# Experiment Design for Simulation Decisions

> Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty  
> Topic 05 of 10  
> Estimated study time: 4 hours  
> Prerequisites: probability, sampling, and discrete-event simulation  
> Version 2026.5

## 1. Why this topic exists

A distribution-centre director must choose whether to add an evening picking shift.
A simulation can show the effect of the extra labour on queueing and late orders.
One simulated month, however, is not a reliable estimate of the expected effect.
The month may happen to contain light demand, favourable service times, or no equipment failures.
A second month can reverse the apparent ranking without any change in the policy.

The decision is not whether the simulated output looks realistic on one animation.
The decision is whether the evidence is strong enough, and economically relevant enough, to commit resources.
Experiment design connects that business question to repeated, controlled simulation runs.
It specifies what varies, what stays fixed, what gets measured, and how uncertainty enters the recommendation.

The director needs a comparison, not two unrelated point estimates.
If both staffing policies encounter the same external demand and equipment environment, much of the irrelevant variation can cancel.
This technique is called common random numbers (CRN).
It can improve precision when the induced correlation is positive, but can make it worse when the correlation is negative.
A shared seed alone does not guarantee that corresponding events use the same random inputs.

A simulation analyst also needs to decide when measurement begins.
An empty queue at time zero may be appropriate for the opening of a new facility.
The same empty queue can distort a model intended to represent a mature operation in steady operation.
Deleting an initial warmup period can reduce that initialization bias, but it also discards information and does not fix a wrong model.

A confidence interval communicates sampling uncertainty from repeated simulation runs under the assumed model.
It is not a guarantee that actual results will lie inside the interval.
It does not automatically include input-data uncertainty, structural errors, parameter drift, or implementation risk.
A narrow interval around a misspecified model can support a very precise bad decision.

The practical deliverable is a decision memo with an auditable experiment ledger.
The ledger records the alternative policies, shared inputs, output measures, random-number design, replication count, uncertainty interval, and decision rule.
A reviewer should be able to reproduce the analysis and understand what the evidence does not establish.

## 2. Learning outcomes

After this topic you can:

1. Define a policy comparison, estimand, time horizon, and unit of replication for an operational decision.
   Exercise 1 asks you to write the experiment contract.
2. Compute a paired difference, its standard error, and a confidence interval from replication outputs.
   Exercise 2 checks the arithmetic.
3. Design common random inputs that preserve the intended marginal distributions of both policies.
   Exercise 3 asks for a matching scheme.
4. Diagnose initial-transient bias and choose a warmup strategy appropriate to the estimand.
   Exercise 4 contrasts terminating and steady-state studies.
5. Select a replication plan using precision, practical significance, and computational cost.
   Exercise 5 requires an explicit plan.
6. Translate simulated service and cost outcomes into an economic decision with uncertainty and constraints.
   Exercise 6 uses the worked-case numbers.
7. Critique model, input, selection, and multiple-comparison risks in a decision brief.
   Exercises 7 and 8 test these judgment calls.

## 3. The core mental model

```text
Business decision and feasible alternatives
                |
                v
Define estimand, horizon, constraints, and materiality
                |
                v
Validate base model and describe uncertain inputs
                |
                v
Design independent replication pairs
  pair 1: environment 1 -> policy A and policy B
  pair 2: environment 2 -> policy A and policy B
  ...
                |
                v
Calculate one outcome difference per pair
                |
                v
Mean difference + uncertainty + practical threshold
                |
                v
Decision, sensitivity, caveats, and monitoring plan
```

The business alternative is an intervention: for example, authorizing one additional shift.
The environment contains the demand, service requirements, disruptions, and other uncertainty that neither policy controls.
The model maps each alternative and environment to outcomes such as late-order cost, throughput, and labour expense.
The comparison should isolate the policy effect as far as the model and experiment design allow.

A replication is a complete simulated realization of the defined experiment.
For a paired design, a replication pair contains one run under each policy in a matched environment.
Pairs must be independent of one another for the ordinary paired t interval used here.
Within a pair, positive dependence is deliberate: common demand or common disruptions expose both policies to similar conditions.

A replication pair is not an individual order.
Orders within a simulation run may share queues, demand conditions, failures, and capacity constraints.
Treating every order as an independent replicate can create a misleadingly narrow interval.
Aggregate to the appropriate replication-level outcome before calculating the ordinary interval.

The central quantity is the difference in outcomes for each matched pair.
If the outcome is annual net saving of B relative to A, positive values favour B.
If the outcome is cost of B minus cost of A, negative values favour B.
Choose a sign convention once and keep it visible in every table and chart.

An interval expresses Monte Carlo uncertainty conditional on the model and its input assumptions.
It is separate from operational variability: one future year can differ substantially from the expected year.
It is also separate from epistemic uncertainty about the demand process, costs, and model structure.
These distinctions matter when management asks both whether the average return is positive and how bad a bad year could be.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Alternative | A specified feasible operating policy | A vague aspiration such as improve service |
| Baseline A | Current or reference policy | Assuming the present operation is costless |
| Candidate B | Intervention compared with A | Changing several unrecorded rules at once |
| Estimand | Population quantity the study aims to estimate, such as expected annual saving in USD/year | Treating any available output as the decision quantity |
| Terminating simulation | Run with a meaningful start and end, such as one shift | Automatically discarding the initial operating period |
| Steady state | Long-run distribution under stable, time-homogeneous assumptions | Assuming every seasonal supply chain has one |
| Initialization bias | Effect of an artificial starting state on the output estimate | Calling all random error bias |
| Warmup | Initial simulated period excluded from measurement, in hours or days | Removing data without defining the target |
| Replication | Complete independent realization of the study horizon | Counting correlated orders as independent runs |
| Replication pair | Two policy runs linked by an intended common environment | Assuming the two runs are independent within a pair |
| Random-number stream | Reproducible sequence assigned to a stochastic source | One global sequence used for all event types |
| Seed | Initialization of a pseudorandom generator | Evidence by itself that inputs remain matched |
| Substream | Disjoint or otherwise managed part of a random stream | An undocumented seed picked by hand |
| Common random numbers | Coupling policy runs to share corresponding exogenous random inputs | Sharing policy-dependent outcomes that should differ |
| Coupling | Joint construction of two outputs with correct policy-specific marginals | Changing a policy's marginal input distribution to force similarity |
| Paired difference | Outcome B minus A, or a declared equivalent, for one pair | Difference between unrelated observation indices |
| Covariance | Joint variation of paired outcomes in squared outcome units | A guarantee of positive correlation |
| Standard deviation | Dispersion of replication outcomes, in outcome units | Precision of the mean |
| Standard error | Estimated sampling dispersion of the mean, in outcome units | Dispersion of individual future outcomes |
| Confidence interval | Procedure-based interval for an unknown model estimand | Probability statement about one fixed parameter after observing the data |
| Confidence level | Long-run coverage of a prespecified interval procedure under its assumptions | Probability that the next observation is in the interval |
| Half-width | Distance from the interval centre to either endpoint, in outcome units | Total interval width |
| Degrees of freedom | Parameter for the t critical value, normally pairs minus one | Number of orders processed |
| Practical threshold | Smallest difference worth acting on, in decision units | A statistical significance threshold |
| Service constraint | Required operating performance criterion with a specified statistic and horizon | A vaguely expressed service target |
| Monte Carlo error | Error from using a finite number of simulation replications | All uncertainty in the decision |
| Input uncertainty | Uncertainty about the real-world input model and its parameters | Run-to-run randomness with inputs fixed |
| Structural uncertainty | Uncertainty about model logic and omitted mechanisms | Sampling noise removable by more replications |
| Batch means | Summary statistics from contiguous parts of a long run | Automatically independent observations |
| Multiple comparisons | Inference involving several alternatives or metrics | Treating each selected winner's nominal interval as selection-proof |
| Holdout | Data or random scenarios not used in fitting or selecting the policy | A second look at the same scenarios |
| Value of information | Expected improvement in decision value from information, net of its cost | The cost saving of the chosen policy alone |
| Forecast | Distribution or estimate of future conditions | A binding customer promise |
| Plan | Proposed action under assumptions | A guaranteed operational result |
| Target | Desired performance level | A prediction or commitment |
| Commitment | An authorized promise or obligation | An unapproved simulation result |

## 5. The mechanics, step by step

### Step 1: Contract for the decision

**Input:** a business owner, an action, available data, and the date by which a choice is needed.
**Output:** a written experiment contract with alternatives and a decision rule.

Name the decision owner and the decision they can actually make.
Specify whether the choices are mutually exclusive, reversible, or staged.
Record the implementation date and the period over which costs and benefits will be counted.
For instance, compare retaining current staffing with adding an evening team for the next 12 months.
Do not describe an operational simulation as an approved staffing commitment.

Define the baseline precisely enough for another analyst to implement it.
Document opening hours, shift capacity, overtime rules, queue priority, and escalation triggers.
Specify the candidate with the same precision.
If the candidate changes both the shift and the dispatch rule, say so; the resulting effect is the combined intervention.

Name the decision quantity before looking at the output.
Expected annual net saving might be the expected baseline cost less the expected candidate cost, including recurring labour.
A service constraint might require the mean late-order rate to remain below a specified percentage.
Alternatively, the business may require a high probability that each individual week meets a service target.
These are distinct estimands; averaging the weekly percentages does not establish the probability of meeting the weekly target.

Separate forecast, plan, target, and commitment.
The demand distribution is a forecast assumption.
The proposed extra shift is a plan.
A 95% on-time rate is a target unless the agreement makes it binding.
A signed service-level agreement or approved hiring decision is a commitment.
The experiment informs these distinctions; it does not erase them.

Choose a materiality threshold in the same currency and time bucket as the estimand.
A USD 10,000/year expected improvement may not justify a disruptive change even if a confidence interval excludes zero.
For irreversible capital expenditure, include the initial cost, useful life, discount rate, and implementation schedule in the economic model.
A recurring annual saving cannot be compared directly with a one-time purchase price without putting them on a common discounted basis.

### Step 2: Choose the experiment horizon

**Input:** the operational process and the decision's time scope.
**Output:** a terminating or steady-state design with a measurement window.

A terminating system has a meaningful end: one promotion, one shift, or one emergency response.
If the real process starts empty at store opening and ends after the backlog is processed, that state is part of the question.
Keep its initial transient in the measured result.
If the business cares about the next 90 calendar days from today's observed backlog, initialize at that backlog and simulate those 90 days.
Discarding the first 30 days would change the question.

A steady-state study asks about long-run behaviour of a stable system under stationary inputs and rules.
Starting an otherwise busy warehouse empty may initially understate queueing and missed dispatches.
Allow a warmup period before collecting measured outcomes if the initial state is artificial.
The appropriate warmup depends on the slowest relevant state, not just the first observed output.
A replenishment backlog may take longer to equilibrate than a pick queue.

Demand seasonality, planned closures, learning, or policy changes can invalidate a simple stationary steady-state interpretation.
A repeating seasonal system may need a cycle-aware design and a full measured number of seasonal cycles.
When the desired answer is calendar-specific, model the calendar and preserve the relevant initial condition instead of assuming a stationary average.
A system whose mean arrival workload exceeds sustainable capacity may not have a stable steady state.
In that case, longer warmup will not repair the problem.

### Step 3: Define the unit and outputs

**Input:** the estimand, horizon, and operational data definitions.
**Output:** a replication-level output schema.

For a 12-month terminating study, one replication is one complete simulated year.
A paired replication contains one baseline year and one candidate year in a matched environment.
Use annual net saving, annual late orders, and annual order count as separate outputs.
Retain numerator and denominator for rate metrics.

The mean of replication-level late-order rates and the pooled late-order rate are generally different.
The former weights each simulated year equally.
The latter weights years by their order counts.
Neither is universally correct; specify which matches the decision before running the experiment.
When assessing a weekly service requirement, store weekly results as well as annual totals.

For a continuing process, independent replications after warmup are often easy to explain.
Another possibility is one long run divided into batches after warmup.
Batch means can remain correlated when batches are too short, and their reliability requires a separate analysis.
Do not apply the independent-replication formula to every individual order or every correlated batch by default.

Specify a rule for unfinished work at the horizon.
Count late orders that remain open, estimate the cost of the remaining backlog, or continue a defined drain-down period.
The choice changes the estimand.
Counting only completed orders can make a policy with a large unfinished backlog appear artificially effective.

### Step 4: Partition sources of randomness

**Input:** the simulation logic and list of exogenous uncertain inputs.
**Output:** a stream and coupling map.

List demand arrivals, order attributes, service requirements, supplier delays, machine failures, and repairs separately.
Some inputs are naturally indexed by order ID, machine ID, or calendar time.
Generate them independently between replication pairs while preserving their intended within-pair relationships.
Record the random generator, seed derivation, stream allocation, and software version.

For arrivals, generate a common external arrival calendar for the two policies in a pair.
Give each common order a stable ID and a common workload requirement.
Allow each policy to schedule and process that order differently.
The service requirement is shared; the realized completion time is a policy outcome and need not be shared.
If a policy prevents an order from arriving or changes customer demand, explicitly model that causal effect rather than forcing identical arrival counts.

For time-based equipment failures, one defensible construction is a common potential failure calendar when the machine is exposed to the same calendar-time hazard.
For usage-based wear, the two policies may accumulate operating hours at different rates.
Share the underlying wear thresholds, not necessarily the calendar time of failure.
The model must retain each policy's correct marginal failure process.
This difference is important: enforcing the same failure time under different utilization can introduce bias.

Use separate streams or keyed random values for distinct exogenous sources.
If both policies consume a single generator sequentially, an additional event in one policy can shift every later draw.
Using the same initial seed then gives different random numbers to corresponding orders.
Stable keyed variates can be associated with pair ID, entity ID, and input type; a stream hierarchy can achieve the same end.
Whichever scheme is chosen, verify with an input trace that corresponding exogenous events match.

Avoid tying a random value to an event that only one policy can create unless its distribution is defined for both alternatives.
If the candidate introduces an additional machine, define its own failure stream without accidentally changing the baseline machine's stream.
If an order is split only under B, define how the parent order's workload is allocated among child tasks.
The coupling should reflect a defensible counterfactual, not merely maximise correlation by changing the marginal model.

### Step 5: Pilot and diagnose coupling

**Input:** a small prespecified pilot with independent replication pairs.
**Output:** estimates of variability, correlation, runtime, and warmup sensitivity.

Run enough pilot pairs to observe rough behaviour, recognizing that a small pilot gives noisy variance estimates.
Compare the paired-difference variance with the variance expected under independent runs.
For outputs A and B,

```text
Var(B - A) = Var(B) + Var(A) - 2 Cov(A, B).
```

Positive covariance reduces the variance of the difference.
Zero covariance gives no gain relative to independent policy outputs with the same marginal variances.
Negative covariance increases it.
The direction and size of covariance are empirical properties of the coupling and model.

For example, suppose both policies' annual cost standard deviations are USD 100,000/year.
With independent runs, the difference variance is 20,000 million squared USD/year.
With correlation 0.8, the paired difference variance is 4,000 million squared USD/year.
The corresponding difference standard deviations are about USD 141,421/year and USD 63,246/year.
This is a variance reduction by a factor of five, not a fivefold reduction in standard deviation.
If the correlation is -0.5, the paired variance becomes 30,000 million squared USD/year and CRN is counterproductive.

Inspect matching traces in addition to aggregate correlation.
A high output correlation does not prove the marginals or correspondence are correct.
A low correlation can reflect genuine policy interactions, faulty synchronization, or both.
Document the diagnosis before switching methods.

### Step 6: Assess initial transient and warmup

**Input:** the target state, initial state, pilot trajectories, and relevant slow state variables.
**Output:** a justified deletion period and measured run length, or an explicit no-warmup decision.

For a terminating study representing an actual opening, use zero warmup unless the estimand explicitly excludes an initial period.
For a steady-state study started from an artificial empty system, compare trajectories from several independent starts.
Plot queue length, work in process, utilization, and backlog over simulated time.
A common diagnostic, Welch's graphical method, averages trajectories across replications and smooths the resulting curve.
A visually stable region can guide a candidate warmup period; it is not a proof of stationarity.

Inspect relevant state variables rather than only the headline average.
An average order delay can appear stable while an inventory pipeline or repair backlog is still changing.
Try plausible longer warmups and check whether the reported result changes materially.
Record the selection rule and sensitivity; choosing a cutoff solely because it produces a preferred result is not defensible.

For matched policies, use the same calendar-time deletion cutoff where the comparison requires the same measured window.
The policies may converge at different rates, so the cutoff must be adequate for both.
Alternatively, initialize each policy from a credible stationary distribution if one is available and justified.
Initializing both with the same arbitrary queue is not equivalent to sampling each policy's own stationary state.

Warmup reduces usable simulation time.
More replications of a short, biased run do not eliminate the initialization bias.
Longer runs after a justified cutoff reduce the influence of the initial state on a long-run average, subject to stable dynamics.
If no defensible stable regime exists, change the estimand to a finite-horizon question rather than claiming a steady-state answer.

### Step 7: Choose the replication count

**Input:** pilot paired differences, a desired half-width, and the computational budget.
**Output:** a documented replication plan and stopping rule.

Let the estimated standard deviation of paired differences be s and the desired half-width be h.
A rough normal approximation for the required number of independent pairs is

```text
n ≈ (z × s / h)^2,
```

where z is the relevant standard-normal critical value.
For a nominal two-sided 95% interval, z is approximately 1.96.
Round up and allow for a t critical value, which is larger at small sample sizes.
For example, s = USD 40,000/year and h = USD 10,000/year give approximately 62 pairs.
This is planning guidance, not a promise of the realized half-width.

The decision may depend on whether an interval lies above a materiality threshold, not on achieving a generic percentage precision.
If the estimated effect is near the threshold, substantially more simulation may be needed.
If the model is clearly inadequate, extra replications are not the first investment.
Compare the cost of further computation and analysis with the value of reducing decision uncertainty.

A fixed number of pairs chosen in advance gives the ordinary fixed-sample interval its usual interpretation under its assumptions.
Repeatedly checking an ordinary interval and stopping at the first favourable crossing can distort coverage.
If adaptive stopping is important, use a sequential procedure with appropriate coverage guarantees or a prespecified and justified two-stage design.
Record every look and the stopping rule.
Do not discard unfavourable runs.

Pilot results can inform a new fixed design, but careless reuse of the same data for data-dependent sample sizing can alter exact coverage.
A simple operational approach is to use the pilot only for planning and then run a fresh, independent confirmatory sample of fixed size.
If pilot data are reused, state the procedure and justify its inference rather than claiming automatic nominal coverage.

### Step 8: Compute a paired interval

**Input:** n independent paired differences d₁, ..., dₙ in a common unit.
**Output:** a mean difference and two-sided confidence interval.

For pair i, define dᵢ as the net economic outcome of B relative to A.
Calculate the sample mean d̄ and sample standard deviation s_d using denominator n - 1.
The estimated standard error is s_d / √n.
For a two-sided 95% interval, use the t critical value with n - 1 degrees of freedom:

```text
mean difference ± t(0.975, n - 1) × s_d / √n.
```

The conventional t interval is exact for independent normally distributed differences.
For sufficiently well-behaved differences and a large sample it can also be approximately useful.
Heavy tails, rare disruptions, unstable variance, or small samples can make the approximation poor.
Inspect the distribution and consult an appropriate method when those conditions matter.
A bootstrap over independent pairs can be informative, but ordinary bootstrap methods are not automatically reliable for rare events or very small samples.

The interval concerns the expected paired difference under the assumed input model.
A 95% procedure would cover the fixed model mean in 95% of hypothetical repetitions of the complete fixed-sample procedure, under its conditions.
After observing one interval, its endpoints are fixed; the frequentist interpretation is not a 95% probability that the fixed mean is inside it.
An interval for a future year's saving requires a prediction or distributional analysis, not merely the confidence interval for the mean.

### Step 9: Include economics and constraints

**Input:** replication-level operational outcomes, cost schedule, and decision thresholds.
**Output:** comparable net benefits and separate constraint evidence.

Convert outcomes to a common currency and horizon.
For example, multiply avoided late orders by an incremental cost per late order, then subtract the incremental shift cost.
Use incremental economic cost, not revenue that would have been earned anyway or accounting allocations unaffected by the decision.
Check whether the late-order cost already includes refunds, lost margin, expediting, and customer service before adding separate line items.
Double counting makes the candidate appear more valuable than it is.

Variable penalties can be simulated at the order level when severity differs.
A single average penalty is appropriate only for the specified approximation and population.
If the penalty is uncertain, vary it in a sensitivity analysis and show the break-even value.
For an investment with an upfront payment I and annual cash benefits B_t, compare net present value:

```text
NPV = -I + Σ B_t / (1 + r)^t.
```

Use consistent nominal cash flows and nominal discount rates, or consistent real cash flows and real rates.
Avoid treating uncertain benefits as guaranteed cash flows.
Compute scenario-level NPV when the annual benefits and their dependence matter.

A service requirement needs its own statistic and uncertainty assessment.
An expected annual cost saving cannot prove that a 95% on-time target is met.
For example, a management rule might require the lower confidence bound for expected service to exceed a threshold.
A contractual rule may instead apply to each month, each customer, or specific exclusions.
Translate the actual wording into a measured statistic before the experiment.

### Step 10: Protect the decision against selection

**Input:** candidate set, planned comparisons, and decision policy.
**Output:** a selection and confirmation protocol.

When testing many staffing levels, schedules, and priority rules, the apparent best result is partly selected for favourable simulation noise.
A nominal interval calculated after choosing the winner on the same runs need not have its advertised coverage for the selected winner.
Reserve independent confirmation scenarios, or use a justified multiple-comparison or ranking-and-selection method.
State whether intervals are per comparison or simultaneous across the family of decisions.

Do not use a holdout repeatedly to tune the policy and still call it untouched confirmation.
Document any exploration and the final locked candidates.
If the confirmation changes the selected policy, obtain new confirmation or acknowledge the selection.
Model selection and statistical comparison do not replace operational feasibility review.

### Step 11: Report, decide, and monitor

**Input:** confirmed outputs, economic thresholds, constraints, and assumptions.
**Output:** a recommendation and a plan to compare reality with the model.

Report the sign convention, point estimate, interval, number of independent pairs, and horizon together.
Show uncertainty in the operational metric as well as net economics.
Distinguish the Monte Carlo interval from sensitivity to input distributions and from structural uncertainty.
State which assumptions would reverse the recommendation.

Give the decision owner an actionable rule.
One possible rule is to adopt B only if the lower confidence bound on expected annual saving exceeds USD 20,000/year and the service constraint is supported.
This is a chosen business rule, not a universal statistical requirement.
A decision-maker may rationally accept more risk, run a pilot, or defer if the evidence is inconclusive.

After approval, monitor actual labour cost, workload, service, disruptions, and implementation changes.
Compare realized outcomes with the forecast distribution, not only with the mean.
Investigate material deviations and update the model with independent operational evidence.
An apparently accurate confidence interval for the model mean does not demonstrate that the deployed system is valid.

## 6. Worked example

### Decision and scope

A regional fulfilment centre experiences late orders after a carrier pickup cutoff.
The director is considering a second evening picking team for a 12-month planning period.
Policy A keeps the current schedule.
Policy B adds the team and otherwise retains the same dispatch rule.
The hypothetical figures below are illustrative, not a company case.

The demand forecast is a distribution of daily orders and order sizes.
The staffing plan is to add the team if the decision rule is satisfied.
The service target is at least 95% of orders dispatched on time, measured over the planning period.
No hiring commitment exists until the director approves it.
The study uses a terminating 12-month horizon initialized from the observed starting backlog.
Because the starting backlog is part of the decision, there is no deleted warmup in this example.

The economic estimand is expected annual net saving of B relative to A, in USD/year.
The team costs an additional USD 120,000/year, including payroll and related expenses.
Each avoided late order is valued at an illustrative incremental USD 50/order.
This unit cost combines expected incremental expediting and customer remedy costs.
It excludes lost revenue and any benefits already included elsewhere.
The decision threshold is USD 20,000/year of expected net saving.
A separate service assessment is required; the small worked sample below does not establish service compliance.

### Design

The analyst constructs five independent scenario pairs for transparent hand calculation.
Each pair has a common arrival calendar, order workload by stable order ID, and common potential equipment disruption inputs.
The two policies can produce different queue states and completion times.
The annual avoided late-order counts are the difference between late orders under A and B in each pair.
The five pairs are independent; the two policies within each pair are intentionally coupled.
The count of five is for instruction, not a recommended production sample size.

| Pair | A late orders/year | B late orders/year | Avoided late orders/year | Net saving, USD/year |
|---:|---:|---:|---:|---:|
| 1 | 7,000 | 3,800 | 3,200 | 40,000 |
| 2 | 7,200 | 3,600 | 3,600 | 60,000 |
| 3 | 6,800 | 4,000 | 2,800 | 20,000 |
| 4 | 7,500 | 3,700 | 3,800 | 70,000 |
| 5 | 7,000 | 3,400 | 3,600 | 60,000 |

For pair 1, avoided late-order benefit is 3,200 × USD 50 = USD 160,000/year.
Subtract the USD 120,000/year labour increment to obtain USD 40,000/year net saving.
The same formula gives the other four values.
The sample mean saving is (40,000 + 60,000 + 20,000 + 70,000 + 60,000) / 5 = USD 50,000/year.
The sample mean avoided count is 3,400 orders/year.
Multiplying 3,400 by USD 50/order and subtracting USD 120,000/year gives the same USD 50,000/year.

### Interval arithmetic

Measure differences in thousands of USD/year for the calculation.
The five differences are 40, 60, 20, 70, and 60.
Their deviations from the mean of 50 are -10, 10, -30, 20, and 10.
The squared deviations sum to 1,600 in squared thousands of USD/year.
The sample variance is 1,600 / 4 = 400 in those squared units.
The sample standard deviation is 20 thousand USD/year.
The standard error is 20 / √5 = 8.944 thousand USD/year.

With four degrees of freedom, the two-sided 95% t critical value is approximately 2.776.
The half-width is 2.776 × 8.944 = 24.83 thousand USD/year.
The interval is approximately USD 25,170 to USD 74,830 per year.
The entire interval is above the USD 20,000/year materiality threshold.
Under the stipulated rule, the economic evidence passes this one criterion.
It does not satisfy the separate service criterion, which has not been calculated.
A five-pair interval also depends strongly on the distributional assumption and is too thin a basis for a consequential hiring decision without further confirmation.

### Break-even and sensitivity

At the sample mean, 3,400 avoided late orders/year require an incremental late-order cost of

```text
(USD 120,000/year + USD 20,000/year) / 3,400 orders/year
= USD 41.18 per avoided late order.
```

That is the point-estimate break-even penalty for meeting the USD 20,000/year materiality threshold.
It is not the threshold for the lower confidence bound, because sampling variation remains.
At USD 40/order, the point estimate becomes USD 16,000/year, below materiality.
At USD 60/order, it becomes USD 84,000/year, before considering uncertainty in the penalty itself.
Thus the estimated incremental economic cost of lateness is decision-critical.
If that cost is mostly an accounting allocation that would not change, the model overstates savings.

The calculation also excludes an implementation cost.
If the team requires a one-time USD 60,000 setup expense and generates a constant USD 50,000/year for three years at 10% per year, the illustrative point-estimate NPV is

```text
-60,000 + 50,000/1.10 + 50,000/1.10² + 50,000/1.10³
≈ USD 64,343.
```

This is a separate three-year illustration, not an additional term in the 12-month confidence interval.
It assumes constant benefits, no residual value, no taxes, and the stated discount rate.
A proper stochastic NPV analysis would propagate annual dependence and uncertainty.

### Decision record

The preliminary economic estimate favours B at the assumed late-order cost.
The director should not treat that as approval: service performance, implementation feasibility, and confirmation remain unresolved.
A production analysis would use a prespecified larger independent set of pairs, a service statistic aligned with the agreement, and sensitivity to the late-order cost.
It would also check whether the forecast demand, backlog initialization, and failure logic are credible for the actual 12-month period.

## 7. Code, query, or calculation

The following self-contained Python 3 example reproduces the arithmetic above.
It uses only the standard library and does not require a simulation package.
The five paired outcomes are synthetic teaching data, not the output of a validated warehouse model.
The t critical value is supplied for exactly four degrees of freedom and a two-sided 95% interval.
Do not reuse that constant for another sample size or confidence level.
The example demonstrates the analysis of a replication table; it does not simulate arrivals or establish that a CRN implementation is correct.

```python
from math import sqrt
from statistics import mean, stdev

# One row per independent replication pair.
# Annual late-order counts share a scenario within each row.
rows = [
    (7000, 3800),
    (7200, 3600),
    (6800, 4000),
    (7500, 3700),
    (7000, 3400),
]
late_cost = 50.0       # incremental USD per avoided late order
shift_cost = 120000.0  # incremental USD per simulated year
threshold = 20000.0    # USD per year
critical_t = 2.776445105  # 97.5th percentile, df = 4

assert len(rows) == 5
avoided = [a - b for a, b in rows]
savings = [count * late_cost - shift_cost for count in avoided]
n = len(savings)
estimate = mean(savings)
sample_sd = stdev(savings)  # sample denominator n - 1
standard_error = sample_sd / sqrt(n)
half_width = critical_t * standard_error
lower = estimate - half_width
upper = estimate + half_width

print(f"pairs: {n}")
print(f"mean avoided late orders/year: {mean(avoided):,.0f}")
print(f"mean saving USD/year: {estimate:,.2f}")
print(f"paired SD USD/year: {sample_sd:,.2f}")
print(f"standard error USD/year: {standard_error:,.2f}")
print(f"95% interval USD/year: [{lower:,.2f}, {upper:,.2f}]")
print(f"economic threshold passed: {lower > threshold}")
print("service constraint: not assessed")
```

Expected output, rounded to cents:

```text
pairs: 5
mean avoided late orders/year: 3,400
mean saving USD/year: 50,000.00
paired SD USD/year: 20,000.00
standard error USD/year: 8,944.27
95% interval USD/year: [25,166.45, 74,833.55]
economic threshold passed: True
service constraint: not assessed
```

The code keeps the pairing because it subtracts A and B within each row before taking a standard deviation.
Sorting the A and B columns separately would destroy the pairing.
If the shift cost varied by scenario, include the pair-specific incremental cost before computing each saving.
If the late-order cost varied by customer, calculate the avoided cost from the appropriate order-level outcomes within each replication.

For a production experiment, store at least the following fields per run:

| Field | Example | Purpose |
|---|---|---|
| Experiment ID | `evening-shift-2026-01` | Identifies the decision and model version |
| Pair ID | `17` | Matches the two policy runs |
| Policy | `A` or `B` | Identifies the intervention |
| Input scenario ID | `scenario-17` | Audits matching |
| Stream configuration | Generator and seed mapping | Supports reproducibility |
| Start and end | Simulated timestamps | Establishes the horizon |
| Warmup cutoff | Timestamp or zero | Defines measured observations |
| Initial state version | Backlog snapshot identifier | Makes initialization auditable |
| Late orders | Integer orders/year | Supports service and cost calculation |
| Total orders | Integer orders/year | Supplies the service-rate denominator |
| Incremental cost | USD/year | Supports net benefit calculation |
| Model revision | Commit or release identifier | Connects outputs to executable logic |
| Runtime status | Complete or failed with reason | Prevents silent omission of failed runs |

For each pair, check that both policies completed and that the matching scenario identity is correct.
A failed run is not automatically a zero or a missing-at-random observation.
Investigate whether failures occur preferentially under severe workload or a particular policy.
If the experiment is rerun after a defect is corrected, preserve the old record and identify the new model version.

## 8. Data, standards, and regulatory context

The experimental design is only as meaningful as the operational inputs and definitions.
Demand arrival timestamps need a defined time zone and treatment of daylight-saving changes.
Order creation time, warehouse release time, and customer promise time are different events.
Choosing one in place of another changes both delay and the service metric.

Service-time records may be censored by shift end, equipment outage, or incomplete transactions.
Completed orders alone can omit the slowest unfinished work.
Backlog snapshots should state whether they contain cancelled orders, holds, and work already picked but not dispatched.
Use an extraction timestamp and a reproducible inclusion rule.

For the CRN map, order identifiers must be stable within a simulated scenario.
A physical enterprise identifier may change when an order is split or merged.
Retain parent-child relationships rather than relying on a generated row order.
Random-number assignment should use the model's stable identity, not the incidental processing sequence.

The GS1 Global Trade Item Number identifies trade items, not independent simulation runs or experiment scenarios.
GS1 standards can support consistent product identification across trading partners, but they do not prescribe a warmup period or a confidence-interval method.
Similarly, a business event standard can improve the meaning of timestamps without making those events statistically independent.
Map the actual fields used by the model and document gaps.

ISO 9001 describes quality-management-system requirements at an organizational level.
It is not a universal legal requirement to run a specific number of simulation replications.
Where an organization is certified, its documented processes and applicable contractual obligations may govern approvals and records.
The applicable edition, certification scope, contract, and jurisdiction require local review.
This lesson does not provide legal advice.

If order histories contain personal data, privacy obligations may apply to collection, retention, sharing, and use.
For example, the EU General Data Protection Regulation applies under its stated territorial and material scope; it is not a blanket rule for every simulated order everywhere.
Use data minimization and appropriate access controls according to the organization's legal obligations.
A random seed or a scenario ID is not, by itself, proof that the underlying data have been anonymized.

Model validation and experiment design are distinct controls.
Validation assesses whether the model is sufficiently credible for its intended use.
Replication estimates conditional sampling uncertainty after the modelling assumptions are set.
A narrow interval does not certify the fidelity of the simulated arrival or service process.
Maintain provenance for source data, transformations, model revision, and experiment output together.

Input-data uncertainty deserves separate treatment when the model's input distributions are estimated from limited history.
For example, use alternative defensible parameter sets or a nested resampling design when appropriate.
The inner simulation's confidence interval conditional on one fitted distribution is not an interval that automatically includes uncertainty in fitting that distribution.
If seasonality or rare disruptions are poorly observed, disclose that limitation rather than inventing a precise probability.

## 9. Industry practice and case evidence

Simulation experiments are used to compare operational alternatives before implementing them in a live system.
A published review by Terzi and Cavalieri surveys simulation in the supply-chain context and discusses the variety of applications and modelling approaches.
It is evidence of use and methodological scope, not evidence that any particular design in this lesson produced savings for a named company.

A frequently cited published manufacturing example is the semiconductor fabrication facility studied by Fowler, Hogg, and Phillips in their work on control and simulation of wafer fabrication.
Such systems motivate careful treatment of queues, utilization, and operational performance.
The existence of a published application does not establish that its numerical results transfer to a distribution centre.
Model structure, input distributions, and decision economics must be re-established for the current setting.

NIST documents simulation as one component of manufacturing systems analysis and digital-twin work.
That is a public institutional source for the importance of explicit representations and model integration.
It should not be read as a certification of the teaching example or its proposed staffing decision.

A typical business workflow can be described without asserting a specific firm's confidential practice:

1. Operations describes the feasible interventions and decision deadline.
2. Finance agrees on incremental cost definitions and materiality.
3. The analyst validates historical inputs and the model's decision-relevant outputs.
4. The experiment owner locks the policy definitions and random-number plan.
5. Independent replication pairs are run and logged.
6. The analyst reports operational performance, economics, uncertainty, and sensitivity separately.
7. The owner chooses implementation, further evidence, or no change.
8. The implementation team monitors realized outcomes against the forecast.

This workflow is a recommended synthesis, not a claim that every company uses these exact steps.
In a high-stakes application, independent review should focus on the estimand, coupling, terminal backlog, input evidence, and actual decision rule.
A visually convincing animation is useful for communicating mechanisms but cannot substitute for those checks.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| One run per policy | Recommendation changes with seed | Natural run variability mistaken for policy effect | Unstable decision | Use independent replication pairs |
| Same seed, misaligned draws | Matched order IDs receive different workloads | Event sequences consume draws in different order | Little or misleading variance reduction | Separate streams or key by entity and source |
| Wrong marginal coupling | Candidate failure rate changes when CRN is enabled | Coupling changes policy-specific input law | Biased comparison | Validate marginal inputs for each policy |
| Negative paired covariance | Difference variance exceeds independent benchmark | Policies react oppositely to shared inputs | Wider interval | Diagnose pilot covariance and consider independent streams |
| Pseudoreplication | Millions of orders claimed as millions of independent samples | Shared queues and shocks correlate outcomes | Understated uncertainty | Analyse independent run-level differences |
| Empty-start bias | Early queues are much shorter than later queues | Artificial initialization favours early output | Biased long-run estimate | Justify warmup or initialize realistically |
| Unnecessary deletion | Real opening period omitted | Warmup removes part of target horizon | Wrong estimand | Preserve genuine starting transient |
| Unstable system | Backlog grows without bound across long runs | Workload exceeds sustainable capacity | No meaningful steady-state estimate | Study finite horizon or change capacity assumptions |
| Incomplete horizon | Large open backlog but only completed orders measured | Slow work is censored | Candidate seems better than it is | Count open work or define drain-down |
| Premature stopping | Runs stop at first favourable interval | Repeated looks select random crossings | Nominal coverage can fail | Fixed sample or valid sequential method |
| Winner selection | Best of many policies reported with ordinary interval | Selection favours favourable noise | Overoptimistic benefit | Independent confirmation or simultaneous method |
| Cost double count | Penalty and lost margin include the same remedy | Same economic effect added twice | Inflated savings | Map each cost to one incremental cash-flow item |
| Service metric mismatch | Annual average passes but contractual months fail | Aggregation hides low-service periods | Operational or contractual exposure | Match statistic and time bucket to obligation |
| Narrow conditional interval | Small Monte Carlo error despite weak inputs | More runs reduce only simulation noise | False confidence | Separate input and structural sensitivity |
| Rare-event blindness | No severe outage appears in a small sample | Low-frequency tail poorly sampled | Underestimated downside | Model plausible scenarios and use suitable rare-event methods |
| Unrecorded changes | Results cannot be reproduced from saved configuration | Model or data changed during study | Audit failure | Version inputs, code, seeds, and decision rule |
| Missing failed runs | Severe scenarios disproportionately absent | Runtime failures are selectively dropped | Biased results | Log and resolve failure mechanism |
| Ignoring implementation | Model assumes staff available from day one | Recruitment and ramp-up omitted | Benefits overestimated | Include realistic implementation timing and costs |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or assess |
|---|---|---|---|
| Independent replication pairs | Standard error formula treats pairs as independent | Shared generator state or reused scenario | Audit seed hierarchy and scenario identities |
| Correct paired marginals | Difference must represent each policy's valid output | Forcing a common failure time despite different wear | Compare generated inputs with each policy's intended law |
| Comparable matched environment | CRN is intended to remove irrelevant environmental noise | Input streams drift when event counts differ | Trace common order IDs and external events |
| Fixed-sample interval | Conventional confidence interpretation uses a prespecified design | Unadjusted repeated stopping | Audit planned and actual number of looks |
| Adequate t approximation | Small-sample interval relies strongly on difference distribution | Extreme skew or rare catastrophic loss | Inspect paired differences and use a justified alternative |
| Valid input distribution | Results are conditional on model inputs | Shift in demand, promotion, or supplier risk | Compare with holdout data and stress plausible scenarios |
| Correct model logic | A simulation reflects real causal mechanisms | Missing capacity, priority, or operational workaround | Face, structural, and output validation |
| Appropriate starting state | Early trajectory should represent the target | Empty start used for a mature backlog | Compare initial state with observed operations; test warmup |
| Stable steady state when claimed | Long-run mean must exist for that target | Demand exceeds effective service capacity | Examine workload, backlog trends, and stationarity assumptions |
| Relevant economic valuation | Avoided delay must translate into actual incremental cost | Accounting allocation or double counted remedy | Reconcile with finance and trace cash flows |
| Correct service definition | Constraint must correspond to the actual obligation | Annual mean substituted for monthly contractual test | Review wording, numerator, denominator, and exclusions |
| Feasible candidate | Simulation policy must be implementable | Staff cannot be hired or labour rule prohibits schedule | Operations and human-resources feasibility review |
| Forecast remains applicable | Decision is forward-looking | Market or operational regime changes | Monitor leading indicators and refresh scenarios |
| No unmodelled selection | Nominal interval assumes the stated comparison | Candidate chosen as best of many on same runs | Preserve selection history and use holdout confirmation |

None of these checks proves that a recommendation will succeed.
They make the decision's dependency on evidence visible.
Simulation replication cannot eliminate uncertainty about whether customers or employees will react differently in the real implementation.

## 12. Exercises

### Exercise 1 — Define the decision

A retailer considers a weekend receiving shift to reduce Monday backlog.
Historical data include shipment arrivals, dock capacity, and labour hours.

**Deliverable:** a one-page experiment contract.
**Acceptance:** name baseline and candidate, one economic estimand with units, one service constraint, the horizon, initial state, and decision owner; distinguish forecast, plan, target, and commitment.
**Time:** 15 minutes.

### Exercise 2 — Calculate a paired interval

Five independent paired savings are USD 40,000, 60,000, 20,000, 70,000, and 60,000 per year.
Use the two-sided 95% t critical value 2.776 with four degrees of freedom.

**Deliverable:** sample mean, sample standard deviation, standard error, and interval.
**Acceptance:** use paired savings as observations, the n - 1 variance denominator, and explain what the interval covers.
**Time:** 15 minutes.

### Exercise 3 — Design common random inputs

Two dispatch policies process the same incoming orders; one policy splits large orders across two machines.
Machine failure depends on accumulated usage rather than elapsed calendar time.

**Deliverable:** a stream or keyed-input map for arrivals, order workloads, splitting, and machine failures.
**Acceptance:** preserve each policy's marginal laws, match common order attributes, explain how usage thresholds translate into different failure times, and specify a trace check.
**Time:** 20 minutes.

### Exercise 4 — Decide whether to warm up

Study A asks about the first 30 days after opening an empty warehouse.
Study B asks about the long-run average delay at a mature, stable warehouse, but the model begins empty.

**Deliverable:** a warmup decision for each study and a diagnostic plan for B.
**Acceptance:** retain the genuine transient in A; for B discuss multiple trajectories, slow state variables, cutoff sensitivity, and whether steady state is plausible.
**Time:** 15 minutes.

### Exercise 5 — Plan precision

A pilot gives a paired-difference standard deviation of USD 40,000/year.
The desired 95% interval half-width is USD 10,000/year.
Use 1.96 for a rough normal calculation.

**Deliverable:** an approximate number of pairs and a confirmatory plan.
**Acceptance:** show the squared formula, round upward, note the t adjustment, and specify how pilot data and stopping are handled.
**Time:** 15 minutes.

### Exercise 6 — Evaluate the economics

Use the worked example's mean of 3,400 avoided late orders/year and USD 120,000/year incremental labour.
The materiality threshold is USD 20,000/year.

**Deliverable:** the break-even cost per avoided late order and an interpretation at USD 40/order.
**Acceptance:** include units, calculate point-estimate net saving, and explain why the calculation does not establish the service constraint or the confidence-bound decision.
**Time:** 15 minutes.

### Exercise 7 — Critique selection and uncertainty

An analyst simulates 50 policy variants, selects the largest observed saving, and reports its ordinary 95% interval from those same runs.
The analyst claims the interval includes every source of uncertainty.

**Deliverable:** a short critique and confirmation protocol.
**Acceptance:** identify selection bias, conditional Monte Carlo uncertainty, a way to protect confirmation, and at least one input or structural sensitivity.
**Time:** 20 minutes.

### Exercise 8 — Build a decision memo

A candidate policy has estimated net saving of USD 55,000/year with a 95% paired interval of USD 8,000 to USD 102,000/year.
The adoption rule requires a lower bound above USD 20,000/year and evidence that monthly service meets the stated requirement.
The model records only an annual average service rate.

**Deliverable:** a decision recommendation and a prioritized evidence plan.
**Acceptance:** do not claim the rule is met; identify the economic interval and service-definition gap; propose a justified next step and document non-Monte-Carlo risks.
**Time:** 20 minutes.

## 13. Self-check answers

### 1. Decision contract rubric

A strong answer names the current receiving schedule as A and defines the proposed weekend shift as B, including its staffing and start date.
An appropriate estimand is expected incremental net cost saving in USD over the next 13 weeks.
A service measure could be the percentage of Mondays with backlog below a specified threshold, with the denominator and cutoff explicitly defined.
The initial backlog should be the actual or forecast state at the start of that period, not automatically an empty system.
The forecast is the arrival distribution; the plan is the proposed shift; the target is the desired backlog performance; the commitment occurs only on authorization or agreement.

### 2. Paired interval

The mean is USD 50,000/year.
The sample standard deviation is USD 20,000/year and the standard error is approximately USD 8,944/year.
The interval is approximately USD 25,170 to USD 74,830/year.
It is a confidence interval for the expected model-based annual saving, not a range guaranteed to contain next year's realized saving.
It assumes independent pairs and, with only five observations, relies heavily on the t-method conditions.

### 3. CRN design rubric

Generate a common external order calendar and stable order IDs for each pair.
Assign each order a common total workload using an order-keyed stream.
Define the candidate's split rule and workload allocation without changing the total workload distribution.
Assign independent wear thresholds by machine and pair; translate them to failure times using each policy's accumulated usage.
Trace the same order's arrival and workload under both policies and compare generated failure-threshold distributions against their intended laws.
Matching realized completion times or forcing identical calendar failures would generally be wrong.

### 4. Warmup

Study A should retain the first 30 days because opening empty is the actual initial condition and the entire period is the target.
For study B, an initial deletion may reduce empty-start bias if a stable long-run regime exists.
Inspect independent trajectories for backlog, queues, utilization, and other slow variables; compare results under longer plausible cutoffs.
Do not claim a steady-state answer if backlog persistently grows or the seasonal process does not support the assumed regime.

### 5. Replication planning

The rough count is (1.96 × 40,000 / 10,000)² = 61.4656, rounded up to 62 pairs.
This normal approximation ignores the larger finite-sample t critical value and the uncertainty in the pilot standard deviation.
Choose a slightly larger justified fixed confirmatory sample or solve the t-based planning equation iteratively.
One clean approach uses the pilot only for planning and runs fresh independent confirmatory pairs with the sample size locked in advance.
Do not repeatedly check an ordinary interval and stop when it first becomes favourable.

### 6. Economics

The point-estimate break-even cost is (120,000 + 20,000) / 3,400 = approximately USD 41.18 per avoided order.
At USD 40/order the point-estimate net saving is 3,400 × 40 - 120,000 = USD 16,000/year.
That is below the USD 20,000/year threshold.
It is not a confidence-bound calculation, and it says nothing by itself about whether the service requirement is met.

### 7. Selection critique rubric

Selecting the largest observed saving among 50 alternatives favours positive noise.
The ordinary interval calculated on the selection runs is not automatically valid for the selected winner.
Lock the candidate and evaluate it on fresh independent scenarios, or use a justified selection-adjusted procedure.
Even a correct Monte Carlo interval is conditional on the assumed model; vary demand, delay costs, and plausible missing mechanisms separately.
Report the family of comparisons and whether any claimed intervals are simultaneous.

### 8. Decision memo rubric

The lower economic bound is USD 8,000/year, below the required USD 20,000/year, so the stated economic rule is not met.
The annual service average does not answer the monthly requirement.
Do not claim the candidate fails economically in truth: the interval is inconclusive relative to the threshold under the chosen rule.
Consider more independent pairs if the expected value of improved precision justifies the study cost, while separately collecting monthly service outputs.
Review input, implementation, and cost uncertainty because more simulation alone will not resolve them.

## 14. Interview questions

1. **What is the replication unit in a paired simulation?**
   A strong answer distinguishes an independent scenario pair from correlated orders within a run.
   A weak answer treats every simulated transaction as independent.
2. **Why can common random numbers improve a comparison?**
   A strong answer uses the covariance term and explains positive correlation of matched outcomes.
   A weak answer promises improvement from using the same seed.
3. **How can same-seed runs lose synchronization?**
   A strong answer describes policy-dependent event counts and separate streams or entity-keyed inputs.
   A weak answer assumes the generator always consumes draws in the same order.
4. **When can CRN make precision worse?**
   A strong answer identifies negative covariance and checks the paired-difference variance.
   A weak answer says the variance must decrease.
5. **When should you discard warmup?**
   A strong answer relates deletion to an artificial start and a credible steady-state estimand.
   A weak answer always deletes the first 10% without examining the question.
6. **What does a 95% confidence interval cover?**
   A strong answer describes repeated-procedure coverage for the model mean under assumptions.
   A weak answer calls it the range of 95% of future years.
7. **Why can more replications fail to make a decision credible?**
   A strong answer separates Monte Carlo error from input and structural bias.
   A weak answer equates precision with validity.
8. **How do you compare a recurring benefit with a capital cost?**
   A strong answer puts incremental cash flows on a common discounted time basis and states assumptions.
   A weak answer subtracts a one-time purchase price from one year's saving without explanation.
9. **What is wrong with picking the best of 50 policies and reporting its ordinary interval?**
   A strong answer recognizes selection and proposes independent confirmation or simultaneous inference.
   A weak answer assumes the nominal confidence level automatically survives selection.
10. **How would you check a service constraint?**
    A strong answer identifies the contractual time bucket, numerator, denominator, exclusions, and appropriate uncertainty.
    A weak answer substitutes annual mean cost for service performance.

## 15. Key terms

- **Alternative:** an implementable policy in the [decision contract](#step-1-contract-for-the-decision).
- **Baseline:** the reference policy against which change is evaluated.
- **Batch means:** summaries from contiguous periods of a long run; see [unit and outputs](#step-3-define-the-unit-and-outputs).
- **Common random numbers:** matched inputs across policies; see [partitioning randomness](#step-4-partition-sources-of-randomness).
- **Confidence interval:** a procedure-based range for a model estimand; see [paired interval](#step-8-compute-a-paired-interval).
- **Confidence level:** long-run coverage of the stated interval procedure.
- **Coupling:** a joint construction preserving each policy's marginal distribution.
- **Covariance:** a measure of joint variation governing the CRN benefit.
- **Decision threshold:** the improvement material enough to justify action.
- **Degrees of freedom:** parameter of the t distribution, here the number of pairs minus one.
- **Estimand:** the precisely defined quantity the study estimates.
- **Forecast:** an estimate or distribution of future conditions, not an authorized action.
- **Holdout:** independent data or scenarios reserved for confirmation.
- **Initialization bias:** distortion caused by an unrepresentative initial state.
- **Input uncertainty:** uncertainty about real-world input models and parameters.
- **Materiality:** the business importance of an effect in decision units.
- **Monte Carlo error:** finite-replication error conditional on the model.
- **Net present value:** discounted incremental benefits less investment; see [economics](#step-9-include-economics-and-constraints).
- **Paired difference:** the difference in policy outcomes within one matched scenario.
- **Plan:** a proposed action that is not necessarily committed.
- **Replication:** a complete realization of the defined simulation experiment.
- **Seed:** generator initialization, not proof of matched inputs.
- **Service constraint:** a specified operating requirement with a defined statistic.
- **Standard error:** estimated sampling variation of the mean.
- **Steady state:** long-run distribution under suitable stable assumptions.
- **Structural uncertainty:** uncertainty about the mechanisms represented in the model.
- **Target:** a desired level of performance, distinct from a forecast.
- **Terminating simulation:** a run with a meaningful start and end.
- **Warmup:** an initial period excluded when appropriate for the estimand.

## 16. Sources

- Law, Averill M. *Simulation Modeling and Analysis*, 5th ed., McGraw-Hill Education, 2015. Standard reference for simulation output analysis, initialization, replications, and variance reduction.
- Welch, Peter D. “The Statistical Analysis of Simulation Results.” In *The Computer Performance Modeling Handbook*, Academic Press, 1983. Describes graphical analysis of initial-transient behaviour.
- L'Ecuyer, Pierre. “Random Numbers for Simulation.” *Communications of the ACM* 33(10), 1990, 85–97. [https://doi.org/10.1145/84537.84555](https://doi.org/10.1145/84537.84555). Background on pseudorandom numbers and simulation.
- Terzi, Sergio, and Sergio Cavalieri. “Simulation in the Supply Chain Context: A Survey.” *Computers in Industry* 53(1), 2004, 3–16. [https://doi.org/10.1016/S0166-3615(03)00104-0](https://doi.org/10.1016/S0166-3615(03)00104-0). Review of supply-chain simulation applications.
- National Institute of Standards and Technology. [Digital Twins](https://www.nist.gov/digital-twins). Public institutional overview; it does not validate this lesson's hypothetical case.
- GS1. [GTIN Management Standard](https://www.gs1.org/1/gtinrules/en/). Identifier guidance, not a simulation-inference standard.
- International Organization for Standardization. [ISO 9001 quality management](https://www.iso.org/standard/62085.html). Organizational quality-management context; applicability depends on certification and contracts.
- European Union. [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Consult the regulation and relevant local advice for actual applicability.
- Python Software Foundation. [`statistics` — Mathematical statistics functions](https://docs.python.org/3/library/statistics.html). Defines the standard-library sample standard deviation used in the code.

## 17. Links

- [Module 20 topic index](_index.md).
- [Twin, shadow, simulation, and model choice](01_twin-shadow-simulation.md).
- [Discrete-event simulation](02_discrete-event-simulation.md).
- [Module 20 survey](../../../course/20_digital_twins_simulation_optimization_uncertainty.md).
- [Simulation and optimization production labs survey](../../../course/30_simulation_optimization_production_labs.md).
