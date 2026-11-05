# Input Modeling for Simulation

> Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty  
> Topic 03 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Module 06, Module 11, Module 15, and basic probability  
> Version 2026.5

## 1. Why this topic exists

A distribution center manager wants to know whether to authorize an overtime shift.
The decision must be made before the next week's orders and equipment failures are known.
A simulation can compare overtime with the normal schedule, but it needs plausible inputs.
A precise-looking output based on the wrong inputs is not reliable evidence.

Suppose historical orders were lower than true customer demand because products were unavailable.
Fitting those fulfilled orders as if they were unconstrained demand understates pressure on the facility.
Suppose service times include lunch breaks because the system records only assignment and completion.
Fitting those times as hands-on processing produces the opposite error.
Suppose equipment failures are recorded only when maintenance files a work order.
Ignoring unreported short stops understates disruption.
These are business-definition errors before they are statistical errors.

Input modeling translates observations and domain knowledge into stochastic quantities for a simulation.
It decides what varies, on what clock, under which conditions, and with what relationships to other inputs.
It also distinguishes variability in the world from uncertainty about the model itself.
The model must be useful for the specific decision, not merely attractive in a goodness-of-fit chart.

The operations manager owns the business decision and the consequences of an error.
The data owner explains how records were generated and what is missing.
The analyst constructs candidate models, checks their implications, and documents uncertainty.
The maintenance lead explains equipment exposure, operational states, and repair reporting.
Finance supplies decision costs and challenges unsupported savings.
A model owner approves updates and monitors whether assumptions continue to hold.

An input model is not a forecast, a staffing plan, a service target, or a customer commitment.
A forecast describes a distribution or estimate of a future quantity given information available at a date.
A plan specifies intended actions such as overtime or inventory allocation.
A target is a desired performance level such as a 95% on-time rate.
A commitment is an obligation or promise to a customer or another party.
The same numeric value may appear in each, but it has a different meaning and owner.
Do not fit a distribution to planned volume and label it actual demand.
Do not replace a contractual service commitment with an estimated service probability.

The payoff from better inputs is a better decision under uncertainty, not an impressive list of distributions.
A simple empirical model can be preferable to a complicated parametric model that misses the tail.
A parametric model can be preferable when data are sparse and its assumptions are defensible.
A scenario may be more honest than a probability when there is no credible frequency estimate.
The right choice depends on what can change the decision and what evidence exists.

## 2. Learning outcomes

After this topic you can:

1. Define the simulation input, observational unit, exposure, decision horizon, and source of each record.
   Exercise 1 checks whether an input is genuinely observed or is censored by operations.
2. Construct candidate demand, service-time, and failure models from operational records.
   Exercise 2 asks you to calculate fitted summaries and explain their units.
3. Diagnose censoring, truncation, missingness, seasonality, changing mix, and dependence.
   Exercises 3 and 4 require you to identify consequences and propose evidence-based remedies.
4. Separate random variation from parameter and structural uncertainty.
   Exercise 5 asks you to design a nested uncertainty experiment.
5. Compare candidate models using diagnostics relevant to a business decision.
   Exercise 6 asks you to select a model without relying on a single p-value.
6. Convert simulated outcomes into an operational comparison while identifying approximations.
   Exercise 7 asks for a cost calculation and a decision rule.
7. Design data controls, validation, monitoring, and model-change governance.
   Exercise 8 asks for a release and monitoring plan.

You should be able to explain the direction of bias before calculating an exact estimate.
You should also be able to say when the available data do not identify the desired quantity.
A successful model communicates what is known, what was assumed, and what could reverse the decision.

## 3. The core mental model

```text
Decision and horizon
        |
        v
Physical process + observation process + intervention history
        |
        v
Records -> lineage / definitions / exposure / quality checks
        |
        v
Conditional input models: demand, service, failures, dependence
        |                         |
        |                         +-> parameter and structural alternatives
        v
Simulation of a specified policy -> output distributions -> decision comparison
        |                                      |
        +------------ validation and monitoring <--- realized outcomes
```

The physical process produces orders, work, uptime, failures, and delays.
The observation process determines which of those events become records.
A stopped scanner, a stockout, or a policy change affects the observation process.
That process can hide or alter the very input you want to estimate.

A conditional input model describes a quantity given relevant context.
For example, pick duration may depend on order size, temperature zone, and experience.
One distribution for all records implicitly fixes the historical mixture of those contexts.
If the planned operating mix changes, the pooled distribution may no longer apply.

A simulation draws values from the input model and applies business rules.
Those draws should respect time, units, feasible ranges, and material dependence.
A scenario changes assumptions or operating conditions in a controlled way.
Multiple replications quantify simulation noise conditional on the selected input model.
They do not automatically quantify uncertainty about the input model itself.

Validation is a loop, not a certificate.
First verify that the code samples what the analyst intended.
Then compare input behavior with independent data and domain knowledge.
Finally compare simulated system outcomes with a historical period under comparable policy.
A good fit to one aggregate output does not prove every input is correct.
Different input errors can offset one another.

The decision is the final unit of usefulness.
If several plausible input models recommend the same policy, the recommendation may be robust.
If they change the ranking, the analyst should expose that instability to the decision owner.
A useful result may be a request to collect one additional field rather than approve automation.

## 4. Definitions and vocabulary

| Term | Meaning | Common misuse or unit |
|---|---|---|
| Decision horizon | Period over which a policy is evaluated | Confused with the historical fitting window; hours or days |
| Observation unit | Entity represented by one record | Treating order lines as independent orders |
| Demand | Customer need or requests before fulfillment constraints | Equating shipped units with all desired units; units per bucket |
| Arrival process | Timing and count of incoming entities | Treating a daily count model as a complete within-day arrival model |
| Interarrival time | Elapsed time between consecutive arrivals | Pooling different operating regimes; minutes |
| Service time | Resource time needed for a defined activity | Equating elapsed lead time with active work; minutes per job |
| Lead time | Elapsed time between defined milestones | Assuming it excludes waiting, holds, or batching; hours |
| Exposure | Time or cycles during which an event could occur | Using calendar time when equipment was off; operating hours or cycles |
| Failure | Defined loss or degradation of an asset's required function | Counting maintenance work orders as failures without classification |
| Repair time | Time from a stated failure milestone to a stated restoration milestone | Confusing active technician time with total downtime; minutes |
| Uptime | Time available and operating under the selected definition | Assuming it equals production time in every system |
| Downtime | Time unable to perform a stated function | Including planned maintenance without saying so; minutes |
| Rate | Expected events per unit exposure under a model | Treating events per calendar day as events per operating hour |
| Intensity | Instantaneous event rate, possibly changing over time or state | Assuming all arrival or failure processes have constant intensity |
| Distribution | Probabilities over possible values | Assuming a named family is automatically correct |
| Empirical distribution | Probability mass assigned to observed values or blocks | Assuming unseen extremes cannot occur |
| Parametric distribution | Family indexed by a finite set of parameters | Selecting by convenience without checking support or tails |
| Support | Values with nonzero model probability | Using a normal model that permits negative durations without a remedy |
| Location | A measure of central tendency or a family parameter | Assuming the mean is always the most relevant number |
| Dispersion | Spread around the center | Using variance when the simulation needs a standard deviation; squared units versus units |
| Quantile | Value at or below which a specified fraction of the distribution falls | Calling the 95th percentile a maximum |
| Tail | Low-probability region of a distribution | Ignoring it when overtime or safety stock is tail-driven |
| Count | Nonnegative integer number of events in a bucket | Treating a fraction of an order as an individual order |
| Poisson distribution | Count family with mean equal to variance for its basic form | Assuming counts are independent or equidispersed in real data |
| Exponential distribution | Nonnegative continuous family with constant hazard | Calling its mean a guaranteed life; time units |
| Weibull distribution | Flexible nonnegative lifetime family with a shape-dependent hazard | Fitting without considering censoring or operating age |
| Lognormal distribution | Positive distribution whose logarithm is normal | Assuming every positive duration is lognormal |
| Hazard | Instantaneous failure intensity among assets surviving to a given age | Confusing hazard with cumulative probability |
| Survival function | Probability that lifetime exceeds a given age | Assuming a censored observation is a failure at the censoring time |
| Censoring | Observation reveals only a bound on the event time | Discarding an asset still operating at the observation cutoff |
| Right censoring | Event time exceeds the last observed event-free time | Recording the last observed time as an exact failure |
| Left truncation | Inclusion requires survival until observation begins | Ignoring that old surviving assets are a selected sample |
| Missingness | A field or event is absent from the record | Assuming missing times are a random sample |
| Selection bias | Observed records differ systematically from the target population | Applying a model for closed jobs to all jobs when long jobs stay open |
| Dependence | Joint behavior not described by marginal distributions alone | Assuming matching each marginal reproduces congestion |
| Correlation | A particular numerical measure of association | Treating zero Pearson correlation as independence |
| Autocorrelation | Dependence across time for a series | Randomly shuffling a persistent demand sequence |
| Seasonality | Repeated change in a distribution with calendar or cycle | Modeling weekday peaks with a single constant mean |
| Nonstationarity | A distribution or relationship changes over time | Treating old and new process data as exchangeable |
| Regime | A state with distinct operating characteristics | Assuming a disruption regime has the same parameters as normal operations |
| Bootstrap | Resampling observations to approximate sampling uncertainty | Treating dependent daily observations as independent |
| Block bootstrap | Resampling contiguous groups to preserve local dependence | Choosing blocks too short to retain important persistence |
| Aleatory uncertainty | Variation represented as inherent randomness for the model purpose | Calling it removable merely by running more simulations |
| Epistemic uncertainty | Limited knowledge of parameters, data, or model structure | Treating one fitted parameter as certain |
| Calibration | Estimation or adjustment of model inputs from data | Fitting against the same outcomes later presented as independent validation |
| Validation | Evidence that a model is adequate for a specified use | Claiming a model is universally true after one test |
| Verification | Checking that implementation matches its specification | Mistaking a correct implementation for a valid model |
| Holdout | Data reserved from fitting and model selection for assessment | Repeatedly tuning on it until performance looks good |
| Common random numbers | Coupling random inputs across policies to improve comparison precision | Assuming a shared seed automatically aligns changed event sequences |
| Replication | One independently generated simulation run under stated inputs | Counting correlated outputs from the same run as independent replications |
| Scenario | A coherent set of assumptions or conditions | Assigning a probability without evidence |
| Sensitivity analysis | Study of how outputs change as assumptions or inputs change | Varying implausible inputs while ignoring decision-sensitive ones |
| Input uncertainty | Uncertainty in the distributions or parameters supplied to the simulation | Reporting only Monte Carlo error conditional on a fixed model |
| Monte Carlo error | Finite-simulation error in estimated results | Treating a narrow simulation interval as evidence that inputs are accurate |

## 5. The mechanics, step by step

### Step 1: Start with a decision and causal boundary

Input: a named owner, alternatives, decision deadline, and performance measures.
Output: an input inventory linked to the decision.
State whether the model concerns one asset, one shift, a site, or a network.
State whether the horizon is one shift, one quarter, or multiple years.
Specify service, cost, and risk measures in their operational units.
List policy alternatives before looking for an input distribution.

For an overtime decision, demand volume, processing effort, and machine downtime matter.
A detailed model of container dwell time may not matter if all containers arrive before the shift.
Conversely, arrival timing matters if a cutoff makes work arriving late impossible to finish.
The correct boundary is determined by the decision, not the available database tables.

Specify the event being simulated.
An order, order line, pallet, case, and individual unit are different entities.
Write down how entities are created, processed, routed, and completed.
Define the resources they consume and whether those resources can operate in parallel.
Record whether the decision changes arrival behavior, service speed, or failure exposure.
If it does, a fixed historical input may be invalid under the proposed intervention.

### Step 2: Map the observation process

Input: source-system schemas, event definitions, and sample records.
Output: a data dictionary and a map from the real process to recorded events.
Identify the source system and field for each input and timestamp.
Record the system owner, extraction date, data window, and time zone.
Explain when an observation begins and ends, including open records at extraction.
Separate event time from ingestion time and correction time.

Check whether the system records attempted orders, accepted orders, shipped orders, or invoices.
Trace a small set of orders from request to completion and compare fields with operations.
Check whether missing orders are missing because of stockouts or channel restrictions.
A sales history alone usually cannot identify every lost request.
Use lost-sales logs, substitutions, external signals, and explicit assumptions where available.
Do not fill the unobserved demand gap with a point estimate and call it measured truth.

For service, distinguish clock time, queue time, active processing, and interruption.
Identify scan gaps, batch completion timestamps, and rounding to the nearest minute.
Record whether simultaneous jobs can share a resource.
If only elapsed time is observable, model it as elapsed time and do not add a separate queue without checking double counting.
If active work is required but unavailable, use observations, experiments, or bounded assumptions.

For failures, define a failure mode and a restoration event before counting records.
Match maintenance records with asset-state and operating-hour records where possible.
Distinguish corrective work, preventive work, inspection, and administrative tickets.
Document work orders that cover multiple failures or multiple assets.
Separate asset age from elapsed time since the most recent repair.
The relevant clock may be calendar hours, running hours, cycles, starts, or throughput.

### Step 3: Clean without erasing process behavior

Input: raw records plus documented business rules.
Output: a reproducible analysis table and a quality report.
Keep an immutable raw extract and record every transformation.
Check duplicate identifiers, impossible sequences, missing endpoints, and unit mismatches.
Resolve daylight-saving transitions with time-zone-aware timestamps.
Keep both the raw and corrected value when a legitimate correction is made.
Count excluded rows and their business importance.

An extreme duration may represent a true difficult job, a recording error, or an interruption.
Use process evidence before removing it.
A long order waiting over a holiday should not automatically become an unusually long active service time.
A real breakdown taking many hours should not be dropped because it makes a chart inconvenient.
Report tail outcomes with and without questionable records when their status is unresolved.

Check whether corrections change the model decision, not just the fitted mean.
If excluding a handful of long stops removes the case for preventive maintenance, investigate those stops.
If late-added orders are systematically absent, a clean-looking table may still be biased.
Small samples call for more explicit assumptions, not silent data fabrication.

### Step 4: Choose time buckets and exposure

Input: process cadence, event timestamps, and policy horizon.
Output: counts, intervals, or exposure-normalized records at a defensible resolution.
Use individual timestamps when the timing of congestion matters.
Use daily counts when intra-day patterns are irrelevant to the specific decision.
Record open hours, breaks, holidays, and incomplete days alongside the counts.
A 6-hour day and a 24-hour day are not automatically comparable exposures.

Daily demand and within-day arrival times are separate modeling choices.
A daily total can be sampled first, followed by arrival times conditional on that total.
Alternatively use a time-varying arrival intensity when the data support it.
Do not assume uniform arrivals merely because only daily totals are available.
If timing matters and is unobserved, treat alternative arrival profiles as scenarios.

For failure modeling, calculate exposure before computing rates.
A machine that was shut down for preventive work was not necessarily at risk of the same operating failure.
An asset with no recorded failure still contributes its observed at-risk exposure.
Repeated failures from one asset are not automatically independent first lifetimes.
Specify whether a repair restores an asset to new, to its prior age, or to some intermediate condition.

### Step 5: Examine demand before fitting

Input: cleaned request records and relevant calendar, product, and stock-availability context.
Output: descriptive diagnostics and candidate demand-generating structures.
Plot counts over time, separated by weekday, promotion, channel, and availability as appropriate.
Examine mean, variance, zero frequency, upper quantiles, and long runs above normal.
Compare comparable periods rather than treating a launch month as an ordinary month.
Investigate whether observed zeros are genuine no-demand periods or missing extracts.

A Poisson count model may be a starting point when events are approximately independent and rare within short intervals.
For a constant-rate Poisson model, the count mean equals its variance in the same fixed exposure.
Variance much larger than the mean can reflect bursts, mixed regimes, autocorrelation, or unobserved drivers.
Variance below the mean can arise from capacity limits, appointment slots, or regular order patterns.
Neither pattern alone identifies the cause.

When appropriate, model a time-varying mean using calendar and promotion information available at the decision date.
A negative binomial count model can represent overdispersion, but it does not automatically represent a known promotion mechanism.
A zero-inflated or hurdle model may be useful for intermittent demand if the separate zero process is meaningful and identifiable.
An empirical distribution can preserve observed shapes for a stable and representative period.
A block resample can retain short-run temporal patterns if blocks are sufficiently long and representative.

For future promotions or market changes, state what is forecast and what is assumed.
The demand forecast is uncertain and may differ from the volume target.
A sales plan may influence future demand, but it is not a realization of demand.
Do not condition a backtest on promotion outcomes that were unknown at the original decision date.

### Step 6: Examine service and repair durations

Input: aligned start and end events and relevant job characteristics.
Output: candidate duration models conditional on important drivers.
Inspect the support: active duration cannot be negative, and a zero may be a real instant or a timestamp artifact.
Compare durations by activity, product family, order size, shift, and equipment condition.
Plot the median and upper tail as well as the mean.
Check whether time is rounded or top-coded.
Look for large spikes at shift boundaries and system batch-update times.

A lognormal or gamma family may describe some positive, right-skewed durations.
A mixture or conditional model can be preferable when separate workflows have different speeds.
A deterministic setup time plus a variable per-unit time may have a useful physical interpretation.
An empirical distribution avoids imposing a parametric shape but cannot generate values outside its observed support.
With a small sample, a single historical maximum is a poor estimate of the worst plausible delay.

If jobs remain open at extraction, completed jobs are a selected sample.
Their durations may systematically exclude the longest jobs.
Use appropriate survival methods or wait for completion when feasible.
Do not treat time observed so far as the final duration of each open job.
Service times can be correlated across jobs sharing congestion, staff, a product mix, or a shift.
Resampling individual records independently destroys that shared context.

Repair durations need their own scope.
A repair may include detection, technician dispatch, diagnosis, parts wait, actual work, and restart checks.
Whether those components overlap with other work determines how to represent them.
A scheduled parts shortage can produce a repair tail not visible in technician wrench time.
Where repair capacity is constrained, represent shared technicians rather than assuming every repair begins immediately.

### Step 7: Model failure occurrence with censoring

Input: event histories, asset exposure, censoring flags, and a repair interpretation.
Output: a failure process and a documented exposure basis.
For a first failure from new under an exponential lifetime model, the survival function is
$S(t)=\exp(-\lambda t)$, where $t$ is exposure and $\lambda$ is failures per exposure unit.
Its hazard is constant and its mean lifetime is $1/\lambda$.
The constant-hazard assumption is a modeling choice, not a physical law.

For independent first lifetimes with exact failure times and right-censored observations,
the exponential likelihood is proportional to $\lambda^d \exp(-\lambda E)$,
where $d$ is the number of failures and $E$ is total exposure through failure or censoring.
When $d>0$, the maximum-likelihood estimate is $\hat\lambda=d/E$.
The right-censored assets contribute exposure but not an additional failure.
If there are no failures, the estimate is on the boundary at zero and $1/\hat\lambda$ is not a finite reliable lifetime estimate.
Use a confidence bound or a justified prior and gather more evidence rather than asserting infinite reliability.

The same count-over-exposure rate estimate can arise for a homogeneous Poisson recurrent-event process.
That is a different physical model from independent first lifetimes.
For recurrent failures, specify risk exposure, downtime exclusion, and repair effect explicitly.
A minimal-repair process can differ materially from a replacement process with changing hazard.
Do not pool first lifetimes and recurrent events without a coherent likelihood.

A Weibull lifetime model allows hazard to increase or decrease with age according to its shape.
Its fitted parameters require an appropriate censored likelihood and a sufficiently informative sample.
When new assets enter observation already aged, consider left truncation and the selection of survivors.
If exposure is observed only after asset installation, document the age at entry.
Check whether preventive replacement removes assets before failure for reasons related to their condition.
Informative censoring can bias ordinary survival estimates and may require an explicit observation model or sensitivity analysis.

### Step 8: Select candidate models using multiple kinds of evidence

Input: candidate families, training data, and decision-critical output measures.
Output: a preferred model and plausible alternatives, with reasons.
First reject distributions with impossible support or implausible mechanics.
Then compare descriptive summaries, histograms or empirical cumulative distributions, and tail quantiles.
Use probability or quantile plots where appropriate and mark uncertainty from small samples.
Compare the observed and modeled frequencies of decision-relevant events.
For capacity, those events may be demand above a staffing threshold or repair time longer than a spare-parts lead time.

A goodness-of-fit test is not a universal acceptance certificate.
Small samples have low power to detect an important mismatch.
Large samples can reject a harmless mismatch that has no decision impact.
If parameters are estimated from the tested sample, account for that when using a test or its reference distribution.
Use out-of-time performance and operational plausibility as well as numerical fit.
Information criteria can compare specified likelihood models fitted to the same observations, but they do not validate unobserved tails.

Use domain knowledge to challenge the model without turning judgment into an undocumented override.
Ask whether the model permits values below setup time or above physical capacity.
Ask whether the shape remains credible in a new operating regime.
Document disagreements and how each alternative affects the policy choice.
A model may be adequate for choosing next-week overtime but inadequate for sizing a ten-year facility.

### Step 9: Preserve dependence and causal ordering

Input: time-aligned demand, service, resource, and failure observations.
Output: a joint sampling scheme, not just a list of marginal distributions.
Measure dependence at the level relevant to the simulation: job, shift, day, asset, and site.
High-demand days may have larger orders, slower processing, or both.
Multiple machines may fail together because they share a power supply or operating environment.
Transport lead times across suppliers may rise together during a regional disruption.
Independently sampling all marginals can understate simultaneous stress.

Estimate and inspect scatterplots, cross-tabulations, lagged relationships, and conditional summaries.
Pearson correlation alone misses nonlinear and tail dependence.
Use domain structure to decide which variables can influence which other variables.
Never use an outcome from after the decision as an input supposedly known before the decision.

One straightforward joint approach resamples entire historical shifts with demand, workload, and downtime together.
That preserves observed within-shift relationships but cannot produce unseen combinations or new regimes.
A conditional approach samples a regime or covariate first, then dependent inputs given that state.
A multivariate or copula approach can be useful with sufficient data and appropriate validation, especially for tail co-movement.
A time-series model can preserve persistence and lagged responses.
Specify what dependence each method preserves and what it destroys.

When evidence is sparse, create plausible stress scenarios rather than inventing precise correlations.
For example, compare independent inputs with a scenario where high demand coincides with slower service.
Do not assign a probability to the stress case unless the probability has a defensible source.
If the recommendation changes between those cases, report the dependency as a decision risk.

### Step 10: Separate sources of uncertainty

Input: fitted data, candidate structures, and the simulation design.
Output: uncertainty decomposition and an experiment plan.
Within a selected input model, each future shift can vary because arrivals and durations are random.
That is aleatory variability relative to the chosen model.
The fitted mean demand or failure rate is also uncertain because the historical sample is finite.
The choice between a constant-rate and a regime-dependent model is structural uncertainty.
Future process changes and unobserved lost demand introduce additional scenario uncertainty.

A nested analysis samples plausible input parameters in an outer loop and runs simulation replications in an inner loop.
For each outer draw, retain the sampled parameters throughout the appropriate simulation horizon.
If a daily rate represents a fixed but unknown facility characteristic, redrawing it independently for every order changes the model.
If rates truly vary by shift, represent that separately as process variability.

Bootstrap independent units only when independence is plausible.
Use shift-level or block resampling when observations share shift conditions or time dependence.
For censored failure data, use a resampling or parametric procedure that retains the censoring structure.
If the source data are too sparse to support a stable bootstrap, show that instability rather than hiding it.
Expert ranges can supplement data when elicitation, evidence, and limitations are documented.

Monte Carlo confidence intervals answer a conditional simulation-precision question.
They do not include structural model uncertainty unless the experiment explicitly varies structure.
A prediction interval for future shift outcomes answers a different question from a confidence interval for mean cost.
Label the interval, population, horizon, and uncertainty components.

### Step 11: Validate the inputs and the resulting system

Input: locked model specification, independent data, and operational expertise.
Output: evidence for a stated decision use and a list of unresolved risks.
Verify sampling code against known distribution properties and small hand calculations.
Check the simulated count, duration, exposure, and unit conventions separately.
Confirm that a censored observation is not accidentally treated as a failure.
Check that the initial state and warm-up treatment match the decision.

Reserve an out-of-time period when the process and policy are comparable.
Compare observed and generated means, spreads, zero rates, tails, and time patterns.
Compare combinations such as high demand with slow service, not merely individual marginals.
Then compare system outputs such as late orders, utilization, queue lengths, and downtime.
A model that matches average throughput but misses backlog peaks may be unsuitable for staffing.

Ask operators whether simulated event sequences are physically plausible.
Trace individual replications through the model to expose impossible event orderings.
If the holdout reveals a mismatch, revise using a development period and reserve fresh evidence if feasible.
Repeated tuning against the same holdout makes it a development set.
When no independent period exists, acknowledge the validation limit and use prospective shadow evaluation.

### Step 12: Compare decisions and prepare handoff

Input: validated inputs or explicitly qualified alternatives, cost definitions, and policy constraints.
Output: a decision recommendation with uncertainty and conditions for reversal.
Use a decision table with expected cost, service-risk measures, and any non-negotiable constraints.
Price lost demand, late work, overtime, downtime, and safety consequences consistently.
Do not count the same lost order as both backlog and permanently lost revenue unless the accounting supports it.
State whether costs are incremental, fixed, avoidable, or sunk.

When comparing policies, use paired scenarios or properly aligned common random numbers if appropriate.
Pairing reduces noise when outcomes respond similarly to shared random conditions.
A common seed alone does not guarantee alignment if a policy changes the sequence of random draws.
Estimate uncertainty for the paired difference, not just two separate policy means.
Check whether policy changes also change the input distributions or exposure.

A release package includes data lineage, units, distributions, fitted parameters, code version, validation period, and known limitations.
It also includes the selected policy, its owner, trigger conditions, and a manual override.
Monitor new data for regime shifts, missingness, and deviations in decision-relevant tails.
Refit on a defined schedule or trigger, subject to review and backtesting.
Retain prior model versions so past decisions can be reproduced.

### Decision pattern: intraday demand and release cutoffs

A daily total does not tell you when work arrives.
Consider a facility that closes its dispatch gate at 18:00 local time.
An order arriving at 08:00 can consume capacity throughout the day.
The same order arriving at 17:55 has only five minutes before the gate closes.
Both contribute one order to the daily total, but their service implications differ.

Define the cutoff event and whether it applies to order receipt, pick completion, or vehicle departure.
Then identify the earliest time at which an order can begin service.
If the data contain event timestamps, group arrivals into operational intervals.
One reasonable starting point may be 30-minute intervals, but the right resolution depends on the decision.
A staffing plan that changes only by shift may not justify one-minute parameter estimates.
A dock appointment system with a 15-minute cutoff may require finer resolution.

For each interval, calculate the count and the number of comparable intervals observed.
Separate days with reduced hours, system outages, and unusual promotions.
Inspect whether the peak is stable in clock time or follows another event such as a sales release.
Use local time for business schedules and preserve an unambiguous time-zone-aware timestamp underneath.
An apparent one-hour shift around daylight saving time may be a conversion artifact.

One model samples a daily count and allocates it among intervals using observed interval shares.
That preserves the sampled daily total by construction.
It also assumes the interval allocation mechanism is adequately described by the chosen shares.
If busy days have a more concentrated afternoon peak, use shares conditional on day type or volume.
If the data do not support conditioning, show a concentrated-arrival stress scenario.

Another model specifies an arrival intensity that changes over the shift.
It can generate individual event times and allows peaks to move with known covariates.
It requires enough timestamped observations to estimate the changing intensity.
It also requires a separate representation of dependence if arrivals cluster beyond the specified intensity.
Check generated interarrival times and peak interval counts against holdout observations.
Do not validate only against the daily total.

Input for the decision: timestamped orders, open hours, cutoff rules, and a comparable historical period.
Output: a distribution of arrivals by interval under each stated day type.
Branch point: if intraday timing does not change any policy comparison, a daily aggregate may be adequate.
If it changes cutoff misses, retain the temporal structure or collect the missing timestamps.

### Decision pattern: product mix and service work

An average job is not necessarily an actual job.
A facility may process a mix of single-line replenishment orders and multi-line emergency orders.
Suppose a single-line order averages 5 minutes of active work and a complex order averages 17 minutes.
If 80% are single-line and 20% are complex, the weighted mean is 7.4 minutes per order.
That calculation describes the mix only if those proportions apply to the target period.

If a promotion changes the mix to 60% single-line and 40% complex, the mean becomes 9.8 minutes per order.
The change is 2.4 minutes per order without any change to either within-type mean.
At 100 orders per shift, expected active work rises by 240 minutes per shift.
Adding another operator may therefore be a mix response rather than evidence that operators slowed down.
The arithmetic is an illustration; actual service distributions and mix must be measured.

A useful input model first samples or forecasts the mix.
It then samples service conditional on the selected order type.
If order size is numeric, consider a physically meaningful setup-plus-work model.
For example, total active minutes could equal setup minutes plus minutes per picked line.
The residual variability should still be checked for dependence, skew, and changing variance.
Do not extrapolate a linear per-line relationship far beyond observed order sizes without evidence.

Order mix can depend on demand volume, promotion, channel, and customer behavior.
Sampling mix independently of the demand regime may miss that relationship.
On a high-volume day, both the number and complexity of orders may be elevated.
That creates more extreme workload than a model that independently mixes all historical jobs.
Use shared day-level drivers, conditional sampling, or resampled complete shifts as appropriate.

Input for the decision: job type, line count, active time, staffing context, and future mix assumptions.
Output: a conditional workload distribution and uncertainty in its composition.
Branch point: if mix is stable and the pooled model performs on holdout, a pooled model may suffice.
If mix is changing, preserve the drivers that explain the difference.
Explain whether any planned process redesign changes those drivers.

### Decision pattern: spare parts and repair delays

Maintenance decisions often depend on both when failures occur and how long restoration takes.
A machine can fail rarely but create severe disruption when a replacement component is unavailable.
A frequent minor fault can have a high event count and little business impact per event.
The policy comparison needs both occurrence and consequence.

Construct a timeline from failure detection to restoration.
Possible milestones include alarm, diagnosis, parts request, parts arrival, repair start, and restart.
Some intervals can overlap; others are strictly sequential.
Technician availability, spare inventory, and supplier lead time affect different parts of the timeline.
A single fitted downtime distribution can hide which policy lever changes the outcome.

For example, holding one spare part can reduce parts waiting for the first failure.
It does not reduce diagnostic time or guarantee a technician is available.
If the part is replenished slowly, a second failure may occur before the spare is replaced.
If failures share a common cause, the spare can be consumed during a regional disruption.
The model should represent the relevant inventory and resource rules before claiming a spare's value.

Check whether repair records include planned maintenance.
Check whether downtime ends when a work order closes or when production actually resumes.
The work-order close date can be later than the physical restart for administrative reasons.
Conversely, an asset marked repaired may need a qualification run before it is available.
Model the milestone that matters to the decision.

Sample repair duration conditional on failure mode if the data support that distinction.
If a particular mode has few events, report its uncertainty and combine information only with justification.
A hierarchical model may share information across similar assets but depends on a defensible similarity assumption.
Do not pool a minor sensor reset and a major gearbox replacement merely because both are called repairs.

Input for the decision: failure mode, exposure, restoration stages, spare availability, and technician capacity.
Output: the distribution of downtime and recovery under each spare-parts policy.
Branch point: use a single repair distribution only when component detail does not alter the decision.
Escalate if the recommendation depends on a rare repair tail that the available records do not identify.

### Decision pattern: incomplete and changing history

A dataset can be both large and unrepresentative.
Ten years of records from an obsolete line may be less useful than six months after a redesign.
The effective sample for the current decision is the comparable evidence, not the total row count.
Document what changed, when it changed, and whether comparable assets remain.

Consider a scanner migration at the start of a quarter.
Before migration, events were uploaded in batches at the end of the shift.
After migration, events are sent near real time.
An apparent reduction in recorded service duration may reflect measurement rather than process improvement.
Compare physical observations with both systems and map the two timestamp definitions.
Keep a regime indicator until comparability is demonstrated.

A second change might alter the physical process itself.
New conveyor equipment can change speed, failure modes, and bottlenecks.
The pre-change failure rate should not be reused automatically for the new asset.
Engineering specifications and vendor tests may be informative, but their operating conditions may differ.
Use them as documented evidence or scenarios, not as site-specific observations.

With a short post-change history, avoid claiming a stable rare-event tail.
Compare plausible parameter ranges and alternative assumptions.
Prioritize measurements that reduce the uncertainty most likely to affect the decision.
For a service-level decision this may be high-percentile processing time rather than another estimate of the mean.
For a maintenance decision it may be running-hour exposure and restoration time rather than ticket count.

Input for the decision: change log, measurement-system history, and post-change observations.
Output: an explicit comparable population and a set of alternatives for unsupported relationships.
Branch point: pool periods only if both the measurement process and relevant operational behavior are comparable.
If not, stratify, adjust with evidence, or report a restricted scope.

### Decision pattern: choosing a validation threshold

Validation criteria should be agreed before examining the final holdout.
Otherwise an analyst can keep adjusting the criterion to match the observed result.
The criterion must relate to the decision rather than a convenient global statistic.
For overtime, errors around the capacity threshold may matter more than errors on very quiet shifts.

One possible diagnostic compares the observed and simulated share of shifts above 240 minutes of workload.
Another compares the 90th percentile of workload and the length of consecutive overloaded runs.
A third checks whether overloads cluster with promotion days as they do in the historical period.
All three depend on a comparable measurement of workload and a sufficiently large holdout.
An observed holdout rate based on five shifts is itself very uncertain.

Specify tolerances in operational terms before fitting when possible.
For example, a model owner might require that a discrepancy not change the preferred staffing policy across plausible costs.
That is a decision-stability criterion, not a universal numerical accuracy rule.
Also specify what happens if the criterion fails: revise the model, collect data, or use a conservative manual plan.
Do not declare success simply because there is no statistically significant difference.

Use calibration plots or summary tables with uncertainty bands appropriate to the data.
Investigate the direction, magnitude, and business consequences of discrepancies.
If several comparisons are made, avoid selecting only the best-looking one.
Retain an audit trail of failed diagnostics and model revisions.
When historical policy differs from the proposed policy, validation of observed outcomes does not establish counterfactual validity.

Input for the decision: a reserved comparable period and decision-relevant performance measures.
Output: an evidence statement tied to a specified scope, horizon, and acceptance criterion.
Branch point: if the model fails on a noncritical feature, document why the decision is unaffected.
If it fails on a decision-sensitive feature, do not rely on the apparent precision of its recommendation.

### Decision pattern: choosing the next measurement

Data collection has a cost, so not every uncertain input deserves the same attention.
Rank input gaps by plausibility, consequence, and whether better data can actually be obtained.
An uncertain input that does not affect the recommendation may be a low-priority measurement target.
A modest uncertainty near a policy threshold can be more valuable to resolve.

Start with two or more defensible values or structures for the uncertain input.
Run the same decision comparison under each, holding other assumptions explicit.
Record whether the preferred policy, service constraint, or cost ranking changes.
If the decision changes, identify which observation would distinguish the alternatives.
Examples include logging operating hours, capturing denied requests, or measuring actual work start times.

Consider measurement error and implementation effects before launching a collection effort.
Manual timing can change operator behavior or miss pauses.
A new sensor can fail most often in the same harsh conditions that cause equipment failure.
A required reason code can be selected inconsistently under workload pressure.
Pilot the measurement protocol and compare it with an independent source where feasible.

Estimate the cost of delaying the decision as well as the potential benefit of information.
An urgent safety decision may need an immediate conservative action rather than a long study.
A reversible overtime reservation may allow a short pilot with prospective outcome monitoring.
A facility investment with a long useful life may justify more extensive evidence.
The value of information depends on whether new evidence can actually change the action.

Input for the decision: current policy comparison, uncertainty ranges, and feasible measurement options.
Output: a prioritized collection plan with an owner, instrument, duration, and success criterion.
Branch point: collect when the expected decision benefit can justify the effort and delay.
Otherwise retain the uncertainty explicitly and choose a robust action or a qualified recommendation.

## 6. Worked example

### Decision and boundaries

A regional parts distributor must decide whether to reserve overtime for one packing station.
The decision concerns the next 240-minute shift and is made before orders arrive.
One operator processes the station's jobs serially.
The company can reserve an additional 60 minutes of overtime at a fixed incremental cost of USD 55 per shift.
The base option has no reserved overtime.
Each unfinished job at the end of the available work window incurs an estimated USD 18 incremental deferral cost.
The cost is a planning estimate, not a customer penalty or guaranteed revenue loss.
There is no contractual service constraint in this illustration.

The forecast is a probability distribution for next-shift demand, not the realized order count.
The plan is either 240 minutes or 300 minutes of available processing time.
The target is to finish all accepted jobs, but this target is not guaranteed by either plan.
No customer commitment is implied by the exercise.
Historical observations below are synthetic teaching data, not evidence about a real company.

The station uses one machine whose failures interrupt the operator's work.
For illustration, each failure consumes 15 minutes of capacity.
Failure exposure is the scheduled 240-minute base shift, including in the overtime option.
Failures during overtime are excluded from this deliberately simplified calculation.
This assumption may favor overtime and needs review before a real decision.
The model also assumes all work is known at the start of the shift.
It represents work remaining in fractional job-equivalents, not individual customer completion times.

### Demand data

Eight comparable shifts recorded the following counts of accepted jobs.
Assume for the exercise that no demand was lost or rejected, and that each shift had the same exposure.

| Shift | Observed jobs per 240-minute shift |
|---:|---:|
| 1 | 22 |
| 2 | 25 |
| 3 | 19 |
| 4 | 28 |
| 5 | 24 |
| 6 | 27 |
| 7 | 21 |
| 8 | 26 |

The total is 192 jobs across 8 shifts.
The sample mean is $192/8=24$ jobs per shift.
The squared deviations from 24 sum to 68 jobs squared.
The sample variance is $68/7=9.714$ jobs squared per shift squared.
The sample standard deviation is approximately 3.117 jobs per shift.
These data are less dispersed than a Poisson distribution with mean 24.
Eight shifts are far too few to conclude that this difference is stable.
An empirical distribution is a natural baseline for the demonstrated code.
A Poisson model with mean 24 is a useful alternative, not a proven best model.

A real study would check weekday, channel, promotions, missing requests, and arrivals within each shift.
If the next shift has an unusually large promotion, these eight ordinary shifts may not be representative.
If lost demand exists, the accepted-job count cannot simply be equated with unconstrained demand.

### Service data

Six independently sampled completed jobs had active processing times of 6, 8, 7, 9, 10, and 8 minutes.
The total is 48 minutes across 6 jobs.
The mean active service time is $48/6=8$ minutes per job.
The squared deviations from 8 total 10 minutes squared.
The sample variance is $10/5=2$ minutes squared.
The sample standard deviation is approximately 1.414 minutes per job.
The sample is illustrative and too small to establish a tail or independence.
The empirical service distribution assigns probability 2/6 to 8 minutes and 1/6 to each other observed duration.
It cannot generate a job longer than 10 minutes.

A first workload check at the mean is $24\times8=192$ minutes per shift before failures.
The nominal 240-minute shift therefore has 48 minutes of spare time at mean demand and service.
That is not a guarantee: variable demand, variable job times, and downtime can exceed the spare time.
A ratio of average available time to average work is not a tail-service measure.

### Failure data

Four comparable independent new assets were observed on an operating-minute clock.
Three failed after 100, 120, and 80 operating minutes respectively.
The fourth was still operating at the observation cutoff after 140 operating minutes.
The fourth lifetime is right censored at 140 minutes, not an observed 140-minute failure.
For the calculation, assume independent exponential first lifetimes, noninformative censoring, and common constant hazard.

Total at-risk exposure is $100+120+80+140=440$ operating minutes.
The number of failures is 3.
The estimated rate is $3/440=0.00681818$ failures per operating minute.
The corresponding estimated exponential mean lifetime is $440/3=146.667$ operating minutes.
Without censoring correction, averaging just the three observed failure times gives 100 minutes.
That naive mean discards the fourth asset's 140 minutes of failure-free exposure.
It is not the same estimator and would misrepresent the specified likelihood.

For the shift model only, make an additional, strong transfer assumption:
use the estimated constant rate as a homogeneous Poisson recurrent-failure rate per scheduled base-shift minute.
A first-lifetime estimate alone does not prove that recurrent failures follow a Poisson process.
The transfer ignores downtime removal from exposure and repair effects.
Under this illustrative assumption, the expected failure count in 240 scheduled minutes is
$240\times3/440=1.63636$ failures per shift.
At 15 minutes per failure, expected downtime is approximately 24.545 minutes per shift.
The mean workload plus mean downtime is approximately $192+24.545=216.545$ minutes.
The resulting 23.455-minute mean slack is not the same as expected unused capacity or a service guarantee.
The nonlinear backlog calculation depends on the joint distribution, not only on that mean.

### Decision calculation

For each simulated shift, draw a historical demand count with replacement.
For each job, draw a historical active service time with replacement.
Draw failures from the illustrative Poisson count model and multiply by 15 minutes.
Let total required time be service minutes plus failure minutes.
The base window is 240 minutes; the overtime window is 300 minutes.
Approximate unfinished work as $\max(0,\text{required minutes}-\text{window minutes})/8$ job-equivalents.
This converts remaining time at the sample mean rate; it does not count actual unfinished discrete jobs.
The base cost is USD 18 times unfinished job-equivalents.
The overtime cost is USD 55 plus USD 18 times unfinished job-equivalents.

For a particular hypothetical draw with 28 jobs, 230 service minutes, and 3 failures,
required time is $230+3\times15=275$ minutes.
The base policy leaves $275-240=35$ minutes, or $35/8=4.375$ job-equivalents.
Its illustrated cost is $4.375\times\text{USD }18=\text{USD }78.75$.
The overtime policy leaves no unfinished work and costs USD 55.
Overtime saves USD 23.75 on that draw, not necessarily on average.
If required work is 230 minutes, the base cost is USD 0 and overtime costs USD 55.
Averaging paired draws estimates the comparison over the stated input model.

Approve overtime only if its estimated benefit and service implications justify the USD 55 reservation,
after accounting for uncertainty and any decision-maker constraints.
The calculation below reports the conditional comparison rather than asserting that eight shifts prove a policy.
Before operational approval, investigate whether the simple approximation and sample represent the upcoming shift.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the summaries above and simulates the two policies on the same draws.
Save it as a local `.py` file and run it with Python 3.
The fixed seed makes a run reproducible in a compatible Python environment.
The confidence interval describes finite-replication error for the paired mean cost difference only.
It does not include uncertainty in the empirical samples, failure-rate estimate, or model structure.
The calculation does not implement a discrete-event queue, job priorities, or overtime failures.

```python
import math
import random
import statistics

# Synthetic records; demand is jobs per 240-minute shift.
demand = [22, 25, 19, 28, 24, 27, 21, 26]
# Active service time per completed job, in minutes.
service = [6, 8, 7, 9, 10, 8]
# First-failure times and one right-censored lifetime, in operating minutes.
failures = [100, 120, 80]
censored = [140]

base_minutes = 240.0
overtime_minutes = 60.0
repair_minutes = 15.0
overtime_cost_usd = 55.0
deferral_cost_usd_per_equivalent = 18.0
replications = 30000
rng = random.Random(202605)

exposure_minutes = sum(failures) + sum(censored)
rate_per_minute = len(failures) / exposure_minutes
failure_count_mean = rate_per_minute * base_minutes
service_mean = statistics.mean(service)


def poisson_count(mean, generator):
    """Knuth sampler; suitable here because the Poisson mean is small."""
    threshold = math.exp(-mean)
    product = 1.0
    count = 0
    while product > threshold:
        count += 1
        product *= generator.random()
    return count - 1


def cost(required_minutes, window_minutes, reserve_cost):
    remaining_minutes = max(0.0, required_minutes - window_minutes)
    job_equivalents = remaining_minutes / service_mean
    return reserve_cost + deferral_cost_usd_per_equivalent * job_equivalents


base_costs = []
overtime_costs = []
savings = []
base_late = 0
overtime_late = 0
for _ in range(replications):
    jobs = rng.choice(demand)
    processing = sum(rng.choice(service) for _ in range(jobs))
    breakdowns = poisson_count(failure_count_mean, rng)
    required = processing + breakdowns * repair_minutes
    base = cost(required, base_minutes, 0.0)
    overtime = cost(required, base_minutes + overtime_minutes, overtime_cost_usd)
    base_costs.append(base)
    overtime_costs.append(overtime)
    savings.append(base - overtime)
    base_late += required > base_minutes
    overtime_late += required > base_minutes + overtime_minutes

mean_savings = statistics.mean(savings)
se_savings = statistics.stdev(savings) / math.sqrt(replications)
# Normal approximation for the Monte Carlo interval of the paired mean.
half_width = 1.96 * se_savings
print(f"Demand mean: {statistics.mean(demand):.3f} jobs/shift")
print(f"Demand sample variance: {statistics.variance(demand):.3f} jobs^2/shift^2")
print(f"Service mean: {service_mean:.3f} minutes/job")
print(f"Service sample variance: {statistics.variance(service):.3f} minutes^2/job^2")
print(f"Failure rate: {rate_per_minute:.6f} failures/minute")
print(f"Mean failure count: {failure_count_mean:.3f} failures/base shift")
print(f"Mean base cost: USD {statistics.mean(base_costs):.2f}/shift")
print(f"Mean overtime cost: USD {statistics.mean(overtime_costs):.2f}/shift")
print(f"Paired savings: USD {mean_savings:.2f}/shift")
print(f"Monte Carlo 95% interval: USD [{mean_savings-half_width:.2f}, "
      f"{mean_savings+half_width:.2f}]/shift")
print(f"Base shifts with unfinished work: {base_late/replications:.3%}")
print(f"Overtime shifts with unfinished work: {overtime_late/replications:.3%}")
print("Positive savings favor overtime under this illustrative model.")
```

Expected deterministic summary lines are a demand mean of 24.000 jobs per shift,
a demand sample variance of 9.714 jobs squared per shift squared,
a service mean of 8.000 minutes per job,
a service sample variance of 2.000 minutes squared per job squared,
a failure rate of 0.006818 failures per minute,
and a mean of 1.636 failures per base shift.
The remaining lines are simulated cost and unfinished-work summaries.
Overtime cannot increase the unfinished-work fraction on a paired draw in this model,
but its reserved cost means that its mean cost need not be lower.
The 95% interval uses a large-sample normal approximation for the mean paired difference.

To explore structural uncertainty, replace the empirical demand draw with a justified alternative,
change the repair-duration assumption, or use a joint shift resample if complete shift records exist.
Do not interpret such changes as equally likely without evidence.
To model input uncertainty, add an outer resampling loop over independent source units,
refit the parameters, and use inner replications for each fitted model.
Preserve the relationship between event and censoring records in a survival analysis.

## 8. Data, standards, and regulatory context

### Data contract

A usable data contract starts with the operational definition of each event.
For demand, specify the difference between request, acceptance, shipment, return, and cancellation.
For service, specify start and end milestones and whether breaks or waiting are included.
For failures, specify asset identity, failure mode, event time, restoration time, and exposure clock.
A field name such as `duration` is insufficient without its start, end, unit, and inclusions.

At minimum retain the source, extraction timestamp, system time zone, and update cadence.
Maintain a stable event identifier and a version or correction indicator when the source permits it.
Record data-quality controls for duplicate events, late-arriving updates, and missing intervals.
Reconcile totals to an authoritative operational report, explaining valid differences.
A total can reconcile even when timestamps or categories are wrong, so reconciliation is necessary but not sufficient.

Privacy matters even in a capacity model.
Individual operator identifiers, schedules, and job histories can reveal personal performance information.
Collect only the attributes needed for the stated analysis and restrict access appropriately.
Aggregate results where feasible without destroying important process differences.
Retention, lawful use, and employee-notice obligations depend on jurisdiction and facts.
The organization must obtain its own legal and labor-relations advice.

### Event and enterprise standards

GS1 EPCIS 2.0 is a standard for sharing visibility event data about objects and business processes.
Its event vocabulary can help record what happened, when, where, and in which business context.
It does not establish the truth of a scan or automatically infer unobserved demand.
Its event data must still be mapped to the observation unit and process definitions used in the simulation.
Use the applicable GS1 specification and trading-partner agreements for implementation details.

The ISA-95 family addresses integration between enterprise and control systems.
It offers a framework for organizing manufacturing operations information and interfaces.
Using ISA-95 terminology does not by itself define a site's failure threshold or prove timestamp quality.
Equipment state mappings and exposure rules remain local modeling decisions.

ISO 22400 addresses key performance indicators for manufacturing operations management.
A stated key performance indicator (KPI) definition can help distinguish availability and related measures.
Do not assume an organization calculates every KPI identically merely because a standard exists.
Record the numerator, denominator, included states, aggregation window, and local deviations.

ISO 14224 provides principles and requirements for collecting and exchanging reliability and maintenance data for equipment in petroleum, petrochemical, and natural gas industries.
Its equipment taxonomy and failure-data concepts can inform a reliability data dictionary.
The standard's domain and edition matter; it is not automatically a legal requirement for every warehouse.
A firm may adopt a standard by contract or policy, and local legal obligations can differ.

### Censoring, missingness, and representativeness

An active job at extract time is right censored if the outcome of interest is completion time.
A machine still operating is right censored for its first failure time.
Censoring has an informative contribution only if the analyst retains the observation and its event-free exposure.
If reporting stops because an asset appears likely to fail, ordinary noninformative-censoring assumptions may be invalid.

Data may be missing because a scanner lost network connectivity during congestion.
Data may be delayed because operators complete records at the end of a shift.
Data may be absent because the business never recorded lost customer requests.
These mechanisms differ and should not receive the same imputation rule.
Compare observed missingness across time, site, workload, and asset state when possible.
Flag unidentifiable quantities and assess plausible bounds or scenarios.

Historical records reflect historical operating policies.
Expedited jobs may have been assigned to the fastest staff.
Maintenance may have been scheduled precisely because an asset was already deteriorating.
Such selection can confound a naive comparison of policy outcomes.
A distribution fitted to historical behavior is not automatically valid under a changed policy.
Record intervention history and consult causal or experimental expertise for consequential changes.

### Regulatory interpretation

Standards and documentation practices are not the same thing as statutory obligations.
Legal requirements depend on activity, location, employment context, contracts, and current law.
A plant's safety rules may constrain machine restart, staffing, or maintenance in ways a cost model must not override.
A simulation recommendation is not permission to bypass a safety requirement.
The named standards in this section are technical references, not a claim of universal legal applicability.
Verify the current edition and applicable contractual or regulatory status before implementation.

## 9. Industry practice and case evidence

A recurring industrial pattern is to connect event data to an operational decision rather than model every process in equal detail.
For input modeling, this means starting with decision-sensitive quantities and their observation quality.
Real implementations combine operational histories, engineering knowledge, scenario analysis, and ongoing review.
The particular probability families and benefits must be established for the organization and use case.

### Reliability data practice

ISO 14224 provides a framework for collecting reliability and maintenance data in its specified industry scope.
Its equipment and event definitions help distinguish failure behavior from undifferentiated maintenance records.
It provides a reference point for specifying failure modes, exposure, and model assumptions.
It does not establish a specific failure distribution for the illustrative packing station.

NIST's Engineering Statistics Handbook describes survival analysis, censoring, reliability functions, and common lifetime models.
Its exponential model is the basis for the constant-hazard explanation in this topic.
Its methods also show why censored operating time carries information.
The fictitious example's transfer to repeated failures is an additional teaching assumption, not a result of the NIST source.

### Supply-chain event practice

GS1 describes EPCIS as a way for applications to capture and share visibility event information.
Its stated purpose supports a data lineage discussion across business partners.
A shared event format does not guarantee that all partners observe the same physical process.
[INFERENCE] If a distributor obtains more consistent event milestones from partners, it may be easier to separate waiting from transit time.
That inference depends on the partners actually capturing the relevant milestones accurately.
No quantified service improvement is claimed here.

### Simulation and input uncertainty

The simulation literature distinguishes input-model uncertainty from uncertainty due to finite simulation replications.
Barton and Schruben's work on resampling methods is a published reference for incorporating input uncertainty in simulation analysis.
Its relevance is the methodological distinction, not an assertion that one bootstrap scheme suits every dependent or censored dataset.
Analysts must choose resampling units and dependence assumptions that match their own data-generating process.

### Using evidence responsibly

A published case can motivate a design, but it cannot supply your facility's parameters.
A reported reduction in downtime elsewhere cannot be used as this site's failure rate.
A vendor case study may describe selected successes and omit failed deployments.
Check the source, dates, population, baseline, measurement, and independence of evaluation.
When a public report is silent on sampling or uncertainty, say so rather than filling the gap.

For an internal decision, maintain an evidence register.
Mark measured values, model estimates, expert judgments, and scenarios separately.
Record the owner and date of each assumption.
Escalate assumptions that are both weakly supported and capable of reversing the decision.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Fulfilled orders treated as demand | Sales fall exactly when availability falls | Stockouts suppress observations | Capacity or stock is undersized | Combine availability and lost-request evidence; scenario the unidentified gap |
| Planned volume treated as realized demand | Input equals the budget every week | A target replaces a random quantity | No realistic overload risk | Separate forecast, plan, target, and commitment |
| Incomplete days pooled with full days | Low counts cluster on holidays | Exposure differs by bucket | Mean arrival rate is understated | Normalize exposure or condition on operating calendar |
| Lead time used as active service | Long durations align with shift breaks | Waiting is mixed with work | Queueing is counted twice | Separate milestones and interruption time |
| Open jobs dropped | Recent records have unusually short durations | Long jobs remain incomplete | Tail and workload are underestimated | Retain censoring and use appropriate duration methods |
| Censored asset marked failed | Failure count increases at extract cutoff | Censoring is coded as event | Failure rate is inflated | Store explicit event indicator and exposure |
| No-failure assets discarded | Only work-order assets appear | Event-free exposure is missing | Failure rate is inflated | Reconcile asset population and operating exposure |
| Calendar hours used as running hours | Idle assets appear unusually reliable | Non-risk time enters denominator | Failure rate per operating hour is understated | Define at-risk operating states |
| Recurrent and first failures pooled | Repair history is absent | Different event processes are mixed | Invalid hazard and maintenance decision | Choose a recurrent-event or lifetime model explicitly |
| Independent inputs under shared stress | Simulated peak backlog is too low | Joint high-demand and slow-service days disappear | Overtime need is understated | Model conditional dependence or resample shifts |
| Pooled regime hides seasonality | Errors recur every promotion or weekday | Distribution ignores a known driver | Wrong staffing on peak days | Condition on information available at decision time |
| Empirical tail treated as a bound | Simulations never exceed historical maximum | Resampling cannot generate unseen extremes | Rare-event risk is understated | Use justified tail model or stress scenario |
| Outlier removal hides real events | Largest stops vanish after cleaning | Difficult observations are deleted | Resilience looks better than reality | Investigate records and show sensitivity |
| Good fit mistaken for validation | Only one p-value is reported | Test does not check process or decision | False confidence | Combine diagnostics, holdout, and operational review |
| Reused holdout | Validation improves after many revisions | Model selection leaks into assessment | Performance is optimistic | Reserve fresh evidence or label the limit |
| Simulation interval mistaken for total uncertainty | Interval is tiny despite eight data points | Only replication error is measured | Decision appears more certain than evidence allows | Add parameter and structural uncertainty |
| Seed mistaken for paired design | Policies consume random draws differently | Corresponding events are no longer aligned | Variance reduction fails or misleads | Key streams to shared exogenous events |
| Policy changes inputs | Staff speed changes after overtime begins | Input assumed invariant under intervention | Policy comparison is biased | Model policy-dependent behavior or run a pilot |
| Timestamp zone error | Negative or repeated durations near clock change | Inconsistent daylight-saving conversion | Spurious durations and peaks | Store time zone and convert consistently |
| Workload approximation called a job count | Report has fractional customer orders | Minutes are divided by average duration | Service risk is miscommunicated | Label job-equivalents or simulate discrete completions |
| Drift ignored | Residuals change after new equipment launch | Historical process is no longer representative | Stale recommendations | Monitor, review, and version refits |

## 11. Assumptions and limits

The worked example is a decision illustration, not an estimate for a real site.
Its quantitative outputs are conditional on synthetic data and explicit simplifications.

| Assumption | Why it is needed here | What breaks it | How to test or respond |
|---|---|---|---|
| Eight shifts are comparable | Empirical demand draws use a common population | Promotions, holidays, or channel shifts | Compare covariates and collect representative shifts |
| Accepted jobs equal relevant demand | The example uses accepted counts | Stockouts, cancellations, or rejected requests | Reconcile requests and availability; bound the missing gap |
| Six service times represent the future mix | Independent empirical draws use a common distribution | Product mix or staffing changes | Stratify and compare future mix to observed mix |
| Service draws are independent | Simple calculation draws each job separately | Shared shift conditions or learning | Examine within-shift correlation; sample blocks |
| Recorded service is active work | The workload model separately adds failures | Waits or breakdowns are already included | Reconcile milestone definitions and sampled traces |
| Assets have common exponential first lifetimes | The rate formula uses a shared constant hazard | Ageing, different assets, or duty cycles | Compare survival and hazard patterns; fit alternatives |
| Censoring is noninformative | Standard likelihood uses censoring as exposure | Monitoring stops when failure is suspected | Investigate dropout reasons and sensitivity |
| First-lifetime rate transfers to recurrent failures | The toy shift requires a repeatable failure count | Repair changes effective age or failure intensity | Collect recurrent histories and model repair effects |
| Scheduled minutes approximate at-risk exposure | The example uses a 240-minute Poisson exposure | Downtime and idle periods reduce operating time | Simulate operating-state exposure explicitly |
| Repair takes 15 minutes | A fixed duration keeps the example interpretable | Parts wait and repair complexity vary | Collect full restoration durations and model their distribution |
| Failures do not occur during overtime | The example samples only base-shift exposure | Equipment also runs during overtime | Extend the exposure clock and compare results |
| All work is present at shift start | Total workload approximates completion | Late arrivals or cutoffs prevent work from starting | Use arrival timestamps and a discrete-event model |
| Backlog scales at 8 minutes per job | Remaining minutes are converted to job-equivalents | Actual last jobs have different durations | Simulate discrete job completion and priorities |
| USD 18 per equivalent is incremental | Cost ranks the alternatives | Penalties, lost sales, or recovery costs differ | Finance validates cost components and ranges |
| Overtime reserve costs USD 55 regardless of use | The policy is a reserved capacity option | Pay depends on actual time or minimum-call rules | Encode the actual labor agreement and decision timing |
| No safety or contractual constraint binds | Cost alone is compared | A minimum service or safe staffing rule applies | Treat requirements as constraints before ranking costs |

A narrow simulation interval does not cure an unsupported assumption.
Changing an assumption can require a different model, not merely another parameter value.
If a decision depends on unobserved tail behavior, state that the evidence is insufficient for a precise probability.
A conservative stress case can show exposure without pretending to know its frequency.

## 12. Exercises

### Exercise 1 — Observation versus demand

A distributor observes 100 shipped orders on a day with six hours of stockout.
It also has a log of 12 rejected requests, but does not log customers who leave without requesting.
The manager proposes using 100 orders as the full demand observation.

Deliverable: a short classification of observed, partially observed, and unobserved quantities.
Acceptance criteria: distinguish shipments from requested demand; identify at least one unidentified component;
name two additional evidence sources and avoid asserting an exact total.
Time: 10 minutes.
Maps to outcome 1.

### Exercise 2 — Calculate fitted summaries

Use demand counts 22, 25, 19, 28, 24, 27, 21, and 26 jobs per shift.
Use first failures at 100, 120, and 80 operating minutes and one right-censored lifetime at 140 minutes.
Assume independent exponential first lifetimes and noninformative censoring.

Deliverable: mean demand, sample variance, total exposure, estimated failure rate, and mean lifetime.
Acceptance criteria: show formulas, units, and why the censored exposure is included without adding a failure.
Time: 15 minutes.
Maps to outcome 2.

### Exercise 3 — Critique a duration extract

A report contains completed jobs only.
It measures assignment-to-completion elapsed time and calls it active processing time.
Long-running open jobs are excluded, and overnight breaks are included in closed jobs.

Deliverable: a table of at least three distinct problems and practical remedies.
Acceptance criteria: identify censoring and waiting separately; discuss likely bias without claiming a universal net direction.
Time: 15 minutes.
Maps to outcome 3.

### Exercise 4 — Model joint stress

A site has 52 weeks of demand and service records.
High-demand weeks tend to have more complex orders and longer processing times.
Management plans to sample demand and service independently from pooled histories.

Deliverable: one proposed joint model, one diagnostic, and one limitation.
Acceptance criteria: preserve a meaningful dependence unit and explain the effect of losing joint stress;
state what the proposed model cannot learn from the record.
Time: 15 minutes.
Maps to outcome 3.

### Exercise 5 — Design a nested experiment

A simulation of next week's policy uses a fitted daily demand rate and a fitted failure rate.
Both estimates are based on limited historical data.
The model owner has enough compute for many runs but not enough evidence for a precise tail model.

Deliverable: outer and inner loop pseudocode, an uncertainty label for each loop, and a reporting plan.
Acceptance criteria: refit or sample inputs in the outer loop; simulate future variability in the inner loop;
distinguish Monte Carlo error, parameter uncertainty, and unquantified structural uncertainty.
Time: 20 minutes.
Maps to outcome 4.

### Exercise 6 — Choose a demand model

A constant Poisson model matches mean demand but misses the upper tail on promotion days.
An empirical model matches observed historical values but contains only two promotion days.
A calendar-conditional model fits better in training but has unstable promotion estimates.

Deliverable: a provisional choice, at least two diagnostics, and a decision-sensitive sensitivity plan.
Acceptance criteria: discuss sample limitations, holdout data, operational plausibility, and the cost of tail errors;
do not choose solely on a p-value.
Time: 20 minutes.
Maps to outcome 5.

### Exercise 7 — Compare one simulated draw

A shift has 230 minutes of service work and 3 failures, each taking 15 minutes.
The base window is 240 minutes; reserved overtime adds 60 minutes at USD 55.
Remaining work is converted at 8 minutes per job-equivalent and costs USD 18 per equivalent.

Deliverable: base and overtime costs and the paired saving on this draw.
Acceptance criteria: show minutes, job-equivalents, currency, and why the single-draw result is not an expected value.
Time: 10 minutes.
Maps to outcome 6.

### Exercise 8 — Set up model governance

A live staffing model will be refreshed monthly and consumes demand, service, and maintenance data.
Management wants an automatic recommendation for the next shift.
The initial historical period includes a recent equipment replacement and a software migration.

Deliverable: a one-page release checklist and monitoring plan.
Acceptance criteria: assign owners; specify data quality, regime, validation, versioning, override,
and escalation triggers; separate safety rules from optimization.
Time: 25 minutes.
Maps to outcome 7.

## 13. Self-check answers

### Answer 1

The 100 shipped orders are observed shipments, assuming the report reconciles.
The 12 rejected requests are observed unsatisfied requests if they are distinct from the shipped orders.
Customers who never submit a request are unobserved.
At least 112 distinct requests may be evidenced under those assumptions, but exact latent demand is not identified.
Check for duplicates, partial shipments, and whether rejections later became accepted orders before adding counts.
Potential evidence includes availability logs, abandoned-cart or inquiry data, substitution records,
customer interviews, and comparable periods without stockouts.
None guarantees recovery of all unobserved demand.

### Answer 2

Demand totals 192 jobs across eight shifts, for a mean of 24 jobs per shift.
Squared deviations total 68 jobs squared.
The sample variance is $68/7=9.714$ jobs squared per shift squared.
Exposure is $100+120+80+140=440$ operating minutes.
The failure count is 3; the censored asset contributed exposure but no observed failure.
Thus $\hat\lambda=3/440=0.006818$ failures per operating minute.
The corresponding exponential mean is $1/\hat\lambda=146.667$ operating minutes.
This is a point estimate, not a promise that a particular asset will last 146.667 minutes.

### Answer 3

Excluding open jobs selects completed observations and may understate long completion durations.
Retain open-job age and a completion indicator, then use a censoring-aware method if assumptions are defensible.
Assignment-to-completion time contains queueing and other waiting, not necessarily active service.
Capture actual start, pause, and completion events or model elapsed time consistently without adding the same wait twice.
Overnight breaks can inflate elapsed duration relative to active work.
Use calendar and operator-state data to separate break time where possible.
The two biases can operate in opposite directions, so the net direction is not determined from these facts alone.

### Answer 4

A defensible baseline is to resample entire weeks, preserving each week's joint demand and observed service mix.
Compare simulated and historical service summaries conditional on high-demand weeks.
This retains observed within-week association but cannot generate novel joint extremes or a future regime absent from history.
If next year's order mix differs, condition on known drivers or add explicit scenarios.
A well-supported conditional model is another acceptable answer.

### Answer 5

One acceptable design draws or resamples historical units in an outer loop,
refits the demand and failure parameters, and stores that fitted model.
For each fitted model, an inner loop simulates many future weeks and calculates the policy difference.
The inner variation represents future randomness conditional on the selected inputs;
finite inner replication also creates Monte Carlo estimation error.
Across outer fits, variation reflects sampling uncertainty to the extent the resampling design is valid.
Alternative model families, unrecorded demand, and process changes need separate structural scenarios.
A confidence interval for the simulation mean under one fixed fit excludes those additional uncertainties.
For dependent daily data, use suitable blocks rather than assuming each day is independent.
For censored failures, retain the event indicator and exposure in the refitting procedure.

### Answer 6

There is no uniquely justified winner from the stated evidence.
A strong answer recognizes that a constant Poisson model misses a decision-relevant promotion tail.
It uses the conditional model as a candidate while acknowledging the unstable promotion estimate.
It compares out-of-time promotion and ordinary periods where available,
checks high quantiles and the probability of exceeding capacity,
and examines whether inputs available at decision time explain the modeled differences.
It compares the recommended policy across empirical, conditional, and plausible stress alternatives.
If only two promotion days exist, a precise tail probability is not supported.
The decision owner may choose a reversible policy or commission additional data collection.

### Answer 7

Required work is $230+3\times15=275$ minutes.
The base plan leaves 35 minutes, equal to $35/8=4.375$ job-equivalents.
The base cost is $4.375\times\text{USD }18=\text{USD }78.75$.
The overtime window is 300 minutes, so there is no remaining work in this draw.
Overtime costs USD 55 and saves USD 23.75 relative to base on this draw.
A different draw can favor base, so calculate an average paired difference before deciding on expected cost.
The approximation produces fractional job-equivalents rather than actual customer order counts.

### Answer 8 rubric

A strong plan names an accountable operations owner, a data owner, and a model owner.
It tests source completeness, timestamp integrity, exposure, and definitions before each refresh.
It separates pre- and post-replacement and pre- and post-migration regimes.
It reserves comparable out-of-time evidence and tests tail and joint behavior.
It versions data, code, parameters, approval, and recommendation.
It defines drift thresholds, investigation ownership, and a safe manual fallback.
It prevents a cost recommendation from overriding safety or contractual constraints.
A weak plan says only to retrain monthly and monitor average accuracy.

## 14. Interview questions

1. How do you model demand when only fulfilled sales are recorded?
   A strong answer distinguishes observed sales from latent demand and explains availability and lost-sales evidence.
   A weak answer assumes shipments equal unconstrained demand.
2. How do you select a service-time distribution?
   A strong answer defines the service boundary, checks support and covariates, and validates decision-relevant tails.
   A weak answer chooses a familiar family because software offers it.
3. What does a right-censored asset contribute to a failure estimate?
   A strong answer includes its event-free exposure and no failure count under the stated likelihood.
   A weak answer either drops the asset or records a failure at the cutoff.
4. When is failures divided by exposure a useful estimate?
   A strong answer states the process, exposure clock, censoring, and constant-rate assumptions.
   A weak answer divides any work-order count by calendar days without examining risk time.
5. How can independent input sampling distort a staffing decision?
   A strong answer explains shared drivers and the loss of simultaneous demand and service stress.
   A weak answer says matching all marginal means is sufficient.
6. What is the difference between a simulation confidence interval and input uncertainty?
   A strong answer distinguishes replication error from limited knowledge of distributions and structure.
   A weak answer claims more simulation runs eliminate sparse historical data.
7. Why might a goodness-of-fit test be insufficient?
   A strong answer discusses test power, estimated parameters, tail impact, and independent validation.
   A weak answer treats a p-value above a threshold as proof of truth.
8. How do you choose the unit to bootstrap?
   A strong answer follows the dependence and observation structure, such as shift or time block.
   A weak answer always resamples individual records without examining shared conditions.
9. How do you compare overtime policies efficiently?
   A strong answer uses aligned paired inputs, incremental costs, and uncertainty for the paired difference.
   A weak answer compares two unrelated point estimates or assumes the same seed always aligns events.
10. What would cause you to reject a model release?
    A strong answer names material data gaps, invalid exposure, decision-sensitive mismatch, or missing control ownership.
    A weak answer releases because the model produces a precise numerical answer.

## 15. Key terms

Each term links to the definitions in [Section 4](#4-definitions-and-vocabulary).

- [Arrival process](#4-definitions-and-vocabulary) — timing and count of incoming work.
- [Aleatory uncertainty](#4-definitions-and-vocabulary) — modeled future variation.
- [Bootstrap](#4-definitions-and-vocabulary) — resampling for sampling uncertainty.
- [Block bootstrap](#4-definitions-and-vocabulary) — resampling dependent sequences in groups.
- [Calibration](#4-definitions-and-vocabulary) — estimating model inputs.
- [Censoring](#4-definitions-and-vocabulary) — observing a bound on an event time.
- [Common random numbers](#4-definitions-and-vocabulary) — coupling policy comparisons.
- [Count](#4-definitions-and-vocabulary) — integer events in a bucket.
- [Demand](#4-definitions-and-vocabulary) — customer need before constraints.
- [Dependence](#4-definitions-and-vocabulary) — joint behavior beyond marginal distributions.
- [Empirical distribution](#4-definitions-and-vocabulary) — distribution based on observed values.
- [Epistemic uncertainty](#4-definitions-and-vocabulary) — limited knowledge of inputs or structure.
- [Exponential distribution](#4-definitions-and-vocabulary) — constant-hazard lifetime family.
- [Exposure](#4-definitions-and-vocabulary) — time or cycles at risk.
- [Failure](#4-definitions-and-vocabulary) — defined loss or degradation of function.
- [Hazard](#4-definitions-and-vocabulary) — instantaneous rate among surviving units.
- [Holdout](#4-definitions-and-vocabulary) — data reserved for assessment.
- [Input uncertainty](#4-definitions-and-vocabulary) — uncertainty about supplied input models.
- [Interarrival time](#4-definitions-and-vocabulary) — elapsed time between arrivals.
- [Lead time](#4-definitions-and-vocabulary) — elapsed time between milestones.
- [Left truncation](#4-definitions-and-vocabulary) — selection by surviving to entry.
- [Monte Carlo error](#4-definitions-and-vocabulary) — error from finite simulation runs.
- [Nonstationarity](#4-definitions-and-vocabulary) — distributional change over time.
- [Poisson distribution](#4-definitions-and-vocabulary) — a count distribution with equal mean and variance.
- [Quantile](#4-definitions-and-vocabulary) — threshold associated with a cumulative probability.
- [Regime](#4-definitions-and-vocabulary) — operating state with distinct behavior.
- [Replication](#4-definitions-and-vocabulary) — one independent simulation run.
- [Right censoring](#4-definitions-and-vocabulary) — lifetime known only to exceed an observed time.
- [Scenario](#4-definitions-and-vocabulary) — coherent set of assumptions.
- [Service time](#4-definitions-and-vocabulary) — defined resource time per activity.
- [Survival function](#4-definitions-and-vocabulary) — probability of exceeding a lifetime.
- [Tail](#4-definitions-and-vocabulary) — low-probability region of outcomes.
- [Validation](#4-definitions-and-vocabulary) — evidence of adequacy for a specific use.
- [Verification](#4-definitions-and-vocabulary) — implementation conformance to specification.
- [Weibull distribution](#4-definitions-and-vocabulary) — lifetime family with flexible hazard shape.

## 16. Sources

Sources below support methods and terminology; none supplies the fictional example's parameters.
Web resources and standards should be checked for current editions before operational use.

1. NIST/SEMATECH, *e-Handbook of Statistical Methods*, Reliability chapter.
   [Reliability](https://www.itl.nist.gov/div898/handbook/apr/apr.htm).
   Covers survival functions, censoring, life distributions, and reliability estimation.
2. NIST/SEMATECH, *e-Handbook of Statistical Methods*, exponential model.
   [Exponential distribution in reliability](https://www.itl.nist.gov/div898/handbook/apr/section1/apr161.htm).
   Provides the constant-hazard model and its reliability relationships.
3. Barton, Russell R., and Lee W. Schruben (1993), "Uniform and Bootstrap Resampling of Empirical Distributions," *Proceedings of the 1993 Winter Simulation Conference*.
   [Penn State publication record](https://pure.psu.edu/en/publications/uniform-and-bootstrap-resampling-of-empirical-distributions).
   Discusses resampling empirical inputs in simulation; applicability depends on the data structure.
4. Sargent, Robert G. (2013), "Verification and Validation of Simulation Models," *Journal of Simulation*, 7, 12–24.
   [DOI: 10.1057/jos.2012.20](https://doi.org/10.1057/jos.2012.20).
   Distinguishes verification and validation and discusses model-use context.
5. GS1, *EPCIS Standard*.
   [EPCIS](https://www.gs1.org/standards/epcis).
   Defines a visibility-event data sharing standard; it does not guarantee event accuracy.
6. International Society of Automation, *ISA-95 Standard: Enterprise-Control System Integration*.
   [ISA-95 overview](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard).
   Provides enterprise and control integration concepts.
7. International Organization for Standardization, *ISO 22400-2:2014*, Automation systems and integration — Key performance indicators for manufacturing operations management — Part 2.
   [ISO catalogue](https://www.iso.org/standard/54497.html).
   Addresses manufacturing operations KPI definitions; check applicability and current status.
8. International Organization for Standardization, *ISO 14224:2016*, Petroleum, petrochemical and natural gas industries — Collection and exchange of reliability and maintenance data for equipment.
   [ISO catalogue](https://www.iso.org/standard/64076.html).
   Provides a reliability and maintenance data reference for its stated industry scope.
9. NASA, *NASA Systems Engineering Handbook*, NASA/SP-2016-6105 Rev. 2.
   [NASA handbook](https://soma.larc.nasa.gov/STP/tdmo/pdf_files/NASA_Systems_Engineering_Handbook_SP-2016-6105_Rev_2.pdf).
   Engineering reference for system lifecycle and risk practices; it does not supply the example's failure rate.

## 17. Links

- [Module 20 index](./_index.md).
- [Twin versus shadow versus simulation](./01_twin-shadow-simulation.md).
- [Discrete-event simulation](./02_discrete-event-simulation.md).
- [Module 20 survey and embedded lab material](../../../course/20_digital_twins_simulation_optimization_uncertainty.md).
- [Module 30 simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md).
- [Probabilistic forecasts: related demand uncertainty topic](../08_hierarchical_intermittent_probabilistic_forecasting/07_probabilistic-forecasts.md).
- [Forecastability and benchmarks: related modeling topic](../07_time_series_forecasting/01_forecastability-benchmarks.md).
- [Installed-base failure forecast: related reliability topic](../../06_domain_electives/31_aftermarket_spare_parts_service/01_installed-base-failure-forecast.md).
