# Validating Supply-Chain Simulation and Digital Twins

> Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty  
> Topic 4 of 10  
> Estimated study time: 4 hours  
> Prerequisites: simulation concepts, discrete-event simulation, and input modelling  
> Version 2026.5

## 1. Why this topic exists

A distribution manager wants to add a second receiving shift before peak season.
A simulation reports that this change would remove most late receipts.
The investment has an annual labor cost, and the proposed roster changes employees' working hours.
An attractive animation is not evidence that the estimate is fit for that decision.

Validation asks whether the model is sufficiently credible for a specified use.
It does not prove that the model is universally true.
A model can be useful for comparing alternative shift patterns and unsuitable for promising individual delivery times.
The required evidence depends on the consequences of the decision, the alternatives and the time available.

A model owner must answer several business questions before presenting a result.
Does the representation preserve the process mechanisms that affect receiving congestion?
Do operators recognize the routing, exceptions, capacity constraints and calendar?
Do simulated results reproduce observations from periods not used to tune the model?
Are discrepancies small enough to avoid changing the recommended decision?
What uncertainty remains after the model has passed the agreed checks?

Validation is not a final chart added to a slide deck.
It starts when the decision, model boundary and data are defined.
It continues after deployment when the real process, input feeds or rules change.
For a live digital twin, a stale event feed or a changed shift calendar can invalidate a previously credible forecast.
A one-off simulation study needs controlled versions and documented limits too.

Validation differs from verification.
Verification asks whether the implementation behaves as its specification intends.
Validation asks whether the implemented representation is adequate for its intended real-world use.
Both matter: software can correctly implement an unrealistic queue, while a plausible conceptual model can be coded incorrectly.

The purpose of this lesson is to turn validation into a decision record.
You will compare face, structural and output evidence; separate calibration from independent testing; quantify bias and uncertainty; and specify who may rely on which result.
The example uses synthetic receiving data, not a claim about a real facility.
The example's numerical conclusions depend on its stated acceptance rule and observations.

## 2. Learning outcomes

After this lesson, you can:

1. Define a decision-specific validation question, including alternatives, horizon, units and consequence of error.
   Exercise 1 asks you to write the decision contract.
2. Distinguish verification from face, structural and output validation.
   Exercise 2 asks you to classify evidence.
3. Design calibration and known-period holdout windows without leaking future information.
   Exercise 3 asks you to choose a split and account for an operational change.
4. Compute signed bias, absolute error and an interval for a period-matched discrepancy.
   Exercise 4 checks the arithmetic and interpretation.
5. Decide whether validation evidence supports an investment, a limited pilot or a stop.
   Exercise 5 requires a conditional recommendation.
6. Identify input, simulation, measurement and structural uncertainty without treating them as interchangeable.
   Exercise 6 asks you to design a sensitivity and replication plan.
7. Specify a governance record, monitoring trigger and accountable response to drift.
   Exercise 7 asks you to draft a release gate.
8. Critique fairness and operational bias across relevant segments, not only overall performance.
   Exercise 8 tests subgroup reasoning and escalation.

Success means that you can explain both what the evidence supports and what it does not support.
A passing historical comparison does not automatically validate an unobserved policy.

## 3. The core mental model

```text
Decision and loss from error
          |
          v
Boundaries, intended use, pre-agreed tolerances
          |
          +--> Process evidence --> face and structural checks
          |
          +--> Input and code evidence --> data checks and verification
          |
          +--> Historical known periods --> calibration --> locked model
                                                    |
                                                    v
                                      independent holdout comparisons
                                                    |
                    +-------------------------------+------------------+
                    |                               |                  |
               KPI bias and error          subgroup/tail checks    uncertainty
                    |                               |                  |
                    +-------------------------------+------------------+
                                                    |
                                                    v
                                  decision impact and risk acceptance
                                                    |
                                                    v
                                      approve / pilot / reject
                                                    |
                                                    v
                                  monitor, investigate, revalidate
```

The arrows are evidence flow, not a promise of causality.
A high historical fit may arise from an incorrect model with compensating errors.
A model can also have modest historical fit but reliably rank alternatives within a narrowly defined decision range.
Neither situation is resolved by the aggregate error alone.

### A claim is conditional

Replace the sentence “the twin is validated” with a scoped claim.
For example: “Version 1.3 reproduces daily receiving throughput within the agreed bias threshold over eight held-out weekdays, under observed arrival and staffing conditions, for comparing similar shift patterns.”
Even this statement would require evidence for the comparable shift patterns, not merely matching throughput.
It is not a claim about a new facility, an unprecedented peak or employee-level decisions.

The conditions include model version, input vintage, population, time window, metric definition, decision horizon and intended action.
Record them alongside the evidence.
An approval that omits these conditions tends to be reused beyond its evidential boundary.

### Three complementary lenses

Face validation asks knowledgeable people whether the representation and behavior seem plausible.
A receiving supervisor might reject a model that unloads trucks at a dock that is closed during the night shift.
A plausible animation alone cannot establish whether the waiting-time distribution is accurate.

Structural validation asks whether the relationships and constraints correspond to the operational mechanism at the required resolution.
It examines routing, entity state, capacity, precedence, calendar, priority rules and mass balance.
An audit trail from specification to code and tests makes the finding reproducible.

Output validation compares defined model outputs with independent observations under matched conditions.
The comparison can include throughput, queues, service levels, cost and tail outcomes.
A single overall average is seldom sufficient for a decision involving peak congestion.

Each lens can catch a different defect.
Face review finds implausible exceptions that the data may not label.
Structural review finds a missing capacity constraint even if average throughput matches.
Output comparison reveals a persistent bias that the process description did not reveal.

### Separate three uncertainties

Random simulation uncertainty arises because one replication is one stochastic realization.
Use independent replications and report uncertainty in estimated simulation summaries.
Input uncertainty arises because historical records only estimate arrival, service and failure behavior.
Explore plausible fitted inputs, parameter intervals or an appropriate resampling procedure.
Structural uncertainty arises from competing reasonable model mechanisms, such as pooled versus dedicated labor.
Compare alternatives where practical and state where the data cannot discriminate.

Measurement uncertainty concerns how observations are recorded and matched.
For instance, a receipt scan may occur after physical unloading.
More simulation replications narrow Monte Carlo error but do not repair a biased scan timestamp or a wrong routing rule.

### The counterfactual boundary

History provides observations under the policy that actually operated.
The proposed second shift did not run in the held-out period, so its historical outcome is not directly observed.
Use structural evidence, policy-range checks and, where feasible, a controlled pilot to support that extrapolation.
If policies alter arrival behavior, overtime, labor productivity or carrier scheduling, encode and test those mechanisms rather than assuming inputs remain fixed.

## 4. Definitions and vocabulary

| Term | Meaning | Common misuse | Unit or convention |
|---|---|---|---|
| Intended use | Decision and operating domain supported by a model | Calling any model valid for all decisions | Decision, horizon and population |
| Verification | Check that the implementation conforms to its specification | Treating a successful run as validation | Test result and version |
| Validation | Evidence that a model is adequate for intended use | Claiming universal proof of truth | Scoped claim and acceptance rule |
| Face validation | Expert assessment of plausibility and behavior | Treating expert approval as an accuracy estimate | Reviewer, scenario and finding |
| Structural validation | Examination of causal logic and operational constraints | Comparing only one output mean | Rules, state transitions and invariants |
| Output validation | Comparison of simulated and observed outcomes | Comparing unmatched periods or definitions | Matched metric and time bucket |
| Calibration | Estimation or adjustment using designated data | Reporting calibration fit as independent accuracy | Training window and parameters |
| Holdout | Observations withheld from fitting and selection | Inspecting and repeatedly tuning to the holdout | Locked period and model version |
| Known period | Historical period with sufficiently reliable observations | Assuming all historical data are comparable | Local dates, calendar and exposure |
| Leakage | Use of unavailable or withheld information in model fitting or inputs | Calling realized future arrivals a prospective forecast | Information available at decision time |
| Historical replay | Run using recorded historical inputs and initial state | Presenting conditional replay as forecast accuracy | Recorded inputs, cutoff and period |
| Prospective test | Test using only information available before outcomes occur | Substituting actual future demand without disclosure | Forecast vintage and evaluation horizon |
| Metric | Defined operational quantity used for assessment | Comparing numerically similar but different definitions | Units, numerator, denominator and cutoff |
| Exposure | Opportunity for an event or outcome to occur | Comparing counts with different volumes | Arrivals, operating hours or orders |
| Throughput | Completed units crossing a defined boundary | Equating arrivals with completions | Receipts/day or units/hour |
| Waiting time | Elapsed time from defined queue entry to service start | Using scan delay as actual queue duration | Minutes and timestamp rule |
| Service level | Fraction satisfying an explicit promise | Averaging incompatible denominators | Percent of eligible orders on time |
| Bias | Mean signed model-minus-observed discrepancy under a defined weighting | Confusing signed bias with absolute error | Same units as the metric |
| Absolute error | Absolute magnitude of a discrepancy | Inferring direction from magnitude | Same units as the metric |
| Mean absolute error | Average absolute discrepancy | Treating it as signed bias | Metric units per time bucket |
| Relative error | Error divided by a stated reference | Dividing by zero or omitting denominator | Percent of observed value |
| Replication | One simulation run with a controlled random stream | Treating correlated days as independent replications | Run identifier and seed |
| Monte Carlo error | Variation in a simulation estimate due to finite random draws | Claiming it covers model misspecification | Standard error or interval |
| Confidence interval | Procedure-based interval for a specified estimand | Treating it as the probability that a fixed parameter lies inside | Level, method and assumptions |
| Prediction interval | Range for a future outcome under stated assumptions | Calling an interval for a mean a prediction interval | Future period or event |
| Paired difference | Difference between observations aligned by unit or period | Pairing unrelated days to reduce apparent variance | Model minus observed |
| Warm-up | Initial period discarded to reduce initialization effects in a suitable steady-state study | Discarding genuine transient peak conditions | Simulated time |
| Initial state | Queues, inventory and resources at simulation start | Starting empty when history begins with backlog | Time-stamped state |
| Regime | Operational conditions governed by a common process | Pooling pre- and post-policy-change data indiscriminately | Dates and conditions |
| Tail | Extreme portion of an outcome distribution | Validating the mean as a proxy for the 95th percentile | Quantile and sample size |
| Tolerance | Pre-agreed discrepancy or risk limit for intended use | Choosing the threshold after seeing results | Metric units or decision loss |
| Structural uncertainty | Uncertainty about the model's mechanisms or boundaries | Treating it as random-seed variation | Alternative structures and scenarios |
| Input uncertainty | Uncertainty in estimated model inputs or their future values | Treating fitted parameters as known facts | Distribution, parameter or scenario |
| Measurement uncertainty | Uncertainty in recorded reference data | Assuming scans are exact ground truth | Timestamp resolution or measurement error |
| Drift | Change in data, operation or model performance over time | Declaring a permanent failure from one noisy point | Metric, window and trigger |
| Challenger | Alternative method used as a comparator | Assuming a complex twin always beats a simple baseline | Same information and decision context |
| Approval | Accountable acceptance for a specified use | Treating a modeler's sign-off as unlimited authority | Scope, owner, expiration and conditions |
| Fairness assessment | Examination of relevant differences in error or impact across groups | Inferring equitable performance from the aggregate | Segment, denominator and consequence |

## 5. The mechanics, step by step

### 5.1 Frame the decision before testing

Write the business decision as a choice among feasible actions.
Include the status quo, at least one alternative and the time by which the decision must be made.
Name the person authorized to approve the action and the person accountable for model use.

Define the population and physical boundary.
A receiving model might cover inbound trailers from gate arrival until the receipt is posted.
Specify whether cross-dock loads, returns and emergency deliveries are inside or outside the boundary.
Use the same scope for the observed and simulated metrics.

List the outcomes that can change the decision.
Throughput may be the primary measure, but queue length, labor hours and late receipts may constrain the proposal.
A policy that increases throughput but violates a labor agreement is not a feasible recommendation.
Distinguish the forecast of demand, the operating plan, a service target and an actual commitment.

Record the loss from incorrect recommendations.
A false positive may buy unnecessary capacity; a false negative may cause missed deliveries and lost sales.
If the costs are asymmetric, tolerances need not be symmetric across scenarios or outcomes.
The business owner should approve tolerances before the holdout results are examined.

**Input:** decision brief, alternatives, operational constraints and cost consequences.
**Output:** a signed intended-use statement and a preliminary validation plan.
**Branch:** if decision ownership or the reference outcome cannot be defined, pause the investment claim and narrow the scope.

### 5.2 Build the evidence inventory

List the systems that record gate events, appointments, labor rosters, dock assignments, warehouse receipts and exceptions.
For every table, record its owner, extraction time, timestamp zone, business key and retention period.
Describe whether the source is a direct observation, a derived value, a manual entry or an imputation.

Match events by a stable shipment or trailer identifier where available.
Investigate duplicates, missing IDs, clock drift, daylight-saving transitions and records corrected after the event.
Do not silently delete awkward events simply because the simulation cannot represent them.
Maintain exclusions with counts, reasons and the effect on the denominator.

Define each metric in executable or auditable terms.
For daily completions, choose the physical completion event or receipt-posting event and the local-day boundary.
If the business uses a 06:00-to-06:00 operating day, do not compare it with calendar-day simulated totals.
Check whether overnight work belongs to the day started or the day completed.

Review whether the observation system itself changed.
A scanner replacement or a new definition of late receipt may cause a measurement break rather than a real process change.
Request operational context before interpreting a sudden distribution shift.

**Input:** source extracts, metadata, operating calendar and definitions.
**Output:** data-quality report and traceable observation table.
**Branch:** if critical timestamps or IDs are missing, use a coarser defensible metric or obtain better observations; do not invent exact waits.

#### An operational data contract

The team should document the input table before scoring the model.
The table below is a compact example for a receiving operation.
Names and tolerances must be adapted to the actual site.

| Field | Meaning | Check before use | Decision relevance |
|---|---|---|---|
| `trailer_id` | Identifier for a physical trailer visit | Uniqueness within a visit and documented reuse | Connects arrivals, work and departure |
| `gate_in_time` | Recorded time of entry into the yard | Local time, clock offset and missingness | Determines arrival and waiting exposure |
| `service_start_time` | Time unloading actually begins | Cannot precede gate entry without an explained exception | Separates waiting from service |
| `service_end_time` | Time unloading and required checks end | Cannot precede start under the defined clock | Defines completion and service duration |
| `dock_id` | Dock used for unloading | Valid dock at that time and documented reassignment | Tests resource capacity and routing |
| `shift_id` | Rostered shift covering the event | Correct cross-midnight attribution | Supports shift-specific checks |
| `exception_code` | Recorded reason for a deviation | Allowed values and missing-code rate | Distinguishes diversions and unusual work |
| `extract_time` | Time the source snapshot was taken | After corrections needed for the reference | Establishes data vintage |

A field may be present yet still unfit for its proposed use.
For example, a service-start scan can be consistently late if staff scan after opening a trailer.
That delay biases service duration downward and waiting time upward.
The combined elapsed time might look correct even though both component metrics are wrong.

Review a sample of timestamp sequences with operators.
Choose examples from each shift and from both normal and busy days.
Include records with missing scans, long elapsed times and overrides.
Sampling only clean records would conceal the exact cases that challenge the twin.

For each field, record its source system and whether its value is available at prediction time.
An end-of-day corrected roster may be a good reference for historical staffing.
It is not necessarily an available input for the morning staffing decision.
An actual gate arrival is appropriate in a conditional historical replay.
It is not the forecast that was available before that trailer arrived.

#### Reconciliation example

Suppose the gate system records 1,020 eligible arrivals over ten operating days.
The receipt system links 980 of those arrivals to a completion or an open backlog entry.
Forty arrivals have no linked disposition in the extract.
The raw link rate is 980 / 1,020 = 96.1%.

That rate does not tell you whether missing records are harmless.
If 35 of the 40 unmatched arrivals occurred during the busiest shift, the missingness is concentrated in a decision-critical segment.
The observed completion rate for that shift could be biased upward or downward depending on the missing outcomes.
Do not fill all 40 as completed or all 40 as late without evidence.

Ask the data owner to classify unmatched IDs.
Possible explanations include identifier reuse, manual diversion, a second visit and missing scans.
Record the number in each resolved category and the remaining unresolved count.
If resolution requires a later corrected extract, preserve both the original vintage and the corrected reference.

As a simple bound, suppose all 40 unmatched arrivals were eligible for a late-rate metric.
If 60 linked arrivals were late, at least 60 / 1,020 = 5.88% were late.
If every unmatched arrival were also late, the upper bound would be 100 / 1,020 = 9.80%.
These are extreme accounting bounds, not a confidence interval.
They assume the 980 linked outcomes are classified correctly and that all 1,020 arrivals belong in the denominator.
If a target is 7% late, uncertainty from missing classification alone can change the apparent pass or fail.

This bound is useful even when its extremes are implausible.
It shows whether resolving missing records can matter to the business decision.
If both bounds are far from the decision threshold, the missingness may be less consequential for that specific metric.
It may still matter for a subgroup, a tail metric or an individual claim.

#### Select checks for the intended analysis

An overall count can be complete while the sequence of events is wrong.
A trailer may be counted as completed on the correct day but linked to the wrong dock.
That defect could leave daily throughput unaffected and corrupt dock utilization.
Validate the fields needed for each claim rather than applying one generic completeness score.

For throughput, check the event that constitutes completion and the operating-day cutoff.
For wait, check both gate-in and service-start timestamps and their ordering.
For resource utilization, check dock identity, occupancy interval and overlaps.
For a late-service rate, check the promise time, eligible population and exclusion rules.
For cost, check the activity basis, currency, time bucket and whether costs are incremental.

When a field fails, decide whether to repair it, exclude the affected claim or broaden the uncertainty assessment.
The right response depends on the source of the problem and the impact on the decision.
State who approved the disposition and retain the original values.
Never overwrite an unfavorable measurement merely to make the model agree.

### 5.3 Partition periods and lock the evaluation

Select a calibration window that covers normal conditions relevant to the intended use.
Select a later holdout window that was not used to tune model parameters or choose among candidate models.
Record dates, calendar, data vintage and reasons for excluding any days.
For seasonal decisions, reserve relevant peak observations when available.

Avoid random row splits that place the same busy day, vehicle or shared disruption in both sets.
A time-based split better represents a future deployment when the model will be used prospectively.
If multiple locations are involved, consider an additional site holdout when transfer to new sites is part of the intended use.

Freeze the conceptual model, code, parameter estimates, metric definitions and acceptance thresholds before examining the holdout.
A documented holdout failure may motivate revision, but then that period has become development data.
Obtain a fresh untouched period or clearly label the later comparison as exploratory.
Repeatedly checking the same holdout while changing rules is adaptive tuning, even when each change appears small.

Identify operational regime changes across the split.
If the site changed from two docks to three, simulate each period with its actual capacity and calendar.
Do not hide a post-change failure by averaging it with a pre-change success.
If there are too few comparable periods, state that uncertainty and use a prospective shadow trial.

**Input:** time-indexed, quality-checked history and known changes.
**Output:** frozen split, versioned parameter set and locked test specification.
**Branch:** if the planned policy is unlike any observed condition, retain the holdout as baseline evidence only and require stronger extrapolation evidence.

### 5.4 Check face validity independently

Ask people with different operational perspectives to review process maps and representative trajectories.
Include planners, frontline operators, maintenance, data owners and the decision owner as appropriate.
Avoid relying only on the sponsor who benefits from the proposed investment.

Present typical and exception scenarios without revealing which answer the modeler wants.
Ask what happens when a carrier is late, a dock fails or a high-priority load arrives.
Capture contradictory accounts; a disagreement may reveal shifts, sites or undocumented local workarounds.

Review behavior at extreme but plausible inputs.
If all docks are unavailable, a model should not complete receipts unless an explicitly represented alternative resource is available.
If arrivals rise while capacity is fixed, a queue should not disappear without a mechanism such as diversion or abandonment.
Operators should review assumptions about breaks, batch sizes and handoffs.

Record reviewer role, date, model version, material shown, observations and unresolved issues.
Face agreement establishes plausibility, not statistical accuracy.
A dissenting review requires investigation; counting favorable votes is not a substitute for resolving a mechanism.

**Input:** process representation, traces, scenario outputs and expert availability.
**Output:** signed review record with corrections and unresolved limitations.
**Branch:** a material contradiction blocks claims that depend on that process until resolved or explicitly bounded.

### 5.5 Verify implementation and inspect structure

Trace each important process rule from source evidence to specification, implementation and test.
For a dock resource, verify capacity, shift schedules, failure behavior and resource release.
For each entity, check that the state transitions follow the physical sequence.
Where the model intentionally simplifies a step, explain why the simplification is unlikely to change the decision.

Use conservation checks.
For a closed receiving boundary, starting backlog plus arrivals must equal completions plus ending backlog plus explicitly recorded removals.
The same relation should hold per simulated replication and, within measurement limits, in the observation data.
A missing removal category can masquerade as a queue discrepancy.

Test deterministic edge cases before stochastic comparisons.
With one arrival, one available dock and a fixed 30-minute service time, the completion should occur 30 minutes after service starts.
With zero arrivals and zero starting backlog, completions should be zero.
With a resource unavailable for the entire horizon and no alternate resource, no work should complete through that resource.
These are verification tests, not validation against the facility.

Inspect event ordering when two events have the same timestamp.
A shift ending at 18:00 and a service starting at 18:00 need a defined ordering and overtime rule.
Use unit-consistent clocks and confirm that service times in minutes are not interpreted as hours.
Check priority preemption, batching and queue discipline against documented operations.

Compare alternative plausible structures if evidence is weak.
For example, one pooled team and two dedicated teams can yield the same average labor hours but different congestion.
If the recommended action changes across these structures, the structural ambiguity is decision material.
Obtain better evidence, design a pilot or report that the decision is not resolved.

**Input:** conceptual model, operational rules, code, tests and audit traces.
**Output:** traceability matrix, test results and documented simplifications.
**Branch:** an invariant failure blocks release until corrected and affected tests are repeated.

### 5.6 Define comparable historical experiments

Choose whether the comparison is a conditional replay or a prospective forecast test.
In a conditional replay, feed the observed arrival sequence and actual resource calendar into the model.
This isolates parts of the process response, but it does not test the ability to forecast arrivals or absences.
Label the result accordingly.

In a prospective test, use the demand forecast, roster and state snapshot available at the historical decision time.
Do not replace a forecast with later realized demand while claiming prospective accuracy.
If both tests are feasible, their difference helps locate error in inputs versus process response.

Initialize each test with the observed or best-estimated backlog, inventory and resource state at its cutoff.
If these states are uncertain, vary them over a plausible range and disclose the effect.
For an actual finite peak window, preserve the genuine starting backlog rather than automatically discarding it as warm-up.
For a steady-state study, justify a warm-up period and verify that it does not erase relevant transient behavior.

Align operating days, units, eligibility rules, sites and exclusions.
A real holiday closure must be reflected in the model calendar.
An observed receipt that belongs to an excluded process should not remain in the reference denominator.
Record the count of matched and unmatched observations and the reason for each mismatch.

Use enough replications to estimate stochastic output summaries with decision-relevant precision.
Predefine a precision target or replication budget.
If using sequential stopping, specify the rule in advance and use an interval method appropriate for that design.
Do not treat every simulated event within one run as an independent replication.

**Input:** locked model, historical inputs, reference outcomes and initialization.
**Output:** reproducible run manifest and matched comparison table.
**Branch:** if alignment or reference quality is insufficient, limit conclusions rather than scoring incomparable series.

### 5.7 Compare outputs at the right resolution

Start with the metrics linked to the decision and the business-approved tolerances.
Compute a signed discrepancy as simulation minus observation for each matched period.
Compute mean signed error and mean absolute error across periods.
State whether each day receives equal weight or whether volume weighting is used.

Inspect direction as well as magnitude.
An average bias near zero can conceal alternating large over- and under-predictions.
For example, errors of +20 and -20 receipts/day have zero mean bias but 20 receipts/day mean absolute error.
A model that meets an average-bias rule may therefore fail an absolute-error or tail rule.

Compare distributions and operational states, not only means.
A queue-length distribution, 90th-percentile wait, utilization by shift and maximum backlog may matter more than average throughput.
Plot observed and simulated outcomes by day and examine residuals by weekday, volume band, shift and facility.
A high aggregate match can conceal systematic underestimation during the busiest shift.

Use a simple challenger with the same information availability.
A recent comparable-day average or a capacity rule may be sufficient for some decisions.
If the simulation is not better or does not provide useful causal and scenario information, consider whether its maintenance cost is justified.
A challenger is not automatically a substitute for a model of untried policies.

Interpret statistical tests in context.
Failure to reject a zero-difference null hypothesis is not proof of equivalence.
A small sample can produce a wide interval even when a business-material bias is plausible.
If equivalence is required, define a business margin beforehand and use an appropriate equivalence procedure or interval rule.

**Input:** matched outputs, observations, segment definitions and precommitted margins.
**Output:** discrepancy table, distribution and segment diagnostics, and challenger comparison.
**Branch:** if aggregate results pass but a critical segment fails, evaluate the business exposure and do not announce an unconditional pass.

### 5.8 Quantify uncertainty without hiding its source

For stochastic simulation, retain replication-level outputs and seeds.
Estimate the simulation mean for each historical period and its Monte Carlo standard error from independent replications.
Increase replications if simulation noise prevents a meaningful comparison with the tolerance.
A narrow Monte Carlo interval is conditional on the chosen model and inputs.

Across historical periods, report the spread and uncertainty of matched discrepancies.
A simple paired t interval assumes that period-level discrepancies are approximately independent and that the interval is suitable for the sample distribution.
Serial correlation, shared disruptions and short time series can make that interval misleading.
Use a dependence-aware design or collect more independent periods when appropriate.

Explore input uncertainty by varying estimated arrival, service, failure and staffing parameters using defensible ranges or resampling.
When a fitted parameter and historical observation come from the same source, preserve their dependence in the resampling design.
Avoid varying inputs independently when the real process links them, such as long service times on high-volume days.

Explore structural uncertainty by testing alternative plausible mechanisms and checking whether they alter the policy ranking.
Review observation uncertainty, including delayed scans and missing exceptions.
Record which uncertainties are quantified, which are stress-tested and which remain unquantified.
Do not add intervals from different sources mechanically without a valid joint method.

**Input:** replications, historical periods, input estimates and competing structures.
**Output:** uncertainty ledger with methods, assumptions and impact on the decision.
**Branch:** if uncertainty crosses the decision boundary, favor a reversible pilot, more evidence or a robust policy rather than reporting a false point estimate.

### 5.9 Translate the evidence into a decision

Use a pre-agreed evidence gate, not a single favorable statistic.
A gate may require acceptable data coverage, no unresolved structural defect, output errors within tolerance and no material subgroup failure.
It may also require that reasonable input and structural alternatives do not reverse the recommendation.

Compare the predicted incremental value of each policy with the cost of being wrong.
A model can miss its throughput threshold yet still support a low-risk pilot if the pilot is reversible and measured carefully.
Conversely, a small average error may be unacceptable when it can reverse a large irreversible investment decision.

Classify the result as approved for a specified use, approved with conditions, deferred for more evidence, or rejected.
Specify any conditions: limited facilities, duration, human review, budget cap, uncertainty range or pilot measurement.
The owner, not the model alone, decides whether residual risk is acceptable.

**Input:** validation evidence, decision economics, risk appetite and feasible alternatives.
**Output:** signed decision memo with limitations and review date.
**Branch:** when evidence is inconclusive, state the unresolved decision rather than silently treating inconclusive as pass.

### 5.10 Monitor the deployed twin

Store the approved model artifact, code revision, parameter values, dependencies, input snapshot or immutable references, seeds and run configuration.
Record the model owner, business owner, independent reviewer and approver.
Separate the ability to change code, approve release and authorize operational decisions where practical.

Monitor input freshness, missing events, schema changes, state reconciliation and output residuals.
Define triggers before release, such as sustained out-of-tolerance bias or a major change to dock capacity.
An alert should name a recipient, response time, investigation path and fallback action.
Avoid automatically retraining or changing operating policy solely because a noisy metric crossed one arbitrary threshold.

Revalidate after material process changes, data-definition changes, deployment to a new facility or expansion of intended use.
Record whether the evidence covers the new situation.
If an operational feed fails, degrade to a safe manual or static process and label outputs stale.
A retired model should no longer appear as an approved decision tool.

**Input:** production telemetry, process-change notices and approved baseline.
**Output:** monitoring record, incident response and periodic review decision.
**Branch:** suspend the relevant use if a critical data feed or control is unreliable; restore only after the owner accepts corrective evidence.

## 6. Worked example

### Decision and scope

A fictional regional distribution center receives consumer goods on weekdays.
Its operations director must decide whether to add an evening receiving shift for a six-week peak.
The alternatives are the existing roster and a proposed shift that adds 14 labor-hours per operating day.
The proposed shift costs USD 4,200 per week including supervision and overhead.
This cost is an assumption for the example, not a market estimate.

The model's intended use is to compare feasible receiving rosters at this site.
The primary historical validation metric is completed receipts per operating day.
A receipt means a trailer whose unloading and receiving checks are completed within the local 06:00-to-06:00 operating day.
Starting and ending backlogs are counted at 06:00 local time.
Cross-dock trailers and customer returns are excluded from both model and reference data.

The demand forecast is a prediction of inbound arrivals.
The current and proposed rosters are plans.
The business target is at most 5% of eligible trailers waiting beyond one operating day.
A carrier appointment is a commitment only if the actual contract says so; the target itself is not a carrier commitment.

Management agrees, before the holdout is opened, that absolute mean signed throughput error must be at most 3 receipts/day.
It also requires mean absolute error at most 5 receipts/day, no unresolved conservation failure, and separate review of busy days.
These numbers are assumed business thresholds, not universal statistical standards.
A pass supports baseline credibility only; the proposed roster still needs counterfactual evidence.

### Data and split

The model team calibrates service-time and absence assumptions on an earlier eight-week period.
It locks the model and evaluates eight later, eligible operating days.
Each holdout day uses actual observed arrivals and the actual roster as inputs.
This is a conditional replay, not an evaluation of advance demand forecasts.

The initial backlog is taken from the physical gate and receipt records at each day's cutoff.
Days containing a documented system outage were excluded before scoring because their reference completions could not be reconstructed.
The count and reason for exclusions are recorded; the resulting evidence does not cover such outages.
The simulated value for each day is the mean of many independent runs, rounded here to a whole receipt/day for teaching.
The code below uses the displayed values and therefore reproduces the displayed calculations.

| Holdout day | Actual arrivals (trailers/day) | Observed completions (receipts/day) | Simulated mean (receipts/day) | Error: simulated minus observed (receipts/day) |
|---|---:|---:|---:|---:|
| 1 | 99 | 96 | 98 | +2 |
| 2 | 108 | 104 | 107 | +3 |
| 3 | 93 | 91 | 90 | -1 |
| 4 | 115 | 109 | 113 | +4 |
| 5 | 104 | 101 | 103 | +2 |
| 6 | 120 | 112 | 117 | +5 |
| 7 | 97 | 95 | 94 | -1 |
| 8 | 111 | 106 | 108 | +2 |

The observations sum to 814 receipts over the eight operating days.
The simulated means sum to 830 receipts over those same eight days.
The difference is +16 receipts across matched day-level means.
The mean signed error is 16 / 8 = +2.0 receipts/day.
The model therefore overestimates throughput on average under this sign convention.

The absolute errors sum to 2 + 3 + 1 + 4 + 2 + 5 + 1 + 2 = 20 receipts.
Mean absolute error is 20 / 8 = 2.5 receipts/day.
Aggregate relative bias is 16 / 814 × 100 = +1.97% of observed completions.
This volume-weighted ratio is not the same as averaging eight daily percentage errors.

For these errors, the sample standard deviation is approximately 2.14 receipts/day.
The standard error of their mean is approximately 2.14 / sqrt(8) = 0.76 receipts/day.
Using a two-sided 95% t critical value of 2.365 with seven degrees of freedom gives an interval of approximately [0.21, 3.79] receipts/day.
This interval describes uncertainty in a mean discrepancy under the stated independent-period approximation.
It is not an interval for a future day, a probability of passing the threshold, or an allowance for structural uncertainty.
Eight consecutive operating days could be correlated; if so, the simple interval may be too narrow.

The point estimates meet the assumed mean-bias and mean-absolute-error gates.
The upper interval endpoint exceeds the 3 receipts/day bias threshold.
If the gate required the entire interval to lie inside ±3 receipts/day, this evidence would not pass that stronger rule.
The stated gate concerns point estimates, so do not silently replace it with an interval rule after seeing the data.
Instead report the residual uncertainty and consider whether more independent days are needed before a consequential commitment.

### Segment and structural checks

Define busy days prospectively as at least 110 observed arrivals per operating day.
Days 4, 6 and 8 qualify, with errors +4, +5 and +2 receipts/day.
Their mean error is 11 / 3 = +3.67 receipts/day.
The remaining five days have errors totaling +5 receipts, or +1.0 receipt/day.
The busy-day estimate is based on only three days; it is a warning, not a stable population estimate.

The aggregate result would hide the larger bias in the conditions most relevant to peak staffing.
The team traces a plausible mechanism: the modeled labor calendar assumes relief coverage during a break, but actual records show relief was not always available.
This is an illustrative investigation finding, not an assertion about the data in the table.
The team must test the explanation with rosters and event traces rather than immediately changing a parameter to fit the holdout.
Any changed model needs fresh holdout evidence.

An inventory-flow check for a sample day starts with 12 trailers in backlog and adds 115 arrivals.
With 109 completed trailers, no diverted trailers and no missing records, ending backlog should be 12 + 115 - 109 = 18 trailers.
If the reference system reports 17 trailers, reconcile one trailer before treating the event ledger as exact.
Possible causes include an unrecorded diversion, a timing cutoff difference or a missing arrival.
The correct resolution requires evidence; adjusting the simulation to hide the discrepancy is not validation.

### Decision implications

Suppose the unmodified model estimates that the proposed shift avoids 45 late trailers per week relative to the baseline.
Suppose the business estimates an avoidable cost of USD 120 per late trailer, and assumes this value is incremental and not double-counted.
The estimated weekly gross benefit is 45 × USD 120 = USD 5,400.
After the assumed USD 4,200 weekly shift cost, estimated net benefit is USD 1,200 per week.
The break-even reduction is USD 4,200 / USD 120 = 35 late trailers per week.

The 45-trailer estimate is a hypothetical model output, not established by the throughput table.
The table alone neither validates late-trailer counts nor the causal effect of the extra shift.
If a reasonable structural or input scenario reduces the predicted improvement to 30 trailers/week, net benefit becomes USD 3,600 - USD 4,200 = -USD 600/week.
That scenario crosses break-even, so an irreversible commitment would be poorly supported without further evidence.

A defensible recommendation is a time-limited, approved pilot if labor and safety constraints allow it.
Predefine comparable peak days, actual cost accounting, late-trailer eligibility, a contemporaneous baseline or defensible comparison, and a stop rule.
A before-and-after comparison alone is vulnerable to changes in arrivals, carrier mix and staffing.
The director retains approval authority and should not interpret this example as a guaranteed saving.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the eight-day calculations and compares the point estimates to the illustrative gates.
The t critical value is fixed for this specific eight-day, two-sided 95% example.
For other sample sizes or interval designs, calculate the appropriate critical value with a vetted statistical library.
The script does not generate simulation replications or establish that period errors are independent.

```python
from math import sqrt
from statistics import mean, stdev

# Daily completions: model means and observed values in receipts/day.
rows = [
    # day, arrivals, observed completions, simulated mean
    (1, 99, 96, 98),
    (2, 108, 104, 107),
    (3, 93, 91, 90),
    (4, 115, 109, 113),
    (5, 104, 101, 103),
    (6, 120, 112, 117),
    (7, 97, 95, 94),
    (8, 111, 106, 108),
]

errors = [simulated - observed for _, _, observed, simulated in rows]
observed_total = sum(observed for _, _, observed, _ in rows)
simulated_total = sum(simulated for _, _, _, simulated in rows)
bias = mean(errors)
mae = mean(abs(error) for error in errors)
relative_bias = 100 * sum(errors) / observed_total

# Illustrative paired t interval across days, conditional on assumptions.
standard_error = stdev(errors) / sqrt(len(errors))
t_critical_df7 = 2.364624
interval = (
    bias - t_critical_df7 * standard_error,
    bias + t_critical_df7 * standard_error,
)

busy_errors = [
    simulated - observed
    for _, arrivals, observed, simulated in rows
    if arrivals >= 110
]
other_errors = [
    simulated - observed
    for _, arrivals, observed, simulated in rows
    if arrivals < 110
]

mean_bias_limit = 3.0  # receipts/day; illustrative pre-agreed gate
mae_limit = 5.0        # receipts/day; illustrative pre-agreed gate
point_gate = abs(bias) <= mean_bias_limit and mae <= mae_limit

print(f"Observed total: {observed_total} receipts")
print(f"Simulated total: {simulated_total} receipts")
print(f"Mean signed error: {bias:+.2f} receipts/day")
print(f"Mean absolute error: {mae:.2f} receipts/day")
print(f"Aggregate relative bias: {relative_bias:+.2f}%")
print(f"95% paired t interval: [{interval[0]:.2f}, {interval[1]:.2f}] receipts/day")
print(f"Busy-day mean error: {mean(busy_errors):+.2f} receipts/day (n={len(busy_errors)})")
print(f"Other-day mean error: {mean(other_errors):+.2f} receipts/day (n={len(other_errors)})")
print(f"Point-estimate throughput gate: {'PASS' if point_gate else 'FAIL'}")

weekly_shift_cost = 4200.0        # USD/week
cost_per_late_trailer = 120.0     # USD/trailer
avoided_late_trailers = 45.0      # trailers/week; hypothetical
break_even = weekly_shift_cost / cost_per_late_trailer
net_benefit = avoided_late_trailers * cost_per_late_trailer - weekly_shift_cost
print(f"Break-even reduction: {break_even:.1f} trailers/week")
print(f"Illustrative net benefit: USD {net_benefit:,.0f}/week")
```

Expected output, with rounding to two decimal places where shown:

```text
Observed total: 814 receipts
Simulated total: 830 receipts
Mean signed error: +2.00 receipts/day
Mean absolute error: 2.50 receipts/day
Aggregate relative bias: +1.97%
95% paired t interval: [0.21, 3.79] receipts/day
Busy-day mean error: +3.67 receipts/day (n=3)
Other-day mean error: +1.00 receipts/day (n=5)
Point-estimate throughput gate: PASS
Break-even reduction: 35.0 trailers/week
Illustrative net benefit: USD 1,200/week
```

The printed pass covers only two throughput point estimates.
It does not certify data quality, structural validity, segment performance, late-trailer prediction or the investment decision.
Production code should retain full-precision replication outputs and validate the source schema and metric definitions.
Do not round daily means before scoring when full-precision values are available.
A real gate should emit all required evidence and an accountable approval, not just a Boolean.

## 8. Data, standards, and regulatory context

### Data lineage and time

A validation dataset should be reconstructible from named sources and an extraction cutoff.
A warehouse management system, yard management system and labor roster can disagree about when work occurred.
Document the meaning of every timestamp, its time zone, precision, correction policy and business event.
An event posted after the fact is not necessarily the time at which the physical activity happened.

Record the data available at the historical decision cutoff separately from later corrections.
A retrospective replay may use corrected outcomes as reference observations, but a prospective input must not include corrections unavailable at the decision time.
If the replay uses actual arrivals, clearly label the comparison as conditional on realized arrivals.

Preserve the relationship between raw events and derived metrics.
For every reported late-trailer rate, retain its eligible population, exclusions, numerator, denominator, threshold and operating calendar.
Version the transformation logic as well as the simulation.
A changed metric definition can cause apparent drift even when operations have not changed.

### Identifiers and interoperability

GS1 publishes standards for identifying parties, locations, trade items and logistic units.
Its Electronic Product Code Information Services (EPCIS) standard provides a framework for sharing visibility event data.
These conventions can improve event interpretation and cross-organization traceability when implemented consistently.
They do not guarantee that a scan is complete, correctly timed or representative of all physical movements.
See the official GS1 standard linked in Sources for the current specification and implementation details.

A local warehouse may use internal trailer IDs rather than a GS1 identifier.
Document the mapping, reuse policy and unresolved collisions.
Do not assume that two systems' identical-looking IDs refer to the same physical object without a verified join rule.
If joins are probabilistic, quantify or review matching error and disclose its influence on validation.

### Model governance standards

ISO 9001 specifies requirements for a quality management system when an organization seeks conformance or certification.
It does not prescribe a universal simulation acceptance threshold or certify a particular twin's predictions.
The ISO 9001 standard and its organization-specific applicability should be checked against the current authoritative edition and scope.

The U.S. National Institute of Standards and Technology (NIST) AI Risk Management Framework is voluntary guidance for managing risks of AI systems.
Where a twin contains an AI component, its governance concepts can inform mapping, measurement, management and oversight.
The framework itself is not a general legal authorization for automated operational decisions and does not validate a simulation.
A non-AI simulation still benefits from explicit ownership, documentation and change control without being mislabeled as AI.

ASME V&V 40 addresses risk-informed credibility assessment for computational modeling in a medical-device context.
Its relationship between context of use, model risk and evidence is a useful analogy for supply-chain validation.
It is not a supply-chain certification requirement, and its medical-device context must not be presented as directly governing a receiving model.
The standards and guidance cited here are starting points, not substitutes for the rules applicable to a particular deployment.

### Privacy, employment and security

A staffing twin may ingest employee rosters, productivity records or location data.
Minimize personal data where aggregate or pseudonymized information is sufficient for the decision.
Limit access, document retention and audit sensitive uses with the organization's privacy, labor and security teams.
Pseudonymization does not by itself make a dataset anonymous.

Employment and privacy obligations depend on jurisdiction, purpose and facts.
For EU processing, the General Data Protection Regulation (GDPR) may be relevant; lawful basis, data minimization, access rights and automated-decision rules require context-specific review.
The GDPR is binding where applicable, unlike voluntary frameworks.
This teaching material is not legal advice and does not determine whether a particular process meets a legal exception or obligation.

Evaluate whether model errors or proposed rosters disproportionately affect shifts, sites, contractors or other relevant groups.
Aggregate error does not establish equitable impact.
Use groups only when appropriate and lawful, guard against identifying small populations, and involve the people affected in interpreting results.
A group comparison indicates a possible disparity; it does not alone establish causation or a legal violation.

Protect model inputs and operational outputs as potentially sensitive business information.
A compromised capacity feed or unauthorized parameter change can make an otherwise sound twin misleading.
Use access controls, change logs, protected releases and incident response proportionate to the operational risk.

## 9. Industry practice and case evidence

### Evidence is not a transferable accuracy claim

Public reports often describe a digital twin's purpose or claimed benefit without publishing the holdout data, metric definitions, uncertainty or comparison design.
A named implementation can demonstrate an operational use case without establishing that a different site's simulation is valid.
For each external claim ask who measured the outcome, what baseline was used, what changed at the same time and whether independent evidence is available.

### Port of Rotterdam

IBM describes work with the Port of Rotterdam on digital information and a port digital twin, including sensor and operational data intended to support port operations.
This is a named public case of the operational ambition to represent a complex physical system using shared data.
The cited IBM material is a provider account; it is not independent proof of generalizable simulation accuracy or a quantified return at another port.

**Inference:** a port model with vessel movement, water and berth information would require validation at multiple time and spatial scales.
A plausible aggregate port throughput would not by itself validate an individual berth occupancy prediction.
The public case alone does not establish a particular historical holdout protocol or the accuracy of a specific model version.
A practitioner should request those details before relying on a comparable deployment.

### BMW production planning

NVIDIA publicly describes BMW Group's use of digital factory tools and simulation for production planning.
The case is relevant because production layouts, material flow and human work are operationally coupled.
The cited provider account describes a project and capabilities; it should not be read as independent validation of every resulting decision or a guarantee of financial benefit.

**Inference:** a facility planning study should compare simulated flow and resource use with observed operation, review the representation with engineers and operators, and test alternative layouts under consistent assumptions.
An attractive virtual factory view is not itself a validation result.
For a new facility there may be no site-specific operating history, so evidence must come from analogous sites, component tests, uncertainty scenarios and later commissioning data.
Those sources are weaker for some claims and should be identified as such.

### Published simulation methodology

Sargent's paper on verification and validation of simulation models describes the distinction between conceptual model validity, computerized model verification, operational validity and data validity.
It emphasizes that validity concerns the model's intended application and requires a combination of techniques.
This methodology supports using expert review, structural checks and historical comparisons together rather than relying on a single test.
The paper does not supply an automatic universal error tolerance for distribution-center staffing.

Law's discussion of verification and validation similarly treats credibility as a practical modeling process involving the problem, data, model and users.
A published method is a guide to study design, not evidence that a particular local model passed its gate.
Record the actual local inputs, test results and sign-offs.

### How to read external claims

A vendor demonstration may be useful evidence of available software features.
It is weaker evidence about a model's real-world predictive accuracy when reference data and evaluation design are unavailable.
A company case study can reveal the decision context but may omit unsuccessful deployments or selection criteria.
A peer-reviewed method can establish a defensible evaluation technique without validating a new deployment.

In a procurement review, request an example validation report rather than only an animation or sales metric.
Ask for definitions, data lineage, a genuinely withheld period, segment errors, uncertainty, known failures and the proposed maintenance responsibility.
Distinguish supplier claims, independently audited outcomes and the buyer's own acceptance testing.
A contract should specify deliverables and acceptance rights, subject to legal and procurement review.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Calibration leakage | Test dates appear in parameter notebooks | Holdout influences tuning | Reported accuracy is optimistic | Freeze artifacts and retain a fresh time-based holdout |
| Output-only validation | One attractive aggregate chart | Compensating errors mask wrong rules | Policy change fails despite good baseline fit | Trace structure and review operators' exceptions |
| Face review only | Stakeholders say the animation looks right | Plausibility is mistaken for accuracy | Unquantified decision error | Add matched quantitative checks and data review |
| Wrong timestamp | Posting time substituted for physical completion | System delay shifts measured service time | False queue or throughput diagnosis | Map event semantics and quantify delay |
| Wrong day boundary | Night-shift totals disagree across systems | Calendar and operating days differ | Artificial daily residuals | Align cutoff, timezone and daylight saving |
| Empty initialization | Simulated queue starts at zero every day | Existing backlog is omitted | Early throughput and wait estimates are biased | Reconstruct initial state or test its uncertainty |
| Unmodeled exceptions | Manual diversions absent from model | Conservation boundary is incomplete | Backlog and tail errors | Explicitly represent or exclude with an auditable rule |
| Misused random seeds | One run drives the recommendation | Monte Carlo noise looks like policy effect | Unstable business case | Use independent replications and precision targets |
| Pseudoreplication | Thousands of events treated as independent runs | Shared system state induces dependence | Intervals are too narrow | Analyze independent runs or properly clustered units |
| Mean-only scoring | Average bias near zero despite large daily errors | Positive and negative errors cancel | Peak failures are concealed | Report absolute errors, tails and conditional slices |
| Segment masking | Overall pass but busiest shift fails | Weighted aggregate hides localized bias | Wrong capacity allocated where needed | Predefine business-critical segments and exposure |
| Unfair impact | One employee group receives worse schedules | Policy and errors vary across affected groups | Disproportionate harm and loss of trust | Assess relevant lawful group impacts and involve stakeholders |
| Unchecked counterfactual | New shift benefit inferred from baseline fit alone | Unobserved behavior changes under intervention | Exaggerated investment return | Review mechanisms, stress assumptions and pilot |
| Invalid uncertainty interval | Daily residuals are autocorrelated | Independence assumption fails | False confidence in a narrow interval | Use a dependence-aware design or more independent periods |
| Misread non-significance | Wide interval crosses zero | Low power is treated as equivalence | Material bias incorrectly dismissed | Set equivalence margin and interpret interval width |
| Unstable denominator | Late rate falls while eligible orders disappear | Exclusions or missing records change | Apparent improvement is not operational | Audit numerator, denominator and exclusions |
| Regime pooling | Capacity changed midway through history | Different processes are averaged | Current behavior is misrepresented | Tag regimes and validate each relevant condition |
| Uncertain reference treated as truth | Scan gaps cluster on busy days | Missingness is related to workload | Apparent model bias is confounded | Assess missingness and measurement error |
| Excessive tuning | Same holdout reviewed after every revision | Test becomes development data | Overfitting and false assurance | Preserve new holdouts or run prospective shadow tests |
| Overbroad approval | One site's pass copied to all sites | Different process and population | Unsupported deployment | Specify approved domain and require transfer validation |
| Stale twin state | Feed lag exceeds model horizon | Simulated state no longer matches reality | Misleading real-time recommendations | Monitor freshness, reconcile and fall back safely |
| Silent model change | Parameter update lacks version record | Evidence no longer matches artifact | Irreproducible recommendation | Version, approve and revalidate material changes |
| Cost double counting | Avoided late cost includes labor already saved elsewhere | Benefits overlap | Inflated business case | Map incremental cost categories and finance ownership |
| Incentive conflict | Sponsor alone chooses thresholds and signs off | Pressure favors favorable interpretation | Selective reporting | Independent review and precommitted gates |

Treat a warning as a prompt for investigation, not automatic proof of a root cause.
Record the check, evidence and owner before closing it.
Some defects require stopping the model; others can be addressed by narrowing the claim.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test or bound it |
|---|---|---|---|
| The business metric matches the decision | Error must be meaningful to the choice | Throughput rises while late receipts or cost worsen | Review metric hierarchy and decision consequences |
| Source events represent physical work | Observations serve as a reference | Delayed scans or missing manual work | Compare with direct observations and audit trails |
| Initial backlog is known | Replay needs the correct starting state | Open trailers are not reconciled at cutoff | Reconcile snapshots and vary uncertain backlog |
| Holdout was not used for tuning | Independent evidence reduces selection bias | Analysts repeatedly inspect and revise against it | Audit access and preserve a fresh period |
| Periods are comparable | Observed-versus-simulated difference has a meaningful interpretation | Changed docks, products, policy or calendar | Tag regimes and replay actual conditions |
| Arrivals and rosters are correctly aligned | Conditional replay isolates process response | Time zone, cutoff or ID mismatch | Check event-level joins and operating-day totals |
| Expert descriptions reflect actual practice | Face and structure review need operational knowledge | Informal workarounds differ by shift | Observe work and seek independent perspectives |
| Simulation rules are implemented correctly | Output discrepancies must not be coding mistakes | Event ordering, units or resource release defects | Use invariants, deterministic cases and code review |
| Replications are independent | Standard simulation error calculations require a suitable sampling design | Shared streams or identical seeds are treated as independent | Inspect seed allocation and replication design |
| Period errors are sufficiently independent for a simple t interval | Standard error uses the effective number of independent periods | Consecutive days share backlog, demand or disruptions | Inspect dependence and use an appropriate alternative |
| Fitted input ranges cover plausible operation | Sensitivity results depend on explored cases | New peak demand or equipment failure outside history | Stress beyond history and update with new data |
| Policy changes do not alter omitted behavior | Counterfactual results rely on represented responses | Carriers reschedule or productivity shifts under new roster | Interview stakeholders and test in a pilot |
| Cost estimates are incremental | Benefit-cost comparison needs consistent accounting | Avoided costs are noncash, duplicated or delayed | Finance review and scenario analysis |
| Segment definitions are relevant and lawful | Disaggregated checks must address real risk | Tiny cells, proxies or unavailable consent | Privacy review, suppression and stakeholder input |
| Baseline comparison transfers to alternative policies | Model mechanisms must remain valid after intervention | New resource or priority rule creates unseen dynamics | Structural review, component tests and controlled deployment |
| Data and model remain current | Historical evidence applies to the deployed version | Process, schema, software or resource capacity changes | Version controls, monitoring and revalidation triggers |

The eight-day example does not identify a true long-run bias with high precision.
Its daily simulation means are supplied rather than generated from replication-level data.
The hypothetical policy benefit is illustrative and has not been validated against a realized counterfactual.
A production decision needs more evidence than this teaching dataset provides.

## 12. Exercises

### Exercise 1 — Write a decision contract

**Time:** 15 minutes.
A fulfillment center considers adding a weekend picking shift before a four-week promotion.
The sponsor asks whether a twin is “accurate enough.”

**Deliverable:** a one-page intended-use statement with two feasible policies, horizon, population, primary metric, constraints, owner and one decision-specific error consequence.
**Acceptance criteria:** distinguish a forecast, plan, target and commitment; give units and denominator; state one condition outside the approved use.

### Exercise 2 — Classify evidence

**Time:** 12 minutes.
Consider four proposed checks: an operator reviews a trace; a unit test checks resource release; the team matches queue distributions to a withheld month; an engineer traces dock capacity rules to the actual roster.

**Deliverable:** classify each check and state what it cannot establish on its own.
**Acceptance criteria:** distinguish face validation, verification, output validation and structural validation without claiming that any one check proves universal validity.

### Exercise 3 — Protect the holdout

**Time:** 20 minutes.
A site has 18 months of daily history, a new dock added in month 13, and a planned policy decision for the current three-dock operation.
A modeler wants to tune repeatedly on months 16–18 and report those months as independent validation.

**Deliverable:** propose a split, a regime treatment and a plan if three-dock history is insufficient.
**Acceptance criteria:** keep tuning separate from final testing, prevent future leakage, explain the capacity change and identify at least one limitation of the resulting evidence.

### Exercise 4 — Compute matched error

**Time:** 15 minutes.
Observed daily completions are 80, 100, 120 and 100 receipts/day.
Simulated daily means are 84, 96, 126 and 98 receipts/day, respectively.
Use simulation minus observation.

**Deliverable:** calculate daily errors, mean signed error, mean absolute error and aggregate relative bias.
**Acceptance criteria:** show arithmetic, units, denominator and one reason a zero or small mean bias can mislead.

### Exercise 5 — Make an investment decision

**Time:** 20 minutes.
A proposed change costs USD 6,000/week.
Avoided late orders are valued at USD 150/order.
The simulation estimates 50 avoided late orders/week, but plausible structural scenarios range from 30 to 55 orders/week.
Assume all benefits are incremental.

**Deliverable:** calculate break-even and the net benefit at the estimate and endpoints; recommend approve, pilot or defer.
**Acceptance criteria:** state the uncertainty that crosses the decision boundary and specify a measurement condition or reason for the chosen action.

### Exercise 6 — Design an uncertainty study

**Time:** 25 minutes.
A queue simulation uses a fitted service distribution, one staffing rule and 20 runs.
The recorded service timestamps are sometimes delayed, and the busiest day has never been observed with the proposed shift.

**Deliverable:** an uncertainty ledger with Monte Carlo, input, measurement and structural entries, plus a proposed investigation for each.
**Acceptance criteria:** distinguish what more seeds can and cannot fix; include at least one policy-ranking sensitivity and one limitation that remains.

### Exercise 7 — Define a release gate

**Time:** 20 minutes.
The twin runs every morning and recommends staffing to a supervisor.
The arrival feed sometimes arrives several hours late and the dock calendar changes quarterly.

**Deliverable:** a release and monitoring record covering owner, intended use, version, approval, freshness trigger, response, fallback and revalidation event.
**Acceptance criteria:** specify who acts and when, avoid automatic approval after an alert, and link the trigger to the decision horizon.

### Exercise 8 — Investigate segment bias

**Time:** 20 minutes.
The overall signed error is +1 receipt/day across 20 days.
The day shift averages -2 receipts/day across 15 days and the night shift averages +10 receipts/day across five days.
The staffing decision chiefly affects night-shift contractors.

**Deliverable:** verify the overall arithmetic, identify a possible business and people impact, and propose an investigation.
**Acceptance criteria:** report both segment sample sizes and units, avoid claiming causation or legal violation, and specify privacy-aware involvement of affected workers.

## 13. Self-check answers

### Exercise 1

A strong statement names the center, four-week promotion, eligible order population and a time-bounded comparison of current staffing against a feasible weekend shift.
It identifies a business owner and a separate model owner.
A plausible primary metric is late eligible orders divided by eligible orders, expressed as a percentage per operating week.
Constraint examples include labor availability, safety and service agreements.

The forecast is predicted order volume; the plan is the proposed shift roster.
The target is a desired late-order rate, and a commitment is an actual contractual or operational promise if one exists.
A material error might cause a costly unnecessary shift or leave orders unserved.
Evidence from this center and promotion does not automatically support another facility or a longer peak.

### Exercise 2

The operator's trace review is face validation: it checks operational plausibility but not numerical accuracy.
The resource-release unit test is verification: it checks implementation against a specified rule, not whether that rule is appropriate.
The withheld-month queue comparison is output validation: it checks matched behavior in the observed domain but cannot alone establish causal structure.
The dock-rule trace is structural validation: it checks the representation of a mechanism but does not alone establish output accuracy.
The checks are complementary and each should be versioned.

### Exercise 3

One defensible design uses a portion of the post-month-13 history for calibration and preserves a later post-change block as a locked test.
The exact split depends on when the decision was made and how much relevant post-change data are available.
Earlier two-dock data can support component estimates only if the differences are represented and their transferability is justified.
Do not apply future observations to a historical prospective forecast.

If months 16–18 are repeatedly used for tuning, they are not an untouched final holdout.
Reserve a later period, collect a prospective shadow test or state that independent evidence is unavailable.
Check whether seasonality and promotional peaks are represented; a time split alone cannot create missing peak evidence.
Document the reduced precision from a short post-change sample.

### Exercise 4

The errors are +4, -4, +6 and -2 receipts/day.
Their sum is +4 receipts, so mean signed error is +1 receipt/day.
The absolute errors sum to 4 + 4 + 6 + 2 = 16 receipts.
Mean absolute error is 16 / 4 = 4 receipts/day.
Observed completions total 400 receipts and aggregate relative bias is 4 / 400 × 100 = +1%.
A small mean signed error hides substantial daily errors that cancel.

### Exercise 5

Break-even is USD 6,000/week divided by USD 150/order, or 40 avoided late orders/week.
At 50 orders/week, net benefit is 50 × USD 150 - USD 6,000 = USD 1,500/week.
At 30 orders/week, it is USD 4,500 - USD 6,000 = -USD 1,500/week.
At 55 orders/week, it is USD 8,250 - USD 6,000 = USD 2,250/week.

The plausible range crosses break-even, so the point estimate alone does not settle the decision.
A reversible pilot with a credible comparison, predeclared outcome definition and measured incremental costs is defensible if safe and feasible.
Deferral is also defensible if a pilot cannot be run safely or its outcomes cannot resolve the uncertainty.
The range is not automatically a probability distribution or confidence interval.

### Exercise 6

Monte Carlo uncertainty concerns finite runs; use independent replications and assess the precision of policy differences.
Input uncertainty concerns fitted service behavior; refit or vary plausible parameters while preserving relevant dependence.
Measurement uncertainty concerns delayed timestamps; compare scans with direct observations and test plausible delay corrections.
Structural uncertainty concerns the staffing rule and the unobserved shift; compare credible rules and seek process evidence.

More random seeds reduce simulation noise conditional on the same model and inputs.
They cannot correct systematic timestamp delay or prove the proposed shift behaves as assumed.
Test whether reasonable variants reverse the policy ranking.
The actual counterfactual remains unobserved until a suitable intervention or stronger evidence is available.

### Exercise 7

A strong gate names the staffing supervisor as business decision owner, a model owner for technical response and an approver independent of the change where feasible.
It specifies the approved site, morning decision horizon, input vintage, code and parameter version, validation report and expiration date.
For example, if the feed is older than the maximum age approved for that morning's roster, the system alerts the named operator before the roster cutoff.
The exact maximum age and response time must be set from operational needs rather than assumed here.

The fallback could be a clearly labeled last-approved manual staffing procedure, not a stale recommendation presented as current.
A dock-calendar change triggers review and, when material, revalidation before using the changed configuration.
The owner investigates the alert and records the decision; crossing back under a threshold does not automatically approve an unreviewed model change.

### Exercise 8

The weighted overall error is (15 × -2 + 5 × 10) / 20 = +1 receipt/day.
The five night-shift observations show +10 receipts/day average overestimation, compared with -2 receipts/day for 15 day-shift observations.
The small night sample limits certainty, but the segment is directly relevant to the staffing choice.
Overestimated night throughput could lead to understaffing and increased workload; that is a possible consequence, not an established outcome.

Check metric definitions, shift-specific arrival mix, roster coverage, missing scans and operational exceptions.
Seek additional comparable night observations and involve contractors and their representatives as appropriate.
Protect small-cell data and use lawful access controls.
Do not conclude that the observed disparity proves a discriminatory mechanism or a legal breach.

## 14. Interview questions

### 1. What does it mean to validate a simulation?

A strong answer defines an intended use and combines evidence about data, process structure, implementation and observed outputs.
It describes conditional confidence rather than proof.
A weak answer says the model is valid because it runs or because a stakeholder likes the animation.

### 2. How do verification and validation differ?

A strong answer distinguishes correct implementation of the specified model from adequacy of that model for the real decision.
It supplies a concrete test for each.
A weak answer treats unit tests or a historical fit as sufficient for both questions.

### 3. Why can a model match average throughput and still be wrong?

A strong answer mentions compensating structural errors, cancellation of signed residuals and poor tails or segment fit.
It proposes mechanism checks and disaggregated comparisons.
A weak answer assumes one aggregate mean validates all policies.

### 4. How would you select a historical holdout?

A strong answer uses an untouched time-relevant period, identifies regime and seasonal coverage, locks artifacts and prevents information leakage.
It explains what to do after tuning against the holdout.
A weak answer randomly splits correlated events and repeatedly adjusts the model until the test passes.

### 5. Is replaying actual demand a forecast test?

A strong answer calls it a conditional replay and explains which process mechanisms it tests.
It contrasts that test with one using demand forecasts available at the decision cutoff.
A weak answer reports replay accuracy as end-to-end forecast accuracy.

### 6. What is the difference between bias and mean absolute error?

A strong answer defines signed model-minus-observed error and explains cancellation.
It reports absolute error separately, with units and weighting.
A weak answer calls both metrics accuracy without specifying direction or denominator.

### 7. What does a confidence interval that crosses zero tell you?

A strong answer discusses the interval's estimand, sampling assumptions and compatibility with material positive or negative effects.
It explains why non-rejection is not equivalence.
A weak answer declares the model unbiased with certainty.

### 8. How would you validate a policy never used historically?

A strong answer uses baseline evidence, mechanism review, plausible input and structural scenarios and, if appropriate, a controlled pilot.
It states the counterfactual limitation and potential behavior changes.
A weak answer assumes baseline fit automatically proves the alternative policy's benefit.

### 9. What does increasing replications solve?

A strong answer says it reduces Monte Carlo uncertainty in estimates under the same model and input assumptions.
It distinguishes this from input, measurement and structural uncertainty.
A weak answer says more random seeds fix a missing process or biased source data.

### 10. What happens after a live twin passes initial validation?

A strong answer names version control, feed freshness, residual monitoring, change triggers, accountable response and a safe fallback.
It limits approval by use and environment.
A weak answer assumes one sign-off lasts indefinitely.

### 11. How do you evaluate uneven error across shifts or groups?

A strong answer checks exposure, sample size, definitions, potential impact and privacy-aware stakeholder evidence.
It distinguishes a disparity from its cause.
A weak answer ignores a critical segment because the overall average passes.

### 12. When should you not use a twin for an investment decision?

A strong answer considers unobservable reference data, unresolved structural defects, decision-sensitive uncertainty, poor comparators and maintenance cost.
It suggests a narrower analysis, additional measurement or a reversible pilot where feasible.
A weak answer treats complexity or a vendor label as proof of business value.

## 15. Key terms

- **[Acceptance tolerance](#5-the-mechanics-step-by-step):** pre-agreed discrepancy or risk limit tied to the decision.
- **[Bias](#6-worked-example):** mean signed discrepancy under a stated comparison and weighting.
- **[Calibration](#5-the-mechanics-step-by-step):** fitting or adjusting a model using designated development evidence.
- **[Challenger](#5-the-mechanics-step-by-step):** alternative approach evaluated with comparable information.
- **[Conditional replay](#5-the-mechanics-step-by-step):** historical test conditioned on recorded inputs and state.
- **[Confidence interval](#6-worked-example):** sampling-procedure interval for a specified quantity under explicit assumptions.
- **[Conservation check](#5-the-mechanics-step-by-step):** accounting identity for entities or material crossing the boundary.
- **[Counterfactual](#3-the-core-mental-model):** outcome under an alternative not observed for the same situation.
- **[Data lineage](#8-data-standards-and-regulatory-context):** trace from original records through transformations to reported metrics.
- **[Decision boundary](#6-worked-example):** point at which the preferred action changes.
- **[Drift](#5-the-mechanics-step-by-step):** change in inputs, operations or performance over time.
- **[Equivalence](#5-the-mechanics-step-by-step):** sufficiently small difference according to a predefined margin and suitable evidence.
- **[Exposure](#4-definitions-and-vocabulary):** opportunity for the measured outcome to occur.
- **[Face validation](#5-the-mechanics-step-by-step):** operational plausibility review by knowledgeable people.
- **[Fairness assessment](#8-data-standards-and-regulatory-context):** examination of relevant differences in model error or decision impact across groups.
- **[Holdout](#5-the-mechanics-step-by-step):** data withheld from fitting and selection for evaluation.
- **[Initial state](#5-the-mechanics-step-by-step):** modeled queue, resources and other system conditions at the start.
- **[Input uncertainty](#3-the-core-mental-model):** uncertainty about model inputs or their estimated distributions.
- **[Intended use](#3-the-core-mental-model):** decision and operating conditions for which evidence is claimed.
- **[Leakage](#5-the-mechanics-step-by-step):** use of information unavailable or reserved for the intended test.
- **[Mean absolute error](#6-worked-example):** average magnitude of matched discrepancies.
- **[Measurement uncertainty](#3-the-core-mental-model):** uncertainty in the reference observations or their interpretation.
- **[Monte Carlo error](#3-the-core-mental-model):** finite-random-sampling uncertainty in a simulation estimate.
- **[Output validation](#3-the-core-mental-model):** comparison of model outcomes with suitable observations.
- **[Paired difference](#6-worked-example):** model-minus-observed difference for a matched unit or period.
- **[Prospective test](#5-the-mechanics-step-by-step):** evaluation using information available before outcomes occur.
- **[Regime](#5-the-mechanics-step-by-step):** set of operating conditions with a particular process configuration.
- **[Replication](#5-the-mechanics-step-by-step):** one run of a stochastic simulation under specified inputs and random streams.
- **[Structural uncertainty](#3-the-core-mental-model):** uncertainty about mechanisms, boundaries or relationships in the model.
- **[Structural validation](#5-the-mechanics-step-by-step):** review of whether model relationships reflect relevant operating mechanisms.
- **[Validation](#3-the-core-mental-model):** assessment of adequacy for a specified real-world use.
- **[Verification](#5-the-mechanics-step-by-step):** check that implementation conforms to its specification.

## 16. Sources

1. Sargent, Robert G. “Verification and validation of simulation models.” *Journal of Simulation* 7, 12–24 (2013). [https://doi.org/10.1057/jos.2012.20](https://doi.org/10.1057/jos.2012.20). Methodological reference on simulation credibility and validation techniques.
2. Law, Averill M. “How to Build Valid and Credible Simulation Models.” *Proceedings of the 2019 Winter Simulation Conference*. [Winter Simulation Conference archive](https://informs-sim.org/wsc19papers/). Consult the proceedings for the paper and its context.
3. National Institute of Standards and Technology. *Artificial Intelligence Risk Management Framework (AI RMF 1.0)*, NIST AI 100-1 (2023). [https://doi.org/10.6028/NIST.AI.100-1](https://doi.org/10.6028/NIST.AI.100-1). Voluntary AI risk-management guidance, not a simulation certification.
4. ASME. *Assessing Credibility of Computational Modeling through Verification and Validation: Application to Medical Devices*, V&V 40. [ASME standards listing](https://www.asme.org/codes-standards/find-codes-standards/assessing-credibility-computational-modeling-through-verification-validation-application-medical-devices). Medical-device context; used here as an explicitly qualified analogy.
5. GS1. *EPCIS Standard*. [https://www.gs1.org/standards/epcis](https://www.gs1.org/standards/epcis). Visibility event standard and current official materials.
6. ISO. *ISO 9001 — Quality management systems*. [https://www.iso.org/standard/62085.html](https://www.iso.org/standard/62085.html). Consult ISO for the current applicable edition and scope.
7. European Union. Regulation (EU) 2016/679, General Data Protection Regulation. [EUR-Lex consolidated text](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Binding where applicable; applicability requires context-specific assessment.
8. IBM. “Port of Rotterdam” customer story. [https://www.ibm.com/case-studies/port-of-rotterdam](https://www.ibm.com/case-studies/port-of-rotterdam). Provider account of port digitalization; not independent validation evidence.
9. NVIDIA. “BMW Group Builds Virtual Factory with NVIDIA Omniverse.” [NVIDIA Blog](https://blogs.nvidia.com/blog/bmw-group-omniverse/). Provider account of digital factory planning; do not generalize its claims to another site.
10. NIST/SEMATECH. *e-Handbook of Statistical Methods*. [https://www.itl.nist.gov/div898/handbook/](https://www.itl.nist.gov/div898/handbook/). Reference for statistical methods and assumptions; choose an interval appropriate to the data and design.

The worked case, data, tolerance, economics and recommendations are invented for instruction.
They are not observations or performance claims from the named organizations.

## 17. Links

- [Module 20 topic map](_index.md)
- [Twin versus shadow versus simulation](01_twin-shadow-simulation.md)
- [Discrete-event simulation](02_discrete-event-simulation.md)
- [Input modelling](03_input-modeling.md)
- [Experiment design](05_experiment-design.md)
- [Simulation-optimization](09_simulation-optimization.md)
- [Module 20 survey](../../../course/20_digital_twins_simulation_optimization_uncertainty.md)
- [Simulation-optimization production labs](../../../course/30_simulation_optimization_production_labs.md)
