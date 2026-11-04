# Statistical Process Control, Capability, and Defect Analytics

> Module 15 — Manufacturing and Production Planning  
> Topic 11 of 11  
> Estimated study time: 4 hours  
> Prerequisites: basic descriptive statistics and [OEE and the loss tree](09_oee-loss-tree.md)  
> Version 2026.5

## 1. Why this topic exists

A production planner needs to know how many conforming units a line can reliably deliver.
A quality engineer needs to know whether a change in measurements signals a new cause.
A customer needs assurance that product meets its specifications.
These are related questions, but they are not interchangeable.

Suppose a filling line repeatedly produces containers close to a maximum fill specification.
The average output may be on schedule while rework, extra inspection, and customer complaints increase.
Expediting more input material will not remove a biased fill setting.
Conversely, adjusting the filler after every random fluctuation can increase variation and make delivery less reliable.

Statistical process control (SPC) provides a disciplined way to distinguish unusually behaving processes from ordinary variation.
A control chart compares ordered process observations with limits estimated from process behavior.
Capability compares process location and spread with specification limits set by product requirements.
A defect analysis describes observed nonconformities and their business consequences.
Each is useful; none replaces the others.

The decisions include whether to stop and investigate, whether to release a lot under an approved plan, whether to adjust a setpoint, and whether a promised production quantity is credible.
The process owner also decides which evidence is sufficient to change a baseline.
A quality manager may temporarily contain material while an assignable cause is investigated.
A planner may revise expected good output and available capacity until the process is restored.
A customer or regulatory agreement may prescribe additional actions beyond this chapter.

This topic treats control charts as operational decision rules, not as proof that every item is good.
It also treats capability indices as estimates conditional on a defensible process model, not as permanent machine ratings.
The examples are fictional and use metric units and US dollars (USD).

## 2. Learning outcomes

By the end of this topic you can:

1. Distinguish customer specifications, process control limits, and operating targets in a containment decision.
   Exercise 1 tests the distinction.
2. Choose an appropriate variable or attribute chart and justify a rational subgroup.
   Exercises 2 and 6 test the choice.
3. Calculate an individuals and moving-range chart and identify an out-of-control signal.
   Exercise 3 tests calculation and response.
4. Calculate Cp and Cpk and explain why stability and an appropriate distribution matter.
   Exercises 4 and 5 test interpretation.
5. Calculate defect and defective-unit rates with correct denominators.
   Exercise 7 tests the distinction.
6. Design a sampling, reaction, and handoff plan that protects delivery without treating all fluctuation as special cause.
   Exercise 8 assesses the complete design.
7. Translate quality loss into expected good output and explicitly distinguish forecast, plan, target, and commitment.
   Exercises 7 and 8 assess the business link.

## 3. The core mental model

```text
Product need -> approved specifications -> acceptance decision
                         ^
                         |
Process -> measurement -> time order -> control chart -> investigate signal
   |                           |
   |                           +-> stable baseline -> capability estimate
   |                                                    |
   +-> defects, scrap, rework -> good output -> capacity and delivery plan
```

A process has inputs, operating settings, people, equipment, and environmental conditions.
Together these generate a distribution of measurements and defects over time.
The measurement system observes that process imperfectly.
The chart uses the order of observations to look for behavior inconsistent with the chosen baseline.

Common-cause variation is the routine variation of a stable system under its current conditions.
Special-cause variation is associated with a distinguishable change in the process or measurement system.
A chart signal is evidence for investigation, not proof of the specific cause.
A lack of signal is not proof that no change occurred.

Control limits describe expected process behavior under an estimated model.
Specification limits describe acceptable product characteristics.
They answer different questions and may have completely different numerical values.
A stable process can consistently miss a customer requirement.
A process may temporarily produce acceptable units while unstable and therefore hard to predict.

When a process is stable and the capability assumptions are reasonable, compare the estimated process distribution with the specification window.
Cp measures potential fit if the process were centered, using a within-process standard deviation.
Cpk accounts for the distance from the mean to the closer specification limit.
Both depend on the characteristic, time period, measurement method, and chosen estimate of variation.

A defect is a nonconformity; a defective unit has at least one nonconformity.
One unit may contain several defects.
A rate of defects per unit can therefore exceed one, while the proportion of defective units cannot.
Use the actual observation denominator and an explicit definition of an opportunity.

Finally translate quality evidence into a planning assumption.
Good output depends on the order and definitions of losses, including whether rework becomes saleable output within the planning horizon.
Do not multiply unrelated or overlapping loss percentages as though they were independent.
The planning number is an estimate, not a guarantee of conformance or delivery.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Characteristic | Measurable product or process property, such as fill mass in grams | Combining different characteristics in one index |
| Unit | One inspected product item | Treating a defect as a unit |
| Defect or nonconformity | One failure to meet a specified requirement | Assuming one defect per unit |
| Defective unit | Unit with one or more nonconformities | Calling every defect a separate unit |
| Specification | Authorized acceptance requirement, with units and revision | Estimating it from recent measurements |
| LSL | Lower specification limit in characteristic units | Calling it a lower control limit |
| USL | Upper specification limit in characteristic units | Using it as a chart limit |
| Target | Preferred characteristic value, not necessarily a specification boundary | Assuming it equals the midpoint |
| Control limit | Statistical boundary for a chart under its baseline model | Treating it as a release criterion |
| Center line | Chart's baseline estimate of process center | Assuming it is a customer requirement |
| UCL | Upper control limit in chart-statistic units | Calling every value below it conforming |
| LCL | Lower control limit in chart-statistic units | Allowing an impossible negative count without interpretation |
| Common cause | Routine system variation under a stable regime | A reason to ignore chronic poor performance |
| Special cause | Distinguishable source of unusual variation | Any measurement not equal to target |
| Signal | Pattern meeting a preselected chart rule | Proof of a specific root cause |
| Rational subgroup | Observations grouped to capture short-term variation under similar conditions | Arbitrary monthly averages hiding shifts |
| I chart | Chart of individual measurements over time | Substituting it for a grouped chart without considering sampling |
| Moving range | Absolute difference between consecutive individual observations | Difference from the specification target |
| X-bar chart | Chart of subgroup means | Chart of every individual reading |
| R chart | Chart of within-subgroup ranges | Chart of process mean |
| p chart | Chart of defective fraction with a binomial model | Chart of defects per unit |
| np chart | Chart of number of defective units with fixed sample size | Using one limit with changing sample sizes |
| c chart | Chart of defect count for a constant inspection opportunity | Counting defective units as defects |
| u chart | Chart of defects per unit or exposure when exposure varies | Ignoring variation in opportunity definition |
| Phase I | Retrospective baseline development and investigation | Quietly deleting inconvenient points |
| Phase II | Prospective monitoring against a fixed approved baseline | Recomputing limits after every new point |
| Sigma | Standard deviation estimate in characteristic units | A universal constant for all time horizons |
| Within sigma | Estimate of short-term variation, conditional on sampling design | A guarantee of long-term performance |
| Overall sigma | Standard deviation across the stated observations | Automatically interchangeable with within sigma |
| Cp | Specification width divided by six within standard deviations | A centeredness or yield measure by itself |
| Cpk | Smaller distance from mean to a specification limit divided by three within standard deviations | A meaningful prediction when process is unstable |
| Pp and Ppk | Analogous indices using overall standard deviation | Silent synonyms for Cp and Cpk |
| DPU | Defects per inspected unit | Proportion of defective units |
| DPMO | Defects per million defined opportunities | Observed parts per million defective |
| Measurement system | Instruments, people, procedure, calibration, and environment | The gauge alone |
| Reaction plan | Assigned actions for a defined signal | An instruction to adjust every observation |
| Containment | Controlled identification and disposition of potentially affected product | Permanent corrective action |

## 5. The mechanics, step by step

### Step 1 — Define the decision and the characteristic

**Input:** product drawing or specification, process flow, and the decision owner's question.
**Action:** identify the characteristic, unit, measurement location, product revision, and acceptance rules.
**Output:** one documented chart purpose and a traceable specification record.

For example, a fill-mass requirement can be two-sided in grams.
A cosmetic inspection can instead classify an item as defective or conforming.
A surface may have multiple scratches, each counted as a separate defect.
The sampling plan must distinguish these cases before data collection starts.

Ask whether the decision is process monitoring, lot acceptance, release, root-cause investigation, or capacity planning.
The same measurement may inform several decisions, but each can have a different sampling plan and authority.
A chart signal does not itself determine product disposition.
A conforming inspection result does not by itself establish process stability.

Record whether the specification is single-sided or two-sided.
A one-sided requirement has only one relevant capability distance.
Do not invent a second specification merely to calculate Cp.
Record whether the target lies at the specification midpoint.
The midpoint is a mathematical convenience, not necessarily the economically or technically preferred setting.

### Step 2 — Verify the measurement system

**Input:** measurement procedure, gauge, reference standards, operators, and environmental conditions.
**Action:** confirm calibration and assess measurement variation relative to the process and specification.
**Output:** a measurement-system assessment and agreed resolution and handling rules.

Calibration establishes traceability under stated conditions; it does not alone establish repeatability and reproducibility in production.
Gauge repeatability and reproducibility studies compare variation from equipment and appraisers using a suitable study design.
Attribute inspections need an assessment of consistency as well as a written defect catalogue.
Blinded repeat inspections can reveal disagreements in borderline visual classifications.

A gauge with coarse resolution can produce many identical readings and a misleadingly small moving range.
Rounding, data-entry truncation, automatic replacement of missing readings, and changes in instruments also alter chart behavior.
Record instrument identifiers and software versions with each observation when practical.
Do not treat instrument changes as process improvements without checking measurement effects.

If the measurement system is inadequate, improve it or explicitly restrict the interpretation of the analysis.
Do not apply a numerical gauge acceptance threshold without understanding the relevant industry method and risk.
A safety-critical dimension may require a stricter system than a low-risk internal indicator.

### Step 3 — Design sampling and rational subgroups

**Input:** process cycle time, changeover pattern, known variation sources, and the cost of delayed detection.
**Action:** select sampling frequency, subgroup size, selection method, and a way to preserve time order.
**Output:** a sampling instruction and a chart design matched to the question.

Consecutive units can represent short-term variation at a particular time.
Measurements spaced across a shift may deliberately include additional within-shift variation.
The distinction affects the within-sigma estimate and thus capability.
Averages across a full day can conceal a transient failure between shifts.

Select units independently of their apparent quality where possible.
Convenience selection of attractive units biases both defect estimates and capability.
Record sample time, line, product, cavity, shift, and relevant maintenance events.
When streams are materially different, chart them separately or model the stream differences explicitly.

Sampling every hour can miss a brief failure between samples.
More frequent sampling can shorten detection delay but uses labor and inspection capacity.
Base the interval on the hazard, process dynamics, containment cost, and response capacity.
If the process can drift rapidly, the last good sample may be a long way from the last good unit.

Consider serial correlation: adjacent readings may be correlated because the process has inertia or an automatic controller.
Standard individuals-chart limits based on adjacent moving ranges may be inappropriate in that situation.
Consult a statistician about modeling, residual charts, or a redesigned sampling interval.
Do not assume that spacing observations further apart automatically removes all correlation.

### Step 4 — Choose the chart family

**Input:** measurement scale, subgroup design, and defect definition.
**Action:** choose a chart whose statistic matches the sampling process.
**Output:** a chart specification, including limits and signal rules.

Use an individuals and moving-range chart when one continuous reading is recorded at each sampling time and its assumptions are reasonable.
The individuals chart monitors location; the moving-range chart provides a view of successive differences.
Investigate a moving-range signal before relying on an individuals-chart interpretation based on that range.

Use X-bar and R charts when rational subgroups of several continuous readings are collected and the range is suitable for subgroup size.
Use X-bar and s charts when subgroup standard deviation is the chosen dispersion statistic.
The chart constants depend on subgroup size and must match the chosen design.
Interpret the dispersion chart before interpreting the means chart.

Use a p chart for the fraction of inspected units that are defective.
The sample size can vary, yielding different control limits for each sample.
For independent Bernoulli outcomes with a common baseline fraction p-bar and sample size n, the approximate limits are p-bar plus or minus three times the square root of p-bar times (1 minus p-bar) divided by n.
Clip a negative lower limit to zero and an upper limit above one to one for display.
The normal approximation can be poor for rare defects or small samples; exact or alternative methods may be needed.

Use an np chart for the count of defective units when the sample size is constant.
Use a c chart for counts of defects when the inspection opportunity is constant and a Poisson model is plausible.
Use a u chart for defects per unit when exposure varies and the Poisson model is plausible.
Clustering, overdispersion, and changing opportunity definitions can invalidate standard count-chart limits.

### Step 5 — Establish a Phase I baseline

**Input:** representative historical observations with timestamps and contextual events.
**Action:** order data, compute provisional limits, investigate signals, and document justified exclusions.
**Output:** approved baseline data, limits, signal rules, and an effective date.

A baseline should represent the operating conditions that the chart will monitor.
Include routine shifts, operators, and normal material sources if they belong to the intended process population.
Avoid mixing distinctly different products or machines just to obtain more observations.
Sample size affects uncertainty: a short series gives unstable estimates even if no signals appear.

For an individuals chart, define moving range MR at time i as the absolute difference between measurement i and measurement i minus one.
There are n minus one moving ranges for n consecutive measurements.
The baseline mean is the sum of all n measurements divided by n.
The mean moving range is the sum of the n minus one ranges divided by n minus one.
For moving ranges of length two, estimate within sigma as mean moving range divided by d2 = 1.128.
The approximate individuals-chart limits are mean plus and minus three times this sigma estimate.
Equivalently, use mean plus and minus 2.660 times mean moving range, with rounding.

For the companion moving-range chart, the center line is mean moving range.
For ranges of two, its approximate lower limit is zero and upper limit is 3.267 times mean moving range.
These constants assume the standard independent normal-observation model used for the conventional chart.
If the process is strongly nonnormal or autocorrelated, nominal false-alarm behavior need not hold.

Investigate each potential special cause using operator logs, material changes, equipment events, and measurement records.
An outlier is not automatically erroneous.
Remove an observation from a revised baseline only when there is a documented, defensible reason and an approved analysis policy.
Keep the original data and all exclusion decisions auditable.
If no cause is found, do not invent one to make the chart look stable.

### Step 6 — Monitor Phase II and react

**Input:** frozen baseline limits, new time-ordered observations, and a written reaction plan.
**Action:** plot observations against the fixed limits and apply the agreed signal rules.
**Output:** an event, investigation, containment decision, and documented closure when required.

One common rule is one point beyond a three-sigma control limit.
Additional run or trend rules can improve sensitivity to smaller shifts but increase false alarms when applied together.
State all rules before monitoring and train operators to apply them consistently.
A three-sigma chart is not a guarantee of exactly one false alarm in any particular operating period.

On a signal, identify the last known acceptable process state and the potentially affected product interval.
The actual containment boundary depends on the process, sampling interval, traceability, and applicable quality procedures.
Notify the designated quality and production owners.
Preserve settings and evidence before resetting equipment where safe and practical.
Assess product against approved acceptance rules; do not use control limits as product specifications.

If the process remains stable but poorly centered, change the system through a planned improvement rather than repeatedly adjusting after ordinary observations.
Record changes and avoid comparing pre-change and post-change measurements as though they were one unchanged process.
Approve a new baseline only when the change is understood and sufficient representative data exist.
Rebaselining is not a device for making an unfavorable chart appear acceptable.

### Step 7 — Assess capability

**Input:** an approved specification, a stable representative process, and a valid measurement system.
**Action:** estimate center and variation and evaluate the distributional model.
**Output:** capability estimates, uncertainty, actual observed nonconformance, and a decision.

For a two-sided characteristic, Cp = (USL minus LSL) / (6 times within sigma).
Cpk = minimum of (USL minus mean) / (3 times within sigma) and (mean minus LSL) / (3 times within sigma).
A centered process has Cp and Cpk approximately equal.
A shifted mean reduces Cpk even when the spread, and therefore Cp, is unchanged.
If the mean lies outside a specification, Cpk can be negative.

A large Cp alone cannot establish that units conform because Cp ignores centering.
A high Cpk is not a universal release decision and is not proof of zero defects.
A threshold such as 1.33 may be a contractual or organizational convention for a specific application; it is not a universal law.
Report the sample size and uncertainty, ideally with confidence intervals appropriate to the design.
For very small samples, point estimates can be highly uncertain.

Normal-based tail predictions require a suitable normal model as well as stability and appropriate sigma.
Histograms, probability plots, engineering knowledge, and residual analysis can help evaluate the model.
A visually plausible histogram is not proof that extreme tails are accurately estimated.
For a skewed, bounded, mixed, or multimodal process, consider a justified nonnormal model, transformation, stratification, or empirical assessment.
Do not silently apply normal tail tables to a distribution that does not support them.

Overall sigma is computed across all observations in the stated period.
Pp and Ppk use overall sigma in the corresponding formulas.
Within and overall variation can differ because of drift, between-shift differences, and the subgroup design.
Neither index repairs an unstable process or substitutes for understanding the source of the variation.

### Step 8 — Analyze defects and turn evidence into a plan

**Input:** inspected units, defects by type and severity, production routing, and cost and capacity data.
**Action:** reconcile denominators, calculate rates, stratify losses, and translate them into good output.
**Output:** a quality-loss view and an explicit production-planning assumption.

Defective fraction equals defective units divided by inspected units.
DPU equals total defects divided by inspected units.
DPMO equals total defects divided by inspected units times opportunities per unit, multiplied by one million.
The last formula assumes that the opportunity count and counting rule are consistently defined.
DPMO is not necessarily the probability that a unit is defective.

A Pareto analysis ranks categories by count, cost, downtime, customer impact, or another declared metric.
A frequent minor cosmetic defect and a rare safety defect require different decisions.
Count alone is not a severity assessment.
Separate scrap from rework and identify whether rework returns to the constrained operation.

If expected final yield is 0.98 good units per started unit and all other losses are already included, a forecast of 9,800 good units requires 10,000 starts.
If rework consumes bottleneck capacity, calculate its additional time rather than simply applying a yield multiplier.
When forecasts change, communicate assumptions to production scheduling, procurement, and customer service.
Do not relabel a statistical forecast as a customer commitment without authorization.

### Step 9 — Distinguish monitoring sensitivity from inspection coverage

**Input:** process cycle rate, sample interval, potential failure duration, and detection rule.
**Action:** calculate the exposure between observations and consider the size of shift the chart needs to detect.
**Output:** a documented sampling interval and escalation tradeoff.

Suppose a line produces 600 units per hour and samples one unit every 30 minutes.
Approximately 300 units are produced between consecutive samples at that constant rate.
This is an exposure calculation, not a guarantee that exactly 300 units are suspect after a signal.
The actual affected interval depends on when the cause began, how quickly the chart detected it, and what traceability proves.
If the process can fail and recover between observations, the entire event may be missed.
If a chart rule needs multiple observations, detection can take longer than one sample interval.

A single three-sigma rule is relatively sensitive to a large abrupt shift.
Smaller persistent changes may require many observations before that rule signals.
Other methods, such as cumulative-sum or exponentially weighted moving-average charts, may detect smaller persistent shifts sooner under suitable assumptions.
They require their own design parameters, limits, and reaction procedures.
They are not simply extra run rules added without considering the resulting alarm rate.

Sampling frequency is a joint quality and production decision.
More samples use operator or laboratory time and may require destructive tests.
Less frequent sampling can enlarge the potentially affected inventory and delay detection.
Calculate a range of plausible containment quantities instead of claiming perfect detection.
If the line stops on each signal, account for the lost capacity and the cost of false alarms.
If operators cannot respond promptly, adding alarms without response capacity can increase alarm fatigue.

For rare but severe hazards, periodic sampling alone may not be an adequate control.
Engineering safeguards, process interlocks, validated inspection, and traceability may be necessary under the applicable risk management process.
An SPC chart does not override a mandatory safety control.
The quality function and relevant technical specialists own the adequacy of the overall control strategy.

### Step 10 — Investigate shifts without overfitting

**Input:** time-stamped signal, process conditions, recent change records, and product genealogy.
**Action:** compare plausible causes with evidence in time order and test the proposed explanation.
**Output:** an investigation record with cause, uncertainty, corrective action, and verification responsibility.

Start by confirming that the signal is real and not an instrument or data-processing error.
Check that the observation was attached to the correct line, unit, and timestamp.
Retain the original value even if a correction is authorized.
If a unit was sampled correctly but the instrument malfunctioned, document the measurement failure separately from product disposition.

Construct a short timeline: last routine observation, machine adjustments, material changes, maintenance, alarm events, and the signal.
Interview operators without assuming they caused the problem.
Look for an observable mechanism that could affect the measured characteristic.
For example, a clogged nozzle could change fill mass, but the chart alone does not prove a clog.
Confirm or refute the hypothesis using the machine inspection and relevant process records.

Avoid selecting the first factor that changed merely because it is convenient.
Several factors may change together at a shift boundary.
An apparent association can arise because a product, machine, and operator all changed at the same time.
Use a controlled comparison or designed experiment when a cause cannot otherwise be isolated and when it is safe to do so.
Never create a hazardous condition simply to reproduce a defect.

Distinguish correction from corrective action.
Correction addresses the immediate product or setting, such as segregating affected cartridges.
Corrective action addresses the mechanism, such as changing a maintenance procedure after evidence confirms nozzle blockage.
Effectiveness checks ask whether the action prevents recurrence over an appropriate period.
They do not consist solely of observing one favorable point.

Close an investigation with explicit uncertainty when the cause remains unknown.
The absence of an identified cause is not permission to claim that the signal was harmless.
The process owner decides the next monitoring and containment steps under the approved quality procedure.
Retain the event for recurring-pattern analysis.

### Step 11 — Separate short-term from long-term variation

**Input:** measurements grouped by time, machine, shift, and other meaningful operating conditions.
**Action:** compare within-subgroup and overall variation without mixing unlike populations.
**Output:** a documented interpretation of short-term capability and longer-term performance.

Imagine five adjacent units are measured every two hours.
The spread among those five units primarily reflects variation within a short window.
Differences between subgroup means may reflect changes between windows.
The subgroup design determines which sources appear in each estimate.
It does not magically label all within-window variation as permanent or all between-window variation as removable.

If the means shift over days, overall sigma may exceed within sigma.
Cp and Cpk based on within sigma can look better than Pp and Ppk based on overall sigma.
That difference is a prompt to investigate time-related variation, not automatic proof of any one cause.
If the process is unstable, neither pair should be presented as a dependable future prediction.
Reporting both without identifying their time frame and sigma definitions invites confusion.

Changing the sampling interval can change the moving-range estimate.
For a positively correlated series, consecutive differences may be small even while the process wanders substantially over a longer horizon.
An apparently high Cpk based on that small moving range may substantially overstate long-horizon performance.
Assess dependence and control dynamics before treating adjacent-range variation as representative.
Preserve the sampling design with the published index so later readers can interpret it.

### Step 12 — Interpret zero observed failures

**Input:** number of independent inspected units, sampling method, detection quality, and number of observed defective units.
**Action:** report the observed count and a suitable uncertainty bound rather than claiming zero risk.
**Output:** an evidence statement with numerator, denominator, confidence method, and inspection limitations.

Suppose an independent random sample of 300 units has no observed defective units.
The observed defective fraction is 0 / 300 = 0%.
That result does not demonstrate that the true defective probability is zero.
Under independent Bernoulli sampling with a constant probability and perfect detection, a common approximate one-sided 95% upper bound is 3 / 300 = 1%.
This approximation is often called the rule of three.

The bound changes if the sample is not random, units are clustered, or inspection misses defects.
It also changes when a different confidence level or exact binomial method is used.
An upper bound of approximately 1% is not evidence of a parts-per-million failure rate.
For a rare risk, a much larger representative sample or a justified process model may be needed.
Neither approach eliminates the need for engineering controls.

When inspections are destructive, sampling itself consumes saleable units and laboratory capacity.
Specify whether the sample is drawn before or after rework, and whether results represent first-pass or final quality.
If a unit can fail multiple characteristics, specify which failure definition the zero count covers.
Do not add together separate zero-failure claims as though they came from independent populations without checking overlap.
If failure severity differs, report each critical category separately where the risk assessment requires it.

### Step 13 — Design the quality-to-planning handoff

**Input:** current good inventory, held inventory, expected yield, available capacity, and order priority.
**Action:** create a time-phased view of starts, expected good receipts, holds, and authorized releases.
**Output:** a planning scenario with an accountable owner and decision date.

First establish a physical balance for each product and period.
Starting inventory plus receipts minus shipments minus scrapped inventory equals ending inventory, subject to the site's defined categories.
Held inventory is physically present but may not be available to promise.
Rework inventory is not automatically finished goods.
Goods should change status only when the authorized quality disposition is recorded.

Then calculate production starts and expected yield over the same time bucket.
Use an estimate representative of the relevant product, line, and conditions.
If a yield estimate already includes ordinary rework, do not add the same recovered units again.
If rework is scheduled in a later week, the planning model must place that receipt in the later week.
Inventory timing can matter even when total eventual yield is unchanged.

Next identify constrained resources.
A scrap event before the bottleneck can reduce the input available to that bottleneck.
A defect found after the bottleneck can consume scarce bottleneck time without producing a saleable unit.
Rework may return to the same constrained resource or require a different cell.
The relevant economic loss depends on routing, contribution margin, available demand, and time, not just the material price.

Communicate a baseline and an adverse scenario.
For example, the baseline might assume a held lot is released after successful inspection.
The adverse scenario might assume part of it is scrapped and replacement production uses an extra shift.
Label both as scenarios until the disposition decision is made.
Customer service should not describe either scenario as a commitment without the appropriate approval.

### Step 14 — Maintain the control system

**Input:** approved chart design, event log, process changes, and business objectives.
**Action:** review chart performance, response times, data quality, and relevance on a scheduled basis.
**Output:** a controlled decision to continue, revise, retire, or rebaseline the chart.

Audit whether operators actually collect the required samples.
Compare planned sampling timestamps with observed timestamps.
Investigate long gaps and suspiciously repeated values.
Review whether alarms were acknowledged, investigated, and closed by the assigned owners.
An attractive chart with no operational response is not an effective process control.

Check the product and process population when a tool, supplier, recipe, or routing changes.
A material process change may require a new baseline after validation.
Do not assume every minor maintenance action invalidates all historical limits.
Conversely, do not retain obsolete limits merely to preserve a favorable performance trend.
Document the technical basis, approval date, and version of each change.

Review economic measures without letting them replace technical controls.
Track held quantity, scrap, rework hours, detection delay, and customer impact separately.
Examine whether quality improvements released bottleneck capacity or simply moved work to another station.
Compare outcomes on a consistent product-mix and time basis.
Any causal benefit claim should state other major changes during the comparison period.

### Worked chart-choice comparisons

**Case A: dimensional measurements.**
One diameter is measured every 20 minutes on a single turning machine.
The measurement is continuous in millimeters.
An individuals and moving-range chart is a candidate if its assumptions fit the time series.
If adjacent readings are strongly autocorrelated because of tool wear, evaluate an alternative model or chart design.
The drawing supplies the specification; it does not supply the statistical chart limits.

**Case B: five consecutive measurements.**
An operator measures five consecutive diameters every two hours.
The within-subgroup range describes short-window variation.
An X-bar and R chart is a candidate with constants for a subgroup size of five.
The sequence of subgroup means can reveal changes between sampling windows.
A subgroup mixing five different machines would answer a different question and could hide machine-specific behavior.

**Case C: damaged cartons.**
A warehouse inspects 100 cartons on one shift and 160 on the next.
Each carton is classified as damaged or not damaged.
A p chart accommodates the different denominators through sample-specific limits.
The classification method and inspected population must remain comparable across shifts.
If damaged cartons arrive in clusters from one pallet, the independence assumption needs investigation.

**Case D: scratches per panel.**
An inspector counts all scratches on each panel.
A panel can have more than one scratch.
With a fixed panel area and a plausible Poisson model, a c chart is a candidate.
If inspected area differs, a u chart using defects per unit area may be more appropriate.
If the inspection lighting changes, apparent defect rates may change even if the process does not.

**Case E: critical rare failures.**
One leak in a large batch has a severe consequence.
A routine p chart with sparse samples may detect a worsening rate only after an unacceptable delay.
Consider the complete hazard-control strategy and the inspection plan, not a chart in isolation.
Exact attribute methods or time-between-event charts may be appropriate with expert design.
The choice requires evidence about the failure mechanism, detection system, and applicable obligations.

### Worked attribute limits

Suppose an established baseline has 40 defective cartons among 2,000 inspected cartons.
The baseline fraction is p-bar = 40 / 2,000 = 0.020 defective cartons per inspected carton.
For a new sample of n = 100 cartons, the approximate standard error is sqrt(0.020 × 0.980 / 100) = 0.0140.
The three-sigma upper limit is 0.020 + 3 × 0.0140 = 0.0620.
The calculated lower limit is negative and is displayed as zero.
An observed fraction of 7 / 100 = 0.070 is above this approximate upper limit.

For a sample of n = 400 cartons, the approximate standard error is sqrt(0.020 × 0.980 / 400) = 0.0070.
The corresponding upper limit is 0.0410.
The lower limit is again clipped to zero.
An observed fraction of 18 / 400 = 0.045 exceeds this sample-specific approximate limit.
The difference in limits arises from sample size, not a changed product specification.
With small expected defect counts, check an exact or other appropriate method instead of relying mechanically on this approximation.

In either sample, investigate the signal according to the reaction plan.
The chart does not determine which cartons are damaged or whether an entire shipment is acceptable.
Record the observed counts and actual sample sizes with the chart.
If the inspection standard changed, do not interpret a shift as a process change without checking detection.
If defective counts include reinspection of the same cartons, reconcile the unit population first.

### Worked defect-cost comparison

Suppose one week has 80 scratched units and four leak failures.
Each scratched unit requires 2 minutes of rework at a stated conversion cost of USD 0.50 per minute.
The direct rework conversion cost is 80 × 2 × USD 0.50 = USD 80.
Suppose each leak failure requires USD 25 of direct investigation and scrap cost.
The direct leak cost is 4 × USD 25 = USD 100.
The two categories should not be compared solely by count: scratches are more frequent, but leaks have a larger stated direct cost.

These invented costs exclude customer harm, capacity constraints, and potential field consequences.
The example does not assign a monetary value to safety or regulatory exposure.
If leak severity is high, risk controls may take priority regardless of the simple cost ranking.
If scratched units also contain leaks, avoid counting the same scrap cost twice.
Validate routing and timing before translating rework minutes into lost throughput.

### Business decision checklist

Before publishing a chart, state the operating population.
Name the line, product, characteristic, and period.
State the unit and measurement resolution.
State the sample selection and chart family.
Name the baseline period and point count.
Show the center line and control limits.
Specify the signal rules.
Name the response owner and escalation threshold.
Separate the specification limits from chart limits.
Identify the product-disposition authority.

Before publishing a capability index, verify stability for the intended claim.
State whether sigma is within or overall.
Report the specification revision and whether the characteristic is one-sided.
State the distributional model if translating capability into estimated tails.
Report sample size and uncertainty.
Disclose exclusions and the sampling interval.
Avoid comparing indices built from incompatible subgroup designs.
Explain the operational decision supported by the estimate.
Retain the source observations and calculation version.
Reassess after a substantive process or specification change.

Before publishing a defect KPI, state whether the numerator counts units or defects.
Identify the denominator and the inspection opportunity.
Define the inspection stage and whether rework is included.
Separate observed data from model-based forecasts.
Identify uninspected production and missing records.
State whether repeated inspections were deduplicated.
Show category severity and financial assumptions separately.
Reconcile the KPI with physical product flow.
Flag changes in detection method or opportunity definition.
Assign an owner for corrective action and planning communication.

## 6. Worked example

### Business situation and decision

A fictional plant fills adhesive cartridges on line A.
The approved fill-mass specification is 98.0 g to 102.0 g inclusive.
The process target is 100.0 g, and a customer order requires 9,800 acceptable cartridges next week.
The scheduler must decide whether to plan 10,000 starts and whether to promise the full order.
The quality engineer must decide whether the fill process warrants investigation.

The plant records one cartridge measurement each hour.
A validated scale is assumed, and the example treats hourly readings as independent for illustration.
The sample is illustrative, short, and not sufficient to qualify a real production baseline.
An actual decision would use a representative Phase I study, traceability, and the plant's release procedure.

### Provisional historical measurements

| Hour | Mass (g) | Moving range (g) |
|---:|---:|---:|
| 1 | 99.6 | — |
| 2 | 100.0 | 0.4 |
| 3 | 99.8 | 0.2 |
| 4 | 100.2 | 0.4 |
| 5 | 100.0 | 0.2 |
| 6 | 100.4 | 0.4 |
| 7 | 100.2 | 0.2 |
| 8 | 100.6 | 0.4 |
| 9 | 100.4 | 0.2 |
| 10 | 100.0 | 0.4 |

The ten measurements sum to 1,001.2 g.
The provisional mean is 100.12 g.
The nine moving ranges sum to 2.8 g.
The mean moving range is 2.8 / 9 = 0.3111 g.
The within-sigma estimate is 0.3111 / 1.128 = 0.2758 g.

The individuals-chart upper limit is 100.12 + 3 × 0.2758 = 100.9474 g.
The lower limit is 100.12 − 3 × 0.2758 = 99.2926 g.
The moving-range chart upper limit is 3.267 × 0.3111 = 1.0164 g.
Its lower limit is 0 g.
All ten historical individual readings and nine ranges are within these provisional limits.
That observation does not establish real-world stability, especially with only ten readings.

### Capability and interpretation

Cp = (102.0 − 98.0) / (6 × 0.2758) = 2.417, approximately.
The upper-side index is (102.0 − 100.12) / (3 × 0.2758) = 2.272, approximately.
The lower-side index is (100.12 − 98.0) / (3 × 0.2758) = 2.562, approximately.
Cpk is the smaller index, approximately 2.272.
The upper specification is closer to the mean, so the upper side governs Cpk.
These values are illustrative point estimates, not a qualified production capability claim.
The short, deliberately regular sequence is not sufficient for a reliable tail estimate.

### New observation and response

The next hourly mass is 101.1 g.
It is within the 98.0 g to 102.0 g product specification.
It is above the provisional 100.9474 g individuals-chart upper control limit.
The new moving range is |101.1 − 100.0| = 1.1 g.
That range also exceeds the provisional 1.0164 g moving-range upper limit.
This is a process signal even though the sampled cartridge conforms to its specification.

Under the agreed reaction plan, the operator notifies quality and the supervisor.
The team identifies the potentially affected production interval using the last confirmed process state and traceability records.
It checks the scale, filler setting, recent material changes, and maintenance events.
Quality determines containment, inspection, and disposition under the authorized procedure.
The planner records the affected quantity as unavailable until disposition is complete.
The engineer does not announce a cause solely from the point and does not silently widen the limits.

### Planning implications

Assume recent, separately validated final yield is 98.0%, including all ordinary scrap and successful rework outcomes.
The forecast from 10,000 starts is 10,000 × 0.98 = 9,800 good cartridges.
The plan is 10,000 starts; the target is 9,800 good cartridges.
The customer commitment is 9,800 cartridges only if an authorized owner accepts the risk and capacity after containment.
The 98.0% yield assumption is not calculated from the ten fill measurements.
The signal creates additional uncertainty and may delay release even if the forecast remains arithmetically unchanged.

If 200 cartridges are held pending disposition, they cannot be counted as available finished goods immediately.
If half are later released and half scrapped, the incremental impact must be reconciled against the original yield assumption to avoid double counting.
If rework occupies the bottleneck, schedule its actual capacity consumption.
A cost comparison should include inspection, scrap, rework, lost margin, and delayed-delivery exposure in USD, with no invented prices.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Copy it into a file and run it with `python file.py`.
It reproduces the example's baseline, chart limits, indices, and next-point signals.
The baseline values are assumed to be in chronological order and on the same measurement scale.
The function rejects a zero moving range because conventional sigma-based capability is then undefined.
It does not establish stability, normality, measurement adequacy, confidence intervals, or release authorization.

```python
from statistics import mean


def analyze_individuals(values, lsl, usl, next_value):
    if len(values) < 2:
        raise ValueError("At least two measurements are required")
    if lsl >= usl:
        raise ValueError("LSL must be below USL")

    ranges = [abs(b - a) for a, b in zip(values, values[1:])]
    center = mean(values)
    mr_bar = mean(ranges)
    sigma = mr_bar / 1.128
    if sigma == 0:
        raise ValueError("Zero moving range: capability is undefined")

    lower = center - 3 * sigma
    upper = center + 3 * sigma
    mr_upper = 3.267 * mr_bar
    cp = (usl - lsl) / (6 * sigma)
    cpk = min((usl - center) / (3 * sigma),
              (center - lsl) / (3 * sigma))
    next_range = abs(next_value - values[-1])
    return {
        "center": center,
        "mr_bar": mr_bar,
        "sigma": sigma,
        "lower": lower,
        "upper": upper,
        "mr_upper": mr_upper,
        "cp": cp,
        "cpk": cpk,
        "next_range": next_range,
        "individual_signal": next_value < lower or next_value > upper,
        "range_signal": next_range > mr_upper,
        "next_conforming": lsl <= next_value <= usl,
    }


baseline = [99.6, 100.0, 99.8, 100.2, 100.0,
            100.4, 100.2, 100.6, 100.4, 100.0]
result = analyze_individuals(baseline, 98.0, 102.0, 101.1)
for name in ("center", "mr_bar", "sigma", "lower", "upper",
             "mr_upper", "cp", "cpk", "next_range"):
    print(f"{name}: {result[name]:.4f}")
for name in ("individual_signal", "range_signal", "next_conforming"):
    print(f"{name}: {result[name]}")
```

Expected output, allowing for ordinary rounding:

```text
center: 100.1200
mr_bar: 0.3111
sigma: 0.2758
lower: 99.2926
upper: 100.9474
mr_upper: 1.0164
cp: 2.4171
cpk: 2.2721
next_range: 1.1000
individual_signal: True
range_signal: True
next_conforming: True
```

The code deliberately freezes baseline limits before evaluating the new observation.
Recalculating limits with that observation would change the question and could mask a shift.
The limits are approximate because published constants are rounded.
The code uses no additional run rules; a production system must implement its approved rules explicitly.

## 8. Data, standards, and regulatory context

### Minimum data contract

Store the observation timestamp with timezone and the actual measurement value in its original unit.
Keep a separate ingestion timestamp when offline uploads or delayed entry are possible.
Order by observation time, not database arrival time.
Record the line, machine, product, product revision, and characteristic identifier.
Record the gauge, gauge resolution, operator or automated source, and measurement procedure revision.

For attribute data record units inspected, defective units, defects by category, and applicable exposure.
Capture whether inspection was full, sampled, destructive, or based on a special investigation.
Mark reinspection and rework records so the same unit is not accidentally counted twice.
Preserve the identifier linking a unit to its lot and production time.
Retain raw measurements, corrections, and an audit trail rather than overwriting inconvenient observations.

Specification records require effective dates, authorization, unit, and the exact acceptance rule.
A rounded display value must not silently replace the raw value used for acceptance.
Missing measurements are missing, not zero and not automatically conforming.
Late data can change an apparent chart sequence; define correction and notification procedures.
Restrict access and retain data according to the organization's applicable quality and privacy requirements.

### Relevant standards and authority

ISO 7870-1 describes general principles of control charts.
ISO 7870-2 addresses Shewhart control charts.
ISO 22514-1 covers general principles and concepts for process capability and performance.
ISO 22514-2 addresses capability and performance for time-dependent process models.
Consult the current licensed editions and the applicable contract before adopting detailed methods.
These standards are voluntary unless incorporated into a binding contract, regulation, or other applicable requirement.
They do not themselves authorize release of a particular lot.

ISO 9001 specifies quality-management-system requirements for organizations seeking conformity to that standard.
It is not a universal statutory mandate to calculate Cpk for every characteristic.
Industry-specific customer requirements may impose additional methods, sample sizes, reporting, or thresholds.
Check the applicable revision and actual customer agreement rather than assuming one threshold applies everywhere.

In the United States, 21 CFR 820 governs quality management system requirements for medical devices within its scope.
The FDA's Quality Management System Regulation became effective February 2, 2026 and incorporates ISO 13485:2016 by reference, with additional provisions.
Applicability depends on the product, activity, jurisdiction, and current rules.
A chart is evidence within a validated quality system, not a substitute for the applicable regulatory requirements.
This educational treatment is not legal or regulatory advice.

### Sampling and retention choices

Document sampling frequency and the risk rationale before collecting a favorable series.
An acceptance-sampling plan answers a lot-disposition question under its stated operating characteristics.
A control chart monitors process behavior; it is not automatically an acceptance-sampling plan.
Record rejected, held, reinspected, and released material so that observed rates have an interpretable denominator.
If only passed units appear in the dataset, calculated performance is subject to selection bias.

## 9. Industry practice and case evidence

The National Institute of Standards and Technology (NIST) Engineering Statistics Handbook presents control-chart methods and process-capability indices.
It distinguishes control limits derived from process data from specification limits reflecting external requirements.
NIST also discusses the sensitivity of capability estimates to distributional assumptions and sample size.
This public technical evidence supports the mechanics taught here; it is not evidence that the fictional plant achieved a particular result.

The Western Electric Statistical Quality Control Handbook popularized supplementary rules for interpreting control-chart patterns.
Using multiple rules changes the overall false-alarm behavior relative to a single point-beyond-three-sigma rule.
A site should choose rules based on its process and response capacity rather than combining every available rule without assessing consequences.

In its public guide to the automotive industry, the International Automotive Task Force describes IATF 16949 as a quality-management-system standard aligned with ISO 9001 and customer-specific requirements.
A supplier must examine the actual applicable standard and customer contract for its obligations.
The existence of an automotive quality framework does not establish that any named supplier met a particular Cpk value.

[INFERENCE] A plant with a well-instrumented bottleneck can often make faster containment decisions if its quality events, machine timestamps, and lot traceability are linked.
That conclusion follows from reducing time spent identifying the affected interval, but the magnitude depends on local systems and must be measured.
[INFERENCE] A stable but incapable fill process is more likely to benefit from a deliberate centering or variation-reduction project than from repeated ad hoc adjustments.
Neither inference substitutes for an actual site investigation or trial.

Compare process interventions with an explicit counterfactual.
For instance, track scrap and rework before and after a documented equipment change, while considering product mix and measurement changes.
Do not attribute an improvement solely to training when the gauge, material, and specifications changed at the same time.
Report the time window, number of units, and uncertainty along with any business saving.
An announced customer case or vendor case study should be labeled as such, not presented as independently audited causal proof.

### Applying the evidence to a plant decision

The following scenarios are teaching applications, not published company case outcomes.
They show how the documented distinction between process control and product acceptance affects operational choices.
Each scenario identifies what is known and what would need local evidence.

**Scenario 1: stable but miscentered dispensing.**
An adhesive dispenser has a stable series of measurements close to its upper specification.
The quality team has verified the gauge and approved a representative baseline.
The mean is not at the process target.
The upper-side capability index is much lower than the lower-side index.
The production team is considering a setpoint change.

The evidence supports investigating a deliberate centering change.
It does not establish the safest adjustment size or show that variance will remain constant after the change.
The team should assess whether a different target affects bonding performance, material usage, or another specification.
An authorized trial can compare before-and-after observations under comparable operating conditions.
The post-change process needs its own stability assessment.
Planning should estimate any downtime and material consumed by the trial separately from expected recurring benefit.

**Scenario 2: unstable but currently conforming packaging.**
Carton seal strength measurements remain inside specification, but several points violate the site's approved chart rules.
The shift supervisor wants to continue at full speed because no sampled units have failed acceptance.
The chart is based on a current and representative baseline, and the gauge is known to be adequate.
The potentially affected interval includes product produced between the last routine observation and the signal.

The chart supports investigation and the site's prescribed reaction.
It does not independently establish how much product is nonconforming.
Quality must determine containment and disposition using the approved procedure.
Planning should identify held inventory and alternative capacity before committing a revised shipping date.
If the cause is an instrument problem, the evidence and affected product decisions still need documentation.
If the cause is a failing heater, the maintenance response and effectiveness check should address the mechanism.

**Scenario 3: apparent improvement after an inspection change.**
A plant reports that scratch defects per million opportunities fell by half in one month.
During that month it also expanded the recorded list of inspection opportunities per panel.
The number of observed scratches and panels inspected did not change.
The larger denominator alone can lower the published DPMO.
That change does not by itself demonstrate an improvement in the physical process.

Recalculate both periods using a comparable opportunity definition where the data permit.
Also report defects per panel and defective panels per inspected panel.
Check whether the lighting, inspector training, or inspection position changed.
If historical data cannot be recoded reliably, disclose the break in comparability.
Use the new definition prospectively and preserve the old definition and revision date.
An operational improvement claim requires evidence beyond a denominator change.

**Scenario 4: production-rate pressure at a bottleneck.**
A constrained work center is measured on units per hour.
Management proposes faster machine speed to improve output.
The quality engineer expects a possible change in defect rate and variation.
The planner expects that faster gross output might reduce late orders.
The decision cannot be made from gross rate alone.

Estimate good units per constrained hour, including expected scrap and the actual rework route.
If the speed change increases failure after the bottleneck, some scarce processing time is lost to unsaleable output.
If rework returns to the bottleneck, reserve that time in the schedule.
If rework occurs on unconstrained equipment, its cost and delay still matter even if it does not use bottleneck time.
Use a controlled trial and monitor safety, quality, and equipment constraints.
Compare the trial with a representative baseline and disclose uncertainty before changing the production commitment.

**Scenario 5: supplier comparison.**
Two material suppliers report Cpk for nominally the same characteristic.
Supplier A uses a within-sigma estimate from consecutive samples on one machine.
Supplier B uses overall sigma from a month of mixed-machine output.
The two values do not measure exactly the same thing.
Differences in specification revisions, units, measurement systems, and sample size can further undermine comparison.

Request the raw summary definitions, study period, sampling design, and stability evidence.
Ask whether each report is Cp/Cpk or Pp/Ppk and which sigma estimate was used.
Compare like with like or commission a mutually agreed study.
Do not award business solely to the supplier with the larger unlabeled number.
Consider supply continuity, incoming inspection evidence, cost, and the applicable qualification process.

**Scenario 6: a claimed zero-defect month.**
A site inspected 300 units and found no defective units.
The monthly presentation describes the process as having zero defects.
That wording could mean zero observed failures, zero failures in all produced units, or zero underlying risk.
Only the first statement is supported by the sample result.
The population coverage and sampling method determine what can be inferred about the rest.

Report zero observed defective units out of 300 inspected units.
Where the Bernoulli assumptions are defensible, report the approximate upper confidence bound described above.
Identify uninspected units and any known inspection blind spots.
Continue controls appropriate to the severity of the failure mode.
Avoid presenting an observed zero as a guarantee to customers.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Specifications used as limits | Chart limits match drawing exactly | Product requirements replace process model | Shifts go undetected or false alarms proliferate | Calculate and label each limit separately |
| Unstable capability | Repeated chart signals | One distribution mixes different regimes | Cpk misstates future risk | Investigate and stabilize before inference |
| Poor gauge resolution | Many identical values | Rounding suppresses apparent variation | Inflated capability | Assess resolution and gauge variation |
| Selected samples | Only attractive units reach inspection | Nonrandom selection | Optimistic defect rate | Define random or systematic selection |
| Product mixing | Two peaks in histogram | Distinct streams share one baseline | Misleading limits | Stratify by process and characteristic |
| Ignored serial correlation | Long runs and slowly drifting values | Adjacent observations depend on each other | Wrong false-alarm rate | Model dependence or redesign chart |
| Overadjustment | Operators move settings after every point | Reaction to common-cause noise | More variation and downtime | Use a predefined reaction rule |
| Hidden rework | Repeated unit IDs counted as new starts | Denominator inflation | Understated loss and capacity | Track unit genealogy and first pass |
| Wrong defect denominator | DPMO labeled as defective ppm | Opportunities confused with units | Incorrect business claim | Publish definitions and reconciliations |
| Uncontrolled rebaseline | Limits change after poor performance | Signals absorbed into baseline | Deterioration normalized | Require approval and retain history |
| Small sample | High Cpk from a few readings | Estimate uncertainty ignored | False confidence in tails | Obtain representative data and intervals |
| Wrong chart | p chart used for multiple defects per unit | Binomial unit model mismatches counts | Invalid limits | Choose chart from data-generating process |
| Delayed reaction | Repeated alarms without closure | No owner or authority | Large containment interval | Assign actions, escalation, and deadline |
| Incomplete opportunity definition | DPMO drops after checklist expansion | Denominator changes without improvement | Artificial KPI gain | Version and audit opportunity definition |
| Release inferred from chart | Stable chart treated as acceptance | Stability confused with conformity | Nonconforming product shipped | Apply authorized product acceptance rules |
| Assumed normal tails | Skewed or mixed data | Model predicts wrong tail probability | Unreliable ppm forecast | Assess model and use justified alternative |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to assess |
|---|---|---|---|
| Representative baseline | Future monitoring compares with past behavior | New material or product mix | Compare operating conditions and strata |
| Stable process for capability prediction | A single distribution must describe future behavior | Shifts, trends, changing variance | Review charts and process history |
| Adequate measurement system | Observed variation should reflect process behavior | Gauge drift or coarse rounding | Calibration and measurement-system study |
| Independent observations for conventional limits | Nominal chart behavior relies on model assumptions | Process inertia or feedback control | Examine autocorrelation and process timing |
| Rational sampling | Within variation must match intended short-term variation | Arbitrary or mixed subgroups | Review collection design and routing |
| Suitable normal model for tail prediction | Normal tails translate indices into expected nonconformance | Skew, mixtures, boundaries | Probability plot and engineering review |
| Correct specification | Capability needs the applicable customer limits | Wrong drawing revision | Verify authorized effective revision |
| Constant defect opportunity | Count chart baseline assumes comparable exposure | Changing inspection scope | Track inspected area or opportunities |
| Comparable inspection | Rate changes should reflect quality, not detection | Inspector or method change | Repeatability study and audit |
| Known yield scope | Good-output forecast requires nonoverlapping losses | Rework and scrap counted twice | Reconcile physical unit flows |
| Sufficient data | Estimates need acceptable uncertainty | Ten convenient observations | Report sample size and intervals |
| Prompt signal response | A chart only helps if acted on | No trained owner or escalation | Review event logs and response time |

A stable process can still have unacceptably high defect levels.
A capability index can be a useful summary while hiding the location, severity, and financial impact of failures.
No finite sample proves zero future defects.
When assumptions are not met, report the limitation rather than presenting the output as a qualified capability claim.

## 12. Exercises

### Exercise 1 — Control and specification limits

A shaft has specification limits of 9.90 mm and 10.10 mm.
Its approved chart limits are 9.94 mm and 10.04 mm.
A new measurement is 10.06 mm.
**Deliverable:** classify the measurement against both sets of limits and describe the immediate process action.
**Acceptance criteria:** distinguish conformity from a chart signal and avoid making an unsupported disposition decision.
**Time:** 5 minutes.

### Exercise 2 — Select a chart

A plant records the number of damaged cartons in samples of varying size each shift.
Each carton is either damaged or not damaged for this inspection.
**Deliverable:** select a chart and state the denominator and one assumption.
**Acceptance criteria:** identify the variable-sample-size defective-unit chart and its binomial-model limitation.
**Time:** 5 minutes.

### Exercise 3 — Moving range and signal

A baseline has center 50.0 mm and mean moving range 0.20 mm.
A new value is 50.6 mm; the immediately previous value was 50.0 mm.
Use d2 = 1.128 and moving-range multiplier 3.267.
**Deliverable:** calculate the individuals upper limit, moving-range upper limit, and both signals.
**Acceptance criteria:** show the formula, units, and response without equating a signal with a product defect.
**Time:** 10 minutes.

### Exercise 4 — Capability

A stable process has mean 20.3 mm, within sigma 0.20 mm, LSL 19.0 mm, and USL 21.0 mm.
**Deliverable:** calculate Cp and Cpk and identify the closer specification.
**Acceptance criteria:** show both one-sided distances and explain why Cp exceeds Cpk.
**Time:** 10 minutes.

### Exercise 5 — Critique a capability claim

An engineer reports Cpk 1.80 from twelve points after combining two machines and ignoring a visible shift.
**Deliverable:** list three reasons the claim is insufficient for prediction and propose an investigation.
**Acceptance criteria:** address stability, mixture, and uncertainty without asserting a specific failure rate.
**Time:** 10 minutes.

### Exercise 6 — Sampling design

A filling line switches material supplier every two days and has a slow within-shift drift.
**Deliverable:** propose a sampling and stratification design and state what each subgroup represents.
**Acceptance criteria:** include time order, supplier identity, detection-delay tradeoff, and measurement-system review.
**Time:** 15 minutes.

### Exercise 7 — Defects and output

An inspection examines 1,000 units, of which 40 have at least one defect.
Inspectors find 55 total defects across four defined opportunities per unit.
A planned production run starts 5,000 units; assume the observed defective fraction is an appropriate forecast of final loss and no rework occurs.
**Deliverable:** calculate defective fraction, DPU, DPMO, and forecast good units.
**Acceptance criteria:** show denominators and distinguish the forecast from a commitment.
**Time:** 10 minutes.

### Exercise 8 — Design the response

A line has a chart signal, a customer shipment due today, and incomplete unit traceability.
**Deliverable:** write a one-page reaction and communication plan for production, quality, and planning.
**Acceptance criteria:** assign owners, define a conservative affected interval and disposition authority, preserve evidence, and state a delivery decision with uncertainty.
**Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

The 10.06 mm measurement is inside the 9.90–10.10 mm specification range.
It exceeds the 10.04 mm upper control limit and is therefore a chart signal under the one-point rule.
Follow the reaction plan, investigate and contain potentially affected production as required.
Conformity of this one unit does not prove that the process is stable or the entire lot is acceptable.

### Exercise 2

Use a p chart for the fraction of damaged cartons in each shift's sample.
The numerator is damaged cartons and the denominator is cartons inspected in that shift.
With varying sample size, calculate limits for each sample size.
The ordinary model assumes comparable independent binary outcomes with a common baseline fraction.
Clustering or changing inspection definitions can violate that model.

### Exercise 3

Within sigma is 0.20 / 1.128 = 0.1773 mm.
The individuals upper limit is 50.0 + 3 × 0.1773 = 50.532 mm.
The new value 50.6 mm exceeds that limit.
The moving-range upper limit is 3.267 × 0.20 = 0.6534 mm.
The new moving range is 0.6 mm, below the range limit.
Investigate the individual signal under the local reaction plan.
No product specification was supplied, so conformity cannot be determined.

### Exercise 4

Cp = (21.0 − 19.0) / (6 × 0.20) = 1.667.
The upper-side index is (21.0 − 20.3) / (3 × 0.20) = 1.167.
The lower-side index is (20.3 − 19.0) / (3 × 0.20) = 2.167.
Cpk is 1.167 and the upper specification is closer.
Cp exceeds Cpk because the mean is displaced from the specification midpoint.
These are conditional estimates, not proof of a particular defect rate.

### Exercise 5

The visible shift undermines a single stable process model.
Combining machines can hide distinct centers, spreads, or causes.
Twelve points provide limited information and substantial uncertainty, especially about tails.
Stratify by machine, verify measurement and specifications, investigate the shift, and gather representative data.
Then estimate capability for the appropriate stable process and report uncertainty.

### Exercise 6 rubric

A strong design records representative time-ordered samples across shifts and supplier changes.
It makes subgroup timing explicit and separates supplier or machine strata where justified.
It evaluates detection delay, inspection burden, traceability, and gauge adequacy.
A weak design samples only at shift start or averages away the entire shift's drift.
There is no universally correct hourly interval without process-risk information.

### Exercise 7

Defective fraction = 40 / 1,000 = 0.04, or 4%.
DPU = 55 / 1,000 = 0.055 defects per unit.
DPMO = 55 / (1,000 × 4) × 1,000,000 = 13,750 defects per million opportunities.
Forecast good output = 5,000 × (1 − 0.04) = 4,800 units.
The 4,800 units are a forecast under the stated assumption, not a customer commitment.
The defect count exceeds the defective-unit count because some units contain multiple defects.

### Exercise 8 rubric

A strong answer gives quality authority over containment and disposition and gives planning responsibility for updating availability.
It defines the interval using available evidence and broadens it when traceability is incomplete.
It preserves settings and samples, names escalation owners, and records the shipment decision and communication time.
It does not release material merely because the customer deadline is near.
It identifies an uncertain delivery forecast and the owner authorized to make any revised commitment.

## 14. Interview questions

1. **What is the difference between a control limit and a specification limit?**
   Strong: process behavior versus product acceptance, with independent sources and uses.
   Weak: calls either one a tolerance determined from the other.
2. **Can a stable process be incapable?**
   Strong: yes; its stable distribution can overlap or lie outside specification.
   Weak: assumes stability guarantees conforming units.
3. **What does a chart signal mean?**
   Strong: a rule-triggered reason to investigate relative to a baseline, not an identified root cause.
   Weak: claims every signal proves a specific defect.
4. **Why chart moving ranges?**
   Strong: to monitor successive variation and support the individuals-chart sigma estimate.
   Weak: uses ranges as customer acceptance limits.
5. **Why can Cp exceed Cpk?**
   Strong: Cp ignores centering and Cpk uses the nearer specification limit.
   Weak: says Cp is always the actual yield.
6. **When would you avoid normal-based ppm predictions?**
   Strong: unstable, mixed, skewed, poorly measured, or inadequately sampled processes.
   Weak: derives extreme-tail certainty from any finite Cpk.
7. **How do p and u charts differ?**
   Strong: p tracks fraction of defective units; u tracks defects per unit of exposure.
   Weak: treats defects and defective units as identical.
8. **What makes a rational subgroup useful?**
   Strong: its collection design separates the variation relevant to the monitoring question.
   Weak: chooses group size without considering time or operating conditions.
9. **How do you respond when a conforming point signals?**
   Strong: investigate and follow containment and disposition procedures separately.
   Weak: ignores the signal because the point is within specification.
10. **How does a defect rate change a production plan?**
    Strong: translates validated yield into good output and considers rework, capacity, and uncertainty.
    Weak: treats a forecast as an unconditional commitment.

## 15. Key terms

- [Acceptance sampling](#8-data-standards-and-regulatory-context): lot-disposition sampling under a defined plan.
- [Attribute chart](#step-4--choose-the-chart-family): chart for counts or fractions.
- [Capability](#step-7--assess-capability): comparison of a modeled stable process with specifications.
- [Center line](#4-definitions-and-vocabulary): baseline central chart statistic.
- [Common cause](#3-the-core-mental-model): routine variation of a stable process.
- [Containment](#step-6--monitor-phase-ii-and-react): controlled identification and disposition of potentially affected product.
- [Control limit](#3-the-core-mental-model): statistical chart boundary.
- [Cp](#step-7--assess-capability): specification width relative to within-process spread.
- [Cpk](#step-7--assess-capability): closer specification distance relative to within-process spread.
- [Defect](#step-8--analyze-defects-and-turn-evidence-into-a-plan): individual nonconformity.
- [Defective unit](#step-8--analyze-defects-and-turn-evidence-into-a-plan): unit with at least one nonconformity.
- [DPMO](#step-8--analyze-defects-and-turn-evidence-into-a-plan): defects per million defined opportunities.
- [DPU](#step-8--analyze-defects-and-turn-evidence-into-a-plan): defects per inspected unit.
- [Individuals chart](#step-5--establish-a-phase-i-baseline): chart of time-ordered individual measurements.
- [Measurement system](#step-2--verify-the-measurement-system): full process of obtaining readings.
- [Moving range](#step-5--establish-a-phase-i-baseline): absolute difference between consecutive observations.
- [Overall sigma](#step-7--assess-capability): standard deviation across the stated observations.
- [p chart](#step-4--choose-the-chart-family): chart of defective fraction.
- [Phase I](#step-5--establish-a-phase-i-baseline): retrospective baseline development.
- [Phase II](#step-6--monitor-phase-ii-and-react): monitoring against a frozen baseline.
- [Ppk](#step-7--assess-capability): closer specification distance relative to overall variation.
- [Rational subgroup](#step-3--design-sampling-and-rational-subgroups): observations grouped for an explicit variation comparison.
- [Reaction plan](#step-6--monitor-phase-ii-and-react): assigned response to a signal.
- [Special cause](#3-the-core-mental-model): distinguishable source of unusual variation.
- [Specification](#step-1--define-the-decision-and-the-characteristic): authorized product requirement.
- [u chart](#step-4--choose-the-chart-family): chart of defects per unit of exposure.
- [Within sigma](#step-7--assess-capability): short-term variation estimate defined by the sampling design.

## 16. Sources

1. NIST/SEMATECH, *e-Handbook of Statistical Methods*, [Process or Product Monitoring and Control](https://www.itl.nist.gov/div898/handbook/pmc/pmc.htm).
   Public technical handbook covering control charts, variation, and capability.
2. NIST/SEMATECH, [What is Process Capability?](https://www.itl.nist.gov/div898/handbook/pmc/section1/pmc16.htm).
   Definitions and conditions for capability indices.
3. NIST/SEMATECH, [Individuals Control Charts](https://www.itl.nist.gov/div898/handbook/pmc/section3/pmc322.htm).
   Individuals and moving-range chart construction.
4. ISO, [ISO 7870-1:2019, Control charts — Part 1: General guidelines](https://www.iso.org/standard/70973.html).
   Standard overview; consult the licensed text and current edition for requirements.
5. ISO, *ISO 7870-2, Control charts — Part 2: Shewhart control charts*.
   Use the ISO catalogue to verify the current edition.
6. ISO, *ISO 22514-1, Statistical methods in process management — Capability and performance — Part 1: General principles and concepts*.
   Use the ISO catalogue to verify the current edition.
7. ISO, *ISO 22514-2, Statistical methods in process management — Capability and performance — Part 2: Process capability and performance of time-dependent process models*.
   Use the ISO catalogue to verify the current edition.
8. FDA, [Quality Management System Regulation](https://www.fda.gov/medical-devices/postmarket-requirements-devices/quality-management-system-regulation-qmsr).
   Official background and effective-date context for US medical-device requirements.
9. Electronic Code of Federal Regulations, [21 CFR Part 820](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-H/part-820).
   Current US regulatory text; check applicability and effective version.
10. International Automotive Task Force, [IATF 16949](https://www.iatfglobaloversight.org/iatf-16949/).
    Official overview of the automotive quality-management standard.
11. Western Electric Company, *Statistical Quality Control Handbook*, 1956.
    Historical source for supplementary chart interpretation rules.

## 17. Links

- [Module 15 topic index](_index.md)
- [Module 15 survey](../../../course/15_manufacturing_production_planning.md)
- [OEE and the loss tree](09_oee-loss-tree.md)
- [Bottlenecks and lean flow](10_bottlenecks-lean-toc.md)
- [Finite scheduling](06_finite-scheduling.md)
- [Production simulation and optimization labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md)
