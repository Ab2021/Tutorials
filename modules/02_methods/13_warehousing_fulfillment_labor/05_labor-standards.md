# Labour Standards and Engineered Standards

> Module 13 — Warehousing, Fulfillment, and Labor Analytics
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 13 topics 01–03
> Version 2026.5

## 1. Why this topic exists

A warehouse manager must decide how many paid hours to schedule before the orders arrive.
A labour standard translates a defined quantity and mix of work into an estimate of time.
Without a consistent method, a staffing plan can simply repeat yesterday’s overtime.
A target copied from a different building can be equally misleading.
The same number of order lines can require very different travel, handling and exception work.

The decision is not merely mathematical.
A standard can influence staffing, budgets, coaching and, in some organisations, incentives.
If it excludes ordinary delays or adequate recovery, it can turn a planning assumption into pressure to work unsafely.
If it credits work that was never observed, it can conceal a bottleneck and cause missed promises.
The standard must therefore be operationally useful and legitimate to the people doing the work.

This lesson develops an engineered standard from observable elements.
It distinguishes elapsed time, normal time, allowances and paid time.
It also explains how to study a job without treating a stopwatch as a substitute for worker knowledge.
A numerical case connects task estimates to shift staffing and a runnable calculation.
The case is fictional; its rates, costs and policies are teaching assumptions, not benchmarks.
The jurisdiction and employment agreement determine actual legal obligations.

> **Note:** A calculated standard is an estimate for a defined method and population.
> It is not a medical assessment, a legal safe-harbour, or a commitment that a particular worker can attain it.
> Health, safety, accommodation, pay and break obligations apply independently of a model.

## 2. Learning outcomes

After completing this topic, you can:

1. **Define** normal, standard, direct, indirect and paid time without mixing denominators.
2. **Decompose** a pick task into measurable elements and identify non-overlapping boundaries.
3. **Calculate** a standard using an explicit performance rating and allowance convention.
4. **Translate** a forecast work mix into paid staffing hours without double-counting time.
5. **Critique** an observation sample for selection, measurement and fairness bias.
6. **Design** an approval, appeal and change-control process with worker participation.
7. **Diagnose** a reported productivity gap using task mix, quality and system constraints.
8. **Interpret** applicable regulatory information without treating the calculation as legal advice.

Exercises 1–8 in section 12 map to these outcomes respectively.
Each exercise asks for a deliverable, a time budget and acceptance criteria.
The answer section provides arithmetic and rubrics for open-ended decisions.

## 3. The core mental model

An engineered standard is a conditional model of a method, not a universal speed limit.
Its output is valid only for work whose method and conditions resemble those studied.

```text
Business decision and worker consultation
                 |
                 v
Defined task + safe method + scope + observed conditions
                 |
                 v
Element observations -> normal-time estimate -> allowance convention
                 |                                  |
                 +----------- versioned standard ---+
                                  |
Workload forecast -> driver quantities -> earned standard hours
                                  |
                  indirect work + paid non-task time
                                  |
                    staffing plan and capacity check
                                  |
          quality, safety, service and worker feedback
                                  |
            review, appeal, correction or re-study
```

Follow the arrows in both directions.
A planning forecast predicts demand; it is not a guarantee of actual volume.
A staffing plan assigns capacity to the forecast; it is not an individual performance score.
A standard describes expected time under stated conditions; it is not itself a production commitment.
A management target is a chosen goal and must be labelled separately from those estimates.

The clock observes a particular worker performing a particular task in a particular setting.
The analyst classifies observed intervals, checks the record and documents interruptions.
An element average describes the observed sample, not every possible future task.
A rating, if used, is a human judgement about the observed pace relative to a defined reference.
The rating is contestable and should never erase unsafe conditions or force a pace.
Allowances recognise time not captured by normal productive element time.
The exact calculation depends on whether the allowance is a proportion of productive time or total time.
Applying both conventions to the same allowance overstates the result; omitting both understates it.

A task standard produces *earned* time when multiplied by valid completed work.
Earned time is a modelled quantity, not time recorded by a clock or compensation owed.
Compare it with actual time only when the scope and denominator match.
Paid time includes more than direct work; a shift may include training, meetings, breaks and downtime.
Keep those items visible in the staffing bridge instead of forcing all paid hours into a pick rate.

The loop is a governance requirement, not a decorative final step.
When slotting, equipment or product mix changes, a once-valid coefficient can become stale.
Workers can identify a missing safety check before a spreadsheet reveals the discrepancy.
Management should preserve a way to challenge a standard without retaliation.
The decision owner must be able to suspend use when evidence is inadequate.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Task | Bounded unit of work with a start and finish, such as one picked line | Treating a shift as one homogeneous task |
| Element | Repeatable component of a task, measured in seconds per occurrence | Overlapping two elements and counting time twice |
| Driver | Observable quantity that triggers an element, such as visits per line | Assuming every line has the same number of units |
| Observation | One recorded occurrence under stated conditions | Calling repeated observations independent when clustered |
| Observed time | Clock time for an element, seconds per occurrence | Treating it as a universal safe pace |
| Performance rating | Documented adjustment of observed pace to a reference pace, dimensionless | Presenting a subjective rating as objective fact |
| Normal time | Observed time multiplied by rating, seconds per occurrence | Calling it paid time |
| Allowance | Explicit provision for defined needs or interruptions | Hiding all waiting in an unexplained percentage |
| Productive-time allowance | Allowance expressed as a fraction of normal time | Dividing by one minus the percentage |
| Total-time allowance | Allowance expressed as a fraction of standard time | Multiplying normal time by one plus the percentage |
| Standard time | Normal time plus the defined allowance, seconds per driver | Treating it as a guaranteed individual achievement |
| Earned time | Completed driver quantity times its applicable standard, hours | Treating modelled hours as recorded paid hours |
| Direct time | Time assigned to producing or processing a defined unit | Excluding essential work because it does not move product |
| Indirect time | Necessary work not attributed to a direct driver, hours | Dropping cleaning and replenishment from staffing |
| Paid time | Compensable time under applicable law and policy, hours | Assuming it equals clocked task time |
| Utilisation | Defined share of available capacity used, percent | Using paid hours and direct hours interchangeably |
| Work sampling | Structured observations of activity states at selected instants | Claiming it yields detailed element durations |
| Time study | Repeated timing of defined work elements | Measuring only the fastest worker |
| Predetermined time system | Published or proprietary elemental motion model | Assuming its coefficients automatically fit the site |
| Delay | Time in a specified interruption category | Classifying normal replenishment waits as exceptional |
| Personal and fatigue allowance | Explicit provision for personal needs and recovery | Assuming a fixed percentage makes any task safe |
| Exception | Documented departure from standard conditions or method | Removing all slow observations as exceptions |
| Version | Identified effective set of method, coefficients and rules | Replacing a rate without preserving history |
| Exposure | Work conditions and duration relevant to risk | Inferring injury risk solely from productivity |
| Workload forecast | Estimated future driver quantities for a period | Calling forecast volume a customer commitment |
| Confidence interval | Range generated by a specified statistical procedure | Claiming it captures all design or bias errors |

## 5. The mechanics, step by step

### 5.1 Define the management decision

State whether the output will inform staffing, budget, process design or individual assessment.
A standard suitable for aggregate capacity planning is not automatically appropriate for individual evaluation.
List the decision owner, workers affected, sites, shifts, dates and planned uses.
Record prohibited or unapproved secondary uses before collecting data.
Invite worker representatives and safety specialists to review the study design.
A supervisor who can discipline the observed worker should not be the only avenue for concerns.

Input: a decision statement and a description of the operational process.
Output: a study charter with boundaries, stakeholders, uses and escalation routes.
Branch: if management cannot agree on the use and protections, do not deploy the standard.
A preliminary observation can still support a non-punitive process diagnosis if its purpose is disclosed.

### 5.2 Define the task and safe method

Write the start and finish event in terms a worker can recognise.
For a pick line, the start may be arrival at the location and the finish a confirmed pick into a container.
If walking between locations is included, define exactly where that travel begins and ends.
Specify equipment, travel path, scan sequence, item handling and required quality checks.
Include safe lifting, equipment inspection and normal security procedures.
Document conditions that change the work: height, weight, congestion, temperature and product accessibility.
Have workers demonstrate the actual method and explain variations before setting the stopwatch.
A method that requires bypassing a guard or skipping a quality check is not an acceptable baseline.

Input: process map, standard operating procedure and worker demonstrations.
Output: approved method and an inclusion/exclusion table.
Branch: if practice differs from the written method, resolve the discrepancy before assuming either is correct.
The study can document both methods while the safe process is being designed.

### 5.3 Choose work elements and drivers

Divide the process into elements that are observable and useful for forecasting.
Use a travel element for each location visit, a handling element for each unit and a transaction element for each line.
A single coefficient per order can hide differences in order size and travel distance.
An excessively fine model can become expensive to maintain and unreliable to measure.
Each element needs an unambiguous start, finish, driver and unit.
Adjacent element boundaries must meet without gaps or overlaps unless gaps are explicitly classified.
For irregular tasks, choose a separate driver such as exceptions per 100 lines or hours per shift.
Check that the count source and timing source refer to the same physical work.

Input: task map, order data and preliminary field observations.
Output: element dictionary and a reconciliation of task duration to element durations.
Branch: if one element changes materially with distance or weight, stratify it or add a driver.
Avoid adding a driver unless it can be measured consistently in live operations.

### 5.4 Design a representative study

Identify the population: shifts, workers, product profiles, zones, equipment and seasons.
Sample across ordinary conditions, including typical congestion and normal replenishment delays.
Document who was selected, who declined, and what selection constraints apply.
Protect voluntary participation and representation consistent with applicable obligations.
Repeat observations across workers, times and days rather than timing one unusually fast hour.
Clustered observations from one worker on one day are not equivalent to independent worker-days.
Choose a precision goal and a sample plan before looking at the result.
A pilot can estimate variation and inform the required sample size, but it does not cure bias.

Input: population roster or aggregate schedule, work mix and practical observation constraints.
Output: sampling plan, strata, observation protocol and privacy notice.
Branch: if rare heavy or awkward items matter, oversample them and weight results back to expected mix.
If a stratum cannot be observed, disclose the coverage gap and avoid claiming a complete standard.

### 5.5 Observe and reconcile

Record a start and finish time, element ID, task ID, worker pseudonym and context for each observation.
Use synchronised clocks and test whether device resolution is appropriate for short elements.
Record interruptions as events with reasons and durations instead of silently deleting them.
Check the total elapsed interval against the sum of elements and classified interruption time.
Discuss unclear observations with the participant while the events are still understandable.
Store source records securely and distinguish corrections from original observations.
Remove an observation only against a documented rule, with both included and excluded summaries retained.
Ordinary waiting is not automatically an outlier just because management wants it eliminated.

Input: observation sheets or instrument logs and a documented classification scheme.
Output: a reconciled dataset, exception register and data-quality assessment.
Branch: if the study changes behaviour substantially, report observation effects and repeat where appropriate.
If a safety concern appears, stop the unsafe work and address it; do not simply time through it.

### 5.6 Estimate normal element time

Calculate a mean or other documented estimator for each element and stratum.
Keep counts and units with every estimate; a mean without a sample size conceals uncertainty.
If performance rating is used, define the reference pace and train observers before data collection.
For the convention in this lesson, normal time equals observed time multiplied by rating.
A rating of 1.05 means the observer judged the observed pace 5% above the reference pace.
That observation’s 10.0 seconds becomes 10.5 normal seconds, not 9.5 seconds.
When ratings vary by observation, adjust each observation before computing the mean.
Compare results among observers and invite a structured review of contested ratings.
Rating is not required for every study: a transparent fixed-reference approach may be preferable.

Input: valid element observations and any pre-specified ratings.
Output: normal seconds per occurrence, variation and sample coverage.
Branch: if observer agreement is poor, revisit the protocol rather than averaging disagreement away.
Do not equate a slower observation with a deficient worker without investigating conditions.

### 5.7 Specify allowances and avoid overlap

List the needs or delays covered, the basis of each allowance and its evidence.
Separate personal needs and recovery from unavoidable operating delays where feasible.
Check legal and contractual requirements independently; an allowance percentage is not legal compliance.
Distinguish a percentage of normal productive time from a percentage of total standard time.
With a 15% productive-time allowance, 60 normal seconds become 69 standard seconds.
With a 15% total-time allowance, 60 normal seconds become 60 / 0.85 = 70.59 standard seconds.
The percentages describe different denominators and cannot be interchanged.
Identify break, delay and indirect-work categories already accounted for elsewhere.
Do not add a paid break as a separate staffing item if the same break is already funded by the standard.
A nominal allowance cannot substitute for task redesign where physical exposure is unsafe.

Input: normal times, exposure review, interruption evidence and allowance policy.
Output: standard times with an auditable calculation and coverage map.
Branch: if the allowance varies materially by task or conditions, use applicable strata rather than one global rate.
Where evidence is inadequate, retain a provisional label and avoid consequential individual use.

### 5.8 Build a work-content model

Multiply each element standard by its corresponding driver quantity.
For a single line with one visit and three units, count one visit, one line and three units.
For an order containing several lines, count the order-level element only once per order.
Add independent element times to obtain total standard seconds, then divide by 3,600 seconds per hour.
A future workload uses forecast driver quantities; actual earned hours use validated completed quantities.
Keep the two outputs in different fields so a forecast is not reported as completed work.
Reconcile totals to the workload population and check duplicate or cancelled tasks.

Input: versioned element rates and driver quantities at a compatible grain.
Output: forecast standard hours or earned standard hours, with period and scope.
Branch: if a driver quantity is missing, report an incomplete estimate and request correction.
Do not replace unknown quantities with zero without an explicit, justified rule.

### 5.9 Translate to a shift plan

Start with forecast standard hours for direct tasks.
Add necessary indirect work and any paid non-task time not already included in the standard.
Divide required paid hours by paid hours per worker-shift and round up for whole workers.
If worker-shift length varies, calculate availability by role, shift and eligibility rather than one average.
Check stations, inventory, equipment, arrival timing and downstream capacity separately.
More scheduled labour cannot overcome a blocked conveyor or unavailable stock.
Distinguish a nominal average staffing requirement from an intraday coverage schedule.
Review the plan with operations and workers; adjust it for known changes and document overrides.

Input: forecast hours, non-overlapping paid-time bridge and availability by shift.
Output: a staffing plan, assumptions and capacity constraints.
Branch: if the forecast exceeds feasible capacity, choose an explicit service, overtime or volume decision.
Do not label an unapproved overtime assumption a staffing commitment.

### 5.10 Validate, govern and maintain

Compare forecasts with actual paid hours at the same process and time grain.
Investigate mix, method, downtime, quality and forecast errors before changing a coefficient.
Review multiple periods; a one-day variance can reflect weather, releases or a system outage.
Use holdout data when available and compare aggregate bias as well as distribution of errors.
Check safety and quality signals alongside throughput; speed alone is not success.
Publish version, effective date, scope, owner, approved uses and review interval.
Provide a channel for workers to inspect relevant assumptions and request correction.
Record complaints, decisions, remedy and appeal outcomes with appropriate confidentiality.
When slotting, equipment or task design changes, assess whether a re-study is required.
Retire superseded versions for new work while preserving historical reproducibility.

Input: operating results, worker feedback, safety evidence and change events.
Output: approved, revised, suspended or retired standard with rationale.
Branch: if the method cannot be performed safely, suspend the affected use and correct the method.
If pay or discipline is affected, involve qualified labour, legal and employee representatives as applicable.
### 5.11 Field decisions that often change the result

Use this checklist while designing the study, not after a disputed target is announced.
Each item is a separate decision about measurement, staffing or worker protection.
#### Field decision 1: Arrival boundary

**Rule:** Mark arrival when the cart stops at the assigned location.

**Risk:** A scanner acknowledgement can lag physical arrival.

**Check:** Compare a sample of video or manual event times, subject to privacy rules.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 2: Location verification

**Rule:** Include the normal scan and visual verification before picking.

**Risk:** Misread labels may add time even when the worker follows the method.

**Check:** Record relabel work separately when it is not part of ordinary verification.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 3: Unit handling

**Rule:** Count each item transferred, including required secure placement.

**Risk:** A three-unit line is not equivalent to a one-unit line.

**Check:** Stratify fragile or heavy items when handling differs materially.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 4: Line confirmation

**Rule:** End after the required confirmation has completed.

**Risk:** Network lag can be ordinary operating time or a separate delay.

**Check:** Use a consistent event definition and reconcile with the delay ledger.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 5: Walking distance

**Rule:** Measure the actual accessible route rather than a straight line.

**Risk:** Blocked aisles and one-way rules change travel.

**Check:** Retain distance bands and observe representative routes.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 6: Cart setup

**Rule:** Decide whether setup occurs per order, batch or shift.

**Risk:** A per-line coefficient overcharges small orders and undercharges large ones.

**Check:** Use an order or batch driver if the count is available.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 7: Container change

**Rule:** Define when a full tote is exchanged and where it goes.

**Risk:** A mean per line hides variability in item volume.

**Check:** Use container changes as a driver when reliable records exist.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 8: Empty travel

**Rule:** Include return travel when the operating loop requires it.

**Risk:** A task may end away from the next assignment.

**Check:** Reconcile start and finish positions across consecutive tasks.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 9: Replenishment wait

**Rule:** Observe the time a picker lacks accessible stock.

**Risk:** Deleting every wait presents an unattainable method.

**Check:** Classify normal waits and separately flag exceptional outages.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 10: Short pick

**Rule:** Define the scan, investigation and escalation steps.

**Risk:** A missing unit changes both work content and service outcome.

**Check:** Model expected exception frequency and time outside the routine element.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 11: Damage handling

**Rule:** Include safe isolation and reporting for damaged items.

**Risk:** A standard that rewards ignoring damage can harm customers.

**Check:** Use quality checks and an exception driver without penalising reports.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 12: Device failure

**Rule:** Record the outage start, restoration and workaround.

**Risk:** A shift-wide incident can dominate one sample day.

**Check:** Keep it in actual paid time and report scenario assumptions separately.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 13: Congestion

**Rule:** Identify aisle, time window and queues at shared resources.

**Risk:** Capacity depends on concurrent work, not just individual speed.

**Check:** Compare low and peak congestion before adopting one coefficient.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 14: Working height

**Rule:** Record whether items are floor, waist or overhead level.

**Risk:** Averages hide different reach and lifting demands.

**Check:** Use safety review and relevant task strata, not a speed adjustment alone.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 15: Item weight

**Rule:** Record weight bands and required mechanical assistance.

**Risk:** A small SKU count may include a large share of handling risk.

**Check:** Validate the product master and inspect representative items.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 16: Training

**Rule:** Classify supervised learning and required refresher activity.

**Risk:** Treating all learning time as direct pick time distorts comparisons.

**Check:** Plan training hours separately and protect worker privacy.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 17: Shift handover

**Rule:** Record the scheduled handover and its actual duration.

**Risk:** Ignoring handover creates an impossible paid-hour budget.

**Check:** Include it once in indirect time or the paid-time bridge.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 18: Cleaning

**Rule:** Specify routine cleanup and sanitation required by the task.

**Risk:** A faster process may simply omit required cleaning.

**Check:** Model planned cleaning separately when it does not scale with lines.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 19: Safety inspection

**Rule:** Include pre-use equipment checks and defect reporting.

**Risk:** Counting only movement makes safety work invisible.

**Check:** Use a shift or equipment driver and preserve the inspection record.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 20: Quality rework

**Rule:** Measure rework and its cause as a separate flow.

**Risk:** Output can rise while errors and downstream labour rise faster.

**Check:** Pair throughput with error rates and rework hours.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 21: Paid breaks

**Rule:** List the paid breaks applicable to the schedule.

**Risk:** Adding the same paid break to both allowance and shift bridge double-counts it.

**Check:** Use one coverage ledger and validate policy with payroll.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 22: Accommodation

**Rule:** Document adjustments without recording unnecessary medical details.

**Risk:** A population standard cannot determine an individual accommodation.

**Check:** Use a confidential, legally reviewed process separate from routine dashboards.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 23: Temporary staff

**Rule:** Record process familiarity and assignment rules.

**Risk:** Pooling all workers can mask differing training conditions.

**Check:** Review aggregate strata without turning employment status into a proxy for ability.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 24: Observer calibration

**Rule:** Compare observers using the same pre-agreed example.

**Risk:** Inconsistent ratings create systematic differences between shifts.

**Check:** Resolve discrepancies before field use and retain the calibration record.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 25: Clock resolution

**Rule:** Use sufficient precision for the shortest measured element.

**Risk:** Whole-second rounding can dominate a two-second element.

**Check:** Test device resolution and aggregate where precision is inadequate.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 26: Selection of days

**Rule:** Include representative release patterns and business days.

**Risk:** A quiet Tuesday may not represent a promotion weekend.

**Check:** Weight the sampled strata to the intended forecast period.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 27: Sample size

**Rule:** Specify the decision precision and practical sampling unit.

**Risk:** Hundreds of lines from one person are not hundreds of independent worker-days.

**Check:** Estimate uncertainty with worker or day clustering when appropriate.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 28: Outlier treatment

**Rule:** Write exclusion criteria before reviewing slow values.

**Risk:** Removing all long times systematically removes real work.

**Check:** Report sensitivity with and without justified exclusions.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 29: Process change

**Rule:** Log effective date and the affected method or location.

**Risk:** Comparing pre-change actuals to a post-change standard confounds the analysis.

**Check:** Use effective-dated versions and re-study material changes.

**Record:** Keep the decision and its effective date with the study version.
#### Field decision 30: Appeal

**Rule:** Provide a route to contest a task classification or standard.

**Risk:** A dashboard correction is not useful if the underlying record stays wrong.

**Check:** Record the decision, correction and any remedy in a controlled workflow.

**Record:** Keep the decision and its effective date with the study version.
## 6. Worked example

### 6.1 The operating decision

A fictional regional distribution centre plans one eight-hour paid shift for a small-item pick zone.
The planning period is one weekday shift in November 2026; all money is in November 2026 US dollars.
The manager needs a staffing estimate, not an individual performance target.
The order forecast is 4,800 completed lines, with an expected 7,200 units and 4,800 location visits.
The line count is a forecast, not a customer commitment or a guaranteed release.
A location visit corresponds to one line in this particular forecast; other processes may differ.
The paid shift is eight hours per scheduled worker, including a paid break described below.

The study covers a scanner-assisted, cart-based process in a single temperature-controlled zone.
Workers helped define the safe method and verified that location scans and handling checks were included.
The example is deliberately small enough to calculate by hand; its observations do not establish a production-ready rate.
A real deployment requires a larger, representative study across workers, shifts and conditions.

### 6.2 Study observations

The timing convention separates three non-overlapping elements.
Travel is measured once per location visit, verification once per line and handling once per unit.
The table gives illustrative mean observed times and a common illustrative rating for each element.

| Element | Driver | Observed seconds per driver | Rating | Normal seconds per driver |
|---|---|---:|---:|---:|
| Travel and position cart | visit | 18.0 | 1.05 | 18.9 |
| Verify location and confirm line | line | 12.0 | 1.00 | 12.0 |
| Pick and place | unit | 8.0 | 1.00 | 8.0 |

The travel calculation is 18.0 seconds × 1.05 = 18.9 normal seconds per visit.
The rating is a judgement in this fictional example, not a statement that a worker should walk faster.
The remaining elements have a neutral rating of 1.00.
The normal-time estimate requires the method and sample caveats already stated.

### 6.3 Allowance and coverage

For this calculation the allowance is 15% of normal time, not 15% of total standard time.
It covers the agreed personal/recovery and ordinary minor interruption categories in the hypothetical study.
The allowance does not cover the separately scheduled paid break, training or shift handover.
This is a teaching assumption, not a recommendation that 15% is sufficient or legally compliant.
The coverage ledger prevents the same paid interval from being added twice.

| Category | Treatment in this case | Boundary |
|---|---|---|
| Normal pick elements | Included in element normal time | Visit, line and unit |
| Defined recovery and minor interruption | 15% of normal time | Productive-time allowance |
| Paid break | Separate paid-time bridge | 0.5 paid hour per worker-shift |
| Handover and equipment checks | Separate indirect hours | 4.0 hours per shift |
| Training | Separate indirect hours | 2.0 hours per shift |
| Major outage | Scenario, not routine standard | Not assumed in base plan |

Standard travel time = 18.9 × 1.15 = 21.735 seconds per visit.
Standard verification time = 12.0 × 1.15 = 13.8 seconds per line.
Standard handling time = 8.0 × 1.15 = 9.2 seconds per unit.
The standard for a line with 1.5 units on average is 21.735 + 13.8 + 1.5 × 9.2.
That is 49.335 standard seconds per forecast line at this mix.
This is not the standard for every individual line; actual unit count changes the work content.

### 6.4 Forecast direct work

| Driver | Forecast quantity per shift | Standard seconds per driver | Standard seconds per shift |
|---|---:|---:|---:|
| Visit | 4,800 visits | 21.735 | 104,328 |
| Line | 4,800 lines | 13.8 | 66,240 |
| Unit | 7,200 units | 9.2 | 66,240 |
| Total | Mixed drivers | Not additive as a rate | 236,808 |

236,808 standard seconds / 3,600 seconds per hour = 65.78 standard hours per shift.
The total is a forecast of standard work content, not actual hours already worked.
The visit count and line count happen to be equal; they refer to distinct, non-overlapping elements.
If travel were already included in the line confirmation time, this calculation would double-count it.

### 6.5 Paid staffing bridge

Add 4.0 indirect hours for handover and checks and 2.0 indirect hours for training.
Required non-break work is therefore 65.78 + 4.0 + 2.0 = 71.78 hours per shift.
Each scheduled worker has 8.0 paid hours minus 0.5 paid break hour = 7.5 hours for that work.
71.78 / 7.5 = 9.5707 worker-shifts, rounded up to 10 whole worker-shifts.
Ten workers supply 80.0 paid hours, of which 5.0 paid hours are the scheduled breaks.
The remaining 75.0 hours cover 71.78 forecast hours, leaving 3.22 hours of nominal headroom.
That headroom is not guaranteed spare capacity: release timing, eligibility and downtime still matter.
The headcount rounding does not imply that overtime is required or approved.
At an assumed fully loaded USD 25 per paid hour, the planned shift cost is 80 × 25 = USD 2,000.
The rate is illustrative and includes all labour cost components only by assumption.

### 6.6 Sensitivity and management interpretation

Suppose the line and visit forecasts stay at 4,800 while units rise from 7,200 to 9,600.
The incremental 2,400 units add 2,400 × 9.2 / 3,600 = 6.1333 standard hours.
The direct forecast becomes 71.9133 hours; adding 6.0 indirect hours gives 77.9133 hours.
77.9133 / 7.5 = 10.3884 worker-shifts, so the base assumptions now imply 11 workers.
The line volume did not change, but the handling mix did.
A lines-per-hour target alone would miss this staffing change.

A different scenario changes only the paid break to 0.75 hour per worker-shift.
At the original mix, available non-break time is 7.25 hours per worker.
71.78 / 7.25 = 9.9007 worker-shifts, still rounded to 10 workers.
This is a calculation illustration, not a recommendation to change break policy.
Before adopting any schedule, check the applicable rules, agreement and actual shift coverage.

If a conveyor outage consumes 8.0 additional paid hours, the base work requires 79.78 non-break hours.
79.78 / 7.5 = 10.6373 worker-shifts, implying 11 workers under this simplified scenario.
Hiring an eleventh worker after the outage cannot recover time already lost; timing and available work matter.
The model is a shift total, not a queueing simulation or an intraday roster.
Report forecast, plan and realised outcomes separately at the end of the shift.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It computes the base shift and the higher-unit sensitivity from section 6.
The allowance is applied to normal time, and the paid break is outside the allowance.
The program rejects negative quantities and an impossible paid-break duration.
It is a teaching calculator, not a production payroll or safety system.

```python
from math import ceil


def calculate_shift(visits, lines, units, indirect_hours, paid_hours,
                    paid_break_hours, hourly_cost):
    quantities = (visits, lines, units, indirect_hours, hourly_cost)
    if any(value < 0 for value in quantities):
        raise ValueError("quantities and cost must be non-negative")
    if paid_hours <= 0 or not 0 <= paid_break_hours < paid_hours:
        raise ValueError("paid hours must exceed non-negative paid break")

    # Observed seconds, performance rating, and productive-time allowance.
    allowance = 0.15
    elements = (
        (visits, 18.0, 1.05),
        (lines, 12.0, 1.00),
        (units, 8.0, 1.00),
    )
    standard_seconds = sum(
        count * observed_seconds * rating * (1 + allowance)
        for count, observed_seconds, rating in elements
    )
    direct_hours = standard_seconds / 3600
    work_hours = direct_hours + indirect_hours
    available_hours = paid_hours - paid_break_hours
    workers = ceil(work_hours / available_hours)
    paid_total = workers * paid_hours
    return {
        "direct_hours": direct_hours,
        "work_hours": work_hours,
        "workers": workers,
        "paid_hours": paid_total,
        "cost_usd": paid_total * hourly_cost,
        "headroom_hours": workers * available_hours - work_hours,
    }


def show(label, result):
    print(label)
    print(f"  direct standard hours: {result['direct_hours']:.2f}")
    print(f"  non-break work hours: {result['work_hours']:.2f}")
    print(f"  workers: {result['workers']}")
    print(f"  paid hours: {result['paid_hours']:.2f}")
    print(f"  cost USD: {result['cost_usd']:.2f}")
    print(f"  nominal headroom hours: {result['headroom_hours']:.2f}")


base = calculate_shift(4800, 4800, 7200, 6, 8, 0.5, 25)
high_units = calculate_shift(4800, 4800, 9600, 6, 8, 0.5, 25)
show("Base", base)
show("Higher units", high_units)
```

Expected output, rounded for display:

```text
Base
  direct standard hours: 65.78
  non-break work hours: 71.78
  workers: 10
  paid hours: 80.00
  cost USD: 2000.00
  nominal headroom hours: 3.22
Higher units
  direct standard hours: 71.91
  non-break work hours: 77.91
  workers: 11
  paid hours: 88.00
  cost USD: 2200.00
  nominal headroom hours: 4.59
```

The higher-unit case has more rounded headroom because staffing is a whole-worker decision.
Do not interpret the headroom as an invitation to remove recovery or required indirect work.
The code does not model worker qualifications, arrival times, breaks by window or station constraints.
It assumes every worker has the same paid shift and break and is eligible for the same tasks.
A production implementation must preserve rate versions and reconcile input quantities to completed tasks.
Floating-point rounding affects displayed values only; no element is rounded before aggregation.

## 8. Data, standards, and regulatory context

### 8.1 Data contracts

The operational system should identify task, order, line, unit quantity, location and timestamps.
A warehouse management system (WMS) may record scans rather than physical start and finish events.
A scan is evidence of a system event, not necessarily a complete measurement of all work.
Keep the task completion state and reversal history so cancelled work is not credited twice.
Reconcile totals against payroll and scheduling at an appropriate aggregate grain.
Payroll hours and task records can differ legitimately because task records omit paid non-task work.
The reconciliation should explain categories rather than force the totals to match by assumption.

A usable standard record includes task and method identifiers, element rates and units.
It also includes study dates, sample coverage, allowance basis, version, owner and approval.
Retain effective dates so historical earned hours can be reproduced using the applicable version.
Restrict person-level observations and record who can access, amend or export them.
Apply retention rules and consultation requirements appropriate to the organisation and jurisdiction.

### 8.2 United States context

The US Occupational Safety and Health Act imposes workplace safety obligations; its general duty clause
requires covered employers to furnish employment and a workplace free from recognised hazards likely
to cause death or serious physical harm, subject to the statute and its applicability.
Occupational Safety and Health Administration (OSHA) warehouse guidance discusses relevant hazards.
An engineered time allowance does not by itself establish compliance with those obligations.
The Fair Labor Standards Act and Department of Labor guidance address compensable hours and overtime
for covered employees; coverage, exemptions and specific facts matter.
Do not infer pay owed from earned standard hours: use actual compensable time under applicable rules.
Federal law is not the only source of obligations; state law, local law and agreements may add requirements.

The Americans with Disabilities Act may require reasonable accommodation for qualified individuals,
subject to statutory conditions and exceptions; a standard does not decide accommodation eligibility.
The Equal Employment Opportunity Commission provides employer guidance on the interactive process.
Keep medical information confidential and seek qualified advice for an actual case.
Do not use a group average or a single pace rating to deny an individual assessment.

As an example of a specific state framework, California Labor Code sections 2100–2107 address
warehouse distribution centre quotas and related disclosures and protections for covered employers and employees.
The statute has definitions, thresholds and exceptions; not every warehouse or metric is covered.
Consult the current law and the California Labor Commissioner’s guidance before applying it.
A quota that interferes with required breaks or health and safety is not made lawful by calling it a standard.
The cited sources should be checked for changes and interpreted with counsel as of the decision date.

### 8.3 Standards and conventions

ISO 6385 describes ergonomic principles in the design of work systems.
It is a design standard, not a numerical picking-rate table and not automatically legislation everywhere.
The National Institute for Occupational Safety and Health (NIOSH) lifting equation supports assessment
of specified two-handed manual lifting tasks; its assumptions and limits must be checked.
Neither a lifting equation nor an allowance percentage captures all injury risks or all work conditions.
Use qualified ergonomics assessment and worker input to evaluate the actual job.

The rating convention and allowance formulas in this lesson are explicitly declared analytical conventions.
Different industrial engineering systems may define ratings or allowances differently.
When importing a vendor standard, request the underlying method, denominator, exclusions and validation evidence.
Do not combine published percentages until their definitions and overlap are understood.

## 9. Industry practice and case evidence

### 9.1 Public evidence versus fictional calculation

The centre and numbers in section 6 are fictional and are not represented as an industry benchmark.
Public cases demonstrate why the governance questions matter, not that the fictional coefficients are accurate.
Evidence of a regulatory finding or a company position should not be generalised to every site.

### 9.2 OSHA and warehouse ergonomics

OSHA announced in January 2023 that it cited Amazon for exposing workers to ergonomic hazards at three warehouses.
The announcement describes the agency’s inspection findings and proposed penalties at the time.
A citation is an agency allegation and its later procedural status may differ; check the case record before relying on it.
It does not establish a causal effect for every productivity measure or a universal safe rate.
**Inference:** A labour-standard governance process should review ergonomic exposure and worker reports
alongside throughput, rather than assuming that a calculated allowance resolves safety concerns.

### 9.3 California quota disclosures

California’s warehouse quota statute provides a public example of governance applied to certain distribution centres.
The law addresses written quota descriptions and prohibits specified adverse action in relation to quotas
that do not allow compliance with applicable meal, rest, bathroom or occupational health and safety rules.
The exact statutory conditions, remedies and current guidance matter to any real application.
**Inference:** Versioned descriptions, access to relevant information and an appeal route make a standard
more auditable even in settings where this particular statute does not apply.
The inference is a management design recommendation, not a claim that every element is legally mandatory everywhere.

### 9.4 How practitioners should use the evidence

Ask whether the documented case shares the site’s work, geography and employment arrangements.
Distinguish an enforcement allegation, an adjudicated finding and a company’s response.
Do not use a headline injury rate as a coefficient in a staffing model.
Do use the case to construct questions for safety, legal and worker representatives.
For example, ask whether a productivity dashboard makes reporting a hazard appear as lost performance.
Ask whether the model includes required equipment checks and safe handling.
Then observe the local operation and document the answer rather than assuming transferability.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Fast-worker sample | One worker supplies most observations | Selection bias | Unattainable rate | Representative multi-worker sample |
| Quiet-shift sample | Peak hours absent | Congestion omitted | Understaffing at peak | Stratify by release and shift |
| Overlapping elements | Element sums exceed task clock | Boundaries overlap | Inflated earned hours | Reconcile start and finish events |
| Missing travel | Task starts only at scan | Work outside system timestamps | Unrealistic staffing | Observe full physical cycle |
| Deleted delays | Most long times labelled abnormal | Normal waits removed | Hidden system losses | Retain delay ledger and inclusion rules |
| Wrong allowance basis | Formula lacks denominator | Productive and total time confused | Incorrect standard | Declare formula and units |
| Double-counted break | Break appears in two bridge rows | Allowance overlaps paid-time adjustment | Overstated staffing | Category coverage ledger |
| Missing break | Paid hours treated as all direct work | Break absent from model | Understaffing and pressure | Explicit paid-time bridge |
| Stale rate | Slotting changed after study | Method and distance drift | Forecast bias | Change triggers and re-study |
| Mix distortion | Units per line rising | Single line coefficient hides handling | Wrong headcount | Model visits, lines and units |
| Unsafe shortcut | Quality or hazard reports fall with speed | Incentive discourages safe method | Injury or defects | Independent safety and quality review |
| False individual ranking | Aggregate rate used on all tasks | Worker assignments differ | Unfair decisions | Matched work and due process |
| Missing indirect work | Meetings absent from plan | Non-unit tasks omitted | Overtime and missed work | Add non-overlapping shift drivers |
| Missing outage | Actual and standard diverge during incident | System availability ignored | Blame shifted to workers | Log events and compare matched periods |
| Non-reproducible history | Rates overwritten in place | No effective-dated version | Unexplainable outcomes | Preserve immutable versions |
| Suppressed reports | Workers fear penalty for stopping | Feedback channel is not trusted | Unseen hazards | Confidential escalation and anti-retaliation controls |
| Invalid clock | Negative or impossible durations | Clock drift or event ordering | Corrupt rates | Synchronise and validate timestamps |
| Forecast treated as actual | Planned lines shown as completed | Data states conflated | False productivity | Separate forecast and actual fields |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Method is stable | Coefficients describe one method | New scanner or changed route | Compare current method with study sheet |
| Element boundaries do not overlap | Addition must be valid | Travel included twice | Reconcile to full-cycle timing |
| Driver counts are accurate | Counts scale work content | Duplicates or cancellations | Reconcile task and order records |
| Sample covers intended work | Estimates must transfer | Missing shift or product group | Compare sample and operational mix |
| Rating is consistently applied | Normal time depends on judgement | Observer disagreement | Calibration and sensitivity analysis |
| Allowance covers specified categories | Standard time must include its scope | Undefined or overlapping categories | Review coverage ledger with workers |
| Paid break is separate | Staffing bridge subtracts it | Same interval funded in allowance | Compare policy and rate documentation |
| Indirect hours are forecast | Direct work is not the whole shift | Training or checks omitted | Reconcile schedules to time categories |
| Work is available throughout shift | Total-hour division assumes usable capacity | Late release or upstream blockage | Inspect arrival and completion timestamps |
| Workers are interchangeable for plan | Whole-worker calculation uses one availability | Different qualifications or restrictions | Build role-by-shift coverage |
| No major outage in base case | Base workload excludes outage | System or equipment failure | Compare downtime logs; run scenarios |
| Safe method remains feasible | Rate assumes compliant task execution | Workload or design creates exposure | Ergonomic review and worker feedback |
| Cost rate includes selected costs | Shift estimate multiplies paid hours | Premiums or benefits omitted | Reconcile finance definition |
| Allowance is adequate for studied conditions | Estimate assumes recovery needs met | Heat, load or duration change | Reassess exposure and worker reports |

These assumptions are conditional, not claims that all warehouses meet them.
A statistical interval around an element mean does not capture omitted breaks, biased selection or unsafe design.
Report uncertainty and structural gaps separately.
When a material assumption fails, restrict the standard’s use until it is corrected or replaced.
## 12. Exercises

Work independently before consulting section 13.
All figures are illustrative and no exercise supplies legal advice.
### Exercise 1: Recall and denominators

**Scenario and task:** Explain normal time, standard time and earned time.

**Deliverable:** A concise written response with any arithmetic and assumptions shown.

**Acceptance criteria:** Use one sentence and one unit for each; distinguish earned from paid hours.

**Suggested time:** 8 minutes.

State the time basis and scope whenever a quantity is calculated.
If a fact is not supplied, label your assumption rather than inventing evidence.
### Exercise 2: Allowance calculation

**Scenario and task:** A measured element takes 20 seconds at rating 1.10. Apply a 12% allowance on normal time.

**Deliverable:** A concise written response with any arithmetic and assumptions shown.

**Acceptance criteria:** Show normal seconds, standard seconds and the formula; explain the allowance basis.

**Suggested time:** 10 minutes.

State the time basis and scope whenever a quantity is calculated.
If a fact is not supplied, label your assumption rather than inventing evidence.
### Exercise 3: Element model

**Scenario and task:** A shift has 1,000 visits at 20 seconds each and 1,500 units at 6 seconds each.

**Deliverable:** A concise written response with any arithmetic and assumptions shown.

**Acceptance criteria:** Calculate total standard hours without adding an unrelated line term.

**Suggested time:** 12 minutes.

State the time basis and scope whenever a quantity is calculated.
If a fact is not supplied, label your assumption rather than inventing evidence.
### Exercise 4: Paid staffing

**Scenario and task:** For 25 work hours, an eight-hour paid shift includes a 0.5-hour paid break outside the standard.

**Deliverable:** A concise written response with any arithmetic and assumptions shown.

**Acceptance criteria:** Calculate whole workers and paid hours; explain why rounding is required.

**Suggested time:** 10 minutes.

State the time basis and scope whenever a quantity is calculated.
If a fact is not supplied, label your assumption rather than inventing evidence.
### Exercise 5: Sampling critique

**Scenario and task:** A manager times one experienced worker for 40 minutes on a quiet shift and removes every delay.

**Deliverable:** A concise written response with any arithmetic and assumptions shown.

**Acceptance criteria:** Identify four distinct biases and propose a repair for each.

**Suggested time:** 15 minutes.

State the time basis and scope whenever a quantity is calculated.
If a fact is not supplied, label your assumption rather than inventing evidence.
### Exercise 6: Governance design

**Scenario and task:** Design a standard approval and appeal workflow for a newly automated pick zone.

**Deliverable:** A concise written response with any arithmetic and assumptions shown.

**Acceptance criteria:** Name owners, worker participation, effective dates, suspension triggers and remedy.

**Suggested time:** 20 minutes.

State the time basis and scope whenever a quantity is calculated.
If a fact is not supplied, label your assumption rather than inventing evidence.
### Exercise 7: Variance diagnosis

**Scenario and task:** Actual paid hours rise while lines stay flat and units per line increase.

**Deliverable:** A concise written response with any arithmetic and assumptions shown.

**Acceptance criteria:** Give three plausible mechanisms, evidence for each and one incorrect causal conclusion to avoid.

**Suggested time:** 15 minutes.

State the time basis and scope whenever a quantity is calculated.
If a fact is not supplied, label your assumption rather than inventing evidence.
### Exercise 8: Regulatory interpretation

**Scenario and task:** A supervisor claims a 15% allowance proves every quota is lawful and safe.

**Deliverable:** A concise written response with any arithmetic and assumptions shown.

**Acceptance criteria:** Refute the claim using two distinct regulatory or safety considerations and name a source to consult.

**Suggested time:** 12 minutes.

State the time basis and scope whenever a quantity is calculated.
If a fact is not supplied, label your assumption rather than inventing evidence.
## 13. Self-check answers

### Answer 1: Denominators

Normal time is observed element time adjusted to the stated reference pace, in seconds per occurrence.
Standard time adds the defined allowance using its stated denominator, also in seconds per occurrence.
Earned time is completed driver count multiplied by applicable standard time, converted to hours.
Paid hours are actual compensable time and need not equal modelled earned hours.

### Answer 2: Allowance

Normal time = 20 seconds × 1.10 = 22 seconds per occurrence.
Standard time = 22 seconds × 1.12 = 24.64 seconds per occurrence.
The 12% is a fraction of normal time; dividing by 0.88 would use a different convention.
The arithmetic does not establish that the rating or allowance is appropriate.

### Answer 3: Element model

Visits require 1,000 × 20 = 20,000 standard seconds.
Units require 1,500 × 6 = 9,000 standard seconds.
The total is 29,000 seconds / 3,600 = 8.0556 standard hours.
The elements must be non-overlapping; no separate line rate was supplied.

### Answer 4: Paid staffing

Each worker supplies 8.0 − 0.5 = 7.5 non-break hours.
25 / 7.5 = 3.3333 worker-shifts, so round up to four whole workers.
Four workers receive 4 × 8 = 32 paid hours in this simplified shift plan.
The plan has 30 non-break hours and 5 hours of nominal headroom.
Availability timing and eligibility still need checking.

### Answer 5: Sampling critique rubric

A strong answer identifies worker selection, time-of-day selection, limited duration and delay exclusion separately.
It proposes sampling across workers, shifts and days; recording ordinary delays; and retaining exclusion evidence.
It also notes clustering and the need for worker involvement and safe-method observation.
A weak answer only says to collect more observations from the same worker in the same quiet period.

### Answer 6: Governance design rubric

A strong answer names operations, industrial engineering, safety, workers and the accountable approver.
It defines study purpose, data access, notice, effective version, permissible use and review interval.
It supplies a confidential challenge channel, investigation deadline, correction and remedy.
It suspends affected use when the method becomes unsafe or evidence no longer supports the standard.
A weak answer gives a supervisor unilateral power to alter a rate without notice or appeal.

### Answer 7: Variance diagnosis rubric

Higher units per line can add handling time; test unit counts and unit-level rates.
A changed travel mix can add distance; test location visits and route measurements.
Outages or indirect work can increase paid hours; test incident and time-category records.
Training, quality rework and release timing are also plausible and require evidence.
Do not conclude that workers slowed down merely from a flat line count and higher paid hours.

### Answer 8: Regulatory interpretation rubric

An allowance is a modelling convention, not proof of compliance with wage, break or safety rules.
Actual compensable hours must be assessed independently of earned hours under applicable law.
Ergonomic hazards and accommodation obligations require assessment of actual work and individual facts.
Consult current OSHA, Department of Labor, EEOC and relevant state guidance with qualified counsel.
A strong answer distinguishes binding law from guidance and avoids asserting universal coverage.

## 14. Interview questions
### Question 1

**Question:** Why can lines per paid hour mislead?

**Strong answer:** Explain mix, indirect work, break coverage and denominator consistency.

**Weak answer:** Treat every line and every paid hour as identical.

**Follow-up:** Ask the candidate to state one observation that would change their conclusion.
### Question 2

**Question:** What is the difference between observed and normal time?

**Strong answer:** Show the declared rating convention and its judgemental nature.

**Weak answer:** Assume the raw clock time is automatically a universal standard.

**Follow-up:** Ask the candidate to state one observation that would change their conclusion.
### Question 3

**Question:** How do allowance denominators change the calculation?

**Strong answer:** Contrast multiplying by 1+a and dividing by 1-a with their bases.

**Weak answer:** Use the formulas interchangeably.

**Follow-up:** Ask the candidate to state one observation that would change their conclusion.
### Question 4

**Question:** What makes a sample representative?

**Strong answer:** Cover workers, shifts, conditions, clustering and intended population.

**Weak answer:** Use the largest count of observations from one worker.

**Follow-up:** Ask the candidate to state one observation that would change their conclusion.
### Question 5

**Question:** How do you handle an interruption?

**Strong answer:** Classify it, keep evidence, and define ordinary versus exceptional treatment.

**Weak answer:** Delete all observations slower than the target.

**Follow-up:** Ask the candidate to state one observation that would change their conclusion.
### Question 6

**Question:** How do you translate standard hours to headcount?

**Strong answer:** Add non-overlapping indirect work, subtract paid breaks from availability and round.

**Weak answer:** Divide direct hours by eight without checking the bridge.

**Follow-up:** Ask the candidate to state one observation that would change their conclusion.
### Question 7

**Question:** When should a standard be re-studied?

**Strong answer:** Identify changes in method, mix, equipment, safety and systematic forecast bias.

**Weak answer:** Change it every time a worker misses a target.

**Follow-up:** Ask the candidate to state one observation that would change their conclusion.
### Question 8

**Question:** How do you prevent double counting?

**Strong answer:** Reconcile element boundaries and allowance coverage to paid-time categories.

**Weak answer:** Assume all time categories are independent by name.

**Follow-up:** Ask the candidate to state one observation that would change their conclusion.
### Question 9

**Question:** What protects workers from misuse?

**Strong answer:** Describe consultation, access, independent review, appeal and non-retaliation.

**Weak answer:** Promise the model is objective because it uses timestamps.

**Follow-up:** Ask the candidate to state one observation that would change their conclusion.
### Question 10

**Question:** Does an allowance make a task safe or lawful?

**Strong answer:** Explain separate ergonomic, wage, break and accommodation obligations.

**Weak answer:** Treat a percentage as a legal safe harbour.

**Follow-up:** Ask the candidate to state one observation that would change their conclusion.
## 15. Key terms

Use the definitions in section 4 and the calculation in sections 5–7.

- **Allowance:** Provision for defined needs or interruptions; see section 5.7.
- **Appeal:** Process for contesting a standard or its application; see section 5.10.
- **Clock time:** Recorded elapsed time; see section 4.
- **Direct time:** Time attributed to a defined production task; see section 4.
- **Driver:** Count or quantity that scales an element; see section 5.3.
- **Earned time:** Completed work multiplied by standard time; see section 5.8.
- **Element:** Bounded repeatable component of work; see section 5.3.
- **Engineered standard:** Conditional work-time estimate built from defined elements; see section 3.
- **Exception:** Documented departure from the defined method; see section 5.5.
- **Forecast:** Estimated future volume, not completed work; see section 6.1.
- **Indirect time:** Necessary work not assigned to a direct driver; see section 5.9.
- **Normal time:** Observed time adjusted by a declared rating; see section 5.6.
- **Observation:** Recorded occurrence of an element; see section 5.5.
- **Paid time:** Compensable time under applicable rules; see section 8.2.
- **Performance rating:** Judgement relating observed pace to a reference; see section 5.6.
- **Productive-time allowance:** Allowance expressed relative to normal time; see section 5.7.
- **Sample coverage:** Extent to which observations reflect the intended population; see section 5.4.
- **Standard time:** Normal time plus allowance on a declared basis; see section 5.7.
- **Task:** Bounded unit of work; see section 5.2.
- **Time study:** Repeated timing of defined elements; see section 5.5.
- **Total-time allowance:** Allowance expressed relative to standard time; see section 5.7.
- **Version:** Effective-dated set of standard assumptions and coefficients; see section 5.10.
- **Work sampling:** Structured observations of activity states; see section 4.
- **Workload mix:** Distribution of driver quantities and conditions; see section 6.6.

## 16. Sources

Sources are provided for independent checking; legal and regulatory material can change.
The numerical case is fictional and is not sourced from the organisations below.

1. US Occupational Safety and Health Act, section 5(a)(1), general duties:
   [29 USC 654](https://uscode.house.gov/view.xhtml?req=granuleid:USC-prelim-title29-section654&num=0&edition=prelim).
2. OSHA, [Warehousing: Overview](https://www.osha.gov/warehousing).
3. US Department of Labor, Wage and Hour Division,
   [Hours Worked under the Fair Labor Standards Act](https://www.dol.gov/agencies/whd/fact-sheets/22-flsa-hours-worked).
4. US Equal Employment Opportunity Commission,
   [Enforcement Guidance on Reasonable Accommodation and Undue Hardship under the ADA](https://www.eeoc.gov/laws/guidance/enforcement-guidance-reasonable-accommodation-and-undue-hardship-under-ada).
5. California Legislative Information, [Labor Code sections 2100–2107](https://leginfo.legislature.ca.gov/faces/codes_displayText.xhtml?lawCode=LAB&division=2.&title=&part=7.&chapter=11.&article=).
6. California Department of Industrial Relations,
   [Warehouse Quotas](https://www.dir.ca.gov/dlse/warehouse-quotas.html).
7. ISO, [ISO 6385:2016, Ergonomics principles in the design of work systems](https://www.iso.org/standard/63785.html).
8. CDC/NIOSH, [Revised NIOSH Lifting Equation](https://www.cdc.gov/niosh/ergonomics/about/rnle.html).
9. OSHA, 18 January 2023, [US Department of Labor cites Amazon for exposing workers to ergonomic hazards](https://www.osha.gov/news/newsreleases/national/01182023).

## 17. Links

- [Module 13 index](_index.md).
- [Warehouse flow system](01_warehouse-flow-system.md).
- [WMS data model](02_wms-data-model.md).
- [Productivity metrics](03_productivity-metrics.md).
- [Module 13 survey and its lab sections](../../../course/13_warehousing_fulfillment_labor.md).
