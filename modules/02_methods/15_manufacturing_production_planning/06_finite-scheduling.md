# Finite Scheduling: Single Machines, Flow Shops, and Job Shops

> Module 15 — Manufacturing and Production Planning  
> Topic 06 of 11  
> Estimated study time: 4 hours  
> Prerequisites: [manufacturing master data](02_manufacturing-master-data.md), [MRP explosion](04_mrp-explosion.md), [capacity requirements](05_capacity-requirements.md)  
> Version 2026.5

## 1. Why this topic exists

A weekly capacity report can show enough machine-hours and still conceal an impossible Tuesday.
Jobs may compete for the same machine at the same instant.
An operator, component, fixture, inspection, or preceding operation may not be available when the report assumes.
A finite schedule assigns specific operations to specific time intervals without exceeding the modeled resource supply.
It turns an aggregate capacity discussion into a concrete sequence and a list of exceptions.

Consider a plant with a machining cell and a heat-treatment oven.
A customer needs a pump housing by the end of the day.
The housing cannot enter the oven until machining and inspection finish.
Meanwhile another order is already using the oven.
A material requirements planning (MRP) due date does not resolve that contention.
The scheduler needs to decide which job runs first, which commitment is exposed, and who can authorize a change.

The production supervisor needs a workable first dispatch, not a list of mathematically attractive but unstaffed start times.
The material planner needs a signal when a missing part invalidates a proposed start.
Customer service needs a revised promise before a missed delivery becomes a surprise.
Quality and maintenance need protected time that a utilization chart may otherwise hide.
The plant manager needs to compare service, work-in-process inventory, overtime, changeovers, and disruption.

A forecast is an estimate of future demand.
A plan is the intended allocation of work.
A target is the desired result, such as 95% on-time delivery.
A commitment is an authorized obligation to a customer or internal recipient.
A schedule is a time-specific plan; it does not itself create or change a customer commitment.

This lesson uses fictional orders and clearly labeled assumptions.
Times are elapsed production hours from the start of a shift unless stated otherwise.
Money, where used, is in United States dollars (USD).
The methods are decision aids, not permission to bypass safety, quality, or authorization.
A feasible modeled schedule can still fail on the floor if the model omits a real constraint.

## 2. Learning outcomes

After studying this topic, you can:

1. **Distinguish** capacity loading from finite scheduling and identify the decision each supports.
   Exercise 1 tests the distinction.
2. **Calculate** completion, flow time, lateness, tardiness, and on-time status for a single-machine sequence.
   Exercise 2 requires an auditable time ledger.
3. **Compare** first-come-first-served, shortest-processing-time, and earliest-due-date rules under stated assumptions.
   Exercise 3 tests competing objectives.
4. **Construct** a two-machine flow-shop sequence using Johnson's rule and its prerequisites.
   Exercise 4 checks both the sequence and the time chart.
5. **Build** a feasible job-shop timeline with routing and machine non-overlap.
   Exercise 5 tests precedence and contention.
6. **Diagnose** calendar, setup, material, labor, and data constraints before releasing a schedule.
   Exercise 6 asks you to identify infeasible proposals.
7. **Design** an exception and rescheduling process that protects commitments and measures change.
   Exercise 7 uses a disruption scenario.
8. **Critique** an optimization result and specify evidence for choosing a rule or solver.
   Exercise 8 requires a validation and decision design.

A good answer gives the time origin, units, feasible interval, objective, and decision owner.
A low lateness score is not enough if the sequence requires an unavailable machine or unapproved process.

## 3. The core mental model

A schedule is an executable hypothesis about the future, not a promise that the future will match it.
Each operation occupies a time interval on every resource it needs.
The intervals must respect both physical resource availability and the order in which the job's operations can occur.

```text
Customer commitments + authorized order priorities
                         |
MRP orders + actual shop status + routings + material availability
                         |
             resource calendars + setups + skills
                         |
                         v
      candidate sequence and operation start / finish times
                         |
        +----------------+-----------------+
        |                                  |
   constraint check                 objective measures
   machine / worker                 tardiness / flow / cost
   precedence / quality             change / service risk
        |                                  |
        +----------------+-----------------+
                         |
             supervisor and planner review
                         |
          released dispatch + exception ownership
                         |
             actual events and deviations
                         |
           repair schedule or escalate decision
```

Read the diagram from left to right and then back through feedback.
The order book and due dates express service requirements.
Routing operations and material readiness describe what each job must consume.
Calendars describe when each resource can supply time.
The scheduler constructs intervals and checks whether intervals overlap illegally.
A score ranks only the feasible options according to agreed priorities.
Approval and communication make the plan actionable.
Execution feedback supplies facts for the next decision.

Three shop structures help you decide how much coordination is needed.
A single-machine problem has one shared processing resource.
A flow shop has several machines visited in the same order by each job.
A job shop allows jobs to have different routes and machine visit orders.
Real facilities may combine all three structures in different work centers.
The labels are approximations, not organizational charts.

Finite does not mean optimal.
A dispatching rule can make a feasible local choice without finding the best global schedule.
An optimizer can prove optimality only for its mathematical model and stated objective.
An omitted cleaning crew or inaccurate processing time can make a mathematically valid answer operationally infeasible.

The time axis is as important as the total number of hours.
A machine available for eight hours across two shifts cannot necessarily run a non-preemptive six-hour job if no continuous six-hour window exists.
A job can wait for material even while a machine is free.
A schedule that uses every minute leaves no protection against ordinary variation.
The right amount of buffer is a business choice informed by variability and service risk.

## 4. Definitions and vocabulary

The table fixes the meaning used throughout this lesson.
A due date and an operation finish time must share a clock and time zone before subtraction.

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Job | Order or lot routed through one or more operations; identifier | Treating a customer order and every sublot as the same object |
| Operation | A required step of a job on eligible resources; operation identifier | Assuming all steps can occur simultaneously |
| Machine | A capacity-constrained resource; resource identifier | Treating a work center containing several machines as one machine without checking interchangeability |
| Routing | Ordered or partially ordered operations and eligible resources; version | Using an obsolete process revision |
| Processing time | Duration for an operation excluding explicitly modeled setup; hours | Applying a per-unit rate as a per-lot duration |
| Setup | Preparation needed before processing; hours per transition | Both adding setup to processing and reserving it separately |
| Sequence-dependent setup | Setup duration determined by predecessor and successor states; hours | Assuming a single average fits all transitions |
| Release time | Earliest time a job may begin; hours from origin | Confusing order creation with material readiness |
| Due date | Required completion time for the stated boundary; hours from origin | Comparing a machine finish with a shipment due date without post-processing |
| Start time | Beginning of a scheduled interval; hours from origin | Treating a planned start as an actual event |
| Completion time | End of the relevant job or operation; hours from origin | Using the first operation finish as job completion |
| Flow time | Completion minus release time; hours | Calling it pure processing time |
| Makespan | Final job completion minus schedule origin; hours | Treating makespan as customer tardiness |
| Lateness | Completion minus due date; signed hours | Truncating early completion to zero |
| Tardiness | Maximum of zero and lateness; hours | Using signed lateness as a penalty |
| Earliness | Maximum of zero and due date minus completion; hours | Assuming early production has no carrying cost |
| Weighted tardiness | Tardiness multiplied by a stated priority weight; weighted hours | Calling a subjective weight a verified monetary loss |
| Dispatch rule | Local criterion for selecting the next eligible job | Assuming the rule proves global optimality |
| FCFS | First-come-first-served, ordered by queue arrival | Ordering by order number when arrival times differ |
| SPT | Shortest processing time, ordered by duration | Assuming it is best for every service objective |
| EDD | Earliest due date, ordered by due date | Assuming it minimizes total tardiness for every instance |
| Critical ratio | Remaining time until due divided by estimated remaining work; dimensionless | Ignoring queues and resource calendars in remaining work |
| Slack | Due time minus current time minus remaining work; hours | Mistaking positive slack for a guaranteed on-time finish |
| Preemption | Interrupting and later resuming an operation | Assuming every physical process permits interruption |
| Non-preemptive | Operation must run continuously once started | Splitting across maintenance without authorization |
| Calendar | Allowed working intervals and closures; timestamps | Using paid shift time as machine-ready time |
| Bottleneck | Resource constraining throughput in the system under consideration | Assuming the busiest machine is always the constraint |
| WIP | Work in process: released but not completed work; jobs or units | Counting planned but unreleased orders as physical WIP |
| Frozen zone | Near-term period in which changes require defined authority | Treating a frozen schedule as immune to safety events |
| Feasibility | Satisfaction of all modeled hard constraints | Equating it with quality, robustness, or optimality |
| Optimality gap | Difference between incumbent objective and a valid solver bound, under its reported convention | Treating the gap as forecast accuracy |
| Rescheduling | Revising an existing schedule after new information | Republishing every minor timing change without considering disruption |
| ISA-95 | International standard series for enterprise-control system integration | Treating an information model as a mandatory dispatch rule |

## 5. The mechanics, step by step

The procedure below produces a reviewed operational plan.
Each step names an input, a decision, an output, and the point at which the process must branch.
It applies whether the arithmetic is performed by hand, a spreadsheet, or a solver.

### Step 1 — Define the decision boundary

Input: the planning horizon, a site, a set of customer or internal requirements, and the next release decision.
Write down the first and last timestamps in the horizon.
Identify whether completion means leaving a machine, passing inspection, packing, or arriving at the customer.
State which orders are firm, planned, or merely forecast.
Do not schedule an unapproved forecast as if it were an accepted customer obligation.
Output: a list of candidate jobs, their status, the boundary, and an owner.
If the boundary excludes a downstream constraint, reserve a credible allowance or extend the model.
For example, a machining finish at 15:00 is not a 15:00 shipment if inspection takes two hours.
If the due date has no time of day, agree an explicit interpretation with the customer team.
Record the interpretation so a later shift does not compare incompatible timestamps.

### Step 2 — Reconcile order and operation status

Input: open manufacturing orders, actual completions, rejects, and quantities still required.
Exclude completed operations and represent in-process work with its remaining duration.
Confirm whether partially processed material can safely pause.
Do not schedule a full original lot a second time after part of it has finished.
Connect each operation to the correct order and routing revision.
Output: a snapshot with a timestamp and remaining work by operation.
If actual status conflicts with the enterprise record, hold the affected plan for supervisor confirmation.
If a job has been physically split, model each transferable sublot and its quantity explicitly.
Retain the parent-child link so completing one sublot does not falsely close the full order.
A snapshot is a starting point, not a substitute for collecting subsequent events.

### Step 3 — Establish operation dependencies

Input: released routing, engineering revision, and quality requirements.
For each job, identify the operations that must finish before the next operation can begin.
Represent inspection, cooling, transport, and cure as actual steps or explicit lags when they matter.
Where parallel steps are permitted, state whether both must finish before assembly.
Output: a precedence graph for each job.
If the graph has a cycle, correct the master data before scheduling.
If an alternate route exists, check qualification for the specific product, lot, and resource.
An alternate route is not automatically usable merely because it appears in a master-data table.
If a transfer batch is smaller than the lot, model quantity and release timing rather than assuming the entire lot transfers at first-piece completion.
Keep the physical precedence even if a proposed due date makes it inconvenient.

### Step 4 — Identify resources and eligibility

Input: operation requirements, equipment capabilities, qualifications, and tooling.
List each machine and all eligible alternatives.
List auxiliary resources consumed at the same time, such as a fixture, crane, operator, or test chamber.
State whether labor is required for setup only or for the full processing interval.
Output: an eligibility and simultaneous-resource requirement ledger.
If two machines share one fixture, they cannot both use that fixture at the same time.
If one operator attends two machines, establish the allowed overlap rather than assuming unlimited attention.
If a job requires certified staff, a generic headcount is not sufficient evidence of coverage.
If a machine is eligible only after engineering approval, treat that approval as a release condition.
Reserve safety-critical capacity independently from commercial priorities.

### Step 5 — Build finite calendars

Input: shifts, breaks, holidays, maintenance, planned cleaning, and labor attendance.
Convert all times into a consistent clock and retain the original time-zone information.
Build available intervals separately for each machine and each shared resource.
Output: non-overlapping availability windows with an effective timestamp.
If work is non-preemptive, require the entire operation to fit in a continuous permitted interval.
If interruption is allowed, document restart time, quality conditions, and who authorizes it.
Do not subtract an availability allowance from processing time after already blocking the same planned maintenance on the calendar.
Do not silently assume overtime is available: identify authorization, worker coverage, and cost.
A calendar is finite only to the extent that its unavailability is actually represented.
A weekly capacity total alone is insufficient to test within-day feasibility.

### Step 6 — Determine readiness and release

Input: material availability, preceding operations, permits, and inspection or engineering holds.
For each job or operation, calculate the earliest lawful and physical release time.
A purchase order promised for Thursday is not material available in stores on Wednesday.
Consider inbound receiving and quality inspection before making material eligible.
Output: a release timestamp or an explicit hold and owner.
If material is uncertain, produce a scenario rather than treating an estimated arrival as fact.
If a quality hold has no estimated clearance, leave the job in an exception queue.
The scheduler may reserve a provisional slot, but label it conditional.
Distinguish a release date for the whole lot from a date at which only part of the lot is available.
Review open holds before every dispatch, not just at the start of the week.

### Step 7 — Estimate run and transition times

Input: remaining quantity, standard cycle rate, setup matrix, and initial equipment state.
Convert rate and quantity into duration using consistent units.
Separate fixed setup from quantity-dependent run time.
For sequence-dependent setups, use the actual predecessor state, including the state at the horizon start.
Output: a duration by operation and a transition duration by feasible predecessor-successor pair.
If cleaning changes with product family, record the direction: A-to-B need not equal B-to-A.
If the setup crew is a shared resource, reserve crew time as well as machine time.
If setup can occur before material release, distinguish anticipatory setup from processing readiness.
If it cannot, require setup to start after readiness.
Check whether the standard includes breaks, scrap, or previous downtime before adding allowances.

### Step 8 — Choose a shop representation

Input: routing graph, machine eligibility, and the decision horizon.
Use a single-machine representation for one constrained resource when other steps are not binding inside the decision window.
Use a flow-shop representation when jobs share the same machine order.
Use a job-shop representation when jobs visit machines in different orders.
Output: a model boundary and a list of omitted or approximated constraints.
If all jobs use the same two machines in the same order, test whether Johnson's rule applies.
If routes differ, do not apply the flow-shop rule as if the shared order were guaranteed.
If a work center contains parallel machines with different qualifications or speeds, model them as distinct eligible resources.
A line that mostly follows one route but has a rework loop may require a hybrid model.
Select the simplest representation that preserves the decision-relevant constraints.

### Step 9 — Agree the objective and hard constraints

Input: customer obligations, safety and quality rules, operating costs, and management priorities.
Separate constraints that cannot be violated from outcomes that can be traded off.
A hard maintenance outage is a calendar constraint, not a small penalty in an objective.
A tardiness target can be a score or an explicit service constraint, depending on the agreement.
Output: a ranked objective, constraints, and authorized exception paths.
If customer service is lexicographically first, first minimize the defined service loss and then optimize cost among equally acceptable solutions.
If using weights, state their units and the rationale for their magnitude.
For monetary weights, include only incremental and defensible costs; avoid inventing penalty revenue.
Make the tie-break rule deterministic, such as due date then stable job identifier.
Record who can approve overtime, a changed shipment, or an alternate route.

### Step 10 — Construct a baseline

Input: the current accepted sequence and the validated snapshot.
Calculate planned operation starts and finishes without violating resource and precedence constraints.
A first-come-first-served (FCFS) sequence is one transparent baseline when queue arrivals are known.
The currently published schedule is another important baseline because changes have costs.
Output: an interval ledger and objective values for the baseline.
If the current sequence is infeasible, explain which constraints it violates before comparing scores.
Do not compare a schedule with maintenance included to another schedule that simply ignores maintenance.
Use identical release dates, due dates, resource calendars, and measurement boundaries across options.
For in-process work, preserve the actually started operation unless interruption is permitted.
Make the comparison reproducible from a timestamped snapshot.

### Step 11 — Apply dispatch rules where appropriate

Input: the eligible queue at a decision time and the agreed score.
FCFS selects the job that arrived first to the relevant queue.
Shortest processing time (SPT) selects the job with the smallest processing duration.
Earliest due date (EDD) selects the job with the earliest stated due time.
Critical ratio selects the smallest ratio of time until due to remaining estimated work.
Output: a selected job and a reason, not merely a rank number.
If no job is eligible, advance to the next release or resource-availability event instead of scheduling unavailable work.
If a rule ties, apply a documented secondary key.
If setup depends on sequence, include its effect in a scenario; an SPT rank based on run time alone may generate excessive cleaning.
A dispatch rule is recalculated when a resource becomes available; it need not produce a fixed all-day list.

### Step 12 — Construct and inspect multi-machine schedules

Input: routing precedence, machine calendars, setup times, and eligible jobs.
For each operation, choose a start no earlier than release, predecessor finish, and resource availability.
Reserve a non-overlapping interval on every required resource.
Output: machine-by-machine and job-by-job interval ledgers.
For a two-machine flow shop satisfying Johnson's assumptions, arrange jobs using the minimum remaining processing time rule.
If the minimum is on the first machine, place that job in the earliest open position.
If the minimum is on the second machine, place that job in the latest open position.
Remove the job and repeat, resolving ties consistently.
This rule minimizes makespan under its assumptions, not every customer-service objective.
For general job shops, a feasible heuristic or optimization model must enforce each route and each resource's non-overlap.

### Step 13 — Check every interval independently

Input: a candidate schedule and the original constraints.
For each resource, sort assigned intervals by start time and check that adjacent intervals do not overlap.
For each job, verify that every successor begins at or after the required predecessor finishes plus any lag.
Verify the start is not earlier than release or the permitted calendar interval.
Output: a violation report and a status of feasible, conditional, or rejected.
A setup interval must also obey machine and crew availability when those resources are required.
A job can start exactly when another ends if the time convention is half-open: [start, finish).
A resource with capacity greater than one requires a cumulative-capacity check, not a simple pairwise no-overlap check.
Validate due dates against the correct operation and include required downstream steps.
Never treat a low objective value as proof that these checks passed.

### Step 14 — Compare business trade-offs

Input: feasible candidate schedules, incremental cost assumptions, and the current commitments.
Compare total tardiness, maximum lateness, number of late jobs, and mean flow time where relevant.
Also compare setup hours, overtime, inventory exposure, and the number of changes to released work.
Output: a decision table that shows both the numerical differences and the approvals needed.
If one option reduces aggregate delay but makes a critical committed order late, escalate the trade-off.
If an overtime option costs USD 240 but avoids a verified USD 400 incremental loss, the net modeled benefit is USD 160.
That comparison is conditional on overtime actually being staffed and approved.
Do not count avoided penalties as cash unless the contract and counterfactual support them.
Do not treat a priority score as a monetary benefit without conversion evidence.
Prefer a transparent non-dominated set to a false single score when the business has not agreed weights.

### Step 15 — Approve, publish, and dispatch

Input: the recommended feasible option, exception list, and authority matrix.
Have the scheduler propose, the supervisor verify physical readiness, and the relevant owner approve exceptions.
Publish the revision, timestamp, effective horizon, assumptions, and first dispatch.
Output: a controlled schedule and a queue of conditional jobs or unresolved blockers.
If a customer commitment changes, the authorized customer-facing owner must communicate it.
If a job is already running, mark it as fixed unless an authorized interruption is safe.
Keep the previous version so actual changes can be measured and explained.
The floor should be able to distinguish released instructions from simulation scenarios.
A worker must be able to stop an unsafe operation regardless of the schedule.
A schedule that lacks a responsible owner is a proposal, not an executable instruction.

### Step 16 — Monitor and repair

Input: actual starts, finishes, scrap, outages, arrivals, and priority changes.
Compare events with the published schedule using common identifiers and timestamps.
Trigger review when a defined threshold, such as a material hold or predicted committed-order miss, is crossed.
Output: a local repair, a revised schedule, or an escalated decision with impact.
If the disruption is small, preserve unaffected jobs rather than unnecessarily resequencing the whole day.
If a bottleneck is down for a long period, rebuild the feasible plan and identify impossible commitments.
Quantify instability as changed job positions, moved start times, or changed customer dates.
Keep the frozen-zone policy explicit, including the authority to override it.
Capture actual reasons for deviation so duration standards and calendars improve.
Review the result at the next shift handover and close the exception only when the owner confirms resolution.

### Rule selection is conditional

The classic result for SPT assumes all jobs are available at time zero, a single continuously available machine, known deterministic processing times, and no sequence-dependent setups or preemption.
Under those conditions SPT minimizes the sum, and therefore the mean, of completion times.
With release times, nontrivial setups, or a different objective, this guarantee need not hold.
EDD minimizes maximum lateness for the analogous single-machine setting, not necessarily total tardiness or the number of late jobs.
Neither result proves that a customer will accept a particular consequence.

Johnson's rule assumes a two-machine flow shop with every job visiting machine 1 then machine 2, known processing times, all jobs available at the start, continuously available machines, and one job at a time on each machine.
Jobs do not overtake within the permutation schedule and processing is non-preemptive.
The standard result minimizes makespan, not weighted tardiness or setup cost.
Sequence-dependent setups, machine breakdowns, blocking, limited buffers, and release dates require separate analysis.

Critical ratio is a dynamic heuristic.
At time `t`, compute `(due - t) / remaining processing` when remaining processing is positive.
A ratio below 1 suggests insufficient pure processing time remains, before allowing for queues and downtime.
A negative ratio means the due time has already passed.
A zero remaining-work denominator requires a separate completion or data check.
A ratio above 1 is not a promise because downstream contention can consume the apparent margin.

## 6. Worked example

### Decision and data

A fictional pump-component plant must schedule four released lots on one finishing machine.
All four lots are physically ready at shift start, `t = 0 hours`.
The due times below refer to completion on that finishing machine, not shipment.
The customer team has already confirmed these as required internal completion times; no new external promise is implied.
Each lot needs one uninterrupted operation.
There is no setup, travel, scrap, or other shared resource in the first comparison.
The machine is continuously available for the base scenario.
All quantities are lot-level; processing time is hours per lot.

| Job | Queue arrival rank | Processing, hours | Due, hours |
|---|---:|---:|---:|
| A | 1 | 7 | 12 |
| B | 2 | 4 | 9 |
| C | 3 | 6 | 18 |
| D | 4 | 3 | 8 |

The total run time is `7 + 4 + 6 + 3 = 20 hours`.
Every non-idling sequence therefore finishes at hour 20 in this simplified case.
That fact does not make the sequences equivalent for individual due dates or flow time.
For each job, `lateness = completion - due` in hours.
Tardiness is `max(0, lateness)` in hours.
A job finishing exactly at its due time is on time.
All release times are zero, so each job's flow time equals its completion time.

### FCFS: A, B, C, D

| Job | Start, hours | Finish, hours | Due, hours | Lateness, hours | Tardiness, hours |
|---|---:|---:|---:|---:|---:|
| A | 0 | 7 | 12 | -5 | 0 |
| B | 7 | 11 | 9 | 2 | 2 |
| C | 11 | 17 | 18 | -1 | 0 |
| D | 17 | 20 | 8 | 12 | 12 |

FCFS completion sum is `7 + 11 + 17 + 20 = 55 job-hours`.
Mean flow time is `55 / 4 = 13.75 hours per job`.
Total tardiness is `0 + 2 + 0 + 12 = 14 job-hours`.
Two of four jobs are late, so on-time completion is 50% of jobs.
Maximum lateness is 12 hours.

### SPT: D, B, C, A

| Job | Start, hours | Finish, hours | Due, hours | Lateness, hours | Tardiness, hours |
|---|---:|---:|---:|---:|---:|
| D | 0 | 3 | 8 | -5 | 0 |
| B | 3 | 7 | 9 | -2 | 0 |
| C | 7 | 13 | 18 | -5 | 0 |
| A | 13 | 20 | 12 | 8 | 8 |

SPT completion sum is `3 + 7 + 13 + 20 = 43 job-hours`.
Mean flow time is `43 / 4 = 10.75 hours per job`.
Total tardiness is 8 job-hours.
One of four jobs is late, so on-time completion is 75% of jobs.
Maximum lateness is 8 hours.

### EDD: D, B, A, C

| Job | Start, hours | Finish, hours | Due, hours | Lateness, hours | Tardiness, hours |
|---|---:|---:|---:|---:|---:|
| D | 0 | 3 | 8 | -5 | 0 |
| B | 3 | 7 | 9 | -2 | 0 |
| A | 7 | 14 | 12 | 2 | 2 |
| C | 14 | 20 | 18 | 2 | 2 |

EDD completion sum is `3 + 7 + 14 + 20 = 44 job-hours`.
Mean flow time is `44 / 4 = 11 hours per job`.
Total tardiness is `0 + 0 + 2 + 2 = 4 job-hours`.
Two of four jobs are late, so on-time completion is 50% of jobs.
Maximum lateness is 2 hours.

### Interpret the base comparison

| Sequence | Mean flow, hours/job | Total tardiness, job-hours | Late jobs, count | Maximum lateness, hours |
|---|---:|---:|---:|---:|
| FCFS | 13.75 | 14 | 2 | 12 |
| SPT | 10.75 | 8 | 1 | 8 |
| EDD | 11.00 | 4 | 2 | 2 |

SPT has the lowest mean flow among these choices and, under the stated assumptions, among all sequences.
EDD has the lowest maximum lateness under its stated assumptions.
In this particular instance, EDD also has less total tardiness than the other two displayed rules.
That last observation is not a general theorem about EDD and total tardiness.
SPT has fewer late jobs here, but job A is eight hours late.
If A is a critical commitment, the supervisor may prefer EDD's smaller maximum delay.
If the service measure is simply number of late jobs, SPT is preferable to EDD on that measure for this instance.
The decision requires an agreed service policy; neither rank is universally correct.
No cost conclusion follows until reliable incremental cost and customer impact are supplied.

### Add maintenance and test non-preemption

Now the machine is unavailable from hour 10 to hour 12 for mandatory maintenance.
The EDD sequence remains D, B, A, C.
D runs from hour 0 to 3 and B from hour 3 to 7.
A needs seven uninterrupted hours and cannot fit between hour 7 and hour 10.
It starts after maintenance at hour 12 and finishes at hour 19.
C runs from hour 19 to hour 25.
The idle interval from hour 7 to 10 is forced by this fixed sequence and the non-preemption assumption.
Another sequence may use the gap; that does not authorize interrupting A.

| Job or event | Start, hours | Finish, hours | Due, hours | Tardiness, hours |
|---|---:|---:|---:|---:|
| D | 0 | 3 | 8 | 0 |
| B | 3 | 7 | 9 | 0 |
| Idle | 7 | 10 | — | — |
| Maintenance | 10 | 12 | — | — |
| A | 12 | 19 | 12 | 7 |
| C | 19 | 25 | 18 | 7 |

The revised completion sum is `3 + 7 + 19 + 25 = 54 job-hours`.
Mean flow is `54 / 4 = 13.5 hours per job`.
Total tardiness is `0 + 0 + 7 + 7 = 14 job-hours`.
Maximum lateness is 7 hours.
This is an evaluation of the fixed EDD order, not a claim that EDD is optimal with the outage.
The scheduler should compare permitted alternate sequences, overtime, or commitment changes rather than deleting maintenance from the calendar.

### Compare a feasible resequencing

Consider D, A, B, C under the same outage and readiness assumptions.
This is a candidate, not an instruction to change the published plan.
It fills the otherwise idle interval by starting A immediately after D.
Both jobs finish before maintenance begins.

| Job or event | Start, hours | Finish, hours | Due, hours | Tardiness, hours |
|---|---:|---:|---:|---:|
| D | 0 | 3 | 8 | 0 |
| A | 3 | 10 | 12 | 0 |
| Maintenance | 10 | 12 | — | — |
| B | 12 | 16 | 9 | 7 |
| C | 16 | 22 | 18 | 4 |

The completion sum is `3 + 10 + 16 + 22 = 51 job-hours`.
Mean flow is `51 / 4 = 12.75 hours per job`.
Total tardiness is `0 + 0 + 7 + 4 = 11 job-hours`.
Maximum lateness is seven hours, and two jobs are late.
Relative to fixed EDD, the candidate reduces total tardiness by three job-hours.
It reduces mean flow by 0.75 hours per job and finishes the whole set three hours earlier.
Those improvements come with an important distributional change.
Job A moves from seven hours late to on time.
Job B moves from on time to seven hours late.
Job C moves from seven hours late to four hours late.
If B supports an especially time-critical customer, the supervisor must not hide its delay in an aggregate improvement.
The customer owner should evaluate that exposure before approving the change.
Neither candidate requires overtime or an interruption of maintenance.
No claim is made that this alternative is globally optimal.
The example illustrates why a feasible gap-filling move can improve throughput while transferring service risk.

### Two-machine flow shop

Three different fictional jobs each visit machine 1 and then machine 2.
All are available at hour 0; machines are continuously available and have an unlimited intermediate buffer.
Processing is non-preemptive, and setup is absent or sequence-independent and already included in the displayed times.

| Job | Machine 1, hours | Machine 2, hours |
|---|---:|---:|
| P | 3 | 8 |
| Q | 6 | 4 |
| R | 7 | 5 |

The smallest remaining time is P's 3 hours on machine 1, so place P first.
Of Q and R, Q's 4 hours on machine 2 is smallest, so place Q last.
R occupies the middle position.
Johnson's sequence is P, R, Q.

| Job | Machine 1 interval, hours | Machine 2 interval, hours |
|---|---|---|
| P | [0, 3) | [3, 11) |
| R | [3, 10) | [11, 16) |
| Q | [10, 16) | [16, 20) |

R finishes machine 1 at hour 10 but waits until machine 2 is free at hour 11.
Q cannot enter machine 2 before both its machine 1 finish and R's machine 2 finish, both at hour 16.
The makespan is 20 hours.
For comparison, P, Q, R yields machine 2 completions at hours 11, 15, and 21, for a makespan of 21 hours.
This is a makespan comparison, not a due-date or labor-cost comparison.
If the intermediate buffer cannot hold R during hour 10 to 11, re-evaluate the model rather than applying the table unchanged.

### Two-route job shop

Two further fictional jobs have opposite machine routes.
Job X needs machine 1 for four hours, then machine 2 for three hours.
Job Y needs machine 2 for five hours, then machine 1 for two hours.
All operations are released at hour 0 and machines are continuously available.

| Resource | First operation, hours | Second operation, hours |
|---|---|---|
| Machine 1 | X1 [0, 4) | Y2 [5, 7) |
| Machine 2 | Y1 [0, 5) | X2 [5, 8) |

X1 finishes at hour 4 but X2 must wait until machine 2 is free at hour 5.
Y1 finishes at hour 5 before Y2 starts at hour 5.
No two intervals on the same machine overlap.
The makespan is 8 hours.
Eight hours is also a lower bound because machine 2 must process `5 + 3 = 8 hours` of work.
The displayed schedule reaches that lower bound and is optimal for makespan under these assumptions.
Johnson's two-machine flow-shop construction is not directly applicable because the two jobs have different routes.

## 7. Code, query, or calculation

The following standard-library Python script reproduces the single-machine comparison and the fixed EDD maintenance scenario.
It constructs a static priority sequence at time zero, not a general dynamic dispatcher or job-shop optimizer.
All jobs are released at zero, processing times are positive, and each job needs one non-preemptive operation.
The maintenance interval is half-open `[10, 12)` and the machine is otherwise continuously available.
The script excludes setup, secondary resources, and transport.
It reports signed lateness separately from nonnegative tardiness.
Copy it into a Python 3 file and run it with `python filename.py`.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    name: str
    processing: int  # hours per lot
    due: int         # hours from shift start
    arrival: int     # FCFS queue rank


JOBS = [
    Job("A", 7, 12, 1),
    Job("B", 4, 9, 2),
    Job("C", 6, 18, 3),
    Job("D", 3, 8, 4),
]


def first_slot(start, duration, closures):
    """Find the earliest uninterrupted slot outside sorted closures."""
    t = start
    for lo, hi in sorted(closures):
        if t + duration <= lo:
            break
        if t < hi and t + duration > lo:
            t = hi
    return t


def schedule(jobs, key, closures=()):
    t = 0
    rows = []
    for job in sorted(jobs, key=key):
        start = first_slot(t, job.processing, closures)
        finish = start + job.processing
        lateness = finish - job.due
        rows.append((job.name, start, finish, lateness, max(0, lateness)))
        t = finish
    return rows


def report(label, rows):
    order = "".join(row[0] for row in rows)
    mean_flow = sum(row[2] for row in rows) / len(rows)
    total_tardy = sum(row[4] for row in rows)
    late_count = sum(row[4] > 0 for row in rows)
    max_lateness = max(row[3] for row in rows)
    print(f"{label}: order={order}, mean_flow={mean_flow:.2f} h/job, "
          f"tardiness={total_tardy} job-h, late={late_count}, "
          f"max_lateness={max_lateness} h")
    for name, start, finish, lateness, tardiness in rows:
        print(f"  {name}: [{start},{finish}), L={lateness}, T={tardiness}")


report("FCFS", schedule(JOBS, lambda j: (j.arrival, j.name)))
report("SPT", schedule(JOBS, lambda j: (j.processing, j.name)))
report("EDD", schedule(JOBS, lambda j: (j.due, j.name)))
report("EDD outage", schedule(JOBS, lambda j: (j.due, j.name), [(10, 12)]))
```

Expected output:

```text
FCFS: order=ABCD, mean_flow=13.75 h/job, tardiness=14 job-h, late=2, max_lateness=12 h
  A: [0,7), L=-5, T=0
  B: [7,11), L=2, T=2
  C: [11,17), L=-1, T=0
  D: [17,20), L=12, T=12
SPT: order=DBCA, mean_flow=10.75 h/job, tardiness=8 job-h, late=1, max_lateness=8 h
  D: [0,3), L=-5, T=0
  B: [3,7), L=-2, T=0
  C: [7,13), L=-5, T=0
  A: [13,20), L=8, T=8
EDD: order=DBAC, mean_flow=11.00 h/job, tardiness=4 job-h, late=2, max_lateness=2 h
  D: [0,3), L=-5, T=0
  B: [3,7), L=-2, T=0
  A: [7,14), L=2, T=2
  C: [14,20), L=2, T=2
EDD outage: order=DBAC, mean_flow=13.50 h/job, tardiness=14 job-h, late=2, max_lateness=7 h
  D: [0,3), L=-5, T=0
  B: [3,7), L=-2, T=0
  A: [12,19), L=7, T=7
  C: [19,25), L=7, T=7
```

The `first_slot` routine checks whether the whole operation fits before each closure.
If it overlaps a closure, the routine moves the start to the closure's end and checks the next closure.
It does not interrupt a running job or reorder jobs to fill the gap.
The use of half-open intervals permits an operation to end exactly when maintenance begins.
The inputs are assumed to describe valid nonempty closures and a nonempty job list.
Mean flow equals mean completion only because every release time is zero.
For real releases, compute each job's completion minus its own release.
Add material, labor, routing, or setup constraints before using a result for an actual release decision.

## 8. Data, standards, and regulatory context

A schedule is only as credible as the data snapshot used to construct it.
A scheduling system needs a shared definition of an order, operation, resource, event, and time boundary.
The following ledger is a practical data contract rather than a claim that a particular standard mandates every field.

### Orders and commitments

Keep the manufacturing order identifier, parent customer order where applicable, product revision, quantity, unit of measure, and status.
Store both the requested date and the authorized committed date when they differ.
Record the completion boundary and time zone for each due timestamp.
Maintain a reason and approval for priority changes.
Retain cancellation and hold states so obsolete orders do not compete for machine time.
A quantity expressed in pieces cannot be multiplied directly by a rate in hours per kilogram.
An estimated customer need is not automatically an accepted commitment.
The customer-facing owner decides whether and how a promise changes.

### Operations and routing revisions

Identify each operation, predecessor, eligible machine, expected duration, and applicable routing version.
Store minimum transfer quantities and lag times where the process uses them.
Record alternate-resource qualification and its effective dates.
Keep engineering and quality holds visible at the operation or lot level.
Capture whether setup and run times are per piece, per batch, or per transition.
Represent rework as a controlled route rather than silently deleting failed work.
Actual order status must be reconciled with the route version in force for that lot.
A revised standard should not silently rewrite the historic basis of a published schedule.

### Resource and calendar records

Identify machines individually when capabilities or calendars differ.
Record shift windows, planned maintenance, breaks, and declared staffing with effective timestamps.
Capture each shared fixture, tool, or crew where simultaneous demand can exceed supply.
Represent the initial machine state when changeover depends on the predecessor product.
Use a consistent time-zone conversion and explicit daylight-saving interpretation for cross-site data.
Mark overtime as conditional until the person with authority approves coverage.
Track the source and last update of calendar changes.
Avoid treating an overall equipment effectiveness percentage as an extra calendar closure when the same losses were already modeled.

### Execution events and reconciliation

Record planned and actual start and finish separately.
Capture quantities completed, rejected, held, and transferred, with units.
Record event time, ingestion time, resource identity, and source system.
Late-arriving events should update the next snapshot without silently changing a previously published version.
Link downtime to a reason and duration rather than inferring every gap as breakdown.
Define whether an event signifies operator entry, machine state, or quality release.
Check duplicates and time-order anomalies before using the records for standards.
Store the exception owner and disposition when the schedule differs from reality.

### ISA-95 and IEC 62264

The International Society of Automation ISA-95 series and the related IEC 62264 series address enterprise-control system integration and information models.
Their scope is useful for aligning enterprise planning and manufacturing operations data across a boundary.
The standards do not by themselves select EDD, prove a schedule feasible, or define a site's customer priority policy.
Implementations still need mapping from local identifiers, states, calendars, and units.
Consult the specific licensed edition and organizational adoption decision for normative details.
The standard is not automatically a law merely because an organization uses it.
As of November 2026, this lesson treats the standards as integration references, not a substitute for applicable regulation.
The official ISA and IEC catalog pages in Sources provide the authoritative scope.

### Quality, safety, and controlled production

The applicable legal and quality obligations depend on product, jurisdiction, and process.
For example, the United States Food and Drug Administration's current good manufacturing practice regulations in 21 CFR Part 211 apply to covered finished pharmaceuticals.
The electronic Code of Federal Regulations is the authoritative source for the applicable text and current amendments.
A finite schedule cannot authorize bypassing required quality review, equipment controls, or documented procedures.
It also does not establish that a particular factory or product is regulated under Part 211.
A site must determine its own applicability with qualified regulatory and quality personnel.
Treat required inspections and holds as constraints, not optional service penalties.
As of November 2026, consult the current rule and site procedures before making a regulated production decision.

### Security and auditability

Limit who can change resource calendars, due dates, routing eligibility, and release status.
Record the actor, timestamp, previous value, and reason for an authorized override.
Keep scenario schedules separate from released instructions.
Protect customer priorities and production information according to site policy.
Ensure that interfaces identify stale data rather than silently supplying old availability.
Reconcile the source system with the schedule when identifiers change.
An audit trail supports accountability but does not prove that the underlying time standard is correct.
Retention and electronic-record requirements depend on the applicable regime and site controls.

## 9. Industry practice and case evidence

### Public case: Toyota and leveling

Toyota publicly describes *heijunka*, or production leveling, as leveling the type and quantity of production over a fixed period.
The Toyota Production System page also describes just-in-time production and the relation between production flow and customer demand.
This is a company-authored description of Toyota's own system, not an independently measured effect size.
The page does not establish the exact algorithm, dispatch parameters, or quantitative result for the fictional examples in this lesson.
Source: Toyota Motor Corporation, [Toyota Production System](https://global.toyota/en/company/vision-and-philosophy/production-system/).

[INFERENCE] A plant using leveling can use a finite schedule to test whether the proposed mix fits its actual machines, staffing, and changeovers.
[INFERENCE] Frequent mix changes may improve responsiveness but can consume more setup time if changeovers are long.
Neither inference is a claim that Toyota uses the specific Python script or numerical data above.
A manager should verify local changeover times and demand patterns before copying a leveling cadence.

### Public technical example: OR-Tools job shop

Google's OR-Tools documentation presents a job-shop scheduling example using operations, machine constraints, and precedence.
The documentation explains the no-overlap requirement for tasks assigned to the same machine and minimizes makespan in its example.
This is official software documentation and an illustrative model, not evidence of a particular plant's achieved savings.
The model demonstrates a representation; a real implementation must add its own calendars, materials, labor, and quality constraints.
Source: [OR-Tools job shop scheduling](https://developers.google.com/optimization/scheduling/job_shop).

[INFERENCE] An industrial team can use such a model to generate candidate sequences and compare them with its published dispatch plan.
[INFERENCE] A solver's advantage should be measured against an equally feasible baseline on historical snapshots and prospective pilots.
These are recommendations, not documented customer outcomes of the OR-Tools example.
A solver result without a valid bound does not establish optimality.
A solver result with a bound establishes a statement about the encoded instance, not about missing real-world constraints.

### Evidence and transfer

Named company descriptions are useful for understanding operating principles.
They usually do not expose enough data to reproduce a plant's exact schedule or isolate a causal performance effect.
Academic scheduling results provide proofs under precise assumptions.
Software documentation provides executable modeling patterns, not automatic business validation.
Use these evidence types together without treating them as interchangeable.
For your own site, retain an unmodified baseline, input snapshot, actual outcomes, and reasons for overrides.
Compare like-for-like service boundaries and account for breakdowns and interventions.
Do not claim causal improvement from one unusually favorable shift.

## 10. Failure modes and red flags

The table connects an early signal to a mechanism and an action.
A mitigation is a control to consider, not proof that the risk has disappeared.

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Double-booked machine | Overlapping intervals in a dispatch list | Two jobs claim the same unit of capacity | A job waits and its finish is false | Check machine interval non-overlap before release |
| Wrong routing order | Successor starts before predecessor finishes | Precedence omitted or reversed | Unworkable or nonconforming work | Validate operation graph and actual status |
| Missing material | Planned start precedes receiving or inspection | Purchase promise treated as stock | Machine stands idle or job is interrupted | Gate release on confirmed usable material |
| Hidden maintenance | Calendar says available during a work order | Separate systems disagree | Candidate schedule cannot run | Reconcile maintenance and production calendars |
| Shared fixture conflict | Two machines start work requiring one fixture | Auxiliary resource omitted | One machine cannot start | Reserve fixture capacity in both intervals |
| Unstaffed machine | Schedule uses a shift without qualified labor | Machine availability confused with labor | Delayed start or unsafe staffing | Model skill and coverage by shift |
| Setup omitted | Frequent cleaning overruns | Transition matrix or initial state absent | More elapsed time and missed dates | Model predecessor-dependent setup and crew |
| Setup counted twice | Planned time much longer than shop observation | Duration already includes separately modeled setup | False overload and unnecessary overtime | Define time components and audit units |
| Impossible non-preemption | Operation crosses outage with no restart rule | Model assumes split processing | Quality risk or missed finish | Fit continuous slots or approve interruption explicitly |
| Wrong due boundary | All machine jobs appear on time but shipments are late | Packing and transport omitted | False customer confidence | Define completion boundary and downstream allowance |
| Stale execution state | Finished lots remain in queue | Late or missing event interface | Duplicate scheduling and false capacity shortage | Reconcile floor snapshot and timestamps |
| Priority churn | Many daily resequences with no recorded cause | Every small change triggers global optimization | Setup disruption and worker distrust | Freeze near-term work with exception authority |
| Overloaded buffer | Jobs finish upstream with nowhere to wait | Infinite intermediate storage assumed | Blocking and loss of throughput | Model buffer and transfer restrictions |
| Bad units | Calculated operation takes minutes instead of hours | Rate, batch size, or unit mismatched | Unrealistic machine load | Validate quantity, rate, and time units |
| Objective mismatch | Low makespan but critical order is late | Metric ignores customer priority | Commercial harm despite a good score | Agree service objective and review individual commitments |
| False solver confidence | Small reported gap but repeated shop overrides | Model excludes practical constraints | Mathematically good, unusable plan | Independently check constraints and investigate overrides |
| Quality hold ignored | Held lot appears on first dispatch | Status not integrated or overwritten | Unapproved processing or release | Preserve hold as a hard gate and assign owner |
| Phantom overtime | Plan assumes extra shift without approval | Capacity option treated as already available | Missed commitments and labor conflict | Confirm authorization, qualified coverage, and cost |
| Time-zone error | Work appears to finish before it starts across sites | Mixed local and universal timestamps | Incorrect precedence and due-date comparison | Normalize timestamps and retain local context |
| Bottleneck moved | Old constrained machine is idle while another queue grows | Product mix or downtime changed | Wrong release priorities | Monitor queue, utilization, and throughput by interval |

Review the reason for overrides as a leading indicator of model quality.
Repeated manual correction may identify a missing constraint, not a lack of operator discipline.
Do not eliminate a quality hold simply to improve a score.

## 11. Assumptions and limits

A result inherits the assumptions of its data, model, and decision boundary.
Test those assumptions before interpreting a guarantee or releasing work.

| Assumption | Why needed here | What breaks it | Practical test |
|---|---|---|---|
| Jobs released at hour 0 | Makes fixed SPT and EDD comparison meaningful | Incoming material or predecessor operation is late | Compare readiness timestamps with actual release events |
| Deterministic durations | Allows exact completion arithmetic | Variable cycle time, scrap, or rework | Compare standards with actual distributions by product |
| One operation per single-machine job | Makes completion equal machine finish | Inspection or packaging follows | Trace each job to the required delivery boundary |
| Continuous base availability | Supports textbook dispatch-rule results | Maintenance, breaks, or breakdowns | Compare schedule intervals to the resource calendar |
| No sequence-dependent setup | Allows the rules to be ranked by processing alone | Product transitions need unequal cleaning | Review observed transition matrix and initial state |
| Non-preemptive processing | Requires continuous slots | Process can or must be interrupted | Verify product procedure and restart requirements |
| One job per machine | Supports pairwise interval check | Batch equipment can process compatible lots together | Confirm physical capacity and batching rules |
| Unlimited flow-shop buffer | Allows R to wait between machines | No floor space or blocking transfer | Inspect storage and transport capacity |
| All flow-shop jobs share route | Supports Johnson's construction | Different route or rework step | Compare approved operation sequences |
| Due dates use same clock | Makes lateness arithmetic valid | Different zones, date-only commitments, or shipping dates | Reconcile timestamp and delivery boundary with order owner |
| Equal job importance in summary | Makes unweighted totals interpretable | Customers or lots have different business impacts | Agree priority policy and report individual jobs |
| Valid fixed outage window | Supports the maintenance comparison | Maintenance duration changes or is movable | Confirm maintenance work order and completion events |
| No material or labor contention | Keeps the example hand-calculable | Shared operator, component, or fixture | Walk the floor and check resource assignments |
| No stochastic arrivals | Supports a static sequence | Urgent jobs arrive mid-shift | Backtest a rolling event-driven dispatch policy |
| Completion at due time is on time | Defines the service metric | Contract requires earlier cutoff or acceptance | Confirm the contractual definition and clock |
| Objective adequately represents business value | Makes candidate ranking decision-useful | Safety, service, or cost omitted | Review outliers and exceptions with decision owners |

The analytical guarantees are not evidence that a real facility has these properties.
A finite schedule does not predict all breakdowns or automatically reduce variability.
Simulation or scenario analysis can test sensitivity, but it does not cure missing source data.
Treat an exact optimizer's answer as conditional on both the input snapshot and the encoded constraints.

## 12. Exercises

Use the stated assumptions rather than silently adding missing resources.
For numerical work, show the job order, intervals, units, and intermediate arithmetic.
A solution or grading rubric follows in Section 13.

### Exercise 1 — Loading versus scheduling

**Type:** recall and explanation.
**Time:** 10 minutes.
A work center has eight available machine-hours on Monday and receives two five-hour non-preemptive jobs, both due Monday.
The aggregate report shows ten hours of demand.
**Deliverable:** explain the overload and name two decisions that a finite schedule would expose.
**Acceptance:** identify the two-hour aggregate gap, distinguish a bucket total from exact intervals, and name an owner for an exception.
Do not assume overtime, split processing, or a second machine without stating the assumption.

### Exercise 2 — Compute a schedule

**Type:** application.
**Time:** 15 minutes.
Jobs E, F, and G are available at hour 0 on one continuous machine.
Their processing times are 4, 2, and 5 hours; their due times are 6, 8, and 10 hours respectively.
Use sequence E, F, G with no setup or preemption.
**Deliverable:** a table of starts, finishes, signed lateness, tardiness, and mean flow time.
**Acceptance:** show all three intervals and correctly identify late jobs, units, and the completion boundary.
A finish exactly at the due time is on time.

### Exercise 3 — Compare priorities

**Type:** application and critique.
**Time:** 15 minutes.
Use the four base jobs A, B, C, and D from Section 6 without maintenance.
Compare SPT with EDD for mean flow, total tardiness, late-job count, and maximum lateness.
**Deliverable:** a four-metric comparison and a recommendation for a manager who values minimizing the worst delay.
**Acceptance:** use the numerical values from the interval ledger and explain why the other rule may still be preferred under a different objective.
State whether the recommendation is a general theorem or instance-specific.

### Exercise 4 — Apply Johnson’s rule

**Type:** application.
**Time:** 20 minutes.
Three jobs U, V, and W each visit machine 1 then machine 2.
Their processing times on machines 1 and 2 are U: (2, 7), V: (5, 3), and W: (4, 6), in hours.
All jobs are ready at zero; both machines are continuous and the buffer is unlimited.
**Deliverable:** a Johnson sequence and start/finish intervals for both machines.
**Acceptance:** identify each placement, enforce precedence and non-overlap, and calculate makespan.
Explain what changes if the buffer cannot hold waiting work.

### Exercise 5 — Check a job-shop proposal

**Type:** application and diagnosis.
**Time:** 15 minutes.
X needs machine 1 for four hours and then machine 2 for three hours.
Y needs machine 2 for five hours and then machine 1 for two hours.
A proposal schedules X1 [0,4), X2 [4,7), Y1 [0,5), Y2 [5,7).
**Deliverable:** identify any violation, repair the schedule, and compute its makespan.
**Acceptance:** check both job precedence and machine overlap, use half-open intervals, and justify whether the repair is makespan-optimal under the stated assumptions.

### Exercise 6 — Identify hard constraints

**Type:** critique.
**Time:** 15 minutes.
A plan puts a six-hour non-preemptive lot on a machine from hour 7 to 13.
The machine is down from hour 10 to 12; the lot's material is usable from hour 8; one qualified operator is available only from hour 9 to 17.
Assume the operator is required throughout processing and the machine is otherwise continuously available.
**Deliverable:** enumerate violations and propose the earliest feasible start and finish.
**Acceptance:** check all three resource/readiness constraints, do not split the lot, and distinguish a feasible slot from an approved overtime decision.
If no additional availability beyond hour 17 is assumed, state the resulting limit.

### Exercise 7 — Design disruption response

**Type:** design.
**Time:** 20 minutes.
At hour 5 a bottleneck fails, with an uncertain repair time between two and six hours.
One job has already started on a different resource; two committed orders are at risk.
**Deliverable:** a short event-to-decision workflow, including scenario versions and an escalation message.
**Acceptance:** protect safety and in-process constraints, use at least two repair scenarios, identify decision authority and customer communication, and quantify schedule changes.
Do not present the repair estimate as a confirmed restart.

### Exercise 8 — Evaluate a proposed optimizer

**Type:** design and critique.
**Time:** 25 minutes.
A vendor reports a 15% improvement in makespan on one undisclosed schedule and proposes automatic publication.
**Deliverable:** a validation plan and a release decision checklist.
**Acceptance:** specify a feasible common baseline, input snapshot, hard-constraint checks, service and stability metrics, repeatable evaluation, and human authority.
Distinguish a mathematical optimality gap from a real-world improvement estimate.
List at least two ways the model can be feasible mathematically but unusable on the floor.

## 13. Self-check answers

### Answer 1 — Loading versus scheduling

Ten required machine-hours minus eight available machine-hours equals a two-machine-hour overload.
The weekly or daily total does not say which job will occupy which interval.
A finite schedule would reveal which job is late and whether either job can fit around a closure.
The supervisor can confirm machine and labor options; the planner can propose a sequence.
The customer-facing owner handles any change to a customer commitment.
Overtime or another machine is a conditional option, not an assumed fact.

### Answer 2 — E, F, G

| Job | Start, hours | Finish, hours | Due, hours | Lateness, hours | Tardiness, hours |
|---|---:|---:|---:|---:|---:|
| E | 0 | 4 | 6 | -2 | 0 |
| F | 4 | 6 | 8 | -2 | 0 |
| G | 6 | 11 | 10 | 1 | 1 |

The intervals are [0,4), [4,6), and [6,11).
G is the only late job, by one hour.
The completion sum is `4 + 6 + 11 = 21 job-hours`.
Since all releases are zero, mean flow is `21 / 3 = 7 hours per job`.
Total tardiness is one job-hour.

### Answer 3 — SPT versus EDD

SPT has mean flow 10.75 hours/job, total tardiness 8 job-hours, one late job, and maximum lateness 8 hours.
EDD has mean flow 11 hours/job, total tardiness 4 job-hours, two late jobs, and maximum lateness 2 hours.
Choose EDD for the stated goal of reducing the worst delay in this example.
Under the stated classic assumptions EDD minimizes maximum lateness, not necessarily total tardiness or late-job count.
SPT is preferable on mean flow and late-job count for these data.
A different service objective can therefore change the recommended policy.

### Answer 4 — U, W, V

The smallest processing time is U's two hours on machine 1; place U first.
The next smallest is V's three hours on machine 2; place V last.
W occupies the remaining middle position.

| Job | Machine 1, hours | Machine 2, hours |
|---|---|---|
| U | [0,2) | [2,9) |
| W | [2,6) | [9,15) |
| V | [6,11) | [15,18) |

The makespan is 18 hours.
W waits from hour 6 until hour 9 for machine 2.
V waits from hour 11 until hour 15.
If no waiting buffer exists, the stated schedule may block machine 1 and must be rebuilt under a blocking constraint.

### Answer 5 — Job-shop repair

X2 [4,7) overlaps Y1 [0,5) on machine 2 from hour 4 to 5.
The proposed job precedences themselves are satisfied, but the machine constraint is not.
Move X2 to [5,8); retain X1 [0,4), Y1 [0,5), and Y2 [5,7).
Both routes and both machine capacities are then respected.
Makespan is eight hours.
Machine 2 requires eight hours of total work, so no feasible schedule can finish before hour 8.
The repair reaches that lower bound and is optimal for makespan under the stated continuous-availability assumptions.

### Answer 6 — Calendar, material, and operator

The proposed start at hour 7 is earlier than material release at hour 8 and operator availability at hour 9.
The proposed processing interval also overlaps maintenance from hour 10 to 12.
Starting at hour 9 would still overlap maintenance, so the first uninterrupted machine slot begins at hour 12.
A six-hour operation would finish at hour 18.
The stated operator is available only through hour 17, so [12,18) is not feasible with that operator.
No feasible six-hour interval exists inside the stated operator window.
A feasible 12-to-18 interval requires confirmed qualified coverage through hour 18 and any needed authorization.
Do not call it approved overtime until the responsible owner agrees.

### Answer 7 — Disruption-response rubric

A strong response records the failure event and protects people and any work already in process.
It constructs at least a two-hour-repair and a six-hour-repair scenario, clearly labeled conditional.
It checks resource calendars, precedence, materials, and any requalification before dispatch.
It reports which commitments are at risk and how much each proposed start or sequence moves.
It assigns maintenance to update repair estimates, the supervisor to confirm shop feasibility, and the planner to compare options.
It assigns the customer-facing owner to communicate an authorized change.
It republishes a version only after the designated authority approves the action.
A weak response merely reruns an optimizer or assumes the shortest repair estimate is certain.

### Answer 8 — Optimizer-validation rubric

A strong response asks for a timestamped common data snapshot and an executable incumbent baseline.
It independently checks machine, labor, material, routing, calendar, setup, and quality constraints.
It reports makespan alongside late commitments, flow, overtime, setup, and schedule changes.
It uses multiple representative historical or prospective instances and records overrides and causes.
It separates a solver's model-based bound from observed operational improvement.
It requires designated supervisor and planner approval before publication and defines a rollback path.
An omitted fixture and stale material status are two examples of mathematical feasibility without operational feasibility.
A weak response accepts one unexplained percentage or grants automatic dispatch solely because a solver reports a small gap.

## 14. Interview questions

Each question checks reasoning rather than recognition of a software brand.

### 1. What makes a schedule finite?

A strong answer explains time-indexed capacity, non-overlap or cumulative limits, and precedence.
It distinguishes modeled feasibility from completeness of the real-world model.
A weak answer says only that the schedule covers a fixed number of days.
Probe for a shared tool or operator that can invalidate a machine-only plan.

### 2. How do SPT and EDD differ?

A strong answer states their priority keys and the assumptions behind their classic single-machine results.
It distinguishes mean completion from maximum lateness and does not claim EDD always minimizes total tardiness.
A weak answer claims one rule is always best for every business objective.
Probe with the A–D comparison and ask which customer gets a different result.

### 3. How do you handle a machine outage?

A strong answer places the outage on the correct calendar and checks whether operations are preemptible.
It compares feasible alternatives and identifies the approval for maintenance or overtime changes.
A weak answer simply deletes the outage to retain the old finish times.
Probe for a job that does not fit before the outage.

### 4. When can you use Johnson’s rule?

A strong answer describes the two-machine common route, deterministic times, availability, and makespan objective.
It explains earliest placement for a minimum on machine 1 and latest placement for a minimum on machine 2.
A weak answer applies the rule unchanged to arbitrary job-shop routes or sequence-dependent cleaning.
Probe for blocking or different release times.

### 5. How would you verify a job-shop plan?

A strong answer checks each operation's predecessor and each machine's occupied intervals.
It also checks release, calendar, auxiliary resource, and approval constraints.
A weak answer inspects makespan alone.
Probe for a schedule with correct job order but a machine overlap.

### 6. Why might an idle machine be rational?

A strong answer mentions waiting for a required material, a future priority, a non-preemptive slot, or a protected maintenance window.
It weighs the decision against service and capacity implications.
A weak answer insists any idle minute is automatically wasteful.
Probe whether idle time is forced by the selected sequence or by the whole feasible problem.

### 7. How do you model setup?

A strong answer distinguishes run time, setup time, predecessor state, and setup crew capacity.
It checks whether the time is already included in the standard and whether anticipatory setup is permitted.
A weak answer adds the same average setup twice or ignores transition direction.
Probe how initial machine state changes the first job's setup.

### 8. How should a schedule be revised?

A strong answer defines triggers, a frozen zone, exception authority, version control, and communication.
It balances recovery benefits against the cost of changing already released work.
A weak answer globally resequences after every minor event without involving the floor.
Probe what happens when a job has already started.

### 9. What does a solver optimality gap mean?

A strong answer connects the incumbent objective with a valid bound under the solver's reporting convention.
It says the gap is conditional on the encoded data, objective, and constraints.
A weak answer treats a small gap as proof of accurate processing times or customer benefit.
Probe how an omitted fixture can invalidate the physical result.

### 10. How do you evaluate a scheduling improvement?

A strong answer compares feasible schedules on common snapshots and reports service, flow, cost, and instability.
It uses repeated representative instances and tracks actual overrides.
A weak answer compares an infeasible old plan with a feasible new one on one favorable day.
Probe for the difference between forecast demand and an accepted order.

## 15. Key terms

Use the linked section to revisit the operational definition or example.

- **Availability** — resource time permitted by a calendar; see [calendar mechanics](#step-5--build-finite-calendars).
- **Bottleneck** — resource constraining system throughput; see [disruption exercise](#exercise-7--design-disruption-response).
- **Calendar** — time windows when a resource can work; see [definitions](#4-definitions-and-vocabulary).
- **Commitment** — authorized obligation, not a forecast; see [decision boundary](#step-1--define-the-decision-boundary).
- **Completion time** — end of the specified job boundary; see [worked example](#6-worked-example).
- **Critical ratio** — remaining time to due divided by remaining processing; see [rule selection](#rule-selection-is-conditional).
- **Dispatch rule** — local selection criterion for the eligible queue; see [dispatch mechanics](#step-11--apply-dispatch-rules-where-appropriate).
- **Due date** — required finish at an explicitly defined boundary; see [order data](#orders-and-commitments).
- **EDD** — earliest-due-date priority; see [EDD example](#edd-d-b-a-c).
- **Feasibility** — satisfaction of modeled hard constraints; see [independent checks](#step-13--check-every-interval-independently).
- **FCFS** — first-come-first-served queue priority; see [FCFS example](#fcfs-a-b-c-d).
- **Flow shop** — jobs share an order of machines; see [flow-shop example](#two-machine-flow-shop).
- **Flow time** — completion minus release; see [worked example](#6-worked-example).
- **Frozen zone** — period with controlled schedule changes; see [monitoring](#step-16--monitor-and-repair).
- **Job shop** — jobs can have different routes; see [job-shop example](#two-route-job-shop).
- **Johnson's rule** — two-machine makespan sequencing procedure; see [rule selection](#rule-selection-is-conditional).
- **Lateness** — signed completion minus due time; see [definitions](#4-definitions-and-vocabulary).
- **Makespan** — elapsed time to final completion from the stated origin; see [flow-shop example](#two-machine-flow-shop).
- **Non-preemption** — no interruption after operation starts; see [maintenance scenario](#add-maintenance-and-test-non-preemption).
- **Operation** — a job step using required resources; see [precedence](#step-3--establish-operation-dependencies).
- **Optimality gap** — solver-reported difference between incumbent and bound; see [interview questions](#9-what-does-a-solver-optimality-gap-mean).
- **Precedence** — requirement that a step follows its predecessor; see [job-shop example](#two-route-job-shop).
- **Processing time** — operation duration excluding separately modeled setup; see [definitions](#4-definitions-and-vocabulary).
- **Release time** — earliest allowed start; see [readiness](#step-6--determine-readiness-and-release).
- **Routing** — operation and eligibility structure for a product; see [routing data](#operations-and-routing-revisions).
- **Sequence-dependent setup** — transition duration dependent on predecessor and successor; see [time estimates](#step-7--estimate-run-and-transition-times).
- **Slack** — due time less current time and remaining work; see [definitions](#4-definitions-and-vocabulary).
- **SPT** — shortest-processing-time priority; see [SPT example](#spt-d-b-c-a).
- **Tardiness** — nonnegative part of lateness; see [worked example](#6-worked-example).
- **WIP** — work released but not completed; see [definitions](#4-definitions-and-vocabulary).

## 16. Sources

Theoretical claims should be read with the assumptions of the cited model.
Company and software pages describe practices or tools; they do not independently prove local results.
All web sources below were identified for this lesson as references; consult the current source for updated details.

1. Johnson, S. M. (1954). “Optimal two- and three-stage production schedules with setup times included.” *Naval Research Logistics Quarterly*, 1(1), 61–68. [https://doi.org/10.1002/nav.3800010110](https://doi.org/10.1002/nav.3800010110).
   Foundational flow-shop sequencing result; check the assumptions before applying it to a real line.
2. Smith, W. E. (1956). “Various optimizers for single-stage production.” *Naval Research Logistics Quarterly*, 3(1–2), 59–66. [https://doi.org/10.1002/nav.3800030106](https://doi.org/10.1002/nav.3800030106).
   Classic single-machine sequencing results and their objective-dependent assumptions.
3. Jackson, J. R. (1955). “Scheduling a production line to minimize maximum tardiness.” *Management Science Research Project*, Research Report 43, University of California, Los Angeles.
   Classic earliest-due-date result for minimizing maximum lateness in its single-machine setting.
4. Google. [Job shop scheduling, OR-Tools documentation](https://developers.google.com/optimization/scheduling/job_shop).
   Official example of interval, precedence, and machine no-overlap modeling; not an industrial outcome study.
5. Toyota Motor Corporation. [Toyota Production System](https://global.toyota/en/company/vision-and-philosophy/production-system/).
   Company explanation of just-in-time and production leveling; company-authored evidence.
6. International Society of Automation. [ISA-95 standard: Enterprise-Control System Integration](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard).
   Official overview of the ISA-95 series and its integration context.
7. International Electrotechnical Commission. [IEC 62264-1: Enterprise-control system integration — Part 1](https://webstore.iec.ch/en/publication/6675).
   Official catalog entry for the related enterprise-control integration standard; consult the current edition.
8. United States Government Publishing Office. [21 CFR Part 211: Current Good Manufacturing Practice for Finished Pharmaceuticals](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-211).
   Current U.S. regulatory text for covered finished pharmaceuticals; applicability depends on the product and activity.

## 17. Links

- [Module 15 index](./_index.md) — topic map and learning sequence.
- [Manufacturing master data](02_manufacturing-master-data.md) — orders, routings, resources, and calendars.
- [MRP explosion](04_mrp-explosion.md) — material requirements and order timing.
- [Capacity requirements](05_capacity-requirements.md) — resource load and aggregate capacity reconciliation.
- [Module 15 survey](../../../course/15_manufacturing_production_planning.md) — the end-to-end production-planning context.
- [Production planning labs survey](../../../course/30_simulation_optimization_production_labs.md) — simulation and optimization practice.
