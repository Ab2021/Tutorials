# Sequence-Dependent Changeovers and SMED

> Module 15 — Manufacturing and Production Planning
> Topic 08 of 11
> Estimated study time: 4 hours
> Prerequisites: topics 02, 05, 06 and 07
> Version 2026.1

## 1. Why this topic exists

A production schedule is not a list of independent jobs.
A line that makes a dark, allergen-containing product before a light, allergen-free product may need a long, validated clean.
The reverse transition can take less time.
The difference changes whether a delivery promise is feasible.

A scheduler needs the current product, the next product, a work-center calendar, labor and cleaning resources, and an approved transition recipe.
A planner also needs lot sizes and due dates.
A plant manager needs the economic trade-off between fewer changeovers and higher inventory, delay, or lost sales.
A quality manager needs proof that a shorter changeover does not bypass sanitation, inspection, or release.
Maintenance needs a safe method for isolating energy and restoring the equipment.

A single average setup time can make a capacity plan look feasible while the detailed schedule fails.
It can also hide a valuable improvement: one frequently used transition may consume most of the loss.
Improving that transition has different value from improving an infrequent one by the same number of minutes.
On a constrained line, recovered time can enable more throughput; on an unconstrained line, it may simply increase idle time.
The business benefit must be demonstrated rather than assumed.

Single-minute exchange of die (SMED) is a structured method for reducing changeover time.
The name means a single-digit number of minutes, not a guarantee or a reason to override quality and safety.
The method distinguishes work requiring the machine to stop from work that can safely be performed while it runs.
It converts eligible work to external work and then streamlines what remains.
Scheduling economics determines whether the resulting capability should change sequence, lot size, staffing, or customer commitments.

This topic teaches both sides of the decision.
You will model directed transitions, measure the actual event, evaluate improvement without double counting, and put the new standard under control.
The numerical examples are illustrative, not a published plant result.

## 2. Learning outcomes

After this topic you can:

1. Define a changeover boundary and distinguish internal and external activities; demonstrate this in Exercise 1.
2. Build and interpret a directed setup matrix, including initial state and missing transitions; demonstrate this in Exercises 2 and 6.
3. Calculate job completion, tardiness, and setup time for a sequence on a finite resource; demonstrate this in Exercise 2.
4. Apply SMED separation, conversion, and streamlining without bypassing safety or quality controls; demonstrate this in Exercise 3.
5. Compare scheduling alternatives using explicit incremental costs and avoid double counting; demonstrate this in Exercises 2 and 4.
6. Assess whether capacity recovered at a bottleneck is monetizable; demonstrate this in Exercise 4.
7. Design event data and a controlled rollout with owners and acceptance criteria; demonstrate this in Exercises 5 and 6.
8. Explain uncertainty, cleaning validation, and the limits of a small deterministic schedule; demonstrate this in Exercises 7 and 8.

## 3. The core mental model

```text
Demand and due dates    Current state and approved transition rules
         |                         |
         +--------> finite scheduler <--------- resource calendars
                            |
                 ordered jobs + setup tasks
                            |
               execute, timestamp, inspect
                            |
          accepted output + deviations + actual loss
                            |
          SMED study -> safer standard -> validated matrix
                            |                    |
                            +---- economics -----+
```

A transition is a directed edge between two states.
A product, recipe, tooling, sanitation class, or packaging format may be part of a state.
The edge from A to B need not match the edge from B to A.
Some edges are prohibited, not merely expensive.
An initial state gives the first job its incoming edge.
The schedule consumes production time and transition time on the same constrained machine unless the process model states otherwise.

A changeover contains multiple activities.
Some must happen while production is stopped: internal work.
Others can happen before the last good unit or after the first good unit while production continues: external work.
The classification depends on the actual safety and quality rules, not on the activity name.
External does not mean free: a technician, staging area, transport, or tool may be scarce.
A one-machine model can therefore overstate feasibility if it ignores those resources.

SMED is an improvement cycle, not a sorting rule.
First measure and separate the activities.
Then convert eligible internal work to external work.
Then simplify, parallelize safely, or eliminate remaining work.
Trial, validate, standardize, and monitor the result.
The scheduler can only rely on the new times after the process owner has approved them.

The economic loop closes when the plant uses the recovered capability.
A shorter transition might support an earlier delivery, more capacity for profitable demand, or smaller lots.
It might instead create idle time.
The value depends on the binding constraint and the alternative use of the released resources.
Count the causal benefit once, net of incremental costs and investment.

## 4. Definitions and vocabulary

| Term | Meaning | Common misuse | Unit or representation |
|---|---|---|---|
| Changeover | Transition from one production state to another | Only wrench time is included | minutes per event |
| Setup | Activities and time needed to prepare the next run | Assuming it is independent of sequence | minutes per directed edge |
| State | Attributes that determine a transition | Treating an item ID as the only relevant attribute | recipe, allergen, tooling and version |
| Directed matrix | Table indexed by prior and next state | Assuming symmetry | minutes or prohibited |
| Initial state | Verified machine condition before the first scheduled job | Starting every horizon from a clean machine | state and timestamp |
| Internal activity | Work requiring the machine to be stopped | Classifying by job title rather than machine condition | minutes |
| External activity | Work safely performed while the machine produces | Assuming no labor or space is needed | minutes and resources |
| SMED | Single-minute exchange of die improvement method | Promise that every transition is under ten minutes | method |
| Last good unit | Last accepted unit of the outgoing run | Last unit discharged regardless of quality | timestamp |
| First good unit | First incoming unit meeting defined acceptance | First unit started without release | timestamp |
| Startup loss | Scrap, speed loss or delays during stabilization | Counting every slow minute as a separate setup cost as well | units, minutes, currency |
| Tardiness | Positive part of completion less due time | Treating early completion as negative tardiness | minutes |
| Lateness | Completion minus due time, possibly negative | Equating it with tardiness | minutes |
| Makespan | Elapsed time from schedule start to last completion | Ignoring an initial setup or downtime | minutes |
| Bottleneck | Resource constraining system throughput over the relevant horizon | Calling every busy machine a bottleneck | resource and period |
| Constraint | Limiting resource, policy, or material | Assuming it remains fixed after improvement | documented condition |
| Holding cost | Incremental cost of carrying inventory | Charging the entire inventory value each period | currency per unit per period |
| Contribution | Incremental revenue less relevant variable cost | Treating revenue as profit | currency per accepted unit |
| Opportunity cost | Value of the best feasible alternative forgone | Adding it to a benefit already representing that alternative | currency per decision horizon |
| Cleaning validation | Evidence that a cleaning process consistently meets defined requirements | Treating a quick visual check as universally sufficient | approved protocol and results |
| Change control | Formal review and authorization of process changes | Updating a planning matrix before approval | versioned record |
| Setup crew | People needed for a transition | Assuming they are always available | people and calendar |
| Sequence-dependent cost | Incremental cost depending on the ordered pair | Setting it proportional to time without evidence | currency per edge |
| Anticipatory setup | Setup allowed before the job or material is ready | Assuming this is always safe or valid | timing rule |
| Release | Authorization for output to proceed | Treating completion of setup as automatic release | status and timestamp |
| Frozen horizon | Window in which changes require controlled exception | Assuming the schedule can be freely rearranged | time window |

## 5. The mechanics, step by step

### Step 1 — Frame the decision

**Input:** demand, customer dates, capacity outlook, constraints, and the planning horizon.
**Output:** a specific decision and a baseline against which to compare it.

1. Identify the production resource and its normal unit of scheduling.
2. Specify the horizon, time zone, shift calendar, and start timestamp.
3. List the jobs to be sequenced and whether splitting is permitted.
4. Record which due dates are customer commitments versus internal targets.
5. Separate the demand forecast from the production plan and from an actual customer commitment.
6. Identify mandatory quality, safety, maintenance, and labor constraints.
7. Decide whether the decision is sequencing existing jobs, reducing setup time, resizing lots, or a combination.
8. Choose the comparison horizon long enough to include downstream inventory and changeover effects.
9. Define who can accept a late job, authorize a change, or reject an unsafe transition.
10. Keep the baseline schedule and the inputs used to produce it.

A meaningful baseline is the best feasible current alternative, not necessarily the sequence used last month.
If the problem is an investment, include the transition into and out of the comparison period.
Otherwise a sequence can appear cheap by shifting its last expensive clean beyond the horizon.

### Step 2 — Define the physical boundary

**Input:** equipment map, production records, quality requirements, and operator observation.
**Output:** an agreed event start and end plus classifications of included losses.

Use last accepted output from the outgoing run to first accepted output from the incoming run as one useful gross boundary.
Record whether acceptance is instantaneous or retrospective.
If laboratory release takes hours, record physical first conforming output and release time separately.
The correct boundary depends on the decision: machine occupancy and customer-available output may differ.

Mark the last outgoing unit, stop, isolation, tooling change, sanitation, verification, restart, first incoming unit, and final release.
Record waiting for tools, people, materials, laboratory results, and decisions.
Use synchronized clocks and preserve the source timestamp.
Distinguish actual production loss from a break that was already planned and unavoidable.
Do not delete an inconvenient delay; code its cause.

If a unit is later rejected, revise the accepted-output boundary under a documented rule.
Keep the originally observed record and the correction history.
If the operation is continuous, define an equivalent transition boundary based on the verified specification window.
Make the boundary stable across before-and-after comparisons.

### Step 3 — Observe the work

**Input:** representative changeovers, authorized observation, event logs, and a process map.
**Output:** a time-ordered activity chart with resource and dependency information.

Observe several shifts and both transition directions when they differ.
Explain the purpose to operators and obtain required consent for recording.
Do not infer individual performance from a short observation.
Use a video or stopwatch only under site policy.
Record each activity's start, finish, worker, tools, machine state, and prerequisite.
Record overlaps rather than summing all labor minutes as elapsed machine time.

Ask the people doing the work why the current step exists.
A search for a tool may conceal a storage or ownership problem.
A long inspection may be required by a validated process even if it looks like waiting.
A repeated adjustment may indicate an unstable fixture, not a slow operator.
Record exceptions, missing materials, and downstream interruptions separately from the standard method.

### Step 4 — Separate internal from external work

**Input:** the observed activity chart and safety and quality constraints.
**Output:** a classified chart and a safe external-preparation plan.

For each activity ask whether the equipment must be stopped for that work to occur safely and correctly.
Preparing an already approved tool in a safe staging area can often be external.
Removing a guard or changing tooling in a hazardous zone generally requires the applicable isolation procedure.
Verification may have to occur after a specific step and cannot simply be moved earlier.
Do not assume a cleaning task can run concurrently with production.

Prepare parts, documents, clean tools, labels, and personnel before the stop where authorized.
Make readiness visible before releasing the outgoing job to finish.
Reserve the staging area and confirm correct versions.
If preparation is incomplete, escalate before the transition rather than hiding the delay in the line stop.
Account for labor that is pulled away from another operation.

### Step 5 — Convert, streamline, and validate

**Input:** classified activities, engineering constraints, and approved change process.
**Output:** a proposed method with a test plan and a risk assessment.

First convert genuinely movable work to external work.
Examples include preparing fixtures offline or presetting an approved adjustment on a duplicate fixture.
Then reduce the remaining internal steps through engineered locating features, quick-release devices, or better access.
Parallel work may shorten elapsed time if the people and space are available and the work is safe.
Its labor minutes can increase even while machine minutes fall.

Do not eliminate an inspection merely because it is slow.
A proposed replacement must be demonstrated to satisfy the same requirement and approved by the responsible function.
Verify ergonomic access, guarding, stored energy, chemical compatibility, and sanitation as applicable.
Write a trial procedure, test worst-case transitions, and define acceptance before the trial.
Track scrap and first-pass quality as well as elapsed time.
A faster transition that raises contamination or product loss is not automatically an improvement.

### Step 6 — Build the transition matrix

**Input:** approved states, measured events, engineering rules, and the effective calendar.
**Output:** a versioned directed matrix with confidence and ownership.

Rows identify the prior state and columns the next state.
Store units explicitly, typically elapsed machine minutes per event.
Add the observed sample size, date range, equipment, method version, and chosen planning statistic.
A median represents a typical event but is not a guarantee of meeting a deadline.
Use a conservative planning time or explicit variability if the service objective requires it.

A diagonal may be zero if consecutive lots require no intervening action.
A mandatory inspection or batch clean can make a same-state transition nonzero.
Represent a forbidden transition explicitly, not as zero and not as an arbitrarily large time that could still be selected.
An unmeasured transition is unknown, not zero.
Route an unknown through engineering review, a validated conservative rule, or a known intermediate state.

For each directed edge document the sanitation rule, tooling requirement, labor need, and other scarce resources.
Include the starting state at the beginning of the horizon.
If the equipment has been idle, store the actual condition and expiry of its clean status.
A family-level matrix is acceptable only if all included products follow the same approved transition rule.

### Step 7 — Construct feasible schedules

**Input:** jobs, processing durations, release times, due dates, initial state, matrix, and resource calendars.
**Output:** a feasible ordered schedule with setup and production intervals.

Start from the actual available time and the verified current state.
For each job, determine its incoming transition.
Determine whether setup can begin before material release and whether any expiration rule applies.
Place the setup in a continuous or interrupted interval according to the real procedure.
Respect the machine, crew, tool, cleaning station, and maintenance calendars.
Then place production without overlap on the same machine.

For the simplified case with all jobs and resources available at time zero, the recurrence is:

```text
completion[k] = completion[k-1] + setup[state[k-1], state[k]] + processing[k]
tardiness[k]  = max(0, completion[k] - due[k])
```

Set `completion[0]` to the machine available time and `state[0]` to its verified initial state.
If job release is required before setup, use the later of the prior completion and job release as the setup start.
If setup can be anticipatory, a different recurrence is required and must model setup expiry and resource availability.
Insert calendar downtime explicitly; elapsed time and working time are not interchangeable.

Calculate completion, tardiness, setup minutes, resource conflicts, and the final state for every candidate.
A transition-minimizing sequence is not necessarily the sequence minimizing tardiness or total economic cost.
For a few jobs, enumerate feasible permutations.
For larger problems use a documented heuristic or solver, with the same feasibility checks.
A solver's objective value is not evidence that its inputs or operational constraints are correct.

### Step 8 — Evaluate economic alternatives

**Input:** feasible alternatives, decision horizon, incremental cost rates, and demand evidence.
**Output:** a comparison of relevant costs and benefits, including uncertainty.

Include setup labor and consumables if they actually change.
Include lost contribution only when demand exists, the relevant constraint binds, and the time can produce accepted saleable units.
Include overtime, expediting, quality loss, inventory carrying, and contractual delay charges when incremental and supported.
Subtract the cost of extra preparation, equipment, validation, and maintenance.
A fixed salary is not necessarily cash saved when minutes are released.

Choose one consistent basis for valuing a capacity minute.
If the setup-hour rate already includes lost contribution, do not add the same throughput benefit again.
If additional output is counted, do not also claim the identical time as avoided opportunity cost.
If tardiness penalties proxy customer loss, do not add the same customer loss again without separating the underlying mechanisms.
Keep transfers between departments separate from whole-business benefit.

For an investment, compare recurring net benefit against installation and ongoing costs.
Simple payback is investment divided by annual net benefit only when that benefit is positive and stable.
For a longer project, discounted cash flow is more informative; state the discount rate and timing.
Report a downside case with lower utilization or lower demand.
If the constraint moves elsewhere, recalculate the benefit.

### Step 9 — Pilot, control, and learn

**Input:** approved method, pilot plan, process owners, baseline, and acceptance criteria.
**Output:** validated standard work and controlled planning parameters or a rejected proposal.

Assign an owner for safety, quality, production, maintenance, scheduling, and the matrix record.
Pilot representative shifts, products, and operators.
Use predefined criteria for safety, quality, elapsed time, repeatability, and resource availability.
Record all trial results, including failures and rework.
Keep the old approved method available until the new one is authorized.

After approval, train affected staff and update work instructions, tooling, planning data, and the effective date together.
Audit the new standard in actual production.
Track transition distributions and invalid or missing matrix edges.
Investigate drift before changing the planning parameter.
Revalidate after material, equipment, formulation, sanitation, or regulatory changes.

## 6. Worked example

### Decision and assumptions

A packaging line has three orders available at minute 0 of a shift.
The current verified state is A.
One line is available continuously, and one setup crew is available whenever needed.
There are no release delays, planned breaks, startup scrap, or setup expiry in this example.
All three orders must be completed without splitting.
The schedule starts at minute 0; due times are minutes after that start.

The demand forecast generated the need for the orders, but the three job quantities and times below are the production plan.
The due times are the commitments used in this illustrative schedule, not a claim that a customer has actually contracted them.
The setup matrix is approved and contains no forbidden transitions.
The diagonal is zero because no intervening batch clean is required in this scenario.

| Job and state | Processing time | Due time |
|---|---:|---:|
| A | 120 min | 150 min |
| B | 90 min | 240 min |
| C | 105 min | 330 min |

| Prior state | Next A | Next B | Next C |
|---|---:|---:|---:|
| A | 0 min | 35 min | 55 min |
| B | 50 min | 0 min | 20 min |
| C | 25 min | 45 min | 0 min |

A to B requires 35 min, while B to A requires 50 min.
C to B requires 45 min, while B to C requires 20 min.
The direction matters.
The setup-hour charge is USD 120 per machine-hour, equal to USD 2 per setup minute.
The illustrative late-delivery charge is USD 2 per tardy job-minute.
These are distinct assumed incremental charges; neither includes the other.
Processing costs are identical across sequences and are omitted from this comparison.
The charge is not asserted to be a market price or contractual rule.

### Candidate A, B, C

| Operation | Start | Finish | Due | Tardiness |
|---|---:|---:|---:|---:|
| A setup | 0 min | 0 min | — | — |
| A production | 0 min | 120 min | 150 min | 0 min |
| A to B setup | 120 min | 155 min | — | — |
| B production | 155 min | 245 min | 240 min | 5 min |
| B to C setup | 245 min | 265 min | — | — |
| C production | 265 min | 370 min | 330 min | 40 min |

Total setup is 0 + 35 + 20 = 55 min.
Total tardiness is 0 + 5 + 40 = 45 job-min.
The schedule ends at minute 370.
Setup charge is 55 min × USD 2/min = USD 110.
Tardiness charge is 45 job-min × USD 2/job-min = USD 90.
Total relevant charge is USD 200 for this set of three jobs.

### Candidate A, C, B

A finishes at minute 120.
A to C takes 55 min, so C finishes at minute 280.
C to B takes 45 min, so B finishes at minute 415.
A and C are on time; B is 175 min late.
Setup totals 100 min and costs USD 200.
Tardiness costs 175 min × USD 2/min = USD 350.
The relevant total is USD 550, or USD 350 more than A, B, C.
The cheapest setup sequence among these two is also less late, but that need not hold in other demand situations.
The code below enumerates all six sequences rather than claiming this pair proves global optimality.

### SMED intervention

An approved fixture and preparation change reduces B to C from 20 min to 12 min.
It does not change A to B, production speed, the job quantities, or the quality acceptance rule.
For A, B, C, A still completes at minute 120 and B at minute 245.
The new B to C setup ends at minute 257.
C completes at minute 362 and is 32 min late.
Total setup is 47 min; total tardiness is 5 + 32 = 37 job-min.
The new setup charge is USD 94 and the tardiness charge is USD 74.
The total is USD 168, a reduction of USD 32 for this particular three-job schedule.

The 8 min saved has two modeled effects: USD 16 less setup charge and USD 16 less tardiness charge.
They are additive only because this example defines them as separate costs.
If the setup charge already represented the entire financial impact of late deliveries, adding the tardiness charge would double count.
The improvement is a planning opportunity, not a customer promise until the trial and schedule are approved.

Suppose this exact schedule recurs 180 times per year and the improvement incurs USD 1,200 per year in incremental preparation cost.
Its illustrative annual net benefit is 180 × USD 32 − USD 1,200 = USD 4,560 per year.
For an upfront investment of USD 9,120, simple payback is USD 9,120 / USD 4,560 per year = 2 years.
This is a scenario calculation, not a forecast: recurrence, rates, and independent cost categories require evidence.
It excludes tax, discounting, equipment residual value, and any change to other sequences.
A complete investment review should compare the best feasible schedules before and after the improvement, not merely hold the sequence fixed.

### Sensitivity: service priorities and directional setups

The following small decision experiments use the same line, processing times, initial state, and setup matrix.
Each scenario evaluates every permutation under its stated due-date and cost assumptions.
The charges are simple illustrative proxies in USD, not actual contractual amounts.
They show how a priority change can alter the preferred sequence and how a fixed sequence responds.
A tie means either listed schedule has the same modeled charge; operational feasibility may still distinguish them.

#### Experiment 1: Baseline commitments

Due times A/B/C are 150/240/330 min; setup costs USD 2/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 45 | 370 | 200 |
| ACB | 100 | 175 | 415 | 550 |
| BAC | 140 | 270 | 455 | 820 |
| BCA | 80 | 245 | 395 | 650 |
| CAB | 115 | 345 | 430 | 920 |
| CBA | 150 | 370 | 465 | 1040 |

The minimum modeled charge is USD 200; the preferred sequence is ABC.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 2: Earlier C commitment

Due times A/B/C are 150/240/270 min; setup costs USD 2/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 105 | 370 | 320 |
| ACB | 100 | 185 | 415 | 570 |
| BAC | 140 | 330 | 455 | 940 |
| BCA | 80 | 245 | 395 | 650 |
| CAB | 115 | 345 | 430 | 920 |
| CBA | 150 | 370 | 465 | 1040 |

The minimum modeled charge is USD 320; the preferred sequence is ABC.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 3: Earlier B commitment

Due times A/B/C are 150/180/330 min; setup costs USD 2/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 105 | 370 | 320 |
| ACB | 100 | 235 | 415 | 670 |
| BAC | 140 | 270 | 455 | 820 |
| BCA | 80 | 245 | 395 | 650 |
| CAB | 115 | 405 | 430 | 1040 |
| CBA | 150 | 430 | 465 | 1160 |

The minimum modeled charge is USD 320; the preferred sequence is ABC.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 4: Later A commitment

Due times A/B/C are 300/240/330 min; setup costs USD 2/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 45 | 370 | 200 |
| ACB | 100 | 175 | 415 | 550 |
| BAC | 140 | 125 | 455 | 530 |
| BCA | 80 | 95 | 395 | 350 |
| CAB | 115 | 195 | 430 | 620 |
| CBA | 150 | 220 | 465 | 740 |

The minimum modeled charge is USD 200; the preferred sequence is ABC.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 5: Higher tardiness exposure

Due times A/B/C are 150/240/330 min; setup costs USD 2/min and tardiness costs USD 5/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 45 | 370 | 335 |
| ACB | 100 | 175 | 415 | 1075 |
| BAC | 140 | 270 | 455 | 1630 |
| BCA | 80 | 245 | 395 | 1385 |
| CAB | 115 | 345 | 430 | 1955 |
| CBA | 150 | 370 | 465 | 2150 |

The minimum modeled charge is USD 335; the preferred sequence is ABC.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 6: Higher setup exposure

Due times A/B/C are 150/240/330 min; setup costs USD 5/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 45 | 370 | 365 |
| ACB | 100 | 175 | 415 | 850 |
| BAC | 140 | 270 | 455 | 1240 |
| BCA | 80 | 245 | 395 | 890 |
| CAB | 115 | 345 | 430 | 1265 |
| CBA | 150 | 370 | 465 | 1490 |

The minimum modeled charge is USD 365; the preferred sequence is ABC.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 7: Simultaneous commitments

Due times A/B/C are 300/300/300 min; setup costs USD 2/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 70 | 370 | 250 |
| ACB | 100 | 115 | 415 | 430 |
| BAC | 140 | 155 | 455 | 590 |
| BCA | 80 | 95 | 395 | 350 |
| CAB | 115 | 135 | 430 | 500 |
| CBA | 150 | 165 | 465 | 630 |

The minimum modeled charge is USD 250; the preferred sequence is ABC.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 8: C priority with later A

Due times A/B/C are 360/300/200 min; setup costs USD 2/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 170 | 370 | 450 |
| ACB | 100 | 195 | 415 | 590 |
| BAC | 140 | 255 | 455 | 790 |
| BCA | 80 | 85 | 395 | 330 |
| CAB | 115 | 130 | 430 | 490 |
| CBA | 150 | 105 | 465 | 510 |

The minimum modeled charge is USD 330; the preferred sequence is BCA.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 9: B priority with later A

Due times A/B/C are 360/180/300 min; setup costs USD 2/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 135 | 370 | 380 |
| ACB | 100 | 235 | 415 | 670 |
| BAC | 140 | 155 | 455 | 590 |
| BCA | 80 | 35 | 395 | 230 |
| CAB | 115 | 250 | 430 | 730 |
| CBA | 150 | 220 | 465 | 740 |

The minimum modeled charge is USD 230; the preferred sequence is BCA.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 10: Loose deadlines

Due times A/B/C are 500/500/500 min; setup costs USD 2/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 0 | 370 | 110 |
| ACB | 100 | 0 | 415 | 200 |
| BAC | 140 | 0 | 455 | 280 |
| BCA | 80 | 0 | 395 | 160 |
| CAB | 115 | 0 | 430 | 230 |
| CBA | 150 | 0 | 465 | 300 |

The minimum modeled charge is USD 110; the preferred sequence is ABC.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 11: Tight deadlines

Due times A/B/C are 120/180/240 min; setup costs USD 2/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 195 | 370 | 500 |
| ACB | 100 | 275 | 415 | 750 |
| BAC | 140 | 390 | 455 | 1060 |
| BCA | 80 | 285 | 395 | 730 |
| CAB | 115 | 435 | 430 | 1100 |
| CBA | 150 | 460 | 465 | 1220 |

The minimum modeled charge is USD 500; the preferred sequence is ABC.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.

#### Experiment 12: Late C commitment

Due times A/B/C are 150/240/500 min; setup costs USD 2/min and tardiness costs USD 2/job-min.

| Sequence | Setup min | Tardiness job-min | Finish min | Relevant cost USD |
|---|---:|---:|---:|---:|
| ABC | 55 | 5 | 370 | 120 |
| ACB | 100 | 175 | 415 | 550 |
| BAC | 140 | 145 | 455 | 570 |
| BCA | 80 | 245 | 395 | 650 |
| CAB | 115 | 345 | 430 | 920 |
| CBA | 150 | 370 | 465 | 1040 |

The minimum modeled charge is USD 120; the preferred sequence is ABC.
The finish time is the final completion, whereas tardiness adds late minutes across jobs.
This result assumes the stated rates and does not override any safety or quality constraint.


## 7. Code, query, or calculation

The following Python 3 program enumerates all six sequences in the worked example.
It assumes all jobs are released at minute 0, one continuously available machine, one always-available setup crew, deterministic directed setups, and no forbidden transitions.
It treats setup and tardiness rates as separate incremental costs.
It reports total tardiness in job-minutes, not elapsed minutes.
It does not model cleanup after the final job, material shortages, calendars, or uncertainty.

```python
from itertools import permutations

processing = {"A": 120, "B": 90, "C": 105}  # minutes
Due = {"A": 150, "B": 240, "C": 330}  # minute from start
setup = {
    "A": {"A": 0, "B": 35, "C": 55},
    "B": {"A": 50, "B": 0, "C": 20},
    "C": {"A": 25, "B": 45, "C": 0},
}  # elapsed minutes
SETUP_USD_PER_MIN = 2
LATE_USD_PER_JOB_MIN = 2


def evaluate(sequence, matrix):
    clock = 0
    previous = "A"  # verified initial state
    setup_minutes = 0
    tardiness = 0
    completion = {}
    for job in sequence:
        transition = matrix[previous][job]
        setup_minutes += transition
        clock += transition + processing[job]
        completion[job] = clock
        tardiness += max(0, clock - Due[job])
        previous = job
    cost = (setup_minutes * SETUP_USD_PER_MIN
            + tardiness * LATE_USD_PER_JOB_MIN)
    return setup_minutes, tardiness, clock, cost, completion


for seq in permutations(processing):
    s, t, finish, cost, _ = evaluate(seq, setup)
    print("".join(seq), s, t, finish, cost)

improved = {state: row.copy() for state, row in setup.items()}
improved["B"]["C"] = 12
print("improved ABC", *evaluate(("A", "B", "C"), improved)[:4])
```

Columns are sequence, setup minutes, total tardiness job-minutes, finish minute, and relevant cost in USD.
The copy prevents the improved matrix from changing the baseline.
Dictionary insertion order gives the displayed sequence order in Python 3.7 and later.

Expected output:

```text
ABC 55 45 370 200
ACB 100 175 415 550
BAC 140 270 455 820
BCA 80 240 395 640
CAB 115 345 430 920
CBA 150 370 465 1040
improved ABC 47 37 362 168
```

The program is a teaching enumeration, not a production scheduler.
If a real edge is missing, reject that candidate and investigate rather than replacing the edge with zero.
For a shift calendar, add intervals for all required resources and calculate a feasible start for each operation.
A setup team shared across machines requires a joint schedule.
When objectives include a real customer promise, check whether monetary penalties capture the contractual and operational consequences.

### Extending the calculation to labor and inventory

The worked example assigns a cost per setup minute.
That assumption is useful for demonstrating an objective, but an actual plant should build the rate from incremental consequences.
The following separate illustration shows why.

Suppose a setup currently needs one technician for 35 min.
The fully incremental technician cost is USD 36 per paid hour.
The labor cost is 35 / 60 h × USD 36/h = USD 21.
A proposed method uses two technicians simultaneously for 15 min.
Its labor cost is 2 × 15 / 60 h × USD 36/h = USD 18.
Machine stop falls by 20 min and paid technician time falls by 5 person-min.
This is not a saving of 20 technician-min.
The hypothetical labor saving is USD 3 per changeover, provided paid hours actually decrease or the freed labor displaces an incremental cost.

Now suppose the second technician was not otherwise available.
If that technician earns an incremental overtime premium, include the premium.
If that technician leaves a different bottleneck unattended, calculate the resulting system impact instead.
If the second technician has idle paid time, there may be no additional cash outlay.
All three answers can be right in different operating conditions.
Use the decision-specific alternative rather than a universal hourly rate.

Next consider a smaller-lot decision made possible by a shorter setup.
Suppose annual demand is 24,000 accepted units and is level throughout a 240-working-day year.
An old policy produces 2,000 units per lot.
It therefore makes 12 lots per year under these simplifying assumptions.
A proposed policy produces 1,000 units per lot and makes 24 lots per year.
Assume the average cycle stock is half the lot size, with no safety stock change.
Old average cycle stock is 1,000 units; new average cycle stock is 500 units.
At an incremental carrying cost of USD 4 per unit-year, the reduction is USD 2,000 per year.

The new policy causes 12 additional changeovers per year.
If an approved setup takes 30 min and has a truly incremental cost of USD 60 per setup, those extra setups cost USD 720 per year.
The simplified net benefit is USD 2,000 − USD 720 = USD 1,280 per year.
The additional machine load is 12 × 30 min = 360 min per year.
That load is feasible only if the applicable machine and crew calendars have room.
If the line is constrained, the opportunity cost of 360 min may exceed the inventory benefit.
If the USD 60 setup cost already includes that opportunity cost, do not add it again.
No service improvement or obsolescence reduction is assumed in this calculation.

Level demand, constant lot size, zero shortages, no safety-stock change, and immediate replenishment are simplifying assumptions.
Seasonality, shelf life, minimum order quantities, yield loss, and batch-release delays can alter the result.
A new policy should be simulated or piloted with actual demand and execution constraints before adoption.
The result is a plan alternative, not authorization to change a committed order.

### Worked observation: elapsed time versus labor time

Suppose the old B to C transition is observed from minute 0 to minute 20.
The following hypothetical activities are recorded on a shared clock.

| Activity | Start | End | Person | Machine status |
|---|---:|---:|---|---|
| Remove outgoing fixture | 0 min | 4 min | Technician 1 | stopped and isolated |
| Clean contact surface | 4 min | 10 min | Technician 1 | stopped |
| Fetch approved incoming fixture | 4 min | 9 min | Technician 2 | stopped |
| Install and align fixture | 10 min | 16 min | Technician 1 | stopped |
| Restart and verify output | 16 min | 20 min | Technician 1 | transition to production |

Elapsed machine transition time is 20 min, not 25 min.
The recorded labor time is 20 + 5 = 25 person-min because two activities overlap.
The incoming fixture must be available before installation at minute 10.
Its retrieval is complete by minute 9, so it does not determine the finish in this observed event.
Simply moving that particular retrieval earlier would reduce work during the stop but would not reduce the observed 20 min elapsed time.
In another event, a late retrieval could become critical; inspect several observations before dismissing the improvement.

Suppose a controlled intervention shortens cleaning from 6 min to 4 min and installation from 6 min to 4 min.
The revised procedure also stages the approved fixture before the stop.
The tasks then occur at 0–4, 4–8, 8–12, and 12–16 min.
The expected elapsed transition is 16 min, a 4 min reduction, if the restart and verification remain 4 min.
This is a hypothetical engineering estimate, not evidence that a specific cleaning step can be shortened.
The responsible quality and safety functions must approve both changes.
The team should also monitor whether earlier staging introduces contamination, mix-ups, or expired clean status.

If a revised fixture instead allowed 8 min of total reduction, the worked example's B to C time would fall from 20 min to 12 min.
The activity breakdown would have to substantiate that additional saving.
Do not use the 16 min illustrative observation as proof of the separate 12 min scheduling assumption.
Both cases use the same concept but describe different proposed methods.

### Operational handoff checklist

Before the schedule is published, the scheduler confirms the current machine state and its effective timestamp.
The scheduler checks that each selected edge is approved for the current recipe, tooling, and sanitation status.
The crew lead confirms labor, tools, and staging space for the planned setup window.
Materials management confirms the incoming order can start when setup finishes.
Quality confirms any required inspection and release step is included.
Maintenance confirms scheduled work and energy-control requirements.
The production owner checks the outgoing and incoming order priorities.
The planner communicates any proposed change to an already committed sequence through the site escalation process.

At execution, the operator records the actual boundary and reason codes.
The team records any unavailable part, missing person, unexpected cleaning, or rejected first unit.
Quality records acceptance and release separately when those events differ.
The scheduler receives the actual finish and revised machine state.
The next job is resequenced only within the authorized decision rights.
An unapproved edge remains unavailable even when the line is late.

After execution, the process owner reviews recurring delay categories.
The data steward checks timestamp corrections and missing transition identifiers.
Finance reconciles claimed savings to actual incremental labor, materials, overtime, or throughput effects.
The engineering owner reviews whether new equipment or a formulation change invalidates the setup standard.
The quality owner evaluates whether changed cleaning or release steps require further evidence.
The scheduler updates future parameters only after the new method is authorized.
This closes the loop from observed loss to a feasible, controlled plan.

### Terminal-state adjustment

The three-job example stops at the final job.
That is appropriate only if the state after the final job has no consequence inside the decision horizon.
Suppose instead that the next shift requires state A before any other work can begin.
The terminal transition must then be included in the horizon or assigned to the next shift consistently.

For A, B, C, the final state is C.
The approved C to A transition takes 25 min.
Adding it after minute 370 makes the machine ready in state A at minute 395.
The setup total becomes 80 min and, at USD 2 per setup-minute, its charge becomes USD 160.
The job tardiness remains 45 job-min because the clean occurs after all three jobs finish.
The relevant charge becomes USD 160 + USD 90 = USD 250, excluding any independent cost of delaying the next shift.

For A, C, B, the final state is B.
The B to A transition takes 50 min.
Adding it after minute 415 makes the machine ready in state A at minute 465.
The setup total becomes 150 min and the setup charge becomes USD 300.
The tardiness charge remains USD 350, so the revised relevant charge is USD 650.
These two comparisons do not establish the best sequence among all six once terminal requirements are included.
Re-enumerate every feasible sequence whenever the horizon or terminal condition changes.

If the next shift actually begins with state C, the appropriate terminal requirement is different.
The scheduler should use the known next job or an explicitly justified terminal policy, not an arbitrary clean-to-A assumption.
If the next job is uncertain, compare plausible scenarios rather than pretending the terminal state has no value.
Document whether terminal cleaning costs were included in the current horizon or the next one, and charge them only once.

## 8. Data, standards, and regulatory context

### Minimum event data

A useful event record includes machine, site, transition ID, outgoing and incoming state, method version, and event start and end.
Capture clock source, time zone, operator or crew identifier where policy permits, and data correction history.
Store discrete activities with start, end, resource, reason code, and overlap information.
Record actual produced quantities, scrap, rework, acceptance and release times, and any deviation.
Link the event to production orders and the approved instruction in effect at that time.

A state key must include attributes relevant to the transition.
A single item code may omit allergen, color, format, equipment, tooling, or clean-status details.
Conversely, a matrix with one row for every commercial item can be unnecessarily large if approved families behave identically.
Document the mapping from item and route to transition family.
Version the mapping and matrix with effective dates.
A retrospective data change should not silently rewrite the planning assumptions used for a historical schedule.

Useful checks include negative or overlapping timestamps, unexplained missing state, impossible sequence, stale method version, and unclosed quality holds.
Compare automated stop events with operator logs and quality records.
Do not equate every machine stop with a changeover.
Missingness can be systematic: the most disrupted transitions may be least well recorded.
Sample them deliberately.
Separate changes in method from changes in mix when reporting trends.

### Interoperability and standards

ISA-95 / IEC 62264 provides models and terminology for integrating enterprise and manufacturing operations information.
Its role here is to help relate equipment, production schedules, material, personnel, and performance information across systems.
It does not prescribe a particular setup matrix, SMED method, or numerical setup time.
The applicable edition and licensed normative text should be checked for a real implementation.

ISO 22400 addresses manufacturing operations management key performance indicators, including definitions and relationships for manufacturing operations.
Use the specific adopted edition and site definitions when comparing indicators.
It does not make a local changeover definition automatically comparable with every other site.
Map the numerator, denominator, clock boundary, and loss classification before comparison.

ISO 9001 provides requirements for a quality management system when an organization adopts or is contractually required to meet it.
It is not itself a universal legal mandate and does not certify that a specific changeover is safe.
Apply the site's documented change control, risk assessment, and process validation requirements.
The authoritative standard and certification scope govern any certification claim.

### Safety and product-specific obligations

In the United States, OSHA's control of hazardous energy rule, 29 CFR 1910.147, applies to covered servicing and maintenance activities, subject to its scope and exceptions.
Determine applicability to the actual task and site with safety personnel.
The rule includes requirements for energy control procedures and training; moving work outside the stop does not waive them.
Other jurisdictions and additional machine guarding, electrical, chemical, or worker-safety rules can apply.
This course is not legal or engineering advice.

For U.S. food facilities subject to FDA's preventive controls rule, 21 CFR Part 117 includes current good manufacturing practice and, where applicable, hazard analysis and risk-based preventive controls.
The facility's hazard analysis and written programs determine relevant controls; allergen and sanitation controls cannot be bypassed for a shorter setup.
For drugs, 21 CFR 211.67 addresses equipment cleaning and maintenance, while other current good manufacturing practice provisions may also apply.
Applicability depends on product, operation, and jurisdiction.
This is a dated educational summary, not a substitute for current law or a validated site procedure.

Cleaning validation is not interchangeable with routine verification.
A process change may require review of residues, sampling methods, worst-case products, acceptance limits, and documented approval.
Whether revalidation is necessary is a quality and regulatory decision based on the actual change and applicable requirements.
A scheduler should consume an approved release status, not infer release from a clock.

### Governance and retention

Assign a process owner to matrix values and a data steward to identifiers and effective dates.
Limit who can change prohibited edges or release states.
Maintain an audit trail for overrides and the rationale for exceptional sequences.
Define retention using legal, quality, contractual, and company requirements applicable to the site.
Protect employee-linked observations according to local privacy law and labor agreements.
The scheduling model should expose the version and timestamp of every decision-critical input.

## 9. Industry practice and case evidence

### Toyota and the method's origin

Shigeo Shingo's book *A Revolution in Manufacturing: The SMED System* describes the development of setup reduction methods associated with Japanese manufacturing and press changeovers.
It is a practitioner account, not a controlled study proving the same numerical outcome at every factory.
The generalizable method is separation of internal and external setup, conversion where possible, and improvement of both.
Do not copy an anecdotal time reduction into a business case without a local baseline.

Toyota's public Toyota Production System material describes just-in-time and jidoka as pillars of its production system.
It emphasizes making only what is needed and stopping for abnormal conditions.
This supports the operational importance of flexible flow and quality, but does not establish a specific setup time or financial benefit for the hypothetical line in this topic.
**Inference:** reducing reliable setup time can make smaller batches more practical when setup is a material part of the lot-sizing trade-off.
The actual result depends on demand, available capacity, and the rest of the system.

### Published SMED case

Cakmakci's 2009 paper, “Process improvement: performance analysis of the setup time reduction—SMED in the automobile industry,” reports an industrial setup-reduction application in the automotive industry.
Consult the paper for its equipment, measured results, and boundaries before transferring a benchmark.
A single case demonstrates application in its setting, not a guaranteed percentage improvement elsewhere.
The useful practice is to document the baseline, proposed work elements, implementation, and measured outcome.

### Food and regulated production

A food plant may sequence products to reduce allergen cross-contact risk and cleaning burden.
An apparent efficiency gain is unacceptable if it conflicts with the site's hazard analysis or validated sanitation process.
The order in which products run is a control decision as well as a capacity decision.
**Inference:** an optimal sequence under setup minutes alone can be infeasible under a validated allergen rule.
The correct operational response is to prohibit the edge or add the approved clean, not to treat the edge as a negotiable cost.

### Typical implementation choices

Plants often start with a loss tree or a Pareto analysis of transitions by total minutes, frequency, and variability.
A rare long transition and a frequent moderate transition require different interventions.
An event with a large total can be driven by high volume rather than poor individual performance.
Check both the per-event distribution and aggregate loss.

A multidisciplinary workshop can expose steps that a planner cannot see in an enterprise system.
Operators know actual access, adjustments, and recurring searches.
Quality knows the release and cleaning controls.
Maintenance knows isolation, wear, and mechanical constraints.
Finance can separate accounting allocation from incremental value.
Scheduling can test whether the faster transition changes a feasible sequence.

Common rollout practice is to pilot a small set of transition families before extending a matrix to all products.
The pilot should include the difficult directions, not only the easiest changeover.
After the pilot, compare the distribution of actual performance under the same measurement boundary.
A shift in product mix or staffing can otherwise be mistaken for a method improvement.
These are practical recommendations, not claims about a named company's results.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Symmetric matrix | Same value copied both ways | Directional cleaning is hidden | Infeasible or expensive sequence | Measure and approve directed edges |
| Unknown set to zero | Missing edge has no owner | Data default becomes free setup | Unrealistic plan | Block unknown edges pending review |
| Wrong initial state | First job assumes clean equipment | Prior run or clean expiry omitted | First order starts late | Verify state at horizon start |
| Quality endpoint omitted | First output fails release | Gross stop ends before accepted output | Capacity overstated | Record physical acceptance and release separately |
| Double counting benefit | Setup rate and contribution both include lost output | Same minute valued twice | Inflated business case | Trace each benefit to one mechanism |
| External work ignored | Staging worker is double-booked | Prep consumes another scarce resource | Work is not ready | Schedule crew and tools explicitly |
| Parallel work unsafe | Crowding around guarded equipment | Jobs overlap without isolation controls | Injury or incident | Risk assess and authorize work |
| Validation bypassed | Changed cleaning has no signed approval | Speed prioritized over product control | Contamination or compliance exposure | Quality change control and trial |
| Average hides variability | Frequent missed planned starts | Tail delays not represented | Tardiness and expediting | Measure distribution and service risk |
| Boundary drift | End timestamp changed after project | Before and after use different definitions | False improvement | Lock and audit event definitions |
| Lot size grows unchecked | Finished stock and obsolescence rise | Fewer setups prioritized alone | Higher working capital and waste | Compare total relevant cost |
| Local improvement at idle machine | More idle time after change | No constrained demand uses saved minutes | Claimed throughput never appears | Verify bottleneck and demand |
| Missing final clean | Final state not valued | Cost moved outside horizon | Biased sequence choice | Model terminal condition |
| Matrix goes stale | New tool or product follows old rule | No effective-date governance | Invalid setup or quality risk | Assign owner and version reviews |
| Setup overlap misread | Labor minutes exceed elapsed time | Concurrent activities summed | Wrong machine capacity estimate | Record intervals and critical path |
| Unit mismatch | Hours entered as minutes | Unvalidated interface conversion | Grossly wrong schedule | Validate units and range checks |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Times are deterministic | Makes the small calculation transparent | Shift and product variability | Compare distributions by edge and shift |
| All jobs available at start | Supports the simple recurrence | Late components or approvals | Compare actual release timestamps |
| Continuous line calendar | Makes clock time equal working time | Breaks or maintenance | Overlay a real shift calendar |
| One crew always available | Avoids multi-resource conflicts | Crew shared with other lines | Check crew schedule and travel |
| No setup expiry | Allows a simple edge model | Clean status expires before use | Review validated hold times |
| No forbidden transitions | Makes every permutation feasible | Allergen or safety prohibition | Review approved transition rules |
| Processing times fixed | Isolates setup effects | Speed or yield varies by sequence | Analyze output and scrap by predecessor |
| Setup and tardiness costs independent | Allows charges to be added | Rate already includes customer loss | Trace rate construction with finance |
| No terminal cleanup | Simplifies three-job horizon | Next horizon needs a costly clean | Include terminal state or next job |
| All units accepted | Keeps output quantity constant | Startup scrap or failed release | Compare first-pass yield by transition |
| Value can recur | Supports annualization | Demand or schedule mix changes | Examine expected annual transition count |
| Improvements remain effective | Supports stable payback | Wear, turnover, or drifting method | Audit and review post-launch data |

A good model can make the wrong decision if the physical and commercial assumptions are wrong.
Run sensitivity cases for timing variability, demand, cost rates, and the location of the constraint.
Use stochastic simulation or robust scheduling when service exposure justifies the extra complexity.
Keep validation and mandatory safety rules as hard constraints, not as probabilistic cost preferences.

## 12. Exercises

### Exercise 1 — Classify observed activities

**Time:** 15 min.
**Deliverable:** an activity table with classification, reason, and a safety or quality check.
A team retrieves a pre-approved fixture from a safe remote rack, isolates the machine, replaces tooling inside the guarded zone, checks alignment, restarts, and verifies accepted output.
Classify each as potentially external or internal and explain one condition that could change the classification.
**Acceptance:** all activities are addressed; isolation and guarded work are not moved into live operation; inspection is not removed merely for speed.

### Exercise 2 — Calculate a schedule

**Time:** 20 min.
**Deliverable:** a timeline and cost calculation for A, B, C using the worked-example matrix.
Show initial state, every transition, completion times, job tardiness, and total relevant charge.
**Acceptance:** A, B, and C finish at minutes 120, 245, and 370; setup is 55 min; tardiness is 45 job-min; cost is USD 200.
Explain why job-minutes can exceed elapsed schedule time.

### Exercise 3 — Design a SMED pilot

**Time:** 25 min.
**Deliverable:** a one-page pilot plan for the B to C transition.
Specify a baseline, at least two improvement hypotheses, responsible functions, trial population, quality and safety gates, and a rollback rule.
**Acceptance:** plan distinguishes external preparation from internal work; keeps the authorized cleaning and isolation controls; requires approval before the planning matrix changes.

### Exercise 4 — Evaluate benefit

**Time:** 20 min.
**Deliverable:** annual net benefit and simple payback under the worked-example recurrence assumption.
Use 180 runs per year, USD 32 saved per run, USD 1,200 per year additional preparation expense, and USD 9,120 initial investment.
**Acceptance:** USD 4,560 per year net benefit and 2 years simple payback; identify one reason either value may not be realized.
Explain the double-counting risk if the setup rate already contains late-delivery loss.

### Exercise 5 — Specify event data

**Time:** 20 min.
**Deliverable:** a data dictionary of at least ten fields plus three validation rules.
Include state, timestamps, quality release, method version, resource, and auditability.
**Acceptance:** units, source, ownership, missing-value treatment, and before/after comparability are explicit.

### Exercise 6 — Critique a matrix

**Time:** 20 min.
**Deliverable:** a corrected policy for a matrix in which A to B is 35 min, B to A is missing, A to A is assumed zero, and the initial state is unknown.
**Acceptance:** no unmeasured edge is invented; the same-state rule and current condition are verified; prohibited, unknown, and zero are distinguished.

### Exercise 7 — Explain operational uncertainty

**Time:** 15 min.
**Deliverable:** a short risk assessment of using the median transition time in a service-critical schedule.
**Acceptance:** distinguishes median from a commitment, proposes a measure of tail risk, and names a practical way to validate the selected planning allowance.

### Exercise 8 — Design an integrated decision

**Time:** 25 min.
**Deliverable:** a comparison plan for reducing lot size after a successful SMED pilot.
Include setup frequency, holding cost, capacity, product availability, cleaning obligations, and the final state beyond the horizon.
**Acceptance:** identifies incremental costs and benefits, names a baseline, treats mandatory requirements as constraints, and avoids counting the same capacity benefit twice.

## 13. Self-check answers

### Exercise 1

Retrieving a prepared fixture can be external if its safe location and preparation do not require entering a guarded zone or interfering with the running line.
Isolation and replacement inside the guarded zone are internal when required to stop the machine safely.
Alignment and verification depend on their actual procedure; some checks require a stopped machine, while final acceptance requires production or test output.
Record restart and verification separately when release follows physical production.
A trained safety and quality authority determines the actual classification.

### Exercise 2

Start in state A at minute 0.
A to A is 0 min; A production ends at minute 120.
A to B takes 35 min; B production ends at minute 245.
B to C takes 20 min; C production ends at minute 370.
The late minutes are A 0, B 5, and C 40, summing to 45 job-min.
The setup cost is USD 110 and tardiness cost is USD 90.
Total relevant cost is USD 200.
Job-minutes accumulate across jobs and are not the same measure as the final clock time.

### Exercise 3

A strong pilot identifies comparable baseline transitions and observes enough shifts to understand variation.
It prepares approved tools before stop and considers a validated locating fixture.
It keeps the prescribed isolation, cleaning, inspection, and release steps.
Safety, quality, operations, and maintenance approve the method and acceptance criteria.
A failed quality result, unsafe observation, or unapproved deviation triggers rollback and review.
A weak plan simply tells operators to work faster and updates the matrix after one favorable run.

### Exercise 4

Gross annual scenario benefit is 180 × USD 32 = USD 5,760.
Subtract USD 1,200 per year of added preparation expense to obtain USD 4,560 per year.
USD 9,120 / USD 4,560 per year gives 2 years of simple payback.
If the transition occurs only half as often, the gross benefit falls while fixed annual expense may remain.
If the setup charge already includes the same late-delivery impact, counting the separate tardiness charge overstates benefit.

### Exercise 5

A strong dictionary contains event ID, site, machine, prior and next state, start and end timestamps, time zone, production order, method version, quality disposition, release time, reason codes, and source system.
For example, validate end not earlier than start, verify state codes against the effective mapping, and require an approved method version on the event date.
Keep raw and corrected records linked.
Missing prior state is unknown, not a clean state.

### Exercise 6

The 35 min A to B entry says nothing definitive about B to A.
Mark B to A unknown until measured or approved under a conservative engineering rule.
Confirm whether A to A needs a batch clean, inspection, or tool adjustment before setting it to zero.
Verify the actual initial state from execution and quality records.
Treat prohibited, unknown, and approved zero as three separate values.

### Exercise 7

Half of a continuous distribution can exceed its median; the median is not a service guarantee.
Review the upper tail and the sample size for each important transition.
Select an allowance consistent with the service objective and resource constraints, then backtest it on later events.
Escalate if there is too little representative data or if the process has changed.

### Exercise 8

A strong comparison evaluates the same demand and service requirements under the best feasible old and new policies.
It calculates the change in transition frequency and capacity load, incremental average inventory, carrying cost, quality loss, and preparation cost.
It tests whether smaller lots improve service or merely shift work to another constraint.
It includes the transition beyond the horizon and maintains approved cleaning and safety rules.
Count either actual incremental contribution or the corresponding opportunity cost once, not both.

## 14. Interview questions

### 1. Why can A to B differ from B to A?

A strong answer identifies directional cleaning, color, tooling, or allergen requirements and proposes measured directed edges.
A weak answer assumes setup is a fixed attribute of the incoming product.

### 2. What are the endpoints of a changeover?

A strong answer defines last accepted outgoing output and first accepted incoming output, while separating physical acceptance from formal release when needed.
A weak answer uses a convenient stop code without checking quality or waiting time.

### 3. What does external setup mean?

A strong answer says work can safely occur while the line is running and still consumes real resources.
A weak answer says it can always be moved before a stop without an isolation or quality review.

### 4. Is a ten-minute SMED target mandatory?

A strong answer explains the single-digit aspiration and prioritizes safety, validation, and economically justified improvement.
A weak answer promises a number regardless of process risk.

### 5. How do you handle an unknown transition?

A strong answer distinguishes unknown from prohibited and approved zero, then seeks a validated conservative route or measurement.
A weak answer silently fills the missing value with zero.

### 6. When is a saved minute worth contribution margin?

A strong answer requires a binding constraint, feasible extra production, demand for accepted output, and incremental contribution.
A weak answer multiplies every saved minute by sales revenue.

### 7. How can setup reduction increase cost?

A strong answer discusses extra staffing, validation, fixture cost, quality loss, and downstream bottlenecks.
A weak answer assumes any lower machine time is automatically cash savings.

### 8. Why might the shortest setup sequence be a poor schedule?

A strong answer considers job due dates, release times, resource constraints, mandatory cleans, and customer penalties.
A weak answer uses setup minutes as the only objective.

### 9. How do you compare before and after fairly?

A strong answer controls event boundaries, product mix, shifts, method version, staffing, and data quality.
A weak answer compares the single fastest new event with the worst old event.

### 10. What happens after a successful pilot?

A strong answer includes approval, training, effective-dated work instructions and matrix updates, monitoring, and change control.
A weak answer immediately applies the pilot time to every product and shift.

## 15. Key terms

- [Changeover](#4-definitions-and-vocabulary): transition from one state to another.
- [Directed matrix](#step-6--build-the-transition-matrix): setup data indexed by prior and next state.
- [Initial state](#step-7--construct-feasible-schedules): verified condition before the first scheduled job.
- [Internal activity](#step-4--separate-internal-from-external-work): work that requires a stopped machine.
- [External activity](#step-4--separate-internal-from-external-work): work safely possible while the machine runs.
- [SMED](#3-the-core-mental-model): structured setup-reduction method.
- [Last good unit](#step-2--define-the-physical-boundary): outgoing accepted output defining one possible boundary.
- [First good unit](#step-2--define-the-physical-boundary): incoming accepted output defining one possible boundary.
- [Setup crew](#step-7--construct-feasible-schedules): personnel required during transition.
- [Release](#8-data-standards-and-regulatory-context): authorization to use output.
- [Tardiness](#step-7--construct-feasible-schedules): positive completion delay relative to due time.
- [Lateness](#4-definitions-and-vocabulary): signed completion time minus due time.
- [Makespan](#4-definitions-and-vocabulary): time to complete the schedule.
- [Bottleneck](#step-8--evaluate-economic-alternatives): resource limiting system throughput.
- [Holding cost](#step-8--evaluate-economic-alternatives): incremental cost of carrying stock.
- [Contribution](#step-8--evaluate-economic-alternatives): revenue less relevant variable cost.
- [Opportunity cost](#step-8--evaluate-economic-alternatives): value of the best feasible alternative forgone.
- [Cleaning validation](#8-data-standards-and-regulatory-context): evidence supporting the cleaning process.
- [Change control](#step-9--pilot-control-and-learn): approval and documentation of process change.
- [Anticipatory setup](#step-7--construct-feasible-schedules): setup permitted before a job is ready.
- [Terminal state](#11-assumptions-and-limits): end-of-horizon condition affecting later work.
- [Planning allowance](#exercise-7): time chosen to address schedule uncertainty.
- [Simple payback](#smed-intervention): initial investment divided by annual net benefit.
- [Sequence-dependent cost](#step-8--evaluate-economic-alternatives): cost dependent on the ordered transition.

## 16. Sources

1. Shingo, Shigeo. *A Revolution in Manufacturing: The SMED System*. Productivity Press, 1985. Practitioner account of the method and its development.
2. Cakmakci, Mehmet. “Process improvement: performance analysis of the setup time reduction—SMED in the automobile industry.” *International Journal of Advanced Manufacturing Technology* 41, 168–179 (2009). [https://doi.org/10.1007/s00170-008-1434-4](https://doi.org/10.1007/s00170-008-1434-4). Industrial case evidence; applicability is context-specific.
3. Toyota Motor Corporation. [Toyota Production System](https://global.toyota/en/company/vision-and-philosophy/production-system/). Company description of just-in-time and jidoka; not independent proof of a setup benefit.
4. International Society of Automation. [ISA-95 standard](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard). Overview of enterprise-control integration; consult the applicable normative edition.
5. International Organization for Standardization. [ISO 22400-2:2014](https://www.iso.org/standard/54497.html), *Automation systems and integration — Key performance indicators (KPIs) for manufacturing operations management — Part 2: Definitions and descriptions*. Check current edition and scope.
6. International Organization for Standardization. [ISO 9001 quality management](https://www.iso.org/iso-9001-quality-management.html). Overview of the quality management standard; applicability is not universal law.
7. U.S. Occupational Safety and Health Administration. [29 CFR 1910.147, Control of hazardous energy](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147). Consult current text and applicability.
8. U.S. Electronic Code of Federal Regulations. [21 CFR Part 117, Current Good Manufacturing Practice, Hazard Analysis, and Risk-Based Preventive Controls for Human Food](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-B/part-117). Consult current text and applicability.
9. U.S. Electronic Code of Federal Regulations. [21 CFR 211.67, Equipment cleaning and maintenance](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-211/subpart-D/section-211.67). Consult current text and applicability.

Regulatory summaries in this topic were prepared for the 2026 course edition.
Check current law and site-specific requirements before implementation.
The worked example, cost rates, and decision experiments are invented teaching data, not source claims.

## 17. Links

- [Module 15 topic index](_index.md).
- [Manufacturing and production planning survey](../../../course/15_manufacturing_production_planning.md).
- [Manufacturing master data](02_manufacturing-master-data.md).
- [Capacity requirements planning](05_capacity-requirements.md).
- [Finite scheduling](06_finite-scheduling.md).
- [Lot sizing](07_lot-sizing.md).
- [OEE and the loss tree](09_oee-loss-tree.md).
- [Bottlenecks, lean, and theory of constraints](10_bottlenecks-lean-toc.md).
- [Statistical process control](11_spc-cpk.md).
- The [module survey](../../../course/15_manufacturing_production_planning.md) includes its lab and assessment context.
