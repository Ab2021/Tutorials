# Discrete-Event Simulation

> Module 20 — Digital Twins, Simulation, and Optimization under Uncertainty  
> Topic 02 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 06, 11 and 15; [Module 20 topic map](_index.md)  
> Version: 2026.5

## 1. Why this topic exists

A distribution-center manager must decide whether to open a second unloading door.
The proposed door has a fixed daily labor and equipment cost.
It may reduce truck waits, but its value depends on when trucks arrive and how long they occupy the first door.
A spreadsheet that divides total work by total capacity cannot show the queue that forms when several trucks arrive together.
Nor does average utilization reveal whether a priority shipment misses its appointment while the dock is busy with lower-priority work.

Discrete-event simulation (DES) represents the system as a sequence of changes at particular times.
A truck arrives, joins a queue, acquires a door, completes unloading and leaves.
Between those events, the modeled state remains unchanged.
The analyst can change capacity or the dispatch rule, replay plausible conditions and compare decision consequences.

DES is useful when waiting, shared capacity, routing, schedules or interactions determine outcomes.
It is not automatically a digital twin.
A standalone DES may be built for one study and run offline.
A digital twin generally adds a maintained relationship to a physical asset or process; the precise definition and synchronization requirements must be agreed for the application.
A visually convincing animation does not establish that this relationship exists or that the model predicts well.

The decision owner is usually an operations manager, network planner or investment committee.
The manager needs a decision that is feasible under labor and safety constraints, not a perfect imitation of every motion.
The analyst needs timestamped operational records, process knowledge, a defined baseline and a review process.
The financial partner needs a consistent comparison of extra capacity cost against avoidable delay and service consequences.

**Business question:** Should the site pay for an additional unloading door on a defined shift, given uncertain arrivals and service times?

**Decision:** Choose a capacity and queue policy for that shift.

**Forecast:** A probability model for future arrivals and work content, estimated from evidence.

**Plan:** A proposed roster and door allocation built using the forecast and operational constraints.

**Target:** A desired service level, such as no more than 10% of trucks waiting over 30 minutes.

**Commitment:** A separately authorized promise to a carrier or customer; a simulation result is not itself a commitment.

A one-off deterministic example can teach event logic.
A decision about recurring staffing needs repeated scenarios, input uncertainty and operational validation.
The model's value is the quality of the decision it supports, not the number of entities drawn on a screen.

## 2. Learning outcomes

After this lesson you can:

1. **Identify** entities, attributes, state variables, events, resources and queues in a real process; demonstrate this in Exercise 1.
2. **Trace** a future-event list and calculate waits, utilization and throughput with explicit tie-breaking; demonstrate this in Exercises 2 and 3.
3. **Specify** the system boundary, time horizon, initialization and closure rules; demonstrate this in Exercise 4.
4. **Implement** a reproducible single-queue dock DES using the Python standard library; demonstrate this in Exercise 5.
5. **Compare** capacity policies with paired replications and explain uncertainty without treating simulated trucks as independent experiments; demonstrate this in Exercise 6.
6. **Critique** input data, validation evidence and operational constraints; demonstrate this in Exercise 7.
7. **Recommend** an actionable business decision with a cost bridge, sensitivity and limitations; demonstrate this in Exercise 8.

An outcome is achieved when you can explain the operational meaning as well as execute the arithmetic.
A model that runs but does not answer a decision question is incomplete.
A model that answers a question under an unexamined assumption is not necessarily decision-ready.

## 3. The core mental model

```text
Operational system                      Model and experiment
------------------                      --------------------
Truck appointment / actual arrival ---> arrival event at time t
Truck identity and work content ------> entity attributes
Door availability --------------------> resource state / capacity
Waiting yard and dispatch rule -------> queue and selection rule
Unloading completion -----------------> departure and resource release
                                              |
                                       future-event list
                                              |
                                  advance clock to next event
                                              |
                                update state, record, schedule
                                              |
                          replication outcomes and paired comparisons
                                              |
                           manager chooses an executable policy
```

Imagine a clock that jumps rather than ticks every second.
At each jump, remove the earliest event from the future-event list.
Advance the clock to its timestamp.
Apply the event's state transitions in a specified order.
Record measurements and schedule any resulting events.
Then repeat until the stopping condition is met.

The future-event list is a set of scheduled future changes, not a queue of trucks waiting for service.
A truck queue contains entities waiting for a constrained resource.
Confusing the two can lead to impossible service order or an event being lost.

The current state is a minimal set of facts needed to determine what happens next under the model rules.
For a simple dock it includes the clock, the ordered waiting trucks, door occupancy and scheduled completion times.
A richer dock may need shift status, driver hours, hazardous-material compatibility and availability of forklift operators.
Add a state variable when omitting it can change the decision or invalidate a constraint.

An entity is a modeled item that moves through the system.
A resource is capacity seized or released by entities.
A queue is a waiting collection with an explicit selection discipline and, where relevant, a finite physical limit.
An event is an instantaneous change in modeled state.
Service is an interval represented by a start event and a later completion event.

Randomness enters through modeled inputs, not through an unexplained change in the clock.
Examples include arrival time, service duration and equipment failure.
For a particular scenario, fixed random draws and policy rules generate a reproducible path.
Different scenarios generate different paths and decision outcomes.

A simulation experiment is a controlled comparison of policies under specified conditions.
It is not a search for a single lucky run.
The output is a distribution or a paired difference across independent replications, plus evidence that the representation is adequate for its purpose.

## 4. Definitions and vocabulary

| Term | Meaning in this lesson | Common misuse | Unit or representation |
|---|---|---|---|
| DES | Model whose state changes at discrete event times | Treating an animated flow as validated DES | Event times and state transitions |
| Entity | Individual modeled object, such as a truck | Treating a whole daily volume as one truck when individual waits matter | Entity ID |
| Attribute | Entity-specific property, such as unload minutes | Changing a fixed property without a rule | Minutes, class or ID |
| State | Information needed to evolve the model | Reporting only an end-of-day average | Queue, busy doors, clock |
| Event | Instantaneous modeled state change | Confusing event timestamp with interval duration | Simulation minutes |
| Arrival | Entry of an entity at the modeled boundary | Using gate arrival as dock arrival without travel time | Minutes from shift start |
| Service start | Entity begins using the required capacity | Starting service before a door is free | Minutes from shift start |
| Completion | Service finishes and capacity is released | Forgetting to release the door | Minutes from shift start |
| Simulation clock | Current model time | Confusing model time with computer runtime | Minutes from shift start |
| Future-event list | Ordered collection of scheduled events | Using it as the waiting-truck queue | Timestamp, type, tie key |
| Resource | Constrained reusable capacity | Assuming doors imply available labor | Doors, crews or forklifts |
| Capacity | Number of units simultaneously available | Treating daily hours as instantaneous capacity | Doors at time t |
| Queue | Entities waiting for a resource | Equating queue length to number in system | Trucks |
| Queue discipline | Rule for selecting the next entity | Assuming all priority rules are first-in-first-out | FIFO, priority, appointment order |
| FIFO | First in, first out among eligible entities | Ignoring ties or exceptions | Arrival order |
| Work-conserving | A compatible idle server takes eligible waiting work | Assuming this when breaks or staging prevent service | Boolean policy property |
| Waiting time | Start time minus arrival time | Counting unloading as waiting | Minutes per truck |
| Flow time | Departure time minus arrival time | Calling flow time just queue delay | Minutes per truck |
| Throughput | Completed entities in a defined period | Using arrivals as completed work | Trucks per shift |
| Work in process (WIP) | Entities in the modeled system | Counting only waiting trucks | Trucks |
| Utilization | Busy resource-time divided by available resource-time | Averaging door percentages without correct weights | Door-minutes / available door-minutes |
| Warm-up | Excluded initial period intended to reduce initialization bias in a steady-state study | Applying it automatically to a real empty-start shift | Minutes |
| Terminating run | Simulation with an operational stopping condition | Extrapolating one shift to a steady state | Shift, day or order book |
| Replication | Independent simulated realization of the experiment | Calling each truck an independent replication | One scenario and outcome |
| Random seed | Initial state for a pseudorandom stream | Treating one seed as evidence of stability | Integer |
| Common random numbers (CRN) | Coupling policies with matching exogenous random inputs | Reusing a seed without preserving the correspondence of draws | Paired scenario IDs |
| Confidence interval (CI) | Sampling-uncertainty interval under a stated method | Claiming it covers input misspecification | Outcome units, confidence level |
| Verification | Checking that implementation follows the specification | Equating correct code with a valid model | Tests and traces |
| Validation | Assessing fitness for a specific intended use | Treating one matching average as proof | Evidence and acceptance criteria |
| Censoring | Incomplete observation of an entity by a measurement cutoff | Dropping trucks still waiting at closing | Remaining count and age |
| Scenario | A coherent set of assumptions and inputs | Treating an arbitrary percentile combination as a probability distribution | Named input set |
| Policy | Decision rule that can be implemented | Treating an uncertain future arrival as a control | Capacity and dispatch rule |
| Bottleneck | Capacity or constraint limiting relevant system performance | Calling the highest-utilized step a permanent bottleneck | Context and time period |

## 5. The mechanics, step by step

### Step 1 — Define the decision and counterfactual

**Input:** The operational decision, decision date, alternatives and owner.

Ask what will change if the study recommends one alternative rather than another.
For the dock, the controllable alternatives are one staffed door or two staffed doors for the shift.
Specify whether a second door can be opened only at the start or dynamically during the shift.
Dynamic opening introduces a trigger, lead time, labor availability and potentially a minimum paid period.

Identify the baseline as the actual executable policy, not an idealized case with free labor and perfect arrivals.
State the decision horizon, review date and cost basis.
A one-day model can support a shift policy if its scenarios represent the conditions in which that policy will be used.
A capital purchase also needs a multi-year financial model and evidence about changes in future demand.

Choose service metrics before seeing the experiment results.
For example, define an excessive wait as strictly greater than 30 minutes from dock arrival to service start.
Specify whether the service-level denominator includes trucks still waiting at close.
A policy may improve average wait while leaving an unacceptable tail for priority freight.

**Output:** A decision statement, a list of executable alternatives and a metric dictionary.

**Branch:** If the decision only concerns annual aggregate flow with no congestion interaction, compare simpler arithmetic before building DES.
If individual queues and timing determine the answer, continue.

### Step 2 — Draw the boundary and map entities

**Input:** A process walk, records and interviews with the shift supervisor.

Choose the entry point: site gate, yard check-in or dock-ready time.
Choose the exit point: unloading completion, document release or gate-out.
The worked example starts at dock-ready time and ends at unloading completion.
It therefore excludes travel through the yard, paperwork and outbound gate queues.

Define one entity as one arriving truck.
Give it an identifier, arrival time, required door-minutes and any priority or compatibility class that affects dispatch.
For a realistic study, define if multiple purchase orders on one truck need separate treatment.
A truck may occupy one door while several order entities are processed inside it; do not count those as independent door arrivals.

List process states: not yet arrived, waiting, in service and completed.
Define permitted transitions and reject impossible transitions such as completed to waiting without a rework event.
If unloaded pallets consume staging capacity, add a staging constraint only if it can block the dock in the decision horizon.
Record why excluded processes are not material to this decision.

**Output:** Boundary diagram, entity schema and a list of excluded interactions.

**Branch:** If the excluded yard queue can block dock-ready arrivals, either expand the boundary or explicitly condition the study on dock-ready arrivals and limit the conclusion.

### Step 3 — Define resources and calendars

**Input:** Staff roster, equipment availability, access rules and service process.

Represent the number of staffed, usable doors over time.
A physical door without a qualified crew is not necessarily available capacity.
If one forklift is shared by two doors, model the forklift as another resource when contention can limit the comparison.
Do not count the same person simultaneously in two crews unless the process supports it.

For each resource, record the number of units, calendar, skill or compatibility, and interruption rules.
Specify what happens at shift end: finish current work, hand over, pause or stop.
Specify whether a scheduled break can interrupt a service already in progress.
A calendar changes available capacity; it is not merely a cost multiplier.

For the teaching example, each door has one dedicated crew and is continuously available from time zero until the last modeled truck completes.
There are no breaks, equipment failures or incompatible trucks.
Those are assumptions, not claims about all docks.

**Output:** Resource table with capacity by time and allocation constraints.

**Branch:** If capacity varies by time, schedule calendar-change events and define the ordering relative to arrivals and completions at the same timestamp.

### Step 4 — Specify queue admission and dispatch

**Input:** Operating procedures, contract priorities, space constraints and exception history.

Specify the eligibility rule and the selection rule separately.
Eligibility says which trucks can use a currently free door.
Selection says which eligible truck goes next.
FIFO is a selection rule; a door's temperature or handling compatibility is an eligibility constraint.

Decide what happens when the yard is full.
A truck may wait outside the modeled boundary, be rerouted, be rejected or block an upstream process.
Do not silently discard it and report a shorter queue.
Specify any priority aging or reservation rule to prevent a low-priority class from waiting indefinitely.

In the example, the waiting area is unlimited and all trucks are eligible for either door.
The queue is FIFO by arrival time, with truck ID as a deterministic secondary key when arrivals tie.
An idle door immediately takes the first waiting truck.
There is no preemption: a new high-priority arrival cannot interrupt unloading.

**Output:** An unambiguous dispatch algorithm and exceptional-case rules.

**Branch:** If priority changes service order, record class-specific waits and service levels as well as overall averages.

### Step 5 — Define the event list and ordering

**Input:** Process transitions and resource rules.

A minimal dock model needs an arrival event and a completion event.
An arrival adds a truck to the queue and may immediately start service.
A completion frees a door, records departure and may start the next eligible truck.
A service start removes a waiting truck, assigns a door and schedules a completion at start plus service duration.

Represent each scheduled event by timestamp, event-type priority, stable tie key and event payload.
The stable key prevents the computer from comparing arbitrary entity objects when timestamps match.
Never schedule an event earlier than the current simulation clock.
Reject negative service durations or malformed timestamps rather than moving the clock backward.

For the worked trace, completion events are processed before arrival events at the same time.
After each event, assign waiting trucks to any free doors.
The queue remains FIFO, so an existing waiting truck takes a freed door before a new arrival joins.
A truck arriving at the exact instant a door becomes free can start at that same time if no earlier truck is waiting.

At time zero, create arrival events for all trucks in the example.
A stochastic model could schedule the next arrival as each arrival occurs, provided the input process is specified consistently.
The event list is ordered by time even if an event was created later than another event.

**Output:** Event table, tie rules and scheduling invariants.

**Branch:** If events can trigger zero-duration transitions, define a deterministic ordering and guard against infinite loops at a single timestamp.

### Step 6 — Specify inputs and their dependence

**Input:** Timestamped arrival and service records with provenance and data-quality checks.

Derive interarrival times from arrivals in the same site and time zone.
Do not estimate them across a shift boundary as if the overnight closure were normal within-shift arrival behavior.
Identify appointments, late arrivals, cancellations and actual dock-ready times separately.
Include planned peaks rather than fitting a single stationary rate to all hours by default.

Measure service duration from actual service start to completion.
Distinguish elapsed time from active work when breaks or blocked equipment are present.
Investigate zero, negative or implausibly long durations and document corrections.
Stratify by load type or work content when that materially changes service duration.

Preserve relevant dependence.
For example, heavy loads may be clustered at specific appointment windows, so independently sampling arrival and service values can understate congestion.
A practical approach is to resample whole days or blocks, preserving the within-block pattern, when enough comparable historical blocks exist.
For a forecast with new operating conditions, describe how the data-generating model changes and why.

**Output:** An input dataset or generative model, units, provenance, sample size and dependence assumptions.

**Branch:** If data are scarce, elicit plausible ranges and run sensitivity cases; do not pretend a fitted distribution is precisely known.

### Step 7 — Define initialization and stopping

**Input:** Business horizon, opening state and reporting requirements.

A terminating shift study starts from a specified opening state and has a defined cutoff.
The initial state can be empty, a measured opening queue or a sampled backlog.
If the site typically has trucks waiting at shift start, starting empty can systematically understate delay.

Distinguish stopping arrivals from stopping the simulation.
One approach admits trucks arriving before shift close and then drains all admitted work.
Another stops all measurements at close and reports unfinished work and its age.
The two approaches answer different business questions and produce different utilization denominators.

In the deterministic worked example, four trucks arrive during the observation period and all are served to completion.
We measure utilization over 0–120 minutes, not until the last departure.
Idle time after the final departure is included.
We assume staff remain available until all four trucks complete.

A long-run steady-state study instead asks about a continuing operation.
It may require a defensible warm-up and long enough observation window.
A warm-up is not a universal fix and should not erase real opening congestion in a terminating shift.

**Output:** Initial state, arrival cutoff, service cutoff, drain policy and measurement windows.

**Branch:** When a backlog carries over across days, model enough consecutive days or explicitly model the opening-backlog distribution.

### Step 8 — Implement the event loop

**Input:** Event specifications, state definitions, input records and metrics.

Initialize the clock, resource state, queue and event list.
Schedule the initial arrivals and any initial calendar changes.
While the event list is nonempty and the stopping condition permits processing, take its earliest event.
Advance the clock, update time-integrated state measures, apply the event and record affected entity timestamps.
Schedule any consequent completion or other future events.

A priority queue or heap is a natural implementation for a future-event list.
A FIFO queue can be implemented with a deque.
The structures serve different purposes even if both have the word queue in their descriptions.
In a large model, use an established simulation framework when it makes resource interactions easier to inspect and maintain.
A custom engine still needs explicit specifications and tests.

Track state area for time-weighted queue length or busy capacity.
Before each event at time `t`, accumulate `(t - previous_time) × previous_state`.
Then divide by the observation-window length for a time average.
An event-sampled average of queue lengths is not generally a time-weighted average.

**Output:** Reproducible implementation and a trace mode that lists each transition.

**Branch:** If multiple resources must be seized together, define the allocation rule to avoid deadlock or unrealistic partial reservations.

### Step 9 — Verify conservation and edge cases

**Input:** A deterministic trace with hand-calculated expected results.

Check that arrivals equal completed plus waiting plus in-service entities at each checkpoint.
Check that the queue never has negative length and door occupancy never exceeds capacity.
For every completed truck, verify `arrival ≤ start ≤ completion` and `completion = start + service duration`.
Verify that no door serves overlapping trucks.
Check that a work-conserving idle compatible door does not coexist with an eligible waiting truck.

Test no arrivals, one truck, simultaneous arrivals, simultaneous completion and arrival, and zero waiting.
Test a duration that ends exactly at the observation boundary.
Test finite queue behavior and shift changes if those features are in scope.
Check denominators when no entities or no available door-minutes are observed.

Verification asks whether the program implements the documented model.
It does not establish that the documented model represents the physical dock well enough.

**Output:** Expected-versus-actual traces and a record of passed and failed checks.

**Branch:** If a conservation check fails, fix the model before interpreting any service-level output.

### Step 10 — Validate for the decision

**Input:** Independent operational evidence and an agreed validation plan.

Ask domain experts to inspect the process map and several event traces.
Compare modeled and observed arrival patterns, service-duration distributions, throughput, queue lengths and waits over a held-out period where data permit.
Check by time of day and load class, not only an overall mean.
If a rare peak matters to the investment, examine that peak separately.

Define practical tolerances before inspecting the final comparison.
For example, a team might require baseline mean wait within 5 minutes and the fraction waiting over 30 minutes within 5 percentage points, subject to adequate sample sizes.
These values are illustrative acceptance criteria, not a universal standard.
A wide uncertainty interval may prevent a meaningful validation conclusion.

Investigate discrepancies rather than tuning several unrelated parameters until one aggregate matches.
A model can match throughput while incorrectly representing the queue because it misses shift closures or priority rules.
Document which decision and operating range the evidence supports.

**Output:** Validation report, residual discrepancies, acceptance decision and owner.

**Branch:** If there is no reliable waiting-time record, instrument a pilot period or use independent observations; describe the limitation until then.

### Step 11 — Design a comparison

**Input:** Validated baseline, candidate policies, scenario generator and performance measures.

Choose a replication unit aligned with the decision, such as one representative shift.
Generate independent scenarios across replication IDs.
Within each replication, feed both policies the same arrivals and service requirements when those quantities are genuinely unaffected by the policy.
This is a paired comparison using common random numbers.

Store exogenous draws by scenario and truck ID before running either policy.
Simply resetting the same seed is insufficient if different policies consume random draws in different orders.
If the policy changes service time through congestion or staffing, use a structural relationship that represents that effect instead of mechanically forcing equality.

Compute each policy's per-replication outcome and its paired difference.
For `n` independent paired differences `d_i`, estimate the mean difference with `sum(d_i)/n`.
The standard error is the sample standard deviation of the `d_i` divided by `sqrt(n)`.
A two-sided interval can use a Student-t critical value with `n-1` degrees of freedom when its assumptions and replication count are appropriate.
Inspect skew and rare-event behavior; a normal-looking mean interval is not a guarantee about extreme waits.

**Output:** Experiment matrix, seed and scenario manifest, replication-level metrics and uncertainty summary.

**Branch:** If paired differences do not reduce variance, inspect the coupling; pairing is not guaranteed to help in every model.

### Step 12 — Translate results into a decision

**Input:** Replication outcomes, cost rates, constraints and implementation details.

Calculate incremental benefit from fewer waiting minutes only when the delay cost is incremental and not already counted elsewhere.
Separate carrier demurrage paid by the site from internal labor cost and customer-service penalties.
Do not monetize the same delay twice as both a demurrage saving and an identical generic delay cost.

Compare the incremental benefit to the incremental cost of the second staffed door.
Include minimum shift payments, overtime, training, equipment and any implementation lead time that differ by policy.
If the second door is already staffed for another task, use the opportunity cost of redeployment rather than assuming it is free.

Report the distribution of outcomes, not just the average.
A policy with positive expected benefit can still breach a hard safety or service requirement in unacceptable scenarios.
Show a break-even delay cost and sensitivity to arrivals, service and staffing assumptions.
Specify who can authorize the policy and how performance will be monitored after adoption.

**Output:** Decision brief with baseline, recommendation, economics, service consequences, uncertainty and a rollback or review trigger.

**Branch:** If the estimated benefit is small relative to model and cost uncertainty, collect data or run a controlled pilot rather than claim a decisive answer.

## 6. Worked example

### Case and assumptions

A regional distributor considers staffing a second receiving door for a two-hour peak.
The four trucks below are fictional, but their timestamps and durations are chosen to resemble a small appointment burst.
All times are minutes after 08:00 local time.
The modeled arrival is dock-ready time, not gate arrival.
Unloading duration is the required door occupancy in minutes.

| Truck | Dock-ready arrival (min) | Unloading duration (min) |
|---|---:|---:|
| A | 0 | 35 |
| B | 12 | 28 |
| C | 19 | 42 |
| D | 47 | 24 |

Policy 1 uses one door with one dedicated crew.
Policy 2 uses two doors with a dedicated crew for each door.
Both use one unlimited FIFO queue, identical truck work requirements and no interruption.
All trucks can use either door.
At a simultaneous completion and arrival, completion is processed first, with the existing queue served before the arrival is admitted.
All four trucks are admitted and processed to completion.
Both doors, when selected, are considered available throughout 0–120 minutes.

The service target is at most 10% of trucks waiting more than 30 minutes.
The target is not a contractual commitment.
The data are one illustrative scenario, not a forecast or proof that the future service target will be met.

### One-door event trace

| Time (min) | Event and resulting action | Queue after action | Busy door |
|---:|---|---|---|
| 0 | A arrives and starts; schedule completion at 35 | empty | A |
| 12 | B arrives | B | A |
| 19 | C arrives | B, C | A |
| 35 | A completes; B starts; schedule completion at 63 | C | B |
| 47 | D arrives | C, D | B |
| 63 | B completes; C starts; schedule completion at 105 | D | C |
| 105 | C completes; D starts; schedule completion at 129 | empty | D |
| 129 | D completes | empty | none |

| Truck | Arrival (min) | Start (min) | Completion (min) | Wait (min) | Flow time (min) |
|---|---:|---:|---:|---:|---:|
| A | 0 | 0 | 35 | 0 | 35 |
| B | 12 | 35 | 63 | 23 | 51 |
| C | 19 | 63 | 105 | 44 | 86 |
| D | 47 | 105 | 129 | 58 | 82 |
| Total | — | — | — | 125 | 254 |

Mean wait is `125 / 4 = 31.25` minutes per truck.
Two of four trucks wait strictly more than 30 minutes, so the tail fraction is 50%.
The last truck completes at minute 129, nine minutes after the 120-minute window.
Three trucks complete by minute 120; all four eventually complete under the drain rule.

Door occupancy from minute 0 to minute 120 is 120 door-minutes.
The door is continuously busy in that window, including the first 15 minutes of D's service.
Utilization over this window is `120 / 120 = 100%`.
Total work is 129 door-minutes, but dividing 129 by 120 would mix service after the window with capacity inside it.

The queue-length area from 0 to 120 is the sum of the individual waits, because all four trucks start by minute 120.
That area is 125 truck-minutes.
The time-average queue length is `125 / 120 = 1.0417` trucks.
The average is a time-weighted quantity, not the average of queue sizes at event timestamps.

### Two-door event trace

| Time (min) | Event and resulting action | Queue after action | Busy doors |
|---:|---|---|---|
| 0 | A arrives and starts on door 1; completion at 35 | empty | A |
| 12 | B arrives and starts on door 2; completion at 40 | empty | A, B |
| 19 | C arrives; both doors busy | C | A, B |
| 35 | A completes; C starts on door 1; completion at 77 | empty | B, C |
| 40 | B completes | empty | C |
| 47 | D arrives and starts on door 2; completion at 71 | empty | C, D |
| 71 | D completes | empty | C |
| 77 | C completes | empty | none |

| Truck | Arrival (min) | Start (min) | Completion (min) | Wait (min) | Flow time (min) |
|---|---:|---:|---:|---:|---:|
| A | 0 | 0 | 35 | 0 | 35 |
| B | 12 | 12 | 40 | 0 | 28 |
| C | 19 | 35 | 77 | 16 | 58 |
| D | 47 | 47 | 71 | 0 | 24 |
| Total | — | — | — | 16 | 145 |

Mean wait is `16 / 4 = 4` minutes per truck.
No truck waits strictly more than 30 minutes, so the tail fraction is 0%.
All four trucks complete by minute 120.
The queue-length area is 16 truck-minutes, giving `16 / 120 = 0.1333` trucks on average.

The doors perform 129 door-minutes of work inside the 120-minute window.
Available capacity is `2 × 120 = 240` door-minutes.
Combined utilization is `129 / 240 = 53.75%`.
Low combined utilization is not automatically waste: the capacity prevents simultaneous-arrival congestion.
However, an idle second crew may have a productive alternative task; the economic comparison should account for that task.

### Economic bridge

Assume the site bears an incremental delay cost of USD 1.50 per waiting truck-minute.
Assume this rate covers the relevant delay consequence once, with no separate charge for the same delay.
Assume the second staffed door costs an incremental USD 120 for the two-hour block.
These rates are illustrative assumptions, not observed carrier contracts or labor quotations.

| Item | Calculation | Result |
|---|---|---:|
| Waiting minutes avoided | 125 − 16 | 109 truck-minutes |
| Avoided delay cost | 109 × USD 1.50/minute | USD 163.50 |
| Extra capacity cost | Given | USD 120.00 |
| Net benefit | 163.50 − 120.00 | USD 43.50 per modeled block |
| Break-even delay cost | 120 / 109 | approximately USD 1.10 per truck-minute |

On this one path, the second door improves the service target and has positive illustrative net benefit.
This is not a recommendation to permanently hire a second crew.
A recurring staffing decision requires a representative mix of shifts, verified cost rates, operational validation and uncertainty analysis.
The observed result is conditional on identical service requirements under both policies and ignores yard spillback and shared equipment.

The capacity choice is a **plan** to evaluate, not a promise of the observed 0% tail fraction.
The four arrival times are a scenario, not a demand forecast.
A subsequent forecast might produce many scenarios with different burst patterns.
A management-approved service promise, if any, remains a separate **commitment**.

### Sensitivity and interpretation

At USD 0.75 per truck-minute, the avoided delay value is `109 × 0.75 = USD 81.75`.
Subtracting USD 120 gives a net cost of USD 38.25 in this scenario.
At USD 2.00 per truck-minute, the avoided delay value is USD 218.00 and net benefit is USD 98.00.
This is a sensitivity to the cost rate, not a probability distribution over future days.

If the extra crew is available only from minute 30, the two-door schedule above is no longer valid.
If a shared forklift permits only one truck to unload at a time, the second physical door alone may add no unloading capacity.
If an arrival is canceled, both the queue and economic outcome change.
Each is a structural or scenario change, not a rounding adjustment.

## 7. Code, query, or calculation

The following standard-library Python program implements the same deterministic event rules.
It also creates independent, synthetic paired scenarios to demonstrate experimentation.
Run it with Python 3.9 or later by saving it as `dock_des.py` and executing `python dock_des.py`.
No third-party packages or input files are required.

The stochastic generator is illustrative, not fitted to a real facility.
Each scenario fixes arrivals and service requirements before either policy runs.
Service durations are strictly positive integer minutes; arrivals may tie and are ordered by ID.
The event loop processes completions before arrivals at a common timestamp.
It drains all admitted trucks and computes busy time only within the 0–120 minute reporting window.
The synthetic comparison uses 1,000 independent scenario IDs; its exact numerical output is not asserted here because the focus is the reproducible method rather than a calibrated business estimate.

```python
from collections import deque
from dataclasses import dataclass
import heapq
import random
import statistics


@dataclass(frozen=True)
class Truck:
    truck_id: str
    arrival: int
    service: int


def simulate(trucks, doors, window=120):
    """FIFO dock; completion precedes arrival at the same timestamp.

    All trucks are admitted, doors stay available until the queue drains,
    and utilization is reported for [0, window).
    """
    if doors < 1 or window <= 0:
        raise ValueError("doors and window must be positive")
    ids = [truck.truck_id for truck in trucks]
    if len(ids) != len(set(ids)):
        raise ValueError("truck IDs must be unique")
    if any(t.arrival < 0 or t.service <= 0 for t in trucks):
        raise ValueError("arrivals must be nonnegative and service positive")

    # Event tuple: time, type priority, stable ID, payload.
    # Completions (0) precede arrivals (1) at a shared time.
    events = [(t.arrival, 1, t.truck_id, t) for t in trucks]
    heapq.heapify(events)
    waiting = deque()
    free = doors
    started = {}
    completed = {}
    busy_in_window = 0

    def dispatch(now):
        nonlocal free, busy_in_window
        while free and waiting:
            truck = waiting.popleft()
            free -= 1
            started[truck.truck_id] = now
            finish = now + truck.service
            busy_in_window += max(0, min(finish, window) - min(now, window))
            heapq.heappush(events, (finish, 0, truck.truck_id, truck))

    while events:
        now, kind, _, truck = heapq.heappop(events)
        if kind == 0:
            free += 1
            completed[truck.truck_id] = now
        else:
            waiting.append(truck)
        dispatch(now)

    waits = {t.truck_id: started[t.truck_id] - t.arrival for t in trucks}
    return {
        "waits": waits,
        "starts": started,
        "completions": completed,
        "total_wait": sum(waits.values()),
        "mean_wait": sum(waits.values()) / len(trucks) if trucks else 0.0,
        "over_30": sum(w > 30 for w in waits.values()) / len(trucks)
                   if trucks else 0.0,
        "completed_by_window": sum(t <= window for t in completed.values()),
        "utilization": busy_in_window / (doors * window),
    }


def make_scenario(seed):
    """Synthetic example: 4–8 trucks arriving in the first 80 minutes."""
    rng = random.Random(seed)
    count = rng.randint(4, 8)
    return [Truck(f"T{i:02d}", rng.randrange(80), rng.randint(20, 45))
            for i in range(count)]


def main():
    example = [Truck("A", 0, 35), Truck("B", 12, 28),
               Truck("C", 19, 42), Truck("D", 47, 24)]
    for doors in (1, 2):
        result = simulate(example, doors)
        print(f"{doors} door(s): waits={result['waits']}, "
              f"total={result['total_wait']} min, "
              f"utilization={result['utilization']:.4f}, "
              f"completed by 120={result['completed_by_window']}")

    # Each paired difference uses the same pre-generated trucks.
    differences = []
    for seed in range(1000):
        trucks = make_scenario(seed)
        one = simulate(trucks, 1)
        two = simulate(trucks, 2)
        differences.append(one["total_wait"] - two["total_wait"])
    mean = statistics.mean(differences)
    se = statistics.stdev(differences) / len(differences) ** 0.5
    print(f"Synthetic paired reduction: {mean:.2f} min/shift; "
          f"Monte Carlo SE: {se:.2f} min/shift")


if __name__ == "__main__":
    main()
```

The first two output lines should be:

```text
1 door(s): waits={'A': 0, 'B': 23, 'C': 44, 'D': 58}, total=125 min, utilization=1.0000, completed by 120=3
2 door(s): waits={'A': 0, 'B': 0, 'C': 16, 'D': 0}, total=16 min, utilization=0.5375, completed by 120=4
```

The third line reports a Monte Carlo mean and standard error in minutes per synthetic shift.
It is a simulation sampling summary, not a calibrated prediction or a complete confidence interval.
Use a suitable t critical value and an independently assessed input model for an interval used in a decision.
The deterministic example is deliberately simple enough to audit by hand.
The code does not implement calendars, finite yards, priority dispatch, resource failures or shared forklifts.
Add those features only with a corresponding specification, data and verification cases.

## 8. Data, standards, and regulatory context

### Minimum operational data contract

| Field | Meaning and unit | Quality check | Decision relevance |
|---|---|---|---|
| Truck or visit ID | Unique visit identifier | No duplicate active visits | Links arrival to completion |
| Site and door | Facility and resource identifier | Valid site and door master data | Capacity and routing |
| Dock-ready timestamp | Time truck can actually be served | Time zone and ordering checked | Queue entry |
| Service-start timestamp | Actual unloading start | Not before dock-ready time | Waiting-time measurement |
| Completion timestamp | Actual release of door | Not before service start | Service duration |
| Load type | Operational handling class | Stable definition over time | Heterogeneity and eligibility |
| Work content | Pallets, cases or another physical quantity | Missingness and unit checks | Service-duration drivers |
| Appointment timestamp | Planned arrival slot | Separate from actual arrival | Arrival process and adherence |
| Crew and equipment calendar | Available capacity over time | Compare with actual staffing | Feasible resource model |
| Exception code | Breakdown, hold or rework reason | Missing-code review | Interruptions and exclusions |

Store original timestamps and transformations with their time zone and daylight-saving rules.
A timestamp without a clear event definition may not be comparable across systems.
For example, a transport-management-system gate event may precede dock readiness by an uncertain yard delay.
An apparent negative wait can be a clock synchronization or definition problem rather than a fast process.

Keep a data dictionary, extraction date, filters, exclusions and version of the input sample.
Distinguish a missing timestamp from a zero-duration event.
Quantify the fraction of records excluded and compare their characteristics with retained records.
If difficult loads are disproportionately missing, the cleaned sample can systematically understate service time.

A simulation clock can use minutes since a shift start, but the source timestamps must first be normalized consistently.
When reporting across multiple shifts, preserve the actual calendar date and local operating calendar.
Avoid pooling closures, holidays and extraordinary disruptions into an ordinary day without explicitly modeling their frequency.

### Identifiers and interoperability

GS1's EPCIS standard defines a common structure for visibility events, including what, when, where, why and how information about business objects.
It can help exchange event records across organizational boundaries.
It does not require a site to build a DES or guarantee that a scanned timestamp equals the physical service-start time.
Check the applicable EPCIS version, trading-partner implementation and event semantics before using it as an input contract.

ANSI/ISA-95 addresses integration between enterprise and control systems and provides models and terminology for manufacturing operations.
It can inform the boundary between enterprise orders and operational resources.
It is not a general-purpose rule for queue discipline, nor does adopting it validate a simulation.
Use the version and scope agreed with the organization and suppliers.

These standards are voluntary unless incorporated into an applicable law, contract or other binding obligation.
A model team should not characterize a standard as universally mandatory merely because a customer uses it.
The authoritative contractual requirements and local legal obligations must be checked for the deployment.

### Privacy, security and operational rules

Driver names, badges, vehicle identifiers and precise movement histories may be personal data in some jurisdictions or contexts.
Whether a particular processing activity falls under the European Union General Data Protection Regulation (GDPR) depends on its facts and scope.
GDPR principles include data minimization and purpose limitation where the regulation applies; Article 5 is the primary text.
A DES generally needs stable pseudonymous visit IDs rather than a driver's name in every model record.
Pseudonymization reduces exposure but does not automatically make data anonymous or remove legal obligations.
Consult the organization's privacy and legal teams about lawful basis, retention, access and cross-border transfer.

Restrict access to facility layout, shipment details and live movements according to the site's security policy.
An offline planning model should not send control commands to doors or equipment.
A model-connected control system needs a separate safety and cyber-risk assessment.
The educational example is not a safety certification or legal opinion.

Working-time, vehicle-safety and site-access rules can constrain executable schedules.
The simulation must encode the specific applicable rules and actual labor agreement rather than assume a universal break schedule.
If a proposed policy violates a binding constraint, reject or redesign it even if simulated cost is lower.

### Measurement and uncertainty

Historical data measure the system under historical policies.
A new policy can change driver behavior, appointment adherence or service work content.
State the causal transport assumption before reusing historical inputs unchanged.
Compare the model's forecast period and operational population with the sample used for fitting.
Data collected only on weekdays may not support a weekend capacity recommendation.

A narrow Monte Carlo interval only reflects repeated sampling from the specified simulation input model.
It does not automatically include uncertainty in fitted parameters, unobserved disruption modes or a wrong process boundary.
Use alternative plausible input models, parameter resampling or scenario analysis when those uncertainties are material.
Document which uncertainty each reported interval includes.

## 9. Industry practice and case evidence

### Evidence and transfer

The following sources document simulation use or practice; none validates the fictional dock example.
A named case is not evidence that every facility will see the same savings.
Where a cited source is a vendor or technology-company case study, its results are claims by the source rather than independently audited outcomes unless otherwise stated.

**NIST's manufacturing simulation work.**
The U.S. National Institute of Standards and Technology (NIST) publishes resources on digital twins and manufacturing systems integration.
Its work discusses the relationship between physical manufacturing systems, data and simulation.
The transferable lesson is to define the connection and evidence needed for the intended application, not to label every offline simulation a twin.
The source does not establish the economics of an extra receiving door.

**Port of Rotterdam and IBM.**
IBM publicly describes work with the Port of Rotterdam involving a digital platform and a digital representation of port operations.
That is a named public technology-company case, not independent evidence of a quantified DES benefit for the dock in this lesson.
Port traffic involves shared berths, movements, weather and other interacting constraints.
[INFERENCE] A port planning study may benefit from event-based modeling when vessel arrivals and resource conflicts drive waiting.
The case should not be used to assert a specific savings percentage or a particular event model without further primary evidence.

**Automotive and manufacturing practice.**
Simulation software documentation and peer-reviewed manufacturing simulation literature describe using event-based models to evaluate lines, buffers and material handling.
An automotive factory's operating details, data quality and union agreements differ from a distribution dock.
Transferring the method requires a new boundary, calibrated inputs and local validation.
A model that correctly predicts a factory line does not automatically validate a yard model.

### From model to an operating decision

Before authorizing a capacity change, the manager needs an implementation design as well as an estimated benefit.
The following fictional receiving-site decision illustrates the handoff.
The current plan uses one staffed door from 06:00 to 14:00.
An alternative adds a second crew from 08:00 to 12:00, when appointments cluster.
The site has two physical doors, but the second requires a qualified forklift operator.
The proposed roster must identify that operator rather than assume one is available.

The study compares three feasible policies:

1. Keep one staffed door for the full shift.
2. Staff two doors during the fixed four-hour peak.
3. Call in the second crew when the queue reaches three trucks, subject to a 45-minute call-in delay.

The third policy needs a precise trigger.
Define queue length as trucks physically ready for a door but not yet in service.
Evaluate the trigger after each arrival and completion event.
Once called, the crew remains on site for at least four paid hours.
The crew cannot be called if a certified operator is unavailable.
If the queue falls during the call-in delay, the call remains active unless the actual labor agreement permits cancellation.
These details affect both waiting and cost; leaving them implicit makes the policy impossible to implement consistently.

For each simulated shift, record total waiting minutes and the number of trucks waiting over 30 minutes.
Also record the longest wait, the number of missed appointment commitments, paid crew-hours and work remaining at 14:00.
If overtime is authorized, record its hours separately from regular staffing.
If the site carries a backlog into the next shift, preserve each truck's accumulated wait rather than reset the clock.
Use the same sampled arrival and workload scenario for all three policies where the policy does not itself change those inputs.
Estimate policy differences across independent shifts, not across trucks within one shift.

Suppose a pilot comparison produces the following illustrative *mean* results per shift:

| Policy | Wait (truck-minutes) | Paid door-hours | Trucks over 30 minutes | Incremental cost versus baseline |
|---|---:|---:|---:|---:|
| One door | 280 | 8 | 3.0 | USD 0 |
| Fixed peak crew | 100 | 12 | 0.6 | USD 160 |
| Triggered call-in | 150 | 11 | 1.2 | USD 120 |

These values are hypothetical; they are not results from the code or a published facility.
At USD 1.20 per waiting truck-minute, the fixed crew avoids `180 × 1.20 = USD 216` in delay cost.
Its illustrative net benefit is USD 56 per shift after the USD 160 incremental staffing cost.
The triggered policy avoids `130 × 1.20 = USD 156` in delay cost.
Its illustrative net benefit is USD 36 per shift after the USD 120 incremental cost.
Under these assumptions, fixed peak staffing has USD 20 higher net benefit than triggered call-in.
The fixed policy also has fewer long-waiting trucks on average.

The averages are insufficient to establish a choice by themselves.
The decision brief should include paired uncertainty intervals and the frequency of shifts that breach the service target.
It should also show results for plausible delay rates and crew costs, including any minimum-call payment.
If a second operator is routinely unavailable, the fixed policy is not executable without recruitment or reassignment.
If the extra operator is reassigned from another activity, include the lost contribution or service effect of that activity.
Management can authorize a limited pilot only after checking safety, staffing, operating procedures and financial assumptions.

During a pilot, retain the baseline definition, measure actual dock-ready and service timestamps, and document deviations from the planned policy.
Compare like-for-like appointment volume and load mix; a quieter pilot week alone does not demonstrate a causal improvement.
Review the policy at a predetermined date with the site manager, labor representative and finance partner.
Escalate if queue spillback, safety incidents or overtime exceed agreed limits.
Update the model with the pilot evidence, but preserve a holdout period for evaluating any recalibration.

### Practical business cases

**Receiving dock staffing.**
A warehouse manager chooses one or two staffed doors for the morning peak.
The relevant entities are trucks; the primary resources are doors and crews.
The outcome is waiting and completion, together with incremental crew cost.
A key failure is treating an unstaffed physical door as available capacity.

**Order-picking wave release.**
A fulfillment manager chooses when to release orders to picking zones.
Entities can be orders or totes; pickers, stations and conveyors can be resources.
A policy that releases too much work may increase congestion and lead time even if every picker looks busy.
The model must represent shared space and actual release rules if those interactions drive the decision.

**Cross-dock transfer.**
A terminal manager compares staging-space and trailer-door allocations.
Pallets may wait for a destination trailer while inbound trucks occupy doors.
A finite staging area can block unloading; unlimited-buffer assumptions can hide that constraint.
Measure missed connections and overtime alongside throughput.

**Hospital supply replenishment.**
A supply team chooses replenishment routes and replenishment frequency for clinical units.
Entities may be replenishment requests, not patients; carts and staff are shared resources.
Safety stock, access windows and urgent requests affect service.
Clinical safety and local governance constrain implementation; a simulation is not a substitute for those controls.

**Repair-parts depot.**
A planner compares technician capacity and spare-part availability.
A repair job may wait for both a technician and a component.
A queue model that includes labor but assumes every spare is instantly available can misidentify the constraint.
Track backorders and downtime as well as labor utilization.

### How to use case evidence responsibly

Separate documented facts, company claims and your inference.
Record the source date, technology, population, baseline and measurement period.
Ask whether the study used a live intervention, retrospective model or counterfactual simulation.
A vendor's promised benefit is not a measured outcome.
A documented outcome is not necessarily causal if other changes occurred at the same time.

Before transferring a result, compare arrival variability, operating hours, labor rules, technology and cost structure.
Use the published case to identify hypotheses and failure modes, not to import a return on investment.
For a high-stakes investment, seek an independently measured local pilot where feasible.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Mean-only arrivals | Smooth model, bursty actual dock | Peaks are averaged away | Underestimated waits | Model time-varying and correlated arrivals |
| Wrong boundary | Gate-to-door time called dock wait | Different clocks and processes are mixed | Misleading service claim | Define entry and exit events explicitly |
| Invisible shared labor | Two doors modeled with one crew | A physical asset is mistaken for staffed capacity | Infeasible benefit | Model shared crew or justified capacity |
| Ignored finite yard | Simulated queue grows without limit | Spillback and rejected trucks disappear | Understated upstream disruption | Add finite space and overflow behavior |
| Incorrect tie order | Results change with record order | Simultaneous events lack a rule | Nonreproducible dispatch | Specify and test stable event priorities |
| Censored waits omitted | Only completed trucks reported | Long-waiting trucks disappear at close | Optimistic tail metric | Drain admitted work or report unfinished ages |
| Wrong utilization denominator | Utilization exceeds 100% | Work outside window divided by inside capacity | Invalid capacity conclusion | Clip busy time and align availability window |
| False independence | Narrow interval from thousands of trucks | Trucks in a shift share congestion | Understated uncertainty | Use independent shift-level replications |
| Seed-only pairing | Paired runs consume different draws | Policy changes random-number call order | No valid entity-level coupling | Pre-generate exogenous scenarios by ID |
| Double-counted benefit | Demurrage and generic delay both monetize same minute | Overlapping cost components | Inflated return | Build an incremental cost ledger |
| Overfitted input | One historical mean matches perfectly | Parameters tuned to the validation period | Poor future performance | Hold out periods and inspect several metrics |
| Stale calendar | Model allows work during planned breaks | Resource availability not maintained | Unrealistic throughput | Version and validate calendars |
| Omitted priorities | Aggregate wait looks acceptable | Critical class is delayed by FIFO | Service failure for critical loads | Represent eligibility and report by class |
| Perfect animation bias | Stakeholders trust visuals over trace | Appearance is confused with validity | Premature deployment | Review assumptions and quantitative evidence |
| No change ownership | Model recommendation has no shift owner | Operational dependencies are not resolved | Study is not implemented | Assign decision, staffing and monitoring owners |
| Unmodeled feedback | Faster dock changes arrival patterns | Historical arrival process no longer holds | Counterfactual bias | Test behavioral responses and sensitivity |

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | Test or response |
|---|---|---|---|
| Dock-ready arrivals are known | Defines the entry event | Gate times substitute for readiness | Compare gate, yard and dock timestamps |
| FIFO dispatch | Makes service order reproducible | Priority, appointments or manual overrides | Audit actual dispatch logs |
| Unlimited queue | Keeps the teaching model focused | Yard space fills or trucks are rejected | Measure peak occupancy and overflow |
| Dedicated crew per door | Makes each modeled door a usable resource | Staff are shared or absent | Check roster and observed utilization |
| Equal service requirement across policies | Enables paired comparison | More doors change congestion or crew skill | Measure service by staffing configuration |
| No breakdowns | Simplifies example trace | Door or forklift downtime is material | Examine downtime and add failure events |
| No service interruption | Represents continuous unloading | Breaks or handovers stop work | Compare elapsed and active service |
| All trucks compatible | Allows either door to serve any truck | Temperature, equipment or safety restrictions | Map load-door eligibility |
| Empty initial system | Defines the initial state | Overnight backlog or early trucks exist | Measure opening backlog by shift |
| All admitted trucks drain | Avoids hiding unfinished waits | Shift closure forces a handover or stop | Model overtime, carryover or censoring |
| Delay rate constant | Supports simple cost arithmetic | Contract charges are nonlinear or capped | Apply actual tariff and contract terms |
| Input sample represents future | Supports scenario relevance | Forecast volume or mix changes | Backtest and run demand/mix sensitivities |
| No endogenous arrivals | Allows fixed paired scenarios | Carriers change behavior under a new policy | Gather response evidence or model alternatives |
| No downstream blocking | Allows completion to release a door | Staging space fills | Inspect blocked-door and staging records |

The deterministic example is a model of four trucks, not a statistically representative sample.
It illustrates the mechanisms that a fuller study must quantify.
The Python demonstration samples a synthetic process, not real operating conditions.
No conclusion about a named facility follows from its numbers.

Simulation does not optimize a policy by itself.
It evaluates policies supplied to it; a separate search or optimization process may propose candidates.
It also cannot correct missing data or an incorrect causal assumption merely by increasing the number of replications.
More runs reduce Monte Carlo error conditional on the model, not structural error in the model.

## 12. Exercises

### Exercise 1 — Map a process

A parcel hub receives cages of parcels, sorts them at two staffed stations and loads outbound trailers.
A cage can wait if both sort stations are busy.
The manager is deciding whether to add a third staffed station for a two-hour peak.

- **Deliverable:** A table of entities, attributes, resources, events, queue, boundary and three performance measures.
- **Acceptance criteria:** Distinguish a cage from a station, specify queue selection and state the start and end timestamps.
- **Time:** 15 minutes.

### Exercise 2 — Trace a single door

Three trucks arrive at minutes 0, 5 and 12 with service durations 10, 9 and 4 minutes respectively.
A single door uses FIFO and serves immediately when free.

- **Deliverable:** Start, completion and wait for each truck; mean wait and number completed by minute 20.
- **Acceptance criteria:** Honor the chronological event sequence and report units.
- **Time:** 15 minutes.

### Exercise 3 — Same-time events

Truck A occupies the only door from minute 0 through completion at minute 10.
Truck B has waited since minute 7; truck C arrives at minute 10.
Service takes 5 minutes for B and 2 minutes for C.
Completion is processed before arrival, and existing waiting trucks receive the free door first.

- **Deliverable:** State the service order and start times for B and C.
- **Acceptance criteria:** Explain why the tie rule preserves the existing FIFO queue.
- **Time:** 10 minutes.

### Exercise 4 — Define shift boundaries

A site accepts arrivals until 18:00 and allows unloading overtime until 19:00.
Any unserved truck at 19:00 is carried over to the next shift.
Management wants both same-day carrier delay and overtime labor hours.

- **Deliverable:** A one-page specification of initial state, arrival cutoff, service cutoff, carryover and metrics.
- **Acceptance criteria:** Include unfinished waiting ages and distinguish shift capacity from overtime capacity.
- **Time:** 20 minutes.

### Exercise 5 — Modify the code

Use the code in Section 7 with two trucks arriving simultaneously at minute 0.
Let truck A require 10 minutes and truck B require 5 minutes; use one door.

- **Deliverable:** The input list and expected waits, starts and completions.
- **Acceptance criteria:** Follow the ID tie-break and show why the result does not depend on the input-list order.
- **Time:** 15 minutes.

### Exercise 6 — Calculate a paired comparison

Across four independent paired shifts, the reduction in waiting time from an added door is 20, 40, 30 and 50 truck-minutes.
The additional door costs USD 30 per shift and waiting costs USD 1 per truck-minute.

- **Deliverable:** Mean wait reduction, mean incremental net benefit and sample standard error of the reduction.
- **Acceptance criteria:** Use the four shift-level paired differences, not individual truck waits; show the denominator.
- **Time:** 20 minutes.

### Exercise 7 — Critique validation

A model matches average daily completions within 1%, but modeled 95th-percentile wait is 8 minutes while observed 95th-percentile wait is 55 minutes.
The input records lack staffing breaks and were fitted on the same month used for evaluation.

- **Deliverable:** Three reasons not to approve a tail-service decision and a prioritized evidence plan.
- **Acceptance criteria:** Address the tail discrepancy, missing calendar and lack of holdout; propose measurable checks.
- **Time:** 20 minutes.

### Exercise 8 — Recommend a policy

A facility can add a staffed door for USD 160 per shift.
A calibrated paired experiment estimates 180 avoided truck-minutes per shift, but the 95% interval is 80–280 minutes.
The site estimates delay cost at USD 1.20 per truck-minute and a binding safety rule requires a forklift operator at each open door.
Only one certified operator is currently rostered.

- **Deliverable:** A decision brief with expected net benefit, interval translated into net benefit, feasibility condition and next action.
- **Acceptance criteria:** Do not recommend opening an unstaffed door; distinguish simulation sampling uncertainty from cost and model uncertainty.
- **Time:** 25 minutes.

## 13. Self-check answers

### Exercise 1

The cage is the entity, with ID, arrival time and parcel count as possible attributes.
Each staffed sort station is a resource; capacity is two stations in the baseline and three in the candidate policy.
Arrival, service start and service completion are events.
The waiting cages form a queue whose discipline must be specified, such as FIFO by arrival with ID tie-break.
One defensible boundary runs from cage arrival at the sort queue to sorting completion.
Measures include cage waiting minutes, cages completed per two-hour peak and time-weighted station utilization.
If the decision concerns trailer departure instead, extend the boundary and include trailer readiness and loading capacity.

### Exercise 2

| Truck | Arrival (min) | Start (min) | Completion (min) | Wait (min) |
|---|---:|---:|---:|---:|
| 1 | 0 | 0 | 10 | 0 |
| 2 | 5 | 10 | 19 | 5 |
| 3 | 12 | 19 | 23 | 7 |

Mean wait is `(0 + 5 + 7) / 3 = 4` minutes per truck.
Two trucks complete by minute 20; the third is still in service.
The arrival at minute 12 joins the queue while truck 2 occupies the door.

### Exercise 3

At minute 10, A completes and B starts immediately because B is already waiting.
C then arrives and queues.
B completes at minute 15, when C starts.
Thus the order is B then C, with start times 10 and 15 minutes respectively.
The event priority and FIFO queue together avoid giving C precedence over B.

### Exercise 4

A strong answer defines the opening backlog, including the age and priority of every carried-over truck.
It admits only trucks arriving before the chosen 18:00 cutoff, with an explicit rule for exactly 18:00.
It models overtime staffing until 19:00 and stops or hands off service according to the stated real operating rule.
It reports completed trucks, waiting minutes, overtime resource-hours and unfinished trucks with waiting age at 19:00.
It describes whether an interrupted truck resumes the same service next shift and how its remaining work is retained.
A weak answer drops unfinished trucks or divides overtime work by normal-shift capacity.

### Exercise 5

Input: `Truck("A", 0, 10)` and `Truck("B", 0, 5)`.
The heap orders simultaneous arrival events by ID, so A starts at minute 0 and completes at minute 10.
B starts at minute 10 and completes at minute 15.
Waits are A = 0 minutes and B = 10 minutes.
The input-list order does not alter the event tuple's ID tie-break.
If the real process uses a different tie-break, change both the specification and implementation.

### Exercise 6

The mean paired reduction is `(20 + 40 + 30 + 50) / 4 = 35` truck-minutes per shift.
At USD 1 per truck-minute, mean avoided delay cost is USD 35 per shift.
After USD 30 in additional staffing, mean net benefit is USD 5 per shift.
The deviations from 35 are −15, 5, −5 and 15 minutes.
Their squared sum is 500 minute-squared.
The sample variance is `500 / 3 = 166.67` minute-squared and sample standard deviation is approximately 12.91 minutes.
The standard error is `12.91 / sqrt(4) = 6.45` truck-minutes per shift.
Four replications provide limited evidence; do not confuse the point estimate with a guaranteed saving.

### Exercise 7

The modeled tail is 47 minutes lower than observed, so an average-throughput match does not establish tail validity.
Missing breaks can reduce modeled congestion and should be compared with roster and clock records.
Fitting and evaluating on the same month risks overfitting and does not demonstrate performance on new periods.
First verify timestamp definitions and sample completeness; then incorporate actual calendars and evaluate on a held-out month.
Compare mean and 95th-percentile waits with uncertainty, time-of-day breakdowns and queue traces.
Set acceptance tolerances with the decision owner before using the model to decide a tail-service target.

### Exercise 8

Expected avoided cost is `180 × USD 1.20 = USD 216` per shift.
Expected net benefit after USD 160 staffing cost is USD 56 per shift.
Applying the same fixed cost and rate to the stated interval yields `80 × 1.20 − 160 = −USD 64` and `280 × 1.20 − 160 = USD 176` per shift.
This transformation assumes the reported interval is for mean avoided minutes and that the rate and capacity cost are fixed.
It does not include uncertainty in those cost assumptions or model structure.
The candidate cannot operate as described with only one certified operator.
Obtain a second certified operator and account for that person's full incremental cost before any pilot.
Given the interval crosses zero, collect further evidence or pilot with approved safety controls and an agreed review trigger.

## 14. Interview questions

1. **When would you use DES rather than a simple capacity ratio?**
   A strong answer connects decisions to time-dependent contention, variability and queue interactions.
   A weak answer says DES is always better because it has more detail.
2. **What is the difference between a waiting queue and the future-event list?**
   A strong answer distinguishes waiting entities from scheduled state changes.
   A weak answer uses the two interchangeably.
3. **What belongs in the model state?**
   A strong answer names the minimum information necessary to determine future transitions and explains omitted variables.
   A weak answer lists only summary output metrics.
4. **How should simultaneous events be handled?**
   A strong answer specifies a deterministic operational tie rule and tests it.
   A weak answer relies on accidental input or container order.
5. **How would you calculate utilization with a shift cutoff?**
   A strong answer clips busy time to the measurement window and divides by available resource-time in the same window.
   A weak answer divides total eventual service by shift capacity.
6. **Why can a validated mean conceal a poor model?**
   A strong answer checks tails, temporal patterns, process structure and held-out observations.
   A weak answer treats one matched aggregate as proof.
7. **What is a replication in a staffing experiment?**
   A strong answer identifies an independent shift scenario and paired policy outcomes.
   A weak answer treats each dependent truck in one queue as an independent replication.
8. **What do common random numbers require?**
   A strong answer preserves comparable exogenous inputs by scenario and entity, and checks that policy-dependent effects are modeled.
   A weak answer assumes resetting a seed guarantees a valid pairing.
9. **What does a narrow confidence interval not tell you?**
   A strong answer distinguishes simulation sampling error from input, structural and cost uncertainty.
   A weak answer says it proves the model is correct.
10. **How do you turn a simulation result into a staffing decision?**
    A strong answer uses incremental costs, operational constraints, uncertainty and an implementation owner.
    A weak answer recommends the highest-utilization policy without considering service or feasibility.

## 15. Key terms

- **Arrival:** entry into the chosen model boundary; see [definitions](#4-definitions-and-vocabulary).
- **Attribute:** entity-specific information used in rules; see [definitions](#4-definitions-and-vocabulary).
- **Boundary:** included process and measurement scope; see [Step 2](#step-2--draw-the-boundary-and-map-entities).
- **Capacity:** simultaneously available resource units; see [Step 3](#step-3--define-resources-and-calendars).
- **Censoring:** incomplete observation at a cutoff; see [Step 7](#step-7--define-initialization-and-stopping).
- **Common random numbers:** paired use of exogenous random inputs; see [Step 11](#step-11--design-a-comparison).
- **Completion:** event that ends service and releases capacity; see [event trace](#one-door-event-trace).
- **Confidence interval:** uncertainty interval for an estimated quantity; see [Step 11](#step-11--design-a-comparison).
- **Discrete-event simulation:** model driven by discrete state changes; see [mental model](#3-the-core-mental-model).
- **Dispatch rule:** policy for choosing eligible waiting work; see [Step 4](#step-4--specify-queue-admission-and-dispatch).
- **Entity:** individual item represented in the flow; see [Step 2](#step-2--draw-the-boundary-and-map-entities).
- **Event:** instantaneous state transition; see [Step 5](#step-5--define-the-event-list-and-ordering).
- **FIFO:** first-in-first-out queue discipline; see [Step 4](#step-4--specify-queue-admission-and-dispatch).
- **Flow time:** completion minus arrival; see [worked example](#one-door-event-trace).
- **Future-event list:** ordered scheduled transitions; see [Step 5](#step-5--define-the-event-list-and-ordering).
- **Initialization:** definition of the opening state; see [Step 7](#step-7--define-initialization-and-stopping).
- **Monte Carlo error:** sampling error from a finite number of simulated runs; see [code](#7-code-query-or-calculation).
- **Policy:** executable operational decision rule; see [Step 1](#step-1--define-the-decision-and-counterfactual).
- **Queue:** entities waiting for service; see [mental model](#3-the-core-mental-model).
- **Replication:** one independently generated scenario outcome; see [Step 11](#step-11--design-a-comparison).
- **Resource:** reusable constrained capacity; see [Step 3](#step-3--define-resources-and-calendars).
- **Scenario:** coherent set of input conditions; see [Step 6](#step-6--specify-inputs-and-their-dependence).
- **Service level:** defined performance measure against a target; see [worked example](#case-and-assumptions).
- **Simulation clock:** model time of the current event; see [mental model](#3-the-core-mental-model).
- **State:** facts needed to evolve the model; see [mental model](#3-the-core-mental-model).
- **Throughput:** completed entities per defined period; see [definitions](#4-definitions-and-vocabulary).
- **Utilization:** busy capacity divided by available capacity in the same window; see [worked example](#two-door-event-trace).
- **Validation:** evidence of fitness for a specific use; see [Step 10](#step-10--validate-for-the-decision).
- **Verification:** checking implementation against specification; see [Step 9](#step-9--verify-conservation-and-edge-cases).
- **Waiting time:** service start minus queue arrival; see [worked example](#one-door-event-trace).

## 16. Sources

1. Jerry Banks, John S. Carson II, Barry L. Nelson and David M. Nicol, *Discrete-Event System Simulation*, 5th ed., Pearson, 2010. Standard textbook on event-scheduling, input modeling, output analysis and validation.
2. Averill M. Law, *Simulation Modeling and Analysis*, 5th ed., McGraw-Hill Education, 2015. Background on simulation studies, verification, validation and experimental analysis.
3. Stewart Robinson, “Simulation model verification and validation: increasing the users' confidence,” *Proceedings of the 1997 Winter Simulation Conference*. [DOI: 10.1109/WSC.1997.640570](https://doi.org/10.1109/WSC.1997.640570). Verification and validation concepts.
4. Barry L. Nelson and David Matejcik, “Using common random numbers for indifference-zone selection and multiple comparisons in simulation,” *Management Science*, 41(12), 1995. [DOI: 10.1287/mnsc.41.12.1935](https://doi.org/10.1287/mnsc.41.12.1935). Paired simulation comparisons.
5. [Python `heapq` documentation](https://docs.python.org/3/library/heapq.html). Heap-based priority queues; consult the version matching your Python runtime.
6. [Python `random` documentation](https://docs.python.org/3/library/random.html). Pseudorandom generators and reproducibility qualifications.
7. [GS1 EPCIS standard](https://www.gs1.org/standards/epcis). Event-data interoperability and version information.
8. [ISA-95 standard overview](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard). Enterprise-control system integration scope.
9. [Regulation (EU) 2016/679, GDPR, official text](https://eur-lex.europa.eu/eli/reg/2016/679/oj). Article 5 sets out principles for processing personal data; applicability depends on the circumstances.
10. [NIST Digital Twins program](https://www.nist.gov/digital-twins). Official manufacturing and digital-twin research context.
11. [IBM, Port of Rotterdam customer story](https://www.ibm.com/case-studies/port-of-rotterdam). Company-published case; treat claims as attributed, not independent validation of this lesson's model.

Source scope and availability should be checked at the time of use.
No source above substantiates the fictional worked example's specific costs or outcomes.

## 17. Links

- [Module 20 topic map and planned sibling topics](_index.md).
- [Module 20 survey: digital twins, simulation and optimization](../../../course/20_digital_twins_simulation_optimization_uncertainty.md).
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md).
- [Terminal simulation: maritime, rail and air operations](../../07_engineering_electives/41_maritime_rail_air_operations/08_terminal-simulation.md).
- [Layout simulation in plant engineering](../../07_engineering_electives/45_plant_layout_detailed_engineering/09_layout-simulation.md).
- The module map lists the adjacent topics on input modeling and validation; consult their status before assigning them.
