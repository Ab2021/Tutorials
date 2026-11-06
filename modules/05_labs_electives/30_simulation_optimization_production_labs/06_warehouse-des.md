# Warehouse Discrete-Event Simulation

> Module 30 — Simulation, Optimization, and Real-Time Production Labs  
> Topic 06 of 10  
> Estimated study time: 3–4 hours  
> Prerequisites: basic Python 3, arithmetic, first-in-first-out queues, and the [environment lab](01_environment-repo.md)  
> Version 2026.5

## 1. Why this topic exists

A fulfillment manager must decide whether another picker, another packer, or a process change will improve the flow of orders.
An average processing time alone cannot answer that question.
Orders arrive unevenly, wait for shared capacity, and leave one station at different times.
A change that helps one station can simply move the queue to the next.

Discrete-event simulation (DES) represents individual state changes at their actual event times.
It provides a controlled sandbox for testing a proposed configuration before changing a shift.
The relevant decision is not whether the simulation looks realistic on screen.
The decision is whether the evidence is adequate to choose a safe operational experiment.

The supervisor needs a credible estimate of delay and throughput.
The labor planner needs a capacity and shift definition.
The operations analyst needs traceable inputs and a comparison that is not dominated by random noise.
The safety owner needs to know which physical and human constraints have been omitted.
The data owner needs a clear account of what timestamps mean and how records were selected.

The miniature line in this lab receives orders, picks them, and packs them.
Each station has a first-in-first-out (FIFO) waiting queue and interchangeable workers.
An order has a release time and one service requirement at each station.
The model does not reserve inventory, route a worker through aisles, or simulate a conveyor.

The main artifact is an executable Python script with an embedded, fully specified fixture.
The fixture is invented for instruction; it is neither a warehouse extract nor a vendor case study.
Its observed timestamps are synthetic holdout observations, not measured production outcomes.
They let you practice detecting a known discrepancy without implying that validation is complete.

A second experiment creates seeded, synthetic replications and compares scenarios using identical order streams.
The fixed fixture is small enough to calculate by hand.
The stochastic experiment illustrates reproducibility and sampling uncertainty rather than estimating any particular site's future performance.
Neither result is a staffing commitment, a forecast, or an approved production change.

## 2. Learning outcomes

By the end of this lab you can:

1. Draw an order's journey through two stations and identify entities, resources, queues, and events.
   Exercise 1 checks the flow and ownership of resources.
2. Compute service starts, departures, waits, and cycle times from an event fixture.
   Exercise 2 checks each timestamp and denominator.
3. Run a deterministic and a seeded stochastic experiment from the same executable artifact.
   Exercise 3 checks reproducibility and scenario pairing.
4. Calculate time-weighted utilization and explain why a high value alone does not establish a bottleneck.
   Exercise 4 checks exposure and capacity assumptions.
5. Identify a plausible constraint using queues, interventions, and downstream effects together.
   Exercise 5 asks you to interpret a capacity comparison.
6. Compare simulated outputs with a held-out observed trace and diagnose a discrepancy.
   Exercise 6 checks cohort matching and possible omitted events.
7. Design tests for event order, conservation, and simultaneous completions.
   Exercise 7 asks for boundary cases and expected behavior.
8. Specify a controlled field-validation and release plan.
   Exercise 8 requires acceptance gates and a rollback owner.

These outcomes distinguish model execution from model validation.
Reproducing the same wrong answer is useful for debugging, but it is not evidence that the model describes a warehouse.

## 3. The core mental model

```text
                     release event
                          |
                          v
                    [pick FIFO queue]
                          |
                    seize picker
                          |
                   pick completion
                          |
                    release picker
                          v
                    [pack FIFO queue]
                          |
                    seize packer
                          |
                   pack completion
                          |
                    release packer
                          v
                    completed order
```

The entity is an order, not an individual item or a worker.
The entity carries its ID, release time, and two required service durations.
The picker and packer pools represent scarce, reusable resources.
A queue holds eligible work when every resource in its pool is busy.
A station starts service immediately when an order is eligible and a resource is free.

A release changes the eligible population.
A pick completion releases a picker and makes an order eligible for packing.
A pack completion releases a packer and removes an order from the system.
Between events the state does not change, even though the simulation clock advances.
This is why event time, rather than a fixed one-minute iteration, is a natural representation.

The implementation uses a valid special case of an event scheduler.
It computes the earliest available resource with a min-heap and schedules each FIFO job at its earliest feasible start.
It first calculates all pick completions, then processes those completions in chronological order at the pack station.
That decomposition is exact for this two-stage feed-forward system because downstream state never blocks or changes the upstream station.
It would not be exact with finite buffers, blocking, shared pick-and-pack workers, recirculation, or feedback.
A general event calendar is necessary when such interactions are part of the decision.

Two events may share a timestamp.
For this lab, completion releases capacity at that timestamp before a waiting order starts.
Within a station, ties in eligibility are ordered by order ID.
An order that completes picking at minute 3 may begin packing at minute 3.
Zero elapsed time between these events does not imply a negative wait or overlapping use of a worker.

The model uses minutes from the beginning of the observation window.
Time zero is a reference, not a time-zone-aware wall-clock timestamp.
The fixed example starts empty and ends only after all five released orders are packed.
Its denominator for utilization is the interval from minute 0 to the final pack completion.
For a real shift, opening work in process and the scheduled observation interval must be handled explicitly.

Follow a unit of work through the diagram:

- On release, the order enters the pick queue or starts picking.
- At pick start, one picker becomes occupied by that order.
- At pick finish, the picker is freed and the order becomes pack-eligible.
- At pack start, one packer becomes occupied.
- At pack finish, the packer is freed and the order leaves the modeled system.
- No order can be completed twice or disappear from the trace.

A queue length is a state at a time; a queue wait is a property of an order.
The mean of queue snapshots at irregular event times is not the time-weighted mean queue length.
The lab computes individual waits, which are directly auditable from timestamps.
If queue-length exposure matters, integrate queue length over each interval between consecutive events.

An intervention is not a forecast.
Adding a worker in the simulator changes an assumption about available capacity.
It does not establish that a qualified worker is available, that space permits parallel activity, or that demand will match the input.
Use it to form a testable hypothesis, not to promise a labor reduction.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Entity | One modeled order with a unique ID. | Treating one order as one item when order sizes vary. |
| Resource | A pool of interchangeable workers with integer capacity, in workers. | Counting a worker twice across simultaneous stations. |
| Pick station | The service step between release and pick completion. | Including unrecorded travel in both arrival delay and service. |
| Pack station | The service step between pick completion and final completion. | Assuming every order has the same packaging requirements. |
| FIFO | First eligible order starts first; ties use order ID. | Assuming arrival order at pick equals arrival order at pack. |
| Event | An instantaneous release, start, or completion at a time in minutes. | Confusing event timestamps with service durations. |
| Event calendar | Time-ordered list of pending state changes. | Processing events in data-file order rather than time order. |
| Event tie | Two events with the same simulated time. | Leaving the order of capacity release and acquisition unspecified. |
| Simulation clock | Elapsed minutes in the synthetic experiment. | Treating simulated minutes as local timestamps without a start date. |
| Release | Time at which an order is eligible for picking, in minutes. | Using order creation when the warehouse receives work later. |
| Pick start | Time a picker begins the order, in minutes. | Using a scan that happens after physical work has already begun. |
| Pick finish | Time picking is complete and packing becomes possible, in minutes. | Using the last item scan without checking other completion work. |
| Pack start | Time a packer begins the order, in minutes. | Treating first pack scan as necessarily identical to actual start. |
| Pack finish | Time the modeled order leaves the system, in minutes. | Equating label creation with carrier handover. |
| Pick wait | Pick start minus release, in minutes per order. | Averaging only immediately started orders. |
| Pack wait | Pack start minus pick finish, in minutes per order. | Including transit already included in pack service. |
| Cycle time | Pack finish minus release, in minutes per order. | Comparing different start or end definitions. |
| Service requirement | Time occupied by one station on one order, in minutes. | Interpreting queue waiting as processing. |
| Busy time | Sum of occupied resource-minutes in the observation window. | Summing overlapping jobs without accounting for pool capacity. |
| Utilization | Busy resource-minutes divided by capacity times exposure minutes. | Dividing by one worker's minutes when there are two workers. |
| Makespan | Last pack finish minus the chosen time origin, in minutes here. | Comparing windows with different starting backlogs. |
| Throughput | Number completed divided by a specified time interval, orders per minute. | Dividing completed orders by service time instead of elapsed time. |
| Work in process | Released orders not yet finished, measured in orders. | Counting orders that have not been released. |
| Conservation | Released equals completed plus remaining work in process. | Ignoring rejected, canceled, or split orders. |
| Bottleneck | A capacity or policy constraint that limits a specified outcome under a specified workload. | Naming the highest-utilization station without an intervention. |
| Warm-up | Initial period discarded to reduce sensitivity to arbitrary starting state. | Discarding real opening work from a finite-shift analysis. |
| Censoring | An order's true completion or wait is not observed within the data window. | Treating an unfinished order as having zero remaining time. |
| Calibration | Estimating model inputs or parameters from a designated data set. | Tuning on the same observations used for final validation. |
| Validation | Assessing fitness for the intended decision against independent evidence. | Calling a successful script run operational validation. |
| Verification | Checking that the implemented model follows its specification. | Treating a good historical fit as proof that code is correct. |
| Replication | One independently seeded realization of a stochastic experiment. | Treating correlated orders within one run as independent replications. |
| Common random numbers | Using the same exogenous random inputs across compared scenarios. | Reusing one global stream when scenarios consume draws in different orders. |
| Paired difference | Scenario result minus baseline result within the same replication. | Comparing only two independently sampled means. |
| Holdout | Observations withheld from fitting and scenario selection. | Calling a tuned-on trace independent validation. |
| Trace | Per-order sequence of timestamps and derived metrics. | Reporting only a summary that cannot be reconciled. |
| Fixture | Explicit small input with known expected behavior. | Presenting invented observations as operational measurements. |

## 5. The mechanics, step by step

### Step 1 — Frame the decision

**Input:** a proposed staffing decision, a defined workload, and a service objective.
**Action:** write one question with a horizon and a metric.
**Output:** a decision statement that can be disproved.

For example: under a given eight-hour shift workload, does one extra packer reduce the fraction of orders waiting more than 15 minutes at packing?
That is different from asking whether a packer is busy.
Specify whether the added person works the full shift or only a peak period.
Identify the owner who can approve the experiment and the safety constraints that cannot be traded away.

Do not infer a staffing recommendation from the fixture in this document.
Five orders and invented durations cannot represent seasonal variation, work mix, absences, or an operating shift.
The fixture instead demonstrates the arithmetic and the mechanics of comparing capacity.

A decision may have multiple outcomes.
Track delay, throughput, resource exposure, and any unacceptable service breach separately.
If an objective is monetary, state the currency, wage treatment, and time horizon.
No wage or savings value is assumed in this lab.

### Step 2 — Draw the system boundary

**Input:** a process map and definitions agreed with a warehouse operator.
**Action:** mark the first and last modeled events and identify every queue within that boundary.
**Output:** a boundary diagram and a list of exclusions.

This lab starts when an order is released for picking.
It ends when packing is complete.
Put-away, replenishment, quality inspection, consolidation, shipping, and delivery are outside the boundary.
An actual site may need any of these steps, especially if they block release or the next station.

Write down whether an order can be canceled, split, combined, or reworked.
The runnable model permits none of these transitions.
Its conservation equation is therefore especially simple: released orders equal packed orders after the drain period.
A production process with cancellation requires a separate cancellation state and denominator.

The entity granularity must match the decision.
If a single order contains multiple lines, the service duration can represent total order work only if that aggregation is justified.
If lines can be picked in parallel and the order waits for the last line, model the fork and join explicitly.
Do not add line-level conclusions to an order-level model.

### Step 3 — Define state and resources

**Input:** the process boundary, worker schedule, and resource assignment rules.
**Action:** define each resource pool, eligibility rule, and queue discipline.
**Output:** an unambiguous state specification.

The state in this example consists of released orders, station availability, order status, and completion timestamps.
The picker pool and packer pool have separate capacities.
Each worker handles at most one order at a time and can start a new one immediately after completing a previous order.
Workers within the same pool are assumed interchangeable.
They do not move between pools in the same run.

A FIFO queue chooses the earliest eligible order first.
For the pick queue, eligibility begins at release.
For the pack queue, eligibility begins at pick finish.
Ties are resolved by order ID so that two reruns do not depend on incidental dictionary order.
In a real facility, priorities, cutoff times, skills, or batching may replace FIFO.

A fixed capacity over the entire horizon is an explicit simplification.
Breaks, meetings, shift handoffs, training, and walking between zones can change availability.
In particular, the synthetic observed holdout contains a one-minute unexplained pack delay.
The deterministic model has no mechanism that generates that delay.

### Step 4 — Specify event semantics

**Input:** eligible times, service durations, station capacities, and queue rule.
**Action:** specify the state change for every event and the ordering of simultaneous events.
**Output:** a transition table and testable invariants.

| Event | Before | Action | After |
|---|---|---|---|
| Release | Order not yet available to pick. | Add order to pick eligibility. | Order waits or begins pick. |
| Pick start | Order eligible; picker free. | Allocate one picker. | Picker busy until finish. |
| Pick finish | Order being picked. | Free picker; make order pack-eligible. | Order waits for pack or begins pack. |
| Pack start | Order pack-eligible; packer free. | Allocate one packer. | Packer busy until finish. |
| Pack finish | Order being packed. | Free packer; record completion. | Order leaves modeled system. |

A worker freed at time `t` is available to start a new order at time `t`.
This convention uses half-open occupancy intervals `[start, finish)`.
Two jobs may touch at the boundary without overlapping.
When multiple orders become eligible at `t`, sort them by ID.
The rule is part of the model, not an implementation accident.

Every event must preserve nonnegative elapsed durations.
For every order, `release <= pick_start <= pick_finish <= pack_start <= pack_finish`.
A station with capacity `c` can have no more than `c` overlapping service intervals.
An order's wait is the difference between its start and eligibility time.

The small script requires strictly positive service times and nonnegative releases.
The assumed times are integer minutes, but the scheduling arithmetic would also work with consistent numeric time units.
Use actual timestamp arithmetic and documented rounding when adapting the method to real records.

### Step 5 — Construct an input contract

**Input:** an order fixture or a cleaned and documented historical extract.
**Action:** define one record per entity and the unit and meaning of every field.
**Output:** a table that can be checked before simulation.

| Field | Type | Unit | Meaning |
|---|---|---|---|
| `id` | Unique string | None | Stable order identifier and tie-breaker. |
| `release` | Nonnegative integer | Minute | Eligibility for the pick queue. |
| `pick` | Positive integer | Minute | Required picker service duration. |
| `pack` | Positive integer | Minute | Required packer service duration. |

The executable embeds five such records.
You do not need network access, a database, or third-party Python packages.
The fixture should remain unchanged while checking the expected trace.
It is safe to edit a copy for exercises; retain the original as the baseline.

Never infer a service duration by subtracting release from completion when that interval includes queueing.
If only release and completion exist, service and waiting times are not separately identifiable without additional assumptions.
Seek start and finish scans, time studies, or direct observations.
Record their limitations before fitting distributions.

Input order must not determine queue order.
The script explicitly sorts by eligibility time and then ID.
Unique IDs prevent tie ambiguity and accidental overwriting of results.
For real data, check duplicates, missing releases, impossible timestamps, and inconsistent units before modeling.

### Step 6 — Schedule the pick stage

**Input:** release times, pick durations, and picker count.
**Action:** maintain a min-heap of worker availability times.
**Output:** pick start and finish for every order.

Initially every picker's next available time is zero.
Visit orders in `(release, id)` order.
Remove the earliest available picker time from the heap.
Start at the maximum of release time and that availability time.
Finish after the order's pick duration and return the new availability time to the heap.

This rule implements FIFO for a single nonpreemptive station with identical workers.
Because jobs do not overtake in the eligibility list, an earlier eligible order receives the earliest possible free worker.
Completion order can nevertheless differ from start order when multiple workers have different service durations.
That distinction matters at the next queue.

A worker cannot be assigned to two overlapping jobs.
If two workers finish together, either identical worker may take the next job.
Worker identity is not reported because this lab assumes interchangeability.
If worker skills or travel matter, worker identity must become part of the state.

### Step 7 — Schedule the pack stage

**Input:** pick completion times, pack durations, and packer count.
**Action:** sort orders by `(pick_finish, id)` and run the same station scheduler.
**Output:** pack starts, pack finishes, and one trace record per order.

Sorting by pick finish is essential when multiple pickers are present.
An order released first may finish picking after another order.
The pack queue is FIFO by pack eligibility, not by the original release sequence.
For a tie, the stated ID rule supplies a reproducible ordering.

Packing does not change picker availability in this model.
A pack queue of any length is allowed, and pickers never wait for a free buffer.
If the real handoff area has a strict capacity, this assumption can overstate pick throughput.
A finite buffer introduces a dependency from the downstream queue back to the upstream resource.
Replace the two-pass scheduler with a coupled event model before evaluating that condition.

### Step 8 — Define metrics before running scenarios

**Input:** a trace and a declared observation window.
**Action:** calculate waits, cycle time, completions, throughput, and utilization using fixed denominators.
**Output:** a metric dictionary and a reconciliation table.

For order `i`, pick wait is `pick_start_i - release_i`.
Pack wait is `pack_start_i - pick_finish_i`.
Cycle time is `pack_finish_i - release_i`.
Total time in the system equals the sum of both waits and both service requirements for this simple flow.

Average pick wait is the sum of pick waits divided by the number of included orders.
The average pack wait has the analogous denominator.
The mean of the order-level cycle times is not the same as makespan.
Makespan runs from minute zero to the last completion in this lab.

The script calculates utilization over `[0, makespan)`.
For a pool with `c` workers, the denominator is `c * makespan` worker-minutes.
The numerator is the sum of all service durations in that pool.
This full-drain calculation is valid because every modeled service interval lies inside the window.
For a truncated window, calculate only the overlap of each interval with the window.

For a scheduled shift, choose the denominator deliberately.
If a worker is not rostered for part of the shift, do not count that period as available capacity unless that is the desired measure.
Report both scheduled-capacity utilization and actual-availability utilization if the distinction matters.
Neither measure alone proves causality.

### Step 9 — Verify the fixed trace

**Input:** the five-order fixture and one worker at each station.
**Action:** hand-calculate the timeline and compare it with the script's per-order trace.
**Output:** an auditable baseline and invariant checks.

The full table appears in section 6.
In brief, the picker finishes at minutes 3, 5, 9, 10, and 12.
The packer finishes at minutes 5, 8, 10, 14, and 16.
The total pick wait is 17 order-minutes, and the total pack wait is 2 order-minutes.
Both stations accumulate 12 worker-minutes of service.

Check conservation: five orders are released, and five are completed after the drain.
Check that every finish is at least its start.
Check that service durations in the trace equal the fixture's required durations.
Check that no single-worker service intervals overlap.
The script includes assertions for the baseline, tied eligibility, and the capacity comparisons.

A test that recomputes the same expression as the production code is weak by itself.
Hand-derived timestamps and deliberately constructed edge cases provide independent checks.
Keep those expectations fixed when refactoring the scheduler.

### Step 10 — Compare interventions

**Input:** baseline trace, candidate capacities, and the same five orders.
**Action:** run baseline `(1 picker, 1 packer)`, extra-picker `(2, 1)`, and extra-packer `(1, 2)` cases.
**Output:** a table of outcomes and a statement of what can and cannot be inferred.

The baseline finishes at minute 16.
Adding a picker finishes the five-order batch at minute 15 and moves more waiting to packing.
Adding a packer finishes the batch at minute 14 while leaving the pick schedule unchanged.
Those outcomes do not make either intervention universally preferable.
Their values depend on the chosen objective, costs, staffing, and input sequence.

Mean pick waiting identifies substantial upstream delay in the baseline.
The capacity experiments show that packing can also limit final completion.
The station with the larger average queue delay need not yield the largest improvement in makespan when its capacity is increased.
The result is an illustration of a shifting constraint, not a definitive ranking of a real warehouse's bottlenecks.

For a real decision, add realistic candidate policies and quantify their full consequences.
A second picker might require a second cart or increase congestion.
A second packer might be physically impossible at a single packing bench.
The current script assumes no such interference and therefore supplies an optimistic capacity experiment.

### Step 11 — Generate paired stochastic replications

**Input:** a generator specification, a seed policy, a number of replications, and candidate capacities.
**Action:** create each order stream once and pass the identical stream to every scenario in that replication.
**Output:** paired scenario differences across replication seeds.

The script uses `random.Random(seed)` and keeps the resulting order list separate from the scenario scheduler.
Each stream contains 30 orders with synthetic integer interarrival and service durations.
The distributions are deliberately simple and are not calibrated to a site.
Seeds 100 through 119 identify 20 independent pseudorandom streams within the script's assumptions.

The baseline and candidate consume the same generated orders within each seed.
That pairing removes variation caused solely by comparing different workloads.
The script reports the mean makespan difference and the standard error of paired differences.
A negative candidate-minus-baseline value means the candidate finished that synthetic batch sooner.
A positive value would indicate a slower completion for that metric.

The sample standard deviation of paired differences uses `n - 1` in its denominator.
The standard error is that sample standard deviation divided by the square root of `n`.
It measures Monte Carlo variability in the estimated mean difference for the stated generator.
It does not account for uncertainty in whether the generator resembles a real warehouse.

The 20 replications are a teaching setting, not a prescribed precision target.
For an actual experiment, decide an acceptable uncertainty width before running it.
Report replication count, seeds, model version, scenario definitions, and uncertainty interval.
For a small sample, a suitable Student-t interval may be preferable to a normal approximation if its assumptions are defensible.
The script reports standard error rather than claiming a formal interval.

Do not seed each scenario separately and assume it received the same order stream.
If scenario branches consume different numbers of random draws, even a shared initial seed can desynchronize subsequent draws.
Pre-generating exogenous work, as this lab does, makes the paired workload explicit.
If future extensions introduce random failures or worker-dependent service, define shared and scenario-specific random streams deliberately.

### Step 12 — Compare against held-out observations

**Input:** independent observed release, start, and finish timestamps for the same defined cohort.
**Action:** compare trace-level and aggregate metrics without altering parameters on the holdout.
**Output:** a validation report listing errors, coverage, and unresolved discrepancies.

The embedded synthetic holdout uses the same five order IDs and includes observed timestamps for both stations.
The final order's observed packing starts one minute later than the model predicts.
Its other observed timestamps match the deterministic fixture.
Consequently the observed mean cycle time is 8.8 minutes and the simulated mean is 8.6 minutes.
The model underestimates that mean by 0.2 minutes for this invented cohort.

The observed total pack waiting is 3 order-minutes versus 2 simulated order-minutes.
The difference is localized to order E rather than distributed evenly across the line.
A plausible hypothesis is a one-minute interruption or unrecorded handoff delay.
The fixture does not identify the actual cause; do not report that hypothesis as an observed fact.

In a real validation, calibrate on earlier shifts and lock the inputs before evaluating later shifts.
Use the same definition of release and finish in the model and observed records.
Include completed and still-open orders consistently; otherwise fast orders can be overrepresented.
Compare distributions, tails, and peak-period queues as well as means.
A five-order synthetic holdout cannot establish predictive validity.

### Step 13 — Translate findings into an experiment plan

**Input:** validation discrepancies, intervention effects, and operational constraints.
**Action:** distinguish a model finding from a field-test proposal.
**Output:** a reviewable decision memo with conditions for proceeding and stopping.

A field plan should name a site, shift, workload, metric definition, owner, and baseline period.
It should specify what changes, when, and how the effect will be measured.
It should cover safety review, labor consultation where required, and the possibility of demand mix changes.
It should state who can stop the trial and how the prior operating mode is restored.

Require acceptance criteria to be set before examining trial results.
If the simulation materially misses historical queue or completion behavior, fix or bound the discrepancy before using it to choose a trial.
A simulation-only effect is a hypothesis; only a properly measured field trial supplies evidence of actual site impact.

## 6. Worked example

### Fixture and assumptions

A training line receives five synthetic orders.
The time origin is the beginning of a finite batch observation, in minutes.
One picker and one packer are available from minute 0 until the batch is drained.
Both stations are FIFO, and IDs break ties in eligibility.
There are no breaks, failures, walking, priorities, or capacity constraints on the intermediate buffer.

| Order | Release, min | Pick service, min | Pack service, min |
|---|---:|---:|---:|
| A | 0 | 3 | 2 |
| B | 1 | 2 | 3 |
| C | 2 | 4 | 1 |
| D | 3 | 1 | 4 |
| E | 4 | 2 | 2 |

These are invented service requirements.
They are not a sample from a real warehouse and should not be interpreted as benchmark rates.

### Pick-stage calculation

Order A is released at minute 0 when the picker is free.
It starts at minute `max(0, 0) = 0` and finishes at `0 + 3 = 3`.
Its pick wait is `0 - 0 = 0` minutes.

Order B is released at minute 1 while A is in service.
It starts at `max(1, 3) = 3` and finishes at `3 + 2 = 5`.
Its pick wait is `3 - 1 = 2` minutes.

Order C is released at minute 2.
It starts at `max(2, 5) = 5` and finishes at `5 + 4 = 9`.
Its pick wait is `5 - 2 = 3` minutes.

Order D is released at minute 3.
It starts at `max(3, 9) = 9` and finishes at `9 + 1 = 10`.
Its pick wait is `9 - 3 = 6` minutes.

Order E is released at minute 4.
It starts at `max(4, 10) = 10` and finishes at `10 + 2 = 12`.
Its pick wait is `10 - 4 = 6` minutes.

The sum of pick waits is `0 + 2 + 3 + 6 + 6 = 17` order-minutes.
Mean pick wait is `17 / 5 = 3.4` minutes per order.
The total picker service is `3 + 2 + 4 + 1 + 2 = 12` worker-minutes.

### Pack-stage calculation

A is eligible for packing at minute 3.
The packer is free, so A runs from minute 3 to minute 5.
A waits `3 - 3 = 0` minutes for packing.

B is eligible at minute 5, exactly when A finishes.
The half-open interval convention allows B to start at minute 5.
B finishes at minute 8 and waits zero minutes.

C becomes eligible at minute 9, after B finishes.
The packer is idle from minute 8 to minute 9.
C starts at minute 9 and finishes at minute 10.
C waits zero minutes.

D becomes eligible at minute 10, exactly when C finishes.
D starts at minute 10 and finishes at minute 14.
D waits zero minutes.

E becomes eligible at minute 12 while D is in service.
E starts at minute 14 and finishes at minute 16.
E waits `14 - 12 = 2` minutes.

The sum of pack waits is `0 + 0 + 0 + 0 + 2 = 2` order-minutes.
Mean pack wait is `2 / 5 = 0.4` minutes per order.
The total packer service is `2 + 3 + 1 + 4 + 2 = 12` worker-minutes.

### Reconciled trace

| ID | Release | Pick start | Pick finish | Pack start | Pack finish | Pick wait | Pack wait | Cycle |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A | 0 | 0 | 3 | 3 | 5 | 0 | 0 | 5 |
| B | 1 | 3 | 5 | 5 | 8 | 2 | 0 | 7 |
| C | 2 | 5 | 9 | 9 | 10 | 3 | 0 | 8 |
| D | 3 | 9 | 10 | 10 | 14 | 6 | 0 | 11 |
| E | 4 | 10 | 12 | 14 | 16 | 6 | 2 | 12 |

All table times and waits are in minutes.
For E, `6 + 2 + 2 + 2 = 12` minutes: pick wait plus pick service plus pack wait plus pack service.
The sum of the cycle times is `5 + 7 + 8 + 11 + 12 = 43` order-minutes.
The mean cycle is `43 / 5 = 8.6` minutes per order.
The makespan is 16 minutes from time zero to the last pack completion.
Batch throughput over that window is `5 / 16 = 0.3125` orders per minute.

Both resources are busy for 12 worker-minutes over a 16-minute exposure.
Picker utilization is `12 / (1 * 16) = 0.75`, or 75%.
Packer utilization is also `12 / (1 * 16) = 0.75`, or 75%.
Identical utilization hides the different waiting patterns and service sequences.

### Capacity scenarios

The same five orders and service requirements are used in all three runs.
Only the capacity named in each row changes.

| Scenario | Pickers | Packers | Makespan, min | Total pick wait, order-min | Total pack wait, order-min |
|---|---:|---:|---:|---:|---:|
| Baseline | 1 | 1 | 16 | 17 | 2 |
| Extra picker | 2 | 1 | 15 | 1 | 19 |
| Extra packer | 1 | 2 | 14 | 17 | 0 |

With two pickers, A and B finish picking at minute 3.
D finishes at minute 4, E at minute 6, and C at minute 7.
The packer receives these orders in `A, B, D, E, C` order.
Their pack intervals are `[3,5)`, `[5,8)`, `[8,12)`, `[12,14)`, and `[14,15)`.
Pack waits are 0, 2, 4, 6, and 7 minutes, totaling 19 order-minutes.
Only C waits for a picker, for one minute.

With two packers, the pick schedule is unchanged.
Each order can begin packing immediately after picking.
D occupies a packer from minute 10 to 14, while E occupies the other from minute 12 to 14.
The last pack finish is minute 14, and no order waits for packing.

The extra packer reduces makespan by `16 - 14 = 2` minutes on this batch.
The extra picker reduces makespan by `16 - 15 = 1` minute.
The extra picker substantially reduces pick waiting but increases pack waiting.
These results are conditional on a tiny deterministic workload and unlimited intermediate storage.

### Synthetic holdout comparison

The following observation table is deliberately invented as a validation exercise.
It is separate from the scheduling code's fixture and must not be described as real warehouse data.

| ID | Observed pick start | Observed pick finish | Observed pack start | Observed pack finish |
|---|---:|---:|---:|---:|
| A | 0 | 3 | 3 | 5 |
| B | 3 | 5 | 5 | 8 |
| C | 5 | 9 | 9 | 10 |
| D | 9 | 10 | 10 | 14 |
| E | 10 | 12 | 15 | 17 |

Observed cycle times are 5, 7, 8, 11, and 13 minutes.
Their mean is `44 / 5 = 8.8` minutes.
The simulated-minus-observed mean difference is `8.6 - 8.8 = -0.2` minutes.
The simulated final completion is one minute earlier than the invented observed final completion.

Observed pack waits are 0, 0, 0, 0, and 3 minutes.
Their mean is `3 / 5 = 0.6` minutes, versus `2 / 5 = 0.4` simulated minutes.
The one-minute idle gap before E is a discrepancy requiring investigation.
It could reflect a break, handoff, missing scan, or another unmodeled cause; the table cannot distinguish these explanations.
A useful diagnostic is the per-order residual, not just the small mean error.

## 7. Code, query, or calculation

### Run the lab

Use Python 3.9 or later and only the standard library.
Create a working directory outside the course source if desired.
Save the following complete block as `warehouse_des.py` in that directory.
Run `python warehouse_des.py`; on systems using `python3`, run `python3 warehouse_des.py` instead.
The code writes no external files and needs no network, credentials, packages, or production data.

```python
from heapq import heapify, heappop, heappush
from math import sqrt
from random import Random
from statistics import mean, stdev

# All times are integer minutes from time zero. IDs resolve eligibility ties.
FIXTURE = [
    {"id": "A", "release": 0, "pick": 3, "pack": 2},
    {"id": "B", "release": 1, "pick": 2, "pack": 3},
    {"id": "C", "release": 2, "pick": 4, "pack": 1},
    {"id": "D", "release": 3, "pick": 1, "pack": 4},
    {"id": "E", "release": 4, "pick": 2, "pack": 2},
]

# Invented observations; these are not a real warehouse extract.
OBSERVED = {
    "A": (0, 3, 3, 5),
    "B": (3, 5, 5, 8),
    "C": (5, 9, 9, 10),
    "D": (9, 10, 10, 14),
    "E": (10, 12, 15, 17),
}


def station(jobs, capacity):
    """jobs are (eligible time, ID, service minutes); return ID -> interval."""
    if capacity < 1:
        raise ValueError("capacity must be at least one")
    available = [0] * capacity
    heapify(available)
    intervals = {}
    for eligible, order_id, duration in sorted(jobs):
        free = heappop(available)
        start = max(eligible, free)
        finish = start + duration
        intervals[order_id] = (start, finish)
        heappush(available, finish)
    return intervals


def simulate(orders, pickers=1, packers=1):
    """Run a finite, empty-start, feed-forward FIFO line to completion."""
    if not orders or len({o["id"] for o in orders}) != len(orders):
        raise ValueError("orders must be nonempty with unique IDs")
    if any(o["release"] < 0 or o["pick"] <= 0 or o["pack"] <= 0
           for o in orders):
        raise ValueError("release must be nonnegative and service positive")
    picked = station([(o["release"], o["id"], o["pick"]) for o in orders],
                     pickers)
    packed = station([(picked[o["id"]][1], o["id"], o["pack"]) for o in orders],
                     packers)
    trace = []
    for o in sorted(orders, key=lambda row: row["id"]):
        ps, pf = picked[o["id"]]
        ks, kf = packed[o["id"]]
        trace.append({"id": o["id"], "release": o["release"],
                      "pick_start": ps, "pick_finish": pf,
                      "pack_start": ks, "pack_finish": kf,
                      "pick_wait": ps - o["release"], "pack_wait": ks - pf,
                      "cycle": kf - o["release"]})
    horizon = max(row["pack_finish"] for row in trace)
    metrics = {
        "makespan": horizon,
        "pick_wait": sum(row["pick_wait"] for row in trace),
        "pack_wait": sum(row["pack_wait"] for row in trace),
        "mean_cycle": mean(row["cycle"] for row in trace),
        "pick_util": sum(o["pick"] for o in orders) / (pickers * horizon),
        "pack_util": sum(o["pack"] for o in orders) / (packers * horizon),
    }
    return trace, metrics


def synthetic_orders(seed, count=30):
    """Generate a workload once per seed, then reuse it across scenarios."""
    rng = Random(seed)
    release = 0
    orders = []
    for i in range(count):
        release += rng.randint(0, 3)
        orders.append({"id": f"{i:03d}", "release": release,
                       "pick": rng.randint(1, 5), "pack": rng.randint(1, 4)})
    return orders


def paired_differences(pickers, packers, seeds):
    differences = []
    for seed in seeds:
        orders = synthetic_orders(seed)
        baseline = simulate(orders, 1, 1)[1]["makespan"]
        candidate = simulate(orders, pickers, packers)[1]["makespan"]
        differences.append(candidate - baseline)
    return mean(differences), stdev(differences) / sqrt(len(differences))


def main():
    trace, base = simulate(FIXTURE)
    expected = [(0, 3, 3, 5), (3, 5, 5, 8), (5, 9, 9, 10),
                (9, 10, 10, 14), (10, 12, 14, 16)]
    actual = [(r["pick_start"], r["pick_finish"], r["pack_start"],
               r["pack_finish"]) for r in trace]
    assert actual == expected
    assert (base["makespan"], base["pick_wait"], base["pack_wait"]) == (16, 17, 2)
    assert len(trace) == len(FIXTURE)
    assert all(r["release"] <= r["pick_start"] <= r["pick_finish"]
               <= r["pack_start"] <= r["pack_finish"] for r in trace)
    assert station([(0, "B", 1), (0, "A", 1)], 1) == {
        "A": (0, 1), "B": (1, 2)}
    print("id release pick_start pick_finish pack_start pack_finish pick_wait pack_wait cycle")
    for r in trace:
        print(*(r[k] for k in ("id", "release", "pick_start", "pick_finish",
                              "pack_start", "pack_finish", "pick_wait", "pack_wait", "cycle")))
    for name, p, k, expected_values in [
        ("baseline", 1, 1, (16, 17, 2)),
        ("extra_picker", 2, 1, (15, 1, 19)),
        ("extra_packer", 1, 2, (14, 17, 0)),
    ]:
        _, m = simulate(FIXTURE, p, k)
        values = (m["makespan"], m["pick_wait"], m["pack_wait"])
        assert values == expected_values
        print(f"{name}: makespan={values[0]} pick_wait={values[1]} pack_wait={values[2]}")
    print(f"baseline mean_cycle={base['mean_cycle']:.1f} "
          f"pick_util={base['pick_util']:.2f} pack_util={base['pack_util']:.2f}")
    observed_cycle = mean(OBSERVED[o["id"]][3] - o["release"] for o in FIXTURE)
    observed_pack_wait = sum(OBSERVED[o["id"]][2] - OBSERVED[o["id"]][1]
                             for o in FIXTURE)
    print(f"observed mean_cycle={observed_cycle:.1f} pack_wait={observed_pack_wait}")
    print(f"simulated_minus_observed mean_cycle={base['mean_cycle'] - observed_cycle:.1f}")
    for name, p, k in [("extra_picker", 2, 1), ("extra_packer", 1, 2)]:
        avg, se = paired_differences(p, k, range(100, 120))
        print(f"{name} paired makespan difference={avg:.2f} min; SE={se:.2f} min")
    print("checks passed")


if __name__ == "__main__":
    main()
```

The fixed portion of the expected output is:

```text
id release pick_start pick_finish pack_start pack_finish pick_wait pack_wait cycle
A 0 0 3 3 5 0 0 5
B 1 3 5 5 8 2 0 7
C 2 5 9 9 10 3 0 8
D 3 9 10 10 14 6 0 11
E 4 10 12 14 16 6 2 12
baseline: makespan=16 pick_wait=17 pack_wait=2
extra_picker: makespan=15 pick_wait=1 pack_wait=19
extra_packer: makespan=14 pick_wait=17 pack_wait=0
baseline mean_cycle=8.6 pick_util=0.75 pack_util=0.75
observed mean_cycle=8.8 pack_wait=3
simulated_minus_observed mean_cycle=-0.2
```

Two additional lines report paired stochastic differences and their standard errors in minutes.
The final line is `checks passed` if the assertions succeed.
The exact stochastic values depend on the documented Python random implementation; record the Python version when sharing results.
The fixed trace does not depend on pseudorandom draws.

### Tests and interpretation

The assertions check the hand-calculated baseline, a simultaneous-eligibility tie, conservation of trace rows, timestamp order, and the three fixed scenario outcomes.
Run the command twice in the same Python environment to check deterministic reproducibility.
Compare the complete outputs, including the seeded stochastic lines.
If an assertion fails, inspect the fixture, the event order, and the Python file before changing an expected value.

The example does not automatically prove that capacity never overlaps for every possible input.
Exercise 7 asks you to add a general overlap check and a simultaneous completion case.
The code does not calculate a confidence interval, optimize staffing, or fit distributions to warehouse observations.
The paired standard error is not validation error and is not evidence of a real-world improvement.

### Optional fixture adaptation

To use a sanitized extract, transform each eligible order into the four input fields in the contract.
Keep the historical event trace separately so model-generated timestamps cannot accidentally replace observed timestamps.
Do not include names, customer addresses, order contents, or other unnecessary sensitive information.
Ensure all records use the same time origin and time unit before passing them to `simulate`.

Historical service durations may be unavailable for unfinished orders or censored by scan practices.
Do not silently drop these records or fill their missing service values with zero.
Record how they were handled and test whether the resulting cohort still represents the intended workload.
Before operational use, replace the simplified two-pass model if real process feedback violates its assumptions.

## 8. Data, standards, and regulatory context

### Data provenance and timestamp semantics

A credible simulation input starts with a data dictionary, not a histogram.
For each timestamp, identify the producing system, clock, time zone, precision, and event definition.
A warehouse management system may record an order release, a handheld scan, or a task confirmation at different points in the physical process.
Those points are not automatically interchangeable.

An order creation timestamp can precede warehouse release by hours.
A first scan can occur after a worker has already traveled to a location.
A label timestamp can precede the actual end of packing.
A change in mobile connectivity can cause delayed upload without a delayed physical event.
Investigate these mechanisms before treating a time difference as service or wait.

Use one stable entity key across release, pick, and pack events.
Preserve a mapping from task IDs to order IDs when a task can contain several orders.
Record whether a split order is represented as one parent, multiple children, or a final consolidated entity.
A join that duplicates an order can inflate throughput and busy time simultaneously.

Convert wall-clock timestamps to a common instant before subtracting them.
Store the original time zone and daylight-saving interpretation in the data contract.
Ambiguous or nonexistent local times require an explicit resolution rule.
A negative interval is a data-quality signal, not a value to clip to zero without investigation.

### Cohort and observation design

Define whether the cohort is orders released during a window or orders completed during a window.
These selections answer different questions.
For a release cohort, follow orders after the window until they finish or explicitly mark them as censored.
For a completion cohort, account for work released before the window.
A model that starts empty cannot reproduce a shift that begins with an unmeasured backlog.

Record downtime, shift breaks, skill restrictions, and changes in station capacity with their start and end times.
If availability records do not exist, say that capacity is an assumption rather than a measured input.
Stratify input observations by work type, order size, zone, and time of day when these affect service.
Avoid fitting a single pooled distribution when the mixture is changing across the comparison periods.

Keep training, calibration, and validation periods distinct.
Select a holdout that includes realistic peaks and different operational days where possible.
If you adjust a parameter after seeing holdout errors, the holdout has become development data.
Reserve a new untouched period for the next independent assessment.

Document sampling, excluded rows, missingness, and any repairs.
Store an immutable data snapshot or a content hash where organizational policy permits.
Record the extraction query, schema version, model version, seed list, and scenario settings.
These artifacts make a result reproducible without exposing personal information in the teaching repository.

### Identifiers and standards

GS1 publishes standards for identifying trade items, locations, logistics units, and event information.
The GS1 Electronic Product Code Information Services (EPCIS) standard describes a way to share visibility event data.
An EPCIS event can provide useful business context, but it does not by itself define the start and end of this particular pick-line queue.
Map local events to a documented process definition before computing waiting or service time.

A barcode, a location identifier, or an event standard does not guarantee complete or correctly ordered scans.
The unit of analysis still needs a stable order or task mapping.
Use the applicable GS1 specification and local implementation agreement rather than inventing identifier semantics.
Adoption of a standard in this educational example is not claimed to be legally required.

### Privacy, labor, and safety

Operational timestamps can be linked to identifiable workers even when the order data contains no customer name.
Restrict access, minimize fields, and follow the organization's retention and privacy rules.
Applicable legal duties vary by jurisdiction and use; this lab is not legal advice.
Do not publish worker-level traces in a public repository.

A capacity simulation does not authorize a change in safe workload, ergonomic practice, staffing agreement, or required training.
Review applicable local safety and labor requirements with qualified owners before a field trial.
The model omits fatigue, injury risk, travel congestion, and quality effects.
These omissions are material if a recommendation changes work intensity.

## 9. Industry practice and case evidence

### What published literature supports

Warehouse design and order picking have been studied extensively in operations research.
De Koster, Le-Duc, and Roodbergen review order-picking design and control, including routing, batching, zoning, and storage decisions.
Those choices affect travel and workload, which this small two-station model intentionally aggregates into service time.
The review supports the importance of design choices; it does not validate the numerical fixture in this lab.

Banks and colleagues describe discrete-event simulation as a method for modeling systems whose states change at discrete points in time.
The present heap scheduler is a restricted feed-forward implementation of that event logic.
A full simulation package may be more appropriate when a system has blocking, resource calendars, or interacting decisions.
Using a package does not remove the need to define events and validate data.

Sargent distinguishes model verification and validation and discusses their application throughout model development.
That distinction is central here: passing the fixture checks says the code follows a hand-specified example.
Matching independent operating observations is a separate, context-dependent question.
No validation status for any real facility is claimed.

### A documented public case and its limits

Amazon Robotics describes mobile robotic fulfillment operations in public company materials.
Those descriptions document an operating approach in which mobile robots bring inventory to employees at workstations.
They are company-authored descriptions, not independent estimates of the staffing or throughput effect of the fixture in this lesson.
The actual systems involve routing, inventory placement, workstations, and constraints absent from this model.

[INFERENCE] A facility using mobile inventory movement may experience a different balance between travel, picking, and station waiting from the manual pick-line abstraction here.
Testing that inference would require site-specific event definitions, worker assignments, and independent operational data.
The public description does not establish the size or direction of a benefit for any proposed change.
Do not transfer a company case's process or performance claim into this synthetic example.

### A practical evidence ladder

Start with process observation and a documented boundary.
Check that the data capture the events the process owner says matter.
Verify the model against deliberately small, hand-worked cases.
Compare a locked baseline against independent operating periods.
Test whether intervention rankings remain stable across plausible demand and service assumptions.
Only then consider a controlled operational trial with a safety and rollback plan.

A single fitted average is weak evidence of operational fidelity.
A trace that matches the order of events is stronger evidence about mechanics but still may not represent peak periods.
Independent shifts with similar delay distributions support a narrower claim about the tested conditions.
A randomized or otherwise credible field comparison can support a claim about actual intervention effects, subject to its design and scope.

Model users should record evidence against each important assumption.
For example, a station may truly be FIFO in ordinary periods but override priority for cutoff orders.
A single station's utilization may look low because another station starves it during part of the day.
A queue may appear short in snapshots because orders wait in an upstream staging area that was excluded.
These alternatives require observation and event data, not more decimal places in the simulation output.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong release event | Simulated wait begins long before warehouse work. | Order creation is substituted for task release. | Queue and cycle time are overstated. | Map source event to operational eligibility and inspect samples. |
| Completion event mismatch | Model is consistently early versus shipment scan. | Packing finish is compared with carrier handover. | A valid model appears wrong or a wrong one appears valid. | Align boundary and endpoint in both data sets. |
| Missing opening backlog | Observed early-shift queues start above zero. | Empty-start assumption discards prior work. | Early delay and throughput are biased. | Seed initial work in process or limit the claim to empty starts. |
| Incomplete drain | Many released orders lack completion. | Observation stops at shift end. | Fast orders dominate the measured cycle time. | Follow the cohort or explicitly analyze censoring. |
| Duplicate order rows | More completions than releases after a join. | One-to-many task scans multiply entities. | Throughput and service are overstated. | Define entity keys and test join cardinality. |
| Swapped time zones | Negative waits cluster around clock changes. | Local timestamps are treated as comparable instants. | Impossible event sequences and distorted distributions. | Normalize instants and document daylight-saving rules. |
| Incorrect tie rule | Results vary after input row reorder. | Equal-time events follow incidental iteration order. | Nonreproducible queue assignment. | Declare and test completion and ID tie ordering. |
| Omitted breaks | Regular gaps appear in observed service. | Constant capacity ignores worker unavailability. | Simulated completion is too early. | Add observed calendars or quantify the limitation. |
| Service includes wait | Fitted service exceeds hands-on observation. | Release-to-finish interval is used as service. | Queue delay is counted twice. | Separate start, finish, and eligibility events. |
| Wrong FIFO population | Pack order follows releases rather than pick finishes. | Upstream order is assumed to persist downstream. | Pack waits are assigned to the wrong orders. | Sort by pack eligibility and test overtaking. |
| Finite buffer omitted | Pickers stop when handoff area is full. | Downstream congestion cannot block upstream resources. | Capacity benefits are overstated. | Implement coupled blocking events and validate buffer size. |
| Shared worker counted twice | Both stations are busy with one cross-trained worker. | Independent resource pools model the same person. | Impossible capacity and understated delay. | Model the shared worker and assignment policy explicitly. |
| Pooled workload shift | Error grows for large orders or peak hours. | One service distribution hides mixtures. | Scenarios are compared under the wrong mix. | Segment by workload drivers and validate each segment. |
| Random streams mispaired | Scenario differences fluctuate despite fixed demand. | Conditional draws desynchronize a shared seed. | Extra noise or misleading comparison. | Pre-generate common exogenous workloads. |
| Overfitted validation | Historical mean matches but later shifts miss. | Parameters were tuned on evaluation records. | False confidence and poor generalization. | Lock calibration and reserve a fresh holdout. |
| Utilization-only bottleneck | High utilization does not predict intervention impact. | Busy time ignores starvation, blocking, and sequence. | Resources are added at the wrong station. | Combine queues, controlled perturbations, and process evidence. |
| Average-only report | Mean error is small but late orders are missed. | Positive and negative errors cancel. | Service failures remain hidden. | Compare tails, per-order residuals, and time-of-day profiles. |
| Unrealistic parallelism | Two workers occupy a one-person workstation. | Capacity is increased without spatial constraints. | Simulated gains are physically infeasible. | Validate station layout, equipment, and skill requirements. |
| Uncontrolled field change | Trial coincides with a promotion or new routing rule. | Multiple causes change simultaneously. | Improvement cannot be attributed to staffing. | Use a comparison design and log concurrent changes. |
| Privacy leakage | Shared traces include identifiable worker IDs. | Operational records are copied without minimization. | Unauthorized disclosure or policy breach. | Use synthetic fixtures, access controls, and approved aggregation. |

A failure table is not a substitute for direct observation.
A red flag tells you where to investigate; it does not prove which mechanism occurred.
Document both the evidence and the competing explanations before modifying the model.

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or bound it |
|---|---|---|---|
| A modeled entity is one order. | Keeps each trace and metric interpretable. | Parallel line picking, splits, or consolidation. | Compare order and task counts; observe split and join steps. |
| All work begins in an empty system. | Defines the finite fixture's initial state. | A shift begins with unfinished work. | Measure opening backlog and rerun with an appropriate initial state. |
| All released work drains. | Gives complete cycle times and a fixed makespan. | A real observation window ends before orders finish. | Follow release cohorts or report censored records. |
| Station capacities are constant. | Makes worker availability representable by a heap. | Breaks, absences, or shift changes. | Compare against roster and downtime event logs. |
| Workers within a pool are interchangeable. | Avoids worker-specific assignment state. | Skills, speed, equipment, or zones differ. | Stratify service and inspect skill-dependent task routing. |
| Pickers and packers are distinct. | Allows independent station scheduling. | Cross-trained staff serve both queues. | Reconcile worker IDs and build a shared-resource model if needed. |
| Pick and pack services are nonpreemptive. | Each order occupies one continuous interval. | Interruptions, priority overrides, or task switching. | Observe interrupted tasks and encode pause/resume when material. |
| Queues are FIFO by eligibility. | Defines an unambiguous scheduling discipline. | Cutoffs, expedited orders, batching, or supervisor overrides. | Compare observed start order with eligibility order. |
| The intermediate buffer is unlimited. | Prevents downstream state from blocking pickers. | Physical handoff space fills. | Observe capacity and blocking; use a coupled event calendar. |
| Service times are known exogenous inputs. | Enables exact paired scenario comparison. | Congestion, learning, or staffing changes alter durations. | Measure duration under different workloads and test sensitivity. |
| Pick completion immediately enables pack eligibility. | Omits handoff and transport events. | Travel, sorting, quality checks, or scan delays intervene. | Measure handoff delay and model it explicitly if material. |
| No rework, cancellation, or failures occur. | Makes entity conservation simple. | Defects, canceled orders, and equipment breakdowns occur. | Count and characterize exceptions; add states when decision-relevant. |
| Order arrivals are synthetic and stationary in the random exercise. | Demonstrates reproducible paired replications. | Actual work has waves, seasonality, or correlated bursts. | Compare interarrival patterns and run historical replay. |
| Time begins at zero in integer minutes. | Keeps the teaching arithmetic simple. | Timestamp precision or shift windows require finer resolution. | Repeat with seconds and actual time-zone-aware instants. |
| Resource exposure lasts through final completion. | Defines the utilization denominator in this batch. | Workers are rostered for a fixed shift or leave early. | Use scheduled availability intervals for the intended KPI. |
| A single small holdout illustrates comparison. | Makes discrepancies visible by hand. | Real-world variability and sampling uncertainty dominate. | Validate against multiple independent periods and peak conditions. |

The assumptions are not claims that a warehouse actually operates this way.
Each assumption either limits the decision or identifies a concrete model extension.
Do not interpret a narrow deterministic outcome as a robust prediction across omitted conditions.

## 12. Exercises

### Exercise 1 — Draw and label the flow

**Time:** 10 minutes.
**Deliverable:** a diagram with the two queues, two resource pools, and four event types.
Identify the order as the entity and state the release and completion boundaries.

**Acceptance criteria:** the diagram makes clear that pick completion makes an order eligible for packing.
It shows the picker being released before the next pick starts at the same time.
It names the FIFO tie rule and notes that the two resource pools are distinct.

### Exercise 2 — Recompute the baseline

**Time:** 20 minutes.
**Deliverable:** a five-row trace with release, pick start and finish, pack start and finish, waits, and cycle time.
Calculate mean cycle time and both station utilizations using the full-drain horizon.

**Acceptance criteria:** every service interval matches the fixture's duration.
The trace contains five completed orders and no overlapping service on either single worker.
Your total wait and utilization denominators match the definitions in section 5.

### Exercise 3 — Run paired experiments

**Time:** 20 minutes.
**Deliverable:** the script output from two runs in the same Python environment and a short provenance record.
Include Python version, seed range, replication count, and the exact scenario capacities.
Explain why the synthetic order list is created once per seed rather than separately per scenario.

**Acceptance criteria:** both runs reproduce their fixed trace and paired results in that environment.
The report labels the stochastic values as invented-workload results, not a production forecast.
It distinguishes paired standard error from validation error.

### Exercise 4 — Change the utilization window

**Time:** 15 minutes.
**Deliverable:** picker and packer utilization for the baseline using a scheduled 20-minute horizon.
Assume each worker remains scheduled from minute 0 through minute 20 and no further work arrives.
Compare those values with the 16-minute full-drain utilization.

**Acceptance criteria:** the numerator is 12 worker-minutes for each station.
The denominator is 20 worker-minutes for each station under the new assumption.
Explain why neither denominator is universally correct without specifying the KPI.

### Exercise 5 — Diagnose the constraint

**Time:** 20 minutes.
**Deliverable:** a paragraph comparing the baseline, extra-picker, and extra-packer cases.
Use both wait totals and makespan changes.
Recommend the next evidence-gathering step, not a production staffing action.

**Acceptance criteria:** identify that extra picking moves waiting to packing.
Recognize that extra packing improves this batch's makespan more than extra picking.
Explain why equal baseline utilization and five synthetic orders do not establish a universal bottleneck.

### Exercise 6 — Validate a held-out trace

**Time:** 20 minutes.
**Deliverable:** a comparison of simulated and observed mean cycle time and total pack wait.
Identify the order that contributes the discrepancy and propose two possible explanations.
List one additional observation that would discriminate between those explanations.

**Acceptance criteria:** use the same five release-cohort orders in both calculations.
Report the signed simulated-minus-observed cycle difference.
Label the observed table as synthetic and the explanations as hypotheses.

### Exercise 7 — Extend verification

**Time:** 30 minutes.
**Deliverable:** a test plan and code for a general station-capacity overlap check.
Add tests for a completion and new eligibility at the same time, simultaneous completions, reversed input order, and two workers.

**Acceptance criteria:** the overlap check treats finish as occurring before start at the same timestamp.
It detects any interval set with more overlapping jobs than workers.
The tests include hand-specified expected intervals and do not use the scheduler itself to manufacture expected values.

### Exercise 8 — Design a real-data validation and field trial

**Time:** 30 minutes.
**Deliverable:** a one-page decision memo and a data dictionary for a proposed warehouse study.
Specify calibration and independent holdout periods, cohort definition, initial work in process, relevant queue metrics, and missing-data handling.
Propose a field comparison, safety review, stop condition, rollback owner, and an acceptance gate fixed before the trial.

**Acceptance criteria:** the memo separates model verification, historical validation, and field evidence.
It states what data are unavailable and which assumptions remain untested.
It does not promise an operational improvement or expose worker-level personal data.

## 13. Self-check answers

### Answer 1 — Flow and state

The entity is an order.
It enters the pick queue at release and requires one picker during its pick service interval.
It enters the pack queue at pick completion and requires one packer during its pack service interval.
Its pack completion ends the modeled journey.

A resource is released at the same timestamp as its prior service finishes.
The next eligible order may start at that time, so intervals are half-open.
Orders tied on eligibility are ordered by ID.
A strong diagram also marks that picker and packer pools are separate and the handoff buffer is unlimited.

### Answer 2 — Trace and metric arithmetic

The baseline pick intervals are A `[0,3)`, B `[3,5)`, C `[5,9)`, D `[9,10)`, and E `[10,12)`.
The baseline pack intervals are A `[3,5)`, B `[5,8)`, C `[9,10)`, D `[10,14)`, and E `[14,16)`.
Pick waits are 0, 2, 3, 6, and 6 minutes.
Pack waits are 0, 0, 0, 0, and 2 minutes.

The five cycle times are 5, 7, 8, 11, and 12 minutes.
Their mean is 8.6 minutes per order.
Each worker is busy for 12 minutes over the 16-minute full-drain interval.
Each pool's utilization is 75% for that specified interval.

### Answer 3 — Reproducibility and pairing

The script generates one list of 30 synthetic orders for each seed.
It then passes that same list to baseline and candidate runs.
The paired difference is candidate makespan minus baseline makespan for that seed.
The mean and standard error summarize the 20 paired differences, not 600 independent order observations.

The fixed seed range makes the pseudo-random experiment repeatable in the same documented environment.
It does not make the generated workload representative of any actual site.
A sound provenance record includes the Python version and a saved copy or hash of the script as well as the seeds and capacities.
The standard error measures variability due to the synthetic replications, not systematic model error.

### Answer 4 — Alternative utilization denominator

The numerator remains 12 worker-minutes for the picker and 12 worker-minutes for the packer.
The scheduled capacity is `1 worker * 20 minutes = 20 worker-minutes` at each station.
Each utilization is `12 / 20 = 0.60`, or 60%.
The earlier full-drain denominator gives `12 / 16 = 0.75`, or 75%.

The difference is entirely due to exposure definition.
If the KPI is busy time during a 20-minute paid schedule, the 20-minute denominator fits the stated assumption.
If the KPI is utilization only until the finite batch drains, the 16-minute denominator fits.
A report must name the window rather than present either percentage without context.

### Answer 5 — Capacity interpretation

The baseline has 17 order-minutes of pick wait and 2 order-minutes of pack wait.
An extra picker changes those totals to 1 and 19 order-minutes, respectively.
Its makespan falls from 16 to 15 minutes.
An extra packer leaves pick wait at 17 order-minutes, removes pack wait, and reduces makespan to 14 minutes.

The larger baseline pick wait points to upstream delay, while the larger makespan reduction from extra packing shows the end-of-batch objective can respond differently.
The equal baseline utilization values cannot settle this question.
Inspect real workload, handoff capacity, worker schedules, and queue traces before designing a field test.
Any statement about a real bottleneck is unsupported by these five invented orders.

### Answer 6 — Holdout discrepancy

Simulated mean cycle time is `43 / 5 = 8.6` minutes.
Synthetic observed mean cycle time is `44 / 5 = 8.8` minutes.
The signed simulated-minus-observed difference is `-0.2` minutes.
Simulated pack waiting totals 2 order-minutes; synthetic observed pack waiting totals 3 order-minutes.

Order E accounts for the one-minute difference: observed packing starts at minute 15 rather than simulated minute 14.
A break or a delayed handoff are hypotheses, not facts established by the table.
A station availability log, direct observation, or handoff event timestamp could help distinguish them.
Five invented records are a mechanics check, not independent evidence of site-level fit.

### Answer 7 — Verification rubric

A correct overlap checker sorts interval boundaries and handles finishes before starts at equal timestamps.
Alternatively, it can count active intervals on each half-open segment between unique event times.
It must fail a fixture with two overlapping intervals for one worker and pass two adjacent intervals for one worker.
It must pass two overlapping intervals for two workers and fail three overlapping intervals for two workers.

A simultaneous-completion test should specify the eligible orders, capacities, service durations, and expected starts by hand.
Reversing input records should preserve the trace because the scheduler sorts by eligibility and ID.
A tie test should establish that ID, rather than insertion order, determines the next job.
The reviewer should reject a test whose only oracle is a second call to the same implementation.

### Answer 8 — Design rubric

A strong memo defines a release cohort, exact event semantics, observation window, and treatment of opening and unfinished work.
It separates input calibration from a locked, chronologically later validation period.
It compares means, tails, queue patterns, and throughput, with a stated tolerance for the intended decision.
It records plausible competing causes of discrepancies and does not silently retune on the holdout.

A credible field plan identifies the intervention, comparison condition, period, owner, and operational safety controls.
It states how demand mix and simultaneous process changes will be measured or controlled.
It names a stop condition and the person authorized to restore the previous configuration.
It describes a successful model result as a reason to consider a trial, not proof of realized savings.

## 14. Interview questions

### 1. What makes this a discrete-event model?

A strong answer identifies state changes at releases and service completions and explains that time jumps between meaningful events.
It notes that the heap implementation is a special-case scheduler for a feed-forward system.
A weak answer says that any program with a loop and a clock is necessarily a correct event simulation.

### 2. Why are the two stations scheduled in separate passes?

A strong answer explains that downstream state cannot block or change upstream behavior under the stated assumptions.
It says that pack jobs must be sorted by pick completion, not original release.
A weak answer assumes this decomposition remains correct with finite buffers or shared workers.

### 3. How do you handle simultaneous events?

A strong answer states that capacity released at `t` is available at `t` and intervals are half-open.
It supplies a deterministic eligibility tie rule and a test for it.
A weak answer relies on incidental input order or leaves event priority undefined.

### 4. Why is average utilization insufficient to identify a bottleneck?

A strong answer discusses starvation, blocking, queue patterns, objective choice, and intervention effects.
It observes that both baseline stations are 75% utilized yet the two capacity interventions affect makespan differently.
A weak answer selects the busiest station without specifying workload or outcome.

### 5. How would you compare staffing scenarios fairly?

A strong answer holds exogenous workloads constant within each replication and reports paired differences across independent seeds.
It distinguishes Monte Carlo uncertainty from uncertainty in model inputs and structural assumptions.
A weak answer compares unrelated single runs or assumes a shared initial seed always guarantees common random numbers.

### 6. What would invalidate the baseline comparison with historical data?

A strong answer mentions mismatched boundaries, initial work in process, censored orders, calendar differences, and data leakage.
It asks for a held-out cohort and per-order or distributional checks.
A weak answer declares the model valid solely because its average cycle time is close.

### 7. How do you calculate station utilization with multiple workers?

A strong answer divides busy worker-minutes by available worker-minutes over a declared observation interval.
It clips intervals to the window and accounts for actual availability when the KPI requires it.
A weak answer divides total service time by elapsed time without multiplying by worker capacity.

### 8. What if some orders are not complete when data collection stops?

A strong answer identifies right censoring and explains why excluding unfinished orders biases completion metrics.
It proposes following the release cohort or using an appropriate censored-data method.
A weak answer sets missing cycle times to zero or removes unfinished orders without disclosure.

### 9. When does the heap scheduler need replacement?

A strong answer names downstream blocking, shared staff, finite buffers, routing feedback, or dynamic priority as interacting state.
It proposes a coupled event calendar with explicit event precedence and regression tests.
A weak answer adds a second pass while assuming that earlier upstream decisions can never need revision.

### 10. What makes a simulation-based operational recommendation credible?

A strong answer combines a decision-specific boundary, verified mechanics, independent validation, sensitivity analysis, and an approved field trial.
It includes worker safety, feasibility, measurement, and rollback.
A weak answer treats a synthetic model output as a measured production saving.

## 15. Key terms

- **Availability:** time a resource can accept work; see [mechanics](#5-the-mechanics-step-by-step).
- **Blocking:** upstream work prevented from leaving by downstream constraints; see [assumptions](#11-assumptions-and-limits).
- **Bottleneck:** constraint limiting an outcome under a specified workload; see [capacity scenarios](#capacity-scenarios).
- **Busy time:** occupied worker-minutes within a defined interval; see [metrics](#step-8--define-metrics-before-running-scenarios).
- **Calibration:** estimation of inputs from designated data; see [data context](#8-data-standards-and-regulatory-context).
- **Capacity:** number of simultaneously available workers in a pool; see [resources](#step-3--define-state-and-resources).
- **Censoring:** incomplete observation of an order's eventual outcome; see [cohort design](#cohort-and-observation-design).
- **Common random numbers:** shared exogenous workloads for paired scenarios; see [replications](#step-11--generate-paired-stochastic-replications).
- **Conservation:** equality of released orders and completed plus remaining orders; see [verification](#step-9--verify-the-fixed-trace).
- **Cycle time:** release-to-pack-finish elapsed minutes; see [worked example](#6-worked-example).
- **Discrete-event simulation:** representation of state changes at event times; see [mental model](#3-the-core-mental-model).
- **Entity:** modeled unit that flows through the system; see [vocabulary](#4-definitions-and-vocabulary).
- **Event calendar:** ordered set of future state transitions; see [event semantics](#step-4--specify-event-semantics).
- **FIFO:** first-in-first-out queue discipline; see [resources](#step-3--define-state-and-resources).
- **Fixture:** explicit input with known expected behavior; see [code](#7-code-query-or-calculation).
- **Holdout:** data withheld from calibration for evaluation; see [validation](#step-12--compare-against-held-out-observations).
- **Makespan:** elapsed time to finish the finite batch; see [reconciled trace](#reconciled-trace).
- **Monte Carlo error:** sampling variability across random replications; see [replications](#step-11--generate-paired-stochastic-replications).
- **Pack eligibility:** time picking finishes and an order may enter packing; see [pack scheduling](#step-7--schedule-the-pack-stage).
- **Paired difference:** candidate result minus baseline on the same input stream; see [replications](#step-11--generate-paired-stochastic-replications).
- **Queue wait:** service start minus eligibility time; see [metrics](#step-8--define-metrics-before-running-scenarios).
- **Release:** time an order becomes eligible for the first station; see [input contract](#step-5--construct-an-input-contract).
- **Replication:** one seeded synthetic workload and scenario evaluation; see [code](#7-code-query-or-calculation).
- **Resource:** reusable capacity such as a picker or packer; see [mental model](#3-the-core-mental-model).
- **Service duration:** time a resource is occupied by an order; see [input contract](#step-5--construct-an-input-contract).
- **Starvation:** lack of available upstream work for a resource; see [failure modes](#10-failure-modes-and-red-flags).
- **Throughput:** completed orders divided by a stated elapsed interval; see [reconciled trace](#reconciled-trace).
- **Trace:** per-order event record and derived quantities; see [worked example](#6-worked-example).
- **Utilization:** busy resource-time divided by available resource-time; see [metrics](#step-8--define-metrics-before-running-scenarios).
- **Validation:** evaluation of model fitness for a stated use against independent evidence; see [holdout comparison](#synthetic-holdout-comparison).
- **Verification:** checking implementation against its specification; see [tests](#tests-and-interpretation).
- **Work in process:** released but unfinished orders; see [boundary](#step-2--draw-the-system-boundary).

## 16. Sources

1. René de Koster, Tho Le-Duc, and Kees Jan Roodbergen, “Design and control of warehouse order picking: A literature review,” *European Journal of Operational Research* 182(2), 2007, 481–501. [DOI: 10.1016/j.ejor.2006.07.009](https://doi.org/10.1016/j.ejor.2006.07.009). Warehouse design and order-picking research context; not evidence for this fixture's numbers.
2. Robert G. Sargent, “Verification and validation of simulation models,” *Journal of Simulation* 7, 2013, 12–24. [DOI: 10.1057/jos.2012.20](https://doi.org/10.1057/jos.2012.20). Model verification and validation concepts.
3. Jerry Banks, John S. Carson II, Barry L. Nelson, and David M. Nicol, *Discrete-Event System Simulation*, 5th edition, Pearson, 2010. Standard DES concepts, event scheduling, and experimental design.
4. [Python `heapq` documentation](https://docs.python.org/3/library/heapq.html). Min-heap behavior used by the station scheduler.
5. [Python `random` documentation](https://docs.python.org/3/library/random.html). Pseudorandom generator, reproducibility notes, and seed behavior.
6. [Python `statistics` documentation](https://docs.python.org/3/library/statistics.html). Sample standard deviation used for paired standard errors.
7. [GS1 EPCIS standard](https://www.gs1.org/standards/epcis). Visibility-event standard and implementation resources; not a mandate to use it in this lab.
8. [Amazon Robotics](https://www.aboutamazon.com/what-we-do/amazon-robotics). Company-authored description of robotic fulfillment operations; not independent validation of performance here.

The fixture and its observed holdout are instructional inventions.
No source above reports their numeric values or verifies this script against an operating site.
The choice of an applicable law, safety standard, or employee-data policy requires jurisdiction- and organization-specific review.

## 17. Links

- [Module 30 topic index](_index.md).
- [Module 30 survey and lab overview](../../../course/30_simulation_optimization_production_labs.md).
- [Environment and repository lab](01_environment-repo.md) for Python setup and reproducibility practices.
- [Inventory policy simulation lab](03_inventory-policy-simulation.md) for scenario comparisons using common demand paths.
- [Demand forecast pipeline lab](02_demand-forecast-pipeline.md) for input-data and backtesting discipline.
- [Warehouse flow system](../../02_methods/13_warehousing_fulfillment_labor/01_warehouse-flow-system.md) for the wider physical process.

The code and fixture in section 7 constitute the self-contained lab artifact.
Sibling availability and course sequencing are described in the module index.
