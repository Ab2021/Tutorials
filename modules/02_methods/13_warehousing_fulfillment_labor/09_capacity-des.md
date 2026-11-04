# Capacity and Discrete-Event Simulation

> Module 13 — Warehousing, Fulfillment, and Labor Analytics
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: topics 01–05 and basic probability
> Version 2026.5

## 1. Why this topic exists

A distribution center manager must decide whether the building can make its peak shipping promise.
A daily total divided by a nominal hourly rate is not enough to answer that question.
Orders arrive in bursts, product mix changes the work per order, and staff and equipment are shared.
A downstream pack queue can fill while pickers are nominally productive.
A carrier cutoff makes the timing of completion as important as the number completed.
A capacity plan that ignores these interactions can authorize too little labor or spend money at the wrong constraint.

Discrete-event simulation (DES) represents individual changes of state in time order.
It does not predict the future; it calculates the consequences of explicit assumptions and sampled variability.
You can compare decisions under matched demand scenarios before committing a peak roster.
The decision owner should still check feasibility with operations, labor relations, safety, and finance.

A forecast is a probability-informed estimate of demand.
A plan is the proposed staffing and operating policy for that demand.
A target is the internal performance objective.
A commitment is the externally promised service level and cutoff.
Never label the output of one random run as a customer commitment.

This lesson follows an illustrative pick-and-pack building with a parcel cutoff.
All examples are invented for teaching, not a claim about a particular operator.
Currency is US dollars (USD), using illustrative 2026 rates.
The model is deliberately smaller than a real warehouse and excludes several mechanisms explicitly.

## 2. Learning outcomes

After this lesson you can:

1. Distinguish theoretical, effective, and time-dependent capacity and explain their decision uses (exercise 1).
2. Translate a shipping promise into entities, boundaries, events, queues, and resource rules (exercise 2).
3. Compute a deterministic capacity screen and identify its limitations (exercise 3).
4. Construct and explain a reproducible event simulation (exercise 4).
5. Compare staffing alternatives with matched demand and quantify cost and service uncertainty (exercise 5).
6. Diagnose data, validation, and regulatory risks before approving a plan (exercise 6).
7. Design an operational experiment and an escalation rule for peak (exercise 7).
8. Explain why utilization and throughput do not alone prove customer service (exercise 8).

## 3. The core mental model

```text
Forecast and promise             Roster, shifts, constraints
        |                                   |
        v                                   v
Arrival events -> pick queue -> pick workers -> pack queue -> pack workers
                       |                            |              |
                       +---- waiting and WIP -------+              v
                                                          completion timestamp
                                                                   |
                                                         cutoff and cost metrics
```

Each order is an entity with an arrival time and a deadline.
A resource is a worker pool; a queue holds entities waiting for an available worker.
An event changes the state at a specific time: arrival, pick completion, or pack completion.
The simulation clock jumps to the next event rather than stepping through every empty minute.
An event calendar orders future events by timestamp and a deterministic tie-breaker.
On an arrival, the model adds work and starts pick immediately if capacity is free.
On pick completion it releases the pick worker and adds the order to the pack queue.
On pack completion it records the completion time and releases the pack worker.

A service result is not just total completions divided by total orders.
You must specify the eligible population, deadline, and treatment of orders still in process.
An order that is unfinished at cutoff counts late even if it eventually completes.
Simulate beyond cutoff when you need the eventual backlog drain time.
The model below drains all accepted orders but tests their recorded completion against cutoff.
The model starts empty; real carry-in must be modeled explicitly when relevant.

The warehouse is a network of interacting constraints, not an assembly of independent averages.
The bottleneck can move as arrival intensity, staffing, or work mix changes.
Increasing picking speed can worsen the pack queue if pack capacity does not increase.
Even when average capacity exceeds average demand, time-local overload creates queues.
A queue may be intentional protection against starvation, but it consumes space and time.
Little's Law, WIP = throughput × flow time, is a useful steady-state cross-check when its conditions hold.
It is not a substitute for a transient, cutoff-specific experiment.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| **Entity** | One modeled order, tote, line, or other moving object; here order. | Treating unlike order sizes as identical without checking. |
| **Arrival** | Entry into modeled boundary at a timestamp in minutes. | Using order creation when release is the actual entry. |
| **Release policy** | Rule determining when eligible orders enter work. | Assuming all demand is instantly released. |
| **Event** | Instantaneous state change at a simulation time. | Confusing event with the duration between events. |
| **Event calendar** | Time-ordered future events, with tie priority. | Using an unordered list without tie behavior. |
| **Simulation clock** | Time of the most recently processed event, minutes. | Assuming runtime equals modeled elapsed time. |
| **State** | Resource occupancy, queue contents, and entity status at a time. | Recording only output totals. |
| **Resource** | Constrained worker, workstation, device, or space. | Treating all staff as interchangeable. |
| **Server** | A unit of a resource processing one entity at a time. | Assuming a worker handles unlimited orders. |
| **Queue** | Ordered waiting entities, measured in orders. | Calling in-service work waiting work. |
| **WIP** | Work in process inside a stated boundary, orders. | Excluding waiting or counting arrivals twice. |
| **Service time** | Active processing time per order, minutes. | Including queue delay and then modeling the delay again. |
| **Cycle time** | Completion minus entry time, minutes per order. | Equating it with touch time. |
| **Utilization** | Busy resource time divided by available resource time. | Using paid hours when the denominator is scheduled hours. |
| **Nominal capacity** | Ideal rate under stated technical assumptions, orders/hour. | Treating it as deliverable throughput. |
| **Effective capacity** | Rate after explicitly modeled losses, orders/hour. | Applying the same allowance twice. |
| **Throughput** | Completed entities per stated interval, orders/hour. | Equating starts with completions. |
| **Bottleneck** | Constraint limiting additional system throughput under a scenario. | Assuming the same station constrains every hour. |
| **Cutoff** | Deadline for completion or handoff, local timestamp. | Treating end of shift and carrier handoff as identical. |
| **On-time fraction** | On-time eligible orders divided by all eligible orders. | Excluding unfinished orders from denominator. |
| **Replication** | One independent simulated trajectory under a scenario. | Calling individual orders independent scenario replications. |
| **Random seed** | Initial state for a pseudorandom stream. | Treating a seed as evidence of accuracy. |
| **Warm-up** | Discarded initial interval for a steady-state model. | Discarding actual opening-period backlog in a finite peak. |
| **Carry-in** | Unfinished work already present at model start, orders. | Assuming the system always opens empty. |
| **Drain time** | Time from a chosen stop point until remaining work completes, minutes. | Reporting drain throughput as before-cutoff throughput. |
| **Common random numbers** | Aligned random inputs across competing alternatives. | Assuming same seed automatically preserves alignment. |
| **Verification** | Check implementation against model specification. | Treating plausible output as proof of correctness. |
| **Validation** | Assess fitness against real behavior for the decision. | Claiming a model is universally valid. |
| **Scenario** | A coherent set of demand and operating assumptions. | Mixing incompatible worst cases without labeling. |
| **Confidence interval** | Interval quantifying sampling uncertainty under assumptions. | Calling it a guarantee for actual peak day. |
| **Service level** | Explicit customer-facing success metric and denominator. | Substituting average cycle time for deadline compliance. |
| **Priority rule** | Policy for selecting the next waiting order. | Assuming first-in first-out when priority orders exist. |
| **Blocking** | Inability to move output because downstream space is full. | Representing an infinite buffer as physical reality. |
| **Starvation** | A resource has no eligible work to process. | Calling it equipment downtime. |
| **Setup** | Time to prepare resource for a different job or batch. | Ignoring sequence dependence. |
| **Rework** | Repeat work required after a quality failure. | Treating corrections as free. |
| **Censoring** | Incomplete observation of durations at extraction. | Dropping open tasks and biasing service times downward. |
| **Calibration** | Choosing input values using observed data. | Using validation data as tuning data without disclosure. |
The unit of analysis must remain consistent from input through output.
If one model entity represents an order, an observed line-level service time needs an order-level conversion.
Averages of averages should be weighted by the relevant population.
A model may use a more detailed entity hierarchy when consolidation materially affects capacity.
Document each conversion and the population it covers.

## 5. The mechanics, step by step

### 5.1 Frame the decision

1. Name the accountable decision owner and the decision date.
   Write down which choices are actually available before peak.
   Examples are adding a pack shift, moving a break, or reserving overtime.
   Do not compare alternatives that cannot be staffed or physically installed in time.
2. State the promise as an observable event and cutoff.
   Separate order acceptance, release, packing, manifesting, and carrier handoff.
   A completed carton is not necessarily a carrier scan.
3. Define the eligible population and the deadline for each order class.
   Make exclusion rules explicit before seeing the outcome.
   A canceled order may be excluded only under the agreed metric definition.
4. State the objective and the binding constraints.
   Examples are lowest incremental labor cost subject to a minimum on-time fraction and safe occupancy.
   Record hard constraints separately from preferences and penalty estimates.
5. Choose the experiment horizon.
   A peak day with an opening backlog is a terminating system, not necessarily a steady-state system.
   Include the lead-in or initialize a realistic backlog and drain after the cutoff.

### 5.2 Map the physical process

6. Walk the flow with an operator and draw its actual branches.
   Capture inbound release, pick, consolidation, pack, manifest, sort, staging, and handoff where material.
   Identify rework loops and where an order can wait.
7. Choose entity grain based on the decision.
   Order-level simulation is convenient for a cutoff but can hide the interaction of lines and cartons.
   Line-level simulation requires explicit order consolidation before shipment.
8. Record resources and their calendars.
   Include headcount, skill eligibility, breaks, equipment availability, and shared duties.
   Model a worker shared between picking and replenishment only once.
9. Define queue disciplines and ownership.
   First-in first-out (FIFO) is a policy, not a law of queueing.
   A real priority rule may prioritize urgent orders and change service across classes.
10. Specify finite buffers when physical congestion affects the decision.
    Determine whether an upstream worker remains occupied while blocked.
    The executable example assumes infinite buffers and therefore cannot assess congestion safety.

### 5.3 Build inputs from evidence

11. Extract arrival timestamps at the actual model boundary.
    Retain the date, hour, order class, and source timezone.
    Separate creation from release so that a release policy is not hidden in the arrival process.
12. Compute active work durations from event pairs for each stage.
    Match start and completion identifiers and remove duplicate events under an explicit rule.
    Separate pauses, equipment downtime, and handoffs from active service when modeled separately.
13. Preserve observations with unfinished work as censored rather than silently dropping them.
    Investigate negative durations, clock skew, duplicate scans, and missing endpoints.
    Compare extracted totals to warehouse management system (WMS) operational counts.
14. Stratify by the variables that change workload.
    Product size, lines per order, channel, packaging, and required handling can all matter.
    Small strata may need pooling or an empirical mixture with an uncertainty range.
15. Preserve relevant dependence.
    Large orders can require both longer picking and longer packing.
    A shared daily demand shock can also affect every hour; independently drawing hourly averages erases it.
16. Model calendars with actual local timestamps and daylight-saving rules.
    Confirm the carrier cutoff, shift start, and break schedule in the same time basis.
    Record any clock conversions and ambiguities in the data lineage.
17. Distinguish structural assumptions from fitted parameters.
    A queue discipline is a policy assumption; mean service time is a parameter estimate.
    Both can change conclusions and both need review.

### 5.4 Construct and verify the event logic

18. Specify the state before writing code.
    At minimum track each queue, each busy count, the event calendar, and completion timestamps.
    State invariants include nonnegative queue lengths and busy counts no larger than available servers.
19. Specify every event transition.
    An arrival enqueues work; completion releases a server and sends the entity downstream.
    Starting service removes an entity from a queue and schedules exactly one completion.
20. Set a deterministic policy for equal-time events.
    The example uses an increasing insertion sequence to break ties.
    Other policies are acceptable if documented and consistently implemented.
21. Decide the effect of a shift ending while an order is in service.
    A nonpreemptive policy can allow work to finish; a preemptive policy must record remaining time.
    The simple model holds resources available until all accepted work drains.
22. Hand-trace a tiny example with one worker and two arrivals.
    Check queue order, start times, completion times, and resource release.
    Repeat with zero arrivals and simultaneous arrivals.
23. Check conservation at the end of each run.
    Accepted orders equal completed orders plus queued and in-service orders.
    At full drain, accepted equals completed; no order completes twice.
24. Verify behavior in limiting cases.
    With one order and idle resources, completion is arrival plus both service times.
    Adding a resource should not be presumed to improve every individual outcome under every policy.
    Under a controlled FIFO deterministic test, it should produce the expected queue behavior.

### 5.5 Validate and experiment

25. Hold out representative historical days for validation.
    Reproduce their arrivals, calendar, known disruptions, and opening work where possible.
    Compare hourly completions, queue buildup, cutoff misses, and resource busy fractions.
26. Review mismatches with operators rather than tuning solely to an aggregate target.
    A good daily throughput total can hide a bad morning release pattern.
    Record which discrepancies remain and whether they affect the decision.
27. Construct feasible candidate plans with a baseline and clear changes.
    Make incremental staffing, overtime, transfer, or release rules explicit.
    Price all alternatives on a consistent incremental basis.
28. Define a demand scenario set before reviewing results.
    Include a central forecast, peak-volume case, arrival-concentration case, and operational disruption.
    Do not call a scenario a percentile without a calibrated probability model.
29. Run independent replications within each scenario.
    A single stochastic trajectory is a demonstration, not a decision-quality estimate.
    Retain seeds, model version, input snapshot, and policy settings for reproduction.
30. Compare plans using matched random inputs where possible.
    Pair outcomes by replication and calculate uncertainty on the per-replication differences.
    Ensure service-time draws remain aligned by order and stage rather than merely reusing a seed.
31. Report distributions, not only average on-time performance.
    A mean across days and the proportion of days meeting a target answer different questions.
    Include the number of simulated days, standard error, and interval method.
32. Test sensitivity to assumptions with the largest potential decision impact.
    Vary service tails, demand concentration, carry-in, absenteeism, and cutoff timing.
    Use a coherent joint stress scenario as well as individual sensitivity changes.
33. Choose a plan only after checking operational and financial feasibility.
    Confirm hiring lead time, skills, station availability, supervision, and safety.
    Specify a trigger for fallback action when actual demand or queue growth exceeds the plan.
34. Monitor after deployment and update the model.
    Compare actual hourly arrivals, completions, queue age, and missed promises with planned bands.
    Assign a named owner to investigate deviations and approve changes.
### Operational checkpoint: Order release

**Mechanism.** A large wave can create a queue even when daily capacity exceeds demand.
**Evidence.** Compare release timestamps to order creation and hourly completion.
**Decision use.** Test a staggered release only if downstream commitment and inventory permit.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Opening backlog

**Mechanism.** Starting empty removes work that may already occupy the first shift.
**Evidence.** Count open eligible orders by stage at the opening timestamp.
**Decision use.** Initialize the queue and remaining service for in-process work.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Carrier handoff

**Mechanism.** Pack completion is earlier than the truck departure event.
**Evidence.** Measure the distribution from pack completion to carrier scan.
**Decision use.** Move the modeled deadline earlier or represent the handoff process.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Shift breaks

**Mechanism.** A daily average hides simultaneous capacity losses.
**Evidence.** Reconcile planned and actual break windows by skill.
**Decision use.** Represent time-varying capacity and check whether staggering is permitted.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Absenteeism

**Mechanism.** Scheduled headcount is not actual available headcount.
**Evidence.** Compare rostered and present workers by hour and skill.
**Decision use.** Use coherent attendance scenarios and price reserve coverage.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Skill restrictions

**Mechanism.** A pooled team can appear to cover a station it cannot operate.
**Evidence.** Map certified or trained workers to eligible activities.
**Decision use.** Use separate resource pools or explicit assignment constraints.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Equipment downtime

**Mechanism.** A nominal worker may be idle because a device is unavailable.
**Evidence.** Join downtime intervals to station and shift calendars.
**Decision use.** Model the equipment constraint or apply a documented availability adjustment, not both.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Replenishment

**Mechanism.** Pick time changes when a forward location is empty.
**Evidence.** Match stockout and replenishment events to delayed picks.
**Decision use.** Add replenishment resources and shortages when they drive the decision.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Large-order mix

**Mechanism.** A stable order count can carry far more work than usual.
**Evidence.** Compare line, unit, and carton distributions by day and channel.
**Decision use.** Sample complete order profiles rather than only mean seconds per order.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Packing material

**Mechanism.** A missing carton size creates workarounds and delays.
**Evidence.** Track material stockouts and alternative packaging use.
**Decision use.** Add a supply constraint in scenarios where it is material.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Priority orders

**Mechanism.** FIFO can hide systematic service failure for a customer class.
**Evidence.** Inspect dispatch rules and queue bypass events.
**Decision use.** Model the actual priority discipline and report results by class.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Batch picking

**Mechanism.** Completion of a batch releases several orders together.
**Evidence.** Measure batch size, formation delay, and sortation time.
**Decision use.** Represent batch formation and downstream consolidation.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Travel congestion

**Mechanism.** More pickers can increase travel time in narrow aisles.
**Evidence.** Compare task durations across staffing and aisle occupancy.
**Decision use.** Use an occupancy-dependent delay only when supported by evidence.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Quality rework

**Mechanism.** Corrections consume capacity and can change deadlines.
**Evidence.** Link exception scans and repeat tasks to original orders.
**Decision use.** Represent rework probability and service duration without double counting.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Dock staging

**Mechanism.** Packed orders can wait for doors or transport equipment.
**Evidence.** Measure queue age and staging occupancy before departure.
**Decision use.** Add finite staging and dock resources if handoff is the commitment.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Inventory availability

**Mechanism.** An unavailable item can stop an entire order.
**Evidence.** Reconcile allocations, shorts, and substitutions with order outcomes.
**Decision use.** Separate eligible available orders from held orders and report both.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Task interleaving

**Mechanism.** The same person may alternate replenishment and picking.
**Evidence.** Reconstruct time use from assignments and location events.
**Decision use.** Model shared capacity and switching rules explicitly.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Overtime

**Mechanism.** Overtime is not a free extension of a shift.
**Evidence.** Confirm wage premium, willingness, approval, and fatigue constraints.
**Decision use.** Price incremental hours and specify the legal and safety review.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Arrival burst

**Mechanism.** Two days with equal total demand can have different cutoff performance.
**Evidence.** Compare within-day arrival profiles conditional on total volume.
**Decision use.** Preserve correlated hourly counts in scenario sampling.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Service-time tail

**Mechanism.** Rare long tasks can occupy a scarce packer near cutoff.
**Evidence.** Inspect upper quantiles and verify extreme records.
**Decision use.** Test an empirical tail and a plausible alternative; do not delete valid extremes.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Weather disruption

**Mechanism.** A site-wide event can affect arrival, attendance, and carrier pickup.
**Evidence.** Identify joint historical disruption days and available contingencies.
**Decision use.** Construct a joint scenario instead of independent optimistic averages.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Data latency

**Mechanism.** Late scans may look like operational delay or early completion.
**Evidence.** Compare device, server, and physical observation timestamps.
**Decision use.** Correct known clock errors and test sensitivity to remaining lag.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Peak season mix

**Mechanism.** Historical normal days may not represent promotions.
**Evidence.** Compare channel and product mix to the peak demand plan.
**Decision use.** Reweight or scenario-test mix, with any extrapolation labeled.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Temporary staff

**Mechanism.** New workers may need training and supervision.
**Evidence.** Measure performance by tenure while respecting privacy.
**Decision use.** Use skill-appropriate service inputs and a realistic ramp schedule.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Rate incentives

**Mechanism.** A measured rate may shift work or quality risk elsewhere.
**Evidence.** Pair speed measures with errors, rework, and worker feedback.
**Decision use.** Optimize system outcomes rather than a single station rate.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Multi-carton orders

**Mechanism.** One packed carton does not complete a multi-carton order.
**Evidence.** Link carton IDs and final carton completion to order IDs.
**Decision use.** Apply the promise to the last required carton or actual handoff.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Order cancellations

**Mechanism.** Removing canceled work after it consumed labor understates load.
**Evidence.** Find cancellation timing relative to release and service.
**Decision use.** Cancel at the appropriate event and retain consumed capacity.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Cutoff changes

**Mechanism.** Earlier carrier departure can invalidate a formerly feasible plan.
**Evidence.** Confirm date-specific departure and handoff requirements.
**Decision use.** Recompute on-time status using the revised cutoff.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: System outage

**Mechanism.** Manual processing can have different speed and error rates.
**Evidence.** Collect incident durations and fallback process records.
**Decision use.** Model an outage and recovery scenario if contingency planning needs it.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### Operational checkpoint: Productivity drift

**Mechanism.** A rate measured during a quiet hour may not hold at peak.
**Evidence.** Compare service distributions by volume, shift, and day.
**Decision use.** Test conditional rates and avoid treating correlation as causal proof.
**Owner.** Have the process owner confirm the rule before translating it into model logic.
**Output.** Record the assumption, source period, uncertainty, and any unresolved discrepancy.
**Interpretation.** A favorable aggregate result does not remove the need to inspect this exposure.
**Boundary.** Include the mechanism when it can change the decision, not merely because data exist.
**Review.** Compare the scenario result with the baseline on the same demand population.
**Action.** Escalate if the proposed policy is not feasible on the floor.
### 5.6 Experiment design in more depth

**Set the comparison unit.**
The replication is a simulated day, not a single order.
Orders on the same day share demand conditions and resource congestion.
Treating all orders as independent observations would understate uncertainty.
For each day, save demand volume, mix, capacity, service, and total cost.
Keep the original order-level output for diagnosis.

**Separate randomness from business uncertainty.**
Monte Carlo noise declines as the number of replications increases.
Uncertainty in a service-time distribution does not disappear merely by repeating draws.
Neither does the possibility that a promotion changes the order mix.
Vary uncertain input assumptions in an outer scenario layer.
Report each assumption and avoid averaging incompatible scenarios without justified weights.

**Pair alternatives deliberately.**
Generate a day of arrivals and order profiles before evaluating either staffing plan.
Attach each order's pick and pack requirements to its identifier.
Both alternatives then encounter the same underlying workload.
Compute each day's plan B minus plan A difference.
An intervention can change the realized queue and completion time without changing the sampled work.

**Choose the number of replications for a precision goal.**
Start with an exploratory batch and estimate variability of paired differences.
Decide what interval width would be useful to the business decision.
Increase the number of independent days until precision is adequate or a declared limit is reached.
Do not stop merely because an intermediate estimate favors the preferred alternative.
Predefine a stopping procedure or disclose adaptive stopping and its implications.

**Keep rare-event risks visible.**
If a severe backlog occurs once in a small run, the estimated frequency is imprecise.
If it never occurs, its true probability is not proven to be zero.
Report event counts and the number of replications.
Use targeted stress scenarios to expose mechanisms, without calling their frequency a probability.
More sophisticated rare-event methods require their own validation.

**Define an economic comparison.**
Calculate incremental paid worker-hours, not just active minutes.
Add only relevant incremental costs, such as approved overtime premium or temporary labor fees.
Do not subtract an avoided late-order penalty unless the agreement actually makes it avoidable.
Keep uncertain customer consequences separate from contractual cash payments.
Show both the monetary comparison and the service constraint.

**Example of a simple paired cost calculation.**
Suppose a reserve packer costs USD 192 for one eight-hour day.
Suppose the baseline has an expected 12 late orders per day under a stated scenario.
Suppose the alternative has an expected 7 late orders per day under that same scenario.
The estimated reduction is 12 − 7 = 5 late orders per day.
The incremental staffing cost is USD 192 / 5 = USD 38.40 per late order avoided.
This ratio is undefined when the estimated reduction is zero and misleading when it is negative.
It is not a net benefit unless a defensible value or cost of lateness is also specified.

**Translate service into a decision rule.**
The target might be at least 98% on time on 95% of operating days.
This differs from an average 98% of orders on time across all days.
Write the threshold and denominator before comparing plans.
State how borderline estimates and uncertainty are handled.
A conservative manager may require a lower confidence bound to exceed the threshold.
That rule must be agreed in advance and is a management choice, not a universal standard.

**Look beyond the average order.**
Report urgent and standard orders separately when their promises differ.
Check whether a policy improves the pooled average by delaying a smaller customer class.
Show counts alongside percentages for small segments.
Track queue age as well as queue length.
An old queue can signal a promise risk that a small total queue conceals.

**Specify an escalation threshold.**
For example, a site may compare the 14:00 backlog with an approved scenario band.
The trigger must be based on measured open work, available capacity, and remaining time.
If the trigger is crossed, an authorized supervisor may activate a preapproved reserve.
The action must include a safe staffing and station-availability check.
Record the trigger time, action, and actual outcome for later learning.

**Use a time-dependent backlog screen.**
Suppose 90 orders remain at 14:00 and the cutoff is 16:00.
If verified remaining system capacity is 40 orders per hour, at most 80 orders can finish in two hours.
The gap is at least 90 − 80 = 10 orders, before considering new arrivals or sequencing.
That is a necessary-capacity warning, not a complete queue simulation.
If additional eligible orders are expected, include them in the remaining work estimate.

**Validate the intervention, not only the baseline.**
A model calibrated at three packers may not describe four packers in a constrained aisle.
Check whether extra workers share tables, printers, replenishment, or supervision.
Historical evidence from comparable staffing levels is preferable where available.
If no such evidence exists, test plausible congestion effects as explicit uncertainty.
Do not claim a causal productivity gain from a correlation alone.

**Interpret finite-horizon results.**
A terminating peak-day simulation uses the actual opening state and stops its service assessment at cutoff.
It can continue afterward to calculate drain time and overtime exposure.
Do not discard the opening period as warm-up when it is part of the decision.
Conversely, a long-run steady-state study may need an initialization-bias strategy.
The analysis design follows the business question rather than a fixed simulation ritual.

**Record reproducibility artifacts.**
Retain the model version, input extracts, transformation rules, seeds, and scenario definitions.
Record the release policy, resource calendars, queue disciplines, and cutoff convention.
Save the expected hand-trace results alongside the implementation.
Document who approved inputs and when the operational constraints were checked.
Protect sensitive worker-level data and use an approved retention period.

**Review the recommendation as an operational contract.**
State the decision, scenario coverage, cost, expected service, and important residual risks.
List what must be true for the plan to be executable on the day.
Identify the fallback when actual conditions exceed those assumptions.
Identify the owner who monitors the leading indicators.
The decision owner can reject a low-cost plan that does not meet a binding safety or service requirement.

## 6. Worked example

### 6.1 Decision and assumptions

A fictional parcel warehouse must decide whether to reserve an additional packer for a peak shift.
The forecast is 420 eligible orders released between 08:00 and 16:00 on a particular 2026 day.
The plan has four pickers and three packers available for eight hours each.
The internal target is at least 98% of eligible orders packed by 16:00.
The illustrative customer commitment is packing by 16:00; actual carrier handoff is outside this example.
Assume an empty opening system and that each order requires one pick and one pack operation.
Each picker averages 4.0 active minutes per order and each packer 3.0 active minutes per order.
Assume each person has 420 effective working minutes in the eight-hour shift after breaks and losses.
These effective minutes already account for allowances; do not subtract the same break a second time.

### 6.2 Deterministic screen

Pick capacity = 4 workers × 420 minutes/worker ÷ 4.0 minutes/order = 420 orders per shift.
Pack capacity = 3 workers × 420 minutes/worker ÷ 3.0 minutes/order = 420 orders per shift.
Demand = 420 orders per shift, so the nominal slack at both stations is zero orders.
Average utilization implied by this screen is 100% of effective available time at both stations.
This screen does not prove that 420 orders can complete by 16:00.
It ignores the need to pick an order before packing it, arrival peaks, variability, and queues.
Any late release can miss the cutoff even with unused capacity earlier in the day.

Add a fourth packer with the same 420 effective minutes.
Pack capacity becomes 4 × 420 ÷ 3.0 = 560 orders per shift.
The pack screen now has 140 orders of slack, while pick remains at 420 orders per shift.
The extra packer cannot fix a shortage of pick capacity or an order released after the deadline.
At USD 24 per paid hour and an eight-hour shift, the added straight-time cost is USD 192 per day.
This is an illustrative incremental cost, not a quote and not an overtime calculation.
A decision to reserve this worker requires a realistic simulation and operational feasibility review.

### 6.3 Why within-day timing matters

Consider two orders, A and B, released at 15:53 and 15:54 respectively.
Assume one idle picker and one idle packer, 4 minutes pick and 3 minutes pack per order.
A picks from 15:53 to 15:57 and packs from 15:57 to 16:00.
B picks from 15:57 to 16:01 and packs from 16:01 to 16:04.
With a cutoff inclusive of exactly 16:00, A is on time and B is late.
The two-order on-time fraction is 1/2 = 50%, despite ample unused time earlier in the day.
Adding only a packer cannot speed B's pick completion in this trace.
If a second picker were available, B could pick 15:54–15:58 and pack 16:00–16:03 with one packer.
With a second picker and second packer, B could instead pack 15:58–16:01 and still miss cutoff.
Thus even simultaneous resource additions cannot reverse an intrinsically late release.

### 6.4 Interpreting an experiment

Suppose a properly validated experiment produces paired daily on-time fractions for each plan.
Compute the fraction meeting the target per day as well as the mean fraction across days.
Also report mean incremental paid hours, the peak queue, and carryover at cutoff.
Do not infer those outputs from the deterministic screen; they require the actual simulation inputs.
The code below demonstrates event logic on a separate small, exactly checkable dataset.
Its output is not a statistical forecast for the 420-order case.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It simulates a FIFO pick-and-pack system with constant per-order service times.
The event list is ordered by time and insertion sequence, including simultaneous events.
Workers remain available while the system drains, and queues are unbounded.
All arrival times and durations are minutes from opening; cutoff is inclusive.
Run the entire block as a script; the expected output follows it.

```python
from collections import deque
from heapq import heappush, heappop
from itertools import count


def simulate(arrivals, pickers, packers, pick_time, pack_time, cutoff):
    if pickers < 1 or packers < 1:
        raise ValueError('Each stage needs at least one worker')
    if pick_time <= 0 or pack_time <= 0:
        raise ValueError('Service times must be positive')
    calendar = []
    serial = count()
    pick_queue, pack_queue = deque(), deque()
    busy_pick = busy_pack = 0
    completion = {}

    def schedule(time, kind, order):
        heappush(calendar, (time, next(serial), kind, order))

    for order, arrival in enumerate(arrivals):
        schedule(arrival, 'arrival', order)

    while calendar:
        now, _, kind, order = heappop(calendar)
        if kind == 'arrival':
            pick_queue.append(order)
        elif kind == 'pick_done':
            busy_pick -= 1
            pack_queue.append(order)
        elif kind == 'pack_done':
            busy_pack -= 1
            completion[order] = now
        else:
            raise ValueError('Unknown event')

        while pick_queue and busy_pick < pickers:
            job = pick_queue.popleft()
            busy_pick += 1
            schedule(now + pick_time, 'pick_done', job)
        while pack_queue and busy_pack < packers:
            job = pack_queue.popleft()
            busy_pack += 1
            schedule(now + pack_time, 'pack_done', job)

    assert len(completion) == len(arrivals)
    assert busy_pick == 0 and busy_pack == 0
    assert not pick_queue and not pack_queue
    on_time = sum(time <= cutoff for time in completion.values())
    return completion, on_time


if __name__ == '__main__':
    done, on_time = simulate([0, 1, 2], 1, 1, 4, 3, 10)
    print('Completion minutes:', [done[i] for i in range(3)])
    print('On time:', on_time, 'of', len(done))
    print('Late:', len(done) - on_time)
```

Expected output:

```text
Completion minutes: [7, 11, 15]
On time: 1 of 3
Late: 2
```

The first order picks during minutes 0–4 and packs during minutes 4–7.
The second picks during minutes 4–8 and packs during minutes 8–11.
The third picks during minutes 8–12 and packs during minutes 12–15.
Two of three orders finish after minute 10, so they are late even though the model drains.
A zero-order input returns an empty completion dictionary and zero on-time orders.
The caller must define an empty-population service fraction rather than dividing by zero.

This artifact verifies event mechanics; it is not a calibrated random peak model.
For a decision experiment, sample arrivals and order-level durations outside the simulator.
Store each order's sampled durations and pass them into both alternatives to preserve pairing.
Represent breaks, varying server counts, carry-in, and the real shipping boundary before use.
For replication differences `d_i`, estimate the mean difference as `sum(d_i)/n`.
Estimate its standard error as the sample standard deviation of `d_i` divided by `sqrt(n)`.
Use an appropriate t interval for independent, approximately well-behaved replication differences.
For rare misses or strongly skewed outputs, consider a justified alternative interval and more runs.
An interval quantifies simulation sampling error conditional on inputs; it does not cover model bias.
## 8. Data, standards, and regulatory context

A WMS is a system of record for operational events, not a guarantee that all physical events were scanned.
Request event-level records with stable order and task identifiers and a documented extraction timestamp.
Join order eligibility and promised deadline to task events without duplicating a task across order lines.
Use an explicit timezone and preserve both original and normalized timestamps.
When clocks disagree, quantify the disagreement rather than silently sorting physically impossible events.
Document data lineage, population filters, event definitions, and exception handling.
An audit should be able to reconstruct a single order's path from raw events to modeled completion.

GS1 identification standards can help trading partners identify items, locations, and logistic units consistently.
They do not prescribe a warehouse simulation or guarantee the accuracy of event timestamps.
The GS1 EPCIS standard provides an event-sharing vocabulary for visibility across organizations.
Its applicability depends on the actual trading-partner implementation and any contract or law.
Treat internal WMS events and EPCIS events as distinct schemas until a mapping is reviewed.

In the United States, the Occupational Safety and Health Administration (OSHA) administers workplace safety rules.
Applicable requirements depend on the activity, equipment, jurisdiction, and facts of the operation.
For example, powered industrial truck training requirements are addressed in 29 CFR 1910.178(l).
A simulation result does not certify compliance with that rule or authorize an untrained worker to drive.
OSHA's warehousing resources describe hazards and prevention; consult current applicable requirements.
Do not infer that a faster staffing policy is safe merely because simulated throughput increases.

If worker-level data are used, limit access and retention according to applicable law and organizational policy.
Aggregate reporting where individual identification is unnecessary for the decision.
Review employee monitoring and scheduling practices with the appropriate legal and labor stakeholders.
This lesson is educational and is not legal advice; rules and contracts may change.
A documented safety review is a deployment gate, not a statistical confidence interval.

## 9. Industry practice and case evidence

Warehouse simulation is used to compare flow configurations before expensive operational changes.
The literature documents both the benefits of simulation and the need for credible input data and validation.
Banks, Carson, Nelson, and Nicol's *Discrete-Event System Simulation* describes general DES methods.
Their framework is methodological evidence, not proof that a particular warehouse will meet a target.

The peer-reviewed warehouse design review by Roodbergen and Vis surveys automated storage and retrieval systems.
It distinguishes design and operational decisions in such systems and cites analytical and simulation approaches.
An automated storage result does not transfer automatically to a manual parcel operation.
Use the reference to identify mechanisms and research methods, not as a promised productivity improvement.

The Kiva Systems paper by Wurman, D'Andrea, and Mountz describes coordinating mobile robots in warehouses.
It explains a real system and its coordination problem; it does not validate the numerical example here.
[INFERENCE] Mobile robots make traffic interaction and equipment availability relevant simulation inputs.
The size of those effects remains site-specific and must be measured.

In practice, a capacity meeting often combines a deterministic screen, a scenario experiment, and operator review.
This is a recommended workflow in this lesson, not a claim that every company follows the same process.
A screened alternative may be rejected for unavailable staff, floor space, safety, or implementation lead time.
A model should show the effect of such constraints rather than presenting an infeasible optimum.
The decision memo should distinguish published evidence, measured site inputs, and analyst judgment.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Daily average only | Good daily capacity but late afternoon backlog | Bursty arrivals exceed local capacity | Late cutoffs | Use hourly arrivals and event timing |
| Wrong deadline | Pack and truck scans differ | Internal completion substituted for handoff | False service claim | Model promised event and transfer |
| Missing backlog | Opening queue absent from model | Work already in process omitted | Overstated spare capacity | Initialize observed carry-in |
| Double allowances | Break excluded from rate and roster | Same lost time applied twice | Understated capacity | Trace denominator and loss accounting |
| Ignoring allowances | Paid time treated as active time | Break and nonproductive work omitted | Overstated capacity | Use explicit calendars or effective time |
| Infinite buffer | Floor staging repeatedly overflows | Model omits physical blocking | Unsafe or infeasible plan | Model space and upstream blocking |
| Wrong mix | Peak lines per order increase | Average service time no longer representative | Optimistic throughput | Condition service on order profile |
| Dropped open tasks | Only completed tasks extracted | Long tasks censored out | Biased service times | Handle censoring and reconcile counts |
| Shared worker duplicated | One name appears in two full rosters | Resource assigned twice | Impossible staffing | Model shared eligibility |
| Single seed | One run drives decision | Random noise mistaken for signal | Unstable selection | Use independent paired replications |
| Seed misalignment | Policies consume random draws in different order | Same seed gives different order workloads | Noisy or biased comparison | Pre-sample by order and stage |
| Tuning to holdout | Validation dates repeatedly retuned | Evidence no longer independent | Overconfidence | Reserve a new holdout |
| No shift boundary | Model runs staff indefinitely | Overtime appears free | Cost and service distorted | Model calendar and overtime policy |
| Wrong priority | Urgent orders miss despite strong average | FIFO differs from operating rule | Customer inequity | Model priority and class metrics |
| Hidden rework | Quality exceptions growing | Repeat tasks omitted | Capacity overstatement | Include measured rework |
| Excess precision | Results reported to hundredths from weak data | Sampling interval confused with input certainty | False confidence | Report sensitivity and input ranges |
| Unsafe staffing | Aisle occupancy exceeds operating limit | More workers assumed always beneficial | Safety and congestion risk | Apply floor limits and review |
| Wrong unit | Order rate compared to line demand | Numerator and service grain differ | Invalid capacity comparison | Convert using measured mix |
| Time-zone error | Negative or shifted durations | Device and server clocks differ | False timing | Normalize and audit timestamps |
| No follow-through | Actual queue departs from plan | No monitoring owner | Failure persists | Set triggers and escalation owner |
## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Orders are independent entities | Makes the small model tractable | Shared batches and order consolidation | Trace orders through batch and carton identifiers |
| FIFO queues | Defines who starts next | Customer or expedite priorities | Compare queue sequence with actual task assignments |
| Constant service time | Makes the teaching trace reproducible | Order heterogeneity and long tails | Inspect conditional service distributions |
| Unlimited buffers | Avoids blocking logic in the example | Physical staging or aisle capacity | Measure occupancy and observed blocked time |
| Empty opening system | Specifies initial conditions | Carry-in backlog or in-service tasks | Count open work at the opening timestamp |
| Stable resources through drain | Allows every order to complete | Shift end, absence, or overtime restriction | Compare calendars and payroll records |
| No quality rework | Keeps the demonstration concise | Exceptions and repacking | Link corrections to original orders |
| Packing equals promise | Makes cutoff observable in the teaching case | Manifest or carrier scan is contractual event | Inspect actual customer and carrier agreements |
| Arrivals are known | Separates event logic from demand forecasting | Uncertain or changed release volumes | Backtest forecast and stress arrival profiles |
| No inventory holds | Makes all arriving orders eligible for work | Shorts or allocation failures | Reconcile holds and release status |
| Independent replications | Supports standard error of replication means | Shared random input or correlated resampling | Review sampling design and seed streams |
| Historical patterns transfer | Makes data-driven calibration useful | New layout, promotion, or worker mix | Compare current drivers with fitting period |
| Constant worker productivity | Simplifies resource representation | Fatigue, learning, congestion, or equipment issues | Stratify rates by shift, tenure, and occupancy |
| No worker sharing | Keeps resource pools distinct | Interleaved replenishment and picking | Reconstruct worker assignments |
| Accurate event clocks | Makes calculated durations meaningful | Clock skew and delayed scanning | Compare source, server, and physical observations |
| Demand scenario probabilities are known | Allows probability-weighted decision metrics | Uncalibrated stress cases | Backtest scenarios or label them unweighted |
A limitation is material if it could reverse the preferred plan or conceal an unacceptable risk.
The executable model deliberately omits randomness, shift calendars, and finite buffers.
It is a teaching artifact, not a production staffing recommendation.
The deterministic example uses effective time while the program uses continuous service time.
Do not combine their outputs without making those different assumptions consistent.

## 12. Exercises

### Exercise 1 — Capacity vocabulary

**Time:** 10 minutes.
**Task:** A station has five workers scheduled for 8 hours each.
Each worker has 60 minutes of breaks and other excluded time per shift.
Processing takes 5 active minutes per order.
Define scheduled and effective worker-minutes and calculate the effective station capacity.
Explain why that value is not guaranteed throughput.
**Deliverable:** Three calculations and one operational explanation.
**Acceptance:** Units are consistent and no allowance is counted twice.

### Exercise 2 — Model boundary

**Time:** 15 minutes.
**Task:** A customer promise is a carrier scan by 18:00 local time.
Your WMS supplies release, pick, pack, manifest, and dock handoff timestamps.
Define the entity, first and last modeled events, resources, two queues, and on-time denominator.
**Deliverable:** A one-page event map and a metric definition.
**Acceptance:** The carrier scan is not silently replaced by pack completion.

### Exercise 3 — Capacity screen

**Time:** 15 minutes.
**Task:** Six pickers each have 390 effective minutes and need 3.9 minutes per order.
Four packers each have 390 effective minutes and need 3.0 minutes per order.
Demand is 550 orders in one shift.
Calculate each stage's effective capacity and identify the deterministic constraint and slack.
**Deliverable:** Arithmetic in orders per shift and a caveat about timing.
**Acceptance:** Capacity is not described as a guarantee of on-time completion.

### Exercise 4 — Hand trace

**Time:** 15 minutes.
**Task:** Use the program's assumptions with arrivals at minutes 0 and 1.
There is one picker and one packer; pick takes 4 minutes and pack takes 3 minutes.
The cutoff is minute 8 inclusive.
**Deliverable:** Start and finish time at each stage and the on-time count.
**Acceptance:** Include queue wait and count the unfinished order as late.

### Exercise 5 — Paired comparisons

**Time:** 20 minutes.
**Task:** For four independent paired replications, plan B minus plan A yields
on-time fraction differences of 0.02, 0.04, 0.00, and 0.02.
Calculate the mean difference and sample standard deviation.
Explain what must be aligned for the pairing to be meaningful.
**Deliverable:** Arithmetic in percentage points and a one-paragraph interpretation.
**Acceptance:** Use the sample denominator and do not claim generalizability from four runs.

### Exercise 6 — Critique

**Time:** 20 minutes.
**Task:** A report says, “Average utilization is 75%, so every order meets the 16:00 cutoff.”
It contains only completed WMS tasks and no arrival profile or opening backlog.
Identify at least four distinct risks and an evidence source for each.
**Deliverable:** Risk–mechanism–evidence table.
**Acceptance:** Address unfinished tasks and time-varying arrivals separately.

### Exercise 7 — Design a decision experiment

**Time:** 30 minutes.
**Task:** Compare reserving one packer versus reserving one picker for a peak day.
The decision must meet a customer deadline without unsafe staging.
**Deliverable:** Scenario definitions, inputs, event rules, validation plan, costs, metrics, and escalation trigger.
**Acceptance:** Both alternatives use comparable demand; the plan includes uncertainty and operational feasibility.

### Exercise 8 — Interpret service

**Time:** 10 minutes.
**Task:** Three simulated days have 100 eligible orders each.
On-time counts are 100, 98, and 92.
The daily target is at least 98% on time.
**Deliverable:** Pooled on-time fraction and fraction of days meeting the target.
**Acceptance:** Keep order-level and day-level denominators distinct.

## 13. Self-check answers

### Answer 1

Scheduled time is 5 × 8 × 60 = 2,400 worker-minutes per shift.
Effective time is 5 × (480 − 60) = 2,100 worker-minutes per shift.
Effective capacity is 2,100 / 5 = 420 orders per shift.
Arrivals, sequencing, variability, and downstream constraints can prevent completing 420 by cutoff.

### Answer 2

An order is the entity, entering at release and leaving at confirmed carrier scan.
Pick and pack workers are resources; carrier handling and dock capacity may also be resources.
Example queues are released orders awaiting picking and picked orders awaiting packing.
The denominator is all eligible promised orders, including orders without a scan by 18:00.
A model ending at pack must explicitly model or conservatively account for downstream handoff.

### Answer 3

Pick capacity is 6 × 390 / 3.9 = 600 orders per shift.
Pack capacity is 4 × 390 / 3.0 = 520 orders per shift.
Pack is the deterministic constraint; its slack relative to 550 orders is −30 orders.
Pick slack is 600 − 550 = 50 orders per shift.
Even if total capacity exceeded demand, the cutoff could still be missed because of timing.

### Answer 4

Order 1 picks from minute 0 to 4 and packs from minute 4 to 7.
Order 2 waits for pick, picks from minute 4 to 8, and packs from minute 8 to 11.
Exactly one of two orders completes by minute 8; the other is late.

### Answer 5

The mean paired difference is (0.02 + 0.04 + 0.00 + 0.02) / 4 = 0.02, or 2 percentage points.
The sum of squared deviations is 0.0008 in squared fraction units.
The sample variance is 0.0008 / 3 and the sample standard deviation is about 0.01633.
That is approximately 1.63 percentage points across the four paired differences.
Pair the same scenario, arrivals, and order-level work requirements across both plans.
Four replications are too few to establish a robust decision.

### Answer 6

Late bursts can exceed available end-of-day capacity despite low average utilization.
An arrival-by-hour extract tests this mechanism.
Open tasks omitted from the extract bias observed durations and hide unfinished work.
A snapshot of open tasks and censored durations tests that mechanism.
Unrecorded opening backlog consumes time; compare the opening WIP snapshot.
The true promise may be carrier handoff, not completion; inspect agreements and scan timestamps.

### Answer 7 rubric

A strong design maps the actual promise and entity boundary and includes carry-in.
It models resource calendars, skill eligibility, queues, and physical staging constraints.
It compares alternatives on paired demand and cost with independent replications.
It validates historical hourly patterns and reports class service, cutoff misses, and uncertainty.
It names feasibility, safety, and escalation owners and specifies a measurable trigger.
No unique numerical answer exists without site data and policy constraints.

### Answer 8

Pooled on-time fraction is (100 + 98 + 92) / 300 = 290 / 300 = 96.67%.
Two of three days meet the 98% daily target, or 66.67% of days.
The two percentages answer different questions and cannot substitute for each other.

## 14. Interview questions
### Question 1

**Question:** Why can a warehouse with 80% average utilization miss a cutoff?
**Strong answer:** Discuss bursts, local overload, precedence, and deadlines.
**Weak answer reveals:** Equating a daily average with every interval.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 2

**Question:** What is the difference between a forecast and a staffing plan?
**Strong answer:** Separate uncertain demand from a chosen resource policy.
**Weak answer reveals:** Treating forecast volume as an approved roster.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 3

**Question:** How would you choose the simulation entity?
**Strong answer:** Tie grain to promise, mix, and consolidation.
**Weak answer reveals:** Always choosing orders regardless of process.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 4

**Question:** What goes into an event calendar?
**Strong answer:** Timestamp, event type, entity, and deterministic tie order.
**Weak answer reveals:** Advancing time without updating state.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 5

**Question:** How do you validate a capacity model?
**Strong answer:** Use independent historical periods and compare time-resolved measures.
**Weak answer reveals:** Matching one daily total and stopping.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 6

**Question:** How do you compare two staffing plans?
**Strong answer:** Use feasible alternatives, paired inputs, costs, service, and uncertainty.
**Weak answer reveals:** Comparing two unrelated single runs.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 7

**Question:** What does common random numbers require?
**Strong answer:** Align workload draws by entity and stage across alternatives.
**Weak answer reveals:** Assuming a common seed alone is sufficient.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 8

**Question:** Why can adding pickers make pack congestion worse?
**Strong answer:** More upstream completions can increase downstream WIP.
**Weak answer reveals:** Assuming local productivity always improves system service.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 9

**Question:** What is the role of Little’s Law?
**Strong answer:** State matching boundaries and steady-state conditions and use as cross-check.
**Weak answer reveals:** Applying it blindly to a transient deadline day.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 10

**Question:** How should unfinished orders be counted at cutoff?
**Strong answer:** Include eligible unfinished orders as late and drain separately.
**Weak answer reveals:** Dropping them from the denominator.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 11

**Question:** What is input uncertainty?
**Strong answer:** Distinguish uncertain distributions and structural assumptions from run noise.
**Weak answer reveals:** Reporting a narrow Monte Carlo interval as total uncertainty.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
### Question 12

**Question:** How does safety enter the decision?
**Strong answer:** Represent relevant physical limits and require an operational safety review.
**Weak answer reveals:** Treating simulated output as regulatory approval.
**Follow-up:** Ask for an example with a stated unit, boundary, and operational decision.
## 15. Key terms
- **Entity:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Arrival:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Release policy:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Event:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Event calendar:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Simulation clock:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **State:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Resource:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Server:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Queue:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **WIP:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Service time:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Cycle time:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Utilization:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Nominal capacity:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Effective capacity:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Throughput:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Bottleneck:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Cutoff:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **On-time fraction:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Replication:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Random seed:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Warm-up:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Carry-in:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Drain time:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Common random numbers:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Verification:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Validation:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Scenario:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Confidence interval:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Service level:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Priority rule:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Blocking:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Starvation:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Setup:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Rework:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Censoring:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
- **Calibration:** See [definitions](#4-definitions-and-vocabulary) and the [mechanics](#5-the-mechanics-step-by-step).
## 16. Sources

- Jerry Banks, John S. Carson II, Barry L. Nelson, and David M. Nicol, *Discrete-Event System Simulation*, 5th ed., Pearson, 2010. General simulation methodology.
- René B. M. de Koster, Tho Le-Duc, and Kees Jan Roodbergen, “Design and control of warehouse order picking: A literature review,” *European Journal of Operational Research* 182 (2007), 481–501. [DOI: 10.1016/j.ejor.2006.07.009](https://doi.org/10.1016/j.ejor.2006.07.009).
- Kees Jan Roodbergen and Iris F. A. Vis, “A survey of literature on automated storage and retrieval systems,” *European Journal of Operational Research* 194 (2009), 343–362. [DOI: 10.1016/j.ejor.2008.01.038](https://doi.org/10.1016/j.ejor.2008.01.038).
- Peter R. Wurman, Raffaello D'Andrea, and Mick Mountz, “Coordinating Hundreds of Cooperative, Autonomous Vehicles in Warehouses,” *AI Magazine* 29(1) (2008), 9–20. [DOI: 10.1609/aimag.v29i1.2082](https://doi.org/10.1609/aimag.v29i1.2082).
- GS1, [EPCIS and CBV standards](https://www.gs1.org/standards/epcis), official standard information.
- OSHA, [Warehousing](https://www.osha.gov/warehousing), official hazard and safety resources.
- U.S. Occupational Safety and Health Administration, [29 CFR 1910.178](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.178), powered industrial trucks; consult current text and applicability.

## 17. Links

- [Module 13 index](_index.md).
- [Warehouse flow system](01_warehouse-flow-system.md).
- [WMS data model](02_wms-data-model.md).
- [Productivity metrics](03_productivity-metrics.md).
- [Little's Law and queueing](04_littles-law-queueing.md).
- [Labor standards](05_labor-standards.md).
- [Module 13 survey and lab sections](../../../course/13_warehousing_fulfillment_labor.md).
