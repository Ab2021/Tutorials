# Picking Strategies

> Module 13 — Warehousing, Fulfillment, and Labor Analytics
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 13 topics 01–05
> Version 2026.5

## 1. Why this topic exists

A warehouse may have sufficient inventory and paid labor yet still miss a carrier cutoff.
The reason can be the way work is released and combined, not the speed of an individual picker.
A picking strategy determines which orders travel together, which workers visit which areas,
and when picked items become complete orders ready for packing.
These decisions exchange travel savings for waiting, sorting, coordination, and exception risk.
The operations manager must choose a policy that works on the actual order mix,
not copy the method with the highest reported units per hour at another site.

A single-order route is easy to trace but repeatedly visits popular locations.
A batch reduces repeated visits but creates a downstream sort and additional handling.
A cluster preserves order identity in separate cart compartments while sharing travel,
but its cart capacity, verification, and sequence constraints are real costs.
A zone divides responsibility by location; balancing and order consolidation become essential.
A wave is a release schedule, not a synonym for batch or zone picking.
These axes can be combined, provided their handoffs and capacity are modeled explicitly.

The decision affects more than picker productivity.
Packers need complete, correctly identified orders at a usable rate.
Inventory control needs traceable exceptions and accurate remaining quantities.
Customer service needs a truthful completion and dispatch promise.
Finance needs total incremental cost, including sort, overtime, and error recovery.
Workers need a safe, understandable process with adequate training and a fair workload.
The choice should be evaluated at the system boundary from release through ship-ready,
using a common population of orders and a common paid-time denominator.

This lesson develops a practical decision process and a reproducible numerical comparison.
The examples are synthetic teaching data, not measurements of a named warehouse.
Forecast volume, staffing plan, service target, and customer commitment are distinct:
a forecast is uncertain demand; a plan allocates capacity; a target is an internal goal;
a commitment is the actual promise made to a customer or carrier.
All illustrative currency is USD for an assumed 2026 planning period.

## 2. Learning outcomes

After this lesson you can:

1. **Classify** discrete, batch, cluster, zone, and wave designs by work unit and handoff (exercise 1).
2. **Reconstruct** the release-to-ship-ready process and identify its order-level dependency (exercise 2).
3. **Compute** end-to-end paid labor cost per shipped order without omitting sorting (exercise 3).
4. **Calculate** cutoff feasibility and distinguish average capacity from time-dependent availability (exercise 4).
5. **Design** a controlled comparison with balanced order mix and prespecified guardrails (exercise 5).
6. **Diagnose** inventory, tote identity, congestion, and zone imbalance failures (exercise 6).
7. **Evaluate** sensitivity to batch size and volume using explicit assumptions (exercise 7).
8. **Recommend** a rollout, exception path, and rollback trigger (exercise 8).

The exercises require decisions and explanations rather than memorized strategy rankings.

## 3. The core mental model

```text
Customer promise + inventory availability + order pool
                       |
                 release policy
                 /           \
        order grouping      release time
             |                  |
      discrete / batch / cluster; optional zone assignment
             |                  |
        pick and verify -> sort / zone consolidation
                                  |
                         complete-order buffer
                                  |
                          pack -> ship-ready -> carrier
```

A release policy controls how much work enters the floor and when.
A grouping policy controls which orders share a trip or container.
A zone policy assigns physical areas to workers or equipment.
The policies interact but are not mutually exclusive labels.
For instance, a wave may release cluster carts into several zones.
A batch may be released continuously rather than in a scheduled wave.
A completed order is the join of every required line, not the first line picked.
A missing unit blocks that join until an authorized resolution occurs.
Track the state of each order through the join, not just picker task completion.

A causal chain makes the trade-off concrete.
More orders per trip can reduce travel per order when locations overlap.
The same increase can fill carts, increase search within containers, and delay order release.
A larger release can improve grouping but create congestion and a downstream burst.
Moving a bottleneck from pick to sort is not an end-to-end capacity improvement.
Comparing elapsed order time with paid worker time prevents this mistake.
Elapsed time may include waiting while no worker touches the order.
Paid time includes simultaneous labor by multiple workers, so it is not equal to elapsed time.
The method should be chosen for the demand and service window it must actually support.
Measure the whole chain before changing a local rate target.

## 4. Definitions and vocabulary

| Term | Operational meaning | Common misuse or unit |
|---|---|---|
| Order | Customer request with one or more lines | Not interchangeable with a unit |
| Order line | Requested stock keeping unit and quantity on an order | One line can contain many units |
| Unit | Individual saleable item | A case or carton may contain multiple units |
| SKU | Stock keeping unit, the inventory identity | Same description does not guarantee same SKU |
| Pick task | Authorized instruction to remove inventory | Task count is not necessarily line count |
| Discrete pick | One order per route or work assignment | Does not necessarily mean one item per trip |
| Batch pick | Combined demand for multiple orders, followed by allocation | Batch size is orders, not necessarily units |
| Cluster pick | Multi-order trip with separate identified order containers | Mixing into one container makes downstream sort necessary |
| Zone pick | Assignment of inventory areas to different workers | An order can require multiple zones |
| Wave | Planned group of work released together | A release schedule, not a picking route |
| Continuous release | Eligible work released as capacity becomes available | Not automatically congestion-free |
| Sortation | Allocation of commingled units to orders | Must be counted in labor and accuracy |
| Put wall | Identified compartments used to sort items into orders | A finite-capacity resource |
| Consolidation | Joining order components from multiple areas | Requires correct identity and completion state |
| Cart capacity | Safe physical compartment, weight, and volume limits | Order count alone is insufficient |
| Touch | A handling event involving an item or container | Definitions differ between sites |
| Travel | Movement between task locations | Distance and time are different measures |
| Dwell | Elapsed time waiting in a state | Not automatically paid processing time |
| Cycle time | Elapsed time across named start and end events | Always specify the boundary |
| Labor content | Sum of paid worker time attributable to work | Can exceed elapsed time through parallel work |
| Throughput | Completed work per time interval | State orders/hour or other unit |
| WIP | Work in process; released but not completed work | Not synonymous with available stock |
| Cutoff | Latest operational time for a defined departure or process | Different from customer delivery promise |
| Short pick | Requested quantity cannot be picked from assigned stock | Do not silently mark the line complete |
| Replenishment | Movement of reserve stock to a forward pick location | A dependency, not free capacity |
| Exception | Deviation requiring a defined decision or intervention | Must have owner and resolution timestamp |
| Scan verification | Matching a scanned identity against expected task state | A scan does not prove physical quantity by itself |
| WMS | Warehouse management system | Recorded events may lag physical work |
| Guardrail | Prespecified limit that constrains a decision | Not a substitute for a measured outcome |
| Order completion | All required lines resolved under the fulfillment policy | A picked line is not a completed order |
| Service level | Fraction meeting a defined promise or cutoff | Specify denominator and timestamp |
| Availability | Eligible inventory accessible for allocation | On-hand includes potentially blocked stock |
| Cohort | Orders selected under a shared eligibility definition | Avoid comparing different demand mixes |
| Counterfactual | Outcome that would have occurred under another policy | Cannot be observed for the same order twice |
| Paid hour | Compensated labor hour included in the analysis | Excluding indirect work biases comparisons |

## 5. The mechanics, step by step

The following procedure makes the decision repeatable.
At each step record the owner, inputs, decision, and output.
Do not begin by installing a new cart or setting a productivity target.

### Step 1 — Define the decision and the service boundary

Input: eligible orders, customer promises, carrier schedules, and process timestamps.
The owner is the operations manager with transport and customer-service participation.
Choose the unit of comparison: a shipped order, not an arbitrary scan event.
Define start as order release and finish as ship-ready confirmation in this lesson.
Also record order arrival and physical departure to see delays outside that boundary.
Exclude canceled orders from the shipped-order denominator but report their count separately.
Define whether substitutions, partial shipments, and backorders satisfy the promise.
Output: a written cohort definition and a service measure with explicit timestamps.
Branch: if the cutoff differs by carrier, stratify by cutoff instead of pooling blindly.
A late release may make an otherwise efficient strategy infeasible for that order.

### Step 2 — Reconstruct demand and physical constraints

Input: order lines, quantities, dimensions, SKU locations, and eligible inventory snapshots.
Calculate orders per interval, lines per order, units per line, and shared locations.
Retain the distribution and peak intervals; an average masks a concentrated arrival burst.
Record hazardous, temperature-controlled, fragile, and age-restricted attributes if relevant.
Map each order to compatible handling requirements before grouping it.
Measure the cart footprint, safe load, compartment count, and aisle restrictions.
Include replenishment locations, pack stations, sort wall slots, and staging capacity.
Output: eligible demand strata and a physical resource map.
Branch: isolate orders that cannot share equipment or are too urgent to wait for a group.
An item dimension recorded at case level must be converted before testing unit-level fit.

### Step 3 — Establish the current baseline

Input: event history, payroll hours, indirect assignments, and exception records.
Link tasks to order lines and labor shifts without multiplying hours during joins.
Count orders released, completed, shipped, canceled, and unresolved by cohort.
Separate direct pick, sort, consolidation, pack, replenishment, and exception handling.
Report paid hours and volume by shift, zone, and operational role.
Record the late-order fraction and the age of work still incomplete at cutoff.
Reconcile paid hours to payroll or timekeeping totals with documented exclusions.
Output: a baseline cost and service distribution with a named observation period.
Branch: if scan timestamps are missing, validate with observation before estimating travel.
A missing scan is not evidence that an activity took zero time.

### Step 4 — Define feasible policies

Input: demand strata, equipment limits, inventory states, and required service.
For discrete picking, assign one order per route and retain identity throughout.
For batch picking, aggregate quantities by location and define the allocation process.
For cluster picking, assign each order to a unique, verified container compartment.
For zone picking, define zone boundaries, transfers, join rules, and escalation owners.
For waves, specify release times, membership, cutoff priority, and maximum released work.
Continuous release is an alternative timing rule and can use the same picking methods.
Output: executable policy descriptions, not just names of strategies.
Branch: reject a design if required equipment, inventory traceability, or safe handling is absent.
A hybrid is valid only when each order follows a clear assignment and exception path.

### Step 5 — Estimate travel and handling jointly

Input: location sequences, route distance, walking speed assumptions, and handling times.
Estimate distinct location visits rather than simply summing lines across grouped orders.
Shared locations can reduce travel, but the result depends on overlap and routing geometry.
Include cart setup, scanning, searching, depositing, and return or handoff time.
For batches, add the downstream sort and any rehandling and rework.
For clusters, include compartment lookup, cart staging, and capacity-driven split trips.
For zones, include handoffs, waiting at the join, and coordination of missing lines.
For waves, include planned waiting before release and congestion after release.
Output: a labor-content estimate by activity and a separate elapsed-time estimate.
Branch: if cart limits or route restrictions bind, split the work before computing savings.
Use observed route times where possible; distance divided by walking speed is only a proxy.

### Step 6 — Model downstream and deadline feasibility

Input: completion timing by pick group, sort and pack capacity, and cutoff calendars.
A batch does not produce a complete order until its required items are allocated.
A zone order does not complete until all required zone contributions are joined.
Calculate expected backlog at each downstream station at short enough intervals.
Check that cumulative arrivals to pack do not exceed cumulative pack capacity plus buffer.
Where the buffer fills, model blocking upstream rather than assuming unlimited staging.
Check the final order's ship-ready time, not only the average order's pick time.
Output: projected ship-ready distribution, queue peaks, and resource bottlenecks.
Branch: reduce release size or alter order priority when a wave arrives too late to clear.
A capacity average can look adequate while a late burst still misses the cutoff.

### Step 7 — Compare total economics under a common denominator

Input: paid hours by role, wage rates, incremental equipment and error recovery costs.
Compute labor cost as hours by role multiplied by the corresponding loaded hourly rate.
Add only incremental nonlabor costs on the same time horizon and population.
Count sort, consolidation, and replenishment changes rather than assuming they are free.
Use shipped orders as denominator and report unshipped and canceled counts alongside it.
Keep service constraints explicit; do not monetize late shipments without a defensible value.
Output: total incremental cost, cost per order, capacity, and service guardrail results.
Branch: a policy that is cheaper but fails an agreed service constraint is not feasible.
Avoid double counting a rework hour in both direct labor and an error-cost allowance.

### Step 8 — Pilot, measure, and decide

Input: feasible designs, randomization unit, comparable shifts, and safety approval.
Allocate eligible work by comparable time block or matched order stratum.
Train workers in both methods and ensure supervisors can override unsafe assignments.
Record assignment before outcomes and retain overrides with reasons.
Prespecify primary cost measure and safety, accuracy, service, and congestion guardrails.
Observe late or incomplete orders until resolution rather than dropping them at shift end.
Compare within demand strata and report sample size and uncertainty.
Output: pilot decision, limitations, rollout plan, and rollback conditions.
Branch: stop immediately for a safety incident or a defined hard guardrail breach.
A numerical improvement in a small uncontrolled pilot is a signal, not proof of causation.

### Step 9 — Operate the selected policy

Input: daily demand forecast, staffing availability, resource status, and promise calendar.
Select eligible orders and release only work that can be safely processed downstream.
Monitor active work, aged incomplete orders, short picks, and queue capacity during the shift.
Assign a person to investigate each exception and document its disposition.
Close the loop by reconciling picked quantities, packed orders, and remaining inventory.
Refresh labor and capacity assumptions when mix, layout, or equipment changes.
Output: a daily execution record and a periodic policy review.
Branch: if a station fails, switch to a documented fallback without losing order identity.
The operating plan is revised as conditions change; a forecast is not a customer promise.

### Decision cases: translating conditions into policy

The following short cases are decision prompts, not universal rules.
Each assumes safe equipment, valid inventory, and a documented exception owner.
Use measured local evidence before treating any candidate as a recommendation.

#### Case 1: Single-line parcel orders

Condition: many orders request one common SKU.
Candidate: batch by location.
Primary risk: sort capacity and container identity.
Decision logic: a fast sort can offset travel savings.
Evidence to collect: count allocation errors and all sort hours.

#### Case 2: Low-overlap specialty orders

Condition: orders share few locations.
Candidate: discrete or small cluster.
Primary risk: extra cluster handling without route overlap.
Decision logic: combination does not guarantee shorter travel.
Evidence to collect: compare observed location visits per order.

#### Case 3: Urgent same-day order

Condition: one order approaches an earlier cutoff.
Candidate: discrete priority lane.
Primary risk: waiting to form a large group consumes promise slack.
Decision logic: urgency changes the value of travel savings.
Evidence to collect: track release-to-ship-ready time.

#### Case 4: Oversized cartons

Condition: some order lines do not fit the cart.
Candidate: separate compatible and oversized work.
Primary risk: unsafe load or split-order join.
Decision logic: a compartment count is not a cube constraint.
Evidence to collect: measure weight, dimensions, and join delay.

#### Case 5: Fragile products

Condition: several lines require protective handling.
Candidate: compatible discrete or partitioned cluster.
Primary risk: damage from shared handling.
Decision logic: a faster trip can increase downstream loss.
Evidence to collect: audit damage and packaging touches.

#### Case 6: Chilled inventory

Condition: orders mix ambient and temperature-controlled items.
Candidate: zone picking with controlled join.
Primary risk: out-of-temperature dwell at consolidation.
Decision logic: each zone has a different exposure limit.
Evidence to collect: timestamp movement and controlled holding.

#### Case 7: Hazardous segregation

Condition: the order pool contains incompatible materials.
Candidate: separate eligible groups.
Primary risk: prohibited co-location in a tote.
Decision logic: eligibility must precede batching.
Evidence to collect: audit classification and compartment assignment.

#### Case 8: High-value merchandise

Condition: a product requires controlled custody.
Candidate: discrete or secured cluster.
Primary risk: untraceable custody transfers.
Decision logic: travel efficiency cannot replace security controls.
Evidence to collect: log custody changes and investigate variances.

#### Case 9: Aisle congestion

Condition: many workers are released to the same narrow aisle.
Candidate: staggered small releases.
Primary risk: large waves create blocking.
Decision logic: the shortest individual route may delay the system.
Evidence to collect: observe aisle occupancy and queue time.

#### Case 10: Late carrier pickup

Condition: carrier collection is delayed after pack.
Candidate: release by the actual applicable cutoff.
Primary risk: overprioritizing one carrier starves another.
Decision logic: a cutoff change affects relative order priority.
Evidence to collect: recalculate slack using current carrier schedule.

#### Case 11: Late order arrival

Condition: orders arrive throughout the shift.
Candidate: continuous release with bounded grouping.
Primary risk: waiting for a full wave delays early orders.
Decision logic: batch quality competes with aging.
Evidence to collect: measure oldest eligible order age.

#### Case 12: Peak arrival burst

Condition: orders cluster shortly before cutoff.
Candidate: small prioritized waves with extra pack capacity.
Primary risk: downstream capacity is exceeded.
Decision logic: total daily capacity hides interval overload.
Evidence to collect: compare arrivals and completions each interval.

#### Case 13: Small overnight shift

Condition: few workers cover picking and packing.
Candidate: simple discrete or small cluster.
Primary risk: frequent role switching and idle stations.
Decision logic: specialization can fail at low volume.
Evidence to collect: measure paid idle time and changeover.

#### Case 14: Large multi-line order

Condition: one order occupies multiple compartments.
Candidate: discrete route or split with explicit join.
Primary risk: mistaking one cart completion for order completion.
Decision logic: physical limits can split a customer order.
Evidence to collect: track all containers under one order ID.

#### Case 15: Single dominant fast mover

Condition: many orders require the same location.
Candidate: batch or cluster after capacity check.
Primary risk: congestion and repeated replenishment.
Decision logic: popular items create both overlap and competition.
Evidence to collect: record stockouts and location queue time.

#### Case 16: Many slow movers

Condition: the catalog has dispersed low-frequency locations.
Candidate: discrete or carefully grouped cluster.
Primary risk: large detours between distant picks.
Decision logic: grouping by quantity alone can increase distance.
Evidence to collect: compare route distance for actual locations.

#### Case 17: Zone imbalance

Condition: one zone has most of the order lines.
Candidate: zone with flexible staffing.
Primary risk: other zones finish while the busy zone accumulates work.
Decision logic: order completion depends on the slowest required zone.
Evidence to collect: measure work content and age by zone.

#### Case 18: Cross-zone orders

Condition: most orders require multiple areas.
Candidate: zone with explicit consolidation.
Primary risk: incomplete orders occupy limited staging.
Decision logic: parallel work can reduce elapsed time but increase handoffs.
Evidence to collect: measure join dwell and complete-order rate.

#### Case 19: Single-zone orders

Condition: most orders are entirely within one area.
Candidate: zone picking without unnecessary transfer.
Primary risk: sending complete orders through a redundant join.
Decision logic: a join is needed only when components must meet.
Evidence to collect: count avoidable transfers.

#### Case 20: Partial inventory availability

Condition: one order line is unavailable at release.
Candidate: hold or authorize partial fulfillment.
Primary risk: releasing incomplete work fills staging.
Decision logic: the fulfillment policy determines what counts as complete.
Evidence to collect: track shortage age and disposition.

#### Case 21: Forward-pick stockout

Condition: reserve stock exists but pick face is empty.
Candidate: trigger replenishment before or during release.
Primary risk: workers arrive before replenishment.
Decision logic: on-hand inventory is not accessible pick capacity.
Evidence to collect: measure replenishment response and repeated visits.

#### Case 22: Inventory discrepancy

Condition: system quantity exceeds physical quantity.
Candidate: exception-controlled pick with recount.
Primary risk: silent substitution corrupts records.
Decision logic: a scan confirms an identity, not available quantity.
Evidence to collect: compare physical count with inventory ledger.

#### Case 23: Unscannable label

Condition: the item barcode is damaged.
Candidate: verified exception workflow.
Primary risk: manual bypass hides incorrect SKU.
Decision logic: the fallback needs independent identity evidence.
Evidence to collect: count overrides and audit their reasons.

#### Case 24: Cart compartment error

Condition: two orders share an apparent compartment.
Candidate: stop and resolve assignment.
Primary risk: cross-order contamination.
Decision logic: identity must remain unique during the assignment.
Evidence to collect: enforce active mapping uniqueness.

#### Case 25: Put wall saturation

Condition: batch picks arrive faster than sorting clears.
Candidate: throttle batch release.
Primary risk: blocked sort stations stop upstream work.
Decision logic: a finite wall cannot absorb unlimited WIP.
Evidence to collect: measure occupied slots and blocked time.

#### Case 26: Pack station starvation

Condition: pick completion is delayed until a large batch finishes.
Candidate: smaller release groups.
Primary risk: packers wait despite sufficient daily volume.
Decision logic: release cadence affects downstream utilization.
Evidence to collect: measure time to first complete order.

#### Case 27: Pack station overload

Condition: many clusters finish at once.
Candidate: smooth release or add feasible pack capacity.
Primary risk: completed carts queue near pack.
Decision logic: more pick throughput can worsen cutoff performance.
Evidence to collect: measure pack queue age and blocked aisles.

#### Case 28: New worker training

Condition: inexperienced workers use multiple compartments.
Candidate: small cluster with supervised checks.
Primary risk: compartment confusion and error recovery.
Decision logic: initial speed is not steady-state performance.
Evidence to collect: separate training hours and audit accuracy.

#### Case 29: Worker absence

Condition: planned zone staffing is unavailable.
Candidate: reassign trained staff or switch to discrete.
Primary risk: an unstaffed zone blocks order joins.
Decision logic: coverage matters more than average staffing.
Evidence to collect: track unresolved orders by missing zone.

#### Case 30: Equipment outage

Condition: a required cart or scanner is unavailable.
Candidate: use a documented fallback.
Primary risk: duplicate or lost assignments on reassignment.
Decision logic: fallback must preserve work state.
Evidence to collect: reconcile open tasks before restart.

#### Case 31: High return from errors

Condition: wrong-item claims rise after batching.
Candidate: reduce batch size and investigate allocation.
Primary risk: saved travel is lost to recovery and customer harm.
Decision logic: accuracy is an end-to-end outcome.
Evidence to collect: link claim to order and allocation events.

#### Case 32: Promotion changes mix

Condition: a campaign changes product affinity.
Candidate: re-estimate grouping with current demand.
Primary risk: historical overlap no longer predicts visits.
Decision logic: policy parameters must follow mix.
Evidence to collect: compare predicted and realized location overlap.

#### Case 33: Seasonal temporary layout

Condition: staging encroaches on a travel aisle.
Candidate: re-route and cap released work.
Primary risk: old travel estimates assume a blocked route.
Decision logic: physical change invalidates the model.
Evidence to collect: observe route times and safety conditions.

#### Case 34: Multiple carriers

Condition: orders have different departure deadlines.
Candidate: prioritize by carrier-specific slack.
Primary risk: a single wave cutoff delays earlier departures.
Decision logic: service constraints belong at order level.
Evidence to collect: monitor lateness by carrier cohort.

#### Case 35: Partial-shipment policy

Condition: customer permits some lines to ship separately.
Candidate: resolve split shipments explicitly.
Primary risk: counting two shipments as two orders distorts economics.
Decision logic: order, shipment, and parcel are different units.
Evidence to collect: report both order and shipment denominators.

#### Case 36: Mixed quantities per line

Condition: some lines require several units.
Candidate: choose groups using unit volume and weight.
Primary risk: counting lines understates cart load.
Decision logic: task count and physical units differ.
Evidence to collect: validate capacity against quantities.

#### Case 37: Reverse logistics contamination

Condition: returns share a staging area with outbound work.
Candidate: physically segregate inventory states.
Primary risk: uninspected stock enters an outbound cluster.
Decision logic: available-to-pick status must be controlled.
Evidence to collect: audit state transitions and location access.

#### Case 38: Priority override

Condition: a supervisor inserts an urgent order into a wave.
Candidate: record override and re-evaluate feasibility.
Primary risk: unrecorded changes bias pilot results.
Decision logic: assignment and actual treatment can differ.
Evidence to collect: retain original assignment and override reason.

#### Case 39: Carrier cutoff moves earlier

Condition: a departure schedule changes midshift.
Candidate: recalculate priorities and communicate.
Primary risk: old wave timing no longer supports promise.
Decision logic: a schedule is an operational input, not a constant.
Evidence to collect: monitor slack and issue timestamp.

#### Case 40: Cart shortage

Condition: the site has fewer carts than planned groups.
Candidate: queue assignments or use another eligible method.
Primary risk: model assumes unlimited simultaneous carts.
Decision logic: cart availability controls work in process.
Evidence to collect: measure cart cycle time and waiting.


### Quantitative decision checks

These independent checks illustrate arithmetic to adapt to observed site data.

#### Shared locations
Four orders each have 6 location visits when picked separately.
Separate routes require 4 × 6 = 24 visits.
If a feasible combined route visits 15 locations, it avoids 9 visits.
Visit savings do not establish time savings without route and handling times.

#### Group setup
A cluster saves 12 minutes of travel but adds 5 minutes of setup.
Its net modeled saving is 12 − 5 = 7 minutes per group.
At USD 24 per hour, that is USD 2.80 of labor content per group.
Include any changed sort, correction, and paid idle time before a financial decision.

#### Cart compartments
A cart has 8 usable compartments and one order needs two.
The remaining 6 compartments can hold at most 6 one-compartment orders.
The theoretical maximum is therefore 7 orders, not 8 orders.
Weight, volume, and incompatibility may lower the feasible count further.

#### Weight limit
A cart has a safe payload limit of 120 kg and already carries 85 kg.
The remaining payload is 120 − 85 = 35 kg.
An additional 38 kg order is not eligible for that loaded cart.
Include the actual cart and equipment specifications in the safety assessment.

#### Volume limit
A cart has 0.8 cubic meters of usable space and 0.5 is occupied.
Its remaining nominal space is 0.3 cubic meters.
An order needing 0.35 cubic meters cannot fit under this simple capacity test.
Packing geometry can reduce usable capacity even when total volume fits.

#### Zone join
Zone A finishes at minute 22 and zone B at minute 31 after release.
If both contributions are required, the earliest join is minute 31.
Zone A waits at least 31 − 22 = 9 minutes for zone B.
Transfer and validation time make the actual join later.

#### Labor cost
Ten hours of picking at USD 24/hour cost USD 240.
Three hours of sorting at USD 20/hour cost USD 60.
The combined labor cost is USD 300 for the measured cohort.
Using only the picker rate would omit USD 60 of downstream work.

#### Error recovery
A process has 4 observed errors and each takes 15 minutes to correct.
Observed correction content is 4 × 15 = 60 minutes.
At USD 24/hour this correction labor costs USD 24.
Customer loss or later claims are not included in this estimate.

#### Wave waiting
An eligible order arrives one minute after an hourly wave release.
If the next release is exactly 60 minutes later, it waits 59 minutes.
That wait occurs before any picking or packing starts.
An urgent exception rule can reduce the wait but must respect downstream capacity.

#### Cutoff slack
A customer order must be ship-ready at 16:00.
It is released at 14:30, giving 90 minutes of available time.
If measured processing and waiting take 100 minutes, it misses by 10 minutes.
The relevant service comparison is against its own cutoff, not a pooled average.

#### Pack backlog
A station starts an interval with 20 waiting orders.
During the interval 55 arrive and 45 are completed.
End-of-interval backlog is 20 + 55 − 45 = 30 orders.
This arithmetic assumes no cancellations or other departures in the interval.

#### Queue capacity
A staging buffer holds 40 carts and currently contains 34.
It has 6 available cart positions.
Releasing 10 carts without departures would require 4 additional positions.
A full buffer can block upstream work and create a safety hazard.

#### Replenishment
A pick face contains 18 units and released demand needs 25.
The visible shortage is 25 − 18 = 7 units.
At least 7 additional eligible units must arrive for all demand to be picked.
Reserve availability and replenishment lead time must also be checked.

#### Completion rate
A cohort contains 300 released orders and 285 are ship-ready by cutoff.
The on-time completion fraction is 285 / 300 = 95%.
The remaining 15 orders require disposition and follow-up.
Do not discard those 15 from the denominator merely because they are incomplete.

#### Split shipments
A cohort has 100 orders and generates 108 shipments.
If total fulfillment cost is USD 270, cost per order is USD 2.70.
Cost per shipment is USD 270 / 108 = USD 2.50.
Both metrics can be valid but answer different operational questions.

#### Wage mix
Six pick hours cost USD 24/hour and four sort hours cost USD 20/hour.
Total cost is 6 × 24 + 4 × 20 = USD 224.
Dividing by 10 hours gives USD 22.40 per paid hour for this mix.
Do not apply that blended rate to a different role mix without recalculation.

#### Station availability
Three packers are scheduled for 4 hours each.
A station outage removes 2 productive worker-hours with no alternative station.
Available productive station time is at most 12 − 2 = 10 worker-hours.
Paid payroll time can remain 12 hours even when productive time falls.

#### Pilot mix
Treatment has 80 single-line and 20 multi-line orders.
Control has 20 single-line and 80 multi-line orders.
An unadjusted average combines strategy differences with order complexity.
Compare within line-count strata or rebalance the assignment before drawing a conclusion.

#### Additional equipment
A cart costs USD 1,200 and is expected to serve 600 comparable windows.
A simple straight-line allocation is USD 2 per window before financing or maintenance.
With 240 orders per window that is USD 2 / 240 = USD 0.0083 per order.
Use actual accounting policy and utilization; this is not a complete investment appraisal.

#### Remaining promise slack
An order has 75 minutes until its ship-ready cutoff.
Expected remaining pick, join, and pack time totals 55 minutes.
Its modeled slack is 75 − 55 = 20 minutes.
Variability and congestion can consume that slack, so it is not a guarantee.


## 6. Worked example

### Decision and cohort

A synthetic distribution center must prepare 240 eligible orders in one four-hour window.
All orders are available and inventory-eligible at release; every order ships if completed.
Each order has two lines, so the cohort contains 480 lines.
The orders have one unit per line, for 480 units in total.
Assume all items are compatible with the chosen carts and there are no shortages.
The internal target is to have all 240 orders ship-ready by the end of the window.
This target is not a customer delivery commitment and does not include transport time.
Management compares discrete routes with four-order cluster carts.
Both alternatives use the same order cohort and the same four-hour window.

### Estimated paid work

The estimates below come from an illustrative engineering study, not observed evidence.
The labor rate is USD 24 per paid hour for every included role.
The common pack work is included even though it does not affect the difference.

| Activity | Discrete hours | Cluster hours | Interpretation |
|---|---:|---:|---|
| Travel and picking | 18.0 | 12.0 | Shared routes reduce repeated travel |
| Cart setup and verification | 0.0 | 2.0 | Cluster requires compartment identity checks |
| Consolidation and corrections | 1.0 | 1.5 | Includes expected routine rehandling |
| Packing | 8.0 | 8.0 | Same order population and assumption |
| Total paid work | 27.0 | 23.5 | Sum of all listed activities |

Discrete cost is 27.0 hours × USD 24/hour = USD 648 per window.
Cluster cost is 23.5 hours × USD 24/hour = USD 564 per window.
The difference is USD 84 per window, or 3.5 paid hours per window.
Per shipped order, the estimates are USD 648 / 240 = USD 2.70 for discrete
and USD 564 / 240 = USD 2.35 for cluster.
The saving is USD 0.35 per shipped order, approximately 12.96% of baseline cost.
This is not a forecast of actual savings until the assumptions are tested.
There is no separate batch sorting charge here because each cluster compartment retains identity.
If items are commingled, add a sorting activity instead of calling that work free.

### Staffing feasibility

Suppose five pick-side workers are available for four hours, giving 20 paid hours.
Suppose two packers are available for four hours, giving 8 paid hours.
Discrete pick-side demand is 18.0 + 1.0 = 19.0 hours.
Cluster pick-side demand is 12.0 + 2.0 + 1.5 = 15.5 hours.
Both fit the aggregate pick-side limit; packing exactly equals its 8-hour limit.
The packing capacity implied by the assumed workload is 240 / 8 = 30 orders per paid hour.
Two packers therefore have an aggregate nominal rate of 60 orders per elapsed hour.
At full utilization, a 240-order pack workload consumes the entire four-hour window.
Any initial wait before the first order reaches pack makes the all-on-time target infeasible
unless pack capacity, available time, or the actual packing content changes.
Thus aggregate staffing feasibility alone does not establish cutoff feasibility.
Neither policy can be certified from these totals without an arrival schedule to pack.

### An explicit timing check

Assume, for a separate sensitivity case, all 240 orders reach pack at hour 1.0.
With two packers at a combined 60 orders/hour and no interruption,
the finish time is hour 1.0 + 240 / 60 = hour 5.0.
That misses the four-hour internal target by one hour.
Three packers at the same individual rate would provide 90 orders/hour.
Their idealized finish time would be hour 1.0 + 240 / 90 = hour 3.667.
That is feasible on this simplified timing assumption, subject to physical station capacity.
The third packer changes the staffing plan and adds cost if the extra hours are paid.
If that worker is paid for all four hours at USD 24/hour, add USD 96 to either plan.
Do not add this cost to the original 8-hour packing table while retaining its two-worker premise.

### Decision and sensitivity

Cluster has an estimated labor advantage, but service feasibility remains unproven.
Pilot cluster with measured arrival-to-pack timestamps and a staffed pack plan.
Keep discrete handling for orders too large, urgent, or incompatible with a cluster cart.
If cluster correction work rises from 1.5 to 5.0 hours, total cluster work becomes 27.0 hours.
At that point its estimated labor cost equals discrete cost before equipment costs.
This is a break-even sensitivity, not an asserted error rate.
Include training, cart depreciation, and any changed replenishment when they are incremental.
Quantify uncertainty in overlap and rework rather than presenting 3.5 hours as guaranteed.
A service failure, unsafe cart load, or untraceable order overrides the apparent labor saving.

## 7. Code, query, or calculation

The following Python 3 script uses only the standard library.
Save it as `picking_compare.py` outside the course repository and run `python picking_compare.py`.
It reproduces the work-content and simple deadline arithmetic in section 6.
Inputs are totals for one 240-order cohort, not historical observations.
It assumes every listed hour is paid at the same rate and every order ships.
Packing availability is modeled independently from pick-side availability.
The burst calculation assumes all orders arrive at pack at once and packers start immediately.
It does not simulate individual orders, congestion, breaks, or random failures.

```python
orders = 240
window_hours = 4.0
rate_usd_per_hour = 24.0
pick_workers = 5
pack_workers = 2

policies = {
    "discrete": {
        "pick": 18.0,
        "setup": 0.0,
        "correction": 1.0,
        "pack": 8.0,
    },
    "cluster": {
        "pick": 12.0,
        "setup": 2.0,
        "correction": 1.5,
        "pack": 8.0,
    },
}

for name, hours in policies.items():
    pick_side = hours["pick"] + hours["setup"] + hours["correction"]
    total = pick_side + hours["pack"]
    cost = total * rate_usd_per_hour
    print(f"{name}: hours={total:.1f}, cost=${cost:.2f}, "
          f"cost/order=${cost / orders:.2f}, "
          f"pick_capacity_ok={pick_side <= pick_workers * window_hours}, "
          f"pack_capacity_ok={hours['pack'] <= pack_workers * window_hours}")

baseline = sum(policies["discrete"].values())
cluster = sum(policies["cluster"].values())
saving = (baseline - cluster) * rate_usd_per_hour
print(f"saving=${saving:.2f}, reduction={(baseline - cluster) / baseline:.2%}")

arrival_hour = 1.0
orders_per_paid_pack_hour = orders / policies["cluster"]["pack"]
for workers in (2, 3):
    finish = arrival_hour + orders / (workers * orders_per_paid_pack_hour)
    print(f"packers={workers}: finish_hour={finish:.3f}, "
          f"within_window={finish <= window_hours}")
```

Expected output:

```text
discrete: hours=27.0, cost=$648.00, cost/order=$2.70, pick_capacity_ok=True, pack_capacity_ok=True
cluster: hours=23.5, cost=$564.00, cost/order=$2.35, pick_capacity_ok=True, pack_capacity_ok=True
saving=$84.00, reduction=12.96%
packers=2: finish_hour=5.000, within_window=False
packers=3: finish_hour=3.667, within_window=True
```

The two `capacity_ok` flags compare total work with total available paid hours.
They deliberately do not claim that orders arrive early enough to use those hours.
The second calculation exposes this difference with a specified arrival time.
For production, replace the fixed arrival assumption with observed or simulated arrival events.
Validate that every hour is allocated once and that each order's status is carried to resolution.
Test mixed wages and paid idle time when the financial decision requires them.

## 8. Data, standards, and regulatory context

The WMS event log should distinguish task creation, release, acceptance, pick confirmation,
exception, container transfer, consolidation, pack completion, and shipment confirmation.
Use immutable event IDs, event time and ingestion time, site time zone, and source system.
Retain order line ID, SKU, requested and confirmed quantity, and unit of measure.
Record location, inventory status, worker or device assignment, and destination container.
A transfer should identify both sending and receiving containers or locations.
For a cluster, retain the mapping of order to compartment for each assignment period.
For a batch, retain the allocation from aggregate picked quantity back to order lines.
For a zone, record which required zones have completed and which remain outstanding.
A wave ID describes release membership and must not replace the order-line key.

Make corrections append-only where possible, with references to the event corrected.
Deduplicate retries using a stable event identifier rather than dropping equal timestamps.
A task can be reassigned; summing task durations across owners may double count time.
A short pick is an exception until resolved by a documented substitution, cancellation,
replenishment, or backorder decision under the applicable business policy.
An order created before midnight and shipped after midnight needs a consistent cohort rule.
Inventory snapshots and event times must be aligned before comparing stock with picks.
Log missing identifiers and late-arriving events rather than silently excluding them.

GS1 identification standards provide a common approach to trade-item and logistics identities.
GS1 barcodes can encode identifiers, but a barcode alone does not establish correct quantity,
condition, ownership, stock availability, or order completion.
The WMS must define the operational meaning of each scan and validate task context.
Whether a particular GS1 identifier is required depends on trading agreements and applicable rules;
the existence of a GS1 standard is not itself a universal legal mandate.
Consult the current GS1 General Specifications and applicable partner implementation guide.

In the United States, OSHA's general industry powered industrial truck standard,
29 CFR 1910.178, applies where its scope and conditions are met.
It includes requirements concerning operator training and safe operation of covered equipment.
The standard does not prescribe discrete, cluster, batch, wave, or zone picking.
Pedestrian carts are not automatically powered industrial trucks; assess the actual equipment.
Other occupational-safety requirements may apply to the facility and jurisdiction.
Evaluate lifting, reach, aisle congestion, traffic separation, and load stability locally.
This educational discussion is not legal advice; consult current official rules and site experts.
Avoid turning individual scan rates into a safety-blind incentive or disciplinary proxy.
Protect employee data with appropriate access, retention, and local privacy review.
Record the applicable regulation version and review date in the site compliance register.

Data quality checks should compare order totals with line totals and task confirmations.
Reconcile outbound inventory decrement to shipped and exception dispositions.
Check that a compartment is not assigned to two active orders simultaneously.
Check that a zone join does not complete before all required zone contributions resolve.
Check that work released after cutoff is not credited to an earlier wave without explanation.
A missing event requires investigation; it is not automatically a zero-duration event.

## 9. Industry practice and case evidence

Peer-reviewed warehouse research treats order picking as a system of interdependent decisions.
De Koster, Le-Duc, and Roodbergen review layout, storage, routing, batching, and zoning.
Their review supports considering these decisions jointly rather than declaring one universal rule.
It does not establish the labor savings of the synthetic case in this lesson.
Petersen and Aase compare picking, storage, and routing policies in a distribution setting.
Their study illustrates the importance of interactions and assumptions in such comparisons.
It does not justify transferring a reported performance result to a different building.
See the original papers in section 16 for method and scope.

A documented operational case is Ocado's customer fulfillment center model.
Ocado Group describes automated grids and robots that retrieve storage bins for stations.
This is evidence of a goods-to-person operational design, not evidence that a particular
manual batch size or the economics in section 6 apply at Ocado.
[INFERENCE] Moving inventory to a station changes the travel component of pick labor;
it does not eliminate order identity, workstation capacity, or replenishment decisions.
Vendor and company descriptions can establish the existence of a design but may emphasize benefits.
Independent operating cost, full utilization, and customer-service impact require separate evidence.

A second public example is Amazon's description of its fulfillment-center processes,
which identifies picking as a stage before packing and shipping.
The description supports treating pick as part of a broader fulfillment flow.
It does not publish a controlled comparison of the five strategy choices in this lesson.
[INFERENCE] A local improvement in one stage need not increase site throughput when another
stage constrains the flow; verify this for the actual site using station-level measurements.
Neither example supplies a comparable labor-cost estimate for the synthetic cohort.

Use external cases as hypotheses about feasible designs and implementation risks.
Before adoption, collect local order mix, physical layout, error rates, and resource calendars.
Compare expected improvements against the cost of training, equipment, and transition.
A pilot should measure both the intended outcome and downstream side effects.
When citing a public case, distinguish the company claim, the researcher result, and your inference.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong compartment | Frequent scan overrides | Order-container mapping lost | Wrong customer receives item | Unique mapping and scan at deposit |
| Batch not sorted | Pick complete but orders open | Commingled units await allocation | False completion claim | Track allocation and order join |
| Zone imbalance | One zone's aged work rises | Uneven work assignment | Orders wait on slow zone | Rebalance trained capacity |
| Wave too large | Aisle occupancy rises | Simultaneous releases exceed space | Congestion and safety exposure | Cap WIP and stagger releases |
| Pack starvation | Packers idle early | Large groups finish late | Lost packing time | Release smaller groups earlier |
| Pack overload | Queue ages rise | Pick completion burst exceeds pack rate | Cutoff misses | Smooth release and plan pack capacity |
| Short pick hidden | Unresolved quantities disappear | Exception marked complete | Inventory and service error | Enforce explicit disposition |
| Incompatible items grouped | Handling override increases | Eligibility rules are incomplete | Damage or safety risk | Filter before grouping |
| Cart overloaded | Cart weight near limit | Orders grouped by count alone | Injury or damaged goods | Enforce weight and cube limits |
| Missing replenishment | Repeat visits to empty face | Reserve movement not synchronized | Travel and order delay | Trigger and monitor replenishment |
| Misleading productivity | Units/hour rises while late orders rise | Local metric ignores downstream work | Lower system service | Measure complete orders and total hours |
| Duplicate event | Quantity exceeds requested units | Scan retries posted twice | Inventory misstatement | Idempotent event handling |
| Wrong cohort | One method gets easier orders | Allocation is not comparable | Biased benefit estimate | Randomize or stratify and retain assignment |
| Unsafe pace | Near misses increase | Incentive rewards speed alone | Injury and disruption | Safety guardrail and stop authority |
| False availability | On-hand includes blocked stock | Status not considered | Unfillable tasks released | Allocate eligible inventory only |
| Missing cutoff | Orders wait in normal wave | Priority ignores promise slack | Late dispatch | Order-level deadline monitoring |
| Untracked transfer | Tote moves without receipt | Zone custody chain breaks | Lost items and rework | Scan both handoff endpoints |
| Suppressed cancellations | Canceled orders excluded without disclosure | Difficult cases disappear from comparison | Cost and service appear better | Report disposition for all released orders |

A warning is not itself a root cause.
Inspect the physical process and event trail before selecting a corrective action.
Assign each unresolved problem to an owner with an escalation deadline.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Eligible inventory is available | Tasks can be executed | Blocked or missing stock | Compare allocation to physical checks |
| Order mix is comparable | Cost comparison is meaningful | Different lines or urgency | Stratify orders and compare distributions |
| Loaded wage is constant | Simple cost arithmetic | Shift or role wage differences | Replace with role-specific payroll rates |
| No paid idle time outside modeled hours | Work content approximates payroll | Fixed shifts with unused time | Reconcile model to scheduled paid hours |
| All orders ship | Per-order denominator is 240 | Cancellations or backorders | Follow each released order to disposition |
| Cluster compartments are unique | Identity is retained | Duplicate assignment or override | Audit active mappings and scan trails |
| Cart limits are respected | Group is physically feasible | Heavy or bulky units | Check item dimensions, mass, and load |
| Pack productivity is constant | Timing calculation is simple | Fatigue, mix, station downtime | Measure intervals and confidence range |
| All orders arrive together in timing case | Closed-form finish is computable | Staggered arrivals | Use event-level queue reconstruction |
| Travel reduction is real | Cluster estimate has a basis | Low SKU overlap or congestion | Observe route times on matched work |
| Correction work is included | Labor comparison is complete | Returns or claims appear later | Follow cohort through claim window |
| Service boundary is stable | Late fraction is comparable | Different cutoff or timestamp meaning | Version metric definition and audit events |
| Worker safety is protected | Policy can be operated | Aisle obstruction or unsafe lifts | Observe work and review incidents |
| Packing stations are available | Extra staff can add capacity | Stations, supplies, or floor space bind | Verify equipment and resource calendar |

These are assumptions about the example, not assertions about a real site.
Use sensitivity ranges or a simulation when variability determines the decision.
The deterministic arithmetic cannot estimate a tail probability or a confidence interval.

## 12. Exercises

Complete each exercise before reading section 13.
State units, time boundaries, and assumptions in each calculation.

### Exercise 1 — Classify the strategy

A worker collects combined quantities for ten orders into one tote.
Another worker allocates the units into ten labeled customer compartments.
A supervisor releases these tasks every hour.
Deliverable: identify grouping method, release method, and downstream work.
Acceptance: distinguish all three and explain why this is not cluster picking.
Time: 8 minutes.

### Exercise 2 — Draw the dependency

Two zones contribute to an order, and one zone reports a short pick.
Deliverable: draw the states from release to ship-ready and identify the join condition.
Acceptance: show both zone contributions, an exception owner, and an authorized resolution;
explain why a completed first zone does not establish order completion.
Time: 10 minutes.

### Exercise 3 — Compare labor cost

For 200 shipped orders, discrete uses 20 pick hours, 2 correction hours, and 6 pack hours.
Cluster uses 14 pick hours, 3 setup hours, 2 correction hours, and 6 pack hours.
Every hour costs USD 25; ignore equipment and assume identical service.
Deliverable: total hours, total labor cost, cost per order, and labor saving.
Acceptance: include every stated activity and use 200 shipped orders for both methods.
Time: 12 minutes.

### Exercise 4 — Test the deadline

A pack station receives 180 ready orders at hour 1.5.
Two packers together complete 60 orders per elapsed hour without interruption.
The ship-ready cutoff is hour 4.0.
Deliverable: calculate the finishing time and the shortfall at cutoff.
Acceptance: distinguish work remaining from total daily nominal capacity.
Time: 10 minutes.

### Exercise 5 — Design a pilot

A site wants to compare four-order cluster and discrete picking over two weeks.
Its morning orders have fewer lines than its evening orders.
Deliverable: a treatment allocation, primary metric, three guardrails, and exception handling plan.
Acceptance: prevent time-of-day mix from masquerading as a treatment effect;
include total labor, order-level completion, and an auditable override record.
Time: 20 minutes.

### Exercise 6 — Diagnose a red flag

A batch pilot raises picker units per hour by 20%, but sort hours double and lateness rises.
Deliverable: explain two plausible mechanisms and list four measurements to collect.
Acceptance: address sorting and time-dependent pack arrivals without asserting a proven cause.
Time: 12 minutes.

### Exercise 7 — Find the break-even correction time

Use the section 6 assumptions and hold all cluster activities except correction fixed.
Deliverable: calculate the correction hours at which cluster equals discrete total paid hours.
Acceptance: show the equation and identify one excluded cost that could change break-even.
Time: 8 minutes.

### Exercise 8 — Recommend a rollout

A site has a reliable cluster identity scan, limited cart capacity, and variable cutoff times.
Deliverable: write a one-page recommendation with eligibility, release, measures, and fallback.
Acceptance: cover oversized and urgent orders, safety, downstream capacity,
a threshold for pausing the pilot, and a named decision owner.
Time: 25 minutes.

## 13. Self-check answers

### Answer 1

The grouping method is batch picking because units for several orders are commingled.
The hourly schedule is a wave release rule.
The second worker performs sortation or allocation to order compartments.
It is not cluster picking as described because order identity is not preserved at the pick deposit.
A batch may be released continuously, and a wave may contain cluster work.

### Answer 2

Release creates work in zone A and zone B.
Zone A can complete while zone B records a shortage and escalates it.
The owner investigates and authorizes replenishment, a substitution, or another allowed disposition.
The join checks all required order lines and permitted dispositions before ship-ready processing.
A short-pick scan alone does not prove that the customer's complete order is ready.

### Answer 3

Discrete total is 20 + 2 + 6 = 28 paid hours.
Cluster total is 14 + 3 + 2 + 6 = 25 paid hours.
Discrete cost is 28 × USD 25 = USD 700, or USD 3.50 per shipped order.
Cluster cost is 25 × USD 25 = USD 625, or USD 3.125 per shipped order.
Labor saving is USD 75 for the cohort, or USD 0.375 per order.
Rounded to cents, cluster cost is USD 3.13 per order and saving is USD 0.38 per order.
Round only after calculating; rounded per-order figures may not add exactly.

### Answer 4

Processing requires 180 / 60 = 3 elapsed hours after the burst arrives.
The finish is hour 1.5 + 3 = hour 4.5, half an hour after cutoff.
Only 2.5 hours are available between arrival and cutoff.
At 60 orders/hour, 150 orders finish by cutoff and 30 remain.
The conclusion assumes no prior work, interruptions, or other constraints.

### Answer 5 rubric

A strong design randomizes or alternates comparable blocks within morning and evening strata.
It records assignment before outcomes and reports overrides and incomplete orders.
Its primary metric is total paid cost per shipped order over a defined cohort and horizon.
Guardrails include order accuracy, cutoff performance, and worker safety.
The team measures pack and correction work and follows all orders to disposition.
An uncontrolled comparison of morning cluster against evening discrete fails the mix criterion.

### Answer 6

One possible mechanism is that commingled items create more sorting and verification work.
Another is that large pick batches arrive at pack too late or in bursts exceeding capacity.
Neither mechanism is proved by the stated observation.
Measure sort paid hours, order arrival timestamps to pack, pack completions by interval,
and order-level lateness; also examine error recovery and released-order mix.

### Answer 7

Discrete has 27.0 total paid hours in section 6.
Cluster excluding corrections has 12.0 + 2.0 + 8.0 = 22.0 paid hours.
Set 22.0 + correction hours = 27.0; break-even correction is 5.0 paid hours.
Incremental cart cost, a changed wage rate, or a service penalty changes the economic threshold.
The equation assumes the same shipped-order count and no other changes.

### Answer 8 rubric

A strong recommendation defines cart fit, compatible inventory, and order-compartment identity.
It directs urgent or oversized orders to a feasible alternative without losing traceability.
It checks pack capacity and tracks completion against each applicable cutoff.
It identifies an operations owner, a safety stop, and a measured pause threshold agreed in advance.
It specifies how outstanding work transfers to the fallback and how results are reconciled.
A recommendation based only on picker units per hour does not meet the acceptance criteria.

## 14. Interview questions

### 1. How does batch differ from cluster picking?

A strong answer distinguishes commingled units plus sorting from separate order compartments.
A weak answer assumes any multi-order cart is cluster regardless of identity handling.

### 2. Can a wave contain discrete orders?

A strong answer says yes: wave describes release timing, not the route or grouping method.
A weak answer treats wave and discrete as mutually exclusive strategies.

### 3. Why might a high picker rate worsen service?

A strong answer discusses release bursts, downstream queues, and order completion dependencies.
A weak answer assumes every local throughput gain increases shipped orders.

### 4. What is the cost denominator?

A strong answer defines shipped orders, paid hours, cohort, and excluded or unresolved work.
A weak answer compares units per direct pick hour across different order mixes.

### 5. What makes a zone order complete?

A strong answer requires all required zone contributions and authorized exception resolution.
A weak answer treats completion in the first zone as completion of the customer order.

### 6. How do you test a strategy fairly?

A strong answer uses comparable randomized or matched blocks, stratifies mix, and retains overrides.
A weak answer attributes a before-after difference to strategy despite a changed promotion.

### 7. How can adequate total hours still miss a cutoff?

A strong answer explains that capacity must be available after work arrives and before cutoff.
A weak answer divides daily volume by daily hours and ignores the arrival schedule.

### 8. What data joins protect cluster identity?

A strong answer links order, line, task, container, compartment, and assignment intervals.
A weak answer relies on a cart identifier without recording which order occupies each compartment.

### 9. What changes your answer for a high-velocity SKU?

A strong answer considers overlap, face replenishment, congestion, and grouping capacity together.
A weak answer assumes popularity guarantees that the largest batch is optimal.

### 10. When should a pilot stop?

A strong answer names a prespecified safety or service trigger, owner, and fallback procedure.
A weak answer continues because the productivity number is favorable despite a safety breach.

## 15. Key terms

The definitions are in [section 4](#4-definitions-and-vocabulary); decision use is in
[section 5](#5-the-mechanics-step-by-step).

- **Order:** customer request, the service and economic cohort unit.
- **Order line:** requested SKU and quantity within an order.
- **Unit:** individual saleable item.
- **SKU:** stock keeping unit identifying inventory.
- **Discrete picking:** one order per route or assignment.
- **Batch picking:** combine orders and subsequently allocate their items.
- **Cluster picking:** combine trips while retaining separate order compartments.
- **Zone picking:** divide physical areas among workers or equipment.
- **Wave:** scheduled release of a group of work.
- **Continuous release:** release work as eligibility and capacity permit.
- **Sortation:** allocate commingled units to orders.
- **Put wall:** location with identified order-sort compartments.
- **Consolidation:** join components of an order.
- **Cart capacity:** safe physical carrying and compartment limits.
- **Touch:** a defined item or container handling event.
- **Travel:** movement between work locations.
- **Dwell:** elapsed waiting time in a state.
- **Cycle time:** elapsed time between named process events.
- **Labor content:** summed worker time attributable to activities.
- **Throughput:** completed work per unit of time.
- **WIP:** released work not yet complete.
- **Cutoff:** latest specified time for a process or departure.
- **Short pick:** inability to pick requested quantity from the assigned stock.
- **Replenishment:** movement into a forward pick location.
- **Exception:** deviation requiring explicit disposition.
- **Scan verification:** comparison of a scanned identity with task expectations.
- **WMS:** warehouse management system.
- **Guardrail:** limit that constrains an operational decision.
- **Cohort:** population selected using a common rule.
- **Counterfactual:** outcome under an alternative unobserved choice.
- **Paid hour:** compensated labor hour included in the comparison.

## 16. Sources

- De Koster, R., Le-Duc, T., and Roodbergen, K. J. (2007).
  “Design and control of warehouse order picking: A literature review.”
  *European Journal of Operational Research*, 182(2), 481–501.
  [https://doi.org/10.1016/j.ejor.2006.07.009](https://doi.org/10.1016/j.ejor.2006.07.009).
  Peer-reviewed overview; use for interacting design decisions, not a site-specific forecast.
- Petersen, C. G., and Aase, G. (2004).
  “A comparison of picking, storage, and routing policies in manual order picking.”
  *International Journal of Production Economics*, 92(1), 11–19.
  [https://doi.org/10.1016/j.ijpe.2003.09.006](https://doi.org/10.1016/j.ijpe.2003.09.006).
  Peer-reviewed comparison; its setting and assumptions limit transferability.
- GS1, [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
  Standards guidance for identification and barcode use; check the current version and partner rules.
- U.S. Occupational Safety and Health Administration,
  [29 CFR 1910.178, Powered industrial trucks](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.178).
  Regulatory text; applicability depends on the equipment and jurisdiction.
- Ocado Group, [Ocado Smart Platform](https://www.ocadogroup.com/solutions/ocado-smart-platform).
  Company description of its fulfillment technology; company claims are not independent evaluations.
- Amazon, [How Amazon fulfillment centers work](https://www.aboutamazon.com/news/operations/how-amazon-fulfillment-centers-work).
  Company description of fulfillment operations; not a controlled strategy comparison.

Regulatory discussion is educational and should be checked against current official sources.
Synthetic example inputs and results in this lesson are not attributed to these sources.

## 17. Links

- [Module 13 index](_index.md).
- [Warehouse as a flow system](01_warehouse-flow-system.md).
- [WMS process and data model](02_wms-data-model.md).
- [Productivity metrics](03_productivity-metrics.md).
- [Little's Law and queueing](04_littles-law-queueing.md).
- [Labor standards](05_labor-standards.md).
- [Module 13 survey and its lab activities](../../../course/13_warehousing_fulfillment_labor.md).
- [Labs and electives index](../../05_labs_electives/_index.md).
