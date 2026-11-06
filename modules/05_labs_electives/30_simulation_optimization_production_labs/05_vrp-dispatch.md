# Vehicle Routing and Dispatch Replay

> Module 30 — Simulation, Optimization, and Real-Time Production Labs
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: basic Python 3, permutations, time arithmetic, CSV concepts, and the [environment lab](01_environment-repo.md); familiarity with optimization is helpful.
> Version 2026.5

## 1. Why this topic exists

A dispatcher must decide which orders to release, which vehicles to use, and in what sequence to visit customers.
The decision is made with incomplete information and a clock that does not stop while optimization runs.
A route with a low distance on paper may miss a delivery appointment or exceed a vehicle's capacity.
A feasible route produced too late may be operationally useless.
A plan that assumes tomorrow's orders were known this morning cannot establish the value of an online policy.

The business question in this lab is whether waiting briefly to batch orders is worthwhile.
Waiting can combine deliveries that would otherwise use separate vehicles.
Waiting can also consume the slack needed to meet a customer's promised window.
The right answer depends on the value of distance, time, service reliability, vehicle availability, and the actual release process.

The planner owns the proposed route.
The dispatcher owns the release decision and may override the plan.
The driver supplies execution updates and follows the approved route subject to safety and local rules.
Customer service owns communication when an appointment is at risk.
Operations management chooses the policy and its service guardrails.
No optimizer output is automatically a customer commitment.

This topic distinguishes four artifacts.
A forecast is an estimate of future order volume or travel time.
A plan is the proposed vehicle assignment and stop sequence.
A target is a desired service or cost level.
A commitment is an agreed customer or carrier obligation.
The toy fixture uses planned deterministic travel minutes, not observed travel forecasts or demonstrated real-world performance.

You will solve a tiny capacitated vehicle routing problem with time windows, abbreviated CVRPTW.
You will then replay two dispatch policies against the same synthetic order releases.
The code uses exhaustive enumeration rather than an external solver so every feasibility rule is visible.
The result is a local educational calculation, not a production route recommendation.
No network calls, maps, credentials, live customer records, or paid services are needed.

> **Note:** The fixture is deliberately small enough to enumerate. Its apparent policy advantage is a property of this one constructed scenario, not evidence of savings in an operating fleet.

## 2. Learning outcomes

After this lab you can:

1. Define a CVRPTW decision, its grain, constraints, and units; exercises 1 and 2 check the formulation.
2. Reconstruct arrivals, waits, service starts, returns, and capacity usage; exercises 2 and 3 check arithmetic.
3. Explain why batching may trade distance against lateness and customer exposure; exercise 4 checks the policy decision.
4. Separate an optimizer's feasibility conclusion from its objective value and from a production service promise; exercises 5 and 6 check interpretation.
5. Execute the Python fixture and explain its deterministic tie-breaking and replay; exercise 7 checks the artifact.
6. Design controls for changing orders, unavailable vehicles, and time-dependent travel; exercises 8 and 9 check deployment judgment.
7. Specify a fair historical replay that respects information available at each dispatch instant; exercise 10 checks evaluation design.

## 3. The core mental model

```text
 orders + release times       vehicles + availability       travel matrix + windows
           |                           |                            |
           +---------------------------+----------------------------+
                                       |
                          validate units, identity, timing
                                       |
                         snapshot at dispatch decision time
                                       |
                         batching rule selects eligible orders
                                       |
                       generate assignments and stop sequences
                                       |
              capacity, window, shift, and return feasibility checks
                      |                                   |
                  infeasible                           feasible plans
                      |                                   |
             exception / escalation             score + deterministic tie-break
                                                          |
                                                proposed dispatch plan
                                                          |
                                  human approval -> release -> execution events
                                                          |
                                 replay next release with vehicles still busy
                                                          |
                          compare service, workload, cost and exception outcomes
```

A route is an ordered sequence, not a set of stops.
The travel leg into a stop determines arrival.
If arrival precedes the window, the driver waits until the window opens.
Service starts at the later of arrival and opening time.
A service-start window is feasible when that start is no later than the closing time.
The driver then spends the stated service duration before traveling to the next stop.
After the final stop, the vehicle returns to the depot.

Capacity is cumulative for the deliveries assigned to a vehicle.
For a pure-delivery route, loading the sum of all assigned quantities at the depot is enough to check capacity.
A pickup-and-delivery route needs a load trajectory, not just a total.
A multi-compartment vehicle may need separate capacity checks for weight, volume, and temperature zone.
The lab uses a single integer load dimension and no pickups.

Batching changes the set of orders visible to a dispatch decision.
An immediate policy sends currently released work now.
A hold policy waits to a declared cutoff before solving the combined batch.
The hold is not clairvoyance if the cutoff and release information are fixed before the later orders arrive.
However, evaluating that rule on one hand-selected fixture cannot establish a general advantage.

A replay is a sequence of decisions and state transitions.
At every decision, include only information available then.
Once a vehicle leaves, the fixture treats its route as immutable and unavailable until it returns.
A real system must decide whether and when an en-route driver may be rerouted.
That choice introduces communication, safety, and commitment costs that this small lab does not simulate.

The optimizer and the replay answer different questions.
The optimizer finds the least-travel feasible arrangement within one snapshot and the enumerated model.
The replay exposes consequences of dispatch timing and vehicle occupancy over multiple snapshots.
Neither can show that the underlying travel matrix, order feed, or customer promise is correct.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Vehicle routing problem (VRP) | Assign and sequence stops on vehicle routes | Calling a single fixed route a fleet plan |
| CVRPTW | Capacitated VRP with customer time windows | Assuming capacity or time feasibility follows from low cost |
| Depot | Origin and return location of each fixture route | Treating a customer stop as a depot |
| Order | One delivery request with release time, demand, and window | Assuming an order exists before its release |
| Stop | One location and service event; one order per stop here | Combining stops without preserving service duration |
| Vehicle | A resource that carries orders and becomes busy when dispatched | Reusing it while its previous route is in progress |
| Capacity | Maximum onboard units per vehicle | Comparing it to the largest order instead of route total |
| Demand | Units loaded for one delivery | Confusing product units with distance or minutes |
| Release time | Earliest minute at which the dispatcher knows the order | Treating appointment opening as order release |
| Batch cutoff | Minute at which a hold policy selects released orders | Choosing the cutoff retrospectively from future outcomes |
| Dispatch instant | Minute a route departs the depot in this fixture | Equating plan creation with actual driver departure |
| Travel matrix | Ordered-pair travel duration, in minutes | Assuming symmetry or triangle inequality without evidence |
| Service duration | Minutes needed at a stop after service starts | Omitting unloading time from downstream arrivals |
| Arrival | Minute the vehicle reaches a location | Treating early arrival as completed service |
| Waiting | Minutes from early arrival to service-window opening | Assuming waiting reduces route duration |
| Service start | Maximum of arrival and window opening | Using window close as an arrival deadline without specifying semantics |
| Time window | Inclusive allowed service-start interval in minutes | Assuming service completion is required inside the window |
| Return time | Minute the vehicle reaches the depot after its final stop | Declaring the vehicle available at the last customer |
| Route duration | Return time minus departure time, in minutes | Confusing it with travel minutes when service or waiting exists |
| Route travel | Sum of matrix minutes for all route legs | Counting only outward or intercustomer legs |
| Objective | Scalar quantity minimized by the solver | Treating omitted service or fixed costs as optimized |
| Feasible solution | Assignment satisfying every modeled hard constraint | Claiming every unmodeled legal or safety requirement is met |
| Infeasibility | No complete assignment exists within the enumerated model | Concluding the business has no possible remedy |
| Unserved order | Released order not assigned in a dispatch attempt | Silently removing it to make the objective look better |
| Tie-break | Deterministic ordering after equal primary scores | Presenting arbitrary deterministic choices as unique optima |
| Replay | Chronological reconstruction of decisions from available information | Comparing policies with different order populations |
| Busy interval | Time from vehicle departure until depot return | Assuming a vehicle can accept another depot route mid-trip |
| Baseline | Prespecified comparison policy | Comparing against an implausibly poor alternative |
| Slack | Time between service start and latest permissible start | Calling average slack a guarantee against all delays |
| Exception | Explicit condition requiring a decision outside the normal plan | Reporting only successful assignments |
| Snapshot | Versioned input state at one decision instant | Mixing order updates from after dispatch into the past |
| Counterfactual | Simulated outcome under an alternative policy | Claiming it is the historical outcome |
| Hard constraint | Condition that must be met for acceptance | Hiding its violation inside a small penalty |
| Soft constraint | Preference that may be violated at an explicit cost | Assuming a penalty makes a promise inviolable |

## 5. The mechanics, step by step

### Step 1: Frame the decision and comparison

1. Identify the dispatcher and the decision instant.
2. Specify the depot, service region, shift, fleet, and order population.
3. Define the action: assign eligible orders, choose routes, and decide when to release.
4. Specify the alternatives before examining replay results.
5. In this fixture, immediate release occurs at minutes 0 and 3.
6. The batch alternative waits until minute 3 and then releases the combined orders.
7. Define service-start windows as inclusive, not as deadlines for service completion.
8. Decide that each vehicle returns to the same depot after its route.
9. Declare that every selected order must be assigned exactly once or reported as unserved.
10. Use route travel minutes as the optimization objective.
11. Also report return times, waiting, fleet use, and unserved orders.
12. Recognize that the objective does not price driver time, waiting, or fixed vehicle costs.
13. Obtain a business owner's agreement on guardrails before interpreting any lower score as a recommendation.
14. Output a decision record containing policy, time horizon, and metrics.

The fixture treats release at minute 3 as an event that can be acted on at minute 3.
In a real event stream, define ordering when an update and dispatch share a timestamp.
A timestamp alone does not establish when data was ingested or available to the decision service.

### Step 2: Construct the input contract

1. Assign a unique order identifier and a stable stop identifier.
2. Record release time in a common clock and state the time zone for real timestamps.
3. Record quantity in one explicit unit and ensure it is nonnegative.
4. Record the earliest and latest permissible service-start times.
5. Record service duration in minutes.
6. Record each vehicle's capacity and earliest availability.
7. Record a travel time for each directed pair of relevant locations.
8. Include travel from and back to the depot.
9. State whether travel times depend on departure time, vehicle type, or traffic.
10. For a static matrix, version its extraction time and methodology.
11. Validate that each order has a matrix row and column.
12. Reject duplicate order identifiers rather than silently merging them.
13. Check that window opening is not later than window closing.
14. Check that all numeric durations and capacities are valid in their declared units.
15. Record which fields are assumed versus sourced from an operational system.
16. Preserve input and code versions so a historical decision can be reconstructed.

The embedded fixture is trusted, synthetic, and intentionally small.
The sample program does not implement all these production input validations.
A production ingestion boundary must reject or quarantine invalid data before optimization.

### Step 3: Define the decision variables

1. Decide whether a vehicle is used for a route.
2. Assign every selected order to exactly one route.
3. Choose a permutation of the assigned orders on that route.
4. Associate each route with a vehicle available at the dispatch instant.
5. Track service-start time for each order.
6. Track return time for each dispatched vehicle.
7. For delivery-only loads, sum assigned demand to determine depot load.
8. If vehicle capacities differ, evaluate each route against the assigned vehicle's capacity.
9. Here all vehicles have the same capacity, so route-to-vehicle identity does not change route feasibility.
10. If drivers differ in skills or restrictions, include compatibility explicitly.
11. If stops can be split, introduce a split policy and quantities; the fixture forbids splitting.
12. Output an assignment and ordered list for each used vehicle.

An arc-based mixed-integer formulation often uses binary variables for whether vehicle k travels from node i to node j.
That formulation also needs flow conservation, one-visit constraints, capacity constraints, and temporal consistency.
Subtour elimination or equivalent ordering constraints are needed to exclude disconnected customer cycles.
The exhaustive enumerator avoids explicit subtour constraints because it builds depot-to-depot sequences directly.
It is pedagogical rather than a replacement for a scalable solver.

### Step 4: Enforce service and capacity feasibility

For an order j after node i, calculate:

```text
arrival_j       = departure_i + travel_minutes[i, j]
service_start_j = max(arrival_j, window_open_j)
wait_j          = service_start_j - arrival_j
departure_j     = service_start_j + service_minutes_j
```

At the depot, `departure_i` is the route dispatch time.
At every customer, require `service_start_j <= window_close_j`.
The earliest side of the window is enforced by the maximum operation.
This definition permits service to finish after the close if service started on time.
If the business requires completion inside the window, check `departure_j <= window_close_j` instead.
Do not mix the two interpretations in the same evaluation.

For each route, require the sum of delivery demand to be at most vehicle capacity.
A route with total demand exactly equal to capacity is feasible.
For an all-delivery route, onboard load decreases after each completed stop.
For a mixed pickup-delivery route, simulate load after every pickup and drop.
A total-demand shortcut cannot establish mixed-route feasibility.

After the final service, add the return leg to the depot.
Use that return time to decide when the vehicle may next depart.
The fixture has no maximum route duration, driver break, shift-end, or depot opening constraint.
Adding such constraints would require new data and explicit checks, not just a note in the objective.

### Step 5: Enumerate the tiny snapshot

1. Sort the selected order identifiers for reproducible iteration.
2. Assign each order to one of the currently available vehicles.
3. For every assignment, list every permutation of each nonempty vehicle group.
4. Evaluate each resulting route using the time and capacity rules.
5. Reject an assignment as soon as any route fails.
6. Sum route travel minutes, including all depot legs.
7. Prefer the lower total travel score.
8. On equal travel, prefer fewer used vehicles.
9. On a further tie, compare the sorted route tuples lexicographically.
10. Map the selected sorted routes to available vehicles in sorted identifier order.
11. Report infeasible if no full assignment is found.
12. Never silently delete a difficult order to obtain a lower score.

Identical vehicles make the sorted-route tie-break safe for this fixture.
For heterogeneous vehicles, sorting routes independently of vehicle identity can change feasibility or cost.
A heterogeneous implementation must retain the vehicle-route pairing throughout search and tie-breaking.
The enumeration repeats some equivalent assignments; the deliberate simplicity is acceptable only at this fixture size.

### Step 6: Replay dispatch decisions

1. Initialize all vehicles at the depot and available at minute 0.
2. Process decision times in chronological order.
3. At each time, collect released, not-yet-dispatched orders.
4. Consider a vehicle available only if its recorded return time is no later than the decision time.
5. Solve a full assignment for the selected orders and currently available vehicles.
6. If feasible, record each route's departure and return time.
7. Mark assigned orders as dispatched.
8. Update each used vehicle's next available time to its route return.
9. If infeasible, record the pending orders as an explicit exception.
10. Keep unresolved orders pending for a later configured decision instant.
11. At the final decision instant, report all orders still pending as unserved.
12. Do not assume a vehicle can be reassigned in the middle of its existing route.
13. Do not use the minute-3 order in the minute-0 optimization.
14. Keep the two policies' order releases, capacities, matrix, and window rules identical.
15. Compare outcomes only after the replay completes.

This replay does not create extra decision instants when a vehicle returns.
An order that cannot be assigned at the last configured instant remains unserved even if a vehicle becomes available later.
A fuller event engine would schedule return-triggered retries and specify their effect on commitments.
That distinction matters when interpreting an infeasible dispatch snapshot.

### Step 7: Check feasibility independently

For each reported route, recompute all legs and customer service starts from the saved inputs.
Check each order appears exactly once across routes or in the final unserved list.
Check that no order departs before release.
Check route demand against the assigned vehicle's capacity.
Check each customer service start against both window bounds.
Check that the next use of a vehicle starts no earlier than its previous return.
Recompute the route travel objective from all matrix legs.
Check that reported service and waiting do not accidentally enter the travel-only score.
Record why any pending order could not be assigned at a decision instant.

The program in section 7 includes fixture assertions for the key results and a deliberately infeasible case.
For production use, a separate verifier should read the output artifact and independently calculate these constraints.
A solver's internal status alone is not an independent verifier.
Model correctness and data quality remain separate from solver correctness.

### Step 8: Compare business outcomes

1. Count released, dispatched, and unserved orders.
2. Report on-time service starts over the full released population.
3. Report route travel minutes and vehicle departures.
4. Report driver duty time, including wait and service, if costed.
5. Examine each customer's delay between release and dispatch.
6. Examine minimum time-window slack, not just an average.
7. Compare performance on identical days, depots, and available input snapshots.
8. Include exceptions, manual overrides, and unassigned orders in the denominator.
9. Estimate incremental cost using explicit rates and assumptions.
10. Report distributions across many operational days, not only a mean.
11. Segment by route density, demand size, and window tightness.
12. Identify operational constraints and approval requirements before recommending rollout.

A travel-minute difference is not automatically a labor saving.
The fleet may pay fixed shifts, waiting, overtime, minimum carrier charges, or idle time.
A customer may value a longer advance notice even if service remains within the window.
Financial translation needs local rate cards and a causal evaluation design.

### Step 9: Decide and document

The educational decision for this fixture is to consider the minute-3 batch because it reduces travel while preserving the modeled hard constraints.
That is a conditional scenario conclusion, not authorization to change a real dispatch process.
A production decision would need evidence from representative days, approved service thresholds, and a rollback plan.
Document the chosen policy, dispatch cutoff, exception owner, model version, input version, and decision timestamp.
Document how late orders, delays, cancellations, and vehicle failures are handled.
Capture the dispatcher override reason and the route actually driven.
Keep customer communication and driver safety under accountable human and operational controls.

## 6. Worked example

### Scenario and fixture

A synthetic depot has two identical vehicles, `V1` and `V2`.
Each vehicle carries at most 5 load units.
Each route starts and ends at depot `D`.
All times below are integer minutes from a common origin.
All locations lie on a line, so the matrix is the absolute difference between positions.
This is a synthetic travel-time construction, not a geographic distance estimate.

| Location | Position in travel minutes from origin |
|---|---:|
| D | 0 |
| A | 2 |
| B | 4 |
| C | 6 |

The full travel-time matrix, in minutes, is:

| From / to | D | A | B | C |
|---|---:|---:|---:|---:|
| D | 0 | 2 | 4 | 6 |
| A | 2 | 0 | 2 | 4 |
| B | 4 | 2 | 0 | 2 |
| C | 6 | 4 | 2 | 0 |

Each order is a single stop with one minute of service.
Windows are inclusive service-start intervals.

| Order and stop | Release minute | Demand units | Window open | Window close | Service minutes |
|---|---:|---:|---:|---:|---:|
| A | 0 | 2 | 0 | 10 | 1 |
| B | 0 | 3 | 4 | 12 | 1 |
| C | 3 | 2 | 8 | 18 | 1 |

Total demand is 7 units.
One vehicle cannot carry all three orders because capacity is 5 units.
Two vehicles together have 10 units of capacity, but aggregate capacity alone is not enough to establish feasible windows.
The route and timing checks below provide the missing evidence for this fixture.

### Immediate dispatch, first decision at minute 0

Only A and B have been released.
Combining A and B uses exactly `2 + 3 = 5` load units.
The least-travel route is `D -> A -> B -> D`.

| Event | Calculation | Minute |
|---|---|---:|
| Depart D | Decision instant | 0 |
| Arrive A | 0 + 2 | 2 |
| Start A | max(2, 0) | 2 |
| Finish A | 2 + 1 | 3 |
| Arrive B | 3 + 2 | 5 |
| Start B | max(5, 4) | 5 |
| Finish B | 5 + 1 | 6 |
| Return D | 6 + 4 | 10 |

The route's travel is `2 + 2 + 4 = 8` minutes.
Its duration is 10 minutes because two minutes of service are included.
There is no waiting on this route.
At minute 3, `V1` remains busy until minute 10.

### Immediate dispatch, second decision at minute 3

C is released at minute 3.
`V2` is the only vehicle at the depot and immediately available.
Dispatch `D -> C -> D` at minute 3.

| Event | Calculation | Minute |
|---|---|---:|
| Depart D | C release | 3 |
| Arrive C | 3 + 6 | 9 |
| Start C | max(9, 8) | 9 |
| Finish C | 9 + 1 | 10 |
| Return D | 10 + 6 | 16 |

The C route's travel is `6 + 6 = 12` minutes.
Its duration is 13 minutes, including one minute of service.
Immediate dispatch therefore travels `8 + 12 = 20` minutes across two departures.
Every order is served within its specified window.
The latest depot return occurs at minute 16.

### Batched dispatch at minute 3

The batch policy holds A and B until minute 3.
At that instant, all three orders are eligible.
A and B each experience a three-minute release-to-dispatch hold.
C experiences no release-to-dispatch hold.

The selected routes are `D -> A -> D` and `D -> B -> C -> D`.
Their loads are 2 and `3 + 2 = 5` units respectively.
Their travel is 4 and 12 minutes respectively, for a total of 16 minutes.
The selected sequence follows the stated deterministic tie-break among equal-travel choices.

For A:

| Event | Calculation | Minute |
|---|---|---:|
| Depart D | Batch cutoff | 3 |
| Arrive A | 3 + 2 | 5 |
| Start A | max(5, 0) | 5 |
| Finish A | 5 + 1 | 6 |
| Return D | 6 + 2 | 8 |

For B and C:

| Event | Calculation | Minute |
|---|---|---:|
| Depart D | Batch cutoff | 3 |
| Arrive B | 3 + 4 | 7 |
| Start B | max(7, 4) | 7 |
| Finish B | 7 + 1 | 8 |
| Arrive C | 8 + 2 | 10 |
| Start C | max(10, 8) | 10 |
| Finish C | 10 + 1 | 11 |
| Return D | 11 + 6 | 17 |

Both routes satisfy capacity and all three service-start windows.
The latest depot return occurs at minute 17, one minute later than immediate dispatch.
All three customer service starts remain within the modeled windows.
A's and B's starts are three and two minutes later than in the immediate policy.
C's start is one minute later.

### Why 16 travel minutes is optimal for this snapshot

Because total demand is 7 units and vehicle capacity is 5 units, at least two routes are necessary.
All locations are to the right of the depot in this artificial metric.
A route visiting C must travel at least 6 minutes outward and 6 minutes back, or 12 travel minutes.
The remaining route must visit at least one of A or B.
The cheapest possible remaining nonempty route is the 4-minute depot-A-depot route.
Thus every feasible two-route solution has travel of at least `12 + 4 = 16` minutes.
The selected routes reach that lower bound and satisfy the windows and capacities.
This argument relies on the fixture's line matrix and does not prove an optimum for a general road network.

### Business interpretation

The batch saves `20 - 16 = 4` travel minutes in this synthetic replay.
Relative to the immediate baseline, the reduction is `4 / 20 = 20%` of modeled travel minutes.
Both policies have three of three orders served on time under the deterministic inputs.
The batch has a later final vehicle return and later customer service starts.
The example does not assign a monetary value to the 4 minutes.
It does not represent traffic variability, order cancellations, driver breaks, or additional customer notification requirements.
The operational recommendation is to investigate a hold policy under real service guardrails, not to claim a realized saving.

### Sensitivity reasoning

If A's latest service start were minute 4, the chosen batch would violate A's window because it starts at minute 5.
The immediate route would still start A at minute 2.
If C were not released until minute 4, it would not be eligible for the minute-3 batch.
If a vehicle were unavailable at minute 3, aggregate capacity might be insufficient even though the nominal fleet has two vehicles.
If service completion had to occur by the closing minute, each service start would need to leave at least one minute for unloading.
These changes must be simulated under an explicitly revised input contract; they are not included in the reported 20% figure.

### Audit every possible two-route partition

The batch snapshot has three indivisible orders and two identical vehicles.
Each feasible partition has one singleton and one pair.
Putting all three orders on one vehicle exceeds capacity.
The three partitions below therefore cover every possible allocation shape.
For each pair, both possible stop sequences must still be checked.
The partition is not a complete route specification until that sequence is chosen.

**Partition 1: A alone, B and C together.**

- The singleton route has 2 units and travels 4 minutes.
- The pair route has `3 + 2 = 5` units and travels 12 minutes in either order.
- The total is 16 travel minutes.
- For B then C, B starts at minute 7 and C starts at minute 10.
- The B-C vehicle returns at minute 17.
- For C then B, C starts at minute 9 and finishes at minute 10.
- The C-to-B leg takes 2 minutes, so B starts at minute 12.
- Minute 12 is exactly B's inclusive latest service start.
- B finishes at minute 13 and the vehicle returns at minute 17.
- Both sequences satisfy the stated service-start windows.
- The code selects B-C because `("B", "C")` sorts before `("C", "B")`.
- This is a tie-break, not a claim that B-C has lower modeled travel.

**Partition 2: B alone, A and C together.**

- The singleton B route has 3 units and travels 8 minutes.
- The pair route has `2 + 2 = 4` units and travels 12 minutes in either order.
- The total is 20 travel minutes.
- For A then C, A starts at minute 5 and finishes at minute 6.
- The A-to-C leg takes 4 minutes, so C starts at minute 10.
- C finishes at minute 11 and the pair vehicle returns at minute 17.
- This A-C sequence is feasible under the service-start windows.
- For C then A, C starts at minute 9 and finishes at minute 10.
- The C-to-A leg takes 4 minutes, so A would start at minute 14.
- A's latest permitted service start is minute 10.
- The C-A sequence is therefore infeasible even though its travel score is 12 minutes.
- The feasible partition costs 4 more travel minutes than partition 1.

**Partition 3: C alone, A and B together.**

- The singleton C route has 2 units and travels 12 minutes.
- The pair route has `2 + 3 = 5` units and travels 8 minutes in either order.
- The total is 20 travel minutes.
- For A then B, A starts at minute 5 and finishes at minute 6.
- B starts at minute 8, finishes at minute 9, and the vehicle returns at minute 13.
- For B then A, B starts at minute 7 and finishes at minute 8.
- A starts at minute 10, exactly its inclusive closing time.
- A finishes at minute 11 and the vehicle returns at minute 13.
- Both pair sequences pass the service-start rule.
- The singleton C route starts service at minute 9 and returns at minute 16.
- This partition is feasible but costs 4 more travel minutes than partition 1.

These checks provide a second way to see why the selected partition is optimal.
The lower-bound proof above is shorter; this enumeration additionally exposes sequence feasibility.
At larger scale, manually listing all partitions and permutations becomes impractical.
A verifier must still check each returned route even when the search is delegated to a solver.

### Examine the timing margin

The batched A route starts service at minute 5 with a latest start of minute 10.
Its service-start slack is therefore 5 minutes.
The batched B route starts at minute 7 with a latest start of minute 12.
Its slack is also 5 minutes.
The batched C route starts at minute 10 with a latest start of minute 18.
Its slack is 8 minutes.
These values describe the deterministic schedule, not an unconditional delay allowance.

A delay on the depot-to-B leg moves B's arrival and potentially also C's arrival.
For example, adding 2 minutes to that leg moves B's service start to minute 9.
With the same subsequent leg and service durations, C starts at minute 12.
Both still meet their respective windows in that particular scenario.
Adding 6 minutes instead moves B's service start to minute 13.
That exceeds B's closing minute 12, so the route fails the hard window.
The 6-minute scenario is not evaluated by the program unless its travel model is changed.

Early-window waiting can absorb some upstream delay without changing a service start.
Consider the C-only route if it departs at minute 0 instead of minute 3.
Its nominal arrival is minute 6, before C's opening minute 8.
It waits 2 minutes and starts service at minute 8.
An additional 1-minute travel delay would change arrival to minute 7 but not the start.
An additional 3-minute delay would change arrival and service start to minute 9.
That illustration concerns timing arithmetic only: C is not released at minute 0 in the actual fixture.
Dispatching it then would violate the release constraint and is not an allowed policy.

### Preserve the decision audit trail

For each policy, record the input fixture version and the list of decision instants.
Record the available vehicle set at each instant, including why each excluded vehicle is busy.
Record the selected order set and the release time of each order.
Record the optimizer's objective, tie-break, route output, and feasibility status.
Record each route's load, departures, service starts, and depot return.
Record any exception even if a later decision eventually serves the order.
For the immediate policy, the minute-3 vehicle set contains V2 but not V1.
For the batch policy, both vehicles remain available at minute 3.
This state difference is part of the policy effect rather than an arbitrary change in the fleet.

An operational audit record should also distinguish the proposed plan from the released plan.
If a dispatcher changes a sequence, retain the original proposal and the approved replacement.
If an order is cancelled after release, preserve the cancellation's effective and received timestamps.
If a travel estimate is corrected after the decision, retain the original estimate used at dispatch.
Use the later correction for a separately labeled analysis, not to rewrite the decision-time record.
The small script prints an inspectable trace but does not persist or secure an audit log.

## 7. Code, query, or calculation

### Prerequisites and execution

Use Python 3.9 or later and its standard library.
Create a local file named `vrp_dispatch_lab.py` containing the complete block below.
Run `python vrp_dispatch_lab.py` from the directory containing the file.
On systems where the Python 3 executable is named `python3`, use `python3 vrp_dispatch_lab.py` instead.
No package installation, network access, solver license, or external data file is required.
The fixture is embedded in the script so its inputs and output are reproducible together.
The script writes no output files and performs no live dispatch.
Assertions are executable fixture checks, not a comprehensive production test suite.

```python
from itertools import permutations, product

# Minute offsets from a shared origin; one demand unit is an abstract load unit.
POSITION = {"D": 0, "A": 2, "B": 4, "C": 6}
ORDERS = {
    "A": {"release": 0, "demand": 2, "open": 0, "close": 10, "service": 1},
    "B": {"release": 0, "demand": 3, "open": 4, "close": 12, "service": 1},
    "C": {"release": 3, "demand": 2, "open": 8, "close": 18, "service": 1},
}
CAPACITY = 5
VEHICLES = ("V1", "V2")


def travel(a, b):
    """Synthetic, symmetric travel minutes; not road-network distance."""
    return abs(POSITION[a] - POSITION[b])


def evaluate_route(route, depart, orders=ORDERS, capacity=CAPACITY):
    """Return route facts, or None if any modeled hard constraint fails."""
    if sum(orders[o]["demand"] for o in route) > capacity:
        return None
    now, previous, drive = depart, "D", 0
    starts = {}
    for order_id in route:
        order = orders[order_id]
        if order["release"] > depart:
            return None
        leg = travel(previous, order_id)
        drive += leg
        arrival = now + leg
        start = max(arrival, order["open"])
        if start > order["close"]:
            return None
        starts[order_id] = start
        now = start + order["service"]
        previous = order_id
    back = travel(previous, "D")
    return {"travel": drive + back, "return": now + back, "starts": starts}


def solve(order_ids, vehicles, depart, orders=ORDERS):
    """Enumerate full assignments and ordered routes for identical vehicles."""
    ids = tuple(sorted(order_ids))
    available = tuple(sorted(vehicles))
    if not ids:
        return []
    if not available:
        return None
    best_key, best_routes = None, None
    for allocation in product(range(len(available)), repeat=len(ids)):
        groups = [tuple(ids[i] for i, v in enumerate(allocation) if v == j)
                  for j in range(len(available))]
        nonempty = [group for group in groups if group]
        for route_set in product(*(permutations(group) for group in nonempty)):
            routes = tuple(sorted(route_set))
            facts = [evaluate_route(route, depart, orders) for route in routes]
            if any(fact is None for fact in facts):
                continue
            key = (sum(fact["travel"] for fact in facts), len(routes), routes)
            if best_key is None or key < best_key:
                best_key, best_routes = key, routes
    if best_routes is None:
        return None
    return [(vehicle, route, evaluate_route(route, depart, orders))
            for vehicle, route in zip(available, best_routes)]


def replay(decisions, orders=ORDERS):
    """Keep failed orders pending; vehicles return before becoming available."""
    available_at = {vehicle: 0 for vehicle in VEHICLES}
    dispatched = set()
    records = []
    exceptions = []
    for now in decisions:
        pending = sorted(o for o, row in orders.items()
                         if row["release"] <= now and o not in dispatched)
        available = [v for v in VEHICLES if available_at[v] <= now]
        plan = solve(pending, available, now, orders)
        if plan is None:
            exceptions.append((now, tuple(pending)))
            continue
        for vehicle, route, fact in plan:
            available_at[vehicle] = fact["return"]
            dispatched.update(route)
            records.append((now, vehicle, route, fact))
    unserved = tuple(sorted(set(orders) - dispatched))
    return records, exceptions, unserved


def summary(label, result):
    records, exceptions, unserved = result
    total = sum(fact["travel"] for _, _, _, fact in records)
    print(f"{label}: travel={total} min; departures={len(records)}; "
          f"unserved={list(unserved)}; exceptions={exceptions}")
    for depart, vehicle, route, fact in records:
        path = "-".join(("D", *route, "D"))
        print(f"  t={depart} {vehicle} {path} "
              f"start={fact['starts']} return={fact['return']}")


immediate = replay((0, 3))
batch = replay((3,))
summary("immediate", immediate)
summary("batch", batch)

# Fixture regression checks, including capacity and no available vehicle.
assert sum(r[3]["travel"] for r in immediate[0]) == 20
assert sum(r[3]["travel"] for r in batch[0]) == 16
assert immediate[2] == batch[2] == ()
assert evaluate_route(("A", "B", "C"), 3) is None
assert solve(("A",), (), 0) is None
assert evaluate_route(("A",), 9) is None  # arrival 11 exceeds close 10
assert immediate[0][0][3]["return"] == 10
assert batch[0][1][3]["starts"] == {"B": 7, "C": 10}
print("fixture checks passed")
```

Expected output on a standard Python interpreter is:

```text
immediate: travel=20 min; departures=2; unserved=[]; exceptions=[]
  t=0 V1 D-A-B-D start={'A': 2, 'B': 5} return=10
  t=3 V2 D-C-D start={'C': 9} return=16
batch: travel=16 min; departures=2; unserved=[]; exceptions=[]
  t=3 V1 D-A-D start={'A': 5} return=8
  t=3 V2 D-B-C-D start={'B': 7, 'C': 10} return=17
fixture checks passed
```

The first assertion checks the immediate travel score; the second checks the batched score.
The third verifies that neither scenario silently drops an order.
The capacity check rejects all three deliveries on one vehicle.
The unavailable-vehicle check returns infeasible rather than inventing a vehicle.
The late-start check rejects service that starts after the close.
The remaining checks pin the relevant vehicle return and batch service starts.

The program uses a static matrix, deterministic service times, and identical vehicles.
It enumerates assignments and permutations, so runtime grows rapidly with the number of orders.
Do not use it for production-sized fleets or sensitive data.
It handles the fixture's fixed decision times only; it is not a real-time event processing system.
The `orders` parameter can vary releases, windows, quantities, and service times for existing locations A, B, and C.
Adding a new order identifier also requires a corresponding `POSITION` entry because identifiers double as location keys here.
The `decisions` argument must be chronological and include the intended horizon; the script assumes rather than validates that contract.
An order released after the last decision is included in `unserved`, not silently treated as served.
A snapshot failure is recorded in `exceptions`; a later configured decision can retry pending orders.

## 8. Data, standards, and regulatory context

### Operational data contract

A real dispatch implementation needs an order source with a stable order key and an event history.
It needs the time the order became usable by the dispatcher, not only a retrospectively corrected order timestamp.
A useful contract distinguishes event time, system ingestion time, and the decision's data-availability cutoff.
Late corrections should be versioned rather than overwriting the snapshot used for a historical replay.

The order table should specify whether one row is a customer order, a delivery line, or a stop.
If several lines are consolidated into one stop, preserve the original order identifiers and the aggregation rule.
The rule must specify how quantities, service times, customer windows, and priority are combined.
Summing units with incompatible packaging or temperature requirements is invalid.
A cancellation event should state when the cancellation was known and whether the vehicle had already departed.
A changed appointment should preserve both the old and new window with effective and availability times.

The fleet table needs vehicle identifiers, capacity by relevant dimension, start location, and availability intervals.
Heterogeneous vehicle and driver restrictions may include skills, accessible loading equipment, or carrier eligibility.
A depot inventory restriction requires inventory at the origin, not just a vehicle capacity field.
A multi-depot fleet requires route origins, permitted returns, and repositioning rules.
The synthetic fixture assumes away all of these distinctions.

The travel table needs origin, destination, estimated duration, units, source, version, and validity interval.
Road travel can be asymmetric; D-to-A need not equal A-to-D.
Time-dependent congestion can change a leg based on its departure time.
Geocoding confidence and address correction history also affect route validity.
The line-based travel matrix in this lab is not a road map and must not be presented as one.

### Clocks and windows

Store real operational instants with explicit time zones or UTC offsets.
Resolve daylight-saving ambiguity before constructing a replay.
State whether endpoints are inclusive and whether the promise concerns arrival, service start, or service completion.
State whether early arrival is allowed and where a vehicle may wait.
A warehouse appointment may permit arrival before a gate opens but prohibit parking on a public road.
The route model must reflect the actual operating agreement rather than infer it from a field name.

Do not compare a local timestamp to a UTC timestamp without conversion.
Do not treat a date-only commitment as a precise time-window close.
Do not replace a customer promise with a system-derived target without recording the change.
A trace should contain planned departure, observed departure, planned service start, and observed service event where available.
Those events have different meanings and may be subject to missing or delayed reporting.

### Standards and identifiers

[GS1](https://www.gs1.org/standards/id-keys) describes identification keys used for entities, locations, and logistics units.
A Global Location Number can identify a party or location when the trading partners use that standard.
A Serial Shipping Container Code can identify a logistics unit where the relevant partners have adopted it.
Those identifiers do not by themselves provide a valid address, a travel time, an appointment, or a feasible route.
Use the governing trading-partner agreement and appropriate GS1 specifications to determine implementation requirements.

The [GS1 Electronic Product Code Information Services standard](https://www.gs1.org/standards/epcis) describes structured supply-chain event data and context.
It can support traceability when implemented with suitable identifiers and event capture.
It does not automatically tell the dispatcher when an event became available to a historical decision.
Capture both business-event and system-availability semantics needed for the replay.
An organization may use a different agreed event schema; interoperability depends on the contract actually implemented.

### Driver rules and privacy

Where U.S. federal hours-of-service rules apply, the [Federal Motor Carrier Safety Administration hours-of-service materials](https://www.fmcsa.dot.gov/regulations/hours-service/summary-hours-service-regulations) describe relevant requirements and exceptions.
Applicability depends on vehicle, operation, jurisdiction, and current rules.
A route passing this lab's capacity and window checks is not certified compliant with those requirements.
Model duty status, breaks, shift limits, and jurisdiction-specific exceptions only with current authoritative guidance and qualified review.
The toy program has no hours-of-service logic.

Delivery addresses and driver traces can be personal or commercially sensitive data.
Apply applicable privacy rules, approved retention, access controls, and data minimization to actual operating records.
The proper legal basis and obligations depend on jurisdiction and use; this lab is not legal advice.
The synthetic fixture contains no real addresses or driver identities.

### Quality and traceability controls

A release pipeline should validate unique identifiers, nonnegative quantities, valid time ordering, and matrix coverage.
It should record source versions and reject missing values rather than substitute a convenient zero.
It should distinguish known-zero travel within the same location from missing travel between two locations.
It should retain the decision's model version, solver configuration, input snapshot identifier, and objective value.
It should persist exceptions and manual changes with timestamps and accountable owners.
It should restrict access to dispatch records and redact data in shared test artifacts.
None of these production controls is implemented by the standalone toy script.

## 9. Industry practice and case evidence

### What the literature establishes

Solomon's 1987 paper, [Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints](https://doi.org/10.1287/opre.35.2.254), studies algorithms and benchmark problem instances for routing with time windows.
The benchmark framing demonstrates that sequence, capacity, and timing interact.
It is not evidence that the three-order fixture reflects a particular carrier's fleet.
Results on standard instances help compare algorithms under defined assumptions.
They do not alone establish savings in a different operating environment.

The book *The Vehicle Routing Problem*, edited by Toth and Vigo, provides a broader account of VRP formulations and algorithms.
Real routing variants add fleet, scheduling, road-network, and operational constraints.
The exact model should follow the decision rights and commitments of the operation.
An exact optimum of the wrong model can be operationally inferior to a reviewed heuristic plan.

### Public carrier example

UPS describes its On-Road Integrated Optimization and Navigation initiative, known as ORION, in its public [technology and innovation materials](https://about.ups.com/us/en/our-company/innovation-driven.html).
That public account is a company description of an operational routing initiative.
It is not an independent validation of this lab's numerical example or batching rule.
The company operates at a scale and with constraints that are not represented by the three synthetic stops.
Claims about a particular saving require their own source, time period, baseline, and verification; none is asserted here.

[INFERENCE] A routing program in a large parcel network must coordinate route recommendations with driver execution, changing volume, and operational exceptions.
The fixture illustrates only the narrower logical distinction between a route plan and a dispatch replay.
It does not reproduce the company's algorithm, input data, business case, or production controls.

### Research and operational evidence are different

A research paper may show an algorithm's performance on published instances.
A company case may describe adoption or a reported business outcome.
An operational replay may estimate what would have happened on historical days under a modeled policy.
A controlled pilot may observe outcomes after a policy is introduced.
None of these evidence types should be substituted for another without identifying the assumptions.

A historical replay is especially vulnerable to missing counterfactuals.
Historical travel times can depend on the actual route and departure time.
A new policy may send vehicles on different roads or at different times.
Reusing the observed duration of a different route as if it were directly observed is unjustified.
A replay must either estimate those counterfactual durations or state what cannot be inferred.

### Translating a replay into a pilot

Start with a representative sample of days and depots.
Prespecify the immediate-release baseline, candidate batch cutoffs, and service guardrails.
Include high-volume, low-volume, peak, disruption, and tight-window periods.
Reconstruct vehicle states and order knowledge at each decision instant.
Use a credible travel-time model and test sensitivity to congestion and service-time variation.
Show the distribution of on-time service, unserved orders, driver workload, and cost components.
Have dispatchers review proposed plans and log overrides without altering the historical input record.

A shadow deployment can compare recommendations with existing dispatch without controlling vehicles.
Shadow mode cannot directly measure customer or driver response to a changed plan.
A staged field trial requires operational approval, safety review, and a predefined rollback criterion.
Report deviations and failed assignments as outcomes, not as records to be removed from the analysis.
When assignment is not randomized, account for differences between depots, days, and drivers before attributing any observed change to the policy.

### Example decision memo

A defensible memo for the fixture would state the decision, alternatives, input assumptions, and simulated results.
It would report 20 versus 16 modeled travel minutes and three of three modeled on-time service starts.
It would report the one-minute later final return under the hold policy.
It would identify the small synthetic fixture and the absence of stochastic travel or driver constraints.
It would recommend further representative evaluation rather than deployment or a financial claim.
An accountable operations owner would decide the next step and service guardrails.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Future order leakage | Minute-0 plan contains a minute-3 order | Replay uses final daily extract | Inflated batching benefit | Filter by actual availability and version snapshots |
| Busy vehicle reused | Two routes overlap for one vehicle | State resets at each solve | Physically impossible plan | Persist return times and verify nonoverlap |
| Capacity ignored | Route load exceeds rated units | Constraint omitted or units mismatched | Unsafe or impossible loading | Check each capacity dimension and validate units |
| Window semantics wrong | Arrival passes but completion promise fails | Start and completion are conflated | Missed commitment | Contract the window event and check that event |
| Service omitted | Later stops consistently arrive late | Only travel legs are modeled | Systematic lateness | Measure and include service duration |
| Waiting omitted | Departures precede window opening | Early arrival treated as service | False feasibility | Use max of arrival and opening |
| Depot return omitted | Vehicle appears free at last stop | Return leg not modeled | Double booking | Include return leg and update availability |
| Unserved orders dropped | Reported on-time rate rises as orders disappear | Denominator excludes failures | Misleading service result | Report all released orders and exceptions |
| Infeasibility hidden | Empty route presented as success | Solver failure mapped to default output | Orders not delivered | Return explicit status and escalate |
| Travel data stale | Consistent lateness by corridor | Static matrix misses congestion | Fragile schedules | Version matrices and calibrate by time period |
| Wrong time zone | Routes shift by an hour on transition day | Local timestamps parsed inconsistently | Missed appointments | Store offsets and test daylight-saving boundaries |
| Duplicate orders | Same customer demand assigned twice | Retries lack idempotency | Duplicate deliveries and cost | Stable identifiers and deduplication policy |
| Cancellation ignored | Driver visits cancelled stop | Event not applied to dispatch state | Wasted travel and poor experience | Version cancellation and define reroute policy |
| No exception owner | Pending jobs remain after solve failure | Escalation path is undefined | Missed commitments | Assign owner, deadline, and fallback process |
| Unrealistic batch hold | Service slack shrinks on tight windows | One cutoff applied indiscriminately | Late service despite lower travel | Use explicit guardrails and stress tests |
| Objective misread | Travel minutes reported as labor savings | Fixed and waiting costs ignored | Incorrect business case | Use a validated cost model and report components |
| Tie instability | Identical inputs yield different routes | Unspecified search ordering | Dispatcher distrust and hard audits | Declare deterministic tie-break and versions |
| Unmodeled driver rules | Route passes lab but exceeds legal shift | Model has no duty constraints | Safety and regulatory exposure | Add applicable constraints and qualified review |
| Bad geocode | Travel matrix points to wrong entrance | Address mapped inaccurately | Missed or impossible access | Verify locations and monitor corrections |
| Selection bias | Only successfully routed orders compared | Failed or overridden plans excluded | Optimistic policy estimate | Retain entire eligible population and override reasons |

A red flag is not proof of a particular incident.
It is a reason to inspect the corresponding data and control.
The consequences depend on operational context and must be assessed by the responsible organization.

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or replace it |
|---|---|---|---|
| All vehicles start at D | Simplifies route construction | Vehicles begin at customer or another depot | Add vehicle-specific start and end locations |
| Two identical vehicles | Makes sorted route-to-vehicle mapping valid | Capacity or eligibility differs by vehicle | Preserve vehicle identity during enumeration |
| Capacity is 5 units | Gives a concrete load limit | Weight, cube, or compartment limits differ | Add dimensions and compare each route with each limit |
| Orders are indivisible | One identifier appears on one route | Split delivery is allowed or needed | Add explicit quantity allocation and split costs |
| All work is delivery-only | Route load is initial demand sum | Pickups increase load mid-route | Track load after every stop |
| Static travel matrix | Makes the arithmetic deterministic | Congestion depends on departure time | Use versioned time-dependent travel estimates |
| Travel is symmetric | Follows from positions on a line | One-way roads or traffic differ | Use a directed matrix |
| Service takes one minute | Provides reproducible stop timing | Unloading varies by order and customer | Estimate a distribution and test scenario sensitivity |
| Start-window semantics | Defines the feasibility test | Commitment is on arrival or completion | Change the checked event and recalculate |
| Routes cannot change en route | Makes vehicle state simple | Live dispatch changes active routes | Model state, communication delay, and change cost |
| No return-triggered retry | Fixed decision instants bound the replay | Pending work should be reconsidered on return | Add return events and explicit retry policy |
| Every selected order must be assigned | Prevents silent omission | Partial assignment and prioritization are allowed | Define penalties, priority, and exception rules explicitly |
| No driver duty constraints | Keeps the toy example small | Shifts, breaks, or legal limits bind | Integrate applicable duty data and rules |
| No route duration limit | Return time only controls reuse | Carrier or customer imposes a limit | Add a hard duration or shift-end constraint |
| All addresses are known | Matrix has every required pair | Address is missing or inaccurate | Validate and quarantine unresolved locations |
| All timestamps share one clock | Enables integer arithmetic | Time zones or daylight saving differ | Normalize and retain source-zone semantics |
| No stochastic disruption | Makes both policies directly reproducible | Travel, service, and order arrivals vary | Use calibrated scenarios and quantify uncertainty |
| Objective is travel minutes | Allows a transparent numerical comparison | Labor, service, emissions, or fixed costs matter | Define a validated multi-component objective |
| Decision times are prespecified | Prevents retrospective cutoff selection | Cutoff tuned on the same outcomes | Separate policy design and holdout evaluation |
| Tiny order count | Makes exhaustive search tractable | Assignments and permutations grow rapidly | Use an appropriate solver and independent verifier |

This ledger describes the code's model, not a list of optional production requirements.
Any real implementation needs a new contract and evidence for its own assumptions.
The example's calculated optimality is restricted to the modeled snapshot and objective.

## 12. Exercises

### Exercise 1 — Define the decision contract

**Time:** 10 minutes.
**Deliverable:** A compact dispatch contract for a hypothetical one-depot delivery shift.
Include the decision owner, release event, planning horizon, route origin and destination, and objective.
Specify whether windows apply to arrival, service start, or completion.
List at least two constraints that are absent from the toy program but may matter in your hypothetical operation.
**Acceptance criteria:** Units and clock are explicit; the contract distinguishes a plan from a commitment; no omitted constraint is presented as enforced.

### Exercise 2 — Calculate a route

**Time:** 10 minutes.
**Deliverable:** A timeline for `D -> B -> C -> D` departing at minute 3 using the worked fixture.
Calculate each arrival, service start, service finish, return, route load, and travel total.
**Acceptance criteria:** You include both customer service minutes and the depot return leg; the stated service starts are within the stated windows.

### Exercise 3 — Check capacity and timing

**Time:** 10 minutes.
**Deliverable:** A feasibility decision for `D -> A -> B -> C -> D` at minute 3.
Explain separately whether it passes capacity and whether timing alone would pass the service-start windows.
**Acceptance criteria:** Sum all three demands and calculate all three service starts; do not claim that a timing-feasible overloaded route is feasible overall.

### Exercise 4 — Evaluate the business trade-off

**Time:** 10 minutes.
**Deliverable:** A short comparison of immediate and batch policies using the supplied fixture.
Give travel difference, percentage relative to immediate, final return difference, and one reason the result is not a production saving.
**Acceptance criteria:** The denominator for the percentage is explicit; your conclusion includes both service and operational trade-offs.

### Exercise 5 — Diagnose window semantics

**Time:** 10 minutes.
**Deliverable:** A calculation for order A with arrival at minute 10 and one minute of service.
Determine feasibility if the close of minute 10 means latest service start, and if it means latest service completion.
**Acceptance criteria:** You show the service finish time and distinguish the two contracts.

### Exercise 6 — Explain infeasibility

**Time:** 10 minutes.
**Deliverable:** A dispatcher-facing exception message for an unassigned order at a decision instant with no available vehicles.
State the snapshot limitation and two operational actions that could be considered.
**Acceptance criteria:** The message does not assert that delivery is impossible forever; it does not silently remove the order or invent a vehicle.

### Exercise 7 — Run and inspect the fixture

**Time:** 15 minutes.
**Deliverable:** The console output and a short explanation of the capacity, late-window, and busy-vehicle checks.
Use the exact script and command from section 7.
Explain why A, B, and C cannot all travel on one vehicle.
**Acceptance criteria:** The fixture assertions pass, both policies list no unserved orders, and route times agree with section 6.
If your environment cannot run Python, report the limitation rather than claiming execution.

### Exercise 8 — Design a heterogeneous fleet extension

**Time:** 20 minutes.
**Deliverable:** A design for a 4-unit vehicle and a 5-unit vehicle with different availability times.
Describe the assignment representation, capacity check, tie-break, and an independent verification case.
**Acceptance criteria:** Your approach keeps route and vehicle identity paired; it does not assume total fleet capacity proves feasibility.

### Exercise 9 — Specify an exception workflow

**Time:** 15 minutes.
**Deliverable:** A state diagram or table for released, planned, dispatched, cancelled, delayed, and unserved orders.
Include who can authorize an override, when a customer is notified, and how a return-triggered retry would work.
**Acceptance criteria:** No order silently disappears; concurrent cancellation and dispatch have an explicit ordering rule; committed plans have an audit trail.

### Exercise 10 — Design a credible replay

**Time:** 25 minutes.
**Deliverable:** A one-page evaluation plan comparing immediate dispatch with a fixed batching policy across historical shifts.
Specify the historical cutoff, eligible population, travel-time counterfactual, baseline, metrics, uncertainty, and rollback criteria for a later pilot.
**Acceptance criteria:** All policies use information available at each decision; failed and overridden plans remain in denominators; the plan separates replay estimates from observed pilot outcomes.

## 13. Self-check answers

### Exercise 1 — Contract rubric

A strong response names an accountable dispatcher and a time at which a specific set of released orders is selected.
It identifies the depot, available vehicles, capacity unit, time unit, window event, and route return policy.
It specifies the metric minimized and the hard constraints that may not be traded away.
It labels driver shifts and road restrictions as additional requirements if they are not actually modeled.
It explicitly distinguishes the proposed dispatch plan from a customer commitment.
A response is incomplete if it treats a low objective as proof of legal or operational feasibility.

### Exercise 2 — Timeline

The route departs D at minute 3 and reaches B at `3 + 4 = 7`.
B service starts at `max(7, 4) = 7` and ends at minute 8.
The B-to-C leg takes 2 minutes, so C is reached at minute 10.
C service starts at minute 10 and ends at minute 11.
The return to D takes 6 minutes, so the vehicle returns at minute 17.
The load is `3 + 2 = 5` units, equal to capacity.
The travel total is `4 + 2 + 6 = 12` minutes.
Both service starts are within their respective inclusive windows.

### Exercise 3 — Separate checks

The load is `2 + 3 + 2 = 7` units, exceeding 5 units of capacity.
Departing at minute 3 reaches A at minute 5 and starts service at minute 5.
After service, B is reached at minute 8 and starts service at minute 8.
After B service, C is reached at minute 11 and starts service at minute 11.
All three starts are within their windows, but the route is infeasible because of capacity.
A complete feasibility conclusion requires every hard constraint to hold simultaneously.

### Exercise 4 — Policy comparison

Immediate travel is 20 minutes and batch travel is 16 minutes.
The batch difference is 4 fewer travel minutes, or `4 / 20 = 20%` relative to immediate.
The final vehicle returns at minute 17 under batching versus minute 16 under immediate dispatch.
All three modeled service starts meet their windows in both policies.
A and B wait three minutes before batch dispatch, and all customer starts occur later than in the immediate scenario.
No financial saving follows automatically from a travel-time difference in a synthetic fixture.

### Exercise 5 — Window event

Arrival at minute 10 produces service start at minute 10 and finish at minute 11.
If closing minute 10 is an inclusive latest service start, the stop is feasible.
If closing minute 10 is an inclusive latest service completion, the stop is infeasible.
The distinction must be agreed before computing service performance.

### Exercise 6 — Exception message rubric

A suitable message identifies the decision minute, affected order identifiers, and the absence of an available vehicle at that instant.
It states that the current snapshot has no complete assignment and that pending work requires attention.
The dispatcher might consider a later return-triggered retry or an approved additional carrier, subject to its own constraints and commitments.
Other actions might include customer communication or an authorized appointment revision.
The message must not assert permanent infeasibility from a single snapshot.

### Exercise 7 — Fixture interpretation

The expected output is shown in section 7; compare it with the output actually produced by your own run.
The capacity assertion rejects 7 units on a 5-unit vehicle.
The late-window assertion rejects A when departure at minute 9 yields arrival at minute 11, after its close at minute 10.
At minute 3 in the immediate replay, V1 is busy until minute 10, so V2 receives C.
The final two route-specific assertions verify a return time and the B/C service starts.
A statement that a learner ran the program is warranted only when their own execution actually occurred.

### Exercise 8 — Heterogeneous fleet rubric

A correct design binds every ordered route to its actual vehicle throughout candidate generation and scoring.
It checks the assigned vehicle's own capacity, availability, start location, and any compatibility constraints.
It uses a deterministic tie-break over vehicle-route pairs rather than sorting routes independently of vehicles.
A useful test gives a route of demand 5 units to the 4-unit vehicle and requires rejection.
A second test ensures a vehicle unavailable at the decision instant is not assigned.

### Exercise 9 — Workflow rubric

A strong workflow records event identifiers, effective times, ingestion times, and the actor responsible for changes.
It defines an ordering or version conflict rule for cancellation and dispatch at the same timestamp.
It retains a pending or unserved state until an authorized decision resolves the order.
It identifies customer notification ownership and the relationship between a revised route and existing commitments.
A return-triggered retry is scheduled from actual or estimated vehicle return, with a documented treatment of late status events.

### Exercise 10 — Replay rubric

A credible plan reconstructs orders and fleet state as known at each dispatch instant, including changes and delays.
It chooses cutoffs and metrics before evaluating a held-out period.
It includes all eligible orders, failed assignments, cancellations, and dispatcher overrides under declared denominator rules.
It models counterfactual travel rather than claiming unobserved travel is directly measured.
It reports uncertainty and segments service outcomes, not merely total route travel.
A later pilot needs an accountable owner, safety guardrails, a control or comparison design, and a rollback rule.
Replay estimates and realized pilot outcomes are reported separately.

## 14. Interview questions

### 1. What makes a route feasible in this lab?

A strong answer checks order release, vehicle capacity, every service-start window, and the vehicle's availability.
It includes depot return when determining the vehicle's next availability.
A weak answer says the shortest route is automatically feasible.

### 2. Why can a lower-travel batching policy still be undesirable?

A strong answer discusses lost slack, later service, longer final return, and customer or driver effects.
It explains that travel minutes alone do not measure all cost.
A weak answer assumes any modeled travel reduction is realized financial savings.

### 3. How do arrival and service start differ?

A strong answer calculates waiting as the difference when a vehicle arrives before a window opens.
It states which event the service commitment constrains.
A weak answer assumes an early arrival is an early service start.

### 4. What is temporal leakage in a dispatch replay?

A strong answer describes using an order, correction, or vehicle update before it was available to the historical decision.
It describes versioned snapshots and availability timestamps as controls.
A weak answer says sorting by event time alone is sufficient for all late-arriving data.

### 5. What does an infeasible solver result mean?

A strong answer confines it to the selected orders, currently available vehicles, specified constraints, and modeled horizon.
It identifies explicit exception handling and possible operational alternatives.
A weak answer declares all delivery impossible or silently drops an order.

### 6. Why must the depot return leg be included?

A strong answer includes it in travel and in the vehicle's next available time.
A weak answer makes the vehicle available at its final customer.

### 7. When is summing route demand sufficient for capacity?

A strong answer limits the shortcut to delivery-only routes loaded at the depot with a compatible single load dimension.
It uses a load trajectory for pickups or other changing loads.
A weak answer uses one total for arbitrary pickup-and-delivery operations.

### 8. Why does the toy enumerator not scale?

A strong answer notes combinatorial growth in assignments and permutations and suggests an appropriate routing solver with a separate verifier.
A weak answer assumes a faster computer makes enumeration practical for every fleet.

### 9. How would you evaluate a new cutoff?

A strong answer prespecifies the policy, reconstructs historical information, uses a fair baseline, and includes exceptions and service guardrails.
It distinguishes a counterfactual replay from actual pilot outcomes.
A weak answer selects the best cutoff on one favorable day and reports its travel reduction as realized savings.

### 10. How would heterogeneous vehicles change this code?

A strong answer keeps each route paired with the vehicle during feasibility and scoring.
It checks capacity, availability, compatibility, and vehicle-specific costs.
A weak answer sorts routes independently and assigns them arbitrarily to vehicles.

## 15. Key terms

- **Arrival:** Reaching a stop before any waiting; see [section 5](#5-the-mechanics-step-by-step).
- **Batch cutoff:** Time at which a held set of released orders is selected; see [section 3](#3-the-core-mental-model).
- **Busy interval:** Departure through depot return; see [section 6](#6-worked-example).
- **Capacity:** Vehicle load limit in stated units; see [section 4](#4-definitions-and-vocabulary).
- **Commitment:** An agreed obligation, distinct from a plan; see [section 1](#1-why-this-topic-exists).
- **Counterfactual:** An estimated outcome under an alternative policy; see [section 9](#9-industry-practice-and-case-evidence).
- **CVRPTW:** Capacitated vehicle routing problem with time windows; see [section 3](#3-the-core-mental-model).
- **Depot:** The start and return location in this fixture; see [section 6](#6-worked-example).
- **Dispatch instant:** The time a route is released and departs; see [section 5](#5-the-mechanics-step-by-step).
- **Exception:** A condition requiring an explicit operational decision; see [section 10](#10-failure-modes-and-red-flags).
- **Feasibility:** Satisfaction of all modeled hard constraints; see [section 5](#5-the-mechanics-step-by-step).
- **Forecast:** An estimate of future quantities or conditions; see [section 1](#1-why-this-topic-exists).
- **Hard constraint:** A condition that may not be violated in an accepted solution; see [section 4](#4-definitions-and-vocabulary).
- **Infeasibility:** Absence of a full modeled assignment in a snapshot; see [section 13](#13-self-check-answers).
- **Objective:** Quantity minimized by the optimizer; see [section 5](#5-the-mechanics-step-by-step).
- **Order release:** Time a request becomes eligible for dispatch; see [section 6](#6-worked-example).
- **Plan:** A proposed action, not an automatically binding commitment; see [section 1](#1-why-this-topic-exists).
- **Replay:** Chronological simulation of repeated dispatch decisions; see [section 3](#3-the-core-mental-model).
- **Return time:** Time a vehicle gets back to its depot; see [section 6](#6-worked-example).
- **Route duration:** Elapsed departure-to-return time including service and wait; see [section 4](#4-definitions-and-vocabulary).
- **Route travel:** Sum of modeled travel legs, excluding service and wait; see [section 4](#4-definitions-and-vocabulary).
- **Service duration:** Time spent servicing a stop; see [section 5](#5-the-mechanics-step-by-step).
- **Service start:** Later of arrival and window opening; see [section 5](#5-the-mechanics-step-by-step).
- **Slack:** Remaining time before the latest permitted service start; see [section 4](#4-definitions-and-vocabulary).
- **Snapshot:** Versioned decision-time input state; see [section 8](#8-data-standards-and-regulatory-context).
- **Target:** A desired level, not a commitment; see [section 1](#1-why-this-topic-exists).
- **Tie-break:** Deterministic ordering among equally scored solutions; see [section 7](#7-code-query-or-calculation).
- **Time window:** The permitted interval for the contractually defined event; see [section 8](#8-data-standards-and-regulatory-context).
- **Travel matrix:** Pairwise travel times with explicit direction and version; see [section 8](#8-data-standards-and-regulatory-context).
- **Unserved order:** An order not assigned within the replay horizon; see [section 7](#7-code-query-or-calculation).
- **Waiting:** Time between early arrival and permitted service start; see [section 3](#3-the-core-mental-model).

## 16. Sources

1. Solomon, Marius M. (1987). [Algorithms for the Vehicle Routing and Scheduling Problems with Time Window Constraints](https://doi.org/10.1287/opre.35.2.254). *Operations Research*, 35(2), 254–265. Peer-reviewed formulation and algorithms; not evidence for this fixture's business effect.
2. Toth, Paolo, and Daniele Vigo, editors. *The Vehicle Routing Problem*. Society for Industrial and Applied Mathematics, 2002. [Publisher page](https://doi.org/10.1137/1.9780898718515). Foundational formulations and methods.
3. Google OR-Tools. [Vehicle Routing Problem with Time Windows](https://developers.google.com/optimization/routing/vrptw). Official documentation for a routing solver and time-window modeling; the code in this topic does not require OR-Tools.
4. GS1. [GS1 identification keys](https://www.gs1.org/standards/id-keys). Official descriptions of identification keys; applicability depends on the trading-partner implementation.
5. GS1. [EPCIS and CBV](https://www.gs1.org/standards/epcis). Official information about supply-chain event standards; the lab does not implement EPCIS.
6. U.S. Federal Motor Carrier Safety Administration. [Hours of Service Regulations summary](https://www.fmcsa.dot.gov/regulations/hours-service/summary-hours-service-regulations). Official overview; consult current requirements and applicable exceptions for an actual operation.
7. UPS. [Innovation driven](https://about.ups.com/us/en/our-company/innovation-driven.html). Company material concerning its technology initiatives; treated as company evidence, not independent verification of a numerical benefit.
8. Python Software Foundation. [`itertools` documentation](https://docs.python.org/3/library/itertools.html). Official documentation for `product` and `permutations` used in the standalone enumeration.

Sources support the concepts attributed to them; the numerical fixture and its calculated outcomes are synthetic course material.
External rules and web pages can change; check the current authoritative version before operational use.

## 17. Links

- [Module 30 topic index](_index.md): the lab sequence and topic map.
- [Module 30 survey](../../../course/30_simulation_optimization_production_labs.md): module-level context and assessment.
- [Environment and repository lab](01_environment-repo.md): local setup and reproducibility practices.
- [Demand forecast pipeline lab](02_demand-forecast-pipeline.md): distinguish an estimated input from a dispatch decision.
- The embedded [section 7 fixture](#7-code-query-or-calculation) is the dataset and executable lab for this topic; it needs no external fixture file.
- [Worked example](#6-worked-example): independently reconstruct the routes and decision before changing the code.
