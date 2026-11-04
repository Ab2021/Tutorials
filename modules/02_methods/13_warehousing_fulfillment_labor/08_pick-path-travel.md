# Pick Path and Travel Time

> Module 13 — Warehousing, Fulfillment, and Labor Analytics
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 13, topics 02, 06, and 07
> Version 2026.5

*Routing inside the four walls and the value of a better sequence.*

## 1. Why this topic exists

A route is an operating instruction, not merely a line drawn on a floor plan.
The warehouse manager must decide whether to change the order in which workers visit pick locations.
A shorter path can release capacity, but it cannot automatically reduce payroll or improve dispatch performance.
The saving depends on the orders released together, the actual walkable network, and what workers do with the time released.
The supervisor also needs to know whether a new route creates congestion, missed cutoffs, unsafe crossings, or more sorting work.

A warehouse management system (WMS) may print locations in alphanumeric order.
That order is convenient for storage administration but can make a picker cross the same aisle repeatedly.
Another facility may use an apparently efficient sequence that ignores a one-way aisle or the location of a drop station.
A third may reduce theoretical distance while sending every picker to the same narrow cross-aisle at the same time.
These are different problems and require different remedies.

This lesson holds slot assignments and the released order set fixed when estimating the direct routing effect.
Slotting changes where inventory resides; batching changes which stops share a tour.
Those choices interact with routing, but crediting all three changes to route sequencing would misstate the business case.
A route is evaluated from its actual start to its actual end, including required return or handoff travel.

The decision is usually whether to pilot a route policy, not whether to accept an abstract optimum.
An analyst supplies distance and time estimates, uncertainty, safety constraints, and operational guardrails.
Operations owns feasibility and worker consultation; finance owns the conversion of time into realizable dollars.
Customer service owns the service thresholds that a route change cannot violate.

## 2. Learning outcomes

After this lesson you can:

1. **Draw** a directed warehouse graph with pick access points, cross-aisles, and start and end nodes; exercise 1 checks it.
2. **Calculate** complete route distance and time without dropping the return leg; exercise 2 checks it.
3. **Compare** feasible sequences and identify when the shorter route is not the faster route; exercise 3 checks it.
4. **Build** a repeatable shortest-path and route enumeration calculation; exercise 4 checks it.
5. **Estimate** capacity and financial value without counting noncash time as realized savings; exercise 5 checks it.
6. **Design** a controlled field pilot with safety, quality, and service guardrails; exercise 6 checks it.
7. **Diagnose** noisy route and location data and state limits; exercise 7 checks it.
8. **Critique** a vendor or internal claim about travel reduction; exercise 8 checks it.

## 3. The core mental model

```text
Released pick lines + inventory + layout version + rules
                         |
                         v
          map each required pick to an access node
                         |
                         v
       feasible network -> pairwise shortest-path costs
                         |
                         v
       route policy -> ordered visits -> executable walk
                         |
              +----------+-----------+
              v                      v
        predicted travel       safety/service checks
              |                      |
              +----------+-----------+
                         v
          field outcomes -> decision and model revision
```

The graph represents legal movement, not the rack numbering scheme.
Nodes mark intersections, stations, and pick access points.
Edges represent traversable aisle segments and contain distance, direction, and any applicable restriction.
A route visits each required access node and starts and ends at the specified operational nodes.
The shortest path between two stops may pass through another stop; passing it is not automatically a pick.
The executable route expands each stop-to-stop transition into legal graph edges.

Distance is a useful intermediate output, not the outcome the customer buys.
Time equals movement time plus handling, waiting, search, exceptions, and other task components within the chosen boundary.
A shorter route may be slower if its narrow crossing is congested or requires extra equipment maneuvers.
A saving in task minutes becomes a business outcome only if the freed capacity can be used or labor spend can actually change.
Keep prediction, plan, target, and commitment separate: the model predicts, the manager plans, a target expresses an aim, and a commitment is an approved obligation.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Pick line | One requested item/location instruction; lines | Assuming one line equals one physical unit |
| Stop | One required location visit; stops | Treating repeated lines at one location as necessarily separate visits |
| Tour | A complete start-to-end route; tours | Omitting its return or handoff leg |
| Access node | Graph point from which a location can be serviced | Placing it through a rack face |
| Edge | Traversable segment joining two graph nodes; meters | Drawing a legal edge across a barrier |
| Directed edge | Movement permitted in one specified direction | Assuming the reverse movement is also allowed |
| Cross-aisle | Connector across storage aisles | Assuming every rack gap is a public passage |
| Depot | Route origin or required end station | Assuming start and end are always identical |
| Distance | Sum of traversed edge lengths; meters | Measuring Euclidean straight-line distance through racks |
| Travel time | Time spent moving within a defined boundary; seconds | Calling total task duration travel time |
| Dwell | Time stationary at a stop; seconds | Assuming every stationary interval is productive handling |
| Shortest path | Least-cost legal path between two nodes under stated edge weights | Confusing it with the best ordering of all required stops |
| Route sequence | Order in which required stops are serviced | Confusing a visit order with expanded walking directions |
| Optimal route | Best feasible route under a specified objective and model | Claiming practical optimality with missing constraints |
| Heuristic | Rule producing a feasible candidate without an optimality guarantee | Calling its result exact |
| S-shape | Policy that traverses selected aisles using a prescribed pattern | Assuming it is optimal for every layout |
| Return policy | Entering and exiting selected aisles from the same end when applicable | Forgetting depot return in the full tour |
| Batch | Orders combined for execution | Assuming batching never requires sorting |
| Cluster | Multiple orders picked together while kept separate | Ignoring cart capacity and separation errors |
| Congestion | Interaction that delays movement; seconds per tour or edge | Treating it as an invariant distance penalty |
| Released capacity | Time made available for other work; hours per period | Equating it with a cash saving |
| Loaded labor rate | Relevant cost per paid labor hour; USD/hour | Applying it to time that cannot be removed or redeployed |
| Mix adjustment | Comparison standardized for demand or task composition | Comparing two unmatched shifts as a causal result |
| Route compliance | Share of assigned route instructions followed under a defined denominator | Treating every deviation as worker error |
| Exception | Event preventing normal execution, such as an empty slot | Silently deleting its time from a comparison |
| Cutoff | Latest operational time for a downstream service commitment | Substituting average completion time for on-time performance |

## 5. The mechanics, step by step

### 5.1 Define the decision and boundary

Write the operational decision in a sentence: whether to change the stop sequence for a defined class of tours.
Name the building, zones, shift, equipment, picking method, and release window.
Specify whether the route begins at a staging station, a previous task location, or a battery station.
Specify whether it ends at pack, a conveyor, the start station, or another handoff.
Include every mandatory leg between those boundaries.
A route that ends at pack must not be evaluated as if the picker vanished at the final shelf.

Choose the primary outcome before seeing results.
Travel seconds per completed tour is appropriate when the route is the intervention and task composition is stable.
Total paid hours, on-time completion, safety, and pick accuracy are useful business outcomes or guardrails.
Do not divide by units when unit quantity changes but the physical stop count does not.
Do not divide by completed tours alone when canceled or abandoned tours create material work.
State whether exceptions and breaks are included and apply the same rule to both policies.

### 5.2 Build a walkable graph

Obtain a current facility drawing and validate it with a floor walk.
Create nodes at cross-aisle intersections, stop access points, route origins, destinations, and controlled doors.
Split an aisle edge at each pick access point when the precise access location matters.
Measure centerline travel distance, not nominal rack length if turns or access offsets are significant.
Record meters and measurement date; do not mix feet and meters without conversion.
Use directed arcs for one-way movement and remove prohibited paths entirely.
Represent a two-way aisle by arcs in both directions where equipment and rules allow both movements.
Do not use a very high penalty as a substitute for a hard safety prohibition if the algorithm might still select it.

Store graph version, effective interval, equipment class, node identifiers, and source drawing.
Model closures as time-bounded changes and preserve historical versions for backtesting.
Treat vertical travel, elevators, and mezzanine stairs explicitly if they belong in scope.
A forklift route and pedestrian route can require different edges even on the same map.
For a blocked route, provide an operational fallback rather than an impossible instruction.
Have a supervisor and safety representative sign off on the walkable network before optimization.

### 5.3 Translate the work into stops

Extract released pick lines and join each to the location valid when the task was released.
A location master may change after a relocation, so an unversioned present-day join can create a false historical route.
Keep order, wave, batch, tote, equipment, priority, and required quantity where available.
Map the actual pick face, not just the stock-keeping unit (SKU), to a graph access node.
One SKU can occupy several locations and one location can contain several SKUs.
A stockout substitution can change the feasible stop set after release.

Aggregate multiple lines at the same access node only if they can be serviced in one visit without violating order separation or precedence.
Retain the original lines so that dwell, accuracy, and task count remain auditable.
If a node cannot be mapped, report the missing work and correct or exclude it under a predeclared rule.
Never silently set an unmapped stop's distance to zero.
Preserve canceled, short, and reassigned work with explicit statuses.
The unit of analysis is a complete tour, not a row in the pick-line table.

### 5.4 Construct the cost matrix

Compute legal shortest paths between the start, all required access nodes, and the end.
Distance weights yield a shortest-distance matrix; time weights yield a shortest-time matrix under the assumed conditions.
For directed edges, cost from A to B can differ from cost from B to A.
An unreachable pair indicates an infeasible tour or a graph defect, not a zero-cost arc.
Retain the actual node path as well as its cost so that directions can be reconstructed.
If edge travel time changes with traffic, a single static matrix is an approximation.
Avoid estimating walking speed from a task duration that includes dwell or waiting.

### 5.5 Choose a route policy

For a small stop set, enumerate permutations and select the least-cost feasible sequence.
A tour with n distinct stops has n! possible sequences before precedence or other restrictions.
Enumeration is transparent for small n but becomes impractical as n increases.
A nearest-neighbor heuristic selects the cheapest next feasible stop; it can be fast but is not guaranteed to minimize the full tour.
A local improvement swaps or reverses portions of a route and accepts feasible improvement; directed edges require recalculation in the new direction.
Warehouse-specific S-shape and return policies are simple to explain and may suit a regular layout.
Compare policies on the same stop set, start, end, edge weights, and constraints.
If the facility already has a sequence rule, replay it rather than comparing with an arbitrary bad baseline.

Add operational constraints before selecting the winner.
Examples include cold-chain precedence, hazardous-material separation, cart capacity, order-specific service deadlines, and required handoffs.
Precedence can require fragile goods to be picked after heavy goods.
A pick may be available only after replenishment; a nominally short sequence can encounter an empty face.
A strictly distance-minimizing route is not necessarily feasible or preferred under these constraints.
Document objective weights rather than hiding safety or service trade-offs inside a single unexplained score.

### 5.6 Estimate time and queue effects

For a simple unobstructed case, movement time equals distance divided by walking speed.
Add turn, handling, waiting, scanning, and exception components only when they are not already embedded in measured edge time.
Use a consistent route boundary and note which components change under the intervention.
If congestion increases with simultaneous route choices, static independent-tour estimates overstate the benefit.
Measure queues at shared intersections and pack handoffs; route changes can move a bottleneck rather than remove it.
When interactions are material, test through staggered field trials or a calibrated discrete-event simulation.
Report mean and tail behavior, including late-tour and queue measures, not only the fastest route.

### 5.7 Quantify operational value

Multiply the predicted seconds saved per comparable tour by eligible tours in a specified period.
Divide by 3,600 seconds per hour to obtain released hours.
Multiply by an hourly cost only for a scenario where those hours can be avoided, redeployed to work otherwise requiring paid hours, or used to avoid documented incremental staffing.
Separate capacity value, avoided overtime, and verified budget savings.
Do not claim the same hour as both overtime avoidance and regular-hour elimination.
Subtract implementation and recurring costs using the same period and currency.
Check whether pack capacity, replenishment, and order release can absorb the faster work.
If those downstream constraints bind, additional picker capacity may accumulate work in process rather than improve shipments.

### 5.8 Validate and deploy

Replay both policies on held-out, complete historical tours using their contemporaneous map.
Inspect extreme differences and impossible edges with an operator.
Test the recommended instructions on the floor with representative carts and workers.
A controlled pilot should balance shifts, demand mix, worker experience, and volume across policies.
Randomize eligible tours or time blocks where practical and record the assignment.
Predefine sample period, outcome, exclusions, safety stops, and the minimum operationally useful effect.
Track adherence, deviations, exceptions, on-time performance, quality, and worker feedback.
If deviations are common, determine whether the instructions, graph, inventory, or measurement boundary are wrong before attributing fault.
Roll back promptly when a safety or service guardrail is breached.

## 6. Worked example

A hypothetical fulfillment center compares two routes for the same released tour.
The route starts and ends at packing station D and visits three pick faces A, B, and C.
All movements are legal for the cart; the graph is bidirectional and the listed distances are shortest walkable paths in meters.
The complete matrix includes the distances needed for all candidate tours.

| From/to | D | A | B | C |
|---|---:|---:|---:|---:|
| D | 0 | 20 | 30 | 40 |
| A | 20 | 0 | 20 | 30 |
| B | 30 | 20 | 0 | 20 |
| C | 40 | 30 | 20 | 0 |

The existing WMS sequence is D–A–C–B–D.
Its distance is 20 + 30 + 20 + 30 = 100 meters per tour.
The proposed sequence is D–A–B–C–D.
Its distance is 20 + 20 + 20 + 40 = 100 meters per tour.
This first comparison shows no saving: an apparently more orderly sequence is not automatically shorter.

Now suppose the facility's verified A-to-C path is 50 meters, not 30 meters, because a barrier blocks the direct cross-aisle.
To keep all pairwise paths mutually consistent, B-to-C remains 20 meters and A-to-B remains 30 meters.
The revised matrix is:

| From/to | D | A | B | C |
|---|---:|---:|---:|---:|
| D | 0 | 20 | 30 | 40 |
| A | 20 | 0 | 30 | 50 |
| B | 30 | 30 | 0 | 20 |
| C | 40 | 50 | 20 | 0 |

The matrix is compatible with a graph whose relevant direct edges have these lengths.
The baseline is 20 + 50 + 20 + 30 = 120 meters per tour.
The candidate is 20 + 30 + 20 + 40 = 110 meters per tour.
The reduction is 10 meters, or 10 / 120 = 8.33% of baseline travel distance.
The reverse sequence D–C–B–A–D also costs 110 meters; ties are acceptable.
The example's equal start and end are an assumption, not a universal warehouse rule.

Assume moving speed is 1.0 meter per second for both policies and no route-dependent waiting.
Baseline movement takes 120 seconds; candidate movement takes 110 seconds.
The model predicts a 10-second travel saving per comparable tour.
Suppose pick handling and fixed handoff together take 180 seconds per tour under both policies.
Total modeled task time falls from 300 to 290 seconds, or 3.33%, not 8.33%.
The handling figure is an illustrative assumption, not a measured standard.

Suppose 1,800 eligible tours occur per week and the same saving applies to all of them.
Predicted released time is 1,800 × 10 / 3,600 = 5.0 hours per week.
If at most 3.0 hours of overtime can actually be avoided at USD 30 per hour, the weekly avoidable spend is USD 90.
The remaining 2.0 hours are available capacity, not an additional cash saving without a documented use.
If the recurring route-service cost is USD 20 per week, modeled net weekly benefit is USD 70 before implementation costs.
An implementation charge of USD 700 would have a simple payback of 10 weeks at that stable net rate.
This is a scenario, not a forecast, staffing commitment, or guarantee; taxes, discounting, seasonality, and deployment risk are excluded.
If downstream packing cannot accept earlier arrivals, the potential may appear as a longer pack queue rather than improved dispatch.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It enumerates all sequences for the revised, symmetric worked-example matrix.
Distances are assumed to be legal shortest-path distances; the calculation does not derive or validate the underlying graph.
It includes the depot return and prints a deterministic lexical tie-break.
In production, use graph-derived costs, directed distances where appropriate, capacity and precedence checks, and a scalable solver for larger tours.

```python
from itertools import permutations

nodes = ('A', 'B', 'C')
distance = {
    'D': {'D': 0, 'A': 20, 'B': 30, 'C': 40},
    'A': {'D': 20, 'A': 0, 'B': 30, 'C': 50},
    'B': {'D': 30, 'A': 30, 'B': 0, 'C': 20},
    'C': {'D': 40, 'A': 50, 'B': 20, 'C': 0},
}

def route_distance(stops):
    path = ('D',) + tuple(stops) + ('D',)
    return sum(distance[a][b] for a, b in zip(path, path[1:]))

baseline = ('A', 'C', 'B')
best = min(permutations(nodes), key=lambda route: (route_distance(route), route))
base_m = route_distance(baseline)
best_m = route_distance(best)
saved_s = (base_m - best_m) / 1.0  # meters / (meters per second)
weekly_hours = saved_s * 1800 / 3600
print('baseline_m:', base_m)
print('best_route:', 'D-' + '-'.join(best) + '-D')
print('best_m:', best_m)
print('saved_seconds_per_tour:', f'{saved_s:.1f}')
print('released_hours_per_week:', f'{weekly_hours:.1f}')
```

Expected output:

```text
baseline_m: 120
best_route: D-A-B-C-D
best_m: 110
saved_seconds_per_tour: 10.0
released_hours_per_week: 5.0
```

The calculation omits congestion and assumes speed and dwell are unchanged.
The output is a model result, not observed labor savings.
If any pair is unreachable, a production implementation must flag infeasibility rather than assign zero distance.
For an asymmetric matrix, reversing the route can have a different cost.
A path matrix also needs an accompanying edge-by-edge path for execution.

### 7.1 Sensitivity laboratory: translating a route estimate into a decision

Each scenario below is hypothetical and uses a single comparable tour population.
The table-style calculations are separate scenarios, not additive savings.
Each assumes a 1 meter/second walking speed and no change in handling unless a risk says otherwise.
The weekly volume and distance improvement are deliberately varied to expose scaling and decision constraints.
These calculations estimate travel capacity only; none establishes realized payroll savings.

#### Scenario 1: Morning release surge
- Baseline travel: 105 meters per tour.
- Candidate travel: 95 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 9.52% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 900 tours per week.
- Weekly travel time released: 900 × 10 / 3,600 = 2.50 hours.
- Interpretation: 2.50 hours is potential capacity, not verified cash.
- Operational complication: simultaneous tours can share a cross-aisle.
- Evidence to collect: measure crossing wait during the release peak.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 2: Late-shift replenishment
- Baseline travel: 110 meters per tour.
- Candidate travel: 95 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 13.64% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 1,000 tours per week.
- Weekly travel time released: 1000 × 15 / 3,600 = 4.17 hours.
- Interpretation: 4.17 hours is potential capacity, not verified cash.
- Operational complication: restocking vehicles may block a walking edge.
- Evidence to collect: compare routes with replenishment activity logged.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 3: Cold-chain priority
- Baseline travel: 115 meters per tour.
- Candidate travel: 95 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 17.39% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 1,100 tours per week.
- Weekly travel time released: 1100 × 20 / 3,600 = 6.11 hours.
- Interpretation: 6.11 hours is potential capacity, not verified cash.
- Operational complication: temperature exposure can constrain stop order.
- Evidence to collect: record elapsed exposure and required precedence.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 4: Fragile-order handling
- Baseline travel: 120 meters per tour.
- Candidate travel: 115 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 4.17% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 1,200 tours per week.
- Weekly travel time released: 1200 × 5 / 3,600 = 1.67 hours.
- Interpretation: 1.67 hours is potential capacity, not verified cash.
- Operational complication: heavy goods may need to be picked before fragile goods.
- Evidence to collect: enforce load sequence before comparing distance.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 5: Mixed-cart equipment
- Baseline travel: 125 meters per tour.
- Candidate travel: 115 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 8.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 1,300 tours per week.
- Weekly travel time released: 1300 × 10 / 3,600 = 3.61 hours.
- Interpretation: 3.61 hours is potential capacity, not verified cash.
- Operational complication: different carts can use different passage widths.
- Evidence to collect: filter graph edges by cart class.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 6: Pack-station backlog
- Baseline travel: 130 meters per tour.
- Candidate travel: 115 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 11.54% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 1,400 tours per week.
- Weekly travel time released: 1400 × 15 / 3,600 = 5.83 hours.
- Interpretation: 5.83 hours is potential capacity, not verified cash.
- Operational complication: earlier arrivals can wait at the handoff.
- Evidence to collect: track handoff queue and dispatch completion.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 7: Scanner reconnection
- Baseline travel: 100 meters per tour.
- Candidate travel: 80 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 20.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 1,500 tours per week.
- Weekly travel time released: 1500 × 20 / 3,600 = 8.33 hours.
- Interpretation: 8.33 hours is potential capacity, not verified cash.
- Operational complication: buffered confirmations can arrive out of sequence.
- Evidence to collect: compare event timestamps with ingestion timestamps.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 8: Seasonal overflow
- Baseline travel: 105 meters per tour.
- Candidate travel: 100 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 4.76% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 1,600 tours per week.
- Weekly travel time released: 1600 × 5 / 3,600 = 2.22 hours.
- Interpretation: 2.22 hours is potential capacity, not verified cash.
- Operational complication: temporary stock can obstruct an otherwise legal passage.
- Evidence to collect: validate the effective map for the period.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 9: New-hire mix
- Baseline travel: 110 meters per tour.
- Candidate travel: 100 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 9.09% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 800 tours per week.
- Weekly travel time released: 800 × 10 / 3,600 = 2.22 hours.
- Interpretation: 2.22 hours is potential capacity, not verified cash.
- Operational complication: experience may influence both speed and exceptions.
- Evidence to collect: balance experience across treatment groups.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 10: Inventory shortage
- Baseline travel: 115 meters per tour.
- Candidate travel: 100 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 13.04% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 900 tours per week.
- Weekly travel time released: 900 × 15 / 3,600 = 3.75 hours.
- Interpretation: 3.75 hours is potential capacity, not verified cash.
- Operational complication: a stockout can force a substitution and detour.
- Evidence to collect: retain short picks and their replanning time.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 11: Multi-line location
- Baseline travel: 120 meters per tour.
- Candidate travel: 100 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 16.67% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 1,000 tours per week.
- Weekly travel time released: 1000 × 20 / 3,600 = 5.56 hours.
- Interpretation: 5.56 hours is potential capacity, not verified cash.
- Operational complication: several lines can share a single physical stop.
- Evidence to collect: retain line counts while consolidating eligible visits.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 12: Different end station
- Baseline travel: 125 meters per tour.
- Candidate travel: 120 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 4.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 1,100 tours per week.
- Weekly travel time released: 1100 × 5 / 3,600 = 1.53 hours.
- Interpretation: 1.53 hours is potential capacity, not verified cash.
- Operational complication: the worker may finish at a different handoff.
- Evidence to collect: include the actual destination in each route.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 13: One-way crossing
- Baseline travel: 130 meters per tour.
- Candidate travel: 120 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 7.69% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 1,200 tours per week.
- Weekly travel time released: 1200 × 10 / 3,600 = 3.33 hours.
- Interpretation: 3.33 hours is potential capacity, not verified cash.
- Operational complication: the reverse direction may be prohibited.
- Evidence to collect: use a directed cost matrix.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 14: Priority order
- Baseline travel: 100 meters per tour.
- Candidate travel: 85 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 15.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 1,300 tours per week.
- Weekly travel time released: 1300 × 15 / 3,600 = 5.42 hours.
- Interpretation: 5.42 hours is potential capacity, not verified cash.
- Operational complication: a dispatch cutoff can make a short tour late.
- Evidence to collect: measure on-time completion by priority.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 15: Cart capacity
- Baseline travel: 105 meters per tour.
- Candidate travel: 85 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 19.05% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 1,400 tours per week.
- Weekly travel time released: 1400 × 20 / 3,600 = 7.78 hours.
- Interpretation: 7.78 hours is potential capacity, not verified cash.
- Operational complication: a combined batch may require an intermediate unload.
- Evidence to collect: include capacity and unload legs.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 16: Narrow aisle
- Baseline travel: 110 meters per tour.
- Candidate travel: 105 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 4.55% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 1,500 tours per week.
- Weekly travel time released: 1500 × 5 / 3,600 = 2.08 hours.
- Interpretation: 2.08 hours is potential capacity, not verified cash.
- Operational complication: turning and yielding can dominate walking distance.
- Evidence to collect: measure travel and yielding separately.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 17: High pick face
- Baseline travel: 115 meters per tour.
- Candidate travel: 105 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 8.70% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 1,600 tours per week.
- Weekly travel time released: 1600 × 10 / 3,600 = 4.44 hours.
- Interpretation: 4.44 hours is potential capacity, not verified cash.
- Operational complication: retrieval time can differ by shelf height.
- Evidence to collect: stratify stop handling by elevation.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 18: Two-zone handoff
- Baseline travel: 120 meters per tour.
- Candidate travel: 105 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 12.50% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 800 tours per week.
- Weekly travel time released: 800 × 15 / 3,600 = 3.33 hours.
- Interpretation: 3.33 hours is potential capacity, not verified cash.
- Operational complication: changing zones can add a transfer operation.
- Evidence to collect: include the handoff in the service boundary.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 19: Battery exchange
- Baseline travel: 125 meters per tour.
- Candidate travel: 105 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 16.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 900 tours per week.
- Weekly travel time released: 900 × 20 / 3,600 = 5.00 hours.
- Interpretation: 5.00 hours is potential capacity, not verified cash.
- Operational complication: equipment may require a mandatory stop.
- Evidence to collect: model the battery stop when actually required.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 20: Break boundary
- Baseline travel: 130 meters per tour.
- Candidate travel: 125 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 3.85% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 1,000 tours per week.
- Weekly travel time released: 1000 × 5 / 3,600 = 1.39 hours.
- Interpretation: 1.39 hours is potential capacity, not verified cash.
- Operational complication: a paid break can split a measured task interval.
- Evidence to collect: exclude or allocate breaks consistently.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 21: Shift change
- Baseline travel: 100 meters per tour.
- Candidate travel: 90 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 10.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 1,100 tours per week.
- Weekly travel time released: 1100 × 10 / 3,600 = 3.06 hours.
- Interpretation: 3.06 hours is potential capacity, not verified cash.
- Operational complication: a route can be handed to another worker.
- Evidence to collect: link the reassignment instead of dropping the task.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 22: Temporary closure
- Baseline travel: 105 meters per tour.
- Candidate travel: 90 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 14.29% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 1,200 tours per week.
- Weekly travel time released: 1200 × 15 / 3,600 = 5.00 hours.
- Interpretation: 5.00 hours is potential capacity, not verified cash.
- Operational complication: construction can disconnect a previously valid edge.
- Evidence to collect: version the closure and provide a fallback.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 23: Item substitution
- Baseline travel: 110 meters per tour.
- Candidate travel: 90 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 18.18% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 1,300 tours per week.
- Weekly travel time released: 1300 × 20 / 3,600 = 7.22 hours.
- Interpretation: 7.22 hours is potential capacity, not verified cash.
- Operational complication: a replacement can be located in a different aisle.
- Evidence to collect: recompute the route after the substitution.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 24: Heavy goods
- Baseline travel: 115 meters per tour.
- Candidate travel: 110 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 4.35% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 1,400 tours per week.
- Weekly travel time released: 1400 × 5 / 3,600 = 1.94 hours.
- Interpretation: 1.94 hours is potential capacity, not verified cash.
- Operational complication: load weight can change safe speed and handling.
- Evidence to collect: stratify by weight and equipment.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 25: Order separation
- Baseline travel: 120 meters per tour.
- Candidate travel: 110 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 8.33% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 1,500 tours per week.
- Weekly travel time released: 1500 × 10 / 3,600 = 4.17 hours.
- Interpretation: 4.17 hours is potential capacity, not verified cash.
- Operational complication: cluster picking needs tote-level integrity.
- Evidence to collect: audit sorting errors as a guardrail.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 26: Pedestrian crossing
- Baseline travel: 125 meters per tour.
- Candidate travel: 110 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 12.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 1,600 tours per week.
- Weekly travel time released: 1600 × 15 / 3,600 = 6.67 hours.
- Interpretation: 6.67 hours is potential capacity, not verified cash.
- Operational complication: a faster route may cross powered-truck traffic.
- Evidence to collect: review segregation and intersection exposure.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 27: Long tail of tours
- Baseline travel: 130 meters per tour.
- Candidate travel: 110 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 15.38% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 800 tours per week.
- Weekly travel time released: 800 × 20 / 3,600 = 4.44 hours.
- Interpretation: 4.44 hours is potential capacity, not verified cash.
- Operational complication: a few very long tours can move the mean.
- Evidence to collect: report distribution and upper-tail results.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 28: Small-order day
- Baseline travel: 100 meters per tour.
- Candidate travel: 95 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 5.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 900 tours per week.
- Weekly travel time released: 900 × 5 / 3,600 = 1.25 hours.
- Interpretation: 1.25 hours is potential capacity, not verified cash.
- Operational complication: fewer multi-stop tours limit sequencing opportunity.
- Evidence to collect: report savings by stop-count band.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 29: Large-order day
- Baseline travel: 105 meters per tour.
- Candidate travel: 95 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 9.52% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 1,000 tours per week.
- Weekly travel time released: 1000 × 10 / 3,600 = 2.78 hours.
- Interpretation: 2.78 hours is potential capacity, not verified cash.
- Operational complication: many-stop tours increase optimization complexity.
- Evidence to collect: compare solver runtime and feasible solutions.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 30: Unequal aisle length
- Baseline travel: 110 meters per tour.
- Candidate travel: 95 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 13.64% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 1,100 tours per week.
- Weekly travel time released: 1100 × 15 / 3,600 = 4.58 hours.
- Interpretation: 4.58 hours is potential capacity, not verified cash.
- Operational complication: a generic serpentine rule can over-traverse long aisles.
- Evidence to collect: use measured segment lengths.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 31: Blocked pick face
- Baseline travel: 115 meters per tour.
- Candidate travel: 95 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 17.39% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 1,200 tours per week.
- Weekly travel time released: 1200 × 20 / 3,600 = 6.67 hours.
- Interpretation: 6.67 hours is potential capacity, not verified cash.
- Operational complication: another worker can delay access to the location.
- Evidence to collect: record location-level waiting separately.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 32: Different labor rates
- Baseline travel: 120 meters per tour.
- Candidate travel: 115 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 4.17% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 1,300 tours per week.
- Weekly travel time released: 1300 × 5 / 3,600 = 1.81 hours.
- Interpretation: 1.81 hours is potential capacity, not verified cash.
- Operational complication: overtime and regular time have different avoidable costs.
- Evidence to collect: value only the hours actually removed.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 33: Minimum staffing
- Baseline travel: 125 meters per tour.
- Candidate travel: 115 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 8.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 1,400 tours per week.
- Weekly travel time released: 1400 × 10 / 3,600 = 3.89 hours.
- Interpretation: 3.89 hours is potential capacity, not verified cash.
- Operational complication: shorter work may not reduce scheduled headcount.
- Evidence to collect: report capacity separately from payroll.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 34: Wave timing
- Baseline travel: 130 meters per tour.
- Candidate travel: 115 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 11.54% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 1,500 tours per week.
- Weekly travel time released: 1500 × 15 / 3,600 = 6.25 hours.
- Interpretation: 6.25 hours is potential capacity, not verified cash.
- Operational complication: release policy changes the crowding pattern.
- Evidence to collect: hold wave timing fixed in the route test.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 35: Promotional mix
- Baseline travel: 100 meters per tour.
- Candidate travel: 80 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 20.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 1,600 tours per week.
- Weekly travel time released: 1600 × 20 / 3,600 = 8.89 hours.
- Interpretation: 8.89 hours is potential capacity, not verified cash.
- Operational complication: a promotion changes SKU and aisle concentration.
- Evidence to collect: compare within representative workload strata.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 36: Split cases
- Baseline travel: 105 meters per tour.
- Candidate travel: 100 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 4.76% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 800 tours per week.
- Weekly travel time released: 800 × 5 / 3,600 = 1.11 hours.
- Interpretation: 1.11 hours is potential capacity, not verified cash.
- Operational complication: case and each picks may use distinct equipment.
- Evidence to collect: segment the routing population by pick method.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 37: Measurement drift
- Baseline travel: 110 meters per tour.
- Candidate travel: 100 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 9.09% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 900 tours per week.
- Weekly travel time released: 900 × 10 / 3,600 = 2.50 hours.
- Interpretation: 2.50 hours is potential capacity, not verified cash.
- Operational complication: scanner latency may differ across devices.
- Evidence to collect: audit device-specific event lag.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 38: Shared shortcut
- Baseline travel: 115 meters per tour.
- Candidate travel: 100 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 13.04% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 1,000 tours per week.
- Weekly travel time released: 1000 × 15 / 3,600 = 4.17 hours.
- Interpretation: 4.17 hours is potential capacity, not verified cash.
- Operational complication: all routes may compete for the same shortcut.
- Evidence to collect: measure the edge under treatment traffic.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 39: Worker override
- Baseline travel: 120 meters per tour.
- Candidate travel: 100 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 16.67% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 1,100 tours per week.
- Weekly travel time released: 1100 × 20 / 3,600 = 6.11 hours.
- Interpretation: 6.11 hours is potential capacity, not verified cash.
- Operational complication: the worker may know of an unrecorded obstruction.
- Evidence to collect: capture the reason before judging compliance.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 40: Return to charge
- Baseline travel: 125 meters per tour.
- Candidate travel: 120 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 4.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 1,200 tours per week.
- Weekly travel time released: 1200 × 5 / 3,600 = 1.67 hours.
- Interpretation: 1.67 hours is potential capacity, not verified cash.
- Operational complication: the route endpoint may be a charging station.
- Evidence to collect: use the correct end node for that tour.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 41: Quality inspection
- Baseline travel: 130 meters per tour.
- Candidate travel: 120 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 7.69% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 1,300 tours per week.
- Weekly travel time released: 1300 × 10 / 3,600 = 3.61 hours.
- Interpretation: 3.61 hours is potential capacity, not verified cash.
- Operational complication: some items require a mandatory inspection stop.
- Evidence to collect: include inspection precedence and time.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 42: Hazardous separation
- Baseline travel: 100 meters per tour.
- Candidate travel: 85 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 15.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 1,400 tours per week.
- Weekly travel time released: 1400 × 15 / 3,600 = 5.83 hours.
- Interpretation: 5.83 hours is potential capacity, not verified cash.
- Operational complication: incompatible products can restrict shared carts.
- Evidence to collect: validate compatibility before route selection.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 43: Multiple floors
- Baseline travel: 105 meters per tour.
- Candidate travel: 85 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 19.05% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 1,500 tours per week.
- Weekly travel time released: 1500 × 20 / 3,600 = 8.33 hours.
- Interpretation: 8.33 hours is potential capacity, not verified cash.
- Operational complication: an elevator or staircase may govern movement.
- Evidence to collect: represent permitted vertical connections.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 44: Slot relocation
- Baseline travel: 110 meters per tour.
- Candidate travel: 105 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 4.55% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 1,600 tours per week.
- Weekly travel time released: 1600 × 5 / 3,600 = 2.22 hours.
- Interpretation: 2.22 hours is potential capacity, not verified cash.
- Operational complication: the same SKU can move between pick faces.
- Evidence to collect: join to the effective-dated location.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 45: Incomplete tours
- Baseline travel: 115 meters per tour.
- Candidate travel: 105 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 8.70% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 800 tours per week.
- Weekly travel time released: 800 × 10 / 3,600 = 2.22 hours.
- Interpretation: 2.22 hours is potential capacity, not verified cash.
- Operational complication: abandoned work can disappear from completed-tour metrics.
- Evidence to collect: count incomplete tours and their elapsed work.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 46: Maintenance downtime
- Baseline travel: 120 meters per tour.
- Candidate travel: 105 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 12.50% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 900 tours per week.
- Weekly travel time released: 900 × 15 / 3,600 = 3.75 hours.
- Interpretation: 3.75 hours is potential capacity, not verified cash.
- Operational complication: a lift or doorway can be unavailable.
- Evidence to collect: use time-bounded edge availability.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 47: Weekend schedule
- Baseline travel: 125 meters per tour.
- Candidate travel: 105 meters per tour.
- Travel reduction: 20 meters per tour.
- Distance reduction: 16.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 20 seconds per tour.
- Eligible workload: 1,000 tours per week.
- Weekly travel time released: 1000 × 20 / 3,600 = 5.56 hours.
- Interpretation: 5.56 hours is potential capacity, not verified cash.
- Operational complication: staffing and order mix can differ by day.
- Evidence to collect: stratify or randomize within day type.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 48: Demand growth
- Baseline travel: 130 meters per tour.
- Candidate travel: 125 meters per tour.
- Travel reduction: 5 meters per tour.
- Distance reduction: 3.85% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 5 seconds per tour.
- Eligible workload: 1,100 tours per week.
- Weekly travel time released: 1100 × 5 / 3,600 = 1.53 hours.
- Interpretation: 1.53 hours is potential capacity, not verified cash.
- Operational complication: released capacity may absorb work without cutting payroll.
- Evidence to collect: identify the specific avoided future hiring.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 49: Labor agreement
- Baseline travel: 100 meters per tour.
- Candidate travel: 90 meters per tour.
- Travel reduction: 10 meters per tour.
- Distance reduction: 10.00% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 10 seconds per tour.
- Eligible workload: 1,200 tours per week.
- Weekly travel time released: 1200 × 10 / 3,600 = 3.33 hours.
- Interpretation: 3.33 hours is potential capacity, not verified cash.
- Operational complication: changes to assignment or measurement may need review.
- Evidence to collect: involve employee representatives and human resources.
- Decision rule: retain the current policy if safety or service guardrails fail.

#### Scenario 50: Service deadline
- Baseline travel: 105 meters per tour.
- Candidate travel: 90 meters per tour.
- Travel reduction: 15 meters per tour.
- Distance reduction: 14.29% of the baseline.
- Assumed movement speed: 1 meter per second.
- Modeled movement saving: 15 seconds per tour.
- Eligible workload: 1,300 tours per week.
- Weekly travel time released: 1300 × 15 / 3,600 = 5.42 hours.
- Interpretation: 5.42 hours is potential capacity, not verified cash.
- Operational complication: a distance objective can delay the most urgent order.
- Evidence to collect: track deadline misses and enforce priority rules.
- Decision rule: retain the current policy if safety or service guardrails fail.


## 8. Data, standards, and regulatory context

A defensible analysis joins the tour assignment, location master, movement network, pick events, exceptions, and labor records at compatible timestamps.
A WMS event records a system action, which might not coincide with the instant a worker physically reaches a location.
Scanner timestamps can represent confirmation after handling, not arrival.
Device clocks, offline buffering, and delayed uploads can reorder events.
Record both event and ingestion times, time zone, identifier scope, and source system.
A historical route must use the layout and location assignments effective at the time of that route.
Worker identifiers are sensitive operational data; limit access, retention, and individual-level reporting to legitimate purposes.
Aggregate pilot reporting where feasible and involve the relevant privacy and employee relations teams.

GS1 identification and barcode standards can support unambiguous product and location identification; they do not prescribe an optimal walking path.
A Global Location Number (GLN), where used, identifies a location or party in the GS1 system; the facility still needs an internal geometry and access map.
OSHA's U.S. walking-working surfaces standard at 29 CFR 1910.22 requires, among other things, walking-working surfaces to be maintained in a clean, orderly, and sanitary condition and free of specified hazards; consult the regulation for its exact scope and obligations.
OSHA's powered industrial truck standard at 29 CFR 1910.178 addresses relevant truck operation and training; applicability depends on the equipment and circumstances.
These U.S. federal references are not a substitute for site-specific legal review, other applicable rules, or local jurisdiction requirements.
An optimization objective never authorizes blocking an exit, bypassing a safety rule, or assigning unauthorized equipment.
Regulatory references are current as consulted for this lesson, November 2026; verify changes and jurisdiction before deployment.

## 9. Industry practice and case evidence

Warehouse route design is an established operations-research subject, but published model results depend on the layout and order distribution studied.
Ratliff and Rosenthal's 1983 paper studies order picking in a rectangular warehouse under specific assumptions and provides an efficient optimal algorithm for its stated setting.
It does not establish that an arbitrary modern multi-block warehouse with congestion, one-way rules, and deadlines has the same solution structure.
Petersen's 1997 evaluation compares routing policies under studied conditions; use it to understand policy sensitivity, not as a site-specific performance guarantee.
De Koster, Le-Duc, and Roodbergen's 2007 review surveys order-picking design and control, including routing and related decisions.
The review supports treating layout, storage assignment, batching, and routing as connected decisions, not multiplying their independent percentage savings without checking interactions.

In practice a company may start with a straightforward WMS location sequence because it is easy to maintain and explain.
A more sophisticated graph is useful only when its location mapping, restrictions, and maintenance process are trustworthy.
A field trial can reveal local exceptions absent from a published model: replenishment carts, seasonal overflow, blocked aisles, and variable pick heights.
[INFERENCE] A site with a large share of one-stop tours has less sequencing opportunity than a comparable site with many multi-stop tours, all else equal.
[INFERENCE] A site whose pack station is already saturated may experience queue growth rather than additional output when pick travel falls.
Neither inference is a named-company result or a measured universal effect.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Missing return leg | Routes stop at final shelf | Boundary is incomplete | Inflated benefit | Include destination in every tour |
| Stale map | Directions cross a new barrier | Layout version mismatch | Impossible route | Version and floor-validate graph |
| Wrong direction | Route instructs reverse travel | One-way rule omitted | Unsafe instruction | Directed edges and safety signoff |
| Missing stop | Some tasks lack coordinates | Failed location join | Understated distance | Reconcile lines and reject missing mappings |
| Queue ignored | Walking improves but elapsed time rises | Shared crossing saturates | No throughput gain | Measure queue and phase release |
| Dwell double counted | Predicted totals exceed observed | Edge time includes handling | Biased comparison | Define exclusive time components |
| Cherry-picked baseline | Old rule looks unusually inefficient | Nonrepresentative replay | False business case | Replay actual eligible tours |
| Worker mix confounding | Policy used only by experienced staff | Assignment is not balanced | Misattributed improvement | Randomize or stratify |
| Batching confounding | Stops per tour change in pilot | Intervention changes multiple levers | Route effect unidentifiable | Hold batch policy fixed |
| Unrealizable savings | Payroll is unchanged | Hours cannot be removed | Inflated financial benefit | Separate capacity and avoidable spend |
| Inventory exception | Frequent stockouts cause detours | Static stops are unavailable | Route instructions fail | Record exceptions and support replanning |
| Excessive monitoring | Individuals are ranked from noisy scans | Event time is misinterpreted | Trust and privacy harm | Govern access and use aggregate measures |
| Quality degradation | Wrong-item scans rise | Speed pressure changes behavior | Rework and customer harm | Accuracy guardrail and stop rule |
| Downstream bottleneck | Pack backlog rises | Faster picking feeds fixed capacity | More work in process | Monitor queue and coordinate releases |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test |
|---|---|---|---|
| Graph matches reality | Legal distances drive route costs | Construction or overflow | Floor walk and version audit |
| Stops are correct | Sequence operates on real work | Substitutions or stockouts | Reconcile released and completed lines |
| Speed is comparable | Converts meters to seconds | Equipment or terrain differs | Stratify direct movement observations |
| Dwell is unchanged | Isolates route effect | Reaching order affects handling | Compare stop-level handling |
| Start and end are fixed | Defines tour boundary | Assignment begins elsewhere | Inspect task chaining |
| Congestion is negligible | Static matrix remains valid | Simultaneous flows interact | Measure crossing wait by load |
| Eligible volume persists | Scales savings by period | Seasonality or mix shift | Replay representative periods |
| Overtime is avoidable | Converts time to cash | Minimum staffing or schedules bind | Finance validates roster decisions |
| Randomization balances mix | Supports causal pilot estimate | Overrides and noncompliance | Check assignment and strata |
| Safety rules stay current | Ensures feasibility | Policy or equipment changes | Scheduled safety review |

The worked example is deliberately small enough to enumerate.
Its optimum applies to its supplied matrix, not to every warehouse tour.
Measurement uncertainty, operational interactions, and policy compliance determine whether the predicted difference survives in practice.

## 12. Exercises

### Exercise 1 — Draw the network

A depot D connects to aisle intersections X and Y by 10 and 14 meters, respectively.
X connects to Y by a one-way edge of 8 meters from X to Y; the reverse is prohibited.
Pick face A is 5 meters beyond X on a two-way spur; face B is 6 meters beyond Y on a two-way spur.
Deliverable: a labeled directed graph and the cost of D–A–B–D.
Acceptance: show directions, both spur returns, and the final depot leg; do not cross the forbidden edge backward.
Time: 12 minutes.

### Exercise 2 — Compare tours

Use the revised matrix from section 6.
Compute D–B–A–C–D and D–A–B–C–D.
Deliverable: each leg and total meters, plus the absolute and percentage distance reduction relative to the first route.
Acceptance: include the depot return and state the baseline denominator.
Time: 10 minutes.

### Exercise 3 — Time is not distance

A baseline travels 120 meters at 1 meter per second and waits 5 seconds.
A candidate travels 100 meters at 1 meter per second and waits 30 seconds; other work is identical.
Deliverable: travel-plus-wait time for both and a policy recommendation conditional on these assumptions.
Acceptance: identify the shorter route and the faster route separately.
Time: 8 minutes.

### Exercise 4 — Extend the calculation

Change the example program so the depot return is replaced by an end node E.
Provide directed costs of your choosing and at least one route where reversal has a different cost.
Deliverable: executable code, input matrix, candidate route costs, and chosen route.
Acceptance: evaluate every candidate using D as start and E as end; explain tie handling and infeasible arcs.
Time: 25 minutes.

### Exercise 5 — Value released time

A policy saves 12 seconds on each of 2,400 tours per week.
Only 4 hours per week can avoid overtime at USD 32 per hour; recurring cost is USD 28 per week.
Deliverable: released hours, avoidable gross spend, net weekly benefit, and unmonetized hours.
Acceptance: do not monetize the other hours without evidence of an additional use.
Time: 10 minutes.

### Exercise 6 — Design a pilot

A facility wants to compare its current sequence with a new WMS route.
Deliverable: a one-page experimental protocol with unit of assignment, period, primary metric, at least three guardrails, and rollback ownership.
Acceptance: address task mix, incomplete work, spillovers at crossings, worker involvement, and downstream queues.
Time: 25 minutes.

### Exercise 7 — Audit a route dataset

A data extract contains confirmed picks but not canceled tasks; current location coordinates replace historical coordinates.
Some scans arrive after a device reconnects, and the end station is missing for 15% of tours.
Deliverable: a defect ledger and a defensible analysis population.
Acceptance: explain the bias mechanism and a remediation or explicit limitation for each defect.
Time: 20 minutes.

### Exercise 8 — Critique an improvement claim

A vendor reports a 20% distance reduction from a simulated best case and claims a 20% payroll reduction.
Deliverable: a short decision memo identifying what additional evidence is required before purchase.
Acceptance: cover baseline selection, time composition, feasibility, demand, implementation cost, and labor realizability.
Time: 15 minutes.

## 13. Self-check answers

### Exercise 1

The allowed path is D–X–A–X–Y–B–Y–D.
Its distance is 10 + 5 + 5 + 8 + 6 + 6 + 14 = 54 meters.
Each spur requires an outbound and return leg, and Y–X is not permitted.
The route may pass through Y more than once; that is not a defect.

### Exercise 2

D–B–A–C–D costs 30 + 30 + 50 + 40 = 150 meters.
D–A–B–C–D costs 20 + 30 + 20 + 40 = 110 meters.
The reduction is 40 meters, or 40 / 150 = 26.67% relative to the specified baseline.
This differs from the section 6 baseline, which is why its percentage differs.

### Exercise 3

The baseline takes 120 + 5 = 125 seconds.
The candidate takes 100 + 30 = 130 seconds.
The candidate is 20 meters shorter but 5 seconds slower under the stated wait assumptions.
Prefer the baseline for this narrowly defined time objective unless another constraint changes the decision.

### Exercise 4

A strong solution explicitly constructs D-to-first, inter-stop, and last-to-E legs.
It calculates directed arc costs without assuming symmetry and tests all permutations.
It reports an unreachable path as infeasible, not as a zero.
It includes a reproducible tie-break and demonstrates a reversal difference with actual inputs.

### Exercise 5

2,400 × 12 / 3,600 = 8 released hours per week.
Four avoidable overtime hours at USD 32 per hour produce USD 128 gross weekly savings.
Subtracting USD 28 per week gives USD 100 net weekly benefit before one-time costs.
The other 4 hours are unmonetized capacity, not an additional USD 128 of verified saving.

### Exercise 6

A strong protocol assigns comparable tours or blocks before outcomes are known and records overrides.
It specifies travel or total task seconds per comparable tour as primary and measures accuracy, safety, on-time shipping, and pack queue as guardrails.
It keeps batching and slotting stable, includes exceptions and incomplete work, and addresses interference between policies at shared crossings.
It names a supervisor empowered to pause the test and involves workers in route validation.

### Exercise 7

Missing canceled work can favor policies that generate more unrecorded incomplete work.
Historical coordinates must be reconstructed from effective-dated location records, or results must be restricted and qualified.
Device event time and ingestion time should be retained separately, with clock and offline delay investigated.
Missing end stations prevent complete tour measurement; recover them from task handoffs or exclude affected tours under a declared rule while reporting the excluded share.
A defensible cohort contains reconciled tours with known starts, ends, stops, and valid maps, with selection bias disclosed.

### Exercise 8

A strong memo requests comparable historical baseline tours and real layout and constraint validation.
It distinguishes travel distance from total labor time and asks for a controlled field test including queues and quality.
It requires a period-specific staffing plan showing which hours can actually be avoided, plus recurring and one-time costs.
A simulated 20% distance reduction alone cannot establish a 20% payroll reduction.

## 14. Interview questions

1. How would you model a one-way cross-aisle?
   A strong answer uses directed edges and checks reachability; a weak answer assumes a symmetric distance matrix.
2. What is the difference between a shortest path and an optimal pick tour?
   A strong answer distinguishes two-node pathfinding from ordering all required visits; a weak answer treats them as identical.
3. Why can a shorter route take longer?
   A strong answer includes congestion, turns, and downstream waits; a weak answer assumes constant speed and no interaction as fact.
4. How would you handle an empty pick face?
   A strong answer preserves the exception and defines safe replanning; a weak answer silently deletes the stop.
5. How do you compare a new sequence fairly?
   A strong answer fixes stop sets and boundaries and uses representative or randomized work; a weak answer compares unmatched shifts.
6. When is saved travel time a cash saving?
   A strong answer identifies avoided labor or documented redeployment and costs; a weak answer multiplies all seconds by the wage rate.
7. Why might reverse routes have different costs?
   A strong answer points to directed movement and time-varying congestion; a weak answer assumes the map is always symmetric.
8. What makes a route infeasible?
   A strong answer includes disconnected edges, equipment restrictions, safety, capacity, and precedence; a weak answer checks distance alone.
9. How can a historical location join mislead?
   A strong answer uses effective-dated layouts and locations; a weak answer substitutes current coordinates without qualification.
10. What would make you stop a pilot?
    A strong answer names predefined safety, quality, or service thresholds and an owner; a weak answer waits for the average to improve.

## 15. Key terms

- **Access node:** the legal service point for a pick face; see section 4.
- **Batch:** orders grouped for execution; see section 4.
- **Capacity:** work the system can process under specified conditions; see section 5.7.
- **Congestion:** interaction-induced delay; see section 5.6.
- **Cross-aisle:** connector between aisle segments; see section 5.2.
- **Depot:** a tour origin or destination; see section 4.
- **Directed edge:** a permitted movement in one direction; see section 5.2.
- **Distance matrix:** pairwise legal path costs; see section 5.4.
- **Dwell:** stationary time at a stop; see section 4.
- **Edge:** a traversable segment in a graph; see section 4.
- **Exception:** departure from normal task execution; see section 5.3.
- **Feasibility:** satisfaction of all hard route constraints; see section 5.5.
- **Heuristic:** a practical rule without a general optimality guarantee; see section 5.5.
- **Mix adjustment:** comparison accounting for workload composition; see section 4.
- **Node:** a location in a route graph; see section 5.2.
- **Pick line:** a requested item/location instruction; see section 4.
- **Precedence:** a required ordering between activities; see section 5.5.
- **Released hours:** time made available by the intervention; see section 5.7.
- **Route compliance:** adherence under a defined denominator; see section 4.
- **Sequence:** order of required stop visits; see section 3.
- **Shortest path:** least-cost legal path between nodes; see section 5.4.
- **Stop:** a required location visit; see section 4.
- **Tour:** complete route from start to end; see section 4.
- **Travel time:** movement time within a defined boundary; see section 4.
- **WMS:** warehouse management system; see section 1.

## 16. Sources

- Ratliff, H. D., and Rosenthal, A. S. (1983), “Order-Picking in a Rectangular Warehouse: A Solvable Case of the Traveling Salesman Problem,” *Operations Research* 31(3), 507–521. https://doi.org/10.1287/opre.31.3.507
- Petersen, C. G. (1997), “An evaluation of order picking routing policies,” *International Journal of Operations & Production Management* 17(11), 1098–1111. https://doi.org/10.1108/01443579710177860
- de Koster, R., Le-Duc, T., and Roodbergen, K. J. (2007), “Design and control of warehouse order picking: A literature review,” *European Journal of Operational Research* 182(2), 481–501. https://doi.org/10.1016/j.ejor.2006.07.009
- GS1, [Global Location Number](https://www.gs1.org/standards/id-keys/gln): identifier background; not a routing standard.
- U.S. Occupational Safety and Health Administration, [29 CFR 1910.22, General requirements](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.22).
- U.S. Occupational Safety and Health Administration, [29 CFR 1910.178, Powered industrial trucks](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.178).

## 17. Links

- [Module 13 index](_index.md)
- [Slotting and ABC](06_slotting-abc.md)
- [Picking strategies](07_picking-strategies.md)
- [Capacity and discrete-event simulation](09_capacity-des.md)
- [Module 13 survey and lab sections](../../../course/13_warehousing_fulfillment_labor.md)
