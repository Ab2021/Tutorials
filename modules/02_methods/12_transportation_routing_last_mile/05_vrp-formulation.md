# The Vehicle Routing Problem

> Module 12 — Transportation, Routing, and Last-Mile Analytics
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 12, topics 01–04
> Version 2026.5

*Capacity, time windows, and the formulation choices that matter.*

---

## 1. Why this topic exists

A dispatcher must turn a set of customer commitments into routes that can actually be driven.
The transport manager must decide how many vehicles to buy or hire, what promises to offer,
and when an order should be outsourced rather than forcing it onto an overloaded route.
A routing model helps make those decisions only if its mathematical rules describe the business.

A short route can be expensive when it requires an extra vehicle.
A cheap route can be unacceptable when it misses a contractual delivery window.
A load that fits by weight may not fit by cube, pallet positions, or axle distribution.
A route whose total driving time is legal may still violate a mandatory break or a facility closure.
A single objective value cannot explain any of these failures without the underlying schedule.

The vehicle routing problem (VRP) chooses which vehicle serves each stop and in what order.
The capacitated vehicle routing problem with time windows (CVRPTW) adds load limits and service intervals.
It is a useful starting point, not a complete description of a live transport operation.
Model formulation translates choices and obligations into variables, costs, and constraints.
That translation determines what the solver is allowed to recommend.

Consider a shipper choosing between two delivery plans.
One uses two contracted vehicles and has 80 road kilometres.
The other uses three vehicles and has 65 road kilometres.
If a vehicle costs USD 180 per day and distance costs USD 1 per kilometre,
the first costs USD 440 and the second USD 605 before other costs.
Minimising kilometres alone selects the more expensive plan.
If the first plan breaches a hard customer window, however, it is not a valid alternative.

This topic builds a formulation that makes such tradeoffs explicit.
It distinguishes a solver proving infeasibility from a search that merely found no solution.
It also shows why relaxing a constraint is a proposal requiring operational approval,
not a mathematical licence to violate a safety rule.
The numbers in instructional examples are invented for teaching, not market benchmarks.

The model is a planning aid.
Dispatch retains responsibility for live road conditions, vehicle defects, actual driver hours,
and customer contact.
A feasible model solution is a candidate plan under the data and rules supplied;
it is not proof that every stop will be served in the real world.

## 2. Learning outcomes

After completing this topic you can:

1. **Translate** an order and fleet policy into a routing decision, objective, and constraints.
2. **Define** directed arc, vehicle-use, visit, load, and time variables with consistent units.
3. **Formulate** a delivery-only CVRPTW with depot flow, capacity, time, and subtour protection.
4. **Calculate** route cost, service start, waiting, and remaining capacity for a candidate plan.
5. **Diagnose** a proven infeasible instance using necessary checks and controlled relaxations.
6. **Compare** fleet, window, outsourcing, and service alternatives on a stated economic basis.
7. **Evaluate** solution status, bound, gap, and out-of-sample operational feasibility.
8. **Design** a deployment handoff that separates a plan from a customer commitment.

Exercises 1–8 in section 12 correspond to these outcomes.
Section 13 gives calculations or an assessment rubric for each exercise.

## 3. The core mental model

A routing model is a contract between commercial promises and physical resources.
Orders and promises define obligations; fleet and working time define available resources.
A directed travel matrix connects stops; rules decide which sequences are allowed.
The objective selects among allowed sequences using a comparable cost measure.

```text
Orders + promises + location data + fleet + shift rules
                        |
                        v
          Versioned feasible decision instance
                        |
         +--------------+----------------+
         |                               |
   Hard constraints                Explicit objective
   safety and commitments          money or priorities
         |                               |
         +--------------+----------------+
                        v
          Solver: routes, times, cost, status
                        |
                        v
      Independent route validation and human approval
                        |
                        v
          Dispatch execution and actual events
                        |
                        v
       Audit, learning, and revised planning inputs
```

The diagram has two feedback loops.
Before release, validation can reject a route that violates an omitted rule or uses stale data.
After execution, actual drive and service times inform the next model version.
Neither loop permits quietly rewriting a customer's agreed promise after a late delivery.

### 3.1 Feasibility comes before ranking

A feasible plan satisfies all hard constraints in the model.
An optimal plan is a feasible plan with the best objective value among all feasible plans,
provided the algorithm establishes that fact.
A good incumbent is simply the best feasible plan found so far.
A solver may return an incumbent without a proof of optimality.
It may also stop with no incumbent without proving that a feasible solution does not exist.

A plan is physically or contractually unacceptable if a material real-world condition is absent
from the model even when the solver calls the plan feasible.
For example, a delivery vehicle may meet aggregate payload but exceed an axle limit.
The correct response is to add or independently enforce the missing condition,
not to assert that the solver made the load legal.

### 3.2 A route is an ordered schedule, not a set of dots

A customer has a physical location, a delivery quantity, an earliest and latest service start,
and a duration of unloading.
Vehicle travel time depends on origin, destination, departure time, restrictions, and road conditions.
Arrival, service start, service completion, and departure are different timestamps.
Early arrival can cause waiting; a model must say where waiting is allowed and who pays for it.
If the customer window is defined for completion instead of start, the constraint must change.

The set of stops assigned to a vehicle is not enough to determine feasibility.
Two sequences through the same stops can produce different driving distances and waiting times.
A tight first appointment can require visiting a geographically distant stop before a nearby one.
An apparently inefficient detour can protect a high-value contractual promise.

### 3.3 Three layers of economic choice

At the first layer, decide whether all mandatory orders can be served with available vehicles.
At the second, compare feasible combinations of vehicle use, distance, working time, and paid extras.
At the third, vary controllable policies: appointment windows, fleet size, or outsourcing.
Keep the baseline obligations unchanged when comparing route algorithms.
When obligations change, label the result as a policy scenario, not an algorithm improvement.

### 3.4 Plan, forecast, target, and commitment

A **forecast** estimates future orders, travel times, or service durations.
A **plan** assigns orders and schedules vehicles using the selected input snapshot.
A **target** is a desired performance level, such as 98% on-time delivery.
A **commitment** is a promise accepted by the customer or a binding operational rule.
A forecast is uncertain; a plan can change; a target may be missed;
a commitment must not be relaxed without the authority specified by the business.

### 3.5 Cost is not distance

A distance-only objective can be appropriate for an existing fixed fleet when incremental vehicle,
driver, and service costs are unaffected by route selection.
That is a narrow assumption, not a universal rule.
If using a vehicle incurs a fixed rental, driver shift, or minimum call-out charge,
include that amount once per used vehicle.
If waiting time costs paid driver time, include it once in a time-based cost term.
Do not add both a fully loaded hourly rate and the same wage again as a stop fee.

### 3.6 The unit of choice changes the answer

A daily dispatch model chooses actual orders and available vehicles for a specific operating day.
A tactical model may instead use representative demand days to compare fleet sizes.
A strategic model may estimate customer regions and depot locations before individual orders exist.
Do not interpret one day's best route as a fleet-sizing proof.
Do not publish a strategic average-time route as an executable dispatch schedule.

## 4. Definitions and vocabulary

| Term | Meaning | Common misuse | Unit or convention |
|---|---|---|---|
| Order | Commercial request with quantity and promise | Assuming one order always means one stop | Order ID |
| Stop | A service event at a location | Treating several incompatible appointments as one stop | Stop ID |
| Depot | Start or end facility for a vehicle | Assuming every route returns to its start | Location ID |
| Node | Depot or customer representation in the graph | Confusing node count with road intersections | Index |
| Arc | Directed move from one node to another | Assuming all road moves are symmetric | Ordered pair |
| Travel matrix | Distance and time for each permitted directed arc | Using straight-line distance as road travel | km, min |
| Service time | Time spent performing work at a stop | Omitting unloading or paperwork | min/stop |
| Arrival | Time vehicle reaches a location | Equating arrival with completed service | Clock time |
| Service start | Time unloading or other work begins | Applying a completion window to start | Clock time |
| Waiting | Time between arrival and permitted service start | Treating waiting as free automatically | min |
| Time window | Allowed interval for a defined event | Failing to specify start versus completion | Clock time |
| Capacity | Maximum permitted resource carried | Using gross vehicle weight as payload | kg, m³, pallets |
| Demand | Quantity to deliver or collect | Mixing pickup and delivery signs | kg, m³, pallets |
| Payload | Permitted freight mass for a specific vehicle | Equating it to vehicle gross weight | kg |
| Cube | Physical volume occupied by freight | Assuming volume alone guarantees packing | m³ |
| Fleet | Vehicles and drivers available in the planning horizon | Counting a vehicle that is already committed | Vehicle IDs |
| Fixed vehicle cost | Incremental cost incurred if a vehicle is used | Including sunk fleet cost in every dispatch | USD/vehicle-day |
| Variable cost | Cost that changes with an operational quantity | Counting fuel twice through two rates | USD/km or USD/min |
| Feasible solution | Assignment and sequence satisfying model constraints | Claiming that it necessarily satisfies omitted rules | Boolean |
| Incumbent | Best feasible solution found so far | Calling it optimal without a proof | Objective units |
| Lower bound | Proven optimistic bound for minimisation | Calling a heuristic estimate a valid bound | Objective units |
| Gap | Difference between incumbent and proven bound | Comparing gaps with different denominators | % |
| Infeasible | No solution satisfies all constraints, if proven | Treating a timeout as proof | Solver status |
| Soft window | Window with permitted, penalised violation | Treating any soft penalty as customer consent | min and USD/min |
| Hard window | A violation is forbidden by the model | Calling a window hard but allowing lateness | min |
| Outsourcing | Assigning service to an external provider | Counting it as free fleet capacity | USD/order or route |
| Subtour | Closed customer cycle not connected to the depot | Assuming in/out degree alone prevents it | Node sequence |
| Big-M | Upper bound used to deactivate an implication | Choosing an arbitrary enormous constant | Depends on row |
| Route duration | End time minus start time under a stated definition | Confusing it with driving time alone | min |
| Split delivery | Serving one order using multiple visits or vehicles | Assuming it is allowed by a standard VRP | Quantity/visit |
| Scenario | Coherent alternative set of inputs and policies | Changing several assumptions without labels | Version ID |
| Robustness | Performance under deviations from nominal inputs | Equating it to deterministic feasibility | Chosen risk measure |
| Cost-to-serve | Relevant economic cost attributable to service | Equating revenue with avoided cost | USD/order or period |

## 5. The mechanics, step by step

The formulation below is a delivery-only, single-depot, finite-fleet CVRPTW.
All customers are mandatory, each is visited once, and each route starts and ends at the depot.
Demand is nonnegative; each vehicle departs with the deliveries assigned to it.
Travel time and distance are deterministic and may be asymmetric.
Service windows constrain service start, and waiting is allowed at customers.
These assumptions are deliberately narrow so that every equation can be interpreted.

### 5.1 Step 1 — Write the business decision and horizon

**Input:** a dated list of released orders, accepted windows, and available vehicles.
**Decision:** assign each stop to one vehicle and sequence each vehicle's stops for one day.
**Output:** a dated candidate schedule and a comparison to the current dispatch baseline.

Freeze the order cutoff before solving.
Record whether cancellations after cutoff are removed and whether new urgent orders enter a replan.
Define whether each order is mandatory, optional with a lost-sales consequence, or eligible for outsourcing.
Define a customer window as service-start, arrival, or completion time.
Select one convention and carry it through the model and validation.

A route spanning midnight needs a monotonic time axis, such as minutes from Monday 00:00.
A clock time alone is ambiguous around midnight and daylight-saving transitions.
Convert source timestamps with the applicable time zone and retain original timestamps for audit.
The worked example uses minutes after 08:00 on a single operating day.

### 5.2 Step 2 — Build the graph and permitted fleet

Let `N` be the set of customer stops and `K` the set of available vehicles.
Use separate start depot `s` and end depot `e` nodes even if they represent the same building.
Let `V = {s} ∪ N ∪ {e}` and let `A` contain only physically permitted directed arcs.
Remove arcs entering `s`, leaving `e`, or going from a node to itself.
A vehicle that is not used takes no arc; it is not forced to make a fictitious empty depot trip.

For each customer `i`, record weight demand `q_i`, optional cube demand `v_i`,
service duration `h_i`, and allowed service-start interval `[a_i,b_i]`.
For each vehicle `k`, record payload `Q_k`, cube capacity `V_k`,
fixed incremental cost `F_k`, and shift interval `[S_k,E_k]`.
For each arc `(i,j)` and vehicle `k`, record distance `d_ij` in km,
travel duration `t_ij` in minutes, and any vehicle-specific access restriction.

Road distance and time must use the same location and restriction assumptions as dispatch.
A bridge restriction may make an arc unavailable to a particular vehicle;
remove that vehicle-arc combination instead of giving it a low travel cost.
A one-way road means `t_ij` need not equal `t_ji`.
An inaccessible or unmapped customer should be a data exception, not a zero-minute arc.

### 5.3 Step 3 — Define decision variables

For each permitted arc and vehicle, let `x_ijk = 1` if vehicle `k` traverses `i → j`.
Otherwise `x_ijk = 0`.
Let `y_ik = 1` if vehicle `k` serves customer `i`.
Let `z_k = 1` if vehicle `k` is used.
These are binary variables.

Let `T_ik` be the service-start time at customer `i` for vehicle `k`.
Let `T_sk` and `T_ek` be the start and end depot times for vehicle `k`.
These time variables are continuous in minutes.
A formulation may use integer minutes for implementation convenience;
then rounding must be conservative and documented.
Let `u_i` be a continuous ordering variable used below to prevent disconnected customer cycles.

When `y_ik = 0`, the value of `T_ik` is an auxiliary value, not an actual customer appointment.
The constraints below bound it within the customer's window even for an unvisited vehicle.
That is harmless provided the corresponding big-M deactivates every unused arc implication.
Only report a time where `y_ik = 1`.

### 5.4 Step 4 — Enforce visit and flow rules

Every customer is served once:

```text
sum(k in K) y_ik = 1                                      for i in N
```

For each customer and vehicle, entry and exit equal the visit decision:

```text
sum(j: (j,i) in A) x_jik = y_ik                            for i in N, k in K
sum(j: (i,j) in A) x_ijk = y_ik                            for i in N, k in K
```

A used vehicle leaves the start depot once and enters the end depot once:

```text
sum(j in N: (s,j) in A) x_sjk = z_k                        for k in K
sum(i in N: (i,e) in A) x_iek = z_k                        for k in K
y_ik <= z_k                                                for i in N, k in K
```

If there are no customers, all `z_k` may be zero.
When a customer is incompatible with vehicle `k`, fix `y_ik = 0` and exclude its vehicle arcs.
Do not model a forbidden stop as merely expensive unless an authorised override really exists.

These equations alone permit a connected depot route plus a separate customer cycle.
The next step removes the latter rather than relying on a solver to guess the intent.

### 5.5 Step 5 — Remove disconnected subtours

Let `n = |N|` and bound the ordering variables `1 <= u_i <= n`.
For each distinct customer pair, impose the Miller–Tucker–Zemlin ordering constraint:

```text
u_i - u_j + n * sum(k in K) x_ijk <= n - 1                for i != j in N
```

The sum is zero when no vehicle uses that directed arc and one when it is used.
The visit and flow constraints ensure that no customer has more than one outgoing arc overall.
When the arc is selected, the equation forces `u_j >= u_i + 1`.
A closed customer cycle would require strictly increasing order back to its own start,
which is impossible.
This ordering formulation is correct for this single-visit model, though it can be computationally weak.
Larger implementations may use valid connectivity cuts, route-based formulations, or specialised solvers.

Positive travel or service times can also exclude cycles through time propagation,
but depending on that property is fragile when zero-duration arcs or rounded times appear.
Explicit connectivity makes the intended logic auditable.

### 5.6 Step 6 — Enforce load capacity

For delivery-only routes, total delivered load assigned to a vehicle cannot exceed payload:

```text
sum(i in N) q_i * y_ik <= Q_k * z_k                        for k in K
```

For an independently measured cube constraint, add:

```text
sum(i in N) v_i * y_ik <= V_k * z_k                        for k in K
```

The payload equation is sufficient for aggregate load along a delivery-only route because the load
starts at the sum of assigned deliveries and decreases at every stop.
It is not sufficient for mixed pickup and delivery: load can rise after a pickup.
That case needs load propagation with explicit pickup and delivery quantities and their order.

Cube is not a packing proof.
Pallet shape, stackability, loading order, temperature, segregation, floor positions,
and axle weights can prevent a geometrically legal and safe load despite aggregate capacity.
An actual load-building process must validate those requirements.
The consolidation lesson examines capacity utilisation and freight economics separately.

### 5.7 Step 7 — Enforce time windows and shifts

Bound customer service-start variables:

```text
a_i <= T_ik <= b_i                                         for i in N, k in K
S_k <= T_sk <= E_k                                         for k in K
S_k <= T_ek <= E_k                                         for k in K
```

For every vehicle-arc combination, impose time propagation:

```text
T_jk >= T_ik + h_i + t_ij - M_ijk * (1 - x_ijk)            for (i,j) in A, k in K
```

Use `h_s = 0`; include depot loading time separately if it consumes the shift.
The inequality allows waiting between arrival and service start.
The time at the end depot is at least the prior service start plus service and travel.
If a stop's promise means service completion by `b_i`, use `T_ik + h_i <= b_i` instead.
If waiting is prohibited at a site, model its permissible location explicitly;
the inequality alone does not determine where waiting occurs.

For an arc with bounded origin time `T_ik <= U_ik` and destination time `T_jk >= L_jk`,
a valid implication constant is:

```text
M_ijk = max(0, U_ik + h_i + t_ij - L_jk)
```

For customers, use `U_ik = b_i` and `L_jk = a_j`;
for depot nodes use the vehicle shift bounds.
If other conditional bounds are used, derive `M_ijk` from those bounds instead.
An excessively large constant weakens the relaxation and can cause numerical difficulties.
An undersized constant can incorrectly exclude valid routes.
Where supported, an indicator constraint can express the implication without a user-supplied `M`.

A maximum duty length `D_k` can be added as `T_ek - T_sk <= D_k`.
If `T_sk` is allowed to float, this definition does not automatically count pre-departure loading,
reporting time, or other paid work; include them explicitly if required.
A rule about continuous driving between breaks is not equivalent to total route duration.
Add a properly specified break model or validate and reject routes outside the model.

### 5.8 Step 8 — Choose an economically consistent objective

For a simple incremental-cost objective, minimise:

```text
sum(k in K) F_k * z_k
+ sum(k in K, (i,j) in A) c_km_k * d_ij * x_ijk
+ sum(k in K) c_min_k * (T_ek - T_sk)
```

Here `F_k` is USD per used vehicle, `c_km_k` is USD per km,
and `c_min_k` is USD per paid route minute.
The last term includes travel, service, and any waiting inside the modelled route span.
It does not necessarily equal payroll: minimum shifts, overtime, paid breaks,
and time before departure or after return require separate modelling.

If all vehicle costs are sunk for a short-run dispatch decision, `F_k` can be zero;
that does not make an additional rental or overtime shift free.
Use fixed cost for the economically avoidable vehicle decision being evaluated.
Revenue is usually constant across mandatory all-served plans and can be excluded.
If service is optional or demand changes, compare incremental contribution and service costs consistently.

For a hard safety constraint, do not assign a finite penalty that allows the solver to buy a violation.
For a negotiable commercial window, a soft alternative might define `late_i >= T_i - b_i`
and add `p_i * late_i` to cost, but only if the customer permits the late outcome.
Where a service priority is non-negotiable, a lexicographic objective may first minimise authorised
exceptions, then minimise cost among equally acceptable solutions.
Publish the priority order and the authority to accept exceptions.

### 5.9 Step 9 — Encode real business constraints deliberately

**Compatibility:** refrigeration, access permits, vehicle height, and customer equipment requirements
can be implemented with permitted vehicle-stop pairs and restricted arcs.
Use effective dates: a vehicle certification or customer restriction can change.

**Multiple resources:** weight and cube can each bind.
Pallet count can be another constraint, but integer pallet positions and packing geometry may still require
an independent load plan.
Using the maximum of utilisation percentages as a feasibility test is valid only for the represented
aggregate capacities, not for physical arrangement.

**Service durations:** unloading can depend on quantity, equipment, and customer practice.
If it is modelled as a fixed per-stop duration, record the approximation.
If service duration depends on the vehicle or quantity, index the parameter accordingly.

**Customer priority:** a must-serve order remains mandatory.
An optional order needs an explicit rejection variable and an economically meaningful penalty
or a separate service-priority objective.
A missing order must appear in an exception report, not disappear from the denominator.

**Multiple trips:** the base model permits one route per vehicle.
Reloads require trip-indexed or reload-node modelling and a driver schedule between trips.
Counting the same physical vehicle twice without non-overlap can create impossible simultaneous routes.

**Pickups and returns:** each pickup-and-delivery request needs paired assignment,
precedence, and a route load calculation after each event.
A pickup assigned to a different vehicle from its delivery is invalid unless transfers are authorised.
Returnable containers and reverse logistics consume capacity even when outbound delivery demand falls.

**Driver regulations:** legal driving, on-duty, off-duty, break, and weekly limits depend on jurisdiction
and operation.
A daily duration cap is at most a simplified proxy, not legal advice.
Use current applicable rules and actual driver histories in a dedicated compliance process.

### 5.10 Step 10 — Pre-screen, solve, and interpret status

Before solving, check whether every mandatory order is eligible for at least one vehicle.
For unsplittable delivery, each order must fit at least one eligible vehicle individually.
Total demand must not exceed total available capacity for the single-trip model.
These are necessary checks, not sufficient proofs of feasibility.
Windows, route duration, vehicle compatibility, or packing can still prevent a solution.

For each customer and eligible vehicle, test earliest possible arrival from the start depot
and latest possible return to the end depot.
Eliminate impossible assignments only when the test uses a valid optimistic travel lower bound.
A direct path that is not the shortest feasible road path is not necessarily a lower bound.

Record solver version, parameters, random seed if relevant, wall-clock limit, input snapshot,
objective, incumbent, bound, reported gap, and status.
For a minimisation problem, a proven lower bound is no greater than the optimum.
If the incumbent costs USD 1,000 and a valid lower bound is USD 900,
the absolute gap is USD 100; using incumbent as denominator gives 10%.
Solvers may use another denominator or special treatment near zero: report their definition.
A 10% bound gap does not mean a 10% operational forecast error.

A completed proof of infeasibility means no solution exists within the encoded model.
A timeout with no solution means only that the search has not found one.
A feasible incumbent with a time limit is usable only after independent validation and approval.
A lower bound without an incumbent is not a route plan.

### 5.11 Step 11 — Diagnose infeasibility without breaking obligations

First validate units, missing records, time zones, travel matrix connectivity, and window semantics.
Then examine simple contradictions such as one order heavier than every eligible vehicle.
If available, an irreducible infeasible subsystem or conflict refiner identifies a conflicting subset
of constraints; the subset is model- and algorithm-dependent and need not identify the business root cause.

Run one controlled relaxation at a time.
Examples include adding an authorised vehicle, expanding an adjustable appointment,
allowing approved outsourcing, or raising a proposed overtime limit.
Keep hard legal and safety constraints hard.
For a relaxable customer window, record the affected customer, minutes, authorisation, and economic cost.
Compare the resulting route to the original service promise, not merely the relaxed objective.

A minimum total violation model can reveal one low-penalty compromise.
It does not prove that the selected compromise is fair, legal, or operationally acceptable.
Penalty weights affect which constraints are relaxed and may obscure alternatives.
Present a short list of actionable options to the owner of each constraint.

### 5.12 Step 12 — Validate and release

Recompute every selected route from the exported ordered stops and original inputs.
Check each mandatory stop once, vehicle availability, eligibility, capacity, start and end,
arc access, service-start windows, elapsed time, and the objective breakdown.
Validate packing and compliance controls that were intentionally outside the optimisation model.

The dispatcher reviews exceptions and feasibility under current conditions.
The customer-facing system issues or confirms promises only through its authorised workflow.
Store the released plan separately from solver candidates, manually edited plans, and actual events.
A manual change triggers revalidation of the affected route and its connected resource commitments.

### 5.13 Formulation variants and selection

The arc formulation is transparent and appropriate for teaching and smaller mixed-integer models.
A set-partitioning formulation chooses from feasible whole routes and covers each customer once.
It can exploit route-generation methods, but a restricted route list may omit the best route.
A specialised routing heuristic can return good plans quickly without providing a global optimality proof.
Choice depends on instance size, time available, operational constraints, and the evidence required.

A static matrix assumes travel time does not change with departure time.
A time-dependent model can represent congestion, but it needs reliable time-bucketed data
and consistent travel-time functions.
A stochastic or robust model explicitly considers uncertainty; it requires scenarios or uncertainty sets.
A naive buffer may improve reliability but can also waste capacity or conceal systematic data bias.
Compare reliability and cost under held-out operating days before adopting a buffer policy.

## 6. Worked example

### 6.1 Decision and assumptions

A regional distributor plans one weekday in an illustrative 2026 USD scenario.
It must deliver three orders, A, B, and C, from one depot.
Two identical vehicles are available, each with 10 units of payload capacity.
The unit can represent a consistently measured load unit; the example omits cube and packing.
The company pays USD 100 per used vehicle plus USD 2 per road kilometre.
Driver time is assumed included in that fixed vehicle charge for this example only.
There are no tolls, overtime, or outsourcing fees in the base comparison.

The operating day starts at 08:00, and times below are minutes after 08:00.
A service-start window is hard, waiting is allowed, and every customer must be served once.
Each vehicle must return to the depot by minute 120.
Travel takes one minute per road kilometre in this constructed matrix.

| Stop | Demand, load units | Service start window, min | Service time, min |
|---|---:|---:|---:|
| A | 4 | 10–35 | 10 |
| B | 5 | 30–60 | 10 |
| C | 6 | 35–75 | 10 |

The depot is denoted D. Distances are symmetric in this example, unlike the general model.

| From / to | D | A | B | C |
|---|---:|---:|---:|---:|
| D | 0 | 10 | 20 | 25 |
| A | 10 | 0 | 10 | 15 |
| B | 20 | 10 | 0 | 10 |
| C | 25 | 15 | 10 | 0 |

Total demand is 15 load units.
One vehicle cannot serve all stops because 15 exceeds 10.
Two vehicles have total capacity 20, so aggregate capacity is sufficient but not a feasibility proof.
The eligible pairs by load are A+B = 9, A+C = 10, and B+C = 11 units.
Therefore B and C cannot share a vehicle without splitting or changing capacity.

### 6.2 Candidate plan P: A then B, and C alone

Vehicle 1 leaves D at minute 0 with 9 load units.
It arrives at A at minute 10 and begins service at minute 10, inside A's 10–35 window.
It finishes at minute 20 with 5 load units remaining.
It travels 10 minutes to B, arrives at minute 30, and starts at minute 30.
B is inside its 30–60 window; service ends at minute 40.
The vehicle returns 20 minutes later, at minute 60, empty.
Its distance is 10 + 10 + 20 = 40 km.

Vehicle 2 leaves D at minute 0 with 6 load units.
It arrives at C at minute 25 and waits 10 minutes for the window to open at minute 35.
Service ends at minute 45, and the 25-minute return ends at minute 70.
Its distance is 25 + 25 = 50 km.
Both returns are before the minute-120 depot deadline.

Plan P uses two vehicles, travels 90 km, and costs:

```text
2 vehicles × USD 100/vehicle + 90 km × USD 2/km = USD 380
```

Waiting is shown as an operational fact but creates no extra charge under the example's stated
fixed vehicle-cost assumption.
If drivers are paid additional variable time, the economics must be recalculated.

### 6.3 Candidate plan Q: A then C, and B alone

Vehicle 1 leaves D at minute 0 with 10 load units.
It starts at A at minute 10, finishes at minute 20, and reaches C at minute 35.
C begins service at minute 35 and finishes at minute 45.
The vehicle returns at minute 70, having driven 10 + 15 + 25 = 50 km.

Vehicle 2 leaves D at minute 0 carrying 5 units and reaches B at minute 20.
It waits until B's window opens at minute 30, serves B until minute 40,
and returns at minute 60 after 20 km of return travel.
Its distance is 20 + 20 = 40 km.

Plan Q also costs USD 200 + USD 180 = USD 380.
Both plans satisfy the stated aggregate capacities and start windows.
They have the same cost under the stated objective, although their load mix differs.
A dispatcher might prefer P or Q based on an additional criterion such as loading order,
customer familiarity, or risk; that preference is not implied by the current objective.

### 6.4 A tempting but infeasible alternative

Putting B and C on the same vehicle appears geographically attractive because their arc is 10 km.
The assigned demand is 5 + 6 = 11 units against a 10-unit capacity.
The plan is infeasible even before checking its timing.
The fact that a vehicle reaches B before C does not reduce its initial load below 11:
both deliveries were on board when it left the depot.
If a vehicle with 12-unit capacity is available, the plan must be recalculated with that vehicle's cost.

A single vehicle visiting all three stops would use fewer fixed vehicle charges,
but its 15-unit initial load exceeds the 10-unit capacity.
An apparently lower objective for that route has no bearing on the base decision.

### 6.5 Policy scenario: add an authorised capacity unit

Suppose vehicle 2 can be replaced by a 12-unit vehicle for an incremental USD 30 per day.
This is a separate scenario, not a change to the baseline.
Let the 12-unit vehicle serve B then C, carrying 11 units, while the original vehicle serves A.
B starts at minute 30 after arrival at minute 20 and 10 minutes waiting.
After B's service ends at minute 40, it reaches C at minute 50 and serves until minute 60.
It returns to D at minute 85.
The B–C route distance is 20 + 10 + 25 = 55 km.
The A-only route distance is 10 + 10 = 20 km.

The combined cost is two base vehicle charges of USD 100,
plus the USD 30 upgrade, plus 75 km at USD 2/km:

```text
USD 200 + USD 30 + USD 150 = USD 380
```

The upgrade saves 15 km or USD 30 in distance cost and costs USD 30.
The modelled savings are therefore zero, before omitted operational differences.
The break-even upgrade cost is USD 30 per day under these assumptions.
If the upgrade cost were USD 20, the scenario would save USD 10;
if USD 40, it would cost USD 10 more.
A larger vehicle must also be permitted at both customer locations and fit the physical load.

### 6.6 Policy scenario: an infeasible window

Now suppose C's service must start by minute 20, with all other baseline data unchanged.
The shortest direct depot-to-C travel time is 25 minutes and no vehicle departs before minute 0.
Every vehicle therefore reaches C no earlier than minute 25.
The instance is infeasible under that deadline, even if total vehicle capacity is increased.

Possible operational remedies include a customer-approved later start,
an earlier authorised depot departure if the shift and facility permit it,
or a qualified external provider with a compatible location and service commitment.
Each option changes a business condition and needs its own cost and approval.
A model must not invent an earlier departure while keeping the same shift constraints.

### 6.7 Decision note

The base data support two equally priced feasible plans, P and Q, at USD 380 each.
The larger-vehicle scenario ties them under the stated incremental USD 30 upgrade cost.
The tightened C window cannot be met from this depot under the stated shift start.
No claim is made that these nominal schedules will survive unexpected traffic or loading delays.
The manager should release a plan only after checking physical loading, driver eligibility,
current travel times, and customer-specific rules omitted from the example.

## 7. Code, query, or calculation

The following standard-library Python enumerates the small worked instance.
It is an independent candidate-plan evaluator, not a scalable VRP solver.
It enumerates all assignments to two labelled vehicles and all stop orders,
checks capacity and service windows, computes waiting and return time,
and reports the lowest-cost feasible plans.
The deterministic matrix, identical vehicles, and cost assumptions are exactly those in section 6.
An unused vehicle incurs no fixed charge.
Run with Python 3; no third-party package or network connection is needed.

```python
from itertools import permutations, product

DEPOT = "D"
STOPS = ("A", "B", "C")
DEMAND = {"A": 4, "B": 5, "C": 6}
WINDOW = {"A": (10, 35), "B": (30, 60), "C": (35, 75)}
SERVICE = {"A": 10, "B": 10, "C": 10}
DIST = {
    "D": {"D": 0, "A": 10, "B": 20, "C": 25},
    "A": {"D": 10, "A": 0, "B": 10, "C": 15},
    "B": {"D": 20, "A": 10, "B": 0, "C": 10},
    "C": {"D": 25, "A": 15, "B": 10, "C": 0},
}
CAPACITY = 10
END = 120
FIXED_USD = 100
USD_PER_KM = 2


def evaluate(route):
    """Return (distance, finish, details), or None for an infeasible route."""
    if not route:
        return (0, 0, [])
    load = sum(DEMAND[s] for s in route)
    if load > CAPACITY:
        return None
    time = 0
    distance = 0
    previous = DEPOT
    details = []
    for stop in route:
        distance += DIST[previous][stop]
        arrival = time + DIST[previous][stop]  # one minute per km
        earliest, latest = WINDOW[stop]
        start = max(arrival, earliest)
        if start > latest:
            return None
        details.append((stop, arrival, start, start - arrival))
        time = start + SERVICE[stop]
        previous = stop
    distance += DIST[previous][DEPOT]
    finish = time + DIST[previous][DEPOT]
    if finish > END:
        return None
    return (distance, finish, details)


best = None
plans = set()
for assignment in product(range(2), repeat=len(STOPS)):
    groups = [tuple(s for s, k in zip(STOPS, assignment) if k == v)
              for v in range(2)]
    for first in permutations(groups[0]):
        for second in permutations(groups[1]):
            routes = (first, second)
            evaluations = tuple(evaluate(r) for r in routes)
            if any(e is None for e in evaluations):
                continue
            distance = sum(e[0] for e in evaluations)
            cost = FIXED_USD * sum(bool(r) for r in routes) + USD_PER_KM * distance
            # Canonicalise identical-vehicle labels, not the order within a route.
            canonical = tuple(sorted(routes))
            if best is None or cost < best:
                best, plans = cost, {canonical}
            elif cost == best:
                plans.add(canonical)

if best is None:
    print("No feasible plan found by complete enumeration of this instance")
else:
    print(f"Best cost: USD {best}")
    for plan in sorted(plans):
        print(" | ".join("-".join((DEPOT, *route, DEPOT)) for route in plan))
```

Expected output is:

```text
Best cost: USD 380
D-A-B-D | D-C-D
D-A-C-D | D-B-D
```

Enumeration is complete only for the explicitly coded assumptions and this finite instance.
It becomes impractical as the number of customers grows.
The result is not a proof that the same cost is achievable under unmodelled packing,
vehicle availability, legal driving, or stochastic traffic constraints.
If windows are changed, update the expected output rather than treating the original answer as a test oracle.

### 7.1 Manual independent check

For each exported route, compute the initial load as the sum of its deliveries.
At each stop subtract its delivered amount and check that the remaining load is nonnegative.
Compute arrival from previous departure plus directed travel time.
Compute service start as the maximum of arrival and window opening.
Check that service start does not exceed the window closing.
Add service time to obtain departure and repeat until the depot return.
Sum vehicle charges and selected arc costs, then reconcile against the model objective.
This manual protocol detects many errors that a single objective-value comparison cannot.

## 8. Data, standards, and regulatory context

### 8.1 Order and location records

At minimum, the planning extract needs an immutable order ID, stop ID, service address,
coordinates or a validated road-network location, demand by resource,
service duration, service window, and the source and timestamp of the promise.
The same physical address can contain several delivery docks with different access rules.
Conversely, several orders can be combined into one stop only when their service conditions agree.

Keep requested, confirmed, and revised windows as separate events.
A customer request is not automatically a carrier commitment.
A late revision must not overwrite the original commitment used for performance reporting.
If an address is geocoded to a road segment rather than its actual entrance,
validate whether trucks can reach the loading entrance from that road segment.

### 8.2 Matrix provenance

Store matrix provider or routing engine, map version, generation date,
vehicle profile, traffic assumption, units, and unreachable-arc policy.
A matrix made for passenger cars can route heavy vehicles through prohibited roads.
A free-flow travel-time matrix is not an estimate of weekday peak-hour driving time.
Record which matrix cells are missing and whether fallback values are estimates or forbidden arcs.

Treat distance and time as separately measured attributes.
The shortest road distance can have a longer travel time than a longer motorway route.
If distance and time come from different paths, their combination might not describe any real path.
Use a consistent route policy or preserve the route alternative behind each matrix pair.

### 8.3 Fleet and driver availability

A fleet master should identify capacity, dimensions, access restrictions, and effective dates.
Vehicle capacity must reflect payload after relevant equipment and operating conditions,
not the manufacturer's gross vehicle weight by itself.
Driver records must reflect actual availability and applicable historical duty information.
A vehicle and a qualified driver are separate resources even if an implementation pairs them in advance.

The United States Federal Motor Carrier Safety Administration publishes hours-of-service rules
for covered commercial drivers; applicability and exceptions depend on the operation.
The FMCSA summary linked in section 16 distinguishes driving, on-duty, rest, and break requirements.
A VRP route duration parameter by itself does not establish regulatory compliance.
For another jurisdiction use the applicable authority and current rules rather than importing US limits.

### 8.4 Messages and identifiers

GS1 standards provide identifiers and business-message conventions for trading partners;
they do not determine the optimisation objective or guarantee accurate source data.
A Global Location Number can help identify a party or location when trading partners agree on its use.
An internal stop ID is still needed to distinguish multiple service events at one location.
Electronic data interchange messages can transmit shipment and status information,
but the relevant implementation guide determines which events and qualifiers are present.

Retain the original message, mapping version, event timestamp, and processing timestamp.
A status sent late should not silently change the time at which an event actually occurred.
Reconciliation should identify duplicate orders, missing windows, and conflicting revisions.
Do not infer consent to a changed delivery window solely from an updated machine record.

### 8.5 Privacy and retention

Customer addresses, contact information, and driver traces can be personal data.
Apply the organisation's applicable privacy and retention policy, restrict access,
and provide only the detail necessary for the intended planning or audit purpose.
A route optimisation case study does not prescribe a jurisdiction-independent retention period.
Protect customer-identifying logs in non-production environments and examples.

### 8.6 Quality controls

A daily control report should count orders received, excluded, outsourced, planned, and released.
Their categories must reconcile, including cancellation and duplicate rules.
Report missing dimensions, stale fleet availability, invalid windows, and unreachable locations separately.
Compare forecast travel times and service durations to actuals by market, time, and stop type.
An aggregate average can hide systematic underprediction for a particular customer class.

## 9. Industry practice and case evidence

### 9.1 UPS and ORION

UPS describes ORION, its On-Road Integrated Optimization and Navigation technology,
as a route optimisation system used to help drivers choose delivery routes.
The company's public description discusses continually updated routes and operational data.
This is company evidence about its own programme, not an independent estimate of benefits
for every carrier or proof that a particular mathematical formulation is in use.
See the UPS source in section 16 for the company's description.

The transferable lesson is the operational feedback loop:
a route is calculated using available information and must be reconciled with execution.
The public description does not establish the objective weights or constraints used in this lesson.
**Inference:** dynamic updating can improve responsiveness when new information is reliable,
but it also requires change control and driver-facing instructions.
No numerical UPS saving is assumed in the worked example.

### 9.2 Public benchmark instances

Solomon's published vehicle routing and scheduling work introduced benchmark problems with
time-window and capacity features for comparing algorithms.
The instances are useful for reproducible research because inputs and feasibility can be shared.
They do not by themselves represent a particular shipper's loading, regulatory, or customer contract rules.
An algorithm's rank on those instances is not an implementation guarantee at a live depot.

The original paper evaluates algorithms under the model and instances described in the paper.
It does not establish that a solver winning a benchmark will minimise a company's full cost-to-serve.
**Inference:** a production evaluation should retain benchmark tests for regressions and add
independently validated instances reflecting its own operating constraints.

### 9.3 Published formulation research

The vehicle routing problem has a long research literature, beginning with work by Dantzig and Ramser
on truck dispatching and followed by many exact and heuristic methods.
Research formulations expose tradeoffs between solution quality, model fidelity, and solve time.
A paper's numerical results are conditional on its instance set and computational setup.
They should not be transcribed as expected business savings for a different network.

### 9.4 Operational adoption

A planner usually needs a route sheet, exceptions, and a reason for changes,
not just a solver objective value.
A transport manager needs the incremental cost relative to a comparable baseline.
A customer-service owner needs affected promises and approval evidence.
A finance owner needs to know which costs are avoidable in the decision horizon.
An implementation should agree these outputs before tuning an optimiser.

A meaningful pilot compares like-for-like orders, fleet availability, windows, and cutoff times.
Where randomisation is impractical, document selection and seasonal differences.
Report both planned and actual cost, service outcomes, and manual changes.
The observed difference is not automatically causal if the routes were run on different days or markets.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Distance-only objective | More routes with fewer km | Vehicle charge omitted | Total spend rises | Include incremental vehicle cost |
| Missing demand | Orders with zero load | Bad master mapping | Overloaded vehicle | Validate units and completeness |
| Cube omitted | Low weight, full trailer | Only mass constrained | Freight cannot fit | Add cube and loading validation |
| Packing ignored | Aggregate cube fits, pallets do not | Geometry omitted | Dock rejection | Physical loading check |
| Wrong window event | Arrival reported as completion | Ambiguous contract field | Missed promise | Define event and map explicitly |
| Wrong time zone | Routes late by fixed hours | Local and UTC mixed | Invalid schedule | Preserve zone and test conversion |
| Unreachable arc set to zero | Implausible short route | Missing-value default | Impossible road plan | Flag and block missing arcs |
| Asymmetry discarded | Return times inconsistent | Matrix symmetrised | Late or illegal route | Preserve directed arcs |
| Shift is too simple | Planned route violates a break | Duration used as legal proxy | Compliance risk | Model or independently validate breaks |
| Duplicate vehicles | Overlapping route assignments | Reloads treated as separate fleet | Resource collision | Enforce non-overlap |
| Hidden dropped orders | Total cost falls abruptly | Mandatory stops removed | Service failures | Reconcile all order statuses |
| Arbitrary big-M | Slow solve or false infeasibility | Weak or invalid implication | Wrong or delayed decisions | Derive bounds or use indicators |
| Subtour omitted | Customer cycle away from depot | Degree constraints insufficient | Undriveable solution | Explicit connectivity constraints |
| Timeout called infeasible | No route after short run | Search status misread | Unnecessary escalation | Distinguish no incumbent from proof |
| Bound called a plan | Dashboard reports lower bound | No feasible incumbent | Dispatch has no schedule | Require validated route export |
| Free overtime | Low plan cost, high payroll | Paid time excluded | Cost understated | Add incremental time rules |
| Stale availability | Assigned vehicle under repair | Asset snapshot too old | Day-of-service disruption | Timestamp and confirm fleet data |
| Inconsistent comparison | Savings appear only on quiet days | Different order or service mix | False benefit claim | Like-for-like baseline and cohorts |
| Penalty substitutes for permission | Late service bought cheaply | Hard promise made soft | Contract breach | Authorise and report exceptions |
| Forecast treated as certainty | Frequent late routes | Nominal travel too optimistic | Service degradation | Backtest and stress scenarios |

Each row is an investigation prompt, not a claim that any particular organisation has the defect.
The first response should preserve the original instance and the solution status before editing data.
Record the cause, owner, correction, and whether previously released plans are affected.

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test |
|---|---|---|---|
| All stops mandatory | Defines service obligation | Optional orders or cancellations | Reconcile order and cancellation status |
| One visit per customer | Simplifies assignment | Split deliveries or multiple appointments | Inspect order and receiving rules |
| Delivery-only load | Total assigned load is enough for aggregate capacity | Pickups or transfers | Trace quantities by event |
| One trip per vehicle | Depot flow identifies a single route | Midday reloads | Compare actual trip logs |
| Known directed travel | Defines nominal schedule | Traffic, incidents, closures | Backtest matrix against actual travel |
| Fixed service duration | Makes start times computable | Quantity or dock variability | Compare actual service by stop type |
| Start-window semantics | Defines promise constraint | Completion or arrival requirement | Inspect contract and source field |
| Waiting permitted | Propagation permits slack | Site prohibits early arrival | Confirm parking and waiting policy |
| Aggregate capacity | Provides necessary resource limits | Packing and axle restrictions | Load-plan and scale checks |
| Static costs | Makes daily plan comparable | Overtime tiers or dynamic tariffs | Reconcile rates to invoices |
| Vehicle availability known | Makes selected fleet executable | Breakdown or driver absence | Confirm before dispatch |
| No traffic uncertainty | Defines deterministic feasibility | Variable travel and service | Scenario and historical replay |
| Fixed depot | Simplifies network | Multiple depots or open routes | Check actual origin and return rules |
| Complete input | Ensures all orders represented | Late or lost data | Reconcile to order source |

The deterministic result is conditional on every row above.
A safety buffer may be justified by empirical performance but is not itself a legal compliance system.
More complicated formulations are not automatically better if their data cannot be maintained.
Choose the simplest model that represents the decisions and obligations in scope,
then make omitted conditions visible in validation and handoff.

## 12. Exercises

### Exercise 1 — Translate a decision

A distributor has six mandatory orders, two vans, delivery windows, and one optional order.
The optional order can be declined for a stated contribution loss.
Write the decision variables, hard constraints, and objective components in business language.

**Deliverable:** a one-page decision contract.
**Acceptance:** distinguishes mandatory and optional service, van use, time, load, and lost contribution.
**Time:** 15 minutes.

### Exercise 2 — Trace a route

Use vehicle 1 on route D–A–B–D from section 6, leaving at minute 0.
Compute load at departure and after each stop, arrival, service start, departure, and return time.

**Deliverable:** a stop-by-stop table.
**Acceptance:** includes units and explicitly checks both windows and the vehicle limit.
**Time:** 15 minutes.

### Exercise 3 — Compare plans

Using the section 6 costs, compare plan P with a hypothetical feasible 65-km plan
that requires three vehicles at the same fixed and variable rates.

**Deliverable:** cost calculation and recommendation under the stated assumptions.
**Acceptance:** counts each used vehicle once and does not confuse kilometres saved with dollars saved.
**Time:** 10 minutes.

### Exercise 4 — Diagnose a contradiction

C's latest service start is changed to minute 20, with no earlier depot departure.
The fastest permitted depot-to-C trip takes 25 minutes.
Is the instance feasible? Name two authorised remedies.

**Deliverable:** a short proof and alternatives.
**Acceptance:** distinguishes a proved timing contradiction from a solver timeout.
**Time:** 10 minutes.

### Exercise 5 — Write a capacity constraint

Two vehicles have payloads 10 and 12 units; A, B, C have demand 4, 5, and 6 units.
Write the capacity inequality for the second vehicle and evaluate assigning B and C to it.

**Deliverable:** symbolic inequality and numerical substitution.
**Acceptance:** identifies the load at departure and notes that physical packing remains unchecked.
**Time:** 10 minutes.

### Exercise 6 — Analyse a search report

A minimisation run ends at a time limit with an incumbent of USD 1,000 and a valid lower bound
of USD 900. A second run ends at a time limit without an incumbent or proof.
Explain what each report establishes.

**Deliverable:** two status statements and the first run's incumbent-denominator gap.
**Acceptance:** does not call either plan optimal or the second instance infeasible.
**Time:** 10 minutes.

### Exercise 7 — Critique a model

A routing model uses symmetric car travel times, weight only, and fixed stop durations.
Its manager wants to release heavy-vehicle routes with customer completion windows.

**Deliverable:** ranked list of at least four risks and validation actions.
**Acceptance:** addresses vehicle access, time-window semantics, cube or packing, and service variability.
**Time:** 20 minutes.

### Exercise 8 — Design an experiment

A depot proposes a five-minute buffer per stop to improve reliability.
Design a comparison against the current planning policy using historical operating days.

**Deliverable:** a short evaluation plan with service, cost, and exception measures.
**Acceptance:** holds order and promise scope comparable, separates planned and actual,
and describes uncertainty and the release decision owner.
**Time:** 25 minutes.

## 13. Self-check answers

### Answer 1

The decisions are vehicle use, assignment, sequence, timing, and acceptance of the optional order.
Mandatory visits, permitted equipment, payload, shifts, and agreed windows are hard constraints.
The optional order needs a binary acceptance or rejection decision.
Its lost contribution or other approved penalty belongs in the objective if it is declined.
Add avoidable vehicle, distance, and paid-time costs without double counting.
Do not make mandatory service optional merely to make the model solve.

### Answer 2

| Event | Load after event, units | Arrival, min | Start, min | Departure, min |
|---|---:|---:|---:|---:|
| Depart D | 9 | — | — | 0 |
| A | 5 | 10 | 10 | 20 |
| B | 0 | 30 | 30 | 40 |
| Return D | 0 | 60 | — | — |

The initial 9 units are within the 10-unit payload.
A starts within 10–35 and B within 30–60.
The return is within the 120-minute depot limit.
Travel distance is 40 km and the route costs USD 180 under the section 6 assumptions.

### Answer 3

Plan P costs `2 × USD 100 + 90 × USD 2 = USD 380`.
The hypothetical plan costs `3 × USD 100 + 65 × USD 2 = USD 430`.
It saves 25 km, worth USD 50, but adds USD 100 for a vehicle.
Plan P is USD 50 cheaper, provided both plans satisfy the same service and other conditions.

### Answer 4

The earliest possible C service start is minute 25, already later than minute 20.
Therefore no feasible route exists under the stated assumptions, regardless of fleet size.
An approved later window or an authorised earlier departure are two possible changes.
An eligible third-party provider is another option if it can meet the original commitment.
Their costs, permissions, and feasibility must be assessed rather than assumed.

### Answer 5

For vehicle 2, `4 y_A2 + 5 y_B2 + 6 y_C2 <= 12 z_2`.
With B and C assigned and the vehicle used, the left side is 11 and the right side is 12.
The aggregate payload constraint is satisfied with one unit of spare capacity.
The inequality says nothing about cube, axle loads, access, or stackability.

### Answer 6

The first run has a feasible incumbent under the encoded model and a USD 900 lower bound.
Its absolute gap is USD 100 and its incumbent-denominator gap is 10%.
The optimum lies between the bound and incumbent, subject to the solver's validity guarantees.
The second run establishes neither feasibility nor infeasibility.
Both require status and validation details before any operational conclusion.

### Answer 7

A strong answer checks heavy-vehicle restrictions and directed road times first,
then maps completion windows to service start plus service duration.
It adds cube or independent loading validation and tests variability in service duration.
It also checks driver compliance, vehicle availability, matrix freshness, and missing orders.
The ranking should reflect severity and evidence, not assume that every risk is equally likely.

### Answer 8

A strong design replays the same order sets, fleet, windows, and travel data under both policies.
It compares vehicle use, predicted cost, planned lateness, and feasibility of released routes.
For observed operations, it tracks actual on-time performance, incremental paid time,
manual overrides, and customer exceptions, with comparable days or random assignment where feasible.
It reports uncertainty and any traffic or demand changes rather than attributing all differences to buffer policy.
Operations owns executable-route approval; the customer-service owner approves changes to promises.

## 14. Interview questions

1. **Why is the shortest route not always the cheapest plan?**
   Strong: distinguishes fixed vehicle, variable distance, time, and service costs.
   Weak: assumes all kilometres have the same economic effect.
2. **What does an arc decision variable represent?**
   Strong: identifies vehicle, ordered node pair, and binary traversal.
   Weak: describes only an unordered group of customers.
3. **Why are flow equations alone insufficient?**
   Strong: constructs a disconnected customer cycle and proposes connectivity constraints.
   Weak: assumes matching arrivals and departures guarantees a depot-connected route.
4. **What does a time-window constraint actually constrain?**
   Strong: specifies arrival, start, or completion and treats waiting explicitly.
   Weak: says merely that a vehicle reaches the customer sometime that day.
5. **How do you choose big-M?**
   Strong: derives it from valid variable bounds or uses an indicator constraint.
   Weak: chooses a huge number without checking numerical or validity effects.
6. **How can a route fit payload but fail at the dock?**
   Strong: explains cube, packing, axle distribution, and access.
   Weak: treats aggregate weight as a complete physical loading model.
7. **What does a time limit without a feasible incumbent mean?**
   Strong: no solution was found within the search; infeasibility is not proven.
   Weak: declares the customer orders impossible.
8. **How would you respond to proven infeasibility?**
   Strong: validates inputs, isolates conflicts, and proposes authorised business alternatives.
   Weak: silently makes legal or customer rules soft.
9. **What is an optimality gap?**
   Strong: compares feasible incumbent and valid bound with a stated denominator.
   Weak: equates it with forecast error or actual service failure rate.
10. **How would you measure a pilot's business value?**
    Strong: compares like-for-like service, incremental cost, exceptions, and actual outcomes.
    Weak: reports simulated kilometres alone as realised profit.

## 15. Key terms

- **Arc** — directed movement between graph nodes; see section 4.
- **Assignment** — selection of a vehicle for a stop; see section 5.3.
- **Big-M** — implication constant derived from bounds; see section 5.7.
- **Capacity** — permitted amount of a resource carried; see section 5.6.
- **Commitment** — accepted promise or obligation; see section 3.4.
- **Completion window** — interval constraining end of service; see section 5.7.
- **Cost-to-serve** — relevant cost of providing service; see section 4.
- **CVRPTW** — capacitated vehicle routing problem with time windows; see section 1.
- **Depot** — route start or end facility; see section 5.2.
- **Directed matrix** — origin–destination data that may differ by direction; see section 8.2.
- **Duty length** — defined elapsed working interval; see section 5.7.
- **Feasible** — satisfying encoded constraints; see section 3.1.
- **Fixed vehicle cost** — incremental charge per used vehicle; see section 5.8.
- **Forecast** — estimate of an uncertain future quantity; see section 3.4.
- **Gap** — incumbent-to-bound difference under a stated convention; see section 5.10.
- **Hard constraint** — rule that cannot be violated in the model; see section 3.1.
- **Incumbent** — best feasible solution found so far; see section 3.1.
- **Infeasible** — no solution satisfies the encoded rules if proven; see section 5.11.
- **Lower bound** — provably optimistic value in a minimisation problem; see section 5.10.
- **Objective** — criterion used to rank feasible plans; see section 5.8.
- **Outsourcing** — authorised external service alternative; see section 5.11.
- **Payload** — permitted freight mass on a vehicle; see section 4.
- **Plan** — candidate assignment and schedule; see section 3.4.
- **Service start** — beginning of work at a customer; see section 3.2.
- **Soft window** — authorised window with modelled violation; see section 5.8.
- **Subtour** — disconnected customer cycle; see section 5.5.
- **Time window** — permitted interval for a named event; see section 4.
- **Vehicle routing problem** — assigning and sequencing stops across vehicles; see section 1.
- **Waiting** — time between arrival and permitted service; see section 3.2.

## 16. Sources

Evidence is separated by type; none of these sources supplies the invented worked-example values.

**Research — Grade A**

- G. B. Dantzig and J. H. Ramser, “The Truck Dispatching Problem,” *Management Science*,
  1959, 6(1), 80–91. [DOI: 10.1287/mnsc.6.1.80](https://doi.org/10.1287/mnsc.6.1.80).
  Foundational vehicle-routing formulation; not a modern operating-cost benchmark.
- M. M. Solomon, “Algorithms for the Vehicle Routing and Scheduling Problems with Time Window
  Constraints,” *Operations Research*, 1987, 35(2), 254–265.
  [DOI: 10.1287/opre.35.2.254](https://doi.org/10.1287/opre.35.2.254).
  Research algorithms and time-window instances; results depend on its experimental setting.
- C. E. Miller, A. W. Tucker, and R. A. Zemlin, “Integer Programming Formulation of Traveling
  Salesman Problems,” *Journal of the ACM*, 1960, 7(4), 326–329.
  [DOI: 10.1145/321043.321046](https://doi.org/10.1145/321043.321046).
  Ordering constraints for eliminating tours; this lesson adapts the idea to multiple vehicles.

**Regulatory and standards — Grade B**

- Federal Motor Carrier Safety Administration, [Hours of Service of Drivers](https://www.fmcsa.dot.gov/regulations/hours-service).
  Consult current rules and applicability; the lesson's simplified shift is not a compliance determination.
- GS1, [Global Location Number](https://www.gs1.org/standards/id-keys/gln).
  Defines an identification standard, not an optimisation or service-quality guarantee.

**Company account — Grade C**

- UPS, [UPS Deploys Purpose-Built Navigation for UPS Service Personnel](https://about.ups.com/us/en/newsroom/press-releases/innovation-driven/ups-deploys-purpose-built-navigation-for-ups-service-personnel.html).
  Company description of its routing technology; claims and scope should be read in the original context.

**Technical documentation**

- Google OR-Tools, [Vehicle Routing Problem](https://developers.google.com/optimization/routing/vrp).
  Illustrates routing model concepts; a library's examples do not encode every business constraint.
- Google OR-Tools, [Vehicle Routing Problem with Time Windows](https://developers.google.com/optimization/routing/vrptw).
  Explains a time-window implementation; check library version and solution status in deployment.

## 17. Links

- [Module 12 index](_index.md).
- [Freight modes and their economics](01_freight-modes.md).
- [Rate structures and accessorials](02_rate-structures.md).
- [Lane and tender analytics](03_lane-tender-analytics.md).
- [Consolidation and cube utilisation](04_consolidation-cube.md).
- [Module 12 survey and Lab 12.1](../../../course/12_transportation_routing_last_mile.md).
- [Module expansion specification](../../../course/MODULE_EXPANSION_SPEC.md).
