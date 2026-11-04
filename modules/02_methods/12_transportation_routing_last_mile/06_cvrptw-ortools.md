# A Runnable CVRPTW in OR-Tools

> Module 12 — Transportation, Routing, and Last-Mile Analytics  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic Python and vehicle-routing terminology  
> Version 2026.1

## 1. Why this topic exists

A transport planner needs a route that can actually be driven, not just a shortest line on a map.
A route that exceeds vehicle capacity or reaches a customer after its receiving window creates recovery cost.
A capacitated vehicle routing problem with time windows (CVRPTW) makes these restrictions explicit.
Google OR-Tools can search for a feasible plan, but it does not understand your operation unless you encode it.
This lesson builds a small model and audits the returned schedule before discussing deployment.

The business decision is whether two available vehicles can serve four committed deliveries in a single shift.
The model chooses stop order and assignment, minimizing total driving minutes.
The solution is a candidate plan, not a driver instruction, customer promise, or proof of global optimality.
The four-customer data are fictional and intentionally small enough for manual review.

## 2. Learning outcomes

After this lesson you can:

1. Translate deliveries, capacities, service times, and windows into a CVRPTW data contract.
2. Build and run a routing model using the Python OR-Tools API.
3. Explain how the capacity and time dimensions enforce different constraints.
4. Audit a solution's customer coverage, vehicle loads, and service-start times.
5. Distinguish no solution found within a search limit from proof of infeasibility.
6. Calculate a simple plan cost and describe costs omitted by the objective.
7. Propose operational controls before releasing a model-generated plan.

Exercises 1–7 assess these outcomes in the same order.

## 3. The core mental model

```text
Orders + fleet + road-time matrix + operating rules
                        |
                        v
       Validate units, windows, capacity, and IDs
                        |
                        v
        Routing model: assignment + stop sequence
            |                        |
      capacity dimension       time dimension
            \                        /
             +---- search ----------+
                        |
               candidate schedule
                        |
            independent operational audit
                        |
         dispatcher accepts, revises, or rejects
```

The model chooses which vehicle visits each customer and in what order.
A dimension accumulates a quantity along a route: delivery load in one dimension and elapsed time in another.
Travel and service consume time; waiting may be inserted to meet a customer's earliest service time.
In this example every customer is mandatory and every vehicle starts and ends at the same depot.
The time window applies to service **start**, not necessarily completion.
An objective ranks feasible plans; it does not automatically encode every economic or safety requirement.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Node | Depot or customer location, identified by integer | Treating routing indices as node IDs |
| Vehicle | One available route resource | Assuming every vehicle must be used |
| Demand | Delivery quantity in units | Mixing parcels with kilograms |
| Capacity | Maximum total route demand in units | Ignoring vehicle-specific payload |
| Transit | Time added between two nodes, minutes | Omitting service at the departing node |
| Service time | Minutes spent at a stop | Counting it twice |
| Time window | Permitted service-start interval, minutes from shift start | Calling it a completion deadline |
| Cumulative variable | Accumulated quantity at a routing index | Assuming it is always a unique time |
| Slack | Waiting time allowed in the time dimension | Calling all slack driving time |
| Horizon | Latest allowed cumulative time, minutes | Confusing it with per-arc waiting allowance |
| Objective | Quantity minimized by the solver | Calling driving minutes total economic cost |
| Feasible | Satisfies encoded constraints | Assuming missing constraints are satisfied |
| Search limit | Budget for finding an incumbent | A certificate of infeasibility |
| Depot | Common start and finish location | Assuming depot loading and unloading are included |

## 5. The mechanics, step by step

### 5.1 Establish the decision and input contract

Freeze an order snapshot and record its creation time.
Each delivery requires a stable ID, location, quantity, service duration, and agreed window.
The fleet snapshot needs available vehicle IDs, usable capacity, shift availability, and depot assignment.
The matrix must cover every origin and destination in the same node order as the demand list.
Use nonnegative integer minutes and integer demand units; state how fractional source values are rounded.
Rounding travel times down can create a plan that is feasible only on paper.

The example assumes all orders are released and loaded at the start depot.
One delivery is represented by one node; no order is split.
All vehicles have the same travel-time matrix and zero loading and unloading time at the depot.
These are modelling assumptions, not OR-Tools defaults.

### 5.2 Check obvious feasibility before solving

Total demand must not exceed the sum of available vehicle capacities.
No indivisible customer's demand may exceed every eligible vehicle's capacity.
Each window must have an earliest time no later than its latest time.
The direct depot-to-customer travel time must permit arrival by the window end.
Return travel plus service must also fit the shift, subject to the actual route.
These are necessary checks, not sufficient checks: customers can still compete for the same narrow time window.
Do not silently delete an impossible delivery to make the solve succeed.

For this instance, demand is 2 + 3 + 2 + 3 = 10 units and capacity is 2 × 5 = 10 units.
Both vehicles must be used, and each must carry exactly 5 units.
That arithmetic does not establish time-window feasibility.

### 5.3 Choose the objective

The code minimizes the sum of driving minutes across all routes.
It excludes waiting, service, fixed vehicle cost, labor overtime, fuel, tolls, and service penalties.
A vehicle fixed cost could be added if its unit is made commensurate with the arc cost.
Do not add dollars to minutes without an explicit conversion and documented assumptions.
All customers are mandatory here; no disjunction permits dropping a customer.

### 5.4 Create indices and callbacks

The routing index manager maps internal routing indices to external node IDs.
Start and end indices for a depot may be different internal indices even though they map to node 0.
The distance callback returns travel minutes between the corresponding nodes.
The time callback adds service at the origin to travel to the destination.
This means the customer's cumulative time is its service-start time.
The final customer service is included on its outgoing arc to the depot.

### 5.5 Constrain capacity

A unary callback returns demand at the node being departed.
The capacity dimension has zero slack and starts at zero for each vehicle.
Its cumulative quantity is demand served before the current node, not remaining physical cargo.
For a delivery-only route, its end value equals total assigned demand and cannot exceed capacity.
It does not model pickups, changing cargo dimensions, weight distribution, or actual onboard inventory.

### 5.6 Constrain time

The time dimension uses travel plus origin service as transit.
Its slack allows up to 120 minutes of waiting per index, not a total route waiting budget.
The horizon limits cumulative time to 180 minutes after shift start.
The example fixes each vehicle start to minute 0 and limits each route's depot return to minute 180.
Each customer cumulative variable is restricted to its service-start window.
To require completion by a deadline, subtract service duration from that deadline before setting the latest start.
If customers have different shifts or depot loading constraints, model them explicitly.

### 5.7 Search and interpret

The first-solution strategy constructs an initial feasible route.
Guided local search then attempts improvements until the time limit expires.
A returned assignment is an incumbent, not necessarily the global optimum.
A `None` result means no assignment was returned; it may reflect infeasibility, invalid data, or an insufficient search budget.
Inspect solver status and use independent diagnostics rather than telling a customer the orders are impossible.
Identical inputs may lead to different feasible sequences or schedules across versions and search runs.

### 5.8 Audit and release

Print every visited node, service-start time, cumulative assigned load, and route end.
Independently check each customer's presence exactly once, window compliance, capacity, and return time.
Check the matrix against actual roads, vehicle restrictions, and traffic before release.
A dispatcher must review the candidate, communicate commitments, and retain a versioned decision record.
If a stop cannot be served, escalate it explicitly instead of hiding it in a route total.

### 5.9 Trace the capacity dimension by hand

For the manual route 0 → 1 → 4 → 0, the unary demand callback reads demand at the node being left.
The load cumulative variable at the depot start is zero.
The transition 0 → 1 adds depot demand zero, so the cumulative value at node 1 is zero.
The transition 1 → 4 adds node 1 demand of 2 units, so the value at node 4 is 2 units.
The transition 4 → 0 adds node 4 demand of 3 units, so the end value is 5 units.
Thus the cumulative value at a customer is not the amount delivered including that customer.
Nor is it the physical payload still on the truck.
It is an accounting variable for demand accumulated along the path.
For this delivery-only model, its final value enforces the route's aggregate capacity.
If a driver asks for remaining physical payload, derive it separately from the initial loaded quantity and deliveries completed.
Do not generalize this accounting convention to mixed pickup-and-delivery routes.
Those require explicit pickup and delivery relationships, load changes with the correct signs, and checks at every point in the sequence.

### 5.10 Trace time slack and schedule feasibility

For the arc from node 1 to node 4, the fixed transit is 10 minutes of service plus 25 minutes of driving.
With service starting at minute 20, the earliest possible next service start without waiting is minute 55.
Node 4 opens at minute 70, so at least 15 minutes of slack is required between these cumulative times.
The model permits that slack at the routing index; its representation does not establish the physical place where the driver waits.
That distinction matters when waiting on a public road or at a customer entrance is prohibited.
The slack maximum of 120 minutes applies to each transition, not to the sum over the route.
An operational rule limiting total paid waiting needs a separate constraint or a subsequent audit.
The printed cumulative times are service-start times, not necessarily exact physical arrival times.
To calculate physical arrival, the operation must choose where waiting occurs and confirm that choice is permitted.
The manual example's arrival calculations assume waiting at the destination.

### 5.11 Separate hard constraints from preferences

A hard capacity bound excludes any route exceeding the specified amount.
A travel-minutes objective ranks the routes that remain.
The latter cannot compensate for a hard capacity violation.
Conversely, a shorter route is not necessarily preferred in practice if its driver cost or customer risk is higher.
Consider two feasible options with 120 and 130 driving minutes.
If the first needs two vehicles and the second needs one, a fixed vehicle charge may reverse their economic order.
At USD 1 per driving minute and USD 40 per used vehicle, the respective proxy costs are USD 200 and USD 170.
That comparison assumes identical omitted service, labor, and other costs.
The example code has no vehicle fixed cost and therefore does not perform that comparison.
Before introducing a monetary objective, agree which costs are avoidable and use a consistent integer currency unit.
Check the magnitude of each term so that converting dollars to cents does not accidentally change relative weights.
Treat a soft late-delivery penalty as a commercial policy decision, not as permission to violate a mandatory contract.

### 5.12 Diagnose an infeasible-looking instance

Start with input integrity rather than modifying the customer's commitment.
Confirm that all time windows use the same date, local zone, and definition of service start.
Check that node IDs index the same rows and columns in the matrix and the same rows in the demand and service arrays.
Check that each customer is reachable from an eligible vehicle's depot and can be served within its window.
An individual earliest-arrival check is necessary but not sufficient: customers can compete for the same vehicle and time.
For example, two 4-unit orders may each fit a 5-unit vehicle, but not on the same route.
If only one vehicle is available, the aggregate 8-unit demand exceeds its capacity.
If two vehicles are available, capacity alone no longer proves feasibility because windows or shifts may still conflict.
Next inspect the reported routing status and the search budget.
With authorization, run diagnostics that remove one family of restrictions at a time on a copy of the instance.
If a diagnostic copy solves after relaxing windows, review the windows and travel estimates; do not call the relaxed plan executable.
If capacity relaxation produces a solution, identify which vehicle or assignment exceeds its real capacity.
Record each changed input and restore all real constraints before proposing a dispatch plan.

### 5.13 Prepare a reproducible planning package

Retain the order snapshot identifier, fleet snapshot identifier, matrix version, and their effective times.
Retain the program revision, OR-Tools version, search settings, run time, status, and objective unit.
Store the complete candidate routes and the results of the independent checks.
Record which data changed between the optimization run and dispatch release.
If an order is canceled after the snapshot, do not count it as a solver omission without reconciling versions.
If a new order arrives after the snapshot, do not claim the solver included it.
The dispatcher must decide whether to rerun, manually amend, or hold the order for a later wave.
A reproducible package is evidence of what was modeled, not evidence that a road, vehicle, or customer was actually available.

### 5.14 Establish the independent audit boundary

The provided assertions check the solution using the same in-memory input arrays used by the model.
They can reveal a route that violates those arrays, but they cannot reveal that an input array contains the wrong customer's window.
An operational audit should join the result to independently controlled order and fleet records using stable IDs.
Check all required orders exactly once, then verify each route's vehicle eligibility and quantity against those records.
Recompute travel and service timing using the approved matrix version and the same waiting convention.
Reconcile a count of unassigned, canceled, and duplicate orders rather than only a total stop count.
Check customer access, restricted roads, driver availability, and cargo handling separately where the model omits them.
An exception is resolved by correcting source data, changing an authorized constraint, or rejecting the candidate.
It is not resolved by removing an assertion or changing a label on the output.

### 5.15 Specify the planning horizon and event semantics

The planning horizon is not automatically the driver's paid shift.
It is the range of values allowed for the cumulative time variable.
Choose an origin instant, such as 08:00 on the operating date.
Retain the time zone and daylight-saving interpretation alongside that origin.
Convert every customer window to minutes from that same instant.
An overnight window may legitimately end after 1,440 minutes.
An overnight window is not automatically invalid because its clock time is earlier.
If the optimization spans a clock change, use elapsed instants rather than subtracting clock labels.
Retain original customer-facing local times for operational review.

For each timestamp, specify the event it constrains.
Arrival, service start, unloading completion, and departure are different events.
The example constrains service start, with both bounds inclusive.
The driver may arrive earlier and wait when waiting is operationally permitted.
The example does not require departure before the upper bound.
If the agreement requires completion, adjust the latest start for service.
If the agreement requires both arrival and completion, represent both conditions.
If setup work begins before the formal appointment, define which activity counts as service.
Never infer contractual semantics merely from a column named `window_end`.

The fixed start of minute zero deserves a separate business decision.
It makes depot waiting part of modeled route duration.
It also prevents the model from choosing a later duty start.
A later departure might reduce paid waiting under a different employment arrangement.
Allowing a flexible start is not enough if loading or vehicle availability is omitted.
Check the actual start range, loading work, and paid-time convention together.
When changing the start convention, update the cost comparison and audit accordingly.

### 5.16 Design the matrix as an operational product

The matrix grain is directed origin, destination, road profile, and effective conditions.
Its entries represent estimated travel minutes, not necessarily distance or paid labor.
A profile suitable for a car may use roads barred to a truck.
One approved truck profile may still be unsuitable for hazardous or oversized cargo.
Capture restrictions relevant to the actual vehicle class before generating arcs.
Do not substitute a straight-line distance for a road travel time without disclosure.
Do not infer that a missing arc has zero cost.
Zero is a valid value for a self-arc but suspicious for distinct locations.

Before solving, compare matrix dimensions with the node list.
Verify the node order, uniqueness, and coordinate version.
Check nonnegative values and the diagonal under the provider's conventions.
Inspect a sample of directional pairs against maps or historical journeys.
Different directions can have different costs because of one-way streets.
Missing or forbidden pairs need an explicit unreachable representation or exclusion.
A very large sentinel number is not a road restriction by itself.
It can distort objectives or integer ranges if used indiscriminately.
Its use requires a documented modeling convention and an audit.

Static travel estimates are conditional on their input traffic assumptions.
If the route crosses a peak period, one static value may be systematically optimistic.
Bucketed or time-dependent travel needs a compatible modeling approach.
Simply replacing one matrix entry after solving invalidates the old schedule.
Run scenarios with appropriately dated matrices to measure sensitivity.
The resulting range is not a calibrated prediction interval unless supported by data.
Track matrix creation time and source with every released plan.

### 5.17 Translate heterogeneous fleet rules

Start from a table of actual eligible vehicles, not a fleet average.
Each row needs its usable capacity, availability, depot, and resource identifier.
A vehicle's nominal payload may exceed the payload available after equipment is loaded.
Its cargo cube may bind before its weight does.
Different capacity measures require separate constraints and units.
One dimension cannot safely compare kilograms with cubic metres.
Compatibility may also depend on temperature, liftgate, site clearance, or driver training.

Build an eligibility matrix between orders and vehicles.
An order with no eligible vehicle needs escalation before solving.
An eligible vehicle is not necessarily available at the required time.
Different start and end depots need vehicle-specific indices.
Vehicle-specific shifts need bounds on each corresponding start and end variable.
Distinct road profiles require appropriate per-vehicle transit and cost handling.
Each additional rule should be checked against the current OR-Tools API documentation.
Do not assume that a capacity dimension enforces eligibility or working-time rules.

For physical loading, aggregate weight and cube are necessary but not always sufficient.
The order may need an unloading sequence compatible with the vehicle layout.
Fragility, axle loads, segregation, and tie-down rules may invalidate a numerical assignment.
Use a qualified loading review or a suitable loading model when those rules matter.
The routing output is a candidate to load, not proof of a safe load plan.
Record which operational checks were modeled and which were performed externally.

### 5.18 Choose integer units and rounding directions

OR-Tools routing callbacks return integer values.
Select a base unit fine enough to represent the business distinction being modeled.
If time is expressed in seconds, every time window and service duration must use seconds.
If cost is expressed in cents, every objective term must use compatible cents.
Document conversion before joining values from different feeds.
A conversion table is more reliable than an informal variable name.

For a hard upper capacity bound, conservative rounding may be appropriate.
For example, round each demand up and usable capacity down in the chosen unit.
That approach can reject a physically feasible route because it adds conservatism.
Quantify the gap before deploying it at scale.
For time, rounding each arc upward can accumulate across a long route.
Rounding each arc downward risks a schedule that is feasible only on paper.
Choose and document the policy; compare the result with unrounded source values.
Ensure the maximum route sum and weighted objective fit the integer ranges used by the solver.
Extreme penalties or a very fine base unit can overflow or dominate other terms.

### 5.19 Model optional work without hiding it

Some orders are forecast work; others are firm commitments.
The decision to defer an order is commercial policy, not a solver default.
If an order may be omitted, define who can approve that omission.
Specify whether omission is allowed for every order or only selected categories.
Specify how the penalty relates to the objective's units.
An arbitrary penalty of 100 means 100 objective units, not 100 dollars by default.
If objective units are driving minutes, a drop penalty trades against driving minutes.
That exchange may not match the customer's agreed remedy or the real cost of failure.

The released result must list dropped or unassigned orders explicitly.
Reconcile them to the original order snapshot using stable IDs.
If an order is canceled after the cutoff, retain the cancellation event separately.
If a mandatory order disappears, block release and investigate.
Do not call a plan complete merely because all remaining modeled nodes were visited.
If optional work competes with mandatory work, audit both groups independently.
Report the penalty's effect separately from physical travel and service costs.
Run a sensitivity analysis before using a penalty to automate commercial trade-offs.

### 5.20 Build an operational exception path

An input failure should identify the source record, not only raise a generic exception.
Route the issue to the owner of the relevant order, site, fleet, or matrix field.
For a missing delivery window, decide whether a published business default is authorized.
Otherwise hold the order rather than guessing a commitment.
For an impossible window, verify the customer's agreement and time zone first.
For a missing vehicle capacity, obtain the usable capacity from the fleet owner.
For a matrix failure, distinguish a provider outage from an inaccessible road.
For a no-assignment result, retain the full run and inspect the status.

The next step depends on the evidence.
A unit mismatch calls for correcting and rerunning the input snapshot.
Aggregate demand above capacity calls for additional resources or a changed service plan.
Search exhaustion may justify a different search budget or strategy.
An individual order with no legal vehicle calls for qualification or escalation.
A relaxed diagnostic model can identify a possible source of conflict.
It must be labeled diagnostic and cannot become an instruction without restoring real constraints.
The dispatcher decides how to communicate an unresolved commitment.

### 5.21 Manage search budgets and comparison

Record the time budget, strategy, software version, and hardware context.
Different time budgets can produce different incumbent qualities.
Compare runs on the same validated input snapshot and objective definition.
Use a feasible manual baseline where one is available.
A feasible baseline gives an upper bound for a minimization objective.
It is not a lower bound or a proof that the baseline is optimal.
If the new incumbent is worse than the baseline, the search has not beaten that baseline.
The operation may retain the baseline if it passes the same release checks.

Compare computation time with the operational decision deadline.
A lower objective discovered after the release cutoff may have no practical value.
Longer searches are also not a substitute for accurate travel or service inputs.
Store no-assignment outcomes as well as successful outcomes.
Track whether a lack of solutions correlates with instance size or particular constraints.
When evaluating strategies, include run-to-run variability and failed runs.
Do not compare a route that drops orders with a baseline serving all orders as if they were equivalent.

### 5.22 Define a release gate

The release gate evaluates the candidate against independently controlled records.
First reconcile the order set, including cancellations and orders added after cutoff.
Then check each vehicle against its actual status and the required equipment.
Recompute capacity and service timing from the proposed sequence.
Check customer access, loading, working-time requirements, and applicable operating rules.
Identify required manual checks that the model cannot perform.
Review low-slack stops and potential downstream customer impact.
Confirm who may accept a documented commercial exception.

The output has three possible states.
**Release** means all mandatory gates passed and an authorized person approved the version.
**Hold** means a resolvable data or resource issue remains open.
**Reject** means a hard rule is violated or a required order is missing.
The dispatcher may request a corrected run or an authorized operational alternative.
Record the gate outcome, reviewer, time, and evidence references.
Send drivers an approved instruction, not an unlabeled optimization artifact.
See [route versus dispatch](07_route-vs-dispatch.md) for execution and exception ownership.

### 5.23 Run a controlled pilot

Agree a decision and comparison period before selecting pilot days.
The baseline should have the same order eligibility and service requirements.
Capture orders offered to planning, not only orders ultimately delivered.
Measure paid time, kilometers, vehicles, failed attempts, and outside recovery charges.
Keep units and cost allocation consistent between baseline and candidate.
Record dispatch overrides with reasons and the actual instruction sent.
Separate optimization opportunity from adherence and actual execution.

Where feasible, randomize comparable days or territories to treatment and control.
Avoid assigning only easy areas or light-volume days to the new process.
If randomization is unavailable, disclose selection and relevant differences.
Check weather, customer mix, volume, fleet availability, and cutoff times.
Report distributions and uncertainty rather than only a single average.
A small pilot may reveal implementation defects without establishing financial impact.
An observed improvement is not automatically caused by the solver.
Set a stop rule for safety, service, or data-quality deterioration.
Give an accountable owner authority to stop the pilot and restore the approved baseline.

### 5.24 Treat model maintenance as an operating process

Assign separate owners for customer commitments, road data, fleet, and solver configuration.
Each owner approves changes to the data they control.
Retain effective dates so a historical plan can be reconstructed.
An updated customer entrance should not rewrite the entrance known on a previous day.
Track discrepancies found by drivers and dispatchers through to resolution.
Do not immediately replace a site service duration from one exceptional delivery.
Compare repeated observations by site and handling conditions.
Record the rationale and expected impact of any parameter change.

Version model code and dependency versions with the planning package.
Review changes to objective weights and constraint semantics as business-policy changes.
An upgrade may change search behavior even when inputs are unchanged.
Compare representative historical instances before changing the production version.
Document the approval, deployment date, and rollback route.
Keep operational access restricted to people who need order and driver information.
Archive artifacts according to the organization's retention and applicable legal policies.

### 5.25 Distinguish uncertainty from controllable slack

Slack can be a deliberate allowance for early arrival or uncertain activity.
It is not itself a probability of being on time.
A route with 20 minutes of window margin can still fail under a larger delay.
Its downstream margin may differ from its current-stop margin.
If one customer has a narrow window, later flexibility elsewhere may not help.
The cost of a missed window may also differ by customer and product.

Use historical data to examine travel and service variation by relevant context.
Compare predicted versus observed times only when the timestamps describe the same events.
If observations are sparse, state the sample limitation rather than inventing a percentile.
Stress the full sequence with coherent scenarios rather than independent arbitrary delays.
For each scenario, recalculate service starts and depot return.
Count which commitments fail and identify the first failure.
The scenario results are conditional on their inputs, not guaranteed probabilities.
If a probabilistic service target is required, use an appropriate statistical model and validation.

### 5.26 Reconcile solver outputs with the external system

Internal routing indices are not business identifiers.
Export stable order IDs and vehicle IDs with every sequence position.
Do not use a node's array position as a durable order identifier.
Persist the mapping from input IDs to nodes for the run.
Export both the plan version and the source snapshot IDs.
Include time zone, origin instant, and the definition of each reported time.
If an external transport system requires clock times, convert from the stored instant.

The receiving system should reject duplicate instruction versions safely.
It should distinguish a new release from a retry of an earlier message.
An acknowledgment should identify the version and the accepting party.
If a message fails, retain it in a visible retry or escalation queue.
Do not assume a successful API call means a driver has accepted the work.
Compare exported route contents with the approved candidate before dispatch.
Changes made after approval require the applicable revision and approval process.

### 5.27 Keep service and economics interpretable

An objective is a decision rule, not an accounting statement.
Driving minutes can be useful when vehicle count and other costs are fixed.
That assumption can fail when a solution activates another vehicle.
Paid time may include loading, waiting, service, breaks, and return travel.
Some carrier charges are fixed per route; others vary with time, distance, or activity.
For each comparison, state the avoidable cost boundary.
Separate estimated costs from actual invoices and payroll.

Suppose two feasible plans cost USD 190 and USD 205 per day under a specified proxy.
The USD 15 difference is meaningful only if service and cost coverage are comparable.
If the cheaper plan omits a required order, it is not an equivalent alternative.
If it needs an additional USD 25 reattempt, its stated proxy was incomplete.
If the USD 25 is uncertain, show the scenario rather than treating it as realized.
Include the cost of any external review or manual rework in a pilot business case.
Do not add an uncertain customer credit twice because it appears in two reports.
Avoid presenting modeled savings as realized cash without reconciliation.

### 5.28 Distinguish implementation checks from business acceptance

Software validation asks whether a model was implemented as specified.
Business validation asks whether that specification represents the operation well enough.
A unit test can confirm that origin service is added to each transit.
It cannot establish that a customer actually takes 10 minutes to unload.
A schedule audit can verify a service-start window in the input table.
It cannot establish that the input table captures the customer's real commitment.
Use independent records and operational review to close those gaps.

For model changes, maintain representative cases with known expected properties.
Examples include one order, a tight time window, a full-capacity route, and a forbidden assignment.
Test behavior at inclusive window boundaries and with zero-demand nodes.
Include a case where total capacity is sufficient but eligibility makes assignment impossible.
Compare properties rather than a single exact sequence when multiple optima can exist.
Retain a human-readable explanation for each expected property.
No collection of small cases proves performance on all production instances.

## 6. Worked example

### 6.1 Inputs

Time is measured in minutes from 08:00 local time; the depot is node 0.
Demand is measured in standardized delivery units, not weight.
The matrix is symmetric for teaching convenience; real road networks need not be symmetric.

| Node | Demand, units | Service, min | Service-start window, min |
|---:|---:|---:|---:|
| 0 depot | 0 | 0 | 0–180 |
| 1 | 2 | 10 | 20–60 |
| 2 | 3 | 10 | 30–80 |
| 3 | 2 | 10 | 60–120 |
| 4 | 3 | 10 | 70–140 |

| From / to | 0 | 1 | 2 | 3 | 4 |
|---|---:|---:|---:|---:|---:|
| 0 | 0 | 10 | 20 | 25 | 30 |
| 1 | 10 | 0 | 10 | 20 | 25 |
| 2 | 20 | 10 | 0 | 15 | 20 |
| 3 | 25 | 20 | 15 | 0 | 10 |
| 4 | 30 | 25 | 20 | 10 | 0 |

Two vehicles each have capacity 5 units and a 180-minute shift.
A manually constructed feasible plan is shown below; the solver may return a different one.

### 6.2 Vehicle A: 0 → 1 → 4 → 0

| Event | Calculation | Time, min |
|---|---|---:|
| Depart depot | Fixed start | 0 |
| Arrive at 1 | 0 + 10 | 10 |
| Start service at 1 | Wait until window opens | 20 |
| Depart 1 | 20 + 10 | 30 |
| Arrive at 4 | 30 + 25 | 55 |
| Start service at 4 | Wait until window opens | 70 |
| Depart 4 | 70 + 10 | 80 |
| Return depot | 80 + 30 | 110 |

Load is 2 + 3 = 5 units.
Driving is 10 + 25 + 30 = 65 minutes.
Waiting is 10 + 15 = 25 minutes; service is 20 minutes.
The route duration is 65 + 25 + 20 = 110 minutes.

### 6.3 Vehicle B: 0 → 2 → 3 → 0

| Event | Calculation | Time, min |
|---|---|---:|
| Depart depot | Fixed start | 0 |
| Arrive at 2 | 0 + 20 | 20 |
| Start service at 2 | Wait until window opens | 30 |
| Depart 2 | 30 + 10 | 40 |
| Arrive at 3 | 40 + 15 | 55 |
| Start service at 3 | Wait until window opens | 60 |
| Depart 3 | 60 + 10 | 70 |
| Return depot | 70 + 25 | 95 |

Load is 3 + 2 = 5 units.
Driving is 20 + 15 + 25 = 60 minutes.
Waiting is 10 + 5 = 15 minutes; service is 20 minutes.
The route duration is 60 + 15 + 20 = 95 minutes.
Every service start lies inside its window and both returns precede minute 180.
Total driving is 65 + 60 = 125 minutes; this is a feasible upper bound on the driving objective, not a proof of optimality.

### 6.4 Illustrative economics

Suppose the carrier charges USD 0.80 per driving minute and USD 40 per dispatched vehicle.
For the manual plan, the proxy cost is 125 × USD 0.80 + 2 × USD 40 = USD 180 per day.
If waiting and service are paid at the same rate, that proxy understates labor cost.
At USD 0.50 per on-duty minute and USD 40 per vehicle, the manual plan instead costs
(110 + 95) × USD 0.50 + 2 × USD 40 = USD 182.50 per day.
These are separate hypothetical costing conventions, not additive charges.
The code optimizes neither convention exactly because it minimizes driving minutes alone.

### 6.5 Stress test: later travel to node 4

Keep the manual assignment and all customer windows unchanged.
Suppose travel from node 1 to node 4 is 45 minutes rather than the planned 25 minutes.
The vehicle starts service at node 1 at minute 20 and leaves at minute 30.
It arrives at node 4 at minute 75 and can begin service immediately.
The original plan started service at minute 70, so the service-start delay is only 5 minutes.
The other 15 minutes of additional travel consume the originally planned waiting time.
After 10 minutes of service, the vehicle leaves node 4 at minute 85.
At the original 30-minute return travel time, it reaches the depot at minute 115.
The 140-minute latest service start and 180-minute return limit still hold.
This is a conditional calculation, not a claim that all travel delays will be absorbed.
If the 1-to-4 travel time is instead 115 minutes, service starts at minute 145 and misses the latest start by 5 minutes.
The planner should update the directional matrix and re-solve rather than assuming the original sequence remains best.

### 6.6 Stress test: an additional indivisible order

The original four deliveries total 10 units, exactly the combined capacity of two 5-unit vehicles.
Suppose an additional confirmed order requires 1 unit and must be served in the same wave.
The new total demand is 11 units while available capacity remains 10 units.
Under the lesson's one-trip, no-split assumptions, serving all five orders is impossible regardless of travel or window details.
This aggregate-capacity calculation is a valid infeasibility certificate for those specific assumptions.
It does not prove that service is impossible if an approved extra vehicle or a second trip becomes available.
The operational choices may include adding capacity, moving a non-mandatory order by agreement, or planning another wave.
Each choice changes the business problem and requires a new snapshot or explicit policy approval.
Do not disguise the additional demand by increasing a vehicle's recorded capacity above its safe usable capacity.

### 6.7 Stress test: service completion instead of service start

The original node 4 window allows service to start through minute 140.
With a 10-minute service duration, a start at minute 140 implies completion at minute 150.
If the actual customer agreement instead requires completion by minute 140, the correct latest service start is minute 130.
In general, latest start equals latest completion minus the modeled service duration.
If service duration is uncertain, a deterministic 10-minute estimate does not guarantee a 140-minute completion.
The business must decide what service-time estimate and reliability standard to use before modeling that requirement.
If the translated latest start precedes the earliest allowed start, flag the individual order as inconsistent.
Do not silently clamp the latest time to the earliest time, because that would create a different commitment.

### 6.8 Translate the model result into a release sheet

A useful release sheet identifies the order snapshot and matrix version at the top.
For each vehicle, list its route ID, driver or carrier assignment, capacity, sequence, and planned service-start times.
For each stop, show the stable order ID, receiving address, contact protocol, quantity, and agreed window.
Flag the distinction between model minutes from 08:00 and local clock times.
For example, minute 70 corresponds to 09:10 on the stated operating date, subject to the stated time zone.
List outstanding exceptions separately, including any order missing from the proposal.
Add the independent audit result and the name and time of the person authorizing release.
The sheet should not convert the model's deterministic times into customer guarantees.
After release, retain the sheet as a versioned historical artifact when dispatch changes the route.

### 6.9 A competing assignment and the value of waiting

Consider another feasible assignment using the same original inputs.
Vehicle A serves 0 → 1 → 2 → 0.
Vehicle B serves 0 → 3 → 4 → 0.
The two route loads are 2 + 3 = 5 units and 2 + 3 = 5 units.
The capacity check therefore passes for each vehicle.
The schedule below assumes destination waiting and zero depot handling time.

| Event | Vehicle A, min | Vehicle B, min |
|---|---:|---:|
| Leave depot | 0 | 0 |
| Arrive first customer | 10 | 25 |
| Begin first service | 20 | 60 |
| Leave first customer | 30 | 70 |
| Arrive second customer | 40 | 80 |
| Begin second service | 40 | 80 |
| Leave second customer | 50 | 90 |
| Return depot | 70 | 120 |

Node 1 starts at minute 20, within 20–60.
Node 2 starts at minute 40, within 30–80.
Node 3 starts at minute 60, within 60–120.
Node 4 starts at minute 80, within 70–140.
Both vehicles return before minute 180.
All four customers are served exactly once.
The plan is feasible under the stated inputs.

Vehicle A drives 10 + 10 + 20 = 40 minutes.
Vehicle B drives 25 + 10 + 30 = 65 minutes.
Together they drive 105 minutes, compared with 125 for the first manual plan.
The new plan is a feasible upper bound of 105 driving minutes.
It is not presented as proof that no better plan exists.
Vehicle A waits 10 minutes and serves for 20 minutes.
Its total duration is 40 + 10 + 20 = 70 minutes.
Vehicle B waits 35 minutes and serves for 20 minutes.
Its total duration is 65 + 35 + 20 = 120 minutes.
The combined duty duration is 190 minutes.
The first manual plan had a combined duration of 205 minutes.

At USD 0.80 per driving minute plus USD 40 per dispatched vehicle,
the alternative proxy is 105 × USD 0.80 + 2 × USD 40 = USD 164 per day.
That is USD 16 below the original USD 180 proxy.
Under the separate USD 0.50 per duty minute plus USD 40 per vehicle convention,
the alternative is 190 × USD 0.50 + 2 × USD 40 = USD 175 per day.
That is USD 7.50 below the original USD 182.50 duty-time proxy.
The numerical differences are forecasts under two assumed cost conventions.
Neither is a measured saving or a complete operating-cost comparison.
The 35-minute wait at node 3 must be physically and contractually permissible.
If that wait cannot occur there, verify an alternative waiting location and timing.

### 6.10 A tight-window conflict that aggregate capacity misses

Consider a separate two-customer instance, not an addition to the original four-node case.
Each order requires 2 units and lasts 10 minutes.
One available vehicle has capacity 5 units.
Both customer service starts must occur between minute 20 and minute 25.
Driving from the depot to either customer takes 20 minutes.
Driving between the customers takes 10 minutes in either direction.
Assume a sufficiently long depot-return horizon.

Aggregate demand is 2 + 2 = 4 units, below capacity 5.
Each customer can individually be reached at minute 20.
But serving either first at minute 20 finishes at minute 30.
The earliest arrival at the other customer is minute 40.
Minute 40 exceeds the second customer's latest start of minute 25.
Reversing the order gives the same contradiction.
For this stated instance, the one vehicle cannot serve both commitments.
Adding a second suitable vehicle could remove this particular conflict.
Its actual availability, capacity, and return timing still need to be checked.
Increasing a search budget cannot make the original restrictions simultaneously true.

### 6.11 A conservative rounding calculation

Suppose a different vehicle has 10.0 kilograms of usable payload.
Three indivisible orders weigh 3.31, 3.31, and 3.31 kilograms.
Their unrounded total is 9.93 kilograms and fits the stated payload.
If the chosen unit is a tenth of a kilogram and each demand is rounded upward,
each modeled demand becomes 34 units, or 3.4 kilograms.
The modeled sum is 102 units, or 10.2 kilograms.
The modeled capacity is 100 units, or 10.0 kilograms.
The conservative integer representation rejects the combined load.
That rejection is not a physical overload under the stated exact measurements.

Using hundredths of a kilogram instead gives three demands of 331 units.
The sum is 993 units against a capacity of 1,000 units.
This finer representation preserves feasibility for this example.
It does not remove measurement error in the source weights.
Use verified weights and an appropriate operational safety margin.
When evaluating rounding policies, distinguish quantization from uncertainty.
Do not fix a rounding artifact by overstating the physical vehicle limit.

### 6.12 Read sensitivity as a decision aid

Return to the 105-minute alternative and its 70- and 120-minute route durations.
Suppose node 3 service increases from 10 to 20 minutes.
Vehicle B starts node 3 at minute 60 and now departs at minute 80.
It arrives at node 4 at minute 90 and starts immediately.
It departs node 4 at minute 100 and returns at minute 130.
Node 4 still starts within its original 70–140 window.
The depot return remains within the 180-minute horizon.
Travel remains 65 minutes for vehicle B.
The route duration grows by 10 minutes because waiting at node 4 was already zero.

Now suppose travel from node 3 to node 4 is 75 minutes rather than 10,
while node 3 service remains the original 10 minutes.
Departure from node 3 is minute 70 and arrival at node 4 is minute 145.
The latest start is minute 140, so the sequence misses by 5 minutes.
The calculation examines this sequence under this changed arc only.
It does not establish that every other assignment is infeasible.
Re-solve the changed instance if the updated arc is accepted as planning input.

## 7. Code, query, or calculation

### 7.1 Installation

Install a supported Python 3 environment and OR-Tools using the official package instructions.
From a shell, create and activate a virtual environment appropriate to your operating system, then run:

```bash
python -m pip install ortools
```

Save the following as `cvrptw.py` and run `python cvrptw.py`.
The example uses public `ortools.constraint_solver` APIs.
It needs no network access after installation and no external data file.
The solver normally finds a solution for this small instance; exact sequences and times may vary.
The assertions audit the returned incumbent and fail visibly if a displayed schedule is inconsistent.

```python
from ortools.constraint_solver import pywrapcp, routing_enums_pb2


def main():
    travel = [
        [0, 10, 20, 25, 30],
        [10, 0, 10, 20, 25],
        [20, 10, 0, 15, 20],
        [25, 20, 15, 0, 10],
        [30, 25, 20, 10, 0],
    ]
    demand = [0, 2, 3, 2, 3]
    service = [0, 10, 10, 10, 10]
    windows = [(0, 180), (20, 60), (30, 80), (60, 120), (70, 140)]
    capacities = [5, 5]
    manager = pywrapcp.RoutingIndexManager(len(travel), len(capacities), 0)
    routing = pywrapcp.RoutingModel(manager)

    def drive(from_index, to_index):
        return travel[manager.IndexToNode(from_index)][manager.IndexToNode(to_index)]

    drive_id = routing.RegisterTransitCallback(drive)
    routing.SetArcCostEvaluatorOfAllVehicles(drive_id)

    def delivered(from_index):
        return demand[manager.IndexToNode(from_index)]

    demand_id = routing.RegisterUnaryTransitCallback(delivered)
    routing.AddDimensionWithVehicleCapacity(demand_id, 0, capacities, True, "Load")

    def elapsed(from_index, to_index):
        node = manager.IndexToNode(from_index)
        return service[node] + drive(from_index, to_index)

    time_id = routing.RegisterTransitCallback(elapsed)
    routing.AddDimension(time_id, 120, 180, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")
    for node in range(1, len(travel)):
        time_dim.CumulVar(manager.NodeToIndex(node)).SetRange(*windows[node])
    for vehicle in range(len(capacities)):
        time_dim.CumulVar(routing.End(vehicle)).SetRange(0, 180)
        routing.AddVariableMinimizedByFinalizer(time_dim.CumulVar(routing.End(vehicle)))

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.seconds = 2
    solution = routing.SolveWithParameters(params)
    if solution is None:
        print(f"No assignment returned; routing status={routing.status()}")
        return

    seen = []
    total_drive = 0
    for vehicle in range(len(capacities)):
        index = routing.Start(vehicle)
        nodes = []
        times = []
        while True:
            node = manager.IndexToNode(index)
            nodes.append(node)
            times.append(solution.Value(time_dim.CumulVar(index)))
            if routing.IsEnd(index):
                break
            index = solution.Value(routing.NextVar(index))
        load = sum(demand[n] for n in nodes[1:-1])
        assert load <= capacities[vehicle]
        assert times[0] == 0 and times[-1] <= 180
        for pos in range(1, len(nodes) - 1):
            node = nodes[pos]
            seen.append(node)
            assert windows[node][0] <= times[pos] <= windows[node][1]
        for pos in range(len(nodes) - 1):
            a, b = nodes[pos], nodes[pos + 1]
            assert times[pos + 1] >= times[pos] + service[a] + travel[a][b]
            total_drive += travel[a][b]
        print(f"vehicle {vehicle}: nodes={nodes}, service/start/end minutes={times}, load={load}")
    assert sorted(seen) == list(range(1, len(travel)))
    assert total_drive == solution.ObjectiveValue()
    print(f"total driving minutes={total_drive}; objective={solution.ObjectiveValue()}")


if __name__ == "__main__":
    main()
```

The printed time values are one concrete assignment to cumulative variables, not a forecast of actual arrivals.
Waiting is implicit in the difference between consecutive times after travel and origin service are deducted.
The audit recomputes load and driving from the printed sequence rather than trusting a displayed cumulative load.
The fixed route starts mean waiting before the first customer is counted within the shift.
If no solution is returned, the script prints the routing status rather than reporting a false feasible plan.
A production implementation should validate the data before calling OR-Tools and persist a full solve record.

### 7.2 Independently replay a proposed sequence

The following standalone calculation checks the alternative in section 6.9.
It does not invoke OR-Tools and does not read the solver's cumulative variables.
Save it separately as `replay.py` and run it with Python 3.
It assumes waiting at the destination, fixed start at minute zero, and no depot service.
The example data are deliberately duplicated for a transparent teaching calculation.
In production, obtain the audit data from independently governed records.

```python
travel = [
    [0, 10, 20, 25, 30],
    [10, 0, 10, 20, 25],
    [20, 10, 0, 15, 20],
    [25, 20, 15, 0, 10],
    [30, 25, 20, 10, 0],
]
demand = [0, 2, 3, 2, 3]
service = [0, 10, 10, 10, 10]
windows = [(0, 180), (20, 60), (30, 80), (60, 120), (70, 140)]
routes = [[0, 1, 2, 0], [0, 3, 4, 0]]
capacities = [5, 5]
visited = []
total_driving = 0

for vehicle, route in enumerate(routes):
    clock = 0
    driving = 0
    load = sum(demand[node] for node in route[1:-1])
    assert route[0] == route[-1] == 0
    assert load <= capacities[vehicle]
    starts = []
    for origin, destination in zip(route, route[1:]):
        clock += service[origin] + travel[origin][destination]
        driving += travel[origin][destination]
        if destination:
            clock = max(clock, windows[destination][0])
            assert clock <= windows[destination][1]
            starts.append(clock)
            visited.append(destination)
    assert clock <= 180
    total_driving += driving
    print(vehicle, "starts", starts, "return", clock,
          "load", load, "driving", driving)

assert sorted(visited) == [1, 2, 3, 4]
print("total driving", total_driving)
```

Expected output is:

```text
0 starts [20, 40] return 70 load 5 driving 40
1 starts [60, 80] return 120 load 5 driving 65
total driving 105
```

The replay selects the earliest schedule under its waiting convention.
It is not a general proof that every later schedule is feasible.
The return time includes service at the final customer.
The audit checks the listed routes, not the availability of a driver or a loading bay.
For production, reject malformed node identifiers before indexing the matrix.
Avoid using Python assertions as the only production control:
assertions can be disabled, so use explicit validation and recorded error handling.

### 7.3 Read the objective and the schedule separately

The routing model's objective is the sum of driving arc costs.
The time dimension is a constraint system, not the objective in this example.
The finalizer chooses values for selected end-time variables after route decisions.
It does not turn total elapsed time into the optimized objective.
Two schedules for the same route can have identical objective values.
They can differ in where waiting occurs and when the driver returns.
The manual replay computes one earliest schedule for its specified waiting convention.

When extending the model, enumerate every objective contribution.
Possible contributions include arc costs, vehicle fixed costs, and authorized penalties.
Check the documentation for the exact behavior of the installed API version.
The assertion that total drive equals objective is specific to the current code.
It must be revised if any additional objective term is introduced.
Continue to report physical driving minutes independently after such a change.
Report penalties separately so an omitted order is visible to an operator.

### 7.4 A boundary checklist for the code

| Boundary | Question | Expected action on failure |
|---|---|---|
| Input arrays | Do all arrays have one entry per node? | Reject the input snapshot |
| Matrix shape | Is every row the same length as the node list? | Regenerate or repair the matrix |
| Node identity | Does each row map to the intended order? | Correct the ID mapping |
| Unit | Are demand and capacity expressed in the same integer unit? | Convert using the approved policy |
| Window | Is earliest start no later than latest start? | Resolve the source commitment |
| Route | Does each required order appear exactly once? | Block release and investigate |
| Time | Is each transition consistent with service and travel? | Reject the invalid schedule |
| Capacity | Is each vehicle's total within its usable capacity? | Reassign or obtain capacity |
| Shift | Is the route within the specified start and end limits? | Replan or escalate |
| External rules | Are omitted constraints checked by a qualified owner? | Hold until those checks pass |

These are acceptance boundaries, not claims that the sample script implements every check.
The script intentionally keeps its input in memory for instructional clarity.
A production pipeline also needs controlled data ingestion and a durable audit record.

## 8. Data, standards, and regulatory context

Keep timestamps as instants with explicit time zone and preserve the customer's local window interpretation.
A daylight-saving transition can make a local time ambiguous or nonexistent.
Resolve it before converting to integer minutes from the planning origin.
Record travel-matrix generation time, road profile, traffic assumption, source, and rounding policy.
Matrix freshness and geographic coverage are operational controls, not solver guarantees.
Customer quantities must use the same capacity unit as the vehicle capacities.
If both mass and cube constrain loading, add separate dimensions and validate their units.

US motor-carrier hours-of-service requirements depend on vehicle and operation; consult current
[FMCSA guidance](https://www.fmcsa.dot.gov/regulations/hours-service/summary-hours-service-regulations).
A 180-minute horizon is an illustrative shift bound, not a compliance model.
Break rules, accumulated duty history, and jurisdictional exceptions require additional data and constraints.
GS1 location identifiers can support consistent location master data where trading partners use them;
see [GS1 identification keys](https://www.gs1.org/standards/id-keys).
An identifier standard does not verify geocoding, access restrictions, or delivery windows.
Treat customer location and driver data as sensitive and apply the applicable local privacy rules.

## 9. Industry practice and case evidence

The [OR-Tools vehicle routing documentation](https://developers.google.com/optimization/routing) describes routing problems,
including capacity and time-window examples and the routing library's dimensions.
That is evidence of a modelling interface, not evidence that this fictional plan was used in industry.
UPS describes its [ORION route optimization system](https://about.ups.com/us/en/our-company/innovation-driven/orion.html) publicly.
The company account is evidence that UPS uses route optimization; it is not independent validation of the economics above.
Inference: a carrier deploying route optimization still needs current operational inputs and an exception process.
Do not transfer a vendor or carrier's reported savings to this four-stop example without a comparable baseline.

A production pilot should compare against the dispatcher's existing practice on comparable days.
Track orders presented, orders served, paid hours, kilometers, failed attempts, and late deliveries.
Include externalities such as reattempt cost and the cost of extra vehicles.
Document whether allocation to pilot and control is randomized and whether service rules changed.
If dispatchers override a plan, preserve both the original plan and the reason for the override.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unit mismatch | Loads far above fleet norms | Parcels compared with kilograms | Invalid capacity decision | Validate unit metadata |
| Stale matrix | Actual travel exceeds predictions across lanes | Old traffic assumptions | Late stops | Version and refresh matrix |
| Missing service time | Dwell absent from schedule | Transit omits work | Unrealistic return | Add origin service to transit |
| Wrong window meaning | Completion after receiving closes | Start treated as completion | Rejected delivery | Convert deadline to latest start |
| Index confusion | Depot appears as customer | Internal index used as node ID | Corrupt route | Always use index manager |
| Insufficient capacity | Total demand exceeds fleet | Necessary condition fails | No feasible assignment | Escalate fleet or orders |
| Hidden dropping | Missing stops in output | Optional-stop penalty used | Unserved commitments | Audit mandatory coverage |
| False infeasibility | Search ends quickly with no assignment | Time limit exhausted | Avoidable escalation | Inspect status and diagnostics |
| Unsupported legal claim | Single shift horizon called compliant | Missing duty history | Driver violation risk | Model and review real rules |
| Incorrect economics | Driving minutes presented as dollars | Objective omits costs | Wrong procurement decision | Recalculate full incremental cost |
| Unrealistic slack | Long waiting seen at many stops | Per-index slack mistaken for route total | Shift not representative | Bound and audit waiting |
| Bad geocode | Driver reports inaccessible entrance | Coordinates do not identify loading point | Failed stop | Verify entrance and restrictions |

## 11. Assumptions and limits

| Assumption | Why used | What breaks it | Test or response |
|---|---|---|---|
| Static travel times | Small deterministic model | Congestion varies by departure | Compare to actual time by period |
| One depot | Simplifies starts and ends | Vehicles based elsewhere | Use per-vehicle start and end nodes |
| One demand unit | Supports one capacity dimension | Weight and cube bind differently | Add separate dimensions |
| No split deliveries | One node per order | Large order needs multiple vehicles | Reframe order allocation explicitly |
| Mandatory customers | Represents firm commitments | Some orders can be deferred | Add transparent optional-stop policy |
| Start at minute zero | Makes schedule easy to audit | Driver shifts differ | Set vehicle-specific start ranges |
| Fixed service duration | Avoids stochastic dwell | Site conditions vary | Estimate site-specific distributions |
| Unlimited labor detail | Shift horizon is a proxy | Breaks or prior duty bind | Include regulatory and employment rules |
| Complete matrix | Callback always returns an arc | Disconnected or prohibited roads | Validate reachability and restrictions |
| Single objective | Demonstrates routing | Cost and reliability trade off | Define approved economic objective |

## 12. Exercises

1. **Data contract**, 15 minutes. Deliver a table of the data needed for each customer and vehicle. Acceptance: include units, window semantics, IDs, and an owner for each field.
2. **Build and run**, 20 minutes. Install OR-Tools and run the provided program. Acceptance: capture the printed routes and distinguish a returned solution from a no-solution status.
3. **Dimension reasoning**, 15 minutes. Explain where service time is charged and why capacity at a customer index is not necessarily its post-delivery total. Acceptance: identify origin transit and unary demand semantics.
4. **Audit**, 20 minutes. Check the manual plan's coverage, loads, windows, and total driving time. Acceptance: show all four checks and arithmetic.
5. **Search result**, 10 minutes. A search returns no assignment after two seconds. Deliver a diagnostic decision. Acceptance: do not assert mathematical infeasibility; identify at least two next steps.
6. **Cost**, 15 minutes. Recalculate the manual plan at USD 0.90 per driving minute and USD 35 per dispatched vehicle. Acceptance: show both components and total.
7. **Release control**, 25 minutes. Design a dispatcher review checklist for a plan with fresh orders. Acceptance: include a data version, feasibility audit, override reason, and notification owner.
8. **Window critique**, 15 minutes. Node 4 must finish by minute 75. Does the manual schedule comply? Acceptance: calculate completion and propose a correct latest service-start bound.

## 13. Self-check answers

1. Each order needs a stable ID, valid location, quantity and unit, service duration, and earliest/latest service-start times with time zone. Fleet data include availability, capacity and unit, shift and start/end depot. A named operational owner must maintain each field.
2. A feasible assignment lists both vehicles, customer sequences, start/end times, loads, and total driving minutes. Exact ordering is not guaranteed. A no-assignment message is not a proof that no feasible schedule exists.
3. Transit from node `a` to `b` includes service at `a` plus driving from `a` to `b`. Demand is accumulated on departure from the current node, so the cumulative load at that node represents previously visited demand.
4. Nodes 1–4 appear once. Both vehicle loads are 5 units. Service starts 20, 70, 30, and 60 are within their respective windows. Driving is 65 + 60 = 125 minutes.
5. Validate capacity and window inputs, inspect the routing status, and consider a longer search or a simpler diagnostic model. Communicate uncertainty while escalating the potential service risk.
6. Driving costs 125 × USD 0.90 = USD 112.50. Vehicles cost 2 × USD 35 = USD 70. Total is USD 182.50 per day, before other costs.
7. A strong checklist records the data snapshot and matrix, independently checks every order and capacity, verifies driver and vehicle availability, records overrides, and assigns a named notification owner. It blocks release when a hard constraint fails.
8. Service starts at minute 70 and takes 10 minutes, so completion is minute 80 and misses the deadline by 5 minutes. The latest allowed start is 75 − 10 = minute 65. If the original earliest start of minute 70 remains binding, this individual window is inconsistent; clarify the customer's requirements before re-solving.

### Answer detail: contract and implementation

For exercise 1, the data contract should distinguish raw and modeled values.
An order ID is the join key, not an array position.
The location record should identify the usable entrance and its coordinate source.
Quantity requires a unit and an effective conversion rule.
The service duration is a planning estimate, not an observed completion time.
The window needs a local date, time zone, endpoints, and event definition.
The customer or commercial owner verifies the commitment.
The fleet owner verifies vehicle capacity and availability.
The road-data owner verifies matrix generation and profile.
The planner reconciles those snapshots at the cutoff.

For exercise 2, include the actual status if no assignment is returned.
Do not manufacture an example route to replace an unsuccessful run.
If an assignment is returned, inspect the printed customer list for duplicates.
Check the reported load against each vehicle's capacity.
Check the time at every customer against that customer's own window.
Recompute the objective from arc travel times.
The exact vehicle numbering is not important if all restrictions hold.
The actual output can differ from both manual schedules in this lesson.

For exercise 3, consider the arc from customer 1 to customer 4.
Its time transit is 10 minutes of service plus 25 minutes of travel.
The cumulative time difference can exceed 35 minutes because of waiting.
The load transition adds customer 1 demand of 2 units.
It does not add customer 4 demand until the route leaves customer 4.
This convention is valid for the modeled delivery-only aggregate check.
It does not describe the truck's actual remaining cargo.

### Answer detail: audit and decision

For exercise 4, compare sets as well as counts.
The set of visited customers is {1, 2, 3, 4}.
Its count is four and each element appears once.
Vehicle A's 2 + 3 units and vehicle B's 3 + 2 units both equal capacity.
The respective service starts are 20 and 70 minutes on A,
and 30 and 60 minutes on B.
Compare each number to the window for the corresponding node.
The return times are 110 and 95 minutes, each within 180 minutes.
Driving totals 65 + 60 = 125 minutes.
The return calculations include the last customer's service time.

For exercise 5, first verify whether the input can be parsed and the solve is valid.
Then inspect demand, fleet eligibility, windows, and route-duration assumptions.
A longer search is appropriate only when there is time before the decision deadline.
One-at-a-time relaxations are diagnostic, not release authorizations.
If aggregate mandatory demand exceeds capacity, identify that specific contradiction.
If no contradiction is found, continue to report the result as unresolved.
Inform the dispatcher early enough to secure backup capacity if needed.

For exercise 6, apply the per-minute charge only to 125 driving minutes.
The USD 112.50 driving charge and USD 70 vehicle charge sum to USD 182.50.
Do not add the earlier USD 180 proxy; it uses different assumptions.
Neither calculation includes a separate paid-waiting charge.
An invoice may apply minimum hours or other terms not shown here.

For exercise 7, a complete design has a stop condition as well as a checklist.
Missing mandatory orders, impossible capacity, or unapproved safety exceptions block release.
The checklist identifies the reviewer and records the decision time.
A changed customer window requires authority from the appropriate commercial owner.
The driver receives only the approved version and acknowledges the instruction.
If a data correction follows approval, create and review a new version.

For exercise 8, node 4's original start at minute 70 produces completion at minute 80.
The new completion limit of minute 75 is exceeded by 5 minutes.
The model's correct latest start is minute 65 when service lasts 10 minutes.
But the original earliest start at node 4 is minute 70.
If that earliest start remains in force, the revised individual window is inconsistent:
minute 70 is later than minute 65.
Clarify whether the customer changed only the deadline or also permits earlier service.
Do not declare an adjusted window feasible by silently changing its earliest bound.

## 14. Interview questions

1. **How does a dimension work?** Strong: cumulative variables, transit, slack, and bounds. Weak: it is merely a reporting field.
2. **Why is the time callback asymmetric?** Strong: origin service plus travel; real road travel may be directional. Weak: service is ignored.
3. **What does a no-solution result prove?** Strong: no incumbent returned under the run conditions. Weak: global infeasibility is certain.
4. **How would you model completion windows?** Strong: subtract service time from the latest completion or model a completion variable. Weak: leave latest start unchanged.
5. **What is the objective's unit?** Strong: total driving minutes. Weak: dollars or route duration without conversion.
6. **How do you audit coverage?** Strong: compare visited IDs with required IDs exactly once. Weak: count route lines only.
7. **What does capacity represent here?** Strong: total delivery demand assigned to each route. Weak: changing physical load for arbitrary pickups.
8. **What must happen before release?** Strong: reconcile current orders, fleet and legal restrictions, then obtain accountable dispatch approval. Weak: send routes automatically because the solver succeeded.

## 15. Key terms

- **Assignment:** allocation of customers to vehicles; see section 5.
- **Capacity:** maximum assigned demand; see section 5.5.
- **Completion deadline:** latest allowed end of service; see section 5.6.
- **Cumulative variable:** quantity at a routing index; see section 4.
- **Customer node:** one modeled delivery; see section 4.
- **Demand:** customer delivery quantity; see section 6.1.
- **Depot:** common route origin and destination; see section 6.1.
- **Dimension:** accumulated constrained quantity; see section 3.
- **Driving time:** arc travel minutes; see section 5.3.
- **Feasible solution:** assignment satisfying encoded constraints; see section 5.7.
- **Finalizer:** mechanism choosing values for selected variables after routing; see section 7.
- **Horizon:** maximum cumulative time; see section 5.6.
- **Incumbent:** best assignment returned by a search; see section 5.7.
- **Index manager:** maps routing indices to nodes; see section 5.4.
- **Mandatory stop:** customer that cannot be dropped; see section 5.3.
- **Objective:** quantity minimized; see section 5.3.
- **Service time:** stop work duration; see section 5.4.
- **Slack:** allowed waiting in the time dimension; see section 5.6.
- **Time window:** allowed service-start interval; see section 4.
- **Transit callback:** function giving an arc's time or cost; see section 5.4.
- **Vehicle route:** ordered sequence from start to end; see section 6.
- **Waiting:** time beyond transit before the next service start; see section 7.

## 16. Sources

- Google OR-Tools, [vehicle routing overview](https://developers.google.com/optimization/routing).
- Google OR-Tools, [vehicle routing with capacity constraints](https://developers.google.com/optimization/routing/cvrp).
- Google OR-Tools, [vehicle routing with time windows](https://developers.google.com/optimization/routing/vrptw).
- Google OR-Tools, [Python installation](https://developers.google.com/optimization/install/python).
- FMCSA, [summary of hours-of-service regulations](https://www.fmcsa.dot.gov/regulations/hours-service/summary-hours-service-regulations); verify the current rules and their applicability.
- GS1, [identification keys](https://www.gs1.org/standards/id-keys).
- UPS, [ORION](https://about.ups.com/us/en/our-company/innovation-driven/orion.html), company description; company claims are not independently verified here.

## 17. Links

- [Module index](_index.md)
- [Freight modes](01_freight-modes.md)
- [Rate structures](02_rate-structures.md)
- [Lane and tender analytics](03_lane-tender-analytics.md)
- [Module survey and lab context](../../../course/12_transportation_routing_last_mile.md)
