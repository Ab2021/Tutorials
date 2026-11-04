# Capacity Requirements Planning

> Module 15 — Manufacturing and Production Planning  
> Topic 05 of 11  
> Estimated study time: 4 hours  
> Prerequisites: [planning cascade](01_planning-cascade.md), [manufacturing master data](02_manufacturing-master-data.md), [MRP explosion](04_mrp-explosion.md)  
> Version 2026.5

## 1. Why this topic exists

A materials plan can say that enough components will be available while the plant lacks the hours to convert them into finished goods.
The resulting shortage may appear only when orders reach the floor, after customer dates and supplier deliveries have been accepted.
Capacity requirements planning (CRP) exposes this mismatch early enough to choose a remedy.

CRP translates dated operations on planned and open manufacturing orders into required resource time.
It compares that load with resource time available on a specific calendar.
The output is a set of dated gaps and a decision queue, not a self-executing schedule.
An overload in a weekly bucket says that the proposed work exceeds the stated weekly capacity.
It does not, by itself, establish the exact order sequence or which customer will be late.

The production planner needs to know whether to move work, add a shift, use another resource, subcontract, or revisit the master production schedule (MPS).
The supervisor needs to know whether the staff, tooling, and maintenance window support the proposal.
The finance partner needs the incremental cost and economic exposure, not merely a utilization percentage.
The customer team needs a credible date before changing a commitment.
Engineering and quality need to approve an alternate process where product or regulatory controls require it.

A capacity number without a calendar can be dangerously persuasive.
Two nominally identical 40-hour machines do not provide 80 available hours if one is down for eight hours of planned maintenance.
A machine can also be idle while a required operator or specialist is unavailable.
Conversely, abundant labor hours do not create machine time at a constrained work center.
The resource and time bucket must match on both sides of the comparison.

A forecast is an estimate of demand; it is not an accepted order.
A plan is an intended course of action; it is not necessarily feasible or approved.
A target is a desired outcome, such as a utilization objective; it is not productive time.
A commitment is an obligation accepted by an authorized party, such as a confirmed customer delivery or a firm subcontract purchase order.
CRP tests a proposed plan against assumptions; it does not turn a forecast into a commitment.

This lesson uses a fictional plant producing precision pump components.
Its figures illustrate decisions and are not evidence about a real company.
The arithmetic separates machine-hours, labor-hours, paid clock time, and incremental cost in United States dollars (USD).
You will see why a seemingly small overload still needs a named owner, a feasible resource action, and a closed-loop check.

## 2. Learning outcomes

After completing this lesson, you can:

1. **Trace** the information flow from dated orders and routings to work-center loads and decisions.
   Exercise 1 checks your causal explanation.
2. **Calculate** setup and run load separately for machine and labor resources.
   Exercise 2 requires a numerical resource ledger.
3. **Derive** effective calendar capacity without counting the same loss twice.
   Exercise 3 tests gross, net, and effective hours.
4. **Diagnose** an overload and explain what a weekly bucket does and does not prove.
   Exercise 4 requires a capacity gap and a timing caveat.
5. **Compare** feasible remedies using incremental economics, delivery risk, and approval constraints.
   Exercise 5 asks for a costed decision.
6. **Critique** the underlying data and the treatment of open orders, yields, and shared resources.
   Exercise 6 asks for a prioritized data investigation.
7. **Design** decision rights and a closed-loop response to a changing plan.
   Exercises 7 and 8 require an accountable workflow and a scenario design.

An excellent result states the unit, date, resource, assumptions, and decision owner.
A correct formula with an infeasible calendar is not a credible production plan.
A cheap remedy that lacks authorization or available material is not an approved remedy.

## 3. The core mental model

CRP is a time-phased resource translation and reconciliation process.
Material requirements planning (MRP) proposes orders and dates based on material logic.
Routings explain which resources each order consumes and at what rate.
Calendars and loss assumptions describe the resource supply available in each time bucket.
The difference is a decision signal.

```text
Forecast + accepted orders + inventory policy
                    |
                    v
          approved MPS / order plan
                    |
                    v
      MRP orders + open shop orders
                    |
                    v
   operation dates + routings + remaining quantity
                    |
                    v
   load by resource, time bucket, and order status
                    |
                    +--------> compare <--------+
                                               |
  shifts + machines + people + closures        |
       - planned downtime - other losses ------+
                    |
                    v
         dated gaps and root-cause checks
                    |
                    v
   options -> economics + risk + authority -> decision
                    |
                    v
       update plan, communicate, and recheck
                    |
                    v
    finite schedule and execution feedback
```

The left-hand branch expresses demand for resource time.
The right-hand branch expresses supply of resource time.
Their units and boundaries must match before subtraction.
The decision loop changes one or both branches and then recomputes the result.

CRP is more detailed than rough-cut capacity planning (RCCP).
RCCP tests an aggregate master schedule against selected critical resources using bills of capacity or representative rates.
CRP normally uses dated manufacturing orders, operation routings, and work-center calendars.
The terms and implementation details vary across systems; examine the actual calculation rather than relying on the label.
Finite scheduling adds sequence and resource-conflict constraints that a bucketed infinite-load calculation does not necessarily represent.

Think of each operation as drawing on a set of capacity accounts.
A setup can require a machine, a technician, and a fixture at the same time.
The same hour may be one machine-hour and one labor-hour; it is not two elapsed hours.
For a resource pool, however, summing like units across independent jobs is appropriate.
Resource requirements are simultaneous constraints, not interchangeable currencies.

A weekly balance can hide a daily collision.
For example, 30 machine-hours of load and 40 machine-hours of weekly capacity do not prove feasibility if all 30 hours require a machine on Monday and only eight hours are available that day.
Conversely, a week showing a 10-hour deficit may be recoverable by moving an eligible operation to an earlier period.
Movement is permitted only when predecessor completion, material availability, expiry, storage, and customer constraints allow it.

The gap is not the decision.
A negative balance requires explanation and an owner.
A positive balance is headroom, not proof of a good sequence.
The planner tests candidates against all affected resources and dates before recommending a change.
The authorized decision maker accepts the cost, service, and risk trade-off.

## 4. Definitions and vocabulary

The following definitions set the boundaries used in this lesson.
System terminology differs, so document the mapping to local fields.

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Capacity requirements planning (CRP) | Time-phased comparison of operation resource load with resource availability, commonly in hours per bucket | Treating a load chart as a feasible sequence |
| Work center | Planning group of one or more resources performing related operations; identified by site and code | Pooling equipment that cannot actually substitute |
| Resource | Constrained machine, person, crew, fixture, or other capability with a calendar | Assuming all hours in a pool are interchangeable |
| Capacity pool | Resources combined under stated interchangeability rules; hours per bucket | Combining incompatible shifts, skills, or qualifications |
| Routing | Ordered operations, resource assignments, standard times, and dependencies for a product or process revision | Treating an obsolete default routing as a validated process |
| Operation | A specified step on an order consuming resources; coded by order and operation number | Counting only the final operation |
| Setup time | Resource time per run, batch, or changeover, usually hours per order | Multiplying a per-order setup by every unit |
| Run time | Resource time proportional to processed quantity, usually hours per unit | Treating elapsed time as both machine and labor time without checking crew |
| Queue time | Time waiting before processing; elapsed hours or days | Treating waiting as active machine load |
| Move time | Elapsed time to transfer work between operations | Assuming it necessarily consumes the bottleneck machine |
| Lot size | Quantity processed together, in units per order or batch | Treating a split order as having no additional setup |
| Gross capacity | Calendar time before stated deductions, in resource-hours per bucket | Calling nameplate time immediately usable productive time |
| Planned downtime | Known shutdown or maintenance time deducted from the relevant calendar, in hours | Deducting the same maintenance twice |
| Net scheduled capacity | Gross calendar hours less explicitly scheduled exclusions | Confusing it with an output target |
| Effective capacity | Net scheduled capacity adjusted for specified expected losses, in productive-equivalent hours | Applying an overlapping overall equipment effectiveness factor twice |
| Load | Standard resource time required by dated operations, in resource-hours | Using quantities or sales dollars as though they were hours |
| Remaining load | Standard time still required to finish an open operation | Reloading completed units and completed setup |
| Capacity gap | Capacity minus load, in like resource-hours and the same bucket | Dividing by capacity and forgetting the absolute deficit |
| Utilization | Here, load divided by effective capacity, expressed as a percentage | Claiming this planned ratio equals observed equipment utilization |
| Infinite loading | Assignment of work without automatically enforcing a hard resource limit | Assuming it guarantees execution |
| Finite scheduling | Sequencing and timing work while enforcing modeled resource constraints | Assuming every modeled schedule survives real disturbances |
| Bucket | A time aggregation interval, such as a day or week | Combining different dates and masking short-term peaks |
| Time fence | Boundary that changes how freely a plan may be changed | Treating a protected interval as physically impossible to change |
| Open order | Released or firm order with work remaining | Adding its original full load to its residual load |
| Planned order | Proposed order not yet necessarily released or committed | Describing an MRP proposal as an accepted customer promise |
| Yield | Good output divided by input or starts, under a stated measurement basis | Applying yield both in order quantity and again to run load |
| Scrap | Units that cannot continue as conforming product; units per operation | Ignoring where in the routing loss occurs |
| Alternate routing | Approved alternative operation sequence or resource | Assuming a theoretical machine substitution is already qualified |
| Bottleneck | Resource or constraint limiting system throughput under the current conditions | Labeling the busiest average resource the permanent bottleneck |
| Overtime | Work outside normal scheduled hours, with calendar and pay rules | Assuming a paid hour produces one full standard hour |
| Subcontracting | Authorized transfer of work to an external supplier | Ignoring transport, quality, lead time, and supplier capacity |
| Incremental cost | Change in relevant future cost between alternatives, USD per scenario | Charging costs that do not change or omitting new costs |
| Opportunity cost | Value of the best forgone alternative, in USD | Treating allocated overhead as a cash cost without evidence |
| Contribution margin | Revenue less relevant variable costs, USD per unit or order | Treating revenue alone as the value of capacity |
| Decision right | Authority to approve a defined action, threshold, or exception | Assuming the person who calculates a plan may approve every change |
| Snapshot | Versioned inputs and assumptions as of a named timestamp | Comparing runs with different data and calling all changes process instability |

## 5. The mechanics, step by step

### Step 1 — State the decision and horizon

Input: the current business question, planning horizon, service obligations, and decision deadline.
Output: a named resource scope, bucket size, owner, and decision window.

1. State what must be decided: for example, whether to authorize overtime for next week.
2. Identify the relevant site, work centers, product families, and resource types.
3. Select a horizon long enough to expose planned order releases and replenishment lead times.
4. Select a bucket small enough to reveal actionable peaks.
5. Use daily or shift detail near execution if weekly totals can conceal conflicts.
6. Retain weekly or monthly views for longer-horizon staffing and capacity decisions.
7. Identify protected time fences and decisions that require escalation.
8. Record the source snapshot and time zone.

A capacity study without a decision horizon tends to produce an attractive chart with no actionable owner.
A 12-month aggregate staffing question and tomorrow's dispatch decision use different granularity.
Neither view replaces the other.

### Step 2 — Establish the order population

Input: dated MPS, MRP proposals, firm planned orders, released shop orders, and execution transactions.
Output: one deduplicated set of operations with remaining quantities and dates.

1. Freeze or identify the versions of the MPS and MRP run being evaluated.
2. Separate forecast-driven planned orders from orders linked to accepted customer commitments.
3. Include open work already released to the shop floor.
4. Include firm planned and proposed work according to the declared scope.
5. Exclude canceled orders and operations outside the horizon unless carry-in or carry-out is relevant.
6. For open orders, use remaining work, not the original total.
7. Identify whether setup has already occurred and whether the next run needs another setup.
8. Reconcile order identifiers across MRP, manufacturing execution, and scheduling systems.
9. Avoid counting a planned order and its converted shop order as two independent loads.
10. Mark demand that is not yet authorized so a scenario is not mistaken for an approved plan.

A firm order may be late even when its remaining workload is correctly recorded.
Compare its due date, current operation, and predecessor status, not only the order header date.
If execution reporting lags, explicitly flag uncertainty in completed and remaining quantities.

### Step 3 — Validate routing and standards

Input: effective-dated routing, production version, order quantity, standard setup and run times, and resource assignments.
Output: applicable operation requirements for each order.

1. Select the correct product, site, revision, and effective date.
2. Check that the operation sequence matches the approved process.
3. Identify the resources required concurrently at each operation.
4. Confirm whether a rate applies per unit, per batch, per hour, or per multiple-cavity cycle.
5. Convert all standards into a common unit such as hours per unit.
6. Identify fixed setup, quantity-dependent run, teardown, inspection, and cleaning time.
7. Distinguish machine time from labor time rather than copying one rate to the other.
8. Check whether a crew size multiplies labor-hours while elapsed time remains unchanged.
9. Determine whether time standards already include an allowance for performance loss.
10. Flag missing or expired standards before publishing a precise load percentage.

For an operation with one setup and a run rate per unit:

```text
machine load = machine setup hours + quantity units × machine hours per unit
labor load   = labor setup hours   + quantity units × labor hours per unit
```

If three operators jointly staff one machine for two elapsed hours, the operation consumes six labor-hours and two machine-hours.
If setup and run overlap or share a crew, model the actual assignment rather than blindly summing incompatible assumptions.
If a machine can process multiple units in parallel, use a validated effective rate or explicit cycle and cavity logic.

### Step 4 — Place operation loads in time

Input: order dates, lead-time offsets, operation dependencies, and scheduling rules.
Output: dated resource loads by operation and bucket.

1. Determine whether operation dates come from a finite schedule or an infinite-load backward/forward calculation.
2. Identify release, start, and due dates and their meanings in the source system.
3. Offset operations using the relevant working calendar, not automatically by calendar days.
4. Keep material availability and predecessor completion constraints visible.
5. Allocate setup to the bucket where it is expected to occur.
6. Allocate run time across buckets using the best supported operation-level timing rule.
7. If only an order due date is known, mark the allocation as an approximation.
8. Carry unfinished work across the horizon boundary rather than silently dropping it.
9. Preserve order, operation, resource, status, and date lineage for drill-down.
10. Check for loads assigned before the planning snapshot or outside operating shifts.

Putting all load in the order due week may be acceptable as a rough proxy only when disclosed and validated.
It can misplace earlier operations or hide a shift conflict.
A routing lead time containing queues should not be converted automatically into active resource load.

### Step 5 — Build machine calendar capacity

Input: installed machines, working shifts, holidays, maintenance, and expected loss assumptions.
Output: effective machine-hours by work center and bucket.

1. Enumerate each eligible machine rather than assuming the number of installed assets equals the number available.
2. Multiply scheduled machines by their scheduled operating hours in the bucket.
3. Remove holidays and approved shutdowns not already absent from the base calendar.
4. Deduct planned maintenance only on the affected machine and time interval.
5. Deduct known reservations or trials if they consume the same resource and are not already modeled as load.
6. Apply a documented expected-loss factor to the remaining scheduled hours when standards are ideal productive hours.
7. Keep separate scenarios if maintenance or breakdown assumptions are uncertain.
8. Do not subtract a planned event explicitly and also include it inside a factor calibrated on that same event.
9. Reconcile hours to the actual plant calendar and shift handover rules.
10. Keep equipment qualifications visible when pooling machines.

One transparent convention is:

```text
net scheduled machine hours = gross machine calendar hours - explicit exclusions
effective machine hours = net scheduled machine hours × residual productive factor
```

The residual factor must describe losses not already captured by explicit exclusions or time standards.
It is not automatically the same as overall equipment effectiveness (OEE).
OEE definitions and measurement boundaries must be checked before using an OEE value in a CRP denominator.
If a standard run rate is already based on historical actual performance, applying that performance loss again would understate capacity.

### Step 6 — Build labor and shared-resource capacity

Input: rostered employees, paid hours, qualifications, absence, training, and applicable work rules.
Output: effective labor-hours by skill pool and bucket, with named assumptions.

1. Count only people qualified and available for the operations in scope.
2. Start from scheduled hours for each relevant person and shift.
3. Subtract known leave, training, meetings, and other exclusions where not already removed.
4. Apply only a residual factor that is consistent with the labor time standard.
5. Separate specialist maintenance, inspection, and setup skills if they are not interchangeable.
6. Model shared staff across work centers without crediting the same person to every center at once.
7. Account for breaks, legal limits, collective agreements, and required supervision using local rules.
8. Check concurrent availability of machine, operator, fixture, and material.
9. Identify open vacancies and temporary staff with effective dates.
10. Document whether paid clock hours, attendance hours, or standard productive hours are being compared.

An employee can supply 40 paid hours without supplying 40 standard productive hours.
The appropriate conversion depends on how standards and calendars were established.
A generic percentage is a modeling assumption, not a substitute for a maintained roster.

### Step 7 — Reconcile load and supply

Input: operation-level load and effective capacity for each resource and bucket.
Output: gap, utilization, and an exception list.

1. Sum machine-hours by work center and date bucket.
2. Separately sum labor-hours by skill and date bucket.
3. Ensure each load category has a matching capacity denominator.
4. Calculate `gap = effective capacity - load` in hours.
5. Calculate `planned utilization = load / effective capacity × 100%` when capacity is positive.
6. Report a zero-capacity bucket with positive load as an exception, not a fabricated percentage.
7. Identify both the largest hour deficits and their customer or safety consequences.
8. Trace each overload to contributing orders, setups, rates, and status.
9. Compare nearby buckets for realistic pull-forward or push-out choices.
10. Record the version and assumptions that produced the exception.

A negative gap means demand for modeled standard resource time exceeds modeled effective supply.
A 110% planned utilization does not mean a machine literally runs 110% of the clock.
An aggregate positive gap may still conceal a shift, qualification, or sequence conflict.

### Step 8 — Diagnose before changing the plan

Input: exception list, order-level drill-down, current shop-floor evidence, and master-data history.
Output: validated exceptions and the causes worth addressing.

1. Check for duplicate orders, incorrect units, and expired routings.
2. Verify that the work has not already been completed or moved.
3. Check maintenance and staffing calendar accuracy with the responsible owner.
4. Look for a setup standard applied per unit rather than per order.
5. Look for a run rate expressed per 100 units but interpreted per unit.
6. Check whether yield was embedded in the quantity or separately applied.
7. Compare standards with comparable actual operations over a useful sample.
8. Investigate unusually high variation rather than adjusting every standard to one observation.
9. Distinguish a transient one-week overload from a recurring structural capacity shortage.
10. Escalate uncertain inputs rather than burying uncertainty in a single precise estimate.

Correcting a genuine data error is not the same as reducing real demand.
Keep the original snapshot and the correction rationale for audit and learning.
A planner should not relax a standard solely to make an overload disappear.

### Step 9 — Generate feasible remedies

Input: validated overload, demand priorities, alternative resources, supplier options, and constraints.
Output: candidate actions with their physical effects and prerequisites.

1. Move work to an earlier bucket if material, shelf life, and space permit.
2. Move work later only when the affected due date and customer consequences are accepted.
3. Authorize overtime or a temporary shift when machine, people, and supervision are simultaneously available.
4. Use an approved alternative machine or routing when capability, tooling, and quality are demonstrated.
5. Subcontract work only after supplier capacity, qualification, material ownership, logistics, and lead time are checked.
6. Split an order when partial completion helps, accounting for additional setup and transfer costs.
7. Change lot size when the inventory, setup, and service consequences justify it.
8. Adjust maintenance only through the maintenance and safety authorization process.
9. Reduce load through a real process improvement only after time savings are demonstrated and approved.
10. Replan the MPS or customer commitment if no operational option closes the gap.

Each option changes specific resource loads, capacity, or dates.
Write those changes explicitly before comparing costs.
A transfer to another machine can create an overload in the receiving work center.
A subcontract may relieve internal run time but retain internal setup, inspection, and receiving work.

### Step 10 — Calculate decision economics

Input: physical scenario differences, cost rates, service exposure, and the decision boundary.
Output: comparable incremental cost and consequence for each feasible option.

1. Establish the baseline plan and a common horizon.
2. Include costs that change because of the option, not historical sunk costs.
3. For overtime, include incremental pay, benefits where applicable, energy, supervision, and incremental scrap if evidenced.
4. For subcontracting, include supplier price, freight, inspection, and avoided internal variable costs.
5. For pull-forward, include incremental inventory carrying cost, storage, and obsolescence exposure.
6. For late work, estimate contractual charges and lost contribution only with explicit assumptions.
7. Avoid counting both lost revenue and lost contribution for the same units.
8. Check whether a labor cost is genuinely avoided or remains fixed during the decision horizon.
9. Express total scenario cost and the physical outcome, not only a cost per hour.
10. If comparing dollars per relieved bottleneck-hour, also show the total cost and service effect.

For a simple incremental choice:

```text
incremental scenario cost = new relevant cost - avoided relevant cost
cost per relieved bottleneck hour = incremental scenario cost / hours relieved
```

The second metric is useful for comparing capacity relief only when the denominator represents the same bottleneck and the options preserve equivalent service and quality.
It does not replace a total-cost comparison when remedies create different quantities or customer outcomes.
Fixed overhead allocations generally do not change just because an order moves; include them only when the allocation represents an actual incremental decision consequence.

### Step 11 — Decide and communicate

Input: feasible options, comparable economics, obligations, and decision rights.
Output: an approved, owned action or a documented escalation.

1. Present the baseline, gap, data confidence, and at least one feasible recovery path.
2. Show service, safety, quality, people, cost, and downstream resource impacts together.
3. Identify the individual authorized to approve overtime, alternate processes, supplier commitments, and customer date changes.
4. Record who recommends, who approves, who executes, and who must be informed.
5. State the threshold that routes a decision to plant leadership or the sales and operations planning (S&OP) forum.
6. Obtain approvals before changing controlled routings or making supplier commitments.
7. Update the production plan and resource calendars in the system of record.
8. Communicate affected dates to procurement, scheduling, customer service, and execution teams.
9. Retain the rejected alternatives and key reasons for the decision.
10. Set an expiry or reapproval condition when input assumptions may change.

The planner owns analysis and coordination unless local policy assigns otherwise.
The plant manager may own a spend decision within a delegated limit.
Quality and engineering retain their independent approval rights for controlled process changes.
Commercial staff must not promise a revised date until the relevant capacity and material checks are complete.

### Step 12 — Close the loop

Input: approved action, revised plan, schedule, execution feedback, and actual cost.
Output: verified implementation, updated forecast, and learning for the next cycle.

1. Recalculate all affected resource and time buckets after the change.
2. Check predecessor and successor operations, materials, and quality holds.
3. Confirm that the finite scheduler can place the recovered work within real shifts.
4. Monitor actual attendance, machine availability, throughput, and supplier acceptance.
5. Compare actual hours with standards using matched quantities and operation definitions.
6. Investigate systematic variance before revising master data.
7. Measure whether the customer date was achieved and whether the total remedy cost matched approval.
8. Escalate when the approved action is no longer feasible.
9. Keep a decision record linked to the original and revised plan snapshots.
10. Feed recurring overload evidence into staffing, investment, sourcing, or demand-planning decisions.

CRP is complete only as an analysis when the scenario has been handed to an authorized decision maker.
The business process is complete when the chosen action is executed or explicitly replaced and the effect is monitored.

### Additional mechanics — Moving work between periods

Pulling work forward changes at least two capacity buckets.
It is not enough to subtract the moved run hours from the overloaded week.
Calculate the earlier week's additional setup, run load, material need, and inventory exposure.
Then recalculate every resource in both periods.

Suppose a 150-unit portion of an order uses 0.060 machine-hours and 0.080 labor-hours per unit.
Moving those units earlier relieves 9.0 machine-hours and 12.0 labor-hours in the later week.
If the move creates a separate earlier run, assume it also requires a 2.0-machine-hour and 2.0-labor-hour setup.
The earlier week therefore receives 11.0 machine-hours and 14.0 labor-hours of additional load.
The original later-week setup remains if the remainder still runs there.

If earlier-week headroom is 12.0 machine-hours and 16.0 labor-hours, the aggregate balances remain positive.
The earlier machine gap falls to 1.0 hour and the earlier labor gap falls to 2.0 hours.
That calculation is conditional on usable material, tooling, qualifications, and compatible shift timing.
If only 100 units of material are available early, the 150-unit transfer is not a feasible candidate.
The planner must not claim a 9.0-hour later-week relief from units that cannot yet be started.

The economic boundary also moves.
Extra setup can consume paid capacity without an incremental wage if staff are already scheduled and no other production is displaced.
If the earlier resource is scarce, the relevant opportunity cost can be the contribution or service value of displaced work.
Additional finished inventory can incur financing, storage, insurance, damage, and obsolescence costs.
Use costs that actually change over the extra holding period rather than an annual charge on all existing inventory.

For an illustrative 150 units with inventory value of USD 18.20 per unit and an annual carrying rate of 20%, one extra week costs:

```text
150 units × USD 18.20/unit × 0.20/year × 1/52 year = USD 10.50
```

That figure excludes incremental setup supplies and assumes a simple annual-rate allocation.
If setup supplies cost another USD 26 and no other costs change, the incremental total is USD 36.50.
The low cost is irrelevant if material is unavailable, the earlier week cannot execute the work, or inventory cannot be stored.
If material can be made available through expediting, add the expediting cost and confirm the supplier's actual date.
Do not subtract the entire original setup from the later week unless no later run remains.

### Additional mechanics — Comparing production alternatives

Different remedies change different physical quantities.
An overtime block adds scheduled time; it does not automatically remove order load.
Subcontracting transfers some load; it does not necessarily eliminate internal inspection or material preparation.
A faster routing changes standard load; it may also require new tooling, qualification, or additional staffing.
Changing the due date shifts load in time; it may affect a customer promise.
An added shift may require an entire crew and supervisor even when the machine is the observed bottleneck.

Use a scenario ledger with one row for each affected resource and period.
Record baseline capacity, baseline load, proposed change, revised capacity, revised load, and revised gap.
Keep labor, equipment, fixtures, and external supplier capacity in separate rows.
Show the causal link between the decision and each changed number.
A single net number without the inputs is difficult to review and easy to misinterpret.

An option is physically feasible only when all material and resource constraints hold at the required dates.
An option is commercially feasible only when the appropriate supplier and customer obligations can be accepted.
An option is financially comparable only when its incremental costs cover the same decision boundary.
An option is authorized only when the relevant decision rights have been exercised.
These are distinct tests; passing one does not imply passing the others.

When alternatives differ in customer impact, do not rank them solely by USD per bottleneck-hour.
Calculate the total expected cost of each outcome and separately describe its delivery, quality, and safety exposure.
If late-delivery penalties are uncertain, show a sensitivity threshold rather than inventing a precise charge.
For example, if one option costs USD 180 more but avoids a possible delay, a verified expected delay cost above USD 180 would reverse a simple cost-only comparison.
That threshold ignores risk aversion and other service considerations, which should be stated separately.

### Additional mechanics — Managing a recurring overload

Repeated short-term overtime can be a symptom of a structural resource gap.
Compare several versioned weekly plans with actual load and capacity before proposing a permanent remedy.
Separate a real recurring workload increase from stale rates, duplicate order data, and calendar errors.
Normalize observed throughput for mix, product revision, and the number of setups.
Examine whether the same resource remains limiting after each recovery action.

If demand is persistently greater than sustainable capacity, options include additional staffing, equipment, process redesign, external supply, or demand shaping.
Each option has a different lead time, capital requirement, risk, and decision authority.
An equipment purchase cannot resolve a near-term shortage if delivery and qualification occur after the affected orders are due.
Hiring does not immediately create qualified capacity if training and supervision are binding.
Supplier diversification may increase resilience but requires ongoing qualification and commercial management.

Distinguish peak capacity from sustainable capacity.
An exceptional overtime week may be possible without being repeatable for months.
Safety, employee fatigue, planned maintenance, quality, and applicable working-time rules constrain repeatability.
Capacity expansion should be evaluated using realistic availability and utilization assumptions, not continuous nameplate operation.
The aggregate production plan should be revised when the expected demand trajectory exceeds the chosen sustainable resource strategy.

### Additional mechanics — A decision record that can be executed

A compact exception record should contain:

1. The planning snapshot, site, resource, and time bucket.
2. The capacity, load, gap, and units before intervention.
3. The orders and customer commitments exposed to the gap.
4. The source and confidence of the calendar and time standards.
5. Each candidate's physical resource effect and date effect.
6. Incremental cost, currency, assumptions, and sensitivity.
7. Safety, quality, people, supplier, and material prerequisites.
8. The recommendation and rejected alternatives.
9. The decision maker, approval time, execution owner, and deadline.
10. The revised plan version and the next verification point.

The owner should distinguish an action that has been proposed from one that is approved and scheduled.
A supplier's tentative offer is not an accepted purchase order.
A supervisor's proposed shift is not an approved roster.
A recalculated weekly surplus is not proof that the scheduler has placed every operation.
If a prerequisite fails, return the exception to the decision queue rather than silently assuming the capacity exists.

## 6. Worked example

### Business setting and boundary

Northbank Pumps is a fictional plant manufacturing precision pump components.
The planning team reviews one milling work center for the week beginning 10 May.
All amounts are illustrative USD; all hours refer to that same week unless stated otherwise.
The plant has two interchangeable qualified mills and a common pool of three qualified operators.
Each operation requires one mill and one operator simultaneously; the labor and machine standards differ because operators also perform associated handling.
The example assumes enough suitable coverage exists to realize the stated labor standards; exact overlapping assignments remain a finite-scheduling check.

A forecast suggested demand for 1,100 component units across three product orders.
The current production plan contains 600 units of A, 400 units of B, and 100 units of C.
The C order is released and has 100 units of work remaining; its setup has not yet been performed.
A and B are planned orders, not released commitments.
The 95% on-time delivery goal is a target, not an order.
A customer has accepted delivery of the C order; A and B dates are internal plan dates in this example.
The team will not change the customer commitment without commercial approval.

All material is assumed available before the planned operations.
No production loss is applied to the order quantities because they are already required starts under the approved yield convention.
All stated times are standard productive resource-hours; each setup occurs once in the week.
The objective is to resolve a one-week overload without changing the planned quantities or due dates.

### Order load

| Order | Status and remaining quantity | Machine setup | Machine run | Labor setup | Labor run |
|---|---:|---:|---:|---:|---:|
| A | Planned, 600 units | 2.0 h/order | 0.060 h/unit | 2.0 h/order | 0.080 h/unit |
| B | Planned, 400 units | 1.5 h/order | 0.050 h/unit | 1.5 h/order | 0.060 h/unit |
| C | Released, 100 units | 1.0 h/order | 0.080 h/unit | 1.0 h/order | 0.100 h/unit |

The machine-hour calculation is:

```text
A: 2.0 h + 600 units × 0.060 h/unit = 38.0 machine-h
B: 1.5 h + 400 units × 0.050 h/unit = 21.5 machine-h
C: 1.0 h + 100 units × 0.080 h/unit =  9.0 machine-h
Total                                         68.5 machine-h
```

The labor-hour calculation is:

```text
A: 2.0 h + 600 units × 0.080 h/unit = 50.0 labor-h
B: 1.5 h + 400 units × 0.060 h/unit = 25.5 labor-h
C: 1.0 h + 100 units × 0.100 h/unit = 11.0 labor-h
Total                                       86.5 labor-h
```

The setup remains once per order because each product is assumed to run once.
If C's setup had already been completed and could be retained, the remaining-load calculation would remove that setup.
If A were split into two independent internal runs, the extra setup would have to be added.
The standard labor total can exceed machine time because handling consumes labor away from the mill.
A common calendar balance does not prove the handling and machine attendance can be placed without overlaps.

### Calendar capacity

Each of the two mills is scheduled for five days at eight hours per day.
One mill has eight hours of planned maintenance, already excluded before the residual factor is applied.
The residual productive factor is 0.85, covering other expected losses not included in maintenance or the ideal standards.

```text
gross machine calendar = 2 mills × 5 days × 8 h/day = 80.0 machine-h
net scheduled machine   = 80.0 h - 8.0 h maintenance = 72.0 machine-h
effective machine       = 72.0 h × 0.85 = 61.2 productive-equivalent machine-h
```

The three operators each have 40 scheduled hours.
Known leave and training total 12 labor-hours and do not overlap the residual loss factor.
The 0.90 residual factor captures other nonproductive time relative to the labor standards.

```text
gross labor calendar = 3 operators × 40 h/operator = 120.0 labor-h
net scheduled labor  = 120.0 h - 12.0 h = 108.0 labor-h
effective labor      = 108.0 h × 0.90 = 97.2 productive-equivalent labor-h
```

These factors are scenario assumptions, not universal manufacturing benchmarks.
They must be calibrated to the local standard and calendar boundaries.
They are not a claim that a particular legal schedule or staffing pattern is permissible.

### Baseline comparison

| Resource | Effective capacity | Required load | Capacity minus load | Planned utilization |
|---|---:|---:|---:|---:|
| Mills | 61.2 machine-h | 68.5 machine-h | -7.3 machine-h | 111.9% |
| Operators | 97.2 labor-h | 86.5 labor-h | +10.7 labor-h | 89.0% |

Machine utilization is `68.5 / 61.2 × 100 = 111.93%`, rounded to 111.9%.
Labor utilization is `86.5 / 97.2 × 100 = 88.99%`, rounded to 89.0%.
The machine gap is negative by 7.3 productive-equivalent machine-hours.
The labor pool has 10.7 productive-equivalent labor-hours of aggregate headroom.
The weekly labor balance alone does not establish that qualified people are free at the same time as the mills.

The baseline is not physically executable under its own weekly machine assumptions.
The 7.3-hour deficit does not identify which order must be late.
A daily or shift-level schedule is required to determine sequencing and concurrency.
The planner should first verify maintenance, standards, order status, and material availability.

### Option 1 — Overtime

The supervisor can provisionally make one mill and one qualified operator available for 10 additional clock-hours in the week.
This availability is a proposed action, not yet an approved commitment.
The same residual productive factors apply to these added hours.
The incremental overtime wage including assumed relevant benefits is USD 38 per clock-hour.
Incremental machine energy is USD 4 per clock-hour.
No additional setup, freight, or quantity change is assumed.

```text
added machine capacity = 10.0 clock-h × 0.85 = 8.5 effective machine-h
added labor capacity   = 10.0 clock-h × 0.90 = 9.0 effective labor-h
revised machine gap    = 61.2 + 8.5 - 68.5 = +1.2 machine-h
revised labor gap      = 97.2 + 9.0 - 86.5 = +19.7 labor-h
incremental cost       = 10.0 clock-h × (USD 38/h + USD 4/h) = USD 420
```

The cost per incremental effective machine-hour supplied is `USD 420 / 8.5 h = USD 49.41/h`.
This is a supply-cost ratio, not a claim that every additional hour is fully used.
The option supplies more capacity than the 7.3-hour deficit; a smaller number of clock-hours might suffice if the work rules and staffing allow it.
In this scenario the supervisor offers a fixed 10-hour block, so the full block is costed.

### Option 2 — Subcontract part of A

An approved supplier has provisionally confirmed capacity for 150 units of A and can meet the required date.
The supplier price is USD 9.00 per unit; internal avoidable variable cost is USD 5.50 per unit.
Incremental freight and receiving together cost USD 75 per shipment.
A remains a single internal run of 450 units and retains its full internal setup.
The avoided internal variable cost is assumed genuinely avoidable in the decision horizon; the plant must verify that assumption.

```text
machine load relieved = 150 units × 0.060 h/unit = 9.0 machine-h
labor load relieved   = 150 units × 0.080 h/unit = 12.0 labor-h
revised machine load  = 68.5 - 9.0 = 59.5 machine-h
revised labor load    = 86.5 - 12.0 = 74.5 labor-h
revised machine gap   = 61.2 - 59.5 = +1.7 machine-h
revised labor gap     = 97.2 - 74.5 = +22.7 labor-h
incremental cost      = 150 × (USD 9.00 - USD 5.50) + USD 75 = USD 600
```

The incremental cost per machine-hour of internal load relieved is `USD 600 / 9.0 h = USD 66.67/h`.
Supplier acceptance is conditional until a purchase order is approved and accepted.
The option assumes no extra internal inspection load beyond the stated receiving cost; verify that assumption before approval.
It also assumes the supplier can use the required specification and that ownership of components and transport dates are settled.

### Decision and sensitivity

| Criterion | Fixed 10-hour overtime block | Subcontract 150 units of A |
|---|---:|---:|
| Incremental cost | USD 420 | USD 600 |
| Resulting machine gap | +1.2 h | +1.7 h |
| Resulting labor gap | +19.7 h | +22.7 h |
| Main dependency | Qualified staffing and executable shift placement | Supplier acceptance, specification, transport, receiving |
| Internal quantity changed | None | 150 units of A transferred |

Both options close the weekly machine gap under the stated assumptions.
Overtime is USD 180 cheaper for the same planned quantities and dates.
The planner recommends overtime, subject to workforce rules, approval, material availability, and a finite schedule confirming simultaneous resource availability.
The plant manager approves the spend within the local delegated authority; the supervisor confirms the roster.
If the schedule cannot place the extra time before the affected due dates, the team must reconsider subcontracting or escalate the plan.

If overtime's residual machine factor falls from 0.85 to 0.70, the block supplies `10 × 0.70 = 7.0` effective machine-hours.
The machine gap would remain `61.2 + 7.0 - 68.5 = -0.3` hours.
The cheaper option would then fail the capacity test unless another feasible adjustment is made.
If the USD 5.50 per unit internal variable cost is not actually avoidable, subcontract incremental cost rises to `150 × USD 9 + USD 75 = USD 1,425`.
This sensitivity changes economics, not the supplier's qualification or delivery risk.

The decision record should retain the version of the plan, the two options, the estimated costs, approver, roster condition, and next check time.
Actual overtime cost and output should be compared with the approved scenario after execution.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
Save it as a local script or paste it into a Python interpreter.
It reproduces the illustrative weekly resource balances and the two recovery options.
It does not create a finite schedule, validate staffing rules, or authorize a decision.

Assumptions:

- All three operations fall in the same weekly bucket.
- Quantities are remaining starts and setup has not been completed.
- Time standards and capacity factors use compatible productive-hour boundaries.
- The 10-hour overtime block is fully paid and uses the same residual factors.
- Subcontracting removes only 150 units of A run load; A's setup remains internal.
- All incremental costs are in USD and omit no other changes in this illustrative comparison.
- Floating-point results are displayed to two decimal places.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Order:
    name: str
    units: int
    machine_setup_h: float
    machine_run_h_per_unit: float
    labor_setup_h: float
    labor_run_h_per_unit: float

    def machine_load(self) -> float:
        return self.machine_setup_h + self.units * self.machine_run_h_per_unit

    def labor_load(self) -> float:
        return self.labor_setup_h + self.units * self.labor_run_h_per_unit


orders = [
    Order("A", 600, 2.0, 0.060, 2.0, 0.080),
    Order("B", 400, 1.5, 0.050, 1.5, 0.060),
    Order("C", 100, 1.0, 0.080, 1.0, 0.100),
]

machine_factor = 0.85
labor_factor = 0.90
machine_capacity = (2 * 5 * 8 - 8) * machine_factor
labor_capacity = (3 * 40 - 12) * labor_factor
machine_load = sum(order.machine_load() for order in orders)
labor_load = sum(order.labor_load() for order in orders)

print(f"Machine: load={machine_load:.2f} h, capacity={machine_capacity:.2f} h, "
      f"gap={machine_capacity - machine_load:+.2f} h")
print(f"Labor:   load={labor_load:.2f} h, capacity={labor_capacity:.2f} h, "
      f"gap={labor_capacity - labor_load:+.2f} h")

# Overtime adds paid clock hours to both resource calendars.
overtime_clock_h = 10.0
ot_machine = machine_capacity + overtime_clock_h * machine_factor
ot_labor = labor_capacity + overtime_clock_h * labor_factor
ot_cost = overtime_clock_h * (38.0 + 4.0)
print(f"Overtime: machine gap={ot_machine - machine_load:+.2f} h, "
      f"labor gap={ot_labor - labor_load:+.2f} h, cost=USD {ot_cost:.2f}")

# The internal A setup remains; only per-unit run time is removed.
a = orders[0]
subcontract_units = 150
sub_machine_load = machine_load - subcontract_units * a.machine_run_h_per_unit
sub_labor_load = labor_load - subcontract_units * a.labor_run_h_per_unit
sub_cost = subcontract_units * (9.0 - 5.5) + 75.0
print(f"Subcontract: machine gap={machine_capacity - sub_machine_load:+.2f} h, "
      f"labor gap={labor_capacity - sub_labor_load:+.2f} h, cost=USD {sub_cost:.2f}")
```

Expected output:

```text
Machine: load=68.50 h, capacity=61.20 h, gap=-7.30 h
Labor:   load=86.50 h, capacity=97.20 h, gap=+10.70 h
Overtime: machine gap=+1.20 h, labor gap=+19.70 h, cost=USD 420.00
Subcontract: machine gap=+1.70 h, labor gap=+22.70 h, cost=USD 600.00
```

The script subtracts the subcontracted run load, not the retained setup.
It reports standard productive-equivalent hours rather than elapsed clock hours.
The 10 clock-hours of overtime increase machine and labor capacity by different effective amounts because the assumed loss factors differ.
In a live implementation, retain the snapshot ID, operation date, resource ID, routing revision, and status alongside each load.
Use decimal arithmetic or integer minor currency units for financial posting when the accounting application requires exact rounding.

## 8. Data, standards, and regulatory context

### Minimum useful data contract

A CRP calculation is only as credible as the joins between orders, routings, resource calendars, and actual execution.
An order header alone rarely identifies enough information to allocate remaining operation load.
The table below is a practical starting contract, not a prescribed enterprise schema.

| Domain | Minimum fields | Business check |
|---|---|---|
| Order | Site, order ID, status, item, revision, quantity, release and due dates | One record per order version; no duplicate converted order |
| Operation | Order ID, operation number, start/end, completed and remaining quantity, setup state | Remaining work reconciles to execution |
| Routing | Item, site, revision, effective dates, sequence, resource, setup and run basis | Approved route applies to the actual product and date |
| Resource | Work-center code, asset or skill identity, qualification and pool | Pooled members are truly substitutable |
| Calendar | Resource, shift, time zone, open hours, maintenance, holidays | No duplicate exclusion or missing exception |
| Staff | Skill, roster, absence, authorization, labor constraints | Available hours are not allocated to multiple centers |
| Material | Availability date, quality status, quantity, expiry | A feasible operation has usable inputs at its start |
| Cost | Currency, effective date, marginal wage, energy, supplier and freight costs | Costs describe changes caused by the scenario |
| Snapshot | Run timestamp, source versions, plan status, owner | A reviewer can reproduce the decision |
| Execution | Operation confirmations, downtime reasons, quantities and timestamps | Actuals use comparable units and boundaries |

Preserve both raw values and converted values when units differ.
For example, a routing may hold minutes per 100 units while the capacity model reports hours per unit.
The conversion must preserve the basis: `minutes per 100 units / 100 / 60 = hours per unit`.
Test the conversion with a known order and an independent manual calculation.

### Master-data controls

A routing revision must be effective for the site, item, and production date being planned.
A replacement machine can have a different setup, run rate, inspection requirement, and staffing requirement.
The receiving work center needs its own calendar and a load check.
An approved alternate route should identify who can activate it and under what conditions.

A setup record needs a basis: per order, batch, tool change, cleaning event, or sequence transition.
A run-rate record needs a quantity basis and a definition of good or started units.
If yield is already embedded in planned starts, do not inflate run load by yield a second time.
If yield is operation-specific, calculate the required starts at each operation before applying its rate.

Actual and standard hours should have comparable boundaries.
A clocked attendance hour may include breaks or waiting that the standard excludes.
A machine-hour may include unattended running while the operator performs another task.
An improvement program needs reason codes and timestamps before it can distinguish rate loss, downtime, and incorrect standard work.

### Standards and legal boundaries

[ISA-95](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard) describes models and terminology for integration between enterprise and manufacturing operations systems.
It can support a shared understanding of production schedules, resources, and performance information.
It does not by itself prescribe this lesson's 0.85 factor, select an overtime policy, or guarantee the correctness of a site's data.
Implementation depends on the specific edition, scope, and system interface used.

[ISO 22400-2](https://www.iso.org/standard/56848.html) addresses definitions and descriptions of manufacturing operations management key performance indicators.
A local OEE or availability measure should be mapped to its precise numerator, denominator, loss categories, and edition before it is reused in capacity planning.
Do not assume that a percentage labeled OEE is interchangeable with a residual productive factor.
The standard's relevance is definitional and measurement-oriented; it is not a universal capacity target.

In the United States, the [Fair Labor Standards Act overtime provisions](https://www.dol.gov/agencies/whd/overtime) are administered by the U.S. Department of Labor.
Their applicability and the calculation of overtime depend on employee classification, workweek, jurisdiction, and other facts.
State law, collective agreements, contracts, and site policies may impose additional requirements.
The example's USD 38 per hour is an assumed incremental rate, not a statement of statutory pay.

In the European Union, the [Working Time Directive 2003/88/EC](https://eur-lex.europa.eu/eli/dir/2003/88/oj) sets minimum safety and health requirements for organization of working time, subject to national implementation and applicable exceptions.
Its detailed application requires local legal and human-resources review.
A capacity model does not authorize shifts that breach applicable working-time, rest, or safety obligations.

Safety, equipment guarding, environmental restrictions, and product-specific regulatory rules may also constrain an alternative.
Those requirements depend on jurisdiction and product; this lesson does not determine compliance.
Consult current primary rules and qualified site personnel before changing a process, roster, or supplier.
References and the legal context are educational, not legal advice; the sources were framed for this course's 2026 edition.

### Governance and retention

Use effective dates and controlled approval for routing changes.
Separate the role that proposes a standard from the role that approves it when local controls require separation.
Record the origin and rationale of maintenance changes and retain the previous calendar version.
Restrict personnel details to the minimum necessary for the decision and local privacy obligations.
Aggregate skill capacity for broad planning when individual records are not needed.

A decision log should capture the original exception, alternatives, calculations, approval, execution owner, and resulting plan version.
Keep units and time zones explicit, especially when plants and suppliers operate across midnight or daylight-saving transitions.
Versioning is essential when actual data arrive late or an MRP run changes the order population.

## 9. Industry practice and case evidence

### How a planning team typically works

In a mature process, a material planner publishes a dated order proposal and exceptions.
The capacity planner checks critical work centers and skill pools against current calendars.
The supervisor verifies near-term realities such as attendance, tooling, and maintenance.
The scheduler then turns an approved plan into an operation sequence that respects modeled constraints.
The team reviews exceptions again as execution information changes.

This division of labor is a general practice description, not a claim that every company uses the same roles or software.
Some plants combine the roles in one person; others use a central planning team and local production control.
The important controls are clear ownership, a comparable resource ledger, and an authorized response to infeasibility.

Near-term plans often use shift-level visibility, while longer-horizon plans use weekly or monthly capacity profiles.
A planner should not interpret a long-range weekly gap as a promise that a specific shift can execute the work.
A schedule can pass a weekly CRP check and still fail because a fixture is shared, a qualified operator is absent, or a predecessor finishes too late.

### Public evidence: Toyota and operational interruption

Toyota publicly described the effects of the 2011 Great East Japan Earthquake and the resulting production and supply-chain disruption in its [2011 annual report](https://global.toyota/en/ir/library/annual/2011/).
The report is a company disclosure, not independent proof of a particular CRP algorithm or a quantified planning benefit.
It illustrates that nominal installed production capability can be constrained by disruptions to facilities and supply networks.

[INFERENCE] A capacity plan for a manufacturer exposed to such disruption would need to revise both resource calendars and material availability rather than merely extend the usual production rate.
The public disclosure does not establish the fictional example's factors, the exact order-level decisions, or the costs of any particular recovery option.
Do not extrapolate a firm-specific recovery outcome from a general statement that a disruption occurred.

### Public evidence: NIST manufacturing extension work

The U.S. National Institute of Standards and Technology's [Manufacturing Extension Partnership](https://www.nist.gov/mep) describes a national network providing technical and business assistance to manufacturers.
NIST describes services and program structure; the existence of the program is not proof that any particular capacity technique produced a stated savings at a particular plant.
A useful operational lesson is to distinguish process improvement support from evidence about the actual timing and size of an individual plant's capacity relief.

[INFERENCE] A manufacturer that reduces validated setup time can potentially free capacity at the affected resource.
The relief must be measured against the number and sequence of actual setups, not assumed from an improvement slogan.
If a setup reduction applies to a nonconstraint, the system-level throughput effect may be negligible.
No quantified outcome is asserted here.

### Comparing service and economics in practice

A decision meeting should distinguish a firm customer order from a forecast-driven plan.
A capacity gap affecting a confirmed shipment may justify a higher incremental cost than one affecting a flexible replenishment proposal.
The economic comparison must still distinguish avoided late penalties, contribution margin, expediting cost, and nonfinancial obligations.
Avoid multiplying a full selling price by delayed units as though all revenue were permanently lost.

A subcontractor can have nameplate capability but lack a qualified process, a material slot, or transport time.
A second internal machine can have free aggregate hours but need a different fixture or an approved program revision.
A labor pool can have aggregate headroom but lack a trained operator on the only usable shift.
Each candidate therefore needs evidence of concurrent feasibility.

Weekly utilization is useful for triage and trend comparison when definitions remain stable.
It is a poor stand-alone performance target when it encourages oversized batches, unnecessary work-in-process, or hidden overtime.
A resource with spare capacity can be strategically useful for resilience or due-date protection.
A high average loading level is not proof of high throughput or customer service.

### Evidence hierarchy

A site should prefer its own validated time studies, maintenance calendars, and execution records for its capacity assumptions.
Public sources can explain concepts, reporting practices, and controls but do not establish a site's machine rate.
Vendor case studies may describe useful practices but should be checked for selection bias and measurement boundaries.
An improvement claim without a baseline, comparable period, and scope is not sufficient to set a planning factor.

Document what is observed, what is assumed, and what is inferred.
For a capacity decision, retain the actual approved hours and resource availability, not just a dashboard image.
Where evidence is weak, show a range or threshold sensitivity and identify the owner who will resolve it.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Duplicate orders | Load doubles after release | Planned and shop order both counted | False overload and unnecessary spend | Reconcile conversion IDs and statuses |
| Completed work reloaded | Open orders show original quantities | Execution confirmations ignored | Exaggerated remaining load | Use current operation and remaining quantity |
| Completed setup reloaded | Setup appears after verified completion | Setup state omitted | Inflated hours | Record setup status and retention rule |
| Missing setup | Many short lots show implausibly low load | Only run rates modeled | Hidden overload | Validate per-order or sequence setup |
| Wrong rate unit | One item dominates load unexpectedly | Minutes per 100 units read as hours per unit | Large false exception | Unit metadata and known-order check |
| Wrong routing revision | Actual process differs from model | Expired default route selected | Misplaced work and cost | Effective-date and approval controls |
| Double-counted downtime | Capacity falls after maintenance update | Maintenance deducted in calendar and factor | Artificially low capacity | Map loss boundaries explicitly |
| Double-counted yield | Starts and load both inflated | Scrap factor applied twice | Excess production and false overload | Declare quantity basis at every operation |
| Hidden labor constraint | Machine capacity is available but jobs wait | Qualified crew unavailable | Missed dates | Model skill calendars and concurrent shifts |
| Shared person counted twice | Two centers each show full staffing | One employee allocated to multiple pools | Unexecutable plan | Common resource identity and allocation rules |
| Bucket masking | Week is green but Monday is red | Weekly aggregation hides peak | Late order despite spare weekly hours | Shift or day check for near-term work |
| False pooling | Available machine cannot make the part | Capability or tooling differs | Rework or delay | Resource qualification and alternate routing |
| Load misplaced in time | Upstream operation starts after due date | All operations assigned to due week | Incorrect exceptions | Use operation offsets and predecessor checks |
| Queue mistaken for run | Lead time far exceeds observed touch time | Waiting time treated as active load | False capacity deficit | Separate elapsed and active time |
| Overtime assumed available | Capacity increases before roster approval | People and supervision not confirmed | Recovery plan fails | Conditional scenario and named approval |
| Unpriced subcontract | Supplier quote omits logistics and inspection | Scope differs across alternatives | Wrong economic choice | Comparable incremental cost and quality review |
| Excess utilization target | Queues and expedite frequency rise | Headroom treated as waste | Long lead times and poor resilience | Manage service and flow, not utilization alone |
| Changed commitment without authority | Customer sees a new date first | Planner acts outside decision rights | Contract and trust risk | Commercial approval and communication control |
| Stale snapshot | Two teams cite different capacity | Inputs refreshed at different times | Conflicting decisions | Version and timestamp each scenario |
| Suppressed uncertainty | Precise percentages with unstable standards | One-point assumptions conceal variance | Overconfidence | Sensitivity analysis and data-owner escalation |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Rates reflect the approved process | Convert units to hours | Product revision or equipment change | Compare route revision to work instructions |
| Setup is once per order | Calculate fixed load | Split lots or sequence-dependent cleaning | Check planned sequence and setup records |
| Orders contain remaining starts | Avoid double counting completed work | Delayed reporting or good-output quantity | Reconcile execution and yield basis |
| Factors exclude explicit maintenance | Avoid duplicate loss | Factor calibrated on total historical downtime | Trace numerator and denominator categories |
| Resource pool is interchangeable | Sum capacity and load | Different tooling or qualifications | Review part-machine qualification matrix |
| Labor and machine hours are compatible | Compare both resource constraints | Different standard-time boundaries | Map attendance and productive time definitions |
| Material is available at operation start | Consider capacity as the immediate constraint | Supplier delay or quality hold | Check dated usable inventory and receipts |
| Supplier option can meet date | Compare subcontract with overtime | No accepted slot or transport delay | Obtain conditional supplier confirmation |
| Avoided internal cost is variable | Estimate incremental subcontract cost | Fixed wages or unavoidable consumables | Finance review of cost behavior |
| Weekly aggregation is adequate for triage | Produce a manageable comparison | All work requires the same day or shift | Run a detailed finite scheduling check |
| Overtime factor matches normal factor | Estimate added productive hours | Fatigue, different staffing, or startup losses | Compare comparable overtime actuals |
| Residual factors are stable | Translate calendar to effective time | Unusual maintenance or performance shift | Scenario analysis and recent comparable history |
| All relevant resources are modeled | Close the real constraint | Shared fixture, inspector, or utility omitted | Walk the routing with execution owners |
| Order due dates are meaningful | Prioritize service consequences | Unapproved or stale promises | Reconcile with customer and commercial records |
| Cost horizon captures consequences | Compare relevant alternatives | Inventory or warranty costs outside horizon | Expand the boundary for material differences |

A bucketed CRP result is not proof of detailed feasibility.
Even a finite model is limited by the completeness and currency of its inputs.
The worked example assumes deterministic rates and no new breakdowns; actual outcomes can differ.
Use sensitivity ranges when uncertainty could reverse the decision.
Do not use this educational model to override safety, labor, quality, or commercial approval controls.

## 12. Exercises

Use the stated assumptions; show units and distinguish planning proposals from commitments.
Unless a question changes them, use the fictional Northbank case and USD currency.

### Exercise 1 — Explain the information flow

**Time:** 10 minutes.
**Task:** Explain how an MRP order becomes a CRP exception and then a decision.
**Deliverable:** A labeled flow of five to eight stages and a short paragraph.
**Acceptance criteria:** Include routing, operation dates, resource calendar, comparison, and decision owner.
Explain why a planned order and a customer commitment are not automatically equivalent.
State one reason a weekly capacity pass does not guarantee a feasible schedule.

### Exercise 2 — Calculate resource load

**Time:** 15 minutes.
**Task:** An order has 350 remaining units, 1.4 machine-hours of setup, 0.072 machine-hours per unit, 1.0 labor-hour of setup, and 0.090 labor-hours per unit.
Setup is still required and occurs once.
**Deliverable:** Machine and labor load in hours, with formulas.
**Acceptance criteria:** Show fixed and variable components separately and report both totals to two decimal places.
State what changes if the setup has already been completed and remains valid.

### Exercise 3 — Construct a calendar

**Time:** 15 minutes.
**Task:** Two machines each have five eight-hour shifts; 10 machine-hours of maintenance are explicitly excluded.
A residual factor of 0.80 applies only after maintenance and excludes the same maintenance loss.
**Deliverable:** Gross, net scheduled, and effective machine-hours.
**Acceptance criteria:** Show each step and explain why another maintenance deduction would be wrong.
State one reason the resulting weekly effective hours may not be available on the required day.

### Exercise 4 — Diagnose a gap

**Time:** 15 minutes.
**Task:** A work center has 59.0 effective machine-hours and 64.5 machine-hours of load in a week.
Its labor pool has 92.0 effective labor-hours and 85.0 labor-hours of load.
**Deliverable:** Gap and utilization for each resource plus a two-sentence diagnosis.
**Acceptance criteria:** Use capacity minus load and identify the machine deficit.
Avoid declaring the plan feasible solely from labor headroom or declaring a specific order late without sequence evidence.

### Exercise 5 — Compare recovery economics

**Time:** 20 minutes.
**Task:** A fixed overtime block supplies 6.0 effective machine-hours at an incremental cost of USD 330.
A qualified subcontract option removes 100 units at 0.060 machine-hours per unit.
Its price is USD 8.20 per unit, avoidable internal cost is USD 5.70 per unit, and freight is USD 40.
Both options preserve the same dates and require no additional setup.
The current gap is minus 5.5 machine-hours.
**Deliverable:** Capacity effect, revised gap, incremental total cost, and cost per effective machine-hour for each option.
**Acceptance criteria:** Show both physical and financial calculations.
Recommend the cheaper feasible option and name two conditions that require confirmation.

### Exercise 6 — Critique data quality

**Time:** 20 minutes.
**Task:** A dashboard shows 150% machine utilization after a planned order is released.
The released order is visible in the shop system, but the MRP proposal also remains visible.
The routing lists 30 minutes per 100 units; the load table labels the value as hours per unit.
Maintenance was deducted from the calendar and is also part of the historical loss factor.
**Deliverable:** A prioritized investigation with three specific defects, owners, and checks.
**Acceptance criteria:** Explain the effect of each defect without assuming the corrected overload is zero.
Give the correct conversion of 30 minutes per 100 units into hours per unit.

### Exercise 7 — Design decision rights

**Time:** 25 minutes.
**Task:** The cheapest recovery involves a new subcontractor and may change a customer delivery date.
Design a decision workflow for a regulated product without assuming this lesson establishes its regulatory requirements.
**Deliverable:** A role/action matrix and a short escalation record.
**Acceptance criteria:** Include planner, supervisor, quality/engineering, procurement, commercial owner, and spend approver.
Distinguish recommendation from approval and supplier commitment from a quote.
Identify a stop condition and the information required before the customer date changes.

### Exercise 8 — Design a sensitivity and monitoring plan

**Time:** 25 minutes.
**Task:** The Northbank overtime productive factor may be 0.70, 0.85, or 0.95.
The 10-hour block costs USD 420 in all scenarios.
**Deliverable:** A three-row machine-capacity scenario table and a monitoring plan.
**Acceptance criteria:** Calculate added capacity and revised gap for each factor.
Identify the minimum factor that exactly closes the 7.3-hour deficit for the fixed block.
Name one trigger for re-escalation and one source of actual evidence.

## 13. Self-check answers

### Exercise 1

A suitable flow is: MPS and order inputs → dated MRP proposals and open orders → operation routing and remaining quantities → time-phased resource load → calendar capacity → gap → validation and authorized decision.
The MRP proposal is a plan output; it becomes an external obligation only through the relevant release and acceptance process.
A weekly surplus can hide a Monday-only machine conflict, an unavailable fixture, or a missing qualified operator.
A strong answer identifies the specific resource, bucket, units, and decision maker.

### Exercise 2

```text
Machine = 1.40 h + 350 units × 0.072 h/unit = 26.60 machine-h
Labor   = 1.00 h + 350 units × 0.090 h/unit = 32.50 labor-h
```

Run components are 25.20 machine-hours and 31.50 labor-hours.
If the setup has already been completed and remains valid, remaining load becomes 25.20 machine-hours and 31.50 labor-hours.
If a later restart requires another setup, include the applicable restart time rather than automatically removing all setup.

### Exercise 3

```text
Gross = 2 × 5 × 8 = 80.0 machine-h
Net scheduled = 80.0 - 10.0 = 70.0 machine-h
Effective = 70.0 × 0.80 = 56.0 machine-h
```

The 0.80 factor is defined to exclude the already deducted maintenance.
Subtracting the 10 hours again would count that same event twice.
The remaining effective hours may be concentrated on different days from the required operation.

### Exercise 4

```text
Machine gap = 59.0 - 64.5 = -5.5 machine-h
Machine utilization = 64.5 / 59.0 × 100 = 109.3%
Labor gap = 92.0 - 85.0 = +7.0 labor-h
Labor utilization = 85.0 / 92.0 × 100 = 92.4%
```

The machine pool is overloaded by 5.5 effective hours; aggregate labor has 7.0 hours of headroom.
The machine deficit needs a remedy and a sequence check.
The data do not establish which order will be late or whether the available labor overlaps the required machine time.

### Exercise 5

```text
Overtime relief = 6.0 machine-h
Overtime revised gap = -5.5 + 6.0 = +0.5 machine-h
Overtime cost per supplied effective hour = USD 330 / 6.0 h = USD 55.00/h
Subcontract relief = 100 × 0.060 = 6.0 machine-h
Subcontract revised gap = -5.5 + 6.0 = +0.5 machine-h
Subcontract incremental cost = 100 × (USD 8.20 - USD 5.70) + USD 40 = USD 290
Subcontract cost per relieved machine-hour = USD 290 / 6.0 h = USD 48.33/h
```

Subcontracting is USD 40 cheaper under the stated equal-service assumptions.
Confirm supplier qualification and accepted capacity, plus the assumed avoidability of the internal cost.
Also confirm material, transport, inspection, and date feasibility before approval.
The cost per hour comparison is meaningful here because each option relieves six hours and preserves the same service outcome.

### Exercise 6

First, the planner and system owner should reconcile planned-to-released order IDs so only one remaining order is counted.
Second, the master-data owner should convert the routing rate correctly:

```text
30 min / 100 units / 60 min per hour = 0.005 h/unit
```

Third, the calendar or industrial engineering owner should separate explicitly deducted maintenance from the historical factor.
All three defects tend to inflate the reported utilization in the described direction.
Their combined correction cannot be quantified from the question alone and does not prove there is no real overload.
Retain the original and corrected snapshots and validate against representative actual operations.

### Exercise 7 rubric

A strong matrix gives the planner analysis and coordination responsibility and the supervisor responsibility for confirming floor feasibility.
It requires quality and engineering approval for the controlled process and procurement approval for the supplier engagement.
It identifies the authorized spend approver and the commercial owner who can approve and communicate any customer date change.
A quote or conditional slot is not an accepted purchase order.
A stop condition could be an unqualified supplier, insufficient material, or no approved customer date.
Before changing the customer date, document the feasible recovery plan, affected commitment, impact, and commercial approval.
A weak answer lets the planner unilaterally approve every change or omits the feedback loop.

### Exercise 8

| Overtime residual factor | Added effective machine-hours | Revised machine gap |
|---:|---:|---:|
| 0.70 | 7.0 h | -0.3 h |
| 0.85 | 8.5 h | +1.2 h |
| 0.95 | 9.5 h | +2.2 h |

The threshold is `7.3 / 10.0 = 0.73`, or 73% of the offered clock-hours, if the baseline and all other assumptions remain unchanged.
At 0.70 the block does not close the modeled gap.
A suitable trigger is a confirmed roster shortfall or a productive factor forecast below 0.73.
Use timestamped machine-state and production confirmations, with consistent definitions, as one source of actual evidence.
The monitoring plan should name the supervisor, review time, and escalation recipient.

## 14. Interview questions

1. **How would you explain CRP to a plant manager?**
   Strong: dated operation load versus dated resource availability, followed by an authorized remedy.
   Weak: a utilization chart automatically generates a feasible schedule.
2. **Why can machine capacity be sufficient while a job still waits?**
   Strong: qualified labor, fixture, material, predecessor, and shift concurrency can constrain execution.
   Weak: all resource-hours are interchangeable.
3. **How do you avoid double-counting losses?**
   Strong: reconcile calendar exclusions, standards, and factor numerator/denominator boundaries.
   Weak: apply every available percentage to gross calendar hours.
4. **What changes when an order is partly complete?**
   Strong: use remaining quantity and setup state, and reconcile confirmations.
   Weak: load the original full order again.
5. **Why is a weekly capacity pass not a schedule?**
   Strong: daily peaks, sequencing, dependencies, and concurrent resources remain unresolved.
   Weak: weekly spare hours guarantee every due date.
6. **How would you compare overtime and subcontracting?**
   Strong: compare incremental total costs, capacity effect, quality, timing, and approval prerequisites.
   Weak: compare supplier price with fully allocated internal cost without considering avoidability.
7. **How would you prioritize two overloaded orders?**
   Strong: examine commitments, feasible alternatives, contribution and penalties, and authorized business priorities.
   Weak: always choose the highest selling price without checking dates or constraints.
8. **What is wrong with a blanket 100% utilization target?**
   Strong: high loading can increase queues and reduce resilience; resource and measurement boundaries matter.
   Weak: all idle time is necessarily waste.
9. **What would you inspect if utilization doubled after MRP?**
   Strong: duplicate order conversion, unit conversions, routing revision, completed work, and calendar changes.
   Weak: immediately hire staff based on the dashboard.
10. **Who can change a customer commitment or process route?**
    Strong: local commercial and quality/engineering authorities approve their respective changes; planner coordinates evidence.
    Weak: whoever runs the capacity report can approve every change.

## 15. Key terms

- [Alternate routing](#4-definitions-and-vocabulary) — approved substitute sequence or resource.
- [Bucket](#4-definitions-and-vocabulary) — date interval used to aggregate hours.
- [Capacity gap](#5-the-mechanics-step-by-step) — capacity minus load in matching units.
- [Capacity pool](#4-definitions-and-vocabulary) — resource group with explicit interchangeability.
- [Capacity requirements planning](#3-the-core-mental-model) — dated resource-load and capacity comparison.
- [Commitment](#1-why-this-topic-exists) — obligation accepted by an authorized party.
- [Contribution margin](#4-definitions-and-vocabulary) — revenue less relevant variable costs.
- [Decision right](#5-the-mechanics-step-by-step) — authority for a specified action.
- [Effective capacity](#6-worked-example) — net calendar time adjusted for defined residual losses.
- [Finite scheduling](#3-the-core-mental-model) — timing and sequencing subject to modeled constraints.
- [Forecast](#1-why-this-topic-exists) — estimate of future demand.
- [Gross capacity](#6-worked-example) — scheduled resource time before declared exclusions.
- [Incremental cost](#5-the-mechanics-step-by-step) — change in relevant future cost between options.
- [Infinite loading](#4-definitions-and-vocabulary) — load placement without enforcing hard capacity.
- [Labor-hour](#6-worked-example) — one person's resource hour under the stated standard.
- [Load](#4-definitions-and-vocabulary) — required resource time from dated operations.
- [Machine-hour](#6-worked-example) — one machine's resource hour under the stated standard.
- [Maintenance exclusion](#6-worked-example) — explicitly unavailable calendar time.
- [Open order](#4-definitions-and-vocabulary) — released or firm work with a remaining requirement.
- [Opportunity cost](#4-definitions-and-vocabulary) — value of the best forgone alternative.
- [Overtime](#6-worked-example) — work outside the normal scheduled hours.
- [Planned order](#4-definitions-and-vocabulary) — proposed manufacturing order awaiting appropriate release.
- [Residual factor](#5-the-mechanics-step-by-step) — adjustment for defined losses not captured elsewhere.
- [Routing](#4-definitions-and-vocabulary) — ordered process steps and associated resource standards.
- [Run time](#4-definitions-and-vocabulary) — quantity-dependent resource time.
- [Setup time](#4-definitions-and-vocabulary) — resource time per stated setup event.
- [Snapshot](#8-data-standards-and-regulatory-context) — versioned inputs at a known time.
- [Subcontracting](#6-worked-example) — transferring approved work to an external supplier.
- [Time fence](#4-definitions-and-vocabulary) — boundary controlling plan-change authority.
- [Utilization](#6-worked-example) — planned load divided by effective capacity under this lesson's convention.
- [Work center](#4-definitions-and-vocabulary) — planning group of related manufacturing resources.
- [Yield](#4-definitions-and-vocabulary) — good output divided by input on a declared basis.

## 16. Sources

The sources below support concepts, standards context, or the named public evidence.
The fictional case, numerical assumptions, calculations, and decision rights are instructional and are not attributed to a source.

1. [ISA-95, Enterprise-Control System Integration](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard), International Society of Automation.
   Source for the scope of enterprise and manufacturing operations integration; consult the applicable edition for detailed requirements.
2. [ISO 22400-2:2014, Automation systems and integration — Key performance indicators for manufacturing operations management — Part 2: Definitions and descriptions](https://www.iso.org/standard/56848.html), International Organization for Standardization.
   Source for manufacturing KPI definitions; check current edition and applicability before implementation.
3. [Overtime Pay](https://www.dol.gov/agencies/whd/overtime), U.S. Department of Labor, Wage and Hour Division.
   Primary agency overview of U.S. federal overtime provisions; state and contractual rules can differ.
4. [Directive 2003/88/EC concerning certain aspects of the organisation of working time](https://eur-lex.europa.eu/eli/dir/2003/88/oj), European Parliament and Council, EUR-Lex.
   Primary EU legal text; national implementation and exceptions matter.
5. [Toyota Annual Report 2011](https://global.toyota/en/ir/library/annual/2011/), Toyota Motor Corporation.
   Company disclosure used only for the documented disruption context; not independent evidence of a specific capacity-planning result.
6. [Manufacturing Extension Partnership](https://www.nist.gov/mep), U.S. National Institute of Standards and Technology.
   Official description of the assistance network, not evidence of a specific plant outcome.
7. Hopp, Wallace J., and Mark L. Spearman, *Factory Physics*, 3rd edition, Waveland Press, 2011.
   Operations management reference for capacity, variability, queues, and production flow.
8. Vollmann, Thomas E., William L. Berry, D. Clay Whybark, and F. Robert Jacobs, *Manufacturing Planning and Control for Supply Chain Management*, 5th edition, McGraw-Hill, 2005.
   Reference for manufacturing planning, capacity planning, and execution context.

## 17. Links

- [Module 15 index](_index.md) — topic map and scope.
- [Planning cascade](01_planning-cascade.md) — decisions from aggregate planning to execution.
- [Manufacturing master data](02_manufacturing-master-data.md) — routings, work centers, and calendars.
- [MRP explosion](04_mrp-explosion.md) — material-driven order proposals and timing.
- [Module 15 survey](../../../course/15_manufacturing_production_planning.md) — production planning overview.
- [Production planning labs survey](../../../course/30_simulation_optimization_production_labs.md) — simulation and optimization lab context.
