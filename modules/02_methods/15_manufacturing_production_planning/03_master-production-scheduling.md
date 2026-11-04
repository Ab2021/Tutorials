# Master Production Scheduling

> Module 15 — Manufacturing and Production Planning  
> Topic 03 of 11  
> Estimated study time: 4 hours  
> Prerequisites: Module 01A (units and time), Module 10 (inventory), and manufacturing master data  
> Version 2026.5

## 1. Why this topic exists

The master production schedule (MPS) turns a demand and supply discussion into dated production decisions.
Without it, sales can see a healthy monthly forecast while the factory lacks the correct product in the week a customer needs it.
A plant can meet its monthly output target and still miss individual customer orders.
A schedule can also appear to cover every order while promising the same available stock to several customers.

The master scheduler owns the proposed quantities and dates for selected end items or planning items.
Sales owns the quality of customer commitments and the communication of revised dates.
Operations owns the reality of the capacity, material, maintenance, and quality constraints it reports.
The supply planning team coordinates a decision when these views disagree.
Ownership is a decision right, not permission to change another team's facts silently.

A useful MPS says which product is to become available, where, how much, and when.
Its output is a plan for supply, not proof that every component and resource is available.
The plan drives dependent material requirements and must survive capacity and execution checks.
A weekly review makes deviations visible before they become an urgent daily negotiation.
A freeze makes near-term changes exceptional rather than impossible.

Consider a medical-device accessories plant with a good monthly order book.
A late customer request pulls a sterile connector into the same week as a scheduled maintenance shutdown.
Accepting the request without checking the bottleneck could delay several existing customers.
Refusing it without checking the actual available inventory could lose profitable business unnecessarily.
The MPS process makes the tradeoff visible and assigns someone to decide.
This is a teaching scenario, not a reported company case.

A forecast is an estimate of future demand, expressed here as units per week.
A production plan is a proposed quantity and receipt date, also in units per week.
A target is a desired outcome, such as keeping 25 units as a buffer.
A commitment is a promise to a customer, with a quantity and a date.
These four objects can disagree without any one of them being wrong.
The job is to reconcile them and record the choices.

The cost of an unstable plan includes setup changes, premium freight, short component runs, overtime, and lost delivery reliability.
A rigid plan also has costs: unfilled orders, obsolete inventory, and missed demand shifts.
The correct policy balances those costs rather than pursuing stability or utilization alone.
This unit gives you a calculation method and an operating cadence for making that balance explicit.

## 2. Learning outcomes

After completing this unit, you can perform the following tasks.

1. **Classify** a forecast, a production plan, a buffer target, and a customer commitment without adding the same demand twice.
   Exercise 1 checks your classification and your explanation.
2. **Calculate** projected available balance (PAB) from beginning inventory, dated receipts, and the selected demand-consumption rule.
   Exercise 2 checks each weekly balance and the buffer breach.
3. **Compute and interpret** discrete available to promise (ATP), including the treatment of a negative interval.
   Exercise 3 requires an allocation audit and a statement of promise limits.
4. **Choose** a planning time fence and a demand time fence for a specific lead-time and change-cost situation.
   Exercise 4 asks you to justify both boundaries and the authority inside them.
5. **Diagnose** a rough-cut capacity problem and compare a feasible schedule change against overtime.
   Exercise 5 requires an explicit calculation and a decision record.
6. **Design** an exception workflow with named owners, a response deadline, and a customer communication rule.
   Exercise 6 tests your governance design.
7. **Critique** the data, assumptions, and measures that make a published MPS unreliable.
   Exercise 7 asks you to identify defects and tests.
8. **Reconcile** a revised order against a baseline and explain the effect on material planning and promises.
   Exercise 8 checks the end-to-end change logic.

The outcomes are deliberately operational.
You should be able to explain your calculations to a planner who has to act on them.
You should also be able to tell the customer-facing team what the calculation does not guarantee.

## 3. The core mental model

### 3.1 The schedule is a controlled buffer

```text
Demand signals                        Supply and execution signals
forecast + customer orders            stock + receipts + capacity + constraints
          |                                           |
          +-------------------+-----------------------+
                              v
                   Demand policy and netting
                              |
                              v
               Master production schedule by item/week
                 |              |                 |
                 v              v                 v
            PAB / buffer    ATP and allocation   MRP / capacity checks
                 |              |                 |
                 +--------------+-----------------+
                                v
                Exceptions -> decision -> published baseline
                                ^
                                |
                 shop confirmations and order changes
```

The MPS is between aggregate business intent and executable detail.
Sales and operations planning (S&OP) typically agrees aggregate supply and demand by family over a longer horizon.
The MPS disaggregates that intent to selected item, location, and time buckets.
Materials requirements planning (MRP) uses parent requirements and supply decisions to calculate dated component needs.
Capacity planning challenges whether the required work fits available resources.
Detailed scheduling sequences specific operations, people, and machines.
Actual production and customer orders then change the next review.

The arrows are feedback loops, not one-way handoffs.
An MRP shortage can make a proposed MPS receipt unattainable.
A bottleneck overload can require a different receipt week.
A customer order cancellation can release stock but not necessarily release a production slot without cost.
A plan should not be published as feasible until the defined checks and owners have acted.
Even then, it is feasible only against the data and detail included in those checks.

### 3.2 Four distinct lenses

The **demand lens** asks what quantity is expected or already ordered.
The **inventory lens** asks what balance remains after dated demand and supply.
The **promise lens** asks what quantity is not already committed under the allocation policy.
The **capacity lens** asks whether work required to create planned receipts fits resource availability.
No single lens replaces the others.
A positive PAB can coexist with no ATP in a particular interval.
Positive ATP can coexist with a capacity or material shortage if the planned receipt is unverified.
An available machine does not imply that a quality-released component exists.

### 3.3 Dates and responsibility

A receipt date is when a finished unit becomes usable for the modeled purpose.
A release date is when the production order must begin under the assumed lead time.
A requested date is the date the customer wants.
A confirmed date is the date the company has promised.
A ship date and a customer arrival date are not automatically the same.
The time difference can include inspection, warehouse handling, transport, and customer receiving calendars.

The master scheduler decides a proposed MPS within delegated policy.
Customer service manages individual order dates and customer notification.
Materials planning verifies purchased and manufactured components.
The plant verifies calendar, labor, bottleneck, and execution status.
A designated escalation owner chooses when one team's constraints cannot all be met.
The decision record preserves both the accepted and rejected alternatives.

### 3.4 The buffer has two meanings

Inventory can buffer uncertainty in demand or supply.
The MPS also buffers organizational disagreement by turning it into dated choices.
Neither buffer is free.
Inventory consumes working capital and can expire or become obsolete.
Decision buffers consume time and need a clear escalation path.
The aim is not to remove all change but to distinguish controlled change from avoidable churn.

## 4. Definitions and vocabulary

The following terms apply throughout this unit.
Products, locations, time buckets, units of measure, and scenario versions must accompany each quantity.
Unless stated otherwise, the examples use finished units in weekly buckets.

| Term | Definition | Common misuse | Grain or unit |
|---|---|---|---|
| Master production schedule (MPS) | Dated supply plan for selected master-scheduled items | Treating it as the machine-level dispatch sequence | Item × location × week, units |
| Master-scheduled item | End item, option, or planning item explicitly controlled in MPS | Assuming every component must be master scheduled | Item and location |
| Demand forecast | Expected demand before all orders are known | Treating a forecast as a firm customer promise | Item × location × week, units |
| Customer order | A recorded customer requirement with quantity and date | Assuming all order statuses are legally firm | Order line, units and date |
| Commitment | A quantity and date communicated under the firm's promise policy | Calling unapproved demand a commitment | Order line, units and date |
| Independent demand | Demand originating outside the parent-component explosion | Calling a component requirement independent without tracing its source | Item × date, units |
| Dependent demand | Requirement derived from a parent or another modeled relationship | Adding parent-derived demand again as a customer order | Component × date, units |
| Forecast consumption | Rule that reduces forecast exposed to netting when eligible orders arrive | Adding full forecast to overlapping orders | Defined item, date window, units |
| Demand time fence (DTF) | Boundary at which the demand source or consumption policy changes | Assuming every system uses the same inside-fence rule | Date and item/location |
| Planning time fence (PTF) | Boundary controlling automatic replanning of near-term supply | Treating it as a universal legal prohibition on changes | Date and item/location |
| Frozen zone | Interval where schedule changes require explicit authority | Assuming no emergency change can ever occur | Date interval, approver |
| Slushy zone | Interval where changes need limited review or incur constraints | Treating the name as a standardized system configuration | Date interval, policy |
| Liquid zone | Interval where ordinary replanning is allowed | Promising all future production before material and capacity checks | Date interval, policy |
| Firm planned order | Planned supply protected from specified automatic changes | Equating firm status with physically released or completed supply | Order, units and dates |
| Released production order | Order authorized for execution | Assuming release ensures completion by the receipt date | Order, operation dates |
| Planned receipt | Supply expected to become available in a particular period | Placing it at order release rather than availability | Item × week, units |
| Scheduled receipt | Existing expected incoming supply, such as an open order | Counting the same production order as both scheduled and planned | Order line, date and units |
| Beginning on-hand | Eligible inventory available at the opening instant | Including blocked, consigned, or already dispatched stock without policy | Item × location, units |
| Projected available balance (PAB) | Time-phased inventory after selected supply and netted demand | Calling PAB a customer promise | End-of-week units |
| Safety stock | Inventory target or reserve used to protect against uncertainty | Automatically assuming it is legally unpromiseable | Item × location, units |
| Available to promise (ATP) | Uncommitted supply available under a stated promise rule | Treating an unvalidated planned receipt as a physical unit | Supply interval, units |
| Discrete ATP | ATP calculated at an opening or supply receipt interval | Reporting zero in a non-receipt week as zero cumulative promise capacity | Receipt interval, units |
| Cumulative ATP | Promise capacity accumulated through a requested date under allocation rules | Adding positive intervals while ignoring earlier shortages | Requested date, units |
| Allocation | Reservation or entitlement of supply for a customer, channel, or purpose | Assuming unallocated ATP can be spent by all channels independently | Allocation bucket, units |
| Backorder | Committed or accepted demand past its required date and not fulfilled | Treating it as a new future order without removing the old one | Order line, units and overdue date |
| Lead time | Assumed elapsed or working time from release to availability | Ignoring calendars, inspection, or actual variability | Working days or weeks |
| Cumulative lead time | Longest relevant chain of component and production lead times under stated assumptions | Summing all parallel paths | Time, with BOM path |
| Lot size | Quantity rule for a production or purchasing order | Treating the nominal lot as a hard constraint without checking policy | Units per order |
| Rough-cut capacity planning (RCCP) | Approximate capacity check for a proposed MPS | Treating an aggregate check as a finite schedule | Resource × period, hours |
| Bottleneck | Resource constraining the system's throughput under the current conditions | Calling every heavily used resource the permanent constraint | Resource and period |
| Load | Required resource time for proposed work | Comparing quantities with hours without a conversion | Resource × period, hours |
| Available capacity | Usable resource time after calendar and policy exclusions | Using nameplate time as net available time | Resource × period, hours |
| Exception | A condition that needs review under a defined rule | Treating every numerical difference as equally urgent | Rule, severity, due date |
| Schedule nervousness | Frequent changes to planned quantities or dates across planning runs | Counting a new order as a reschedule without a baseline | Change count or units per comparable cohort |
| Pegging | Trace from a supply or requirement to the demand or parent it supports | Inferring a unique customer allocation from unpegged inventory | Order/requirement relationship |
| Baseline | Published, versioned plan used for comparison | Comparing plans at different grains or different horizon dates | Version and publication time |
| Rescheduling | Moving or changing an existing planned supply or demand date | Counting a replacement order as additional net supply | Order ID, old/new values |

Vocabulary differs among enterprise planning systems.
The calculation and configuration, not a screen label, determine what a number means.
Document local synonyms where they differ from this unit.

## 5. The mechanics, step by step

### Step 1 — Establish the decision and horizon

**Input:** the business decision, the customer service promise, the planning cycle, and cumulative lead times.
**Action:** select the items, locations, dates, and time bucket appropriate to that decision.
**Output:** a scoped MPS calendar with an owner and publication cutoff.

Start with the question the plan must answer.
For a stocked finished good, the question may be how many units should become available in each week.
For a configured product, a planning bill or option-level plan may be more useful than a forecast for every possible end configuration.
For a make-to-order item, customer orders and capacity may dominate forecast-driven stock replenishment.
Do not transfer a make-to-stock rule unchanged to a project or engineer-to-order business.

The horizon must reach far enough to expose supply decisions that must be made now.
A horizon shorter than the relevant cumulative lead time hides future component releases.
A horizon much longer than useful data may create false precision.
Weekly buckets make a weekly MPS readable but conceal within-week sequence and timing.
A daily execution schedule may be necessary inside the first week.

Define whether the week begins on Monday and ends on Sunday, or use the plant's actual calendar.
Define the timezone for the snapshot and the cutoff for order changes.
State whether availability means manufacturing completion, quality release, or warehouse availability.
Use one definition throughout a version of the plan.

### Step 2 — Choose the master-scheduled item

**Input:** product structure, customer order patterns, aggregation rules, and control points.
**Action:** select the level where demand can be forecast and supply can still be controlled.
**Output:** a list of item-location combinations and their relationship to parent and component plans.

Master scheduling at finished-item level gives clear customer visibility.
It can create excessive volatility if thousands of low-volume variants share the same components.
Scheduling an intermediate product can better match a postponement strategy.
In that case, the final configuration still needs its own promise and assembly-capacity rules.

Record the mapping from family-level S&OP to items.
Reconcile the sum of item plans to the family target after unit conversion.
Explain any gap due to mix, inventory strategy, or a deliberate capacity choice.
Do not force the sum to agree by hiding unmet demand.

Check whether an item is supplied by more than one plant.
A plant-level MPS and a network-level demand plan need a sourcing rule.
Transfers between locations consume transport and can have different receipt and availability dates.
A shared global inventory total is not immediately available to every customer.

### Step 3 — Take a trustworthy snapshot

**Input:** on-hand inventory, open orders, purchase and production receipts, forecasts, and status codes.
**Action:** capture data at a common as-of time and reconcile quantity and status.
**Output:** a versioned item-location-week table with provenance and exceptions.

Exclude units that are not eligible under the stated availability policy.
Examples include scrapped material, unapproved quality holds, and stock in a different legal entity.
A conditional release can be included only as a scenario with an owner and release date.
Identify stock already allocated to another channel or order.
Do not subtract the same allocation twice if the source on-hand value is already net of it.

Open order quantities should be remaining quantities, not original quantities when partial shipments occurred.
Check that the date is the date appropriate for the planning calculation.
A requested delivery date may need conversion to a warehouse ship date.
A forecast may already exclude firm orders or may require explicit consumption.
Record that semantic choice with the extract.

Separate existing scheduled receipts from proposed MPS receipts.
A production order converted from a planned order must replace the planned order, not duplicate it.
Identify late receipts and orders that are already past due.
Use an explicit backorder treatment rather than losing these records outside the horizon.

### Step 4 — Select the demand-consumption rule

**Input:** forecasts, orders, order statuses, DTF, and forecast-consumption configuration.
**Action:** calculate the demand used for each planning bucket without double counting overlapping signals.
**Output:** a dated netted-demand series and a trace from order to forecast consumption.

A common simplified rule for one bucket is `demand = max(forecast, orders)`.
This is appropriate only when forecast and orders describe the same demand population and bucket.
If a special project order is not included in the forecast, add that incremental order separately.
If a forecast has already been netted by the source system, applying `max` again can understate demand.

Production systems may consume forecast backward, forward, or within a specific window.
They may use customer, channel, product family, or planning item as a consumption key.
An order in week 2 can consume forecast in week 3 under one configuration but not another.
The rule affects both the total and the timing of net requirements.
Do not call a weekly `max` calculation an exact reproduction of an unspecified enterprise resource planning (ERP) configuration.

A DTF may specify orders-only demand inside the boundary and forecast consumption outside it.
Another organization may retain forecast inside the fence for unbooked demand.
Both can be defensible if documented and consistently applied.
The worked example deliberately uses weekly `max` in every bucket to keep the calculations auditable.
Its frozen zone therefore controls supply changes, not a change in the demand formula.

**Branch:** if orders and forecast cover different populations, segment them first.
If the order is a substitute for forecast demand, consume it once.
If the order is incremental, add it once.
If the overlap is unknown, flag the ambiguity and use a scenario rather than claiming certainty.

### Step 5 — Calculate projected available balance

**Input:** eligible opening stock, netted demand, scheduled receipts, and proposed MPS receipts.
**Action:** roll the inventory equation forward one period at a time.
**Output:** end-of-period PAB and any negative-balance or buffer exception.

For week `t`, let `PAB[t-1]` be the previous end balance.
Let `S[t]` be existing scheduled receipts and `M[t]` be non-duplicated MPS receipts.
Let `D[t]` be demand after the selected consumption rule.
Then `PAB[t] = PAB[t-1] + S[t] + M[t] - D[t]`.
The opening PAB is the eligible beginning inventory after any explicitly modeled adjustments.

A negative PAB represents a projected shortage under this convention.
Some systems instead show a positive stock balance and a separate backorder quantity.
The numerical presentation differs; the unmet requirement must still be carried forward.
Do not silently reset negative inventory to zero without tracking the shortfall.

Compare PAB with safety stock, not only with zero.
Below-buffer inventory can be a deliberate choice, but it is an exception to the target.
The buffer is not itself another customer order unless the replenishment logic explicitly treats it as a target requirement.
When no feasible receipt can prevent a breach, quantify the date and size and assign a decision owner.

### Step 6 — Propose supply quantities and dates

**Input:** PAB gaps, lot rules, lead times, capacity, and relevant S&OP boundaries.
**Action:** select candidate receipts that cover the demand and inventory policy where feasible.
**Output:** a versioned proposed MPS with quantities, receipt dates, release dates, and reasons.

A lot-for-lot plan matches each modeled shortage exactly.
A fixed-lot or minimum-order policy may make it preferable or necessary to bring additional inventory forward.
A campaign can reduce changeovers but increase inventory and delay a different product.
Compare these choices on service, cash, setup, and feasibility, not utilization alone.

Offset receipt dates to release dates using the appropriate production and working calendar.
Include inspection and quality release if availability is defined after inspection.
If the calculated release lies in the past, label the receipt as an expedite or an already-late requirement.
A planning system may display it in the current period, but it has not made time available retroactively.

Record whether each quantity is proposed, firm planned, released, or completed.
The statuses control what changes require review and what execution can already affect.
A firm planned order is still a plan; a released order can still fail.
Do not mix the statuses into a single unlabeled supply row in the decision record.

### Step 7 — Check rough-cut capacity

**Input:** proposed order releases, resource usage rates, setup allowances, and net calendar capacity.
**Action:** calculate load by critical resource and relevant period.
**Output:** load-versus-capacity gaps and feasible alternatives for review.

A simple product-rate calculation is `load = planned units × hours per unit`.
Add setup, cleaning, qualification, and other resource time if they are not already included in the rate.
Apply yield consistently: if the rate is per started unit and yield is less than 100%, convert good units to starts.
Use the resource and operation date, not automatically the finished-goods receipt week.

Capacity must reflect maintenance, holidays, staffing, and any approved overtime.
Do not compare a 24-hour nominal day with an 8-hour staffed shift.
Also do not subtract downtime twice if the available-capacity calendar already excludes it.
RCCP is an approximation; its value is to expose material gaps early.

**Branch:** when a week is overloaded, compare shifting a receipt, overtime, a qualified alternate resource, and a revised customer date.
For each option state its cost, dependencies, lead time, and approval owner.
A faster line is not an available alternative unless the route, materials, people, and quality approvals support it.
Do not announce a capacity-based promise before the relevant option is authorized.

### Step 8 — Calculate available to promise

**Input:** eligible opening stock, eligible dated receipts, confirmed but unshipped orders, and allocation rules.
**Action:** calculate uncommitted supply at the start and at each receipt interval.
**Output:** discrete ATP by supply interval, an allocation audit, and a customer-facing promise limit.

Discrete ATP is a policy calculation, not a universal formula.
A common convention subtracts customer orders until the next MPS receipt from the current interval's supply.
For the first interval, use opening eligible stock plus any receipt in that first interval.
For a later interval, use that interval's eligible receipt.
Include orders due in the receipt week and before the next receipt.
Forecast demand is used for inventory planning, but is not subtracted in this particular ATP convention.

A negative discrete interval means its own receipt does not cover the orders assigned to that interval.
One possible policy reallocates the deficit to uncommitted ATP in earlier intervals, most recent first.
This is permissible only if those earlier units remain physically available and are not reserved elsewhere.
If earlier ATP is insufficient, report the residual shortage and stop automatic promising.
Do not use a future receipt to claim an earlier due date without an explicit late-delivery decision.

A positive ATP value is not an instruction to promise every unit.
Customer allocation, safety-stock policy, contractual priority, material confidence, and timing may restrict use.
The company may reserve the buffer or allow it to be used under authorization.
State which policy applies rather than subtracting the buffer in some places and not others.

When no receipt occurs in a week, discrete ATP is normally not newly created in that week.
Unconsumed earlier ATP may remain available for a later requested date.
A cumulative view must retain earlier commitments, allocation changes, and the timing of receipts.
It cannot be constructed by adding a future receipt to a promise due before that receipt.

### Step 9 — Set time fences and permissions

**Input:** lead-time distribution, change costs, capacity commitments, customer lead times, and authority matrix.
**Action:** define the frozen, slushy, and liquid zones and specify what each action requires.
**Output:** effective fence dates, demand rules, and an approved exception path.

Time fences are dates that roll as the planning date advances.
They are not permanent week labels.
A frozen interval protects work that has already consumed material, setup, or customer commitment.
A slushy interval allows controlled adjustments while preserving important constraints.
A liquid interval allows routine replanning, subject to the normal feasibility and approval process.

The PTF controls supply replanning behavior in the local system.
The DTF controls which demand sources or consumption rules are used, if configured that way.
They need not coincide.
Neither boundary should be set from a textbook default without checking actual lead times and business costs.

Firming a plan can prevent automatic date or quantity changes.
It does not prevent a human from changing the record if their permissions allow it.
Define who may override, what evidence is required, and who communicates downstream effects.
A genuine safety or quality stop overrides ordinary production objectives and requires its own escalation.

### Step 10 — Reconcile material feasibility

**Input:** MPS releases, effective bills of materials, inventory, supplier commitments, and component lead times.
**Action:** send the approved candidate to MRP or an equivalent material check, then review shortages.
**Output:** component exceptions linked to parent items, dates, and decisions.

The MPS is not the complete component plan.
MRP explodes parent releases through the effective bill, nets component supply, and offsets lead times.
If the component is unavailable at release, the finished receipt may be infeasible.
Pegging helps identify which planned receipts and customer commitments are exposed.

Validate revision, effective date, substitute eligibility, and quality status before accepting a material workaround.
A substitute can require engineering or customer approval.
A later component purchase date may be consistent in the system but impossible for the supplier.
An expedite is a proposed action until the supplier confirms the date and quantity.

### Step 11 — Publish a controlled baseline

**Input:** demand, supply, material, capacity, and customer decisions after review.
**Action:** approve a plan version and communicate its scope, exceptions, and remaining risks.
**Output:** a timestamped baseline and a list of actions with owners and deadlines.

A published plan includes the assumptions under which it was approved.
Record the source snapshot, scenario, calculation policy, item-location scope, and fence dates.
Attach the approved capacity and material exceptions rather than concealing them.
Make clear whether an unconfirmed supply date is included in ATP.

The plan should reconcile with the aggregate S&OP decision at the agreed family and period grain.
Explain variances caused by item mix, changed order timing, or newly reported constraints.
The purpose is transparency, not forcing false agreement between differently timed snapshots.
An approved change must propagate to dependent plans and affected customer commitments.

### Step 12 — Run exception governance

**Input:** differences between current facts and the approved baseline, with explicit exception thresholds.
**Action:** triage by impact and response deadline, decide, communicate, and close with evidence.
**Output:** an auditable action record and updated operational plan.

Each exception should identify item, location, affected period, quantity, and version.
It should also identify the cause, exposure, decision owner, response deadline, and next action.
Use a stable exception ID to avoid reopening the same issue under a new name each day.
If a condition persists, update its age and escalation rather than generating duplicate tasks.

An imminent missed customer commitment may require same-day review.
A future buffer breach may be reviewed in the weekly planning meeting if its component release is not yet due.
A master-data inconsistency should go to the data owner even when no immediate order is late.
Severity depends on time to act and consequence, not only on the unit count.

A planner proposes options and reports their implications.
The plant confirms whether execution resources are actually available.
Sales or customer service confirms whether a customer date or allocation can change.
The designated decision owner resolves conflicts within delegated authority.
Escalate unresolved decisions before the last feasible action date.

Close the exception only after the approved action is reflected in the systems that use it.
For a changed receipt, that can include MPS, MRP, capacity, ATP, and customer communication.
A comment saying "expedite" is not proof that the supplier accepted a new date.
Record the accepted date and the evidence of confirmation.

### Step 13 — Measure stability and service together

**Input:** comparable baseline versions, actual receipts, customer commitments, and change reasons.
**Action:** measure schedule changes at a fixed horizon and compare them with service and cost.
**Output:** a review of avoidable churn, justified overrides, and resulting outcomes.

One quantity-weighted instability measure is absolute change in planned units inside a defined window divided by baseline planned units in the same window.
Match item, location, and original calendar date across the two snapshots.
Specify whether canceled orders, new orders, and date shifts are counted separately.
A move of 20 units from one week to another creates 20 units removed and 20 units added in an absolute bucket-difference measure.
That is 40 units of bucket movement, not 40 physical units produced twice.

Report a companion order-based metric if double-sided bucket movement is confusing.
Report the baseline denominator and treatment of zero baseline explicitly.
Avoid calling a percentage a forecast accuracy or service measure.
A perfectly stable plan can be consistently late.

Review customer service, inventory exposure, overtime, and expedite cost beside stability.
Segment changes by root cause: demand, supply, capacity, data, quality, or management override.
Use a reason code and audit trail to separate necessary changes from preventable churn.
A metric should support discussion, not penalize the reporting of a necessary change.

## 6. Worked example

### 6.1 Business situation and snapshot

A fictional plant makes a single stocked connector kit, item K, at plant P.
The planning date is Monday before week 1.
The six-week horizon is chosen for instruction; a real horizon should cover the relevant cumulative lead time.
All figures are good finished kits, not component units.
One week is the planning bucket and all quantities in a bucket are assumed available before that week's demand.
Within-week sequencing and customer delivery transit are not represented.

The plant has 140 eligible kits at the start of week 1.
This inventory is physically available, quality released, and not already allocated outside the orders shown.
The inventory target is 25 kits at each week's end.
The target is a management buffer, not a customer commitment.
For this example ATP is gross of the buffer, and using the last 25 kits requires separate authorization.
There are no other scheduled receipts.

Sales provides an unconstrained forecast for the same customer population as the recorded orders.
The firm order row contains remaining quantities, not original order quantities.
The selected planning rule is the larger of forecast and orders in each week.
There are no incremental project orders in the data.
The rule applies across all six weeks despite the supply freeze.

| Week | Forecast, kits | Firm orders, kits | Netted demand, kits | Proposed MPS receipt, kits |
|---:|---:|---:|---:|---:|
| 1 | 68 | 61 | 68 | 0 |
| 2 | 72 | 78 | 78 | 120 |
| 3 | 75 | 60 | 75 | 0 |
| 4 | 80 | 70 | 80 | 140 |
| 5 | 70 | 40 | 70 | 0 |
| 6 | 65 | 30 | 65 | 120 |
| Total | 430 | 339 | 436 | 380 |

Forecast is an estimate of demand, not a promise of 430 kits.
The 339 ordered kits are customer commitments for this illustration.
The proposed 380-kit MPS is a supply plan, not a customer order.
The 25-kit buffer is a target and not an additive demand row.
The netted demand is 436 kits because week 2 orders exceed the forecast by 6 kits.

### 6.2 Roll the inventory forward

The formula is previous PAB plus that week's MPS receipt minus that week's netted demand.
No other scheduled receipt or inventory adjustment exists in this scenario.
The calculation uses end-of-week balances.

| Week | Opening PAB, kits | MPS receipt, kits | Netted demand, kits | Ending PAB, kits | Above 25-kit target, kits |
|---:|---:|---:|---:|---:|---:|
| 1 | 140 | 0 | 68 | 72 | 47 |
| 2 | 72 | 120 | 78 | 114 | 89 |
| 3 | 114 | 0 | 75 | 39 | 14 |
| 4 | 39 | 140 | 80 | 99 | 74 |
| 5 | 99 | 0 | 70 | 29 | 4 |
| 6 | 29 | 120 | 65 | 84 | 59 |

Week 1: `140 + 0 - 68 = 72` kits.
Week 2: `72 + 120 - 78 = 114` kits.
Week 3: `114 + 0 - 75 = 39` kits.
Week 4: `39 + 140 - 80 = 99` kits.
Week 5: `99 + 0 - 70 = 29` kits.
Week 6: `29 + 120 - 65 = 84` kits.
As a cross-check, `140 + 380 - 436 = 84` kits.
The minimum end balance is 29 kits in week 5, only 4 kits above the buffer target.
There is no end-of-week shortage in this simplified view.
There may still be a shortage on an individual day before a receipt arrives.

### 6.3 Calculate discrete ATP before adjustment

The first interval is week 1 because the next MPS receipt is in week 2.
The second interval covers weeks 2 and 3 because the following receipt is in week 4.
The third covers weeks 4 and 5 because the following receipt is in week 6.
The final interval covers week 6 through the end of the modeled horizon.
Later orders are not assumed to be zero in reality; they are outside this example.

| Interval start | Eligible supply, kits | Orders in interval, kits | Raw discrete ATP, kits |
|---:|---:|---:|---:|
| Week 1 | 140 | 61 | 79 |
| Week 2 | 120 | 78 + 60 = 138 | -18 |
| Week 4 | 140 | 70 + 40 = 110 | 30 |
| Week 6 | 120 | 30 | 90 |

The negative 18-kit interval is a coverage deficit relative to its own 120-kit receipt.
It does not mean total stock is negative in week 2.
The earlier interval has 79 kits of uncommitted ATP that can cover this 18-kit deficit under the stated policy.
After reallocating 18 kits from week 1, adjusted interval ATP is 61, 0, 30, and 90 kits.
The allocation trace is `week 1: 79 - 18 = 61` and `week 2: -18 + 18 = 0` kits.
The adjusted total is `61 + 0 + 30 + 90 = 181` kits.
The conservation check is `140 + 380 - 339 = 181` kits.

A request in week 3 could use up to 61 kits of the earlier uncommitted ATP in this static example.
That statement assumes no intervening orders, no allocations, and the same receipt timing.
A request in week 4 could use that remaining earlier ATP plus up to 30 newly available kits, subject to all policy checks.
A week 3 promise cannot use the week 4 receipt.
A week 6 promise may also depend on the 120-kit week 6 receipt actually being achievable.

ATP is greater than PAB because this convention subtracts customer orders, whereas PAB also protects forecast demand.
At the end of week 6, the difference is `181 - 84 = 97` kits, equal to `436 - 339` kits of forecast demand not yet ordered.
Promising all 181 kits would consume coverage intended for unbooked forecast demand and possibly the 25-kit buffer.
Whether that is acceptable is a commercial and inventory-policy decision, not a mathematical fact about ATP.

### 6.4 Challenge capacity

Assume a one-week production lead time and an illustrative critical-resource rate of 0.4 staffed hours per good kit.
Assume the rate already includes setup and expected yield losses for this calculation.
The required load is assigned to the week before each receipt.
The plant confirms 48 staffed hours of capacity in each of weeks 1, 3, and 5.
Week 2 has at least 8 otherwise unused staffed hours on the same qualified resource.
These are planning assumptions; the plant must confirm material availability and detailed sequence.

| Receipt week | Receipt, kits | Release/load week | Load at 0.4 hours/kit | Available, hours | Gap, hours |
|---:|---:|---:|---:|---:|---:|
| 2 | 120 | 1 | 48 | 48 | 0 |
| 4 | 140 | 3 | 56 | 48 | 8 overload |
| 6 | 120 | 5 | 48 | 48 | 0 |

The week 4 receipt requires `140 × 0.4 = 56` staffed hours in week 3.
The 48-hour capacity leaves an 8-hour overload.
At the same rate, the gap is `8 / 0.4 = 20` kits.
The original MPS therefore passes the simplified inventory check but fails this capacity check.
It should not be described as fully feasible.

### 6.5 Compare two decisions

**Option A: overtime.**
Add 8 approved staffed hours in week 3 and retain the 140-kit week 4 receipt.
At an incremental cost of USD 75 per staffed hour, the added cost is USD 600.
This assumes the resource, qualified staff, and required materials are available for those hours.
If the quoted USD 75 per hour is already a total rather than incremental rate, the comparison needs consistent costing.

**Option B: pull forward 20 kits.**
Release 20 kits in week 2 and receive them in week 3.
Reduce the week 4 receipt from 140 to 120 kits.
Week 2 load rises by `20 × 0.4 = 8` hours and week 3 load falls by the same amount.
The stated spare week 2 capacity and one-week lead time make the rough-cut arithmetic fit.
The operation must allow a 20-kit split, and components, quality review, and detailed sequence must be confirmed.

Option B changes PAB in week 3 from 39 to 59 kits.
Week 4 PAB remains 99 kits because the total of the two receipts is unchanged.
Weeks 5 and 6 remain 29 and 84 kits respectively.
At an illustrative carrying cost of USD 0.10 per kit per week, the extra one-week inventory costs USD 2.
This excludes any incremental setup, handling, material expedite, and quality costs.
If the split creates a USD 650 setup, the apparent USD 598 advantage over overtime disappears.

The preferred preliminary decision is Option B only if the split has no material extra setup and all feasibility checks pass.
Otherwise compare the full incremental cost of Option A with the split and any service impact.
A new receipt in week 3 also changes the discrete ATP interval boundaries.
Recalculate ATP from the updated orders and receipts rather than carrying forward the old interval values.
The master scheduler records the approved option and the capacity owner confirms the revised slot.
Customer service is notified if any confirmed date or allocation changes.

### 6.6 Apply the time fence

For this example the frozen supply zone is weeks 1 and 2, and weeks 3 and 4 are slushy.
Weeks 5 and 6 are liquid for ordinary replanning.
The PTF boundary is the end of week 2 in this local policy.
The DTF is also set at the end of week 2, but the demand calculation retains weekly `max` on both sides.
The DTF is therefore a review marker here rather than a switch to orders-only demand.

Option B adds a week 3 receipt, outside the stated frozen receipt zone, but requires a week 2 release.
The release touches the frozen execution window, so the plant must approve it even though the receipt is outside the fence.
An emergency change to the week 2 receipt would need the designated frozen-zone approver.
The rule is based on both release and receipt impact, not only the screen's displayed receipt date.

### 6.7 Record the exception

Exception ID `K-P-W3-CAP-01` identifies the 8-hour week 3 overload.
The cause is a 140-kit receipt whose release requires 56 staffed hours against 48 available hours.
The response deadline is before the week 2 release cutoff for Option B.
The master scheduler owns the proposal and the plant capacity lead owns the availability confirmation.
Materials planning confirms components for the 20-kit earlier release.
The production manager approves the schedule change within the delegated authority.
If a promised date changes, customer service owns the communication and acknowledgement.
Closure requires the approved revised schedule, material confirmation, and updated ATP calculation.

## 7. Code, query, or calculation

The following Python 3 program reproduces the original six-week PAB and discrete ATP calculation.
It uses only the standard library and can be copied into a file and run with `python mps_example.py`.
It is intentionally not a replacement for an ERP, finite scheduler, or transaction-safe order-promising service.
The code applies weekly `max` demand, uses receipts at the start of the bucket, and treats all displayed orders as firm and unshipped.
No safety stock is deducted from ATP; the 25-kit target is checked separately.
The final ATP interval ends at the supplied horizon, so omitted later orders are not covered.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Week:
    number: int
    forecast: int
    orders: int
    receipt: int


weeks = [
    Week(1, 68, 61, 0),
    Week(2, 72, 78, 120),
    Week(3, 75, 60, 0),
    Week(4, 80, 70, 140),
    Week(5, 70, 40, 0),
    Week(6, 65, 30, 120),
]
opening_stock = 140
buffer_target = 25


def projected_balances(rows, opening):
    balance = opening
    results = []
    for row in rows:
        demand = max(row.forecast, row.orders)
        balance += row.receipt - demand
        results.append((row.number, demand, balance))
    return results


def discrete_atp(rows, opening):
    # First interval starts at the opening; later intervals start at receipts.
    starts = [0] + [i for i, row in enumerate(rows) if i > 0 and row.receipt]
    intervals = []
    for position, start in enumerate(starts):
        end = starts[position + 1] if position + 1 < len(starts) else len(rows)
        supply = rows[start].receipt + (opening if start == 0 else 0)
        booked = sum(row.orders for row in rows[start:end])
        intervals.append({
            "week": rows[start].number,
            "raw": supply - booked,
            "adjusted": supply - booked,
        })

    # Earlier positive ATP covers later deficits, most recent first.
    # A residual negative value means the earlier supply was insufficient.
    for i in range(len(intervals)):
        if intervals[i]["adjusted"] >= 0:
            continue
        deficit = -intervals[i]["adjusted"]
        for j in range(i - 1, -1, -1):
            available = max(0, intervals[j]["adjusted"])
            transfer = min(deficit, available)
            intervals[j]["adjusted"] -= transfer
            deficit -= transfer
            if deficit == 0:
                break
        intervals[i]["adjusted"] = -deficit
    return intervals


pab = projected_balances(weeks, opening_stock)
atp = discrete_atp(weeks, opening_stock)
print("week demand PAB buffer_gap")
for week, demand, balance in pab:
    print(week, demand, balance, balance - buffer_target)
print("ATP start_week raw adjusted")
for interval in atp:
    print(interval["week"], interval["raw"], interval["adjusted"])
print("ATP total", sum(interval["adjusted"] for interval in atp))
load = 140 * 0.4
print("week 3 load, capacity, overload hours", load, 48, load - 48)
```

Expected output:

```text
week demand PAB buffer_gap
1 68 72 47
2 78 114 89
3 75 39 14
4 80 99 74
5 70 29 4
6 65 84 59
ATP start_week raw adjusted
1 79 61
2 -18 0
4 30 30
6 90 90
ATP total 181
week 3 load, capacity, overload hours 56.0 48 8.0
```

The `buffer_gap` column is PAB minus the 25-kit target; it is not a separate demand deduction.
The negative raw ATP in week 2 is resolved by consuming 18 kits of earlier ATP.
If an interval remains negative after the transfer loop, the example policy exposes a residual shortage.
A production service would also need transaction-safe reservations, eligibility checks, allocation policy, dates finer than a week, and durable audit records.
Those are outside the scope of this teaching calculation.

## 8. Data, standards, and regulatory context

### 8.1 Minimum data contract

An MPS extract needs an item identifier and a location identifier that are stable for the period being compared.
It needs the unit of measure, bucket start and end, timezone, and as-of timestamp.
It needs the source system and version for forecast, order, inventory, and receipt records.
It also needs order status and remaining quantity rather than only order creation quantity.
The data owner should define the quality status and ownership criteria for eligible stock.

For each receipt, preserve a unique order or plan-line identifier.
Preserve the original release date, current release date, receipt date, quantity, and firming status.
If a planned order converts to a production order, maintain the relationship between the records.
That relationship prevents a transition from being interpreted as two supplies.
Revisions require valid-from and valid-to dates rather than simply overwriting history.

For each demand record, retain the customer or channel where policy allows.
Record requested and confirmed dates separately.
Record the date used in the planning calculation and the conversion from delivery date to ship date.
For a forecast, record the forecast version, planning population, aggregation level, and consumption window.
This allows a reviewer to trace whether booked demand was already consumed.

Inventory reconciliation should include beginning physical stock, blocked stock, allocated stock, in-transit stock, and adjustments.
Different systems can use the same word for different eligibility rules.
A table labeled `available` is not a sufficient definition.
A physical count can be correct while the availability status is wrong.

### 8.2 Grain, units, and calendars

The minimum grain is normally item × location × date or week × version × supply or demand type.
Order-level detail is required to trace an individual promise.
Capacity data needs resource × period × shift or another appropriate time grain.
A product conversion from cases to kits must use the effective packing definition.
A rate per started unit and a plan in good units need an explicit yield conversion.

A weekly receipt is not necessarily available at the beginning of the week.
If demand arrives before the physical completion date, weekly PAB can overstate service.
An item can have sufficient weekly quantity but an impossible daily order sequence.
Use finer-grained checks for the frozen interval and for time-critical orders.

Calendar assumptions must include plant working days, holidays, maintenance, and staffed shifts.
Transport and customer calendars are distinct from plant calendars.
A lead time of one week in this unit is a simplified elapsed-bucket assumption.
A real implementation should use the local working calendar and capture variability.

### 8.3 Standards and interoperability

[ISA-95](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard) is a family of enterprise-control integration standards.
It provides terminology and models for exchanging information between business planning and manufacturing operations.
It does not prescribe a single ATP formula, time-fence length, or customer allocation decision.
Use the applicable part and version, and check a licensed copy when implementing a conformance requirement.

[GS1 identification standards](https://www.gs1.org/standards/id-keys/gtin) describe the Global Trade Item Number (GTIN) for identifying trade items.
A GTIN can support consistent product identification across trading partners.
It does not replace an internal plant-specific manufacturing item, routing, revision, or planning policy.
Pack-level differences and local crosswalks must still be maintained.

Electronic data interchange (EDI) can exchange purchase orders, acknowledgements, and shipment information between partners.
The agreed implementation guide determines which message fields mean requested dates, confirmed dates, quantities, and changes.
An inbound order message is not automatically a capacity-feasible commitment.
The parties still need a response policy, status mapping, and exception process.

These standards are conventions or contractual requirements only where applicable.
They are not blanket legal obligations on every MPS calculation.
The applicable regulation depends on product, jurisdiction, market, and business activity.
This unit is educational, not legal or regulatory advice.

### 8.4 Regulated and quality-controlled inventory

In regulated manufacturing, production and quality records can constrain what can be released for use or shipment.
For example, U.S. finished pharmaceutical manufacturing is subject to applicable current good manufacturing practice requirements in [21 CFR Part 211](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-211).
As of the version consulted for this unit, 21 CFR 211.165 addresses testing and release for distribution of drug products.
The regulation applies within its legal scope; it does not mandate this unit's ATP or weekly MPS algorithm.
A site quality unit and legal advisers should determine applicable requirements and release status.

A quality hold should not be transformed into an available quantity merely to eliminate a planning shortage.
A proposed release date may be modeled as conditional supply if it is clearly labeled and approved for scenario use.
A conditional supply should not be represented as an unconditional customer promise without the applicable policy.
Lot, expiry, serialization, and customer qualification can restrict which physical units serve a particular order.
A total in units can therefore overstate usable supply for a specific customer.

### 8.5 Controls and data stewardship

Use completeness checks for missing item, location, date, status, and unit.
Use uniqueness checks for duplicate order lines and converted planned orders.
Reconcile totals to the source systems at the same timestamp and grain.
Validate that open quantities are nonnegative and partial shipments are reflected.

Investigate receipts whose release dates are in the past without execution evidence.
Investigate an inventory status that changes from blocked to eligible without an approved event.
Investigate a forecast consumption rule that changes after a system upgrade.
Assign each failure to the owner who can repair the source, not only to the reporting analyst.

Record who can edit time fences, firm a plan, override an allocation, or approve a customer date.
Retain both the old and new value and the effective timestamp for critical changes.
Restrict customer detail in analytical extracts according to the organization's privacy and access rules.
A planning dashboard should expose data freshness so users know when ATP may have changed.

## 9. Industry practice and case evidence

### 9.1 What public evidence can and cannot establish

Public descriptions often disclose planning-system features rather than a company's exact MPS formula.
A vendor's documentation can establish how its own software is designed to behave.
It cannot establish that every customer configured it that way or achieved a specific business result.
A company report can establish what the company said publicly, not the complete counterfactual impact of a planning change.
The following examples distinguish those evidence types.

### 9.2 Oracle planning and discrete ATP

Oracle's public [Available to Promise documentation](https://docs.oracle.com/cd/A60725_05/html/comnls/us/mrp/atp.htm) describes ATP in its master scheduling and supply chain planning software.
This is product documentation, not independent proof of a particular plant's performance.
It illustrates that ATP is tied to a defined supply-and-order convention and system setup.
The exact treatment of negative ATP, consumption, and time fences must be checked against the deployed product and version.

The worked example in this unit is a teaching convention, not a claim that every Oracle configuration produces identical values.
Its explicit backward allocation avoids presenting a negative interval as if it were free supply.
A business evaluating a package should request test cases with orders before and after each receipt.
The acceptance test should also cover partial shipment, cancellation, reservation, and a past-due order.
The business should inspect whether promises are protected against concurrent users.

### 9.3 SAP and allocation-aware promising

SAP's official documentation describes [product allocation in advanced available-to-promise](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/f132c385e0234fe68ae9ff35b2da178c/cfc8aa5747c46c10e10000000a441470.html).
This is documentation of a vendor capability, not evidence of a specific customer's implementation.
It shows why a global available quantity may be further constrained by an allocation policy.
A channel or customer may have a quota even when aggregate supply appears positive.

The operational implication is that allocation ownership matters as much as the arithmetic.
A quota should have an effective period, eligible demand, and authority for exceptions.
A customer-facing employee needs to know whether an allocation is consumed on order entry, confirmation, or shipment.
The implementation must also define how unused allocation is released or carried forward.
These are governance design questions, not claims about one universal SAP configuration.

### 9.4 A named manufacturer: Toyota

Toyota publicly describes its [Toyota Production System](https://global.toyota/en/company/vision-and-philosophy/production-system/), including just-in-time and jidoka.
The source is a company description of its production philosophy.
It supports the observation that synchronized flow and responding to abnormalities are central themes in Toyota's stated approach.
It does not disclose a universal master-schedule time fence or prove that the six-week calculation above is used by Toyota.

**Inference:** a company pursuing synchronized flow needs reliable signals and a way to respond to disruptions across planning and execution.
That inference does not mean a completely frozen plan is always desirable.
Small lots, stable supplier processes, and fast feedback may change the economic value of a long fence.
A different manufacturer with long component lead times or costly qualification may need a different policy.
The evidence does not justify copying another plant's numeric boundary.

### 9.5 Comparing operating contexts

A consumer-goods plant may forecast at family level and use inventory to protect service across promotion uncertainty.
The MPS challenge is to avoid producing the wrong mix before the promotion's true demand is known.
A common decision is whether to build base product early and postpone final packaging.
That option only works if packaging capacity and materials are available when the mix is known.

A capital-equipment producer may work from a small number of large customer orders.
The limiting factor can be engineering completion or a long-lead purchased component rather than finished-goods inventory.
The planning discussion should then emphasize milestone confidence and order-specific capacity.
A high aggregate ATP number would be a poor substitute for an order-level feasibility check.

A food or pharmaceutical producer may face expiry and quality-release constraints.
A gross inventory quantity can hide lots that expire before the customer can use them.
The MPS must use the relevant usable shelf life and approved release status.
Moving production earlier may avoid a capacity overload while worsening remaining shelf life.

### 9.6 What to seek before adopting a practice

Ask whether a published result describes one site or an entire network.
Ask whether it measures delivery reliability, inventory, schedule adherence, or only system adoption.
Ask for the baseline, time window, product mix, and concurrent changes.
Ask whether a quoted improvement was independently audited.
Do not infer a causal benefit from a vendor case study that lacks a comparator.
Use the organization's own before-and-after plan versions and decision records to evaluate a change.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Forecast and orders double counted | Demand spikes as forecast converts to orders | Both full quantities enter the same demand population | Excess production and inventory | Define consumption keys and reconcile order conversion |
| Forecast already netted, then netted again | Demand falls when an order is entered | A second consumption pass removes real demand | Underproduction | Record source semantics and test an order lifecycle |
| Incorrect order status | Canceled orders continue to consume ATP | Status mapping treats inactive lines as firm | Lost sales or excess inventory | Reconcile open order counts and status codes |
| Partial shipment ignored | Remaining order equals original order | Shipment quantity is not deducted | ATP understated and demand overstated | Use remaining quantity and match shipment events |
| Duplicate planned and firm receipt | Supply jumps when an order is released | Conversion creates a second supply line | False PAB and promises | Link converted records and enforce one active supply |
| Blocked stock treated as eligible | Quality hold appears in ATP | Inventory status not filtered | Promise against unusable supply | Define eligibility and test status changes |
| Safety target silently treated as a firm order | PAB calculation has an unexplained extra deduction | Buffer is added as demand rather than a target | Inflated production requirements | Separate target, reserve, and demand fields |
| Negative ATP clipped to zero | Sum of displayed ATP exceeds uncommitted supply | Interval shortage is hidden instead of allocated | Double promising | Carry deficit backward per policy and report residual |
| Future receipt used for an earlier promise | Customer date precedes receipt date | Cumulative total ignores timing | Late shipment | Evaluate promise at the requested date |
| Supply interval ignores later horizon orders | Final ATP is unusually high | Orders after the modeled horizon are omitted | Overcommitment at the horizon edge | Extend the horizon or restrict the promise |
| Multiple users promise the same quantity | Concurrent confirmations exceed ATP | Read and reservation are not atomic | Double allocation | Use transaction-safe reservation and reconciliation |
| ATP ignores customer allocations | Some channels consume another channel's stock | Aggregate quantity is shown without quota | Contract or priority failure | Apply effective allocation policy at promise time |
| Buffer shown as unconditionally free | Sales promises the final reserve | ATP and inventory target have different policies | Higher risk of later stockout | State reserve authority and approval rule |
| Receipt placed in wrong bucket | PAB is healthy but daily orders miss | Receipt is assumed at week start when it arrives later | In-week shortage | Model timestamp or perform a daily check |
| Release offset ignores calendar | A planned release falls on a shutdown day | Calendar days replace working days | Unrealistic receipt date | Offset through the correct plant calendar |
| Past-due release silently moved to today | Repeated expedite messages | Fixed lead time is not recoverable | Chronic lateness and firefighting | Show lateness, owner, and realistic recovery date |
| Capacity rate is stale | Actual hours consistently exceed standard | Routing or setup assumptions are outdated | Published MPS overload | Compare standard against confirmed actuals and update master data |
| Setup counted twice | Capacity load exceeds actual practice | Rate includes setup and setup hours are added again | Unnecessary overtime or rejected demand | Document rate basis and reconcile the load calculation |
| Setup omitted | Several small lots fit only on paper | Per-unit rate ignores changeovers | Missed schedule and overtime | Include order-level setup or use a conservative RCCP factor |
| Yield applied inconsistently | Good-unit receipt exceeds likely output | Started units and good units are mixed | Material and capacity shortfall | Convert starts and good units using a stated yield |
| Freeze bypassed silently | Near-term dates change without approval | Users have ungoverned override rights | Churn and downstream mistrust | Audit edits and require a named exception approver |
| Freeze is absolute in practice | Customer impact grows despite available recovery | No emergency path or decision authority exists | Avoidable missed commitments | Define a controlled override and escalation clock |
| Plan compared at moving horizons | Instability rises every week without clear changes | Different calendar weeks are compared | Misleading KPI and misplaced action | Compare identical item-location-calendar cohorts |
| Aggregate capacity hides mix | Family plan fits but a specific line is overloaded | Products use different bottleneck resources | Infeasible item-level schedule | Check resource-specific item loads |
| Material and capacity checks disconnected | Capacity passes while components are late | A schedule is approved without MRP feedback | Idle capacity and delayed orders | Peg shortages to MPS and close both checks |
| Customer communication omitted | ERP date changes but customer expects the old date | Planning change is not sent to service | Surprise late delivery | Assign communication owner and confirm acknowledgement |
| Duplicate exceptions proliferate | Many alerts have the same item, date, and cause | Each run creates a new alert without state | Alert fatigue | Use stable IDs, aging, and closure evidence |
| Wrong version used | Different teams report different MPS quantities | Extracts have different cutoffs or scenario IDs | Conflicting decisions | Publish a versioned baseline and common snapshot |
| False precision beyond horizon | Far-future ATP is treated as a confirmed production slot | Capacity or supplier confirmation is absent | Unreliable long-range promise | Apply confidence gates and promise horizon limits |
| Data fixes lack ownership | The same invalid route returns after every run | Analyst patches an extract but source remains wrong | Repeated manual work | Assign source-system steward and validate correction |

A red flag is a prompt for investigation, not proof of a single cause.
Several failures can coexist, such as an incorrect calendar and a stale capacity rate.
An exception log should identify the verified mechanism and the evidence used.

## 11. Assumptions and limits

| Assumption | Why the example needs it | What breaks it | How to test or relax it |
|---|---|---|---|
| One item and one plant | Keeps the calculations visible | Shared stock, alternate plants, or transfers | Add location-specific supply and transit time |
| Opening 140 kits are eligible | Establishes the starting balance and ATP | Holds, ownership restrictions, or unrecorded allocations | Reconcile stock status and reservations at the cutoff |
| Orders and forecast cover the same population | Makes weekly `max` a meaningful consumption rule | Special orders or forecasts already net of bookings | Trace a sample of orders to forecast segments |
| The order row is firm and unshipped | Makes booked demand valid for ATP | Cancellations, partial shipments, or tentative orders | Compare remaining quantities to order and shipment status |
| Receipts are usable before weekly demand | Enables bucket-level PAB and ATP | Midweek completion after customer due dates | Recalculate by day or event timestamp |
| Forecast and order buckets align | Allows direct weekly comparison | Different cutoffs, timezones, or delivery-date basis | Normalize calendars and compare underlying dates |
| No external allocations exist | Allows all 140 opening kits into the example ATP | Contractual reserves or another channel's claim | Reconcile allocation ledger and subtract once |
| No later orders are modeled | Bounds the final ATP interval | Orders exist after week 6 | Extend the horizon and state the next receipt boundary |
| Supply quantities are nonduplicated | Keeps conservation checks valid | Planned orders convert without removing original records | Trace order conversion identifiers |
| One-week production lead time | Makes release-week capacity simple | Queue, inspection, or supplier variability | Examine historical release-to-availability distributions |
| Resource rate is 0.4 hours per good kit | Converts output into load | Different mix, actual yield, or omitted setup | Compare standard and actual hours by route and lot |
| Resource hours are net staffed hours | Makes load-capacity comparisons commensurate | Maintenance or staffing losses are already or not yet included | Reconcile shift calendar, exclusions, and overtime |
| A 20-kit split is allowed | Makes Option B technically possible | Minimum lot, cleaning, or qualification rules | Confirm routing, lot policy, and quality constraints |
| Materials exist for an earlier release | Supports the pull-forward decision | A component arrives after week 2 | Run MRP and obtain supplier or store confirmation |
| Carrying rate is USD 0.10 per kit-week | Gives an illustrative comparison | Different capital, storage, or expiry costs | Use finance-approved incremental cost and time |
| Orders are not reprioritized | Preserves the ATP allocation trace | A customer priority or channel quota changes | Re-run allocation with the approved priority policy |
| Weekly netting represents demand adequately | Keeps an auditable six-row example | Within-week demand spikes or substitution | Compare daily history and scenario-test the peak |
| Capacity check includes the relevant constraint | Makes the demonstrated overload actionable | Another resource or labor skill also constrains output | Review routing, cross-resource load, and finite sequence |
| Supply feasibility is not guaranteed by ATP | Prevents overstating a formula's authority | Users treat a planned receipt as certain | Gate promises by status, risk, and confirmation |
| Time fences are locally configured | Clarifies who may change the plan | Product-specific system semantics differ | Verify settings, permissions, and override logs |

The example is deterministic.
It does not estimate service probability, optimize lot sizes, or model demand correlation.
It does not calculate a full component explosion or an operation-level sequence.
The 25-kit buffer is an illustrative policy, not a statistically derived safety stock.
The numbers are useful for learning the control logic, not for setting a real plant's thresholds.

## 12. Exercises

Use units and dates in every numerical answer.
State a convention whenever the exercise leaves a policy choice open.

### Exercise 1 — Classify the signals

**Type:** recall and explanation.  
**Time:** 10 minutes.  
**Deliverable:** a four-row table and one sentence explaining overlap.

A planner receives a forecast of 95 kits for week 4.
Customers have placed 62 kits of firm orders for that same forecast population and week.
The proposed week 4 production receipt is 80 kits.
Management sets an end-of-week inventory target of 20 kits.

Identify which value is a forecast, a commitment, a plan, and a target.
Calculate the weekly planning demand under the `max` rule.
Explain why 95 plus 62 is not the correct demand under that rule.

**Acceptance criteria:** all four categories are correct; the demand is in kits; overlap is explained without treating the target as demand.

### Exercise 2 — Roll forward PAB

**Type:** calculation.  
**Time:** 15 minutes.  
**Deliverable:** a three-row table with demand, receipt, end PAB, and buffer status.

Beginning eligible inventory is 84 kits.
Forecast is 43, 56, and 49 kits in weeks 1, 2, and 3.
Firm orders are 39, 61, and 30 kits in those weeks and cover the same population.
Receipts are 0, 50, and 40 kits respectively.
The target is 20 kits at each week end.
Use weekly `max` demand and assume receipts precede demand within a week.

**Acceptance criteria:** every end balance is shown; the first target breach is identified; the final stock conservation equation is stated.

### Exercise 3 — Allocate a negative ATP interval

**Type:** calculation and critique.  
**Time:** 20 minutes.  
**Deliverable:** raw and adjusted interval ATP, transfer trace, and one promise limitation.

Opening eligible inventory is 90 kits.
MPS receipts are 70 kits in week 2 and 80 kits in week 4.
Firm orders are 40, 50, 45, and 30 kits in weeks 1 through 4.
There are no additional receipts or orders through the horizon end in week 4.
Use discrete intervals and reallocate a negative interval against earlier positive ATP, most recent first.

**Acceptance criteria:** interval boundaries are correct; any deficit is transferred explicitly; total adjusted ATP reconciles to eligible supply less firm orders.
Your limitation must address either timing, buffer policy, or feasibility.

### Exercise 4 — Design time fences

**Type:** application and design.  
**Time:** 20 minutes.  
**Deliverable:** a short policy and authority matrix.

A plant has a two-week internal production lead time.
A key component requires six weeks from supplier commitment to arrival.
A customer can normally revise an order until three weeks before delivery.
The factory incurs substantial cleaning cost if a released campaign changes.
The plant has no single universally correct PTF or DTF.

Propose frozen, slushy, and liquid zones and a demand-consumption policy.
Name at least three actions requiring approval and their owners.
Explain how the component lead time influences the longer planning horizon without automatically becoming an absolute freeze.

**Acceptance criteria:** dates and authority are explicit; supply and demand fences are distinguished; a controlled emergency override is included.

### Exercise 5 — Compare capacity options

**Type:** calculation and decision.  
**Time:** 20 minutes.  
**Deliverable:** a load calculation and an incremental-cost comparison with a conditional recommendation.

A receipt of 150 kits requires 0.5 staffed hours per kit in the preceding week.
The resource has 65 staffed hours available.
Overtime costs an additional USD 90 per staffed hour.
Alternatively, 20 kits can be produced one week earlier using otherwise unused capacity.
The early batch requires an additional USD 140 setup and USD 0.15 per kit-week in carrying cost for one week.
Assume both options satisfy quality, material, and due-date constraints.

**Acceptance criteria:** load and overload are in hours; both incremental costs are in USD; the recommendation identifies the assumptions that could reverse it.

### Exercise 6 — Govern an exception

**Type:** workflow design.  
**Time:** 20 minutes.  
**Deliverable:** a one-page exception record and escalation sequence.

An approved week 2 receipt of 120 kits is at risk because a component supplier reports a two-day delay.
A customer has a confirmed order due at the end of week 2.
The planning snapshot is two days old and the material status is not yet confirmed.
The release date is inside the frozen zone.

Specify severity, response deadline, evidence needed, owner, options, approval, and customer communication.
Include a closure condition and a rule for an unresolved decision.

**Acceptance criteria:** the record separates facts from assumptions; names roles rather than a generic team; links the supply decision to ATP and customer communication.

### Exercise 7 — Audit a planning extract

**Type:** critique.  
**Time:** 15 minutes.  
**Deliverable:** a defect-and-test table with at least five rows.

The extract contains a forecast already net of orders, but the analyst applies weekly `max` with the gross orders.
An old planned order and its released replacement both appear as receipts.
Quality-held inventory appears as eligible opening stock.
Capacity is in machine hours while the routing rate is in staffed hours.
The planning and order data were extracted on different days.

For each defect state the likely direction or ambiguity of bias and a concrete test.
Add one governance control that prevents recurrence.

**Acceptance criteria:** all five supplied defects are addressed; uncertainty is acknowledged where direction depends on configuration; each test has a source or reconciliation target.

### Exercise 8 — Replan after an order change

**Type:** integrated application.  
**Time:** 25 minutes.  
**Deliverable:** revised demand and PAB for the affected weeks, an ATP interpretation, and a decision note.

Return to the six-week worked example and its original receipts.
Before any shipments, week 5 firm orders increase from 40 to 82 kits.
The increase is for the same demand population as the week 5 forecast.
All other inputs stay unchanged.
Use the original weekly `max` demand and gross-of-buffer ATP convention.

Recalculate week 5 and week 6 PAB.
Calculate the raw week 4 interval ATP for weeks 4 and 5.
Using the original negative-ATP reallocation policy, calculate the adjusted discrete ATP intervals.
State whether the customer request should be automatically confirmed and why.

**Acceptance criteria:** the change in PAB is reconciled; the raw and adjusted interval effects are distinct; capacity, timing, allocations, and buffer policy are considered before confirmation.

## 13. Self-check answers

### Answer 1 — Signal classification

The 95-kit value is the week 4 demand forecast.
The 62-kit value is the firm customer commitment.
The 80-kit value is the proposed production receipt plan.
The 20-kit value is an end-of-week inventory target.
Under weekly `max`, planning demand is `max(95, 62) = 95` kits.
Adding 95 and 62 would count the same customer population twice under the stated overlap assumption.
If the 62 kits were truly incremental to the forecast, the answer would differ.

### Answer 2 — PAB

Week 1 demand is `max(43, 39) = 43` kits.
Week 1 PAB is `84 + 0 - 43 = 41` kits, above the 20-kit target.
Week 2 demand is `max(56, 61) = 61` kits.
Week 2 PAB is `41 + 50 - 61 = 30` kits, above the target.
Week 3 demand is `max(49, 30) = 49` kits.
Week 3 PAB is `30 + 40 - 49 = 21` kits, above the target by 1 kit.
There is no target breach in the three weeks.
The conservation check is `84 + 50 + 40 - 43 - 61 - 49 = 21` kits.
A daily shortage is still possible if the weekly timing assumption is false.

### Answer 3 — ATP

Week 1 is the first interval and has `90 - 40 = 50` kits of raw ATP.
Weeks 2 and 3 form the second interval and have `70 - 50 - 45 = -25` kits of raw ATP.
Week 4 is the final interval and has `80 - 30 = 50` kits of raw ATP.
Transfer 25 kits from week 1 to cover the week 2 interval deficit.
The adjusted interval values are 25, 0, and 50 kits.
Their total is 75 kits.
The conservation check is `90 + 70 + 80 - 40 - 50 - 45 - 30 = 75` kits.
The week 4 receipt cannot support a promise due in week 3.
An inventory reserve or unverified receipt could further restrict the quantity that may actually be promised.

### Answer 4 — Time-fence rubric

A strong answer specifies the exact date convention and how the rolling boundaries advance.
It distinguishes the six-week component commitment from the two-week internal production lead time.
One defensible policy freezes released work for two weeks, reviews weeks 3 through 6 under controlled change, and replans later periods routinely.
That is a proposal, not a universal best practice.
The component commitment may need earlier purchasing or a supplier flexibility agreement even when the finished-goods schedule is not frozen for six weeks.
A strong answer states whether the DTF changes the demand calculation or only triggers review.
It names a planner, an operations approver, a materials owner, and a customer communication owner.
It identifies an emergency override and records the cost and customer impact.

### Answer 5 — Capacity options

The load is `150 × 0.5 = 75` staffed hours.
The overload is `75 - 65 = 10` staffed hours.
Overtime adds `10 × USD 90 = USD 900`.
Producing 20 kits earlier removes `20 × 0.5 = 10` hours from the overloaded week.
The early-batch cost is `USD 140 + 20 × USD 0.15 = USD 143`.
Under the stated assumptions the early batch saves `USD 900 - USD 143 = USD 757` compared with overtime.
An unmodeled expedite, material shortage, expiry loss, or extra setup above that saving could reverse the recommendation.

### Answer 6 — Exception rubric

A strong record treats the supplier's reported delay as a risk, not as a confirmed missed customer delivery.
It obtains a current component status, recovery date, remaining order quantity, and actual stock before promising a response.
The materials planner obtains supplier confirmation and the plant checks feasible recovery capacity.
The master scheduler compares a split, alternate material if approved, overtime, and a revised receipt.
A delegated manager approves any frozen-zone change.
Customer service assesses the affected order and communicates a revised confirmed date when authorized.
The deadline precedes the last feasible recovery action, not merely the weekly meeting.
Closure requires recorded evidence, updated plan and ATP, and customer acknowledgement when a commitment changes.
An unresolved decision escalates to a named manager before the response deadline.

### Answer 7 — Extract audit rubric

A second consumption pass can understate or distort demand; compare the forecast definition and an order-level consumption trace.
A duplicate planned and released receipt overstates supply; join the conversion identifiers and reconcile active quantities.
Held stock in opening availability overstates eligible supply; reconcile quantity by quality status to the inventory ledger.
Machine hours and staffed hours are not directly interchangeable; reconcile the capacity unit to the routing and staffing model.
Different extract dates can bias results either way; align as-of timestamps and explain intervening transactions.
A durable control assigns a source owner, validation rule, exception response, and recurring reconciliation.
Patching only the analyst's copy is not a durable source correction.

### Answer 8 — Changed order

Week 5 netted demand becomes `max(70, 82) = 82` kits, an increase of 12 kits.
Week 5 PAB becomes `99 - 82 = 17` kits.
Week 6 PAB becomes `17 + 120 - 65 = 72` kits.
The week 5 balance is 8 kits below the 25-kit target, although it remains positive.

The raw ATP for the week 4 interval becomes `140 - 70 - 82 = -12` kits.
The raw ATP sequence is 79, -18, -12, and 90 kits.
First cover the week 2 deficit of 18 kits from week 1, leaving 61 kits there.
Then cover the week 4 deficit of 12 kits from the remaining week 1 ATP, leaving 49 kits there.
The adjusted ATP sequence is 49, 0, 0, and 90 kits, totaling 139 kits.
The conservation check is `140 + 380 - (339 + 42) = 139` kits.
The order quantity increased by 42 kits even though PAB fell by only 12 kits, because 30 kits consume previously unbooked forecast.

Do not automatically confirm solely from these static values.
Check whether the earlier 49-kit ATP is still unreserved and physically available for week 5.
Check the week 4 receipt's capacity and material feasibility, the allocation policy, and the buffer exception.
A customer commitment also needs the correct daily ship and arrival timing.

## 14. Interview questions

### 1. What does an MPS decide that a demand forecast does not?

A strong answer distinguishes a dated supply proposal from an estimate of market demand.
It includes item, location, quantity, timing, and an owner.
It explains that the MPS must be checked against material and capacity.
A weak answer says the two are the same number or that the forecast guarantees production.

### 2. Why can `max(forecast, orders)` be wrong?

A strong answer discusses overlapping demand populations, prior consumption, and cross-bucket windows.
It gives an incremental order or a pre-netted forecast as a counterexample.
A weak answer treats the formula as an ERP-independent law.

### 3. How does PAB differ from ATP?

A strong answer identifies the demand basis and the purpose of each measure.
It notes that ATP depends on the selected allocation and receipt policy.
A weak answer assumes a positive inventory projection is automatically a customer promise.

### 4. What does a negative discrete ATP interval tell you?

A strong answer says interval orders exceed the supply assigned to that interval.
It describes a permitted earlier-supply allocation and the need to expose any residual shortage.
It also states that future supply cannot satisfy an earlier promise without a changed date.
A weak answer simply sets the negative value to zero and loses the deficit.

### 5. How would you set a planning time fence?

A strong answer considers actual lead times, release status, costs of change, and delegated authority.
It describes a measured review and a controlled exception path.
A weak answer selects a fixed number of weeks solely because another company uses it.

### 6. Why can a plan with positive PAB still fail?

A strong answer mentions within-week timing, material availability, and bottleneck load.
It distinguishes a rough-cut check from a finite schedule.
A weak answer assumes a positive end-of-week total proves all orders can ship on time.

### 7. How do you compare overtime with pulling production forward?

A strong answer uses incremental costs and checks extra setup, holding, quality, and material feasibility.
It recognizes the customer-service impact and records assumptions.
A weak answer compares only the labor rate or only the inventory balance.

### 8. What should happen when a frozen order must change?

A strong answer names the authority, records the reason and alternatives, and propagates the change.
It includes material, capacity, ATP, and customer communication effects.
A weak answer either silently changes the order or claims the freeze makes all change impossible.

### 9. How would you measure MPS nervousness?

A strong answer compares aligned calendar cohorts across versioned snapshots.
It states the denominator and distinguishes bucket movement from physical units.
It reviews stability alongside service and cost.
A weak answer compares different rolling windows or rewards stability regardless of missed orders.

### 10. How do you know whether a customer can be promised the displayed ATP?

A strong answer checks transaction-safe reservation, date, eligibility, allocation, and supply confidence.
It includes any approval required to use a buffer.
A weak answer copies the aggregate ATP value into an order without checking other claims.

### 11. What would you ask after an ERP upgrade changes ATP?

A strong answer asks about configuration, data definitions, order statuses, consumption, and interval logic.
It requests reconciled before-and-after test cases for the same snapshot.
A weak answer assumes the new value is correct because the system generated it.

### 12. How do you close a capacity exception?

A strong answer requires an approved option, confirmed resource and material availability, and updated dependent plans.
It also checks affected customer commitments and retains evidence of communication.
A weak answer closes the alert when someone writes "investigating" or "expedite" in a comment.

## 15. Key terms

Each term points to the detailed definition in [Section 4](#4-definitions-and-vocabulary).

- [Allocation](#4-definitions-and-vocabulary): an entitlement or reservation against eligible supply.
- [Available to promise (ATP)](#4-definitions-and-vocabulary): uncommitted supply under a stated promise policy.
- [Backorder](#4-definitions-and-vocabulary): an accepted requirement that remains unfulfilled after its required date.
- [Baseline](#4-definitions-and-vocabulary): a published plan version used for comparison.
- [Bottleneck](#4-definitions-and-vocabulary): a resource constraining throughput in the modeled circumstances.
- [Commitment](#4-definitions-and-vocabulary): a promised quantity and date.
- [Cumulative ATP](#4-definitions-and-vocabulary): promise capacity accumulated through a requested date under the chosen rules.
- [Cumulative lead time](#4-definitions-and-vocabulary): the longest relevant lead-time chain through the supply structure.
- [Customer order](#4-definitions-and-vocabulary): a customer requirement recorded with quantity, date, and status.
- [Demand forecast](#4-definitions-and-vocabulary): an estimate of demand not yet fully known.
- [Demand time fence](#4-definitions-and-vocabulary): a boundary associated with the local demand or consumption policy.
- [Dependent demand](#4-definitions-and-vocabulary): a requirement derived from a parent or other modeled relationship.
- [Discrete ATP](#4-definitions-and-vocabulary): uncommitted supply calculated in receipt-defined intervals.
- [Exception](#4-definitions-and-vocabulary): a condition requiring action under a defined rule.
- [Firm planned order](#4-definitions-and-vocabulary): planned supply protected from specified automatic changes.
- [Forecast consumption](#4-definitions-and-vocabulary): reduction of overlapping forecast when eligible orders arrive.
- [Frozen zone](#4-definitions-and-vocabulary): an interval where schedule changes require explicit authority.
- [Independent demand](#4-definitions-and-vocabulary): demand not generated by a parent-component explosion.
- [Lead time](#4-definitions-and-vocabulary): assumed time from release to usable receipt.
- [Liquid zone](#4-definitions-and-vocabulary): an interval permitting ordinary replanning.
- [Load](#4-definitions-and-vocabulary): resource time required by planned work.
- [Lot size](#4-definitions-and-vocabulary): the quantity rule for a planned or released order.
- [Master production schedule (MPS)](#4-definitions-and-vocabulary): dated planned supply for selected items.
- [Pegging](#4-definitions-and-vocabulary): the trace connecting a requirement or supply to its source.
- [Planned receipt](#4-definitions-and-vocabulary): supply expected to become usable in a period.
- [Planning time fence](#4-definitions-and-vocabulary): a boundary controlling automatic supply replanning.
- [Projected available balance (PAB)](#4-definitions-and-vocabulary): time-phased inventory after selected demand and supply.
- [Released production order](#4-definitions-and-vocabulary): an order authorized for execution.
- [Rough-cut capacity planning (RCCP)](#4-definitions-and-vocabulary): approximate resource-load testing of a proposed MPS.
- [Safety stock](#4-definitions-and-vocabulary): an inventory target or reserve against uncertainty.
- [Schedule nervousness](#4-definitions-and-vocabulary): changes to plan quantities or dates across comparable versions.
- [Scheduled receipt](#4-definitions-and-vocabulary): an existing expected incoming supply order.
- [Slushy zone](#4-definitions-and-vocabulary): an interval in which changes require limited review.

## 16. Sources

Sources are linked so a reader can inspect the original context.
Vendor documentation establishes product behavior or terminology, not independent evidence of customer outcomes.
Accessed for course reference as publicly available links; verify the current edition before applying a rule to an implementation.

1. Association for Supply Chain Management (ASCM), [Supply Chain Dictionary](https://www.ascm.org/topics/dictionary/).
   A terminology reference for planning concepts; local system definitions can differ.
2. Oracle, [Available to Promise (ATP)](https://docs.oracle.com/cd/A60725_05/html/comnls/us/mrp/atp.htm).
   Official product documentation on master scheduling and ATP concepts; version-specific configuration matters.
3. SAP, [Product Allocation in Advanced Available-to-Promise](https://help.sap.com/docs/SAP_S4HANA_ON-PREMISE/f132c385e0234fe68ae9ff35b2da178c/cfc8aa5747c46c10e10000000a441470.html).
   Official vendor documentation on allocation capability; not a guarantee of a particular customer's process.
4. International Society of Automation, [ISA-95 Standard: Enterprise-Control System Integration](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard).
   Standards-family overview for interfaces between enterprise and manufacturing systems.
5. GS1, [Global Trade Item Number (GTIN)](https://www.gs1.org/standards/id-keys/gtin).
   Official identifier reference; a GTIN does not by itself encode plant-specific production rules.
6. U.S. National Archives and Records Administration, [21 CFR Part 211 — Current Good Manufacturing Practice for Finished Pharmaceuticals](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-C/part-211).
   Primary regulatory text; consult the applicable current version and jurisdiction.
7. Toyota Motor Corporation, [Toyota Production System](https://global.toyota/en/company/vision-and-philosophy/production-system/).
   Company description of its production philosophy; not evidence of a particular ATP formula or time-fence length.
8. Orlicky, Joseph, *Material Requirements Planning: The New Way of Life in Production and Inventory Management*, McGraw-Hill, 1975.
   Foundational book on time-phased material planning; the exact implementation must be checked against current system behavior.
9. Vollmann, Thomas E., William L. Berry, D. Clay Whybark, and F. Robert Jacobs, *Manufacturing Planning and Control for Supply Chain Management*, 5th edition, McGraw-Hill/Irwin, 2005.
   Reference on the planning hierarchy and master scheduling; use as conceptual background rather than a regulatory source.

## 17. Links

- [Module 15 index](./_index.md): topic sequence and module context.
- [The planning cascade and S&OP](./01_planning-cascade.md): decisions and handoffs surrounding master scheduling.
- [Manufacturing master data](./02_manufacturing-master-data.md): items, bills, routing, calendars, and effective dates.
- [Module 15 survey](../../../course/15_manufacturing_production_planning.md): the complete planning cascade and broader production context.
- [Production labs survey](../../../course/30_simulation_optimization_production_labs.md): related simulation, optimization, and production exercises.
- [Production labs module index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md): lab module navigation.
