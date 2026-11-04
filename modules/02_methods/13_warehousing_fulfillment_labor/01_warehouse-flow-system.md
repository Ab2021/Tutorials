# The Warehouse as a Flow System

> Module 13 — Warehousing, Fulfillment, and Labor Analytics  
> Topic 01 of 10  
> Estimated study time: 4 hours  
> Prerequisites: basic units, inventory and transportation concepts  
> Version 2026.5

## 1. Why this topic exists

A warehouse manager does not sell movement; the customer buys an accurate order available when promised.
The warehouse must convert inbound goods into usable inventory and then into loaded shipments without losing identity, quality, or safety.
Every local improvement must be judged by its effect on that end-to-end promise and its cost.
A picker can be busy all shift while the shipping dock misses its carrier departure.
A receiving team can unload quickly while unverified pallets block put-away and prevent orders from being fulfilled.
A manager who sees only departmental productivity may reward both outcomes.

The business decision is not simply how many workers to schedule.
It is where to place the next qualified worker, whether to change a release or cutoff, and when to buy capacity.
The answer depends on the constraint at the relevant time, the mix of work, the buffers, and what downstream stations can actually consume.
Capacity that cannot reach a customer because another operation is constrained is not automatically economic capacity.

This lesson builds a physical and financial flow map for a distribution center (DC).
It follows receiving, put-away, storage, replenishment, picking, packing, staging and shipping.
It treats returns, holds, shortages, and rework as real paths rather than disappearing exceptions.
It distinguishes material flow from information flow, and a plan from an actual customer commitment.
The numerical case is illustrative, not a claim about a named facility.

A supervisor can use the model at a shift huddle; a finance partner can use it in a capacity investment review.
An analyst can use the same map to determine which timestamps and quantities are needed before publishing a bottleneck report.
Safety and product integrity are constraints on all alternatives, not optional adjustments after the productivity calculation.

### The decision changes over the operating day

At 07:00 the receiving manager may decide which supplier vehicles need scarce dock doors.
At 09:00 the outbound supervisor may decide whether to release a new wave of orders.
At 11:00 inventory control may need to resolve stock that exists physically but is not eligible in the system.
At 13:00 transportation may need to secure an additional collection or notify customers of a likely miss.
These decisions use related data, but they do not have the same deadline or the same available actions.
The correct forecast at 07:00 cannot replace an updated estimate at 13:00.

The cost of waiting also changes with time.
An additional packer available for four hours before cutoff can prevent a miss.
The same person starting after the carrier leaves can reduce backlog but cannot reverse a missed handoff.
Similarly, an inbound pallet received after its customer order's release may help tomorrow even if it cannot help today.
Time, eligibility and custody therefore belong in the economic case alongside productive rates.

The most expensive error is often a false explanation of a miss.
If the dashboard shows a high pick rate, a manager might conclude that packing needs a faster printer.
But the actual problem could be incomplete orders waiting for replenishment, a late label interface, or a full staging lane.
Each remedy has a different owner, cash cost, implementation time and risk.
This lesson supplies a way to separate those mechanisms before committing labor or capital.

### Who needs the answer

The warehouse supervisor needs an achievable plan for the next interval, not just the previous day's average.
The inventory controller needs to know which stock records prevent allocation and what evidence can release them.
The transportation planner needs a credible ready-to-load forecast before carrier options expire.
Customer service needs a list of affected orders and revised promises rather than an unqualified throughput percentage.
Finance needs the difference between an avoidable cash cost and a transfer of work between cost centers.
The safety owner needs a veto over a plan that exceeds safe occupancy or requires unauthorized equipment use.

These stakeholders may see different counts because they observe different events.
The supervisor may count picked cartons; customer service may count complete orders.
Transportation may count loaded trailers; finance may count invoiced shipments.
None of these counts is intrinsically wrong, but they cannot be substituted for one another without a bridge.
The bridge must explain partial orders, cancellations, multi-carton orders and late carrier proof.
The lesson uses good customer-order handoff as the final service boundary and names other boundaries explicitly.

### What success does not mean

A successful warehouse is not necessarily the one with the highest observed utilization.
It is a warehouse that can satisfy an agreed service at an acceptable total economic cost and risk.
The best action may leave some upstream labor idle to avoid flooding a constrained downstream area.
It may also add a controlled buffer before a constrained process so that normal variability does not starve it.
Both choices require a reasoned decision, not a blanket rule that all idle time or all WIP is bad.

The analysis here concerns flow through one building and its immediate carrier handoff.
It does not optimize network location, carrier routing, or individual employee performance.
Those decisions affect the building, but their objectives and constraints must be evaluated separately.
When the site cannot meet a commitment, an honest exception report is preferable to relabeling a late shipment as on time.

## 2. Learning outcomes

After completing this lesson, you can:

1. Draw the physical and information flows and identify every handoff and major buffer; see exercise 1.
2. Reconcile opening stock, receipts, shipments, and adjustments without double counting; see exercise 2.
3. Convert station capacities expressed in different units to a common demand basis; see exercise 3.
4. Estimate a bottleneck and remaining backlog for a defined shift and product mix; see exercise 4.
5. Compare a staffing intervention with its incremental contribution and service risk; see exercise 5.
6. Design a release, escalation, and recovery rule with named owners; see exercise 6.
7. Critique a flow metric using its event boundaries, population and quality exclusions; see exercise 7.
8. Present an investment recommendation with sensitivity and a falsifiable pilot; see exercise 8.

These are operational decisions rather than exercises in maximizing utilization.
State the time window and the unit for every answer.
Show what is assumed rather than presenting an estimate as a measured fact.

## 3. The core mental model

```text
supplier appointment + advance shipment notice
             |                               information
             v
arrival -> unload -> inspect -> receive -> put-away -> reserve storage
              |          |                     |             |
              v          v                     v             v
         dock queue    hold             put-away queue   replenishment
                                                               |
customer order -> promise -> release -> pick face <- replenishment
                                        |
                                        v
                                   pick -> pack -> stage -> load -> depart
                                     |       |        |       |
                                     v       v        v       v
                                  short    rework    queue   carrier proof

customer return -> identify -> inspect -> restock / repair / dispose
```

Arrows indicate material or decision dependencies, not a promise that all facilities use the same sequence.
The order management system determines what is promised; a warehouse management system (WMS) directs and records work.
Some cross-dock receipts move directly to outbound staging; they must still pass the relevant identification and control checks.
Reserve storage is not immediately pickable unless location, status, allocation and replenishment permit it.
A physical pallet and a database record can disagree; neither should silently overwrite the other.

Each station has an input queue, an effective processing rate, a downstream handoff, and failure or rework paths.
A queue protects the next station from short supply interruptions but consumes space and increases elapsed time.
Too little work at a constrained station causes starvation; too much work upstream creates congestion and hides problems.
A controlled buffer has an owner, a unit, a maximum, an age threshold, and an escalation path.

Throughput is completed good output per elapsed time at a specified boundary.
Capacity is the sustainable rate under specified staffing, mix, equipment availability and quality constraints.
The smallest *comparable* effective capacity can constrain flow, but the constraint can move by hour or product family.
Release rate is a decision; shipments are an observed result; forecast demand is uncertain; a customer commitment is contractual or policy-defined.
An average rate does not establish that an individual order will make a specific carrier cutoff.

The conservation identity is opening work plus arrivals minus completed departures plus net corrections equals closing work.
Apply it separately to each buffer and use consistent units and event boundaries.
For a stable system over a sufficiently representative period, Little's Law relates average work in process (WIP), throughput and flow time: L = λW.
It does not justify dividing a single peak-hour snapshot by a day-average rate when arrivals and departures are unbalanced.
A buffer is not automatically waste: it can provide recovery time, but its capacity and service impact must be explicit.

### Four clocks for one order

An order has a customer clock, an inventory clock, a warehouse-work clock and a carrier clock.
The customer clock starts at the agreed order event and ends at the promised delivery or shipment event.
The inventory clock tracks when all required stock becomes both physically accessible and legally eligible.
The work clock tracks release, picking, packing and readiness for dispatch.
The carrier clock tracks cutoff, vehicle availability and the handoff evidence accepted by the service agreement.
These clocks overlap, but one clock cannot be inferred reliably from another.

For example, an order released at 08:00 may be picked at 10:00 and packed at 10:15.
If the last carton waits at the dock until a 15:30 carrier acceptance, warehouse processing and customer wait are different intervals.
The 10:15 pack scan is useful for pack productivity, but it is not evidence of the 15:30 acceptance.
If a required line was not eligible until 09:30, the 90-minute wait before that event also needs a reason.
A complete analysis reports the longest material waits and the boundary at which each began.

### Flow efficiency and the cost of waiting

Suppose a complete order spends 18 minutes in active work and 162 minutes waiting between release and carrier handoff.
Its elapsed time is 180 minutes and its active-time fraction is 18 / 180 = 10%.
This fraction describes the order's time distribution; it is not the employee utilization rate.
Reducing one 30-minute queue by 20 minutes improves elapsed time if the downstream path can absorb the order.
It does not necessarily save 20 labor minutes, because waiting cartons do not continuously consume a worker.
The economic value may instead be fewer missed departures, less occupied space or a later feasible order cutoff.

Conversely, removing all waiting can make a constrained station repeatedly run out of work.
A buffer before that station is an insurance decision against variability in its upstream supply.
The manager pays for the insurance in occupied space, handling and longer flow time.
The appropriate amount changes with replenishment reliability, service priority and the cost of a missed cutoff.
Measure those trade-offs before choosing a fixed number of cartons.

### Conservation is not the same as productivity

Imagine a pack queue starts with 40 cartons, receives 160 and completes 150 in one hour.
Its closing balance is 40 + 160 - 150 = 50 cartons.
The 150 completed cartons describe the station's gross output at that boundary.
If 5 cartons need correction and remain in the queue, define whether the completion event reverses or the rework returns as an entry.
Use one convention consistently so that the 5 cartons do not disappear or appear twice.
Neither the closing balance nor gross output alone reveals how many complete orders reached a carrier.

A second queue can grow at the same time the first queue shrinks.
If packing produces faster than shipping can load, the pack queue improves while staging deteriorates.
Viewing both buffers prevents an apparent local gain from being mistaken for a system gain.
The same logic applies upstream when unloaded pallets accumulate in a put-away queue.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Receipt | Verified inbound quantity, units or pallets | Equating a truck arrival with available stock |
| Advance shipment notice (ASN) | Supplier's expected shipment detail | Treating an unverified notice as a count |
| Put-away | Movement from receipt to an authorized location | Assuming a suggested destination proves completion |
| Reserve | Bulk storage location and its stock | Assuming it is already at the pick face |
| Pick face | Location supplying picking work | Ignoring its replenishment needs |
| Replenishment | Transfer from reserve to pick face | Treating movement as new inventory |
| Order | Customer request at header grain | Assuming one order always equals one carton |
| Order line | Requested item on an order | Treating all lines as equal effort |
| Unit | An individual item in a specified unit of measure | Mixing cases and eaches |
| Carton | Shipping package | Treating every carton as a complete order |
| Throughput | Good output per time at a boundary | Counting starts as completions |
| Capacity | Sustainable potential completions per time under conditions | Calling a theoretical machine rate actual capacity |
| Utilization | Used capacity divided by compatible available capacity | Comparing different time windows |
| Bottleneck | Resource constraining system output in the selected period | Naming the busiest person without proving flow impact |
| WIP | Work inside defined system boundaries | Including completed orders or excluding holds inconsistently |
| Buffer | Work or space held between stages | Assuming a buffer has no cost or limit |
| Cycle time | Elapsed time between defined events | Confusing labor time with elapsed time |
| Lead time | Elapsed time from requested start to required outcome | Omitting release or carrier wait |
| Queue age | Time since work entered a waiting state | Using average age to hide an old tail |
| Cutoff | Latest event time accepted for a service | Assuming order release time is carrier acceptance |
| Allocation | Reservation of eligible stock for demand | Double counting allocated and available stock |
| Hold | Stock or work blocked pending a disposition | Deleting held units from the ledger |
| Exception | A departure from normal flow requiring a decision | Treating all exceptions as completed work |
| Rework | Additional work correcting an output | Counting corrected work twice as new throughput |
| Yield | Good output divided by input at defined stage | Using gross output as shippable output |
| Cross-dock | Transfer from inbound to outbound with limited storage | Skipping receiving controls |
| Dock-to-stock | Time between specified receipt and usable-stock events | Comparing sites with different endpoints |
| On-time shipment | Shipment meeting a specified commitment and proof event | Measuring label creation instead of carrier handoff |
| Cost-to-serve | Attributable cost of fulfilling a defined demand | Dividing total cost by incompatible volume |
| Contribution | Incremental revenue less incremental variable cost | Calling revenue alone the benefit |
| Constraint | Limiting resource or policy during a particular interval | Assuming the constraint never moves |

Additional terms used in the operating examples:

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Order-equivalent | Native work divided by work per order for a stated mix | Treating the result as a count of actual complete orders |
| Eligible inventory | Stock meeting status, location and allocation rules for demand | Counting a held pallet as ready to promise |
| Productive labor hour | An hour measured on the denominator of the selected work rate | Applying a paid-hour rate to net hours |
| Paid labor hour | An hour on the payroll clock, including applicable nonproductive time | Assuming every paid hour is available for direct work |
| Resource-minute | One minute of a specified labor or equipment resource | Adding minutes from resources that are not interchangeable |
| Starvation | A station lacks ready input during available processing time | Calling any empty queue a defect |
| Blocking | A station cannot pass completed work to its next authorized destination | Attributing all low output to operator speed |
| Release | Authorization to begin a defined unit of work | Treating release as completion or customer promise |
| Backlog | Open demand at a stated boundary and time | Assuming every open order is late |
| Aged backlog | Open work segmented by elapsed age and due time | Relying on mean age alone |
| Partial shipment | Dispatch of some, but not all, required order contents | Calling the order complete without the agreed rule |
| Work content | Resource time required for a specific class of work | Assigning equal time to all lines |
| First-pass yield | Good output without correction divided by eligible first attempts | Dividing corrected completions by first attempts |
| Recovery | Action that avoids or reduces a service failure | Counting a late recovery as an on-time handoff |
| Incremental cost | Cost that changes because an option is chosen | Charging the same fixed overhead to both alternatives |
| Counterfactual | Outcome expected without the proposed action | Assuming every delayed order would cancel |
| Cutoff slack | Time remaining between predicted readiness and the required handoff | Treating a forecast as guaranteed capacity |
| Exposure | Quantity or value at risk under a defined failure scenario | Adding a penalty to lost contribution when it already includes it |

## 5. The mechanics, step by step

### Step 1: Define the service and economic decision

Input: committed orders, promise times, cutoffs, planned arrivals and the decision horizon.
Output: a decision statement with an owner and a measurable success criterion.
Example: decide by 10:00 whether to borrow two trained packers for the 14:00 carrier departure.
State the customer population, affected service, units, time zone, and the latest acceptable evidence of handoff.
Ask whether the constraint is physical capacity, an information hold, a staffing rule, or an earlier customer cutoff.
The branch for a safety or product-integrity concern is containment and authorized review, not faster release.
Identify incremental contribution, expediting cost and penalty exposure separately; do not sum them if contracts already include the same loss.

### Step 2: Map paths and inventory boundaries

Input: floor walk, operating procedure, WMS event definitions, interviews with front-line staff.
Output: a directed map showing actual movement, controls, queues and alternate paths.
Walk an inbound pallet from arrival to available stock, then trace an order from release to carrier possession.
Mark every change of custody and the person or system authorizing it.
Record separate return, damage, quality-hold, short-pick and cancellation paths.
For cross-dock work draw the direct path, but retain receiving and shipment reconciliation.
Do not assume a location named staging represents one physical area or one operational state.

### Step 3: Establish the baseline and reconcile quantities

Input: beginning and ending queue counts, arrival and departure events, reversals, adjustments and stock status.
Output: a reconciled table for each stage and an exception list with owners.
For a queue in units, closing = opening + entering - exiting + net corrections.
A transfer between two internal locations is an exit from one queue and an entry to the other, not a receipt into the building.
Match event timestamps to actual time zone and identify cancelled or corrected records.
Count physical stock in a sample of locations and investigate identity or status differences before relying on the ledger.
Quantify missing events; an unreconciled difference is not automatically shrink.

### Step 4: Translate demand into work by station

Input: order profile, units per line, lines per order, cartonization, inbound mix and routing.
Output: station workloads in their native units and conversion assumptions.
For example, forecast orders × expected lines/order gives expected lines; lines × expected units/line gives expected units.
Use measured joint distributions when possible: multiplying independent averages can misrepresent correlated order characteristics.
Split e-commerce each-pick from pallet dispatch if their work content and routes differ.
Account for replenishments, quality checks and rework as workload, not invisible overhead.
For a shared station, calculate each class's resource minutes and add them before comparing to available minutes.

### Demand decomposition: two channels sharing a building

Suppose a shift is expected to process 360 parcel orders and 120 store orders.
Parcel orders average 2.0 lines and 1.1 cartons; store orders average 7.0 lines and 4.0 cartons.
The parcel forecast therefore creates 720 lines and 396 cartons.
The store forecast creates 840 lines and 480 cartons.
Total expected work is 1,560 lines and 876 cartons, not 480 interchangeable units of work.
The forecast-weighted mean is 3.25 lines/order and 1.825 cartons/order.
Neither mean says how many minutes the two order classes consume.
If store lines require case handling and parcel lines require each picking, use separate observed rates.
Suppose parcel picking takes 1.2 minutes/line and store picking takes 0.8 minutes/line.
The pick workload is 720 × 1.2 + 840 × 0.8 = 1,536 labor minutes.
That is 25.6 net labor hours before any unmodeled indirect work.
Applying the parcel rate to all 1,560 lines would predict 31.2 hours, overstating workload by 5.6 hours.
The reverse error is possible if the smaller class has the higher handling time.
The building-level mean also changes if the channel mix shifts even when each channel's performance is unchanged.

Now suppose the 120 store orders must be loaded by 12:00, whereas parcel departure is at 16:00.
A shift-wide total hides a four-hour difference in usable time.
The manager needs a sequence that protects store loading while preserving enough work for parcel departure.
The relevant question is not which channel has the larger total volume.
It is whether each channel's work can pass through every shared resource before its own cutoff.
Separate capacity reservations can help, but they should be released when the protected work is complete.
Otherwise idle reserved capacity can coexist with an overdue queue in the other channel.

### Receipt controls and inventory eligibility

Before the truck arrives, the appointment record provides expected time, carrier and door requirements.
The supplier's ASN provides an expectation, not proof that the goods have arrived or are correct.
At arrival, record actual arrival and distinguish yard waiting from dock unloading.
At unloading, identify each handling unit and check observable damage without obstructing safe vehicle movement.
At verification, compare product, quantity, lot and condition with the purchase order and ASN as applicable.
Where policy allows blind counts, staff count without being led by the expected quantity.
An overage, shortage or damaged package receives a reason code and an owner.
Do not silently post the expected quantity merely to finish the truck.
Receipt posting updates the inventory ledger; it may or may not make inventory eligible for allocation.
The exact transition depends on inspection, ownership, quality release and the configured WMS state.

Consider an ASN for 40 pallets, each expected to contain 48 cases.
The expected total is 1,920 cases.
Physical verification finds 39 complete pallets and one pallet with 44 acceptable cases and four damaged cases.
The received physical total is still 1,920 cases if all four damaged cases are present.
The immediately acceptable quantity is 1,916 cases, assuming no other restrictions.
The four damaged cases remain separately identifiable and must not be reported as saleable stock.
If the four cases are missing rather than damaged, physical total is 1,916 and the discrepancy is a shortage.
Those are different financial and custody events even though the available quantity is identical.
Record the reason and evidence before deciding on a supplier claim or disposal.
The receiving supervisor owns the operational record; quality and finance own the applicable disposition.

### Put-away, reserve and replenishment

A put-away instruction should respect location capacity, compatibility, handling equipment and product restrictions.
The destination must be confirmed after the move, not assumed from the task suggestion.
A full location, blocked aisle or unsuitable rack creates an exception requiring an authorized alternative.
An employee should not improvise a location that violates load limits or segregation rules.
Reserve capacity is measured in usable positions, volume and weight, not just nominal floor area.
Empty positions can be unusable for a particular pallet because of height, weight or compatibility.
Inventory control should distinguish a confirmed position from a temporary staging position.

Replenishment links reserve stock to the pick face.
If a face holds 120 units, begins with 80 units and is expected to supply 220 units before the next replenishment window, it needs at least 140 additional units.
That minimum does not automatically specify one transfer: the physical face cannot hold 220 units at once.
Two or more timed replenishment moves may be required as picks create space.
If a transfer takes 12 minutes including travel and confirmation, two moves consume 24 equipment-minutes before congestion.
A forklift assigned to inbound put-away during those same minutes cannot perform both tasks.
If reserve stock is held or the fork operator is unavailable, the pick face can empty despite positive building inventory.
Monitor face stock and open replenishment tasks early enough to cover travel and queue time.

### Shared-resource workload

Suppose one forklift is available for 360 productive minutes during a shift.
Put-away requires 22 moves at 8 minutes each, or 176 minutes.
Replenishment requires 18 moves at 7 minutes each, or 126 minutes.
Returns disposition requires 8 moves at 6 minutes each, or 48 minutes.
The total is 350 minutes, leaving 10 minutes of planned capacity.
Nominal loading is 350 / 360 = 97.2%.
That is an arithmetic workload ratio, not an estimate of queue waiting time.
It leaves little room for travel variation, blocked locations, charging or urgent tasks.
An extra 30-minute equipment interruption would increase the shortfall to 20 minutes.
The manager can reschedule nonurgent moves, obtain qualified additional capacity, or reduce demand on this resource.
Each option must preserve replenishment and product disposition obligations.
Do not report the 10-minute planned margin as proof that all moves will finish before their individual deadlines.

### Step 5: Estimate effective capacity by time window

Input: qualified staffing, available minutes, productive rate, break plan, equipment uptime and expected mix.
Output: an hourly capacity profile, not merely a building-wide daily average.
Use net productive hours with a rate measured on the same denominator; do not subtract breaks twice.
The planned capacity of a station is hours × rate × applicable availability factor when the rate excludes downtime.
A rate already observed over paid time embeds some losses; multiplying by another loss factor double counts them.
Check whether skilled labor, equipment, storage position, floor space or a service policy is the limiting resource.
A forklift shared by put-away and replenishment cannot be fully assigned to both simultaneously.

### Step 6: Compare compatible rates and trace constraint propagation

Input: the common-basis capacity profile and the current buffer state.
Output: the likely constraint and the downstream and upstream consequences.
Divide native-unit capacity by required native units per order to estimate order-equivalent capacity for a fixed mix.
The lowest effective order-equivalent rate is a useful candidate, not proof of causal constraint.
Verify that its input is available, its output has a destination, and added capacity actually increases good shipped orders.
A full downstream buffer can block an apparently slow station; the true actionable constraint may be downstream.
A short of eligible stock can starve pickers even when physical inventory exists in reserve.
Examine both queue age and queue quantity; a small queue containing overdue priority work can be urgent.

### Step 7: Manage buffers and release

Input: capacity by interval, order priorities, available inventory and dock staging limits.
Output: a release plan with thresholds and explicit recovery actions.
Set a finite WIP cap by area and use a visible oldest-age threshold for expedited orders.
Release only work that has eligible inventory, capacity to proceed, and a realistic route to its departure.
Hold a small protective buffer before the constraint where variability justifies it.
If the buffer is empty, investigate upstream supply; if downstream staging is full, slow release and clear the dock.
Do not use a WIP cap as a reason to hide orders that remain unfulfilled; keep unreleased work visible.

### Step 8: Select and authorize a countermeasure

Input: feasible options, labor skills, economic rates, safety constraints and remaining time.
Output: a named action, start and stop time, expected effect, and rollback trigger.
Compare moving trained staff, changing release sequence, overtime, an extra carrier pickup and temporary capacity.
Ask what work is displaced when staff move; a packer borrowed from receiving can create tomorrow's inventory shortage.
Do not claim the full local productivity gain as a system gain if another stage cannot absorb it.
Calculate incremental contribution from additional good orders and subtract incremental labor, carrier and rework costs.
Separate recovered late orders from genuinely incremental sales, because their economic values differ.

### Step 9: Control the shift and learn

Input: timestamped actuals and exception queues against the plan.
Output: periodic operating review and a documented forecast-versus-actual variance.
At each huddle compare arrivals, departures, age, quality, capacity and remaining time to cutoff.
Escalate missing inventory to inventory control, missing carrier capacity to transportation, and unsafe conditions to the safety owner.
When actual performance differs from plan, record whether the cause was mix, staffing, equipment, data or execution.
Review counterfactual assumptions before claiming a saving; compare similar shifts and record confounders.
Reset buffer targets after a structural change instead of permanently running in emergency mode.

### Interval planning: why a daily total is not a schedule

The shift plan must state when work arrives as well as how much can be processed.
Consider a 12:00 departure with 240 cartons required by 11:45 for loading.
From 08:00 to 10:00 the pack team can process 80 cartons per hour, but only 40 ready cartons arrive per hour.
From 10:00 to 12:00 another 160 ready cartons arrive per hour.
There is no opening ready queue, and unused early capacity cannot be stored for later.
Assume that arrivals are even within each hour and that work completed by a boundary is available at that boundary.

| Interval | Ready arrivals | Pack capacity | Completed | Closing ready queue |
|---|---:|---:|---:|---:|
| 08:00–09:00 | 40 cartons | 80 cartons | 40 cartons | 0 cartons |
| 09:00–10:00 | 40 cartons | 80 cartons | 40 cartons | 0 cartons |
| 10:00–11:00 | 160 cartons | 80 cartons | 80 cartons | 80 cartons |
| 11:00–11:45 | 120 cartons | 60 cartons | 60 cartons | 140 cartons |

By 11:45 the team has completed 220 cartons, not the 240 required.
The total 08:00–12:00 theoretical capacity is 320 cartons, but the relevant deadline is earlier.
The first two hours lose 80 cartons of potential capacity to starvation.
The last 15 minutes before departure are reserved for loading and cannot repair the shortfall under the stated rule.
Twenty cartons are exposed even though a daily aggregate appears sufficient.
If cartons belong to multi-carton orders, the number of affected orders depends on which cartons are missing.

The supervisor could accelerate upstream preparation before 10:00, add packing capacity after 10:00, or change the loading plan.
Each option requires a distinct feasibility check.
An upstream change helps only if the work can arrive early enough and is eligible.
Additional pack capacity helps only if qualified staff and workstation space exist during the late interval.
A later loading plan helps only if the carrier and safe loading procedure permit it.
The team should also sequence complete priority orders, not simply maximize the number of easy cartons.

### Turning a queue into a release decision

Suppose the safe ready-to-pack buffer limit is 300 cartons.
At 10:00 there are 240 cartons in the buffer and 30 more already released upstream that will arrive shortly.
Only 30 cartons of buffer headroom remain before accounting for any expected completions.
If pack is expected to complete 50 cartons before the next review, a provisional release of up to 80 cartons could fit.
That calculation assumes all 50 completions actually depart the buffer and the 30 in-flight cartons arrive.
It is not permission to release 80 cartons regardless of uncertainty.

If a printer failure reduces expected completion to 20 cartons, the same calculation permits only 50 cartons.
If the next review is delayed, the supervisor should refresh the estimate rather than reuse an old number.
An order-level release rule must convert cartons to orders using the actual planned cartons for that batch where available.
When cartonization is uncertain, reserve space for the upper plausible count and update after packing.
Unreleased orders stay visible on the customer commitment board.

The release owner checks three conditions before authorizing work:

1. All required inventory is eligible, or the approved partial-shipment rule is explicitly applied.
2. The receiving buffer has safe space after allowing for work already in transit.
3. The remaining pack, staging and loading schedule can plausibly meet the order's service.

When the buffer reaches its limit, stop new releases to that path rather than obstructing an aisle.
When the oldest urgent order reaches its escalation threshold, contact the owning team and customer service.
When an order cannot make cutoff, preserve its real status and document the revised plan.
The threshold should be set from actual space and recovery time, not copied from another building.

### Slotting and replenishment as a flow decision

Slotting places products where they can be stored and retrieved under physical and product rules.
A high-frequency item placed far from the pack path may increase travel on every order.
Moving it closer can reduce travel, but the new location may have a smaller face and need more replenishments.
The decision therefore trades picker travel against forklift work, face capacity and congestion.
It also has one-time move, system-update and verification costs.

Suppose an item generates 180 picks per shift.
A proposed location saves 10 seconds of picker travel per pick.
The gross travel saving is 1,800 seconds, or 30 minutes per shift.
The smaller face needs two extra replenishments at 9 forklift-minutes each, adding 18 forklift-minutes per shift.
The two resource types are not interchangeable without a staffing and equipment assumption.
If the forklift is already the bottleneck, the extra replenishments may harm service despite the picker saving.

At USD 24 per picker-hour and USD 30 per forklift-hour, the direct time values are USD 12 and USD 9 per shift respectively.
The apparent net is USD 3 per shift before move cost, congestion, equipment cost and any impact on throughput.
These are avoidable cash savings only if the hours can actually be removed or used for valuable extra work.
If the move costs USD 150 and the USD 3 is truly recurring cash benefit, simple payback is 50 comparable shifts.
If the minutes cannot be monetized, a 50-shift payback claim is unsupported.
Before the move, validate face dimensions, handling compatibility and the replenishment window.
Afterward, compare completed orders, replenishment lateness and travel under comparable item demand.

### Packing, quality and completeness

Packing converts picked items into a shipment configuration accepted by the customer and carrier.
The work includes identifying contents, selecting packaging, applying required protection and recording the correct label.
A carton may leave pack successfully while another carton for the same order is still missing.
Order completion requires all required cartons and any applicable checks, unless an authorized partial shipment is allowed.

Suppose 200 cartons enter pack and 190 pass the first inspection.
First-pass yield is 190 / 200 = 95%.
If the other 10 are corrected and all pass later, eventual good output is 200 cartons.
Counting 200 eventual completions plus 10 corrections as 210 good cartons inflates throughput.
The correction time still belongs in the resource workload and the defect analysis.
Record the original defect, corrective action and final disposition with linked identifiers.

A faster pack procedure can increase gross cartons per hour while decreasing first-pass yield.
If 100 cartons require 60 minutes initially and 10 require 6 minutes each to correct, total work is 120 minutes.
The effective good-output rate is 100 / 2 = 50 cartons per labor-hour, assuming all corrections succeed.
Comparing the initial 100 cartons per hour with a competitor's corrected-good rate would be misleading.
Also track safety, damage and customer claims; rework is not the only cost of a packaging failure.

### Staging and carrier handoff

Staging is a finite area where ready freight waits for an assigned outbound movement.
Its capacity is constrained by physical dimensions, segregation, aisle clearance and loading sequence.
A floor-space count may overstate usable capacity when large cartons block access to other shipments.
Designate a lane, identify the shipment and record entry and exit times.
Do not place goods in an exit, in front of emergency equipment or in an unsafe travel path.

Suppose 96 cartons are staged at 13:00 in an area with a safe limit of 120 cartons.
Forty additional cartons are expected from pack by 13:30 and 30 cartons are expected to load in the same interval.
If the load happens before the arrivals, occupancy rises only to 106 cartons.
If all arrivals precede loading, occupancy reaches 136 cartons and exceeds the limit.
The same net change of 10 cartons therefore does not establish a safe peak.
The operating plan must coordinate timing, not merely the interval totals.

The loading lead can allocate the next vehicle and confirm that the carrier is ready.
The pack lead can throttle output to the affected lane when the occupancy threshold is approached.
If a vehicle is late, transportation owns the carrier escalation while the warehouse protects the safe area.
The final service event should be documented according to the agreement: for example, carrier scan or signed custody transfer.
A label or a staged carton is not automatically a carrier handoff.

### Returns and reverse flow

A return is a new inbound flow with uncertain identity, condition and disposition.
The original order identifier can help establish entitlement, but it does not establish resale eligibility.
At arrival, record the return authorization where applicable, item identity, quantity, condition and custody.
Quality or another authorized role decides restock, repair, vendor return or disposal under local policy.
Until that decision, returned stock should not inflate the available-to-promise balance.

Suppose 100 units are returned in a day: 72 are accepted for resale, 12 require repair, 10 are damaged and 6 cannot yet be identified.
The total disposition categories reconcile to 100 units.
Only the 72 accepted units may become eligible, and only after required system and location updates.
The 12 repair units are still owned inventory if policy says so, but they are not automatically saleable.
The 6 unidentified units need a controlled exception location and an owner.
If the original sale and the return are both recorded, do not count the return as a supplier receipt.

Reverse-flow work consumes receiving, inspection, storage and sometimes forklift capacity.
When those resources are shared with outbound replenishment, a return surge can indirectly delay customer orders.
The answer is not to bypass inspection; it is to schedule capacity and make the competing deadlines visible.

## 6. Worked example

### Decision and baseline

A fictional distributor operates an eight-hour day shift, 08:00–16:00, with a carrier pickup at 16:00.
The commitment is carrier handoff by 16:00 for eligible orders released by 08:00; later orders use a different promise.
Forecast demand is 1,050 orders for this service; the authorized operating plan is 1,000 orders.
The forecast is not a commitment, and the plan does not change the customer promise.
There are 1,000 eligible orders at release, all with available inventory and no initial WIP.
The mix averages 2.4 lines/order, 1.25 units/line, and 1.1 cartons/order.
For this calculation the mix remains fixed and orders can flow between stations during the shift.
Station rates are net of breaks and the stated availability; they refer to good completed work.
The shift's aggregate capacities are estimates, not proof that every order reaches the carrier on time.

| Stage | Staff and net labor hours | Rate | Native capacity | Order-equivalent capacity |
|---|---:|---:|---:|---:|
| Pick | 12 × 7.0 = 84 h | 31 lines/labor h | 2,604 lines | 2,604 / 2.4 = 1,085 orders |
| Pack | 6 × 7.0 = 42 h | 25 cartons/labor h | 1,050 cartons | 1,050 / 1.1 = 954.5 orders |
| Ship | 4 × 7.0 = 28 h | 40 cartons/labor h | 1,120 cartons | 1,120 / 1.1 = 1,018.2 orders |

Pack is the candidate constraint: about 954 whole order-equivalents versus 1,000 eligible orders.
Picking is not the immediate bottleneck even though it employs more people.
At this aggregate resolution the planned shortfall is 1,000 - 954 = 46 orders, with rounding down for complete orders.
The 1,050-order forecast would exceed release by another 50 orders, but those orders are not part of this shift's stated commitment.
Do not add that forecast gap to the 46 committed orders and call the sum a service failure.

### Check that the constraint can actually run

The pack station needs 1,050 cartons for 954.5 order-equivalents.
Pick can supply 2,604 / 2.4 × 1.1 = 1,193.5 carton-equivalents, above pack capacity.
Shipping can handle 1,120 cartons, also above pack capacity.
Thus, under the fixed-mix assumption, another qualified packer can potentially increase completed orders.
In reality, start-up delay, replenishment timing and staged-carton accumulation must be checked hourly.
If pack has no work for its first hour, the shift total overstates its reachable output before cutoff.

### Intervention A: borrow one qualified packer

A trained associate can move from an unconstrained indirect assignment for 4.0 net hours.
The assignment requires no replacement and does not cause an inbound or safety shortfall; this is an assumption to verify with its owner.
The additional pack capacity is 4.0 h × 25 cartons/h = 100 cartons.
Pack capacity becomes 1,150 cartons, or 1,045.5 order-equivalents.
Shipping now limits capacity to 1,120 cartons or 1,018.2 order-equivalents.
The demand is 1,000 orders, requiring 1,100 cartons, so 1,000 orders are feasible on aggregate capacity.
The maximum increase attributable to the intervention is 46 whole orders, not 90.9, because demand is finite.

Assume each recovered order has USD 8 incremental contribution *only if* the order would otherwise be lost.
Assume the borrowed hours create USD 112 of incremental labor or backfill cost.
The estimated contribution is 46 × USD 8 = USD 368; net incremental benefit is USD 368 - USD 112 = USD 256.
This calculation excludes unquantified customer trust, carrier charges and any downstream losses.
If customers merely receive their orders tomorrow and sales are retained, the USD 8 lost-sale assumption is inappropriate.
At USD 2 avoidable late-service cost per order instead, the estimated benefit is USD 92 - USD 112 = minus USD 20.
The break-even avoidable loss is USD 112 / 46 = USD 2.43 per recovered order, rounded to cents.

### Intervention B: overtime after cutoff

Four packer hours after 16:00 also add 100 cartons of capacity but cannot meet a 16:00 handoff by themselves.
If the carrier accepts a paid 18:00 pickup, the decision additionally requires carrier confirmation and a feasible loading schedule.
Suppose the added carrier pickup costs USD 180 and overtime costs USD 140; total incremental cost is USD 320.
Under the USD 8 genuinely lost contribution assumption, 46 recovered orders produce USD 368, for USD 48 net.
This is lower than intervention A's USD 256 and relies on a later pickup being contractually acceptable.
If it misses the original promise, it may recover sales but cannot be counted as an on-time shipment.
The comparison is illustrative and does not include fixed site cost already incurred under either option.

### Shift control

At 12:00 the supervisor should compare actual completed cartons and aging orders with the remaining pack and ship capacity.
The decision should be revised if the order mix differs, a printer fails, replenishment is delayed, or the carrier changes cutoff.
Track the oldest order and the expected load time, not only total completed volume.
If additional pack output fills a finite staging buffer, stop excess release and clear staging safely.
At close, reconcile every released order into shipped, cancelled, held, or still-open status.
The finance partner should audit whether the supposedly lost orders would actually have cancelled.

### Hour-by-hour feasibility of intervention A

The aggregate case needs a timing check before a supervisor can authorize it.
For this illustration, assume picking supplies 275 ready cartons in each two-hour interval.
The six baseline packers jointly complete 262.5 cartons per interval.
The four shipping associates jointly handle 280 cartons per interval.
The borrowed packer works the final four net hours, adding 50 cartons in each of the last two intervals.
Assume within-interval processing is sufficiently smooth and a ready carton can move through successive stages within the interval.
The table reports carton-equivalents, so fractional entries are a planning approximation.

| Interval | Ready arrivals | Pack capacity | Packed | Pack queue at end | Ship capacity | Shipped | Ship queue at end |
|---|---:|---:|---:|---:|---:|---:|---:|
| 08:00–10:00 | 275 | 262.5 | 262.5 | 12.5 | 280 | 262.5 | 0 |
| 10:00–12:00 | 275 | 262.5 | 262.5 | 25.0 | 280 | 262.5 | 0 |
| 12:00–14:00 | 275 | 312.5 | 300.0 | 0 | 280 | 280.0 | 20.0 |
| 14:00–16:00 | 275 | 312.5 | 275.0 | 0 | 280 | 280.0 | 15.0 |

At 12:00, 25 carton-equivalents await packing.
In the next interval, 25 + 275 = 300 are available, below the boosted 312.5 capacity.
Shipping receives 300 but can process only 280, leaving 20 staged.
In the final interval, shipping receives 275 more and can process 280, leaving 15 staged at 16:00.
Thus this particular staffing timing ships 1,085 carton-equivalents by cutoff, not all 1,100.
The aggregate calculation identified a potentially sufficient total, but the interval example shows why its timing must be verified.

One remedy is to schedule the borrowed packer for 10:00–14:00 rather than the last four hours.
Under the same assumptions, pack completes 262.5, 287.5, 275 and 275 cartons across the four intervals.
The corresponding shipped quantities are 262.5, 280, 280 and 277.5 cartons.
The final ship queue is zero and the cumulative ship total is 1,100 carton-equivalents by 16:00.
In the second interval, the 7.5-carton staging queue absorbs pack output above ship capacity.
The next interval ships that queue plus part of the new output and leaves 2.5 cartons.
The final interval clears the remainder if the carrier handoff can occur continuously through 16:00.

This schedule is still not a proof about individual orders.
Actual carton counts are integer, the final load needs its own time, and shipment groups may have different deadlines.
The 1,100 cartons must also be the complete set for the 1,000 committed orders.
If the carrier requires loading complete by 15:45, replan against 15:45 rather than using the 16:00 endpoint.
If a safe staging cap is below the model's 7.5-carton peak, the schedule must be modified.
The value of the table is to expose a decision dependency that the shift total hid.

### Demand, costs and the customer counterfactual

The baseline estimate exposes 46 orders to a miss; it does not prove that 46 orders cancel.
Suppose a later analysis estimates 10 cancellations among those 46 without intervention.
If each canceled order would have contributed USD 8 and each of the remaining 36 would incur USD 2 in avoidable late-service cost, the estimated gross benefit is:

```text
10 orders × USD 8/order + 36 orders × USD 2/order = USD 152
USD 152 - USD 112 incremental labor = USD 40 estimated net benefit
```

The USD 2 late cost is assumed not to be included in the USD 8 lost contribution.
If a cancelled order also incurs an independent credit, add it only when the contract and accounting treatment support it.
The ten cancellations are an illustrative counterfactual estimate, not an observation from the fictional shift.
If only 5 orders cancel, gross benefit is USD 40 + USD 82 = USD 122 and net benefit is USD 10.
If none cancel, gross benefit is USD 92 and net benefit is negative USD 20, as above.
The decision depends on both the probable outcome and the range of plausible outcomes.

An expected-value calculation does not settle every service decision.
A contract may require an action to avoid a critical failure even when a simple average calculation is unfavorable.
Conversely, a penalty may be capped or waived, so it should not be assumed to apply to every late order.
The decision owner records the service obligation separately from the financial estimate.
Afterward, compare the observed missed and canceled orders with the assumed exposure.
This comparison can improve the next estimate, though the unobserved outcome without intervention still requires a defensible comparison.

## 7. Code, query, or calculation

The following standard-library Python calculation reproduces the aggregate capacity and decision comparison.
It is a steady-mix arithmetic screen, not a scheduling simulation or a guarantee of on-time delivery.
Run it with Python 3; it requires no files or third-party packages.
Counts of complete orders are rounded down; intermediate order-equivalent capacities remain fractional.

```python
from math import floor

orders = 1000
lines_per_order = 2.4
cartons_per_order = 1.1
pick_lines = 12 * 7.0 * 31
pack_cartons = 6 * 7.0 * 25
ship_cartons = 4 * 7.0 * 40


def completed(pack_capacity):
    limits = (
        orders,
        pick_lines / lines_per_order,
        pack_capacity / cartons_per_order,
        ship_cartons / cartons_per_order,
    )
    return floor(min(limits))


baseline = completed(pack_cartons)
intervention = completed(pack_cartons + 4.0 * 25)
recovered = intervention - baseline
incremental_cost_usd = 112
lost_contribution_usd_per_order = 8
net_usd = recovered * lost_contribution_usd_per_order - incremental_cost_usd
print(f"baseline={baseline} orders")
print(f"intervention={intervention} orders")
print(f"recovered={recovered} orders")
print(f"net=${net_usd:.2f}")
```

Expected output:

```text
baseline=954 orders
intervention=1000 orders
recovered=46 orders
net=$256.00
```

The use of floor is appropriate only for a homogeneous order-equivalent approximation.
Actual individual orders have integer line and carton counts and must be scheduled as discrete work.
Sensitivity: at 1.2 cartons/order, baseline pack capacity is 875 orders and ship capacity is 933 orders.
After the same 100-carton addition, pack supports 958 orders but ship supports only 933; the constraint moves to shipping.
The code assumes no lost sales unless the economic input explicitly states them; use an independently evidenced value.
If a station's historical rate is measured on paid hours, input paid hours rather than reducing them again for breaks.

### A second calculation: corrected workload and capacity

Use the following hand calculation when task events contain both first attempts and correction work.
The example is illustrative and is measured over one shift.

| Quantity | Value | Boundary |
|---|---:|---|
| Initial pack attempts | 480 cartons | First attempt enters pack |
| First-pass good | 456 cartons | Pass required check without correction |
| Corrected good | 18 cartons | Pass after a linked correction |
| Still unresolved | 6 cartons | Not accepted by shift end |
| Total pack labor | 20 labor-hours | Includes first attempts and corrections |

First-pass yield is 456 / 480 = 95%.
Eventual good yield by shift end is (456 + 18) / 480 = 98.75%.
Good output is 474 cartons and good output per labor-hour is 474 / 20 = 23.7 cartons.
The 6 unresolved cartons remain visible as open work, not as good output.
If there were 24 correction events, counting them in addition to the 480 first attempts would not produce 504 distinct cartons.
The record should link each correction to its original carton and preserve the final status.

If the next shift has 18 compatible net labor-hours at the same good-output rate, a simple estimate is 18 × 23.7 = 426.6 good cartons.
That estimate requires comparable mix, rework, availability and material supply.
It is not evidence that 426 complete orders will ship.
If the 6 open cartons are worked again, they are part of next-shift workload but not new customer demand.
Avoid comparing an output rate that includes correction hours against one that excludes them.

### Sensitivity of the staffing decision

Keep the 46-order maximum and USD 112 incremental cost from intervention A.
The table varies the assumed avoidable value per recovered order and holds all other conditions fixed.

| Avoidable value per recovered order | Gross benefit | Net benefit | Interpretation |
|---:|---:|---:|---|
| USD 0 | USD 0 | minus USD 112 | No quantified financial recovery |
| USD 2 | USD 92 | minus USD 20 | Delayed orders with low avoidable cost |
| USD 2.43 | USD 111.78 | minus USD 0.22 | Rounded break-even approximation |
| USD 4 | USD 184 | USD 72 | Positive if value is actually avoidable |
| USD 8 | USD 368 | USD 256 | Requires the stated lost-contribution assumption |

The exact break-even value is USD 112 / 46, approximately USD 2.4348 per order.
Rounding to USD 2.43 for reporting leaves a small negative residual.
If only 30 orders can actually be recovered, break-even becomes USD 112 / 30 = USD 3.73 per order, rounded.
The number of recoverable orders depends on timing, inventory, order completeness and carrier capacity.
Recalculate both the numerator and denominator when the countermeasure changes.

## 8. Data, standards, and regulatory context

The WMS task log, order ledger, labor records, inventory ledger and carrier events each have distinct grains.
The order is not the carton; the carton is not the task; an employee clock punch is not an operation completion.
Join through stable identifiers, deduplicate corrected events and preserve history rather than overwriting status changes.
Record event time, ingestion time, time zone, source, reason code and correction linkage where available.
A missing scan is an unknown event, not evidence that the underlying movement did not happen.
Compare the physical observation against the WMS record and investigate differences before altering the ledger.

GS1 identification standards describe globally interoperable identifiers such as the Global Trade Item Number (GTIN) and Serial Shipping Container Code (SSCC).
An SSCC can identify a logistics unit; it does not by itself establish a correct physical count or a successful carrier handoff.
The relevant GS1 General Specifications define the application identifiers and encoding rules; trading partners must agree on implementation.
An ASN may use electronic data interchange (EDI) or an application programming interface; neither format guarantees accuracy.
Supplier expectations and receipt discrepancies need independent verification and contractually agreed resolution.

In the United States, OSHA's powered industrial truck standard, 29 CFR 1910.178, includes operator training and evaluation requirements.
Applicable obligations depend on the workplace and jurisdiction; this educational lesson is not legal advice.
A staffing option that borrows a forklift operator requires documented authorization for the relevant equipment and workplace.
OSHA's warehousing materials discuss hazards including forklifts, material handling and loading docks.
A productivity target does not override equipment guarding, traffic separation, safe lifting or required breaks.
Follow applicable law, collective agreements and local procedures; consult qualified safety and legal professionals.

Inventory status may encode saleable, quarantine, damaged, expired or otherwise restricted stock.
The meaning and authority for each transition must be defined with quality and finance, particularly for regulated goods.
For temperature-sensitive stock, an apparently available quantity may be ineligible until condition evidence is reviewed.
Customer, worker and shipment identifiers can be sensitive; restrict access and minimize retention according to policy and law.
Metrics for individual performance need context, documented purpose and safeguards against unsafe incentives.

A data contract for this lesson should specify the site, event, key, timestamp, unit of measure, status and inclusion rule.
Publish whether ship time means label printed, dock scan, vehicle departure or carrier acceptance.
Publish whether dock-to-stock ends at receipt posting, put-away confirmation or stock made eligible for allocation.
Local definitions are legitimate only if their differences are visible when comparing buildings.

### Event dictionary for a traceable order

The following is a proposed local contract, not a universal WMS schema.
Each site must map its actual events and identify the authoritative producer.

| Event | Suggested grain and key | When recorded | What it establishes | What it does not establish |
|---|---|---|---|---|
| Vehicle arrival | Site and vehicle visit ID | Vehicle reaches agreed site boundary | Arrival for yard timing | Goods unloaded or verified |
| Handling unit scan | Site and SSCC or local unit ID | Unit is scanned at receiving | Identity read at a location | Contents independently counted |
| Receipt verification | Receipt line and verification version | Count and condition accepted | Verified quantity under the local procedure | Allocation eligibility in every system |
| Quality release | Stock lot and decision version | Authorized hold release | Specified quality restriction removed | Location is accessible or stock unallocated |
| Put-away confirmation | Movement task and completion version | Physical placement confirmed | Recorded destination and task completion | Item is not subject to another hold |
| Allocation | Order line and allocation version | Eligible quantity is reserved | Demand linked to eligible stock | Order has been picked |
| Pick completion | Pick task and completion version | Required quantity confirmed picked | Task output under the scan procedure | Complete customer order |
| Pack acceptance | Carton and acceptance version | Carton passes required check | Carton ready for next step | Every carton in the order is ready |
| Staging entry | Carton and staging visit | Carton enters identified lane | Recorded staging custody | Carrier has accepted the shipment |
| Carrier acceptance | Shipment and proof reference | Agreed handoff event occurs | Evidence at the defined service boundary | Final customer delivery |
| Return disposition | Return line and decision version | Authorized inspection decision | Assigned return path | Automatic resale eligibility before posting |

Retain the original source event identifier and correction relationship.
An update to an event is not necessarily an additional movement.
If one source reuses a task identifier for multiple attempts, add an attempt or version key.
When timestamps arrive late, separate the event's occurrence time from the time the platform received it.
Specify whether reporting periods use local site time or another agreed clock.
Resolve daylight-saving transitions and duplicate local clock times with offsets or normalized timestamps.

### Joining grains without multiplying work

An order can have several lines, each line can have multiple pick tasks, and an order can produce several cartons.
Joining all three child tables directly at order ID can produce a many-to-many multiplication.
For one order with 3 lines, 4 pick tasks and 2 cartons, a naive join can produce 3 × 4 × 2 = 24 rows.
Summing line quantities over those rows would count each line eight times if all combinations appear.
Summing carton records would count each carton twelve times.
Those counts are artifacts of the join, not evidence of more work.

Aggregate each child table to the intended reporting grain before joining.
For an order-level service report, derive distinct required lines, eligible quantities, accepted cartons and handoff status separately.
Then join one record per order and reporting version, retaining exceptions when a key is missing.
For task-level productivity, retain the task grain and do not attach unrelated carton rows without a bridge.
A bridge table should state the relationship and effective period when cartons contain multiple order lines.
Reconcile totals before and after each transformation.

### Definitions that change management action

**Open order:** an order not yet complete under the published customer rule at the reporting instant.
A canceled order is not an open order, but cancellation must remain visible in a separate outcome category.
A partially shipped order stays open unless the contract explicitly defines the remaining quantity as canceled or waived.
The report should show the promised event and time, not infer lateness from order creation alone.

**On-time shipment rate:** number of eligible orders with complete carrier handoff by the agreed cutoff divided by eligible orders due in the cohort.
Publish whether the cohort includes canceled orders, customer-requested delays, and approved partial shipments.
Late-arriving carrier proof should update the result under a stated correction and publication policy.
Report missing proof separately rather than assuming it is on time or late.
For example, 92 proven on time, 5 proven late and 3 missing proofs out of 100 due orders yield a proven on-time rate of 92%.
The final rate cannot exceed 95% if the 5 proven late remain late, and could remain 92% if all 3 missing are late.

**Dock-to-stock:** elapsed time from a named arrival event to the defined allocation-eligible event.
Report the population, median, tail and incomplete or held receipts.
An average of completed receipts alone can improve while the oldest held receipts remain excluded.
For a cohort of 100 receipts, show how many have a valid endpoint and how many are unresolved at the reporting time.
Do not set unresolved duration to zero.

**Queue age:** reporting time minus queue-entry time for each item still in the queue.
Choose a unit such as carton, pallet or order and retain priority and due time.
The oldest age helps surface an exception, but a single old low-priority item does not describe the whole queue.
Pair an age distribution with quantity, commitment exposure and reason codes.
If work exits and returns for rework, define whether the original clock continues or a new visit is created.

### Data quality checks before a bottleneck claim

1. Count source records by site, day, event type and missing-key status.
2. Check that event timestamps are plausible and that documented reversals link to an original event.
3. Identify arrivals recorded after completion and route them for investigation rather than silently sorting them away.
4. Reconcile beginning inventory, verified receipts, external shipments and net adjustments at a compatible unit.
5. Reconcile each internal queue separately, using transfers as exits and entries rather than new building receipts.
6. Compare physical samples with recorded location and status.
7. Inspect missing carrier proof and unresolved quality holds as explicit populations.
8. Compare paid hours, net hours and event coverage before computing labor rates.
9. Segment by order class and interval before attributing a change to a person or a process.
10. Publish remaining limitations alongside any decision recommendation.

These checks do not require a perfect system before any action is taken.
They do require that the size and direction of known uncertainty be visible.
If scans are missing during a network outage, a manual count may support an immediate safety decision.
The later data correction should retain its source and reason, not pretend that a scanner recorded the movement.

## 9. Industry practice and case evidence

### DHL and warehouse automation

DHL Supply Chain publicly reported a deployment of autonomous mobile robots with Locus Robotics in its warehouses.
The company's announcement discusses using robots to support order fulfillment and identifies the partnership.
This is a company account, evidence grade C, not an independent controlled estimate of total warehouse benefit.
The fact of a robot deployment does not establish that the same equipment fits a site with a different order profile.
[Inference] A practical adoption review should examine handoffs, replenishment, pack capacity, charging and exception work as well as pick travel.
The source does not provide the worked example's rates, contribution or payback.

### Amazon and fleet coordination

Amazon stated in July 2025 that it had deployed more than one million robots and introduced DeepFleet for fleet coordination.
This is a company claim, evidence grade C, and its fleet scale does not imply a comparable business case for a smaller DC.
The announcement is evidence of the company's stated operating direction, not proof of an independently audited improvement at every site.
[Inference] Fleet movement efficiency can shift the limiting operation to packing, replenishment, maintenance or dispatch.
Test that inference with station event data and capacity before purchasing equipment.
Do not treat an announced technology or quoted local percentage as a building-wide or customer-wide benefit.

### Research evidence

De Koster, Le-Duc and Roodbergen review design and control of warehouse order picking in the European Journal of Operational Research.
Their review provides a framework for interacting decisions including layout, storage assignment, routing and batching.
It is a peer-reviewed secondary source, evidence grade A for the literature it reviews, not a measurement of this fictional site's labor cost.
Its setting and publication date should be considered when translating findings to a particular facility.
[Inference] Since design and control interact, evaluate picking changes against downstream pack, inventory availability and dispatch timing.
The paper does not support assigning the same numerical improvement to all warehouses.

A useful case-study protocol records baseline, demand mix, hours, service definition, capital, operating costs and measurement period.
Ask whether quality and safety changed, whether the comparison had seasonal controls, and who funded the claim.
Where evidence is absent, present an assumption and a pilot plan, not an invented benchmark.

### How to transfer evidence to a local decision

An external case can identify a possible mechanism without providing the parameters for a local investment model.
For the DHL example, the public announcement supports that the company reported a robotics partnership and deployment.
It does not establish this lesson's pack rate, the amount of labor displaced, or the number of on-time orders saved.
Those quantities must come from local observations or be labeled as assumptions.

Translate a published claim through five questions:

1. What operational boundary did the source measure: travel, picks, cartons, or customer orders?
2. What population and mix were included, and were difficult exceptions omitted?
3. What comparison period or control supports the claimed change?
4. Which implementation and recurring costs were included?
5. What changed in downstream flow, safety, quality and service?

If the source cannot answer a question, record the gap rather than fill it with the vendor's most favorable estimate.
For a warehouse with a constrained dock, faster picking may increase staging congestion rather than completed shipments.
That is a testable local hypothesis, not a claim that every robot deployment produces congestion.
The same discipline applies to a published layout change or labor standard.

### A practical four-week comparison

Before changing the process, collect order mix, qualified hours, good output, queue ages and carrier handoff for comparable shifts.
Define a comparison group if feasible, such as a similar area not receiving the intervention.
Record known differences in product profile, customer cutoff, replenishment and staffing.
Random assignment can improve interpretation where safe and practical; an operationally selected comparison has more selection risk.

During the pilot, retain the same event definitions and include all eligible orders, including exceptions.
Observe the complete path from release to handoff rather than only the station that changed.
Record implementation support and manual workarounds, not just recurring license cost.
Set thresholds for safety incidents, defect rate, backlog age and customer service before reviewing results.
Authorize the supervisor to pause the pilot if a guardrail is breached.

At the end, compare the intervention area's change with the comparison area's change.
For example, if on-time performance rises from 90% to 94% in the pilot area and from 91% to 93% in the comparison area, the difference in changes is 2 percentage points.
That calculation is descriptive; different order mix or other simultaneous changes can still explain the difference.
Report cohort sizes and uncertainty, and do not describe a small uncontrolled pilot as proof of a permanent effect.
Scale only after confirming the downstream capacity and incremental economics for the proposed operating range.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Releasing all work at once | WIP and travel congestion rise | Competing jobs occupy finite space | Orders miss cutoff | Cap releases and age-prioritize |
| Counting receipt as available | Pick shorts despite inbound arrival | Inspection or put-away unfinished | False promise | Use eligible inventory status |
| Wrong capacity units | Pallets compared with orders | Work content ignored | Wrong bottleneck | Convert by measured mix |
| Double-counted break loss | Net hours reduced twice | Incompatible denominator | Excess staffing | Document rate basis |
| Full outbound staging | Packed cartons cannot move | Ship cannot consume output | Pack blockage | Set staging cap and clear lane |
| Starved pack | Empty pack queue with open orders | Pick or replenishment late | Idle bottleneck | Protect a controlled buffer |
| Hidden quality hold | Stock exists but cannot allocate | Status excluded from planning | Last-minute shortages | Report holds and owners |
| Rework counted twice | Output exceeds distinct orders | Rescans treated as completions | Inflated productivity | Deduplicate at order grain |
| Borrowed skill unavailable | Absences or authorization gaps | Capacity assumed fungible | Unsafe or infeasible plan | Maintain skill and safety matrix |
| Late handoff misreported | Labels printed before carrier arrival | Wrong event boundary | Misstated service | Use carrier acceptance evidence |
| One-day average hides peak | Queues grow before dispatch | Capacity arrives too late | Cutoff missed | Model intervals and cutoffs |
| Replenishment neglected | Empty faces with reserve stock | Forklift shared elsewhere | Picker starvation | Schedule and monitor replenishment |
| Inventory discrepancy ignored | Repeated location corrections | Physical and logical stock diverge | Phantom availability | Count, trace and resolve |
| Financial double count | Saving exceeds avoidable cost | Labor and contribution overlap | Misleading investment case | Separate revenue, cost and cash |
| Unsafe speed incentive | Near misses and damage rise | Only volume is rewarded | Injury and poor quality | Pair speed with safety and quality |

### Reading warning signs as causal chains

**High pick productivity and late orders.**
Start with the order-completion and carrier-handoff records, not the picker leaderboard.
Check whether picked lines belong to orders missing other lines, whether pack is starved or blocked, and whether staging is full.
An increase in picked lines can coexist with a decrease in complete customer orders.
The next action depends on which of these mechanisms is observed.

**Low recorded labor productivity.**
First check whether the numerator contains all completed work and the denominator matches the rate definition.
A scanner outage can hide completions, while paid hours can include training or an equipment interruption.
Next check whether work was available and whether downstream space existed.
Only after those checks should the team investigate the work method.
An individual comparison without mix and assignment context can reward unsafe shortcuts.

**A growing hold queue.**
Separate newly received holds from cases already overdue for disposition.
Show count, age, reason, owner and customer exposure.
A queue may grow because quality reviewers lack capacity, because supplier defects increased, or because release events are missing.
Removing a hold without authority does not fix any of those causes.
Escalate the underlying reason while protecting restricted stock.

**A disappearing backlog.**
Check whether orders shipped, canceled, moved to another status, or vanished from the reporting population.
A dashboard filter that excludes unreleased orders can make backlog appear to improve when no customer work is complete.
Reconcile the order cohort from opening to closing status before claiming a recovery.
Keep late orders visible even when they were successfully shipped after cutoff.

## 11. Assumptions and limits

| Assumption | Why used | What breaks it | Test or response |
|---|---|---|---|
| Fixed mix | Convert rates to orders | Seasonal or channel shift | Recalculate by order class |
| Net rates | Avoid double-counted losses | Rate excludes downtime | Reconcile hours to paid time |
| Available inventory | Isolate capacity issue | Holds or missing stock | Check status and physical counts |
| Sufficient input to pack | Pack can use its capacity | Late picks or replenishment | Compare hourly arrival profile |
| Sufficient downstream space | Pack completions can leave | Staging congestion | Measure occupancy and dwell |
| Staff are qualified | Borrowed capacity is feasible | Training or equipment restriction | Check skill matrix and supervisor |
| Uniform contribution | Illustrate economics | Product and customer margins differ | Segment contribution and cancellation |
| Orders truly lost | Calculate incremental sales benefit | Orders only delayed | Compare cancellation evidence |
| Carrier accepts pickup | Translate output to shipment | Capacity or cutoff changes | Obtain carrier confirmation |
| No material rework change | Use good-output rates | Errors increase under pressure | Track defect and rework rate |
| Aggregate approximation | Screen options quickly | Start-up and interval variability | Use a time-phased model or simulation |
| Reliable events | Reconstruct flow | Missing or late scans | Reconcile against observations |

None of these assumptions is a substitute for a safe and legally compliant operating procedure.
Uncertainty in a material assumption should lead to a range and an owner for measurement.
A short pilot must include a fallback if a queue grows or safety deteriorates.

### Uncertainty register for a live decision

For each material input, record a value, source, observation window, uncertainty and owner.
An observed rate from a quiet Tuesday may not represent a promotion with more lines per order.
A carrier cutoff taken from an old schedule may not represent today's actual capacity.
A calculated resource requirement may be precise to a decimal while the rate itself varies substantially.
Do not let formatting imply more certainty than the data supports.

| Input | Low-side scenario | High-side scenario | Decision implication |
|---|---|---|---|
| Ready work to pack | Late replenishment delays arrivals | Work arrives as scheduled | Added pack labor may be starved in the low case |
| Cartons per order | Fewer cartons under the actual mix | More cartons under the actual mix | The same carton capacity supports fewer orders in the high case |
| Carrier availability | Planned pickup unavailable | Pickup confirmed with sufficient space | Output may need a revised promise in the low case |
| Avoidable value | Orders remain and ship later | Orders would otherwise cancel | The same intervention has different financial value |
| Data coverage | Missing scans affect a key interval | Complete validated event trail | Physical observation may be needed before acting |

Scenario values must be plausible for the decision and should not be presented as statistical confidence bounds without evidence.
Sensitivity can identify which assumption is worth measuring first.
If the decision changes only when carrier capacity changes, call the carrier before refining the labor rate.
If every plausible scenario violates a safety limit, reject the unsafe option rather than averaging the outcomes.
Document the selected option, its owner and the condition that triggers reconsideration.

### When the simple capacity model is inadequate

Use a more detailed time-phased model when multiple departures have different priorities.
Model individual tasks or batches when order completeness, batching or sequence materially changes outcomes.
Use observed distributions rather than a single mean when variability drives buffer overflow or cutoff risk.
Represent shared equipment explicitly when the same forklift serves receiving and replenishment.
Preserve separate units and conversion rules for each demand class.

Simulation can compare alternatives, but a detailed model is not automatically accurate.
Validate event boundaries, routing, initial queues and resource calendars against observed operations.
Compare model outputs with held-out periods and investigate material discrepancies.
Include random-seed and scenario sensitivity if stochastic arrivals or task times are modeled.
For a near-term shift decision, a transparent interval table may be more useful than an unvalidated simulation.

## 12. Exercises

For calculations, state the unit, time window, event boundary and assumptions.
Do not treat a calculated average rate as proof of individual-order service.

### Exercise 1 — Draw the system

A distributor receives mixed pallets, inspects a sample, puts accepted stock in reserve, replenishes pick faces and ships parcels.
A portion of receipts is held for damage; returned goods require inspection before being sold again.
**Deliverable:** a flow diagram and a short explanation; suggested time: 20 minutes.
**Acceptance:** show physical and information flows, all major queues, the hold and return paths, and a named owner for each handoff.
Identify one point where WMS status can differ from physical location.

### Exercise 2 — Reconcile a buffer

At 08:00 a put-away queue contains 140 pallets.
During the shift 275 verified pallets enter, 310 leave for storage, and 5 duplicate arrivals are reversed.
**Deliverable:** closing queue in pallets and a reconciliation equation; suggested time: 10 minutes.
**Acceptance:** apply the reversal once, distinguish internal transfer from a building receipt, and flag physical discrepancies.

### Exercise 3 — Compare capacities

An order mix has 2.5 lines/order and 1.2 cartons/order.
Pick produces 2,250 lines/shift; pack produces 1,140 cartons/shift; ship produces 1,080 cartons/shift.
**Deliverable:** capacity of each station in order-equivalents and the candidate constraint; suggested time: 15 minutes.
**Acceptance:** show unit conversion and identify a tie if one exists; do not claim confirmed causality from rates alone.

### Exercise 4 — Recover a queue

A staging queue starts with 120 cartons, receives 680 cartons and dispatches 710 cartons during an eight-hour shift.
The next shift has a safe staging limit of 100 cartons.
**Deliverable:** closing queue, distance to limit and one control action; suggested time: 10 minutes.
**Acceptance:** use conservation, quantify headroom and identify a physical-space risk.

### Exercise 5 — Evaluate staff movement

A temporary shift of trained labor can recover at most 30 otherwise lost orders.
Avoidable contribution is USD 6 per lost order; incremental backfill costs USD 135.
**Deliverable:** net benefit and break-even contribution per recovered order; suggested time: 15 minutes.
**Acceptance:** show both calculations and explain why the result changes if orders are only delayed.

### Exercise 6 — Design a release rule

At 10:00 a pick-to-pack buffer is near its safe 400-carton limit, while 70 urgent orders remain unreleased.
The carrier cutoff is 14:00; some urgent orders lack eligible stock.
**Deliverable:** a 10-line operating rule including owner, thresholds, priority, escalation and fallback; suggested time: 25 minutes.
**Acceptance:** protect safety, keep unreleased orders visible, check eligibility, and avoid promising shipment without capacity.

### Exercise 7 — Critique a metric

Site A reports dock-to-stock from dock arrival to receipt posting; Site B reports from receipt posting to allocation eligibility.
Site A reports 2.0 hours and Site B reports 1.5 hours.
**Deliverable:** a comparison critique and a common definition; suggested time: 15 minutes.
**Acceptance:** identify both boundary differences and propose events, population, exclusions and a tail measure.

### Exercise 8 — Propose a pilot

A supplier offers a pick automation system and claims 30% faster picking, but gives no site-level customer outcome or total-cost data.
**Deliverable:** a one-page decision brief and a four-week pilot design; suggested time: 35 minutes.
**Acceptance:** specify a baseline, matched comparison or explicit limitations, throughput, service, quality, safety, incremental cash flows, and stop criteria.
Name at least two potential downstream constraints and disclose uncertainty.

### Exercise 9 — Test a cutoff schedule

At 08:00 there is no ready-to-pack queue.
From 08:00–09:00, 30 cartons arrive and pack can complete 60 cartons.
From 09:00–10:00, 90 cartons arrive and pack can complete 60 cartons.
From 10:00–11:00, 90 cartons arrive and pack can complete 60 cartons.
The carrier requires all 180 cartons for the priority cohort to be packed by 11:00.
The first two hours' 120 arrivals are priority cartons; the final hour brings 60 priority and 30 nonpriority cartons.
Assume arrivals and completions flow continuously within each hour and priority cartons are served first.

**Deliverable:** an hourly queue table, the packed total by 11:00, the shortfall and one feasible type of intervention.
Suggested time: 20 minutes.
**Acceptance:** show unused early capacity, do not carry unused capacity to later hours, and distinguish carton shortfall from affected orders.
Explain which additional timing or resource information is needed before claiming the intervention will work.

### Exercise 10 — Reconcile a trace and value a recovery

A cohort contains 100 orders due for handoff by 16:00.
Carrier proof shows 86 complete on-time orders and 8 complete late orders.
Two orders were canceled before release and four have no carrier proof at the reporting time.
The agreed reporting policy excludes the two pre-release cancellations from the due-order denominator.
Of the four unknown orders, a physical investigation finds two staged but not handed over by 16:00; two remain unresolved.
Assume staged-but-not-handed-over means late under the commitment.

A proposed process change costs USD 90 per comparable shift.
The team estimates it would avoid 10 late orders per shift, each with USD 12 of avoidable service cost.
The USD 12 is independent of the USD 90 cost and does not include lost contribution.

**Deliverable:** report the proven on-time rate and its possible range after investigation, then calculate the estimated net benefit.
Suggested time: 20 minutes.
**Acceptance:** state the denominator and unresolved population, do not classify missing proof as success, and describe one assumption needed to claim the estimated benefit.

## 13. Self-check answers

### Exercise 1

A valid route is arrival → unload → inspection → receipt → put-away queue → reserve → replenishment → pick face → pick → pack → staging → load → carrier.
Add an inspection-to-hold branch and a return-to-disposition branch.
A warehouse receiving clerk owns the inbound count, inventory control owns discrepancies, and shipping owns the carrier handoff; local roles may differ.
A pallet may physically be on the dock after the WMS shows it as available if a receipt was posted before put-away.
A correct answer explicitly separates that system status from verified physical custody.

### Exercise 2

Closing queue = 140 + 275 - 310 - 5 = 100 pallets.
The reversal removes a duplicate entry; it is not a physical removal of five additional pallets.
Moving pallets into storage changes the put-away queue but does not reduce whole-building inventory.
If the physical queue is 103 pallets, the 3-pallet difference requires investigation, not an unsupported shrink entry.

### Exercise 3

Pick capacity = 2,250 / 2.5 = 900 order-equivalents/shift.
Pack capacity = 1,140 / 1.2 = 950 order-equivalents/shift.
Ship capacity = 1,080 / 1.2 = 900 order-equivalents/shift.
Pick and ship are tied candidates at 900 order-equivalents/shift.
Confirm actual starvation, blocking and timing before deciding which intervention increases shipments.

### Exercise 4

Closing staging queue = 120 + 680 - 710 = 90 cartons.
Headroom to the 100-carton safe limit is 10 cartons.
Monitor inflow and dispatch by interval; temporarily stop release if the safe limit is about to be exceeded.
Do not place overflow in a fire exit or travel aisle while waiting for the next shift.

### Exercise 5

Gross avoidable contribution = 30 × USD 6 = USD 180.
Net benefit = USD 180 - USD 135 = USD 45.
Break-even contribution = USD 135 / 30 = USD 4.50 per recovered order.
If the orders would ship later rather than disappear, use evidenced avoidable late costs instead of assuming lost contribution.

### Exercise 6

A strong rule appoints a shift supervisor, checks stock eligibility and carrier capacity, and reserves a safe flow path.
The rule stops or throttles release at the agreed buffer limit and escalates aged urgent work and stock holds separately.
It makes unreleased orders visible to customer service and transport, with a timed reassessment and documented fallback.
There is no unique numeric release threshold without pack and ship rates, order mix and staging conditions.

### Exercise 7

The reported intervals have different start and end events and therefore cannot rank the sites.
Define, for example, arrival at dock to stock eligible for allocation after required checks.
Use the same eligible receipt population, event timezone, exclusions and correction policy at both sites.
Report median and 95th percentile alongside missing-event coverage and a count of unresolved holds.
The 95th percentile needs enough comparable observations to be interpretable.

### Exercise 8

A strong pilot measures complete good orders handed to the carrier, not just picker travel or vendor-reported cycles.
It includes upstream replenishment and downstream pack and shipping as potential constraints.
It records installation, support, integration, maintenance, training and labor displacement or redeployment costs.
It compares similar mix and peak periods, notes selection bias, sets safety and quality guardrails and specifies a rollback.
A four-week result does not establish peak-season performance or lifetime payback without further evidence.

### Exercise 9

| Hour | Opening queue | Arrivals | Capacity | Completed | Closing queue |
|---|---:|---:|---:|---:|---:|
| 08:00–09:00 | 0 | 30 | 60 | 30 | 0 |
| 09:00–10:00 | 0 | 90 | 60 | 60 | 30 |
| 10:00–11:00 | 30 | 90 | 60 | 60 | 60 |

All entries are cartons.
Total packed by 11:00 is 30 + 60 + 60 = 150 cartons.
The shortfall against 180 required cartons is 30 cartons.
Thirty cartons of capacity are unused in the first hour because only 30 cartons arrive.
The unprocessed queue is 60 cartons, larger than the priority shortfall because 210 cartons arrived in total.
The unprocessed queue contains 30 priority and 30 nonpriority cartons under the stated priority rule.
The priority shortfall is therefore 30 cartons, not the entire 60-carton queue.

An intervention might move at least 30 of the later arrivals into the first hour if they are eligible and upstream work can be accelerated.
Alternatively, it could add at least 30 cartons of usable packing capacity after 09:00, with material and space available.
The 30-carton figure assumes that added output belongs to the required priority set.
Confirm qualifications, downstream loading, actual arrival timing and the identity of the priority cartons before committing.
The number of affected customer orders cannot be computed without the carton-to-order mapping.

### Exercise 10

The due-order denominator is 100 - 2 = 98 orders.
The proven on-time rate is 86 / 98 = 87.76%, approximately 87.8%.
After the physical investigation, 10 orders are known late: 8 with late proof plus 2 staged past cutoff.
Two orders remain unresolved.
If both unresolved orders are later proven on time, the rate is 88 / 98 = 89.80%, approximately 89.8%.
If neither is proven on time, the rate remains 86 / 98 = 87.76%.
The result is therefore between approximately 87.8% and 89.8% under the stated denominator and outcome rules.

Estimated avoidable cost is 10 × USD 12 = USD 120 per comparable shift.
Net estimated benefit is USD 120 - USD 90 = USD 30 per shift.
The estimate assumes that all 10 late orders are genuinely avoided and that the USD 12 cost is avoidable for each.
It also assumes the change does not introduce other costs or displace critical work.
Actual evaluation needs a defensible counterfactual and a stable service definition.

## 14. Interview questions

1. **How do you identify the bottleneck?** Strong: compare compatible effective capacities and confirm starvation, blockage and response to added capacity. Weak: name the team with the most workers.
2. **Why can more picker output fail to improve on-time delivery?** Strong: pack, staging or carrier cutoff can constrain the system. Weak: assume all picked lines immediately become shipped orders.
3. **How do you measure dock-to-stock?** Strong: specify start, end, population, status and tail. Weak: compare site averages without event definitions.
4. **What is the value of a buffer?** Strong: absorb variability at a space and time cost, with limits and ownership. Weak: call all WIP either necessary or waste.
5. **What changes when mix changes?** Strong: workload per order, routing and capacity conversion change. Weak: assume constant orders per labor hour.
6. **What is the difference between availability and presence?** Strong: eligible status, location and allocation matter in addition to physical quantity. Weak: count every pallet as immediately saleable.
7. **How would you value an intervention?** Strong: incremental recoverable contribution or avoidable cost less incremental cost, without double counting. Weak: multiply every local unit by gross revenue.
8. **What validates a staffing move?** Strong: qualification, displaced work, timing, safety and downstream capacity. Weak: assume all labor is interchangeable.
9. **What is the limitation of Little's Law?** Strong: consistent boundaries and representative stable averages are essential. Weak: use one snapshot and an unrelated throughput figure.
10. **How do you assess an automation claim?** Strong: check source, baseline, mix, cost, service, quality and counterfactual. Weak: apply the vendor's percentage to all site costs.

## 15. Key terms

- [Allocation](#4-definitions-and-vocabulary) — commitment of eligible inventory to demand.
- [ASN](#4-definitions-and-vocabulary) — supplier notice of expected shipment contents.
- [Bottleneck](#4-definitions-and-vocabulary) — resource limiting output under specified conditions.
- [Buffer](#4-definitions-and-vocabulary) — work or space between stages.
- [Capacity](#4-definitions-and-vocabulary) — potential output over a stated period.
- [Carton](#4-definitions-and-vocabulary) — shipping package.
- [Constraint](#4-definitions-and-vocabulary) — limiting resource or operating policy.
- [Contribution](#4-definitions-and-vocabulary) — incremental revenue less incremental variable cost.
- [Cost-to-serve](#4-definitions-and-vocabulary) — attributable cost to fulfill demand.
- [Cross-dock](#4-definitions-and-vocabulary) — direct inbound-to-outbound transfer.
- [Cutoff](#4-definitions-and-vocabulary) — latest time for a defined service event.
- [Cycle time](#4-definitions-and-vocabulary) — elapsed time between defined events.
- [Dock-to-stock](#4-definitions-and-vocabulary) — elapsed time to specified inventory availability.
- [Exception](#4-definitions-and-vocabulary) — deviation requiring a decision.
- [Hold](#4-definitions-and-vocabulary) — blocked stock or work awaiting disposition.
- [Lead time](#4-definitions-and-vocabulary) — elapsed time from requested start to outcome.
- [Order line](#4-definitions-and-vocabulary) — requested item record on an order.
- [Pick face](#4-definitions-and-vocabulary) — location supplying picking.
- [Put-away](#4-definitions-and-vocabulary) — movement to authorized storage.
- [Queue age](#4-definitions-and-vocabulary) — elapsed waiting time.
- [Receipt](#4-definitions-and-vocabulary) — verified inbound quantity.
- [Replenishment](#4-definitions-and-vocabulary) — movement from reserve to pick face.
- [Reserve](#4-definitions-and-vocabulary) — bulk storage.
- [Rework](#4-definitions-and-vocabulary) — corrective processing.
- [Throughput](#4-definitions-and-vocabulary) — good completed output per time.
- [Utilization](#4-definitions-and-vocabulary) — used versus available compatible capacity.
- [WIP](#4-definitions-and-vocabulary) — work within defined boundaries.
- [Yield](#4-definitions-and-vocabulary) — good output divided by input.

## 16. Sources

- De Koster, R., Le-Duc, T., and Roodbergen, K. J. (2007), “Design and control of warehouse order picking: A literature review,” *European Journal of Operational Research*, 182(2), 481–501. [https://doi.org/10.1016/j.ejor.2006.07.009](https://doi.org/10.1016/j.ejor.2006.07.009). Peer-reviewed review; evidence grade A.
- Little, J. D. C. (1961), “A Proof for the Queuing Formula: L = λW,” *Operations Research*, 9(3), 383–387. [https://doi.org/10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383). Foundational peer-reviewed result; evidence grade A.
- GS1, [General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications), current published edition. Identifier and barcode standards; consult the edition applicable to the implementation.
- U.S. Occupational Safety and Health Administration, [29 CFR 1910.178, Powered industrial trucks](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.178). Official regulatory text; evidence grade B.
- U.S. Occupational Safety and Health Administration, [Warehousing](https://www.osha.gov/warehousing). Official hazard and safety resources; evidence grade B.
- Amazon, [Amazon deploys its one millionth robot and introduces a new generative AI model](https://www.aboutamazon.com/news/operations/amazon-million-robots-generative-ai-foundation-model) (2025). Company account; evidence grade C.
- DHL, [DHL Supply Chain and Locus Robotics](https://www.dhl.com/global-en/home/press/press-archive/2022/dhl-supply-chain-and-locus-robotics-expand-partnership.html) (2022). Company account; evidence grade C.

## 17. Links

- [Module 13 index](_index.md)
- [Module 13 survey](../../../course/13_warehousing_fulfillment_labor.md)
- [Process flows and the WMS data model](02_wms-data-model.md)
- [Productivity metrics](03_productivity-metrics.md)
- [Module 12 transportation index](../12_transportation_routing_last_mile/_index.md)
