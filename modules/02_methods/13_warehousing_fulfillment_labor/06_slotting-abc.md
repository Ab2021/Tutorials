# Slotting and ABC

> Module 13 — Warehousing, Fulfillment, and Labor Analytics
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 13 topics 01–03
> Version 2026.5

## 1. Why this topic exists

A warehouse can ship every order and still waste substantial labor moving between locations.
Slotting decides which stockkeeping unit (SKU) occupies which eligible location and how much
inventory is presented there. It is a recurring operating decision, not a one-time sort by sales.
A picker spends time walking, searching, reaching, confirming, and sometimes waiting for stock.
A move that shortens travel can increase replenishment or congestion, so the whole process matters.

The operations manager needs a layout that can be executed safely during a busy shift.
The finance partner needs the savings expressed in paid hours and cash, not just meters walked.
The inventory team needs enough forward stock to meet demand between replenishments.
The safety lead needs physical compatibility and ergonomic restrictions enforced before optimization.
A supervisor needs move instructions that preserve traceability and can be reversed if a pilot fails.

ABC analysis ranks items by a chosen measure of work and partitions cumulative contribution.
It is a screening device: it does not prove that every A item belongs at the closest location.
A fast-moving bulky item may replenish too often in a small forward slot.
A hazardous item may not be permitted near the packing area at all.
Two items frequently ordered together can benefit from proximity, but only when the picking
method actually visits both locations in the same tour and the aisle can absorb the traffic.

This lesson makes the decision at SKU × pick-location × planning-period grain.
It distinguishes historical observation, a demand forecast, a proposed layout, and a committed move.
All example money is illustrative 2026 USD; rates and decisions are not external benchmarks.
The example uses a deliberately simplified additive travel proxy, then names what it omits.
Use measured routes or simulation before treating the proxy as a staffing commitment.

## 2. Learning outcomes

After completing this lesson you can:

1. Define the SKU, order-line, pick-face, replenishment, and location grains (exercise 1).
2. Compute velocity and cumulative-contribution ABC classes with a stated boundary rule (exercise 2).
3. Screen a proposed assignment against dimensions, capacity, safety, and stock constraints (exercise 3).
4. Calculate travel and replenishment labor for competing layouts (exercise 4).
5. Compute pair support and confidence without treating association as causation (exercise 5).
6. Evaluate a proposed move using a consistent horizon and a defensible break-even (exercise 6).
7. Design a controlled pilot, guardrails, and rollback plan (exercise 7).
8. Diagnose unstable rankings, missing events, and transfer effects (exercise 8).

## 3. The core mental model

```text
orders + picks + location master + inventory + replenishments
                         |
                         v
         reconcile event grain and planning period
                         |
                         v
     forecast visits and units; calculate ABC and affinity
                         |
                         v
       eliminate physically and operationally illegal slots
                         |
                         v
 compare feasible assignments: travel + replenishment + moves
                         |
                         v
  pilot -> measure service, safety and labor -> retain or roll back
```

Velocity describes how often work arrives, not its value to the customer.
Travel is the distance or time implied by the physical route and picking policy.
Affinity describes co-occurrence within a specified order or batch population.
Eligibility defines which assignments are legal and practical before their cost is scored.
The objective then compares feasible assignments using the same demand and cost horizon.
The final decision includes execution cost, uncertainty, and a safe implementation window.

A location close to the depot is valuable when it saves repeated travel.
Its capacity is scarce: allocating it to one SKU denies it to another.
The opportunity cost is therefore the best feasible alternative use of that location.
A ranking alone cannot represent this competition.
A slow SKU at a near slot may be justified by weight, an accessibility need, or segregation.

**Visit frequency** and **unit frequency** serve different decisions.
A line for 20 units can require one visit but consumes 20 units of forward stock.
If orders are batched, several lines may be picked in one visit.
Therefore order-line counts are a proxy for visits only when the picking process supports it.
Replenishment depends on stock consumed, usable capacity, starting stock, and the trigger policy.
Never add historical replenishments to a hypothetical slot without adjusting for capacity.

Pair affinity is conditional on how work is grouped.
A frequent pair in customer orders may not appear together in a zone picker's route.
A large order contributes many pairs; a high-volume item co-occurs with almost everything.
Support, confidence, and lift describe different features of the same population.
None by itself proves that placing items together will reduce total route length.

## 4. Definitions and vocabulary

| Term | Operational meaning | Misuse or unit |
|---|---|---|
| SKU | Distinct item and pack configuration identified for inventory control | Not a customer order |
| Pick face | Forward location from which a picker draws stock | One SKU may have several faces |
| Reserve | Stock held outside the forward pick face | Not necessarily available immediately |
| Velocity | Rate of a chosen activity per planning period | Always name visits, lines, or units |
| ABC | Classes based on ranked cumulative contribution | Cutoffs are policy, not physical laws |
| Pick line | Item row on an order or work assignment | Can require multiple physical visits |
| Visit | A picker stops at a location on a tour | Not necessarily one order line |
| Tour | Sequence of stops in one picker route | Route distance is not sum of depot round trips |
| Replenishment | Transfer from reserve to a forward face | Counts depend on starting inventory and policy |
| Usable capacity | Safe, accessible quantity in a location | Units or cases per face, not gross cube alone |
| Support | Orders containing both items divided by all eligible orders | Proportion, with denominator stated |
| Confidence | Orders containing both divided by orders containing the antecedent | Directional and not causal |
| Lift | Confidence divided by marginal prevalence of the consequent | Undefined when consequent prevalence is zero |
| Travel proxy | Simplified estimate used to compare candidate assignments | Not necessarily actual route length |
| Move cost | One-time labor and other incremental expense of a slot change | Do not mix with recurring cost |
| Payback | Upfront cost divided by incremental net recurring savings | Undefined for nonpositive savings |
| Eligibility | Physical, safety, system, and operational conditions for an assignment | A hard constraint, not a preference |
| Cube | Three-dimensional volume | Volume alone cannot establish fit |
| Congestion | Interference among people, equipment, or replenishment at shared space | A source of delay and risk |
| Slotting horizon | Period over which forecast work and costs are compared | Match it to move cost and payback |
| Stockout | Forward face lacks stock when work requires it | Reserve stock may still exist |
| Cutoff | Policy threshold for cumulative contribution classes | State how boundary-crossing items are treated |
| Guardrail | Metric whose unacceptable movement stops or changes a pilot | Not a guarantee of safety |
| Counterfactual | Estimated outcome without the proposed change | Must be comparable in mix and time |

## 5. The mechanics, step by step

### Step 1 — Define the decision and horizon

Input: facility boundary, decision owner, current picking method, and planned decision date.
Output: a documented population, time bucket, and the choice to be made.
Select a horizon long enough to recover one-time moves but short enough to forecast credibly.
For seasonal stock, compare separate demand scenarios rather than assuming a flat year.
Specify whether the question concerns one zone, the entire building, or a single pick method.
Record the current layout and actual picking rules before changing either.
If a layout change and a batching change are simultaneous, their separate effects are not identified.
Identify whether the desired benefit is throughput, labor hours, service, or safety.
A reduction in engineered minutes is not automatically a reduction in payroll.
The decision maker must know whether labor can be redeployed or scheduled differently.

### Step 2 — Reconstruct trustworthy work

Input: orders, line details, pick tasks, location history, and stock movements.
Output: reconciled activity by SKU, date, face, order, and physical visit when available.
Remove cancellations and reversals according to event type, not by deleting negative quantities.
Retain corrections in an audit table so the net activity can be explained.
Separate eaches from cases and convert using the effective pack configuration at event time.
Join locations using the assignment valid at the event timestamp, not today's location master.
Check that each event joins to one SKU and one location; duplicate joins inflate work.
Compare aggregated shipped quantities against fulfillment and inventory control totals.
Investigate missing picker scans: missing scans are not evidence of zero visits.
Record stockouts and substitutions because observed picks can be censored demand.
Do not infer demand for unavailable items solely from the volume successfully picked.
Exclude abnormal shutdown days only with a written reason and an alternative scenario.

### Step 3 — Choose velocity and calculate ABC

Input: a common period of reconciled SKU activity and the operating decision.
Output: ordered item contributions, cumulative share, and reproducible classes.
For single-order picking, completed lines can approximate required pick stops.
For batch or cluster picking, derive visits from task or tour records when possible.
Use units to estimate replenishment workload, not as an automatic substitute for stops.
Sort descending on the chosen measure; break ties deterministically by SKU identifier.
Divide each SKU's activity by the total eligible activity, and accumulate the shares.
This lesson assigns a SKU to A if the cumulative share *before* adding it is below 80%.
Otherwise it is B if the prior cumulative share is below 95%; remaining SKUs are C.
Thus the item crossing a boundary remains in the preceding class.
State that convention because assigning by ending cumulative share gives different classes.
An all-zero population has no meaningful ABC share: stop and investigate the input.
Segment incompatible handling or temperature zones before ranking locations for placement.
Compare rolling windows and evaluate whether volatile SKUs are consistently fast or seasonal.

### Step 4 — Measure affinity in the right population

Input: deduplicated SKU presence within each eligible order, wave, or batch.
Output: pair counts, support, confidence, lift, and an operationally relevant candidate list.
Choose the grouping unit that reflects the route whose travel you want to change.
For order affinity, each order contributes at most once to a particular pair.
For batch affinity, reconstruct actual batches; do not substitute customer orders silently.
Keep the full eligible population as the support denominator, including singleton orders.
Exclude cancelled lines consistently before both pair counts and marginal counts.
Set a minimum pair count before ranking lift, because rare coincidences yield unstable ratios.
Check that the two items can actually share a zone and that proximity does not cause congestion.
Use a holdout period to see whether the association persists, and inspect seasonal changes.
If picking is by zone, proximity across zones may not help either picker.
Avoid double counting an affinity benefit already captured by route-based travel estimates.

### Step 5 — Build the feasible assignment set

Input: location dimensions, load ratings, zone restrictions, SKU dimensions, handling rules.
Output: an eligibility matrix for each candidate SKU-location pair.
Check width, depth, height, orientation, access clearance, and safe load independently.
Translate capacity to the same unit as forecast consumption and account for dunnage or gaps.
Apply temperature, lot, expiry, hazardous-material, food-contact, and segregation rules as applicable.
Confirm the face supports the actual handling unit and picker equipment.
A calculated volume fit does not prove that a package can pass through the opening.
Review ergonomic reach and lift requirements with safety and frontline personnel.
Reserve empty or shared locations needed for seasonal demand and operating flexibility.
If no feasible face remains, flag the SKU for review rather than force an illegal assignment.
Hard constraints are never offset by a large calculated travel saving.

### Step 6 — Estimate work for each candidate

Input: forecast visits, units, route data, capacity, inventory policy, and cost rates.
Output: comparable recurring labor and service estimates for each feasible assignment.
At an early screening stage use visits × incremental minutes per visit as a travel proxy.
The incremental factor must be calibrated against observed routes for the actual pick method.
With multi-stop tours, summing depot round trips per item overstates travel.
A shared aisle or stop can also make a simple sum miss congestion and pair interactions.
Compute replenishments from demand, opening stock, capacity, and the actual trigger rule.
Count only replenishment work changed by the slotting decision, not unrelated reserve movements.
Multiply tasks by measured minutes per task, including any relevant setup or travel.
Convert total minutes to hours before multiplying by a consistent loaded labor rate.
Do not monetize both travel minutes and the same productivity improvement as separate benefits.
Track unpriced service, safety, and capacity impacts as explicit decision criteria.

### Step 7 — Compare assignments, uncertainty, and move cost

Input: feasible candidate layouts, recurring work, one-time cost, and scenario weights if justified.
Output: a recommended layout and a documented no-change comparison.
Use identical forecast periods, rates, and workload for every layout.
Add one-time move labor, labels, system configuration, verification, and downtime if incremental.
Net recurring savings equal old recurring cost minus new recurring cost, including replenishment.
Payback equals one-time cost divided by positive recurring savings per period.
If savings are zero or negative, report no simple payback rather than a negative payback.
For a fixed horizon, subtract the one-time cost from cumulative recurring savings.
For long horizons or material investment, use discounted cash flow and explicit residual assumptions.
Run low-demand, high-demand, and capacity-stress cases; report which choice changes.
Treat a modeled savings range as uncertainty, not as a guaranteed reduction in headcount.

### Step 8 — Execute, measure, and refresh

Input: approved assignment, inventory balances, move window, and authorized staff.
Output: verified location records and a monitored operational pilot.
Freeze or sequence concurrent work so that stock is not picked from an ambiguous location.
Count and identify stock, move it, scan the new location, and reconcile the WMS record.
Confirm labels, lot and expiry data, quantities, reserve linkage, and replenishment rules.
Perform an independent physical check before reopening the affected face.
Brief operators on the new path and provide a direct issue-escalation channel.
Keep the previous mapping and define when and how an authorized rollback occurs.
Compare pilot and control periods at similar demand, order mix, and staffing levels.
Review replenishment labor, stockouts, errors, travel, throughput, congestion, and safety together.
Refresh when demand, assortment, layout, constraints, or the pick method changes materially.
Avoid churn: require a benefit threshold above move cost and measurement uncertainty.
### Decision situations: what changes the answer

The following field cases are decision checks, not universal rules.
Each one identifies the observation, the mechanism, evidence to request, and the decision implication.
#### 1. Frozen versus ambient

- Observation: A chilled item cannot use an ambient face.
- Mechanism: Temperature eligibility precedes travel ranking.
- Evidence: Confirm the approved temperature range and actual zone monitoring.
- Decision: A short walk is not compensation for product exposure.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 2. Overweight case

- Observation: A case exceeds a shelf load rating.
- Mechanism: A location can fit dimensionally but fail structurally.
- Evidence: Verify package mass, pallet configuration, and rated load.
- Decision: Assign a rated lower location or retain the current position.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 3. Oversized carton

- Observation: The case volume fits but its longest side does not.
- Mechanism: Cube is necessary but not sufficient for access.
- Evidence: Compare each orientation against opening and internal dimensions.
- Decision: Reject the face when no permitted orientation fits.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 4. High unit velocity

- Observation: The item sells many eaches on few lines.
- Mechanism: Many units do not imply many picker stops.
- Evidence: Compare line, unit, and visit rates over the same period.
- Decision: Size the face for consumption before awarding the closest slot.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 5. Low unit velocity

- Observation: Many orders request one small unit.
- Mechanism: Frequent stops can dominate replenishment demand.
- Evidence: Measure route stops and typical units per stop.
- Decision: A small nearby face may be worthwhile if allowed.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 6. Case and each pick

- Observation: One SKU appears in two handling forms.
- Mechanism: The effective SKU-location decision includes handling unit.
- Evidence: Separate case picks from each picks and preserve conversion history.
- Decision: Score each pick method against its own locations.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 7. Multiple pick faces

- Observation: A fast SKU has two forward faces.
- Mechanism: SKU-level totals alone cannot explain location-level work.
- Evidence: Link each visit to the face active at the time.
- Decision: Evaluate split allocation and replenishment independently.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 8. Frequent short picks

- Observation: Tasks are often shorted at the face.
- Mechanism: Completed picks understate desired work.
- Evidence: Compare ordered, allocated, picked, and shorted quantities.
- Decision: Include availability scenarios rather than rank only successes.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 9. Cancelled demand

- Observation: An order is cancelled after allocation.
- Mechanism: Raw order lines can overstate physical workload.
- Evidence: Identify cancellations and reversals with event timestamps.
- Decision: Count actual activity and forecast future demand separately.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 10. Duplicate task events

- Observation: A task emits creation, confirmation, and correction messages.
- Mechanism: Event count differs from completed task count.
- Evidence: Deduplicate by task identity and event semantics.
- Decision: Use final valid task state and retain the audit trail.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 11. Location history

- Observation: A SKU moved during the lookback window.
- Mechanism: Joining to today’s location misattributes past travel.
- Evidence: Use effective-dated SKU-location assignments.
- Decision: Split historical activity at the move timestamp.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 12. Pack conversion

- Observation: A supplier changes case quantity.
- Mechanism: Static conversion distorts capacity and unit demand.
- Evidence: Apply the conversion valid on the transaction date.
- Decision: Recalculate usable face capacity in current units.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 13. Intermittent demand

- Observation: One project order dominates a quiet item.
- Mechanism: A mean hides demand clustering and uncertainty.
- Evidence: Inspect weekly counts and the underlying customer event.
- Decision: Use scenario forecasts and avoid irreversible moves on one spike.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 14. Peak season

- Observation: A seasonal item outranks staples for a few weeks.
- Mechanism: A full-year average hides a short high-load window.
- Evidence: Compute period-specific velocity and duration.
- Decision: Compare temporary slots against two moves and peak benefit.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 15. New SKU

- Observation: An item has little historical activity.
- Mechanism: Zero recorded picks does not mean zero future demand.
- Evidence: Use launch plan, analogous products, and forecast uncertainty.
- Decision: Assign a provisional eligible face with an early review.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 16. Discontinued SKU

- Observation: A fast historical item is being retired.
- Mechanism: Past velocity may overstate future visits.
- Evidence: Obtain confirmed end date and remaining inventory.
- Decision: Limit the horizon and avoid moves unlikely to pay back.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 17. Replenishment trigger

- Observation: Stock is topped up at a minimum level.
- Mechanism: Capacity division alone misses policy timing.
- Evidence: Model opening balance, threshold, order size, and schedule.
- Decision: Simulate the actual rule or clearly label the approximation.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 18. Partial replenishment

- Observation: Reserve cannot supply a complete face.
- Mechanism: Nominal capacity overstates available transfer quantity.
- Evidence: Use reserve balance and the actual task size distribution.
- Decision: Model additional trips and stockout risk when supply is partial.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 19. Empty face at opening

- Observation: A daily calculation assumes a full face.
- Mechanism: Opening inventory affects the number of refills.
- Evidence: Record beginning balance before calculating tasks.
- Decision: Use the correct inventory balance or a full simulation.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 20. Shared replenishment trip

- Observation: One operator refills several nearby faces.
- Mechanism: Multiplying a full travel time by every task overstates cost.
- Evidence: Measure shared-trip time and incremental handling time.
- Decision: Allocate common travel once and compare complete tours.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 21. Replenishment interference

- Observation: Refills block the same aisle used by pickers.
- Mechanism: Lower replenishment count can have nonlinear service effects.
- Evidence: Observe time-of-day overlap and aisle occupancy.
- Decision: Move work to a safe window or include interference in the pilot.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 22. Zone picking

- Observation: Items are picked by separate zone operators.
- Mechanism: Customer-order affinity may not create a common route.
- Evidence: Map each pair to the actual zone and handoff policy.
- Decision: Use within-zone batches to estimate travel interaction.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 23. Batch picking

- Observation: Several customer orders share one location visit.
- Mechanism: Order-line count overstates stop count.
- Evidence: Reconstruct batches and deduplicate item visits within each.
- Decision: Use actual visit counts for the batch policy.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 24. Cluster cart

- Observation: A picker handles multiple orders with compartments.
- Mechanism: Picking travel and sorting effort change together.
- Evidence: Record stops, cart capacity, and sorting time per tour.
- Decision: Compare total route and sorting labor, not distance alone.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 25. Single-line orders

- Observation: Some orders contain only one SKU.
- Mechanism: Excluding singleton orders inflates pair support.
- Evidence: Include all eligible orders in the support denominator.
- Decision: Report pair count and denominator with every affinity value.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 26. Rare affinity

- Observation: Two uncommon items appear together twice.
- Mechanism: High lift from tiny counts can be unstable.
- Evidence: Set a minimum count and compare a later holdout period.
- Decision: Use the pair as a hypothesis rather than a move instruction.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 27. Popular item affinity

- Observation: A popular item co-occurs with many items.
- Mechanism: High confidence may merely reflect high marginal popularity.
- Evidence: Compute support, both directional confidences, and lift.
- Decision: Prioritize incremental route savings over raw pair counts.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 28. Heavy-item pairing

- Observation: Two heavy products are frequently ordered together.
- Mechanism: Adjacency can compromise safe lifting or cart stability.
- Evidence: Consult safe handling rules and observe actual cart loading.
- Decision: Preserve handling sequence and keep unsafe locations ineligible.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 29. Forward capacity

- Observation: A face has attractive travel but little storage.
- Mechanism: Small capacity can cause frequent reserve work.
- Evidence: Convert safe usable cube to units and model replenishments.
- Decision: Choose the minimum total-cost feasible face.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 30. Reserve proximity

- Observation: Forward location changes but reserve stays fixed.
- Mechanism: Replenishment travel changes with both endpoints.
- Evidence: Measure reserve-to-face routes and equipment requirements.
- Decision: Evaluate travel for replenishment as well as picking.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 31. Congested golden zone

- Observation: Many A items compete for the nearest aisle.
- Mechanism: Adding people can increase queueing and obstruction.
- Evidence: Measure aisle occupancy and delay by interval.
- Decision: Spread fast items across suitable nearby aisles if needed.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 32. Accessibility

- Observation: A location is close but requires an awkward reach.
- Mechanism: Distance is not a complete measure of physical workload.
- Evidence: Review lift height, mass, reach, and task frequency.
- Decision: Apply the site ergonomic policy as a constraint.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 33. Lot control

- Observation: The SKU has expiry and lot restrictions.
- Mechanism: Combining lots can break rotation and traceability.
- Evidence: Verify lot identity and expiration in both systems and labels.
- Decision: Preserve lot segregation and required rotation during moves.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 34. Hazard segregation

- Observation: Two products have incompatible storage requirements.
- Mechanism: Affinity cannot override a segregation rule.
- Evidence: Review current applicable rules and site-specific compatibility.
- Decision: Exclude the combination and obtain qualified approval.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 35. Inventory accuracy

- Observation: System stock differs from physical stock.
- Mechanism: A move can propagate a discrepancy into the new face.
- Evidence: Count and reconcile before authorizing the transfer.
- Decision: Resolve variance under inventory-control procedure.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 36. Stale location status

- Observation: A face appears free but is reserved or blocked.
- Mechanism: The location master may lag real operations.
- Evidence: Check live status, physical inspection, and pending tasks.
- Decision: Do not assign until the reservation or block is resolved.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 37. Partial implementation

- Observation: Only some proposed moves are completed.
- Mechanism: A portfolio benefit may depend on interacting assignments.
- Evidence: Record executed versus proposed mapping and cutover time.
- Decision: Evaluate the layout actually in use, not the approved plan.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 38. Staffing conversion

- Observation: Travel hours decrease but schedules remain unchanged.
- Mechanism: Engineered hours do not automatically become cash savings.
- Evidence: Check labor contracts, fixed staffing, overtime, and redeployment.
- Decision: Separate capacity release from realizable payroll reduction.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 39. Simultaneous promotion

- Observation: A promotion starts during the pilot.
- Mechanism: Mix changes can masquerade as a layout effect.
- Evidence: Track orders, lines, units, item mix, and promotion dates.
- Decision: Use comparable control periods or matched zones.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
#### 40. Rollback

- Observation: A moved face produces unexpected errors.
- Mechanism: Benefits cannot justify uncontrolled inventory confusion.
- Evidence: Retain the old mapping, inventory trace, and stop authority.
- Decision: Suspend tasks and reverse only through verified system moves.
- Owner: the slotting analyst documents the issue and the operating owner accepts the disposition.
- Recheck: confirm the disposition after any material change to the relevant inputs.
## 6. Worked example

### Decision and population

A fictional fulfillment center is considering swapping two SKUs between two existing faces.
The forecast is for 20 operating days in one month of 2026.
The forecast is a planning estimate, not a customer commitment or actual observed volume.
Both items fit both locations, pass the site's safety checks, and have reserve stock available.
Both locations are in the same picking zone and have identical handling equipment.
No affinity adjustment is included in this two-item comparison.
The current picking method makes the stated per-visit travel proxy reasonable for screening.
A route study is still required before committing a staffing reduction.

| Item | Forecast pick visits/month | Forecast units/month | Current face | Proposed face |
|---|---:|---:|---|---|
| A | 1,200 | 2,400 | Far | Near |
| B | 400 | 800 | Near | Far |

| Face | Incremental travel | Usable capacity | Replenishment task time |
|---|---:|---:|---:|
| Near | 0.5 min/visit | 100 units | 6 min/task |
| Far | 1.5 min/visit | 400 units | 6 min/task |

Assume each face starts full and can be replenished instantly when empty.
All monthly quantities divide evenly by the relevant capacity, and there is no end-of-month top-up.
Under those special assumptions, replenishments equal `max(0, ceil(demand/capacity) - 1)`.
This is an illustrative finite-period inventory calculation, not a general replenishment formula.
The labor rate is USD 24 per productive labor-hour, including the relevant incremental burden.
The swap costs USD 240 once, including verified physical moves, labels, and system work.
No demand growth, discounting, downtime, or other incremental effect is assumed.

### ABC from visits

The two-item total is 1,200 + 400 = 1,600 visits per month.
A contributes 1,200 / 1,600 = 75% of visits; its prior cumulative share is 0%.
B contributes 400 / 1,600 = 25%; its prior cumulative share is 75%.
With the prior-share 80% rule, both are A: B crosses the threshold and stays in A.
This small example demonstrates why an ABC label is not a fixed percentage of SKUs.
The decision should use actual visit counts and cost, not a claim that all A items must be near.

### Current layout

A at Far produces 1,200 × 1.5 = 1,800 travel minutes per month.
B at Near produces 400 × 0.5 = 200 travel minutes per month.
Total current travel is 2,000 minutes, or 33.333 hours per month.
A replenishments: `ceil(2,400 / 400) - 1 = 5` tasks per month.
B replenishments: `ceil(800 / 100) - 1 = 7` tasks per month.
Current replenishment is 12 × 6 = 72 minutes per month.
Total current work is 2,072 minutes, or 34.533 hours per month.
At USD 24/hour, current modeled labor cost is USD 828.80 per month.

### Proposed layout

A at Near produces 1,200 × 0.5 = 600 travel minutes per month.
B at Far produces 400 × 1.5 = 600 travel minutes per month.
Proposed travel is 1,200 minutes, or 20 hours per month.
A replenishments: `ceil(2,400 / 100) - 1 = 23` tasks per month.
B replenishments: `ceil(800 / 400) - 1 = 1` task per month.
Proposed replenishment is 24 × 6 = 144 minutes per month.
Total proposed work is 1,344 minutes, or 22.4 hours per month.
At USD 24/hour, proposed modeled labor cost is USD 537.60 per month.

### Decision and sensitivity

Travel decreases by 800 minutes per month.
Replenishment increases by 72 minutes per month.
Net modeled labor decreases by 728 minutes, or 12.133 hours per month.
Net recurring modeled value is USD 291.20 per month at the stated rate.
Simple payback is USD 240 / USD 291.20 per month = 0.824 months.
Over three identical months, modeled net value after the move is USD 633.60.
That calculation is 3 × USD 291.20 − USD 240, before discounting.
At 20 operating days/month, the payback corresponds to about 16.5 operating days.
If released hours cannot reduce paid hours or avoid overtime, USD 291.20 is capacity value,
not necessarily cash savings; the benefit must be presented accordingly.

Suppose replenishment instead takes 30 minutes per task in both faces.
The difference in replenishments is 12 tasks per month, adding 360 minutes per month.
Net savings then become 800 − 360 = 440 minutes, or USD 176 per month.
Payback becomes USD 240 / USD 176 per month = 1.364 months.
The layout still wins in this scenario, but the economics are materially weaker.
At 800 / 12 = 66.667 minutes per extra task, the modeled recurring savings are zero.
Above that time, this simplified comparison favors the current layout.
Real replenishment timing, congestion, and shared travel can change all of these estimates.

### Affinity side calculation

Suppose 1,000 eligible orders include 200 with A, 100 with B, and 60 with both.
Each order is counted once even if it contains multiple lines for the same SKU.
Support(A,B) = 60 / 1,000 = 6% of eligible orders.
Confidence(A→B) = 60 / 200 = 30% of A-containing orders.
Confidence(B→A) = 60 / 100 = 60% of B-containing orders.
Lift = 0.30 / 0.10 = 3.0, equivalently 0.60 / 0.20 = 3.0.
These figures are descriptive and do not supply the missing travel benefit.
Confirm common tours and feasible locations before considering an affinity-based move.
## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Save it as `slotting_example.py` and run `python slotting_example.py`.
It reproduces the worked comparison and the pair statistics.
All input demand and task times are forecasts for one illustrative month.
The replenishment calculation assumes full opening faces, instant refills at empty,
no final top-up, and a constant task time; it does not model partial tasks or queues.
The model is a comparison, not an optimization of all feasible assignments.

```python
from math import ceil

items = {
    "A": {"visits": 1200, "units": 2400},
    "B": {"visits": 400, "units": 800},
}
faces = {
    "near": {"travel_minutes": 0.5, "capacity": 100},
    "far": {"travel_minutes": 1.5, "capacity": 400},
}
current = {"A": "far", "B": "near"}
proposed = {"A": "near", "B": "far"}
rate_per_hour = 24.0
replenishment_minutes = 6.0
move_cost = 240.0


def assess(layout):
    travel = 0.0
    refills = 0
    for sku, face_name in layout.items():
        item = items[sku]
        face = faces[face_name]
        travel += item["visits"] * face["travel_minutes"]
        refills += max(0, ceil(item["units"] / face["capacity"]) - 1)
    refill_time = refills * replenishment_minutes
    labor_cost = (travel + refill_time) / 60 * rate_per_hour
    return travel, refills, refill_time, labor_cost


def show(label, result):
    travel, refills, refill_time, cost = result
    print(f"{label}: travel={travel:.0f} min, "
          f"refills={refills}, refill={refill_time:.0f} min, "
          f"cost=USD {cost:.2f}/month")


old = assess(current)
new = assess(proposed)
show("Current", old)
show("Proposed", new)
savings = old[3] - new[3]
print(f"Savings: USD {savings:.2f}/month")
if savings > 0:
    print(f"Payback: {move_cost / savings:.3f} months")
else:
    print("Payback: none at nonpositive recurring savings")

orders, a_orders, b_orders, both = 1000, 200, 100, 60
support = both / orders
confidence_ab = both / a_orders
confidence_ba = both / b_orders
lift = confidence_ab / (b_orders / orders)
print(f"Support={support:.3f}; confidence A->B={confidence_ab:.3f}; "
      f"confidence B->A={confidence_ba:.3f}; lift={lift:.3f}")
```

Expected output:

```text
Current: travel=2000 min, refills=12, refill=72 min, cost=USD 828.80/month
Proposed: travel=1200 min, refills=24, refill=144 min, cost=USD 537.60/month
Savings: USD 291.20/month
Payback: 0.824 months
Support=0.060; confidence A->B=0.300; confidence B->A=0.600; lift=3.000
```

Before adapting this program to live data, define missing and zero-capacity handling.
Check location eligibility separately; the program assumes both layouts are feasible.
Replace the finite-period refill rule with a transaction-level simulation if needed.
Add route interactions only when calibrated evidence supports the additional terms.
Represent currencies, time zones, and effective dates consistently in the input contract.
Review the output with the warehouse supervisor before creating any move task.

## 8. Data, standards, and regulatory context

### Minimum data contract

| Dataset | Grain | Important fields | Reconciliation |
|---|---|---|---|
| Order lines | Order × line × revision | SKU, requested units, status, order time | Ordered and cancelled totals |
| Pick tasks | Task × event | picker or device, SKU, face, quantity, timestamp, status | Completed tasks and shipped units |
| Location master | Location × effective period | dimensions, load, zone, restrictions, status | Physical inspection and occupancy |
| Slot history | SKU × location × effective period | start, end, assignment type | No overlapping unintended assignments |
| Inventory | SKU × lot × location × snapshot | available, held, allocated, quantity | Inventory control balances |
| Replenishment | Task × event | source, destination, quantity, timestamps | Source and destination movements |
| Labor | Shift × role or work assignment | paid time, task time, breaks, rate | Payroll and timekeeping totals |
| Orders for affinity | Order × distinct SKU | order identifier, eligible status | Distinct eligible order count |
| Safety rules | Rule × applicability period | hazard, zone, handling constraints, owner | Current approved site procedures |

A warehouse management system (WMS) records tasks and locations, but its configuration
and event semantics determine whether a task is a physical visit or an administrative record.
Confirm whether timestamps represent creation, assignment, start, completion, or receipt.
A task may be reassigned; use the completion event for completed-work measurement.
A shared device identifier is not necessarily a person identifier.
Limit access to identifiable worker-level data and apply the organization's retention policy.
These privacy practices are governance choices; applicable legal duties vary by jurisdiction.

### Identifiers and measurement

GS1 publishes standards for identifying trade items and locations.
A Global Trade Item Number (GTIN) can identify a trade item at a defined packaging level.
A Global Location Number (GLN) can identify a location or party in an agreed context.
Neither identifier by itself describes a facility's slot coordinates, safe load, or availability.
Internal location codes still need a governed mapping to physical positions.
Do not assume one GTIN equals one internal SKU without checking packaging and master data.
Use a consistent system of units and document conversions between eaches, cases, and pallets.

### Safety and legal boundaries

In the United States, OSHA's materials-handling rule at 29 CFR 1910.176 includes
requirements concerning safe storage and aisles; consult the official text and applicability.
The regulation is binding where applicable; this educational lesson is not legal advice.
The relevant duties depend on the product, facility, jurisdiction, and task.
Do not infer that one federal rule covers all hazardous substances, fire codes, or food rules.
For any affected facility, qualified personnel must identify applicable requirements and
translate them into site-specific location eligibility and operating procedures.
A standard, supplier guideline, or internal ABC cutoff is not automatically a legal mandate.
Verify current texts and site conditions when decisions are made.

### Data-quality controls

Record the source system, extraction timestamp, period, and timezone for every dataset.
Measure unmatched SKU and location records both by row count and by activity volume.
A low unmatched row percentage can still hide a large share of workload.
Test for duplicate task IDs, overlapping effective dates, negative net balances, and impossible capacity.
Investigate whether missing location events cluster on a shift, device, or work type.
A location may be physically usable but system-blocked; both states matter to execution.
Compare planned task duration with measured elapsed and active time before assigning rates.
Document whether breaks, travel outside the zone, indirect work, and waiting are included.
Keep personal information out of analytical extracts when aggregated measures suffice.

## 9. Industry practice and case evidence

### What the literature supports

De Koster, Le-Duc, and Roodbergen's 2007 review describes warehouse order-picking
design and control, including storage assignment and routing, as interdependent decisions.
The review synthesizes research; it is not evidence of a universal percentage saving.
Its relevance here is the need to evaluate slotting alongside picking and routing choices.

Petersen and Aase's 2004 simulation study compares routing, storage, and picking policies.
A simulated policy comparison depends on the study's layout, order patterns, and assumptions.
It is evidence that policies interact, not proof that any policy wins in every building.
Read the model and experimental settings before applying the results elsewhere.

Kiva Systems' mobile-robot approach, described by Wurman, D'Andrea, and Mountz,
brings inventory pods to workstations rather than requiring a human picker to visit every face.
That is a different travel and capacity system from the manual-forward-pick example above.
[INFERENCE] A slotting objective for such a system must account for pod access,
robot travel, workstation workload, and inventory placement rather than reuse walking distance.
The cited publication documents the system concept; it does not validate this lesson's example.

### How to use external cases

Identify the facility, date, operational process, intervention, and measured outcome.
Separate a named company's reported result from a causal estimate with a control group.
Ask whether the case involved storage assignment, order batching, automation, or all three.
Check whether the stated denominator is orders, lines, units, paid hours, or productive hours.
A case reported as a percentage improvement cannot be applied to another facility without
checking starting performance, product mix, technology, and operational constraints.
A vendor claim can suggest a hypothesis but is not independent validation of the benefit.
For a local investment, measured pilot evidence is more actionable than a borrowed benchmark.

### Implementation patterns

A common operating pattern is to use velocity as a candidate-generation screen.
The analyst then filters candidates for physical and operational eligibility.
Supervisors inspect proposed changes because the system record may miss physical obstacles.
Inventory control executes moves through traceable tasks rather than physical relocation alone.
The site monitors realized effects and changes its assumptions when the pilot disagrees.
These are recommended steps for this lesson, not a claim about every company's practice.
## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Unit velocity used as visits | Large case orders dominate rank | Units are not stops | Near slots allocated to infrequent work | Measure physical visits |
| Partial-pick censoring | High shorts but low completed picks | Missing stock suppresses observed activity | Under-slot constrained items | Reconcile requested and shorted work |
| Duplicate event joins | Activity exceeds shipped totals | Multiple events multiply a task | Inflated benefits | Deduplicate by event semantics |
| Stale assignment history | Historical work appears at new location | Current master overwrites history | Misstated baseline | Effective-date the join |
| Capacity based only on cube | Packages do not fit opening | Shape and clearance ignored | Unusable face | Validate orientation and access |
| No opening-stock assumption | Refill totals disagree by one or more | Period boundaries omitted | Incorrect task comparison | Model opening, closing, trigger |
| Refill time omitted | Pick travel improves but reserve labor rises | Work shifts between roles | Net labor increases | Include incremental replenishment |
| Pair-count denominator wrong | Support changes when singleton orders are added | Eligible population excluded | Inflated affinity | Include all eligible orders |
| Rare-pair overfit | High lift on only two orders | Small sample noise | Unnecessary moves | Minimum count and holdout |
| Ignored pick method | Pair affinity high but separate zones pick items | Items do not share a tour | No route improvement | Match affinity to work grouping |
| Double-counted route saving | Affinity and travel both claim the same distance | Overlapping benefit terms | Inflated return | Use a single route cost model |
| Congestion omitted | Near aisle has queue or blocked access | Concentrated demand competes for space | Delays and safety risk | Observe and cap occupancy |
| Unsafe assignment | Layout favors a prohibited shelf | Hard rule treated as penalty | Injury or noncompliance | Enforce eligibility before scoring |
| Cash confused with capacity | Paid hours remain unchanged | Time savings not schedulable | Financial overstatement | Separate cash and capacity |
| Simultaneous process changes | Layout and batching change together | Treatment is confounded | Attribution fails | Isolate changes or use controls |
| Move executed only physically | Scans fail or inventory appears missing | System and physical states diverge | Errors and stockouts | Verify both states at cutover |
| Seasonal churn | Same SKU moves repeatedly | Short-term noise drives ranking | Move cost exceeds benefit | Hysteresis and longer scenarios |
| Poor rollback plan | Staff cannot identify previous stock position | Old mapping or lot trail missing | Prolonged disruption | Preserve mapping and stop authority |
| Wrong rate denominator | Productive hours valued at paid-hour rate without adjustment | Labor burden counted inconsistently | Mispriced benefit | State rate and hour basis |
| Lack of ownership | Unresolved discrepancy remains after pilot | Nobody can approve correction | Permanent degraded operation | Assign named operational authority |

### Investigation sequence

Start with the baseline population and confirm that the activity totals reconcile.
Next validate the physical constraints and capacity of each affected face.
Then compare pick travel, replenishment, and move costs on the same period.
Finally investigate omitted interactions: congestion, batching, service, and safety.
If the system and physical inventory disagree, halt the move until the variance is resolved.
A clean spreadsheet is not a substitute for a safe and executable task sequence.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test or response |
|---|---|---|---|
| Demand forecast applies to horizon | Compares future assignments | Promotion, launch, or demand shock | Compare low, base, and high scenarios |
| Visits predict travel work | Connects activity to distance | Batching or shared tours | Reconstruct tours and measure routes |
| Incremental travel is additive | Makes screening tractable | Shared aisle segments and route interaction | Validate against sampled paths |
| Faces start full | Determines finite-period refills | Opening face partially empty | Use observed opening stock |
| Refills occur instantly at empty | Makes the example calculable | Delays, minimum triggers, or queues | Simulate actual trigger and schedule |
| Refill task time is constant | Converts task count to minutes | Distance, equipment, or shared trips | Measure time by source and destination |
| Both locations are eligible | Allows a direct swap | Dimension, load, or segregation failure | Physical and safety review |
| All demand can be supplied | Avoids reserve stockouts | Upstream shortage or holds | Check inventory availability and lead time |
| The rate reflects incremental cost | Monetizes modeled hours | Fixed staffing or mixed labor categories | Separate avoidable from capacity value |
| Move cost is complete | Enables payback | Downtime, labels, or checks omitted | Obtain a verified execution estimate |
| No material congestion change | Avoids additional queue term | Concentration in narrow aisle | Observe occupancy and delay by interval |
| No other assignment changes | Isolates the two-SKU comparison | Competing SKU displaced | Evaluate the full feasible portfolio |
| Stable item identity | Makes historical aggregation meaningful | Pack conversion or SKU substitution | Maintain effective-dated identity mapping |
| Forecast quantities are divisible | Simplifies worked arithmetic | Partial final face and variable ending balance | Use ceiling and period-end balance explicitly |
| No discounting | Supports simple short payback | Long investment horizon or capital cost | Use discounted cash flow when material |

These assumptions describe the example, not universal warehouse properties.
A single violated assumption does not automatically invalidate every use of a model.
It does require a sensitivity analysis, a revised method, or a narrower decision claim.

## 12. Exercises

### Exercise 1 — Define the grain

**Time:** 12 minutes.
A batch contains three customer orders, each requesting SKU X once.
The picker visits X's face once and takes six units in total.
A separate replenishment operator later refills the face once.
**Deliverable:** report order lines, picker visits, picked units, and replenishment tasks.
**Acceptance:** identify all four counts and explain why none is interchangeable.

### Exercise 2 — Compute ABC

**Time:** 18 minutes.
Four SKUs have 50, 30, 15, and 5 visits per week, respectively, in descending order.
Use the prior-cumulative-share convention: A below 80%, B from 80% to below 95%, C otherwise.
**Deliverable:** a table of share, prior cumulative share, ending cumulative share, and class.
**Acceptance:** show boundary treatment and reconcile to 100 visits per week.

### Exercise 3 — Screen eligibility

**Time:** 20 minutes.
A case measures 60 × 40 × 30 cm and weighs 18 kg.
A shelf opening is 55 × 45 cm, usable depth is 65 cm, and rated load is 15 kg.
Assume the case may be rotated and only one case is proposed.
**Deliverable:** decide whether to assign the case and identify missing operating checks.
**Acceptance:** assess fit and mass independently and reject any load-rating violation.

### Exercise 4 — Compare layouts

**Time:** 25 minutes.
Use the two-item example and assume 10 minutes per replenishment task rather than six.
Keep all other assumptions, the USD 24/hour rate, and USD 240 move cost unchanged.
**Deliverable:** compute each layout's monthly travel, replenishment time, labor cost, savings,
and simple payback.
**Acceptance:** retain the correct 12-task difference and show units for each result.

### Exercise 5 — Interpret affinity

**Time:** 15 minutes.
Among 500 eligible orders, 80 contain X, 50 contain Y, and 20 contain both.
**Deliverable:** compute support, confidence in both directions, and lift.
**Acceptance:** use the full order denominator for support and explain one limit of the result.

### Exercise 6 — Evaluate the investment

**Time:** 15 minutes.
A feasible move costs USD 900 once and is forecast to save 50 labor-hours/month.
At USD 25/hour, only 40% of released time is expected to reduce paid labor.
Assume this realization rate stays constant for six months and there are no other changes.
**Deliverable:** calculate monthly modeled capacity value, expected cash saving, simple cash
payback, and undiscounted six-month net cash benefit.
**Acceptance:** distinguish capacity from cash and apply the move cost exactly once.

### Exercise 7 — Design a pilot

**Time:** 30 minutes.
A warehouse proposes moving 20 fast SKUs across three aisles before peak.
Some are bulky and replenishment is shared with a neighboring zone.
**Deliverable:** a one-page pilot plan specifying baseline, comparison, eligibility,
measurements, move sequence, stop rules, rollback, and decision owner.
**Acceptance:** protect safety and traceability, account for both zones, and avoid claiming
causation solely from an uncontrolled before-and-after comparison.

### Exercise 8 — Diagnose the ranking

**Time:** 20 minutes.
A completed-picks report ranks an item C after it was out of stock for half the month.
An order-line feed shows numerous shorted requests, while a batch log has missing IDs.
**Deliverable:** identify the biases, data checks, and a provisional decision.
**Acceptance:** explain censored demand, visit-grain uncertainty, and the need for an
eligible temporary assignment rather than inventing a precise visit count.

## 13. Self-check answers

### Answer 1

The three orders contain three order lines for X.
The batch has one recorded picker visit to X, with six units picked.
The later refill is one replenishment task, not another customer pick visit.
A single visit can satisfy multiple order lines, and one line can contain multiple units.
Different work measures serve different labor and capacity calculations.

### Answer 2

| SKU rank | Visits/week | Share | Prior cumulative | Ending cumulative | Class |
|---|---:|---:|---:|---:|---|
| 1 | 50 | 50% | 0% | 50% | A |
| 2 | 30 | 30% | 50% | 80% | A |
| 3 | 15 | 15% | 80% | 95% | B |
| 4 | 5 | 5% | 95% | 100% | C |

The denominator is 100 visits/week.
The third item's prior share equals 80%, so it is B rather than A.
The fourth item's prior share equals 95%, so it is C rather than B.
This convention is explicit; another convention may assign boundary items differently.

### Answer 3

A 40 × 30 cm orientation fits the 55 × 45 cm opening, with 60 cm depth within 65 cm.
The case weighs 18 kg, exceeding the shelf's 15 kg rated load.
The proposed assignment is therefore ineligible despite the dimensional fit.
Confirm access clearance, orientation constraints, handling method, and the actual shelf rating.
Do not average ratings across shelves or assume a safety factor is available for normal use.

### Answer 4

The current layout retains 2,000 travel minutes and 12 refill tasks per month.
At 10 minutes/task, current replenishment is 120 minutes and total work is 2,120 minutes.
Current modeled labor cost is 2,120 / 60 × USD 24 = USD 848 per month.
The proposed layout retains 1,200 travel minutes and 24 refill tasks per month.
Its replenishment is 240 minutes and total work is 1,440 minutes per month.
Proposed modeled labor cost is 1,440 / 60 × USD 24 = USD 576 per month.
Savings are USD 272 per month; simple payback is USD 240 / USD 272 = 0.882 months.
These are modeled values, not automatically realized cash savings.

### Answer 5

Support is 20 / 500 = 4% of eligible orders.
Confidence X→Y is 20 / 80 = 25% of X-containing orders.
Confidence Y→X is 20 / 50 = 40% of Y-containing orders.
Y prevalence is 50 / 500 = 10%, so lift is 0.25 / 0.10 = 2.5.
The association does not establish a travel saving or demonstrate a causal relationship.
Actual work grouping, legal proximity, and route length still need examination.

### Answer 6

Modeled capacity value is 50 hours/month × USD 25/hour = USD 1,250/month.
Expected cash saving is 40% × USD 1,250 = USD 500/month.
Cash payback is USD 900 / USD 500 per month = 1.8 months.
Six-month undiscounted net cash benefit is 6 × USD 500 − USD 900 = USD 2,100.
The remaining USD 750/month is not assumed to be cash savings.

### Answer 7 — Rubric

A strong plan defines the affected SKU-location population and a stable comparison group.
It records eligible faces and signed safety review before moving any stock.
It measures visits, route time, refill work in both zones, errors, and stockouts.
It specifies comparable operating intervals and changes in order mix.
It identifies who may stop work, the numerical or qualitative guardrail, and escalation.
It preserves the prior mapping and verifies inventory before and after a controlled rollback.
It reports cash and capacity separately, and states the limitations of causal attribution.

### Answer 8 — Rubric

A strong answer recognizes that stockouts censor completed picks relative to demand.
It reconciles shorted requests, cancellations, substitutions, and availability by period.
It checks whether missing batch IDs systematically affect particular shifts or devices.
It does not equate order lines with visits while batch grouping is unknown.
It assigns a provisional safe face and schedules a review after better evidence is available.
## 14. Interview questions

### 1. What does ABC classify?

A strong answer identifies the ranked measure, population, period, and cumulative cutoff rule.
It explains why a class is a screening label rather than an assignment decision.
A weak answer treats A as a universal fixed fraction of SKUs or uses revenue without context.
Ask the candidate how their classification handles a SKU crossing the 80% boundary.

### 2. When are units a poor proxy for travel?

A strong answer distinguishes units, lines, visits, and tours with a batch-picking example.
It connects units to stock consumption and visits to picker work.
A weak answer assumes each unit creates a separate physical stop.
Ask what source system establishes the actual visit grain.

### 3. How would you model a forward face?

A strong answer includes usable capacity, opening inventory, demand, trigger, and refill time.
It distinguishes an illustrative ceiling calculation from an actual replenishment policy.
A weak answer divides monthly units by gross volume and stops.
Ask how the answer changes if the face starts partially empty.

### 4. What does affinity tell you?

A strong answer defines support, directional confidence, lift, and the relevant order population.
It checks whether the items are picked on a common tour and whether the pair is stable.
A weak answer infers causal travel savings from lift alone.
Ask why singleton orders must remain in the support denominator.

### 5. How do you handle hard constraints?

A strong answer filters infeasible assignments before comparing costs.
It includes dimensions, load, access, applicable safety restrictions, and system status.
A weak answer permits a safety violation if the calculated travel saving is large enough.
Ask who approves an ambiguous restriction and what happens until the review is complete.

### 6. Why might a near face be a bad choice?

A strong answer considers replenishment frequency, reserve travel, congestion, and ergonomics.
It evaluates the alternative use of the scarce face by another SKU.
A weak answer assumes distance to pack is the only relevant variable.
Ask for a numerical example in which replenishment outweighs travel savings.

### 7. How do you reconstruct a baseline?

A strong answer uses effective-dated location assignments and valid completed task events.
It reconciles quantities and checks missing scans and stockout-censored demand.
A weak answer joins historic picks to today's location and counts every event as a task.
Ask how a reassigned task with several status messages is counted.

### 8. How do you calculate return?

A strong answer compares identical time horizons, includes incremental moves and refills,
and distinguishes modeled capacity hours from genuinely avoidable paid labor.
A weak answer multiplies all engineered hours by a wage and calls the result guaranteed cash.
Ask what simple payback means when recurring savings are zero.

### 9. What would you measure in a pilot?

A strong answer includes route time, visits, refill tasks, errors, stockouts, safety, and mix.
It uses a comparable baseline or control and reports remaining confounding.
A weak answer measures only lines per hour after changing both layout and batching.
Ask how neighboring zones might absorb work shifted by the pilot.

### 10. How would you roll back a failed change?

A strong answer names a stop authority, a retained old mapping, and verified inventory moves.
It reconciles physical and system state before work resumes.
A weak answer tells staff to move the boxes back without updating location records.
Ask how lot and expiry attributes remain traceable during the reversal.

### 11. How frequently should slots be refreshed?

A strong answer balances forecast changes and recurring benefits against move cost and noise.
It uses scenarios or a minimum benefit threshold to avoid unnecessary churn.
A weak answer prescribes a universal weekly schedule without considering workload or seasonality.
Ask how the decision differs for a short promotion and a permanent assortment change.

### 12. What changes in a goods-to-person facility?

A strong answer considers pod access, robot and workstation workloads, and storage constraints.
It recognizes that a human walking-distance proxy may no longer represent system travel.
A weak answer assumes the nearest manual shelf rule transfers unchanged to mobile pods.
Ask what evidence is needed to calibrate the alternative objective.

## 15. Key terms

Each term points to the section that develops its operational meaning.

- [ABC classification](#5-the-mechanics-step-by-step) — cumulative activity segmentation.
- [Affinity](#3-the-core-mental-model) — co-occurrence in a defined order or batch population.
- [Baseline](#8-data-standards-and-regulatory-context) — reconciled reference for comparison.
- [Batch](#5-the-mechanics-step-by-step) — orders grouped for a picking tour.
- [Capacity](#4-definitions-and-vocabulary) — safe usable quantity in a face.
- [Cash benefit](#6-worked-example) — avoidable expenditure, not merely released time.
- [Confidence](#6-worked-example) — conditional co-occurrence proportion.
- [Congestion](#10-failure-modes-and-red-flags) — interference at shared space.
- [Counterfactual](#4-definitions-and-vocabulary) — outcome estimated without a change.
- [Cutoff](#5-the-mechanics-step-by-step) — class boundary and crossing convention.
- [Demand censoring](#10-failure-modes-and-red-flags) — lost observation when stock is unavailable.
- [Eligibility](#5-the-mechanics-step-by-step) — hard conditions for an assignment.
- [Face](#4-definitions-and-vocabulary) — location from which stock is picked.
- [Forecast](#6-worked-example) — uncertain future demand estimate.
- [Guardrail](#10-failure-modes-and-red-flags) — condition that prompts intervention.
- [Lift](#6-worked-example) — confidence relative to marginal prevalence.
- [Location history](#8-data-standards-and-regulatory-context) — effective-dated mapping of stock positions.
- [Move cost](#6-worked-example) — one-time incremental cost of a change.
- [Order line](#4-definitions-and-vocabulary) — item row in a customer order.
- [Payback](#6-worked-example) — move cost divided by positive recurring savings.
- [Pick task](#8-data-standards-and-regulatory-context) — recorded unit of assigned or completed work.
- [Replenishment](#5-the-mechanics-step-by-step) — transfer from reserve to forward stock.
- [Reserve](#4-definitions-and-vocabulary) — stock outside the forward face.
- [Rollback](#5-the-mechanics-step-by-step) — verified reversal of an implemented assignment.
- [Route proxy](#3-the-core-mental-model) — simplified estimate of incremental travel.
- [SKU](#4-definitions-and-vocabulary) — stockkeeping unit under inventory control.
- [Slotting](#1-why-this-topic-exists) — assignment of stock to eligible locations.
- [Stockout](#4-definitions-and-vocabulary) — unavailable forward stock when required.
- [Support](#6-worked-example) — pair count divided by all eligible orders.
- [Tour](#4-definitions-and-vocabulary) — ordered sequence of picking stops.
- [Velocity](#5-the-mechanics-step-by-step) — activity rate in a named unit and period.
- [Visit](#3-the-core-mental-model) — physical stop at a location.
- [WMS](#8-data-standards-and-regulatory-context) — warehouse management system.
- [Zone picking](#5-the-mechanics-step-by-step) — allocation of picks to physical zones.

## 16. Sources

1. De Koster, R., Le-Duc, T., and Roodbergen, K. J. (2007).
   “Design and control of warehouse order picking: A literature review.”
   *European Journal of Operational Research*, 182(2), 481–501.
   [https://doi.org/10.1016/j.ejor.2006.07.009](https://doi.org/10.1016/j.ejor.2006.07.009).
   Peer-reviewed review; useful for the interaction among warehouse design decisions.
2. Petersen, C. G. and Aase, G. (2004).
   “A comparison of picking, storage, and routing policies in manual order picking.”
   *International Journal of Production Economics*, 92(1), 11–19.
   [https://doi.org/10.1016/j.ijpe.2003.09.006](https://doi.org/10.1016/j.ijpe.2003.09.006).
   Simulation study; results depend on its modeled facility and assumptions.
3. Wurman, P. R., D'Andrea, R., and Mountz, M. (2008).
   “Coordinating Hundreds of Cooperative, Autonomous Vehicles in Warehouses.”
   *AI Magazine*, 29(1), 9–20.
   [https://doi.org/10.1609/aimag.v29i1.2082](https://doi.org/10.1609/aimag.v29i1.2082).
   Describes the Kiva system; not evidence for the numerical case in this lesson.
4. GS1, [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Official identification and barcode standards; check the current version and applicability.
5. U.S. Occupational Safety and Health Administration,
   [29 CFR 1910.176, Handling materials—general](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.176).
   Official U.S. regulation; consult the current text and qualified safety personnel.

The worked numbers are fictional instructional assumptions, not results from these sources.
External evidence does not replace a site-specific assessment of safety or performance.

## 17. Links

- [Module 13 index](_index.md): topic sequence and module scope.
- [Warehouse flow system](01_warehouse-flow-system.md): flow, buffers, and downstream effects.
- [WMS data model](02_wms-data-model.md): event and location records.
- [Productivity metrics](03_productivity-metrics.md): rate denominators and comparability.
- [Module 13 survey](../../../course/13_warehousing_fulfillment_labor.md): module overview and lab material.
