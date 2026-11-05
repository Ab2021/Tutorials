# Batching and Routing for Quick Commerce

> Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics
> Topic 7 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 1, 5, and 6
> Version 2026.5

## 1. Why this topic exists

A store receives three orders within two minutes.
Each customer has a delivery commitment.
A picker can visit an aisle once for several orders, and a courier can deliver several nearby orders in one trip.
Those apparent efficiencies do not necessarily belong to the same order grouping.

If the store waits to form a larger pick batch, the first customer spends part of the promise waiting.
If the picker collects several baskets on one route, a wrong tote can send the correct item to the wrong household.
If the dispatcher combines two deliveries, the second customer receives the order after an extra stop.
A courier assigned before the bags are ready may wait unpaid or occupy scarce supply.
A faster picker route can therefore make the whole service slower.

The operational decision is not simply “combine nearby orders.”
It is which compatible orders to collect together, when to release them, which courier to use, in what delivery sequence, and when to abandon the batch.
The decision must use the information available at that moment, not information revealed afterward.

A shift lead needs a release rule that staff can execute under pressure.
A dispatch controller needs a way to distinguish a feasible planned route from an optimistic line on a map.
A product owner needs to understand the customer risk of waiting for another order.
A courier needs realistic travel and handoff time, safe loads, and clear pay.
A food-safety owner needs temperature and exposure limits that remain binding even when batching looks profitable.

The expensive error is not merely a late delivery.
It includes lost customer trust, compensation, cold-chain exposure, picker congestion, courier waiting, and the opportunity cost of tying up a vehicle.
The apparent benefit can also be overstated when the counterfactual assumes that two couriers would otherwise be idle and instantly available.

This topic treats batching as a controlled, rolling decision rather than a one-time route optimization.
It separates physical feasibility, service risk, and economics.
It shows a small reproducible example before introducing broader operating controls.

### Decisions and ownership

| Decision | Accountable owner | Inputs | Output |
|---|---|---|---|
| Set promise and service guardrails | Operations and product leads | Customer commitments, capacity, observed lateness | Policy limits by service tier |
| Release a pick wave | Store shift lead or approved automation | Ready orders, picker capacity, due times | Pick batch and release time |
| Sequence aisle visits | Store operations | Item locations, precedence, congestion | Executable pick path |
| Assign delivery batch | Dispatch controller or approved automation | Courier state, packed orders, travel estimates | Courier and candidate route |
| Accept work and travel safely | Courier | Offer, load, route and pay information | Acceptance, rejection, or safety escalation |
| Set food handling limits | Food-safety owner | Product type, packaging and local rules | Hard compatibility and exposure constraints |
| Change objective weights | Operations owner with analytics | Measured cost, service and distributional effects | Versioned, approved policy |
| Pause an unsafe or degraded policy | On-duty operations lead | Exceptions, telemetry, staff reports | Revert to safe dispatch rule |

Automation may recommend a decision; it does not remove ownership of the policy or the exception path.
A service-level target is not the same thing as an individual customer commitment.
A plan is a proposed execution, and an estimated arrival is a prediction rather than a guarantee.

## 2. Learning outcomes

After this unit you can:

1. **Distinguish** pick batching, delivery batching, and routing by their resources and failure modes.
   Exercise 1 checks the distinction.
2. **Calculate** travel, handoff, arrival, lateness, and incremental route cost for a small candidate set.
   Exercises 2 and 3 test the arithmetic.
3. **Screen** a candidate batch against time, capacity, compatibility, and sequence constraints.
   Exercise 4 requires an explicit feasibility record.
4. **Choose** between waiting and releasing using remaining slack, uncertainty, and an executable fallback.
   Exercise 5 asks for a release decision.
5. **Critique** an evaluation that reports average savings without protecting late-tail service or worker welfare.
   Exercise 6 tests this outcome.
6. **Design** a rolling operational control and a staged experiment with named owners and stop rules.
   Exercises 7 and 8 provide the design brief.
7. **Explain** the assumptions of a small executable routing calculation and why it is not a production optimizer.
   Exercise 3 asks you to modify and interpret it.

A satisfactory answer keeps minutes, distance, load, and currency distinct.
It also reports a rejected candidate when that candidate is attractive on distance but fails a hard constraint.

## 3. The core mental model

```text
Customer orders + commitments + basket facts
                    |
                    v
     Release decision: wait, pick alone, or pick together
                    |
                    v
   Pick route -> separate totes -> scan -> pack and stage
                    |
                    v
   Available couriers + compatible packed orders + travel estimates
                    |
                    v
      Enumerate route candidates and screen hard constraints
                    |
                    v
      Score feasible routes -> offer -> accept -> depart
                    |
                    v
       Stop 1 -> handoff -> stop 2 -> handoff -> completion
                    |
                    v
    Actual events -> service, cost, safety, and fairness feedback
                    |
                    +----> replan when reality changes
```

There are two linked but distinct consolidation problems.
A **pick batch** shares movement inside the store.
A **delivery batch** shares courier movement outside the store.
The first can contain orders that should never share a delivery route; the second can combine orders picked separately.
Treating the groups as identical hides important constraints.

A route is a sequence, not a set.
For orders A and B, A-then-B and B-then-A have the same members but different arrival times, exposure durations, and lateness.
The last customer pays for previous travel and handoffs.
The first customer may also pay if the store waits to find a partner.

Time is a budget.
At decision time, remaining slack equals the commitment time minus the current time minus the remaining estimated work.
Remaining work includes picking, packing, courier arrival or readiness, loading, road travel, access, and handoff.
Use conservative estimates or explicit uncertainty for high-stakes decisions; a point estimate alone gives no probability of success.

Capacity is multidimensional.
A courier may have enough mass capacity but insufficient insulated volume.
A picker may have enough tote positions but too many temperature zones to maintain safe separation.
A delivery sequence may meet deadlines but violate a maximum exposure time.
These are feasibility conditions, not merely extra costs.

Batching saves duplicated setup, walking, and travel only when the shared resource would otherwise repeat that work.
It adds waiting, detours, sorting, and coordination.
The correct comparison is with a feasible alternative available at the same decision time, including its opportunity cost.
It is not with a fictional ideal solo trip.

Uncertainty changes the decision.
A route that reaches a customer one second before a promise under average traffic is not robust.
Traffic, picking exceptions, building access, and customer availability are correlated in some operating conditions.
When a scenario changes, recompute the entire timeline rather than adding independent percentile times without justification.

Control is continuous.
A candidate can be feasible when offered but infeasible when the courier accepts late or an item is missing.
An operator needs a cutoff for waiting, a recheck before dispatch, and a rule to split, reorder, substitute, or notify when reality changes.
Changing the route after acceptance can affect pay and fairness; the applicable agreement and local rules govern what is allowed.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Order | Customer request with order identifier and line items | Treating multiple lines as separate orders |
| Order line | Requested product and quantity | Assuming one line equals one physical unit |
| Pick batch | Orders collected on one picker tour | Equating it to a delivery batch |
| Pick wave | Orders released together for picking | Assuming simultaneous release guarantees a common route |
| Delivery batch | Orders assigned to a shared courier trip | Assuming all members were picked together |
| Route | Ordered sequence of travel and service events | Comparing unordered sets only |
| Stop | Physical service location in a route | Assuming one stop always means one order |
| Aisle walking | Picker movement between stock locations, usually metres or minutes | Using straight-line shelf distance as actual path |
| Pick time | Time to locate, scan and collect items, minutes | Omitting item handling and exception time |
| Sort time | Time to separate and verify order contents, minutes | Counting it as free when picking in bulk |
| Pack time | Time to pack and label an order, minutes | Assuming all orders finish when the first item is picked |
| Stage | Place prepared goods in a controlled handoff area | Treating staged perishable goods as indefinitely safe |
| Commitment | Delivery time or window communicated under the applicable customer promise | Treating a model estimate as a promise |
| ETA | Estimated time of arrival, a forecast at a stated prediction time | Treating a point estimate as guaranteed |
| Deadline | Time used for feasibility screening, in a named time zone | Confusing promised end with an internal target |
| Slack | Deadline less current time and remaining work, minutes | Omitting waiting and handoff time |
| Lateness | Positive part of completion time minus deadline, minutes | Averaging early arrivals as negative lateness |
| Tardiness | Another term for nonnegative lateness, minutes | Confusing lateness with total elapsed time |
| Detour | Added route time or distance relative to a defined comparator | Comparing against an infeasible baseline |
| Dwell | Time spent stationary at a location, minutes | Attributing all dwell to a courier |
| Handoff | Transfer of goods at a stop, including access and confirmation | Treating arrival at building as completed delivery |
| Capacity | Limit on load, volume, tote positions, or other resource | Using mass as a proxy for every constraint |
| Compatibility | Whether orders may share a tote, compartment, route, or vehicle | Assuming geographical proximity implies compatibility |
| Cold-chain exposure | Time and conditions outside specified temperature control | Equating route duration with actual product temperature |
| Service level | Share meeting a precisely defined commitment | Averaging rates without order counts |
| Late-tail metric | A statistic describing severe lateness, such as p95 positive delay | Reporting a percentile without defining its population |
| Opportunity cost | Value of a resource's best feasible alternative use | Treating the accounting wage as the full marginal cost |
| Rolling horizon | Repeatedly solve with current information and updated state | Assuming plans remain valid until route completion |
| Reoptimization | Revising a plan after a trigger or time interval | Reassigning accepted work without operational rules |
| Hard constraint | Condition candidates must meet to be eligible | Assigning a small penalty to an unsafe action |
| Soft objective | Quantity traded off within the feasible set | Using it to override safety |
| Baseline | Executable comparator under the same conditions | Comparing with an idealized or historical policy alone |
| Shadow mode | Generate recommendations without executing them | Claiming simulated savings are realized savings |
| Assignment | Matching work to a specific resource | Assuming a suggested offer was accepted |
| Censoring | Missing completion or observation beyond the measurement window | Treating open orders as on-time |

## 5. The mechanics, step by step

### Step 1 — Define the decision and clock

**Input:** current orders, resource states, promise definitions, and a single decision timestamp.

**Action:** choose whether this decision concerns pick release, courier assignment, route resequencing, or an exception.
Use one clock and a named time zone; retain the original customer commitment and any internal planning deadline separately.
An internal deadline may precede the customer deadline to reserve uncertainty buffer.
Do not silently move the customer's commitment when a route becomes inconvenient.

**Output:** a decision record containing the eligible order population and the decision time.

**Branch:** if an order is cancelled, held for verification, or lacks a required fact, remove it from normal eligibility and use the appropriate exception queue.
A missing promise is not permission to deliver indefinitely late.

### Step 2 — Reconstruct remaining work

**Input:** event log, item locations, picker progress, pack status, courier availability, travel predictions, and stop-access data.

For an unpicked order, calculate remaining pick and pack time.
For a picked order, use actual completion and any outstanding packing or verification.
For an assigned courier, distinguish time to the store, loading, departure, driving, and customer handoff.
For each component, distinguish observed time from forecast time.
Avoid double counting travel to the store when the courier is already present.

A simplified expected completion for the first customer is:

```text
completion_1 = decision_time
             + wait_for_batch + remaining_pick_pack
             + wait_for_courier + loading
             + travel_store_to_1 + handoff_1
```

This sum applies only if the components occur sequentially.
When picking and courier approach run in parallel, readiness is the later of their completion times, not their sum.
If loading cannot begin until both are ready, use `max(packed_at, courier_at_store) + load_time`.

**Output:** a time-stamped state and a remaining-work estimate per order and resource.

**Branch:** if an event is stale, use a conservative fallback or ask for an operational update rather than presenting it as current.

### Step 3 — Set hard eligibility rules

**Input:** service tier, item properties, load dimensions, worker capability, food-handling rules, and operating policy.

Reject a candidate if it exceeds a physical vehicle or tote limit.
Check both mass and volume; add compartment limits where relevant.
Enforce incompatible products and required separation.
Respect promised time windows, access restrictions, and any maximum allowed product exposure under the applicable handling plan.
Do not interpret a model-estimated arrival as a food-temperature measurement.

Protect worker safety: a distance-saving route is not valid if it requires unsafe riding, prohibited access, or a load beyond the courier's safe capacity.
Apply any required break, working-time, or offer rules as constraints determined with local experts.
This educational policy does not determine employment status or legal compliance.

**Output:** eligible order-resource pairs and explicit rejection reasons.

**Branch:** when no candidate survives, route to a single-order or manual escalation path if feasible; otherwise notify the accountable operator and customer according to policy.
Never force an infeasible batch merely to clear a queue.

### Step 4 — Form pick-batch candidates

**Input:** eligible unpicked orders, item locations, available pickers, cart/tote limits, and release deadlines.

Group potential orders with overlapping aisles or locations.
A good candidate reduces repeated aisle entry without delaying an urgent order beyond its buffer.
Measure walking on the real traversable aisle network, including one-way restrictions, stairs, and congestion.
Shelf-coordinate Euclidean distance is a lower bound at best, not a route a person can necessarily walk.

Retain each order's identity in a separate tote or an equivalent controlled sorting process.
Estimate incremental sort and verification work.
Account for duplicate items, large or fragile items, and temperature zones.
A picker may visit ambient aisles first and chilled locations later if the site's food-safety procedure requires it.

**Output:** candidate pick sets, proposed paths, estimated walking and handling time, and order-level ready times.

**Branch:** if adding the next order saves walking but exceeds the first order's release cutoff or tote capacity, release the smaller batch.
If one item is missing, follow substitution or shortage policy rather than holding all other orders indefinitely.

### Step 5 — Choose a pick path

**Input:** pick set, store graph, current picker location, pick locations, and any precedence rules.

Sequence visits to minimize a suitable measure of work while respecting aisle direction and temperature handling.
For a tiny store, a fixed aisle sweep can be more reliable than a brittle optimizer.
For a larger site, compare a shortest-path or traveling-salesperson heuristic against the existing route.
Use observed walking time as well as distance: a shorter route through a congested aisle can be slower.

Do not optimize walking distance in isolation.
A route that visits a slow-to-find item early may expose cold goods unnecessarily; a route that defers an urgent customer's final item can delay the entire delivery.
Where several pickers share an aisle, congestion depends on the simultaneous plan, not only on one path.

**Output:** an executable path and expected order-level completion times.

**Branch:** if a picker reports a blockage or unavailable item, refresh location and remaining work and either replan or apply a simple fallback route.

### Step 6 — Decide whether to wait for another order

**Input:** current candidates, forecast arrivals, oldest order's remaining slack, and incremental benefit of a larger batch.

An arrival forecast is uncertain; a future compatible order is not yet a customer commitment in hand.
The release policy should set a maximum wait and a minimum remaining buffer.
Compute the oldest order's latest release from its deadline minus conservative remaining work.
The chosen wait cannot exceed this latest release or any shorter safety or policy limit.

Waiting may be justified when the expected saved work exceeds the incremental service and resource cost and hard constraints remain satisfied.
That comparison must include the possibility that no compatible order arrives.
If no defensible estimate of that probability exists, use a bounded rule and measure its outcomes rather than inventing precision.

**Output:** release now or wait until a stated timestamp, with an owner and trigger.

**Branch:** release immediately when the timer expires, an urgent order loses its buffer, or upstream conditions deteriorate.
Do not reset the oldest order's timer every time another order arrives.

### Step 7 — Build delivery candidates

**Input:** packed or predictably ready orders, couriers, vehicle capabilities, travel-time matrix, and stop details.

Start with single-order alternatives and add small compatible combinations.
For each set, enumerate feasible stop sequences where the set is small enough.
For larger sets, use a time-limited heuristic with a known feasible fallback.
The travel matrix must include direction, travel mode, decision time, and any important access restrictions.
A bicycle travel estimate is not interchangeable with a car estimate.

For each candidate, calculate departure from the later of order readiness and courier availability, plus loading.
At each stop add travel and handoff before proceeding to the next stop.
Keep route completion separate from each order's delivery completion.
The final return to store may matter for courier availability and economics even though it does not affect customers already served.

**Output:** ordered candidates with event timelines, load profiles, and capacity checks.

**Branch:** if an order is not ready at the expected time, recompute before offering or dispatching.
If the courier is already on a route, only consider modifications that the operational policy, safety controls, and agreement allow.

### Step 8 — Calculate service feasibility and risk

**Input:** candidate timelines, order commitments, travel and handoff uncertainty, and service buffers.

For each order, calculate `lateness = max(0, completion - commitment)`.
A strict no-lateness screen rejects candidates with positive predicted lateness.
If policy allows an at-risk decision when all options are late, record that exception explicitly and choose according to an approved escalation rule.
Do not label an already-late route feasible under a no-lateness rule.

For uncertainty, use empirical residuals or scenarios from comparable historical situations, constructed without future information.
Simulate shared delays jointly when traffic or store congestion affects multiple route legs.
Compare the share of scenarios in which each order meets its deadline with the policy threshold.
If sample size is small or distributions have shifted, show the uncertainty rather than claiming a calibrated probability.

**Output:** per-order predicted completion, buffer, lateness, and scenario risk.

**Branch:** if point-estimate feasible but buffer insufficient, release earlier, split, choose another courier, or escalate.
A generous average cannot compensate for an unacceptable last-stop risk.

### Step 9 — Compare economics within the feasible set

**Input:** feasible route alternatives, incremental picking effort, courier pay, vehicle cost, compensation, and resource availability.

A transparent objective may combine incremental labour minutes, travel distance, and expected late cost.
Convert each term to a common unit only when the conversion is defensible.
If a policy weight is a preference rather than a measured cost, label it as a weight.
Show absolute units beside the composite score so stakeholders can understand the trade-off.

Use marginal rather than total cost where possible.
If a courier would otherwise wait at the store, the relevant comparison differs from a courier who would deliver another order.
Include the return or repositioning leg when it changes the next available time.
Compare pick and delivery savings separately to avoid crediting the same saved minute twice.

**Output:** ranked feasible options, the chosen comparator, and sensitivity to key weights.

**Branch:** if ranking changes under plausible assumptions, present the choice as uncertain and apply the approved service-first policy.
Never hide a capacity, safety, or legally required restriction inside a tunable penalty.

### Step 10 — Offer, confirm, and execute

**Input:** selected route, courier availability, predicted readiness, offer terms, and execution policy.

Send an offer with the information required by the operating agreement and local rules.
A recommendation is not an assignment until the relevant acceptance or assignment event occurs.
Record offer time, response, expiration, and reason where available.
If the offer is rejected or expires, recompute using the new time and current resource state.

At packing, scan each order into its correct bag and each bag into the correct delivery batch.
At pickup, confirm the courier receives all expected bags and any required temperature-controlled container.
At each stop, confirm the correct order and record a completed handoff using the approved proof process.
Protect customer data; show only the information required to deliver.

**Output:** an accepted and traceable execution plan, with a documented fallback.

**Branch:** if one order is not ready or a courier reports an unsafe load, do not pressure the courier to accept it.
Split, wait within the permitted buffer, or escalate according to policy.

### Step 11 — Monitor and replan

**Input:** live status, missed scans, new orders, travel changes, and exception reports.

Monitor actual against planned timestamps and the earliest remaining customer deadline.
Set explicit triggers: late pack, courier no-show, route delay beyond a buffer, failed handoff, temperature excursion, cancellation, or system outage.
A trigger should name the action and owner, not merely change a dashboard color.

For a stalled first stop, recalculate the effect on all subsequent customers.
Depending on policy and feasibility, contact the customer, resequence, split to another courier, or return goods safely.
Do not promise a new time that operations cannot support.
Respect the courier's agreement when changing the sequence or adding work.

**Output:** a revised plan or documented exception, plus customer and worker communications where needed.

**Branch:** if routing data are unavailable, use a preapproved conservative manual procedure with a cap on batch size and clear escalation.
Reconcile manual decisions once systems recover.

### Step 12 — Learn and change policy

**Input:** completed and open orders, rejected candidates, actual times, costs, complaints, and incident records.

Compare intended and executed routes.
Separate model prediction error from policy decisions and execution exceptions.
Track the share of eligible orders actually batched, not only the average batch size of those that were batched.
Report service by stop position, area, time of day, basket type, and promise tier where sample size permits.

Evaluate candidate policies against the same demand and resource constraints.
Use randomized assignment when feasible and safe; otherwise describe selection bias and confounding.
A shadow replay can reveal obvious infeasibility but cannot establish realized savings, because a different policy changes courier availability and subsequent decisions.

**Output:** a versioned policy review with service, safety, worker, and economic outcomes.

**Branch:** stop or revert if predeclared guardrails fail; investigate before increasing exposure.
A model retrain is not a substitute for correcting a bad process or missing scan.

### Operational design detail — picking configurations

The choice of a picking configuration determines what the system can safely assume about order separation.
The floor supervisor should document the physical workflow before setting batch-size limits.
A routing model cannot repair an ambiguous tote procedure.

**Single-order picking.**
One picker collects one customer's items before beginning another order.
The approach minimizes sorting complexity and makes ownership obvious.
Its disadvantage is repeated aisle travel when several customers need nearby items.
It is a useful fallback when scans, carts, or staff training are unavailable.
Its baseline must include the actual start and finish location, not an imagined teleport back to the packing bench.

**Multi-tote picking.**
One cart carries a dedicated tote for each order.
The picker visits a location once and places the required quantity into each corresponding tote.
A screen should identify the item, quantity, and destination tote at the point of placement.
The picker must be able to correct an accidental placement without losing the audit trail.
The practical batch limit may be the number of safe tote positions rather than the number of orders the software can display.
Bulky items can consume more than one slot and make the nominal limit misleading.

**Bulk pick followed by sort.**
The picker collects aggregated quantities and sorts by order at a separate station.
This can reduce handling at shelves but shifts effort and error risk to the sorting station.
It also introduces an additional queue and may delay individual order readiness.
A combined pick is not complete for delivery purposes until the individual order is sorted, verified, and packed.
Compare end-to-end completion and error rates with multi-tote picking, not only shelf walking.

**Zone picking.**
Different staff collect different categories or store zones and then consolidate each customer's goods.
The slowest required zone can determine readiness for the whole order.
Parallel zone times therefore combine through a maximum, followed by consolidation, rather than a simple sum.
Additional staffing may reduce order elapsed time while increasing total worker-minutes.
If refrigerated items are collected in a separate zone, the staging location and timing must follow the site's approved handling procedure.

**Pick-to-order versus pick-to-stock.**
This unit concerns fulfilling identified customer orders from store inventory.
Replenishing a forward pick face is a different activity and has its own capacity and inventory controls.
Do not claim a batch-picking saving by shifting replenishment work outside the measurement window.
Record aisle interruptions caused by replenishment because they can change actual walking speed.

### Operational design detail — the walking network

Model the store as traversable nodes and edges when a map is needed.
Nodes can represent intersections, shelf access points, staging locations, and packing benches.
Edges represent legal walking paths and carry distance or expected travel time.
One-way aisles, temporary obstructions, and staff-only doors change the feasible graph.
Location coordinates should be versioned with the effective date of a layout change.

Shelf adjacency alone is insufficient.
Two shelves may be physically close but separated by a wall or a one-way aisle.
An aisle may have more than one entrance, and a shortest path can depend on the picker's current position.
For chilled or frozen goods, the relevant path may be constrained by the order in which product classes can be collected.
If the layout changes, stale location coordinates can produce a route that staff cannot follow.

Collect observed durations from enough comparable tours to see whether route length predicts time.
Use distance plus handling and exception time rather than converting every metre at a single universal speed.
A heavy cart, a crowded promotion, or a blocked aisle can slow the same path.
Picker experience and shift conditions can also affect observed durations.
Do not infer individual worker performance from route time without considering those conditions and the intended use of the data.

Validate a path by walking it with store staff.
Ask whether each shelf can be approached from the proposed side and whether the cart can turn safely.
Compare the suggested sequence with established safety and food-handling procedures.
If a shortest path conflicts with those procedures, the path is infeasible.
Document the discrepancy and update the graph or constraints.

### Operational design detail — incremental pick value

For a proposed order added to a pick batch, separate four quantities.
First, estimate the additional walking needed to reach its item locations.
Second, estimate its item handling, scans, and tote placement.
Third, estimate incremental sorting, verification, and packing complexity.
Fourth, estimate any waiting imposed on orders already in the batch.

The baseline is the work needed to pick that same order under the current alternative.
If it would otherwise be picked by another available picker, the comparison differs from placing it behind a long queue for the same picker.
When labour is already committed to another task, moving that labour changes other work as well.
The pick decision should not count a saving that is offset by a larger packing bottleneck.

Consider an order with 5 item lines, 3 of which lie on an existing route.
If the other 2 lines require an additional aisle, the true added walk includes entering and leaving that aisle.
It also includes handling all 5 lines, not only the 2 new locations.
If two units of the same item are needed for different customers, verify each allocation separately.
Treat a substituted product as a changed line requiring the site's approval and scan procedure.

Batch size should be justified by marginal rather than total benefit.
Moving from one to two orders may eliminate a repeated aisle tour.
Moving from two to three may save little walking while adding a third tote and more sorting.
An average saving across all batches can conceal a negative marginal value for larger batches.
Track incremental walking, waiting, sorting, and error rates by proposed size.

### Operational design detail — urgency and priority

First-in-first-out is understandable, but it can fail when promises and remaining work differ.
Earliest deadline first is also incomplete if one order needs much more work.
Least slack incorporates both the deadline and estimated remaining work.
It still depends on the quality of the remaining-work estimate.

Suppose A is due in 15 minutes and needs 12 minutes of remaining work.
Suppose B is due in 13 minutes and needs 6 minutes of remaining work.
A has 3 minutes of slack and B has 7 minutes of slack.
Earliest deadline would prioritize B, while least slack would prioritize A.
The example ignores variability and shared resources; it demonstrates the distinction rather than prescribing a universal rule.

An urgent order should not automatically be paired with a low-urgency order merely because their destinations are nearby.
Check whether the partner's readiness delays departure.
Check whether serving the partner first consumes the urgent order's remaining buffer.
If the urgent order must go first, test the second customer's deadline and product exposure separately.
If neither sequence is acceptable, use separate trips or escalate.

Avoid priority starvation.
Repeatedly giving new urgent orders precedence can leave older, less urgent orders waiting until they too become urgent.
Include age, release cutoffs, and an escalation queue in the policy.
Monitor waiting time distributions, not just the percentage of urgent orders released.
The policy owner should define how different service tiers are treated before the shift begins.

### Operational design detail — uncertainty and buffers

Separate a point estimate from a distribution of possible outcomes.
The median travel time can be useful for staffing, but a deadline decision also needs a view of adverse outcomes.
The relevant prediction is conditional on the current time, mode, location, and operating conditions.
Historical data from an unrelated mode or geography may not transfer.

A fixed buffer is easy to explain and audit.
It can be too conservative for a short, predictable trip and too small for a long, uncertain one.
A risk-based buffer can vary with route and context, provided the risk model is sufficiently calibrated.
The choice is a policy decision, not a purely mathematical result.
An operator should be able to fall back to fixed conservative limits if the model is unavailable.

Evaluate calibration by comparing predicted on-time probabilities with observed frequencies over suitable groups.
If routes assigned a 90% on-time probability meet their promises far less often, investigate bias and drift.
Report sample size and uncertainty around the observed frequency.
An aggregate calibration result can hide poor performance for particular areas or stop positions.
Where sample sizes are small, avoid false precision and consider pooling only when defensible.

Use scenarios that preserve important relationships.
Heavy rain may slow both courier approach and road travel.
A store backlog may delay several orders ready for the same trip.
An apartment access problem may affect all orders in the same building.
Randomly sampling each leg independently can understate the chance of a large shared delay.

Do not confuse a safety buffer with excess capacity.
The buffer protects a commitment against uncertainty in a particular execution.
Excess capacity provides resources to handle variation across the operation.
Removing all available capacity can make the nominal buffer unachievable when a courier is delayed.
Monitor both utilization and the ability to recover from disruptions.

### Operational design detail — route sequencing

A short total route can still be a poor service sequence.
The first stop's time affects every later stop.
The last stop's handoff does not affect earlier customers but does affect courier availability and product exposure.
The sequence should be evaluated against each customer's commitment, not only final route completion.

For two orders, compare both sequences explicitly.
Calculate store-to-first travel, first handoff, inter-stop travel, and second handoff.
Use directional travel times because one-way streets or access patterns can make A-to-B differ from B-to-A.
Then add the return or next-positioning leg for resource availability.
If either sequence fails a hard rule, exclude it even if its total travel is shorter.

For three or more orders, the number of possible sequences grows rapidly.
Screen obvious incompatibilities before enumerating sequences.
A geographically close order with a very late readiness time can invalidate an otherwise attractive route.
Use a maximum computation time and preserve the best known feasible option.
Record whether the result was exhaustive or heuristic so that a heuristic recommendation is not represented as a proven optimum.

An accepted trip is not a blank slate for continuous optimization.
Repeated sequence changes can create navigation confusion and unexpected worker effort.
Set a materiality threshold and an event trigger for changes.
For example, an urgent customer deadline or a road closure may justify reviewing the sequence; a negligible estimated saving may not.
The actual rules for changing an accepted offer depend on the operating agreement and applicable requirements.

### Operational design detail — loading and staged orders

Loading begins only when the courier and required orders are ready.
If A is packed at 12:03 and B at 12:07, the pair is not fully ready until 12:07.
If the courier arrives at 12:05, the courier waits 2 minutes for B before loading can begin.
If loading requires 2 minutes, departure is 12:09.
This calculation assumes all items can safely wait in their staging conditions and no other constraint is breached.

The same example has different implications for the two orders.
A has been packed for 6 minutes by departure; B has been packed for 2 minutes.
Those elapsed times do not by themselves establish product temperature.
Use the site's validated staging method and monitoring to determine whether the route remains permitted.
If B is delayed further, A may need a solo departure rather than indefinite waiting.

Record separately the time an order is packed, staged, scanned to the courier, and physically picked up.
A batch can be marked ready in software while one bag is still missing from the rack.
Conversely, an order can be physically ready while a scan has not yet synchronized.
An exception process should reconcile these states without allowing an unverified bag to leave.

The cost of courier waiting is not always the same as the store's cost of idle time.
Compensation rules and the worker's alternative opportunities matter.
Measure courier arrival-to-pickup time by store and by batch size.
Where a worker contests a timestamp, preserve both the system event and the correction process.
Use the information to fix readiness estimates rather than assigning blame from a single noisy event.

### Operational design detail — stopping and splitting

Define a batch-splitting authority before an exception occurs.
The automated system may recommend a split, while the dispatcher controls a manual override.
The courier's consent or other protections may be required under the applicable terms.
Food safety may require specific handling when an order is removed from a route.
Every split should preserve order identity and record who has physical custody of each bag.

Before pickup, splitting may mean sending the ready order alone and leaving the other in controlled staging.
Recalculate the waiting order's new pickup estimate and customer commitment risk.
Do not count the initial batching decision as executed if no shared trip occurred.
Record the reason, such as late pack, rejected offer, capacity issue, or changed travel estimate.

After pickup, splitting is operationally harder.
The first courier may need to return a bag or transfer it at an approved location.
The transfer itself adds time, handling, and custody risk.
An apparently available second courier may not be able to reach the transfer location safely or in time.
Model the actual transfer sequence before declaring it a feasible recovery.

If no recovery can meet the original promise, communicate according to the approved customer policy.
Do not erase the original promise from measurement.
Resolve cancellation, return, refund, and food disposition through the appropriate policies.
Afterward, distinguish a forecast error from a preventable operational failure.
Both may warrant action, but the corrective action will differ.

### Operational design detail — monetary comparison

Suppose a local analysis values incremental courier occupation at a planning rate of USD 0.40 per minute.
The worked example's 13-minute difference would imply USD 5.20 of illustrative resource value before other effects.
That is `13 courier-minutes × USD 0.40 per courier-minute`.
It is not a verified reduction in courier pay, a quote, or a universal labour rate.

Suppose additional sorting costs USD 0.30 per batch and expected customer compensation changes by USD 0.20 per batch.
If these were credible incremental estimates, the illustrative net would be `USD 5.20 - USD 0.30 - USD 0.20 = USD 4.70`.
This arithmetic does not include changes in courier availability, vehicle cost, lost demand, or food handling.
It also cannot establish that the batch meets service and safety constraints.
Hard constraints are screened before an economic comparison.

Avoid multiplying a total route duration by a rate if the comparator already includes some of that duration.
Avoid adding the same delay cost once as a service penalty and again as compensation unless both represent separate real effects.
Distinguish a transfer payment from a social cost when the analysis includes both platform and worker perspectives.
State the viewpoint: platform contribution, total operating resource cost, customer experience, or combined welfare.
Different viewpoints can rank the same policies differently.

### Operational design detail — evaluation and rollout

Start with a process baseline before changing the decision rule.
Document current batch eligibility, actual stop sizes, release waits, and manual overrides.
Measure the fraction of all eligible orders served in batches and the distribution of batch sizes.
A rising average among only batched orders can coexist with fewer orders being batched overall.
Keep denominators consistent across before and after periods.

In shadow mode, save the recommendation and all inputs available at the time.
Compare whether the recommendation would have passed known hard constraints.
Compare predicted outcomes with actual outcomes only where their interpretation is valid.
The actual courier path was generated by the existing policy, not the shadow policy.
Consequently the shadow system's predicted savings are not realized savings.

If randomization is appropriate, consider store-time clusters instead of individual orders when couriers are shared.
Individual-order treatment can spill into control orders by consuming the same limited courier pool.
Cluster assignment reduces some interference but can reduce statistical power.
The analysis plan should account for clustering and time variation.
Discuss design and sample-size requirements with an analyst before exposure begins.

Set guardrails before launching.
Examples include original-promise on-time rate, severe lateness, wrong-order handoffs, product handling incidents, and courier waiting.
Specify the population, measurement window, threshold, owner, and response for each guardrail.
A stop rule should be executable even if an economic improvement is statistically promising.
Protect against sparse data by including immediate incident escalation as well as aggregate thresholds.

After the pilot, report both the intention-to-treat population and execution details where appropriate.
Include rejected offers, failed assignments, cancellations, and late-arriving completion events.
Explain missing data and any deviations from the predeclared analysis.
Do not attribute differences to the policy if the study design cannot support that conclusion.
Retain a rollback procedure until the policy has demonstrated stable operation.

## 6. Worked example

### Setting and assumptions

At 12:00 local time, a dark store has two packed, compatible orders, A and B.
Both are ready for collection and can be carried in separate insulated compartments of one courier bag.
The customer commitment is completed handoff, not arrival at the building.
A has a 12:19 deadline and B has a 12:22 deadline.
The planning team also evaluates separate trips using two couriers who are actually available at 12:00.
This availability is an explicit scenario assumption, not a general rule.

All travel times below are deterministic estimates for the same travel mode and conditions.
Each courier needs 2 minutes to load at the store.
Each customer handoff takes 2 minutes.
Times are measured in minutes from 12:00.
The return to store is included in resource time, not in customer completion.

| Directed leg | Travel time | Interpretation |
|---|---:|---|
| Store to A | 7 min | First outbound option |
| Store to B | 8 min | Alternative outbound option |
| A to B | 4 min | Second delivery leg |
| B to A | 5 min | Reverse direction differs |
| A to store | 7 min | Repositioning after A |
| B to store | 8 min | Repositioning after B |

The bag limit is 12 kg and 40 L.
Order A occupies 4 kg and 12 L; order B occupies 5 kg and 14 L.
The combined load is 9 kg and 26 L, so it passes both capacity checks.
No product or compartment incompatibility exists under this hypothetical site's rules.
The site has independently confirmed that the illustrative route duration is within its validated handling limits; this example does not specify a universal food-safety threshold.

### Separate trips

Courier 1 loads A during minutes 0–2.
The courier travels 7 minutes and arrives at A at minute 9, or 12:09.
Handoff ends at minute 11, or 12:11.
A's lateness is `max(0, 11 - 19) = 0` minutes.
The courier returns in 7 minutes and is available at minute 18.

Courier 2 loads B during minutes 0–2.
The courier travels 8 minutes and arrives at B at minute 10, or 12:10.
Handoff ends at minute 12, or 12:12.
B's lateness is `max(0, 12 - 22) = 0` minutes.
The courier returns in 8 minutes and is available at minute 20.

The sum of courier occupied times is `18 + 20 = 38` courier-minutes.
This is not 38 minutes of elapsed time; the trips occur concurrently.
Both customer orders meet their commitments under the deterministic assumptions.

### Shared trip, A then B

One courier loads both orders during minutes 0–2.
The courier travels 7 minutes to A, arrives at minute 9, and completes A at minute 11.
The courier travels 4 minutes from A to B, arrives at minute 15, and completes B at minute 17.
The courier returns from B in 8 minutes, becoming available at minute 25.

A completes at 12:11, 8 minutes before its 12:19 deadline.
B completes at 12:17, 5 minutes before its 12:22 deadline.
Both have zero deterministic lateness.
The shared trip occupies 25 courier-minutes.
Compared with the two-courier baseline, it saves `38 - 25 = 13` courier-minutes.
It delays B's completed handoff by `17 - 12 = 5` minutes.
It does not delay A in this example.

### Shared trip, B then A

The courier loads during minutes 0–2, travels 8 minutes to B, and completes B at minute 12.
The courier travels 5 minutes from B to A, arrives at minute 17, and completes A at minute 19.
A finishes exactly at its 12:19 deadline; B finishes at 12:12.
The courier returns from A in 7 minutes and is available at minute 26.

This ordering is point-estimate feasible but leaves A no time buffer.
Its occupied time is 26 courier-minutes, one more than A-then-B.
A 3-minute delay at B would make A 3 minutes late if the delay propagates unchanged.
Under a 3-minute minimum planning buffer, B-then-A is rejected while A-then-B passes.
The 3-minute buffer is a hypothetical policy choice, not a statistical guarantee.

| Option | A completion | B completion | Minimum buffer | Courier occupied time | Deterministic late orders |
|---|---:|---:|---:|---:|---:|
| Separate trips | 12:11 | 12:12 | 8 min | 38 courier-min | 0 |
| A then B | 12:11 | 12:17 | 5 min | 25 courier-min | 0 |
| B then A | 12:19 | 12:12 | 0 min | 26 courier-min | 0 |

The minimum buffer in the first row is `min(19 - 11, 22 - 12) = 8` minutes.
For A-then-B it is `min(19 - 11, 22 - 17) = 5` minutes.
For B-then-A it is `min(19 - 19, 22 - 12) = 0` minutes.

### Picking extension

Suppose these orders had not yet been picked.
Separately, the two pick tours would require 180 m and 160 m of walking.
A shared tour would require 240 m because both orders use the same aisles.
At a planning walking speed of 60 m/min, separate walking requires `340 / 60 = 5.67` picker-minutes.
Shared walking requires `240 / 60 = 4.00` picker-minutes.
That is a walking saving of 1.67 picker-minutes before sorting.

Suppose shared picking adds 1.0 picker-minute of tote sorting and verification.
The net estimated picker saving is `1.67 - 1.00 = 0.67` picker-minutes.
Item handling is assumed identical in both alternatives and therefore cancels.
If waiting for the second order adds 3 minutes to A's release time, A's customer delay may be larger than this picker saving.
A labour saving and an individual customer's delay are not interchangeable units.

This extension is a separate counterfactual; the delivery calculations above start with both orders already packed at 12:00.
Do not add a fresh pick duration to those delivery times without recomputing both readiness and courier schedules.

### Decision and sensitivity

Choose A-then-B if the 3-minute buffer is the applicable policy, the courier accepts, and the assumptions still hold at dispatch.
This choice saves 13 courier-minutes against the stated two-courier alternative while preserving 5 minutes of minimum point-estimate buffer.
It also adds 5 minutes to B's handoff relative to a separate trip.
The owner should check whether the second courier could have served other work; that opportunity is not included in the simple occupied-time comparison.

If the A-to-B leg takes 7 minutes instead of 4, B completes at minute 20 and retains 2 minutes of buffer.
The route still meets the point-estimate deadline but fails the 3-minute planning buffer.
If B's handoff takes 6 minutes instead of 2, B completes at minute 21 and has 1 minute of buffer.
If loading starts 6 minutes late, A completes at minute 17 and B at minute 23; B is 1 minute late.
These are one-at-a-time scenarios, not probability estimates.

The correct operational response depends on when the delay becomes known.
Before dispatch, the controller can choose separate couriers if they remain available.
After leaving the store, the controller needs an approved exception plan and accurate customer communication.
The calculation does not prove that batching is superior across all orders or all days.

## 7. Code, query, or calculation

The following Python 3 example uses only the standard library.
Save it as `batch_example.py` outside the course file and run `python batch_example.py`.
It enumerates the two shared sequences and the two separate trips from the worked example.
Times are minutes from 12:00; capacity is kilograms and litres.
It is deliberately deterministic and does not optimize pick routes, model probability, or establish food safety.

```python
from itertools import permutations

LOAD_MIN = 2
HANDOFF_MIN = 2
MIN_BUFFER = 3
CAP_KG = 12
CAP_L = 40

orders = {
    "A": {"deadline": 19, "kg": 4, "litres": 12},
    "B": {"deadline": 22, "kg": 5, "litres": 14},
}
travel = {
    ("S", "A"): 7,
    ("S", "B"): 8,
    ("A", "B"): 4,
    ("B", "A"): 5,
    ("A", "S"): 7,
    ("B", "S"): 8,
}


def route(sequence):
    """Return a deterministic trip timeline, including return to store."""
    if sum(orders[o]["kg"] for o in sequence) > CAP_KG:
        return None
    if sum(orders[o]["litres"] for o in sequence) > CAP_L:
        return None
    clock = LOAD_MIN
    previous = "S"
    delivered = {}
    for order in sequence:
        clock += travel[(previous, order)]
        clock += HANDOFF_MIN
        delivered[order] = clock
        previous = order
    occupied = clock + travel[(previous, "S")]
    buffers = {o: orders[o]["deadline"] - delivered[o] for o in sequence}
    late = {o: max(0, -buffers[o]) for o in sequence}
    return {
        "delivered": delivered,
        "buffers": buffers,
        "late": late,
        "occupied": occupied,
        "buffer_feasible": min(buffers.values()) >= MIN_BUFFER,
    }


for sequence in permutations(orders):
    result = route(sequence)
    print(
        "-".join(sequence),
        "delivered=", result["delivered"],
        "min_buffer=", min(result["buffers"].values()),
        "occupied=", result["occupied"],
        "passes=", result["buffer_feasible"],
    )

solo = [route((order,)) for order in orders]
solo_occupied = sum(result["occupied"] for result in solo)
batched = route(("A", "B"))
print("solo_occupied=", solo_occupied)
print("A-B_saving=", solo_occupied - batched["occupied"])
```

Expected output, allowing for ordinary spacing around printed values:

```text
A-B delivered= {'A': 11, 'B': 17} min_buffer= 5 occupied= 25 passes= True
B-A delivered= {'B': 12, 'A': 19} min_buffer= 0 occupied= 26 passes= False
solo_occupied= 38
A-B_saving= 13
```

`passes` is the hypothetical minimum-buffer rule, not a prediction of actual on-time probability.
The code assumes all orders are ready at time zero, all relevant legs exist, and one 2-minute loading event serves the batch.
It assumes two simultaneous couriers are available for the separate-trip comparator.
It does not calculate courier compensation, verify compatibility, use traffic distributions, or model a return that ends elsewhere.
Add those only with valid data and an explicitly defined operating policy.

For a third order, the number of permutations grows to six; for `n` distinct stops it is `n!`.
Exhaustive enumeration is useful for a tiny example, not a guarantee of scalable real-time routing.
The production system also needs a time limit, valid input checks, a feasible fallback, and observable decision records.

## 8. Data, standards, and regulatory context

### Event data contract

| Field | Meaning | Quality check | Use |
|---|---|---|---|
| `order_id` | Stable identifier for the customer order | Unique within source and linked across systems | Join events without merging customers |
| `created_at` | Order creation time with offset or UTC | Present and not after completion | Queue and waiting time |
| `promised_by` | Original customer commitment end | Preserved despite later updates | Service measurement |
| `decision_at` | Timestamp of the routing decision | Not after information used in that decision | Leakage control |
| `pick_started_at` | Start of physical picking | Ordered relative to release | Picker work |
| `pick_completed_at` | Last required item picked or documented exception | Before pack completion | Readiness |
| `packed_at` | Order packed and ready for handoff | Valid state and scan | Dispatch eligibility |
| `courier_arrived_at` | Courier arrives at pickup location | Distinguish GPS estimate from verified event | Waiting attribution |
| `pickup_at` | Bags transferred to courier | No earlier than packed and assigned under normal flow | Exposure and route start |
| `stop_arrived_at` | Arrival at the customer location | Not substituted for completion | Travel analysis |
| `handoff_at` | Customer handoff completed | Required completion evidence | Commitment and lateness |
| `route_id`, `sequence` | Execution trip and stop order | Versioned when changed | Stop-position analysis |
| `load_kg`, `load_l` | Actual or validated estimated load | Nonnegative and unit-consistent | Capacity screening |
| `temperature_class` | Required handling class | Controlled vocabulary with effective dates | Compatibility |
| `cancelled_at`, `reason` | Cancellation state and reason | Retain historical state | Avoid excluding difficult orders |
| `policy_version` | Rule or model version active at decision | Joinable to release record | Audit and experiment |

Event timestamps need an authoritative clock and a policy for late-arriving events.
When the device is offline, ingestion time and event time differ.
Preserve both rather than silently overwriting one with the other.
A missing handoff is not an on-time delivery; it is an unresolved observation until reconciled.

Use historical data as it existed at the prediction time.
An eventual actual travel time, a later cancellation, or a future courier acceptance cannot be used as a feature in the earlier decision.
Retain the candidate set and reason for rejection if policy evaluation needs to explain why a route was not chosen.
Limit access to addresses, phone numbers, proof-of-delivery media, and other personal data to legitimate delivery and audit needs.
Set retention and deletion according to applicable requirements and the organization's approved policy.

### Identifiers and interoperability

GS1 defines identifiers and data standards for products, locations, logistic units, and event sharing.
A Global Trade Item Number (GTIN) identifies a trade item; it does not replace an order identifier, tote scan, or proof that a particular unit went to the right customer.
A Global Location Number (GLN) can identify a location when the trading partners use it; a local shelf coordinate is still needed for indoor picking.
Electronic Product Code Information Services (EPCIS) provides a standardized event-sharing model, but adopting it is a design choice unless a specific contract or law requires it.
See the GS1 sources in section 16 for the current specifications.

A travel matrix is not a universal data standard.
Document its origin, travel mode, geography, update cadence, directional convention, and timestamp.
Measure whether predicted times remain calibrated in rain, peak periods, and multi-unit buildings.
A map's geometric distance cannot replace a permitted pedestrian or vehicle path.

### Food, safety, and labour context

Food safety is jurisdiction- and product-specific.
In the United States, the Food and Drug Administration's Food Code is a model for retail food regulation; state, local, tribal, and territorial authorities determine adoption and modifications.
The code and local requirements must be reviewed for the actual operation.
A route optimizer should receive site-approved handling constraints; it should not infer a universally safe time from this example.

In the European Union, Regulation (EC) No 852/2004 establishes hygiene requirements for food business operators, including temperature-control obligations where necessary.
Its application depends on the activity and national enforcement context.
Hazard analysis and critical control point (HACCP) procedures are a food-safety management approach, not a routing algorithm.
A food-safety professional must determine controls, limits, monitoring, and corrective action for the specific products and jurisdiction.

Worker classification, working time, traffic rules, insurance, and pay obligations vary by jurisdiction and contract.
Routing design must not assume an independent-contractor classification or permit unsafe speeds.
Legal and safety owners should translate applicable obligations into versioned operational constraints.
The sources are references for further review, not legal advice; requirements and local adoption should be checked as of the decision date.

## 9. Industry practice and case evidence

### Evidence discipline

A published vehicle-routing result can establish that a method worked in a defined benchmark.
It does not establish that the same savings occur in a particular quick-commerce store.
A company description can establish what a company says it offers, not independently verified economic or safety outcomes.
The following cases separate documented facts from operational inference.

### Case 1 — Dynamic pickup and delivery research

Savelsbergh and Sol describe the general pickup and delivery problem and its constraints in *Transportation Science*.
The research frames pickup and delivery as a route problem with precedence and resource restrictions.
Its general model is useful for seeing why membership and ordering cannot be separated.

**Inference for a quick-commerce operator:** represent store pickup, delivery windows, and vehicle capacity explicitly rather than clustering by distance alone.
The paper is not evidence that any specific grocer achieved a particular saving.
Real-time food delivery also adds uncertain preparation, customer availability, and worker acceptance.

### Case 2 — Dynamic routing review

Pillac and coauthors review dynamic vehicle routing and distinguish static information from requests revealed over time.
The distinction matters when a new order appears after a courier has departed.
An offline solution that sees the full day's orders has information unavailable to a live dispatcher.

**Inference:** test a rolling-horizon policy with timestamped input snapshots and a feasible fallback.
Compare it with the real baseline under the same information and capacity constraints.
A hindsight-optimal route is an upper bound or diagnostic, not an achievable service promise.

### Case 3 — Instacart shopper batching

Instacart's shopper-facing materials describe batches and the work involved in shopping and delivering customer orders.
The company controls that documentation and may change details by market and over time.
The material demonstrates that a commercial platform presents work as batches, not that every batch improves total cost or customer outcomes.

**Inference:** show shoppers distinct order identities and verify handoffs when multiple orders share a trip.
Measure the extra sorting and the last customer's delay, rather than treating a batch count as a productivity outcome by itself.
Consult current local shopper terms for actual pay and assignment rules.

### Case 4 — DoorDash delivery offers

DoorDash's Dasher guidance describes delivery offers and, in some markets or circumstances, multiple deliveries on an offer.
This is company-authored operational guidance, not independent validation of routing performance.
It also may not describe the same service flow as a dark-store grocer.

**Inference:** distinguish an offer from an acceptance and record any subsequent change.
Evaluate courier waiting, route duration, and service by stop order alongside platform cost.
Do not transfer company-specific terms to another operator or jurisdiction.

### Translate evidence into decisions

Use field observations to verify the store map, tote handling, and customer-access delays.
Use a time-split evaluation to test prediction quality under seasonality and operating changes.
Run a bounded pilot only after the safety owner approves the constraints and the dispatcher has a reversible fallback.
Track both labor productivity and individual customer outcomes.
If a claimed saving relies on two couriers who were not actually available, revise the baseline.
If an improvement exists only in the average while severe late deliveries increase, the policy may fail its service contract despite apparent efficiency.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Waiting without a cutoff | Oldest order age rises | Timer restarts as orders arrive | First order misses promise | Fixed oldest-order release deadline |
| Optimizing only distance | Distance down, lateness up | Handoffs and congestion omitted | Late customers | Compare full timeline and stop buffers |
| Merged pick and delivery groups | Frequent manual splits | Same grouping assumed for different resources | Infeasible route or wasted sorting | Model two grouping decisions |
| Wrong tote | Scan overrides increase | Multi-order items mixed | Wrong customer receives goods | Separate totes and scan verification |
| Unavailable courier | Offers expire often | Plan assumes acceptance | Late departure | Model offer and fallback time |
| Stale traffic matrix | Residuals rise in a region | Weather or road change not reflected | Systematic lateness | Refresh data and use conservative fallback |
| Last-stop neglect | Second-stop late rate rises | Earlier service delay propagates | Unequal customer experience | Stop-position guardrails |
| Capacity proxy | Bag is full below weight limit | Volume or shape ignored | Unsafe or failed pickup | Separate mass, volume and compartment checks |
| Cold-chain breach | Long staging or missing temperature records | Route duration used as proxy for temperature | Product safety risk | Site-approved handling limits and monitoring |
| No return leg | Courier availability prediction is early | Repositioning omitted | Next assignment fails | Model end location and next availability |
| Future-information leakage | Replay looks much better than pilot | Actual outcomes used in decision features | Invalid policy estimate | Point-in-time snapshots |
| Survivor bias | Only completed orders reported | Cancellations and open orders dropped | Inflated on-time rate | Include and classify all eligible orders |
| Wrong promise version | Service improves after edits to deadline | Original commitment overwritten | Misreported performance | Immutable original promise and history |
| Handoff omitted | Arrival looks on time, complaints rise | Building access or customer contact excluded | Actual completion misses promise | Record arrival and completed handoff |
| Unsafe courier pressure | Speed exceptions or complaints increase | Objective rewards implausible travel | Injury or noncompliance | Safe-time constraints and escalation |
| Oversized batch | Sorting and mispick rates increase | Shared walking saving overwhelmed by complexity | Longer pick and rework | Limit size and measure incremental work |
| Reoptimization churn | Frequent route changes after acceptance | System reacts to every small update | Confusion and extra work | Trigger thresholds and change rules |
| False saving | Batch labour down but queue grows | Resource opportunity cost omitted | System-wide cost increases | Compare feasible system alternatives |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Accurate commitments | Correct deadline calculation | Promise edits or time-zone errors | Reconcile immutable customer messages |
| Orders ready at decision time in example | Makes departure timeline reproducible | Item missing or packing delay | Compare packed scans to decision time |
| Two solo couriers available in example | Makes stated comparator executable | Competing work or refusal | Replay actual available supply and offers |
| Travel matrix representative | Predicts each route leg | Weather, traffic, road closure | Backtest residuals by area and hour |
| Handoff time constant in example | Keeps arithmetic transparent | Apartment access or absent customer | Measure distribution by building type |
| Load estimates correct | Capacity screen | Unrecorded packaging or bulky items | Weigh and measure sample bags |
| Products compatible | Permits shared route | Temperature, allergen, or handling conflict | Validate with food-safety owner |
| Single loading duration | Avoids double counting | Additional bags require more time | Observe loading against batch size |
| Return to same store | Defines occupied-time comparison | Courier ends elsewhere or accepts another job | Track actual end location and next job |
| Walking speed 60 m/min | Converts example metres to minutes | Congestion, heavy loads, aisle restrictions | Measure actual picker paths and durations |
| Sorting adds one minute | Calculates incremental pick effort | Complex baskets or verification failures | Time study across batch sizes |
| Forecasts use contemporaneous information | Avoids leakage | Event updates overwrite history | Audit feature timestamps |
| Buffer policy is 3 minutes | Defines example screen | Site uses risk-based or tier-specific rule | Review approved policy and outcomes |
| Scenario delays are illustrative | Shows direction of sensitivity | Correlated delays or distribution shift | Use joint, out-of-sample scenarios |

The example's 13 courier-minute saving is a resource-time difference, not verified profit.
It excludes fixed costs, variable pay, compensation, downstream assignments, and changes in customer behavior.
The pick extension is a separate hypothetical and cannot be combined with the packed-order timeline without rebuilding it.
The point estimates do not establish a 95% or any other on-time probability.

## 12. Exercises

### Exercise 1 — Separate the decisions

**Time:** 10 minutes.
A store collects orders A and B on one cart, then sends A with a bicycle courier and B with a car courier.
A third order C is picked alone and delivered with A.

**Deliverable:** list the pick batches and delivery batches.
Identify one reason these partitions differ.

**Acceptance criteria:** every order appears once in each relevant grouping; the answer distinguishes store walking from road routing.

### Exercise 2 — Recalculate a delivery route

**Time:** 15 minutes.
Use the worked example's A-then-B route, but increase the A-to-B travel leg from 4 to 7 minutes.
Keep all other values unchanged.

**Deliverable:** completed handoff times, lateness, minimum buffer, and occupied courier time.

**Acceptance criteria:** include the return leg and apply the 3-minute buffer separately from the deadline.

### Exercise 3 — Modify the calculation

**Time:** 15 minutes.
In the code, increase B's volume from 14 L to 30 L.

**Deliverable:** state which shared candidates are rejected by capacity and which solo routes remain possible.
Explain what else the example script would need to do to print rejected candidates without attempting to index `None`.

**Acceptance criteria:** calculate combined volume; recognize that the existing printing loop assumes a non-`None` result.
Do not claim the original script handles this change without modification.

### Exercise 4 — Screen a candidate

**Time:** 15 minutes.
A courier has a 10 kg and 35 L bag.
Three orders weigh 3 kg, 4 kg, and 4 kg and occupy 8 L, 10 L, and 9 L.
All deadlines would be met in a point forecast.
The first two are compatible; the third cannot share a compartment with the first under site policy.

**Deliverable:** assess the three-order batch and the first-two-order batch.

**Acceptance criteria:** calculate both capacities and identify the independent compatibility issue; do not use timely arrival to override a hard constraint.

### Exercise 5 — Release or wait

**Time:** 15 minutes.
At 14:00, A is due by 14:24.
A conservative remaining pick, pack, load, travel and handoff sequence needs 20 minutes after release.
The site's maximum batching wait is 3 minutes from now, and the minimum additional buffer is 2 minutes.

**Deliverable:** latest permitted release time and a rule for what happens if no partner arrives.

**Acceptance criteria:** show the arithmetic; distinguish the 3-minute maximum wait from the tighter deadline-based limit.

### Exercise 6 — Critique a dashboard

**Time:** 20 minutes.
A report says average courier minutes per completed order fell from 20 to 16 after batching.
It excludes cancellations, reports no sample sizes, and gives no service result by stop position.
The rollout coincided with a quieter demand period.

**Deliverable:** identify at least four threats to the claimed impact and specify four complementary measures.

**Acceptance criteria:** address selection, confounding, customer service, and resource availability; avoid claiming that the observed difference proves causation.

### Exercise 7 — Design an exception control

**Time:** 25 minutes.
A courier has accepted two orders; the first customer cannot be reached and the second order has 6 minutes of remaining buffer.
The applicable agreement limits unilateral changes to accepted work.

**Deliverable:** a decision tree with trigger, decision owner, communication, permitted alternatives, and audit events.

**Acceptance criteria:** preserve safety and agreement constraints, recompute the second order's timeline, and include a no-feasible-option escalation.

### Exercise 8 — Design a pilot

**Time:** 30 minutes.
A city operator wants to increase the maximum delivery batch size from two to three.

**Deliverable:** a pilot plan with unit of randomization, eligibility, baseline, primary metric, service and safety guardrails, stop rule, and analysis population.

**Acceptance criteria:** address interference between shared couriers, power or sample-size planning without inventing a number, and inclusion of cancellations and open orders.
Name an accountable owner for launch and rollback.

## 13. Self-check answers

### Exercise 1

Pick batches are `{A, B}` and `{C}`.
Delivery batches are `{A, C}` and `{B}`.
A and B share aisle walking, while A and C can have compatible delivery locations and timing.
An order can belong to a different group in each stage without any inconsistency.

### Exercise 2

Loading ends at minute 2.
A handoff completes at minute `2 + 7 + 2 = 11`.
B completes at minute `11 + 7 + 2 = 20`.
A has `19 - 11 = 8` minutes of buffer; B has `22 - 20 = 2` minutes.
Neither is late, but the 2-minute minimum buffer fails the 3-minute policy.
Return takes 8 minutes, so occupied time is `20 + 8 = 28` courier-minutes.

### Exercise 3

The combined volume is `12 + 30 = 42` L, greater than the 40 L bag limit.
Both A-then-B and B-then-A return `None` from `route` because they contain the same load.
A alone uses 12 L and B alone uses 30 L, so each passes the volume check; their weights also pass.
The printing loop must check `if result is None` and print an infeasibility reason or skip dictionary indexing.
A production implementation should retain a structured rejection reason rather than a bare `None`.

### Exercise 4

All three orders weigh `3 + 4 + 4 = 11` kg, exceeding the 10 kg limit.
Their volume is `8 + 10 + 9 = 27` L, below the 35 L limit.
The first and third are also incompatible in the specified compartment, an independent failure.
The first two weigh 7 kg and occupy 18 L; they are compatible and meet the stated point-forecast deadlines.
Additional uncertainty and other operational constraints still require review.

### Exercise 5

The deadline-based latest release is `14:24 - 20 min - 2 min = 14:02`.
The maximum-wait cutoff is 14:03.
Use the earlier cutoff, 14:02, unless a deterioration requires earlier release.
If no compatible partner arrives, release A alone at or before 14:02 and follow the normal exception path if solo execution becomes infeasible.

### Exercise 6

Strong answers identify exclusion of cancellations and open orders, confounding from the quieter period, lack of sample sizes, and lack of a comparable counterfactual.
They also question whether courier availability and return or repositioning were included.
Useful measures include on-time completion against original promise, late minutes by stop position, cancellations per eligible order, courier occupied time including return, misdelivery rate, and worker waiting.
Stratification alone may not eliminate confounding; a safe randomized design can improve causal interpretation.

### Exercise 7 rubric

A strong decision tree starts a customer-contact timer and recalculates the remaining stop's completion using current travel and handoff information.
It identifies the dispatcher or on-duty lead, the courier's role, and the policy governing route changes.
It includes a permitted split or resequence if feasible, a safe return or escalation if not, and prompt customer communication.
It records contact attempts, timestamps, revised route, courier response, and final disposition.
A weak response silently skips the first customer or forces the courier to violate the agreement.

### Exercise 8 rubric

A strong plan chooses an experimental unit large enough to limit courier-pool interference, such as store-time clusters if operationally suitable.
It compares a maximum of three orders with the existing maximum of two, subject to identical hard constraints.
It specifies eligible orders before treatment, measures all assigned orders including cancellations and open cases, and states how incomplete observations are resolved.
A primary economic measure is paired with lateness, severe delay, misdelivery, food safety, and courier welfare guardrails.
An operations owner and safety owner approve launch; an on-duty owner can roll back immediately.
Sample size is based on baseline variance, minimum meaningful effect, clustering, and acceptable error rates rather than a made-up target.

## 14. Interview questions

1. **Why can pick and delivery batches differ?**
   Strong: explains aisle overlap versus delivery geography and timing, with separate capacity and identity controls.
   Weak: assumes the same order set must persist throughout.
2. **What makes a route feasible?**
   Strong: checks sequence, readiness, capacity, compatibility, commitments, and policy constraints.
   Weak: treats short distance as sufficient.
3. **How would you calculate the second stop's completion?**
   Strong: includes first travel, first handoff, inter-stop travel, and second handoff.
   Weak: uses direct store-to-second travel only.
4. **When should a store stop waiting for another order?**
   Strong: uses a fixed maximum wait and latest release based on remaining work and buffer.
   Weak: waits indefinitely for a target batch size.
5. **What is wrong with adding independent p95 times?**
   Strong: explains that component dependence and the distribution of sums matter.
   Weak: asserts the result is automatically the p95 of total time.
6. **How do you compare a batch with solo delivery?**
   Strong: uses a feasible simultaneous resource baseline, including return and opportunity cost.
   Weak: assumes unlimited couriers and ignores subsequent work.
7. **What would you measure by stop position?**
   Strong: reports completion against original promise, delay, and customer outcomes with counts.
   Weak: reports only the average batch duration.
8. **What happens after a courier declines?**
   Strong: advances the clock, refreshes availability and promises, and executes a fallback.
   Weak: treats an offer as an accepted assignment.
9. **How would you model perishables?**
   Strong: uses site-approved product-specific handling controls and measured exposure where available.
   Weak: declares every route under an arbitrary duration safe.
10. **How would you evaluate a new batching policy?**
    Strong: distinguishes shadow replay from causal outcomes and addresses shared-resource interference.
    Weak: attributes a before-and-after change solely to the policy.

## 15. Key terms

- **Assignment:** allocation of work to a resource; see [mechanics](#step-10--offer-confirm-and-execute).
- **Batch:** orders consolidated for a defined stage; see [mental model](#3-the-core-mental-model).
- **Buffer:** time reserved between predicted completion and deadline; see [worked example](#6-worked-example).
- **Capacity:** a resource limit such as kilograms or litres; see [definitions](#4-definitions-and-vocabulary).
- **Commitment:** a customer-facing promised delivery time or window; see [data contract](#event-data-contract).
- **Compatibility:** permission to share a resource under product and handling rules; see [eligibility](#step-3--set-hard-eligibility-rules).
- **Courier availability:** time and place at which a courier can undertake work; see [candidate construction](#step-7--build-delivery-candidates).
- **Deadline:** the timestamp used to screen completion; see [definitions](#4-definitions-and-vocabulary).
- **Delivery batch:** orders on one courier trip; see [mental model](#3-the-core-mental-model).
- **Detour:** additional route work relative to a named comparator; see [definitions](#4-definitions-and-vocabulary).
- **ETA:** estimated arrival time, a forecast rather than a commitment; see [mental model](#3-the-core-mental-model).
- **Feasibility:** satisfying all applicable hard constraints; see [eligibility](#step-3--set-hard-eligibility-rules).
- **Handoff:** transfer at a customer stop; see [event data](#event-data-contract).
- **Lateness:** positive completion delay relative to deadline; see [service risk](#step-8--calculate-service-feasibility-and-risk).
- **Opportunity cost:** value of the best feasible alternative use; see [economics](#step-9--compare-economics-within-the-feasible-set).
- **Pick batch:** orders collected on one picker tour; see [pick candidates](#step-4--form-pick-batch-candidates).
- **Pick path:** sequence of store locations visited; see [pick path](#step-5--choose-a-pick-path).
- **Release cutoff:** latest permitted time to stop waiting and start work; see [release decision](#step-6--decide-whether-to-wait-for-another-order).
- **Reoptimization:** revising a plan when conditions change; see [monitoring](#step-11--monitor-and-replan).
- **Return leg:** movement after the last delivery to the relevant next location; see [worked example](#6-worked-example).
- **Rolling horizon:** repeated planning with information available at each decision time; see [mental model](#3-the-core-mental-model).
- **Route:** an ordered sequence of movements and service events; see [definitions](#4-definitions-and-vocabulary).
- **Shadow mode:** recommendations observed but not executed; see [policy learning](#step-12--learn-and-change-policy).
- **Slack:** time remaining after accounting for estimated work; see [mental model](#3-the-core-mental-model).
- **Stop position:** order in which a customer is served; see [failure modes](#10-failure-modes-and-red-flags).
- **Tote:** a container used to separate or transport picked goods; see [pick candidates](#step-4--form-pick-batch-candidates).

## 16. Sources

These sources are reference pointers for this educational unit; check current editions and local applicability before deployment.

1. Savelsbergh, M. W. P., and Sol, M. “The General Pickup and Delivery Problem.” *Transportation Science* 29(1), 1995, pp. 17–29. [https://doi.org/10.1287/trsc.29.1.17](https://doi.org/10.1287/trsc.29.1.17). General formulation of pickup, delivery, and route constraints; not a quick-commerce impact estimate.
2. Pillac, V., Gendreau, M., Guéret, C., and Medaglia, A. L. “A Review of Dynamic Vehicle Routing Problems.” *European Journal of Operational Research* 225(1), 2013, pp. 1–11. [https://doi.org/10.1016/j.ejor.2012.08.015](https://doi.org/10.1016/j.ejor.2012.08.015). Dynamic information and routing context.
3. GS1, [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications). Product and identification standards; consult the applicable edition.
4. GS1, [EPCIS and CBV](https://www.gs1.org/standards/epcis). Event-data interoperability standards; use depends on implementation and agreements.
5. U.S. Food and Drug Administration, [FDA Food Code](https://www.fda.gov/food/fda-food-code/food-code-2022). Model retail food code; adoption and amendments are local.
6. European Union, [Regulation (EC) No 852/2004 on the hygiene of foodstuffs](https://eur-lex.europa.eu/eli/reg/2004/852/oj). Primary legal text; check consolidated law and national implementation for current application.
7. Instacart, [Shopper Help Center](https://www.instacart.com/company/shopper-community/). Company-authored shopper materials; operational descriptions can change.
8. DoorDash, [Dasher Central](https://dasher.doordash.com/). Company-authored courier guidance; terms and features vary by market.

## 17. Links

- [Module 22 index](_index.md).
- [Promise tiers and dark stores](01_promise-tiers-dark-stores.md).
- [Courier supply and demand](05_courier-supply-demand.md).
- [Module 22 survey and lab sections](../../../course/22_quick_commerce_on_demand_delivery.md).
