# The Five Visibility Layers

> Module 19 — Control Towers, Visibility, and Event Intelligence  
> Topic 1 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 03, 06, and 12  
> Version 2026.5

## 1. Why this topic exists

A shipment can be visible on a map and still be invisible to the person deciding whether to promise a customer order.
The position says where a vehicle was reported; it does not say which orders are inside, whether the goods are saleable, or whether a customer will accept a partial delivery.
A useful control tower connects these questions without pretending that every observation has the same meaning.

Consider a regional distributor waiting for a supplier's shipment before a promotion.
The carrier reports the vehicle is late.
The receiving dock is already booked, the warehouse has some stock, and several customers have competing commitments.
A planner who sees only the shipment may expedite everything.
A customer-service representative who sees only the order may promise stock twice.
A buyer who sees only the purchase order may assume the supplier can replace the shortfall.
A network analyst may see spare capacity at another node but not the product or transport constraints that make it usable.

The expensive mistake is not merely an inaccurate map.
It is choosing or communicating an action before checking the evidence needed for that action.
A control tower should shorten the time between an actionable change and a responsible decision, while preserving uncertainty and authority boundaries.
It should also make the cost of a wrong action explicit.

Five complementary layers provide a practical organizing model:

1. **Shipment:** movement and milestones of a defined consignment or transport leg.
2. **Inventory:** quantity and disposition of stock at a location and time.
3. **Order:** demand, supply commitments, allocations, and their fulfillment status.
4. **Supplier:** a supplier's commitments, capacity, dependencies, and ability to respond.
5. **Network:** relationships and feasible flows among suppliers, sites, lanes, and customers.

These layers are views of connected business objects, not five independent truths.
A shipment may contain several orders; an order may be split across shipments and lots.
Inventory can be in a warehouse, in transit, reserved, damaged, or unavailable for another reason.
Supplier and network views add context, but a capacity report is not proof that a particular unit will arrive on time.

The primary audience is the person who must decide: transport controller, inventory planner, order manager, buyer, site manager, or network lead.
They need a shared language for what is observed, inferred, committed, and still unknown.
They also need explicit permission to act: visibility is not authority to spend, change a customer promise, or share a partner's confidential data.

This topic shows how to choose the minimum evidence for a decision, reconcile the layers, and assign ownership.
It deliberately does not design a complete event platform or prediction model.
Those topics become useful after you know what decision the data is supposed to support.

## 2. Learning outcomes

After this topic you can:

1. **Classify** a business question into the visibility layer or combination of layers it requires; see exercises 1 and 2.
2. **Draw** the object and information relationships between supplier, shipment, inventory, order, and network; see exercise 3.
3. **Calculate** a time-bounded available-to-promise quantity without counting reserved or in-transit units twice; see exercises 4 and 5.
4. **Choose** a response using explicit service, timing, and cost assumptions; see exercise 6.
5. **Assign** decision, execution, escalation, and communication ownership; see exercise 7.
6. **Critique** freshness, identity, completeness, and uncertainty for a control-tower view; see exercise 8.
7. **Specify** a practical visibility improvement and a measurable operating result; see exercises 9 and 10.

You should be able to defend why a layer is necessary, not simply name it.
You should also distinguish an observed fact from an estimate and a proposed action from an authorized commitment.

## 3. The core mental model

A visibility layer is a decision-oriented view over evidence about objects and relationships.
The layers overlap; none is a mandatory database or screen.

```text
Supplier commitment/capability -- supplies --> item / lot
         |                                        |
         | releases                               | belongs to
         v                                        v
Purchase order line --> shipment / leg --> receipt --> inventory by site/status
         |                    |                         |
         |                    | contributes to          | can be allocated to
         +--------------------+-------------------------v
                                               customer order line
                                                           |
Network: nodes + lanes + capacities + constraints --------+
                                                           v
                       impact estimate -> owner -> decision -> action -> outcome
```

The arrows express business relationships, not necessarily direct database joins.
An advance shipment notice can connect a purchase-order line to handling units, while a receipt can establish what actually arrived.
A carrier location report is evidence about transport, not proof of receipt or quality acceptance.
A network lane is a feasible relationship subject to capacity, transit time, product compatibility, and policy; a line on a map alone does not make a transfer feasible.

### A question-to-evidence chain

Start with a decision and deadline.
Then identify the affected business object, required observation, source, timestamp, uncertainty, and authorized actor.
For example, the question “Can we honor order C-17 by Friday?” requires a customer-order commitment, usable inventory, inbound timing, and feasible movement to the customer.
A supplier recovery option may also be relevant.
No single location ping answers that question.

Use three distinct clocks:

- **Event time:** when the source says a business event occurred.
- **Recorded time:** when the tower received or stored it.
- **Decision time:** when a person or system must choose an action.

A record received recently can describe an old event.
A forecast estimated time of arrival (ETA) can change without any physical movement.
Show its issue time and uncertainty, and never silently replace an actual receipt with a forecast arrival.

### The five lenses

| Lens | Primary object | Question it can answer | Question it cannot answer alone |
|---|---|---|---|
| Shipment | Consignment, handling unit, leg | Where is the shipment and which milestone is reported? | Which customer commitments will fail? |
| Inventory | Item, lot, site, status | How much usable stock is present or expected? | Whether a customer has agreed to an alternative |
| Order | Purchase or customer order line | What was requested, promised, allocated, or fulfilled? | Whether the supplier has spare production capacity |
| Supplier | Supplier, facility, item commitment | What supply is committed and what recovery is plausible? | Whether a vehicle has physically arrived |
| Network | Nodes, lanes, resources, dependencies | Which alternatives and common exposures exist? | Whether an individual order is already fulfilled |

### Observe, interpret, decide, learn

1. **Observe:** capture source evidence and its coverage.
2. **Interpret:** resolve identities and calculate potential impact, preserving unknowns.
3. **Decide:** compare feasible choices under constraints and decision rights.
4. **Act:** issue an approved instruction to the executing party.
5. **Learn:** confirm execution and outcome; measure whether the intervention helped.

A dashboard that stops after observation is a visibility product, not necessarily an operating control loop.
An automated instruction without a controlled decision policy is not a substitute for a decision owner.

## 4. Definitions and vocabulary

The following terms have operational meanings in this topic.
Local systems may use different names; document the mapping before combining their quantities.

| Term | Definition | Common misuse | Unit or coding context |
|---|---|---|---|
| Visibility layer | A view organized around a class of business questions | Treating a layer as a source system | Named layer |
| Business object | Entity whose state is tracked | Assuming an identifier is globally unique | Typed identifier |
| Shipment | Goods moved under a transport arrangement | Equating one shipment with one order | Shipment ID |
| Consignment | Goods associated with a transport movement or instruction | Assuming all parties share the same ID | Carrier or shipper ID |
| Transport leg | One segment between two nodes | Treating a multi-leg trip as one transit time | Origin, destination, mode |
| Milestone | Reported transport or operational event | Treating an estimated milestone as actual | Event type and time |
| ETA | Estimate of arrival at a specified node | Confusing point estimate with guarantee | Timestamp and time zone |
| Receipt | Record of goods received at a site | Assuming receipt equals inspection release | Units and site |
| On-hand | Quantity recorded as physically held | Assuming all on-hand is usable | Item units per site |
| Available stock | On-hand after applicable exclusions | Subtracting the same reservation twice | Units at a stated time |
| Reservation | Stock set aside for a demand or purpose | Counting it as free and allocated | Units and reservation ID |
| Allocation | Assignment of supply to demand | Treating a proposed assignment as a shipment | Units per order line |
| Inventory status | Disposition such as saleable, held, or damaged | Treating all statuses as saleable | System status code |
| Lot | Group of units with shared traceability attributes | Assuming every unit in a lot is co-located | Lot code |
| Purchase order | Buyer's order to supplier | Treating the order as physical supply | Order and line IDs |
| Customer order | Demand record with requested and committed terms | Treating request and promise as identical | Order and line IDs |
| Promise | Agreed customer-facing commitment | Treating a model prediction as a promise | Quantity, time, location |
| Plan | Intended internal course of action | Treating an unapproved plan as a contract | Quantity and period |
| Forecast | Estimate of future demand or event | Treating forecast as observed fact | Units or timestamp; interval |
| Target | Desired performance level | Treating a target as a measured outcome | Metric and period |
| Supplier commitment | Supplier's accepted quantity and date | Treating an unconfirmed purchase order as accepted | Units and date |
| Supplier capacity | Potential output over a period subject to constraints | Treating theoretical capacity as available capacity | Units per day or week |
| Node | Facility, supplier site, or customer location in a network | Assuming all nodes can stock every item | Location ID |
| Lane | Feasible transport connection between nodes | Assuming the existence of a route guarantees capacity | Origin, destination, mode |
| Dependency | A relationship through which disruption may propagate | Assuming a direct supplier is the only dependency | Node or component relationship |
| Lead time | Elapsed time from specified start to specified end | Omitting endpoints or variability | Hours or days |
| Freshness | Age of evidence relative to the decision | Using ingestion time as the event time | Minutes or hours |
| Coverage | Share of in-scope objects with usable evidence | Reporting only connected partners as the denominator | Numerator and denominator |
| Completeness | Presence of required fields or events | Confusing field completeness with correctness | Percentage by field or object |
| Provenance | Source and transformation history of a value | Presenting a calculated value as a source fact | Source ID and timestamps |
| Decision owner | Person accountable for selecting an action | Assigning a queue without authority | Role and authority limit |
| Executor | Person or system carrying out an approved action | Assuming the decision owner performs every task | Role or system |
| Escalation | Transfer of decision to an authorized higher level | Escalating after the action deadline | Trigger, recipient, deadline |
| Exception | State requiring attention against a defined rule | Treating every status update as an exception | Object, rule, severity |
| Service level | Defined measure of fulfillment performance | Comparing metrics with different denominators | Percentage and time window |
| Counterfactual | Estimated result without the intervention | Treating an unobserved outcome as measured fact | Scenario and assumptions |

## 5. The mechanics, step by step

The following procedure is intentionally business-led.
You can use a spreadsheet, a query, or a commercial platform, provided that the same definitions and ownership controls apply.

### Step 1 — Name the decision, owner, and deadline

**Input:** a business outcome at risk and a person who can act.
**Output:** a decision statement with an action deadline.

Write the decision as a verb and object: “allocate available units to customer orders by 10:00 local time,” not “improve visibility.”
Record the customer or operational consequence of delay, the scope of products and sites, and the latest time an alternative can be executed.
The deadline is often earlier than the customer's delivery date because booking, picking, transit, and communication consume time.

Ask who is accountable for choosing among alternatives.
Separate that person from the operator who books freight, the analyst who calculates impact, and the person authorized to revise a customer promise.
If authority is conditional on spend, record the threshold and escalation path.
A control tower can recommend an action; the accountable role still approves it under policy.

**Branch:** if no meaningful action remains before the deadline, route the case to customer communication and recovery rather than creating an operational alert that cannot change the outcome.
If a future action is possible, set a review time based on the next decision point.

### Step 2 — Define the decision grain

**Input:** the decision statement and affected demand or supply.
**Output:** a typed object key and time horizon.

Choose the smallest grain at which the decision changes.
For allocation this may be item, customer-order line, destination, and due time.
For transport booking it may be a handling unit, shipment leg, origin, and departure time.
For supplier recovery it may be supplier facility, product, production period, and confirmed quantity.

Distinguish a header from its lines.
A purchase order with ten lines is not fully late merely because one line is late.
Similarly, a shipment can carry stock for several orders and a customer order can span several shipments.
Persist relationship quantities as well as IDs, so a split does not duplicate the entire shipment quantity on each order.

Include an effective timestamp or interval for mutable relationships.
An order may be reallocated after the first snapshot.
A join that uses the latest assignment to explain an earlier decision will misstate what the decision maker knew at that time.

**Branch:** if an object cannot be identified unambiguously, retain it as unmatched, assign an owner to resolve it, and treat its impact as unknown rather than silently dropping it.

### Step 3 — Establish what shipment visibility actually observes

**Input:** carrier feeds, shipper notices, milestone definitions, and shipment identifiers.
**Output:** a timeline with provenance, event age, and a defensible status.

Identify the transport object: shipment, container, trailer, package, or vehicle.
A vehicle position can cover multiple shipments and must be associated with the correct load and leg.
A container identifier does not prove the contents after a transload unless the transload relationship is recorded.

Separate planned departure, actual departure, estimated arrival, actual arrival, and receiving confirmation.
Define each milestone's location and source.
A geofence entry can indicate proximity to a warehouse, but it does not establish unloading, count, or quality release.
A carrier-reported delay and a receiver's dock appointment are different observations; retain both when they conflict.

Show the last event time and its receipt time.
Measure coverage against all in-scope shipments, including those with no feed.
Show a stale state when a feed has exceeded the business tolerance rather than displaying an old position as current.
For a timestamp in an unfamiliar zone, convert using an explicit time-zone rule and preserve the original value.

**Branch:** if the ETA is late but there is no reliable item-level manifest, contact the shipper or supplier for the contents before claiming a specific customer impact.
If the manifest exists, use its quantities as expected contents, not as proof that every unit is received and saleable.

### Step 4 — Establish usable inventory, not just a balance

**Input:** stock by item, site, status, lot if relevant, reservations, inbound receipts, and outbound commitments.
**Output:** a time-phased supply ledger with mutually exclusive quantities.

Start with a timestamped physical or book balance.
Exclude damaged, quality-held, expired, or otherwise unavailable units according to policy.
Account for reservations exactly once: either start with an unreserved balance, or start with total saleable stock and subtract reservations, but do not do both.
Make the chosen definition visible.

Distinguish a stock snapshot from a flow.
A truck that is physically in transit may be included in an enterprise-owned inventory balance, but it is not necessarily in the destination site's on-hand balance.
The ledger must state whether in-transit units are already included before adding expected receipts.
A receipt replaces an expected inbound quantity as the shipment moves through state; it is not an extra unit of supply.

For each future bucket, add only inbound quantities expected to be available by the decision's required time.
Apply receipt, inspection, put-away, picking, and transport delays where relevant.
If the ETA is uncertain around the cutoff, show a scenario or interval rather than declaring the units definitely available.
Do not combine units of different products, packaging, or quality grades without a valid conversion.

**Branch:** if a physical count contradicts the book balance, use the agreed operational control for the discrepancy and escalate the uncertainty.
Do not hide the mismatch in a single confident available-to-promise number.

### Step 5 — Connect the order and its commitment

**Input:** customer and purchase order lines, accepted commitments, allocation rules, and fulfillment events.
**Output:** order-level remaining need and a clear customer promise.

For a customer order line, identify requested quantity, accepted quantity, promised date and location, shipped quantity, delivered quantity if known, cancellations, and existing allocations.
Specify whether partial deliveries are allowed and whether the commitment is measured at shipment, delivery, or another milestone.
A requested date is not necessarily an accepted promise.
A revised forecast is not automatically a customer-approved change.

For a purchase-order line, identify requested quantity, supplier-confirmed quantity and date, shipped quantity, received quantity, and outstanding balance.
A purchase order is a commercial instruction; supplier acceptance and shipment evidence are separate facts.
Avoid subtracting shipment and receipt from the same outstanding balance twice: receipts may be part of shipped quantity.

Connect order lines to shipments through explicit line-to-handling-unit or allocation relationships where available.
Record the linked quantity and its effective time.
If a load is mixed or partially allocated, do not infer that every order line is delayed by the full shipment quantity.

**Branch:** if the order's promise is missing or disputed, customer service owns confirmation before the tower communicates a breach.
If the promise is known but no feasible supply exists, evaluate an authorized alternative or escalate.

### Step 6 — Assess supplier response and exposure

**Input:** supplier confirmation, facility and item mapping, production or release status, and contact/contract terms.
**Output:** a time-bounded view of feasible supplier options with named confirmation owners.

Ask which supplier facility makes or releases the item, not only which corporate supplier is on the purchase order.
Check the supplier's accepted quantity, date, and any known constraint on material, labor, equipment, or transport.
If the supplier reports spare capacity, ask whether it is available to this buyer and this product, when it can be released, and whether required components are present.
A statement that a factory has spare aggregate capacity does not establish a recoverable quantity for a specific order.

Map known critical sub-tier dependencies where evidence and permissions allow.
Record unknown dependencies explicitly.
A buyer may know the tier-one supplier but not the sub-tier facility; that is a coverage gap, not proof of diversification.
Treat supplier-provided future capacity as a commitment only after the supplier confirms terms through the appropriate commercial process.

**Branch:** if the supplier can offer a partial early release, compare its arrival and handling time with customer deadlines.
If the alternative requires a new commercial commitment, the buyer or authorized procurement role approves it before transport is booked.

### Step 7 — Test the network alternative

**Input:** site and lane graph, eligible inventory, transit and cutoff times, capacity, and business constraints.
**Output:** feasible alternatives and eliminated options with reasons.

Represent a node with its location, function, and relevant operating calendar.
Represent a lane with origin, destination, mode, transit-time assumption, capacity, cost, and booking cutoff.
A feasible alternative must satisfy the entire path, not just one edge.
For example, spare stock at another site is irrelevant if the transfer cannot be picked and delivered before the promise.

Check whether inventory at the alternative site is genuinely unallocated and whether the stock is legally and commercially eligible for the destination.
Consider lot restrictions, temperature control, shelf life, customer approval, and local site obligations where applicable.
Use stated or confirmed capacity, not the visual existence of a route, to support the decision.
Keep fixed costs and per-unit costs separate.

Network visibility also reveals shared exposure.
Several apparently independent supplier names may rely on the same port, factory, or component.
However, a network map is only as complete as its supplier and dependency data.
Mark unverified edges rather than implying a complete graph.

**Branch:** if the route or capacity is unconfirmed, label the option conditional and request confirmation before spending or promising.
If no route can satisfy the current commitment, evaluate customer prioritization or a revised commitment with the responsible owner.

### Step 8 — Reconcile the layers at a common cutoff

**Input:** the five layer views, timestamps, relationship tables, and decision horizon.
**Output:** a time-bounded impact statement and confidence qualifier.

Choose a decision cutoff and describe what was known then.
Source systems update at different speeds, so a dashboard assembled at 09:00 may combine an 08:58 carrier event, an 08:00 warehouse snapshot, and yesterday's supplier confirmation.
Display each important source age separately.
Do not assign the freshest timestamp to the whole case.

Join from the affected shipment or item to the relevant order lines using quantity-bearing relationships.
Reconcile total linked quantities against source totals.
Investigate residuals: unallocated supply, unmatched order lines, over-allocation, duplicate events, and conflicting statuses.
A residual is a useful operational signal; it is not necessarily a data error until the definitions have been checked.

At the decision horizon, compute free usable supply and compare it with competing commitments.
Separate observed stock from expected inbound and proposed transfers.
The result can be a range or scenario table when availability depends on uncertain arrivals.
Never convert an unknown into zero or into a confirmed unit without explaining the consequence.

**Branch:** if material evidence is stale or missing, either obtain confirmation before the decision deadline or choose a conservative action with its potential cost disclosed.
Escalate if no authorized person can accept the uncertainty.

### Step 9 — Compare actions using business economics

**Input:** feasible options, incremental costs, customer consequences, constraints, and uncertainty.
**Output:** a recommendation with assumptions and alternatives.

Compare the incremental cost of each option against the same baseline.
Use a consistent currency and period, and state whether lost contribution, contractual penalties, expedited freight, labor, spoilage, and future recovery are included.
A penalty is not necessarily equivalent to lost margin; avoid counting the same financial harm twice.
If costs cannot be estimated reliably, present a service-versus-cost trade-off instead of inventing a precise net benefit.

Identify hard constraints first: safety, legal obligations, customer-specific terms, and approved spend limits.
Then compare service outcomes and cost for feasible choices.
An allocation rule may prioritize confirmed contractual commitments, strategic customers, or earliest due dates, but the rule must be authorized and applied consistently.
The highest revenue order is not automatically the correct priority.

Use scenarios for uncertain inbound timing or quantities.
A point ETA is not a probability distribution.
If a probability is used, disclose its source and calibration; do not invent one to make expected-value arithmetic look rigorous.
A sensitivity threshold can still show how large a loss would need to be to justify an expedite.

**Branch:** if an option exceeds the owner's spend limit or changes an external promise, send it to the designated approver with a deadline.
If the deadline passes without approval, follow the documented default policy rather than assuming silence is consent.

### Step 10 — Record ownership and execute

**Input:** recommended option, authority matrix, and required communication.
**Output:** an approved or rejected decision and traceable execution.

A practical record contains case ID, affected object IDs, evidence cutoff, options considered, selected option, decision owner, approver if different, executor, deadline, and reason.
Link the record to the evidence snapshot or version so that later changes do not rewrite the history of the decision.
Record the customer's promise separately from the internal plan.

The transport controller may book an approved lane; the inventory planner may release a reservation; customer service may communicate a revised promise.
Each action needs confirmation from its executor.
A recommended transfer that was never booked is not a completed intervention.
A booked transfer that misses its cutoff should return to the exception queue with an owner and next decision time.

Escalation should specify a trigger, recipient, contact path, and response deadline.
It is not sufficient to assign “the business” or a shared mailbox.
Provide a fallback when the primary owner is unavailable.
Where automation can execute within policy, specify the guardrails, audit record, and human override.

### Step 11 — Confirm outcomes and learn

**Input:** action confirmations, actual receipts, delivery evidence, and case history.
**Output:** a closed case with measured outcome and learning.

Confirm whether the chosen action happened, whether the expected supply arrived, and whether the customer commitment was met.
Distinguish an operational success from an analytical success: a late delivery after a sound decision may reflect a new disruption, while a successful delivery can occur despite poor data.
Record these separately.

Measure decision latency from actionable evidence to authorized decision, not from every raw event.
Measure execution latency from approval to confirmation.
Track coverage, stale evidence, unmatched relationships, alert workload, service, and incremental intervention cost with explicit denominators.
Review avoided losses as estimates unless a defensible counterfactual exists.

Close the loop with the source owner when a recurring mismatch is caused upstream.
Do not merely teach users to ignore an alert that is systematically wrong.
When a policy changes, version the decision rule and communicate it to affected roles.

### A compact decision-rights matrix

| Decision or action | Accountable role | Typical executor | Consulted role | Escalate when |
|---|---|---|---|---|
| Confirm shipment milestone | Transport operations lead | Carrier or transport analyst | Receiving site | Evidence conflicts near cutoff |
| Release or reallocate stock | Inventory planning lead | Warehouse or planning system | Order management | Policy or customer priority is disputed |
| Approve expedite spend | Budget owner within delegated limit | Transport controller | Buyer, finance if needed | Spend exceeds delegated limit |
| Revise customer promise | Customer-service manager under commercial policy | Account representative | Sales and inventory planning | Contract terms or priority require higher approval |
| Obtain supplier recovery commitment | Procurement lead | Buyer | Supplier and production planner | Capacity or commercial terms exceed delegation |
| Activate alternative network path | Network operations lead | Site and transport teams | Quality and trade compliance as applicable | Capacity, eligibility, or spend is unresolved |

This is an example, not a universal authority matrix.
Replace role names, thresholds, and approval paths with the organization's actual delegation of authority.

### Layer-by-layer questions for an operational handoff

The following prompts make the general procedure usable during a short handoff between shifts.
They are deliberately ordered from the immediate decision to evidence, not from available screens to possible stories.
Record the answer and its source; an unanswered prompt is an explicit uncertainty, not a reason to invent a value.

#### Shipment handoff

1. Which shipment and leg are relevant to this decision?
   Name the shipper and carrier identifiers and any crosswalk used.
2. What is the last actual milestone and when did it occur?
   State the source and event time, not only when the dashboard refreshed.
3. What arrival is predicted and at which location?
   Record the estimate issue time, any interval, and whether it includes remaining intermediate stops.
4. What goods does the manifest associate with the leg?
   Identify quantities, units, and any missing order or lot relationships.
5. What does the carrier still need to confirm?
   Give the transport owner a deadline before the booking or customer-communication cutoff.

If the carrier has only a vehicle-level feed, distinguish the vehicle from the load.
If a shipment has been transloaded, use the most recent supported association between goods and conveyance.
If no recent evidence exists, the right status is unknown or stale, not automatically delayed or on time.

#### Inventory handoff

1. What site and item does the balance describe?
   Name the unit of measure and snapshot time.
2. Which statuses are unavailable for the immediate decision?
   Include quality hold, damage, expiry, or other locally applicable restrictions.
3. Which stock is already reserved?
   Explain whether the source balance already excludes those reservations.
4. Which inbound quantities are not yet included in on-hand?
   Record their expected availability after receiving and release, not only their dock arrival.
5. What count or status discrepancy could materially alter the choice?
   Assign the site owner to confirm it before action if the decision allows time.

For example, a site can have 100 physical cases and zero free cases if all saleable stock is reserved.
That does not mean the site is empty; it means no stock is available for a new commitment under the stated allocation policy.
Conversely, a system may show a positive free balance that cannot be used because the associated lot is not eligible for the destination.

#### Order handoff

1. Which order lines are affected, and what quantities remain open?
   Separate requested, accepted, allocated, shipped, and delivered quantities.
2. What is the exact customer promise?
   Include the quantity, destination, promised milestone, date, and time zone.
3. Are partial deliveries, substitutions, or revised dates allowed?
   Cite the applicable customer agreement or obtain customer-service confirmation.
4. What is the allocation priority when supply is insufficient?
   State the approved policy rather than assuming that all customers are interchangeable.
5. Who may communicate or change the promise?
   Give that person an evidence cutoff and a communication deadline.

The quantity remaining to deliver is not always the same as the quantity remaining to allocate.
An order may be allocated but not shipped, shipped but not yet delivered, or partially delivered with a balance still open.
Choose the quantity appropriate to the decision and do not subtract the same movement at more than one stage.

#### Supplier handoff

1. Which legal supplier and physical facility support the item?
   Distinguish the contracting entity from the production or dispatch site.
2. What quantity and date has the supplier actually accepted?
   Keep the purchase-order request and supplier confirmation separate.
3. What goods have been released, and what evidence supports the release?
   A planned production completion is not an actual dispatch.
4. What recovery option has the supplier specifically offered?
   Capture item, quantity, release date, constraints, and the expiry of the offer.
5. Who can agree to changed commercial terms?
   Route price, minimum quantity, or contract changes through procurement authority.

If the supplier has not responded, mark the recovery quantity unknown.
If a sub-tier dependency is absent from the map, mark the dependency coverage incomplete.
Neither absence justifies asserting that the supplier has no exposure.

#### Network handoff

1. Which alternative origin, destination, and mode are proposed?
   Specify the entire path, including transfer and final-delivery legs.
2. Is eligible stock truly free at the alternative origin?
   Protect its existing commitments and confirm site handling capacity.
3. What is the latest feasible booking and dispatch time?
   Include calendars and any time-zone differences.
4. Is the lane capacity confirmed for the relevant product and quantity?
   Record equipment, temperature, quality, and destination restrictions where applicable.
5. Who accepts the incremental cost and risk?
   Identify the approver, executor, and fallback if a booking fails.

The output of this handoff is an actionable decision record, not a complete enterprise model.
When a prompt changes the outcome, make the associated source and assumption visible in the case.
When it does not change the outcome, avoid delaying the action merely to collect a more attractive dashboard field.

### Choosing the first improvement

A team usually cannot improve all five layers at once.
Choose a narrow decision where a missing relationship repeatedly prevents action.
For example, if carriers provide timely milestones but shipment lines cannot be connected to customer orders, improving map refresh frequency does not resolve the principal decision gap.
The first improvement may instead be a quantity-bearing shipment-to-order link and an unmatched-record queue.

Define a baseline over all in-scope shipments and order lines, not only those that already have complete links.
Measure the proportion linked, the time needed to identify affected commitments, and the number of cases that reach an authorized decision before their cutoff.
Track service and incremental action cost as outcomes, while acknowledging that multiple process changes may affect them.
Review a sample of linked records against source documents to detect links that are present but wrong.

Assign a business owner for the decision and a data owner for the relationship.
The business owner defines what makes an impact actionable and who can spend.
The data owner resolves mapping defects and agrees on the correction path with trading partners.
Neither role can substitute for the other: perfect data without authority creates an informative queue, and authority without trustworthy data invites poor decisions.

Expand the pilot only after the team can explain its denominator, uncertainty, and actual operational response.
The purpose of expansion is to improve a repeatable decision, not to maximize the number of feeds connected.

## 6. Worked example

### Business situation

A distributor serves two customers from East distribution center (DC).
The item is a standard ambient product measured in cases; one case is the indivisible planning unit.
At 08:00 on Monday, East has a delayed inbound shipment and must decide whether to transfer stock from West DC.
All times below are local to both DCs; assume there is no time-zone difference.
Amounts are in US dollars (USD).

The customer-facing commitments are:

| Customer order line | Promised quantity | Promised delivery | Existing East allocation | Partial delivery accepted? |
|---|---:|---|---:|---|
| A-14 | 84 cases | Tuesday 16:00 | 84 cases | No |
| B-29 | 67 cases | Tuesday 16:00 | 36 cases | Yes |
| Total | 151 cases | Tuesday 16:00 | 120 cases | Mixed |

The allocation policy gives A-14 priority because its contractual terms do not permit a partial delivery.
B-29 may receive a partial delivery, but its customer must be informed of any shortfall.
These are example policy assumptions, not general rules about customers.

East's inventory ledger at 08:00 is:

| Component | Quantity | Interpretation |
|---|---:|---|
| Book on-hand | 138 cases | Physical book balance at East |
| Quality hold | 11 cases | Not saleable until inspection release |
| Damaged | 7 cases | Not saleable |
| Saleable | 120 cases | 138 − 11 − 7 cases |
| Existing allocations | 120 cases | 84 to A-14 and 36 to B-29 |
| Free saleable | 0 cases | 120 − 120 cases |

The 11 held cases are not assumed to be released by the deadline.
Damaged and quality-held quantities are disjoint.
The example does not add the 120 allocated cases again when computing free stock.
The customer demand remaining beyond those allocations is 151 − 120 = **31 cases**.

### Shipment and supplier evidence

Supplier S-8 confirmed a purchase-order line for 96 cases.
Its advance shipment notice assigns all 96 cases to shipment SH-41.
The carrier reported departure on Sunday at 18:00.
At 07:45 Monday, its latest carrier ETA at East is Tuesday 15:00, with no calibrated arrival interval supplied.
East requires four hours after actual arrival for receipt, release, picking, and outbound departure; the customer's delivery route requires another two hours.
Under that point ETA the earliest modeled customer arrival is Tuesday 21:00, after the Tuesday 16:00 promise.
The ETA is a forecast, not a confirmed arrival.
The inbound 96 cases are not included in East's 138-case on-hand balance.

Supplier S-8 reports that the 96 cases have shipped and cannot be split or recalled at the origin.
It has not confirmed a second production release before Tuesday.
That is a supplier constraint for this scenario, not evidence of a general factory capacity limit.
The buyer owns verification of any new supplier option.

### Network alternative

West DC has 54 saleable, unreserved cases after its own confirmed allocations.
The network operations lead confirms a lane with capacity for 40 cases, a Monday 11:00 booking cutoff, and arrival at East by Tuesday 08:00 if dispatched as planned.
East needs four hours after arrival for receipt, picking, and dispatch; the customer route needs two more hours.
Thus the planned customer arrival is Tuesday 14:00, two hours before the promise.
The transfer is feasible under these stated times; it is not guaranteed.
The lane is eligible for the item and customer destinations.

The lane charges a fixed USD 110 plus USD 6 per transferred case.
Assume the same departure and fixed charge apply to any quantity from 1 to 40 cases.
The transfer does not compromise West's existing commitments because the 54 cases are already net of West's reservations.
The inventory planner has authority to allocate stock; the transport manager can approve expedite spend up to **USD 400 per incident**.
The customer-service manager owns any customer promise change.

### Reconciliation and decision

At 08:00, A-14 has all 84 cases allocated and B-29 has 36 of 67 cases allocated.
The relevant shortage is therefore 31 cases, not 151 cases and not 96 cases.
The inbound shipment has 96 expected cases but cannot be treated as timely supply under its point ETA and processing assumptions.
West's 54 free cases exceed the need, and the lane's 40-case capacity also exceeds it.
The minimum transfer that fills both commitments is 31 cases.

The incremental transfer cost is:

```text
fixed charge                         USD 110
31 cases × USD 6/case               USD 186
incremental transfer cost           USD 296
```

USD 296 is below the USD 400 per-incident approval limit.
If there is no transfer and the forecast timing holds, B-29 is short 31 cases at the promise time.
The scenario does not assign a monetary value to the shortfall, and therefore does not claim a positive financial return.
The business accepts USD 296 for the service improvement under its stated policy.
If the transfer fails, customer service must reconsider the promise; the model does not guarantee delivery.

| Option | Supply usable by customer promise under assumptions | Incremental cost | Main issue |
|---|---:|---:|---|
| Wait for SH-41 | 120 cases | USD 0 | B-29 short 31 cases if ETA holds |
| Transfer 31 cases | 151 cases | USD 296 | Transfer and processing must meet plan |
| Transfer 40 cases | 160 cases | USD 350 | Nine extra cases are not needed for these commitments |
| Request supplier second release | Unknown | Unknown | No supplier confirmation or timing evidence |

The 160-case figure is total supply available for the two orders plus nine excess cases; it is not a claim that orders require 160 cases.
A 40-case transfer costs USD 54 more than a 31-case transfer and provides no additional benefit for these two commitments under the example assumptions.
There may be other legitimate demand, but it is outside the case and must not be invented.

### Who acts and when

1. At 08:10, the inventory planner approves reserving 31 West cases for B-29, subject to West confirmation.
2. The West site confirms stock and pick capacity by 09:00.
3. The transport manager approves the USD 296 incident cost and the transport controller books the lane before 11:00.
4. The planner records a transfer allocation of 31 cases against B-29; it remains expected supply until received and released.
5. Customer service monitors the transfer and retains the original promise; no new customer promise is made from the ETA alone.
6. East confirms actual receipt and release on Tuesday; the outbound operator confirms dispatch.
7. Customer service verifies delivery and closes or escalates the case.

If West cannot confirm the 31 cases or the lane misses the cutoff, the proposed transfer is no longer a confirmed option.
The inventory planner escalates the shortage and customer service contacts B-29 before the agreed communication deadline.
The original inbound 96 cases, once received and released, replenish inventory; they do not retroactively become on-time supply.

### Facts, forecasts, plan, target, and commitment

| Category | Case item |
|---|---|
| Observed record | East's 08:00 book balance and recorded allocations |
| Supplier commitment | S-8 accepted 96 cases; it reports these shipped |
| Forecast | Carrier ETA Tuesday 15:00, with no supplied interval |
| Internal plan | Transfer 31 cases and reach the customer Tuesday 14:00 |
| Customer commitment | Deliver A-14 and B-29 by Tuesday 16:00 |
| Target | Honor both customer commitments without unauthorized spend |
| Unknown | Actual arrival, release, execution, and delivery outcome |

This separation prevents a planned transfer or predicted ETA from being reported as an actual delivery.
It also makes the business decision auditable at the time it was made.

## 7. Code, query, or calculation

The following Python 3 example reproduces the case's deterministic allocation and cost calculation using only the standard library.
It is a decision aid, not an enterprise available-to-promise engine.
Copy it into a file and run it with `python visibility_case.py`.

```python
from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class Order:
    order_id: str
    promised_cases: int
    allocated_cases: int
    priority: int  # lower number is allocated first


@dataclass(frozen=True)
class Stock:
    on_hand: int
    quality_hold: int
    damaged: int
    allocated: int

    def free(self) -> int:
        # Status quantities are disjoint; allocations are of saleable stock.
        saleable = self.on_hand - self.quality_hold - self.damaged
        if min(self.on_hand, self.quality_hold, self.damaged, self.allocated) < 0:
            raise ValueError("negative stock quantity")
        if saleable < 0 or self.allocated > saleable:
            raise ValueError("inconsistent inventory ledger")
        return saleable - self.allocated


def allocate_shortage(orders: list[Order], incoming_cases: int) -> dict[str, int]:
    """Assign free incoming cases against unmet demand in policy priority order."""
    if incoming_cases < 0:
        raise ValueError("negative incoming quantity")
    result = {}
    remaining = incoming_cases
    for order in sorted(orders, key=lambda item: (item.priority, item.order_id)):
        if not 0 <= order.allocated_cases <= order.promised_cases:
            raise ValueError(f"invalid allocation for {order.order_id}")
        needed = order.promised_cases - order.allocated_cases
        assigned = min(needed, remaining)
        result[order.order_id] = assigned
        remaining -= assigned
    return result


def customer_arrival(site_arrival: datetime) -> datetime:
    # Four hours for site processing and two hours for outbound delivery.
    return site_arrival + timedelta(hours=6)


def main() -> None:
    orders = [Order("A-14", 84, 84, 1), Order("B-29", 67, 36, 2)]
    east = Stock(on_hand=138, quality_hold=11, damaged=7, allocated=120)
    assert sum(order.allocated_cases for order in orders) == east.allocated
    free_east = east.free()
    shortage = sum(order.promised_cases - order.allocated_cases for order in orders)
    shortage_after_free = max(0, shortage - free_east)

    west_free = 54
    lane_capacity = 40
    transfer_cases = min(shortage_after_free, west_free, lane_capacity)
    assignments = allocate_shortage(orders, free_east + transfer_cases)
    remaining_shortage = shortage - sum(assignments.values())

    fixed_usd = 110
    per_case_usd = 6
    transfer_cost_usd = (fixed_usd + per_case_usd * transfer_cases) if transfer_cases else 0
    spend_limit_usd = 400

    promise = datetime(2026, 5, 5, 16, 0)  # Tuesday local time
    inbound_eta = datetime(2026, 5, 5, 15, 0)
    transfer_eta = datetime(2026, 5, 5, 8, 0)
    inbound_customer_time = customer_arrival(inbound_eta)
    transfer_customer_time = customer_arrival(transfer_eta)

    print(f"East free saleable: {free_east} cases")
    print(f"Unmet commitments: {shortage} cases")
    print(f"Proposed transfer: {transfer_cases} cases")
    for order_id, cases in assignments.items():
        print(f"Additional allocation {order_id}: {cases} cases")
    print(f"Remaining shortage: {remaining_shortage} cases")
    print(f"Transfer cost: USD {transfer_cost_usd}")
    print(f"Within USD {spend_limit_usd} incident limit: {transfer_cost_usd <= spend_limit_usd}")
    print(f"Inbound customer time: {inbound_customer_time:%a %H:%M}; on time: {inbound_customer_time <= promise}")
    print(f"Transfer customer time: {transfer_customer_time:%a %H:%M}; on time: {transfer_customer_time <= promise}")


if __name__ == "__main__":
    main()
```

Expected output:

```text
East free saleable: 0 cases
Unmet commitments: 31 cases
Proposed transfer: 31 cases
Additional allocation A-14: 0 cases
Additional allocation B-29: 31 cases
Remaining shortage: 0 cases
Transfer cost: USD 296
Within USD 400 incident limit: True
Inbound customer time: Tue 21:00; on time: False
Transfer customer time: Tue 14:00; on time: True
```

The code assumes that the given allocations are valid and refer to these two orders only.
It treats the transfer ETA as a deterministic scenario, not a guaranteed outcome.
It does not verify actual stock, booking, calendars, shelf life, customer acceptance, or authorization.
The `assert` checks a case consistency assumption; production validation should not depend on assertions being enabled.
It calculates a proposed allocation; no source system is updated and no customer promise is made.
If the quantity transferred is below the shortage, the code reports the residual rather than claiming full service.
In practice, use timezone-aware timestamps and explicit business calendars; the naive timestamps here intentionally share one local zone and assume continuous elapsed hours.

## 8. Data, standards, and regulatory context

### Data contracts by layer

| Layer | Minimum identifiers | Critical fields | Quality question |
|---|---|---|---|
| Shipment | Shipper shipment ID, carrier ID, leg ID | Event type, location, event and receipt time, ETA | Is this the right shipment and how old is the evidence? |
| Inventory | Item, site, status, lot where relevant | On-hand, held, allocated, timestamp, unit | Are statuses mutually exclusive and quantities comparable? |
| Order | Order ID and line ID, trading party | Quantity, unit, request, promise, allocation, fulfillment | Is the promise accepted and is the line relationship current? |
| Supplier | Supplier and facility IDs, item | Confirmed quantity/date, capability, evidence date | Is reported capacity available to this buyer and item? |
| Network | Node and lane IDs | Mode, capacity, transit time, calendar, restrictions | Is the alternative actually feasible and confirmed? |

An interface contract should state ownership, permitted use, refresh cadence, late-data handling, retention, and correction procedures.
It should also state whether a missing record means zero, not applicable, or unknown.
A null supplier capacity is not a zero-capacity confirmation.

Use a master-data crosswalk for partner identifiers and product units.
A global identifier can support interoperability, but a global code alone does not establish a correct link to a local system's record.
Maintain valid-from and valid-to times for mappings, and retain provenance when a mapping changes.
For quantity joins, document pack conversions and rounding rules; a case and an each are not interchangeable without a conversion.

### GS1 and event standards

[GS1 EPCIS](https://www.gs1.org/standards/epcis) defines a standard for sharing event data about the what, when, where, why, and how of products and assets.
It can help trading partners exchange visibility events across organizational boundaries.
EPCIS 2.0 provides a standardized event model and interfaces; adoption and implementation agreements determine what each participant actually sends.
It does not guarantee event completeness, accurate timestamps, inventory ownership, customer allocation, or permission to take a business action.

[GS1 identification keys](https://www.gs1.org/standards/id-keys) include identifiers for products, locations, and logistics units.
A Global Trade Item Number (GTIN), Global Location Number (GLN), and Serial Shipping Container Code (SSCC) can support cross-party identification where used consistently.
A GTIN identifies a trade item, not a particular shipment or customer order.
An SSCC identifies a logistics unit; it is not automatically a unique identifier for an entire order.
A GLN identifies a location or party according to the applicable GS1 rules; it does not itself describe operating capacity.

The [GS1 Global Traceability Standard](https://www.gs1.org/standards/gs1-global-traceability-standard/current-standard) describes traceability principles and information sharing across supply chains.
Traceability establishes relationships and event history for scoped objects; it does not by itself solve the tower's prioritization or spending decision.
Apply the edition and partner agreement relevant to the implementation.

### Commercial messages and internal systems

Electronic data interchange (EDI) and application programming interfaces (APIs) can carry order, shipment, and inventory messages.
Message format alone does not make a supplier confirmation enforceable or a shipment notice an actual receipt.
Document which party sends each message, what the message means, and which field is authoritative for each decision.
Where a message replaces a previous version, retain the version and effective time.

An enterprise resource planning (ERP) system often holds orders and financial ownership; a warehouse management system (WMS) often records warehouse movements and statuses; a transportation management system (TMS) often records transport plans and execution.
These are common roles, not universal guarantees.
Reconcile definitions with actual local configuration before deciding which system is authoritative.

### Regulation and data rights

Regulatory obligations depend on the product, jurisdiction, parties, and date; the standards above are not automatically binding law.
For regulated food in the United States, the [US Food and Drug Administration's Food Traceability Rule page](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods) describes requirements for persons covered by the rule and lists relevant compliance updates.
Check current FDA materials and legal advice for applicability and compliance dates; an EPCIS feed alone is not a compliance determination.
The rule does not prescribe every control-tower decision or authorize sharing data with every participant.

Location feeds may identify drivers or reveal sensitive supplier operations.
Apply applicable privacy and employment rules and contractual restrictions to the specific facts.
Define access by purpose, retain only what is necessary under the organization's policy, and log access to sensitive information.
A commercial partner's willingness to share one operational milestone does not imply permission to disclose its customer list or production capacity to competitors.

### Operational data-quality measures

Define each measure before using it as a target:

| Measure | Numerator | Denominator | Interpretation limit |
|---|---|---|---|
| Shipment milestone coverage | In-scope shipments with required usable milestone | All in-scope shipments due in period | A milestone can be wrong despite being present |
| Fresh evidence rate | In-scope objects within stated age threshold | All in-scope objects requiring freshness | Threshold differs by decision deadline |
| Order-link coverage | Impacted shipment quantity linked to order lines | Total impacted shipment quantity | Linked quantity may still be incorrect |
| Inventory status completeness | Stock records with valid status | All in-scope stock records | Valid code does not prove physical count |
| Supplier confirmation coverage | In-scope lines with current supplier confirmation | All in-scope supplier lines | Confirmation is not a guarantee |
| Network feasibility coverage | Proposed options with verified lane and constraints | All proposed options | A verified lane can still fail in execution |

Report the time period, object population, source, and exclusions with every percentage.
Stratify by partner and lane so a high average cannot hide a critical blind spot.

## 9. Industry practice and case evidence

### Case 1: Walmart and food traceability

Walmart publicly described its work with IBM on food traceability, including tracing produce through supply-chain records; see [Walmart's 2018 announcement](https://corporate.walmart.com/news/2018/09/24/walmart-announces-new-food-traceability-initiative-using-blockchain-technology).
The announcement describes requirements for suppliers of certain leafy greens to contribute traceability information.
It is company evidence about a specific initiative, not an independent evaluation of every downstream control-tower outcome.

The visibility lesson is that a product-to-source trace requires more than a current vehicle position.
Lot, location, and transaction relationships allow a firm to narrow the scope of an investigation.
A shipment layer may help locate product in transit, while an inventory layer identifies stock at sites and an order layer identifies affected customers.

**Inference:** if a distributor can link affected lots to customer-order lines, it can direct a targeted customer notification more effectively than a party with only a broad supplier name.
That conclusion depends on complete and accurate links; the cited announcement does not establish the performance of the hypothetical distributor in this lesson.
Traceability also does not remove the need for a quality or food-safety owner to decide the response.

### Case 2: Maersk and remote container monitoring

Maersk describes remote container management for refrigerated cargo, including monitoring conditions and location, in its [Remote Container Management product information](https://www.maersk.com/transportation-services/remote-container-management).
This is company product information and should be read as a description of a service, not independent evidence of a universal cost or service improvement.

A temperature observation can be highly relevant to a perishable shipment.
It is not, by itself, a disposition decision: a quality owner needs the product-specific limits, duration, sensor reliability, and applicable handling procedures.
The inventory layer must distinguish held from saleable goods, while order visibility identifies potentially affected commitments.
Supplier and network information can identify replacement or alternative supply, subject to confirmation.

**Inference:** connecting a documented temperature excursion to a quality hold and an affected order can reduce the chance that an operator treats a physically arriving shipment as automatically usable.
No quantitative benefit is asserted here.

### Case 3: DHL and supply-chain risk monitoring

DHL describes its supply-chain risk management offering and the use of visibility to assess disruption exposure in its [Resilience360 material](https://www.dhl.com/global-en/home/insights-and-innovation/thought-leadership/trend-reports/resilience360.html).
Commercial material describes offered capabilities and should not be treated as independent validation of a particular customer's avoided loss.

The transferable practice is to connect an external disruption to facilities, suppliers, lanes, and business exposure rather than forward every news item to every planner.
A network layer can show which sites or lanes may be affected.
Supplier and order evidence is still required to decide which customer commitments are at risk.
A mapped site in a disruption zone does not prove that it is closed or that a particular shipment is delayed.

**Inference:** a tower that records source confidence and assigns a supplier-contact owner can turn a network warning into a verifiable operational case.
The case is not closed until the owner confirms the relevant condition and chooses an action.

### What to take from public cases

Named cases illustrate specific practices, not proof that a technology or operating model transfers unchanged to every firm.
Ask what was actually observed, the scope and denominator, who supplied the evidence, and whether the result was independently evaluated.
Avoid converting a vendor's capability claim into a measured return on investment.
For your own implementation, compare before and after using consistent populations and account for seasonality, process changes, and the difficulty of observing the no-intervention outcome.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Map mistaken for delivery | Vehicle is near destination but no receipt | Location is a proxy for a different event | Premature customer promise | Require actual receipt and release evidence |
| Old event shown as current | Position timestamp is much older than dashboard refresh | Ingestion time masks event age | False confidence | Display both clocks and stale threshold |
| Double-counted inbound | Stock rises when a receipt arrives without reducing inbound | Expected receipt and actual receipt coexist as supply | Overpromising | Use mutually exclusive states and reconcile flows |
| Reserved stock treated as free | Same cases appear in multiple allocations | Reservation semantics differ between systems | Competing commitments | Define free stock and reconcile reservations |
| Held stock treated as saleable | Negative quality outcomes follow receipt | Physical presence confused with release | Unsafe or invalid fulfillment | Enforce disposition status and quality approval |
| Shipment joined to every order | Sum of order-linked cases exceeds shipment cases | Many-to-many join omits linked quantity | Exaggerated impact | Use quantity-bearing links and residual checks |
| Missing link treated as no impact | Unmatched shipment lines are excluded | Inner join drops unknowns | Missed at-risk customers | Retain unmatched records and assign resolution owner |
| Request called a promise | Requested and accepted dates differ | Commercial state is ignored | False breach or false reassurance | Store requested and confirmed dates separately |
| Supplier capacity assumed available | Aggregate capacity has no item or customer scope | Theoretical capacity confused with committed supply | Failed recovery plan | Obtain specific written confirmation |
| Network route assumed feasible | Map line lacks capacity or calendar | Topology mistaken for executable flow | Missed booking and commitment | Validate lane, cutoff, eligibility, and capacity |
| Prediction treated as guarantee | One ETA appears without age or interval | Estimate presented as actual | Unapproved or unrealistic promise | Label forecast and escalate uncertain cutoff cases |
| No decision owner | Case remains open in shared queue | Responsibility and authority are unclear | Action deadline passes | Assign named role, fallback, and deadline |
| Approval after booking | Transport action exceeds delegated limit | Decision and execution rights conflated | Unauthorized cost | Enforce approval before execution |
| Duplicate alerts | Same delay appears for each event | No case correlation or lifecycle | Attention fatigue | Group by affected decision and update one case |
| False precision in economics | Avoided loss reported as exact | Unobservable counterfactual treated as fact | Misleading investment decisions | Show scenario assumptions and ranges |
| Incomplete network map | All visible suppliers appear independent | Sub-tier or common-port dependencies absent | Concentration risk missed | Track mapping coverage and unknown edges |
| Wrong time zone | Events appear to arrive before departure | Timestamp converted without source zone | Incorrect lateness and sequence | Preserve original zone and convert explicitly |
| Plan mistaken for execution | Transfer is allocated but not booked | Proposed action used as completed evidence | Unmet promise | Capture booking and receipt confirmations |
| Stale authority matrix | Employee approves under an obsolete threshold | Policy change not propagated | Governance breach | Version and periodically review delegation |
| Sensitive data overshared | Multiple partners see detailed capacity | Access based on convenience rather than purpose | Commercial or privacy harm | Apply least privilege and contractual controls |

A red flag is a reason to investigate, not automatic proof of a bad decision.
The mitigation must address the underlying definition, source, or authority problem rather than merely suppress the visible symptom.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test it |
|---|---|---|---|
| Item units are comparable | Quantities must be added and allocated | Mixed pack sizes or incompatible grades | Reconcile item master and conversion rules |
| Inventory statuses are disjoint | Held and damaged units must not be subtracted twice | A damaged lot also appears in the hold bucket | Compare record-level status assignments |
| Allocations refer to saleable stock | Free quantity must not go negative or double count | Reservations include in-transit or held stock | Trace reservations to stock IDs and statuses |
| Order promises are current | Impact depends on the accepted deadline | Customer accepted a change in another channel | Reconcile customer confirmation and order history |
| Shipment-to-order links carry quantities | Split loads require proportional impact | One ID links to multiple lines without quantities | Compare linked sums with shipment manifest |
| Event and receipt times are meaningful | Freshness and sequence require clocks | Device clock drift or missing zone | Compare source clocks and known milestones |
| ETA is relevant to the named node | Arrival calculation depends on the endpoint | ETA means port arrival rather than DC arrival | Inspect source definition and route |
| Processing time is representative | A feasible route must include site work | Backlog, inspection, or closure changes duration | Compare actual elapsed times by site and shift |
| West stock is genuinely free | Transfer must not harm existing commitments | Other allocations are absent or stale | Confirm reservations and site manager approval |
| Lane capacity is usable | Map alternative must be executable | No carrier booking or equipment availability | Obtain booking confirmation before cutoff |
| Supplier response is specific | A recovery option needs committed supply | Generic capacity statement without item/date | Request item-level quantity and release date |
| Decision rights remain current | Approved action needs delegated authority | Spend policy or role has changed | Verify current delegation and escalation contacts |
| Data access is permitted | Cross-party analysis depends on lawful sharing | Contract limits onward use or privacy rights apply | Review agreements and applicable policy |
| Scenario costs are incremental | Options must be compared on the same basis | Cost already included in baseline or omitted | Reconcile with finance and commercial terms |

The worked example is deterministic and has one item, two orders, one alternative lane, and one shared local time zone.
Real networks can have uncertain travel times, competing demand, multiple product substitutions, calendars, and capacity interactions.
The framework identifies the evidence and authority needed; it does not solve an arbitrary network optimization problem.

## 12. Exercises

### Exercise 1 — Classify questions

**Time:** 10 minutes.
**Deliverable:** a five-row table assigning each question to one or more layers and stating one missing fact.

1. Where was container X last reported?
2. How many saleable cases of item P are free at site D?
3. Which accepted customer lines depend on shipment X?
4. Can supplier S release 50 additional cases this week?
5. Which other site can deliver the item before the customer cutoff?

**Acceptance criteria:** identify the primary layer for each question, add another layer where needed, and avoid presenting a forecast or map edge as proof of execution.

### Exercise 2 — Separate facts from claims

**Time:** 10 minutes.
**Deliverable:** classify the following as observed record, forecast, plan, commitment, or target, and identify its owner.

- Carrier predicts arrival at 14:00 Tuesday.
- Customer accepted delivery of 60 cases by Wednesday 12:00.
- Warehouse recorded receipt of 52 cases at 09:15.
- Planner proposes transferring eight cases from another site.
- Manager aims for 98% on-time delivery this month.

**Acceptance criteria:** use each category once and do not describe receipt as quality release unless separately supported.

### Exercise 3 — Draw the object model

**Time:** 20 minutes.
**Deliverable:** an annotated diagram connecting a supplier facility, purchase-order line, shipment, receipt, inventory record, customer-order line, and alternative network lane.

Include one shipment serving two order lines and one order line split across two shipments.
Show where linked quantities, event times, and effective dates belong.

**Acceptance criteria:** many-to-many links are explicit; no implied one-to-one relationship; each edge has enough information to prevent quantity multiplication.

### Exercise 4 — Calculate free stock

**Time:** 10 minutes.
**Deliverable:** a calculation in cases and an explanation of the reservation convention.

At 07:00, a site has 215 cases on hand, including 13 quality-held and 9 damaged cases.
These two statuses are disjoint.
There are 146 cases reserved from the remaining saleable stock.

**Acceptance criteria:** subtract each exclusion once, report saleable and free cases, and state the 07:00 snapshot.

### Exercise 5 — Calculate a time-bounded shortage

**Time:** 15 minutes.
**Deliverable:** a two-row supply ledger and shortage calculation.

A customer has a confirmed demand of 90 cases by 17:00.
The site has 38 unreserved saleable cases available now.
A shipment of 45 cases is forecast to arrive at 13:00; receipt and release require two hours.
An additional 20 cases are forecast to arrive at 16:00 and require the same processing time.
Assume delivery from the site takes one hour and all forecasts occur exactly as stated for this exercise.

**Acceptance criteria:** include only supply that can reach the customer by 17:00, calculate the shortage, and distinguish forecast from actual.

### Exercise 6 — Compare a transfer decision

**Time:** 20 minutes.
**Deliverable:** a recommendation, incremental cost, and one sensitivity or limitation statement.

A site is short 23 cases before a customer promise.
An approved alternative site has 30 free cases and a confirmed 25-case lane capacity.
The transfer costs USD 95 fixed plus USD 7 per case.
The buyer can approve up to USD 300 per incident.
Assume arrival and processing meet the promise if the transfer is booked by noon.
No reliable lost-margin or penalty estimate is available.

**Acceptance criteria:** calculate the smallest full-service transfer and cost, compare it with authority, and avoid claiming a quantified return.

### Exercise 7 — Design ownership and escalation

**Time:** 20 minutes.
**Deliverable:** a role matrix for a late inbound shipment that may affect two customers.

Include decision owner, executor, customer communication owner, escalation trigger, fallback, and deadline.
One proposed action exceeds the planner's spending authority.

**Acceptance criteria:** approval precedes booking, customer promise changes have an authorized owner, and no responsibility is assigned only to an unnamed team.

### Exercise 8 — Audit a dashboard

**Time:** 20 minutes.
**Deliverable:** at least five defects, each with consequence and correction.

The dashboard shows “on time” because it refreshed at 09:00.
Its last carrier position is from 22:00 the prior day.
It counts 100 in-transit cases and the same 100 cases as received.
It drops shipment lines without a customer-order match.
It labels an unconfirmed alternative lane as “available.”

**Acceptance criteria:** identify clock, double-counting, unmatched-record, and feasibility problems; add one decision-rights or evidence-quality issue.

### Exercise 9 — Specify a pilot

**Time:** 25 minutes.
**Deliverable:** a one-page pilot charter for one item family, one lane, and two customer-service teams.

Include the business decision, five-layer evidence, scope, owner, baseline, measures with denominators, access controls, and stop conditions.

**Acceptance criteria:** distinguish observations from commitments, include an execution confirmation, and name at least one missing-data escalation.

### Exercise 10 — Critique a public claim

**Time:** 20 minutes.
**Deliverable:** a short evidence assessment of one source in section 9.

State what the source actually claims, its evidence type, what it does not establish, and one inference relevant to a tower.
Propose a measurement plan that would test the inference in your own organization.

**Acceptance criteria:** label inference, avoid attributing an invented outcome to the source, and define the comparison population and period.

## 13. Self-check answers

### Exercise 1

1. Shipment visibility provides the last reported position; the event time and carrier source are needed to judge freshness.
2. Inventory visibility provides stock by site and status; reservations and exclusions are needed for free stock.
3. Shipment and order visibility are both needed, joined through quantity-bearing links; a manifest or allocation relationship may be missing.
4. Supplier visibility is primary; obtain a specific item, quantity, date, and confirmation rather than relying on theoretical capacity.
5. Network and inventory visibility are necessary; order timing and lane capacity determine whether the alternative can satisfy the promise.

### Exercise 2

- Predicted arrival is a **forecast**, owned operationally by the transport function and sourced from the carrier or prediction service.
- Accepted customer delivery is a **commitment**, owned for customer communication by the authorized commercial or service role.
- Warehouse receipt is an **observed record**, owned by the receiving operation; it is not proof of quality release.
- Proposed transfer is an internal **plan**, owned by the inventory planner until approved and executed.
- The 98% monthly on-time figure is a **target**, owned by the manager responsible for the metric and its definition.

### Exercise 3 rubric

A strong diagram identifies both order-line and shipment-line quantities, distinguishes planned from actual receipt, and shows inventory status before allocation.
It shows the alternative lane as conditional on capacity and time.
It preserves timestamps and effective relationship dates.
A weak diagram joins order headers directly to a truck and treats every resulting row as the full shipment quantity.

### Exercise 4

Saleable stock is 215 − 13 − 9 = **193 cases** at 07:00.
Free stock is 193 − 146 = **47 cases** at the same snapshot time.
The calculation assumes the 146 reserved cases are included in the 193 saleable cases and that hold and damage do not overlap.
If the source already reports 47 cases as free, subtracting reservations again would be wrong.

### Exercise 5

The first inbound reaches the site at 13:00, becomes usable at 15:00, and reaches the customer at 16:00 under the stated forecast.
It contributes 45 cases to the modeled on-time supply.
The second inbound becomes usable at 18:00 and reaches the customer at 19:00, so it contributes zero cases by 17:00.
Modeled on-time supply is 38 + 45 = **83 cases**.
The modeled shortage is 90 − 83 = **7 cases**.
This is a scenario based on exact forecast arrivals, not a guarantee that 83 cases will be delivered.

### Exercise 6

Transfer **23 cases**, because that is the shortage and it is below both 30 free cases and 25 cases of lane capacity.
Cost is USD 95 + 23 × USD 7 = **USD 256**.
The amount is below the USD 300 per-incident limit by USD 44.
The transfer is feasible only under the stated timing and booking assumptions.
Without a credible estimate of avoided loss, the answer can describe a service improvement at USD 256, not a quantified return on investment.

### Exercise 7 rubric

A strong answer assigns a specific accountable role to the allocation, a budget owner to the excess spend, an executor to the booking, and customer service to any promise change.
It specifies a response deadline before the booking cutoff and a named fallback if the primary owner is unavailable.
A weak answer says “the tower handles it,” books first, and asks for approval later.

### Exercise 8

The 09:00 refresh is not evidence that the 22:00 carrier position is current; display event and ingestion times separately.
The same 100 cases cannot be both expected inbound and newly received in the supply ledger; use state transitions and reconcile totals.
Unmatched lines must remain visible as unknown exposure with an owner rather than disappearing from an inner join.
An unconfirmed lane is a conditional option, not available capacity; request a carrier or site confirmation.
The dashboard also needs an accountable person and deadline for the decision, not only a status color.
A sixth issue is that “on time” needs a defined promised milestone, relevant ETA, and uncertainty, not just a refresh time.

### Exercise 9 rubric

A strong charter names a decision that can be changed, a bounded population, a responsible role, and a baseline measured on the same population and period definition as the pilot.
It defines coverage and freshness denominators, records approval and execution separately, and limits access to partner data.
It states what happens when a shipment cannot be linked to an order and when the pilot should stop.
A weak charter measures only screen usage and assumes all linked data is accurate.

### Exercise 10 rubric

A strong assessment identifies the source as company, vendor, standard, or independent research; quotes or paraphrases only supported claims; and labels its own inference.
Its measurement plan specifies a comparable population, period, outcome, data source, and limitations of the counterfactual.
A weak assessment repeats a marketing claim as an independently measured causal benefit.

## 14. Interview questions

1. **What can a shipment location answer, and what can it not answer?**
   A strong answer identifies the transport object, event age, and source, and distinguishes location from receipt, release, and customer impact.
   A weak answer says a map proves delivery or stock availability.
2. **How do you compute free inventory without double counting?**
   A strong answer states the source balance definition, exclusions, reservations, and time horizon.
   A weak answer adds on-hand, in-transit, and allocated quantities without checking overlap.
3. **How do you connect a delayed shipment to affected orders?**
   A strong answer uses line-level, quantity-bearing relationships and reports unmatched quantities.
   A weak answer joins shipment and order headers and assigns the full load to every order.
4. **What does a supplier's capacity report mean?**
   A strong answer asks about product, facility, period, available components, competing demand, and confirmation.
   A weak answer treats nameplate capacity as a confirmed release.
5. **When is an alternative lane feasible?**
   A strong answer checks stock eligibility, booking cutoff, capacity, transit, processing, and customer deadline.
   A weak answer points to a line on a network map.
6. **How should event time and dashboard refresh time be displayed?**
   A strong answer preserves source event time and receipt time, applies a decision-specific freshness threshold, and marks stale evidence.
   A weak answer uses only the latest dashboard refresh timestamp.
7. **Who can change a customer promise?**
   A strong answer refers to the actual delegation, separates forecast from commitment, and records approval and communication.
   A weak answer lets the ETA model automatically revise the promise without policy.
8. **How would you choose whether to expedite?**
   A strong answer compares feasible incremental costs and service impacts under explicit uncertainty and authority limits.
   A weak answer claims a precise return from an invented avoided-loss number.
9. **How do you measure coverage?**
   A strong answer defines all in-scope objects as the denominator and stratifies by partner, lane, and period.
   A weak answer counts only connected shipments and calls the result complete coverage.
10. **How do you close an exception?**
    A strong answer requires decision, execution confirmation, actual outcome, and a record of unresolved uncertainty.
    A weak answer closes the case when a recommendation appears on a screen.

## 15. Key terms

Each term links to the detailed definition or procedure in this topic.

- [Allocation](#4-definitions-and-vocabulary) — assignment of supply to demand.
- [Available stock](#4-definitions-and-vocabulary) — stock after applicable exclusions.
- [Business object](#4-definitions-and-vocabulary) — entity whose state is tracked.
- [Commitment](#6-worked-example) — accepted external obligation with terms.
- [Consignment](#4-definitions-and-vocabulary) — goods associated with transport.
- [Coverage](#8-data-standards-and-regulatory-context) — usable evidence relative to an in-scope population.
- [Customer order](#4-definitions-and-vocabulary) — demand with requested and accepted terms.
- [Decision owner](#5-the-mechanics-step-by-step) — accountable role for the choice.
- [Dependency](#4-definitions-and-vocabulary) — relationship through which exposure can propagate.
- [ETA](#4-definitions-and-vocabulary) — estimated arrival at a specified location.
- [Event time](#3-the-core-mental-model) — time a source says the event occurred.
- [Exception](#4-definitions-and-vocabulary) — state requiring attention against a rule.
- [Forecast](#6-worked-example) — estimate, not a commitment.
- [Freshness](#8-data-standards-and-regulatory-context) — age of evidence relative to a decision.
- [Inventory status](#4-definitions-and-vocabulary) — disposition such as saleable or held.
- [Lane](#4-definitions-and-vocabulary) — potential transport connection with constraints.
- [Lot](#4-definitions-and-vocabulary) — traceable grouping of units.
- [Milestone](#4-definitions-and-vocabulary) — reported transport or operational event.
- [Network visibility](#3-the-core-mental-model) — evidence about nodes, lanes, and dependencies.
- [On-hand](#4-definitions-and-vocabulary) — recorded physical stock at a site.
- [Order visibility](#3-the-core-mental-model) — evidence about demand and commitments.
- [Plan](#6-worked-example) — intended internal action, not proof of execution.
- [Promise](#4-definitions-and-vocabulary) — agreed customer-facing quantity and timing.
- [Provenance](#4-definitions-and-vocabulary) — source and transformation history.
- [Receipt](#4-definitions-and-vocabulary) — record that goods arrived at a site.
- [Reservation](#4-definitions-and-vocabulary) — stock set aside for a purpose.
- [Shipment visibility](#3-the-core-mental-model) — evidence about movement and milestones.
- [Supplier visibility](#3-the-core-mental-model) — evidence about supplier commitments and capability.
- [Target](#6-worked-example) — desired metric, not an observed outcome.
- [Visibility layer](#4-definitions-and-vocabulary) — decision-oriented view of related evidence.

## 16. Sources

Sources are cited for the particular claims made above; a standard or company page is not independent proof of a tower's business benefit.
Consult current versions and applicable legal advice before implementation.

1. GS1, [EPCIS standard](https://www.gs1.org/standards/epcis). Event-sharing standard and implementation resources.
2. GS1, [GS1 identification keys](https://www.gs1.org/standards/id-keys). Descriptions of identifiers for products, parties, locations, and logistics units.
3. GS1, [Global Traceability Standard](https://www.gs1.org/standards/gs1-global-traceability-standard/current-standard). Traceability principles and information-sharing framework.
4. US Food and Drug Administration, [FSMA Final Rule on Requirements for Additional Traceability Records for Certain Foods](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods). Official regulatory information and updates; applicability and current dates must be checked.
5. Walmart, [Walmart Announces New Food Traceability Initiative Using Blockchain Technology](https://corporate.walmart.com/news/2018/09/24/walmart-announces-new-food-traceability-initiative-using-blockchain-technology), 2018. Company account of its produce traceability initiative.
6. Maersk, [Remote Container Management](https://www.maersk.com/transportation-services/remote-container-management). Company description of refrigerated-container monitoring.
7. DHL, [Resilience360](https://www.dhl.com/global-en/home/insights-and-innovation/thought-leadership/trend-reports/resilience360.html). Company material on supply-chain risk monitoring.
8. Hau L. Lee, V. Padmanabhan, and Seungjin Whang, “Information Distortion in a Supply Chain: The Bullwhip Effect,” *Management Science*, 1997, [DOI: 10.1287/mnsc.43.4.546](https://doi.org/10.1287/mnsc.43.4.546). Research on information distortion; it does not validate a specific control-tower product.
9. Martin Christopher and Helen Peck, “Building the Resilient Supply Chain,” *The International Journal of Logistics Management*, 2004, [DOI: 10.1108/09574090410700275](https://doi.org/10.1108/09574090410700275). Research framing supply-chain resilience and risk.

## 17. Links

- [Module 19 topic index](_index.md).
- [Module 19 survey and its lab sections](../../../course/19_control_towers_visibility_events.md).
- [Event streams, IoT, telematics, and EPCIS 2.0](02_event-streams-epcis.md) — planned sibling topic.
- [Event time versus processing time](03_event-time-processing-time.md) — planned sibling topic.
- [Decision rights and escalation](09_decision-rights.md) — planned sibling topic.
- [Simulation, optimization, and production labs](../../../course/30_simulation_optimization_production_labs.md) — related course labs.

Sibling topic links refer to files listed as planned in the module index; availability changes as the module is expanded.
