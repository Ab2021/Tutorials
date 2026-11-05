# Returns Networks: From Customer Handoff to Recoverable Inventory

> Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise  
> Topic 07 of 10  
> Estimated study time: 4 hours  
> Prerequisites: topics 01, 03, and 06  
> Version 2026.5

## 1. Why this topic exists

A customer who has received the wrong size wants an easy, trustworthy way to return it.
The retailer needs the right item back, an accurate refund, and a defensible inventory record.
Those needs are connected, but they are not the same decision.
A fast refund can precede physical inspection.
A physical receipt does not mean that an item is sellable.
A low-cost transportation route can leave seasonal merchandise waiting until its value has fallen.

A returns network determines where the customer can hand over an item, where it is inspected, and where it goes next.
Its design includes stores, carrier drop-off points, consolidation locations, distribution centers (DCs), repair partners, and recovery outlets.
It also includes the information that links each movement to the original sale and the authority to release a refund.
A network map without event ownership is not an operating design.

The commercial decision is not simply whether return-to-store costs less than return-to-DC.
Store return can save a parcel movement while consuming scarce selling-floor labor and space.
Direct return to a DC can pool specialist inspection but add transit, waiting, and customer inconvenience.
The economic comparison must use incremental costs, recovery value, timing, and a common customer and item population.
It must also separate costs already caused by the return from costs that change with the route.

The customer-experience owner needs a clear drop-off promise and refund status.
The store operations owner needs capacity at the counter and a safe holding area.
The reverse logistics owner needs predictable flows, scans, and carrier handoffs.
Finance needs a consistent basis for recovery and cost allocation.
Inventory control needs evidence before stock becomes available to promise (ATP).
Merchandising needs to know whether returned stock is likely to sell before its value decays.

A poor design creates both visible and hidden losses.
A customer may contact support repeatedly because an item was handed over but no event was recorded.
A store may accept merchandise it cannot legally or safely handle.
A DC may receive an unidentifiable parcel and hold the refund while the customer waits.
An item may be counted twice if a store makes it sellable before a transfer is reversed.
A refund may be paid twice when separate systems treat the same item as two returns.

This unit treats routing as an operational and economic decision, not as a fraud-scoring exercise.
Disposition alternatives and fraud controls have their own topics; here they are considered only insofar as they constrain network routing.
All numeric examples are illustrative, not measurements of a named company.

## 2. Learning outcomes

After this unit you can:

1. **Map** the physical, financial, and information paths of a return and identify an owner for each handoff.
   Exercise 1 asks for a three-lane map and control points.
2. **Classify** eligible customer drop-off and inspection nodes without confusing receipt with sellability.
   Exercise 2 asks you to route different item classes.
3. **Calculate** route-level incremental cost, expected recovery, and contribution per return under explicit assumptions.
   Exercise 3 reproduces the worked example.
4. **Measure** initiation-to-refund and handoff-to-sellable lead times with the correct event population.
   Exercise 4 tests timestamp handling and censoring.
5. **Choose** capacity and consolidation rules while recognizing the customer-service and inventory-age trade-offs.
   Exercise 5 requires a capacity and dispatch decision.
6. **Critique** a proposed experiment for selection bias, customer exclusion, and double-counted savings.
   Exercise 6 requests an evaluation plan.
7. **Design** an exception and reconciliation process for missing, damaged, or mismatched merchandise.
   Exercise 7 requires accountable owners and evidence.
8. **Recommend** a network policy with constraints, sensitivity, and decision rights.
   Exercise 8 asks for a management recommendation.

These outcomes cover both arithmetic and implementation.
Correct arithmetic on an infeasible route is not a sound recommendation.
A satisfactory design explains how a route operates on the worst ordinary day, not only in the average week.

## 3. The core mental model

A return is a linked set of customer, product, money, and information flows.
The flows can run at different speeds and through different organizations.
Draw all four before comparing routes.

```text
CUSTOMER       initiate --> choose eligible option --> hand off --> status/refund
                              |                         |
PHYSICAL          store / carrier / partner --------> receive --> inspect
                         |                                  |         |
                         +-- consolidation --> DC ----------+         +--> saleable stock
                                                                    +--> repair/recovery
                                                                    +--> controlled disposal
INFORMATION     authorization --> item ID --> custody scans --> condition --> inventory event
MONEY           original payment --> refund decision --> payment rail --> settlement
CONTROL         policy owner     custody owner     operations owner    finance/inventory owner
```

The diagram is a process model, not a requirement that every return visits every node.
A store can inspect and resell an eligible item locally.
A store can also accept the item solely as a collection point and ship it elsewhere.
A customer can mail directly to a DC or use a carrier partner.
A bulky or unsafe product may require a scheduled specialist pickup instead of either route.
Some low-value items may be refunded without physical return under an approved policy.
That last case has refund and control events but no reverse product movement.

The customer route and the physical destination must be recorded separately.
A return handed in at a store may ultimately be inspected at a regional DC.
Calling that transaction simply a "store return" obscures the later transport and handling.
Similarly, a prepaid label identifies an intended destination, not proof that a parcel was received.

A useful unit of analysis is an authorized return line: an original order line and a quantity being returned.
A parcel can contain multiple lines, and a line can be split across parcels.
A customer may return several units of the same SKU, each with a different condition.
A robust data model therefore connects authorization, item quantity, package, and scan events without assuming a one-to-one relationship.

The network objective is constrained contribution, not the shortest route.
For an eligible route, expected contribution equals recoverable value minus incremental reverse-flow costs.
The original payment and refund are relevant to total order economics, but cancel in a route comparison when the same refund is due under either route.
If the route changes the refund amount or the probability of a completed refund, include that difference explicitly.
Customer experience is often a constraint or an additional outcome because a credible monetary value for loyalty is rarely available.
Do not silently insert an unsupported lifetime-value benefit.

A route decision has at least three clocks.
The first clock is customer effort: how long until the customer can hand off the item.
The second is money: how long from the policy's refund trigger until the payment is initiated and then received.
The third is product: how long until inspection and a valid sellable or other disposition decision.
Optimizing one clock can make another worse.
A store handoff can improve customer access while a weekly store-to-DC shuttle delays sellable inventory.
A carrier's initial acceptance scan can trigger a refund while the item takes days to reach inspection.

The control principle is evidence before state transition.
Handoff evidence supports a custody change.
Inspection evidence supports a condition grade.
An authorized disposition supports inventory release or a recovery transfer.
A payment instruction and payment-provider confirmation support distinct refund statuses.
No single event proves all four.

## 4. Definitions and vocabulary

| Term | Definition | Common misuse | Unit or identifier |
|---|---|---|---|
| Return initiation | Customer requests or starts a return. | Treating a started request as a physical handoff. | UTC timestamp and return ID. |
| Return authorization | Policy decision allowing specified item quantities to enter a return process. | Assuming authorization proves eligibility after inspection. | Authorization ID and line quantity. |
| Return line | Specific original order line and quantity requested back. | Equating it with a parcel. | Order-line ID, SKU, units. |
| Return-to-store | Customer delivers an item to a retail location. | Assuming the store must resell it. | Store ID and handoff timestamp. |
| Return-to-DC | Customer or collection point sends product to a distribution center. | Ignoring intermediate carrier or consolidation nodes. | Destination node and receipt timestamp. |
| Drop-off partner | Third party accepting custody on the retailer's behalf. | Treating a partner scan as an inspection. | Partner ID, site ID, scan ID. |
| First carrier scan | Carrier records acceptance of a shipment. | Assuming it proves contents or condition. | Tracking ID and timestamp. |
| Custody | Accountable possession of an item or package. | Assuming a label creation event transfers custody. | From/to party and event time. |
| Receipt | A node records a physical item or container arriving. | Using trailer arrival as item-level verification. | Node ID, quantity, timestamp. |
| Inspection | Examination against documented condition criteria. | Assuming inspection always implies resale. | Inspector, grade, reason, time. |
| Triage | Initial routing or condition classification. | Treating a preliminary triage as final disposition. | Reason and grade codes. |
| Disposition | Authorized outcome for an item after review. | Equating it with the customer's refund decision. | Resell, repair, recovery, or disposal code. |
| Sellable inventory | Stock meeting physical, quality, and system requirements for sale. | Adding all received stock to ATP. | SKU, location, units, state. |
| Quarantine | Controlled status preventing normal sale or movement until review. | Treating a holding shelf as a complete system control. | Location, reason, aging. |
| Consolidation | Combining returns for a later movement or processing batch. | Assuming consolidation has no waiting cost. | Batch ID and dispatch time. |
| Reverse lead time | Elapsed time between specified return milestones. | Reporting one ambiguous average for all stages. | Hours or calendar days. |
| Refund initiation | Retailer submits an approved payment instruction. | Claiming the customer has received funds immediately. | Payment instruction ID, timestamp, currency. |
| Refund settlement | Payment rail or issuer completes the relevant transaction stage. | Assuming the merchant controls the issuer's posting time. | Provider status and timestamp. |
| Refund liability | Amount owed to the customer under the applicable policy and accounting treatment. | Treating it as identical to recovery value. | Currency per authorized return. |
| Recovery value | Proceeds or retained inventory value attributable to the returned item, net of stated adjustments. | Counting gross retail price as realized recovery. | Currency per unit at a defined date. |
| Incremental cost | Cost changed by selecting a route for the decision horizon. | Including fixed overhead that will not change. | Currency per return. |
| Contribution | Defined recovery or revenue less specified variable costs. | Calling a route comparison a complete profit-and-loss statement. | Currency per return. |
| Counterfactual | Outcome that would have occurred under another feasible policy. | Comparing users who self-select different channels as if randomized. | Defined population and period. |
| Right-censoring | Outcome has not yet occurred by the data extraction time. | Assigning incomplete returns a duration of zero. | Start time and observation cutoff. |
| Service-level agreement (SLA) | Explicit target or commitment for a defined event and population. | Reporting an internal target as a customer commitment. | Percent within hours or days. |
| Chain of custody | Evidence of successive transfers of responsibility. | Assuming a handoff occurred because a manifest exists. | Item or container and event chain. |
| Exception | A case requiring a controlled path outside the standard flow. | Closing a case by deleting the unresolved event. | Exception code, owner, age. |
| Forecast | An estimate of future demand or return volume. | Calling a forecast an approved capacity plan. | Units per interval and uncertainty. |
| Plan | An authorized allocation of capacity and actions. | Presenting the plan as a guaranteed outcome. | Staff-hours, units, period. |
| Target | An internal performance aspiration. | Calling an uncommunicated target a promise. | Metric and threshold. |
| Commitment | A promise made to a customer or contractual counterparty. | Using an average as a promise for each case. | Event, deadline, conditions. |

## 5. The mechanics, step by step

### 5.1 Define the decision and the population

1. State the decision in operational terms.
   Example: whether to enable store acceptance for eligible online apparel returns at 12 stores for eight weeks.
   Specify whether the store will perform final inspection or only collect items for a DC.
   The owner is the reverse logistics lead, with store operations and customer experience as decision partners.
2. Define the population before observing outcomes.
   Include the order channels, product classes, geography, return window, and customer eligibility.
   Exclude regulated, hazardous, oversized, or otherwise unsupported products only when the reason and alternative route are explicit.
   A network design that excludes difficult returns cannot be generalized to all returns.
3. Specify the decision horizon.
   A six-week pilot can use avoidable labor hours and carrier charges as incremental costs.
   A three-year network redesign may also change leases, systems, staffing, and asset commitments.
   Capacity that is fixed this week can become avoidable over a longer horizon.
4. Separate forecast, plan, target, and commitment.
   Forecast weekly returns and uncertainty from historical cohorts.
   Plan staff-hours and dispatch capacity for an agreed service level.
   Set an internal target for inspection backlog.
   Publish only customer commitments that the network and policy can support.
5. Define success and guardrails before routing changes.
   Include incremental economics per eligible return, customer effort, lead-time distribution, loss, and exception aging.
   A route should not be declared successful solely because its shipping invoice fell.

**Inputs:** order and return history, item constraints, sites, customer policy, forecast range.
**Output:** a versioned scope and decision charter with an accountable sponsor.
**Branch:** if the required return is not eligible for store handling, offer a compliant alternative rather than silently rejecting the customer.

### 5.2 Segment the reverse flow

1. Group items by handling needs rather than by selling channel alone.
   Apparel, sealed beauty products, temperature-sensitive goods, large furniture, electronics, and damaged items can require different facilities and equipment.
2. Add physical attributes that affect feasibility.
   Dimensions, weight, battery type, leakage risk, packaging condition, and serial-number requirements matter to transport and receipt.
   A product code by itself may not reveal the condition of an individual returned unit.
3. Add value and time sensitivity.
   A seasonal garment that loses value quickly may justify faster inspection.
   A durable basic item may tolerate a less frequent consolidation movement.
   The estimate must reflect actual expected recovery, not the original list price.
4. Segment demand by customer geography and access.
   Driving distance is not the same as an accessible or practical return option.
   Store hours, mobility, public transport, rural coverage, and printer access can affect usable choice.
5. Segment uncertainty as well as average volume.
   Peak-period return surges can arrive after the sales peak.
   A product recall or promotion can shift both volume and condition mix.
   Capacity and eligibility policies should identify who can revise them when the forecast fails.

**Inputs:** item master, condition and reason codes, customer access data, past return cohorts.
**Output:** feasible item-by-route matrix and volume forecast by node and time bucket.
**Branch:** uncertain or unsafe condition goes to a controlled exception workflow, not directly into regular sellable stock.

### 5.3 Map current and candidate flows

1. Draw the actual customer handoff, not just the address on the label.
   Record whether the customer packages the item, needs a label, and receives a receipt.
2. Draw every custody transfer.
   Distinguish store collection, carrier pickup, consolidation, DC receipt, and downstream recovery partner.
   Name the operator responsible for each transition.
3. Draw inventory state changes separately.
   A store receipt can create a return-in-transit or quarantine record.
   It should not automatically create an available unit if inspection has not occurred.
4. Draw the refund decision and payment events separately.
   The policy can trigger approval on a first scan, store acceptance, or inspection, subject to its controls.
   The finance and customer policy owners must approve the selected trigger.
5. Mark fixed versus optional nodes.
   Direct shipment to a DC is not inherently direct if the parcel moves through a carrier hub.
   For retailer-controlled cost and cycle time, identify which stages are material and observable.
6. Record alternative paths.
   A store may resell an unopened eligible item locally or consolidate a damaged item to a specialist.
   An unidentified unit may wait in exception handling instead of joining the normal flow.

**Inputs:** process observations, carrier contracts, store and DC standard operating procedures.
**Output:** candidate network diagrams with event, custody, and decision owners.
**Branch:** if an event cannot be associated with a return line or container, route it to reconciliation rather than inventing a match.

### 5.4 Establish policy and authorization

1. Display return eligibility and the actual customer choices before the customer commits to a route.
   Show the location, hours, label or packaging requirement, potential fee, and expected refund timing where applicable.
2. Connect the request to the original order line and available quantity.
   Account for partial shipments, prior returns, exchanges, and cancellations.
   Reject a duplicate authorization only after allowing an operator to resolve legitimate splits or corrections.
3. Assign a unique authorization and a stable item-level reference where the system supports it.
   Associate the authorization with the original payment and the selected route.
4. Create a customer receipt or confirmation at the handoff.
   The receipt should identify what was accepted and the next status without claiming inspection is complete when it is not.
5. Specify whether a refund is conditional on inspection.
   If approval occurs before inspection, record the associated exposure and exception process.
   Do not confuse the retailer's policy with a universal legal rule.

**Inputs:** policy version, order data, customer request, item eligibility.
**Output:** authorized quantities, route options, and a traceable customer instruction.
**Branch:** missing order information goes to a documented manual lookup or policy exception, subject to privacy and access controls.

### 5.5 Design store handoff and local processing

1. Confirm the store has trained staff, safe space, and a working system before accepting a return class.
   The reverse logistics owner approves route eligibility; the store manager owns execution at the site.
2. At the counter, identify the return and count the units accepted.
   Where policy requires, verify SKU, serial number, and visible condition.
   Record discrepancies rather than overwriting the customer's request.
3. Provide the customer with a timestamped receipt.
   If network connectivity fails, use a controlled offline procedure with unique records and later reconciliation.
   A handwritten receipt without a way to reconcile the inventory event is not an adequate steady-state design.
4. Separate local resale candidates from quarantined, hazardous, or DC-bound merchandise.
   Physical segregation and system status must agree.
   Avoid leaving returned items on the selling floor before the release criteria have been satisfied.
5. For local inspection, document the pass criteria and authorized staff.
   If the item passes, release it to the correct SKU and store location exactly once.
   If it fails or is uncertain, place it in the specified exception or transfer stream.
6. For collection-only stores, create a container and manifest.
   Record which return lines and quantities are in the container.
   Transfer custody when the carrier accepts it, not when the manifest is printed.
7. Capacity-plan counter time and back-room occupancy.
   Account for normal sales traffic and for return peaks by hour.
   If holding capacity is full, trigger a pre-agreed overflow or pickup procedure.

**Inputs:** authorization, item, inspection rules, store staffing and storage capacity.
**Output:** handoff evidence, item state, and either a local release or traceable onward shipment.
**Branch:** a mismatch between requested and presented item is an exception, not a reason to relabel the item without evidence.

### 5.6 Design return-to-DC and consolidation

1. Choose a customer handoff method and an authorized destination.
   Direct parcel, partner drop-off, and store consolidation have different customer effort and transport cost.
2. Decide whether the customer or a partner packages the goods.
   Specify acceptable packaging and carrier restrictions for the product class.
   A label alone does not guarantee safe transport.
3. Create a tracking identifier and connect it to the authorization.
   Support multiple return lines per parcel and multiple parcels per authorization.
4. Define the first evidence of physical custody.
   Label creation is an intention; an acceptance scan is a handoff event.
   A carrier may scan only the package exterior, so the contents remain unverified.
5. At a consolidation node, set dispatch triggers.
   A fixed schedule is simple but can lengthen waits for items that arrive just after dispatch.
   A fill threshold improves load utilization but may leave low-volume stores waiting indefinitely.
   Combine a maximum dwell time with a volume or scheduled departure rule when appropriate.
6. At DC receipt, reconcile expected containers and observed units.
   Record short, over, damaged, and unidentified receipts separately.
   Preserve container-level events even when item-level inspection occurs later.
7. Assign an inspection queue based on safety, aging, value decay, and contractual requirements.
   A first-in-first-out queue is transparent, but it may not minimize loss when product value decays at different rates.
   Any priority override needs an owner and a reason code.
8. Record final disposition and inventory movement.
   A DC may route items to local stock, another selling location, repair, liquidation, donation, or disposal.
   The disposition economics topic addresses valuation in depth; the network must still carry each feasible destination and cost.

**Inputs:** carrier and site capacity, package IDs, manifests, product handling rules.
**Output:** traceable receipt, inspection queue, and authorized inventory or recovery movement.
**Branch:** no matching authorization or damaged packaging triggers controlled research, not automatic refund denial or stock creation.

### 5.7 Set staffing and storage capacity

1. Forecast arrivals by node and interval, including return lag after delivery and the expected impact of promotions.
   Use a range rather than a single average when data support it.
2. Convert expected arrivals into work.
   For example, 120 units per day at 3 minutes per unit require 360 labor-minutes, or 6 productive labor-hours per day.
   Add separate setup, exception, transfer, and supervision work where those activities are incremental.
3. Distinguish paid hours from productive processing hours.
   Breaks, customer interruptions, and competing tasks reduce usable capacity.
   Do not count the same staff-hour as fully available to both selling and returns.
4. Check peak demand against capacity by hour or shift.
   A daily total can hide a Saturday afternoon queue.
   A practical plan uses staffing schedules and service-time distributions, not just daily average minutes.
5. Estimate holding space from arrivals, dwell time, and item volume.
   For a stable system, Little's law relates average work in process to average throughput multiplied by average elapsed time.
   It is not a guarantee for a bursty or unstable system.
6. Define overflow triggers and authority.
   The store manager can request a pickup; the network control owner can authorize rerouting or temporary capacity.
   A customer-facing option should not remain open when the site cannot accept the relevant class safely.

**Inputs:** arrival forecast, service-time observations, staffing constraints, storage limits.
**Output:** approved capacity plan, peak guardrails, and escalation thresholds.
**Branch:** when demand exceeds capacity, select a controlled alternative and communicate it rather than hiding the queue in unscanned stock.

### 5.8 Calculate route economics

1. Fix a common unit and population.
   Use currency per accepted eligible unit or per completed return, but do not mix units and parcels without conversion.
   State currency, price date, time horizon, and treatment of tax.
2. Define the baseline and candidate.
   The baseline is the route that would otherwise have occurred, not an idealized route with zero handling.
3. Include route-specific avoidable costs.
   Examples are counter labor, parcel postage, supplies, transfer cost, DC inspection, customer support, and avoidable recovery processing.
   If a carrier charge is per parcel and the decision unit is an item, allocate using observed items per parcel and test the assumption.
4. Include recovery as expected realizable value at a defined point.
   Multiply the probability of each outcome by its net recovery, consistently including or excluding downstream cost.
   If estimated recovery already includes repair cost, do not deduct the same cost again.
5. Include the cost of time where supported.
   Seasonal markdown or obsolescence should be reflected in recovery estimates at the actual expected release time.
   A financing charge can be included separately if it is material and not already embedded in the value estimate.
6. Treat refund and original sale consistently.
   For the same eligible unit with the same refund on both routes, both are common and cancel in the incremental comparison.
   For total order economics, include sale, refund, outbound fulfillment, and all return costs under a clearly defined accounting basis.
7. Compare expected contribution and constraints.
   Choose the highest expected feasible contribution only after customer, safety, capacity, and policy constraints have been met.
   Sensitize uncertain differences rather than claiming precision from average inputs.
8. State who owns approval.
   Finance approves costing and treatment of fixed versus variable expenses.
   Operations validates capacity and work content.
   Customer experience validates the availability and communicated promise.
   The network sponsor decides after reviewing trade-offs and exception risks.

**Inputs:** cost rates, recovery model, timing, route feasibility, common baseline.
**Output:** route contribution and sensitivity with documented cost ownership.
**Branch:** if the customer policy or refund differs between routes, explicitly model the difference and confirm it is permitted before comparison.

### 5.9 Measure lead time and customer experience

1. Define event clocks and timestamp sources.
   Record initiation, authorization, handoff, carrier acceptance, node receipt, inspection, release, refund approval, and payment submission where applicable.
   Preserve the original event time, receipt time, time zone, and source.
2. Define denominators before selecting a percentile.
   Handoff-to-receipt uses physically handed-off returns with a known receipt, while initiated-to-handoff also includes customer action.
   Refund-initiation performance should use a cohort with the applicable trigger and policy.
3. Separate calendar time from business time.
   If a customer commitment says business days, use a published business calendar and holiday treatment.
   Operational elapsed time can also be reported in calendar hours for capacity management.
4. Report distributions and incompleteness.
   Median and 90th percentile reveal different parts of the customer experience.
   Recent cases that have not reached the end event are right-censored and must not be treated as zero-duration completions.
5. Compare cohorts at equal observation windows or use appropriate survival methods.
   A completed-only sample systematically omits slow open cases.
   Report open-case counts and age buckets beside completed-case durations.
6. Measure access and effort.
   Report route availability, travel or access proxies where responsibly collected, counter wait, label and packaging requirements, contact rate, and exception rate.
   A fast conditional refund is not a good experience if the customer cannot reach the chosen drop-off point.
7. Reconcile the customer message with actual events.
   "Received by carrier" and "inspected by retailer" are distinct messages.
   If a payment provider has not confirmed settlement, do not present merchant initiation as money in the customer's account.

**Inputs:** event log, policy version, customer communication records, reporting cutoff.
**Output:** an event dictionary, cohort dashboard, and service exceptions with owners.
**Branch:** missing or contradictory timestamps go to a data-quality queue and are counted as unknown, not silently discarded.

### 5.10 Reconcile, improve, and govern

1. Reconcile counts between authorization, handoff, transit, receipt, inspection, disposition, and refund.
   Compare units with units and packages with packages.
   Timing differences mean a daily mismatch is not automatically a loss; age the unresolved cases.
2. Identify duplicates and missing links.
   A retry may generate duplicate messages without representing a second physical event.
   Use stable idempotency keys and retain an audit trail of corrections.
3. Investigate variances by owner and age.
   Store operations owns unresolved counter handoffs; carrier management owns carrier acceptance-to-delivery gaps; DC operations owns receipt-to-inspection backlog.
   Finance owns refund reconciliation; inventory control owns inventory-state reconciliation.
4. Use a recurring review with agreed thresholds.
   Review customer impacts, safety incidents, aged stock, cost, and route feasibility alongside throughput.
   The policy owner can change eligibility, while the finance owner validates the resulting economics.
5. Test a change against a credible counterfactual.
   Randomize eligible customers or locations when practical, and record whether assigned routes were actually used.
   If randomization is not possible, compare similar pre-defined cohorts and disclose residual selection bias.
6. Close the loop with an explicit decision.
   Continue, change, pause, or retire the route based on evidence and guardrails.
   Preserve the policy version and effective date so that later analysis can explain observed outcomes.

**Inputs:** reconciliations, cohort metrics, exception records, experimental design.
**Output:** approved action, owner, due date, and traceable policy change.
**Branch:** a safety or legal issue can require immediate suspension regardless of short-run economic performance.

### 5.11 Design a local-resale branch

A collection-only store and a local-resale store have different process and capacity requirements.
Local resale is attractive only if the store can establish item identity, condition, price eligibility, and ownership.
It also needs a valid location and a controlled transition into saleable stock.
The expected benefit is earlier recovery or a transfer avoided; neither is automatic.
The physical item must have a plausible local selling opportunity.

1. Confirm that the item is eligible for the store's assortment and handling capability.
   The online SKU may not be ranged in that store.
   Even if the item can technically be sold, a single odd size or color may have little expected demand.
2. Inspect against a documented condition standard.
   Sealed packaging, missing accessories, wear, and product-specific safety attributes may change the outcome.
   Store staff should know when to stop and refer an uncertain case to a specialist.
3. Verify the product and inventory identifiers.
   If an online-only SKU has no valid store item record, do not substitute a similar SKU to make the register accept it.
   The master-data owner must correct the record or direct the product to another approved route.
4. Establish the appropriate sale price or recovery basis.
   A full-price sale may be unrealistic if the season is ending.
   A clearance path can have lower handling cost but lower net recovery.
5. Record the release from return quarantine to the specific store's available state.
   The transaction must reference the inspection and prevent duplicate release.
   A later physical transfer must decrement the originating store state.
6. Monitor whether the released unit actually sells.
   A release to inventory is not cash recovery.
   Measure days to sale, markdown, unsold stock, and any subsequent transfer.
7. Reconcile the local outcome to the route model.
   Include local inspection and merchandising labor as well as avoided transport.
   Do not count both the inventory valuation at release and the later sale proceeds as separate recovery.

**Inputs:** product eligibility, inspection standards, local demand, price policy, store inventory system.
**Output:** a traceable local release or a documented alternative disposition.
**Branch:** if the store cannot establish identity or quality, preserve the item in quarantine and send it to the designated exception owner.

For illustration, assume a unit can be sent to the DC at USD 6.15 in reverse costs and USD 23.40 expected recovery, as in section 6.
Suppose local processing would cost USD 4.00 and produce USD 20.00 in expected net recovery.
Local contribution would be USD 16.00, versus USD 17.25 for forwarding to the DC.
The apparent transport saving does not compensate for lower expected local recovery.
If local recovery instead rises to USD 22.00 with costs unchanged, local contribution becomes USD 18.00.
That comparison is valid only if both recovery estimates use the same valuation horizon and include equivalent downstream costs.

### 5.12 Model dispatch frequency and value decay

Dispatch frequency is a policy choice with operational constraints.
More frequent pickups can reduce waiting time but raise vehicle, handling, or stop charges.
Less frequent pickups can improve consolidation but consume space and delay inspection.
The effect differs by store volume and product mix.

1. Measure actual item arrivals by store and day, including weekly and seasonal patterns.
   Avoid deriving every store's schedule from a regional average.
2. Record the time from item acceptance to departure.
   Include cases that are still waiting at the observation cutoff.
3. Obtain transport costs at the appropriate charging unit.
   A fixed charge per stop behaves differently from a cost per unit or per container.
4. Estimate expected inventory value as a function of release date for time-sensitive categories.
   Use actual sell-through and markdown evidence where available.
   Do not assume a constant daily loss without testing that assumption.
5. Specify a dispatch rule with both capacity and time limits.
   A rule might dispatch on a scheduled day or earlier when occupancy reaches a threshold.
   A separate maximum dwell limit prevents a small store from waiting indefinitely for a full container.
6. Simulate plausible peak and low-volume scenarios before implementation.
   Include a missed pickup, an interrupted store shift, and a delayed DC receipt.
7. Confirm escalation and ownership.
   Store operations owns accurate staging; the carrier owns agreed pickups; the network control owner resolves missed capacity.

For a simple illustrative comparison, suppose weekly pickup costs USD 120 per store-week for 100 units.
Transport cost is USD 1.20 per unit.
Twice-weekly pickup costs USD 180 per store-week for the same 100 units, or USD 1.80 per unit.
If faster pickup improves expected recovery by USD 0.80 per unit and changes no other costs, its net advantage is USD 0.20 per unit.
If the recovery improvement is only USD 0.30 per unit, the faster option loses USD 0.30 per unit.
These calculations assume the same volume, feasible capacity, and comparable unit mix.
They do not establish which schedule meets the customer promise or any safety requirement.

### 5.13 Specify practical reconciliation controls

The return network should reconcile at multiple levels because a single total can hide offsetting errors.
Two missing items and two unrecorded items do not constitute a balanced process merely because the net variance is zero.
Separate unit, package, money, and inventory-state ledgers.

1. At authorization, retain the maximum approved quantity and prior returned quantity by original line.
   Track cancellations and changes as events, not overwrites.
2. At customer handoff, count what was actually accepted.
   Capture a receipt reference and identify differences from the authorization.
3. At dispatch, record the container, its stated contents, and the seal where applicable.
   Mark its departure only when custody has actually changed.
4. At receipt, count containers and record shortages or damage.
   Item inspection can follow later, but the container must remain traceable in the meantime.
5. At item inspection, associate observed units with authorized lines using evidence.
   An unmatched item remains a documented exception, not an invented match.
6. At inventory release, ensure each qualifying unit leaves quarantine exactly once.
   A correction should reverse or amend the original event in a traceable way.
7. At refund, reconcile approved amounts to payment instructions and provider statuses.
   Investigate duplicate or failed instructions without assuming they correspond to duplicate physical units.
8. At close, age outstanding differences and compare them with approved thresholds.
   Give each exception one accountable owner and a next review time.

For example, a store accepts 40 units on Monday, dispatches 35, and retains five in a secured container.
The DC receives 34 identifiable units from that shipment; one unit remains under investigation.
The correct physical bridge is 34 received plus one unresolved in transit or receipt variance plus five still at the store, totaling 40 units.
The one unresolved unit is not automatically lost, and the five retained units are not missing.
If 38 customer refunds have already been initiated, that number belongs to the money ledger.
It cannot be added to or subtracted from the physical bridge.
Finance should reconcile those 38 instructions separately to their authorized lines and payment outcomes.

### 5.14 Handle backlog, surge, and service recovery

A return queue is not merely a throughput issue.
It affects space, customer status, refund exceptions, and the time-sensitive value of merchandise.
The appropriate response depends on which queue is growing.

1. Separate intake backlog from inspection backlog and refund backlog.
   Each has a different starting event, ending event, and operational owner.
2. Measure both count and age.
   A small queue of very old unresolved items can be more urgent than a large queue of new arrivals.
3. Segment by risk and required handling.
   Safety-sensitive and legally time-bound cases may require a different route from ordinary apparel.
   A fast lane must have objective criteria and should not conceal the age of the remaining queue.
4. Diagnose the constraint before adding capacity.
   More counter staff will not fix a DC inspection bottleneck.
   More parcel pickups will not fix missing item identifiers.
5. Establish an overflow plan that preserves evidence.
   Temporary storage needs defined custody, inventory state, security, and a recovery schedule.
   A rented trailer is not a substitute for a tracked location and a safe handling plan.
6. Communicate truthful status to customers.
   Explain the last confirmed event and next action, rather than inventing a delivery or refund date.
   Support staff need an escalation path for customers whose cases exceed policy thresholds.
7. Recover capacity without creating duplicate work.
   If overtime changes costs, include it in the economic review.
   If staff are moved from another activity, record the lost capacity there.
8. Review the incident after the queue returns to normal.
   Compare forecast, plan, actual arrivals, effective capacity, and any missed commitments.
   Update the capacity plan and supplier agreements where evidence supports a change.

The service owner decides whether and how to update the customer.
Operations owns clearing the queue, finance owns payment reconciliation, and safety can suspend an unsafe flow.
Analytics supports the decision with complete cohorts and age distributions; it does not replace operational accountability.

### 5.15 Make the customer journey observable

A reverse network is only usable if the customer can understand and complete the next step.
Customer experience depends on the entire journey, not solely the processing time after an item enters a facility.
Design each message around an event the retailer can actually substantiate.

1. Before route selection, show the applicable options and conditions.
   A customer should be able to see whether a printer, package, appointment, or original payment method is needed.
   If the option has a fee, disclose the amount and when it applies.
2. Present the actual location and hours when the customer chooses a handoff.
   Opening hours can change, so the source and refresh frequency should be owned.
   If the selected site becomes unavailable, provide a workable alternative and update the customer.
3. Confirm the authorized items and quantities.
   A customer returning multiple items should not be required to infer whether they may travel together.
   Provide instructions that distinguish separate parcels from multiple lines in one parcel.
4. At handoff, issue a durable confirmation.
   The confirmation should identify the accepted return and record the date and location.
   It should explain whether condition verification or another step remains.
5. Send status updates only when the underlying event has occurred.
   A transport event can establish carrier acceptance but not the contents of a sealed parcel.
   An inspection event can establish condition only to the extent of the documented inspection.
6. Explain the refund in the correct stages.
   If the refund is approved but not yet submitted, say so.
   If it has been submitted, distinguish that from the issuer's posting to the customer's account.
7. Provide a support route when evidence is missing.
   Customer support should see the return reference, last verified milestone, and responsible operational queue.
   It should not need unrestricted access to full payment credentials.
8. Review unequal access to the offered routes.
   Compare availability and completion across geography and practical access constraints where collection is appropriate and lawful.
   Preserve an alternative for customers who cannot use the preferred drop-off.

Consider a customer who creates a return on Monday, visits a store on Wednesday, and receives a handoff receipt.
The store transfers the parcel on Friday and the DC inspects it on Monday.
Initiation-to-handoff is two calendar days, while handoff-to-inspection is five calendar days.
If the retailer initiates a refund on Wednesday at handoff, the refund clock and inspection clock follow different paths.
No conclusion about the issuer's posting date follows from the Wednesday initiation event.
If the customer never reaches the store, reporting only completed handoff-to-inspection times omits that experience entirely.

## 6. Worked example

### 6.1 Decision and assumptions

A retailer is considering a return-to-store route for online apparel in a single metropolitan region.
The comparison covers 1,000 eligible returned units over an eight-week pilot.
The unit is one accepted item; assume one item per customer parcel for the direct-DC baseline.
Currency is US dollars (USD), excluding sales tax, at the same price date.
The counterfactual is the existing prepaid customer-to-DC parcel route.
The proposed route accepts items at a store and consolidates them for DC inspection.
The store is a collection point, not a local resale point.
Both routes pay the same USD 50 refund per eligible unit under the same policy.
The original sale, refund, and outbound cost therefore cancel in the incremental route comparison.
They remain necessary in a full order profit-and-loss calculation.

The retailer forecasts 1,000 eligible units; this is a forecast, not a customer commitment.
The operations plan reserves enough counter and DC capacity for that forecast and a peak buffer.
The internal target is at least 95% of handed-off units received by the DC within five calendar days.
The customer commitment in this illustration is a handoff receipt immediately and refund initiation within two business days after the applicable policy trigger.
The payment provider's subsequent posting time is not controlled by the retailer.
The example does not assume that the internal target and customer commitment measure the same event.

Assume the sites can safely handle this apparel and that participating customers have both options available.
The expected net recovery estimates below include condition mix and value decay through the stated inspection time.
They exclude the route handling and transportation costs listed separately.
All rates are illustrative, and real estimates should be derived from comparable cohorts and supplier agreements.

### 6.2 Baseline: customer-to-DC parcel

| Component | Calculation or assumption | USD per unit |
|---|---|---:|
| Prepaid parcel transport | Contract rate, one item per parcel | 5.20 |
| Packaging and label support | Incremental average | 0.55 |
| DC receipt and inspection | Incremental labor and materials | 2.10 |
| Customer support | Expected incremental contact cost | 0.40 |
| Total reverse-flow cost | 5.20 + 0.55 + 2.10 + 0.40 | 8.25 |
| Expected net recovery | Condition mix and timing adjusted | 24.00 |
| Route contribution | 24.00 − 8.25 | 15.75 |

The USD 24.00 recovery is not the original selling price.
It is the expected realizable amount attributable to the returned unit, on the stated basis.
At 1,000 units, expected reverse-flow cost is USD 8,250.
Expected net recovery is USD 24,000.
Expected route contribution is USD 15,750.
These totals exclude the common refund and original order flows.

### 6.3 Candidate: store handoff and DC consolidation

| Component | Calculation or assumption | USD per unit |
|---|---|---:|
| Store counter and sorting | Incremental labor and supplies | 2.30 |
| Store-to-DC consolidation | Incremental transfer and handling | 1.45 |
| DC receipt and inspection | Incremental labor and materials | 2.10 |
| Customer support | Expected incremental contact cost | 0.30 |
| Total reverse-flow cost | 2.30 + 1.45 + 2.10 + 0.30 | 6.15 |
| Expected net recovery | Condition mix and timing adjusted | 23.40 |
| Route contribution | 23.40 − 6.15 | 17.25 |

At 1,000 units, expected reverse-flow cost is USD 6,150.
Expected net recovery is USD 23,400.
Expected route contribution is USD 17,250.
The candidate saves USD 2.10 per unit in reverse-flow cost but loses USD 0.60 per unit in expected recovery.
The net expected advantage is USD 1.50 per unit, or USD 1,500 for 1,000 units.
This is a route-level contribution difference, not a complete estimate of total corporate profit.
It excludes any unmeasured customer retention effect and any one-time implementation expense.
If implementation costs USD 900 for the pilot, the eight-week pilot net benefit is USD 600 before other omitted effects.
A decision on a longer rollout must include the costs and benefits that actually change over that longer horizon.

### 6.4 Lead-time and capacity interpretation

Assume the baseline averages one calendar day from customer handoff to carrier acceptance and three more calendar days to DC receipt.
It then averages two calendar days from DC receipt to inspection.
The mean handoff-to-inspection time is 1 + 3 + 2 = 6 calendar days.
Assume the store route averages two calendar days waiting for consolidation, one day of transfer, and the same two days to inspection.
The mean handoff-to-inspection time is 2 + 1 + 2 = 5 calendar days.
These are sums of assumed mean durations, not a derived 95th percentile.
They also exclude the time from initiation to customer handoff and any refund payment-rail delay.

The candidate's lower recovery despite a shorter illustrative average is possible because its recovery estimate includes a different assumed condition mix.
However, a fair causal comparison needs comparable item and customer mix; otherwise that difference may be selection, not route effect.
For a controlled pilot, stratify by item category and season and compare assigned groups.
A capacity review should also inspect the distribution of consolidation waiting time, not just its mean.
A once-weekly pickup may create a long tail even when the mean appears acceptable.

Suppose peak store arrivals are 90 units per day and processing takes four productive minutes per unit.
The workload is 360 productive minutes, or six productive hours per day.
If a paid hour supplies 45 productive minutes for this activity, the plan needs eight paid hours per peak day before any separate exception allowance.
Those hours must not also be credited in full to normal selling work.
If one store lacks the space or staff to handle that peak, its route is not feasible merely because the regional average is favorable.

### 6.5 Sensitivity and decision

The candidate breaks even against the baseline when its recovery is USD 21.90 per unit, holding costs and baseline recovery fixed.
That follows from USD 21.90 − USD 6.15 = USD 15.75.
The candidate's assumed recovery can therefore fall by USD 1.50 from USD 23.40 before the advantage disappears.
Alternatively, with recovery fixed, its reverse-flow cost can rise to USD 7.65 per unit before break-even.
That is USD 1.50 above the assumed USD 6.15.
Changing several inputs at once requires a joint sensitivity, not adding separate break-even headrooms.

Recommendation: run the eight-week controlled pilot at sites that meet the capacity, access, and handling constraints.
The sponsor owns the go/no-go decision; finance validates costs, store operations validates capacity, and customer experience owns the communicated promise.
Pause a site if it cannot maintain safe custody or if unresolved handoffs exceed the approved threshold.
At the end, compare assigned eligible cohorts, open-case aging, actual incremental cost, recovery after a common observation window, and customer effort.
Do not extrapolate to hazardous products, rural customers, or local-store resale from this collection-only apparel pilot.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
It reproduces the worked example and shows a small event-level duration calculation.
Save it as a local script and run `python returns_example.py`.
The route model uses expected values per eligible unit; it does not forecast demand or optimize the network.
A missing completion event is reported as open, not as a completed zero-day duration.
Dates in this teaching data are UTC and durations are calendar days.

```python
from datetime import datetime, timezone
from statistics import median

# USD per accepted unit; recovery already reflects condition mix and timing.
routes = {
    "direct_dc": {
        "recovery": 24.00,
        "costs": {
            "parcel": 5.20,
            "packaging": 0.55,
            "dc_inspection": 2.10,
            "support": 0.40,
        },
    },
    "store_to_dc": {
        "recovery": 23.40,
        "costs": {
            "store_handling": 2.30,
            "consolidation": 1.45,
            "dc_inspection": 2.10,
            "support": 0.30,
        },
    },
}


def contribution(route):
    """Expected USD per unit after the listed reverse-flow costs."""
    return route["recovery"] - sum(route["costs"].values())


baseline = contribution(routes["direct_dc"])
candidate = contribution(routes["store_to_dc"])
units = 1000
one_time_cost = 900.0
print(f"Direct DC contribution: ${baseline:.2f}/unit")
print(f"Store-to-DC contribution: ${candidate:.2f}/unit")
print(f"Incremental contribution: ${candidate - baseline:.2f}/unit")
print(f"Pilot net after setup: ${(candidate - baseline) * units - one_time_cost:.2f}")
print(f"Break-even candidate recovery: ${baseline + sum(routes['store_to_dc']['costs'].values()):.2f}/unit")

# One row per return line; None means not completed at the observation cutoff.
def utc(text):
    return datetime.fromisoformat(text).replace(tzinfo=timezone.utc)


cutoff = utc("2026-02-10T00:00:00")
records = [
    {"id": "A", "handoff": utc("2026-02-01T00:00:00"), "inspection": utc("2026-02-05T00:00:00")},
    {"id": "B", "handoff": utc("2026-02-02T00:00:00"), "inspection": utc("2026-02-08T00:00:00")},
    {"id": "C", "handoff": utc("2026-02-07T00:00:00"), "inspection": None},
]
completed_days = []
open_ages = []
for row in records:
    start = row["handoff"]
    end = row["inspection"]
    if start > cutoff or (end is not None and end < start):
        raise ValueError(f"Invalid chronology for {row['id']}")
    if end is None or end > cutoff:
        open_ages.append((cutoff - start).total_seconds() / 86400)
    else:
        completed_days.append((end - start).total_seconds() / 86400)

print(f"Completed: {len(completed_days)}; median: {median(completed_days):.1f} calendar days")
print(f"Open: {len(open_ages)}; oldest: {max(open_ages):.1f} calendar days")
```

Expected output:

```text
Direct DC contribution: $15.75/unit
Store-to-DC contribution: $17.25/unit
Incremental contribution: $1.50/unit
Pilot net after setup: $600.00
Break-even candidate recovery: $21.90/unit
Completed: 2; median: 5.0 calendar days
Open: 1; oldest: 3.0 calendar days
```

The completed median is descriptive of two completed cases only.
It is not an estimate of the median for all three cases, because the third case has not yet completed.
A production implementation must additionally address duplicate events, partial quantities, late-arriving scans, time zones, business calendars, and policy versions.
It should use decimal arithmetic or integer currency units where accounting-grade rounding is required.
The example intentionally leaves customer retention unmonetized.

## 8. Data, standards, and regulatory context

### 8.1 Minimum data contract

| Entity | Required fields | Control and interpretation |
|---|---|---|
| Original order line | Order ID, line ID, SKU, quantity, sale and payment references. | Limit matching to authorized users; preserve partial quantity history. |
| Return authorization | Return ID, order line, requested quantity, reason, policy version. | Distinguish request, approval, cancellation, and expiration. |
| Customer option | Eligible routes, selected route, displayed fee and promise version. | Record what was actually shown, not just today's policy. |
| Item or quantity | SKU, quantity, condition where known, serial where required. | Permit split parcels and mixed condition without creating duplicate units. |
| Package or container | Tracking or container ID, contents manifest, origin and destination. | A manifest is an assertion until receipt is reconciled. |
| Event | Event ID, type, event time, ingestion time, source, node, actor. | Retain corrections and distinguish duplicate messages from new physical events. |
| Inspection | Item reference, grade, criteria version, inspector, timestamp. | A preliminary scan is not final condition evidence. |
| Inventory movement | From and to state, SKU, quantity, location, transaction ID. | Quarantine and available inventory must not both claim the same unit. |
| Refund | Approval, amount, currency, payment instruction, provider status. | Merchant initiation and customer posting are separate stages. |
| Exception | Code, evidence, owner, opened and resolved timestamps. | Keep unresolved cases visible and age them. |
| Cost | Route, basis, period, currency, rate, inclusion rules. | Separate parcel-level from item-level cost and avoid double counting. |

Event data should be modeled as an appendable history with explicit correction events.
A current status alone cannot reconstruct how long an item waited at each node.
Use common identifiers across order management, store point of sale, carrier, warehouse management, and finance systems.
Access should follow the minimum necessary role; a warehouse operator may not need full payment credentials or customer contact history.
Retain personal data according to the applicable policy and law, and restrict exports and logs accordingly.

### 8.2 Event and identity standards

GS1's EPCIS standard provides a common structure for sharing visibility event information about objects, locations, time, and business context.
It can support interoperable custody and process events when trading partners agree on identifiers and semantics.
EPCIS does not itself verify that a scan is truthful, mandate a particular refund policy, or make an item sellable.
GS1 identification keys and barcodes can identify products, locations, and logistics units, but local implementation still needs a mapping to the retailer's return line and quantities.
The GS1 General Specifications and EPCIS documentation are cited below.
Use the current version appropriate to the trading partner agreement; the citations are references, not a claim that every retailer must use GS1.

Electronic data interchange (EDI) may transmit shipment, receipt, or financial messages between partners.
The chosen transaction set and implementation guide depend on the agreement.
A transmitted advance notice is not the same thing as a verified physical receipt.
When identifiers or messages are retransmitted, consumers should use stable business keys and idempotent processing.
Reconciliation must account for corrections, partial receipts, and timing differences.

### 8.3 Legal and safety boundaries

Return rights, disclosure duties, and refund deadlines depend on jurisdiction, transaction type, product, and the applicable facts.
In the United States, the Federal Trade Commission's Mail, Internet, or Telephone Order Merchandise Rule addresses shipment and certain delayed or unshipped order situations; do not treat it as a universal post-delivery returns law.
State law and the disclosed merchant policy may also affect the available remedy and process.
For qualifying EU distance contracts, Directive 2011/83/EU sets a general withdrawal framework with exceptions and conditions; local implementation and later amendments matter.
Counsel should review the applicable current law and customer-facing policy before rollout.
No example in this unit establishes a legally sufficient refund deadline.

Transport of lithium batteries, aerosols, chemicals, or other dangerous goods can be subject to specific carrier and legal requirements.
The US Department of Transportation Pipeline and Hazardous Materials Safety Administration publishes hazardous materials transport guidance.
Whether a particular return is regulated depends on its material, condition, packaging, mode, and jurisdiction.
Do not allow an ordinary store or parcel route for a hazardous return merely because it exists for apparel.
Establish safe acceptance, packaging, training, and escalation with qualified specialists.

Personal information associated with orders, addresses, and returns can be subject to privacy requirements that vary by jurisdiction.
The EU General Data Protection Regulation sets requirements for processing personal data within its scope, including data minimization and security principles.
An identifier can still be personal data when it is linkable to a person.
Use role-based access, appropriate retention, and a reviewed legal basis where applicable.
This is educational context, not legal advice.

### 8.4 Data quality controls

Measure missing return IDs, unmatched packages, duplicate event IDs, negative durations, and events received after the reporting cutoff.
Partition late-arriving data by event time and ingestion time so historical reports can be restated transparently.
Check that units handed off equal units received, still in transit, or in a documented variance after allowing for timing.
Reconcile payment instructions independently from physical units; an early refund can legitimately precede inspection.
Audit the inventory transition from quarantine to available stock, including reversal and correction paths.
Maintain a common reason-code dictionary and train sites before comparing their reason-code distributions.

## 9. Industry practice and case evidence

### 9.1 Public examples and what they do not prove

UPS describes its Happy Returns service as offering a network of return locations and box-free, label-free returns for participating merchants and eligible returns.
The published service describes a customer handoff model and a partner network, not proof that every merchant or item qualifies.
[Source: UPS Happy Returns, linked in section 16.]
**Inference:** a shared drop-off network can change packaging, consolidation, and customer access trade-offs relative to individual parcels.
The cited service description does not establish a universal cost saving or a particular refund lead time.

Amazon's customer help pages describe available return options and explain that options depend on the item and circumstances.
A displayed option can include a physical drop-off and may specify whether a box or label is needed.
[Source: Amazon Customer Service return guidance, linked in section 16.]
**Inference:** option eligibility is an operational control as well as a customer interface decision.
The public guidance does not disclose the marginal cost of each route or guarantee every customer the same options.

Walmart's public returns information describes store and mail return options for eligible purchases, subject to its policy and item-specific terms.
[Source: Walmart return policy, linked in section 16.]
**Inference:** a retailer with stores and e-commerce can make a store a customer handoff node even when downstream processing differs by product.
The policy alone does not reveal whether a specific returned item is resold locally, sent to a DC, or processed by a partner.

These examples are service descriptions, not controlled experiments.
They establish that named businesses publicly offer particular options under stated conditions.
They do not establish treatment effects on retention, customer satisfaction, loss, emissions, or unit economics.
A service that is attractive in one geography or product category may not work elsewhere.

### 9.2 Translating practice into a design

Start with the consumer's actual menu of options.
Document the items for which each option is available, the customer's packaging burden, and the handoff proof.
Then trace the less visible downstream process.
A convenient store counter does not eliminate the need for safe storage, transfer, inspection, and reconciliation.
A partner drop-off may aggregate parcels, but the retailer still needs an event-to-order link and a clear refund trigger.

Review observed differences in customer effort alongside the cost model.
An option selected by customers who live near a store may appear cheaper or faster because of geography or product mix.
A randomized offer or a carefully defined comparison can distinguish some of the route effect from selection.
Even a randomized offer estimates the effect of offering a route, not necessarily the causal effect of using it.
Report assignment, actual use, and crossover separately.

A useful operating review compares the promise with observed milestone distributions.
If refunds are initiated quickly but inventory remains uninspected, finance and inventory operations face different backlogs.
If stock is released quickly but customers cannot see a receipt, support contacts can remain high.
Connect these outcomes using return IDs while keeping each team's accountable metric distinct.

### 9.3 Evidence quality and transfer limits

Official policy pages are suitable evidence of published customer options at the time accessed.
They are not independent audits of compliance or proof of profitability.
Company descriptions may change, and options can differ by region, product, seller, and customer.
Check live terms before designing a customer promise.
Use internal event data and validated costs for the decision; use public cases to identify plausible design patterns, not as substitute input rates.
Where a case suggests a hypothesis, label the inference and test it in the eligible population.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation and owner |
|---|---|---|---|---|
| Unscanned handoff | Receipts do not match counter counts. | Customer leaves before a durable event is captured. | Disputed custody and support contacts. | Store manager reconciles receipts and offline records. |
| Label mistaken for acceptance | Many labels exist with no carrier scan. | Intention is counted as physical movement. | False in-transit stock and misleading status. | Carrier operations defines and monitors acceptance events. |
| Duplicate refund | Multiple payment instructions share a return line. | Retries or separate channels lack idempotency. | Cash loss and reconciliation work. | Finance owns unique refund keys and exception review. |
| Premature inventory release | Available units rise before inspection. | Receipt event is incorrectly mapped to sellable state. | Unsellable stock is promised to customers. | Inventory control requires condition and release evidence. |
| Store overload | Counter wait and back-room occupancy rise. | Return volume exceeds planned labor or space. | Poor service and unsafe storage. | Store operations activates overflow capacity. |
| Consolidation delay | Aged containers await pickup. | Fill threshold has no maximum dwell time. | Value decay and delayed inspection. | Reverse logistics sets dispatch and aging triggers. |
| Lost package link | Tracking exists without a return ID. | Integration loses a many-to-many association. | Unidentified receipt and delayed refund. | Integration owner reconciles container and item references. |
| Mixed-condition release | Damaged items share a sellable batch. | Batch-level receipt substitutes for unit inspection. | Quality failures and extra rework. | DC quality uses item-level or defensible sampling controls. |
| Hidden customer burden | Completion drops among customers without printers. | Route cost excludes effort and access barriers. | Abandoned returns and inequitable experience. | Customer experience measures option accessibility. |
| False savings | Freight savings exceed reported total savings but labor is absent. | Shared store work is treated as free. | Incorrect rollout decision. | Finance includes avoidable labor and opportunity cost. |
| Misleading speed metric | Completed-only median improves while open backlog rises. | Slow cases are censored out. | Service deterioration is concealed. | Analytics reports open ages and cohort completeness. |
| Incorrect condition mix | Candidate appears to have lower recovery. | Different products or customers self-select. | A route is credited or blamed for mix effects. | Experiment owner stratifies and reports assignment effects. |
| Unsafe product accepted | Staff cannot identify a leaking or damaged battery. | Eligibility rules are too broad or training is absent. | Injury, transport breach, or facility damage. | Safety owner restricts route and provides escalation. |
| Refund-status overclaim | Support says funds arrived when only submitted. | Provider event semantics are conflated. | Customer distrust and repeated contacts. | Payments owner maps status language to verified events. |
| Unreconciled partial return | Returned quantity exceeds remaining eligible quantity. | Line, unit, and parcel quantities are conflated. | Duplicate stock or missed customer credit. | Order management enforces quantity ledger with correction path. |
| Policy drift | Store and online instructions conflict. | Multiple systems use different effective dates. | Denied handoffs and inconsistent treatment. | Policy owner versions and publishes eligibility centrally. |
| Exception queue stagnates | Unidentified units age beyond review threshold. | No named owner or escalation. | Refund delays and recovery value loss. | DC manager assigns aging-based ownership. |
| Optimizing average only | Mean processing time is stable but peak queues grow. | Capacity is insufficient during concentrated arrivals. | Missed commitments and staff overtime. | Workforce planner models arrival peaks and buffers. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Common refund across routes. | Lets refund cancel from incremental comparison. | Route-specific fees or different eligibility. | Compare policy versions and model the incremental refund explicitly. |
| One item per baseline parcel. | Converts parcel cost to unit cost. | Bundled or split shipments. | Measure item-per-parcel distribution and recalculate. |
| Recovery excludes listed route costs. | Prevents double counting. | Recovery was estimated net of inspection or transport. | Reconcile definitions with finance and rebuild the bridge. |
| Estimated costs are incremental. | Decision should use costs changed by the route. | Staffing or capacity cannot actually be adjusted. | Observe schedules, contracts, and the decision horizon. |
| Stable condition mix. | Allows recovery comparison on a common basis. | Store users return different products or conditions. | Stratify or randomize and report mix-adjusted results. |
| Adequate store capacity. | Enables the route without damaging service. | Peak demand or competing work exceeds staffing. | Measure by shift and plan buffer and overflow. |
| Correct custody events. | Lead-time and loss measures depend on actual events. | Missing, duplicate, or late scans. | Reconcile source records and report data quality. |
| Common observation window. | Makes cohort outcomes comparable. | Recent returns remain open or recovery is not realized. | Mature cohorts equally and report censored cases. |
| Route eligibility is enforceable. | Prevents unsafe or unsupported flows. | Staff override rules or item data is wrong. | Audit exceptions and test with safety specialists. |
| USD and price date are consistent. | Makes additions and comparisons meaningful. | Currency conversion or inflation is mixed. | Convert using a documented basis and sensitivity. |
| No monetized retention effect. | Avoids unsupported value estimates. | Route changes future customer purchasing materially. | Measure longer-term outcomes separately before valuation. |
| Mean stage times can be added. | Provides an illustrative expected total. | Inputs describe different populations or conditional processes. | Align cohorts and calculate actual end-to-end durations. |
| Capacity cost is linear over the pilot range. | Allows a constant per-unit rate. | A threshold triggers another shift or vehicle. | Model step costs and simulate plausible volumes. |
| Customer access is adequate. | Assures the candidate is a real option. | Distance, opening hours, disability, or transport constraints. | Measure access and preserve an appropriate alternative. |
| Legal and safety rules are current. | Eligibility and promises must be lawful and safe. | Law, carrier rules, or product condition changes. | Obtain current jurisdiction-specific professional review. |

These assumptions are not independent.
A peak-volume change can simultaneously increase labor cost, consolidation delay, and recovery loss.
A one-variable sensitivity is useful for diagnosis but not a substitute for a joint scenario.

## 12. Exercises

Use the illustrative USD assumptions in section 6 unless the exercise says otherwise.
State any additional assumptions, identify the decision owner, and show units.

### Exercise 1 — Draw the linked flows

**Time:** 20 minutes.
**Task:** Map a customer returning two order lines in one package to a store that forwards the package to a DC.
Include an early refund on handoff and a later item inspection.
**Deliverable:** A diagram with physical, information, inventory, and money lanes, including at least five named events.
**Acceptance criteria:** Distinguish package from return lines, show custody change, quarantine, refund initiation, and final release; assign owners.

### Exercise 2 — Select eligible routes

**Time:** 20 minutes.
**Task:** Consider sealed apparel, a leaking aerosol, and an oversized dining table.
A small store has counter staff but no hazardous-material training or oversized storage.
**Deliverable:** A route and escalation decision for each product.
**Acceptance criteria:** Do not assign unsupported goods to the store; give a feasible process to investigate or arrange an authorized alternative without asserting a specific legal classification.

### Exercise 3 — Reproduce route contribution

**Time:** 20 minutes.
**Task:** Calculate the cost, contribution, per-unit difference, and 1,000-unit pilot result after USD 900 setup.
**Deliverable:** A calculation table with USD per unit and total USD.
**Acceptance criteria:** Show why common refund cancels and explain why the result is not total order profit.

### Exercise 4 — Interpret the event cohort

**Time:** 20 minutes.
**Task:** Use the three timestamp records in section 7.
Compute the median completed handoff-to-inspection duration and the open case's age at cutoff.
**Deliverable:** Two metrics and an interpretation of their denominators.
**Acceptance criteria:** Do not assign zero days to the open case or claim the completed median represents all cases.

### Exercise 5 — Capacity and dispatch

**Time:** 25 minutes.
**Task:** A site receives 105 units per peak day and needs four productive minutes per unit.
A paid hour contains 42 productive minutes available for returns.
The collection space holds 240 units and the site dispatches only when 300 units accumulate.
**Deliverable:** Paid labor-hours required per peak day and a revised dispatch control.
**Acceptance criteria:** Show arithmetic; identify why the existing threshold is infeasible; include a maximum dwell rule and a named owner.

### Exercise 6 — Evaluate an offer experiment

**Time:** 30 minutes.
**Task:** Design an eight-week comparison of the offer of store return versus the existing route.
Some customers assigned the store option will still use mail.
**Deliverable:** Assignment unit, primary metric, guardrails, cohort window, and analysis approach.
**Acceptance criteria:** Separate offer assignment from route actually used, address selection and censoring, and do not claim a causal effect of use from simple user comparisons.

### Exercise 7 — Resolve a custody mismatch

**Time:** 25 minutes.
**Task:** A store manifest lists ten items; the DC receives a sealed container with nine identifiable items and one unidentifiable item.
One customer's refund was already initiated.
**Deliverable:** A reconciliation and escalation procedure with owner, evidence, and status for each step.
**Acceptance criteria:** Preserve the manifest, do not create a fictitious match, do not reverse the refund automatically, and keep inventory unavailable pending resolution.

### Exercise 8 — Recommend a network policy

**Time:** 35 minutes.
**Task:** Management wants every store to accept all online returns immediately because the pilot has a USD 1.50 per-unit advantage.
Develop a recommendation for the next decision meeting.
**Deliverable:** A one-page decision memo with feasibility, economics, customer access, safety, evidence, ownership, and stop conditions.
**Acceptance criteria:** Distinguish the apparel collection pilot from other products and geographies; include at least two sensitivity factors and a measurement plan.

## 13. Self-check answers

### Exercise 1

The customer request produces one authorization with two line references.
The store records both accepted quantities and links them to one package or container.
The customer receives a handoff receipt; the store owns the physical handoff record.
The refund owner can initiate payment under the approved early-refund policy without asserting that inspection is complete.
The store creates a manifest and the carrier accepts custody at a recorded event.
The DC receives the container and reconciles each item to its return line.
The DC places both units in quarantine until inspection.
Inventory control releases only qualifying quantities to their correct locations; exceptions remain unavailable.
Finance reconciles the payment instruction and later provider status independently of inventory release.

### Exercise 2

Sealed apparel can be accepted at the store if the policy and item-level inspection or forwarding process allow it.
The store should not accept the leaking aerosol into its ordinary apparel stream.
A trained safety team must determine material classification and an appropriate authorized handling route.
The table exceeds the site's storage capability, so arrange an appropriate scheduled collection or other approved bulky-item process.
Do not assume the customer must simply keep either item; customer policy and legal obligations require review.
The network policy owner approves the route matrix, and safety or specialist operations owns exceptional handling.

### Exercise 3

Direct-DC reverse-flow cost is USD 5.20 + USD 0.55 + USD 2.10 + USD 0.40 = USD 8.25 per unit.
Its contribution is USD 24.00 − USD 8.25 = USD 15.75 per unit.
Store-to-DC cost is USD 2.30 + USD 1.45 + USD 2.10 + USD 0.30 = USD 6.15 per unit.
Its contribution is USD 23.40 − USD 6.15 = USD 17.25 per unit.
The difference is USD 1.50 per unit, or USD 1,500 for 1,000 units.
After USD 900 setup, the pilot difference is USD 600.
The common USD 50 refund and original sale cancel only for this like-for-like route comparison.
Full order profit requires the sale, refund, outbound costs, taxes and other relevant components on a consistent basis.

### Exercise 4

Case A takes four calendar days and case B takes six calendar days.
Their completed-case median is (4 + 6) / 2 = five calendar days.
Case C is open and has aged three calendar days at the cutoff.
The completed denominator is two; the open denominator is one.
The completed median omits the eventual duration of case C and cannot stand alone as the cohort's overall median.

### Exercise 5

Required productive work is 105 × 4 = 420 minutes per peak day.
At 42 productive minutes per paid hour, the requirement is 420 / 42 = ten paid hours per peak day.
That excludes separately estimated exception and supervisory work.
A 300-unit dispatch threshold exceeds the 240-unit holding limit and therefore cannot safely be reached under the stated constraint.
Set a lower volume threshold compatible with the space limit and a maximum dwell time, with extra headroom for variability.
The store manager monitors occupancy; reverse logistics owns pickup capacity and escalation.
The specific threshold and dwell time require site and carrier evidence rather than an arbitrary universal value.

### Exercise 6 rubric

A strong design randomizes eligible customers or comparable stores before route choice and defines the intention-to-treat effect of offering the option.
It measures incremental contribution per assigned eligible return using a common observation window and records actual route use separately.
It reports open returns and age, customer effort, safety, service, and site capacity guardrails.
It plans for cluster effects if stores are randomized and checks assignment balance and contamination.
A simple comparison of store users and mail users is not causal because they selected their route.
If the trial is too small for a precise economic estimate, report uncertainty rather than asserting equivalence.

### Exercise 7 rubric

Preserve the container seal, manifest, receipt scan, and item descriptions as evidence.
Record nine matched units and one unidentified unit, all with an explicit location and inventory state.
The DC operations owner opens an exception and contacts the store and carrier using the container reference.
Investigate whether the unidentified unit is the tenth item without forcing a match.
Finance checks the already-initiated refund against policy and payment status; physical uncertainty alone does not justify automatic reversal.
Inventory control keeps the unmatched unit unavailable until an authorized decision and records any correction as an auditable event.
Escalate by age and close only with documented evidence and an accountable resolution.

### Exercise 8 rubric

A strong memo recommends only routes that meet item, safety, capacity, and customer-access constraints.
It uses USD 1.50 per unit as an estimate for the specified pilot, not a universal saving.
It tests at least recovery value, store labor, transfer costs, or volume-dependent step costs.
It requests an appropriately designed pilot or phased expansion, with open-case and customer-experience guardrails.
It names the network sponsor, finance, store operations, customer experience, inventory control, and safety decision rights.
It includes a stop condition for unsafe handling or loss of custody and a review date.
It avoids claiming a retention, emissions, or profit improvement that the evidence does not support.

## 14. Interview questions

1. **What distinguishes return-to-store from local resale?**
   A strong answer separates customer handoff from inspection and final stock location.
   A weak answer assumes every store return becomes store inventory.
2. **When does a refund event occur relative to physical receipt?**
   A strong answer describes policy-dependent triggers and separates approval, initiation, and payment posting.
   A weak answer treats a carrier label as proof of receipt and refund settlement.
3. **How would you compare store and DC routes economically?**
   A strong answer defines a common population, incremental cost, recovery, timing, and common refund treatment.
   A weak answer compares only transport invoices or gross retail price.
4. **How do you avoid overstating a route's speed?**
   A strong answer defines event pairs and cohorts, reports open-case aging, and distinguishes elapsed from business time.
   A weak answer averages completed cases without disclosing incomplete returns.
5. **Why can consolidation reduce cost and still destroy value?**
   A strong answer explains transfer utilization versus dwell time, value decay, and maximum-wait controls.
   A weak answer assumes fewer shipments always improve contribution.
6. **What evidence supports a change from quarantine to available stock?**
   A strong answer identifies verified item, condition criteria, authorized release, location, and an auditable inventory transaction.
   A weak answer assumes any received container is fully sellable.
7. **How would you handle a package containing several return lines?**
   A strong answer models many-to-many package and line relationships and reconciles quantities independently.
   A weak answer assigns one package ID as the sole item identifier.
8. **How would you test the customer impact of offering a new route?**
   A strong answer separates randomized offer assignment from selected use and states guardrails and observation windows.
   A weak answer attributes all differences between self-selected users to the route.
9. **Who owns an unresolved handoff?**
   A strong answer assigns the store, carrier, DC, finance, and inventory control their respective events and escalation roles.
   A weak answer assigns every discrepancy to an unnamed analytics team.
10. **How should a hazardous or damaged item enter the network?**
    A strong answer checks product, condition, carrier, jurisdiction, training, and specialist escalation.
    A weak answer applies ordinary parcel instructions to every product.
11. **What happens when store capacity is fixed in the short run?**
    A strong answer evaluates opportunity cost, peak workload, service constraints, and the decision horizon.
    A weak answer calls labor free because employees are already scheduled.
12. **What would make the worked example's conclusion reverse?**
    A strong answer calculates the USD 1.50 per-unit break-even gap and tests jointly changing costs and recovery.
    A weak answer asserts the store route is always cheaper because it has fewer parcels.

## 15. Key terms

- [Authorization](#4-definitions-and-vocabulary) — permission to process a specified return request.
- [Available-to-promise](02_available-to-promise.md) — inventory that can be committed after relevant constraints.
- [Business day](#59-measure-lead-time-and-customer-experience) — a policy-defined operating day; use the appropriate published calendar.
- [Chain of custody](#4-definitions-and-vocabulary) — evidence of responsibility at successive handoffs.
- [Cohort](#59-measure-lead-time-and-customer-experience) — a population selected by a common event and period.
- [Consolidation](#56-design-return-to-dc-and-consolidation) — combining returns for a later movement.
- [Contribution](#58-calculate-route-economics) — defined recovery less specified incremental costs.
- [Counterfactual](#4-definitions-and-vocabulary) — outcome under the alternative feasible policy.
- [Customer commitment](#3-the-core-mental-model) — a communicated promise with defined conditions.
- [Disposition](#4-definitions-and-vocabulary) — authorized outcome after review of the returned item.
- [Drop-off partner](#4-definitions-and-vocabulary) — an organization accepting a customer handoff.
- [Event time](#81-minimum-data-contract) — when a recorded operational event occurred.
- [Exception](#510-reconcile-improve-and-govern) — a case requiring a controlled nonstandard process.
- [Forecast](#51-define-the-decision-and-the-population) — an uncertain estimate of future volume.
- [Handoff](#55-design-store-handoff-and-local-processing) — a transfer of physical responsibility.
- [Incremental cost](#58-calculate-route-economics) — cost that changes with the decision.
- [Inspection](#56-design-return-to-dc-and-consolidation) — examination against condition criteria.
- [Inventory release](01_inventory-visibility.md) — transition into a recorded eligible inventory state.
- [Lead time](#59-measure-lead-time-and-customer-experience) — elapsed time between explicitly defined milestones.
- [Manifest](#56-design-return-to-dc-and-consolidation) — a stated list of expected container contents.
- [Plan](#51-define-the-decision-and-the-population) — an authorized allocation of resources and actions.
- [Quarantine](#55-design-store-handoff-and-local-processing) — controlled state preventing ordinary sale pending review.
- [Receipt](#4-definitions-and-vocabulary) — recorded arrival of a container or item at a node.
- [Recovery value](#58-calculate-route-economics) — expected or realized value under stated netting and timing rules.
- [Refund initiation](#4-definitions-and-vocabulary) — submission of a payment instruction.
- [Return line](#3-the-core-mental-model) — an original order line and quantity requested back.
- [Return-to-DC](#56-design-return-to-dc-and-consolidation) — a flow whose processing destination is a distribution center.
- [Return-to-store](#55-design-store-handoff-and-local-processing) — a customer handoff at a retail location.
- [Right-censoring](#59-measure-lead-time-and-customer-experience) — an end event not yet observed at the cutoff.
- [Sellable stock](#55-design-store-handoff-and-local-processing) — product cleared for sale in the relevant system and location.
- [Service-level agreement](#4-definitions-and-vocabulary) — a defined performance target or contractual commitment.
- [Target](#51-define-the-decision-and-the-population) — an internal desired metric, distinct from a customer promise.
- [Triage](#4-definitions-and-vocabulary) — preliminary classification for subsequent handling.
- [Unit of analysis](#3-the-core-mental-model) — the entity to which a metric or comparison applies.

## 16. Sources

Sources below support the named concepts or public-service descriptions, not the illustrative costs or outcomes.
Customer policies and regulations change; consult the current text and relevant jurisdiction before operational use.

1. GS1, [EPCIS and CBV standards](https://www.gs1.org/standards/epcis): event-sharing standards and implementation resources.
2. GS1, [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications): identification and barcode specifications.
3. Federal Trade Commission, [A Business Guide to the FTC's Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule): scope and shipment-related requirements.
4. Federal Trade Commission, [A Businessperson's Guide to Federal Warranty Law](https://www.ftc.gov/business-guidance/resources/businesspersons-guide-federal-warranty-law): warranty context; not a universal returns policy.
5. European Parliament and Council, [Directive 2011/83/EU on consumer rights](https://eur-lex.europa.eu/eli/dir/2011/83/oj): distance-contract withdrawal framework and exceptions, subject to amendments and national implementation.
6. European Parliament and Council, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj): privacy principles and obligations within its scope.
7. US Department of Transportation, Pipeline and Hazardous Materials Safety Administration, [Hazardous Materials](https://www.phmsa.dot.gov/hazmat): transport safety rules and guidance.
8. UPS, [Happy Returns](https://happyreturns.com/): published returns service and participating drop-off network.
9. Amazon Customer Service, [Returns and Refunds](https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7): customer-facing return guidance; availability depends on circumstances.
10. Walmart, [Returns and Refunds](https://www.walmart.com/help/article/returns-and-refunds/): published return-policy guidance and options.
11. Little, J. D. C. (1961), “A Proof for the Queuing Formula: L = λW,” *Operations Research*, 9(3), 383–387, [DOI: 10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383): relationship between average throughput, time, and work in process under appropriate conditions.
12. Rogers, D. S., and Tibben-Lembke, R. S. (2001), “An Examination of Reverse Logistics Practices,” *Journal of Business Logistics*, 22(2), 129–148, [DOI: 10.1002/j.2158-1592.2001.tb00007.x](https://doi.org/10.1002/j.2158-1592.2001.tb00007.x): reverse-logistics practice and definitions.

## 17. Links

- [Module 23 topic index](_index.md)
- [Module 23 survey and embedded lab](../../../course/23_omnichannel_fulfillment_returns.md)
- [Inventory visibility](01_inventory-visibility.md)
- [Available-to-promise](02_available-to-promise.md)
- [Order routing and distributed order management](03_order-routing-dom.md)
- [BOPIS and curbside](05_bopis-curbside.md)
- [Promise accuracy](06_promise-accuracy.md)
