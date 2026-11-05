# Inventory Visibility

> Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise  
> Topic 01 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 13 and 21  
> Version 2026.5

## 1. Why this topic exists

A shopper sees one unit available for pickup, pays, and travels to the store.
The associate searches the shelf and backroom but cannot find it.
The order must be substituted, rerouted, delayed, or canceled.
The visible quantity was a claim about the world, not the world itself.
Inventory visibility turns that claim into an auditable, time-bounded decision input.

The commercial cost is not confined to a lost sale.
A failed pick consumes store labor and customer-service time.
A replacement shipment may require express freight or a second package.
A delayed refund ties up a customer's money.
An inaccurate positive balance can also prevent replenishment, leaving a shelf empty for days.
An inaccurate negative balance hides saleable stock and produces false stockouts.

Different participants need different answers.
A store associate needs a searchable physical location and last verified quantity.
A digital product page needs a cautious, current indication of eligible stock.
An order management system needs an atomic reservation against that stock.
A replenishment planner needs the corrected balance and transaction history.
Finance needs explainable inventory adjustments and valuation controls.
A merchant needs to understand lost sales without treating every search as a confirmed shortage.

Visibility does not guarantee that a pickup or delivery promise will be met.
The stock may be damaged, reserved, late to receive, or inaccessible to the picker.
Even a perfect count cannot create carrier capacity or store labor.
Conversely, an imperfect count can still support useful promises if exposure is controlled.
The goal is a measured trade-off between availability, accuracy, customer experience, and cost.

This topic covers the evidence chain from physical movement to book balance to customer-facing decision.
It explains perpetual inventory, count accuracy, phantom stock, and controls.
It does not equate a displayed balance with available-to-promise (ATP).
Routing economics, detailed ATP policy, and downstream promise measurement are separate topics.

## 2. Learning outcomes

After working through this topic you can:

1. Reconstruct a perpetual inventory balance from an opening position and signed movements.
   Exercise 1 checks the arithmetic and identifies missing events.
2. Distinguish physical on-hand, recorded on-hand, eligible stock, reservations, and customer-facing quantity.
   Exercise 2 applies the distinction to a store pickup decision.
3. Compute exact-match and unit-weighted accuracy with explicit denominators and timing.
   Exercise 3 compares measures that answer different questions.
4. Diagnose phantom stock without treating a failed search as conclusive proof of loss.
   Exercise 4 requires evidence and a reversible immediate response.
5. Design idempotent, ordered, auditable movement and adjustment controls.
   Exercise 5 assesses duplicates, late events, and segregation of duties.
6. Set and evaluate a conservative exposure rule against fill and lost-sale outcomes.
   Exercise 6 requires an off-policy limitation and a prospective test.
7. Explain how an inventory signal contributes to, but cannot by itself establish, a customer promise.
   Exercise 7 covers freshness, reservation, capacity, and downstream evidence.
8. Build a count program and escalation plan with risk-based coverage and independent review.
   Exercise 8 defines ownership, thresholds, and checks on bias.

## 3. The core mental model

```text
physical units and events
  | receipt / sale / move / damage / return / count
  v
capture + validation + event identity + event time
  |                         |
  | accepted                +--> exceptions, duplicate/late-event review
  v
perpetual book by SKU x location x inventory status
  | count evidence ---> reconcile, investigate, approve adjustment
  v
eligible on-hand - active reservations - policy protection
  | freshness / confidence / channel and location rules
  v
sellable signal ---> atomic reservation ---> pick and handover
  |                                           |
  +------ published promise and evidence <----+
```

Read the diagram from left to right, but do not treat it as a single database transaction across every system.
Physical movements happen in a store or facility; systems learn about them through captured events.
A book balance is the result of applying accepted events to an opening position.
A cycle count is an observation of physical stock, not automatically a correction of the book.
An adjustment changes the book only after the count is reconciled to concurrent events and approved.

A useful inventory record is a tuple, not just an integer.
It identifies the item, node, ownership, inventory status, unit of measure, effective time, processing time, and source.
For promise decisions, it also needs freshness, committed quantities, and whether the stock is accessible for the requested fulfillment mode.
A balance at midnight cannot safely answer a live pickup query after a morning of unprocessed sales.
A balance by distribution center does not establish that a specific store has a sellable unit.

Think of three distinct quantities.
Physical on-hand is what exists in the counted scope at the observation time.
Recorded on-hand is what the ledger says exists in that scope at a specified effective time.
Exposable quantity is a policy result after eligibility, reservations, protection, and freshness checks.
These quantities can all differ without any arithmetic error.
Only compare like with like: the same SKU, location, status, unit, ownership, and time cut.

Availability is also a temporal claim.
An item may be available when a product page is rendered and gone when checkout attempts to reserve it.
Cache invalidation cannot eliminate that race by itself.
Reservation must be atomic at the authoritative boundary, and expiration or cancellation must release it exactly once.
An inquiry is not a reservation; a reservation is not a pick; a pick is not customer handover.
Record each state transition separately.

Accuracy has two directions.
Book greater than physical creates a positive book error and possible phantom stock.
Book less than physical creates a negative book error and concealed stock.
Both can arise from omitted, duplicated, misclassified, or mistimed events.
The same absolute error can have different customer impact depending on whether the decision is for one unit or a pallet.
A location with a book balance of one and zero physical units is especially risky for a one-unit pickup promise.
A location with 101 recorded and 100 physical units has the same absolute error but a different exposure.

Do not confuse observed reliability with an assumed probability.
A 90% historical pick success rate is conditional on the orders, stores, and policies observed.
It is not automatically the probability that the next item at a different store exists.
Censored observations, selection of easy picks, and changes to safety buffers all alter the observed population.

## 4. Definitions and vocabulary

The table defines the terms used throughout this topic.
Quantities refer to eaches unless a different unit is stated.

| Term | Meaning and unit | Frequent misuse |
|---|---|---|
| SKU | Stock keeping unit: a specific sellable item and variant, coded by the enterprise | Treating a style or product family as a size-specific SKU |
| Node | A store, warehouse, or other inventory-holding location with a stable identifier | Assuming a city is a specific physical location |
| Physical on-hand | Units physically present in a precisely defined scope at a stated time | Counting goods owned by another party or in transit |
| Book on-hand | Units in the perpetual inventory ledger for that scope and effective time | Calling it a verified physical count |
| Perpetual inventory | Continuously maintained book balance updated from recorded movements | Assuming every movement is captured instantly |
| Movement | A recorded change in quantity, location, ownership, or status | Counting both sides of a transfer as independent net creation |
| Opening balance | Starting quantity at the beginning of the reconciliation interval | Mixing it with an opening snapshot from another time |
| Receipt | Acceptance of incoming units into a defined ownership and status | Equating advance shipping notice quantity to accepted units |
| Sale | Event that consumes inventory under the defined sales-posting policy | Reducing stock both on reservation and on shipment as if both were sales |
| Transfer | Movement between nodes or locations, with in-transit status as appropriate | Assuming dispatch proves receipt |
| Return | Customer or internal reverse movement requiring inspection and disposition | Treating all returned units as immediately saleable |
| Adjustment | Authorized ledger correction with reason, evidence, and actor | Using an adjustment to hide a missing operational event |
| Cycle count | Scheduled or triggered count of a subset of inventory | Counting only easy or high-accuracy locations and generalizing |
| Blind count | Count performed without showing the expected book quantity to the counter | Assuming blindness alone removes all count error |
| Recount | Independent or controlled repeat observation of a disputed count | Repeating until the book value is accepted |
| Phantom stock | Positive recorded stock that cannot be located or is absent from the relevant physical scope | Declaring theft from a single failed pick |
| Concealed stock | Physical stock not represented in the relevant recorded balance | Calling all negative variances shrink |
| Variance | Usually physical count minus book quantity, in units | Omitting sign convention or comparison time |
| Shrink | Inventory loss recognized under the organization's accounting policy | Treating every timing difference as proven theft |
| Accuracy | Agreement of a record and an observation under a defined measure and scope | Reporting one percentage without a denominator |
| Exact-match rate | Exactly matching counted SKU-node records divided by eligible counted records | Interpreting it as unit-weighted accuracy |
| Unit error rate | Sum of absolute unit variances divided by a stated unit denominator | Netting positive and negative errors before taking absolute values |
| Count freeze | Operational pause or logical time boundary used to align physical and book counts | Assuming a long count window is an instantaneous snapshot |
| Event time | Time at which a physical or business event is effective | Substituting ingestion time without checking latency |
| Ingestion time | Time at which a system receives an event | Assuming arrival order is occurrence order |
| Idempotency key | Stable event identity used to prevent repeated application | Generating a new key on each retry |
| Source of truth | System authorized to make a specific decision for a defined data element | Assuming one system is authoritative for every status |
| Reservation | Temporary or committed claim on eligible stock, with quantity and lifecycle | Treating a page view as a reservation |
| Available-to-promise | Policy-computed quantity or supply available for a specific promise | Treating raw book on-hand as customer-eligible supply |
| Safety buffer | Quantity withheld from exposure to absorb uncertainty or policy risk | Calling it a physical stock correction |
| Freshness | Age or lag of the relevant information relative to a decision time | Using cache age while ignoring upstream feed lag |
| Sellable status | Disposition eligible for a specific channel under defined rules | Treating quarantined or damaged stock as saleable |
| Pick failure | Inability to pick a requested unit under a defined search procedure | Treating every failure as proof the unit does not exist |
| Reservation conflict | Competing claims against the same eligible units | Assuming an available page display prevents overselling |
| Line fill | Share of eligible order lines fulfilled under a stated policy and time window | Substituting unit fill without saying so |
| Observation bias | Difference between observed records and the full decision population | Claiming accuracy from only locations that were counted |

## 5. The mechanics, step by step

### Step 1: Define the decision and comparison grain

Input: a proposed pickup or delivery use case and a customer promise window.
Output: a written data contract for the decision.
Choose a grain at least as detailed as SKU, node, ownership, and inventory status.
Add lot, expiration, serial number, or bin when the operation requires them.
Define the unit of measure and conversion; a case is not automatically an each.
Specify which physical locations are in scope, including backroom and display.
Specify whether consigned, damaged, reserved, and in-transit goods belong in each measure.
Name the source authoritative for each field rather than designating a universal system of truth.

A store may own a unit that is not eligible for same-day pickup.
A distribution center's quantity may support shipping but not pickup at the store.
A floor model may be physically present but unavailable for sale under local policy.
Capture those restrictions explicitly rather than silently subtracting them twice.
The decision time is the checkout or page-render time, not the time of a later investigation.

### Step 2: Establish the opening position and event vocabulary

Input: a trusted baseline and a list of movement types.
Output: a signed event ledger with documented posting rules.
For a single scope, use:

```text
closing book = opening book + receipts + transfer-ins + accepted saleable returns
               - sales - transfer-outs - damages removed from saleable stock
               + signed adjustments
```

Each term contains only events that change the chosen scope.
If a receipt is placed in quarantine, increase quarantine on-hand, not saleable on-hand.
A transfer out of a store and into transit is not a network-wide loss.
A later receipt from transit into another store must not create a second unit.
A cancellation of an unpicked order may release a reservation without changing physical or book on-hand.
State whether sales decrement on tender, pick, shipment, or another event, then apply that rule consistently.

A useful event contains a stable event identifier, source, SKU, node, status, signed quantity, unit, event time, ingestion time, and reference document.
For a transfer it also references both origin and destination or an in-transit ledger.
For a correction it references the original event or count evidence and carries a reason.
Identity is not the same as payload: if a repeated identity arrives with a different payload, quarantine it for investigation.
Rejecting or quarantining a bad event needs an observable exception queue and an owner.

### Step 3: Handle duplicate, missing, and late events

Input: event stream plus event identity and watermarks.
Output: an auditable ledger and exception report.
Apply an event with a known identity at most once.
Retrying transport must not post a second sale or receipt.
Track source sequence gaps where sequence numbers are available.
Reconcile document totals and source-to-target counts rather than inferring completeness from an apparently fresh timestamp.

Event time and ingestion time answer different questions.
A 10:00 sale arriving at 10:12 affects the 10:00 physical state, but it was not known to a 10:05 decision.
Keep both the as-known-at-decision view and the later corrected effective-time view.
Do not rewrite historical promise evidence as though the late event was known earlier.
If a late event crosses a count boundary, replay or reconcile the affected interval before approving a correction.
A monitoring threshold should reflect expected feed latency and customer risk, not an arbitrary universal standard.

### Step 4: Observe physical stock and align the cut

Input: a selected count scope, a book snapshot, and a movement log.
Output: a physical count comparable to a book quantity at the same cut.
Count by SKU, node, status, and unit of measure, including all relevant bins and backroom locations.
A blind first count reduces anchoring to the expected quantity.
Scan the actual variant; visual similarity is not enough for size or color.
Make sure returns awaiting inspection and customer-reserved parcels are classified consistently with the scope.

If movements continue during counting, record the exact count interval and movement times.
For a simple forward roll, physical at cut equals physical observed plus inbound movements after observation and before cut minus outbound movements in that interval.
Apply only movements actually affecting the observed scope, and reconcile any unknown movement before adjusting.
An alternative is a controlled freeze, but its operational cost and compliance need to be explicit.
An ambiguous time window is a reason to investigate, not a reason to overwrite the ledger.

### Step 5: Compute and interpret variance

Input: aligned physical and book quantities.
Output: signed variance, absolute error, and a case for investigation.
Use `variance = physical - book` throughout this topic.
A negative variance indicates fewer observed units than the book claims.
A positive variance indicates more observed units than the book claims.
The sign convention is arbitrary, but mixing conventions within a dashboard is not.
A zero net variance across records can hide substantial errors of opposite signs.
Calculate absolute error at the record grain before aggregating.

A possible exact-match metric is matching counted SKU-node-status records divided by all valid counted records.
A possible unit error rate is the sum of absolute variance divided by the sum of physical units over the same valid records.
This unit denominator becomes zero for an entirely empty physical sample; report the rate as undefined, not zero.
Another organization may use book units as denominator; label it and do not compare rates without aligning definitions.
Report sample size, selection method, time window, and whether records with zero balances were included.

### Step 6: Investigate before adjustment

Input: nonzero aligned variance or a pick exception.
Output: documented probable cause, unresolved exception, or approved adjustment proposal.
Search the shelf, nearby bins, backroom, receiving area, and reserved-order staging where appropriate.
Check the exact variant, recent transfer documents, receipt discrepancies, cancellations, and unposted point-of-sale transactions.
Review whether a return was physically received but remains in inspection or another ownership status.
Ask whether the item was removed as damaged without a status movement.
Consider transaction duplication and timing before assigning physical loss.

Record search coverage and the result, including a failed search.
A failed pick is valuable risk evidence but not definitive proof that the stock was never present.
The unit may be misplaced or inaccessible, and the search may have been incomplete.
Suppress risky exposure while investigating if policy calls for it; do not silently fabricate an inventory movement.
If a recount confirms the variance, an authorized person reviews the evidence and approves an adjustment under policy.
Preserve the original count, recount, approval, reason code, and before-and-after balances.
Avoid allowing the person who made the adjustment to be its sole independent approver.

### Step 7: Convert evidence into a promise input

Input: book on-hand, eligible status, active reservations, protection rule, freshness, and channel constraints.
Output: a timestamped quantity or a conservative unavailable/unknown result.
For a simple illustration, `exposable = max(0, eligible book - active reservations - buffer)`.
This is an example policy, not a universal ATP definition.
The eligible book excludes ineligible statuses before this formula; do not subtract the same damaged stock again.
The buffer is a decision protection, not a ledger correction and not proof that the withheld units are absent.
Apply freshness and node-capability rules before exposing the result.

At checkout, an authoritative service must atomically check and reserve available eligible quantity.
Two customers seeing the same unit may both reach checkout; at most one reservation should succeed.
A stale or unavailable source may require hiding inventory, showing uncertainty, or using a separately approved fallback.
The chosen behavior must be explicit because a permissive fallback raises oversell risk and a restrictive one hides real stock.
Release expired or canceled reservations once, and reconcile them against picks and shipments.
The displayed promise should also incorporate store operating hours, pick capacity, and transport where relevant.

### Step 8: Monitor outcomes and change policy with controls

Input: count results, event health, exposures, reservations, pick attempts, and customer outcomes.
Output: a segmented scorecard and corrective action.
Track exact-match rate and absolute unit error alongside pick success and customer promise outcomes.
Segment by node, SKU velocity, recorded balance, product category, event latency, and fulfillment mode.
Include counts of records and units; a percentage with a denominator of two is unstable.
Track data-feed lag and missing-event exceptions separately from physical inventory accuracy.

Monitor false unavailability as well as false availability.
A strict one-unit buffer can reduce failed picks while suppressing genuine one-unit stock.
Estimate the trade-off using a randomized or otherwise defensible prospective comparison when feasible.
Historical outcomes alone may be selected by the policy that decided which units were exposed.
Set escalation thresholds before observing results, assign an owner, and define review frequency and rollback.
A policy is not successful merely because the cancellation rate falls if valid customer demand is hidden.

### Operational decision patterns

The following patterns apply the same ledger, observation, and promise principles to specific events.
Each is a compact playbook; the thresholds are locally chosen, not industry mandates.

#### Receipt short against advance notice

- **Signal:** A shipment notice says 24 units but receiving scans 22 units.
- **Immediate decision:** Post 22 accepted units; record a two-unit document discrepancy.
- **Reason:** Do not credit two units solely because they were expected.
- **Accountability:** Receiving resolves the carrier or supplier claim and retains scans.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Receipt in quarantine

- **Signal:** A sealed delivery contains 12 units pending quality inspection.
- **Immediate decision:** Place 12 units in quarantine rather than eligible stock.
- **Reason:** A release event moves approved units to saleable status once.
- **Accountability:** Quality owns disposition; inventory control reconciles both statuses.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Point-of-sale feed delay

- **Signal:** A sale occurred at 10:00 but reaches the ledger at 10:12.
- **Immediate decision:** Keep event and ingestion times and mark the affected feed stale.
- **Reason:** At 10:05 the book may overstate stock by one unit.
- **Accountability:** Operations investigates lag and the promise service applies its stale-data policy.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Duplicate sale retry

- **Signal:** The same sale event is delivered twice after a timeout.
- **Immediate decision:** Deduplicate using the stable source event identity.
- **Reason:** A second decrement would hide a real unit.
- **Accountability:** The integration owner tracks duplicate volume and conflicting payloads.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Return awaiting inspection

- **Signal:** A shopper hands back an item in unknown condition.
- **Immediate decision:** Record physical receipt into a non-saleable inspection status.
- **Reason:** Refund approval and inventory saleability are separate decisions.
- **Accountability:** Inspection records disposition and an authorized status change.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Return released for sale

- **Signal:** Inspection accepts a returned unit in unopened condition.
- **Immediate decision:** Move one unit from inspection to saleable with a unique event.
- **Reason:** The move changes eligibility but does not create a second physical unit.
- **Accountability:** The returns team records condition and inventory control reconciles statuses.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Store transfer dispatched

- **Signal:** A store sends five units to another store.
- **Immediate decision:** Decrease origin and increase in-transit quantity at dispatch.
- **Reason:** Destination availability must not rise before the agreed receipt event.
- **Accountability:** The transfer owner follows open transit records and aging.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Store transfer short received

- **Signal:** Four of five dispatched units arrive at the destination.
- **Immediate decision:** Receive four units and retain one unit as an unresolved transit discrepancy.
- **Reason:** Do not make the missing unit saleable at either store.
- **Accountability:** Investigate the packing, carrier, and receiving evidence.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Saleable item damaged

- **Signal:** A shelf unit is found broken before an order is picked.
- **Immediate decision:** Move it from saleable stock to damage status when confirmed.
- **Reason:** Removing eligibility is different from physically disposing of the unit.
- **Accountability:** The store records evidence and the disposition owner authorizes write-off.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Misplaced item found

- **Signal:** A unit recorded in the backroom appears on a different shelf.
- **Immediate decision:** Correct the location if both locations are inside the same node scope.
- **Reason:** Do not inflate node on-hand if the unit was already counted there.
- **Accountability:** The store corrects the bin and reviews the cause of misplacement.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Wrong variant picked

- **Signal:** A similar shirt in another size is scanned at pick.
- **Immediate decision:** Reject the scan and search for the exact requested SKU.
- **Reason:** A style-level quantity cannot satisfy a size-specific promise.
- **Accountability:** The associate records the exception and the catalog owner fixes ambiguous labels.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Missing shelf label

- **Signal:** The item exists but is stored under an incorrect shelf label.
- **Immediate decision:** Verify the identifier on the unit and update the location information.
- **Reason:** A failed search can reflect discoverability rather than stock absence.
- **Accountability:** The store manager reviews labeling and searches adjacent locations.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Count overlaps a sale

- **Signal:** A counter sees eight units at 14:00 and one unit sells at 14:03.
- **Immediate decision:** Roll the count to the chosen cut using verified movement timing.
- **Reason:** Comparing the 14:00 count with a 14:05 book without the sale creates a false variance.
- **Accountability:** Inventory control retains the time alignment and event evidence.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Competing reservations

- **Signal:** Two checkouts attempt to reserve the last eligible unit.
- **Immediate decision:** Use one atomic check-and-reserve at the authoritative service.
- **Reason:** A cached product-page quantity is not a concurrency lock.
- **Accountability:** Order management tracks conflicts and confirms release on cancellation.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Reservation expires

- **Signal:** A shopper abandons checkout after a timed hold.
- **Immediate decision:** Release the hold once after the defined expiry.
- **Reason:** Do not increase book on-hand when no sale decrement occurred.
- **Accountability:** Order management reconciles expired holds and duplicate release messages.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Pick search unsuccessful

- **Signal:** An associate cannot locate a unit recorded as available.
- **Immediate decision:** Record the search scope, time, and reason; invoke the exception policy.
- **Reason:** One unsuccessful search is evidence of risk, not proof of theft.
- **Accountability:** The store recounts or investigates before a permanent ledger correction.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Stale store feed

- **Signal:** The last accepted store update is outside the locally approved freshness window.
- **Immediate decision:** Return an unknown or conservative result according to the channel policy.
- **Reason:** A recent cache refresh does not prove an upstream feed is current.
- **Accountability:** The integration owner restores the feed and reviews affected orders.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Negative book balance

- **Signal:** A sale posts against zero recorded units, creating minus one.
- **Immediate decision:** Preserve the event and investigate a missing receipt or timing mismatch.
- **Reason:** Clamping the ledger to zero destroys evidence of the problem.
- **Accountability:** Inventory control reconciles the event chain and authorizes any correction.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Uncounted display stock

- **Signal:** A counter checks the shelf but excludes a promotional display.
- **Immediate decision:** Recount every location in the documented count scope.
- **Reason:** A partial search is not a complete physical count.
- **Accountability:** The store manager updates the location map and count instructions.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### High-value item exception

- **Signal:** A small count variance involves a high-value item.
- **Immediate decision:** Require independent recount and approval under local policy.
- **Reason:** A low unit count can still have material financial exposure.
- **Accountability:** Finance and loss prevention review evidence without assuming intent.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Replenishment blocked by phantom

- **Signal:** A book quantity remains positive while the shelf is empty.
- **Immediate decision:** Investigate and correct confirmed variance through the approved workflow.
- **Reason:** The false balance can suppress replenishment and prolong the stockout.
- **Accountability:** Replenishment and inventory control monitor duration and repeat exceptions.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Buffer hides real stock

- **Signal:** A one-unit buffer makes a verified single unit unavailable online.
- **Immediate decision:** Measure missed exposure and customer demand alongside failed picks.
- **Reason:** Higher apparent fill may come from refusing more opportunities.
- **Accountability:** The channel owner tests a different buffer with safeguards.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Late event before count cut

- **Signal:** A pre-count sale arrives after an adjustment is proposed.
- **Immediate decision:** Reconstruct the effective-time book and reevaluate the variance.
- **Reason:** Posting both the adjustment and late sale without reconciliation double-corrects.
- **Accountability:** Inventory control retains the as-known history and correction linkage.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Manual adjustment override

- **Signal:** An associate requests a positive adjustment without count evidence.
- **Immediate decision:** Hold the request for the required evidence and separate approval.
- **Reason:** A convenient correction can mask a missing receipt or abuse.
- **Accountability:** The authorized reviewer checks source documents and actor history.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Customer-facing item unavailable

- **Signal:** The service knows the node cannot fulfill the requested pickup mode.
- **Immediate decision:** Exclude that node or present an explicit unavailable result.
- **Reason:** Physical ownership alone does not establish eligible fulfillment.
- **Accountability:** The channel owner monitors node capability and operating-hour data.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### End-of-day feed reconciliation

- **Signal:** The store reports a daily count of sales that differs from received ledger events.
- **Immediate decision:** Compare unique source identities and document totals for the same time interval.
- **Reason:** A current last-event timestamp does not prove every event arrived.
- **Accountability:** The integration owner resolves gaps and confirms replay idempotency.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Count sample excludes empty bins

- **Signal:** Only positive book records are selected for review.
- **Immediate decision:** Add a documented sample of zero-book locations and scan the physical scope.
- **Reason:** Positive-only samples understate concealed stock.
- **Accountability:** Inventory control reports sample design and coverage with the accuracy metric.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Wrong ownership status

- **Signal:** A vendor-owned consignment unit is mixed with company-owned stock.
- **Immediate decision:** Count and record ownership separately under the contractual policy.
- **Reason:** Physical presence does not itself establish company ownership.
- **Accountability:** The inventory owner reconciles consignment records with the supplier.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Seasonal surge

- **Signal:** A promotion increases sales, receipt volume, and store workload.
- **Immediate decision:** Review lag, reservation conflicts, and count coverage at a shorter cadence.
- **Reason:** A buffer calibrated in a quiet period may be insufficient.
- **Accountability:** Operations sets a temporary policy with an end time and rollback owner.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

#### Store closure during promise window

- **Signal:** Inventory is recorded but the location is closed for the requested pickup.
- **Immediate decision:** Apply operating-hours and pickup-capability rules before promising.
- **Reason:** Correct stock does not imply that an associate can pick it on time.
- **Accountability:** The store owner maintains hours and exception calendars.
- **Evidence:** Retain the source event or observation, item and node, effective time, and decision.
- **Customer effect:** Review whether the displayed availability or an existing commitment needs correction.
- **Control test:** Compare the resulting state with the physical and event evidence after resolution.
- **Escalation:** Route unresolved discrepancies to the named owner rather than silently overwriting the balance.

### Step 9: Design a count program that can support its claims

Counting everything every day is usually impractical.
Counting nothing until the annual physical inventory leaves errors in customer-facing decisions.
A workable program combines routine coverage, targeted investigation, and an independent measurement sample.
Do not combine those three purposes into a single unexplained accuracy percentage.

Start by defining the count population.
It includes SKU-node-status records and physical locations where unrecorded units could exist.
Document whether zero-book records, new items, discontinued items, and temporary displays are in the frame.
Inventory that is not in the frame cannot be represented by the resulting sample.
For example, sampling only records with positive book quantities cannot reliably discover unrecorded stock at zero-book locations.

For operational remediation, prioritize based on risk.
Useful signals include recent failed picks, low positive balances, high sales velocity, repeated adjustments, feed incidents, and high unit value.
The relative priority weights are business choices and should be documented and reviewed.
High-risk targeting helps find actionable problems quickly.
It does not, by itself, estimate the chain-wide rate of error.

For a population estimate, draw a separate probability sample with known selection probabilities.
Stratify by store, category, or book-balance band when those distinctions matter.
If strata are sampled at different rates, use an appropriate weighted estimator.
Keep the selection frame and weights alongside the reported result.
Estimate uncertainty using a method appropriate to the sampling design.
Do not report a weighted estimate as though it were a simple unweighted average.

Assign a count coordinator and publish a calendar.
The coordinator confirms that counters have item identifiers, location maps, and access to the relevant stock.
The store owner confirms that staging and backroom areas are in scope.
The inventory-control owner defines variance thresholds and recount procedures.
Finance defines evidence and approval requirements for adjustments under the applicable policy.
The integration owner supplies the event stream and feed-health evidence for the count window.

At the count, the counter records the observation, time, and scope.
The reviewer checks whether movement continued and whether the count must be rolled to the book cut.
A separate reviewer authorizes an adjustment if the documented policy requires it.
For a disputed count, preserve both the first and second observation.
If the discrepancy cannot be resolved, retain an open exception with an owner and due date.
The open exception should not be converted into a fictitious root cause merely to close the report.

After the count, review repeat patterns.
For example, repeated size substitutions can indicate catalog or labeling problems.
Repeated transfer shortages can indicate shipping, receiving, or transit-reconciliation problems.
Repeated apparent overages after delayed receipts can indicate a posting-time problem.
The corrective action should address the event or process that creates the error, not just the individual balance.

### Step 10: Choose customer-facing behavior for uncertainty

A visibility service should distinguish at least three states: available, unavailable, and unknown.
Available means the policy's conditions are met at a stated time.
Unavailable means the system has sufficient evidence that the requested offer cannot be made under the policy.
Unknown means the evidence required to make the decision is missing or too stale.
Conflating unknown with unavailable can hide demand and obscure an integration outage.
Conflating unknown with available can expose customers to a promise that cannot be supported.

Choose the presentation for each channel and risk level.
For example, a product page may say that pickup availability cannot currently be checked.
Checkout may refuse a new reservation until the authoritative service recovers.
A customer-service agent may investigate another node, provided the alternative promise is confirmed.
These are example choices, not mandatory wording or universal rules.
The business should document the behavior, its owner, and its effect on accessibility and customer experience.

For an existing reservation, an outage does not automatically mean the unit has disappeared.
Keep the reservation state and the customer commitment separate from the current ability to make new claims.
If the order cannot be fulfilled, use the established customer communication and remedy process.
If the feed recovers, reconcile events accumulated during the outage before relaxing restrictions.
Do not release all holds and reopen availability merely because the service is responding again.

Record the decision evidence at both display and commitment.
The record should include the item, node, time, source freshness, eligibility result, quantity requested, and policy version.
At commitment, add reservation identity and the promise communicated to the customer.
When a later pick fails, record the search result, reason, and resolution.
This evidence distinguishes a stale-data exposure from a valid reservation followed by a physical execution failure.

### Step 11: Reconcile an event feed at two boundaries

An end-to-end feed may pass through a store system, integration service, ledger, cache, and order service.
Checking only the cache's most recent refresh time is insufficient.
A newly refreshed cache can contain an old ledger snapshot.
Likewise, a ledger can have a recent event while an earlier transaction is missing.
Check both timeliness and completeness.

At the source-to-ledger boundary, compare unique event IDs and transaction totals for a defined interval.
When source sequence numbers exist, inspect missing sequences and out-of-order arrival.
Compare signed quantities by event type, SKU, and node where practical.
Investigate differences before replaying a source batch.
Replay with the original event IDs so accepted events are not posted a second time.
If a source corrects an event, identify the correction explicitly instead of reusing an ID with changed content.

At the ledger-to-decision boundary, compare a sampled decision snapshot with the authoritative balance as known at that instant.
Check the mapping of statuses, active reservations, buffers, and node capability.
Measure the age of the oldest contributing source, not merely the age of the last cache update.
For an unavailable upstream source, record whether the decision used a fallback.
A fallback is an explicit operational policy, not proof that its result was accurate.

Maintain separate measures for message processing and inventory accuracy.
Message delivery success measures the integration path.
Count agreement measures observed physical-to-book agreement.
Pick success measures fulfillment under the selection and search process actually used.
Promise success measures the customer's outcome under the agreed definition.
Improvement in one measure does not establish improvement in all the others.

### Step 12: Quantify the cost of different decisions

An exposure rule has at least two competing error costs.
A false positive exposes stock that cannot be fulfilled.
A false negative withholds stock that could have fulfilled demand.
The first can produce cancellation, labor, service recovery, and customer inconvenience.
The second can produce lost contribution or move the customer to another channel.
The two categories are not necessarily symmetric in cost.

Suppose a prospective test records 100 eligible demand opportunities.
Policy A accepts 80 order lines and fulfills 76 of them.
Policy B accepts 70 order lines and fulfills 68 of them.
Policy A's observed accepted-line fill rate is `76 / 80 = 95%`.
Policy B's observed accepted-line fill rate is `68 / 70 = 97.14%`, rounded.
Policy B appears better on accepted-line fill but fulfills eight fewer lines in this illustrative comparison.
The numbers alone do not establish causation because the opportunity populations might differ.

For a simplified financial comparison, suppose each fulfilled line contributes USD 12 before failure-handling costs.
Suppose each failed accepted line adds USD 5 of incremental handling cost.
Under those assumptions, A yields `76 × USD 12 - 4 × USD 5 = USD 892`.
B yields `68 × USD 12 - 2 × USD 5 = USD 806`.
This calculation excludes customer lifetime effects, substitution, and capacity constraints.
It also assumes contribution is not already net of the USD 5 handling cost.
Change those assumptions and the decision may change.

To estimate the trade-off credibly, predefine the opportunity denominator.
Include views or demand attempts that the policy rejected, not only orders accepted for picking.
Use comparable groups and record whether customers subsequently bought through another node.
Avoid calling every unavailable page view a lost sale; many views do not represent purchase intent.
Where practical, use a randomized test with operational and customer-protection guardrails.
Report uncertainty and the distribution across stores instead of relying only on a network average.

### Step 13: Govern root causes and preventive action

Record a discrepancy as an observation first.
Then classify the root cause only when the evidence supports it.
Useful reason categories can include posting delay, omitted movement, duplicate event, wrong item, location error, damage, and unknown.
Each category needs a definition so that different stores code similar cases consistently.
An unknown category is preferable to a misleadingly precise label.

Review both frequency and materiality.
A frequent one-unit error on a fast-moving last-unit pickup offer may have substantial customer impact.
A rare high-value discrepancy may need a separate financial and security review.
Summarize counts, units, value, duration, and affected customer commitments where evidence permits.
Do not infer an individual's intent from a variance dashboard.

Assign each preventive action to the process owner.
The integration team owns source gaps and replay behavior.
The store operations team owns physical search, labeling, and count discipline.
The returns team owns inspection and disposition handoffs.
The order-management team owns reservation lifecycle and conflict behavior.
Finance and inventory control own their respective adjustment and review obligations.

Define what successful correction would look like before changing the process.
For a missing sales feed, a possible outcome is fewer unreconciled source event IDs.
For misplaced goods, a possible outcome is fewer repeat failed searches after location corrections.
For delayed returns, a possible outcome is shorter time from inspection to the correct status event.
Monitor for unintended consequences, including lower exposure caused by a more restrictive policy.
Close the action only when the owner has reviewed evidence against the defined outcome.

## 6. Worked example

### Scenario and scope

A retailer promises pickup from Store S17 for a specific blue jacket, size M, SKU J42.
The physical scope is company-owned stock at S17, including shelf, backroom, and pickup staging.
The unit of measure is one jacket.
The book snapshot and count are aligned at 14:00 local time on the same day.
No movement crosses the count cut in this constructed example.
The dollar amounts below are illustrative US dollars, not company evidence.

At 08:00 the saleable book contains 19 units.
The store accepts a delivery of 7 saleable units.
A point-of-sale feed posts 6 sales.
A transfer dispatch removes 2 units to transit.
A damaged unit moves from saleable to damage status.
A previously inspected return adds 1 saleable unit.
There are no other accepted saleable events in this interval.

| Event | Signed saleable change, units | Running book, units |
|---|---:|---:|
| Opening, 08:00 | — | 19 |
| Accepted receipt | +7 | 26 |
| Sales | -6 | 20 |
| Transfer dispatched | -2 | 18 |
| Damage moved out of saleable | -1 | 17 |
| Inspected return released | +1 | 18 |

Thus the recorded saleable on-hand is `19 + 7 - 6 - 2 - 1 + 1 = 18 units`.
The transfer is not a network-wide disappearance: the 2 units are in the in-transit scope.
The damaged unit remains physically present in a different, non-saleable status until disposition.

### Physical observation and correction

A blind count finds 16 saleable units in the defined scope at 14:00.
A second count confirms 16 units, including the staging location.
Variance is physical minus book, or `16 - 18 = -2 units`.
The book overstates saleable stock by 2 units, an example of a positive book error.
The team checks nearby bins, the backroom, recent receipts, point-of-sale events, transfer documents, and return status.
The investigation cannot establish a specific root cause from the evidence available.
It therefore records an unresolved cause, not an assertion of theft.
An independent authorized reviewer approves an adjustment of minus 2 units.
The new book is 16 units, with count, recount, review, and signed adjustment retained.

Suppose two units are already reserved for active pickup orders and a one-unit policy buffer applies.
The simple exposable quantity after the correction is `max(0, 16 - 2 - 1) = 13 units`.
Before the correction, the same formula would have produced `max(0, 18 - 2 - 1) = 15 units`.
The 2-unit difference is a potentially unsafe overstatement; it is not evidence that 2 customers actually saw or ordered those units.
The buffer is withheld from sale by policy but remains part of the 16-unit physical and book balance.
Assume reservations are included in on-hand and have not already been decremented by the sale-posting convention.

A customer asks for 2 jackets at 14:02.
Under the assumed policy, the reservation service can atomically reserve 2 of the 13 exposable units if no intervening event changes the result.
The remaining exposable quantity would then be 11 units.
The service still needs store opening hours and picking capacity before a pickup time can be committed.
Successful reservation does not prove successful pickup; the pick and handover remain separate events.

### Accuracy example across four records

The same count program observes four independent SKU-node-status records.
All observations use a common time cut and one unit per item.

| Record | Book units | Physical units | Physical minus book, units | Exact match? |
|---|---:|---:|---:|---|
| J42 at S17 | 18 | 16 | -2 | No |
| K11 at S17 | 9 | 9 | 0 | Yes |
| L07 at S18 | 4 | 6 | +2 | No |
| M20 at S18 | 0 | 0 | 0 | Yes |
| Total | 31 | 31 | 0 | 2 of 4 |

Net variance is zero units, but the sum of absolute errors is `2 + 0 + 2 + 0 = 4 units`.
The exact-match rate is `2 / 4 = 50%` of counted records.
The physical-denominator unit error rate is `4 / 31 = 12.90%` of physical units, rounded to two decimal places.
The corresponding unit accuracy convention `1 - error rate` is 87.10% here, but it can become negative when errors exceed physical units.
It is therefore safer to name and publish the underlying error rate and denominator.
The example includes a zero-book, zero-physical record; omitting it would change the exact-match denominator.
These four records are illustrative and cannot estimate chain-wide accuracy.

### Customer and financial interpretation

Assume an avoidable failed pickup costs USD 8 in incremental labor and service recovery, excluding any lost margin.
If the two overstated units had each produced one avoidable failed pickup, the illustrative incremental exposure would be `2 × USD 8 = USD 16`.
This is a conditional scenario, not an observed realized loss or a forecast.
Do not add that amount to a gross-margin estimate without checking whether the loss components overlap.
An alternative policy might hold back more units, but that can suppress legitimate demand.
The business decision requires both failed-pick cost and the contribution lost by withholding available inventory.

## 7. Code, query, or calculation

The following standard-library Python program reproduces the count calculations and a simple promise input.
Copy it into a file and run with Python 3.
The sample deliberately uses physical units as the unit-error denominator.
It does not implement a live inventory service, event replay, atomic reservation, or a statistical confidence interval.
A production implementation needs durable storage, transactional reservation, and source-specific validation.

```python
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class Count:
    sku: str
    node: str
    book: int
    physical: int

    @property
    def variance(self) -> int:
        return self.physical - self.book


def metrics(rows: list[Count]) -> dict[str, Optional[float]]:
    if not rows:
        return {"exact_match": None, "unit_error": None}
    exact = sum(r.variance == 0 for r in rows) / len(rows)
    physical_units = sum(r.physical for r in rows)
    absolute_error = sum(abs(r.variance) for r in rows)
    unit_error = absolute_error / physical_units if physical_units else None
    return {"exact_match": exact, "unit_error": unit_error}


def exposable(eligible_book: int, reserved: int, buffer: int) -> int:
    if min(eligible_book, reserved, buffer) < 0:
        raise ValueError("inputs must be nonnegative")
    return max(0, eligible_book - reserved - buffer)


def pct(value: Optional[float]) -> str:
    return "undefined" if value is None else f"{100 * value:.2f}%"


rows = [
    Count("J42", "S17", 18, 16),
    Count("K11", "S17", 9, 9),
    Count("L07", "S18", 4, 6),
    Count("M20", "S18", 0, 0),
]
result = metrics(rows)
for row in rows:
    print(f"{row.sku}/{row.node}: variance={row.variance:+d} units")
print(f"exact match: {pct(result['exact_match'])}")
print(f"unit error: {pct(result['unit_error'])}")
print(f"post-count exposable: {exposable(16, 2, 1)} units")
```

Expected output:

```text
J42/S17: variance=-2 units
K11/S17: variance=+0 units
L07/S18: variance=+2 units
M20/S18: variance=+0 units
exact match: 50.00%
unit error: 12.90%
post-count exposable: 13 units
```

The program treats each input row as already aligned and valid.
It does not prove that the observation was blind, complete, or at the same cut as the book.
It computes absolute error before aggregation, so opposite variances cannot cancel.
The empty sample and a zero physical-unit denominator produce undefined rates instead of misleading zeros.
The exposable function is intentionally separate from the accuracy calculation.
Do not use it to mutate a book balance or to claim an order has been reserved.

For operational deployment, add tests against duplicate events and concurrent reservation attempts.
Also validate unit conversions, time zones, source completeness, reservation lifecycle, and status mappings.
Preserve an immutable decision snapshot when a promise is displayed or committed.
Compare that snapshot with later corrected history rather than replacing it.

## 8. Data, standards, and regulatory context

### Minimum data contract

A visibility feed should specify field definitions, ownership, valid values, and latency objectives.
The data contract is an operational agreement, not evidence that every message arrived.
It should cover identity, physical scope, event semantics, time, quantity, and status.
The following fields are sufficient to begin discussion, not a universal schema.

| Field | Purpose | Typical failure |
|---|---|---|
| Source event ID | Deduplicate retries | New identity generated for the same retry |
| Source system | Establish origin and authority | Unattributed manual feed |
| SKU and variant | Identify the exact item | Style identifier substituted for variant |
| Node and sublocation | Locate and scope the quantity | Closed or merged node reused silently |
| Ownership | Exclude stock belonging to another party | Consignment mixed with owned stock |
| Status | Distinguish saleable, damaged, and inspection stock | Return posted directly to saleable |
| Quantity and unit | Measure the movement | Case-to-each conversion omitted |
| Effective timestamp | Place the event in business time | Local time interpreted as UTC |
| Ingestion timestamp | Measure arrival latency | Only one timestamp retained |
| Reference document | Trace receipt, sale, return, or transfer | Adjustment without count evidence |
| Reason and actor | Explain corrections and approvals | Shared login obscures accountability |
| Reservation identity | Trace allocation and release | Cancellation releases the wrong hold |
| Schema version | Interpret fields consistently | Producer changes meaning without notice |

### Identification and event standards

GS1 defines globally administered identifiers including the Global Trade Item Number (GTIN) and Global Location Number (GLN).
A GTIN identifies a trade item at the relevant packaging level; it does not itself identify the physical quantity currently at a store.
An internal SKU may map to a GTIN, but the mapping can be many-to-one or otherwise governed by local catalog design.
A GLN identifies a party or location under GS1 rules; it does not prove inventory presence or a facility's pickup capability.
Use the GS1 General Specifications and implementation rules appropriate to the item and jurisdiction.

GS1 Electronic Product Code Information Services (EPCIS) specifies a way to share visibility event data about what, when, where, why, and how.
EPCIS events can support traceability across organizations when the participants capture and exchange consistent data.
The standard does not guarantee a complete count, accurate source capture, or atomic order reservations.
A compliant event interface is therefore not a substitute for reconciliation and physical controls.

Electronic data interchange (EDI) documents may communicate shipment notices, purchase orders, or other commercial messages.
Their actual transaction set, version, partner agreement, and semantics determine what can be inferred.
An advance shipment notice is a claim about a shipment, not proof of accepted receipt.
A received quantity requires a documented receiving event or an explicitly controlled exception.

### Financial and legal context

International Accounting Standard 2 (IAS 2) addresses inventories and measurement under International Financial Reporting Standards (IFRS).
It is not a prescription for this topic's specific cycle-count frequency or customer-facing safety buffer.
Where IFRS applies, finance should evaluate adjustments and valuation under the applicable accounting framework.
Other entities may use different accounting standards and local legal requirements.
Do not assume that an operational variance is automatically a recognized financial loss at the same moment.

In the United States, the Federal Trade Commission's Mail, Internet, or Telephone Order Merchandise Rule addresses shipment representations and delays for covered orders.
The rule is not a universal global pickup-time standard and its applicability depends on the facts.
Merchants should consult current regulator guidance and counsel about promises, consent, refunds, and communication.
A visibility control alone does not establish compliance with delivery or consumer-protection obligations.
This educational material is not legal or accounting advice.

For traceable or regulated products, applicable lot, expiration, serial, recall, and chain-of-custody rules can constrain the required grain.
The obligations differ by product, country, and role; identify them before simplifying data to SKU-node totals.
Privacy controls also matter because reservations may link inventory events to identifiable customer orders.
Limit access, define retention, and avoid embedding unnecessary customer information in shared stock feeds.

## 9. Industry practice and case evidence

### Evidence hierarchy

Public evidence describes particular firms, periods, and operating settings.
It rarely reveals a full inventory accuracy denominator or a controlled estimate of causal effect.
Separate a company's reported approach from your own inference about why it might work.
Neither a vendor demonstration nor an implementation announcement proves a universal improvement.

### Inventory record inaccuracy as a documented retail problem

DeHoratius and Raman studied inventory records at a large retailer and reported substantial record inaccuracy in their published research.
The paper is useful because it treats inaccurate records as an operational phenomenon rather than just a system availability issue.
Its results relate to its studied retailer and period; they are not a current benchmark for every chain.
The practical implication is to measure record-to-physical agreement directly instead of assuming the ledger is correct.

### RFID in retail

Radio-frequency identification (RFID) can automate some item identification and support frequent stock observations.
The University of Arkansas research by Hardgrave and colleagues studied RFID and inventory record accuracy in a retail setting.
Consult the original paper for its intervention, sample, definitions, and limitations before comparing its reported outcome with another operation.
**Inference:** faster, more complete scans may improve observation frequency, but tag coverage, placement, read conditions, and event integration still matter.
A tag read near a store does not necessarily establish ownership, condition, shelf accessibility, or customer reservation status.

### Target and store fulfillment

Target's annual reporting describes using stores as fulfillment hubs for digital orders, including same-day services.
This is evidence that store inventory and store execution matter to a real omnichannel operating model.
It is not evidence that Target achieves a particular SKU-level inventory accuracy or that any specific buffer is used.
**Inference:** when stores fulfill digital demand, poor location and status accuracy can create costs in both the digital order and store labor.
The inference follows from the mechanics in this topic, not from a disclosed causal estimate in the report.

### Applying external evidence responsibly

Before adopting an external benchmark, record the publication date, population, metric, denominator, and intervention.
Check whether the measure is physical count accuracy, scan read rate, pick success, or on-time customer fulfillment.
These outcomes are related but cannot be substituted for one another.
Ask whether the reported improvement is absolute percentage points or a relative percentage change.
Also ask whether the result reflects a controlled comparison or a before-and-after association.
A local pilot should collect both benefits and suppressed demand, with a predeclared evaluation period.

## 10. Failure modes and red flags

Each control requires a named owner and an exception-resolution path.
The table gives starting points, not universal thresholds.

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Missing sales | Source-to-ledger gap | Sales not posted | Phantom stock | Reconcile event IDs and replay safely |
| Duplicate receipt | Repeated source reference | Retry applied twice | False availability | Stable idempotency key |
| Late transaction | Rising event latency | Old event arrives after decision | Misleading historical reconstruction | Retain event and ingestion times |
| Wrong variant | High substitution or scan mismatch | Style used instead of SKU | Failed customer pick | Scan exact item identifier |
| Transfer not received | Aging transit balance | Dispatch confused with receipt | Destination oversell | Two-sided transfer reconciliation |
| Uninspected return | Saleable balance jumps on refund | Status conflated with payment | Damaged item offered | Inspection status and release event |
| Missing damage movement | Damaged units found in sellable count | Status not updated | False availability | Record controlled status transition |
| Overdue feed | Old source watermark | Integration outage | Stale customer display | Monitor upstream lag and fail by policy |
| Double reservation | Conflicting accepted holds | Nonatomic check and update | Oversell | Transactional check-and-reserve |
| Stuck reservation | Holds exceed active orders | Release missing or duplicated | Concealed sellable stock | Reconcile order and hold lifecycles |
| Incomplete count | Variance disappears on wider search | Backroom or display omitted | Bad adjustment | Document scope and independent recount |
| Moving count cut | Count and book disagree during active trading | Events not time-aligned | False variance | Freeze or roll movements to a common cut |
| Adjustment abuse | Repeated unexplained overrides | Weak approval and audit | Financial and operational risk | Evidence, limits, separate review |
| Hidden offsetting errors | Net variance near zero with many discrepancies | Positive and negative errors cancel | False confidence | Sum absolute errors at record level |
| Biased count sample | Only convenient locations counted | High-risk locations excluded | Inflated accuracy | Probability or risk sample with coverage |
| Excessive buffer | Fewer failed picks but more unavailable views | Genuine stock withheld | Lost contribution | Measure both suppression and failures |
| Uncontrolled release | Canceled reservation releases twice | Non-idempotent lifecycle | Excess exposure | State machine and idempotent release |
| Unmapped ownership | Consigned stock mixed with owned stock | Scope mismatch | Invalid promises and valuation | Separate ownership and contract rules |
| Negative balance clamped | System shows zero despite negative event chain | Evidence discarded | Persistent root cause | Preserve ledger and investigate |
| False certainty | Green status despite unknown feed health | Missingness treated as accuracy | Incorrect promise | Explicit unknown state and escalation |

A high exact-match rate does not eliminate the possibility of severe errors on a few high-demand items.
A low error rate does not reveal whether failures are concentrated at the last unit.
Review the distribution and the resulting customer outcomes alongside the aggregate.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Correct SKU mapping | Matches a physical variant to a record | Reused or ambiguous identifiers | Scan audit and catalog mapping review |
| Comparable scope | Makes book and count meaningful | Missing display or mixed ownership | Walk the count boundary and location map |
| Aligned time cut | Prevents timing from looking like loss | Unlogged movement during count | Reconcile timestamps and movement log |
| Reliable count | Makes observation useful | Double count or missed bin | Blind independent recount of a sample |
| Complete events | Supports perpetual balance | Source outage or sequence gap | Source-to-target document reconciliation |
| Idempotent application | Prevents duplicated postings | Retry with changed identity | Inject duplicate event in a controlled test |
| Status semantics consistent | Determines saleable eligibility | Return posted straight to saleable | Trace status transitions and disposition |
| Reservations distinct from book | Avoids double subtraction | Sale decremented on hold in another system | Trace one order through each state |
| Buffer policy applicable | Defines conservative exposure | Surge or category mix changes | Segment outcome and suppression rates |
| Stable units | Makes arithmetic comparable | Case-pack conversion or fractional unit | Audit unit mappings and conversion factors |
| Count sample representative | Supports broader inference | Counts target only easy records | Compare coverage and selection probabilities |
| Decision snapshot retained | Enables promise audit | Later corrections overwrite history | Reconstruct a past decision as known then |
| Costs are incremental | Supports business comparison | Labor or margin counted twice | Reconcile assumptions with finance |
| Physical unit can be picked | Links count to fulfillment | Item inaccessible or location unknown | Compare count, search, and pick outcomes |

Sampling uncertainty matters even if every selected record was counted correctly.
Risk-based samples are useful for control, but raw results may not represent the whole estate.
Use weighting or a separately designed probability sample when estimating a population metric.
A small illustrative sample cannot establish a service-level probability or set a universal buffer.

## 12. Exercises

For each exercise state units, time cut, scope, and assumptions.
Record an uncertainty rather than inventing a missing fact.

### Exercise 1 — Ledger reconstruction

A store starts with 23 saleable units at 08:00.
It receives 9, sells 8, sends 3 to transit, removes 2 to damage, and releases 1 inspected return.
Deliverable: a signed movement table and closing book quantity.
Acceptance: every event appears once; transit and damage are described as status or scope changes.
Time: 10 minutes.

### Exercise 2 — Promise input

At checkout, eligible book is 12 units, active reservations are 3 units, and a policy buffer is 2 units.
The store's source feed is current, and the customer requests 4 units.
Deliverable: the simple exposable calculation and a description of what must happen before confirming.
Acceptance: arithmetic is correct and the answer distinguishes inquiry, atomic reservation, and operational capacity.
Time: 10 minutes.

### Exercise 3 — Accuracy metrics

At a common cut the book and physical quantities for three records are `(10, 8)`, `(5, 5)`, and `(1, 3)` units.
Deliverable: signed variances, exact-match rate, absolute unit error, and physical-denominator error rate.
Acceptance: absolute values are taken before aggregation and denominator is explicit.
Time: 15 minutes.

### Exercise 4 — Phantom-stock diagnosis

A picker cannot find the final recorded unit for a pickup order.
The shelf label is correct, but the backroom search was interrupted and the point-of-sale feed is 30 minutes late.
Deliverable: immediate customer-protection action, three investigation checks, and a proposed adjustment decision.
Acceptance: the answer does not assert theft or overwrite the book without evidence and approval.
Time: 15 minutes.

### Exercise 5 — Event-control design

A receipt retry produces the same source ID twice, and a separate earlier sale arrives after a count.
Deliverable: a processing sequence with idempotency, effective-time treatment, reconciliation, and approval controls.
Acceptance: duplicate receipt is applied once; the late sale is not double-counted with an adjustment.
Time: 20 minutes.

### Exercise 6 — Buffer evaluation

A proposed buffer of one unit lowers observed failed-pick rate from 6% to 3% of accepted order lines.
During the same period, product-page unavailability increases from 8% to 17% of eligible views.
The periods differ in promotions and stock mix.
Deliverable: a critique and prospective evaluation design including customer and financial outcomes.
Acceptance: distinguish selection effects from causal impact and include suppressed demand and operational safeguards.
Time: 25 minutes.

### Exercise 7 — Promise audit

A customer saw pickup availability at 11:00, reserved at 11:04, and received a cancellation at 11:40.
A 10:50 sale reached the stock service at 11:15; the store was open and had labor capacity.
Deliverable: a timeline of physical event time, information arrival, displayed claim, reservation, and outcome.
Acceptance: preserve the as-known-at-decision view and identify missing evidence before concluding root cause.
Time: 20 minutes.

### Exercise 8 — Count governance

A chain has 100 stores with variable shrink risk, velocity, and feed health.
Deliverable: a count and exception program with roles, a coverage plan, escalation, and a measurement plan.
Acceptance: include independent approval, count-cut treatment, zero-book coverage, and a way to assess sampling bias.
Time: 30 minutes.

## 13. Self-check answers

### Exercise 1

The closing book is `23 + 9 - 8 - 3 - 2 + 1 = 20 units`.
The 3 units sent to transit leave the store scope, not necessarily the network.
The 2 damaged units leave the saleable status, not necessarily physical ownership.
The inspected return enters saleable status only once.
If the sale posting rule already decremented stock at an earlier pick, verify that the 8 sales are not duplicate decrements.

### Exercise 2

Exposable is `max(0, 12 - 3 - 2) = 7 units`.
Four units fit the simplified policy at the inquiry instant.
The authoritative service must atomically reserve 4 units before the order is committed.
It must also check the relevant store hours and pick capacity.
A current feed and enough quantity do not by themselves establish a guaranteed customer handover.

### Exercise 3

The variances are `8 - 10 = -2`, `5 - 5 = 0`, and `3 - 1 = +2` units.
Exactly one of three records matches, so the exact-match rate is `1 / 3 = 33.33%`.
Absolute error is `2 + 0 + 2 = 4 units`.
Physical total is `8 + 5 + 3 = 16 units`.
The physical-denominator error rate is `4 / 16 = 25%`.
Net variance is zero, which conceals both mismatches.

### Exercise 4 rubric

A strong answer protects the open order through the defined exception process and communicates uncertainty.
It checks the remaining backroom, pending point-of-sale events, and staging or recent transfers.
It records what was searched and when, then arranges a complete count or controlled recount.
An approved correction follows confirmed aligned evidence; a 30-minute feed delay alone is not proof of missing stock.
A weak answer simply writes off one unit or labels the associate's failed search as theft.

### Exercise 5 rubric

A strong sequence detects the repeated receipt identity and applies its quantity once.
If the same identity has inconsistent payloads, it isolates the conflict rather than choosing silently.
It records the earlier sale's effective and arrival times and reconstructs the relevant count cut.
It adjusts only the remaining confirmed difference, with evidence and independent authorization.
The answer preserves what was known at earlier customer decisions.

### Exercise 6 rubric

The observed failed-pick rate falls by 3 percentage points among accepted lines.
The observed unavailability rate rises by 9 percentage points among eligible views.
These denominators describe different populations, so subtracting the two rates is not meaningful.
Promotions and mix changed, and customers excluded by the buffer cannot produce accepted-line failures.
A strong evaluation randomizes comparable stores or traffic where safe, or uses a defensible matched comparison.
It measures exposure, attempted orders, fulfilled contribution, failed picks, customer recovery cost, and confidence intervals.
It sets guardrails, duration, and rollback criteria before launch.

### Exercise 7 rubric

A strong timeline shows the sale occurred at 10:50 but was unavailable to the stock service until 11:15.
At 11:00 and 11:04 the service may have operated on an overstated book, depending on its actual state and feed-health policy.
The cancellation at 11:40 is an outcome, not proof of the reason.
Inspect the exact item, balances, reservation result, picker search, feed health, and cancellation reason.
Retain the original decision snapshots even after the late event corrects effective-time history.

### Exercise 8 rubric

A strong design combines risk-triggered counts with broad coverage, including zero-book locations.
It assigns separate count, review, adjustment approval, feed ownership, and store accountability.
It documents a freeze or time-roll method, recount thresholds, evidence retention, and escalation.
It reports sample selection and coverage, and uses a probability sample or appropriate weighting for population inference.
It monitors both stock accuracy and customer outcomes, with periodic control review.

## 14. Interview questions

1. **How do you distinguish book from physical on-hand?**
   Strong: defines scope and time, and explains recorded events versus observation.
   Weak: assumes the ledger is a verified physical count.
2. **What creates phantom stock?**
   Strong: gives missing sale, wrong status, transfer, and count examples while separating proof from suspicion.
   Weak: attributes every discrepancy to theft.
3. **Why can net variance be zero when accuracy is poor?**
   Strong: shows positive and negative errors cancel and calculates absolute error at record grain.
   Weak: uses only the net total.
4. **How do you compare a count with live trading?**
   Strong: aligns a cut with a freeze or verified event roll and preserves timing evidence.
   Weak: compares an arbitrary snapshot with an all-day count.
5. **What does a buffer do?**
   Strong: describes withheld exposure, reduced risk, and suppressed legitimate demand.
   Weak: calls the buffer a physical adjustment or claims it guarantees fulfillment.
6. **How do you prevent double selling the last unit?**
   Strong: describes an atomic authoritative reservation and idempotent lifecycle.
   Weak: relies on both shoppers seeing a fresh product page.
7. **How should a late sale affect a past promise audit?**
   Strong: retains both as-known and corrected effective-time views.
   Weak: overwrites history and claims the decision service knew the late event.
8. **How do you handle a failed pick?**
   Strong: protects the customer, documents the search, investigates, and adjusts only on approved evidence.
   Weak: treats one failed search as conclusive loss.
9. **How would you evaluate a count program?**
   Strong: specifies sample coverage, denominator, uncertainty, physical alignment, and customer outcomes.
   Weak: quotes a single accuracy percentage without a sample frame.
10. **What does a GS1 identifier guarantee?**
    Strong: explains identification and the limits of identification for presence and eligibility.
    Weak: assumes an identifier proves an item is physically available.

## 15. Key terms

The section links below lead to the definitions and operational treatment in this topic.

- [SKU](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Node](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Physical on-hand](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Book on-hand](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Perpetual inventory](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Movement](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Receipt](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Transfer](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Return](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Adjustment](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Cycle count](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Blind count](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Recount](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Phantom stock](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Concealed stock](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Variance](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Shrink](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Exact-match rate](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Unit error rate](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Count freeze](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Event time](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Ingestion time](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Idempotency key](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Source of truth](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Reservation](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Available-to-promise](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Safety buffer](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Freshness](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Sellable status](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Pick failure](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Reservation conflict](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Line fill](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.
- [Observation bias](#4-definitions-and-vocabulary): see its scope, unit, and common misuse.

## 16. Sources

The links below identify original research, standards bodies, regulators, and company reporting.
Check the current editions and local applicability before operational or compliance use.

1. DeHoratius, Nicole, and Ananth Raman. “Inventory Record Inaccuracy: An Empirical Analysis.”
   *Management Science*, 54(4), 2008, 627–641.
   [https://doi.org/10.1287/mnsc.1070.0789](https://doi.org/10.1287/mnsc.1070.0789).
   Empirical evidence on retail record inaccuracy; its results are specific to the studied setting.
2. Hardgrave, Bill C., John A. Aloysius, and Sandeep Goyal.
   “RFID-Enabled Visibility and Retail Inventory Record Inaccuracy: Experiments in the Field.”
   *Production and Operations Management*, 22(4), 2013, 843–856. [https://doi.org/10.1111/poms.12010](https://doi.org/10.1111/poms.12010).
   Research on RFID and retail inventory accuracy; consult the paper for methodology and scope.
3. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Defines identification and barcode application rules; it does not guarantee physical presence.
4. GS1. [EPCIS and CBV](https://www.gs1.org/standards/epcis).
   Visibility-event standards and documentation; conformance does not guarantee complete events.
5. IFRS Foundation. [IAS 2 Inventories](https://www.ifrs.org/issued-standards/list-of-standards/ias-2-inventories/).
   Official overview of inventory accounting requirements under IFRS.
6. US Federal Trade Commission. [A Business Guide to the FTC's Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule).
   Official guidance on covered shipment representations, delays, and related duties.
7. Target Corporation. [Annual reports](https://investors.target.com/financial-information/annual-reports).
   Company reporting on store and digital operations; not independent evidence of a specific inventory-accuracy result.

## 17. Links

- [Module 23 index](_index.md).
- [Module 23 survey](../../../course/23_omnichannel_fulfillment_returns.md).
- [Available-to-promise](02_available-to-promise.md): convert eligible supply into a promise.
- [Order routing and distributed order management](03_order-routing-dom.md): select a fulfillment node.
- [Promise accuracy](06_promise-accuracy.md): compare the customer commitment with the outcome.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md): practice evaluating operational policies.

Sibling topic links follow the module's planned topic map; availability depends on publication of those topics.
