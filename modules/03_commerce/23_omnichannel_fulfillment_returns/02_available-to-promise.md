# Available-to-Promise: From Inventory to a Defensible Customer Commitment

> Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise  
> Topic 2 of 10  
> Estimated study time: 4 hours  
> Prerequisites: inventory fundamentals and order lifecycle  
> Version 2026.5

## 1. Why this topic exists

An inventory balance is not a promise.
A retailer may show 20 units of a product in a store while five are already committed, two are damaged, and three are held back for walk-in demand.
A website that offers all 20 units sells the same stock more than once.
A website that offers none may lose orders that could have been fulfilled profitably.

Available-to-promise (ATP) turns a supply position, a time horizon, and existing obligations into a quantity that can be committed under a defined policy.
Available-to-sell (ATS) is often the immediately marketable subset, but vendors and firms use these names differently.
The actual contract is the calculation, its timestamp, and its ownership, not the label.

The decision crosses teams.
Merchandising wants availability; stores need stock for walk-in customers; operations must pick and ship; finance wants profitable orders; customer service bears broken promises.
A promise engine must arbitrate among these objectives without counting the same unit twice.

This topic distinguishes a display quote from an atomic reservation and a customer commitment.
It covers physical eligibility, future receipts, time-phased demand, holds, safeguards, concurrency, exception handling, and trade-offs.
Delivery-date calculation also needs carrier and capacity information; stock alone cannot guarantee arrival.

**Example business decisions**

- Should a store expose its last sellable unit online?
- Can a planned inbound shipment support an order for next Tuesday?
- When should an abandoned checkout release its inventory?
- Should an unconfirmed supplier receipt count toward a customer promise?
- How much safety stock is worth protecting against a false-availability incident?
- What happens to a confirmed order when a cycle count finds missing stock?

The object of analysis is not maximizing a displayed quantity.
It is generating incremental, profitable, feasible commitments while honoring existing customers.
A lower availability figure can be better when the alternative is a costly cancellation.
Equally, a blanket buffer can conceal bad data and unnecessarily suppress sales.

## 2. Learning outcomes

After this topic, you can:

1. **Distinguish** physical stock, eligible stock, ATS, ATP, reservations, allocations, and customer commitments; demonstrate the distinction in Exercise 1.
2. **Compute** a point-in-time sellable balance and a time-phased, no-double-counting ATP; demonstrate in Exercises 2 and 3.
3. **Trace** an order from quote to hold, conversion, release, and fulfillment; demonstrate in Exercise 4.
4. **Diagnose** overselling caused by concurrency, stale data, or duplicate events; demonstrate in Exercise 5.
5. **Choose** protections for channels, unreliable receipts, and inventory uncertainty; demonstrate in Exercise 6.
6. **Evaluate** a buffer using contribution and failure costs rather than only an availability rate; demonstrate in Exercise 7.
7. **Design** a measurement and exception process that preserves existing promises; demonstrate in Exercise 8.
8. **Explain** how standards, system boundaries, and customer-facing terms constrain but do not define the ATP policy; use the interview questions to check your explanation.

Each exercise asks for a concrete deliverable.
Numerical exercises have worked answers in section 13.
Design exercises have a rubric because more than one policy can be defensible.

## 3. The core mental model

```text
Physical observations and inbound evidence
                 |
                 v
       Eligibility and data quality
   status, location, ownership, timestamps
                 |
                 v
  Supply and obligations by node and time
  on-hand + eligible receipts - obligations
                 |
                 v
   Protection and feasibility constraints
    buffer, channel rules, capacity, transit
                 |
                 v
      Quote (advisory, expires quickly)
                 |
        atomic conditional reservation
                 |
       +---------+---------+
       |                   |
   convert to order     expire/release
       |                   |
 customer commitment    inventory returns
       |                 to candidate pool
 pick -> ship -> deliver -> possible return
```

The first boundary is **physical versus eligible**.
A unit may exist but be unavailable because it is quarantined, customer-owned, reserved, damaged, or in the wrong location.
The second boundary is **eligible versus uncommitted**.
The same physical unit cannot back both an existing order and a new one.
The third boundary is **available versus deliverable**.
The right inventory is not enough if the store cannot pick before cutoff or the carrier cannot reach the customer in time.

A quote is an observation or estimate, not necessarily a claim on inventory.
A hold reserves a specific quantity for a limited period; whether it is binding depends on the business contract.
A confirmed order creates a customer obligation even when internal systems later discover a shortage.
Treat that obligation as something to manage, not a number to erase.

A node is a stock-holding or fulfillment location, such as a store or distribution center (DC).
The inventory key normally includes product, node, inventory status, owner, and sometimes lot, serial number, or expiration date.
The policy must also identify the channel, promise date, unit of measure, and the time at which inputs were observed.

**Time is part of the inventory key for decisions.**
An inbound receipt arriving Friday cannot support a Wednesday pickup.
A Friday receipt also cannot support two separate Friday orders unless enough units remain after both obligations are deducted.
Future supply is a confidence-weighted operational decision, not a physical fact.
A booked transfer, an advanced shipping notice, and an unconfirmed purchase order carry different uncertainty.

**Conservation is the governing invariant.**
For a given pool and period, active claims plus protected units must not exceed the eligible supply that can support them.
At the commitment boundary, check this invariant atomically.
A fast cache is useful for browsing; it is not, by itself, proof that an order has secured stock.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Stock keeping unit (SKU) | Commercial item variant, such as a particular size and color; identifier | Treating a style as interchangeable with each variant |
| Node | Store, DC, or other inventory location; location identifier | Pooling distant stock without accounting for transit |
| On-hand | Recorded physical quantity at a node; units | Assuming every recorded unit is present or sellable |
| Physical count | Observed units at a stated time | Assuming a count remains current after transactions |
| Inventory status | Condition or disposition of a unit, such as saleable or quarantine | Counting quarantine in the sellable pool |
| Eligible inventory | Stock meeting ownership, condition, channel, and product rules; units | Deducting a blocked unit again after excluding its status |
| Allocated stock | Stock assigned under a defined internal policy; units | Assuming every allocation is a customer order |
| Reservation | A claim against capacity or inventory for a named purpose; units and expiry | Treating a reservation record as a physical movement |
| Soft hold | Temporary inventory claim, typically during checkout; units and expiry | Calling an unreserved browse quote a hold |
| Hard reservation | Claim tied to an accepted order or another controlled obligation; units | Assuming it guarantees a physical unit exists |
| Commitment | Obligation made to a customer for quantity and service date | Treating a forecast or an internal target as a commitment |
| Available-to-sell (ATS) | Policy-defined quantity immediately offered for sale; units | Assuming all systems use the same formula |
| Available-to-promise (ATP) | Quantity that can be newly committed for a defined scope and date; units | Using undated on-hand as date-specific ATP |
| Capable-to-promise (CTP) | Promise based on resources such as production or labor as well as material | Assuming materials alone imply capability |
| Safety buffer | Deliberate protection against uncertainty; units or a rule | Treating it as a physical defect or a second reservation |
| Channel protection | Quantity or capacity held for a channel or segment | Treating protected stock as usable simultaneously by all channels |
| Future receipt | Inbound supply expected at a time; units and confidence | Treating a purchase order as already received |
| Lead time | Elapsed time from one operational event to another; hours or days | Replacing a distribution with a single precise value |
| Cutoff | Latest time for a process or shipment on a given day and local time zone | Ignoring holidays, daylight saving, or local calendars |
| Promise date | Date or time interval communicated to a customer | Confusing shipment date with arrival date |
| Order management system (OMS) | System coordinating order states and often sourcing decisions | Assuming every OMS has authoritative physical counts |
| Event time | Time a business event occurred, preferably with time zone | Replacing it with arrival time when events are late |
| Ingestion time | Time the receiving system processed an event | Assuming it is the physical event time |
| Idempotency key | Stable identifier used to make retries produce one logical effect | Generating a new key for each retry |
| Version | Monotonic revision or concurrency token for a record | Assuming a wall clock alone establishes a total order |
| Phantom inventory | Recorded stock that is physically absent or inaccessible | Solving every instance by adding an arbitrary buffer |
| Stockout | Inability to meet demand from available supply; definition depends on scope | Counting a hidden unit as customer-accessible stock |
| Backorder | Accepted demand to be fulfilled later under an explicit policy | Silently promising immediate fulfillment with no stock |
| Return disposition | Decision about whether returned goods are resellable, repairable, or otherwise handled | Adding a return to ATS when only a return was initiated |
| Split shipment | One order sourced or shipped in more than one part | Ignoring extra freight and customer communication |
| Freshness | Age and completeness of data relative to its source and service-level objective | Equating a recent timestamp with a complete feed |

Quantities in this unit are integer eaches unless stated otherwise.
A case pack must be converted consistently before comparing it with eaches.
A unit may be eligible for one channel but not another, so eligibility is always scoped.

## 5. The mechanics, step by step

### 5.1 Establish the decision and service level

**Input:** request SKU, quantity, channel, destination or pickup node, desired arrival or pickup date, and current time.
**Output:** a precisely scoped promise question.

Ask: “Can we accept two units of SKU A for pickup at Store 17 by 16:00 local time on Tuesday?”
This is materially different from asking whether two units exist in the network.
State whether substitutions, split sourcing, partial fulfillment, or later dates are allowed.
A browse response may be approximate; accepting payment and confirming a delivery requires stronger controls.

Define the promise service level: a quoted arrival window, not merely the earliest possible ship date.
Include processing, store pick capacity, carrier pickup, transit, holidays, and destination restrictions.
A separate service may compute the delivery calendar; the ATP decision must use a compatible as-of time and version.
If no eligible route meets the date, positive stock does not make the request feasible.

### 5.2 Define the supply pool and ownership

**Input:** inventory records by SKU, node, status, owner, and observation time.
**Output:** eligible on-hand at each eligible node.

Start with the recorded on-hand quantity for the specific SKU and unit of measure.
Exclude damaged, recalled, quarantined, expired, customer-owned, or otherwise restricted units when those statuses are included in the raw figure.
Do not subtract them again if the input already contains only saleable inventory.
Record precisely which system owns each status and when it was updated.

Exclude nodes that cannot serve the channel or destination.
For example, a store may be pickup-only, a DC may lack hazmat authorization, and a temperature-sensitive item may not be shippable on a proposed route.
A store's inventory can support a ship-from-store order only if operational rules and capacity permit it.
If no node is eligible, return a reason code rather than a fabricated availability number.

Treat a set of substitutes as separate SKUs unless substitution is explicitly permitted and customer consent is handled.
A bundle requires every component and potentially packaging or assembly capacity.
For a simple kit, component feasibility is the minimum of `floor(component ATP / units required per kit)` across components, subject to shared-component reservations.
Do not independently promise each kit when several kits compete for the same components.

### 5.3 Reconcile the observation

**Input:** snapshot, recent events, sequence or version, count adjustments, and source health.
**Output:** a usable inventory view plus a confidence or freshness flag.

Compare source event time, ingestion time, and processing lag.
A recently ingested event can still describe an old physical movement.
Look for gaps in ordered sequences, missing point-of-sale transactions, delayed returns, and duplicated adjustments.
A snapshot must identify the event watermark it includes; otherwise applying all events again can double count transactions.

Use a store count or another authoritative process to correct physical balance.
Record count time and reconcile movements occurring during the count.
A count adjustment changes the supply estimate; it does not automatically delete customer obligations.
If the source is stale beyond policy, suppress or narrow promises, or route to a better-observed node.
Choose the fail-closed rule in advance, and document any permitted exception.

### 5.4 Classify existing claims

**Input:** order lines, temporary holds, allocations, and their lifecycle states.
**Output:** non-overlapping active claims by SKU, node, and relevant date.

A hold in checkout and the order created from that hold represent the same units during conversion.
The conversion must transfer the claim, not add a second claim.
A canceled order must not retain an active reservation after the cancellation is finalized.
A picked item is often still physically on hand until shipment; whether it is deducted in the stock snapshot or the claims ledger must be explicit.

For each claim, retain a unique ID, quantity, state, creation time, expiry if any, and order or cart reference.
Check that a claim is counted once in the pool used for ATP.
If a downstream physical movement already reduced eligible on-hand, remove or reclassify the corresponding claim in the same accounting model.
Otherwise shipment can cause a double deduction.

A channel allocation may be a partition of a common pool, not an additional physical deduction.
When one channel consumes its allocation, reduce the corresponding remaining allocation as well as the common pool.
Never let multiple channel ledgers independently promise the same common units.

### 5.5 Apply protections

**Input:** eligible stock, existing claims, safety policy, and channel entitlements.
**Output:** the quantity permitted for the requesting channel.

A simple immediate rule is:

```text
ATS = max(0, eligible on-hand - active claims - safety buffer - other protected stock)
```

All terms are units at the same SKU, node, timestamp, and accounting boundary.
The safety buffer and protected stock must be distinct; if one already includes the other, subtract it only once.
The `max` makes the customer-facing offer nonnegative, but retain the unrounded raw balance for shortage alerts.
If raw balance is negative, zero ATS does not resolve the existing shortage.

A buffer can reflect count error, uncertain demand between updates, or a decision to keep units on shelves for store shoppers.
Calibrate by SKU-node risk and update cadence where evidence supports it.
A high-value, low-count item may warrant different controls from a fast-moving item with accurate scanning.
An absolute buffer is simple but can conceal improvements and disproportionately suppress low-stock items.

Protected stock is a business entitlement, not an independent source of supply.
Set who can release it, when it expires, and how overrides are logged.
For example, three units protected for stores until 15:00 may be released to digital channels at 15:00 if pickup capacity and physical stock remain.
A release must be atomic with other claims or it can create a brief double offer.

### 5.6 Build the time-phased supply and demand ledger

**Input:** current available stock, dated receipts, existing commitments, and date rules.
**Output:** projected uncommitted inventory at each decision bucket.

Choose an hourly or daily bucket that matches the decision.
Map every event to the time it becomes usable, not merely its nominal calendar date.
A supplier delivery at 17:00 cannot support a 14:00 pickup without receiving and handling capacity.
Within a bucket, define whether receipts are available before demand; use finer time resolution if that convention changes the decision.

A basic balance is:

```text
projected balance[t] = projected balance[t-1]
                     + eligible receipts[t]
                     - committed demand[t]
```

The opening balance already excludes current active claims and protections only if those items are not also in the dated demand ledger.
The opening and future buckets must form one non-overlapping accounting boundary.
Keep unknown inbound out of firm ATP unless an explicit, approved risk policy permits it.
For a partially reliable receipt, a conservative quantity or delayed date may be used, but label it as a policy estimate rather than observed stock.

Past-due orders and receipts need special handling.
A past-due order is still an obligation; it does not disappear because its date has passed.
A past-due unreceived purchase order is not on-hand and should be revalidated or excluded, not automatically rolled forward each day as reliable supply.

### 5.7 Protect downstream commitments

For a requested date `d`, first calculate the projected balance in every bucket with no new order.
A new order of `q` units promised on `d` consumes `q` in every subsequent projected balance until the horizon ends, under a fungible-pool model.
Therefore a conservative new ATP for `d` is the minimum projected balance from `d` through the protected horizon, floored at zero.

```text
ATP[d] = max(0, min(projected_balance[t] for t >= d))
```

This definition prevents a new early order from taking stock already needed by a later order.
It assumes that later eligible receipts and obligations are known within the horizon and that stock is fungible within the pool.
It is not a universal vendor formula; a system may use receipt buckets, pegging, allocation partitions, or different horizons.
State the exact implementation and confirm its invariants.

If an existing projection is negative, raise an exception for affected commitments.
Do not hide the deficit solely by clipping the displayed ATP to zero.
A finite horizon also creates risk just beyond its end; choose it to cover the relevant commitment and replenishment window.
Reservations outside that window must not be forgotten when deciding whether to extend it.

### 5.8 Check fulfillment capability and total economics

**Input:** inventory candidates, capacity calendars, transit, product restrictions, and fulfillment costs.
**Output:** feasible options and a recommended promise.

A candidate option needs sufficient inventory before pick, sufficient labor or automation capacity, a valid carrier cutoff, and transit time compatible with the delivery promise.
For pickup, check store opening hours and pick queue as well as stock.
For delivery, account for zone, carrier service, packaging, weather or other risk policy, and nonworking days.
If the order can be split, evaluate each line and the combined cost and customer experience.

Rank feasible alternatives using contribution after incremental product, handling, packaging, transportation, and expected failure costs.
Do not optimize freight alone if it increases store labor or damages store availability.
If the top option is unavailable at reservation time, recalculate rather than silently using a different promise date.
In a regulated or contractual setting, changes to the promised service may require disclosure or customer consent.

### 5.9 Quote, reserve, and confirm

**Input:** feasible option, requested quantity, idempotency key, and current version.
**Output:** either a valid reservation or a failure requiring a refreshed quote.

A browse quote can read a materialized availability view with a stated timestamp and expiry.
At checkout, perform an atomic conditional write against the authoritative reservation pool.
The condition must check all relevant stock, protection, and time-phased constraints for the requested quantity.
The write creates exactly one claim, or creates none.
A compare-and-swap version, row lock, serializable transaction, or other correctly implemented concurrency control can provide this property within its scope.

An idempotency key must be stable for a logical request.
A retry with the same key and same payload should return the original result rather than create another reservation.
A conflicting payload for the same key should be rejected or handled explicitly.
A timeout is not proof that the server failed; query the request outcome before retrying with a new identity.

Where one order spans multiple pools, a single database transaction may not cover the whole operation.
Use a carefully designed coordinator or saga with durable state, idempotent compensation, and a deadline.
Do not claim atomicity across independent systems unless it is actually provided.
A customer confirmation should follow successful acquisition of all required claims and required payment or fraud decisions, according to the business contract.

### 5.10 Manage holds and exceptions

A hold needs an explicit expiration instant, measured using a consistent server-side time basis.
Its release must be conditional on still being active and expired.
Order conversion and expiry may race; only one transition may win.
A delayed expiry worker should not make an already expired hold indefinitely valid, but it must not release a hold that was successfully converted.

On cancellation, reduce the appropriate active claim once.
If goods have already shipped, cancellation may instead create a return process; inventory is not available until physical receipt and inspection support that status.
On pick failure, report a stock exception, update inventory evidence, and seek another feasible source or customer-approved alternative.
Do not simply delete the failed commitment to make the metric look healthy.

When inbound supply is delayed, recalculate affected commitments and prioritize under an agreed, fair policy.
Contact customers with a revised option when required by the applicable terms and law.
Escalate material shortages and document the override, decision owner, and customer impact.
The earliest recorded customer promise, not a later silently revised one, should remain available for service measurement.

### 5.11 Measure and revise

**Input:** immutable quote, hold, order, inventory, fulfillment, and exception events.
**Output:** a reconciled performance scorecard and policy changes.

Measure quote-to-hold conversion, reservation failures, expired holds, promise-date attainment, customer cancellations, pick failures, and units suppressed by buffers.
Segment by SKU, node, channel, date, and freshness band.
Record numerator, denominator, exclusions, event time, and observation window for every metric.
A promise breach rate should be calculated on orders with outcomes observed, not just orders that shipped.
Include canceled orders and customer-notified delays according to an explicit policy.

Compare buffer alternatives using a controlled experiment or a carefully adjusted observational analysis.
Monitor both sales and customer harm; do not infer causality from a single before-and-after period with changing demand.
Recalibrate when count quality, lead time, channel mix, or operating hours change.
A metric is a diagnostic, not permission to weaken existing obligations.

## 6. Worked example

### 6.1 Scenario and boundary

A home-goods retailer sells SKU L42, a lamp, through a store and a website.
All quantities below are individual lamps.
At 09:00 local time on Monday, Store S has a raw recorded on-hand balance of 27 units.
The store's status ledger identifies two damaged units and one unit in quarantine; these three units are included in the raw balance.
No other exclusions apply.

The retailer has four already confirmed customer units, held as hard reservations and not yet removed from the raw on-hand balance.
There are also two units in unexpired checkout holds.
The store protects three uncommitted units for walk-in customers and keeps a two-unit safety buffer for count uncertainty.
The protection and buffer do not overlap.
The immediate digital offer is calculated after all of these deductions.

```text
Raw recorded on-hand                       27 units
Less damaged                                2 units
Less quarantine                             1 unit
Eligible on-hand                           24 units
Less hard reservations                      4 units
Less checkout holds                         2 units
Less walk-in channel protection             3 units
Less safety buffer                          2 units
Immediate digital ATS                      13 units
```

The calculation is `27 - 2 - 1 - 4 - 2 - 3 - 2 = 13 units`.
Neither the safety buffer nor the channel protection is an existing customer order.
If the two checkout units become an order, the hold is converted; no second deduction occurs.
If a hold expires unconverted, digital ATS rises by the released quantity only if all other inputs remain unchanged.

The 13 units are not a network-wide quantity.
They apply to this SKU, this node, the website channel, and the snapshot time.
A later count, store sale, hold, release, or policy change can change the result.

### 6.2 Future supply and existing demand

For the same node and digital pool, management wants to accept promises over Monday through Thursday.
Take the 13-unit ATS as the opening balance after the existing four hard reservations and two holds have already been deducted.
Those six units must not be deducted again in the dated ledger.
The walk-in protection and safety buffer remain fixed over the horizon in this simplified example.

A confirmed transfer of eight eligible units is scheduled for Tuesday before the store's pick cutoff.
Receiving capacity is available and the receiving process is expected to finish before the cutoff.
This is an approved planning assumption, not proof of physical receipt.
An additional supplier purchase order for six units is due Wednesday but is unconfirmed, so the firm-promise policy excludes it.

Two previously accepted digital commitments, not included in the six opening claims, require five units on Wednesday and seven units on Thursday.
The internal forecast is nine units of additional demand over the week; it is neither an order nor a customer commitment and is not subtracted as if it were one.
The sales plan is to accept 15 units over the week; it is a target, not guaranteed supply.

| Bucket | Opening balance | Eligible receipts | New dated existing commitments | Closing projected balance |
|---|---:|---:|---:|---:|
| Monday | 13 units | 0 units | 0 units | 13 units |
| Tuesday | 13 units | 8 units | 0 units | 21 units |
| Wednesday | 21 units | 0 units | 5 units | 16 units |
| Thursday | 16 units | 0 units | 7 units | 9 units |

For Monday, `min(13, 21, 16, 9) = 9 units`.
For Tuesday, `min(21, 16, 9) = 9 units`.
For Wednesday, `min(16, 9) = 9 units`.
For Thursday, the balance is `9 units`.
The 13-unit immediate ATS is a snapshot measure; the nine-unit Monday ATP protects future commitments across this horizon.
They answer different questions and must not be presented as interchangeable.

Suppose a new customer asks for eight units on Monday.
All projected balances after the reservation fall by eight units: five, 13, eight, and one unit, respectively.
The customer can be offered the order if the pick and delivery capability checks also pass.
The remaining Monday ATP is one unit, not five units, because the Thursday commitment needs protection.

A simultaneous request for four units must fail the same atomic check after the eight-unit reservation succeeds.
Offering the first customer eight units and the second four units from independently cached nine-unit ATP would oversell the protected horizon by three units.
At least one request must receive a changed quantity, a later feasible option, or a rejection.
The system must not promise a later option without checking the dated ledger and capacity again.

### 6.3 Inbound disruption and response

Assume instead the eight-unit Tuesday transfer is delayed beyond Thursday before the new eight-unit order is accepted.
The projected balances become 13, 13, eight, and one unit.
Monday ATP is now one unit.
The six-unit unconfirmed purchase order remains excluded.
If the transfer delay is discovered only after the eight-unit order was confirmed, projected Thursday balance becomes negative seven units.
That is an existing-commitment shortage, not permission to erase the order.

Operations should identify affected order lines, find feasible alternative inventory, and compare the cost and timing of transfer, split fulfillment, or customer-approved delay.
A manager should follow the agreed priority and communication policy.
Preserve the original commitment and all changes in the audit trail.
If no replacement exists, the outcome may be a cancellation or delayed delivery; measure it honestly.

### 6.4 Business trade-off

Suppose protecting two additional units would prevent an estimated 0.06 failed orders per day, each costing USD 70 in contribution loss, service labor, and goodwill proxy.
Estimated avoided loss is `0.06 × USD 70 = USD 4.20 per day`.
Suppose the tighter offer also loses an estimated 0.4 otherwise profitable orders per day at USD 8 contribution each, or USD 3.20 per day.
Under these assumptions, the net expected benefit is USD 1.00 per day.
This is an illustrative expected-value calculation, not a measured result.
The outcome depends on whether the incremental orders truly disappear, on the estimate of customer harm, and on the confidence interval around both rates.
A small estimated net benefit may not justify an irreversible policy without a controlled test.

## 7. Code, query, or calculation

The following standard-library Python example reproduces the time-phased case and demonstrates a single-process atomic reservation boundary.
It uses integer units, a fixed protected horizon, end-of-bucket commitments, and receipts available before commitments in the same bucket.
The opening balance already deducts the six immediate claims, buffer, and channel protection.
The lock illustrates concurrency within one Python process only; a production multi-process or distributed service needs an authoritative transactional store or equivalent mechanism.
This sample does not calculate delivery dates, persist records, handle retries by idempotency key, or implement hold expiry.
Those omissions are explicit boundaries, not production-ready behavior.

```python
from dataclasses import dataclass
from threading import Lock


@dataclass(frozen=True)
class Bucket:
    name: str
    receipt: int
    committed: int


class PromiseLedger:
    def __init__(self, opening: int, buckets: list[Bucket]):
        if opening < 0:
            raise ValueError("opening balance must be nonnegative")
        if any(b.receipt < 0 or b.committed < 0 for b in buckets):
            raise ValueError("receipt and commitment must be nonnegative")
        self.opening = opening
        self.buckets = buckets
        self.new_orders = [0] * len(buckets)
        self.lock = Lock()

    def balances(self) -> list[int]:
        balance = self.opening
        result = []
        for bucket, new_order in zip(self.buckets, self.new_orders):
            balance += bucket.receipt
            balance -= bucket.committed + new_order
            result.append(balance)
        return result

    def atp(self, start: int) -> int:
        if not 0 <= start < len(self.buckets):
            raise ValueError("invalid bucket")
        return max(0, min(self.balances()[start:]))

    def reserve(self, start: int, quantity: int) -> bool:
        if quantity <= 0:
            raise ValueError("quantity must be positive")
        with self.lock:
            if quantity > self.atp(start):
                return False
            self.new_orders[start] += quantity
            return True


ledger = PromiseLedger(
    13,
    [
        Bucket("Monday", 0, 0),
        Bucket("Tuesday", 8, 0),
        Bucket("Wednesday", 0, 5),
        Bucket("Thursday", 0, 7),
    ],
)
print("Initial balances:", ledger.balances())
print("Initial Monday ATP:", ledger.atp(0))
print("Reserve 8 Monday:", ledger.reserve(0, 8))
print("Balances after reservation:", ledger.balances())
print("Remaining Monday ATP:", ledger.atp(0))
print("Reserve another 4 Monday:", ledger.reserve(0, 4))
```

Expected output:

```text
Initial balances: [13, 21, 16, 9]
Initial Monday ATP: 9
Reserve 8 Monday: True
Balances after reservation: [5, 13, 8, 1]
Remaining Monday ATP: 1
Reserve another 4 Monday: False
```

The method checks every downstream balance after a candidate reservation date.
A request for a later date does not consume earlier inventory availability in the model, but it still reduces later balances.
The horizon must cover obligations that require protection.
The sample does not enforce receipt confidence; the caller must construct eligible buckets under an approved policy.
It also does not handle negative starting projections as a separate shortage queue; in practice, expose those deficits before clipping ATP to zero.

For durable use, retain request IDs and payload hashes, make reserve and release transitions idempotent, and store an immutable event trail.
A transaction must guard the exact pool and all future buckets affected by a reservation.
If eligibility, capacity, or inbound confidence changes outside that transaction, reconcile the impact on existing commitments.
Test race conditions with multiple clients, process crashes, duplicate messages, expiry races, and replayed snapshots in the production architecture.

## 8. Data, standards, and regulatory context

### 8.1 Minimum data contract

| Data element | Example | Why needed | Control |
|---|---|---|---|
| Product identity | SKU L42 | Distinguishes exact variant | Validate mapping and effective dates |
| Location | Store S | Determines stock and service options | Maintain active location master |
| Unit of measure | each | Prevents case/eaches mismatch | Store conversion factors with version |
| On-hand quantity | 27 units | Starting physical estimate | Reconcile source and transactions |
| Status | damaged, quarantine | Determines eligibility | Define mutually exclusive states |
| Owner | retailer or third party | Determines right to sell | Enforce ownership restrictions |
| Observation time | Monday 09:00 local | Measures source age | Preserve time zone and source time |
| Ingestion time | Monday 09:01 local | Measures pipeline lag | Monitor gaps and late arrivals |
| Version or sequence | source revision | Supports ordering and replay | Detect missing or duplicate events |
| Receipt ID and date | transfer arriving Tuesday | Supports future supply | Record confidence and usable time |
| Claim ID and state | checkout hold | Prevents double counting | Enforce unique transition identity |
| Claim expiry | 09:15 local | Controls temporary lock | Evaluate using trusted server time |
| Customer promise | Thursday by 20:00 | Determines obligation | Preserve original and revisions |
| Calendar | carrier cutoff | Establishes feasibility | Version holidays and local time |
| Policy version | buffer rule 4 | Explains decision | Attach version to every quote |
| Reason code | source stale | Explains suppressed availability | Standardize and monitor frequency |

For each field, define owner, source system, units, refresh frequency, allowed values, retention, and recovery behavior.
An apparently complete feed can still omit an entire node; completeness monitoring must cover expected entities as well as latency.
A snapshot and its event watermark must travel together to avoid applying already included events twice.

### 8.2 Identification and exchange

GS1's Global Trade Item Number (GTIN) identifies trade items, while Global Location Numbers (GLNs) identify parties or locations where adopted.
These identifiers can make cross-enterprise data matching more reliable; they do not automatically prove physical inventory accuracy.
A GTIN is not necessarily the same as a retailer's internal SKU.
Maintain mappings at the correct pack and variant level, including effective dates.

GS1 Electronic Product Code Information Services (EPCIS) describes a standard for sharing visibility event data, including what, when, where, why, and how in its information model.
It can support traceability and event exchange across partners.
It does not prescribe this topic's ATS formula, establish a universal inventory authority, or guarantee that every event is present.
Adoption and compliance depend on trading-partner and applicable legal requirements.

Electronic data interchange (EDI) can communicate purchase orders, acknowledgments, shipment notices, and other business documents using agreed transaction sets.
A shipment notice is evidence about a planned or dispatched shipment, not proof that sellable inventory has been received and inspected.
Match document quantities and identifiers to physical receiving events and resolve discrepancies.
Implementations differ by standard, version, and partner agreement; do not infer a field's meaning from its label alone.

### 8.3 Consumer commitments and privacy

In the United States, the Federal Trade Commission's Mail, Internet, or Telephone Order Merchandise Rule addresses shipment representations, delays, and customer options for covered orders.
Consult the current rule and authoritative guidance for scope, timing, exceptions, and obligations; this course does not replace legal advice.
A business must not treat its internal ATP estimate as a substitute for compliance with applicable customer-facing shipment rules.
Other jurisdictions may apply different consumer, cancellation, privacy, and product-safety requirements.

A promise record may include an address, order history, and customer contact details.
Use access controls, purpose limitation, retention policies, and applicable privacy requirements.
Analytical datasets should minimize personal data and use aggregation or pseudonymous identifiers where appropriate.
Pseudonymization alone does not necessarily make a record anonymous.

Record policy changes and overrides with a named role, reason, time, and approval.
Audit logs should make it possible to reconstruct what was known when the customer was promised a date.
A current inventory snapshot is not sufficient evidence of a historical decision.

### 8.4 Decision record: an auditable contract

A reconstructable decision record joins several independent pieces of evidence.
The record need not duplicate every source payload if immutable references remain available.
It must identify which versions were used for the actual decision.

| Record field | Example value | Audit question |
|---|---|---|
| Request ID | `req-81` | Which logical request initiated this decision? |
| Request type | reserve | Was this a quote or an exclusive claim? |
| Request quantity | 2 eaches | What did the customer request? |
| Requested service | pickup | What kind of fulfillment was evaluated? |
| Requested window | Tuesday 14:00–16:00 local | Which service commitment mattered? |
| SKU and pack | L42, each | Which exact item was evaluated? |
| Candidate node | Store S | Which pool supplied the units? |
| Source watermark | inventory version 518 | Which stock events were included? |
| Existing claim watermark | claims version 902 | Which prior claims were considered? |
| Receipt scenario | transfer T12, 8 eaches | Which future supply was assumed? |
| Buffer version | rule B4 | Which protection policy was applied? |
| Calendar version | store calendar C9 | Which operational hours were used? |
| Capacity version | pick capacity P3 | Which labor assumption was used? |
| Computed ATP | 9 eaches | What did the algorithm calculate? |
| Reservation result | accepted, 2 eaches | What claim was actually created? |
| Idempotency key | `cart-43-attempt-1` | Can a retry be recognized? |
| Expiration instant | Tuesday 09:15 local | When does a temporary hold end? |
| Promise communicated | Tuesday 14:00–16:00 local | What did the customer see? |
| Policy override | none | Was a normal control bypassed? |
| Actor or service | checkout service | Who initiated the transition? |
| Decision timestamp | Tuesday 09:00 local | When was the decision made? |
| Outcome link | order line O17-1 | What eventually happened? |

The example identifiers are fictional and are not personal information.
The order link should remain stable if a fulfillment location changes.
A location change does not justify replacing the original promise or reservation history.
If a rule version cannot be recovered, the organization cannot fully explain why the offer was made.

### 8.5 Reconciliation ownership

Data ownership must be explicit at every boundary.
The inventory team owns the interpretation of physical statuses.
The order team owns accepted customer order states.
The reservation service owns the active claim state it creates.
The fulfillment team owns evidence of picks, substitutions, shortages, and handoffs.
The analytics team owns the reproducible measurement definitions.
An operational owner must resolve mismatches rather than only reporting them.

Use a reconciliation cycle appropriate to transaction volume and risk.
For each SKU-node pool, compare the inventory snapshot with the last applied event position.
Compare active accepted orders with their inventory claims.
Compare open claims with valid carts or order lines.
Compare completed shipments with the corresponding inventory movements.
Compare returned merchandise with receipt and inspection events.
Investigate every material unexplained difference.

An unmatched accepted order is potentially under-reserved.
An unmatched claim is potentially suppressing available stock.
A negative balance is a shortage even when the storefront displays zero.
A late event can change a historical reconstruction, so retain both event time and arrival time.
The correction must be idempotent and traceable to a specific discrepancy.

### 8.6 Serving different clients

The product listing page, store associate, marketplace, and checkout may use different latency budgets.
They should still share the same definition of the underlying resource.
Otherwise an associate can promise stock that the website has already reserved.
Record whether a response is a quote, an active hold, or a confirmed commitment.
Do not let a client interpret a positive quote as an exclusive allocation.

For a read-only quote, provide the observed time, scope, and validity period where useful.
For a reservation, return the claim ID, quantity, expiration, and status.
For a rejection, return a useful reason without revealing sensitive operational information.
For a partial availability response, make clear whether partial fulfillment is allowed.
For a customer-visible alternative, recalculate price and delivery terms if they differ.

An external marketplace may require updates asynchronously.
Feed delay can produce customer orders after the local ATP has changed.
The business can reserve a bounded allocation for that marketplace or use another explicit risk policy.
The marketplace allocation cannot also be offered in full by the direct channel.
Release unused allocations under an agreed schedule and with an auditable transition.

### 8.7 Managing exceptions through a queue

A decision system needs a work queue as well as a formula.
Without a work queue, negative balances remain mathematical observations rather than operational actions.
Each exception should identify the affected pool, customer commitments, due times, and responsible team.
The queue should distinguish a new data defect from an already known incident.
Link multiple affected orders to the same root cause when evidence supports it.

Prioritize first by customer impact and time remaining to recover.
Then consider product restrictions, feasible alternatives, and contractual obligations.
Treat protected customer groups and legally required priorities consistently with applicable policies.
Do not prioritize solely by the margin of an order if that contradicts customer commitments.
Document the decision and the facts available at the time.

Possible actions include finding another eligible node, expediting a transfer, requesting an approved substitution, or offering a revised service.
Each action has a cost and may consume capacity needed by another customer.
Check the resulting ledger before accepting an alternative.
If the recovery fails, preserve the incident and customer communication history.
Close the incident only after the outcome and any stock correction are recorded.

### 8.8 Worked reconciliation sequence

Consider an independent pool of 10 eligible eaches at 10:00.
There are no future receipts, safety buffers, or other claims in this miniature sequence.
The store and web channel share this one pool.
The arithmetic below tracks accounting events rather than predicting physical behavior.

1. A web cart receives an active hold for three eaches.
2. Available quantity becomes seven eaches.
3. The cart converts to a confirmed order for the same three eaches.
4. Available quantity remains seven eaches because the claim changes state.
5. A store sale records two additional eaches as physically sold.
6. Eligible on-hand becomes eight eaches while the order claim remains three.
7. Available quantity becomes five eaches.
8. The order is picked, but the physical inventory snapshot still includes the picked units.
9. Available quantity remains five eaches while the claim is active.
10. Shipment records a three-unit physical decrement and retires the corresponding claim.
11. Eligible on-hand becomes five eaches and active claims become zero.
12. Available quantity remains five eaches.

The shipment transaction may be implemented as multiple events in a real system.
If so, the read model must avoid treating the intermediate state as a new saleable surplus.
A temporary double deduction is also possible if the physical decrement arrives before the claim is retired.
Both cases require an explicit transition convention and monitoring.
The order of events in a message queue is not necessarily the order of physical events.

Now suppose the customer requests a return for one shipped each.
The request alone does not increase the five-unit available quantity.
After the item is physically received, inspected, and marked eligible, on-hand becomes six eaches.
Only then can the return contribute to ATP under this simplified policy.
If the item is damaged on return, available quantity remains five eaches.

### 8.9 Demand, policy, and service interactions

A customer may abandon a cart because the hold duration is too short.
A very long hold can instead lock inventory for customers who will never buy.
Compare conversion, expiry, repeated holds, and stock suppression across hold-duration policies.
Account for different checkout steps, payment authentication, and customer accessibility needs.
Do not infer the best duration from the number of expired holds alone.

A fixed safety buffer may remove all digital availability for a low-stock SKU.
That can be appropriate when stock records are unreliable or the cost of a missed promise is high.
It can be wasteful when accurate serial scanning makes the final unit dependable.
Use the risk of the marginal unit, not just average accuracy across the whole catalog.
Review low-stock behavior separately from high-stock behavior.

Store protection can improve local availability but displace profitable digital demand.
The correct measure is the incremental value of the protected stock in its alternative use.
Estimate whether a protected unit would actually sell in the store before replenishment.
Consider labor, markdown exposure, cross-channel substitution, and customer retention.
Use a bounded override process for exceptional orders rather than ad hoc individual requests.

An order can be inventory-feasible and still be operationally infeasible.
Suppose one picker has capacity for six units before cutoff and four are already scheduled.
Even if ten units are in stock, only two additional units fit that capacity constraint.
An order for three units needs another feasible slot, another node, or an approved later promise.
Reserving stock without securing relevant capacity can create a hidden overcommitment.

Likewise, carrier capacity and service may vary by destination.
A node that can hand over a package Tuesday evening may not deliver to a remote address by Wednesday.
Check promised arrival against the current route calendar.
If the carrier service is unavailable, report a feasible alternative rather than silently extending the date.
Preserve which service and transit assumptions supported the original offer.

### 8.10 Policy change controls

Before changing a buffer or reservation rule, write down the intended business outcome.
Specify the affected SKU-node-channel population.
State the baseline and the proposed policy in quantities, time windows, and source versions.
List customer groups and commitments that the change could affect.
Identify a rollback threshold based on both service and lost-sales outcomes.

Run the new policy in shadow mode when it is safe and useful.
Shadow mode computes a decision without creating a customer promise.
Compare its decision with the existing policy using the same historical inputs.
Include cases where the new rule would reject an order the current rule accepted.
Also include cases where it would offer stock the existing policy suppressed.

Shadow results cannot prove how customers or operations would respond to a changed offer.
For causal evidence, assign comparable eligible traffic or pools to policies when feasible.
Avoid assigning overlapping pools to different treatments if stock can be consumed across them.
Use an observation period long enough to see cancellations and delayed fulfillment.
Predefine the primary outcome, guardrails, and the treatment of incomplete orders.

After release, monitor customer impact and operational exceptions.
Compare the actual events with the decisions and assumptions in the policy record.
Roll back or pause if a predefined guardrail is breached.
Existing customer commitments remain obligations even if the policy version is rolled back.
Retain the history of which version generated each accepted order.

### 8.11 Source quality and policy governance

An inventory quantity can be numerically precise while operationally unreliable.
For instance, a balance of 14 eaches may be correct as of yesterday but incomplete today.
Freshness must be assessed against the volatility of the pool and the business consequence of an error.
A slow-moving, periodically counted item and a busy store shelf may justify different thresholds.
Use observed error and incident rates to revise those thresholds.

An inbound receipt date can also look precise without being reliable.
Track supplier promised date, shipment date, actual arrival, and usable receipt separately.
Compare the full distribution of lateness and short receipts, not just the average.
Use data by supplier, lane, product class, and season only where sample sizes support it.
When evidence is sparse, make the uncertainty visible instead of manufacturing precision.

Governance should assign an owner to every decision rule.
The owner approves changes, understands the customer impact, and receives exception reports.
The engineering owner ensures the rule is implemented consistently and safely.
The data owner ensures the inputs have definitions, lineage, and monitoring.
The operations owner confirms that the promised work is actually feasible.

Approval does not replace measurement.
A rule can be implemented exactly as specified yet have unacceptable business consequences.
Review customer outcomes, staff workload, suppressed availability, and contribution together.
Correct source defects where possible rather than continually increasing safeguards.
Retire a rule when it no longer addresses an observed risk.

## 9. Industry practice and case evidence

### 9.1 What evidence supports

Target's 2023 annual report describes its stores as fulfillment hubs for digital orders and discusses services including Order Pickup, Drive Up, and same-day delivery.
This establishes a public omnichannel operating model with stores participating in digital fulfillment.
It does not disclose the exact ATS formula or prove the effectiveness of any particular safety buffer.
**Inference:** a retailer combining store and digital demand needs a policy for competing claims on store stock.
The inference is a design implication, not a reported Target implementation detail.

Walmart's published help material describes pickup and delivery ordering and notes that substitutions or out-of-stock situations may occur.
The public customer-facing process illustrates that availability and physical picking can diverge.
It does not establish Walmart's internal reservation algorithm, expected out-of-stock rate, or financial outcome.
**Inference:** pick exceptions require customer communication and operational recovery even when a website accepted an order.

Salesforce's B2C Commerce documentation describes inventory availability concepts and reservation behavior in a particular commercial platform.
It is useful for understanding one implementation, but platform terminology and default behavior are not universal.
Review the product version and configuration before mapping the platform's fields to your own ATP ledger.
Vendor documentation is evidence of documented product behavior, not independent proof of customer impact.

### 9.2 Typical operating pattern

Many omnichannel designs use a responsive read model for browsing and a stronger reservation operation during checkout.
This pattern trades freshness, latency, and cost; it is not guaranteed to be present at any named company above.
High-volume browse traffic can tolerate bounded staleness if users are told availability is subject to change.
A confirmed order needs a durable claim and a feasible service plan.

Store staff may pick from the same shelf as walk-in customers.
That shared stock creates competition between point-of-sale events and online orders.
Faster scanning, frequent counts, and modest channel protection can reduce conflict, but each has a cost.
The right balance depends on demand, count accuracy, pick time, and customer outcomes.

A future receipt may be usable for a promise when the inbound is sufficiently reliable, the receiving operation has capacity, and the customer date includes appropriate uncertainty.
A supplier acknowledgement may justify a different risk class from a purchase order that has not been acknowledged.
Different classes should have visible eligibility and escalation rules.
A probability estimate should be calibrated against actual usable receipt times, not only scheduled dates.

### 9.3 Evidence hierarchy and interpretation

Public annual reports and customer help pages describe selected business practices; they do not expose all operational details.
Vendor documentation describes product features but is not independent evidence of business results.
Internal order, inventory, and fulfillment event data can show the local failure mechanism, but observational comparisons may be confounded by demand or assortment changes.
A randomized or carefully designed quasi-experimental test gives stronger causal evidence for a buffer change when operationally and ethically feasible.

State whether a result is a public fact, an illustrative assumption, an internal observation, or an inference.
For example, a drop in oversells after a buffer increase does not alone establish that the buffer caused the improvement if data latency also improved.
Compare suppressed sales, cancellation rates, contribution, and customer outcomes over comparable populations.
Avoid applying one retailer's unverified rule to a different assortment or operating model.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Phantom stock | Pick failures rise at low balances | Record exceeds physical stock | Cancellations and wasted labor | Counts, root-cause analysis, calibrated buffer |
| Duplicate claim | Orders appear twice in reservation audit | Hold and order both deducted | Lost sales from false stockouts | Atomic hold conversion and reconciliation |
| Missing claim | Orders exceed reservation records | Acceptance bypasses claim service | Oversell | Enforce a single commitment boundary |
| Cache race | Two checkouts succeed against one unit | Both read stale availability | Double promise | Conditional authoritative reservation |
| Expiry race | Confirmed orders lose holds | Expiry and conversion both succeed | Stock reassigned to another order | Conditional state transition |
| Duplicate retry | One request creates two holds | New identity on timeout retry | Lost availability or duplicate order | Stable idempotency key and result lookup |
| Future receipt optimism | Orders late despite ample projected stock | Unconfirmed inbound treated as certain | Missed promises | Eligibility tiers and usable-time tracking |
| Downstream starvation | Late orders fail after early sales | New order ignores later commitments | Broken existing promises | Minimum future balance or equivalent pegging |
| Double status deduction | ATS lower than reconciliation | Blocked stock excluded twice | Unnecessary lost sales | Document snapshot boundary |
| Double shipment deduction | ATS falls when order ships | Claim and physical decrement overlap | False stockout | Coordinated ledger transition |
| Misapplied channel buffer | All channels show protected units | Each pool treats protection as shared | Oversell or inequitable allocation | Partition and atomic release rules |
| Wrong units | Case quantities appear as eaches | Conversion not applied | Material overstatement | Versioned unit conversions |
| Stale point-of-sale feed | Lag or sequence gaps increase | Store sales arrive late | Online promise on sold units | Freshness thresholds and fail-closed policy |
| Wrong time zone | Orders miss a local cutoff | UTC and local calendar mixed | Late delivery | Zoned timestamps and calendar tests |
| Unavailable labor | On-time inventory but unpicked orders | Stock check ignores capacity | Missed pickup window | Capacity-aware promise and queue control |
| Return counted too early | Saleable stock rises on return initiation | No physical receipt or inspection | Invalid resale promises | Add only after eligible disposition |
| Clipped shortage hidden | Displayed ATP is zero while breaches rise | Negative raw balance is discarded | Existing orders receive no intervention | Monitor negative balances and exception queue |
| Inconsistent snapshot replay | Balances jump after restart | Events included in snapshot replayed again | Incorrect available stock | Persist snapshot watermark and deduplicate |
| Incorrect bundle logic | Kit orders fail despite component stock | Shared components promised to multiple kits | Oversold components | Reserve components together |
| Silent promise revision | Reported on-time rate improves suddenly | Original date overwritten | Misleading service measurement | Preserve original and revised promises |

For each incident, capture the affected SKU, node, pool, promise, event sequence, and customer outcome.
Separate a local operational mistake from a systematic policy defect.
A permanent buffer increase is not a substitute for correcting a broken event feed or duplicate reservation logic.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| SKU units are fungible within a pool | Allows a single aggregate balance | Lot, serial, expiry, or customer restrictions | Compare eligibility at unit or lot level |
| All active claims are captured | Prevents double selling | Offline orders or bypassing channels | Reconcile order lines to claims |
| Exclusions do not overlap | Prevents double deductions | Saleable snapshot also subtracts quarantine | Trace each status to the source |
| Receipt timing is usable timing | Supports future commitments | Unloading or inspection delay | Compare usable timestamps with scheduled dates |
| Demand buckets cover obligations | Protects future orders | Orders outside horizon or late events | Audit commitments through the full horizon |
| Each event is applied once | Maintains conservation | Duplicate replay or missing watermark | Rebuild balances and compare sequence IDs |
| Buffer remains available | Supports uncertainty protection | Unauthorized overrides or store sales | Audit overrides and physical outcomes |
| Channel protection is exclusive | Prevents shared-pool double count | Independent systems sell the same partition | Reconcile allocations against common stock |
| Capacity calendar is valid | Links stock to deliverability | Labor absence or carrier disruption | Compare planned and actual throughput |
| Returns are inspected before resale | Keeps ineligible goods out of stock | Automatic credit posts inventory immediately | Trace returns from receipt to disposition |
| Lock or transaction covers all claims | Prevents concurrent oversell | Multiple services write independent pools | Concurrent integration and failure tests |
| Expected costs are comparable | Supports buffer economics | Missing customer harm or displaced demand | Sensitivity analysis and controlled experiment |

A deterministic ledger does not produce a probability of on-time delivery by itself.
Receipt uncertainty, inventory error, demand, labor, and transit can be correlated.
Simply multiplying independent probabilities without evidence can overstate confidence.
For high-impact promises, model joint uncertainty or use conservative scenarios and monitor calibration.

The worked example assumes no replenishment or obligation after Thursday.
Real policies need a horizon long enough to protect the commitments they accept.
Extending the horizon can reveal a deficit that a short calculation hides.
No safety buffer guarantees stock, and no reservation creates physical inventory.

## 12. Exercises

### Exercise 1 — Vocabulary and boundaries

**Time:** 12 minutes.  
**Deliverable:** a table classifying six facts as physical supply, exclusion, claim, policy protection, forecast, or customer commitment.

Facts: 18 recorded units; two quarantined units; a three-unit checkout hold; a two-unit walk-in protection; a forecast of seven units of demand; and a confirmed order for four units.
Explain whether a checkout quote necessarily creates a claim.

**Acceptance criteria:** each fact is classified once; the answer identifies that a forecast is not a confirmed order and a quote need not reserve stock.

### Exercise 2 — Immediate ATS

**Time:** 15 minutes.  
**Deliverable:** a calculation in units and a sentence describing its scope.

A node records 31 units including three damaged units.
There are five hard-reserved units and two held units, all included in that balance.
The channel protection is four units and the separate buffer is three units.
Calculate ATS for the digital channel.

**Acceptance criteria:** all deductions occur exactly once; the result is nonnegative and labeled in units; the answer distinguishes the digital pool from the total network.

### Exercise 3 — Future commitments

**Time:** 20 minutes.  
**Deliverable:** a daily ledger and ATP for Monday and Wednesday.

Opening available stock is 12 units.
Tuesday has an eligible receipt of six units.
Wednesday has an existing commitment of four units and Thursday has an existing commitment of nine units.
Use the minimum future balance through Thursday.
Calculate the balances and the maximum new Monday commitment.
Explain why a 12-unit Monday offer is unsafe.

**Acceptance criteria:** the answer shows all four balances, the downstream minimum, and the effect of an excessive new order on Thursday.

### Exercise 4 — Lifecycle and expiry

**Time:** 20 minutes.  
**Deliverable:** a state diagram with quote, active hold, converted order, expired hold, canceled order, and fulfilled order.

Show how a hold can become an order without a second deduction.
Explain what must happen when conversion and expiry occur concurrently.
Describe when a shipped return may become saleable.

**Acceptance criteria:** only one transition wins the expiry race; release is idempotent; return stock is not saleable merely because a customer requests a return.

### Exercise 5 — Concurrent checkout diagnosis

**Time:** 20 minutes.  
**Deliverable:** a short incident explanation and a control design.

Two users each read one remaining unit from a five-second-old cache.
Both submit a checkout request and both are confirmed.
A retry after timeout creates a second hold for one user.

**Acceptance criteria:** distinguish stale reads from duplicate retries; require an atomic check-and-reserve and stable idempotency identity; describe an auditable customer recovery process.

### Exercise 6 — Policy critique

**Time:** 25 minutes.  
**Deliverable:** a decision memo with three risks, three controls, and a fallback.

A proposal counts every open purchase order on its planned arrival date, releases all store protection at midnight, and permits any order to use the same network-wide pool.
Assess promise reliability, time zones, and shared supply.

**Acceptance criteria:** classify inbound confidence, test feasible arrival and receiving time, define shared-pool controls, and state an exception owner.

### Exercise 7 — Buffer economics

**Time:** 15 minutes.  
**Deliverable:** a calculation in USD per day and a recommendation with one uncertainty.

A one-unit buffer is expected to avoid 0.08 failed orders per day at USD 50 per failure.
It is also expected to suppress 0.5 profitable orders per day at USD 6 contribution each.
Compute the expected daily net benefit.

**Acceptance criteria:** show avoided loss and lost contribution separately; label the estimate and name a method for validating the assumptions.

### Exercise 8 — Scorecard and recovery design

**Time:** 30 minutes.  
**Deliverable:** a one-page proposal for metrics, event data, escalation, and experiment design.

A retailer reports that 99% of shipped orders arrived on time, but it excludes canceled orders and overwrites the original promise when an order is delayed.
Design a scorecard that exposes customer experience and a process to address a negative projected balance.

**Acceptance criteria:** define numerator, denominator, observation window, and segmentation; preserve original promises; include cancellations and shortages; propose a fair prioritization and customer communication process.

## 13. Self-check answers

### Answer 1

Recorded on-hand is the physical supply estimate, not proof of physically sellable stock.
Quarantined units are an eligibility exclusion.
A checkout hold is an active claim if the system actually reserves it.
Walk-in protection is a policy partition, not a customer order.
The seven-unit forecast is an estimate of possible future demand.
The confirmed order is a customer commitment and normally has a corresponding claim.
A quote can be advisory and may expire without any reservation.

### Answer 2

`31 - 3 - 5 - 2 - 4 - 3 = 14 units`.
The 14 units are digital ATS for the specified node and snapshot, assuming all deductions are distinct and no other constraints apply.
A fulfillment date still requires capacity and transit checks.

### Answer 3

The projected balances are 12 units Monday, 18 units Tuesday, 14 units Wednesday, and five units Thursday.
Monday ATP is `min(12, 18, 14, 5) = 5 units`.
Wednesday ATP is `min(14, 5) = 5 units`.
A new 12-unit Monday commitment would leave Thursday at `5 - 12 = -7 units`.
That seven-unit deficit affects existing obligations even though the Monday snapshot initially showed 12 units.

### Answer 4

A sound diagram permits `quote -> active hold -> converted order -> fulfilled`, and separately `active hold -> expired`.
Cancellation of an accepted order triggers a state-appropriate release only once.
Conversion and expiration must be competing conditional transitions on the same active hold; one succeeds and the other observes the resulting state.
A returned item becomes eligible only after physical receipt and whatever inspection and disposition the policy requires.
**Rubric:** award credit for exclusive transitions, correct release timing, and an audit trail; do not award full credit to a diagram that treats the quote as a guaranteed claim.

### Answer 5

The cache read is not an exclusive claim, so two readers can both observe the same unit.
An atomic conditional reservation against the shared authority allows at most one one-unit claim to succeed.
The retry must use the same idempotency key and recover the original result rather than create a second hold.
If both customers already received confirmations, preserve both commitments, search for alternative supply, and communicate the outcome under policy and applicable terms.
**Rubric:** a cache refresh alone is insufficient; the answer must address both concurrency and retry identity.

### Answer 6

An open purchase order may be late, canceled, short, or unusable pending inspection.
Classify receipts by confidence and usable time, exclude or conservatively adjust uncertain supply, and monitor actual receipt performance.
Use each node's local time zone and calendar for any release event.
A shared pool needs one authority or a provably exclusive partition; a network sum alone ignores location, transit, and capacity.
Assign a named owner for exceptions and use a safe fallback such as a later quote or unavailable status when data is unreliable.
**Rubric:** accept alternatives if they preserve existing obligations and explicitly account for supply uncertainty and shared claims.

### Answer 7

Avoided loss is `0.08 × USD 50 = USD 4 per day`.
Lost contribution is `0.5 × USD 6 = USD 3 per day`.
Estimated net benefit is `USD 4 - USD 3 = USD 1 per day`.
A controlled experiment can assess whether failures and lost orders actually change by those amounts.
The estimate should be reconsidered if displaced customers buy later or failure costs differ by segment.

### Answer 8

One useful outcome is the fraction of accepted orders meeting the original promise among orders whose promised window has elapsed, with cancellations and partial outcomes handled explicitly.
Publish the numerator, denominator, exclusions, cutoff date, and separately the cancellation and revised-promise rates.
Segment by node, channel, SKU risk, and data freshness.
Preserve original promise events and reason-coded revisions.
A negative balance triggers an owner to identify affected orders, evaluate alternatives, apply a documented priority, and communicate with customers.
**Rubric:** credit clear definitions, immutable history, action ownership, and a test with a comparable control; a shipped-only on-time rate is not an adequate customer outcome metric.

## 14. Interview questions

1. **What is the difference between ATS and ATP?**
   A strong answer gives a scoped operational definition and discusses timing and existing obligations.
   A weak answer assumes a universal formula or equates both with on-hand.
2. **Why can a store show stock yet be unable to promise it?**
   A strong answer covers eligibility, claims, channel protection, stock accuracy, and capacity.
   A weak answer considers only the stock count.
3. **How do you prevent two customers from buying the last unit?**
   A strong answer distinguishes an advisory read from an atomic conditional reservation.
   A weak answer relies only on a fast cache.
4. **What happens when a hold converts to an order?**
   A strong answer transfers one claim atomically and handles expiry races.
   A weak answer adds a second reservation or assumes a timer always fires on time.
5. **How do you handle a retry after a timeout?**
   A strong answer uses a stable idempotency key and retrieves the previous outcome.
   A weak answer assumes timeout means no write occurred.
6. **Why can Monday ATP be lower than Monday inventory?**
   A strong answer protects downstream commitments and calculates the minimum future balance.
   A weak answer ignores orders already promised for later dates.
7. **When should future supply be eligible?**
   A strong answer distinguishes planned, confirmed, dispatched, received, and usable supply, with explicit risk rules.
   A weak answer counts every open purchase order at face value.
8. **How would you choose a safety buffer?**
   A strong answer compares marginal failure reduction with lost contribution and tests uncertainty.
   A weak answer applies the same arbitrary number everywhere indefinitely.
9. **How do you measure promise accuracy?**
   A strong answer preserves the original commitment and includes canceled and delayed outcomes with clear denominators.
   A weak answer measures only successfully shipped orders against revised dates.
10. **What does a negative projected balance mean?**
    A strong answer identifies a shortage affecting existing obligations and proposes recovery.
    A weak answer clips the value to zero and declares the problem solved.
11. **How would you account for returns?**
    A strong answer waits for receipt, inspection, and eligible disposition before adding stock.
    A weak answer adds stock when a return label is created.
12. **What changes in a distributed implementation?**
    A strong answer describes durable authority, idempotency, transaction scope, and compensating actions without claiming unsupported global atomicity.
    A weak answer assumes a process-local lock protects every server.

## 15. Key terms

- **[Allocation](#4-definitions-and-vocabulary):** policy assignment of stock or capacity.
- **[ATP](#5-the-mechanics-step-by-step):** new quantity that can be promised for a defined date and pool.
- **[ATS](#5-the-mechanics-step-by-step):** policy-defined quantity currently offered for sale.
- **[Backorder](#4-definitions-and-vocabulary):** accepted demand to be fulfilled later.
- **[Buffer](#5-the-mechanics-step-by-step):** stock deliberately protected from offer.
- **[Channel protection](#5-the-mechanics-step-by-step):** entitlement reserved for a specified channel.
- **[Claim](#5-the-mechanics-step-by-step):** recorded reservation or other obligation against a pool.
- **[Commitment](#3-the-core-mental-model):** obligation communicated or accepted for a customer.
- **[Concurrency](#5-the-mechanics-step-by-step):** simultaneous operations competing for shared supply.
- **[Cutoff](#4-definitions-and-vocabulary):** latest permitted operational time for a service.
- **[Eligibility](#5-the-mechanics-step-by-step):** whether an item can serve a specific request.
- **[Event time](#8-data-standards-and-regulatory-context):** time the underlying event occurred.
- **[Expiry](#5-the-mechanics-step-by-step):** end of a temporary hold's validity.
- **[Forecast](#6-worked-example):** estimated future demand, not an order.
- **[Freshness](#8-data-standards-and-regulatory-context):** age and completeness of information.
- **[Future receipt](#5-the-mechanics-step-by-step):** inbound stock expected to become usable later.
- **[Hard reservation](#4-definitions-and-vocabulary):** controlled claim associated with an obligation.
- **[Hold](#5-the-mechanics-step-by-step):** temporary claim with an explicit lifecycle.
- **[Idempotency](#5-the-mechanics-step-by-step):** ensuring retry has one logical effect.
- **[Node](#3-the-core-mental-model):** inventory or fulfillment location.
- **[On-hand](#4-definitions-and-vocabulary):** recorded quantity physically present.
- **[Phantom inventory](#10-failure-modes-and-red-flags):** recorded stock not actually usable or present.
- **[Promise date](#4-definitions-and-vocabulary):** customer-facing service date or window.
- **[Quote](#3-the-core-mental-model):** proposed service option, not necessarily a claim.
- **[Reservation](#5-the-mechanics-step-by-step):** controlled claim against available supply.
- **[Return disposition](#5-the-mechanics-step-by-step):** decision determining whether returned stock is usable.
- **[SKU](#4-definitions-and-vocabulary):** specific commercial product variant.
- **[Snapshot watermark](#8-data-standards-and-regulatory-context):** event position included in a stock snapshot.
- **[Time-phased ledger](#5-the-mechanics-step-by-step):** projected supply less obligations by period.
- **[Unit of measure](#8-data-standards-and-regulatory-context):** quantity basis such as each or case.

## 16. Sources

1. GS1, [GTIN Management Standard](https://www.gs1.org/1/gtinrules/). Product identification rules; consult the applicable version and implementation guidance.
2. GS1, [Global Location Number](https://www.gs1.org/standards/id-keys/gln). Identification of parties and locations; not an inventory accuracy standard.
3. GS1, [EPCIS and CBV](https://www.gs1.org/standards/epcis). Visibility event exchange standards and specifications.
4. U.S. Federal Trade Commission, [Business Guide to the FTC's Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule). Regulatory guidance for covered merchandise orders; consult current law and counsel.
5. Target Corporation, [2023 Annual Report](https://corporate.target.com/getmedia/eba4c76f-a33f-4f4c-9dce-683e907ac4e1/2023-Annual-Report-Target-Corporation.pdf). Public description of the retailer's operating model; not evidence of its internal ATP formula.
6. Walmart Business, [Substitutions for Store Pickup and Delivery Items](https://business.walmart.com/help/article/substitutions-for-store-pickup-and-delivery-items/b94b3c9413b14315acd003296238cdaa). Customer-facing substitution guidance; availability and implementation can change.
7. Salesforce, [Customize Inventory Management](https://developer.salesforce.com/docs/commerce/b2c-commerce/guide/b2c-inventory-for-developers.html). Vendor documentation; confirm product version and settings before applying.
8. Silver, Edward A., David F. Pyke, and Douglas J. Thomas, *Inventory and Production Management in Supply Chains*, 4th edition, CRC Press, 2016. Inventory and replenishment background; not a specification of this example's ATP algorithm.

The example retailer, quantities, costs, and business outcomes in this topic are illustrative.
Public company materials support only the specific claims attributed to them; causal implications are labeled as inference.
Regulatory and standards references should be checked for current applicability before deployment.

## 17. Links

- [Module 23 topic index](_index.md).
- [Module 23 survey](../../../course/23_omnichannel_fulfillment_returns.md).
- [Inventory visibility](./_index.md#3-topics) — sibling topic listed in the module index.
- [Order routing and distributed order management](./_index.md#3-topics) — sibling topic listed in the module index.
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md) — relevant practice in modeling and decision evaluation.
