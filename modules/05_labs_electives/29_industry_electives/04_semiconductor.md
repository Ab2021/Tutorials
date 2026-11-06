# Semiconductor supply-chain planning

> Module 29 — Industry Electives: Sector Primers
> Topic 04 of 08
> Estimated study time: 4 hours
> Prerequisites: Modules 14, 15, and 28; basic inventory and probability
> Version 2026.5

## 1. Why this topic exists

A semiconductor planner must commit scarce manufacturing capacity before demand is certain.
A chip can pass through fabrication, wafer sort, assembly, final test, and distribution before a customer receives a usable part.
Each stage has its own lead time, capacity limit, yield, and ownership boundary.
A customer order for a finished device does not automatically identify which wafer, package, or test resource can satisfy it.
The same wafer technology can feed products with different package and qualification requirements.
Conversely, a part number may be qualified at only one fabrication plant or assembly site.

The immediate buyer may be a distributor, an electronics manufacturer, or a contract manufacturer rather than the company whose product contains the chip.
The end customer may be an automotive manufacturer, a cloud provider, a phone maker, or an industrial-equipment producer.
Each sees different demand and places different contractual commitments.
An original equipment manufacturer (OEM) can report a production forecast while its contract manufacturer places the binding purchase order.
A distributor can hold stock while the chip supplier sees only the distributor's replenishment order.
These distinctions matter when demand weakens or a shortage encourages duplicate orders.

The practical decision is not simply how many chips to forecast.
It is how many wafers to start, where to reserve capacity, which products to prioritize, and when to communicate feasible delivery dates.
The planner also decides whether to hold unfinished inventory, buy capacity options, qualify an alternate, or accept the cost of a missed shipment.
A foundry, a fabless designer, an integrated device manufacturer, and an equipment supplier face related but distinct decisions.
The fabless company purchases manufacturing services and owns the product design; it generally does not own the wafer fab.
An integrated device manufacturer can own manufacturing assets as well as product designs, although it may also outsource some stages.
The foundry sells fabrication services; its own planning objective cannot be assumed to match that of any one customer.

A downstream shortage can idle a vehicle assembly line even when the missing device is inexpensive relative to the vehicle.
High replacement cost at the customer's assembly line does not prove that the chip supplier can legally or technically transfer a qualified part immediately.
Many chips require product-specific software, board qualification, reliability evidence, or manufacturing change approval.
A wafer already in process may be dedicated to another product and cannot be treated as fungible finished goods.
These constraints make a single aggregate inventory number a poor substitute for an allocation plan.

This primer uses explicitly synthetic examples to teach calculations.
It does not claim that the hypothetical figures describe a named company or current market conditions.
Public evidence is identified separately in section 9.
The regulatory discussion is educational, dated, and not a legal determination.
A credible planning recommendation shows its assumptions, identifies the decision owner, and states what evidence would change the recommendation.

## 2. Learning outcomes

After this topic, you can:

1. Map a semiconductor product from design through qualified customer delivery, identifying each buyer and capacity owner.
   Exercise 1 checks the map and the handoffs.
2. Calculate expected good units from wafer starts, die per wafer, and separate stage yields.
   Exercise 2 checks units, multiplication, and rounding.
3. Translate finished-unit demand into bottleneck capacity while recognizing the difference between expected yield and delivery assurance.
   Exercises 3 and 4 test feasibility and allocation.
4. Construct a time-phased capacity and inventory plan without mixing forecast, plan, target, and commitment.
   Exercise 5 asks for a decision table and a reconciliation.
5. Diagnose a chip bullwhip using aligned order and consumption series while identifying data limitations.
   Exercise 6 tests the calculation and interpretation.
6. Evaluate a proposed allocation using contract, qualification, financial, and customer-continuity criteria.
   Exercise 7 asks for an auditable recommendation.
7. Design a decision brief and monitoring plan that identifies uncertainty, standards, and escalation owners.
   Exercise 8 uses a scoring rubric.
8. Adapt the runnable lab in section 7 and explain why its simple model does not guarantee delivery.
   The worked example and lab form the calculation reference.

A correct result is not just a number.
It includes the time bucket, source, version, constraint, and person authorized to make the decision.
For a design exercise, a defensible alternative can score well if it states its assumptions and reconciles to the data.

## 3. The core mental model

### 3.1 Material, demand, and capacity flow

```text
End-use consumption / installed demand
              |
              v   forecast, inventory, actual consumption
OEM / system owner -- design approval --> qualified part and site list
              |                              |
              v                              v
EMS / distributor purchase order --> fabless or integrated supplier
                                          |
                             wafer reservation / start decision
                                          v
Equipment + materials --> wafer fab --> wafer sort --> assembly / package
                              ^                            |
                              |                            v
                       bottleneck hours                final test
                                                           |
                                              finished goods / delivery
                                                           v
                                                customer production
```

An electronics manufacturing services provider (EMS) builds products for a customer and may place orders on its behalf.
Signals flow upstream; constrained material flows downstream.
Capacity reservations and supplier commitments are negotiated in both directions.
Yield and lead time change the amount and timing of material at every handoff.
A downstream consumption signal may travel through several organizations before it reaches the fab.
That delay makes it easy to confuse a change in orders with a change in true end-use demand.

### 3.2 Three linked but distinct plans

The demand plan estimates what customers may need, by product, location, and date.
The supply plan chooses feasible starts and production routes subject to resource and material constraints.
The allocation plan assigns available qualified units to customers according to agreed priorities and rules.
All three plans must share a product hierarchy, calendar, and version, but they are not interchangeable.
An unconstrained forecast is not a delivery promise.
A capacity reservation is not necessarily a completed wafer, and a completed wafer is not necessarily a passing finished device.
A delivery commitment must reflect available-to-promise rules, qualifications, and contractual authority.

The fourth object is the financial plan.
It estimates margin, inventory exposure, expedite costs, and penalties, with assumptions made explicit.
High margin may be one allocation criterion, but contractual obligations and safety-critical customer continuity can override a simple margin ranking.
The decision owner, not an optimizer alone, approves the trade-off.

### 3.3 Delay, feedback, and amplification

Consider a stylized shortage.
A customer sees lead time rise and increases an order to protect production.
A distributor interprets the order as stronger consumption and increases its own replenishment request.
A chip supplier books more starts and capacity.
Long manufacturing lead times prevent a fast supply response.
When end-customer demand normalizes or duplicate orders are canceled, previously started wafers may still arrive.
Inventory then rises and the next order is reduced sharply.
The oscillation can propagate upstream, but this story is a causal hypothesis, not proof for any particular dataset.

Distinguish an actual change in consumer demand from an order-policy response.
Distinguish inventory placed for an approved strategic build from speculative duplicate orders.
Time aggregation, product mix, allocation rules, price changes, and promotional launches can also create apparent amplification.
Use comparable time series before claiming bullwhip.

### 3.4 Why average yield is not service level

A yield of 90% means an expected proportion under the specified measurement definition.
It does not mean every lot will achieve exactly 90%, and it does not imply a 90% probability of meeting demand.
Lot-to-lot dependence, common process excursions, bin mix, and a limited number of wafers change the distribution of output.
Expected output is useful for a baseline plan.
A customer commitment needs a distribution, a buffer policy, or a justified conservative scenario as well.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Fabless designer | Company that designs chips while contracting wafer fabrication. | Assuming it has direct control of all fab tools. |
| Foundry | Manufacturer that fabricates wafers for customers. | Treating every foundry process as interchangeable. |
| Integrated device manufacturer (IDM) | Company with both product and manufacturing activities. | Assuming all its manufacturing is internal. |
| OEM | Company responsible for the branded system or vehicle. | Treating an OEM forecast as an order from the legal buyer. |
| EMS | Contract manufacturer that assembles electronic systems. | Assuming the EMS owns end-customer demand. |
| Distributor | Intermediary that buys or handles devices for downstream customers. | Equating its orders to end consumption. |
| Wafer | Substrate on which many individual dies are manufactured; counted in wafers. | Calling it a finished chip. |
| Die | Individual semiconductor circuit cut from a wafer; counted in dies. | Assuming all gross dies pass test. |
| Gross die per wafer | Geometrically available dies per wafer for a particular design; dies/wafer. | Mixing it with already-good dies. |
| Wafer start | Wafer released into a defined fabrication flow; wafers/week or month. | Calling it a finished wafer shipment. |
| Wafer sort | Electrical testing performed on the wafer before singulation. | Ignoring its test program and bin rules. |
| Assembly | Packaging and interconnection of dies into devices. | Assuming any package line handles any device. |
| Final test | Post-assembly checks and binning of devices. | Treating a passing die as automatically saleable. |
| Yield | Accepted units divided by units entering the stated stage; dimensionless. | Multiplying an overall yield by its component yields again. |
| Bin | Classification of devices by test results or specifications. | Counting a lower-performance bin as the requested grade. |
| Cycle time | Elapsed time through an operation or flow, usually days or weeks. | Confusing processing time with queue plus processing time. |
| Lead time | Elapsed time from a specified trigger to specified receipt. | Mixing quoted, actual, and promised lead times. |
| Bottleneck | Resource that limits feasible throughput under the stated mix and period. | Assuming a resource remains the bottleneck for every product mix. |
| Tool-hour | One hour of usable time on an identified equipment group. | Equating calendar hours with effective productive hours. |
| Utilization | Workload divided by available capacity for a specified resource and period. | Treating 100% average utilization as delay-free. |
| Work in process (WIP) | Material started but not yet finished at the chosen boundary. | Counting WIP as immediately shippable stock. |
| Re-entrant flow | Repeated visits by a wafer to a process family at different manufacturing steps. | Counting only the first visit in capacity load. |
| Qualification | Evidence and approval that a part, process, or site meets stated requirements. | Assuming technical similarity equals customer approval. |
| Allocation | Explicit assignment of constrained supply to eligible demand. | Treating an allocation percentage as a shipment commitment. |
| ATP | Available to promise: supply available for a new commitment after applicable reservations. | Treating physical stock already committed elsewhere as free. |
| Forecast | Estimated future demand under stated assumptions. | Calling it a purchase obligation. |
| Plan | Chosen operating actions and resource assignments. | Treating it as a guaranteed outcome. |
| Target | Desired performance or volume. | Entering it into the order book as actual demand. |
| Commitment | An authorized promise under specified commercial terms. | Assuming an unaccepted request is a promise. |
| Bullwhip | Greater variability in an upstream order signal than in comparable downstream demand. | Declaring bullwhip from a single spike. |
| Sell-in | Shipments into a channel, such as from supplier to distributor. | Calling them consumption by the end user. |
| Sell-through | Channel sales to the next customer, as defined by the reporting agreement. | Assuming all sell-through is final use. |
| Non-cancelable, non-returnable (NCNR) | Contract term limiting cancellation or return, subject to actual agreement and applicable law. | Assuming every order is NCNR. |
| Process node | A technology-generation label used by a manufacturer. | Interpreting every label as the same physical dimension across firms. |
| Backlog | Accepted, unfilled orders at a specified date. | Treating all customer requests as firm backlog. |
| Allocation eligibility | Technical, contractual, and regulatory permission to assign a particular supply lot to an order. | Assuming a qualified alternate part is already approved. |
| Pegging | Mapping supply, WIP, or planned production to specific requirements. | Allocating the same supply to two orders. |

## 5. The mechanics, step by step

The procedure below starts with a business decision and ends with a reviewable commitment.
It is deliberately more detailed than a single optimization model because the input definitions and approvals determine whether the answer can be used.

### Step 1 — Define the decision and its clock

**Input:** a shortage, capacity-booking deadline, product launch, or inventory-risk signal.
**Output:** a decision statement naming the owner, action, horizon, and review date.
For example: decide this month how many wafer starts to reserve for each eligible product family over the next quarter.
Specify whether the scope is a fab, a foundry contract, an assembly site, or the end-to-end network.
Specify whether the decision can still be changed after release.
A production planner may choose start quantities, while sales approves customer-facing allocation and legal reviews contractual terms.
The final promise should have one accountable owner.

The horizon must cover the longest relevant commitment and supply delay, not merely the next accounting month.
Use weekly buckets when a one-week shift can idle an assembly line.
Use a monthly or quarterly view for longer-term capacity reservation, but do not hide weekly infeasibility inside a monthly average.
A useful output is a table with decision date, action, lead-time assumption, earliest output date, and owner.

**Branch:** if the shortage is already within the fixed lead-time window, adding new wafer starts may arrive too late.
Consider qualified finished stock, WIP already in the route, approved substitutions, customer rescheduling, or a negotiated transfer.
If no eligible supply can arrive in time, record the shortfall and escalate rather than creating a fictitious start.

### Step 2 — Map product and demand ownership

**Input:** product master, customer orders, bills of material, design ownership, and channel records.
**Output:** a versioned mapping from end product and customer to orderable chip, eligible route, and commercial buyer.
Record part number, revision, die design, package, grade, approved fab, approved assembly site, and approved test flow.
Record whether the customer can accept a substitute and which authority approves the change.
Separate the system owner from the contracting party and ship-to location.
A forecast submitted by an OEM and a purchase order submitted by its EMS may represent the same requirement.
Do not add them without a clear ownership and deduplication rule.

A design win is evidence of a potential future demand stream, not proof of a current order.
A purchase order is evidence of an order, not necessarily of final consumption.
A blanket capacity agreement may create obligations that are not visible in the normal order table.
Capture contract references and expiration dates without exposing restricted prices to unauthorized users.

**Branch:** if a qualified route is missing from the master, quarantine the record for review.
Do not make a product eligible merely because its package code looks similar.
If an alternate is approved only after a future date, model that date explicitly.

### Step 3 — Build a time-aligned demand picture

**Input:** accepted orders, forecast versions, sell-through, customer inventory, production schedules, and cancellations.
**Output:** an unconstrained demand series and an evidence-quality flag by part, customer, and week.
Keep order date, requested delivery date, promised delivery date, and actual ship date separate.
Use the same calendar and unit conversion when comparing upstream orders to downstream consumption.
Measure the number of pieces of the exact sellable device, not wafers, unless a documented conversion is applied.

Create separate columns for forecast, target, firm order, backlog, actual shipments, and actual consumption.
Record the observation date so that a historical forecast can be evaluated without later revisions leaking into the past.
Note whether cancellations are netted against new orders or represented as separate signed events.
Reconcile each customer's opening backlog plus accepted orders minus shipments and cancellations to closing backlog.
Flag discrepancies before using backlog as a capacity signal.

Check the demand owner: a distributor's sell-in and an EMS order may both be responses to the same OEM build plan.
Ask whether the reported customer inventory includes in-transit stock and consigned stock.
When customer inventory is unavailable, use a scenario rather than labeling channel shipments as consumption.
A sudden order increase with stable sell-through and rising channel stock warrants investigation; it does not prove duplicate ordering.

**Branch:** for a new product with little history, build a bottom-up scenario from launch volume, units per system, and qualified adoption.
Include uncertainty in the launch date and the share of systems using the device.
For a mature part, compare history, inventory, lifecycle notices, and customer build schedules.
Do not combine incompatible product revisions solely to increase sample size.

### Step 4 — Turn gross dies into saleable output

**Input:** wafer starts, gross dies per wafer, stage yields, bin rules, and routing.
**Output:** expected saleable units per start and a yield-risk description.
A simple multiplicative model is:

```text
expected saleable units = wafer starts × gross dies per wafer
                         × wafer-sort yield × assembly yield × final-test yield
```

Each yield in this expression is conditional on entering its own stage.
For example, final-test yield must use assembled units entering final test as its denominator.
If the available data instead gives an all-stage yield from gross die to final shipment, use that once rather than multiplying component yields again.

Sort yield can vary with design, process, wafer lot, and time.
Assembly yield can vary with package and subcontractor.
Final-test acceptance can vary with test limits and customer grade.
Model relevant bins separately when a customer requires a particular grade.
A passing lower grade is not automatically an eligible higher-grade unit.

For expected values, use a properly estimated conditional yield at each stage.
For supply assurance, model yield uncertainty and dependence at lot or process level where data supports it.
Do not assume independent unit failures when a single process excursion can affect an entire lot.
Compare forecasts with realized good units by lot and route, and investigate systematic bias.
A yield-improvement project is not free capacity until its effect is qualified, measured, and available inside the planning horizon.

**Branch:** if yield is unstable or a new design is ramping, use a low-yield scenario and specify its origin.
If sample size is small, show a range and seek engineering review rather than reporting a precise-looking point estimate.
If a yield measure already excludes engineering wafers or scrapped lots, record that selection before extrapolation.

### Step 5 — Translate starts into constrained load

**Input:** routing, process times, re-entrant visits, calendar, maintenance, and site capability.
**Output:** resource load and capacity by period, with the constraint named.
For each product and resource, sum required time across all visits to that resource group.
Multiply hours per wafer by planned wafers to obtain the capacity load.
The applicable capacity is usable productive time after the calendar and agreed availability assumptions, not nominal equipment ownership.
A maintenance schedule, tool qualification, engineering work, and the product mix can all reduce available time.
Avoid subtracting a loss twice if the supplied capacity already includes it.

A resource can be scarce because of a particular process capability even when another tool group is idle.
An alternate plant may have capacity but lack the necessary process or customer qualification.
Capacity purchased from a foundry may be limited by a contract, a reservation, a minimum volume, or a cancellation term.
Obtain the contract-specific details from the authorized owner rather than inferring them from a generic utilization figure.

Re-entrant manufacturing means a wafer can visit the same tool family several times.
An aggregate hours-per-start factor is acceptable only when its routing and product mix are consistent with the planning question.
A static monthly model can miss a weekly queue or a bottleneck that moves with mix.
For major decisions, add time-phased operations and WIP positions or use a validated discrete-event simulation.

**Branch:** if bottleneck demand exceeds capacity, compare feasible alternatives: defer, reallocate, buy qualified external capacity, improve yield, or reduce downstream demand.
Check each alternative's earliest feasible date and incremental economics.
Do not declare an unconstrained tool group to be available for a process it cannot perform.

### Step 6 — Time-phase WIP and finished supply

**Input:** opening inventories, WIP by stage, starts, expected yields, release schedules, and transport lead times.
**Output:** a weekly supply projection by eligible part and location.
For each bucket, opening stock plus eligible receipts minus assigned shipments equals closing stock.
For each WIP lot, use remaining route and expected completion date rather than the lead time of a new start.
Apply only yields for stages not already completed.
A lot that passed wafer sort should not incur wafer-sort yield again.
Separate quality hold, engineering lots, and blocked inventory from available supply.

Define the planning convention for arrivals and shipments inside a bucket.
If receipts are available at the end of the week, they cannot cover shipments due at the beginning of that week.
For a coarse model, explicitly state the within-bucket assumption and sensitivity to its reversal.
Include packaging substrates and test capacity if they can constrain delivery, not just wafer starts.
An end-to-end plan is feasible only when each required stage has both capacity and the right material.

Record age and date of inventory in the right unit: wafers, bare dies, or packaged parts.
Use a conversion rule only where engineering confirms that the material can become the requested device.
When quality status changes, update both the supply projection and all pegged orders.

**Branch:** if a lot is late, replan its downstream resource slots and notify affected customers.
If a lot is placed on hold, remove it from ATP until the relevant release authority acts.
If material is already allocated, a transfer requires explicit deallocation and a new approved assignment.

### Step 7 — Allocate qualified supply and document priorities

**Input:** eligible supply, firm orders, customer criticality, contract rules, and any agreed strategic reserves.
**Output:** allocation by customer, product, lot or bucket, and reason code.
First enforce hard eligibility: part, grade, site, date, quality status, geography, and approved use.
Then identify legally and commercially binding obligations with qualified counsel and commercial owners.
Only after those constraints are clear should you apply a discretionary policy.

Possible policies include proportional allocation, contractual priority, customer production continuity, agreed reservation rights, or a weighted score.
Each has trade-offs.
Proportional allocation can be simple but may idle all customers if none receives a minimum viable quantity.
A continuity rule can prevent a shutdown but requires credible customer inventory and alternative-source evidence.
A margin rule can improve short-term economics while harming contractual compliance or long-term relationships.
A weighted score must not hide a hard rule inside an unexplained coefficient.

Make the choice auditable by preserving the inputs, time of decision, approver, allocation version, and override reason.
Do not double-allocate the same finished stock, WIP, or capacity reservation.
Calculate customer shortfall as eligible demand minus allocated quantity, with the demand definition stated.
If the plan uses forecast rather than accepted orders, label its allocation provisional.

**Branch:** if multiple products share a bottleneck, compare the incremental contribution per constrained hour only after hard commitments and eligibility are respected.
If a customer requests an exception, assess the effect on every displaced customer rather than adding supply on paper.
If information is insufficient, set a review deadline and identify which owner must provide it.

### Step 8 — Model scenarios and select a plan

**Input:** baseline and alternative demand, yield, lead-time, and capacity assumptions.
**Output:** comparable scenarios and a recommendation with decision thresholds.
At minimum compare base yield, a lower-yield case, and a demand or timing shock.
Calculate expected units, bottleneck use, shortages, inventory, and changes in customer allocation.
If costs are used, state whether they are contribution, full cost, penalty, or cash flow.
Do not add the same contractual penalty to multiple customers without evidence.

A scenario is a coherent combination of assumptions, not an independently probable forecast for every variable.
If demand rises when capacity is most constrained, modeling those risks independently can understate exposure.
If probabilities are unavailable, present stress results without inventing probabilities.
A probability of service requires a specified stochastic model and validated input distributions.

Use value of information to decide which missing data to obtain.
For instance, improved distributor inventory visibility is valuable if it could change a large start decision before the order deadline.
It is less valuable to the immediate decision if capacity has already been irreversibly committed, though it may still improve future planning.

### Step 9 — Convert the plan into commitments

**Input:** approved allocation, valid ATP, customer order, contract, and shipment calendar.
**Output:** a dated customer communication and an internally pegged supply record.
Separate a planning estimate from a contractual promise in the message.
State the quantity, unit, part and revision, ship-from location, expected ship date, and assumptions where appropriate.
A customer request date is not a confirmed date until the authorized process accepts it.
Do not promise the same expected yield buffer to multiple customers.
If the company uses a confidence threshold for commitments, calculate it under the approved method rather than relabeling expected output as guaranteed.

Where an order cannot be filled, communicate the uncovered quantity, next review date, and approved mitigation.
Record any condition such as customer qualification of an alternate site.
Commercial terms and trade restrictions must be reviewed by the responsible teams before release.

### Step 10 — Monitor, learn, and correct

**Input:** actual starts, stage receipts, yields, tool availability, demand, cancellations, and shipments.
**Output:** exception decisions and updated model parameters with an audit trail.
Review leading indicators before waiting for missed shipments.
Examples include held lots, declining sort yield, rising queue time, late packaging material, and customer inventory falling faster than planned.
Track forecast error by horizon and demand owner, not just a single overall number.
Track allocation adherence and override reasons, including the impact on displaced orders.

Use a fixed cadence for the normal plan and an exception path for urgent disruptions.
Define a threshold, owner, and response for each important indicator.
A threshold should lead to an action such as engineering investigation or allocation review, not an automatic promise change without authority.
After each period, reconcile planned starts to actual starts and planned completions to actual good output.
Learn whether the difference arose from demand, yield, capacity, timing, or master-data error.

Archive the model version and inputs used for each material decision.
Otherwise a retrospective can mistakenly compare today's revised forecast with yesterday's decision.
Feedback is useful only when the decision context is reproducible.

## 6. Worked example

### 6.1 Decision and synthetic context

All numbers in this example are synthetic and do not represent any actual manufacturer.
A fabless chip supplier must choose a four-week wafer-start allocation for two already-qualified products.
The supplier buys a bottleneck capacity reservation from one foundry.
Its two buyers are an automotive electronics supplier purchasing product A and an industrial controller manufacturer purchasing product B.
The downstream vehicle OEM provides a build forecast but is not the direct contractual buyer in this example.
The four-week wafer-start window is the decision period, not the promised delivery period.

The supplier expects the resulting finished devices to become available in a later delivery window after fabrication, sort, package, and test.
For arithmetic simplicity, the example assumes all starts in the four-week window arrive in that later window.
Actual weekly dates and stage constraints must be checked separately before any customer-facing promise.
There is no eligible finished stock, prior WIP, packaging limit, or alternate fab in this synthetic case.
All starts are whole wafers.

The requested later-window demand is:

| Item | Product A | Product B |
|---|---:|---:|
| Customer forecast | 105,000 devices | 85,000 devices |
| Firm accepted orders within forecast | 90,000 devices | 70,000 devices |
| Sales target | 110,000 devices | 90,000 devices |
| New delivery commitment from this exercise | None yet | None yet |

The forecast is the planning input.
The accepted orders are commercial obligations requiring the contract owner to review timing and remedies; the example does not assume their contractual priority.
The target is not added to the forecast.
Neither the forecast nor the planned allocation automatically changes the accepted order or constitutes a new delivery commitment.

### 6.2 Technical conversion

| Parameter | Product A | Product B |
|---|---:|---:|
| Gross die per wafer | 1,000 dies | 800 dies |
| Wafer-sort yield | 90% | 95% |
| Assembly yield | 95% | 90% |
| Final-test yield | 95% | 95% |
| Bottleneck load | 2.0 tool-hours/wafer | 1.5 tool-hours/wafer |

All stage yields are conditional on the units entering that stage.
For A, expected good devices per wafer are:

```text
1,000 × 0.90 × 0.95 × 0.95 = 812.25 devices/wafer
```

For B, expected good devices per wafer are:

```text
800 × 0.95 × 0.90 × 0.95 = 649.80 devices/wafer
```

These are expected values, not guarantees for any particular wafer or lot.
Fractional expected devices can appear in an expectation even though an actual shipment contains whole devices.

### 6.3 Unconstrained requirements and bottleneck

Divide forecast demand by expected yield and round wafer starts up because starts must be whole wafers:

```text
A: 105,000 / 812.25 = 129.27..., so 130 wafers
B:  85,000 / 649.80 = 130.81..., so 131 wafers
```

The supplier has 400 productive bottleneck tool-hours for this four-week start window.
The forecast-covering starts would require:

```text
A: 130 wafers × 2.0 tool-hours/wafer = 260.0 tool-hours
B: 131 wafers × 1.5 tool-hours/wafer = 196.5 tool-hours
Total: 456.5 tool-hours
Shortage: 56.5 tool-hours
```

The forecast is therefore infeasible at the stated yield and capacity.
The 400 tool-hours are assumed to be already net of planned downtime.
Subtracting maintenance again would understate available time.

### 6.4 Proposed allocation

Commercial and operations owners select a provisional plan of 110 A wafers and 120 B wafers.
The selection is an illustrative policy decision, not the output of a claimed optimal algorithm.
It is made after an assumed review that both products are eligible for the foundry route and that no other contractual priority changes the choice.
It still requires a legal and commercial review of the accepted orders and their promised dates.

```text
A capacity = 110 × 2.0 = 220 tool-hours
B capacity = 120 × 1.5 = 180 tool-hours
Total      = 400 tool-hours
```

The expected finished output is:

```text
A = 110 × 812.25 = 89,347.5 expected devices
B = 120 × 649.80 = 77,976.0 expected devices
```

Against forecast, the expected shortfalls are:

```text
A = 105,000 − 89,347.5 = 15,652.5 expected devices
B =  85,000 − 77,976.0 =  7,024.0 expected devices
```

A is also below its 90,000-device accepted-order quantity in expected output by 652.5 devices.
B exceeds its 70,000-device accepted-order quantity in expected output by 7,976 devices.
This aggregate comparison does not establish whether either order is late; order dates and delivery timing are not supplied.
It also does not establish that the expected surplus is safe to promise to a different customer.
The decision owner should review whether accepted orders require a different mix, mitigation, or customer negotiation.

### 6.5 A feasible alternative and its trade-off

Suppose the owner moves 10 A wafers to B while holding total starts at 230 wafers.
The resulting 100 A and 130 B wafers use:

```text
100 × 2.0 + 130 × 1.5 = 395 tool-hours
```

This is feasible and leaves 5 tool-hours idle, but it is not a one-for-one exchange in bottleneck capacity.
It produces:

```text
A = 100 × 812.25 = 81,225 expected devices
B = 130 × 649.80 = 84,474 expected devices
```

The alternative reduces B's forecast shortfall from 7,024 to 526 expected devices.
It increases A's forecast shortfall from 15,652.5 to 23,775 expected devices.
It also increases A's expected gap to its accepted-order quantity to 8,775 devices.
Without contract terms, dates, and customer impact, neither the baseline nor this alternative can be declared commercially best.
To use all capacity, an additional feasible whole-wafer combination could be evaluated, subject to demand and route constraints.

### 6.6 Yield stress

Hold starts fixed at 110 A and 120 B wafers.
Assume A wafer-sort yield falls from 90% to 85% while all other yields remain unchanged.
This is a stress scenario, not a probability estimate.

```text
A stressed good devices/wafer = 1,000 × 0.85 × 0.95 × 0.95
                              = 767.125
A stressed expected output     = 110 × 767.125
                              = 84,383.75 devices
A stressed forecast gap        = 105,000 − 84,383.75
                              = 20,616.25 devices
```

The stress reduces A's expected output by 4,963.75 devices relative to the base case.
B remains at 77,976 expected devices under this deliberately isolated stress.
The supplier should not present this single scenario as the complete risk distribution.
Common-cause losses could also affect B, and lower yield can coincide with longer cycle time.

### 6.7 Decision brief

**Decision:** provisionally reserve 110 A and 120 B wafer starts, subject to commercial and engineering approval.
**Constraint:** 400 productive tool-hours in the four-week start window.
**Forecast:** 105,000 A and 85,000 B devices for the later delivery window.
**Plan:** expected output of 89,347.5 A and 77,976 B devices under baseline yield.
**Target:** 110,000 A and 90,000 B devices, shown for context only.
**Commitment:** no new customer promise is authorized by this calculation.
**Key exposure:** A's expected production is below its accepted-order volume, even before yield stress.
**Immediate action:** retrieve order-level dates and contract terms, confirm existing WIP and approved alternatives, and review the allocation.
**Trigger:** if observed A sort yield trends below the agreed engineering threshold, rerun the supply plan and escalate.
**Owner:** supply planning prepares the analysis; commercial, engineering, and the contract owner approve their respective decisions.
**Next review:** before the reservation becomes irreversible and again when lot-level yield information becomes available.

## 7. Code, query, or calculation

### 7.1 Lab purpose and boundaries

The following Python 3 program reproduces the base and stressed calculations.
It uses only the standard library and can be copied into a file named `semiconductor_lab.py`.
It models a single aggregate start window and a later common output window.
The quantities are synthetic expected values.
The code is a transparent arithmetic model, not a production allocation engine or service-level simulation.
It does not issue orders, promise dates, infer contractual priority, or optimize the allocation.

Assumptions:

- Product yields are stage-conditional and fixed at the stated planning values.
- All wafers use the same qualified route for their product.
- Capacity is measured in productive bottleneck tool-hours for the start window.
- All planned starts are whole wafers and all expected outputs arrive in one later window.
- There is no opening stock, other WIP, downstream bottleneck, or expiration.
- Forecasts are demand estimates; accepted orders and targets are separate fields.
- The stress changes only A's wafer-sort yield.
- A fractional expected output is not an executable fractional device shipment.

```python
from dataclasses import dataclass, replace
from math import ceil


@dataclass(frozen=True)
class Product:
    name: str
    gross_die_per_wafer: int
    sort_yield: float
    assembly_yield: float
    final_test_yield: float
    tool_hours_per_wafer: float
    forecast_devices: int
    accepted_order_devices: int
    target_devices: int

    def expected_good_per_wafer(self) -> float:
        return (
            self.gross_die_per_wafer
            * self.sort_yield
            * self.assembly_yield
            * self.final_test_yield
        )

    def forecast_starts(self) -> int:
        return ceil(self.forecast_devices / self.expected_good_per_wafer())


def report(product: Product, starts: int) -> None:
    good_per_wafer = product.expected_good_per_wafer()
    output = starts * good_per_wafer
    hours = starts * product.tool_hours_per_wafer
    print(
        f"{product.name}: starts={starts} wafers, "
        f"hours={hours:.1f} tool-hours, "
        f"expected={output:.2f} devices, "
        f"forecast_gap={max(0.0, product.forecast_devices - output):.2f} devices, "
        f"order_gap={max(0.0, product.accepted_order_devices - output):.2f} devices"
    )


def main() -> None:
    a = Product("A", 1000, 0.90, 0.95, 0.95, 2.0, 105000, 90000, 110000)
    b = Product("B", 800, 0.95, 0.90, 0.95, 1.5, 85000, 70000, 90000)
    capacity = 400.0
    plan = [(a, 110), (b, 120)]
    needed = [(p, p.forecast_starts()) for p in (a, b)]
    needed_hours = sum(p.tool_hours_per_wafer * n for p, n in needed)
    planned_hours = sum(p.tool_hours_per_wafer * n for p, n in plan)

    print("Forecast-covering wafer starts:")
    for product, starts in needed:
        print(f"{product.name}: {starts} wafers")
    print(f"Required: {needed_hours:.1f} tool-hours")
    print(f"Capacity: {capacity:.1f} tool-hours")
    print(f"Capacity gap: {max(0.0, needed_hours - capacity):.1f} tool-hours")
    print("Provisional plan:")
    for product, starts in plan:
        report(product, starts)
    print(f"Plan load: {planned_hours:.1f} tool-hours")
    print(f"Plan feasible: {planned_hours <= capacity}")
    print("A sort-yield stress:")
    report(replace(a, sort_yield=0.85), 110)


if __name__ == "__main__":
    main()
```

Run `python semiconductor_lab.py` in a Python 3 environment.
Expected output, with decimals rounded for display:

```text
Forecast-covering wafer starts:
A: 130 wafers
B: 131 wafers
Required: 456.5 tool-hours
Capacity: 400.0 tool-hours
Capacity gap: 56.5 tool-hours
Provisional plan:
A: starts=110 wafers, hours=220.0 tool-hours, expected=89347.50 devices, forecast_gap=15652.50 devices, order_gap=652.50 devices
B: starts=120 wafers, hours=180.0 tool-hours, expected=77976.00 devices, forecast_gap=7024.00 devices, order_gap=0.00 devices
Plan load: 400.0 tool-hours
Plan feasible: True
A sort-yield stress:
A: starts=110 wafers, hours=220.0 tool-hours, expected=84383.75 devices, forecast_gap=20616.25 devices, order_gap=5616.25 devices
```

### 7.2 Interpreting the output

The `order_gap` is a comparison of expected total output with accepted-order quantity.
It is not an assessment of contractual breach or a calculation of late units.
The program clips negative gaps to zero for readability, but that does not make the remaining expected quantity uncommitted inventory.
The target is recorded in the data structure but deliberately not added to forecast demand.
The program does not assert that capacity-feasible starts imply packaging or delivery feasibility.

To extend the lab, add weekly arrival buckets and a stock-balance table.
Represent each lot's stage and remaining yield only once.
Next add an eligibility matrix that blocks disallowed site, package, and customer combinations.
Then add order-level required and promised dates, with a documented within-week timing convention.
Before using any simulation, estimate a defensible lot-level yield distribution and dependence structure.
For a planning system, also add data-quality checks, authorization, audit logging, and an approved decision process.

## 8. Data, standards, and regulatory context

### 8.1 Minimum data contract

A reliable data contract identifies a record's owner, definition, time basis, permitted uses, and refresh cadence.
It must describe which organization can correct a record and how changes propagate to downstream plans.
Commercial confidentiality can restrict access even where technical access is possible.
Use minimum necessary fields and approved aggregation when sharing customer or process data.

| Dataset | Grain and essential fields | Decision use | Control |
|---|---|---|---|
| Product master | Part, revision, die, package, grade, lifecycle status. | Determine what is being planned. | Version and effective dates. |
| Qualification | Customer, part, manufacturing site, process, valid dates. | Restrict eligible supply. | Approval evidence and revocation. |
| Orders | Buyer, order line, quantity, requested and promised dates, status. | Measure accepted obligations. | Reconcile changes and cancellations. |
| Forecast | Owner, part, delivery bucket, issue date, quantity. | Estimate unconstrained demand. | Preserve historical vintages. |
| Channel inventory | Owner, location, part, date, quantity and stock status. | Interpret sell-in against sell-through. | Avoid double counting consigned stock. |
| WIP | Lot, stage, route, quantity, status, projected date. | Estimate future supply. | Separate hold and already-consumed yields. |
| Yield | Lot, operation, input quantity, accepted quantity, bin. | Convert starts to output. | Check denominators and exclusions. |
| Capacity | Tool group, site, bucket, available hours, routing load. | Identify constrained starts. | State downtime and re-entrant visits. |
| Delivery | Lot, part, quantity, ship date, destination, actual receipt. | Reconcile plan to execution. | Match shipment and order identifiers. |
| Contract | Parties, capacity rights, order terms, relevant effective dates. | Establish permitted allocation. | Restricted access and legal review. |

A single record can be technically valid and still misleading.
For example, a customer may submit the same demand to two distributors during a shortage.
Entity resolution can identify overlapping part, end customer, and dates, but only an authorized commercial investigation can establish whether the demand is duplicated.
Do not silently delete one order based solely on a statistical similarity.

Preserve source timestamps and the time zone or business calendar.
A forecast issued on Monday must not be evaluated using a later corrected demand file as though it had been available on Monday.
An inventory snapshot should specify whether in-transit and consigned goods are included.
For lot traceability, retain parent-child relationships across split, merge, assembly, and test operations where relevant and permitted.
A lot identifier without its lineage is insufficient to trace a device to all relevant manufacturing stages.

### 8.2 Industry standards and conventions

SEMI publishes standards for semiconductor manufacturing equipment and materials, including equipment communication standards.
For example, SEMI E10 addresses equipment reliability, availability, and maintainability terminology, while SEMI E30 defines a generic model for equipment communications and control.
Consult the applicable licensed edition and implementation agreements before defining a production interface.
The existence of a SEMI standard does not guarantee that different plants calculate usable planning capacity identically.
Document local definitions and the treatment of downtime.

JEDEC publishes semiconductor standards, including JESD47 for stress-test-driven qualification of integrated circuits.
The applicable revision, product class, customer contract, and other relevant standards determine what evidence is needed.
A qualification standard does not by itself constitute a customer's approval of every plant, process change, or end use.
Automotive customers may additionally require Automotive Electronics Council (AEC) component qualification documents such as AEC-Q100 for relevant integrated circuits.
An AEC qualification reference is not a substitute for an individual customer agreement or a complete quality system.

IPC publishes electronics assembly standards; their applicability depends on the product and contract.
The International Organization for Standardization (ISO) publishes ISO 9001 quality-management requirements.
IATF 16949 is an automotive quality-management system standard, used together with ISO 9001 as specified by its issuer and customer requirements.
These standards do not set wafer-start quantities or provide an allocation rule.
Certification status, product qualification, and customer approval are distinct facts.

Identifiers and business messages can follow agreed standards, but a standard-format order does not resolve duplicate demand or contractual ambiguity.
For an external interface, document the meaning of order type, requested date, ship date, unit of measure, and change event.
Version the mapping from customer part numbers to the supplier's die and package revisions.
When a standard defines optional fields, a trading-partner agreement should say which fields are mandatory for that exchange.

### 8.3 Trade controls and legal boundaries

As of November 2026, export-control applicability requires a transaction-specific and current review.
The U.S. Bureau of Industry and Security (BIS) administers the Export Administration Regulations (EAR), which can apply to specified items and transactions.
Classification, origin, destination, end user, end use, and applicable jurisdiction affect the analysis.
Rules and license requirements change; consult the current regulations and a qualified trade-compliance specialist.
A planning system must not treat a destination as eligible solely because inventory is physically available there.
Screening and license conditions belong in the release process, with access-controlled decision records.

The European Union's dual-use regulation provides a control framework for specified dual-use items and activities.
Member-state implementation, product classification, destination, and current amendments can matter.
This primer does not determine whether any example product is controlled or whether any transaction needs a license.
A semiconductor subsidy program, industrial policy, or capacity announcement is not a general authorization to export.
The synthetic example assumes that all proposed routes are compliant, but that assumption must be verified in a real transaction.

Use appropriate controls for customer data, manufacturing recipes, and design information.
Confidentiality agreements and applicable law may restrict sharing even within a supply-chain collaboration.
A data-sharing arrangement should define purpose, retention, access, aggregation, and incident response.
Avoid distributing detailed customer demand to a competing customer simply to justify an allocation.

### 8.4 Measurement checks

Check that the sum of lot-level accepted output cannot exceed the measured input without an explained rework or split/merge rule.
Check that yields are in the range from zero to one and use a consistent denominator.
Check whether reported yield includes only tested units or also accounts for scrapped and untested material.
Check that a capacity measure has not already deducted the same downtime represented in its calendar.
Check that a forecast vintage precedes the decision it is used to evaluate.
Check whether the named buyer, ship-to, and end user are the same entity.
Check whether a lot is actually released, qualified, and legally eligible for its proposed destination.
Exceptions require review; a data-quality flag is not an instruction to alter history.

## 9. Industry practice and case evidence

### 9.1 U.S. Department of Commerce semiconductor supply-chain request

The U.S. Department of Commerce published results from its semiconductor supply-chain request for information in January 2022.
The department reported low median inventory of key semiconductor products among responding consumers and described mismatches between supply and demand.
Its announcement also discusses constrained production and the difficulty of quickly increasing supply.
See the original department release in section 16 for the population, framing, and reported figures.
The request was not a randomized sample of every semiconductor product or buyer.
It should not be used as a current estimate of inventory for a particular part or firm.

**Planning implication, inference:** collecting customer inventory and consumption information can improve the distinction between a shortage-driven order and end-use demand.
The implication depends on participants reporting comparable definitions and on the planner having the authority to use the information.
The public release does not demonstrate that any specific proposed allocation rule is optimal.

### 9.2 Automotive semiconductor exposure

The U.S. Government Accountability Office (GAO) examined semiconductor shortages affecting the automotive industry in a 2022 report.
It describes factors associated with the shortage and the federal response.
The report is a public source for the sector's exposure; consult it for the exact scope and evidence rather than attributing a single cause to every automaker.
Automotive semiconductor demand passes through multiple tiers and includes many device types and manufacturing technologies.
A single vehicle-production decline cannot be assigned to a particular chip without product-specific evidence.

**Planning implication, inference:** an OEM, tier-one supplier, distributor, and chipmaker should reconcile forecasts and firm orders at an agreed part and time grain.
They should also document which alternatives have actually been approved.
This inference is a planning recommendation, not a GAO finding that the synthetic case's allocation is correct.

### 9.3 Bullwhip evidence and interpretation

Lee, Padmanabhan, and Whang's 1997 management-science literature describes the bullwhip effect and mechanisms including demand signal processing, order batching, price variation, and shortage gaming.
The result is a general supply-chain framework, not proof that each semiconductor order increase is gaming.
Real chip markets also experience product launches, capacity investment cycles, supply disruptions, and technology transitions.
The analyst must test plausible explanations with the relevant demand, inventory, and order records.

A basic diagnostic compares the coefficient of variation (CV) of upstream orders with the CV of downstream consumption over aligned periods.
For a nonnegative series with positive mean, CV is standard deviation divided by mean.
A ratio greater than one suggests relative order amplification in that window; it does not identify the cause.
Use the same dispersion convention, temporal aggregation, product scope, and currency or unit basis for both series.
If consumption is seasonal or trending, also examine comparable residuals or seasonal adjustments and document the method.

Suppose synthetic weekly end consumption is 90, 110, 90, 110 devices and upstream orders are 70, 130, 70, 130 devices.
Both series have mean 100 devices/week.
Using population standard deviation, consumption has standard deviation 10 devices/week and CV 0.10.
Orders have standard deviation 30 devices/week and CV 0.30.
The ratio is 3.0 for these four weeks.
This deliberately constructed example illustrates relative variability; it cannot establish a general semiconductor market effect.

Lag, aggregation, and sample length matter.
An order series could be shifted relative to consumption because of normal lead time, and a four-week sample is too short for robust conclusions.
If one series is shipments constrained by allocation rather than true demand, its apparent low variability may be an artifact.
A more useful study examines multiple comparable periods, forecast vintages, channel inventory, and cancellation history.
Show the data lineage and sensitivity to time bucket and lag before recommending a policy change.

### 9.4 Manufacturing capacity is not instantly interchangeable

A semiconductor fab requires specialized equipment, process integration, materials, and qualified products.
The U.S. Department of Commerce's 2022 supply-chain findings discuss the difficulty of quickly expanding production.
Specific expansion timelines and capability depend on the fab, equipment, process, and qualification requirements.
A new fab announcement is therefore not evidence of near-term supply for a particular part.
The analyst must identify the specific route, date, approved capacity, and customer qualification status.

**Planning implication, inference:** a planning team should maintain a product-to-route eligibility map and monitor capacity, WIP, and yield together.
Adding an unqualified route to an optimization model as immediately available can make a mathematically feasible plan operationally impossible.
The exact value of a second source requires costs, timing, yield, and customer approval evidence.

### 9.5 Evidence hierarchy for a decision

Use official regulatory text to establish applicable legal requirements, interpreted by qualified specialists.
Use an executed contract to determine the parties' actual commercial obligations, subject to counsel's interpretation.
Use observed lot and tool data to estimate plant-specific output and delays.
Use peer-reviewed research to identify mechanisms and methods, not to substitute for local measurements.
Use company statements as statements of that company's reported position, with the date and scope recorded.
Use a synthetic example only to illustrate arithmetic or a decision process.
Never present synthetic yield or capacity figures as an industry benchmark.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Forecast treated as order | Target and orders share one field. | Distinct demand signals are added. | Excess starts or false obligations. | Preserve signal type and contract owner. |
| Duplicate channel demand | Similar part and dates appear at multiple tiers. | Same OEM requirement is counted twice. | Bullwhip and excess stock. | Reconcile with authorized counterparties; do not delete automatically. |
| Gross die treated as good die | Output equals gross dies times starts. | Yield losses are omitted. | Unrealistic supply promise. | Apply stage-conditional yields with audited denominators. |
| Yield applied twice | WIP already passed a stage but model applies its yield. | Stage boundary is lost. | Understated supply. | Record current stage and apply only remaining yields. |
| Lot correlation ignored | Many wafers fail together. | Common process excursion affects multiple units. | Overstated service confidence. | Model lot or common-cause risk where data allows. |
| Tool capacity overstated | Plan uses installed calendar hours. | Maintenance and qualification are omitted. | Infeasible starts. | Use documented productive capacity. |
| Re-entrant load omitted | Routing lists only the first tool visit. | Repeated operations are not counted. | Hidden overload. | Sum all relevant visits or validate aggregate factors. |
| Route eligibility assumed | Alternate plant appears without approval. | Technical or customer qualification is missing. | Unusable output or prohibited shipment. | Enforce effective-dated qualification rules. |
| WIP counted as finished stock | All inventory appears in ATP. | Remaining lead time and losses are ignored. | Missed delivery. | Track stage, status, and remaining yield. |
| Weekly timing hidden | Monthly supply meets monthly demand but early orders slip. | Receipts arrive after required shipments. | Stockout despite aggregate balance. | Model finer buckets and receipt convention. |
| Allocation exceeds supply | Total customer allocations exceed released stock. | Same lot is pegged multiple times. | Conflicting promises. | Atomic reservations and reconciliation. |
| Contract priority overlooked | Planner has no access to order terms. | A soft score displaces a binding obligation. | Commercial or legal exposure. | Contract-owner review before discretionary ranking. |
| Channel stock confused with consumption | Sell-in rises while sell-through is flat. | Inventory build is interpreted as end demand. | Unnecessary starts. | Compare aligned sell-in, sell-through, and inventory. |
| Export eligibility inferred from geography | Shipment planned without classification. | Current transaction-specific controls ignored. | Prohibited or delayed shipment. | Trade-compliance release and current review. |
| Improvement assumed before validation | Yield benefit appears only in a project slide. | Unproven change is booked as capacity. | Output miss. | Use measured, qualified effective dates. |
| Scenario called a probability | One low-yield case is labeled 95% service. | No probability distribution is modeled. | False confidence. | State stress assumptions; estimate distributions separately. |
| Backlog not reconciled | Closing backlog differs from order movements. | Cancellations or duplicates are missing. | Inflated demand. | Reconcile opening plus orders minus shipments and cancellations. |
| Metrics hide customer harm | Aggregate fill rate rises but one plant has no minimum quantity. | Average suppresses distribution of shortage. | Customer shutdown. | Review customer- and date-level consequences. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or relax it |
|---|---|---|---|
| Stage yields are conditional. | Multiplication requires compatible denominators. | Overall yield is multiplied by component yields. | Audit stage input and accepted counts. |
| Base yield represents the future. | Converts starts to expected output. | New design, process excursion, or mix shift. | Backtest by product, lot, and time; stress the estimate. |
| A fixed tool-hour factor represents each product. | Converts starts to bottleneck load. | Routing, visit count, or equipment mix changes. | Compare with current routing and measured tool history. |
| Productive capacity is known. | Defines the feasible start limit. | Unplanned downtime or missing maintenance. | Compare planned versus realized availability. |
| The delivery window is common. | Makes the example calculable without weekly scheduling. | Lots complete at different dates. | Use lot-level remaining cycle-time distributions. |
| Other stages have enough capacity. | Isolates the wafer bottleneck. | Package substrate or test shortage. | Build a stage-by-stage capacity balance. |
| All planned routes are qualified. | Makes starts potentially usable for the customers. | Customer-specific site or revision restrictions. | Review effective-dated qualification evidence. |
| No opening inventory or WIP exists. | Avoids double counting in the example. | Existing material is available or held. | Add lot-level opening balances and status. |
| Forecast quantities represent distinct demand. | Avoids duplicated production. | EMS, distributor, and OEM report the same need. | Reconcile demand ownership and channel inventory. |
| Unit demand is a useful planning target. | Makes shortages comparable in pieces. | Different customer criticality or product value. | Add qualified business impact and contract rules. |
| A point estimate is appropriate for baseline arithmetic. | Enables an interpretable first calculation. | Volatile, correlated yields or uncertain lead times. | Add validated scenarios or a stochastic model. |
| Constraint remains the same. | Allows one-resource analysis. | Product mix or calendar moves the bottleneck. | Recalculate all resource loads each period. |
| A proportional or selected allocation is permissible. | Provides an illustrative decision. | Contracts or regulations impose other priorities. | Obtain commercial and legal review. |
| Exact output can be treated as an expectation. | Simplifies expected-value arithmetic. | Executable shipment requires whole devices and lot constraints. | Use integer quantities and lot-level simulation for commitments. |
| Customer order status is accurate. | Supports the order-gap comparison. | Cancellation or revision is missing. | Reconcile order-event history with contract owner. |
| No model output is a new commitment. | Protects the distinction between plan and promise. | A recipient treats the analysis as an offer. | Label the document and use authorized ATP workflow. |

The model's simplicity is intentional for instruction.
It should be replaced or extended when the decision depends on weekly operations, correlated failures, or multiple constrained stages.
Additional complexity is useful only if its parameters and decision ownership can be supported.
A highly detailed model with stale qualification data may be less reliable than a transparent constrained calculation with current inputs.

## 12. Exercises

### Exercise 1 — Map the decision network

**Time:** 15 minutes.
**Type:** recall and application.
**Scenario:** A vehicle OEM forecasts 40,000 control units for a quarter.
Its tier-one supplier buys a microcontroller through a distributor; the designer contracts a foundry and an outsourced assembly and test provider.
The OEM has approved only one package and one manufacturing route.

**Deliverable:** draw the material flow and information flow, labeling each buyer, supplier, and approval owner.
**Acceptance criteria:** distinguish OEM forecast from the tier-one order; identify at least three manufacturing stages; explain why an alternative route needs approval.
State one missing fact that would affect the plan.

### Exercise 2 — Calculate expected output

**Time:** 12 minutes.
**Type:** calculation.
A product has 1,200 gross dies per wafer, 92% wafer-sort yield, 96% assembly yield, and 95% final-test yield.
A plan starts 75 wafers.
Each yield is conditional on the previous stage.

**Deliverable:** calculate expected good devices per wafer and total expected devices.
**Acceptance criteria:** show the multiplication, preserve units, and explain why the result does not guarantee a shipment quantity.
Round only the displayed total to two decimal places.

### Exercise 3 — Test a bottleneck plan

**Time:** 12 minutes.
**Type:** application.
Product X requires 1.8 tool-hours per wafer and product Y requires 2.5 tool-hours per wafer.
A proposed four-week plan starts 80 X wafers and 90 Y wafers.
Available productive capacity is 350 tool-hours.

**Deliverable:** calculate total load, feasibility, and the minimum reduction in Y wafer starts if X is held fixed.
**Acceptance criteria:** use whole wafers and show remaining unused hours.
Assume no other constraint and no demand limit for this exercise only.

### Exercise 4 — Compare allocation choices

**Time:** 15 minutes.
**Type:** critique and calculation.
Two customers each request 10,000 devices.
Only 12,000 eligible devices are available before their requested date.
Customer A needs at least 8,000 devices to maintain a production line; customer B can operate at a lower rate with 4,000 devices.
No contractual priority is provided.

**Deliverable:** compare proportional allocation with an 8,000/4,000 split.
**Acceptance criteria:** calculate each customer's shortfall and explain the line-continuity trade-off.
Identify at least two missing facts before approving the split.
Do not assume that a line-continuity claim overrides contract terms.

### Exercise 5 — Reconcile a weekly inventory plan

**Time:** 15 minutes.
**Type:** application.
Opening released stock is 4,000 devices.
Receipts of 3,000 and 2,000 devices arrive at the start of weeks 1 and 2 respectively.
Planned shipments are 5,000 and 3,500 devices in weeks 1 and 2 respectively.
No other receipts or shipments occur.

**Deliverable:** calculate opening stock, receipts, shipments, and closing stock for each week.
**Acceptance criteria:** show the balance equation and identify whether the two-week plan is feasible under the stated receipt convention.
Explain what changes if the week-1 receipt arrives after the week-1 shipment.

### Exercise 6 — Diagnose variability

**Time:** 15 minutes.
**Type:** calculation and critique.
Weekly downstream consumption is 80, 120, 80, 120 devices.
Weekly upstream orders are 60, 140, 60, 140 devices.
Use population standard deviations and identical four-week windows.

**Deliverable:** calculate both means, standard deviations, CVs, and the ratio of order CV to consumption CV.
**Acceptance criteria:** show the calculation and give two reasons why the result does not prove shortage gaming.

### Exercise 7 — Critique the worked-case decision

**Time:** 20 minutes.
**Type:** critique.
An executive proposes promising the base-case expected output of 89,347.5 A devices and 77,976 B devices immediately.
They also propose using the sales target as an additional demand line.

**Deliverable:** write a brief response listing at least four specific defects and two actions before any commitment.
**Acceptance criteria:** address yield uncertainty, fractional expectations, dates and contractual authority, and double counting of target versus forecast.
Identify one missing downstream constraint.

### Exercise 8 — Build a sector decision brief

**Time:** 35 minutes.
**Type:** design.
Use the worked case or an appropriately anonymized real planning decision.
Design a one-page decision brief for the owner of a capacity reservation.

**Deliverable:** state the decision, time horizon, buyer and end user, current forecast, plan, target, commitment, constraint, and at least two scenarios.
Include an allocation rule, one alternative, a risk trigger, and owners for action and approval.
**Acceptance criteria:** quantities reconcile to capacity; eligibility and contractual constraints precede the discretionary allocation; assumptions and sources are labeled.
Include a time-phased next step and identify one data item that could change the recommendation.
Use the rubric in section 13.

## 13. Self-check answers

### Answer 1 — Network map

A suitable chain is foundry to wafer sort to assembly and final test to distributor to tier-one supplier to vehicle production.
The designer purchases foundry services and arranges assembly and test in the scenario.
The tier-one supplier buys through the distributor; the OEM provides a forecast and approves the relevant device and route.
The exact ownership of wafer sort or finished inventory is not specified and should be marked unknown.
An alternate route may differ in process, site, or reliability and needs the applicable customer and engineering approvals.
A missing fact could be the tier-one supplier's actual accepted order, existing inventory, or requested delivery schedule.

### Answer 2 — Expected yield

```text
Good devices/wafer = 1,200 × 0.92 × 0.96 × 0.95
                   = 1,006.848 devices/wafer
Expected output    = 75 × 1,006.848
                   = 75,513.60 devices
```

This is an expected value under the stated fixed yields.
Actual output is an integer and can be lower or higher, especially when lot-level outcomes are correlated.
A promise needs appropriate supply assurance and eligibility checks.

### Answer 3 — Capacity

```text
X load = 80 × 1.8 = 144 tool-hours
Y load = 90 × 2.5 = 225 tool-hours
Total  = 369 tool-hours
Gap    = 369 − 350 = 19 tool-hours
```

Reducing Y by seven wafers saves only 17.5 tool-hours and remains infeasible.
Reducing Y by eight wafers saves 20 tool-hours.
The revised plan uses 349 tool-hours and leaves one tool-hour unused.
This answer assumes that removing eight Y starts is commercially and technically permitted.

### Answer 4 — Allocation

Proportional allocation gives each customer 6,000 devices and a 4,000-device shortfall.
Under the scenario, that leaves A below its stated 8,000-device minimum while B can operate at a lower rate.
The 8,000/4,000 allocation gives A a 2,000-device shortfall and B a 6,000-device shortfall.
It meets A's claimed minimum and B's stated lower-rate quantity, but its fairness and legality are not established.
Before approval, obtain the contracts and promised dates, validate both customers' inventory and minimum operating quantities, and check whether any alternatives exist.

### Answer 5 — Stock balance

```text
Week 1: 4,000 opening + 3,000 receipts − 5,000 shipments = 2,000 closing
Week 2: 2,000 opening + 2,000 receipts − 3,500 shipments =   500 closing
```

Under the stated start-of-week receipt assumption, stock remains nonnegative and the aggregate plan is feasible.
If the week-1 receipt arrives after the shipment, only 4,000 devices are available when 5,000 are needed.
The plan has a 1,000-device timing shortfall even though the end-of-week arithmetic still produces 2,000 devices.
Exact shipment timing and any backorder policy must be specified.

### Answer 6 — Variability

Both means are 100 devices/week.
For consumption, deviations are −20, 20, −20, and 20; the population variance is 400 and standard deviation is 20 devices/week.
For orders, deviations are −40, 40, −40, and 40; the population variance is 1,600 and standard deviation is 40 devices/week.
Thus consumption CV is 0.20, order CV is 0.40, and the ratio is 2.0.
This is evidence of relative variability in the constructed window, not proof of its cause.
Order batching, lead-time timing, promotions, inventory policy, and the very short sample are alternative explanations or limitations.
The consumption data might also be constrained shipments rather than true demand.

### Answer 7 — Commitment critique

Expected output is not a service guarantee; lot-level yield can be lower.
A fractional expected device is not an executable shipment quantity.
The analysis contains no order-level dates, no approved ATP, and no authorization for a new promise.
It omits potential assembly, package-material, test, and logistics constraints.
Adding a target to the forecast can double count demand and confuses desired performance with a demand signal.
Before a promise, confirm qualified available supply and stage timing, assess yield risk, and obtain commercial and contract-owner approval.
Review existing order terms and communicate any uncovered quantities through the authorized process.

### Answer 8 — Design rubric

Score the brief on five dimensions, from 0 to 2 points each:

| Dimension | 0 points | 1 point | 2 points |
|---|---|---|---|
| Decision and ownership | Unclear decision. | Decision or owner present. | Decision, owner, deadline, and horizon clear. |
| Reconciliation | Capacity or quantities conflict. | Arithmetic mostly clear. | Units, capacity, timing, and gaps reconcile. |
| Eligibility and obligations | Omitted. | Mentioned but not applied. | Hard route and contract checks precede allocation. |
| Uncertainty | One unsupported point estimate. | One stress with limited interpretation. | At least two coherent scenarios and decision triggers. |
| Communication | Forecast and commitment conflated. | Some distinctions or actions present. | Forecast, plan, target, commitment, owners, and next review explicit. |

A strong brief scores 8–10 points without a zero in eligibility or reconciliation.
A different allocation can be strong if its assumptions, consequences, and approval path are explicit.
Do not award full credit for a polished recommendation whose numbers exceed capacity or whose route is unqualified.

## 14. Interview questions

1. **How would you distinguish the buyer from the end customer in a chip shortage?**
   A strong answer maps the OEM, tier-one or EMS buyer, distributor, chip supplier, and contract holder.
   A weak answer assumes the organization forecasting demand necessarily placed the purchase order.
2. **How do you convert wafer starts into expected finished devices?**
   A strong answer multiplies gross die by correctly defined stage-conditional yields and identifies bin restrictions.
   A weak answer treats gross dies as shipped devices or applies an overall yield twice.
3. **Why is expected yield insufficient for a customer promise?**
   A strong answer discusses variability, lot correlation, timing, released stock, and commitment authority.
   A weak answer calls a 90% yield a 90% service probability.
4. **What makes a bottleneck capacity calculation credible?**
   A strong answer includes re-entrant visits, effective tool availability, routing, product mix, and time buckets.
   A weak answer multiplies installed tools by calendar hours and ignores qualification.
5. **How would you test for bullwhip in the chip channel?**
   A strong answer aligns order and consumption definitions, computes relative variability, and checks inventory, lag, seasonality, and alternative explanations.
   A weak answer labels one large order as manipulation.
6. **How would you allocate supply across customers in a shortage?**
   A strong answer checks eligibility and obligations first, compares transparent policies, and records impacts and approvals.
   A weak answer maximizes margin without checking contracts or feasibility.
7. **Why might an alternate foundry fail to solve a near-term shortage?**
   A strong answer cites process capability, route qualification, lead time, available capacity, and customer approval.
   A weak answer assumes a wafer is interchangeable across every site.
8. **What is the difference between forecast, plan, target, and commitment?**
   A strong answer gives each a distinct decision role and owner.
   A weak answer adds them all to create a single demand number.
9. **How would you use WIP in an ATP calculation?**
   A strong answer uses lot status, remaining stage yield, timing, eligibility, and existing pegs.
   A weak answer treats every in-process wafer as finished available stock.
10. **What would you do if a customer asks you to reallocate an already-promised lot?**
    A strong answer identifies displaced orders, contractual authority, quality and trade checks, decision ownership, and customer communication.
    A weak answer promises the lot twice or changes priority without approval.

## 15. Key terms

- **Allocation:** assignment of eligible scarce supply; see [mechanics](#step-7--allocate-qualified-supply-and-document-priorities).
- **Assembly:** packaging and interconnection after fabrication; see [definitions](#4-definitions-and-vocabulary).
- **ATP:** available to promise after applicable reservations; see [commitments](#step-9--convert-the-plan-into-commitments).
- **Backlog:** accepted unfilled orders; see [demand picture](#step-3--build-a-time-aligned-demand-picture).
- **Bin:** classification of tested devices; see [yield mechanics](#step-4--turn-gross-dies-into-saleable-output).
- **Bottleneck:** a resource limiting throughput for a stated mix and period; see [capacity](#step-5--translate-starts-into-constrained-load).
- **Bullwhip:** relative amplification of upstream order variability; see [evidence](#93-bullwhip-evidence-and-interpretation).
- **Capacity reservation:** a commercial claim on specified capacity, not finished output; see [decision clock](#step-1--define-the-decision-and-its-clock).
- **Commitment:** authorized promise under specified terms; see [decision brief](#67-decision-brief).
- **Cycle time:** elapsed time through a defined manufacturing flow; see [definitions](#4-definitions-and-vocabulary).
- **Die:** individual circuit on a wafer; see [technical conversion](#62-technical-conversion).
- **EMS:** contract electronics manufacturer; see [mental model](#31-material-demand-and-capacity-flow).
- **Fabless designer:** chip designer contracting wafer production; see [definitions](#4-definitions-and-vocabulary).
- **Final test:** post-assembly device assessment; see [yield mechanics](#step-4--turn-gross-dies-into-saleable-output).
- **Forecast:** estimate of future demand; see [demand picture](#step-3--build-a-time-aligned-demand-picture).
- **Foundry:** producer of wafers for customers; see [mental model](#31-material-demand-and-capacity-flow).
- **Gross die per wafer:** available die positions before yield losses; see [technical conversion](#62-technical-conversion).
- **Lead time:** elapsed time between specified order and receipt events; see [decision clock](#step-1--define-the-decision-and-its-clock).
- **Lot:** traceable group of material with shared processing history; see [data contract](#81-minimum-data-contract).
- **OEM:** original equipment manufacturer responsible for the end system; see [network](#31-material-demand-and-capacity-flow).
- **Pegging:** linking a supply quantity to a requirement; see [allocation](#step-7--allocate-qualified-supply-and-document-priorities).
- **Plan:** chosen operating actions; see [decision brief](#67-decision-brief).
- **Qualification:** required evidence and approval for a part or route; see [standards](#82-industry-standards-and-conventions).
- **Re-entrant flow:** repeated visits to a tool family; see [capacity](#step-5--translate-starts-into-constrained-load).
- **Sell-in:** shipment into a channel; see [demand picture](#step-3--build-a-time-aligned-demand-picture).
- **Sell-through:** channel sale to the next customer; see [bullwhip evidence](#93-bullwhip-evidence-and-interpretation).
- **Target:** desired performance, not additional demand; see [decision brief](#67-decision-brief).
- **Tool-hour:** productive time on a specified resource group; see [bottleneck calculation](#63-unconstrained-requirements-and-bottleneck).
- **Wafer start:** release of a wafer into fabrication; see [worked example](#61-decision-and-synthetic-context).
- **Wafer sort:** electrical test before singulation; see [technical conversion](#62-technical-conversion).
- **WIP:** work in process, not automatically shippable stock; see [time phasing](#step-6--time-phase-wip-and-finished-supply).
- **Yield:** accepted fraction of the specified stage input; see [yield mechanics](#step-4--turn-gross-dies-into-saleable-output).

## 16. Sources

Sources are publicly identifiable; some standards require purchase or registration for the full text.
Access the current edition and transaction-specific legal texts when applying the material.

1. U.S. Department of Commerce, [Results from Semiconductor Supply Chain Request for Information](https://www.commerce.gov/news/blog/2022/01/results-semiconductor-supply-chain-request-information), January 25, 2022.
   Official summary of the request, reported inventory, and supply constraints; its responding population is not the whole market.
2. U.S. Government Accountability Office, [Semiconductors: Supply Chain Disruptions Impacted the Automotive Industry](https://www.gao.gov/products/gao-22-105162), GAO-22-105162, 2022.
   Official review of automotive semiconductor disruptions and the federal response.
3. Hau L. Lee, V. Padmanabhan, and Seungjin Whang, “The Bullwhip Effect in Supply Chains,” *MIT Sloan Management Review*, 1997, [article](https://sloanreview.mit.edu/article/the-bullwhip-effect-in-supply-chains/).
   General supply-chain mechanisms; not a chip-specific causal finding.
4. SEMI, [Standards](https://www.semi.org/en/standards).
   Official entry point for semiconductor equipment and materials standards, including current edition information.
5. JEDEC, [Standards and Documents](https://www.jedec.org/standards-documents).
   Official catalog for JESD47 and other semiconductor standards; applicability depends on the relevant edition and agreement.
6. Automotive Electronics Council, [Component Technical Committee documents](http://www.aecouncil.com/AECDocuments.html).
   Official access point for AEC-Q100 and related automotive component qualification documents.
7. ISO, [ISO 9001 quality management systems](https://www.iso.org/standard/62085.html).
   Official description of the 2015 edition; check the current edition for an actual assessment.
8. International Automotive Task Force, [IATF 16949](https://www.iatfglobaloversight.org/iatf-169492016/).
   Official information on the automotive quality-management system standard.
9. U.S. Bureau of Industry and Security, [Export Administration Regulations](https://www.bis.gov/regulations/ear).
   Official entry point for current U.S. export-control rules; classification and applicability require specialist review.
10. European Union, [Regulation (EU) 2021/821 establishing a Union regime for the control of exports, brokering, technical assistance, transit and transfer of dual-use items](https://eur-lex.europa.eu/eli/reg/2021/821/oj).
    Official regulation as published; consult the consolidated text and applicable amendments for a current decision.
11. IPC, [Standards](https://www.ipc.org/standards).
    Official catalog of electronics-industry standards; a contract and product scope determine applicability.

## 17. Links

- [Module 29 topic index](_index.md).
- [Module 29 survey](../../../course/29_industry_electives.md).
- [CPG and retail sibling](01_cpg-retail.md).
- [Automotive sibling](03_automotive.md).
- [Simulation and optimization production labs index](../30_simulation_optimization_production_labs/_index.md).

Sibling topics listed as planned in the module index may not yet be available in a particular checkout.
Use the simulation and optimization lab index to locate an available exercise; the synthetic Python lab in section 7 is self-contained.
