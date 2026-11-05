# BOPIS and Curbside: The Store as a Pickup Operation

> Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise  
> Topic 05 of 10  
> Estimated study time: 4 hours  
> Prerequisites: inventory visibility and available-to-promise  
> Version 2026.5

## 1. Why this topic exists

Buy online, pick up in store (BOPIS) converts a store into a small fulfillment operation.
A customer buys remotely, the store finds and stages the goods, and the customer collects them.
Curbside changes the final handoff: the customer remains outside and an associate brings the order out.
The promise therefore depends on both digital inventory and physical store capacity.

A store manager may see sales increase while associates see queues grow.
An e-commerce manager may see fast average pickup times while a minority of customers wait much longer.
A merchandising manager may see substitution rates rise without knowing whether customers accepted the replacements.
A finance partner may see attractive sales but no incremental contribution after labor and cannibalization.
Each observation addresses a different decision.
You need one operating model that connects them without pretending that one metric explains the whole experience.

The costly mistake is treating an accepted order as a fulfilled order.
An order can be accepted when the item is not actually on the shelf.
It can be picked but staged in the wrong place.
It can be ready but handed to the wrong person.
It can be handed over on time with an unwanted substitute.
It can satisfy the customer and still consume more contribution than it creates.

The store owns the real work but usually does not control every demand driver.
Marketing sets promotions, digital teams set checkout promises, and inventory systems report availability.
Operations must translate those inputs into a feasible promise and staffing plan.
An explicit escalation rule is more useful than expecting a picker to solve a system-wide shortage alone.

The choice is not simply whether to offer pickup.
You must decide which products and stores qualify, how much capacity to expose, how substitutions are authorized, and when to stop taking orders.
You must also decide which customers receive priority when a queue forms.
A defensible rule protects safety and accessibility without hiding service differences by channel.

This unit teaches the operating and analytical choices behind those decisions.
The numerical case is synthetic and illustrates a method, not a claim about any retailer.
All currency in that case is US dollars, and the period is one operating day unless stated otherwise.

## 2. Learning outcomes

After completing this unit, you can:

1. **Map** an order from customer promise to handoff, identify owners, and distinguish physical from system states.
   Exercise 1 tests the map and the evidence required for each transition.
2. **Compute** line-level pick accuracy, order completeness, substitution acceptance, and pickup wait with explicit denominators.
   Exercises 2 and 3 test the calculation and interpretation.
3. **Choose** a substitution and exception policy that protects customer consent and product safety.
   Exercise 4 tests policy design and an escalation decision.
4. **Estimate** workload, productive labor, paid labor, and hourly capacity under stated assumptions.
   Exercises 5 and 6 test the labor model and its capacity implications.
5. **Assess** a contribution case using incremental demand, cannibalization, and avoidable costs.
   Exercise 7 tests the decision rather than gross sales alone.
6. **Diagnose** data gaps, queue-tail behavior, and misleading comparisons between stores.
   Exercise 8 tests the measurement plan.
7. **Design** a controlled operating change with guardrails and ownership.
   Exercise 9 tests the pilot and evaluation design.

The observable outcome is a decision a store team can execute and audit.
It is not merely a dashboard of attractive averages.

## 3. The core mental model

```text
Customer sees promise + substitution choices
                  |
                  v
Order accepted --> inventory and capacity reserved
                  |
                  v
Release --> find and verify items --> resolve missing items
                  |                         |
                  |                         +--> consent / refund / cancel
                  v
Stage with location and temperature controls
                  |
                  v
Ready notification --> customer arrives / checks in
                  |
                  v
Queue --> retrieve --> verify recipient --> handoff / receipt
                  |
                  v
Feedback, reconciliation, inventory correction, improvement
```

The flow has two interdependent clocks.
The **preparation clock** starts at acceptance and ends when the order is truly ready.
The **handoff clock** starts at arrival or valid check-in and ends when the customer receives the order.
A ready-order target cannot substitute for a curbside wait target.
Both clocks must use recorded events and a declared treatment for missing timestamps.

There are also two constrained resources.
Inventory must exist, be sellable, and remain available long enough for a picker to find it.
Labor and staging capacity must exist in the relevant time window, not merely somewhere during the day.
Capacity shown at checkout should be the lower of these feasible limits after operational buffers.
The specific allocation algorithm belongs to the order-routing and available-to-promise topics.

A promise is a commitment shown to a customer, not the same thing as a forecast or a staffing plan.
The forecast predicts demand; the plan assigns resources; the target states an internal ambition.
The commitment is what the individual customer was told at a particular time.
Keep the original commitment and any subsequent revisions so that a later revision cannot erase a miss.

Errors propagate through the chain.
A phantom unit causes a search, which consumes labor and delays subsequent orders.
A late order reaches staging after the customer has already arrived, creating a mixed queue of ready and unready orders.
An unapproved replacement may look complete operationally but generate a complaint or return.
Conversely, a cautious substitution policy may reduce apparent completeness while preserving customer trust.

The feedback loop must return information to the right owner.
A picker reports a missing shelf unit, the inventory owner investigates the record, and digital availability is updated under an authorized process.
A handoff delay caused by too few associates is a staffing and capacity decision, not an inventory correction.
Repeated contact failures are a customer-communication design problem.
A monthly average alone cannot tell these mechanisms apart.

### Decisions and accountability

| Decision | Accountable owner | Inputs | Escalation or control |
|---|---|---|---|
| Offer a pickup window | Digital promise owner | Eligible inventory, capacity, cutoff | Store operations can pause an unsafe or infeasible window. |
| Allocate associates by interval | Store manager | Forecast arrivals, work standards, other duties | Regional operations resolves a persistent capacity deficit. |
| Approve a substitute | Customer under published policy | Alternatives, price, allergy or other restrictions | Picker refunds or escalates if authorization is absent. |
| Correct an inventory record | Inventory control | Search evidence, counts, transaction history | Separate count and adjustment authority where feasible. |
| Release an order | Trained handoff associate | Ready state, identity or pickup credential, physical goods | Supervisor resolves mismatch or suspected fraud. |
| Fund expansion | Finance and channel leadership | Incremental contribution, service and risk | Pilot stops if pre-agreed guardrails fail. |

Roles are illustrative; assign named roles for the actual organization.
A single person may perform several roles in a small store, but responsibilities and audit trails remain distinct.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| BOPIS | Buy online, pick up in store; a fulfillment option. | Assuming every pickup is curbside. |
| Curbside | Pickup delivered to an outside designated handoff point. | Treating the customer's parking time as invisible. |
| Order | A customer's checkout transaction; count in orders. | Equating orders with distinct customers. |
| Order line | A requested product or variant on an order; count in lines. | Equating a line with a unit when quantity exceeds one. |
| Unit | One physical item; count in units. | Mixing units and lines in one accuracy denominator. |
| Requested item | The product and quantity originally requested. | Rewriting the request after substitution. |
| Exact pick | A verified requested item and quantity with no replacement. | Counting a similar item as exact. |
| Exact-line rate | Exact fulfilled lines divided by eligible requested lines. | Calling it order accuracy without further checks. |
| Order completeness | Orders with all required lines fulfilled according to a stated rule divided by eligible orders. | Inferring it from line accuracy alone. |
| Pick accuracy | Correctly picked items divided by audited eligible items under a declared rule. | Using only scanned items and excluding missing picks. |
| Substitution | An alternative item offered or supplied in place of the requested item. | Assuming customer approval merely because an alternative exists. |
| Offer rate | Lines with a replacement offered divided by substitution-eligible unavailable lines. | Dividing by all lines without saying so. |
| Acceptance rate | Accepted offers divided by offers with a resolved customer response. | Silently removing no-response cases. |
| No-response rate | Offers without a response by cutoff divided by offers sent. | Classifying no response as affirmative consent. |
| Short | Requested quantity not supplied; count in units or lines. | Treating a canceled line as a complete pick. |
| Exception | A deviation requiring resolution, such as a missing or damaged item. | Counting every repeated contact as a separate affected order. |
| Promise | The customer-visible readiness or pickup commitment with timestamp and terms. | Replacing the original promise with the latest estimate. |
| Ready | All required work is complete, the order is staged, and it can be released. | Marking ready when a picker merely starts. |
| Staging | Holding picked items in a controlled location before handoff. | Ignoring cold-chain or access controls. |
| Check-in | Recorded signal that the customer is present and requesting service. | Treating an early app tap as verified physical arrival. |
| Arrival | Customer's physical presence at the pickup point. | Assuming every customer uses the app. |
| Handoff | Release of the order to the authorized recipient. | Using associate dispatch as handoff completion. |
| Wait | Elapsed time from a declared start event to handoff; minutes. | Comparing different start definitions without adjustment. |
| Percentile | Value at or below which a specified share of observations falls under a stated method. | Calling a 95th percentile an average. |
| Right-censoring | An observation whose final event has not occurred by the reporting cutoff. | Dropping still-waiting customers from live wait reports. |
| Productive time | Measured time spent on defined order work; minutes. | Treating it as all paid time. |
| Allowance | Time reserved for breaks, interruptions, and other nonproductive activity. | Applying an allowance twice. |
| Utilization | Workload divided by available service capacity over a specified period. | Assuming 100% utilization yields acceptable waits. |
| Capacity | Maximum sustainable throughput under stated resources and service constraints. | Treating a daily average as peak-hour capacity. |
| Incremental contribution | Change in contribution caused by the initiative after relevant costs. | Calling total pickup sales incremental. |
| Cannibalization | Demand shifted from another channel rather than newly created. | Assuming every channel shift is a lost sale. |
| Service recovery | Corrective action after a service failure, such as a refund or contact. | Omitting recovery costs from economics. |
| Accessible pickup | A pickup process usable by customers with different access needs. | Making app use or vehicle ownership the only path. |
| Cold chain | Temperature-controlled handling needed to maintain product condition. | Treating all staged items as ambient goods. |
| Event time | Time an operational event happened. | Substituting message ingestion time without disclosure. |
| Event ingestion time | Time the system received an event. | Assuming late messages imply late physical work. |

## 5. The mechanics, step by step

### Step 1 — Define the operating promise

**Input:** store hours, assortment, current inventory, customer-facing terms, historical demand, and site constraints.
**Owner:** digital promise owner with store operations and customer experience.
**Output:** a versioned, customer-visible promise and eligibility rule.

Specify whether the promise is a readiness deadline, a collection window, a handoff wait target, or a combination.
State the time zone and the store's actual operating hours.
Describe the cutoff for same-day orders and what happens when the store closes before the customer arrives.
If a customer selects a future window, distinguish the scheduled window from the earliest time the order is ready.
Do not present an internal forecast as a guaranteed handoff time.

Check constraints before accepting an order.
A large appliance may require a different crew and vehicle loading procedure from a small grocery bag.
Hazardous or age-restricted products may require special handling or be excluded.
Refrigerated items require available temperature-controlled space.
A pickup location may be temporarily unsafe because of weather or construction.
The eligibility rule should capture those constraints before checkout, not after a customer parks.

Agree an operational stop rule.
For example, the store manager may pause new curbside slots when the safe staging area is full.
The digital system must receive the pause quickly enough to avoid accepting orders the store cannot support.
Record the pause reason and start and end times so that the effect on sales and service can later be evaluated.
Offer a viable alternative only if it is actually available; do not silently move an order to a different store.

### Step 2 — Reserve inventory and workload capacity

**Input:** requested stock-keeping units (SKUs), quantities, available-to-promise values, and slot capacity.
**Owner:** order management and inventory systems; store operations owns local capacity settings.
**Output:** accepted, rejected, or alternative-offered order with a durable reservation record.

A shelf balance is not automatically a promiseable balance.
Existing reservations, damaged items, safety buffers, and timing of inventory updates matter.
The inventory topic explains the underlying reliability problem; here the practical question is whether a picker can fulfill the commitment.
Use the same reservation identity when the picker later reports a short.
Avoid a design that permits two channels to consume the same last unit without an atomic reservation or equivalent control.

Reserve labor capacity in a time bucket that reflects when work occurs.
A 5 p.m. collection may require picking at 3 p.m., while the handoff consumes staff at 5 p.m.
Daily headcount does not guarantee sufficient simultaneous coverage.
Separate picking, exception resolution, staging, and handoff workloads where they use different constraints.
Allow for cross-trained associates only if the store can actually release them from other tasks.

If the capacity system is unavailable, use an approved fallback that limits exposure rather than inventing inventory or capacity.
The owner must decide whether to stop accepting new orders, extend a displayed window with consent, or route customers to another supported option.
Already accepted orders need an explicit recovery plan.
Log the outage and affected promises for later reconciliation.

### Step 3 — Release work in an operationally sensible sequence

**Input:** accepted orders, promised deadlines, store layout, product handling rules, and available associates.
**Owner:** store fulfillment lead.
**Output:** a prioritized pick task or batch with a traceable order association.

Sequence by the real deadline and product needs, not simply by the order in which tasks entered a screen.
Perishable items may be picked closer to handoff or staged in temperature-controlled storage.
An order with a near deadline may need priority over a larger batch that is operationally convenient.
Batching can reduce walking time, but only if items remain attributable to the correct orders.
A batch that improves average pick time while causing late orders is not necessarily an improvement.

Record the release and start events separately.
Release-to-start captures waiting for an associate; start-to-finish captures task execution.
If these are combined, a staffing delay can be mistaken for slow individual work.
Review tasks that remain open near the deadline and assign ownership for intervention.
Avoid measuring associates only by speed, which can encourage shortcuts on verification and substitutions.

### Step 4 — Find and verify the requested product

**Input:** product identifier, variant, quantity, location hint, and customer instructions.
**Owner:** trained picker.
**Output:** verified exact item, verified shortage, or an exception requiring resolution.

Use the location hint to reduce search time, but do not treat it as proof that an item exists.
Verify the specific variant and quantity against the order.
Where suitable, scan the item identifier and associate it with the order container.
A successful scan does not prove the product is undamaged, in date, or placed in the right bag.
Physical inspection and container control remain necessary.

If the expected location is empty, follow a bounded search policy.
Check approved secondary locations and recent replenishment information before recording a short.
Set a search-time limit appropriate to the product and promise; excessive searching consumes capacity for other customers.
A shortage may reflect theft, misplacement, late receipts, or a record error, so do not infer the cause from a single failed search.
Record the reason using controlled codes and route recurrent patterns to inventory control.

For variable-weight goods, define the tolerated weight range and price calculation before picking.
For multipacks, distinguish one sellable pack from the number of individual contents.
For serial-controlled goods, verify the serial number if the retailer's policy requires it.
These product-specific rules belong in the task instructions and audit design.
They should not be inferred from a generic scan-success metric.

### Step 5 — Resolve a missing or unsuitable item

**Input:** shortage, customer substitution preferences, approved alternatives, prices, and a response deadline.
**Owner:** picker proposes; customer authorizes under the published policy; supervisor handles exceptions.
**Output:** exact item, authorized replacement, declined replacement, refund, or cancellation.

Store the original request separately from the replacement.
Show the alternative's identity, size, price, and any material differences the customer needs to decide.
A replacement that is larger or cheaper is not automatically acceptable if dietary restrictions or compatibility matter.
A generic preference such as “allow substitutions” must be interpreted in the context of the retailer's actual terms and product risks.
Do not infer affirmative consent from a customer failing to respond to a message.

Define a deterministic no-response path before the deadline.
For example, if the customer's saved preference is “ask first,” an unanswered offer leads to a short and refund rather than an unapproved item.
If the customer has explicitly preauthorized a class of replacements, apply the specified limits and retain evidence of that authorization.
For an age-restricted or otherwise regulated item, use a policy reviewed for the relevant jurisdiction.
If an alternative cannot be made safely, do not substitute it merely to improve completeness.

Separate offer, response, and execution.
An offered substitute may be accepted but still unavailable by the time the picker returns.
A customer may decline or change the preference before the stated cutoff.
Record the offer version, response, final picked item, and final charged amount.
Notify the customer of the outcome before or at handoff and make corrections possible through a supported channel.

### Step 6 — Stage and mark ready

**Input:** verified items, container identifiers, handling requirements, order identity, and staging locations.
**Owner:** staging associate or picker under store procedure.
**Output:** complete physical staging and a trustworthy ready event.

Assign each container to the order and to an actual staging location.
An order may span ambient, refrigerated, frozen, and secure locations.
The handoff task must show every required location without exposing unnecessary customer details.
Maintain appropriate temperature and duration controls according to product requirements and local policy.
A full refrigerator is a capacity constraint, not a reason to leave cold items at ambient temperature.

Perform a completeness check before marking ready.
Confirm that all approved items and authorized shorts have a final status.
Confirm that the number of bags or totes is recorded, including any controlled items held separately.
A ready notification should not be sent while an unresolved substitution or missing bag remains.
If an order later becomes unready, preserve both the original event and the correction.

Record the actual time staging is complete, not simply when the application updates.
An offline device may transmit an event later than it occurred.
Keep both event time and ingestion time, with a reason or quality flag for material discrepancies.
Physical audits of a sample of ready orders can detect systematic premature notifications.

### Step 7 — Recognize arrival and manage the queue

**Input:** readiness status, customer arrival or check-in, accessible contact options, queue order, and available handoff staff.
**Owner:** handoff lead or store manager.
**Output:** a visible queue with a documented start time and service priority.

Offer at least one usable path for customers who cannot or do not use the app.
For example, a clearly posted phone number or staffed desk can complement a digital check-in.
A customer who checks in early should not automatically start a physical-arrival wait clock unless that is the stated metric.
Capture both app check-in and observed arrival where feasible.
If physical arrival is unobservable, label the metric check-in-to-handoff rather than claiming true arrival-to-handoff.

Keep ready and unready orders distinguishable.
A customer who arrives before an order is ready should receive a realistic update and a clear choice where feasible.
Do not exclude that customer from the end-to-end experience simply because the operational queue starts later.
For analysis, show arrival-to-ready and ready-or-arrival-to-handoff separately.
The total customer wait is still arrival-to-handoff if arrival is observable.

Use a published and fair queue rule, with exceptions for safety, accessibility, or product handling where appropriate.
An associate should not need to guess whether a longer-waiting customer is visible in another application.
Provide a safe route between store and vehicle, including weather and traffic precautions.
Avoid requiring associates to enter a customer's vehicle; loading procedures should follow local safety policy.

### Step 8 — Retrieve and hand over

**Input:** queue entry, staging locations, pickup credential, bag count, and customer authorization.
**Owner:** handoff associate.
**Output:** correct order released to the authorized recipient and handoff event recorded.

Retrieve all containers, including cold and secure items.
Check the order identifier against the pickup credential using the retailer's approved procedure.
Avoid displaying a full customer name or other unnecessary personal data on an exterior bag or sign.
Use a process for authorized alternate collectors and for customers who cannot produce an expected credential.
Do not improvise around age or identity checks when a product or local rule requires them.

At the vehicle, confirm the correct customer and order without exposing the complete order to bystanders.
Communicate authorized substitutions and shorts in an accessible way.
If a customer rejects an item at handoff, record the rejection and initiate the applicable refund or return process.
Mark the handoff at actual completion, not when the associate leaves the staging area.
Capture an exception if a customer leaves, cannot be located, or disputes the order.

### Step 9 — Close, reconcile, and learn

**Input:** physical handoff, payments, refunds, uncollected orders, exception events, and customer feedback.
**Owner:** store manager for physical reconciliation; finance and order systems for financial reconciliation.
**Output:** closed order, accurate inventory state, documented recovery, and improvement actions.

Reconcile picked quantities, substitutions, charges, and refunds.
Release or dispose of uncollected items under product-specific policy and the stated collection deadline.
Return sellable items to inventory only after verifying their condition and handling history.
A digital cancellation alone does not prove that the physical goods have returned to a sellable location.

Review misses by mechanism rather than by associate name alone.
A cluster of stockouts may indicate receiving or inventory-record latency.
A queue at one curbside entrance may indicate traffic layout rather than inadequate average staffing.
Slow substitution responses may point to unclear customer communication or poor connectivity.
Assign each corrective action an owner, a deadline, and a measurable result.

### Step 10 — Review the operating control loop

**Input:** a consistent weekly scorecard, cost data, capacity forecasts, and customer complaints.
**Owner:** store operations and channel owner, with finance and customer experience.
**Output:** decisions on capacity, service design, and investment.

Show exact-line fulfillment, order-level completeness, readiness, handoff wait distribution, complaints, and relevant cost together.
Segment by store, hour, fulfillment mode, and customer journey where sample sizes permit.
Display counts beside percentages and suppress or aggregate unstable small cells under the organization's privacy policy.
Compare against the original promise, not a later modified promise.
Review open and missing-event orders separately from completed orders.

A practical weekly meeting distinguishes three questions.
Did customers receive what they requested and when they expected it?
Did the store have enough safe, trained capacity at the relevant time?
Did the service create incremental contribution after relevant displacement and costs?
A favorable answer to one does not automatically answer the other two.

### Step 11 — Test a change before expanding it

**Input:** baseline service, current cost, proposed intervention, and an evaluation design.
**Owner:** channel sponsor, finance, store operations, and customer experience.
**Output:** a decision with quantified uncertainty and pre-agreed guardrails.

Specify the change narrowly enough to identify its mechanism.
For example, adding an evening handoff associate changes capacity at a specific time.
Changing picking technology, promotional exposure, and the pickup promise simultaneously makes it difficult to attribute the result.
Document which activities the new associate would otherwise have performed.
If another employee must cover those activities, include that replacement work in incremental cost.

Choose a comparison before observing the results.
Randomly assign comparable stores where feasible and check whether customers can switch between treated and control stores.
If neighboring stores share demand, assign geographic clusters or explicitly measure the displacement.
When assignment is not random, compare pre-period trends and explain why remaining differences may bias the estimate.
Do not call an uncontrolled before-and-after comparison a randomized experiment.

Define the decision horizon to include setup costs and relevant delayed effects.
A customer may return a substitute after the reporting week, creating recovery and reverse-logistics cost.
A shift in staffing can create overtime in a different department.
Charge such effects to the decision when they are incremental and material.
Separate one-time implementation costs from the ongoing operating run rate.

Choose a primary economic metric and a small set of service and safety guardrails.
For instance, contribution per eligible store-day can be the economic metric.
Readiness against the original promise, the 90th percentile of completed wait, and open-wait ages can be guardrails.
Also monitor independent pick audits and customer complaints by mode.
Specify minimum event coverage so that a falling wait metric cannot be achieved by losing timestamps.

Interpret results at the assignment unit.
Orders within the same store share staff, assortment, local demand, and other conditions.
Treating every order as statistically independent can understate uncertainty.
Show the estimate, uncertainty interval, sample counts, and exclusions together.
The sponsor should make the scale, modify, or stop decision against thresholds set before the results are known.

If the pilot improves service but reduces contribution, describe the tradeoff explicitly.
An executive may fund a service objective for strategic reasons, but should identify its cost and owner.
If the pilot improves contribution while safety or consent guardrails fail, do not scale the unsafe process.
Record the decision and the next review date so temporary exceptions do not become undocumented permanent policy.

## 6. Worked example

### Case and decision

A hypothetical general-merchandise store is considering a daily pickup offer.
Management needs to decide whether to expand the offer after a controlled pilot.
The example uses one day of observed workload and a separate, explicitly assumed demand effect.
It is illustrative; it is not a claim about an actual company.

The forecast is 100 orders per day, each with an average of 2.4 requested lines.
The staffing plan is based on those 100 orders and the work standards below.
The internal target is at least 95% exact-line fulfillment and at most 5 minutes median curbside wait.
The customer commitment is that orders accepted for the day will be ready by their displayed individual deadlines.
No wait-time guarantee is assumed in the checkout commitment.

| Input | Value | Interpretation |
|---|---:|---|
| Accepted and eligible orders | 100 orders | All included in this example's fulfillment cohort. |
| Requested eligible lines | 240 lines | One unit per line for this simplified case. |
| Exactly fulfilled lines | 228 lines | Original item and quantity supplied. |
| Accepted replacement lines | 8 lines | All were explicitly authorized and supplied. |
| Unfilled lines | 4 lines | Refunded or otherwise settled under the order terms. |
| Curbside orders | 60 orders | Customer collects at the vehicle. |
| In-store pickup orders | 40 orders | Customer collects at a service point. |
| Orders with an exception | 20 affected orders | Each incurs one 3-minute resolution task in this example. |

The exception assumption is independent of the 12 non-exact lines.
The 20 affected orders include issues such as contact or staging questions, not just missing lines.
No claim is made that the 12 non-exact lines appear on 12 different orders.
We therefore cannot calculate the exact share of fully complete orders from these aggregates.

### Fulfillment measures

Exact-line fulfillment is:

`228 exact lines / 240 requested lines = 95.0%`.

The service target of at least 95% is met exactly in this daily sample.
That does not establish that the underlying process consistently meets the target.
Authorized fulfillment including replacements is:

`(228 exact + 8 authorized replacement lines) / 240 requested lines = 98.33%`.

The unfilled-line rate is `4 / 240 = 1.67%`.
If all 12 non-exact lines were eligible for substitution and exactly 10 offers were made, the offer rate would be `10 / 12 = 83.33%`.
If 8 offers were accepted and 2 declined, the acceptance rate among resolved offers would be `8 / 10 = 80.0%`.
This additional offer example assumes no unanswered offers; otherwise no-response must be reported separately.
These rates should not be relabeled as order accuracy.

The order-level completeness rate requires the order identifier for each line.
If the 12 non-exact lines fall on 12 different orders, 88 orders are exact-complete.
If they all fall on 5 orders, 95 orders are exact-complete.
Both allocations are compatible with the line totals, assuming the affected orders have enough lines.
The line aggregate alone therefore cannot identify the exact-complete order count.
A line-level event table resolves the ambiguity.

### Labor calculation

Assume the following measured productive work standards.
A standard is an input to this case, not an industry benchmark.
The standards include no breaks or unassigned time.
They do not claim that all work is perfectly divisible into fractional associates.

| Activity | Driver | Standard | Productive minutes |
|---|---:|---:|---:|
| Find and verify | 240 lines | 1.4 minutes per line | 336 minutes |
| Stage | 100 orders | 1.2 minutes per order | 120 minutes |
| Common handoff work | 100 orders | 1.5 minutes per order | 150 minutes |
| Extra curbside travel and loading | 60 orders | 2.5 minutes per curbside order | 150 minutes |
| Extra in-store counter work | 40 orders | 1.0 minute per in-store order | 40 minutes |
| Resolve exceptions | 20 affected orders | 3.0 minutes per affected order | 60 minutes |
| **Total** | | | **856 minutes** |

The common handoff standard applies to both channels.
The channel-specific standards are additional, so the calculation does not count the same work twice.
Productive labor is `856 / 60 = 14.2667 hours` per day.
Suppose productive work accounts for 80% of paid time after breaks, interruptions, and other allowances.
Required paid time is `14.2667 / 0.80 = 17.8333 hours` per day.
At a fully loaded wage of $22.00 per paid hour, labor cost is `$392.33` per day after rounding.
This is not a roster: staffing must be placed in the actual hours of demand, subject to shift lengths and coverage rules.

Add $35.00 per day for incremental bags and supplies and $50.00 per day for avoidable pickup supervision and equipment.
These amounts are assumptions about costs that would change with the program.
Total relevant daily operating cost is `$392.33 + $35.00 + $50.00 = $477.33`.
Cost per completed or accepted order is not automatically the same; here the denominator is the 100 accepted orders.
The relevant cost is `$4.77 per accepted order` after rounding.
If an equipment payment is already unavoidable regardless of this decision, exclude that sunk amount from the incremental case.

### Illustrative wait calculation

Suppose 10 observed curbside waits in minutes are:
`1, 2, 2, 3, 3, 4, 5, 7, 9, 14`.
All 10 are completed and use valid check-in-to-handoff timestamps.
The median is the mean of the fifth and sixth ordered values: `(3 + 4) / 2 = 3.5 minutes`.
Using the nearest-rank definition, the 90th percentile is the ninth value, or 9 minutes.
The mean is `50 / 10 = 5.0 minutes`.
The example meets the 5-minute median target, but the 14-minute maximum highlights a long wait for one customer.
Ten observations are too few for a stable tail estimate; the example demonstrates the calculation only.
It also says nothing about actual arrival-to-check-in time.

### Incremental business case

Assume a controlled comparison supports the following forecast for the 100 pickup orders.
Forty orders are incremental to the retailer, while 60 shift from an existing store channel.
Each incremental order contributes $15.00 before pickup operating cost.
Each shifted order creates a $3.00 loss of contribution relative to its prior-channel counterfactual.
The $3.00 is a net difference in product and channel contribution before the pickup costs already counted below.
It must not include those same pickup costs again.

Incremental contribution before pickup cost is:

`40 × $15.00 - 60 × $3.00 = $420.00 per day`.

After the $477.33 per day relevant pickup cost, incremental contribution is:

`$420.00 - $477.33 = -$57.33 per day`.

The business case is negative on these assumptions despite a favorable exact-line rate and median wait.
For 100 orders, with all other assumptions unchanged, break-even incremental share `s` solves:

`100 × [15s - 3(1 - s)] = 477.33`.

Thus `s = (4.7733 + 3) / 18 = 0.43185`, or about 43.2% of orders.
This threshold is sensitive to the contribution assumptions, labor standard, order mix, and any omitted recovery or capital cost.
It is not evidence that the actual incremental share is 43.2%.

### Decision and ownership

Do not expand solely on the one-day service result.
The channel owner and finance partner should obtain a credible incremental-demand estimate across comparable stores and time periods.
The store manager should measure actual interval-level labor and tail waits, including incomplete observations.
The customer experience owner should inspect substitution consent and complaints.
The decision owner may extend a controlled pilot if service and safety guardrails hold and the experiment can resolve the economic uncertainty.
If no credible path to positive incremental contribution exists, change the operating model or stop the offer rather than relabeling shifted sales as growth.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Copy it into a file and run `python pickup_case.py`.
It reproduces the synthetic case's arithmetic and prints an interval-capacity example.
No external data, package, network access, or credentials are required.
The program deliberately does not invent order-level accuracy from line totals.

```python
from math import ceil, factorial
from statistics import mean, median

# All quantities are synthetic and refer to one operating day.
orders = 100
requested_lines = 240
exact_lines = 228
accepted_replacements = 8
unfilled_lines = 4

curbside_orders = 60
instore_orders = 40
exception_orders = 20

# Productive work standards; each component is additional to the others.
minutes = {
    "pick": requested_lines * 1.4,
    "stage": orders * 1.2,
    "common_handoff": orders * 1.5,
    "curbside_extra": curbside_orders * 2.5,
    "instore_extra": instore_orders * 1.0,
    "exception": exception_orders * 3.0,
}
productive_minutes = sum(minutes.values())
productive_fraction = 0.80
paid_hours = productive_minutes / 60 / productive_fraction
loaded_wage = 22.0
supplies = 35.0
avoidable_fixed = 50.0
pickup_cost = paid_hours * loaded_wage + supplies + avoidable_fixed

incremental_orders = 40
shifted_orders = orders - incremental_orders
incremental_order_contribution = 15.0
shifted_order_delta = -3.0
contribution_before_pickup = (
    incremental_orders * incremental_order_contribution
    + shifted_orders * shifted_order_delta
)
net_incremental = contribution_before_pickup - pickup_cost
break_even_share = (pickup_cost / orders - shifted_order_delta) / (
    incremental_order_contribution - shifted_order_delta
)

waits = [1, 2, 2, 3, 3, 4, 5, 7, 9, 14]

def nearest_rank(values, percentile):
    """Return the nearest-rank percentile for nonempty observations."""
    if not values or not 0 < percentile <= 1:
        raise ValueError("need values and 0 < percentile <= 1")
    ordered = sorted(values)
    return ordered[ceil(percentile * len(ordered)) - 1]

def erlang_c(arrival_rate, service_rate, servers):
    """M/M/c queue: probability of waiting and mean queue wait in minutes.

    Rates are per hour; independent Poisson arrivals, exponential service,
    identical servers, FIFO, infinite waiting space and stable utilization
    are assumptions, not properties established by the input data.
    """
    if arrival_rate < 0 or service_rate <= 0 or servers < 1:
        raise ValueError("rates and server count must be valid")
    offered_load = arrival_rate / service_rate
    utilization = offered_load / servers
    if utilization >= 1:
        raise ValueError("queue is not stable")
    terms = sum(offered_load ** n / factorial(n) for n in range(servers))
    tail = offered_load ** servers / (
        factorial(servers) * (1 - utilization)
    )
    wait_probability = tail / (terms + tail)
    wait_minutes = 60 * wait_probability / (
        servers * service_rate - arrival_rate
    )
    return utilization, wait_probability, wait_minutes

assert exact_lines + accepted_replacements + unfilled_lines == requested_lines
assert curbside_orders + instore_orders == orders

print(f"Exact-line rate: {exact_lines / requested_lines:.2%}")
print(
    "Authorized fulfillment: "
    f"{(exact_lines + accepted_replacements) / requested_lines:.2%}"
)
print(f"Unfilled-line rate: {unfilled_lines / requested_lines:.2%}")
print(f"Productive minutes: {productive_minutes:.0f}")
print(f"Paid labor hours: {paid_hours:.2f}")
print(f"Pickup cost: ${pickup_cost:.2f}")
print(f"Cost per accepted order: ${pickup_cost / orders:.2f}")
print(f"Net incremental contribution: ${net_incremental:.2f}")
print(f"Break-even incremental share: {break_even_share:.2%}")
print(f"Mean wait: {mean(waits):.1f} minutes")
print(f"Median wait: {median(waits):.1f} minutes")
print(f"Nearest-rank p90 wait: {nearest_rank(waits, 0.90)} minutes")

# Hypothetical peak hour, not the observed ten waits above.
rho, p_wait, queue_minutes = erlang_c(60, 25, 3)
print(f"Peak-hour utilization: {rho:.1%}")
print(f"Probability of queueing: {p_wait:.1%}")
print(f"Mean queue wait: {queue_minutes:.2f} minutes")
```

Expected output, with ordinary Python floating-point rounding:

```text
Exact-line rate: 95.00%
Authorized fulfillment: 98.33%
Unfilled-line rate: 1.67%
Productive minutes: 856
Paid labor hours: 17.83
Pickup cost: $477.33
Cost per accepted order: $4.77
Net incremental contribution: $-57.33
Break-even incremental share: 43.19%
Mean wait: 5.0 minutes
Median wait: 3.5 minutes
Nearest-rank p90 wait: 9 minutes
Peak-hour utilization: 80.0%
Probability of queueing: 64.7%
Mean queue wait: 2.59 minutes
```

The queue calculation is a separate hypothetical peak-hour scenario.
It assumes 60 arrivals per hour, three continuously available associates, and 25 service completions per associate-hour.
The offered load is `60 / 25 = 2.4` associate-equivalents and utilization is `2.4 / 3 = 0.80`.
The model estimates time waiting before service, not the customer's total handoff time.
Under the assumed exponential service rate, mean service time adds `60 / 25 = 2.4 minutes`.
Thus the modeled mean queue-plus-service time is approximately `2.59 + 2.40 = 4.99 minutes`.
The model does not capture early check-ins, correlated arrivals, walking routes, shared picking duties, or no-shows.
Use observed arrival and service distributions or a validated simulation when these assumptions are not reasonable.
An unstable input, such as 75 or more arrivals per hour with these service resources, is rejected rather than assigned a finite steady-state wait.

## 8. Data, standards, and regulatory context

### Event and entity design

A reliable analysis starts with an order and line identity that persists across systems.
The original order, requested line, accepted substitution, and final supplied product must remain distinguishable.
Keep the physical container and staging-location identities separate from customer identity.
A single order can contain multiple containers and multiple temperature zones.
A single customer can place several orders, and one authorized collector can collect another person's order.
A data model that assumes one bag, one person, and one order is fragile.

| Entity or event | Minimum useful fields | Quality control |
|---|---|---|
| Order acceptance | Order ID, store, timestamp, original promise, channel | Require a unique event ID and preserve the original promise. |
| Requested line | Order ID, line ID, SKU, requested quantity and unit | Retain the original even after a replacement. |
| Reservation | Line ID, store, quantity, reservation state and time | Reconcile duplicate or expired reservations. |
| Pick action | Line ID, actual SKU, quantity, picker task, time, result | Distinguish attempted scan, verified pick, and short. |
| Substitute offer | Offer ID, line ID, alternative, price, policy version, time | Link response to the correct offer version. |
| Customer response | Offer ID, response, time, evidence or channel | Preserve decline and no-response outcomes. |
| Staging | Container ID, location, handling class, timestamp | Audit physical location and missing containers. |
| Ready | Order ID, event time, ingestion time, final item status | Reject or flag ready events with unresolved required work. |
| Arrival or check-in | Order ID, event type, timestamp, source | Do not conflate app check-in with observed presence. |
| Handoff | Order ID, timestamp, outcome, credential method | Record failed and successful attempts separately. |
| Financial settlement | Line ID, charge, adjustment, refund, currency | Reconcile charges to the actual fulfilled goods. |

Use a stable event identifier so retrying a message does not create a second handoff or second refund.
Event histories should retain corrections rather than silently overwrite the original fact.
Store local timestamps with time zone or offset and retain an unambiguous instant for elapsed-time calculations.
Daylight-saving transitions and device clock errors can otherwise create negative or implausible durations.
A late event is not necessarily a late service outcome; compare event time and ingestion time.

### Cohorts and denominators

For readiness, choose a cohort such as all orders accepted on a given day with a promise in the reporting horizon.
Count orders not ready by the deadline as misses, including orders that later cancel, unless a separately published policy defines an excusable customer-driven cancellation.
Report exclusions and their counts instead of dropping inconvenient cases.
A completion-day cohort can omit late orders that remain open, so it is unsuitable by itself for measuring promise performance.

For wait, define start and end events precisely and report the share with valid timestamps.
Completed-only wait distributions are biased downward while long-waiting orders remain open.
A live dashboard should show open waits and their current age alongside completed waits.
For formal time-to-event estimation, choose a method appropriate to censoring and competing outcomes such as abandonment.
Do not insert zero minutes for missing timestamps.

For substitutions, report unavailable eligible lines, offers, accepted offers, declined offers, and unanswered offers.
A rate among resolved offers should be accompanied by the count and share of unresolved offers.
An authorized preselection and a real-time acceptance are different evidence paths.
Keep each policy version with the order, since consent language can change over time.

For accuracy, distinguish a process scan rate from independent audit accuracy.
A scan can be technically successful while the wrong variant, quantity, container, or customer receives the item.
A representative audit samples completed and failed orders, includes missing lines, and verifies ground truth without relying solely on the operational scan.
If audits are concentrated on complaints, estimate selection bias before treating the results as representative.

### Product identifiers and traceability

[GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications) describe GS1 identification keys and barcode application rules.
A Global Trade Item Number (GTIN) identifies a trade item under GS1 rules; a retailer's internal SKU may identify a different level of assortment.
A barcode scan can help match a trade item to an order, but it does not by itself verify quantity, condition, customer authorization, or order-bag association.
Use the current specification and the retailer's licensed implementation guidance for actual barcode design.
Do not assume every locally assigned SKU is a GTIN.

For food, lot, date, and handling information may be important for recalls and quality.
The US Food and Drug Administration's [Food Traceability Rule page](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods) describes additional recordkeeping requirements for certain foods and covered entities.
Applicability depends on the food, activity, exemptions, and current compliance dates; consult current official guidance and counsel rather than treating every pickup order as covered.
A retailer may maintain additional traceability as a voluntary control even when a particular legal rule does not apply.
The example in this unit does not assert a particular legal compliance deadline.

### Privacy and access

Pickup systems may process names, phone numbers, order contents, precise arrival times, and vehicle descriptions.
Collect only what is necessary for the operational purpose, restrict access by role, and set a documented retention schedule.
Use short-lived pickup credentials where appropriate and avoid exposing unnecessary information on public signage.
The [California Privacy Rights Act information page](https://cppa.ca.gov/regulations/) provides links to California Privacy Protection Agency regulations.
Whether a business or data use is covered depends on current law and facts; privacy counsel should review notices, rights handling, retention, and vendor arrangements.
These controls are design guidance, not a substitute for a legal determination.

The US Department of Justice's [ADA guidance](https://www.ada.gov/) describes obligations and resources concerning accessibility.
Specific obligations depend on the service, business, location, and current law.
As an operational design choice, provide a usable non-app path, clear signage, and a way to request assistance.
Do not assume that a single channel accommodates every disability or that a vehicle-based service can replace an accessible store entrance.
Test the actual journey with representative users and accessibility specialists.

### Safety and payments

Review workplace safety, traffic movement, lifting, and weather procedures with qualified local owners.
The US Occupational Safety and Health Administration provides [retail industry safety resources](https://www.osha.gov/retail).
Those resources are not a site-specific determination of compliance.
Store-specific assessments should cover moving vehicles, visibility, slip hazards, manual handling, and emergency escalation.
Set safe thresholds for pausing outside handoff; communicate an alternative before directing a customer elsewhere.

Payment authorization and final capture may occur at different times, especially when substitutions or variable-weight goods change the total.
Use the payment provider's documented integration and applicable payment-card security requirements.
Do not store full payment credentials in pickup event logs.
Reconcile final charges and refunds to the actual items and the customer's agreed price treatment.
A favorable operational substitution metric does not excuse an incorrect charge.

## 9. Industry practice and case evidence

### What public cases can and cannot establish

Public company material is useful for identifying an operating model and a customer-facing feature.
It is usually less useful for proving a causal effect on store profit or a universal staffing standard.
A retailer's stated service promise is not evidence that every order met the promise.
A financial result for a company is not the incremental profit of a specific pickup feature.
The cases below identify documented features and then label analytical implications separately.

### Target: order pickup and drive-up

Target's official [Order Pickup and Drive Up help pages](https://help.target.com/help/subcategoryarticle?childcat=Order+Pickup+%26+Drive+Up&parentcat=Fulfillment+Services) describe its pickup services and customer instructions.
The exact availability, procedures, and terms can change by store and date; consult the live pages for current details.
The existence of distinct pickup modes is a documented service design, not proof of any particular wait distribution.

**Inference:** when a retailer offers both in-store and outside handoff, it needs separate capacity assumptions for counter work and outdoor travel.
A single cost per order can conceal differences in walking distance, weather exposure, parking geometry, and customer arrival patterns.
An analyst should measure both shared work and mode-specific incremental work rather than multiplying total store labor by an arbitrary channel percentage.

**Decision lesson:** the store manager needs a capacity view by mode and interval.
The digital team needs a way to stop offering a mode when its safe capacity is unavailable.
The customer needs a clear explanation of the available alternative.
None of these decisions follows from a national marketing claim alone.

### Walmart: pickup and substitution choices

Walmart Business's [substitution help page](https://business.walmart.com/help/article/substitutions-for-store-pickup-and-delivery-items/b94b3c9413b14315acd003296238cdaa) describes customer-facing substitution functionality for eligible orders.
Features and terms vary, so current help and the customer's actual order terms govern a real transaction.
A customer's ability to set or review preferences is a documented workflow feature; it does not establish a particular acceptance rate.

**Inference:** preferences must be linked to the original order line and the specific policy in effect at purchase.
Operational systems should distinguish a customer-approved replacement from an item the picker merely regarded as similar.
A positive substitution experience depends on the price, size, and characteristics of the offered item, not just the existence of a replacement.

**Decision lesson:** monitor no-response and rejection separately from acceptance.
If substitutions rise because stock records deteriorate, celebrating a high acceptance rate would hide the underlying inventory problem.
Investigate the root cause before expanding the set of automatically eligible alternatives.

### IKEA: collection point and booking constraints

IKEA describes [Click and Collect](https://www.ikea.com/us/en/customer-service/services/click-collect/) as a customer pickup service, with details and availability subject to location and current terms.
The relevant public fact is that pickup is an offered collection method with stated customer instructions.
It does not reveal the retailer's internal labor standards, store-level unit economics, or causal demand lift.

**Inference:** bulky products require a different handling and vehicle-capacity design than a small parcel.
A reservation that considers only the number of orders may overbook staff or physical space when item dimensions vary.
A useful capacity model can include size or handling classes, but its coefficients need local measurement.

**Decision lesson:** classify eligibility and resource needs at product level.
Do not infer that every item can use the same handoff route or staffing standard.
A site may have adequate average order capacity while lacking safe space for a few oversized orders.

### Evidence hierarchy for an internal decision

Use transaction and operational events to describe what happened, after checking coverage and data quality.
Use independent audits and customer contact data to find mistakes that the transaction system misses.
Use an experiment or a defensible quasi-experiment to estimate incremental demand and contribution.
Use public cases to generate candidate designs, not to substitute for a local counterfactual.

A randomized store rollout can compare treated and control stores if operational spillovers are limited.
Where randomization is infeasible, matched stores and pre-period trends can support a qualified comparison.
Check seasonality, promotions, assortment, and local competition before attributing differences to pickup.
A treatment store's sales increase is not automatically incremental if nearby control stores lose the same customers.
State the unit of assignment and the likely spillover boundary.

Measure impacts over a long enough horizon to include repeat use, refunds, and returns where material.
Predefine primary economic and service outcomes, the analysis population, and stopping rules.
If the sample is small, report uncertainty and directional evidence instead of a precise universal effect.
A negative pilot is useful if it prevents scaling an unprofitable or unsafe operating model.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation and owner |
|---|---|---|---|---|
| Phantom stock | Repeated shorts on apparently available SKUs | Perpetual balance diverges from shelf | Broken promise and wasted search | Inventory control investigates counts and adjustment causes. |
| Double reservation | Same last unit assigned twice | Non-atomic channel allocation | One customer receives a short | Order-system owner fixes reservation consistency and reconciles duplicates. |
| Premature ready event | Ready orders still have open pick tasks | Status updates precede physical staging | Customer arrives to an unready order | Fulfillment lead enforces readiness checks and audits samples. |
| Wrong variant | Scan overrides or high exchange rate | Similar packaging and weak validation | Wrong item or avoidable return | Picker verifies identifier and variant; quality owner audits. |
| Missing bag | Bag count differs from staging retrieval | Order spans multiple locations | Partial handoff and return trip | Handoff lead checks container count and all locations. |
| Unapproved substitute | Acceptance evidence absent | No response treated as consent | Charge dispute and customer harm | Policy owner defines no-response path and retains evidence. |
| Unsafe replacement | Customer complaint about incompatibility | Generic similarity ignores restrictions | Health, safety, or usability risk | Merchandising defines exclusions; picker escalates. |
| Cold-chain breach | Staging duration or temperature out of range | Insufficient cold capacity or slow handoff | Product quality loss | Store manager monitors capacity and follows product disposition policy. |
| Hidden curbside queue | Rising open check-ins absent from completed dashboard | Live cases excluded | Tail waits understated | Analytics owner shows open ages and completion coverage. |
| Early app check-in | Negative arrival-to-check-in or large early interval | Customer signals before physical arrival | Misleading wait comparisons | Capture distinct events and label the metric accurately. |
| Queue starvation | Some customers repeatedly bypassed | Priority rule is inconsistent across channels | Inequitable wait and complaints | Handoff lead uses a visible queue and documented exceptions. |
| Incorrect recipient | Credential mismatch or missing handoff verification | Weak identity and container controls | Fraud, loss, privacy exposure | Store manager requires authorized verification and exception review. |
| Unsafe loading | Near misses around pickup bays | Vehicle and pedestrian paths conflict | Associate or customer injury | Safety owner redesigns traffic flow and pauses unsafe operation. |
| Inaccessible check-in | Calls for help or abandoned arrivals | App-only or unclear instructions | Exclusion and missed orders | Customer experience owner supplies accessible alternatives. |
| Labor underestimation | High overtime despite low productive minutes | Breaks, travel, or other work omitted | Cost overrun and burnout | Workforce planner measures paid time and interval coverage. |
| Slot overbooking | Late readiness concentrated in promotions | Daily forecast hides peak | Cascading misses | Capacity owner caps interval-level acceptance. |
| Misleading substitution rate | High acceptance but many no-responses | Denominator excludes unresolved offers | Overstated customer approval | Analytics owner reports all offer outcomes. |
| Inaccurate clock | Negative durations or late event ingestion | Device clock drift or offline sync | Invalid wait and promise metrics | Data engineering stores event and ingestion times and flags anomalies. |
| Duplicate refund | Multiple settlement events for one line | Retry lacks idempotency | Financial leakage | Payments owner uses stable transaction IDs and reconciliation. |
| Uncollected order | Growing aged staging inventory | No expiry or customer contact | Lost capacity and inventory distortion | Store manager follows expiry, contact, and disposition rules. |
| Misleading pilot economics | Sales up but comparable stores down | Channel or store cannibalization | Negative incremental contribution | Finance estimates a counterfactual and includes displaced margin. |
| Double-counted savings | Labor credited to two initiatives | Shared capacity is allocated twice | Inflated business case | Finance maintains a single cost baseline and ownership ledger. |

Investigate clusters rather than assuming every incident is independent.
For example, a promotion may simultaneously increase stockouts, picker search time, late readiness, and curbside queues.
Correcting only the handoff staffing may reduce the visible queue while leaving the cause intact.
Record the causal hypothesis and check whether the intervention changes the upstream and downstream measures.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Every requested line is recorded once | Defines the fulfillment denominator | Duplicate lines, canceled lines, lost events | Reconcile checkout, order management, and settlement records. |
| A line has one unit in the example | Makes the line and unit quantities equivalent | Multi-unit or variable-weight lines | Calculate both line and unit measures in production. |
| Pick standards are representative | Converts workload into labor | Layout, season, product mix, learning | Sample work across stores, shifts, and handling classes. |
| Productive fraction is 80% | Converts productive minutes into paid hours | Break policy, interruptions, idle periods differ | Compare with paid schedules and observed work sampling. |
| Labor is available when needed | Connects daily work to service capacity | Peaks, breaks, competing store tasks | Model short intervals and construct feasible rosters. |
| Fixed pickup cost is avoidable | Makes it relevant to the decision | Equipment or supervision remains after closure | Trace each cost to the decision and contract terms. |
| Incremental order contribution is $15.00 | Values genuinely new demand | Mix, returns, discounts, or margin change | Estimate net contribution by cohort and follow returns. |
| Shifted-order delta is minus $3.00 | Values displaced channel demand | Prior channel or basket changes | Estimate a counterfactual by customer and store cohort. |
| Forty orders are incremental | Separates growth from channel shift | Self-selection and local spillovers | Use a controlled rollout or qualified quasi-experiment. |
| Costs are not double counted | Allows contribution components to add | Shifted delta already includes pickup labor | Maintain an explicit inclusion and exclusion ledger. |
| Wait sample is complete | Makes displayed descriptive statistics interpretable | Still-waiting and missing-event customers excluded | Report open cases, missingness, and cohort cutoff. |
| Check-in represents the chosen start | Gives the wait metric a defined meaning | Early or absent digital check-in | Capture physical arrival where feasible and report both. |
| Erlang-C assumptions hold in its example | Provides a simple queue estimate | Bursty arrivals, variable routing, shared tasks | Compare with actual distributions or use simulation. |
| Customer preference is current and specific | Supports authorized replacement | Preference changed, ambiguous, or expired | Store policy version and offer-specific response. |
| Price and refund policy is applied correctly | Connects operations to settlement | Delayed capture, incorrect override, tax differences | Reconcile final receipt and refund by line. |
| Product handling needs are known | Enables safe staging and release | Misclassified product or missing temperature history | Validate product master and audit actual handling. |

The one-day example is not a statistical performance assessment.
The queue model is steady-state and does not establish customer experience in a transient peak.
The business case excludes financing, taxes, and corporate overhead unless they change with the decision.
It should include incremental returns, recovery, and capital costs if those are material in the real case.
Legal and accessibility obligations require current, jurisdiction-specific review.

## 12. Exercises

### Exercise 1 — Trace an order

**Time:** 20 minutes.
**Task:** A customer orders three lines, one is unavailable, and an approved replacement is staged in a refrigerated tote.
The customer checks in before arriving and later collects at curbside.
Map the state transitions, timestamps, and physical containers from order acceptance through settlement.
**Deliverable:** a one-page sequence diagram and an owner for each exception.
**Acceptance criteria:** retain the original line, the replacement approval, both check-in and arrival, all staging locations, and the actual handoff.
Explain which events support readiness, actual wait, and financial reconciliation.

### Exercise 2 — Compute line metrics

**Time:** 15 minutes.
**Task:** Of 500 requested lines, 460 are exactly fulfilled, 25 have an approved replacement, and 15 are unfilled.
Of the 40 non-exact lines, 30 receive an offer; 25 offers are accepted, 3 declined, and 2 unanswered.
**Deliverable:** five rates with formulas and denominators: exact, authorized fulfillment, unfilled, offer, and acceptance among resolved offers.
**Acceptance criteria:** the rates reconcile to the supplied counts and the unanswered share is stated separately.
Explain why these figures do not identify exact-complete orders.

### Exercise 3 — Analyze wait

**Time:** 15 minutes.
**Task:** Six completed check-in-to-handoff waits are 2, 3, 3, 4, 8, and 16 minutes.
Two additional checked-in customers are still waiting at 10 and 12 minutes at the reporting cutoff.
**Deliverable:** completed-case mean, median, nearest-rank 90th percentile, and a short interpretation.
**Acceptance criteria:** do not count open waits as completed durations or drop them without disclosure.
Identify the event needed to measure physical arrival-to-handoff wait.

### Exercise 4 — Design substitutions

**Time:** 25 minutes.
**Task:** A customer selected “ask first” for a missing product and has not responded to a proposed alternative.
The order is due in 15 minutes, and the alternative differs in size and ingredients.
**Deliverable:** a decision table for accept, decline, no response, and alternative unavailable after acceptance.
**Acceptance criteria:** the original request remains visible, no response is not treated as approval, and refund or escalation ownership is explicit.
Explain what changes if a customer has previously given specific, valid authorization.

### Exercise 5 — Build a labor estimate

**Time:** 20 minutes.
**Task:** A store expects 80 orders and 200 lines per day.
Picking takes 1.5 productive minutes per line, staging 2 minutes per order, and handoff 3 minutes per order.
Exception work adds 40 productive minutes per day.
Productive work is 75% of paid time, and loaded labor is $24.00 per paid hour.
**Deliverable:** productive minutes, paid hours, and labor cost for one day.
**Acceptance criteria:** show the allowance conversion and do not round intermediate hours before computing cost.
Name two reasons this daily total is not a shift schedule.

### Exercise 6 — Check peak capacity

**Time:** 20 minutes.
**Task:** At the busiest hour, 45 handoffs arrive and two available associates can each complete 20 handoffs per hour.
Assume this is a sustained hour and that no other associate can help.
**Deliverable:** offered load, nominal hourly capacity, utilization, and a practical intervention.
**Acceptance criteria:** identify that demand exceeds nominal capacity and do not report a finite steady-state wait for this setup.
Explain why adding a third associate may still require measurement of service variability.

### Exercise 7 — Evaluate incrementality

**Time:** 20 minutes.
**Task:** A pickup program serves 200 orders per day.
A controlled estimate attributes 70 orders to incremental demand; 130 shift from another channel.
Incremental orders contribute $12.00 each before pickup cost, and shifted orders lose $2.00 each relative to their counterfactual before pickup cost.
Avoidable pickup cost is $650.00 per day.
**Deliverable:** daily incremental contribution, a decision recommendation, and one uncertainty that could reverse it.
**Acceptance criteria:** separate new and shifted orders, subtract the cost once, and avoid equating total sales with incremental value.

### Exercise 8 — Audit a dashboard

**Time:** 25 minutes.
**Task:** A dashboard reports 99% pick accuracy using successful scans only and a 2-minute mean wait using completed app check-ins only.
It excludes canceled orders and orders with missing timestamps.
**Deliverable:** a measurement critique and a replacement scorecard with at least five measures.
**Acceptance criteria:** specify each numerator, denominator, cohort, missingness measure, and owner.
Include a tail or open-wait measure and an independent quality check.

### Exercise 9 — Design a pilot

**Time:** 35 minutes.
**Task:** A retailer proposes a dedicated curbside associate during the evening peak in 20 stores.
The proposal claims faster waits and higher incremental profit but has no counterfactual.
**Deliverable:** an evaluation plan including assignment, outcomes, guardrails, analysis horizon, and decision rights.
**Acceptance criteria:** address store selection, spillovers, customer experience, safety, relevant labor cost, and statistical uncertainty.
Specify a stop rule and explain how the analysis will treat open orders and missing events.

## 13. Self-check answers

### Exercise 1

A sound sequence is acceptance and reservation, release, exact picks and shortage, offer, customer approval, replacement pick, refrigerated staging, readiness, early check-in, physical arrival, retrieval, verification, handoff, and settlement.
Link every event to the order; link the offer and final actual SKU to the original line.
Retain the refrigerated tote location as well as any ambient locations.
The store manager owns physical staging and a handoff exception; the customer or authorized policy provides substitution permission.
Finance and order systems reconcile the final charge and any refund.
The difference between physical arrival and handoff measures actual customer wait when both events are observed.
Check-in-to-handoff is a different measure when check-in preceded arrival.

### Exercise 2

Exact fulfillment is `460 / 500 = 92%`.
Authorized fulfillment is `(460 + 25) / 500 = 97%`.
Unfilled is `15 / 500 = 3%`.
Offer rate among the non-exact eligible lines is `30 / 40 = 75%`.
Acceptance among resolved offers is `25 / (25 + 3) = 89.29%`.
Unanswered share is `2 / 30 = 6.67%` of offers.
Acceptance among all offers is `25 / 30 = 83.33%`; it is a different denominator and should be labeled accordingly.
Line counts do not indicate how the non-exact lines are distributed across orders.

### Exercise 3

The completed waits sum to 36 minutes, so the completed-case mean is `36 / 6 = 6 minutes`.
The median is `(3 + 4) / 2 = 3.5 minutes`.
Nearest-rank p90 uses rank `ceil(0.90 × 6) = 6`, so it is 16 minutes.
The two open cases have already waited 10 and 12 minutes, but their final durations are unknown.
They must appear as open, censored observations, not as completed 10- and 12-minute waits.
A physical-arrival timestamp, with a reliable handoff timestamp, is needed for arrival-to-handoff time.
The sample is too small to support a stable tail estimate.

### Exercise 4

If accepted, verify the alternative still exists, record the approval, pick it, and apply the agreed price rule.
If declined, short the original line and issue the applicable refund or adjustment.
If unanswered under “ask first,” do not substitute; follow the stated short-and-refund or escalation policy before the deadline.
If the accepted alternative is unavailable, do not pretend it was supplied; offer another option if time and policy permit or short the line.
A previously recorded specific authorization can permit a replacement only within its stated scope and applicable rules.
The rubric requires a named owner, evidence, customer notification, and a final financial state for every branch.

### Exercise 5

Picking is `200 × 1.5 = 300 productive minutes`.
Staging is `80 × 2 = 160 productive minutes`.
Handoff is `80 × 3 = 240 productive minutes`.
Including 40 exception minutes, total productive time is `740 minutes`.
Paid time is `740 / 60 / 0.75 = 16.4444 hours`.
Labor cost is `16.4444 × $24.00 = $394.67 per day` after rounding.
A schedule also needs interval demand, simultaneous coverage, breaks, and feasible shift lengths.
A daily total cannot guarantee any particular peak-hour service level.

### Exercise 6

Offered load is `45 / 20 = 2.25` associate-equivalents.
Two associates provide `2 × 20 = 40 handoffs per hour` of nominal capacity.
Utilization is `45 / 40 = 112.5%` over the sustained hour.
A stable steady-state queue does not exist under these assumptions, so do not quote a finite steady-state waiting time.
Add capacity, reduce accepted arrivals, redistribute appointments with customer agreement, or change the process safely.
With three associates, nominal utilization becomes `45 / 60 = 75%`, but burstiness, service-time variation, and other duties can still produce long waits.
Observe the actual queue or model those effects before promising a wait.

### Exercise 7

New-order contribution is `70 × $12.00 = $840.00 per day`.
Shifted-order effect is `130 × -$2.00 = -$260.00 per day`.
After pickup cost, incremental contribution is `$840.00 - $260.00 - $650.00 = -$70.00 per day`.
On these point estimates, expansion reduces contribution; investigate improvements or stop rather than claim success from 200 orders.
The uncertainty in the incremental-order estimate or in avoidable labor could change the result.
A complete decision would include uncertainty and any material service or strategic effects stated separately.

### Exercise 8

A strong answer says successful scans exclude missing products, wrong-container outcomes, and orders never scanned.
It defines exact-line fulfillment over all eligible requested lines, independent audit accuracy over sampled eligible lines, and readiness against original promised deadlines.
It defines completed wait with a stated start and end, a percentile with its method, and open-wait age plus timestamp coverage.
It reports cancellation and missingness counts rather than silently excluding them.
The owner of operational definitions, the owner of data pipelines, and the owner of corrective action are named.
A weak answer merely adds more charts without changing denominators or addressing selection bias.

### Exercise 9

A strong plan randomizes comparable stores where feasible and specifies how nearby-store spillovers will be handled.
It measures incremental contribution against controls, including the dedicated associate's incremental paid hours and any displaced work.
It defines readiness, exact fulfillment, wait distribution, safety incidents, and accessibility complaints as guardrails.
It records the assignment unit, pre-period, analysis horizon, sample size rationale, and uncertainty interval before rollout.
It includes open orders and reports missing event coverage; a stop rule pauses the change after a serious safety event or predefined sustained service deterioration.
Store operations owns staffing and safety; finance owns economic validation; a named sponsor decides scale, modification, or stop.
A weak plan compares only treated-store sales before and after a promotion and declares causality.

## 14. Interview questions

1. **Why can a store meet a line-level accuracy target while customers are unhappy?**
   A strong answer distinguishes line and order outcomes, unauthorized substitutes, long waits, and selection bias.
   A weak answer assumes every correct scan implies a successful customer journey.
2. **How would you define a pickup wait?**
   A strong answer specifies start, end, cohort, early check-in, missing events, and open cases.
   A weak answer says “the average in the app” without naming timestamps.
3. **How would you handle a missing item when the customer does not respond?**
   A strong answer uses the actual consent policy, product restrictions, a deadline, and a documented short or escalation.
   A weak answer assumes silence authorizes any replacement.
4. **What does 80% utilization mean for curbside service?**
   A strong answer states the workload and capacity units and explains why variation still creates queues.
   A weak answer assumes 20% nominal spare capacity guarantees no waiting.
5. **How do you estimate staff hours from an order forecast?**
   A strong answer separates line and order drivers, productive standards, paid-time allowance, and peak coverage.
   A weak answer multiplies daily orders by a single unvalidated number and calls it a roster.
6. **What evidence would support a claim that pickup creates incremental demand?**
   A strong answer identifies a counterfactual, controls, spillovers, and uncertainty.
   A weak answer treats all pickup sales as new sales.
7. **Why preserve the original promise after a revision?**
   A strong answer explains customer commitment, auditability, and selection of misses.
   A weak answer overwrites it with the latest estimate and reports only the revised target.
8. **How would you distinguish an inventory problem from a labor problem?**
   A strong answer traces availability, search, queue, and readiness events by time and product.
   A weak answer blames slow pickers solely from a late-order count.
9. **What is wrong with using completed waits only on a live dashboard?**
   A strong answer identifies right-censoring and shows open wait ages and coverage.
   A weak answer assumes unfinished customers are irrelevant until handoff.
10. **What would cause you to pause curbside service?**
    A strong answer names a safety or capacity threshold, accountable authority, customer communication, and recovery path.
    A weak answer continues accepting orders solely to protect revenue.
11. **How would you audit substitution quality?**
    A strong answer samples original requests, offers, responses, physical goods, and final charges.
    A weak answer reports only the number of replacement items scanned.
12. **What makes an operating cost relevant to an expansion decision?**
    A strong answer asks whether it changes with the decision and prevents double counting against channel contribution.
    A weak answer allocates all historical corporate overhead without examining avoidability.

## 15. Key terms

- [Arrival](#4-definitions-and-vocabulary): customer presence at the pickup point.
- [Available-to-promise](02_available-to-promise.md): inventory quantity that can be committed under the applicable rules.
- [BOPIS](#4-definitions-and-vocabulary): buy online, pick up in store.
- [Capacity](#4-definitions-and-vocabulary): sustainable throughput under specified resources and constraints.
- [Cannibalization](#4-definitions-and-vocabulary): movement of demand from an existing channel.
- [Check-in](#4-definitions-and-vocabulary): recorded request for pickup service.
- [Cold chain](#4-definitions-and-vocabulary): temperature-controlled handling of sensitive goods.
- [Commitment](#3-the-core-mental-model): a promise actually shown to a customer.
- [Curbside](#4-definitions-and-vocabulary): handoff at an outside pickup point.
- [Event time](#4-definitions-and-vocabulary): when an action occurred, distinct from ingestion.
- [Exact-line rate](#6-worked-example): exact requested lines divided by eligible requested lines.
- [Exception](#4-definitions-and-vocabulary): a deviation requiring operational resolution.
- [Handoff](#4-definitions-and-vocabulary): release to an authorized recipient.
- [Incremental contribution](#6-worked-example): contribution change caused by the offer after relevant costs.
- [Inventory visibility](01_inventory-visibility.md): information about physical and recorded stock.
- [Nearest-rank percentile](#7-code-query-or-calculation): ordered value at rank `ceil(p × n)`.
- [No-response rate](#4-definitions-and-vocabulary): unanswered offers divided by offers sent.
- [Order completeness](#4-definitions-and-vocabulary): share of orders satisfying a defined complete-order rule.
- [Pick accuracy](#4-definitions-and-vocabulary): correct picks divided by audited eligible items.
- [Productive time](#4-definitions-and-vocabulary): measured time on specified order work.
- [Promise accuracy](06_promise-accuracy.md): performance against the promise actually shown.
- [Queue](#step-7--recognize-arrival-and-manage-the-queue): customers awaiting service under a stated priority rule.
- [Ready](#4-definitions-and-vocabulary): staged and genuinely available for release.
- [Reservation](#step-2--reserve-inventory-and-workload-capacity): allocation of stock or capacity to a commitment.
- [Right-censoring](#8-data-standards-and-regulatory-context): final duration not yet observed at cutoff.
- [Service recovery](#4-definitions-and-vocabulary): corrective action following a failure.
- [Short](#4-definitions-and-vocabulary): requested quantity not supplied.
- [Staging](#step-6--stage-and-mark-ready): controlled holding before handoff.
- [Substitution](#step-5--resolve-a-missing-or-unsuitable-item): authorized alternative for a requested item.
- [Utilization](#7-code-query-or-calculation): offered work divided by service capacity.
- [Wait](#4-definitions-and-vocabulary): elapsed time between explicitly specified journey events.

## 16. Sources

Sources are public starting points; access dates and local applicability should be checked for an operational implementation.
Company help pages describe current customer-facing features, not audited operational outcomes.

1. GS1, [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications). Identification keys and barcode application rules.
2. US Food and Drug Administration, [FSMA Final Rule on Requirements for Additional Traceability Records for Certain Foods](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods). Scope, exemptions, and current implementation guidance.
3. California Privacy Protection Agency, [Regulations](https://cppa.ca.gov/regulations/). Current California privacy regulatory materials.
4. US Department of Justice, [ADA.gov](https://www.ada.gov/). Federal accessibility guidance and resources.
5. US Occupational Safety and Health Administration, [Retail Industry](https://www.osha.gov/retail). Retail workplace safety resources.
6. Target, [Order Pickup and Drive Up](https://help.target.com/help/subcategoryarticle?childcat=Order+Pickup+%26+Drive+Up&parentcat=Fulfillment+Services). Company customer-service information; procedures may change.
7. Walmart Business, [Substitutions for Store Pickup and Delivery Items](https://business.walmart.com/help/article/substitutions-for-store-pickup-and-delivery-items/b94b3c9413b14315acd003296238cdaa). Company customer-service information; eligibility and terms may change.
8. IKEA US, [Click and Collect](https://www.ikea.com/us/en/customer-service/services/click-collect/). Company description of the collection service.
9. Little, J. D. C. (1961), “A Proof for the Queuing Formula: L = λW,” *Operations Research*, 9(3), 383–387. [DOI: 10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383). Foundational queue relationship; it does not validate the example's arrival or service assumptions.

## 17. Links

- [Module 23 topic index](_index.md)
- [Module 23 survey and its lab sections](../../../course/23_omnichannel_fulfillment_returns.md)
- [Inventory visibility](01_inventory-visibility.md)
- [Available-to-promise](02_available-to-promise.md)
- [Order routing and distributed order management](03_order-routing-dom.md)
- [Ship-from-store economics](04_ship-from-store.md)
- [Promise accuracy](06_promise-accuracy.md)
- [Returns networks](07_returns-networks.md)
- [Channel P&L](10_channel-pnl.md)
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md)

Sibling topic links follow the module index and may be unavailable until their planned files are produced.
