# Returns and Reverse Logistics

> Module 21 — E-commerce Supply Chain Analytics  
> Topic 07 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 07, 10, and 12; basic probability and contribution accounting  
> Version 2026.5

## 1. Why this topic exists

A customer buys an item online without trying, touching, or testing it first.
After delivery, the item may not fit, may differ from the description, or may arrive damaged.
The customer wants a simple remedy.
The retailer must decide whether and how to accept the item, issue a refund, recover value, and prevent the next avoidable return.
Those decisions connect customer experience, inventory, product quality, transport, finance, and legal obligations.

A returns manager may be asked to reduce the reported return rate.
A warehouse manager may be asked to clear a growing inspection queue.
A finance director may need a credible reserve and contribution forecast.
A merchant may need to know whether a new size guide is worth funding.
All four need a consistent unit of analysis and a view of the full loop.

An apparently low return rate can merely mean that recent orders have not had time to come back.
A low cost per parcel can conceal expensive refund leakage, a lost resale window, or a damaged customer relationship.
A high return rate can reflect an assortment mix or customer policy, not a process failure.
A fast refund can improve trust while making fraud controls harder if the evidence is poorly designed.
No single metric resolves these trade-offs.

This topic follows an item from delivery through return initiation, transport, inspection, disposition, and financial settlement.
It separates a customer remedy from what happens physically to an item.
A customer may receive a refund without shipping the item back, and an item may be received before a refund is approved.
Inventory recovery is not the same event as financial settlement.

The business cases and all amounts in the worked example are illustrative, in U.S. dollars, excluding sales tax.
They are not estimates of any named company's performance.
The approach is educational, not legal, tax, or accounting advice.
Consult the applicable specialists before changing a consumer policy or a reported financial estimate.

## 2. Learning outcomes

After this topic you can:

1. **Define** unit, order, revenue, and parcel return rates and select compatible numerators and denominators.
   Exercise 1 checks grain and event definitions.
2. **Calculate** a mature cohort rate and explain why an immature cohort needs adjustment.
   Exercise 2 checks age and observation windows.
3. **Map** the reverse flow and assign owners to its decisions, exceptions, and evidence.
   Exercise 3 asks for a process and ownership map.
4. **Evaluate** inspection and disposition using incremental value, capacity, and safety constraints.
   Exercise 4 tests the disposition calculation.
5. **Reconcile** a return to contribution without double counting inventory or refunds.
   Exercise 5 calculates economic loss and recovery.
6. **Compare** a prevention intervention with its cost and causal uncertainty.
   Exercise 6 develops an experiment and a business case.
7. **Design** a customer policy and a control plan that respect the relevant legal and operational constraints.
   Exercise 7 requires a policy trade-off and rollout decision.
8. **Forecast** reverse workload using cohort volume, initiation timing, transit, and processing delays.
   Exercise 8 checks a small capacity calculation.

Success means presenting an explicit decision, its evidence, its owner, and the conditions under which the decision changes.
A lower return rate is not automatically the best outcome.
A credible recommendation also protects customers, meets mandatory obligations, and accounts for net contribution.

## 3. The core mental model

### 3.1 A closed loop with separate clocks

```text
Sale -> fulfillment -> delivery -> eligibility clock -> customer decision
                                                     |
                               keep                  | return request
                                 |                   v
                                 |            authorization / remedy
                                 |                   |
                                 |          drop-off / collection / no-return remedy
                                 |                   |
                                 |          transport -> receipt -> inspection
                                 |                                  |
                                 |               +------------------+------------------+
                                 |               |                  |                  |
                                 |            restock          repair/resale       recycle/dispose
                                 |               |                  |                  |
                                 +---------------+------------------+------------------+
                                                 |
                           settled refund, cost, value recovery, customer outcome
                                                 |
                                  reason and root cause -> prevention
```

The horizontal flow is a sequence of physical and information events, not a guarantee that every item follows every step.
The refund may precede receipt, follow inspection, or be issued without a return.
A package can contain multiple units from multiple order lines.
One customer request can be split across multiple shipments.
A returned unit can be inspected twice but should not be counted twice as received.

The clock used for eligibility is specified in policy and applicable law.
The clock used for analysis must also be specified: order date, ship date, or delivery date.
For a post-delivery return rate, delivery date is usually an interpretable starting point.
A return initiated 20 days after delivery and received 29 days after delivery is an initiation at age 20, not 29.

### 3.2 What management is deciding

A good returns system makes several linked decisions.
It tells the customer what remedies and deadlines are available.
It selects a reverse channel and grants an authorization when required.
It determines when evidence is sufficient for a refund.
It routes the physical item to a location that can process it safely.
It chooses a disposition and makes inventory available only after the required controls.
It attributes avoidable reasons to a responsible process and funds corrective action.

These decisions have different owners.
Product and customer experience own the design of the customer journey.
Legal and compliance interpret mandatory rules and approve policy wording.
Returns operations own physical execution and queue management.
Merchandising and product quality own assortment and supplier actions.
Finance owns the economic definitions, estimates, and reconciliation.
Data owners ensure that each event can be linked and audited.
Risk teams design proportionate abuse controls and review appeals.

### 3.3 Four connected views

**Customer view:** time and effort from request to remedy, clarity, and fairness.
A speed improvement has value only if the remedy is correct and accessible.
Report outcomes for customers who needed assistance, not only portal users.

**Unit view:** whether an identified delivered unit was requested, received, inspected, and recovered.
The denominator for a unit rate is delivered units, not return parcels.
The same unit must not become multiple returns because of duplicate event messages.

**Network view:** where returns arrive, how long they wait, and whether processing capacity matches the expected flow.
A low transport rate can be expensive if it delays resale until a season ends.

**Economic view:** sale proceeds retained or refunded, product cost, reverse cost, recovered value, and any incremental customer or policy cost.
Use consistent accounting and economic boundaries.
Cash timing, recognized expense, and decision-relevant incremental cost need not coincide.

The closed loop matters because a return-reason dashboard without corrective action only reports yesterday's problem.
Likewise, a policy change without an estimate of later cohorts can make the current month look better while creating delayed claims.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Order | Customer purchase transaction; one order can contain several lines and units. | Assuming one order equals one item or one parcel. |
| Order line | An item, variant, price, and quantity within an order. | Treating a line with quantity three as one physical unit. |
| Delivered unit | One physical unit recorded as delivered under the agreed delivery definition. | Counting shipped, canceled, or undelivered units in a delivery cohort. |
| Delivery cohort | Units delivered in a specified calendar period, grouped by delivery date. | Comparing cohorts at different ages without adjustment. |
| Event time | When the event occurred, in a stated time zone. | Using ingestion time as the customer event time. |
| Observation cutoff | Latest event time included in a snapshot. | Treating late-arriving records as evidence that no return happened. |
| Age | Elapsed time from cohort anchor to an event or cutoff, in days. | Using transit time as time since delivery. |
| Eligibility window | Policy or statutory time interval for a customer to exercise a specified right. | Assuming one window applies to every country, product, and remedy. |
| Initiation | Customer request or other defined start of a return. | Treating initiation as proof of receipt or refund. |
| Return authorization | A recorded permission or reference for a defined return request. | Assuming an authorization guarantees a physical arrival. |
| Return parcel | A physical reverse shipment with a carrier tracking identifier. | Equating parcel count with item count. |
| Receipt | Physical arrival of an identified item at the receiving node. | Calling an inbound carrier scan a warehouse receipt. |
| Inspection | Recorded assessment against product, safety, and condition rules. | Automatically treating unopened packaging as proof of saleability. |
| Disposition | Authorized final or interim route for an item, such as restock, repair, or recycle. | Treating a proposed route as a completed physical action. |
| Returnless remedy | Refund, replacement, or other remedy without requiring physical return. | Counting its missing inbound scan as a lost parcel. |
| Exchange | A remedy that replaces an item under specified pricing and availability rules. | Ignoring its outbound shipping, inventory, and financial flows. |
| Refund | Money or other value returned to a customer for a transaction. | Treating refund cash as identical to total economic loss. |
| Gross unit return rate | Distinct delivered units with a defined return event divided by eligible delivered units. | Including exchanges, cancellations, or returnless remedies inconsistently. |
| Order incidence | Orders with at least one qualifying return divided by eligible delivered orders. | Interpreting it as the fraction of units returned. |
| Value return rate | Refunded or returned merchandise value divided by a compatible sales-value base. | Dividing this month's refunds by this month's sales and calling it a cohort rate. |
| Reason code | Customer-selected or agent-selected reason for a request, from a controlled list. | Assuming the customer's selection proves root cause. |
| Root cause | Evidence-supported underlying cause of a failure. | Equating the first available reason code with the verified cause. |
| Resaleable yield | Restocked, saleable units divided by received units or another explicitly stated base. | Comparing yields with different exclusions. |
| Recovery proceeds | Incremental economic proceeds or avoided purchase cost from the chosen disposition. | Adding gross resale revenue while also counting the full saved product cost. |
| Net recovery | Recovery proceeds less incremental costs of realizing them. | Forgetting refurbishment, selling fees, or incremental freight. |
| Reverse cost | Incremental cost of receiving and handling a return, according to a stated boundary. | Including original cost twice or omitting outsourced fees. |
| Contribution | Revenue less costs included in a specified decision boundary. | Comparing versions with different allocations or time horizons. |
| Maturity | Extent to which the relevant return event window has elapsed and data are complete. | Calling a cohort mature solely because its calendar month ended. |
| Right censoring | Incomplete observation of future events for a unit at the snapshot cutoff. | Assuming no observed return means no future return. |
| Lag distribution | Probability distribution of days between two defined events. | Mixing initiation, parcel receipt, and refund lags. |
| Backlog | Work that has entered but not completed a defined processing stage. | Measuring an authorization backlog as if it were a physical inspection queue. |
| Cycle time | Elapsed or working time between named events, in stated units. | Mixing medians with tail percentiles or excluding stalled cases. |
| Salvage value | Proceeds from a lower-value route, net or gross as specified. | Treating an offer as realized proceeds. |
| Disposition constraint | A safety, legal, contractual, inventory, or capacity rule limiting routes. | Choosing the highest spreadsheet value for an ineligible item. |
| Reserve | A financial estimate or provision for returns under the entity's accounting framework. | Treating the operational forecast as a journal entry without accounting review. |
| Abuse | Behavior that violates an applicable policy or law, as established by evidence. | Classifying every frequent returner or accessibility need as fraud. |

## 5. The mechanics, step by step

### 5.1 Frame the decision and the population

**Input:** decision brief, policy version, transaction scope, and reporting cutoff.
**Action:** name the population and outcome before retrieving data.
**Output:** a one-page metric and decision contract.

Specify whether the decision concerns a product, supplier, channel, country, node, or full portfolio.
Specify whether the question is operational capacity, customer policy, financial exposure, or prevention.
A return-cost forecast and an experiment may use the same events but different comparison populations.
The operations team needs future arrivals by node and day.
The merchant may need incremental contribution per sold unit over a season.

Write down the inclusion rules for delivered units.
State whether replacements, gifts, samples, canceled units, and marketplace sales are included.
State the treatment of returnless remedies, warranty claims, chargebacks, and failed delivery.
They may matter commercially but should not be silently mixed with customer-initiated post-delivery physical returns.

Specify the event defining the numerator.
Initiated returns answer a customer-behavior question.
Received returns answer an inbound workload question.
Completed refunds answer a cash or settlement question.
All are useful, but the rates differ and must not share an unlabeled chart.

Choose the denominator before slicing.
For a unit rate, use eligible delivered units and count distinct returned units at most once.
For order incidence, use delivered orders and count an order once even when several lines return.
For revenue rate, define whether the numerator is merchandise value requested, approved, or refunded.
Use the corresponding original merchandise value and a consistent discount and tax convention.

### 5.2 Build the event ledger

**Input:** order lines, delivery scans, requests, return shipments, inspection outcomes, settlements, and inventory movements.
**Action:** link events to a stable delivered-unit or line-quantity identity.
**Output:** a reproducible as-of ledger with source lineage and exceptions.

The ideal record includes order ID, line ID, unit ID or allocatable quantity, product variant, seller, customer market, node, and delivery time.
Add request ID, request time, reason code and version, parcel ID, carrier scans, receipt time, inspection outcome, and disposition completion time.
Link refunds, replacements, fees, and inventory movements to the same business object.
Keep original currency and event timestamp alongside converted amount and reporting currency.

Where no physical unit identifier exists, use a line-level quantity ledger.
Record each claim and receipt quantity with a stable event ID.
Apply deterministic allocation rules when multiple returns reference a line.
Flag quantities exceeding delivered quantity and route them to review rather than clipping them silently.
A physical package does not uniquely identify its contents.
A customer can send items from two orders in one parcel or split one line across two parcels.

Preserve events as append-only facts where possible.
A corrected refund or reversed receipt should be represented by a compensating event or a versioned correction.
Retain source system, ingestion time, event time, and snapshot cutoff.
Deduplicate repeated messages by source event identity, not simply by identical timestamp and amount.
A legitimate second refund can have the same amount as the first.

Reconcile the ledger to order management, carrier, warehouse, payment, and general ledger totals.
A reconciliation should report counts, quantities, and amounts by source and time period.
Document expected timing differences, open exceptions, and the owner and age of each exception.
Do not force a warehouse-receipt count to equal a refund count: their clocks and populations differ.

### 5.3 Measure rates at compatible grain

**Input:** delivered-unit ledger and qualifying event definition.
**Action:** form fixed delivery cohorts and count distinct qualifying units within the specified horizon.
**Output:** a labeled cohort rate and completeness assessment.

For cohort \(c\) and initiation horizon \(H\) days:

\[
R_c(H) = \frac{\text{distinct delivered units in }c\text{ with a qualifying initiation by age }H}
{\text{eligible units delivered in }c}.
\]

The denominator contains the whole eligible delivery cohort, including units with no return.
The numerator contains at most one qualifying initiation per unit.
Use an inclusive or exclusive boundary consistently; this lesson uses age from zero through day \(H\), inclusive.
If the customer is entitled to initiate at any time on the final local calendar day, compute age under that policy rather than substituting a 24-hour timestamp rule.

For an order incidence rate, identify orders with at least one qualifying unit return and divide by eligible orders.
A 20% unit rate does not imply that 20% of orders have a return.
For value, measure the returned merchandise amount and the delivered merchandise amount on a compatible cohort and allocation basis.
A high-value item can dominate a value rate even when unit incidence is low.

A calendar ratio is different:

\[
\text{calendar ratio in March} = \frac{\text{units received in March}}{\text{units delivered in March}}.
\]

It is useful for describing flow balance, but March receipts can come from February deliveries.
It is not the probability that a March delivery will be returned.
It can even exceed 100% when a large earlier cohort returns during a small current sales period.
Label it as a flow ratio and do not use it as a mature cohort outcome.

### 5.4 Handle maturity, censoring, and lag

**Input:** cohort dates, event times, cutoff, policy window, and historical lag curves.
**Action:** separate fully observed cohorts from partially observed cohorts.
**Output:** mature actual rates and explicitly modeled immature-cohort estimates.

A cohort is mature for an initiation metric only after every included delivery has passed the chosen horizon and the data-latency allowance has elapsed.
A calendar month cohort therefore matures relative to its last delivery, not its first.
Receipt-based rates need an additional transit and receiving lag, and very late or lost parcels require an explicit tail rule.
Extensions, holiday policies, statutory remedies, and country differences can make a single horizon inappropriate.

If the delivery cohort has an observed 30-day return rate but only 15 days of follow-up for its newest units, that is not a 30-day actual.
Show the observed-to-date rate and the age distribution.
For an early estimate, compare each unit or daily cohort to a historical cumulative initiation curve at the same age.
A simple nowcast divides observed returns by the historical fraction of eventual returns expected by that age.
The estimate depends on stable lag and comparable product, season, channel, and policy mix.

For example, 72 observed initiations at day 10 with an estimated 60% of final initiations observed by day 10 imply 120 expected eventual initiations.
If 1,000 units were delivered and have the same follow-up age, the point estimate is 12%.
It is not 12% observed, and the historical 60% can be wrong.
Use uncertainty intervals or scenarios, especially when only a few mature cohorts exist.
Do not divide an aggregate by one lag fraction if its members have substantially different ages.

For variable follow-up or changing risk over time, estimate an age-specific event curve with a survival or competing-risk method.
The event definition and censoring assumptions still matter.
If customers can return a unit for distinct competing remedies, specify whether those events preclude or alter the event of interest.
A returnless refund and a physical return are not necessarily independent events.

Test historical curves by pretending older mature cohorts were observed at earlier cutoffs.
Compare the early estimate with the subsequently observed final rate.
Stratify or recalibrate if timing changes after a policy, channel, promotion, or carrier change.
Data that arrive late can imitate a behavioral lag shift, so inspect ingestion delay separately.

### 5.5 Diagnose reasons and root causes

**Input:** customer reasons, quality findings, product attributes, delivery conditions, and contact records.
**Action:** distinguish reported reason from verified cause and prioritize a manageable intervention.
**Output:** a ranked cause hypothesis with accountable owner and evidence grade.

Design reason codes that customers understand and that map to actions.
Examples include fit too small, fit too large, item damaged, wrong item, not as described, changed mind, and other.
Keep a separate inspection reason and a separate root-cause field.
A customer may select "damaged" because the package was dented, while inspection finds an intact product.
Another customer may select "changed mind" because it is the easiest portal option.

Track missing, other, changed, and conflicting codes by channel and agent.
Do not train an automated prevention model on a reason field without examining how the field was collected.
A default option, a policy that treats defects differently, or a language mismatch can distort the distribution.
Use a sample of photos, contacts, physical inspections, and supplier evidence to validate coding.
Handle sensitive images and customer messages under the privacy and retention policy.

Compare rates within an appropriate product and customer context.
A shoe category can have a higher fit-return rate than consumables even with a well-functioning process.
A supplier can look worse because it sells harder-to-fit sizes or recently launched products.
Standardize for observed mix or use matched comparisons, and report sample sizes and uncertainty.
Never present an adjusted observational association as proof that a supplier caused the difference.

Rank opportunities by addressable volume times economic loss per avoidable return, adjusted for feasibility.
Protect safety and legal duties even when direct economics appear unfavorable.
Examples of controllable causes include wrong variant picking, misleading dimensions, a supplier defect, and carrier handling damage.
Examples of customer choice include ordering multiple sizes to compare at home; its remedy may be better sizing information rather than a punitive policy.

### 5.6 Design the customer policy

**Input:** applicable obligations, product restrictions, customer needs, historical behavior, and cost-to-serve.
**Action:** define eligibility, remedy, channel, timing, evidence, and exception handling.
**Output:** approved policy text, implementation rules, and operational service measures.

State the eligible products, relevant time window, starting event, geographic scope, and condition requirements.
Distinguish voluntary change-of-mind returns from defective, misdescribed, or otherwise legally protected remedies.
Specify who pays reverse postage, any disclosed fees, refund method, and treatment of promotional discounts.
Ensure policy text and checkout disclosures match the configured workflow.
Apply the policy version that governs each transaction; do not silently apply a newly restrictive rule retroactively.

Offer an accessible route for customers who cannot use a portal or printer.
Explain how to package, label, drop off, or arrange collection where relevant.
Provide a way to correct an incorrect reason or tracking reference and an escalation path for disputed decisions.
Support accessible communications and language needs in the markets served.

Choose refund timing deliberately.
Refund on carrier acceptance can reduce wait time but leaves exposure to empty or mismatched parcels.
Refund after receipt or inspection reduces some uncertainty but adds waiting and workload.
A risk-based process can vary the evidence requested while preserving applicable rights and a human appeal route.
Do not infer wrongdoing from disability, postcode, protected characteristics, or the mere fact of using a return right.

A returnless remedy can be appropriate when reverse handling costs exceed expected recoverable value, or transport creates disproportionate burden.
It requires clear product, safety, tax, and abuse controls.
The customer must not be instructed to dispose of hazardous or recalled goods in an unsafe way.
Document the authority and customer communication for each category.
A replacement should include its own outbound cost, stock requirement, and subsequent return exposure.

### 5.7 Plan reverse transport and receipt

**Input:** initiated-return forecast, geography, product characteristics, carrier contracts, and site capacity.
**Action:** choose collection, drop-off, consolidation, and receiving routes.
**Output:** a capacity and service plan by node and period.

Convert initiation forecasts into expected parcel handoffs using historical conversion from request to shipment.
Convert handoffs to receipts using an age distribution for transit and warehouse receiving.
Use a separate units-per-parcel distribution when planning inspection labor.
Account for returns never sent, canceled requests, mixed parcels, and repeat carrier scans.

Consolidation can lower line-haul cost while increasing delay and lost resale value.
Estimate both the transport saving and the value of earlier inventory availability.
For seasonal goods, a delay that crosses the markdown date may have a disproportionate cost.
For a low-value, non-seasonal item, a longer consolidated route may be economically sensible.

Choose nodes based on inspection capability, product safety, geographic demand, inventory location, and labor availability.
A local receipt node can verify contents before transferring items to a specialized repair center.
A specialist node may be mandatory for lithium batteries, medical goods, or recalled products.
Use documented handling rules and consult the relevant carrier and safety requirements.

At receipt, reconcile package identity, quantity, and condition to the authorization when one exists.
Provide a process for unannounced and mismatched returns rather than silently discarding them.
Quarantine unsafe, counterfeit-suspect, recalled, or unidentifiable items for an authorized decision.
A carrier's delivered scan is not a substitute for an item-level receiving event.

### 5.8 Inspect and choose disposition

**Input:** identified unit, condition evidence, allowed routes, expected proceeds, incremental costs, and capacity.
**Action:** select the highest feasible incremental value route subject to safety, legal, and service constraints.
**Output:** an auditable disposition decision and completed inventory or disposal event.

List feasible routes first: restock as new, open-box sale, repair and resale, return to vendor, donation, recycling, or compliant disposal.
Do not compare an infeasible new-sale route for an opened hygiene product with an eligible resale route.
The rules depend on product, condition, jurisdiction, supplier contract, and actual safety assessment.
Record who approved exceptions and the evidence used.

For each route, calculate expected incremental proceeds or avoided cost less incremental transport, inspection, refurbishment, selling, and disposal costs.
Use the same valuation basis and decision time for all options.
For example, a saleable returned item that substitutes for one replenishment unit may be valued at avoided landed purchase cost, adjusted for time and demand.
Alternatively, value the incremental eventual resale contribution, but include all future selling costs and do not also add the avoided cost.

Compare a repair route with a liquidation route using expected successful repair yield.
If repair costs $6 per attempted unit and produces $20 net proceeds only 60% of the time, with no proceeds otherwise, expected value is $6 per attempted unit.
If liquidation yields $7 net per attempted unit, choose liquidation on those assumptions unless constraints or broader effects change the decision.
A capacity bottleneck can make a slightly lower-value fast route preferable for some units.
The opportunity cost of constrained repair capacity belongs in the decision.

Restock only after required inspection and system status changes.
Separate physical putaway from available-to-promise inventory.
Prevent an item in quarantine from appearing as sellable stock.
Reconcile units received to units in each completed disposition, in process, or under investigation.

### 5.9 Reconcile the financial loop

**Input:** original transaction economics, refund, reverse costs, and realized or expected recovery.
**Action:** use a consistent baseline and count each economic effect once.
**Output:** contribution bridge per returned unit and per delivered unit.

One useful baseline is the contribution if a delivered item is kept.
A full refund removes retained merchandise revenue; the original product cost may still be sunk for the customer transaction.
If the returned item is recovered, it can reduce the loss by creating value in the selected recovery channel.
Subtract reverse freight, handling, and any nonrecoverable payment or packaging cost that the baseline has not already included.

The incremental loss relative to a kept sale is not simply refund plus product cost.
If the kept-sale baseline already includes the original product cost, adding that cost again to the incremental change double counts it.
Likewise, if net recovery already deducts inspection and freight, do not subtract those costs a second time.
A saleable unit's avoided purchase cost and its gross future selling price are alternative valuation approaches unless the full future transaction is modeled consistently.

For a cohort with heterogeneous returns, use the sum of unit-level incremental losses divided by eligible delivered units.
A simplified estimate is return probability times average incremental loss per returned unit.
That simplification assumes the conditional average applies to the population and that cost, disposition, and return probability are correctly paired.
The exercise may require separate segments when high-risk items have different recovery values.

For accounting reporting, consult the applicable framework, entity policy, and finance owner.
For example, the revenue recognition framework may require estimates of refund obligations and rights to recover products.
Do not assume a managerial contribution bridge is itself the prescribed financial statement treatment.

### 5.10 Forecast workload and expose uncertainty

**Input:** future delivery forecast, return probability by segment, lag curves, parcel conversion, and site routing.
**Action:** convolve deliveries with expected initiation and receipt lags.
**Output:** expected arrivals, workload, and range of plausible outcomes by site and day or week.

For delivery cohort \(c\), segment \(s\), and age \(a\), expected initiations equal delivered units times eventual return probability times the probability mass of initiation at age \(a\).
Apply a separate distribution for time from initiation to receipt and a probability that an initiated return is actually shipped.
If one parcel contains multiple units, translate units into parcels only with an appropriate consolidation assumption.
Do not use a unit lag curve as a parcel arrival curve without checking the mapping.

A deterministic expected workload is not a staffing commitment.
Use high-volume scenarios, variable transit, absenteeism, service targets, and existing backlog to assess capacity.
Forecast weekly or daily work by node according to when the item is expected to need processing.
A one-day transport delay can shift work across staffing periods without changing total returns.

Reconcile arrivals, completed inspections, and closing backlog:

\[
B_{t+1} = B_t + A_t - C_t,
\]

where \(B_t\) is opening units awaiting inspection, \(A_t\) are newly received units, and \(C_t\) are completed inspections in the period.
Completions cannot exceed opening backlog plus arrivals or available processing capacity.
Track aging as well as total backlog; a stable total can conceal very old units.

### 5.11 Prioritize prevention and measure causal impact

**Input:** reason and economics baseline, candidate intervention, eligible population, and implementation cost.
**Action:** design a comparison that estimates incremental outcomes, not just a rate change.
**Output:** a funded or rejected intervention with an owner, guardrails, and review date.

Possible interventions include correcting a product photograph, revising dimensions, improving a size guide, changing packaging, fixing a pick process, or holding a defective supplier lot.
Each should map to a hypothesized mechanism and a measurable intermediate outcome.
For a size guide, the mechanism might be better size selection; monitor guide use and fit-related returns.
For packaging, the mechanism might be less transit damage; monitor damage reports and confirmed inspection results.

Where feasible, randomize at a unit that avoids interference: customer, product page, store, or supplier lot.
Set the exposure date and freeze assignment rules before observing results.
Follow delivery cohorts through the full agreed return horizon.
Measure conversion, delivered units, return rate, net contribution, service contacts, and accessibility alongside the targeted reason.
A reduced return rate obtained by deterring legitimate claims is not necessarily a successful intervention.

Calculate benefit using the change in the number of returns per eligible delivered unit, not just the percentage change in a rate.
Apply the appropriate incremental loss per avoided return and subtract intervention costs.
If the intervention changes conversion or basket mix, include the incremental margin and fulfillment implications in a wider business case.
Avoid counting an avoided return and its separately listed component costs twice.

### 5.12 Govern decisions and learn

**Input:** operational dashboard, exceptions, experiment results, and financial reconciliation.
**Action:** set review cadence, assign decisions, and close the loop on interventions.
**Output:** a signed decision log with next review and rollback triggers.

Operations reviews arrivals, backlog, cycle time, and safety exceptions daily or weekly.
Customer experience reviews remedy timeliness, contacts, appeals, and accessibility.
Merchandising and quality review product and supplier cohorts on a cadence appropriate to mature data.
Finance reviews the contribution bridge, cash settlement, and forecasts with accounting.
Legal reviews changes to policy and jurisdiction-specific controls.

Each action has one accountable decision maker and named contributors.
A proposal should state the baseline, counterfactual, expected effect, uncertainty, implementation cost, and stopping rule.
Assign an owner to the actual remediation, not merely to the dashboard.
Record when the action was deployed and which customers or products were affected.

A post-implementation review checks whether measured outcomes match the hypothesis.
If they do not, distinguish execution failure, data failure, and an incorrect causal hypothesis.
Update estimates using mature cohorts rather than revising history without an audit trail.

## 6. Worked example

### 6.1 Decision brief

An illustrative retailer sells a jacket through its own online channel.
The merchant is deciding whether to fund a sizing guide before the next season.
Operations must also plan the reverse flow for the present cohort.
Finance wants an expected contribution per delivered unit, not just return postage.
Customer experience must preserve a clear return option.

All figures in this case are assumptions, not external claims.
The currency is U.S. dollars; amounts exclude tax and financing effects.
The unit of analysis is one delivered jacket.
The cohort contains 1,000 eligible jackets delivered on the same day.
The horizon for a qualifying initiation is 30 days after delivery, inclusive.
The observation cutoff is after day 30 plus a data-latency allowance.
All 1,000 units have complete follow-up for this initiation metric.

The retailer observes 180 distinct units with qualifying return initiations.
Of those, 162 are physically received and 18 receive a returnless refund under the hypothetical policy.
The 162 received units complete inspection.
There are no unshipped requests or unresolved cases in this simplified example.
All 180 returned units receive full merchandise refunds.

### 6.2 Rates and population

The initiated unit return rate is:

\[
180 / 1{,}000 = 18.0\%.
\]

The physical receipt rate per delivered unit is:

\[
162 / 1{,}000 = 16.2\%.
\]

The returnless remedy incidence is:

\[
18 / 1{,}000 = 1.8\%.
\]

The physical receipt share among initiated units is 162 / 180 = 90.0%.
That is a process mix for this simplified cohort, not a universal conversion assumption.
A 16.2% received rate would understate customer remedy demand if the intended metric included returnless refunds.
An 18.0% initiation rate cannot by itself predict receipt labor without the channel mix.

Suppose 40 returned jackets came from orders that each contained two returned jackets, and the remaining 140 came from different orders.
There are then 20 + 140 = 160 orders with at least one return.
If the 1,000 jackets came from 800 eligible orders, order incidence is 160 / 800 = 20.0%.
The order incidence is not the 18.0% unit rate.
These additional order quantities are illustrative and do not alter the unit-level financial bridge.

### 6.3 Original sale economics

Assume each jacket has the following economics if kept:

| Component | Amount per delivered unit |
|---|---:|
| Merchandise revenue retained | $80.00 |
| Original landed product cost | $32.00 |
| Original outbound fulfillment cost | $8.00 |
| Kept-sale contribution | $40.00 |

The kept-sale contribution is $80.00 − $32.00 − $8.00 = $40.00 per unit.
Outbound fulfillment cost is assumed nonrecoverable after delivery.
The original landed product cost is included once in the kept baseline.
Payment costs and other overhead are excluded from both alternatives for simplicity.
A live business case would include them if they differ between alternatives.

### 6.4 Physical disposition and recovery

The 162 received units have the following completed routes:

| Route | Units | Net recovery per unit | Total net recovery |
|---|---:|---:|---:|
| Restock saleable | 100 | $30.00 | $3,000.00 |
| Repair and resale | 40 | $12.00 | $480.00 |
| Liquidate | 22 | $4.00 | $88.00 |
| Total | 162 | — | $3,568.00 |

Net recovery is valued at the decision point after any route-specific refurbishment and selling costs.
For restock, $30.00 is the estimated avoided replacement cost after a timing adjustment.
It is not the $80.00 future retail selling price.
The repair and liquidation figures are net of their route-specific costs and reflect expected realization.
Common reverse freight and inspection are not included in the net recovery figures.
The 18 returnless-remedy units have zero physical recovery for the retailer.

The received-unit resaleable restock yield is 100 / 162 = 61.7%, rounded.
Including repair and resale, 140 / 162 = 86.4% are routed to a resale channel.
These are route shares, not proof of future sell-through for the repair items.
The physical receipt and disposition counts reconcile: 100 + 40 + 22 = 162 units.

### 6.5 Reverse costs and contribution bridge

The common reverse cost is $10.00 per physically received unit.
It includes reverse transport and inspection but excludes the route-specific costs already netted from recovery.
For 162 units, common reverse costs are 162 × $10.00 = $1,620.00.
The 18 returnless refunds incur no additional physical handling cost in this case.
Assume no refund processing fee or replacement shipment.

The total merchandise refunded is 180 × $80.00 = $14,400.00.
Relative to the all-kept baseline, the cohort's incremental return loss is:

\[
\$14{,}400 + \$1{,}620 - \$3{,}568 = \$12{,}452.
\]

The loss per initiated return is $12,452 / 180 = $69.18, rounded.
The loss per delivered unit is $12,452 / 1,000 = $12.452.
The baseline all-kept cohort contribution is 1,000 × $40.00 = $40,000.00.
After returns, the cohort contribution is $40,000.00 − $12,452.00 = $27,548.00.
Expected contribution per delivered unit under this observed cohort mix is $27.548.

A direct check reaches the same value.
Retained revenue is 820 × $80.00 = $65,600.00.
Subtract 1,000 × $32.00 = $32,000.00 in original product cost.
Subtract 1,000 × $8.00 = $8,000.00 in outbound cost.
Subtract $1,620.00 in common reverse cost and add $3,568.00 in recovery.
The result is $65,600 − $32,000 − $8,000 − $1,620 + $3,568 = $27,548.

Notice that the refund is not $80 plus the original $32 product cost in the incremental bridge.
The $32 was already incurred in the all-kept baseline.
The recovery value is an offset, and its route-specific costs must not be deducted again.
The direct check makes the baseline and sign convention visible.

### 6.6 A disposition choice

Consider one of the 40 repair candidates before it is routed.
An inspection has already been paid and is common to all feasible choices.
Repair is expected to yield $24.00 in proceeds if successful, with a 75% success probability.
The repair attempt costs $6.00 whether or not it succeeds.
On failure, assume no further proceeds or disposal cost.
The expected net recovery is 0.75 × $24.00 − $6.00 = $12.00.
A liquidation buyer offers $8.00 net at the same decision point.
Repair is expected to add $4.00 per candidate, assuming capacity and safety eligibility.

If success probability falls below 58.3%, liquidation becomes preferable on these assumptions.
The break-even calculation solves \(24p - 6 = 8\), so \(p = 14/24\).
If the repair shop is at capacity, the opportunity cost of displacing a higher-value repair may reverse the decision.
If an item fails a safety rule, the higher expected value does not authorize repair or resale.

### 6.7 Prevention business case

The proposed guide costs $9,000 once to develop and deploy.
Assume 100,000 comparable jackets will be delivered in the evaluation season.
The base case assumes the initiated rate falls from 18% to 16%, with no conversion or mix change.
This is a two-percentage-point absolute reduction, not a 2% relative reduction.
Expected avoided returns are 100,000 × 0.02 = 2,000 units.

If each avoided return saves the cohort's average incremental loss of $69.1778, gross benefit is approximately $138,356.
After the $9,000 guide cost, estimated incremental contribution is approximately $129,356.
This is a forecast, not a realized saving or a commitment.
The assumption that avoided fit returns have the same mix and loss as all returns is particularly material.
If avoided returns are mostly saleable jackets with higher recovery, the appropriate loss per avoided return may be lower.

At an assumed $50.00 loss per avoidable return, break-even avoided returns are $9,000 / $50 = 180 units.
For 100,000 delivered units, the break-even absolute rate reduction is 180 / 100,000 = 0.18 percentage points.
A 0.10-percentage-point improvement avoids 100 returns and yields $5,000 before guide cost: a $4,000 loss.
A 0.50-percentage-point improvement avoids 500 returns and yields $25,000 before cost: a $16,000 gain.
These calculations exclude changes in conversion, margin, ongoing guide expense, and customer behavior.

The merchant owns the funding decision after finance validates the boundary.
The product team owns the randomized guide implementation and customer accessibility.
Operations owns the evidence on reason and recovery mix.
The decision should wait for legal and customer-experience review and use mature-cohort results before claiming success.

## 7. Code, query, or calculation

The following standard-library Python 3 example reproduces the worked case's cohort and contribution arithmetic.
It uses synthetic aggregate records, not actual customer data.
All deliveries share a date, all return events are observed within 30 days, and all returned units are distinct.
The example validates the population and disposition reconciliation but is not a production ledger.
It deliberately does not estimate immature cohorts or prove the causal effect of the guide.
Copy it into a file such as `returns_case.py` and run `python returns_case.py`.

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class Case:
    delivered: int
    requested: int
    received: int
    returnless: int
    sale_price: float
    product_cost: float
    outbound_cost: float
    reverse_cost: float
    # Route name, unit count, net route recovery per unit.
    routes: tuple[tuple[str, int, float], ...]


def summarize(case: Case) -> dict[str, float]:
    if case.delivered <= 0:
        raise ValueError("delivered units must be positive")
    if not (0 <= case.requested <= case.delivered):
        raise ValueError("requested units must fit the delivered cohort")
    if min(case.received, case.returnless) < 0:
        raise ValueError("channel counts cannot be negative")
    if case.received + case.returnless != case.requested:
        raise ValueError("all requests must be received or returnless")
    if any(count < 0 for _, count, _ in case.routes):
        raise ValueError("route counts cannot be negative")
    if sum(count for _, count, _ in case.routes) != case.received:
        raise ValueError("routes must reconcile to received units")

    refunds = case.requested * case.sale_price
    reverse = case.received * case.reverse_cost
    recovery = sum(count * value for _, count, value in case.routes)
    all_kept = case.delivered * (
        case.sale_price - case.product_cost - case.outbound_cost
    )
    loss = refunds + reverse - recovery
    contribution = all_kept - loss
    return {
        "request_rate": case.requested / case.delivered,
        "receipt_rate": case.received / case.delivered,
        "refunds": refunds,
        "reverse": reverse,
        "recovery": recovery,
        "loss": loss,
        "contribution": contribution,
        "contribution_per_unit": contribution / case.delivered,
        "loss_per_return": loss / case.requested if case.requested else 0.0,
    }


def prevention_case(
    delivered: int,
    baseline_rate: float,
    treatment_rate: float,
    loss_per_avoided_return: float,
    intervention_cost: float,
) -> tuple[float, float]:
    avoided = delivered * (baseline_rate - treatment_rate)
    benefit = avoided * loss_per_avoided_return - intervention_cost
    return avoided, benefit


def main() -> None:
    case = Case(
        delivered=1000,
        requested=180,
        received=162,
        returnless=18,
        sale_price=80.0,
        product_cost=32.0,
        outbound_cost=8.0,
        reverse_cost=10.0,
        routes=(
            ("restock", 100, 30.0),
            ("repair", 40, 12.0),
            ("liquidate", 22, 4.0),
        ),
    )
    result = summarize(case)
    avoided, benefit = prevention_case(
        delivered=100_000,
        baseline_rate=0.18,
        treatment_rate=0.16,
        loss_per_avoided_return=result["loss_per_return"],
        intervention_cost=9_000.0,
    )
    print(f"Initiated rate: {result['request_rate']:.1%}")
    print(f"Received rate: {result['receipt_rate']:.1%}")
    print(f"Refunds: ${result['refunds']:,.2f}")
    print(f"Reverse cost: ${result['reverse']:,.2f}")
    print(f"Net recovery: ${result['recovery']:,.2f}")
    print(f"Incremental return loss: ${result['loss']:,.2f}")
    print(f"Cohort contribution: ${result['contribution']:,.2f}")
    print(f"Contribution per unit: ${result['contribution_per_unit']:.3f}")
    print(f"Avoided returns: {avoided:,.0f}")
    print(f"Guide benefit after cost: ${benefit:,.2f}")


if __name__ == "__main__":
    main()
```

Expected output, with standard rounding:

```text
Initiated rate: 18.0%
Received rate: 16.2%
Refunds: $14,400.00
Reverse cost: $1,620.00
Net recovery: $3,568.00
Incremental return loss: $12,452.00
Cohort contribution: $27,548.00
Contribution per unit: $27.548
Avoided returns: 2,000
Guide benefit after cost: $129,355.56
```

In a production implementation, use decimal currency arithmetic or integer minor units with an explicit rounding policy.
Join actual refunds to transactions instead of assuming every request produces a full refund.
Model partial refunds, replacements, cancellation, return shipping charges, and outstanding cases explicitly.
Use route-specific distributions rather than assuming each item receives an identical average recovery.
The code's rejection of unreconciled channels is intentional for this completed synthetic cohort; a live ledger needs an explicit open-case state.
If an observed treatment rate is higher than baseline, the prevention calculation correctly reports negative avoided returns and a negative benefit.
The result is still only as reliable as the causal estimate and the loss assigned to the affected items.

## 8. Data, standards, and regulatory context

### 8.1 Data contracts and identity

A return analysis depends on linkable records across commercial, physical, and financial systems.
Write a data contract for each source before building a dashboard.
Specify the record grain, primary key, source owner, event time, time zone, update cadence, correction mechanism, and retention period.
Specify which fields are optional and which failures must be escalated.
The following minimum contract separates facts that are frequently collapsed.

| Dataset | Required grain | Important fields | Primary control |
|---|---|---|---|
| Delivered sales | Order line or unit | Quantity, product, seller, price, delivery timestamp | Reconcile eligible quantities to order system. |
| Requests | Return request line | Request ID, quantity, initiation time, reason, policy version | Detect duplicate IDs and excess claimed quantity. |
| Shipments | Parcel event | Carrier, tracking ID, event, event time | Distinguish repeated scans from distinct packages. |
| Receipt | Received item or allocated quantity | Node, receipt time, identifier, condition | Reconcile received units to request and exceptions. |
| Inspection | Inspection decision | Rules version, result, inspector or system | Ensure quarantine and unsafe cases are not sellable. |
| Disposition | Completed movement | Route, time, quantity, destination | Reconcile received stock to open and completed states. |
| Settlement | Financial event | Refund ID, amount, currency, payment reference | Match to payment and general ledger where applicable. |
| Cost | Activity or rate period | Freight, labor, repair, fee and currency | Separate actuals, accruals, and assumptions. |
| Policy | Version and market | Eligibility, start and end, permitted remedies | Preserve the rule in force at transaction time. |
| Product | Variant and effective period | Category, size, supplier, hazards, restrictions | Version attributes and restricted routes. |

The data owner should preserve unmatched records in a controlled exception table.
A missing request link can be a customer service problem, an integration failure, or a legitimate unannounced return.
Deleting it makes the metric look cleaner but breaks operational reconciliation.
An unresolved difference should have an age, reason, and owner.

Delivery timestamps often arrive from carrier feeds and may be corrected after the fact.
Maintain both the original and corrected event, along with the time the correction became available.
A historical as-of report should use only information available at that cutoff.
A restated report can use later corrections if labeled and reconciled to the original.

Line quantities require particular care.
A customer can buy two identical units and return one.
A partial refund can apply to one unit, a shipping charge, or the entire order.
Allocate amounts under a documented rule and keep unallocated amounts visible.
If serial numbers are legally or operationally relevant, preserve their relationship to shipment and return without exposing them unnecessarily.

### 8.2 Common identifiers and event conventions

GS1 publishes identification and event-data standards used across many supply chains.
A Global Trade Item Number (GTIN) identifies a trade item; a serial identifier can distinguish individual instances when assigned.
A Serial Shipping Container Code (SSCC) identifies a logistics unit such as a parcel or pallet when used.
Electronic Product Code Information Services (EPCIS) defines a way to share event information about objects and their business context.
See the GS1 sources in section 16 for current definitions and specifications.

These identifiers and standards are conventions or contractual requirements unless a specific law or agreement makes them mandatory for a transaction.
A GTIN does not identify the customer order, prove delivery, or establish the condition of a returned item.
An SSCC identifies a logistics unit, not necessarily each item inside it.
An EPCIS event can record a business event but does not guarantee that the captured event is correct.
A data-quality and reconciliation process remains necessary.

A return label's carrier tracking number may link transport events but may be reused, missing, or unrelated to the order-line identity.
The system should store crosswalks between order, authorization, package, and unit or line quantities.
Make identifier scope and uniqueness explicit, including carrier and time period when needed.
Version mapping rules so that a later correction does not silently rewrite a previously reported cohort.

### 8.3 Customer rights and policy

As of November 2026, the European Union Consumer Rights Directive provides a general framework for withdrawal from many distance contracts, subject to conditions and exceptions.
The official directive text and amendments should be read with national implementing law and the relevant transaction facts.
The applicable starting point, remedies for faulty goods, exclusions, and cost allocation can differ by product and circumstance.
A retailer's voluntary policy must not be assumed to replace mandatory consumer rights.
Legal counsel should approve customer-facing text and the operational rules for each market.

In the United States, return and refund obligations depend on applicable federal and state law, product and transaction facts, disclosures, and the retailer's commitments.
The Federal Trade Commission provides consumer-facing guidance on returns and refunds.
Do not generalize one state rule to the whole country or assume that a posted policy can waive a mandatory remedy.
For a defective or recalled product, additional obligations may apply.

A merchant operating across borders must map seller, destination, contract, marketplace rules, and policy version.
A marketplace may administer a remedy while the merchant or seller bears part of the economic cost.
The contract and applicable law determine responsibility; a platform label alone does not.
Obtain local legal and tax review for duties, taxes, consumer remedies, and disposition.

### 8.4 Product, safety, and environment

A returned item can contain a battery, personal data, a regulated material, or a defect that makes transport or resale unsafe.
Review current carrier dangerous-goods requirements and local environmental, product-safety, waste, and recall rules.
A suspected recall item should be isolated and handled under the applicable recall instructions.
An electronic device may require verified data erasure before refurbishment or resale.
An item's commercial value is not permission to resell it.

A disposition dashboard should distinguish donation, recycling, energy recovery, and landfill where the evidence supports the distinction.
Use downstream documentation rather than assuming a recycler recovered a particular fraction.
Environmental comparisons require an explicit system boundary, transport mode, displacement assumption, and measurement method.
Avoid claiming that returnless refunds are always environmentally preferable: replacement purchases and customer disposal can change the result.

### 8.5 Privacy, controls, and security

Return records can contain customer identity, addresses, photos, health-related inferences, and payment references.
Collect only what is necessary for the stated purpose and define access and retention controls.
Keep payment credentials in approved payment systems rather than duplicating them in analytics tables.
Use role-based access, audit logs, encryption where appropriate, and a process for responding to data incidents.
Privacy obligations vary by jurisdiction and processing context; obtain qualified advice.

A fraud model is a decision-support tool with potential false positives.
Document the target, features, review threshold, evidence, appeal mechanism, and monitoring by relevant population.
Do not use an opaque score as the only basis for denying a legally required remedy.
Separate confirmed abuse from unusual but legitimate return behavior.
Monitor both false denials and missed abuse, and protect access to customer explanations.

## 9. Industry practice and case evidence

### 9.1 What public cases can and cannot establish

Public return-policy pages describe current customer-facing options, not necessarily the internal operating cost or causal effect of those options.
Company communications may be updated without preserving earlier versions.
A case should therefore identify the page, access period, and exact scope of the claim.
It should not invent return rates, cost savings, or experiment results.
The links in section 16 are starting points for checking current policy details.

### 9.2 Amazon: channel and item-specific options

Amazon's customer help material describes how eligible customers can start a return and receive available return options.
The available method can depend on the item, seller, and location.
The practical lesson is to model the actual offered channel at the request level rather than assigning every order to one reverse route.
A customer's return flow can also differ between products on the same order.

**[INFERENCE]** A retailer offering multiple channels needs a mapping from authorization to parcel and receipt, plus a way to compare cost and experience by offered and chosen option.
The public help page does not establish Amazon's cost per return, restock yield, or the effect of a specific channel on retention.
Any comparison with another retailer needs comparable country, item, and service definitions.

### 9.3 IKEA: customer policy and product eligibility

IKEA publishes a returns policy with conditions and processes that vary by market.
Its public U.S. policy is a concrete example of why a return policy needs eligibility, proof-of-purchase, and condition rules.
Check the live page and transaction date before applying its terms.
The policy itself is not evidence that every returned item is resold or that any particular disposition is optimal.

**[INFERENCE]** A retailer with large or assembled goods must plan collection, inspection, damage, and recovery differently from a parcel apparel merchant.
The policy page alone cannot quantify those differences.
A useful internal evaluation would compare total recovery and customer effort by product and channel.

### 9.4 Patagonia: repair and resale as distinct routes

Patagonia's Worn Wear program publicly describes repair and trade-in or resale options for eligible products.
The program illustrates a route other than immediate disposal or resale as new.
Eligibility, condition, credit, and channel terms are program-specific and may change.
A trade-in is not identical to a return for a refund, so it must not be included in a return rate without a deliberate definition.

**[INFERENCE]** A repair-capable network requires grading, parts, skilled capacity, quality assurance, and demand for repaired goods.
The public program page does not provide the contribution per repaired unit for this lesson's hypothetical jacket.
An analyst should obtain actual repair yield, turnaround time, incremental cost, and resale proceeds before selecting that route.

### 9.5 Evidence hierarchy and transfer

The sources in section 16 include standards bodies, regulators, a peer-reviewed research paper, and company program pages.
Regulator and standards sources are useful for definitions and primary requirements but do not give a universal business-case parameter.
A company policy page is evidence of the described policy at the relevant time, not independent proof of performance.
Research may establish an association or modeled mechanism in a particular setting, not an effect guaranteed in another assortment.

When transferring evidence, check product, market, customer population, time period, outcome definition, and counterfactual.
A fashion return study may not generalize to a bulky appliance.
An early-refund program may affect customers who choose that channel differently from customers who do not.
A cost per parcel does not transfer directly to a cost per returned unit.
Use local data and a controlled evaluation for a local investment decision.

### 9.6 Operations practices worth testing locally

A daily reverse backlog meeting can separate items awaiting receipt, inspection, repair, and final disposition.
It should identify blocked units, aging, and the owner of each exception.
It should not promote a unit to sellable status merely to improve a cycle-time metric.
A supplier quality review can compare verified defects against delivery cohorts and purchase lots.
It should include denominator and mix information, not just a count of complaints.

A customer-experience review can monitor refund time from request and from parcel handoff separately.
It should include contacts, appeals, and failed portal journeys.
A lower average time can hide a small population waiting much longer, so report a tail percentile and unresolved cases.
A finance review can compare forecast and actual recovery by disposition route.
It should distinguish expected recovery from realized proceeds and explain timing differences.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Immature cohort called final | Recent rate is systematically lower than older rates. | Later initiations are not yet observed. | False improvement and underplanned capacity. | Report age, mature cohorts, and validated lag nowcasts. |
| Calendar ratio called probability | Returns exceed sales in a quiet month. | Numerator and denominator come from different cohorts. | Misleading rate and policy decision. | Use delivery cohorts for incidence and label flow ratios. |
| Unit and order grains mixed | Order-level rate changes when basket size changes. | Multi-unit orders are counted inconsistently. | Wrong category comparison. | Publish separate order and unit definitions. |
| Duplicate scans inflate receipt | Several receipts share a source event ID. | Carrier or warehouse message is replayed. | Phantom workload and recovery. | Deduplicate by event identity and reconcile quantities. |
| Parcel count substitutes for units | Units per parcel changes after consolidation. | Mixed and split parcels break one-to-one mapping. | Staffing and rate errors. | Link package contents to line quantities. |
| Returnless remedies omitted | Refunds rise while physical receipts fall. | Metric only counts inbound units. | Understated customer remedy and financial exposure. | Report returnless and physical channels separately. |
| Reason code treated as fact | Portal code conflicts with inspection or contacts. | Customer selection is influenced by wording or incentives. | Wrong supplier or product action. | Retain source, validate sample, and separate verified cause. |
| Wrong valuation baseline | Savings exceed refund value plus all avoided costs. | Product cost or recovery is counted twice. | Inflated business case. | Use a kept-sale bridge and independent direct reconciliation. |
| Recovery recorded before realization | Forecast proceeds rise while cash and stock remain flat. | Proposed disposition is treated as completed. | Overstated contribution or available inventory. | Separate planned, completed, and realized states. |
| Inspection backlog hidden | Median cycle time stays low while old cases grow. | Unfinished units excluded from cycle-time calculation. | Lost resale window and customer complaints. | Report open-case age and tail service measures. |
| Fast refund without controls | Refunds cannot be matched to received goods. | Authorization and evidence are weak. | Leakage and disputes. | Use proportionate controls, reconciliation, and appeal. |
| Punitive policy optimizes one metric | Claims fall while contacts and complaints rise. | Legitimate customers face friction. | Customer harm and possible legal exposure. | Monitor remedy access and legal compliance as guardrails. |
| Policy change confounds comparison | New cohort differs in window, assortment, and price. | Several causes change simultaneously. | Invalid attribution. | Randomize where feasible and preserve policy versions. |
| Unsafe disposition | Quarantine units appear in available inventory. | Inventory status and physical controls are disconnected. | Safety incident and recall exposure. | Block sale until authorized release and audit movements. |
| Carrier scan treated as receipt | Tracking says delivered but no item is found. | Package-level event does not verify contents. | Premature closure and unresolved customer claims. | Record item receipt and investigate mismatches. |
| Site averages hide bottlenecks | Total capacity appears sufficient but one node ages rapidly. | Product skill and geography constrain routing. | Missed service and markdown exposure. | Forecast by site, skill, period, and queue. |
| Historical lag becomes stale | Nowcasts consistently underpredict later returns. | Policy, promotion, or customer mix changes timing. | Late labor and reserve corrections. | Backtest by cohort and monitor lag drift. |
| Fraud label substitutes for evidence | Frequent-return customers are automatically denied. | Model confuses behavior and misconduct. | Unfair treatment and legal risk. | Review evidence, permit appeal, and monitor false positives. |
| Cross-border costs omitted | Recovered items incur unexpected duty or tax. | Route and transaction responsibilities were not mapped. | Negative net recovery. | Include market-specific costs and specialist review. |
| Prevention evaluated on requests alone | Return rate falls while delivered volume or margin falls. | Friction or conversion loss changes the denominator. | Apparent improvement destroys value. | Measure randomized contribution and customer guardrails. |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Delivery identity is reliable. | It anchors cohort membership and elapsed age. | Missing or corrected scans, split deliveries. | Reconcile to carrier and customer evidence; retain corrections. |
| Qualifying events link to delivered units. | Distinct units are needed for a unit rate. | Mixed parcels, unannounced returns, duplicate requests. | Audit links and quantify unmatched and excess quantities. |
| Initiation horizon is appropriate. | It defines the comparable outcome. | Extensions, warranty claims, differing local rights. | Segment by policy and market; report later events separately. |
| Data latency is understood. | Mature observations must include late feeds. | Batch failures or delayed manual entries. | Measure event-to-ingestion delay and completeness by source. |
| Historical lag predicts current timing. | A nowcast needs an age curve. | Promotion, policy, season, or carrier changes. | Backtest at past cutoffs and use scenarios or updated curves. |
| Returns are distinguishable from other remedies. | Populations need consistent definitions. | Exchanges, chargebacks, failed deliveries, or returnless remedies overlap. | Define mutually exclusive or explicitly overlapping event flags. |
| Route values use the same decision boundary. | Disposition options must be comparable. | One route includes common cost while another omits it. | Build a line-item bridge at the decision point. |
| Restocked units displace future purchases. | Avoided landed cost is used as recovery. | No demand, obsolete inventory, or excess stock. | Compare demand and inventory; use alternative liquidation value. |
| Conditional average loss fits avoided returns. | Simple prevention arithmetic multiplies rate change by a loss. | Intervention targets unusually high- or low-recovery returns. | Estimate cost and recovery for the affected reason and segment. |
| No conversion or mix change in the guide example. | Isolates return savings. | Guide alters purchasing or product choice. | Measure delivered volume, mix, and contribution in the experiment. |
| Repair yield is stable. | Expected repair value uses one success probability. | New defect mix, parts shortage, or technician learning. | Track yield and cost by defect, batch, and period. |
| Transport and processing capacity are available. | Expected routing assumes units can enter each route. | Peak backlog or specialist-site outage. | Model capacity and queue age; include a feasible fallback. |
| Monetary amounts are comparable. | Costs and recovery are combined. | Currency, tax, timing, or accounting differences. | Preserve source amounts, conversion date, and tax boundary. |
| The illustrative policy is lawful. | The numerical scenario needs a remedy rule. | Product, contract, or jurisdiction imposes different rights. | Obtain market-specific legal review before implementation. |
| Observational reason patterns identify a cause. | Prioritization needs a plausible mechanism. | Selection bias, default reason codes, concurrent changes. | Validate physical evidence and use a controlled comparison. |
| Average values describe tail risk. | A point estimate simplifies planning. | Few high-cost products or very delayed cases dominate. | Segment distributions and report percentiles and scenarios. |

These assumptions are not guarantees.
A useful decision record states which assumptions are estimates, who owns them, and when they will be refreshed.
If an assumption is untestable with current data, show the range over which the recommendation changes.

## 12. Exercises

### Exercise 1 — Choose the grain

**Time:** 15 minutes.
A cohort contains 400 delivered orders and 600 delivered units.
Sixty orders each have at least one returned unit, and 90 distinct units have qualifying return initiations.
Those units travel in 70 return parcels.

**Deliverable:** calculate the order incidence and unit initiation rate; explain why parcels are not a valid replacement numerator.
**Acceptance criteria:** use matching denominators, label both rates, and identify at least one many-to-many parcel relationship.
**Outcome:** 1.

### Exercise 2 — Maturity and nowcast

**Time:** 20 minutes.
A daily delivery cohort contains 2,000 units.
At day 10, 144 distinct units have initiated returns.
A comparable, validated historical curve suggests 60% of eventual 30-day initiations are observed by day 10.

**Deliverable:** compute the observed-to-date rate and a simple expected 30-day rate.
Name two reasons the estimate could be biased.
**Acceptance criteria:** distinguish observed from modeled values and state the lag stability assumption.
**Outcome:** 2.

### Exercise 3 — Reverse flow and ownership

**Time:** 25 minutes.
A retailer offers drop-off and home collection, plus returnless refunds for some low-value products.
Some parcels contain items from multiple orders, and refunds may be issued before physical receipt.

**Deliverable:** draw a flow from delivery to disposition and assign accountable owners to policy, receipt, refund, and unsafe-product exceptions.
**Acceptance criteria:** show at least two nonphysical or exceptional paths, distinguish parcel from item, and include an appeal route.
**Outcome:** 3.

### Exercise 4 — Select disposition

**Time:** 15 minutes.
A received item can be liquidated for $9 net.
Repair costs $7 per attempt and yields $28 net proceeds before repair cost with 70% probability; failure yields zero.
Both routes are safe, permitted, and have unconstrained capacity.

**Deliverable:** calculate expected net recovery for repair and choose the better route.
Find the repair success probability at which the routes have equal expected value.
**Acceptance criteria:** include the cost on every repair attempt and show the threshold equation.
**Outcome:** 4.

### Exercise 5 — Contribution reconciliation

**Time:** 20 minutes.
A delivered unit would have earned $70 revenue less $25 product cost and $7 outbound cost if kept.
It is fully refunded, incurs $9 reverse cost, and produces $18 net recovery.
The net recovery excludes the $9 common reverse cost.

**Deliverable:** calculate the incremental loss against the kept baseline and the returned-unit contribution.
Check the answer using a direct revenue-and-cost bridge.
**Acceptance criteria:** count original product cost once and do not deduct recovery costs twice.
**Outcome:** 5.

### Exercise 6 — Prevention business case

**Time:** 25 minutes.
A packaging change is forecast to reduce returns by 0.4 percentage points across 50,000 delivered units.
The appropriate incremental loss per avoided return is $35.
The change costs $4,000 in the period.

**Deliverable:** calculate avoided returns, gross benefit, net benefit, and break-even absolute rate reduction.
Propose a randomized evaluation and two customer or commercial guardrails.
**Acceptance criteria:** use percentage points correctly; label the estimate as forecast; describe the unit of randomization and follow-up horizon.
**Outcome:** 6.

### Exercise 7 — Policy and controls

**Time:** 30 minutes.
A proposal would refund every customer at first carrier scan and automatically reject all later requests from high-return customers.
The retailer operates in several jurisdictions and sells both apparel and battery-powered devices.

**Deliverable:** write a one-page decision memo recommending approval, modification, or rejection.
Identify mandatory-rule review, safety controls, exception handling, and a measurable pilot.
**Acceptance criteria:** separate confirmed abuse from frequent use, provide an accessible appeal, name owners, and state stop conditions.
**Outcome:** 7.

### Exercise 8 — Capacity and backlog

**Time:** 20 minutes.
A site begins a week with 120 units awaiting inspection.
It expects 500 units to arrive and can inspect at most 540 units that week.
Assume all arrivals are available for inspection and no units are removed through another route.

**Deliverable:** calculate the minimum possible closing backlog and identify two additional inputs needed for a staffing decision.
**Acceptance criteria:** use the flow-balance equation, respect the completion cap, and distinguish an expected forecast from a staffing commitment.
**Outcome:** 8.

## 13. Self-check answers

### Exercise 1

Order incidence is 60 / 400 = 15.0% of eligible orders.
Unit initiation rate is 90 / 600 = 15.0% of eligible delivered units.
The rates happen to be numerically equal, but they describe different populations.
Seventy parcels are neither 70 returned units nor 70 returned orders.
A parcel can contain several units or items from different orders; an order's items can also be split into multiple parcels.
The data contract must preserve the relationships rather than assume one-to-one mapping.

### Exercise 2

The observed-to-date rate is 144 / 2,000 = 7.2%.
Estimated 30-day initiations are 144 / 0.60 = 240 units.
The modeled 30-day rate is 240 / 2,000 = 12.0%.
This is not a 12.0% actual at day 10.
It assumes the historical cumulative share applies to this cohort and the event feed is complete.
A changed policy window or new assortment can alter customer timing.
Late ingestion can also make the observed count artificially low.
Validate the estimate on mature comparable cohorts observed at the same age.

### Exercise 3 rubric

A strong diagram includes customer request, authorization or exception, drop-off or collection, parcel transport, item receipt, inspection, and disposition.
It shows returnless remedies without a physical receipt and shows that a refund can precede or follow physical inspection.
It preserves an item-to-parcel crosswalk and accounts for mixed parcels and missing authorizations.
Customer experience or product owns policy design; legal approves applicable requirements.
Operations owns receipt and quarantine; finance or the payments owner controls settlement reconciliation.
A product-safety owner resolves unsafe-item exceptions under documented rules.
A customer can appeal a disputed decision through an accessible channel.
A weak answer labels every scan as a refund and assigns the entire process to a carrier.

### Exercise 4

Repair expected net recovery is 0.70 × $28 − $7 = $12.60 per attempt.
Liquidation provides $9.00 per unit.
Repair has $3.60 higher expected value under the stated assumptions.
At equality, \(28p - 7 = 9\), so \(p = 16/28 = 57.1\%\), rounded.
Below that success probability, liquidation has higher expected value.
This conclusion changes if costs, failure proceeds, capacity, or eligibility differ.

### Exercise 5

Kept contribution is $70 − $25 − $7 = $38.
Incremental return loss is $70 refund + $9 reverse cost − $18 recovery = $61.
Returned-unit contribution is $38 − $61 = −$23.
The direct bridge is $0 retained merchandise revenue − $25 product cost − $7 outbound − $9 reverse + $18 recovery = −$23.
The two methods agree because they use the same boundary.
Adding another $25 to the incremental loss would double count the original cost.

### Exercise 6

A 0.4-percentage-point reduction is 0.004 in probability units.
Expected avoided returns are 50,000 × 0.004 = 200.
Gross benefit is 200 × $35 = $7,000.
Net benefit is $7,000 − $4,000 = $3,000 for the period.
Break-even avoided returns are $4,000 / $35 = 114.29 expected units.
Break-even absolute rate reduction is 114.29 / 50,000 = 0.2286 percentage points, approximately 0.229 points.

One design randomizes eligible orders or another appropriate noninterfering unit to old and new packaging before shipment.
Prespecify the population, assignment, and 30-day post-delivery initiation horizon, adjusted for applicable policy and data latency.
Monitor delivered volume, product damage, customer contacts, and contribution in addition to returns.
If packaging is assigned at a warehouse or batch level, account for clustering and avoid claiming order-level independence.
The forecast is not a measured causal effect.

### Exercise 7 rubric

A strong memo modifies or rejects automatic blanket rejection until legal and risk review establishes a compliant, evidence-based process.
It distinguishes a carrier acceptance scan from receipt and inspection, and assesses exposure from an early refund.
It keeps defective-product and statutory remedies distinct from a voluntary change-of-mind policy.
It includes battery-safe transport, quarantine, and relevant carrier instructions.
It specifies accessible human review and appeal, policy versioning, privacy controls, and customer communication.
A pilot identifies a population, comparison, outcome window, remedy-time and false-denial guardrails, and stop authority.
Legal owns jurisdictional interpretation; customer experience owns the journey; risk, operations, and finance own their relevant controls.
A weak memo treats a frequent returner as proven fraudulent or assumes one global policy is lawful everywhere.

### Exercise 8

Available work is 120 + 500 = 620 units.
At most 540 can be inspected, so the minimum closing backlog is 620 − 540 = 80 units.
If arrivals come too late for the available shifts, the actual backlog can be larger.
A staffing plan needs arrival timing and uncertainty, labor productivity and attendance, skill mix, service targets, and opening backlog age.
The 500 arrivals are an expectation, not a commitment that each shift will receive a known quantity.

## 14. Interview questions

1. **How would you define a return rate for an apparel category?**
   A strong answer selects a delivery cohort, eligible unit denominator, initiation or receipt event, horizon, and treatment of exchanges and returnless refunds.
   A weak answer divides current-month parcels by current-month orders without explaining timing or grain.
2. **Why do recent cohorts often look better than mature cohorts?**
   A strong answer describes right censoring, age alignment, data latency, and a validated nowcast or mature-cohort comparison.
   A weak answer assumes fewer returns must mean better product quality.
3. **How would you link a mixed return parcel to two orders?**
   A strong answer preserves parcel, request, order-line, and item or quantity identities and handles exceptions.
   A weak answer uses tracking number as the unique unit key.
4. **When does a returnless refund make sense?**
   A strong answer compares expected reverse cost and recovery, customer burden, safety, policy, and risk controls.
   A weak answer says it is always cheaper because freight is avoided.
5. **How would you choose between repair and liquidation?**
   A strong answer compares feasible expected net values at the same decision point, including success yield and constrained capacity.
   A weak answer selects the route with the highest gross sale price.
6. **How can a refund be $80 but the incremental loss differ?**
   A strong answer starts from a kept-sale baseline, includes reverse cost and recovery, and avoids recounting original product cost.
   A weak answer adds every cost and future sale value without a boundary.
7. **What would you do with a large inspection backlog?**
   A strong answer segments by site, age, safety, seasonality, skill, and expected recovery; it reconciles arrivals and completions.
   A weak answer closes cases in the system without verifying physical disposition.
8. **What does a customer reason code prove?**
   A strong answer treats it as a customer-reported signal and validates root cause with independent evidence.
   A weak answer assigns supplier penalties directly from unverified portal selections.
9. **How would you evaluate a sizing guide?**
   A strong answer defines a randomized exposure, mature outcome window, contribution measure, and conversion and customer guardrails.
   A weak answer compares two months with different promotions and calls the difference causal.
10. **What should a returns forecast feed into operations?**
    A strong answer models delivered cohorts, initiation lag, shipment conversion, transit, routing, and units per parcel by period.
    A weak answer multiplies current sales by a single return rate and assumes all work arrives immediately.
11. **What must change for a cross-border return?**
    A strong answer identifies the seller, customer rights, reverse transport, duty and tax treatment, product restrictions, and contract responsibility.
    A weak answer assumes the domestic policy and freight rate apply globally.
12. **How would you use an abuse score responsibly?**
    A strong answer requires relevant evidence, monitoring of false denials, legally appropriate remedies, accessible review, and an appeal path.
    A weak answer treats frequent returns or a model score as conclusive proof of fraud.

## 15. Key terms

- [Backlog](#510-forecast-workload-and-expose-uncertainty): work awaiting a defined stage; see also the queue balance in section 5.10.
- [Cohort](#53-measure-rates-at-compatible-grain): units grouped by a common delivery period.
- [Contribution](#59-reconcile-the-financial-loop): revenue less costs within an explicit boundary.
- [Customer remedy](#56-design-the-customer-policy): refund, replacement, repair, or other resolution as applicable.
- [Cycle time](#4-definitions-and-vocabulary): elapsed or working time between named events.
- [Delivery event](#52-build-the-event-ledger): the recorded event anchoring a post-delivery cohort.
- [Disposition](#58-inspect-and-choose-disposition): authorized route for a returned unit.
- [Event ledger](#52-build-the-event-ledger): linked and versioned operational and financial facts.
- [Exchange](#4-definitions-and-vocabulary): replacement remedy with its own physical and financial flows.
- [Gross unit return rate](#53-measure-rates-at-compatible-grain): qualifying distinct units divided by eligible delivered units.
- [Initiation](#53-measure-rates-at-compatible-grain): defined start of a return request.
- [Inspection](#58-inspect-and-choose-disposition): condition and eligibility assessment.
- [Lag distribution](#54-handle-maturity-censoring-and-lag): timing distribution between specified events.
- [Maturity](#54-handle-maturity-censoring-and-lag): extent to which the defined outcome window is observed.
- [Net recovery](#58-inspect-and-choose-disposition): proceeds or avoided cost less route-specific incremental costs.
- [Nowcast](#54-handle-maturity-censoring-and-lag): estimate of a later cohort outcome from incomplete current observation.
- [Order incidence](#53-measure-rates-at-compatible-grain): eligible orders with a qualifying return divided by eligible orders.
- [Parcel](#57-plan-reverse-transport-and-receipt): physical reverse shipment that can contain multiple items.
- [Policy version](#56-design-the-customer-policy): effective set of rules governing an applicable transaction.
- [Prevention](#511-prioritize-prevention-and-measure-causal-impact): intervention intended to avoid a return's underlying cause.
- [Quarantine](#58-inspect-and-choose-disposition): controlled holding state before an authorized release or other route.
- [Receipt](#57-plan-reverse-transport-and-receipt): recorded physical arrival of an identified item.
- [Refund](#59-reconcile-the-financial-loop): value returned to the customer.
- [Repair yield](#58-inspect-and-choose-disposition): fraction of repair attempts achieving the defined successful outcome.
- [Resaleable yield](#64-physical-disposition-and-recovery): specified resaleable units divided by a stated base.
- [Return authorization](#57-plan-reverse-transport-and-receipt): reference or permission for a defined request.
- [Returnless remedy](#56-design-the-customer-policy): customer resolution without physical return.
- [Reverse cost](#59-reconcile-the-financial-loop): included incremental costs of the return loop.
- [Right censoring](#54-handle-maturity-censoring-and-lag): observation ends before all possible events occur.
- [Root cause](#55-diagnose-reasons-and-root-causes): supported underlying cause rather than a reported symptom.
- [Salvage value](#58-inspect-and-choose-disposition): value realized through a lower-value disposition route.
- [Value return rate](#53-measure-rates-at-compatible-grain): returned or refunded merchandise value over a compatible sales-value base.

## 16. Sources

The following sources are real, checkable starting points; company policy pages can change.
The descriptions below distinguish primary documentation from research and company claims.
Access the current source and applicable version before making an operational or legal decision.

1. **GS1, “GS1 General Specifications.”** Identification and barcode standards, including trade-item and logistics-unit identifiers.  
   <https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications>  
   Evidence: standards body; conventions or binding obligations depend on the applicable contract or law.
2. **GS1, “EPCIS.”** Standard for sharing visibility event data across enterprises.  
   <https://www.gs1.org/standards/epcis>  
   Evidence: standards body; event capture does not itself verify physical truth.
3. **European Parliament and Council, Directive 2011/83/EU on consumer rights.** Official text of the consumer-rights framework, including distance contracts and withdrawal rules.  
   <https://eur-lex.europa.eu/eli/dir/2011/83/oj>  
   Evidence: legislation; check amendments, national implementation, exceptions, and facts.
4. **U.S. Federal Trade Commission, “Returns and Refunds.”** Consumer guidance on return and refund policies.  
   <https://consumer.ftc.gov/articles/returns-and-refunds>  
   Evidence: government guidance; state and transaction-specific requirements still need review.
5. **IFRS Foundation, IFRS 15, “Revenue from Contracts with Customers.”** Revenue recognition standard; includes application guidance for sales with a right of return.  
   <https://www.ifrs.org/issued-standards/list-of-standards/ifrs-15-revenue-from-contracts-with-customers/>  
   Evidence: accounting standard body; consult the applicable framework and current authoritative text.
6. **Fleischmann, M., Bloemhof-Ruwaard, J. M., Dekker, R., van der Laan, E., van Nunen, J. A. E. E., and Van Wassenhove, L. N. (1997), “Quantitative models for reverse logistics: A review,” European Journal of Operational Research, 103(1), 1–17.**  
   <https://doi.org/10.1016/S0377-2217(97)00230-0>  
   Evidence: peer-reviewed review; model assumptions and period limit direct transfer to a present retailer.
7. **Amazon Customer Service, “Return Items You Ordered.”** Public instructions for initiating eligible returns.  
   <https://www.amazon.com/gp/help/customer/display.html?nodeId=GKM69DUUYKQWKWX7>  
   Evidence: company policy information; not independent evidence of cost or causal effect.
8. **IKEA U.S., “Return policy.”** Public market-specific customer policy and conditions.  
   <https://www.ikea.com/us/en/customer-service/returns-claims/return-policy/>  
   Evidence: company policy; check current terms and transaction date.
9. **Patagonia, “Worn Wear.”** Public description of repair, trade-in, and used-product activities.  
   <https://wornwear.patagonia.com/>  
   Evidence: company program information; trade-in is not automatically a refund return.

The examples in sections 6, 7, 12, and 13 are invented teaching scenarios.
They should not be attributed to any source or company above.

## 17. Links

- [Module 21 topic index](_index.md).
- [Module 21 survey](../../../course/21_ecommerce_supply_chain_analytics.md).
- [Fulfillment models](01_ecommerce-fulfillment-models.md): commercial ownership and physical execution.
- [Long-tail forecasting](03_long-tail-forecasting.md): sparse demand and hierarchy.
- [Inventory placement](04_inventory-placement.md): location and availability of recovered stock.
- [Delivery promise](05_delivery-promise.md): customer commitment and service design.
- [Simulation and optimization production labs survey](../../../course/30_simulation_optimization_production_labs.md): methods for testing capacity and routing scenarios.
