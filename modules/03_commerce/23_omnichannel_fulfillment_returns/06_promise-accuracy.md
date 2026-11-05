# Promise Accuracy: Measure What the Customer Was Actually Told

> Module 23 — Omnichannel Fulfillment, Returns, and the Customer Promise  
> Topic 06 of 10  
> Estimated study time: 4 hours  
> Prerequisites: inventory visibility, available-to-promise, and order routing  
> Version 2026.5

## 1. Why this topic exists

A checkout can show a customer an attractive delivery date that the operation cannot reliably meet.
A distribution center can meet its dispatch target while a parcel arrives late.
A store can finish picking on time while the customer never receives the ready notification.
The merchant may nevertheless report excellent performance if its metric uses a later revised date, excludes cancellations, or counts only orders with clean carrier scans.

Promise accuracy connects a customer-facing commitment to the eventual customer-facing outcome.
It is not the same as carrier on-time performance or warehouse adherence to a dispatch plan.
The difference matters because the customer bought an end-to-end service, not a collection of independently optimized tasks.

A promise can be a calendar date, a time window, a range, or a pickup-ready deadline.
The calculation must preserve the wording, time zone, channel, conditions, and date actually displayed at the decision point.
If the system stores only its current estimate, the historical customer experience cannot be reconstructed reliably.

A promise owner needs answers to several different questions:

- Did the merchant meet the commitment that induced the purchase?
- Can the customer use the goods when they expected to use them?
- Where in the network are failures accumulating, including orders not yet resolved?
- Does a faster promise improve conversion enough to pay for its service and failure costs?
- Which action can the responsible team take before the next customer is disappointed?

These questions require distinct measures and decisions.
A high percentage on a small, selected set of delivered parcels is not evidence that the whole ordering experience is healthy.
A wider promise window may raise the percentage met while making the offer less useful.
A team rewarded only for speed or only for on-time performance can move harm outside its own scorecard.

This lesson builds an immutable promise record, connects it to delivery and pickup events, defines an accountable cohort, and turns the result into a control and business case.
The examples are illustrative, not actual company results.
Legal and privacy references are educational rather than legal advice.

## 2. Learning outcomes

After completing this topic you can:

1. Specify which displayed and accepted promise versions to retain and identify the primary commitment for measurement.
2. Calculate mature-cohort on-time performance, unresolved exposure, and bounds without excluding difficult orders.
3. Reconcile orders, lines, parcels, delivery attempts, cancellations, and pickup-ready events at their proper grains.
4. Diagnose whether a gap is associated with promise design, fulfillment execution, carrier transit, or measurement quality.
5. Design operational triggers and accountable interventions with safeguards against shifting failures between teams.
6. Evaluate an incremental business case that separates contribution, service recovery, operating cost, and causal uncertainty.
7. Propose privacy, data quality, incentive, and accessibility controls for a production scorecard.
8. Communicate the difference between a statistical estimate, a forecast, a target, and a customer commitment.

Exercises 1–8 and their self-checks cover these outcomes in the same order.

## 3. The core mental model

The unit of analysis is a promise made to a customer about an identifiable order or order component.
Every outcome is compared with that promise under a rule defined before the result is known.

```text
Offer rendered -> customer selects -> order accepted -> fulfillment events
       |                  |                  |                 |
       v                  v                  v                 v
 immutable offer     accepted promise    order/line links   parcel or pickup
 and context         and version         and revisions      event evidence
                           |                                    |
                           +---------- cohort ledger -----------+
                                           |
                  +------------------------+------------------+
                  v                        v                  v
           customer outcome        operational stages    data exceptions
                  |                        |                  |
                  +--------------> decision <----------------+
                                           |
                          revise capacity, routing, promise,
                          communication or data controls
```

### Four clocks

1. The **display clock** records when the customer saw the offer.
2. The **commitment clock** records the deadline implied by the accepted promise.
3. The **event clock** records when fulfillment and delivery actually occurred.
4. The **observation clock** records when events arrived in the analytics system.

These clocks are not interchangeable.
An event that happened on Monday but reached the warehouse on Thursday belongs to Monday operationally, while a Monday report may not yet have observed it.
Persist both event and ingestion timestamps.

### Forecast, plan, target, and commitment

A forecast is a probabilistic estimate of a future outcome, such as a 90% probability of delivery by Friday.
A plan assigns resources or schedules work to achieve an intended outcome.
A target is a management objective, such as 96% of eligible orders meeting their commitment.
A commitment is what a customer was told they could expect, such as delivery by Friday at 21:00 local time.
A target can be missed even when individual commitments were reasonable.
A forecast can change after purchase without changing the original customer commitment.

### Measure the customer and the operation separately

The primary service measure compares the accepted promise with actual fulfillment.
A diagnostic measure compares individual events with their internal plans.
A revised-promise measure describes whether subsequent communications were met.
A customer-experience measure can include cancellations, refunds, contacts, and missing items.
All are useful; none should silently replace the original-promise measure.

### Lead with the customer-relevant event

For delivery, use a defensible proof-of-delivery event when available, not label creation or handover to a carrier.
For pickup, measure a valid ready notification against the pickup-ready promise and track collection separately.
For a whole-order promise, every required item must be available to the customer before the deadline.
A line, parcel, and order each answer a different question, and none may be relabeled as another.

### Accuracy is not inherently speed

Promising ten days and delivering in nine can be accurate but uncompetitive.
Promising tomorrow and delivering the next day can be commercially attractive but systematically unreliable.
Pair accuracy with the promised lead-time distribution, realized lead time, conversion, cost, and customer outcomes.
Compare like customer, item, lane, and season mixes before drawing conclusions.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Offer | A rendered fulfillment option, including wording and eligibility; one display event | Assuming every calculated estimate was shown |
| Display event | Evidence that an offer was rendered in a session at an instant | Treating a server response as confirmed visibility |
| Accepted promise | Promise attached to the selected option when the order is accepted | Using the latest internal estimate |
| Original promise | First accepted commitment for the defined transaction | Substituting a later carrier update |
| Revision | Later communicated change with timestamp, wording, and reason | Overwriting the earlier version |
| Deadline | Latest instant included by the promise, stored in UTC with local interpretation | Comparing date-only strings in mixed time zones |
| Time window | Inclusive or exclusive interval presented to the customer | Assuming the start is the deadline |
| Lead time | Duration from a named start event to a named end event, in hours or days | Comparing durations with different starting clocks |
| Cohort | Population grouped by a fixed event and period, such as acceptance week | Grouping only by eventual delivery week |
| Eligible order | Order within the published population definition | Excluding late orders after observing the outcome |
| Mature cohort | Cohort whose deadlines and observation lag have passed under a stated policy | Assuming every mature record is resolved |
| Unresolved | No valid terminal outcome is available at the analysis cutoff | Counting it as on time |
| Right censoring | Outcome is not observed by the analysis cutoff | Treating incomplete observations as completed |
| On-time | A valid relevant event is at or before the applicable deadline | Treating a carrier's internal SLA as the customer promise |
| On-time rate | On-time count divided by a specified eligible or resolved population | Omitting the denominator and exclusions |
| Coverage | Fraction of the target population with valid required evidence | Hiding missing promise or delivery data |
| Order grain | One customer transaction; one row per order | Averaging parcel rates and calling the result order rate |
| Line grain | One ordered product line, potentially multiple units | Counting only the first fulfillment of a split line |
| Parcel grain | One carrier package or tracking identifier | Treating a single delivered parcel as a completed order |
| Split order | An order fulfilled in more than one shipment or node | Ignoring the last required component |
| Proof of delivery | Carrier or merchant evidence of completed delivery | Assuming every scan is accurate or customer receipt is proven |
| Attempt | An attempted handoff that may not complete delivery | Counting an attempted delivery as delivered by default |
| Pickup-ready event | Authorized fulfillment completion and customer notification for collection | Using pick completion when notification is required |
| Cancellation | An order or line ended without ordinary completion | Erasing it from the cohort |
| Customer-requested change | Documented request that alters delivery or pickup | Treating every post-order change as consent |
| Service recovery | Action such as a refund, credit, or expedited replacement after a failure | Counting a credit as reversing a missed original promise |
| First-attempt delivery | Successful completion on the first delivery attempt | Equating first attempt with proof of delivery |
| Exception | A record needing an explicit rule or review | An automatic exclusion from all reporting |
| ETA | Estimated time of arrival; a prediction that can change | Assuming every ETA was a customer commitment |
| Available-to-promise (ATP) | Inventory and supply available for a future commitment under defined rules | Equating book on-hand inventory with sellable stock |
| Distributed order management (DOM) | Assignment and coordination of orders across fulfillment nodes | Assuming the lowest shipping cost always meets the promise |
| Buy online, pick up in store (BOPIS) | Digital purchase followed by physical store collection | Treating customer collection delay as a late ready notification |
| Event time | When an event occurred, with time zone or offset | Replacing it with ingestion time |
| Ingestion time | When an event entered the data platform | Claiming it is the time of physical delivery |
| Attribution | Assignment of a reason or cost under a documented rule | Proof that the attributed factor caused the miss |
| Guardrail | A measure that limits harm while optimizing a primary objective | A metric without a threshold or decision owner |
| Contribution | Revenue less explicitly defined incremental variable costs, in a named currency | Counting gross sales as profit |
| Intent-to-treat | Analysis by originally assigned experimental policy | Removing customers whose treatment was inconvenient |
| Confidence interval | A range from a specified statistical procedure reflecting sampling uncertainty | A guarantee about a particular future order |

## 5. The mechanics, step by step

### Step 1 — Decide what the customer was promised

**Input:** rendered offer, selected service, checkout transaction, and confirmation message.
**Output:** immutable promise version and evidence of its presentation or acceptance.
**Owner:** product manager, supported by the checkout engineering lead.

Record the actual customer-facing text and a structured representation.
A calculated date returned by a service is not sufficient evidence that the interface rendered it.
Capture a rendering event where feasible, while recognizing that rendering does not prove the customer read the text.
Link the selected option to the resulting order through a stable, privacy-protective identifier.
If checkout changes the option or address before acceptance, record the newly displayed offer and the final accepted promise.

Retain at least:

- Promise version identifier and immutable creation time.
- Transaction, session, and selected-option identifiers, subject to retention policy.
- Channel, locale, device context, and content or template version.
- Rendered words, date or range, price, cutoff conditions, and any holiday qualification.
- Destination time zone and, where applicable, store time zone.
- Earliest and latest promised instants or local dates, with date-only interpretation.
- Service type: delivery, pickup-ready, appointment, or another explicitly defined commitment.
- Selection and order-acceptance events, including whether acceptance succeeded.
- Model, rule, and input-version identifiers needed to reproduce the decision.
- Subsequent customer-visible revision, reason, channel, timestamp, and evidence of delivery of the message.

Avoid storing a precise street address in an analytics fact when a protected location key and service area suffice.
Do not infer that an email was read merely because the message was sent.

**Branch:** If a shown offer cannot be linked to an accepted order, retain it for offer and conversion analysis if consent and policy permit; do not invent an order promise.
If an accepted order lacks a recorded promise, flag the record as unknown and report capture coverage.
A reconstructed promise may support a sensitivity analysis but must not masquerade as the captured original.

### Step 2 — Establish a canonical deadline

**Input:** accepted promise text and structured attributes.
**Output:** deadline instant, local date, interval rule, and parsing status.
**Owner:** promise product owner and analytics engineering lead.

For a date-only promise, define the end of the promised local day using the published service terms.
A practical representation is the exclusive start of the following local day, converted to Coordinated Universal Time (UTC).
Then an event is on time if its timestamp is strictly earlier than that exclusive bound.
This avoids inventing precision such as 23:59:59.999 and handles daylight-saving transitions when a current time-zone database is used.
For an explicitly promised time, document whether the stated instant is inclusive and preserve that precision.

A range such as “Tuesday–Thursday” normally measures the latest date for the on-time deadline; early delivery may need a separate acceptability rule.
A scheduled appointment may have both early and late failure boundaries.
A promise such as “within two business days” needs the applicable location, cutoff, weekends, holidays, acceptance instant, and calendar version.
Never silently replace an ambiguous message with a fabricated deadline.

**Branch:** If the customer selected a delivery slot after checkout, use the accepted slot for the slot measure and preserve the earlier commitment for its own measure.
If the address changes, record both the initial commitment and the customer-requested change; apply a predeclared exception policy, not an after-the-fact rewrite.
If the time zone is unknown, mark the deadline unknown and quantify the affected population.

### Step 3 — Define the population before observing outcomes

**Input:** order acceptance log and policy scope.
**Output:** dated cohort contract with denominator, exclusions, and status rules.
**Owner:** customer-experience metric owner, with finance and operations approval.

A defensible primary cohort groups accepted orders by acceptance week and original service type.
Every accepted order in scope enters once, including orders later canceled or not yet delivered.
Track test orders, duplicates, and fraud-confirmed transactions under objective, documented rules.
Publish their counts and reasons instead of silently deleting them.
The policy should distinguish merchant-caused cancellations from documented customer-requested cancellations.

For a strict end-to-end promise measure, count merchant-caused cancellation as a failure.
Report customer cancellations as a separate terminal status and show both an all-accepted view and a predeclared service-eligible view.
Do not label the latter an all-order rate.
If customer intent is unknown, retain an unknown-reason category rather than assigning fault.

Separate order-level delivery and pickup-ready cohorts unless their event definitions are made explicit.
An order with delivery and pickup components requires an order-level completion rule and component-level diagnostics.
Report by the original selected option, not by the eventual fulfillment path alone.

### Step 4 — Build the event and entity graph

**Input:** order lines, allocation records, shipments, tracking events, cancellations, and notifications.
**Output:** an as-of ledger with one row per metric grain and linked evidence.
**Owner:** data engineering lead, with carrier integration and store operations.

An order can have multiple lines; a line can have multiple allocations and parcels.
A parcel can include units from several lines.
Do not join these tables and then count the resulting rows as independent orders.
Aggregate completion at the component grain first, then roll up to an order.

For a whole-order delivery promise, completion is the latest valid delivery time across all required components.
For a line with multiple units, all required units must be accounted for unless a partial-delivery promise was explicitly accepted.
Replacements, substitutions, and cancellations need a declared rule about whether they satisfy the original commitment.
Record the accepted substitution and its timing; do not assume a different product is equivalent.

Use event identifiers and source sequence numbers to deduplicate retries.
Preserve source event time, source time zone, ingestion time, correction time, and source system.
Retain both an append-only event history and an as-of materialization for a dated report.
Corrections should create a new version with provenance; a historical report should be reproducible as it was known at the time.

**Branch:** If a carrier provides only a date without a time, evaluate date-based promises at compatible precision and mark a timed promise indeterminate.
If proof conflicts with a customer claim of nonreceipt, retain the carrier-evidence result and track the dispute and resolution separately.
Do not assert that a scan proves the customer physically received the parcel.

### Step 5 — Treat pickup as a different customer outcome

**Input:** pick completion, staging, notification, store hours, and collection events.
**Output:** pickup-ready performance and a separate collection outcome.
**Owner:** store operations leader and digital product owner.

A ready promise is met only when the ordered eligible items are actually staged and the required ready message is issued by the deadline under the published policy.
If a message says ready but the items cannot be found, the event is not a valid ready outcome.
If the promise says a customer may collect during store hours, the hours and relevant closure calendar are part of the interpretation.
Do not credit a midnight notification as usable pickup if the store is closed and the commitment promised access before closure.

Collection is controlled partly by the customer.
Track collection time, abandonment, hold expiration, and handoff accuracy separately from readiness.
A customer who collects tomorrow does not make a valid ready-today commitment late.
Conversely, a quick customer handoff cannot repair a notification sent after the promised ready time.

### Step 6 — Choose the observation cutoff and maturity rule

**Input:** cohort dates, promise deadlines, event arrival history, and reporting cutoff.
**Output:** mature and open cohorts with explicit unresolved status.
**Owner:** analytics lead and operational metric owner.

Let the reporting cutoff be an explicit instant, not merely a dashboard refresh date.
Only use event information ingested by that cutoff for an as-of report.
For settled reporting, choose a grace period based on measured source latency, such as a high percentile of delivery-event arrival delay.
A cohort is eligible for the settled view after every relevant deadline plus the chosen grace period has elapsed.
A grace period reduces, but cannot eliminate, missing events.
Continue to show unresolved records rather than assuming they were delivered.

For each cohort, publish at least:

- Accepted and eligible order counts.
- On-time, late, merchant-canceled, customer-canceled, and unresolved counts under mutually exclusive rules.
- Promise-capture and outcome-evidence coverage.
- Original-promise on-time share and its denominator.
- Resolved-only share, clearly labeled, if used for diagnosis.
- Observation cutoff, maturity rule, revisions, and late-arriving corrections.

A live view should show open commitments, approaching deadlines, already overdue unresolved orders, and aged missing events.
It is not a substitute for a mature cohort.
A delivery-week report is useful for workload, but it does not answer how a specific acceptance cohort performed.

### Step 7 — Calculate rates, bounds, and uncertainty

**Input:** the cohort ledger and eligibility policy.
**Output:** reproducible point measures and transparent unknown-outcome bounds.
**Owner:** analytics lead.

For a simple strict all-accepted measure:

```text
on_time_rate = on_time_orders / accepted_orders
```

Merchant cancellations, late deliveries, customer cancellations, and unresolved outcomes remain in that denominator.
This is a conservative operational share rather than a claim that every unresolved or customer-canceled order was physically late.
Report the status breakdown beside it.

A resolved-only rate can be useful when defined as on-time divided by on-time plus late plus merchant-canceled orders.
Exclude customer cancellations only if the declared definition does so; show their number alongside the rate.
Never compare a resolved-only metric with a target defined on all accepted orders.

If unresolved orders are the only uncertain outcomes and every known non-on-time outcome is classified as failure under the chosen policy:

```text
lower_bound = on_time / eligible
upper_bound = (on_time + unresolved) / eligible
```

These are identification bounds under the stated classification assumptions, not confidence intervals.
If a missing promise or disputed cancellation could change eligibility or classification, calculate separate sensitivity cases and disclose them.
A confidence interval for a sampled rate addresses sampling uncertainty under its model; it does not repair selection bias or missing data.
Orders sharing a disruption or customer are correlated, so simple independent-order intervals can overstate precision.

### Step 8 — Slice without hiding mix or small cells

**Input:** complete cohort with attributes captured as of acceptance and operational event history.
**Output:** prioritized diagnostic views and comparable benchmarks.
**Owner:** analytics partner with the relevant operations owners.

Useful acceptance-time dimensions include offer type, channel, region, original promise lead time, product category, and customer-selected service.
Operational dimensions include assigned node, actual node, carrier, lane, capacity period, split status, and exception reason.
Retain both original and actual values when rerouting occurs.

Show counts with every percentage.
A node with 1 miss in 5 orders has an 80% rate; it is not necessarily worse in underlying reliability than a node with 40 misses in 200 orders.
Suppress or aggregate small cells where privacy policy requires it.
Distinguish a within-segment change from a shift in the mixture of easy and difficult orders.
A weighted comparison using a fixed reference mix can be informative if there is sufficient common support.

A carrier may appear late because the warehouse handed over after cutoff.
Compute stage intervals and compare with the actual plan for each stage before assigning an owner.
An association between a node and a miss is a prioritization signal, not proof of cause.

### Step 9 — Diagnose stage-level exposure

**Input:** deadline, event timestamps, planned cutoffs, and latest capacity forecasts.
**Output:** stage exception queue with responsible action and due time.
**Owner:** control tower or equivalent operations lead.

A practical stage map is:

1. Acceptance to allocation: check reservation and inventory confidence.
2. Allocation to pick start: check queue age and labor availability.
3. Pick to pack: check missing stock, substitutions, and quality holds.
4. Pack to carrier handover: check pickup schedule and missed cutoff.
5. Handover to last-mile delivery: check carrier scans, transit variance, and exceptions.
6. Delivery to evidence: check delayed or contradictory event feeds.

Compute remaining slack as the customer deadline minus the current time and an evidence-based remaining process duration.
A negative slack forecast is an intervention signal, not an already observed delivery failure.
Use service-specific distributions rather than a universal average.
An uncertain event feed may require a data investigation rather than an unnecessary customer alert.

For each exception, assign a decision, owner, trigger, escalation time, customer impact, and evidence of closure.
Possible interventions include rerouting before pick, allocating additional labor, replacing inventory, upgrading transport, or communicating a change.
Do not promise an upgraded delivery without verifying inventory, capacity, and carrier feasibility.

### Step 10 — Separate original and revised commitments

**Input:** versioned customer communications and any explicit acceptance of a change.
**Output:** original-promise outcome plus revision and recovery measures.
**Owner:** customer-experience lead.

When a carrier disruption changes the ETA, preserve the original accepted commitment.
Record whether a revised estimate was sent, when it was sent, and whether it was actually presented in the customer's chosen channel.
If the customer explicitly agrees to a new slot, preserve that acceptance and measure the new slot separately.

Report both original-promise accuracy and revised-promise accuracy for the relevant population.
Add revision frequency, notice lead time, and service-recovery cost.
A revision sent after the old deadline is not advance notice.
A communication can reduce uncertainty but does not retroactively make the original commitment accurate.

### Step 11 — Use controls that change behavior

**Input:** mature-cohort results, live exposure, capacity, and customer feedback.
**Output:** time-bound actions, escalation, and an audit trail.
**Owner:** network operations lead; commercial and product owners retain decision rights for customer promises.

Set separate triggers for early warning and confirmed deterioration.
For example, review a lane if its rolling seven-day mature cohort falls below 94% on at least 200 eligible orders, or if its overdue unresolved count exceeds a predefined threshold.
The numbers are illustrative and require calibration to actual volumes and costs.
Do not run a low-volume node on a volatile percentage alone; use pooled periods or a counts-based exception review.

At each review, identify the root-cause hypothesis, evidence, customer exposure, proposed action, owner, and review date.
A decision to extend a promise must consider conversion and customer usefulness, not only the on-time rate.
A decision to accelerate a promise must include capacity, cost, and tail risk.
Return a failed pilot to a safe offer or constrain it to a verified region if guardrails are breached.

### Step 12 — Govern incentives and the business case

**Input:** scorecard, cost model, causal evidence, and decision rights.
**Output:** balanced objectives and an investment decision with downside sensitivity.
**Owner:** general manager and finance partner.

Assign a customer-facing outcome to a cross-functional owner who can convene product, stores, distribution, and carrier teams.
Functional owners retain diagnostic measures for their own controllable stages.
Do not penalize a store for an unverified carrier scan or reward a carrier for a late handover it did not control.

Pair the original-promise metric with promise speed, conversion, cancellations, contact rate, cost per accepted order, and evidence coverage.
Audit the treatment of exclusions, date revisions, partial shipments, and high-risk customers.
If teams can improve the metric by extending every promise, the scorecard is incomplete.
If they can improve conversion by making unrealistic promises, the scorecard is also incomplete.

Estimate incremental contribution from an intervention relative to a feasible counterfactual.
Include additional volume only when evidence supports causality; include margin, shipping, labor, credits, contacts, and operating costs without double counting.
Use a controlled experiment where feasible and a carefully qualified observational comparison otherwise.
Track realized results after rollout against the same population and cost definitions.

### Step 13 — Write the metric contract before launch

The metric contract is a decision document, not merely a query comment.
Its purpose is to make two independent implementations reach the same classification.
It should be approved by the business owner and by the teams that supply evidence.
Store a version and effective date with every published result.
Changes in policy can otherwise look like changes in operational performance.

Use the following specification checklist.
Each item prevents a different source of disagreement:

1. **Metric name:** say whether the unit is an order, line, unit, parcel, or customer.
2. **Business question:** state which customer commitment is being evaluated.
3. **Population:** list acceptance channels, services, regions, and order types in scope.
4. **Start event:** define the system and timestamp that establish cohort membership.
5. **Promise version:** specify which accepted version is primary and how revisions are stored.
6. **Deadline:** state the interpretation of dates, times, ranges, and local calendars.
7. **Completion:** specify the customer-relevant event and required evidence.
8. **Splits:** specify the rollup when required quantities travel in multiple parcels.
9. **Substitutions:** distinguish an accepted substitute from an unfulfilled requirement.
10. **Cancellations:** specify treatment by scope, reason, and evidence of customer request.
11. **Unknowns:** state how missing promises, outcomes, and reason codes appear.
12. **Maturity:** define the observation cutoff and the late-arrival allowance.
13. **Numerator:** define exactly what counts as an on-time success.
14. **Denominator:** define exactly which eligible records are counted.
15. **Companion measures:** list speed, coverage, cost, and customer-outcome guardrails.
16. **Revisions:** define when a historical result is restated and how it is labeled.
17. **Dimensions:** distinguish values at acceptance from values after fulfillment.
18. **Ownership:** name an accountable role and data-source stewards.
19. **Approvals:** identify product, operations, finance, privacy, and legal reviewers as needed.
20. **Versioning:** define how a policy change is communicated and compared with prior periods.

For each item, identify at least one example at the boundary.
A boundary example is more useful than a broad statement that timestamps should be correct.
For instance, state whether a 17:00 event satisfies a “by 17:00” promise.
State whether a delivery event at the exclusive midnight boundary belongs to the preceding day.
State which customer-facing message governs if the option changes between cart and order confirmation.
Where the evidence cannot settle an example, the correct classification may be unknown.
The number of unknowns is then part of the decision, not a reason to invent precision.

### Step 14 — Specify state transitions and precedence

An order can acquire several operational statuses over its life.
The scorecard needs one analytical status at an observation cutoff without deleting the underlying events.
Define a deterministic precedence policy and publish it with the metric.
The policy below is an example for a whole-order delivery commitment.
It is not a universal legal or contractual rule.

1. Start with an accepted order and its immutable original deadline.
2. Identify all required units and accepted substitutions at the relevant times.
3. If the promise is unavailable, assign an unknown-promise flag.
4. Independently determine whether the order has a qualifying customer-requested cancellation.
5. If all required units have valid delivery evidence by the deadline, classify on time.
6. If all required units have valid delivery evidence after the deadline, classify late.
7. If required units were merchant-canceled and the original commitment cannot be completed, classify merchant-canceled.
8. If a qualifying customer cancellation prevents completion, classify customer-canceled under the published policy.
9. If no terminal classification can be established by the cutoff, classify unresolved.
10. Retain disputes, missing evidence, and data defects as separate flags.

The precedence must address conflicts rather than merely assume they do not occur.
For example, a system may record cancellation after the carrier has already delivered.
An operations analyst should inspect the effective timestamps and source evidence.
If the cancellation was only a delayed administrative update, the delivered outcome may govern.
If the timeline is contradictory, retain an exception and avoid silently picking the convenient event.

Partial cancellation requires additional care.
Suppose an order contains two required units and one is canceled by the merchant.
Delivering the other unit does not fulfill an all-items commitment.
The order is not on time at the whole-order grain unless a valid accepted change altered the requirement.
Report the delivered unit separately if unit-level service is also useful.
Avoid counting the same order once as delivered and again as canceled in a mutually exclusive order-level table.

A customer who cancels after a missed deadline presents another policy question.
The original commitment may already have failed before the customer requested cancellation.
Preserve both the deadline comparison and the later cancellation event.
Define whether the headline reports this as late, customer-canceled after deadline, or another explicit status.
Do not use a broad customer-cancellation code to erase an already observed failure.
The original event history allows auditors to calculate an alternative sensitivity view.

### Step 15 — Reconcile a production cohort

Reconciliation starts with independent control totals, not with the final rate.
An apparently plausible rate can result from offsetting errors in the numerator and denominator.
The order-management system, promise service, fulfillment system, and carrier feed each need an agreed extraction boundary.
Use the same observation cutoff where the systems support it.

Perform the following checks in sequence:

1. Count accepted transactions in the source acceptance ledger.
2. Remove only documented test, duplicate, or other policy-excluded transactions.
3. Compare the resulting population with the analytics cohort count.
4. Count cohort orders with a linked selected offer.
5. Count selected offers with valid rendered wording and acceptance evidence.
6. Count promises with an interpretable deadline and destination time zone.
7. Count required lines and units for each order.
8. Reconcile required quantities to delivered, canceled, substituted, and open quantities.
9. Check whether each shipment or parcel maps to the correct order requirements.
10. Check event uniqueness using source identifiers and revision sequence.
11. Identify events dated before acceptance or after the observation cutoff.
12. Identify terminal events with missing or incompatible timestamp precision.
13. Classify each accepted order once under the approved precedence policy.
14. Confirm that exclusive status counts sum to the eligible population.
15. Compare the snapshot with prior snapshots to quantify late-arriving changes.
16. Review anomalies by source, service, region, and event type.

The quantity equation must be appropriate to the order model.
A useful control is required quantity = completed quantity + canceled quantity + open quantity, after accounting for approved substitutions.
Do not apply it blindly if a source stores cumulative quantities and another stores increments.
Document the treatment of returns: a return after successful delivery is not automatically a failure of the original delivery commitment.
It may be relevant to customer experience and channel economics, but it is a different event.

Create a reconciliation exception with a named owner and due date.
The source owner should correct the source or provide evidence for a documented analytical adjustment.
An adjustment should be reversible and versioned.
Avoid hand-editing a dashboard result without retaining the correction rationale.
Escalate a data issue when it affects a material number of customers or changes a decision.
If a source is unavailable, publish the affected coverage and avoid a false sense of precision.

### Step 16 — Interpret apparent trends

Suppose a network reports a rise from 92% to 95% on-time performance.
Before concluding that execution improved, inspect the promise lead-time distribution.
If next-day offers became less common and five-day offers became more common, the network may simply have made easier commitments.
Show the rate within comparable promised-lead-time groups and the share of orders in each group.
Also show the overall result, because a customer experiences the offer mix as well as within-group execution.

Inspect population changes in the same way.
A new marketplace, a new region, or a new heavy-goods assortment may have different delivery risks.
Reporting only the total can hide improvement within each segment or create the appearance of improvement from a favorable mix.
If standardizing to a fixed reference mix, publish the weights and show the unadjusted rates too.
Do not standardize into combinations with no observations and then present the result as measured fact.

Inspect the event pipeline before interpreting an abrupt improvement.
A carrier outage may cause late parcels to be unresolved and therefore excluded from a resolved-only rate.
A change in the observation lag may delay the recognition of failures.
A new reason-code rule may reclassify merchant cancellations as customer requests.
A code release may alter the meaning of the delivered event.
Each is a potential measurement change, not necessarily an operating improvement.

Inspect the denominator for business-driven selection.
A merchant may stop offering a fast service in difficult postal areas.
The remaining service population may be more reliable but less representative of all potential customers.
This could be a rational commercial decision, but the performance comparison should say that eligibility changed.
Count suppressed offers and affected demand where permitted.
Do not infer discrimination from geography alone; investigate access and impact using approved methods.

Finally, distinguish a baseline movement from a tested intervention.
Holiday periods, weather, promotional campaigns, and carrier capacity can all move the result.
A before-and-after chart is a useful monitoring device but does not by itself establish causality.
Where a control group is unavailable, document alternative explanations and report the result as an association.

### Step 17 — Design an experiment without changing the estimand

The business may ask whether a more reliable promise is worth its cost.
Define the policy comparison before choosing a randomization unit.
For example, compare the current offer logic with a capacity-aware logic for eligible sessions.
The estimand could be incremental contribution per eligible session over a defined observation window.
That estimand includes customers who choose not to purchase because the displayed promise is slower.
It is different from contribution per delivered order.

Use stable assignment so a customer or session does not alternate between contradictory options.
Capture the assigned policy, the option actually rendered, and the accepted promise separately.
An assignment can fail to render because of a technical error; retain the assignment for an intent-to-treat analysis.
Track that failure as an implementation-quality measure.
Define any per-protocol analysis as secondary because conditioning on exposure can introduce selection bias.

Shared resources complicate the experiment.
Treatment and control orders may consume the same store labor, inventory, carrier capacity, and delivery slots.
Treatment may therefore change the experience of control customers.
Randomizing by region or time block may reduce some spillovers, but can introduce other differences and reduce the number of independent units.
Choose the design with the operations team and an analyst before rollout.
Document the expected interference and the assumptions needed for interpretation.

Predefine the main outcome and the guardrails.
Possible outcomes include contribution per eligible session, conversion, original-promise accuracy, cancellations, and service-recovery cost.
Specify the analysis period and how long outcome data are allowed to mature.
Avoid repeatedly checking a conventional significance threshold and stopping at the first favorable result without an appropriate sequential design.
Plan how to handle missing outcomes and whether the result is reported at session, order, customer, or cluster grain.
If the experiment is too small to resolve the economically relevant difference, say so.
A non-significant result does not demonstrate that the policies are equivalent.

Include an operational stop rule independent of statistical significance.
For example, the control-tower owner may suspend a treatment if a verified capacity incident threatens a material number of commitments.
Define who can stop it, which evidence triggers review, and how affected customers are informed.
Retain those orders in the original assignment analysis.
Safety and customer commitments take precedence over completing an experiment.

### Step 18 — Turn the scorecard into a weekly decision

A useful weekly review begins with customer exposure rather than a league table of teams.
The chair is the cross-functional promise owner.
The attendees include product, fulfillment, carrier management, customer care, data, and finance representatives.
Each attendee brings the same metric version and observation cutoff.

Use a fixed agenda:

1. Review accepted volume, promise speed, and original-promise performance for mature cohorts.
2. Review unresolved and missing-promise counts, including any concentration by source.
3. Compare current results with a like-for-like baseline and the approved target.
4. Review current at-risk commitments and the age of unresolved exceptions.
5. Examine a small number of high-impact, evidence-backed failure pathways.
6. Confirm whether proposed interventions have feasible inventory and capacity.
7. Estimate customer benefit, incremental cost, and potential side effects.
8. Assign an owner, deadline, and measurable closure condition to each action.
9. Escalate cross-functional decisions that no individual team can make.
10. Record decisions and revisit prior actions using mature outcome data.

Example decision rights make the meeting actionable:

| Decision | Accountable role | Required consultation | Evidence before action |
|---|---|---|---|
| Change displayed delivery window | Promise product owner | Commercial, operations, legal as needed | Capacity, conversion, and wording impact |
| Allocate additional store labor | Store operations leader | Store manager and finance | Queue, staffing, and customer exposure |
| Change carrier service | Carrier management leader | Fulfillment and finance | Lane performance, cutoff, and incremental cost |
| Correct event mapping | Data engineering lead | Source owner and analytics | Sampled evidence and historical impact |
| Issue broad customer communication | Customer-experience leader | Legal and operations | Affected population and verified alternative |
| Approve recurring investment | General manager | Finance and functional owners | Counterfactual, sensitivity, and ownership |

The accountable role is responsible for a decision, not necessarily for executing every task.
Record who can authorize exceptions and who pays for the incremental cost.
A carrier-management team cannot fix a warehouse cutoff it does not control.
A store manager cannot repair a broken checkout promise renderer without engineering support.
When a decision spans functions, the promise owner arbitrates trade-offs rather than pushing the miss out of one team's denominator.

## 6. Worked example

### Situation and commitment

A fictional retailer sells home goods in one delivery region.
During the acceptance week beginning 4 May 2026, it accepts 200 delivery orders.
The checkout offers delivery by 8 May 2026 local time.
The merchant's documented interpretation is completion before 00:00 on 9 May in the destination time zone.
The customer-facing promise is a commitment; a transport model's 95% delivery prediction is a forecast; a 96% on-time objective is a target; a scheduled carrier pickup is a plan.

The order ledger is built as of 15 May after the original deadlines and a seven-day event-arrival allowance.
All 200 orders have captured promise text and a known deadline.
This example assumes no customer-requested cancellations and no unknown eligibility.
Five orders still lack a valid final outcome despite the allowance.

| Original outcome status | Orders | Treatment in strict all-accepted share |
|---|---:|---|
| All required items delivered by original deadline | 176 | On time |
| Completed after original deadline | 13 | Not on time |
| Merchant canceled before completion | 6 | Not on time |
| Unresolved as of 15 May | 5 | Unknown; retained in denominator |
| Total accepted | 200 | Denominator |

The strict on-time share is 176 / 200 = **88.0%**.
It is a conservative share of all accepted orders, not a statement that all five unresolved orders were late.
The resolved service rate is 176 / (176 + 13 + 6) = 176 / 195 = **90.26%**, rounded to two decimal places.
The five unresolved orders are not silently removed from the headline.
The known on-time lower bound is 176 / 200 = **88.0%**.
If every unresolved order turns out to have been delivered on time, the upper bound is 181 / 200 = **90.5%**.
Even that upper bound is below the illustrative 96% target.
The interval is not a 95% confidence interval; it reflects only the specified unresolved-outcome uncertainty.

### Grain and a split shipment

Order A contains two lines in two parcels.
Parcel A1 is delivered at 16:00 local time on 8 May.
Parcel A2 is delivered at 10:00 local time on 9 May.
The whole-order commitment requires both lines by the deadline.
Therefore order A is late even though one of its two parcels is on time.
A parcel-level rate of 1 / 2 = 50% for this order is not an order-level outcome.
If the customer had accepted separate line-specific promises, each line would be evaluated against its own captured commitment instead.

Order B is picked and packed on 7 May but the merchant cancels on 8 May because the only item is damaged.
Counting warehouse dispatch performance alone would omit this customer failure.
Order B remains a merchant-canceled order in the 200-order cohort.

Order C has a carrier delivery scan at 20:30 on 8 May, ingested at 06:00 on 10 May.
It is on time by event time and appears in the 15 May as-of report.
A report generated on 9 May could not know that outcome and should show it as unresolved at that earlier observation cutoff.
A later correction to the carrier event would be recorded as a new version, not a silent rewrite of the historical report.

### Diagnostic decomposition

The following exclusive categories are assigned using the earliest verified operational exception; they do not prove a unique cause.
They cover the 19 known failures, not the five unresolved orders.

| First verified exception | Known failures | Example control |
|---|---:|---|
| Inventory could not be picked | 6 | Verify stock and reserve before confirming |
| Missed carrier handover cutoff | 5 | Add capacity or move the cutoff earlier |
| Carrier transit exception after timely handover | 5 | Review lane capacity and backup service |
| Address or access issue not resolved | 2 | Validate contact and initiate exception workflow |
| Other verified issue | 1 | Review individually |
| Total | 19 | Do not extrapolate to unresolved orders |

The six inventory failures include the six merchant cancellations in this illustrative table.
The other 13 are late deliveries.
The association points first to inventory and cutoff investigations; it does not establish how many misses a proposed fix will prevent.
Reconcile every reason assignment to an order and allow an unknown reason when evidence is insufficient.

### A revised date

Suppose 10 of the 13 late deliveries received an updated date before the original deadline.
Eight of those 10 then met the revised date.
The revised-date rate is 8 / 10 = 80% for orders with that notice, not 80% of the original cohort.
None of the 13 becomes on time against its original promise because of the revision.
Report notice coverage as 10 / 13, or 76.92%, for the late-delivered subgroup; this is retrospective and excludes cancellations and unresolved orders.

### Pickup comparison

A separate store accepts 40 BOPIS orders promising ready by 18:00 local time.
Thirty-six have verified staged goods and a ready notification by 18:00.
Two become ready after 18:00 and two are merchant-canceled.
The ready-on-time share is 36 / 40 = 90%.
A customer collects one of the 36 on-time orders the next morning; that does not change its ready outcome.
Do not combine this 90% with the delivery percentage without labeling their different customer events.

### Illustrative business decision

The retailer is considering a verified-inventory and earlier-handover intervention.
Assume a future month has 10,000 accepted orders under either policy and comparable mix.
The current policy is expected to produce a 10% failure rate; the intervention is expected to reduce that to 7%.
These are assumptions, not estimates inferred from the 200-order example.
The difference is 3 percentage points, or 300 fewer failures per month.

Assume each avoided failure saves an average of €8 in incremental credits and contacts, with no overlapping cost categories.
Gross service-recovery saving is 300 × €8 = €2,400 per month.
The intervention adds €0.18 per accepted order in variable labor and transport cost, or €1,800 per month.
It also costs €400 per month to operate the data and control process.
Net monthly contribution change before implementation cost and conversion effects is €2,400 − €1,800 − €400 = **€200**.

At a 1-percentage-point improvement, savings are 100 × €8 = €800 and the change is €800 − €1,800 − €400 = **−€1,400 per month**.
At a 5-percentage-point improvement, savings are 500 × €8 = €4,000 and the change is **€1,800 per month**.
The break-even reduction is €2,200 / (€8 × 10,000) = 0.0275, or **2.75 percentage points**.
If implementation costs €12,000 once and the €200 monthly result persists, undiscounted simple payback is 60 months.
That ignores discounting, uncertainty, demand changes, and future changes in cost.
A general manager should not approve the investment on this base case alone without considering strategic benefits and a credible test.

Do not add avoided refunds as savings if the €8 already includes them.
Do not add revenue from customers who would have ordered anyway.
If the promise becomes slower, model any conversion loss separately, including contribution rather than gross sales.
If reliability increases retention, count it only with credible incremental evidence and avoid double counting future contribution in another valuation.

## 7. Code, query, or calculation

The following standalone Python 3 example uses only the standard library.
It computes the delivery example at order grain from already validated, as-of status records.
It deliberately requires an explicit status for every accepted order.
It does not parse timestamps, deduplicate carrier events, determine eligibility, or validate customer claims; those are upstream controls described above.
The example encodes the status counts as compact input and expands them to order records so the denominator is visible.

```python
from collections import Counter

# Snapshot: cohort accepted 4–10 May 2026; report as of 15 May 2026.
# Each record is one accepted delivery order and has a captured promise.
status_counts = {
    "on_time": 176,
    "late": 13,
    "merchant_canceled": 6,
    "unresolved": 5,
}
orders = [
    {"order_id": f"{status}-{index:03d}", "status": status}
    for status, count in status_counts.items()
    for index in range(count)
]
allowed = {"on_time", "late", "merchant_canceled", "unresolved"}
if any(row["status"] not in allowed for row in orders):
    raise ValueError("Unknown status: revise the classification contract")
if len({row["order_id"] for row in orders}) != len(orders):
    raise ValueError("Order IDs must be unique at the order grain")

counts = Counter(row["status"] for row in orders)
total = len(orders)
if not total:
    raise ValueError("Empty cohort has no meaningful rate")
on_time = counts["on_time"]
resolved = on_time + counts["late"] + counts["merchant_canceled"]
unresolved = counts["unresolved"]
strict = on_time / total
resolved_rate = on_time / resolved if resolved else None
upper = (on_time + unresolved) / total

print(f"accepted={total}; resolved={resolved}; unresolved={unresolved}")
print(f"strict_on_time={strict:.2%}")
print(f"resolved_on_time={resolved_rate:.2%}" if resolved_rate is not None
      else "resolved_on_time=undefined")
print(f"outcome_bounds=[{strict:.2%}, {upper:.2%}]")

monthly_orders = 10_000
failure_reduction = 0.03  # absolute rate change, not 3% of failures
saving_per_avoided_failure = 8.0  # EUR
incremental_cost_per_order = 0.18  # EUR
monthly_fixed_cost = 400.0  # EUR
avoided = monthly_orders * failure_reduction
net = (avoided * saving_per_avoided_failure
       - monthly_orders * incremental_cost_per_order
       - monthly_fixed_cost)
break_even = ((monthly_orders * incremental_cost_per_order + monthly_fixed_cost)
              / (monthly_orders * saving_per_avoided_failure))
print(f"avoided_failures={avoided:.0f}/month; net=EUR {net:.2f}/month")
print(f"break_even_reduction={break_even:.2%}")
```

Expected output:

```text
accepted=200; resolved=195; unresolved=5
strict_on_time=88.00%
resolved_on_time=90.26%
outcome_bounds=[88.00%, 90.50%]
avoided_failures=300/month; net=EUR 200.00/month
break_even_reduction=2.75%
```

### Production adaptation

Read actual records from a governed order ledger rather than the example dictionary.
Calculate completion from required components before joining the result to the one-row-per-order cohort.
Treat a missing deadline or contradictory delivery event as a separate evidence exception.
Maintain an as-of cutoff on both source events and subsequent corrections.
Add customer-canceled and unknown-promise statuses under the organization's published metric contract.
A customer-canceled order cannot simply be included in the upper bound unless the bound's assumptions define it as an uncertain success.
Test time-zone conversions across daylight-saving transitions and compare boundary instants using the promised precision.
Validate the code against hand-calculated records covering splits, partial cancellations, late scans, duplicate events, and revisions.

## 8. Data, standards, and regulatory context

### A minimal data contract

| Dataset | Grain | Critical fields | Control |
|---|---|---|---|
| Promise offer | One rendered offer version | text, time, locale, option, deadline, version | Render-to-selection linkage and capture coverage |
| Order acceptance | One accepted transaction | order ID, accepted instant, selected offer, service | Unique order ID and source reconciliation |
| Order requirement | One line or unit requirement | ordered quantity, product, accepted substitutions | Reconcile required units to fulfillment |
| Allocation | One assignment version | order line, node, quantity, validity interval | Preserve original and rerouted assignment |
| Parcel mapping | One component-to-parcel link | parcel ID, line or unit quantity | No duplicate quantity across active parcels |
| Fulfillment event | One source event | type, event time, ingestion time, source, revision | Deduplication and source provenance |
| Communication | One message or rendered revision | version, channel, send or render time, status | Distinguish sent, rendered, and read |
| Cancellation | One status change | scope, reason, request source, effective time | Preserve unknown reason and audit changes |
| Pickup readiness | One store readiness event | staged quantity, notification, store hours | Reconcile physical readiness and message |
| Customer contact | One contact | protected order key, reason, contact time | Minimize sensitive free text |

Use a contract owner, schema version, freshness objective, access classification, and incident procedure for every feed.
Define nullability, uniqueness, allowed status transitions, timestamp precision, and late-arrival handling.
A data-quality dashboard should show both count and share affected by each defect.
A 99.9% populated tracking field may still be misleading if failures are concentrated in a particular carrier or service.

### Identifiers and traceability

GS1 provides standards for identifying objects and capturing and sharing supply-chain event information.
EPCIS 2.0 specifies a way to represent traceability events including what, when, where, why, and how.
Adopting an event standard can improve interoperability; it does not by itself prove that an event occurred or that a customer saw a promise.
Use the applicable standard and trading-partner agreements where relevant rather than assuming every carrier emits the same fields.
The GS1 source in section 16 describes EPCIS; the merchant still needs an order-to-parcel mapping and a documented customer metric.

### Personal data and jurisdiction

A delivery address, tracking history linked to a person, customer contact, and device identifiers can be personal data.
In the European Union, the General Data Protection Regulation (GDPR) sets principles including purpose limitation, data minimization, accuracy, storage limitation, and integrity and confidentiality; see Article 5.
The lawful basis and obligations depend on the processing activity and circumstances, not on a generic analytics label.
Privacy and legal teams should determine applicable retention, access, data-subject rights, transfer, and vendor controls.
A tokenized order ID can remain personal data if it can be linked back to a person.

Apply least-privilege access, encryption where appropriate, retention schedules, and auditable joins.
Prefer a coarse service region in broad analytics views and restrict precise addresses and customer messages.
Suppress small groups where identification risk is material, especially when slicing by location or accessibility needs.
Do not infer a protected characteristic from address or disability-related service interactions for performance targeting.

### Consumer-facing claims

The U.S. Federal Trade Commission's Mail, Internet, or Telephone Order Merchandise Rule addresses shipping representations and delay-related obligations for covered merchandise orders.
Its applicability, permitted notice and consent mechanisms, and remedies depend on the facts and governing law.
An internal on-time metric is not a legal determination of compliance.
The EU Consumer Rights Directive also addresses delivery and remedies in its applicable scope; national implementation and exceptions matter.
Route proposed promise wording and customer-change workflows through counsel for each market.
Avoid claiming that one jurisdiction's rules automatically apply worldwide.

### Clock and calendar controls

Store an absolute instant and a named local time zone where possible.
An offset alone does not describe future daylight-saving changes.
Use a maintained time-zone database and store the calendar version used for business-day commitments.
Keep observed event precision: a date-only scan cannot establish whether delivery occurred before a 15:00 promise.
Monitor clock skew, source time-zone defaults, and events that appear to precede order acceptance.

## 9. Industry practice and case evidence

### Customer-facing tracking

UPS describes tracking statuses and delivery progress in its public tracking support materials.
The existence of a carrier tracking interface demonstrates a source of operational status, not that every scan is complete or proves receipt.
A merchant integrating the feed still needs to map a tracking number to the right order components and preserve the commitment that preceded shipment.
**Inference:** comparing the retailer's original promise with a later carrier tracking estimate requires retaining both histories rather than replacing one with the other.

USPS publishes service performance information and explains its measurement approach for mail and package services.
Its published service measures concern the postal network and their stated service standards and measurement methods.
They are not automatically the same as a retailer's checkout promise, which can also include processing before carrier acceptance.
**Inference:** a retailer should reconcile carrier service performance with its own order-level experience rather than use either number as a proxy for the other.

### Store fulfillment and pickup

Target describes order pickup and drive-up services in its public customer guidance, including notification when an order is ready.
This documents a customer-facing process, not a quantified performance result for the example in this lesson.
**Inference:** the ready message, staged product, and store hours should be joined when evaluating a merchant's pickup-ready promise.
Customer collection and store readiness remain separate operational events.

### Evidence hierarchy

Carrier documentation can establish the meaning of published statuses but not the truth of every event.
Company help pages describe a process but may change and do not establish causal impact.
Internal logs can establish what a system recorded, subject to capture and measurement error.
A randomized experiment can support a causal comparison if assignment, exposure, outcome capture, and interference are addressed.
An observational association can prioritize investigation but may reflect differences in geography, inventory, product, or customer mix.

Do not transfer a carrier's published service level to a merchant's checkout conversion model without checking definitions and population.
Document the source date and retrieve current guidance before relying on operating or legal details.
The cases here support measurement design, not an asserted universal benchmark.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Promise overwritten | Most orders have only the newest date | Revision replaces accepted version | Original misses disappear | Immutable versioned offers and audit trail |
| Offer not actually rendered | Server counts exceed client render events | Calculated option differs from visible option | Wrong customer commitment | Link rendered selection to order |
| Missing hard orders | Capture rate drops for disrupted carriers | Missingness is correlated with lateness | Inflated accuracy | Publish coverage and outcome bounds |
| Cancellations disappear | Accepted count exceeds evaluated count | Only delivered orders enter metric | Service failure hidden | Cohort from acceptance and explicit statuses |
| Parcel/order confusion | More metric rows than orders | One-to-many join duplicates orders | Denominator distortion | Aggregate required components before order join |
| Partial order credited | First parcel delivered promptly | Latest required component ignored | Whole-order miss called success | Complete only after all required components |
| Wrong local day | Midnight errors cluster in time zones | UTC date compared to local date | False late or on-time classification | Convert using named zone and precision |
| Late-arriving events ignored | Recent cohorts improve after restatement | Ingestion time substituted for event time | Spurious trend and false alarms | As-of snapshots and maturity lag |
| Customer change assumed | Many misses assigned to customer request | Unverified revision categorized as consent | Inflated original-promise metric | Capture request, channel, and agreement |
| Pickup ready only on paper | Customer complaints despite strong ready rate | Message issued before goods are staged | Wasted customer trip | Require verified physical readiness |
| Store hours ignored | Ready messages arrive after closing | Timestamp passes while service is inaccessible | Customer cannot collect as promised | Apply commitment wording and store calendar |
| Attempts counted as delivery | High attempt rate but nonreceipt contacts | Attempt event used as proof of completion | Missed handoffs hidden | Separate attempted and delivered events |
| Revision treated as success | Revised accuracy rises while original falls | Moving deadline resets score | Incentive to communicate instead of fix | Report original and revised rates separately |
| Slower promises rewarded | On-time rate rises as lead time increases | Optimization ignores offer attractiveness | Lost sales and customer value | Pair speed, conversion, and accuracy |
| Aggressive promises rewarded | Conversion rises and credits spike | Commercial team ignores execution risk | Profit and trust deteriorate | Shared scorecard and failure-cost guardrail |
| Blame assigned by final carrier | Late parcels were handed over after cutoff | Stage responsibility conflated | Wrong corrective action | Compare event stages with internal plan |
| Small cells overinterpreted | Extreme rates from few orders | Sampling noise or correlation | Unnecessary operational changes | Show counts, pool cautiously, review evidence |
| False economic savings | Credits and refunds counted twice | Overlapping cost categories | Inflated investment case | Finance-owned cost dictionary |
| Customer data overexposed | Address and contact text in broad dashboard | No minimization or role controls | Privacy and security harm | Protected keys, access controls, retention |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or manage |
|---|---|---|---|
| Selected offer is captured | Establishes actual commitment | Client event fails or option changes | Reconcile selection and confirmation; report capture |
| Acceptance identifies population | Prevents outcome-selected cohorts | Duplicate or rejected orders counted | Reconcile to order-management acceptance ledger |
| Promise can be interpreted | Converts wording into a deadline | Ambiguous range, missing zone, special terms | Review wording and flag unparseable cases |
| Required components are known | Defines complete order | Split, substitution, or cancellation is unlinked | Reconcile quantities from order to parcels |
| Events reflect physical activity | Supports outcome classification | Mis-scans, fraud, or stale feeds | Audit samples against disputes and source evidence |
| Event time is trustworthy | Compares event to deadline | Clock skew or incorrect time zone | Check impossible sequences and source offsets |
| Observation cutoff is reproducible | Allows consistent restatement | Corrections overwrite history | Version source events and snapshot as-of state |
| Missing outcomes are bounded | Quantifies unresolved uncertainty | Missing promises or eligibility also unknown | Add separate sensitivity cases and coverage |
| Statuses are mutually exclusive | Prevents double counting | A cancellation and delivery share an order | Define precedence and retain underlying evidence |
| Cause labels are reliable | Prioritizes corrective action | Reason codes are mandatory guesses | Audit samples; allow unknown and multiple causes |
| Comparison groups are comparable | Supports performance diagnosis | Mix shifts across regions or products | Stratify and standardize with common support |
| Treatment effect is incremental | Supports business case | Selection or seasonality explains the change | Randomize or use a defensible counterfactual |
| Unit economics persist | Converts service benefit to money | Costs or volumes vary with scale | Run low/base/high scenarios and monitor |
| One outcome fits the promise | Makes a rate meaningful | Pickup and delivery promise different events | Publish service-specific measures |
| Customer harm is observable | Measures experience beyond scans | Nonreceipt or inaccessible handoff is unreported | Combine evidence with claims and survey channels |
| Rules apply to the relevant market | Guides communication and retention | Local law or contract differs | Obtain jurisdiction-specific legal review |

A mature cohort can still have missing or disputed outcomes.
A precise percentage cannot repair a biased population.
Service metrics are descriptive unless a design supports causal claims.
The €8 per avoided failure in the example is an assumption, not an industry estimate.

## 12. Exercises

### Exercise 1 — Capture the accepted commitment

A mobile checkout shows “arrives Friday,” then the customer changes the address and selects “Saturday by 18:00.”
The order confirmation says Saturday by 18:00 in the destination time zone.
Two days later, the merchant emails an estimate of Sunday.

**Deliverable:** List the immutable promise and communication records and identify the primary customer commitment.
**Acceptance criteria:** Distinguish both rendered offers, the accepted option, and the later estimate; preserve time zone and event times; do not treat an email as consent.
**Time:** 15 minutes.

### Exercise 2 — Cohort arithmetic

A mature acceptance cohort contains 500 orders: 440 on time, 30 late, 10 merchant-canceled, and 20 unresolved.
Assume all have captured deadlines and no customer cancellations.

**Deliverable:** Compute the strict on-time share, resolved-only rate, and lower and upper outcome bounds.
**Acceptance criteria:** Show denominators, retain unresolved orders in the strict population, and explain what the bounds do and do not mean.
**Time:** 15 minutes.

### Exercise 3 — Split order and time zone

An order promises all items by 12 June local time, interpreted as before the start of 13 June.
One parcel arrives at 17:30 on 12 June and the second at 00:15 on 13 June in the same local time zone.
A dashboard marks the order on time using the first parcel and UTC calendar dates.

**Deliverable:** Classify the whole order and describe the correction.
**Acceptance criteria:** Use the last required component, preserve local-time interpretation, and explain why UTC date alone is insufficient.
**Time:** 15 minutes.

### Exercise 4 — Diagnose a fulfillment gap

A carrier is blamed for 50 late orders.
Of those, 30 were handed over after the merchant's planned carrier cutoff, 12 were handed over on time but experienced transit exceptions, and eight lack adequate event evidence.

**Deliverable:** Propose an investigation and two immediate controls.
**Acceptance criteria:** Do not assign the eight unknowns to the carrier; distinguish association from cause; name an owner and evidence for each control.
**Time:** 20 minutes.

### Exercise 5 — Live control design

A network operates high-volume and low-volume stores with different cutoff times.
Its dashboard shows only a monthly average of delivered parcel speed.

**Deliverable:** Design a one-page control specification for customer promise exposure.
**Acceptance criteria:** Include cohort, grain, deadline, maturity, unresolved queue, a count-aware trigger, owner, escalation, and a guardrail against extending all promises.
**Time:** 25 minutes.

### Exercise 6 — Incremental business case

An intervention handles 20,000 accepted orders per month.
Assume it reduces failure rate by 2 percentage points, saves €6 per avoided failure, costs €0.07 per order, and adds €500 fixed monthly cost.

**Deliverable:** Calculate monthly net contribution change and the break-even absolute failure-rate reduction.
**Acceptance criteria:** Use incremental costs once, show arithmetic and units, and state one major uncertainty.
**Time:** 20 minutes.

### Exercise 7 — Experiment and incentives

The commercial team proposes a faster promise in half the eligible regions.
Store teams are paid only on ready-on-time percentage and may widen the offered window.

**Deliverable:** Propose an experimental comparison and a balanced incentive scorecard.
**Acceptance criteria:** Define assignment, outcome denominator, conversion and cost measures, interference risks, original-promise accuracy, and a fairness or privacy safeguard.
**Time:** 30 minutes.

### Exercise 8 — Data and governance review

A vendor proposes sending raw addresses, full contact transcripts, and unversioned delivery estimates to a shared dashboard indefinitely.

**Deliverable:** Write a concise approval or rejection memo and specify a safer data contract.
**Acceptance criteria:** Address minimization, role access, retention, immutable accepted promise, event evidence, capture coverage, and jurisdiction-specific review.
**Time:** 20 minutes.

## 13. Self-check answers

### 1. Accepted commitment

Retain the Friday offer and its render time as the earlier displayed option.
Retain the Saturday option, its rendered wording, destination time zone, selection, and order acceptance.
The Saturday by 18:00 commitment is the primary accepted commitment in this scenario.
Retain the Sunday email as a later estimate, including send and delivery status if known.
The email is not proof the customer accepted a revised commitment.
Measure the original Saturday commitment separately even if the merchant later reports performance against Sunday.

### 2. Cohort arithmetic

Strict share = 440 / 500 = **88.0%**.
Resolved-only share = 440 / (440 + 30 + 10) = 440 / 480 = **91.67%**.
Lower bound = 440 / 500 = **88.0%**.
Upper bound = (440 + 20) / 500 = **92.0%**.
The bounds assume the unresolved outcomes are the only uncertainty and the 40 known late or merchant-canceled orders are failures under this policy.
They are not confidence intervals and do not describe uncertainty about missing promises or eligibility.

### 3. Split order

The order is late because the second required parcel arrived after the local deadline.
The first parcel's timely arrival does not complete the all-items promise.
Join parcel events to required line quantities, take the latest valid completion, and compare it with the exclusive local deadline converted to an absolute instant.
A UTC calendar date without the destination time zone can move an event across a local date boundary.
If a parcel event is only date-precision, do not invent a time for a timed promise.

### 4. Diagnosis rubric

A strong answer investigates why 30 orders missed handover and tests whether the planned cutoff was itself feasible.
The warehouse or store operations lead owns a queue and capacity review, with pick, pack, and handover timestamps as evidence.
The carrier-management lead investigates the 12 timely handovers with transit exceptions, using tracking and lane records.
The data-integration lead resolves the eight unknowns rather than assigning them to either group.
A good answer considers multiple contributing causes and does not claim a verified causal effect without testing.

### 5. Control-design rubric

A complete proposal starts with accepted-order cohorts and uses the original customer deadline at order grain.
It states observation cutoff and maturity, retains canceled and unresolved orders, and distinguishes pickup from delivery.
It assigns an owner to approaching-deadline and overdue queues, with explicit escalation and closure evidence.
It uses counts and appropriate aggregation for low-volume stores rather than a fragile percentage alone.
It pairs on-time performance with promise lead time, conversion, and cost so widening the window is visible.

### 6. Business case

Avoided failures = 20,000 × 0.02 = **400 orders per month**.
Savings = 400 × €6 = **€2,400 per month**.
Variable cost = 20,000 × €0.07 = **€1,400 per month**.
Net = €2,400 − €1,400 − €500 = **€500 per month**.
Break-even reduction = (€1,400 + €500) / (20,000 × €6) = 0.015833, or approximately **1.58 percentage points**.
The assumed causal failure reduction, unit saving, and possible effect on conversion all require evidence.
The calculation excludes one-time implementation cost and taxes.

### 7. Experiment and incentives rubric

Randomize eligible comparable units where feasible and record the assigned offer even when a customer does not order.
Use eligible sessions for conversion and accepted orders for original-promise performance, with clear linkage and exposure measurement.
Compare incremental contribution, service recovery, lead time, cancellations, and capture coverage by assigned policy.
Check shared inventory, carrier capacity, and customers crossing regions for interference; use clusters or another justified design when necessary.
Do not condition the causal analysis only on completed deliveries.
Limit collection and report small-cell results under an approved privacy policy.
Tie store incentives to controllable readiness while a cross-functional owner holds the end-to-end result.

### 8. Governance rubric

Reject indefinite, broadly accessible raw data as proposed pending a lawful and proportionate design review.
Use a protected order key, service region where sufficient, role-specific access, and a documented retention period.
Preserve immutable displayed and accepted promise versions and separate later estimates.
Document event time, ingestion time, source, corrections, capture coverage, and unresolved outcomes.
Seek legal and privacy review for applicable markets and vendor transfers.
A privacy-preserving design does not require deleting the evidence needed to assess the commitment; it requires retaining the minimum necessary evidence under controlled access.

## 14. Interview questions

1. **How would you define promise accuracy?** Strong: names original customer commitment, relevant completion event, grain, denominator, and cutoff. Weak: says delivered parcels divided by shipped parcels without dates.
2. **What if the promise is revised?** Strong: retains original and reports revised performance separately. Weak: replaces the original deadline.
3. **How do you handle unresolved orders?** Strong: reports counts, maturity, conservative shares, and appropriate bounds. Weak: drops all missing outcomes.
4. **What is the difference between a forecast and a commitment?** Strong: distinguishes a probabilistic estimate from customer-facing wording. Weak: treats any ETA as an accepted promise.
5. **How do split shipments affect the metric?** Strong: maps required quantities to parcels and uses the last required completion for an all-items promise. Weak: credits the first parcel.
6. **How would you measure BOPIS?** Strong: distinguishes staged goods and ready notice from customer collection. Weak: calls collection time the ready time.
7. **Why might a carrier be blamed incorrectly?** Strong: checks handover against cutoff and separates transit from fulfillment delay. Weak: assigns all late deliveries to the last carrier.
8. **Can accuracy improve while customer value falls?** Strong: explains that slower promises can inflate the percentage and proposes speed and conversion guardrails. Weak: equates a higher percentage with a better offer unconditionally.
9. **How would you evaluate a reliability investment?** Strong: uses incremental contribution, avoided failure costs, counterfactual, and sensitivity. Weak: multiplies gross sales by a guessed retention uplift.
10. **What can a standard event feed prove?** Strong: describes event semantics, source quality, and limits of proof of receipt. Weak: assumes any scan proves customer satisfaction.
11. **How do you compare two nodes fairly?** Strong: examines volume, mix, time period, uncertainty, and controllable stages. Weak: ranks percentages without denominators.
12. **How would you govern the data?** Strong: includes minimization, provenance, retention, access, and legal review. Weak: assumes hashed identifiers are automatically anonymous.

## 15. Key terms

- [Accepted promise](#4-definitions-and-vocabulary): the selected customer-facing commitment at order acceptance.
- [Allocation](#step-4--build-the-event-and-entity-graph): assignment of a required quantity to a fulfillment node.
- [As-of cutoff](#step-6--choose-the-observation-cutoff-and-maturity-rule): the latest observation time included in a report.
- [ATP](#4-definitions-and-vocabulary): available inventory and supply under defined commitment rules.
- [BOPIS](#step-5--treat-pickup-as-a-different-customer-outcome): online purchase with store collection.
- [Capture coverage](#step-6--choose-the-observation-cutoff-and-maturity-rule): fraction with required promise or outcome evidence.
- [Cohort](#step-3--define-the-population-before-observing-outcomes): population grouped by a fixed event and period.
- [Commitment](#forecast-plan-target-and-commitment): expectation expressed to the customer.
- [Contribution](#step-12--govern-incentives-and-the-business-case): revenue less defined incremental variable costs.
- [Deadline](#step-2--establish-a-canonical-deadline): latest instant consistent with the commitment.
- [DOM](#4-definitions-and-vocabulary): distributed order management across fulfillment nodes.
- [Event time](#four-clocks): when the physical or system event occurred.
- [Forecast](#forecast-plan-target-and-commitment): probabilistic estimate of an outcome.
- [Guardrail](#step-12--govern-incentives-and-the-business-case): constraint protecting against optimization harm.
- [Ingestion time](#four-clocks): when the analytics system received an event.
- [Mature cohort](#step-6--choose-the-observation-cutoff-and-maturity-rule): cohort past the specified observation allowance.
- [On-time rate](#step-7--calculate-rates-bounds-and-uncertainty): on-time count divided by a stated population.
- [Order grain](#step-4--build-the-event-and-entity-graph): one observation per customer transaction.
- [Outcome bounds](#step-7--calculate-rates-bounds-and-uncertainty): limits under explicit assumptions about unknown outcomes.
- [Pickup-ready event](#step-5--treat-pickup-as-a-different-customer-outcome): staged goods and required notification under policy.
- [Plan](#forecast-plan-target-and-commitment): intended schedule and resource allocation.
- [Promise version](#step-1--decide-what-the-customer-was-promised): immutable record of a customer-facing offer or revision.
- [Proof of delivery](#step-4--build-the-event-and-entity-graph): source evidence of delivery, subject to limitations.
- [Resolved-only rate](#step-7--calculate-rates-bounds-and-uncertainty): rate among explicitly defined resolved outcomes.
- [Service recovery](#step-10--separate-original-and-revised-commitments): remedial action after a service failure.
- [Split order](#grain-and-a-split-shipment): order fulfilled across multiple components or parcels.
- [Target](#forecast-plan-target-and-commitment): management objective, not a customer promise.
- [Unresolved](#step-6--choose-the-observation-cutoff-and-maturity-rule): no valid terminal outcome by the reporting cutoff.

## 16. Sources

The examples and calculations in this lesson are illustrative.
The following sources support the specific concepts indicated; none validates the example's numerical assumptions.
Access current versions and jurisdiction-specific guidance before using them operationally.

1. GS1, [EPCIS standard](https://www.gs1.org/standards/epcis): traceability event information and interoperability.
2. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj), especially Article 5: personal-data processing principles.
3. U.S. Federal Trade Commission, [Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/business-guidance/resources/business-guide-ftcs-mail-internet-or-telephone-order-merchandise-rule): shipping representations and delay-related requirements in its scope.
4. European Union, [Directive 2011/83/EU on consumer rights](https://eur-lex.europa.eu/eli/dir/2011/83/oj): consumer rights including delivery provisions, subject to implementation and scope.
5. U.S. Postal Service, [Service performance](https://about.usps.com/what/strategic-plans/service-performance/): published network service performance and methodology context.
6. UPS, [Tracking support](https://www.ups.com/us/en/support/tracking-support.page): carrier tracking information and terminology.
7. Target, [Order Pickup](https://www.target.com/c/order-pickup/-/N-54vrw): description of its customer pickup service.
8. Internet Assigned Numbers Authority, [Time Zone Database](https://www.iana.org/time-zones): database for named time-zone rules.
9. NIST/SEMATECH, [Engineering Statistics Handbook](https://www.itl.nist.gov/div898/handbook/): statistical methods and interpretation of uncertainty.

## 17. Links

- [Module 23 index](_index.md).
- [Module 23 survey and Lab 23.1, ATP and promise](../../../course/23_omnichannel_fulfillment_returns.md#lab-231--atp-and-promise).
- [Lab 23.5, promise failure analysis](../../../course/23_omnichannel_fulfillment_returns.md#lab-235--promise-failure-analysis).
- [Inventory visibility](01_inventory-visibility.md): confidence in stock and event data.
- [Available-to-promise](02_available-to-promise.md): inventory and capacity behind a commitment.
- [Order routing and DOM](03_order-routing-dom.md): selecting a feasible fulfillment path.
