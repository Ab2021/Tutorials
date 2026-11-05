# Peak Planning for E-commerce Fulfillment

> Module 21 — E-commerce Supply Chain Analytics  
> Topic 8 of 10  
> Estimated study time: 4 hours  
> Prerequisites: order-grain forecasting, inventory placement, delivery promise, and carrier selection  
> Version 2026.5

## 1. Why this topic exists

A seasonal promotion can be commercially successful and operationally disastrous at the same time.
Orders arrive faster than a warehouse can release them, a carrier accepts fewer parcels than expected, and customers receive promises that no available route can meet.
The problem is not simply that the annual demand forecast was wrong.
The business must synchronize decisions with different lead times across inventory, labor, facilities, parcel transport, and customer communication.

Peak is a temporary period in which demand, mix, or variability pushes one or more resources near a binding limit.
That period may be Black Friday, a holiday, a product launch, a payday event, or a weather-driven surge.
A peak is defined relative to capacity, not by a particular date.
A modest increase in orders can be a peak when an inbound dock or a carrier pickup is already full.

The commercial team needs a credible answer before it commits media spending or publishes a delivery date.
The fulfillment manager needs to know which shifts to staff and whether temporary workers will be productive by the event.
The carrier manager needs dated, written pickup commitments, not an undated assurance that a carrier can help.
The inventory planner needs product in the correct node before the last receiving window.
The customer-service leader needs a playbook for honest status updates and remedy decisions.
The executive sponsor must decide which trade-offs are acceptable when all objectives cannot be met.

Peak planning therefore combines a forecast with an executable operating plan.
The forecast describes uncertain demand; it is not a promise from the customer.
The plan allocates resources and describes actions under named conditions.
A target expresses desired performance; it does not create warehouse or transport capacity.
A commitment is an authorized, dated obligation, such as an accepted carrier allocation or a customer delivery promise.
Do not treat these four concepts as interchangeable.

An inaccurate unit of measure is expensive during peak.
A warehouse may advertise capacity in units per hour while carrier contracts constrain parcels per pickup.
An order can contain several units and can split into multiple parcels.
A receiving queue can fill even when outbound orders are low.
Customer-service contacts can rise after the physical operation has recovered.
Every constraint needs a unit, a time bucket, a location, and an owner.

The central question is not whether total weekly capacity exceeds weekly orders.
It is whether the right resources are available at the right node before each cutoff, for a sufficiently wide range of plausible demand.
This lesson builds that answer, quantifies backlog and recovery, and turns the analysis into decisions that people can execute.

## 2. Learning outcomes

By the end of this topic you can:

1. **Translate** order scenarios into units, labor hours, and parcels by node and time bucket; exercise 1 checks the conversions.
2. **Compute** effective capacity, utilization, and backlog while respecting the order of fulfillment stages; exercises 2 and 3 check the arithmetic.
3. **Choose** labor, inventory, and carrier contingencies with trigger thresholds and named decision rights; exercise 4 tests the decision.
4. **Evaluate** the feasibility of a delivery promise using the remaining process and transport windows; exercise 5 applies it.
5. **Critique** a readiness dashboard and identify misleading averages or missing evidence; exercise 6 tests this skill.
6. **Design** a peak command cadence and customer-service response that distinguishes operational status from customer commitments; exercise 7 uses a rubric.
7. **Stress-test** coupled failures and describe limits of a deterministic plan; exercise 8 asks for a scenario design.
8. **Reproduce** a small, dependency-free Python capacity model and explain its assumptions; the worked example and code provide the check.

An acceptable outcome includes both a number and the action that number supports.
A utilization calculation without a decision threshold is not yet a control.
A contingency without an owner, activation deadline, and measurable capacity is not an executable plan.

## 3. The core mental model

```text
Demand scenarios by node and day
       |          product mix and order profile
       v
Inventory available -> release -> pick/pack -> carrier handoff -> delivery
       |                  |          |               |             |
       |              order cap   labor + assets   parcel cap    transit
       +------------------+----------+---------------+-------------+
                                      |
                        backlog, lateness, and cost
                                      |
                  triggers -> owners -> authorized actions
                                      |
                 customer promise and service communication
                                      ^
              actual events and exceptions refresh the plan
```

The diagram is a flow network with stocks and constraints.
Inventory and unfinished work are stocks: they carry from one period into the next.
Order arrival, release, pick, pack, and handoff are flows: each is measured over a period.
The flow through any stage cannot exceed both its available work and its usable capacity.
A late process upstream reduces the time remaining downstream.
Capacity that existed yesterday cannot necessarily be stored for use tomorrow.

The bottleneck is conditional on the period and product mix.
Monday may be labor-constrained, Tuesday may be carrier-constrained, and Wednesday may be inventory-constrained.
A weekly total can show spare capacity while a particular pickup misses thousands of parcels.
Likewise, the fastest average warehouse may still fail a cutoff if arrivals are concentrated immediately before it.
Use time buckets short enough to resolve decisions; a day is useful for a first plan, but a shift or pickup window may be required for execution.

The demand scenario supplies the load, not the answer.
The planner converts orders into the appropriate workload at each stage using explicit conversion factors.
Capacity is then adjusted for availability, attendance, training, breaks, and maintenance, without applying the same loss twice.
An executable plan also respects precedence: receiving must precede availability; work must be packed before carrier handoff.

A backlog is not automatically a missed promise.
An order can wait within its promised processing window and still arrive on time.
Conversely, a small backlog of already-promised expedited orders can be more damaging than a larger backlog of flexible orders.
Backlog must therefore be aged and segmented by customer promise, not merely totaled.

The feedback loop is as important as the original plan.
Actual orders, available stock, productive hours, throughput, and accepted parcels update the projection.
A threshold triggers a pre-authorized decision, such as opening a shift or changing the promise for future checkouts.
Changing a future promise does not erase commitments already made.
Customer service needs the affected orders and the expected recovery time, not a vague statement that the network is busy.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Peak | Period when load or variability approaches a practical resource limit; day or shift | Treating a fixed holiday as the only possible peak |
| Forecast | Estimated future demand, expressed as orders per node per time bucket | Calling the forecast an operational commitment |
| Scenario | Internally consistent set of demand and operating assumptions | Multiplying every adverse assumption without checking plausibility |
| Plan | Chosen resource allocation and response for a stated scenario | Assuming a plan has already been contracted |
| Target | Desired performance, such as 98% on-time delivery | Treating an aspiration as physical capacity |
| Commitment | Authorized obligation with scope and deadline | Counting an unconfirmed supplier offer as firm |
| Order | Customer transaction, counted as orders | Assuming one order equals one unit or one parcel |
| Order line | A distinct item entry on an order, counted as lines | Confusing lines with units |
| Unit | One physical sellable item, counted as units | Using units as if they were parcels |
| Parcel | Package tendered to a carrier, counted as parcels | Assuming a split order always creates one parcel |
| Workload | Resource consumption implied by demand, such as standard labor hours | Comparing workload directly with orders without conversion |
| Standard time | Expected labor time per work unit under stated conditions, seconds per unit | Applying an experienced-worker standard to untrained staff |
| Productive hour | Paid hour available for the specified work after defined losses | Deducting breaks twice |
| Nominal capacity | Theoretical maximum before stated losses, units per hour | Treating equipment nameplate speed as reliable output |
| Effective capacity | Achievable output under stated staffing, mix, and availability, units per bucket | Assuming all planned workers attend and perform at standard |
| Utilization | Load divided by available capacity in the same unit and bucket | Claiming a 100% utilization target leaves recovery margin |
| Bottleneck | Constraint limiting additional end-to-end throughput in a period | Assuming the same bottleneck persists all peak |
| Headroom | Capacity less expected load, in the same unit and bucket | Calling uncontracted contingent capacity firm headroom |
| Buffer | Deliberate inventory, time, capacity, or cash protection against uncertainty | Adding a buffer without identifying what it protects |
| Backlog | Accepted work not completed at a defined stage and cutoff, orders or parcels | Equating all backlog with late orders |
| Aging | Breakdown of backlog by elapsed time or promise deadline | Reporting only the total backlog |
| Release | Authorization for an order to enter fulfillment | Assuming payment acceptance means inventory has been allocated |
| Pick | Retrieval of items from storage for an order | Counting a picked item as a shipped order |
| Pack | Placing goods into shipment-ready packaging | Assuming packed parcels have been accepted by a carrier |
| Tender | Presentation of parcels to a carrier for acceptance | Treating a label scan as carrier acceptance |
| Cutoff | Latest time an action can qualify for a specified downstream window | Applying a carrier cutoff to every service and destination |
| Allocation | Quantity of inventory or transport capacity reserved under a rule | Assuming every allocation is contractually guaranteed |
| Absenteeism | Scheduled workers not present, expressed as a share of scheduled workers | Subtracting absent workers and then applying attendance again |
| Learning curve | Relationship between experience and productivity under a defined task mix | Assuming all new hires achieve standard output immediately |
| Service level | Precisely defined share of eligible orders meeting a service outcome | Omitting the denominator or whether the promise changed |
| Recovery time | Time until backlog and service return to an agreed state | Equating a stopped backlog increase with full recovery |
| Trigger | Observable condition that starts a specified decision process | Using a threshold that is measured after the action deadline |
| Lead time | Time between an action and its usable effect | Ignoring hiring, training, linehaul, or carrier onboarding time |
| Readiness | Evidence that required resources and controls can operate as planned | Treating a meeting attendance list as a test |
| Contingency | Specific alternate action with capacity, cost, timing, and owner | Counting mutually exclusive options as simultaneous capacity |
| Promise | Customer-facing expected delivery commitment, with its eligible conditions | Silently extending existing customer promises |
| WISMO | “Where is my order” customer contact, counted as contacts | Assuming every late parcel produces exactly one contact |
| Force majeure | Contract-defined treatment of specified extraordinary events | Assuming it automatically excuses any service failure |

## 5. The mechanics, step by step

### Step 1 — Set the decision and the planning horizon

**Input:** event calendar, historical order curves, commercial calendar, and current commitments.

**Action:** identify the peak event and work backward from the customer delivery deadline.
List the last practical dates for procurement, inventory transfer, hiring, training, carrier allocation, and checkout promise changes.
Identify a planning bucket for each decision.
A multi-month purchase decision can use weeks, while pickup decisions need days or hours.

**Output:** a dated decision calendar and a named accountable executive.

**Branch:** if the lead time of a required option has already expired, remove it from the available contingency set.
Do not describe an option as available merely because it was feasible in an earlier planning cycle.
A late purchase order may still help recovery, but it cannot protect a date it cannot reach.

Assign one owner to the integrated plan.
Commercial owns promotional inputs and the decision to change campaign exposure.
Inventory planning owns item availability and transfer dates.
Site operations owns safe capacity and shift execution.
Transport procurement owns carrier commitments and cutoff interpretation.
Customer experience owns customer messaging and remedy policy.
Finance validates incremental economics and delegated authority.
The executive sponsor resolves conflicts; no single functional forecast should silently determine the outcome.

### Step 2 — Define service and guardrails

**Input:** customer promise rules, service history, contracts, and financial policy.

**Action:** specify the metric numerator, denominator, event timestamp, exclusions, and reporting lag.
For example, on-time delivery can mean orders delivered by the date shown at checkout divided by all orders with a delivery commitment due in the measured period.
If the metric is parcel-based instead, say so: a multi-parcel order can fail if one parcel arrives late.
Track promise changes separately from actual performance to avoid improving a metric by extending future dates.

**Output:** a service objective and explicit operating guardrails.

Examples include a maximum backlog age, a cap on overtime, a minimum safe staffing level, and a limit on discretionary transport cost.
A service objective is not an instruction to violate safety policy or applicable working-time rules.
The sponsor decides which trade-off has priority when constraints conflict.

### Step 3 — Build a demand profile, not a single total

**Input:** historical orders by timestamp and node, promotion plan, product launch calendar, and known channel changes.

**Action:** create a baseline and at least one credible high-demand case.
Separate the number of orders from units per order, parcels per order, destination mix, expedited share, and product handling mix.
Use consistent time zones and define whether timestamps refer to customer order placement, payment approval, or warehouse release.
Exclude or separately model cancellations and fraud holds rather than silently treating every order as released work.

**Output:** order arrivals and workload drivers by node, bucket, and scenario.

**Branch:** if the promotion is new and has little history, use a range and analogs with documented adjustments.
Do not claim a precise probability without an empirical basis.
If a promotion shifts orders between days rather than creating incremental demand, include the offset in later buckets.
Do not apply a promotional multiplier to a forecast that already contains the uplift.

Model the relationship among assumptions.
A high-demand case might also have more multi-unit orders because a bundle promotion drives both.
A carrier outage is a distinct stress unless evidence supports coupling it with that demand case.
A deliberately severe combined stress can be useful, but label it a stress test rather than an estimated percentile.

### Step 4 — Translate demand into stage-specific load

**Input:** scenario orders, units per order, lines per order, standard times, and parcels per order.

**Action:** multiply orders by the appropriate factor for each stage.
For 1,000 orders at 1.6 units per order, the pick load is 1,600 units before considering inventory constraints.
At 1.15 parcels per order, carrier demand is 1,150 parcels, not 1,000 parcels.
If parcel splitting varies by node or inventory availability, use a segmented factor.

For labor planning, multiply each workload type by its time standard and divide by 60 minutes per hour.
Include setup, replenishment, packaging, exception handling, and indirect work either in the standards or as separate capacity deductions, not both.
For transport, distinguish shipped parcels, pickup slots, linehaul trailers, and destination delivery capacity.
An aggregate parcel count can hide a congested destination or a service-level restriction.

**Output:** comparable workload in units appropriate to each resource.

**Branch:** when conversion factors are unstable, calculate a range and monitor the actual factor during the event.
A fragile-goods promotion can increase pack time and dimensional weight without increasing order count.
A new box rule can change parcels per order and the carrier allocation required.

### Step 5 — Establish reliable site capacity

**Input:** roster, attendance history, task standards, shift calendar, equipment uptime, and process constraints.

**Action:** calculate productive hours from scheduled hours using explicitly defined losses.
For example, 100 workers scheduled for 8 paid hours with 90% attendance and 75% productive time after breaks and indirect work provide 540 productive hours.
At 20 units per productive hour, the resulting labor capacity is 10,800 units, before considering equipment, space, and upstream work.
If a rate already measures units per paid hour, do not apply the productive-time factor again.

Apply training assumptions by cohort.
Experienced workers, new temporary staff, and cross-trained staff can have different rates and error rates.
Account for supervisors and trainers who are unavailable for normal processing while onboarding.
Check whether overtime reduces quality or attendance in following shifts.
A high hourly rate observed on a quiet day should not automatically become the peak standard.

**Output:** effective capacity by process, shift, and node, with a derivation.

**Branch:** if the sorter, packing stations, replenishment crew, or fire-safe floor space is limiting, adding pickers alone will not increase end-to-end output.
Use the lowest feasible stage capacity after ensuring every stage is measured in compatible units.
Where order mix makes a single conversion unreliable, retain separate work types.

### Step 6 — Protect labor availability and safety

**Input:** workload, staffing gap, hiring lead time, attendance distribution, training plan, and local rules.

**Action:** determine the gap between required and available productive hours by shift.
Source temporary labor early enough for screening, onboarding, and supervised practice.
Set staffing commitments with the agency in writing, including reporting location, schedule, cancellation terms, and replacement process.
Plan supervisors, trainers, equipment, transport to site, and meal breaks for the larger workforce.

Cross-training expands flexibility only if the worker is actually available and competent at the time of the switch.
Do not assign the same person to two concurrent capacities.
Separate planned overtime from an emergency option and price it using fully loaded rates.
Monitor injury reports, fatigue, quality errors, and turnover alongside throughput.

**Output:** roster by role and shift, training completion evidence, and a safe escalation path.

**Branch:** if hiring falls short before the deadline, lower promotional exposure, route eligible work elsewhere, or revise future promises.
A last-minute productivity target is not a substitute for missing people.

#### Labor-sizing worksheet

Use this worksheet before requesting additional people.
It separates demand for work from the number of people scheduled.
The figures below are a separate illustrative shift, not inputs to the three-day example.

1. Forecast 7,200 standard orders for the shift.
2. Forecast 800 complex orders for the same shift.
3. Assign 2.0 productive minutes to each standard order.
4. Assign 4.5 productive minutes to each complex order.
5. Multiply 7,200 by 2.0 to obtain 14,400 productive minutes.
6. Multiply 800 by 4.5 to obtain 3,600 productive minutes.
7. Add the workloads to obtain 18,000 productive minutes.
8. Divide by 60 minutes per hour to obtain 300 productive hours.
9. Add 20 productive hours for exception handling if it is excluded from the order standards.
10. The resulting workload is 320 productive hours.
11. Assume one scheduled worker-hour yields 0.72 productive hours.
12. Divide 320 by 0.72 to obtain 444.44 scheduled worker-hours.
13. Round up to 445 scheduled worker-hours for this workload.
14. If each worker is scheduled for an 8-hour shift, divide 445 by 8.
15. The minimum whole-person roster is 56 workers for that shift.
16. Confirm that the roster includes the required qualifications and coverage by task.
17. Confirm that 56 people have workstations and equipment for the complete shift.
18. Check supervision and training separately if they are not included in the 0.72 factor.
19. Check carrier capacity and the pickup cutoff before treating 8,000 orders as shippable.
20. Retain the unrounded calculation so that future mix changes can be modeled.

If complex orders increase from 800 to 1,200 while total orders remain 8,000, the mix changes the answer.
The revised standard-order count is 6,800.
Standard orders consume 6,800 × 2.0 = 13,600 productive minutes.
Complex orders consume 1,200 × 4.5 = 5,400 productive minutes.
Together they consume 19,000 productive minutes, or 316.67 productive hours.
Adding 20 exception hours gives 336.67 productive hours.
At 0.72 productive hours per scheduled hour, that requires 467.60 scheduled hours.
At 8 hours per worker, the minimum whole-person roster becomes 59 workers.
The order count did not change, but the task mix increased the required roster by three people.

The calculation is sensitive to the 0.72 conversion.
If it already includes expected absence, apply it to the scheduled roster directly.
If the 0.72 figure excludes absence, reduce the roster for attendance first.
Do not combine rates with different denominators without converting them.
The resulting roster is a planning estimate, not a safe staffing prescription.
The site must validate task standards, worker capability, breaks, and local requirements.

#### Staffing readiness by cohort

A worker on the schedule is not automatically a qualified worker at the station.
Track a cohort's readiness as a sequence of observable states.
The following checks help prevent planned headcount from being counted before it is useful.

| State | Evidence to collect | Owner's decision |
|---|---|---|
| Requested | Agency requisition with shift and role | Confirm request was sent before the lead-time deadline |
| Accepted | Named or otherwise contractually confirmed supply | Decide whether remaining gap needs a second source |
| Cleared | Required screening and access completed | Release the worker for site onboarding |
| Oriented | Site safety and procedural orientation recorded | Allow supervised practice only where appropriate |
| Practiced | Required supervised work completed | Decide whether the worker can progress to independent work |
| Qualified | Role-specific assessment and authorization recorded | Count the worker in the qualified roster |
| Scheduled | Shift, station, and supervisor assigned | Resolve duplicate assignments and coverage gaps |
| Attended | Shift check-in recorded | Update actual productive capacity |
| Productive | Work is completed at a measured rate and quality | Update the current shift forecast |
| Cross-trained | Additional role qualification is current | Use as a contingency without double-counting |

The cohort record should also identify the date on which each qualification expires.
Where the task requires equipment certification or other authorization, follow the applicable site and legal requirements.
Do not infer qualification from the number of days a person has been on site.
An experienced worker assigned to train others cannot simultaneously be counted at full individual production.
The supervisor should decide which work receives a trainee when quality risk differs by product.

#### Shift handover controls

Capacity planning fails if the next shift inherits work without context.
Use a short, timestamped handover with fields that can be acted on.
It should accompany the system record rather than replace it.

1. Report the actual count of available workers by role.
2. Report the qualified workers who are absent or temporarily reassigned.
3. List workstations and equipment out of service.
4. Report the opening count of orders not yet released.
5. Report the opening count of orders released but not picked.
6. Report the count picked but not packed.
7. Report the count packed but not accepted by a carrier.
8. Segment each queue by its earliest customer promise or required cutoff.
9. List inventory holds that prevent allocation.
10. List the remaining carrier slots by pickup and service.
11. Identify the next decision deadline and its owner.
12. Identify any customer communications already sent.
13. Record the current source-system refresh time.
14. Name the person accepting responsibility for the next update.

The queue counts need defined boundaries so that an order is not counted in two stages.
If a stage permits partial orders, report order and parcel counts separately.
The handover should highlight changes since the previous review rather than bury them in a large report.
The incoming lead should acknowledge material unresolved items and escalation deadlines.

#### Labor contingency decision card

Use a decision card to turn an attendance exception into an authorized action.
The following is a template; its numbers must be locally approved.

**Situation:** the 06:00 shift has fewer qualified packers than the roster requires.
**Observation:** qualified attendance is 42 people against a plan of 50 people.
**Measurement time:** 06:15 local site time.
**Data owner:** site operations lead.
**Business impact:** the pack queue may exceed the next pickup window.
**First check:** remove canceled orders and verify that late arrivals are not already on site.
**Second check:** compare actual pack rate and current order mix with the planning standard.
**Available option:** call the pre-qualified cross-trained team.
**Activation constraint:** the team requires one hour to reach the pack line.
**Resource interaction:** moving that team reduces replenishment capacity.
**Approval owner:** the shift manager within the pre-authorized labor budget.
**Escalation owner:** the site manager if the budget or safety limit would be exceeded.
**Customer impact:** identify orders whose promised dates are at risk, not merely all open orders.
**Communication owner:** customer experience lead.
**Next review:** 07:15, or earlier if the carrier cutoff changes.
**Close condition:** the pack queue is projected to fit the next eligible pickup without violating safety limits.

The card does not authorize ignoring a labor agreement or applicable law.
Its purpose is to make authority, lead time, and dependencies visible before an emergency.

#### Balancing labor across competing work

Replenishment, picking, packing, returns, and inbound receiving compete for space and labor.
The best local throughput at one task may reduce end-to-end shipments.
For example, moving all replenishment workers to picking can temporarily raise pick output.
When pick faces empty locations, the site then loses more capacity than it gained.
Similarly, receiving extra inbound stock can consume a dock needed for the last outbound pickup.
Compare alternatives using the earliest constrained customer commitment and the full chain of dependencies.

Use a task-priority list approved before peak.
The list should identify work that must continue for safety, legal compliance, or system integrity.
It should specify when customer-committed outbound work takes priority over discretionary work.
It should also identify the minimum replenishment and receiving staffing that keeps outbound flow viable.
Returns work may be deferred only within applicable policy and obligations, and only after checking storage and refund impacts.
The shift lead records any deviation, its expected duration, and the owner who approved it.

#### Measuring training effectiveness

Measure output and quality together.
A temporary worker's first-day rate may be lower because the training plan deliberately limits task complexity.
Compare like tasks and similar time windows before concluding that an individual or cohort is underperforming.
Use supervised observations to find missing materials, confusing instructions, and workstation constraints.
Correct a process defect before attributing its effect to workers.

A useful cohort view includes scheduled hours, attended hours, training hours, productive hours, completed tasks, and error counts.
Define each error consistently and distinguish detection time from the time the error occurred.
Track rework hours separately if the standard does not already include them.
If quality deteriorates when the rate rises, the apparent gain may be offset by returns, reshipments, and customer contacts.
The operations leader owns the decision to change a standard or work method.
Human-resources and safety owners should review any incentive or schedule that could create unsafe pressure.

#### Backward planning from the activation deadline

For each staffing option, calculate the last decision time rather than relying on the event start date.
Suppose a qualified reserve requires 90 minutes to confirm, 60 minutes to travel, and 30 minutes for check-in.
If the reserve must begin productive work at 18:00, the latest theoretical request is 15:00.
The calculation is 18:00 minus 3 hours of sequential lead time.
If the call center closes at 14:30, the operational deadline is 14:30 instead.
If the reserve's qualification expires before the event, the option is unavailable until it is renewed.
If the forecast refresh is at 15:00, it is too late to support the 14:30 decision.
Move the review earlier or arrange a later confirmation mechanism.

Add a sensible decision margin when travel, confirmation, and check-in are uncertain.
That margin is an explicit planning choice, not an extra quantity silently deducted from every step.
Record the owner who may accept the fixed cost of calling the reserve.
Record a cancellation deadline and any minimum payment.
Check whether the reserve shares the same transport corridor as the primary labor force.
If it does, a weather disruption may make both unavailable.

#### Inventory and labor interaction

Peak stock placement changes the amount and timing of warehouse work.
A pre-positioned pallet can reduce peak-day receiving pressure but consumes storage space earlier.
A late transfer can require overtime for unloading, inspection, put-away, and replenishment.
An inventory plan should therefore include its labor and dock consequences.
Likewise, a labor plan must include the physical stock and handling equipment needed for the work.

For an item group, record expected pallet arrivals and the receiving minutes per pallet.
Multiply arrivals by the standard to estimate receiving workload.
Add inspection or exception time only if it is not already included in that standard.
Compare the workload with qualified receiving capacity in the available window.
Then verify that put-away and system availability occur before order allocation.
A truck appointment is not the same event as stock becoming available to promise.
Where a receiving gate fails, identify the purchase orders and customer offers affected.
The inventory owner chooses the stock response; the site owner confirms its execution feasibility.

#### Operational interpretation of capacity

There are several different capacity statements, each useful for a different decision.
Do not combine them without checking their definitions.

| Statement | Appropriate use | Missing information to request |
|---|---|---|
| Equipment nameplate rate | Initial engineering ceiling | Availability, mix, downstream capacity |
| Historical best hour | Investigating potential process speed | Repeatability and labor conditions |
| Planned shift capacity | Labor and schedule budgeting | Attendance and actual workload |
| Effective shift capacity | Base operating scenario | Uncertainty and correlated failures |
| Contracted carrier allocation | Pickup planning | Actual acceptance and delivery restrictions |
| Actual accepted parcels | Execution and reconciliation | Downstream transit and final delivery |
| Customer on-time deliveries | Service assessment | Original promise, denominator, delivery scans |

Avoid adding historical best hours across several shifts as though all can occur simultaneously.
Measure constrained throughput at the boundary that matters for the decision.
Review each change in assumptions with the owner of that resource.
When evidence is weak, mark the capacity estimate as uncertain and plan an earlier validation.

#### Customer-service staffing and recovery

Peak workload extends beyond physical fulfillment.
Late tracking scans, failed delivery attempts, stock substitutions, and refunds generate customer contacts.
Model service demand separately from warehouse demand because its peak may occur later.
A simple workload estimate is contacts multiplied by handling minutes per contact.
Add follow-up work only when it is not already included in the handling-time measure.

For example, suppose 1,200 customers are expected to contact the business in a day.
If handling takes 6 productive minutes per contact, the direct workload is 7,200 minutes.
That is 120 productive hours before any excluded follow-up or supervision.
If 75% of scheduled time is available for that workload, the requirement is 160 scheduled hours.
This calculation assumes one contact per expected contact record, not one per affected order.
It does not establish a waiting-time service level because arrivals and agent availability vary during the day.
Use shorter intervals and a queueing model if a waiting-time decision requires them.

Plan the contact channels independently when skills or hours differ.
Phone, chat, email, and marketplace messaging can have different concurrency and response-time commitments.
Do not add their staffing as if every agent can handle every channel concurrently.
Account for language coverage, escalation authority, and peak-specific training.
Measure repeat contacts as well as total contacts to detect unresolved issues.

A disruption message should include what happened when it is confirmed.
It should distinguish a carrier-accepted parcel from a label that has only been created.
It should state the next update time even if an accurate new delivery date is unavailable.
It should explain the available remedy only after the responsible policy owner has approved it.
It should avoid exposing personal information to people who do not need it.
It should be consistent across the order page, outbound notification, and agent guidance.

Create an escalation path for customers with time-sensitive or otherwise material needs.
The policy owner decides prioritization and remedies in light of applicable obligations.
Track whether an agent's requested operational action is actually completed.
The customer-service lead reports unresolved cases and recurring root causes to the command team.
The site lead reports the best current recovery estimate and its uncertainty back to service.
An improvement in warehouse throughput is not proof that the contact queue has recovered.
Close the incident only after physical backlog, customer contacts, and required remedies have been reviewed.

#### Short-form readiness review

The review chair asks each owner for evidence rather than a verbal assurance.
For inventory, inspect the last usable receipt date and the actual receiving status.
For labor, inspect qualified attendance coverage by critical role and shift.
For transportation, inspect allocation confirmation, pickup times, and a successful acceptance event.
For systems, inspect the representative order's allocation, label, tracking, and exception results.
For service, inspect the affected-order query, approved communication, and remedy authority.
For finance, inspect contingency prices and the person authorized to incur them.
For commercial, inspect the promotion schedule and the process for changing future offers.
For safety, inspect the staffing, equipment, and escalation controls required by the site.
The chair records each exception, decision deadline, owner, and next review.
If the critical evidence is absent, the sponsor decides whether to change the launch plan.

### Step 7 — Place and protect inventory

**Input:** available-to-promise stock, inbound purchase orders, transfer lead times, receiving capacity, product eligibility, and node demand.

**Action:** project item-level stock by node and time bucket.
Start with usable on-hand inventory, add receipts only when they can be received and made available, subtract allocated demand, and track transfers as outflow and later inflow.
Separate quarantine, damaged stock, customer returns awaiting inspection, and already-reserved inventory from free stock.
Account for the inbound dock and put-away workload needed to make stock sellable.

Reserve scarce stock using an explicit policy approved by commercial and inventory owners.
The policy may prioritize committed orders, channels, or customer classes, subject to the business's obligations.
Do not count the same stock as available at two nodes.
If an item is present at a remote node, the resulting transfer or split shipment must still meet the promise and transport constraints.

**Output:** dated inventory coverage and the projected stockout or excess by node and item group.

**Branch:** if an inbound receipt misses its last useful receiving window, activate an alternate source or remove the affected offer for future orders.
A forecasted receipt is not physical availability.
A substitution requires product, commercial, and customer-communication approval where applicable.

### Step 8 — Contract and sequence transportation

**Input:** parcel forecast by origin, destination, service, pickup window, carrier rate card, and contract.

**Action:** obtain written allocations by date and service, including acceptance windows and escalation contacts.
Distinguish booked pickup capacity from destination delivery capacity.
Confirm package dimensions, weight limits, hazardous-material restrictions, address eligibility, and peak surcharges.
Check label generation, manifest exchange, scan events, and invoicing before volume starts.

The usable parcel capacity is the smaller of the carrier's accepted allocation and the number of shipment-ready parcels available by cutoff.
A backup carrier is not usable merely because it has a rate card.
It must have an allocation, a suitable service and geography, technical integration, labels, packaging compatibility, and a tested pickup.
If a second carrier shares the same linehaul bottleneck or regional depot, its capacity may not be independent.

**Output:** dated routing plan, capacity reservation evidence, and fallback route by eligible order segment.

**Branch:** if expected load exceeds a firm allocation, choose among additional contracted allocation, earlier fulfillment, an eligible alternate carrier, a slower future promise, or reduced demand.
Do not assume excess parcels will be accepted at the regular rate.

### Step 9 — Simulate queues and promise feasibility

**Input:** opening work in process, arrivals, stage capacities, inventory, pickup windows, and delivery lead times.

**Action:** advance work through stages in chronological order.
Carry unfinished work forward rather than discarding it at midnight.
At each stage, completed work cannot exceed available work or capacity.
Do not use a negative backlog as a credit against later demand.
Represent same-day transit only when the actual cutoff and service make it possible.

For a simplified one-stage daily approximation:

```text
available orders = opening backlog + released orders
processed orders = min(available orders, effective daily capacity)
closing backlog = available orders - processed orders
```

This is useful for a rough plan but cannot prove a specific customer delivery promise.
A promise assessment also needs order placement time, inventory allocation, processing time, pickup acceptance, destination transit, and the promised date.
Measure the age of unfinished work and prioritize according to an approved policy.

**Output:** throughput, backlog, utilization, and projected late-risk by bucket and scenario.

**Branch:** if backlog grows, estimate recovery using spare capacity after new arrivals.
If spare capacity is zero or negative, the backlog will not recover without a change in demand or capacity.
If an option activates after the latest useful decision time, it is not a valid protection for that promise.

### Step 10 — Compare scenarios and decide options

**Input:** baseline, demand-upside, and disruption scenarios; feasible interventions; cost and service guardrails.

**Action:** compare options on customer impact, activation time, capacity, reversibility, cost, and operational risk.
Avoid adding all optional capacities together unless they can operate concurrently and do not consume the same labor, dock, or carrier resource.
Show an unmitigated scenario and the same scenario with the proposed response.
When probabilities are defensible, expected cost can support a choice; when they are not, show scenario outcomes and decision-maker risk tolerance.

A capacity option has value only if it is available before the protected cutoff and can address the binding constraint.
A second pickup cannot fix stock that is not available.
An extra shift cannot clear parcels after the carrier's last accepted pickup unless a later pickup is also arranged.
An expensive intervention can still be justified by avoided customer harm or contractual exposure, but quantify what is known and identify what is not.

**Output:** approved base plan, pre-authorized options, and unresolved risks.

**Branch:** if no feasible option protects the published promise, change the offer for future orders and escalate the treatment of existing commitments.
Do not silently reclassify old commitments as successful under a new promise.

### Step 11 — Turn the plan into a readiness gate

**Input:** plan, owners, contracts, training records, integrations, and contingency tests.

**Action:** run a readiness review at dates when corrective action is still possible.
Each item needs an owner, evidence, a due date, a status, and a decision if it fails.
Use red for a condition that blocks the agreed plan, amber for an actionable risk, and green only for evidence of readiness.
Specify the meaning of each color before reporting begins.

A useful gate includes carrier test pickup and scan, staffing attendance and trained coverage, inventory receiving completion, packaging supply, system load and failover, and customer-service knowledge articles.
Test a representative order end to end through checkout, allocation, pick, pack, label, tender, and customer tracking.
A tabletop test should name the exact person who decides and the time required to activate an option.

**Output:** a signed readiness decision: proceed, proceed with controls, or change the commercial plan.

**Branch:** if an evidence item is missing, mark it unknown or not ready rather than assuming success.
An unresolved red item needs an accountable acceptance of risk, not merely a status color.

### Step 12 — Execute, communicate, and learn

**Input:** actual orders, inventory events, staffing attendance, stage completions, carrier acceptance, delivery scans, and contact volumes.

**Action:** establish a cadence appropriate to the event: daily planning before peak, shift-level reviews during peak, and more frequent checks near cutoffs.
Publish one timestamped operational view with actuals, the next forecast, aged backlog, capacity remaining, and trigger status.
Reconcile source-system latency before treating a missing scan as a physical failure.
Keep decision records with evidence, owner, time, affected orders, and next review.

Customer service receives a current explanation, affected-order list, next update time, and permitted remedies.
Tell customers what is known and what remains uncertain.
Do not announce an unverified delivery date just to reduce contacts.
If an existing promise is at risk, proactively communicate according to the applicable policy and obligations.

**Output:** operational actions and customer communications traceable to a decision.

After peak, reconcile forecast versus actual by bucket, mix, and node.
Separate demand error from execution losses and from changes in the customer promise.
Evaluate the accuracy of trigger lead times, whether contingencies delivered their stated capacity, and whether staff were safe.
Record the next-cycle change, owner, and evidence needed to close it.

## 6. Worked example

### Setting and assumptions

A fictional retailer operates one fulfillment center during a three-day promotion.
The example is illustrative; it is not evidence about a named company.
The retailer sells ordinary, parcel-eligible products and accepts orders throughout each day.
The planning bucket is one calendar day.
No opening backlog exists before day 1.
All orders in this example are inventory-available and eligible for the same carrier service.
Each order produces exactly one parcel; this simplifies the model and is not a general rule.
All orders are assumed available early enough for the same day's processing and pickup.
Actual intraday arrivals would require a finer model.

The demand forecast is 8,200, 10,600, and 9,400 orders on days 1, 2, and 3.
These figures are forecasts, not customer commitments.
The service target is to finish each day with no more than 500 orders awaiting carrier handoff.
That internal target is not the same as an on-time delivery guarantee.
The customer-facing promise depends on destination transit and is not proved by this daily model.

The base roster provides 480, 520, and 500 scheduled worker-hours for the three days.
The attendance assumption is 90% of scheduled hours.
After attendance, 80% of attended time is productive for the measured fulfillment activity.
The measured rate is 25 orders per productive hour for the assumed mix.
The labor capacity is therefore scheduled hours × 0.90 × 0.80 × 25 orders per productive hour.
Other warehouse equipment has at least that much capacity in the base case.
Carrier accepted allocations are 9,000, 9,500, and 10,000 parcels per day.
Because one order equals one parcel here, the daily system capacity is the smaller of labor capacity and carrier allocation.

The retailer has a written option for a second pickup on day 2.
It adds 1,500 accepted parcels on day 2 if activated by the end of day 1.
The option costs USD 2,400, regardless of whether all additional slots are used.
The extra pickup does not add warehouse labor.
The owner of the transport budget can activate it; operations confirms physical readiness.

### Base case without the option

| Day | Forecast arrivals, orders | Scheduled hours | Productive hours | Labor capacity, orders | Carrier capacity, parcels | System capacity, orders |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 8,200 | 480 | 345.6 | 8,640 | 9,000 | 8,640 |
| 2 | 10,600 | 520 | 374.4 | 9,360 | 9,500 | 9,360 |
| 3 | 9,400 | 500 | 360.0 | 9,000 | 10,000 | 9,000 |

For day 1, 480 × 0.90 × 0.80 = 345.6 productive hours.
At 25 orders per productive hour, labor can process 8,640 orders.
The carrier can accept 9,000 parcels, so labor is the limiting stage at 8,640 orders.
Available work is 0 opening backlog + 8,200 arrivals = 8,200 orders.
The site hands off 8,200 orders and closes with 0 orders of backlog.

For day 2, 520 × 0.90 × 0.80 × 25 = 9,360 orders of labor capacity.
The 9,500-parcel carrier allocation is higher, so labor still limits throughput.
Available work is 0 + 10,600 = 10,600 orders.
The site hands off 9,360 orders and closes with 1,240 orders of backlog.
That exceeds the 500-order internal target by 740 orders.

For day 3, 500 × 0.90 × 0.80 × 25 = 9,000 orders of labor capacity.
Available work is 1,240 + 9,400 = 10,640 orders.
The site hands off 9,000 and closes with 1,640 orders of backlog.
The backlog grows by 400 orders on day 3 even though carrier capacity is 10,000 parcels.

| Day | Opening backlog, orders | Arrivals, orders | Handoff, orders | Closing backlog, orders | Limiting capacity |
|---:|---:|---:|---:|---:|---|
| 1 | 0 | 8,200 | 8,200 | 0 | Labor ceiling |
| 2 | 0 | 10,600 | 9,360 | 1,240 | Labor |
| 3 | 1,240 | 9,400 | 9,000 | 1,640 | Labor |

The day 2 carrier option alone does not improve the result.
It raises the day 2 carrier allocation from 9,500 to 11,000 parcels, while labor remains 9,360 orders.
The retailer would spend USD 2,400 for no modeled throughput gain.
This does not imply the carrier option is never useful; it means the option does not address this scenario's bottleneck by itself.

### Combined option

Suppose operations can add 100 scheduled worker-hours on day 2 under the same attendance, productive-time, and rate assumptions.
The added labor provides 100 × 0.90 × 0.80 × 25 = 1,800 orders of modeled capacity.
Day 2 labor capacity rises to 11,160 orders.
With the original carrier allocation, throughput would still be capped at 9,500 parcels.
With both the extra labor and the second pickup, day 2 system capacity is min(11,160, 11,000) = 11,000 orders.
The 10,600 available orders can all be handed off on day 2.

Day 3 starts with no backlog, receives 9,400 orders, and hands off 9,000 orders.
Closing backlog is 400 orders, below the 500-order internal target.
The combined option meets the stated daily backlog target in this simplified scenario.
It does not establish that every customer receives an order by the checkout promise.
The labor cost is not specified, so the example does not claim a total cost or a cost-optimal decision.
The known transport option cost is USD 2,400; finance must add the actual incremental labor and any downstream costs before approval.

### Stress and recovery

If day 2 arrivals are 12,000 orders rather than 10,600, the combined option processes 11,000 and leaves 1,000 orders.
Day 3 then starts with 1,000, receives 9,400, processes 9,000, and closes with 1,400 orders.
The same option no longer meets the 500-order target.
The decision owner must consider an additional feasible day 3 action, reduced promotional exposure, or a changed future promise.
None is automatically assumed to exist.

If a later day has 8,000 arrivals and 9,000 processing capacity, its maximum net backlog reduction is 1,000 orders per day under these assumptions.
A 1,400-order backlog would need at least two such days to clear: 400 orders remain after the first day, and the second can clear the remainder.
This assumes all remaining inventory, labor, and transport capacity actually materialize.
Aged customer commitments may require a faster or differently prioritized response than the aggregate recovery estimate implies.

## 7. Code, query, or calculation

The following Python 3 program uses only the standard library.
Copy it into a file and run `python peak_plan.py`.
It implements the daily approximation above, with same-day availability and one parcel per order.
It does not model individual delivery dates, inventory shortages, or intraday cutoffs.
The program floors capacity to whole orders; the example's capacities are already integers.

```python
from dataclasses import dataclass
from math import floor


@dataclass(frozen=True)
class Day:
    arrivals: int
    scheduled_hours: float
    carrier_parcels: int


def simulate(days, extra_hours=None, extra_parcels=None):
    extra_hours = extra_hours or {}
    extra_parcels = extra_parcels or {}
    backlog = 0
    rows = []
    for day_number, day in enumerate(days, start=1):
        hours = day.scheduled_hours + extra_hours.get(day_number, 0)
        productive_hours = hours * 0.90 * 0.80
        labor = floor(productive_hours * 25)
        carrier = day.carrier_parcels + extra_parcels.get(day_number, 0)
        capacity = min(labor, carrier)
        available = backlog + day.arrivals
        handoff = min(available, capacity)
        closing = available - handoff
        rows.append((day_number, backlog, day.arrivals, labor,
                     carrier, handoff, closing))
        backlog = closing
    return rows


def report(name, rows):
    print(name)
    print('day opening arrivals labor carrier handoff closing')
    for row in rows:
        print(*row)


days = [Day(8200, 480, 9000), Day(10600, 520, 9500),
        Day(9400, 500, 10000)]
report('base', simulate(days))
report('combined', simulate(days, extra_hours={2: 100},
                            extra_parcels={2: 1500}))
```

Expected output:

```text
base
day opening arrivals labor carrier handoff closing
1 0 8200 8640 9000 8200 0
2 0 10600 9360 9500 9360 1240
3 1240 9400 9000 10000 9000 1640
combined
day opening arrivals labor carrier handoff closing
1 0 8200 8640 9000 8200 0
2 0 10600 11160 11000 10600 0
3 0 9400 9000 10000 9000 400
```

The input data are illustrative and should be replaced with validated local inputs for an operational decision.
A production model should separate stages, use order-level or bucketed promise deadlines, and record inventory and carrier eligibility.
It should also handle actual cutoff times, partial days, cancellations, and different conversion rates when those matter.
A floor operation is appropriate for whole orders only because the example's capacity is expressed in orders.
The output is a scenario calculation, not a statistical confidence interval.

## 8. Data, standards, and regulatory context

### Minimum data contract

For each dataset, record source, owner, refresh interval, event-time field, time zone, grain, and known latency.
Record the as-of timestamp used to make a decision so later corrections do not rewrite history.
A useful minimum set is:

| Dataset | Required grain | Critical fields | Common defect |
|---|---|---|---|
| Orders | Order and event | Placement time, channel, node, promise, cancellation | Order count duplicated by line joins |
| Order lines | Order line | Item, quantity, allocated node, status | Units confused with lines |
| Inventory | Item, node, status, time | Available, reserved, quarantined, inbound | Inbound stock counted as sellable |
| Labor | Person or aggregated role and shift | Scheduled, attended, productive hours, qualification | Same person assigned twice |
| Operations | Stage, node, interval | Started and completed work, exceptions, downtime | Scan timestamp mistaken for physical completion |
| Carrier | Pickup, service, destination, interval | Allocation, accepted parcels, cutoff, scans | Labels counted as carrier acceptance |
| Customer service | Contact and order | Contact reason, timestamp, resolution, affected promise | Repeated contacts counted as distinct affected orders |
| Costs | Resource and period | Rate, fixed fee, overtime, surcharge, currency | Comparing cost estimates with different inclusions |

Reconcile order totals before joining line or parcel tables.
One-to-many joins multiply rows; compute order-level measures at order grain or explicitly deduplicate.
Reconcile opening inventory plus receipts minus allocations and adjustments to closing inventory.
Reconcile opening backlog plus arrivals minus completions and cancellations to closing backlog at each stage.
Investigate differences rather than hiding them in a generic adjustment.

For an on-time metric, preserve the promise actually shown when the order was placed.
Store subsequent revisions as separate events with reason, authorization, and communication status.
This makes it possible to compare performance against original commitments and against any later customer-agreed changes.
A dashboard that overwrites the original promise cannot answer that question reliably.

### Identifiers and event exchange

GS1 standards provide common identification and event-sharing conventions.
A Global Trade Item Number (GTIN) identifies a trade item; it does not by itself identify a specific customer's order or a unique parcel.
A Serial Shipping Container Code (SSCC) can identify a logistics unit where implemented.
The GS1 Electronic Product Code Information Services (EPCIS) standard defines a way to share visibility events, including what, when, where, why, and how information.
Adopting an identifier or event format does not guarantee that a scan is timely or physically accurate.
Agree event definitions and reconciliation rules with each carrier and trading partner.

Electronic Data Interchange (EDI) is a family of structured business-message conventions, not a single universal carrier contract.
A trading-partner implementation guide specifies the actual message version, fields, timing, and error-handling agreement.
Confirm how allocations, manifests, acknowledgments, and tracking events are exchanged in the relevant relationship.
A technically accepted message is not necessarily proof that a physical parcel was collected.

### Employment, customer promises, and privacy

Applicable employment, workplace safety, consumer-protection, and privacy requirements depend on jurisdiction and facts.
The following are examples of primary sources, not a complete legal checklist.
In the United States, the Occupational Safety and Health Administration (OSHA) provides warehouse safety resources and enforces applicable occupational safety rules.
The U.S. Department of Labor explains federal Fair Labor Standards Act overtime rules and notes that state law can also apply.
A peak roster should be reviewed for the applicable rules, worker classification, breaks, and safe work practices; do not assume a planning model determines legal compliance.

The U.S. Federal Trade Commission's Mail, Internet, or Telephone Order Merchandise Rule addresses shipment representations and certain delay and refund obligations for covered orders.
Its detailed conditions matter; consult the rule and counsel for the actual transaction rather than assuming every late parcel has the same remedy.
Contract terms and other consumer laws may also apply.
The European Union's General Data Protection Regulation (GDPR) imposes requirements on processing personal data within its scope, including principles of data minimization and security.
Customer-support lists should contain only the information required for the authorized task, with appropriate access and retention controls.
This material is educational and is not legal advice.

## 9. Industry practice and case evidence

### A documented delivery disruption

In December 2013, UPS publicly discussed service disruptions around Christmas after unusually high shipping volume and severe weather affected parts of its network.
The company's 2013 annual report discusses the impact of peak volume and weather on its operations.
The public account is evidence that parcel networks can experience simultaneous load and disruption; it does not provide the fictional retailer's capacity factors or prove any single company's planning process.
[Inference] A retailer dependent on a shared parcel network should monitor accepted pickup capacity and downstream delivery performance separately, and should not assume that a warehouse handoff guarantees delivery by the promised date.

A useful case discussion asks which facts were available before cutoff, which were observable only afterward, and what alternatives were actually contracted.
A second carrier may reduce exposure, but it is not an independent hedge if both carriers rely on the same constrained infrastructure or weather-affected geography.
The correct decision depends on the specific service, location, and contract.

### Retail sales concentration

The U.S. Census Bureau publishes monthly retail trade data and definitions, including nonstore retailing.
Those series are useful context for seasonality, but they are not a forecast of a particular fulfillment center's orders, parcels, or hourly workload.
[Inference] A retailer should combine external seasonality information with its own checkout, promotion, product mix, and node-level event data.
National monthly sales cannot by themselves establish an hourly carrier allocation.

### Illustrative operational case: labor is not the only constraint

A site expects 6,000 orders in a shift and has labor capacity for 6,600 orders.
It also has a firm carrier allocation of 5,500 parcels, with one parcel per order.
Even if the shift processes all 6,000 orders, only 5,500 parcels can be handed off within the allocation.
The first 500 orders of backlog are transport-constrained; another labor shift alone cannot eliminate that handoff backlog.
A feasible response could be a confirmed extra pickup, an earlier eligible pickup, or a reduced offer for future orders.
The choice requires the cutoff, service eligibility, cost, and customer promise, none of which follows from the aggregate counts alone.

### Illustrative operational case: inventory and customer service

A promotion forecasts 4,000 units of an item at the east node and 1,000 units at the west node.
The east node has 2,500 usable units, while the west has 3,000 units.
Network stock totals 5,500 units against 5,000 forecast units, but east has a 1,500-unit local shortfall before transfers.
A transfer of 1,500 units can protect the plan only if it arrives, is received, and becomes available before east-node orders need allocation.
If the transfer misses that window, a remote shipment might be possible only if its route meets the customer's promise and has capacity.
Otherwise, future checkout availability or promise must change, and existing affected orders need an authorized response.
Customer service should receive the order list and verified next update, not the reassuring but misleading network-wide stock total.

### What the cases teach

Public cases establish that disruption can occur; illustrative cases isolate mechanisms.
Neither type substitutes for local data or establishes the probability of an event.
The operational lesson is to test coupled constraints, preserve original promises, and choose actions while the relevant lead time remains.
Every claimed contingency needs an executable route from decision to physical effect.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Weekly averaging | Daily arrival spikes exceed pickup allocation | Excess demand concentrates before a cutoff | Late handoffs despite weekly spare capacity | Plan by node and pickup window |
| Unit confusion | Orders, units, and parcels reported in one column | Multipack and split rates are omitted | Understated labor or transport | Define grain and conversion factors |
| Duplicate loss factors | Paid-hour rate multiplied by an additional break factor | Same time loss is deducted twice | Capacity is understated and spend is misdirected | Document rate denominator and reconcile |
| Training shortfall | New-hire qualification rate trails plan | Planned headcount is not productive capacity | Backlog and errors increase | Cohort rates, training gates, early hiring |
| Fatigue and unsafe overtime | Absence, incidents, or errors rise across shifts | Sustained load erodes safe performance | Harm, disruption, and lower throughput | Set safe limits and monitor leading indicators |
| Inventory counted too early | Inbound stock is still at dock at cutoff | Receiving and put-away are ignored | Orders cannot be allocated | Model usable receipt time and dock capacity |
| Network stock illusion | Aggregate stock is healthy but local stock is short | Product is at the wrong node | Split shipment or missed promise | Plan by item and node with transfer lead time |
| Carrier allocation assumed | No written dated acceptance limit exists | A forecast is mistaken for an allocation | Excess parcels remain at the site | Secure and test written allocation |
| Backup not operational | Labels, scans, or pickup test incomplete | Commercial agreement is not execution readiness | Contingency activates too late | End-to-end test before activation deadline |
| Correlated alternatives | Two routes share a depot or linehaul | Apparent diversification shares a failure point | Both options fail together | Map shared dependencies and stress them |
| Trigger too late | Threshold is crossed after agency call deadline | Decision latency exceeds option lead time | Capacity cannot arrive in time | Set trigger using activation lead time |
| Promise overwrite | Dashboard retains only current delivery date | Original commitment disappears | Reported service looks better than experience | Preserve immutable original promise events |
| Unaged backlog | Total backlog is stable while oldest orders age | New work is processed ahead of urgent work | Promised orders become late | Age by promise and apply approved priority |
| Misread scans | Many parcels have labels but no acceptance event | Data event is confused with physical handoff | Customer status and recovery estimates are wrong | Reconcile manifest, pickup, and acceptance |
| Broken communication | Support receives no affected-order list | Operations and service use different data | Repeated contacts and inconsistent remedies | Share timestamped list and approved script |
| Unpriced option | Option shown as free capacity | Fixed fees, overtime, or surcharges omitted | Margin loss and delayed approval | Price incremental costs and authority in advance |
| False readiness green | Status is green without evidence | Attendance at a review is treated as completion | Untested process fails during peak | Require owner, test, evidence, and expiry |
| Permanent recovery deficit | Backlog grows while arrivals exceed daily capacity | No positive net recovery capacity exists | Late work persists indefinitely | Add capacity or reduce incoming load |
| Forecast double uplift | Promotion applied in two systems | Baseline already includes event effects | Excess inventory or unnecessary capacity | Version and reconcile forecast assumptions |
| Customer-data overexposure | Broad spreadsheet includes unnecessary personal data | Incident response bypasses access controls | Privacy and security exposure | Minimize fields and restrict access |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Daily arrivals are available before cutoff | Allows daily flow model | Late-day orders miss same-day pickup | Compare arrival timestamps with processing windows |
| One order equals one parcel | Makes order and carrier capacity comparable | Split orders or consolidation | Measure parcels per shipped order by mix |
| Inventory is available | Isolates labor and carrier effects | Late receipts, quarantine, stockouts | Reconcile allocatable inventory by item and node |
| Attendance is 90% | Converts roster hours to expected attendance | Weather, illness, agency shortfall | Compare actual attendance by cohort and shift |
| Productive time is 80% | Converts attendance to productive hours | Breaks or indirect tasks differ from plan | Audit task time and ensure no double deduction |
| Rate is 25 orders per productive hour | Converts hours into output | Mix, training, or congestion changes | Measure completed eligible orders per productive hour |
| Stages share a daily bucket | Allows a simple minimum-capacity approximation | Pack completes after pickup cutoff | Model stage timestamps and cutoff windows |
| Carrier accepts allocation | Treats allocation as usable capacity | Restrictions, force majeure, missed manifest | Confirm contract, acceptance events, and eligibility |
| No equipment bottleneck | Allows labor capacity to represent warehouse output | Sorter, pack stations, or dock constraints | Compare stage output and downtime with the plan |
| Backlog is homogeneous | Allows aggregate queue calculation | Different promises and priority rules | Segment by age, service, and due date |
| Extra labor has base productivity | Quantifies the combined option | New staff need supervision or overtime degrades output | Measure cohort productivity and quality |
| Options can operate together | Supports combined labor and pickup scenario | Shared dock, staffing, or service restrictions | Run an integrated feasibility and cutoff check |
| Demand scenario is known for calculation | Permits reproducible arithmetic | Actual demand differs from forecast | Update with observed arrivals and scenario range |
| Transport option cost is fixed | Makes one known cost visible | Volume charges or surcharges apply | Check signed contract and invoice terms |

The illustrative model is deterministic.
It does not assign probability to any scenario and does not claim to optimize the network.
Its day-level outputs cannot determine individual on-time delivery without the missing event and transit information.
Extend the model only when the decision requires the additional complexity and data support it.

## 12. Exercises

For calculations, show units and intermediate steps.
State whether your result is a forecast, plan, target, or commitment.

### Exercise 1 — Convert orders to workload

A node forecasts 4,800 orders for one day.
The expected mix is 1.7 units per order and 1.2 parcels per order.
**Deliverable:** units to pick and parcels to tender.
**Acceptance:** both quantities use the correct conversion and are labeled as forecasts.
**Time:** 5 minutes.

### Exercise 2 — Calculate effective labor capacity

A shift schedules 400 worker-hours.
Attendance is expected to be 95%, and 75% of attended time is productive.
The measured rate is 24 orders per productive hour.
**Deliverable:** productive hours and order capacity.
**Acceptance:** show both factors once and identify the rate denominator.
**Time:** 8 minutes.

### Exercise 3 — Carry backlog forward

Opening backlog is 300 orders.
Day 1 has 4,800 arrivals and capacity for 4,500 orders.
Day 2 has 4,200 arrivals and capacity for 4,700 orders.
Assume inventory and carrier capacity are not tighter constraints.
**Deliverable:** completed orders and closing backlog for each day.
**Acceptance:** carry day 1's backlog into day 2 and do not create negative backlog.
**Time:** 10 minutes.

### Exercise 4 — Choose the limiting intervention

The plan has 10,000 orders available, warehouse capacity of 9,000 orders, and a carrier allocation of 8,000 parcels.
Assume one parcel per order and all orders meet cutoff.
Option A adds 1,000 orders of warehouse capacity; option B adds 1,000 carrier parcels.
**Deliverable:** throughput for no option, A only, B only, and both.
**Acceptance:** identify which option changes throughput and distinguish a physical ceiling from demand.
**Time:** 10 minutes.

### Exercise 5 — Check the promise window

An order is placed at 15:00 on Monday.
Allocation and packing take two hours and the carrier cutoff is 16:00 Monday.
The next pickup is Tuesday at 16:00 and transit takes two calendar days after pickup.
Assume no other pickup or faster service.
**Deliverable:** earliest delivery day and whether a Wednesday delivery promise is feasible.
**Acceptance:** explain why Monday's cutoff is missed and identify the resulting delivery day.
**Time:** 8 minutes.

### Exercise 6 — Critique readiness

A dashboard shows green because weekly carrier capacity is 50,000 parcels against a 48,000-parcel forecast.
It does not show daily allocation, pickup cutoff, contract evidence, or split-shipment rate.
**Deliverable:** four specific questions and the evidence needed to answer them.
**Acceptance:** cover timing, commitment, unit conversion, and operational testing.
**Time:** 12 minutes.

### Exercise 7 — Design a command response

At 10:00 on peak day, paid orders run 25% above the hourly plan.
An extra shift requires six hours' notice; a second pickup must be booked by 12:00.
The original customer promise remains in force for already accepted orders.
**Deliverable:** a one-page decision record with trigger, owner, timeline, data checks, customer-service action, and next review.
**Acceptance:** include both action deadlines, a named decision role, the affected-order population, and a distinction between future promise changes and existing commitments.
**Time:** 20 minutes.

### Exercise 8 — Design a coupled stress

A carrier depot and a temporary-labor agency both depend on the same metropolitan transport corridor.
Severe weather could reduce attendance and delay pickups simultaneously.
**Deliverable:** a scenario table, backlog calculation approach, and two pre-event mitigations.
**Acceptance:** model the dependencies explicitly, avoid inventing a probability, and explain how each mitigation can be activated before its deadline.
**Time:** 20 minutes.

## 13. Self-check answers

### Exercise 1

The forecast pick workload is 4,800 orders × 1.7 units per order = 8,160 units.
The forecast tender workload is 4,800 orders × 1.2 parcels per order = 5,760 parcels.
Neither figure is a carrier commitment or a confirmed inventory reservation.
If actual split behavior changes, the parcel forecast must be updated.

### Exercise 2

Attended time is 400 scheduled hours × 0.95 = 380 attended hours.
Productive time is 380 × 0.75 = 285 productive hours.
Capacity is 285 productive hours × 24 orders per productive hour = 6,840 orders.
This is a modeled capacity, not a guarantee that inventory, equipment, or carrier capacity will support that output.

### Exercise 3

Day 1 available work is 300 + 4,800 = 5,100 orders.
Completed work is min(5,100, 4,500) = 4,500 orders.
Closing backlog is 600 orders.
Day 2 available work is 600 + 4,200 = 4,800 orders.
Completed work is min(4,800, 4,700) = 4,700 orders.
Closing backlog is 100 orders.

### Exercise 4

Without an option, throughput is min(10,000, 9,000, 8,000) = 8,000 orders.
With A only, throughput remains min(10,000, 10,000, 8,000) = 8,000 orders.
With B only, throughput is min(10,000, 9,000, 9,000) = 9,000 orders.
With both, throughput is min(10,000, 10,000, 9,000) = 9,000 orders.
Option B increases throughput by 1,000 orders; adding A on top of B does not raise it further because carrier capacity remains 9,000.
This does not establish which option is economical without costs and service consequences.

### Exercise 5

Packing ends at 17:00 Monday, after the 16:00 cutoff.
The next pickup is Tuesday.
Two calendar days after Tuesday is Thursday.
A Wednesday promise is infeasible under the stated assumptions.
The example assumes the parcel is actually accepted at Tuesday's pickup and transit takes exactly two calendar days.

### Exercise 6

A strong response asks for dated pickup allocations and forecast parcels per pickup, not merely weekly totals.
It asks for the signed or otherwise confirmed allocation and its restrictions.
It requests measured parcels per order and the effect of split shipments.
It asks for a test pickup and acceptance scan, plus evidence that the backup route is operational.
A fifth useful question asks whether the delivery transit can meet each promised destination date.

### Exercise 7 rubric

A strong decision record states who may authorize the extra shift and second pickup and records the latest decision time for each.
It compares the live order curve with the remaining daily capacity after validating data latency and mix.
It gives customer service a timestamped list of at-risk orders and an approved next-update time.
It preserves existing commitments and describes any changed future checkout promise separately.
It states when the next review occurs and what decision would follow each trigger.
A weak answer says only to hire more workers or tell customers there is a delay.

### Exercise 8 rubric

A strong scenario reduces attendance and pickup availability in the same weather period and explains their common cause.
It carries unfinished work forward, checks each stage and cutoff, and ages work by promise where data permit.
It names two feasible protections, such as a geographically independent route and pre-positioned trained staffing, with costs, owners, activation deadlines, and shared dependencies.
It does not call an arbitrary severe scenario a 95th percentile.
A weak answer adds nominal backup capacities without checking whether the same corridor disables both.

## 14. Interview questions

1. **Why can a site miss promises when weekly capacity exceeds weekly demand?**
   A strong answer discusses time buckets, cutoff concentration, stage precedence, and aged commitments.
   A weak answer compares only two weekly totals.
2. **How do you convert orders into a labor plan?**
   A strong answer specifies mix, work standards, productive-hour definition, and indirect work.
   A weak answer multiplies headcount by a rate with no unit or loss assumptions.
3. **What is the difference between a forecast and a carrier commitment?**
   A strong answer names uncertainty, written allocation, date, service, and restrictions.
   A weak answer treats a carrier's verbal willingness as guaranteed capacity.
4. **When is extra warehouse labor ineffective?**
   A strong answer identifies another binding stage, such as carrier acceptance or unavailable stock.
   A weak answer assumes any additional labor increases shipments.
5. **How do you decide when to activate a contingency?**
   A strong answer works backward from cutoff and includes lead time, trigger, authority, and cost.
   A weak answer waits until the promise is already missed.
6. **What makes a backup carrier real?**
   A strong answer checks service eligibility, dated allocation, integration, pickup, and shared dependencies.
   A weak answer points only to an unsigned rate card.
7. **How can a service metric improve while customer experience worsens?**
   A strong answer describes overwritten promises, changed denominators, or long future promises.
   A weak answer treats a single percentage as sufficient evidence.
8. **What should customer service receive during a disruption?**
   A strong answer includes affected orders, known facts, uncertainty, update time, and authorized remedies.
   A weak answer invents a precise delivery date without supporting evidence.
9. **How do you estimate backlog recovery?**
   A strong answer uses spare capacity after new arrivals and checks promise age and bottlenecks.
   A weak answer divides backlog by total capacity without accounting for new demand.
10. **How do you handle correlated stress events?**
    A strong answer models shared causes and dependencies without assigning unsupported probabilities.
    A weak answer assumes backup routes are independent because they have different names.

## 15. Key terms

- [Peak](#4-definitions-and-vocabulary): load or variation approaching a resource limit.
- [Forecast](#4-definitions-and-vocabulary): an estimate of uncertain demand.
- [Scenario](#4-definitions-and-vocabulary): a coherent set of assumptions.
- [Plan](#4-definitions-and-vocabulary): chosen resources and actions.
- [Target](#4-definitions-and-vocabulary): desired performance.
- [Commitment](#4-definitions-and-vocabulary): an authorized obligation.
- [Order](#4-definitions-and-vocabulary): a customer transaction.
- [Order line](#4-definitions-and-vocabulary): an item entry within an order.
- [Unit](#4-definitions-and-vocabulary): a physical sellable item.
- [Parcel](#4-definitions-and-vocabulary): a package tendered for transport.
- [Workload](#4-definitions-and-vocabulary): demand expressed in resource consumption.
- [Standard time](#4-definitions-and-vocabulary): expected time per work unit.
- [Productive hour](#4-definitions-and-vocabulary): hour available for specified work.
- [Effective capacity](#4-definitions-and-vocabulary): achievable output after stated losses.
- [Utilization](#4-definitions-and-vocabulary): load divided by capacity.
- [Bottleneck](#4-definitions-and-vocabulary): the limiting stage for throughput.
- [Headroom](#4-definitions-and-vocabulary): capacity above expected load.
- [Backlog](#4-definitions-and-vocabulary): accepted work not completed at a stage.
- [Aging](#4-definitions-and-vocabulary): breakdown of work by elapsed time or deadline.
- [Allocation](#4-definitions-and-vocabulary): reserved stock or capacity under a rule.
- [Cutoff](#4-definitions-and-vocabulary): latest qualifying action time.
- [Tender](#4-definitions-and-vocabulary): presenting a parcel for carrier acceptance.
- [Service level](#4-definitions-and-vocabulary): defined fraction meeting an outcome.
- [Recovery time](#4-definitions-and-vocabulary): time until an agreed recovery state.
- [Trigger](#4-definitions-and-vocabulary): observable condition initiating a decision.
- [Lead time](#4-definitions-and-vocabulary): delay from action to usable effect.
- [Readiness](#4-definitions-and-vocabulary): evidence that a plan can operate.
- [Contingency](#4-definitions-and-vocabulary): a specified alternative action.
- [Promise](#4-definitions-and-vocabulary): customer-facing delivery commitment.
- [WISMO](#4-definitions-and-vocabulary): a customer inquiry about order location.

## 16. Sources

Sources are cited for the specific claims above; they do not validate the fictional calculations.
Accessed for course context as of the lesson's version; check current text and local applicability before use.

1. [GS1, EPCIS standard](https://www.gs1.org/standards/epcis), official standard overview and specifications.
2. [GS1, GTIN](https://www.gs1.org/standards/id-keys/gtin), identifier definition.
3. [GS1, SSCC](https://www.gs1.org/standards/id-keys/sscc), logistics-unit identifier definition.
4. [U.S. Occupational Safety and Health Administration, Warehousing](https://www.osha.gov/warehousing), safety resources and applicable regulatory context.
5. [U.S. Department of Labor, Overtime Pay](https://www.dol.gov/agencies/whd/overtime), federal overtime overview and links to applicable rules.
6. [U.S. Federal Trade Commission, Mail, Internet, or Telephone Order Merchandise Rule](https://www.ftc.gov/legal-library/browse/rules/mail-internet-or-telephone-order-merchandise-rule), rule text and related materials.
7. [EUR-Lex, Regulation (EU) 2016/679](https://eur-lex.europa.eu/eli/reg/2016/679/oj), General Data Protection Regulation official text.
8. [UPS, 2013 Annual Report](https://investors.ups.com/financials/annual-reports), company reporting on its operations; consult the 2013 report for the cited peak context.
9. [U.S. Census Bureau, Monthly Retail Trade](https://www.census.gov/retail/index.html), official series and methodology.
10. [National Institute of Standards and Technology, Time and Frequency Division](https://www.nist.gov/pml/time-and-frequency-division), context for consistent time measurement; operational time-zone and cutoff rules remain contract-specific.

## 17. Links

- [Module 21 index](_index.md)
- [Module 21 survey](../../../course/21_ecommerce_supply_chain_analytics.md)
- [Order-grain forecasting](02_order-grain-forecasting.md)
- [Inventory placement](04_inventory-placement.md)
- [Delivery promise](05_delivery-promise.md)
- [Carrier selection](06_carrier-selection.md)
- [Simulation and optimization production labs survey](../../../course/30_simulation_optimization_production_labs.md)
