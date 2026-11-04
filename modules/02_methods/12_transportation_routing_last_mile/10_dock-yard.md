# Dock Scheduling and Yard Management

> Module 12 — Transportation, Routing, and Last-Mile Analytics
> Topic 10 of 10
> Estimated study time: 3 hours
> Prerequisites: topics 01–03
> Version 2026.5

## 1. Why this topic exists

A truck arriving at a warehouse transfers responsibility across transport, security, yard operations, receiving and payment. An appointment is a reservation for a scarce combination of door, labor, equipment and time. It is not proof that the vehicle will arrive on time or that freight will be ready to unload. A full calendar can coexist with idle doors when arrivals bunch or shipments are not ready.

Long dwell can increase carrier charges, delay the next load, consume yard capacity and obscure a warehouse productivity problem. Yet reducing one clock does not necessarily improve the system: moving a queue from the yard to a public road, or pressuring a driver to skip a safety step, is not an operational improvement.

The decision is how to allocate and revise appointments, manage exceptions and account for the full handoff. Examples below use fictional 2026 USD and assumptions, not industry benchmarks or legal advice.

## 2. Learning outcomes

You can:

1. Map an appointment and truck visit from booking to release.
2. Define gate-to-gate dwell, queue and service time without overlapping intervals.
3. Calculate capacity and utilization while accounting for variable handling time.
4. Compare appointment alternatives using incremental total cost and constraints.
5. Specify a data reconciliation and detention evidence process.
6. Design exception handling for early, late and missing arrivals.
7. Evaluate a pilot without mistaking queue displacement for improvement.

Exercises and answers cover these outcomes.

## 3. The core mental model

```text
Carrier requests slot -> facility confirms promise -> ETA updates
        -> gate arrival -> check-in -> yard staging -> door assignment
        -> loading/unloading -> paperwork/release -> gate-out
                             |                    |
                         exceptions          clocks and evidence
                             \                    /
                      appointment and yard decisions
```

The appointment, predicted arrival, actual arrival and assigned door are separate records. A carrier may request 10:00, receive a 10:30 booking, predict 10:45, and arrive at 10:55. Changing the prediction does not change the agreement unless an authorized party actually rebooks it.

A door is only one constraint. Door compatibility, required labor, material-handling equipment, product temperature, trailer type, staging space and safety rules can all limit throughput. A schedule that assigns every door every minute leaves no allowance for uncertainty or changeovers. Conversely, idle time can be a deliberate buffer with economic value when it avoids a much more expensive queue.

### Three decisions, three owners

The transport buyer decides what service and accessorial terms to buy from a carrier.
The facility manager decides what combinations of door, crew and equipment can be promised.
The yard coordinator decides how to use available resources as actual arrivals and exceptions unfold.
These decisions share information but do not have the same objective or authority.

A procurement saving can disappear if a low-priced carrier receives appointment windows it cannot reliably meet.
A facility can reduce its labor budget while increasing total network costs if trucks wait longer and miss their next loads.
A yard coordinator can minimize today's visible queue by accepting unplanned loads, while tomorrow's confirmed loads inherit the backlog.
The practical unit of analysis is consequently a visit embedded in a network of promises and resources.

The analysis should answer four questions before it proposes a scheduling algorithm.
First, what has the site promised, and to whom?
Second, which resources must be available simultaneously to deliver that promise?
Third, what evidence establishes what actually happened?
Fourth, who bears the incremental cost of each feasible alternative?
Those questions remain necessary whether the calendar is a spreadsheet or a commercial yard-management system.

### The same delay can have different consequences

A 30-minute delay on an inbound replenishment truck may threaten a production line.
The same delay on a drop trailer delivered before its required date may have no immediate customer effect.
A delayed outbound vehicle can miss a customer receiving window and incur a second appointment or a failed delivery.
A driver may also lose the opportunity to complete another load, even where no detention is invoiceable.
Conversely, an early inbound truck may be useful only if freight, labor and a compatible door are ready.

Separate the physical delay, the contractual consequence and the commercial relationship.
The physical delay is measured from events; the contractual consequence follows the relevant agreement.
The relationship consequence may appear later as a carrier declining future tenders or pricing a lane differently.
Do not claim that every minute of driver delay causes a particular rate increase.
Instead, measure the delay and compare carrier feedback, tender performance and negotiated prices over comparable periods.

### Flow and inventory are connected

The yard holds trailers while they wait for a door, a tractor, documentation or the next move.
That inventory consumes space even if the receiving crew is working efficiently.
A facility that accepts more arrivals than it can release eventually runs out of usable positions.
At that point the appointment plan must change, or the queue simply moves outside the fence.
The correct objective is safe end-to-end flow, not the highest possible number of arrivals at the gate.

### Reading map

Use sections 3–5 to define the process and build a feasible plan.
Section 6 follows the arithmetic from door capacity through charges and a physical visit.
Sections 7–11 cover implementation, evidence, operational red flags and limits.
Use sections 12 and 13 to practice before applying the methods to a real facility.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Appointment | Confirmed reserved service interval | Equating request with confirmed booking |
| ETA | Forecast of a defined arrival event | Treating forecast as a reservation |
| Gate-in | Verified entry event, timestamp | Assuming truck is at a door |
| Check-in | Administrative readiness event | Assuming all freight documents are valid |
| Door-in | Trailer positioned at assigned door | Treating door assignment as physical arrival |
| Service start | Agreed start of loading or unloading | Using it interchangeably with gate-in |
| Service end | Completion of physical handling | Assuming paperwork is complete |
| Gate-out | Verified exit event | Assuming door was occupied until exit |
| Dwell | Duration between explicitly named events | Using one undefined dwell metric |
| Queue | Waiting for a constrained resource | Treating all time on site as queue |
| Detention | Contractual charge for specified delay | Assuming every dwell minute is billable |
| Free time | Contract-defined nonbillable interval | Assuming one industry-wide rule |
| Door occupancy | Door-in to door-out duration | Treating it as unloading labor time |
| Yard move | Trailer relocation in a yard | Omitting the move from resource planning |
| Utilization | Busy time / available time for a resource | Assuming 100% is a safe target |
| No-show | Appointment not fulfilled under defined rules | Labeling a late arrival a no-show prematurely |

### Additional operational terms

| Term | Meaning and unit | Decision or common error |
|---|---|---|
| Booking horizon | Time between confirmation and the reserved interval, in hours or days | Short notice limits capacity adjustment |
| Slot width | Duration of the offered arrival or service window, in minutes | It is not automatically the expected handling duration |
| Grace period | Contractual or policy tolerance around an appointment, in minutes | Do not presume a universal tolerance |
| Cut-off | Time after which the plan follows an exception process | It does not prohibit authorized changes |
| Readiness | Required freight, documentation, crew and equipment are available | Physical arrival alone is insufficient |
| Live load | Trailer handled during a driver's visit | Driver time and door time can overlap |
| Drop-and-hook | Tractor drops or collects a trailer separately from handling | Gate visit and trailer dwell have different endpoints |
| Trailer dwell | Duration a particular trailer remains in a specified location or facility | Do not attribute all trailer dwell to the driver |
| Yard jockey | Vehicle or operator used to move trailers within a site | Door availability does not imply move capacity |
| Turn time | Site-specific duration for a truck visit, in minutes | Always name its endpoints |
| Door compatibility | Whether a door can safely handle a particular visit | An incompatible assignment is not merely a penalty |
| Changeover | Time and work between distinct handling activities | May require both door and labor capacity |
| Occupancy rate | Busy resource time divided by resource time available | A daily average does not reveal hourly peaks |
| Saturation | Demand approaches or exceeds effective capacity | High utilization can produce unstable queues |
| Staging capacity | Safe, usable holding positions at a point in time | Painted positions are not always usable |
| Yard inventory | Trailers physically present in a defined yard area | A booked but absent trailer is not inventory |
| External arrival | Documented arrival at a defined point outside the gate | Coverage may be incomplete or uncertain |
| Schedule adherence | Actual performance against a specified schedule rule | Denominator and comparison version matter |
| Exception | A visit requiring a decision outside the ordinary workflow | A reason code does not itself resolve it |
| Carrier charge | Amount due under a specified agreement, in currency | A cost estimate is not an approved invoice |
| Ingestion time | Time an event reached the analytics system | It need not be the physical event time |
| Event correction | A new version of a previously recorded event | Overwriting raw evidence destroys auditability |
| Spillover | An intervention affects untreated visits or periods | It can bias a pilot comparison |
| Counterfactual | Outcome expected without the intervention | A before period alone may not establish it |

### Four clocks that should not be merged

The **appointment clock** describes the promised window and any agreed tolerance.
The **physical clock** describes movement and handling from observed events.
The **contractual clock** is the interval used to determine a potential charge.
The **forecast clock** describes a predicted event and when that prediction was made.
They can begin at different times and answer different questions.

For example, a truck booked at 10:00 may arrive at 09:40, start service at 10:20 and depart at 11:15.
The physical gate-to-gate duration depends on its gate timestamps, not the appointment alone.
The contract might start at 10:00, at check-in or at another agreed event.
An ETA of 09:45 generated at 08:00 is a forecast, not an observation that arrival occurred at 09:45.
Store each clock separately before attempting a comparison.

## 5. The mechanics, step by step

### 5.1 Establish the operating contract

Define which shipments require appointments, who books them, the slot interval, early and late tolerance, cancellation rules, carrier communication and escalation authority. A booking request is not confirmed until the facility accepts it. Record original and revised slots with who authorized each change and when.

Specify the target of the arrival forecast. An ETA at the public gate is not an ETA at the assigned door. The facility can reserve labor based on a forecast while preserving the original appointment for service measurement. If an authorized rebooking changes obligations, keep both versions and the applicable contract context.

### 5.2 Build the event chain

Record appointment request, confirmation, gate-in, check-in, staging, door-in, service start, service end, door-out, document release and gate-out. Each event needs a source, time zone, event time, ingestion time and correction history. Some events can occur simultaneously; some visits require multiple doors or handling stages. Preserve the actual sequence instead of forcing all visits into one simplified template.

Define intervals before calculating them. Gate-to-gate dwell is gate-out minus gate-in. Door occupancy is door-out minus door-in. Waiting from gate-in to service start includes check-in and yard activity; do not add those components again to total dwell. If the truck waits outside the gate, gate-to-gate dwell misses that queue. A geofence or check-in call can supply a separate external-arrival timestamp, subject to quality and privacy controls.

### 5.3 Assess demand and resource capacity

Forecast visits by day and time using expected inbound and outbound demand, cancellation, no-show and seasonality. Segment by handling type and equipment needs. A full-truckload pallet unload, floor-loaded container and live outbound load may occupy a door for very different durations. Averages alone hide the tail that causes knock-on delays.

For a simple upper bound, compatible door-hours divided by average occupancy gives a nominal visit capacity. This ignores shared labor, yard moves, shift breaks, setup, variance and arrival bunching. Determine whether labor or equipment is the bottleneck before using a door-only calculation for booking.

For each proposed slot, test compatibility, available labor, expected duration, staging capacity and the next appointment. Protect a deliberate buffer for uncertainty. A deterministic schedule based on average service time may look feasible while regularly running late. Use historical service-time distributions, simulations or conservative quantiles appropriate to the risk and validate with operations.

### 5.4 Schedule and communicate

Publish available slots based on actual resource capacity. Capture the carrier's requested and confirmed interval, reference number, load attributes, required equipment, contact method and cancellation deadline. A slot is a capacity commitment subject to its documented conditions, not an unconditional guarantee that unloading starts on arrival.

At a cut-off, freeze the near-term plan enough for labor and yard teams to act, while retaining an exception path. Sequence on readiness and safety as well as appointment time. For example, an early truck can be staged if space permits; it should not automatically displace an on-time truck with a confirmed slot. Communicate changes to the carrier and warehouse rather than silently moving a calendar entry.

### 5.5 Manage the live yard

At gate-in, validate identity, load reference, safety requirements and yard space. Assign a staging location and maintain the current trailer location through each move. When the destination door is ready, check equipment compatibility, secure the trailer and apply the site's safety procedure before handling begins. The specific safety procedure belongs to the site and applicable law, not a generic analytics rule.

When a truck is late, compare updated arrival probabilities and service requirements against the remaining schedule. Options include holding the door briefly, swapping with a compatible ready truck, rebooking or escalating. The decision must consider downstream trucks, driver time and labor availability, not just the late truck's predicted arrival. Record the reason and the authorized decision.

### 5.6 Reconcile costs and contractual clocks

Detention is determined by the contract and documented facts, not by an arbitrary gate-to-gate threshold. Determine when the contractual clock starts and stops, exclusions, free time, applicable rate and approval process. Preserve original scans and corrections. An early arrival may not start the same clock as an on-time arrival; verify the agreement rather than generalizing.

Compare carrier invoice, appointment, gate and handling records. Differences can arise from time-zone conversion, missing scans, rebooking, driver check-in and different definitions of readiness. Resolve disputes with an audit trail. Report both total operational dwell and contractual billable time; one cannot substitute for the other.

### 5.7 Measure the system and pilot changes

Track on-time arrival against the applicable confirmed appointment, schedule adherence, external wait, gate-to-gate dwell, door occupancy, service time, overtime, safety incidents, yard occupancy and carrier charges. Report counts and distributions, including upper tails, not only averages. Separate inbound and outbound visits and distinguish live loads from drop-and-hook.

Before changing slot length or introducing a booking system, define a comparison population and outcome window. A pilot that cuts on-site dwell by forcing trucks to queue outside has moved the queue rather than reduced total waiting. Compare external and internal time, carrier and warehouse costs, service and safety. Where feasible, randomize comparable days or facilities; otherwise explain confounding from volume, staffing and season.

### 5.8 Build a resource-feasible appointment book

Create a compatibility matrix before publishing slots.
Rows can represent doors and columns can represent handling requirements such as refrigeration, trailer configuration and loading equipment.
An incompatible combination is forbidden, not merely expensive.
Have the facility owner approve the matrix and date each version.

Translate expected work into resource requirements.
A visit may need one door, two trained workers and one forklift for different portions of its occupancy.
The door can be available while a shared forklift is already committed elsewhere.
Check overlapping requirements at the level of each constrained resource.

Account for shift start, shift end, breaks and planned maintenance.
If the operation changes crews between shifts, allow for a safe handover where needed.
Do not assume that an appointment ending at the shift boundary can absorb an overrun without overtime.
Make the cost and authorization of overtime visible in the schedule.

Use duration distributions appropriate to load type and readiness, not only a single overall mean.
The amount of buffer depends on the cost of waiting, labor availability and acceptable service risk.
Backtest proposed buffers against past arrival and handling patterns.
Historical performance can be a poor guide after a change in product mix or operating procedure.

### 5.9 Handle exceptions without losing the audit trail

For an early arrival, confirm the applicable booking and availability of safe staging.
If an earlier compatible opening exists, consider its effect on already confirmed appointments and labor.
Record whether the carrier accepted an earlier service opportunity.
Early service is not evidence that the original appointment was different.

For a late arrival, obtain an updated estimated time of arrival and reason where available.
Compare holding the slot, swapping with a ready compatible load, and rebooking.
Evaluate the effect on downstream visits, labor and the driver's remaining working time.
Only an authorized party can confirm a revised booking; retain both booking versions.

For a no-show, wait until the agreed tolerance and verification process have elapsed.
Attempt the specified contact channel and record the outcome.
Release capacity according to the site policy while preserving the appointment history.
A missing gate event alone is not conclusive if the gate system was unavailable.

For an incompatible or unready load, identify the missing prerequisite before assigning a door.
Examples include a missing reference, unavailable handling equipment or an unresolved safety issue.
Escalate to the designated owner and provide the carrier with an expected next update.
Do not bypass a safety control to improve the measured service-start time.

### 5.10 Reconstruct a visit and a charge

Join the appointment to the physical visit using reliable shipment and carrier references.
One appointment can cover several trailers, and a trailer can have multiple moves.
Resolve the mapping before aggregating timestamps; otherwise a join can duplicate both dwell and charges.
Keep unmatched visits in an exception queue.

Construct non-overlapping intervals with explicit endpoints.
For a simple live unload, these may be external arrival to gate-in, gate-in to door-in, door-in to door-out, and door-out to gate-out.
They sum to external arrival through gate-out only when all events exist and are ordered.
Check negative or implausible durations against the source, time zone and correction history.

Use the actual contract to define billable time.
The start may depend on a confirmed appointment, readiness or check-in, and exclusions may apply.
Compare the resulting evidence with the carrier's invoice and document disagreements.
Operational dwell is still useful even when none of it is chargeable.

### 5.11 Govern the yard as a system

Maintain a current record of trailer identity, location, status and last confirmed movement.
A planned move is not a completed move.
Reconcile the digital yard map with physical checks and investigate missing or duplicate locations.
Assign ownership for stale records and for correcting move events.

Set a practical maximum staging occupancy that reflects safe circulation, not just painted spaces.
When occupancy approaches the limit, coordinate inbound flow and alternative staging with carriers.
Measure any waiting displaced outside the gate as well as the internal queue.
Do not treat external congestion as a successful reduction in dwell.

Review the schedule and yard state with transport, gate, warehouse and labor leads.
Publish the next decision point for each unresolved exception.
Use a brief retrospective on recurring causes to update compatibility, duration assumptions or booking rules.
Keep contractual changes separate from operational forecasts and internal targets.

### 5.12 Choose the commercial promise

**Input:** demand by shipment type, carrier requirements, customer deadlines and the resource calendar.
**Output:** an appointment product with documented commercial assumptions.
Specify the booking horizon, slot width, service conditions, cancellation process and exception channel.
State whether the slot is an arrival window or a target service start.
These alternatives allocate risk differently.

A wide arrival window gives the facility flexibility but may increase time protected by the carrier.
A service-start commitment requires the facility to reserve sufficient resources and manage readiness.
An exact-time promise is valuable only if the facility can honor it.
Compare the full-system consequences, not just the number of available calendar slots.
Record the assumptions in procurement and operating documents.

A dedicated carrier may consistently choose an early slot.
A long-haul carrier with an uncertain upstream pickup may need a wider window or rebooking mechanism.
Neither should be promised a service level the facility cannot deliver.
Ask carriers how the available windows affect equipment use, driver scheduling and future tenders.
Do not assume that a carrier's unmeasured benefit is already a shipper cash saving.

An additional receiving shift may reduce waiting, protect production and avoid downstream overtime.
It also costs labor, supervision, utilities and equipment.
Compare alternatives over the same time period and volume.
List incremental labor, equipment, software, carrier charges and customer-service consequences.
Count released capacity as a benefit only if there is an actionable use for it.

### 5.13 Translate the promise into bookable supply

**Input:** door compatibility, shift roster, equipment plan and handling requirements.
**Output:** capacity by time interval and visit class.
Start with each door's actual open interval, not the building's nominal opening hours.
Remove maintenance, cleaning, handover and other unavailable work.
Do not subtract a break twice if the labor calendar already excludes it.

Record the number and skills of workers available in each interval.
Specify workers and equipment needed for each part of a visit.
A two-person crew may be needed during handling but not during paperwork after door-out.
A forklift shared by several doors cannot be committed to all of them at once.
The combined resource calendars determine the binding supply.

Add compatibility restrictions before offering slots.
A temperature-controlled shipment may require a specific door and handling procedure.
A trailer configuration may require a particular leveler or unloading method.
The local safety and operations owners decide eligibility.
The scheduler enforces the approved rules rather than trading away a safety constraint.

Reserve capacity for forecast error and interruptions.
Publish remaining capacity by visit class, rather than allowing every carrier to select every empty-looking door.
If demand exceeds supply, escalate before issuing a confirmation.
Choices include another shift, a different day or a carrier-accepted window.
Overbooking is a policy decision with measurable downside, not a substitute for a capacity calculation.

### 5.14 Forecast demand and handling time

**Input:** historical visits, future orders, promotions and booking patterns.
**Output:** a forecast by interval and handling category, with a range and exclusions.
Segment only as finely as the evidence supports.
For sparse categories, combine similar loads or use operational estimates with uncertainty noted.
Preserve rare high-impact categories rather than hiding them in an overall mean.

Distinguish booked demand from demand that would have booked if capacity were available.
A full calendar can conceal refused requests and manual calls.
Capture requests, refusals, cancellations and rebookings where possible.
If only confirmed appointments exist, call the series observed bookings, not total latent demand.
Record changes in eligibility and no-show definitions before comparing periods.

Arrival variation and handling variation can be related.
A late load may also have missing documents or encounter a shift change.
Sampling each independently in a simulation can understate congestion.
Use observed joint patterns where representative data exist.
Test peak days, long handling events and realistic sequences, not only the average day.

### 5.15 Construct and revise a daily plan

**Input:** confirmed appointments, current ETAs, readiness and resource availability.
**Output:** a door and crew sequence plus unresolved decisions.
Begin with confirmed obligations and safety and compatibility restrictions.
Check whether freight, documents, workers and equipment are actually ready.
Represent each visit with an expected interval and resource consumption.

For a manual plan, order visits by appointment and check each resource conflict.
Move a visit only if the new sequence respects its promise and other constraints.
Allow for duration variation and define a procedure for larger overruns.
List unassigned visits with reason, owner and next update time.
An unassigned visit must not disappear because the calendar has no space for it.

Before changing the plan, identify everyone affected.
Inform the warehouse team if a door sequence changes and the carrier if its commitment changes.
A revised ETA is not permission to change a confirmed appointment.
If the carrier accepts a new window, record agreement time and the previous booking.
Retain both original and latest commitments for service reporting.

### 5.16 Run the gate-to-door handoff

**Input:** approaching visit identity, appointment, gate capacity and yard state.
**Output:** an admitted or held visit with a known next step.
Capture an external-arrival timestamp and source if the measurement is reliable.
A geofence requires attention to location uncertainty, consent and false crossings.
A driver call establishes a call time, not necessarily physical position.

At the gate, verify the reference, vehicle and load under site policy.
Check whether a safe staging location or suitable door is available.
Record admission or the reason for holding the vehicle.
If the yard is full, communicate an approved alternative holding instruction.
Do not direct drivers to a public road without an authorized safety and traffic plan.

Create or update the physical visit record and assign a confirmed location.
Request a move when a compatible door and crew are ready.
A requested move is a plan; its completion establishes the physical location.
Confirm the trailer is secured and released for handling under the site's procedure.
Assign an owner to a failed handoff rather than inventing a timestamp.

### 5.17 Coordinate live loads and drop trailers

In a live load, the driver's gate-to-gate time is operationally important.
The tractor, trailer, door and crew interact in one visit, but their resource intervals differ.
In a drop operation, a carrier may leave one trailer and later collect another.
The driver can depart quickly while the dropped trailer remains for hours or days.
Measure each trailer's dwell and the driver's visit separately.

For a drop, capture the inbound trailer identity, condition and physical location.
Create a handling task tied to the trailer and shipment, not just the departed driver.
Record completion when a yard jockey moves the trailer to a door.
After handling, track it to a designated location and mark it available for collection.
Release the correct trailer only after the site's verification and safety steps.

For a hook, check that the tractor receives the intended trailer and documents.
The collecting reference can differ from the reference used when the trailer arrived.
Allow many-to-many relationships between shipments, appointments and visits where needed.
Do not infer that a driver waited through a multi-day trailer dwell.
Trailer-pool charges require their own contractual rules and evidence.

### 5.18 Control staging and yard-move capacity

Count usable positions by type and time, including trailer and refrigeration restrictions.
Reserve space for circulation, emergency access and site-defined safety requirements.
Account for blocked, damaged or temporarily unavailable positions.
The safe working limit may be below the number of painted spaces.
Have the operating owner approve it and define an earlier escalation threshold.

Track each position as available, occupied, blocked or unknown.
Unknown is not equivalent to available.
Compare physical counts with digital inventory at a defined cadence.
Record corrections without erasing the underlying move history.
Use an approved manual record during an outage and reconcile it after recovery.

Yard jockeys, tractors and operators are shared resources.
Estimate moves per visit and move-duration distributions, including travel and handoff.
A physically empty door may remain unusable if no mover can position the next trailer.
Prioritize moves using safety, commitments, readiness and downstream effects.
Count repositioning caused by a poor initial assignment as real capacity consumption.

### 5.19 Apply a late-arrival decision rule

**Input:** confirmed appointment, arrival forecast, ready work and remaining resources.
**Output:** hold, swap, rebook or escalate, with a timestamp and reason.
Check the applicable tolerance and whether the ETA is sufficiently fresh.
Identify the next decision time and the latest point another load could use the capacity.
Evaluate a range when the ETA is uncertain.

Holding protects the original visit if it arrives soon.
The downside is idle door and labor time if it does not.
Swapping with a compatible ready load uses the interval but the replacement may overrun.
Rebooking can restore a feasible plan but impose carrier delay and require agreement.
Escalate when safety, customer priority or decision authority is unclear.

Compare incremental costs over the same horizon.
Include downstream delay, additional moves and labor, charge exposure and credible service impacts.
Exclude costs common to all options from the choice calculation.
Show the range of an unknown cost at which the recommendation changes.
The dispatcher must reject a plan that violates a non-negotiable constraint.

### 5.20 Handle overruns, cancellations and interruptions

An overrun becomes actionable when work cannot release a resource at its planned time.
Identify whether load condition, equipment, labor, paperwork or an external dependency is responsible.
Update expected release and identify affected visits.
Consider a compatible alternative door or resequencing, including move time and resource overlap.
Do not call a delayed visit on time merely because its appointment was changed afterward.

For a cancellation, record receipt time and who confirmed it.
Release capacity after applying the agreed rule and checking whether service is still needed.
Preserve both transactions if another carrier accepts the opening.
Measure usable recovered capacity separately from the number of cancellations.
A slot canceled too late to fill may have little practical value.

For a power, network or equipment interruption, activate the site's contingency procedure.
Identify which records can be captured manually and who may make assignments.
Communicate operating status and next update time to affected carriers.
Reconcile manual entries with automated records and flag uncertain times after recovery.
Never invent events merely to make an event chain complete.

### 5.21 Define the management scorecard

**Input:** reconciled visits, appointment versions, resource calendars and financial data.
**Output:** a scorecard with owners, denominators and action thresholds.
Report eligible visits and data completeness beside every percentage.
Calculate on-time arrival using the applicable appointment and agreed tolerance.
Also report performance against original commitments when changes are frequent.

Show median and upper-tail external wait, gate-to-gate dwell and door occupancy by visit type.
Include observation counts and the share of visits still open at the cutoff.
Report utilization by resource and interval, not only as a daily door average.
Show yard occupancy against the safe operating limit and distinguish unknown positions from empty ones.
Include unplanned moves and missing handling prerequisites as process indicators.

Show approved and disputed carrier charges separately.
Reconcile exposure to the financial ledger rather than assuming all exposure was invoiced.
Track incremental labor, overtime and software using the same period as benefits.
Report customer-service and safety guardrails beside the economic result.
A favorable average is not success if a vulnerable class bears disproportionate delay.

### 5.22 Run a defensible pilot

Define the eligible population before rollout, including excluded visit types and reasons.
Choose the decision supported, such as whether to purchase software or alter slot lengths.
Specify the comparison, outcome window, implementation cost and guardrails before observing results.
Randomize comparable days or facilities where practical while protecting commitments.
Otherwise document differences in volume, load mix, staffing and season.

Track adherence to the intervention, not only assignment to it.
A treatment day may revert to manual scheduling after an outage.
Where randomized, compare eligible visits by assigned condition as the main analysis.
Also report adoption and deviations to explain feasibility.
Do not remove unusually bad treatment days after seeing the results.

Measure spillovers to neighboring days, carriers and external staging.
Postponing trucks into a control period contaminates the comparison.
Report unresolved visits, missing events and uncertainty in the estimated difference.
Test whether cost results survive plausible unmeasured external waiting.
Then decide whether to scale, modify or stop the intervention.

## 6. Worked example

A fictional facility receives 40 comparable live-unload trucks per day. Four compatible doors operate for 8 hours per day. Each truck occupies a door for an average of 40 minutes. Assume door availability and labor are aligned for this simple capacity bound; breaks, variance and safety activities are included in the observed occupancy only if explicitly measured.

Available door time is 4 × 8 × 60 = 1,920 door-minutes per day. Required average occupancy is 40 × 40 = 1,600 door-minutes per day. The nominal utilization is 1,600 / 1,920 = 83.3%. The arithmetic upper bound is 1,920 / 40 = 48 trucks per day, not a promise that 48 randomly arriving trucks can be handled on time. With variable arrivals and service duration the required buffer may be material.

The facility considers a booking and staging change. In the baseline, 40 visits have 90 minutes mean gate-to-gate dwell, of which 20 minutes per visit on average are contractually billable at USD 60 per hour. In the proposed scenario, mean billable time is forecast to fall to 8 minutes per visit. The rate is a fictional contract assumption and applies only to the billable intervals, not to every dwell minute.

Baseline carrier charge: 40 × (20 / 60) × USD 60 = USD 800 per day.

Proposed carrier charge: 40 × (8 / 60) × USD 60 = USD 320 per day.

Expected carrier-charge saving: USD 480 per day.

Suppose the intervention adds USD 180 per day in staffing and USD 80 per day in software and administration, both incremental over the relevant horizon. Expected net saving is USD 480 − USD 180 − USD 80 = USD 220 per day. This is a forecast, not an observed causal result. It excludes any change in external waiting, customer service or other costs; those omissions can reverse the decision.

The break-even reduction in average billable minutes is USD 260 / (40 × USD 1 per minute) = 6.5 minutes per truck. The assumed reduction is 12 minutes. If only 5 minutes are saved, the carrier-charge benefit is USD 200 per day and the intervention loses USD 60 per day before other effects.

### Trace one physical visit

Suppose a driver reaches the external queue at 09:12 and enters the gate at 09:27.
The trailer reaches the assigned door at 09:42 and handling begins at 09:48.
Handling ends at 10:18, the trailer leaves the door at 10:25, and it exits the gate at 10:37.
All times are local and on the same fictional operating day.

External waiting is 15 minutes.
Gate-in to door-in is 15 minutes.
Door occupancy is 43 minutes, including 6 minutes before handling, 30 minutes of handling and 7 minutes after handling.
Door-out to gate-out is 12 minutes.
External arrival to gate-out is 85 minutes: 15 + 15 + 43 + 12.

Gate-to-gate dwell is 70 minutes, from 09:27 to 10:37.
It already includes the 43 minutes of door occupancy.
Adding door occupancy again to gate-to-gate dwell would overstate total time.
The intervals describe one visit; the daily 40-minute mean occupancy above describes a different aggregate.

### Apply a hypothetical contract

For illustration only, assume this carrier's contract starts its clock at 09:30 and stops it at gate-out.
Assume 45 minutes of free time, no exclusions, and a continuous USD 60 per hour charge thereafter.
The contractual interval is 67 minutes, from 09:30 to 10:37.
Billable time is therefore 67 − 45 = 22 minutes.
The charge is 22 / 60 × USD 60 = USD 22.

If the contract instead starts the clock at door-in, the interval is 55 minutes.
With the same hypothetical free time and rate, only 10 minutes are billable, for USD 10.
This comparison is not an interpretation of a real carrier agreement.
It demonstrates why the contractual clock cannot be inferred from the operational dwell metric.

### Test a queue-displacement scenario

Suppose an intervention reduces mean gate-to-gate dwell by 20 minutes per truck but increases external waiting by 15 minutes per truck.
Across 40 trucks, the net reduction in total observed external-arrival-to-exit time is 40 × (20 − 15) = 200 truck-minutes per day.
That is 3 hours and 20 minutes across the day's visits, not 20 minutes saved per truck end to end.
Any unobserved waiting before the external-arrival event could reduce the apparent benefit further.

If carrier compensation or road safety depends on the external wait, include those effects in the decision.
If an external queue cannot be measured reliably, do not present the on-site reduction as a system-wide saving.
Compare like periods and load types, and record staffing and arrival-volume differences.
These additional assumptions are separate from the USD 220 daily forecast above.

### Extend the case: resource bottleneck

The daily calculation assumes labor is aligned with the four doors.
Now suppose the facility has two forklifts, each available for 7 productive hours per day.
Each of the 40 visits needs one forklift for 25 minutes.
Assume these productive hours already exclude breaks and maintenance.
This is a separate scenario, not a revision to the initial USD 220 forecast.

Available forklift time is 2 × 7 × 60 = 840 forklift-minutes per day.
Required forklift time is 40 × 25 = 1,000 forklift-minutes per day.
The shortfall is 160 forklift-minutes per day.
The average-time forklift bound is 840 / 25 = 33.6 visits per day.
Thus the labor-and-equipment assumption is not met even though average door utilization is 83.3%.

The 33.6 figure is a workload bound, not a schedulable fraction of a truck.
A practical plan needs whole visits and compatible time slots.
Adding a third forklift for three productive hours supplies 180 forklift-minutes per day.
It covers the average daily shortfall by 20 minutes, but does not prove the hourly schedule is feasible.
The third forklift also requires an available qualified operator where the work demands one.

Suppose the extra forklift and operator cost USD 150 per day on an incremental basis.
Suppose alternatively that 7 visits per day can be shifted to another facility at USD 30 per visit.
The shift costs USD 210 per day before any other transport or service effects.
On these limited assumptions, the extra resource is USD 60 per day cheaper.
Do not select it until its actual hours cover the peak and its operator and safety requirements are confirmed.

### Extend the case: hourly conflicts

Consider just one hour, 10:00–11:00.
Two compatible doors are open and a single forklift is available.
Two trucks each need 40 door-minutes and 30 forklift-minutes.
If both trucks start at 10:00 and each needs the forklift for its first 30 minutes, the plan conflicts.
The existence of 120 door-minutes and 60 forklift-minutes in the hour does not make simultaneous use feasible.

If all other requirements permit, schedule the first forklift interval at 10:00–10:30.
Schedule the second at 10:30–11:00.
The second truck occupies its door from 10:30 until 11:10, so it extends beyond the hour.
If the next commitment starts at 11:00, this revised sequence still conflicts with that commitment.
The planner must move the second visit, obtain another resource or revise the next commitment with authority.

The example illustrates why capacity is both a total and a time profile.
A daily shortage may be hidden by unused capacity at the wrong hour.
Conversely, a daily total that looks tight may be feasible if work can move to an agreed off-peak window.
Check the actual dates and intervals for each resource.
Only confirmed and compatible shifts are available to the schedule.

### Extend the case: staging headroom

Suppose the site has 30 painted trailer positions.
Four are unavailable because of blocked access or maintenance, and two must remain clear under the site's safety plan.
The effective safe maximum is 30 − 4 − 2 = 24 occupied positions.
At 09:00, 19 positions are occupied and two have unknown status.
If the unknown positions might be occupied, conservatively available headroom is 24 − 19 − 2 = 3 positions.

Four trucks are due to enter before a known departure.
Admitting all four without another action could exceed the working limit.
The coordinator can verify the unknown locations, accelerate an already-ready authorized release, or offer an approved alternative staging plan.
The coordinator must not assume an unknown position is empty.
Any outside delay must be included in the measured end-to-end visit.

Now suppose the physical check confirms that both unknown positions are empty.
Headroom becomes 24 − 19 = 5 positions.
All four arrivals can be admitted on this simplified space calculation.
The check does not establish that their doors, crews or equipment are available.
It also does not make all five positions interchangeable if certain trailers need special facilities.

### Extend the case: late-load choice

At 10:00, truck A has the confirmed slot but its updated arrival is uncertain.
A compatible ready truck B is waiting.
Assume A has a 60% chance of arriving by 10:15 and a 40% chance of arriving at 10:45.
These probabilities are fictional forecasts, not observed frequencies.
Assume all actions below satisfy the relevant safety and contract rules.

The manager compares holding the door for A with serving B first.
Under hold, forecast incremental cost is USD 30 if A arrives by 10:15 and USD 150 if it arrives at 10:45.
Under swap, forecast incremental cost is USD 90 in the early-A case and USD 40 in the late-A case.
These costs already include the modeled downstream waiting and additional moves.
They exclude costs identical under both choices.

Expected hold cost is 0.60 × USD 30 + 0.40 × USD 150 = USD 78.
Expected swap cost is 0.60 × USD 90 + 0.40 × USD 40 = USD 70.
Swap is cheaper by USD 8 under the stated probabilities and costs.
That difference is small relative to the uncertain inputs.
Communicate the swap to affected parties and obtain any authorization required before execution.

Let `p` be the probability of A arriving by 10:15.
Hold costs `30p + 150(1 − p) = 150 − 120p` dollars.
Swap costs `90p + 40(1 − p) = 40 + 50p` dollars.
The two options break even at `p = 110 / 170`, approximately 64.7%.
Above that threshold hold is cheaper; below it swap is cheaper, given the stated cost scenarios.

This threshold is not a universal late-truck policy.
It changes with each visit's consequences, any contractual entitlement, resource availability and arrival uncertainty.
The model must not assign a finite cost to an action forbidden by a safety constraint.
If the ETA feed is stale, obtain an update before relying on a finely calculated threshold.
Keep the human decision and its reason in the exception record.

### Extend the case: nonlinear charges

Suppose two visits have 10 and 50 billable minutes under a fictional contract.
The rate is USD 60 per hour, with each positive charge rounded up to a full hour.
The first visit costs USD 60 and the second also costs USD 60.
Total charge is USD 120, although the two billable intervals total only 60 minutes.
Applying the rate once to the combined 60 minutes would incorrectly produce USD 60.

Suppose an intervention reduces both visits by 10 billable minutes.
The first visit now has zero billable minutes and incurs no charge under the stated rule.
The second has 40 billable minutes and still rounds to USD 60.
The total is USD 60, a saving of USD 60.
The result depends on the explicit per-visit rounding rule; another contract can yield a different answer.

### Extend the case: yard flow and Little's law

Suppose a stable drop-trailer operation completes an average of 12 trailer visits per day.
Suppose mean trailer time in the defined yard system is 1.5 days.
Under a stable system with consistent boundaries, Little's law gives average inventory of 12 × 1.5 = 18 trailers.
The relevant time is the trailer's time in the defined system, not the driver's gate visit.
The estimate is a long-run average, not a peak-space requirement.

If mean trailer time falls to 1.0 day at the same throughput, the corresponding average is 12 trailers.
The implied reduction is six trailers on average.
Peak occupancy could still exceed the safe limit when arrivals cluster.
Little's law does not specify the arrival pattern or a required safety buffer.
It should not be applied as a steady-state prediction during an accumulating backlog.

### Extend the case: operating decision

The original forecast offers USD 220 per day before omitted costs.
The forklift shortage demonstrates that the assumed capacity alignment must be checked before treating that forecast as feasible.
The staging example shows that a door-only plan can fail even with physical door capacity.
The late-load example shows that a narrow expected-cost advantage may be sensitive to ETA quality.
The nonlinear billing example shows why per-visit contract evaluation can differ from an aggregate estimate.

The proposed decision is therefore conditional.
First validate the resource calendars and the space and movement plan.
Then estimate charges at the contract and visit level and identify outside waiting.
Pilot the change with service and safety guardrails and record actual adoption.
Only the measured incremental outcome can support an investment decision for this facility.

## 7. Code, query, or calculation

The following Python 3 calculation reproduces the example. It assumes a continuous billable rate and no minimum charge or rounding rule. Actual contracts may differ.

```python
trucks = 40
doors = 4
hours = 8
occupancy_minutes = 40
baseline_billable = 20
proposed_billable = 8
rate_per_hour = 60.0
extra_cost = 180.0 + 80.0

available = doors * hours * 60
utilization = trucks * occupancy_minutes / available
baseline = trucks * baseline_billable / 60 * rate_per_hour
proposed = trucks * proposed_billable / 60 * rate_per_hour
net = baseline - proposed - extra_cost
break_even_minutes = extra_cost / (trucks * rate_per_hour / 60)
print(f"Door utilization: {utilization:.1%}")
print(f"Net saving: ${net:.2f}/day")
print(f"Break-even reduction: {break_even_minutes:.1f} min/truck")
```

Expected output:

```text
Door utilization: 83.3%
Net saving: $220.00/day
Break-even reduction: 6.5 min/truck
```

This calculation does not simulate queueing or validate the rate. Recalculate at the contract's actual rounding interval, minimum and exclusions, using individual visit records where charges are nonlinear.

### A visit-level calculation

The next standalone Python 3 example checks a complete, ordered visit and calculates a fictional charge.
All times are on the same date and use the same local time zone.
The assumed contract starts at check-in, provides 45 free minutes and charges USD 60 per hour continuously.
There are no exclusions, minimums or rounding rules.
The example is not a general contract interpreter.

```python
from datetime import datetime


def timestamp(hhmm):
    return datetime.strptime("2026-05-12 " + hhmm, "%Y-%m-%d %H:%M")


events = {
    "external_arrival": timestamp("09:12"),
    "gate_in": timestamp("09:27"),
    "check_in": timestamp("09:30"),
    "door_in": timestamp("09:42"),
    "service_start": timestamp("09:48"),
    "service_end": timestamp("10:18"),
    "door_out": timestamp("10:25"),
    "gate_out": timestamp("10:37"),
}

ordered = list(events.values())
if ordered != sorted(ordered):
    raise ValueError("Visit events are out of order")


def minutes(start, end):
    return (events[end] - events[start]).total_seconds() / 60


external_wait = minutes("external_arrival", "gate_in")
yard_wait = minutes("gate_in", "door_in")
door_occupancy = minutes("door_in", "door_out")
exit_time = minutes("door_out", "gate_out")
gate_dwell = minutes("gate_in", "gate_out")
total = minutes("external_arrival", "gate_out")
contract_minutes = minutes("check_in", "gate_out")
billable_minutes = max(0, contract_minutes - 45)
charge = billable_minutes / 60 * 60

print(f"External wait: {external_wait:.0f} min")
print(f"Gate to door: {yard_wait:.0f} min")
print(f"Door occupancy: {door_occupancy:.0f} min")
print(f"Door out to gate out: {exit_time:.0f} min")
print(f"Gate dwell: {gate_dwell:.0f} min")
print(f"External arrival to exit: {total:.0f} min")
print(f"Billable: {billable_minutes:.0f} min; charge: ${charge:.2f}")
```

Expected output:

```text
External wait: 15 min
Gate to door: 15 min
Door occupancy: 43 min
Door out to gate out: 12 min
Gate dwell: 70 min
External arrival to exit: 85 min
Billable: 22 min; charge: $22.00
```

The disjoint intervals sum to 85 minutes.
Gate dwell is the last three intervals, not an additional interval to add to them.
The billable clock starts at check-in in this example, not necessarily in another contract.
For production, handle time zones, daylight-saving ambiguity, multiple visits and corrections explicitly.
Missing or conflicting events require reconciliation; do not replace them with invented defaults.

## 8. Data, standards, and regulatory context

Use stable shipment, appointment, visit, carrier, tractor, trailer, door and event identifiers. A shipment may involve multiple visits and a trailer may serve multiple shipments. Do not assume a one-to-one relationship. Record event source, time zone, ingestion time, event version, operator and correction reason. Reconcile the appointment ledger with gate, warehouse and carrier records.

GS1 [EPCIS](https://www.gs1.org/standards/epcis) provides a common model for supply-chain events. It does not define a detention contract or guarantee that a gate scan is accurate. Electronic data interchange or application programming interface messages may communicate appointment and shipment information, but the mapping, ownership and timing must be agreed with each partner.

In the United States, OSHA's [Powered Industrial Trucks standard, 29 CFR 1910.178](https://www.ecfr.gov/current/title-29/section-1910.178) includes requirements relevant to covered powered industrial truck operations. Applicability and the complete site safety program require qualified review. A scheduling model must not override safety rules, driver hours rules or site procedures in pursuit of utilization. Other jurisdictions and operations have different requirements.

Carrier contracts and applicable law govern detention and related charges. This curriculum does not state a universal free-time allowance or a legal entitlement to charge. Vehicle location and driver identifiers can be personal data; apply appropriate access, retention and privacy controls, with local legal review where needed.

### Event model and ownership

Separate the appointment ledger from the physical event ledger.
The appointment ledger records a request, confirmation, cancellation and authorized revisions.
The event ledger records observations such as gate entry, move completion and release.
A third ledger can record contract terms and the calculation version applied to an invoice.
Each has a different owner and correction process.

| Object | Grain | Required relationship | Accountable operational owner |
|---|---|---|---|
| Appointment | One booking and version | May relate to one or more shipment references | Booking team |
| Visit | One defined gate-to-gate visit | May relate to several shipments or trailers | Gate or yard team |
| Trailer stay | One trailer's defined presence at a site | Can span multiple driver visits | Yard team |
| Move | One requested and completed relocation | References trailer and locations | Yard team |
| Handling task | One loading or unloading activity | References visit, shipment or trailer as appropriate | Warehouse team |
| Charge claim | One invoice line or contractual claim | Linked to supporting evidence and contract | Accounts payable or transport |

Define the grain before building any dashboard.
An appointment may be canceled and replaced without a physical visit.
A visit can exist without an appointment when an unplanned truck arrives.
A trailer can remain in the yard after the driver has departed.
A handling task can span multiple doors or shifts.
These cases should be represented, not discarded as impossible duplicates.

Every event needs an event type, event time, source, location and entity reference.
Preserve the original source identifier and ingestion time.
The correction record should include the corrected value, author, reason and effective time.
Use a traceable rule to identify the currently accepted event version.
Retaining the raw event permits later review of a disputed correction.

### Reconciliation controls

1. Count appointments, physical visits and invoice claims separately for the reporting period.
2. Identify appointments with no matched visit and visits with no matched appointment.
3. Separate expected cancellations from unexplained missing visits.
4. Match by stable identifiers and additional evidence, not by time proximity alone.
5. Route ambiguous matches to an accountable person instead of selecting one arbitrarily.
6. Validate event order within each physical visit.
7. Check time-zone conversion, daylight-saving transitions and cross-midnight visits.
8. Distinguish a missing event from an event that occurred at the same timestamp as another.
9. Reconcile approved charges to the financial ledger without multiplying them across joined events.
10. Publish match rate, unresolved count and correction age next to the resulting metrics.

For example, one visit has four gate or door events and two invoice lines.
A direct join on visit ID creates eight rows.
Summing the invoice amount on that joined table quadruples the charge.
Aggregate each source at its intended grain before combining measures.
Keep the underlying rows available for investigation.

An open visit at the reporting cutoff has an unknown final dwell.
Dropping all open visits preferentially removes longer visits from a recent cohort.
Report the count and age of open visits and use a sufficiently mature cohort for completed-dwell comparisons.
Where survival methods are appropriate, state their censoring assumptions and data requirements.
Do not assign a gate-out timestamp merely because a reporting period ended.

### Timestamp quality and time zones

Event time is when the event is asserted to have happened.
Ingestion time is when the receiving system obtained the message.
The difference can reveal a delayed integration but is not itself physical waiting.
A manually entered correction can have an old event time and a recent ingestion time.
Both should remain available in the audit trail.

Store a time-zone identifier or offset with local timestamps where available.
Convert to a common timeline before subtracting events from different systems.
Preserve the original value and source because conversions can be corrected.
A repeated local clock time during a daylight-saving transition is ambiguous without further information.
Send unresolved cases for review rather than choosing a convenient offset.

Door-in recorded by a planned assignment is not the same evidence as a completed physical move.
Gate-out from an automatic sensor can also be wrong if it associates the wrong trailer or tractor.
Use source-specific quality checks and an operational process to resolve disagreements.
The most recently received message is not automatically the most accurate event.
Document which evidence is suitable for operational reporting and which is accepted under a contract.

### Access and partner exchange

Provide partners with the minimum data needed for booking, execution and dispute resolution.
Separate permissions to view, create and approve changes.
An integration can transmit an ETA frequently without granting authority to rebook.
Authenticate the source, acknowledge receipt and distinguish acceptance from message delivery.
Define recovery procedures for duplicates, retries, outages and delayed messages.

Driver identifiers, contact details and precise vehicle locations may be sensitive.
Limit access by role and purpose, and define retention and deletion requirements with the relevant owners.
Do not expose a detailed yard map or another carrier's loads in a shared booking interface.
Apply site security and applicable privacy rules in addition to operational convenience.
Seek qualified legal review for jurisdiction-specific obligations.

### Standards and obligations

GS1 EPCIS provides a vocabulary and information model for sharing supply-chain event information.
It can help partners describe what happened, where, when and in what business context.
The standard does not independently verify a physical move or confer a right to payment.
An agreed event mapping and quality controls remain necessary.
Refer to the linked GS1 material for the current specification and implementation guidance.

The cited US powered industrial trucks regulation is a legal source, not a generic dock scheduling recipe.
Determine whether it applies to the equipment, workplace and activity under review.
The facility's qualified safety owners must translate applicable requirements into procedures.
Other rules may apply to other equipment, traffic, drivers or jurisdictions.
The scheduling team should escalate uncertainty rather than substitute this lesson for legal or safety advice.

Contracts may incorporate different free time, start events, exclusions and rounding.
Keep a dated record of the agreement applicable to each claim.
An electronic event shared under a standard may be useful evidence without being the contractually agreed controlling timestamp.
Review conflicting clauses and applicable law with the proper commercial and legal owners.
The calculation should identify its source agreement and effective period.

## 9. Industry practice and case evidence

The US Federal Motor Carrier Safety Administration (FMCSA) has published research on commercial vehicle detention; its [Detention Time Study](https://www.fmcsa.dot.gov/research-and-analysis/research/detention-time-study) describes the agency's work on detention. Definitions, population and study limitations matter: a national research finding is not a contractual rule and does not establish the savings at a particular warehouse.

The US Department of Transportation's Bureau of Transportation Statistics maintains [Freight Analysis Framework](https://www.bts.gov/faf) data describing freight movement at an aggregate level. This can inform broad freight context but cannot directly supply door-level service distributions or invoice-grade detention times for a facility. [INFERENCE] Local gate and warehouse events are therefore needed for the operating decisions in this topic.

A facility considering a commercial appointment product should ask for its own baseline, implementation cost, reference sites and independent evidence. Vendor descriptions of capability do not demonstrate causality. Pilot results should report volume, staffing, external queues, exception handling and uncertainty, including any periods excluded from the analysis.

### What public evidence can and cannot answer

The FMCSA detention study is a relevant starting point for understanding the agency's research program.
Its population, data collection and operational definitions should be read before applying a result.
A national or multi-site result cannot establish the average delay at a particular warehouse.
Differences in load type, carrier participation, season and site procedures can all matter.
Use local measurements and contracts to make a facility-level business case.

The Freight Analysis Framework describes broad freight movements.
Its geographic and commodity aggregation serves a different purpose from an appointment ledger.
It can help a planner understand regional flows when considering future capacity.
It cannot tell the planner which truck occupied a particular door on Tuesday.
It also cannot validate a detention invoice or establish a driver's exact waiting time.

### Evaluate a vendor's proposed product

Ask the vendor to demonstrate an actual booking, revision, cancellation and exception workflow.
Require a demonstration of permissions: who can request, confirm, override and cancel a slot?
Check whether the original promise remains visible after a revision.
Ask how the product handles shared labor, forklifts and specialized doors.
An attractive calendar is not evidence that resource constraints are enforced.

Ask for the input events and the definitions behind every displayed metric.
Verify whether gate arrival, check-in and service start are distinct fields.
Ask how a missing gate-out, duplicate message or manual correction is represented.
Check how a drop trailer is linked to separate driver visits.
Request an export suitable for independent financial and operational reconciliation.

Discuss implementation rather than only subscription price.
Estimate integration, training, carrier onboarding, support, equipment and process-change costs.
Ask what happens during a network outage and how manual events are reconciled.
Identify the facility owner responsible for maintaining resource calendars and compatibility.
Include the ongoing workload of keeping data and permissions current.

### Interpret reference-site claims

Request the number and type of sites, period, eligible visits and baseline used in any claim.
Ask whether the result is a measured outcome, forecast, customer testimonial or modeled estimate.
Check whether external waiting and excluded visits were measured.
Ask whether staffing, customer mix and shipment volume changed at the same time.
If the evidence is unavailable, treat the claim as a hypothesis to test, not a proven benefit.

A reference site can be informative even without a controlled trial.
It can reveal adoption difficulties, integrations, exception handling and operating responsibilities.
It cannot by itself prove that the same percentage saving will occur elsewhere.
Compare the site's handling mix, yard footprint, carrier base and resource constraints with your own.
Document material differences before transferring any assumption.

### Translate evidence into a decision

Use the same horizon for forecast benefits and implementation costs.
Identify one-time costs separately from recurring daily costs.
Specify whether financial benefits are approved payments avoided, reduced overtime or usable capacity.
Show the downside if adoption or performance is below forecast.
Finance and operations should agree on which benefits are counted and who owns them.

If a pilot improves average dwell but worsens the upper tail, investigate before scaling.
A small number of delayed trucks may create major carrier or customer consequences.
If savings are positive but outside queues are unmeasured, present a conditional result.
If safety performance deteriorates, apply the predefined stop or escalation rule.
Document the decision and assumptions so later outcomes can be compared with the forecast.

## 10. Failure modes and red flags

| Failure | Warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Ambiguous dwell | Teams report different values | Different start and end events | Disputes | Publish event definitions |
| Queue displacement | Gate dwell falls, roadside queue rises | Trucks held outside | No system saving | Measure external arrival |
| Full calendar | Late cascade after one overrun | No variability buffer | Poor service | Capacity and uncertainty allowance |
| Door-only capacity | Doors idle despite bookings | Labor or equipment constraint | Missed plan | Joint resource check |
| Promise overwritten | Historical on-time rate improves | Slot changed after arrival | Misleading KPI | Version confirmed appointments |
| Incompatible door | Trailer cannot be handled | Equipment requirement omitted | Additional moves | Validate compatibility |
| Duplicate visits | Charge exceeds source record | Join fan-out | Invoice errors | Preserve native keys |
| Incorrect clock | Billable time equals total dwell | Contract ignored | Payment dispute | Apply contract-specific rules |
| Missing yard move | Trailer location unknown | Unrecorded relocation | Search and delay | Scan moves and reconcile |
| Stale ETA | Assigned door waits for absent truck | Feed delayed | Idle labor | Freshness and manual fallback |
| Unsafe pressure | Shortcuts after KPI change | Time target dominates safety | Injury risk | Safety stop rule |
| Immature outcomes | Recent visits lack gate-out | Data cutoff | Biased dwell | Report unresolved visits |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test |
|---|---|---|---|
| Compatible doors available | Simple capacity bound | Specialized equipment | Segment visits |
| Labor aligned with doors | Door-hour calculation | Shift gaps or absence | Compare staffing roster |
| Average occupancy representative | Required time calculation | Long tail | Inspect distribution |
| Billing is linear | Example cost | Rounding or minimums | Review contract |
| Costs incremental | Net saving | Allocated fixed cost | Finance review |
| Volume unchanged | Per-day comparison | Rerouted freight | Compare eligible demand |
| Outside queue unchanged | Scope of example | Gate holding policy | Measure external arrival |
| Event clocks accurate | Dwell calculation | Bad clocks or late feeds | Reconcile sources |
| No safety deterioration | Operational viability | Pressure to cut checks | Safety guardrail |
| Forecast reduction realized | Expected benefit | Adoption and variability | Controlled pilot |

## 12. Exercises

1. **Event chain** (10 minutes). List five ordered events from gate-in to gate-out and define one dwell interval. Acceptance requires explicit endpoints.
2. **Capacity** (10 minutes). Three doors operate 8 hours, with 45 minutes average occupancy per truck. Compute the nominal upper bound. Acceptance requires a caveat about variability.
3. **Dwell** (10 minutes). A truck gates in at 09:00, starts service at 10:15, finishes at 11:00 and gates out at 11:30. Compute gate-to-gate dwell and pre-service time. Acceptance requires avoiding double counting.
4. **Charges** (15 minutes). Ten visits each incur 15 billable minutes at USD 80 per hour. Calculate total charges without rounding or minimums. Acceptance requires conversion of minutes to hours.
5. **Break-even** (15 minutes). A change costs USD 200 per day and serves 20 visits per day. If billable time costs USD 1 per minute, calculate the reduction per visit needed to break even. Acceptance requires the equation.
6. **Critique** (15 minutes). Gate dwell falls by 30 minutes while external waiting rises by 40 minutes. What can you conclude? Acceptance requires the full system boundary and limitations.
7. **Design** (25 minutes). Design a pilot of revised slots, including unit of assignment, primary cost metric, service and safety guardrails and treatment of outside queues. Acceptance requires a plausible comparison and event reconciliation.

8. **Shared-resource capacity** (20 minutes).
   A site has four doors, each open for 8 hours per day.
   The mean door occupancy is 48 minutes per visit.
   Two shared forklifts each provide 6 productive hours per day.
   Every visit requires 30 forklift-minutes.
   Forecast demand is 28 visits per day.
   Assume all doors and visits are compatible and all times are productive.
   **Deliverable:** calculate both average-workload bounds and the forklift shortfall at forecast demand.
   **Acceptance:** identify the binding average resource and explain why the bound is not an hourly schedule.

9. **Contract and reconciliation** (25 minutes).
   A fictional carrier checks in at 08:10 and exits at 10:25 on the same day.
   Its contract specifies 90 minutes of free time after check-in.
   After free time, charge USD 80 per hour continuously, with no rounding or exclusions.
   The invoice claims USD 100.
   The event feed has two service events and the invoice has one line.
   **Deliverable:** compute the evidenced charge and difference, and explain a join control.
   **Acceptance:** show the clock, convert minutes to hours, and avoid multiplying the invoice line.

10. **Yard and late-arrival decision** (30 minutes).
    The safe limit is 20 occupied positions.
    There are 16 confirmed occupied positions and two unknown positions.
    Three trucks are approaching and no departure is confirmed.
    One truck is late; its current ETA has not been refreshed for 90 minutes.
    Another truck is on time and compatible with an available door.
    **Deliverable:** propose an admission and sequencing process with named owners and evidence.
    **Acceptance:** account conservatively for unknown positions, preserve commitments and identify the next update.
    Explain how external waiting and any safety restrictions enter the decision.

### Additional guidance for exercises 1–7

For exercise 1, list actual events rather than mixing an ETA with a physical observation.
Indicate whether your selected dwell includes handling.
For exercise 2, show the available door-minutes before dividing by occupancy.
Mention at least one non-door resource that can reduce the feasible total.
For exercise 3, name the endpoints of both durations.

For exercise 4, calculate the amount per visit before multiplying by the number of visits.
For exercise 5, distinguish saved billable minutes from saved total dwell minutes.
For exercise 6, assume the stated changes concern the same visits and non-overlapping intervals.
Then state why a different mix or a missing external-arrival event would limit the conclusion.
For exercise 7, state what would cause the pilot to stop or require escalation.

## 13. Self-check answers

1. One sequence is gate-in, check-in, door-in, service start, service end and gate-out. Gate-to-gate dwell runs from gate-in to gate-out; door occupancy needs a separately recorded door-out.
2. 3 × 8 × 60 / 45 = 32 trucks per day as a deterministic upper bound. Variable arrivals, changeovers, labor and breaks can reduce achievable throughput.
3. Gate-to-gate dwell is 150 minutes. Pre-service time is 75 minutes. The latter is contained within the former and should not be added to it.
4. 10 × 15 / 60 × USD 80 = USD 200. The stated arithmetic ignores contractual rounding and minimums.
5. USD 200 / (20 × USD 1 per minute) = 10 minutes per visit. This assumes every saved minute was billable and no other costs change.
6. The measured combined waiting time rises by 10 minutes per truck if both changes apply to the same comparable population and intervals do not overlap. Investigate mix, data quality and other costs before claiming a causal effect.
7. A strong design randomizes comparable days or facilities when feasible, measures total incremental cost per eligible visit and external plus internal wait, sets safety and service stop rules, retains appointment versions and reconciles unresolved events. It reports uncertainty and spillovers.

### 8. Shared-resource capacity

Four doors provide 4 × 8 × 60 = 1,920 door-minutes per day.
At 48 door-minutes per visit, the average-workload bound is 40 visits per day.
Two forklifts provide 2 × 6 × 60 = 720 forklift-minutes per day.
At 30 forklift-minutes per visit, their bound is 24 visits per day.
The forklifts are the binding average resource under these assumptions.

The 28 forecast visits require 28 × 30 = 840 forklift-minutes.
The shortfall is 840 − 720 = 120 forklift-minutes per day.
An extra two productive forklift-hours could cover the daily arithmetic shortage.
The actual schedule still needs an operator and enough availability at the required times.
Arrival bunching, handling variation and breaks can make the practical capacity lower.

### 9. Contract and reconciliation

The check-in-to-exit interval is 135 minutes.
Subtracting 90 free minutes leaves 45 billable minutes.
At USD 80 per hour, the evidenced charge is 45 / 60 × USD 80 = USD 60.
The claimed USD 100 exceeds that amount by USD 40.
The difference is a dispute to investigate, not proof of an invalid invoice.

Confirm the applicable agreement, timestamp evidence and any terms omitted from the exercise.
Keep the invoice line at its native grain when combining it with service events.
A direct join to two service events would duplicate that line.
Aggregate or select the relevant evidence before summing the invoice.
Record the reviewer, resolution and any correction to the underlying events.

### 10. Yard and late-arrival decision rubric

Count both unknown positions conservatively as potentially occupied.
The safe headroom is 20 − 16 − 2 = 2 positions.
Three approaching trucks cannot all be admitted on that conservative evidence without another change.
The yard owner should verify the two unknown positions and report the result promptly.
The gate owner should use the site-approved holding procedure until capacity is confirmed.

The dispatcher should request a fresh ETA for the late truck.
The on-time truck can be considered for the available compatible door if it is ready and all safety conditions hold.
Check the original appointment, labor and equipment before confirming that sequence.
If the late truck must be rebooked, obtain the required authorization and carrier agreement.
Record the original commitment, decision, reason and next communication time.

Capture the outside-arrival and gate-in events where reliable.
Report external waiting along with yard and door intervals.
Do not direct vehicles to an unsafe holding location to improve the gate metric.
A strong answer also considers trailer type and whether positions are interchangeable.
It assigns an owner to each unresolved location, ETA and carrier communication.

## 14. Interview questions

1. What is dwell? Strong answers name its endpoints; weak answers assume one universal definition.
2. Does 48-truck nominal capacity mean 48 appointments are safe? Strong answers discuss variation and other resources; weak answers use average occupancy alone.
3. What is the difference between ETA and appointment? Strong answers preserve both records; weak answers silently rebook when the ETA changes.
4. How do you calculate detention? Strong answers start with the contract and evidence; weak answers apply an assumed universal threshold.
5. How do you reduce waiting without shifting it? Strong answers measure outside and inside the gate; weak answers measure only on-site time.
6. What happens when a truck is late? Strong answers consider compatibility, downstream schedules and communication; weak answers always displace the next truck.
7. How can joins corrupt costs? Strong answers address shipment-to-visit cardinality; weak answers assume every key is one-to-one.
8. How do you evaluate a pilot? Strong answers include counterfactual, guardrails and uncertainty; weak answers attribute all before/after changes to the software.

## 15. Key terms

- **Appointment:** confirmed service interval; see section 4.
- **Availability:** resource time open for use; see section 5.
- **Billable time:** interval chargeable under a contract; see section 5.
- **Buffer:** capacity reserved against variation; see section 5.
- **Check-in:** administrative readiness event; see section 4.
- **Detention:** contract-defined charge for delay; see section 5.
- **Door occupancy:** door-in through door-out; see section 4.
- **Dwell:** time between named events; see section 5.
- **ETA:** forecast for a specified arrival event; see section 3.
- **External queue:** waiting before gate entry; see section 5.
- **Free time:** contractual nonbillable period; see section 4.
- **Gate-in:** verified entry event; see section 4.
- **Gate-out:** verified exit event; see section 4.
- **Live unload:** unloading while a truck visit is active; see section 5.
- **No-show:** unfulfilled appointment under stated rules; see section 4.
- **Queue:** waiting for a constrained resource; see section 4.
- **Service end:** completion of physical handling; see section 4.
- **Service start:** beginning of physical handling; see section 4.
- **Staging:** holding at a designated yard location; see section 5.
- **Utilization:** busy resource time divided by available time; see section 4.
- **Yard move:** trailer relocation; see section 4.

## 16. Sources

- GS1, [EPCIS](https://www.gs1.org/standards/epcis), supply-chain event standard.
- US Occupational Safety and Health Administration, [29 CFR 1910.178](https://www.ecfr.gov/current/title-29/section-1910.178), powered industrial trucks; current official regulatory text.
- FMCSA, [Detention Time Study](https://www.fmcsa.dot.gov/research-and-analysis/research/detention-time-study), agency research resource.
- US Bureau of Transportation Statistics, [Freight Analysis Framework](https://www.bts.gov/faf), official aggregate freight data.

## 17. Links

- [Module index](_index.md)
- [Freight modes](01_freight-modes.md)
- [Rate structures and accessorials](02_rate-structures.md)
- [Lane and tender analytics](03_lane-tender-analytics.md)
- [Last mile](08_last-mile.md)
- [ETA prediction](09_eta-prediction.md)
- [Module survey](../../../course/12_transportation_routing_last_mile.md)
- [Simulation and optimization labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md)
