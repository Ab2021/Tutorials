# Event Streams, IoT, Telematics, and EPCIS 2.0

> Module 19 — Control Towers, Visibility, and Event Intelligence  
> Topic 02 of 10  
> Estimated study time: 4 hours  
> Prerequisites: Modules 03, 06 and 12; Module 19 topic 01  
> Version 2026.5

## 1. Why this topic exists

A control tower cannot act on a delayed shipment if it cannot distinguish a missing message from a missing truck.
An operator needs evidence about the physical move, the business commitment it affects, and the time left to intervene.
That evidence arrives from carriers, warehouses, trading partners, sensors and people.
Each source observes only part of the movement.

A truck's global navigation satellite system (GNSS) position can indicate that the vehicle is near a warehouse.
It does not, by itself, establish that the intended pallet is on that vehicle.
A dock scan can establish that a particular identifier was read at a dock.
It does not establish the temperature of every item in the pallet.
A carrier's arrival message can report a milestone.
It does not necessarily prove that the gate was crossed at the stated time.

These differences matter because different interventions cost money.
A planner may reserve another delivery appointment, contact a carrier, or authorize expedited transport.
A quality specialist may put goods on hold after a temperature excursion.
A customer-service representative may revise a delivery commitment.
Each action has a different owner, decision deadline and evidentiary threshold.

Events are observations or assertions about changes and conditions.
They are not automatically the truth, and a continuous stream is not the same as complete visibility.
A source may send a duplicate after retrying, report an event hours late, or use a location identifier that another company cannot resolve.
A sensor may be offline while its last reading still appears green on a dashboard.
The resulting apparent certainty can be more dangerous than an explicit unknown.

Electronic Product Code Information Services (EPCIS) is a GS1 standard for sharing visibility events.
EPCIS 2.0 adds capabilities including sensor data and modern web interfaces, but it does not supply commercial agreements, trustworthy measurements or a common intervention policy by itself.
The standard can make independently produced statements more interpretable; it cannot make every statement correct.

This lesson starts with a business decision and works backward to event collection, interpretation and quality controls.
The example concerns temperature-sensitive food shipped on a truck, but its methods also apply to other traceable goods where legal and product-specific rules may differ.
The numbers are illustrative, not an industry benchmark.

## 2. Learning outcomes

After this lesson you can:

1. **Distinguish** a physical occurrence, an observation, an event record and an operational decision.
   Exercise 1 checks this distinction.
2. **Map** a business intervention to the minimum identifiers, timestamps and source evidence needed to support it.
   Exercise 2 asks for a decision-to-data map.
3. **Interpret** EPCIS event dimensions and choose among major event types without treating telemetry as proof of custody.
   Exercises 3 and 7 test the mapping.
4. **Calculate** coverage, timeliness, duplication and expected operating value with explicit denominators.
   Exercises 4 and 5 provide numeric checks.
5. **Design** ingestion and exception handling for duplicates, corrections, late data, sensor gaps and conflicting sources.
   Exercises 6 and 7 ask for a design and a critique.
6. **Evaluate** a partner data-sharing proposal, including access, semantics and quality responsibilities.
   Exercise 8 asks for a short agreement.
7. **Explain** why a data-quality score or a standardized payload is insufficient evidence of an avoided loss.
   Exercise 5 and the interview questions test this distinction.

A successful answer identifies what is known, what is inferred, who can act and when the evidence expires.
It does not merely name a technology.

## 3. The core mental model

```text
Physical world:  pallet loaded ---- truck moves ---- temperature changes ---- delivery
                     |                 |                     |                   |
Observed by:      dock reader       vehicle unit          sensor             receiving scan
                     |                 |                     |                   |
Producer assertion: identifier + business step / timestamp / location / sensor report
                     |                 |                     |                   |
Contract & transport: authenticated source, agreed identifiers, event schema, delivery
                     |                 |                     |                   |
Event processing: validate -> preserve raw -> deduplicate -> link -> assess quality
                     |                          |                  |
                     |                          |                  +-> UNKNOWN / HOLD
                     |                          +-> corrected or late evidence
                     +-> accepted event -> shipment and product state with provenance
                                               |
Decision:  evidence + exposure + deadline + authority -> investigate / hold / release
                                               |
Feedback:  action, outcome, corrections and measured value -> improve rules and contracts
```

The physical world and a record about it are different layers.
An event is a source's assertion that something occurred or a measurement was made.
The control tower must retain enough provenance to explain who asserted it, what was measured and when the record became available.
A derived state such as “in transit” is an interpretation of several assertions under a documented rule.
It is not an event directly sensed by all parties.

An event stream is a sequence of delivered event records, often but not necessarily delivered continuously.
The ordering in which the system receives them can differ from the order in which the underlying occurrences happened.
The stream can contain retries, revisions and gaps.
For this lesson, an event is considered actionable only after it has been associated with the right shipment or product and checked against a decision-specific quality rule.

Four relationships matter.

1. **Identity:** which object, shipment, vehicle, party and location are involved?
2. **Time:** when did the occurrence happen, when was it captured, and when did the tower learn of it?
3. **Meaning:** is this a physical observation, a business milestone, a sensor reading, or an interpretation?
4. **Authority:** who is allowed to use it and to act on its consequence?

Think of every visibility claim as a sentence with an evidence label.
“Carrier C reports truck V at site L at 09:42 UTC” is narrower than “shipment S was delivered at 09:42 UTC.”
The second sentence requires a supported association between V and S, plus evidence that delivery actually occurred.
A reliable system can explain that chain.

Data quality is relative to a decision.
A reading received 50 minutes after measurement may be adequate for a weekly scorecard but useless for preventing a 30-minute excursion.
A precise location may be unnecessary for proof of receipt but essential to confirm that a vehicle is heading toward the correct site.
A standard schema improves a shared interpretation only if the partners also agree on identifiers, business processes, event definitions and access rights.

No single record is entitled to erase all contradictory evidence.
For example, a sensor report may be legitimate while a seal scan shows the sensor was not associated with the pallet in question.
A tower should retain both observations, mark the association uncertain and route the case to its owner.
Absence of a reading is not evidence that the reading would have been normal.

## 4. Definitions and vocabulary

| Term | Definition and unit or coding system | Common misuse |
|---|---|---|
| Physical occurrence | An actual movement, handoff, condition or other real-world change. | Treating a database update as proof the physical occurrence happened. |
| Observation | What a person or device perceives about an occurrence or condition. | Assuming every observation is complete and unbiased. |
| Event record | A structured assertion describing an occurrence or observation and its context. | Calling an event record the physical object itself. |
| Event stream | A sequence of delivered event records; delivery need not be continuous or ordered. | Assuming a stream guarantees completeness or exactly-once business effects. |
| Source system | The application or device that produces an assertion. | Treating the integrating broker as the physical observer. |
| Producer | A party or component publishing an event. | Assuming the producer owns the goods. |
| Consumer | A party or component receiving and interpreting events. | Assuming the consumer may redistribute them freely. |
| Internet of Things (IoT) | Networked physical devices that measure or interact with their environment. | Assuming connectivity means continuous or accurate measurement. |
| Telematics | Vehicle or asset data communicated remotely, such as position, motion or status. | Treating the vehicle's coordinates as direct proof of each pallet's location. |
| GNSS | Satellite-based positioning; coordinates are commonly latitude and longitude in degrees. | Confusing position precision with certainty of shipment identity. |
| Latitude and longitude | Coordinates describing a geographic position, conventionally in decimal degrees. | Assuming coordinates identify a legal site or loading bay without mapping. |
| Geofence | A defined geographic area used to detect entry or exit. | Treating a geofence crossing as proof of unloading or acceptance. |
| Sensor | Device that measures a property, such as temperature in degrees Celsius (°C). | Assuming every sensor measures product core temperature. |
| Sampling interval | Time between intended sensor measurements, in minutes or seconds. | Confusing reporting frequency with measurement frequency. |
| Calibration | Comparison and, where appropriate, adjustment against a reference under documented conditions. | Assuming a calibrated device is always installed correctly. |
| Excursion | A measurement or derived interval outside an agreed range. | Assuming one air-temperature reading proves product is unsafe. |
| EPCIS | GS1 standard for sharing supply-chain visibility event data. | Equating an EPCIS message with a verified physical fact. |
| Core Business Vocabulary (CBV) | GS1 vocabulary for interoperable business-step, disposition and other values used with EPCIS. | Inventing local codes and assuming they have standard meaning. |
| Event type | EPCIS structure describing a class of event, including ObjectEvent and AggregationEvent. | Confusing event type with the business step. |
| ObjectEvent | EPCIS event concerning one or more objects and their business context. | Assuming it always implies transfer of ownership. |
| AggregationEvent | EPCIS event describing a parent–child association or disassociation. | Treating an old association as indefinitely valid. |
| TransactionEvent | EPCIS event associating or disassociating objects with business transactions. | Treating a purchase-order association as a physical scan. |
| TransformationEvent | EPCIS event linking input objects to output objects in a transformation. | Assuming every repack is only a location change. |
| AssociationEvent | EPCIS event expressing a persistent association between objects, distinct from ordinary aggregation. | Using the term as a generic synonym for any database join. |
| Action | EPCIS field expressing how an event affects an association or object context where applicable: ADD, OBSERVE or DELETE. | Interpreting DELETE as deleting the historical event record. |
| Business step | The process context, often expressed with a CBV business-step identifier. | Assuming the same label has identical local implementation everywhere. |
| Disposition | Business condition or status of objects, often expressed with a CBV identifier. | Treating it as an independent physical measurement. |
| Read point | Specific place at which an event was observed, such as a dock door. | Treating it as the future destination. |
| Business location | Business site where objects are located following the event. | Assuming it always has the precision of a reader or GPS coordinate. |
| Event time | Time at which the event is asserted to have occurred, including relevant offset context. | Using arrival time as a substitute without saying so. |
| Record time | EPCIS repository time at which an event was recorded. | Assuming it is necessarily the device measurement time. |
| Ingestion time | Local tower time at which the receiving pipeline accepts or records a message. | Treating it as the event's occurrence time. |
| Latency | Difference between two specified timestamps, expressed here in minutes. | Reporting a latency figure without saying which timestamps it compares. |
| Time zone offset | Local offset from Coordinated Universal Time (UTC), such as +02:00. | Treating the offset as a complete time-zone rule for future dates. |
| Event identifier | Identifier for an individual event assertion, if present. | Using the object identifier as though it identified each occurrence. |
| Object identifier | Identifier for a particular physical or digital object, such as a serialized trade item. | Confusing a product class with an individual instance. |
| Global Trade Item Number (GTIN) | GS1 identifier for a trade item class, not necessarily an individual instance. | Assuming a GTIN alone identifies one serialized package. |
| Serial Shipping Container Code (SSCC) | GS1 identifier for a logistic unit such as a pallet. | Assuming a pallet SSCC directly identifies its current vehicle. |
| Global Location Number (GLN) | GS1 identifier for a party or location. | Assuming two partners' local location codes are interchangeable. |
| Master data | Relatively stable reference data used to interpret identifiers and attributes. | Assuming it never changes or that every partner has the same version. |
| Digital link or URI | A Uniform Resource Identifier used to identify or refer to an entity or vocabulary term. | Assuming dereferenceability or public access is required for all uses. |
| Business transaction reference | Reference to an order, shipment or other transaction associated with an event. | Treating a reference alone as proof of shipment contents. |
| Lineage or provenance | Evidence of an assertion's origin, transformations and supporting relationships. | Keeping only the latest display value and losing its source. |
| Duplicate | Repeated delivery of the same logical assertion. | Collapsing two legitimate observations merely because their fields are similar. |
| Idempotency | Property that repeated processing of the same input does not repeat its business effect. | Assuming the transport itself prevents all duplicates. |
| Correction | An additional assertion that identifies or supersedes erroneous prior information under an agreed process. | Silently overwriting history without an audit trail. |
| Error declaration | EPCIS mechanism for indicating an event is in error, with reason and, where relevant, corrective event references. | Assuming a correction means the original event was never received. |
| Watermark | A stream-processing estimate of event-time progress used to manage lateness. | Treating it as a guarantee that no later record will arrive. |
| Coverage | Fraction of an explicitly defined expected population with qualifying evidence. | Dividing received readings by themselves instead of by expected readings. |
| Completeness | Presence of required fields or expected events for the stated use. | Treating field completeness as proof of physical completeness. |
| Timeliness | Arrival within an explicitly stated decision window. | Using an average that hides late high-impact records. |
| Accuracy | Agreement with a suitable reference, under a defined measurement method. | Inferring truth from internal consistency alone. |
| Consistency | Agreement among fields, sources or rules where agreement is expected. | Assuming consistency proves all sources are independently correct. |
| Exception | A condition requiring investigation or action under an agreed operational rule. | Assuming every unusual value deserves the same priority. |
| Decision right | Authority to make a specific operational or commercial choice. | Assuming dashboard access confers authority to release held goods. |
| Service-level agreement (SLA) | An agreed service target, such as a percentage of messages received within a stated time. | Assuming an SLA supplies an independent measurement of compliance. |
| Personally identifiable information (PII) | Information that identifies or can identify a person, depending on the applicable definition. | Assuming all vehicle location data is automatically anonymous. |

## 5. The mechanics, step by step

### Step 1 — Start with an intervention, not a feed

**Input:** a customer promise, a product handling rule and a list of possible operational actions.
**Output:** one decision statement with an owner, deadline and evidence threshold.

Describe the choice in an action verb.
For example: “The quality desk decides whether to place pallet P on hold before receiving accepts it.”
The decision is not “show temperatures on a map.”
The latter is a display feature and says nothing about who will act.

Specify the object at risk and the period during which an intervention remains possible.
If the truck reaches the receiver at 12:00 UTC, a warning received at 12:45 UTC cannot support a pre-arrival hold.
It may still support investigation, recall or an after-action review.
Do not count that late warning as an avoided pre-arrival loss.

Distinguish the following terms in the operating plan:

- **Forecast:** the predicted arrival or predicted temperature outcome, with uncertainty where available.
- **Plan:** the intended truck route, appointment and monitoring schedule.
- **Target:** the desired outcome, such as 95% timely coverage for eligible trips.
- **Commitment:** the agreed delivery or handling obligation, which may be conditional and contract-specific.

A plan is not a record of what happened.
A target is not proof of actual performance.
An ETA forecast is not a carrier's legally binding delivery commitment unless the agreement explicitly makes it one.

Choose the cost of a wrong intervention and the cost of a missed event separately.
A false hold creates labor, delay and potentially product deterioration.
A missed excursion may lead to acceptance of product requiring review.
If outcomes are asymmetric, the rule for a warning need not be the same as the rule for release.
The party with quality authority sets the release rule; the analytics team does not infer it from a graph.

### Step 2 — Define the population and decision grain

**Input:** the intervention and applicable handling policy.
**Output:** an eligibility rule, unit of analysis and expected observation schedule.

State whether the monitored unit is a trip, vehicle, container, pallet, case, individual item or sensor.
A trip can contain multiple pallets with different temperature requirements.
A pallet can move between trucks or be split into cases.
A sensor can measure the air near a door, not the core of every product on the vehicle.

For a shipment-level intervention, define how goods are associated with the trip and when that association begins and ends.
An event linked only to a vehicle does not automatically apply to all its historical cargo.
A valid association needs an effective time interval and a source.
When the vehicle changes, close the old association and establish the new one.
A late reassignment must be able to revise a prior derived state without destroying its history.

Construct an expected population independently of the received stream.
An eligible trip manifest can contain trip ID, planned departure, scheduled arrival, required monitoring profile, assigned sensor and responsible party.
If the manifest is incomplete, report that limitation as well as the resulting coverage.
Using only trips that already transmitted messages as the denominator hides entirely silent trips.

For periodic sensors, define the expected schedule from the actual monitoring start and end where those times are known.
Specify how to handle partial intervals, maintenance windows, reassignment and trip cancellation.
A denominator based on a planned 6-hour trip will differ from one based on the actual 7-hour trip.
Neither is intrinsically right; label the policy and keep the two quantities distinct.

### Step 3 — Inventory producers and their claims

**Input:** carrier, warehouse, device and partner process maps.
**Output:** a source inventory with ownership and expected evidence.

For each producer record the accountable organization, technical endpoint, physical observer, event definitions and operating hours.
The technical sender may be a carrier's aggregator while the original observation came from a driver's mobile device.
Preserve both where available.

| Source | Direct observation or assertion | Useful decision | Important limitation |
|---|---|---|---|
| Dock reader | A code was read at a reader at a time. | Check loading sequence. | An unread code is not proof of absence. |
| Warehouse management system | A user or process confirmed a business step. | Associate goods with a dispatch. | Confirmation can be delayed or entered manually. |
| Vehicle telematics | A device reported vehicle position or status. | Anticipate approach or route deviation. | A vehicle position does not establish its cargo. |
| Temperature logger | A probe measured a condition at a point. | Trigger investigation of an excursion. | Probe placement and calibration determine interpretation. |
| Carrier milestone feed | Carrier asserted pickup, departure or delivery. | Coordinate appointment and customer communication. | The milestone may be inferred or manually entered. |
| Receiver | Receiver confirmed quantity or condition at receipt. | Close or dispute the shipment. | Receipt may be posted after the physical handoff. |

Ask each source what it does during a network outage.
A device might buffer readings, drop them, or change sampling intervals to conserve battery.
A vehicle message may be generated on movement, at fixed intervals, or only when polled.
A supplier may publish shipment events in batches once per day.
These behaviors determine whether a feed can serve a time-critical decision.

### Step 4 — Agree on identity and associations

**Input:** object lists, master data and identifiers from all parties.
**Output:** a time-bounded crosswalk and evidence for each important relationship.

Identify the shipment reference used commercially, the logistic-unit identifier printed on each pallet, the vehicle identifier, the sensor identifier and the destination identifier.
Do not put them into one generic `id` field.
The identifier for the delivery order, the SSCC on a pallet and the identifier for a vehicle have different scopes and lifetimes.

A useful association record contains source identifier, target identifier, relationship type, validity start, validity end, producer, evidence event and confidence or verification status where applicable.
For instance, a warehouse loading scan can associate pallet P with trailer T from 08:12 UTC until an unloading scan or a superseding transfer.
A telematics message for T at 09:30 UTC is then indirect evidence about P only while the association remains valid.
If the seal breaks or the trailer is exchanged, the tower must not blindly extend the relationship.

Partner identifiers may be reused or recycled.
A carrier trip number could be unique only within that carrier and a calendar period.
Namespace it with the issuer and, where needed, effective dates.
A crosswalk should resolve a code to a site or party version that was valid at event time, not simply the newest master-data row.

Serial-level identifiers can support traceability, but serial detail may be unnecessary for every decision.
Use the lowest granularity that meets the business and regulatory need.
Higher granularity increases data volume, access risk and the cost of correcting association errors.

### Step 5 — Define the event contract

**Input:** decisions, source inventory, identifiers and partner capabilities.
**Output:** a versioned contract that a producer and consumer can test.

An event contract needs more than a sample payload.
Write down the business meaning of each milestone, mandatory identifiers, allowed values, units, time semantics and expected delivery window.
Name the party responsible for initial data, corrections and operational support.
State whether a missing value means unknown, not applicable or not yet captured.

A minimal contract for a temperature report might require:

1. A producer namespace and stable event identifier, or a documented fallback deduplication key.
2. A sensor identifier and, when known, a valid shipment, pallet or vehicle association.
3. Measurement time with an explicit offset or UTC representation.
4. Temperature value in °C, measurement type and sensor placement or profile.
5. A report creation or reception time where available.
6. A rule for corrections, late delivery and replay.
7. Data retention, access restrictions and escalation contacts.

Define how a recipient learns that a sensor has stopped reporting.
A status heartbeat, an expected schedule and a source outage notice each convey different evidence.
If none exists, the tower can detect that an expected message did not arrive but cannot safely infer why.

Document semantic versions of local extensions and mapping tables.
A syntactically valid field can still have an incompatible meaning after a source changes its process.
For example, a carrier may redefine “arrived” from a driver button press to geofence entry.
The names remain the same while the evidence and false-positive behavior change.

### Step 6 — Choose the relevant EPCIS representation

**Input:** the physical and business relationship to communicate.
**Output:** an appropriate EPCIS event type and vocabulary mapping.

EPCIS expresses visibility events using dimensions often summarized as what, when, where, why and how.
“What” identifies objects or quantities.
“When” covers event time and offset.
“Where” can include read point and business location.
“Why” gives business context such as business step and disposition.
“How” can carry sensor-related information.
These dimensions help recipients interpret an event, but they are not a guarantee that every field is populated.

Use an ObjectEvent for an observation or business event concerning identified objects when that is the right semantic fit.
For example, a warehouse may report a shipping event for a pallet at a dock.
The business step identifies the process context; the event type does not by itself say whether the action was shipping, receiving or another step.

Use an AggregationEvent to describe association or disassociation of child objects and a parent, such as cases packed on a pallet.
A relationship established at packing is not proof that it still held after a later break or repack.
Capture the changes that terminate or alter the relationship.

Use a TransactionEvent where the purpose is to associate or disassociate objects and business transactions.
A transaction reference may help connect the physical event to an order, but it does not prove physical loading by itself.
Use a TransformationEvent when inputs are transformed into outputs, for example when ingredients become a new batch or cases are repacked under a new identifier.
A transformation requires appropriate traceability links; a simple association is not a substitute for product genealogy.

AssociationEvent supports a more persistent association where the standard's semantics fit, such as a component associated with an asset.
Choose between association types from the normative specification and the actual process, not from the English word “association.”
Use CBV values where applicable and explicitly govern any permitted extensions.
A private label with the same spelling as a standard term is not automatically equivalent.

EPCIS supports sensor elements that can report measured conditions, including temperature, with metadata and standardized identifiers.
The presence of a sensor report does not validate the sensor, its placement or its relationship to the product.
Agree whether the report is an individual reading, summary, interval or other representation before calculating exposure.

### Step 7 — Acquire and transport evidence safely

**Input:** event contract and partner connection details.
**Output:** authenticated event delivery with traceable receipt and recoverable failure handling.

A feed may arrive through an EPCIS interface, a message broker, an application programming interface (API), a file exchange or another contracted channel.
The transport is distinct from the event's semantic content.
An EPCIS-shaped event does not require a particular broker, and a broker does not itself supply EPCIS semantics.

Authenticate the sender, authorize the feed and encrypt data in transit according to the organizations' security policy.
Protect credentials and rotate them on an agreed schedule.
Separate the ability to submit events from the ability to query another party's history.
Record receipt time and the source endpoint without treating either as proof that the physical event occurred.

Acknowledge only at a clearly defined durability boundary.
A response saying “accepted” should not be represented to users as “validated and acted upon” unless that is actually what it means.
On transient failure, a producer may resend.
The consumer therefore needs idempotent processing even if transport delivery is normally reliable.

Preserve a raw copy or equivalent auditable representation with access controls and retention appropriate to the contract.
Keep the parsed version, validation result and mapping version separately enough to reconstruct why a decision was made.
Do not expose raw location or personal data to every dashboard user merely because it is technically available.

### Step 8 — Validate without inventing certainty

**Input:** received event and contract version.
**Output:** accepted, quarantined or rejected event with a reason.

Validate in layers:

1. **Transport:** authenticated sender, intact message, permitted endpoint and size.
2. **Syntax:** parseable data and required structural fields.
3. **Semantics:** allowed event type, field combinations, identifiers and units.
4. **Reference data:** known partner, object, sensor or location under the applicable version.
5. **Plausibility:** reasonable timestamps and values under documented thresholds.
6. **Business context:** valid association to the decision grain and appropriate process state.

A plausible value is not necessarily accurate.
A temperature of 4.0 °C is syntactically and physically plausible but may be a stale copy or come from the wrong sensor.
Similarly, a valid GLN may identify a different facility from the one in the purchase order.
Validation should identify the nature of the uncertainty instead of silently correcting the value.

Choose a disposition for each validation failure.
A malformed event may be rejected to the producer.
An otherwise valid event with an unknown crosswalk may be quarantined and routed to master-data support.
A late but valid event may be accepted into history while marked too late for a specific decision.
The rule should be observable and should not discard the original evidence.

Avoid silently imputing a healthy temperature during a sensor gap.
If an algorithm estimates a missing value for analysis, retain the estimate separately from measured readings, label the method and uncertainty, and do not use it as direct evidence of compliance unless the governing procedure permits it.

### Step 9 — Process time, retries and corrections

**Input:** validated events, timestamp fields and correction policy.
**Output:** reproducible event history and decision state as known at each point in time.

Record at least the asserted occurrence or measurement time and the tower's receipt time.
Where supplied and relevant, retain producer creation time and EPCIS repository record time separately.
Convert timestamps to a common comparison basis while preserving the original offset.
A naive timestamp with no known offset should not be silently treated as UTC.

Calculate delivery latency as tower receipt time minus event time when those clocks are comparable.
A negative result can indicate clock skew, offset errors or incorrect semantics; it is not negative physical travel time.
Measure latency distributions by source and event type as well as by overall population.

Use an event identifier plus producer namespace when the contract promises uniqueness.
If there is no stable event identifier, a deterministic composite key may be necessary, but it can merge distinct events or fail to merge retries.
State its fields, scope and collision risk.
Never use a pallet ID alone as a deduplication key: the pallet legitimately generates many different events.

A retry should not generate a second business action.
A genuinely new reading at the same temperature should remain a separate event if it has a distinct measurement identity or time.
A correction should reference the earlier assertion or follow a documented reconciliation process.
EPCIS error declarations can mark an event as erroneous and reference corrective events where applicable.
Retain the erroneous record and its declaration in the audit trail rather than rewriting the past invisibly.

Out-of-order delivery can temporarily produce an impossible sequence such as receipt before shipping.
The receiving timestamp alone is not a reason to reject a valid earlier shipping event.
Recompute the derived sequence under an explicit lateness policy and retain the original decision-time view.
A watermark can govern when a provisional aggregation closes, but later evidence may still arrive and require a correction.

### Step 10 — Connect telemetry to business context

**Input:** normalized telemetry, current and historical associations, and handling rules.
**Output:** decision-ready evidence with uncertainty and provenance.

A GPS point needs an asset identity and an association to a shipment before it can inform the shipment's progress.
A geofence crossing needs a polygon version, timestamp, position uncertainty and a rule for dwell or repeated boundary crossings.
Arrival at a site does not mean the trailer is at the dock, unloaded, counted or accepted.

A temperature reading needs a unit, sensor identity, time, measurement location and product applicability.
If the sensor measures return-air temperature, do not label it “product core temperature.”
If multiple products have different allowable ranges, apply each relevant product's rule to the appropriate sensor association.
For any medical, pharmaceutical or food product, the applicable handling and disposition rule must come from the responsible quality organization and applicable requirements, not from this illustrative lesson.

An isolated out-of-range sample establishes that the measured point was outside a threshold at the sampled time, subject to measurement uncertainty.
It does not establish the exact start, end or duration of the excursion between samples.
A continuous duration can be bounded or estimated only with additional assumptions about behavior between measurements.
Escalate uncertainty instead of presenting interpolated duration as an observed fact.

### Step 11 — Measure quality against the expected population

**Input:** eligible trips, expected observations, received events and quality rules.
**Output:** a scorecard with denominators, exclusions and decision-specific thresholds.

Define metrics before looking at the result.
The following are illustrative, not universal service levels:

| Metric | Numerator | Denominator | Interpretation |
|---|---|---|---|
| Trip coverage | Eligible trips with at least one qualifying reading. | All eligible trips. | Whether any evidence arrived, not whether monitoring was continuous. |
| Slot coverage | Expected slots matched by a qualifying unique reading. | Expected slots. | Coverage of the agreed measurement schedule. |
| Timely slot coverage | Expected slots with qualifying reading received by the allowed deadline. | Expected slots. | Availability in time for a stated decision. |
| Timeliness among received | Qualifying unique readings received within the allowed latency. | All qualifying unique readings received. | Feed speed conditional on receiving a reading. |
| Duplicate delivery rate | Duplicate message deliveries under a documented key. | All message deliveries. | Transport or producer retry burden, not necessarily business error. |
| Association completeness | Qualifying readings linked to an eligible trip by a valid association. | Qualifying readings for which a trip link is expected. | Whether the measurement can be applied to the operating population. |
| Correction rate | Original events later declared erroneous or corrected. | Original events in a mature cohort. | Frequency of correction; a short follow-up can undercount it. |

Define the matching window and whether multiple readings can satisfy one expected slot.
A slot may have one credited reading, while excess readings remain in the history.
Report data by carrier, lane, device type and decision-critical period where population sizes support interpretation.
Do not hide a silent carrier behind strong coverage from a larger partner.

For auditability, retain the metric version, eligibility list, time-zone treatment and source-data cutoff.
A report produced the next day may include late readings that were not available to the operator during the trip.
Show both the as-known-at-decision measure and the later reconciled measure.

### Step 12 — Turn quality results into a bounded workflow

**Input:** decision-ready events, quality score and action policy.
**Output:** an exception with a named owner, due time and disposition.

Use at least three statuses: evidence supports the rule, evidence indicates a potential exception, and insufficient evidence.
An unknown state must not be shown as green solely because no alarm was received.
The operational response to unknown may differ from the response to a measured excursion.

An alert should identify the object, evidence, event time, received time, relevant threshold, association, uncertainty and recommended next check.
It should also name who owns the next action and the deadline.
A quality hold is not the same as a final rejection, and a prediction of delay is not a confirmed missed appointment.

The workflow must record acknowledgement, investigation, decision, disposition and closure reason.
If new evidence arrives, link it to the existing case or reopen it under a defined rule.
Avoid opening a fresh case for each retry of the same event.
For persistent sensor excursions, define whether several readings form one incident and when it ends.

### Step 13 — Evaluate operating value and improve the contract

**Input:** action logs, outcomes, cost assumptions and comparison design.
**Output:** an estimated value range and prioritized fixes.

Start with avoided losses that an action could plausibly influence before its deadline.
Estimate eligible volume, baseline event frequency, timely detection, effective response and unit value separately.
Then subtract the recurring feed cost, monitoring cost and incremental investigation or intervention costs.
Do not count the full value of every alerted shipment as a saved loss.

Separate a prospective business case from measured benefit.
A pilot's observed correlation between alerts and fewer claims may reflect different routes, seasons or customer mix.
A comparison group, staggered rollout or other suitable design can strengthen causal interpretation if feasible and ethical.
Use actual action records to check whether alerts changed decisions rather than merely showing information already known.

Hold regular partner reviews using examples of failed associations, late messages and disputed milestones.
Agree on corrective owners and deadlines rather than blaming the “data” as an abstract system.
Revisit the contract when a device, process, threshold or commercial obligation changes.
A quality score that rises because the denominator was narrowed is not necessarily an operational improvement.

### Step 14 — Test interoperability with a partner

**Input:** the signed event contract, representative operational records and access to a partner test environment.
**Output:** a documented acceptance decision and a prioritized defect list.

An interoperability test is not complete when a message passes a schema validator.
The receiver must be able to connect the source's assertion to the correct business decision.
Run a small set of scenarios with known expected outcomes, and include both the partner's operational owner and the technical integrator.
The following scenarios are useful because each checks a different boundary.

**Scenario A: normal shipping.**
The warehouse scans one identified pallet and associates it with a dispatch.
The partner sends an event containing the pallet identifier, shipping business context, source location and event time.
The tower resolves the location and links the pallet to the correct eligible shipment.
The expected result is one auditable shipping assertion, not an inferred proof of delivery.
Record the source message, contract version, resolved identities and receipt time.
If the identifiers resolve but the commercial order does not, classify that as a relationship defect rather than a syntax defect.

**Scenario B: split and repack.**
The warehouse removes one case from a pallet and places it on a different pallet before departure.
The event sequence must end the relevant old association and create the new one according to the chosen business process.
Query the associations at a time before the split and again after it.
The expected result is different membership at the two times without loss of the historical relationship.
If a case appears on both pallets at the same time, investigate whether this is a real operational overlap, a missing event or a mapping error.
Do not automatically delete one relationship solely to make the screen look consistent.

**Scenario C: replay after an outage.**
Disconnect the consumer in the test environment while the producer generates three distinct measurements.
On reconnection, deliver the three measurements and retry one of them with the same event ID.
The expected result is four delivery records, three logical measurements and no repeated business effect for the retry.
Compare the operator's as-known state during the outage with the reconciled history after replay.
The latter can be more complete but must not be presented as though it was available during the outage.
If the partner generates a new event ID for the retry, record that as a contract issue and agree how to identify a repeat.

**Scenario D: wrong unit or unknown code.**
Send a test temperature with an unsupported unit and a location code not present in the agreed reference data.
The expected result is a visible validation failure or quarantine under the contract.
The system must not treat an unknown unit as Celsius merely because Celsius is common on the lane.
The support team should be able to identify the producer, message, reason and responsible owner.
Once the partner corrects the data or mapping, retain the original disposition and the correction trail.

**Scenario E: conflict across sources.**
Have the carrier report arrival at 11:40 UTC and the warehouse report gate entry at 12:05 UTC for the same trailer.
The difference may reflect different definitions rather than a clock error.
Show both assertions with source and meaning, then apply any agreed source-priority rule only for the specific derived decision.
For example, use the warehouse gate event for gate-queue reporting if that is the agreed measurement.
Do not erase the carrier's earlier report, which could refer to arrival at an external staging area.
When the meanings are truly incompatible and cannot be resolved, preserve the conflict as an open exception.

**Scenario F: permission boundary.**
Give a receiver access to its own shipment and attempt to query an unrelated customer's vehicle history.
The expected result is access to the authorized shipment information and denial of the unrelated query.
Check exported reports and event replays as well as the primary dashboard.
A system that hides a field on the screen but exposes it through a query endpoint has not enforced the boundary.
Document the test identity, role, intended purpose and result without copying sensitive production data into the test report.

**Acceptance decision.**
Separate blocking defects from improvements.
A broken identifier association or unauthorized data exposure can prevent a pilot from being safe or interpretable.
A cosmetic dashboard label might be deferred if it does not misrepresent evidence and the owner agrees.
The business owner should accept the semantic result; the technical owner should accept transport, validation and access behavior.
Set a retest date and name the party responsible for each unresolved defect.

### Step 15 — Operate the feed across its lifecycle

**Input:** accepted partner feed, assigned support contacts and agreed service measures.
**Output:** an operating cadence that detects degradation and accommodates change.

During onboarding, compare a sample of events with source-system records and, where practical, the physical operation.
Choose the sample to include handoffs, missing scans and changes of vehicle rather than only successful journeys.
Record both confirmed matches and discrepancies.
Agreement among two systems copied from the same source is weaker evidence than comparison against an independent observation.

During daily operations, review missing feeds and unassigned high-priority exceptions.
Use a freshness indicator that says when the last observation was made and when it was received.
A dashboard that refreshes every minute can still display a reading measured hours earlier.
Escalate according to decision risk: an outage on an actively monitored perishable trip may need a faster response than a delay in a retrospective finance feed.

During weekly reviews, compare actual latency and coverage with the agreed target by source and event type.
For each breach, identify the affected population and the operational consequences, not only the system's uptime.
An endpoint can be available while the producer sends the wrong trailer ID.
Likewise, a short outage may have no material impact if the affected events were not needed for a live decision and replay is complete.
Both the service metric and the impact assessment belong in the review.

During periodic contract reviews, inspect changes in device firmware, event semantics, master data, partner ownership and commercial terms.
Test a proposed change against both old and new consumers before its effective date.
Keep a versioned mapping so historical records can still be interpreted using the definitions that applied at the time.
If an event field is retired, identify every downstream metric and decision that depends on it before removing it.

When a partner leaves or access expires, revoke credentials and stop new queries according to the agreement.
Retain or delete historical data as required by the applicable legal and contractual schedule.
Ensure an audit can still explain prior decisions using information the organization is entitled to retain.
Closing the integration is an operational process, not simply switching off an endpoint.

## 6. Worked example

### Situation and decisions

Northbank Foods is a fictional chilled-food distributor.
It contracts carrier Cedar to deliver a mixed load of refrigerated pallets to a regional distribution center.
The organizations have agreed to monitor trailer return-air temperature during each eligible trip.
The illustration is not a product safety rule and does not prescribe a legal limit.

For the example, the quality team sets a **review trigger** at a measured temperature above 8.0 °C.
A trigger requires an investigation and may lead to a hold; it does not automatically prove spoilage.
The quality desk, not the analytics service, owns the release decision.
The receiver owns the dock appointment, and the transport planner owns carrier contact.

On a particular Tuesday the planned departure is 08:00 UTC and the planned arrival is 12:00 UTC.
The customer delivery commitment is an appointment window from 11:45 to 12:15 UTC, subject to the contract.
At 10:00 UTC the arrival forecast is 11:55 UTC; it is a prediction, not the commitment.
The data-service target is to receive at least 95% of expected temperature slots within 15 minutes of measurement over the monthly eligible population.
That target is not the same as the customer's delivery commitment.

The monitoring plan calls for a reading every 30 minutes at the following eight scheduled slots:
08:00, 08:30, 09:00, 09:30, 10:00, 10:30, 11:00 and 11:30 UTC.
The planned denominator is eight slots for this example; the manifest confirms monitoring was required for the full interval.
A reading belongs to a slot only if the sensor ID matches, its measurement time is exactly that slot in this simplified example, and the sensor-to-trailer association is valid.
The receiver may later revise that population if the manifest is corrected; such a revision is not silently backdated.

The shipment contains pallet `P-17` and pallet `P-18`.
A warehouse aggregation event connects cases to the pallets; a loading scan associates both pallets with trailer `T-4` at 07:55 UTC.
A temperature sensor `S-9` is installed in the trailer return-air stream, not inside either pallet.
The sensor-to-trailer installation record is valid from 07:30 UTC until removal after delivery.
There is no direct measurement of the internal temperature of either pallet.

### Evidence received

The following table presents measurement and tower receipt times in UTC.
Event IDs are assigned by the sensor gateway and unique within its producer namespace.
An exact retry preserves the same event ID.

| Delivery | Event ID | Measured time | Received time | Return air, °C | Interpretation |
|---:|---|---|---|---:|---|
| 1 | E01 | 08:00 | 08:04 | 4.2 | On time. |
| 2 | E02 | 08:30 | 08:36 | 4.4 | On time. |
| 3 | E03 | 09:00 | 09:12 | 5.1 | On time. |
| 4 | E03 | 09:00 | 09:13 | 5.1 | Retry of delivery 3, not a new measurement. |
| 5 | E04 | 09:30 | 10:05 | 8.6 | Valid but 35 minutes late. |
| 6 | E05 | 10:00 | 10:08 | 8.9 | On time; arrived after E04 in this example. |
| 7 | E06 | 10:30 | 10:39 | 7.8 | On time. |
| 8 | E07 | 11:00 | 11:51 | 5.0 | Valid but 51 minutes late. |

There is no 11:30 reading by the analysis cutoff of 12:00 UTC.
The missing slot is unknown, not a normal temperature.
The seventh unique reading is available for historical review but is received too late to meet the 15-minute service target.
The 09:30 and 10:00 measurements exceed the example's 8.0 °C review threshold.
They are two samples, not proof that the temperature stayed above 8.0 °C continuously between them.

A carrier telematics event puts trailer `T-4` near the destination at 11:42 UTC.
A geofence entry at 11:44 UTC suggests arrival at the site boundary.
The receiver records unloading at 12:03 UTC and acceptance at 12:10 UTC, both entered at 12:14 UTC.
The geofence is not used as proof of acceptance.
A held pallet should not be released solely because a vehicle reached the site.

### Calculate coverage and latency

There are eight expected slots and seven unique measurements that match a slot.
Thus slot coverage by 12:00 UTC is:

`7 unique covered slots / 8 expected slots = 87.5%`.

The timely unique measurements are E01, E02, E03, E05 and E06.
E04 and E07 exceed the 15-minute threshold.
Timely slot coverage is therefore:

`5 timely slots / 8 expected slots = 62.5%`.

Timeliness among received unique readings is:

`5 timely readings / 7 unique readings = 71.4%`, rounded to one decimal place.

The two metrics answer different questions.
The first penalizes the missing slot; the second measures speed conditional on receipt.
Neither should be called simply “timeliness” without its denominator.

There are eight delivered messages, one of which is a duplicate of E03.
Duplicate delivery rate is:

`1 duplicate delivery / 8 delivered messages = 12.5%`.

This rate uses deliveries, not unique events, as its denominator.
The 09:30 measurement has latency `10:05 − 09:30 = 35 minutes`.
The 11:00 measurement has latency `11:51 − 11:00 = 51 minutes`.
The duplicate does not create an additional reading, additional slot or second alert.

### Act at the time evidence becomes available

At 10:05 UTC, the tower receives E04, showing 8.6 °C at 09:30 UTC.
The temperature is above the review trigger, but it is already 35 minutes old.
The quality desk is alerted with the original measurement time, receipt time, sensor placement and trailer association.
The planner contacts the carrier to check refrigeration equipment and seek a current reading.
The tower does not infer that the product is unsafe or that the excursion lasted exactly 35 minutes.

At 10:08 UTC, E05 arrives with a 10:00 UTC measurement of 8.9 °C.
It is another out-of-range observation, not a duplicate.
The quality desk continues the same incident rather than creating a separate case for every high sample.
At 10:39 UTC, E06 reports 7.8 °C at 10:30 UTC.
That sample is below the illustrative review threshold but does not reconstruct the intervening temperature curve.

At 11:30 UTC, the expected slot becomes due.
At 11:45 UTC, no qualifying reading has arrived within the 15-minute window.
The system flags a monitoring gap, separate from the existing temperature incident.
By 12:00 UTC the 11:30 slot remains missing, so the gap remains unknown.
The quality desk decides whether the available evidence and applicable product procedure require a hold; the example does not prescribe a release.

### Estimate potential operating value

Suppose the distributor considers extending this service to 2,400 eligible trips per year.
The following are **planning assumptions**, not findings about Northbank or an industry benchmark:

- Baseline: 4% of eligible trips have a consequential temperature incident, or `2,400 × 0.04 = 96` incidents per year.
- Without the proposed feed, an effective intervention already occurs on 25% of those incidents.
- With the proposed feed and operating workflow, an effective intervention would occur on 55% of those incidents.
- The estimated avoidable loss per additional effective intervention is USD 1,800.
- Annual incremental feed, support and investigation cost totals USD 34,000.

The incremental effective-intervention rate is `55% − 25% = 30 percentage points`.
Expected additional effective interventions are `96 × 0.30 = 28.8 per year`.
The fractional figure is an expectation across years, not a claim that part of one trip is saved.
Gross expected benefit is `28.8 × USD 1,800 = USD 51,840 per year`.
Estimated net value is `USD 51,840 − USD 34,000 = USD 17,840 per year`.

The calculation assumes incident frequency, effectiveness and avoidable loss remain stable and that the cost already includes all incremental investigations.
It also assumes the 55% effectiveness estimate is incremental to the existing process and does not double-count another initiative.
No data in the single trip proves any of these assumptions.
If the uplift is only 15 percentage points, net value becomes `(96 × 0.15 × USD 1,800) − USD 34,000 = −USD 8,080 per year`.
This sensitivity makes the operating response, not just connection to the feed, a decisive uncertainty.

A controlled pilot should measure eligible trips, incidents, data available before each action deadline, actual intervention, outcomes and cost.
Where feasible, compare similar eligible trips with and without the new workflow while accounting for route, product and seasonal differences.
Do not treat raw alert counts as avoided losses.

## 7. Code, query, or calculation

The following Python 3 example reproduces the monitoring calculations above using only the standard library.
Copy it into a file and run `python event_quality.py`.
It uses an explicit expected schedule and a snapshot cutoff, deduplicates retries by producer and event ID, and preserves late records.
It deliberately does not claim to validate EPCIS, implement a streaming system, infer a continuous temperature curve or decide product disposition.

Assumptions for this bounded demonstration:

- All times in the input are timezone-aware UTC timestamps.
- The event ID is stable for retries and unique within the producer namespace.
- Identical keys represent the same measurement; the earliest receipt is credited.
- A measurement matches one expected slot only when its timestamp equals that slot exactly.
- All listed readings belong to the correct sensor and valid trailer association.
- Only deliveries received by 12:00 UTC are included in the snapshot.
- The 15-minute threshold is inclusive.
- An out-of-range sample exceeds, rather than equals, 8.0 °C.

```python
from datetime import datetime, timedelta, timezone

UTC = timezone.utc


def at(hour, minute):
    return datetime(2026, 5, 12, hour, minute, tzinfo=UTC)


expected = [
    at(8, 0), at(8, 30), at(9, 0), at(9, 30),
    at(10, 0), at(10, 30), at(11, 0), at(11, 30),
]
cutoff = at(12, 0)
max_latency = timedelta(minutes=15)
review_threshold_c = 8.0

# producer, event ID, measurement time, receipt time, temperature °C
deliveries = [
    ("gateway", "E01", at(8, 0), at(8, 4), 4.2),
    ("gateway", "E02", at(8, 30), at(8, 36), 4.4),
    ("gateway", "E03", at(9, 0), at(9, 12), 5.1),
    ("gateway", "E03", at(9, 0), at(9, 13), 5.1),
    ("gateway", "E04", at(9, 30), at(10, 5), 8.6),
    ("gateway", "E05", at(10, 0), at(10, 8), 8.9),
    ("gateway", "E06", at(10, 30), at(10, 39), 7.8),
    ("gateway", "E07", at(11, 0), at(11, 51), 5.0),
]

snapshot = [row for row in deliveries if row[3] <= cutoff]
unique = {}
for row in sorted(snapshot, key=lambda item: item[3]):
    producer, event_id, measured, received, temp_c = row
    key = (producer, event_id)
    if key in unique:
        prior = unique[key]
        if (measured, temp_c) != (prior[2], prior[4]):
            raise ValueError(f"Conflicting retry for {key}")
    else:
        unique[key] = row

duplicate_count = len(snapshot) - len(unique)
expected_set = set(expected)
matched = [row for row in unique.values() if row[2] in expected_set]
covered_slots = {row[2] for row in matched}
timely_slots = {
    row[2]
    for row in matched
    if timedelta(0) <= row[3] - row[2] <= max_latency
}
timely_readings = [
    row for row in matched
    if timedelta(0) <= row[3] - row[2] <= max_latency
]
review_ids = sorted(row[1] for row in matched if row[4] > review_threshold_c)


def pct(numerator, denominator):
    return 100.0 * numerator / denominator if denominator else 0.0


print(f"Expected slots: {len(expected)}")
print(f"Unique matched readings: {len(matched)}")
print(f"Slot coverage: {pct(len(covered_slots), len(expected)):.1f}%")
print(f"Timely slot coverage: {pct(len(timely_slots), len(expected)):.1f}%")
print(f"Timeliness among received: {pct(len(timely_readings), len(matched)):.1f}%")
print(f"Duplicate deliveries: {duplicate_count}/{len(snapshot)} "
      f"({pct(duplicate_count, len(snapshot)):.1f}%)")
print("Review-trigger event IDs:", ", ".join(review_ids))
```

Expected output:

```text
Expected slots: 8
Unique matched readings: 7
Slot coverage: 87.5%
Timely slot coverage: 62.5%
Timeliness among received: 71.4%
Duplicate deliveries: 1/8 (12.5%)
Review-trigger event IDs: E04, E05
```

The program accepts different event IDs at the same scheduled time as separate measurements, while counting that slot only once for slot coverage.
Its conditional timeliness denominator counts matched unique readings, not distinct slots.
The example happens to have at most one unique reading per slot.
Production implementations must define matching tolerances, competing readings, association changes, device clock errors, units and missing-data behavior explicitly.
A real quality system should report an undefined denominator as unavailable rather than the convenience function's `0.0%`; this dataset has nonzero denominators.

## 8. Data, standards, and regulatory context

### EPCIS 2.0 and the business agreement

GS1 publishes EPCIS 2.0 as a visibility-data standard for sharing information about the movement and status of objects.
The standard defines event structures, interfaces and other interoperability mechanisms.
The GS1 EPCIS and CBV resources in section 16 are the authority for exact field requirements and semantics.
Implementers should consult the applicable release and conformance materials rather than treating this lesson's summary as a schema.

The standard supports exchange, but a partner contract must still answer questions such as:

- Which business steps are actually observed and published?
- Which objects and granularity are in scope?
- Which party assigns identifiers, and when can they be reused?
- How are sensor placement, units, calibration and validity communicated?
- What delivery latency is operationally useful?
- How are corrections, replays and conflicting sources handled?
- Who can query, retain, redistribute or delete which data under the applicable rules?
- Who receives a quality exception and has authority to resolve it?

A schema-valid event can be commercially unusable if the receiver cannot map the object to a relevant order.
A standard vocabulary can describe “shipping” while partners disagree about whether the event means a dock scan, gate departure or a batch confirmation.
Specify the local physical trigger and cite the intended vocabulary term.
Do not assume a standards label alone resolves the discrepancy.

### Identifiers and master data

GS1 identifiers such as GTINs, SSCCs and GLNs have different roles.
A GTIN identifies a trade item type; a serial can distinguish an individual instance where used.
An SSCC identifies a logistic unit.
A GLN can identify a location or party in relevant applications.
Follow GS1 allocation rules and partner agreements rather than fabricating production identifiers.

Identifier syntax is not a substitute for a reliable ownership and lifecycle process.
A valid-looking logistic-unit code can still be attached to the wrong pallet.
A location code can identify a corporate entity when the operation needs a particular receiving door.
Master-data correction must preserve historical meaning at the time an event was made.

### Sensor data and evidence boundaries

A sensor report should identify the property, unit, time, device and relevant context.
A temperature measurement in °C is not interchangeable with a different unit unless the conversion is explicit and tested.
For example, 46.4 °F converts to `(46.4 − 32) × 5/9 = 8.0 °C`.
Rounding before comparing against a threshold can change whether a borderline value triggers review; document precision and comparison rules.

Sampling frequency, clock accuracy, sensor placement, calibration and buffering affect interpretability.
A missing interval is a data-quality issue and may be a product-quality issue depending on the governing procedure.
EPCIS can convey sensor information; it does not prescribe universal safe temperatures, sampling intervals, calibration procedures or disposition rules.
Those come from the product, process, qualified quality function and applicable law.

### Food traceability and regulatory scope

The U.S. Food and Drug Administration (FDA) Food Traceability Rule under section 204 of the Food Safety Modernization Act (FSMA) establishes additional traceability recordkeeping requirements for certain foods and persons, subject to the rule's scope and exemptions.
FDA materials identify critical tracking events and key data elements; consult the current FDA page and applicable text for the entities, records, dates and enforcement position relevant to a specific operation.
This lesson, dated 2026.5, does not assert that all food shipments are covered or that EPCIS use alone satisfies the rule.
A voluntary technical standard and a binding legal requirement are different things.
Legal and quality teams must determine applicability and current obligations.

Where the U.S. FDA Drug Supply Chain Security Act (DSCSA) applies, product tracing requirements are sector-specific and separate from the food rule.
Do not generalize an interpretation for food to pharmaceuticals.
Similarly, EU and other jurisdictions may impose different traceability, privacy and product-handling obligations.
Consult current primary sources and professional advice for each jurisdiction.

### Privacy, security and commercial access

Vehicle traces can reveal driver movements or commercially sensitive customer locations.
Whether a trace is personal data depends on identifiability and the applicable law; pseudonymization alone does not automatically make it anonymous.
The EU General Data Protection Regulation (GDPR), where applicable, establishes obligations including data minimization and security, subject to its terms and legal basis.
The regulation does not mandate a particular control-tower architecture or prohibit all location sharing.

Define a permitted purpose, access scope, retention period and handling for disputes before opening a multi-party feed.
A receiver may need proof of delivery without needing continuous access to a driver's complete route.
A manufacturer may need aggregate exception information but not another carrier's customer list.
Segregate tenant access and audit queries as well as event publication.

## 9. Industry practice and case evidence

### Standards adoption does not eliminate operating work

GS1 describes EPCIS as a standard for capturing and sharing visibility event data and identifies applications in traceability and other supply-chain processes.
Its documentation is authoritative about intended standard behavior, not independent evidence of a particular company's savings.
A practical program must still fund scanning, master data, exception ownership and partner onboarding.

**Inference:** A participant with excellent warehouse scans but no agreed shipment-to-pallet association can produce valid-looking EPCIS events while a recipient remains unable to identify which customer orders are at risk.
That is a process and data-contract problem, not a reason to abandon a standard.

### Walmart and food traceability

Walmart publicly described work with IBM on blockchain-based food traceability and reported that tracing the origin of mangoes in a test fell from days to seconds.
The company later communicated traceability requirements for leafy greens to suppliers.
These are company-reported cases, not independent proof that every implementation produces the same outcome.
The public material describes a particular traceback exercise; it should not be interpreted as a measured reduction in foodborne illness or a universal return on investment.

**Inference:** The operational value of faster traceback depends on the identity and completeness of records across participants, as well as on a decision process capable of acting on the result.
A fast lookup of incomplete or incorrectly linked records would not by itself isolate affected goods.
The source supports the existence and reported timing of Walmart's case, while this lesson's inference is a general interpretation.

### Maersk and remote container management

Maersk describes remote container management for refrigerated containers, including access to reefer conditions and location information.
This is a carrier's description of a service, not an independent estimate of avoided spoilage.
It illustrates that condition and location can be provided together for operating decisions.
It does not establish that a container-air measurement equals the condition of every item inside it.

**Inference:** A beneficial workflow requires a customer or carrier to receive, interpret and act on a warning before the intervention deadline.
If a warning arrives after delivery, it may support investigation but cannot reverse a missed pre-delivery action.
The case does not provide the numerical assumptions used in section 6.

### FDA outbreak response context

FDA's Food Traceability Rule materials describe why faster access to relevant records matters during foodborne illness investigations and specify the rule's framework for applicable foods and activities.
This is a regulatory purpose and framework, not evidence that every company using EPCIS complies automatically.
The legal scope, required records and enforcement information must be checked in current FDA materials.

### Practice transfer

For a new implementation, select one decision with measurable lead time.
Document the source-to-object relationship before expanding the number of feeds.
Observe how often an operator acts, whether the action was possible before the deadline, and whether the outcome changed.
Use product and route cohorts to avoid comparing incomparable shipments.
Treat the public examples as motivation to investigate, not as promised benefits.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Silent sensor | No heartbeat or measurement for an expected slot. | Battery, network or device failure interrupts evidence. | Unknown condition displayed as normal. | Generate a separate missing-data exception and record last valid reading. |
| Wrong sensor association | One sensor appears on two concurrent trailers. | Installation or reassignment record is stale. | Temperature applied to the wrong goods. | Time-bound associations, enforce conflict checks and investigate overlaps. |
| Vehicle mistaken for cargo | Vehicle reaches site but no load scan exists. | GPS is used as a proxy for a pallet. | False delivery or loss claim. | Require shipment association and the appropriate handoff evidence. |
| Duplicated messages | Same producer event ID arrives repeatedly. | Network retries or replay. | Inflated counts or repeated actions. | Idempotency key, retry monitoring and one incident per logical event. |
| Distinct events collapsed | Different measurements share a coarse key. | Deduplication uses only pallet or minute. | Real readings and excursions disappear. | Use stable event identity and test collision scenarios. |
| Late reading | Receipt latency grows near intervention time. | Buffering or batch upload. | An actionable warning arrives after the decision. | Publish latency by decision window and escalate lag. |
| Clock skew | Negative latency or impossible sequence. | Device clock or offset is wrong. | Incorrect timeline and misleading duration. | Preserve timestamps, monitor clock health and quarantine ambiguous time. |
| Unit mismatch | Sudden jump after onboarding a new device. | Fahrenheit is read as Celsius or vice versa. | False alarm or missed excursion. | Require units, convert explicitly and validate plausible ranges. |
| Misleading sensor placement | Temperature looks normal while receiver reports warm product. | Return-air sensor does not measure product core. | Unsupported product-release decision. | Document placement and use product-specific quality evidence. |
| Master-data drift | Unknown or conflicting site mappings increase. | Partner renames or reuses a location code. | Events route to the wrong receiver. | Version crosswalks and resolve using event-time validity. |
| Incomplete denominator | Coverage improves after silent trips disappear from the feed. | Received events define the population. | False impression of improved service. | Use an independent eligible manifest and reconcile exclusions. |
| Threshold without owner | Alerts accumulate without acknowledgement. | No one has authority or time to respond. | The feed creates work without preventing losses. | Assign decision rights, due times and escalation. |
| Silent correction | Historical status changes without a reason. | Prior events are overwritten. | Audit and dispute reconstruction become impossible. | Retain original, error declaration, corrective event and action history. |
| Semantic mismatch | Two partners dispute what “arrived” means. | Geofence and proof-of-delivery share a label. | Incorrect customer promise or charge dispute. | Specify physical trigger, source and acceptable evidence. |
| Overbroad access | Users can query unrelated driver routes. | One shared feed has excessive permissions. | Privacy and commercial exposure. | Purpose-based access, tenant isolation and retention limits. |
| False precision | Dashboard shows exact excursion duration from two samples. | Interpolation is presented as observation. | Unjustified hold or release. | Show sampled values, gaps and explicit interval assumptions. |
| Bad batching | All updates arrive after daily cut-off. | Source system exports once a day. | No timely intervention despite high eventual completeness. | Contract on decision-time receipt, not just eventual delivery. |
| Suppressed conflicts | A preferred source overwrites another without a trace. | Source-priority rule is implemented as deletion. | Disputes cannot be reconstructed. | Preserve assertions and show the reconciliation rule and provenance. |

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test or challenge it |
|---|---|---|---|
| Manifest defines eligible trips. | Coverage needs a denominator independent of messages. | Unrecorded subcontracted trips or cancellations. | Reconcile manifest to dispatch and invoices; document exclusions. |
| Event IDs are stable for retries. | Deduplication should remove retries without removing observations. | A gateway generates a new ID on every resend. | Replay known messages and compare producer IDs and payloads. |
| Clock semantics are agreed. | Latency compares event and receipt times. | Device clock drifts or timestamps have unknown offsets. | Compare with synchronized reference and inspect negative latencies. |
| Sensor is associated with the correct trailer. | The measurement must apply to a shipment. | Sensor moved without an installation event. | Audit install and removal scans against physical checks. |
| Temperature measurement represents the intended property. | Review trigger must correspond to the stated handling process. | Probe moved from return air to door frame. | Inspect installation and compare to a reference under the operating procedure. |
| A measurement is useful before the deadline. | An intervention requires lead time. | Offline buffering delays an alert until after acceptance. | Measure decision-time availability by trip and incident. |
| One slot can be credited once. | Slot coverage must not exceed 100%. | Multiple transmissions are counted as separate slots. | Reconcile unique slot keys and inspect duplicate rates. |
| A review trigger is not an automatic rejection. | Operational response depends on product and quality rules. | Dashboard is configured to dispose goods automatically. | Review decision rights and trace sample incidents through closure. |
| Sources use consistent milestone definitions. | Cross-partner comparisons require comparable events. | Carrier changes from driver-entered to inferred arrival. | Inspect source documentation and validate against physical samples. |
| Public case claims are transferable only with care. | Case evidence can guide design but differs from local conditions. | Product, route, scale or process differs. | Run a locally appropriate pilot and record baseline differences. |
| Value assumptions capture incremental response. | Gross benefit must reflect change caused by the workflow. | Existing team would have taken the same action anyway. | Compare with a baseline or defensible counterfactual. |
| Costs include incremental operations. | Net value must deduct the whole relevant cost. | Investigation labor or carrier support is excluded. | Reconcile finance, service and case-handling costs. |

The lesson's code is a transparent calculation, not a production ingestion service.
It does not implement authorization, schema validation, durable storage, watermarks, replay or EPCIS capture interfaces.
Its deterministic sample data should not be interpreted as a measured carrier performance distribution.

## 12. Exercises

### Exercise 1 — Separate occurrence from assertion

**Time:** 10 minutes.

A truck sends coordinates inside a warehouse geofence at 09:15 UTC.
The carrier sends “delivered” at 09:20 UTC.
The receiver scans pallet P at 09:45 UTC and records a shortage at 10:00 UTC.

**Deliverable:** Four labeled statements: observation, assertion, supported conclusion and unsupported conclusion.
**Acceptance criteria:** Distinguish the vehicle from the pallet; do not treat geofence entry as receipt; identify the evidence and time for the shortage.

### Exercise 2 — Map a decision to data

**Time:** 20 minutes.

A customer will reject an order if a dock appointment is missed, unless the planner obtains a new slot before 10:30 UTC.
A carrier provides position every 10 minutes and a milestone file every evening.

**Deliverable:** A five-row decision-to-data table naming decision, owner, deadline, required associations and fallback when evidence is missing.
**Acceptance criteria:** Separate forecast from commitment; explain why an evening milestone cannot support a 10:30 UTC intervention; state what a position does not prove.

### Exercise 3 — Choose event semantics

**Time:** 15 minutes.

A warehouse packs cases into a pallet, ships the pallet, and later repacks one case into a new product configuration.
An order reference is associated with the shipped objects.

**Deliverable:** A mapping from each operation to the EPCIS event type or types you would investigate, with one sentence of reasoning for each.
**Acceptance criteria:** Explain the parent–child association, physical shipping context, transaction reference and transformation separately; avoid claiming that the event type alone proves truth.

### Exercise 4 — Calculate quality measures

**Time:** 15 minutes.

A trip has 12 expected slots.
Ten unique readings match slots; seven arrived within the allowed 10 minutes.
There are 13 delivered messages, including three duplicate deliveries.

**Deliverable:** Slot coverage, timely slot coverage, timeliness among received, and duplicate delivery rate, each with numerator and denominator.
**Acceptance criteria:** Use 12 expected slots, 10 unique readings and 13 deliveries in the appropriate measures; round to one decimal place; state what the missing slots imply.

### Exercise 5 — Calculate incremental value

**Time:** 20 minutes.

A network has 1,500 eligible trips per year; 6% have an incident.
Current operations effectively intervene on 20% of incidents; the proposed workflow would intervene on 40%.
Assume USD 1,200 of avoidable loss per additional effective intervention and USD 14,000 annual incremental cost.

**Deliverable:** Annual incident expectation, additional effective interventions, gross benefit and net value, plus two assumptions needing evidence.
**Acceptance criteria:** Use the incremental 20 percentage-point uplift, not the full 40%; label estimates and avoid claiming observed savings.

### Exercise 6 — Critique the dashboard

**Time:** 20 minutes.

A dashboard shows 100% healthy trips because every trip that sent at least one reading had no reported alarm.
It includes no manifest, timestamps, sensor association history or missing-data indicator.

**Deliverable:** A critique with at least five specific defects and a replacement scorecard outline.
**Acceptance criteria:** Include an independent denominator, unknown status, timeliness, association validity and alert ownership; distinguish a normal observation from no observation.

### Exercise 7 — Design late and corrected events

**Time:** 25 minutes.

A producer sends event X twice, then sends a declaration that X was erroneous and provides a corrected event Y.
A separate earlier loading event arrives after both.

**Deliverable:** A timeline showing receipt order, physical-event order where known, retained records and the state as known before and after correction.
**Acceptance criteria:** Process the retry idempotently; retain X, its error declaration and Y; permit a valid late event to revise a derived state; preserve decision-time history.

### Exercise 8 — Draft a partner contract

**Time:** 30 minutes.

A manufacturer wants a carrier's GPS and temperature feed for a high-value lane.
The carrier is concerned about exposing drivers and unrelated customers.

**Deliverable:** A one-page contract outline with data fields, meaning, identities, latency, quality measures, correction, access, retention, owners and escalation.
**Acceptance criteria:** Set a decision-specific deadline and denominator; distinguish vehicle location from cargo evidence; describe access minimization and a process for unknown or late data; avoid asserting that EPCIS alone settles legal obligations.

## 13. Self-check answers

### 1. Occurrence and assertion

The geofence observation is a device's reported vehicle position inside the configured boundary at 09:15 UTC.
The carrier's 09:20 UTC delivered message is a business assertion, not necessarily a physical scan.
The pallet scan supports that pallet P was observed at the receiver at 09:45 UTC, subject to reader and identity reliability.
The 10:00 UTC shortage record supports that the receiver reported a shortage at that time; it does not by itself identify when or where the loss occurred.
“Pallet P was received at 09:15 UTC because the truck crossed the geofence” is unsupported.

### 2. Decision-to-data map rubric

A strong answer assigns the new-slot decision to the planner and sets a deadline before 10:30 UTC.
It includes shipment-to-vehicle association, destination and appointment, current position time, an arrival forecast and its uncertainty.
It distinguishes the contractual appointment from the forecast.
It notes that a vehicle may be at a location without the intended goods or a completed handoff.
If current evidence is missing, it routes a carrier call or other approved fallback to an owner before the deadline.
An evening file may help reconstruct the day, but arrives too late for the specified decision.

### 3. Event semantics

An AggregationEvent can express the cases' parent–child relationship with the pallet.
An ObjectEvent can express a shipping observation for the pallet in the appropriate business context.
A TransactionEvent can express association with the order where its semantics match the process.
A TransformationEvent can express input-to-output relationships when the repack actually transforms the identified product configuration.
Depending on the physical process, disaggregation and new aggregation may also be appropriate.
The applicable standard and business scenario determine the exact payload; the English operation name alone is not enough.

### 4. Quality calculation

Slot coverage is `10 / 12 = 83.3%`.
Timely slot coverage is `7 / 12 = 58.3%`.
Timeliness among received readings is `7 / 10 = 70.0%`.
Duplicate delivery rate is `3 / 13 = 23.1%`.
Two expected slots have no matching unique reading in the stated snapshot.
Those slots are unknown, not automatically normal or necessarily product failures.

### 5. Value calculation

Expected incidents are `1,500 × 0.06 = 90 per year`.
The incremental intervention rate is `40% − 20% = 20 percentage points`.
Expected additional effective interventions are `90 × 0.20 = 18 per year`.
Gross expected benefit is `18 × USD 1,200 = USD 21,600 per year`.
Net expected value is `USD 21,600 − USD 14,000 = USD 7,600 per year`.
The incident rate, intervention effectiveness, unit loss and incremental cost require evidence.
A comparison should account for actions already taken without the new feed.

### 6. Dashboard critique rubric

A strong answer identifies that the dashboard excludes completely silent trips and does not measure full-period coverage.
It also flags the absence of event and receipt times, sensor identity, assignment validity, calibration or placement context, and an unknown state.
A replacement starts with eligible trips and expected slots, reports timely and eventual coverage separately, and shows source, last measurement and exception owner.
A complete answer explains that no reported alarm is not evidence of normal conditions.

### 7. Late and corrected events rubric

A strong timeline keeps both deliveries of X in the delivery audit while applying the logical event once.
It retains X, the declaration that X is in error, Y and their relationship.
It records the state and evidence available when the operator acted, then shows the revised state after correction.
The late loading event is evaluated by its event time and association rules rather than rejected merely because it arrived last.
Any new action is linked to the existing case and requires an appropriate owner.

### 8. Partner contract rubric

A strong outline defines an eligible population and decision deadline, not just a technical endpoint.
It lists producer and object identifiers, event and receipt times, location precision, temperature units and placement, source semantics and valid associations.
It specifies a deduplication key, correction method, late-data path, quality denominators and escalation owner.
It restricts access by purpose and party, addresses retention and security, and states that legal applicability requires review.
It explains how gaps appear to operators and who can authorize a hold or release.

## 14. Interview questions

1. **How would you tell whether a truck has delivered a pallet?**
   A strong answer separates vehicle position, load association, unloading and receiver confirmation.
   A weak answer calls a geofence crossing proof of delivery.
2. **What does EPCIS add to a stream?**
   A strong answer describes shared event semantics and relevant event dimensions while naming remaining agreement and data-quality work.
   A weak answer claims EPCIS guarantees accurate or complete data.
3. **How do you calculate sensor coverage?**
   A strong answer defines eligible trips or slots independently of the stream, matching rules and cutoff.
   A weak answer divides received records by received records.
4. **What happens when a device reconnects after an outage?**
   A strong answer expects buffering or loss, late events, possible retries, a decision-time gap and later reconciliation.
   A weak answer assumes every historical measurement was available to an operator in real time.
5. **How would you deduplicate?**
   A strong answer uses an agreed producer-scoped event identity, preserves distinct measurements and makes downstream effects idempotent.
   A weak answer drops every event after the first for each pallet.
6. **A temperature sample exceeds a threshold. What can you conclude?**
   A strong answer states the measured property, time, sensor placement and uncertainty, then invokes the quality review procedure.
   A weak answer declares all product unsafe from a single return-air reading.
7. **How should a correction be handled?**
   A strong answer keeps original and corrective assertions, their relationship and the state as known when actions occurred.
   A weak answer overwrites the prior event and loses the audit trail.
8. **What makes a feed commercially valuable?**
   A strong answer connects timely evidence to an authorized action and estimates incremental benefit net of cost.
   A weak answer counts dashboard views or all alerted shipment value as savings.
9. **How would you compare two carriers' timeliness?**
   A strong answer defines comparable event types, population, clocks, latency windows and exclusions, then reports denominators and distributions.
   A weak answer compares averages from differently defined milestones.
10. **How do you protect multi-party telemetry?**
    A strong answer applies purpose-limited access, tenant isolation, retention and audit controls alongside contractual review.
    A weak answer assumes a standards-compliant identifier makes location data public.

## 15. Key terms

- [Physical occurrence](#4-definitions-and-vocabulary) — an actual real-world change or condition.
- [Observation](#4-definitions-and-vocabulary) — what a device or person perceives.
- [Event record](#4-definitions-and-vocabulary) — a structured assertion about an occurrence.
- [Event stream](#3-the-core-mental-model) — a sequence of delivered event records.
- [IoT](#4-definitions-and-vocabulary) — connected devices that measure or interact with the environment.
- [Telematics](#4-definitions-and-vocabulary) — remotely communicated vehicle or asset information.
- [Geofence](#4-definitions-and-vocabulary) — a defined geographic region used for position-based events.
- [Sensor](#4-definitions-and-vocabulary) — a device measuring a specified property.
- [EPCIS](#8-data-standards-and-regulatory-context) — GS1 visibility event standard.
- [CBV](#8-data-standards-and-regulatory-context) — GS1 vocabulary for interoperable business context.
- [ObjectEvent](#step-6--choose-the-relevant-epcis-representation) — event about identified objects and their context.
- [AggregationEvent](#step-6--choose-the-relevant-epcis-representation) — event about parent–child associations.
- [TransactionEvent](#step-6--choose-the-relevant-epcis-representation) — event relating objects and business transactions.
- [TransformationEvent](#step-6--choose-the-relevant-epcis-representation) — event relating input and output objects.
- [AssociationEvent](#step-6--choose-the-relevant-epcis-representation) — event expressing an appropriate persistent association.
- [Business step](#4-definitions-and-vocabulary) — process context of an event.
- [Disposition](#4-definitions-and-vocabulary) — asserted business condition of an object.
- [Read point](#4-definitions-and-vocabulary) — location at which an event was observed.
- [Event time](#step-9--process-time-retries-and-corrections) — asserted time of the occurrence.
- [Record time](#4-definitions-and-vocabulary) — time an EPCIS repository recorded an event.
- [Latency](#step-9--process-time-retries-and-corrections) — difference between explicitly named timestamps.
- [SSCC](#8-data-standards-and-regulatory-context) — identifier for a logistic unit.
- [GLN](#8-data-standards-and-regulatory-context) — identifier for a party or location.
- [Master data](#step-4--agree-on-identity-and-associations) — reference data for interpreting identifiers and attributes.
- [Provenance](#3-the-core-mental-model) — origin and transformation history of evidence.
- [Idempotency](#step-9--process-time-retries-and-corrections) — repeated processing without repeated business effect.
- [Error declaration](#step-9--process-time-retries-and-corrections) — a way to indicate an EPCIS event is erroneous.
- [Coverage](#step-11--measure-quality-against-the-expected-population) — qualifying evidence divided by a specified expected population.
- [Timeliness](#step-11--measure-quality-against-the-expected-population) — evidence arrival relative to a specified deadline.
- [Decision right](#step-12--turn-quality-results-into-a-bounded-workflow) — authority to make a specified operational decision.

## 16. Sources

1. GS1, [EPCIS and CBV standards](https://www.gs1.org/standards/epcis): standard resources and current releases.
   Primary source for EPCIS capabilities and specification links; consult the exact applicable version.
2. GS1, [EPCIS 2.0.0 standard](https://ref.gs1.org/standards/epcis/2.0.0/): normative event structures, interfaces and semantics.
   Use this rather than the lesson's conceptual summary for implementation.
3. GS1, [Core Business Vocabulary 2.0.0](https://ref.gs1.org/standards/cbv/2.0.0/): vocabulary terms and identifiers used with EPCIS.
4. GS1, [GS1 identification keys](https://www.gs1.org/standards/id-keys): roles and resources for identification keys including GTIN, GLN and SSCC.
5. U.S. Food and Drug Administration, [Food Traceability Rule](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-requirements-additional-traceability-records-certain-foods): official scope, records and current implementation information.
   Consult current FDA information before making compliance decisions.
6. U.S. Food and Drug Administration, [Drug Supply Chain Security Act](https://www.fda.gov/drugs/drug-supply-chain-integrity/drug-supply-chain-security-act-dscsa): official context for the separate U.S. drug-tracing framework.
7. European Union, [Regulation (EU) 2016/679, General Data Protection Regulation](https://eur-lex.europa.eu/eli/reg/2016/679/oj): primary legal text; applicability is fact-specific.
8. Walmart, [Blockchain: a new era of transparency in the food system](https://corporate.walmart.com/news/2016/10/19/blockchain-a-new-era-of-transparency-in-the-food-system): company account of the traceability initiative.
9. Walmart, [Food traceability initiative](https://corporate.walmart.com/content/dam/corporate/documents/newsroom/2018/09/24/leafy-greens-supplier-letter.pdf): supplier communication on leafy-greens traceability.
10. Maersk, [Remote Container Management](https://www.maersk.com/transportation-services/reefer-logistics/remote-container-management): carrier description of reefer monitoring capabilities; not independent evidence of savings.

## 17. Links

- [Module 19 index](./_index.md) — sequence, prerequisites and topic map.
- [Module 19 survey and lab sections](../../../course/19_control_towers_visibility_events.md) — overview and hands-on work.
- [Visibility layers](./01_visibility-layers.md) — sibling topic on the objects visible to a tower.
- [Event time versus processing time](./03_event-time-processing-time.md) — sibling topic on ordering and lateness.
- [Exception management and queue design](./07_exception-queue.md) — sibling topic on operational ownership.
- [Module 30 lab index](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md) — simulation and production lab map.

Sibling topic links follow the module index and may remain unavailable until those planned files are published.
