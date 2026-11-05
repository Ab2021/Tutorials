# Food Safety, Cold Chain, and Courier Welfare

> Module 22 — Quick Commerce, Food Delivery, and On-Demand Logistics
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: Topics 1, 4, 5, 6, and 7
> Version 2026.5

## 1. Why this topic exists

A fast order can fail before it reaches the customer even when the app says it arrived on time.
A chilled product can warm during receiving, picking, staging, or the last delivery leg.
A courier can feel pressure to ride unsafely to protect an acceptance score or a customer promise.
A missed temperature measurement can turn an otherwise uneventful trip into an unresolvable product-safety question.
An apparently profitable batch can shift uncompensated time and risk to a worker.

Quick commerce compresses decisions, not the underlying biological, physical, and human constraints.
A platform must decide what it can promise only after it knows what it can keep safe.
The restaurant, store, platform, courier, and customer may each control a different segment of the journey.
The handoffs between them are therefore part of the control system.

This unit asks an operations lead to design a delivery system that can meet a service promise without making product safety or worker welfare negotiable.
It combines hazard analysis and critical control point (HACCP) thinking with practical event data, exception handling, courier protections, and a transparent economic comparison.
HACCP is a systematic method for identifying, evaluating, and controlling significant food-safety hazards.
A HACCP plan is not synonymous with a temperature dashboard.
Whether a particular operation must have a HACCP plan, and its exact required form, depend on jurisdiction and activity.

The relevant decision makers are different people.
A food-safety professional establishes product-specific rules and disposition authority.
A store lead checks and records conditions, contains suspect product, and stops the affected flow.
A dispatch lead maintains a feasible plan and changes the promise when safety cannot be maintained.
A courier can identify damage, unsafe loads, unsafe weather, and inaccessible handoffs, and must have a usable channel to report them.
A finance owner evaluates resource requirements and the economic consequences without treating unvalidated risk savings as guaranteed revenue.

### Decisions and accountability

| Decision | Accountable owner | Operational input | Record or outcome |
|---|---|---|---|
| Define product scope and intended use | Food-safety lead | Product, supplier, destination, use | Approved product profiles |
| Decide significant hazards and controls | Food-safety lead with HACCP team | Flow map, hazard evidence, prerequisites | Signed hazard analysis |
| Approve critical limits where needed | Qualified food-safety authority | Validation evidence and applicable law | Versioned HACCP plan |
| Check incoming and stored product | Store manager | Deliveries, equipment, readings | Receiving and monitoring records |
| Hold or release suspect product | Authorized food-safety decision maker | Time, temperature, identity, evidence | Disposition and rationale |
| Stop unsafe work | Courier and on-duty operations lead | Load, road, weather, fatigue or incident | Stop and escalation record |
| Set route and promise guardrails | Dispatch owner with safety and workforce leads | Exposure budget, travel uncertainty, capacity | Approved dispatch rules |
| Set compensation and facilities | Workforce operations and legal owners | Paid-time definition, worker feedback, local law | Policy and monitored outcomes |
| Fund equipment or staffing | Operations and finance owners | Incremental cost, capacity, uncertainty | Investment decision |
| Change or restart a control | Food-safety and operations owners | Verification, corrective action, tests | Approval and effective version |

A target is a desired aggregate result, not a substitute for a required safety limit.
A plan is an intended course of action, not proof that the action happened.
A forecast is an estimate, not a customer commitment or an observed product temperature.
A customer commitment cannot authorize unsafe handling or unsafe riding.

### Scope and boundary

The examples concern chilled, packaged ready-to-eat food in a hypothetical delivery operation.
They also discuss insulated transport and courier welfare in general.
They do not establish legal thresholds, worker classification, clinical advice, or safe shelf life for an actual product.
Raw food, hot food, frozen food, medicines, and other products require separate hazard analyses and may require different controls.
Apply the current local rules and qualified professional judgment before operating.

The central business question is not whether safety has a positive standalone return in every forecast.
Some controls are required and must be funded regardless of expected savings.
For discretionary alternatives, compare feasible ways of meeting the same safety and welfare requirements.
Make uncertainty, ownership, and the counterfactual visible.

## 2. Learning outcomes

After this unit you can:

1. **Map** a chilled delivery flow and identify hazards and handoff owners.
   Exercise 1 asks for a product-flow and ownership map.
2. **Distinguish** prerequisite controls, operational checks, critical control points, and critical limits.
   Exercise 2 tests whether a proposed control is actually a critical control point.
3. **Calculate** observed and unknown exposure intervals without interpolating unsupported product temperatures.
   Exercises 3 and 4 test this calculation.
4. **Choose** an excursion response that contains product, protects traceability, and assigns disposition to an authorized person.
   Exercise 5 requires a reasoned incident record.
5. **Design** dispatch and worker-welfare guardrails, including a practical stop-work path.
   Exercise 6 tests the decision design.
6. **Compute** incremental operating cost and a bounded business case without inventing incident probabilities.
   Exercise 7 tests the counterfactual and break-even arithmetic.
7. **Evaluate** sensor coverage, worker outcomes, and intervention evidence while recognizing missing data and confounding.
   Exercise 8 asks for an evaluation plan.
8. **Explain** the limits of the runnable monitoring calculation.
   Exercise 4 asks you to modify and interpret it.

An adequate answer names the evidence available when the decision was made.
It separates measured conditions from modelled exposure and distinguishes food disposition from customer service recovery.
It also states who can override, pause, investigate, and authorize a restart.

## 3. The core mental model

```text
Product profile + intended consumer + local requirements
                       |
                       v
          Flow map + hazard analysis + prerequisite programs
                       |
                       v
    Approved controls, limits, monitoring and response ownership
                       |
                       v
 Supplier -> receive -> store -> pick -> pack -> stage -> handoff
                       |                         |
                       |                         v
                       |                  courier + route + load
                       |                         |
                       v                         v
         identity and condition events -> deliver / return / dispose
                       |                         |
                       +----------+--------------+
                                  v
                 monitoring + worker voice + incident reports
                                  |
                                  v
              hold / stop -> assess -> disposition / safe restart
                                  |
                                  v
               verify control -> learn -> approve policy change
```

The physical product flows left to right.
Evidence follows the product, the bag, the storage equipment, and the worker through their separate identifiers.
Product temperature is a physical property of the food, not an attribute that can be inferred perfectly from an order status.
Air temperature, surface temperature, package-core temperature, and logger temperature are different observations.
A sensor in one bag does not measure every package in another bag.

A hazard is a biological, chemical, or physical agent with potential to cause an adverse health effect.
Examples include pathogen growth, allergen cross-contact, chemical contamination, and glass fragments.
Warmth is generally a condition that can support a biological hazard; a single reading alone does not identify an organism or establish illness.
A damaged package may also compromise hygiene even when it is cold.

Prevention begins before dispatch.
Supplier approval, cleaning, pest control, personal hygiene, allergen separation, equipment maintenance, and staff training are prerequisite programs.
Some operating conditions may be monitored operationally without being a critical control point (CCP).
A CCP is a step at which a control can be applied and is essential to prevent, eliminate, or reduce a significant hazard to an acceptable level.
The HACCP team determines CCPs through product- and process-specific analysis; do not label every temperature check a CCP by default.

Each identified CCP needs an appropriate critical limit and monitoring arrangement under the applicable HACCP framework.
A critical limit separates acceptability from unacceptability at that CCP.
An alert threshold may be more conservative than a critical limit so staff can intervene before a deviation.
Neither a generic alert threshold nor this educational example establishes the legal or safe limit for a real product.

The cold chain is a sequence of controlled environments, not one insulated bag.
The relevant clock may include receiving, unrefrigerated picking, staging, loading, travel, and delivery or failed delivery.
The time at one stage cannot be treated as independent of the time already used upstream.
An apparently quick final leg cannot erase an earlier unknown or excessive exposure.
Likewise, refrigeration can slow warming or microbial growth, but putting suspect food back into a refrigerator does not automatically make it safe.

Courier welfare is a system property as well.
A routing objective, an incentive, a promise, and the availability of toilets or rest locations can shape worker behavior together.
A courier should not be forced to trade a safe route, hydration, or an incident report against access to future work.
The system needs a means to report without retaliation and a human response path.

A safe decision is made in this order:

1. Determine whether product and worker constraints are satisfied or whether required evidence is absent.
2. Stop, hold, split, reassign, or shorten the offer when the constraints cannot be demonstrated.
3. Among feasible options, compare customer service and economic outcomes.
4. Observe actual execution and intervene when the facts change.
5. Learn from failures and near misses without retrospectively changing what was known at decision time.

This order prevents a low cost per order from compensating for a failed mandatory control.
It also avoids presenting safe-work protection as a soft preference that an optimizer can buy away.

## 4. Definitions and vocabulary

| Term | Definition and unit where relevant | Common misuse |
|---|---|---|
| Hazard | Biological, chemical, or physical agent with potential adverse health effect | Calling every service delay a food-safety hazard |
| Risk | A function of probability and severity of an adverse effect | Treating hazard presence as a measured probability |
| HACCP | Hazard analysis and critical control point system | A synonym for any checklist |
| Prerequisite program | Foundational hygienic conditions and activities | Assuming prerequisites need no monitoring or ownership |
| Hazard analysis | Evaluation of hazards and controls for a product and process | Copying a generic list without evaluating the operation |
| CCP | Step where control is essential for a significant hazard | Declaring every monitoring point a CCP |
| Critical limit | Criterion separating acceptability from unacceptability at a CCP | Using an unvalidated alert level as the critical limit |
| Operational limit | Internal working threshold set to trigger action before a limit | Treating it as automatically mandated by law |
| Monitoring | Planned observations or measurements to assess control | Equating a sensor heartbeat with a valid reading |
| Deviation | Failure to meet a critical limit or other specified requirement | Treating an unresolved data gap as a proven deviation |
| Corrective action | Predetermined action when control is lost, including product and cause | Only fixing the device and ignoring affected food |
| Correction | Action to address an immediate detected nonconformity | Assuming an immediate fix prevents recurrence |
| Verification | Activities, other than monitoring, that determine whether the system operates as intended | Calling every routine measurement verification |
| Validation | Evidence that a control measure can achieve the intended outcome | Assuming repeated readings prove the control was designed correctly |
| Cold chain | Controlled temperature conditions across product handling and distribution | Assuming one insulated bag controls the whole chain |
| Product temperature | Temperature of the food at a specified measurement point, °C | Treating a nearby air reading as identical |
| Air temperature | Temperature of air near a sensor, °C | Using it as proof of food-core temperature |
| Core temperature | Temperature at an appropriate internal product location, °C | Assuming a surface infrared scan measures it |
| Time-temperature exposure | Duration and conditions experienced across relevant stages, min and °C | Summing only the delivery leg |
| Excursion | Condition outside a specified permitted range or operating rule | Assuming all excursions imply the same disposition |
| Unknown interval | Period for which required condition evidence is absent | Counting missing readings as in-range |
| Sensor calibration | Comparison with a traceable reference under defined conditions | Resetting a logger without a documented comparison |
| Sensor accuracy | Closeness of an indicated value to a reference under stated conditions | Assuming claimed accuracy applies after damage or at all temperatures |
| Probe placement | Physical location of a sensor relative to food and cooling sources | Placing a probe beside a gel pack and calling it worst case |
| Lot | Identified quantity associated with production or handling conditions | Treating an order ID as a supplier lot ID |
| Traceability | Ability to follow product identity and movement through relevant stages | Assuming a customer receipt traces all units |
| Hold | Segregation and status block pending an authorized decision | Leaving stock sellable while emailing someone |
| Disposition | Authorized determination of what happens to affected product | Letting dispatch decide based on customer lateness |
| Recall | Removal or correction process for product that has entered distribution, as applicable | Calling an internal stock hold a completed recall |
| FEFO | First-expired, first-out stock rotation | Using arrival order when expiry dates differ |
| Tamper evidence | Packaging feature indicating possible interference | Treating an intact seal as proof of temperature control |
| Chain of custody | Documented responsibility and transfer of an item | Treating a GPS event as a signed product handoff |
| Insulated container | Container designed to slow heat transfer | Assuming it creates cold without a conditioned source |
| Coolant | Material or device providing cooling capacity | Assuming one pack suits every load and duration |
| Exposure budget | Approved time allowance under specified validated conditions, min | An automatically safe time outside refrigeration |
| Route feasibility | Whether a candidate satisfies all hard constraints with available evidence | Comparing only estimated road travel |
| Utilization | Share of a defined time denominator spent in a defined activity, % | Calling busy time income or welfare |
| Paid time | Time compensated under the relevant agreement and law | Assuming all app-on time or all waiting is paid |
| Net earnings | Gross compensation less specified worker-borne costs, currency per chosen period | Omitting vehicle costs or comparing mismatched time windows |
| Fatigue | Reduced capability associated with sustained work, sleep loss, or related factors | Inferring an individual's fitness from one utilization figure |
| Stop-work right | Policy or legal entitlement to cease unsafe activity as applicable | Publishing a rule without a usable and non-retaliatory process |
| Near miss | Event that could have caused harm but did not, as recorded | Treating a high reporting rate as necessarily worse safety |
| Denominator | Population or exposure over which a rate is calculated | Dividing worker incidents by orders when worker-hours are the exposure of interest |
| Counterfactual | What would have happened under a specified alternative | Assuming the alternative is zero cost or zero risk |
| Expected loss | Probability-weighted consequence over a specified period | Presenting an unsupported point estimate as observed savings |

## 5. The mechanics, step by step

### Step 1 — Define the product and operating boundary

**Input:** the actual product, labeling, intended use, suppliers, sales channels, destinations, and delivery methods.
**Output:** an approved scope statement and a product profile.

Name the product form, packaging, shelf life, storage instructions, and whether it is ready to eat.
Record whether the consumer must cook the item and whether vulnerable populations are expected users.
A vacuum-packed chilled meal, a raw poultry tray, and an ice cream tub should not share one undifferentiated control plan.
Record relevant allergens and how a damaged package could affect other orders.
Identify the legal entity responsible at each stage and the jurisdiction through which the product moves.

Map product from the supplier to final delivery and include returns, replacements, refunds, and disposal.
Include a restaurant handoff if a restaurant prepares the meal.
Include store receiving and storage if a dark store holds inventory.
Include any third-party cross-dock or transfer vehicle.
Identify the customer-contact boundary and what happens when the customer is absent.

Branch when the product is hot, frozen, raw, unpackaged, or otherwise outside the chilled ready-to-eat profile.
Do not transfer a numerical control to a new product without evidence that it is appropriate.
The food-safety lead signs the scope and reviews it whenever a product, packaging system, or process materially changes.

### Step 2 — Draw and verify the real flow

**Input:** the scoped process and a walk-through at representative operating times.
**Output:** an agreed flow diagram with handoffs, delays, and rework loops.

Follow at least one actual item through receiving, storage, picking, packing, staging, loading, travel, and handoff.
Observe both ordinary and peak conditions.
Ask workers how exceptions really occur, rather than drawing only the intended system.
Document where stock waits outside refrigeration and where measurement is possible.
Record the location of each temperature zone, transfer point, and staging area.

Add loops for canceled orders, inaccessible customers, route changes, and returns.
Represent courier waiting separately from food waiting: the two clocks may start at different times.
Record who controls each transition and what information passes to the next owner.
A route trace does not prove a bag was closed or that food was placed inside it.
A store scan does not prove the driver physically received the correct lot.

Validate the diagram by walking the operation and correcting mismatches.
Repeat the review when a new store layout, bag type, or batching policy changes the flow.
The shift lead owns day-to-day adherence; the HACCP team owns the controlled diagram.

### Step 3 — Establish prerequisite programs

**Input:** supplier agreements, site conditions, people, equipment, and current hygiene procedures.
**Output:** maintained prerequisite programs and evidence of execution.

Approve suppliers and specify receiving conditions, product identification, packaging integrity, and escalation contacts.
Define cleaning schedules and approved methods for chillers, totes, insulated bags, and reusable equipment.
Separate raw and ready-to-eat products and address allergen cross-contact.
Provide handwashing access and practical procedures for workers handling product.
Maintain pest control, potable water, waste handling, and appropriate personal hygiene.

Maintain refrigerator and freezer equipment under a schedule with a repair and escalation path.
Give insulated bags a cleaning, damage-inspection, replacement, and ownership process.
Train pickers, packers, dispatchers, and couriers on the steps they actually perform.
Make instructions understandable in the languages and formats required by the workforce.
Record attendance, observation, retraining, and coverage for new starters and contractors.

Prerequisites can fail during a peak even when paperwork exists.
Observe the physical process and resolve unavailable supplies, broken zippers, or inaccessible sinks.
An audit score is not a substitute for preventing contaminated product from being shipped.
The site manager assigns owners and documents completion and exceptions.

### Step 4 — Analyze hazards by product and step

**Input:** product profile, verified flow, hazard information, and prerequisite controls.
**Output:** a documented analysis of significant hazards and relevant control measures.

For each step, list reasonably foreseeable biological, chemical, and physical hazards.
Identify how contamination, survival, growth, or introduction could occur.
Examples include growth during warm staging, allergen cross-contact during packing, sanitizer residue, and packaging fragments.
Consider ingredient and supplier controls as well as controls within the delivery network.
Assess significance with a consistent method that considers likelihood and severity in context.
Record supporting evidence and the rationale, not just a numeric score.

List existing controls and identify gaps.
If a prerequisite reliably controls a hazard, explain the dependency and how that prerequisite is maintained.
If the hazard remains significant, identify an appropriate control measure at the relevant step.
A temperature reading may detect a problem but cannot by itself remove a pathogen.
A sealed package limits some contamination pathways but does not control growth inside it.

Reassess after a complaint trend, recall, formulation change, new courier model, or new scientific evidence.
The multidisciplinary team should include food safety, store operations, delivery operations, engineering, and worker representation where appropriate.
The food-safety lead owns the conclusion and escalation of unresolved evidence gaps.

### Step 5 — Determine CCPs and critical limits

**Input:** significant hazards, control measures, product use, and the applicable HACCP framework.
**Output:** a justified CCP determination and, where applicable, critical limits.

Evaluate whether a specific step is essential to prevent, eliminate, or reduce a significant hazard to an acceptable level.
Consider whether a later step reliably controls the hazard before consumption.
A ready-to-eat chilled product may have no later cooking step available as a control.
Do not assume that every delivery leg or every thermometer is a CCP.
Conversely, do not omit a necessary control merely because it is difficult to measure.

For each CCP, establish a measurable or observable critical limit appropriate to the control.
Identify the measurement location, unit, instrument, sampling frequency, and uncertainty.
Establish how an operator recognizes a deviation and what action is required.
Limits must be supported by applicable requirements and validation, not a desired service time.
An internal warning threshold can be set before the critical limit to create intervention time.
The warning is a management choice; it does not redefine the validated limit.

If evidence is insufficient to set or monitor a necessary limit, redesign the process or obtain qualified validation.
Do not dispatch using an invented threshold and promise to improve the documentation later.
The authorized HACCP team signs the determination and controlled plan version.

### Step 6 — Design receiving and storage controls

**Input:** supplier delivery, approved product specifications, measurement equipment, and storage capacity.
**Output:** accepted, rejected, or held product with traceable records.

Check identity, quantity, lot or batch information, date marking, packaging integrity, and delivery documentation.
Apply the approved receiving-temperature method to the specified sample or shipment.
Sampling must be representative of the risk and should not assume the easiest-to-reach package is the warmest.
A non-destructive between-pack reading, a surface reading, and a food-core reading have different limitations.
Use the approved instrument and method; avoid contaminating food during measurement.

If product does not meet the receiving criteria, identify and segregate it immediately.
Notify the supplier and the food-safety owner according to the agreed procedure.
Do not accept a verbal assurance in place of the required evidence.
Document the affected quantity, measurement, instrument, time, and disposition.

Place accepted chilled goods into appropriate storage promptly.
Maintain first-expired, first-out (FEFO) rotation where suitable, and remove damaged or expired stock under the approved policy.
Monitor equipment conditions with alarms that reach a named person.
A refrigerator display measures the equipment condition at its sensor, not necessarily every product's core temperature.
Investigate blocked airflow, overloaded shelving, open doors, defrost cycles, and power loss.

If a refrigerator alarms, protect workers and product first.
Do not merely silence the alarm or restart the device.
Mark potentially affected lots, establish the exposure window, and assess product according to the authorized procedure.
Record unavailable data explicitly.
The store lead owns containment; the authorized food-safety person owns product disposition.

### Step 7 — Validate the pack and exposure budget

**Input:** product thermal properties, starting condition, bag design, coolant, loading, ambient conditions, and route profiles.
**Output:** a documented set of operating conditions under which a delivery configuration is acceptable.

Test realistic high-load and low-load cases, not only an ideal tightly packed bag.
Include representative ambient heat, door openings, wait time, delivery sequence, and failed-handoff scenarios.
Include a damaged or poorly closed bag as a challenge to assess detectability, not as an accepted operating condition.
Assess cold spots near coolant and warm locations at the edge or top of a load.
Use product or validated surrogate measurements where needed, not a single conveniently located bag-air probe.

Document the initial product temperature and coolant conditioning.
Record load mass and configuration, packaging, ambient temperature, time points, and instruments.
Consider measurement uncertainty, sensor lag, and calibration.
Set the test duration long enough to represent the route and a defensible delay margin.
Determine the permitted configurations and their limits with a qualified food-safety authority.

A validated time allowance applies only within the conditions used to justify it.
Changing the bag, coolant, product size, ambient range, or delivery sequence can invalidate the evidence.
An insulation claim from a vendor is not automatically a product-specific validation.
If the configuration falls outside the approved envelope, redesign or hold rather than extrapolate casually.

### Step 8 — Pick, pack, and stage deliberately

**Input:** order lines, product conditions, assigned totes, and approved packaging configuration.
**Output:** a traceable, segregated, correctly conditioned order ready for handoff.

Use the approved pick sequence to minimize unnecessary time outside controlled storage.
Record the time at which relevant chilled product leaves the controlled environment when the plan requires it.
Prevent chilled goods from sitting beside hot food or chemicals.
Protect ready-to-eat food from raw-product leakage and allergen cross-contact.
Check packaging for damage and date marking before sealing.

Use appropriately conditioned coolant and an inspected, clean insulated container.
Do not place coolant where it can damage or freeze a product unless the design permits it.
Close the bag properly and use tamper-evident packaging where the operating procedure calls for it.
Separate orders so a courier can deliver the correct bag without unnecessary opening.
Record the link between order, lot when captured, bag, and pack completion.

A staging refrigerator may maintain conditions better than an ambient rack, but its actual performance still needs monitoring.
A bag waiting for a courier consumes some of its allowable time under the validated plan.
If the bag or courier is not ready, use the documented holding location and re-evaluate remaining time.
The packer checks the product and configuration; the shift lead owns deviations and unresolved capacity.

### Step 9 — Make the dispatch decision with hard guardrails

**Input:** ready times, remaining exposure allowances, route scenarios, courier availability, load limits, and customer commitments.
**Output:** an offer that is feasible under the approved food-safety and worker-safety rules.

Calculate the exposure already used at the actual decision time.
Include previous handling outside the relevant controlled environment as defined by the product plan.
Estimate remaining staging, loading, route, handoff, and a defensible delay margin.
For a multi-drop route, calculate each order's own timeline through its handoff.
The second stop can consume a substantially larger share of the available budget than the first.

Reject a route that exceeds the approved envelope or lacks required evidence.
Do not offset an unsafe route with a higher delivery fee or a customer discount.
If the route is infeasible, split the batch, obtain another courier, use approved chilled staging, change the offer, or cancel under the applicable policy.
Never assume that extending a customer promise extends product safety.

Check courier vehicle and bag capacity, weight distribution, and required separation.
Account for access, stairs, weather, road restrictions, and reasonable handoff time.
The dispatch system should not suggest unsafe speeds as the means of meeting a promise.
A worker can decline or stop an unsafe task through the available process; the operation must respond without punitive assumptions.
The dispatch owner maintains the rules, while the food-safety owner controls product constraints and the workforce owner controls welfare safeguards.

### Step 10 — Transfer custody and execute the trip

**Input:** packed order, identified courier, approved route, and handoff instructions.
**Output:** a recorded custody transfer and monitored delivery or exception.

At handoff, verify the correct bags, seals, loading, and order count.
Record actual transfer time rather than the time an assignment was proposed.
Tell the courier what to do if packaging leaks, a bag opens, a route changes, or a customer is unavailable.
Provide a reachable support channel that functions while a trip is active.

The courier transports the load using the approved configuration.
The worker should not need to manipulate a phone while riding or driving.
Route changes must be reassessed against remaining time, capacity, and safety conditions.
Any automated replan should preserve the ability to stop and report.

At delivery, record the actual handoff or the approved proof of delivery.
Preserve customer privacy and avoid collecting more location or personal data than needed.
A photo can establish a doorstep event but cannot prove product temperature.
If the recipient is absent, follow the approved failed-delivery procedure rather than leaving chilled food indefinitely.
Do not assume a return can be resold; apply the product disposition policy and local rules.

### Step 11 — Monitor temperature and data quality

**Input:** product or environmental measurements, device metadata, event timestamps, and expected sampling schedule.
**Output:** timely alerts and an auditable record that distinguishes compliant, noncompliant, and unknown conditions.

Choose what is measured and why: product temperature, bag air, refrigerator air, or a validated proxy.
Specify sensor identity, unit, location, sampling interval, time zone, accuracy, and calibration status.
Maintain a mapping between sensor, container, order, and time interval.
For manual measurements, record the person, instrument, method, and measurement time.

Check readings for missing timestamps, duplicate events, impossible values, clock drift, and stale telemetry.
Treat data gaps as unknown, not as proof of compliance or proof of failure.
Use a defined maximum valid gap when summarizing a time series.
If an alert arrives late, record both the event time and the alert receipt time.
A dashboard that updates after the order has been delivered may support investigation but cannot prevent that delivery.

Measure coverage as eligible exposures with valid evidence divided by eligible exposures, with the definition fixed in advance.
Stratify coverage by store, time, route, bag, and relevant product group.
Report the number of unknown intervals beside any compliance percentage.
Investigate whether missingness is concentrated during equipment failures, peak load, or particular courier groups.
The data owner maintains lineage and alert delivery; food safety decides how evidence informs disposition.

### Step 12 — Respond to an excursion or missing evidence

**Input:** alert or report, affected product identity, condition history, and the approved incident procedure.
**Output:** containment, documented assessment, disposition, customer or authority action where applicable, and corrective action.

First protect people and isolate the affected product or route.
Pause release of potentially affected inventory and prevent an automated system from reassigning it.
If product is in transit, contact the courier using a safe communication channel and provide a specific instruction.
Do not ask a courier to read or respond while operating a vehicle.

Record the first known abnormal or unknown time and the last verified acceptable time.
Preserve the raw sensor readings, equipment state, event logs, lot identifiers, and worker observations.
Do not overwrite raw evidence with a corrected estimate.
Identify all potentially affected product, including orders already delivered and stock sharing the same equipment.
A successful customer handoff is not a food-safety clearance.

An authorized food-safety decision maker evaluates the product, temperature evidence, time, uncertainty, packaging, and applicable validated guidance.
Where records are incomplete, distinguish what is known from a conservative bound and from an unsupported assumption.
Disposition may include continued hold, disposal, supplier return, or release only when justified by the authorized procedure.
The operator should not independently invent a release threshold.

If product has entered distribution, follow the incident and recall assessment process and applicable notification obligations.
Customer communication should be accurate about what is known and what action is requested.
Document the responsible authority, timestamps, scope, customer contacts, and outcomes.
Separately correct the equipment or process and verify the correction before restart.
A replacement device can restore capability; it does not automatically resolve product already exposed.

### Step 13 — Build courier welfare into normal operations

**Input:** worker agreements, applicable local requirements, route and pay data, incident reports, and worker feedback.
**Output:** a policy and an executable set of safeguards with accountable owners.

Define what time is paid and which costs the worker bears.
Measure offer-to-acceptance, waiting at collection, active travel, handoff, and relevant return or repositioning time separately.
Compare gross and estimated net earnings using consistent time windows.
Do not infer the applicable legal definition of working time from a metric label.

Offer routes that can be completed at lawful, reasonable speeds and with safe load handling.
Include collection delays, access friction, and realistic handoff time in planning.
Provide accessible information about distance, expected time, pay, load, and material changes, subject to local policy and law.
Design for adverse weather, visibility, traffic, and safe parking or stopping.
Create a process for pausing service or changing the promise when conditions make safe work infeasible.

Provide practical access to drinking water, toilets, rest locations, and emergency contacts where the operating model allows.
Set procedures for fatigue, illness, harassment, collisions, and other incidents without requiring a worker to prove a diagnosis to report concern.
Ensure workers can refuse unsafe work and seek help without retaliation, consistent with applicable law and contractual arrangements.
Audit whether app penalties, automated deactivation, or opaque acceptance metrics undermine this policy.

Consult workers about feasibility and distributional effects.
Use multiple channels, including anonymous reporting where appropriate, and track whether a report receives a response.
Pay attention to workers with different vehicles, shifts, neighborhoods, languages, and accessibility needs.
Protect sensitive information, restrict access, and avoid using safety reports as a covert productivity ranking.
The workforce owner and safety owner jointly maintain safeguards; local management must have authority to act in real time.

### Step 14 — Verify, audit, and improve

**Input:** monitoring records, incidents, calibration records, training, worker feedback, and independent review.
**Output:** evidence that controls operate as designed, with tracked improvements.

Review routine monitoring records for completion, exceptions, and follow-through.
Verify instrument performance and calibration at the specified frequency and after damage or repair.
Observe actual work, including busy shifts and handoffs involving third parties.
Test whether alerts reach an available person and whether that person can stop the affected flow.
Reconcile held inventory against transaction records to identify accidental release.

Analyze repeat incidents by mechanism, not only by the last person who touched the product.
Examine incentives, staffing, packaging, layout, equipment, and data availability.
Separate a correction that handles today's product from corrective action that prevents recurrence.
Assign an owner, deadline, verification method, and evidence for closure.

Trend product holds, exposure evidence coverage, validated route feasibility, injury reports, worker waiting, and complaints together.
A sudden drop in incident reports can indicate improvement or suppressed reporting.
Review both outcomes and leading signals such as worker feedback and alert response times.
Use a governance meeting with operations, safety, workforce, analytics, and finance represented.
Document decisions, unresolved risk, and who can authorize a change or restart.

### Step 15 — Compare feasible business alternatives

**Input:** safety requirements, worker safeguards, baseline and proposed operating designs, and incremental cash flows.
**Output:** an explicitly bounded investment decision and monitoring plan.

First eliminate options that do not meet applicable requirements or approved safety controls.
Then compare the incremental cost of the remaining feasible alternatives over the same time horizon.
Include equipment, depreciation or rental, cleaning, coolant, paid worker time, training, maintenance, and operational overhead as relevant.
Include avoided costs only if they are incremental to the decision and the evidence is credible.

Separate observed direct cash effects from uncertain avoided harm, reputation, or regulatory exposure.
Do not double count a refund in both a customer-loss estimate and a separate compensation line.
Show low and high cases for uncertain inputs and identify what would change the decision.
When the evidence cannot support a probability, provide a break-even analysis rather than inventing a point estimate.
Mandatory baseline controls are not optional simply because a spreadsheet shows a negative net benefit.

Assign an owner for each assumption and update the case after real operating evidence is available.
Measure adoption, coverage, service effects, and worker outcomes alongside cost.
A lower cost per order is not a success if the benefit comes from unpaid waiting or lower control coverage.

## 6. Worked example

### Scenario and product scope

Harbor Market is a fictional dark store selling packaged, chilled ready-to-eat grain bowls.
The calculations are illustrative and are not a food-safety validation or a legal temperature rule.
The food-safety team has approved a specific packaging, starting-temperature, coolant, and ambient-condition configuration for this exercise.
Under that hypothetical approved configuration, each bowl may spend at most 45 minutes from removal from controlled storage to customer handoff.
This 45-minute figure is an assumed internal operating envelope, not a general safe time for chilled food.
If the actual configuration falls outside the validation, the time rule cannot be used as evidence of safety.

The store is comparing a two-order delivery batch with two individual deliveries.
Both bowls are in date and have intact packaging.
The store has lot-to-order traceability for both units.
The dispatcher must check the second customer's timeline, not just the first customer's arrival.
All times below use the same local clock and date.

| Event or assumption | Order A | Order B |
|---|---:|---:|
| Removed from controlled storage | 12:00 | 12:03 |
| Packed and staged in approved configuration | 12:06 | 12:08 |
| Proposed courier handoff | 12:12 | 12:12 |
| Proposed first delivery | 12:25 | — |
| Proposed second delivery | — | 12:38 |
| Customer commitment | 12:30 | 12:40 |
| Hypothetical total exposure allowance | 45 min | 45 min |

The pack and stage events do not reset the exposure clock in this example.
They are inside the approved configuration, but the hypothetical rule counts the entire interval from storage removal to delivery.
Actual plans may use a different validated definition.
The local time arithmetic is straightforward because no time-zone or daylight-saving transition occurs in this scenario.

### Feasibility at dispatch

Order A's planned exposure is 12:25 minus 12:00, or 25 minutes.
Order B's planned exposure is 12:38 minus 12:03, or 35 minutes.
The planned margin to the hypothetical 45-minute envelope is 20 minutes for A and 10 minutes for B.
The customer commitment slack is 5 minutes for A and 2 minutes for B.
The food-safety margin and customer slack are different measures.

The dispatch owner applies an internal planning rule requiring at least 8 minutes of exposure margin for this route's delay uncertainty.
That 8-minute buffer is a scenario assumption, not a claim that it guarantees a particular percentile.
At 12:12, the proposed batch has 20 minutes and 10 minutes of margin, so it passes that particular planning rule.
It also meets both point-estimate customer commitments.
The dispatcher still has to confirm the approved container, courier load, and safe route conditions.
A forecast of on-time arrival is not a guarantee that those conditions will remain true.

Now consider a plausible 6-minute delay before the first delivery that propagates to both stops.
The revised predicted handoffs are 12:31 and 12:44.
Exposure becomes 31 minutes for A and 41 minutes for B.
Both remain below the hypothetical 45-minute envelope at those predictions, but B now has only 4 minutes of exposure margin.
Both customer commitments are missed by 1 and 4 minutes, respectively.
Because the internal planning buffer is no longer met, the dispatch team must re-evaluate the route under the approved operating rule.
It cannot conclude that a customer refund restores the required safety margin.

A further 5-minute delay would put B at 12:49, or 46 minutes from storage removal.
That exceeds the hypothetical operating envelope by 1 minute.
The approved response is to stop or redirect the affected flow and place the product under food-safety review, not to guess that one extra minute is harmless.
Whether a particular event constitutes a CCP deviation depends on the actual approved HACCP plan; this example only establishes an internal envelope breach.

### A temperature alert and missing evidence

A logger in B's bag records the following bag-air observations after handoff.
For the exercise, an internal alert threshold is strictly greater than 8.0 °C.
That threshold is hypothetical, is not a product-core measurement, and is not a legal threshold or a critical limit.
The logger records every 5 minutes when functioning.
The analytics team accepts an interval as observed only if adjacent readings are no more than 6 minutes apart.
Each valid reading represents the bag-air condition until the next reading, solely for the illustrative classification below.
This last-observation rule is a computational convention, not proof of the true temperature trajectory.

| Time | Bag-air reading | Treatment of following interval |
|---|---:|---|
| 12:12 | 5.0 °C | 12:12–12:17 observed at or below alert threshold |
| 12:17 | 5.8 °C | 12:17–12:22 observed at or below alert threshold |
| 12:22 | 8.4 °C | 12:22–12:27 observed above alert threshold |
| 12:27 | 9.1 °C | 12:27–12:37 unknown: 10-minute gap exceeds 6 minutes |
| 12:37 | 6.5 °C | Endpoint only; no following interval in this record |

The analyzed record spans 25 minutes from 12:12 through 12:37.
There are 10 minutes classified at or below the alert threshold, 5 minutes classified above it, and 10 minutes unknown.
Valid-interval coverage is 15 divided by 25, or 60%.
The recorded above-threshold share of valid time is 5 divided by 15, or 33.3%.
It is not valid to call the unknown 10 minutes compliant, or to claim that the product was above 8.0 °C for all 10 minutes.
It is also not valid to convert the air reading directly into bowl-core temperature.

At 12:22 the logger sends an alert, received by operations at 12:23.
Operations directs the courier through a safe contact channel to hold B at the next safe stop.
The courier reports an unzipped bag and closes it; the device comes back online at 12:37.
Closing the bag is an immediate correction, not proof that B is fit to release.
A food-safety professional receives the product profile, lot, event times, readings, unknown interval, and bag report.
For this example the professional keeps B on hold because the available evidence does not satisfy the hypothetical release procedure.
B is not delivered to the customer; the customer receives a replacement or refund under the service policy.
The disposition of the held unit is recorded separately from the refund.

The 12:37 reading is a logger recovery event, not evidence that all earlier exposure was safe.
The incident owner investigates why the bag was unzipped and why telemetry was absent.
The possible causes include a damaged closure, inadequate training, packing design, and device connectivity.
A root cause cannot be chosen without additional evidence.

### Courier welfare decision

The courier had been offered a two-stop route with an estimated 26 minutes from collection to final handoff.
The offer disclosed the load, both stops, and the compensation for that route under the fictional policy.
After the safety stop, the worker spent 8 additional minutes waiting for instructions.
In the scenario, the policy compensates that additional wait; this is an assumption, not a statement of any particular jurisdiction's law.
The worker is not penalized for stopping safely or reporting the open bag.
Dispatch records the wait and separates it from riding time.

Suppose the original route payment was USD 7.80 and the additional 8 minutes were paid at USD 0.30 per minute.
The additional payment is 8 × USD 0.30 = USD 2.40.
Total courier compensation for this trip is USD 10.20 before any separately applicable reimbursements.
These amounts do not establish legal adequacy or worker net earnings.
They show why an intervention must account for the paid time that a real exception creates.

### Incremental business case

The operation is choosing between two feasible configurations after safety approval.
Option S uses a standard approved bag with a dedicated courier for each order.
Option B uses a validated two-order bag and batch where the route is feasible.
The following numbers are fictional average incremental costs per pair of eligible orders.
They exclude merchandise margin, revenue, taxes, and fixed costs common to both options.

| Cost item | S: two solo deliveries | B: approved batch | Difference B minus S |
|---|---:|---:|---:|
| Courier compensation, including modeled waiting | USD 13.20 | USD 9.80 | −USD 3.40 |
| Bag, coolant, cleaning, and device allocation | USD 0.80 | USD 2.00 | +USD 1.20 |
| Packing and monitoring labor | USD 1.00 | USD 1.40 | +USD 0.40 |
| Additional service recovery allowance | USD 0.40 | USD 0.90 | +USD 0.50 |
| Total per pair | USD 15.40 | USD 14.10 | −USD 1.30 |

The modeled saving is USD 1.30 per eligible pair, or USD 0.65 per order within these pairs.
It is not USD 1.30 per order.
At 1,000 eligible pairs per week, the modeled saving is USD 1,300 per week before any implementation cost not already allocated above.
If 1,000 is the number of orders rather than pairs, the calculation must be changed.

The service recovery allowance is an assumed incremental operating cost, not a measured risk probability.
The calculation includes only costs stated in the table; it does not monetize foodborne illness, worker injury, reputational harm, or compliance.
If the batch has a one-time implementation cost of USD 13,000 and the USD 1,300 weekly saving persists, simple undiscounted payback is 10 weeks.
This payback ignores tax, financing, seasonality, training disruption, and uncertainty.
The scenario should not be deployed merely because this arithmetic is positive.

If actual additional courier waiting raises B's compensation by USD 1.50 per pair, B costs USD 15.60 per pair.
B is then USD 0.20 more expensive than S per pair.
If the feasible batch share is lower than expected, total weekly savings fall even when the per-pair result remains positive.
The team should measure actual paid time, product holds, coverage, and feasible-pair volume before claiming realized savings.
The food-safety and workforce guardrails remain binding in either financial case.

## 7. Code, query, or calculation

The following standard-library Python program reproduces the example's interval classification and business-case arithmetic.
It deliberately does not calculate food-core temperature or determine product disposition.
It uses the left reading to classify a valid interval and treats intervals longer than the configured gap as unknown.
The final reading is an endpoint, not a new interval.
The calculation assumes ordered readings from one sensor on one date and a threshold comparison of strictly greater than 8.0 °C.
It does not extrapolate before the first reading or after the last reading.

```python
from datetime import datetime

# Fictional bag-air readings, not validated product-core temperatures.
readings = [
    ("2026-06-01 12:12", 5.0),
    ("2026-06-01 12:17", 5.8),
    ("2026-06-01 12:22", 8.4),
    ("2026-06-01 12:27", 9.1),
    ("2026-06-01 12:37", 6.5),
]
threshold_c = 8.0
max_gap_minutes = 6.0


def parse_readings(rows):
    parsed = [
        (datetime.strptime(timestamp, "%Y-%m-%d %H:%M"), value)
        for timestamp, value in rows
    ]
    if len(parsed) < 2:
        raise ValueError("At least two readings are needed")
    if any(right[0] <= left[0] for left, right in zip(parsed, parsed[1:])):
        raise ValueError("Timestamps must be strictly increasing")
    return parsed


def summarize_intervals(rows, threshold, max_gap):
    if max_gap <= 0:
        raise ValueError("Maximum gap must be positive")
    totals = {"at_or_below": 0.0, "above": 0.0, "unknown": 0.0}
    for (start, value), (end, _) in zip(rows, rows[1:]):
        minutes = (end - start).total_seconds() / 60.0
        if minutes > max_gap:
            totals["unknown"] += minutes
        elif value > threshold:
            totals["above"] += minutes
        else:
            totals["at_or_below"] += minutes
    return totals


parsed = parse_readings(readings)
minutes = summarize_intervals(parsed, threshold_c, max_gap_minutes)
valid = minutes["at_or_below"] + minutes["above"]
span = valid + minutes["unknown"]
coverage = 100.0 * valid / span if span else 0.0
above_share = 100.0 * minutes["above"] / valid if valid else None

solo_cost = 13.20 + 0.80 + 1.00 + 0.40
batch_cost = 9.80 + 2.00 + 1.40 + 0.90
saving_per_pair = solo_cost - batch_cost
weekly_pairs = 1000
weekly_saving = saving_per_pair * weekly_pairs
implementation_cost = 13000.0
payback_weeks = implementation_cost / weekly_saving

print(f"At or below threshold: {minutes['at_or_below']:.0f} min")
print(f"Above threshold: {minutes['above']:.0f} min")
print(f"Unknown: {minutes['unknown']:.0f} min")
print(f"Valid interval coverage: {coverage:.1f}%")
if above_share is None:
    print("Above-threshold share of valid time: undefined")
else:
    print(f"Above-threshold share of valid time: {above_share:.1f}%")
print(f"Solo cost per pair: USD {solo_cost:.2f}")
print(f"Batch cost per pair: USD {batch_cost:.2f}")
print(f"Saving per pair: USD {saving_per_pair:.2f}")
print(f"Weekly saving: USD {weekly_saving:.2f}")
print(f"Simple payback: {payback_weeks:.1f} weeks")
```

Expected output:

```text
At or below threshold: 10 min
Above threshold: 5 min
Unknown: 10 min
Valid interval coverage: 60.0%
Above-threshold share of valid time: 33.3%
Solo cost per pair: USD 15.40
Batch cost per pair: USD 14.10
Saving per pair: USD 1.30
Weekly saving: USD 1300.00
Simple payback: 10.0 weeks
```

The program is a teaching calculation, not a production monitoring service.
For monetary production calculations, use a suitable decimal representation and a documented rounding policy.
The script does not handle time-zone transitions, device calibration, out-of-range values, duplicate-device reconciliation, or late-arriving events.
Its sensor value is not validated against a food measurement.
The fixed 6-minute gap criterion is an exercise assumption; a real rule needs a documented rationale.
The calculation also does not assign the 12:37 endpoint any duration.

A production implementation would need a product-specific monitoring plan, a system clock and time-zone strategy, quality controls, alert routing, audit logs, and a tested human response.
It would also need an explicit rule for beginning and ending each exposure interval.
The code intentionally does not make an automatic release or disposal decision.

## 8. Data, standards, and regulatory context

### Data needed for operational control

A useful product record identifies the item, product category, storage conditions, expiry, supplier, lot, and applicable control profile.
An order record identifies ordered and substituted items, pack time, customer commitment, and cancellation or delivery status.
A custody record identifies the responsible party, transfer time, and bag or container.
A temperature record identifies the sensor, physical position, timestamp, value, unit, and calibration status.
A worker record should contain only the information needed for the relevant operational and welfare purpose.

A join key must have a known scope and lifetime.
A reusable bag ID alone is insufficient without the interval during which it contained a particular order.
A courier assignment event is not necessarily a custody transfer.
A customer-delivered event is not a measurement of food condition.
Keep raw events immutable and store subsequent corrections as separately attributable changes.

### Standards and legal applicability

The Codex Alimentarius Commission's *General Principles of Food Hygiene*, CXC 1-1969, includes HACCP principles and guidance.
Codex guidance is an international reference, not itself a universal statute automatically binding every delivery operation.
Authorities can incorporate or adapt its principles in law, and the applicable local requirements govern.
Consult the current edition and the rules relevant to the facility and product.

In the United States, the Food and Drug Administration (FDA) Food Code is a model for retail and food service.
The 2022 edition is a reference; state, local, tribal, and territorial adoption and amendments determine what is enforceable in a given location.
Certain operations or processes may have additional legal requirements, including HACCP plan requirements.
Do not infer that all app-based deliveries are governed identically or that a model-code provision has been adopted unchanged.

In the European Union, Regulation (EC) No 852/2004 addresses hygiene of foodstuffs and includes HACCP-based procedures for food business operators, with relevant qualifications and implementing context.
Its temperature and cold-chain provisions must be read in context, together with other applicable product-specific and national requirements.
The identity and obligations of each operator depend on the facts.
The primary legal text and current consolidated provisions should be checked with competent local advisers.

GS1 identification and event standards can help trading partners exchange product and movement information.
A GS1 identifier or Electronic Product Code Information Services (EPCIS) event does not by itself validate product safety or prove a physical handoff.
A business must define data quality, access, event semantics, and party responsibilities in its implementation.
Standardization helps interoperability; it does not supply missing observations.

### Temperature and evidence

The approved food-safety plan must specify which food and measurement technique the limit concerns.
The readings in the example are bag-air observations and should not be compared mechanically with a product-core limit.
Probe accuracy, response time, placement, sampling frequency, and thermal contact affect interpretation.
Calibration records should include reference, date, range, result, and action when outside tolerance.
A device that reports a plausible number can still be wrongly placed or miscalibrated.

Where continuous monitoring is unavailable, specify manual checks and the resulting uncertainty.
A missing check should trigger the plan's exception procedure rather than an invented reading.
Time synchronization matters when reconstructing the order of packing, handoff, alarm, and response.
Retain both device event time and server receipt time where practicable.
Retention periods, personal-data rules, and access controls depend on applicable law and policy.

### Courier and employment context

Courier status, compensation obligations, working-time rules, occupational safety duties, and rights vary by jurisdiction and arrangement.
This unit does not determine whether a courier is an employee, contractor, or another category.
A platform's internal label does not resolve that legal question.
Local counsel and the responsible operations team should review the specific model.

The International Labour Organization's guidance on occupational safety and health provides a general prevention framework.
It does not automatically establish a specific wage, break, or classification rule for the fictional operation.
Worker voice and a usable incident-reporting channel are practical design features regardless of the legal categorization question.
Separate legal compliance checks from internal targets that exceed the legal minimum.

### Minimum evidence schema

| Field | Example | Control purpose | Caution |
|---|---|---|---|
| Product ID | Bowl SKU | Join to approved product profile | A SKU may contain several lots |
| Lot ID | Supplier lot | Bound affected stock | Check capture accuracy |
| Order ID | Customer transaction | Identify delivered units | Does not replace lot identity |
| Bag ID | Reusable container | Link sensor and load | Record start and end of association |
| Sensor ID | Logger serial | Connect readings and calibration | Multiple sensors can share a bag |
| Measurement type | Bag air | Interpret values | Do not relabel as core temperature |
| Event time | 12:22 local or UTC | Reconstruct sequence | Preserve time zone and clock quality |
| Receipt time | 12:23 local or UTC | Measure alert latency | Not the physical event time |
| Exposure start | Removal timestamp | Apply the approved clock | Definition is product-plan specific |
| Custody handoff | Actual transfer | Assign operational ownership | Assignment is not handoff |
| Exception state | Hold | Block further release | Integrate state with fulfillment system |
| Disposition | Authorized outcome | Support traceability | Restrict release authority |
| Wait duration | Minutes after stop | Assess work and response | Define paid-time denominator |
| Worker report | Bag closure issue | Surface control failure | Protect from retaliation and misuse |

The minimum schema is a design aid, not a promise that every field is available or legally permissible in every implementation.
Data minimization and controlled access matter especially for location and worker-level records.

## 9. Industry practice and case evidence

### Case 1 — FDA model retail framework

The FDA publishes the Food Code as a model for jurisdictions regulating retail and food-service operations.
Its structure addresses food protection, equipment, personnel, and management responsibilities.
The 2022 edition also provides a reference framework for temperature control and specific process requirements.
The published code is not proof that a particular state has adopted it, nor that an operation complies.

For a delivery design, the useful lesson is that the food chain starts before the courier receives a bag.
Receiving, storage, handling, employee practices, and management response all matter.
[INFERENCE] A platform that measures only vehicle travel temperature risks missing earlier failures in the store or restaurant.
That inference follows from the multi-stage food flow; it is not a reported FDA estimate of platform performance.

### Case 2 — EU hygiene regulation

Regulation (EC) No 852/2004 places hygiene duties on food business operators within its scope.
It includes provisions concerning maintenance of the cold chain and HACCP-based procedures.
Specific application depends on the type of operation, relevant exceptions, and other applicable law.
The legal text does not validate a particular bag or a delivery-duration target.

[INFERENCE] Operators using multiple contractors need an explicit handoff and responsibility map to demonstrate that no stage falls between owners.
The regulation does not prescribe the fictional responsibility table in this lesson.
The map is an operational design response, not a claim about a specific enforcement action.

### Case 3 — Codex HACCP principles

Codex publishes the *General Principles of Food Hygiene* and its HACCP guidance.
The framework distinguishes hazard analysis, CCP determination, critical limits, monitoring, corrective actions, verification, and documentation.
These concepts provide a common language for designing controls across food businesses.
They do not remove the need for product-specific validation and legal review.

[INFERENCE] A generic list of temperatures copied into an app is insufficient to demonstrate a complete HACCP system.
The inference follows because such a list does not by itself establish hazards, control points, monitoring, responses, or verification.
It is not a finding about a named company's implementation.

### Case 4 — Food delivery and worker conditions

The International Labour Organization's *World Employment and Social Outlook 2021* examines the role of digital labour platforms in transforming work.
It discusses working conditions, earnings, and the operation of platform work across contexts.
Its aggregate analysis does not establish the conditions of every delivery worker or every market.
Methods, time periods, and samples must be checked before transferring any statistic.

[INFERENCE] A delivery platform should measure waiting time and worker-borne costs rather than equating order payment with hourly net income.
That is an operational implication, not a numerical estimate from the report.
Averages should be accompanied by distribution and denominator definitions.
Worker consultation can reveal conditions that a route log cannot observe.

### Case 5 — Heat and outdoor work

The U.S. Occupational Safety and Health Administration (OSHA) publishes heat-exposure guidance and prevention resources for employers and workers.
These materials describe prevention measures such as water, rest, shade, acclimatization, and planning appropriate to conditions.
The guidance is not a blanket statement of all legal obligations in every jurisdiction.

[INFERENCE] Quick-commerce capacity plans should include weather-related stop or slowdown options instead of assuming a constant safe courier speed.
That design recommendation does not establish a universal temperature at which all delivery must stop.
The local risk assessment must account for humidity, work intensity, clothing, acclimatization, and available facilities.

### Reading evidence responsibly

Regulatory sources describe requirements or guidance within stated scope.
An observational worker survey describes respondents and the study design, not automatically the causal effect of one dispatch policy.
A vendor's bag performance claim describes its test conditions, which may not match the operation's food, loading, or ambient exposure.
A company case study may describe a practice without independently establishing its effectiveness.

A credible local evaluation specifies the intervention, comparison, period, population, and outcomes before looking at results.
Document which routes and products are eligible for a safety-approved change.
Measure both intended outcomes and harms that may shift elsewhere in the system.
Stop the change if safety or worker protections fail, regardless of apparent short-term savings.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Wrong product profile | Similar items share an unexplained rule | Distinct hazards and thermal properties are pooled | Inappropriate control | Approve product-specific profiles |
| Unmapped waiting | Pack-to-handoff time grows | Food waits outside the assumed flow | Exposure budget is understated | Walk the process at peak and log handoffs |
| Unowned handoff | Both parties say the other checks the bag | Responsibility ends between organizations | Defect passes unnoticed | Assign check, acceptance, and escalation |
| Sensor beside coolant | Consistently low readings near a cold pack | Probe placement misses warm product | False confidence | Validate representative placement |
| Empty device battery | Missing packets increase | No valid observations arrive | Unknown exposure | Monitor heartbeat and have manual fallback |
| Clock drift | Sensor time precedes packing | Device and server clocks differ | Wrong reconstruction | Synchronize clocks and preserve receipt time |
| Alert fatigue | Alerts are routinely closed without action | Excessive false positives or unclear ownership | Real excursion missed | Tune with evidence and audit response |
| Door alarm muted | Alarm clears without a temperature review | Staff reset equipment before containment | Affected stock remains sellable | Require incident and hold workflow |
| Stock hold bypass | Held lots appear in pickable inventory | Inventory and safety systems are not integrated | Suspect food is shipped | Block release and reconcile transactions |
| Overloaded bag | Zipper strain or crushed packaging | Volume and weight exceed design | Warming, leakage, or injury | Enforce validated capacity limits |
| Mixed raw and ready-to-eat food | Leaks or unseparated totes | Cross-contamination at pack or transport | Foodborne illness | Segregate and use appropriate barriers |
| Unverified return | Returned food re-enters inventory | Custody and condition are unknown | Unsafe resale | Apply authorized return disposition |
| Delay hidden by batching | Last-stop exposure rises | Extra stops consume uncounted time | Envelope breach | Evaluate each order's route timeline |
| False compliance from missing data | Compliance improves as coverage falls | Unknown intervals are omitted | Risk is concealed | Report coverage and unknowns |
| Overconfident interpolation | A smooth line spans a long outage | Model assumption is presented as observation | Wrong disposition | Preserve gaps and assess bounds |
| Incorrect limit label | Dashboard alert called a CCP limit | Operational threshold is confused with HACCP limit | Incorrect escalation and audit evidence | Label and govern each threshold |
| No paid waiting | Workers report rushing through checks | Safety activity is economically penalized | Unsafe conduct and unfair cost shifting | Define compensation and monitor wait |
| Unsafe route incentives | Frequent late penalties and unrealistic ETAs | Workers are pressured to speed | Collision or injury | Realistic routing and non-retaliatory escalation |
| Suppressed reports | Near misses fall while complaints rise | Workers fear deactivation | Hazards remain hidden | Independent reporting and response audit |
| Unusable support | Long contact times during an incident | Worker cannot obtain safe instructions | Delayed containment | Staff and test a live escalation channel |
| Inadequate facilities | Workers report no water or toilets | Site access was omitted from design | Heat stress and poor welfare | Provide practical access and feedback |
| Averaged earnings | High mean with low lower-tail outcomes | Waiting and costs vary across workers | Distributional harm hidden | Report time-consistent distributions |
| Misleading business case | Only equipment spend is counted | Labor, cleaning, or wait cost omitted | Unfunded operation | Compare complete incremental costs |
| Invalid validation transfer | New bag adopted without retest | Thermal performance changes | Control envelope no longer applies | Review and revalidate changes |
| Premature restart | Equipment repaired but stock not assessed | Device correction confused with product disposition | Affected product released | Separate restart and disposition approvals |
| Blame-only investigation | Repeated incidents assigned to individuals | System incentives and design ignored | Recurrence and reduced reporting | Analyze workload, process, and equipment |

Review the failure table as a linked system rather than a menu of independent risks.
A missed courier handoff can create both an exposure and unpaid waiting.
A broken sensor can reduce apparent incident counts while worsening actual detection.
A lower average delivery time can coexist with a worse last-stop tail.
The correct response addresses the causal mechanism and checks whether the intervention creates a new hazard.

## 11. Assumptions and limits

| Assumption | Why it is needed | What breaks it | How to test or challenge it |
|---|---|---|---|
| Product identity and lot are recorded | Bounds affected units | Missing or wrong scans | Reconcile stock, orders, and supplier records |
| The product profile matches the actual item | Selects applicable controls | Substitution or formulation change | Audit SKU-to-profile mappings |
| The hypothetical 45-minute envelope is approved | Enables example feasibility arithmetic | Changed bag, load, product, or ambient range | Review actual validation evidence |
| Storage-removal time is accurate | Starts the example exposure clock | Manual entry delay or clock drift | Compare observed and recorded events |
| Bag-air logger represents only bag air | Avoids unsupported core claims | Sensor is mislabeled as food temperature | Inspect placement and metadata |
| A 6-minute gap is a valid analytical cutoff | Classifies example intervals | Device behavior or monitoring plan differs | Review sampling and response requirements |
| Left-value hold is used for valid intervals | Makes calculation reproducible | Conditions fluctuate between readings | Compare with higher-frequency observations |
| Unknown time stays unknown | Avoids invented evidence | Dashboard fills missing values silently | Inspect raw-versus-derived records |
| Route predictions are available at decision time | Supports forward-looking dispatch | Retrospective actuals are used as predictions | Freeze and audit decision-time data |
| A delay affects both stops as specified | Supports scenario calculation | Reordering or traffic changes propagate differently | Recompute actual route scenarios |
| Courier waits are paid in the fictional policy | Supports compensation arithmetic | Agreement or implementation differs | Audit pay and worker statements |
| Eligible pairs are counted once | Converts per-pair savings to weekly savings | Order count is confused with pair count | Reconcile eligible orders and trips |
| All table costs are incremental | Makes the options comparable | Common or duplicated costs are included | Review cost ownership and counterfactual |
| Safety and welfare constraints remain binding | Prevents cost from buying away protection | Optimizer permits unsafe exceptions | Test rejection and stop-work controls |
| Reported incidents reflect detection as well as occurrence | Prevents simplistic trend claims | Reporting changes after a policy change | Compare feedback, coverage, and audit evidence |
| An authorized person can be reached | Makes incident response executable | No out-of-hours coverage | Test escalation at representative times |
| Customers receive accurate communication | Supports incident containment | Contact data or message process fails | Test contact and delivery acknowledgement |
| Rules are current and locally applicable | Avoids misplaced legal conclusions | Law, adoption, or product scope changes | Obtain periodic local legal and safety review |

The numerical scenario is deliberately narrow.
It does not estimate the risk of foodborne illness from a reading or derive a product's safe life.
The business case is a comparison of specified costs, not a valuation of health or worker safety.
A real operation must validate its own products and processes and obtain applicable legal advice.

## 12. Exercises

### Exercise 1 — Flow and ownership map

**Time:** 20 minutes.
**Deliverable:** a one-page flow and responsibility table for a chilled ready-to-eat meal sold by a restaurant through a third-party courier platform.

Include preparation, packing, staging, courier handoff, transport, customer handoff, failed delivery, and incident escalation.
Identify at least one information handoff and one physical handoff.
Name a person or role who can hold product and a role who can authorize release.

**Acceptance criteria:** every stage has an owner; the product and evidence paths are distinct; unknown ownership is flagged rather than assumed.
The map must include a branch for a courier reporting a damaged bag.
It must not treat the delivery platform's customer refund as product disposition.

### Exercise 2 — Hazards and controls

**Time:** 25 minutes.
**Deliverable:** a hazard-analysis excerpt for picking and packing the meal, with three hazards and controls.

Include one biological, one chemical or allergen-related, and one physical hazard.
Identify a prerequisite program and a possible monitoring point.
Explain what evidence and qualified decision process would be needed before labeling that point a CCP.

**Acceptance criteria:** the answer links each hazard to a mechanism and control; it does not designate a CCP solely because a thermometer is present.
It distinguishes a critical limit from an operational alert threshold.

### Exercise 3 — Route exposure arithmetic

**Time:** 15 minutes.
**Deliverable:** a table of exposure and remaining margins for two orders.

Order C leaves controlled storage at 14:02 and is planned for delivery at 14:31.
Order D leaves controlled storage at 14:07 and is planned for delivery at 14:45.
For this exercise only, both orders have an approved 45-minute storage-removal-to-handoff envelope.
The dispatch planning rule requires at least 8 minutes of remaining margin.

Calculate each planned exposure and margin.
Decide whether the proposed route meets this internal planning rule.
Then add 4 minutes to both delivery times and recalculate.

**Acceptance criteria:** report minutes for each order in both scenarios and identify the order that fails the planning buffer after delay.
Do not interpret the hypothetical 45-minute envelope as a general food-safety rule.

### Exercise 4 — Monitoring and missingness

**Time:** 25 minutes.
**Deliverable:** a completed interval table and a short explanation.

At 15:00, 15:05, 15:10, and 15:20, a bag-air logger reads 4.5, 8.5, 9.0, and 5.0 °C, respectively.
Use the same left-reading convention as the code, a threshold of strictly greater than 8.0 °C, and a maximum valid gap of 6 minutes.
Calculate at-or-below, above, and unknown minutes, plus valid-time coverage.
Explain why the result does not establish the temperature of the food.

**Acceptance criteria:** the last 10-minute interval is unknown; the final reading contributes no additional interval.
A modified code input should produce the same totals as the hand calculation.

### Exercise 5 — Incident and disposition

**Time:** 20 minutes.
**Deliverable:** an incident record for a bag-air alert followed by a missing 12-minute sensor interval.

Record the last known acceptable observation, first alert, order and lot identities, containment, evidence preserved, and escalation owner.
Assume the courier is riding when the alert arrives.
Describe a safe contact method and the actions pending food-safety review.

**Acceptance criteria:** the record does not fabricate product temperature, duration above a limit, or an automatic release decision.
It separates product disposition from a refund and identifies how to handle potentially delivered units.

### Exercise 6 — Welfare and dispatch design

**Time:** 25 minutes.
**Deliverable:** a one-page policy for a severe-rain demand peak.

Specify a safe-work escalation or stop rule, offer information, realistic time allowance, wait compensation policy, facilities, and a way to report without retaliation.
Identify a person with authority to slow or pause service.
Explain how you would review worker effects across vehicle types and shifts.

**Acceptance criteria:** no route depends on unsafe speed; a worker can reach a human; service and compensation effects are explicit.
The answer does not assert a universal legal classification or wage rule.

### Exercise 7 — Incremental economics

**Time:** 20 minutes.
**Deliverable:** a cost table and short decision memo.

A compliant solo option costs USD 16.20 per pair.
A compliant batch option costs USD 14.70 per pair before an additional paid-wait allowance of USD 0.80 per pair.
There are 600 eligible pairs per week and an implementation cost of USD 9,450.
Compute incremental saving per pair, weekly saving, and simple undiscounted payback.

**Acceptance criteria:** include paid waiting, use pairs consistently, and show the formula.
State at least two excluded or uncertain effects and explain why safety approval remains a prerequisite.

### Exercise 8 — Evaluation and governance

**Time:** 30 minutes.
**Deliverable:** an evaluation plan for a new validated bag and revised dispatch rule.

Specify eligibility, comparison group or staggered rollout, primary operational outcomes, evidence coverage, worker outcomes, and stop criteria.
Identify confounders including weather and demand, and say how you will address incomplete sensor data.
Name the decision owners for product safety, worker welfare, data quality, and operational rollout.

**Acceptance criteria:** distinguish planned from observed outcomes; avoid interpreting missing observations as compliant.
Report both the average and distribution of waiting or earnings, and predefine a pause when a hard safeguard fails.
Include a plan for worker feedback and a review of whether complaints are being suppressed.

## 13. Self-check answers

### Exercise 1 — Suggested approach

The restaurant owns preparation, approved packing, and the physical staging conditions it controls.
The courier accepts custody only after the correct bags and condition have been checked under the agreed procedure.
The platform owns assignment, support, and the information it sends, subject to the actual agreements and applicable law.
The roles in an actual deployment can differ; legal accountability cannot be inferred from this educational allocation.

A successful answer shows the food traveling through the flow separately from order status and sensor evidence.
For a damaged bag, the courier stops at a safe location and contacts support.
Operations contains the affected order, and the authorized food-safety role decides disposition using available evidence.
A refund is recorded as a customer-service action and does not establish the fate of the food.

### Exercise 2 — Suggested approach

A biological hazard is pathogen growth when chilled ready-to-eat food waits too long in a warm staging area.
An appropriate control may include validated time and temperature conditions and a monitored staging procedure.
An allergen hazard is cross-contact from a leaking or incorrectly packed neighboring item.
Segregation, packaging integrity, correct labels, and cleaning are candidate controls.
A physical hazard is a fragment of damaged packaging entering or contaminating food.
Receiving and packing inspections can help detect damaged packages.

Supplier approval, cleaning, and hygiene can be prerequisite programs.
A staging measurement can be an operational check, but its status as a CCP depends on the hazard analysis and whether control at that step is essential.
The team needs an appropriate critical limit, monitoring method, corrective action, and validation if it is a CCP.
An early warning can be set more conservatively than the critical limit, but the warning does not automatically become the critical limit.

### Exercise 3 — Calculation

For C, 14:31 minus 14:02 is 29 minutes.
Its margin is 45 minus 29, or 16 minutes.
For D, 14:45 minus 14:07 is 38 minutes.
Its margin is 45 minus 38, or 7 minutes.
D already fails the 8-minute planning buffer even though its point estimate is inside the hypothetical 45-minute envelope.
Therefore, the proposed route does not meet the stated planning rule.

After adding 4 minutes, C is delivered at 14:35 and has 33 minutes of exposure and 12 minutes of margin.
D is delivered at 14:49 and has 42 minutes of exposure and 3 minutes of margin.
D still fails the planning buffer.
No food-safety conclusion follows beyond the stated hypothetical operating envelope.

### Exercise 4 — Calculation

The interval 15:00–15:05 contributes 5 minutes at or below the threshold.
The interval 15:05–15:10 contributes 5 minutes above the threshold.
The interval 15:10–15:20 is 10 minutes long and exceeds the 6-minute maximum gap, so it is unknown.
The final reading at 15:20 is an endpoint and contributes no following duration.
Total duration is 20 minutes, valid duration is 10 minutes, and coverage is 50%.
The above-threshold share of valid time is 5 divided by 10, or 50%.
The recorded values are bag air, so they do not directly establish the food's core temperature.

### Exercise 5 — Rubric

A strong record preserves raw readings and states the unknown period explicitly.
It marks the relevant product and prevents further release while the investigation proceeds.
It contacts the courier only through a channel that permits stopping safely before interaction.
It assigns product disposition to an authorized food-safety person and records the decision and rationale.
It considers delivered units, notifications, and applicable recall procedures without assuming a recall is always required.
A weak record changes the missing interval into an assumed safe period or lets the dispatcher release the food to avoid a refund.

### Exercise 6 — Rubric

A strong policy defines an actionable escalation based on local conditions and assigns a named on-duty authority.
It includes reasonable travel and handoff time, clear offers, a stated waiting-payment policy, and access to support.
It provides facilities and a safe way to pause or refuse work, with protection against retaliation.
It evaluates effects by relevant worker groups without exposing unnecessary personal data.
It consults workers and checks whether app incentives undermine the policy.
A weak answer simply tells couriers to ride carefully while keeping impossible service commitments.

### Exercise 7 — Calculation

The batch cost including waiting is USD 14.70 plus USD 0.80, or USD 15.50 per pair.
The incremental saving is USD 16.20 minus USD 15.50, or USD 0.70 per pair.
At 600 pairs per week, the saving is 600 × USD 0.70 = USD 420 per week.
Simple undiscounted payback is USD 9,450 divided by USD 420 per week, or 22.5 weeks.
The calculation assumes eligible volume and unit costs remain constant and excludes discounting.
A strong memo also considers uncertain service recovery, maintenance, feasible-pair volume, and impacts on worker waiting.
The result compares two assumed compliant choices; it cannot justify operating an unapproved unsafe option.

### Exercise 8 — Rubric

A strong evaluation predefines the eligible population and a credible comparison, such as a staggered rollout with suitable controls.
It records the intervention and control versions and adjusts or stratifies for weather, demand, product mix, store, and time.
It reports eligible orders, sensor coverage, unknown intervals, product holds, route exposure, paid waiting, and worker-reported safety outcomes.
It compares distributions as well as averages and explains any changes in denominator or reporting behavior.
A hard food-safety or worker-safety failure pauses the affected rollout before a benefit claim is made.
Owners are named for the control plan, workforce safeguards, telemetry, and release decisions.
A weak plan counts missing sensor periods as compliant or calls a before-and-after difference causal without addressing confounding.

## 14. Interview questions

### 1. What is the difference between a hazard and an excursion?

A strong answer identifies a hazard as an agent with potential adverse health effect and an excursion as departure from a specified condition.
It explains that a temperature excursion can increase biological risk without proving contamination or illness.
A weak answer treats every late delivery as confirmed unsafe food.

### 2. Is every temperature check a CCP?

A strong answer describes product-specific hazard analysis and the need for an essential control at a step.
It distinguishes prerequisites, operational checks, and CCPs, with appropriate limits and monitoring.
A weak answer labels all thermometers CCPs solely because they measure temperature.

### 3. What would you do when an alert is followed by missing sensor data?

A strong answer contains product, preserves raw data, records the unknown interval, and escalates for authorized disposition.
It distinguishes event time from alert receipt and product temperature from bag air.
A weak answer interpolates a safe value and releases the product without review.

### 4. How would you validate an insulated bag?

A strong answer specifies product, starting temperature, loading, coolant, ambient range, route, sensor placement, and uncertainty.
It tests representative and challenging conditions and controls changes to the configuration.
A weak answer relies only on a vendor claim or one reading next to a cold pack.

### 5. Why can a batch be on time but unsafe?

A strong answer accounts for exposure before dispatch, staging, each stop, and the approved time-temperature envelope.
It distinguishes service commitments from food-safety constraints.
A weak answer assumes that any delivery before the customer promise is safe.

### 6. What should happen after an equipment alarm?

A strong answer isolates affected product, assesses the exposure window, preserves evidence, repairs the cause, and authorizes disposition separately from restart.
It identifies the roles responsible for each step.
A weak answer simply resets the alarm and resumes picking.

### 7. How do you measure courier earnings fairly?

A strong answer defines gross and net earnings, worker-borne costs, paid and unpaid time, time windows, and relevant distributions.
It does not infer legal status from an analytics field.
A weak answer divides one payment by travel time while ignoring collection and waiting.

### 8. How can a promise create safety risk?

A strong answer explains how unrealistic ETAs, penalties, or hidden waiting can pressure unsafe speed or skipped checks.
It proposes feasible route times, stop-work escalation, and safeguards against retaliation.
A weak answer blames the courier without considering system design.

### 9. How would you evaluate a new bag and dispatch rule?

A strong answer uses an approved eligibility envelope, a credible comparison, coverage and unknowns, and worker outcomes.
It distinguishes association from causation and defines stop criteria before rollout.
A weak answer reports only a lower average delivery cost after the change.

### 10. How should finance evaluate safety investments?

A strong answer compares feasible compliant options with complete incremental costs, a consistent counterfactual, and sensitivity analysis.
It uses break-even analysis when incident probabilities are unsupported.
A weak answer asserts that prevention automatically pays for itself or treats required controls as optional.

## 15. Key terms

- **Alert threshold:** an internal trigger for action; see [CCPs and limits](#step-5--determine-ccps-and-critical-limits).
- **Calibration:** comparison of an instrument with a reference; see [monitoring](#step-11--monitor-temperature-and-data-quality).
- **CCP:** a step at which an essential control addresses a significant hazard; see [CCP determination](#step-5--determine-ccps-and-critical-limits).
- **Chain of custody:** documented responsibility and transfers; see [handoff](#step-10--transfer-custody-and-execute-the-trip).
- **Cold chain:** controlled conditions across product movement; see [mental model](#3-the-core-mental-model).
- **Corrective action:** action addressing affected product and loss of control; see [excursion response](#step-12--respond-to-an-excursion-or-missing-evidence).
- **Critical limit:** criterion separating acceptability from unacceptability at a CCP; see [CCPs and limits](#step-5--determine-ccps-and-critical-limits).
- **Custody transfer:** physical handoff and associated record; see [execution](#step-10--transfer-custody-and-execute-the-trip).
- **Disposition:** authorized decision on affected product; see [excursion response](#step-12--respond-to-an-excursion-or-missing-evidence).
- **Excursion:** departure from a specified operating condition; see [worked example](#6-worked-example).
- **Exposure budget:** validated time allowance under defined conditions; see [pack validation](#step-7--validate-the-pack-and-exposure-budget).
- **FEFO:** first-expired, first-out stock rotation; see [receiving and storage](#step-6--design-receiving-and-storage-controls).
- **HACCP:** hazard analysis and critical control point system; see [hazard analysis](#step-4--analyze-hazards-by-product-and-step).
- **Hazard:** agent with potential adverse health effect; see [definitions](#4-definitions-and-vocabulary).
- **Hold:** segregation and blocking of product pending a decision; see [excursion response](#step-12--respond-to-an-excursion-or-missing-evidence).
- **Lot:** identified quantity associated with production or handling; see [data schema](#minimum-evidence-schema).
- **Monitoring:** planned observations used to assess control; see [temperature monitoring](#step-11--monitor-temperature-and-data-quality).
- **Near miss:** event with potential for harm but no recorded harm; see [welfare](#step-13--build-courier-welfare-into-normal-operations).
- **Net earnings:** compensation less specified worker-borne costs; see [welfare](#step-13--build-courier-welfare-into-normal-operations).
- **Operational limit:** working threshold used to trigger action; see [CCPs and limits](#step-5--determine-ccps-and-critical-limits).
- **Prerequisite program:** foundational hygienic condition or activity; see [prerequisites](#step-3--establish-prerequisite-programs).
- **Product temperature:** temperature of food at a specified measurement location; see [temperature evidence](#temperature-and-evidence).
- **Recall:** applicable removal or correction process for distributed product; see [excursion response](#step-12--respond-to-an-excursion-or-missing-evidence).
- **Risk:** probability and severity of an adverse effect; see [definitions](#4-definitions-and-vocabulary).
- **Stop-work process:** operational means of stopping unsafe activity and escalating; see [welfare](#step-13--build-courier-welfare-into-normal-operations).
- **Traceability:** ability to follow product identity and movement; see [data schema](#minimum-evidence-schema).
- **Unknown interval:** period without required condition evidence; see [worked example](#a-temperature-alert-and-missing-evidence).
- **Validation:** evidence that a control can achieve its intended outcome; see [pack validation](#step-7--validate-the-pack-and-exposure-budget).
- **Verification:** activities establishing whether controls operate as intended; see [verification](#step-14--verify-audit-and-improve).
- **Worker voice:** worker input into identifying and resolving operational hazards; see [welfare](#step-13--build-courier-welfare-into-normal-operations).

## 16. Sources

These sources support the concepts in this educational unit.
Check current editions, local adoption, and the applicability of each instrument before using it operationally.
The fictional example's numerical limits and costs are assumptions, not values taken from these sources.

1. Codex Alimentarius Commission, [*General Principles of Food Hygiene*, CXC 1-1969](https://www.fao.org/fao-who-codexalimentarius/codex-texts/codes-of-practice/en/).
   International food-hygiene and HACCP reference; consult the current text and local implementation.
2. U.S. Food and Drug Administration, [*2022 FDA Food Code*](https://www.fda.gov/food/fda-food-code/food-code-2022).
   Model retail food code; adoption and modification are jurisdiction-specific.
3. European Parliament and Council, [Regulation (EC) No 852/2004 on the hygiene of foodstuffs](https://eur-lex.europa.eu/eli/reg/2004/852/oj).
   Primary EU legal text; consult current consolidated law and relevant national context.
4. U.S. Food and Drug Administration, [Food Safety Modernization Act: Sanitary Transportation of Human and Animal Food](https://www.fda.gov/food/food-safety-modernization-act-fsma/fsma-final-rule-sanitary-transportation-human-and-animal-food).
   Describes a U.S. transportation rule with defined scope, exemptions, and responsibilities; applicability must be checked.
5. GS1, [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications).
   Identification and barcode specifications, not proof of a product's physical condition.
6. GS1, [EPCIS and Core Business Vocabulary](https://www.gs1.org/standards/epcis).
   Event-data interoperability; implementation and business agreements define actual event use.
7. International Labour Organization, [*World Employment and Social Outlook 2021: The role of digital labour platforms in transforming the world of work*](https://www.ilo.org/publications/flagship-reports/world-employment-and-social-outlook-2021-role-digital-labour-platforms).
   Research on platform work and conditions; consult its methods and population before transferring findings.
8. U.S. Occupational Safety and Health Administration, [Heat](https://www.osha.gov/heat-exposure).
   Heat-exposure prevention resources; applicable duties depend on the jurisdiction and facts.
9. International Labour Organization, [Occupational safety and health](https://www.ilo.org/topics/safety-and-health-work).
   General occupational safety and health resources, not a jurisdiction-specific ruling on courier status.
10. World Health Organization, [Food safety](https://www.who.int/news-room/fact-sheets/detail/food-safety).
    Public-health overview of foodborne hazards and prevention; not a delivery-specific validation protocol.

## 17. Links

- [Module 22 index](_index.md).
- [Promise tiers and dark stores](01_promise-tiers-dark-stores.md).
- [Replenishment, shelf life, and waste](04_replenishment-waste.md).
- [Courier supply and demand](05_courier-supply-demand.md).
- [ETA prediction](06_eta-prediction.md).
- [Batching and routing](07_batching-routing.md).
- [Module 22 survey and lab sections](../../../course/22_quick_commerce_on_demand_delivery.md).
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md).
