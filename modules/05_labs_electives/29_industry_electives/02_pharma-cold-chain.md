# Pharma and Cold Chain: Quality, Traceability, and Scarce Supply

> Module 29 — Industry Electives: Sector Primers  
> Topic 02 of 08  
> Estimated study time: 4 hours  
> Prerequisites: Modules 14, 15, and 28; basic inventory arithmetic  
> Version 2026.5

## 1. Why this topic exists

A hospital pharmacist needs enough medicine for patients this week.
A distribution planner sees boxes at a regional depot.
A temperature alarm has just arrived for one truck.
The boxes may be physically present but cannot simply be treated as available stock.
A quality decision, a traceability reconciliation, and a realistic arrival time determine whether they can be used.
The planner must find a feasible supply plan while an authorized quality professional assesses the evidence.
The quality professional must make an independent disposition without pressure to meet the shipment target.

In pharmaceuticals, the cost of a wrong decision is not limited to a missed forecast or a spoiled pallet.
A product that has lost quality may harm a patient.
A good product mistakenly destroyed may worsen a shortage.
An untraceable package can prevent an investigation or a lawful transaction.
A forecast that uses quarantined stock as available can create a false promise to a clinic.
A route model that ignores remaining shelf life can deliver unusable stock efficiently.

The decision buyer varies by use case.
The supply-chain director buys better allocation and resilience decisions.
The responsible quality function owns product disposition and quality-system controls.
Distribution operations owns monitoring, carrier coordination, and physical segregation.
Regulatory and compliance staff interpret the applicable product and market requirements.
Hospital pharmacy and public-health buyers care about usable arrival, authorized presentation, and continuity of care.
Finance needs visibility into inventory at risk, but book value is not a substitute for a patient-safety assessment.

The sector is not one homogeneous cold chain.
Some medicines are stable at controlled room temperature; others require refrigeration, freezing, or ultra-low temperatures.
A product's approved labeling and qualified shipping configuration determine its controls.
Different countries, product classes, trading partners, and study protocols bring different requirements.
A clinical trial's blinding and randomization constraints differ from commercial distribution.
A national immunization program may have an ethical allocation mandate that differs from a commercial wholesaler's service objective.

This chapter follows one synthetic product through a two-echelon network.
It separates the quality gate from the allocation calculation.
It covers good distribution practice (GDP), serialized identity, temperature records, shortage allocation, and governance.
Its numeric examples are instructional, not clinical stability evidence or a real manufacturer's operating data.
Its code implements an allocation policy, not a product-release decision.

> **Note:** Educational material is not legal, medical, or quality-release advice.
> Consult current local rules, product labeling, approved procedures, and qualified personnel before making an operational decision.
> Regulatory information below is characterized as of November 2026; dates and enforcement policies must be checked again for the transaction at issue.

## 2. Learning outcomes

After this topic, you can:

1. **Map** the decision, buyer, network, and independent quality authority for a medicine shipment; see Exercise 1.
2. **Distinguish** a temperature alert, an excursion investigation, and an authorized disposition; see Exercises 2 and 6.
3. **Reconcile** package identity and traceability events without equating a barcode scan with product authentication or release; see Exercise 3.
4. **Calculate** usable inventory and a reproducible, constrained allocation under scarcity; see Exercise 4.
5. **Compare** allocation policies using service, priority, waste, and access consequences; see Exercises 5 and 8.
6. **Run and challenge** the Python lab, including a change of status or a data gap; see Exercise 7.
7. **Describe** the scope and limits of GDP, GS1 standards, U.S. drug tracing, and EU verification; see Exercise 6.
8. **Write** a decision brief with owners, evidence, assumptions, and escalation triggers; see Exercise 8.

The intended output is a defensible decision and a controlled record, not a claim that a model decides patient treatment.
For every numerical answer, give the unit, time horizon, and relevant eligibility assumption.
For every proposed automation, specify who can stop it and what evidence is retained.

## 3. The core mental model

```text
Product authorization + approved controls + customer need
                        |
          Manufacturing and QA batch release
                        |
       Qualified pack-out and shipment assignment
                        |
  Serialized identity + custody events + temperature evidence
                        |
         Detect alert -> contain -> investigate
                        |
          Quality disposition, with audit trail
                        |
  Authorized usable stock + validated traceability + shelf life
                        |
 Forecast + need + site capacity + allocation policy
                        |
   Feasible allocation -> approval -> dispatch -> receipt
                        |
  Reconcile receipt, monitor outcomes, correct and learn
```

Follow the physical product and the evidence separately.
A shipment may reach its destination even though its temperature record is missing.
A perfect event message may describe a shipment that did not actually arrive.
The two streams must be matched to the same product, shipment, location, and time window.
Disagreement is an exception requiring investigation, not an invitation to choose the convenient source.

There are several gates, not one universal release flag.
The manufacturer releases a batch through its quality system.
A distribution incident can subsequently place some or all affected stock on hold.
A later authorized disposition resolves that hold for a defined quantity and scope.
A traceability exception can separately prevent a transaction even when no temperature problem is known.
An allocation approval does not override either gate.

Represent each batch-location quantity in mutually exclusive states.
A useful planning partition is available, quality hold, traceability hold, expired, and already committed.
When a unit has multiple issues, maintain separate underlying flags and define a deterministic reporting precedence.
That avoids both double counting and hiding the second unresolved issue.
Use the underlying flags, not just the reporting category, for eligibility.
For example, release of a quality hold does not resolve a missing serial event.

The flow is dynamic.
A forecast is an uncertain estimate of future demand.
A plan is a proposed distribution of supply.
A target is an intended performance level.
A commitment is an authorized promise to a recipient.
The commitment must be revised if qualified supply, site capability, or delivery assumptions change.

Scarcity adds a policy choice.
The model can apply an agreed priority rule or solve a constrained optimization problem.
It cannot discover the ethically correct policy from shipment data alone.
The policy must specify the population or sites served, which needs qualify, and how ties and exceptions are handled.
An allocation that maximizes boxes delivered might disadvantage sites with lower cold-storage capacity or worse transport access.
Measure these effects rather than assuming that equal quantities mean equitable access.

### Three clocks

The physical clock measures when exposure, movement, and receipt occur.
The information clock measures when an event was recorded and became available for the decision.
The governance clock measures when a qualified person reviewed and approved a change of status.
A late sensor upload can change what is known after an allocation has already been approved.
A later quality decision must not be backdated to make the earlier plan appear informed.
Store event time, ingestion time, decision time, and record revisions separately.

### The unit of decision

A planner may allocate in packs, vials, doses, or patients served.
Those units are not interchangeable without a product-specific conversion and an operational assumption.
A vial's nominal dose count does not guarantee that every dose is administered.
A serial identifier generally applies at a defined trade-item packaging level, not to an abstract patient.
A batch-level hold can affect multiple cases and sites; a sensor alarm might affect only a subset.
Define the affected scope with evidence before changing inventory states.

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Active pharmaceutical ingredient | Substance intended to furnish pharmacological activity; product-specific mass units | Treating ingredient provenance as finished-pack traceability |
| Batch or lot | Defined quantity produced under a common process and identified for traceability | Assuming every pack in a batch has the same shipping history |
| Batch release | Authorized confirmation under the applicable quality system that a batch can enter the next permitted stage | Assuming later shipping incidents cannot change status |
| Disposition | Authorized decision about identified affected material, such as release, continued hold, return, or destruction | Letting an allocation model silently decide release |
| GDP | Good distribution practice; quality-system controls for distribution, with requirements depending on jurisdiction | Treating one region's GDP text as universally binding |
| GMP | Good manufacturing practice, concerned with manufacturing quality systems and controls | Assuming GMP release guarantees subsequent distribution conditions |
| Temperature excursion | Exposure outside a specified condition or threshold, assessed against the applicable procedure | Assuming every excursion automatically destroys product |
| Alert | Monitoring-system signal that a defined condition or rule was met | Equating an alert with a verified excursion or a release decision |
| Hold or quarantine | Controlled status restricting movement or use pending a decision | Counting held units as ordinary available inventory |
| Stability data | Evidence relating quality to conditions and time for a particular product and configuration | Applying another product's allowed exposure to the shipment |
| Labeled storage condition | Approved product-specific storage requirement, usually expressed in °C | Treating a generic 2–8 °C range as applicable to all products |
| Transit qualification | Evidence that a specific shipping configuration can maintain required conditions under defined challenges | Assuming qualification covers every route, season, or packing variation |
| Mapping | Characterization of temperature distribution across a space or configuration | Assuming one logger is automatically representative |
| Calibration | Comparison of an instrument against a traceable reference with recorded result and uncertainty | Treating a calibration certificate as proof every subsequent reading is correct |
| Sampling interval | Time between recorded measurements, in minutes or seconds | Assuming every peak between readings is observed |
| Data logger | Instrument recording temperature and possibly other conditions over time | Assuming its sensor was inside every product pack |
| Mean kinetic temperature | A temperature summary intended to reflect non-linear thermal effects | Using one summary as an automatic disposition rule |
| FEFO | First-expiry-first-out inventory rotation | Allocating an expired or held batch merely because it expires first |
| Remaining shelf life | Time between the relevant delivery or use date and expiry, in days | Calculating it at dispatch when the policy specifies receipt |
| GTIN | GS1 Global Trade Item Number identifying a trade-item class and packaging level | Treating a GTIN as a unique package serial |
| Serial number | Identifier distinguishing an individual instance within its product-identifier context | Assuming a serial alone is globally unique |
| Lot number | Manufacturer-assigned batch identifier, represented as text | Assuming lot identifies an individual pack |
| GLN | GS1 Global Location Number identifying a party or location in the relevant context | Treating a GLN as a physical sensor coordinate |
| SSCC | GS1 Serial Shipping Container Code identifying a logistics unit | Assuming a pallet identifier is the serial of every contained pack |
| Aggregation | Recorded parent-child relationship between logistics or packaging units | Assuming the contents stayed unchanged after opening or repacking |
| EPCIS | GS1 Electronic Product Code Information Services standard for sharing event data | Confusing an event format with legal compliance or verified physical truth |
| Custody event | Record of a handoff, observation, or status change involving identified objects | Assuming every scan proves continuous custody |
| DSCSA | U.S. Drug Supply Chain Security Act and its implementing framework | Extending its requirements without checking product, entity, and transaction scope |
| FMD | EU Falsified Medicines Directive framework and associated safety-feature rules | Treating EU verification and U.S. transaction tracing as identical |
| Qualified trading partner | Partner satisfying applicable legal and internal qualification requirements | Assuming a commercial master-data record proves qualification |
| Demand forecast | Estimate of future requested quantity over a stated period | Calling a shipment allocation a demand forecast |
| Unmet demand | Eligible requested quantity not fulfilled in the stated horizon | Equating all orders with clinically validated patient need |
| Allocation | Assignment of scarce eligible supply to recipients under a stated rule | Treating the proposed allocation as an irrevocable commitment |
| Fill rate | Fulfilled quantity divided by eligible requested quantity for a defined period | Averaging site percentages without naming the weighting |
| CAPA | Corrective and preventive action within a quality system | Closing an investigation merely because a dashboard changed |
| Audit trail | Attributable history of relevant changes and actions | Keeping only the latest status without who changed it or why |

The table defines operational terms for this chapter.
Legal definitions and exact record fields must be checked against the applicable authoritative source.

## 5. The mechanics, step by step

### Step 1 — Frame the decision and authority

**Input:** product, market, recipients, decision deadline, operating horizon, and known constraints.
**Output:** a decision charter naming the accountable owner and the permitted choices.

1. Name the actionable unit: a shipment, a batch-location quantity, a serialized pack, or a site allocation.
2. State the deadline in a time zone and distinguish event time from the time at which evidence is available.
3. Identify who owns supply planning and who owns quality disposition.
4. Identify legal, regulatory, and product-specific conditions that constrain the decision.
5. Define the population represented by each request, without collecting unnecessary patient-level information.
6. Record which choices are reversible and which require escalation before commitment.
7. Agree the operating horizon, such as the next seven calendar days.

The owner of a commercial service target should not be able to change a quality status to improve that target.
A quality decision may depend on the actual excursion, configuration, stability information, and investigation.
The supply planner can prepare scenarios for several possible outcomes without anticipating the quality decision.

**Branch:** if authority or product requirements are unknown, keep the affected quantity in a controlled state and escalate.
Do not fill the gap with a generic temperature limit or an assumed release.

### Step 2 — Map product, network, and handoffs

**Input:** approved product master, routing information, supplier and customer master, and shipping configuration.
**Output:** a versioned map of nodes, lanes, custody changes, and monitoring coverage.

1. Identify manufacturer, packaging site, central depot, regional depot, carrier, wholesaler, and recipient as applicable.
2. Map the route actually used, not just the planned route.
3. Record the handoff points where responsibility and monitoring change.
4. Assign location identifiers and the time zone of each operational event.
5. Identify the required storage condition for the specific presentation and lifecycle stage.
6. Record pack-out configuration, logger placement, coolant, and qualified duration.
7. Capture recipient capabilities, including receiving hours and storage capacity.
8. Identify outsourcing agreements and the responsibilities for deviation reporting and records.

A depot can be commercially open but physically unable to accept the delivery.
A regional refrigerator can have 200-pack nominal capacity but only 120-pack free capacity.
Use free qualified capacity at the planned arrival time, not the equipment nameplate value.
A recipient must also have authorization and sufficient remaining shelf life under the applicable rules and agreements.

**Branch:** if the actual shipment differs from the qualified configuration, the lane cannot simply inherit the old qualification.
Record the difference and route it to the responsible quality process.

### Step 3 — Establish master data and identifiers

**Input:** product master, packaging hierarchy, approved customer master, and supplier identifiers.
**Output:** a reconciled key map connecting trade item, lot, serial, logistics unit, shipment, and location.

A GTIN identifies a trade-item configuration.
A serial identifies a specific instance in its relevant product context.
The lot and expiry connect a pack to batch and dating information.
The logistics unit identifier relates a case or pallet to a movement.
An aggregation record connects parent and child identifiers at a point in time.

For each identifier, document its issuer, format, source, and effective dates.
Validate that the packaging level agrees with the unit expected by the allocation system.
Check that repacking, split cases, returns, and commissioning or decommissioning events update the hierarchy.
A shipment may contain multiple batches; a batch may span multiple shipments.
Use explicit many-to-many associations rather than assuming a single batch per truck.

**Branch:** if the serial-to-shipment relationship is missing, identify the potentially affected scope conservatively.
A missing link is not proof that the package is counterfeit, nor proof that it is safe.
Escalate under the organization's traceability and suspect-product procedures.

### Step 4 — Ingest temperature and context

**Input:** logger records, device metadata, shipment events, environmental context, and product condition.
**Output:** an immutable source record and a time-aligned, quality-checked analytical view.

1. Capture readings with the original timestamps, units, device ID, and calibration status.
2. Record the measurement interval and any device-reported gaps or battery events.
3. Retain the original file and record its provenance.
4. Convert time zones only in a derived view; preserve the original timestamps and offset.
5. Convert temperature units explicitly and retain the original unit.
6. Associate the logger with its package or qualified load position and deployment period.
7. Cross-check arrival, loading, unloading, and handoff timestamps.
8. Flag duplicate timestamps, implausible jumps, out-of-order readings, and unobserved intervals.
9. Link readings to the actual product and shipping configuration.

The device measures at a sensor, not at every point inside every package.
An external air sensor is not automatically equivalent to a product-core measurement.
A five-minute interval may miss a shorter peak.
The absence of a recorded excursion is weaker evidence when the logger has a gap.
A temperature limit requires an inclusive or exclusive boundary and an approved rule for duration.

**Branch:** if readings are missing, preserve the affected stock and evidence for review.
Do not forward-fill a long gap and present the result as measured temperature.
Do not silently delete a spike because it makes the route look bad.

### Step 5 — Detect, contain, and document an exception

**Input:** observations and approved monitoring thresholds.
**Output:** a deviation record, controlled stock scope, notification, and preserved evidence.

A monitoring rule may create an alert when a reading exceeds an operational threshold.
The first responder checks the sensor, timestamp, affected shipment, and immediate containment instructions.
The responder identifies physical units that might be affected and prevents unauthorized movement.
Record whether the stock is in a vehicle, a warehouse, or already received by a customer.
Notify the functions and partners identified by the applicable procedure.
Preserve original readings, shipping documents, pack-out records, and a chronology of actions.
Record uncertainty in the beginning and end of the event.

Do not assume that the first out-of-range reading is the exact start of exposure.
If the previous reading was in range, the start lies somewhere in the interval between readings unless additional evidence narrows it.
Similarly, the return to range may have occurred before the first in-range reading.
When the sensor is placed outside the product, the recorded air exposure is not necessarily the product exposure.
A second logger may improve coverage but does not automatically eliminate all uncertainty.

**Branch:** an apparent alert caused by a documented device or mapping error may be resolved through the authorized investigation.
Until then, a planner should not treat the unresolved affected quantity as available.

### Step 6 — Make the independent quality decision

**Input:** deviation evidence, product-specific stability information, approved procedures, and distribution history.
**Output:** a documented, authorized disposition for a defined set of units.

A qualified reviewer evaluates the product identity, shipment configuration, exposure history, and data completeness.
The reviewer checks the relevance of stability data to the formulation, packaging, temperature, duration, and cumulative prior exposure.
The reviewer documents assumptions, missing evidence, rationale, and decision authority.
Possible outcomes include continued hold, release subject to conditions, return, or destruction as allowed by the governing procedure.
The example deliberately supplies a disposition as an authorized input; it does not calculate one.

A published general rule for temperature control is not a substitute for product-specific stability evidence.
A model's confidence score is not a quality release.
A business-critical shortage does not make an unsupported product-quality claim true.
Conversely, an alarm alone does not prove all affected medicine must be destroyed.
Keep product disposition separate from the investigation into why the alarm occurred.

**Branch:** if evidence is insufficient, document continued hold and the information required to resolve it.
If the issue spans customers or markets, coordinate with the relevant responsible functions and authorities as required.
Do not communicate release externally before the authorized decision is effective.

### Step 7 — Reconcile identity and traceability

**Input:** physical counts, serialized event records, shipment notices, partner responses, and applicable rules.
**Output:** a reconciled exception list and a transaction-eligible quantity.

1. Confirm the correct product identifier, lot, expiry, serial, and packaging level.
2. Compare shipped and received quantities by product and shipment.
3. Compare the expected serial or aggregation hierarchy with actual observations where required.
4. Record missing, duplicate, unexpected, or unreadable identifiers.
5. Check the sender, recipient, event type, event time, and event location.
6. Check whether required transaction information is available to the authorized partner.
7. Investigate discrepancies without overwriting the original event.
8. Record corrections with actor, time, reason, and evidence.
9. Release only the quantity whose identity, status, and transaction conditions have been resolved.

A shipment event tells you that a source asserted a business event.
It does not independently prove the physical contents, the environmental history, or the quality of the medicine.
An identifier can be copied, scanned incorrectly, or associated with the wrong parent.
A discrepancy may reflect latency or a genuine problem; the investigation should distinguish them.
Check actual current market-specific law and authorized trading-partner obligations before concluding that an event set is sufficient.

**Branch:** missing required data may block a transaction even if the physical stock is cold and in date.
A quality release does not by itself repair an identity exception.
A valid serial record does not by itself establish appropriate temperature exposure.

### Step 8 — Build the available-to-promise supply ledger

**Input:** physical inventory, holds, expiry, commitments, traceability status, and expected replenishment.
**Output:** a timestamped ledger of units eligible for the allocation horizon.

Start from physical on-hand units by batch, location, and packaging presentation.
Deduct units that are expired at the relevant use or receipt date.
Deduct units already committed to another approved plan.
Exclude units with unresolved quality or traceability blocks.
Exclude units not authorized for the recipient's market or presentation.
Apply minimum remaining shelf life and lane or customer constraints.
Do not sum overlapping holds as if they were disjoint; evaluate eligibility at the smallest supported grain.

Make expected inbound supply a separate scenario from on-hand released stock.
Show an arrival probability or scenario only if there is evidence for it.
A forecast batch completion date is not a released inventory quantity.
Where timing matters, distinguish a pack available on day one from a pack arriving after the site's treatment window.

**Branch:** if physical and system counts disagree, reconcile before creating firm commitments.
A provisional scenario can expose the range, but it must label uncertain units explicitly.
The snapshot should identify the extraction time and all known unresolved discrepancies.

### Step 9 — Estimate need without disguising uncertainty

**Input:** orders, consumption, on-hand stock at sites, service horizon, and known operational changes.
**Output:** a site-level demand or need estimate with provenance and uncertainty.

An order may reflect consumption, pipeline filling, double ordering, precautionary behavior, or contract incentives.
Historical shipments may be censored by earlier shortages.
Use stock-on-hand and consumption evidence when available rather than blindly extrapolating shipments.
Avoid exposing individual patient records where aggregated information is sufficient.
Where patient-level data is necessary, establish a lawful basis, access controls, and retention rules.

For the instructional allocation, site requests are treated as eligible need over seven days.
That is an explicit simplifying assumption, not a claim that orders equal patient need in practice.
If a site cannot administer all requested packs before expiry, reduce its feasible demand or change the time horizon.
If requests are old, ask sites to reconfirm them before treating them as commitments.

**Branch:** if need data is unreliable, publish sensitivity ranges and a short reconfirmation deadline.
Avoid interpreting unobserved demand as zero demand, particularly at under-resourced sites.

### Step 10 — Define constraints and policy before optimizing

**Input:** eligible supply, site demand, free capacity, route feasibility, priority policy, and delivery horizon.
**Output:** a documented mathematical and operational decision rule.

Let `x[i]` be whole packs allocated to site `i`.
The supply constraint is `sum(x[i]) <= eligible supply`.
For each site, allocation cannot exceed eligible request or free qualified receiving capacity.
If the site cannot be served within the horizon, its feasible allocation is zero for that run.
Additional constraints may include market authorization, minimum shelf life, contractual rules, and allocation increments.

A simple priority policy sorts sites by approved priority, then a deterministic site ID.
It gives each site as much as possible up to its feasible demand before moving to the next.
This is easy to audit but may leave lower-priority sites with no stock.
A proportional policy divides supply across feasible demands and rounds to whole packs.
That can spread scarcity but may conflict with clinical priority or minimum viable shipment sizes.
Neither policy is universally correct.
The accountable body must select it and document the rationale before seeing the preferred winner.

For a more detailed model, index stock by batch and destination.
Use `x[b,i]` for packs from batch `b` to site `i`.
Constrain batch allocations by eligible batch stock.
Constrain site totals by request and capacity.
Set `x[b,i]` to zero if expiry, product authorization, route, or site constraints fail.
Use integer variables when fractional packs cannot be shipped.
State how ties, fairness, transport cost, and unmet need enter the objective.
Do not use a cost weight as a covert replacement for a non-negotiable quality gate.

**Branch:** if hard constraints make all requests infeasible, report the infeasibility and its cause.
Do not relax a safety, authorization, or quality requirement merely to make a solver return a number.

### Step 11 — Execute and monitor

**Input:** approved allocation, batch assignments, shipment plan, and controlled inventory status.
**Output:** confirmed dispatch, receipt, and exception records.

Reserve stock against the approved plan with a time-limited reservation.
Recheck the eligibility and availability of the actual packs at pick time.
Use FEFO only within the set of eligible stock and feasible destination shelf-life constraints.
Verify pack-out and qualified route, including planned handoffs.
Send the required transaction and shipment information through approved channels.
Confirm receipt, quantity, temperature evidence, and discrepancy resolution.
Update the commitment when a recipient or route changes.

A plan is not execution evidence.
A dispatch is not proof of receipt.
A receipt is not proof of administration.
Measure each separately and prevent the dashboard from collapsing them into one completed flag.

**Branch:** if status changes between planning and dispatch, stop the affected pick and rerun the plan on the new snapshot.
If the recipient loses storage capacity in transit, invoke an approved contingency; do not improvise storage limits.

### Step 12 — Learn and govern

**Input:** outcome records, overrides, deviations, cost, service, and feedback from affected sites.
**Output:** a reviewed performance and improvement record.

Track time from alarm to containment, investigation completion, and authorized disposition.
Track missing data and traceability exceptions separately from confirmed excursions.
Track expired and destroyed packs, but do not label all held packs as waste.
Measure site-level fill rate, time without usable supply, and unfilled eligible demand.
Stratify service by geography and site type when such reporting is lawful and meaningful.
Record when an override improved safety, fixed bad data, or introduced an avoidable inequity.
Review repeated deviations for systemic root causes and appropriate CAPA.

An apparent reduction in excursions after a new sensor rollout might reflect different detection coverage.
Compare measurement coverage and definitions before claiming improvement.
A before-and-after chart alone does not prove that a model caused better patient outcomes.
A controlled evaluation should state the comparison, potential confounding, and outcome measurement limitations.

## 6. Worked example

### Scenario and boundary

All entities and quantities in this section are synthetic.
A manufacturer distributes a refrigerated injectable medicine in whole packs.
The actual approved storage range and stability data are not specified here.
A distribution center holds inventory for three clinics over the next seven calendar days.
The calculation is in packs, not doses or patients.
The requested delivery date and clinic capacities refer to the same seven-day horizon.

The supply director wants a defensible allocation by 14:00 UTC.
The quality unit independently reviews an alarm on one shipment.
The traceability team investigates missing serialized shipment information on part of the same physical lot.
The clinics report their requested packs and free qualified storage capacity.
The allocation committee has approved a priority order before seeing the specific run.
The order reflects a hypothetical policy, not a universal clinical ranking.

### Inventory snapshot

| Inventory group | Physical packs | Quality status | Traceability status | Eligible packs at 14:00 UTC |
|---|---:|---|---|---:|
| Lot A, central depot | 184 | Released; no open incident | Reconciled | 184 |
| Lot B, central depot, group B1 | 54 | Unresolved alarm | Reconciled | 0 |
| Lot B, central depot, group B2 | 22 | Unresolved alarm | Missing event | 0 |
| Lot C, central depot | 35 | Released | Reconciled, but expired for horizon | 0 |
| Total | 295 | Mixed | Mixed | 184 |

The groups B1 and B2 form a partition of the 76-pack shipment affected by the alarm.
B2 has both an unresolved quality alarm and a missing traceability event.
Subtracting 76 and then another 22 from the 295-pack physical total would double count B2.
The expired group C is not available even though it is physically present and traceable.
The initial eligible quantity is therefore 184 packs, not 295 packs and not 162 packs.
No other unreported commitment, capacity constraint, or inbound delivery is assumed.

The temperature logger recorded an out-of-range observation on the B shipment.
This observation triggers a hold and investigation under the synthetic procedure.
It does not determine whether the product is fit for use.
The precise temperature, duration, and allowable exposure are deliberately not invented.
A quality reviewer must use the actual product-specific evidence and the approved procedure.

### Site requests and priority

| Site | Approved priority | Requested packs | Free qualified capacity, packs | Feasible request, packs |
|---|---:|---:|---:|---:|
| North | 1 | 96 | 88 | 88 |
| Central | 2 | 83 | 70 | 70 |
| South | 3 | 74 | 80 | 74 |
| Total | — | 253 | 238 | 232 |

For each site, feasible request is the lesser of requested packs and free qualified capacity.
North: `min(96, 88) = 88` packs.
Central: `min(83, 70) = 70` packs.
South: `min(74, 80) = 74` packs.
The sum is 232 feasible packs, although total nominal free capacity is 238 packs.
The extra six South capacity slots cannot satisfy a different site's request without a different distribution arrangement.
The difference between requested and feasible quantity is 21 packs, caused here by site capacity.

### Initial plan: priority allocation

The policy fills North first, then Central, then South, without exceeding each site's feasible request.
Starting eligible supply is 184 packs.
North receives 88 packs, leaving `184 - 88 = 96` packs.
Central receives 70 packs, leaving `96 - 70 = 26` packs.
South receives 26 packs, leaving zero packs.

| Site | Allocation, packs | Unfilled requested packs | Unfilled feasible packs |
|---|---:|---:|---:|
| North | 88 | 8 | 0 |
| Central | 70 | 13 | 0 |
| South | 26 | 48 | 48 |
| Total | 184 | 69 | 48 |

Requested fill rate is `184 / 253 = 72.73%` after rounding to two decimal places.
Feasible-request fill rate is `184 / 232 = 79.31%`.
These answer different questions.
The first includes site-capacity limitations; the second isolates shortage relative to feasible requests.
South receives `26 / 74 = 35.14%` of its requested packs.
The priority rule protects the first two sites at the expense of South.
That consequence is observable even if the priority itself was approved.

### Later event: quality disposition with an independent traceability hold

At 16:00 UTC, after the initial plan, an authorized quality reviewer decides that the 76 B packs may be released from the temperature hold.
This is a synthetic decision supplied as an input, not a conclusion from a temperature calculation.
The 22 packs in group B2 still have an unresolved traceability exception.
The traceability process has not yet cleared those packs for the relevant transaction.
Thus only the 54 packs in B1 become newly eligible.

If no packs have yet been physically dispatched and the previous reservations can be revised, the new eligible total is `184 + 54 = 238` packs.
The feasible requests total 232 packs, so six packs remain unallocated under this horizon and capacity snapshot.
Re-running the priority rule gives North 88, Central 70, and South 74 packs.
The plan allocates 232 packs; requested fill rate becomes `232 / 253 = 91.70%`.
Feasible-request fill rate becomes 100%.
The remaining 21 requested packs cannot be placed in the stated free capacity.
The six unused eligible packs are not waste; they remain stock subject to later allocation and expiry controls.
The 22 B2 packs remain unavailable pending resolution of the separate traceability issue.

If 184 packs have already been dispatched, do not rewrite the historical plan as though the later evidence existed at 14:00.
Allocate the additional 54 packs against updated need, capacity, and actual receipts.
Confirm whether South still has 48 packs of feasible unmet demand and whether the other site constraints changed.
An approved plan is a snapshot, not a permanent truth.

### Compare a proportional policy

For a policy comparison, return to the initial 184-pack eligible snapshot.
A proportional rule shares the 184 packs in proportion to feasible requests of 88, 70, and 74 packs.
The unrounded quotas are approximately 69.79, 55.52, and 58.69 packs.
Their floors are 69, 55, and 58 packs, totaling 182 packs.
Assign the remaining two whole packs to the largest fractional remainders: North, then South.
The resulting allocations are North 70, Central 55, South 59 packs.
The sum is 184 packs and no site exceeds its feasible request.

| Measure | Priority rule | Proportional rule |
|---|---:|---:|
| North allocated | 88 packs | 70 packs |
| Central allocated | 70 packs | 55 packs |
| South allocated | 26 packs | 59 packs |
| Total allocated | 184 packs | 184 packs |
| South requested fill rate | 35.14% | 79.73% |
| Total requested fill rate | 72.73% | 72.73% |

The rules achieve the same aggregate fill rate in this example but distribute shortfall differently.
The proportional rule does not automatically produce equitable patient access.
The requests may be biased, the sites may serve different populations, and packs may not translate into equal patient benefit.
The decision committee needs evidence on these differences and a documented mandate.

### Decision brief

**Decision:** approve the 14:00 priority allocation of 88, 70, and 26 packs, subject to pick-time eligibility checks.
**Owner:** supply allocation committee; quality retains independent disposition authority.
**Evidence:** 184 eligible packs, site requests and capacities, and approved priority order.
**Excluded:** 76 packs on temperature hold, including 22 with a separate traceability exception, and 35 expired packs.
**Risk:** South experiences a 48-pack feasible shortfall under the selected policy.
**Escalation:** notify South, check alternatives and capacity, and update if authorized disposition or supply changes.
**Revision:** at 16:00, the synthetic quality decision permits a new plan for 54 additional eligible packs, not all 76.
**Not claimed:** the disposition rationale, patient-level outcomes, or legal sufficiency of any particular transaction record.

## 7. Code, query, or calculation

### Lab objective

Run a transparent priority allocation against the worked example.
The code evaluates distinct eligibility flags at inventory-group level.
A group is available only when quality, traceability, expiry, and commitment conditions all permit it.
The program does not infer temperature stability or authorize any product disposition.
The two snapshots represent the evidence available at 14:00 and 16:00 UTC, respectively.

### Assumptions

- Python 3.9 or later; only the standard library is used.
- Each inventory record is a disjoint physical group measured in whole packs.
- The `uncommitted` field already excludes stock promised elsewhere.
- Expiry is expressed as a Boolean eligibility check for this fixed delivery horizon.
- Requests and capacities are whole packs and do not change between the two snapshots.
- Priority is approved in advance; ties are broken lexically by site name.
- All eligible packs are interchangeable for the three sites in this synthetic example.
- The code has no database, audit trail, regulatory validation, or production access control.
- A real implementation must handle batch-to-site eligibility, timestamps, and stale snapshots explicitly.

Save the following as `cold_chain_lab.py` and run `python cold_chain_lab.py`.

```python
from dataclasses import dataclass, replace
from typing import List, Tuple


@dataclass(frozen=True)
class InventoryGroup:
    name: str
    packs: int
    quality_released: bool
    trace_reconciled: bool
    in_date_for_horizon: bool
    uncommitted: bool

    def eligible_packs(self) -> int:
        eligible = (
            self.quality_released
            and self.trace_reconciled
            and self.in_date_for_horizon
            and self.uncommitted
        )
        return self.packs if eligible else 0


@dataclass(frozen=True)
class Site:
    name: str
    priority: int
    request: int
    capacity: int

    def feasible_request(self) -> int:
        return min(self.request, self.capacity)


def allocate_priority(
    groups: List[InventoryGroup], sites: List[Site]
) -> Tuple[List[Tuple[str, int]], int]:
    """Apply an approved priority policy to a fixed eligible snapshot."""
    remaining = sum(group.eligible_packs() for group in groups)
    allocation = []
    for site in sorted(sites, key=lambda item: (item.priority, item.name)):
        quantity = min(remaining, site.feasible_request())
        allocation.append((site.name, quantity))
        remaining -= quantity
    return allocation, remaining


def show(label: str, groups: List[InventoryGroup], sites: List[Site]) -> None:
    eligible = sum(group.eligible_packs() for group in groups)
    allocation, remaining = allocate_priority(groups, sites)
    print(f"{label}: eligible={eligible} packs")
    for name, quantity in allocation:
        print(f"  {name}: {quantity} packs")
    print(f"  unallocated: {remaining} packs")


groups = [
    InventoryGroup("A", 184, True, True, True, True),
    InventoryGroup("B1", 54, False, True, True, True),
    InventoryGroup("B2", 22, False, False, True, True),
    InventoryGroup("C", 35, True, True, False, True),
]
sites = [
    Site("North", 1, 96, 88),
    Site("Central", 2, 83, 70),
    Site("South", 3, 74, 80),
]

show("14:00 UTC", groups, sites)

# An authorized reviewer has released the quality hold on both B groups.
# B2 still fails the separate traceability gate.
later = [
    replace(group, quality_released=True)
    if group.name in ("B1", "B2") else group
    for group in groups
]
show("16:00 UTC", later, sites)
```

Expected output:

```text
14:00 UTC: eligible=184 packs
  North: 88 packs
  Central: 70 packs
  South: 26 packs
  unallocated: 0 packs
16:00 UTC: eligible=238 packs
  North: 88 packs
  Central: 70 packs
  South: 74 packs
  unallocated: 6 packs
```

The second run is a new plan for the entire unchanged horizon, not an instruction to ship the first allocation twice.
A production system must atomically account for reservations, dispatches, and receipts between snapshots.
If a group is partially committed, split it into disjoint groups or represent the eligible uncommitted quantity directly.
If one batch cannot serve a particular site, a single pooled supply number is insufficient.
If a site has a minimum viable shipment, the greedy policy may need a different feasibility rule.
If a quality disposition changes after the snapshot, the system must block affected picks and preserve the earlier decision record.

### Optional hand calculation: interval uncertainty

Suppose a logger reads in range at 10:00 and above the operational threshold at 10:05.
The actual threshold crossing could have occurred at any time between those readings.
Suppose it is above threshold at 10:15 and back in range at 10:20.
Based only on these samples, the recorded above-threshold period spans ten minutes, while the actual crossing-to-return interval is not exactly observed.
Its plausible boundary-based duration is greater than or equal to ten minutes and may approach twenty minutes, subject to behavior between samples.
The reading sequence alone cannot exclude additional unobserved crossings.
This is a measurement illustration, not a stability acceptance criterion.

### Lab extensions

1. Set B2's traceability flag to true in the later snapshot and explain why eligible stock becomes 260 packs but allocation remains 232 packs.
2. Lower South's free capacity to 40 packs and recompute both plans.
3. Add a site with priority 2 and explain the alphabetical tie-break.
4. Sketch an inventory-by-batch assignment model when one batch has too little remaining shelf life for South.
5. Design a lock or reservation transaction that prevents two simultaneous runs from promising the same stock.
6. Document why replacing missing temperature readings with interpolated values would not create measured evidence.

These extensions change assumptions; they do not create a validated production application.
A regulated deployment needs an approved lifecycle, access control, traceability, change control, and documented verification appropriate to its intended use.

## 8. Data, standards, and regulatory context

### Product, shipment, and event data

A minimal analytical data model needs more than a shipment table.
Keep product identity and packaging hierarchy separate from batch and serial instances.
Keep location and partner master data separate from physical events.
Keep sensor observations separate from the shipment-to-device association.
Keep quality decisions and traceability exceptions as controlled records with effective times.
Keep allocation decisions, reservations, dispatches, and receipts as separate facts.

| Dataset | Grain | Essential fields | Important limitation |
|---|---|---|---|
| Product master | Product and packaging level | Identifier, presentation, unit, approved conditions, effective dates | A master-data label is not a stability study |
| Batch record extract | Batch and market | Lot, expiry, release status, decision reference | Status can change after a shipping event |
| Serial master | Individual trade item | Product identifier, serial, lot, expiry, state | Does not by itself show actual location |
| Aggregation | Parent-child relationship and time | Parent, child, event time, action | Can become stale after case opening |
| Shipment | Shipment leg | Origin, destination, actual times, carrier, configuration | Planned route can differ from actual route |
| Sensor observation | Device and time | Value, unit, original time, calibration metadata | Only observes sensor position at sampled times |
| Device assignment | Device and deployment interval | Device, shipment or package, placement, start/end | Incorrect mapping contaminates the excursion scope |
| Deviation | Incident and affected scope | Detection, containment, rationale, status, owner | An alarm is not itself a disposition |
| Trading event | Identified object and event | What, when, where, why, partner, source | Event message is an assertion, not physical proof |
| Inventory | Batch, location, time, status | Physical count, availability flags, commitments | Overlapping holds must not be subtracted twice |
| Demand | Site, product, horizon | Requested quantity, source, date, uncertainty | Orders may not equal patient need |
| Decision | Allocation run and site | Snapshot ID, rule version, quantities, approvals | A plan does not prove fulfillment |

Data quality checks should include completeness, uniqueness, validity, timeliness, and reconciliation.
Their thresholds should match the intended decision.
A 99% event-completeness rate may conceal the specific missing event for the shipment being dispatched.
A low average latency may conceal one delayed upload that arrived after the release decision.
Record source lineage and the version of transformation rules.
Do not replace an original event when a correction is required; retain the correction history.

### GDP and quality management

The European Commission's 2013 Guidelines on Good Distribution Practice of medicinal products for human use describe a quality-system approach to wholesale distribution in the EU context.
They address matters including personnel, premises, documentation, operations, complaints, recalls, outsourced activities, self-inspections, and transport.
The text's applicability depends on the activity, entity, product, and jurisdiction.
The World Health Organization (WHO) also publishes GDP guidance for pharmaceutical products, useful for understanding risk-based distribution controls in a global setting.
WHO guidance is not automatically the law of every jurisdiction.
National legislation, licenses, product authorizations, contracts, and current regulatory guidance must be considered locally.
See the official sources in Section 16.

GDP does not mean that every shipment must use a universal logger threshold.
Temperature requirements are product- and configuration-specific.
GDP also does not give an analyst authority to release material.
The quality system should define the relevant deviation, investigation, and disposition controls.
Qualification demonstrates performance within defined conditions; changes can invalidate that evidence for the new use.

### U.S. DSCSA

The U.S. Drug Supply Chain Security Act was enacted in 2013.
FDA describes the law's enhanced system for tracing certain prescription drugs at package level through the U.S. supply chain.
Covered trading partners and transactions have specific duties, exceptions, exemptions, and implementation details.
FDA has issued guidance and time-limited exemptions or enforcement policies for particular circumstances.
Do not infer a current deadline or a universal exemption from an old summary.
Check the FDA DSCSA page, current guidance, applicable statutes, and the partner's facts at the time of the transaction.

Product identification, transaction information exchange, verification, and suspect or illegitimate product processes are related but distinct concepts.
An EPCIS message may support an interoperable exchange, but a message format alone cannot establish full DSCSA compliance.
A scanned identifier does not prove the medicine is authentic or was stored correctly.
A missing event should be investigated using applicable procedures rather than assumed to prove falsification.

### EU safety features and verification

Directive 2011/62/EU amended the EU framework to address falsified medicinal products.
Commission Delegated Regulation (EU) 2016/161 describes safety features for medicinal products for human use within its scope, including a unique identifier and anti-tampering device.
The regulation includes detailed applicability, responsibilities, and exceptions.
The EU approach includes verification and decommissioning processes under its own legal framework.
It should not be presented as identical to the U.S. transaction-tracing regime.
A 2D code or successful verification is not a temperature certificate.
Read the current consolidated texts and national implementation requirements for the product and transaction.

### GS1 identifiers and EPCIS

GS1 defines identifiers including GTIN, GLN, and SSCC, and publishes the EPCIS standard for sharing visibility event data.
An EPCIS event can describe what objects were involved, when and where an event occurred, and the business context.
Depending on the version and implementation, the standard supports additional information, including sensor-related data.
Trading partners must agree on profiles, data quality, access, and operational handling.
GS1 is a standards organization, not a substitute for a regulator or a product's quality authority.
Using a standard supports interoperability but does not automatically validate each record or satisfy every jurisdictional duty.

### Privacy, security, and controlled records

A serial or shipment record need not contain patient identity to support many distribution decisions.
Minimize personal data and separate sensitive clinical data from routine operational analytics.
Apply the applicable privacy law and organizational policy to any personal data, including site contacts.
Use role-based access to separate quality disposition, allocation planning, and system administration.
Protect source files and event records from unauthorized alteration.
Record who changed a status, when, why, and under what authority.
A technical audit trail does not replace a quality procedure for review and approval.

## 9. Industry practice and case evidence

### Case A — WHO vaccine distribution guidance

**Documented source:** WHO publishes guidance on good storage and distribution practices for medical products and resources on vaccine management.
These sources recognize distribution controls, suitable equipment, monitoring, and documented procedures as part of protecting product quality.
The documents are guidance; national implementation and product-specific instructions determine operational requirements.
They do not establish that a particular fictional shipment survived a temperature excursion.

**Transfer inference:** A network with multiple handoffs benefits from connecting custody, sensor, and quality records at shipment level.
This inference follows from the need to investigate what happened and to which stock.
It is not a measured result from the WHO document.
A site should test whether device placement and event coverage support that linkage.

**Decision application:** Before a route change, compare the proposed route and pack-out with qualified conditions.
If the change is outside the qualification, ask the appropriate quality function to determine the required assessment.
Do not rely only on historical average transit time.

### Case B — COVID-19 vaccine temperature requirements

**Documented source:** CDC maintains vaccine storage and handling resources, including product-specific information and manufacturer links.
COVID-19 vaccine products and presentations have had different temperature and handling instructions, which have changed over time.
The source should be consulted at the time of use for the specific presentation.
Do not extrapolate one product's storage period or thawing rule to another.

**Transfer inference:** Product master data must be versioned by presentation and effective date.
A generic field called `refrigerated` is insufficient when allowable storage differs by state, formulation, or time since thaw.
This is a design implication, not an evaluation of a specific implementation.

**Decision application:** If a planned route includes a transition from frozen storage to refrigerated storage, establish which approved clock starts at the transition.
Make the clock available to recipients and record the actual transition, not merely the planned date.
A product's approved instructions and local procedures govern the actual handling decision.

### Case C — U.S. package-level tracing

**Documented source:** FDA describes DSCSA's objective of establishing an interoperable system to identify and trace certain prescription drugs as they move through the U.S. supply chain.
FDA also publishes guidance on implementation, including circumstances affected by exemptions and enforcement policies.
The public material does not establish that any particular partner has exchanged complete or correct data.

**Transfer inference:** A partner-onboarding plan should include event testing, exception ownership, and reconciliation in addition to barcode scanning.
Interoperability depends on trading partners sending compatible, accurate data on time.
A technical test cannot by itself establish that physical flows match every message.

**Decision application:** When an inbound shipment's transaction data is delayed, distinguish the physical-receipt workflow from the legal and procedural conditions for onward movement.
Escalate using the current rules and the trading partner agreement.
Avoid treating a missing message as either a confirmed counterfeit or an inconsequential delay.

### Case D — Public shortage information

**Documented source:** FDA maintains a searchable database of drug shortages and related information for the United States.
The database is a public source for reported shortage status and related updates, not a real-time view of every clinic's inventory.
Status, affected presentations, and expected availability can change.
A shortage listing alone does not show each site's clinical need or authorize a private allocation rule.

**Transfer inference:** Scarcity planning should combine external shortage signals with verified local inventory and demand.
A national status can prompt a review, but the actual allocation needs product, location, timing, and capacity data.
The absence of a listing does not prove that all local sites have sufficient stock.

### What evidence would support impact?

To evaluate a new alerting process, define the outcome before implementation.
Possible outcomes include time from a verified alarm to containment and the fraction of incidents with complete source data.
Measure false alarms, missed coverage, and device uptime as well as alarm counts.
Compare routes or periods with similar products, monitoring coverage, and operating conditions where possible.
Record simultaneous packaging, carrier, or seasonal changes that could explain observed differences.
A lower destruction count may reflect better protection, different disposition criteria, or less complete monitoring.
Do not attribute the change to the model without a credible comparison and a reviewed quality context.

To evaluate allocation, track both aggregate service and site-level distributions.
Measure eligible demand, delivery date, accepted quantity, and stockout duration using consistent definitions.
Consult affected sites about whether reported requests represent actual need.
A model may improve overall fill while worsening access for a small remote site.
Use the governance process to decide whether that trade-off is acceptable.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Hold treated as supply | Available quantity rises while deviation remains open | ERP status not linked to planning snapshot | Infeasible or unsafe commitment | Gate on authoritative current status and recheck at pick |
| Overlapping holds double counted | Exclusions exceed physical stock | Independent totals subtracted without unit-level intersection | Artificial shortage and misallocation | Evaluate independent flags on disjoint groups |
| Alarm mistaken for destruction | Automatic write-off on first alert | No product-specific investigation | Avoidable waste and shortage | Quarantine and authorized disposition |
| Alarm mistaken for release | Planner clears status to meet demand | Commercial and quality roles conflated | Potential distribution of unsuitable stock | Independent authority and access segregation |
| Logger mapped to wrong shipment | Device ID reused or shipment link missing | Faulty pack-out or data association | Wrong affected scope | Verify deployment and retain chain of evidence |
| Sensor gap hidden by interpolation | Long flat segment or late upload | Derived values presented as readings | False assurance | Mark missing intervals and escalate uncertainty |
| Single logger overgeneralized | Unmapped load positions | Sensor not representative of entire load | Local hot or cold spot missed | Qualified placement and mapping |
| Time zone shifted | Events appear before loading | Local time treated as UTC | Incorrect exposure duration or custody sequence | Preserve original offset and conversion lineage |
| Duplicate serial | Same identifier appears in incompatible places | Scan error, duplication, or suspect event | Traceability block or investigation failure | Exception workflow and source reconciliation |
| Stale aggregation | Parent contents differ after repacking | Child events not updated | Wrong units identified in recall or shipment | Capture disaggregation and reaggregation |
| Serial equated with quality | Valid scan shown as green safety status | Identity and temperature gates collapsed | False confidence in product suitability | Separate identity, temperature, and quality statuses |
| Expiry checked at dispatch only | Arrival has insufficient remaining life | Long transit or recipient requirement omitted | Rejection or waste | Apply shelf-life check at relevant receipt or use time |
| FEFO ignores eligibility | Oldest batch is quarantined | Rotation treated as release | Unlawful or unsuitable pick | Apply FEFO only within eligible feasible stock |
| Nominal rather than free capacity | Recipient cannot unload | Existing inventory ignored | Diversion, exposure, and delay | Confirm free qualified capacity at arrival |
| Forecast confuses orders with need | Ordering spikes after allocation rumor | Double ordering or precautionary orders | Unfair and unstable allocation | Reconcile consumption, stock, and site confirmations |
| Fairness hidden in weights | One site systematically loses | Revenue or distance proxy dominates | Unequal access without explicit policy | Review site-level outcomes and approve objectives |
| Stale snapshot reused | Stock was dispatched after extract | No reservation or refresh | Double promise | Atomic reservation and snapshot expiry |
| Later evidence backdated | Dashboard shows release before review | Status overwritten instead of versioned | Misleading audit record | Preserve event, decision, and effective times |
| Partner master stale | Shipment to unqualified or wrong location | Qualification not maintained | Transaction or service failure | Effective-dated partner checks |
| New route inherits old qualification | Route duration or pack-out changed | Qualification boundaries ignored | Uncontrolled distribution condition | Change control and quality review |
| KPI rewards low alert count | Device uptime falls while alerts fall | Under-detection looks like improvement | Undetected incidents | Report coverage and uptime alongside alerts |
| User override unrecorded | Allocation differs from model without reason | Manual process outside controlled workflow | No accountability or learning | Capture actor, rationale, approval, and outcome |

A red flag is a trigger for investigation, not proof of a specific cause.
Document whether the signal is a confirmed event, a data-quality concern, or an inference.
The most important mitigation is often an ownership boundary rather than a more complex algorithm.

## 11. Assumptions and limits

| Assumption | Why needed here | What breaks it | How to test or respond |
|---|---|---|---|
| Site requests represent eligible seven-day need | Makes the example allocatable | Double orders, old requests, or censored demand | Compare with consumption and reconfirm with sites |
| Packs are interchangeable across sites | Allows pooled supply | Different market labels, authorizations, or shelf-life minima | Build batch-to-site eligibility matrix |
| Site capacity is free qualified capacity | Limits feasible receipt | Other shipments arrive or equipment fails | Confirm free capacity at arrival and refresh |
| Groups are physically disjoint | Prevents double subtraction | Same pack appears in two group records | Reconcile serial or inventory-level membership |
| Quality status is current and authoritative | Defines eligible stock | Late decision or stale extract | Record snapshot and decision effective times |
| Traceability status is independent of quality | Preserves both gates | Organization uses an incomplete combined status | Maintain underlying exceptions separately |
| Whole packs are the indivisible unit | Makes allocation integer | Different pack sizes or permitted case-only shipment | Normalize units and add packaging constraints |
| Priority is preapproved | Makes greedy policy legitimate for the example | Policy changed after seeing results | Version approval and retain decision record |
| All deliveries reach sites within horizon | Allows static capacity comparison | Delays, lane closures, or receiving-hour conflicts | Add time-indexed route and arrival constraints |
| No new consumption occurs during planning | Makes two snapshots comparable | Sites use stock or new patients arrive | Refresh demand and existing inventory |
| No dispatch occurs between example runs | Avoids double shipment on rerun | First plan is partially executed | Subtract actual dispatches and reservations |
| Expiry eligibility is known for horizon | Excludes lot C | Incorrect master data or destination rules | Verify dating and remaining shelf life by recipient |
| Sensor readings represent a known position | Supports excursion investigation | Logger is displaced, missing, or misassigned | Confirm pack-out evidence and calibration |
| Source guidance remains current | Supports educational description | Legal text or regulator policy changes | Check official current sources for each decision |

The example has no stochastic transit-time model, no product stability curve, and no clinical prioritization algorithm.
It does not model dose yield, administration wastage, individual patients, reimbursement, or substitution.
A real network may have multiple products, country labels, route capacities, and batch-specific constraints.
The simple allocation policy is interpretable precisely because it omits those complexities.
Do not use its output in operations without the controls, validation, and domain review appropriate to the intended use.

## 12. Exercises

### Exercise 1 — Map the decision and buyer

**Type:** recall and application. **Time:** 15 minutes.
A depot reports a temperature alarm on a shipment containing two lots for three hospitals.
The commercial director wants a same-day allocation.

**Deliverable:** a one-page decision map with five named roles, three gates, and a time line.
**Acceptance criteria:** distinguish supply planning from quality disposition; include traceability and site capacity; state at least one escalation trigger.

### Exercise 2 — Reconstruct an uncertain excursion

**Type:** calculation and critique. **Time:** 20 minutes.
A logger reads in range at 09:00, out of range at 09:10 and 09:20, and in range at 09:30.
The sampling interval is ten minutes.
There is no other evidence about crossings between readings.

**Deliverable:** a short timeline, the observed out-of-range span, and a statement of what cannot be determined.
**Acceptance criteria:** report ten minutes between the two out-of-range samples; recognize uncertainty around start, end, and unobserved crossings; do not make a product-disposition decision.

### Exercise 3 — Reconcile an identity exception

**Type:** application. **Time:** 20 minutes.
A case is recorded as containing 24 serialized packs.
The physical count is 24, but the received event contains 23 expected serials and one serial absent from the original aggregation.
The case was opened during transit.

**Deliverable:** an exception triage list and a proposed evidence request.
**Acceptance criteria:** identify the missing and unexpected serials; investigate disaggregation and reaggregation; retain original records; avoid declaring either counterfeit or compliant without further evidence.

### Exercise 4 — Allocate under a hold

**Type:** calculation. **Time:** 20 minutes.
There are 150 physical packs: 90 available, 40 on both quality and traceability hold, and 20 expired.
Site A requests 55 packs with capacity for 50; site B requests 60 with capacity for 60.
The approved priority order is A then B.

**Deliverable:** eligible supply, allocations, remaining supply, and each site's unfilled request.
**Acceptance criteria:** do not subtract the two overlapping holds twice; use capacity; show units and arithmetic.

### Exercise 5 — Compare policy outcomes

**Type:** calculation and critique. **Time:** 20 minutes.
In Exercise 4, compare the priority result with a 90-pack proportional allocation over feasible requests of 50 and 60 packs.
Use largest-remainder rounding to whole packs.

**Deliverable:** two allocations and a two-sentence policy trade-off.
**Acceptance criteria:** allocations sum to 90 packs; neither exceeds its feasible request; describe who gains and loses without claiming an ethically universal winner.

### Exercise 6 — Apply regulatory and quality boundaries

**Type:** critique. **Time:** 20 minutes.
An analyst writes: “The GS1 barcode proves the product stayed cold, and EU FMD is the same as U.S. DSCSA, so the model may release the lot.”

**Deliverable:** a corrected paragraph and a list of three authoritative sources to check.
**Acceptance criteria:** separate identity, environmental evidence, legal jurisdiction, and authorized disposition; avoid a generic universal temperature allowance.

### Exercise 7 — Modify the lab

**Type:** coding and design. **Time:** 25 minutes.
In the Section 7 program, set the later B2 traceability status to reconciled.
Then change South's capacity from 80 to 40 packs in both runs.

**Deliverable:** the revised eligible totals, allocations, and an explanation of any unallocated stock.
**Acceptance criteria:** preserve the independent quality and traceability gates; report both snapshots; do not count an unallocated pack as automatically wasted.

### Exercise 8 — Governance and scenario brief

**Type:** design. **Time:** 30 minutes.
A remote site with a small refrigerator is last under a priority rule and repeatedly has unmet demand.
An executive proposes optimizing revenue per shipment to reduce transport cost.

**Deliverable:** a one-page proposal with decision owner, two policy options, hard constraints, data limitations, equity measures, and a review trigger.
**Acceptance criteria:** explain why safety and quality gates are not tradeable weights; identify an affected-site feedback mechanism; distinguish observed data from proposed inference.

## 13. Self-check answers

### Answer 1

A strong map includes the carrier or depot operator detecting and containing the incident, quality assessing and disposing of affected material, a traceability owner reconciling event data, a planner preparing feasible allocation scenarios, and the recipient confirming capacity.
Regulatory or compliance staff may advise on market-specific duties.
The quality, traceability, and allocation gates are distinct.
Escalate if affected scope, logger evidence, or authorization is unknown.
The supply director may approve a plan but must not override quality disposition.

### Answer 2

The first and last observed out-of-range readings are ten minutes apart.
The first crossing lies somewhere between 09:00 and 09:10 if the sequence is interpreted as a continuous crossing.
The return lies somewhere between 09:20 and 09:30 under the same simplification.
That suggests a boundary-based duration between ten and thirty minutes for a single continuous excursion, with endpoints limited by sampling resolution.
Additional unobserved crossings remain possible.
A product-quality conclusion requires the relevant approved evidence, context, and authorized reviewer.

### Answer 3

The count matches, but the identities do not.
Compare the expected serial set with the received serial set to identify one missing and one unexpected serial.
Request the opening, disaggregation, repacking, and custody records, plus partner event timestamps.
Preserve both source messages and record any correction with its rationale.
Apply the relevant exception and suspect-product procedures while the discrepancy is unresolved.
The observation alone establishes neither falsification nor compliance.

### Answer 4

Eligible supply is 90 packs.
The 40 packs have two holds but are one physical group, so they are excluded once.
The 20 expired packs are also excluded.
Site A's feasible request is `min(55, 50) = 50` packs.
It receives 50 packs, leaving 40 packs for site B.
B receives 40 packs; no packs remain.
A has five unfilled requested packs; B has 20.
The total unfilled request is 25 packs.

### Answer 5

Feasible requests total 110 packs.
The unrounded proportional shares are `90 × 50 / 110 = 40.909...` and `90 × 60 / 110 = 49.090...` packs.
Floors are 40 and 49 packs; one remaining pack goes to A's larger remainder.
The proportional allocation is A 41 and B 49 packs.
The priority allocation was A 50 and B 40 packs.
The first protects A's priority; the second spreads shortage across both sites.
The ethical assessment needs the policy mandate and information on patient need and access.

### Answer 6

A GS1 identifier supports identification and event interoperability; it is not proof of temperature history.
A product-specific, investigated temperature record and quality decision are separate from identity verification.
EU safety-feature rules and U.S. DSCSA have different legal scopes and mechanisms.
A model can support analysis but does not independently authorize lot release.
Relevant sources include the GS1 EPCIS standard, FDA's DSCSA resources, and the EU delegated regulation linked in Section 16.
Consult product-specific instructions and the applicable GDP guidance as well.

### Answer 7

At 14:00, both B groups still have a quality hold, so eligible supply remains 184 packs.
The new feasible requests are North 88, Central 70, and South 40 packs, totaling 198.
The first allocation is 88, 70, and 26 packs, leaving zero packs.
At 16:00, B1 and B2 are both quality-released and traceability-reconciled, so eligible supply is `184 + 54 + 22 = 260` packs.
The allocation is 88, 70, and 40 packs, totaling 198.
The remaining 62 packs are eligible but unallocated under this horizon and capacity assumption.
They may be available for a later plan, subject to expiry and other controls.

### Answer 8 rubric

A strong proposal names an accountable allocation body and independent quality authority.
It presents at least two explicit rules and quantifies site-level fill and shortfall.
It keeps authorization, product status, shelf life, and qualified capacity as hard feasibility checks.
It tests whether requests understate need at the remote site and offers a way for that site to correct the record.
It describes uncertainty, competing objectives, data protection, and a revision trigger.
A weak proposal maximizes revenue without recognizing the access consequence or treats an unresolved quality hold as a cost penalty.

## 14. Interview questions

### 1. How would you respond to a temperature alarm during a scarce-product shipment?

A strong answer distinguishes immediate containment, evidence preservation, product-specific investigation, and independent disposition.
It plans alternative supply without assuming release or automatic destruction.
A weak answer clears the alarm because the product is urgently needed.

### 2. What is the difference between a lot and a serial number?

A strong answer explains a batch as a production grouping and a serial as an individual instance identifier in product context.
It also explains that a lot can span shipments with different environmental histories.
A weak answer uses the two fields interchangeably.

### 3. What does a valid EPCIS event prove?

A strong answer describes an event assertion with identity, time, place, and business context, subject to data quality and source trust.
It explicitly separates event validity from physical custody, temperature suitability, and legal compliance.
A weak answer calls the event proof of product safety.

### 4. How do you calculate available supply with overlapping holds?

A strong answer evaluates independent eligibility flags on a disjoint physical unit or group.
It avoids subtracting the same stock twice and preserves each unresolved issue.
A weak answer sums separate hold reports without reconciling their intersection.

### 5. When is FEFO not enough?

A strong answer first filters for release, traceability, product authorization, and destination shelf life.
It uses FEFO only among feasible batches.
A weak answer always picks the oldest physical lot.

### 6. How would you choose an allocation objective?

A strong answer requests a documented mandate, identifies hard constraints, compares site-level outcomes, and involves accountable decision-makers.
It distinguishes requests, patient need, and capacity.
A weak answer assumes revenue or aggregate fill is always the correct objective.

### 7. How do you handle missing temperature data?

A strong answer preserves the gap, checks device deployment and available corroborating evidence, and escalates under the quality procedure.
It states the loss of certainty and does not invent measurements.
A weak answer fills the gap with the mean and labels the shipment compliant.

### 8. How do EU verification and U.S. tracing differ?

A strong answer identifies separate legal frameworks, checks scope and current implementation, and avoids claiming one event standard is the law.
It cites current official sources rather than memorizing an old deadline.
A weak answer treats all jurisdictions and product categories as identical.

### 9. What would you monitor after deploying an allocation model?

A strong answer includes eligible demand, site fill, stockout duration, overrides, stale snapshots, and safety or traceability exceptions.
It checks reporting coverage and disparities across sites.
A weak answer reports only total packs shipped.

### 10. What would invalidate the worked example's pooled-stock model?

A strong answer names batch-specific market authorization, expiry, recipient shelf-life limits, lane restrictions, and timing.
It proposes a batch-by-site feasibility model and controlled status inputs.
A weak answer assumes all packs of the same drug are interchangeable everywhere.

## 15. Key terms

- **Aggregation:** parent-child packaging relationships; see [Section 5](#step-3--establish-master-data-and-identifiers).
- **Allocation:** assignment of eligible scarce supply; see [Section 6](#6-worked-example).
- **Audit trail:** attributable history of changes; see [Section 8](#privacy-security-and-controlled-records).
- **Batch:** production grouping with a lot identifier; see [Section 4](#4-definitions-and-vocabulary).
- **Batch release:** authorized production-quality decision; see [Section 3](#3-the-core-mental-model).
- **Calibration:** comparison with a reference; see [Section 5](#step-4--ingest-temperature-and-context).
- **CAPA:** corrective and preventive action; see [Section 5](#step-12--learn-and-govern).
- **Capacity:** free qualified space at a site; see [Section 6](#site-requests-and-priority).
- **Commitment:** authorized promise distinct from a forecast or plan; see [Section 3](#3-the-core-mental-model).
- **Disposition:** authorized outcome of an investigation; see [Section 5](#step-6--make-the-independent-quality-decision).
- **DSCSA:** U.S. drug tracing framework; see [Section 8](#us-dscsa).
- **EPCIS:** GS1 event-sharing standard; see [Section 8](#gs1-identifiers-and-epcis).
- **Excursion:** exposure outside a defined condition; see [Section 5](#step-5--detect-contain-and-document-an-exception).
- **FEFO:** first-expiry-first-out within eligible stock; see [Section 5](#step-11--execute-and-monitor).
- **FMD:** EU falsified medicines framework; see [Section 8](#eu-safety-features-and-verification).
- **Forecast:** uncertain estimate of demand; see [Section 3](#3-the-core-mental-model).
- **GDP:** good distribution practice; see [Section 8](#gdp-and-quality-management).
- **GLN:** GS1 location or party identifier; see [Section 4](#4-definitions-and-vocabulary).
- **GTIN:** GS1 trade-item identifier; see [Section 4](#4-definitions-and-vocabulary).
- **Hold:** restriction pending decision; see [Section 3](#3-the-core-mental-model).
- **Logger:** device recording environmental observations; see [Section 5](#step-4--ingest-temperature-and-context).
- **Need:** demand estimate requiring context; see [Section 5](#step-9--estimate-need-without-disguising-uncertainty).
- **Priority rule:** approved ordering of recipients; see [Section 6](#initial-plan-priority-allocation).
- **Proportional allocation:** sharing relative to a defined denominator; see [Section 6](#compare-a-proportional-policy).
- **Quarantine:** controlled restriction on material; see [Section 4](#4-definitions-and-vocabulary).
- **Remaining shelf life:** time to expiry at a relevant decision point; see [Section 5](#step-8--build-the-available-to-promise-supply-ledger).
- **Serial:** instance identifier in product context; see [Section 5](#step-3--establish-master-data-and-identifiers).
- **Stability data:** product-specific evidence of quality over conditions and time; see [Section 5](#step-6--make-the-independent-quality-decision).
- **Traceability:** linking product identity to relevant events and partners; see [Section 5](#step-7--reconcile-identity-and-traceability).
- **Unmet demand:** requested quantity not fulfilled under a defined horizon; see [Section 6](#initial-plan-priority-allocation).

## 16. Sources

Sources below are authoritative starting points, not a substitute for checking the current text and jurisdiction.
The cited documents support the stated framework and case facts; the synthetic calculations are the author's instructional examples.

1. European Commission. [Guidelines of 5 November 2013 on Good Distribution Practice of medicinal products for human use (2013/C 343/01)](https://eur-lex.europa.eu/legal-content/EN/TXT/?uri=CELEX:52013XC1123(01)). EU GDP guidance and quality-system context.
2. World Health Organization. [WHO good storage and distribution practices for medical products](https://www.who.int/publications/m/item/trs1025-annex7). WHO Technical Report Series 1025, Annex 7, 2020; global guidance, subject to local implementation.
3. U.S. Food and Drug Administration. [Drug Supply Chain Security Act (DSCSA)](https://www.fda.gov/drugs/drug-supply-chain-integrity/drug-supply-chain-security-act-dscsa). Overview and links to current guidance and implementation information.
4. U.S. Food and Drug Administration. [Drug Shortages](https://www.fda.gov/drugs/drug-safety-and-availability/drug-shortages). Public shortage information and database access; not a site-level inventory feed.
5. European Union. [Directive 2011/62/EU](https://eur-lex.europa.eu/eli/dir/2011/62/oj). Falsified medicines amendments to the EU medicines framework.
6. European Commission. [Commission Delegated Regulation (EU) 2016/161](https://eur-lex.europa.eu/eli/reg_del/2016/161/oj). Detailed safety-feature rules within their stated scope; consult consolidated text and current implementation.
7. GS1. [EPCIS and CBV standard](https://www.gs1.org/standards/epcis). Event visibility standard and implementation resources.
8. GS1. [GS1 identification keys](https://www.gs1.org/standards/id-keys). Definitions and scope of GTIN, GLN, SSCC, and related keys.
9. U.S. Centers for Disease Control and Prevention. [Vaccine Storage and Handling Toolkit](https://www.cdc.gov/vaccines/hcp/admin/storage/toolkit/index.html). Vaccine storage and handling resources; verify current product-specific instructions.
10. World Health Organization. [Immunization supply chain](https://www.who.int/teams/immunization-vaccines-and-biologicals/essential-programme-on-immunization/supply-chain). Program resources on vaccine supply and logistics.

The course survey contains a historical summary of implementation timing.
For any live U.S. DSCSA transaction, consult current FDA materials rather than relying on the survey's date shorthand.
No named organization above is asserted to have used the synthetic allocation algorithm or achieved its numerical results.

## 17. Links

- [Module 29 topic index](_index.md).
- [Module 29 survey](../../../course/29_industry_electives.md).
- [CPG and retail elective](01_cpg-retail.md): compare shelf replenishment and retail allocation.
- [Humanitarian logistics elective](05_humanitarian.md): compare need, access, and allocation governance.
- [Sector playbook lab](08_sector-playbook-lab.md): turn this analysis into a decision brief.
- [Logistics, pharma, and humanitarian cases](../../04_value_governance/28_domain_case_library/08_logistics-pharma-cases.md): public case evidence and transfer tests.
- [Simulation and optimization production labs](../30_simulation_optimization_production_labs/_index.md): extend the allocation into network and scenario models.

Some elective siblings are listed as planned in the module index; follow their links when those topics are published.
