# Disruption Detection

> Module 14A — Supply Chain Risk, Resilience, and Disruption Analytics  
> Topic 09 of 10  
> Estimated study time: 4 hours  
> Prerequisites: disruption timeline, exposure mapping, time-to-recover and time-to-survive  
> Version 2026.5

## 1. Why this topic exists

A supplier can miss a production commitment before its customer sees a late shipment.
A vessel can be delayed before the purchase order's promised arrival date changes.
A weather alert can describe a hazard near a plant that has no exposure to it.
An article can name a supplier whose affected subsidiary does not supply the buyer.
Each observation calls for a different decision, not simply a red dashboard tile.

A buyer wants to know which orders need attention today.
A transport planner wants to know whether an alternative route can still make the delivery window.
A plant manager wants to know when a component shortage would constrain production.
A finance partner wants to know whether the possible loss justifies the response cost.
A continuity lead wants clear ownership while facts remain incomplete.

A warning is valuable only if it arrives early enough to change a feasible action.
The earliest possible signal is not necessarily the best one.
A vague rumor may be early but can consume capacity needed for verified cases.
An accurate confirmation after the last booking deadline may have no preventive value.
A warning programme therefore manages usefulness, timeliness, coverage, credibility, and cost together.

Consider a circuit-board assembler that depends on one specialty connector.
The supplier reports that output may be curtailed by a regional storm.
The port feed also reports increasing vessel dwell time.
A news article repeats the supplier's announcement.
These may be three observations of one developing event, not three independent events.
The buyer must relate the observations to the supplier site, the relevant shipment, the inventory position, and the remaining response time.

The expensive errors run in both directions.
An unneeded air shipment can consume budget and divert scarce capacity.
An ignored credible warning can lead to a production stop and lost contribution.
An alert that reaches the right person after the booking deadline can still be operationally useless.
A warning system that reports no alerts during a data outage can create false confidence.

This lesson designs the detection-to-decision process.
It covers supplier, port, weather, and news signals without claiming that any feed provides ground truth on its own.
It also shows how to measure false alarms and missed events in the presence of interventions.
All case numbers in the worked example are illustrative, not a claim about a real company.

## 2. Learning outcomes

After this lesson you can:

1. Map a warning to an exposed node, shipment, product, decision owner, and deadline.
   Exercise 1 checks the mapping.
2. Distinguish event time, publication time, receipt time, detection time, and actionable lead time.
   Exercise 2 checks the clock arithmetic.
3. Build an incident-level confusion matrix and compute precision, recall, and false-positive rate.
   Exercise 3 checks these calculations.
4. Estimate the economic value of a warning and compare it with response and monitoring costs.
   Exercise 4 checks the decision calculation.
5. Critique correlated feeds, stale data, ambiguous entities, and outcome leakage.
   Exercise 5 checks the critique.
6. Design escalation thresholds, decision rights, and a time-bounded fallback.
   Exercise 6 checks the operating design.
7. Compare warning thresholds under different base rates and capacity limits.
   Exercise 7 checks the comparison.
8. Propose a prospective evaluation that separates observed outcome from the effect of an intervention.
   Exercise 8 checks the evaluation design.

State the unit, time zone, event definition, and decision horizon in every calculation.
Treat estimated probabilities and costs as assumptions unless there is suitable evidence for them.

## 3. The core mental model

Detection converts observations into a time-limited business option.
The complete chain is:

```text
Supplier | port | weather | news observations
                 |
                 v
     Timestamp, provenance, freshness
                 |
                 v
  Resolve entity + geography + incident
                 |
                 v
     Match to exposed nodes and orders
                 |
                 v
 Estimate impact, uncertainty, and deadline
                 |
                 v
   Triage -> investigate -> authorize action
                 |                 |
                 v                 v
        monitor/close        execute/review
                 \                 /
                  v               v
              Record outcome and learn
```

The flow is not a claim that detection causes a disruption.
A weather forecast describes a possible hazard.
An exposure map identifies a facility that might be affected.
A supplier update may provide evidence of an operational consequence.
A decision compares feasible responses with the remaining time and likely loss.

### Four distinct questions

**Hazard:** What has happened or might happen?
A storm track is a hazard forecast, not a confirmed plant outage.

**Exposure:** Which facilities, shipments, suppliers, or customers could be affected?
A plant outside the forecast area may still depend on an exposed upstream port.

**Consequence:** What is the likely effect on usable supply and customer service?
An exposed site with spare capacity elsewhere may have little immediate product impact.

**Actionability:** Can a decision maker still change the outcome at an acceptable cost?
If an alternate shipment takes ten days and the shortage begins in two days, that response cannot prevent the first shortage.

### An alert is not an incident

A feed item is an observation from a source.
An alert is a rule or model output that requests attention.
An incident groups observations about a distinct disruption at a defined scope.
A case is the managed workflow for a potentially affected business decision.
One incident can affect many cases, and one case can include several related incidents.
Define the grain before counting alerts or estimating value.

### Two clocks and one deadline

The impact clock runs from the present to the first decision-relevant consequence.
The response clock is the duration needed to authorize and execute a specific action.
A booking or qualification deadline is the latest time that action can still prevent the consequence.
Effective warning lead time is measured from when a responsible decision maker can act, not from an inaccessible publication timestamp.
If the effective lead time is less than the response duration, the action may still reduce later losses but cannot prevent the initial impact.

### The decision, not the score, owns the outcome

A score can rank cases when review capacity is scarce.
A score is not automatically a calibrated probability.
A buyer may decide to investigate a low-probability event because the investigation is cheap and the potential loss is severe.
A high-probability event may require no expenditure if sufficient inventory and recovery capacity already exist.
Capture the reasoning and later update it rather than converting uncertainty into false precision.

### Why multiple feeds do not automatically mean independent evidence

A supplier announcement, a reporter's summary, and a vendor alert quoting that article can all originate from one statement.
Counting all three as independent confirmation exaggerates certainty.
A weather alert and an independently measured plant outage are different types of evidence, but their relationship still requires judgment.
Keep the source chain and incident link visible to the analyst.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| Hazard | A potentially damaging event or condition, described by location and time | Calling a forecast impact a confirmed operational loss |
| Exposure | A mapped entity, flow, or asset that could encounter the hazard | Assuming geographical proximity proves interruption |
| Vulnerability | Susceptibility to a given hazard and severity | Treating every exposed site as equally fragile |
| Disruption | A defined adverse deviation from expected supply or operation | Counting each article about the same disruption as a new event |
| Signal | An observation potentially informative about a disruption | Treating an unverified signal as a fact |
| Feed | A delivery channel for observations | Treating a feed provider as the original source |
| Provenance | Origin and transformation history of information | Removing quoted source or publication time |
| Entity resolution | Matching a name or identifier to a known organization or location | Assuming all sites sharing a company name are the same facility |
| Incident | A grouped event at a stated scope and time | Merging unrelated events merely because they share a supplier |
| Case | A tracked decision workflow for an exposed business object | Assuming one incident always means one affected order |
| Event time | When the physical or operational event occurred, in a specified time zone | Confusing it with when a report was received |
| Valid time | Period to which a forecast, alert, or measurement applies | Treating a forecast as an observation of the present |
| Publication time | When the originator released the information | Assuming all users could act at that moment |
| Receipt time | When the organisation received the item | Using the vendor's publication time instead |
| Detection time | When the system or analyst first identified the defined risk | Using a retrospective label added later |
| Acknowledgement time | When an accountable recipient accepted the alert | Treating delivery to an unattended queue as acknowledgement |
| Actionable warning time | When relevant evidence reached an empowered decision maker with a feasible action | Claiming lead time from a notification nobody saw |
| Impact time | Start of the defined business consequence | Substituting article publication time |
| Lead time | Impact time minus the chosen warning timestamp, in hours or days | Omitting which timestamps define it |
| Response duration | Time required for authorization and execution of an action | Counting transit but omitting approval or inspection |
| Decision deadline | Latest time at which the chosen action can achieve its specified outcome | Assuming a deadline applies to every response option |
| Time-to-survive | Duration the network can sustain demand without the disrupted supply, under stated assumptions | Treating a single inventory number as guaranteed survival |
| Time-to-recover | Time to restore a specified level of capability after disruption | Assuming full recovery and first partial output are identical |
| Precision | True-positive alerts divided by all evaluated alerts | Calling it recall or assuming it transfers to a new base rate |
| Recall | Detected positive events divided by all eligible positive events | Ignoring missed events outside the feed |
| False-positive rate | False-positive negative opportunities divided by all negative opportunities | Dividing false positives by all alerts |
| False negative | A positive opportunity without a timely qualifying warning | Classifying an intervention-prevented event as negative automatically |
| Base rate | Share of defined opportunities that are positive in a population and horizon | Using a per-shipment rate as a per-supplier-week rate |
| Threshold | Decision rule or cutoff for alerting or action | Treating it as universal across loss sizes and response costs |
| Calibration | Agreement between forecast probability and observed frequency over comparable cases | Equating ranking accuracy with probability accuracy |
| Alert fatigue | Reduced effective response caused by excessive or low-value notifications | Solving it solely by hiding all borderline alerts |
| Deduplication | Grouping observations that refer to the same event | Deleting independent evidence along with duplicate notifications |
| Coverage | Fraction of the eligible exposure population for which a feed or mapping is available | Inferring coverage from the count of alerts |
| Freshness | Age of information relative to its update expectation | Treating an unchanged stale record as a fresh negative |
| Expected loss | Probability-weighted loss over specified outcomes, in currency per decision | Presenting it as a guaranteed realized loss |
| Value of information | Improvement in expected decision value caused by information, net of its cost when specified | Counting gross loss as benefit without considering available actions |
| Forecast | Estimate of a future state or probability | Treating it as a contractual commitment |
| Plan | Chosen intended operating action | Assuming it is already executed |
| Target | Desired performance level | Treating it as a forecast or guarantee |
| Commitment | A party's agreed obligation, subject to contract and practical constraints | Treating an informal estimate as a binding promise |

## 5. The mechanics, step by step

### Step 1 — Define the business decision

**Input:** product priorities, service obligations, current plans, and decision rights.
**Output:** a decision statement with an owner, scope, horizon, and consequence.

Start with a decision such as reserving alternate freight for an exposed component.
Name the component and the plant that consumes it.
Specify whether the concern is a late order, a line stop, or a customer miss.
State the earliest decision deadline for each feasible action.
A monitor should not be purchased solely because it produces attractive maps.

Separate a decision to investigate from a decision to spend.
A planner may investigate a possible supplier delay within existing authority.
Expediting, reserving external capacity, and changing customer allocation may require different approvals.
The same evidence can support one action but not another.

Ask what happens if no warning arrives.
The baseline might be routine expediting after the promised ship date is missed.
The baseline is not necessarily complete inaction.
Economic comparisons must use the actual best available baseline process.

### Step 2 — Define the prediction target and opportunity

**Input:** the decision statement and available history.
**Output:** a target event, unit of analysis, horizon, and eligibility rules.

For example, define a positive opportunity as an exposed supplier-site-week in which a component shipment becomes at least five days late within the following fourteen days.
The threshold, time window, and unit are business choices, not universal standards.
Exclude supplier-site-weeks without relevant open demand if the intended decision concerns only open orders.
Record exclusions before seeing the outcomes.

A different target could be the first production shortfall exceeding one shift.
That target incorporates inventory and production schedules as well as supplier lateness.
A model trained for late shipments must not be presented as a calibrated probability of a plant shutdown without validation for that shutdown target.

Decide whether an event can span multiple weeks.
If it can, avoid giving one prolonged incident many more chances to be detected than a short incident without stating that design.
Use incident-level recall for incidents and opportunity-level false-positive rate for opportunities only when their denominators and relationship are clearly described.
For the worked confusion matrix below, every eligible opportunity has exactly one final binary label.

Freeze the prediction horizon and the evaluation cut-off.
A case not yet old enough to observe its complete outcome window is censored, not automatically negative.
Include late-arriving data only if it would have been available at the decision time for the simulated warning.

### Step 3 — Map the network and its exposure

**Input:** supplier and site master, bills of material, order and shipment records, lanes, and inventory.
**Output:** a dated exposure graph with confidence and known gaps.

Use site identifiers rather than supplier names alone.
Link the legal entity, physical facility, manufactured item, and customer plant where possible.
A corporate office address is not a substitute for the factory location.
If a tier-two source is known only at country level, store that uncertainty instead of inventing coordinates.

Link shipments to the actual origin, departure port, transshipment point, and destination.
A purchased lane is not necessarily the route taken by every shipment.
If the route changes, preserve the prior route and its valid dates.
Mark unconfirmed routes as planned rather than observed.

Translate the hazard area into exposure candidates using appropriate geometry and time.
A location outside a warning polygon can still be affected by a road closure or shared power network.
A location inside the polygon may not experience the severe condition.
A geospatial overlap is a screening criterion, not a conclusion that an outage occurred.

Trace the affected item through the bill of material.
Check substitution, approved alternatives, inventory, outstanding orders, and customer allocation rules.
Inventory at one plant may not be transferable before another plant's deadline.
Use a time-phased view of supply and demand rather than adding all inventory in a region without timing.

### Step 4 — Specify the supplier signal contract

**Input:** supplier commitments, historical delivery patterns, and communication channels.
**Output:** structured supplier observations with accountable ownership.

Useful observations include revised confirmed production dates, short shipments, quality holds, capacity restrictions, and changes in raw-material availability.
A missed milestone can be leading evidence if it precedes the customer impact.
A milestone is only useful if its definition and expected timing are agreed.

Record the supplier site, item, order, source contact or system, reported time, effective time, and next expected update.
Keep both the prior commitment and the revision.
A revised date may reflect an improved estimate rather than a new physical delay.
Repeated optimistic revisions may nevertheless indicate systematic reporting risk.

Distinguish silence from an explicit all-clear.
If a supplier has not reported by the agreed deadline, create a data-quality or contact task.
Do not infer normal output from missing messages.
When a supplier gives a verbal update, record the content, attribution, and time without converting an estimate into a contractual promise.

Give suppliers a low-friction way to report uncertainty.
A reporting process that penalizes every early warning may encourage concealment until the deadline is missed.
Agree escalation contacts and out-of-hours coverage for critical dependencies.
Verify authority and avoid circulating confidential capacity information beyond the permitted audience.

### Step 5 — Specify the port and transport signal contract

**Input:** shipment milestones, carrier schedules, port data, and booked service requirements.
**Output:** a shipment-specific estimate of delay and a route-specific exposure record.

Possible signals include vessel arrival changes, berth congestion, terminal dwell, gate delays, blank sailings, and missed transshipment connections.
Each describes a different part of the journey.
A port-wide average does not establish the delay of a particular container.
A vessel estimate can change before a berth is assigned.

Capture the milestone type and whether the time is planned, estimated, or actual.
Do not treat an estimated arrival as an actual discharge.
A vessel's arrival is not the same as customs release, truck pickup, warehouse receipt, or usable stock after inspection.
Each handoff may consume the remaining response window.

A broad congestion indicator can trigger a check of exposed shipments.
For action, reconcile it with carrier booking, terminal status, and customer-required dates where available.
If the vessel has already been diverted, an alert about the original port may be irrelevant to that shipment.
If multiple carriers relay the same port announcement, record their common origin.

Review the time zone and business calendar for cutoffs.
A booking deadline on a weekend may require action during the previous working day.
Use actual carrier and customs rules for the lane; no generic transit assumption guarantees release.

### Step 6 — Specify the weather signal contract

**Input:** authoritative forecasts and warnings, dated site and route locations, and operating thresholds.
**Output:** hazard exposure candidates with validity periods and uncertainty.

Different weather products have different meanings.
A forecast is a prediction; an observation describes a measured condition; an official warning describes a specified hazard and area under the issuing authority's definitions.
Do not translate an alert category into a guaranteed outage probability.

Store the issue time, valid period, issuing body, hazard type, area, severity wording, and update or cancellation reference when available.
Preserve the actual text and geometry needed for audit, subject to source terms.
When a warning changes, update the case while retaining the earlier version that drove the prior decision.

Compare the hazard to site vulnerability.
Flood elevation, backup power, employee safety, road access, and upstream utility dependencies can matter more than the name of the city.
A facility may be closed as a precaution without suffering physical damage.
Worker safety and emergency instructions take precedence over a cost optimization model.

The U.S. National Weather Service API is one official source for U.S. forecasts, observations, and alerts.
Its documentation describes update and geolocation details; coverage and product semantics are not a global weather standard.
For other regions, use the relevant local meteorological authority and verify coverage.
If a provider's alert feed stops updating, escalate the feed-health failure separately from the absence of a weather alert.

### Step 7 — Specify the news signal contract

**Input:** licensed news, official statements, and source metadata.
**Output:** attributed, deduplicated incident candidates awaiting verification.

An article can report an event before the supplier's next scheduled update.
It can also be outdated, speculative, translated inaccurately, or about a similarly named company.
Capture the headline, publication time, original source, language, relevant quotation, and link or permitted record.
Separate an article's publication time from the event time it reports.

Resolve the company and site before escalating to a buyer.
A company with several factories may report a fire at an unrelated location.
A closure of a distributor is not necessarily a closure of the manufacturer.
If resolution is uncertain, label it as unresolved and set an investigation task.

Trace syndicated articles to a common original report.
A count of ten publications quoting one press release is not ten independent confirmations.
Retain materially different first-hand accounts rather than discarding all sources after clustering.
Correct and retract cases when credible updates change the facts.

Respect licensing and access restrictions.
Avoid treating automated sentiment as verified event severity.
A language model's summary is a navigation aid, not evidence that an event occurred.
A reviewer needs the original supporting material and a route to challenge an entity match.

### Step 8 — Normalize time and preserve lineage

**Input:** source records with timestamps, versions, and ingestion metadata.
**Output:** an auditable as-of view and a current view.

Store timestamps with time zones or normalized to coordinated universal time, while retaining the original representation.
Record event, publication, ingestion, detection, acknowledgement, decision, and action times separately.
For forecasts, also record the validity interval.
When the event time is uncertain, keep a range and report a range of lead times.

Never overwrite an earlier forecast with a revised one in the evaluation dataset.
The later revision may contain knowledge unavailable at the decision time.
Use an as-of join: only information received before the decision timestamp can inform the historical decision replay.
An update arriving after the impact may help adjudicate the outcome but cannot count as an early warning.

Check for clock skew, missing time zones, daylight-saving transitions, and delayed batch ingestion.
A source may publish at 09:00 local time while the company receives the record at 14:00 in another zone.
Convert both before subtracting them.
The relevant delay may be human acknowledgement, not system ingestion.

### Step 9 — Combine evidence without counting it twice

**Input:** normalized observations, source lineage, event candidates, and network matches.
**Output:** incident clusters with evidence status and confidence.

Cluster by event type, physical location, time window, and affected organization.
A broad storm may have distinct consequences at several facilities.
A supplier may have two unrelated fires during the same week.
The clustering rule should not force these into one incident merely because the names match.

Distinguish primary reports, repetitions, independent observations, and contradictions.
A weather bulletin and a supplier's confirmed closure provide different information.
If the supplier's statement merely quotes the weather bulletin, the two sources still share evidence about the hazard.
An independent site-status check may change the impact estimate without making the forecast itself more accurate.

A simple rule can label evidence as unverified, corroborated, or confirmed against a defined source policy.
Document what qualifies for each state.
A confirmed event is not necessarily a confirmed impact on the buyer's orders.
Keep event confidence separate from exposure confidence and consequence confidence.

If a statistical model combines signals, validate it on cases that preserve their correlations.
Do not multiply individual likelihood ratios unless conditional independence is defensible.
Train and evaluate by time and incident groups so that related articles do not leak between train and test sets.
Check probability calibration in the deployed population and horizon.

### Step 10 — Estimate impact and the remaining option window

**Input:** incident evidence, exposed orders, available inventory, demand, recovery scenarios, and feasible actions.
**Output:** a range of impact times, losses, and action deadlines.

Estimate the earliest time the plant would run out of usable material under a stated demand and receipt plan.
Subtract planned demand over time and add only receipts that remain plausible under the scenario.
Identify inventory reserved for other orders and stock blocked by quality.
If the shortage date is a range, retain the range rather than reporting a precise hour without support.

For each response, add investigation, approval, supplier preparation, transit, customs, and inspection time as appropriate.
A partial shipment can avert part of the loss even if it cannot prevent the first shortage.
Document the quantity that can arrive by each relevant date.
The fastest possible theoretical route is not a commitment that capacity will be available.

Define actionable lead time as impact time minus the time the accountable owner receives sufficient evidence and authority to act.
For a particular response, subtract its duration to obtain the decision slack.
Positive slack means the action is possible under the stated duration assumption.
Zero slack is fragile to even a small delay; negative slack means the specified prevention objective is no longer achievable.

If impact time is uncertain, calculate earliest and latest plausible slack.
Use the conservative bound for an irreversible action only if the organisation's risk policy supports it.
If no action can prevent the first loss, consider later recovery, customer communication, or allocation instead of pretending the warning has no remaining use.

### Step 11 — Set a decision threshold and escalation path

**Input:** loss and response cost estimates, capacity constraints, authority matrix, and incident confidence.
**Output:** a documented rule for monitoring, investigation, response, and closure.

A threshold may be an explicit probability, a rule based on days of cover, or a combination of evidence and exposure.
State the level at which an analyst reviews a case separately from the level at which an executive authorizes an expensive action.
A severity label alone does not identify who must do what by when.

Give every case an owner and a backup.
Specify acknowledgement time, next review time, and escalation if the owner is unavailable.
Include a quiet-hours and holiday path for decisions whose deadlines cannot wait.
A case with no acknowledgement should not silently expire.

A typical progression is observe, verify, prepare, authorize, execute, and review.
Preparing a quote or holding an internal planning call is usually less irreversible than booking premium freight.
Record the trigger, evidence, deadline, authority, and decision at each step.
Escalate immediately for safety or legal requirements under the organisation's policies rather than waiting for an economic score.

Set conditions for de-escalation.
A revised official warning, a verified recovery milestone, or a rerouted shipment can remove the original exposure.
De-escalation should not erase the earlier decision record.
If evidence remains ambiguous at the deadline, the decision maker must choose using the remaining uncertainty and record the rationale.

### Step 12 — Evaluate the warning, not just the feed

**Input:** eligible opportunities, as-of alerts, adjudicated outcomes, decision timestamps, actions, and costs.
**Output:** performance and value by decision population and response deadline.

Create a confusion matrix at the declared opportunity grain.
A true positive is an eligible positive opportunity with a qualifying timely warning.
A false negative is a positive opportunity without such a warning.
A false positive is a negative opportunity with a qualifying warning.
A true negative is a negative opportunity without one.

Define timely before looking at the result.
A warning after the relevant action deadline can be informative for recovery but is not a timely prevention warning under that definition.
If deadlines differ by action, report performance for each action or make the chosen action explicit.
Do not count the same repeated notification as multiple independent true positives.

Measure precision as true positives divided by true positives plus false positives.
Measure recall as true positives divided by true positives plus false negatives.
Measure the false-positive rate as false positives divided by false positives plus true negatives.
Report numerators and denominators, not just percentages.
An apparently high precision computed from a handful of positives is uncertain.

Report warning lead-time distribution among true positives, including the fraction that beat each response deadline.
Also report missed events and the distribution of acknowledgement delay.
The median lead time among detected cases alone hides cases never detected.
For uncertain event times, use a range and explain the convention.

The observed result may have been changed by action.
A timely warning that led to a successful diversion can be incorrectly labeled a false alarm if the target is the prevented downstream shortage.
Maintain an intervention-independent hazard or supplier-event label where feasible, plus a separate observed outcome and action record.
Use prospective comparison or carefully designed causal analysis to estimate avoided losses; do not declare all avoided losses observed facts.

### Step 13 — Operate and improve the service

**Input:** performance review, case feedback, source availability, and business changes.
**Output:** revised data contracts, thresholds, mappings, and operating responsibilities.

Track feed availability and stale records separately from risk status.
A missing feed should produce an unknown state and a fallback assignment, not a green status.
Review exposure coverage and ambiguous entity matches as well as model accuracy.

Sample negative opportunities for adjudication.
A programme that reviews only alerted cases cannot reliably calculate missed events or false-positive rate.
Inspect a sample of closed cases for source attribution, owner acknowledgement, and correct action deadlines.

Revisit thresholds when base rates, costs, response options, or reviewer capacity change.
The same threshold may be inappropriate for a critical single-source component and a readily substitutable part.
Record the effective date and owner of any rule change so that subsequent results can be compared fairly.

Run periodic exercises on a plausible event.
A tabletop can test contact details and handoffs, but it does not prove that real premium freight capacity will be available.
Test supplier and carrier commitments separately from the alert workflow.
Use the lessons to revise the playbook and train backups.

## 6. Worked example

### Case and assumptions

Northline Devices is a fictional medical-device assembly business.
Its Ohio plant consumes connector C-47, made at a supplier site on the U.S. Gulf Coast.
The following observations, dates, probabilities, and costs are invented for instruction.
They are not forecasts or claims about an actual supplier or weather event.

The demand forecast is 1,200 connectors per day over the next week.
The production plan consumes 1,200 connectors per day if customer orders remain unchanged.
The service target is to maintain the planned production schedule without a connector-related stop.
The supplier commitment is a shipment of 6,000 conforming connectors arriving at the plant on Thursday, subject to the actual contract.
A supplier's current estimate that the shipment may slip is not itself a revised contractual commitment.
All times below are in U.S. Eastern Time, and all currency is U.S. dollars.

At Monday 08:00 the plant has 5,400 usable connectors after excluding quality-held stock.
Assume consumption at a constant 50 connectors per hour for this illustrative clock calculation.
Ignore other receipts and transfers in the base case.
The 5,400 units cover 108 hours of consumption.
Without the Thursday receipt, projected exhaustion is Friday 20:00.
This projected time is the defined business impact in the timing calculation.
A different production calendar or usable inventory figure changes the result.

The normal ocean and road route cannot deliver an alternative lot before the projected exhaustion time.
A qualified alternate supplier has 3,000 conforming connectors available.
Its quote allows air shipment and receipt within 48 hours after authorization, including the assumed inspection time.
The offer is a time-limited option, not guaranteed capacity until the supplier accepts the purchase order.
The action deadline to protect against Friday 20:00 exhaustion is therefore Wednesday 20:00.
The 3,000 units provide 60 hours of coverage at the assumed consumption rate.
They cannot cover an arbitrarily long outage.

### Signal sequence

| Time | Source and evidence | Interpretation | Next step |
|---|---|---|---|
| Monday 09:00 | National weather warning forecasts a coastal hazard during the supplier's operating window | Hazard near a relevant site; no confirmed output loss | Check supplier location and operating plan |
| Monday 11:00 | Supplier says a shutdown is possible and a firm update is due Tuesday | Relevant first-hand uncertainty, not a confirmed late shipment | Request affected lot and production status |
| Monday 12:30 | A news article quotes the same supplier statement | Repetition, not independent confirmation | Link article to existing incident |
| Tuesday 07:00 | Port feed reports congestion at the booked port | Potential lane delay, but not proof that C-47 is affected | Match booking and vessel status |
| Tuesday 08:00 | Planner receives an integrated case with item, exposure, and action option | First actionable warning under the case definition | Triage before Wednesday deadline |
| Tuesday 10:00 | Supplier reports a likely two-day production interruption for the relevant lot | More direct evidence; still uncertain until shipment and recovery | Re-estimate shipment date |
| Tuesday 12:00 | Carrier confirms the booked sailing remains available but has an estimated one-day delay | A separate transport observation, not a guaranteed arrival | Update the time-phased supply estimate |
| Tuesday 14:00 | Operations and procurement review the alternate quote | Decision meeting with authority and remaining slack | Choose whether to authorize |

The weather warning was published Monday 09:00.
The accountable planner received the actionable case Tuesday 08:00.
Impact is projected for Friday 20:00.
The publication-to-impact interval is 107 hours.
The actionable lead time is 84 hours.
The 48-hour alternate response leaves 36 hours of decision slack at Tuesday 08:00.
At Tuesday 14:00 the remaining lead time is 78 hours and the slack is 30 hours.
The alternate order can still prevent the initial shortage under the assumed timing and quantity.

An alert received Thursday 12:00 would give 32 hours of lead time.
Its slack would be minus 16 hours for the 48-hour action.
It could still support later recovery or customer communication, but not the initial prevention objective.
The relevant timestamp is when the decision owner can actually act, not when an article appeared.

### Triage and response decision

The case team estimates a 30% probability that the original shipment will be too late to prevent a production loss, conditional on the current evidence and no alternate action.
This probability is an illustrative judgment for this single decision, not an empirically calibrated model output.
If the shipment is too late, the estimated incremental loss is $180,000 without the alternate action.
The amount represents contribution and other incremental costs under one stated accounting boundary; it is not revenue plus contribution.

The alternate action costs $18,000 regardless of whether the original shipment would have been late.
If the damaging late-shipment scenario occurs, the alternate lot is assumed to reduce the loss from $180,000 to $35,000.
The residual $35,000 reflects remaining delay and downstream recovery costs after the limited alternate lot is used.
If the damaging scenario does not occur, assume the alternate lot has no additional incremental loss beyond the $18,000 action cost.
This assumption needs checking for excess inventory, expiry, and contract effects.

Without action, expected incremental loss is:

`0.30 × $180,000 + 0.70 × $0 = $54,000`.

With action, expected incremental loss is:

`$18,000 + 0.30 × $35,000 + 0.70 × $0 = $28,500`.

Expected benefit of acting rather than not acting is `$54,000 − $28,500 = $25,500`.
Under these assumptions the decision is to authorize the alternate order before the deadline.
A realized loss may differ greatly from the expected loss.
A confirmed order, approved quality status, available flight, and cash authority are prerequisites to execution.

The break-even probability solves `p × ($180,000 − $35,000) = $18,000`.
Thus `p = $18,000 / $145,000 = 12.41%` approximately.
Above this probability, the action has lower expected cost under the fixed assumptions.
Below it, waiting or taking another action may be preferable.
At exactly the threshold the two options have equal expected cost before considering risk appetite and other constraints.

If response cost rises to $30,000, the threshold becomes approximately 20.69%.
If the alternate lot can avert only $80,000 of loss, the original $18,000 action cost implies a 22.5% threshold.
If it cannot arrive in time, the assumed $145,000 reduction is not available and must be re-estimated.
The probability and avoidable loss are conditional on the action and current evidence; neither should be treated as fixed across all scenarios.

### Warning programme valuation

A single-case decision benefit is not the annual value of the monitoring system.
To value the information, compare a policy that uses the warning with the best baseline policy that lacks that information.
For an illustrative cohort, suppose there are 1,000 eligible supplier-site-weeks per year and 50 have the defined adverse event.
Assume a warning policy identifies 40 of the 50 positive opportunities in time and flags 95 of the 950 negative opportunities.
The remaining counts are 10 missed positives and 855 correctly unflagged negatives.
The assumed 5% base rate, 80% recall, and 10% false-positive rate apply only to this hypothetical cohort and target.

The programme generates 135 flagged opportunities: 40 true positives and 95 false positives.
Its precision is `40 / 135 = 29.63%`.
Its recall is `40 / 50 = 80%`.
Its false-positive rate is `95 / 950 = 10%`.
The low precision despite 80% recall illustrates the effect of a low base rate.
It does not by itself prove that the programme destroys value.

Assume each timely true-positive warning permits an action that avoids $20,000 of loss before action cost.
Assume every flag triggers an action costing $3,000, including flags later classified as false positives.
Assume no action without a flag, no benefit in negative opportunities, and no capacity constraint.
Gross annual avoided loss is `40 × $20,000 = $800,000`.
Annual action cost is `135 × $3,000 = $405,000`.
If annual monitoring and triage cost is $150,000 and is not already included above, net annual expected value is `$245,000`.
The result is conditional on all assumptions and assumes comparable baseline outcomes and no interactions among cases.

If the action cost is $7,000 per flag instead, the action cost is $945,000.
Net annual value is `$800,000 − $945,000 − $150,000 = −$295,000`.
A less expensive verification step, a different threshold, or a different action could be worthwhile, but none is assumed to fix this result automatically.
If the 40 warnings are timely but the action cannot actually prevent the stated loss, the gross benefit estimate also fails.
Avoid counting the same avoided production stop separately as lost sales, lost revenue, and lost contribution.

### Review and close

After execution, record whether the alternate supplier accepted the order and when usable stock arrived.
Record the original supplier's actual delivery date and the plant's observed production impact.
Keep the forecast probability, decision time, and information available then unchanged for evaluation.
A successful alternate shipment does not prove the original shipment would have caused the full estimated loss.
The unobserved no-action outcome requires a counterfactual estimate, not an assertion.

Close the case only after ownership, customer communication, open purchase orders, and any excess inventory are resolved.
Review whether the actionable warning beat the deadline and whether response times matched assumptions.
Record what evidence would have changed the decision.
The purpose is a better next decision, not retrospective blame for an uncertain forecast.

## 7. Code, query, or calculation

The following self-contained Python example calculates the hypothetical cohort metrics and expected annual value above.
It requires only the Python standard library and does not call external data services.
Counts represent mutually exclusive eligible supplier-site-week opportunities with fully observed labels.
The alert is assumed timely for the selected action.
The benefit applies only to true positives and the action cost applies to every alert.
No causal effect is estimated by the code; the financial inputs are assumptions.

```python
# Python 3; illustrative annual supplier-site-week cohort.
positive = 50
negative = 950
true_positive = 40
false_positive = 95
benefit_per_true_positive = 20_000.0  # USD, gross avoided loss
cost_per_alert = 3_000.0              # USD, response cost
annual_monitoring_cost = 150_000.0    # USD, excludes response cost

false_negative = positive - true_positive
true_negative = negative - false_positive
alerts = true_positive + false_positive
precision = true_positive / alerts if alerts else float('nan')
recall = true_positive / positive if positive else float('nan')
false_positive_rate = false_positive / negative if negative else float('nan')
gross_benefit = true_positive * benefit_per_true_positive
response_cost = alerts * cost_per_alert
net_value = gross_benefit - response_cost - annual_monitoring_cost

print(f'TP={true_positive} FP={false_positive} '
      f'FN={false_negative} TN={true_negative}')
print(f'Precision={precision:.2%} Recall={recall:.2%} '
      f'FPR={false_positive_rate:.2%}')
print(f'Gross benefit=${gross_benefit:,.0f} '
      f'Response cost=${response_cost:,.0f}')
print(f'Annual net value=${net_value:,.0f}')
```

Expected output:

```text
TP=40 FP=95 FN=10 TN=855
Precision=29.63% Recall=80.00% FPR=10.00%
Gross benefit=$800,000 Response cost=$405,000
Annual net value=$245,000
```

If there are no alerts, precision is undefined rather than zero.
If there are no positive opportunities, recall is undefined.
If there are no negative opportunities, the false-positive rate is undefined.
The example prints `nan` for these undefined quantities.
The arithmetic assumes integer counts that are consistent with their denominators.
A real production pipeline must validate input counts, decision eligibility, and outcome maturity before calculation.

### Read the output as a decision report

The 135 alerts are a workload count at the opportunity grain.
They are not necessarily 135 independent storms or 135 different suppliers.
The 95 false positives are costly because the example applies the same $3,000 response to every alert.
In practice, a cheap investigation may precede a more expensive intervention.
A two-stage workflow needs separate probabilities and costs for each branch.

The $800,000 gross benefit is a model assumption applied to 40 cases.
It is not a measured sum of avoided losses.
Each case may differ in severity, response effectiveness, and remaining lead time.
For a realistic portfolio, estimate case-level incremental value and sum across cases without overlapping loss attribution.

The $150,000 monitoring cost should include only costs incremental to the programme and not already in the per-alert action cost.
If analysts would have investigated some events without the system, deduct only incremental workload and measure incremental benefit against that baseline.
If a feed subscription serves other business purposes, explicitly allocate or exclude the relevant share rather than counting the full cost twice.

### Manual threshold calculation

Let `p` be the probability of the adverse event under no action.
Let `L0` be its loss without action and `L1` its residual loss with action.
Let `C` be the cost of the action in either event state.
Assume for this calculation that the action creates no additional loss in the no-event state beyond `C`.

Expected cost without action is `p × L0`.
Expected cost with action is `C + p × L1`.
The action is economically preferred when `p × (L0 − L1) > C`.
If `L0 − L1` is positive, the break-even probability is `C / (L0 − L1)`.
If the action cannot reduce the event loss, there is no positive expected-loss saving to cover a positive cost under these assumptions.
If the action has a salvage value, an expiry loss, or a cost that depends on the event state, add those terms to the appropriate branches.

### A two-stage decision

Suppose a $500 supplier investigation takes four hours and produces a more informative update before the response deadline.
The investigation does not itself prevent loss.
Its value is the expected improvement in the subsequent decision compared with deciding now, less $500.
To calculate it, specify the probability of each possible update, the posterior event probability for each update, and the best action in each branch.
Do not add the investigation's gross value to the full action benefit if that benefit already assumes the investigation occurred.
If the four-hour delay misses the booking deadline, the update can destroy the option rather than improve it.

### Compare base rates

Hold recall at 80% and false-positive rate at 10% for illustration.
At a 5% base rate across 1,000 opportunities, the matrix is 40 true positives and 95 false positives.
At a 20% base rate across 1,000 opportunities, it is 160 true positives and 80 false positives.
Precision increases from 29.63% to 66.67%, although recall and false-positive rate have not changed.
This arithmetic does not prove that the same model will retain those rates in a new population.
Population shift can change sensitivity, false-positive rate, calibration, and achievable lead time.

### Interpreting time

Suppose impact is projected for Friday 20:00 and the actionable warning arrives Tuesday 08:00.
From Tuesday 08:00 to Friday 08:00 is 72 hours.
From Friday 08:00 to Friday 20:00 is another 12 hours.
The total is 84 hours.
Subtracting the 48-hour response leaves 36 hours of slack.
This simple calculation assumes continuous elapsed hours, not working hours.
Where weekends or carrier cutoffs matter, use the actual operating calendar and its exceptions.

## 8. Data, standards, and regulatory context

### A minimum data contract

A signal record needs an identifier, source, source version, original publication time, receipt time, and event or forecast-valid time.
It needs the event type, status, scope, location, and supporting text or reference.
A supplier record should identify the site and affected item when known.
A shipment record should identify the milestone and distinguish planned, estimated, and actual values.
A case record should link the observations to a decision owner, deadline, action, and outcome.

Make missingness explicit.
An unknown site, unverified article, or delayed feed is not a negative event.
Retain correction and cancellation messages and associate them with the earlier version.
Where a source prohibits redistribution, store and display only permitted content and metadata.

### Weather and public alerts

The U.S. National Weather Service documents an application programming interface for forecasts, alerts, and observations.
It describes a required user-agent identifier, rate limits, geographic query behavior, and product-specific caveats.
The interface supports Common Alerting Protocol (CAP) among its response formats.
A CAP message can encode alert information, including update and cancellation relationships under the OASIS specification.
CAP is a message format, not proof of supply-chain exposure or a universal operating threshold.

Weather alert coverage is jurisdiction-specific.
A U.S. feed should not be represented as complete worldwide coverage.
Use the relevant issuing authority and check which hazards and regions its products cover.
Archived observations and initial forecasts may need separate retention arrangements because live services can have limited historical windows.

### Business continuity

ISO 22301:2019 specifies requirements for a business continuity management system.
It is not a statutory obligation for every organisation merely because the standard exists.
A contract, regulation, or certification commitment may impose separate requirements in a particular context.
The standard does not prescribe the illustrative 30% probability, the two-day shipping assumption, or a universal alert threshold used here.
An alerting dashboard alone is not evidence of certification or effective continuity management.

### Personal, commercial, and safety information

Supplier contacts, precise facilities, shipment details, and customer allocations can be commercially sensitive.
Apply access controls, retention policies, and relevant legal obligations to the actual data and jurisdictions.
Do not publish a supplier's unverified financial distress or a site-level vulnerability indiscriminately.
Involve legal and privacy teams for applicable requirements; this lesson does not provide legal advice.

Emergency and worker-safety instructions are governed by the responsible authority and the organisation's safety procedures.
A cost-benefit threshold in this lesson is not a reason to disregard an evacuation order.
Record the safety escalation separately from any commercial recovery decision.

### Interoperability and identity

A standard message structure can reduce translation effort, but an identifier still needs a trustworthy mapping to the relevant entity and site.
Maintain historical mappings when suppliers change names, facilities change operators, or ports change terminal identifiers.
A single organization identifier does not prove which factory made a particular lot.
Where actual shipment identifiers are absent, mark the linkage as provisional.

Data latency is part of the service contract.
Agree expected update frequency, maintenance windows, failure notifications, and correction behavior with each source.
Monitor publication-to-receipt delay separately from receipt-to-owner delay.
No one service-level target proves that a particular alert is useful before its business deadline.

### Evaluation governance

Fix the target definition and cohort before computing precision or recall.
Keep an as-of feature snapshot and a later outcome record with separate timestamps.
An outcome adjudicator should know the definitions and record uncertain cases rather than forcing all evidence into a binary label.
If the outcome window has not elapsed, retain the case as immature.

Maintain a model or rule inventory that records owners, training populations, effective dates, known limits, and rollback criteria.
For probability outputs, check calibration in relevant subgroups and time periods.
Use confidence intervals or uncertainty ranges when the number of events is small.
A stable overall metric can hide poor coverage for a critical region or supplier tier.

## 9. Industry practice and case evidence

### Port of Los Angeles: operational visibility

The Port of Los Angeles announced its Port Optimizer Control Tower in February 2021.
The port described a data tool for tracking cargo and metrics including truck turn time.
Its February 2022 announcement described additional Port Optimizer components and terminal data coverage.
These are primary-source descriptions of the port's own service, not independent estimates of avoided disruption losses.

The case shows why a port-level signal can be operationally useful: it can direct planners toward conditions that affect a route.
It does not demonstrate that a particular buyer's container will be delayed by a particular number of days.
[INFERENCE] A buyer would still need to link a port metric to its booking, terminal, handoff, and required receipt date before deciding whether to divert cargo.
[INFERENCE] The usefulness of such visibility depends on update latency and on whether a feasible alternative exists before the relevant deadline.

A truck turn-time metric measures a different process from vessel berth waiting.
A fast gate does not prove the container has been discharged or released by customs.
Likewise, an increase in aggregate traffic does not prove every cargo owner faces the same delay.
Use the source's precise definition and time window before placing a threshold on a metric.

### Suez Canal blockage: disruption and propagation

A.P. Moller–Maersk's 2021 annual report discusses the March 2021 Suez Canal blockage, the queue of vessels, and related disruption.
The company's report is a primary account of its operating context, not an independent counterfactual valuation of any alert programme.
The blockage illustrates that a transportation incident can affect flows beyond the immediate location and that recovery can outlast the initial event.

[INFERENCE] A buyer with cargo on affected services would need vessel and booking-level exposure, not merely an alert that the canal was blocked.
[INFERENCE] A useful response could depend on whether cargo was already at sea, whether diversion was feasible, and the stock available at destination.
The report does not establish that an earlier warning could have prevented every affected delay.
Do not infer the fictional Northline probabilities or action effectiveness from this case.

### Supplier reporting in practice

A supplier's production milestone can arrive earlier than a late-delivery event.
Its practical value depends on an agreed definition, a reliable contact, and a willingness to report uncertainty.
A reported schedule slip is stronger evidence for a specific order than a generic news sentiment score, but it may still be revised.
A buyer should compare commitment revisions with actual outcomes over time.

[INFERENCE] An incentive that punishes every provisional warning may encourage suppliers to wait until the delay is certain.
A better operating arrangement can reward timely, accurate updates while retaining accountability for actual commitments.
This is a design hypothesis to test with suppliers, not a claim about a named firm's policy.

### Weather and protective action

Official weather products can provide early hazard information for exposed locations.
The National Weather Service publishes forecasts, observations, and alert products with distinct meanings and technical constraints.
A weather product alone does not state whether a supplier will close or whether a given order will be late.
The case team must combine it with facility-specific exposure and operating evidence.

[INFERENCE] A storm warning may justify a low-cost readiness call before it justifies a premium freight booking.
Whether the more expensive action is warranted depends on the action deadline, loss, probability, and available alternative.
These decision rules are illustrative, not official weather-service guidance.

### News corroboration

News can reveal a developing incident before a scheduled supplier report.
It can also repeat one source many times or identify a company without identifying the affected site.
Source lineage, entity matching, and a path to correction are therefore operating requirements, not optional decorations.

[INFERENCE] A control tower that sends separate alerts for every syndicated story can inflate apparent event counts and overwhelm reviewers.
The appropriate remedy is incident clustering with preserved evidence, not indiscriminate deletion of all but one source.
A human reviewer should be able to inspect the original material and record contradictory evidence.

### What the evidence does and does not support

The cited port and carrier sources establish that these named organisations described the stated tools or disruptions.
They do not validate a universal early-warning return on investment.
The worked example's counts, probability, costs, and lead times are constructed assumptions.
To generalize, evaluate a comparable population, response policy, and observation period, including missed events and interventions.

### Translating a port indicator into an order decision

Consider an importer whose port dashboard shows median container dwell increasing from four days to seven days.
These numbers are a separate illustration, not observations from the cited port case.
The median is calculated over containers completed in a given reporting period.
It describes the middle completed container, not necessarily the current status of containers still waiting.
If the slowest containers remain uncompleted, the reported median can improve while some individual shipments are still stuck.

The importer has three open orders at the same port.
Order A is already released and scheduled for truck collection tomorrow.
Order B has been discharged but is awaiting inspection.
Order C is still aboard a vessel waiting for a berth.
The same seven-day port statistic should not be added mechanically to all three delivery estimates.

For Order A, confirm the pickup appointment, chassis availability, and trucking commitment.
For Order B, seek an inspection estimate, but do not treat it as a guaranteed release.
For Order C, obtain the carrier's vessel and terminal milestone estimates.
Each order has a different decision deadline and a different potential response.

Suppose the customer's required receipt for Order C is ten days from now.
The carrier forecasts discharge in three days, inspection in two more days, and inland delivery in two further days.
The forecast total is seven days, leaving three days of nominal slack.
That arithmetic assumes the stages are sequential and the quoted days are comparable calendar days.
It also assumes no additional customs hold, missed appointment, or receiving delay.

If the terminal estimate moves back by four days, the same simple plan now arrives in eleven days.
The nominal slack becomes negative one day.
That change may justify seeking an alternative, depending on its cost and actual availability.
It does not by itself prove a customer failure; the later stages may improve or the customer may accept a revised delivery.

Store the prior and revised estimates and the time each was available.
Ask whether the decision maker could have acted before the alternative booking deadline.
On retrospective review, do not substitute the final discharge time for the estimate available when the booking decision was made.
Compare the eventual outcome with the decision made under the evidence actually available then.

### Converting supplier milestones into warning logic

Assume a supplier promises three milestones for a lot: materials released, production completed, and shipment dispatched.
The milestones are illustrative contractual or operational checkpoints, not universal supplier standards.
Each has a planned date, actual date if complete, and latest reported forecast if incomplete.
Keep the difference between a forecast and an accepted commitment visible.

An analyst can calculate milestone variance as actual completion time minus planned completion time.
For an unfinished milestone, calculate forecast variance separately as latest reported forecast minus planned completion time.
Do not mix completed actual variances with incomplete forecasts in a distribution without labeling the difference.
If an actual completion date arrives late through a batch feed, preserve both its event and receipt times.

A rule might trigger an investigation when the forecast production-completion date slips by more than one day.
The rule must also specify which products, sites, and orders are eligible.
It should suppress duplicate notifications for the same unchanged incident while retaining every meaningful revision.
It should not suppress a separate incident merely because the same supplier is involved.

If the supplier reports a two-day slip but has one day of recoverable capacity, the delivery consequence is not automatically two days.
Ask for the feasible recovery plan, including overtime, material availability, quality release, and transport cutoffs.
The recovery plan is a forecast unless it is backed by an enforceable or accepted commitment.
Compare the remaining time with the buyer's own inventory and order requirements.

The owner of the rule should review both caught and missed milestone failures.
A missed actual shipment may be preceded by a milestone that was never reported.
The remedy could be better reporting coverage rather than a lower numeric threshold.
Conversely, frequent harmless one-day revisions may justify a threshold change if the cost and loss trade-off support it.

### Revising probabilities without false precision

Suppose a planner initially assigns a 10% chance that an order will be late by at least five days.
The value is an illustrative prior for a defined order population and horizon.
A supplier update arrives indicating a relevant production interruption.
The decision team now estimates 30%, as in the worked case, but must explain the basis for that revision.
The estimate could come from a validated conditional-frequency model, a documented expert judgment, or a structured combination.

An expert estimate should record the reference class and major evidence.
The reference class could be past interruptions at comparable sites during comparable operating seasons.
If only two historical cases exist, acknowledge that the numerical probability is highly uncertain.
If the case is unprecedented, compare a range of plausible probabilities with the action threshold.
This sensitivity analysis is more honest than presenting a point estimate to two decimal places.

For example, suppose the plausible probability range is 10% to 35% and the break-even threshold is 12.41%.
The range crosses the threshold.
Expected-value preference therefore changes within the plausible range.
A decision maker may investigate further if that investigation fits within the deadline and has positive expected value.
Alternatively, a documented risk constraint may justify acting at the lower end, but that is a policy decision rather than a mathematical implication of the point estimate.

After a case closes, compare the prediction with the outcome without assuming one observation validates or invalidates the probability.
A 30% probability permits the event not to occur in most comparable cases.
Calibration requires a collection of comparable forecasts and independently adjudicated outcomes.
Check whether cases assigned approximately 30% experience the defined event at roughly that rate, with uncertainty due to sample size.
Do not include future information in the stored probability.

### Handling limited review capacity

Suppose the desk can investigate twenty new cases per shift but the detection service generates forty candidates.
An alert threshold cannot be chosen independently of this capacity.
Rank cases by expected incremental decision value where credible estimates exist.
Also preserve mandatory safety, regulatory, and contractual escalation channels.

A useful ranking estimate considers event probability, avoidable loss, response effectiveness, and investigation cost.
It should also consider the time remaining before the next feasible action expires.
A large loss next month may be less urgent than a smaller preventable loss with a deadline in one hour.
The final prioritization rule should be explicit and reviewed for systematic exclusion of poorly mapped suppliers.

If reviewers do not have capacity for all forty candidates, log the twenty not reviewed as unreviewed.
Do not silently call them negative cases.
Sample unreviewed candidates and low-scored opportunities to estimate what the triage process misses.
Measure delay from queue entry to acknowledgement, including cases that expire in the queue.

An analyst may reduce workload by grouping related alerts into a single case.
Grouping can free time without discarding distinct supporting observations.
Another option is a low-cost supplier confirmation before committing a costly response.
Whether that step improves outcomes depends on its accuracy, cost, and delay.
Evaluate those trade-offs rather than assuming more automation always improves performance.

### A staged implementation plan

In the first stage, select one product family, one owner, and a limited set of critical suppliers.
Document the business decision, response alternatives, and exposure mappings.
Use existing supplier contacts and available official hazard information where permitted.
The objective is to learn whether signals can be linked to decisions, not to claim enterprise-wide coverage.

In the second stage, run the case workflow alongside the existing process.
Record alerts and decisions even when no premium action is taken.
Capture the manual baseline's detection time and actions without interfering with safety procedures.
Review a sample of opportunities with no alerts so that missed events are visible.

In the third stage, compare observed workload, acknowledgement delay, and action feasibility with the initial assumptions.
Refine the entity mappings and supplier reporting contracts before widening the population.
Use a prospective comparison where feasible to estimate incremental benefit.
If the event count is low, present a range and the observation period rather than annualizing a small result without qualification.

In the fourth stage, expand only where a decision owner and feasible response are identified.
Different products may need different alert thresholds because their losses and substitutes differ.
Different geographies may require different official weather providers and languages.
Maintain a single common case vocabulary while allowing source-specific semantics.

Finally, rehearse fallback operation.
Test whether an unstaffed mailbox, unavailable feed, or stale mapping is detected by a named person.
Test whether a backup approver can act before the deadline.
Review exercise results separately from evidence that actual capacity has been reserved or delivered.

### Questions for an operating review

Which material risks are not covered by any available feed?
Which mapped sites have uncertain coordinates or unresolved upstream dependencies?
How many warnings arrived after the applicable action deadline?
How many cases were not acknowledged before their response option expired?
How many negative opportunities were independently sampled?
How many severe events occurred without any qualifying warning?
How many false alarms arose from incorrect entity matching rather than an uncertain forecast?
How many repeated notifications came from one incident?
What did investigators do when primary and secondary sources contradicted each other?
Were response costs and available alternative capacity checked at the time of decision?
Which actions prevented observed shortages and therefore require counterfactual evaluation?
Which policy change would most improve the next decision, and who owns it?

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Stale supplier data | No update after agreed deadline | Missing report treated as normal operation | Missed exposure | Mark unknown, contact supplier, escalate feed health |
| Corporate address used as factory | Many sites map to one headquarters | Master data describes legal entity rather than physical facility | Hazard mapped to wrong location | Maintain site-level verified coordinates and validity dates |
| Duplicate news | Many alerts share one quoted statement | Syndication counted as independent confirmation | Overconfidence and fatigue | Cluster by incident and retain original source lineage |
| Incorrect entity match | Article names a similarly named company | Name matching ignores location and legal identity | Unnecessary action or reputational harm | Require disambiguating identifiers and human review |
| Hazard equals loss | Every weather warning becomes a plant-stop alert | Exposure and vulnerability are skipped | Excess premium actions | Verify affected site, capacity, and orders |
| Port average equals shipment delay | Aggregate dwell metric drives every order estimate | Shipment-specific milestones are absent | Incorrect diversion or missed delivery | Match carrier booking and actual milestones |
| Late alert counted as timely | High recall with little decision slack | Evaluation uses publication rather than actionable time | False confidence in prevention | Evaluate against response-specific deadlines |
| False-negative denominator absent | Report contains only alerted cases | Negative and missed opportunities never sampled | Recall cannot be estimated | Build eligible cohort and independently adjudicate outcomes |
| Intervention changes label | Effective diversions later count as false alarms | Observed downstream shortage is prevented | Valuable alerts appear inaccurate | Separate hazard label, action, and observed outcome |
| Outcome leakage | Historical score uses later revised status | As-of snapshots are missing | Overstated backtest performance | Preserve versions and use receipt-time joins |
| Correlated inputs | Model confidence jumps with syndicated stories | Evidence is multiplied as if independent | Inflated probability | Group sources and validate with incident-held-out data |
| No response owner | Alert sits unacknowledged in queue | Notification is mistaken for action | Deadline passes | Assign primary, backup, acknowledgement and escalation time |
| Misstated inventory | Safety stock includes held or reserved lots | Usable stock and allocations not reconciled | Shortage earlier than predicted | Time-phase usable quantities and quality status |
| Imaginary backup capacity | Alternate route is assumed available | Quote is treated as booking or commitment | Response fails in real event | Verify qualification, capacity, and acceptance |
| Double-counted benefit | Avoided revenue and contribution both included | Accounting boundaries overlap | Inflated return | Use incremental contribution and reconcile loss categories |
| Cost omitted | Monitoring cost excludes analyst time | Only software subscription counted | Misleading business case | Include incremental triage, action, and maintenance costs |
| Outdated rule | Alert volume changes after new network route | Threshold and exposure map not reviewed | Poor precision or missed exposure | Version rules and monitor performance by population |
| Feed outage looks green | No alerts when ingestion fails | Absence of data displayed as absence of risk | Unobserved disruption | Health check, unknown state, manual fallback |
| Safety overridden | Low economic score delays evacuation | Business loss rule used for safety action | Injury or legal breach | Follow authority and safety policy independently |
| Incomplete outcome | Recent case classified negative too early | Horizon has not elapsed | Underestimated positives | Mark censored until outcome window matures |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or respond |
|---|---|---|---|
| Site and item links are correct | Exposure depends on actual production location | Supplier switches plants or master data is stale | Reconcile supplier attestations, lots, and shipments |
| Timestamps are comparable | Lead time requires a common clock | Time-zone errors or delayed batches | Audit source and receipt timestamps and preserve originals |
| The target has a fixed definition | Metrics need consistent positive labels | A late order is redefined after the fact | Freeze threshold, horizon, and cohort before evaluation |
| Outcomes mature within the chosen window | A negative needs complete follow-up | Long disruption or late reporting | Mark immature cases censored and extend observation |
| An action is actually feasible | Value requires an executable alternative | Capacity is sold, unqualified, or unavailable | Test acceptance, qualification, booking and inspection |
| Action effectiveness is plausible | Avoided loss drives the calculation | Alternate supply arrives too late or is too small | Simulate quantity by date and compare exercises with actual delivery |
| Costs are incremental and nonoverlapping | Expected value must compare like with like | Contribution, revenue, and penalties are counted twice | Reconcile accounting boundaries with finance |
| The baseline is accurately specified | Information has value only relative to a decision without it | Existing manual process already detects and acts | Compare against the best feasible baseline workflow |
| Model rates transfer to deployment | Cohort metrics inform forward estimates | New region, season, supplier, or base rate | Monitor by subgroup and recalibrate prospectively |
| Observations can be grouped correctly | Duplicates distort confidence and counts | Similar names or simultaneous independent incidents | Audit clusters and allow reversible manual correction |
| An event label is observable | Confusion matrix requires adjudication | Effective action prevents the downstream shortage | Separate hazard, exposure, action, and outcome; estimate counterfactual |
| Decision makers act within service times | Actionable lead time depends on acknowledgement | Absences, holidays, or unstaffed queues | Measure actual handoff times and exercise backups |
| Consumption and inventory assumptions hold | Projected shortage depends on both | Demand spikes, quality holds, or unplanned receipts | Use time-phased scenarios and update the range |
| Sources remain available and licensed | Monitoring requires continuity of access | Rate limits, outages, or contract changes | Monitor health and establish permitted fallback sources |

The expected-value example is risk neutral and excludes financing, tax, and interaction effects unless separately stated.
A decision maker may also impose safety, contractual, or tail-risk constraints.
State such constraints explicitly rather than hiding them inside an unexplained multiplier.
A small number of rare events cannot reliably establish a precise probability for a new hazard.

## 12. Exercises

### Exercise 1 — Exposure and evidence map

**Time:** 20 minutes.
A news report says that Meridian Components has closed a plant after flooding.
Your purchase order names Meridian Components, but the supplier master has three sites in two countries.
A weather warning covers one site, and the order was last confirmed by another.

**Deliverable:** a one-page evidence and exposure map with a case owner and first investigation.
**Acceptance criteria:** distinguish the hazard, named company, physical site, item, shipment, and customer plant.
Identify at least two unresolved links and state how to verify them.
Explain why the article and warning do not alone establish an order delay.
Map the result to Learning outcome 1.

### Exercise 2 — Lead time and slack

**Time:** 15 minutes.
A signal is published Monday 06:00 UTC and received by the responsible planner Monday 12:00 UTC.
The projected shortage starts Thursday 18:00 UTC.
A qualified response takes 60 elapsed hours after authorization.

**Deliverable:** calculate publication-to-impact lead time, actionable lead time, response deadline, and slack at receipt.
**Acceptance criteria:** give all results in hours and include the calendar day and time for the deadline.
Say whether an alert received Wednesday 12:00 UTC can prevent the initial shortage with this response.
Map the result to Learning outcome 2.

### Exercise 3 — Confusion matrix

**Time:** 20 minutes.
Over 500 mature supplier-site-weeks, 40 are positive for a specified late-delivery target.
A timely alert was issued in 30 positive and 46 negative opportunities.

**Deliverable:** a four-cell matrix, precision, recall, and false-positive rate.
**Acceptance criteria:** show numerators, denominators, and percentages; distinguish alert precision from the false-positive rate.
State one limitation of the sample and define how timeliness was determined.
Map the result to Learning outcome 3.

### Exercise 4 — Action threshold

**Time:** 20 minutes.
Without action, an adverse event produces $120,000 of incremental loss.
An action costs $15,000 and reduces the event loss to $20,000.
Assume zero other incremental loss in the no-event state and an event probability of 25%.

**Deliverable:** expected cost of acting and not acting, the expected saving, and the break-even probability.
**Acceptance criteria:** include the action cost in both event states and do not count avoided revenue separately.
Explain what changes if the action cannot arrive on time.
Map the result to Learning outcome 4.

### Exercise 5 — Critique a dashboard

**Time:** 20 minutes.
A dashboard displays ten independent confirmations because ten articles repeat the same supplier statement.
It uses the final revised estimated arrival time to replay an alert supposedly issued three days earlier.
A missing weather feed is displayed as green.

**Deliverable:** identify at least three defects and propose a correction for each.
**Acceptance criteria:** address source dependence, as-of information, and missing-data state.
Name one consequence for decision quality for each defect.
Map the result to Learning outcome 5.

### Exercise 6 — Escalation design

**Time:** 30 minutes.
A critical part has two days of usable stock.
A supplier reports a possible shutdown but cannot yet quantify duration.
A premium shipment can arrive in 36 hours if authorized within the next six hours.
The buyer can spend up to $5,000; the premium option costs $28,000.

**Deliverable:** a time-stamped escalation plan through decision and closure.
**Acceptance criteria:** assign a primary and backup owner, authority, acknowledgement deadline, investigation, decision deadline, and fallback.
Distinguish preparing a quote from committing expenditure.
Include an uncertainty statement and a safety override.
Map the result to Learning outcome 6.

### Exercise 7 — Base rate and threshold

**Time:** 20 minutes.
Two regions each have 1,000 eligible opportunities.
Region A has a 2% event rate and Region B a 10% event rate.
In each, assume recall is 75% and false-positive rate is 5%.

**Deliverable:** compute the true and false positives and precision in each region.
**Acceptance criteria:** show the negative denominator and explain why precision differs.
Name two reasons the assumed common rates might not transfer to reality.
Map the result to Learning outcome 7.

### Exercise 8 — Prospective evaluation

**Time:** 30 minutes.
A buyer plans to route warnings to one team and retain the existing manual process for comparison.
Some warnings will trigger diversions that prevent observed shortages.
The team wants to claim the annual value of the programme after six months.

**Deliverable:** outline an evaluation design and a minimal decision log.
**Acceptance criteria:** specify eligible population, assignment or comparison, outcome horizon, treatment, intervention records, timeliness, and costs.
Explain how to handle prevented outcomes, contamination, incomplete follow-up, and uncertainty.
Do not equate the value of all averted losses with observed facts.
Map the result to Learning outcome 8.

## 13. Self-check answers

### Exercise 1 rubric

A strong map distinguishes the company from its three physical sites.
It links the affected site to the specific item and open order only after verification.
A contact at the supplier can confirm whether the closure affects the producing site, production lot, and shipment date.
The planner can verify the order's site and downstream inventory in internal records.
The source article and weather warning may concern the same hazard and are not automatically independent evidence of a late order.
An accountable procurement or planning owner should receive a deadline for the first follow-up.

### Exercise 2 calculation

Monday 06:00 to Thursday 18:00 is 84 elapsed hours.
Monday 12:00 to Thursday 18:00 is 78 elapsed hours.
The latest authorization for a 60-hour response is Tuesday 06:00 UTC.
At receipt, the decision slack is `78 − 60 = 18` hours.
Wednesday 12:00 is 30 hours before the projected shortage.
It is 30 hours too late for the specified 60-hour prevention response, although it may still aid recovery.
All calculations assume continuous elapsed hours and unchanged impact and response times.

### Exercise 3 calculation

True positives are 30 supplier-site-weeks.
False negatives are `40 − 30 = 10` supplier-site-weeks.
The number of negative opportunities is `500 − 40 = 460`.
False positives are 46, and true negatives are `460 − 46 = 414`.
Precision is `30 / (30 + 46) = 39.47%`.
Recall is `30 / 40 = 75%`.
False-positive rate is `46 / 460 = 10%`.
Timeliness must be tied to an action deadline fixed before evaluation.
The 40 positive observations may be too few for precise subgroup conclusions.

### Exercise 4 calculation

Without action, expected loss is `0.25 × $120,000 = $30,000`.
With action, expected loss is `$15,000 + 0.25 × $20,000 = $20,000`.
Expected saving is `$10,000` under the stated assumptions.
The event loss reduction is `$120,000 − $20,000 = $100,000`.
The break-even probability is `$15,000 / $100,000 = 15%`.
At 25% the action has lower expected cost, provided it is feasible and its effect is as assumed.
If it arrives too late, re-estimate the residual event loss and available benefit; the 15% threshold is no longer valid automatically.

### Exercise 5 rubric

Ten articles quoting one statement do not supply ten independent confirmations.
Group them into an incident while preserving source lineage and any genuinely independent reporting.
A replay that uses a revised arrival estimate before that estimate was received leaks future information.
Use an as-of snapshot based on receipt time and retain later revisions only for outcome adjudication.
A missing weather feed is unknown, not green.
Trigger a feed-health alert, use a documented fallback, and avoid claiming full coverage during the outage.

### Exercise 6 rubric

A strong plan assigns the buyer to investigate immediately and an authorized manager to decide on the $28,000 expenditure.
It sets a short acknowledgement window and names an out-of-hours backup.
It checks stock, the supplier's affected lot, alternate qualification, capacity, and flight acceptance before the six-hour deadline.
Preparing the quote does not itself commit the $28,000.
If the manager cannot be reached, the plan escalates through a named authority path before the deadline rather than allowing a silent expiry.
The case remains uncertain; a safety instruction is followed independently of the commercial score.
The closure record includes the decision, timestamps, actual receipts, and customer impact.

### Exercise 7 calculation

Region A has 20 positive and 980 negative opportunities.
At 75% recall, it has 15 true positives and 5 false negatives.
At a 5% false-positive rate, it has 49 false positives and 931 true negatives.
Precision is `15 / (15 + 49) = 23.44%`.
Region B has 100 positive and 900 negative opportunities.
It has 75 true positives, 25 false negatives, 45 false positives, and 855 true negatives.
Precision is `75 / (75 + 45) = 62.5%`.
The same assumed rates produce different precision because the event base rates differ.
Different source coverage, hazards, supplier reporting, or operating response can also change the rates themselves.

### Exercise 8 rubric

A strong design defines eligible opportunities and freezes the target and timely-warning deadline.
Where feasible and ethical, assign comparable units to access the new warning or the existing process, retaining safety-critical escalation for all.
If random assignment is not feasible, specify a credible comparison and its selection and confounding limits.
Log the warning, acknowledgement, decision, action, cost, incident evidence, and observed outcome at the case level.
Use an intervention-independent event measure where possible and estimate the counterfactual downstream outcome rather than relabeling prevented shortages as false alarms.
Account for sharing between teams, incomplete follow-up, sample size, seasonal variation, and uncertainty around incremental value.
A six-month estimate is not automatically representative of an annual rare-event distribution.

## 14. Interview questions

1. **What makes a warning actionable?**
   A strong answer links an exposed decision, owner, feasible response, and deadline.
   A weak answer treats publication of any alert as sufficient.
2. **How would you distinguish a hazard from a supply disruption?**
   A strong answer traces exposure and operational consequence.
   A weak answer assumes a storm warning proves a shipment is late.
3. **Why can three news articles be one piece of evidence?**
   A strong answer traces syndication to a common original source.
   A weak answer counts publishers without checking attribution.
4. **Which timestamps would you preserve?**
   A strong answer includes event, publication, receipt, acknowledgement, decision, and action times.
   A weak answer retains only a final updated timestamp.
5. **What happens to precision when the base rate falls?**
   A strong answer describes the effect assuming fixed recall and false-positive rate.
   A weak answer assumes precision is independent of population.
6. **How do you measure missed events?**
   A strong answer builds an eligible cohort and independently adjudicates outcomes.
   A weak answer examines only cases that generated alerts.
7. **Why can a successful intervention look like a false alarm?**
   A strong answer separates the hazard label from the prevented downstream outcome.
   A weak answer assumes no observed shortage means the warning had no value.
8. **When should a firm pay for an expedite?**
   A strong answer compares conditional expected avoidable loss with incremental cost and checks feasibility.
   A weak answer acts on severity alone or counts gross revenue as avoided profit.
9. **What would you do during a feed outage?**
   A strong answer marks coverage unknown and invokes an owned fallback.
   A weak answer displays green because no alert arrived.
10. **How would you evaluate a new warning system?**
    A strong answer compares against an existing baseline, records interventions, and reports uncertainty.
    A weak answer multiplies every alert by the maximum possible loss.

## 15. Key terms

- [Actionable warning time](#4-definitions-and-vocabulary): time sufficient evidence reaches an empowered owner.
- [Alert fatigue](#4-definitions-and-vocabulary): loss of effective attention from excessive notifications.
- [As-of view](#step-8--normalize-time-and-preserve-lineage): data available by a specified decision time.
- [Base rate](#4-definitions-and-vocabulary): positive share of the defined opportunity population.
- [Calibration](#4-definitions-and-vocabulary): correspondence between predicted probabilities and comparable outcomes.
- [Case](#4-definitions-and-vocabulary): managed workflow for an exposed decision.
- [Common Alerting Protocol](#8-data-standards-and-regulatory-context): standardized public-alert message format.
- [Confusion matrix](#step-12--evaluate-the-warning-not-just-the-feed): counts of true and false positives and negatives.
- [Decision deadline](#4-definitions-and-vocabulary): latest feasible action time for a specified objective.
- [Decision slack](#step-10--estimate-impact-and-the-remaining-option-window): actionable lead time less response duration.
- [Deduplication](#4-definitions-and-vocabulary): grouping observations referring to the same event.
- [Entity resolution](#4-definitions-and-vocabulary): matching names and identifiers to the right entity or site.
- [Event time](#4-definitions-and-vocabulary): time of the physical or operational event.
- [Expected loss](#4-definitions-and-vocabulary): probability-weighted loss over stated outcomes.
- [Exposure](#4-definitions-and-vocabulary): an asset or flow potentially affected by a hazard.
- [False negative](#4-definitions-and-vocabulary): positive opportunity without a qualifying timely warning.
- [False-positive rate](#4-definitions-and-vocabulary): false positives divided by all negative opportunities.
- [Freshness](#4-definitions-and-vocabulary): age relative to expected information updates.
- [Hazard](#4-definitions-and-vocabulary): potentially damaging event or condition.
- [Incident](#4-definitions-and-vocabulary): grouped disruption event at a defined scope.
- [Lead time](#4-definitions-and-vocabulary): time between a named warning timestamp and impact.
- [Outcome leakage](#step-8--normalize-time-and-preserve-lineage): using information unavailable when a historical decision was made.
- [Precision](#4-definitions-and-vocabulary): fraction of evaluated alerts that are true positives.
- [Provenance](#4-definitions-and-vocabulary): origin and transformation history of evidence.
- [Recall](#4-definitions-and-vocabulary): fraction of positive opportunities detected in time.
- [Response duration](#4-definitions-and-vocabulary): time required to authorize and execute an action.
- [Signal](#4-definitions-and-vocabulary): observation potentially informative about an event.
- [Time-to-recover](#4-definitions-and-vocabulary): time to restore a specified operating capability.
- [Time-to-survive](#4-definitions-and-vocabulary): time demand can be sustained without disrupted supply.
- [Value of information](#4-definitions-and-vocabulary): improvement in expected decision value from information.

## 16. Sources

Evidence grades follow the module index: A peer-reviewed or audited; B regulatory or official statistic; C company case or investor claim; D vendor marketing; E press summary or anecdote.
The categories describe source type, not a guarantee that every claim is independently verified.

1. U.S. National Weather Service, [API Web Service](https://www.weather.gov/documentation/services-web-api), official technical documentation.
   Grade B for its description of the government service; see product-specific caveats and updates.
2. OASIS, [Common Alerting Protocol Version 1.2](https://docs.oasis-open.org/emergency/cap/v1.2/CAP-v1.2-os.html), OASIS Standard, 1 July 2010.
   Standards-body source for message structure and lifecycle, not evidence of business impact.
3. ISO, [ISO 22301:2019, Security and resilience — Business continuity management systems — Requirements](https://www.iso.org/standard/75106.html).
   Standards-body source; the published standard and its amendments govern interpretation, not this summary.
4. Port of Los Angeles, [Control Tower launch](https://portoflosangeles.org/references/news_022421_controltower_launch), 24 February 2021.
   Grade C for the port's description of its own tool and operational metrics.
5. Port of Los Angeles, [Port Optimizer components](https://portoflosangeles.org/references/2022-news-releases/news_022822_port_optimizer), 28 February 2022.
   Grade C for the port's description of its own service; not an independent return-on-investment study.
6. A.P. Moller–Maersk, [Annual Report 2021](https://investor.maersk.com/static-files/b4df47ef-3977-412b-8e3c-bc2f02bb4a5f).
   Grade C for the company's account of the Suez Canal blockage and operational context.

The numerical example, exercises, thresholds, and economic assumptions are instructional constructions.
They should not be attributed to these sources.

## 17. Links

- [Module 14A index](_index.md).
- [The disruption timeline](01_disruption-timeline.md).
- [Mapping and exposure](03_mapping-exposure.md).
- [Time-to-recover and time-to-survive](04_ttr-tts.md).
- [Scenario analysis](06_scenario-analysis.md).
- [Buffering versus flexibility](07_buffering-flexibility.md).
- [Dual sourcing and nearshoring](08_dual-sourcing-nearshoring.md).
- [Module 14A survey](../../../course/14A_resilience_disruption_analytics.md).
- [Control towers and visibility events](../../../course/19_control_towers_visibility_events.md).
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md).
