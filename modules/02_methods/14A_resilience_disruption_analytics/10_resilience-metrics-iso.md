# Resilience Metrics and ISO 22301

> Module 14A — Supply Chain Risk, Resilience, and Disruption Analytics  
> Topic 10 of 10  
> Estimated study time: 4 hours  
> Prerequisites: disruption timeline, exposure mapping, time-to-recover and time-to-survive, scenario analysis  
> Version 2026.5

## 1. Why this topic exists

A continuity director reports that 98% of critical suppliers have signed a continuity questionnaire.
A production planner reports that the sole qualified supplier of a controller needs 12 days to resume deliveries.
The plant can maintain its priority customers for only seven days without those deliveries.
Both reports may be accurate, but only the second exposes a decision that cannot wait for the next annual audit.
The signature shows that a document exists; it does not establish that a replacement line can produce acceptable parts in time.

A buyer needs authority to reserve alternate capacity before a disruption, not a higher dashboard score after one.
An operations manager needs to know which customers receive limited output and who can authorize that allocation.
An information-technology (IT) manager needs to know whether production data can be restored before the factory needs it.
A board needs the exposure that remains after planned actions, the funding decision, and the evidence behind the recovery claim.
Their questions share data, but their decisions and tolerances differ.
A useful measurement system connects them rather than replacing them with a single weighted number.

Business continuity management gives organizations a structured way to identify critical activities, plan responses, exercise them, and improve.
ISO 22301:2019 specifies requirements for a business continuity management system (BCMS).
Certification concerns the management system within an assessed scope; it is not a guarantee that every supplier will recover by a chosen time.
An educational example cannot establish certification, legal compliance, or the adequacy of any organization's continuity arrangements.
The actual standard, its amendments and applicable contractual and legal obligations need review by qualified practitioners.

The operating problem is dynamic.
A plant can restore machines yet remain unable to ship because material is unavailable, quality release is incomplete, or a carrier has no equipment.
A supplier can pass a tabletop discussion while a real transfer fails due to a missing tool or an expired qualification.
A nominal alternate can be available on paper while the same upstream factory serves both sources.
A positive recovery plan therefore needs a defined scenario, a clock, a performance threshold, usable resources, and evidence of activation.

This topic builds a decision-ready scorecard around those elements.
The emphasis is readiness, service performance during a disruption, measurable recovery, and governance.
It explains why a target, a tested outcome, a forecast, a plan, and a contractual commitment are different kinds of information.
It also shows how to report missing evidence without presenting missing data as a pass.
The numbers in the worked case are invented for teaching; they are not a statement about a real company.

## 2. Learning outcomes

After completing this topic you can:

1. **Distinguish** management-system evidence from evidence of operational recovery; apply this distinction in exercise 1.
2. **Define** a critical service, scope, disruption clock, measurement threshold, and owner; apply them in exercise 2.
3. **Compute** service loss, time-to-recover, time-to-survive, recovery gap, and backlog clearance under stated conventions; apply them in exercises 3 and 4.
4. **Interpret** maximum tolerable period of disruption, recovery time objective, and recovery point objective without treating any one as a universal service guarantee; apply them in exercise 5.
5. **Evaluate** test coverage and results using denominators, evidence quality, and unresolved findings; apply them in exercise 6.
6. **Compare** recovery options under cost, service, and common-cause constraints; apply them in exercise 7.
7. **Design** an actionable scorecard and governance loop that exposes residual risk and missing information; apply them in exercise 8.
8. **Explain** what ISO 22301 contributes and what certification does not prove; apply this distinction in exercise 1.

The outcomes require interpretation as well as calculation.
A correct recovery number without its scenario or service threshold is incomplete.
A recommendation without an accountable decision owner and a date is also incomplete.

## 3. The core mental model

```text
Business priorities and impact over time
                 |
                 v
   Critical service + customers + scope
                 |
                 v
 Threat / scenario -> dependencies -> available resources
                 |                         |
                 v                         v
      Service through the event      Response and recovery actions
                 |                         |
                 +------------+------------+
                              v
       TTS, TTR, service loss, backlog and recovery evidence
                              |
                              v
      Compare with tolerances, objectives and decision thresholds
                              |
                              v
      Owner chooses: accept / reduce / transfer / avoid / escalate
                              |
                              v
        Exercise -> findings -> remediation -> retest -> review
```

Read the diagram from the customer backward as well as from the hazard forward.
A service definition identifies the activity and performance that matter to a customer.
A component-level recovery estimate matters because it constrains that activity, not because a component has an inherently meaningful score.
Dependency mapping connects the two.
For example, a recovered machine without a released component or a booking slot may not restore shipments.

The diagram separates the **ability to survive** from the **time needed to recover**.
Time-to-survive (TTS) describes the period for which a stated service threshold can be sustained under a defined disruption.
Time-to-recover (TTR) describes elapsed time from a specified event until a stated capability is restored.
When the service and capability clocks use compatible start points and scope, a positive TTR-minus-TTS gap indicates exposure to a threshold breach.
The gap does not itself quantify the number of missed units, the damage to specific customers, or the time needed to clear a backlog.
Those outputs require a time-phased flow calculation.

A business impact analysis (BIA) considers how impacts grow with time and identifies priorities and requirements for recovery.
Maximum tolerable period of disruption (MTPD) is a limit beyond which adverse impacts become unacceptable in the defined context.
A recovery time objective (RTO) is a target for resuming a specified activity or resource at an agreed level after disruption.
A recovery point objective (RPO) expresses the maximum acceptable data loss in time, measured between a recoverable data point and the disruption.
For operational use, define clock origins and service levels explicitly; terminology and formal obligations must be checked against the applicable standard and organizational policy.
RTO is a target, not a prediction or an observed recovery time.
RPO is not the time required to restore a database.

The management loop has two complementary evidence tracks.
**System evidence** includes assigned responsibilities, approved plans, review records, and corrective-action controls.
**Performance evidence** includes a timed exercise, actual service outcomes, resource availability, and the results of a retest.
A signed plan may make an exercise possible, but it is not a substitute for exercising the plan.
Conversely, a successful one-off exercise does not establish a maintained management system or prove every scenario will succeed.

Use a small portfolio of measurements instead of combining incomparable units into a single score.
A capacity shortfall cannot be canceled out by a high plan-review rate.
A decision maker can nevertheless see a concise summary: threshold breach, exposure, recommended action, owner, evidence, and date.
The underlying metrics should remain visible so that trade-offs and uncertainty are not hidden.

## 4. Definitions and vocabulary

The following definitions are practical conventions for this topic, not substitutes for the licensed standard.
Define exact organizational usage in the metric dictionary before comparing reports.

| Term | Working definition and unit | Common misuse |
|---|---|---|
| Critical service | Defined output for named customers and operating scope | Naming a department without describing its output |
| Service threshold | Minimum acceptable output or timeliness, such as 90% of daily priority demand | Using an annual average to hide a daily breach |
| Priority demand | Eligible demand entitled to the stated service policy, units/day | Applying a threshold to all demand after excluding difficult customers |
| Disruption | Event or condition that impairs normal delivery or operations | Treating the first report as necessarily the physical start |
| Scenario | Coherent description of a shock, affected resources, duration, and conditions | Calling a scenario an empirically estimated probability |
| Scope | Sites, services, suppliers, products, and time covered | Assuming a corporate certificate covers an excluded site |
| Dependency | Resource, activity, party, or information needed for service | Stopping at the first-tier supplier |
| Common cause | One event affecting multiple apparently separate resources | Counting two suppliers with the same upstream plant as independent |
| Baseline | Normal expected performance used as a comparison, units/day | Confusing a forecast with a guaranteed order |
| Forecast | Estimate of uncertain future conditions | Calling forecast demand a binding commitment |
| Plan | Chosen set of activities and allocations | Assuming planned alternate capacity is contracted |
| Target | Desired performance, such as 24 hours | Reporting a target as a measured result |
| Commitment | A documented undertaking, subject to terms and verification | Treating a nonbinding supplier aspiration as reserved capacity |
| TTS | Elapsed time until a stated service threshold is first breached, hours or days | Comparing stock cover under one scenario to TTR under another |
| TTR | Elapsed time to a specified restored capability, hours or days | Equating equipment repair with customer service recovery |
| Recovery gap | TTR minus TTS for compatible clocks, hours or days | Treating a negative gap as proof of no loss under all scenarios |
| Recovery trajectory | Output or service over time during and after the shock | Reporting only the final restored state |
| Fill rate | Eligible units fulfilled divided by eligible demand, percentage | Confusing unit fill with order-level on-time delivery |
| On-time delivery | Eligible orders or lines delivered by the promised time divided by eligible orders or lines | Changing the denominator to exclude delayed orders |
| Backlog | Unfilled demand still eligible for later fulfillment, units | Treating permanently lost demand as recoverable |
| Backlog clearance time | Elapsed time until the defined backlog is zero, hours or days | Declaring recovery on first normal-output day |
| Lost sales | Demand that cannot be recovered later, units or currency | Counting the same units in backlog and lost sales |
| BIA | Structured analysis of activities and impacts over time | Ranking activities without considering recovery dependencies |
| MTPD | Time after which impacts of disruption become unacceptable, hours or days | Treating it as a prediction of supplier recovery |
| RTO | Target time for resumption at a defined level, hours or days | Treating a target as evidence the capability exists |
| RPO | Maximum acceptable age of recovered data at disruption, hours | Treating RPO as system restoration duration |
| Recovery time actual | Observed elapsed restoration time under a specific event or test | Generalizing a tabletop estimate to real performance |
| Recovery point actual | Time between disruption and most recent usable restored data point | Measuring backup schedule rather than recoverability |
| BCMS | Business continuity management system | Assuming certification guarantees continuity of all services |
| ISO 22301 | International standard specifying BCMS requirements | Treating a self-scored checklist as certification |
| Exercise | Planned activity to practice, validate, or improve capabilities | Treating attendance as a successful recovery |
| Tabletop | Discussion-based exercise of roles and decisions | Claiming it proves physical capacity or data restoration |
| Functional test | Exercise of a specific capability under defined conditions | Calling an isolated software restore an end-to-end recovery |
| End-to-end test | Coordinated test across relevant dependencies and outcomes | Omitting customer delivery or quality release and claiming full coverage |
| Finding | Documented deviation, weakness, or improvement opportunity | Closing a finding because an action was assigned |
| Corrective action | Change addressing a finding and its cause | Equating a revised document with verified effectiveness |
| Retest | Re-execution of relevant conditions to check corrective action | Using a less demanding scenario without disclosing the change |
| Coverage | Eligible tested items divided by a defined eligible population | Reporting percentage without exclusions or time window |
| Evidence age | Time since an observation or test, days or months | Treating old evidence as valid after a material change |
| Exposure | Scope and potential consequence of a disruptive condition | Equating exposure with event probability |
| Residual risk | Risk remaining after controls and planned actions | Calling residual risk zero because a plan exists |
| Risk appetite | Authorized willingness to accept specified outcomes | Using a slogan instead of a threshold and escalation route |
| Control | Measure intended to affect risk or its consequences | Assuming a control functions because it is documented |
| Recovery capacity | Usable production or service rate after activation, units/day | Counting nameplate capacity before yield and allocation |
| Usable inventory | Released and accessible stock compatible with priority demand, units | Counting quarantined stock as available |
| Bottleneck | Resource constraining flow under the relevant conditions | Assuming the bottleneck remains fixed after a disruption |
| Coverage gap | Eligible exposure without current relevant evidence or protection | Hiding unknowns by dropping them from the denominator |
| Decision threshold | Pre-agreed condition requiring an action or escalation | Using colors that do not trigger an owner response |

## 5. The mechanics, step by step

### Step 1 — Define the decision and its owner

**Input:** management mandate, customer obligations, critical products, and known disruptions.
**Output:** a one-sentence decision, accountable owner, decision date, and explicit alternative actions.

Start with a choice such as whether to reserve an alternate production lane for a critical controller.
Name the executive who can authorize funding and the operating owner who must activate the lane.
State what information can change the decision, including a test result or a contractual capacity commitment.
A board review should not receive a dashboard without a proposed decision or a stated reason for no action.

Identify the cost of a wrong decision in both directions.
Too little protection can cause service failure and customer harm.
Too much protection can consume working capital and capacity that another critical service needs.
Some impacts, including safety or statutory obligations, may be constraints rather than prices to optimize.
Do not convert an unquantified safety impact into a currency amount merely to make a spreadsheet balance.

If no authority exists for the required response, the scorecard must flag a governance gap.
Do not assign the matter to a committee without naming an accountable individual.
For shared suppliers, determine which business unit can reserve capacity and how competing claims will be resolved.
Document a delegate in case the principal decision maker is unavailable.

### Step 2 — Define the service and population

**Input:** product and customer segmentation, order obligations, site and lane lists.
**Output:** a versioned service definition and an eligible population for each metric.

Specify the product family, customer class, geography, delivery point, measurement period, and service standard.
A daily unit fill rate can coexist with an order-level delivery measure, but their denominators differ.
Include demand that is canceled or delayed according to the stated policy; do not silently exclude it after the event.
For a product with patient or safety implications, distinguish the priority segment from ordinary replenishment.

Define the population as of a specified date.
For example, include all active facilities that make the controller and all direct suppliers of its critical subassemblies.
Keep a list of excluded facilities and the reason for exclusion.
A percentage derived from ten tested suppliers is not comparable with one derived from two suppliers selected for convenience.
Freeze the denominator for a reporting period, then explain additions and removals in the next period.

Branch on service complexity.
For one product and one customer, a single daily threshold may suffice.
For several customer segments, measure each segment separately when their obligations or harm differ.
For a network with several delivery locations, retain location-level results before aggregating.
A network average can meet target while a region receives no product.

### Step 3 — Map dependencies and set the disruption clock

**Input:** bill of materials, supplier and site mappings, transport routes, IT systems, workforce and utilities.
**Output:** a dependency chain and a common clock for each scenario.

Map physical and informational prerequisites in the order needed to restore the service.
Include tier-two resources where evidence exists and mark unknown locations explicitly.
Map tooling, inspection, quality release, packaging, transport bookings, customer acceptance, and production data.
A backup facility requiring a missing specialist is not fully available merely because its building is intact.
Record the source, date, and owner for each dependency assertion.

Choose a clock origin such as the first missed supplier receipt or the physical loss of supplier output.
The two origins can differ by inventory and transit lead time.
If supplier TTR starts at physical failure but customer TTS starts at missed receipt, translate one clock before subtracting.
Do not silently compare elapsed durations with different origins.
Document the time zone, working versus calendar days, and whether a boundary is inclusive.

Draw the sequence from detection to first usable receipt.
Include notification, decision, contracting, setup, qualification, production, quality release, transport, and receipt.
Some tasks may overlap, but only when resources, approvals, and precedences allow it.
The duration is the length of the feasible critical path, not necessarily the sum of every task and not merely the shortest task.
Record which assumptions about overlap have been exercised.

### Step 4 — Build the impact timeline

**Input:** demand forecast, contract priorities, inventory, normal supply, disruption scenario, and impact owners.
**Output:** service and consequence by time bucket plus an agreed impact threshold.

A BIA asks how the consequences of an interruption change with duration.
Model operational, customer, financial, legal, safety, and reputational effects separately where relevant.
For instance, an early shortfall may be absorbed by customer stock while a later shortfall stops a customer's line.
Do not assume that impacts increase smoothly or can always be monetized.

Use a consistent time bucket such as one calendar day.
For each day, show usable opening stock, usable receipts, demand, served units, backlog or lost units, and closing stock.
State whether today's receipts are available before or after today's demand.
Specify whether spare capacity clears backlog before it serves new orders.
If priorities change by customer, apply the documented allocation policy before aggregating results.

Elicit an MTPD with the business owner, subject to applicable policy and professional review.
State the service, starting event, and specific unacceptable consequence.
A single corporate MTPD is rarely meaningful for all activities.
An RTO should be set with the required level of resumed activity and should be consistent with the consequences identified in the BIA.
If the desired RTO is shorter than demonstrated capability, the gap is a decision and funding issue, not a reason to relabel the objective.

### Step 5 — Measure TTS and TTR on compatible terms

**Input:** impact timeline, recovery plan, relevant test results, and scenario assumptions.
**Output:** TTS, TTR, their gap, confidence and evidence provenance.

Define TTS as the elapsed time to the first breach of the chosen service threshold.
In a daily model, report the first failing day and the boundary convention rather than implying hour-level precision.
If 100% of demand is served on days 1–7 and 70% on day 8, a first-breach convention gives TTS of seven completed days.
If only a 90% daily threshold is required, recalculate against that threshold; do not reuse a 100% measure without adjustment.

Define TTR as time to restore a specified capability.
Supplier production, first usable receipt, normal daily throughput, and backlog clearance are different endpoints.
Record each relevant endpoint rather than choosing whichever makes the result look best.
For an actionable comparison with customer TTS, use an endpoint that actually protects the specified customer service.

Calculate `gap = TTR - TTS` only after reconciling clock origins, scope, and units.
A positive gap flags a period in which the threshold is at risk if the modeled trajectory is realized.
A zero or negative gap does not prove that every order is on time, because recovery may be partial, late in the day, or dependent on an untested action.
Describe uncertainty as an interval or discrete alternative scenarios when warranted.
Do not turn a supplier's optimistic promise into a statistical percentile without a distribution and evidence.

### Step 6 — Measure service loss and recovery shape

**Input:** the same day-level demand and served units used in the service definition.
**Output:** cumulative unmet demand, service trajectory, and clearance date.

For each day calculate unmet units as eligible demand minus units served against that demand.
If unmet demand remains recoverable, add it to a distinct backlog state.
If the order is lost, record it as lost demand and do not serve it later.
A late delivery may reduce backlog but does not retroactively make the original delivery on time.

For a fixed horizon, total service loss in unit-days can be represented by the sum of daily shortfall units.
This is a discrete area between a baseline service curve and actual or simulated service.
It depends on the chosen horizon, baseline, and demand treatment.
If demand changes during the disruption, use actual eligible demand or an explicitly labeled scenario, not an unexplained normal-day baseline.

If capacity returns to normal demand exactly, backlog does not clear without extra capacity or reduced future demand.
Calculate the clearance rate as post-recovery available supply less current demand, subject to allocation and yield.
If the rate is positive and constant, clearance days equal opening backlog divided by that rate, rounded up for whole-day operation.
If the rate is zero or negative, report that clearance is not achieved under the modeled conditions.
Do not divide by zero or claim the backlog will disappear automatically.

### Step 7 — Assess capability and evidence

**Input:** plans, contracts, observations, exercise records, incident reports, and findings.
**Output:** evidence-tagged readiness measures and a list of unverified assumptions.

Classify the source of each recovery claim.
A supplier estimate is a forecast; an approved internal sequence is a plan.
A signed reservation may be a commitment but is limited by its terms and any competing allocation rights.
A timed exercise provides observed evidence only for the conditions actually tested.
A real incident can be informative but may differ in scope or severity from the planned scenario.

Capture the exercise date, service and sites covered, scenario, start and stop clocks, success criterion, actual result, exceptions, and assessor.
Identify which dependencies were simulated rather than physically activated.
State whether a test was announced, whether capacity was genuinely reserved, and whether customer or regulatory acceptance was included.
A test can be useful even when it fails; the failure must remain visible until remediation is verified.

When reporting pass rate, define an eligible test as a planned, executed, and scored test under the chosen scope and period.
Show tests passed over tests executed alongside tests executed over tests scheduled.
Unexecuted tests are not passes and should not be silently dropped from the coverage view.
A single easy test and a single severe test should not be treated as interchangeable samples.
If the denominator is zero, report not measurable and explain why rather than reporting 0% or 100%.

### Step 8 — Choose decision-relevant indicators

**Input:** decision, impact timeline, dependencies, evidence, and policy thresholds.
**Output:** a balanced dashboard with action rules and accountable owners.

Select a small number of indicators spanning leading readiness, service impact, recovery, and governance.
For every indicator specify a numerator, denominator, unit, window, scope, source, freshness rule, owner, and threshold.
A metric with no decision attached may still serve diagnosis but should not crowd the executive summary.
A metric whose data cannot be produced consistently needs a data-improvement action, not an invented value.

A practical dashboard can show the following views side by side:

| View | Indicator | Decision supported | Typical caveat |
|---|---|---|---|
| Exposure | Critical services with a positive recovery gap | Where to invest or escalate | Depends on scenarios and comparable clocks |
| Readiness | Critical paths with a current end-to-end exercise | Where to test next | Coverage does not prove a pass |
| Readiness | Verified alternate capacity available by required day | Whether an alternate can actually help | Subject to shared allocation and qualification |
| During event | Priority demand filled on time each day | When to allocate or invoke contingencies | Does not show future inventory exhaustion |
| Recovery | Days to first usable receipt and to stable throughput | Whether recovery meets the objective | A single test may be optimistic |
| Recovery | Backlog and projected clearance day | When to add overtime or reset promises | Depends on demand and usable excess capacity |
| Governance | Overdue high-severity findings | Whether management must intervene | Severity and due dates require discipline |
| Economics | Annual mitigation cost and scenario loss | Which feasible option to fund | Scenario loss is not expected annual loss |
| Data | Critical dependencies with unknown upstream site | Where to request supplier evidence | Known sites are not necessarily independent |

Set thresholds in relation to consequences and decision rights.
A red condition should name an action, the person who takes it, and the deadline.
A green condition should specify what was actually evidenced and its expiration date.
Unknown should remain a distinct state, not a reassuring green or an arbitrary red performance value.
A material unknown can still trigger escalation because the decision cannot rely on absent evidence.

#### Metric card A — Critical service recovery gap

**Decision:** prioritize intervention for services whose recovery exceeds their survival period.
**Population:** all approved critical-service and scenario pairs as of the reporting date.
**Numerator:** eligible pairs with compatible TTR greater than TTS.
**Denominator:** eligible pairs with both clocks quantified under the same scenario.
**Companion count:** eligible pairs lacking either clock or a consistent origin.
**Unit:** pairs and percentage of quantified pairs; individual gaps in hours or days.
**Window:** current scenario version, with evidence and assumptions dated.
**Source:** BIA, inventory model, supplier recovery evidence, and dependency map.
**Owner:** continuity analytics lead, with service owners approving assumptions.
**Trigger:** escalate any positive gap for a service subject to a no-breach constraint.
**Response:** identify the bottleneck, compare feasible interventions, and set a decision date.
**Interpretation limit:** a gap does not measure the number of late units or prove an outage will occur.
**Example:** three quantified pairs have gaps of +5, 0, and −2 days; one further pair is unknown.
Report one of three quantified pairs with a positive gap, plus one unknown pair.
Do not report one of four as if the unknown pair had a nonpositive gap.

#### Metric card B — Tested critical-path coverage

**Decision:** allocate the next exercise slot to an untested or materially changed dependency.
**Population:** the frozen list of critical paths and required scenario classes for the period.
**Numerator:** paths with an executed test meeting the agreed evidence and freshness criteria.
**Denominator:** all eligible paths, including paths with no test.
**Companion counts:** scheduled, executed, passed, failed, not executed, and evidence expired.
**Unit:** paths and percentage, with test type shown separately.
**Window:** a management-approved evidence period or shorter period after material change.
**Source:** exercise register, attendance record, event timestamps, and test outputs.
**Owner:** exercise manager; each capability owner confirms the path definition.
**Trigger:** a critical path lacks current evidence or a high-severity scenario is untested.
**Response:** schedule an appropriate test or request explicit time-limited acceptance.
**Interpretation limit:** coverage states what was tested, not how well it performed.
**Example:** seven of ten paths have current tests; four passed, three failed, and three were untested.
Coverage is 70%; the pass rate among tested paths is four of seven, or 57.14%.
An aggregate pass rate must not erase the three untested paths or their different consequences.

#### Metric card C — Verified alternate capacity by deadline

**Decision:** determine whether alternate production can bridge the modeled shortfall.
**Population:** product-site-scenario combinations that require an alternate source.
**Numerator:** usable alternate units committed and demonstrably deliverable by the deadline.
**Denominator:** usable units required to maintain the specified service until primary recovery.
**Unit:** units and percentage for a specific date and scenario.
**Window:** the period from disruption to first primary usable receipt.
**Source:** capacity reservation, qualification record, production test, and transport evidence.
**Owner:** procurement director, with quality and logistics confirmation.
**Trigger:** verified quantity is below required quantity or a reservation is withdrawn.
**Response:** secure capacity, increase buffer, adjust the service plan, or escalate the gap.
**Interpretation limit:** nominal supplier capacity is not usable quantity and may be shared.
**Example:** 600 usable units are needed by day 10; 450 are reserved and tested for that date.
Coverage is `450 / 600 = 75%`, leaving a 150-unit gap before any uncertainty adjustment.
A 900-unit nameplate capacity at the alternate site does not eliminate the gap.

#### Metric card D — Priority on-time service

**Decision:** activate allocation, customer communication, or contingency transport.
**Population:** all eligible priority units due within each reporting interval.
**Numerator:** units delivered on time and meeting the required acceptance conditions.
**Denominator:** units due in the same interval, including late and unfilled units.
**Unit:** percentage per day and customer segment, with unit counts.
**Window:** event day and cumulative event period, separately.
**Source:** order promise, delivery timestamp, acceptance record, and exception register.
**Owner:** customer operations manager.
**Trigger:** daily service below the authorized segment threshold.
**Response:** apply the allocation policy and notify the decision owner and affected customers.
**Interpretation limit:** delivered units are not necessarily acceptable if quality or temperature conditions failed.
**Example:** 950 of 1,000 eligible units arrive by cutoff in acceptable condition.
The daily on-time rate is 95%; fifty late or rejected units remain in the denominator.
If one customer receives none, the combined rate still needs a customer-level breakdown.

#### Metric card E — Backlog clearance projection

**Decision:** authorize overtime, capacity reallocation, or revised customer dates.
**Population:** orders that remain valid and eligible for later fulfillment.
**Numerator:** current eligible backlog, in units, broken down by due date and priority.
**Denominator:** expected usable excess capacity after meeting the agreed allocation for current orders.
**Unit:** units and projected whole days, with a dated clearance estimate.
**Window:** from the current reporting point until the modeled backlog reaches zero.
**Source:** open-order ledger, capacity plan, confirmed material, and labor schedule.
**Owner:** master scheduler, with customer operations validating priorities.
**Trigger:** projected clearance exceeds the agreed customer or operational limit.
**Response:** obtain additional capacity or revise commitments through authorized channels.
**Interpretation limit:** a forecasted clearance date is conditional on demand, yield, and allocation.
**Example:** 700 units of backlog and 140 excess usable units/day imply five days to clear.
If excess capacity falls to 70 units/day, the projection becomes ten days.
If excess capacity is zero, the backlog does not clear under constant assumptions.

#### Metric card F — Data recovery objective achievement

**Decision:** invest in backup frequency, restore automation, reconciliation, or staffing.
**Population:** critical applications and data sets supporting the named service.
**Numerator:** tested data sets meeting both their own RPO and restoration RTO.
**Denominator:** critical data sets scheduled for evaluation in the reporting period.
**Companion counts:** tested, untested, RPO failures, RTO failures, and reconciliation failures.
**Unit:** data sets and percentages, with hours reported for each actual outcome.
**Window:** the relevant test or incident and evidence-validity period.
**Source:** immutable backup timestamps, restore logs, reconciliation records, and test report.
**Owner:** application recovery owner, with the business owner validating usability.
**Trigger:** either objective is missed or an essential data set is untested.
**Response:** correct the limiting step and perform a representative restore and reconciliation.
**Interpretation limit:** the aggregate may hide which data set prevents shipment.
**Example:** a three-hour recovery point meets a four-hour RPO.
The same test's ten-hour restoration misses an eight-hour RTO.
The data set fails the combined measure despite meeting one of its two objectives.

#### Metric card G — High-severity remediation

**Decision:** direct management attention to unresolved capability weaknesses.
**Population:** findings classified high severity under the approved classification policy.
**Numerator:** open findings past their approved due date, including overdue retests.
**Denominator:** all open high-severity findings, shown alongside the absolute count.
**Unit:** findings, percentage, and days overdue for each finding.
**Window:** as of the review date, with aging since discovery and since due date.
**Source:** finding register, corrective-action evidence, and approval record.
**Owner:** continuity program manager; action owners remain accountable for fixes.
**Trigger:** any overdue finding affecting a no-breach service or a material dependency.
**Response:** escalate resources, temporary controls, or time-limited risk acceptance.
**Interpretation limit:** a fall in open findings is not improvement if severity was downgraded without justification.
**Example:** two of five open high-severity findings are overdue, or 40%.
Report the two affected services and next action dates, not only the percentage.
If no high-severity findings are open, show zero open findings and mark the percentage not applicable.

### Step 9 — Compare interventions and residual exposure

**Input:** baseline scenario, feasible options, cost estimates, and service constraints.
**Output:** a decision table showing cost, impact, uncertainty, and remaining gaps.

Compare options against the same demand, clock, and disruption assumptions.
Potential levers include buffer stock, reserved alternate capacity, faster qualification, alternate transport, and recovery of critical data.
Inventory may extend TTS; a faster production transfer may shorten TTR.
A combination may do both, but its combined benefit must be simulated rather than added blindly.

Show one-time and recurring costs separately and specify the period used for comparisons.
Include qualification, reservation, carrying, quality, transport, and activation costs when material.
Avoid counting the same cost once as a mitigation expense and again inside the loss estimate.
Calculate monetary loss only for impacts with defensible values and boundaries.
When probability is unsupported, compare conditional scenario outcomes and break-even assumptions instead of producing expected annual loss.

Test decision sensitivity to demand, yield, activation delay, available capacity, and common-cause disruption.
If the recommendation changes under a small plausible variation, request better evidence or choose a staged decision.
Some options may be infeasible even if they appear cheapest because a customer or safety constraint is breached.
Record residual risks, including the scenarios excluded from the analysis.

### Step 10 — Govern, exercise, correct, and learn

**Input:** chosen action, operating plan, test calendar, thresholds, and risk acceptance authority.
**Output:** controlled execution, validated findings, and management review.

Assign a named owner for the decision, a separate owner for the operational capability, and an evidence custodian.
Schedule tests according to risk, change, and policy; do not assert a universal frequency from this teaching example.
A material site move, product change, supplier change, or system migration may invalidate prior evidence before its scheduled expiry.
Record the condition that triggers an unscheduled review.

An exercise plan states the scenario, objectives, roles, start and stop times, dependencies, success thresholds, and safety boundaries.
Preserve timestamps and actual outputs, not only participant recollections.
Classify deviations, assess their impact, assign a corrective-action owner and due date, and retain the original finding.
Close a finding only when the authorized reviewer has evidence that the correction is effective for the relevant failure mode.
Where a full retest is impossible, record the alternative evidence and remaining uncertainty.

Review the dashboard with people authorized to commit resources.
For accepted risk, state the scenario, exposure, reason, owner, acceptance period, and review trigger.
Escalate overdue critical actions and unresolved threshold breaches rather than allowing a color to become permanent background.
Feed lessons from incidents and exercises into the BIA, dependency map, plans, contracts, and next test.
This feedback loop is useful whether or not the organization seeks certification.

## 6. Worked example

### Case and decision

Aster Controls is a fictional manufacturer of industrial monitoring units.
Its priority customer group requires a controller module made by one qualified supplier.
The leadership team must decide whether to reserve an alternate production lane and carry a targeted buffer.
The scenario is an outage at the primary supplier, not a forecast of an outage or its probability.
All currency amounts are illustrative US dollars (USD).

The decision owner is the vice president of operations.
The procurement director owns the supplier capacity contract.
The plant manager owns execution of the allocation and production plan.
The quality manager owns release of any alternate-source modules.
The decision deadline is the next quarterly capacity negotiation.

### Baseline facts and assumptions

| Input | Teaching value | Interpretation |
|---|---:|---|
| Priority demand forecast | 480 units/day | Scenario forecast; not a customer commitment |
| Plant allocation plan | Serve priority demand first | Chosen rule; can require contractual review |
| Service target | At least 95% of daily priority units on time | Target, not observed outcome |
| Usable controller stock | 3,360 units at event start | Released and at the plant |
| Primary usable receipts | 0 units/day through day 12 | Scenario assumption |
| Primary recovery | 480 usable units/day from start of day 13 | Forecast under this scenario, not proven fact |
| Post-recovery normal capacity | 480 units/day | Meets new demand but cannot clear backlog |
| Temporary overtime capacity | 120 additional units/day | Assumed available from day 13 if approved |
| Unit contribution at risk | USD 85/unit | Illustrative incremental figure, not revenue |
| Scenario horizon | 28 calendar days | Daily periods; receipts before demand |
| Unmet-demand policy | Fully backlogged | No lost sales in this calculation |
| Start clock | Start of day 1, first missed receipt | All clocks use the same origin |

The 3,360 usable units are already released and belong to the modeled service.
They are not shared with another product and do not expire during the horizon.
Demand is constant only for clarity; actual demand could differ substantially.
The service measure is on-time fulfillment of each day's priority demand.
When overtime clears an old order, that order remains late for the daily on-time measure.

### Establish the survival and recovery clocks

Daily demand is 480 units and no primary receipts arrive during the outage.
Usable stock covers `3,360 / 480 = 7` complete days.
Days 1 through 7 each meet all 480 units of priority demand.
At the start of day 8, the modeled stock is zero and no new units arrive.
Day 8 on-time fill is 0%, below the 95% daily target.
Using completed days before the first breach, TTS is seven days.

The first usable primary receipt arrives at the start of day 13.
The elapsed TTR to that receipt is 12 days from the start of day 1.
The difference is `12 - 7 = 5 days`.
The five-day gap identifies days 8 through 12 as exposed under these assumptions.
It does not include the later duration needed to clear late orders.

### Build the daily result

| Days | Opening situation | Usable receipts for period | Current demand | On-time service | Backlog at end |
|---|---|---:|---:|---:|---:|
| 1–7 | 3,360 units in stock | 0 units | 3,360 units | 3,360 units | 0 units |
| 8–12 | No remaining stock | 0 units | 2,400 units | 0 units | 2,400 units |
| 13–27 | Primary output plus overtime | 600 units/day | 480 units/day | 480 units/day | Falls by 120 units/day |
| 28–32 | Same capacity and demand | 600 units/day | 480 units/day | 480 units/day | Falls by 120 units/day to zero |

The 2,400-unit backlog is `5 days × 480 units/day`.
At the start of day 13, the plan allocates 480 units/day to current demand and 120 units/day to backlog.
The clearance rate is therefore `600 - 480 = 120 units/day`.
Clearance takes `2,400 / 120 = 20` days, days 13 through 32 inclusive.
At the end of day 28, 16 recovery days have cleared 1,920 units, leaving 480 units.
The 28-day scenario horizon is sufficient to calculate on-time service, but not the eventual clearance date; the clearance calculation extends the same assumptions to day 32.
Without overtime, the 480 units/day primary output would match new demand and the backlog would remain at 2,400 units indefinitely in this simplified model.

Over days 1–28, eligible demand is `28 × 480 = 13,440 units`.
On-time units are all but the five disrupted days: `23 × 480 = 11,040 units`.
The on-time unit fill rate is `11,040 / 13,440 = 82.14%`.
The result does not mean that 17.86% of demand is permanently lost; these units are assumed backlogged.
The service curve has five days at 0% followed by restoration to 100% for new demand.
That recovery shape is more informative than a single average.

### Add a business-impact and data-recovery view

Suppose the BIA records that a five-day priority-service interruption becomes unacceptable at the end of day 5.
For this teaching case, the MTPD is five days of interruption of the defined priority service, measured from its first breach, not from the first missed receipt.
This origin is deliberately different from the supplier-receipt clock.
An RTO of three days after the first priority-service breach is the business target for restoring at least 95% daily service.
Under the baseline scenario, the breach starts at the beginning of day 8 and service resumes at the beginning of day 13, five days later.
The forecast therefore misses the three-day RTO by two days and reaches the stated five-day MTPD boundary.
Whether reaching a boundary is acceptable must be stated by policy; this case treats it as an escalation.
The physical TTR-to-first-receipt of 12 days and the service RTO of three days cannot be compared directly without translating their starting events.

Suppose the order database has an RPO of four hours and an IT restoration RTO of eight hours.
At the simulated disruption time of 12:00, the latest usable restored transaction snapshot is timestamped 09:00.
The actual recovery point is three hours, meeting the four-hour RPO in that exercise.
If the system becomes usable at 22:00 on the same day, restoration takes ten hours and misses the eight-hour IT RTO by two hours.
Meeting RPO does not imply meeting restoration RTO, and neither result proves the supplier can ship a controller.
The business owner must check whether the restored orders reconcile with later transactions and whether production can use them.

### Compare three actions

The following amounts are hypothetical annualized costs for a one-year comparison, including stated preparation and carrying costs.
The modeled incident loss includes expedited operations and customer-service costs, not a monetization of every reputational effect.
No incident probability is assumed.

| Option | Annual readiness cost | Conditional incident loss | Modeled priority service | Residual issue |
|---|---:|---:|---|---|
| A: baseline | USD 0 incremental | USD 260,000 | Five failed days; 2,400 late units | 20-day backlog clearance with overtime |
| B: extra stock only | USD 95,000 | USD 80,000 | One failed day; 480 late units | Stock may expire or be allocated elsewhere |
| C: stock plus alternate lane | USD 155,000 | USD 30,000 | No failed day in this scenario | Shared upstream resin and contract capacity remain risks |

Option B adds 1,920 usable units, four days of demand, bringing total stock to 5,280 units.
It covers days 1–11, so the first failing day is day 12 and the backlog peaks at 480 units before primary receipts resume.
With 120 excess units/day after primary recovery, the remaining backlog clears in four days, on day 16.
Its modeled service-breach duration is one day, not zero.

Option C adds the same four-day stock buffer and has a qualified alternate lane.
The alternate commitment is for 480 usable units arriving at the start of day 12, subject to contract terms and successful quality release.
Primary receipts resume at the start of day 13, so there is no unmet demand in the modeled sequence.
The commitment is not evidence of performance until its activation, quantity, release, and transport have been tested.
An upstream resin disruption could disable both suppliers; that scenario is outside this comparison.

The comparison is conditional on the specified incident.
It is invalid to subtract USD 230,000 of conditional loss reduction from USD 155,000 of annual cost without a probability and a common horizon.
Under a deliberately simplified model with at most one such incident per year, risk-neutral monetary valuation, and unchanged annual costs, C is cheaper than A in expectation when `p × (260,000 - 30,000) > 155,000`.
The break-even incident probability is `155,000 / 230,000 = 67.39% per year`.
This is an arithmetic threshold, not an estimated probability or a recommendation to assign one.
C can still be chosen for a non-negotiable service constraint even when the monetary break-even condition is not met.

The incremental annual cost of C versus B is USD 60,000.
The conditional incident loss improvement is USD 50,000 and the modeled service improvement is one fewer breach day.
Under the same one-event monetary model, the break-even probability is `60,000 / 50,000 = 120% per year`, outside the 0–100% probability range.
On these monetary assumptions alone C cannot beat B within that model, but B violates a hypothetical no-breach priority-service constraint while C satisfies it in the modeled scenario.
Changing customer impacts, option costs, event frequency model, or the service constraint can change the recommendation.

### Decision and monitoring

Assume management has approved a rule that no priority-service day may fall below 95% in this scenario.
Recommend option C conditionally, subject to a signed capacity reservation, validated quality release, and an end-to-end exercise before the next contracting cycle.
Until those conditions are satisfied, report the baseline exposure and flag C as a planned control rather than achieved readiness.
The contract should identify the 480-unit allocation, lead time, competing customer rights, and quality obligations.
The procurement director should verify whether the alternate has independent resin supply.
The operations vice president must accept the untested common-cause exposure or fund a separate mitigation.

Track four indicators at each monthly operating review: current tested alternate quantity by day 12, priority stock in usable units, overdue high-severity exercise findings, and the scenario-specific TTR–TTS gap.
Trigger an immediate review if usable stock falls below 5,280 units or an alternate-lane commitment changes.
Recalculate the timeline if demand changes materially or if quality release takes longer than assumed.
The business owner should review the MTPD and RTO when customer obligations or impact assumptions change.
The recommendation is not a claim that the fictional company complies with ISO 22301 or that the option handles every disruption.

## 7. Code, query, or calculation

The following Python 3 example reproduces the baseline and two options as a daily flow model.
It requires only the standard library.
It assumes receipts are available at the start of each day, constant priority demand, a 28-day reporting horizon, and no lost sales.
Current demand has priority over backlog, and an alternate receipt is a one-time quantity.
The primary rate plus overtime begins on day 13; overtime serves old backlog only after current demand.

```python
from math import ceil


def simulate(stock, alternate_day=None, alternate_units=0):
    demand = 480
    backlog = 0
    on_time = 0
    first_breach = None
    for day in range(1, 29):
        if day == alternate_day:
            stock += alternate_units
        if day >= 13:
            stock += 600  # 480 normal plus 120 overtime
        served = min(stock, demand)
        stock -= served
        on_time += served
        if served / demand < 0.95 and first_breach is None:
            first_breach = day
        backlog += demand - served
        cleared = min(stock, backlog)
        stock -= cleared
        backlog -= cleared
    return first_breach, backlog, on_time / (28 * demand)


for label, stock, day, quantity in [
    ("A", 3360, None, 0),
    ("B", 5280, None, 0),
    ("C", 5280, 12, 480),
]:
    breach, backlog, fill = simulate(stock, day, quantity)
    print(f"{label}: first_breach={breach}, backlog_day28={backlog}, "
          f"on_time_fill={fill:.2%}")

print(f"C versus A break-even: {155000 / (260000 - 30000):.2%}")
print(f"A backlog clearance days after recovery: {ceil(2400 / 120)}")
```

Expected output:

```text
A: first_breach=8, backlog_day28=480, on_time_fill=82.14%
B: first_breach=12, backlog_day28=0, on_time_fill=96.43%
C: first_breach=None, backlog_day28=0, on_time_fill=100.00%
C versus A break-even: 67.39%
A backlog clearance days after recovery: 20
```

The result for option B is a 28-day average, not proof that every day meets the 95% daily threshold.
Option B has one day at 0% despite its 96.43% aggregate on-time rate.
The code labels a day as a breach when its on-time fill is strictly below 95%.
The current-demand-first policy is explicit and might conflict with a real contract; changing it changes the results.
A realistic production model may require order-level promises, lead times, yields, capacity constraints, and multiple sites.
Those extensions must not be inferred from this calculation.

To use a spreadsheet instead, enter one row per calendar day with opening usable stock, receipts, current demand, on-time served, prior backlog cleared, closing backlog, and closing stock.
Compute on-time served as the smaller of available stock and current demand under this policy.
Add current shortfall to backlog and then clear backlog with any remaining available stock.
Check that opening stock plus receipts equals current served plus backlog cleared plus closing stock for each day.
Check that opening backlog plus current shortfall equals backlog cleared plus closing backlog.
A negative stock or backlog is a sign of a broken formula or incompatible data.
The spreadsheet and code both ignore additional demand after day 28 when reporting the 28-day fill rate.

## 8. Data, standards, and regulatory context

### ISO 22301 and continuity management

ISO identifies ISO 22301:2019 as *Security and resilience — Business continuity management systems — Requirements*.
The standard provides requirements for a BCMS, not a public ranking of suppliers or a universal recovery target.
Its management-system structure addresses organizational context, leadership, planning, support, operation, performance evaluation, and improvement.
The relevant organizational scope and the actual text determine the details of application.
The discussion here is educational and does not reproduce or interpret the full standard as an audit checklist.

A BIA and risk assessment inform continuity strategies and plans.
Exercises and performance evaluation provide feedback on the management system and capabilities.
Management review and corrective action support ongoing improvement.
An organization should use the current licensed standard, relevant guidance, contractual terms, and competent advice when determining its own obligations.
The existence of a certificate does not tell you whether a particular site, product, supplier, or scenario was within its scope.

ISO 22313:2020 provides guidance on using ISO 22301; it is not itself the requirements standard.
ISO/TS 22317:2021 provides guidance on BIA and can help structure analysis of impacts and dependencies.
ISO 31000:2018 provides risk-management guidelines rather than a BCMS certification guarantee.
These standards are complementary, not interchangeable proof of operational recovery.
Publication years identify the referenced editions; check ISO for the currently applicable edition and amendments before an engagement.

### Certification, audit, and evidence

A certification claim should name the legal entity, sites and activities in scope, certification body, certificate dates, and any exclusions.
Validate the certificate with the issuing body or relevant register where available.
A certificate shows the result of a management-system assessment within scope and at a point in time, subject to ongoing arrangements.
It does not establish that a specific spare part is in stock or that a particular recovery time was demonstrated end to end.
It is not a substitute for operational due diligence, supplier confidentiality agreements, or contractual recovery rights.

Different evidence answers different questions:

| Evidence | What it can support | What it cannot establish by itself |
|---|---|---|
| Approved continuity policy | Leadership has documented direction | A site can produce during a disruption |
| BIA record | Impacts and priorities were analyzed | Every impact assumption remains current |
| Supplier questionnaire | Supplier reported answers at a date | Independent validation of physical recovery |
| Certificate | Assessed management system within stated scope | All suppliers meet a specific TTR |
| Tabletop record | Roles and decision process were discussed | Real transport, capacity, or IT restore worked |
| Timed functional test | Specified function operated under test conditions | All dependencies were exercised |
| End-to-end exercise | Defined chain delivered under recorded conditions | Every untested hazard will have the same outcome |
| Actual incident record | Outcome observed under that incident's conditions | Performance in a more severe or different event |
| Signed reservation | Contractual right subject to terms | Immediate production without setup and release |
| Corrective-action retest | Relevant correction worked in the retest | The original failure is impossible in every scenario |

Document evidence limitations rather than treating the strongest available document as a universal assurance.
Assessors should distinguish independent evidence from management assertion.
Supplier information may be commercially sensitive; use appropriate access control and aggregation.
The evidence custodian should retain a traceable source and record when evidence becomes stale.

### Metric data model

A minimum service table contains service ID, product, site, customer segment, owner, eligible population, and effective dates.
A scenario table contains shock, affected nodes, origin time, duration assumptions, dependencies, and version.
An observation table contains date, metric, numerator, denominator, result, unit, scope, and evidence reference.
An exercise table contains planned and actual dates, scenario, success criterion, observed result, exceptions, and assessor.
A finding table contains severity, owner, due date, root cause, corrective action, retest evidence, and closure approval.
A decision table contains alternatives, authorized choice, residual exposure, acceptance owner, and review date.

Use stable identifiers rather than matching supplier names alone.
A supplier may have multiple sites and a site may be shared by several brands or legal entities.
Record the production site and the upstream dependency separately from the contracting entity.
Preserve historical mappings so that an old test is not accidentally attributed to a newly qualified facility.
Use effective dates when a product specification or customer commitment changes.

Data quality checks include missing site identifiers, duplicate tests, expired evidence, conflicting clocks, and unscored exercises.
A test can cover several dependencies, but do not count it as several independent tests without disclosing the unit of analysis.
A roll-up must retain the count of unknown or untested items.
Store both actual timestamps and their time zones where elapsed hours matter.
Use a documented calendar when a supplier quotes working days but a customer experiences calendar days.

### Legal and contractual boundaries

ISO standards are not automatically statutes merely because they are published.
A contract, regulator, procurement rule, or sector-specific law can make particular continuity obligations binding for a specified organization.
Applicability, reporting deadlines, assurance, and record-retention requirements vary by jurisdiction and sector.
Obtain advice from the relevant legal and compliance owners before using a scorecard as evidence of compliance.
This topic does not determine whether any particular firm is subject to a regulation.

Customer service agreements may set delivery, notice, allocation, and recovery commitments.
Supplier agreements may set reservation, force-majeure, audit, subcontracting, and data-access terms.
A supplier's continuity objective does not override the contract or guarantee priority during a common disruption.
A legal commitment that lacks physical capacity also needs an operational remedy.
Include the commercial owner when interpreting contractual evidence.

## 9. Industry practice and case evidence

### Practice: connect supplier mapping with recovery scenarios

A common supply-chain practice is to map critical components to production sites and upstream dependencies.
The analytic purpose is to identify whether a single disruption can affect several products or supposedly independent suppliers.
A site map alone does not show when usable output will resume.
Add production capacity, qualification, inventory position, transport, and customer demand before calculating a recovery trajectory.
Treat unmapped upstream sites as unknown, not independent.

Sheffi and Rice describe the resilient enterprise as combining operational practices and organizational capabilities rather than relying on a single contingency measure.
Their discussion supports using both physical options and response capability when evaluating continuity.
It does not provide a universal readiness score or validate the fictional case's parameters.
Christopher and Peck discuss supply-chain resilience and the importance of network-level approaches.
Their work provides conceptual support for examining dependencies beyond a firm's boundary.

### Public case: Toyota and the 2011 earthquake

Toyota's account of its response to the Great East Japan Earthquake describes disruption and subsequent efforts to build disaster resilience.
A University of Tokyo research paper describes Toyota's RESCUE supply-chain database and its use in identifying affected suppliers after disasters.
Those accounts describe a real practice, not an independent guarantee that every downstream component is immediately recoverable.
The relevant lesson is to connect visibility to action: identify affected sites, establish contact, and decide on supply alternatives.
A database that lists a site but lacks current capacity or recovery evidence may still leave the decision unresolved.

[INFERENCE] A buyer assessing a Toyota-like network would ask how recently the upstream site list was refreshed and whether recovery capacity was physically exercised.
[INFERENCE] The buyer would also ask whether multiple listed suppliers share an upstream factory, utility, or transport corridor.
These are analytical applications of the public case, not claims that Toyota failed to perform those checks.
A named case cannot provide transferable TTR values without matching product, network, scenario, and period.

### Public case: NIST contingency planning guidance

The US National Institute of Standards and Technology (NIST) publishes guidance on contingency planning for federal information systems.
NIST Special Publication 800-34 Revision 1 describes contingency planning, BIA, recovery strategies, testing, training, and exercises for information systems.
It is an official public-sector guidance source, not ISO 22301 and not automatically a legal requirement for every private supplier.
Its useful operational lesson is to define information dependencies and test restoration rather than relying solely on backup schedules.
A successful database restore still needs reconciliation and integration with the physical supply chain.

### Evidence grading and transfer

For this topic, official standards-body and government publications establish what those bodies publish; they do not validate a particular supplier's claims.
Peer-reviewed or professional research provides concepts and empirical findings within the studied contexts.
A company's public narrative is evidence of what the company reports, and may omit sensitive failures or unfavorable counterfactuals.
Vendor marketing and unsourced anecdotes are weaker evidence of actual recovery performance.
Do not combine evidence grades into a numerical readiness score without a defensible measurement model.

Before applying a case, ask whether the same customers, service threshold, disruption type, and response resources are present.
Ask whether the public account reports observed outcomes, a plan, or an aspiration.
Ask what was excluded and whether the outcome can be reproduced with independent records.
Record the source date and avoid treating a historical capability as current.
The next section translates these limitations into practical warnings.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Checklist replaces capability | Nearly all plans approved, no timed tests | Paper completion substitutes for activation | Alternate fails during a real event | Test capacity, release, and delivery end to end |
| Scope mismatch | Certificate names a different site | Corporate assurance is applied to an excluded activity | Buyer assumes unsupported coverage | Check certificate scope and operational evidence |
| Incompatible clocks | Supplier TTR starts before transit; TTS starts after stockout | Different origins make subtraction invalid | Gap is understated or overstated | Translate to one origin and show endpoints |
| Repair mistaken for recovery | Supplier reports machine repair only | Qualification and shipping are omitted | Customer service resumes later than promised | Measure first usable receipt and stable throughput |
| RTO treated as a fact | Target displayed without actual test time | Desired outcome is confused with capability | Management accepts an unproven control | Show target, observed result, scenario, and date |
| RPO confused with restore time | Backup interval shown as system availability | Data age and elapsed restoration are different | Operations cannot access timely orders | Test both data point and usable-system restoration |
| Average hides daily failure | Monthly fill meets target despite days at zero | Aggregation offsets severe local failures | Priority customers stop operations | Report daily or segment-level minimums |
| Backlog disappears in the model | Forecast returns to demand rate with no excess capacity | Flow conservation is violated | Customers are promised impossible clearance | Track backlog and available excess each period |
| Unknown becomes green | Missing supplier evidence excluded from report | Denominator is selectively reduced | Coverage appears higher than reality | Show eligible, tested, untested, and unknown counts |
| Easy-test bias | Only successful tabletop tests included | Difficult scenarios are not executed | Pass rate looks strong but coverage is weak | Report scheduled, executed, passed, and scenario severity |
| Common-cause blindness | Two suppliers share a resin plant | Nominal diversity does not remove shared exposure | Both options fail together | Map upstream dependencies and test joint disruption |
| Expired evidence | Last test predates a site or product change | Prior test conditions no longer match | An obsolete plan is trusted | Trigger review on material changes and set freshness limits |
| Unreserved alternate | Supplier says capacity may be available | Competing customers can claim the same output | Recovery plan cannot obtain the required units | Verify reservation, allocation rights, and actual availability |
| Quality release omitted | Alternate produces parts but release takes weeks | Production is confused with usable output | First usable receipt misses the threshold | Include qualification and release in critical path |
| Data restore unverified | Backups are scheduled but never restored | Corrupt or incomplete backups remain undetected | Orders cannot be reconstructed | Restore and reconcile a representative sample |
| Action closed on assignment | Finding marked complete when owner is named | Effectiveness is not tested | Same failure recurs | Require evidence and authorized closure or retest |
| Composite masks severity | High document score offsets a critical capacity gap | Unrelated measures are weighted together | Board misses a service constraint | Show separate thresholded indicators and decisions |
| Unowned risk acceptance | Dashboard has permanent red items | No one has authority or deadline | Exposure persists without a conscious choice | Record acceptance owner, period, and review trigger |
| Unpriced double count | Lost contribution appears in both incident loss and mitigation cost | Categories overlap | Option economics are distorted | Reconcile cost boundaries and cash flows |
| Scenario presented as forecast | A severe case is assigned an invented annual probability | Plausibility is confused with frequency evidence | Expected loss is misleading | Use conditional comparison and break-even analysis |
| Gaming the denominator | Suppliers are removed after a test failure | Reported pass rate improves without better capability | Decision maker sees false improvement | Freeze populations and disclose changes |
| Single successful exercise generalized | One announced drill is treated as universal proof | Real conditions and dependencies differ | Overconfidence in untested disruptions | Label scenario and run risk-based complementary tests |

A red flag is a prompt to investigate, not automatic proof of negligence.
For each warning, record the evidence, exposure, owner, and decision date.
A failed test may improve readiness if it leads to a verified correction.
Conversely, a passing test that excludes the bottleneck can create false assurance.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or challenge it |
|---|---|---|---|
| Demand is 480 units/day | Provides a reproducible service timeline | Demand spikes or product mix changes | Stress higher and lower daily demand and inspect order history |
| Starting stock is 3,360 usable units | Establishes the baseline TTS | Quality holds, expiry, or competing allocations | Reconcile physical, quality, and reservation records |
| Receipts occur before demand each day | Defines the daily boundary | Receipts arrive after the customer cutoff | Use hourly timing or shift receipts to the next day |
| Primary receipt begins on day 13 | Defines the scenario recovery endpoint | Repair, release, or transport takes longer | Compare supplier estimate with timed exercise and actual data |
| Primary output is 480 units/day | Supports current daily demand after recovery | Ramp-up, yield loss, labor, or material shortage | Test a lower and slower ramp profile |
| Overtime adds 120 units/day | Makes backlog clearance calculable | Labor, tooling, or approvals constrain overtime | Verify shift plans, contracts, and demonstrated throughput |
| All unmet demand is backlogged | Prevents double counting late and lost units | Customer cancels or substitutes | Segment orders by cancellation and substitution rules |
| Current orders have priority | Fixes allocation order in the model | Contract requires first-in-first-out or emergency priority | Review obligations and simulate alternate allocation rules |
| Extra stock remains dedicated | Makes option B's TTS extension available | Another product consumes the same stock | Check ownership and impose a reservation control |
| Alternate arrives with 480 usable units | Establishes option C service performance | Quality, transport, or shared capacity fails | Execute end-to-end transfer and review contract rights |
| One scenario has at most one event per year in break-even example | Makes probability-times-loss comparison simple | Multiple or correlated incidents occur | Use an appropriate frequency model when evidence supports it |
| Incident losses are comparable | Allows conditional option comparison | Loss boundaries or customer impacts differ | Reconcile categories and perform sensitivity analysis |
| Unacceptable impact begins at a stated boundary | Creates a teachable MTPD | Stakeholders disagree or obligations change | Revisit BIA with customer, legal, and business owners |
| Tested conditions represent a relevant scenario | Gives an exercise decision value | Actual event is broader or simultaneous | Maintain a scenario portfolio and identify untested assumptions |
| Data is accessible and current | Supports reproducible reporting | Supplier restrictions or stale mappings | Record provenance, access limits, and evidence age |
| No probability is available for the case | Prevents invented expected loss | Reliable frequency and consequence data become available | Review evidence and model uncertainty before using it |

The model does not establish a probability distribution, legal obligation, customer damages, or enterprise-wide risk.
Its deterministic values are scenario inputs, not confidence bounds.
A single simulation cannot demonstrate performance under a cyber event that also disables the alternate lane.
Testing additional cases is a decision about risk coverage, not a license to present every combination as equally likely.

## 12. Exercises

Use the definitions and clock conventions in this topic unless an exercise explicitly changes them.
Show units, the start and end of each clock, and any assumptions needed to solve the problem.
Where evidence is missing, label it rather than inventing a result.

### Exercise 1 — Assurance versus capability

**Time:** 10 minutes.
A supplier presents an ISO 22301 certificate for its headquarters and a two-year-old tabletop record.
Its only factory for your component is not listed on the certificate.
The supplier says it can restore production in four days but provides no transport or quality-release evidence.

**Deliverable:** A short note distinguishing three supported statements from three unsupported conclusions.
**Acceptance criteria:** Identify scope, evidence age, and the difference between production restoration and usable receipt.
Explain one next action without claiming that the supplier is noncompliant.

### Exercise 2 — Define a service and metric

**Time:** 15 minutes.
A company ships temperature-sensitive reagents to two hospitals.
The hospitals have different delivery cutoffs and priority classes.
A manager proposes the metric “percentage of locations resilient.”

**Deliverable:** A metric dictionary entry and an escalation rule for one critical service.
**Acceptance criteria:** Specify the service, eligible population, numerator, denominator, unit, time window, source, owner, threshold, and missing-data treatment.
Explain how the second hospital's exposure remains visible.

### Exercise 3 — Reconcile clocks and gap

**Time:** 15 minutes.
A supplier loses production at 00:00 on Monday.
The first missed plant receipt occurs 48 hours later.
The plant maintains its specified service for 96 hours after that missed receipt.
A qualified replacement receipt is forecast 168 hours after the supplier's production loss.

**Deliverable:** TTS and TTR expressed from the production-loss origin, the recovery gap, and an interpretation.
**Acceptance criteria:** Show the 48-hour origin translation and specify whether the result proves actual customer loss.

### Exercise 4 — Calculate service and backlog

**Time:** 20 minutes.
Daily demand is 200 units, usable opening stock is 600 units, and no receipts arrive for five days.
From the start of day 6, usable receipts are 250 units/day.
All missed demand is backlogged and current demand is served before backlog.
Use a 100% daily on-time threshold and calendar-day buckets.

**Deliverable:** First breach day, backlog at the end of day 5, clearance day, and on-time fill for days 1–10.
**Acceptance criteria:** Show the daily or equivalent period flow and distinguish late service from lost demand.

### Exercise 5 — Interpret BIA and IT recovery

**Time:** 15 minutes.
A business owner sets an activity RTO of six hours and an MTPD of 12 hours from the same disruption event.
During a test, the activity resumes at the required level after eight hours.
The latest usable database record is two hours old at the event, against an RPO of one hour.

**Deliverable:** A three-row target-versus-actual assessment and a concise action statement.
**Acceptance criteria:** Correctly identify the RTO result, RPO result, and relationship to MTPD without calling the activity RTO a measured achievement.

### Exercise 6 — Audit test coverage

**Time:** 15 minutes.
Twelve critical paths are eligible for an annual exercise program.
Eight were scheduled, six executed, and four of the six executed tests passed their predefined criteria.
Two failed tests have open high-severity findings; one of the four passing tests was a tabletop only.

**Deliverable:** Calculate execution against eligible paths, execution against scheduled paths, and pass rate among executed tests.
State how many eligible paths were not executed and identify two important limitations.
**Acceptance criteria:** Provide numerators and denominators, do not label the unexecuted paths passes, and distinguish test types.

### Exercise 7 — Make an investment choice

**Time:** 20 minutes.
Option X costs USD 70,000 per year and reduces a specified incident loss from USD 300,000 to USD 180,000.
Option Y costs USD 130,000 per year and reduces the same incident loss to USD 40,000.
Assume at most one such incident per year and no other costs or benefits for the calculation.
The probability is unknown and management has a mandatory service constraint that only Y satisfies in the scenario.

**Deliverable:** Break-even probabilities versus no action for X and Y, plus a recommendation under the stated constraint.
**Acceptance criteria:** Show equations, distinguish a threshold from a forecast, and state at least one residual uncertainty.

### Exercise 8 — Design governance and retesting

**Time:** 25 minutes.
An end-to-end alternate-lane exercise missed its 72-hour target because inspection staffing was unavailable on the weekend.
The supplier revised its plan and assigned an inspection manager.
No retest has been performed and the next peak season begins in six weeks.

**Deliverable:** A six-line executive dashboard entry and an action plan through retest and risk acceptance.
**Acceptance criteria:** State target, observed status, evidence limitation, accountable owner, due date or trigger, and decision.
Include a retest that exercises weekend inspection and an escalation if it cannot be completed in time.
Do not close the finding merely because a manager was assigned.

## 13. Self-check answers

### Exercise 1

Supported: a certificate was presented; its stated scope lists headquarters; a tabletop was recorded two years ago.
The certificate's authenticity and current validity still require verification.
Unsupported: the component factory is certified; the factory can deliver usable parts in four days; the two-year-old tabletop proves present end-to-end capability.
A reasonable next action is to request current factory-specific evidence and a timeline through quality release and delivery.
The available facts do not establish a breach of a particular contractual or legal requirement.

### Exercise 2

One valid design measures hospital A's eligible reagent units delivered by its agreed cutoff divided by hospital A's eligible units due that day.
The unit is percentage per day, the owner is the service operations manager, and the source is reconciled order and delivery timestamps.
A target could be 98% if authorized by the relevant business and contract owners; the figure is an exercise assumption, not a universal standard.
Unknown delivery timestamps should be counted and escalated separately rather than assumed on time.
Hospital B receives its own metric and threshold; a combined percentage must not replace either result.
A strong answer also defines how canceled orders and temperature excursions enter the denominator.

### Exercise 3

The missed receipt occurs 48 hours after production loss.
The service threshold is sustained for another 96 hours, so the translated TTS is `48 + 96 = 144 hours` from production loss.
The replacement TTR is 168 hours from the same origin.
The gap is `168 - 144 = 24 hours`.
The model flags a 24-hour threshold exposure, conditional on the stated forecast and definitions.
It does not establish the actual number of missed units or prove that the forecasted replacement will arrive.

### Exercise 4

Stock covers `600 / 200 = 3` full days.
The first breach occurs on day 4.
Days 4 and 5 each create 200 backlogged units, for 400 units at the end of day 5.
From day 6, 250 receipts less 200 current demand leaves 50 units/day for backlog.
The backlog takes `400 / 50 = 8` days to clear, on day 13, counting days 6–13 inclusive.
Days 1–10 have 2,000 units of eligible demand and 1,600 units served on time.
The on-time fill is `1,600 / 2,000 = 80%`.
The 400 units are late, not permanently lost, under the exercise's stated policy.

### Exercise 5

| Measure | Target or limit | Observed | Interpretation |
|---|---:|---:|---|
| Activity RTO | 6 hours | 8 hours | Missed by 2 hours |
| MTPD | 12 hours | 8-hour interruption | Activity resumed 4 hours before the stated limit |
| RPO | 1 hour | 2 hours of data age | Missed by 1 hour |

The test passes neither the RTO nor the RPO, even though activity resumed before the MTPD.
Investigate the slow recovery and data-loss gap, assign actions, and retest under relevant conditions.
Meeting a maximum tolerable limit is not the same as meeting a more demanding recovery target.

### Exercise 6

Execution against eligible paths is `6 / 12 = 50%`.
Execution against scheduled paths is `6 / 8 = 75%`.
Pass rate among executed tests is `4 / 6 = 66.67%`.
Six eligible paths were not executed, including the two scheduled but not executed.
The two failed tests remain visible as open high-severity findings.
The four passes include a tabletop, so they do not all demonstrate the same operational capability.
The exercise does not say whether the six executed tests cover all important hazards or dependencies.

### Exercise 7

For X, avoided conditional loss is `300,000 - 180,000 = USD 120,000`.
Its break-even probability is `70,000 / 120,000 = 58.33% per year` under the simplified model.
For Y, avoided conditional loss is `300,000 - 40,000 = USD 260,000`.
Its break-even probability is `130,000 / 260,000 = 50% per year` under that model.
Neither threshold estimates the event probability.
The stated mandatory service constraint selects Y among these options, subject to confirming that its capability and contract are credible.
A residual uncertainty is whether Y handles a common-cause event or has been tested under the actual scenario.
The financial calculation omits costs and benefits that would need inclusion in a real decision.

### Exercise 8

A strong dashboard shows the 72-hour target and the actual elapsed time if known; otherwise it states that the measured time was not supplied.
It marks the test failed and the staffing issue open, with the inspection owner, operating executive, and retest date before peak season.
The retest must include an actual weekend inspection resource, realistic handoff, quality release, and evidence of elapsed time.
The original finding remains open until an authorized reviewer accepts evidence of effectiveness.
If the retest cannot occur before peak season, escalate the residual exposure and obtain time-limited acceptance or an alternative control.
A revised document alone is weaker evidence than a relevant operational retest.

## 14. Interview questions

1. **What does a certificate tell a buyer?**
   A strong answer identifies an assessed management system, certificate scope and dates, and the need for capability evidence.
   A weak answer says certification guarantees that all supplier sites will deliver on time.
2. **How would you compare TTR and TTS?**
   A strong answer aligns scenario, origin, unit, service threshold, and endpoint before subtraction.
   A weak answer subtracts a supplier's repair estimate from unrelated inventory cover.
3. **Why can a 98% monthly fill rate be misleading?**
   A strong answer describes daily, segment, and location aggregation hiding a severe outage.
   A weak answer treats the average as proof that every critical customer was served.
4. **What is the difference between an RTO and an observed recovery time?**
   A strong answer calls the former an objective and the latter an outcome under recorded conditions.
   A weak answer presents an approved target as a test result.
5. **How is RPO different from system restoration time?**
   A strong answer measures acceptable data age separately from elapsed time to usable operation.
   A weak answer assumes a short backup interval means the system can be restored quickly.
6. **What happens to backlog when restored capacity equals new demand?**
   A strong answer explains that backlog remains unless demand falls or excess capacity appears.
   A weak answer assumes normal production automatically clears late orders.
7. **How would you report tests with a low execution rate?**
   A strong answer gives eligible, scheduled, executed, passed, and untested counts with scope.
   A weak answer reports only the pass percentage among the easiest tests.
8. **Why is a tabletop useful but insufficient for a physical transfer?**
   A strong answer values role rehearsal while distinguishing it from actual capacity, quality, and transit evidence.
   A weak answer dismisses all discussion exercises or calls them equivalent to a live transfer.
9. **What makes an alternate supplier credible?**
   A strong answer includes capacity rights, qualification, upstream dependencies, timing, and tested delivery.
   A weak answer counts a second vendor name as an independent and ready source.
10. **How would you choose a control without an event probability?**
    A strong answer compares conditional scenarios, constraints, cost, and break-even thresholds while labeling uncertainty.
    A weak answer invents a probability or asserts that worst-case loss is expected annual loss.
11. **When can an exercise finding be closed?**
    A strong answer requires accountable review and evidence that the relevant correction works, with residual limits disclosed.
    A weak answer closes it when an action is assigned or a plan is rewritten.
12. **What belongs in a board-level resilience decision?**
    A strong answer includes the critical service, exposure, choices, costs, evidence, residual risk, owner, and review trigger.
    A weak answer gives a composite score with no decision or underlying service threshold.

## 15. Key terms

Each link returns to the working definition table in section 4.

- [Backlog](#4-definitions-and-vocabulary) — eligible demand awaiting later fulfillment.
- [Backlog clearance time](#4-definitions-and-vocabulary) — time until the defined backlog is zero.
- [BCMS](#4-definitions-and-vocabulary) — business continuity management system.
- [BIA](#4-definitions-and-vocabulary) — analysis of activities and impacts over time.
- [Commitment](#4-definitions-and-vocabulary) — documented undertaking subject to terms.
- [Common cause](#4-definitions-and-vocabulary) — one event affecting multiple resources.
- [Corrective action](#4-definitions-and-vocabulary) — change addressing a finding and its cause.
- [Coverage](#4-definitions-and-vocabulary) — eligible tested items relative to a defined population.
- [Critical service](#4-definitions-and-vocabulary) — defined output for a named customer scope.
- [Decision threshold](#4-definitions-and-vocabulary) — condition requiring a specified response.
- [Dependency](#4-definitions-and-vocabulary) — resource or activity required for service.
- [Evidence age](#4-definitions-and-vocabulary) — time since observation or test.
- [Exercise](#4-definitions-and-vocabulary) — planned activity to practice or validate capabilities.
- [Exposure](#4-definitions-and-vocabulary) — scope and potential consequence of a condition.
- [Fill rate](#4-definitions-and-vocabulary) — eligible units fulfilled divided by eligible demand.
- [Finding](#4-definitions-and-vocabulary) — recorded weakness or improvement opportunity.
- [Forecast](#4-definitions-and-vocabulary) — estimate of uncertain future conditions.
- [ISO 22301](#4-definitions-and-vocabulary) — BCMS requirements standard.
- [MTPD](#4-definitions-and-vocabulary) — disruption duration beyond which impacts become unacceptable.
- [On-time delivery](#4-definitions-and-vocabulary) — eligible deliveries meeting a promised time.
- [Plan](#4-definitions-and-vocabulary) — chosen activities and allocations.
- [Recovery gap](#4-definitions-and-vocabulary) — compatible TTR less TTS.
- [Recovery trajectory](#4-definitions-and-vocabulary) — service or output over time.
- [Residual risk](#4-definitions-and-vocabulary) — risk remaining after controls.
- [Retest](#4-definitions-and-vocabulary) — relevant re-execution to check a correction.
- [Risk appetite](#4-definitions-and-vocabulary) — authorized willingness to accept specified outcomes.
- [RPO](#4-definitions-and-vocabulary) — maximum acceptable age of recovered data at disruption.
- [RTO](#4-definitions-and-vocabulary) — target time to resume at a defined level.
- [Scenario](#4-definitions-and-vocabulary) — coherent specification of a disruptive condition.
- [Service threshold](#4-definitions-and-vocabulary) — minimum acceptable service for a defined scope.
- [Tabletop](#4-definitions-and-vocabulary) — discussion-based exercise of roles and decisions.
- [Target](#4-definitions-and-vocabulary) — desired rather than observed performance.
- [TTR](#4-definitions-and-vocabulary) — elapsed time to a stated recovered capability.
- [TTS](#4-definitions-and-vocabulary) — elapsed time to first breach of a stated service threshold.
- [Usable inventory](#4-definitions-and-vocabulary) — released, accessible stock compatible with demand.

## 16. Sources

Standards and official guidance:

- ISO, [ISO 22301:2019, *Security and resilience — Business continuity management systems — Requirements*](https://www.iso.org/standard/75106.html). The ISO catalogue identifies the standard and its scope; consult the full applicable text for requirements.
- ISO, [ISO 22313:2020, *Security and resilience — Business continuity management systems — Guidance on the use of ISO 22301*](https://www.iso.org/standard/75107.html). Guidance complements rather than replaces the requirements standard.
- ISO, [ISO/TS 22317:2021, *Security and resilience — Business continuity management systems — Guidelines for business impact analysis*](https://www.iso.org/standard/79000.html). Use for BIA method and terminology, subject to the current edition.
- ISO, [ISO 31000:2018, *Risk management — Guidelines*](https://www.iso.org/standard/65694.html). General risk-management guidance, not proof of operational recovery.
- Swanson, M., Bowen, P., Phillips, A. W., Gallup, D., and Lynes, D. [*Contingency Planning Guide for Federal Information Systems*, NIST SP 800-34 Rev. 1](https://doi.org/10.6028/NIST.SP.800-34r1), 2010. Official US federal guidance on information-system contingency planning.

Research and industry evidence:

- Christopher, M., and Peck, H. “Building the Resilient Supply Chain.” *The International Journal of Logistics Management*, 15(2), 2004, pp. 1–14. [https://doi.org/10.1108/09574090410700275](https://doi.org/10.1108/09574090410700275). Conceptual supply-chain resilience and network design.
- Sheffi, Y., and Rice, J. B. “A Supply Chain View of the Resilient Enterprise.” *MIT Sloan Management Review*, 47(1), 2005. [MIT Sloan article](https://sloanreview.mit.edu/article/a-supply-chain-view-of-the-resilient-enterprise/). Discusses resilience capabilities and enterprise response.
- Pettit, T. J., Croxton, K. L., and Fiksel, J. “Ensuring Supply Chain Resilience: Development and Implementation of an Assessment Tool.” *Journal of Business Logistics*, 34(1), 2013, pp. 46–76. Research on resilience assessment; an assessment tool is not a substitute for scenario-specific evidence.
- Toyota Motor Corporation, [*Five Years on: Toyota's Efforts to Build a Disaster Resilient Future*](https://global.toyota/en/detail/11373994). Company account of earthquake response and resilience.
- University of Tokyo Manufacturing Management Research Center, [*Strengthening Purchasing and Supply Chain Management Functions*](https://merc.e.u-tokyo.ac.jp/mmrc/dp/pdf/MMRC514_2019.pdf), 2019. Research paper describing Toyota's RESCUE database and post-earthquake supply-chain management.

Source types matter: standards-body descriptions define published document scope, government guidance has its stated audience, academic work has its study context, and company material is a first-party account.
Neither the fictional Aster figures nor exercise assumptions are empirical findings from these sources.

## 17. Links

- [Module 14A topic index](_index.md)
- [Disruption timeline](01_disruption-timeline.md)
- [Mapping and exposure](03_mapping-exposure.md)
- [Time-to-recover and time-to-survive](04_ttr-tts.md)
- [Scenario analysis](06_scenario-analysis.md)
- [Buffering versus flexibility](07_buffering-flexibility.md)
- [Dual sourcing and nearshoring](08_dual-sourcing-nearshoring.md)
- [Module 14A survey](../../../course/14A_resilience_disruption_analytics.md)
- [Lab 14A.2 — TTR/TTS stress test](../../../course/14A_resilience_disruption_analytics.md#lab-14a2--ttrtts-stress-test)
- [Lab 14A.4 — Scenario analysis and board memo](../../../course/14A_resilience_disruption_analytics.md#lab-14a4--scenario-analysis-and-board-memo)
