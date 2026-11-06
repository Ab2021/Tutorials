# Logistics, Pharma, and Humanitarian Cases

> Module 28 — Domain Case Library and Sector Playbooks
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Modules 07, 10, and 26; topics 01–05 of this module
> Version 2026.5

## 1. Why this topic exists

A parcel route, a container vessel, a vaccine shipment, and an aid convoy can all be described as flows through a network.
They cannot all be governed by the same objective.
A missed delivery window can have a commercial penalty.
An unverified medicine can threaten patient safety.
A delayed relief shipment can reach a community after the critical need has passed.
A route calculation is only useful if its inputs, operating permissions, and consequences are understood.

Six cases illustrate different decision settings.
UPS's On-Road Integrated Optimization and Navigation (ORION) illustrates route recommendations in a dense parcel network.
The 2017 cyberattack on A.P. Moller - Maersk illustrates operational dependence on information systems.
Diversions during the Red Sea security crisis illustrate a network shock and a longer alternative route.
COVID-19 vaccine allocation illustrates scarce supply and competing ethical objectives.
The U.S. Drug Supply Chain Security Act (DSCSA) illustrates traceability obligations and trading-partner coordination.
The Haiti earthquake response illustrates coordination when physical and information infrastructure are damaged.

The common task is not to copy an organization's technology.
It is to specify a decision, identify constraints, and establish what evidence can actually support.
A company statement about its own route system is evidence of its claim.
It is not automatically an independently controlled evaluation of that system.
An official humanitarian situation report is a contemporaneous operational record.
It is not a complete accounting of unmet need or a causal impact study.
A regulatory requirement is a legal constraint for its covered parties and transactions, not a universal process design.

You will produce a case-to-decision memo with an explicit transfer test.
You will also calculate an allocation under supply, cold-chain, and access constraints.
The example is synthetic: it is not a reconstruction of an actual allocation, route, incident, or disaster.
Separate the following throughout:

- **Evidence:** a source's documented statement or observation, with date and scope.
- **Claim:** an attributed assertion that may not have independent verification.
- **Causal inference:** an estimate of an intervention's effect against an explicit counterfactual.
- **Transfer:** a conditional proposition about another setting, not a demonstrated result there.
- **Uncertainty:** missing information, competing explanations, and limits on generalization.

> **Note:** This topic is educational, not legal, medical, security, or emergency-response advice.
> Requirements, authorizations, outbreak conditions, and operating conditions change.
> Consult current authoritative guidance and qualified local decision-makers before implementation.

## 2. Learning outcomes

After studying this chapter, you can:

1. **Classify** statements about the six cases as evidence, claim, inference, or uncertainty.
   Exercise 1 tests classification against the source and the relevant decision.
2. **Describe** the decision, operating constraints, and missing counterfactual in each case.
   Exercise 2 asks for a case anatomy.
3. **Calculate** feasible vaccine allocations and unused capacity in a synthetic network.
   Exercise 3 checks arithmetic and units.
4. **Compare** a priority-based rule with proportional allocation without confusing equity with aggregate throughput.
   Exercise 4 requires both calculations and a value judgment.
5. **Design** a measurement plan for route recommendations or disruption recovery.
   Exercise 5 asks for a counterfactual and operational guardrails.
6. **Apply** the five-question transfer test to a new network.
   Exercise 6 asks for a conditional go/no-go recommendation.
7. **Distinguish** a product identifier, an event record, and an end-to-end chain-of-custody claim.
   Exercise 7 uses a missing-event scenario.
8. **Write** a humanitarian logistics decision brief that states uncertainty and affected-community feedback.
   Exercise 8 requires a decision owner and a revision trigger.

The outcomes are observable: a good submission includes calculations, source boundaries, and a decision.
Naming a famous case without showing the decision logic does not demonstrate transfer.

## 3. The core mental model

The same analytical pipeline applies across all six cases, but its constraints and accountability differ.

```text
  Operating context + decision authority + time horizon
                         |
          Data and documented source claims
                         |
       Validate identity, coverage, timing, units
                         |
      Feasible actions ---- hard safety/legal bounds
             |                      |
       Objective and trade-offs <---- affected people
             |
       Plan -> execution -> exceptions -> feedback
             |                 |
       Observe outcome     manual fallback
             |                 |
       Counterfactual and uncertainty
             |
       Transfer test -> local trial, revision, or reject
```

A **decision** has an owner, alternatives, a deadline, and an actionable unit.
Examples are a driver's next stop, a carrier's route choice, a public authority's dose allocation, or a cluster's cargo-priority list.
A model can recommend an alternative, but the accountable owner accepts or rejects it.
The decision cannot be evaluated from a vendor slogan or a single after-period total.

Data must be usable at the point of decision.
A daily stop forecast does not establish whether a loading dock is available at a particular hour.
A shipment event with an unknown timestamp cannot establish whether a cold-chain excursion occurred in transit.
A report of a reopened road does not establish whether every bridge on the route can carry the planned vehicle.
Record both the measurement and when it became available.

Constraints define the feasible set before optimization.
For parcel routing, constraints may include vehicle capacity, pickup windows, working time, and safe access.
For medicines, restrictions may include product authorization, storage conditions, and applicable trading-partner duties.
For relief, physical access, safety, and the agency's mandate can rule out an otherwise efficient route.
Safety is not a weighted penalty to trade away casually for a small mileage saving.

The objective expresses a preference within that feasible set.
Minimum distance is not identical to minimum labor cost or maximum on-time delivery.
Maximum doses shipped is not identical to maximum doses administered.
Maximum cargo tonnage is not identical to equitable access to needed assistance.
When objectives conflict, disclose the trade-off and who has authority to choose it.

Execution closes the loop.
A recommended stop sequence may be overridden for a blocked street.
A cargo plan may change after a security incident.
A vaccination site may receive less than its allocation if its refrigerator fails.
Capture overrides, reasons, timestamps, and consequences rather than treating deviations as mere noise.

A **counterfactual** is what would have happened to the same or comparable units without the intervention.
A before-and-after comparison can be useful operationally, but seasonality, mix, weather, and concurrent changes can bias it.
Randomized rollout may be possible for low-risk route recommendations; it may be infeasible or unethical in an emergency allocation.
When experimental comparison is inappropriate, use transparent scenarios, historical comparators, and sensitivity analysis without claiming experimental certainty.

Transfer requires more than surface similarity.
A maritime reroute and an earthquake response both involve network disruption.
Their legal authorities, infrastructure ownership, data, beneficiaries, and measures of harm differ.
Transfer the discipline of mapping bottlenecks and alternatives; do not assume the same ranking rule is appropriate.

## 4. Definitions and vocabulary

| Term | Definition and unit or code | Common misuse |
|---|---|---|
| Decision unit | Object acted upon, such as a stop, shipment, or district | Combining incompatible grains in one average |
| Planning horizon | Time between decision and the end of its effect, in hours or days | Treating a strategic capacity decision as a dispatch decision |
| Forecast | Estimate of a future quantity, such as doses requested per week | Calling it a guaranteed delivery |
| Plan | Intended action conditional on assumptions | Treating it as an executed event |
| Target | Desired metric, such as percent on time | Treating a target as a measurement |
| Commitment | Authorized promise to a recipient, with quantity and date | Promising inventory that has not been released |
| Route | Ordered sequence of locations or network edges | Confusing the recommendation with the route driven |
| Stop | A visit location, sometimes containing multiple parcels | Equating stop count with parcel count |
| ORION | UPS's On-Road Integrated Optimization and Navigation routing system | Assuming its reported benefits apply to a different carrier |
| Vehicle capacity | Maximum feasible load, in parcels, weight, volume, or other dimension | Checking weight while ignoring volume |
| Time window | Permitted or promised arrival interval, in local time with zone | Optimizing distance while omitting service windows |
| Override | Authorized departure from a recommendation | Treating every override as a mistake |
| Counterfactual | Outcome under the alternative not taken | Using a target as if it were observed control data |
| Treatment effect | Difference between outcomes under intervention and control | Attributing a raw before–after difference solely to a tool |
| Cyberattack | Malicious action against digital systems or their availability | Assuming every outage involved stolen data |
| Recovery time objective | Desired maximum restoration interval for a service | Claiming it is the actual recovery time |
| Recovery point objective | Maximum tolerable data loss expressed as a time interval | Confusing restored software with restored data integrity |
| Reroute | Alternative physical path for a shipment or vessel | Assuming all origin–destination pairs gain the same transit time |
| Lead time | Time from a specified start event to a specified end event | Comparing estimates with different clock boundaries |
| Buffer stock | Inventory held against uncertainty, in usable units | Counting expired or unreleased stock as usable |
| Allocation | Assignment of scarce supply to recipients | Assuming allocation guarantees administration |
| Administration | Delivery of a dose to a person, recorded as doses or people | Treating doses shipped as people fully protected |
| Equity | A declared fairness criterion sensitive to relevant differences in need or access | Assuming equal shares always produce equitable outcomes |
| Cold chain | Temperature-controlled handling from storage through administration | Inferring product suitability from a single endpoint reading |
| Excursion | Temperature outside a specified range for a time interval | Automatically declaring every excursion usable or unusable without assessment |
| DSCSA | U.S. Drug Supply Chain Security Act | Treating it as a global rule for every medical product |
| Trading partner | An entity in a covered drug transaction, in the applicable statutory category | Assuming all logistics providers have identical legal obligations |
| Product identifier | Prescribed identifier associated with a package or homogeneous case | Treating a product code alone as proof of transaction history |
| Transaction information | Data about a covered transaction as defined by applicable law | Treating a physical delivery note as necessarily sufficient |
| Transaction statement | Required statement about transaction conditions, as defined by applicable law | Assuming it validates every physical handling event |
| Global Trade Item Number (GTIN) | GS1 identifier for a trade item | Using GTIN alone as a unique serial number |
| Serial number | Identifier distinguishing a particular instance within an assigned context | Treating a duplicate or malformed serial as automatically authentic |
| Event data | Records of an event such as shipping or receiving, with attributes and timestamps | Assuming all events are complete or independently verified |
| Logistics Cluster | Humanitarian coordination mechanism supporting logistics in an emergency | Assuming it owns every response asset or makes all beneficiary decisions |
| Throughput | Amount passing a named point per time period, such as tonnes per day | Treating throughput as proof of equitable last-mile delivery |
| Last mile | Final movement to the service point or recipient | Assuming arrival at a central warehouse equals delivery to households |
| Access | Ability to physically and safely reach a place or service | Equating a mapped road with an open route |
| Evidence grade | Course shorthand for source type, A through E | Treating grade as proof of a specific causal claim |
| Sensitivity analysis | Recalculation under varied assumptions | Presenting a single scenario as a probability distribution |
| Stockout | Unavailability of usable stock at the needed place and time | Ignoring stock in quarantine or stock that cannot be administered |
| Opportunity cost | Value of the best alternative foregone | Counting sunk expenditure as a future incremental cost |
| Provenance | Record of a datum's origin and transformations | Assuming a copied assertion becomes independent corroboration |

## 5. The mechanics, step by step

### Step 1 — Specify the decision, not the famous name

**Input:** a proposed use of a named case and an operational question.
**Output:** a one-sentence decision statement with owner, alternatives, unit, and deadline.

1. Name the owner who can actually change the decision.
2. State the exact choice, such as two feasible routes or two allocation rules.
3. Identify the decision unit: a vehicle-day, vessel voyage, package, dose, or district-week.
4. Set the decision deadline and review cadence.
5. State the action if information is late or unavailable.
6. Separate the question of whether a method is useful from the question of whether it is legally permitted.

For ORION, a useful question might be whether dispatchers should offer a stop sequence for a vehicle-day.
For Maersk, a useful question might be how to sustain critical bookings if a shared platform is unavailable.
For the Red Sea, it might be which routing options remain available for a specified origin–destination pair.
For vaccination, it might be how many released doses each site should receive this week.
For DSCSA, it might be how to resolve an exception before a covered product transaction proceeds.
For Haiti, it might be which cargo and route to prioritize given a verified access constraint.

**Branch:** if no one owns the choice, identify the authority before building an optimizer.
If the named action cannot be implemented by the deadline, choose a decision at the right horizon.
A vessel diverted after departure has fewer options than a vessel not yet booked.
A vaccination appointment cannot be retroactively satisfied by a forecast correction.

### Step 2 — Build a bounded evidence card

**Input:** primary source, publication date, original claim, and source context.
**Output:** a claim ledger separating observation, attribution, and unknowns.

For each source:

1. Record the author and whether that organization operated, regulated, measured, or merely reported the event.
2. Quote or accurately paraphrase only the proposition supported by the source.
3. Record geography, population, period, unit, and denominator if available.
4. Identify whether the number is observed, estimated, forecast, or claimed.
5. Capture the comparison group, if any.
6. Record material missing information alongside the claim.
7. Check whether another source is independent rather than copying the same statement.
8. Preserve an access date in a live project.

The course source ladder labels regulatory and official statistics B, and company accounts C.
That label helps identify the source's role.
It does not mean every B-source conclusion is causal or every C-source observation is false.
An official situation report may be accurate about a reported operational status and incomplete about unreported communities.
A company can accurately report its own system while having incentives to emphasize favorable outcomes.

**Branch:** if an effect size has no defined denominator, do not convert it into a return on investment.
If a source says a route is planned, do not record it as completed.
If the only source is marketing, use it to formulate a test rather than to approve a high-risk rollout.

### Step 3 — Map the physical and information networks

**Input:** facilities, transport links, systems, products, and handoffs.
**Output:** a node-edge map with failure dependencies.

1. Draw origin, intermediate facilities, destination, and available transport edges.
2. Mark the capacity and operating hours of each constrained node.
3. Map the information needed to authorize or execute each edge.
4. Identify shared identity, booking, communications, and event-data services.
5. Identify suppliers, carriers, government agencies, and communities with decision rights.
6. Mark single points of failure, including shared credentials and upstream services.
7. Describe fallback modes and the information they require.
8. Note where a physical event and its digital record may diverge.

A parcel vehicle can be physically available while its route device is offline.
A container can be at a port while the booking or release record is inaccessible.
A medicine can be physically received while a required transaction-data exception remains unresolved.
A humanitarian warehouse can hold cargo that cannot cross a damaged bridge.
A dose may arrive at a district but be unavailable to patients because outreach staff are absent.

**Branch:** if digital service loss prevents safe release, use an authorized exception process rather than inventing a release record.
If a route is physically unavailable, a software recommendation cannot create capacity.
If a fallback depends on the same identity service as the primary system, it is not independent.

### Step 4 — Establish data grain and validity

**Input:** data dictionary and sample records, where authorized.
**Output:** a point-in-time dataset and a list of exclusions and unknowns.

For route data, distinguish vehicle-day, stop, package, and scan.
Record planned and actual timestamps, distance source, service windows, and override reasons.
A package scan is not necessarily a measured vehicle arrival.
Avoid training on realized traffic information that was unavailable at dispatch.

For incident data, distinguish service outage, restoration, backlog clearance, and customer recovery.
Collect incident start, detection, containment, restoration, and validation times.
Document clock synchronization and missing logs.
Do not treat the first restored server as the end of customer disruption.

For vessel data, specify departure and arrival definitions, origin–destination pair, vessel class, and routing period.
Compare door-to-door lead time with door-to-door lead time, not a sea passage against a port-to-port estimate.
Record changes in freight rates, port waiting, and sailing frequency separately.
Do not ascribe all variation in a rate to one security event.

For allocation, distinguish eligible population, requested doses, available released doses, site capacity, and actual administrations.
Use the same period for every quantity.
Track open-vial wastage and second-dose obligations where relevant to the product and policy.
Store sensitive person-level information only under an authorized governance framework.

For traceability, document identifier structure, serialization, transaction parties, event time, and receipt reconciliation.
A product code without a serial does not uniquely identify each package.
An event stream without verified completeness cannot alone establish an unbroken chain.
Distinguish correction, duplicate, late, and missing events.

For relief, record source, observation time, verification time, location precision, access status, and affected population estimate.
Mark old road-status reports as stale rather than interpreting missing updates as continued access.
Use local reports without exposing individuals to unnecessary harm.

**Branch:** when a data field is missing, decide whether it blocks action, can be bounded, or requires an approved manual procedure.
Never fill a missing temperature record with an assumed normal temperature.
Never infer need solely from the quantity a community was able to request.

### Step 5 — Specify objectives and constraints

**Input:** decision statement, network, data, and policy requirements.
**Output:** a feasible-set definition and an ordered or weighted objective.

1. List physical constraints in their own units.
2. List legal, clinical, security, and humanitarian constraints with an owner and source.
3. Identify constraints that must not be traded against cost.
4. Define the performance metric and its denominator.
5. State any fairness criterion and its intended beneficiary group.
6. Specify a service or safety guardrail.
7. Define a manual exception and escalation path.
8. Check that the inputs represent the same time horizon.

Example route constraints include available driver hours, vehicle load, and delivery windows.
Example allocation constraints include released stock, site capacity, and eligible demand.
Example humanitarian constraints include road passability, driver safety, and access permissions.
A condition such as verified temperature history can determine whether stock is usable at all.

A priority objective may first protect high-risk groups, then maximize throughput among feasible options.
A weighted-sum objective may instead exchange a unit of equity against a unit of distance.
The latter requires an explicit justification for the weights and may conceal unacceptable outcomes.
A lexicographic rule makes the order of priorities visible but can produce unused capacity if not carefully specified.

**Branch:** if constraints conflict, report infeasibility and escalate.
Do not silently relax a legal or safety condition to return a mathematically feasible plan.
If the objective lacks a legitimate decision owner, present alternatives rather than choosing values on the analyst's authority.

### Step 6 — Generate and compare feasible options

**Input:** a validated feasible set and at least one baseline.
**Output:** an option table with outcomes and assumptions.

For parcel routing, compare the current dispatch practice with a recommendation, subject to the same windows.
For cyber resilience, compare a tested fallback with the present recovery process.
For maritime diversion, compare available route and timing options for a specified shipment.
For vaccines, compare at least two transparent allocation rules with the same stock and site limits.
For DSCSA, compare compliant exception-handling paths with the help of the responsible legal and quality functions.
For humanitarian logistics, compare routes and cargo priorities against verified safety and access constraints.

Use a baseline that could actually have been selected at the decision time.
Include incremental resources, timing, and expected service consequences.
Avoid double-counting the same benefit as both reduced mileage and the fuel expense already included in cost per mile.
Report a range when a critical input is uncertain.

A feasible option is not necessarily desirable.
An allocation that uses all doses may be unacceptable if it systematically excludes remote communities.
An alternative with the shortest expected voyage may have an unacceptable security exposure.
A fallback that restores booking quickly may still compromise record integrity.
An aid route that can carry cargo may not be safe for local drivers or recipients.

### Step 7 — Design measurement before implementation

**Input:** chosen option, baseline, risks, and review owner.
**Output:** an evaluation plan with a credible comparison and guardrails.

For a route trial, consider random assignment of comparable vehicle-days when operationally and ethically feasible.
Predefine on-time service, safety, mileage, driver time, and override rates.
Stratify or adjust for geography, stop density, and workload where needed.
Account for spillovers if drivers share workload or routes.

For resilience, record actual recovery times in exercises and incidents.
Measure service restoration, data reconciliation, and backlog clearance separately.
A tabletop exercise can reveal dependencies, but it is not proof of performance under a live attack.
Avoid deliberately disrupting critical care or emergency services merely to obtain an experiment.

For allocation, compare shipped, administered, wasted, and unserved doses by relevant group and geography.
Define whether the denominator is eligible population, appointments, requests, or estimated need.
Preserve privacy when reporting small groups.
Monitor downstream service effects rather than treating the optimization objective as the only outcome.

For humanitarian operations, compare delivery to independently estimated needs and access where practicable.
Use community feedback and complaints channels that do not expose recipients to retaliation.
Recognize that data collection itself can consume scarce response capacity.
Document missing areas and do not label absent observations as zero need.

**Branch:** if randomization is inappropriate, write down the observational comparison and its limitations.
A difference-in-differences design, for example, needs a defensible parallel-trends assumption and no unaddressed spillovers.
Without a credible counterfactual, report change over time as descriptive, not causal.

### Step 8 — Execute, monitor, and escalate

**Input:** approved plan, named owners, and operational controls.
**Output:** an execution log and a documented revision or closure decision.

1. Release only the inventory, route, or record approved for use.
2. Record plan version and the time each recipient received it.
3. Capture actual movements and service events as they occur.
4. Reconcile planned and actual quantities at each handoff.
5. Investigate discrepancies according to risk and ownership.
6. Monitor safety, service, equity, and data-quality guardrails.
7. Escalate when a predeclared trigger is crossed.
8. Preserve original records and document corrections.
9. Make a time-stamped decision to continue, revise, or stop.
10. Communicate changes to affected operators and recipients.

A route override may be a useful signal that a map is wrong.
An unexpected cold-chain excursion may require product-specific assessment.
A missing serial event may require partner reconciliation.
A deteriorating security situation may invalidate a maritime or aid route.
A district with no reporting may need direct verification, not an automatic zero allocation.

### Step 9 — Apply the transfer test

**Input:** original case card, proposed destination, and decision risk.
**Output:** transfer decision: adapt and test, gather evidence, or reject analogy.

Ask the five questions in the module survey:

1. Is the decision the same?
2. Are product economics and consequences comparable?
3. Is the data environment sufficiently similar?
4. Are operational constraints similar?
5. Is the evidence strong enough for the risk?

For each question, record yes, no, or unknown and supporting evidence.
Three or more no answers make the case an analogy rather than a plan under the course rule.
Unknown is not an automatic yes; seek information or design a bounded test.
Even fewer than three no answers do not override a safety or legal prohibition.

Transfer the reusable mechanism, not the headline outcome.
For example, a route-recommendation feedback loop may transfer from parcel delivery to home health visits.
A mileage-saving percentage does not transfer without a comparable baseline, constraints, and validation.
A crisis network map may transfer from maritime diversion to relief logistics.
A commercial profit objective does not automatically transfer to humanitarian allocation.

### Step 10 — Close the learning loop

**Input:** observed outcomes, exception records, and stakeholder feedback.
**Output:** revised case card with evidence grade, limitations, and next decision.

Retain unsuccessful trials and rejected options alongside successful examples.
Document changes in assumptions, network topology, or policy since the original decision.
Preserve the distinction between a hypothesis supported by a case and a locally measured effect.
A case library that contains only reported successes will bias future decisions.
Review whether the people affected by the decision had a meaningful opportunity to correct the record.
Archive a time-stamped source snapshot where rights and policy permit.

## 6. Worked example

### Decision and evidence boundary

A fictional regional health authority must allocate a week's released vaccine stock to three fictional districts.
The calculation is not an account of a WHO, national, or Haiti decision.
It uses a simplified priority rule to teach scarcity, capacity, and access constraints.
A real allocation requires product-specific clinical advice, local law, community consultation, and current policy.

The decision owner is the regional immunization manager.
The decision deadline is Monday at 08:00 local time.
The planning horizon is one week.
The decision unit is a dose assigned to a district site.
The manager can revise the plan if a site loses capacity before shipment.

The authority has 1,000 released doses this week.
The total is usable stock, not a forecast and not a promise of future supply.
Assume each eligible person needs one dose for this exercise only.
Assume all doses are interchangeable and their remaining shelf life covers the week.
Assume expected wastage is zero so the arithmetic isolates allocation.
These assumptions are not general claims about any vaccine.

The weekly forecast of eligible demand is shown below.
It is an estimate, not a commitment or actual administration.
Each site also reports the number it can administer safely during the week.
The priority group is defined by the fictional authority's approved policy.

| District | Priority demand, doses/week | Other eligible demand, doses/week | Safe administration capacity, doses/week |
|---|---:|---:|---:|
| North | 420 | 280 | 500 |
| Central | 310 | 390 | 450 |
| Remote | 190 | 210 | 240 |
| **Total** | **920** | **880** | **1,190** |

Total forecast demand is 1,800 doses/week.
Available stock is 1,000 doses/week, so forecast demand exceeds stock by 800 doses/week.
Capacity exceeds stock by 190 doses/week in aggregate.
Aggregate spare capacity alone does not show that capacity is distributed in the right places.

### Rule A: priority first, then use remaining capacity

The fictional policy says to allocate all feasible priority demand first.
If doses remain, allocate them to other eligible demand in the order Remote, North, Central.
This secondary ordering is an explicit access-oriented policy choice, not a mathematical implication of the data.
It may be replaced after local consultation.
No district may exceed its forecast demand or safe capacity.

Priority demand fits within each district's capacity:

- North: 420 priority doses are at most 500 doses of capacity.
- Central: 310 priority doses are at most 450 doses of capacity.
- Remote: 190 priority doses are at most 240 doses of capacity.
- Total priority allocation: 420 + 310 + 190 = 920 doses.
- Remaining stock: 1,000 - 920 = 80 doses.

Remaining site capacity after priority allocation is:

- North: 500 - 420 = 80 doses.
- Central: 450 - 310 = 140 doses.
- Remote: 240 - 190 = 50 doses.

Give Remote the first 50 remaining doses.
This reaches its site capacity and leaves 30 doses.
Give North the next 30 doses.
Central receives no other-eligible doses under this ordering.

| District | Priority allocation, doses | Other allocation, doses | Total allocation, doses | Spare capacity, doses |
|---|---:|---:|---:|---:|
| North | 420 | 30 | 450 | 50 |
| Central | 310 | 0 | 310 | 140 |
| Remote | 190 | 50 | 240 | 0 |
| **Total** | **920** | **80** | **1,000** | **190** |

The plan assigns all 1,000 doses.
All 920 forecast priority doses are covered under the simplifying assumptions.
Other-eligible demand covered is 80 / 880 = 9.09%, rounded to two decimal places.
The aggregate forecast demand covered is 1,000 / 1,800 = 55.56%.
These percentages are planned coverage of forecast demand, not observed patient coverage.

Forecast unmet demand is:

- North: 700 - 450 = 250 doses.
- Central: 700 - 310 = 390 doses.
- Remote: 400 - 240 = 160 doses.
- Total: 250 + 390 + 160 = 800 doses.

The allocation is a plan.
A district commitment should be issued only after the authority confirms shipment and operational readiness.
Administrations and wastage must be measured later.
Nothing in the calculation proves that the demand forecast is accurate or that the plan is equitable.

### Rule B: proportional to total forecast demand

For comparison, allocate each district a share of the 1,000 doses proportional to total forecast demand.
All district shares are within their stated capacities.
Calculate exact quotas before rounding:

- North: 1,000 × 700 / 1,800 = 388.888… doses.
- Central: 1,000 × 700 / 1,800 = 388.888… doses.
- Remote: 1,000 × 400 / 1,800 = 222.222… doses.

The floor allocations total 388 + 388 + 222 = 998 doses.
The two largest fractional remainders belong to North and Central.
Give each one additional dose; the resulting allocation is 389, 389, and 222 doses.
The total is exactly 1,000 doses.
Each allocation respects its site's capacity and total demand.

Rule B does not specify how a district divides its doses between priority and other eligible people.
It therefore cannot by itself guarantee full priority coverage.
North's 389 doses, for example, are fewer than its 420 forecast priority doses.
At least 31 forecast North priority doses would remain uncovered even if every North dose went to a priority recipient.
Rule A and Rule B have the same aggregate planned throughput but embody different distributional commitments.
The calculation does not establish which ethical rule a real authority should use.

### Sensitivity: Remote capacity falls

Suppose a cold-chain problem reduces Remote's safe capacity from 240 to 170 doses for the week.
This is a scenario, not an observed event.
Remote priority demand is 190 doses, so the priority-first rule cannot fully satisfy it.
The authority assigns 170 priority doses to Remote, 420 to North, and 310 to Central.
That uses 900 doses and leaves 100 doses.
With Remote full, the secondary order gives North 80 other-eligible doses and Central 20.

The revised allocation is North 500, Central 330, Remote 170 doses.
All 1,000 doses remain assigned and every site is within revised capacity.
Priority coverage falls from 920 to 900 doses, a reduction of 20 doses.
The aggregate total hides the loss of access in Remote.
A temporary mobile clinic or a later delivery might mitigate the constraint, but neither is assumed to exist.

Before committing the revised plan, check whether redistribution serves eligible residents from Remote without unreasonable travel burdens.
Confirm product-specific storage and transport requirements.
Consult affected local representatives and the clinical authority.
If the refrigerator's temperature history is unknown, quarantine or assessment procedures depend on product and policy; do not assume the stock is safe.

### What the example can and cannot establish

The arithmetic establishes feasibility under stated inputs and rule A.
It does not establish that people appear at sites, that demand is evenly observable, or that all doses are administered.
It does not estimate infections, deaths, cost-effectiveness, or causal health benefit.
It does not implement WHO policy or any national vaccine program.
It demonstrates that equal throughput can mask different priority coverage and geographic access.

The data needed for a real decision would include verified inventory release, product-specific schedules, site operating hours, and local eligibility.
It would also include demand uncertainty, transportation constraints, spoilage, actual utilization, and a legitimate fairness process.
The owner should reassess the plan if a capacity, shipment, or eligibility assumption changes.

## 7. Code, query, or calculation

The following Python 3 program reproduces both priority-first scenarios and the proportional comparison.
It uses only the standard library.
The algorithm assumes integer doses, nonnegative demand, one period, and a fixed secondary district ordering.
It is an educational allocation calculation, not clinical or regulatory decision software.
Priority is allocated in input order if stock cannot cover all feasible priority demand.
That tie-breaking rule must be explicitly approved or replaced in a real use.
The demonstration has enough stock to cover feasible priority demand, so that tie does not affect its result.

```python
from fractions import Fraction


DISTRICTS = [
    {"name": "North", "priority": 420, "other": 280, "capacity": 500},
    {"name": "Central", "priority": 310, "other": 390, "capacity": 450},
    {"name": "Remote", "priority": 190, "other": 210, "capacity": 240},
]
STOCK = 1000
SECONDARY_ORDER = ["Remote", "North", "Central"]


def priority_first(rows, stock, secondary_order):
    """Allocate integer doses; input order resolves priority scarcity."""
    result = {row["name"]: {"priority": 0, "other": 0} for row in rows}
    by_name = {row["name"]: row for row in rows}
    remaining = stock

    for row in rows:
        amount = min(row["priority"], row["capacity"], remaining)
        result[row["name"]]["priority"] = amount
        remaining -= amount

    for name in secondary_order:
        row = by_name[name]
        allocation = result[name]
        spare = row["capacity"] - allocation["priority"]
        amount = min(row["other"], spare, remaining)
        allocation["other"] = amount
        remaining -= amount

    return result, remaining


def proportional_if_feasible(rows, stock):
    """Largest-remainder allocation, rejecting capacity-infeasible quotas.

    This demonstration does not redistribute capacity violations.
    It is only applied when stock does not exceed total demand and
    the resulting rounded district allocations satisfy site capacity.
    """
    demands = [row["priority"] + row["other"] for row in rows]
    total = sum(demands)
    if total == 0 or stock > total:
        raise ValueError("Requires positive demand and stock <= total demand")
    quotas = [Fraction(stock * demand, total) for demand in demands]
    amounts = [quota.numerator // quota.denominator for quota in quotas]
    left = stock - sum(amounts)
    # Stable input order resolves any equal fractional remainder.
    ranking = sorted(
        range(len(rows)),
        key=lambda i: (-(quotas[i] - amounts[i]), i),
    )
    for i in ranking[:left]:
        amounts[i] += 1
    if any(amount > row["capacity"] for amount, row in zip(amounts, rows)):
        raise ValueError("Proportional allocation exceeds a site's capacity")
    return {row["name"]: amount for row, amount in zip(rows, amounts)}


def show(label, rows):
    allocation, remaining = priority_first(rows, STOCK, SECONDARY_ORDER)
    print(label)
    for row in rows:
        name = row["name"]
        p = allocation[name]["priority"]
        other = allocation[name]["other"]
        print(f"  {name}: priority={p}, other={other}, total={p + other}")
    covered = sum(value["priority"] for value in allocation.values())
    print(f"  priority covered={covered}; unused stock={remaining}")


show("Base scenario", DISTRICTS)
print("Proportional:", proportional_if_feasible(DISTRICTS, STOCK))
reduced = [
    {**row, "capacity": 170} if row["name"] == "Remote" else dict(row)
    for row in DISTRICTS
]
show("Remote capacity falls", reduced)
```

Expected output:

```text
Base scenario
  North: priority=420, other=30, total=450
  Central: priority=310, other=0, total=310
  Remote: priority=190, other=50, total=240
  priority covered=920; unused stock=0
Proportional: {'North': 389, 'Central': 389, 'Remote': 222}
Remote capacity falls
  North: priority=420, other=80, total=500
  Central: priority=310, other=20, total=330
  Remote: priority=170, other=0, total=170
  priority covered=900; unused stock=0
```

The code does not model uncertainty, appointments, second doses, transport, consent, or stock reservations.
It also assumes unique district names and that every district appears exactly once in the secondary order.
The proportional function rejects capacity violations rather than silently redistributing them.
For real use, specify the legally authorized tie-break and escalation when priority demand exceeds stock.
Validate every input and test the policy against actual operational and equity requirements before implementation.

## 8. Data, standards, and regulatory context

### Evidence and decision records

For all six cases, maintain a source ledger with publisher, document date, claim, geography, metric, and limitations.
Keep a separate operational data dictionary with field definitions, units, source systems, refresh cadence, and data owner.
A publisher's authority to describe an event does not itself prove a causal mechanism.
A timestamped record proves that an event was recorded, not necessarily that the recorded physical event occurred correctly.

A decision record should preserve the forecast available at the time.
Keep later actuals and revised forecasts in separate fields.
Store the version of the optimization objective and the constraints used to generate a recommendation.
Record human overrides without assuming they were either correct or incorrect.
Restrict access to personal or sensitive location data according to applicable law and policy.

### Parcel routing

A route system may ingest location, stop sequence, parcel, vehicle, driver, and traffic information.
Address matching and geographic coordinates can be wrong or stale.
A route recommendation's granularity and refresh frequency determine what it can react to.
A static dispatch plan should not be evaluated as though it knew a road closure that occurred later.

Safety rules and labor obligations depend on jurisdiction and employment arrangements.
This chapter does not state a universal driving-hours rule.
Consult the relevant operator, labor, and transportation requirements for the location.
A company's stated route-system features are not a technical standard binding other carriers.

### Cyber continuity

A recovery plan needs inventories of applications, data, identities, suppliers, and dependencies.
Define recovery time and recovery point objectives for each critical service.
An objective is not evidence that a restoration was achieved.
Test whether offline procedures remain usable when authentication, communications, and ordinary backups are unavailable.
Validate restored data and reconcile transactions created during fallback.

Security controls and incident reporting obligations are jurisdiction- and organization-specific.
The Maersk incident does not establish which controls would have prevented every attack.
The source record supports the existence and reported operational effect of the attack, not a universal recovery prescription.

### Maritime diversion

A maritime record should distinguish scheduled, estimated, and actual departure and arrival.
Identify the route taken and whether the measure includes canal transit, port waiting, and inland transport.
Carrier schedules and quoted rates can change after publication.
UN Trade and Development (UNCTAD) reports aggregate maritime conditions; it does not predict the journey of an individual shipment.

Security, sanctions, insurance, customs, crew safety, and contractual conditions may constrain available routes.
The relevant maritime operator and competent authorities must evaluate those constraints.
No numerical transit-time addition in this chapter should be used as a universal sailing estimate.

### Vaccines and cold chain

The WHO Strategic Advisory Group of Experts on Immunization (SAGE) values framework published in September 2020 offers ethical guidance for COVID-19 vaccine allocation and prioritization.
It addresses allocation between countries and prioritization within countries.
It is guidance, not a substitute for a country's legal authority or a product-specific label.
The framework does not prescribe the synthetic rule in section 6.

Separate vaccine doses from persons, courses, and fully vaccinated people.
Product schedules and storage requirements differ and can change with authorization or guidance.
Record product, lot, expiry, quantity, temperature data, excursion duration, and disposition by an authorized party.
A temperature excursion is an event requiring assessment; its effect cannot be inferred without product-specific information.

Allocation data can encode unequal access.
A low appointment count may reflect travel costs, language barriers, disability access, or absent registration channels rather than low need.
Use appropriate aggregation and safeguards for sensitive health data.
Consult communities on criteria and provide a correction and appeal pathway where feasible.

### U.S. pharmaceutical traceability

The DSCSA is a U.S. law concerning the pharmaceutical distribution supply chain and covered prescription drug products and transactions.
Requirements differ by trading-partner role and statutory provision.
FDA's DSCSA pages explain enhanced drug distribution security and relevant waivers or exemptions.
The legally relevant statute, FDA guidance and notices, product coverage, party status, and transaction date must be checked for a specific case.
This chapter is not a compliance determination.

At a high level, the framework includes exchange of transaction information and transaction statements and package-level product identification for covered circumstances.
A product identifier generally combines a standardized numerical identifier, lot number, and expiration date as specified in the statute; the standardized numerical identifier includes a standardized product identifier and unique serial number.
These concepts must not be collapsed into a bare GTIN.
An identifier by itself does not prove that a product is genuine, authorized, or correctly stored.

FDA has published exemptions beyond a stabilization period for certain trading partners.
Eligibility, conditions, and duration are specified in FDA notices and can vary by category and situation.
Do not interpret an exemption as a blanket extension for all parties or all duties.
Consult current FDA materials and qualified counsel when determining applicability.

GS1 identification and event standards can help trading partners structure identifiers and share event data.
A GS1 standard is a technical convention, not by itself an FDA approval or a complete legal determination.
Electronic Product Code Information Services (EPCIS) can represent what, when, where, why, and how events occurred, depending on the data captured.
Even a syntactically valid EPCIS message can be incomplete, late, duplicated, or factually wrong.
Reconciliation and investigation remain necessary.

### Humanitarian coordination

The Logistics Cluster publishes operational information and coordination material for humanitarian logistics.
Its Haiti situation reports are snapshots of what was known and reported at the time.
Access status, capacity, partners, and routes can change quickly.
Reports do not necessarily measure all communities, all cargo, or all harms.

Distinguish shared logistics services from the mandate and accountability of each responding organization.
A cluster report describing a route or facility does not grant a separate organization permission to use it today.
Local authorities, responders, and affected communities have knowledge and decision rights that cannot be inferred from a central dashboard.
Protect sensitive location and beneficiary data in accordance with applicable humanitarian data practices.

## 9. Industry practice and case evidence

### Case 1 — UPS ORION

**Context and decision.** UPS operates a parcel-delivery network with repeated routing and delivery decisions.
ORION is the company's route-optimization system.
A driver or dispatcher must execute a route under real operating constraints.

**Documented company account.** In its UPSNav announcement, UPS describes navigation that provides turn-by-turn directions derived from its routing software and identifies UPSNav as an update to ORION. [1]
The release describes expected operational and service benefits from the technology.
It is a company source, not an independent controlled trial of each claimed outcome.

**Intervention.** The documented system provides route-derived navigation to service personnel.
Local route maps, vehicle constraints, stops, and driver execution are relevant to the operational mechanism.
The public announcement does not supply all data, assignment rules, or a matched control needed to estimate a causal effect for any particular route.

**Inference.** Better sequencing could reduce distance when accurate network and stop data identify avoidable detours.
It could also fail to reduce total cost if recommendations create service failures or require more handling time.
These are testable hypotheses, not findings proven by the release.

**Transfer.** A dense urban home-service network may test a route recommendation with comparable time windows and local overrides.
A sparse rural medical service may face different travel times, patient needs, and risk constraints.
Do not copy a UPS savings claim into its business case without local measurement.

**Unknowns.** The publicly cited release does not establish the counterfactual miles for the same routes, the full deployment cost, or outcomes for every driver.
No particular numerical saving is asserted here.

### Case 2 — Maersk and NotPetya

**Context and decision.** Maersk operates shipping and related logistics services whose execution depends on shared information systems.
On 28 June 2017 the company reported that it had been hit as part of a global cyberattack it called Petya, with systems down across multiple sites and business units. [2]
The destructive campaign is commonly referred to as NotPetya; the cited contemporaneous company notice uses Petya.

**Documented company account.** The notice describes a company-wide operational disruption and the company's response at that moment.
It is primary evidence of what Maersk publicly reported, not a complete incident reconstruction.
Do not infer from the notice alone the exact attack path, all recovery costs, or the impact on each customer.

**Mechanism.** Shared digital infrastructure can connect a failure in one system to bookings, documentation, terminal operations, and customer communication.
That dependency is a plausible general mechanism.
Which specific dependencies caused which losses in this incident requires more detailed evidence than the notice alone.

**Transfer.** An operator can map critical services, dependencies, and manual fallbacks.
It should test restoration and data reconciliation against its own recovery objectives.
An offline spreadsheet is not a viable fallback if it depends on unavailable credentials or cannot be reconciled safely afterward.

**Causal boundary.** A reported drop in a financial metric during the incident period would not automatically equal the attack's causal cost.
A valid estimate would require a defensible no-attack comparison and treatment of unrelated market changes.
A later restored application does not prove that all customer backlogs were cleared at the same time.

### Case 3 — Red Sea diversions

**Context and decision.** Security threats in the Red Sea prompted changes in maritime routing during the crisis beginning in late 2023.
Carriers and shippers had to choose among available routes, schedules, and service commitments under uncertainty.

**Official evidence.** UNCTAD describes security threats causing redirection of ship arrivals and transits, with repercussions for trade and transport. [3]
Its analysis discusses diversions around the Cape of Good Hope.
This is an institutional analysis of the disruption, not a measurement of the causal delay of every shipment.

**Mechanism.** A longer route can increase sailing time, vessel utilization, fuel use, and inventory in transit.
The magnitude depends on origin, destination, speed, vessel, port congestion, and alternatives.
The route change can interact with capacity and schedule changes throughout the network.

**Transfer.** Map alternative routes, exposed shipments, inventory buffers, and customer commitments.
Stress-test several possible lead times rather than inserting one universal extra-day assumption.
A firm with short shelf-life products or time-sensitive production inputs may have different costs from a firm with durable inventory.

**Uncertainty.** Rates and lead times also move with demand, capacity, congestion, contracts, and other shocks.
Observing a rate increase after an attack is not sufficient to attribute the entire increase to that attack.
A planned diversion is not an observed arrival.

### Case 4 — COVID-19 vaccine allocation

**Context and decision.** Limited early vaccine supply required decisions about allocation between countries and prioritization within countries.
Those choices affected who could obtain vaccination and when.

**Official evidence.** WHO published the SAGE values framework on 14 September 2020 to guide allocation and prioritization of COVID-19 vaccination. [4]
It articulates values and objectives for allocation decisions; it is not a retrospective causal evaluation of any particular allocation.
Its publication preceded many subsequent changes in product availability, evidence, and policy.

**Mechanism.** A prioritization rule can direct scarce doses toward groups with particular risks or social roles.
Whether it improves health or equity depends on vaccine properties, uptake, eligibility, implementation, and the counterfactual.
A rule on paper is not equivalent to delivered access.

**Transfer.** Make objectives explicit; separate global allocation from local prioritization; collect evidence on access and administration.
A hospital supply allocation can borrow the idea of transparent criteria and appeals.
It cannot inherit a vaccine priority list without checking clinical and ethical relevance.

**Unknowns.** The framework alone does not reveal the actual distribution of doses, effectiveness of outreach, or outcomes in any district.
Section 6's districts and figures are synthetic and are not attributed to WHO.

### Case 5 — DSCSA traceability

**Context and decision.** U.S. trading partners need to manage product and transaction data for covered prescription drugs under applicable DSCSA requirements.
A receiving organization may need to investigate a missing or inconsistent transaction record before proceeding.

**Official evidence.** FDA explains the DSCSA and publishes notices about waivers and exemptions. [5][6]
Those notices are the source for their specific terms, conditions, and dates.
They are not proof that every partner or product is covered by the same exception.

**Intervention.** The legal and operational framework supports product identification, transaction information, and verification processes in the covered supply chain.
Trading partners may implement interoperable systems and exception workflows to meet applicable duties.
GS1 standards can provide technical building blocks, but a data format is not synonymous with legal compliance.

**Causal boundary.** Serialization may enable investigations that would otherwise be harder.
A serial number alone does not establish authenticity, prevent diversion, or prove that a temperature condition was maintained.
No reduction in counterfeiting is quantified or attributed to DSCSA here.

**Transfer.** Separate item identity, custody events, ownership or transaction records, and environmental monitoring in any traceability design.
Check which records are actually required for the applicable jurisdiction and role.
A non-U.S. food supply chain may reuse the data-model lesson, not the U.S. legal rule.

### Case 6 — Haiti earthquake logistics

**Context and decision.** Following the January 2010 Haiti earthquake, responders faced urgent logistics and coordination needs.
Infrastructure damage and access limitations affected the movement of relief supplies.

**Official operational evidence.** The Logistics Cluster's 1 February 2010 consolidated situation report records contemporaneous logistics coordination and operating updates. [7]
The 15 February report discusses road and border-routing constraints, including vehicle limitations on a route. [8]
These are dated reports, not current road advice or a complete impact evaluation.

**Mechanism.** If a road or port has limited capacity, central cargo arrival may not translate into last-mile delivery.
Shared information and coordination can help organizations avoid duplicate effort and identify bottlenecks.
The degree to which any specific coordination action caused better outcomes is not measured by the cited situation reports.

**Transfer.** Track cargo by destination and need, check routes with local operators, and refresh access information frequently.
Treat affected people as participants with knowledge and agency, not as passive demand points.
Design a fallback for communications loss and use a feedback channel appropriate to the context.

**Unknowns.** A logistics report is not a census of everyone reached or missed.
A reported operational route may not have been equally safe or accessible to every community.
No specific lives-saved claim follows from the cited reports.

### Cross-case comparison

| Case | Primary decision | Binding or physical constraint | What the cited source supports | What it does not establish |
|---|---|---|---|---|
| ORION | Stop sequence and navigation | Road network, time windows, safe execution | UPS's description of its system | A locally transferable causal saving |
| NotPetya | Continue and restore services | Digital dependencies and integrity | Maersk's contemporaneous disruption report | Complete attack path or causal financial cost |
| Red Sea | Route and schedule | Security and route availability | UNCTAD's observed redirection and analysis | Exact delay for each shipment |
| Vaccines | Allocate scarce doses | Supply, safety, and ethical policy | WHO's stated values framework | Actual causal benefit of a local allocation |
| DSCSA | Exchange and investigate records | Applicable U.S. legal duties | FDA's explanation and notices | Automatic authenticity from a serial |
| Haiti | Prioritize and move relief | Damaged infrastructure and access | Dated operational situation reports | Complete coverage or lives saved |

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Copying a route saving | Same percentage appears in every proposal | Network and baseline differences ignored | Inflated business case | Run a bounded local comparison |
| Optimizing distance alone | Rising late arrivals or unsafe overrides | Objective omits service and safety | Harm despite lower miles | Set hard guardrails and track service |
| Stale road data | Frequent driver overrides at the same location | Map not refreshed | Infeasible routes | Add operator feedback and map ownership |
| Treating outage as only an IT issue | Booking or release processes have no manual owner | Business dependencies omitted | Prolonged operational disruption | Map services and exercise cross-functional recovery |
| Untested fallback | Backup instructions require primary identity service | Shared dependency defeats redundancy | Fallback fails during incident | Exercise under realistic dependency loss |
| Equating restoration and recovery | Server is up but backlog grows | Data or transactions remain unreconciled | Customers still lack service | Measure restoration, reconciliation, and backlog separately |
| Universal diversion assumption | One transit-time uplift used for all lanes | Origin, destination, and vessel differences ignored | Misstated inventory need | Segment by lane and scenario |
| Double-counted disruption cost | Fuel appears inside rate and again as a separate charge | Overlapping cost categories | Exaggerated cost estimate | Reconcile cost definitions and contracts |
| Allocation by requests only | Remote communities show low requests and low access | Barriers suppress observed demand | Need is systematically missed | Combine estimates with community input |
| Shipments confused with vaccinations | Dashboard has no administration field | Proxy treated as outcome | False coverage confidence | Track administered, wasted, and unserved doses |
| Hidden priority tie-break | First district in file always receives scarce doses | Input order becomes policy | Arbitrary inequity | Approve and publish tie-break before execution |
| Ignoring site capacity | Planned doses exceed refrigerator or staffing limit | Aggregate constraint substitutes for local constraint | Waste or delayed administration | Validate capacity by site and period |
| Treating a serial as proof | Matching code is accepted without investigation | Identity is confused with authenticity | Suspect product moves onward | Reconcile and investigate under applicable procedure |
| Assuming one DSCSA deadline | A presentation lists one date for every party | Role-specific notices ignored | Incorrect compliance decision | Review current FDA notices for the exact role |
| Treating event data as physical truth | Late or duplicate events never reconciled | Data-quality problems hidden | False chain-of-custody claim | Reconcile records with physical receipt |
| Using an old situation report as current | No fresh route confirmation | Access changes after publication | Unsafe or blocked relief movement | Verify access with local authorities and operators |
| Tonnage as sole relief metric | Central warehouse throughput rises while complaints increase | Last-mile access omitted | Unequal or mistimed assistance | Measure delivery and community feedback |
| Overclaiming causality | Before–after outcome labeled model effect | Concurrent changes not controlled | Wrong scale-up decision | Use credible comparison or label descriptive |
| Ignoring unobserved areas | Missing districts shown as zero need | Reporting gaps mistaken for absence | Systematic exclusion | Mark missingness and prioritize verification |
| No exception owner | Discrepancies accumulate without disposition | Accountability split across parties | Delays, unsafe workarounds | Assign owner, clock, and escalation path |

Each failure mode is a prompt for a control, not proof that a named organization suffered that failure.
A control is effective only if its owner can execute it under the conditions being tested.
For instance, a manual route override requires a channel that still works during the relevant outage.
A temperature alarm requires an authorized person who can assess and document disposition.
A community feedback mechanism requires a safe and accessible way to respond.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test or monitor |
|---|---|---|---|
| Forecast demand is a useful proxy for need | Allocation uses a demand denominator | Barriers suppress requests | Compare estimates, outreach, and community feedback |
| All doses in the example are interchangeable | Allows a single stock pool | Product, eligibility, or schedule differs | Reconcile lot and product eligibility |
| One dose corresponds to one eligible person in the example | Makes dose coverage interpretable | Multi-dose schedules or repeat doses | Model people and product schedule separately |
| No wastage in the example | Isolates the allocation rule | Open-vial loss or expiry occurs | Track actual wastage and recalculate usable supply |
| Capacity is known for the week | Defines feasible assignment | Staff or cold-chain outage | Confirm capacity before release and monitor daily |
| Secondary order is approved | Makes remaining-dose assignment deterministic | Policy changes or local priorities differ | Obtain documented authority and consultation |
| Route inputs reflect decision-time information | Avoids hindsight bias | Late road closure or leaked future data | Preserve point-in-time snapshots |
| Comparison groups are comparable | Supports effect estimation | Different route density or seasonality | Randomize where feasible or examine pre-trends |
| Source claims have the stated scope | Makes evidence cards reproducible | A later summary drops conditions | Read original dated document and capture context |
| Reports capture material incidents | Supports operational situational awareness | Communication failures hide events | Cross-check independent field reports |
| Serial and event records correspond to physical objects | Supports reconciliation | Duplication, tampering, or scanning errors | Investigate mismatches and sample physical records |
| Route alternatives are authorized and safe | Makes alternatives actionable | Security or permit conditions change | Obtain current competent-authority assessment |
| Manual fallback is independent enough | Allows continuity when primary systems fail | Shared identity or network dependency | Exercise with dependencies deliberately unavailable |
| Aggregate outcomes are comparable | Supports cross-option comparison | Denominators, periods, or geography differ | Normalize definitions and show subgroup results |
| Ethical weights are legitimate | Allows a ranking of options | Stakeholders disagree or lack representation | Consult decision authority and affected groups |
| Cited operational reports are historical | Prevents use as live route advice | Reader treats publication as current | Display dates and refresh before any operation |

The example's feasibility is conditional on the stated assumptions.
The six real cases differ in time, sector, scale, and source quality.
No quantitative effect is transferred from one case to another.
The absence of a cited effect is not proof that an effect was zero.
It means this chapter does not claim a quantified effect on the basis of the cited evidence.

## 12. Exercises

### Exercise 1 — Evidence classification

**Time:** 15 minutes.
**Task:** Classify each statement as source-supported evidence, attributed claim, inference, or unsupported causal claim.

1. UPS describes UPSNav as using directions derived from routing software.
2. Every carrier implementing route recommendations will reduce costs by the same percentage.
3. A longer voyage can increase inventory in transit if demand continues during travel.
4. A dated Haiti situation report is a complete census of unmet need.
5. WHO's SAGE framework provides values guidance for COVID-19 vaccine allocation.
6. An observed reduction in miles after a tool rollout proves the tool caused all of it.

**Deliverable:** Six classifications, each with one sentence of reasoning.
**Acceptance criteria:** Identify the source and distinguish possibility from observed or causal effect.

### Exercise 2 — Case anatomy

**Time:** 25 minutes.
**Task:** Choose either Maersk or the Red Sea disruption.
Describe context, decision owner, decision, data, action, outcome evidence, transfer, and limitations.
Use the dated primary source in section 16.

**Deliverable:** One eight-row case card.
**Acceptance criteria:** At least two unknowns; no fabricated incident cost or shipment delay; a clear counterfactual limitation.

### Exercise 3 — Allocation calculation

**Time:** 25 minutes.
**Task:** Using section 6, compute the base priority-first allocation and spare capacity by district.
Then reduce Remote capacity to 170 doses and recompute.

**Deliverable:** Two tables, each with priority, other, total, and spare capacity in doses.
**Acceptance criteria:** Respect site capacity and total stock; show the 20-dose change in priority coverage.

### Exercise 4 — Compare rules

**Time:** 25 minutes.
**Task:** Compute proportional allocation by total demand using the largest-remainder method.
Compare it with priority-first allocation on aggregate throughput and the minimum uncovered North priority demand.

**Deliverable:** A table and a 150-word decision note.
**Acceptance criteria:** Allocate exactly 1,000 whole doses; obtain 389, 389, and 222; distinguish a fairness value judgment from arithmetic.

### Exercise 5 — Measurement design

**Time:** 30 minutes.
**Task:** Design a route-recommendation evaluation for a fictional parcel operator.
Name a comparison, unit, outcome, two guardrails, two confounders, and an override process.

**Deliverable:** A one-page measurement plan.
**Acceptance criteria:** Comparison is feasible at decision time; no causal claim from a raw before–after change; safety and service are monitored.

### Exercise 6 — Transfer test

**Time:** 20 minutes.
**Task:** Consider applying an ORION-like recommendation to a rural mobile clinic.
Answer the five transfer questions, marking yes, no, or unknown with a reason.

**Deliverable:** A five-row table and a conditional recommendation.
**Acceptance criteria:** Include clinical and access constraints; treat unknown as unresolved; do not import a UPS benefit percentage.

### Exercise 7 — Traceability exception

**Time:** 20 minutes.
**Task:** A fictional receiver has a physical case and a product code, but the associated electronic transaction data are missing.
Describe what is known, what remains unknown, who should be involved, and what should happen next.
Do not decide legal applicability from this short scenario.

**Deliverable:** A four-part exception note.
**Acceptance criteria:** Distinguish item identity from transaction evidence and authenticity; use an authorized hold or escalation process as applicable; cite current FDA materials for legal review.

### Exercise 8 — Humanitarian logistics brief

**Time:** 35 minutes.
**Task:** An earthquake has made one of two roads intermittently passable.
A dated situation report identifies the constraint but does not establish present conditions.
Prepare a brief for the logistics coordinator.

**Deliverable:** One page covering decision, cargo priority, route verification, safety, community feedback, uncertainty, and revision trigger.
**Acceptance criteria:** No claim that an old report grants access; no invented beneficiary count; named authority and fallback.

## 13. Self-check answers

### Exercise 1

1. The UPS announcement supports an attributed company description of its system.
   It does not independently demonstrate a measured effect.
2. Unsupported generalization and causal claim.
   Networks, constraints, and baselines differ; no universal percentage follows from the source.
3. Inference about a plausible mechanism.
   More time in transit can mean more inventory in transit when flow continues, but actual effects require lane-specific data.
4. Unsupported claim.
   The situation report is a dated operational snapshot, not a complete census.
5. Source-supported description of WHO's published framework.
   Its existence and purpose do not establish the impact of a particular national allocation.
6. Unsupported causal claim.
   Workload, geography, weather, and other concurrent changes may explain part of the difference.

### Exercise 2 rubric

A strong card identifies the actual source and its date.
It names a decision and owner without asserting a person's identity not present in the source.
It distinguishes a reported disruption or diversion from a measured causal loss.
It identifies at least two missing facts, such as per-customer impact, comparable counterfactual, or current route availability.
Its transfer recommendation is conditional on a local dependency or lane assessment.
A weak card repeats a headline without specifying an actionable choice.

### Exercise 3

Base scenario:

| District | Priority | Other | Total | Spare capacity |
|---|---:|---:|---:|---:|
| North | 420 | 30 | 450 | 50 |
| Central | 310 | 0 | 310 | 140 |
| Remote | 190 | 50 | 240 | 0 |

All quantities are doses for one week.
Priority coverage is 920 doses and total allocation is 1,000 doses.
The 80 remaining doses after priority go first to Remote, which can accept 50, then to North, which receives 30.

Reduced-capacity scenario:

| District | Priority | Other | Total | Spare capacity |
|---|---:|---:|---:|---:|
| North | 420 | 80 | 500 | 0 |
| Central | 310 | 20 | 330 | 120 |
| Remote | 170 | 0 | 170 | 0 |

Priority coverage is 900 doses.
The difference is 920 - 900 = 20 doses.
Total allocated doses remain 1,000, so aggregate throughput misses the local priority shortfall.

### Exercise 4

Total demand is 1,800 doses.
The exact quotas are 388.888…, 388.888…, and 222.222… doses.
After flooring, two doses remain.
The largest remainders are North and Central, so the rounded allocations are 389, 389, and 222 doses.
North's priority demand is 420 doses, so at least 420 - 389 = 31 priority doses are uncovered there.
Both rules allocate 1,000 doses in aggregate.
The priority-first rule covers all forecast priority demand in the base scenario, whereas proportional allocation alone cannot do so.
Choosing between them requires an authorized ethical and policy judgment; arithmetic alone is insufficient.

### Exercise 5 rubric

A strong plan states the unit and the feasible baseline.
It proposes randomized vehicle-days or a defensible alternative where randomization is not feasible.
It records miles per comparable vehicle-day, on-time service, safety, and driver overrides.
It accounts for route density and weather and documents spillover or missing-data risks.
It describes who can override a recommendation and how reasons feed back into map correction.
A weak plan compares only one month before and after without acknowledging concurrent changes.

### Exercise 6 rubric

A strong answer identifies a similar sequencing decision but different clinical consequences and potentially sparse geography.
It treats local location accuracy, visit duration, patient access, and clinical time windows as unknown until checked.
It recommends a bounded local trial only if approved safety and access constraints are met.
It does not assume UPS's company-reported outcomes transfer.

### Exercise 7 rubric

The physical case and code are observations, subject to verification.
The missing transaction data remain missing; the product code does not prove authenticity or establish a compliant transaction.
The receiver should involve its designated quality, compliance, and trading-partner contacts.
Use the organization's authorized exception, hold, and investigation procedures as applicable.
A qualified party must determine current legal obligations from the relevant statute and FDA notices.
Do not invent or backfill an event as though it were observed.

### Exercise 8 rubric

A strong brief uses the old report as a lead and obtains fresh route verification.
It names the coordinator, the safety authority, the information source, and the decision deadline.
It prioritizes cargo using an explicit need criterion and identifies who can revise it.
It includes local community input, communication fallback, and a trigger for stopping or rerouting.
A weak brief reports aggregate tonnes as proof that all affected people received assistance.

## 14. Interview questions

### 1. What does ORION demonstrate?

A strong answer says UPS describes a route-derived navigation capability and that a local effect needs its own baseline and evaluation.
A weak answer presents a company claim as a universal causal saving.

### 2. How would you evaluate a routing recommendation?

A strong answer defines the vehicle-day, a feasible comparator, and service and safety guardrails.
It addresses route mix, overrides, and point-in-time data.
A weak answer reports only total mileage before and after rollout.

### 3. What does the Maersk incident teach about resilience?

A strong answer distinguishes digital restoration, data integrity, and business recovery.
It proposes dependency mapping and a tested fallback without inventing the incident's root cause.
A weak answer assumes restoring a server restores all customer service.

### 4. How would you model a Red Sea diversion?

A strong answer uses lane-specific route options, uncertain lead times, inventory consequences, and security constraints.
A weak answer applies a single delay to all shipments and attributes every cost increase to one event.

### 5. Why is a vaccine allocation not the same as an administration?

A strong answer distinguishes released stock, assigned doses, shipment, receipt, administration, and wastage.
It discusses access barriers and relevant denominators.
A weak answer calls every shipped dose a vaccinated person.

### 6. How can an allocation use all doses and still fail an equity objective?

A strong answer uses the Remote capacity scenario: total allocation remains 1,000 doses while priority coverage falls by 20 doses.
It explains that equity requires a declared criterion and subgroup measurement.
A weak answer equates maximum throughput with fairness by definition.

### 7. Does a serialized product code prove authenticity?

A strong answer says no; identifiers, transaction data, verification, and physical handling are separate issues.
It calls for the applicable authorized investigation.
A weak answer treats a scanned code as an unconditional authenticity certificate.

### 8. What is the role of FDA exemptions in a DSCSA assessment?

A strong answer checks the specific party, date, covered requirement, conditions, and current notice.
A weak answer announces a blanket exemption for everyone.

### 9. What is the limitation of a Haiti logistics situation report?

A strong answer recognizes a dated operational snapshot that may omit inaccessible or unreported areas.
A weak answer treats reported throughput as complete proof of beneficiary coverage.

### 10. What transfers across these six cases?

A strong answer transfers decision framing, constraint mapping, traceable evidence, and feedback.
It explicitly rejects untested transfer of outcome percentages or legal rules.
A weak answer says every network should maximize the same efficiency metric.

## 15. Key terms

- **[Allocation](#4-definitions-and-vocabulary):** assignment of scarce supply to recipients.
- **[Administration](#4-definitions-and-vocabulary):** delivery of a dose to a person.
- **[Baseline](#5-the-mechanics-step-by-step):** feasible comparator available at decision time.
- **[Capacity](#6-worked-example):** maximum safely usable site throughput in the period.
- **[Case evidence](#9-industry-practice-and-case-evidence):** documented information about a bounded event or intervention.
- **[Cold chain](#8-data-standards-and-regulatory-context):** controlled-temperature handling throughout relevant stages.
- **[Commitment](#6-worked-example):** authorized promise distinct from a plan or forecast.
- **[Counterfactual](#3-the-core-mental-model):** outcome under an alternative action not taken.
- **[Cyber continuity](#8-data-standards-and-regulatory-context):** sustaining critical services through digital disruption.
- **[Decision unit](#4-definitions-and-vocabulary):** object acted on, such as a vehicle-day.
- **[DSCSA](#8-data-standards-and-regulatory-context):** U.S. drug supply-chain security law.
- **[Equity](#6-worked-example):** declared fairness criterion, not necessarily equal shares.
- **[Event data](#8-data-standards-and-regulatory-context):** records describing a reported event and its attributes.
- **[Exception](#10-failure-modes-and-red-flags):** discrepancy requiring authorized handling.
- **[Forecast](#6-worked-example):** estimate of future demand, not a promise.
- **[GTIN](#4-definitions-and-vocabulary):** GS1 trade-item identifier, not a unique serial by itself.
- **[Last mile](#9-industry-practice-and-case-evidence):** final movement toward the service point or recipient.
- **[Lead time](#8-data-standards-and-regulatory-context):** elapsed time between explicitly named events.
- **[Logistics Cluster](#9-industry-practice-and-case-evidence):** humanitarian logistics coordination mechanism.
- **[ORION](#9-industry-practice-and-case-evidence):** UPS route-optimization system.
- **[Override](#5-the-mechanics-step-by-step):** authorized departure from a recommendation.
- **[Plan](#6-worked-example):** intended allocation or action under assumptions.
- **[Priority group](#6-worked-example):** group favored by an explicitly authorized policy.
- **[Product identifier](#8-data-standards-and-regulatory-context):** prescribed item-identification information.
- **[Provenance](#8-data-standards-and-regulatory-context):** source and transformation history of a datum.
- **[Recovery point objective](#8-data-standards-and-regulatory-context):** maximum tolerable data-loss interval.
- **[Recovery time objective](#8-data-standards-and-regulatory-context):** desired restoration interval.
- **[Reroute](#9-industry-practice-and-case-evidence):** alternative physical path for a shipment.
- **[Sensitivity analysis](#6-worked-example):** comparison under changed assumptions.
- **[Serial number](#8-data-standards-and-regulatory-context):** identifier for an individual instance in its coding context.
- **[Throughput](#10-failure-modes-and-red-flags):** quantity passing a specified point per period.
- **[Traceability](#9-industry-practice-and-case-evidence):** ability to link relevant identifiers and records through a supply chain.
- **[Transfer test](#5-the-mechanics-step-by-step):** structured check of decision, economics, data, constraints, and evidence.
- **[Uncertainty](#11-assumptions-and-limits):** incomplete knowledge affecting interpretation or choice.

## 16. Sources

Sources below are primary institutional or company sources for the specific statements attributed to them.
Publication date and institutional role matter; source type is not a substitute for causal identification.
URLs and regulatory materials should be checked for updates before operational use.

1. UPS, [“UPS Deploys Purpose-Built Navigation For UPS Service Personnel”](https://about.ups.com/us/en/newsroom/press-releases/innovation-driven/ups-deploys-purpose-built-navigation-for-ups-service-personnel.html).
   Company announcement describing UPSNav and its relationship to ORION; company account, course grade C.
2. A.P. Moller - Maersk, [“Cyber attack update,” 28 June 2017](https://investor.maersk.com/news-releases/news-release-details/cyber-attack-update).
   Contemporaneous company announcement about affected systems; company account, grade C.
3. UN Trade and Development, [“Red Sea Crisis and implications for trade facilitation in Africa”](https://unctad.org/news/red-sea-crisis-and-implications-trade-facilitation-africa).
   Institutional analysis of route redirection and trade implications; official analysis, grade B for its attributed observations.
4. World Health Organization, [“WHO SAGE values framework for the allocation and prioritization of COVID-19 vaccination,” 14 September 2020](https://www.who.int/publications/i/item/who-sage-values-framework-for-the-allocation-and-prioritization-of-covid-19-vaccination).
   Primary ethical guidance, not an evaluation of section 6's synthetic allocation; official guidance, grade B.
5. U.S. Food and Drug Administration, [“Drug Supply Chain Security Act”](https://www.fda.gov/drugs/drug-supply-chain-integrity/drug-supply-chain-security-act-dscsa).
   Regulatory overview; consult the applicable statute and current agency materials for specific obligations, grade B.
6. U.S. Food and Drug Administration, [“Waivers and Exemptions Beyond the Stabilization Period”](https://www.fda.gov/drugs/drug-supply-chain-security-act-dscsa/waivers-and-exemptions-beyond-stabilization-period).
   Agency notices and links on specific exemptions; applicability and conditions require individual review, grade B.
7. Logistics Cluster, [“Situation Report (Consolidated) - 1 February 2010”](https://logcluster.org/en/documents/situation-report-consolidated-1-february-2010).
   Contemporaneous Haiti response situation report; operational primary source, grade B for reported facts.
8. Logistics Cluster, [“Situation Report (Consolidated) - 15 February 2010”](https://logcluster.org/en/documents/situation-report-consolidated-15-february-2010).
   Contemporaneous report on logistics and access; not current route guidance, grade B for reported facts.
9. GS1, [EPCIS standard](https://www.gs1.org/standards/epcis).
   Standards-body description of event-data exchange; technical convention, not an independent compliance determination.
10. U.S. Congress, [Drug Supply Chain Security Act, Public Law 113-54, Title II](https://www.congress.gov/113/plaws/publ54/PLAW-113publ54.pdf).
    Original enacted statutory text; check subsequent amendments and current codification before applying it.

## 17. Links

- [Module 28 topic index](_index.md).
- [Case anatomy](01_case-anatomy.md).
- [Evidence grades](02_evidence-grades.md).
- [The transfer test](03_transfer-test.md).
- [Sector playbooks](05_sector-playbooks.md).
- [Module 28 survey](../../../course/28_domain_case_library.md).
- [Simulation and optimization production labs](../../05_labs_electives/30_simulation_optimization_production_labs/_index.md).
