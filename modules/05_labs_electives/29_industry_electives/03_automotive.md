# Automotive: Sequenced Supply, Changeovers, and Recall Exposure

> Module 29 — Industry Electives: Sector Primers  
> Topic 03 of 08  
> Estimated study time: 4 hours  
> Prerequisites: Modules 14, 15, and 28  
> Version 2026.5

## 1. Why this topic exists

An assembly plant can have thousands of parts available and still be unable to finish a vehicle because one required component is missing.
The buyer's problem is not simply to minimize the purchase price of each part.
It is to deliver the correct, approved part, at the correct plant, in the correct variant and sequence, at a cost and risk the business can accept.

A seat set that arrives on time but in the wrong vehicle order can be as disruptive as a late truck.
A chip that is plentiful at a distributor may be unusable if it has not been qualified for the electronic control unit.
A model-year change can leave an apparently healthy inventory position filled with parts that no longer fit the build schedule.
A safety defect can require a recall whose affected population is much larger than the supplier's known defective lot if traceability is incomplete.

Different decision makers own different parts of this problem:

- The plant material planner protects near-term build continuity.
- The just-in-sequence supplier dispatches variant-specific material against a production sequence.
- Purchasing negotiates price, capacity, tooling, and commercial responsibility.
- Engineering controls approved specifications and design changes.
- Quality decides containment, disposition, and traceability requirements.
- Finance compares the cost of mitigation with the distribution of losses.
- Legal and regulatory teams determine applicable duties and communications.
- The original equipment manufacturer (OEM), suppliers, dealers, and regulators may each have separate obligations and information.

These actors optimize different objectives and cannot resolve the trade-off by looking at inventory turns alone.
A plant may deliberately carry a small strategic buffer for a long-lead semiconductor while keeping bulky seats on a short sequence loop.
The correct policy depends on the part, the network, the feasible response time, and the value of an interrupted or incorrectly built vehicle.

This chapter builds a decision process for these trade-offs.
The numerical company, parts, costs, and case scenarios are **synthetic**, not claims about a real manufacturer.
Public cases in Section 9 are separately attributed.
Regulatory discussion is educational, not legal advice; requirements vary by jurisdiction, product, and date.

## 2. Learning outcomes

After completing this unit, you can:

1. Draw a tiered automotive supply network and identify a hidden common dependency; demonstrate this in Exercise 1.
2. Calculate stock coverage, sequence timing, and the condition under which a buffer cannot protect the line; demonstrate this in Exercises 2 and 3.
3. Compare feasible continuity options using expected incremental cost and capacity constraints; demonstrate this in Exercise 4.
4. Construct a model-year cutover plan using part effectivity and inventory disposition; demonstrate this in Exercise 5.
5. Bound an affected vehicle population from lot-to-vehicle genealogy and explain the effect of missing records; demonstrate this in Exercise 6.
6. Critique a supplier scorecard and an apparently favorable average service measure; demonstrate this in Exercise 7.
7. Design an executable response and ownership model for a combined shortage and quality event; demonstrate this in Exercise 8.
8. Explain the distinction between an industry standard, a contract, and a legal requirement; demonstrate this in Exercises 6 and 8.

Use the answer section for calculation and recall questions.
For open designs, compare your work with the rubric rather than treating one plan as uniquely correct.

## 3. The core mental model

### Physical flow and information flow

```text
Raw material -> Tier 3 -> Tier 2 -> Tier 1 -> OEM assembly -> dealer -> vehicle owner
                 |         |         |            |
                 +------ shared foundry / logistics node ------+

OEM sales and build plan -> call-offs -> supplier capacity and material plans
Frozen vehicle sequence -> JIS instruction -> sequenced rack -> line-side scan
Part lot / serial -> component -> vehicle identification number -> field history
Engineering release -> approved effectivity -> supplier and plant cutover controls
```

The physical arrows run toward the vehicle, but information must travel in both directions.
An OEM's direct supplier is commonly called Tier 1.
That supplier buys from Tier 2; Tier 2 may buy from Tier 3.
These labels describe position relative to a particular OEM, not an intrinsic rank of a company.
A single supplier can occupy different tiers in different programs.

The shared node in the diagram matters because two apparently independent Tier 1 suppliers may depend on the same foundry, molding plant, port, utility, or sub-tier tool.
A dual-source contract is not equivalent to independent physical capacity.
A meaningful risk map follows part, plant, tool, and sub-tier relationships far enough to identify the constraint.

The same vehicle is also a temporal object.
Before it is built, it appears as a forecast, a planned order, a scheduled build, and then a sequence instruction.
After it is built, its vehicle identification number (VIN) connects it to components, repairs, and field information.
None of these records automatically proves the others are correct.

### Three linked clocks

**The replenishment clock** starts when supply needs to be ordered and ends when usable material is available.
It includes production, transport, receiving, inspection, and release.
A long average lead time is important, but uncertainty and the time needed to recover from a disruption can be more important.

**The sequence clock** starts when a specific vehicle order is frozen or broadcast and ends at the line-side consumption point.
It determines how much time a JIS supplier has to assemble, pick, transport, unload, and verify the right variant in the right order.
A supplier may have sufficient daily capacity but still miss this much shorter deadline.

**The change and traceability clock** covers design releases, effective build dates or serial ranges, and the period during which component-to-vehicle records must support investigation.
Its relevant duration can exceed a single production week by years.
A rushed change can appear to solve today's shortage while creating a later field exposure.

### Separate objective from constraints

The primary business objective is continuity and customer safety at an acceptable total cost, not maximum inventory or minimum price in isolation.
Safety, product approval, sequence correctness, and regulatory compliance are constraints on candidate actions.
If an action fails a mandatory constraint, do not assign it a low expected cost and select it anyway.
For feasible actions, compare incremental costs under explicit demand and disruption scenarios.

**Forecast** means an estimate of possible future vehicle demand.
**Plan** means the quantity and mix the organization intends to build.
**Target** is a desired performance level, such as a service goal.
**Commitment** is an accepted obligation or allocation, subject to its stated terms.
A forecast is not a supplier commitment merely because a spreadsheet labels it as demand.

### Units and direction

Measure a line rate in vehicles per hour and a component requirement in parts per vehicle.
Multiply them to obtain parts per hour.
Express coverage in hours only after accounting for usable stock and the mix actually needed.
A single average coverage number can conceal an immediate shortage of one variant.

For a homogeneous part:

```text
usable coverage (hours) = usable available parts / consumption rate (parts/hour)
```

For a mixed sequence, calculate the first scheduled position that cannot be filled, rather than assuming all variants can substitute for each other.
The earliest unfillable position is a more useful dispatch signal than aggregate inventory days.

## 4. Definitions and vocabulary

| Term | Definition and unit | Common misuse |
|---|---|---|
| OEM | Vehicle maker that specifies or assembles the vehicle | Assuming the OEM makes every component |
| Tier 1 | Supplier shipping a component or system directly to the OEM in a given network | Treating the direct supplier as the only risk source |
| Tier 2 / Tier 3 | Suppliers upstream of the direct supplier | Assuming tier labels imply independent ownership or sites |
| Part number | Controlled identifier for an engineering item and usually a revision context | Assuming two similar descriptions are interchangeable |
| Variant | A configuration, such as left/right or trim color, requiring a specific item | Pooling all variants into one stock total |
| Bill of materials (BOM) | Structured component requirements for a product configuration | Using an obsolete or unapproved BOM for allocation |
| Effectivity | Rule stating where and when a revision applies, such as VIN, date, or build number | Assuming a calendar date alone describes a cutover |
| Model year | Manufacturer and regulatory vehicle designation that may differ from calendar year | Assuming every part changes on January 1 |
| Changeover | Transition between approved configurations, tooling, or production processes | Treating a design release as proof all sites are ready |
| Engineering change | Controlled change to design or manufacturing requirements | Assuming a requested change is already approved |
| Just-in-time (JIT) | Supplying material near its consumption time to limit inventory | Equating it with zero inventory everywhere |
| Just-in-sequence (JIS) | Supplying items in the order and variants required by a specific build sequence | Treating punctual arrival as proof of correct order |
| Call-off | Customer release or instruction against a supply arrangement | Treating every horizon as equally binding |
| Frozen sequence | Ordered build list for a defined time window with controlled change rules | Assuming no resequencing or exceptions can occur |
| Takt / line rate | Available production time per unit, or its inverse in units per hour | Confusing planned rate with actual throughput |
| Cycle time | Time required for a process to complete a unit | Assuming it equals takt at every station |
| Pitch | Defined production interval for a group of units | Treating pitch as necessarily one vehicle |
| Line-side stock | Material available at the consumption point | Counting blocked or inaccessible stock as usable |
| Supermarket | Controlled point-of-use replenishment buffer | Calling an unbounded storage area a pull system |
| Kanban | Signal to replenish a defined quantity under agreed rules | Treating any inventory report as a kanban |
| Expedite | Exceptional acceleration of supply, often using premium transport | Assuming transport alone can create unbuilt capacity |
| Line stop | Period in which the assembly line cannot operate as planned | Multiplying all elapsed hours by cost without checking recovery |
| Bottleneck | Resource that constrains throughput in the stated period | Assuming the same resource constrains every scenario |
| Capacity reservation | Agreed entitlement to production or supply capacity | Treating an informal forecast as reserved capacity |
| Tooling | Molds, dies, fixtures, or other dedicated production equipment | Assuming a second supplier has an approved duplicate tool |
| Traceability / genealogy | Recorded links among material, processes, components, and finished vehicles | Assuming shipment receipt proves installation in a specific VIN |
| Lot / batch | Defined quantity produced or controlled together | Assuming all suppliers use the same lot boundary |
| Serial number | Identifier for an individual instance | Calling a lot number an individual serial |
| VIN | Vehicle identification number assigned to a vehicle | Treating a production order number as a VIN without a reliable mapping |
| Containment | Action to isolate suspect material while facts are established | Presenting a hold as a final root-cause determination |
| Recall | Corrective action for an identified population under applicable rules | Assuming every quality campaign is legally a safety recall |
| Resequencing | Changing the planned vehicle order | Assuming every downstream supplier can absorb unlimited changes |
| Obsolescence | Loss in value when inventory cannot be used or sold as planned | Counting all old stock as scrap before checking approved uses |
| Electronic data interchange (EDI) | Structured exchange of business messages between organizations | Assuming message delivery proves physical receipt or use |
| Advanced shipment notice (ASN) | Electronic notice describing a shipment before arrival | Assuming an ASN replaces receiving and quality checks |
| Probability | Frequency or subjective estimate of an event under a specified horizon | Mixing a weekly probability with an annual loss |
| Expected cost | Probability-weighted sum of costs in the defined scenarios | Treating it as the worst possible loss or a safety approval |

## 5. The mechanics, step by step

### Step 1 — State the decision and its owner

Write a decision sentence before extracting data.
For example: "By 14:00 today, the plant materials manager must select the feasible seat-delivery protection for next week's 400-vehicle build."
Specify who approves spending, who can stop production, and who can accept a product substitution.
Record the latest decision time and the time at which each mitigation must start.

Inputs are the build horizon, line schedule, current stock, and authority matrix.
Output is a decision record with one accountable owner and an escalation deadline.
If the quality disposition is unknown, treat suspect material as unavailable until the authorized function releases it.
A purchasing manager's willingness to use it is not a quality release.

Do not collapse strategic and operational decisions.
A supplier nomination may have a multi-year horizon; a truck dispatch may have a 30-minute horizon.
The same data can inform both, but the feasible actions and costs are different.

### Step 2 — Map the product and network

Start with the plant, product family, model year, and relevant build configurations.
Expand the approved BOM to identify required part numbers, revisions, and quantities per vehicle.
Join each part to its receiving plant, direct supplier site, and current transport lane.
For critical parts, add known sub-tier sites, tooling, raw materials, and logistics dependencies.

Ask the direct supplier to validate the map rather than assuming the purchase-order address is the manufacturing location.
Record evidence date and confidence for each relationship.
A tier map with unknown fields is better than one that silently assumes independence.
Avoid demanding proprietary sub-tier details that the supplier cannot legally or contractually disclose without an agreed information-sharing arrangement.
A confidential common-node identifier or attestation can still support a concentration analysis.

For each node, capture:

- Site and country, with manufacturing and shipping sites distinguished.
- Part, revision, and vehicle programs served.
- Rated and demonstrated capacity in parts per hour and per shift.
- Tool ownership, duplicate tooling, and qualification status.
- Recovery time estimate and evidence supporting it.
- Alternate supplier and alternate site, if approved.
- Transport mode, lane, customs boundary, and typical variability.
- Shared utility, common foundry, resin, or other material dependency.
- Inventory ownership, location, condition, and whether it is reachable in time.

The result is a dependency graph, not merely a supplier ranking.
A site with low annual spend may be critical if it makes the only approved safety component.

### Step 3 — Translate the build schedule into consumption

For each vehicle order, select its approved BOM using configuration and effectivity.
Multiply each required quantity by the scheduled vehicle count.
For a sequenced part, preserve the position and required variant; do not aggregate away order.
Include expected schedule changes, but keep the committed frozen sequence distinct from the mutable forecast.

Example: 60 vehicles per hour each consume one seat set.
If 40% of builds require black seats, average black-seat demand is 24 sets per hour.
That average does not ensure the next ten vehicles have only four black-seat requirements.
If the next ten vehicles all need black seats and only six black sets are available, the seventh vehicle in that run is the first at risk.

Check that quantity units match.
A seat set per vehicle is not the same as individual seats per vehicle.
A carton, rack, or pallet conversion must be explicit and versioned.

### Step 4 — Establish usable material and time to shortage

Reconcile on-hand balances with receiving, quality holds, line-side location, and inbound status.
Classify stock as usable now, usable after a defined release, in transit, planned but not shipped, or unavailable.
Do not count a truck as available at the line merely because an ASN exists.
For in-transit material, estimate arrival at the point of consumption, not only arrival at the gate.

For a single interchangeable part, divide usable stock by consumption rate.
For multiple variants, replay the sequence against available stock and confirmed arrivals.
If a scheduled arrival occurs at the same timestamp as consumption, define whether receiving and verification finish before or after that consumption.
Do not hide this boundary condition inside a simulation.

A simple recovery decision compares:

```text
remaining usable coverage + confirmed usable arrivals
versus
consumption until the next feasible replenishment
```

The first shortage may precede the next truck even if the end-of-shift balance is positive.
In a JIS flow, determine the first incorrect or missing sequence position as well as the first zero balance.

### Step 5 — Assess JIT and JIS response windows

For a JIS supplier, subtract each required activity from the time the assembly line will consume the item:

1. Sequence message transmission and acknowledgement.
2. Supplier pick or build and inspection.
3. Rack loading, dispatch, and transport.
4. Plant gate, unloading, and movement to line side.
5. Scan, verification, and contingency margin.

The remaining margin is the time available for errors and variability.
If it is negative under normal conditions, the design is infeasible without an earlier signal, local buffer, faster process, or changed schedule.
If it is positive only at the mean travel time, check a high-percentile travel time and correlated delays.
A percentile from a small or unrepresentative sample is not a guarantee.

Resequencing has a cost and a limit.
It may avoid one shortage while moving another variant forward or disrupting paint, body, or supplier sequencing constraints.
Confirm which vehicles can move, who authorizes the change, and how updated sequence messages reach every affected supplier.
A local dispatcher should not invent a substitute vehicle order.

### Step 6 — Identify feasible interventions

List actions before assigning costs.
Typical candidates include a local buffer, additional supplier shift, premium freight, approved alternate site, controlled resequencing, prebuilt assemblies, or a scheduled plant pause.
Each candidate needs a start time, completion time, quantity, owner, and evidence of feasibility.

For an alternate site, verify tooling, capacity, quality approval, and applicable customer-specific requirements.
A quote alone is not an approved supply route.
For premium freight, verify that the material exists and can be released before the truck or aircraft departs.
For a buffer, check part shelf life, revision, physical space, financing, ownership, and obsolescence risk.
For prebuilt JIS material, check variant mismatch and sequence changes, not just available floor space.

Reject actions that violate engineering, safety, legal, or operational constraints.
Document rejected actions and the constraint, so that a later review does not mistake them for overlooked choices.

### Step 7 — Compare incremental decision economics

Set a consistent baseline and horizon.
Include costs that differ between feasible actions: inventory carrying, incremental transport, overtime, qualification, expected lost contribution or interruption cost, and expected obsolescence.
Avoid charging the same lost vehicle twice as both a line-stop rate and a lost contribution amount if the rate already contains that contribution.
Consider recovery: overtime may recover units, while a permanently lost build may reduce contribution.

For each scenario, calculate total incremental cost under each action.
Multiply scenario cost by the scenario probability, ensuring probabilities are mutually exclusive and sum to one.
Use the same time horizon for fixed costs, disruption probabilities, and interruption losses.
Then compare expected cost, worst modeled loss, cash timing, and strategic constraints.

Expected cost is not the only decision criterion.
A rare but severe event may exceed a liquidity, safety, or customer-service limit even when the expected value favors accepting it.
Conversely, treating an arbitrary worst case as certain can justify excessive cost.
Present scenario assumptions and sensitivity thresholds to the approving executive.

### Step 8 — Plan the model-year cutover

Start from an approved engineering change and its exact effectivity rules.
A model-year designation is not itself an inventory instruction.
The transition may depend on plant, configuration, VIN range, component serial range, or date and shift.
Old and new vehicles can be built in parallel, and service parts may require the old revision after production switches.

For each part, create a cutover table:

- Old and new part numbers and approved revisions.
- Last eligible old build and first eligible new build.
- Configuration and plant applicability.
- Quantity per vehicle and firm versus forecast builds.
- Old and new supplier inventory by location and ownership.
- Pipeline quantities, transit times, and cancellation terms.
- Open engineering, tooling, packaging, and inspection approvals.
- Required service stock, approved rework, return, or scrap route.
- Owner, milestone, and decision deadline.

Compute old-part demand only for eligible remaining builds, plus authorized service or contingency requirements.
Subtract usable stock and confirmed receipts to identify a shortage or excess.
Do not subtract suspect stock until quality has released it.
Do not declare an excess scrap until service, approved rework, supplier return, and contractual ownership have been reviewed.

A cutover gate should require engineering release, supplier readiness, material availability, plant work instructions, traceability, and a rollback or containment plan.
If one gate fails, the accountable change board decides whether an approved temporary deviation exists or the build must change.
A commercial desire to consume old inventory does not authorize installation outside effectivity.

### Step 9 — Build recall-ready genealogy

At receiving, capture supplier, part revision, lot or serial, quantity, and physical location.
At transformation, link input lots to output component lots or serial numbers.
At assembly, link the actual component or controlled lot to the actual VIN, position, timestamp, and station.
At shipment and field service, maintain correction and repair records under applicable retention and privacy rules.

Track three distinct states: known affected, known not affected with evidence, and unknown.
Unknown is not equivalent to unaffected.
If installation records are incomplete, the possible population may include all vehicles built during an exposure window, subject to the investigation's actual evidence and legal assessment.
Do not promise that better traceability always permits a smaller legally defined recall; regulators and technical facts determine the population.

Validate the graph in both directions.
A forward query asks which VINs may contain a suspect lot.
A backward query asks which lots and processes contributed to one VIN.
Reconcile recorded installations to production counts and investigate missing, duplicate, impossible, or late records.
A scan at a dock establishes receipt, not installation.

### Step 10 — Operate and learn

Run a short daily exception review for near-term shortages and a longer periodic review for structural risks.
Display the next risk time, affected sequence positions, confirmed response, and owner.
Use a common clock and time zone; distinguish planned, confirmed, and actual arrival times.
Escalate when a deadline is crossed, not when a dashboard changes color without an agreed action.

After the event, compare actual to planned lead time, quantity, and cost.
Separate forecast error, supplier capacity, execution variability, master-data defects, and change-control failures.
Update parameters only with an explanation of how the new evidence generalizes.
A successful expedite may have prevented a stop; it is not evidence that the original system was resilient.

### Operating design: from analytics to a buyer's decision

An automotive analytics use case is useful only if its output changes a decision before the decision deadline.
The same dataset may support several decisions, but the appropriate unit of analysis differs.

| Decision | Buyer or accountable operator | Decision cadence | Useful output |
|---|---|---|---|
| Premium freight | Plant materials manager | Intraday | Earliest shortage and latest dispatch time |
| Supplier overtime | Supplier operations and purchasing | Daily or weekly | Required incremental units by shift |
| Buffer sizing | Plant logistics and finance | Weekly or monthly | Cost and protection by variant and horizon |
| Alternate source nomination | Purchasing and engineering | Program or annual | Approved capacity, dependence, and qualification cost |
| Changeover release | Engineering change board | Milestone-driven | Readiness and affected build boundaries |
| Suspect-lot containment | Quality | Immediately upon credible signal | Locations, quantities, and possibly affected VINs |
| Campaign remedy allocation | Service operations and regulatory team | Periodic during an action | Approved remedy supply and prioritized demand |
| Tooling investment | Program management and finance | Program milestone | Capacity, expected loss reduction, and cash requirement |

Each decision has a different counterfactual.
For freight, compare arrival with and without the expedite.
For a source nomination, compare several years of feasible production and disruption scenarios.
For a containment decision, the first question is what must be held and investigated, not what expected cost makes a safety constraint disappear.

An output should name the next action, not just the risk score.
For example, a risk score of 85 out of 100 does not say whether to release a truck, increase a shift, or stop a build.
An actionable output says that the first missing black seat is scheduled at 10:42 local time and that a verified truck must leave the supplier by 09:35 to arrive in time.
The times here are illustrative and require validated travel and handling assumptions.

### A practical sequence-event contract

For each vehicle sequence position, retain the following separate fields.
They are a suggested internal contract, not a claim that all trading partners use one common schema.

1. The plant identifier and the assembly line identifier.
2. The sequence version, issue time, and source system.
3. A stable vehicle order identifier and its VIN when assigned.
4. The sequence position and planned point-of-use time.
5. The configuration and required part number and revision.
6. The required variant, including side or color where relevant.
7. The supplier acknowledgement time and acknowledgement status.
8. The picking or build completion time and inspection status.
9. The rack identifier, rack position, and dispatch time.
10. The shipment and carrier identifiers.
11. The actual gate arrival and line-side availability times.
12. The installation event, including the actual component identifier.
13. The exception reason and who authorized any correction.

When a new sequence version supersedes an old one, store both versions and their effective times.
Do not overwrite the historical order and then use the new order to explain an earlier rack.
If a message arrives out of order, compare its version and effective time with the currently acknowledged version.
The exception process must say which version is authoritative until reconciliation is complete.

There are at least three distinct latency measures:

- **Communication latency:** sequence issued to supplier acknowledgement.
- **Execution latency:** acknowledged instruction to verified dispatch.
- **Delivery latency:** verified dispatch to usable line-side arrival.

Track these separately because their corrective actions differ.
A network outage, an overloaded picking station, and a closed road can produce the same late delivery but require different owners.
For each measure, show distribution and sample size rather than only the mean.
Also identify whether the observations represent ordinary operation, weather disruption, or a holiday schedule.

### Quantifying variant exposure

Suppose a synthetic rack contains 12 black sets and 8 gray sets.
The next 20 build positions require 13 black and 7 gray sets.
Total supply equals total demand at 20 sets, but the rack is one black set short.
The surplus gray set has no value for that immediate sequence unless an approved schedule change creates a compatible position.

For each variant, compare cumulative required units with cumulative usable units at every event time.
The first point where required units exceed usable units is the variant's first shortage.
The minimum of these shortage times across variants is the line-level risk time.
If arrivals are scheduled between consumption events, include their receipt and release times explicitly.
An estimated arrival cannot be substituted for a confirmed release without labeling the assumption.

An implementation can use a running balance:

```text
balance(variant, time) =
    opening usable units
    + cumulative released arrivals
    - cumulative scheduled consumption
```

A nonnegative end-of-day balance does not guarantee that the balance remained nonnegative during the day.
A late morning arrival can restore the evening total after a morning interruption has already occurred.
Maintain event order at equal timestamps, such as receipt before or after consumption, according to the actual operating rule.

### Contracting and allocation under shortage

The purchaser should distinguish four quantities that are often conflated.

1. **Forecast quantity:** a planning estimate with a specified horizon and update frequency.
2. **Released quantity:** a customer instruction under the agreed release process.
3. **Supplier commitment:** quantity and timing the supplier has accepted under the applicable terms.
4. **Physically available quantity:** approved, released material that can reach the plant in time.

If a supplier commits 1,000 parts but has only 700 approved parts available, the remaining 300 are not immediately usable.
The commercial commitment still matters, but it does not physically fill the gap.
The planner must locate the constraint, estimate the recovery time, and decide how to allocate the 700 parts.

Allocation may involve multiple assembly plants, service obligations, and vehicle configurations.
An illustrative policy ranks eligible builds by contractual priority, operational feasibility, and approved customer commitments.
It then checks whether preserving one line creates a larger shortage at another.
The policy should be approved before a crisis and reviewed for antitrust, contractual, and customer constraints where applicable.
It should not instruct a team to conceal a quality issue or favor a configuration that cannot legally be built.

For shared constraints, avoid multiplying the same physical capacity by the number of purchase agreements.
If two programs each show a commitment for 60 units per hour but depend on a common upstream tool with verified capacity of 100 units per hour, the combined demand of 120 units per hour exceeds that tool by 20 units per hour.
The arithmetic alone does not establish which contract has priority.
It does establish that the allocation requires an explicit operational and commercial decision.

### Capacity and recovery stress test

Consider a synthetic supplier with normal demonstrated capacity of 55 modules per hour.
Normal plant consumption is 50 modules per hour, leaving five modules per hour of recovery headroom.
Assume one hour of supplier production is lost while the plant continues consuming from inventory.
The resulting backlog is 50 modules if there was no preexisting backlog.
At five additional modules per hour, recovery takes 50 / 5 = 10 production hours after normal output resumes.

This is not the same as a one-hour disruption to the assembly plant.
The plant can continue only if usable stock covers the drawdown while the supplier catches up.
Transport, shift limits, rejects, and variant-specific capacity may lengthen the recovery.
If maximum approved output is only 50 modules per hour, there is no positive recovery headroom at the unchanged plant rate.
The backlog then persists until demand is reduced or additional approved capacity is introduced.

Ask the supplier to demonstrate the capacity calculation against:

- Actual good units, rather than theoretical machine cycles.
- Changeovers and maintenance included in the available time.
- Staffing that is actually available for the extra shift.
- Sub-tier parts and materials needed for the extra production.
- Tool and process approval at the proposed site.
- Packing, carrier, and receiving capacity after production.
- The required product mix, not only an aggregate daily total.

Use a different scenario for a permanent tool failure than for a one-hour road closure.
The recovery clock and feasible interventions are different.
A large local buffer may bridge a road closure but be insufficient for a months-long tool replacement.

### Recall investigation decision tree

Start with a credible suspect component, process, or failure signal.
Do not assume the signal proves a safety defect or determines the final action.

1. Identify the suspect part, revision, lot, and time window from source evidence.
2. Freeze relevant records and hold potentially affected stock according to the quality procedure.
3. Trace shipments to supplier and plant locations, including transfers and rework.
4. Trace actual component installations to VINs and vehicle positions where records support it.
5. Reconcile installation records with production totals and investigate missing or duplicate links.
6. Include service replacement paths where relevant, rather than limiting the search to initial assembly.
7. Classify records as linked, demonstrably excluded, or unresolved, with supporting evidence.
8. Escalate the technical assessment and any reporting or notification question to authorized functions.
9. If an action is required, reconcile remedy supply, dealer execution, and completion records.
10. Preserve the history of evidence and scope changes with timestamps and decision owners.

The distinction between physical suspect units and vehicle candidates is important.
One lot may have been split into several component lots and shipped to multiple plants.
Conversely, an exposure window may contain vehicles that never received a suspect unit.
The investigation needs enough detail to avoid both unsupported exclusion and unsupported certainty.

### A sensitivity check before approving spend

The Section 6 buffer comparison changes when more than one assumption changes.
For a weekly cost of USD 2,400, the break-even probability is the buffer cost divided by the avoided loss per event, provided the buffer fully avoids that event.
If avoided loss is USD 12,000 per event, the threshold is 20% per week.
If avoided loss is USD 24,000 per event, the threshold is 10% per week.
These are separate sensitivity scenarios, not additional events to add to the original expected cost.

If a buffer only prevents half the interruption in a scenario, use the avoided half rather than the full interruption cost.
For example, if a one-hour delay would cost USD 18,000 and the buffer reliably avoids only 30 minutes, the modeled avoided loss is USD 9,000.
At USD 2,400 per week, the corresponding break-even probability is 2,400 / 9,000 = 26.67% per week.
This calculation assumes interruption cost scales linearly with duration; actual recovery and minimum restart costs can make that assumption false.

Do not mix a probability per shipment with a cost per week without converting the event model.
If shipment delays are correlated, multiplying a per-shipment probability by the number of shipments can overstate or understate the probability of at least one weekly delay.
Use a joint distribution or explicit scenarios when common weather, labor, or supplier events affect multiple shipments.

## 6. Worked example

### Decision and synthetic facts

The following numbers describe a fictional plant and are not company evidence.
All money is in US dollars (USD), and the horizon is one production week.
The plant plans 400 vehicles in five shifts of 80 vehicles each.
Each vehicle needs one approved seat set.
The planned line rate is 40 vehicles per hour during production, so each shift contains two production hours in this deliberately simplified example.
Other downtime and labor scheduling are excluded from the arithmetic.

The current approved Tier 1 seat supplier receives subassemblies from a Tier 2 foam plant.
A normal truck brings 80 seat sets at the start of each shift, sequenced to the frozen build list.
The plant has no unrestricted safety stock at line side at the beginning of the week.
The supplier has confirmed the planned 400 sets but cannot make up a delayed truck during the same shift.
The accepted sequence includes 240 black and 160 gray seat sets over the week.
Black and gray sets are not interchangeable for a given vehicle order.

A local warehouse can hold 40 approved, already sequenced sets at the start of the week.
The warehouse and verification cost is USD 2,400 for that week, including all incremental handling and carrying costs.
Assume the 40 sets exactly match the affected next sequence and can be rotated without obsolescence within the week.
This is a strong assumption: a generic mixed buffer may not protect the needed sequence.

There are two mutually exclusive delivery scenarios:

| Scenario | Probability per week | Impact without buffer |
|---|---:|---|
| On-time deliveries | 0.80 | No shortage |
| One truck delayed by one production hour | 0.20 | One hour without 40 required seat sets |

The one-hour delay is recoverable only through a one-hour line interruption.
The incremental interruption cost is USD 18,000 per hour and already includes the net cost of recovery and lost contribution; do not add either again.
No safety or quality constraints are violated by either option.
Probabilities are assumptions, not measured supplier performance.

### Option A: no local buffer

The on-time scenario costs USD 0 incrementally.
The delayed scenario costs one hour × USD 18,000 per hour = USD 18,000.
Expected weekly incremental cost is:

```text
0.80 × USD 0 + 0.20 × USD 18,000 = USD 3,600/week
```

The largest modeled weekly loss is USD 18,000.
It is a scenario maximum, not a statement that larger disruptions are impossible.

### Option B: 40-set local buffer

Forty sets cover 40 vehicles/hour × one hour = 40 vehicles.
Under the stated exact-variant and release assumptions, they cover the delayed truck's production-hour consumption.
The fixed weekly cost is USD 2,400 in either scenario, and there is no modeled interruption cost.
Thus:

```text
0.80 × USD 2,400 + 0.20 × USD 2,400 = USD 2,400/week
```

Option B saves USD 1,200/week in expected incremental cost relative to Option A.
Its break-even disruption probability is USD 2,400 / USD 18,000 = 0.1333, or 13.33% per week.
Below that probability, with all other assumptions fixed, the no-buffer option has lower expected cost.
The threshold changes if a delayed truck causes a shorter or longer unrecovered stop, or if the buffer is not sequence-correct.

A finance owner should also consider the USD 18,000 modeled tail exposure under Option A, rather than presenting the USD 1,200 expected difference as a guarantee of realized savings.
If the 40 buffered sets are not approved for the next vehicles, the modeled avoided interruption is overstated.

### A sequence counterexample

Suppose the next ten vehicles require black seats in positions 1 through 7 and gray seats in positions 8 through 10.
The only accessible stock is six black and four gray sets.
Aggregate stock equals ten sets, but position 7 cannot be supplied.
Four gray sets do not substitute for the seventh black set.
A sequence-aware replay identifies position 7; an aggregate balance incorrectly reports full coverage.

### Model-year transition

The seat bracket changes from revision A to approved revision B at a specified build boundary.
Remaining authorized revision-A production is 126 vehicles at one bracket per vehicle.
Authorized service reserve is 18 brackets.
Usable old-revision inventory is 150 brackets and no further old receipts are committed.
Required old-revision stock is 126 + 18 = 144 brackets.
Potential excess is 150 - 144 = 6 brackets, subject to inventory accuracy and approved disposition.

The excess does not authorize using revision A on revision-B vehicles.
If the six brackets can be returned for USD 20 each rather than scrapped for zero, the recoverable value is USD 120 before transaction costs.
If the service reserve changes, recalculate rather than applying the original excess number.

New-revision demand is 274 vehicles.
Confirmed usable revision-B stock is 180 brackets and a confirmed receipt of 80 arrives before the first B build.
The remaining shortage is 274 - 180 - 80 = 14 brackets, assuming the 274-vehicle plan is unchanged and all stock is usable.
The 14 brackets require an approved purchase, schedule change, or other authorized action before their first required build, not necessarily before the first B build.

### Genealogy and exposure

A suspect foam lot was used in 100 seat assemblies.
Installation records link 92 assemblies to 92 distinct VINs, with no duplicates.
Eight assemblies lack installation records; all eight could have been installed among 20 additional VINs produced during the relevant window.
Assume those 20 VINs are distinct from the 92 recorded VINs and no other paths exist.

The evidence identifies 92 known linked VINs.
It does not identify which of the 20 additional VINs, if any, received the eight untraced assemblies.
At most eight of those 20 additional VINs can contain one of the eight assemblies under the one-assembly-per-vehicle assumption.
The candidate investigation population is 92 + 20 = 112 VINs, not 100 confidently identified VINs.
The actual scope of any field action requires technical and regulatory determination; it cannot be inferred solely from this arithmetic.

A complete installation record could reduce uncertainty, but only if its coverage and accuracy are verified.
For example, 100% scan compliance reported by a station is unconvincing if production reconciliation shows missing vehicles or manually reused serials.

### Decision brief

Choose the local buffer for this synthetic week on expected cost, provided sequence matching and release are demonstrated before the first shift.
The plant materials manager owns buffer availability; quality owns release; finance validates the interruption rate.
Engineering owns the A/B effectivity gate.
Quality and regulatory counsel own investigation and any field-action decisions.
Reassess if the weekly disruption probability falls below 13.33%, the buffer fails to match the sequence, or the assumed interruption cost changes.

## 7. Code, query, or calculation

The following Python 3 lab uses only the standard library.
It reproduces the expected-cost comparison, cutover arithmetic, genealogy bounds, and a sequence replay.
Run it as a script or paste it into a Python notebook cell.
All inputs are synthetic and quantities are deliberately small enough to audit by hand.
The replay consumes one seat per vehicle; there are no arrivals, substitutions, defects, or resequencing.

```python
from collections import Counter

# Costs are USD per week; probabilities refer to the same week.
delay_probability = 0.20
interruption_hours = 1.0
interruption_cost_per_hour = 18_000
buffer_cost_per_week = 2_400
buffer_sets = 40
line_rate_per_hour = 40

no_buffer = (
    delay_probability * interruption_hours * interruption_cost_per_hour
)
coverage_hours = buffer_sets / line_rate_per_hour
if coverage_hours < interruption_hours:
    raise ValueError("This simplified buffer model requires full coverage")
with_buffer = buffer_cost_per_week
break_even = buffer_cost_per_week / (
    interruption_hours * interruption_cost_per_hour
)

old_builds = 126
service_reserve = 18
old_usable_stock = 150
old_excess = old_usable_stock - old_builds - service_reserve

new_builds = 274
new_usable_stock = 180
new_confirmed_receipt = 80
new_shortage = max(0, new_builds - new_usable_stock - new_confirmed_receipt)

known_vins = {f"KNOWN-{i:03d}" for i in range(1, 93)}
possible_extra_vins = {f"WINDOW-{i:03d}" for i in range(1, 21)}
untraced_assemblies = 8
candidate_vins = known_vins | possible_extra_vins
max_additional_affected = min(untraced_assemblies, len(possible_extra_vins))

sequence = ["black"] * 7 + ["gray"] * 3
stock = Counter({"black": 6, "gray": 4})
first_unfillable = None
for position, variant in enumerate(sequence, start=1):
    if stock[variant] == 0:
        first_unfillable = position
        break
    stock[variant] -= 1

print(f"No buffer expected cost: USD {no_buffer:,.0f}/week")
print(f"Buffer expected cost: USD {with_buffer:,.0f}/week")
print(f"Buffer break-even probability: {break_even:.2%}/week")
print(f"Buffer coverage: {coverage_hours:.1f} production hour")
print(f"Old revision potential excess: {old_excess} brackets")
print(f"New revision shortage: {new_shortage} brackets")
print(f"Known linked VINs: {len(known_vins)}")
print(f"Candidate investigation VINs: {len(candidate_vins)}")
print(f"Maximum additional affected VINs: {max_additional_affected}")
print(f"First unfillable sequence position: {first_unfillable}")
```

Expected output:

```text
No buffer expected cost: USD 3,600/week
Buffer expected cost: USD 2,400/week
Buffer break-even probability: 13.33%/week
Buffer coverage: 1.0 production hour
Old revision potential excess: 6 brackets
New revision shortage: 14 brackets
Known linked VINs: 92
Candidate investigation VINs: 112
Maximum additional affected VINs: 8
First unfillable sequence position: 7
```

The code intentionally refuses to price a partially covering buffer as though it eliminated the full stop.
It is not an operational dispatch system: it does not model timed arrivals, multiple shifts, variable travel times, sequence changes, or the legal definition of a recall population.
The known and candidate VIN sets illustrate set operations, not actual vehicle identifiers.
For an operational implementation, obtain real authorized data, model each event timestamp and variant, reconcile inputs, and document the approval process.

### Lab extension

1. Change the delay probability to 0.10 per week; recalculate both expected costs.
2. Change the sequence to alternate black and gray while keeping the same stock; identify the first unfillable position.
3. Add a quality hold on 20 buffer sets and explain why the simplified full-coverage model must stop rather than claim full protection.
4. Replace the known and candidate sets with authorized, de-identified records; check intersection and uniqueness before counting.
5. Produce a one-page brief stating the chosen action, its owner, the break-even threshold, and three assumptions that could reverse the decision.

Do not upload VINs or supplier-confidential data to an unapproved environment.

## 8. Data, standards, and regulatory context

### Minimum data contract

A planning dataset needs a grain and a timestamp for every record.
The following is an illustrative contract, not a prescribed standard:

| Dataset | Grain and key | Critical fields | Control |
|---|---|---|---|
| Build schedule | One scheduled vehicle position and version | Plant, planned timestamp, configuration, VIN/order mapping, freeze state | Retain sequence versions and change timestamps |
| BOM/effectivity | One component rule and revision | Vehicle configuration, part, quantity, effective boundary | Engineering approval and overlapping-rule checks |
| Supplier capacity | Site, part, period | Approved rate, shifts, planned downtime, allocation | Distinguish quoted, tested, and committed capacity |
| Inventory | Site, part, revision, lot, status | Quantity, unit, quality status, ownership, location | Reconcile to physical counts and holds |
| Inbound shipment | Shipment, handling unit, part/lot | ASN time, actual departure, estimated and actual arrival | Match receipt and quantity; preserve exceptions |
| Sequence message | Plant, schedule version, position | Variant, message time, acknowledgement, supplier | Detect missing messages, duplicates, and out-of-order updates |
| Installation genealogy | VIN, component position, event | Actual part, revision, lot or serial, station, timestamp | Reconcile vehicle builds and investigate gaps |
| Change control | Change and plant applicability | Approval, effectivity, work instruction, supplier readiness | Prevent unapproved use and preserve audit trail |
| Quality event | Event, part, lot, disposition | Symptom, containment, investigation, status, decision owner | Restrict access and retain provenance |
| Field record | VIN and repair or complaint | Date, symptom, remedy, dealer record | Apply privacy, retention, and authorized access rules |

Normalize timestamps to a common time basis and retain the source time zone.
Record late-arriving or corrected events with both event time and ingestion time.
Version the BOM and the production schedule; a correct current master does not reconstruct what was authorized when a historical vehicle was built.

A planned arrival is not a receipt, a receipt is not a quality release, and a release is not proof of installation.
Each state needs its own event and authoritative owner.
Measure missingness, duplicates, referential integrity, timeliness, and reconciliation against independent production totals.

### Industry standards and commercial rules

IATF 16949 is an automotive quality management system standard published by the International Automotive Task Force (IATF).
The official IATF publication and customer-specific requirements define the applicable certification and contractual framework.
It is not, by itself, a universal statute mandating identical practices for every automotive supplier.
Certification does not demonstrate that every lot is defect-free or that a particular change is approved.

AIAG and VDA publish automotive quality planning and production process guidance, including Advanced Product Quality Planning (APQP) and Production Part Approval Process (PPAP) materials.
An OEM contract or customer-specific requirement determines which edition, forms, and approval steps apply to a relationship.
A PPAP package is evidence in an approval process, not a blanket authorization for any later site, process, or material change.

The Odette organization publishes automotive supply-chain and logistics recommendations, including electronic message and label guidance.
EDI standards and trading-partner implementation guides define message structure and interpretation; the contract defines which messages and service levels apply.
A successful interchange acknowledgement validates communication at a specified layer, not physical delivery or sequence correctness.
Map each sender's identifiers and codes to a governed internal model and retain the original message.

The ISO 3779 VIN standard describes the content and structure of the VIN.
Jurisdictional law can impose separate identification and record requirements.
A VIN identifies a vehicle; component genealogy must be captured and maintained separately.

### United States recall context

As of the cited U.S. National Highway Traffic Safety Administration (NHTSA) guidance and regulations, manufacturers have duties associated with safety defects and noncompliance, including reporting and recall processes under applicable federal law.
The regulatory text at 49 CFR Part 573 addresses defect and noncompliance responsibility and reporting; Part 577 addresses owner notification requirements.
The precise deadlines, scope, and obligations depend on the facts and the responsible entity.
Consult current regulations and counsel rather than treating this chapter as an operational compliance checklist.

NHTSA's recall database and public reports allow investigation of documented campaigns and their population and remedy status.
A campaign population is not necessarily the number of defective components.
Use the official campaign record, its publication date, and any amendments; avoid generalizing a specific recall's facts to all vehicles from the same manufacturer or supplier.

### European Union context

Regulation (EU) 2018/858 establishes an EU framework for approval and market surveillance of motor vehicles and their systems, components, and separate technical units.
It includes obligations relevant to economic operators and corrective action; applicability depends on the product, actor, and specific provision.
Use the consolidated legal text and qualified advice for a current determination.
The European Commission Safety Gate provides public information about relevant dangerous non-food products and alerts, but it does not substitute for the governing law or national authority instructions.

### Privacy and data sharing

VIN-linked service and ownership data may be personal data when it identifies or can be linked to a person.
Use purpose limitation, access controls, retention rules, and appropriate contractual and jurisdiction-specific safeguards.
A pseudonymized VIN can still be personal data when reidentification is possible.
For supplier collaboration, share only the data necessary to isolate and remedy the issue, with a documented lawful basis and access owner.

## 9. Industry practice and case evidence

### Case 1: Toyota and the 2011 Japan earthquake

**Documented fact.** Toyota's public reporting for the fiscal year ended March 2012 discusses production disruption following the March 2011 Great East Japan Earthquake and subsequent recovery.
The disaster affected production and supply networks across the industry; the cited report is the source for Toyota's own reported context.
It does not establish the performance of the fictional seat-buffer policy in Section 6.

**Analytical inference.** Mapping only direct suppliers is insufficient when a disaster can interrupt multiple upstream operations and infrastructure at once.
A practical continuity plan should include shared-site exposure, recovery time, substitute qualification, and the timing of restart.
These are general decision implications, not a claim that one specific sub-tier caused all of Toyota's disruption.

Questions for the buyer:

- Which approved parts share a physical sub-tier site?
- How long can available stock protect each plant at the actual build mix?
- Can an alternate tool and site produce approved parts before coverage expires?
- How will the OEM and supplier validate supplier-reported recovery dates?

### Case 2: Takata air bag inflators

**Documented fact.** NHTSA's Takata recall information describes a large, multi-manufacturer recall involving certain air bag inflators and the safety risk of inflator rupture.
NHTSA publishes current recall information and updates; vehicle-specific eligibility should be checked through authoritative recall lookup tools.
The campaign demonstrates the importance of identifying affected products and coordinating remedies over an extended period.

**Analytical inference.** A shared component supplier can create correlated exposure across nominally separate vehicle programs and manufacturers.
Better component-to-vehicle records can improve investigation, but the legally appropriate population still depends on evidence, regulator decisions, and applicable requirements.
Do not infer a precise counterfactual recall size from this general observation.

Questions for quality and procurement:

- Can a suspect component lot be traced to vehicle builds and service replacements?
- Where are records incomplete, and how broad is the resulting candidate population?
- Are replacement parts available and approved at the required rate?
- How will field repairs be reconciled back to the VIN record?

### Case 3: semiconductor allocation as an automotive planning problem

**Documented fact.** The U.S. Department of Commerce's 2022 report on semiconductor supply-chain responses describes tight inventories, capacity constraints, and disruptions affecting downstream sectors including automotive.
Its findings concern the surveyed supply chain and period; they are not current allocation figures for any named vehicle program.

**Analytical inference.** A Tier 1 electronic module supplier can be contractually available while an upstream semiconductor is constrained.
A vehicle maker should map approved devices, fabrication and assembly sites, firm commitments, and qualification lead times before declaring its supply dual-sourced.
Substituting a functionally similar chip may require engineering and customer approval; its availability on a broker website is not enough.

### Case 4: fictional model-year change

A synthetic OEM moves a camera module to a new connector at a plant-specific build boundary.
The old camera has ten weeks of supplier lead time, and a distributor offers additional old stock at a discount.
The buyer should not purchase solely on price: first compare firm remaining old builds and service requirements with usable old stock and open receipts.
Then check whether the supplier's connector, test equipment, packaging, and installation instructions are approved for the new module.
A low old-part price can become expensive if the material is unusable after the boundary.

This is a teaching case, not a public-company claim.
A reasonable decision might be to split a firm order from an option quantity, if the supplier accepts that commercial structure and engineering confirms the relevant effectivity.
The optimal agreement depends on cancellation terms, forecast uncertainty, lead time, and service obligations.

### Case 5: fictional JIS transport disruption

A synthetic seat supplier has enough weekly output, but its only delivery lane passes through a tunnel that occasionally closes.
The line's local sequence-correct stock covers 25 minutes, while an alternate route takes 50 minutes after a closure is detected.
Without another feasible action, the alternate route cannot prevent a shortage that occurs before it arrives.

The team can consider an earlier departure, a local sequenced supermarket, preapproved resequencing, a second lane, or a scheduled pause.
Each option must be checked for variant match and capacity, not merely ranked by transport price.
A warehouse that contains the wrong seats changes total inventory but may not change the first unfillable position.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Aggregate stock masks variant shortage | One color or side is repeatedly short | Incompatible parts are pooled | Wrong build or stop despite positive total stock | Replay the frozen sequence by variant |
| Late JIS message | Acknowledgement lag rises | Supplier has less time to pick and ship | Correct material arrives after use time | Monitor message-to-acknowledgement and escalate |
| Undetected resequencing | Supplier and plant sequence versions differ | Old instruction drives rack order | Line-side order mismatch | Version messages and acknowledge change |
| Hidden shared sub-tier | Two suppliers identify the same plant or tool | One upstream event disables both paths | Common-mode shortage | Map physical sites and qualify independence |
| Unrealistic alternate | Backup site lacks approved tool | Contractual alternate cannot make approved output | Recovery takes longer than coverage | Test and approve alternate before disruption |
| Expedite without material | Truck is booked before supplier release | Transport cannot move unbuilt or held parts | Premium cost without recovery | Confirm quantity, release, and pickup time |
| Average lead-time planning | Late-tail frequency grows | Mean conceals correlated delay | Repeated emergency stops | Use distributions and stress scenarios |
| Overstated line-stop cost | Finance sees duplicate contribution charges | Interruption rate and lost sales overlap | Overinvestment in mitigation | Reconcile cost components and recovery |
| Inventory hold omitted | System available quantity differs from quality status | Suspect material counted as usable | Unauthorized installation or surprise shortage | Join disposition and restrict use |
| Wrong effectivity | BOM has overlapping revision rules | Model-year label substitutes for build boundary | Wrong part installed or unnecessary scrap | Approve and test effective rules by VIN/configuration |
| Cutover excess misclassified | Old production ends but service demand remains | Production and service inventory combined incorrectly | Service shortage or premature scrap | Reserve approved service stock before disposition |
| Weak genealogy | Installation scans do not reconcile to builds | Dock receipt is mistaken for vehicle installation | Wider investigation candidate set | Capture actual installation and reconcile completeness |
| Duplicate serials | One serial maps to multiple incompatible events | Scanning or identifier reuse corrupts graph | False affected and unaffected results | Enforce uniqueness with controlled exception workflow |
| Blind trust in ASN | ASN timestamp exists but receiving is incomplete | Electronic announcement treated as usable stock | Coverage overstated | Match shipment, receipt, inspection, and line-side movement |
| Single metric scorecard | Monthly on-time average looks healthy despite stoppages | Aggregate percentage hides high-impact misses | Risk remains invisible | Add sequence accuracy and time-to-shortage outcomes |
| Privacy leakage | Raw VIN-owner files circulate broadly | More personal data shared than required | Legal and reputational harm | Limit fields, access, purpose, and retention |

An early warning is useful only when it has a defined threshold, owner, and response deadline.
For example, an acknowledgement delay is actionable when the remaining margin is insufficient for normal pick, transit, and line-side verification.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| One seat set per vehicle | Converts vehicles to sets | Special build consumes a different quantity | Validate configuration-specific BOM |
| Fixed 40 vehicles/hour | Converts stock to coverage | Line accelerates, stops, or varies mix | Compare time-stamped actual consumption |
| Buffer matches next sequence | Makes all 40 sets usable | Sequence changes or variant mix differs | Replay actual positions and scan rack order |
| One-hour disruption | Defines modeled interruption | Delay lasts longer or recurs | Examine historical tails and stress scenarios |
| Delay probability is 0.20/week | Weights scenarios | Supplier conditions or seasonality change | Backtest against relevant weeks and confidence range |
| USD 18,000/hour is incremental | Values a stop once | Rate embeds duplicated or excluded costs | Finance reconciles contribution, recovery, and labor |
| Warehouse cost is USD 2,400/week | Makes options comparable | Costs vary with volume or ownership | Obtain complete incremental quote and terms |
| Stock is quality-released | Makes quantity available | Hold, damage, or revision mismatch | Reconcile quality and physical records |
| Old and new requirements are fixed | Enables cutover arithmetic | Forecast, service need, or effectivity changes | Version build plan and engineering approval |
| Known and possible VIN sets do not overlap | Makes 92 + 20 valid | Duplicate or incorrectly joined identifiers | Check set intersection and VIN validity |
| One assembly per vehicle | Bounds additional affected vehicles by eight | Multiple installations or service replacements occur | Inspect product architecture and service history |
| Records cover all relevant paths | Makes the candidate window meaningful | Material transfers or unrecorded rework exist | Reconcile supplier outputs, shipments, and builds |
| Scenario probabilities are exclusive | Avoids double counting | Delays and quality holds overlap | Build a joint scenario tree |
| Commercial action is authorized | Makes cost comparison actionable | Contract, quality, or safety approval is missing | Check sign-offs before execution |

These assumptions are deliberately transparent.
A real decision should add supplier-specific evidence, uncertainty ranges, and the effect of correlated events.
Expected cost alone does not establish legal compliance, product safety, or a recall population.

## 12. Exercises

### Exercise 1 — Map the tiers

A vehicle maker buys a cockpit from Supplier A and an electronic controller from Supplier B.
Both buy a critical chip made at Factory C, through different distributors.
Draw the directed network and mark the common physical dependency.

**Deliverable:** A labeled network diagram and two questions for supplier validation.

**Acceptance criteria:** Correct tier relationships, identification of Factory C as a common node, and distinction between distributor and manufacturing site.

**Time:** 15 minutes.

### Exercise 2 — Calculate coverage

A line consumes 48 approved modules per production hour.
There are 72 usable modules at line side and no confirmed arrival before consumption.
A truck is expected to require two production hours to reach usable line-side status.

**Deliverable:** Coverage in production hours and the uncovered interval.

**Acceptance criteria:** Correct units and explicit statement that expected arrival is not a confirmed receipt.

**Time:** 10 minutes.

### Exercise 3 — Replay the sequence

The next six vehicles require variants A, B, A, A, B, B in that order.
Available stock is two A units and four B units; no arrivals occur.

**Deliverable:** The first unfillable position and the remaining stock immediately before that position.

**Acceptance criteria:** Preserve order, do not substitute variants, and show the intermediate consumption.

**Time:** 10 minutes.

### Exercise 4 — Choose a buffer

Over one week, a single disruption occurs with probability 0.15 and costs USD 20,000 if unprotected.
An approved buffer eliminates that modeled loss and costs USD 2,400 per week in every scenario.
All other incremental costs are zero.

**Deliverable:** Expected weekly cost of each option, preferred option, and break-even probability.

**Acceptance criteria:** Same horizon for all inputs, correct probability weighting, and one limitation of expected value.

**Time:** 15 minutes.

### Exercise 5 — Plan a cutover

An old revision is authorized for 90 remaining production vehicles and 15 service units.
There are 120 usable units in stock and 10 more confirmed old-revision units arriving.
The new revision needs 210 units; 160 are usable and 30 will arrive before the relevant builds.
One unit is required per vehicle or service unit.

**Deliverable:** Potential old excess and new shortage, with two checks before disposition.

**Acceptance criteria:** Include the service reserve and confirmed receipts; do not propose installation outside approved effectivity.

**Time:** 15 minutes.

### Exercise 6 — Bound an investigation population

A suspect lot supplied 50 assemblies.
Forty distinct VINs have verified installation links.
Ten assemblies have missing installation records, and the only possible additional installations are among 25 other distinct VINs.
Assume no service replacement, one assembly per vehicle, and no other paths.

**Deliverable:** Known linked count, candidate investigation count, and maximum additional affected count.

**Acceptance criteria:** Distinguish a candidate set from actual affected vehicles and explain why the legal recall population is not determined by the calculation.

**Time:** 15 minutes.

### Exercise 7 — Critique a scorecard

A supplier delivered 99 of 100 trucks on time in a month.
The one late truck contained the only approved component for a frozen sequence and caused a four-hour production interruption.

**Deliverable:** Three additional measures and an escalation rule.

**Acceptance criteria:** At least one measure captures impact or criticality and one measures sequence correctness; identify why 99% alone is insufficient.

**Time:** 20 minutes.

### Exercise 8 — Design a combined response

A Tier 1 supplier reports a suspect sub-tier lot during a model-year change.
Some old-revision stock is already in transit, the new revision is not yet approved, and tomorrow's build includes both configurations.

**Deliverable:** A one-page decision brief with immediate containment, sequence-aware coverage, cutover gate, genealogy query, decision owners, and stakeholder communications.

**Acceptance criteria:** Quarantine is separate from root-cause conclusion; approval is not inferred from availability; include decision deadlines, data uncertainty, and legal escalation.

**Time:** 35 minutes.

### Exercise 9 — Extend the lab

Adapt Section 7 to calculate expected cost for a 0.10 weekly delay probability, and trace a sequence of alternating black and gray seats with six black and four gray sets.

**Deliverable:** Updated output and a short explanation of which assumptions remain unchanged.

**Acceptance criteria:** No-buffer expected cost is recalculated, sequence stock is decremented in order, and the first shortage is identified without substitution.

**Time:** 25 minutes.

## 13. Self-check answers

### Answer 1

The OEM receives the cockpit from A and the controller from B.
Each direct supplier buys through its own distributor, and both supply paths converge at Factory C.
Factory C is the common physical dependency even though invoices and distributors differ.
Ask whether both devices are actually fabricated at the same site and whether any approved alternate site and capacity exist.
Verify the relationship for the relevant part revision and program rather than relying on company names alone.

### Answer 2

Coverage is 72 modules / 48 modules per production hour = 1.5 production hours.
If the expected usable arrival is two production hours away, the modeled gap is 2.0 - 1.5 = 0.5 production hour.
At 48 modules/hour, that interval corresponds to 24 modules of demand, before considering other actions or downtime.
An expected arrival is uncertain; the actual shortage may occur earlier or later.

### Answer 3

Position 1 consumes A, leaving one A and four B units.
Position 2 consumes B, leaving one A and three B units.
Position 3 consumes A, leaving zero A and three B units.
Position 4 requires A and is therefore the first unfillable position.
The remaining B units cannot be substituted without an authorized configuration change.

### Answer 4

Without the buffer, expected weekly cost is 0.15 × USD 20,000 = USD 3,000.
With the buffer, expected weekly cost is USD 2,400.
The buffer has USD 600 lower expected weekly cost under the stated assumptions.
Break-even probability is USD 2,400 / USD 20,000 = 0.12, or 12% per week.
This expected value does not show the full distribution of disruption severity or guarantee that the buffer works for every variant.

### Answer 5

Old-revision requirement is 90 + 15 = 105 units.
Available plus confirmed old units total 120 + 10 = 130 units, so potential excess is 25 units.
New-revision supply is 160 + 30 = 190 units against a requirement of 210, giving a 20-unit shortage.
Confirm timing and quality release of the receipts and verify actual effectivity and service reserve before disposing of old stock.
The old excess cannot automatically cover the new shortage.

### Answer 6

There are 40 known linked VINs.
The candidate investigation set contains 40 + 25 = 65 distinct VINs.
At most ten of the 25 additional VINs contain one of the untraced assemblies under the stated one-assembly-per-vehicle and no-other-path assumptions.
The specific additional VINs are unknown, and a legally appropriate recall scope requires the relevant technical and regulatory determination.

### Answer 7

A strong answer includes line-stop minutes or incremental cost attributable to late deliveries, sequence and variant accuracy, and time from risk detection to usable replenishment.
A further useful measure is on-time arrival weighted by critical part or by threatened sequence position.
Escalate when projected usable arrival is later than the earliest unfillable position minus the minimum response margin.
The 99% truck measure weights the critical shipment the same as every other truck and hides the four-hour impact.

### Answer 8 rubric

A strong brief immediately holds the suspect lot and identifies its location without claiming a confirmed defect cause.
It recomputes usable, variant-specific coverage for both old and new configurations and names the first decision deadline.
It assigns quality to disposition, engineering to effectivity and new-revision approval, materials to coverage, and legal/regulatory staff to applicable reporting and field-action decisions.
It queries both forward lot-to-VIN and backward VIN-to-lot links, marks unknown records, and plans communications with affected suppliers and the plant.
It states what happens if the new revision is not approved before the first required build.
It does not use an unapproved substitute merely to avoid a stop.

### Answer 9

At a 0.10 weekly delay probability, the no-buffer expected cost is 0.10 × USD 18,000 = USD 1,800/week.
The buffer remains USD 2,400/week under the unchanged assumptions, so no buffer is cheaper on expected cost by USD 600/week.
For ten alternating positions starting with black, the requirements are five black and five gray.
With six black and four gray sets, position 10 is the first unfillable position because it requires the fifth gray set.
The result changes if the sequence starts with gray or includes arrivals, so state the exact sequence.

## 14. Interview questions

1. **How do JIT and JIS differ?**
   A strong answer distinguishes timing from order and variant correctness, and explains why a punctual wrong sequence can interrupt the line.
   A weak answer treats the acronyms as interchangeable.

2. **How would you decide which parts deserve a buffer?**
   A strong answer weighs time to recover, usage variability, interruption impact, variant and obsolescence risk, and approved alternatives.
   A weak answer uses annual spend or inventory turns alone.

3. **What makes two suppliers genuinely independent?**
   A strong answer checks physical sites, sub-tier materials, tools, transport, and qualified capacity under a common event.
   A weak answer assumes two purchase orders eliminate common-mode risk.

4. **What is your first calculation when a JIS truck is late?**
   A strong answer identifies the earliest unfillable sequence position and compares its consumption time to confirmed usable arrival, including plant handling.
   A weak answer divides total seats by average demand without checking variants.

5. **How would you value a one-hour line stop?**
   A strong answer separates lost contribution, recovery, overtime, and other incremental costs, and checks for double counting and recoverable production.
   A weak answer applies a generic headline number without defining its contents.

6. **How do you manage an old-to-new part transition?**
   A strong answer starts with approved effectivity, remaining build and service demand, stock and pipeline, new-part readiness, and a cutover gate.
   A weak answer assumes a calendar model year automatically makes all old stock unusable.

7. **What does an ASN prove?**
   A strong answer says it is a shipment notice and distinguishes dispatch, physical receipt, release, and installation.
   A weak answer treats the message as proof that usable parts reached the line.

8. **How would you estimate possible recall exposure?**
   A strong answer traverses lot-to-component-to-VIN links, quantifies missingness, reconciles production, and separates candidates from legally determined scope.
   A weak answer counts shipments as affected vehicles or calls all unknown VINs confirmed defective.

9. **What would you do if the cheapest alternate part is unapproved?**
   A strong answer excludes it until engineering, quality, and any applicable customer approvals are obtained, and considers feasible interim actions.
   A weak answer installs it because the dimensions appear similar.

10. **What is the difference between a standard and a regulation?**
    A strong answer identifies the issuing body, legal jurisdiction, contractual incorporation, relevant edition, and responsible reviewer.
    A weak answer assumes certification guarantees legal compliance or product safety.

11. **How would you evaluate a supplier with 99% on-time delivery?**
    A strong answer examines criticality, sequence accuracy, stop impact, sample size, and variation by plant and part.
    A weak answer treats one aggregate percentage as complete evidence of resilience.

12. **How would you communicate an uncertain event to executives?**
    A strong answer states the decision deadline, known facts, assumptions, feasible actions, expected and tail costs, and named owners.
    A weak answer presents an unsupported point estimate as certain or omits the consequences of waiting.

## 15. Key terms

- **Advanced shipment notice:** Electronic shipment advice; see [data controls](#8-data-standards-and-regulatory-context).
- **BOM:** Configuration-specific component requirements; see [mechanics](#5-the-mechanics-step-by-step).
- **Break-even probability:** Probability at which two expected costs are equal; see [worked example](#6-worked-example).
- **Call-off:** Customer release against a supply arrangement; see [vocabulary](#4-definitions-and-vocabulary).
- **Candidate population:** Vehicles requiring investigation, not necessarily known defective; see [genealogy](#step-9--build-recall-ready-genealogy).
- **Capacity reservation:** Agreed supply entitlement, distinct from a forecast; see [network mapping](#step-2--map-the-product-and-network).
- **Changeover:** Controlled transition between product or process configurations; see [cutover](#step-8--plan-the-model-year-cutover).
- **Common-mode risk:** One physical event affecting multiple supply paths; see [case evidence](#9-industry-practice-and-case-evidence).
- **Containment:** Isolation pending investigation; see [failure modes](#10-failure-modes-and-red-flags).
- **Coverage:** Usable material divided by consumption rate under stated assumptions; see [mental model](#3-the-core-mental-model).
- **Effectivity:** Approved applicability boundary for a change; see [cutover](#step-8--plan-the-model-year-cutover).
- **Expected cost:** Scenario-weighted incremental cost over a defined horizon; see [economics](#step-7--compare-incremental-decision-economics).
- **Frozen sequence:** Controlled ordered build list; see [JIS response](#step-5--assess-jit-and-jis-response-windows).
- **Genealogy:** Links between input material, components, and vehicles; see [traceability](#step-9--build-recall-ready-genealogy).
- **JIS:** Delivery in vehicle build order and required variants; see [mental model](#3-the-core-mental-model).
- **JIT:** Supply near consumption to manage inventory; see [vocabulary](#4-definitions-and-vocabulary).
- **Line stop:** Interruption of planned production; see [economics](#step-7--compare-incremental-decision-economics).
- **Lot:** Defined batch of material; see [data controls](#8-data-standards-and-regulatory-context).
- **Model year:** Vehicle designation not necessarily equal to a calendar year; see [cutover](#step-8--plan-the-model-year-cutover).
- **OEM:** Original equipment manufacturer; see [network](#3-the-core-mental-model).
- **Obsolescence:** Loss of value when material no longer has its planned use; see [cutover](#step-8--plan-the-model-year-cutover).
- **PPAP:** Production Part Approval Process, subject to applicable customer requirements; see [standards](#8-data-standards-and-regulatory-context).
- **Recall:** Corrective action subject to applicable requirements; see [regulatory context](#8-data-standards-and-regulatory-context).
- **Resequencing:** Authorized change to build order; see [JIS response](#step-5--assess-jit-and-jis-response-windows).
- **Tier 1:** Direct supplier in a specified OEM network; see [network](#3-the-core-mental-model).
- **Usable stock:** Available inventory that is correctly located, approved, and not held; see [coverage](#step-4--establish-usable-material-and-time-to-shortage).
- **Variant:** Configuration-specific item that may not substitute for another; see [sequence counterexample](#a-sequence-counterexample).
- **VIN:** Vehicle identification number, not proof of installed components; see [data controls](#8-data-standards-and-regulatory-context).

## 16. Sources

Sources below support the specific facts and frameworks attributed in the text; they do not validate the synthetic numbers or the inferred recommendations.
Access current editions and applicable customer requirements before operational use.

1. [IATF, IATF 16949 overview and publications](https://www.iatfglobaloversight.org/iatf-169492016/): automotive quality management system standard and oversight information.
2. [AIAG, Quality resources](https://www.aiag.org/quality): automotive quality publications, including APQP and PPAP information.
3. [Odette, Standards](https://www.odette.org/standards): automotive supply-chain standards and implementation resources.
4. [ISO 3779, Road vehicles — Vehicle identification number (VIN) — Content and structure](https://www.iso.org/standard/52200.html): VIN structure standard; use the current edition where applicable.
5. [eCFR, 49 CFR Part 573](https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/part-573): U.S. defect and noncompliance responsibility and reports.
6. [eCFR, 49 CFR Part 577](https://www.ecfr.gov/current/title-49/subtitle-B/chapter-V/part-577): U.S. defect and noncompliance notification.
7. [NHTSA, Takata recall spotlight](https://www.nhtsa.gov/equipment/takata-recall-spotlight): documented inflator risks and recall information.
8. [NHTSA, Recalls](https://www.nhtsa.gov/recalls): vehicle recall lookup and public recall information.
9. [European Union, Regulation (EU) 2018/858](https://eur-lex.europa.eu/eli/reg/2018/858/oj): motor vehicle approval and market-surveillance framework.
10. [European Commission, Safety Gate](https://ec.europa.eu/safety-gate/): public product safety alerts.
11. [Toyota Motor Corporation, Annual Report 2012](https://global.toyota/en/ir/library/annual/): Toyota's annual-report archive; select the 2012 report for the earthquake and production context.
12. [U.S. Department of Commerce, Results from Semiconductor Supply Chain Request for Information](https://www.commerce.gov/news/blog/2022/01/results-semiconductor-supply-chain-request-information): January 2022 survey findings and semiconductor supply constraints.

The cited regulations are jurisdiction-specific and may be amended.
A standard, publication, or public case should not be used as a substitute for the current law, the relevant contract, and qualified engineering and legal review.

## 17. Links

- [Module 29 topic index](./_index.md)
- [Module 29 survey](../../../course/29_industry_electives.md)
- [Sibling topic: CPG and retail](./01_cpg-retail.md)
- [Sibling topic: pharma and cold chain](./02_pharma-cold-chain.md)
- [Module 30 simulation and optimization labs index](../30_simulation_optimization_production_labs/_index.md)

The sibling index lists planned topics; a linked topic may not yet be available in an incomplete course checkout.
Section 7 provides a self-contained computational lab for this chapter.
