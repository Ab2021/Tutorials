# Bottlenecks, Little's Law, Lean, JIT, Kanban, CONWIP, and TOC

> Module 15 — Manufacturing and Production Planning  
> Topic 10 of 11  
> Estimated study time: 4 hours  
> Prerequisites: Topics 05, 06, 08, and 09; basic arithmetic  
> Version 2026.5

## 1. Why this topic exists

A plant can keep every department busy and still miss customer dates.
A job may wait for a heat-treatment oven, then wait again for an inspector.
Meanwhile, upstream machines create more work that has nowhere to go.
The visible problem is late delivery; the physical cause may be an overloaded resource, unreliable flow, or the wrong release policy.

A production manager must choose which orders to release, where to protect capacity, and when to intervene.
A planner must distinguish an actual physical limit from a scheduling artifact.
A finance partner must decide whether an improvement deserves capital, overtime, inventory, or neither.
An operator must be able to stop unsafe work without being penalized for a utilization target.
These decisions cannot be made from machine utilization alone.

Work in process (WIP) is money and elapsed time held between release and completion.
It also absorbs disruption and keeps a constrained resource supplied.
Removing all WIP can lower output when arrivals or processing times vary.
Adding WIP without a release rule can lengthen queues without increasing sales.
The right amount is a controlled tradeoff, not a universal minimum.

Consider a plant with a weekly order commitment that exceeds the bottleneck's reliable good-unit capacity.
A new dispatching board cannot make that commitment feasible.
The business choices include changing the promise, adding qualified capacity, changing mix, reducing losses, or declining work.
The choice has consequences for customers, labor, safety, and working capital.

Lean identifies and removes waste while respecting people and maintaining quality.
Just-in-time (JIT) coordinates replenishment with consumption rather than producing early merely to keep machines busy.
Kanban uses explicit authorization signals and limits to regulate replenishment.
Constant work in process (CONWIP) caps the total released WIP within a defined loop.
The theory of constraints (TOC) focuses improvement and control on the system constraint.
They overlap, but none is a synonym for the others.

This topic gives you a way to measure flow, locate a constraint, design a pull loop, and evaluate the results.
It treats Little's law as an accounting relationship, not a promise that reducing inventory automatically improves throughput.
It also makes the safety and service implications of inventory reduction explicit.
The goal is a decision that works under variation, not an attractive dashboard.

## 2. Learning outcomes

After this topic you can:

1. **Identify** the likely capacity constraint for a product mix using effective good-unit load and explain why the answer can move.
   Exercise 1 checks the calculation and the evidence you would request.
2. **Compute** average flow time, throughput, or WIP using consistent boundaries and time units.
   Exercise 2 checks both arithmetic and interpretation.
3. **Distinguish** an active constraint from a large queue, a busy machine, and a policy restriction.
   Exercise 3 asks you to critique an apparent bottleneck.
4. **Size** a Kanban loop and interpret the rounding and variability allowance.
   Exercise 4 checks the card count and its assumptions.
5. **Design** a CONWIP release and return rule, including exceptions and traceability.
   Exercise 5 asks for a boundary, a control, and a safe escalation path.
6. **Apply** the TOC focusing steps to a resource and identify the commercial decision after capacity changes.
   Exercise 6 uses a mix decision and asks for its limits.
7. **Evaluate** a pull-system pilot using service, quality, safety, and flow measures.
   Exercise 7 asks for a comparison and decision rule.
8. **Explain** when JIT, local Kanban, CONWIP, and a protected constraint buffer are inappropriate or complementary.
   Exercise 8 asks you to defend a hybrid design.

Each result should state the time horizon, product or order population, and physical boundary.
An answer that omits units or assumes away downtime is not decision-ready.

## 3. The core mental model

```text
 Customer demand and due-date policy
                |
                v
      Approved backlog / promise
                |
     release permission and material
                |
                v
 [A: fabricate] -> [buffer] -> [B: constraint] -> [C: finish] -> ship
        ^                           |                         |
        |                           +-- lost B time is lost   |
        |                               system opportunity    |
        +------------- CONWIP card returned ------------------+

 A local Kanban can separately authorize replenishing a component
 into a point-of-use supermarket when a downstream user consumes it.
```

The customer-facing system converts material and labor into good completed units.
The constraint is the element that limits that conversion under the current mix and policy.
In the diagram, B is the active physical constraint for the chosen horizon.
A and C have protective capacity: some unused time lets them recover from variation and serve B.
Keeping A continuously busy can build an upstream queue without increasing B's output.

A finite buffer ahead of B protects it against disruption at A.
The buffer consumes floor space and cash, so it should have an explicit purpose and limit.
A downstream buffer or shipping stock may protect delivery from disruption after B.
The right location depends on where variation occurs and what customer service requires.
A buffer is not evidence of failure by itself.

CONWIP places a cap on the total cards in the loop from release through completion.
A returned card authorizes release of the next eligible job; it does not choose the job by itself.
The dispatch rule, promise dates, mix, and material availability still matter.
A card is permission to release, not a safety permit or permission to bypass quality control.

A local Kanban controls a particular replenishment loop, often for a part at a point of use.
Consumption frees a card or electronic signal; replenishment fills a defined container.
The local loop can sit inside a CONWIP-controlled line, but the same physical units must be counted consistently.
Neither label proves the system is pull-based if users can print unlimited cards to evade the limit.

Little's law relates three **long-run averages over the same boundary and unit population**:

```text
average WIP = average throughput rate x average flow time
       L    =            lambda      x          W
```

If average WIP is 360 good-equivalent units and average throughput is 120 units per day,
average flow time is 3 days, provided the defined population and observation period are compatible.
The identity does not identify a bottleneck or give a service-level percentile.
It does not say that cutting WIP in half while holding everything else fixed is physically achievable.

Variability is central to the tradeoff.
A resource planned at nearly 100% utilization has little slack to absorb late arrivals or long jobs.
Queues can grow sharply even when average demand is slightly below average capacity.
Reducing processing or arrival variability can improve both flow and service without simply shifting cost elsewhere.
A finite WIP cap can reveal problems earlier, but a cap that is too low can starve the constraint.

Think in four linked decisions:

- **Promise:** which demand is forecast, planned, targeted, or committed?
- **Capacity:** which resources can deliver the required good output at the required mix?
- **Release:** what authorizes entry to a bounded system?
- **Recovery:** who acts when buffers, quality, safety, or customer service are threatened?

## 4. Definitions and vocabulary

| Term | Meaning and unit | Common misuse |
|---|---|---|
| Boundary | Physical or logical entry and exit used for measurement | Combining line WIP with plant shipments |
| Population | Jobs or units counted consistently throughout a calculation | Counting only good output but all scrap WIP without adjustment |
| WIP | Released work not yet completed within the boundary, in units or jobs | Treating unreleased orders as shop-floor WIP |
| Throughput rate | Average completions per unit of time, for example good units/day | Calling a machine's theoretical speed throughput |
| Flow time | Elapsed time from boundary entry to exit, including waiting | Using touch time alone |
| Cycle time | Ambiguous term: often time between completions or processing time | Assuming everyone uses the same definition |
| Little's law | Average WIP equals average throughput rate times average flow time under its conditions | Applying it to mismatched windows or populations |
| Stable flow | A process whose long-run averages exist and are meaningful over the chosen horizon | Assuming a short unchanged snapshot proves stability |
| Bottleneck | Resource or restriction limiting system output under a particular mix and horizon | Calling the longest queue the constraint without evidence |
| Constraint | Anything limiting system performance, physical or policy-based | Assuming it must be a machine |
| Capacity | Maximum sustainable output under stated conditions, in good units/time | Using nameplate speed as reliable capacity |
| Load | Required resource time for a mix and time bucket | Comparing weekly load to daily capacity |
| Utilization | Busy time divided by available time, with a defined denominator | Treating 100% as universally desirable |
| Starvation | Resource is ready but lacks required work or material | Counting planned idle time as starvation |
| Blocking | Resource cannot transfer output because downstream space or capacity is unavailable | Calling all downtime blocking |
| Variation | Dispersion in arrivals, processing, quality, or availability | Confusing an average with a guarantee |
| Queue | Work waiting for a resource or decision | Assuming queue size alone identifies the active limit |
| Buffer | Deliberately held time, capacity, or inventory protecting against variation | Treating an uncontrolled backlog as a designed buffer |
| Protective capacity | Surplus nonconstraint capacity used to recover and support the constraint | Eliminating all spare capacity as waste |
| Takt time | Available production time divided by customer demand in the same period | Treating takt as actual machine cycle time |
| Lead time | Elapsed time from a specified request or trigger to specified receipt | Mixing supplier replenishment time with manufacturing flow time |
| Lean | Management approach to improve value flow and remove waste with respect for people | Equating Lean with indiscriminate headcount or inventory cuts |
| JIT | Producing or replenishing the needed item, quantity, and time in a coordinated system | Assuming zero inventory and zero safety margin |
| Pull | A downstream signal authorizes limited upstream replenishment or release | Calling any electronic schedule a pull signal |
| Push | Release based on an upstream plan without a downstream WIP authorization limit | Treating all planning forecasts as physical push |
| Kanban | Signal and authorization governing a defined replenishment loop | Equating a visual card with a complete control policy |
| Container quantity | Units per replenishment card or container | Allowing partially filled containers without defining accounting |
| Supermarket | Controlled point-of-use stock replenished from consumption | Treating an unlimited warehouse as a pull supermarket |
| CONWIP | A release control that limits total WIP across a specified loop | Assuming it also limits every individual queue |
| Card | Physical or electronic permission tied to controlled work | Printing replacement cards without reconciliation |
| FIFO | First in, first out; a sequence rule | Assuming it resolves all due-date or mix conflicts |
| TOC | Theory of constraints: manage and improve the limiting factor | Maximizing every resource's utilization |
| Drum-buffer-rope | TOC scheduling concept: constraint pace, protection, and synchronized release | Treating the rope as a substitute for capacity |
| Drum | Constraint or chosen pacing schedule | Assuming a theoretical machine speed is a feasible schedule |
| Rope | Release timing or signal tied to the drum | Releasing whenever upstream machines are idle |
| Buffer penetration | Consumption of planned protection measured in units or time | Escalating without a consistent buffer definition |
| First-pass yield | Good output without rework divided by input, under a stated definition | Counting reworked units as first-pass good |
| Rework | Additional processing to correct nonconformance | Removing work from WIP while it still occupies capacity |
| Scrap | Units that will not complete the normal flow | Continuing to count scrapped units as future good completions |
| Setup | Resource time needed to switch or prepare production | Ignoring mix-dependent setup time |
| Effective capacity | Capacity after stated losses, mix, and quality adjustments | Applying the same loss twice |
| Safety stock | Inventory held against uncertainty in demand or supply | Assuming a fixed percentage guarantees service |
| Service level | A defined delivery-performance measure, such as order fill rate | Comparing unlike service measures as if equivalent |
| Forecast | Estimate of future demand, not an order commitment | Treating a forecast as a contractual promise |
| Plan | Chosen intended production and release quantities | Calling an infeasible plan a commitment |
| Target | Desired performance used to guide improvement | Presenting a target as an observed result |
| Commitment | Agreed obligation to deliver a stated quantity and date | Assuming all backlog is equally promised |
| Workload | Remaining standard or expected processing time in a queue | Comparing job counts when job sizes differ greatly |
| Changeover | Transition between products, including sequence-dependent activities | Assuming all switches consume equal time |
| Andon | Visual or other signal requesting attention to an abnormal condition | Penalizing people for raising the signal |

Definitions must follow local data contracts.
For example, a job may contain several units, and a serial unit may visit a resource twice.
Do not sum job counts and unit counts as if they were interchangeable.

## 5. The mechanics, step by step

### 5.1 Start with the customer decision

1. Name the decision and owner.
   The owner might be the operations manager, scheduler, or order-promising team.
   Decide whether the question is next shift's release, next month's capacity, or next year's investment.
   Each horizon can have a different constraint.
2. Describe the deliverable.
   Example: a feasible weekly production commitment with an explicit backlog and escalation rule.
   A reduction in WIP is a means, not the customer outcome.
3. Separate four demand concepts.
   Forecast demand is uncertain future demand.
   The plan is the quantity the plant intends to make.
   A target is the desired metric or output.
   A commitment is the quantity and date agreed with a customer.
4. Define the service metric before looking at improvement.
   For example, on-time-in-full orders divided by eligible orders due in the week.
   Specify the due date source, exclusions, order amendments, and treatment of partial shipment.
5. State the economic and safety guardrails.
   Include labor hours, material exposure, quality holds, floor-space limits, and customer penalties where relevant.
   No rate target authorizes bypassing a required inspection or lockout procedure.

**Input:** demand and commitment records, business constraints, and decision horizon.
**Output:** a decision statement with a named owner and measurable outcome.
**Branch:** if customer orders are below sustainable capacity, the immediate constraint may be market demand rather than a machine.

### 5.2 Map the flow and measurement boundary

1. Draw the actual physical route for each major product family.
   Include receiving, fabrication, inspection, rework, finishing, and shipment where relevant.
2. Mark the release point and completion point.
   For example, release may be the scan of a kit into fabrication and completion the acceptance scan at final inspection.
3. Decide whether a unit in quality hold remains inside the boundary.
   Normally it does until accepted, scrapped, or explicitly transferred out.
   Rework loops need a documented entry and exit as well.
4. Identify outside dependencies.
   A component shortage can prevent release even if the line has capacity.
   A downstream shipping hold can make completed product accumulate beyond the chosen boundary.
5. Record units and calendars.
   A production day might be 450 available minutes, not 24 hours.
   Time measured as calendar days includes nights and weekends; time measured as operating days does not.
6. Capture which product families share each resource.
   Treat a shared oven serving another line as a shared constraint candidate, not as dedicated capacity.
7. Map data events to physical events.
   A backdated completion or delayed scan creates apparent negative or inflated flow times.

**Input:** routing, floor walk, transaction timestamps, and product-family list.
**Output:** a boundary diagram and event dictionary that all measures can use.
**Branch:** if traceability is weak, conduct a physical sample before relying on system timestamps.

### 5.3 Measure current flow

1. Take repeated WIP snapshots at consistent intervals, or compute a time-weighted average from entry and exit events.
   Include all locations inside the boundary, including inspection and rework.
2. Count completed units crossing the same boundary in the same period.
   If only good completions count, handle scrap explicitly rather than silently ignoring it.
3. Compute throughput per unit of elapsed time.
   Use the same day definition as the WIP and flow-time calculation.
4. Calculate Little's-law implied average flow time, WIP divided by throughput.
   Compare it with time-stamped cohort flow times only after resolving window and population differences.
5. Inspect the trend in WIP and backlog.
   If WIP rises persistently, arrivals exceed departures over that period and a stationary approximation may mislead.
6. Inspect distributions, not only means.
   A 3-day average does not say 95% of jobs finish within 3 days.
   Long tails matter to delivery promises.
7. Separate elapsed waiting from processing and transport where the data support it.
   An average can conceal a long quality hold for one family.
8. Reconcile entries, completions, scrap, transfers, and opening and closing WIP.
   The balance is opening WIP + entries - completions - scrap/transfers out = closing WIP.

**Input:** WIP snapshots or events, completion events, and exception dispositions.
**Output:** a consistent baseline and a documented data-quality assessment.
**Branch:** if there is a material upward or downward WIP trend, present the exact finite-period balance and avoid presenting WIP/throughput as an observed cohort average.

### 5.4 Test candidate constraints

1. Calculate required work by resource for the planned mix.
   Multiply planned quantities by expected processing time and add sequence-dependent setups and known rework load.
2. Calculate available resource time in the same horizon.
   Start from the actual staffed calendar, then subtract planned breaks and maintenance.
3. Apply reliable availability, performance, and quality assumptions consistently.
   If downtime is already embedded in observed effective throughput, do not subtract it again.
4. Compare load with capacity, including the effect of common resources shared by several lines.
   A utilization figure based on one product at ideal speed is not a mix-aware capacity calculation.
5. Observe the floor over multiple shifts and representative mix.
   Record starvation, blocking, setups, breakdowns, and queue age, not just machine-running indicators.
6. Test whether the suspected resource is continuously supplied and its output is used.
   A large queue in front of a machine can result from a release batch, a quality hold, or downstream blocking.
7. Consider a nonphysical constraint.
   A rule requiring a weekly release of every product, an approval queue, a supplier quota, or insufficient demand may limit performance.
8. Confirm the limit against actual completed output and any capacity changes already known.
   A one-shift outage can create a temporary bottleneck that differs from the structural constraint.

**Input:** mix, routings, calendars, losses, observations, and policies.
**Output:** ranked candidates, confidence, and evidence needed to distinguish them.
**Branch:** if the constraint moves with mix or time, re-evaluate by scenario; do not permanently label a machine as the bottleneck.

### 5.5 Apply the TOC focusing steps

1. **Identify** the constraint.
   State the horizon, the product mix, and the evidence for the limiting resource or policy.
2. **Exploit** it.
   Use its available time effectively without compromising safety, quality, or maintenance.
   Examples include staging approved material, arranging breaks, removing avoidable setup delays, and preventing known defects before the constraint.
3. **Subordinate** other decisions to it.
   Release and upstream schedules should support the constraint's actual plan.
   Upstream excess production is not a win merely because local utilization rises.
4. **Elevate** if the business case warrants it.
   Options include qualified overtime, alternate routing, a second fixture, supplier capacity, or capital investment.
   Include implementation delay, maintenance, training, and incremental cost.
5. **Repeat** after the constraint changes.
   More capacity at B can move the limit to A, C, a supplier, or demand.
   Do not let an old policy continue to constrain the system after its rationale disappears.

For a bottleneck with a full order book, an hour lost at the constraint can mean lost system throughput opportunity.
The actual financial loss depends on whether the hour is recovered later, whether demand exists, and the contribution from the mix.
An hour saved at a nonconstraint does not necessarily increase system output.
It may still be valuable for labor, maintenance, resilience, or flexibility.

**Input:** validated constraint and economic context.
**Output:** a prioritized action, a protected schedule, and a trigger for re-identification.
**Branch:** if a safety or quality requirement conflicts with throughput, the requirement takes precedence; recalculate the feasible promise.

### 5.6 Establish a drum, buffer, and rope

1. Set the **drum** to a feasible sequence at the constraint.
   Reflect product mix, effective rates, setup, maintenance, and promised dates.
2. Set a **buffer** to protect the drum from upstream variability.
   Express it in time or standard workload when products have different processing times.
   Specify the target, upper limit, trigger levels, and owner.
3. Define the **rope** as the rule for releasing work in time to reach the drum.
   It can use planned upstream lead time plus a protection allowance.
   Release cannot exceed material availability, WIP limits, or safety constraints.
4. Monitor buffer penetration.
   If the next scheduled job is missing or quality-blocked, escalate early enough to recover.
   A color on a board is useful only if someone owns the response.
5. Review what caused buffer consumption.
   A late component, a routing error, and a breakdown require different corrective action.
6. Protect downstream delivery as well.
   A constraint can be busy while finished goods miss customers because of downstream failures.
7. Update the rope when the route, variability, or constraint schedule changes.

**Input:** drum schedule, measured upstream variation, available space, and delivery priorities.
**Output:** a synchronized release plan and an escalation rule.
**Branch:** if the buffer regularly empties, investigate causes before simply adding cards or blaming operators.

### 5.7 Select a pull-control architecture

1. Identify the problem to control.
   Local replenishment of a common component is different from regulating total WIP across a mixed-model line.
2. Choose a local Kanban when a defined item or family has a repeatable replenishment loop.
   Define the consuming point, supplying process, container quantity, and replenishment lead time.
3. Choose CONWIP when a common route or boundary can be controlled by a shared release limit.
   A released job holds a card until it completes or is formally removed from the loop.
4. Consider drum-buffer-rope when a known constraint drives the sequence and release needs synchronization.
   CONWIP may cap total WIP while the drum sets the sequence, but the rules must not contradict each other.
5. Retain a push plan where appropriate for long procurement lead times, regulatory release, or seasonal capacity reservation.
   Forecast-based purchasing does not require uncontrolled shop-floor release.
6. Decide how priorities are set.
   Cards authorize work; they do not by themselves resolve due-date conflicts or incompatible product sequences.
7. Define who owns the cap and who can approve exceptions.
   The owner must reconcile card balances and retain a record of each override.

**Input:** routing repeatability, demand pattern, replenishment reliability, and operating boundaries.
**Output:** a selected architecture and a rationale, including what it does not control.
**Branch:** intermittent high-value custom jobs may need workload-based control rather than equal unit cards.

### 5.8 Size a local Kanban loop

A common starting rule is:

```text
cards = ceiling[demand rate x replenishment lead time x (1 + allowance) / container quantity]
```

Demand rate and replenishment lead time must use compatible units.
The allowance is an explicit heuristic for uncertainty or policy; it is not a statistically guaranteed service level.
Lead time runs from the defined signal to the defined replenishment availability, including waiting, production, transport, and receiving where applicable.

1. Choose one item and one loop boundary.
   Pooling unlike products can conceal a shortage in one item.
2. Estimate representative consumption, including relevant seasonality or known mix.
   An average may be insufficient for a highly intermittent or promotional item.
3. Measure the replenishment lead-time distribution.
   Use actual elapsed signal-to-availability time, not only processing time.
4. Choose container quantity from handling, ergonomics, quality, and packaging constraints.
   Do not change physical pack size without approval where handling or traceability is affected.
5. Choose a documented allowance and calculate the initial integer card count.
   Round upward because a fraction of a physical card is not an authorization.
6. Check whether the resulting total covers the intended cycle and protection stock.
   Check floor space, expiration, working capital, and replenishment capacity.
7. Simulate or pilot under realistic demand and supply variation.
   A formula alone cannot guarantee no stockouts.
8. Set review triggers for demand, lead-time, supplier, route, or pack-size change.
   Avoid changing the count daily in response to noise.

A card usually represents a container, but the exact meaning must be explicit.
When a container is partially consumed, the card's state and remaining quantity require a rule.
The nominal card quantity is a limit on authorized stock in the loop, not necessarily the observed average inventory.

**Input:** demand, lead time, allowance, and container quantity.
**Output:** initial card count, loop rules, and review conditions.
**Branch:** if replenishment capacity cannot cover average consumption, more cards only delay the shortage; fix capacity or change the promise.

### 5.9 Set the CONWIP cap and dispatch rule

1. Define entry and exit events.
   A card may attach to a job at release and return after final quality acceptance.
2. Choose the card's unit of control.
   One card per standard-sized job is simple; heterogeneous jobs may require workload classes or weighted authorization.
3. Estimate an initial cap from measured flow, practical protection, and simulation or pilot evidence.
   Little's law can explain the current average, but does not uniquely determine the best cap.
4. Separate the release queue from released WIP.
   The order backlog can be prioritized without consuming a shop-floor card.
5. When a card returns, release the next eligible job according to an explicit priority rule.
   Eligibility includes material, routing, quality status, and capacity constraints.
6. Specify what happens if the highest-priority job is not eligible.
   Skipping it may protect flow but requires a visible exception and a promised-date review.
7. Define card handling for scrap, cancellation, split lots, merges, rework, and outside processing.
   A card should not disappear merely because a unit is difficult to process.
8. Define a cap-change approval and expiry.
   An emergency override should be logged, bounded, and reviewed, not silently made permanent.
9. Audit the invariant.
   Active cards plus free cards should equal the authorized total, adjusted only by approved changes.
10. Track the age and workload of WIP, not only its count.
    A cap can be respected while one job is repeatedly bypassed.

**Input:** boundary, product/job definitions, baseline flow, and priority policy.
**Output:** card rules and initial cap with accountable ownership.
**Branch:** if jobs differ drastically in hours, a fixed job count may permit excessive workload; test a workload-based equivalent.

### 5.10 Integrate quality and safety

1. Place quality checks where defects can be prevented or detected before consuming scarce capacity.
   Do not remove required release checks to improve apparent throughput.
2. Retain traceability through card transfers, split lots, and rework.
   The card is a control token, not a substitute for serial or batch genealogy.
3. Define a hold location and status visible to planning.
   Count held work within the relevant boundary until disposition.
4. Allow any authorized operator to stop for an unsafe condition.
   Escalation must not depend on the state of a buffer.
5. Use safe material-handling limits when selecting container quantity and storage locations.
   A smaller number of large containers can increase ergonomic risk.
6. Include cleaning, sanitation, calibration, and preventive maintenance in the feasible calendar.
   A planned stop can protect long-run throughput and compliance.

**Input:** approved procedures, risk assessments, inspection plan, and traceability requirements.
**Output:** release rules and WIP limits that remain subordinate to those requirements.
**Branch:** if a change affects a regulated process, obtain required quality and engineering approval before the pilot.

### 5.11 Pilot, compare, and learn

1. Choose a bounded area with a stable route and accountable owner.
   Include upstream suppliers and downstream customers in the design.
2. Record a baseline across representative shifts and product mix.
   Measure throughput, WIP, flow time, service, quality, safety, and labor outcomes.
3. State the intended mechanism.
   Example: fewer premature releases reduce queue age while preserving a protective buffer ahead of B.
4. Set an initial cap and escalation thresholds.
   Specify minimum buffer, maximum age, and the action for a missed promise.
5. Train the affected team and rehearse exceptions before launch.
   A system that works only when the designer is present is not ready.
6. Compare like with like.
   Record demand, mix, staffing, downtime, and seasonality; use a comparable reference period or area when possible.
7. Examine both averages and tails.
   Improvement in average flow time with worse oldest-order age may harm customer service.
8. Review daily problems without punishing reporting.
   Distinguish a learning signal from an instruction to work around the limit.
9. Decide to maintain, adjust, roll back, or expand using pre-agreed criteria.
   Investigate adverse safety, quality, and service outcomes immediately.
10. Recalculate the promise when the sustainable rate changes.
    Do not sell assumed improvement before it is demonstrated and approved.

**Input:** baseline, pilot plan, training, and guardrails.
**Output:** evidence and a decision with named owners.
**Branch:** if the pilot reduces WIP but starves the constraint or worsens delivery, revise the system rather than claiming success from inventory alone.

### 5.12 Review economics and distributional effects

1. Estimate carrying cost from the change in average inventory value, not the full sales price.
2. Estimate incremental contribution from additional **sold** good output, not merely units produced.
3. Count overtime, changeovers, expedited freight, setup, and implementation costs.
4. Identify costs transferred to suppliers or customers.
   A local inventory reduction financed by forced supplier stock may not improve the network.
5. Distinguish a one-time cash release from a recurring annual carrying-cost saving.
6. Consider resilience against supplier outage, demand surges, and transport disruption.
7. Document worker impacts, including workload, walking distance, overtime, and ability to stop the line.
8. Present low, base, and high scenarios when key inputs are uncertain.

**Input:** observed performance, cost assumptions, and customer demand.
**Output:** an economic and operational decision, with uncertainty and stakeholder impacts.
**Branch:** if the benefit depends on additional sales that are not contracted, label them as an assumption rather than realized savings.

### 5.13 Stress-test demand and disruption

An average-demand card count does not describe the consequences of a sustained surge.
Before adopting the count, specify a small set of plausible operating scenarios with the planning team.
Each scenario should preserve the same loop boundary and identify the duration of the disturbance.

1. Test a demand increase.
   For example, a 20% increase for two weeks requires more replenishment capacity as well as more inventory.
   Calculate whether the supplier can replenish the extra consumption after existing backlog is cleared.
2. Test a supplier interruption.
   Specify the interruption length, beginning inventory, open replenishment orders, and recovery rate.
   An interruption followed by a slow recovery can cause a shortage even after deliveries resume.
3. Test a quality hold.
   Quarantine suspect lots, preserve genealogy, and assume held material is unavailable until disposition.
   Do not let a card signal override the quality decision.
4. Test a shared-resource outage.
   Include all product families competing for the resource and any qualification limits on alternate equipment.
   An available alternate machine is not capacity if qualified tooling or trained operators are unavailable.
5. Test a mix shift.
   Translate the changed mix into resource minutes, including sequence-dependent setup.
   A stable unit volume can still overload the constraint if the average workload per unit rises.
6. Record the response in advance.
   Identify the person authorized to change priorities, request overtime, notify customers, or temporarily adjust a cap.
   Give each exception an expiry and a review point.
7. Compare responses on service, cost, quality, and worker impact.
   A response that avoids one stockout by creating an unsafe handling condition is not acceptable.
8. Revisit the customer promise when available supply cannot cover commitments.
   Prioritize transparently using contractual and business rules rather than releasing every order simultaneously.

**Input:** demand scenarios, replenishment history, recovery capacity, quality procedures, and escalation authority.
**Output:** a documented set of stress outcomes and actions, not a claim that the selected scenarios cover every risk.
**Branch:** where a disruption would threaten safety or a critical customer, consider explicitly approved protection beyond an average-based heuristic.

## 6. Worked example

### 6.1 Setting and decision

A fictional plant makes a valve body through machining, heat treatment, and final assembly.
The planning horizon is one five-day production week.
The plant runs one 450-minute staffed shift per day after breaks.
Each unit passes once through each operation under the normal route.
The week has a forecast of 640 units, a production plan of 600 units, and a customer commitment of 580 good units.
The improvement target is 600 good units per week, not a promise to customers.
The customer commitment is spread across five daily delivery windows.
The plant must decide whether to release all planned work immediately and whether an overtime shift is needed.

The available production time is 2,250 minutes per week at each resource.
Planned maintenance is already excluded from that number.
The assumed effective capacity factors below cover unplanned stoppage and performance losses, but not the explicitly listed setups.
The table uses expected resource minutes per good finished unit, including the expected effect of scrap and rework.
Therefore no additional yield factor should be applied to the resulting capacity.

| Resource | Effective processing minutes/week after loss and setup | Expected minutes/good unit | Capacity, good units/week |
|---|---:|---:|---:|
| Machining A | 1,950 | 2.8 | 696.4 |
| Heat treatment B | 1,860 | 3.2 | 581.25 |
| Assembly C | 1,980 | 3.0 | 660.0 |

The fractional capacities are calculation results, not fractional valves shipped.
Heat treatment B is the lowest-capacity resource for this single-product mix.
At 600 units, its required time is 600 x 3.2 = 1,920 minutes.
The gap is 1,920 - 1,860 = 60 minutes per week.
At 580 units, required time is 580 x 3.2 = 1,856 minutes.
That leaves only 4 minutes of modeled protection at B.
The commitment is technically inside the average capacity estimate but has little margin for an unusually bad week.
It should not automatically be characterized as reliably achievable.

The machining requirement at 600 units is 1,680 minutes, leaving 270 minutes.
The assembly requirement is 1,800 minutes, leaving 180 minutes.
Those margins are protective capacity in this scenario, not proof that the resources should be staffed or operated identically in every week.
If the product mix changes, or assembly fails, the constraint may move.

### 6.2 Baseline flow

The measured boundary begins when a kit is released to machining and ends when a good valve passes final inspection.
All units in queue, in process, and on quality hold within that route count as WIP.
For the illustrative baseline, no units are scrapped or transferred out within the observation window.
Assume the four-week period is representative and sufficiently stable for the average relation.
Time-weighted average WIP is 348 units.
Average good throughput is 116 units per production day.
The implied average flow time is:

```text
348 units / (116 units/production day) = 3.0 production days
```

The four-week average is 580 units per week, consistent with 116 units per day over five production days.
This does not prove that each order takes exactly 3 days.
It also does not establish that a 3-day promise is safe for 95% of orders.
The planner still needs a flow-time distribution and current backlog age.

Suppose the opening WIP on a particular day is 340 units, releases are 120 units, and accepted completions are 116 units.
With no scrap or transfers, closing WIP is 340 + 120 - 116 = 344 units.
The 4-unit increase shows that releases exceeded completions on that day.
One day alone does not establish a persistent trend.

### 6.3 Improvement options

The team observes that B is occasionally starved for approved material because machining batches the wrong sequence.
It also observes that B spends an avoidable 48 minutes per week waiting for staged fixtures.
These are illustrative measurements, not guaranteed savings.
The team proposes pre-staging verified fixtures and aligning releases to the B sequence.
If the full 48 minutes becomes additional effective processing time, B has 1,908 minutes per week.
Its estimated capacity becomes 1,908 / 3.2 = 596.25 good units per week.
The 600-unit target still needs 12 minutes more than this estimate permits.
The 580-unit commitment now has 52 minutes of modeled margin, still subject to variability.

The planning options are:

| Option | Capacity implication | Decision concern |
|---|---|---|
| Keep current operating practice | About 581 good units/week at B | Commitment has negligible average margin |
| Recover 48 effective minutes | About 596 good units/week at B | Verify the recovery and its sustainability |
| Add 64 effective minutes after recovery | About 616 good units/week at B | Requires qualified coverage and approved cost |
| Release 640 units immediately | Does not change B capacity | Raises WIP and may hide priorities |

The third option adds 64 effective minutes after the fixture improvement, for 1,972 minutes at B.
The calculation is 1,972 / 3.2 = 616.25 good units per week.
The plant must verify that the additional time is genuinely available, that other resources remain adequate, and that demand justifies it.
If the extra minutes require paid overtime, labor rules, fatigue, and setup costs must be considered.
No unsupported claim of realized capacity is made until measured.

### 6.4 Pull-loop design

The team keeps the 580-unit customer commitment separate from the 600-unit internal target.
It does not release all 640 forecast units.
It uses B's approved schedule as the drum and starts with a 1.0-day workload buffer ahead of B.
At 116 good units per production day and 3.2 B-minutes per unit, that buffer is approximately 371.2 B-minutes.
This is an initial pilot protection level, not an optimized or statistically guaranteed buffer.
The release rope is tied to the scheduled B sequence and an estimated upstream lead time.
The team records the upstream lead-time distribution and reviews buffer penetration each shift.

A separate component used at assembly is controlled by local Kanban.
Its average consumption is 116 components per production day.
Measured replenishment lead time is 1.4 production days.
An initial 20% allowance is chosen as a policy heuristic.
Each approved container holds 25 components.

```text
Raw cards = 116 components/day x 1.4 days x 1.20 / 25 components/card
          = 7.7952 cards
Initial authorized cards = ceiling(7.7952) = 8 cards
Nominal authorized quantity = 8 x 25 = 200 components
```

The unrounded coverage requirement is 194.88 components.
The eight-card loop has 200 components of nominal authorization, including replenishment in process.
It does not guarantee uninterrupted supply and is not necessarily 200 components of average stock on hand.
If consumption or lead time is strongly variable, the team must use empirical scenarios or a service-level model rather than treating 20% as a guarantee.

The team also pilots a CONWIP cap of 300 unit-equivalent cards over the machining-to-final-inspection boundary.
A unit receives one card at release; the card returns after good acceptance or controlled scrap disposition.
The numerical cap is a trial decision, not an outcome or a direct consequence of Little's law.
If average WIP reaches 300 units while average throughput remains 116 good units/day under compatible stable conditions, implied average flow time is 2.586 production days.
That is a conditional projection, not proof the lower cap will preserve throughput.
The pilot stops or adjusts if B is starved, safety deteriorates, quality holds are hidden, or customer service worsens.

### 6.5 Illustrative economics and decision

Assume the trial achieves a sustained 48-unit reduction in average WIP, from 348 to 300 units.
At USD 80 of average inventory cost per unit, the one-time working-capital release is USD 3,840.
At an assumed 20% annual carrying rate, the recurring annual carrying-cost reduction is USD 768.
These are different economic quantities; they must not be added as if both recur annually.
The calculation ignores tax, financing details, implementation expense, and changes in other inventory.

If verified fixture staging increases saleable output by 10 good units per week, and all 10 units are actually sold at an incremental contribution of USD 30 per unit, the gross incremental contribution is USD 300 per week.
This is conditional on demand and contribution assumptions and excludes the cost of the staging change.
If the units only accumulate as unsold inventory, that sales contribution has not been realized.

The recommended pilot is to stage fixtures, protect B's schedule, and test the 300-card cap while retaining the 580-unit commitment.
The production manager owns the daily buffer review.
The planner owns the release sequence and reports any order at risk.
Quality owns disposition of held units and preserves their card status.
The business should not commit to 600 units per week until enough representative evidence supports the higher promise and any needed capacity is approved.

## 7. Code, query, or calculation

The following standard-library Python 3 program reproduces the example's main calculations.
It is deliberately a deterministic calculator, not a discrete-event simulation.
It does not estimate variability, optimize a cap, or prove customer service.
Run it as `python flow_example.py` after saving the block as `flow_example.py`.
The input values use production days, good units, minutes, and USD as shown.
The 20% Kanban allowance is a policy parameter, not a computed probability.
The 300-unit WIP cap is a proposed average for a conditional comparison, although a cap and an average are not generally equal.

```python
from math import ceil

# Weekly time already excludes planned maintenance, and each listed
# effective-minute value already excludes other stated losses and setups.
resources = {
    "machining": (1950.0, 2.8),
    "heat_treatment": (1860.0, 3.2),
    "assembly": (1980.0, 3.0),
}

forecast_units = 640
plan_units = 600
commitment_units = 580
production_days_per_week = 5

capacity = {
    name: minutes / minutes_per_good_unit
    for name, (minutes, minutes_per_good_unit) in resources.items()
}
constraint = min(capacity, key=capacity.get)

print(f"Forecast: {forecast_units} units/week")
print(f"Plan: {plan_units} units/week")
print(f"Commitment: {commitment_units} units/week")
for name, units_per_week in capacity.items():
    print(f"{name} capacity: {units_per_week:.2f} good units/week")
print(f"Constraint for this mix: {constraint}")

b_minutes, b_minutes_per_unit = resources["heat_treatment"]
plan_load = plan_units * b_minutes_per_unit
commitment_load = commitment_units * b_minutes_per_unit
print(f"Plan gap at B: {plan_load - b_minutes:.1f} minutes/week")
print(f"Commitment margin at B: {b_minutes - commitment_load:.1f} minutes/week")

recovered_minutes = 48.0
extra_minutes = 64.0
recovered_capacity = (b_minutes + recovered_minutes) / b_minutes_per_unit
extra_capacity = (b_minutes + recovered_minutes + extra_minutes) / b_minutes_per_unit
print(f"B after recovery: {recovered_capacity:.2f} good units/week")
print(f"B after recovery and extra time: {extra_capacity:.2f} good units/week")

average_wip = 348.0
good_units_per_day = 116.0
baseline_flow_days = average_wip / good_units_per_day
proposed_wip = 300.0
conditional_flow_days = proposed_wip / good_units_per_day
print(f"Baseline implied flow: {baseline_flow_days:.3f} production days")
print(f"Conditional flow at 300 average WIP: {conditional_flow_days:.3f} production days")

lead_time_days = 1.4
allowance = 0.20
container_units = 25
raw_cards = good_units_per_day * lead_time_days * (1 + allowance) / container_units
cards = ceil(raw_cards)
print(f"Raw Kanban cards: {raw_cards:.4f}")
print(f"Authorized Kanban cards: {cards}")
print(f"Nominal loop quantity: {cards * container_units} components")

unit_inventory_cost = 80.0
annual_carrying_rate = 0.20
cash_release = (average_wip - proposed_wip) * unit_inventory_cost
annual_carrying_saving = cash_release * annual_carrying_rate
print(f"Conditional working-capital release: USD {cash_release:,.2f}")
print(f"Conditional annual carrying saving: USD {annual_carrying_saving:,.2f}")

weekly_good_units = good_units_per_day * production_days_per_week
print(f"Baseline throughput: {weekly_good_units:.0f} good units/week")
```

Expected output:

```text
Forecast: 640 units/week
Plan: 600 units/week
Commitment: 580 units/week
machining capacity: 696.43 good units/week
heat_treatment capacity: 581.25 good units/week
assembly capacity: 660.00 good units/week
Constraint for this mix: heat_treatment
Plan gap at B: 60.0 minutes/week
Commitment margin at B: 4.0 minutes/week
B after recovery: 596.25 good units/week
B after recovery and extra time: 616.25 good units/week
Baseline implied flow: 3.000 production days
Conditional flow at 300 average WIP: 2.586 production days
Raw Kanban cards: 7.7952
Authorized Kanban cards: 8
Nominal loop quantity: 200 components
Conditional working-capital release: USD 3,840.00
Conditional annual carrying saving: USD 768.00
Baseline throughput: 580 good units/week
```

For real data, replace the static dictionary with a validated mix-aware resource-load calculation.
Keep the resource-time denominator and good-unit definition explicit.
The minimum capacity calculation is appropriate here because the example has one common product route and no other binding constraints.
It is not a complete finite-capacity scheduling model.
A mixture of products with different routings, sequence-dependent setups, or shared labor requires a more detailed model.
A simulation or pilot can estimate starvation and service under specified stochastic assumptions, but neither eliminates the need to validate inputs.

## 8. Data, standards, and regulatory context

### 8.1 Data needed for a defensible result

| Data element | Minimum content | Decision supported | Common defect |
|---|---|---|---|
| Demand | Item, quantity, requested and committed date, status | Promise and takt | Forecast mixed with firm order |
| Release event | Job or unit ID, time, route, quantity | CONWIP boundary entry | Backdated release |
| Completion event | Same ID, time, accepted quantity | Throughput and card return | Completion before inspection |
| Scrap disposition | Quantity, reason, time, card treatment | WIP reconciliation | Scrap counted as good output |
| Rework event | Original ID, route, work and status | Load and genealogy | Hidden second visit |
| WIP snapshot | Location, status, item, quantity, timestamp | Average WIP and buffer | Double count at transfer |
| Resource calendar | Shift, break, maintenance, staffing | Feasible capacity | Nameplate hours used as staffed hours |
| Processing standard | Item, operation, resource, time basis | Mix-aware load | Standard omits setup |
| Setup matrix | From-item, to-item, duration | Feasible sequence | Average setup hides difficult transitions |
| Downtime | Start, end, reason, planned/unplanned | Effective capacity | Overlap counted twice |
| Quality hold | ID, start, release/disposition, reason | WIP and risk | Held stock disappears from the system |
| Card ledger | Card ID, loop, state, associated work | Authorization audit | Duplicate or missing cards |
| Supplier lead time | Signal and available-at-use timestamps | Kanban sizing | Transit time mistaken for total replenishment time |
| Cost | Inventory value, carrying rate, incremental costs | Economics | Revenue used as incremental contribution |
| Safety record | Incidents, near misses, ergonomic and fatigue observations | Guardrail | Underreporting driven by target pressure |

A daily snapshot should have an agreed cutoff and location list.
A time-weighted average is preferable when WIP changes materially between snapshots.
A unit sampled in a long queue is more likely to appear in a snapshot than a quickly completed unit; do not treat the snapshot as an unbiased cohort sample.
Reconcile event totals with physical counts and investigate material discrepancies.

### 8.2 Time, identity, and grain

A transaction timestamp may record entry into a system rather than physical movement.
Compare it with observed scans and floor events before using it for lead-time promises.
Store time zones and daylight-saving handling for cross-site flows.
Document whether elapsed time includes nonworking days.
Do not divide a calendar-day throughput by an operating-day lead time.

The grain matters.
A job can contain multiple serial units, and a container can hold multiple lots.
Job-level CONWIP and unit-level WIP are not interchangeable without a conversion rule.
For mixed job sizes, use standard workload as well as counts.

Use a documented matching key across release, hold, rework, scrap, and completion.
A missing identifier can make work vanish from one part of a dashboard while remaining on the floor.
If units split or merge, record the relationship and preserve required genealogy.

### 8.3 Standards and requirements

ANSI/ISA-95 is a family of standards for enterprise-control system integration.
Its models and terminology can help define exchanges between business and manufacturing operations systems.
It does not prescribe a particular Kanban card count or establish that a plant is Lean.
Applicability depends on the organization's selected architecture and contractual obligations.

GS1 identification and barcode standards can support consistent identification and scanning of trade items, logistic units, and locations.
Their use is a convention or contractual requirement in many contexts, not a general legal mandate for every factory.
A barcode standard does not by itself ensure correct inventory events or quality genealogy.

ISO 9001 specifies requirements for a quality management system when an organization chooses or is required by contract to conform.
Certification is not universally mandatory for all manufacturing plants.
A pull-system redesign should be controlled under the organization's applicable change, documentation, and quality procedures.
ISO 9001 does not mandate a particular WIP limit or prove a customer promise is feasible.

In the United States, OSHA's lockout/tagout standard, 29 CFR 1910.147, applies to covered servicing and maintenance of machines and equipment involving hazardous energy.
Its scope and exceptions must be assessed for the actual operation.
A faster changeover or protected constraint schedule does not waive applicable hazardous-energy controls.
The standard is a legal requirement for covered employers and activities, not a general endorsement of a production method.

In U.S. food operations, applicable FDA current good manufacturing practice and preventive-controls requirements may include controls relevant to the facility and product under 21 CFR Part 117.
Applicability and exemptions depend on the facts.
Inventory reduction must not undermine required temperature control, sanitation, traceability, or other applicable controls.
Do not infer that every manufacturing facility is covered by the food rule.

These references are current as cited in the source section and must be checked against current law and local procedures before implementation.
This lesson is educational, not legal or regulatory advice.

### 8.4 Governance and data use

Assign ownership of route standards, calendars, card counts, and exception approvals.
Retain a change history so that a before/after comparison reflects changed definitions.
Restrict unnecessary personal identifiers in productivity data.
Consult local privacy, labor, and collective-agreement requirements before using individual performance records.
Aggregate operational metrics can still reveal a shift or team; use them for improvement rather than retaliation.

## 9. Industry practice and case evidence

### 9.1 Toyota Production System

Toyota describes the Toyota Production System in terms of just-in-time and jidoka.
Its public explanation describes producing what is needed, when needed, and in the amount needed, and stopping to address abnormalities.
Toyota also describes the use of Kanban to transmit production and conveyance instructions.
These are documented descriptions of Toyota's system, not proof that a copied card count will work in another factory.
See the official Toyota sources in section 16.

The practical lesson is that a signal is embedded in a broader operating system.
Equipment reliability, trained people, quality controls, material handling, supplier coordination, and problem escalation all affect whether a pull loop works.
**Inference:** a plant adopting only cards while retaining uncontrolled release and punitive stop-line incentives should not expect the same behavior as a complete production system.

Toyota's public material should not be read as saying every supply chain must have zero inventory.
Protection depends on demand, replenishment, disruption exposure, and the cost of failure.
A company can pursue JIT while maintaining deliberately placed buffers.

### 9.2 CONWIP research

Spearman, Woodruff, and Hopp's 1990 paper introduced CONWIP as a pull alternative for production systems.
The paper provides a research basis for controlling total WIP across a line rather than only individual local stages.
Its results depend on the modeled system and should not be converted into a universal numerical WIP limit.
A factory's mix, variability, release discipline, and routing determine the performance of a chosen cap.

**Inference:** in a mixed-model line, one total WIP limit may be easier to govern than a large number of product-specific loops.
That simplicity can be offset by weak control of individual queues or unfairness among job classes.
The appropriate comparison uses the same demand, capacity, priorities, and service measures.

### 9.3 Little's law research

Little's 1961 paper gives a formal foundation for the relationship between average number in a system, arrival rate, and time in the system under stated conditions.
The relationship is not specific to manufacturing.
It applies to a defined system and compatible averages; the plant must not combine unrelated populations.

**Inference:** the law is useful as a diagnostic reconciliation and a way to communicate the cost of excess WIP.
It does not determine how quickly a particular order will finish, or the best WIP cap under stochastic conditions.
If the plant increases output by releasing more material into an already saturated line, the law does not promise a shorter wait.

### 9.4 Theory of constraints in practice

Goldratt and Cox's *The Goal* popularized a narrative about managing constraints in manufacturing.
It is an influential management teaching text, not a controlled empirical trial of a specific plant intervention.
The focusing steps and drum-buffer-rope concepts are useful decision heuristics when their assumptions are made explicit.

A practical constraint review brings production, maintenance, quality, planning, and commercial staff together.
The review asks whether a planned hour at the constraint can actually create a saleable good unit.
If the market cannot absorb more output, the business constraint may be orders or pricing rather than a machine.
If a downstream process blocks output, a machine can appear busy without increasing customer deliveries.

### 9.5 What transfers and what does not

Transfer the questions, not an unexamined numeric setting.
A high-volume repetitive assembly line may have predictable routes and small containers.
A project shop may have long, variable jobs, engineering approvals, and expensive shared equipment.
A per-unit Kanban design copied between the two can misrepresent workload and priority.

A pharmaceutical or food line may have a release hold that dominates physical processing time.
A semiconductor line can have reentrant routes, shared tools, long recipes, and product-family differences.
An emergency spare-parts line may value availability during rare disruptions more than a low average inventory level.
These are scenario illustrations, not claims about a named firm's results.

A credible internal case records baseline, intervention, comparison period, mix, demand, staffing, and adverse outcomes.
It distinguishes a measured result from a target and from an inference about causality.
A short before/after comparison without controls can confuse a demand decline with a successful flow redesign.

## 10. Failure modes and red flags

| Failure | Early warning sign | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Releasing to keep A busy | Upstream queue grows while B output is flat | Local utilization drives excess release | Longer flow time and cash tied up | Tie release to constraint schedule and cap |
| Misidentified constraint | Large queue appears only after batch releases | Queue reflects timing, not sustainable capacity | Improvement misses real limit | Compare load, losses, starvation, and blocking |
| Moving constraint ignored | B becomes idle after improvement | A, C, supplier, or demand now limits output | Old policy restricts performance | Repeat constraint assessment after change |
| WIP cap too low | B starves after upstream disruption | Insufficient protective inventory | Lower throughput and missed deliveries | Review variation and increase protection if justified |
| WIP cap too high | Cards rarely bind and old jobs age | Authorization no longer limits releases | Long queues and poor visibility | Audit card use and reduce through controlled trials |
| Unlimited card printing | Card total rises during every shortage | Exceptions bypass authorization | Hidden push system | Controlled approval, expiry, and reconciliation |
| Kanban based on bad lead time | Frequent expedites despite average coverage | Waiting or transport omitted | Stockouts and emergency freight | Measure signal-to-availability distribution |
| Fixed allowance sold as guarantee | Stockout risk is not measured | Heuristic percentage ignores tail dependence | False service confidence | Pilot and model demand/lead-time variability |
| Scrap vanishes from accounting | WIP ledger does not reconcile | Units removed without disposition | Lost cards and misleading yield | Record scrap and return cards by rule |
| Rework hides outside boundary | Held jobs lack visible card status | Inconsistent entry and exit definitions | Understated WIP and load | Include rework and holds or record transfers explicitly |
| Setup omitted | Actual B time exceeds planned load | Product sequence consumes capacity | Infeasible commitment | Include setup matrix and calendar |
| Quality sacrificed | Defects or inspection overrides rise | Output target rewards bypasses | Customer escapes and rework | Quality gate, stop authority, and balanced measures |
| Safety compromised | Near misses or fatigue reports rise | Smaller buffers pressure unsafe recovery | Injury or regulatory breach | Stop-work authority and risk review |
| Supplier inventory displaced | Plant WIP falls while supplier stock rises | Risk and cash moved upstream | Network cost rises or supply weakens | Measure end-to-end inventory and contracts |
| Unfair dispatch | Old or difficult orders remain in queue | New easy jobs repeatedly chosen | Missed commitments and customer harm | Age limits and explicit escalation |
| Average masks instability | WIP trends up but average appears acceptable | Arrival exceeds completion | Growing backlog and unreliable lead time | Inspect time series and finite-period balance |
| Card count confused with on-hand | Loop seems overstocked during replenishment | In-transit and on-hand units mixed | Wrong replenishment decision | Show card state and physical location |
| Excessive overtime at B | Absence, fatigue, or maintenance backlog rises | Elevation ignores sustainability | Safety and reliability deteriorate | Limit overtime and evaluate alternatives |
| Using ideal rate twice | Capacity estimate exceeds observed good output | Losses or yields applied inconsistently | Overpromising | Reconcile standards and effective-rate definitions |
| Forecast treated as order | Inventory builds for unconfirmed demand | Planning signal mistaken for commitment | Obsolescence and cash exposure | Separate forecast, plan, target, and commitment |

A single red flag is a reason to investigate, not proof of root cause.
Use the physical flow, transaction record, and operator observations together.
Correct the control rule or source of variation rather than hiding the symptom.

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | How to test |
|---|---|---|---|
| Defined system boundary | WIP, rate, and time must refer to the same system | Completion or hold is treated differently across measures | Reconcile a sample of traced units |
| Compatible populations | Little's law uses one consistent flow population | WIP includes scrap while throughput counts only good units | Trace dispositions and model exits consistently |
| Meaningful long-run averages | Average relation needs stable, finite behavior | Persistent WIP accumulation or structural shift | Plot WIP and compare arrivals with departures |
| Same time basis | Rate times time must produce units | Mixing calendar and production days | Inspect calendars and convert all rates |
| Representative product mix | Resource load depends on mix | New routing or recipe changes processing time | Recalculate by scenario and compare actual mix |
| Reliable capacity estimate | Plan needs feasible resource time | Breakdown, quality loss, or setup not included | Compare planned and observed good output |
| No double-counted losses | Effective time and unit standard must align | Yield deducted from both time and unit rate | Trace each loss to exactly one calculation |
| Card discipline | WIP cap constrains actual release | Duplicate, missing, or bypassed cards | Reconcile free plus active cards to authorization |
| Fixed container quantity | Simple Kanban formula assumes a known quantity per card | Partial containers or pack-size changes | Audit physical containers and card states |
| Replenishment can meet demand | Cards regulate flow but cannot create capacity | Average consumption exceeds replenishment capacity | Compare sustained demand and supply rates |
| Allowance is appropriate | Initial card count needs a protection choice | Demand spikes or correlated supplier failures | Backtest and stress scenarios; monitor stockouts |
| Little's law is not causal | Identity alone cannot predict a policy intervention | Reduced cap starves the constraint | Pilot or simulate under stated variability |
| Jobs are comparable | A unit or job cap should reflect workload | Large jobs consume much more time than small ones | Compare processing-hour distribution by job |
| Output is saleable | Additional output creates value only with demand | Finished goods accumulate unsold | Match production with customer orders and inventory |
| Safety and quality remain intact | A feasible operation must meet mandatory controls | Pressure to bypass inspection or safe maintenance | Audit adherence and stop authority |
| No unmodeled shared constraint | Single-resource capacity comparison assumes independent availability | Shared operator or fixture limits multiple resources | Map common resources and observed contention |
| Pilot is comparable | Before/after attribution requires similar conditions | Mix, staffing, demand, or season changes | Stratify results and document confounders |
| Buffer location is appropriate | Protection must cover the actual disruption | Variation occurs downstream or at a supplier | Record cause and location of buffer depletion |

For a finite observation window, exact flow accounting remains useful even when steady-state assumptions fail.
A cohort average can differ from WIP divided by throughput because jobs cross the measurement boundaries at different times.
Inspect start and end inventories, censoring, and sample selection before interpreting the difference as a calculation error.

## 12. Exercises

### Exercise 1 — Identify a candidate constraint

A line has 1,800 effective processing minutes per week at A and 1,650 at B.
Each good unit requires 2.5 minutes at A and 3.0 minutes at B.
The current commitment is 530 good units per week.
Assume both operations are required once and the time standards already include expected quality losses.

- **Deliverable:** calculate each resource's good-unit capacity and identify the candidate constraint.
- **Acceptance criteria:** show units and arithmetic; state the margin against commitment; name two observations needed before treating the result as reliable.
- **Time:** 10 minutes.

### Exercise 2 — Apply Little's law with a boundary

A stable finished-unit line has time-weighted average WIP of 420 units and completes 140 good units per operating day.
Assume the same population, boundary, and calendar and no scrap in the period.
A manager says every unit therefore finishes within three days.

- **Deliverable:** compute the implied average flow time and critique the claim.
- **Acceptance criteria:** include units, distinguish average from a percentile, and name one reason a short unstable window may mislead.
- **Time:** 10 minutes.

### Exercise 3 — Diagnose an apparent bottleneck

A large queue appears before a mill every Monday after a weekly release of all orders.
The mill is starved on Thursdays, while the shared inspection station has a persistent queue and runs overtime.
The mill's reported utilization is 95%, but its denominator excludes planned waiting and setup.

- **Deliverable:** rank at least two constraint hypotheses and specify the next evidence to collect.
- **Acceptance criteria:** explain why the Monday queue and reported utilization are not conclusive; include a policy or shared-resource possibility.
- **Time:** 15 minutes.

### Exercise 4 — Size a Kanban loop

A component is consumed at 73 units per operating day.
Replenishment takes 1.6 operating days from signal to point-of-use availability.
The initial allowance is 15%, and the approved container holds 20 units.

- **Deliverable:** calculate unrounded and authorized cards and nominal authorized quantity.
- **Acceptance criteria:** round in the correct direction; explain what the allowance and nominal quantity do not guarantee.
- **Time:** 10 minutes.

### Exercise 5 — Design a CONWIP rule

A mixed-model cell has jobs ranging from 1 to 12 standard processing hours and a quality hold after its last operation.
The existing release rule lets supervisors start jobs whenever an upstream machine is free.

- **Deliverable:** propose the loop boundary, authorization unit, release priority, hold treatment, exception control, and audit invariant.
- **Acceptance criteria:** address unequal job sizes, missing material, scrap, rework, age, safety, and quality; identify an initial cap as a trial rather than an optimum.
- **Time:** 25 minutes.

### Exercise 6 — Manage constrained capacity

A bottleneck has 1,200 available minutes in a week.
Product X requires 4 minutes and earns USD 28 incremental contribution per sold good unit.
Product Y requires 6 minutes and earns USD 36 incremental contribution per sold good unit.
There is demand for at most 200 units of X and 200 units of Y.
Assume no other constraints, setups, minimum orders, or contractual priorities.

- **Deliverable:** choose an allocation maximizing contribution under these assumptions, and calculate total contribution.
- **Acceptance criteria:** show contribution per bottleneck minute, units of each product, consumed minutes, and at least two reasons the rule could fail in practice.
- **Time:** 20 minutes.

### Exercise 7 — Evaluate a pilot

Before a pilot, average WIP is 500 units, output is 100 good units per day, and 92% of eligible orders are on time in full.
During the pilot, average WIP is 360 units, output is 90 good units per day, and on-time-in-full performance is 84%.
The mix and staffing changed at the same time, and a safety near miss was reported.
Assume the same system boundary and comparable stable averages for the arithmetic only.

- **Deliverable:** calculate both implied flow times and recommend whether to expand the pilot.
- **Acceptance criteria:** include the confounding factors, service and safety guardrails, and a concrete investigation or rollback decision.
- **Time:** 20 minutes.

### Exercise 8 — Choose a hybrid design

A factory makes a stable high-volume family and rare engineered orders on a shared heat-treatment resource.
A common fastener is replenished from a supplier with variable delivery, while some finished goods require regulatory release.

- **Deliverable:** draw or describe a control architecture using a drum/buffer/rope, local Kanban, CONWIP, or an alternative as justified.
- **Acceptance criteria:** identify each boundary and signal, protect regulatory holds and traceability, define an exception owner, and explain at least two tradeoffs.
- **Time:** 25 minutes.

## 13. Self-check answers

### Answer 1

A has capacity 1,800 / 2.5 = 720 good units per week.
B has capacity 1,650 / 3.0 = 550 good units per week.
B is the candidate constraint in this simple common-route model.
The 530-unit commitment requires 1,590 B-minutes, leaving 60 B-minutes, or 20 units of nominal capacity.
Check the actual mix, setup and downtime inclusion, quality loss, blocking, and observed throughput before promising the output.
A 20-unit average margin does not guarantee service during a bad week.

### Answer 2

Flow time is 420 units / (140 units per operating day) = 3 operating days on average.
It is not a maximum, percentile, or promised completion time for every unit.
If WIP is rising because release exceeds output, the short-period averages may not represent stable long-run flow.
A mismatched completion boundary or a change in product mix can also distort the comparison.

### Answer 3

Inspection is a strong candidate because its queue persists and it requires overtime, but that evidence alone is not conclusive.
The mill could still be limiting for a particular mix or at a different horizon.
The Monday batch-release rule could create a temporary queue at the mill without making it the sustainable system constraint.
Measure workload, actual available hours, setups, starvation, blocking, and throughput at both resources over representative weeks.
Include the inspection station's other customers and validate the utilization denominator.

### Answer 4

Raw cards = 73 x 1.6 x 1.15 / 20 = 6.716 cards.
Round upward to 7 authorized cards.
Nominal loop quantity is 7 x 20 = 140 units.
The 15% allowance is a heuristic, not a stated service probability.
The 140 units include authorized replenishment in the loop and need not equal average on-hand inventory.

### Answer 5 — Rubric

A strong design defines card attachment at a clear release event and return at accepted completion or controlled disposition.
It keeps quality holds visible and does not return a card simply to conceal work.
It uses workload classes or another justified weighted control because jobs range from 1 to 12 hours.
It defines how missing material affects eligibility and how aged orders are escalated.
It records approved exceptions with owners and expiry, and reconciles authorized, active, and free capacity tokens.
It protects safety and regulatory/quality rules, and treats the initial cap as an experimentally evaluated choice.
A weak answer states only a card number or assumes equal jobs without addressing the consequences.

### Answer 6

X yields USD 28 / 4 = USD 7 per bottleneck minute.
Y yields USD 36 / 6 = USD 6 per bottleneck minute.
Make 200 units of X first, using 800 minutes.
The remaining 400 minutes permit at most 66 whole units of Y, using 396 minutes.
Total consumed time is 1,196 minutes and total contribution is 200 x USD 28 + 66 x USD 36 = USD 7,976.
The remaining 4 minutes cannot make another whole Y unit under the stated assumptions.
The calculation can fail if setups depend on sequence, other resources bind, demand differs, contractual obligations require a different mix, or contribution is not truly incremental.
A longer horizon may permit different integer allocation or inventory decisions.

### Answer 7

Before the pilot, implied flow time is 500 / 100 = 5 operating days.
During the pilot, implied flow time is 360 / 90 = 4 operating days.
The lower implied average accompanies lower throughput and lower on-time-in-full service.
Do not expand on the WIP result alone.
Investigate the near miss immediately and restore safe operating conditions.
Review mix, staffing, demand, starvation, and order-age distributions before attributing the outcome to the cap.
If guardrails are breached, pause or roll back the affected rule under the pre-agreed procedure.

### Answer 8 — Rubric

A strong design may pace the shared heat-treatment resource with a feasible drum schedule and a workload-based upstream buffer.
It may use a local fastener Kanban based on measured supplier replenishment and explicit disruption protection.
A CONWIP cap may control a suitable common line boundary, while rare engineered work uses a distinct workload or priority policy.
The design identifies how the shared resource arbitrates between families and how aged orders are protected.
Regulatory release remains a required hold and is represented accurately in WIP and traceability.
The answer names owners for card exceptions and shortages, and recognizes that low inventory can increase disruption risk or shift costs to suppliers.
Equivalent architectures are acceptable if their boundaries, controls, and tradeoffs are coherent.

## 14. Interview questions

1. **How would you identify a manufacturing constraint?**
   A strong answer compares mix-aware load with reliable capacity and observes starvation, blocking, and queues over a stated horizon.
   A weak answer chooses the busiest-looking machine from one snapshot.
2. **What does Little's law say, and what does it not say?**
   A strong answer states consistent long-run averages and rejects a causal guarantee or percentile interpretation.
   A weak answer assumes halving WIP must halve delivery time without affecting output.
3. **Why might a buffer be valuable?**
   A strong answer explains protection from variability and its cash, space, and service tradeoffs.
   A weak answer calls every buffer waste or claims a larger buffer always fixes insufficient capacity.
4. **What is the difference between local Kanban and CONWIP?**
   A strong answer contrasts local replenishment authorization with a total loop release limit and explains their possible combination.
   A weak answer says both are just cards and therefore identical.
5. **How do you size a Kanban loop?**
   A strong answer defines demand, full replenishment lead time, container quantity, uncertainty allowance, rounding, and validation.
   A weak answer treats a standard percentage as a guaranteed stockout probability.
6. **What happens when the constraint moves?**
   A strong answer repeats the focusing steps and revises the schedule and policies.
   A weak answer keeps maximizing the old resource's utilization indefinitely.
7. **Why is high nonconstraint utilization potentially harmful?**
   A strong answer connects excess release to queue growth while recognizing legitimate cost and resilience tradeoffs.
   A weak answer says idle capacity is always waste.
8. **How would you handle a quality hold in CONWIP?**
   A strong answer keeps the work and card visible until a defined disposition and protects traceability.
   A weak answer returns the card immediately to improve the dashboard without accounting for the held work.
9. **How would you evaluate a WIP reduction?**
   A strong answer includes throughput, delivery, quality, safety, mix, inventory value, and customer or supplier effects.
   A weak answer reports only the reduction in physical stock.
10. **When would you refuse to promise a higher output?**
    A strong answer checks demonstrated reliable capacity, demand, maintenance, quality, and variability before commitment.
    A weak answer converts a best-case machine rate directly into a customer promise.

## 15. Key terms

- [Boundary](#4-definitions-and-vocabulary): the defined entry and exit for a flow measure.
- [Bottleneck](#54-test-candidate-constraints): a resource limiting output under a particular mix and horizon.
- [Buffer](#56-establish-a-drum-buffer-and-rope): deliberate protection against variability.
- [Card](#59-set-the-conwip-cap-and-dispatch-rule): controlled authorization associated with work or a container.
- [CONWIP](#59-set-the-conwip-cap-and-dispatch-rule): a total WIP authorization limit across a loop.
- [Constraint](#55-apply-the-toc-focusing-steps): the limiting factor in the current system.
- [Container quantity](#58-size-a-local-kanban-loop): physical units represented by one local replenishment card.
- [Drum](#56-establish-a-drum-buffer-and-rope): the feasible pace and sequence at the constraint.
- [Effective capacity](#54-test-candidate-constraints): sustainable output after explicitly defined losses.
- [Flow time](#53-measure-current-flow): elapsed time inside the measurement boundary.
- [Forecast](#51-start-with-the-customer-decision): an estimate of future demand.
- [JIT](#3-the-core-mental-model): coordinated production and replenishment to need.
- [Kanban](#58-size-a-local-kanban-loop): a controlled replenishment signal and authorization.
- [Lean](#1-why-this-topic-exists): systematic improvement of value flow with respect for people.
- [Little's law](#3-the-core-mental-model): the relationship between compatible average WIP, throughput, and flow time.
- [Load](#54-test-candidate-constraints): required resource time for the stated mix.
- [Plan](#51-start-with-the-customer-decision): the chosen intended quantity and timing.
- [Protective capacity](#3-the-core-mental-model): nonconstraint slack available for recovery.
- [Pull](#57-select-a-pull-control-architecture): release or replenishment governed by downstream authorization.
- [Queue](#54-test-candidate-constraints): work waiting for a process or decision.
- [Rope](#56-establish-a-drum-buffer-and-rope): release control synchronized to the drum.
- [Service level](#51-start-with-the-customer-decision): a precisely defined measure of customer service.
- [Starvation](#10-failure-modes-and-red-flags): lack of required work at a ready resource.
- [Throughput rate](#53-measure-current-flow): completed units per stated time.
- [TOC](#55-apply-the-toc-focusing-steps): a management approach focused on the limiting factor.
- [Variability](#3-the-core-mental-model): dispersion in demand, processing, availability, or quality.
- [WIP](#53-measure-current-flow): released work not yet completed within the boundary.
- [Workload](#59-set-the-conwip-cap-and-dispatch-rule): remaining resource time represented by work.

## 16. Sources

1. Little, J. D. C. (1961). “A Proof for the Queuing Formula: L = λW.” *Operations Research*, 9(3), 383–387. [https://doi.org/10.1287/opre.9.3.383](https://doi.org/10.1287/opre.9.3.383). Primary mathematical source for the average relationship and its assumptions.
2. Spearman, M. L., Woodruff, D. L., and Hopp, W. J. (1990). “CONWIP: A Pull Alternative to Kanban.” *International Journal of Production Research*, 28(5), 879–894. [https://doi.org/10.1080/00207549008942761](https://doi.org/10.1080/00207549008942761). Foundational CONWIP research; assess model assumptions before transferring conclusions.
3. Toyota Motor Corporation. [Toyota Production System](https://global.toyota/en/company/vision-and-philosophy/production-system/). Company description of JIT and jidoka; a primary source for Toyota's own account, not independent proof of every claimed outcome.
4. Toyota Motor Corporation. [Just-in-Time](https://global.toyota/en/company/vision-and-philosophy/production-system/just-in-time.html). Company explanation of JIT and Kanban; consult for the described system rather than a universal numerical design.
5. Goldratt, E. M., and Cox, J. (1984). *The Goal: A Process of Ongoing Improvement*. North River Press. Influential management narrative, not a controlled case study.
6. International Society of Automation. [ANSI/ISA-95 Enterprise-Control System Integration](https://www.isa.org/standards-and-publications/isa-standards/isa-95-standard). Overview of the standards family and its scope; detailed requirements depend on the applicable edition and contract.
7. GS1. [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications). Identification and data-capture specification; check the applicable edition and trading-partner requirements.
8. International Organization for Standardization. [ISO 9001 — Quality management systems](https://www.iso.org/standard/62085.html). Standard overview; conformance and certification obligations depend on context.
9. U.S. Occupational Safety and Health Administration. [29 CFR 1910.147, Control of hazardous energy (lockout/tagout)](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147). Binding for covered U.S. activities, subject to its scope and exceptions.
10. U.S. Food and Drug Administration / eCFR. [21 CFR Part 117, Current Good Manufacturing Practice, Hazard Analysis, and Risk-Based Preventive Controls for Human Food](https://www.ecfr.gov/current/title-21/chapter-I/subchapter-B/part-117). Check scope, exemptions, and the current text for the specific operation.

Regulatory references were identified for this educational lesson as of November 2026.
They do not replace current legal review or the applicable controlled standard edition.

## 17. Links

- [Module 15 index](_index.md)
- [Module 15 survey](../../../course/15_manufacturing_production_planning.md)
- [Capacity requirements planning](05_capacity-requirements.md)
- [Finite scheduling](06_finite-scheduling.md)
- [Changeovers and SMED](08_changeovers-smed.md)
- [OEE and the loss tree](09_oee-loss-tree.md)
- [SPC and capability](11_spc-cpk.md)
- [Simulation and optimization production labs](../../../course/30_simulation_optimization_production_labs.md)
