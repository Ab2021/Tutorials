# Automation and Micro-Fulfillment

> Module 13 — Warehousing, Fulfillment, and Labor Analytics
> Topic 10 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 13, topics 01–09
> Version 2026.5

## 1. Why this topic exists

A fulfillment director facing an expensive peak can buy equipment, redesign work, outsource volume,
or change the customer promise. Each choice commits a different amount of capital, floor space,
management attention, and operational flexibility. A vendor demonstration cannot establish which
option is best for the actual order mix, peak profile, building, and workforce.

Automated storage and retrieval systems (AS/RS), autonomous mobile robots (AMRs), and goods-to-person
(GTP) workstations alter where inventory waits and where people spend time. They do not eliminate
receiving, replenishment, packing, exception handling, maintenance, or outbound dispatch. A machine
that accelerates picking can make a downstream packing queue longer. A small urban micro-fulfillment
center (MFC) may shorten the delivery journey while duplicating inventory and increasing rent.

The financial decision is also different from the technical decision. A system can work reliably,
meet a throughput specification, and still fail to earn its investment. Conversely, a financially
attractive average-year model may conceal an infeasible peak hour or a safety constraint. This lesson
puts physical flow, staffing, service and cash flows in the same decision frame.

The reader is a site leader, analyst, engineer or finance partner preparing a comparable investment
case. All numerical examples are illustrative, not vendor quotations or claims about a real facility.
The worked case uses constant 2026 US dollars (USD), excludes tax and financing, and treats benefits
as incremental cash flows versus a feasible manual alternative. A forecast is not a commitment.

A decision paper should answer three separate questions. Can the system process the required mix?
Can the organization operate and recover it safely? Does the incremental value justify the capital
and risk? A negative answer to any one requires redesign, renegotiation, or rejection.

## 2. Learning outcomes

After this lesson you can:

1. **Map** manual and automated order flows, including replenishment and exceptions (exercise 1).
2. **Compare** AS/RS, AMR, GTP, and MFC options against item and building constraints (exercise 2).
3. **Calculate** a peak-hour capacity requirement using availability and headroom (exercise 3).
4. **Separate** staffing hours from realizable annual cash savings (exercise 4).
5. **Compute** incremental cash flow, payback and net present value (NPV) (exercise 5).
6. **Challenge** double counting, demand uncertainty, and service assumptions (exercise 6).
7. **Design** a test, fallback and acceptance plan with measurable gates (exercise 7).
8. **Evaluate** inventory placement and local delivery trade-offs for an MFC (exercise 8).

## 3. The core mental model

Automation changes a connected flow, not an isolated rate.

```text
Demand and promise -> release policy -> eligible order lines
                                            |
Supplier -> receive -> quality -> replenish -> storage and retrieval
                                            |              |
                                      manual bypass     robot / AS/RS
                                            |              |
                                            +-> pick / exception
                                                      |
                                                  pack -> stage -> dispatch
                                                      |
                                                 customer outcome

Controls: WMS <-> execution controller <-> equipment; events -> monitoring
Resources: people, space, power, network, stock, maintenance and cash
```

The upper path is the customer's work; the lower path is physical and information infrastructure.
A stockout, control-system outage, full pack buffer or blocked aisle interrupts the same customer
promise even if the robot fleet remains available. Measure delivered orders, not robot motion.

Three boundaries are needed. The **physical boundary** includes both the automated cell and its
feeding and draining processes. The **economic boundary** includes all incremental cash costs and
benefits caused by choosing it over the baseline. The **accountability boundary** names the owner
of each interface, acceptance test, incident and residual risk. Leaving a boundary implicit is a
common way to turn a local improvement into a network-wide loss.

Compare options on a shared demand trace and service promise. First rule out unsafe or physically
infeasible designs. Next test peak throughput and recovery. Then value the feasible alternatives over
the same horizon and decision date. Finally test the result under unfavorable but plausible changes.
A high mean NPV does not excuse a cash shortfall during deployment or a single point of failure.

## 4. Definitions and vocabulary

Each term below has an operational meaning and a common interpretation error.

### AS/RS

- Meaning: automated storage and retrieval system.
- Common misuse: treating storage density as end-to-end throughput.
- Measure or unit: locations and totes/hour.

### AMR

- Meaning: autonomous mobile robot that navigates a defined operating environment.
- Common misuse: assuming unlimited fleet throughput.
- Measure or unit: robots and missions/hour.

### GTP

- Meaning: goods-to-person presentation of stock at a workstation.
- Common misuse: counting presentation as completed customer orders.
- Measure or unit: lines/hour.

### MFC

- Meaning: small fulfillment operation near demand, typically serving a local catchment.
- Common misuse: assuming every local warehouse must be automated.
- Measure or unit: orders/day and square metres.

### WMS

- Meaning: warehouse management system owning configured inventory and task processes.
- Common misuse: assuming it controls every motor.
- Measure or unit: events and inventory records.

### WES

- Meaning: warehouse execution system sequencing work across resources.
- Common misuse: assuming every supplier uses the same system boundary.
- Measure or unit: tasks and events.

### WCS

- Meaning: warehouse control system coordinating equipment-level commands.
- Common misuse: confusing command receipt with physical completion.
- Measure or unit: commands and acknowledgements.

### Throughput

- Meaning: completed work crossing a specified boundary per time.
- Common misuse: equating peak machine cycles with shipped orders.
- Measure or unit: lines/hour or orders/hour.

### Availability

- Meaning: fraction of scheduled time equipment is ready for its intended operation.
- Common misuse: equating readiness with production rate.
- Measure or unit: scheduled hours.

### Utilization

- Meaning: fraction of a resource capacity consumed during a stated interval.
- Common misuse: treating 100% as a safe target.
- Measure or unit: percent.

### Headroom

- Meaning: capacity reserved beyond forecast load.
- Common misuse: counting design capacity as a demand commitment.
- Measure or unit: percent of capacity.

### Tote

- Meaning: container conveying or storing stock.
- Common misuse: assuming all products fit one standard tote.
- Measure or unit: totes.

### SKU

- Meaning: stock keeping unit identifying a distinct inventory item.
- Common misuse: confusing SKU count with units.
- Measure or unit: SKU.

### Order line

- Meaning: one ordered SKU position in an order.
- Common misuse: equating lines and units.
- Measure or unit: lines.

### Replenishment

- Meaning: movement that restores pickable inventory.
- Common misuse: assuming storage always presents stock instantly.
- Measure or unit: units/hour.

### Downtime

- Meaning: time a resource cannot perform its intended function.
- Common misuse: excluding fault recovery or scheduled maintenance without disclosure.
- Measure or unit: minutes per shift.

### OEE

- Meaning: overall equipment effectiveness, commonly availability × performance × quality.
- Common misuse: mixing incompatible denominators.
- Measure or unit: percent.

### Bottleneck

- Meaning: resource limiting system output in a given operating regime.
- Common misuse: assuming it remains fixed under all mixes.
- Measure or unit: orders/hour.

### NPV

- Meaning: sum of dated incremental cash flows discounted to a decision date.
- Common misuse: discounting accounting profit instead of cash.
- Measure or unit: USD.

### Payback

- Meaning: time until undiscounted cumulative incremental cash flow recovers initial investment.
- Common misuse: treating it as a full lifecycle return.
- Measure or unit: years.

### Ramp

- Meaning: period of increasing throughput and learning after go-live.
- Common misuse: assuming steady-state output on day one.
- Measure or unit: months.

### Cutover

- Meaning: controlled transfer from old to new operation.
- Common misuse: assuming a software switch alone transfers physical inventory.
- Measure or unit: date and scope.

### Service level

- Meaning: fraction of defined customer obligations met.
- Common misuse: using a denominator that excludes failed orders.
- Measure or unit: percent of eligible orders.

### Safety stock

- Meaning: inventory held to buffer uncertainty.
- Common misuse: assuming decentralization leaves network inventory unchanged.
- Measure or unit: units.

### Acceptance test

- Meaning: contractual evidence that specified requirements are met.
- Common misuse: using only a vendor simulation as proof.
- Measure or unit: test cases and thresholds.

## 5. The mechanics, step by step

### 5.1 Establish the decision and the counterfactual

Set the investment decision date, demand geography, service promise, lease term, and planning horizon.
Describe the manual option that can actually process the forecast, including hiring, space, equipment,
and service costs. A frozen baseline understates costs if forecast growth would otherwise require a
new shift. Conversely, claiming avoided expansion without a funded or necessary expansion overstates value.
Document whether the decision is to automate an existing site, build a new site, or decentralize stock.
These are different counterfactuals and require different cost boundaries.

### 5.2 Reconstruct demand at decision resolution

Use timestamped order lines and units, including cancellations, stockouts and substitutions. Separate
order creation, release, pick completion and dispatch so delayed work is not mistaken for new demand.
Use at least a representative peak trace where available. Segment by customer promise, channel,
number of lines per order, units per line, cube, weight, fragility, temperature and hazardous properties.
Record whether the sample includes promotions, outages and seasonal closures. A daily average is not a
peak-hour arrival profile. Distinguish an arrival peak from a released-work peak created by batching.

### 5.3 Translate products into eligibility

A product is eligible only if its dimensions, mass, packaging integrity and handling constraints fit the
specified container, gripper and presentation method. Confirm maximum and minimum dimensions and
weight, orientation, crush risk, leak risk, allergen segregation, temperature and regulatory needs.
Measure eligibility by forecast lines and units, not simply by number of SKUs. Calculate the excluded
fraction and design a staffed bypass. A system covering 80% of SKUs may handle far less than 80% of
lines if high-volume products are oversized. Recheck after assortment and package changes.

### 5.4 Select technology candidates

A fixed AS/RS can offer dense storage and repeatable retrieval but needs building envelope, fire-system
compatibility, clear maintenance access and a recovery plan. Its design may be hard to relocate or
extend. Specify retrievals and putaways under the actual inventory distribution, not empty-machine cycles.

AMRs can move carts, shelves or totes along dynamically controlled routes. They require navigable floor,
charging capacity, traffic rules, reliable localization and integration. A fleet sized on uncongested
travel times may queue at lifts, narrow intersections, handoff stations or chargers. Adding robots can
increase congestion rather than capacity. Pedestrian separation and safe operating modes matter.

GTP describes a workflow, not a single storage technology. A workstation may receive goods from fixed
AS/RS, mobile shelves or conveyors. Rate depends on product presentation, scan steps, container changes,
operator reach, order mix, replenishment and exception handling. Ergonomic improvements are a design
hypothesis until tested with workers and a qualified assessor.

An MFC is a network placement and operating model, not necessarily a robot. Local proximity can
reduce delivery distance and improve response, but higher rent, split inventory, replenishment trips,
small-order peaks and last-mile economics can dominate. Compare with a central site and store picking
on the same customer demand and service promise. Model incremental delivery cost, inventory carrying
cost, waste and service benefit only once. Do not treat revenue transfer from another site as new revenue.

### 5.5 Size every limiting resource

Translate peak order arrivals into eligible lines per hour, then into presentation, pick, pack and dispatch
work. Use an observed or defensible design mix, not an unrelated vendor demonstration. Adjust nominal
rates for availability, performance loss and the defined quality loss, without counting the same loss twice.
Provide explicit headroom for variability; a high utilization target is not proof of a feasible queue.
Validate shared equipment and station interactions using trace-driven simulation when appropriate.
Test failure of one station, robot, network link or power supply, including the time to recover the backlog.

### 5.6 Design work and ownership

Map receiving, replenishment, pick, exception, pack, maintenance, supervision, sanitation and recovery.
Assign scheduled staffing by shift and skill, including breaks, training and absences. Distinguish hours
freed from positions removed, overtime avoided and additional output enabled. Savings become cash only
when a cost can actually be avoided, a vacancy not filled, a contract reduced or an operating schedule
changed. Treat severance and training as incremental transition costs where applicable. Consult workers
and safety specialists before imposing a new rate or changing walking and lifting patterns.

### 5.7 Build a lifecycle comparison

Include equipment, integration, installation, construction, permitting, network, testing, inventory
migration, training, spares and contingency at their expected payment dates. Include maintenance, software,
power, insurance, support, consumables, staffing and facilities costs after go-live. Add decommissioning
or residual value only when a defensible cash amount exists. Use consistent real cash flows and a real
discount rate, or nominal flows and nominal rate. Compare all candidates over the same horizon and define
what happens after the horizon. Do not add depreciation as a cash cost when capital expenditure is
already included; tax effects require a separately specified tax model.

### 5.8 Test, contract, deploy and measure

Make the statement of work specify boundaries, data, error recovery, safety responsibilities, response
times, spare parts, license terms and source-system dependencies. Acceptance must be based on a
representative item and order mix, defined measurement intervals, agreed exclusions and end-to-end
service rather than theoretical component cycles. Test normal load, peak load, degraded mode, restart,
inventory reconciliation and cybersecurity recovery. Name a failure owner and a rollback trigger.
After go-live reconcile physical inventory, orders and cash savings to the agreed baseline. Retire a
benefit claim that cannot be observed rather than changing its denominator after the fact.

### Operating decision: Demand arrival

**Input.** Use hourly order creation and promised dispatch.

**Action.** Measure timestamped arrivals for ordinary and peak days.

**Why it matters.** Averages hide concentrated arrivals and create queues.

**Decision test.** Use percentiles and replay actual traces, preserving correlations.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Release policy

**Input.** Use release events and backlog age.

**Action.** Separate arrivals from waves released by the WMS.

**Why it matters.** Batch release can create an avoidable artificial peak.

**Decision test.** Compare smoothed release with the same dispatch deadlines.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Line conversion

**Input.** Use orders, lines and units by order.

**Action.** Compute lines per order for each channel and hour.

**Why it matters.** A fixed conversion misses multi-line promotion baskets.

**Decision test.** Use the joint order-size distribution rather than independent averages.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Product fit

**Input.** Use item dimensions, weight and orientation.

**Action.** Measure physical samples and package variants.

**Why it matters.** Catalog dimensions may describe the case rather than the each.

**Decision test.** Validate the highest-volume items before signing.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Excluded flow

**Input.** Use lines and units outside machine constraints.

**Action.** Route noneligible items through a staffed bypass.

**Why it matters.** Excluded orders may still require consolidation with automated picks.

**Decision test.** Measure end-to-end order completion including bypass wait.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Stock coverage

**Input.** Use eligible demand and inventory location.

**Action.** Check whether required stock is actually in the automated zone.

**Why it matters.** Eligibility alone does not guarantee on-hand stock.

**Decision test.** Include replenishment and reserve stock in the design.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Storage slots

**Input.** Use location dimensions and usable capacity.

**Action.** Convert physical positions into usable item capacity.

**Why it matters.** Empty-position allowance and incompatible items reduce usable slots.

**Decision test.** Calculate storage feasibility by item size class.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Retrieval sequence

**Input.** Use tote retrieval and return events.

**Action.** Measure retrievals per order and batching effects.

**Why it matters.** Multiple presentations may be needed for a single customer order.

**Decision test.** Replay the observed order and stock allocation pattern.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Workstation rate

**Input.** Use scan and completion timestamps.

**Action.** Measure completed good lines per staffed hour.

**Why it matters.** Demonstration rate omits walking, dunnage and exceptions.

**Decision test.** Use the planned item mix and paid-hour denominator.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Staffed capacity

**Input.** Use shift schedules and skill availability.

**Action.** Translate paid hours to available workstation coverage.

**Why it matters.** Unstaffed machines provide no usable throughput.

**Decision test.** Include breaks, training and absence coverage.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Packing

**Input.** Use carton selection and pack completion.

**Action.** Measure pack minutes by basket and product.

**Why it matters.** GTP can move the bottleneck downstream.

**Decision test.** Size pack and carton supply for the same peak trace.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Dispatch

**Input.** Use carrier collection times and dock resources.

**Action.** Calculate completion deadlines backwards from pickup.

**Why it matters.** High picking throughput is useless after the cutoff.

**Decision test.** Measure orders ready before the actual carrier cutoff.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Replenishment

**Input.** Use reserve stock, movements and lead time.

**Action.** Forecast units drawn and trigger replenishment in advance.

**Why it matters.** A fast workstation starves if stock cannot reach it.

**Decision test.** Test simultaneous replenishment and picking load.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Traffic congestion

**Input.** Use robot paths and station queues.

**Action.** Track missions, speed and waits by segment.

**Why it matters.** Adding vehicles can saturate a narrow aisle.

**Decision test.** Test fleet increments against completed orders not missions.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Charging

**Input.** Use charge state and charging station usage.

**Action.** Model charging policy and available electrical capacity.

**Why it matters.** A nominal fleet cannot work while all batteries charge.

**Decision test.** Stress simultaneous low-charge and peak-demand conditions.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Availability

**Input.** Use scheduled time and logged interruption.

**Action.** Define uptime and planned maintenance consistently.

**Why it matters.** Vendor availability may exclude waiting on another subsystem.

**Decision test.** Report both equipment readiness and end-to-end production.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Quality

**Input.** Use wrong picks, damages and rework.

**Action.** Record quality failures at order-line and order levels.

**Why it matters.** Counting a scan as a good customer line ignores later correction.

**Decision test.** Count only accepted, correct output at the chosen boundary.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Inventory integrity

**Input.** Use stock movements and reconciliation differences.

**Action.** Trace each tote and quantity through state changes.

**Why it matters.** A restart can duplicate a movement or strand a tote.

**Decision test.** Reconcile physical stock to WMS after fault recovery.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Interfaces

**Input.** Use message IDs, acknowledgements and retries.

**Action.** Map system ownership and transaction states.

**Why it matters.** A transport acknowledgement does not prove physical completion.

**Decision test.** Test duplicate, delayed and out-of-order messages.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Maintenance

**Input.** Use fault records and spare-part lead times.

**Action.** Specify preventive and corrective work windows.

**Why it matters.** A low annual service fee may exclude critical parts.

**Decision test.** Price the full support scope and local response coverage.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Facilities

**Input.** Use floor loading, power and emergency egress.

**Action.** Survey the actual site with qualified specialists.

**Why it matters.** The building may not support racks or required clearances.

**Decision test.** Obtain engineering review before capital authorization.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Fire protection

**Input.** Use commodity classification and local requirements.

**Action.** Assess stored goods, rack configuration and suppression.

**Why it matters.** Higher-density storage can change the protection design.

**Decision test.** Obtain approval from the authority having jurisdiction.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Worker safety

**Input.** Use person-machine interactions and task observations.

**Action.** Assess hazards through operating and maintenance modes.

**Why it matters.** Protecting normal production while ignoring jam clearing is insufficient.

**Decision test.** Verify isolation and recovery procedures with affected workers.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Cybersecurity

**Input.** Use assets, accounts and network flows.

**Action.** Inventory controller dependencies and remote access.

**Why it matters.** Vendor connectivity can become an uncontrolled entry path.

**Decision test.** Apply least privilege and test restoration from backups.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Local assortment

**Input.** Use catchment demand and shelf life.

**Action.** Select SKUs using local demand and service needs.

**Why it matters.** Replicating the central range increases slow stock and waste.

**Decision test.** Measure local fill rate against carrying and spoilage cost.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Last mile

**Input.** Use delivery routes, windows and drop density.

**Action.** Model incremental route cost by catchment and time.

**Why it matters.** Shorter distance to one customer does not guarantee cheaper routes.

**Decision test.** Compare routes with realistic vehicle and time-window constraints.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Inventory pooling

**Input.** Use demand variability and target fill rate.

**Action.** Estimate stock required at each facility.

**Why it matters.** Splitting a pooled stock position can increase safety stock.

**Decision test.** Calculate working capital and waste across all sites.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Transition

**Input.** Use training, parallel running and shutdown windows.

**Action.** Schedule cutover costs and temporary capacity.

**Why it matters.** A learning curve can increase overtime during ramp.

**Decision test.** Budget the ramp and retain a tested manual fallback.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Commercial contract

**Input.** Use prices, service terms and obligations.

**Action.** Separate fixed, variable and contingent charges.

**Why it matters.** Per-pick fees can change the economics at high volume.

**Decision test.** Evaluate volume bands and exit obligations over the horizon.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Operating decision: Benefit ownership

**Input.** Use staff schedules, invoices and cost centers.

**Action.** Assign each claimed saving to an accountable manager.

**Why it matters.** Freed hours may be redeployed without reducing cash cost.

**Decision test.** Verify realized changes against payroll or supplier invoices.

**Output.** Record the assumption, owner and measurement interval in the option comparison.

**Branch.** If the evidence is weak, collect a sample or run a bounded test before committing capital.

### Fixed automation check: Structural loading

**Action:** Compare fully loaded rack reactions with a structural engineer’s approved slab design.

**Reason:** A floor that supports pallets may not support concentrated equipment loads.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Fixed automation check: Clear height

**Action:** Measure usable height below sprinklers, lighting, ducts and structural members.

**Reason:** A nominal building height can overstate storage positions.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Fixed automation check: Retrieval mix

**Action:** Measure retrieval and putaway demand in the same peak interval.

**Reason:** Simultaneous replenishment can consume travel and lift capacity.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Fixed automation check: Access for maintenance

**Action:** Plan safe access to lifts, drives and obstructed storage positions.

**Reason:** A dense design without access prolongs recovery.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Fixed automation check: Expansion

**Action:** Identify physical extension paths and control-system license limits.

**Reason:** Adding storage positions does not necessarily increase station throughput.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Mobile robots check: Floor condition

**Action:** Survey slopes, joints, surface friction and thresholds along all routes.

**Reason:** A route demonstrated on a clean flat floor may fail at the site.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Mobile robots check: Localization

**Action:** Test navigation around moving people, carts and changing shelves.

**Reason:** A map can become stale as the building changes.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Mobile robots check: Traffic management

**Action:** Identify intersections, one-way routes and pedestrian crossings.

**Reason:** A blocked intersection can idle several otherwise healthy robots.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Mobile robots check: Charging schedule

**Action:** Check charger count, power, battery state and planned peak duty.

**Reason:** Total daily charging energy alone does not prove peak availability.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Mobile robots check: Fleet degradation

**Action:** Remove a robot from the peak test and measure backlog recovery.

**Reason:** A spare robot has value only if it can be charged and deployed in time.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Goods-to-person check: Presentation consistency

**Action:** Measure tote arrival spacing and product accessibility at the station.

**Reason:** A fast retrieval average can still produce long operator idle intervals.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Goods-to-person check: Ergonomic reach

**Action:** Observe reach, repetition, force and posture with actual products.

**Reason:** Less walking can be offset by repetitive upper-body work.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Goods-to-person check: Order consolidation

**Action:** Trace multi-zone orders from all pick paths to the pack location.

**Reason:** Fast automated lines cannot complete an order while a manual line is missing.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Goods-to-person check: Container changes

**Action:** Include empty totes, full order containers and dunnage replenishment.

**Reason:** Ignoring changeovers inflates measured workstation productivity.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Goods-to-person check: Error correction

**Action:** Time wrong-item, damaged-item and missing-stock exception flows.

**Reason:** Excluding exception work moves cost outside the reported rate.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Micro-fulfillment check: Delivery density

**Action:** Build routes using real customer geography, windows and fleet constraints.

**Reason:** A nearby node may still have expensive sparse delivery routes.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Micro-fulfillment check: Assortment

**Action:** Measure local line demand and substitution outcomes by SKU.

**Reason:** A small footprint may force exclusions that worsen customer service.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Micro-fulfillment check: Replenishment frequency

**Action:** Plan supplier or central-depot movements using actual local storage limits.

**Reason:** A smaller local footprint can increase inbound trip frequency.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Micro-fulfillment check: Inventory pooling

**Action:** Model inventory and safety stock at every affected network node.

**Reason:** Stock split between sites may increase total buffer inventory.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Micro-fulfillment check: Perishables

**Action:** Estimate age at receipt, expiration and disposal by product class.

**Reason:** Slow local demand can turn a speed advantage into higher spoilage.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Investment check: Capital boundary

**Action:** Include design, software, building work, integration, testing and training in the quote.

**Reason:** Comparing equipment-only prices with an all-in baseline creates a false advantage.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Investment check: Cash timing

**Action:** Place each capital payment and benefit in its actual expected period.

**Reason:** A late ramp or early payment changes NPV even if totals stay fixed.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Investment check: Volume sensitivity

**Action:** Recompute benefits and variable costs at lower and higher demand.

**Reason:** A fixed-cost system can have poor economics when demand falls.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Investment check: Residual exposure

**Action:** State relocation, removal and disposal costs at lease end.

**Reason:** Ignoring exit obligations can make a short lease appear profitable.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Investment check: Financing

**Action:** Evaluate the project using unlevered cash flows before separately considering financing.

**Reason:** Mixing interest expense with a discount rate that already reflects financing double counts cost.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Implementation check: Parallel operation

**Action:** Define how inventory and orders are assigned while both systems run.

**Reason:** Two active systems can each believe they own the same stock.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Implementation check: Cutover gate

**Action:** Require trained staff, reconciled inventory and passed safety checks before release.

**Reason:** A fixed launch date does not establish operational readiness.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Implementation check: Rollback

**Action:** Specify physical and software steps for returning work to manual flow.

**Reason:** Switching off the controller does not automatically recover stranded stock.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Implementation check: Supplier support

**Action:** Define escalation, response time, parts ownership and access windows.

**Reason:** A response-time promise is not necessarily a repair-time commitment.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

### Implementation check: Post-launch review

**Action:** Compare paid hours, service and exceptions against the approved baseline.

**Reason:** An unchanged benefit forecast can hide missed savings after go-live.

**Evidence:** Record the measurement, date, source and owner in the decision file.

**Gate:** Resolve any material gap before relying on the associated throughput or benefit.

## 6. Worked example

### 6.1 Decision and demand

A regional retailer is considering a GTP installation in an existing leased warehouse. The alternative
is to retain manual picking with a feasible staffing plan. All figures are illustrative 2026 USD.
The 5-year horizon reflects the remaining lease; no renewal or resale value is assumed. The forecast
is 1,000,000 order lines/year, of which 800,000 lines/year fit the automated flow. The other 200,000
lines/year remain manual in both alternatives. Demand is a forecast, not a volume commitment.

The busiest design hour contains 600 eligible lines/hour. Each of two stations has a nominal rate of
400 lines/hour while running. Availability is 90% of scheduled time and performance is 90% of the
available-time rate; assume no additional quality loss in the throughput calculation. This does not
claim that a shift-average factor guarantees a particular hour's output.

Effective design capacity = 2 × 400 × 0.90 × 0.90 = **648 lines/hour**.
The load is 600 / 648 = **92.59%** of that capacity, with 48 lines/hour, or **7.41%** of capacity,
remaining. If the design criterion is 15% headroom, 600 / 0.85 = **705.88 lines/hour** is required.
The two-station design therefore fails that criterion. Adding a third identical, independently fed and
staffed station gives 972 lines/hour before shared-resource constraints. The commercial case below
assumes the initial investment already includes three stations; independent station performance and
upstream and downstream capacity still require testing. The third station is not free.

### 6.2 Labor and cash benefit

The manual alternative needs 10,000 paid pick hours/year for the 800,000 eligible lines, or 80
lines/paid hour. The automated design needs 4,500 paid operating hours/year for those lines, including
replenishment and exceptions inside the cell. It also needs 1,000 paid maintenance hours/year.
The unchanged manual bypass is excluded from both alternatives. At USD 30/paid hour, the difference
is (10,000 − 4,500 − 1,000) × 30 = **USD 135,000/year** of potential labor savings. This becomes
cash only if the staffing plan actually removes those paid hours; otherwise it is a capacity benefit.

The design also avoids USD 45,000/year in independently evidenced overtime and temporary labor not
already included in the 10,000 hours; avoids USD 30,000/year of damage expense; and adds USD
20,000/year in power and consumables plus USD 40,000/year in software and external maintenance.
Steady incremental annual cash flow is 135,000 + 45,000 + 30,000 − 20,000 − 40,000 = **USD
150,000/year**. No revenue uplift is credited. Distinct payroll records must verify that overtime was
not already counted in paid pick hours.

Initial installed capital, integration and training total USD 480,000 at time zero. Year 1 is a ramp year
with only 60% of steady cash benefit, or USD 90,000; years 2–5 each generate USD 150,000. The real
discount rate is 10% per year, with cash received at each year end. No tax, financing, inflation,
terminal value or incremental working capital is modeled. These are simplifying assumptions, not
assertions that those cash flows never exist.

| Year | Incremental cash flow, USD | Discount factor at 10% | Present value, USD |
|---:|---:|---:|---:|
| 0 | -480,000 | 1.0000 | -480,000 |
| 1 | 90,000 | 0.9091 | 81,818 |
| 2 | 150,000 | 0.8264 | 123,967 |
| 3 | 150,000 | 0.7513 | 112,697 |
| 4 | 150,000 | 0.6830 | 102,452 |
| 5 | 150,000 | 0.6209 | 93,138 |

NPV is approximately **USD 34,072**, subject to rounding. Undiscounted cumulative cash flow becomes
positive during year 4: after year 3 the unrecovered balance is USD 90,000; 90,000 / 150,000 = 0.6
year, so simple payback is **3.6 years** under a uniform year-4 cash accrual assumption. Annual
cash-flow timing used for NPV remains year-end; interpolated payback uses the explicitly different
within-year assumption. The two measures answer different questions.

### 6.3 Sensitivity and decision

If steady benefits are 20% lower, year-1 cash flow falls to USD 72,000 and years 2–5 to USD 120,000.
NPV becomes approximately **negative USD 68,742**. If initial cost rises by USD 60,000 with no
other change, NPV becomes approximately **negative USD 25,928**. These are separate scenarios,
not probability-weighted forecasts. The margin is narrow relative to both risks.

Proceed to detailed design only if the three-station quote includes the stated scope, a trace-based
capacity test confirms peak performance and recovery, and the finance owner confirms each saving is
cash-realizable. Otherwise revise the design or keep the manual alternative. A positive base NPV alone
is not authorization to ignore the failed two-station headroom test.

## 7. Code, query, or calculation

The following standard-library Python calculates the stated cash flows and two separate sensitivities.
Run with Python 3; no external package or file is needed. It assumes year-end cash flows and the
uniform within-year accrual convention only for interpolated simple payback.

```python
rate = 0.10
capex = 480_000
steady = 150_000
ramp = 0.60
horizon = 5

def cash_flows(initial, annual, ramp_fraction):
    return [-initial, annual * ramp_fraction] + [annual] * (horizon - 1)

def npv(discount_rate, flows):
    return sum(flow / (1 + discount_rate) ** year
               for year, flow in enumerate(flows))

def payback(flows):
    balance = flows[0]
    for year, flow in enumerate(flows[1:], start=1):
        if flow > 0 and balance + flow >= 0:
            return year - 1 + (-balance / flow)
        balance += flow
    return None

base = cash_flows(capex, steady, ramp)
downside = cash_flows(capex, steady * 0.8, ramp)
overrun = cash_flows(capex + 60_000, steady, ramp)
capacity = 3 * 400 * 0.90 * 0.90
required = 600 / 0.85
print(f"capacity={capacity:.0f} lines/hour")
print(f"required={required:.2f} lines/hour")
print(f"base NPV=USD {npv(rate, base):,.0f}")
print(f"simple payback={payback(base):.2f} years")
print(f"benefit downside NPV=USD {npv(rate, downside):,.0f}")
print(f"cost overrun NPV=USD {npv(rate, overrun):,.0f}")
```

Expected output, rounded to the displayed precision:

```text
capacity=972 lines/hour
required=705.88 lines/hour
base NPV=USD 34,072
simple payback=3.60 years
benefit downside NPV=USD -68,742
cost overrun NPV=USD -25,928
```

The calculation does not simulate queues or establish cash realization. If net annual benefit can
change sign, cumulative cash may cross zero more than once; report all crossings rather than assuming
this simple function is a complete risk model. A missing payback within the horizon returns `None`.

## 8. Data, standards, and regulatory context

Build a data dictionary before estimating performance. The order-line key should connect order ID,
line ID, SKU, quantity, event timestamp, event type, location and exception reason. Use an unambiguous
time zone, retain event time and ingestion time separately, and document late or corrected events.
Protect customer and employee identifiers with role-based access and a stated retention schedule.
Task events should distinguish assignment, start, interruption, completion and reversal. Machine
events should include component, fault, state, timestamp and incident identifier. Link events through
stable correlation identifiers; a device message is not necessarily a completed inventory transaction.

Inventory records need unit of measure, lot or serial where relevant, quantity, location, status and
last confirmed movement. Item data need dimensions, weight, package level, hazardous attributes,
handling limits and effective dates. Finance needs incremental cost center, paid hours, fully loaded
rate, capital timing, service contract and ownership of benefits. Reconcile totals to the general
ledger and operational counts; do not assume that a WMS event is a payroll hour.

GS1 identifiers and barcode standards can support consistent product, logistic-unit and location
identification. Their applicability depends on trading partner and implementation requirements;
using a GS1 identifier does not itself prove that physical content is correct. ISO 3691-4 addresses
safety requirements for driverless industrial trucks and their systems; applicability depends on the
specific machine and jurisdiction. The ISO 12100 risk-assessment framework supports systematic hazard
identification, risk reduction and documentation; certification or compliance is not established by
reading the standard. Applicable US OSHA requirements, including control of hazardous energy under
29 CFR 1910.147 where applicable, must be evaluated for the actual equipment and work. Local building,
fire and electrical authorities determine applicable approvals. These are educational references,
not a claim that a particular configuration is compliant; competent local review remains necessary.

A data contract should define schema version, event meaning, idempotency key, retry and ordering rules,
maximum tolerated delay, escalation and reconciliation owner. Record both missing events and late
arrivals. Do not overwrite a fault with a later normal status: retain the event history needed to
reconstruct duration. Protect remote supplier access and test backup restoration, not only backups.

## 9. Industry practice and case evidence

**Ocado Group and Kroger.** Kroger has publicly described customer fulfillment centers using Ocado
technology in its corporate announcements. This is company-reported evidence that a grocery retailer
has deployed a highly automated fulfillment model, not independent proof that any particular local
MFC has positive NPV. A large customer fulfillment center and a smaller urban MFC have different
capacity, inventory pooling and last-mile economics. The analyst must not transfer a site's throughput
or return to a different geography without the underlying operating data.

**Amazon and mobile robotics.** Amazon describes its use of mobile robots in its fulfillment network.
The company is the source of the operational description and has incentives to emphasize benefits.
It does not follow that a smaller retailer can achieve the same utilization, fleet economics, safety
outcomes or learning curve. Inference: the case illustrates goods moving toward people and the need
to coordinate equipment with human workflows; it is not a controlled comparison of alternatives.

**Public evidence quality.** Vendor brochures can identify candidate technologies and dimensions,
but their demonstrations frequently omit the buyer's product mix and site-specific cost. Treat
published nominal rates as hypotheses for a contractual test, not verified expected performance.
A peer-reviewed study, a regulator's official guidance, a company announcement and a vendor claim
answer different questions. Prefer dated operating data and a matched counterfactual when making
financial claims. Ask whether results include ramp, maintenance, failed orders and manual bypass.

## 10. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Peak overload | Queue age increases before cutoff | Peak rate exceeds usable capacity | Late dispatch | Replay peaks and reserve headroom |
| Bad item data | Frequent fit overrides | Incorrect dimensions or units | Jams and bypass work | Physically verify high-volume items |
| Starved station | Empty presentation buffer | Replenishment is undersized | Paid idle time | Size concurrent replenishment |
| Pack bottleneck | Completed picks accumulate | Downstream labor is fixed | Late orders | Test full flow and add pack capacity |
| Fleet congestion | Missions slow as fleet grows | Shared aisle saturates | Lower marginal throughput | Simulate traffic and redesign routes |
| Interface loss | Unacknowledged commands grow | Messages fail or duplicate | Inventory mismatch | Idempotency and reconciliation |
| Benefit double count | Payroll and overtime overlap | Same hours valued twice | Inflated NPV | Tie each benefit to a unique cost |
| No cash conversion | Hours freed but payroll unchanged | Staff redeployed rather than reduced | Missing savings | Obtain staffing and finance signoff |
| Poor ramp | Training hours exceed plan | Learning and defects ignored | Overtime and delayed returns | Phase deployment and budget ramp |
| Unsafe jam clearing | Frequent manual interventions | Hazardous energy remains present | Serious injury | Risk assessment and isolation |
| Service dependency | Vendor response is slow | Unavailable parts or remote support | Extended outage | Specify spares and response coverage |
| Inventory duplication | Local stock rises after launch | Demand pooling is lost | Working capital and waste | Model stock by node and expiry |
| Lease mismatch | Payback exceeds remaining term | Equipment cannot be relocated cheaply | Stranded capital | Align horizon and lease terms |
| Misleading uptime | Vendor metric excludes upstream waits | Boundary is too narrow | Hidden customer failures | Measure end-to-end output |
| Cyber incident | Unmanaged remote accounts | Insecure supplier access | Operational disruption | Restrict access and test recovery |
| Faulty recovery | System restarts with mismatched stock | Commands were replayed | Duplicate picks | Test restart and reconciliation |
| Uncovered bypass | Excluded SKU share increases | Assortment changes | Manual backlog | Monitor eligible line share |
| Premature acceptance | Only nominal rate is tested | No representative mix or degraded test | Contractual remedies weaken | Agree test protocol before award |

## 11. Assumptions and limits

| Assumption | Why needed | What breaks it | Test |
|---|---|---|---|
| Forecast volume is representative | Sizes the resource and cash flows | Promotion or channel shift | Replay alternative demand traces |
| Eligible mix remains stable | Determines automated volume | Packaging or assortment change | Audit dimensions and line mix monthly |
| Staffing savings are cash | Converts hours to benefit | Staff are retained without cost avoidance | Compare actual paid schedules and baseline |
| Three stations share no hidden bottleneck | Supports nominal capacity | Shared lift or pack station saturates | Test integrated peak performance |
| Rate factors can be multiplied | Provides a first capacity estimate | Correlated faults and variable mix | Compare with empirical shift output |
| Ramp lasts one year | Dates benefits | Integration or learning delays | Use staged readiness and update forecast |
| Manual counterfactual is feasible | Makes the comparison meaningful | Labor shortage or space shortfall | Document shifts and space requirements |
| Discount rate matches cash basis | Makes NPV internally consistent | Nominal cash with real rate | Review financial assumptions |
| No terminal value | Avoids unsupported residual benefit | Equipment has material resale or disposal cost | Obtain binding or credible estimates |
| Safety controls remain effective | Enables continued operation | Unreviewed modifications | Periodic risk review and inspections |
| Item inventory is available | Connects demand to throughput | Replenishment delays or stockouts | Track stockout and presentation delays |
| No network demand transfer credited | Avoids fictitious sales | Incremental market demand demonstrably exists | Compare demand with a defensible control |

## 12. Exercises

Each exercise requires the named deliverable and acceptance criteria. Use illustrative assumptions when
real data are unavailable and mark them as assumptions, not observations.

### Exercise 1: Flow map

**Time:** 12 minutes.

**Deliverable:** Draw the current and proposed order and inventory flows.

**Acceptance criteria:** Include receiving, replenishment, automated and manual bypass, packing, dispatch and exception recovery.

### Exercise 2: Technology selection

**Time:** 15 minutes.

**Deliverable:** Compare fixed AS/RS, AMR and manual flow for a leased facility with uncertain volume.

**Acceptance criteria:** Identify one constraint, one benefit and one failure mode for each; justify a conditional recommendation.

### Exercise 3: Capacity test

**Time:** 10 minutes.

**Deliverable:** For two 400-line/hour stations with 90% availability and 90% performance, calculate capacity and headroom against 600 lines/hour.

**Acceptance criteria:** Show 648 lines/hour, 7.41% headroom and the decision against a 15% requirement.

### Exercise 4: Cash realization

**Time:** 12 minutes.

**Deliverable:** Calculate annual potential savings from 10,000 baseline hours, 4,500 operating hours and 1,000 maintenance hours at USD 30/hour.

**Acceptance criteria:** Show USD 135,000/year and explain what makes it cash rather than redeployed capacity.

### Exercise 5: Investment return

**Time:** 20 minutes.

**Deliverable:** Calculate NPV and simple payback for the worked case.

**Acceptance criteria:** Use the 10% rate and ramp, show approximately USD 34,072 and 3.6 years, and distinguish timing conventions.

### Exercise 6: Challenge the case

**Time:** 15 minutes.

**Deliverable:** Review a claimed USD 150,000/year benefit that includes the same overtime in baseline labor and avoided overtime.

**Acceptance criteria:** Identify double counting, propose a reconciliation, and explain the directional effect on NPV.

### Exercise 7: Acceptance and recovery

**Time:** 20 minutes.

**Deliverable:** Draft an integrated acceptance plan for the three-station design.

**Acceptance criteria:** Specify representative mix, pass thresholds, normal and degraded tests, reconciliation, owner and rollback trigger.

### Exercise 8: Micro-fulfillment decision

**Time:** 20 minutes.

**Deliverable:** Design a comparison of a local MFC with central delivery.

**Acceptance criteria:** Include service promise, delivery routes, inventory, rent, replenishment, waste and incremental rather than transferred sales.

## 13. Self-check answers

### Exercise 1

A valid map shows inbound receipt and quality checks feeding reserve stock, then replenishment feeding
the automated inventory. Eligible lines go to retrieval and workstation picking; ineligible lines go
through a manual bypass. Both streams join before packing and dispatch. Draw exception handling and
inventory reconciliation as explicit loops. A map omitting the bypass or inbound replenishment is
incomplete because the service promise still applies to those flows.

### Exercise 2

A strong comparison explains that fixed AS/RS can increase density but may be hard to relocate;
AMRs can be redeployed more readily but need aisle, traffic and charging feasibility; manual flow
requires less equipment commitment but can face hiring and peak capacity limits. None is universally
best. A short lease and uncertain demand increase the value of flexibility; a binding quote, exit cost,
physical study and comparable demand trace are needed before selecting a design.

### Exercise 3

Two stations deliver 2 × 400 × 0.90 × 0.90 = 648 lines/hour. Spare capacity is 48 lines/hour.
Headroom is 48 / 648 = 7.41%, below the 15% criterion. The required effective capacity is
600 / 0.85 = 705.88 lines/hour. This arithmetic does not prove that a third station can be fed.

### Exercise 4

The difference is 10,000 − 4,500 − 1,000 = 4,500 paid hours/year. At USD 30/hour, potential
savings are USD 135,000/year. Confirm that the hourly rate is comparable, the maintenance hours
are incremental, and schedules or invoices actually decline. Redeployment without cost avoidance is
not a cash saving, even if it improves capacity or quality.

### Exercise 5

Discount the USD 90,000 year-1 benefit and four USD 150,000 year-end benefits at 10%, then subtract
USD 480,000. NPV is approximately USD 34,072. After year 3, USD 90,000 is unrecovered; uniform
year-4 accrual gives 90,000 / 150,000 = 0.6 year and simple payback of 3.6 years. Discounted
payback differs, and interpolated accrual is not the same assumption as year-end NPV cash receipts.

### Exercise 6

Remove the duplicate overtime benefit or remove the overlapping hours from the baseline, not both.
Trace time records to payroll and supplier invoices. Recalculate each year's cash flow and NPV; a
positive duplicate removal decreases NPV. The exact adjustment requires the overlap amount and its
timing, which the question does not supply.

### Exercise 7

A strong plan defines a representative order trace, product mix, inventory and staffing; the end-to-end
completion boundary; a peak output and on-time threshold; and a sustained observation window.
It tests a station failure, backlog recovery, restart, duplicate messages and physical-to-WMS inventory
reconciliation. Name operator, supplier and customer owners, a safety stop, and a documented rollback
trigger. A supplier-only demonstration of nominal cycles is insufficient.

### Exercise 8

A strong answer compares both network designs over the same demand, promise and horizon. It includes
local facility costs, stock and shrink, inbound replenishment, actual delivery-route costs and displaced
central activity. Revenue counts only if incremental to the network. Test sensitivity to local density,
assortment, waste and peak concentration. No numerical winner is justified without those inputs.

## 14. Interview questions

### Question 1

**Question:** Why does a robot rate not prove customer throughput?

**Strong answer:** Describe upstream supply, staffing, packing and dispatch constraints and a shared boundary.

**Weak answer:** Quote only a machine cycle rate.

### Question 2

**Question:** How do you calculate available capacity?

**Strong answer:** State scheduled hours, availability, performance and quality boundaries without double counting.

**Weak answer:** Multiply unrelated vendor percentages with no units.

### Question 3

**Question:** What is the difference between freed hours and savings?

**Strong answer:** Connect saved paid hours to a schedule, vacancy, contract or invoice reduction.

**Weak answer:** Assume all theoretical hours immediately reduce payroll.

### Question 4

**Question:** When does an MFC make sense?

**Strong answer:** Compare local demand density and service value against stock duplication, rent and replenishment.

**Weak answer:** Claim proximity automatically lowers total cost.

### Question 5

**Question:** How would you size a fleet?

**Strong answer:** Use mission trace, travel, traffic, charging, stations and degraded scenarios.

**Weak answer:** Divide average distance by robot speed and stop.

### Question 6

**Question:** What is an acceptable automation test?

**Strong answer:** Use representative mix, end-to-end throughput, agreed exclusions, safety and recovery.

**Weak answer:** Accept a short empty-system demonstration.

### Question 7

**Question:** How would you prevent double counting?

**Strong answer:** Reconcile benefits to distinct baseline costs and assign one owner per claim.

**Weak answer:** Add productivity, overtime and headcount benefits without examining overlap.

### Question 8

**Question:** Why might 90% utilization be risky?

**Strong answer:** Explain variability, correlated arrivals, queue growth and service deadlines.

**Weak answer:** Assume unused capacity is always waste.

### Question 9

**Question:** What should happen after a network outage?

**Strong answer:** Describe safe stop, transaction states, restart, physical reconciliation and fallback.

**Weak answer:** Restart equipment and assume records are correct.

### Question 10

**Question:** How would you decide with positive NPV and negative downside?

**Strong answer:** Discuss risk appetite, cash exposure, sensitivity, contract options and capacity gates.

**Weak answer:** Treat positive base NPV as unconditional approval.

## 15. Key terms

The section 4 definitions provide units and common misuses; the links below point to the corresponding
operational discussion.

- **AS/RS:** automated storage and retrieval system; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **AMR:** autonomous mobile robot that navigates a defined operating environment; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **GTP:** goods-to-person presentation of stock at a workstation; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **MFC:** small fulfillment operation near demand, typically serving a local catchment; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **WMS:** warehouse management system owning configured inventory and task processes; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **WES:** warehouse execution system sequencing work across resources; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **WCS:** warehouse control system coordinating equipment-level commands; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Throughput:** completed work crossing a specified boundary per time; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Availability:** fraction of scheduled time equipment is ready for its intended operation; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Utilization:** fraction of a resource capacity consumed during a stated interval; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Headroom:** capacity reserved beyond forecast load; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Tote:** container conveying or storing stock; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **SKU:** stock keeping unit identifying a distinct inventory item; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Order line:** one ordered SKU position in an order; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Replenishment:** movement that restores pickable inventory; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Downtime:** time a resource cannot perform its intended function; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **OEE:** overall equipment effectiveness, commonly availability × performance × quality; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Bottleneck:** resource limiting system output in a given operating regime; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **NPV:** sum of dated incremental cash flows discounted to a decision date; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Payback:** time until undiscounted cumulative incremental cash flow recovers initial investment; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Ramp:** period of increasing throughput and learning after go-live; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Cutover:** controlled transfer from old to new operation; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Service level:** fraction of defined customer obligations met; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Safety stock:** inventory held to buffer uncertainty; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

- **Acceptance test:** contractual evidence that specified requirements are met; see [definitions](#4-definitions-and-vocabulary) and [mechanics](#5-the-mechanics-step-by-step).

## 16. Sources

- [GS1 General Specifications](https://www.gs1.org/standards/barcodes-epcrfid-id-keys/gs1-general-specifications): identification and barcode standards; consult the applicable edition.
- [ISO 3691-4, Industrial trucks — Safety requirements and verification — Part 4](https://www.iso.org/standard/83545.html): driverless industrial trucks and their systems; verify the edition and jurisdictional application.
- [ISO 12100, Safety of machinery — General principles for design](https://www.iso.org/standard/51528.html): risk assessment and risk reduction framework.
- [OSHA 29 CFR 1910.147](https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147): US control of hazardous energy requirements, including applicable scope and exceptions.
- [NIST Cybersecurity Framework 2.0](https://www.nist.gov/cyberframework): cybersecurity risk-management framework; not equipment certification.
- [Kroger and Ocado customer fulfillment center announcement](https://www.thekrogerco.com/2021/04/14/kroger-and-ocado-announce-official-opening-of-customer-fulfillment-center-in-monroe-ohio/): company account of an automated grocery fulfillment deployment.
- [Amazon Robotics](https://www.aboutamazon.com/news/operations/amazon-robotics-robots-fulfillment-center): company description of robotics in fulfillment; not independent financial validation.

## 17. Links

- [Module 13 index](_index.md).
- [Warehouse flow system](01_warehouse-flow-system.md).
- [WMS data model](02_wms-data-model.md).
- [Productivity metrics](03_productivity-metrics.md).
- [Little's Law and queueing](04_littles-law-queueing.md).
- [Labor standards](05_labor-standards.md).
- [Slotting and ABC](06_slotting-abc.md).
- [Pick path and travel](08_pick-path-travel.md).
- [Module 13 survey and lab context](../../../course/13_warehousing_fulfillment_labor.md).
