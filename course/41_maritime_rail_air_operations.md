# Module 41 — Maritime, Rail, and Air Cargo Operations Engineering

> **Program:** Supply Chain Analytics, AI, Data Science, and Forecasting (2026 Edition)
> **Time:** 26 hours (12 lecture, 10 lab, 4 project)
> **Track fit:** Advanced elective; Logistics and Network Engineering tracks
> **Prerequisites:** Modules 11, 12, 20, 36
> **Tools:** Python, SimPy, OR-Tools, terminal and timetable data
> **Industry question:** "The ship arrives, the train path is fixed, and the freighter is full — what can operations actually change?"

---

## 1. Why industry cares

Module 36 teaches the commercial side of global freight: rates, Incoterms, landed cost, and trade policy. This module teaches the **operations engineering** underneath it: how a terminal, a rail network, and an air cargo operation physically move goods, and where the capacity really is.

Three features dominate:

- **Capacity is lumpy and shared.** A berth, a rail path, and an aircraft position are indivisible. One delayed vessel can displace several others.
- **The bottleneck moves.** Quay cranes, yard equipment, gate lanes, rail sidings, and truck docks take turns being the constraint.
- **Recovery is a scheduling problem.** A disruption creates a queue, and the queue must be sequenced, not merely waited out.

The analytics problem is **capacity, scheduling, and recovery under stochastic arrivals and fixed physical resources**.

## 2. Learning outcomes

By the end of this module, you can:

- Model vessel, train, and aircraft movements as resource-constrained systems.
- Calculate berth, quay crane, yard, rail, and air cargo capacity.
- Diagnose which terminal resource is the binding constraint.
- Build a berth allocation and crane scheduling model.
- Analyse rail timetable capacity, block trains, and yard shunting.
- Model air cargo build-up, ULD constraints, and belly versus freighter capacity.
- Simulate disruption and compare recovery policies.
- Evaluate equipment and automation investments.
- Account for safety, dangerous goods, and security operations.
- Communicate an operations plan in the language of a terminal or network controller.

## 3. Mental model

```text
   ARRIVAL ──> BERTH / PATH / STAND ──> HANDLING ──> STORAGE / BUILD ──> DEPARTURE
      │              │                     │               │                 │
   schedule       allocation            cranes           yard / ULD        slot
   variability    and queue             and labour       and staging       and cutoff
      │
   diversion, weather, congestion

   RAIL:  origin yard ──> network path ──> destination yard ──> customer
   AIR:   truck ──> cargo terminal ──> build-up ──> aircraft ──> breakdown ──> truck
```

Every node has a **capacity**, a **service time distribution**, and a **queue discipline**. The system performance is set by the node with the least slack, not by the average of all nodes.

### 3.1 The three modes share one structure

| Element | Maritime | Rail | Air |
|---|---|---|---|
| Unit of capacity | Berth, crane, slot | Path, siding, wagon | Aircraft position, ULD |
| Arrival pattern | Vessel schedule, weather | Timetable, freight demand | Flight schedule, belly capacity |
| Handling resource | Quay and yard equipment | Shunting locomotive, terminal | Build-up and breakdown staff |
| Storage | Container yard | Yard and siding | Cargo terminal |
| Cutoff | Vessel closing | Train departure | Flight cutoff |

## 4. Core concepts

### 4.1 Berth and quay capacity

A terminal's quay capacity is a function of cranes, productivity, and time:

\[
\text{Moves per day} = \text{cranes} \times \text{moves per crane-hour} \times \text{operating hours} \times \text{availability}
\]

For four cranes at 30 moves per crane-hour, 20 operating hours, and 85% availability:

\[
4 \times 30 \times 20 \times 0.85 = 2{,}040 \text{ moves per day}
\]

A 4,000-move vessel needs about two days of berth time. If two vessels arrive on the same day, the second waits, and the queue propagates.

### 4.2 Berth allocation

Berth allocation assigns vessels to quay positions and time windows. Constraints:

- Vessel length, draft, and beam.
- Crane availability and productivity.
- Tide and weather windows.
- Pilotage and towage.
- Priority contracts and liner schedules.

The objective is usually a weighted combination of waiting time, berth utilisation, and schedule reliability. It is a scheduling problem with release dates and sequence-dependent setup.

### 4.3 Yard and gate operations

The yard buffers the mismatch between vessel and landside flow. Key decisions:

- **Block and stack allocation:** where to put each container.
- **Rehandle minimisation:** avoid digging out boxes.
- **Dwell management:** free time, demurrage, and long-stay boxes.
- **Gate and rail evacuation:** truck appointments, rail loading.

Yard congestion is a **throughput** problem: if arrivals exceed evacuation, dwell rises until the yard is full and the terminal stops accepting.

### 4.4 Rail operations

Rail capacity has three layers:

| Layer | Decision | Constraint |
|---|---|---|
| Network | Path and timetable | Track capacity, signalling |
| Train | Block train or single wagon | Length, weight, gauge |
| Yard | Shunting and marshalling | Sidings, locomotive, labour |

Path capacity is not a single number: it depends on train length, speed, headway, and the mix of passenger and freight services. A timetable that looks feasible on average can fail at peak.

### 4.5 Air cargo operations

Air cargo moves in **ULDs** (unit load devices) in freighter aircraft or in the belly of passenger aircraft. Key concepts:

- **Build-up:** packing cargo into ULDs to a load plan.
- **Breakdown:** unloading and segregating at destination.
- **Cutoff:** the time after which cargo misses the flight.
- **Belly capacity:** available only after passenger baggage and mail.
- **Charter and freighter:** used when scheduled capacity is insufficient.

The capacity unit is the **aircraft position**, and the constraint is often the cargo terminal's build-up rate, not the aircraft.

### 4.6 Intermodal and hinterland

Ports and airports connect to the hinterland by rail, road, and barge. The hinterland link is often the real constraint:

- Rail terminal capacity and path availability.
- Road congestion and truck appointment systems.
- Barge water levels and terminal slots.

A port with free berth capacity but blocked hinterland is still congested.

### 4.7 Disruption and recovery

Disruptions — weather, equipment failure, labour action, congestion, or a blocked channel — create queues. Recovery policies:

| Policy | Effect | Risk |
|---|---|---|
| First-come, first-served | Fair | Slow recovery for priority cargo |
| Priority by contract | Protects key customers | Perceived unfairness |
| Largest ship first | Clears big queues | Starves small vessels |
| Cut and run | Skip a port | Contract breach and rehandling |
| Extra equipment | Adds capacity | Cost and availability |

The right policy depends on the objective and the contractual obligations. Simulation is the practical way to compare them.

### 4.8 Safety, dangerous goods, and security

Operations engineering must respect:

- **Dangerous goods** segregation on vessels, in yards, and on aircraft.
- **ISPS** port security and access control.
- **Cargo screening** for air cargo.
- **Weight and lashing** rules for containers and ULDs.
- **Fatigue and labour** rules for controllers and crews.

A plan that violates a segregation rule is not a plan.

### 4.9 Worked example — quay crane bottleneck

A terminal has four quay cranes, each capable of 32 moves per hour at full rate. Actual productivity is 26 moves per hour because of yard delays. The terminal works 18 hours per day at 90% availability.

\[
\text{Effective moves per day} = 4 \times 26 \times 18 \times 0.90 = 1{,}684.8
\]

If the terminal handles 1,800 moves per day, it is short by about 115 moves, or 6.4%. Options: add a crane shift, improve yard productivity, or reduce vessel calls. Adding a fifth crane at the same productivity gives 2,106 moves, which covers the demand but may simply move the bottleneck to the yard.

### 4.10 Worked example — rail path capacity

A freight corridor can run 24 paths per day in each direction. A block train carries 5,000 tonnes. The mine needs to move 90,000 tonnes per day.

\[
\text{Required trains} = 90{,}000 / 5{,}000 = 18
\]

With 24 paths, there is 25% headroom for maintenance and passenger services. If passenger services take six paths, freight capacity falls to 18, which exactly meets demand with no slack. A single disruption then causes a backlog that cannot be cleared within the day.

### 4.11 Reading a stated capacity figure honestly

Terminals, ports, and corridors publish capacity numbers, and those numbers are rarely comparable. Before a capacity figure enters a plan, establish:

- **What is counted.** Moves, TEU, tonnes, trains, or aircraft positions.
- **At what productivity.** Design rate, average rate, or best shift.
- **Over what hours.** 24/7, two shifts, or seasonal.
- **With what availability.** Gross or net of maintenance and weather.
- **Against what demand mix.** A terminal tuned for transhipment behaves differently under import-heavy volumes.

A terminal that quotes 3,000 moves per day at design productivity may deliver 2,100 at realistic yard conditions. The gap is not dishonesty; it is a different definition. The analyst's job is to reconstruct the definition, then apply the same one to the demand side.

The same discipline applies to rail paths. A "path" may be a nominal timetable slot that assumes a given train length, speed, and headway. Adding longer or slower trains consumes more than one path in practice, even when the timetable counts it as one. Capacity is only meaningful as a **matched pair**: a resource definition and a demand profile.

## 5. Data, standards, and workflow

### 5.1 Data sources

| Data | Source | Use |
|---|---|---|
| Vessel schedule and AIS | Carrier / AIS | Arrival and queue |
| Berth and crane log | Terminal operating system | Productivity |
| Yard inventory | TOS | Dwell and rehandle |
| Gate and rail | TOS / rail system | Evacuation |
| Train timetable | Infrastructure manager | Path capacity |
| Wagon and locomotive | Rail operator | Fleet and cycle |
| Air cargo manifest | Airline / forwarder | Build and cutoff |
| ULD and load plan | Ground handler | Build-up |
| Weather | External feed | Disruption |

### 5.2 Standards

- **IMO** conventions: SOLAS, MARPOL, IMDG, ISPS.
- **IATA** Dangerous Goods Regulations and cargo standards.
- **IATA ULD** regulations and load control.
- **UIC** and national rail standards; **ERTMS/ETCS** for signalling.
- **ISO 668** container dimensions and **ISO 1496** testing.
- **ISO 9001** and **ISO 45001** for terminal operations.
- **TSA** and equivalent air cargo security programmes.

### 5.3 Workflow

1. Plan the arrival and berth or path.
2. Allocate equipment and labour.
3. Handle and store the cargo.
4. Build or load the unit.
5. Depart on schedule.
6. Recover from disruption.
7. Review productivity and safety.

## 6. Industry case studies

### 6.1 Container terminal automation

**Context:** Terminals invested in automated stacking cranes and unmanned vehicles.
**Intervention:** Automation of yard and horizontal transport.
**Evidence:** Industry and academic studies (grade B/C); results vary by terminal.
**Outcome:** Automation raised throughput and safety in some terminals, but required high utilisation and stable volume to pay back.
**What transferred:** automation changes the bottleneck and the labour model; it is not a universal win.
**What did not:** specific productivity gains.

### 6.2 Rail freight corridor capacity

**Context:** A corridor mixed passenger and freight traffic.
**Intervention:** Timetable redesign and infrastructure upgrades.
**Evidence:** Infrastructure manager reports (grade B).
**Outcome:** Capacity improved where headway and signalling were the constraint; not where terminals were.
**What transferred:** network capacity is only useful if terminals can absorb the trains.
**What did not:** specific path counts.

### 6.3 Air cargo during a capacity shock

**Context:** Passenger flight reductions removed belly capacity.
**Intervention:** Freighter charters and cargo-only passenger flights.
**Evidence:** IATA and airline reporting (grade B/C).
**Outcome:** Rates rose and capacity shifted to freighters; some cargo moved to sea.
**What transferred:** air cargo capacity is a derivative of passenger networks and is highly cyclical.
**What did not:** specific rates.

### 6.4 A composite failure — the terminal that optimized the wrong resource

**Context:** A terminal added quay cranes to reduce vessel waiting.
**Intervention:** Crane investment without yard or gate investment.
**Evidence:** Composite teaching case.
**Outcome:** Vessel waiting fell briefly, then yard congestion rose and dwell increased.
**What transferred:** the bottleneck moves; capacity must be added where the queue forms.
**What did not:** specific volumes.

## 7. Labs

### 7.1 Lab 41.1 — Berth and crane capacity

Build a berth allocation model for a week of vessel calls. Measure waiting time and berth utilisation under two priority rules.

**Deliverable:** Allocation model and comparison.

### 7.2 Lab 41.2 — Terminal simulation

Simulate a container terminal with quay cranes, yard equipment, and gate lanes. Identify the binding constraint.

**Deliverable:** SimPy model and diagnosis.

### 7.3 Lab 41.3 — Rail path and yard

Model a rail corridor with passenger and freight paths and a marshalling yard. Test the effect of a terminal delay.

**Deliverable:** Capacity analysis and recovery plan.

### 7.4 Lab 41.4 — Air cargo build-up

Model cargo terminal build-up with ULD constraints and flight cutoffs. Find the build-up capacity needed for a peak.

**Deliverable:** Build-up model and staffing plan.

### 7.5 Lab 41.5 — Disruption recovery

Simulate a port closure and compare first-come, priority, and largest-first recovery policies.

**Deliverable:** Recovery analysis and policy recommendation.

## 8. Assignment / project

Design the operations plan for a terminal, corridor, or air cargo hub.

**Required components:**

1. Demand and arrival profile.
2. Capacity model by resource.
3. Bottleneck diagnosis.
4. Scheduling and allocation rules.
5. Storage and hinterland plan.
6. Disruption recovery plan.
7. Safety and security assessment.
8. Metrics and governance.
9. Risks, assumptions, and limits.

**Length:** 10–14 pages plus model.

## 9. Assessment rubric

| Criterion | Weight | Novice | Competent | Expert |
|---|---:|---|---|---|
| Capacity model | 25% | Rule of thumb | Resource calculation | Bottleneck and queueing |
| Scheduling | 20% | Lists calls | Allocation model | Priority and recovery |
| Modal operations | 20% | Describes mode | Mode-specific constraints | Cross-mode interface |
| Storage and hinterland | 15% | Mentions yard | Dwell and evacuation | System throughput |
| Safety and security | 10% | Names rules | Compliance plan | Operational integration |
| Communication | 10% | Descriptive | Clear recommendation | Decision-ready and honest |

## 10. Regulatory, ethical, and commercial lens

- **Safety:** dangerous goods, lashing, weight, and fatigue rules are mandatory.
- **Security:** ISPS, cargo screening, and access control apply to ports and aircraft.
- **Environment:** emissions from ships, ports, and air cargo are regulated and reported.
- **Labour:** automation changes jobs; consultation and transition are part of the plan.
- **Competition:** terminal and corridor access can raise competition issues.
- **Community:** port and airport operations affect noise, traffic, and local air quality.

## 11. Failure modes and red flags

| Failure | Early warning | Mechanism | Consequence | Mitigation |
|---|---|---|---|---|
| Crane-only investment | Yard dwell rising | Bottleneck moved | Congestion | System capacity plan |
| Average-based capacity | Peak queues | Variability ignored | Waiting and demurrage | Simulation |
| Timetable optimism | Trains late | Path conflict | Backlog | Path and yard analysis |
| Air cutoff ignored | Cargo left behind | Build-up too slow | Missed flight | Cutoff planning |
| Hinterland blindness | Port full | Rail/road constraint | Dwell | Hinterland capacity |
| Recovery by habit | Queue persists | No priority rule | Contract breach | Recovery policy |
| DG segregation error | Inspection finding | Procedural gap | Safety and fine | Training and checks |
| Automation overreach | Low utilisation | Fixed cost | Poor payback | Business case |

## 12. Interview and discussion questions

1. What is the difference between berth capacity and terminal capacity? *(Hint: quay, yard, gate.)*
2. How do you calculate quay crane capacity? *(Hint: cranes × rate × hours × availability.)*
3. Why does adding cranes sometimes make congestion worse? *(Hint: the bottleneck moves.)*
4. What is a rail path, and what limits it? *(Hint: headway, signalling, mix.)*
5. Why is air cargo capacity derivative? *(Hint: belly capacity of passenger flights.)*
6. What is a ULD, and why does it matter? *(Hint: unit of build-up and load.)*
7. How do you recover from a port closure? *(Hint: queue sequencing and priorities.)*
8. What is the hinterland constraint? *(Hint: rail, road, barge evacuation.)*
9. How do dangerous goods change terminal operations? *(Hint: segregation and storage.)*
10. When does terminal automation pay back? *(Hint: volume, utilisation, labour.)*
11. What metric would you use for terminal performance? *(Hint: moves per hour, vessel turnaround, dwell.)*
12. How do you validate a terminal simulation? *(Hint: face, structural, output against history.)*

## 13. Key terms

| Term | Definition |
|---|---|
| Berth | Quay position for a vessel |
| Quay crane | Crane that moves containers between ship and dock |
| Moves per hour | Crane productivity measure |
| Yard dwell | Time a container spends in the terminal |
| Rehandle | Moving a container to reach another |
| Rail path | Timetable slot for a train |
| Block train | Train carrying one commodity or customer |
| Shunting | Marshalling wagons in a yard |
| ULD | Unit load device for air cargo |
| Build-up | Packing cargo into a ULD or aircraft |
| Cutoff | Latest time cargo can be accepted |
| Belly capacity | Cargo space on a passenger aircraft |
| ISPS | International Ship and Port Facility Security Code |
| IMDG | International Maritime Dangerous Goods Code |
| TEU | Twenty-foot equivalent unit |

## 14. Sources and further reading

### Standards and regulation

- IMO, SOLAS, MARPOL, IMDG, ISPS. https://www.imo.org/
- IATA, Dangerous Goods Regulations and ULD standards. https://www.iata.org/
- ISO 668 and ISO 1496 container standards. https://www.iso.org/
- UIC and ERTMS/ETCS rail standards. https://uic.org/ https://www.era.europa.eu/
- TSA air cargo security. https://www.tsa.gov/

### Reports and official sources

- UNCTAD, Review of Maritime Transport. https://unctad.org/
- World Bank, Container Port Performance Index. https://www.worldbank.org/
- OECD/ITF, rail and freight transport outlook. https://www.itf-oecd.org/
- IATA, air cargo market analysis. https://www.iata.org/

### Peer-reviewed and professional

- Notteboom, T., Rodrigue, J.-P., container terminal operations.
- Bierlaire, M., rail scheduling and timetabling research.
- Cordeau, J.-F., et al., berth allocation and quay crane scheduling.
- Boysen, N., et al., air cargo and terminal operations research.

### Industry

- International Association of Ports and Harbors. https://www.iaphworldports.org/
- Airports Council International, cargo guidance. https://aci.aero/
- Terminal operator annual reports and productivity disclosures.

> **Next:** [Module 42](42_treasury_credit_financial_supply_chain.md) turns to the financial flows that fund the physical network.
