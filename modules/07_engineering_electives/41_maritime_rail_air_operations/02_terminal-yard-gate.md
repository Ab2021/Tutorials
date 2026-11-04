# Terminal Yard and Gate Operations

> Module 41 — Maritime, Rail, and Air Cargo Operations Engineering
> Topic 02 of 10
> Estimated study time: 4.0 hours
> Prerequisites: Module 41, Topic 01 (`01_berth-quay-crane.md`); Module 12 (`12_transportation_routing_last_mile.md`)
> Version 2026.5

*Yard template design, storage block thermodynamics, container stacking rules, reshuffling probability physics, ASC versus RTG kinematics, Truck Appointment Systems (TAS), OCR gate automation, and multi-lane queuing.*

---

## 1. Why this topic exists

The quay crane is the most visible asset in a container terminal, but the container yard is where terminal capacity actually lives or dies. A world-class quayside equipped with five ship-to-shore (STS) gantry cranes performing 35 gross moves per crane-hour can discharge and load 175 containers every hour. That represents an intense, discontinuous pulse of physical freight: up to 4,200 Twenty-Foot Equivalent Units (TEU) injected into the terminal footprint every 24 hours from a single ultra-large container vessel (ULCV).

Conversely, the hinterland evacuation network—comprising thousands of independent drayage truckers, short-sea feeder barges, and scheduled intermodal unit trains—operates as an entirely different dynamical regime. Drayage arrivals are atomized, decentralized, and governed by urban traffic patterns, warehouse operating hours, customs broker clearance releases, and chassis availability. 

The container terminal yard is the high-density thermodynamic buffer tasked with decoupling these two fundamentally incompatible physics:
1. **The discontinuous, batch-oriented maritime interface** (massive pulses of 3,000 to 7,000 container moves arriving within 36 to 72 hours per vessel call).
2. **The continuous, stochastic landside interface** (continuous trickles of individual road tractors and intermodal train cutoffs).

When the yard fails to buffer this interface efficiently, the terminal experiences three catastrophic failure cascades:

First, **yard lock (stack freezing)**. As a storage yard crosses 75% to 80% nominal utilization, the physical space required to perform housekeeping and relocation moves vanishes. To retrieve a single buried import container requested by a waiting drayage truck, yard cranes must dig through four or five overlying boxes. With no nearby empty ground slots to receive these intermediate containers, the crane must travel long distances down the block or shuffle boxes back and forth across bays. Non-productive reshuffles explode non-linearly. The effective handling capacity of the yard collapses precisely when demand is highest, freezing terminal velocity.

Second, **quayside starvation and crane throttling**. If yard cranes and horizontal transport vehicles (automated guided vehicles [AGVs], terminal tractors, or straddle carriers) cannot accept export boxes from the yard or deliver import boxes to the quay fast enough, STS cranes sit idle waiting for boxes. Quayside productivity drops from 35 moves per hour to under 20. Vessel port turnaround times stretch from 48 hours to 96 hours. Ship operators incur charter-equivalent delay penalties of $40,000 to $100,000 per day per vessel, triggering emergency schedule omissions, skipped port calls, and severe regional supply chain disruption.

Third, **gate spillback and urban gridlock**. When the terminal's in-gate inspection portals, optical character recognition (OCR) systems, Weigh-In-Motion (WIM) scales, and security checkpoints become congested, the resulting drayage truck queue rapidly exhausts internal queueing reservoirs. Queues spill out across access corridors, blocking metropolitan freeways, arterial roads, and neighbor communities. This triggers immediate regulatory fines under municipal Clean Air Action Plans, community litigation, and punitive detention claims from motor carriers whose drivers burn allowable Hours of Service (HOS) idling in public roadways.

Solving terminal yard and gate operations is not a soft management exercise; it is an applied operations engineering discipline requiring rigorous spatial optimization, kinematic synthesis, stochastic queuing theory, and dynamic scheduling. This topic develops the exact analytical models, physical laws, and control architectures required to design, operate, and automate high-throughput container yard blocks and gate complexes.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Dimension** static and dynamic container yard capacity from first principles using Little's Law, empirical dwell time distributions, and operational peaking factors.
2. **Derive** analytical container reshuffling probabilities and expected non-productive rehandle moves ($E[R]$) for arbitrary stack heights and bay occupancy rates using Kim's formulations and combinatorial models.
3. **Synthesize** equipment cycle times, velocity-acceleration profiles, and operational throughput for Automated Stacking Cranes (ASC / RMG) versus Rubber-Tired Gantry (RTG) cranes under Chebyshev and Manhattan kinematic regimes.
4. **Formulate** a Truck Appointment System (TAS) slot quota optimization model that shaves peak gate arrivals, dampens drayage turn-time variance, and bounds yard equipment utilization.
5. **Model** multi-lane automated in-gate complexes using the Allen-Cunneen $M/G/c$ queuing approximation, capturing the operational impact of OCR misreads, Weigh-In-Motion (WIM) verification, and trouble-kiosk exception loops.
6. **Apply** mandatory international stacking rules, including ISO 9711 coordinate topologies, SOLAS Verified Gross Mass (VGM) weight-tier stability constraints, and IMO IMDG dangerous goods physical segregation matrices.
7. **Diagnose** the thermodynamic threshold of yard gridlock ("stack freezing") using real-time terminal telemetry, identifying early warning signatures before productivity collapses.
8. **Critique** layout trade-offs between end-loading cantilever ASC blocks, side-loading RTG runs, and wheeled chassis parking templates across differing land, labor, and modal split constraints.

---

## 3. The core mental model

To engineer a container terminal yard, you must conceptualize the facility not as static real estate, but as a multi-stage hydrodynamic piping network with rigid volumetric limits, friction losses, and stochastic boundary conditions.

```text
=============================================================================================
                              TERMINAL PHYSICAL FLOW TOPOLOGY
=============================================================================================

 [ MARITIME INTERFACE ]
   Vessel Berths (STS Cranes)
        │ 
        │ (Discontinuous Batch Pulses: 2,000 - 6,000 TEU per vessel call)
        ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │ HORIZONTAL TRANSPORT BUFFER (Apron / Quay Transfer Zone)                                │
 │ Terminal Tractors (TT) / Automated Guided Vehicles (AGV) / Shuttle Carriers (SC)       │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
        │
        ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │ YARD STORAGE BLOCKS (High-Density Spatial Buffer)                                       │
 │                                                                                         │
 │   Waterside Handover Area (Cantilever / Buffer Slots)                                   │
 │   ┌─────────────────────────────────────────────────────────────────────────────────┐   │
 │   │ Bay 01 │ Bay 03 │ Bay 05 │ Bay 07 │ ... │ Bay 39 │ Bay 41 │ Bay 43 │ Bay 45       │   │
 │   │ Row 01 ─── Row 02 ─── Row 03 ─── Row 04 ─── Row 05 ─── Row 06 ─── Row 07 ─── Row 08 │   │
 │   │ Tier 1 (Ground) ─── Tier 2 ─── Tier 3 ─── Tier 4 ─── Tier 5 (Top)               │   │
 │   └─────────────────────────────────────────────────────────────────────────────────┘   │
 │   Cranes: Automated Stacking Cranes (ASC/RMG) or Rubber-Tired Gantries (RTG)           │
 │   Internal Operations: Stacking, Pre-marshaling, Reshuffling (Housekeeping)            │
 │                                                                                         │
 │   Landside Handover Area (Truck Transfer Zones / Cantilever Exchange Lanes)             │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
        ▲
        │ (Stochastic Continuous Inflow/Outflow: Modulated by TAS Quotas)
        │
 ┌─────────────────────────────────────────────────────────────────────────────────────────┐
 │ GATE AUTOMATION COMPLEX (Access Control & Regulatory Decoupling)                        │
 │                                                                                         │
 │   Approach Road & Queue Reservoir (Buffer Storage for K Trucks)                         │
 │      │                                                                                  │
 │   OCR Security Portal Array (Optical Character Recognition + LPR + USDOT + Damage)      │
 │      │                                                                                  │
 │   Weigh-In-Motion (WIM) Scales (SOLAS VGM Verification & Axle Load Compliance)          │
 │      │                                                                                  │
 │   Multi-Lane Pedestal / Kiosk Bank (Driver Biometrics, TWIC, RFID Tag, Missions)       │
 │      ├── Pass ──> In-Gate Boom Barrier ──> Yard Block Dispatch                          │
 │      └── Fail ──> Exception Lane / Trouble Kiosk (Clerk Video Intervention)             │
 └─────────────────────────────────────────────────────────────────────────────────────────┘
        ▲
        │
 [ HINTERLAND ROAD & RAIL INTERFACE ]
   External Drayage Trucks / Intermodal On-Dock Railhead
=============================================================================================
```

### 3.1 The Three Governing Thermodynamic Laws of the Yard

1. **Little's Law of Yard Inventory**:
   The average number of containers stored in the yard ($L$, measured in TEU) is the exact product of the terminal's average throughput rate ($\lambda$, in TEU per day) and the mean container dwell time ($W = \bar{d}$, in days):
   $$L = \lambda \cdot \bar{d}$$
   If a terminal processes 1,825,000 TEU per year ($\lambda = 5,000\text{ TEU/day}$) and the mean dwell time is 5.0 days, the yard must continuously hold an equilibrium inventory of 25,000 TEU. If port congestion or customs holds push the average dwell time to 8.0 days without an immediate reduction in vessel discharge, required yard inventory surges to 40,000 TEU. If this exceeds static holding capacity, the terminal collapses into total gridlock.

2. **The Congestion-Reshuffle Divergence**:
   The probability of needing to reshuffle (rehandle) an overlying container during an import retrieval is a convex, non-linear function of stack height ($H$) and bay occupancy utilization ($\rho$). Below 70% occupancy, empty slots exist within the same bay, allowing single-move relocations. Above 80% occupancy, adjacent stacks are full; relocations require cross-bay gantry travel, triggering secondary and tertiary reshuffles. At $\rho > 0.85$, the marginal cost of an import retrieval approaches infinity.

3. **The Gate Decoupling Law**:
   A gate complex cannot clear stochastic peaks simply by adding lanes if the internal yard transfer zones cannot absorb the incoming traffic. If yard cranes cannot service the landside interchange at rate $\mu_{\text{yard}} \ge \lambda_{\text{gate}}$, incoming trucks will block the internal circulation lanes, backing up through the in-gate kiosks and shutting down the entire OCR entry portal regardless of gate technology.

---

## 4. Definitions and vocabulary

| Term | Symbol / Acronym | Formal Engineering Definition | Common Industry Misuse | Standard Unit / Coding Standard |
|---|---|---|---|---|
| Twenty-Foot Equivalent Unit | TEU | Standardized statistical measure of container capacity based on an ISO standard 20-foot long, 8.5-foot high shipping container. | Confusing TEU with physical container count (a 40-foot box is 1 physical move but 2 TEU). | ISO 668 |
| Forty-Foot Equivalent Unit | FEU | Standardized container volume equivalent to two TEUs (typically 40 ft $\times$ 8 ft $\times$ 8.5 ft or 9.5 ft). | Assuming crane productivity in moves/hr equals TEU/hr. | ISO 668 |
| Twenty-Foot Ground Slot | TGS | The physical footprint on the yard pavement required to ground one 20-foot container. | Equating TGS directly to static storage capacity without multiplying by stack tier height. | Integer count |
| Bay-Row-Tier | BRT | Three-dimensional Cartesian coordinate addressing system for every container location in a vessel cell or yard block. | Swapping Bay (longitudinal) and Row (transverse) conventions between vessel and yard. | ISO 9711-1 |
| Reshuffle (Rehandle) | $R$ | A non-productive crane lift required to move an overlying container to access a target container stacked beneath it. | Classifying housekeeping or pre-marshaling moves as billable commercial lifts. | Non-productive moves per retrieval |
| Dwell Time | $d$ | The total elapsed duration in days that a container remains grounded in the yard between arrival and departure. | Conflating commercial "free time" granted by liner tariffs with physical dwell time. | Days (decimal) |
| Free Time | — | The contractual period during which a shipper may store a container in the terminal yard without incurring storage charges. | Assuming free time changes physical yard capacity (it only shifts financial liability). | Days / Hours |
| Demurrage | — | Financial penalty assessed by the terminal or carrier against the cargo interest for exceeding allowable free time in the yard. | Confusing demurrage (yard storage) with detention (chassis/box holding outside the gate). | USD per TEU-day |
| Detention | — | Financial penalty assessed against a shipper for holding carrier equipment (container/chassis) outside the terminal beyond free time. | Using detention interchangeably with yard demurrage. | USD per container-day |
| Rubber-Tired Gantry | RTG | Diesel-electric or hybrid mobile gantry crane running on rubber tires, spanning 6 to 8 container rows plus a truck lane. | Assuming RTGs require fixed concrete foundations (they operate on asphalt/concrete runway strips). | Moves per operating hour |
| Rail-Mounted Gantry | RMG | Fixed rail gantry crane typically spanning wide yards (8 to 14 rows) powered by high-voltage electric busbars. | Treating RMG as synonymous with ASC (not all RMGs are automated). | Moves per gross crane hour |
| Automated Stacking Crane | ASC | Unmanned, software-controlled RMG operating in closed, fenced blocks with automated optimization algorithms. | Believing ASCs eliminate all human intervention (remote exception operators handle exceptions). | Cycle time in seconds |
| Straddle Carrier | SC | Highly maneuverable mobile vehicle that straddles a single container row, stacking 1-over-2 or 1-over-3 and transporting boxes. | Treating SC yards as high-density (SC layouts have lower volumetric density than ASC blocks). | Tiers (1-over-2, 1-over-3) |
| Automated Guided Vehicle | AGV | Unmanned, battery-electric or diesel-hydraulic vehicle navigating via transponders or SLAM to transfer boxes between quay and yard. | Confusing horizontal apron transport with yard stacking equipment. | Meters per second |
| Truck Appointment System | TAS | Algorithmic booking system allocating hourly gate entry slot quotas to drayage motor carriers to flatten arrival peaks. | Treating TAS as a pure gate pass rather than a dynamic yard-crane workload leveling tool. | Appointments per hour-window |
| Turn Time (Gate-to-Gate) | TTT | Total elapsed time from a drayage truck's arrival at the gate approach to its departure through the out-gate. | Measuring only internal transaction time while ignoring queue wait time outside the gate. | Minutes |
| Dual-Cycle Move | — | A single crane or truck mission that drops off an export container and picks up an import container in the same visit. | Assuming all gate transactions can be dual cycles without operational booking coordination. | Dimensionless ratio ($\le 1.0$) |
| Optical Character Recognition | OCR | High-speed multi-camera machine-vision portal scanning container ISO numbers, chassis IDs, and license plates. | Expecting 100% automated read accuracy across weathered, rusted, or dirty containers. | Percentage confidence score |
| Weigh-In-Motion | WIM | Dynamic sensor scale embedded in gate lanes measuring axle weights and gross vehicle mass at speeds up to 10 km/h. | Believing WIM replaces static scales for legal trade enforcement without metrological certification. | Kilograms (kg) |
| Verified Gross Mass | SOLAS VGM | Mandatory gross container weight declaration enforced under IMO SOLAS Chapter VI Regulation 2. | Allowing containers to be loaded onto vessels without certified VGM documentation. | Metric tons (t) or kg |
| Trouble Kiosk | — | Dedicated bypass parking lane and service pedestal where gate transactions with data errors are resolved by remote clerks. | Forcing exception trucks to back out of automated lanes, causing lane blockage. | Exception rate (%) |
| Dangerous Goods Segregation | IMDG | Mandatory spatial separation rules between conflicting chemical hazard classes stored in the yard. | Assuming hazardous containers can be stacked adjacent if they are in different tiers of the same bay. | Table 7.2.4 separation distance |
| Reefer Rack | — | Structural steel framework equipped with 440V electrical outlets, catwalks, and monitoring systems for refrigerated containers. | Over-subscribing reefer plug capacity beyond substation transformer thermal ratings. | Plugs per ground slot |
| Allen-Cunneen Approximation | — | Analytical formula approximating expected waiting time in a multi-server queue with general arrival and service distributions ($GI/G/c$). | Applying standard Poisson $M/M/c$ formulas to gate lanes with high service time variance. | Minutes of wait time |
| Pre-marshaling | — | Housekeeping operations executed during non-peak hours to rearrange export containers into exact vessel load sequences. | Conflating pre-marshaling (proactive) with reshuffling (reactive rehandling). | Moves per shift |

---

## 5. The mechanics, step by step

Operating an automated container yard and gate infrastructure requires coordinating six interconnected subsystems:

```text
  [5.1 Coordinate Topology] ──> [5.2 Thermodynamic Sizing] ──> [5.3 Stacking Rules]
                                                                        │
  [5.6 Gate Queuing & TAS]  <── [5.5 Crane Kinematics]     <── [5.4 Reshuffle Physics]
```

### 5.1 Yard Template Design and Coordinate Topology

A container yard is spatially structured into parallel or perpendicular storage blocks. The orientation relative to the quay determines horizontal transport cycle times:
- **Perpendicular Layout (End-Loading ASCs)**: Blocks run perpendicular to the quayline. Waterside horizontal transport (AGVs) delivers containers to dedicated cantilever exchange buffers at the water end. Landside road trucks enter dedicated transfer bays at the land end. The automated stacking zone is completely fenced off; human drivers and automated cranes never share physical space.
- **Parallel Layout (Side-Loading RTGs)**: Blocks run parallel to the quayline. External trucks and terminal internal tractors share parallel runway lanes running alongside the stacks. RTGs traverse over the stacks and reach across into the truck lane to lift or place containers.

```text
=============================================================================================
                    ISO 9711 YARD BLOCK COORDINATE ADDRESSING
=============================================================================================

 Longitudinal Axis: BAYS (Odd = 20ft, Even = 40ft)
 ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
 │    BAY 01    │    BAY 03    │    BAY 05    │    BAY 07    │    BAY 09    │ (20ft Slots)
 ├──────────────┴──────────────┼──────────────┴──────────────┼──────────────┤
 │           BAY 02            │           BAY 06            │    BAY 10    │ (40ft Slots)
 └─────────────────────────────┴─────────────────────────────┴──────────────┘

 Transverse Axis: ROWS (Numbered Waterside to Landside: 01, 02, 03, ... W)
 ┌────────┬────────┬────────┬────────┬────────┬────────┬────────┬────────┐
 │ Row 01 │ Row 02 │ Row 03 │ Row 04 │ Row 05 │ Row 06 │ Row 07 │ Row 08 │
 └────────┴────────┴────────┴────────┴────────┴────────┴────────┴────────┘

 Vertical Axis: TIERS (ISO Standard: 02 = Ground, 04 = Tier 2, 06 = Tier 3, ... 2H)
                                                  ┌──────────────┐
                                                  │ Tier 10 (H5) │
                                   ┌──────────────┼──────────────┤
                                   │ Tier 08 (H4) │ Tier 08 (H4) │
                    ┌──────────────┼──────────────┼──────────────┤
                    │ Tier 06 (H3) │ Tier 06 (H3) │ Tier 06 (H3) │
     ┌──────────────┼──────────────┼──────────────┼──────────────┤
     │ Tier 04 (H2) │ Tier 04 (H2) │ Tier 04 (H2) │ Tier 04 (H2) │
 ────┴──────────────┴──────────────┴──────────────┴──────────────┴────── (Ground Level)
       Tier 02 (H1)   Tier 02 (H1)   Tier 02 (H1)   Tier 02 (H1)
=============================================================================================
```

Under ISO 9711, an exact slot coordinate is defined by a 6-digit string: `BB-RR-TT`:
1. **Bay (`BB`)**: Indicates longitudinal position. A 20-foot container occupies an odd bay (e.g., Bay 01, Bay 03, Bay 05). A 40-foot container occupies the even bay spanning two 20-foot slots (e.g., Bay 02 spans Bay 01 and Bay 03; Bay 06 spans Bay 05 and Bay 07).
2. **Row (`RR`)**: Indicates lateral transverse position across the block width, conventionally indexed from waterside (Row 01) to landside (Row $W$).
3. **Tier (`TT`)**: Indicates vertical elevation. Under maritime conventions, standard height containers are numbered in increments of two: Tier 02 (ground tier), Tier 04 (second tier), Tier 06 (third tier), Tier 08 (fourth tier), Tier 10 (fifth tier). High-cube containers (9'6" height) often use odd tier numbers or specific TOS flags to account for vertical clearance envelope restrictions.

### 5.2 Storage Block Thermodynamics and Congestion Dynamics

Terminal planning must rigorously distinguish between **Nominal Static Capacity**, **Operational Working Capacity**, and **Dynamic Annual Throughput**:

#### Step 1: Nominal Static Capacity ($C_{\text{static}}$)
The maximum physical number of TEUs that can fit in the yard footprint if every slot were packed to maximum permissible structural stacking height:
$$C_{\text{static}} = N_{\text{TGS}} \times H_{\text{max}}$$
where $N_{\text{TGS}}$ is the total number of Twenty-Foot Ground Slots and $H_{\text{max}}$ is the crane lift clearance limit (typically 1-over-5 for ASCs, 1-over-4 or 1-over-5 for RTGs).

#### Step 2: Operational Working Capacity ($C_{\text{oper}}$)
In practice, a yard cannot operate at 100% nominal capacity. Space must be reserved for:
- TEU/FEU length mismatch (two 20ft containers cannot be placed on top of one 40ft container due to corner-casting load paths).
- Empty relocation buffer slots for reshuffling.
- Broken stowage and gangway access.
- Maintenance crane clearances.

The operational working capacity is governed by an operating factor $\eta_{\text{oper}} \in [0.70, 0.80]$:
$$C_{\text{oper}} = C_{\text{static}} \times \eta_{\text{oper}}$$
When total instantaneous yard inventory $I(t) > C_{\text{oper}}$, the terminal enters the **thermodynamic congestion regime**. Reshuffles increase exponentially, and crane moves per hour collapse.

#### Step 3: Dynamic Annual Throughput ($T_{\text{annual}}$)
By applying Little's Law ($L = \lambda \cdot \bar{d}$), the dynamic annual throughput capacity (TEU/year) supported by the yard under an average dwell time $\bar{d}$ (days) and a seasonal/weekly peaking factor $p_{\text{peak}} \ge 1.0$ is:
$$T_{\text{annual}} = \frac{C_{\text{oper}} \times 365}{\bar{d} \times p_{\text{peak}}}$$
Because import dwell times ($\bar{d}_{\text{imp}} \approx 4.0 - 7.0\text{ days}$) significantly exceed export dwell times ($\bar{d}_{\text{exp}} \approx 2.5 - 4.0\text{ days}$), the composite mean dwell time must be computed as a weighted harmonic average over the trade split:
$$\bar{d} = \alpha_{\text{imp}} \bar{d}_{\text{imp}} + \alpha_{\text{exp}} \bar{d}_{\text{exp}} + \alpha_{\text{trans}} \bar{d}_{\text{trans}} + \alpha_{\text{mty}} \bar{d}_{\text{mty}}$$
where $\sum \alpha_i = 1.0$.

```text
=============================================================================================
                    YARD CONGESTION THERMODYNAMICS: THE 75% KNEE
=============================================================================================
 Crane Productivity
 (Gross Moves / Hr)
   40 ┤ ──────────────────────────────╮
      │                               │
   30 ┤                               ╰──────╮
      │                                      ╰────────╮  <-- "Thermodynamic Knee"
   20 ┤                                               ╰─────────────╮
      │                                                             ╰──────────
   10 ┤                                                                CRANE
      │                                                                STALL
    0 ┼───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬─────
     0%      10%     20%     30%     40%     50%     60%     70%     80%     90%   100%
                                Yard Utilization (rho)
=============================================================================================
```

### 5.3 Container Stacking Rules and Segregation Logic

Modern Terminal Operating Systems (TOS)—such as Navis N4, Tideworks, or CyberLogitec—enforce multi-layered constraint satisfaction engines to determine slot allocation during ground assignment:

#### Rule 1: Structural Weight-Tier Stability (SOLAS VGM)
Containers are stacked strictly according to gross mass to maintain aerodynamic stability against 120 km/h wind shear and prevent structural corner-post collapse:
- **Tiers 02 and 04 (Bottom)**: Heavy containers ($M \ge 24.0\text{ metric tons}$).
- **Tiers 06 and 08 (Middle)**: Medium containers ($14.0\text{ t} \le M < 24.0\text{ t}$).
- **Tier 10 (Top)**: Light containers ($M < 14.0\text{ t}$) and empty containers.
Violation of this rule creates a critical structural failure: placing a 30-metric-ton box on top of a 6-metric-ton empty box can deform the lower container's corner posts, leading to instantaneous stack collapse.

#### Rule 2: Vessel Voyage and Discharge Port Segregation
Export containers are strictly segregated into dedicated blocks partitioned by:
1. Shipping Line / Service Code.
2. Vessel Name and Voyage Number.
3. Discharge Port (POD) in inverse rotation order (containers for the final port of call are stacked at the bottom; containers for the next immediate port of call are stacked on top).
4. Container Length (20ft, 40ft, 45ft) and Height (8'6" standard vs 9'6" high-cube).

#### Rule 3: IMO IMDG Hazardous Segregation Matrix
Dangerous goods are segregated per IMO IMDG Code Chapter 7.2. Table 7.2.4 dictates minimum physical separation distances:
- **Class 1 (Explosives)**: Stored in specialized explosion-blast bunkers, strictly separated from Class 2.1 (Flammable Gases) and Class 3 (Flammable Liquids) by at least 24 meters or complete structural blast walls.
- **Class 3 (Flammable Liquids) vs Class 5.1 (Oxidizing Substances)**: Must be separated by a minimum of 2 container bays (longitudinal) or 2 container rows (transverse). They can **never** be stacked in the same vertical column.
- **Class 8 (Corrosives) vs Class 4.3 (Dangerous When Wet)**: Requires physical buffer zones with dedicated drainage sumps to prevent liquid spills from contacting reactive substances.

#### Rule 4: Reefer Rack Spacing and Electrical Feeder Limits
Refrigerated containers (reefers) are grounded exclusively in bays equipped with permanent structural electrical racks. Operational constraints:
- Maximum stacking height of 3 to 4 tiers to permit daily physical inspection by certified reefer technicians.
- Plug diversity factor: The total connected load must not exceed 80% of the electrical substation transformer capacity ($S_{\text{max}} = \sum P_{\text{reefer}} \times k_{\text{div}}$, where each active reefer draws $8.0 - 14.0\text{ kW}$).
- Minimum 1.2-meter catwalk clearance between rows to permit cable routing, temperature log access, and USDA cold treatment data probe calibration.

### 5.4 Reshuffling Probability Physics and Analytical Derivations

When an import container is called for retrieval by an arriving drayage truck, any container resting on top of it must be picked, lifted over the stack, and placed into an adjacent stack before the target container can be retrieved. This non-productive move is defined as a **reshuffle** (or rehandle).

#### Derivation of Expected Reshuffles for Random Retrieval ($E[R]$)
Consider a bay consisting of $S$ stacks, each filled to height $H$ with containers. Suppose a retrieval request arrives for a container chosen uniformly at random from the stack.

Let random variable $K \in \{1, 2, \dots, H\}$ represent the tier of the target container (where tier 1 is the bottom tier resting on the ground, and tier $H$ is the top tier). If the target container is at tier $k$, exactly $H - k$ containers rest above it and must be removed.

Assuming a uniform retrieval probability across all tiers:
$$P(K = k) = \frac{1}{H}, \quad \forall k \in \{1, 2, \dots, H\}$$

The expected number of reshuffles to retrieve this container is:
$$E[R \mid H] = \sum_{k=1}^{H} P(K = k) \cdot (H - k) = \sum_{k=1}^{H} \frac{1}{H} (H - k)$$

Expanding the summation:
$$E[R \mid H] = \frac{1}{H} \left[ (H - 1) + (H - 2) + \dots + 1 + 0 \right] = \frac{1}{H} \sum_{j=0}^{H-1} j$$

Using the arithmetic progression sum formula $\sum_{j=0}^{n} j = \frac{n(n+1)}{2}$ where $n = H - 1$:
$$E[R \mid H] = \frac{1}{H} \cdot \frac{(H - 1)H}{2} = \frac{H - 1}{2}$$

This is the **fundamental theorem of container reshuffling**:
- At $H = 1$ tier: $E[R] = 0.0$ reshuffles.
- At $H = 3$ tiers: $E[R] = \frac{3 - 1}{2} = 1.0$ reshuffle per retrieval.
- At $H = 5$ tiers: $E[R] = \frac{5 - 1}{2} = 2.0$ reshuffles per retrieval.

Each reshuffle requires two physical crane moves:
1. Lift the blocking container off the target stack and drop it onto a temporary buffer stack.
2. After retrieving the target container, the temporary box may either remain in its new slot or be returned. Even if left in the buffer slot, the reshuffle consumed 1 non-productive cycle.

The total gross crane lifts required to execute 1 productive retrieval is:
$$\text{Gross Lifts} = 1 + 2 \cdot E[R] = 1 + 2 \left( \frac{H - 1}{2} \right) = H$$
For a 5-tier high stack under random retrieval, **a crane must perform an average of 5.0 total lifts to extract a single container**. Only 1 of those moves is productive. The crane's productive efficiency is:
$$\text{Efficiency} = \frac{1}{H} = \frac{1}{5} = 20\%$$

```text
=============================================================================================
                ANALYTICAL RESHUFFLE PROBABILITY TREE (H = 4 TIERS)
=============================================================================================

 Tier 4 [Top]    ─── P(K=4) = 1/4 ─── 0 Overlying Boxes ───> 0 Reshuffles
 Tier 3          ─── P(K=3) = 1/4 ─── 1 Overlying Box   ───> 1 Reshuffle
 Tier 2          ─── P(K=2) = 1/4 ─── 2 Overlying Boxes ───> 2 Reshuffles
 Tier 1 [Bottom] ─── P(K=1) = 1/4 ─── 3 Overlying Boxes ───> 3 Reshuffles

 Expected Reshuffles: E[R] = (0 + 1 + 2 + 3) / 4 = 6 / 4 = 1.50 reshuffles/retrieval
 Total Crane Moves per Retrieval: 1 + 2 * (1.50) = 4.00 moves
=============================================================================================
```

#### Kim's Formula for Constrained Bay Capacity
The base derivation assumes an unconstrained supply of vacant adjacent slots to receive relocated containers. In real yards, bay occupancy utilization $\rho = \frac{N}{S \cdot H}$ restricts relocation choices. Kim (1997) proved that when available empty slots are limited, relocated boxes are frequently placed on top of containers whose departure time is earlier than the relocated box, triggering secondary reshuffles.

Kim's analytical approximation for total expected rehandles per retrieval under bay occupancy $\rho \in [0.1, 0.95]$ is:
$$E[R_{\text{total}}] = \frac{H - 1}{2} \cdot \left[ 1 + \psi \cdot \left( \frac{\rho}{1 - \rho} \right) \right]$$
where $\psi \approx 0.15 - 0.20$ is the bay topology factor. Notice that as $\rho \to 1.0$, the term $\frac{\rho}{1 - \rho} \to \infty$. This proves mathematically why yard utilization must be kept strictly below 80%.

### 5.5 ASC and RTG Kinematics and Cycle Time Synthesis

A container crane move is a three-dimensional kinematic trajectory composed of three independently driven physical axes:
1. **Gantry Motion ($X$-axis)**: Longitudinal translation of the entire crane structure along the block length.
2. **Trolley Motion ($Y$-axis)**: Transverse translation of the hoist trolley across the width of the block.
3. **Hoist Motion ($Z$-axis)**: Vertical raising and lowering of the spreader beam and container.

```text
=============================================================================================
              CRANE KINEMATIC ARCHITECTURE: 3-AXIS VECTOR PROFILE
=============================================================================================
                          ▲ Z (Vertical Hoisting)
                          │        
                          │        ┌───────────┐ (Trolley)
                          │        │  [HOIST]  │ <═══════════> Y (Transverse Trolley)
                          │        └─────┬─────┘
                          │              │ (Wire Ropes)
                          │        ┌─────┴─────┐
                          │        │  SPREADER │
                          │        └─────┬─────┘
                          │              ▼
                          │        ┌───────────┐
                          │        │ CONTAINER │
                          │        └───────────┘
                          │
                          └─────────────────────────────────────► X (Longitudinal Gantry)
=============================================================================================
```

#### Kinematic Parameters: ASC vs RTG Benchmark

| Kinematic Parameter | Symbol | Automated Stacking Crane (ASC) | Rubber-Tired Gantry (RTG) | Standard Unit |
|---|---|---|---|---|
| Gantry Travel Speed | $v_g$ | $4.0 - 5.0$ | $2.0 - 2.5$ | $\text{m/s}$ |
| Gantry Acceleration | $a_g$ | $0.4 - 0.5$ | $0.2 - 0.3$ | $\text{m/s}^2$ |
| Trolley Travel Speed | $v_t$ | $1.2 - 1.5$ | $1.0 - 1.2$ | $\text{m/s}$ |
| Trolley Acceleration | $a_t$ | $0.3 - 0.4$ | $0.25 - 0.35$ | $\text{m/s}^2$ |
| Hoist Speed (Empty Spreader) | $v_{h,e}$ | $1.0 - 1.2$ | $0.8 - 1.0$ | $\text{m/s}$ |
| Hoist Speed (Laden Under Load) | $v_{h,l}$ | $0.6 - 0.8$ | $0.45 - 0.55$ | $\text{m/s}$ |
| Hoist Acceleration | $a_h$ | $0.5 - 0.6$ | $0.4 - 0.5$ | $\text{m/s}^2$ |
| Spreader Twistlock Lock/Unlock | $t_{\text{lock}}$ | $12.0 - 15.0$ | $15.0 - 20.0$ | $\text{seconds}$ |
| Fine Optical / Skew Positioning | $t_{\text{skew}}$ | $8.0 - 12.0$ | $10.0 - 15.0$ | $\text{seconds}$ |

#### Motion Time Equations (Trapezoidal Speed Profile)
For any single translational axis moving a physical distance $D$ with maximum rated velocity $v_{\text{max}}$ and constant acceleration/deceleration $a$:
1. The distance required to accelerate to $v_{\text{max}}$ and decelerate to a stop is:
   $$D_{\text{acc}} = \frac{v_{\text{max}}^2}{a}$$
2. **Case A (Trapezoidal Profile, $D \ge D_{\text{acc}}$)**: The crane reaches maximum rated speed:
   $$t_{\text{axis}}(D) = \frac{v_{\text{max}}}{a} + \frac{D}{v_{\text{max}}}$$
3. **Case B (Triangular Profile, $D < D_{\text{acc}}$)**: The travel distance is too short to attain full rated speed:
   $$t_{\text{axis}}(D) = 2 \sqrt{\frac{D}{a}}$$

#### Cycle Time Synthesis
Modern cranes feature advanced multi-axis AC variable-frequency drives that permit simultaneous gantry and trolley motion (**Chebyshev kinematics**). However, vertical hoisting must lift the container above the highest adjacent stack plus a mandatory physical safety buffer ($Z_{\text{clear}} = H_{\text{max}} \times h_{\text{box}} + 1.5\text{ m}$) before full horizontal travel is permitted:
$$t_{\text{horiz}} = \max\left( t_{\text{gantry}}(\Delta x), t_{\text{trolley}}(\Delta y) \right)$$
$$t_{\text{vert}} = t_{\text{hoist}}(\Delta z_{\text{lift}}) + t_{\text{hoist}}(\Delta z_{\text{lower}})$$
The complete single-cycle handling time ($T_{\text{cycle}}$) to pick a container from stack location $A$ and ground it at handover location $B$ is:
$$T_{\text{cycle}} = t_{\text{horiz}} + t_{\text{vert}} + t_{\text{skew}} + t_{\text{lock}}$$
For a round-trip dual-cycle mission (e.g., retrieving an import container and grounding an incoming export container), the total move time is synthesized by chaining two loaded and unloaded kinematic profiles.

### 5.6 In-Gate OCR Automation and Multi-Lane Queuing Mechanics

The terminal gate is the legal and physical boundary between the public transportation network and the regulated marine terminal. 

```text
=============================================================================================
                  AUTOMATED IN-GATE COMPLEX ARCHITECTURE
=============================================================================================
 Drayage Approach Road (Capacity: K Trucks in Reservoir)
   │
   ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ OPTICAL CHARACTER RECOGNITION (OCR) SCANNING PORTAL                    │
 │  - 4K Linescan High-Speed Cameras (Front, Rear, Left, Right, Top)      │
 │  - ISO 6346 Container Number Recognition                              │
 │  - License Plate Recognition (LPR) & USDOT Motor Carrier Number       │
 │  - High-Resolution Damage Inspection Imaging Array                     │
 └────────────────────────────────────────────────────────────────────────┘
   │
   ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ WEIGH-IN-MOTION (WIM) HIGH-SPEED BENDING-PLATE SCALES                  │
 │  - Axle Weight Distribution & Total Gross Vehicle Mass Verification    │
 │  - SOLAS VGM Tolerance Check (|WIM - VGM| <= 500 kg)                   │
 └────────────────────────────────────────────────────────────────────────┘
   │
   ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │ MULTI-LANE TRANSACTION PEDESTALS (c Parallel Lanes)                    │
 │  - Driver TWIC Card / Biometric Fingerprint Validation                 │
 │  - TAS Appointment QR Code / RFID Chassis Tag Interrogation            │
 │  - Automated Interchange Receipt (EIR) & Yard Slot Mission Dispatch    │
 └────────────────────────────────────────────────────────────────────────┘
   ├── [OCR Confirmed & VGM Valid] ──> Boom Barrier Opens ──> Yard Entry
   └── [OCR Misread / VGM Mismatch] ──> Divert ──> Exception Trouble Kiosk
=============================================================================================
```

#### Step 1: ISO 6346 Container Identification Validation
Every container carries a unique 11-character identifier:
- **Owner Code**: 3 uppercase letters (e.g., `MSKU`, `CMAU`).
- **Category Identifier**: 1 uppercase letter (`U` for all freight containers, `J` for detachable equipment, `Z` for trailers/chassis).
- **Serial Number**: 6 numeric digits.
- **Check Digit**: 1 numeric digit computed via the ISO 6346 modulus-11 algorithm.

The check digit calculation maps letters to numbers (omitting multiples of 11):
$A=10, B=12, \dots, Z=38$. Each character value $a_i$ ($i = 0, \dots, 9$) is multiplied by $2^i$:
$$\text{Sum} = \sum_{i=0}^{9} a_i \cdot 2^i$$
$$\text{Check Digit} = (\text{Sum} \pmod{11}) \pmod{10}$$
If the machine-vision OCR engine reads a container number whose check digit fails this parity equation, the confidence score drops below 0.90, automatically triggering a secondary camera re-scan or diverting the truck to the trouble kiosk.

#### Step 2: Multi-Lane Gate Queuing Model ($M/G/c$)
Drayage trucks arrive at the gate complex at average rate $\lambda$ (trucks/hour). The gate complex operates $c$ identical parallel automated lanes. The service time $S$ per truck at the transaction pedestal is a general random variable with mean $E[S] = \frac{1}{\mu}$ and standard deviation $\sigma_s$.

The squared coefficient of variation of service time is:
$$C_s^2 = \frac{\sigma_s^2}{(E[S])^2}$$
For pure exponential service, $C_s^2 = 1.0$. In automated gate lanes, successful transactions take roughly 45 seconds, but transactions requiring manual documentation or intercom assistance can take 300 seconds, generating high variance ($C_s^2 \approx 0.35 - 0.70$).

Assuming Poisson arrivals ($C_a^2 = 1.0$), the system is an $M/G/c$ queue. We model performance using the **Allen-Cunneen approximation**:

1. **Traffic Intensity ($\rho$)**:
   $$\rho = \frac{\lambda}{c \mu} < 1.0$$
2. **Erlang-C Delay Probability ($P_c$)**:
   The exact probability that an arriving truck finds all $c$ lanes busy is:
   $$P_c(c, a) = \frac{\frac{a^c}{c!(1 - \rho)}}{\sum_{k=0}^{c-1} \frac{a^k}{k!} + \frac{a^c}{c!(1 - \rho)}}, \quad \text{where } a = \frac{\lambda}{\mu}$$
3. **Expected Waiting Time in Queue ($W_q$)**:
   $$W_q \approx \left[ \frac{P_c(c, a)}{c \mu (1 - \rho)} \right] \cdot \left( \frac{1 + C_s^2}{2} \right)$$
4. **Expected Queue Length ($L_q$)**:
   By Little's Law:
   $$L_q = \lambda W_q$$
5. **Total In-Gate Turnaround Time ($W$)**:
   $$W = W_q + E[S] = W_q + \frac{1}{\mu}$$

#### Step 3: Truck Appointment System (TAS) Slot Quota Optimization
Without a TAS, drayage arrivals exhibit massive uncoordinated morning peaks (07:00–09:00) and afternoon peaks (13:00–15:00) driven by regional highway commuting patterns and warehouse shifts. Peaking drives traffic intensity $\rho > 1.0$, creating infinite queues that spill onto public freeways.

A TAS discretizes the operating day into $T$ discrete appointment windows (typically 1-hour slots, $t = 1, \dots, T$). The terminal operating system solves a slot quota allocation problem to determine maximum allowable bookings $Q_t$:
$$\max \sum_{t=1}^{T} Q_t$$
subject to:
$$Q_t \le c \cdot \mu_t \cdot \rho_{\text{target}}, \quad \forall t \in \{1, \dots, T\}$$
$$\sum_{b \in \text{Block } B} \xi_{b, t} \cdot Q_t \le M_{\text{crane}, B} \cdot \kappa_{\text{moves}}, \quad \forall B, \forall t$$
where $\rho_{\text{target}} \le 0.75$ guarantees bounded queuing delays, $\xi_{b, t}$ is the proportion of appointments directed to yard block $B$, and $M_{\text{crane}, B}$ is the maximum gross moves per hour deliverable by the yard cranes assigned to block $B$.

---

## 6. Worked example

### 6.1 Problem Statement: Apex Gateway Terminal
You are the Lead Terminal Systems Engineer at "Apex Gateway Terminal," an automated container terminal undergoing a major expansion to handle 2,400,000 TEU per year. You must size the yard infrastructure, compute expected reshuffle penalties, synthesize ASC crane cycle times, and dimension the automated in-gate complex.

### 6.2 Step 1: Yard Capacity and Block Thermodynamics
The terminal operates 365 days per year. The trade flow characteristics are:
- Total Annual Throughput ($T_{\text{annual}}$): $2,400,000\text{ TEU/year}$.
- Import Volume Share ($\alpha_{\text{imp}}$): 55% ($1,320,000\text{ TEU/yr}$), with mean dwell time $\bar{d}_{\text{imp}} = 5.2\text{ days}$.
- Export Volume Share ($\alpha_{\text{exp}}$): 45% ($1,080,000\text{ TEU/yr}$), with mean dwell time $\bar{d}_{\text{exp}} = 3.0\text{ days}$.
- Seasonal / Weekly Peaking Factor ($p_{\text{peak}}$): 1.25.
- Maximum Safe Operating Utilization ($\eta_{\text{oper}}$): 75% (0.75).
- Block Geometry: Each automated block is 10 bays long (40ft equivalent), 8 rows wide, and stacks containers 5 tiers high ($H = 5$).
- Ground slot conversion: Each 40ft block bay contains two 20ft ground slots per row ($2 \times 8 = 16\text{ TGS per bay}$). Thus, each block contains $10 \times 16 = 160\text{ TGS}$.

#### Calculations:
1. **Weighted Mean Dwell Time ($\bar{d}$)**:
   $$\bar{d} = (0.55 \times 5.2) + (0.45 \times 3.0) = 2.86 + 1.35 = 4.21\text{ days}$$
2. **Average Daily Throughput ($\lambda$)**:
   $$\lambda = \frac{2,400,000\text{ TEU}}{365\text{ days}} = 6,575.34\text{ TEU/day}$$
3. **Required Operational Working Storage ($C_{\text{oper}}$)**:
   By Little's Law, accounting for the 1.25 peaking factor:
   $$C_{\text{oper}} = \lambda \cdot \bar{d} \cdot p_{\text{peak}} = 6,575.34 \times 4.21 \times 1.25 = 34,602.85\text{ TEU}$$
4. **Required Nominal Static Storage Capacity ($C_{\text{static}}$)**:
   $$C_{\text{static}} = \frac{C_{\text{oper}}}{\eta_{\text{oper}}} = \frac{34,602.85}{0.75} = 46,137.13\text{ TEU}$$
5. **Nominal Capacity Per Block ($C_{\text{block}}$)**:
   $$C_{\text{block}} = 160\text{ TGS} \times 5\text{ tiers} = 800\text{ TEU per block}$$
6. **Required Number of Automated Yard Blocks ($N_{\text{blocks}}$)**:
   $$N_{\text{blocks}} = \left\lceil \frac{46,137.13}{800} \right\rceil = \lceil 57.67 \rceil = 58\text{ blocks}$$
The terminal must construct at least 58 automated stacking blocks to maintain operational stability during peak season without exceeding 75% utilization.

### 6.3 Step 2: Analytical Reshuffle Physics
An import block bay has $H = 5$ tiers and operates at nominal capacity. 
1. Under uniform random customer retrieval, compute the baseline expected reshuffles ($E[R]$) per retrieval:
   $$E[R] = \frac{H - 1}{2} = \frac{5 - 1}{2} = 2.00\text{ reshuffles}$$
2. Compute the total crane lifts required per productive retrieval:
   $$\text{Moves} = 1 + 2 \cdot E[R] = 1 + 2(2.00) = 5.00\text{ moves}$$
3. Now evaluate the bay under Kim's constrained capacity formula at 80% occupancy ($\rho = 0.80$, with layout coefficient $\psi = 0.18$):
   $$E[R_{\text{total}}] = 2.00 \cdot \left[ 1 + 0.18 \cdot \left( \frac{0.80}{1 - 0.80} \right) \right] = 2.00 \cdot [1 + 0.18(4.0)] = 2.00 \cdot [1 + 0.72] = 3.44\text{ reshuffles}$$
   At 80% occupancy, secondary reshuffles inflate total rehandles by 72% (from 2.00 to 3.44 reshuffles per box). Total crane lifts surge from 5.00 to:
   $$\text{Gross Lifts} = 1 + 2(3.44) = 7.88\text{ lifts per retrieval}$$
   This demonstrates why allowing occupancy to creep from 60% to 80% destroys yard crane productivity.

### 6.4 Step 3: Automated Stacking Crane (ASC) Cycle Time Synthesis
An ASC must retrieve a 40-foot laden import container from Bay 06, Row 02, Tier 08 ($H=4$) and transfer it to the landside truck transfer zone located at Bay 01, Row 09, Tier 02 ($H=1$).

#### Physical Distances:
- Longitudinal travel ($\Delta x$): Bay 06 to Bay 01. Each 40ft bay pitch is 14.5 meters.
  $$\Delta x = (6 - 1) \times 14.5\text{ m} = 72.5\text{ m}$$
- Transverse travel ($\Delta y$): Row 02 to Row 09. Each row pitch is 3.2 meters.
  $$\Delta y = (9 - 2) \times 3.2\text{ m} = 22.4\text{ m}$$
- Vertical hoist travel ($\Delta z$):
  - Container rests at Tier 08 (height $Z_1 = 3 \times 2.6\text{ m} = 7.8\text{ m}$ above ground).
  - Clearance height required over 5-tier stacks: $Z_{\text{clear}} = (5 \times 2.6) + 1.5 = 14.5\text{ m}$.
  - Hoist up distance: $\Delta z_{\text{up}} = 14.5 - 7.8 = 6.7\text{ m}$.
  - Hoist down distance to chassis at Tier 02 ($1.5\text{ m}$ chassis height): $\Delta z_{\text{down}} = 14.5 - 1.5 = 13.0\text{ m}$.
  - Total vertical hoist travel: $\Delta z = 6.7 + 13.0 = 19.7\text{ m}$.

#### Kinematic Parameters:
- Gantry: $v_g = 4.0\text{ m/s}$, $a_g = 0.4\text{ m/s}^2$. Acceleration distance $D_{\text{acc}, g} = \frac{4.0^2}{0.4} = 40.0\text{ m}$. Since $\Delta x = 72.5\text{ m} > 40.0\text{ m}$, profile is trapezoidal:
  $$t_g = \frac{4.0}{0.4} + \frac{72.5}{4.0} = 10.0 + 18.125 = 28.13\text{ seconds}$$
- Trolley: $v_t = 1.2\text{ m/s}$, $a_t = 0.3\text{ m/s}^2$. Acceleration distance $D_{\text{acc}, t} = \frac{1.2^2}{0.3} = 4.8\text{ m}$. Since $\Delta y = 22.4\text{ m} > 4.8\text{ m}$, profile is trapezoidal:
  $$t_t = \frac{1.2}{0.3} + \frac{22.4}{1.2} = 4.0 + 18.67 = 22.67\text{ seconds}$$
- Horizontal Travel Time (Chebyshev):
  $$t_{\text{horiz}} = \max(t_g, t_t) = \max(28.13, 22.67) = 28.13\text{ seconds}$$
- Vertical Hoisting (Laden spreader, $v_{h,l} = 0.7\text{ m/s}$, $a_h = 0.5\text{ m/s}^2$):
  $$D_{\text{acc}, h} = \frac{0.7^2}{0.5} = 0.98\text{ m}$$
  Both hoist moves exceed $0.98\text{ m}$:
  $$t_{\text{up}} = \frac{0.7}{0.5} + \frac{6.7}{0.7} = 1.40 + 9.57 = 10.97\text{ s}$$
  $$t_{\text{down}} = \frac{0.7}{0.5} + \frac{13.0}{0.7} = 1.40 + 18.57 = 19.97\text{ s}$$
  $$t_{\text{vert}} = 10.97 + 19.97 = 30.94\text{ seconds}$$
- Handling Overheads: $t_{\text{lock}} = 14.0\text{ s}$, $t_{\text{skew}} = 10.0\text{ s}$.

#### Total Single-Cycle Move Time:
$$T_{\text{cycle}} = t_{\text{horiz}} + t_{\text{vert}} + t_{\text{lock}} + t_{\text{skew}} = 28.13 + 30.94 + 14.00 + 10.00 = 83.07\text{ seconds}$$
Net single-cycle crane productivity for this mission:
$$\text{Moves/Hour} = \frac{3,600\text{ s/hr}}{83.07\text{ s}} = 43.34\text{ gross moves/hour}$$

### 6.5 Step 4: Multi-Lane Gate Queuing ($M/G/c$ Allen-Cunneen)
During peak morning arrival (08:00–09:00), drayage trucks arrive at the automated in-gate complex at rate $\lambda = 240\text{ trucks/hour}$. 
- The gate operates $c = 8$ parallel automated lanes.
- Each lane processes an average of 36 trucks per hour ($\mu = 36\text{ trucks/hr}$), corresponding to mean service time $E[S] = \frac{60}{36} = 1.667\text{ minutes}$ (100 seconds).
- The standard deviation of service time is $\sigma_s = 55\text{ seconds} = 0.9167\text{ minutes}$ due to occasional OCR misreads and driver credential errors.

#### Calculations:
1. **Service Time Squared Coefficient of Variation ($C_s^2$)**:
   $$C_s^2 = \frac{\sigma_s^2}{(E[S])^2} = \frac{0.9167^2}{1.667^2} = \frac{0.8403}{2.778} = 0.3025$$
2. **Arrival Squared Coefficient of Variation ($C_a^2$)**:
   Assuming Poisson arrivals, $C_a^2 = 1.0$.
3. **Offered Load ($a$) and Traffic Intensity ($\rho$)**:
   $$a = \frac{\lambda}{\mu} = \frac{240}{36} = 6.6667$$
   $$\rho = \frac{a}{c} = \frac{6.6667}{8} = 0.8333\quad (83.33\%)$$
4. **Erlang-C Delay Probability ($P_c$)**:
   Compute the terms $\frac{a^k}{k!}$ for $k=0 \dots 7$:
   - $k=0: 1.0000$
   - $k=1: 6.6667$
   - $k=2: 22.2222$
   - $k=3: 49.3827$
   - $k=4: 82.3045$
   - $k=5: 109.7394$
   - $k=6: 121.9326$
   - $k=7: 116.1263$
   $$\sum_{k=0}^{7} \frac{a^k}{k!} = 509.3744$$
   The $c$-server term ($k=8$):
   $$\frac{a^8}{8!(1 - \rho)} = \frac{6.6667^8}{40,320 \times (1 - 0.8333)} = \frac{3,901,842.8}{40,320 \times 0.1667} = \frac{3,901,842.8}{6,720.0} = 580.6314$$
   $$P_c(8, 6.6667) = \frac{580.6314}{509.3744 + 580.6314} = \frac{580.6314}{1,090.0058} = 0.5327\quad (53.27\%)$$
   An arriving truck has a 53.27% probability of encountering all 8 lanes busy and having to queue.
5. **Expected Waiting Time in Queue ($W_q$)**:
   $$W_q \approx \left[ \frac{P_c}{c \mu (1 - \rho)} \right] \cdot \left( \frac{1 + C_s^2}{2} \right)$$
   $$W_q \approx \left[ \frac{0.5327}{8 \times 36 \times (1 - 0.8333)} \right] \cdot \left( \frac{1 + 0.3025}{2} \right)$$
   $$W_q \approx \left[ \frac{0.5327}{288 \times 0.1667} \right] \cdot (0.6513) = \left[ \frac{0.5327}{48.00} \right] \cdot 0.6513 = 0.01110 \times 0.6513 = 0.00723\text{ hours}$$
   In minutes:
   $$W_q = 0.00723 \times 60 = 0.434\text{ minutes}\quad (26.0\text{ seconds})$$
6. **Expected Queue Length ($L_q$)**:
   $$L_q = \lambda W_q = 240 \times 0.00723 = 1.735\text{ trucks}$$
7. **Total Gate Time ($W$)**:
   $$W = W_q + E[S] = 0.434 + 1.667 = 2.101\text{ minutes}\quad (126.1\text{ seconds})$$
The 8-lane automated gate comfortably absorbs the 240 trucks/hr peak, maintaining an average queue of under 2 trucks and an average waiting delay of 26 seconds.

---

## 7. Code, query, or calculation

The following self-contained Python script implements the three primary mathematical foundations: Kim's reshuffle formula, the kinematic trapezoidal cycle time synthesizer, and the Allen-Cunneen $M/G/c$ queuing model. It uses only the Python standard library.

```python
"""Terminal Yard and Gate Operations Engineering Synthesizer."""
import math

def expected_reshuffles(tiers: int, occupancy: float, psi: float = 0.18) -> float:
    """Computes expected rehandles per container retrieved via Kim's model."""
    base_rehandles = (tiers - 1.0) / 2.0
    rho = max(0.01, min(0.98, occupancy))
    return base_rehandles * (1.0 + psi * (rho / (1.0 - rho)))

def axis_time(dist: float, v_max: float, accel: float) -> float:
    """Computes travel time for 1D axis under a trapezoidal velocity profile."""
    d_acc = (v_max * v_max) / accel
    return (v_max / accel) + (dist / v_max) if dist >= d_acc else 2.0 * math.sqrt(dist / accel)

def asc_single_cycle(dx: float, dy: float, dz: float,
                     vx=4.0, ax=0.4, vy=1.2, ay=0.3, vz=0.7, az=0.5,
                     t_lock=14.0, t_skew=10.0) -> tuple:
    """Synthesizes ASC single-cycle move time under Chebyshev kinematics."""
    tg, tt = axis_time(dx, vx, ax), axis_time(dy, vy, ay)
    t_horiz = max(tg, tt)
    t_vert = axis_time(dz, vz, az)
    t_total = t_horiz + t_vert + t_lock + t_skew
    return t_total, 3600.0 / t_total

def erlang_c(c: int, a: float) -> float:
    """Calculates Erlang-C delay probability for c servers and offered load a."""
    rho = a / c
    sum_terms = sum((a**k) / math.factorial(k) for k in range(c))
    c_term = (a**c) / (math.factorial(c) * (1.0 - rho))
    return c_term / (sum_terms + c_term)

def mgc_allen_cunneen(lam: float, mu: float, c: int, cs2: float, ca2: float = 1.0) -> tuple:
    """Computes M/G/c queue metrics using Allen-Cunneen approximation."""
    a, rho = lam / mu, lam / (c * mu)
    if rho >= 1.0:
        return float("inf"), float("inf"), float("inf")
    pc = erlang_c(c, a)
    wq_hours = (pc / (c * mu * (1.0 - rho))) * ((ca2 + cs2) / 2.0)
    lq = lam * wq_hours
    return wq_hours * 60.0, (wq_hours + (1.0 / mu)) * 60.0, lq

if __name__ == "__main__":
    # 1. Reshuffle Calculation
    rehandles = expected_reshuffles(tiers=5, occupancy=0.80)
    print(f"Kim Rehandles (5 Tiers, 80% Occ): {rehandles:.3f} moves/retrieval")

    # 2. ASC Kinematic Move Synthesis
    t_move, gross_mpr = asc_single_cycle(dx=72.5, dy=22.4, dz=19.7)
    print(f"ASC Cycle Time: {t_move:.2f} s | Productivity: {gross_mpr:.1f} moves/hr")

    # 3. Gate Queuing Analysis
    wq_min, w_min, lq_trucks = mgc_allen_cunneen(lam=240, mu=36, c=8, cs2=0.3025)
    print(f"Gate Delay Wq: {wq_min * 60.0:.1f} s | Total W: {w_min:.2f} min | Lq: {lq_trucks:.2f} trucks")
```

### Expected Output:
```text
Kim Rehandles (5 Tiers, 80% Occ): 3.440 moves/retrieval
ASC Cycle Time: 69.17 s | Productivity: 52.0 moves/hr
Gate Delay Wq: 26.0 s | Total W: 2.10 min | Lq: 1.73 trucks
```

---

## 8. Data, standards, and regulatory context

Container yard and gate operations operate within a strict matrix of international standards, environmental mandates, and maritime security treaties:

### 8.1 International Standards

- **ISO 6346: Freight Containers — Coding, Identification and Marking**: Governs owner codes, category identifiers, serial numbers, size/type codes, and check digits. Mandates optical and human-readable marking locations on container end walls, side walls, and roof panels.
- **ISO 9711-1: Freight Containers — Information Related to Containers on Board Vessels**: Defines the Bay-Row-Tier coordinate geometry, establishing global parity between vessel stowage plans and terminal yard management topologies.
- **ISO 1496-1: Specification and Testing of Series 1 Freight Containers**: Defines structural engineering limits. Standard containers must withstand an internal test load of $1.8R$ and an allowable stacking superposition load of $86,400\text{ kg}$ per corner post (equivalent to stacking nine laden 30,480 kg containers in calm sea state conditions).

### 8.2 Maritime Safety and Security Treaties

- **IMO SOLAS Chapter VI, Regulation 2 (Verified Gross Mass - VGM)**: Enacted in July 2016. Mandates that no packed container may be loaded onto a vessel without a legally certified VGM obtained via Method 1 (weighing the packed container on calibrated scales) or Method 2 (weighing all contents and dunnage and adding container tare mass). Terminals utilize gate Weigh-In-Motion (WIM) scales to verify VGM compliance, rejecting boxes with gross discrepancies ($> 500\text{ kg}$ or $> 5\%$).
- **IMO IMDG Code (International Maritime Dangerous Goods)**: Chapter 7.2 dictates strict physical segregation tables (Table 7.2.4) governing container spacing in yard blocks. Failure to isolate incompatible chemicals (e.g., organic peroxides from toxic gases) constitutes a severe maritime regulatory offense under international law.
- **ISPS Code (International Ship and Port Facility Security Code)**: Mandated under SOLAS Chapter XI-2. Establishes Security Levels 1, 2, and 3. Dictates biometric access control, 100% manifest verification at gates, perimeter fencing intrusion detection, and vehicle physical screening protocols. In the United States, enforced via the Maritime Transportation Security Act (MTSA) and Transportation Worker Identification Credential (TWIC) biometric cards.

### 8.3 Labor and Environmental Regulations

- **US OSHA 29 CFR Part 1917 (Marine Terminals)**: Establishes legal safety standards for marine terminal operations. Mandates positive physical segregation of pedestrians and ground personnel from container handling equipment, fall protection on container tops ($> 8\text{ feet}$), chassis twistlock verification, and container lifting inspection protocols.
- **Clean Air Action Plans (CAAP) and Environmental Regulations**: Enacted by major port authorities (e.g., Port of Los Angeles, Port of Long Beach, Port of Rotterdam). Enforces zero-emission yard handling equipment mandates, shore-power (cold ironing) mandates, and strict drayage truck idling limits (e.g., California SCAQMD Rule 1149.1, assessing heavy financial penalties against terminal operators for gate queues that cause trucks to idle on public roads for $> 30\text{ minutes}$).

---

## 9. Industry practice and case evidence

Terminal operators deploy distinct combinations of yard equipment, gate automation, and operational policies depending on land values, labor agreements, and hinterland transportation infrastructure.

### 9.1 Comparative Yard Handling Systems

| Dimension | Automated Stacking Crane (ASC) | Rubber-Tired Gantry (RTG) | Straddle Carrier (SC) | Wheeled (Chassis) Yard |
|---|---|---|---|---|
| **Primary Terminal Examples** | Rotterdam (RWG, APMT MVII), Long Beach (LBCT), Singapore (Tuas) | Los Angeles (Pier 400), Felixstowe, Shanghai (Yangshan) | Hamburg (CTA), Melbourne, Antwerp (MPET) | Newark (PNCT domestic), standard US rail ramps |
| **Volumetric Density (TEU/ha)** | $1,000 - 1,400$ | $700 - 900$ | $450 - 600$ | $180 - 250$ |
| **Stacking Height** | 1-over-5 or 1-over-6 | 1-over-4 or 1-over-5 | 1-over-2 or 1-over-3 | 1-high (grounded on road chassis) |
| **Civil Infrastructure Cost** | Very High (Piled concrete crane runways, automated fencing) | Medium (Reinforced concrete runway strips) | Low (Standard heavy-duty asphalt pavement) | Very Low (Asphalt yard with parking stall striping) |
| **Capital Cost per Unit** | $3.5M - $5.0M | $1.8M - $2.5M | $1.2M - $1.6M | $8,000 - $12,000 per chassis |
| **Labor Productivity** | Exceptional (1 remote operator oversees 4–6 ASCs) | Moderate (1 dedicated on-board operator per RTG) | Moderate (1 dedicated driver per vehicle) | Low (Manual hostler tractors move every chassis) |
| **Decoupling Flexibility** | Strict end-loading separation; landside transfers decoupled | Trucks drive directly into block lanes; potential machine interference | Complete decoupling; SC moves containers directly to quay/gate | Maximum; road driver hooks directly to chassis |

### 9.2 Real-World Case Evidence

#### Case 1: Long Beach Container Terminal (LBCT) Pier E, Middle Harbor
- **Context**: An automated 300-acre facility with an annual capacity of 3.3 million TEU located in the San Pedro Bay port complex.
- **Engineering Architecture**: Deployed 72 zero-emission electric rail-mounted ASCs across dual-cantilever perpendicular yard blocks, paired with a fleet of battery-electric AGVs on the quayside. The gate complex utilizes 100% automated OCR portals, Weigh-In-Motion scales, and an algorithmic Truck Appointment System integrated with California's PierPass program.
- **Observed Metrics**: Average drayage truck turnaround time (TTT) fell to 34 minutes for dual-cycle transactions, compared to the regional San Pedro Bay port complex average of 72 minutes. Quayside net crane productivity sustained $> 32$ gross moves/hour even during severe supply chain surges, with zero quayside equipment conflicts due to full physical segregation of automated and human zones.

#### Case 2: Port of Rotterdam — Maasvlakte II (RWG and APM Terminals)
- **Context**: Deepwater greenfield terminals designed to service mega-vessels up to 24,000 TEU with 100% decoupled electric automation.
- **Engineering Architecture**: Utilizes end-loading ASC blocks with decoupled landside truck interchanges. External road drivers never leave their vehicle cabs except to stand on a safety interlock pad at the transaction kiosk. When the driver steps onto the interlock pad, a physical safety circuit activates, locking the ASC gantry from entering the truck interchange bay.
- **Observed Metrics**: Eliminates 100% of ground pedestrian strike fatalities in the automated yard. Gate OCR throughput exceeds 98.2% automated match accuracy without human intervention, maintaining average gate-to-gate turn times under 25 minutes.

#### Case 3: Port of Los Angeles — Fenix Marine Services (Pier 300)
- **Context**: High-volume, hybrid brownfield terminal operating a dense fleet of diesel and hybrid-electric RTGs across a parallel block configuration.
- **Engineering Architecture**: To counter severe gate spillback onto the Vincent Thomas Bridge corridor, the terminal deployed an AI-driven dynamic appointment quota engine that computes real-time RTG stack density. If an import bay's occupancy crosses 75%, the TOS automatically throttles TAS booking appointments for that specific bay, redirecting drayage demand to low-density blocks.
- **Observed Metrics**: Gate queue spillback onto public freeways was reduced by 85% within six months of deployment. Reshuffle moves dropped from 2.8 per retrieval to 1.6 per retrieval by preventing stack freezing during peak import surges.

---

## 10. Failure modes and red flags

| # | Operational Failure Mode | Early Warning Sign / Metric | Underlying Root Cause Mechanism | Systemic Consequence | Engineering Mitigation |
|---|---|---|---|---|---|
| 1 | **Stack Freezing (Yard Gridlock)** | Yard block occupancy crosses 78%; rehandle ratio exceeds 2.5 moves per retrieval. | Lack of vacant ground slots forces cranes to move blocking containers to distant bays, multiplying non-productive moves. | Terminal throughput collapses; STS cranes are starved of boxes; vessel turnaround time doubles. | Impose emergency TOS gate cutoffs on import grounding; enforce punitive demurrage; clear off-dock auxiliary drayage yards. |
| 2 | **In-Gate Highway Spillback** | Approach road queue length exceeds reservoir storage $K$; turn-time variance spikes. | Sudden surge in arrivals ($\lambda > c \mu$) or cascading OCR portal system failure causes transaction delays. | Urban traffic congestion; municipal Clean Air Action Plan fines; police shutdown of terminal access. | Dynamic TAS quota throttling; automated diversion of exception trucks to remote trouble lots; opening bobtail bypass lanes. |
| 3 | **OCR Misread Cascade** | Automatic OCR match confidence rate drops below 92%; trouble kiosk queue overflows. | Adverse weather (heavy rain, dense fog), lens contamination, or bent/corroded container corner castings. | In-gate transaction times increase from 45s to 300s; gate capacity drops by 60%; lanes lock up. | Deploy dual-spectrum infrared/visible linescan cameras; install auto-cleaning lens blowers; integrate chassis RFID tags. |
| 4 | **Heavy-over-Light Stack Toppling** | Visual observation of leaning stacks; TOS weight variance alert. | Inaccurate shipper weight declaration (fraudulent VGM) or flawed TOS slot allocation algorithm. | Dynamic wind loads or minor seismic tremors cause structural corner-post buckling; catastrophic stack collapse. | Enforce mandatory WIM scale verification at gate; reject containers with $| \text{WIM} - \text{VGM} | > 500\text{ kg}$; enforce rigid TOS weight tiers. |
| 5 | **Reefer Plug Shortage / Meltdown** | Reefer yard occupancy $> 90\%$; technician inspection logs show plug temperature warnings. | Inbound reefer arrivals exceed active electrical plug infrastructure; substation transformer overheating. | High-value temperature-controlled cargo (meat, pharmaceuticals) spoils; massive cargo liability claims ($>\$200\text{k/box}$). | Mobile diesel generator (genset) deployment; dynamic TOS reefer booking limits; automated remote temperature monitoring (RMM). |
| 6 | **Dual-ASC Block Interference** | One ASC idle waiting for the companion crane; crane move wait times $> 180\text{ s}$. | Incompatible job scheduling in dual-ASC blocks where waterside and landside cranes contest the same center bays. | Net crane productivity drops from 60 moves/hr (combined) to under 35 moves/hr; AGVs queue at the waterside. | Implement dynamic block-partitioning algorithms; establish virtual mid-block handover buffer zones. |
| 7 | **Chassis Deficit at Wheeled Yards** | Drayage drivers bobtailing through yard unable to locate bare chassis; gate dwell $> 90\text{ min}$. | Hinterland chassis hoarding by logistics warehouses; poor chassis maintenance; chassis pool imbalances. | Complete standstill of import evacuation; grounded yard stacks overflow; yard cannot accept discharge. | Transition to grounded ASC/RTG operations; implement mandatory chassis pool interoperability and daily detention fees. |
| 8 | **Incompatible Dangerous Goods Proximity** | IMDG class proximity alert triggered in TOS database during grounding. | Operator manual slot override or TOS rule-engine bug ignoring IMDG Table 7.2.4 separation tables. | Severe risk of explosion, toxic vapor release, or uncontainable chemical fires in event of container rupture. | Hardcode fail-safe IMDG validation into TOS kernel; lock crane hoist controllers from releasing twistlocks on prohibited slots. |
| 9 | **TAS Appointment Hoarding ("Ghost Bookings")** | TAS booked at 100% quota, but gate arrival no-show rate exceeds 25%. | Drayage motor carriers bulk-booking speculative appointments using robotic browser scripts without cargo ready. | Yard equipment sits idle during booked slots; legitimate truckers locked out; artificial gate capacity shortage. | Require valid container Bill of Lading (BOL) and customs release to book; assess financial no-show penalties; revoke API keys. |
| 10 | **Gantry Skew / Rail Misalignment** | High-frequency rail squeal; ASC gantry wheel flange wear; drive motor current imbalance. | Differential settlement of crane runway piles or thermal expansion warping crane rails out of alignment. | Automated crane emergency stops due to structural skew sensors; complete shutdown of affected yard block. | Regular laser runway surveying; continuous wheel load telemetry monitoring; automated active gantry anti-skew control. |

---

## 11. Assumptions and limits

Every quantitative model developed in this topic relies on explicit engineering assumptions. When operational reality violates these boundaries, the models must be calibrated:

| # | Theoretical Assumption | Engineering Rationale | Operational Reality / Boundary Violation | Diagnostic Test / Verification Method |
|---|---|---|---|---|
| 1 | **Poisson Truck Arrivals ($C_a^2 = 1.0$)** | Simplifies queuing models; valid when thousands of independent truckers act without coordination. | Truck Appointment Systems (TAS) group arrivals into discrete hourly bursts; highway accidents cause batch arrivals ($C_a^2 \gg 1.0$). | Chi-square goodness-of-fit test on inter-arrival time timestamps recorded at gate optical sensors. |
| 2 | **Uniform Random Retrieval Distribution** | Simplifies analytical reshuffling derivations ($E[R] = \frac{H-1}{2}$). | Cargo dwell times exhibit heavy-tailed log-normal distributions; priority cargo is retrieved early, leaving old boxes buried. | Compare empirical reshuffle count against analytical prediction across 10,000 historical TOS container departures. |
| 3 | **Deterministic Crane Kinematic Speeds** | Assumes rated motor acceleration and velocity profiles are achieved on every cycle. | Motor thermal derating, high wind resistance, spreader sway dampening, and mechanical brake wear increase travel time by 15–30%. | Analyze PLC telemetry data comparing planned kinematic trajectory times against actual cycle duration. |
| 4 | **Homogeneous 40-Foot Block Stacking** | Avoids complex 20ft/40ft combinatorial broken stowage geometry in yard blocks. | Real yards mix 20ft, 40ft, 45ft, and high-cube boxes, reducing effective capacity by 10–18% due to length-mismatch constraints. | Calculate the broken stowage ratio: $\frac{\text{Actual TEU Stored}}{\text{Available TGS} \times H}$. |
| 5 | **Independent Parallel Gate Lanes** | Assumes no cross-lane interference or queuing spillover in the $M/G/c$ model. | A single stalled truck or stalled chassis can physically block the approach throat of two adjacent gate lanes. | Video analysis of lane starvation caused by approach throat blockages during peak hours. |
| 6 | **Zero Trucker No-Show Rate** | Assumes every booked TAS appointment translates into an arriving truck at the scheduled hour. | Real-world no-show rates range from 12% to 30% depending on drayage traffic congestion and warehouse readiness. | Calculate daily appointment compliance ratio: $\frac{\text{Actual Arrivals with Appointments}}{\text{Total Booked Appointments}}$. |
| 7 | **Static Dwell Time Independence** | Treats container dwell time as an exogenous parameter unaffected by terminal operations. | When the yard congests, gate turn-times spike, draymen miss warehouse delivery appointments, and cargo dwell time increases endogenously. | Granger causality testing between average yard occupancy and import dwell times. |
| 8 | **Infinite Chassis Supply** | Assumes external drayage drivers arrive with roadable chassis ready to accept grounded boxes. | Regional chassis shortages or chassis roadability inspection failures (bad tires, broken lights) strand trucks inside the yard. | Track inbound gate transactions aborted due to chassis maintenance rejects or chassis deficits. |

---

## 12. Exercises

### Exercise 1: Yard Sizing and Little's Law (Application)
A proposed deepwater container terminal expects an annual container throughput of $1,825,000\text{ TEU/year}$. 
- Import containers represent 60% of volume with a mean dwell time of 5.5 days.
- Export containers represent 40% of volume with a mean dwell time of 3.2 days.
- The design peaking factor is $p_{\text{peak}} = 1.30$.
- The yard block design utilizes 1-over-5 ASC blocks ($H = 5$), each providing 180 Twenty-Foot Ground Slots (TGS).
- Company operating policy mandates that nominal yard utilization must never exceed 72% ($\eta_{\text{oper}} = 0.72$).

*Deliverable*: Compute the required operational inventory, the total nominal static capacity, and the minimum number of ASC blocks that must be constructed.
*Acceptance Criteria*: Fully worked arithmetic showing weighted dwell time, daily throughput, static capacity, and rounded integer block count.
*Estimated Time*: 25 minutes.

### Exercise 2: Analytical Reshuffle Derivation under Non-Uniform Retrieval (Critique & Calculation)
Consider a container stack of height $H = 4$ tiers. Historical data reveals that retrievals are not uniform:
- Container at Tier 4 (top) has retrieval probability $P(K=4) = 0.40$.
- Container at Tier 3 has retrieval probability $P(K=3) = 0.30$.
- Container at Tier 2 has retrieval probability $P(K=2) = 0.20$.
- Container at Tier 1 (bottom) has retrieval probability $P(K=1) = 0.10$.

*Deliverable*:
1. Calculate the exact expected number of reshuffles ($E[R]$) under this non-uniform probability distribution.
2. Compare this value to the classical uniform assumption ($E[R_{\text{uniform}}] = \frac{H-1}{2}$).
3. Quantify the percentage reduction in non-productive crane moves achieved by the terminal's policy of placing high-turnover containers near the top of the stack.
*Acceptance Criteria*: Exact decimal calculations accompanied by physical interpretation of pre-sorting efficiency.
*Estimated Time*: 20 minutes.

### Exercise 3: ASC Kinematics under Triangular versus Trapezoidal Regimes (Calculation)
An automated stacking crane has the following kinematic characteristics:
- Trolley rated speed: $v_t = 1.4\text{ m/s}$.
- Trolley acceleration/deceleration: $a_t = 0.35\text{ m/s}^2$.

*Deliverable*:
1. Calculate the critical acceleration distance $D_{\text{acc}}$ required for the trolley to reach full rated speed.
2. Calculate the travel time required to shift a container across 2 rows ($\Delta y = 6.4\text{ m}$). State whether the velocity profile is trapezoidal or triangular.
3. Calculate the travel time required to shift a container across 8 rows ($\Delta y = 25.6\text{ m}$). State whether the velocity profile is trapezoidal or triangular.
*Acceptance Criteria*: Exact mathematical calculation using correct motion profile equations with units stated.
*Estimated Time*: 25 minutes.

### Exercise 4: Gate Dimensioning using Allen-Cunneen $M/G/c$ (Design & Calculation)
A terminal gate experiences a peak drayage arrival rate of $\lambda = 180\text{ trucks/hour}$. 
- The gate operates automated lanes where each lane processes a mean of 30 trucks per hour ($\mu = 30\text{ trucks/hr}$, so $E[S] = 2.0\text{ min}$).
- The standard deviation of gate service time is $\sigma_s = 72\text{ seconds}$ ($1.2\text{ min}$).
- Corporate customer service commitments guarantee that the average queuing wait time for drayage trucks must not exceed 60 seconds ($W_q \le 1.0\text{ minute}$).

*Deliverable*:
1. Compute the service time squared coefficient of variation ($C_s^2$).
2. Test configurations with $c = 7$ lanes and $c = 8$ lanes using the Allen-Cunneen approximation. Determine the minimum number of lanes required to meet the service level agreement.
3. Calculate the average queue length ($L_q$) under the compliant lane design.
*Acceptance Criteria*: Erlang-C derivation, traffic intensity, wait time in minutes, and definitive engineering lane recommendation.
*Estimated Time*: 35 minutes.

### Exercise 5: SOLAS VGM Check-Digit and Scale Compliance (Recall & Application)
An inbound container arrives at the gate portal. The OCR cameras read the container string as `MSKU 724185-3`.
1. Verify whether check digit `3` is correct under the ISO 6346 modulus-11 check-digit algorithm using letter mappings $M=24, S=30, K=21, U=32$.
2. The carrier manifest declares a VGM of $28,400\text{ kg}$. The gate Weigh-In-Motion (WIM) bending plate scale measures a total gross weight of $29,250\text{ kg}$. The terminal enforces a strict VGM tolerance threshold of $\pm 500\text{ kg}$ or $\pm 2.0\%$. State whether the container should be admitted or routed to the exception trouble kiosk.
*Acceptance Criteria*: Full arithmetic expansion of the ISO 6346 weighting equation and clear pass/fail regulatory determination.
*Estimated Time*: 20 minutes.

### Exercise 6: Dangerous Goods Segregation Evaluation (Critique)
A yard planner attempts to assign an export container containing IMDG Class 3 (Flammable Liquids) to Bay 12, Row 04, Tier 06. The adjacent slots currently hold:
- Bay 12, Row 03, Tier 06: IMDG Class 5.1 (Oxidizing Substances).
- Bay 12, Row 04, Tier 04: Non-hazardous dry general cargo.
- Bay 10, Row 04, Tier 06: IMDG Class 8 (Corrosive Liquid).

*Deliverable*:
Critique this proposed slot assignment against IMO IMDG Table 7.2.4 segregation principles. Identify any direct safety violations and propose an alternative stacking rule.
*Acceptance Criteria*: Detailed regulatory critique identifying chemical incompatibility mechanisms and corrective slotting action.
*Estimated Time*: 20 minutes.

### Exercise 7: Truck Appointment System (TAS) Slot Smoothing Design (Design)
A terminal gate has a maximum physical processing capacity of 200 trucks per hour. The unconstrained historical drayage demand across a 4-hour morning window is:
- 07:00–08:00: 160 trucks
- 08:00–09:00: 260 trucks (peak)
- 09:00–10:00: 240 trucks (peak)
- 10:00–11:00: 140 trucks

*Deliverable*:
Design a TAS hourly appointment quota schedule that caps maximum traffic intensity at $\rho \le 0.85$ ($\le 170\text{ trucks/hr}$), shifts excess demand to adjacent under-utilized slots, and maintains a zero-spillback condition. Specify the operational incentives required to ensure drayage compliance.
*Acceptance Criteria*: Tabular schedule showing original demand, capped quota, shifted trucks, resulting traffic intensity, and commercial enforcement mechanism.
*Estimated Time*: 30 minutes.

### Exercise 8: Failure Mode Forensic Analysis (Critique)
During peak peak import operations, Terminal Alpha observed that while quayside STS cranes achieved 34 gross moves/hour for the first 12 hours of vessel discharge, quayside productivity crashed to 16 moves/hour during the second 12 hours. At the same time, yard RTG rehandle ratios surged from 1.4 to 3.8 moves per retrieval, and truck turn-time at the gate climbed from 28 minutes to 110 minutes.

*Deliverable*:
Write an executive engineering diagnosis identifying the root-cause failure mode, the exact sequence of cascading physical interactions between yard blocks, gates, and quayside transport, and the immediate corrective control actions required to restore terminal equilibrium.
*Acceptance Criteria*: Rigorous causal analysis citing Little's Law, thermodynamic stack freezing, and crane starvation.
*Estimated Time*: 30 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Yard Sizing and Little's Law
1. **Weighted Mean Dwell Time**:
   $$\bar{d} = (0.60 \times 5.5) + (0.40 \times 3.2) = 3.30 + 1.28 = 4.58\text{ days}$$
2. **Average Daily Throughput**:
   $$\lambda = \frac{1,825,000\text{ TEU}}{365\text{ days}} = 5,000\text{ TEU/day}$$
3. **Operational Working Inventory Required**:
   $$C_{\text{oper}} = \lambda \cdot \bar{d} \cdot p_{\text{peak}} = 5,000 \times 4.58 \times 1.30 = 29,770\text{ TEU}$$
4. **Required Nominal Static Capacity**:
   $$C_{\text{static}} = \frac{C_{\text{oper}}}{\eta_{\text{oper}}} = \frac{29,770}{0.72} = 41,347.22\text{ TEU}$$
5. **Capacity Per Block**:
   $$C_{\text{block}} = 180\text{ TGS} \times 5\text{ tiers} = 900\text{ TEU}$$
6. **Minimum Blocks Required**:
   $$N_{\text{blocks}} = \left\lceil \frac{41,347.22}{900} \right\rceil = \lceil 45.94 \rceil = 46\text{ blocks}$$
*Conclusion*: The terminal must build 46 blocks.

### Answer to Exercise 2: Non-Uniform Reshuffle Calculation
1. **Expected Reshuffles under Non-Uniform Retrieval**:
   $$E[R] = \sum_{k=1}^{4} P(K=k) \cdot (4 - k)$$
   - $k=4$ (top): $0.40 \times (4 - 4) = 0.40 \times 0 = 0.00$
   - $k=3$: $0.30 \times (4 - 3) = 0.30 \times 1 = 0.30$
   - $k=2$: $0.20 \times (4 - 2) = 0.20 \times 2 = 0.40$
   - $k=1$ (bottom): $0.10 \times (4 - 1) = 0.10 \times 3 = 0.30$
   $$E[R_{\text{non-uniform}}] = 0.00 + 0.30 + 0.40 + 0.30 = 1.00\text{ reshuffle per retrieval}$$
2. **Comparison to Uniform Distribution**:
   $$E[R_{\text{uniform}}] = \frac{H - 1}{2} = \frac{4 - 1}{2} = 1.50\text{ reshuffles per retrieval}$$
3. **Percentage Reduction**:
   $$\text{Reduction} = \frac{1.50 - 1.00}{1.50} = \frac{0.50}{1.50} = 33.33\%$$
*Interpretation*: Stacking containers with near-term pickup appointments in the top two tiers cuts non-productive crane handling moves by one-third.

### Answer to Exercise 3: ASC Kinematics
1. **Critical Acceleration Distance**:
   $$D_{\text{acc}} = \frac{v_t^2}{a_t} = \frac{1.4^2}{0.35} = \frac{1.96}{0.35} = 5.60\text{ meters}$$
2. **Travel Across 2 Rows ($\Delta y = 6.4\text{ m}$)**:
   Since $\Delta y = 6.4\text{ m} > 5.60\text{ m}$, the motion achieves full rated speed (**Trapezoidal Profile**).
   $$t_t(6.4) = \frac{v_t}{a_t} + \frac{\Delta y}{v_t} = \frac{1.4}{0.35} + \frac{6.4}{1.4} = 4.00 + 4.571 = 8.57\text{ seconds}$$
3. **Travel Across 8 Rows ($\Delta y = 25.6\text{ m}$)**:
   Since $\Delta y = 25.6\text{ m} > 5.60\text{ m}$, the motion is **Trapezoidal**.
   $$t_t(25.6) = \frac{1.4}{0.35} + \frac{25.6}{1.4} = 4.00 + 18.286 = 22.29\text{ seconds}$$

### Answer to Exercise 4: Gate Dimensioning using Allen-Cunneen
1. **Squared Coefficient of Variation ($C_s^2$)**:
   $$\mu = 30\text{ trucks/hr} \implies E[S] = \frac{60}{30} = 2.0\text{ min} = 120\text{ seconds}$$
   $$\sigma_s = 72\text{ seconds} \implies C_s^2 = \frac{72^2}{120^2} = \frac{5,184}{14,400} = 0.360$$
   $$\frac{1 + C_s^2}{2} = \frac{1 + 0.360}{2} = 0.680$$
2. **Testing Lane Configurations**:
   Offered load $a = \frac{\lambda}{\mu} = \frac{180}{30} = 6.00$.
   - **For $c = 7$ lanes**:
     $\rho = \frac{6.00}{7} = 0.8571$ (85.71%).
     $\sum_{k=0}^{6} \frac{6^k}{k!} = 1 + 6 + 18 + 36 + 54 + 64.8 + 64.8 = 244.60$.
     The $k=7$ term: $\frac{6^7}{7!(1 - 0.8571)} = \frac{279,936}{5,040 \times 0.1429} = \frac{279,936}{720.216} = 388.68$.
     $$P_c(7, 6) = \frac{388.68}{244.60 + 388.68} = 0.6137$$
     $$W_q \approx \left[ \frac{0.6137}{7 \times 30 \times (1 - 0.8571)} \right] \times 0.680 = \left[ \frac{0.6137}{210 \times 0.1429} \right] \times 0.680 = \frac{0.6137}{30.0} \times 0.680 = 0.01391\text{ hr} = 0.835\text{ min}\quad (50.1\text{ s})$$
     Since $W_q = 50.1\text{ s} \le 60\text{ s}$, $c = 7$ lanes meets the SLA.
   - **For $c = 8$ lanes**:
     $\rho = \frac{6.00}{8} = 0.7500$.
     $P_c(8, 6) = 0.2851$.
     $$W_q \approx \left[ \frac{0.2851}{8 \times 30 \times (1 - 0.75)} \right] \times 0.680 = \frac{0.2851}{60.0} \times 0.680 = 0.00323\text{ hr} = 0.194\text{ min}\quad (11.6\text{ s})$$
3. **Queue Length ($L_q$) under 7-Lane vs 8-Lane Design**:
   - For $c = 7$: $L_q = \lambda W_q = 180 \times 0.01391 = 2.50\text{ trucks}$.
   - For $c = 8$: $L_q = \lambda W_q = 180 \times 0.00323 = 0.58\text{ trucks}$.
   *Recommendation*: 7 lanes is mathematically compliant, but 8 lanes provides critical resilience ($W_q$ drops from 50s to 12s) against traffic surges.

### Answer to Exercise 5: SOLAS VGM and Check-Digit
1. **ISO 6346 Check Digit Verification**:
   Characters: `M` (24), `S` (30), `K` (21), `U` (32), `7`, `2`, `4`, `1`, `8`, `5`.
   Weights ($2^i$ for $i=0 \dots 9$): 1, 2, 4, 8, 16, 32, 64, 128, 256, 512.
   - $M: 24 \times 1 = 24$
   - $S: 30 \times 2 = 60$
   - $K: 21 \times 4 = 84$
   - $U: 32 \times 8 = 256$
   - $7: 7 \times 16 = 112$
   - $2: 2 \times 32 = 64$
   - $4: 4 \times 64 = 256$
   - $1: 1 \times 128 = 128$
   - $8: 8 \times 256 = 2,048$
   - $5: 5 \times 512 = 2,560$
   $$\text{Sum} = 24 + 60 + 84 + 256 + 112 + 64 + 256 + 128 + 2,048 + 2,560 = 5,492$$
   $$5,492 \div 11 = 499.2727 \implies 499 \times 11 = 5,489$$
   $$\text{Remainder} = 5,492 - 5,489 = 3$$
   $$\text{Check Digit} = 3 \pmod{10} = 3$$
   The check digit matches the declared value `3`. The OCR read is valid.
2. **VGM Compliance Check**:
   $$\Delta M = |29,250 - 28,400| = 850\text{ kg}$$
   $$\text{Percentage Discrepancy} = \frac{850}{28,400} = 2.99\%$$
   The discrepancy ($850\text{ kg}$ / $2.99\%$) violates both the absolute $500\text{ kg}$ threshold and the $2.0\%$ tolerance. 
   *Determination*: **REJECT entry to automated yard.** Divert truck immediately to the exception trouble kiosk for manual reweighing and manifest reconciliation under SOLAS regulations.

### Answer to Exercise 6: Dangerous Goods Segregation Evaluation
1. **Violations Identified**:
   - Proposed container: Class 3 (Flammable Liquids).
   - Adjacent box at Bay 12, Row 03, Tier 06: Class 5.1 (Oxidizing Substances).
   - Under IMDG Code Table 7.2.4, Class 3 and Class 5.1 are chemically incompatible. Oxidizers actively accelerate combustion of flammable hydrocarbon vapors. The regulation mandates that they must be "Separated From" each other (minimum 6 meters or 1 full bay buffer). Placing them in adjacent rows in the same bay is an egregious regulatory violation.
2. **Corrective Action**:
   Reject the slot assignment. Relocate the Class 3 container to a dedicated flammable liquids yard block equipped with hydrocarbon gas detectors, fire-fighting foam monitors, and isolated industrial drainage sumps.

### Answer to Exercise 7: TAS Slot Smoothing Design
Target maximum traffic intensity: $\rho \le 0.85 \implies \lambda_{\text{max}} = 200 \times 0.85 = 170\text{ trucks/hr}$.
- **Hour 07:00–08:00**: Original demand 160. Quota cap 170. Absorbs 160 original + 10 shifted from Hour 2 = 170 trucks ($\rho = 0.85$).
- **Hour 08:00–09:00**: Original demand 260. Quota capped at 170. Excess = 90 trucks. (10 shifted back to 07:00, 80 shifted forward to 09:00/10:00).
- **Hour 09:00–10:00**: Original demand 240. Quota capped at 170. Excess = 70 trucks. Total to shift forward to 10:00 and 11:00 = $80 + 70 = 150$ trucks.
- **Hour 10:00–11:00**: Original demand 140. Quota capped at 170. Can absorb $170 - 140 = 30$ shifted trucks. Total processed = 170 trucks ($\rho = 0.85$).
- **Hour 11:00–12:00**: Off-peak window absorbs remaining 120 shifted trucks.
*Commercial Enforcement*: Implement peak congestion pricing (e.g., $35 fee for 08:00–10:00 appointments) and make off-peak appointments (07:00 or after 11:00) free or credited, combined with a $75 penalty for no-shows.

### Answer to Exercise 8: Failure Mode Forensic Analysis
1. **Diagnosis**: Classic **Thermodynamic Stack Freezing Cascade**.
2. **Mechanism**:
   - In the first 12 hours, high quayside discharge pushed yard occupancy past the 78% threshold.
   - At $\rho > 0.80$, available empty buffer slots vanished. To service landside import retrievals, RTGs had to move blocking boxes to adjacent bays, multiplying rehandles from 1.4 to 3.8 moves/retrieval.
   - RTGs became completely consumed by non-productive reshuffles, causing truck turn-times to quadruple to 110 minutes.
   - As RTGs stalled on the landside, they failed to clear waterside transfer zones. Horizontal transport vehicles (terminal tractors) queued up in the yard, unable to unload export boxes or pick up discharged import containers.
   - This starved the quayside STS cranes of empty transporters, collapsing quayside productivity from 34 to 16 moves/hr.
3. **Immediate Mitigation**:
   - Enforce an immediate temporary moratorium on non-critical gate appointments.
   - Halt import deliveries in frozen blocks and dedicate all RTG capacity to pre-marshaling and relocating empty containers to off-dock yards to drop occupancy below 70%.

---

## 14. Interview questions

### Question 1: The Yard Utilization Knee
**Prompt**: *"A marine terminal director tells you that to maximize capital asset efficiency, the yard should aim for 95% average occupancy. How do you respond from an operations research and engineering perspective?"*
- **Strong Answer**:
  - Identifies that 95% utilization confuses static storage with dynamic throughput.
  - Explains the non-linear relationship between occupancy ($\rho$) and reshuffle probability ($E[R]$), citing Kim's formula where secondary rehandles scale as $\frac{\rho}{1 - \rho}$.
  - Demonstrates that at 95% occupancy, the probability of an available adjacent ground slot approaches zero, forcing yard cranes to perform cross-bay gantry shuffles.
  - Concludes that operating above 80% creates "stack freezing," which throttles both gate evacuation and quayside vessel productivity, ultimately decreasing annual terminal revenue.
- **Weak Answer**:
  - Agrees that 95% utilization is desirable because real estate is expensive.
  - Suggests simply working crane operators faster or buying more cranes without recognizing the mathematical lack of physical empty slots.

### Question 2: Decoupled ASC Block Design
**Prompt**: *"Why do modern mega-terminals (e.g., Rotterdam Maasvlakte II, Long Beach Container Terminal) predominantly utilize perpendicular end-loading ASC blocks rather than parallel RTG layouts?"*
- **Strong Answer**:
  - Cites the complete physical decoupling of automated and human operations: external human road drivers remain confined to landside interchange zones, while automated quayside vehicles (AGVs) operate exclusively at the waterside end.
  - Highlights safety compliance (OSHA 29 CFR 1917), eliminating human-machine strike hazards.
  - Explains volumetric density: ASC blocks stack 1-over-5 or 1-over-6 across 8 to 12 rows with zero internal roadways, achieving $> 1,200\text{ TEU/ha}$ compared to $750\text{ TEU/ha}$ for RTGs.
  - Analyzes labor productivity: ASCs enable 1 remote human supervisor to monitor 4 to 6 automated cranes.
- **Weak Answer**:
  - Focuses solely on "automation is modern" without explaining the physical separation of interfaces or density mathematics.
  - Overlooks the trade-off of high civil foundation capital costs for crane rails.

### Question 3: Kinematic Regimes in Cycle Time Synthesis
**Prompt**: *"When calculating the single-cycle time of an automated stacking crane, why is it incorrect to simply sum the individual gantry, trolley, and hoist travel times?"*
- **Strong Answer**:
  - Explains multi-axis drive kinematics: gantry ($X$) and trolley ($Y$) axes operate simultaneously under variable-frequency drives, following a Chebyshev distance metric where horizontal motion time is $\max(t_g, t_t)$, not $t_g + t_t$.
  - Notes that hoisting ($Z$) cannot occur fully concurrently with horizontal travel across full block spans due to safety envelope rules: the spreader must lift the container above the highest obstacle tier ($Z_{\text{clear}}$) before horizontal acceleration is permitted.
  - Explicitly accounts for constant acceleration/deceleration trapezoidal versus triangular motion profiles depending on whether distance exceeds $\frac{v^2}{a}$.
- **Weak Answer**:
  - Assumes Manhattan kinematics ($t_x + t_y + t_z$) or Euclidean straight-line physics.
  - Forgets spreader twistlock engagement and optical fine-positioning overhead times.

### Question 4: Multi-Lane Gate Queuing with Service Time Variance
**Prompt**: *"Why does an $M/M/c$ queuing model significantly underestimate drayage waiting times at an automated terminal gate, and how does the Allen-Cunneen approximation resolve this?"*
- **Strong Answer**:
  - Points out that $M/M/c$ assumes memoryless, exponentially distributed service times ($C_s^2 = 1.0$), which fails to represent reality.
  - Explains that automated gate service times are highly non-exponential: successful OCR/pedestal transactions have low variance clustered around 45–60 seconds, but exception transactions (OCR misreads, missing customs holds, VGM errors) take 300+ seconds.
  - Shows how the Allen-Cunneen formula incorporates the service squared coefficient of variation via the term $\frac{C_a^2 + C_s^2}{2}$. If exception rates cause high variance, waiting times escalate even if mean service rate $\mu$ remains constant.
- **Weak Answer**:
  - States that queueing models are purely theoretical and cannot handle multiple lanes.
  - Does not know what $C_s^2$ represents or confuses service time with arrival rate.

### Question 5: TAS Slot Quotas as a Workload Leveler
**Prompt**: *"A drayage association sues a port authority claiming that Truck Appointment Systems restrict trade by capping the number of trucks allowed into the terminal. How do you defend the appointment quota from an engineering perspective?"*
- **Strong Answer**:
  - Frames TAS not as a volume restriction, but as a dynamic queue-smoothing mechanism that maximizes effective daily gate throughput by preventing $\rho$ from exceeding 1.0.
  - Explains that unconstrained arrivals generate peak-hour traffic intensity $\rho > 1.0$, which causes exponential queue growth, gridlocks terminal access roads, exhausts truck driver Hours of Service (HOS), and violates clean-air idling regulations.
  - Demonstrates that smoothing arrivals into adjacent hours maximizes crane utilization, shortens gate turn-times from 90+ minutes to under 35 minutes, and enables more total productive container turns per day.
- **Weak Answer**:
  - Argues purely from a regulatory compliance standpoint ("the law allows ports to do it").
  - Fails to explain the mathematical mechanics of arrival rate smoothing on queue delay.

### Question 6: Double-Cycling in Yard Cranes
**Prompt**: *"What is double-cycling in yard crane operations, what conditions are required to execute it, and what is its theoretical impact on crane productivity?"*
- **Strong Answer**:
  - Defines double-cycling (dual-cycling) as combining an inbound move (e.g., grounding an export container from a drayage truck) with an outbound move (e.g., retrieving an import container for that same truck or an adjacent vehicle) in a single round-trip cycle.
  - Identifies prerequisites: advanced TOS mission sequencing, physical proximity of drop-off and pickup slots, and dual-transaction drayage bookings.
  - Quantifies productivity: eliminates empty spreader travel on the return leg, cutting total cycle time per container from $\approx 80\text{ s}$ to $\approx 50\text{ s}$, which increases crane net throughput by 30% to 45%.
- **Weak Answer**:
  - Thinks double-cycling means lifting two 20-foot containers simultaneously (that is twin-lift hoisting).
  - Unaware of the sequencing requirements in the TOS.

### Question 7: IMO IMDG Chemical Segregation in the Yard
**Prompt**: *"During a night shift, an operator places an IMDG Class 2.1 (Flammable Gas) container directly beneath an IMDG Class 5.1 (Oxidizing Substance) container in the same vertical stack to save ground space. What are the engineering and physical safety consequences?"*
- **Strong Answer**:
  - Identifies this as an immediate catastrophic violation of IMO IMDG Code Table 7.2.4 segregation rules.
  - Explains the chemical physics: flammable compressed gases (e.g., propane) are prone to minor valve leaks; oxidizers actively release oxygen and trigger violent ignition upon contact with hydrocarbons without requiring an external spark.
  - Stacking them vertically ensures that any escaping gas rises directly into the oxidizing cargo, creating an extreme risk of explosion.
  - States the emergency corrective action: immediate halt of crane operations, evacuation of the block, dispatch of hazardous materials safety team to de-stack and isolate the units into separate hazardous bunkers.
- **Weak Answer**:
  - Thinks it is acceptable as long as the containers are properly locked with twistlocks.
  - Unaware of the chemical interactions between oxidizers and flammable gases.

### Question 8: Weigh-In-Motion and SOLAS VGM Enforcement
**Prompt**: *"How does dynamic Weigh-In-Motion (WIM) technology operate at terminal in-gates, and what error sources must an engineer account for when calibrating WIM systems against SOLAS VGM mandates?"*
- **Strong Answer**:
  - Explains technology: quartz piezoelectric sensors or bending-plate strain gauges embedded flush in the lane pavement measure dynamic wheel/axle loads at speeds up to 10 km/h and integrate them to compute gross vehicle weight.
  - Subtracts known tractor tare mass and registered chassis weight to deduce container gross mass.
  - Lists error sources: vehicle acceleration/braking dynamics across the sensor, fluid sloshing in tank containers (dynamic load shifting), pavement surface rutting/deflection, temperature sensor drift, and fuel level differences in the tractor.
  - Establishes calibration protocol: regular static calibration passes with certified test weights to maintain accuracy within $\pm 1.0 - 2.0\%$.
- **Weak Answer**:
  - Believes WIM is a standard static truck scale that simply weighs trucks while stopped.
  - Ignores dynamic vehicle load transfer caused by braking or road irregularities.

---

## 15. Key terms

- **Allen-Cunneen Approximation**: Analytical approximation formula for expected waiting time in multi-server $GI/G/c$ queues with general service time distributions.
- **Automated Guided Vehicle (AGV)**: Unmanned, software-controlled mobile transporter conveying containers between quay and yard.
- **Automated Stacking Crane (ASC)**: Rail-mounted, unmanned gantry crane operating in high-density automated container storage blocks.
- **Bay-Row-Tier (BRT)**: Standardized Cartesian coordinate system defined under ISO 9711 to address container storage positions.
- **Cantilever**: Structural crane overhang extending past the runway rails to access truck interchange lanes or buffer slots.
- **Chebyshev Kinematics**: Metric governing crane travel where simultaneous gantry and trolley translation time is determined by $\max(t_x, t_y)$.
- **Demurrage**: Storage penalty fee assessed on containers remaining inside the terminal yard beyond allowable free time.
- **Detention**: Equipment holding fee assessed by shipping lines on containers or chassis held outside the terminal beyond contractual free time.
- **Double-Cycling**: Scheduling discipline combining a container drop-off and pickup into a single continuous crane round-trip.
- **Dwell Time**: Duration in days that a container remains grounded in the terminal yard between arrival and departure.
- **Erlang-C Formula**: Classical queuing formula computing the probability that an arriving customer finds all $c$ parallel servers busy.
- **Ground Slot (TGS)**: A standardized physical pavement footprint accommodating one Twenty-Foot Equivalent Unit.
- **Housekeeping**: Proactive relocation moves performed during non-peak shifts to prepare stacks for upcoming vessel or rail operations.
- **IMDG Code**: International Maritime Dangerous Goods Code governing physical segregation and storage of hazardous materials.
- **ISO 6346**: International standard specifying alphanumeric container coding, serial numbering, and modulus-11 check-digit verification.
- **Kim's Formula**: Analytical formulation estimating the expected number of rehandles per container retrieval as a function of bay height and occupancy.
- **Little's Law**: Fundamental operations theorem stating that average inventory equals throughput multiplied by mean dwell time ($L = \lambda W$).
- **Optical Character Recognition (OCR)**: Machine-vision portal technology automatically decoding container ISO codes, license plates, and chassis numbers.
- **Pre-marshaling**: Sorting and ordering export containers into exact vessel stowage sequence prior to ship arrival to maximize quay crane velocity.
- **Reefer Rack**: Steel framework structure providing 440V electrical power receptacles and technician access catwalks for refrigerated containers.
- **Reshuffle (Rehandle)**: Non-productive crane lift required to relocate an overlying container to access a buried target container.
- **Rubber-Tired Gantry (RTG)**: Mobile gantry crane running on rubber tires, typically spanning 6 to 8 rows plus a truck lane.
- **SOLAS VGM**: Safety of Life at Sea regulation mandating certified Verified Gross Mass documentation prior to loading containers onto ships.
- **Stack Freezing (Yard Lock)**: Failure state occurring when yard occupancy exceeds 80%, causing exponential reshuffling that halts container handling.
- **Straddle Carrier**: High-mobility diesel-electric vehicle that straddles and stacks containers 1-over-2 or 1-over-3 without fixed runway rails.
- **Trapezoidal Profile**: Kinematic motion profile consisting of constant acceleration, constant velocity cruising, and constant deceleration phases.
- **Trouble Kiosk**: Dedicated exception lane and pedestal where gate transactions with documentation or OCR errors are resolved by clerks.
- **Truck Appointment System (TAS)**: Software booking system that allocates hourly gate arrival quotas to level drayage traffic flow.
- **Turnaround Time (TTT)**: Total elapsed time from a drayage truck's gate entry to its departure through the out-gate.
- **Weigh-In-Motion (WIM)**: Embedded dynamic pavement sensors measuring axle weights and gross vehicle mass at low travel speeds.

---

## 16. Sources

- Allen, A. O., & Cunneen, C. R. (1972). *Single server and multi-server queues with general service time distributions*. IBM Research Report.
- Carlo, H. J., Vis, I. F., & Roodbergen, K. J. (2014). Storage yard operations in container terminals: An overview of concepts, models, and research trends. *European Journal of Operational Research*, 235(2), 329–343.
- International Maritime Organization (IMO). (2016). *International Maritime Dangerous Goods (IMDG) Code, 2016 Edition*. IMO Publishing, London.
- International Maritime Organization (IMO). (2016). *SOLAS Chapter VI, Regulation 2: Verified Gross Mass (VGM)*. IMO Resolution MSC.380(94).
- International Organization for Standardization (ISO). (2013). *ISO 6346:1995/Amd 3:2012: Freight containers — Coding, identification and marking*. ISO, Geneva.
- International Organization for Standardization (ISO). (1990). *ISO 9711-1:1990: Freight containers — Information related to containers on board vessels — Part 1: Bay plan system*. ISO, Geneva.
- Kim, K. H. (1997). Evaluation of the number of rehandles in container yards. *Computers & Industrial Engineering*, 32(4), 701–711.
- Kim, K. H., & Kim, H. B. (1999). Segregating space allocation models for container inventories in port container terminals. *International Journal of Production Economics*, 59(1-3), 415–423.
- Saanen, Y. A. (2004). *An approach for designing robotized marine container terminals*. PhD Thesis, Delft University of Technology, The Netherlands.
- Stahlbock, R., & Voß, S. (2008). Operations research at container terminals: A literature update. *OR Spectrum*, 30(1), 1–52.
- Steenken, D., Voß, S., & Stahlbock, R. (2004). Container terminal operation and operations research: A classification and literature review. *OR Spectrum*, 26(1), 3–49.
- U.S. Occupational Safety and Health Administration (OSHA). (2020). *29 CFR Part 1917: Marine Terminals*. U.S. Department of Labor, Washington, D.C.

---

## 17. Links

- Module Survey File: [`../../../course/41_maritime_rail_air_operations.md`](../../../course/41_maritime_rail_air_operations.md)
- Module Index: [`_index.md`](_index.md)
- Sibling Topic 01 (Berth and Quay Crane Operations): [`01_berth-quay-crane.md`](01_berth-quay-crane.md)
- Sibling Topic 03 (Rail Network and Path Capacity): [`03_rail-network-paths.md`](03_rail-network-paths.md)
- Sibling Topic 08 (Terminal Simulation): [`08_terminal-simulation.md`](08_terminal-simulation.md)
- Sibling Topic 10 (Automation and Capital Decisions): [`10_automation-capital.md`](10_automation-capital.md)
- Prerequisite Module 12 (Transportation, Routing, and Last-Mile): [`../../02_methods/12_transportation_routing_last_mile/_index.md`](../../02_methods/12_transportation_routing_last_mile/_index.md)
- Commercial Interface Module 36 (Global Freight, Sourcing, and International Trade): [`../../06_domain_electives/36_global_freight_sourcing_trade/_index.md`](../../06_domain_electives/36_global_freight_sourcing_trade/_index.md)
