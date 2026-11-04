# Terminal Automation Engineering, Capital Budgeting, and Decoupled Stacking Logistics

> Module 41 — Maritime, Rail, and Air Cargo Operations Engineering
> Topic 10 of 10
> Estimated study time: 4.5 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 12 (Transportation Routing & Last Mile), Module 20 (Operations Research & Mathematical Programming), Topic 41.01 (Berth and Quay Crane Operations), Topic 41.02 (Terminal Yard and Gate), Topic 41.08 (Terminal Simulation)
> Version 2026.5

*Automated container terminal engineering, horizontal transport architecture (battery-electric AGVs vs Lift-AGVs vs manual terminal tractors), automated stacking crane systems (cantilever ASCs vs end-loading ASCs vs automated hybrid e-RTGs), Remote Operating Station (ROS) ergonomic ratios and tele-operation latency budgets, subterranean and quayside civil engineering retrofit costs, discounted cash flow (DCF) capital budgeting, Levelized Cost per Move (LCPM), closed queueing network fleet sizing under battery state-of-charge (SoC) charging cycles, and labor productivity substitution economics.*

---

## 1. Why this topic exists

In global intermodal container supply chains, container terminal automation represents the single largest, most complex capital investment decision a port authority, marine terminal operator (MTO), or global logistics concessionaire will ever execute. Over the past two decades, the container shipping industry has undergone an unprecedented vessel up-sizing cycle. Ultra-Large Container Vessels (ULCVs) exceeding 24,000 Twenty-Foot Equivalent Units (TEU)—such as the Evergreen A-class (24,004 TEU) and MSC Tessa-class (24,116 TEU)—measure 400 meters in length and 61.5 meters in beam (24 container rows across deck). These vessels call at deepwater hub ports demanding 5,000 to 8,000 container moves within strict 36-to-48-hour berth windows to maintain global liner schedule integrity.

Meeting these berth performance requirements under traditional manual terminal operations has reached physical, cognitive, and spatial limits:
1. **Human Fatigue and Kinematic Limits:** A human quay crane operator seated in a cabin 45 to 55 meters above the quay deck experiences constant neck, spine, and ocular strain while manipulating 40-to-65-ton suspended loads. Manual gross crane productivity typically peaks at 28 to 32 moves per hour during early shift hours but steadily degrades to 20 to 24 moves per hour under fatigue, night conditions, or adverse weather.
2. **Horizontal Congestion and Yard Collisions:** Manual internal transfer vehicles (ITVs / terminal tractors) operating in mixed-traffic yards generate extreme variance in travel and positioning times. Dispatched manually or via basic radio-data terminals, human drivers experience bunching at quay crane backreaches, miss optimal yard crane handover slots, and create severe safety hazards. Ground personnel and pin-men working on foot in active container yards face high mortality and injury risks from heavy equipment blind spots.
3. **The Brownfield Spatial Squeeze:** Major gateway terminals located in metropolitan estuaries (e.g., Los Angeles, Long Beach, New York/New Jersey, Rotterdam, Hamburg, Hong Kong) cannot expand their geographical footprint due to urban encroachment, environmental marine preserves, and prohibitive dredging costs. The only mechanism to expand terminal throughput from 1.5 million TEU to 3.0+ million TEU is vertical densification: transitioning from low-density manual straddle carriers (1-over-2 or 1-over-3 stacking) or rubber-tyred gantry cranes (RTGs, 1-over-5 stacking with wide truck lanes) to high-density Automated Stacking Cranes (ASCs, 1-over-5 or 1-over-6 stacking in 8-to-10-wide blocks).

However, terminal automation introduces massive capital risk. A full-scale greenfield automated container terminal—such as Rotterdam World Gateway (RWG), APM Terminals Maasvlakte II, Long Beach Container Terminal (LBCT Pier E), or PSA Singapore Tuas Phase 1—requires between $1.5 billion and $4.0 billion in total capital expenditure (CAPEX). Converting an existing operational manual facility (brownfield automation) requires $500 million to $1.2 billion, alongside years of operational disruption, subterranean civil reconstruction, and complex labor union renegotiations.

Terminal automation replaces flexible, variable operating costs (OPEX) with massive, irreversible fixed capital charges (depreciation, debt service, software licensing, and specialized engineering maintenance). In financial terms, automation induces extreme **operating leverage**. If container volumes remain high and the automated system hits its rated productivity targets (30 to 35 sustained gross moves per crane hour), the terminal achieves an extraordinarily low marginal cost per container move. Conversely, if equipment interface bottlenecks, equipment control system (ECS) dispatch software deadlocks, or global macroeconomic downturns depress terminal throughput, the operator faces catastrophic financial losses, unable to shed fixed capital costs as a manual terminal sheds casual labor hours.

Operations engineers and capital budgeting directors must evaluate four tightly coupled engineering domains:
- **Horizontal Transport Kinematics:** Selecting between standard battery-electric Automated Guided Vehicles (AGVs), active Lift-AGVs, and manual or autonomous terminal tractors, balancing vehicle CAPEX against quay crane decoupling efficiency.
- **Yard Stacking Topology:** Dimensioning perpendicular end-loading ASCs, cantilever ASCs, or automated hybrid electric RTGs (e-RTGs) based on quayside-to-landside container volume splits and civil foundation soil bearing capacities.
- **Human-in-the-Loop Ergonomics:** Sizing Remote Operating Stations (ROS), establishing operator-to-crane ratios, and budgeting tele-operation video streaming latency to prevent remote landing bottlenecks.
- **Infrastructure Electrification and Telemetry:** Upgrading quayside crane runway beams to absorb massive wheel point loads, deploying high-power inductive or conductive opportunity fast-charging grids, and rolling out private 5G Ultra-Reliable Low-Latency Communication (URLLC) networks.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Evaluate and select** horizontal transport architectures (battery-electric AGVs, active Lift-AGVs, manual terminal tractors) based on buffer decoupling physics, cycle dead-time, and quay crane net productivity.
2. **Formulate and solve** closed queueing network models for electric AGV fleet sizing and opportunity charging infrastructure under battery State of Charge (SoC) and Depth of Discharge (DoD) constraints.
3. **Analyze and specify** automated container stacking yard configurations, comparing cantilever ASCs, end-loading ASCs, and automated hybrid e-RTGs across stacking density, wheel load bearing requirements, and landside/waterside decoupling.
4. **Dimension Remote Operating Station (ROS)** operator-to-crane allocation ratios using Erlang multiserver queueing formulations and ergonomic cognitive workload thresholds.
5. **Quantify subterranean and civil engineering retrofit expenditures** for brownfield container terminals, including crane runway pile driving, quay wall tieback strengthening, and medium-voltage power distribution.
6. **Construct multi-decade discounted cash flow (DCF) models** to compute Net Present Value (NPV), Internal Rate of Return (IRR), and Levelized Cost per Move (LCPM) across terminal concession lifecycles.
7. **Model labor substitution economics**, calculating crane gang-hour reductions, severance liabilities, collective bargaining friction, and productivity crossover thresholds against union wage escalations.
8. **Specify industrial wireless telemetry architectures**, contrasting 3GPP Rel-16/17 private 5G URLLC with legacy Wi-Fi and leaky feeder systems under container stack multipath interference.

---

## 3. The core mental model

The engineering foundation of an automated container terminal is the **Three-Tier Decoupled Buffer Architecture**. In a manual terminal, physical operations are tightly coupled: a Ship-to-Shore (STS) crane cannot complete a cycle until a terminal tractor positions a chassis beneath the spreader; a yard RTG cannot complete an outbound lift until a road truck or internal tractor parks in the transfer lane. This coupling causes high stochastic variance to propagate across the entire terminal, resulting in chronic equipment starvation and waiting dead-time.

An automated terminal eliminates stochastic coupling by inserting passive physical buffers between each automated subsystem. This architecture divides the terminal into five discrete physical and digital zones, illustrated below:

```text
========================================================================================
                      CONTAINER TERMINAL AUTOMATION ARCHITECTURE
========================================================================================

 [ OCEAN ]           20,000 - 24,000 TEU Ultra-Large Container Vessel (ULCV)
    ~~~   ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
          |        |        |        |        |        |        |        |
          +--------+--------+--------+--------+--------+--------+--------+
                     [ QUAY CRANE MAIN TROLLEY (Automated / Semiautomatic) ]
                                         |
                                  [ PORTAL SILL ]
                                         v
 [ BUFFER 1 ]     === QUAYSIDE DECOUPLING BUFFER PLATFORM / CASSETTE ===
                  (Dual-trolley secondary hoist absorbs cycle time variance)
                                         |
                                         v
 [ HORIZONTAL ]   === BATTERY-ELECTRIC LIFT-AGV TRANSPORT GRID ===
                  - Automated Guided Vehicles navigate via GNSS RTK / Transponders
                  - Fast Conductive Opportunity Charging Pits (350-450 kW DC)
                  - Private 5G SA URLLC (<5 ms radio latency, 99.999% availability)
                                         |
                                         v
 [ BUFFER 2 ]     === WATERSIDE TRANSFER ZONE (WTZ) GROUND BUFFERS ===
                  (Lift-AGVs drop containers onto ground racks without crane waiting)
                                         |
                                         v
 [ YARD STACK ]   === AUTOMATED STACKING CRANE (ASC) BLOCKS (1-over-5 / 1-over-6) ===
                  - Twin Cantilever ASCs per block (Perpendicular layout)
                  - Electric conductor busbar / motorized cable reel (zero direct emissions)
                  - Autonomous stacking, internal housekeeping, and pre-marshaling
                                         |
                                         v
 [ BUFFER 3 ]     === LANDSIDE TRANSFER ZONE (LTZ) EXCHANGE LANES ===
                  (ASCs pre-stage outbound containers; road trucks park in secure stalls)
                                         ^
                                         | (Remote Video Tele-Operation / 50 ms glass-to-glass)
 [ REMOTE OPS ]   === REMOTE OPERATING STATIONS (ROS) CONSOLE POOL ===
                  - Supervisory ratio: 1 human operator per 4 to 8 ASCs
                  - Operator manually performs final twistlock landing onto road chassis
                                         |
                                         v
 [ LANDSIDE ]     === AUTOMATED GATE SYSTEM (AGS) & INTERMODAL RAIL SIDING ===
                  (Optical Character Recognition OCR, Weigh-in-Motion, RFID, Biometrics)
========================================================================================
```

### The Three Decoupling Interfaces

1. **Quayside Dual-Trolley Buffer (Buffer 1):** The primary trolley hoists the container from the vessel cell guide and lowers it onto an elevated intermediate platform built inside the crane portal sill (or an automated lashing platform). Secondary hoist trolleys (or automated twistlock handling robots) retrieve the container from the platform and lower it to the quay deck. This decouples the long hoist cycle over the ship's hold from horizontal yard transport arrivals.
2. **Waterside Transfer Zone (Buffer 2):** Utilizing active **Lift-AGVs** equipped with hydraulic lifting frames, vehicles deposit containers directly onto passive steel ground supports (stools or cassettes) in the Waterside Transfer Zone (WTZ). The Lift-AGV departs immediately to execute its next dispatch task. The Automated Stacking Crane (ASC) retrieves the container from the stool when kinematically optimal, eliminating vehicle-crane rendezvous dead-wait times.
3. **Landside Transfer Zone (Buffer 3):** Outbound containers are pre-marshaled into the end bays of the ASC block prior to road truck arrival. When an Over-The-Road (OTR) external truck enters the Landside Transfer Zone (LTZ) stall, automated optical vehicle detection systems verify truck position and alignment. A remote human operator at an ROS console takes over the hoist control only for the final 1.5-meter descent to align the container casting with the truck chassis twistlocks.

### Hierarchical Supervisory Control Stack

```text
+--------------------------------------------------------------------------------+
|                     TOS: Terminal Operating System                             |
|  - Vessel stowage planning, berth scheduling, yard block allocation, gate appt |
+--------------------------------------------------------------------------------+
                                       | (REST / OpenAPI / WebSockets)
                                       v
+--------------------------------------------------------------------------------+
|                   ECS: Equipment Control System (TOS Gateway)                  |
|  - Real-time dynamic scheduling, job dispatching, deadlock-free route routing  |
+--------------------------------------------------------------------------------+
               |                                               |
               v                                               v
+-----------------------------+                 +--------------------------------+
|  FMS: Fleet Management      |                 |  CAS: Crane Automation System  |
|  - AGV traffic dispatching  |                 |  - ASC trajectory planning     |
|  - Collision avoidance      |                 |  - Anti-sway & skew control    |
|  - Battery SoC management   |                 |  - Target position profiling   |
+-----------------------------+                 +--------------------------------+
               |                                               |
               +-----------------------+-----------------------+
                                       |
                                       v
+--------------------------------------------------------------------------------+
|                Private 5G Standalone (SA) URLLC Wireless Mesh                  |
|  - 3GPP Rel-16, sub-6 GHz band (n77/n78), dual-redundant core, <5 ms latency   |
+--------------------------------------------------------------------------------+
```

---

## 4. Definitions and vocabulary

| Term | Definition | Common Misuse | Standard / Unit |
| :--- | :--- | :--- | :--- |
| **AGV (Automated Guided Vehicle)** | Unmanned, battery-electric wheeled chassis that transports containers horizontally along predefined paths using transponders or SLAM. | Confusing standard AGVs with Lift-AGVs that possess active lifting capabilities. | ISO 3691-4; km/h, metric tons |
| **Lift-AGV** | An advanced AGV equipped with independent hydraulic or electric lift platforms capable of placing/picking containers onto passive ground stools. | Assuming it requires an external crane to load and unload the vehicle deck. | EN ISO 3691-4; kW, seconds |
| **ITV / Terminal Tractor** | A heavy-duty prime mover (manual or autonomous) designed to pull container roll-trailers or semi-trailers in port yards. | Conflating manual terminal tractors with automated AGVs in civil layout planning. | SAE J3016; km/h, hours |
| **ASC (Automated Stacking Crane)** | A rail-mounted gantry crane operating autonomously within an enclosed container yard block. | Calling an ASC an "RTG"; ASCs are rail-mounted and achieve millimeter-precision positioning. | FEM 1.001, ISO 4301; moves/hour |
| **Cantilever ASC** | An ASC whose main gantry girders extend laterally beyond the runway rails to service external truck lanes or AGV lanes without rail obstruction. | Assuming cantilever ASCs operate identically to end-loading ASCs without side-access. | DIN 15018; meters outreach |
| **Automated e-RTG (ARTG)** | A rubber-tyred gantry crane retrofitted with electric power (busbar or cable reel), automated steering (DGPS/lidar), and remote control. | Believing e-RTGs achieve the same high stacking densities and travel speeds as rail ASCs. | IEEE 1584, ISO 12100; m/min |
| **Dual-Trolley STS Crane** | Ship-to-Shore container crane featuring a primary sea-side trolley and a secondary portal trolley separated by an intermediate platform. | Assuming both trolleys operate over the vessel hold; secondary trolley operates only over the quay. | PIANC WG 208; gross moves/hour |
| **ROS (Remote Operating Station)** | An ergonomic desk equipped with multi-monitor displays, joysticks, and PLC interfaces allowing a human operator to control cranes remotely. | Treating an ROS as a passive video monitor rather than an active deterministic control loop. | ISO 11064; ms glass-to-glass latency |
| **LCPM (Levelized Cost per Move)** | The total discounted lifetime capital and operating expenditures divided by the total discounted container moves over the asset lifecycle. | Miscalculating LCPM by dividing undiscounted OPEX by container moves, ignoring capital costs. | ISO 15686-5; USD ($) per move |
| **SoC (State of Charge)** | The available battery capacity expressed as a percentage of the rated nominal capacity ($0\%\le \text{SoC} \le 100\%$). | Allowing operational SoC to drop below depth-of-discharge limits, accelerating cell degradation. | IEC 62660; percentage (%) |
| **DoD (Depth of Discharge)** | The percentage of battery capacity that has been discharged relative to overall nominal capacity ($\text{DoD} = 100\% - \text{SoC}$). | Equating deep cycling (80–90% DoD) with shallow cycling (20–40% DoD) cell cycle longevity. | IEC 61960; percentage (%) |
| **BSS (Battery Swap Station)** | A mechanized facility where an automated robot extracts a depleted vehicle battery pack and replaces it with a fully charged pack in <5 minutes. | Overlooking the substantial land footprint and spare battery inventory costs required for BSS. | NFPA 855; minutes per swap |
| **Opportunity Fast-Charging** | High-power DC conductive or inductive charging systems (350–600 kW) that charge vehicles during operational pauses (e.g., at buffer waits). | Underestimating peak electrical grid demand charges generated by simultaneous rapid charging. | IEC 61851-23; kW, amperes |
| **URLLC** | Ultra-Reliable Low-Latency Communication service class within 5G mobile networks providing $<1\text{ ms}$ radio latency and $99.999\%$ packet reliability. | Using standard commercial 5G (eMBB) or standard Wi-Fi for deterministic crane safety interlocks. | 3GPP Rel-16/17; ms, packet error rate |
| **GMPH** | Gross Moves Per Hour: total container moves performed by a crane divided by total elapsed operational hours (including breakdown and wait time). | Reporting Net Crane Moves Per Hour (which excludes operational delays) in capital feasibility studies. | Moves per crane hour |
| **Rehandle Ratio** | The average number of non-productive container shifts (shuffles) required to retrieve a target container from a stacked block. | Assuming automated yards eliminate rehandling; without predictive algorithms, rehandles increase. | Non-productive moves / total lifts |
| **LTZ / WTZ** | Landside Transfer Zone (truck interchange) and Waterside Transfer Zone (AGV/quay interchange) at the terminal yard block boundaries. | Designing symmetric LTZ and WTZ layouts ignoring physical differences between AGVs and road trucks. | Physical terminal zones |
| **Concession Agreement** | Long-term lease contract (typically 20 to 50 years) granted by a Port Authority to a private terminal operator governing throughput and capital commitments. | Treating automation investments as perpetual assets without discounting to the concession expiry date. | UNCTAD Port Reform; years |

---

## 5. The mechanics, step by step

### Step 1: Horizontal Transport Kinematics and Decoupled Cycle Modeling

Horizontal transport vehicles circulate in a closed network between the quayside STS cranes and the container yard blocks. The operational cycle of a vehicle consists of four kinematic phases: loaded travel, unloading exchange, empty repositioning travel, and loading exchange.

```text
   +-------------------------------------------------------------------------------+
   |                               AGV CYCLE PHASES                                |
   |                                                                               |
   |   (Quay Crane)  --[Loaded Travel]--> (Yard Block) --[Unloading Exchange]-->   |
   |        ^                                                               |      |
   |        |                                                               v      |
   |   [Loading Exchange] <--[Repositioning Empty]-- (Buffer / Charging Pit)       |
   +-------------------------------------------------------------------------------+
```

The round-trip travel time $T_{\text{travel}}$ over an average one-way routing distance $D_{\text{avg}}$ is governed by vehicle acceleration $a_{\text{acc}}$, steady cruise speed $v_{\text{cruise}}$, deceleration $a_{\text{dec}}$, and cornering speed restrictions:

$$t_{\text{acc}} = \frac{v_{\text{cruise}}}{a_{\text{acc}}}, \quad d_{\text{acc}} = \frac{1}{2} a_{\text{acc}} t_{\text{acc}}^2 = \frac{v_{\text{cruise}}^2}{2 a_{\text{acc}}}$$

$$t_{\text{dec}} = \frac{v_{\text{cruise}}}{a_{\text{dec}}}, \quad d_{\text{dec}} = \frac{v_{\text{cruise}}^2}{2 a_{\text{dec}}}$$

$$t_{\text{cruise}} = \frac{D_{\text{avg}} - (d_{\text{acc}} + d_{\text{dec}})}{v_{\text{cruise}}}$$

$$T_{\text{one-way}} = t_{\text{acc}} + t_{\text{cruise}} + t_{\text{dec}} + \sum_{k=1}^{n_{\text{turns}}} \tau_{\text{turn}, k}$$

where $\tau_{\text{turn}, k}$ is the negotiation delay at curve $k$ (typically 4 to 8 seconds for 90-degree curved guidance paths). Total kinematic travel time for the round-trip is:

$$T_{\text{travel}} = 2 \times T_{\text{one-way}}$$

The critical performance differentiator between standard AGVs, Lift-AGVs, and manual terminal tractors lies in the **exchange handling time** ($T_{\text{exch}}$):

- **Standard AGV (Passive Deck):**
  $$T_{\text{exch}} = T_{\text{wait, STS}} + T_{\text{lift, STS}} + T_{\text{wait, ASC}} + T_{\text{lift, ASC}}$$
  Because the standard AGV cannot lift or ground a container independently, it must remain stationary beneath the crane spreader until the crane executes the hoist and landing maneuver. If the ASC is occupied with internal yard reshuffling, the standard AGV incurs dead-wait time ($T_{\text{wait, ASC}} \approx 60 \text{ to } 180 \text{ seconds}$).
- **Lift-AGV (Active Hydraulic Deck):**
  $$T_{\text{exch}} = T_{\text{drop, buffer}} + T_{\text{pick, buffer}} \approx 15\text{ s} + 15\text{ s} = 30\text{ seconds}$$
  The Lift-AGV drives over a passive quayside or yard ground stool, lowers its hydraulic frame, deposits the container in $<15\text{ seconds}$, and clears the block immediately. Dead-wait time is mathematically eliminated ($T_{\text{wait}} \to 0$).
- **Manual Terminal Tractor (TT / ITV):**
  $$T_{\text{exch}} = T_{\text{position}} + T_{\text{pinning}} + T_{\text{lift}} + T_{\text{driver\_variance}} \approx 90 \text{ to } 150\text{ seconds}$$
  Requires precise reverse parking, human visual alignment, manual chassis twistlock locking/unlocking, and driver behavioral variability.

### Step 2: Battery Thermodynamics, State of Charge (SoC), and Power Consumption

The instantaneous power demand $P_{\text{traction}}(t)$ of an electric AGV operating on level port asphalt is derived from longitudinal vehicle dynamics:

$$P_{\text{traction}}(t) = \frac{1}{\eta_{\text{drive}}} \left[ m_{\text{gross}} \cdot a(t) + m_{\text{gross}} \cdot g \cdot C_{\text{rr}} + \frac{1}{2} \rho_{\text{air}} C_{\text{d}} A_{\text{f}} v(t)^2 \right] v(t)$$

where:
- $m_{\text{gross}} = m_{\text{tare}} + m_{\text{container}}$ (tare mass $\approx 30,000\text{ kg}$; average laden container payload $\approx 15,000 \text{ to } 25,000\text{ kg}$).
- $g = 9.81\text{ m/s}^2$ (gravitational acceleration).
- $C_{\text{rr}} \approx 0.012 \text{ to } 0.018$ (rolling resistance coefficient for heavy solid rubber or industrial radial tires on port concrete/asphalt).
- $\rho_{\text{air}} \approx 1.225\text{ kg/m}^3$ (air density at sea level).
- $C_{\text{d}} \approx 0.70 \text{ to } 0.85$ (aerodynamic drag coefficient of a bluff-body container).
- $A_{\text{f}} \approx 8.5\text{ m}^2$ (frontal cross-sectional area of vehicle plus high-cube container).
- $\eta_{\text{drive}} \approx 0.88 \text{ to } 0.92$ (combined electrical inverter, permanent magnet motor, and planetary gearbox efficiency).

Total energy consumption per container move $E_{\text{move}}$ (in $\text{kWh}$) incorporates traction, auxiliary base loads (sensors, 5G URLLC industrial PC, steer-by-wire, HVAC), and hydraulic lift work:

$$E_{\text{move}} = \int_{0}^{T_{\text{travel}}} P_{\text{traction}}(t) \, dt + P_{\text{aux}} \cdot T_{\text{cycle}} + E_{\text{lift}}$$

where $P_{\text{aux}} \approx 3.5 \text{ to } 5.0\text{ kW}$ continuous, and $E_{\text{lift}} \approx 0.4 \text{ to } 0.6\text{ kWh}$ per lift event. For typical terminal geometries ($D_{\text{avg}} = 700\text{ m}$ one-way), $E_{\text{move}}$ averages $5.5 \text{ to } 7.0\text{ kWh/move}$.

#### Battery Chemistry Selection Matrix

| Battery Parameter | Lithium Iron Phosphate ($\text{LiFePO}_4$ / LFP) | Lithium Titanate Oxide ($\text{Li}_4\text{Ti}_5\text{O}_{12}$ / LTO) |
| :--- | :--- | :--- |
| **Cell Gravimetric Energy Density** | $140 - 170\text{ Wh/kg}$ | $70 - 90\text{ Wh/kg}$ |
| **Maximum Continuous C-Rate** | $1.0\text{C} - 2.0\text{C}$ (Fast charge: 30–60 min) | $4.0\text{C} - 6.0\text{C}$ (Ultra-fast charge: 10–15 min) |
| **Cycle Life to 80% Retained Capacity** | $3,500 - 5,000\text{ cycles}$ (@ 80% DoD) | $15,000 - 25,000\text{ cycles}$ (@ 80% DoD) |
| **Thermal Runaway Onset Temperature** | $\approx 270^\circ\text{C}$ (Highly stable) | $>300^\circ\text{C}$ (Extremely safe, non-graphitic anode) |
| **Capital Cost per Installed kWh** | $\$220 - \$320\text{ /kWh}$ | $\$650 - \$950\text{ /kWh}$ |
| **Operational Paradigm** | Opportunity charging / Battery Swap | Continuous high-power opportunity charging |

To prevent premature cell capacity loss, the operational State of Charge ($\text{SoC}$) is constrained within a cyclic operational window $[\text{SoC}_{\text{min}}, \text{SoC}_{\text{max}}]$, typically $[20\%, 80\%]$, yielding usable battery energy:

$$E_{\text{usable}} = B_{\text{rated}} \times (\text{SoC}_{\text{max}} - \text{SoC}_{\text{min}})$$

The maximum number of container moves executed between charge events is:

$$k_{\text{moves}} = \left\lfloor \frac{E_{\text{usable}}}{E_{\text{move}}} \right\rfloor$$

The required conductive fast-charging time $T_{\text{charge}}$ at charger station power $P_{\text{charger}}$ (typically $350 \text{ to } 450\text{ kW}$) with charger conversion efficiency $\eta_{\text{ch}} \approx 0.92$ is:

$$T_{\text{charge}} = \frac{E_{\text{usable}}}{P_{\text{charger}} \cdot \eta_{\text{ch}}} \times 3600 \quad [\text{seconds}]$$

### Step 3: Closed Queueing Network Fleet Sizing under Charging Constraints

To dimension the horizontal transport fleet, the terminal is modeled as a **closed queueing network** with $N_{\text{AGV}}$ circulating customers. Containers arrive at the quayside according to the combined production rate of $N_{\text{STS}}$ operational Ship-to-Shore cranes, each producing gross moves at mean rate $\mu_{\text{STS}}$:

$$\lambda_{\text{peak}} = N_{\text{STS}} \times \mu_{\text{STS}}$$

Each container move incurs an effective vehicle service duration $T_{\text{eff}}$ comprising round-trip travel time, buffer handling, and amortized battery charging overhead:

$$T_{\text{eff}} = T_{\text{travel}} + T_{\text{exch}} + \frac{1}{k_{\text{moves}}} \left( T_{\text{charge}} + W_{\text{queue, ch}} \right)$$

where $W_{\text{queue, ch}}$ is the expected queueing delay at the fast-charging stations.

The minimum theoretical number of active vehicles $N_{\text{active}}$ required to support quayside throughput without starving the STS cranes is:

$$N_{\text{active}} = \lambda_{\text{peak}} \times \frac{T_{\text{eff}}}{3600}$$

Accounting for scheduled preventative maintenance, unscheduled breakdowns, tire wear replacements, and software calibration, the total required fleet size $N_{\text{fleet}}$ incorporates operational fleet availability $\eta_{\text{avail}}$ (typically $85\% \text{ to } 90\%$) and an operational traffic congestion contingency factor $\alpha_{\text{buffer}} \approx 1.05 \text{ to } 1.10$:

$$N_{\text{fleet}} = \left\lceil \frac{N_{\text{active}} \cdot \alpha_{\text{buffer}}}{\eta_{\text{avail}}} \right\rceil$$

#### Closed Queueing Network Formulation (Mean Value Analysis - MVA)

For detailed terminal layout planning, the circulation of $N$ autonomous vehicles across $M$ service stations (quayside buffer, waterside block exchange, charging bays, and travel lanes) is modeled as a product-form Gordon-Newell closed queueing network. Let:
- $V_i$ represent the visit ratio of a vehicle to station $i \in \{1, \dots, M\}$ relative to a reference quayside move.
- $S_i$ represent the mean service time at station $i$ (with $S_{\text{travel}} = T_{\text{travel}}$, $S_{\text{quay}} = T_{\text{exch, quay}}$, $S_{\text{yard}} = T_{\text{exch, yard}}$, and $S_{\text{ch}} = T_{\text{charge}}$).

Using Mean Value Analysis (MVA), the system is evaluated recursively for population $n = 1, 2, \dots, N$:

1. **Expected Residence Time at Station $i$:**
   $$R_i(n) = \begin{cases} S_i \cdot [1 + Q_i(n-1)], & \text{single-server station} \\ S_i, & \text{infinite-server (pure travel delay)} \\ S_i \cdot \left[ 1 + \frac{Q_i(n-1)}{c_i} \right], & \text{multiserver station with } c_i \text{ parallel servers} \end{cases}$$

2. **System Throughput (Move Production Rate):**
   $$\Lambda(n) = \frac{n}{\sum_{i=1}^{M} V_i \cdot R_i(n)}$$

3. **Mean Queue Length at Station $i$:**
   $$Q_i(n) = \Lambda(n) \cdot V_i \cdot R_i(n)$$

The fleet size $N^*$ is determined as the smallest integer population $n$ such that quayside throughput achieves parity with crane capacity: $\Lambda(n) \ge \lambda_{\text{peak}}$.

#### Charger Station Sizing via Multiserver Queuing ($M/M/c$)

The demand rate for charging sessions $\lambda_{\text{ch}}$ entering the charging station pool is:

$$\lambda_{\text{ch}} = \frac{\lambda_{\text{peak}}}{k_{\text{moves}}} \quad [\text{charging events / hour}]$$

Mean charging service rate per charger stall is $\mu_{\text{ch}} = \frac{3600}{T_{\text{charge}}} \text{ [charges/hour]}$. Offered charging workload intensity in Erlangs is:

$$a_{\text{ch}} = \frac{\lambda_{\text{ch}}}{\mu_{\text{ch}}}$$

For stability, the number of automated charging stalls $c_{\text{ch}}$ must satisfy $c_{\text{ch}} > a_{\text{ch}}$. The probability of an AGV finding all charging stations occupied, requiring waiting, is determined by the Erlang-C formula:

$$P_{\text{wait}} = C(c_{\text{ch}}, a_{\text{ch}}) = \frac{\frac{a_{\text{ch}}^{c_{\text{ch}}}}{c_{\text{ch}}!} \left( \frac{c_{\text{ch}}}{c_{\text{ch}} - a_{\text{ch}}} \right)}{\sum_{k=0}^{c_{\text{ch}}-1} \frac{a_{\text{ch}}^k}{k!} + \frac{a_{\text{ch}}^{c_{\text{ch}}}}{c_{\text{ch}}!} \left( \frac{c_{\text{ch}}}{c_{\text{ch}} - a_{\text{ch}}} \right)}$$

Expected queue waiting time prior to plugging into a charger is:

$$W_{\text{queue, ch}} = \frac{P_{\text{wait}}}{c_{\text{ch}} \mu_{\text{ch}} - \lambda_{\text{ch}}} \times 3600 \quad [\text{seconds}]$$

The number of fast chargers $c_{\text{ch}}$ is dimensioned such that $P_{\text{wait}} \le 0.05$ (less than $5\%$ probability of queueing upon arriving with low SoC).

### Step 4: Stacking Yard Kinematics and Crane Architecture Selection

Container terminals select between three automated stacking configurations:

```text
(A) END-LOADING ASC BLOCKS               (B) CANTILEVER ASC BLOCKS
+------------------------------------+   +------------------------------------+
| WTZ |                              |   |         [Cantilever Arm]           |
| (AGV|        STACKING AREA         |   |----+--------------------------+----|
|Exch)|  (Perpendicular to Quay)     |   |AGV |                          |Truck|
|     |                              |   |Lane|      STACKING AREA       |Lane|
|-----+------------------------------|   |    |  (Perpendicular/Parallel)|    |
|                              | LTZ |   |----+--------------------------+----|
|        STACKING AREA         |(OTR |   |         [Cantilever Arm]           |
|                              |Truck|   +------------------------------------+
+------------------------------------+   
```

1. **End-Loading ASCs:** Stacks are aligned perpendicular to the quay. The ASC travels longitudinally down the block rails to pick or place containers at the extreme block ends: the Waterside Transfer Zone (WTZ) at one end, and the Landside Transfer Zone (LTZ) at the other.
   - *Advantage:* Highly compact footprint; zero side roads between blocks; maximum storage density ($1,100 \text{ to } 1,400\text{ TEU/hectare}$).
   - *Disadvantage:* Long gantry travel distances; potential gantry interference if two cranes share the same rail track.
2. **Cantilever ASCs:** Gantry girders extend laterally over one or both rail tracks, creating an outreach cantilever arm (typically 6 to 9 meters).
   - *Advantage:* AGVs and road trucks drive directly beneath the cantilever arm parallel to the stack. The ASC can exchange containers at any longitudinal bay along the stack perimeter, drastically reducing gantry travel cycle time.
   - *Disadvantage:* Broader land footprint required for perimeter truck lanes; higher crane structural weight and wheel loads.
3. **Automated Hybrid e-RTGs (ARTGs):** Rubber-tyred gantries electrified via conductor busbars or cable reels, operating along fixed virtual tracks using DGPS and ground transponders.
   - *Advantage:* Minimal subterranean civil infrastructure; lower initial CAPEX; ideal for brownfield conversion where existing asphalt pavement cannot support rail beams.
   - *Disadvantage:* Lower stacking density ($750 \text{ to } 950\text{ TEU/ha}$); lower gantry positioning speed; complex automated lane-changing mechanics.

#### Subterranean Runway Wheel Load Mechanics

Rail-mounted ASCs impose severe concentrated dynamic wheel point loads on civil foundations. For an 8-wheel or 16-wheel gantry bogie supporting a fully laden ASC ($m_{\text{crane}} \approx 280\text{ to } 350\text{ metric tons}$, rated spreader load $m_{\text{load}} = 40 \text{ to } 65\text{ metric tons}$ under twin-lift), peak static wheel load $P_{\text{wheel, static}}$ is:

$$P_{\text{wheel, static}} = \frac{(m_{\text{crane}} + m_{\text{trolley}} + m_{\text{rated\_load}}) \cdot g}{n_{\text{wheels}}}$$

Factoring in wind lateral shear forces ($F_{\text{wind}}$), skewing loads, and dynamic hoisting impact factor ($\phi_{\text{dyn}} \approx 1.25 \text{ to } 1.40$ according to FEM 1.001):

$$P_{\text{wheel, dynamic}} = \phi_{\text{dyn}} \cdot P_{\text{wheel, static}} + \frac{M_{\text{wind, overturning}}}{L_{\text{track}} \cdot (n_{\text{wheels}} / 2)}$$

In deep alluvial port soils, dynamic wheel point loads exceeding $350 \text{ to } 550\text{ kN per wheel}$ cannot be supported by shallow sub-base asphalt. They require **continuous reinforced concrete runway beams** supported by driven pre-stressed concrete or steel pipe piles ($30 \text{ to } 45\text{ meters}$ depth down to dense bearing strata), with continuous heavy crane rail (DIN 536 type A100 or A120 rail profiles).

#### Automated Crane Anti-Sway and Trajectory Profiling

To achieve rated autonomous cycle speeds without human cabin intervention, ASCs and automated STS cranes utilize closed-loop electronic **anti-sway control algorithms**. A container suspended from a trolley by hoist wire ropes of instantaneous length $L(t)$ behaves dynamically as a spherical pendulum with natural period $T_{\text{pendulum}}$ and natural angular frequency $\omega_n(t)$:

$$T_{\text{pendulum}}(t) = 2\pi \sqrt{\frac{L(t)}{g}}, \quad \omega_n(t) = \sqrt{\frac{g}{L(t)}}$$

During trolley acceleration and deceleration, kinematic inertia induces load sway angle $\theta(t)$ governed by:

$$\ddot{\theta}(t) + 2 \zeta \omega_n(t) \dot{\theta}(t) + \omega_n^2(t) \theta(t) = -\frac{\ddot{x}_{\text{trolley}}(t)}{L(t)}$$

where $\zeta \approx 0.005$ is structural air damping. To cancel residual sway upon arrival at the target yard bay ($x_{\text{target}}$), the crane drive controller applies an **Input Shaping (Zero Vibration - ZV)** filter to the trolley acceleration reference signal $\ddot{x}_{\text{ref}}(t)$. The ZV shaper convolves the target command with two impulse vectors separated by half the pendulum oscillation period:

$$I_1 = \frac{1}{1 + K}, \quad t_1 = 0; \qquad I_2 = \frac{K}{1 + K}, \quad t_2 = \frac{T_{\text{pendulum}}}{2} = \pi \sqrt{\frac{L}{g}}$$

where $K = e^{-\frac{\zeta \pi}{\sqrt{1 - \zeta^2}}} \approx 1.0$. By tuning trolley deceleration timing to inject an opposing kinematic impulse exactly $T_{\text{pendulum}}/2$ later, the container arrives at the target slot with near-zero residual oscillation ($\theta_{\text{residual}} < 0.15^\circ$), permitting immediate twistlock disengagement or soft-landing.

### Step 5: Remote Operating Station (ROS) Sizing and Tele-Operation Ergonomics

Modern ASCs execute $90\% \text{ to } 95\%$ of all stacking moves fully autonomously using laser range finders (lidar), 3D time-of-flight cameras, and target-position profiling PLCs. Human operator intervention from an ROS console is required only during high-risk exception events:
1. Landing a container onto an external Over-the-Road (OTR) truck chassis twistlocks ($T_{\text{ros, land}} \approx 30 \text{ to } 50\text{ seconds}$).
2. Clearing twistlock latching/unlatching errors or optical character recognition (OCR) read mismatches.
3. Handling damaged containers or non-standard flat racks / open tops.

#### ROS Operator Multiserver Queue Dimensioning

Let $p_{\text{int}}$ represent the probability that a crane move requires human tele-operation intervention ($p_{\text{int}} \approx 0.08 \text{ to } 0.12$ in mature automated terminals). Total arrival rate of intervention requests to the central ROS control room from $N_{\text{ASC}}$ operational cranes is:

$$\lambda_{\text{ros}} = \sum_{j=1}^{N_{\text{ASC}}} \mu_{\text{ASC}, j} \cdot p_{\text{int}} \quad [\text{interventions / hour}]$$

With mean tele-operation service duration $1/\mu_{\text{ros}}$ (typically 45 seconds $\implies \mu_{\text{ros}} = 80\text{ interventions/hour/operator}$), the offered human tele-operation workload in Erlangs is:

$$a_{\text{ros}} = \frac{\lambda_{\text{ros}}}{\mu_{\text{ros}}}$$

The number of active ROS operator desks $c_{\text{ros}}$ is dimensioned using the Erlang-C formula to satisfy the strict performance criterion that the probability of an automated crane waiting for a human operator exceeds 1 second is negligible ($P(W_{\text{ros}} > 1\text{ s}) < 0.01$). This yields typical operational staffing ratios of:
- **1 ROS operator per 4 to 8 ASCs** during peak vessel loading/unloading operations.
- **1 ROS operator per 12 to 16 ASCs** during off-peak night yard housekeeping and internal reshuffling.

#### Tele-Operation Latency Budget (<50 ms Glass-to-Glass)

To preserve operator situational awareness, prevent motion sickness, and avoid kinetic "spreader oscillation over-correction," the round-trip tele-operation system must satisfy an end-to-end latency budget $\le 50\text{ ms}$:

```text
[ Camera Capture ] ---> [ H.265/AV1 Encode ] ---> [ 5G URLLC Radio ] ---> [ Edge Decode ] ---> [ ROS Display ]
   (3-5 ms)                 (8-12 ms)                 (2-4 ms)               (6-10 ms)             (8-16 ms)
```

$$\tau_{\text{glass-to-glass}} = \tau_{\text{sensor}} + \tau_{\text{encode}} + \tau_{\text{network}} + \tau_{\text{decode}} + \tau_{\text{display}} \le 45\text{ ms}$$

### Step 6: Subterranean and Civil Engineering Retrofit Cost Breakdown

Brownfield conversion of an operational manual terminal into an automated facility requires extensive subterranean civil reconstruction. The civil capital cost schedule must account for:

1. **Quayside Infrastructure Upgrades:** Strengthening the seaside crane runway beam and tieback anchor walls to support heavier dual-trolley automated STS cranes (wheel loads scaling from $800\text{ kN/m}$ to $>1,400\text{ kN/m}$).
2. **ASC Runway Beams and Piling:** Installing parallel rail tracks along each yard block. Cost per meter of double-track runway includes pre-cast concrete beams, continuous elastomeric sole plates, galvanized tie rods, and vertical piles spaced at 4-to-6-meter centers:
   $$C_{\text{runway}} = L_{\text{block}} \times 2 \times c_{\text{track\_linear\_cost}}$$
   where $c_{\text{track\_linear_cost}} \approx \$1,800 \text{ to } \$2,600\text{ /meter}$.
3. **Pavement Reconstruction for AGVs:** Heavy industrial roller-compacted concrete (RCC) with high surface flatness (DIN 18202 Table 3 Group 4) to ensure millimeter-accurate guidance sensor tracking:
   $$C_{\text{pavement}} = A_{\text{AGV\_grid}} \times c_{\text{RCC\_sqm}}$$
   where $c_{\text{RCC\_sqm}} \approx \$120 \text{ to } \$180\text{ /m}^2$.
4. **Medium-Voltage (MV) Electrification Grid:** High-capacity $11\text{ kV} \text{ or } 33\text{ kV}$ ring networks, SF6 gas-insulated switchgear, step-down harmonic transformer substations ($11\text{ kV} \to 400\text{ V} / 690\text{ V}$), and underground trenching to supply high-power opportunity chargers and crane busbars.
5. **Private 5G URLLC Mesh Infrastructure:** Deployment of redundant gNodeB macro-cells, distributed antenna systems (DAS) along light towers to eliminate container canyon shadowing, and localized on-premise User Plane Function (UPF) edge computing servers.

### Step 7: Capital Budgeting, DCF, and Levelized Cost per Move (LCPM)

Capital budgeting for terminal automation requires a multi-decade discounted cash flow (DCF) framework spanning the concession period $T_{\text{horizon}}$ (typically 20 to 30 years).

#### 1. Net Present Value (NPV)
$$NPV = -CAPEX_0 + \sum_{t=1}^{T_{\text{horizon}}} \frac{\Delta OPEX_t + \Delta REV_t - CAPEX_{\text{sustaining}, t}}{(1 + WACC)^t} + \frac{V_{\text{salvage}}}{(1 + WACC)^{T_{\text{horizon}}}}$$

where $WACC$ is the Weighted Average Cost of Capital:

$$WACC = \left( \frac{E}{V} \cdot K_{\text{e}} \right) + \left( \frac{D}{V} \cdot K_{\text{d}} \cdot (1 - \tau_{\text{tax}}) \right)$$

#### 2. Levelized Cost per Move (LCPM)
The Levelized Cost per Move ($LCPM$) expresses the minimum cost-recovery revenue required per container move to amortize all lifetime capital outlays and cover operational expenditures:

$$LCPM = \frac{CAPEX_0 + \sum_{t=1}^{T_{\text{horizon}}} \frac{OPEX_t + CAPEX_{\text{sustaining}, t}}{(1 + WACC)^t}}{\sum_{t=1}^{T_{\text{horizon}}} \frac{Q_t}{(1 + WACC)^t}} \quad [\$ / \text{move}]$$

where $Q_t$ is the container move volume processed in year $t$.

### Step 8: Labor Substitution Economics and Union Transition Dynamics

The primary economic justification for container terminal automation is the radical reduction of direct operating labor hours per container move.

#### Crane Gang Labor Structure Comparison

```text
+------------------------------------+   +------------------------------------+
|       MANUAL CRANE GANG            |   |     AUTOMATED CRANE GANG           |
|         (12 - 14 Workers)          |   |          (3 - 4 Workers)           |
|                                    |   |                                    |
| - 1 STS Crane Operator (in cabin)  |   | - 0.5 Supervisory STS Specialist   |
| - 1 Relief Crane Operator          |   | - 0.0 AGV Drivers (Autonomous)     |
| - 6 Terminal Tractor Drivers (ITV) |   | - 0.25 ROS Operator (1 per 4 ASCs) |
| - 2 Yard RTG / Top-Pick Operators  |   | - 2.0 Twistlock Deck Technicians   |
| - 2 Pin-Men / Ground Lashers       |   | - 0.5 Automation Systems Tech      |
| - 1 Hatch Tender / Signal-man      |   |                                    |
| - 1 Deck-man / Clerk               |   |                                    |
+------------------------------------+   +------------------------------------+
```

In high-wage unionized jurisdictions (e.g., US West Coast ILWU, US East/Gulf Coast ILA, Northern European ports), a fully loaded crane gang hour costs $\$1,200 \text{ to } \$2,200\text{ /hour}$ including base wages, overtime, healthcare, pension contributions, and container royalties. Transitioning to automated operations yields a **65% to 75% reduction in direct operational Full-Time Equivalents (FTEs)**.

However, the capital budget must explicitly incorporate **labor transition friction costs**:
- **Voluntary Severance Packages (VSP) / Early Retirement:** Typically $\$150,000 \text{ to } \$300,000$ per displaced senior worker.
- **Retraining Programs:** Reskilling longshore labor into certified mechatronics technicians, PLC diagnostics specialists, and ROS supervisory controllers.
- **Container Royalty and Fund Guarantees:** Contractual obligations negotiated in collective bargaining agreements (e.g., ILA automation royalty funds) requiring fixed royalty payments per container move to protect displaced union health and pension funds.

---

## 6. Worked example

### Problem Statement

A major marine terminal concessionaire is evaluating the full brownfield automation conversion of a 1,200-meter quay facility handling an annual throughput $Q = 1,500,000\text{ container moves/year}$ ($2,400,000\text{ TEU/year}$). The concession has exactly 20 years remaining. The terminal operator's $WACC = 8.0\%$.

The terminal operates 6 Super Post-Panamax dual-trolley STS cranes along the quay line, each delivering an average peak rate $\mu_{\text{STS}} = 32\text{ moves/gross crane hour}$, generating a combined quayside peak demand:

$$\lambda_{\text{peak}} = 6 \times 32 = 192\text{ moves/hour}$$

The engineering team must:
1. Size the battery-electric Lift-AGV fleet and 350 kW DC fast-charging infrastructure.
2. Size the Remote Operating Station (ROS) tele-operation console pool.
3. Establish total project CAPEX across civil retrofits, cranes, vehicles, and IT.
4. Calculate lifetime annual OPEX for the manual baseline versus the automated terminal.
5. Compute the project Net Present Value (NPV), Internal Rate of Return (IRR), and Levelized Cost per Move (LCPM).
6. Determine the volume crossover threshold where automation achieves financial parity with manual operations.

---

### Step-by-Step Calculation

#### 1. Horizontal Transport Kinematics and Battery Fleet Sizing

- **Travel Geometry:** One-way average distance $D_{\text{avg}} = 700\text{ m}$; round-trip distance $D_{\text{RT}} = 1,400\text{ m}$.
- **Kinematic Profile:** Cruising speed $v = 4.5\text{ m/s}$ ($16.2\text{ km/h}$); acceleration $a = 0.5\text{ m/s}^2$; deceleration $a = 0.5\text{ m/s}^2$.
  $$t_{\text{acc}} = \frac{4.5}{0.5} = 9.0\text{ s}; \quad d_{\text{acc}} = \frac{1}{2}(0.5)(9.0)^2 = 20.25\text{ m}$$
  $$t_{\text{dec}} = 9.0\text{ s}; \quad d_{\text{dec}} = 20.25\text{ m}$$
  $$d_{\text{cruise}} = 1,400 - (20.25 + 20.25) = 1,359.5\text{ m}$$
  $$t_{\text{cruise}} = \frac{1,359.5}{4.5} = 302.11\text{ s}$$
  Two 90-degree curve turns per round trip @ $5.0\text{ s}$ each $\implies 10.0\text{ s}$.
  $$T_{\text{travel}} = 9.0 + 302.11 + 9.0 + 10.0 = 330.11\text{ seconds}$$

- **Decoupled Exchange Time:** Lift-AGV drops container on passive stool in $15\text{ s}$ and picks up container in $15\text{ s}$:
  $$T_{\text{exch}} = 15.0 + 15.0 = 30.0\text{ seconds}$$
  $$T_{\text{cycle, base}} = 330.11 + 30.0 = 360.11\text{ seconds} = 6.002\text{ minutes}$$

- **Energy Consumption and Charging:**
  - Rated battery pack capacity: $B_{\text{rated}} = 280\text{ kWh}$ Lithium Iron Phosphate ($\text{LiFePO}_4$).
  - Usable SoC window: $20\% \text{ to } 80\% \implies \Delta \text{SoC} = 60\%$.
    $$E_{\text{usable}} = 280 \times 0.60 = 168.0\text{ kWh}$$
  - Specific energy consumption: $3.2\text{ kWh/km} \times 1.4\text{ km} + 1.8\text{ kWh (aux/lift)} = 6.28\text{ kWh/move}$.
  - Moves executed per full charge cycle:
    $$k_{\text{moves}} = \frac{168.0}{6.28} = 26.75\text{ moves}$$
  - Conductive opportunity charging at $P_{\text{ch}} = 350\text{ kW}$ with efficiency $\eta_{\text{ch}} = 0.92$:
    $$P_{\text{effective}} = 350 \times 0.92 = 322.0\text{ kW}$$
    $$T_{\text{charge}} = \frac{168.0\text{ kWh}}{322.0\text{ kW}} \times 3,600 = 1,878.26\text{ seconds} = 31.30\text{ minutes} = 0.5217\text{ hours}$$
  - Amortized charging overhead per container move:
    $$\tau_{\text{ch, amortized}} = \frac{1,878.26\text{ s}}{26.75\text{ moves}} = 70.21\text{ seconds/move}$$
  - Total effective cycle time per move:
    $$T_{\text{eff}} = T_{\text{travel}} + T_{\text{exch}} + \tau_{\text{ch, amortized}} = 330.11 + 30.0 + 70.21 = 430.32\text{ seconds} = 0.1195\text{ hours}$$

- **Fleet Sizing Calculation:**
  $$N_{\text{active}} = \lambda_{\text{peak}} \times T_{\text{eff}} = 192\text{ moves/hr} \times 0.1195\text{ hours} = 22.94\text{ vehicles}$$
  With fleet operational availability $\eta_{\text{avail}} = 88\%$:
  $$N_{\text{fleet}} = \left\lceil \frac{22.94}{0.88} \right\rceil = \lceil 26.07 \rceil = 27\text{ Lift-AGVs}$$

- **Fast Charger Stall Sizing:**
  $$\lambda_{\text{ch}} = \frac{\lambda_{\text{peak}}}{k_{\text{moves}}} = \frac{192}{26.75} = 7.1776\text{ charging arrivals/hour}$$
  Service rate per charger stall $\mu_{\text{ch}} = 1 / 0.5217\text{ hr} = 1.9168\text{ charges/hour}$.
  Offered workload:
  $$a_{\text{ch}} = \frac{7.1776}{1.9168} = 3.7446\text{ Erlangs}$$
  Testing multiserver capacity $c_{\text{ch}}$:
  - For $c = 5$ chargers: $a/c = 3.7446/5 = 0.7489$. Erlang-C wait probability $P_{\text{wait}} = 24.2\%$.
  - For $c = 6$ chargers: $a/c = 3.7446/6 = 0.6241$. Erlang-C wait probability $P_{\text{wait}} = 6.8\%$.
  - For $c = 7$ chargers: $a/c = 3.7446/7 = 0.5349$. Erlang-C wait probability $P_{\text{wait}} = 1.6\%$.
  Selecting $c_{\text{ch}} = 6\text{ fast-charging stalls}$ satisfies operational queue threshold targets with mean waiting time under $2.5\text{ minutes}$.

---

#### 2. Stacking Yard Cranes and ROS Pool Dimensioning

- **Yard Crane Sizing:** 12 perpendicular yard blocks. To support peak quayside throughput (192 moves/hr) plus landside gate truck appointments (estimated 96 moves/hr) with peak total demand of $288\text{ moves/hr}$, each block is equipped with 2 Cantilever ASCs:
  $$N_{\text{ASC}} = 12 \times 2 = 24\text{ Cantilever ASCs}$$
  Average production required per ASC $= 288 / 24 = 12.0\text{ moves/hour}$, well within rated autonomous capacity ($28 \text{ to } 32\text{ moves/hr}$).

- **ROS Tele-Operation Sizing:**
  - Probability of move requiring manual ROS intervention: $p_{\text{int}} = 8.0\%$.
  - Total arrival rate to ROS operator pool:
    $$\lambda_{\text{ros}} = 288\text{ moves/hour} \times 0.08 = 23.04\text{ interventions/hour}$$
  - Mean intervention duration: $1/\mu_{\text{ros}} = 45\text{ seconds} = 0.0125\text{ hours} \implies \mu_{\text{ros}} = 80.0\text{ interventions/hour/desk}$.
  - Offered tele-operation workload:
    $$a_{\text{ros}} = \frac{23.04}{80.0} = 0.288\text{ Erlangs}$$
  - Dimensioning with $c_{\text{ros}} = 3\text{ active ROS consoles}$:
    Traffic intensity per desk $\rho = 0.288 / 3 = 0.096$.
    Erlang-C delay probability $P_{\text{wait}} \approx 0.0028$ ($<0.3\%$). Crane waiting for an operator is virtually zero. Total control room desks installed: 4 (3 operational + 1 hot standby).

---

#### 3. Capital Expenditure (CAPEX) Schedule

| Subsystem Component | Scope / Physical Sizing | Unit Cost | Total Cost ($M) |
| :--- | :--- | :--- | ---: |
| **Quay Wall & STS Retrofits** | 6 Cranes: dual-trolley conversion & rail tiebacks | $\$8.0\text{M /crane}$ | $\$48.0\text{M}$ |
| **ASC Runway Beams & Piling** | 12 Blocks $\times 2$ tracks $\times 350\text{ m} = 8,400\text{ m}$ | $\$2,200\text{ /m}$ | $\$18.48\text{M}$ |
| **Pavement & Civil Infrastructure** | Sub-base remediation, RCC paving, drainage | Lump sum | $\$110.0\text{M}$ |
| **Electrical Substation & Grid** | $33\text{ kV}$ substations, transformers, trenching | Lump sum | $\$55.0\text{M}$ |
| **Private 5G SA URLLC Network** | Redundant core, 8 gNodeB towers, fiber mesh | Lump sum | $\$6.5\text{M}$ |
| **Automated Stacking Cranes** | 24 Cantilever ASCs (twin-box girder, electric) | $\$5.0\text{M /unit}$ | $\$120.0\text{M}$ |
| **Lift-AGV Fleet** | 27 Battery-electric Lift-AGVs ($280\text{ kWh}$ LFP) | $\$900,000\text{ /unit}$ | $\$24.3\text{M}$ |
| **Fast-Charging Stations** | 6 Conductive $350\text{ kW}$ DC opportunity chargers | $\$400,000\text{ /unit}$ | $\$2.4\text{M}$ |
| **ROS Control Room & Ergonomics** | 4 Ergonomic ROS stations, server core, displays | Lump sum | $\$3.2\text{M}$ |
| **TOS / ECS Software Integration** | Navis N4 / TEAMS ECS, emulation & tuning | Lump sum | $\$28.0\text{M}$ |
| **Union Severance & Retraining** | 220 displaced workers buyout/retraining fund | $\$180,000\text{ /worker}$ | $\$39.6\text{M}$ |
| **Contingency & EPC Management** | Engineering, procurement, construction ($5\%$) | Fixed allowance | $\$22.42\text{M}$ |
| **TOTAL INITIAL CAPEX ($CAPEX_0$)**| | | **$\$477.90\text{M}$** |

---

#### 4. Operational Expenditure (OPEX) Comparison

- **Manual Terminal Baseline (1,500,000 moves/year):**
  - Direct Operating Labor: 14-man gang @ $\$1,850\text{ /gang-hour}$. At $32\text{ moves/gang-hour} \implies \$57.81\text{ /move}$. Factoring in gate, rail, and yard support labor across all shifts $\implies \$32.50\text{ /move}$ blended baseline.
  - Equipment Fuel (Diesel): $3.8\text{ liters diesel/move} \times \$1.20\text{ /liter} = \$4.56\text{ /move}$.
  - Maintenance & Spare Parts: Manual equipment wear $\implies \$7.24\text{ /move}$.
  - Insurance, Facilities, and Overhead: $\$3.50\text{ /move}$.
  - **Total Manual OPEX:** $\$47.80\text{ /move} \implies \text{Annual OPEX} = 1.5\text{M} \times \$47.80 = \mathbf{\$71.70\text{M/year}}$.

- **Automated Terminal (1,500,000 moves/year):**
  - Direct Operating Labor: Supervisory ROS operators, exception handling, gate automation clerks $\implies \$9.20\text{ /move}$.
  - Electrical Power: $16.0\text{ kWh total/move} \times \$0.15\text{ /kWh} = \$2.40\text{ /move}$.
  - Robotics, Mechatronics, and Software Maintenance: Specialized ASC/AGV maintenance $\implies \$9.80\text{ /move}$.
  - Software Licensing, 5G SLA, and Cloud Services: $\$2.50\text{ /move}$.
  - **Total Automated OPEX:** $\$23.90\text{ /move} \implies \text{Annual OPEX} = 1.5\text{M} \times \$23.90 = \mathbf{\$35.85\text{M/year}}$.

- **Annual Operational Cost Savings:**
  $$\Delta OPEX = \$71.70\text{M} - \$35.85\text{M} = \mathbf{\$35.85\text{M/year}}$$

---

#### 5. Discounted Cash Flow (DCF), NPV, and LCPM Results

Over the 20-year concession horizon at $WACC = 8.0\%$:
- The uniform series present worth annuity factor is:
  $$PVIFA(8\%, 20) = \frac{1 - (1 + 0.08)^{-20}}{0.08} = \frac{1 - 0.21455}{0.08} = 9.8181$$
- **Present Value of OPEX Savings:**
  $$PV(\Delta OPEX) = \$35.85\text{M} \times 9.8181 = \$351.98\text{M}$$
- **Net Present Value (NPV):**
  $$NPV = -CAPEX_0 + PV(\Delta OPEX) = -\$477.90\text{M} + \$351.98\text{M} = \mathbf{-\$125.92\text{M}}$$
  *Note:* At $1.5\text{M moves/year}$, the project exhibits negative NPV against an uncapitalized manual status quo because the high initial capital outlay cannot be fully amortized by labor savings alone.

- **Levelized Cost per Move (LCPM):**
  - Discounted Move Volume over 20 years:
    $$PV(Q) = 1,500,000 \times 9.8181 = 14,727,150\text{ discounted moves}$$
  - Present Value of Automated Lifetime Costs (Initial CAPEX + 20-yr OPEX):
    $$PV(\text{Cost}_{\text{auto}}) = \$477.90\text{M} + (\$35.85\text{M} \times 9.8181) = \$477.90\text{M} + \$351.98\text{M} = \$829.88\text{M}$$
    $$LCPM_{\text{auto}} = \frac{\$829,880,000}{14,727,150} = \mathbf{\$56.35\text{ /move}}$$
  - Manual Terminal Lifetime Costs (assuming sustaining maintenance CAPEX of $\$80.0\text{M}$ PV):
    $$PV(\text{Cost}_{\text{manual}}) = \$80.00\text{M} + (\$71.70\text{M} \times 9.8181) = \$80.00\text{M} + \$703.96\text{M} = \$783.96\text{M}$$
    $$LCPM_{\text{manual}} = \frac{\$783,960,000}{14,727,150} = \mathbf{\$53.23\text{ /move}}$$

---

#### 6. Volume Sensitivity and Operational Leverage Crossover

Because automation shifts cost structure from variable OPEX to fixed CAPEX, its financial viability depends heavily on container volume:

| Annual Moves ($Q$) | Total Discounted Moves ($PV(Q)$) | $PV(\text{Cost}_{\text{auto}})$ | $LCPM_{\text{auto}}$ | $PV(\text{Cost}_{\text{manual}})$ | $LCPM_{\text{manual}}$ | Incremental NPV | Standalone NPV |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **1,200,000** | $11.78\text{M}$ | $\$759.48\text{M}$ | **$\$64.46$** | $\$643.17\text{M}$ | **$\$54.59$** | $-\$116.32\text{M}$ | $-\$196.32\text{M}$ |
| **1,500,000** | $14.73\text{M}$ | $\$829.88\text{M}$ | **$\$56.35$** | $\$783.96\text{M}$ | **$\$53.23$** | $-\$45.92\text{M}$ | $-\$125.92\text{M}$ |
| **1,695,737** | $16.65\text{M}$ | $\$875.87\text{M}$ | **$\$52.60$** | $\$875.87\text{M}$ | **$\$52.61$** | **$\$0.00\text{M}$** | $-\$80.00\text{M}$ |
| **2,200,000** | $21.60\text{M}$ | $\$994.13\text{M}$ | **$\$46.03$** | $\$1,112.47\text{M}$ | **$\$51.50$** | **$+\$118.34\text{M}$** | **$+\$38.34\text{M}$** |
| **2,500,000** | $24.55\text{M}$ | $\$1,064.53\text{M}$ | **$\$43.37$** | $\$1,253.27\text{M}$ | **$\$51.06$** | **$+\$188.73\text{M}$** | **$+\$108.73\text{M}$** |

```text
  Levelized Cost per Move ($/move)
    $70 |        \  (Automated: High CAPEX amortized poorly at low volume)
    $65 |         \
    $60 |          \
    $55 |-----------\----------------- (Manual: Flat variable labor cost)
    $50 |            \
    $45 |             \________ (Automated dominates: $43.37/move at 2.5M)
        +-----------------------------------------------
             1.2M    1.5M   1.70M    2.2M    2.5M
                       Annual Container Moves
```

**Crossover Threshold:** At exactly **$1,695,737\text{ moves/year}$** ($\approx 1.70\text{M moves}$), the automation investment achieves full economic parity against the manual baseline ($\text{Incremental NPV} = 0$, $LCPM_{\text{auto}} = LCPM_{\text{manual}} = \$52.60\text{ /move}$). Above this volume, automation delivers massive positive financial returns, reducing unit cost per move by $\$7.69\text{ /move}$ at $2.5\text{M moves/year}$ and generating over $\$188\text{M}$ in incremental net present value.

---

## 7. Code, query, or calculation

The following self-contained Python script computes the exact engineering fleet sizing for battery-electric Lift-AGVs, multiserver fast-charging stalls, and performs a 20-year DCF capital budgeting sensitivity analysis across terminal volume tiers.

```python
"""
Terminal Automation Capital Budgeting and Fleet Sizing Engine
Calculates Lift-AGV fleet requirements, charging queues, NPV, and LCPM.
"""
import math

def analyze_terminal_automation(
    annual_moves=1_500_000,
    wacc=0.08,
    years=20,
    peak_mph=192.0,
    cycle_dist_m=1400.0,
    agv_speed_mps=4.5,
    battery_kwh=280.0,
    charge_kw=350.0,
    charge_eff=0.92,
    kwh_per_km=3.2,
    capex_total_m=477.90,
    manual_opex_per_move=47.80,
    auto_opex_per_move=23.90,
    manual_sustaining_capex_m=80.0
):
    # 1. Kinematics and Energy
    t_travel = cycle_dist_m / agv_speed_mps + 19.0  # accel + turns
    t_exch = 30.0  # Lift-AGV decoupled buffer time
    energy_move = (cycle_dist_m / 1000.0) * kwh_per_km + 1.8
    usable_kwh = battery_kwh * 0.60  # 20% to 80% SoC
    moves_per_charge = usable_kwh / energy_move
    t_charge_sec = (usable_kwh / (charge_kw * charge_eff)) * 3600.0

    # 2. Fleet Sizing
    t_eff_sec = t_travel + t_exch + (t_charge_sec / moves_per_charge)
    active_agvs = (peak_mph / 3600.0) * t_eff_sec
    fleet_agvs = math.ceil(active_agvs / 0.88)  # 88% availability

    # 3. Charger Sizing (Erlang-C approximation)
    charge_arrivals_per_hr = peak_mph / moves_per_charge
    charger_service_rate = 3600.0 / t_charge_sec
    offered_load = charge_arrivals_per_hr / charger_service_rate
    chargers_needed = math.ceil(offered_load * 1.5)  # headroom for P_wait < 0.05

    # 4. DCF and LCPM Economics
    pvifa = (1.0 - (1.0 + wacc) ** -years) / wacc
    pv_moves = annual_moves * pvifa
    pv_savings = annual_moves * (manual_opex_per_move - auto_opex_per_move) * pvifa
    npv_inc = pv_savings - ((capex_total_m - manual_sustaining_capex_m) * 1e6)
    npv_std = pv_savings - (capex_total_m * 1e6)

    pv_auto_costs = (capex_total_m * 1e6) + (annual_moves * auto_opex_per_move * pvifa)
    lcpm_auto = pv_auto_costs / pv_moves

    pv_man_costs = (manual_sustaining_capex_m * 1e6) + (annual_moves * manual_opex_per_move * pvifa)
    lcpm_man = pv_man_costs / pv_moves

    return {
        "fleet_agvs": fleet_agvs,
        "chargers": chargers_needed,
        "npv_inc_m": npv_inc / 1e6,
        "npv_std_m": npv_std / 1e6,
        "lcpm_auto": lcpm_auto,
        "lcpm_man": lcpm_man
    }

if __name__ == "__main__":
    baseline = analyze_terminal_automation()
    print(f"Lift-AGV Fleet Required: {baseline['fleet_agvs']} units")
    print(f"Fast Chargers Required: {baseline['chargers']} stalls")
    print(f"Baseline (1.5M Moves) -> Inc NPV: ${baseline['npv_inc_m']:.2f}M | Std NPV: ${baseline['npv_std_m']:.2f}M | "
          f"LCPM Auto: ${baseline['lcpm_auto']:.2f} | LCPM Manual: ${baseline['lcpm_man']:.2f}")

    print("\nVolume Sensitivity Analysis:")
    for vol in [1_200_000, 1_500_000, 1_695_737, 2_200_000, 2_500_000]:
        res = analyze_terminal_automation(annual_moves=vol)
        print(f"Vol: {vol/1e6:.2f}M -> Inc NPV: ${res['npv_inc_m']:6.2f}M | "
              f"Auto: ${res['lcpm_auto']:5.2f} | Man: ${res['lcpm_man']:5.2f} | "
              f"Advantage: ${res['lcpm_man'] - res['lcpm_auto']:+5.2f}/move")
```

### Script Execution Output
```text
Lift-AGV Fleet Required: 27 units
Fast Chargers Required: 6 stalls
Baseline (1.5M Moves) -> Inc NPV: $-45.92M | Std NPV: $-125.92M | LCPM Auto: $56.35 | LCPM Manual: $53.23

Volume Sensitivity Analysis:
Vol: 1.20M -> Inc NPV: $-116.32M | Auto: $64.46 | Man: $54.59 | Advantage: -$9.87/move
Vol: 1.50M -> Inc NPV: $-45.92M | Auto: $56.35 | Man: $53.23 | Advantage: -$3.12/move
Vol: 1.70M -> Inc NPV: $  0.01M | Auto: $52.60 | Man: $52.61 | Advantage: $+0.00/move
Vol: 2.20M -> Inc NPV: $118.34M | Auto: $46.03 | Man: $51.50 | Advantage: $+5.48/move
Vol: 2.50M -> Inc NPV: $188.73M | Auto: $43.37 | Man: $51.06 | Advantage: $+7.69/move

---

## 8. Data, standards, and regulatory context

Automated container terminal engineering operates within a complex matrix of international safety standards, industrial cybersecurity regulations, environmental decarbonization mandates, and civil engineering codes:

### 1. Safety and Machinery Directives
- **ISO 3691-4:2023 (Industrial Trucks — Safety Requirements and Verification — Part 4: Driverless industrial trucks and their systems):** Establishes binding safety requirements for AGVs, Lift-AGVs, and autonomous terminal tractors. Mandates dual-channel safety laser scanners (PL-d / SIL 2 according to ISO 13849-1 / IEC 62061), defining dynamically switching protective warning and stopping fields based on speed, steering angle, and braking distance under wet pavement conditions.
- **EN 15011 / FEM 1.001 (Cranes — Bridge and Gantry Cranes):** Governs structural calculations, fatigue life classes, and dynamic wheel load impact factors for rail-mounted automated stacking cranes (ASCs) and ship-to-shore (STS) cranes.
- **OSHA 29 CFR Part 1917 (Marine Terminals):** In the United States, automated terminal perimeters must comply with OSHA machine guarding regulations. Mandates physical interlocking fencing, light curtains, and trapped-key interlock systems around the entire automated yard zone. If an unauthorized human breaches the perimeter fence, the Equipment Control System (ECS) must trigger an immediate Category 0 emergency stop across all cranes and AGVs in that block.

### 2. Industrial Cybersecurity and Wireless Telemetry
- **IEC 62443 Series (Security for Industrial Automation and Control Systems - IACS):** Mandates strict network segmentation into discrete "Zones and Conduits." Terminal Operating Systems (TOS) in the corporate/enterprise zone must be isolated from the operational Equipment Control System (ECS), crane PLCs, and vehicle onboard computers via industrial demilitarized zones (IDMZs), data diodes, and stateful inspection firewalls.
- **3GPP Release 16/17 (5G System; Ultra-Reliable Low-Latency Communication - URLLC):** Specifies private cellular architectures operating in dedicated industrial spectrum bands (e.g., CBRS 3.5 GHz in the US, n77/n78 3.7–3.8 GHz in Europe). Mandates dual active radio links, Coordinated Multi-Point (CoMP) transmission, and local 5G Core User Plane Function (UPF) deployment on-terminal to guarantee $<5\text{ ms}$ packet latency and $99.999\%$ packet delivery reliability under dense metal container interference.

### 3. Decarbonization and Environmental Mandates
- **IMO 2023 Strategy on Reduction of GHG Emissions from Ships:** Mandates net-zero greenhouse gas emissions from international maritime shipping by or around 2050, with indicative checkpoints for 2030 (at least $20\%$, striving for $30\%$ reduction) and 2040 ($70\%$, striving for $80\%$).
- **European Union AFIR (Alternative Fuels Infrastructure Regulation):** Mandates that major TEN-T container ports install shore-side electricity supply (cold ironing / OPS) for container vessels by 2030, directly coupling terminal electrical grid substation design with quayside crane automation power planning.
- **California Air Resources Board (CARB) At-Berth and Cargo Handling Equipment (CHE) Regulations:** Mandates zero-emission transitions for all yard tractors, RTGs, and handling equipment across California ports (Long Beach, Los Angeles, Oakland), effectively prohibiting new diesel equipment and requiring full battery-electric or hydrogen infrastructure.

### 4. Electrical Grid Safety and Stationary Storage Codes
- **NFPA 855 (Standard for the Installation of Stationary Energy Storage Systems):** Governs large-scale lithium-ion Battery Energy Storage Systems (BESS) installed to buffer terminal peak fast-charging demand. Mandates thermal runaway off-gas detection, deflagration explosion venting compliant with NFPA 68, high-expansion foam or water-mist deluge suppression systems, and minimum 3.0-meter physical separation between modular battery enclosures.
- **IEEE 1584 / NFPA 70E (Arc-Flash Hazard Calculations and Electrical Safety in the Workplace):** Mandates incident energy analysis across medium-voltage ($11\text{ kV} / 33\text{ kV}$) substations and high-power fast-charging pit rectifiers, establishing flash protection boundaries and Personal Protective Equipment (PPE) categories for terminal mechatronic technicians.

---

## 9. Industry practice and case evidence

Automated container terminals have evolved across three distinct design generations over the past 30 years:

```text
+-------------------+-----------------------------------+-----------------------------------+
| GENERATION        | CHARACTERISTIC ARCHITECTURE       | REPRESENTATIVE FACILITIES         |
+-------------------+-----------------------------------+-----------------------------------+
| 1st Generation    | End-loading ASCs, diesel-electric | ECT Delta Rotterdam (1993)        |
| (1993 - 2005)     | standard AGVs, legacy Wi-Fi       | CTA Altenwerder Hamburg (2002)    |
+-------------------+-----------------------------------+-----------------------------------+
| 2nd Generation    | Cantilever ASCs, battery AGVs,    | Euromax Rotterdam (2008)          |
| (2006 - 2018)     | battery swap stations, fiber mesh | RWG Rotterdam (2015), APMT MVII   |
|                   |                                   | LBCT Pier E Long Beach (2016)     |
+-------------------+-----------------------------------+-----------------------------------+
| 3rd Generation    | Decoupled Lift-AGVs, opportunity  | PSA Tuas Singapore (Phase 1, 2022)|
| (2019 - Present)  | fast-charging, private 5G URLLC,  | Qingdao Qianwan (Phase 1-3)       |
|                   | AI dynamic TOS dispatching        | APM Terminals MedPort Tangier     |
+-------------------+-----------------------------------+-----------------------------------+
```

### 1. Long Beach Container Terminal (LBCT - Pier E), Port of Long Beach, USA
- **Context & Scale:** A \$1.5 billion, 300-acre brownfield redevelopment combining three aging breakbulk and manual container piers into a single mega-terminal with 3.3 million TEU annual capacity.
- **Technical Selection:** 14 ZPMC electric dual-hoist STS cranes, 72 Terex battery-electric Lift-AGVs with automated battery swap stations, and 70 rail-mounted cantilever ASCs arranged perpendicular to the quay.
- **Operating Evidence:** Operates in one of the world's most expensive labor jurisdictions under the International Longshore and Warehouse Union (ILWU). LBCT eliminated 100% of direct diesel emissions from yard transport and stacking, achieving gross crane productivities exceeding 33 moves/hour. Its success proved that full automation can operate harmoniously within an established union collective bargaining framework by providing extensive workforce retraining and high-paying electrical/mechatronic maintenance careers.

### 2. Rotterdam World Gateway (RWG) and APM Terminals Maasvlakte II, Netherlands
- **Context & Scale:** Built concurrently on reclaimed North Sea deepwater land (Maasvlakte 2), both terminals opened in 2015 as the world's first fully zero-emission automated container terminals.
- **Technical Selection:** RWG deployed 14 automated dual-trolley STS cranes, 59 Lift-AGVs with automated battery swap stations, and 50 cantilever ASCs. APMT MVII eliminated traditional STS crane cabins entirely, operating 100% remote-controlled STS cranes from an office building 2 kilometers away.
- **Operating Evidence:** Demonstrated that active Lift-AGVs reduce required fleet size by $22\%$ compared to the 1st generation standard AGVs deployed at nearby ECT Delta, due to zero dead-wait time at the quayside and yard buffers. However, initial commissioning suffered from Equipment Control System (ECS) software bugs, requiring 18 months of software optimization to achieve rated vessel turnaround speeds.

### 3. PSA Singapore — Tuas Mega-Port
- **Context & Scale:** The largest automated container terminal project in human history. Upon completion in the 2040s, Tuas will consolidate all Singapore container operations into a single 65-million TEU facility with 66 berths across 26 kilometers of quay.
- **Technical Selection:** Phase 1 (opened 2022) utilizes over 200 fully automated battery-electric AGVs operating on a nationwide private 5G standalone slice, high-power DC opportunity charging grids, and automated double-cantilever rail cranes.
- **Operating Evidence:** Replaced centralized transponder grid navigation with differential GNSS RTK combined with multi-sensor lidar SLAM and private 5G telemetry. Tuas verified that opportunity fast-charging eliminates the massive real estate footprint of automated battery swap stations, reducing civil land capital outlays by over $\$40\text{M}$ per terminal phase.

### 4. Cautionary Failures and Lessons Learned
- **TraPac Terminal (Port of Los Angeles, USA):** Experienced severe capital expenditure overruns during its brownfield automated straddle carrier (AutoStrad) and ASC retrofit between 2014 and 2018. Complex physical interfaces between manual over-the-road trucks and automated handling zones generated chronic gate bottlenecks. Pavement differential settlement along rail crane tracks required emergency grouting and track realignments, adding tens of millions in unbudgeted civil expenditures.
- **Patrick Terminals (Brisbane / Sydney, Australia):** Pioneer of radar-guided automated straddle carriers (AutoStrads). While achieving significant labor cost reductions, early deployments suffered from heavy maritime salt-spray fog blinding early microwave radar sensors, illustrating the necessity of multi-spectral sensor fusion (lidar, radar, optical cameras, and IMUs) in harsh marine environments.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign | Engineering Mechanism | Consequence | Mitigation Strategy |
| :-: | :--- | :--- | :--- | :--- | :--- |
| 1 | **ECS Circular Deadlock** | AGV travel velocities drop to zero; queue accumulation in cross-lanes. | Circular resource-wait chain where 4+ vehicles block each other's clearance envelopes. | Total gridlock; STS crane starvation; quayside operations halt. | Implement Dijkstra/Banker's deadlock-free routing algorithms with reservation grids. |
| 2 | **Winter Battery SoC Collapse** | Rapid battery depletion rates; unexpected low-voltage cutoffs. | Sub-zero ambient temperatures increase internal battery cell impedance; HVAC cabin draw. | AGVs stall on active runways; crane production drops by 40%. | Install battery thermal management systems (BTMS) with pre-heating and LTO chemistry. |
| 3 | **ROS Video Latency Spikes** | Operator overshoots container landing; video jitter and tearing. | Wireless packet loss, bufferbloat in switch queues, or H.265 encoder frame drop. | Remote operators cannot land spreaders; landside truck gate queues back up. | Deploy private 5G URLLC with prioritized QoS Class Identifier (5QI 82/83) and edge compute. |
| 4 | **Runway Differential Settlement** | Excessive rail clip fatigue; crane gantry skew alarms; rail wear. | Uneven consolidation of reclaimed port fill soil beneath runway pile foundations. | Rail structural shearing; ASC gantry wheel derailment or drive motor burnout. | Deep pile-driving to solid bedrock; hydraulic leveling jacks; continuous fiber-optic strain monitoring. |
| 5 | **Charging Pantograph Arcing** | Electrical pitting on contact shoes; scorch marks on charging hoods. | Misalignment of vehicle docking position or conductive salt-air crust on contact pads. | Charging interruption; fire hazard; reduced power delivery rate. | Automated ultrasonic contact alignment; motorized wiping brushes; sacrificial graphite shoes. |
| 6 | **Optical Chassis Detection Failure** | ROS alarms on truck landing; manual override required for 100% of moves. | Rusted, bent, or non-standard twistlock chassis pins on third-party drayage trucks. | Operator takeover rate spikes from 8% to 50%; ROS console pool saturated. | Mandatory terminal chassis inspection gates; AI stereoscopic vision neural net models. |
| 7 | **5G Multipath Signal Shadowing** | Packet drops when AGV navigates into deep container yard canyons. | Dense steel container stacks reflect and attenuate sub-6 GHz RF signals. | Telemetry loss triggers vehicle safe-stop; intermittent transport halts. | High-tower distributed antenna systems (DAS); multi-beam MIMO; dual-SIM multi-carrier routers. |
| 8 | **Twistlock Latching Jam** | Spreader hoisted with 3 twistlocks locked and 1 unlocked. | Damaged container corner casting; eccentric load distribution; mechanical wear. | Container drops or tilts violently; catastrophic structural damage to ship/crane. | Automated twistlock load-cell tension sensors; dual-hoist safety interlocks; machine vision. |
| 9 | **Macroeconomic Volume Slump** | Throughput falls below the calculated crossover threshold ($<1.74\text{M moves}$). | High fixed debt service and depreciation cannot be reduced as container volumes fall. | Severe negative cash flow; financial covenant breaches; terminal insolvency. | Flexible phased CAPEX expansion; contractual minimum volume guarantees (Take-or-Pay). |
| 10 | **Labor Automation Moratorium** | Union grievance filings, slowdowns, or jurisdictional strike notices. | Collective bargaining disputes regarding workforce displacement and automation scope. | Terminal shutdown; shipping alliances divert vessel strings to competing ports. | Early workforce transition agreements; guaranteed pension funds; automation royalty profit-sharing. |
| 11 | **OT Perimeter Cyber Intrusion** | Unauthorized network traffic on crane PLC network; telemetry drift. | Insecure remote vendor maintenance VPN or compromised industrial gateway. | Malicious hijacking of heavy crane machinery; physical asset destruction. | Strict IEC 62443 compliance; air-gapped IDMZs; continuous network behavioral anomaly detection. |

---

## 11. Assumptions and limits

| # | Assumption | Why It Is Needed | What Breaks It | How to Test / Validate |
| :-: | :--- | :--- | :--- | :--- |
| 1 | **Stationary Crane Demand** | Allows closed queueing network equations to calculate steady-state fleet size. | Ship hatch changes, lashing delays, and crane breakdowns generate bursty arrivals. | Discrete-event simulation (Topic 41.08) with non-homogeneous Poisson processes. |
| 2 | **Deterministic Battery Energy** | Uses a constant energy consumption ($E_{\text{move}} = 6.28\text{ kWh}$) for charging calculations. | Severe headwinds ($>50\text{ km/h}$), heavy container payloads, and winter heating. | Empirical CAN-bus telemetry logging under loaded, windy operational trials. |
| 3 | **Uniform Yard Routing Distances** | Permits using an average round-trip travel distance ($1,400\text{ m}$). | Skewed vessel stowage where all cargo concentrates in the farthest yard blocks. | Dynamic block allocation algorithms in TOS; spatial travel-distance distribution tests. |
| 4 | **Perfect Buffer Decoupling** | Assumes Lift-AGVs never wait for cranes ($T_{\text{wait}} = 0$) at ground stools. | Ground stools full due to ASC maintenance or crane hardware faults. | Sizing transfer zone buffer capacity to exceed peak queue length at $99.5\%$ confidence. |
| 5 | **Zero 5G Packet Drop** | Assumes deterministic telemetry communication without radio packet loss. | Extreme metal reflection, antenna failure, or industrial RF interference. | Drive testing with continuous UDP packet-train jitter and loss logging. |
| 6 | **Constant Wage Escalation** | Assumes manual labor wage inflation mirrors baseline corporate cost projections. | Union negotiations secure aggressive wage spikes (e.g., $30\% - 60\%$ over 6-year pacts). | Scenario modeling with stochastic wage growth rates ($3\%$ to $8\%$ annual inflation). |
| 7 | **Linear Crane Scaling** | Assumes adding ASCs scales yard throughput linearly without interference. | Rail interference between twin cranes on the same block track during simultaneous moves. | Gantry collision-avoidance simulation modeling with twin-crane clearance buffers. |
| 8 | **Zero Year-20 Salvage Value** | Conservatively assumes heavy automated assets have zero net residual value. | Modern cranes and civil runway beams have useful structural lives exceeding 30–40 years. | Asset appraisal benchmarking; secondary crane market valuation studies. |
| 9 | **Unconstrained Grid Power** | Assumes local electric utility can deliver peak charging power ($>15\text{ MW}$). | Municipal substation capacity bottlenecks or transmission interconnect delays. | Utility interconnection feasibility studies; on-site battery energy storage systems (BESS). |
| 10 | **Constant Rolling Resistance** | Assumes tire rolling resistance $C_{\text{rr}} = 0.016$ is invariant across seasonal weather. | Heavy standing water, slush, or port debris increases rolling drag by $25\% - 40\%$. | Dynamic tire pressure monitoring and pavement drainage capacity audits. |
| 11 | **Homogeneous Cell Degradation** | Assumes all vehicle battery packs age uniformly at identical capacity fade rates. | Variations in charging C-rates, thermal gradients, and driverless dispatch skew cell SOH. | Fleet-wide battery management system (BMS) cloud analytics and State of Health (SOH) tracking. |

---

## 12. Exercises

### Exercise 1: AGV vs Lift-AGV Buffer Decoupling Mechanics (Recall)
Explain the mechanical and operational distinction between a standard passive-deck AGV and an active Lift-AGV. Describe how this distinction impacts quayside quay crane wait times, horizontal fleet size, and overall terminal operating leverage.
- **Deliverable:** Technical memorandum (300–450 words) with a comparative kinematic cycle diagram.
- **Acceptance Criteria:** Correctly identifies the presence of the hydraulic lift platform and passive ground stools; correctly shows the elimination of $T_{\text{wait, STS}}$ and $T_{\text{wait, ASC}}$; references empirical fleet reduction percentages.
- **Time Estimate:** 25 minutes.

### Exercise 2: Fleet Sizing Under Severe Environmental Headwinds (Calculation)
An automated terminal deploys battery-electric Lift-AGVs with nominal tare mass $m_{\text{tare}} = 30,000\text{ kg}$ carrying a 40-foot container ($m_{\text{container}} = 26,000\text{ kg}$). The vehicle operates in severe coastal storm conditions with a sustained headwind $v_{\text{wind}} = 15.0\text{ m/s}$ ($54\text{ km/h}$). Frontal area $A_{\text{f}} = 8.5\text{ m}^2$, $C_{\text{d}} = 0.85$, $\rho_{\text{air}} = 1.25\text{ kg/m}^3$, tire rolling resistance $C_{\text{rr}} = 0.016$, and driveline efficiency $\eta_{\text{drive}} = 0.88$.
1. Calculate the steady-state mechanical traction power $P_{\text{traction}}$ required to cruise at $v_{\text{vehicle}} = 4.0\text{ m/s}$ directly into the headwind.
2. Determine the percentage increase in traction power compared to calm conditions ($v_{\text{wind}} = 0$).
- **Deliverable:** Fully worked mathematical derivation with all intermediate force equations.
- **Acceptance Criteria:** Correctly applies relative airspeed velocity $(v_{\text{vehicle}} + v_{\text{wind}})$ in aerodynamic drag; calculates power within $\pm 1.5\%$ precision.
- **Time Estimate:** 35 minutes.

### Exercise 3: Remote Operating Station (ROS) Sizing via Erlang-C (Application)
An automated terminal operates 32 Cantilever ASCs. Each ASC handles an average of 25 container moves per hour. Due to poor lighting and legacy chassis twistlock damage, the probability of requiring human ROS operator intervention is $p_{\text{int}} = 12.0\%$. Mean operator intervention time is 48 seconds per event.
1. Compute the total intervention arrival rate $\lambda_{\text{ros}}$ (interventions per hour) and the offered workload $a_{\text{ros}}$ (Erlangs).
2. Using the Erlang-C formula, determine the minimum number of operational ROS desks $c_{\text{ros}}$ required such that the probability of a crane waiting for an operator is $P_{\text{wait}} \le 0.02$ ($2\%$).
3. If each ROS desk is staffed by an operator earning $\$140,000\text{ /year}$ fully loaded, calculate total annual ROS direct labor cost assuming 24/7/365 coverage (4.2 shifts per desk).
- **Deliverable:** Step-by-step queueing calculation spreadsheet or Python script output with staffing cost summary.
- **Acceptance Criteria:** Correct calculation of Erlang intensity; accurate application of Erlang-C formula; correct multi-shift labor factor.
- **Time Estimate:** 40 minutes.

### Exercise 4: Civil Engineering Runway Soil Bearing Capacity (Calculation)
A rail-mounted ASC has a total self-weight $m_{\text{crane}} = 310\text{ metric tons}$. The rated hoist capacity under twin-lift is 65 metric tons. The crane is supported by two 8-wheel bogies (one on each rail runway, total 16 wheels). Dynamic impact factor $\phi_{\text{dyn}} = 1.30$. During a gale event, lateral wind load creates an overturning moment $M_{\text{wind}} = 4,200\text{ kNm}$. The crane rail track gauge is $L_{\text{track}} = 32.0\text{ meters}$.
1. Calculate the peak dynamic wheel load $P_{\text{wheel, max}}$ on the critical leeward rail.
2. If the local port geotechnical report establishes a maximum allowable soil bearing capacity of $180\text{ kPa}$ without deep piling, explain why a shallow spread footing cannot support this crane rail and calculate the minimum required pile group capacity.
- **Deliverable:** Structural load calculation report with geotechnical commentary.
- **Acceptance Criteria:** Correct application of static dead load, live load, dynamic impact, and overturning moment; correct wheel load in kilonewtons (kN).
- **Time Estimate:** 30 minutes.

### Exercise 5: LCPM Sensitivity to Trade Recession Volume Shock (Financial)
Using the worked example baseline ($CAPEX_0 = \$477.9\text{M}$, Automated OPEX $=\$23.90\text{ /move}$, Manual OPEX $=\$47.80\text{ /move}$, Manual sustaining CAPEX $=\$80.0\text{M}$ PV, $WACC = 8.0\%$, 20-year horizon):
1. Compute the project NPV and Levelized Cost per Move ($LCPM$) if a prolonged global geopolitical crisis reduces terminal volume by $25\%$ to $Q = 1,125,000\text{ moves/year}$.
2. Contrast this with a trade boom scenario where volume expands by $+30\%$ to $Q = 1,950,000\text{ moves/year}$.
3. Formulate an executive briefing explaining how operating leverage drives the divergence between these two financial outcomes.
- **Deliverable:** Financial comparison table and a 250-word executive memo.
- **Acceptance Criteria:** Exact discounted cash flow figures; correct LCPM calculation for both volume tiers; sound managerial commentary on fixed-cost absorption.
- **Time Estimate:** 45 minutes.

### Exercise 6: Battery Swap Stations (BSS) vs In-Ground Opportunity Charging (Critique)
Critique the operational, economic, and civil infrastructure trade-offs between Automated Battery Swap Stations (BSS) and In-Ground Conductive Opportunity Fast Charging (350–500 kW) for a fleet of 50 automated container vehicles.
- **Deliverable:** Comparative evaluation matrix across 6 criteria (Terminal Real Estate Footprint, Spare Battery Inventory CAPEX, Mechanical Reliability, Peak Grid Power Demand, Vehicle Structural Weight, Battery Cell Degradation) followed by an evidence-based recommendation.
- **Acceptance Criteria:** Explicitly references real terminal precedents (e.g., LBCT/RWG for BSS vs PSA Tuas for opportunity charging); cites specific degradation and electrical tariff mechanics.
- **Time Estimate:** 40 minutes.

### Exercise 7: Private 5G URLLC Antenna Architecture Design (Design)
Design the physical RF network topology for an automated container yard consisting of 16 ASC blocks, where container stacks reach 6 high ($15.5\text{ meters}$ elevation). The layout measures $1,200\text{ m} \times 600\text{ m}$.
1. Specify antenna mounting locations, heights, and beam downtilt angles to overcome extreme multipath signal reflection and shadowing in container canyons.
2. Formulate the redundancy architecture for the 5G Core, User Plane Function (UPF), and fiber backhaul to ensure 99.999% availability ($<5.26\text{ minutes}$ downtime per year).
- **Deliverable:** Systems architecture specification document with layout diagram and failover logic.
- **Acceptance Criteria:** Explains how to eliminate shadow zones; specifies on-premise local UPF deployment; complies with 3GPP Rel-16 URLLC parameters.
- **Time Estimate:** 50 minutes.

---

## 13. Self-check answers

### Solution to Exercise 1: AGV vs Lift-AGV Decoupling Mechanics
A standard AGV possesses a passive deck. When transporting a container, it requires an external crane (STS or ASC) to hoist the container onto its twistlock pins and offload it at destination. Consequently, the vehicle must remain stationary under the crane until the crane completes its cycle. If the crane is delayed by hatch changes or internal stack shuffling, the AGV incurs dead-wait time ($60 \text{ to } 180\text{ seconds}$). This couples the horizontal transport cycle directly to the crane cycle.

An active Lift-AGV incorporates two independent hydraulic or electromechanical lifting platforms built into its chassis. At the waterside transfer zone or quayside buffer, the Lift-AGV drives over passive steel ground stools (cassettes), lowers its platform, deposits the container in $<15\text{ seconds}$, and departs immediately. The crane retrieves the container from the stool asynchronously. This eliminates vehicle waiting dead-time, decouples horizontal transport from crane kinematics, and reduces the required vehicle fleet by $20\% \text{ to } 25\%$ for equivalent quayside throughput. In capital budgeting, this fleet reduction saves tens of millions in initial vehicle CAPEX and eliminates quay crane starvation during peak vessel working windows.

---

### Solution to Exercise 2: Fleet Sizing Under Severe Environmental Headwinds
#### 1. Mechanical Traction Power Calculation
- Gross mass $m_{\text{gross}} = 30,000 + 26,000 = 56,000\text{ kg}$.
- Vehicle ground speed $v_{\text{vehicle}} = 4.0\text{ m/s}$.
- Relative airspeed velocity $v_{\text{air}} = v_{\text{vehicle}} + v_{\text{wind}} = 4.0 + 15.0 = 19.0\text{ m/s}$.
- Rolling resistance force:
  $$F_{\text{rr}} = m_{\text{gross}} \cdot g \cdot C_{\text{rr}} = 56,000 \times 9.81 \times 0.016 = 8,789.76\text{ N}$$
- Aerodynamic drag force:
  $$F_{\text{drag}} = \frac{1}{2} \rho_{\text{air}} C_{\text{d}} A_{\text{f}} v_{\text{air}}^2 = \frac{1}{2} (1.25) (0.85) (8.5) (19.0)^2 = 4.515625 \times 361 = 1,630.14\text{ N}$$
- Total tractive force required at wheels:
  $$F_{\text{traction}} = F_{\text{rr}} + F_{\text{drag}} = 8,789.76 + 1,630.14 = 10,419.90\text{ N}$$
- Steady-state mechanical traction power at wheels:
  $$P_{\text{wheel}} = F_{\text{traction}} \cdot v_{\text{vehicle}} = 10,419.90 \times 4.0 = 41,679.6\text{ W} = 41.68\text{ kW}$$
- Power delivered by electric motor/inverter:
  $$P_{\text{traction}} = \frac{P_{\text{wheel}}}{\eta_{\text{drive}}} = \frac{41.68}{0.88} = \mathbf{47.36\text{ kW}}$$

#### 2. Comparison with Calm Conditions ($v_{\text{wind}} = 0$)
- In calm air, $v_{\text{air}} = 4.0\text{ m/s}$:
  $$F_{\text{drag, calm}} = \frac{1}{2} (1.25) (0.85) (8.5) (4.0)^2 = 4.515625 \times 16 = 72.25\text{ N}$$
  $$F_{\text{traction, calm}} = 8,789.76 + 72.25 = 8,862.01\text{ N}$$
  $$P_{\text{traction, calm}} = \frac{8,862.01 \times 4.0}{0.88} = \frac{35,448.0}{0.88} = 40.28\text{ kW}$$
- **Percentage Increase:**
  $$\Delta P = \frac{47.36 - 40.28}{40.28} \times 100\% = \mathbf{+17.58\%}$$
*Takeaway:* Aerodynamic drag scales quadratically with relative airspeed, increasing total traction power draw by nearly $18\%$ during gale conditions, which accelerates battery SoC depletion and increases daily charging frequency.

---

### Solution to Exercise 3: Remote Operating Station (ROS) Sizing via Erlang-C
#### 1. Arrival Rate and Offered Workload
- Total crane moves per hour: $32\text{ ASCs} \times 25\text{ moves/hr} = 800\text{ moves/hour}$.
- Intervention probability $p_{\text{int}} = 0.12$.
- Intervention arrival rate:
  $$\lambda_{\text{ros}} = 800 \times 0.12 = 96.0\text{ interventions/hour}$$
- Service rate per desk: $1/\mu = 48\text{ seconds} = 0.01333\text{ hours} \implies \mu = 3,600 / 48 = 75.0\text{ interventions/hour/desk}$.
- Offered workload intensity:
  $$a_{\text{ros}} = \frac{\lambda_{\text{ros}}}{\mu} = \frac{96.0}{75.0} = \mathbf{1.28\text{ Erlangs}}$$

#### 2. Minimum Operational ROS Desks
To satisfy $P_{\text{wait}} \le 0.02$:
- Minimum desks for stability: $c > 1.28 \implies c \ge 2$.
- Testing $c = 3$ desks:
  $$\rho = \frac{1.28}{3} = 0.4267$$
  $$P_{\text{wait}} = C(3, 1.28) = \frac{\frac{1.28^3}{6} \left(\frac{3}{3 - 1.28}\right)}{1 + 1.28 + \frac{1.28^2}{2} + \frac{1.28^3}{6}\left(\frac{3}{3 - 1.28}\right)} = \frac{0.3495 \times 1.7442}{1 + 1.28 + 0.8192 + 0.6096} = \frac{0.6096}{3.7088} = 0.1644 \quad (16.4\% > 2\%)$$
- Testing $c = 4$ desks:
  $$\rho = \frac{1.28}{4} = 0.3200$$
  $$\frac{a^4}{4!} \left(\frac{4}{4 - 1.28}\right) = \frac{2.6844}{24} (1.4706) = 0.11185 \times 1.4706 = 0.1645$$
  $$\sum_{k=0}^{3} \frac{1.28^k}{k!} = 1 + 1.28 + 0.8192 + 0.3495 = 3.4487$$
  $$P_{\text{wait}} = \frac{0.1645}{3.4487 + 0.1645} = \frac{0.1645}{3.6132} = 0.0455 \quad (4.55\% > 2\%)$$
- Testing $c = 5$ desks:
  $$\rho = \frac{1.28}{5} = 0.2560$$
  $$\frac{a^5}{5!} \left(\frac{5}{5 - 1.28}\right) = \frac{3.436}{120} (1.3441) = 0.02863 \times 1.3441 = 0.03848$$
  $$\sum_{k=0}^{4} \frac{1.28^k}{k!} = 3.4487 + 0.11185 = 3.5605$$
  $$P_{\text{wait}} = \frac{0.03848}{3.5605 + 0.03848} = \frac{0.03848}{3.5990} = \mathbf{0.0107 \quad (1.07\% \le 2\%)}$$
**Result:** Exactly **$c_{\text{ros}} = 5\text{ operational ROS desks}$** are required. Staffing ratio is $32 / 5 = 6.4\text{ ASCs per operator}$.

#### 3. Annual Direct Labor Cost
- 24/7/365 coverage requires $4.2\text{ FTEs per desk}$.
- Total operators required: $5\text{ desks} \times 4.2 = 21\text{ FTE operators}$.
- Total Annual Labor Cost: $21 \times \$140,000 = \mathbf{\$2,940,000\text{ /year}}$.

---

### Solution to Exercise 4: Civil Engineering Runway Soil Bearing Capacity
#### 1. Peak Dynamic Wheel Load Calculation
- Total mass $m_{\text{total}} = m_{\text{crane}} + m_{\text{load}} = 310,000 + 65,000 = 375,000\text{ kg}$.
- Total static gravitational load:
  $$W_{\text{total}} = 375,000 \times 9.81 = 3,678.75\text{ kN}$$
- Total wheels $= 16$ ($8\text{ wheels per rail}$).
- Static load per wheel:
  $$P_{\text{static}} = \frac{3,678.75}{16} = 229.92\text{ kN}$$
- Overturning moment from wind: $M_{\text{wind}} = 4,200\text{ kNm}$.
- Overturning vertical force on leeward rail (track gauge $L_{\text{track}} = 32.0\text{ m}$):
  $$\Delta F_{\text{rail}} = \frac{M_{\text{wind}}}{L_{\text{track}}} = \frac{4,200}{32.0} = 131.25\text{ kN}$$
- Additional wind force distributed over the 8 leeward wheels:
  $$\Delta P_{\text{wind, wheel}} = \frac{131.25}{8} = 16.41\text{ kN}$$
- Peak dynamic wheel load ($\phi_{\text{dyn}} = 1.30$):
  $$P_{\text{wheel, max}} = \phi_{\text{dyn}} \cdot P_{\text{static}} + \Delta P_{\text{wind, wheel}} = (1.30 \times 229.92) + 16.41 = 298.90 + 16.41 = \mathbf{315.31\text{ kN}}$$

#### 2. Geotechnical Soil Bearing Commentary
A dynamic point load of $315.31\text{ kN per wheel}$, with wheels spaced at $1.2 \text{ to } 1.5\text{ meters}$ along the bogie, creates a continuous linear load exceeding $220 \text{ to } 260\text{ kN/meter}$. A shallow footing would require a foundation width exceeding $1.5\text{ meters}$ to keep bearing pressures below the $180\text{ kPa}$ allowable limit. More critically, shallow footings in port reclamation fill suffer from long-term differential settlement, which shears crane rails and trips crane skew interlocks. Therefore, the runway requires deep concrete or steel pipe piles driven down to dense bearing strata (bearing capacity $\ge 1,200 \text{ to } 1,800\text{ kN per pile}$), spaced at 4-to-6-meter intervals beneath continuous reinforced concrete beams.

---

### Solution to Exercise 5: LCPM Sensitivity to Trade Recession Volume Shock
- Annuity Factor ($WACC = 8.0\%$, 20 yrs): $PVIFA = 9.8181$.
- Automated Costs: $CAPEX_0 = \$477.90\text{M}$; Unit OPEX $= \$23.90\text{ /move}$.
- Manual Costs: PV CAPEX $= \$80.00\text{M}$; Unit OPEX $= \$47.80\text{ /move}$.

#### 1. Recession Scenario ($Q = 1,125,000\text{ moves/year}$)
- Discounted Moves: $PV(Q) = 1,125,000 \times 9.8181 = 11,045,362\text{ moves}$.
- Annual Automated OPEX: $1.125\text{M} \times \$23.90 = \$26.8875\text{M}$.
- $PV(\text{Cost}_{\text{auto}}) = \$477.90\text{M} + (\$26.8875\text{M} \times 9.8181) = \$477.90\text{M} + \$263.98\text{M} = \$741.88\text{M}$.
  $$LCPM_{\text{auto}} = \frac{\$741,880,000}{11,045,362} = \mathbf{\$67.17\text{ /move}}$$
- Annual Manual OPEX: $1.125\text{M} \times \$47.80 = \$53.775\text{M}$.
- $PV(\text{Cost}_{\text{manual}}) = \$80.00\text{M} + (\$53.775\text{M} \times 9.8181) = \$80.00\text{M} + \$527.97\text{M} = \$607.97\text{M}$.
  $$LCPM_{\text{manual}} = \frac{\$607,970,000}{11,045,362} = \mathbf{\$55.04\text{ /move}}$$
- **Project NPV:** $-\$477.90\text{M} + [(\$53.775\text{M} - \$26.8875\text{M}) \times 9.8181] = -\$477.90\text{M} + \$263.98\text{M} = \mathbf{-\$213.92\text{M}}$.
- *Result:* In a recession, automation severely underperforms, adding $+\$12.13\text{ /move}$ in unit cost compared to manual operations.

#### 2. Trade Boom Scenario ($Q = 1,950,000\text{ moves/year}$)
- Discounted Moves: $PV(Q) = 1,950,000 \times 9.8181 = 19,145,295\text{ moves}$.
- Annual Automated OPEX: $1.950\text{M} \times \$23.90 = \$46.605\text{M}$.
- $PV(\text{Cost}_{\text{auto}}) = \$477.90\text{M} + (\$46.605\text{M} \times 9.8181) = \$477.90\text{M} + \$457.57\text{M} = \$935.47\text{M}$.
  $$LCPM_{\text{auto}} = \frac{\$935,470,000}{19,145,295} = \mathbf{\$48.86\text{ /move}}$$
- Annual Manual OPEX: $1.950\text{M} \times \$47.80 = \$93.21\text{M}$.
- $PV(\text{Cost}_{\text{manual}}) = \$80.00\text{M} + (\$93.21\text{M} \times 9.8181) = \$80.00\text{M} + \$915.14\text{M} = \$995.14\text{M}$.
  $$LCPM_{\text{manual}} = \frac{\$995,140,000}{19,145,295} = \mathbf{\$51.98\text{ /move}}$$
- **Project NPV:** $-\$477.90\text{M} + [(\$93.21\text{M} - \$46.605\text{M}) \times 9.8181] = -\$477.90\text{M} + \$457.57\text{M} = \mathbf{-\$20.33\text{M}}$ (approaching parity; crossover occurs at $1.742\text{M moves}$ when comparing against zero manual CAPEX).
- *Result:* At $1.95\text{M moves}$, automated LCPM is $\$3.12\text{ /move}$ cheaper than manual, saving the terminal over $\$6.0\text{M/year}$ in net operating costs.

#### 3. Executive Commentary
Automation transforms variable operational labor into fixed capital assets. In a volume downturn, high fixed depreciation and debt service cannot be cut, resulting in severe negative operating leverage and rising unit costs ($+\$12.13\text{ /move}$ penalty). In high-volume scenarios, fixed costs are diluted across millions of moves, unlocking dramatic economies of scale and expanding operating margins.

---

### Rubric for Exercise 6: BSS vs Opportunity Fast Charging
- **Terminal Real Estate Footprint (Weight: 20%):** BSS requires $1,500 \text{ to } 2,500\text{ m}^2$ dedicated building footprint, safety setbacks, and entrance/exit switch lanes. Opportunity charging pits occupy zero net real estate when built beneath quayside crane backreaches or transfer zones.
- **Spare Battery Inventory CAPEX (Weight: 20%):** BSS requires a battery-to-vehicle ratio of $1.25 \text{ to } 1.35$ (for 50 vehicles $\implies 15$ spare battery packs @ $\$75,000 = \$1.125\text{M}$ plus $\$3.5\text{M}$ mechanized swap station robotics). Opportunity charging requires exactly 1 battery per vehicle and lower station CAPEX.
- **Mechanical Reliability & Wear (Weight: 15%):** BSS involves automated multi-axis hydraulic robots lifting 8-to-12-ton battery packs thousands of times per year, introducing high mechanical failure rates and alignment wear. Conductive pantographs have few moving parts.
- **Peak Grid Power Demand (Weight: 15%):** BSS allows slow, steady, controlled battery charging (e.g., $50 \text{ to } 80\text{ kW}$ per pack), reducing peak grid spikes and avoiding utility maximum-demand surcharges. Opportunity charging draws massive intermittent pulses ($350 \text{ to } 500\text{ kW}$ simultaneously across 6–8 chargers $\implies 3.0 \text{ to } 4.0\text{ MW}$ instantaneous load), requiring costly on-site battery energy storage systems (BESS) or high utility capacity contracts.
- **Vehicle Structural Weight & Degradation (Weight: 15%):** Opportunity charging vehicles carry smaller battery packs ($180 - 250\text{ kWh}$) but undergo shallow micro-cycles (high C-rate), requiring LTO or advanced LFP cells. BSS vehicles carry larger packs ($280 - 350\text{ kWh}$) operating under lower C-rates.
- **Recommendation Synthesis (Weight: 15%):** Sound justification recommending Opportunity Fast Charging for modern greenfield terminals (following PSA Tuas precedent) due to land minimization, capital efficiency, and mechanical simplicity, backed by on-site BESS to shave electrical demand peaks.

---

### Rubric for Exercise 7: Private 5G URLLC Antenna Architecture Design
- **RF Antenna Placement (Weight: 25%):** Rejects mounting antennas at ground level. Places directional multi-sector MIMO antennas on high terminal floodlight towers ($35 \text{ to } 45\text{ meters}$ elevation) spaced at $200 \text{ to } 300\text{ meters}$, with downtilt angles of $8^\circ \text{ to } 14^\circ$ looking directly down the longitudinal container block canyons to illuminate the ground transfer lanes without obstruction from stacked steel containers.
- **Multipath Mitigation (Weight: 25%):** Specifies 3GPP Rel-16 Coordinated Multi-Point (CoMP) transmission and antenna diversity (dual circular polarization) to eliminate fading dips caused by reflections off metallic container walls.
- **Core and Edge UPF Redundancy (Weight: 25%):** Mandates dual-homed, geographically separated on-premise edge servers running the 5G User Plane Function (UPF) and Equipment Control System (ECS) gateways. Radio control plane traffic may connect to cloud cores, but user plane vehicle control loops must remain strictly on-premise.
- **Availability and Latency Compliance (Weight: 25%):** Demonstrates mathematical compliance with $99.999\%$ uptime ($<5.26\text{ minutes/year}$ downtime) via dual active radio modems on each AGV/ASC connecting to independent radio cells, achieving deterministic glass-to-glass latency $<45\text{ ms}$.

---

## 14. Interview questions

### Question 1: Capital Budgeting Metrics
**Prompt:** In assessing a \$600M terminal automation proposal, the Chief Financial Officer prefers Net Present Value (NPV), while the Chief Operating Officer insists on Levelized Cost per Move (LCPM). How do these metrics differ, when do they diverge, and which metric should govern the final board investment decision?
- **Strong Answer Contains:** Explains that NPV measures aggregate discounted dollar value creation over the concession horizon, directly evaluating whether total cash inflows/savings exceed the cost of capital ($WACC$). LCPM calculates the unit cost-recovery price per container move required to amortize initial CAPEX, sustaining CAPEX, and lifetime OPEX. They diverge when terminal volume projections are uncertain or variable: a project may have an acceptable LCPM under full capacity utilization, but if volume drops, the true realized cost per move explodes due to fixed capital depreciation, rendering the NPV negative. Concludes that NPV must govern the capital allocation decision because it directly incorporates volume risk and shareholder value creation, whereas LCPM serves as an operational pricing benchmark.
- **Weak Answer Reveals:** Equates LCPM to simple accounting cost per move without discounting; fails to recognize how volume variance distorts unit cost amortization; cannot explain how $WACC$ links the two metrics.

### Question 2: AGV Fleet Decoupling Physics
**Prompt:** Why do Lift-AGVs reduce required fleet size compared to standard AGVs, and how does this affect quayside crane productivity?
- **Strong Answer Contains:** Identifies the active lifting frame and passive ground stools as the physical decoupling mechanism. Standard AGVs have passive decks and must remain stationary under the crane spreader until the crane executes the lift or setdown, incurring $60 \text{ to } 180\text{ seconds}$ of dead-wait time if the yard crane is engaged in reshuffling. Lift-AGVs drop containers onto ground stools in $<15\text{ seconds}$ and immediately depart. This eliminates crane-vehicle synchronization waiting, lowering effective cycle time by $20\% \text{ to } 30\%$ and reducing the required fleet size by the same ratio. Quayside STS cranes never suffer from horizontal transport starvation.
- **Weak Answer Reveals:** Believes Lift-AGVs are simply faster in cruising speed; misses the concept of passive ground stools; unaware of crane-vehicle dead-wait synchronization bottlenecks.

### Question 3: Remote Operating Station (ROS) Latency Limits
**Prompt:** What is the maximum acceptable glass-to-glass video latency for an ROS tele-operation console controlling an ASC landing on an external road truck, and what physical phenomena occur if this budget is exceeded?
- **Strong Answer Contains:** Specifies an upper threshold of $45 \text{ to } 50\text{ ms}$ total glass-to-glass latency (camera sensor capture $\to$ H.265 encode $\to$ 5G URLLC radio $\to$ edge decode $\to$ monitor display refresh). Explains that exceeding $60 \text{ to } 80\text{ ms}$ triggers **operator-induced oscillation (PIO)**: the human operator observes an alignment error, applies joystick correction, but due to visual lag, applies correction after the spreader has already passed the target, causing violent pendulum swinging. This causes landing misses, twistlock damage, operator motion sickness, and severe truck turnaround delays.
- **Weak Answer Reveals:** Suggests that standard internet streaming latency (200–500 ms) is acceptable; unfamiliar with operator-induced oscillation; cannot itemize the latency budget stages.

### Question 4: Civil Engineering Runway Piling in Brownfield Conversions
**Prompt:** Why can't a brownfield terminal converting from manual rubber-tyred gantry cranes (RTGs) to rail-mounted ASCs simply lay crane rails directly on the existing yard asphalt?
- **Strong Answer Contains:** Compares RTG dispersed wheel loads on rubber tires (typically 4 or 8 tires spreading loads over asphalt) with the concentrated dynamic point loads of rail-mounted ASCs ($>300 \text{ to } 500\text{ kN per wheel}$ on steel rails). Explains that port terminals are built on reclaimed, dredged alluvial soils prone to secondary consolidation. Unpiled asphalt foundations will experience differential settlement under high-load steel wheels. Differential settlement causes track gauge distortion, excessive rail clip fatigue, structural gantry racking, drive motor over-torque alarms, and wheel derailment. Deep driven piles down to bedrock are structurally mandatory.
- **Weak Answer Reveals:** Thinks rails are only needed to guide the crane; unaware of soil bearing capacity limits, dynamic impact factors, or differential settlement mechanics.

### Question 5: Battery Charging Sizing and Grid Queuing
**Prompt:** When sizing an opportunity charging network for electric AGVs, how do you mathematically model the risk of an AGV running out of battery charge while waiting in line for an open charging stall?
- **Strong Answer Contains:** Formulates the charging station as an $M/M/c$ or $M/G/c$ multiserver queueing system where arrival rate $\lambda_{\text{ch}} = \lambda_{\text{peak}} / k_{\text{moves}}$ and service rate $\mu_{\text{ch}} = 1 / T_{\text{charge}}$. Applies the Erlang-C formula to compute the waiting probability $P_{\text{wait}}$ and expected queue waiting duration $W_{\text{queue}}$. Integrates this queue waiting time into the minimum operational State of Charge ($\text{SoC}_{\text{reserve}}$) threshold: the vehicle must initiate its charging dispatch when remaining battery energy exceeds $E_{\text{reserve}} = P_{\text{aux}} \cdot W_{\text{queue, 99\%}} + E_{\text{travel, stall}}$, ensuring that even under a 99th-percentile queue delay, the vehicle never suffers an in-transit low-voltage shutdown.
- **Weak Answer Reveals:** Suggests simply dividing total daily energy by charger power; ignores stochastic queueing delays; unaware of how queue waiting time impacts minimum battery safety reserve thresholds.

### Question 6: Labor Union Transition and Collective Bargaining
**Prompt:** When presenting a terminal automation business case in a unionized port with strong collective bargaining rights, what strategies prevent catastrophic labor unrest and align labor with capital investment?
- **Strong Answer Contains:** Emphasizes that successful automation projects treat labor transition costs as a mandatory capital line item (severance, early retirement, guaranteed pension contributions, container royalty funds). Recommends establishing transparent, multi-year transition committees, guaranteeing no involuntary layoffs by absorbing headcounts through natural attrition, and creating accredited union training academies to upskill dockworkers into high-paying mechatronics, PLC diagnostics, and ROS operations careers. Cites the Long Beach LBCT / ILWU model.
- **Weak Answer Reveals:** Suggests treating union opposition purely as a legal confrontation or firing the workforce; ignores collective bargaining contractual protections; fails to budget for transition friction and severance packages.

### Question 7: Private 5G URLLC vs Wi-Fi in Container Terminals
**Prompt:** Why have modern automated terminals transitioned from industrial Wi-Fi meshes to Private 5G Standalone (SA) URLLC networks?
- **Strong Answer Contains:** Explains that container yards are dynamic RF environments consisting of moving walls of solid steel (containers stacked 5–6 high). Wi-Fi operating in unlicensed 2.4/5 GHz spectrum suffers from severe multipath cancellation, non-deterministic CSMA/CA contention delays, channel interference from ship radars and external trucks, and slow client roaming handoffs (100–500 ms) as vehicles transition between access points. Private 5G SA operating in dedicated, interference-free spectrum utilizes cellular beamforming, centralized scheduler grants, and seamless soft handovers with deterministic radio latency $<5\text{ ms}$ and $99.999\%$ reliability under 3GPP Rel-16 URLLC.
- **Weak Answer Reveals:** Claims Wi-Fi is simply "slower" in bandwidth; unaware of unlicensed channel contention, roaming disconnects, or steel container multipath physics.

### Question 8: Operating Leverage During Freight Recessions
**Prompt:** An automated terminal and a manual terminal both operate at $50\%$ operating margins during peak freight booms. When global trade volumes drop by $30\%$, why does the automated terminal experience a far more severe collapse in net earnings?
- **Strong Answer Contains:** Explains the concept of **operating leverage** ($\text{DOL} = \frac{\% \Delta \text{EBIT}}{\% \Delta \text{Revenue}}$). A manual terminal has a high variable cost structure dominated by hourly longshore labor; when volume falls, the operator cancels crane gangs and sheds labor shifts, preserving operating cash flow. An automated terminal has substituted variable labor with massive fixed capital costs (interest on debt, depreciation of $\$500\text{M}+$ infrastructure, fixed software licenses, preventive maintenance contracts). These fixed costs cannot be scaled down during volume slumps, causing operating income to collapse rapidly and pushing the terminal below its breakeven crossover volume.
- **Weak Answer Reveals:** Assumes automated terminals always have higher margins under all conditions; fails to distinguish between fixed capital depreciation and variable labor costs.

---

## 15. Key terms

- **AGV (Automated Guided Vehicle):** Unmanned wheeled platform guided by transponders, lasers, or GNSS.
- **Lift-AGV:** AGV equipped with independent hydraulic lifting platforms for buffer decoupling.
- **ASC (Automated Stacking Crane):** Rail-mounted gantry crane operating autonomously in a yard block.
- **Cantilever ASC:** ASC with gantry girders extending laterally beyond rails to service side lanes.
- **Automated e-RTG (ARTG):** Rubber-tyred gantry crane retrofitted with electric power and autonomous steering.
- **Dual-Trolley STS Crane:** Quayside crane with primary and secondary trolleys separated by an intermediate platform.
- **Remote Operating Station (ROS):** Ergonomic multi-monitor desk for remote supervisory human tele-operation.
- **LCPM (Levelized Cost per Move):** Lifetime discounted CAPEX and OPEX divided by lifetime discounted moves.
- **State of Charge (SoC):** Available battery energy expressed as a percentage of nominal capacity ($0-100\%$).
- **Depth of Discharge (DoD):** Percentage of battery capacity consumed during discharge ($100\% - \text{SoC}$).
- **Opportunity Fast-Charging:** High-power DC conductive charging executed during short operational pauses.
- **Battery Swap Station (BSS):** Robotic depot that mechanically replaces depleted battery packs in $<5\text{ minutes}$.
- **URLLC (Ultra-Reliable Low-Latency Communication):** 3GPP 5G service class delivering $<5\text{ ms}$ latency and $99.999\%$ reliability.
- **Gross Moves Per Hour (GMPH):** Total container moves performed divided by total elapsed operational crane hours.
- **Rehandle Ratio:** Average non-productive container shuffling moves required per target container retrieval.
- **Waterside Transfer Zone (WTZ):** Interchange area at the quay-facing end of a yard block for AGV handoffs.
- **Landside Transfer Zone (LTZ):** Interchange area at the hinterland-facing end of a yard block for road truck handoffs.
- **Operating Leverage:** Ratio of fixed to variable operational costs determining profit sensitivity to volume shocks.
- **Crossover Volume:** The annual container throughput where automated LCPM equals manual LCPM.
- **Gordon-Newell Network:** Closed queueing network mathematical framework used for vehicle fleet sizing.
- **Erlang-C Formula:** Queueing model formula determining waiting probabilities in multiserver systems.
- **Runway Beam:** Continuous reinforced concrete beam supporting rail-mounted crane tracks.
- **Differential Settlement:** Uneven vertical subsidence of foundation soil along a crane runway track.
- **Operator-Induced Oscillation (PIO):** Kinetic spreader instability caused by delayed tele-operation visual feedback.
- **Glass-to-Glass Latency:** Total end-to-end delay from camera photon capture to monitor display presentation.
- **Equipment Control System (ECS):** Real-time software routing, dispatching, and controlling terminal equipment.
- **Terminal Operating System (TOS):** High-level software managing vessel planning, yard allocation, and billing.
- **Take-or-Pay Contract:** Concession or carrier agreement guaranteeing minimum revenue regardless of volume.

---

## 16. Sources

- **PIANC Working Group 208 (2014):** *Planning and Design of Automated Container Terminals.* PIANC General Secretariat, Brussels, Belgium. ISBN: 978-2-87223-217-8.
- **ISO 3691-4:2023:** *Industrial trucks — Safety requirements and verification — Part 4: Driverless industrial trucks and their systems.* International Organization for Standardization, Geneva, Switzerland.
- **3GPP Technical Specification 38.300 (Rel-17, 2022):** *NR; NR and NG-RAN Overall Description; Stage 2 (Release 17).* 3rd Generation Partnership Project.
- **IEC 62443-3-3:2018:** *Security for industrial automation and control systems — Part 3-3: System security requirements and security levels.* International Electrotechnical Commission, Geneva.
- **Vis, I. F. A., & de Koster, R. (2003):** Transshipment of containers at a container terminal: An overview. *European Journal of Operational Research*, 147(1), 1–16. https://doi.org/10.1016/S0377-2217(02)00293-X
- **Carlo, H. J., Vis, I. F. A., & Roodbergen, K. J. (2014):** Transport operations in container terminals: Literature review, trends, research directions and categorization framework. *European Journal of Operational Research*, 236(1), 1–13. https://doi.org/10.1016/j.ejor.2013.11.023
- **Steenken, D., Voß, S., & Stahlbock, R. (2004):** Container terminal operation and operations research—a classification and literature review. *OR Spectrum*, 26(1), 3–49. https://doi.org/10.1007/s00291-003-0157-0
- **Saanen, Y. A. (2004):** *An approach for designing robotized maritime container terminals.* Doctoral dissertation, Delft University of Technology, Netherlands. ISBN: 90-5638-124-1.
- **Port of Long Beach (2021):** *Middle Harbor Redevelopment Project: Terminal Automation Environmental & Operational Performance Report.* Long Beach Harbor Commission, CA.
- **UNCTAD (2023):** *Review of Maritime Transport 2023: Towards a Green and Just Transition.* United Nations Conference on Trade and Development, Geneva. UNCTAD/RMT/2023.

---

## 17. Links

- **Parent Module Map:** [`_index.md`](_index.md)
- **Sibling Topic 41.01 (Berth and Quay Crane Operations):** [`01_berth-quay-crane.md`](01_berth-quay-crane.md)
- **Sibling Topic 41.02 (Terminal Yard and Gate Operations):** [`02_terminal-yard-gate.md`](02_terminal-yard-gate.md)
- **Sibling Topic 41.08 (Terminal Simulation & Queuing):** [`08_terminal-simulation.md`](08_terminal-simulation.md)
- **Root Survey File:** [`../../../course/41_maritime_rail_air_operations.md`](../../../course/41_maritime_rail_air_operations.md)
