# Port Operations and Congestion

> Module 36 — Global Freight, Ports, and Trade Policy
> Topic 05 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11 (Safety Stock & Service Levels), Module 14 (Network Design), Module 19 (International Logistics & Customs), Topic 36.04 (Modal Choice and Emissions)
> Version 2026.5

---

## 1. Why this topic exists

Over 80 percent of global merchandise trade by volume moves across maritime supply chains. In containerized freight, the seaport is not an inert transfer node or a simple storage lot. It is a high-velocity, highly stochastic, capital-intensive dynamic interface between oceanic linehaul networks and inland multimodal transport. 

When container terminals operate smoothly, a 20,000 twenty-foot equivalent unit (TEU) ultra-large container vessel (ULCV) can berth, discharge and load 4,500 container moves across five quay cranes within 36 hours, and clear those containers into hinterland rail and trucking corridors within 48 to 72 hours. When terminals fail, the breakdown does not remain localized. It triggers catastrophic nonlinear supply chain contagion across the global economy.

The financial and operational scale of port congestion became undeniable during the 2021–2022 global supply chain disruption. At the height of the crisis in late 2021, over 100 container vessels were held at anchor or drift areas off the San Pedro Bay ports of Los Angeles and Long Beach. Over 500,000 TEUs of cargo sat immobilized on ships outside the port gate, representing more than $40 billion in locked working capital. Transpacific ocean transit times doubled from 20 days to over 50 days, spot freight rates surged by 800 percent from $2,000 to over $18,000 per forty-foot equivalent unit (FEU), and ocean carriers and beneficial cargo owners (BCOs) absorbed tens of millions of dollars per week in vessel charter losses, inventory carrying costs, and demurrage penalties.

The root cause of port gridlock is rarely a failure of a single machine or labor gang. Rather, it is the fundamental mathematical coupling among four interdependent terminal subsystems:
1. **The Berth and Anchorage Subsystem:** Vessel navigation, tidal access channels, tug and pilotage availability, and berth allocation.
2. **The Quay Subsystem:** Ship-to-shore (STS) gantry crane operations, spreader cycle dynamics, and gang scheduling.
3. **The Yard Stacking Subsystem:** Rubber-tyred gantry (RTG), rail-mounted gantry (RMG), or automated stacking crane (ASC) blocks, horizontal transport buffers, and container stacking mechanics.
4. **The Gate and Intermodal Rail Subsystem:** Ingate/outgate optical character recognition (OCR) inspection portals, trouble kiosks, drayage truck appointment systems (TAS), and on-dock intermodal rail yards.

Because these subsystems operate in strict physical series, a capacity degradation in any one subsystem instantly propagates upstream and downstream. If import dwell time in the yard rises due to a shortage of warehouse space or truck chassis in the hinterland, yard stacking utilization surges past 80 percent. At that threshold, yard cranes spend the majority of their operating cycles "reshuffling" (digging out buried containers) rather than servicing trucks or quay shuttles. 

With yard cranes paralyzed by reshuffling, horizontal transport vehicles (terminal tractors or automated guided vehicles) cannot drop off discharged import containers and back up onto the quay apron. Quay cranes stall under hatch covers, vessel service times stretch from 36 hours to 96 hours, berths cannot clear for scheduled arrivals, and incoming ships are turned away to anchorage. At anchorage, queuing delays escalate nonlinearly according to the Pollaczek-Khinchine and Erlang-C formulations.

Logistics engineers, data scientists, and procurement executives who treat port dwell as an exogenous random variable or who model port throughput using uncoupled deterministic averages make multi-million-dollar inventory and routing blunders. Mastering port operations requires understanding the mathematical optimization of berth allocation, the stochastic queuing dynamics of anchorages and gates, the combinatorial mechanics of yard reshuffling, and the diagnostic protocols that isolate true operational bottlenecks from commercial disputes over demurrage.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Deconstruct** modern container terminal operations into four tightly coupled subsystems (Berth, Quay, Yard, Gate), mapping container and information workflows, buffer capacities, and failure propagation vectors.
2. **Formulate** the dynamic continuous Berth Allocation Problem (BAP) and Quay Crane Assignment Problem (QCAP) as a Mixed-Integer Linear Program (MILP) to minimize total vessel turnaround time, anchorage waiting time, and schedule tardiness.
3. **Quantify** anchorage queue formation, berth occupancy, and vessel waiting times using Pollaczek-Khinchine ($M/G/1$) and Erlang-C ($M/M/s$) queuing models, evaluating the severe non-linear impact of service-time variance ($C_s^2$) and utilization ($\rho$).
4. **Model** yard crane interference, non-passing track constraints, and the physics of container re-handling, calculating expected reshuffle moves as a function of stacking height ($H$) and block occupancy ($U$).
5. **Design** an optimized Truck Appointment System (TAS) and gate configuration, applying multi-server queuing theory ($M/G/c$) to size appointment quotas, eliminate highway queuing spillover, and maximize dual-transaction matchbacks.
6. **Diagnose** the structural root causes of container dwell time inflation across import, export, and empty flows using parametric duration analysis (log-normal and Weibull models) and audit decision trees.
7. **Construct** an operational terminal simulation and queuing evaluation engine in Python to evaluate terminal resilience, calculate bottleneck shift points, and forecast operational recovery trajectories under volume surges.

---

## 3. The core mental model

A container terminal is a multi-stage closed-loop queuing network bounded by oceanic shipping on the waterside and inland road/rail networks on the landside. The core mental model is the **Coupled Four-Stage Terminal Flow Pipeline**.

Container movements are discrete, but the capacity of each subsystem acts as a physical buffer. If any buffer reaches saturation, backpressure propagates upstream, while flow starvation propagates downstream.

```text
=============================================================================================================
                      THE FOUR-STAGE COUPLED CONTAINER TERMINAL PIPELINE
=============================================================================================================

[ WATERSIDE LOGISTICS ]
 Inbound Vessels ──► [ Anchorage Queue ] ──► [ Nav Channel & Pilot ] ──► [ Berths 1 .. s ]
                           ▲ (Holding Area)                                    │
                           │                                                   ▼
===========================│===================================================│=============================
[ QUAY CRANE SUBSYSTEM ]   │                                                   │
                           │   ┌───────────────────────────────────────────────┘
                           │   ▼
                     [ STS Quay Cranes 1 .. K ] ◄── Spreader Cycle (Discharge / Load Moves)
                               │         ▲
                               ▼         │ Horizontal Transfer (AGV / Terminal Tractor / Straddle Carrier)
===============================│=========│===================================================================
[ YARD STACKING SUBSYSTEM ]    │         │
                               ▼         │
                       [ Yard Buffer / Stacking Blocks ]
                       - ASC / RMG / RTG Crane Gantries
                       - Import Stacks (High Reshuffling Risk: H tiers)
                       - Export Stacks (Marshalling / Segregation)
                       - Empty Container Depot & Reefer Racks
                               │         ▲
                               │         │ Internal Retrieval / Delivery
===============================│=========│===================================================================
[ LANDSIDE GATE & RAIL ]       │         │
                               ▼         │
                       [ Terminal Gate Complex ]          [ On-Dock Intermodal Rail ]
                       - OCR Portals & WIM Scales         - Cantilever RMGs
                       - Driver Kiosks / Trouble Pedestal - Double-Stack Block Trains
                       - Truck Appointment System (TAS)
                               │                                  │
                               ▼                                  ▼
                    [ Regional Drayage / Highway ]      [ Class I / National Rail Network ]
=============================================================================================================
```

### The Dynamics of Subsystem Coupling

The four subsystems do not operate independently. Their performance is governed by strict conservation of flow and capacity constraints:

1. **Waterside-Quay Interface (Berth Allocation):**
   A vessel cannot discharge containers until it is moored at a quay with adequate draft, length overall ($LOA$), and allocated Ship-to-Shore (STS) cranes. The vessel service rate is determined by the number of assigned cranes multiplied by the net crane productivity (gross crane moves per hour minus interference and lashing delays).

2. **Quay-Yard Interface (Horizontal Transport Buffer):**
   Between the quay and the yard stacks operates a fleet of internal transfer vehicles: Terminal Tractors (TT), Automated Guided Vehicles (AGV), or Straddle Carriers (SC). If the yard cranes take too long to receive an import container (due to crane digging or queue congestion in the block), the horizontal transport vehicles queue up along the quay crane back-reach. When all transfer vehicles are queued, the STS crane cannot place containers down and must idle—a condition known as **crane starvation** or **quay lock**.

3. **Yard-Gate Interface (Landside Servicing):**
   Yard cranes (RTGs or ASCs) have dual responsibilities: they must service waterside transfers (discharging imports and feeding exports to the vessel) and landside transfers (loading highway drayage trucks and on-dock rail wagons). If landside drayage trucks arrive in uncoordinated surges, yard cranes divert capacity to truck loading, starving the quay, or conversely, prioritize the ship, causing truck queues at the gate to spill out onto public highways.

4. **The Yard Utilization Tipping Point:**
   The container yard is the central buffer of the entire terminal. When yard occupancy is below 70 percent, empty ground slots are readily accessible. An import container stacked at tier 3 or 4 can be uncovered by moving covering containers to adjacent empty slots in the same bay.
   
   When yard occupancy surpasses the **75 to 80 percent tipping point**, adjacent empty slots disappear. To dig out a buried container, the yard crane must lift the top containers, gantry down the block to find an open slot in another bay, drop the obstruction, gantry back, lift the target container, and then repeat the move. Expected crane cycle time triples from 75 seconds to over 220 seconds. Net yard crane capacity collapses just when demand is highest. The terminal enters a self-reinforcing congestion death spiral.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Industry Misuse | Standard Unit / Dimension |
| :--- | :--- | :--- | :--- |
| **Twenty-Foot Equivalent Unit (TEU)** | Standard measure of container volume based on a nominal 20-foot ISO intermodal shipping container ($20' \times 8' \times 8.5'$). | Treating 1 physical container as 1 TEU (a 40-foot container is 2 TEUs; a 45-foot container is 2.25 TEUs). | TEU (Dimensionless count) |
| **Berth Allocation Problem (BAP)** | The mathematical scheduling problem of assigning arriving vessels to specific quay space and time windows. | Conflating discrete berth allocation (fixed individual berths) with continuous berth allocation (quay treated as a continuous length). | Optimization model / Coordinates $(m, t)$ |
| **Quay Crane Assignment Problem (QCAP)** | The allocation of specific ship-to-shore gantry cranes to berthed vessels over discrete time horizons. | Assuming crane productivity scales linearly with crane count without factoring in crane interference. | Integer (crane-hours) |
| **Berth Productivity (BMPH)** | Total container moves executed across all cranes assigned to a vessel divided by total hours moored at berth. | Conflating net operating hours with gross berth hours (which includes lashing, unlashing, and draft surveys). | Gross Crane Moves / Berth Hour |
| **Crane Productivity (CMPH)** | The number of container lifts completed by a single quay crane within one operating hour. | Using gross moves instead of net moves (moves adjusted for hatch cover shifts and gear handling). | Container Lifts / Net Crane Hour |
| **Ship-to-Shore (STS) Crane** | Rail-mounted gantry crane operating along the quay wall to load and unload containers from cellular container ships. | Confusing an STS crane with yard gantry cranes (RTG or RMG). | Equipment classification |
| **Rubber-Tyred Gantry (RTG)** | Mobile gantry crane running on rubber tires, used to stack containers 1-over-4 to 1-over-6 high across 6 to 8 container rows. | Assuming RTGs have fixed rail paths; RTGs can change lanes via 90-degree wheel turns. | Equipment classification |
| **Automated Stacking Crane (ASC)** | Unmanned, rail-mounted gantry crane deployed in perpendicular or parallel automated yard blocks. | Assuming ASCs never experience interference; dual ASCs on the same rail track clash frequently. | Equipment classification |
| **Internal Transfer Vehicle (ITV)** | Tractor-trailer combination (terminal tractor/yard hustler) or AGV dedicated to moving containers between quay and yard. | Conflating intra-terminal transfer equipment with external street-legal drayage trucks. | Fleet count / Vehicle units |
| **Import Dwell Time** | Elapsed duration from the moment a container is discharged from a vessel to its physical exit through the terminal outgate. | Measuring dwell time only from customs release rather than physical discharge. | Days / Hours |
| **Export Dwell Time** | Elapsed duration from the moment an export container enters the ingate to its physical loading onto a vessel. | Confusing dwell time with the export cargo cut-off window. | Days / Hours |
| **Free Time** | Contractually permitted period during which an import or export container may occupy terminal space without incurring demurrage. | Believing free time is mandated by port authorities rather than set by carrier service contracts or terminal tariffs. | Calendar or Working Days |
| **Demurrage** | Financial penalty assessed by the terminal operator or ocean carrier for container storage in the yard beyond granted free time. | Conflating demurrage (storage inside terminal) with detention (equipment retention outside terminal). | USD / Day / TEU |
| **Detention (Per Diem)** | Financial penalty assessed by the equipment owner (ocean carrier or leasing company) for holding empty containers or chassis outside the terminal past free time. | Conflating detention with port demurrage or terminal storage charges. | USD / Day / Equipment Unit |
| **Reshuffling (Re-handling)** | Unproductive crane move required to lift, relocate, and restack non-target containers covering a target container. | Omitting reshuffles from yard operational cost and crane productivity models. | Moves per target retrieval |
| **Yard Utilization Rate ($U$)** | Ratio of currently occupied ground slots (factoring stacking height) to maximum nominal static capacity. | Assuming a yard can operate at 100% capacity; capacity above 80% causes crane productivity collapse. | Percentage (%) |
| **Truck Appointment System (TAS)** | Digital reservation platform requiring drayage carriers to book specific hourly time windows to pick up or drop off containers. | Viewing TAS purely as a traffic control tool rather than a dynamic yard crane workload leveling mechanism. | Protocol / Operational Policy |
| **Gate Turn Time (GTT)** | Total elapsed time from an external truck arriving at the terminal queue/OCR portal to its departure through the outgate. | Measuring only pedestal-to-pedestal time while ignoring approach highway queue time. | Minutes |
| **Trouble Ticket** | An exception flag generated at the ingate pedestal due to missing documentation, customs hold, seal mismatch, or weight discrepancy. | Underestimating trouble ticket frequency; a 5% exception rate can double gate turn times. | Process exception state |
| **Dual Transaction (Matchback)** | A single drayage gate visit that executes both an inbound drop-off (export or empty) and an outbound pickup (import or empty). | Assuming all visits are dual; single "bobtail" visits waste 50% of gate and yard transfer capacity. | Dimensionless ratio ($\le 1.0$) |
| **Verified Gross Mass (VGM)** | Certified total weight of a packed container declared by the shipper under IMO SOLAS Chapter VI regulations. | Accepting shipper estimates without calibrated Weigh-in-Motion (WIM) or weighbridge verification. | Kilograms (kg) or Metric Tons |
| **On-Dock Rail (ODR)** | Rail intermodal terminal located directly inside the marine terminal footprint, serviced by terminal transfer vehicles. | Conflating on-dock rail with near-dock or off-dock intermodal container transfer facilities (ICTF). | Modal infrastructure |
| **Anchorage Waiting Time ($W_q$)** | Duration a vessel spends at designated maritime anchorage grounds waiting for an assigned berth to become available. | Measuring vessel delay only against initial carrier schedule rather than port call confirmation. | Hours / Days |
| **Pollaczek-Khinchine (P-K)** | Exact analytical queuing formula for the expected waiting time in an $M/G/1$ single-server queue with general service distribution. | Applying exponential ($M/M/1$) formulas when service time variance deviates significantly from the mean. | Formula / Mathematical relation |
| **Erlang-C Formula** | Analytical formula for the probability that an arriving entity must wait in an $M/M/s$ multi-server queuing system. | Using Erlang-C when vessel arrivals are strictly scheduled and deterministic. | Probability ($P_w \in [0, 1]$) |
| **Port Congestion Surcharge (PCS)** | Emergency fee levied by ocean carriers on cargo owners to offset vessel idling and operating costs at congested ports. | Believing PCS is a terminal fee; it is an ocean carrier tariff assessment. | USD / TEU |

---

## 5. The mechanics, step by step

Understanding terminal congestion requires breaking down the physical and analytical mechanics of all four subsystems, starting from maritime arrival and ending at the highway/rail gate interface.

```text
=============================================================================================================
                               TERMINAL SUB-PROCESS FLOWCHART
=============================================================================================================

1. Maritime Arrival      2. Berth Allocation     3. Quay Handling       4. Yard Management     5. Landside Gate
   ┌───────────────┐        ┌──────────────┐        ┌──────────────┐       ┌──────────────┐       ┌──────────────┐
   │ Vessel Enters │  ETA   │ Solve BAP /  │ Berth  │ Deploy STS   │ Cycle │ Assign Block │ Dwell │ TAS Booking  │
   │ Port Waters / ├───────►│ QCAP MILP    ├───────►│ Cranes       ├──────►│ & Manage Re- ├──────►│ & OCR Gate   │
   │ Anchorage     │ Queue  │ Optimization │ Window │ Disch / Load │ Times │ handling     │ Days  │ Processing   │
   └───────────────┘        └──────────────┘        └──────────────┘       └──────────────┘       └──────────────┘
```

---

### Step 1: Vessel Arrival and Anchorage Queuing Mechanics

Vessel traffic arriving at a port approaches through designated shipping channels and drops anchor in maritime anchorage sectors if a berth is unavailable. 

#### A. The Stochastic Arrival and Service Process
While ocean carriers operate published liner schedules with fixed weekly arrival windows (proforma schedules), weather conditions, chokepoint delays, and prior port dwell introduce severe stochastic jitter. Realized inter-arrival times follow an exponential distribution or a displaced log-normal distribution:
$$f(t) = \frac{1}{t \sigma_a \sqrt{2\pi}} \exp\left( -\frac{(\ln t - \mu_a)^2}{2\sigma_a^2} \right)$$

Vessel service time $S$ at berth is not constant. It is a function of:
- Total container moves (discharge, load, restow).
- Number of assigned quay cranes ($q_v$).
- Gross crane productivity ($\mu_c$ moves/hour).
- Non-productive handling: lashing/unlashing, twistlock removal, hatch cover moves, boom-up clearance, draft surveys, and labor shift changes.

Service time variance $\sigma_s^2$ is driven by vessel size mix, stowage plan complexity, crane mechanical failures, and adverse weather (high winds forcing STS cranes to boom up).

#### B. Single-Berth or Priority Berth Queuing: The Pollaczek-Khinchine Formula
For a dedicated berth or single specialized channel modeled as an $M/G/1$ queue (Poisson arrivals at rate $\lambda$, general independent service times with mean $\mathbb{E}[S] = 1/\mu$ and variance $\sigma_s^2$):

1. **Traffic Intensity (Server Utilization):**
   $$\rho = \lambda \mathbb{E}[S] = \frac{\lambda}{\mu} < 1$$
   If $\rho \ge 1$, the queue grows infinitely; the anchorage backlog cannot clear.

2. **Squared Coefficient of Variation ($C_s^2$):**
   $$C_s^2 = \frac{\sigma_s^2}{(\mathbb{E}[S])^2}$$

3. **Second Moment of Service Time:**
   $$\mathbb{E}[S^2] = \sigma_s^2 + (\mathbb{E}[S])^2 = (\mathbb{E}[S])^2 (1 + C_s^2)$$

4. **Expected Anchorage Waiting Time ($W_q$):**
   According to the Pollaczek-Khinchine (P-K) formula:
   $$W_q = \frac{\lambda \mathbb{E}[S^2]}{2(1 - \rho)} = \frac{\rho \mathbb{E}[S]}{2(1 - \rho)} \left(1 + C_s^2\right)$$

5. **Expected Total Time in Port ($W$) and Anchorage Queue Length ($L_q$):**
   $$W = W_q + \mathbb{E}[S]$$
   $$L_q = \lambda W_q = \frac{\rho^2}{2(1 - \rho)} \left(1 + C_s^2\right)$$

> **Note:** Notice the term $(1 + C_s^2)$. 
> - If service times are strictly deterministic ($C_s^2 = 0$, an $M/D/1$ queue), $W_q$ is exactly half of the waiting time of an exponential queue ($M/M/1$, where $C_s^2 = 1$). 
> - If operational disruptions cause high variability ($C_s = 1.2, C_s^2 = 1.44$), waiting time increases by 22 percent above an exponential queue, and nearly 2.5 times that of a deterministic queue. Berth scheduling discipline directly compresses anchorage waiting time without building new berths.

#### C. Multi-Berth Common User Terminals: The Erlang-C Model
For a terminal with $s$ identical parallel berths modeled as an $M/M/s$ system:

1. **Offered Load ($a$):**
   $$a = \frac{\lambda}{\mu} = \lambda \mathbb{E}[S]$$

2. **Berth Utilization ($\rho$):**
   $$\rho = \frac{a}{s} = \frac{\lambda}{s \mu} < 1$$

3. **Probability of System Idleness ($P_0$):**
   $$P_0 = \left[ \sum_{k=0}^{s-1} \frac{a^k}{k!} + \frac{a^s}{s! (1 - \rho)} \right]^{-1}$$

4. **Erlang-C Probability of Waiting ($P_w$ or $C(s, a)$):**
   $$P_w = \frac{\frac{a^s}{s! (1 - \rho)}}{\sum_{k=0}^{s-1} \frac{a^k}{k!} + \frac{a^s}{s! (1 - \rho)}} = \frac{a^s}{s! (1 - \rho)} P_0$$

5. **Expected Waiting Time at Anchorage ($W_q$):**
   $$W_q = \frac{P_w}{s \mu - \lambda} = \frac{P_w \cdot \mathbb{E}[S]}{s(1 - \rho)}$$

6. **General Service Time Correction (Lee-Longton Approximation for $M/G/s$):**
   When service times are general with squared coefficient of variation $C_s^2$:
   $$W_q(M/G/s) \approx \left(\frac{1 + C_s^2}{2}\right) W_q(M/M/s)$$

---

### Step 2: The Continuous Berth Allocation Problem (BAP) MILP Formulation

Once a vessel is cleared to enter the port channel, the terminal operator must assign it a physical position along the continuous quay wall and a precise berthing time window.

```text
=============================================================================================================
                  2D SPACE-TIME RECTANGLE PACKING IN CONTINUOUS BAP
=============================================================================================================
Berth
Length
  ^
L │                    ┌──────────────────┐
  │                    │ Vessel 3         │
  │     ┌──────────────┤ (LOA_3, p_3)     │
  │     │ Vessel 1     └──────────────────┘
  │     │ (LOA_1, p_1)
  │     └──────────────┐        ┌─────────────────────────┐
  │                    │        │ Vessel 2 (LOA_2, p_2)   │
0 └────────────────────┴────────┴─────────────────────────┴────────► Time (Hours)
  0                    b_1      b_2                       b_3
```

In continuous BAP, the quay is modeled as a continuous line of length $L_{\text{quay}}$, and the planning horizon is $T$. Each vessel $i \in V$ is represented as a 2D rectangle in space-time:
- Horizontal dimension: Duration of stay $p_i$.
- Vertical dimension: Length overall plus safety margin $L_i = LOA_i + \Delta_L$.

#### Mathematical Formulation

**Sets and Parameters:**
- $V$: Set of arriving vessels, indexed by $i, j \in \{1, 2, \dots, N\}$.
- $a_i$: Estimated arrival time (ETA) of vessel $i$.
- $d_i$: Requested departure time (proforma schedule window) of vessel $i$.
- $L_i$: Total berth length required by vessel $i$ (including safety clearances).
- $L_{\text{quay}}$: Total usable length of the continuous quay.
- $p_i$: Service time of vessel $i$ (assumed fixed or crane-dependent).
- $m_i^*$: Optimal berthing coordinate along the quay (closest to assigned import/export yard blocks, minimizing yard drayage).
- $c_1, c_2, c_3$: Cost weighting factors for anchorage waiting time, departure delay (tardiness), and yard drayage deviation.
- $M$: A sufficiently large positive scalar (Big-M).

**Decision Variables:**
- $b_i \ge 0$: Berthing start time of vessel $i$.
- $m_i \in [0, L_{\text{quay}} - L_i]$: Berthing position of vessel $i$ along the quay.
- $c_i = b_i + p_i$: Departure time of vessel $i$.
- $z_{ij}^x \in \{0, 1\}$: Binary variable = 1 if vessel $i$ is completely to the left of vessel $j$ along the quay ($m_i + L_i \le m_j$).
- $z_{ij}^y \in \{0, 1\}$: Binary variable = 1 if vessel $i$ departs before vessel $j$ berths ($c_i \le b_j$).

**Objective Function:**
$$\min Z = \sum_{i \in V} \left[ c_1 (b_i - a_i) + c_2 \max(0, c_i - d_i) + c_3 |m_i - m_i^*| \right]$$

**Subject to Constraints:**
1. **Arrival Feasibility:** A ship cannot berth before it physically arrives at the port:
   $$b_i \ge a_i \quad \forall i \in V$$

2. **Spatial Quay Boundaries:** The vessel must physically fit within the quay boundaries:
   $$m_i + L_i \le L_{\text{quay}} \quad \forall i \in V$$
   $$m_i \ge 0 \quad \forall i \in V$$

3. **Spatial Non-Overlap Constraints (Big-M formulation):**
   $$m_i + L_i \le m_j + M(1 - z_{ij}^x) \quad \forall i, j \in V, i \ne j$$
   $$m_j + L_j \le m_i + M(1 - z_{ji}^x) \quad \forall i, j \in V, i \ne j$$

4. **Temporal Non-Overlap Constraints (Big-M formulation):**
   $$b_i + p_i \le b_j + M(1 - z_{ij}^y) \quad \forall i, j \in V, i \ne j$$
   $$b_j + p_j \le b_i + M(1 - z_{ji}^y) \quad \forall i, j \in V, i \ne j$$

5. **Disjunctive 2D Separation Constraint:** At least one separation condition must hold for any pair of vessels:
   $$z_{ij}^x + z_{ji}^x + z_{ij}^y + z_{ji}^y \ge 1 \quad \forall i, j \in V, i < j$$

#### Quay Crane Assignment Problem (QCAP) Coupling
In advanced formulations (Berth Allocation and Crane Assignment Problem, BACAP), service time $p_i$ is an endogenous decision variable that decreases as more quay cranes $q_i(t)$ are allocated to vessel $i$:
$$p_i \approx \frac{\text{Total Container Moves}_i}{\sum_{t} q_i(t) \cdot \mu_c \cdot (1 - \beta_{\text{crane}} (q_i(t) - 1))}$$
where $\beta_{\text{crane}} \approx 0.05 - 0.08$ is the crane interference factor (diminishing marginal returns per crane due to physical clearance constraints along the ship). Furthermore, STS cranes cannot cross over one another because they run on the same set of continuous rails.

---

### Step 3: Horizontal Transport and Quay-Yard Interface Dynamics

When an STS crane hoists a container off the ship, it lowers it through the crane portal onto an internal transfer vehicle (ITV).

#### Horizontal Transport Fleet Sizing
The required fleet size $N_{\text{ITV}}$ to sustain a targeted quay crane throughput is determined by closed queuing network equations:
$$\text{Cycle Time } T_{\text{cycle}} = t_{\text{load\_crane}} + t_{\text{travel\_loaded}} + t_{\text{queue\_yard}} + t_{\text{offload\_yard}} + t_{\text{travel\_empty}} + t_{\text{queue\_quay}}$$

To prevent crane starvation, the arrival rate of empty ITVs under the crane must match or exceed the crane lift rate $\mu_c$:
$$N_{\text{ITV}} \ge \mu_c \cdot \left(\frac{T_{\text{cycle}}}{60}\right) \cdot q_{\text{vessel}}$$

If yard cranes become congested, $t_{\text{queue\_yard}}$ expands. $T_{\text{cycle}}$ surges from 12 minutes to 30 minutes. If $N_{\text{ITV}}$ is fixed, the effective ITV arrival rate drops below $\mu_c$, and the STS crane is forced to idle with the spreader in mid-air.

---

### Step 4: Yard Operations, Stacking Mechanics, and Crane Interference

The container yard is organized into rectangular blocks consisting of 20 to 30 bays (longitudinal), 6 to 8 rows (lateral), and 4 to 6 tiers (vertical).

```text
=============================================================================================================
                    CROSS-SECTION OF A TYPICAL CONTAINER STACKING BAY
=============================================================================================================
Tier
  ▲
5 │    [   ]         [   ]         [   ]  ◄── Top Tier Containers (Must be moved to access Tier 1)
4 │    [ C1]         [ C4]         [ C7]
3 │    [ C2]         [ C5]         [ C8]
2 │    [ C3]         [TARGET]      [ C9]
1 │    [   ]         [   ]         [   ]  ◄── Ground Slots (Tier 1)
  └──────┬─────────────┬─────────────┬──────► Rows
       Row 1         Row 2         Row 3
```

#### A. Stacking Physics and Re-Handling (Reshuffling) Formula
Containers are retrieved on demand by external trucks or waterside shuttles. In import stacks, retrieval sequences are highly uncertain because truckers arrive according to customer schedules, customs releases, and carrier availability.

If a target container is stored at tier $k$ (where tier 1 is the bottom slot on the ground and tier $H$ is the highest container currently stacked in that column):
$$\text{Number of unproductive reshuffles required } R_k = H - k$$

Assuming uniform retrieval probability across all $H$ containers in a stack of height $H$:
$$\mathbb{E}[R(H)] = \frac{1}{H} \sum_{k=1}^H (H - k) = \frac{1}{H} \left( \frac{H(H - 1)}{2} \right) = \frac{H - 1}{2}$$

| Stacking Height ($H$) | Expected Reshuffles per Retrieval ($\mathbb{E}[R]$) | Net Lifts per Target Container ($1 + \mathbb{E}[R]$) | Operational Impact |
| :---: | :---: | :---: | :--- |
| **1-over-2 ($H=2$)** | 0.50 | 1.50 | Low density; minimal reshuffling; huge yard footprint. |
| **1-over-3 ($H=3$)** | 1.00 | 2.00 | Typical for low-density chassis or straddle carrier yards. |
| **1-over-4 ($H=4$)** | 1.50 | 2.50 | Standard RTG operation; 60% of crane moves are unproductive. |
| **1-over-5 ($H=5$)** | 2.00 | 3.00 | Automated stacking yard (ASC); 67% of crane moves are unproductive. |
| **1-over-6 ($H=6$)** | 2.50 | 3.50 | Extreme density (Singapore, Hong Kong); requires smart housekeeping. |

#### B. The Occupancy Penalty and Non-Linear Explosion
The formula $\mathbb{E}[R] = (H - 1)/2$ assumes that the crane can immediately drop reshuffled containers into an adjacent empty slot in the exact same bay. 

When yard occupancy $U = \text{Occupied Slots} / \text{Total Slots}$ exceeds 75 percent, adjacent slots are frequently occupied. The yard crane must hoist the obstructing container, gantry down the block to another bay, lower it, return to the bay, and repeat. 
Kim (1997) and Dekker et al. show that the effective reshuffle penalty scales nonlinearly with yard utilization:
$$\mathbb{E}[R_{\text{effective}}] \approx \frac{H - 1}{2} \cdot \left( 1 + \frac{\alpha U^2}{1 - U} \right)$$
where $\alpha$ is a block layout coefficient ($\approx 0.15 - 0.25$). As $U \to 1.0$, the denominator $(1 - U) \to 0$, and expected crane handling time per container explodes toward infinity.

#### C. Yard Crane Interference on Shared Rails
In modern automated terminals, blocks are serviced by Rail-Mounted Gantry Cranes (RMGs) or Automated Stacking Cranes (ASCs). Often, two ASCs share the same physical rail track over a block:
- ASC 1 services the waterside apron (interfacing with AGVs).
- ASC 2 services the landside interchange (interfacing with road trucks).

Because the two cranes run on the same physical rails, they cannot cross each other:
$$x_{\text{crane\_1}}(t) + \Delta_{\text{safety}} \le x_{\text{crane\_2}}(t) \quad \forall t$$

When ASC 1 must access a bay near the landside end, ASC 2 must yield its position, abort or pause truck servicing, and travel to a parking spur or buffer zone. The resulting interference factor $\beta_{\text{yard}}$ reduces the combined productivity of twin ASCs by 15 to 25 percent:
$$\text{Net Block Moves/Hour} = (CMPH_1 + CMPH_2) \cdot (1 - \beta_{\text{yard}})$$

---

### Step 5: Gate Operations and Truck Appointment Systems (TAS)

The terminal gate controls landside access. Drayage trucks move through several sequential inspection layers:
1. **Approach and Staging:** Queuing lanes on public access roadways or terminal entry aprons.
2. **OCR Portal:** Optical Character Recognition scans the container ISO code, license plate, USDOT number, and chassis VIN at 10 mph.
3. **Weigh-In-Motion (WIM) Scales:** Verifies axle loads and measures gross weight against the IMO SOLAS Verified Gross Mass (VGM).
4. **Driver Pedestals / Kiosks:** Driver inserts RFID badge/TWIC, confirms booking number, takes physical paper routing slip (or digital push notification).
5. **Exception Gate (Trouble Kiosk):** Trucks with missing customs clearance, seal mismatches, or unpaid demurrage are diverted out of active lanes.

```text
=============================================================================================================
                              TERMINAL GATE PROCESSING PIPELINE
=============================================================================================================

Inbound Drayage ──► [ Approach Queue ] ──► [ OCR Portal & WIM ] ──► [ Ingate Pedestals ] ──► [ Yard Entry ]
                                                                             │
                                                                   Exception ▼ (Trouble)
                                                                    [ Trouble Parking ]
```

#### Gate Queuing Formulation ($M/G/c$)
If trucks arrive uncoordinated at peak hours (e.g., 08:00–10:00 after local warehouses open), arrival rate $\lambda_{\text{gate}}$ exceeds the aggregate service rate of $c$ active gate lanes ($c \cdot \mu_{\text{gate}}$). 
Under an $M/G/c$ queuing approximation (Cosmetatos formula):
$$W_q(M/G/c) \approx \Psi(c, \rho, C_s^2) \cdot W_q(M/M/c)$$
where $W_q(M/M/c)$ is derived from the multi-server Erlang-C equation, and $\Psi$ is a correction factor based on the squared coefficient of variation of gate processing times $C_{s,\text{gate}}^2$.

#### Truck Appointment System (TAS) Quota Sizing
To prevent gate queuing collapse, modern terminals mandate a Truck Appointment System (TAS). The operating day is partitioned into time windows $t \in \{1, 2, \dots, T\}$ (typically 1-hour slots). 

The terminal optimization engine calculates hourly appointment quotas $Q_t$ constrained by:
1. Gate lane capacity: $Q_t \le c_{\text{gate}} \cdot \mu_{\text{gate}} \cdot \Delta t \cdot \rho_{\text{target}}$ (where $\rho_{\text{target}} \le 0.85$ to ensure queues do not form).
2. Yard crane dispatch capacity per block $b$: $\sum_{i \in \text{Block } b} \text{Appointments}_{i, t} \le \text{Net Moves per Hour}_b \cdot \Delta t$.
3. Dual-transaction incentives: Prioritizing bookings where a driver drops an export/empty and picks up an import, doubling terminal lane asset efficiency:
   $$\text{Asset Productivity Ratio} = \frac{\text{Import Lifts} + \text{Export Lifts}}{\text{Total Gate Visits}}$$

---

### Step 6: Dwell Time Decomposition and Hinterland Bottleneck Diagnosis

Container dwell time in the yard represents the operational inventory buffer of the supply chain. Dwell time is not normally distributed; it follows a heavy-tailed, positively skewed log-normal or Weibull distribution:
$$f(t; \lambda_{\text{dwell}}, k) = \frac{k}{\lambda_{\text{dwell}}} \left(\frac{t}{\lambda_{\text{dwell}}}\right)^{k-1} e^{-(t/\lambda_{\text{dwell}})^k}$$

#### The Dwell Time Free-Time Cliff
Under standard marine terminal tariffs:
- Days 1 to 4: Free Time (no storage fees). Pickups occur steadily.
- Day 5: Expiration of Free Time. Pickups peak sharply on Day 4 and early Day 5 as shippers race to avoid charges.
- Days 6+: Demurrage accrual. Containers remaining past Day 6 often exhibit an "abandonment threshold." If demurrage charges exceed the commercial value of the freight (or the margin of the goods), beneficial cargo owners stop prioritizing retrieval, letting dwell escalate to 30+ days.

```text
Container
Retrieval
Rate
  ^
  │              Peak Pickups (Avoiding Demurrage)
  │                    ┌─┐
  │                   ┌┘ └┐
  │                  ┌┘   └┐
  │     ┌────────────┘     └────────────────────────┐
  │    ┌┘                                           └─── Long Dwell Tail
  │   ┌┘                                                 (Disputes, Customs Holds,
  │  ┌┘                                                   Insolvent Consignees)
0 └──┴───────────────┴─────┴────────────────────────┴────────► Dwell Time (Days)
     0               3     4 (Free Time Limit)      10
```

#### The Four Dwell Disruption Vectors
When average dwell expands from 3.5 days to 8.0 days, total containers in the yard more than double ($L = \lambda \cdot W$), driving yard occupancy past the 80 percent tipping point:
1. **Warehouse / DC Congestion:** Consignees have full warehouses and intentionally use the port as a high-cost storage yard.
2. **Chassis Dislocation:** Draymen cannot pick up import containers because empty containers mounted on chassis are blocked from returning to the terminal due to empty-return restrictions.
3. **Intermodal Rail Car Shortage:** Class I railroads fail to supply sufficient double-stack well cars to clear on-dock rail tracks, stranding inland-bound imports for weeks.
4. **Documentation / Customs Holds:** Partner Government Agency (PGA) inspections (CBP, FDA, USDA) or missing bills of lading freeze container movement.

---

## 6. Worked example

### Problem Setup: Apex Gateway Terminal (AGT)

Apex Gateway Terminal is an international container facility operating:
- A continuous quay wall of $L_{\text{quay}} = 1,600 \text{ meters}$.
- $s = 4$ nominal berthing positions for mega-vessels.
- 16 Super Post-Panamax STS Quay Cranes (4 cranes per nominal berth).
- 32 Automated Stacking Cranes (ASCs) operating in 16 yard blocks (2 cranes per block, non-passing rails).
- A 12-lane gate complex ($c_{\text{gate}} = 12$).

The terminal faces severe seasonal peak volume. We will analyze the system end-to-end:
1. Assess waterside anchorage waiting times using Erlang-C and Pollaczek-Khinchine.
2. Solve the Berth Allocation schedule for four incoming vessels.
3. Compute yard re-handling dynamics and crane cycle degradation under 70% vs. 85% yard occupancy.
4. Size the Truck Appointment System (TAS) gate quotas to stabilize turn times.

---

### Step 1: Anchorage Queuing Evaluation

Vessel arrivals during the peak month occur at an average rate of $\lambda = 0.080 \text{ vessels per hour}$ (1 vessel every 12.5 hours, or 13.44 vessels per week).

The mean vessel service time at berth is $\mathbb{E}[S] = 40.0 \text{ hours}$ (representing an average exchange of 3,600 moves across allocated cranes).

1. **System Traffic Intensity and Berth Utilization:**
   $$\text{Offered Load } a = \lambda \cdot \mathbb{E}[S] = 0.080 \times 40.0 = 3.20 \text{ Erlangs}$$
   $$\text{Berth Utilization } \rho = \frac{a}{s} = \frac{3.20}{4} = 0.80 \quad (80.0\%)$$

2. **Calculate Erlang-C Multi-Server Wait Time ($M/M/4$):**
   First, compute state probabilities for $k = 0, 1, 2, 3$:
   $$\frac{a^0}{0!} = \frac{3.20^0}{1} = 1.0000$$
   $$\frac{a^1}{1!} = \frac{3.20^1}{1} = 3.2000$$
   $$\frac{a^2}{2!} = \frac{3.20^2}{2} = \frac{10.24}{2} = 5.1200$$
   $$\frac{a^3}{3!} = \frac{3.20^3}{6} = \frac{32.768}{6} = 5.4613$$
   $$\sum_{k=0}^{3} \frac{a^k}{k!} = 1.0000 + 3.2000 + 5.1200 + 5.4613 = 14.7813$$

   Compute the saturated state term ($k = 4$):
   $$\frac{a^s}{s!(1 - \rho)} = \frac{3.20^4}{24 \times (1 - 0.80)} = \frac{104.8576}{24 \times 0.20} = \frac{104.8576}{4.80} = 21.8453$$

   Compute $P_0$ (probability that all berths are empty):
   $$P_0 = \frac{1}{14.7813 + 21.8453} = \frac{1}{36.6266} = 0.027303$$

   Compute Erlang-C Probability of Waiting $P_w$:
   $$P_w = 21.8453 \times P_0 = 21.8453 \times 0.027303 = 0.5964 \quad (59.64\%)$$

   Compute Expected Anchorage Waiting Time ($W_q$):
   $$\mu = \frac{1}{\mathbb{E}[S]} = \frac{1}{40.0} = 0.025 \text{ vessels/hour}$$
   $$W_q = \frac{P_w}{s\mu - \lambda} = \frac{0.5964}{(4 \times 0.025) - 0.080} = \frac{0.5964}{0.100 - 0.080} = \frac{0.5964}{0.020} = 29.82 \text{ hours}$$

   Expected vessels at anchorage:
   $$L_q = \lambda \cdot W_q = 0.080 \times 29.82 = 2.386 \text{ vessels}$$

3. **Impact of Service Time Variance (Lee-Longton $M/G/4$ Approximation):**
   Suppose winter fog and crane maintenance cause service time standard deviation to be $\sigma_s = 24.0 \text{ hours}$:
   $$C_s = \frac{\sigma_s}{\mathbb{E}[S]} = \frac{24.0}{40.0} = 0.60 \implies C_s^2 = 0.36$$
   $$W_q(M/G/4) \approx \left(\frac{1 + 0.36}{2}\right) \times 29.82 = 0.68 \times 29.82 = 20.28 \text{ hours}$$

   If severe stowage chaos increases variance to $\sigma_s = 48.0 \text{ hours}$ ($C_s = 1.20, C_s^2 = 1.44$):
   $$W_q(M/G/4) \approx \left(\frac{1 + 1.44}{2}\right) \times 29.82 = 1.22 \times 29.82 = 36.38 \text{ hours}$$
   
   A high-variance terminal forces vessels to spend an extra 16.1 hours at anchor compared to a low-variance terminal, even with identical average handling speeds.

---

### Step 2: Berth Allocation Problem (BAP) Schedule

Four vessels arrive with the following parameters:

| Vessel ($i$) | Vessel Name | LOA + Margin ($L_i$) | ETA ($a_i$) | Due Date ($d_i$) | Moves ($M_i$) | Cranes Assigned ($q_i$) | Net Rate ($\mu_v$) | Service Time ($p_i$) |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **V1** | *Pacific Titan* | $400 \text{ m}$ | $0.0 \text{ h}$ | $36.0 \text{ h}$ | 3,600 | 4 | 120 moves/h | $30.0 \text{ h}$ |
| **V2** | *Atlantic Star* | $350 \text{ m}$ | $4.0 \text{ h}$ | $44.0 \text{ h}$ | 3,000 | 4 | 120 moves/h | $25.0 \text{ h}$ |
| **V3** | *Orient Express* | $300 \text{ m}$ | $8.0 \text{ h}$ | $40.0 \text{ h}$ | 2,400 | 3 | 90 moves/h | $26.7 \text{ h}$ |
| **V4** | *Coastal Courier*| $200 \text{ m}$ | $12.0 \text{ h}$| $36.0 \text{ h}$ | 1,200 | 2 | 60 moves/h | $20.0 \text{ h}$ |

**Quay Constraints:**
- Total quay length $L_{\text{quay}} = 1,600 \text{ m}$.
- Sum of all vessel lengths $= 400 + 350 + 300 + 200 = 1,250 \text{ m} \le 1,600 \text{ m}$.
- Because the sum of lengths is strictly less than $L_{\text{quay}}$, all four vessels can physically berth simultaneously along the quay wall without spatial conflict, provided they can be assigned non-overlapping coordinates.

**Spatial and Temporal Berthing Plan:**
Let us assign berthing positions $m_i$ along the quay:
- Vessel 1 (*Pacific Titan*): Assign to $m_1 = 0 \text{ m}$. Occupies $[0, 400 \text{ m}]$.
- Vessel 2 (*Atlantic Star*): Assign to $m_2 = 400 \text{ m}$. Occupies $[400, 750 \text{ m}]$.
- Vessel 3 (*Orient Express*): Assign to $m_3 = 750 \text{ m}$. Occupies $[750, 1,050 \text{ m}]$.
- Vessel 4 (*Coastal Courier*): Assign to $m_4 = 1,050 \text{ m}$. Occupies $[1,050, 1,250 \text{ m}]$.

**Temporal Schedule:**
Since space is available, each vessel berths immediately upon arrival ($b_i = a_i$):
- V1: Berths at $b_1 = 0.0 \text{ h}$, departs at $c_1 = 0.0 + 30.0 = 30.0 \text{ h}$. Tardiness: $\max(0, 30.0 - 36.0) = 0 \text{ h}$.
- V2: Berths at $b_2 = 4.0 \text{ h}$, departs at $c_2 = 4.0 + 25.0 = 29.0 \text{ h}$. Tardiness: $\max(0, 29.0 - 44.0) = 0 \text{ h}$.
- V3: Berths at $b_3 = 8.0 \text{ h}$, departs at $c_3 = 8.0 + 26.7 = 34.7 \text{ h}$. Tardiness: $\max(0, 34.7 - 40.0) = 0 \text{ h}$.
- V4: Berths at $b_4 = 12.0 \text{ h}$, departs at $c_4 = 12.0 + 20.0 = 32.0 \text{ h}$. Tardiness: $\max(0, 32.0 - 36.0) = 0 \text{ h}$.

**Quay Crane Rail Verification:**
Total cranes required across all vessels between $t = 12.0$ and $t = 25.0$:
$$\sum q_i = 4 + 4 + 3 + 2 = 13 \text{ cranes} \le 16 \text{ total available cranes}$$
The schedule is fully feasible in space, time, and crane capacity. Total waiting time is $0.0 \text{ hours}$.

---

### Step 3: Yard Reshuffling and Stacking Crane Degradation

Consider an import block stacking containers 4 tiers high ($H = 4$). Base crane cycle time under ideal conditions (no reshuffling) is $t_{\text{base}} = 75 \text{ seconds}$ per move ($CMPH = 3,600 / 75 = 48 \text{ moves/hour}$).

1. **Theoretical Expected Reshuffles:**
   $$\mathbb{E}[R] = \frac{H - 1}{2} = \frac{4 - 1}{2} = 1.50 \text{ reshuffles per container retrieval}$$

2. **Scenario A: Normal Yard Occupancy ($U = 0.70$ or 70%):**
   At 70% occupancy, adjacent slots are available. Each reshuffle takes an average of 45 seconds (short cycle):
   $$T_{\text{retrieve}} = t_{\text{base}} + (\mathbb{E}[R] \times t_{\text{reshuffle}}) = 75 + (1.50 \times 45) = 75 + 67.5 = 142.5 \text{ seconds}$$
   $$\text{Net Retrieval Moves per Hour} = \frac{3,600}{142.5} = 25.26 \text{ target containers/hour}$$

3. **Scenario B: Congested Yard Occupancy ($U = 0.85$ or 85%):**
   At 85% occupancy, empty slots in the active bay are scarce. Using the non-linear occupancy penalty formula with $\alpha = 0.20$:
   $$\text{Penalty Factor } \Phi(U) = 1 + \frac{\alpha U^2}{1 - U} = 1 + \frac{0.20 \times (0.85)^2}{1 - 0.85} = 1 + \frac{0.20 \times 0.7225}{0.15} = 1 + \frac{0.1445}{0.15} = 1 + 0.9633 = 1.9633$$
   $$\mathbb{E}[R_{\text{eff}}] = 1.50 \times 1.9633 = 2.945 \text{ reshuffles per retrieval}$$

   Because reshuffled containers must be moved to distant bays, the average reshuffle cycle time increases to 85 seconds:
   $$T_{\text{retrieve, congested}} = 75 + (2.945 \times 85) = 75 + 250.3 = 325.3 \text{ seconds}$$
   $$\text{Net Retrieval Moves per Hour} = \frac{3,600}{325.3} = 11.07 \text{ target containers/hour}$$

4. **Productivity Collapse Ratio:**
   $$\frac{\text{Net Capacity at 85\%}}{\text{Net Capacity at 70\%}} = \frac{11.07}{25.26} = 0.438 \quad (56.2\% \text{ loss of retrieval capacity})$$
   A 15 percentage point increase in yard occupancy cuts net yard crane output by more than half, explaining why terminals freeze when yard inventory passes 80%.

---

### Step 4: Gate Sizing and Truck Appointment System (TAS) Quotas

The terminal operates $c_{\text{gate}} = 12$ ingate lanes.
- Average processing time per truck at the pedestal: $\mathbb{E}[S_{\text{gate}}] = 2.0 \text{ minutes} = 1/30 \text{ hours}$ ($\mu_{\text{gate}} = 30 \text{ trucks/lane/hour}$).
- Total theoretical gate service capacity: $12 \times 30 = 360 \text{ trucks/hour}$.

1. **Unrestricted Peak Demand Surge (No TAS):**
   Without appointments, 320 trucks arrive between 08:00 and 09:00 ($\lambda_{\text{gate}} = 320 \text{ trucks/hour}$).
   $$\rho_{\text{gate}} = \frac{\lambda_{\text{gate}}}{c_{\text{gate}} \cdot \mu_{\text{gate}}} = \frac{320}{360} = 0.8889 \quad (88.89\%)$$

   Processing times have variance due to trouble tickets ($\sigma = 1.2 \text{ min}, C_{s,\text{gate}} = 0.60, C_s^2 = 0.36$).
   Using the Erlang-C multi-server formula for $s=12, a = 320 / 30 = 10.667$:
   $$P_w \approx 0.442$$
   $$W_{q,\text{gate}}(M/M/12) = \frac{P_w}{c\mu - \lambda} = \frac{0.442}{360 - 320} = \frac{0.442}{40} = 0.01105 \text{ hours} = 0.663 \text{ minutes}$$
   $$L_q = 320 \times 0.01105 = 3.54 \text{ trucks}$$

   Now consider a 15% volume surge or 2 lane closures due to OCR failures ($c = 10, \text{Capacity} = 300 \text{ trucks/hour}$):
   $$\rho = \frac{320}{300} = 1.067 \quad (\rho > 1.0)$$
   The queue becomes unstable. Over 60 minutes, $320 - 300 = 20$ unserviced trucks accumulate each hour. Within two hours, 40 trucks with 75-foot tractor-trailers back up 3,000 feet onto the public freeway, triggering municipal police intervention.

2. **TAS Smoothing Optimization:**
   The terminal introduces a mandatory TAS capping arrivals at $\rho_{\text{target}} \le 0.75$:
   $$Q_t = \lfloor c_{\text{gate}} \cdot \mu_{\text{gate}} \cdot \rho_{\text{target}} \rfloor = \lfloor 12 \times 30 \times 0.75 \rfloor = 270 \text{ appointments/hour}$$
   - Import pickup quota: 150 slots/hour (matched to yard crane block retrieval capacity).
   - Export drop-off quota: 80 slots/hour.
   - Dual-transaction priority slots: 40 slots/hour.
   
   At $\lambda = 270 \text{ trucks/hour}$ across 12 lanes ($\rho = 0.75$):
   $$P_w \approx 0.148 \implies W_q = \frac{0.148}{360 - 270} = \frac{0.148}{90} = 0.00164 \text{ hours} = 5.9 \text{ seconds}$$
   Gate turn times drop from 45+ minutes to under 12 minutes door-to-door. Highway queues are completely eliminated.

---

## 7. Code, query, or calculation

The following self-contained Python script computes Pollaczek-Khinchine ($M/G/1$) anchorage waiting metrics, Erlang-C ($M/M/s$) multi-berth performance, and yard crane cycle degradation under variable occupancy.

```python
"""Terminal Queuing and Yard Degradation Computational Engine."""
import math

def pollaczek_khinchine(lam: float, mean_s: float, std_s: float) -> dict:
    """Computes M/G/1 queue metrics for vessel anchorage or single-server channel."""
    rho = lam * mean_s
    if rho >= 1.0:
        raise ValueError(f"Unstable system: rho={rho:.3f} >= 1.0")
    cs_sq = (std_s / mean_s) ** 2
    e_s2 = (std_s ** 2) + (mean_s ** 2)
    wq = (lam * e_s2) / (2.0 * (1.0 - rho))
    return {"rho": rho, "Cs_sq": cs_sq, "Wq_hours": wq, "Lq_vessels": lam * wq}

def erlang_c(s: int, lam: float, mean_s: float) -> dict:
    """Computes M/M/s multi-berth queue metrics via analytical Erlang-C formula."""
    a = lam * mean_s
    rho = a / s
    if rho >= 1.0:
        raise ValueError(f"Unstable system: rho={rho:.3f} >= 1.0")
    sum_terms = sum((a**k) / math.factorial(k) for k in range(s))
    saturated = (a**s) / (math.factorial(s) * (1.0 - rho))
    p0 = 1.0 / (sum_terms + saturated)
    pw = saturated * p0
    mu = 1.0 / mean_s
    wq = pw / (s * mu - lam)
    return {"rho": rho, "P_wait": pw, "Wq_hours": wq, "Lq_vessels": lam * wq}

def yard_cycle_degradation(h: int, util: float, base_sec: float = 75.0) -> dict:
    """Calculates expected reshuffles and net retrieval moves under yard occupancy."""
    if not (0.0 <= util < 1.0):
        raise ValueError("Yard utilization must be in range [0, 1).")
    e_r = (h - 1) / 2.0
    penalty = 1.0 + (0.20 * (util**2)) / max(0.01, 1.0 - util)
    eff_reshuffles = e_r * penalty
    reshuffle_time = 45.0 if util <= 0.75 else 85.0
    net_cycle = base_sec + (eff_reshuffles * reshuffle_time)
    net_mph = 3600.0 / net_cycle
    return {"eff_reshuffles": eff_reshuffles, "net_cycle_sec": net_cycle, "net_mph": net_mph}

if __name__ == "__main__":
    # 1. Anchorage Queuing Evaluation
    ec = erlang_c(s=4, lam=0.080, mean_s=40.0)
    print(f"Erlang-C (s=4, rho={ec['rho']:.2f}): P(wait)={ec['P_wait']:.4f}, Wq={ec['Wq_hours']:.2f}h")

    # 2. P-K Formula with Service Variability
    pk = pollaczek_khinchine(lam=0.020, mean_s=40.0, std_s=24.0)
    print(f"P-K M/G/1 (Cs^2={pk['Cs_sq']:.2f}): Wq={pk['Wq_hours']:.2f}h, Lq={pk['Lq_vessels']:.2f}")

    # 3. Yard Degradation at 70% vs 85% Occupancy
    yd_70 = yard_cycle_degradation(h=4, util=0.70)
    yd_85 = yard_cycle_degradation(h=4, util=0.85)
    print(f"Yard 70% Util -> Net Cycle: {yd_70['net_cycle_sec']:.1f}s, Output: {yd_70['net_mph']:.1f} moves/h")
    print(f"Yard 85% Util -> Net Cycle: {yd_85['net_cycle_sec']:.1f}s, Output: {yd_85['net_mph']:.1f} moves/h")
```

### Assumptions and Expected Output

**Assumptions:**
- Stationary Poisson arrival process for Erlang-C and Pollaczek-Khinchine models.
- Independent, identical service channels with no simultaneous multi-server breakdown.
- Yard reshuffle penalty calibrated with empirical block coefficient $\alpha = 0.20$.

**Expected Console Output:**
```text
Erlang-C (s=4, rho=0.80): P(wait)=0.5964, Wq=29.82h
P-K M/G/1 (Cs^2=0.36): Wq=108.80h, Lq=2.18
Yard 70% Util -> Net Cycle: 164.6s, Output: 21.9 moves/h
Yard 85% Util -> Net Cycle: 325.3s, Output: 11.1 moves/h
```

---

## 8. Data, standards, and regulatory context

Container terminal operations function under strict statutory, safety, and international technical standards. Non-compliance immediately triggers regulatory holds, port state detentions, or operational bans.

```text
=============================================================================================================
                          GOVERNANCE AND STANDARDS HIERARCHY
=============================================================================================================
  Maritime Safety & Weight ──► IMO SOLAS VI Reg 2 (Verified Gross Mass - VGM)
  Commercial / Billing     ──► US FMC OSRA 2022 (Demurrage & Detention Reform; 46 CFR Part 541)
  Digital Standards        ──► DCSA (Just-In-Time Port Call Standards / Track & Trace 2.2)
  Security & Identification──► ISO 6346 (Container Markings) & MTSA / ISPS Code (TWIC & Biometrics)
  Decarbonization / Power  ──► CARB At-Berth Regulation (Cold Ironing) & EU FuelEU Maritime
=============================================================================================================
```

### 1. IMO SOLAS Chapter VI, Regulation 2: Verified Gross Mass (VGM)
Enacted July 1, 2016, by the International Maritime Organization (IMO) under the International Convention for the Safety of Life at Sea (SOLAS):
- **Statutory Mandate:** A packed container cannot be loaded onto a cellular container vessel unless its Verified Gross Mass (VGM) has been obtained by the shipper and communicated to the terminal representative and ship master sufficiently in advance of the vessel stowage plan creation.
- **Permitted Methods:**
  - *Method 1:* Weighing the fully packed and sealed container using calibrated, certified scales.
  - *Method 2:* Weighing all individual cargo packages, pallets, dunnage, and securing material, and adding the tare mass of the container printed on the container door.
- **Terminal Enforcement:** Modern marine terminals operate Weigh-In-Motion (WIM) scales embedded in the ingate lanes. If an ingate container weight deviates by more than $\pm 500 \text{ kg}$ (or $\pm 5\%$) from the declared electronic VGM document, the system generates an automated "VGM Discrepancy Hold," locking the container out of the vessel load list.

### 2. US Federal Maritime Commission (FMC) and OSRA 2022
The Ocean Shipping Reform Act of 2022 (Public Law 117-146) radically altered demurrage and detention enforcement across United States container ports:
- **Demurrage and Detention Billing Requirements (46 CFR Part 541):** Common carriers and terminal operators are prohibited from assessing demurrage or detention charges unless the invoice includes 13 mandatory statutory data elements (including container availability date, free time start/end dates, applicable tariff rates, and specific identification of why charges were incurred). Invoices missing any required element forfeit the right to collection.
- **Incentive Principle:** Demurrage and detention fees are legally classified as financial incentives to encourage cargo fluidity. When an ocean carrier or terminal operator fails to provide physical access to a container (e.g., container buried in a closed yard block, lack of appointment slots in the TAS, or terminal gate closures), assessing demurrage constitutes an "unreasonable practice" under 46 U.S.C. § 41102(c).
- **Empty Return Mandates:** Regulators penalize terminals that issue blanket bans on empty container returns while simultaneously charging detention on the unreturnable equipment.

### 3. Digital Container Shipping Association (DCSA) Port Call Standards
The DCSA (an association of the top global ocean carriers including MSC, Maersk, CMA CGM, and Hapag-Lloyd) establishes digital interface standards for terminal-carrier coordination:
- **Just-In-Time (JIT) Port Arrival Standards:** Replaces the historical "first-come, first-served" oceanic race to anchorage with digital handshakes. The terminal transmits the confirmed *Requested Time of Berthing* (RToB) 24 to 48 hours prior to arrival, allowing the vessel to practice "virtual arrival" (slow steaming to reduce bunker consumption and emissions while maintaining berth alignment).
- **Standardized Port Call Timestamps:** Establishes unambiguous definitions for operational milestones: *Pilot Boarding Place Arrival*, *All Fast at Berth*, *Commenced Discharge*, *Completed Cargo Operations*, and *Unmoored*.

### 4. Environmental and Decarbonization Mandates
- **California Air Resources Board (CARB) At-Berth Regulation:** Requires container vessels berthing at California ports (Los Angeles, Long Beach, Oakland) to shut down auxiliary diesel generators and connect to the municipal electric grid ("cold ironing" or shore power) or deploy CARB-approved barge-mounted emissions capture systems. Terminals face fines exceeding $10,000 per violation day for unexcused shore-power connection delays.
- **EU FuelEU Maritime and ETS Expansion:** As of 2024, maritime emissions inside European Union ports are integrated into the EU Emissions Trading System (ETS). Vessels incur carbon allowance costs for every metric ton of $CO_2$ emitted while waiting at European port anchorages.

---

## 9. Industry practice and case evidence

Real-world port operations operate at the intersection of mathematical theory and brutal physical constraints. The following empirical cases illustrate subsystem failure, automation trade-offs, and operational breakthroughs.

```text
=============================================================================================================
                                     SUMMARY OF CASE EVIDENCE
=============================================================================================================
  Port of Los Angeles / Long Beach (2021-2022) ──► Cascading Subsystem Collapse & Chassis Gridlock
  Port of Rotterdam Maasvlakte II              ──► Full Automation & Landside/Waterside ASC Decoupling
  PSA Singapore Tuas Mega-Port                 ──► High-Density Mathematical Optimization & Single Window
=============================================================================================================
```

### Case 1: The San Pedro Bay Port Congestion Crisis (Los Angeles and Long Beach, 2021–2022)

The twin ports of Los Angeles and Long Beach handle approximately 35 percent of all US containerized waterborne imports. Between October 2021 and February 2022, the ports suffered the most severe congestion crisis in modern maritime history.

*Empirical Facts:*
- **Anchorage Queue Explosion:** Over 109 container ships dropped anchor or drifted off the coast of Southern California.
- **Yard Saturation:** Terminal yard occupancy exceeded 88 percent across all major terminals (APM Terminals Pier 400, Long Beach Container Terminal, Fenix Marine Services).
- **Dwell Time Inflation:** Average import container dwell time in the yard jumped from an historical average of 3.2 days to over 9.5 days. Street dwell for container chassis exceeded 10 days.

*The Cascading Failure Vector:*
1. Inland distribution centers in the Inland Empire (Riverside/San Bernardino) reached 100% capacity due to consumer goods inventory accumulation.
2. Warehouses could not unload incoming drayage trailers. Truck chassis remained immobilized under loaded containers in warehouse parking lots, creating a regional chassis deficit.
3. Without chassis, draymen could not pull import containers out of the marine terminal yard.
4. Terminal yards exceeded 85% occupancy. RTG cranes spent two-thirds of their work shifts reshuffling containers.
5. Inbound vessels could not discharge; STS crane moves dropped from 32 CMPH to below 18 CMPH.
6. The Marine Exchange of Southern California ran out of designated anchorage space, forcing incoming mega-vessels to drift 150 miles offshore in open ocean waters.

*The Resolution Protocol:*
The congestion was broken not by adding berths, but through systemic operational intervention:
- **Emergency Safety and Air Quality Protocol:** The Marine Exchange introduced a dynamic queuing reservation system where ships were assigned queue positions based on their departure time from Asia, rather than arrival at the harbor pilot station, allowing ships to slow steam across the Pacific.
- **The Threat of the "Container Dwell Fee":** The port authorities announced an escalating daily fee ($100 per container on Day 9, increasing by $100 per day thereafter) assessed directly against ocean carriers for lingering containers. Although never formally collected, the threat forced ocean carriers to charter dedicated "sweeper ships" to evacuate over 150,000 empty containers that were blocking yard capacity.

### Case 2: Port of Rotterdam Maasvlakte II (Automation and Decoupled Stacking)

At the outer deepwater port of Rotterdam, terminal facilities such as APM Terminals MedPort and Rotterdam World Gateway (RWG) were engineered specifically to eliminate subsystem interference through complete physical decoupling.

*Operational Architecture:*
- **Zero Personnel on Quay Apron:** Fully automated Lift-AGVs (battery-powered) transit between five remote-controlled Super Post-Panamax STS cranes and automated stacking blocks.
- **Perpendicular ASC Block Layout:** Stacking blocks run perpendicular to the quay wall rather than parallel. The waterside end of each block is serviced exclusively by automated AGVs, while the landside end is serviced exclusively by external drayage trucks and rail shuttles.
- **Double-Ended Crane Automation:** Two separate ASCs operate per block, but physical separation rails and dedicated handover bays prevent crane collision while eliminating interference between waterside vessel discharge and landside truck pickup.

*Observed Productivity:*
Net vessel operating rates regularly surpass 150 berth moves per hour across four cranes with continuous 24/7 reliability, proving that decoupling landside and waterside operations eliminates yard crane cross-interference.

### Case 3: PSA Singapore and the Tuas Mega-Port Development

PSA Singapore handles over 37 million TEUs annually, with transshipment comprising more than 85 percent of total volume. Transshipment terminals are fundamentally different from gateway terminals: containers do not exit through a truck gate; they transfer from one arriving vessel to another connecting vessel.

*System Innovations:*
- **High-Density Stacking (1-over-6):** Stacking up to 6 tiers high requires massive predictive intelligence. PSA uses deep reinforcement learning algorithms to match stowage plans with incoming connection windows, minimizing reshuffling moves to less than 0.3 per container transfer.
- **PORTNET Community Single Window:** A unified EDI integration linking customs, terminal operators, linehaul carriers, feeder lines, and bunker suppliers. Berth allocations are dynamically adjusted in 15-minute increments based on real-time AIS vessel tracking and terminal crane gangs.

---

## 10. Failure modes and red flags

The table below catalogs operational failure modes across terminal subsystems, detailing observable early warning signs, underlying mechanisms, systemic consequences, and validated countermeasures.

| # | Failure Mode | Early Warning Sign | Operational Mechanism | Systemic Consequence | Countermeasure / Mitigation |
| :-: | :--- | :--- | :--- | :--- | :--- |
| **1** | **Yard Saturation Gridlock** | Yard occupancy exceeds 75%; gross crane moves/hour drop by >20%. | Available empty ground slots disappear; crane re-handling moves explode nonlinearly. | Crane cycle times triple; horizontal transport queues back up onto quay; vessel discharge halts. | Implement empty container sweeper vessels; shorten import free time; impose punitive progressive dwell surcharges. |
| **2** | **Quay Crane Interference (Clash)** | Cranes assigned to adjacent bays operate below 15 moves/hour. | Minimum boom-to-boom safety distance ($\approx 35 \text{ m}$) violated; cranes wait for adjacent booms to clear. | Loss of 30–50% of expected crane gang output; vessel service window slips past proforma schedule. | Enforce dynamic crane assignment algorithms with hard physical distance separation constraints. |
| **3** | **Gate Highway Queue Spillover** | Drayage truck line extends past terminal staging gate onto public highway. | Ingate processing rate falls below peak arrival rate ($\rho_{\text{gate}} > 1.0$); OCR failure rate spikes. | Municipal traffic citations; driver detention claims; port authority gate closure orders. | Deploy mandatory Truck Appointment System (TAS) with hourly quota caps; open bypass lanes for bobtails. |
| **4** | **Chassis Dislocation Deficit** | Drayage drivers arrive "bobtail" (without chassis); terminal chassis pools empty. | Export/empty containers mounted on chassis are stranded at customer warehouses; terminal bans empty returns. | Inability to dray import containers off the terminal; import yard dwell increases rapidly. | Mandate "dual-transaction" gate appointments; establish off-dock "peel-off" container staging yards on neutral chassis. |
| **5** | **VGM Discrepancy Lock** | High volume of containers diverted to terminal trouble parking area. | Ingate weigh-in-motion scale detects weight $> 5\%$ or $> 500 \text{ kg}$ variance from shipper EDI VGM declaration. | Container excluded from vessel load stowage plan; terminal yard space consumed by dead inventory. | Automated electronic notification to shipper; on-terminal certified re-weighing service with administrative fee. |
| **6** | **Twin-ASC Rail Clash** | Landside truck turn times exceed 90 minutes while quay cranes wait for containers. | Twin ASCs on single block rail fight for access to middle bays; one crane forced to idle in parking spur. | Symmetrical capacity loss across both waterside and landside interfaces. | Establish centralized virtual exchange zones ("handshake bays") in the middle of the block; prioritize waterside moves. |
| **7** | **On-Dock Rail Stranding** | Rail container dwell exceeds 6 days; intermodal tracks blocked with mixed cuts. | Class I railroad fails to supply double-stack well cars; switch engine power unavailable. | Rail tracks saturate; import containers bound for inland rail cannot be discharged from ships. | Contractual car-supply service level agreements (SLAs) with Class I carriers; dray rail containers to off-dock ICTF. |
| **8** | **Reefer Plug Exhaustion** | Inbound reefer vessels held at anchor; mobile generator sets deployed. | Import reefer containers dwell past 3 days; refrigerated container racks reach 98% electrical plug capacity. | High risk of high-value perishable cargo spoilage; catastrophic cargo liability claims under COGSA. | Dynamic reefer surcharge tariff; prioritized outbound drayage appointment slots; portable genset buffer banks. |
| **9** | **Tidal Draft / Air Draft Lock** | Arriving ultra-large container ship anchors outside harbor channel for 8+ hours. | Vessel arrival delayed past high tide window; vessel draft exceeds channel depth or air draft under bridges. | Unplanned anchorage waiting time; missed pilotage windows; domino delays for follow-on port calls. | Dynamic under-keel clearance (DUKC) systems using real-time bathymetry, swell, and tide sensors. |
| **10** | **Hazardous IMDG Segregation Conflict** | Yard planner unable to find valid slot for incoming hazardous container. | Incompatible hazardous classes (e.g., IMDG Class 5.1 oxidizers near Class 3 flammables) assigned adjacent bays. | Discharging crane stops work on entire ship bay; regulatory inspection fine by USCG / Port State Control. | Automated IMDG segregation rules engine integrated directly into terminal operating system (TOS) stowage module. |
| **11** | **Labor Gang Starvation** | Quay cranes stand idle with spreader elevated during operational shift. | Terminal fails to order sufficient longshore gangs from union hiring hall 24 hours in advance. | Terminal pays dead-time wages; ship incurs thousands of dollars per hour in charter delay costs. | Predictive gang ordering based on high-accuracy AIS arrival telemetry and crane productivity forecasting. |
| **12** | **Demurrage Free-Time Cliff Surge** | Gate volume spikes by 300% on Friday afternoon. | Beneficial cargo owners rush to pull containers on final day of free time before weekend demurrage accrues. | Ingate congestion collapse; yard crane capacity overwhelmed; trucks turned away without containers. | Stagger free-time expiration by vessel discharge date/hour rather than calendar days; align free time with TAS slots. |

---

## 11. Assumptions and limits

The analytical frameworks and optimization formulations presented in this topic rest upon foundational engineering assumptions. Operating outside these boundaries invalidates the models.

| # | Assumption | Operational Rationale | Boundary Condition / What Breaks It | Validation and Testing Protocol |
| :-: | :--- | :--- | :--- | :--- |
| **1** | **Poisson Vessel Arrivals ($M/M/s$, $M/G/1$)** | Simplifies stochastic arrival modeling in open sea lanes. | Breaks during severe coastal storms, canal closures, or proforma schedule compression where ships arrive in dense bunches. | Run Anderson-Darling or Chi-Square goodness-of-fit test on empirical inter-arrival times; use $G/G/s$ if variance is high. |
| **2** | **Continuous Linear Quay Wall** | Assumes the entire berth length is uniform, unobstructed, and has identical water depth. | Physical bollard spacing, shallow water patches, crane rail breaks, and protruding pier infrastructure create discrete barriers. | Map physical berth constraints into the BAP model as forbidden spatial coordinate exclusions. |
| **3** | **Linear Crane Productivity Scaling** | Assumes adding a 4th or 5th crane increases gross vessel moves proportionally. | Physical boom clearance constraints, ship bay geometry, and crane interference cause diminishing marginal returns above 4 cranes. | Apply sub-linear productivity scaling factor: $CMPH(q) = q \cdot CMPH_{\text{base}} \cdot (1 - \beta(q - 1))$. |
| **4** | **Uniform Random Yard Retrieval** | Assumes any container in a vertical stack has equal probability of being picked by a truck. | Shippers prioritize specific containers based on manufacturing schedules; "peel-off" piles retrieve strictly top-to-bottom. | Analyze historical retrieval distributions; apply non-uniform rank-order statistics if retrievals correlate with import dwell. |
| **5** | **Steady-State Queuing Equilibrium ($\rho < 1$)** | Required for analytical P-K and Erlang-C formulas to converge to finite waiting times. | Peak season surges or labor strikes frequently push arrival rates past service capacity ($\rho \ge 1.0$) for days at a time. | Use discrete-event dynamic simulation (SimPy / AnyLogic) to capture transient queue accumulation during overloaded intervals. |
| **6** | **Independent Gate Lane Operations** | Assumes each gate lane operates as an independent parallel server with identical processing rates. | A single truck with complex documentation or oversized cargo can block a lane; shared trouble kiosks create cross-lane correlation. | Model gate complex as a network of parallel queues with probabilistic balking and reneging to exception lanes. |
| **7** | **Deterministic Crane Cycle Times** | Simplifies horizontal transport fleet sizing calculations. | Mechanical spreader jams, twistlock sticking, wind gusts, and human operator fatigue induce significant variance. | Measure cycle time distributions via automated TOS telemetry; incorporate stochastic buffers into ITV dispatch rules. |
| **8** | **Full Trucker Compliance with TAS** | Assumes drayage drivers arrive strictly within their reserved 1-hour appointment window. | Traffic congestion on public freeways causes 15–25% of trucks to arrive early, late, or miss appointments entirely ("no-shows"). | Enforce grace periods (e.g., $\pm 45 \text{ min}$); track carrier compliance scores; over-allocate appointment quotas by no-show rate. |
| **9** | **Static Container Dwell Behavior** | Assumes container dwell time distribution is invariant to terminal congestion. | When congestion builds, consignees intentionally delay pickups because warehouse space is exhausted (feedback dwell inflation). | Model dwell as an endogenous function of regional warehouse vacancy rates and terminal demurrage tariff pricing. |
| **10** | **Homogeneous Container Sizes** | Assumes all containers are standard 40-foot units (2 TEUs) with identical handling profiles. | Stacking 20-foot containers into 40-foot slots (or vice versa) creates slot stranding; out-of-gauge (OOG) cargo requires specialized gear. | Segregate yard blocks into dedicated 20-foot, 40-foot, and high-cube/OOG zones within the terminal operating system. |

---

## 12. Exercises

The following exercises reinforce mathematical, algorithmic, and operational competencies across container terminal design, queuing analysis, and congestion mitigation.

### Exercise 1: Erlang-C Berth Sizing and Congestion Surcharge Analysis
*Focus:* Multi-server queuing calculations and capital expenditure evaluation.  
*Time:* 35 minutes.  
*Deliverable:* Complete step-by-step mathematical calculation of berth waiting times and financial risk assessment.

A common-user container terminal currently operates $s = 3$ berths. Average vessel inter-arrival time is 16.0 hours ($\lambda = 0.0625 \text{ vessels/hour}$). Average vessel service time at berth is $\mathbb{E}[S] = 36.0 \text{ hours}$ ($\mu = 0.02778 \text{ vessels/hour}$).
1. Calculate the current berth utilization $\rho$, the probability of waiting $P_w$, and the expected waiting time at anchorage $W_q$.
2. Ocean carriers announce that if average anchorage waiting time exceeds 18.0 hours, they will impose an emergency Port Congestion Surcharge (PCS) of $250 \text{ per TEU}$. Given an annual terminal throughput of 1,200,000 TEUs, calculate the potential annual surcharge cost to shippers.
3. The port authority can construct a 4th berth ($s = 4$) at an annualized capital financing cost of $18,000,000$. Re-calculate $P_w$ and $W_q$ for $s = 4$. Based strictly on avoiding the congestion surcharge, does the investment make financial sense?

*Acceptance Criteria:* Accurate calculation of $P_0, P_w, W_q$ to two decimal places; clear financial trade-off comparison.

---

### Exercise 2: Pollaczek-Khinchine Variability Mitigation
*Focus:* $M/G/1$ queue modeling, squared coefficient of variation, and scheduling discipline.  
*Time:* 30 minutes.  
*Deliverable:* Quantitative comparison table showing waiting time and queue length across three operational variance regimes.

A specialized container feeder channel operates as an $M/G/1$ queue with arrival rate $\lambda = 0.040 \text{ vessels/hour}$ and mean service time $\mathbb{E}[S] = 20.0 \text{ hours}$.
1. Compute the traffic intensity $\rho$.
2. Compute the expected waiting time $W_q$ and anchorage queue length $L_q$ under three scenarios:
   - Scenario A: High operational chaos ($\sigma_s = 25.0 \text{ hours}$).
   - Scenario B: Standard exponential service ($\sigma_s = 20.0 \text{ hours}$).
   - Scenario C: Advanced digital scheduling and standardized crane gangs ($\sigma_s = 8.0 \text{ hours}$).
3. Explain the operational mechanism through which variance reduction cuts waiting time without increasing the number of physical berths or speed of cranes.

*Acceptance Criteria:* Correct calculation of $C_s^2$, $W_q$, and $L_q$ for all three scenarios; concise, rigorous physical explanation.

---

### Exercise 3: Continuous Berth Allocation Problem (BAP) Conflict Resolution
*Focus:* Mixed-Integer Linear Programming constraints and 2D space-time packing.  
*Time:* 40 minutes.  
*Deliverable:* Space-time coordinate diagram, mathematical non-overlap verification, and total delay calculation.

A container terminal has a continuous quay wall of $L_{\text{quay}} = 900 \text{ meters}$. Two vessels are scheduled:
- Vessel A: $LOA_A + \Delta_L = 400 \text{ m}$, ETA $a_A = 0 \text{ h}$, Service duration $p_A = 24 \text{ h}$.
- Vessel B: $LOA_B + \Delta_L = 450 \text{ m}$, ETA $a_B = 6 \text{ h}$, Service duration $p_B = 20 \text{ h}$.
- Vessel C: $LOA_C + \Delta_L = 350 \text{ m}$, ETA $a_C = 8 \text{ h}$, Service duration $p_C = 16 \text{ h}$.

1. Can all three vessels berth simultaneously at $t = 8 \text{ h}$? Show the mathematical inequality.
2. If Vessel A berths at coordinate $m_A = 0 \text{ m}$ at $b_A = 0 \text{ h}$, and Vessel B berths at coordinate $m_B = 400 \text{ m}$ at $b_B = 6 \text{ h}$, determine the earliest feasible berthing time $b_C$ for Vessel C.
3. State which Big-M non-overlapping constraint ($z_{ij}^x$ or $z_{ij}^y$) separates Vessel C from Vessel A and Vessel B.

*Acceptance Criteria:* Precise spatial and temporal coordinates; correct application of MILP non-overlapping logic.

---

### Exercise 4: Yard Reshuffling and Block Stacking Height Optimization
*Focus:* Stacking physics, re-handling ratios, and crane cycle time degradation.  
*Time:* 30 minutes.  
*Deliverable:* Tabular calculation comparing 3-high, 4-high, and 5-high stacking configurations under high utilization.

An import container block handles 1,200 container retrievals per week. The terminal is considering whether to operate RTG cranes at 3-high, 4-high, or 5-high stacking ($H \in \{3, 4, 5\}$).
1. Calculate the theoretical expected reshuffles per retrieval $\mathbb{E}[R]$ for $H = 3, 4, 5$ assuming random uniform retrievals.
2. If the terminal operates at an elevated yard utilization of $U = 0.80$, compute the effective reshuffles $\mathbb{E}[R_{\text{eff}}]$ using the non-linear penalty factor with $\alpha = 0.20$.
3. Given a base retrieval cycle of 70 seconds and an average reshuffle cycle of 60 seconds, compute the total annual crane operating hours required to service the 1,200 weekly retrievals for each stacking height.

*Acceptance Criteria:* Correct arithmetic across all three heights; clear demonstration of how vertical density increases operating expense.

---

### Exercise 5: Truck Appointment System (TAS) Quota Sizing
*Focus:* Gate queuing design, multi-server performance, and arrival smoothing.  
*Time:* 35 minutes.  
*Deliverable:* Hourly appointment schedule and gate utilization analysis.

A terminal has 10 active ingate lanes ($c_{\text{gate}} = 10$). Mean transaction time per truck at the pedestal is 2.5 minutes ($\mu_{\text{gate}} = 24 \text{ trucks/lane/hour}$).
1. Calculate the maximum theoretical throughput capacity of the gate complex per hour.
2. To guarantee that average gate queue wait time remains below 2 minutes, terminal policy mandates that gate utilization $\rho_{\text{gate}}$ must not exceed 0.80. Calculate the maximum allowable hourly appointment quota $Q_t$.
3. Prior to TAS, peak uncoordinated morning demand was 280 trucks/hour. What was the gate utilization $\rho$ during that peak? Did the queue stabilize?
4. How should the terminal structure appointments to incentivize dual transactions (matchbacks)? Provide a mathematical expression showing how dual transactions expand effective gate capacity.

*Acceptance Criteria:* Quantitative gate sizing; proper Erlang utilization limits; rigorous formulation of dual-transaction efficiency.

---

### Exercise 6: Terminal Bottleneck Shift Critique
*Focus:* Systems thinking, theory of constraints, and cross-subsystem dynamics.  
*Time:* 30 minutes.  
*Deliverable:* A 300-word written technical critique analyzing an ill-conceived terminal investment proposal.

A terminal director proposes spending $60,000,000 to purchase 4 new Super Post-Panamax STS cranes with hoisting speeds of 40 moves/hour to eliminate a 3-day vessel anchorage backlog. The terminal’s import yard is currently operating at 86% occupancy, and local drayage trucks experience an average gate turn time of 95 minutes.
Critique this proposal using the Theory of Constraints and the Four-Stage Coupled Pipeline model:
1. Explain why adding faster quay cranes will fail to reduce anchorage waiting times.
2. Identify where the true system constraint lies.
3. Predict what will physically happen to the quay apron if the new cranes are installed without addressing the yard.

*Acceptance Criteria:* Rigorous operational reasoning citing subsystem backpressure, ITV cycle times, and yard tipping points.

---

### Exercise 7: FMC OSRA 2022 Compliance Audit
*Focus:* Regulatory compliance, demurrage billing rules, and tariff validation.  
*Time:* 30 minutes.  
*Deliverable:* Audit memorandum assessing the legal validity of a $14,200 demurrage invoice.

A beneficial cargo owner receives a demurrage invoice from a terminal operator for 5 containers held in the yard for 12 days past the 4-day free-time limit ($5 \text{ containers} \times 8 \text{ demurrage days} \times \$355/\text{day} = \$14,200$).
An audit of terminal logs reveals:
- On Days 5 and 6, the terminal gate was closed due to a union holiday.
- On Days 7 and 8, the containers were located in "Block 4G," which was placed on a temporary operational lockdown because yard cranes were clearing export stacks for an outbound vessel.
- On Day 9, the trucker attempted to book a TAS appointment, but the TAS platform displayed "Zero Appointments Available" for import pickups.
- The invoice sent to the BCO omits the specific container availability date and the certifying officer signature.

Evaluate the legal validity of this invoice under US FMC OSRA 2022 regulations (46 CFR Part 541). Calculate the revised lawful demurrage charge.

*Acceptance Criteria:* Identification of statutory billing deficiencies; application of the FMC "Incentive Principle"; exact recalculation of legally collectible demurrage.

---

### Exercise 8: Automated Container Terminal Architecture Design
*Focus:* Green-field terminal design, layout selection, and buffer engineering.  
*Time:* 45 minutes.  
*Deliverable:* Architectural blueprint specification detailing equipment choices, yard layout, and decoupling mechanisms.

You are the lead terminal automation architect designing a new 2,000,000 TEU/year automated terminal. 
1. Choose between a **parallel yard block layout** (blocks parallel to the quay) and a **perpendicular yard block layout** (blocks perpendicular to the quay). Justify your choice based on horizontal transport path crossings and crane interference.
2. Specify the horizontal transport technology (Terminal Tractors vs. Straddle Carriers vs. Battery-Electric AGVs) and explain how it decouples the STS crane cycle from the yard stacking cycle.
3. Design a physical buffer mechanism to prevent landside drayage volatility from starving the waterside quay cranes.

*Acceptance Criteria:* Comprehensive technical rationale; clear architectural trade-offs; explicit handling of decoupling buffers.

---

## 13. Self-check answers

### Answer to Exercise 1: Erlang-C Berth Sizing
1. **Current System Metrics ($s = 3$):**
   $$\text{Offered Load } a = \lambda \cdot \mathbb{E}[S] = 0.0625 \times 36.0 = 2.25 \text{ Erlangs}$$
   $$\text{Utilization } \rho = \frac{a}{s} = \frac{2.25}{3} = 0.75 \quad (75.0\%)$$

   Compute state summation for $k = 0, 1, 2$:
   $$\frac{2.25^0}{1} = 1.0000, \quad \frac{2.25^1}{1} = 2.2500, \quad \frac{2.25^2}{2} = 2.5313 \implies \sum_{k=0}^2 \frac{a^k}{k!} = 5.7813$$
   $$\text{Saturated term } (k=3): \frac{2.25^3}{6 \times (1 - 0.75)} = \frac{11.3906}{1.50} = 7.5938$$
   $$P_0 = \frac{1}{5.7813 + 7.5938} = \frac{1}{13.3751} = 0.074766$$
   $$P_w = 7.5938 \times 0.074766 = 0.5678 \quad (56.78\%)$$
   $$W_q = \frac{P_w}{s\mu - \lambda} = \frac{0.5678}{(3 \times 0.02778) - 0.0625} = \frac{0.5678}{0.08333 - 0.0625} = \frac{0.5678}{0.02083} = 27.26 \text{ hours}$$

2. **Surcharge Exposure:**
   Since current waiting time ($27.26 \text{ h}$) exceeds the 18.0-hour threshold, the $250/\text{TEU}$ surcharge triggers:
   $$\text{Annual Risk} = 1,200,000 \text{ TEUs} \times \$250/\text{TEU} = \$300,000,000/\text{year}$$

3. **Expanded System ($s = 4$):**
   $$\rho = \frac{2.25}{4} = 0.5625 \quad (56.25\%)$$
   $$\sum_{k=0}^3 \frac{a^k}{k!} = 5.7813 + \frac{2.25^3}{6} = 5.7813 + 1.8984 = 7.6797$$
   $$\text{Saturated term } (k=4): \frac{2.25^4}{24 \times (1 - 0.5625)} = \frac{25.6289}{24 \times 0.4375} = \frac{25.6289}{10.50} = 2.4408$$
   $$P_0 = \frac{1}{7.6797 + 2.4408} = \frac{1}{10.1205} = 0.098809$$
   $$P_w = 2.4408 \times 0.098809 = 0.2412 \quad (24.12\%)$$
   $$W_q = \frac{0.2412}{(4 \times 0.02778) - 0.0625} = \frac{0.2412}{0.11111 - 0.0625} = \frac{0.2412}{0.04861} = 4.96 \text{ hours}$$
   
   *Conclusion:* Adding the 4th berth slashes anchorage wait time from 27.26 hours to 4.96 hours (well below the 18.0-hour threshold). The annualized capex cost of $18,000,000 avoids a potential $300,000,000 carrier surcharge. The investment is overwhelmingly justified.

---

### Answer to Exercise 2: Pollaczek-Khinchine Variability
1. **Traffic Intensity:**
   $$\rho = \lambda \cdot \mathbb{E}[S] = 0.040 \times 20.0 = 0.80 \quad (80.0\%)$$

2. **Waiting Metrics Across Regimes:**
   - **Scenario A (High Chaos: $\sigma_s = 25.0 \text{ h}$):**
     $$C_s = \frac{25.0}{20.0} = 1.25 \implies C_s^2 = 1.5625$$
     $$W_q = \frac{\rho \mathbb{E}[S]}{2(1 - \rho)} (1 + C_s^2) = \frac{0.80 \times 20.0}{2 \times (1 - 0.80)} (1 + 1.5625) = \frac{16.0}{0.40} \times 2.5625 = 40.0 \times 2.5625 = 102.50 \text{ hours}$$
     $$L_q = \lambda W_q = 0.040 \times 102.50 = 4.10 \text{ vessels}$$

   - **Scenario B (Standard Exponential: $\sigma_s = 20.0 \text{ h}$):**
     $$C_s = 1.00 \implies C_s^2 = 1.0000$$
     $$W_q = 40.0 \times (1 + 1.0000) = 40.0 \times 2.0 = 80.00 \text{ hours}$$
     $$L_q = 0.040 \times 80.00 = 3.20 \text{ vessels}$$

   - **Scenario C (Advanced Scheduling: $\sigma_s = 8.0 \text{ h}$):**
     $$C_s = \frac{8.0}{20.0} = 0.40 \implies C_s^2 = 0.1600$$
     $$W_q = 40.0 \times (1 + 0.1600) = 40.0 \times 1.1600 = 46.40 \text{ hours}$$
     $$L_q = 0.040 \times 46.40 = 1.86 \text{ vessels}$$

3. **Operational Mechanism:**
   Queues build up when short inter-arrival times coincide with long service times. In Scenario A, service times frequently run to 45+ hours, blocking the channel while subsequent vessels continue to arrive according to the Poisson process. Stabilizing service times ($\sigma_s \to 8 \text{ h}$) eliminates long tail blockages, cutting waiting times by 54.7% ($102.5 \text{ h} \to 46.4 \text{ h}$) with zero increase in nominal crane speed.

---

### Answer to Exercise 3: Continuous BAP Conflict Resolution
1. **Simultaneous Berthing Check:**
   $$\sum_{i \in \{A, B, C\}} L_i = 400 + 450 + 350 = 1,200 \text{ meters}$$
   Since total required quay space ($1,200 \text{ m}$) strictly exceeds available quay length $L_{\text{quay}} = 900 \text{ m}$, all three vessels cannot berth simultaneously at $t = 8 \text{ h}$.

2. **Earliest Berthing Time for Vessel C:**
   - Vessel A occupies $[0, 400 \text{ m}]$ from $t = 0 \text{ h}$ to $t = 0 + 24 = 24 \text{ h}$.
   - Vessel B occupies $[400, 850 \text{ m}]$ from $t = 6 \text{ h}$ to $t = 6 + 20 = 26 \text{ h}$.
   - At $t = 8 \text{ h}$, the remaining unallocated quay space is from $850 \text{ m}$ to $900 \text{ m}$ ($50 \text{ meters}$).
   - Vessel C requires $350 \text{ m}$. It cannot fit spatially until either Vessel A or Vessel B departs.
   - Vessel A departs at $t = 24 \text{ h}$. Upon A's departure, coordinates $[0, 400 \text{ m}]$ become available.
   - Since Vessel C requires $350 \text{ m} \le 400 \text{ m}$, Vessel C can berth at coordinate $m_C = 0 \text{ m}$ at time $b_C = 24.0 \text{ hours}$.

3. **Separation Logic:**
   - Vessel C is separated from Vessel A in **time** ($z_{AC}^y = 1$: Vessel A departs before Vessel C berths).
   - Vessel C is separated from Vessel B in **space** ($z_{CB}^x = 1$: Vessel C at $[0, 350 \text{ m}]$ is to the left of Vessel B at $[400, 850 \text{ m}]$).

---

### Answer to Exercise 4: Yard Reshuffling
1. **Theoretical Expected Reshuffles ($\mathbb{E}[R]$):**
   - $H = 3$: $\mathbb{E}[R] = (3 - 1)/2 = 1.00$
   - $H = 4$: $\mathbb{E}[R] = (4 - 1)/2 = 1.50$
   - $H = 5$: $\mathbb{E}[R] = (5 - 1)/2 = 2.00$

2. **Effective Reshuffles at $U = 0.80$ ($\alpha = 0.20$):**
   $$\text{Penalty Factor } \Phi(0.80) = 1 + \frac{0.20 \times (0.80)^2}{1 - 0.80} = 1 + \frac{0.20 \times 0.64}{0.20} = 1 + 0.64 = 1.64$$
   - $H = 3$: $\mathbb{E}[R_{\text{eff}}] = 1.00 \times 1.64 = 1.64 \text{ reshuffles}$
   - $H = 4$: $\mathbb{E}[R_{\text{eff}}] = 1.50 \times 1.64 = 2.46 \text{ reshuffles}$
   - $H = 5$: $\mathbb{E}[R_{\text{eff}}] = 2.00 \times 1.64 = 3.28 \text{ reshuffles}$

3. **Annual Crane Hours ($1,200 \text{ retrievals/week} \times 52 \text{ weeks} = 62,400 \text{ retrievals/year}$):**
   - Cycle time: $T_{\text{cycle}} = 70 + (\mathbb{E}[R_{\text{eff}}] \times 60) \text{ seconds}$.
   - $H = 3$: $T = 70 + (1.64 \times 60) = 70 + 98.4 = 168.4 \text{ s}$. Total hours $= (62,400 \times 168.4) / 3,600 = 2,918.9 \text{ hours}$.
   - $H = 4$: $T = 70 + (2.46 \times 60) = 70 + 147.6 = 217.6 \text{ s}$. Total hours $= (62,400 \times 217.6) / 3,600 = 3,771.7 \text{ hours}$.
   - $H = 5$: $T = 70 + (3.28 \times 60) = 70 + 196.8 = 266.8 \text{ s}$. Total hours $= (62,400 \times 266.8) / 3,600 = 4,624.5 \text{ hours}$.
   
   *Conclusion:* Stacking 5-high instead of 3-high saves ground footprint but requires an additional $1,705.6 \text{ crane operating hours}$ per year (+58.4% operating expense) strictly due to wasted reshuffling moves.

---

### Answer to Exercise 5: TAS Quota Sizing
1. **Theoretical Capacity:**
   $$10 \text{ lanes} \times 24 \text{ trucks/lane/hour} = 240 \text{ trucks/hour}$$

2. **Maximum Appointment Quota ($Q_t$):**
   $$Q_t = \lfloor 240 \times 0.80 \rfloor = 192 \text{ appointments/hour}$$

3. **Pre-TAS Peak Assessment:**
   $$\rho_{\text{peak}} = \frac{280}{240} = 1.167 \quad (\rho > 1.0)$$
   The queue was completely unstable. Demand exceeded capacity by 40 trucks/hour, producing exponential queue growth, gridlock on access corridors, and turn times exceeding 2 hours.

4. **Dual-Transaction Optimization:**
   In a single transaction, 1 truck visit = 1 container move (0.40 moves/minute of gate capacity). In a dual transaction, 1 truck visit = 2 container moves (1 export drop + 1 import pickup).
   $$\text{Effective Container Flow} = \lambda_{\text{gate}} \cdot (1 + f_{\text{dual}})$$
   where $f_{\text{dual}}$ is the fraction of dual transactions. If $f_{\text{dual}}$ increases from 0.10 to 0.50, effective gate container throughput increases by 36.4% with zero additional lanes or physical asphalt.

---

### Answer to Exercise 6: Terminal Bottleneck Shift Critique Rubric
A complete critique must score on three core dimensions:
1. **Theory of Constraints Identification (40%):** Recognizes that the system bottleneck currently resides in the yard (86% occupancy) and the gate (95 min turn time), NOT at the berth.
2. **Coupling Dynamics (30%):** Explains that faster quay cranes will discharge containers faster onto the quay apron, but because yard cranes are saturated with reshuffles ($U = 0.86$), transfer vehicles (ITVs) will find nowhere to ground the boxes. The apron will choke with queued ITVs, and the new $60M cranes will be forced into "crane starvation" (idling with spreaders up).
3. **Correct Capital Allocation (30%):** Recommends allocating capital to off-dock container storage, empty evacuation sweepers, automated gate OCR kiosks, and TAS software rather than waterside steel.

---

### Answer to Exercise 7: FMC OSRA 2022 Compliance Audit
1. **Statutory Non-Compliance:**
   - Incomplete Invoice: The invoice omits the mandatory container availability date and certifying officer signature, which violates 46 CFR § 541.6. Under OSRA 2022, an invoice lacking mandatory elements is non-enforceable.
2. **Application of the Incentive Principle (46 U.S.C. § 41102(c)):**
   - Days 5 and 6: Terminal was closed. Shippers could not physically access containers. Demurrage cannot accrue during terminal gate closures.
   - Days 7 and 8: Block 4G locked down by terminal operations. Cargo was inaccessible through no fault of the cargo owner. Demurrage unlawful.
   - Day 9: Terminal appointment system had zero available slots. A terminal cannot charge storage fees while actively locking truckers out of appointment access.
3. **Lawful Recalculation:**
   - Total billed days past free time: 8 days (Days 5 through 12).
   - Inaccessible days: 5 days (Days 5, 6, 7, 8, 9).
   - Lawful accrual days: 3 days (Days 10, 11, 12).
   - Legally collectible demurrage: $5 \text{ containers} \times 3 \text{ days} \times \$355/\text{day} = \$5,325$. (Savings to BCO: $\$8,875$).

---

### Answer to Exercise 8: Automated Container Terminal Architecture Design Rubric
1. **Block Layout (40%):** Strong answer selects a **perpendicular yard block layout**. Perpendicular blocks physically decouple waterside AGV flow from landside drayage truck flow. Parallel layouts force external street trucks and internal terminal AGVs into the same travel corridors, creating cross-traffic bottlenecks and collision risks.
2. **Horizontal Transport Decoupling (30%):** Selects Lift-AGVs or Straddle Carriers capable of grounding containers onto buffer frames beneath the STS crane backreach. This completely decouples crane cycle time from vehicle arrival jitter.
3. **Buffer Engineering (30%):** Proposes dedicated landside exchange grids, on-dock rail cantilever spurs, and automated container parking racks that absorb highway arrival surges without passing variance into active stacking blocks.

---

## 14. Interview questions

The following questions evaluate candidates on deep operational engineering, queuing mathematics, and trade compliance for roles such as Director of Port Operations, Maritime Logistics Data Scientist, or Global Supply Chain Network Architect.

### Q1: "If our container berth utilization is currently running at 82%, why shouldn't we accept a contract for an additional weekly liner service that would push utilization to 89%?"
*Strong Answer:*  
"Berth queuing follows non-linear queuing dynamics (Erlang-C for multi-berth and Pollaczek-Khinchine for single channels). In the Erlang-C formulation, average waiting time is inversely proportional to $(1 - \rho)$. At $\rho = 0.82$, $(1 - \rho) = 0.18$. At $\rho = 0.89$, $(1 - \rho) = 0.11$. Pushing utilization from 82% to 89% represents a 64% increase in the congestion multiplier. Because vessel arrivals and service times have high stochastic variance ($C_s^2 > 0.3$), that 7 percentage point increase in volume will more than double average anchorage waiting times from 20 hours to over 50 hours. The resulting vessel delays will cause carriers to miss proforma windows, incur tens of thousands of dollars per day in vessel charter losses, and prompt shipping lines to levy Port Congestion Surcharges or omit our port entirely."  
*Weak Answer:*  
"Because 89% is too close to 100%, and if a crane breaks down, we might get backed up."

---

### Q2: "What is the physical and mathematical mechanism that causes container yard operations to collapse when occupancy exceeds 80%?"
*Strong Answer:*  
"The collapse is driven by the physics of container re-handling and non-linear search times. In an import block, containers are retrieved randomly. Expected reshuffles per retrieval for a stack of height $H$ is $(H - 1)/2$ only if empty adjacent slots are available in the same bay. When occupancy exceeds 75–80%, open slots vanish. Cranes cannot drop an obstructing container in the same bay; they must gantry down the block to drop it in an empty slot three bays over, gantry back, retrieve the target box, and repeat. As Kim (1997) formalized, the reshuffle penalty scales proportionally to $\alpha U^2 / (1 - U)$. When $U \to 0.85$, net crane productivity drops by more than 50% from 25 moves/hour to under 11 moves/hour. Yard cranes spend all their energy shuffling obstructions rather than servicing trucks, which halts horizontal transport and causes quay cranes to stall."  
*Weak Answer:*  
"The yard gets full, so drivers have to wait longer because there is no space to put new containers."

---

### Q3: "How does the Pollaczek-Khinchine formula demonstrate the value of digitizing vessel arrivals and standardizing crane gang operations?"
*Strong Answer:*  
"The P-K formula for an $M/G/1$ queue defines waiting time as $W_q = \frac{\rho \mathbb{E}[S]}{2(1-\rho)} (1 + C_s^2)$. The term $(1 + C_s^2)$ decouples the effect of average speed from operational variability. If a terminal operates with high variability—such as uncoordinated arrivals, crane breakdowns, and stowage confusion—$C_s$ may be 1.2 ($C_s^2 = 1.44$). If we invest in DCSA Just-In-Time digital port call handshakes, predictive stowage planning, and standardized crane lashing procedures, we can cut service time standard deviation in half, reducing $C_s^2$ from 1.44 to 0.36. This reduces waiting time by:
$$\frac{1 + 0.36}{1 + 1.44} = \frac{1.36}{2.44} = 0.557 \quad (\text{a } 44.3\% \text{ reduction in vessel delay})$$
without purchasing new cranes or dredging deeper berths. Variability reduction is pure operational leverage."  
*Weak Answer:*  
"It shows that if you have better software and fewer breakdowns, ships spend less time waiting around."

---

### Q4: "What is the difference between Discrete and Continuous Berth Allocation, and why is the continuous version computationally harder?"
*Strong Answer:*  
"In the Discrete Berth Allocation Problem (BAP), the quay wall is partitioned into fixed, predefined berths, each accommodating one ship at a time. It models as a classic multi-processor scheduling problem or parallel machine assignment. In the Continuous BAP, the quay wall has no physical divisions; it is treated as a continuous physical resource of length $L$. A vessel can be berthed at any coordinate $m$ along the wall, provided it does not overlap with other vessels in space or time. Continuous BAP is a 2D space-time rectangle packing problem with dynamic arrival constraints ($b_i \ge a_i$). It is strongly NP-hard because the spatial decision variable $m_i$ and temporal variable $b_i$ are continuous, requiring $O(N^2)$ binary disjunctive constraints ($z_{ij}^x, z_{ij}^y$) to prevent rectangle overlap."  
*Weak Answer:*  
"Discrete means you have distinct berths like Berth 1 and Berth 2, while continuous means the whole dock is open. Continuous is harder because you have more choices."

---

### Q5: "How should a marine terminal design a Truck Appointment System (TAS) to optimize internal yard crane productivity rather than just gate line flow?"
*Strong Answer:*  
"A naive TAS only throttles aggregate gate arrivals to prevent highway queues. An advanced TAS synchronizes gate quotas with the real-time capacity of specific yard blocks and equipment. First, quotas must be partitioned down to individual yard blocks: if Block 3 has one RTG capable of 22 net moves/hour, the TAS must not issue more than 15 import appointments for Block 3 in that hour window. Second, the TAS must dynamically incentivize dual transactions (matchbacks). A dual transaction allows a truck to drop an export and pick up an import in one trip, doubling the productivity of gate and terminal transfer assets. Third, appointments must dynamically adapt to vessel working schedules: if an STS crane is intensively discharging imports into Block 2, the TAS should suppress landside appointments for Block 2 during that shift to prevent twin-crane rail clashes."  
*Weak Answer:*  
"You build a website where truckers pick a time slot so they don't all show up at 8:00 AM."

---

### Q6: "Under US FMC OSRA 2022 rules, what operational conditions invalidate a terminal's right to assess demurrage fees on an import container?"
*Strong Answer:*  
"Under OSRA 2022 and 46 CFR Part 541, demurrage is legally defined as an incentive mechanism to promote cargo fluidity. Therefore, demurrage cannot be assessed if the cargo owner or drayman was physically prevented from retrieving the container due to operational barriers created by the terminal or ocean carrier. Invalidation conditions include: (1) container located in a closed or inaccessible yard block; (2) terminal gates closed due to weather, labor holidays, or IT outages; (3) the Truck Appointment System having zero available appointments within the free-time window; (4) equipment holds, such as a missing chassis or container mounted on an out-of-service chassis; and (5) missing statutory data elements on the invoice, such as availability dates or mandatory dispute policies. If any of these conditions occur, the terminal's right to collect demurrage is forfeited."  
*Weak Answer:*  
"If the terminal has long lines or the computer system crashes, you don't have to pay."

---

### Q7: "Why do automated container terminals (like Rotterdam Maasvlakte II) use perpendicular yard blocks rather than parallel yard blocks?"
*Strong Answer:*  
"In parallel block layouts, the long axis of the block runs parallel to the quay wall. This forces external road trucks and internal quay transfer vehicles (AGVs/terminal tractors) to share the same longitudinal circulation roadways, creating severe cross-traffic congestion, intersection bottlenecks, and safety hazards. In perpendicular layouts, the blocks run at a 90-degree angle to the quay. This allows complete physical decoupling: the waterside end of the block interfaces exclusively with unmanned AGVs transiting to the quay cranes, while the landside end of the block interfaces exclusively with external road trucks and rail shuttles. The two traffic streams never cross paths, and automated stacking cranes (ASCs) can transfer containers between the two domains without human-machine conflict."  
*Weak Answer:*  
"Perpendicular blocks fit better on the land and make it easier for trains to come in."

---

### Q8: "What is the 'peel-off' or 'free-flow' pile strategy, and how does it alter yard crane reshuffling dynamics?"
*Strong Answer:*  
"A peel-off pile replaces random-access container stacking with strict bulk delivery. Instead of sorting containers by individual consignee and placing them in random import stacks (which requires 1.5 to 2.5 reshuffles per retrieval), the terminal pre-dotes hundreds of containers for a single high-volume BCO or drayage consortium into a dedicated block stacked strictly by size. When external drayage trucks arrive, the yard crane simply 'peels off' whatever container is on the top of the stack and places it on the next available chassis. Reshuffles drop to exactly zero ($\mathbb{E}[R] = 0$). Truck turn times drop from 60 minutes to under 15 minutes, crane moves jump from 12 to 35+ moves/hour, and high-density yard space is utilized at maximum asset efficiency."  
*Weak Answer:*  
"It's when you peel containers off the top of the pile so you don't have to dig for them."

---

## 15. Key terms

- **Twenty-Foot Equivalent Unit (TEU):** Standard measure of containerized cargo volume based on a $20' \times 8' \times 8.5'$ ISO container.
- **Berth Allocation Problem (BAP):** Optimization formulation assigning vessels to spatial quay coordinates and time slots.
- **Quay Crane Assignment Problem (QCAP):** Mathematical programming model allocating STS cranes to berthed vessels over time.
- **Ship-to-Shore (STS) Crane:** High-capacity rail-mounted gantry crane operating on the quay wall to handle container moves.
- **Gross Moves Per Hour (GMPH):** Total container lifts executed by a crane divided by total elapsed operating hours.
- **Net Moves Per Hour (NMPH):** Container lifts executed adjusted for non-handling delays (hatch covers, lashing, boom clearance).
- **Rubber-Tyred Gantry (RTG):** Mobile gantry crane running on rubber tires, stacking containers 1-over-4 to 1-over-6 in yard blocks.
- **Automated Stacking Crane (ASC):** Unmanned rail-mounted gantry crane operating in automated container terminal blocks.
- **Internal Transfer Vehicle (ITV):** Dedicated yard tractor or AGV shuttling containers between quay cranes and yard stacks.
- **Crane Starvation (Quay Lock):** Condition where a quay crane idles because no horizontal transport vehicle is available to receive or feed a container.
- **Reshuffling (Re-handling):** Unproductive crane moves required to dig out a buried target container from a stack.
- **Yard Tipping Point:** The 75–80% yard occupancy threshold where re-handling moves explode and crane productivity collapses.
- **Pollaczek-Khinchine (P-K) Formula:** Analytical queuing equation computing mean waiting time in an $M/G/1$ queue as a function of service time variance.
- **Erlang-C Formula:** Queuing formulation for the probability of waiting in an $M/M/s$ multi-server queuing network.
- **Truck Appointment System (TAS):** Digital scheduling platform regulating landside drayage arrivals via hourly quota caps.
- **Gate Turn Time (GTT):** Elapsed duration from truck arrival at the terminal gate queue to outgate departure.
- **Dual Transaction (Matchback):** A single truck visit combining an inbound container drop-off with an outbound pickup.
- **Weigh-In-Motion (WIM):** Ingate dynamic scale technology verifying container weight compliance with SOLAS VGM rules.
- **Verified Gross Mass (VGM):** Mandatory certified container weight under IMO SOLAS Chapter VI Regulation 2.
- **Demurrage:** Storage penalty assessed on containers occupying terminal yard space beyond contractually granted free time.
- **Detention (Per Diem):** Equipment usage fee assessed on containers or chassis held outside the marine terminal past free time.
- **Free Time:** Grace period granted under carrier tariffs during which container storage incurs no demurrage.
- **Ocean Shipping Reform Act of 2022 (OSRA 2022):** US statutory reform prohibiting unfair demurrage/detention practices and mandating billing transparency.
- **Just-In-Time (JIT) Port Call:** DCSA digital coordination standard enabling vessels to adjust sailing speed to match confirmed berth availability.
- **Cold Ironing (Shore Power):** Providing shoreside electrical power to a berthed vessel, allowing auxiliary diesel engines to shut down.
- **On-Dock Rail (ODR):** Intermodal rail terminal located inside the marine terminal footprint, eliminating highway drayage legs.
- **Chassis Pool:** Shared inventory of intermodal chassis managed by equipment leasing consortia or terminal operators.
- **Trouble Kiosk:** Exception gate lane where trucks with customs holds, missing bookings, or weight discrepancies are processed.
- **Peel-Off Pile (Free-Flow):** Terminal stacking strategy where containers for a single large account are stacked together and pulled strictly top-to-bottom.
- **Port Congestion Surcharge (PCS):** Carrier emergency fee levied on cargo owners to offset vessel anchorage delays at congested ports.

---

## 16. Sources

1. **Bierwirth, C., & Meisel, F.** (2010). A survey of berth allocation and quay crane scheduling problems in container terminals. *European Journal of Operational Research*, 202(3), 615–627. https://doi.org/10.1016/j.ejor.2009.05.031
2. **Bierwirth, C., & Meisel, F.** (2015). A follow-up survey of berth allocation and quay crane scheduling problems in container terminals. *European Journal of Operational Research*, 244(3), 675–689. https://doi.org/10.1016/j.ejor.2014.12.030
3. **Carlo, H. J., Vis, I. F., & Roodbergen, K. J.** (2014). Storage yard operations in container terminals: An overview of operations research-oriented research. *European Journal of Operational Research*, 235(2), 322–340. https://doi.org/10.1016/j.ejor.2013.10.033
4. **Cordeau, J. F., Laporte, G., Legato, P., & Moccia, L.** (2005). Models and solutions for the berth allocation problem. *Annals of Operations Research*, 136(1), 291–315. https://doi.org/10.1007/s10479-005-2063-4
5. **Kim, K. H.** (1997). Evaluation of the number of rehandles in container yards. *Computers & Industrial Engineering*, 32(4), 701–711. https://doi.org/10.1016/S0360-8352(97)00024-7
6. **Steenken, D., Voß, S., & Stahlbock, R.** (2004). Container terminal operation and operations research: a classification and literature review. *OR Spectrum*, 26(1), 3–49. https://doi.org/10.1007/s00291-003-0157-z
7. **United States Federal Maritime Commission (FMC).** (2022). *Demurrage and Detention Billing Requirements* (Docket No. 22-04, 46 CFR Part 541). Federal Register, 87(34), 9872–9885.
8. **United Nations Conference on Trade and Development (UNCTAD).** (2023). *Review of Maritime Transport 2023: Towards a green and just transition*. United Nations Publications. ISBN: 978-92-1-002894-3.
9. **Digital Container Shipping Association (DCSA).** (2022). *DCSA Interface Standard for Just-in-Time Port Calls, Version 1.2*. DCSA Standards Architecture Committee.
10. **International Maritime Organization (IMO).** (2016). *Guidelines Regarding the Verified Gross Mass of a Container Carrying Cargo* (MSC.1/Circ.1475). IMO Maritime Safety Committee.

---

## 17. Links

- **Upward Index:** [`_index.md`](_index.md) — Module 36 Course Outline and Topic Map
- **Sibling Topics:**
  - [`01_incoterms-documents.md`](01_incoterms-documents.md) — Incoterms, Risk Allocation, and Trade Documentation
  - [`02_landed-cost.md`](02_landed-cost.md) — Total Landed Cost Architecture and Financial Modeling
  - [`04_modal-choice-emissions.md`](04_modal-choice-emissions.md) — Multimodal Freight Selection, Cost-Transit Boundaries, and Carbon Accounting
  - [`06_demurrage-detention.md`](06_demurrage-detention.md) — Demurrage and Detention: Root Cause Analysis, Dispute Audits, and Carrier Tariffs
  - [`07_freight-procurement-rate-risk.md`](07_freight-procurement-rate-risk.md) — Ocean Freight Procurement, Index-Linked Contracts, and Surcharge Hedging
  - [`08_chokepoints-resilience.md`](08_chokepoints-resilience.md) — Maritime Chokepoint Vulnerability, Geopolitical Disruption, and Routing Resiliency
- **Foundational Curriculum Modules:**
  - [Module 11: Safety Stock and Inventory Buffering](../../02_methods/11_safety_stock_inventory_buffering/_index.md)
  - [Module 14: Network Design and Facility Location](../../02_methods/14_network_design_facility_location/_index.md)
  - [Module 19: International Logistics and Customs Execution](../../03_commerce/19_international_logistics_customs/_index.md)
- **Root Survey File:** [`../../../course/36_global_freight_ports_trade_policy.md`](../../../course/36_global_freight_ports_trade_policy.md)
