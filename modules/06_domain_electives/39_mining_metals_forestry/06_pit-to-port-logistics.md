# Pit-to-Port Logistics: Heavy-Haul Rail, Overland Conveyors, Slurry Hydraulics, and Terminal Demurrage Optimization

> Module 39 — Mining, Metals, and Forestry Supply Chains
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 15 (Production Planning and Scheduling), Module 38 (Process Industries, Commodities, and Bulk Supply Chains), Module 39 Topic 03 (Drilling, Blasting, and Load-Haul Operations), Topic 04 (Grade Blending and Specification), and Topic 05 (Processing and Recovery)
> Version 2026.5

---

## 1. Why this topic exists

In bulk mineral commodity supply chains—such as iron ore, metallurgical coal, thermal coal, bauxite, and base-metal concentrates—the mine site is rarely situated adjacent to deepwater shipping infrastructure. Mineral deposits are geologically confined to remote inland cratons, continental interiors, and high-altitude cordilleras, frequently situated 200 to 1,000 kilometers from the nearest navigable coastline. Consequently, the pit-to-port logistics corridor is not an auxiliary transport function; it is the physical and economic spine of the entire extractive enterprise.

The logistical movement of bulk solids represents between 30% and 60% of the total cash operating costs (Free on Board, or FOB, cash cost per dry metric tonne) of delivered mineral products. In high-volume operations producing between 20 million and 350 million wet metric tonnes per annum (Mtpa), unit margins are intensely sensitive to small perturbations in transport velocity, asset utilization, and handling losses. An uncoordinated logistics corridor destroys value through four primary mechanisms:

1. **Extreme Capital Asset Interlocking and Captive Dependency:** Heavy-haul freight corridors rely on specialized, captive infrastructure: privately owned single-track railways operating at 35-to-40-tonne axle loads, multi-kilometer overland conveyors traversing rugged topography, high-pressure slurry pipelines running across private easements, and dedicated multi-berth marine terminals. Because these physical networks are entirely captive to the mine assets, there is no open spot market for third-party transport substitution. A derailment, conveyor tear, or pipeline blockage brings the entire cash-generating capacity of the mining complex to an immediate halt once upstream surge capacities saturate.
2. **Dynamic Bottlenecks and Buffer Saturation:** The pit-to-port pipeline consists of series-coupled discrete and continuous systems: batch extraction and crushing, continuous overland transport, batch train loading, discrete rail transit, batch rotary car dumping, continuous stacker-reclaimer blending, and batch vessel hatch loading. The buffer capacities separating these stages—Run-of-Mine (ROM) pads, train loadout silos, stockyard windrows, and surge bins—are constrained by strict spatial and capital limits. A four-hour mechanical breakdown at a marine terminal rotary car dumper propagates upstream across a single-track rail network, forcing loaded trains into passing sidings, starving inland loadout silos, halting mine production, and stranding haul trucks at the primary crusher.
3. **Severe Ocean Demurrage Exposure:** Global marine bulk transport operates under strict Charter Party contracts. Dry bulk carriers—ranging from 60,000 Deadweight Tonne (DWT) Ultramax vessels to 180,000 DWT Capesize and 400,000 DWT Valemax ore carriers—accrue punitive financial penalties ("demurrage") if terminal loading exceeds agreed contractual laytime. Demurrage rates fluctuate between $15,000 and $45,000 per vessel per day. When a terminal queue extends to 25 or 40 vessels off ports such as Port Hedland, Dalrymple Bay, or Ponta da Madeira due to mismatched rail arrivals or stockyard shortfalls, demurrage costs rapidly escalate to millions of dollars per week, eroding operating margins.
4. **Physical Degradation, Slurry Sanding, and Quality Loss:** Granular bulk materials change physically during transport. Lump iron ore breaks down through mechanical drops at conveyor transfers and car dumpers, generating sub-6.3 mm fines that reduce product realization prices. Slurry pipelines transporting copper or iron concentrate risk catastrophic "sanding" if slurry velocity drops below the critical deposition velocity, causing unrecoverable pipeline plugging, or severe pipe wall erosion if driven at excessive velocities. At the port, excessive moisture leads to cargo liquefaction risks under maritime safety codes (IMSBC Code Transportable Moisture Limit), while insufficient moisture causes severe dust emission penalties.

Mastering pit-to-port logistics requires a unified mathematical and operational understanding of heavy-haul rail dynamics, fluid mechanics of slurry transport, overland conveyor mechanical design, automated bulk yard scheduling, and maritime berth allocation optimization. This topic details the engineering mechanics, mathematical algorithms, and operational protocols required to run a synchronized, low-variance pit-to-port supply chain.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Calculate** the cycle times, train throughput, and required rolling stock fleet size (locomotives and freight wagons) for captive heavy-haul rail networks under axle load, speed profile, and passing siding constraints.
2. **Formulate and solve** the single-track train scheduling problem with passing loop conflicts using dynamic priority dispatching and mixed-integer linear programming formulations.
3. **Compute** belt tension profiles, rolling resistance coefficients ($f$), and electrical power balances—including regenerative downhill braking power—for long-distance overland conveyors according to DIN 22101 and ISO 5048 standards.
4. **Determine** the critical deposition velocity ($V_c$), non-Newtonian Bingham plastic rheology parameters, and friction head losses for high-concentration mineral slurry pipelines to establish safe hydraulic operating envelopes.
5. **Analyze** rotary car dumper cycle times, indexer-positioner kinematics, and stockyard surge bin dynamics to identify terminal discharge capacity limits.
6. **Design** kinematic anti-collision envelopes and coordinate bounding boxes for automated bucket-wheel stacker-reclaimers operating in multi-machine longitudinal stockyards.
7. **Construct** vessel hatch-loading sequences that respect maximum allowable ship hull bending moments and shear forces under International Association of Classification Societies (IACS) standards while synchronizing with ship deballasting pumping rates.
8. **Formulate** a Mixed-Integer Linear Program (MILP) for the Berth Allocation and Shiploader Scheduling Problem (BAP/SSP) to minimize total vessel turnaround time and port demurrage penalties under tidal and stockyard availability constraints.

---

## 3. The core mental model

The pit-to-port logistics network is a **stochastic, tightly coupled discrete-continuous queuing network with finite buffer capacities and strict maritime deadline penalties**. Material flows through alternating regimes of continuous bulk transport and batch-mode conveyance, bounded by inland surge buffers and terminal stockyards.

### 3.1 The Integrated Flow Architecture

The physical network consists of five physical transformation stages coupled by four primary inventory buffers:

```text
==============================================================================================================
                                THE PIT-TO-PORT LOGISTICAL CONTINUUM
==============================================================================================================

   INLAND EXTRACTION & PROCESSING            INLAND TRANSPORT CORRIDOR                EXPORT MARINE TERMINAL
  ┌─────────────────────────────────┐   ┌────────────────────────────────┐   ┌───────────────────────────────┐
  │  Open-Pit Mine / Beneficiation  │   │     Captive Transport Mode     │   │   Stockyard, Berth & Wharf    │
  │                                 │   │                                │   │                               │
  │  ┌───────────┐   ┌───────────┐  │   │  [Mode A: Heavy-Haul Rail]     │   │  ┌─────────────────────────┐  │
  │  │  Primary  │   │ Secondary │  │   │  - 35-40t Axle Load Rakes      │   │  │   Rotary Car Dumper     │  │
  │  │  Crusher  │──►│ Processing│  │   │  - Single-Track + Loops        │   │  │   (Tandem/Triple/Quad)  │  │
  │  └─────┬─────┘   └─────┬─────┘  │   │               OR               │   │  └───────────┬─────────────┘  │
  │        │               │        │   │  [Mode B: Overland Conveyor]   │   │              │                │
  │        ▼               ▼        │   │  - 5-25 km High-Tension Belt   │   │              ▼                │
  │  ┌───────────────────────────┐  │   │  - Regenerative Downhill Braking│  │  ┌─────────────────────────┐  │
  │  │ Train Loadout (TLO) Silos │──┼───┤               OR               ├───┼─►│   Stockyard Windrows    │  │
  │  │ or Slurry Pump Station    │  │   │  [Mode C: Slurry Pipeline]     │   │  │ (Automated Stacker-    │  │
  │  └───────────────────────────┘  │   │  - Non-Newtonian Hydraulics    │   │  │  Reclaimers: CAS LiDAR) │  │
  └─────────────────────────────────┘   │  - Positive Displacement Pumps │   │  └───────────┬─────────────┘  │
                                        └────────────────────────────────┘   │              │                │
                                                                             │              ▼                │
                                                                             │  ┌─────────────────────────┐  │
                                                                             │  │  Berth & Shiploaders    │  │
                                                                             │  │  (Tidal Windows, Hatch  │  │
                                                                             │  │   Sequencing, IACS S1A) │  │
                                                                             │  └───────────┬─────────────┘  │
                                                                             └──────────────┼────────────────┘
                                                                                            │
                                                                                            ▼
                                                                             ┌───────────────────────────────┐
                                                                             │   Ocean-Going Bulk Vessels    │
                                                                             │ (Capesize/Newcastlemax/Valemax│
                                                                             │  NOR Laytime Clock Running!)  │
                                                                             └───────────────────────────────┘
==============================================================================================================
```

### 3.2 The Closed-Loop Feedback and Information Architecture

Logistical decisions cannot be made in an upstream-to-downstream forward-push manner. Bulk shipping operations are pulled by vessel arrival notices and Charter Party laytime clocks. The terminal must synchronize train dispatches from inland mines days in advance to assemble dedicated cargo parcels (often 150,000 to 260,000 tonnes of specific grade blends) on stockyard pads immediately prior to vessel berthing.

```text
==============================================================================================================
                             CLOSED-LOOP PIT-TO-PORT CONTROL TOPOLOGY
==============================================================================================================

  14-Day Vessel Stem ──► 7-Day Berth Plan ──► Stockyard Pad ──► Train Sequence ──► Mine Loadout Silo
     (ETA Window)          (Tidal Draft)        Allocation          Schedule          Inventory Target
          │                      │                   │                   │                   │
          ▼                      ▼                   ▼                   ▼                   ▼
  ┌───────────────┐      ┌───────────────┐   ┌───────────────┐   ┌───────────────┐   ┌───────────────┐
  │ Vessel Queue  │◄─────┤ Shiploader    │◄──┤ Reclaimer     │◄──┤ Car Dumper    │◄──┤ Train Inbound │
  │ & Demurrage   │      │ Execution     │   │ & Conveyors   │   │ Unloading     │   │ Rakes Transit │
  └───────────────┘      └───────────────┘   └───────────────┘   └───────────────┘   └───────────────┘
          │                      ▲                   ▲                   ▲                   ▲
          │                      │                   │                   │                   │
          └──────────────────────┴───────────────────┴───────────────────┴───────────────────┘
                                REAL-TIME DYNAMIC BOTTLENECK CONTROL
==============================================================================================================
```

If the rail schedule fails to deliver the required grade blend on time, the reclaimer starves, the shiploader idles, the vessel exceeds its laytime allowance, and demurrage accrues every hour until completion. Conversely, if trains arrive faster than the terminal can dump and stack, inbound rail rakes accumulate on terminal holding tracks, backing up into mainline passing loops and freezing the entire railway network.

---

## 4. Definitions and vocabulary

The following table establishes the formal mathematical, physical, and contractual terminology governing pit-to-port logistics operations.

| Term | Definition | Mathematical Symbol / Unit | Common Misconception or Operational Pitfall |
| :--- | :--- | :--- | :--- |
| **Axle Load** | The total gross weight exerted by a single wheelset on the rail infrastructure. | $L_{\text{axle}}$ ($\text{metric tonnes}$, $\text{t/axle}$) | Assuming track capacity is solely limited by rail steel weight; subgrade modulus and bridge fatigue are often the binding structural limits. |
| **Gross Train Weight (GTW)** | The combined weight of all locomotives, empty freight wagons (tare), and transported mineral payload. | $W_{\text{train}}$ ($\text{metric tonnes}$, $\text{t}$) | Conflating payload with gross weight when computing braking distances, tractive effort, and kinetic energy on downhill gradients. |
| **Distributed Power (DP)** | Locomotive placement architecture where remote locomotive units are spaced inside or at the rear of the train rake, controlled via wireless telemetry. | Ratio: $\text{Lead} : \text{Mid} : \text{Tail}$ | Assuming all locomotives must sit at the head; in long trains ($>2.0\text{ km}$), head-only tractive effort shears couplers via excessive in-train draft forces. |
| **Train Cycle Time** | Total round-trip duration for a train rake: loading, loaded transit, dumping, empty transit, and operational queuing. | $T_{\text{cycle}}$ ($\text{hours}$) | Calculating cycle time purely as distance divided by speed, ignoring passing loop dwell, crew changes, and terminal queueing variance. |
| **Passing Loop (Siding)** | A secondary track section parallel to a single-track mainline allowing trains traveling in opposite directions (or different speeds) to pass. | $L_{\text{loop}}$ ($\text{meters}$) | Neglecting deceleration and acceleration transition times when scheduling meets; train length must include safety buffer clearance. |
| **Headway** | The minimum safe temporal or spatial separation required between consecutive trains operating in the same direction on a shared track block. | $h_{\text{min}}$ ($\text{minutes}$ or $\text{km}$) | Treating headway as constant; headway expands non-linearly with train mass, downhill gradient, and braking reaction latency. |
| **Overland Conveyor (OLC)** | Continuous trough-belt conveyor system transporting bulk solids over long distances across undulating natural terrain. | Length $L$ ($\text{km}$), Capacity $Q$ ($\text{t/h}$) | Assuming steady-state power draw governs motor sizing; starting and abortive stopping transient tensions govern belt tensile rating. |
| **Regenerative Braking** | Mechanical-to-electrical energy conversion by drive motors on downhill conveyor flights or electric locomotives descending grades. | $P_{\text{regen}}$ ($\text{kW}$ or $\text{MW}$) | Failing to install dynamic braking resistor banks or grid-tie substations capable of absorbing regenerative surges without tripping the local grid. |
| **Belt Turnover** | Mechanism at conveyor head or tail terminals that twists the return belt 180 degrees so the clean side rests on return idlers. | Dimensionless angle ($180^{\circ}$) | Neglecting turnover transition distance; insufficient transition length causes high edge stresses and catastrophic belt carcass delamination. |
| **Concentration by Weight** | Ratio of solid dry mineral mass to total slurry mixture mass in a slurry pipeline. | $C_w$ (dimensionless or $\%$) | Treating $C_w$ as interchangeable with volumetric concentration $C_v$; slurry hydraulics and settling kinetics depend fundamentally on $C_v$. |
| **Concentration by Volume** | Ratio of solid mineral volume to total slurry mixture volume. | $C_v$ (dimensionless or $\%$) | Underestimating the exponential viscosity surge that occurs when $C_v$ approaches dense packing limits ($C_v > 40\%$). |
| **Critical Deposition Velocity** | The minimum mean slurry flow velocity required to maintain solid mineral particles in suspension without forming a stationary bed. | $V_c$ ($\text{m/s}$) | Operating too close to $V_c$ to save pumping energy; minor flow surges or particle size coarsening will induce pipeline sanding and plugging. |
| **Bingham Plastic Fluid** | Non-Newtonian fluid exhibiting a finite yield stress that must be exceeded before flow initiates, followed by linear plastic viscosity. | Yield stress $\tau_y$ ($\text{Pa}$), Viscosity $\eta_p$ ($\text{Pa}\cdot\text{s}$) | Applying Newtonian Navier-Stokes head loss equations to concentrated slurries, massively underestimating laminar start-up pressure requirements. |
| **Slack Flow** | Open-channel flow condition inside a slurry pipeline when the internal hydraulic grade line drops below the physical pipeline elevation profile. | Dimensionless state | Allowing slack flow on downhill pipeline segments; causes severe internal pipe cavitation, destructive column separation, and catastrophic pipe rupture. |
| **Rotary Car Dumper** | Mechanical apparatus that clamps loaded ore wagons and rotates them 160° to 180° along their longitudinal axis to discharge bulk cargo. | Cycle time $t_d$ ($\text{seconds/dump}$) | Assuming car dumpers require train uncoupling; heavy-haul trains utilize rotary couplers allowing dumping while remaining coupled. |
| **Indexer-Positioner** | High-precision hydraulic or electro-mechanical arm that advances a train rake forward by exactly one or two car lengths into the dumper. | Velocity $v_p$ ($\text{m/s}$), Stroke ($\text{m}$) | Failing to account for wagon draft gear slack run-in and run-out, which causes severe shock loading and wagon derailment during indexing. |
| **Collision Avoidance System (CAS)**| Dynamic sensor and control framework preventing mechanical contact between moving stockyard machines (stackers, reclaimers). | Spatial envelope $(x, y, z, \theta, \phi)$ | Relying exclusively on GPS; satellite multipath and structural shadowing require redundant 3D LiDAR, radar, and hardwired PLC interlocks. |
| **Transportable Moisture Limit** | Maximum gross moisture content at which a granular mineral cargo is considered safe for maritime transport without risk of liquefying. | TML ($\%$ moisture by weight) | Shipping concentrates near Flow Moisture Point (FMP); ship vibration during ocean transit induces liquefaction, cargo shifting, and capsizing. |
| **Notice of Readiness (NOR)** | Formal legal document tendered by a vessel master to terminal operators declaring the vessel is physically ready to receive cargo in all holds. | Timestamp (ISO 8601) | Tendering invalid NOR prior to customs clearance, hold cleanliness certification, or anchoring within port limits, invalidating laytime inception. |
| **Laytime** | Contractually agreed duration permitted to the charterer to complete loading or discharging of a vessel without incurring additional fees. | Days or Weather Working Days ($\text{WWD}$) | Confusing calendar days with Weather Working Days; rain, wind, or tidal surges may legally pause the laytime clock under maritime rules. |
| **Demurrage** | Liquidated financial compensation paid by the cargo shipper/charterer to the vessel owner for time spent loading beyond agreed laytime. | Rate $D$ ($\$/\text{day}$) | Viewing demurrage as a minor operational friction; across large export volumes, cumulative demurrage can eliminate operating profitability. |
| **Despatch** | Financial bonus paid by the vessel owner to the charterer for completing vessel loading ahead of the contractual laytime expiration. | Rate: standardly $0.5 \times D$ ($\$/\text{day}$) | Assuming despatch compensates for disrupted train schedules; rapid loading can sub-optimize inland rail logistics and grade homogenization. |
| **IACS Hatch Pouring Sequence** | Structured multi-pass cargo loading schedule designed to keep ship hull longitudinal stresses within allowable safety thresholds. | Metric tonnes per hold per pass | Loading holds to completion in a single pass; uneven mass distribution generates excessive bending moments and hull girder structural fracture. |

---

## 5. The mechanics, step by step

Achieving high-throughput, low-variance pit-to-port logistics requires rigorous application of physical and mathematical formulations across four engineering domains: heavy-haul rail networks, overland conveyor/slurry hydraulics, stockyard automation, and marine terminal berth allocation.

### 5.1 Captive Heavy-Haul Rail Operations and Dynamic Sequencing

Heavy-haul rail represents the primary mass-transit mechanism for iron ore, bauxite, and coal. Modern systems operate rakes measuring 2.0 to 3.5 kilometers in length, hauling 20,000 to 42,000 tonnes of payload per train.

#### 5.1.1 Train Resistance and Tractive Physics

The tractive effort $F_{\text{TE}}$ required by diesel-electric or electric locomotives must overcome total train resistance $R_{\text{total}}$, calculated via the empirical Davis Equation expanded for heavy-haul track dynamics:

$$R_{\text{total}} = R_{\text{rolling}} + R_{\text{grade}} + R_{\text{curve}} + R_{\text{accel}}$$

The base train resistance per unit mass $r_0$ ($\text{N/kN}$ or $\text{N/t}$) is expressed as:

$$r_0 = A + B \cdot v + C \cdot v^2$$

where:
- $A$ represents journal friction and rolling resistance between wheel tread and rail head ($\text{N/t}$). For heavy-haul welded rail on concrete sleepers with 40-tonne axle loads, $A \approx 6.5\text{--}8.5\text{ N/t}$.
- $B$ represents mechanical flange friction and track depression wave resistance ($\text{N}\cdot\text{s}/(\text{t}\cdot\text{m})$), typically $0.05\text{--}0.12\text{ N}\cdot\text{s}/(\text{t}\cdot\text{m})$.
- $C$ represents aerodynamic drag coefficient per unit mass ($\text{N}\cdot\text{s}^2/(\text{t}\cdot\text{m}^2)$), heavily dependent on train cross-sectional area and inter-wagon gap turbulence ($C \approx 0.0015\text{--}0.0035\text{ N}\cdot\text{s}^2/(\text{t}\cdot\text{m}^2)$).
- $v$ is train velocity ($\text{m/s}$).

Grade resistance $R_{\text{grade}}$ is strictly gravitational:

$$R_{\text{grade}} = W_{\text{train}} \cdot g \cdot \sin(\theta) \approx W_{\text{train}} \cdot g \cdot i$$

where $W_{\text{train}}$ is gross train weight ($\text{kg}$), $g = 9.81\text{ m/s}^2$, and $i$ is track gradient (dimensionless slope, where $1\% = 0.01$). On a $1.0\%$ adverse gradient, grade resistance contributes $98.1\text{ N/t}$, completely dominating rolling friction.

Curve resistance $R_{\text{curve}}$ accounts for wheel flange slipping and scuffing on curved track:

$$R_{\text{curve}} = W_{\text{train}} \cdot g \cdot \left(\frac{k_c}{R_{\text{radius}}}\right)$$

where $R_{\text{radius}}$ is curve radius ($\text{m}$) and $k_c \approx 0.6\text{--}0.8\text{ m}$ for standard gauge heavy-haul track.

#### 5.1.2 Axle Load Limits and Track Structural Integrity

Track infrastructure limits maximum axle load $L_{\text{axle}}$ based on Talbot's equation for rail bending stress under dynamic wheel loading:

$$\sigma_{\text{rail}} = \frac{P_{\text{dynamic}} \cdot x_0}{4 \cdot Z_{\text{rail}}} \le \sigma_{\text{allowable}}$$

where:
- $P_{\text{dynamic}} = \phi_{\text{dynamic}} \cdot \frac{L_{\text{axle}} \cdot g}{2}$ is dynamic wheel load ($\text{N}$).
- $\phi_{\text{dynamic}} = 1 + \frac{k_d \cdot v}{D_{\text{wheel}}}$ is the dynamic amplification factor ($k_d \approx 0.004\text{--}0.006$, $D_{\text{wheel}}$ in meters).
- $x_0 = \sqrt[4]{\frac{4 \cdot E \cdot I_{\text{rail}}}{u_{\text{track}}}}$ is the characteristic track length ($\text{m}$), where $E$ is steel elastic modulus ($207\text{ GPa}$), $I_{\text{rail}}$ is rail moment of inertia ($68\text{ kg/m}$ or $136\text{ RE}$ rail: $I \approx 3.9 \times 10^{-5}\text{ m}^4$), and $u_{\text{track}}$ is track subgrade modulus ($\text{N/m}^2$).
- $Z_{\text{rail}}$ is rail section modulus ($\text{m}^3$).

Operating at 40-tonne axle loads ($L_{\text{axle}} = 40\text{ t}$, wheel load $= 200\text{ kN}$) requires premium micro-alloyed or hypereutectoid heat-treated steel rail ($>370\text{ HBW}$) and heavy concrete sleepers spaced at $0.60\text{ m}$ centers to prevent rail head surface fatigue, corrugation, and ballast crushing.

#### 5.1.3 Train Cycle Time and Fleet Sizing Formulation

Total round-trip cycle time for rake $k$ is calculated as:

$$T_{\text{cycle}} = T_{\text{load\_wait}} + T_{\text{load}} + T_{\text{transit\_loaded}} + T_{\text{dump\_wait}} + T_{\text{dump}} + T_{\text{transit\_empty}} + T_{\text{maint\_dwell}}$$

Let $Q_{\text{annual}}$ be the annual target throughput ($\text{dry metric tonnes per annum}$), $w_{\text{payload}}$ be payload capacity per train rake ($\text{tonnes}$), and $\mu_{\text{moist}}$ be average bulk moisture fraction. Annual wet tonnage is $Q_{\text{wet}} = Q_{\text{annual}} / (1 - \mu_{\text{moist}})$.

The total number of annual train trips required is:

$$N_{\text{trips}} = \left\lceil \frac{Q_{\text{wet}}}{w_{\text{payload}}} \right\rceil$$

Annual operating availability is $H_{\text{net}} = 8760 \times A_{\text{network}} \times (1 - U_{\text{maint}})$, where $A_{\text{network}}$ is network operating efficiency and $U_{\text{maint}}$ accounts for planned track maintenance windows.

The active train fleet size (number of operating train rakes $N_{\text{rakes}}$) is governed by:

$$N_{\text{rakes}} = \left\lceil \frac{N_{\text{trips}} \cdot \bar{T}_{\text{cycle}}}{H_{\text{net}}} \right\rceil$$

#### 5.1.4 Dynamic Single-Track Train Scheduling (Meet-Pass Conflict Resolution)

Captive mining railways primarily utilize single-track infrastructure with discrete passing loops (sidings) situated at intervals of 20 to 45 kilometers. Scheduling trains $i \in \mathcal{T}_{\text{loaded}}$ (mine to port) and $j \in \mathcal{T}_{\text{empty}}$ (port to mine) across track blocks $b \in \mathcal{B}$ requires enforcing non-overlapping occupancy constraints.

Let $t_{i, b}^{\text{arr}}$ and $t_{i, b}^{\text{dep}}$ be the arrival and departure times of train $i$ at block $b$. For any single-track segment $b$ bounded by passing sidings $s$ and $s+1$, two trains moving in opposite directions cannot simultaneously occupy block $b$:

$$t_{j, b}^{\text{dep}} \ge t_{i, b}^{\text{arr}} + h_{\text{clear}} - M \cdot (1 - y_{i,j,b})$$

$$t_{i, b}^{\text{dep}} \ge t_{j, b}^{\text{arr}} + h_{\text{clear}} - M \cdot y_{i,j,b}$$

where:
- $y_{i,j,b} \in \{0, 1\}$ is a binary decision variable equal to 1 if train $i$ clears block $b$ before train $j$ enters, and 0 otherwise.
- $h_{\text{clear}}$ is the mandatory signal clearing and safety headway interval (typically $4\text{ to }8\text{ minutes}$).
- $M$ is a sufficiently large positive constant.

If train $i$ enters siding $s$ to allow opposing train $j$ to pass on the main track, train $i$ must decelerate, wait until train $j$ clears the block plus safety headway $h_{\text{clear}}$, and accelerate back to mainline running speed, incurring a siding dwell delay:

$$\Delta t_{\text{siding}} = t_{\text{decel}} + (t_{j, s}^{\text{pass}} - t_{i, s}^{\text{arr}}) + h_{\text{clear}} + t_{\text{accel}}$$

Dynamic dispatch engines solve this conflict graph using receding-horizon Mixed-Integer Linear Programming or discrete-event constraint propagation to minimize cumulative delay across all active rakes.

---

### 5.2 Long-Distance Overland Conveyors and Mineral Slurry Pipelines

Where continuous conveyance is viable, overland conveyors and slurry pipelines replace or supplement heavy-haul rail corridors.

#### 5.2.1 Overland Conveyor Belt Tensions and Power Balance

Long-distance overland conveyors (OLCs) span 5 to 25 kilometers per single flight. Belt tension calculations follow DIN 22101 and ISO 5048. Total required motor drive force $F_U$ at the drive pulley circumference equals total motion resistance:

$$F_U = F_H + F_N + F_{St}$$

1. **Main Resistance ($F_H$):** Frictional resistance of idler bearings and rubber indentation rolling resistance along carrying and return runs:

$$F_H = f \cdot L_{\text{conv}} \cdot g \cdot \left[ m'_R + (2 m'_B + m'_G) \cdot \cos(\delta) \right]$$

where:
- $f$ is the artificial friction coefficient ($0.012\text{--}0.016$ for low-rolling-resistance rubber compounds on precision idlers).
- $L_{\text{conv}}$ is conveyor center-to-center length ($\text{m}$).
- $m'_R$ is the mass per unit length of rotating idler parts ($\text{kg/m}$).
- $m'_B$ is the mass per unit length of the conveyor belt carcass and covers ($\text{kg/m}$).
- $m'_G$ is the mass per unit length of conveyed bulk material ($\text{kg/m}$), where $m'_G = \frac{Q_{\text{mass}}}{3.6 \cdot v_{\text{belt}}}$.
- $\delta$ is conveyor inclination angle.

2. **Lift Resistance ($F_{St}$):** Gravitational energy to elevate or lower cargo over net vertical lift $H_{\text{lift}}$:

$$F_{St} = m'_G \cdot g \cdot H_{\text{lift}}$$

On downhill conveyors where $H_{\text{lift}} < 0$, if $|F_{St}| > F_H + F_N$, the conveyor becomes **regenerative**: the bulk material drives the belt, and drive motors function as induction generators feeding power back into the mine electrical grid:

$$P_{\text{shaft}} = \frac{F_U \cdot v_{\text{belt}}}{1000 \cdot \eta_{\text{drive}}} \quad (\text{kW})$$

If $P_{\text{shaft}} < 0$, net electrical power generated is $P_{\text{regen}} = |P_{\text{shaft}}| \cdot \eta_{\text{drive}}$.

3. **Euler-Eytelwein Slip Limit:** To prevent drive pulley slippage, the ratio of tight-side tension $T_1$ to slack-side tension $T_2$ must satisfy:

$$\frac{T_1 - T_{\text{cent}}}{T_2 - T_{\text{cent}}} \le e^{\mu \cdot \theta_{\text{wrap}}}$$

where $\mu$ is the friction coefficient between drive pulley lagging and belt bottom cover ($0.35\text{--}0.45$ for grooved ceramic lagging), $\theta_{\text{wrap}}$ is drive wrap angle in radians, and $T_{\text{cent}} = m'_B \cdot v_{\text{belt}}^2$ is centrifugal belt tension.

#### 5.2.2 Mineral Slurry Pipeline Hydraulics and Critical Deposition Velocity

Slurry pipelines transport finely ground mineral concentrates (copper, magnetite, phosphate) suspended in water over hundreds of kilometers.

1. **Concentration Conversions:** Solids concentration by weight $C_w$ and solids concentration by volume $C_v$ are related via solid specific gravity $S_s = \rho_s / \rho_w$:

$$C_v = \frac{C_w}{S_s - C_w \cdot (S_s - 1)}$$

Slurry mixture density is:

$$\rho_m = \rho_w \cdot \left[ 1 + C_v \cdot (S_s - 1) \right] = \frac{\rho_w}{1 - C_w \cdot \left(1 - \frac{1}{S_s}\right)}$$

2. **Critical Deposition Velocity ($V_c$):** Slurry pipelines must operate above the deposition velocity $V_c$ to prevent solid particles from settling onto the pipe invert. Durand's modified formulation governs:

$$V_c = F_L \cdot \sqrt{2 \cdot g \cdot D_{\text{pipe}} \cdot \left(\frac{S_s - S_L}{S_L}\right)}$$

where:
- $D_{\text{pipe}}$ is internal pipe diameter ($\text{m}$).
- $S_L$ is carrier liquid specific gravity ($1.0$ for water).
- $F_L$ is Durand's factor, parameterized by solid particle median diameter $d_{50}$ and volumetric concentration $C_v$. For fine mineral concentrates ($d_{50} \approx 35\text{--}75\ \mu\text{m}$) at $C_v \approx 0.30\text{--}0.38$, $F_L$ typically ranges between $1.05$ and $1.25$.

Safe operational velocity is maintained within a strict window:

$$1.15 \cdot V_c \le V_{\text{oper}} \le 2.5\text{ m/s}$$

Operating below $1.15 \cdot V_c$ introduces bed formation and plugging risks; operating above $2.5\text{ m/s}$ accelerates internal pipe wall abrasive wear.

3. **Rheology and Head Loss:** Concentrated mineral slurries exhibit non-Newtonian Bingham plastic behavior governed by yield stress $\tau_y$ ($\text{Pa}$) and plastic viscosity $\eta_p$ ($\text{Pa}\cdot\text{s}$):

$$\tau = \tau_y + \eta_p \cdot \dot{\gamma}$$

The transition from laminar to turbulent flow is determined by the Hedstrom number $He$ and Bingham Reynolds number $Re_B$:

$$He = \frac{\rho_m \cdot \tau_y \cdot D_{\text{pipe}}^2}{\eta_p^2}, \quad Re_B = \frac{\rho_m \cdot V \cdot D_{\text{pipe}}}{\eta_p}$$

Critical Reynolds number $Re_{c}$ for laminar-turbulent transition is:

$$Re_{c} \approx \frac{He}{8} \cdot \left(1 - \frac{4}{3} x_c + \frac{1}{3} x_c^4\right)$$

where $x_c = \tau_y / \tau_w$. Slurry pipelines are engineered to operate in the turbulent regime ($Re_B > Re_c$) to maintain homogeneous particle suspension while minimizing viscous drag.

Hydraulic friction gradient $i_m$ (meters of slurry head loss per meter of pipe, $\text{m/m}$) is calculated via the Darcy-Weisbach equation with slurry friction factor $f_m$:

$$i_m = \frac{\Delta h_f}{L} = f_m \cdot \frac{V^2}{2 \cdot g \cdot D_{\text{pipe}}}$$

Total pumping pressure required across pipeline length $L_{\text{pipe}}$ with elevation change $\Delta z$ is:

$$P_{\text{pump}} = \rho_m \cdot g \cdot \left( i_m \cdot L_{\text{pipe}} + \Delta z \right) \quad (\text{Pa})$$

#### 5.2.3 Slack Flow Prevention

When traversing mountain passes descending toward coastal ports, the static downward gradient can exceed the hydraulic friction gradient ($|\Delta z / L| > i_m$). Under these conditions, the internal pipeline pressure drops below atmospheric pressure ($P_{\text{pipe}} < P_{\text{vapor}}$), inducing **slack flow** (cavitation, vapor cavity formation, and severe column separation). 

Slack flow destroys pipeline linings and causes pipe collapse or catastrophic overpressure water-hammer waves upon column rejoining. Prevention requires installing fixed orifice choke stations or backpressure control valves at terminal locations to hold the Hydraulic Grade Line (HGL) above the physical pipeline elevation profile across all steady-state and transient operating regimes.

---

### 5.3 Bulk Terminal Stockyard Automation and Stacker-Reclaimer Coordination

Marine bulk terminals decouple rail or pipeline transport from vessel loading using extensive stockyards capable of storing 5% to 15% of annual terminal throughput.

#### 5.3.1 Rotary Car Dumper Cycle Time Mechanics

At the terminal rail receiving station, train rakes enter the car dumper via automated indexer-positioners. Modern heavy-haul networks use tandem or triple rotary dumpers operating on rotary-coupled wagons:

```text
==============================================================================================================
                                  ROTARY CAR DUMPER CYCLE TIMING (90-SECOND BASE)
==============================================================================================================
  Step                                    Duration (s)   Cumulative Time (s)
  ─────────────────────────────────────────────────────────────────────────────
  1. Indexer arm advances rake 2 cars     [18.0 s]       |██████              18.0 s
  2. Wheel chocks engage & clamps descend [ 6.0 s]       |██                  24.0 s
  3. Dumper barrel rotates 0° to 165°     [24.0 s]       |████████            48.0 s
  4. Ore discharges into receiving hopper [10.0 s]       |███                 58.0 s
  5. Barrel returns 165° to 0°            [20.0 s]       |███████             78.0 s
  6. Clamps release & wheel chocks retract[ 4.0 s]       |█                   82.0 s
  7. Indexer clears for next stroke       [ 8.0 s]       |███                 90.0 s
==============================================================================================================
```

The theoretical instantaneous throughput capacity of a tandem dumper discharging two 120-tonne payload wagons every $t_d = 90\text{ seconds}$ is:

$$Q_{\text{dumper\_theo}} = \frac{2 \times 120\text{ t}}{90\text{ s}} \times 3600\text{ s/h} = 9,600\text{ t/h}$$

Effective long-term throughput $Q_{\text{dumper\_eff}}$ accounts for train rake swap delays ($t_{\text{swap}} \approx 15\text{--}30\text{ minutes}$ between rakes), hopper level choke interlocks, and mechanical maintenance:

$$Q_{\text{dumper\_eff}} = Q_{\text{dumper\_theo}} \cdot \text{OEE}_{\text{dumper}}$$

where $\text{OEE}_{\text{dumper}} = A_{\text{avail}} \cdot P_{\text{perf}} \cdot Q_{\text{qual}}$ (typically $0.65\text{--}0.78$ in sustained heavy-haul operations).

#### 5.3.2 Stockyard Stacking and Reclaiming Blending Control

Stacking creates homogenized product piles using Chevron or Windrow stacking methods. Reclaiming is executed via automated Bucket-Wheel Stacker-Reclaimers (BWSR) operating bench by bench:

- **Chevron Method:** The stacker travels longitudinally along the center line of the pile, depositing material layer upon layer directly above the center.
- **Windrow Method:** The stacker deposits multiple smaller parallel longitudinal berms across the yard width, minimizing particle roll-down segregation of lump and fines.

The theoretical variance reduction ratio $R_{\text{var}}$ achieved through stacking $N_{\text{layers}}$ layers and reclaiming across the transverse face is:

$$R_{\text{var}} = \frac{\sigma^2_{\text{reclaimed}}}{\sigma^2_{\text{inbound}}} \approx \frac{1}{N_{\text{layers}}}$$

#### 5.3.3 Stacker-Reclaimer Kinematic Collision Avoidance Systems (CAS)

In automated stockyards, multiple machines (e.g., Stacker 1, Stacker-Reclaimer 2, Reclaimer 3) operate along shared or adjacent rail tracks. Each machine $m$ possesses three primary kinematic degrees of freedom:
1. Rail travel position $x_m(t)$ along track length $L_{\text{yard}}$ ($\text{m}$).
2. Boom slew angle $\theta_m(t)$ relative to machine centerline ($\text{rad}$).
3. Boom luffing angle $\phi_m(t)$ relative to horizontal plane ($\text{rad}$).

The instantaneous spatial boundary of machine $m$ is represented by a set of dynamic geometric primitives $\mathcal{P}_m(x_m, \theta_m, \phi_m)$. The anti-collision constraint mandates that the minimum 3D Euclidean distance between any two machines $m$ and $k$ exceeds a certified safety buffer $D_{\text{safe}}$:

$$\min_{p_1 \in \mathcal{P}_m, p_2 \in \mathcal{P}_k} \| p_1 - p_2 \|_2 \ge D_{\text{safe}} + v_{\text{rel}} \cdot t_{\text{stop}} + \frac{1}{2} a_{\text{max}} \cdot t_{\text{stop}}^2$$

where $v_{\text{rel}}$ is relative approach velocity, $a_{\text{max}}$ is emergency braking deceleration, and $t_{\text{stop}}$ is system latency plus braking application time. Machine Programmable Logic Controllers (PLCs) continuously execute spatial intersection checks against shared 3D point-cloud envelopes generated by dual redundant LiDAR scanners and millimeter-wave radar arrays.

---

### 5.4 Marine Berth Dispatching, Hatch Sequencing, and Demurrage Optimization

The marine terminal is the commercial interface where bulk cargo transfers into chartered vessels. Operations are governed by naval architecture constraints and Charter Party commercial law.

#### 5.4.1 Bulk Carrier Longitudinal Hull Strength and Hatch Sequencing

Bulk carriers (e.g., 180,000 DWT Capesize with 9 cargo holds) are susceptible to structural failure during loading. Under International Association of Classification Societies (IACS) Unified Requirement S1A, loading must never induce still-water bending moments ($BM$) or shear forces ($SF$) exceeding permissible structural design envelopes:

```text
==============================================================================================================
                                BULK CARRIER HATCH LOADING SEQUENCE (2-PASS IACS)
==============================================================================================================

  Vessel Holds:    [H1]     [H2]     [H3]     [H4]     [H5]     [H6]     [H7]     [H8]     [H9]
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  Pass 1 (50-60%): 10,000t           12,000t           14,000t           12,000t           10,000t  (Trim odd)
                   ──────►           ──────►           ──────►           ──────►           ──────►
  Pass 1 (cont.):           11,000t           13,000t           13,000t           11,000t           (Trim even)
                            ──────►           ──────►           ──────►           ──────►
  ──────────────────────────────────────────────────────────────────────────────────────────────────────────
  Pass 2 (Final):  Remaining hold capacities adjusted to target final draft and longitudinal trim (even keel)
==============================================================================================================
```

Loading holds in a single continuous pass induces catastrophic hull **sagging** (excessive weight amidships causing deck compression failure) or **hogging** (excessive weight in end holds causing keel buckling). Therefore, shiploading follows a structured multi-pass plan (typically 2 to 3 passes per hold), synchronized with the vessel's deballasting pumps. If the terminal shiploader loading rate ($Q_{\text{load}} \approx 8,000\text{ t/h}$) exceeds the ship's ballast discharge capacity ($Q_{\text{ballast}} \approx 3,000\text{--}4,000\text{ t/h}$), loading must be throttled or stopped, idling the berth.

#### 5.4.2 Demurrage and Despatch Economics

Charter parties establish contractual laytime rules:
- **Tender of Notice of Readiness (NOR):** Transmitted when the vessel arrives at anchorage, holds are inspected and approved, and the ship is in all respects ready to load.
- **Laytime Commences:** In accordance with charter terms (e.g., "12 hours after valid NOR tendered, or upon berthing, whichever is earlier").
- **Laytime Allowed ($T_{\text{allowed}}$):** Computed from agreed cargo quantity $W_{\text{cargo}}$ and daily loading rate $R_{\text{contract}}$ (in Weather Working Days of 24 consecutive hours, WWD):

$$T_{\text{allowed}} = \frac{W_{\text{cargo}}}{R_{\text{contract}}} \quad (\text{days})$$

- **Laytime Used ($T_{\text{used}}$):** Total elapsed time from laytime commencement to completion of loading, minus contractually excepted periods (e.g., bad weather preventing loading, terminal equipment breakdowns where specified by contract terms).

Demurrage or despatch is computed as:

$$\Delta T_{\text{laytime}} = T_{\text{used}} - T_{\text{allowed}}$$

$$\text{Financial Settlement} = \begin{cases} \Delta T_{\text{laytime}} \cdot D, & \text{if } \Delta T_{\text{laytime}} > 0 \quad (\text{Shipper pays Demurrage}) \\ |\Delta T_{\text{laytime}}| \cdot d_{\text{rate}}, & \text{if } \Delta T_{\text{laytime}} < 0 \quad (\text{Shipowner pays Despatch}) \end{cases}$$

where $D$ is contractual demurrage rate ($\$/\text{day}$) and despatch rate $d_{\text{rate}} = 0.5 \cdot D$.

#### 5.4.3 Mathematical Formulation: Berth Allocation and Shiploader Scheduling Problem (BAP/SSP)

Consider a marine terminal with a set of berths $\mathcal{B} = \{1, \dots, B\}$ and a set of calling vessels $\mathcal{V} = \{1, \dots, V\}$. Each vessel $j \in \mathcal{V}$ possesses:
- Expected arrival time $A_j$.
- Requested cargo tonnage $W_j$.
- Maximum draft requirement $d_j^{\text{draft}}$.
- Laytime allowed $T_j^{\text{allowed}}$.
- Daily demurrage penalty rate $D_j$.

Let $x_{j, b} \in \{0, 1\}$ equal 1 if vessel $j$ is assigned to berth $b$, and 0 otherwise. Let $t_j^{\text{start}}$ be berthing start time, and $t_j^{\text{finish}}$ be loading completion time. Handling duration at berth $b$ is $H_{j,b} = W_j / Q_{b}^{\text{eff}}$, where $Q_{b}^{\text{eff}}$ is the effective shiploader rate at berth $b$.

The optimization objective minimizes total port turnaround time and demurrage costs:

$$\min \sum_{j \in \mathcal{V}} \left[ C_{\text{wait}} \cdot (t_j^{\text{start}} - A_j) + \frac{D_j}{24} \cdot \max\left(0, (t_j^{\text{finish}} - A_j - T_j^{\text{allowed}})\right) \right]$$

subject to:

1. **Vessel Non-Preemption and Single Assignment:**
$$\sum_{b \in \mathcal{B}} x_{j,b} = 1 \quad \forall j \in \mathcal{V}$$

2. **Arrival Precedence:**
$$t_j^{\text{start}} \ge A_j \quad \forall j \in \mathcal{V}$$

3. **Berth Non-Overlapping Constraints:** For any two vessels $j$ and $k$ assigned to the same berth $b$ ($x_{j,b} = x_{k,b} = 1$):
$$t_k^{\text{start}} \ge t_j^{\text{finish}} + t_{\text{clear}} - M \cdot (1 - z_{j,k})$$
$$t_j^{\text{start}} \ge t_k^{\text{finish}} + t_{\text{clear}} - M \cdot z_{j,k}$$
where $z_{j,k} \in \{0, 1\}$ enforces that either vessel $j$ precedes vessel $k$ or vice versa, and $t_{\text{clear}}$ is berth unmooring/mooring tug clearance time.

4. **Tidal Window Constraints:** For deep-draft vessels ($d_j^{\text{draft}} > d_{\text{channel}}^{\text{low}}$), departure time $t_j^{\text{finish}}$ must coincide with a high-tide clearance window $\mathcal{W}_{\text{tide}} = [\tau_{\text{open}}, \tau_{\text{close}}]$:
$$\tau_{\text{open}, k} \le t_j^{\text{finish}} \le \tau_{\text{close}, k} \quad \text{for some tidal cycle } k$$

5. **Stockyard Reclaim Line Constraints:** Simultaneous reclaiming across berths cannot exceed total yard conveyor trunkline capacity:
$$\sum_{j \in \mathcal{V}(t)} Q_j^{\text{reclaim}} \le Q_{\text{yard\_trunkline}}^{\text{max}} \quad \forall t$$

---

## 6. Worked example

An iron ore producer in Western Australia operates a captive heavy-haul rail corridor linking an inland processing hub to an export marine terminal. The enterprise plans to evaluate throughput capacity, train fleet requirements, pipeline hydraulic feasibility for a secondary concentrate stream, and vessel demurrage performance.

### 6.1 Rail Network Parameters and Cycle Time Calculation

The rail corridor possesses the following parameters:
- Annual export commitment: $Q_{\text{annual}} = 55.0\text{ Mtpa}$ (dry).
- Free cargo moisture: $\mu_{\text{moist}} = 6.5\%$.
- Wet annual tonnage: $Q_{\text{wet}} = \frac{55.0}{1 - 0.065} = 58.824\text{ Mtpa}$.
- One-way mainline route distance: $L_{\text{rail}} = 410\text{ km}$ (single-track, 14 passing loops).
- Axle load limit: $L_{\text{axle}} = 40.0\text{ t/axle}$.
- Wagon specification: 4-axle gondola cars.
- Gross wagon weight limit: $4 \times 40.0\text{ t} = 160.0\text{ t}$.
- Wagon tare mass: $w_{\text{tare}} = 24.5\text{ t}$.
- Wagon net payload: $w_{\text{payload}} = 160.0 - 24.5 = 135.5\text{ t}$.
- Rake configuration: $N_{\text{wagons}} = 240\text{ wagons}$ per train.
- Total payload per train rake:
  $$W_{\text{rake\_payload}} = 240 \times 135.5\text{ t} = 32,520\text{ wet metric tonnes}$$
- Total gross train weight (excluding locomotives):
  $$W_{\text{rake\_gross}} = 240 \times 160.0\text{ t} = 38,400\text{ t}$$

Operational time components for a single round trip are:
1. **Train Loading Time ($T_{\text{load}}$):** Automated batch weigh-bin loadout operating at $7,500\text{ t/h}$:
   $$T_{\text{load}} = \frac{32,520\text{ t}}{7,500\text{ t/h}} = 4.336\text{ h} \quad (4\text{ h } 20\text{ min})$$
2. **Loaded Mainline Transit ($T_{\text{transit\_loaded}}$):** Average loaded running speed $\bar{v}_{\text{loaded}} = 65.0\text{ km/h}$:
   $$T_{\text{transit\_loaded}} = \frac{410\text{ km}}{65.0\text{ km/h}} = 6.308\text{ h} \quad (6\text{ h } 18\text{ min})$$
3. **Loaded Siding Delays ($T_{\text{siding\_delay}}$):** Expected meet-pass delays based on 14 loops:
   $$T_{\text{siding\_delay}} = 1.750\text{ h} \quad (1\text{ h } 45\text{ min})$$
4. **Terminal Receiving and Dump Wait ($T_{\text{dump\_wait}}$):** Queueing at port yard:
   $$T_{\text{dump\_wait}} = 1.250\text{ h} \quad (1\text{ h } 15\text{ min})$$
5. **Car Dumping Time ($T_{\text{dump}}$):** Tandem rotary car dumper ($2\text{ cars/cycle}$ at $85\text{ s/cycle}$):
   $$N_{\text{cycles}} = \frac{240\text{ cars}}{2} = 120\text{ cycles}$$
   $$T_{\text{dump}} = \frac{120 \times 85\text{ s}}{3600\text{ s/h}} = 2.833\text{ h} \quad (2\text{ h } 50\text{ min})$$
6. **Empty Mainline Transit ($T_{\text{transit\_empty}}$):** Average empty running speed $\bar{v}_{\text{empty}} = 78.0\text{ km/h}$:
   $$T_{\text{transit\_empty}} = \frac{410\text{ km}}{78.0\text{ km/h}} = 5.256\text{ h} \quad (5\text{ h } 15\text{ min})$$
7. **Inland Yard Servicing and Crew Dwell ($T_{\text{dwell}}$):** Inspection, refueling, brake tests:
   $$T_{\text{dwell}} = 2.500\text{ h} \quad (2\text{ h } 30\text{ min})$$

Total round-trip train cycle time is:
$$T_{\text{cycle}} = 4.336 + 6.308 + 1.750 + 1.250 + 2.833 + 5.256 + 2.500 = 24.233\text{ hours}$$

#### Train Trips and Fleet Sizing

Annual trips required:
$$N_{\text{trips}} = \frac{58,823,529\text{ t}}{32,520\text{ t/trip}} = 1,808.84 \implies 1,809\text{ trips/year}$$

Daily train dispatch rate:
$$N_{\text{daily}} = \frac{1,809\text{ trips}}{365\text{ days}} = 4.956\text{ trains/day} \quad (\approx 1\text{ train every } 4.84\text{ hours})$$

Active operating rakes required under network availability $A_{\text{net}} = 88.0\%$:
$$H_{\text{annual\_avail}} = 8,760\text{ h} \times 0.88 = 7,708.8\text{ operating hours/year}$$
$$\text{Total Fleet Hours Required} = 1,809\text{ trips} \times 24.233\text{ h/trip} = 43,837.5\text{ rake-hours}$$
$$N_{\text{active\_rakes}} = \frac{43,837.5}{7,708.8} = 5.687\text{ rakes}$$

Rounding up to the nearest integer yields **6 active train rakes**. Providing one additional rake for scheduled heavy workshop maintenance yields a total capital requirement of **7 train rakes** (totaling $7 \times 240 = 1,680\text{ ore wagons}$, plus 21 heavy-haul diesel-electric locomotives under a 3-unit Distributed Power configuration per rake).

---

### 6.2 Secondary Stream: Slurry Pipeline Hydraulic Feasibility

A secondary magnetite concentrate stream produces $Q_{\text{conc}} = 8.5\text{ Mtpa}$ (dry) transported via an adjacent 120-kilometer buried slurry pipeline.
- Solid mineral specific gravity: $S_s = 4.90$ ($\rho_s = 4,900\text{ kg/m}^3$).
- Carrier liquid: Water ($\rho_w = 1,000\text{ kg/m}^3$).
- Target solids concentration by weight: $C_w = 64.0\%$ ($0.640$).
- Particle sizing: $d_{50} = 42\ \mu\text{m}$, top size $d_{95} = 90\ \mu\text{m}$.
- Internal pipe diameter: $D_{\text{pipe}} = 0.450\text{ m}$ ($450\text{ mm}$).
- Annual pipeline operating hours: $H_{\text{pipe}} = 8,000\text{ h/year}$.

#### Step 1: Volumetric Concentration and Slurry Density

$$C_v = \frac{C_w}{S_s - C_w \cdot (S_s - 1)} = \frac{0.640}{4.90 - 0.640 \times (4.90 - 1.0)} = \frac{0.640}{4.90 - 2.496} = \frac{0.640}{2.404} = 0.2662 \quad (26.62\%)$$

Slurry mixture density is:
$$\rho_m = \rho_w \cdot [1 + C_v \cdot (S_s - 1)] = 1,000 \times [1 + 0.2662 \times 3.90] = 1,000 \times [1 + 1.0382] = 2,038.2\text{ kg/m}^3$$

#### Step 2: Critical Deposition Velocity ($V_c$)

For $d_{50} = 42\ \mu\text{m}$ at $C_v = 0.2662$, empirical Durand parameter $F_L = 1.10$.
$$V_c = F_L \cdot \sqrt{2 \cdot g \cdot D_{\text{pipe}} \cdot \left(\frac{S_s - S_L}{S_L}\right)}$$
$$V_c = 1.10 \cdot \sqrt{2 \times 9.81 \times 0.450 \times \left(\frac{4.90 - 1.00}{1.00}\right)} = 1.10 \cdot \sqrt{8.829 \times 3.90} = 1.10 \cdot \sqrt{34.433} = 1.10 \times 5.868 = 1.614\text{ m/s}$$

To ensure safe margin against deposition, operational velocity is selected at $1.20 \times V_c$:
$$V_{\text{oper}} = 1.20 \times 1.614\text{ m/s} = 1.937\text{ m/s}$$

#### Step 3: Volumetric Throughput and Flow Velocity Validation

Required dry solids mass flow rate:
$$\dot{M}_s = \frac{8,500,000\text{ t}}{8,000\text{ h}} \times \frac{1,000\text{ kg/t}}{3600\text{ s/h}} = 295.14\text{ kg/s}$$

Total slurry mixture mass flow rate:
$$\dot{M}_m = \frac{\dot{M}_s}{C_w} = \frac{295.14}{0.640} = 461.15\text{ kg/s}$$

Slurry volumetric flow rate:
$$\dot{Q}_m = \frac{\dot{M}_m}{\rho_m} = \frac{461.15\text{ kg/s}}{2,038.2\text{ kg/m}^3} = 0.2262\text{ m}^3\text{/s} \quad (814.3\text{ m}^3\text{/h})$$

Cross-sectional area of $450\text{ mm}$ pipe:
$$A_{\text{pipe}} = \frac{\pi \cdot (0.450)^2}{4} = 0.15904\text{ m}^2$$

Actual mean slurry pipeline velocity:
$$V_{\text{actual}} = \frac{\dot{Q}_m}{A_{\text{pipe}}} = \frac{0.2262\text{ m}^3\text{/s}}{0.15904\text{ m}^2} = 1.422\text{ m/s}$$

**Hydraulic Warning:** $V_{\text{actual}} (1.422\text{ m/s}) < V_c (1.614\text{ m/s})$. The selected $450\text{ mm}$ pipe diameter is too large for the design mass rate, which will cause particle settling and sanding.

**Engineering Correction:** Downsize internal pipe diameter to $D_{\text{pipe}} = 380\text{ mm}$ ($0.380\text{ m}$):
$$A_{\text{pipe\_new}} = \frac{\pi \cdot (0.380)^2}{4} = 0.11341\text{ m}^2$$
$$V_{\text{actual\_new}} = \frac{0.2262}{0.11341} = 1.994\text{ m/s}$$

Recalculate $V_c$ for $380\text{ mm}$ pipe:
$$V_{c\_\text{new}} = 1.10 \cdot \sqrt{2 \times 9.81 \times 0.380 \times 3.90} = 1.10 \cdot \sqrt{29.077} = 1.10 \times 5.392 = 1.483\text{ m/s}$$
$$V_{\text{actual\_new}} = 1.994\text{ m/s} = 1.34 \times V_c \quad (\text{Safely above deposition; satisfies hydraulic stability})$$

#### Step 4: Friction Gradient and Pumping Power

For turbulent flow of fine concentrate at $V = 1.994\text{ m/s}$, Darcy friction factor $f_m = 0.0185$.
$$i_m = f_m \cdot \frac{V^2}{2 \cdot g \cdot D_{\text{pipe}}} = 0.0185 \cdot \frac{(1.994)^2}{2 \times 9.81 \times 0.380} = 0.0185 \cdot \frac{3.976}{7.456} = 0.0185 \times 0.5333 = 0.009865\text{ m head / m pipe}$$

Total friction head loss over $L_{\text{pipe}} = 120,000\text{ m}$ (assuming zero net elevation change, $\Delta z = 0$):
$$\Delta h_f = 0.009865 \times 120,000\text{ m} = 1,183.8\text{ m of slurry head}$$

Total required pump discharge pressure:
$$P_{\text{pump}} = \rho_m \cdot g \cdot \Delta h_f = 2,038.2\text{ kg/m}^3 \times 9.81\text{ m/s}^2 \times 1,183.8\text{ m} = 23,670,000\text{ Pa} = 23.67\text{ MPa} \quad (236.7\text{ bar})$$

This pressure exceeds the rating of a single positive displacement station. The corridor is divided into **two intermediate pump stations**, each operating high-pressure piston-diaphragm pumps discharging at $11.84\text{ MPa}$ ($118.4\text{ bar}$).

Total hydraulic power required across the corridor:
$$P_{\text{hydraulic}} = \dot{Q}_m \cdot P_{\text{pump}} = 0.2262\text{ m}^3\text{/s} \times 23.67 \times 10^6\text{ Pa} = 5,354,154\text{ W} = 5.354\text{ MW}$$
Assuming mechanical pump efficiency $\eta_{\text{pump}} = 86\%$, electrical motor power consumption is:
$$P_{\text{elec}} = \frac{5.354\text{ MW}}{0.86} = 6.226\text{ MW}$$

---

### 6.3 Marine Terminal Demurrage Settlement Calculation

A Capesize dry bulk carrier, the *M/V Pilbara Enterprise*, arrives to load iron ore:
- Vessel Cargo Quantity: $W_{\text{cargo}} = 175,000\text{ wet metric tonnes}$ ($\pm 10\%$ owner's option).
- Charter Party Agreed Loading Rate: $R_{\text{contract}} = 40,000\text{ t/WWD}$ (Weather Working Day of 24 consecutive hours, Sundays and Holidays Included — SHINC).
- Demurrage Rate: $D = \$32,000.00\text{ per day}$ (pro-rata).
- Despatch Rate: $d_{\text{rate}} = \$16,000.00\text{ per day}$ (pro-rata).

#### Calculation of Allowed Laytime

$$T_{\text{allowed}} = \frac{175,000\text{ t}}{40,000\text{ t/day}} = 4.375\text{ days} = 105.000\text{ hours} \quad (4\text{ days } 9\text{ hours } 00\text{ minutes})$$

#### Chronological Statement of Facts (SOF)

1. **Monday 06:00:** Vessel arrives at pilot boarding station and tenders Notice of Readiness (NOR).
2. **Monday 12:00:** Contractual NOR notice time expires (6-hour turn time); laytime clock begins.
3. **Monday 12:00 to Tuesday 18:00:** Vessel idles at anchorage waiting for berth (30.0 hours elapsed).
4. **Tuesday 18:00 to Tuesday 20:00:** Pilotage and berthing at Berth 2 (2.0 hours).
5. **Tuesday 20:00 to Wednesday 04:00:** Loading Hold 1, 3, 5 (Pass 1). Shiploader active (8.0 hours).
6. **Wednesday 04:00 to Wednesday 14:00:** Tropical thunderstorm with gale-force winds. Port authority orders loading stopped, shiploader boomed up, hatches closed. This period is recorded as bad weather under WWD terms:
   $$\text{Weather Interruption} = 10.0\text{ hours} \quad (\text{Deducted from Laytime Used})$$
7. **Wednesday 14:00 to Friday 02:00:** Shiploader loading resumes and completes Hold 2, 4, 6, 7, 8, 9 (Pass 1 and Pass 2): 36.0 hours.
8. **Friday 02:00 to Friday 08:00:** Terminal rotary car dumper mechanical chain failure halts reclaim line. Cargo unavailable on pad. Terminal breakdown is **not** an excepted period under this charter party:
   $$\text{Terminal Delay} = 6.0\text{ hours} \quad (\text{Laytime continues to run!})$$
9. **Friday 08:00 to Friday 22:30:** Final trimming pass, draft survey, deballasting completed. Cargo completed at Friday 22:30 (14.5 hours).

#### Laytime Accounting

Total elapsed duration from laytime commencement (Monday 12:00) to completion (Friday 22:30):
$$\text{Total Elapsed Time} = 4\text{ days } 10\text{ hours } 30\text{ minutes} = 106.500\text{ hours}$$

Deduct contractually excluded time (rain/weather under WWD terms):
$$\text{Excluded Weather Time} = 10.000\text{ hours}$$

Net Laytime Used:
$$T_{\text{used}} = 106.500 - 10.000 = 96.500\text{ hours} = 4.02083\text{ days}$$

Comparison against Allowed Laytime:
$$\Delta T_{\text{laytime}} = T_{\text{used}} - T_{\text{allowed}} = 96.500\text{ h} - 105.000\text{ h} = -8.500\text{ hours} \quad (\text{Vessel completed EARLY})$$

The vessel completed loading $8.500\text{ hours}$ ($0.35417\text{ days}$) prior to laytime expiration.

#### Financial Settlement (Despatch Earned)

$$\text{Despatch Payable} = 0.35417\text{ days} \times \$16,000.00\text{/day} = \$5,666.67$$

The mining enterprise (charterer) earns a despatch credit of **$5,666.67** from the shipowner. Had the terminal delay extended an additional 12 hours, the laytime allowance would have been exceeded by $3.5\text{ hours}$, converting this outcome into a demurrage penalty of $\$4,666.67$.

---

## 7. Code, query, or calculation

The following self-contained Python script implements a discrete Berth Allocation and Shiploader Demurrage Optimization engine. It processes an arriving fleet of bulk carriers, assigns vessels to berths under shiploader rate constraints, and calculates laytime, demurrage, and despatch balances.

```python
"""Berth Allocation and Demurrage Optimization Engine for Bulk Terminals."""

from dataclasses import dataclass
from typing import List, Dict

@dataclass
class BulkCarrier:
    name: str
    arrival_hr: float
    tonnage: float
    contract_rate_tpd: float
    demurrage_per_day: float
    berth_assigned: int = 0
    start_hr: float = 0.0
    finish_hr: float = 0.0

def schedule_terminal(vessels: List[BulkCarrier], berth_rates: Dict[int, float]) -> None:
    # Sort vessels by Earliest Due Date / Arrival priority
    vessels.sort(key=lambda v: (v.arrival_hr, v.tonnage / v.contract_rate_tpd))
    berth_available = {b: 0.0 for b in berth_rates}

    print(f"{'Vessel':<18} {'Berth':<6} {'Arr(h)':<7} {'Start':<7} {'End(h)':<7} {'Lay(h)':<7} {'Result':<10} {'Amount ($)':>10}")
    print("-" * 80)
    
    total_demurrage = 0.0
    total_despatch = 0.0

    for v in vessels:
        # Assign to the berth that minimizes completion time
        best_b, best_start, best_finish = None, 1e9, 1e9
        for b, rate in berth_rates.items():
            start = max(v.arrival_hr, berth_available[b])
            load_hours = v.tonnage / rate
            finish = start + load_hours
            if finish < best_finish:
                best_b, best_start, best_finish = b, start, finish

        v.berth_assigned = best_b
        v.start_hr = best_start
        v.finish_hr = best_finish
        berth_available[best_b] = best_finish + 1.5  # 1.5h unmooring clearance

        # Laytime Accounting (WWD 24h SHINC)
        allowed_hours = (v.tonnage / v.contract_rate_tpd) * 24.0
        used_hours = v.finish_hr - v.arrival_hr
        delta_hours = used_hours - allowed_hours

        if delta_hours > 0:
            demurrage = (delta_hours / 24.0) * v.demurrage_per_day
            total_demurrage += demurrage
            status, cost = "DEMURRAGE", demurrage
        else:
            despatch = (abs(delta_hours) / 24.0) * (0.5 * v.demurrage_per_day)
            total_despatch += despatch
            status, cost = "DESPATCH", despatch

        print(f"{v.name:<18} {v.berth_assigned:<6} {v.arrival_hr:<7.1f} {v.start_hr:<7.1f} "
              f"{v.finish_hr:<7.1f} {allowed_hours:<7.1f} {status:<10} {cost:>10.2f}")

    print("-" * 80)
    print(f"Net Port Liability: Demurrage=${total_demurrage:,.2f} | Despatch=${total_despatch:,.2f} | "
          f"Balance=${total_demurrage - total_despatch:,.2f}")

if __name__ == "__main__":
    fleet = [
        BulkCarrier("Ore_Titan", arrival_hr=2.0, tonnage=180000, contract_rate_tpd=45000, demurrage_per_day=32000),
        BulkCarrier("Pacific_Pride", arrival_hr=4.0, tonnage=110000, contract_rate_tpd=35000, demurrage_per_day=24000),
        BulkCarrier("Cape_Venture", arrival_hr=12.0, tonnage=210000, contract_rate_tpd=50000, demurrage_per_day=36000),
        BulkCarrier("Iron_King", arrival_hr=26.0, tonnage=175000, contract_rate_tpd=40000, demurrage_per_day=30000)
    ]
    # Two export berths: Berth 1 (modern 7,500 t/h), Berth 2 (standard 5,500 t/h)
    berths = {1: 7500.0, 2: 5500.0}
    schedule_terminal(fleet, berths)
```

```text
Expected Output:
Vessel             Berth  Arr(h)  Start   End(h)  Lay(h)  Result     Amount ($)
--------------------------------------------------------------------------------
Ore_Titan          1      2.0     2.0     26.0    96.0    DESPATCH     48000.00
Pacific_Pride      2      4.0     4.0     24.0    75.4    DESPATCH     27714.29
Cape_Venture       1      12.0    27.5    55.5    100.8   DESPATCH     42975.00
Iron_King          2      26.0    26.0    57.8    105.0   DESPATCH     45738.64
--------------------------------------------------------------------------------
Net Port Liability: Demurrage=$0.00 | Despatch=$164,427.92 | Balance=$-164,427.92
```

---

## 8. Data, standards, and regulatory context

Pit-to-port supply chains operate under intersecting jurisdictions governing rail safety, pressurized pipelines, environmental dust emissions, and international maritime law.

### 8.1 Rail Infrastructure and Rolling Stock Standards

Heavy-haul networks must comply with standards promulgated by national rail safety regulators:
- **AREMA (American Railway Engineering and Maintenance-of-Way Association):** Chapter 4 (Rail), Chapter 16 (Economics of Railway Location and Operation), and Chapter 30 (Ties). AREMA specifies wheel-rail contact stress envelopes, subsurface ballast depths, and minimum structural track modulus ($u_{\text{track}} \ge 28\text{ N/mm}^2$ for 40-tonne axle loading).
- **AS 7635 / AS 7637 (Rail Industry Safety and Standards Board, Australia):** Defines track geometry tolerances, rail flaw ultrasonic testing frequencies, and braking stopping distances for automated train operations (e.g., Rio Tinto AutoHaul).
- **AAR (Association of American Railroads) Interchange Rules:** Establishes standards for Type E/F rotary couplers, draft gear shock attenuation, and dynamic distributed power wireless communication protocols.

### 8.2 Overland Conveyors and Pressure Piping Codes

- **DIN 22101 / ISO 5048 (Continuous Mechanical Handling Equipment):** Calculation of operating resistances, belt tension distributions, and driving forces for belt conveyors. Mandates safety factors on belt carcass tensile ratings (minimum dynamic safety factor $S_F \ge 5.0$ for steel-cord belts under transient acceleration and abortive braking).
- **ASME B31.11 / ASME B31.4 (Slurry and Liquid Transportation Piping Systems):** Governs design, material selection, ultrasonic inspection, wall thickness corrosion/erosion allowances, and hydrostatic pressure testing for long-distance slurry pipelines. Enforces minimum wall thickness calculated via Barlow's formula:

$$t_{\text{pipe}} = \frac{P_{\text{design}} \cdot D_{\text{outer}}}{2 \cdot S_{\text{allow}} \cdot E_{\text{joint}} \cdot F_{\text{design}}} + C_{\text{erosion}}$$

where $F_{\text{design}}$ is the design safety factor ($0.72$ for open terrain, dropping to $0.40\text{--}0.50$ at river crossings and populated infrastructure), and $C_{\text{erosion}}$ is sacrificial steel thickness allocated for slurry abrasion across design asset life.

### 8.3 Maritime Codes and Solid Bulk Cargo Safety

- **IMO IMSBC Code (International Maritime Solid Bulk Cargoes Code):** Categorizes bulk cargoes into three hazard classes:
  - **Group A:** Cargoes that may liquefy if shipped at a moisture content exceeding their Transportable Moisture Limit (TML). Iron ore fines, nickel laterite, and copper/zinc concentrates belong to Group A.
  - **Group B:** Cargoes possessing chemical hazards (e.g., self-heating coal emitting methane and carbon monoxide).
  - **Group C:** Cargoes neither liable to liquefy nor possess chemical hazards (e.g., dry lump iron ore, sized quartz).
  Shippers must legally certify cargo moisture, Flow Moisture Point (FMP), and TML using audited Proctor-Fagerberg or flow-table test procedures within 7 days prior to vessel loading.
- **IACS Unified Requirement S1A (Loading Manuals for Bulk Carriers):** Mandates that every vessel possesses an approved computer loading instrument calculating real-time still-water bending moments, shear forces, and local double-bottom panel buckling stresses during every step of hatch pouring.

### 8.4 Commercial Maritime Contracts and Laytime Rules

- **BIMCO (Baltic and International Maritime Council) Laytime Definitions (2013):** Standardizes contractual interpretations of Notice of Readiness (NOR), Weather Working Days (WWD), Sundays and Holidays Included (SHINC), and Strike Exclusions.
- **Port Authority Environmental Regulations:** Mandates boundary PM10 and PM2.5 particulate limits, requiring chemical dust suppressants (surfactants/polymers) applied at conveyor transfer chutes, car dumpers, and stockyard stockpiles, coupled with washdown water recycling systems.

---

## 9. Industry practice and case evidence

Operating pit-to-port networks at high throughput requires specialized logistics automation. The following public cases provide operational evidence.

### 9.1 Rio Tinto AutoHaul: Autonomous Heavy-Haul Rail (Pilbara, Western Australia)

Rio Tinto operates the world's largest automated heavy-haul freight network across the Pilbara region of Western Australia, hauling over 330 Mtpa of iron ore to ports at Dampier and Cape Lambert across 1,700 kilometers of heavy-haul track.

- **Operational Challenge:** Traditional operations required over 40 locomotive driver shift changeovers per day at remote siding locations. Driver changes introduced 20 to 45 minutes of idling delay per train, disrupting meet-pass schedules and inducing high fuel consumption variance due to inconsistent manual throttle and braking control.
- **Implemented Architecture:** The **AutoHaul** system implements GoA4 (Grade of Operation 4—unattended train operation). Rakes measuring 2.4 kilometers in length carrying 28,000 tonnes of ore are controlled autonomously via computer systems interfaced with European Train Control System (ETCS) Level 2 / digital signalling, satellite positioning, forward-looking obstacle-detection radar, and collision detection video analytics. Control is centralized 1,500 kilometers away at the Operations Centre in Perth.
- **Documented Results:** Eliminating crew changes removed over 1.2 million kilometers of annual light-vehicle road transit for crew transfers. AutoHaul achieved a 6% reduction in cycle time variance and a 3% improvement in overall network fuel efficiency through algorithmic throttle-brake speed profiling, freeing approximately three train paths per day across shared single-track bottlenecks without adding physical passing sidings.

### 9.2 Vale Carajás Logistics Corridor: Broad-Gauge Quad Dumping (Pará to Maranhão, Brazil)

Vale operates the Estrada de Ferro Carajás (EFC), an 892-kilometer railway linking the S11D and Carajás iron ore complexes in Pará state to the Ponta da Madeira Maritime Terminal in São Luís.

- **Network Characteristics:** EFC operates on 1,600 mm broad-gauge track carrying 37.5-tonne axle loads. Trains run in 330-wagon configurations (measuring 3.5 kilometers in length) with Gross Train Weight exceeding 42,000 tonnes, hauled by GE/Wabtec Evolution Series locomotives in Distributed Power mode.
- **Terminal Integration:** Ponta da Madeira handles the world's largest bulk carriers—400,000 DWT Valemax vessels. To match this loading rate, the terminal uses quad rotary car dumpers capable of overturning four 110-tonne wagons simultaneously (discharging 440 tonnes every 90 seconds). This feeds dedicated stockyard reclaim lines feeding twin shiploaders operating at combined sustained rates exceeding 16,000 tonnes per hour.
- **Engineering Reality:** At 42,000 tonnes GTW, in-train longitudinal dynamic forces (slack run-in and run-out) exceed 2,000 kN. Vale uses electronically controlled pneumatic (ECP) braking across all 330 wagons, applying brake shoes simultaneously along the 3.5-kilometer train length within 2 seconds (compared to 30 to 45 seconds for conventional pneumatic brake pipes), preventing wagon derailment caused by in-train compression shock waves.

### 9.3 Anglo American Minas-Rio: Long-Distance Slurry Pipeline (Minas Gerais to Rio de Janeiro, Brazil)

The Minas-Rio integrated iron ore complex transports pellet-feed magnetite concentrate from an open-pit mine and beneficiation plant in Conceição do Mato Dentro to the offshore port of Açu across a 529-kilometer buried slurry pipeline.

- **Physical Corridor:** The 24-to-26-inch pipeline traverses 32 municipalities, dropping from an inland elevation of 800 meters above sea level to coastal tidewater. It incorporates two high-pressure pump stations equipped with multiple quintuplex positive-displacement piston-diaphragm pumps, along with a coastal valve and choke station.
- **Hydraulic Management:** The system moves approximately 24 Mtpa of high-grade magnetite slurry at $C_w \approx 67\%$. Because the pipeline traverses steep descents in the Serra dos Aimorés mountain range, pressure monitoring systems continuously adjust terminal choke orifices to eliminate slack flow cavitation. Internal acoustic monitoring arrays and mass-balance computational systems detect pressure drops within seconds to isolate sections via automated emergency block valves in the event of pipeline breach.

### 9.4 Richards Bay Coal Terminal (RBCT, South Africa)

RBCT is one of the world's largest export coal terminals, with an installed design capacity of 91 Mtpa, fed by Transnet Freight Rail's 580-kilometer Coal Line from Mpumalanga.

- **Operational Dynamic:** Transnet operates 200-wagon coal rakes on 1,067 mm (Cape gauge) track at 26-tonne axle loads, operating dual-voltage (25 kV AC / 3 kV DC) electric locomotives.
- **Logistical Vulnerability:** RBCT demonstrates the operational vulnerability of captive pit-to-port corridors to upstream infrastructure deterioration. Cable theft, locomotive spare parts shortages, and track maintenance backlogs reduced rail delivery rates from a peak of 76 Mtpa to below 50 Mtpa between 2021 and 2023. The mismatch forced inland coal producers to deploy thousands of road haulage trucks to bypass rail, increasing road transport costs from $12/t to over $45/t, accelerating road degradation, and creating truck traffic queues at port entry gates.

---

## 10. Failure modes and red flags

The table below outlines common failure modes across pit-to-port logistics networks, detailing early warning signs, underlying mechanisms, consequences, and mitigations.

| Failure Mode | Early Warning Sign | Underlying Physical / Logistical Mechanism | Operational & Financial Consequence | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Slurry Pipeline Sanding** | Gradual increase in pumping discharge pressure coupled with localized acoustic sensor hiss. | Slurry flow velocity drops below critical deposition velocity ($V < V_c$), causing coarse particles to settle and form a stationary dune bed on pipe invert. | Catastrophic pipeline plugging; line must be depressurized, located, physically cut, and flushed, causing multi-week shutdowns ($>\$50\text{M}$ loss). | Install automated variable-frequency pump loops; monitor minimum volumetric velocity; maintain emergency high-pressure flush water reservoirs. |
| **Slack Flow Cavitation** | Sudden pressure oscillations downstream of mountain crests; audible knocking along pipeline. | Hydraulic Grade Line drops below pipe elevation profile ($P < P_{\text{vapor}}$), generating vacuum voids that collapse destructively upon rejoining. | Internal pipe lining detachment, metal fatigue, column-separation water-hammer bursts during transient pump trips. | Install automated variable orifice choke stations and vacuum-breaker air valves at hydraulic crests; maintain positive backpressure. |
| **Train Coupler Shearing** | In-train telemetry records sudden spike in longitudinal draft force ($>1,800\text{ kN}$). | Unsynchronized throttle or braking transitions across Distributed Power (DP) locomotive units induce violent slack run-out. | Train break-in-two on mainline; halts corridor traffic for 12 to 36 hours; damages wagons and track switches. | Deploy synchronous ECP braking; implement predictive DP throttle sequencing algorithms; enforce strict coupler fatigue ultrasonic testing. |
| **Rail Gauge Corner Spalling** | Fine surface cracks on outer rail head observed during ultrasonic car inspection; track geometry car records gauge spread. | Excessive contact stress from 40t axle loads exceeding rail steel shear yield strength, causing Rolling Contact Fatigue (RCF). | Sudden catastrophic rail break under train passage, leading to high-speed loaded train derailment. | Implement cyclical rail grinding profiles to eliminate micro-cracks; apply top-of-rail friction modifiers; upgrade to hypereutectoid heat-treated steel. |
| **Conveyor Belt Longitudinal Rip**| Localized motor current spike on carrying idlers; tension imbalance between drive pulleys. | Oversize tramp iron (e.g., failed loader tooth) wedges in transfer chute, piercing moving belt carcass and slitting belt longitudinally. | Destruction of multi-kilometer steel-cord belt flight; replacement lead time 4 to 8 months; logistics corridor halted. | Install electromagnetic tramp metal cross-belt separators; deploy continuous embedded inductive loop rip-detection systems. |
| **Regenerative Braking Inversion**| Electrical substation DC bus overvoltage alarms; motor drive fault code "Overspeed Trip". | Downhill conveyor load exceeds kinetic braking threshold during emergency abortive stop, saturating dynamic braking resistor banks. | Runaway downhill conveyor; mechanical disc brakes glaze and burn out; high-speed belt ejection and structural collapse. | Install fail-safe spring-applied hydraulic multi-disc brakes on low-speed pulleys; size dynamic braking resistor banks with $2.5\times$ thermal safety margin. |
| **Rotary Dumper Car Derailment** | Positioner hydraulic pressure exceeds relief setpoint; indexer arm laser distance measurement disagrees with mechanical stop. | Slack action between wagon couplers causes wagon wheelset to ride up on rail lead-in check rail during positioner indexing stroke. | Dumper barrel jammed; structural damage to indexer arm; terminal receiving capacity halted for 48 to 96 hours. | Implement closed-loop wheel proximity sensing; install laser positioner arm profiling; replace mechanical stops with hydraulic retardation buffers. |
| **Stacker-Reclaimer Collision** | Machine proximity warning siren activates; PLC safety relay trip on slew drive. | GPS multipath reflection off metal stockyard structures causes machine position drift; blind spot in operator visual field. | Structural boom collapse, bucket wheel shearing; loss of reclaiming capacity for 6 to 12 months ($>\$20\text{M}$ replacement). | Mandate triple-redundant CAS: dual-station RTK-GPS fused with 3D LiDAR spatial point-clouds and hardwired fail-safe mechanical limit interlocks. |
| **Vessel Cargo Liquefaction** | Moisture assay of shiploader cross-stream sample approaches Flow Moisture Point ($w > \text{TML}$). | Heavy rain on open stockyard windrows raises concentrate/fines moisture above safe maritime stability limit. | Cargo liquefies under ship vibration during ocean transit, shifting cargo, causing vessel instability and potential capsizing. | Construct covered stockyard storage sheds; reject cargo above TML; enforce automated inline continuous moisture analyzer interlocks. |
| **Hull Bending Moment Exceedance**| Vessel loading computer stress alarm sounds; ship master halts shiploader feed. | Shiploader operator deviates from IACS 2-pass hatch plan, placing too much tonnage in midship hold (H5) during initial pass. | Permanent plastic deformation of ship hull girder; cracked deck plates; immediate port state detention and repair litigation. | Automate shiploader hatch traverse programming via PLC recipe control; cross-interlock terminal DCS with ship stability computer. |
| **Stockyard Pad Stranding** | Reclaimer bucket wheel runs empty before vessel hold reaches certified bill of lading tonnage. | Mass-balance inventory drift caused by uncalibrated belt scales, bulk density shifts, and unrecorded stacker drop losses. | Vessel starved at berth; laytime expires while waiting for emergency train dispatch; severe demurrage incurred. | Implement continuous LiDAR volumetric drone surveying of stockpiles; calibrate belt weightometers weekly to ISO 9870 standards. |
| **Demurrage Cascade Trap** | Anchorage vessel queue increases from normal 5 vessels to $>25$ vessels; average waiting time $>14\text{ days}$. | Inland rail or processing throughput falls $10\%$ below target for 3 consecutive weeks, while vessel stemming notices remain unadjusted. | Terminal pays millions of dollars in monthly demurrage; shipowners refuse to charter vessels to the port without exorbitant freight premiums. | Implement dynamic vessel de-stemming protocols; throttle vessel arrival stems in response to inland production variance; use virtual arrival speed adjustments. |

---

## 11. Assumptions and limits

Logistical optimization models rely on physical and operational assumptions that bound their validity. The following ledger states these constraints explicitly.

| Assumption | Operational Rationale | Boundary Condition / Failure Point | Diagnostic Test |
| :--- | :--- | :--- | :--- |
| **Constant Slurry Rheology** | Slurry hydraulic models assume yield stress $\tau_y$ and plastic viscosity $\eta_p$ remain constant along the pipeline flight. | Severe particle size grinding excursions or clay contamination (e.g., smectite) exponentially increase yield stress, doubling friction loss. | Continuous inline capillary or rotational rheometer sampling at pump station intake manifolds. |
| **Deterministic Train Cycle Time** | Rail fleet planning models frequently assume mean cycle times $\bar{T}_{\text{cycle}}$ represent operational reality. | High network utilization ($>85\%$ of track capacity) induces exponential queueing delays at passing sidings, expanding cycle time variance. | Track siding dwell distribution analysis; check if variance-to-mean ratio ($VMR = \sigma^2 / \mu$) exceeds 0.25. |
| **Zero In-Train Coupler Slack** | Longitudinal train dynamics models often treat rakes as rigid bodies when calculating tractive effort and braking distance. | In long trains ($>2.0\text{ km}$), cumulative draft gear slack ($2.0\text{--}4.0\text{ m}$) induces dynamic shock waves that decouple train sections. | High-frequency accelerometers installed on lead, mid-train, and tail locomotive units to record draft shock waves. |
| **Uniform Idler Friction ($f$)** | Overland conveyor motor sizing assumes an invariant DIN 22101 artificial friction factor $f$ across the entire flight. | Ambient temperatures below $0^{\circ}\text{C}$ stiffen rubber compound backing, increasing rolling resistance by $30\text{--}60\%$, overloading drives. | Temperature-dependent dynamic mechanical analysis (DMA) testing of conveyor belt bottom cover rubber compound. |
| **Homogeneous Stockpile Blending** | Stacking models assume Chevron/Windrow layers blend perfectly according to the $\sigma^2 / N_{\text{layers}}$ relationship. | Particle rolling segregation causes coarse lump ore to accumulate at pile toes while fine particles concentrate at pile apex. | Cross-stream automated mechanical sampling of reclaimer belt discharge analyzed by sizing sieve screen arrays. |
| **Unconstrained Deballasting** | Shiploader scheduling assumes vessel loading can proceed at maximum continuous shiploader nameplate capacity. | Vessel ballast pumps cannot match high-capacity shiploaders ($8,000\text{--}12,000\text{ t/h}$), forcing shiploader downtime to prevent hull overstress. | Review ship deballasting capacity declaration prior to vessel berthing; cross-check against terminal pour plan. |
| **Continuous Weather Working Days** | Berth throughput planning assumes laytime accrues continuously under standard maritime definitions. | Local climatic shifts (monsoons, tropical cyclones, sea swell exceeding $1.5\text{ m}$) legally suspend the contractual laytime clock under WWD terms. | Statistical review of historical 10-year port wave-rider buoy data and meteorological rainfall downtime logs. |
| **Static Track Modulus ($u_{\text{track}}$)**| Rail bending stress calculations assume uniform track bed elasticity along hundreds of route kilometers. | Subgrade saturation during wet seasons drops track modulus by $>50\%$, accelerating rail fatigue and subgrade pumping failures. | Automated track recording vehicle measuring dynamic rail vertical deflection under full 40-tonne wheel load. |
| **Infinite Surge Bin Dampening** | Train loadout and car dumper scheduling models assume surge bins absorb discrete-to-continuous transfer shocks. | Surge bins hold only 15 to 45 minutes of run time; minor equipment trips immediately propagate across connected systems. | Dynamic discrete-event simulation (DES) modeling bin fill levels under stochastic equipment failure distributions. |
| **Negligible Bulk Cargo Degradation**| Quality models assume particle size distribution remains unchanged between mine loadout and vessel hatch. | Multiple high-drop transfers (chutes, stacker boom drops, car dumper hoppers) degrade lump ore, generating $2\text{--}5\%$ additional fines. | Drop-tower shatter testing and Tumbler Index (TI) testing according to ISO 3271 standards. |

---

## 12. Exercises

### Exercise 1: Heavy-Haul Rail Fleet Sizing and Passing Loop Headway (Calculation)
A heavy-haul bauxite railway must transport $Q_{\text{annual}} = 36.0\text{ Mtpa}$ (dry) from an inland mine to a coastal alumina refinery. Free moisture is $\mu = 8.0\%$. Mainline distance is $280\text{ km}$ (single-track). Axle load limit is $35.0\text{ t/axle}$ on 4-axle wagons. Wagon tare mass is $22.0\text{ t}$. Trains run in 180-wagon configurations. Loading rate is $6,000\text{ t/h}$. Loaded speed is $60\text{ km/h}$, empty speed is $75\text{ km/h}$. Dumper operates at $2\text{ wagons/dump}$ with a $95\text{ s}$ cycle. Loaded siding delays average $1.5\text{ h}$, empty siding delays $1.0\text{ h}$, port dump wait $1.2\text{ h}$, and inland dwell $2.0\text{ h}$. Network operating availability is $85\%$.
- **Deliverable:** Calculate net wagon payload, total gross train weight (excluding locomotives), round-trip cycle time ($T_{\text{cycle}}$), annual trips required, and minimum number of active train rakes required.
- **Acceptance Criteria:** Correct arithmetic steps; final active rake count rounded up to integer; cycle time calculated to two decimal places.
- **Estimated Time:** 30 minutes.

### Exercise 2: Single-Track Meet-Pass Conflict Resolution (Application)
Two loaded heavy-haul trains (T101, T103) travel south from Mine to Port. Two empty trains (T202, T204) travel north from Port to Mine. A single-track segment connects Siding Alpha (km 50) and Siding Beta (km 95). Block distance is $45\text{ km}$. Running speeds are $60\text{ km/h}$ for loaded trains and $75\text{ km/h}$ for empty trains. Deceleration to enter a siding takes $4\text{ minutes}$; acceleration to exit takes $5\text{ minutes}$. Minimum safety headway between trains moving in the same direction is $12\text{ minutes}$. Clearing headway for opposing movements is $6\text{ minutes}$. T101 departs Siding Alpha at 08:00. T202 arrives at Siding Beta at 08:15.
- **Deliverable:** Construct a time-space conflict diagram (or tabular schedule); determine whether T101 and T202 should meet at Siding Alpha or Siding Beta to minimize total combined delay; calculate exact arrival, departure, and dwell times for both trains.
- **Acceptance Criteria:** Explicit mathematical validation that track segment occupancy never overlaps; proper inclusion of headway and acceleration/deceleration penalties.
- **Estimated Time:** 45 minutes.

### Exercise 3: Slurry Pipeline Critical Velocity and Head Loss (Calculation)
An iron ore concentrate slurry pipeline transports $12.0\text{ Mtpa}$ (dry) over $180\text{ km}$. Specific gravity of solids is $S_s = 4.80$. Solids concentration by weight is $C_w = 65\%$. Median particle diameter is $d_{50} = 45\ \mu\text{m}$. Pipe internal diameter is $D_{\text{pipe}} = 0.500\text{ m}$. Durand factor $F_L = 1.15$. Annual operating availability is $8,200\text{ hours}$. Darcy slurry friction factor is $f_m = 0.0175$. Elevation drop is $150\text{ m}$ from mine to port ($\Delta z = -150\text{ m}$).
- **Deliverable:** Compute volumetric concentration $C_v$, mixture density $\rho_m$, critical deposition velocity $V_c$, actual operational velocity $V_{\text{actual}}$, verify hydraulic stability ($V_{\text{actual}} \ge 1.15 \cdot V_c$), total friction head loss ($\text{m}$), and net pump discharge pressure required ($\text{MPa}$).
- **Acceptance Criteria:** Correct formulas applied; explicit step-by-step arithmetic; friction and static elevation head properly summed.
- **Estimated Time:** 35 minutes.

### Exercise 4: Overland Downhill Conveyor Regenerative Power Balance (Calculation)
A downhill overland conveyor spans $L = 6.5\text{ km}$ with a net vertical drop of $H = -220\text{ m}$. Design mass throughput is $Q = 4,500\text{ t/h}$. Belt speed is $v_{\text{belt}} = 5.5\text{ m/s}$. DIN artificial friction factor is $f = 0.014$. Mass of rotating idler parts is $m'_R = 38\text{ kg/m}$. Mass of belt carcass and covers is $m'_B = 42\text{ kg/m}$. Combined drive and gearbox mechanical efficiency is $\eta = 92\%$.
- **Deliverable:** Compute mass of conveyed bulk material per meter ($m'_G$), main resistance force ($F_H$), lift resistance force ($F_{St}$), net motor shaft power ($P_{\text{shaft}}$), determine whether the system is regenerative, and calculate net electrical power generated ($\text{kW}$) fed back into the electrical grid.
- **Acceptance Criteria:** Correct application of DIN 22101 power balance; negative signs properly interpreted for regenerative generation.
- **Estimated Time:** 30 minutes.

### Exercise 5: Rotary Car Dumper and Stockyard Surge Sizing (Application)
A marine bulk export terminal receives ore from heavy-haul trains. The tandem rotary car dumper has a mechanical cycle time of $85\text{ seconds}$ per dump (2 wagons of $125\text{ t}$ payload each). Train rakes consist of 200 wagons. Train swap time at the dumper dumper is $25\text{ minutes}$. The dumper discharges into a 2,500-tonne surge bin beneath the hopper. The surge bin feeds a stockyard stacking conveyor operating at a maximum capacity of $7,500\text{ t/h}$.
- **Deliverable:** Compute the instantaneous dumping rate during active dumping, total net time to unload a 200-car train (including swap time), effective hourly train unloading throughput, and calculate whether the 2,500-tonne surge bin will overflow during active train dumping.
- **Acceptance Criteria:** Mathematical mass balance showing surge bin accumulation rate; clear determination of whether conveyor take-away rate constrains dumper operation.
- **Estimated Time:** 30 minutes.

### Exercise 6: Shiploader Hatch Pouring Plan and Hull Stress Critique (Critique)
A terminal operations supervisor drafts a loading plan for a 170,000 DWT Capesize bulk carrier with 9 holds. To minimize shiploader travel time along the wharf, the supervisor proposes loading each hold to its final target capacity in a single continuous pass, starting at Hold 1 and progressing sequentially to Hold 9.
- **Deliverable:** Critique this proposed loading plan from naval architectural, structural integrity, and commercial perspectives. Detail the exact failure mechanisms under IACS Unified Requirement S1A, explain the interaction with vessel deballasting rates, and outline the correct multi-pass pour structure.
- **Acceptance Criteria:** Identification of sagging and hogging bending moments, shear stress limits, deballasting bottlenecks, and draft/trim stability control.
- **Estimated Time:** 40 minutes.

### Exercise 7: Terminal Demurrage and Berth Allocation Optimization (Design)
Design an integrated operational protocol and priority dispatching rule for a marine bulk terminal operating 3 berths with heterogeneous shiploader rates ($10,000\text{ t/h}$, $7,500\text{ t/h}$, $5,000\text{ t/h}$) serving Capesize, Panamax, and Handymax vessels. The protocol must resolve trade-offs between minimizing immediate vessel demurrage penalties, avoiding stockyard product pad starvation, and managing deep-draft tidal window departures.
- **Deliverable:** Provide a formal algorithmic flowchart (or step-by-step decision rules), define objective function components, and specify real-time data inputs required from rail, stockyard, and marine radar systems.
- **Acceptance Criteria:** Comprehensive design addressing the interaction between inland train delivery schedules, stockyard reclaim line availability, and maritime laytime contracts.
- **Estimated Time:** 45 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Heavy-Haul Rail Fleet Sizing
1. **Wagon Net Payload:**
   - Gross weight limit per wagon: $4\text{ axles} \times 35.0\text{ t/axle} = 140.0\text{ t}$.
   - Net payload per wagon: $w_{\text{payload}} = 140.0\text{ t} - 22.0\text{ t (tare)} = 118.0\text{ t}$.
2. **Train Rake Payload and Gross Weight:**
   - Payload per rake: $W_{\text{rake\_payload}} = 180 \times 118.0\text{ t} = 21,240\text{ wet tonnes}$.
   - Gross weight per rake: $W_{\text{rake\_gross}} = 180 \times 140.0\text{ t} = 25,200\text{ t}$.
3. **Cycle Time Components ($T_{\text{cycle}}$):**
   - Annual wet tonnage: $Q_{\text{wet}} = 36.0 / (1 - 0.08) = 39.1304\text{ Mtpa}$.
   - Loading time: $T_{\text{load}} = 21,240\text{ t} / 6,000\text{ t/h} = 3.54\text{ h}$.
   - Loaded transit: $T_{\text{transit\_loaded}} = 280\text{ km} / 60\text{ km/h} = 4.67\text{ h}$.
   - Loaded siding delay: $1.50\text{ h}$.
   - Port dump wait: $1.20\text{ h}$.
   - Dumping time: 180 wagons $/ 2 = 90\text{ cycles} \times 95\text{ s} = 8,550\text{ s} = 2.375\text{ h} \approx 2.38\text{ h}$.
   - Empty transit: $T_{\text{transit\_empty}} = 280\text{ km} / 75\text{ km/h} = 3.73\text{ h}$.
   - Empty siding delay: $1.00\text{ h}$.
   - Inland dwell: $2.00\text{ h}$.
   - Total cycle time: $T_{\text{cycle}} = 3.54 + 4.67 + 1.50 + 1.20 + 2.38 + 3.73 + 1.00 + 2.00 = 22.02\text{ hours}$.
4. **Annual Trips and Fleet Size:**
   - Annual trips: $N_{\text{trips}} = 39,130,435\text{ t} / 21,240\text{ t/trip} = 1,842.3 \implies 1,843\text{ trips/year}$.
   - Available annual hours: $8,760 \times 0.85 = 7,446\text{ operating hours/year}$.
   - Total fleet hours required: $1,843 \times 22.02\text{ h} = 40,582.86\text{ rake-hours}$.
   - Active rakes required: $40,582.86 / 7,446 = 5.45 \implies \mathbf{6\text{ active rakes}}$ (plus 1 maintenance spare $= 7$ total rakes).

### Answer to Exercise 2: Single-Track Meet-Pass Conflict Resolution
1. **Unconstrained Running Times across the 45 km Block:**
   - Loaded train T101 ($60\text{ km/h}$): $45 / 60 = 0.75\text{ h} = 45\text{ minutes}$.
   - Empty train T202 ($75\text{ km/h}$): $45 / 75 = 0.60\text{ h} = 36\text{ minutes}$.
2. **Option A: Meet at Siding Alpha (km 50):**
   - T101 is held at Siding Alpha. T202 departs Siding Beta at 08:15.
   - T202 transit time: 36 min $+ 4\text{ min decel} = 40\text{ min}$. T202 arrives at Siding Alpha at 08:55.
   - T101 can depart Siding Alpha after T202 clears plus safety headway ($6\text{ min}$) and acceleration ($5\text{ min}$): Departure at $08:55 + 11\text{ min} = 09:06$.
   - Delay to T101 (scheduled 08:00): $66\text{ minutes}$ delay. Delay to T202: $0\text{ minutes}$. Total delay: $66\text{ minutes}$.
3. **Option B: Meet at Siding Beta (km 95):**
   - T101 departs Siding Alpha at 08:00. Transit time: 45 min $+ 4\text{ min decel} = 49\text{ min}$. T101 arrives at Siding Beta at 08:49.
   - T202 is held at Siding Beta from 08:15 until T101 arrives (08:49) $+ 6\text{ min headway} + 5\text{ min accel} = 09:00$.
   - Delay to T101: $0\text{ minutes}$. Delay to T202 (held from 08:15 to 09:00): $45\text{ minutes}$.
4. **Optimal Dispatch Decision:**
   - **Option B (Meet at Siding Beta)** is optimal. It minimizes total network delay ($45\text{ min}$ vs $66\text{ min}$) and prioritizes the high-momentum loaded train (T101), saving fuel and avoiding adverse grade deceleration.

### Answer to Exercise 3: Slurry Pipeline Critical Velocity and Head Loss
1. **Volumetric Concentration ($C_v$):**
   $$C_v = \frac{0.65}{4.80 - 0.65 \times (4.80 - 1.0)} = \frac{0.65}{4.80 - 2.47} = \frac{0.65}{2.33} = 0.27897 \quad (27.90\%)$$
2. **Mixture Density ($\rho_m$):**
   $$\rho_m = 1,000 \times [1 + 0.27897 \times (4.80 - 1.0)] = 1,000 \times [1 + 1.0601] = 2,060.1\text{ kg/m}^3$$
3. **Critical Velocity ($V_c$):**
   $$V_c = 1.15 \cdot \sqrt{2 \times 9.81 \times 0.500 \times \left(\frac{4.80 - 1.0}{1.0}\right)} = 1.15 \cdot \sqrt{9.81 \times 3.80} = 1.15 \cdot \sqrt{37.278} = 1.15 \times 6.1056 = 7.021\text{ m/s ???}$$
   *Re-evaluating units:* Durand equation requires submerged specific gravity: $(S_s - S_L)/S_L = 3.80$. $2 \times 9.81 \times 0.50 = 9.81$. $9.81 \times 3.80 = 37.278$. $\sqrt{37.278} = 6.1056$. However, Durand factor $F_L$ for fine slurries ($45\ \mu\text{m}$) is standardly applied as $F_L \approx 0.25\text{--}0.35$ in classic Durand charts (or $1.15$ when using the Wasp-modified $F_L$ based on particle Froude number). Using the stated $F_L = 1.15$ yields $V_c = 7.02\text{ m/s}$ if uncorrected, but for fine concentrates, standard industry $F_L' = F_L \times \sqrt{d_{50}/D} \approx 0.28$, yielding $V_c \approx 1.71\text{ m/s}$. Applying the exercise's specific direct inputs:
   $$\text{Mass flow rate: } \dot{M}_s = \frac{12,000,000\text{ t}}{8,200\text{ h}} \times \frac{1,000}{3,600} = 406.50\text{ kg/s}$$
   $$\dot{M}_m = \frac{406.50}{0.65} = 625.39\text{ kg/s}$$
   $$\dot{Q}_m = \frac{625.39}{2,060.1} = 0.30357\text{ m}^3\text{/s}$$
   $$A_{\text{pipe}} = \frac{\pi \times (0.500)^2}{4} = 0.19635\text{ m}^2$$
   $$V_{\text{actual}} = \frac{0.30357}{0.19635} = 1.546\text{ m/s}$$
4. **Friction Loss and Pressure:**
   $$i_m = 0.0175 \times \frac{(1.546)^2}{2 \times 9.81 \times 0.500} = 0.0175 \times \frac{2.390}{9.81} = 0.0175 \times 0.2436 = 0.004263\text{ m/m}$$
   $$\text{Friction head: } \Delta h_f = 0.004263 \times 180,000\text{ m} = 767.34\text{ m}$$
   $$\text{Net Head Required: } H_{\text{net}} = \Delta h_f + \Delta z = 767.34\text{ m} + (-150.0\text{ m}) = 617.34\text{ m of slurry}$$
   $$\text{Net Pump Pressure: } P_{\text{pump}} = \rho_m \cdot g \cdot H_{\text{net}} = 2,060.1 \times 9.81 \times 617.34 = 12,476,000\text{ Pa} = \mathbf{12.48\text{ MPa}} \quad (124.8\text{ bar})$$

### Answer to Exercise 4: Overland Downhill Conveyor Regenerative Power Balance
1. **Bulk Material Mass per Meter ($m'_G$):**
   $$m'_G = \frac{4,500\text{ t/h} \times 1,000\text{ kg/t}}{3600\text{ s/h} \times 5.5\text{ m/s}} = \frac{1,250\text{ kg/s}}{5.5\text{ m/s}} = 227.27\text{ kg/m}$$
2. **Main Resistance Force ($F_H$):**
   $$\cos(\delta) = \cos(\arcsin(-220 / 6500)) = \cos(-1.939^{\circ}) \approx 0.9994$$
   $$F_H = 0.014 \times 6,500\text{ m} \times 9.81 \times [38 + (2 \times 42 + 227.27) \times 0.9994]$$
   $$F_H = 892.71 \times [38 + 311.27 \times 0.9994] = 892.71 \times [38 + 311.08] = 892.71 \times 349.08 = 311,627\text{ N} = 311.63\text{ kN}$$
3. **Lift Resistance Force ($F_{St}$):**
   $$F_{St} = m'_G \cdot g \cdot H = 227.27\text{ kg/m} \times 9.81\text{ m/s}^2 \times (-220\text{ m}) = -490,495\text{ N} = -490.50\text{ kN}$$
4. **Net Motor Shaft Power ($P_{\text{shaft}}$):**
   $$F_U = F_H + F_{St} = 311.63\text{ kN} - 490.50\text{ kN} = -178.87\text{ kN}$$
   Since $F_U < 0$, the system is **regenerative**.
   $$P_{\text{shaft}} = \frac{-178.87\text{ kN} \times 5.5\text{ m/s}}{1} = -983.78\text{ kW}$$
5. **Electrical Power Generated to Grid:**
   $$P_{\text{regen}} = |P_{\text{shaft}}| \cdot \eta_{\text{drive}} = 983.78\text{ kW} \times 0.92 = \mathbf{905.08\text{ kW}}$$
   The conveyor continuously delivers **$905.1\text{ kW}$** of electrical power back into the mining electrical distribution grid.

### Answer to Exercise 5: Rotary Car Dumper and Stockyard Surge Sizing
1. **Instantaneous Dumping Rate:**
   $$Q_{\text{inst}} = \frac{2 \times 125\text{ t}}{85\text{ s}} \times 3,600\text{ s/h} = 10,588.24\text{ t/h}$$
2. **Train Unloading Duration:**
   $$N_{\text{cycles}} = \frac{200}{2} = 100\text{ cycles} \times 85\text{ s} = 8,500\text{ s} = 141.67\text{ minutes} = 2.361\text{ hours}$$
   $$\text{Total Time per Train} = 141.67\text{ min} + 25\text{ min (swap)} = 166.67\text{ minutes} = 2.778\text{ hours}$$
   $$\text{Effective Hourly Throughput} = \frac{200 \times 125\text{ t}}{2.778\text{ h}} = \frac{25,000\text{ t}}{2.778\text{ h}} = \mathbf{9,000.0\text{ t/h}}$$
3. **Surge Bin Mass Balance:**
   - Inflow rate during active dumping: $Q_{\text{in}} = 10,588.24\text{ t/h}$.
   - Outflow rate via take-away conveyor: $Q_{\text{out}} = 7,500.00\text{ t/h}$.
   - Net accumulation rate in surge bin:
     $$\Delta \dot{M} = 10,588.24 - 7,500.00 = 3,088.24\text{ t/h} = 0.8578\text{ tonnes/second}$$
   - Active dumping duration per 85-second cycle is approximately $12\text{ seconds}$ of mass drop into hopper, but continuous cycling over 100 cycles represents an extended accumulation phase:
     $$\text{Total active dumping time} = 2.361\text{ hours}$$
     $$\text{Theoretical accumulated mass} = 3,088.24\text{ t/h} \times 2.361\text{ h} = 7,291.3\text{ tonnes}$$
   - **Conclusion:** A 2,500-tonne surge bin **will overflow**. The bin can absorb only:
     $$t_{\text{overflow}} = \frac{2,500\text{ t}}{3,088.24\text{ t/h}} = 0.8095\text{ hours} = 48.57\text{ minutes}$$
   - **Operational Remedy:** The dumper cannot operate continuously at $85\text{ s/cycle}$ unless the takeaway conveyor is expanded to $\ge 10,600\text{ t/h}$, or automated dumper cycle throttling (introducing a 35-second dwell between dumps) is enforced to match the $7,500\text{ t/h}$ takeaway limit.

### Rubric for Exercise 6: Shiploader Hatch Pouring Plan Critique
A strong critique addresses four core domains:
1. **Naval Architecture & IACS S1A Violation:** Single-pass loading causes extreme longitudinal stress imbalance. When Holds 1, 2, and 3 are loaded while Holds 7, 8, and 9 are empty, the ship experiences severe hogging. As midship holds (Hold 5) are filled to capacity without end hold balance, excessive sagging bending moments occur, exceeding maximum permissible still-water bending moments ($BM_{\text{max}}$) and shear forces ($SF_{\text{max}}$), risking structural fracture of deck plates or double-bottom buckling.
2. **Deballasting Synchronization Bottleneck:** Capesize vessels carry 50,000 to 60,000 tonnes of water ballast. Ship deballasting pumps discharge at 3,000 to 4,000 t/h. A modern shiploader operating at 8,000 t/h in a single pass will outrun the vessel's deballasting rate, forcing the ship to exceed draft limits prematurely, inducing excessive forward or aft trim, and requiring emergency stops.
3. **Mooring and Gangway Safety:** Rapid concentrated loading in single holds induces abrupt list and trim changes, placing dangerous shock loads on mooring lines and winches, risking line snap-back.
4. **Correct Multi-Pass Structure:** Detail a standard 2-pass (or 3-pass) sequence. Pass 1 fills 50% to 65% of each hold's capacity across alternating odd/even holds (e.g., H1, H3, H5, H7, H9, followed by H2, H4, H6, H8), permitting continuous deballasting. Pass 2 tops up holds to final cargo distribution, with the final 10% reserved for Hold 4 or 6 as the "trimming hold" to achieve certified even-keel departure draft.

### Rubric for Exercise 7: Terminal Demurrage and Berth Allocation Optimization
A complete system design must include:
1. **Decision Rules / Objective Formulation:** Clear mathematical objective function minimizing weighted turnaround time, demurrage penalties, and product blending shortfalls.
2. **Multi-Berth Matching Logic:** Assign deep-draft Capesize vessels to high-rate berths (Berth 1: 10,000 t/h) with deep dredge pockets; assign Panamax/Handymax to lower-rate berths (Berth 2 & 3).
3. **Tidal Window Synchronization:** Deep-draft vessels must complete loading with sufficient lead time to unmoor and transit the port access channel during high-tide clearance windows.
4. **Inland Stockyard Integration:** Real-time interlock ensuring a vessel is only berthed if $\ge 80\%$ of its dedicated cargo blend is physically assembled on yard pads, with the remaining $20\%$ confirmed on inbound mainline trains within 6 hours of dumper arrival.
5. **Dynamic De-stemming & Virtual Arrival:** Protocol allowing the terminal to notify incoming vessels 72 hours prior to arrival to reduce sea steaming speed (saving bunker fuel) when terminal bottlenecks would force them into anchorage demurrage queues.

---

## 14. Interview questions

### Question 1: How does Distributed Power (DP) technology fundamentally alter the throughput capacity and physical constraints of a heavy-haul bulk railway?
- **What a strong answer contains:** Explains that in trains exceeding 1.5 to 2.0 kilometers in length with gross weights above 25,000 tonnes, concentrating all locomotive traction at the head generates draft forces that exceed the tensile yield limit of standard AAR Type E/F couplers ($>1,800\text{--}2,000\text{ kN}$), risking knuckle failure. Distributing locomotive units (e.g., 2 locomotives at head, 1 mid-train, 1 at tail) divides in-train tensile and compressive forces, reduces coupler strain, and enables rakes of 240 to 330 cars. Additionally details pneumatic brake propagation: radio-controlled DP units initiate brake pipe pressure reductions simultaneously from multiple points along the train, reducing pneumatic propagation time from 40 seconds to under 8 seconds. This shortens emergency stopping distances, allows tighter signal headway blocks, and increases single-track corridor capacity.
- **What a weak answer reveals:** Focuses solely on "adding more horsepower" to go faster, missing coupler stress physics, pneumatic brake propagation dynamics, and headway block compression.

### Question 2: In mineral concentrate slurry pipelines, what physical mechanisms dictate the boundaries of the safe operating velocity window, and what happens if you violate them?
- **What a strong answer contains:** Identifies the lower boundary as the critical deposition velocity ($V_c$), derived via Durand or Wasp models, where fluid drag and turbulent eddies can no longer overcome gravitational settling of the median and top particle fractions ($d_{50}, d_{95}$). Operating below $1.15 \cdot V_c$ causes dune formation, pipe invert sanding, and catastrophic blockage requiring mechanical pipe excavation. Identifies the upper boundary (typically $2.2\text{--}2.5\text{ m/s}$) governed by abrasive slurry erosion, where particle kinetic energy removes the internal polyurethane or steel sacrificial wear liner, accelerating wear proportionally to $V^{2.5\text{--}3.0}$. Notes that hydraulic gradient and pump power escalate with $V^2$, making high velocities economically non-viable.
- **What a weak answer reveals:** Mentions only that "slurry might get stuck if it moves too slow," lacking quantitative definitions of $V_c$, turbulent suspension mechanics, or erosion wear kinetics.

### Question 3: Why do rotary car dumpers in modern heavy-haul mining operations utilize rotary couplers, and what are the operational trade-offs?
- **What a strong answer contains:** Details that rotary couplers allow wagons to be overturned 160° to 180° along their longitudinal axis while remaining coupled to the train rake. Eliminating the requirement to uncouple and recouple 240 individual wagon air hoses and mechanical knuckles reduces cycle time per 2-car dump from 6 to 8 minutes (uncoupled) down to 75 to 90 seconds (coupled). This enables sustained terminal dumping throughput exceeding 8,000 to 10,000 t/h. Highlights the trade-offs: rotary couplers require precise indexer-positioner alignment to prevent torsional binding; they introduce complex mechanical wear surfaces requiring rigorous non-destructive magnetic particle inspection; and coupler failure inside the dumper barrel halts the entire terminal intake.
- **What a weak answer reveals:** Thinks rotary couplers are used just to save manual labor, without understanding cycle time arithmetic, brake pipe pneumatic continuity, or indexer alignment precision.

### Question 4: How do automated Bucket-Wheel Stacker-Reclaimers avoid physical collisions in high-density stockyards, and why is standard RTK-GPS insufficient on its own?
- **What a strong answer contains:** Explains that stockyards operate multiple 1,500-to-2,500-tonne machines along parallel or shared tracks with long rotating booms (50–60 meters). A complete Collision Avoidance System (CAS) maps dynamic 3D kinematic bounding envelopes based on track travel ($x$), boom slew ($\theta$), and luffing angle ($\phi$). Demonstrates why RTK-GPS is insufficient: satellite multipath reflections off adjacent metallic stockyard structures, gantry cranes, and shiploaders create positioning jitter; satellite line-of-sight can be obstructed during high-angle boom operations; and GPS cannot detect physical stockpile crest shifts, unexpected foreign obstacles, or manual service vehicles. Therefore, systems mandate multi-sensor fusion: RTK-GPS fused with 3D LiDAR scanners, millimeter-wave radar point clouds, high-integrity optical shaft encoders, and hardwired fail-safe mechanical PLC exclusion zones.
- **What a weak answer reveals:** Assumes a GPS receiver on the cab is completely sufficient, ignoring rotational boom kinematics, satellite multipath, and redundant fail-safe requirements.

### Question 5: Under International Association of Classification Societies (IACS) standards, why is multi-pass hatch loading mandatory for bulk carriers, and how does it interface with terminal shiploader dispatching?
- **What a strong answer contains:** Cites IACS Unified Requirement S1A. Single-pass loading places tens of thousands of tonnes in isolated holds while adjacent holds remain empty, generating extreme longitudinal shear forces and still-water bending moments (excessive hogging or sagging) that exceed the structural design limits of the ship's hull girder, risking deck compression buckle or keel fracture. Explains that 2-pass or 3-pass sequences distribute weight evenly along the keel. Details the operational interface: shiploaders must traverse repeatedly between hatches, incurring non-loading travel and boom repositioning time. Crucially links this to deballasting rates: if the terminal shiploading rate outpaces the ship's deballasting pumps, loading must pause, directly impacting berth occupancy and laytime calculations.
- **What a weak answer reveals:** Believes multi-pass loading is done merely to keep the ship level in the water for aesthetic or navigation reasons, completely missing hull girder bending moments, shear stress limits, and ballast pump synchronization.

### Question 6: What constitutes a legally valid Notice of Readiness (NOR) under standard BIMCO charter party terms, and what are the operational consequences of a defective NOR?
- **What a strong answer contains:** A legally valid NOR requires three conditions: (1) the vessel has arrived at the agreed geographical destination (within port limits or designated anchorage); (2) the vessel is physically ready in all respects to load the contracted cargo (all cargo holds clean, dry, free of prior cargo residue/rust scale, and passed by certified independent marine surveyors); and (3) all legal, customs, and health clearances (free pratique) are obtained or obtainable without delay. If an NOR is tendered prematurely (e.g., holds fail inspection due to coal dust contamination prior to loading grain or iron ore), the NOR is legally invalid *ab initio*. Laytime does not commence until a fresh, valid NOR is tendered after holds are re-cleaned and certified. If the terminal experiences loading delays in the interim, the shipowner cannot claim demurrage, potentially saving the charterer hundreds of thousands of dollars.
- **What a weak answer reveals:** Thinks an NOR is simply an email sent by the ship captain announcing arrival time, without understanding hold inspection certification, geographical port limits, or laytime inception mechanics.

### Question 7: How do you optimize the trade-off between inland rail cycle times and marine terminal vessel demurrage when facing an unexpected 3-day rail corridor outage?
- **What a strong answer contains:** Frames the problem as a dynamic cross-echelon buffer management problem. A 3-day rail outage halts inland replenishments, creating immediate risk of stockyard pad starvation. Steps: (1) Audit current stockyard inventory by grade; determine which vessels in the 7-day stem can be fully loaded from existing stockyard pads; (2) Re-sequence the Berth Allocation Plan (BAP) to berth vessels whose cargoes are 100% physically available, bypassing vessels whose cargoes require inbound rail delivery; (3) Contact vessels whose cargoes cannot be assembled and issue "Virtual Arrival" instructions under BIMCO clauses, allowing them to reduce steaming speed at sea, saving bunker fuel while contractually shifting arrival dates to defer laytime inception; (4) Upon rail restoration, sequence high-grade block trains to replenish specific starving pads rather than running balanced FIFO trains.
- **What a weak answer reveals:** Suggests simply loading whatever vessels arrive first until the yard runs out of ore, accepting demurrage as an uncontrollable act of god.

### Question 8: Why is the Transportable Moisture Limit (TML) a critical supply chain KPI in base-metal concentrate and iron ore fines logistics, and where in the chain must it be controlled?
- **What a strong answer contains:** Explains the physical phenomenon of cargo liquefaction under the IMO IMSBC Code (Group A cargoes). Granular bulk materials with high fines content and moisture above the Flow Moisture Point (FMP) appear solid when dry-stacked. However, engine vibration and ship rolling during ocean transit compact the cargo, decreasing inter-particle void volume. Pore water pressure increases until effective inter-granular friction drops to zero, converting solid cargo into a viscous slurry. The shifting fluid sloshes across holds, causing catastrophic vessel list and capsizing within minutes. TML is defined as $90\%$ of FMP. It must be controlled upstream: at mine dewatering filters, covered stockyard windrows, and rotary dumper sampling points. If rainfall elevates moisture above TML, terminal shiploaders must legally halt loading, even if laytime expires and demurrage accumulates.
- **What a weak answer reveals:** Confuses TML with simple customer payment penalties for excess water, failing to recognize it as a mandatory maritime safety threshold preventing vessel sinking.

---

## 15. Key terms

The following glossary defines key terms across heavy-haul rail, overland conveyors, slurry hydraulics, terminal stockyards, and maritime logistics:

- **Axle Load Limit:** The maximum permissible gross weight borne by a single wheelset on track infrastructure, dictating rail steel grade, sleeper spacing, and wagon payload capacity.
- **BAP (Berth Allocation Problem):** The mathematical optimization formulation that schedules incoming vessels to discrete or continuous terminal berths to minimize waiting time and demurrage costs.
- **Bingham Plastic:** A non-Newtonian fluid model characterizing mineral slurries that exhibit a yield stress threshold ($\tau_y$) followed by constant plastic viscosity ($\eta_p$).
- **Bucket-Wheel Stacker-Reclaimer (BWSR):** A large rail-mounted bulk-handling machine equipped with a counterweighted boom and bucket wheel used to stack and reclaim materials in longitudinal stockyards.
- **Charter Party:** The legal contract between a shipowner and a charterer (cargo shipper) defining freight rates, laytime allowances, demurrage, and despatch terms.
- **Chevron Stacking:** A longitudinal stockpile stacking method where material is deposited in successive layers along the centerline, reducing grade variance across transverse cuts.
- **Collision Avoidance System (CAS):** A fail-safe multi-sensor system (fusing LiDAR, radar, and GPS) that prevents structural contact between moving stockyard machines.
- **Critical Deposition Velocity ($V_c$):** The minimum slurry flow velocity required to maintain solid particles in turbulent suspension and prevent bed formation on the pipe invert.
- **Davis Equation:** An empirical quadratic formulation modeling train resistance as a function of journal friction, wheel-rail flange contact, and aerodynamic drag.
- **Deballasting Rate:** The volumetric discharge capacity of a vessel's ballast pumps, acting as an operational ceiling on maximum continuous shiploading rates.
- **Demurrage:** Contractual financial damages paid by a cargo charterer to a shipowner for time spent loading a vessel beyond agreed laytime limits.
- **Despatch:** An agreed financial bonus (typically half the demurrage rate) paid by a shipowner to a charterer for completing vessel loading ahead of contractual laytime.
- **Distributed Power (DP):** The operation of remote locomotive units placed throughout a train rake controlled via wireless telemetry to manage in-train draft forces.
- **Durand Equation:** A classical semi-empirical hydraulic equation used to predict critical deposition velocity in settling slurry pipelines.
- **Euler-Eytelwein Relation:** The fundamental capstan friction equation establishing maximum belt tension ratios on conveyor drive pulleys before slippage occurs.
- **Flow Moisture Point (FMP):** The moisture content percentage at which a granular bulk material begins to flow and exhibit liquid behavior under mechanical vibration.
- **Gross Train Weight (GTW):** The total physical mass of an entire train rake, including locomotives, empty wagon tare, and conveyed cargo payload.
- **Headway:** The minimum safe operational separation in time or distance required between consecutive train movements operating across a shared track block.
- **IMSBC Code:** The International Maritime Solid Bulk Cargoes Code published by the IMO governing safe stowage, handling, and transport of solid bulk materials.
- **Indexer-Positioner:** An automated electro-hydraulic arm that indexes train rakes forward through a rotary car dumper without decoupling wagons.
- **Laytime:** The contractually allowed duration specified in a charter party for loading or discharging bulk cargo without incurring demurrage penalties.
- **Meet-Pass Siding:** A track siding adjacent to a single-track mainline allowing trains to pass or overtake one another to resolve operational conflicts.
- **Notice of Readiness (NOR):** The formal declaration tendered by a ship master stating that the vessel has arrived and is ready in all respects to load cargo.
- **Overland Conveyor (OLC):** A high-tension trough-belt conveyor system transporting bulk solids across undulating natural terrain over multi-kilometer flights.
- **Regenerative Conveyor:** A conveyor flight descending a net downward gradient where gravitational cargo forces exceed frictional losses, generating electrical power.
- **Rotary Car Dumper:** A terminal mechanism that rotates rail wagons 160° to 180° along their longitudinal axis to discharge bulk cargo via gravity.
- **Rotary Coupler:** A specialized rail coupler allowing individual wagons to rotate in a car dumper barrel while remaining mechanically and pneumatically coupled.
- **Slack Flow:** An open-channel cavitation condition in slurry pipelines occurring when the hydraulic grade line falls below the physical pipeline elevation profile.
- **Transportable Moisture Limit (TML):** The maximum legal cargo moisture content ($90\%$ of FMP) permitted for safe ocean transport of Group A bulk cargoes.
- **Weather Working Day (WWD):** A legal laytime definition denoting a 24-hour day on which weather conditions permit loading of the contracted cargo without disruption.

---

## 16. Sources

The formulations, design criteria, and regulatory mandates presented in this topic are drawn from peer-reviewed literature, engineering standards, and international maritime codes:

1. **American Railway Engineering and Maintenance-of-Way Association (AREMA).** (2024). *Manual for Railway Engineering*. Volume 1 (Track) and Volume 4 (Operations and Economics). AREMA, Lanham, MD.
2. **Deutsches Institut für Normung (DIN).** (2022). *DIN 22101: Continuous conveyors — Belt conveyors for loose bulk materials — Basics for calculation and dimensioning*. Beuth Verlag, Berlin.
3. **International Maritime Organization (IMO).** (2023). *International Maritime Solid Bulk Cargoes Code (IMSBC Code) and Supplement*. 2023 Edition. IMO Publishing, London.
4. **International Association of Classification Societies (IACS).** (2021). *Unified Requirement S1A: Additional Requirements for Loading Conditions, Loading Manuals and Loading Instruments for Bulk Carriers*. Rev.7. IACS, London.
5. **Wasp, E. J., Kenny, J. P., & Gandhi, R. L.** (1977). *Solid-Liquid Flow: Slurry Pipeline Transportation*. Trans Tech Publications, Clausthal-Zellerfeld.
6. **Wilson, K. C., Addie, G. R., Sellgren, A., & Clift, R.** (2006). *Slurry Transport Using Centrifugal Pumps*. 3rd Edition. Springer, New York. [DOI: 10.1007/978-0-387-31055-8]
7. **Bierwirth, C., & Meisel, F.** (2015). A follow-up survey of berth allocation and quay crane scheduling problems in container and bulk terminals. *European Journal of Operational Research*, 244(3), 675–689. [DOI: 10.1016/j.ejor.2014.12.030]
8. **Thomas, D. G.** (1965). Transport characteristics of suspension: VIII. A note on the slip velocity of particles in turbulent flow and its relationship to critical deposition velocity. *AIChE Journal*, 11(2), 373–375.
9. **CEMA (Conveyor Equipment Manufacturers Association).** (2020). *Belt Conveyors for Bulk Materials*. 7th Edition with Metric Conversion. CEMA, Naples, FL.
10. **BIMCO.** (2013). *Charter Party Laytime Definitions 2013*. Baltic and International Maritime Council, Bagsvaerd, Denmark.
11. **Touran, A., & Lopez, R.** (2006). Modeling rail freight cycle times using Monte Carlo simulation. *Journal of Transportation Engineering*, 132(6), 464–470.
12. **BHP & Rio Tinto Technical Operational Disclosures.** (2020–2024). *Pilbara Iron Ore Logistics and Autonomous Rail Systems Technical Briefings*. Public Investor Presentation Series, Perth & Melbourne.

---

## 17. Links

### Module Navigation
- **Module Index:** [Module 39 — Mining, Metals, and Forestry Supply Chains](_index.md)
- **Previous Topic:** [Topic 05 — Processing and Recovery: Crushers, Mills, Smelters, and Metallurgical Balances](05_processing-recovery.md)
- **Next Topic:** [Topic 07 — Stockpiles and Bulk Inventory: Buffer, Blend, Degradation, and Working Capital](07_stockpiles-inventory.md)

### Related Sibling Topics
- [Topic 01 — Reserves, Grade, and Confidence: Mineral Resource Estimation and Reconciliation](01_reserves-grade-confidence.md)
- [Topic 02 — Mine Planning and Cut-Off Grade: Strategic Scheduling and Price Controllability](02_mine-planning-cutoff.md)
- [Topic 03 — Drilling, Blasting, and Load-Haul Operations](03_drilling-blasting-loadhaul.md)
- [Topic 04 — Grade Blending and Specification Management](04_grade-blending.md)
- [Topic 08 — Commodity Pricing, Quotation Periods, and Offtake Contracts](08_commodity-offtake.md)

### Foundation & Elective Modules
- [Module 11 — Network Design Optimization](../../02_methods/11_network_design/_index.md)
- [Module 15 — Production Planning and Scheduling](../../02_methods/15_production_planning_scheduling/_index.md)
- [Module 38 — Process Industries, Commodities, and Bulk Supply Chains](../38_process_industries_commodities/_index.md)
- [Survey File: Mining, Metals, and Forestry Supply Chains](../../../course/39_mining_metals_forestry.md)
