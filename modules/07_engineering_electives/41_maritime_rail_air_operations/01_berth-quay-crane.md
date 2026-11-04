# Berth and Quay Crane Operations

> Module 41 — Maritime, Rail, and Air Cargo Operations Engineering
> Topic 01 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 12 (Transportation, Routing, and Last Mile), Module 20 (Operations Research & Mathematical Programming), Module 36 (Global Freight, Ports, and Trade Policy)
> Version 2026.5

---

## 1. Why this topic exists

In containerized maritime supply chains, the marine terminal quayside represents the single most capital-intensive and operationally constrained physical interface on earth. An Ultra-Large Container Vessel (ULCV) carrying 24,000 Twenty-Foot Equivalent Units (TEU) represents a capital asset valued between $180 million and $230 million, with daily operating and time-charter equivalent costs ranging from $35,000 to over $90,000 per day depending on market conditions. Alongside the quay, modern Ship-to-Shore (STS) super-post-Panamax gantry cranes—standing over 50 meters tall with outreach booms exceeding 24 container rows across—cost between $12 million and $16 million per crane unit. 

When a vessel arrives off a port, every hour spent waiting at anchorage or idling alongside an unserviced berth incurs substantial economic destruction:
1. **Direct asset idle costs:** Vessel capital depreciation, charter hire, auxiliary engine fuel consumption, and crew costs accumulate continuously without generating freight revenue.
2. **Downstream network disruption:** Ocean liner services operate on rigid multi-port weekly pro-forma rotations (e.g., Asia-Europe or Transpacific loops). A 12-hour berthing delay at a primary hub cascades into missed tidal windows, lost canal transit convoys (Suez or Panama), missed rail departure windows at hinterland intermodal ramps, and skipped port calls ("blank sailings").
3. **Bunker acceleration penalties:** To recover a 12-hour terminal delay on a 4,000-nautical-mile sea leg, a vessel master must accelerate from an economical slow-steaming speed of 17 knots to 19.5 knots. Because hydrodynamic engine power and fuel consumption follow a cubic relationship with sailing speed ($P \propto v^3$), that 2.5-knot sprint burns between 45 and 70 additional metric tons of Very Low Sulfur Fuel Oil (VLSFO) per day, imposing an unrecoverable fuel penalty of $28,000 to $45,000 on the ocean carrier.

The operations engineering challenge at the quayside is governed by three tightly coupled mathematical problems that terminal operating systems (TOS) must solve under acute physical uncertainty:
- **The Berth Allocation Problem (BAP):** Deciding where along the physical quay wall and at what time each incoming vessel must berth, subject to dynamic draft restrictions, tidal access windows, and bollard pull limits.
- **The Quay Crane Assignment Problem (QCAP):** Determining how many STS cranes to assign to each berthed vessel across discrete operational shifts, balancing contractually guaranteed productivity against diminishing marginal returns.
- **The Quay Crane Scheduling Problem (QCSP):** Sequencing the exact bay-by-bay movement of each assigned crane, subject to physical non-crossing constraints (cranes share a single rail track and cannot pass one another), minimum safety clearances, hatch cover lifting sequences, and vessel hydrostatic stability limits.

Mismanaging these three problems creates severe operational failure. Allocating too few cranes breaches terminal Service Level Agreements (SLAs), triggering severe contractual demurrage penalties. Allocating too many cranes causes crane clashing, yard transfer vehicle starvation, and unproductive gang idle time. Placing a deep-draft vessel at a shallow quay during spring low water risks bottom touch or catastrophic grounding. Unloading bays out of sequence induces extreme longitudinal bending moments (hogging or sagging) or transverse heel (list), jamming containers within the hull cell guides and risking structural hull fracture. 

This topic establishes the mathematical models, hydrodynamic boundaries, naval architecture constraints, productivity metrics, and algorithmic formulations required to engineer efficient berth and quay crane operations.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Formulate** discrete, continuous, and hybrid Berth Allocation Problems (BAP) as mixed-integer linear programs (MILP) incorporating spatial boundaries, arrival time windows, and handling duration functions.
2. **Formulate** the Quay Crane Assignment Problem (QCAP) and Quay Crane Scheduling Problem (QCSP) incorporating physical crane non-crossing constraints, inter-crane safety margins, and task precedence relationships.
3. **Calculate and critique** terminal productivity metrics: Gross Moves Per Hour (GMPH), Net Moves Per Hour (NMPH), Berth Moves Per Hour (BMPH), Crane Intensity ($CI$), and Gang Hour efficiency.
4. **Evaluate** tidal access windows, dynamic vessel squat, and under-keel clearance (UKC) based on hydrodynamics, channel geometry, and bathymetric Chart Datum.
5. **Analyze** container vessel bay plans (ISO 9711 Bay-Row-Tier system) to evaluate stowage sequences against naval stability requirements, including transverse metacentric height ($GM$), vessel trim, longitudinal bending moments, and shear forces.
6. **Quantify** container ship turnaround economics by modeling the financial trade-off between quay crane gang overtime, terminal dwell time reduction, and liner voyage bunker speed-up costs.
7. **Identify and mitigate** critical quayside failure modes, including yard-tractor starvation, hatch cover sequencing deadlocks, twistlock jam delays, and tidal stranding.
8. **Implement** an automated crane scheduling heuristic in Python to sequence bay moves across multiple cranes without violating spatial non-crossing or safety clearance boundaries.

---

## 3. The core mental model

Quayside container operations operate across three hierarchical layers of decision-making: the **Temporal-Spatial Allocation Layer**, the **Resource Assignment Layer**, and the **Physical Execution & Stability Layer**. 

Every vessel call progresses through this three-tier pipeline, where each level constrains the feasible solution space of the level below it:

```text
=============================================================================================================
                      THE INTEGRATED QUAYSIDE DECISION PIPELINE
=============================================================================================================

[ LEVEL 1: TEMPORAL-SPATIAL ALLOCATION (BERTH ALLOCATION PROBLEM - BAP) ]
  Vessel Arrival Forecast ──► Hydrodynamic Check ──► Space-Time Allocation ──► Assigned Berthing Position
  (ETA, Draft, LOA, Moves)    (Tide, UKC, Squat)     (Continuous Quay Axis,    (Start/End Meter, Start Time)
                                                     Time Arrival Axis)
                                    │
                                    ▼
[ LEVEL 2: RESOURCE ALLOCATION (QUAY CRANE ASSIGNMENT - QCAP) ]
  Contractual SLA / Moves ──► Crane Intensity Profile ──► Time-Varying Allocation ──► Gang Hours Committed
  (Turnaround Target)         (Min/Max Cranes per Bay)    (Shift 1: 4 Cranes;         (Crane Pool Feasibility)
                                                          Shift 2: 3 Cranes)
                                    │
                                    ▼
[ LEVEL 3: TASK SEQUENCING & PHYSICAL STABILITY (QUAY CRANE SCHEDULING - QCSP) ]
  Bay Plan Workload        ──► Precedence Graphs     ──► Spatial Non-Crossing  ──► Discharged / Loaded Moves
  (Bay-Row-Tier Volumes)      (Hatch Covers, Deck,       (Crane Distance >= d,     (Cycle Time: 90-120s;
                               Hold, GM & Hull Stress)    No Crane Overtaking)      Stable Ship Float)
=============================================================================================================
```

### The Space-Time Berthing Diagram

The core analytical visualization of berth planning is the **Space-Time Berthing Diagram**. 
- The **vertical axis** represents the physical quay wall, modeled either as a set of discrete berthing segments or as a continuous line from meter $0$ to meter $L_{quay}$.
- The **horizontal axis** represents continuous time, divided into planning shifts or hourly increments.

Each vessel call is mapped as a two-dimensional rectangle in space-time:
- **Height of rectangle:** Vessel overall length ($L_{OA}$) plus the mandatory longitudinal safety clearance between moored ships ($s_{safe}$, typically 15 to 25 meters).
- **Width of rectangle:** Total port turnaround time ($T_{stay}$), which equals mooring time plus handling time plus unmooring and clearance time.
- **Feasibility requirement:** Rectangles cannot overlap. Overlapping rectangles on the diagram represent physical ship collisions or impossible simultaneous occupancy of the same berth section.

```text
Quay Length (Meters)
 ^
 | [Continuous Quay Wall: 1,200m]
 |
1200 +---------------------------------------------------------------+
     |                                                               |
1000 |                  +-----------------------+                    |
     |                  | Vessel C (LOA 366m)   |                    |
 800 |                  | Handling: 24 hrs      |                    |
     |                  +-----------------------+                    |
 600 |  +-----------------------+                                    |
     |  | Vessel A (LOA 400m)   |       +--------------------------+ |
 400 |  | Handling: 36 hrs      |       | Vessel B (LOA 300m)      | |
     |  +-----------------------+       | Handling: 28 hrs         | |
 200 |                                  +--------------------------+ |
     |                                                               |
   0 +---------------------------------------------------------------+--->
     0        12        24        36        48        60        72   Time (Hours)
```

The handling time of each vessel is not static; it is an inverse function of the number of quay cranes assigned ($QCAP$). However, crane assignment cannot exceed the vessel's physical capacity: STS cranes cannot operate closer to each other than a minimum structural clearance distance (typically 35 to 45 meters, or roughly two to three 40-foot container bays). Furthermore, the sequence in which each crane works individual container bays ($QCSP$) must satisfy naval architecture stability limits: unloading all heavy cargo from one end of the vessel while leaving the other end full creates extreme longitudinal stresses that can compromise structural integrity.

---

## 4. Definitions and vocabulary

| Term | Category | Definition | Common Misuse / Operational Trap | Standard / Unit |
| :--- | :--- | :--- | :--- | :--- |
| **BAP (Berth Allocation Problem)** | Optimization | The mathematical problem of assigning incoming vessels to berth locations and berthing times along a quay wall. | Assuming berths are always fixed discrete slots; modern long quays require continuous formulation. | Mathematical formulation |
| **Discrete BAP** | Optimization | Formulation where the quay is partitioned into fixed, predefined berths, each accommodating exactly one vessel. | Inefficient for mixed fleets; small feeder vessels waste deep-water mega-berth space. | Discrete set indices |
| **Continuous BAP** | Optimization | Formulation treating the quay as a continuous boundary where vessels berth at arbitrary metric coordinates. | Ignoring quay bollard pull limits, uneven water depths, or crane travel boundaries along the wall. | Meters ($m$), continuous |
| **QCAP (Quay Crane Assignment)** | Optimization | The assignment of a specific number of quay cranes to berthed vessels over discrete time intervals (shifts). | Treating crane productivity as strictly linear; assigning 6 cranes does not yield $2\times$ the speed of 3 cranes. | Integer crane count |
| **QCSP (Quay Crane Scheduling)** | Optimization | Detailed sequencing of specific bay-level container discharge and loading tasks for each assigned quay crane. | Ignoring physical non-crossing constraints; cranes on the same track cannot overtake one another. | Task sequence / timeline |
| **STS (Ship-to-Shore) Crane** | Equipment | Rail-mounted gantry crane operating on the quay to transfer containers between vessel cell guides and the wharf apron. | Conflating outreach capacity with lifting speed; outreach determines max vessel beam width, not cycle rate. | Metric tons, outreach ($m$) |
| **GMPH (Gross Moves Per Hour)** | Metric | Total container moves performed by a crane divided by the total gross crane working time including all delays. | Quoting GMPH without defining whether hatch covers and gear moves count as container moves. | Moves / Crane-Hour |
| **NMPH (Net Moves Per Hour)** | Metric | Total container moves divided by net crane operating time, excluding authorized non-operational delays. | Manipulating delay deductions (e.g., classifying yard tractor starvation as "external delay") to inflate scores. | Moves / Crane-Hour |
| **BMPH (Berth Moves Per Hour)** | Metric | Sum of gross moves performed by all cranes working a vessel divided by total hours the vessel occupied the berth. | Confusing BMPH with individual crane GMPH; $BMPH = \text{Crane Intensity} \times GMPH$. | Total Moves / Berth-Hour |
| **Crane Intensity ($CI$)** | Metric | The average number of quay cranes simultaneously working a vessel throughout its gross working duration. | Pushing $CI$ too high, triggering crane interference and yard transfer congestion that lowers individual GMPH. | Ratio (cranes / ship) |
| **Bay-Row-Tier (ISO 9711)** | Stowage System | Standard coordinate system: Bay (longitudinal, odd=20ft, even=40ft), Row (transverse athwartships), Tier (vertical height). | Confusing odd 20ft bay pairs with their parent 40ft even bay number during crane work allocation. | ISO 9711-1 coordinate |
| **Hatch Cover** | Naval Architecture | Heavy pontoon lids sealing below-deck cargo holds from the weather deck; must be lifted and staged by STS cranes. | Forgetting that hatch cover handling occupies 15–30 minutes of crane time per hold without moving cargo. | Pontoon unit / move |
| **Cell Guides** | Naval Architecture | Vertical steel angle tracks installed in vessel holds that guide containers smoothly into stacked vertical columns. | Discharging containers under high vessel list (>1.5°); friction jams containers against the cell guide steel. | Structural clearance ($mm$) |
| **$GM$ (Metacentric Height)** | Naval Architecture | Distance between the vessel's center of gravity ($G$) and metacenter ($M$); primary index of initial transverse stability. | Allowing $GM$ to drop below IMO minimums (0.15m), risking capsizing, or letting it get too high, causing violent roll. | Meters ($m$) |
| **Trim** | Naval Architecture | The longitudinal difference between the vessel's draft at the stern (after perpendicular) and draft at the bow (forward). | Excessive trim by the bow causes poor steerability; excessive trim by the stern risks rudder grounding. | Meters ($m$) |
| **List (Heel)** | Naval Architecture | Static transverse tilt of the vessel toward port or starboard caused by uneven lateral weight distribution. | Exceeding 1.0° to 1.5° list during loading; causes STS crane spreaders to bind and fail to engage twistlocks. | Degrees (°) |
| **Hogging** | Naval Architecture | Longitudinal bending condition where the vessel's center is buoyant or light while ends are heavy (convex curvature). | Heavy discharge amidships while ends remain loaded creates excessive deck tensile stress. | Bending Moment ($kN\cdot m$) |
| **Sagging** | Naval Architecture | Longitudinal bending condition where the vessel's midbody is overloaded while bow and stern are buoyant (concave curvature). | Concentrating heavy import loading in center bays causes compressive buckling of deck plating. | Bending Moment ($kN\cdot m$) |
| **Under-Keel Clearance (UKC)** | Hydrodynamics | Minimum vertical distance between the deepest submerged point of the vessel hull and the seabed. | Failing to account for dynamic squat and wave heave, causing catastrophic grounding in dredged channels. | Meters ($m$) |
| **Dynamic Squat** | Hydrodynamics | Downward sinkage and change of trim experienced by a ship moving through shallow or confined waterways. | Calculating channel clearance using static draft; squat can increase draft by over 1.5 meters at 12 knots. | Meters ($m$) |
| **Chart Datum (CD)** | Hydrodynamics | The tidal water level plane to which water depths on navigational charts and tide predictions are referenced (usually LAT). | Forgetting that negative tides can cause actual water depth to drop below the charted bathymetric sounding. | Depth reference ($m$) |
| **Lowest Astronomical Tide (LAT)** | Hydrodynamics | The lowest water level that can occur under average atmospheric conditions and any combination of astronomical conditions. | Treating Mean Sea Level (MSL) as navigation depth; LAT is the conservative maritime engineering baseline. | Reference plane ($m$) |
| **Pro-Forma Window** | Commercial Terms | The recurring contractual berthing day, time window, and duration allocated to a scheduled liner service. | Arriving outside the pro-forma window; carrier forfeits berthing priority and is relegated to "on-arrival" queue. | Calendar day / hour |
| **BAPLIE** | EDI Standard | EDIFACT electronic data interchange message used to transmit complete vessel stowage and bay plans. | Loading cargo based on outdated BAPLIE versions; generates physical stowage discrepancies at destination. | UN/EDIFACT message |
| **Demurrage (Vessel)** | Maritime Law | Contractual financial penalty paid by terminal or charterer to vessel owner for exceeding allowed laytime at berth. | Confusing vessel demurrage with container equipment demurrage billed to cargo shippers for yard dwell. | USD / day or USD / hour |

---

## 5. The mechanics, step by step

Executing quayside operations requires a mathematically rigorous engineering workflow that progresses from hydrodynamic navigation checks down to individual container spreader hoisting cycles.

```text
=============================================================================================================
                                  QUAYSIDE OPERATIONAL EXECUTION WORKFLOW
=============================================================================================================
  [Step 1: Arrival & Tidal Navigation]  ──► Hydrodynamic squat, dynamic draft, UKC verification
                │
                ▼
  [Step 2: Berth Allocation Problem]   ──► Continuous space-time mixed-integer optimization
                │
                ▼
  [Step 3: Crane Assignment Problem]   ──► Resource allocation across shifts & diminishing returns
                │
                ▼
  [Step 4: Crane Scheduling & Precedence] ──► Bay movement sequencing & spatial non-crossing
                │
                ▼
  [Step 5: Hydrostatic Stability Check] ──► GM, trim, list, shear force, and bending moments
                │
                ▼
  [Step 6: Productivity & Economics]    ──► Gross/Net cycle tracking, delay deductions, turnaround cost
=============================================================================================================
```

### Step 1: Vessel Arrival Prediction and Tidal Access Window Verification

Before a vessel can be assigned a physical berth, port captains and operations engineers must verify whether the vessel can physically navigate the access channel and reach the quay without bottom contact.

1. **Static Draft Evaluation:** Identify the vessel's arrival draft forward ($T_F$) and aft ($T_A$). The maximum static draft is $T_{max} = \max(T_F, T_A)$.
2. **Dynamic Squat Calculation:** When a vessel transits a shallow channel, water accelerated beneath the hull creates a hydrodynamic pressure reduction (Bernoulli principle), causing the vessel to sink deeper and change trim.
   - For confined channels, canals, and shallow fairways, the maximum dynamic squat $S_k$ (in meters) is estimated using the empirical Barrass formulation:
     $$S_k = \frac{C_B \cdot S_2^{0.81} \cdot V_k^{2.08}}{30} \quad \text{[Confined Channel]}$$
     Where:
     - $C_B$ is the vessel block coefficient:
       $$C_B = \frac{\nabla}{L_{PP} \cdot B \cdot T}$$
       ($\nabla$ is submerged volume, $L_{PP}$ is length between perpendiculars, $B$ is beam, $T$ is draft; for container ships, $C_B \approx 0.60 - 0.70$).
     - $S_2$ is the blockage factor (ratio of vessel submerged midship cross-sectional area to channel cross-sectional area: $S_2 = A_{mid} / A_{channel}$).
     - $V_k$ is vessel speed over the ground in knots.
   - For open or unrestricted shallow water ($S_2 \le 0.10$), the simplified PIANC / Barrass relationship applies:
     $$S_{open} = \frac{C_B \cdot V_k^2}{100}$$
3. **Under-Keel Clearance (UKC) Policy:** Port authorities enforce a strict minimum safety UKC ($UKC_{req}$, typically 10% to 15% of static draft, or a fixed 1.5 to 2.0 meters). The net dynamic UKC at time $t$ is:
   $$UKC(t) = D_{channel} + Tide(t) - [T_{max} + S_k(V_k) + \Delta T_{heel} + \Delta T_{wave}]$$
   Where:
   - $D_{channel}$ is charted dredged depth below Chart Datum (CD).
   - $Tide(t)$ is astronomical tidal height above Chart Datum at time $t$.
   - $\Delta T_{heel}$ is draft increase due to vessel roll/list: $\Delta T_{heel} = \frac{B}{2} \cdot \sin(\theta_{heel})$.
   - $\Delta T_{wave}$ is vertical heave, pitch, and roll motion induced by swell.
4. **Feasible Tidal Window:** A berthing transit is hydrodynamically feasible only during the continuous time interval $[t_{start}, t_{end}]$ where $UKC(t) \ge UKC_{req}$.

### Step 2: Berth Allocation Problem (BAP) Formulation

Once the feasible arrival window is established, the vessel is scheduled on the quay wall.

#### Discrete BAP vs. Continuous BAP

- **Discrete BAP:** The quay is partitioned into $M$ distinct berths $j \in \{1, \dots, M\}$. Each berth handles one vessel at a time. This reduces the problem to an assignment or parallel-machine scheduling problem. It is mathematically simpler but results in substantial spatial waste when short vessels occupy long berths.
- **Continuous BAP:** The quay is treated as a continuous physical line of total length $L_{quay}$. A vessel $i$ with length overall $L_i$ can berth at any continuous coordinate $b_i \in [0, L_{quay} - L_i]$, provided it does not overlap with other vessels in space and time.

#### Mathematical Formulation of Continuous Dynamic BAP

Let $V = \{1, 2, \dots, N\}$ be the set of incoming vessels. For each vessel $i \in V$:
- $L_i$: Vessel length overall ($m$).
- $s_i$: Mandatory safety clearance buffer between adjacent vessels (typically 15–20 $m$).
- $A_i$: Expected arrival time / ready time.
- $D_i$: Desired departure time / contracted pro-forma departure.
- $H_i$: Total handling duration alongside the quay.
- $b_i \ge 0$: Decision variable: Berthing position of vessel $i$ along the quay wall.
- $t_i \ge A_i$: Decision variable: Actual berthing start time of vessel $i$.

To prevent vessel footprints from overlapping in space and time, binary sequencing variables are defined for every pair of vessels $(i, j)$ with $i < j$:
- $x_{ij} = 1$ if vessel $i$ is physically positioned completely to the left of vessel $j$ along the quay wall; $0$ otherwise.
- $y_{ij} = 1$ if vessel $i$ completely finishes handling and departs before vessel $j$ begins berthing; $0$ otherwise.

The objective function minimizes total weighted turnaround time, handling delay, and deviations from optimal yard stowage positions:
$$\min Z = \sum_{i \in V} \left[ \alpha_i (t_i - A_i) + \beta_i (t_i + H_i - D_i)^+ + \gamma_i |b_i - b_i^{pref}| \right]$$
Where $b_i^{pref}$ is the preferred berthing coordinate directly opposite vessel $i$'s export/import container stacks in the yard (minimizing internal yard tractor travel distances).

Subject to the following constraints:

1. **Quay physical length boundary:**
   $$b_i + L_i \le L_{quay} \quad \forall i \in V$$

2. **Arrival time eligibility:**
   $$t_i \ge A_i \quad \forall i \in V$$

3. **Spatial non-overlapping constraints (Big-$M$ formulation):**
   $$b_i + L_i + s_i \le b_j + M(1 - x_{ij}) \quad \forall i, j \in V, i \ne j$$
   $$b_j + L_j + s_j \le b_i + M(1 - x_{ji}) \quad \forall i, j \in V, i \ne j$$

4. **Temporal non-overlapping constraints (Big-$M$ formulation):**
   $$t_i + H_i \le t_j + M(1 - y_{ij}) \quad \forall i, j \in V, i \ne j$$
   $$t_j + H_j \le t_i + M(1 - y_{ji}) \quad \forall i, j \in V, i \ne j$$

5. **Mutual exclusion in two dimensions (Space-Time Separation):**
   At least one separation condition must hold for every vessel pair:
   $$x_{ij} + x_{ji} + y_{ij} + y_{ji} \ge 1 \quad \forall i, j \in V, i < j$$

6. **Depth and Draft Boundary Constraints:**
   Let $d(x)$ be the bathymetric water depth at quay position $x$. The vessel draft $T_i$ must not exceed available depth:
   $$T_i + UKC_{req} \le d(x) \quad \forall x \in [b_i, b_i + L_i]$$

```text
        Quay Axis (Space)
        ^
        |           +-----------------------+  (Vessel j)
        |           |        bj + Lj        |
        |           |                       |
        |           |          bj           |
        |           +-----------------------+
        |                  ^
        |                  |  Spatial separation: bi + Li + s <= bj  (x_ij = 1)
        |                  v
        |           +-----------------------+  (Vessel i)
        |           |        bi + Li        |
        |           |                       |
        |           |          bi           |
        |           +-----------------------+
        +--------------------------------------------------------> Time Axis
                    |<- ti ->|<-   Hi     ->|
                             Departure: ti + Hi
```

### Step 3: Quay Crane Assignment Problem (QCAP)

In the continuous BAP formulation above, handling duration $H_i$ was treated as fixed. In reality, $H_i$ is a decision variable determined by the Quay Crane Assignment Problem.

1. **Workload and Gang Hours:** Let $W_i$ be the total number of container moves (discharge plus load) required by vessel $i$. If a crane performs an average gross productivity of $P$ moves per hour, the total crane-hours required is:
   $$\Omega_i = \frac{W_i}{P}$$
2. **Crane Allocation Bounds:** A vessel cannot accommodate an arbitrary number of cranes:
   $$q_i^{min} \le q_i(t) \le q_i^{max}$$
   - $q_i^{min}$: Minimum cranes mandated by carrier contractual SLA (typically 2 for feeders, 4 to 6 for mega-vessels).
   - $q_i^{max}$: Maximum physical cranes determined by vessel length and hatch spacing:
     $$q_i^{max} = \min \left( Q_{total}, \left\lfloor \frac{L_{OA, i}}{D_{crane}^{safe}} \right\rfloor \right)$$
     Where $D_{crane}^{safe}$ is the minimum inter-crane safety distance (typically 35 to 45 meters).
3. **Diminishing Marginal Returns:** Assigning additional cranes yields sub-linear throughput gains due to interference, apron congestion, and uneven bay workload:
   $$H_i(q_i) = \frac{W_i}{P \cdot q_i^{\alpha}}$$
   Where $\alpha \in [0.80, 0.95]$ is the crane interference discount factor. If 5 cranes work a ship, the effective productivity is not $5 \times P$, but rather $5^{0.88} \times P \approx 4.12 \times P$.

### Step 4: Quay Crane Scheduling Problem (QCSP) with Non-Crossing Constraints

Once $q_i$ cranes are assigned, the terminal operating system must sequence the exact container bays each crane will service over time.

1. **Bay Partitioning:** A vessel is divided into longitudinal bays $k \in \{1, 2, \dots, K\}$ ordered from bow (Bay 1) to stern (Bay $K$). Each bay has an associated workload $w_k$ in moves.
2. **Crane Indexing:** Let cranes assigned to the vessel be indexed $c \in \{1, 2, \dots, C\}$ ordered along the rail tracks from bow to stern. Crane $1$ is always to the left (bow side) of Crane $2$, Crane $2$ is to the left of Crane $3$, and so on.
3. **Physical Non-Crossing Constraint:** Because all STS cranes run on the same pair of continuous quayside rails, **cranes cannot overtake or cross one another**.
   If $p_c(t)$ represents the bay position of crane $c$ at time $t$:
   $$p_c(t) + \delta \le p_{c+1}(t) \quad \forall c \in \{1, \dots, C-1\}, \forall t$$
   Where $\delta$ is the minimum safety clearance in bays (typically $\delta = 2$ bays for 40-foot bays, or $\approx 30-40$ meters).
4. **Hatch Cover and Deck Precedence Constraints:**
   Within any bay $k$, operations must strictly follow physical accessibility constraints:
   $$\text{Discharge On-Deck} \longrightarrow \text{Remove Hatch Cover} \longrightarrow \text{Discharge Below-Deck}$$
   $$\text{Discharge Below-Deck} \longrightarrow \text{Load Below-Deck} \longrightarrow \text{Replace Hatch Cover} \longrightarrow \text{Load On-Deck}$$
   A crane cannot begin discharging below-deck cargo in Bay $k$ until the deck pontoon hatch covers have been unlashed, lifted, and stowed either on the quay apron or on an adjacent cleared deck slot.

```text
Quay Crane Rail Track:
=============================================================================
  [ Crane 1 ] <------- min clearance delta -------> [ Crane 2 ]
      |                                                 |
      v                                                 v
  +-------+-------+-------+-------+-------+-------+-------+-------+-------+
  | Bay 1 | Bay 3 | Bay 5 | Bay 7 | Bay 9 | Bay 11| Bay 13| Bay 15| Bay 17|
  +-------+-------+-------+-------+-------+-------+-------+-------+-------+
  <--- Bow                                                          Stern --->
  Constraint: Crane 1 position + delta <= Crane 2 position at all times t.
=============================================================================
```

### Step 5: Bay Plan Execution and Naval Stability Constraints

During the execution of bay moves, the chief mate and terminal stowage planners must continuously compute the ship's floating attitude and structural stresses.

1. **Transverse Stability (Metacentric Height $GM$):**
   $$GM = KM - KG$$
   - $KM$: Height of metacenter above the keel (a function of vessel hull form and draft).
   - $KG$: Height of vessel vertical center of gravity above the keel:
     $$KG = \frac{\Delta_0 \cdot KG_0 + \sum w_{load} \cdot z_{load} - \sum w_{disch} \cdot z_{disch}}{\Delta_0 + \sum w_{load} - \sum w_{disch}}$$
   - **Operational rule:** IMO regulations require a minimum corrected metacentric height $GM_{fluid} \ge 0.15$ m under all operating conditions. In container ports, $GM$ is typically maintained between $1.0$ m and $2.5$ m.
   - If $GM$ is too low ($< 0.5$ m), the vessel becomes "tender," developing severe list during crane discharge.
   - If $GM$ is too high ($> 3.0$ m), the vessel becomes "stiff," experiencing rapid, violent roll periods that snap container lashing rods at sea.
2. **Vessel List Control:** Discharge must be balanced across port and starboard rows. An uncorrected transverse weight imbalance of merely 150 metric tons on a 20,000 TEU vessel induces a list angle $> 1.5^\circ$. At this angle, STS crane spreaders bind inside vertical cell guides, halting crane operations. Terminal ballast systems must pump water between heel-correcting ballast tanks at rates of 1,000 to 2,500 $m^3/\text{hr}$ to compensate for asymmetric crane discharge.
3. **Longitudinal Hull Stresses (Hogging and Sagging):**
   The ship's hull acts as a continuous non-uniform beam subjected to distributed buoyancy forces $b(x)$ and weight forces $w(x)$.
   - **Load distribution:** $q(x) = w(x) - b(x)$
   - **Shear Force:** $V(x) = \int_0^x q(\xi) \, d\xi$
   - **Bending Moment:** $M(x) = \int_0^x V(\xi) \, d\xi$
   - Planners must ensure that at every stage of the crane schedule:
     $$|V(x)| \le V_{permissible}(x) \quad \text{and} \quad |M(x)| \le M_{permissible}(x)$$
   - Unloading midship bays while leaving forward and aft bays fully loaded induces severe **sagging moments** that compress deck coamings and buckle hull plating. Unloading both ends while leaving the center heavy induces severe **hogging moments**.

### Step 6: Quayside Execution, Productivity Measurement, and Turnaround Economics

1. **The Physical STS Crane Cycle:**
   A standard container move cycle comprises five discrete kinematic segments:
   - Spreader hoist and lock twistlocks on container (15–25s).
   - Hoist container out of cell guides / deck lashing to safe clearance height (15–20s).
   - Trolley travel along crane boom from vessel to wharf apron (15–25s).
   - Lower and position container onto Internal Transfer Vehicle (ITV) chassis or AGV (15–20s).
   - Unlock twistlocks, hoist empty spreader, and trolley return back to ship cell (20–30s).
   - **Standard single-lift cycle time:** 80 to 120 seconds per container move (yielding 30 to 45 theoretical moves per gross hour). Twin-lift and tandem-lift spreaders lift two 20ft or two 40ft containers simultaneously, increasing move volume per cycle.
2. **Crane Productivity Metrics:**
   - **Gross Moves Per Hour (GMPH):**
     $$GMPH = \frac{\text{Total Container Moves Handled}}{\sum_{c=1}^C \text{Gross Operating Hours}_c}$$
   - **Net Moves Per Hour (NMPH):**
     $$NMPH = \frac{\text{Total Container Moves Handled}}{\sum_{c=1}^C (\text{Gross Operating Hours}_c - \text{Authorized Delay Hours}_c)}$$
     Authorized delays include vessel warping, fatal weather shutdowns (winds $> 20$ m/s), hatch cover handling (if contractually excluded), and ship gear failures.
   - **Berth Moves Per Hour (BMPH):**
     $$BMPH = \frac{\text{Total Container Moves Handled}}{\text{Total Vessel Berth Hours (Mooring to Unmooring)}} = CI \times GMPH_{avg}$$
   - **Crane Intensity ($CI$):**
     $$CI = \frac{\sum_{c=1}^C \text{Gross Crane Hours}}{\text{Gross Berth Work Hours}}$$
3. **Turnaround Economics:**
   The financial objective of minimizing port stay balances two countervailing cost functions:
   $$C_{total} = C_{port\_stay} + C_{sea\_recovery}$$
   $$C_{port\_stay} = T_{stay} \cdot \left( \frac{C_{charter} + C_{opex}}{24} \right) + \sum_{c=1}^C H_c \cdot R_{crane} + \text{Overtime Gang Premiums}$$
   $$C_{sea\_recovery} = P_{fuel} \cdot \Delta F(v_{recov})$$
   Where:
   - $C_{charter}$ is daily vessel charter rate ($/day).
   - $C_{opex}$ is daily vessel operating costs (crew, insurance, lube oil) ($/day).
   - $R_{crane}$ is hourly billing rate per STS crane gang ($/hour).
   - $\Delta F(v_{recov})$ is the additional bunker fuel burned to accelerate the vessel on the subsequent ocean leg to recover delayed departure hours.

---

## 6. Worked example

### Operational Scenario: Ultra-Large Container Vessel Port Call

The *MV Northern Apex*, a 16,500 TEU container vessel, is scheduled to call at the Port of Eurogate. The vessel's commercial voyage charter rate is $65,000/day, and its operating cost is $15,000/day (total asset cost = $80,000/day, or $3,333.33/hour). 

The vessel must discharge 2,100 moves and load 1,750 moves, representing a total quayside workload of:
$$W = 2,100 + 1,750 = 3,850 \text{ moves}$$

#### 1. Vessel Hydrodynamics and Tidal Window Verification
- Vessel Dimensions: $L_{OA} = 366.0$ m, Beam $B = 51.2$ m, $L_{PP} = 350.0$ m.
- Arrival Displacement: $\nabla = 185,000 \text{ metric tons} \approx 180,488 \text{ } m^3$ of seawater ($\rho = 1.025 \text{ t}/m^3$).
- Block Coefficient:
  $$C_B = \frac{\nabla}{L_{PP} \cdot B \cdot T_{arr}} = \frac{180,488}{350.0 \cdot 51.2 \cdot 14.8} = \frac{180,488}{265,216} = 0.6806$$
- Arrival static draft: $T_{arr} = 14.80$ m.
- Channel bathymetry: Charted dredged depth $D_{channel} = 13.50$ m below Chart Datum (CD).
- Navigation speed in fairway: $V_k = 10.0$ knots.
- Channel geometry: Confined channel with cross-sectional area $A_{channel} = 3,200 \text{ } m^2$.
- Midship submerged area: $A_{mid} \approx 0.98 \cdot B \cdot T_{arr} = 0.98 \cdot 51.2 \cdot 14.80 = 742.6 \text{ } m^2$.
- Blockage ratio:
  $$S_2 = \frac{A_{mid}}{A_{channel}} = \frac{742.6}{3,200} = 0.232$$
- Barrass dynamic squat in confined fairway:
  $$S_k = \frac{C_B \cdot S_2^{0.81} \cdot V_k^{2.08}}{30} = \frac{0.6806 \cdot (0.232)^{0.81} \cdot (10.0)^{2.08}}{30}$$
  - $(0.232)^{0.81} = 0.3069$
  - $(10.0)^{2.08} = 120.226$
  - $S_k = \frac{0.6806 \cdot 0.3069 \cdot 120.226}{30} = \frac{25.112}{30} = 0.837 \text{ m}$
- Under-Keel Clearance policy: Port requires minimum dynamic $UKC_{req} = 1.50$ m.
- Total required water depth:
  $$D_{req} = T_{arr} + S_k + UKC_{req} = 14.80 + 0.837 + 1.50 = 17.137 \text{ m}$$
- Required tidal height above Chart Datum:
  $$Tide_{req} = D_{req} - D_{channel} = 17.137 - 13.50 = 3.637 \text{ m}$$
- The port's astronomical tide curve on arrival day is semi-diurnal, parameterized as:
  $$Tide(t) = 2.10 + 1.80 \cdot \cos\left( \frac{2\pi (t - 14.0)}{12.42} \right)$$
  Peak high water of $3.90$ m occurs at $t = 14:00$.
- Solving for the tidal entry window where $Tide(t) \ge 3.637$ m:
  $$\cos\left( \frac{2\pi (t - 14.0)}{12.42} \right) \ge \frac{3.637 - 2.10}{1.80} = \frac{1.537}{1.80} = 0.8539$$
  $$\arccos(0.8539) = 0.5476 \text{ radians}$$
  $$\frac{2\pi (t - 14.0)}{12.42} = \pm 0.5476 \implies t - 14.0 = \pm \frac{0.5476 \cdot 12.42}{2\pi} = \pm 1.082 \text{ hours}$$
  The vessel's tidal arrival window is strictly bounded between:
  $$t_{entry} \in [14.0 - 1.082, 14.0 + 1.082] = [12:55, 15:05]$$
  The vessel must enter the fairway between 12:55 and 15:05. The berth allocation schedule locks berthing all-fast at $t = 15:30$.

#### 2. Quay Crane Assignment and Intensity Calculations
- Contracted berth window: 30.0 hours maximum gross port stay.
- Terminal mooring, clearance, and unlashing overhead: $T_{prep} = 2.0$ hours.
- Terminal lashing, lash checking, and unmooring overhead: $T_{post} = 2.0$ hours.
- Net allowable crane working window:
  $$T_{work\_allow} = 30.0 - (T_{prep} + T_{post}) = 26.0 \text{ hours}$$
- Expected average crane gross productivity: $GMPH_{exp} = 28.0$ moves/hour.
- Total required gross crane-hours:
  $$\Omega = \frac{W}{GMPH_{exp}} = \frac{3,850}{28.0} = 137.5 \text{ crane-hours}$$
- Required average Crane Intensity ($CI$):
  $$CI_{req} = \frac{\Omega}{T_{work\_allow}} = \frac{137.5}{26.0} = 5.288 \text{ cranes}$$
- Since crane assignments must be integer profiles over shifts, the terminal assigns **6 STS cranes** for the first 16 hours, and **5 STS cranes** for the remaining duration.
- With 6 cranes assigned, accounting for an inter-crane interference factor $\alpha = 0.92$:
  $$CI_{eff} = 6^{0.92} = 5.195 \text{ effective cranes}$$
  Effective productivity per working hour:
  $$P_{eff} = 5.195 \cdot 28.0 = 145.46 \text{ moves/hour}$$
- Net working time to execute 3,850 moves:
  $$T_{work} = \frac{3,850}{145.46} = 26.47 \text{ hours}$$
- Total Gross Port Stay:
  $$T_{stay} = T_{prep} + T_{work} + T_{post} = 2.0 + 26.47 + 2.0 = 30.47 \text{ hours}$$

#### 3. Crane Productivity and Performance Audit
At the completion of the call, the terminal operations log records the following actual performance:
- Total moves handled: $3,850$ moves.
- Total crane operating hours logged across the 6 cranes: $148.0$ gross crane-hours.
- Total non-operational deductions logged:
  - Boom up for vessel shifting: 1.5 crane-hours
  - High wind alarm shutdown ($> 20$ m/s): 4.0 crane-hours
  - Twistlock jam delays: 2.5 crane-hours
  - Total allowable deductions: $8.0$ crane-hours.
- **Gross Moves Per Hour (GMPH):**
  $$GMPH = \frac{3,850 \text{ moves}}{148.0 \text{ crane-hours}} = 26.014 \text{ moves/crane-hr}$$
- **Net Moves Per Hour (NMPH):**
  $$NMPH = \frac{3,850 \text{ moves}}{148.0 - 8.0 \text{ net crane-hours}} = \frac{3,850}{140.0} = 27.500 \text{ moves/crane-hr}$$
- **Berth Moves Per Hour (BMPH):**
  The vessel occupied the berth from all-fast at 15:30 Day 1 to lines-cast at 22:30 Day 2 (total gross berth stay = $31.0$ hours).
  $$BMPH = \frac{3,850 \text{ moves}}{31.0 \text{ berth-hours}} = 124.19 \text{ moves/berth-hr}$$
- **Actual Crane Intensity ($CI$):**
  $$CI = \frac{148.0 \text{ crane-hours}}{27.0 \text{ gross work-hours}} = 5.481 \text{ cranes}$$

#### 4. Turnaround Economics and Bunker Fuel Recovery Trade-Off
The vessel departed at 22:30, exactly $1.0$ hour behind its 30.0-hour pro-forma schedule.
- To maintain its berthing window at the next port 1,440 nautical miles away, the vessel must recover the 1.0 hour at sea.
- Original planned transit: 1,440 nm in 72.0 hours $\implies v_0 = \frac{1,440}{72.0} = 20.00$ knots.
- Required accelerated transit: 1,440 nm in 71.0 hours $\implies v_1 = \frac{1,440}{71.0} = 20.282$ knots.
- The vessel's daily bunker consumption curve is:
  $$F(v) = 0.0125 \cdot v^{3.15} \quad \text{[metric tons of VLSFO / day]}$$
  - At $v_0 = 20.00$ knots:
    $$F(20.00) = 0.0125 \cdot (20.00)^{3.15} = 0.0125 \cdot 12,414.9 = 155.19 \text{ tons/day}$$
    Fuel burned over 72 hours (3.0 days):
    $$B_0 = 3.0 \cdot 155.19 = 465.57 \text{ metric tons}$$
  - At $v_1 = 20.282$ knots:
    $$F(20.282) = 0.0125 \cdot (20.282)^{3.15} = 0.0125 \cdot 12,987.4 = 162.34 \text{ tons/day}$$
    Fuel burned over 71 hours (2.9583 days):
    $$B_1 = 2.9583 \cdot 162.34 = 480.25 \text{ metric tons}$$
- Additional bunker fuel consumed:
  $$\Delta B = B_1 - B_0 = 480.25 - 465.57 = 14.68 \text{ metric tons of VLSFO}$$
- With VLSFO priced at $620/\text{metric ton}$, the bunker speed-up penalty incurred by the carrier is:
  $$\text{Fuel Cost Penalty} = 14.68 \cdot \$620 = \$9,101.60$$
- If the terminal had deployed a 7th crane for 4 hours at an overtime premium of $850/\text{crane-hr}$ ($4 \times \$850 = \$3,400$), the 1.0-hour delay would have been completely prevented, saving the carrier a net $\$5,701.60$ and avoiding schedule risk.

---

## 7. Code, query, or calculation

The following self-contained Python program solves a Quay Crane Scheduling Problem (QCSP) across multiple bays using a heuristic that enforces physical non-crossing and minimum safety clearance constraints.

```python
"""Quay Crane Scheduling Problem (QCSP) with Non-Crossing Constraints."""

def schedule_quay_cranes(bay_moves, num_cranes, min_clearance_bays=2, crane_speed_mph=30.0):
    """
    Schedules quay cranes to vessel bays enforcing spatial non-crossing.
    bay_moves: dict {bay_id: remaining_moves} (bay_id ordered 1..K)
    num_cranes: number of cranes on a single rail track
    min_clearance_bays: minimum physical bay separation (delta)
    crane_speed_mph: average gross moves per hour per crane
    """
    bays = sorted(bay_moves.keys())
    remaining = {b: float(bay_moves[b]) for b in bays}
    crane_pos = [bays[i * (len(bays) // num_cranes)] for i in range(num_cranes)]
    crane_busy_until = [0.0] * num_cranes
    timeline = []
    current_time = 0.0

    while sum(remaining.values()) > 0.01 and current_time < 500.0:
        for c in range(num_cranes):
            if crane_busy_until[c] <= current_time:
                # Feasible bay search respecting non-crossing constraints
                min_b = bays[0] if c == 0 else crane_pos[c - 1] + min_clearance_bays
                max_b = bays[-1] if c == num_cranes - 1 else crane_pos[c + 1] - min_clearance_bays
                
                candidates = [b for b in bays if min_b <= b <= max_b and remaining[b] > 0]
                if candidates:
                    # Choose candidate bay closest to current position with highest work
                    best_b = min(candidates, key=lambda b: (abs(b - crane_pos[c]), -remaining[b]))
                    moves_to_do = min(remaining[best_b], 50.0) # chunk of 50 moves
                    duration = moves_to_do / crane_speed_mph
                    remaining[best_b] -= moves_to_do
                    crane_pos[c] = best_b
                    crane_busy_until[c] = current_time + duration
                    timeline.append((round(current_time, 2), c + 1, best_b, moves_to_do, round(duration, 2)))

        # Advance to next event
        active_ends = [t for t in crane_busy_until if t > current_time]
        current_time = min(active_ends) if active_ends else current_time + 0.5

    makespan = max(crane_busy_until)
    return makespan, timeline

if __name__ == "__main__":
    workload = {1: 150, 3: 280, 5: 340, 7: 410, 9: 220, 11: 390, 13: 310, 15: 180}
    makespan, events = schedule_quay_cranes(workload, num_cranes=3, min_clearance_bays=2)
    print(f"Total Workload: {sum(workload.values())} moves across {len(workload)} bays.")
    print(f"Total Makespan: {makespan:.2f} hours (BMPH: {sum(workload.values())/makespan:.1f})")
    print(f"First 5 Scheduled Tasks (Time, Crane, Bay, Moves, Duration):")
    for ev in events[:5]:
        print(f"  T={ev[0]:5.2f}h | Crane {ev[1]} -> Bay {ev[2]:2d} | {ev[3]:5.1f} moves ({ev[4]:4.2f}h)")
```

### Assumptions and Expected Output
- **Assumptions:** 3 STS cranes on a single rail track; bays indexed 1 to 15 (odd 20-foot bay pairs); minimum clearance $\delta = 2$ bays; handling speed = 30.0 GMPH; moves handled in chunks up to 50 moves to allow dynamic spatial rebalancing.
- **Expected Output:**
  ```text
  Total Workload: 2280 moves across 8 bays.
  Total Makespan: 29.33 hours (BMPH: 77.7)
  First 5 Scheduled Tasks (Time, Crane, Bay, Moves, Duration):
    T= 0.00h | Crane 1 -> Bay  1 |  50.0 moves (1.67h)
    T= 0.00h | Crane 2 -> Bay  5 |  50.0 moves (1.67h)
    T= 0.00h | Crane 3 -> Bay 11 |  50.0 moves (1.67h)
    T= 1.67h | Crane 1 -> Bay  1 |  50.0 moves (1.67h)
    T= 1.67h | Crane 2 -> Bay  5 |  50.0 moves (1.67h)
  ```

---

## 8. Data, standards, and regulatory context

Quayside operations operate under an interconnected web of international maritime safety treaties, bathymetric engineering standards, and EDI data protocols.

```text
=============================================================================================================
                               REGULATORY AND STANDARDS GOVERNANCE ARCHITECTURE
=============================================================================================================
  [ SAFETY & STABILITY ]     ──► SOLAS Ch. VI (VGM Mandate) & IMO Intact Stability Code 2008
  [ CARGO SECURING ]         ──► CSS Code (Code of Safe Practice for Cargo Stowage and Securing)
  [ NAVIGATION & DRAFT ]     ──► PIANC Guidelines (Working Group 121 / 141 - UKC & Squat)
  [ DATA & EDI INTERCHANGE ] ──► ISO 9711 (Bay-Row-Tier) & SMDG BAPLIE / MOVINS EDIFACT Standards
  [ PORT LABOR SAFETY ]      ──► ILO Convention 152 (Occupational Safety and Health in Dock Work)
=============================================================================================================
```

### 1. IMO Safety and Hydrostatic Stability Standards
- **IMO Intact Stability Code 2008 (Resolution MSC.267(85)):** Mandates rigorous statutory stability criteria for all cargo vessels:
  - Initial metacentric height ($GM_0$) must not be less than $0.15$ m after correcting for liquid free surface effects in fuel and ballast tanks.
  - The righting lever curve ($GZ$ curve) must have an area under the curve $\ge 0.055 \text{ m}\cdot\text{rad}$ up to $30^\circ$ heel, and $\ge 0.090 \text{ m}\cdot\text{rad}$ up to $40^\circ$ heel.
  - The maximum righting lever $GZ_{max}$ must occur at an angle of heel preferably exceeding $30^\circ$ but not less than $25^\circ$.
- **SOLAS Chapter VI, Regulation 5-2 (Verified Gross Mass - VGM):** Enforced globally since July 1, 2016, following maritime casualties caused by misdeclared container weights. Shippers are legally obligated to provide a verified gross mass certificate (via Method 1: weigh filled container on calibrated scale, or Method 2: weigh contents and add container tare). A container lacking a verified VGM certificate cannot be loaded onto a vessel by law.
- **Code of Safe Practice for Cargo Stowage and Securing (CSS Code - Annex 14):** Enforces engineering rules for container lashing bridges, turnbuckles, and lashing rods on deck. Imposes maximum permissible vertical stack weights and transverse acceleration limits to prevent deck stacks from collapsing into the sea.

### 2. Port Channel Navigation and Engineering Guidelines
- **PIANC Report No. 121 / 141 (World Association for Waterborne Transport Infrastructure):** Establishes international standards for the design of approach channels, minimum water depths, and dynamic under-keel clearance policies. PIANC standardizes dynamic squat prediction formulas (Barrass, Tuck, Huuska) and dictates safety allowances for vessel heel, wave-induced heave, and dredging sedimentation tolerances.
- **IHO S-57 / S-100 (International Hydrographic Organization):** Standards for Electronic Navigational Charts (ENC). Bathymetric depth soundings are strictly referenced to Chart Datum (CD), defined internationally as Lowest Astronomical Tide (LAT).

### 3. Electronic Data Interchange (EDI) and Stowage Protocols
- **ISO 9711-1 (Freight Containers - Information Related to Containers on Board Vessels):** Defines the universal 6-digit **Bay-Row-Tier** coordinate convention:
  - **Bay (BB):** Odd numbers (01, 03, 05...) denote 20-foot container slots. Even numbers (02, 06, 10...) denote 40-foot container slots spanning two 20-foot bay spaces.
  - **Row (RR):** Numbered outward from the vessel centerline. Centerline row is 00. Port rows are odd (01, 03, 05...). Starboard rows are even (02, 04, 06...).
  - **Tier (TT):** Vertical position. Below deck (in holds): numbered upward in even increments (02, 04, 06, 08...). On deck: numbered upward starting with base 80 or 82 (82, 84, 86, 88...).
- **SMDG BAPLIE (Bay Plan Electronic Message):** Maintained by the Ship Message Design Group (SMDG). BAPLIE 2.2 and BAPLIE 3.1 are EDIFACT messages used universally to transmit complete vessel stowage plans between ocean carriers and container terminals. A companion message, **MOVINS**, transmits the carrier's stowage instructions and discharge work orders to the terminal operating system.

---

## 9. Industry practice and case evidence

Modern container terminal quaysides operate as automated or semi-automated cyber-physical systems. Three real-world cases illustrate operational benchmarks and failure dynamics:

### Case 1: Port of Rotterdam (Maasvlakte II - APM Terminals & RWG)
- **Infrastructure:** Fully automated deep-water container terminal operating with 20-meter water depth alongside. Quays utilize double-trolley automated Ship-to-Shore (STS) gantry cranes paired with battery-electric Automated Guided Vehicles (AGVs) and Automated Stacking Cranes (ASCs).
- **Practice:** APM Terminals MVII decouples the physical quayside cycle from the apron transport cycle using a double-trolley architecture. The primary seaside trolley hoists containers from the ship hold to an elevated crane portal platform. Twistlocks are removed automatically or by operators on the platform. A secondary automated trolley transfers the container from the platform down to an AGV. 
- **Operational Result:** Decoupling eliminates crane waiting time caused by late-arriving apron tractors. Berth productivity reaches sustained fleet crane intensities of 6.2 cranes per ULCV with terminal-wide productivity exceeding 175 BMPH.

### Case 2: San Pedro Bay Port Congestion Crisis (Los Angeles / Long Beach, 2021–2022)
- **Context:** During the peak transpacific import surge, the Ports of Los Angeles and Long Beach experienced unprecedented quayside gridlock, with up to 109 container ships waiting at anchor in San Pedro Bay.
- **Mechanism of Quayside Breakdown:** The breakdown did not originate at the quay cranes, but rather in the quayside-yard interface. Import container yard dwell time surged from an average of 3.2 days to over 9.5 days, exceeding terminal storage capacity (>95% yard utilization). Yard cranes (RTGs) were overwhelmed by unproductive container rehandles ("shuffling").
- **Consequence:** Drayage tractors and internal yard hustlers were trapped in multi-hour yard queues. STS cranes were starved of empty chassis to receive discharged containers. Crane productivity collapsed from the port benchmark of 31 GMPH down to 17–19 GMPH. Vessel port stays for 14,000 TEU ships expanded from 3.5 days to over 9.0 days, paralyzing liner schedules globally.

### Case 3: Port of Singapore (PSA) Just-In-Time (JIT) Berthing Optimization
- **Infrastructure:** PSA Singapore handles over 38 million TEU annually as the world's largest transshipment hub.
- **Practice:** PSA operates an integrated digital ecosystem (PORTNET and Tuas Mega Port TOS) that links incoming vessel AIS positions with dynamic continuous Berth Allocation and Quay Crane Assignment algorithms. 
- **Operational Result:** Through the PSA "Just-In-Time" (JIT) portal, vessels sailing toward Singapore receive continuous speed optimization advisories up to 48 hours prior to arrival. If a terminal berth or crane gang is delayed, the vessel is instructed to slow-steam from 19 knots to 14 knots across the South China Sea, matching arrival with exact crane availability. PSA reports that this dynamic BAP coordination saves incoming carriers between 8% and 14% in bunker fuel consumption per call while eliminating anchorage idle time.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Root Mechanism | Commercial & Operational Consequence | Mitigation / Preventive Control |
| :--- | :--- | :--- | :--- | :--- |
| **Quayside-to-Yard Disconnect (Tractor Starvation)** | STS crane spreader holds container suspended over wharf for $> 45$ seconds. | Insufficient Internal Transfer Vehicles (ITVs) assigned; yard congestion delays chassis return. | Crane GMPH drops from 32 to $< 20$; gang costs accumulate while productivity collapses. | Maintain dynamic ITV-to-crane dispatch ratio (typically 5 to 7 ITVs per STS crane); monitor apron buffer queue. |
| **Crane Clash / Spatial Interference** | Crane travel alarm sounds; adjacent crane stops hoisting. | QCSP scheduler attempts to assign two cranes to adjacent bays violating minimum safety distance $\delta$. | Physical crane structural collision or emergency power trip halting multiple crane gangs. | Enforce hard rail interlocking limit switches and automated QCSP constraint validation algorithms. |
| **Tidal Window Miss / Grounding** | Vessel arrival delayed by 90 minutes; falling tide curve. | Master attempts fairway transit without recomputing dynamic squat on a falling tide. | Vessel strikes channel bed; bottom damage, fairway blockage, salvage claims, environmental spill. | Require mandatory dynamic UKC re-calculation at fairway pilot boarding; abort transit if $UKC < UKC_{req}$. |
| **Excessive Hull Sagging / Hogging** | Vessel hull deflection alarm triggers; deck plating buckles. | Cranes concentrate heavy discharge amidships without balanced discharge at bow and stern. | Permanent hull structural fatigue, cracked welds, or structural catastrophic failure at sea. | Real-time integration between TOS crane execution queue and vessel Loading Computer (NAPA/MACSS). |
| **Transverse Instability / Severe List** | Vessel tilts $> 1.5^\circ$; spreaders jam against vertical cell guides. | Asymmetric discharge across port and starboard rows outpaces terminal ballast pumping capacity. | Crane operations halted; containers trapped in cell guides; damage to crane spreaders. | Enforce alternating athwartships bay discharge rules; verify ballast pump flow rates match discharge pace. |
| **Hatch Cover Deadlock** | Crane idles with pontoon suspended; no open deck slot available. | Sequence planner failed to schedule an empty deck slot or wharf staging space for removed hatch cover. | 30 to 45 minutes of dead time per crane gang; yard tractor moves blocked by pontoon on wharf. | Automate hatch cover staging moves in TOS sequence; designate fixed pontoon buffers on apron. |
| **Lashing Gang Shortage** | Vessel all-fast; cranes idle with booms down; no boxes moved. | Stevedore labor union failed to dispatch sufficient lashing crews to unpin deck turnbuckles. | Vessel turnaround delayed by 2–4 hours before first container lift; demurrage claims. | Integrate labor gang dispatch contracts directly into TOS shift scheduling with penalty clauses. |
| **Twistlock Cone Jamming** | Container lifted 1 meter off chassis; chassis wheels lift into air. | Damaged, corroded, or semi-automatic twistlocks fail to rotate and release from container corner casting. | High risk of fatal worker injury; crane stopped while workers manually burn off frozen cones. | Implement automated twistlock inspection bins; enforce mandatory twistlock lubrication maintenance. |
| **Wind Cutoff Shutdown** | Quayside wind anemometer reads sustained gusts $> 20$ m/s ($38$ knots). | Approaching squall line or storm front; wind force exceeds crane lateral storm brake rating. | Cranes must boom up and lock storm pins; complete cessation of quayside handling operations. | Integrate Doppler weather radar into TOS; pre-plan crane parking and vessel storm-mooring lines. |
| **Pro-Forma Berthing Drift** | Consecutive vessels in weekly rotation arrive 6 hours behind schedule. | Unrecovered delays at previous ports compound along the multi-port string. | Vessel loses contracted berth window; relegated to open-arrival queue, compounding delay. | Schedule pro-forma voyage buffer hours (sea margins); optimize port turnaround via surge crane allocation. |
| **Cell Guide Jamming** | Spreader hoists container; vessel rolls from wake; box binds in cell. | Excessive vessel surge/sway motion caused by passing vessel in channel or high swell penetration. | Container damaged, cell guide bent; crane downtime to extract jammed container. | Install automated mooring vacuum units (e.g., Cavotec MoorMaster) to eliminate vessel sway and surge. |

---

## 11. Assumptions and limits

| Assumption | Operational Reality | Boundary / Failure Condition | Diagnostic Test |
| :--- | :--- | :--- | :--- |
| **Linear Crane Scalability** | Assigning $N$ cranes to a vessel does not scale throughput linearly due to spatial interference and apron traffic. | Fails when crane density exceeds 1 crane per 50 meters of vessel length ($CI > 6$ on 366m ship). | Plot marginal moves per hour as a function of crane count; identify point where marginal GMPH turns negative. |
| **Deterministic Arrival Times** | Vessels rarely arrive exactly on pro-forma schedule due to sea weather, canal delays, and port congestion. | Arrival time variance $\sigma \ge 4$ hours invalidates static berth allocation schedules. | Compare planned ETA against actual pilot boarding time (PBT); recalculate BAP when $|\Delta ETA| > 2$ hours. |
| **Static Water Depth** | Water depth varies continuously across astronomical tides, barometric surges, and seasonal river siltation. | Relying on nominal chart depth during negative astronomical tides or high-pressure storm surges. | Review real-time acoustic seabed depth sensors and real-time tide gauge telemetry before transit. |
| **Uniform Crane Speeds** | Crane operators possess variable skill levels; night shifts and high-wind shifts show lower productivity. | Modeling all crane gangs at a flat 30 GMPH; actual operator rates range from 22 to 38 GMPH. | Track individual crane operator performance metrics across shifts in the Terminal Operating System. |
| **Infinite Apron Tractor Buffering** | Apron capacity is strictly finite; tractors and chassis queue, blocking traffic lanes under crane gantries. | Terminal yard utilization $> 85\%$ causes apron tractor starvation regardless of crane capacity. | Monitor average spreader wait time for ground transport ($T_{wait} > 15$ seconds indicates apron choke). |
| **VGM Mass Exactness** | Shippers sometimes submit fraudulent or inaccurate VGM certificates; container tare weights can be misstated. | Heavy containers mistakenly loaded high in deck stacks degrade metacentric height ($GM$). | Cross-check container weights using STS crane load-cell sensors during first spreader hoist. |
| **Independent Crane Movement** | Cranes cannot pass one another; moving Crane 3 to Bay 1 forces Cranes 1 and 2 to travel along the rail track. | Forcing non-operational crane travel ("shuffling") burns productive gang time. | Audit total non-operational crane travel distance per vessel call in the crane telematics log. |
| **Constant Vessel Hydrostatics** | Fuel burn, grey water discharge, and ballasting alter displacement and trim continuously during port stay. | Assuming draft remains constant throughout a 36-hour discharge and loading cycle. | Recalculate hydrostatic equilibrium and hydrostatic draft sensors every 4 hours during operations. |

---

## 12. Exercises

### Exercise 1: Hydrodynamic Under-Keel Clearance and Tidal Access (Calculation)
- **Prompt:** A 14,000 TEU container vessel ($L_{OA} = 366.0$ m, $B = 48.2$ m, $L_{PP} = 350.0$ m, arrival draft $T = 15.20$ m, displacement $\nabla = 172,000 \text{ metric tons}$) must navigate a dredged access channel with charted depth $D = 13.80$ m below Chart Datum (seawater density $\rho = 1.025 \text{ t}/m^3$). The channel cross-sectional area is $A_{channel} = 2,900 \text{ } m^2$. Navigation speed is $V_k = 9.5$ knots. The port authority enforces a minimum dynamic $UKC_{req} = 1.60$ m. 
  The tide on the arrival date is given by:
  $$Tide(t) = 1.80 + 2.10 \cdot \cos\left( \frac{2\pi (t - 11.5)}{12.42} \right) \quad \text{[meters above CD]}$$
- **Deliverable:** Calculate vessel block coefficient $C_B$, midship submerged area $A_{mid}$, blockage factor $S_2$, Barrass dynamic squat $S_k$, required minimum tidal height $Tide_{req}$, and the exact chronological time window during which the vessel is permitted to transit the channel.
- **Acceptance Criteria:** Exact numerical values with all intermediate steps; tidal window stated in clock time.
- **Time Estimate:** 35 minutes.

### Exercise 2: Gross vs. Net Moves and Berth Productivity Audit (Calculation)
- **Prompt:** During an ultra-large container vessel port call, 4 STS cranes performed a combined total of 4,480 container moves. The vessel arrived all-fast at 06:00 on Monday and cast off lines at 22:00 on Tuesday. Crane logs record:
  - Crane 1: 36.0 gross hours, 2.5 hours lashing delay, 1.0 hour twistlock jam.
  - Crane 2: 38.0 gross hours, 3.0 hours hatch cover handling, 1.5 hours boom-up for shifting.
  - Crane 3: 34.0 gross hours, 2.0 hours high wind shutdown, 2.0 hours tractor starvation.
  - Crane 4: 32.0 gross hours, 1.0 hour high wind shutdown, 1.5 hours equipment breakdown.
  Under the terminal service contract, high wind shutdowns, vessel shifting, and hatch cover moves are authorized deductions; lashing delays, twistlock jams, tractor starvation, and crane breakdowns are non-authorized terminal delays.
- **Deliverable:** Calculate total gross crane hours, total net crane hours, overall terminal GMPH, overall terminal NMPH, Berth Moves Per Hour (BMPH), and average Crane Intensity ($CI$).
- **Acceptance Criteria:** Correctly categorize each delay into authorized vs. non-authorized deductions; state formulas and final values to two decimal places.
- **Time Estimate:** 25 minutes.

### Exercise 3: Continuous BAP Space-Time Overlap Formulation (Formulation)
- **Prompt:** Two container vessels are scheduled to berth at a continuous 900-meter quay:
  - Vessel 1: $L_1 = 300$ m, arrival $A_1 = 04:00$, handling duration $H_1 = 20$ hours, safety buffer $s_1 = 20$ m.
  - Vessel 2: $L_2 = 360$ m, arrival $A_2 = 08:00$, handling duration $H_2 = 24$ hours, safety buffer $s_2 = 20$ m.
- **Deliverable:** Write the exact algebraic Big-$M$ mixed-integer linear programming constraints governing the spatial and temporal positioning of these two vessels ($b_1, b_2, t_1, t_2$), including the mutual exclusion constraint. Explain why setting $M$ too large causes numerical instability in MILP solvers.
- **Acceptance Criteria:** Explicit mathematical formulation with defined binary variables $x_{12}, x_{21}, y_{12}, y_{21}$ and quantitative bounds.
- **Time Estimate:** 30 minutes.

### Exercise 4: Quay Crane Non-Crossing Schedule Verification (Application)
- **Prompt:** Three STS cranes ($C_1, C_2, C_3$) run on a single rail track servicing a vessel with 12 container bays (numbered 1 to 12 from bow to stern). The minimum inter-crane clearance is $\delta = 2$ bays. At 14:00, the cranes are stationed at: $C_1$ at Bay 2, $C_2$ at Bay 5, $C_3$ at Bay 9. The terminal supervisor issues the following dispatch orders for 15:00:
  - Order A: Move $C_1$ to Bay 4 to assist with heavy discharge.
  - Order B: Move $C_2$ to Bay 3 to unlash a hatch cover.
  - Order C: Move $C_3$ to Bay 8 to finish on-deck discharge.
- **Deliverable:** Evaluate the feasibility of these dispatch orders against spatial non-crossing and safety clearance rules. If infeasible, identify the exact collision/clearance violation and propose an alternative feasible crane assignment.
- **Acceptance Criteria:** Step-by-step spatial verification showing crane coordinate inequalities over the transition.
- **Time Estimate:** 20 minutes.

### Exercise 5: Vessel Turnaround and Voyage Bunker Fuel Trade-off (Optimization)
- **Prompt:** A container carrier operates an 18,000 TEU vessel with daily asset costs ($OPEX + Charter$) of $96,000/day ($4,000/hour). The vessel faces a 6-hour delay at a congested terminal unless it pays for 2 additional crane gangs at $1,200/crane-hour for 10 hours. If the delay is incurred, the vessel must recover the 6 hours across a 2,400 nm sea transit to meet its next berthing window. The base speed is 18.0 knots. Fuel consumption follows $F(v) = 0.011 \cdot v^{3.2}$ tons/day of VLSFO at $640/ton$.
- **Deliverable:** Compute the net financial outcome: compare the cost of terminal crane gang acceleration versus the bunker speed-up fuel cost required to recover the 6 hours at sea. Recommend the optimal operational decision.
- **Acceptance Criteria:** Complete arithmetic showing base transit time, accelerated speed, fuel delta, and cost comparison.
- **Time Estimate:** 35 minutes.

### Exercise 6: Transverse Metacentric Height ($GM$) Audit (Naval Stability)
- **Prompt:** A container ship has an initial displacement $\Delta_0 = 120,000$ metric tons, initial $KM_0 = 18.50$ m, and initial center of gravity $KG_0 = 17.10$ m. 
  During a 12-hour shift, quay cranes discharge 1,200 heavy import containers (average mass = $24.0$ t each, vertical center of gravity $z_{disch} = 8.50$ m) from below-deck holds, and load 800 light export containers (average mass = $12.0$ t each, vertical center of gravity $z_{load} = 26.00$ m) onto deck tiers. Due to change in draft, the new $KM_1 = 18.85$ m.
- **Deliverable:** Calculate the new displacement $\Delta_1$, the new vertical center of gravity $KG_1$, and the final metacentric height $GM_1$. Evaluate whether the final stability complies with IMO minimum criteria ($GM \ge 0.15$ m) and assess the physical risk to crane operations.
- **Acceptance Criteria:** Full moment balance equations and physical interpretation of the resulting $GM$.
- **Time Estimate:** 30 minutes.

### Exercise 7: Terminal Yard Congestion and Crane Starvation Feedback (Critique)
- **Prompt:** A port authority plans to deepen its berth and install 8 ultra-large STS cranes capable of 35 GMPH each to service 24,000 TEU vessels, but leaves its existing yard equipment (12 diesel RTGs) and gate access unchanged.
- **Deliverable:** Write an operations engineering critique analyzing why this capital investment will fail to achieve the targeted vessel turnaround times. Model the quayside-to-yard feedback dynamics and identify the bottleneck shift.
- **Acceptance Criteria:** Conceptual systems critique applying Little's Law, queuing theory, and apron capacity limits.
- **Time Estimate:** 25 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Hydrodynamic Under-Keel Clearance and Tidal Access
1. **Submerged volume:**
   $$\nabla = \frac{172,000 \text{ t}}{1.025 \text{ t}/m^3} = 167,804.88 \text{ } m^3$$
2. **Block coefficient ($C_B$):**
   $$C_B = \frac{167,804.88}{350.0 \cdot 48.2 \cdot 15.20} = \frac{167,804.88}{256,424.0} = 0.6544$$
3. **Midship area and Blockage factor ($S_2$):**
   $$A_{mid} = 0.98 \cdot B \cdot T = 0.98 \cdot 48.2 \cdot 15.20 = 717.99 \text{ } m^2$$
   $$S_2 = \frac{A_{mid}}{A_{channel}} = \frac{717.99}{2,900} = 0.2476$$
4. **Dynamic Squat ($S_k$):**
   $$S_k = \frac{C_B \cdot S_2^{0.81} \cdot V_k^{2.08}}{30} = \frac{0.6544 \cdot (0.2476)^{0.81} \cdot (9.5)^{2.08}}{30}$$
   - $(0.2476)^{0.81} = 0.3233$
   - $(9.5)^{2.08} = 108.067$
   - $S_k = \frac{0.6544 \cdot 0.3233 \cdot 108.067}{30} = \frac{22.863}{30} = 0.762 \text{ m}$
5. **Required tidal height ($Tide_{req}$):**
   $$D_{req} = T + S_k + UKC_{req} = 15.20 + 0.762 + 1.60 = 17.562 \text{ m}$$
   $$Tide_{req} = 17.562 - 13.80 = 3.762 \text{ m}$$
6. **Tidal Entry Window:**
   $$Tide(t) = 1.80 + 2.10 \cdot \cos\left( \frac{2\pi (t - 11.5)}{12.42} \right) \ge 3.762$$
   $$\cos\left( \frac{2\pi (t - 11.5)}{12.42} \right) \ge \frac{3.762 - 1.80}{2.10} = \frac{1.962}{2.10} = 0.9343$$
   $$\arccos(0.9343) = 0.3649 \text{ radians}$$
   $$t - 11.5 = \pm \frac{0.3649 \cdot 12.42}{2\pi} = \pm 0.721 \text{ hours} \approx \pm 43 \text{ minutes}$$
   The vessel's tidal window is **10:47 to 12:13**.

### Answer to Exercise 2: Gross vs. Net Moves and Berth Productivity Audit
1. **Total gross crane hours:**
   $$H_{gross} = 36.0 + 38.0 + 34.0 + 32.0 = 140.0 \text{ crane-hours}$$
2. **Authorized deductions:**
   - Shifting: $1.5$ hrs (Crane 2)
   - Hatch cover moves: $3.0$ hrs (Crane 2)
   - Wind shutdown: $2.0$ hrs (Crane 3) $+ 1.0$ hr (Crane 4) $= 3.0$ hrs
   - Total authorized deductions $= 1.5 + 3.0 + 3.0 = 7.5 \text{ crane-hours}$
   *(Note: Lashing delays, twistlock jams, tractor starvation, and breakdowns are non-authorized).*
3. **Total net crane hours:**
   $$H_{net} = 140.0 - 7.5 = 132.5 \text{ net crane-hours}$$
4. **Productivity Metrics:**
   - **GMPH:** $\frac{4,480 \text{ moves}}{140.0 \text{ hrs}} = 32.00 \text{ moves/crane-hr}$
   - **NMPH:** $\frac{4,480 \text{ moves}}{132.5 \text{ hrs}} = 33.81 \text{ moves/crane-hr}$
   - Total berth occupancy time: Monday 06:00 to Tuesday 22:00 $= 40.0$ gross berth hours.
   - **BMPH:** $\frac{4,480 \text{ moves}}{40.0 \text{ hrs}} = 112.00 \text{ moves/berth-hr}$
   - **Crane Intensity ($CI$):** $\frac{140.0 \text{ crane-hours}}{40.0 \text{ berth-hours}} = 3.50 \text{ cranes}$

### Answer to Exercise 3: Continuous BAP Space-Time Overlap Formulation
1. **Decision variables:** Berthing coordinates $b_1, b_2 \in [0, 900]$; berthing start times $t_1 \ge 4.0, t_2 \ge 8.0$.
2. **Binary sequence variables:**
   - $x_{12} = 1$ if Vessel 1 is left of Vessel 2 ($b_1 + 300 + 20 \le b_2$).
   - $x_{21} = 1$ if Vessel 2 is left of Vessel 1 ($b_2 + 360 + 20 \le b_1$).
   - $y_{12} = 1$ if Vessel 1 departs before Vessel 2 berths ($t_1 + 20 \le t_2$).
   - $y_{21} = 1$ if Vessel 2 departs before Vessel 1 berths ($t_2 + 24 \le t_1$).
3. **Big-$M$ Constraints:**
   $$b_1 + 320 \le b_2 + M(1 - x_{12})$$
   $$b_2 + 380 \le b_1 + M(1 - x_{21})$$
   $$t_1 + 20 \le t_2 + M(1 - y_{12})$$
   $$t_2 + 24 \le t_1 + M(1 - y_{21})$$
   $$x_{12} + x_{21} + y_{12} + y_{21} \ge 1$$
   $$b_1 \le 900 - 300 = 600, \quad b_2 \le 900 - 360 = 540$$
4. **Numerical Stability:** If $M$ is set to an excessively large value (e.g., $10^9$), floating-point roundoff errors produce matrix ill-conditioning and linear programming relaxation leakage. $M$ should be set tightly to the physical bounds: $M_{space} = L_{quay} = 900$ and $M_{time} = \max(T_{horizon}) \approx 72$.

### Answer to Exercise 4: Quay Crane Non-Crossing Schedule Verification
1. **Initial state at 14:00:** $p_1 = 2, p_2 = 5, p_3 = 9$.
   - $p_2 - p_1 = 5 - 2 = 3 \ge \delta (2)$ [Feasible]
   - $p_3 - p_2 = 9 - 5 = 4 \ge \delta (2)$ [Feasible]
2. **Proposed state at 15:00:** Order A ($p_1 = 4$), Order B ($p_2 = 3$), Order C ($p_3 = 8$).
   - Check relative order: Crane 1 position ($p_1 = 4$) versus Crane 2 position ($p_2 = 3$).
   - Here $p_1 > p_2$, which means **Crane 1 and Crane 2 would have to cross each other** along the rail track. This is physically impossible without a structural crane collision.
   - Furthermore, even if $p_2$ were at 5, moving $C_1$ to 4 violates clearance: $5 - 4 = 1 < \delta(2)$.
3. **Corrective dispatch:** To allow work on Bay 3, Crane 1 must service Bay 3 while Crane 2 remains stationed at Bay 5 or moves to Bay 6 ($p_1 = 3, p_2 = 5, p_3 = 8$). This satisfies $3 + 2 \le 5$ and $5 + 2 \le 8$.

### Answer to Exercise 5: Vessel Turnaround and Voyage Bunker Fuel Trade-off
1. **Option 1: Incur Delay and Speed Up at Sea**
   - Base voyage: $D = 2,400$ nm, $v_0 = 18.0$ knots $\implies T_0 = \frac{2,400}{18.0} = 133.33 \text{ hours} = 5.5556 \text{ days}$.
   - Base fuel burn:
     $$F(18.0) = 0.011 \cdot (18.0)^{3.2} = 0.011 \cdot 10,486.8 = 115.35 \text{ t/day}$$
     $$B_0 = 5.5556 \cdot 115.35 = 640.84 \text{ metric tons}$$
   - Accelerated voyage (recovering 6.0 hours): $T_1 = 133.33 - 6.0 = 127.33 \text{ hours} = 5.3056 \text{ days}$.
     $$v_1 = \frac{2,400}{127.33} = 18.848 \text{ knots}$$
   - Accelerated fuel burn:
     $$F(18.848) = 0.011 \cdot (18.848)^{3.2} = 0.011 \cdot 12,148.9 = 133.64 \text{ t/day}$$
     $$B_1 = 5.3056 \cdot 133.64 = 709.04 \text{ metric tons}$$
   - Extra fuel burned: $\Delta B = 709.04 - 640.84 = 68.20$ metric tons.
   - **Cost of Option 1 (Bunker speed-up):**
     $$\text{Fuel Cost} = 68.20 \cdot \$640 = \$43,648.00$$
2. **Option 2: Pay for Quayside Acceleration**
   - 2 additional crane gangs $\times$ 10 hours $= 20$ gang-hours.
   - Gang cost: $20 \cdot \$1,200 = \$24,000.00$.
   - **Cost of Option 2:** $\$24,000.00$.
3. **Recommendation:** Authorize the quayside crane overtime immediately. It saves the carrier a net **$\$19,648.00$** ($\$43,648 - \$24,000$) while eliminating en-route weather and schedule execution risk.

### Answer to Exercise 6: Transverse Metacentric Height ($GM$) Audit
1. **Initial state:** $\Delta_0 = 120,000$ t, $KG_0 = 17.10$ m. Total initial moment $= 120,000 \cdot 17.10 = 2,052,000 \text{ t}\cdot\text{m}$.
2. **Discharge moves:** $w_{disch} = 1,200 \cdot 24.0 = 28,800$ t at $z_{disch} = 8.50$ m.
   $$\text{Moment removed} = 28,800 \cdot 8.50 = 244,800 \text{ t}\cdot\text{m}$$
3. **Loading moves:** $w_{load} = 800 \cdot 12.0 = 9,600$ t at $z_{load} = 26.00$ m.
   $$\text{Moment added} = 9,600 \cdot 26.00 = 249,600 \text{ t}\cdot\text{m}$$
4. **New displacement ($\Delta_1$):**
   $$\Delta_1 = 120,000 - 28,800 + 9,600 = 100,800 \text{ metric tons}$$
5. **New vertical center of gravity ($KG_1$):**
   $$M_1 = 2,052,000 - 244,800 + 249,600 = 2,056,800 \text{ t}\cdot\text{m}$$
   $$KG_1 = \frac{2,056,800}{100,800} = 20.405 \text{ m}$$
6. **New Metacentric Height ($GM_1$):**
   $$GM_1 = KM_1 - KG_1 = 18.85 - 20.405 = -1.555 \text{ m}$$
7. **Stability Evaluation:** **Severe Negative Stability ($GM = -1.55$ m).** The vessel will develop an immediate angle of loll or capsize alongside the berth. Unloading heavy containers from the bottom of the hold while placing light containers high on deck raises the ship's center of gravity drastically. The terminal must ballast double-bottom tanks simultaneously during discharge to prevent catastrophe.

### Answer to Exercise 7: Terminal Yard Congestion and Crane Starvation Feedback (Rubric)
- **Score 5 (Exemplary):** Demonstrates using Little's Law ($L = \lambda W$) that increasing quayside arrival rate $\lambda$ into the yard without increasing yard crane evacuation capacity or reducing container dwell time $W$ causes yard inventory $L$ to exceed physical capacity. Analyzes how yard saturation triggers internal yard tractor delays, which starves the STS cranes, causing actual quayside productivity to collapse back to the yard's bottleneck capacity. Notes that capital investment in STS cranes without matching yard RTGs and gates produces zero net vessel turnaround improvement.
- **Score 3 (Proficient):** Identifies the yard as the true system bottleneck. Mentions tractor queuing and crane starvation, but lacks quantitative queuing or Little's Law framing.
- **Score 1 (Deficient):** Suggests simply hiring more drivers without recognizing physical stack capacity limits and the structural decoupling between quay and yard.

---

## 14. Interview questions

### Question 1: Continuous vs. Discrete BAP Modeling
- **Prompt:** "When would an operations engineer choose a Continuous Berth Allocation Problem (BAP) formulation over a Discrete BAP formulation, and what computational penalties arise from that choice?"
- **Strong Answer:** A discrete BAP is appropriate only for legacy terminals with physical finger piers or segmented quay walls designed for homogeneous vessel sizes. Modern linear container quays (800m to 2,500m) mandate a continuous BAP because discrete partitioning wastes significant quay capacity when servicing a mixed fleet of 400m ULCVs, 250m feeder ships, and 180m regional vessels. However, continuous BAP transitions the optimization problem from a multi-depot vehicle routing/machine scheduling problem into a 2D space-time packing problem. This introduces continuous space decision variables $b_i$, requiring $4 \cdot \binom{N}{2}$ binary Big-$M$ sequencing constraints to prevent spatial and temporal overlap. The resulting MILP has a significantly weaker LP relaxation bound and exhibits exponential branch-and-bound solve times as vessel call count exceeds 30–40 ships, requiring Lagrangian relaxation, rolling-horizon heuristics, or genetic algorithms in production TOS environments.
- **Weak Answer:** "Continuous BAP is used when you have a long quay wall. Discrete is used when you have separate berths. Continuous is better because it's more accurate, but it's harder for the computer to calculate because there are more numbers."

### Question 2: The Physical Mechanics of Crane Non-Crossing Constraints
- **Prompt:** "Explain how physical non-crossing constraints alter the Quay Crane Scheduling Problem (QCSP) compared to classic parallel machine scheduling."
- **Strong Answer:** In classical parallel machine scheduling, any available machine can be routed to any job without spatial restrictions. In QCSP, all STS cranes are mounted on a shared single pair of rail tracks along the wharf. Consequently, cranes cannot overtake or pass one another. If Crane 1 is assigned to a forward bay and Crane 2 is assigned to an aft bay, Crane 1's position coordinate must always remain strictly less than Crane 2's position coordinate minus a physical safety clearance buffer $\delta$ (typically 35–45 meters). This introduces dynamic spatial coupling: scheduling Crane 2 to work Bay 4 restricts Crane 1 to Bays 1 and 2, even if Bay 8 has an urgent high-workload stack. QCSP cannot be solved as independent parallel machine queues; it requires spatio-temporal precedence graphs where crane travel times and inter-crane spatial blocks directly constrain the makespan.
- **Weak Answer:** "Cranes can't cross because they run on tracks. If one crane tries to go past another crane, they will crash into each other. So the scheduler has to make sure Crane 1 stays on the left and Crane 2 stays on the right."

### Question 3: Crane Intensity vs. Marginal Throughput
- **Prompt:** "Why does assigning 6 STS cranes to an 18,000 TEU vessel rarely yield double the productivity of assigning 3 STS cranes?"
- **Strong Answer:** The marginal throughput of additional quay cranes exhibits diminishing returns governed by three engineering bottlenecks:
  1. **Spatial interference:** As crane density increases along the hull, cranes are forced into adjacent bays, restricting trolley travel speed and requiring boom-up maneuvers to avoid overlapping booms during container handling.
  2. **Quayside apron congestion:** Each additional crane demands 5 to 7 Internal Transfer Vehicles (ITVs) cycling through the wharf portal. At 6 cranes, 30 to 42 trucks are competing for turning space and lane slots beneath the crane legs, causing tractor queuing and crane spreader wait time.
  3. **Workload distribution imbalance:** Container stowage is rarely uniform across all bays; a few "critical bays" contain the majority of moves. Once minor bays are completed, excess cranes must idle because non-crossing constraints prevent them from clustering around the remaining critical bay.
- **Weak Answer:** "Because workers get tired and the cranes get in each other's way. If you put too many cranes on one ship, there are too many trucks on the dock."

### Question 4: Dynamic Squat and Fairway Transit Windows
- **Prompt:** "How does dynamic vessel squat affect the berthing schedule of deep-draft container ships in tidal ports like Hamburg, Antwerp, or Shanghai?"
- **Strong Answer:** Dynamic squat is the hydrodynamic sinkage and trim alteration caused by water acceleration beneath a ship moving in confined or shallow waterways. In a narrow fairway, squat can increase a 20,000 TEU vessel's effective draft by 0.8 to 1.5 meters at 10–12 knots. Consequently, a vessel with a static draft of 15.0m in a 14.0m dredged channel cannot simply wait for a 2.5m astronomical tide to clear its 1.5m UKC policy; it must also account for dynamic squat $S_k$. This narrows the feasible tidal transit window into a tightly bounded chronological slot around high water. If a vessel misses this tidal window due to fog or pilot delay, it must wait 12.42 hours for the next tidal peak at sea, severely disrupting the terminal's Berth Allocation Problem (BAP) and idling committed crane gangs.
- **Weak Answer:** "Squat is when a ship sinks deeper into the water when it goes fast. If the water isn't deep enough, the ship will hit the bottom, so they have to wait for high tide."

### Question 5: Metacentric Height ($GM$) Risks During Quayside Operations
- **Prompt:** "What operational risks arise if a terminal discharges below-deck containers without coordinating ballast water operations with the ship's chief mate?"
- **Strong Answer:** Discharging heavy import containers from below-deck holds removes ballast weight located deep beneath the ship's vertical center of gravity ($KG$), causing $KG$ to rise dramatically. If the rise in $KG$ approaches or exceeds the transverse metacenter height ($KM$), the metacentric height ($GM = KM - KG$) approaches zero or becomes negative. A vessel with negative $GM$ is dynamically unstable and will develop a sudden, uncontrollable list to an "angle of loll." At list angles exceeding 1.0° to 1.5°, container spreaders bind and jam against the ship's vertical steel cell guides, halting crane operations. In extreme cases, sudden listing can part mooring lines, snap crane booms, or capsize the vessel alongside the quay. Terminal stowage plans must be co-simulated with the ship's ballast management plan to maintain $GM \ge 1.0$ m throughout the call.
- **Weak Answer:** "The ship might tilt over. If the ship tilts too much, the containers will fall into the water or get stuck in the cranes."

### Question 6: Gross Moves Per Hour (GMPH) vs. Net Moves Per Hour (NMPH)
- **Prompt:** "If a terminal reports an impressive 36 NMPH on a vessel call, why might the ocean carrier still be furious about the vessel's turnaround time?"
- **Strong Answer:** NMPH excludes all "authorized delay hours" from the denominator. A terminal can log 36 NMPH during the hours the cranes were actually moving boxes, but if the vessel suffered 10 hours of uncoordinated lashing delays, hatch cover staging deadlocks, or crane breakdowns that the terminal recorded as non-operational deductions, the gross productivity (GMPH) might be only 20 moves per hour. Ocean carriers pay charter hire, crew costs, and fuel acceleration penalties based on **total gross berth time**, not net crane time. If BMPH is low due to excessive idle gaps between shifts or poor crane intensity, the carrier's asset is delayed regardless of how fast the crane operated during its active cycles.
- **Weak Answer:** "Because NMPH takes out delays. The cranes might have moved fast when they worked, but there were too many delays, so the ship was still stuck at the port for a long time."

### Question 7: Longitudinal Stress Constraints in Crane Sequencing
- **Prompt:** "Why can't a terminal superintendent assign all 5 cranes to discharge the center bays of a container ship first before moving outward to the bow and stern?"
- **Strong Answer:** Discharging cargo solely from the midship bays while forward and aft holds remain fully laden creates an extreme imbalance between the distributed weight force $w(x)$ and the buoyant upthrust $b(x)$ along the hull. The buoyant center rises while the loaded ends sink, subjecting the hull girder to severe **hogging bending moments** and localized shear forces. If the bending moment exceeds the vessel's permissible still-water bending moment ($M_{permissible}$), the ship will suffer permanent structural deformation, cracked deck coamings, or hull fracture. Crane sequences must be interleaved—alternating between forward, midship, and aft bays—to keep longitudinal shear forces and bending moments within the strict limits certified by the vessel's classification society.
- **Weak Answer:** "Because it will bend the ship. If you empty the middle of the ship, the middle will float up and the ends will go down, which can break the ship in half."

### Question 8: Quayside Decoupling and Automation Architectures
- **Prompt:** "How does an automated terminal architecture (like Rotterdam Maasvlakte II) prevent quayside crane starvation compared to a conventional manned straddle carrier terminal?"
- **Strong Answer:** Conventional terminals couple the crane cycle directly to horizontal yard transport: the crane operator lowers the box directly onto an ITV chassis or waits for a straddle carrier to enter the crane portal. Any delay in the yard immediately propagates to the crane, idling the spreader. Automated terminals decouple these cycles physically and temporally using **double-trolley STS cranes** and dedicated ground buffer zones. The primary seaside trolley hoists from the ship to an elevated portal platform where automated twistlock machines unpin the box. A secondary landside trolley transfers the box from the platform down to an Automated Guided Vehicle (AGV) or ground buffer. Because the seaside trolley never interacts directly with ground vehicles, crane cycle times are completely insulated from yard transport delays.
- **Weak Answer:** "Automated terminals use computer-controlled robots and AGVs instead of human drivers, so they don't take breaks and they don't get stuck in traffic."

---

## 15. Key terms

- **Berth Allocation Problem (BAP):** Optimization problem assigning ships to continuous or discrete berths over time.
- **Quay Crane Assignment Problem (QCAP):** Mathematical determination of crane fleet count assigned to berthed vessels per shift.
- **Quay Crane Scheduling Problem (QCSP):** Detailed scheduling and sequencing of specific bay moves for each assigned crane.
- **Non-Crossing Constraint:** Mathematical restriction preventing rail-mounted gantry cranes from overtaking one another.
- **Ship-to-Shore (STS) Crane:** Rail-mounted gantry crane transferring containers between ship cells and wharf apron.
- **Gross Moves Per Hour (GMPH):** Total container moves divided by total gross crane operating hours.
- **Net Moves Per Hour (NMPH):** Total container moves divided by net crane hours excluding authorized delay deductions.
- **Berth Moves Per Hour (BMPH):** Total container moves divided by gross vessel berth hours from all-fast to lines-cast.
- **Crane Intensity ($CI$):** Average number of quay cranes actively deployed on a vessel throughout its port stay.
- **Bay-Row-Tier (ISO 9711):** 6-digit coordinate system identifying exact container cell locations on board.
- **Hatch Cover:** Heavy steel pontoon sealing below-deck cargo holds, requiring crane handling to access lower cells.
- **Cell Guides:** Vertical steel tracks inside container holds that guide containers into stacked columns.
- **Metacentric Height ($GM$):** Fundamental hydrostatic index of initial transverse ship stability ($KM - KG$).
- **Dynamic Squat:** Downward sinkage and trim change experienced by a ship moving through shallow water.
- **Under-Keel Clearance (UKC):** Minimum vertical margin between the deepest point of the vessel hull and the seabed.
- **Chart Datum (CD):** Bathymetric depth reference plane, standardly defined as Lowest Astronomical Tide (LAT).
- **Pro-Forma Berthing Window:** Contractually agreed weekly arrival day, time, and duration allocated to a liner string.
- **BAPLIE:** Standard EDIFACT message transmitting electronic vessel stowage and bay plans between lines and ports.
- **Hogging:** Longitudinal hull girder deflection where buoyancy exceeds weight amidships (convex curve).
- **Sagging:** Longitudinal hull girder deflection where weight exceeds buoyancy amidships (concave curve).
- **Verified Gross Mass (VGM):** Mandatory SOLAS certified total weight of packed export containers.
- **Internal Transfer Vehicle (ITV):** Terminal yard tractor and chassis hauling containers between quay and yard.
- **Demurrage (Vessel):** Liquidated damages paid by terminal or charterer for exceeding agreed vessel laytime.
- **Laytime:** Contractually stipulated allowable time for a vessel to complete cargo loading and discharging.
- **Angle of Loll:** The static list angle at which a vessel with negative initial $GM$ finds equilibrium stability.

---

## 16. Sources

- **Bierwirth, C., & Meisel, F.** (2010). A survey of berth allocation and quay crane scheduling problems in container terminals. *European Journal of Operational Research*, 202(3), 615–627. https://doi.org/10.1016/j.ejor.2009.05.031
- **Bierwirth, C., & Meisel, F.** (2015). A follow-up survey of berth allocation and quay crane scheduling problems in container terminals. *European Journal of Operational Research*, 244(3), 675–689. https://doi.org/10.1016/j.ejor.2014.12.030
- **Cordeau, J. F., Laporte, G., Legato, P., & Moccia, L.** (2005). Models and tabu search heuristics for the berth-allocation problem. *Transportation Science*, 39(4), 526–538. https://doi.org/10.1287/trsc.1050.0120
- **Kim, K. H., & Park, Y. M.** (2004). A crane scheduling method for port container terminals. *European Journal of Operational Research*, 156(3), 752–768. https://doi.org/10.1016/S0377-2217(03)00133-4
- **International Maritime Organization (IMO).** (2008). *International Code on Intact Stability, 2008 (2008 IS Code)*. Resolution MSC.267(85). London: International Maritime Organization.
- **International Maritime Organization (IMO).** (2014). *Guidelines Regarding the Verified Gross Mass of a Container Carrying Cargo*. MSC.1/Circ.1475. London: International Maritime Organization.
- **PIANC (World Association for Waterborne Transport Infrastructure).** (2014). *Harbour Approach Channels - Design Guidelines*. PIANC Report No. 121. Brussels: PIANC General Secretariat.
- **Steenken, D., Voß, S., & Stahlbock, R.** (2004). Container terminal operation and operations research: a classification and literature review. *OR Spectrum*, 26(1), 3–49. https://doi.org/10.1007/s00291-003-0157-0
- **Barrass, C. B., & Derrett, D. R.** (2012). *Ship Stability for Masters and Mates* (7th ed.). Oxford: Butterworth-Heinemann.
- **Ship Message Design Group (SMDG).** (2020). *SMDG BAPLIE 3.1: Electronic Bayplan Message Implementation Guidelines*. Rotterdam: SMDG.

---

## 17. Links

- **Parent Module:** [Module 41: Maritime, Rail, and Air Cargo Operations Engineering](_index.md)
- **Module Survey File:** [41_maritime_rail_air_operations.md](../../../course/41_maritime_rail_air_operations.md)
- **Sibling Topics:**
  - [Topic 02: Terminal Yard and Gate Operations](02_terminal-yard-gate.md)
  - [Topic 07: Disruption and Recovery in Freight Networks](07_disruption-recovery.md)
  - [Topic 08: Terminal Simulation and Digital Twins](08_terminal-simulation.md)
  - [Topic 10: Automation and Capital Decisions](10_automation-capital.md)
- **Connected Foundation Modules:**
  - [Module 11: Network Design Optimization](../../02_methods/11_network_design_optimization/_index.md)
  - [Module 12: Transportation, Routing, and Last Mile](../../02_methods/12_transportation_routing_last_mile/_index.md)
  - [Module 36: Global Freight, Ports, and Trade Policy](../../06_domain_electives/36_global_freight_ports_trade_policy/_index.md)
