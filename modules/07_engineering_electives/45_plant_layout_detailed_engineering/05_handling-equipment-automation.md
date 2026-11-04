# Handling Equipment, Automation Architectures, and Civil/Structural Interfaces

> Module 45 — Plant Layout, Facility Design, and Detailed Engineering Interfaces
> Topic 05 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 10 (Inventory Optimization & Multi-Echelon), Module 15 (Warehouse Operations), Module 45 Topic 01 (Facility Types and Flow), Module 45 Topic 03 (Capacity and Space Calculation)
> Version: 2026.5

---

## 1. Why this topic exists

In industrial facility engineering, material handling equipment (MHE) and automated storage and retrieval systems (AS/RS) represent the physical bridge between static facility geometry and dynamic logistics velocity. Modern fulfillment centers, cross-docks, and plant warehouses are no longer passive brick-and-mortar storage shells; they are highly integrated, cyber-physical machines. The equipment specified to transport, store, and sort goods dictates not only the operating expense (OpEx) of warehouse labor, but also the civil, structural, electrical, and life-safety specifications of the building envelope itself.

MHE and automation decisions represent capital investments ranging from $15 million for an Autonomous Mobile Robot (AMR) sortation fleet to over $120 million for a high-bay clad-rack AS/RS installation. These decisions are physically and financially irreversible once concrete is cast and structural steel is erected:
- A software routing policy in a Warehouse Management System (WMS) or Warehouse Execution System (WES) can be updated overnight.
- In contrast, altering an automated crane aisle, changing slab flatness tolerances, reinforcing under-designed concrete against punching shear, or adding electrical substation capacity to support fast-charging vehicle fleets incurs catastrophic capital expenditure, prolonged facility shutdown, and structural remediation costs.

When material handling systems are specified in isolation from civil, structural, and electrical engineering, facilities fail at the interfaces:
- Specifying an automated Very Narrow Aisle (VNA) turret truck or high-bay AS/RS crane without specifying defined-traffic concrete floor tolerances ($F_{min}$ per ACI 117 or DIN 15185) leads to mast oscillation. At a 16-meter lift height, a millimeter-scale surface irregularity at the wheel contact patch amplifies into a 40-millimeter lateral mast displacement, causing high-speed collisions with structural rack uprights.
- Specifying automated guided vehicles (AGVs) or heavy counterbalanced lift trucks without analyzing slab punching shear and wheel contact stress leads to floor delamination, subgrade settlement, and structural joint faulting.
- Sizing an AMR fleet using nominal average travel speeds while ignoring acceleration/deceleration trapezoids, turning kinematic envelopes, battery opportunity charging schedules, and intersection deadlock dynamics results in chronic throughput deficits of 25% to 40% during peak operating shifts.
- Installing massive lithium-ion or lead-acid charging zones without engineered thermal management, busbar feeder sizing, and mandatory National Fire Protection Association (NFPA) / Occupational Safety and Health Administration (OSHA) hydrogen dilution ventilation creates catastrophic fire and deflagration hazards that halt facility occupancy certification.

Facilities engineers, industrial architects, and supply chain analytics specialists must master the coupled mechanics of MHE kinematics, automation cycle-time physics, civil slab engineering, and electrical utility sizing. This topic provides the mathematical foundations, governing standards, structural interaction formulas, and design workflows required to engineer material handling automation systems that operate at peak design throughput while maintaining structural integrity and life safety.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Derive** kinematic swept paths, turning radii, and right-angle stacking aisle clearances for manual lift trucks, reach trucks, and VNA guided vehicles in compliance with ISO 3691-4 and OSHA 1910.178.
2. **Formulate and compute** single-command and dual-command cycle times and hourly throughput for Unit-Load AS/RS cranes using the Chebyshev rectilinear travel metric and FEM 9.851 standards.
3. **Model** the kinematics and throughput dynamics of Mini-Load cranes, tier-captive and roaming shuttle systems, and 3D cube storage (AutoStore) grids, including digging depth probability distributions.
4. **Dimension** Autonomous Mobile Robot (AMR) and AGV fleets using closed-form kinetic equations that incorporate loaded travel, empty repositioning, battery charging profiles, and traffic congestion deratings.
5. **Analyze** high-speed sortation kinematics across sliding shoe, cross-belt, and tilt-tray technologies, calculating divert angles, gapping belt physics, and mechanical discharge limits.
6. **Specify** concrete slab flatness ($F_F$) and levelness ($F_L$) per ACI 117 / ASTM E1155 and superflat defined-traffic tolerances ($F_{min}$) per ACI 360R and DIN 15185 to eliminate dynamic mast sway.
7. **Evaluate** structural slab loading, verifying baseplate bearing pressures, critical punching shear per ACI 318, and seismic base shear and building-rack interaction per RMI ANSI MH16.1.
8. **Size** facility electrical charging infrastructure, calculating peak kVA demand, diversity factors, and mandatory hydrogen dilution ventilation airflow rates per OSHA 1910.178 and NFPA 2.

---

## 3. The core mental model

The governing mental model for material handling automation and facility integration is the **Automation-Kinematic-Structural Coupling Triangle**. 

Material handling systems do not exist in a kinematic vacuum. Every mechanical acceleration, vehicle turn, payload transfer, and battery recharge exerts an equal and opposite demand on the civil foundations, structural framing, and electrical utilities of the building. Successful facility design requires simultaneous equilibrium across all three vertices of this coupling triangle.

```text
                             [TOP APEX]
                   MHE & AUTOMATION KINEMATICS
                     (Payloads, Velocities,
                    Accelerations, Cycle Times)
                             /        \
                            /          \
      Dynamic Wheel Loads, /            \  Electrical Demand,
      Mast Sway Moments,  /              \ Fast Charging C-Rates,
      Guide Rail Forces  /                \ Thermal Dissipation,
                        /                  \ Battery Chemistry
                       /                    \
                      /                      \
        [BOTTOM LEFT]                          [BOTTOM RIGHT]
     CIVIL & STRUCTURAL                     ELECTRICAL & UTILITY
         INTERFACES                              INTERFACES
    (Slab Flatness FF/FL,                  (Substation kVA, Switchgear,
   F-min, Punching Shear,                  Busway Drops, NFPA Venting,
  Subgrade k, RMI Seismic)                 Harmonic Distortion THD)
```

### 3.1 Kinematic Interaction: Velocity vs. Slab Precision

When MHE operates at high speeds and elevations, the kinematic system interacts dynamically with the civil floor slab. Consider an AS/RS crane or a VNA turret truck operating at mast height $H$ with wheel track width $W_w$. A minute transverse elevation irregularity $\Delta z$ across the wheel track produces an angular tilt $\theta \approx \frac{\Delta z}{W_w}$. 

At mast tip elevation $H$, this angular tilt produces a static lateral displacement $\delta_{static} = H \cdot \theta = H \cdot \frac{\Delta z}{W_w}$. During horizontal acceleration $a_x$, dynamic inertia amplifies this deflection:

$$\delta_{dynamic} = \delta_{static} \left(1 + \frac{a_x \cdot H}{g \cdot W_w}\right)$$

A vertical floor variation of just $\Delta z = 2.0\text{ mm}$ across a $W_w = 1.5\text{ m}$ wheelbase at $H = 18\text{ m}$ produces a static deflection of $24\text{ mm}$. Under dynamic deceleration, mast tip displacement exceeds $45\text{ mm}$. If operating clearances inside the storage rack are $50\text{ mm}$, the crane or turret mast strikes the structural rack upright, shearing pallet loads, tripping emergency stops, and buckling the storage system. Consequently, higher automation speeds demand mathematically stricter slab tolerances.

### 3.2 The AS/RS Chebyshev Kinematic Plane

For automated storage cranes operating simultaneously in horizontal travel ($X$) and vertical hoist ($Y$), cycle time is governed by the Chebyshev ($L_\infty$) metric, where travel time between two coordinates is the maximum—not the sum—of the independent axis travel times:

```text
 Elevation (Y)
    ^
    |         +---------------------------------------+
    |         |                   . (Target X2, Y2)   |
  H |---------|------------------/--------------------+  <-- Rack Face
    |         |                 /                     |
    |         |                /                      |
    |         |               /                       |
    |         |  Chebyshev   /                        |
    |         |  Isochrone  /                         |
    |         |            /                          |
  0 +---------+-----------+---------------------------+-----> Length (X)
   (I/O)                  Lx                          L
              <----------->
              t_x = t_y (Equalized Travel Boundary)
```

The crane reaches peak operational efficiency when the normalized rack dimensions equalize horizontal travel time $T_x$ and vertical travel time $T_y$. When $T_x \approx T_y$, the crane operates along the diagonal of the Chebyshev space, eliminating idle waiting time on either axis.

### 3.3 AMR Trip Cycle Decomposition

An Autonomous Mobile Robot does not provide 100% productive transport. Sizing an automated fleet requires decomposing the robot's operating timeline into productive transport, kinematic transitions, traffic interference, and thermodynamic recovery (charging):

```text
+---------------------------------------------------------------------------------------+
|                              TOTAL SHIFT DURATION (100%)                              |
+------------------------------------+--------------------------+-----------------------+
|        AVAILABLE OPERATING         |     TRAFFIC & QUEUE      |  BATTERY CHARGING &   |
|            TIME (eta_batt)         |     LOSSES (eta_traffic) |  RECOVERY (1-eta_batt)|
+------------------+-----------------+--------------------------+-----------------------+
|  LOADED TRAVEL   |  EMPTY RETURN   | INTERSECTION CONFLICTS,  | CONSTANT-CURRENT /    |
|   (v_loaded,     |   (v_empty,     | DEADLOCK AVOIDANCE,      | CONSTANT-VOLTAGE      |
|    accel/decel)  |    accel/decel) | QUEUING AT CONVEYORS     | FAST CHARGING (C-RATE)|
+------------------+-----------------+--------------------------+-----------------------+
|  TRANSFER TIME   |  TRANSFER TIME  | CLEAR-PATH SEARCH DELAY, | THERMAL COOLDOWN,     |
|   (PICK / LIFT)  |  (DROP / LOWER) | HUMAN AVOIDANCE SLOWDOWN | BATTERY CELL BALANCING|
+------------------+-----------------+--------------------------+-----------------------+
```

Ignoring any component of this decomposition leads directly to under-fleeting, operational choke points, and failure to meet the facility's peak dispatch volume.

---

## 4. Definitions and vocabulary

| Term | Definition | Common Misuse / Pitfall | Unit / Standard Reference |
|---|---|---|---|
| **Chebyshev Metric ($L_\infty$)** | A distance metric where the distance between two vectors is the greatest of their differences along any coordinate dimension: $d_\infty(p, q) = \max_i(\|p_i - q_i\|)$. | Confusing with Manhattan ($L_1$) distance; AS/RS cranes move in $X$ and $Y$ simultaneously, so total travel time is $\max(t_x, t_y)$, not $t_x + t_y$. | Seconds ($\text{s}$) / FEM 9.851 |
| **FEM 9.851** | Federation Europeenne de la Manutention standard defining calculation principles for cycle times and throughput of storage and retrieval machines. | Assuming cycle time equals rack length divided by top speed; ignores acceleration/deceleration trapezoids and pickup/deposit station dwells. | Standard / FEM 9.851 |
| **Shape Factor ($T$)** | The ratio of crane travel time along the rack length to travel time along the rack height: $T = T_x / T_y$. | Assuming a rack is square in physical meters; a square travel rack has equal travel times ($T=1$), not equal physical dimensions. | Dimensionless ratio |
| **Single-Command Cycle ($T_{sc}$)** | An AS/RS operating cycle consisting of either a single storage transaction or a single retrieval transaction between the I/O station and a rack location. | Believing a single command performs both storage and retrieval; that is a dual command. | Seconds ($\text{s}$) / FEM 9.851 |
| **Dual-Command Cycle ($T_{dc}$)** | An AS/RS operating cycle where the crane deposits an inbound pallet into an empty rack location, travels between locations, retrieves an outbound pallet, and returns to I/O. | Modeling dual-command travel time as twice the single-command time; dual commands eliminate one round-trip transit. | Seconds ($\text{s}$) / FEM 9.851 |
| **Floor Flatness ($F_F$)** | A numerical index evaluating concrete surface waviness and curvature over short 300 mm intervals per ASTM E1155. | Confusing with floor levelness ($F_L$); $F_F$ governs vehicle bumpiness, while $F_L$ governs overall tilt. | Dimensionless / ASTM E1155 |
| **Floor Levelness ($F_L$)** | A numerical index evaluating concrete slab conformance to a horizontal plane over 3-meter spans per ASTM E1155. | Specifying high $F_L$ for elevated suspended slabs where structural deflections make it unachievable without post-pour grinding. | Dimensionless / ASTM E1155 |
| **Superflat Floor ($F_{min}$)** | An absolute profile tolerance standard defined for narrow-aisle defined-traffic paths, measuring elevation differences across wheel tracks. | Applying random-traffic $F_F / F_L$ specs to VNA aisles; VNA turrets require defined-wheel-path $F_{min}$ profiling. | Dimensionless / ACI 360R, DIN 15185 |
| **Punching Shear ($v_u$)** | A localized two-way shear stress failure where a concentrated point load (rack baseplate or MHE wheel) punches through a concrete slab. | Checking only one-way beam shear; punching shear governs thin slabs under high point loads. | Megapascals ($\text{MPa}$) / ACI 318 |
| **Subgrade Modulus ($k$)** | The measure of soil stiffness under slab loading, defined as applied pressure divided by vertical deflection. | Assuming subgrade stiffness is infinite; low $k$ values cause slab flexure, joint faulting, and rack settlement. | $\text{MPa/m}$ or $\text{psi/in}$ (pci) |
| **Post-Tensioned (PT) Slab** | A concrete floor slab reinforced with high-strength prestressed steel tendons, eliminating contraction joints over large floor bays. | Drilling rack anchor bolts into a PT slab without ground-penetrating radar (GPR); severing a live tendon causes explosive failure. | Civil Structural Standard |
| **RMI ANSI MH16.1** | Rack Manufacturers Institute specification governing the design, testing, and utilization of industrial steel storage racks, including seismic loads. | Treating industrial pallet racks as furniture or equipment rather than building-like structural frames subject to seismic codes. | Standard / RMI ANSI MH16.1 |
| **Seismic Response Coefficient ($C_s$)** | The factor multiplied by effective seismic weight to determine total design horizontal base shear: $V = C_s W$. | Ignoring stored inventory mass; rack seismic mass includes 100% dead weight plus a code-mandated fraction of rated live storage. | Dimensionless / ASCE 7 |
| **Tier-Captive Shuttle** | An automated shuttle vehicle dedicated permanently to a single vertical rack storage tier, transferring cartons to an end-of-aisle vertical lift. | Assuming shuttle throughput is bounded by the vehicle; end-of-aisle vertical lifts are the primary throughput bottleneck. | Architecture / MHE |
| **Roaming Shuttle** | An automated shuttle that moves between vertical levels via an integrated lift, reducing vehicle capital cost at the expense of lower velocity. | Specifying roaming shuttles for high-throughput fulfillment where lift queues throttle shuttle utilization. | Architecture / MHE |
| **3D Cube Storage (AutoStore)** | An ultra-dense automated storage architecture where bins are stacked directly on top of each other inside an aluminum grid, retrieved by overhead robots. | Assuming instantaneous bin retrieval; deeper bins require "digging" (relocating top bins), introducing non-linear retrieval delays. | Proprietary Architecture |
| **Dig Depth ($d$)** | The number of non-target inventory bins that must be removed and temporarily restaged to access a requested SKU inside a 3D cube grid. | Assuming uniform retrieval time; dig depth follows a combinatorial distribution dependent on bin velocity slotting. | Integer count of bins |
| **Natural Navigation (SLAM)** | Simultaneous Localization and Mapping; AMR navigation using onboard LiDAR, cameras, and wheel odometry without physical floor infrastructure. | Assuming SLAM works in featureless or dynamic cross-docks; dynamic pallet walls cause laser localization drift. | Navigation Technology |
| **QR Grid Navigation** | AGV/AMR navigation using downward-facing optical cameras reading 2D matrix data codes affixed to the concrete floor on 1-meter centers. | Overlooking floor maintenance; damaged or dirty QR codes cause vehicle emergency stops and fleet traffic jams. | Navigation Technology |
| **Fleet Availability Factor ($\eta_{batt}$)** | The fraction of a working shift an automated vehicle is operational after deducting battery charging, cell balancing, and maintenance. | Assuming robots run 24 hours continuously without modeling battery C-rate charging profiles. | Dimensionless ($0.0 - 1.0$) |
| **Traffic Congestion Factor ($\eta_{traffic}$)** | The operational derating factor reflecting vehicle speed reductions, queuing, and rerouting caused by multi-robot spatial interference. | Sizing AMR fleets assuming empty aisles; at high robot densities, congestion can reduce effective throughput by 15% to 30%. | Dimensionless ($0.0 - 1.0$) |
| **Sliding Shoe Sorter** | A high-speed linear slat conveyor utilizing divert shoes that glide across extruded aluminum slats to push cartons into takeaway spurs. | Operating at divert angles above $30^\circ$ for high-speed parcel sortation, inducing severe carton impact and tumbling. | Kinematic Conveyor System |
| **Cross-Belt Sorter** | A continuous loop conveyor where individual carriers house motorized bi-directional belt segments that actively discharge parcels at high speed. | Ignoring parcel induction synchronization; misaligned belt velocities cause packages to miss divert chutes. | Kinematic Conveyor System |
| **Tilt-Tray Sorter** | A continuous carrier sortation loop where trays tilt mechanically or electro-pneumatically to discharge cartons using gravitational slide. | Sorting parcels with varying friction coefficients without dynamic tilt control, causing light envelopes to stall or fly off. | Kinematic Conveyor System |
| **C-Rate** | The rate of battery charge or discharge normalized against total capacity; $1\text{C}$ charges a battery fully in 1 hour; $2\text{C}$ in 30 minutes. | Specifying $3\text{C}$ fast charging without verifying thermal dissipation and lithium-ion cell degradation limits. | Inverse hours ($\text{hr}^{-1}$) |
| **Opportunity Charging** | The practice of charging automated vehicle batteries during idle periods, shift breaks, or while dwelling at loading/unloading stations. | Expecting full battery recharge during 20-second pallet transfers without high-voltage charging contact shoes. | Power Delivery Strategy |
| **Class I, Division 2** | An NFPA 70 / National Electrical Code (NEC) hazardous location classification where flammable gases (e.g., hydrogen) may be present under abnormal conditions. | Failing to classify lead-acid battery rooms; inadequate ventilation causes explosive hydrogen concentrations above 1% LFL. | NFPA 70 / NEC Article 500 |
| **Lower Flammable Limit (LFL)** | The lowest concentration of a flammable gas in air capable of producing a flash of fire in the presence of an ignition source (4.0% for $H_2$). | Operating battery charging areas without safety margins; OSHA mandates dilution ventilation to keep hydrogen below 25% of LFL (1.0%). | Percentage by volume ($\%$) |

---

## 5. The mechanics, step by step

```text
=============================================================================================================
                           AUTOMATION & CIVIL INTERFACE DESIGN PROCEDURE
=============================================================================================================
 [STEP 1: Kinematic Envelopes & Aisle Geometry]
         |--> Select MHE archetype (Counterbalance, Reach, VNA Turret, AMR, AS/RS)
         |--> Calculate turning radii, chassis swept paths, and OSHA clear clearances
         v
 [STEP 2: AS/RS & Storage Kinematics]
         |--> Determine velocity/acceleration profiles (trapezoidal vs. triangular)
         |--> Apply FEM 9.851 Chebyshev metrics to compute Single (T_sc) & Dual (T_dc) command cycles
         |--> Evaluate Shuttle / AutoStore grid kinetics & digging distributions
         v
 [STEP 3: Fleet Sizing & Traffic Dynamics]
         |--> Decompose mission travel paths (loaded, empty, transfer dwells, turns)
         |--> Apply battery derating (eta_batt) and traffic interference derating (eta_traffic)
         |--> Solve closed-form fleet sizing equation: N = ceil(sum(Q_i * T_trip,i) / (3600 * eta_b * eta_t))
         v
 [STEP 4: Sortation System Kinematics]
         |--> Determine sorter technology (Sliding Shoe, Cross-Belt, Tilt-Tray)
         |--> Calculate divert angles (theta), induction speeds, gapping belt physics, and pitch
         v
 [STEP 5: Civil & Structural Interface Engineering]
         |--> Establish floor flatness/levelness: ACI 117 (FF/FL) and defined-traffic (F_min / DIN 15185)
         |--> Calculate dynamic mast sway: delta = H * (delta_z / W_w) * dynamic_amp
         |--> Check slab structural capacity: Wheel point loads, baseplate bearing, ACI 318 punching shear
         |--> Calculate seismic base shear & building-rack interaction per RMI ANSI MH16.1
         v
 [STEP 6: Power, Charging & Utility Infrastructure]
         |--> Size battery packs, charging C-rates, and opportunity charging schedules
         |--> Calculate peak facility electrical demand (kVA) with diversity factors
         |--> Size OSHA / NFPA hydrogen dilution exhaust ventilation for battery rooms
=============================================================================================================
```

### 5.1 Step 1: MHE Kinematic Envelopes and Aisle Sizing

The physical geometry of a warehouse aisle is dictated by the kinematic envelope of the handling vehicle operating within it. The aisle width must accommodate the vehicle chassis, the front and rear swing overhangs, the physical dimensions of the pallet load, and statutory operating clearances.

For standard counterbalance lift trucks and reach trucks performing a $90^\circ$ right-angle stack, the minimum clear stacking aisle width ($W_a$) is derived from the Pythagorean geometry of the vehicle's turning center:

$$W_a = R_t + \sqrt{(L_{load} + x)^2 + \left(\frac{W_{load}}{2}\right)^2} + C_{clear}$$

Where:
- $R_t$ is the outer turning radius of the vehicle chassis ($\text{m}$).
- $L_{load}$ is the length of the pallet load in the direction of handling ($\text{m}$).
- $W_{load}$ is the width of the pallet load across the forks ($\text{m}$).
- $x$ is the load distance, measured from the center of the front load axle to the front face of the fork vertical shank ($\text{m}$).
- $C_{clear}$ is the total statutory operating clearance margin ($\text{m}$), mandated by OSHA 1910.178 and ISO 3691-4 (typically $150\text{ mm}$ to $250\text{ mm}$ on each side, totaling $300\text{ mm}$ to $500\text{ mm}$).

```text
               +-----------------------------+
               |         PALLET LOAD         | W_load
               +-----------------------------+
                            | x |
     <----------------------O---*------------------------>
     Turning Center         Front Axle
           \                    |
            \                   | Vehicle Chassis
             \                  |
              \                 v
               \__________ R_t (Chassis Turning Radius)
```

For Very Narrow Aisle (VNA) man-up or man-down turret trucks, the vehicle does not execute a $90^\circ$ turn inside the aisle. Instead, the chassis remains stationary along the aisle axis while a trilateral head or rotating fork carriage traverses laterally to access rack faces on either side. The VNA clear aisle width ($W_{vna}$) is strictly linear:

$$W_{vna} = \max\left(W_{truck}, W_{load}\right) + 2 \cdot C_{side}$$

Where:
- $W_{truck}$ is the overall width across the truck chassis or outriggers ($\text{m}$).
- $C_{side}$ is the mechanical side clearance between the widest point of the truck/load and the rack face or pallet overhang ($\text{m}$).
- Under wire guidance (inductive guide wire embedded in a concrete saw-cut at the aisle centerline), $C_{side} \ge 100\text{ mm}$ per side.
- Under structural floor rail guidance (continuous low-profile angle steel anchored to the slab), $C_{side}$ can be compressed to $75\text{ mm}$ to $100\text{ mm}$, yielding clear aisle widths between $1.65\text{ m}$ and $1.85\text{ m}$ for standard $1,200 \times 1,000\text{ mm}$ pallets.

### 5.2 Step 2: AS/RS Kinematics and FEM 9.851 Cycle Time Modeling

Unit-Load Automated Storage and Retrieval Systems (AS/RS) utilize single- or twin-mast rail-guided cranes traveling down long, narrow storage aisles. The crane operates two independent drive mechanisms: horizontal travel ($X$-axis) along a floor-mounted rail and vertical hoist ($Y$-axis) along the structural mast.

#### Kinematic Velocity Profiles
Each axis operates under a trapezoidal velocity-time profile governed by peak velocity ($v$), acceleration ($a$), and jerk ($j$). Assuming constant acceleration (ignoring jerk for primary cycle time formulation per FEM 9.851), the distance required to reach full rated speed is:

$$s_{acc} = \frac{v^2}{2a}, \quad t_{acc} = \frac{v}{a}$$

If the travel distance $D$ exceeds $2 \cdot s_{acc}$, the motion profile is trapezoidal, and total travel time is:

$$t(D) = 2 \cdot t_{acc} + \frac{D - 2 \cdot s_{acc}}{v} = \frac{v}{a} + \frac{D}{v} \quad \left(\text{for } D \ge \frac{v^2}{a}\right)$$

If $D < 2 \cdot s_{acc}$, the crane never achieves peak velocity, resulting in a triangular motion profile:

$$t(D) = 2 \cdot \sqrt{\frac{D}{a}} \quad \left(\text{for } D < \frac{v^2}{a}\right)$$

#### Chebyshev Rack Normalization
Let the storage rack have usable length $L$ and usable height $H$. The full-dimension travel times along each axis are:

$$T_x = t(L), \quad T_y = t(H)$$

Per FEM 9.851 and the classical formulations of Bozer & White (1984), define the normalized rack shape parameter $b$:

$$T_{max} = \max(T_x, T_y), \quad T_{min} = \min(T_x, T_y), \quad b = \frac{T_{min}}{T_{max}} \le 1.0$$

When $b = 1.0$, the rack is kinematically "square," meaning an automated crane requires identical times to traverse the full length or the full height.

#### Single-Command and Dual-Command Travel Times
Assuming randomized storage and retrieval across the rack face, the expected one-way travel time $E[T_{sc\_travel}]$ between the Input/Output (I/O) station located at the bottom corner $(0,0)$ and a uniformly distributed storage location $(X, Y)$ is derived by integrating the Chebyshev metric over the normalized unit rack:

$$E[T_{sc\_travel}] = T_{max} \int_0^1 \int_0^1 \max(x, b \cdot y) \, dx \, dy = T_{max} \left(\frac{1}{3} + \frac{b^2}{30}\right)$$

A Single-Command Cycle ($T_{sc}$) consists of the crane traveling from the I/O station to a rack location, depositing or retrieving a load, and returning empty to the I/O station:

$$T_{sc} = 2 \cdot E[T_{sc\_travel}] + 2 \cdot t_{p/d} = 2 \cdot T_{max} \left(\frac{1}{3} + \frac{b^2}{30}\right) + 2 \cdot t_{p/d}$$

Where $t_{p/d}$ is the time required for the crane's telescopic shuttle forks to extend, deposit/retrieve a pallet, and retract to home position.

A Dual-Command Cycle ($T_{dc}$) consists of:
1. Traveling from I/O station $(0,0)$ to deposit location $(X_1, Y_1)$.
2. Executing deposit ($t_{p/d}$).
3. Traveling between deposit location $(X_1, Y_1)$ and retrieval location $(X_2, Y_2)$ (inter-rack travel).
4. Executing retrieval ($t_{p/d}$).
5. Traveling from retrieval location $(X_2, Y_2)$ back to the I/O station $(0,0)$.
6. Discharging the retrieved load at I/O ($2 \cdot t_{p/d}$ total pickup/deposit dwells at I/O).

The expected travel time between two independent, uniformly distributed random points inside the rack is:

$$E[T_{between}] = T_{max} \left(\frac{1}{3} + \frac{b^2}{6} - \frac{b^3}{30}\right)$$

Therefore, the total expected Dual-Command cycle time is:

$$T_{dc} = 2 \cdot E[T_{sc\_travel}] + E[T_{between}] + 4 \cdot t_{p/d}$$

$$T_{dc} = T_{max} \left(\frac{2}{3} + \frac{b^2}{15} + \frac{1}{3} + \frac{b^2}{6} - \frac{b^3}{30}\right) + 4 \cdot t_{p/d} = T_{max} \left(1 + \frac{7b^2}{30} - \frac{b^3}{30}\right) + 4 \cdot t_{p/d}$$

The single-command hourly throughput ($\lambda_{sc}$) and dual-command hourly throughput ($\lambda_{dc}$) per crane are:

$$\lambda_{sc} = \frac{3600}{T_{sc}} \text{ [pallets/hour]}, \quad \lambda_{dc} = \frac{7200}{T_{dc}} \text{ [pallets/hour] (yielding } \frac{3600}{T_{dc}} \text{ dual cycles/hr)}$$

### 5.3 Step 3: Mini-Load, Shuttle, and 3D Cube Storage Systems

#### Shuttle Systems: Tier-Captive vs. Roaming
In tote and carton handling, traditional Mini-Load cranes are increasingly supplanted by Shuttle Systems.
- **Tier-Captive Shuttles:** Each vertical level contains an independent shuttle traveling at high horizontal acceleration ($a_x \ge 2.0\text{ m/s}^2, v_x \ge 4.0\text{ m/s}$). Carton vertical movement is performed by dedicated end-of-aisle reciprocating lifts. The system throughput is decoupled from aisle length, but strictly bounded by the vertical lift cycle time:

$$\lambda_{shuttle\_system} = \min\left(\sum_{k=1}^K \lambda_{shuttle, k}, \quad \lambda_{lift\_in} + \lambda_{lift\_out}\right)$$

Where a dual-car vertical reciprocating lift with cycle time $t_{lift} = 4.5\text{ s}$ imposes a hard physical ceiling of:

$$\lambda_{max} = \frac{3600}{4.5} = 800\text{ totes/hour per aisle}$$

- **Roaming Shuttles:** Shuttles change tiers via a vertical lift mechanism. Capital equipment cost is lower, but system throughput is heavily penalized by vertical hoist queueing.

#### 3D Cube Storage (AutoStore) Kinematics
In 3D cube storage, inventory totes are stacked directly inside dense vertical cells without operating aisles. A fleet of autonomous robots traverses an aluminum top grid ($X-Y$), hoisting totes vertically through cell columns.

```text
       Top Grid Robots (vx = 3.1 m/s, ax = 0.8 m/s^2)
        [R1] ------>          [R2]
      ===+=====================+===================  <-- Top Grid Deck
         | [Bin 1: Non-Target] | [Bin A]
         | [Bin 2: Non-Target] | [Bin B]
         | [Bin 3: Non-Target] | [Bin C]
         | [Bin 4: TARGET SKU] | [Bin D]
         +---------------------+-------------------+  <-- Concrete Slab
```

If a requested SKU is located at depth $d$ below the surface (where $d = 0$ is the top bin), the robot must perform $d$ "digs"—extracting the $d$ overlying non-target bins, placing them into adjacent vacant cells on the grid, retrieving the target bin, and subsequently restoring the non-target bins.

The expected retrieval time for a bin at depth $d$ inside a grid of height $H_{grid}$ is:

$$T_{retrieve}(d) = t_{travel\_grid} + (2d + 1) \cdot \left[2 \cdot \sqrt{\frac{h_{bin} \cdot d_{avg}}{a_{hoist}}} + t_{clamp}\right]$$

Under randomized inventory slotting in a grid $N_{tiers}$ deep, the probability distribution of dig depth $d$ is uniform: $P(d) = \frac{1}{N_{tiers}}$, yielding an expected dig count of $E[d] = \frac{N_{tiers}-1}{2}$. Under Pareto velocity slotting (ABC stratification where fast-moving A-items are dynamically restored to the top 2 tiers during return cycles), the expected dig depth is compressed:

$$E[d]_{stratified} = \sum_{d=0}^{N_{tiers}-1} d \cdot P(d) \ll \frac{N_{tiers}-1}{2}$$

For a 16-tier AutoStore grid, pure random slotting yields $E[d] = 7.5$ digs per retrieval (severely degrading robot throughput), whereas natural Pareto turnover reduces $E[d]$ to between $1.2$ and $2.1$ digs.

### 5.4 Step 4: AMR and AGV Fleet Sizing and Traffic Dynamics

Determining the required fleet size of Autonomous Mobile Robots ($N_{AMR}$) or Automated Guided Vehicles ($N_{AGV}$) requires balancing total transport work against the effective operating availability of a single robotic unit.

The closed-form governing fleet sizing equation is:

$$N_{AMR} = \left\lceil \frac{\sum_{i=1}^M Q_i \cdot T_{trip, i}}{3600 \cdot \eta_{batt} \cdot \eta_{traffic}} \right\rceil$$

Where:
- $M$ is the total number of distinct transport flow routes.
- $Q_i$ is the peak design material flow demand on route $i$ ($\text{trips/hour}$).
- $T_{trip, i}$ is the total round-trip cycle time for route $i$ ($\text{seconds}$).
- $\eta_{batt}$ is the battery availability factor ($0.0 < \eta_{batt} \le 1.0$).
- $\eta_{traffic}$ is the traffic congestion and routing derating factor ($0.0 < \eta_{traffic} \le 1.0$).

#### Dissection of Trip Cycle Time ($T_{trip, i}$)
The total cycle time for an individual mission decomposes into kinematic, transfer, and queueing components:

$$T_{trip, i} = t_{acc/dec} + \frac{D_{loaded, i}}{v_{loaded}} + t_{drop} + \frac{D_{empty, i}}{v_{empty}} + t_{pick} + t_{turns} + t_{queue}$$

Where:
- $D_{loaded, i}$ and $D_{empty, i}$ are the loaded transport distance and empty repositioning travel distance ($\text{m}$).
- $v_{loaded}$ and $v_{empty}$ are the rated top cruising velocities ($\text{m/s}$). Loaded speed is typically derated by 15% to 25% to prevent load tipping during emergency stops ($a_{estop} \ge 1.5\text{ m/s}^2$).
- $t_{pick}$ and $t_{drop}$ are the physical pallet/cart engagement and disengagement times (docking, pin engagement, hydraulic lift actuation; typically $15\text{ s}$ to $30\text{ s}$).
- $t_{turns}$ is the cumulative rotational time consumed executing in-place pivot turns or curved arcs ($\sum \frac{\Delta \theta}{\omega_{turn}}$).
- $t_{queue}$ is the stochastic waiting time at pickup/deposit conveyor spurs.

#### Modeling the Battery Availability Factor ($\eta_{batt}$)
Under an opportunity fast-charging regime utilizing Lithium Iron Phosphate ($\text{LiFePO}_4$) chemistry:
- The robot operates down to a minimum State of Charge (SoC) of $20\%$ and recharges to $80\%$ SoC.
- The effective usable battery capacity is $\Delta SoC = 0.60 \cdot C_{batt}$ ($\text{Watt-hours}$).
- If average power consumption during travel and idling is $P_{avg}$ ($\text{Watts}$), the run time between charges is:

$$t_{run} = \frac{0.60 \cdot C_{batt}}{P_{avg}} \text{ [hours]}$$

- Charging at a constant-current rate of $R_c$ (C-rate, e.g., $1.5\text{C}$):

$$t_{charge} = \frac{\Delta SoC}{R_c \cdot C_{batt}} + t_{dock\_overhead} = \frac{0.60}{R_c} + t_{dock\_overhead} \text{ [hours]}$$

The theoretical battery availability factor is:

$$\eta_{batt} = \frac{t_{run}}{t_{run} + t_{charge}}$$

For example, if an AMR consumes $400\text{ W}$, carries a $1,200\text{ Wh}$ battery pack, and charges at $1.5\text{C}$ with $0.05\text{ hr}$ docking overhead:
- $t_{run} = (0.60 \cdot 1200) / 400 = 1.80\text{ hours}$.
- $t_{charge} = (0.60 / 1.5) + 0.05 = 0.45\text{ hours}$.
- $\eta_{batt} = 1.80 / (1.80 + 0.45) = 0.80$.

#### Modeling the Traffic Congestion Factor ($\eta_{traffic}$)
The traffic factor $\eta_{traffic}$ accounts for velocity throttling at blind intersections, path rerouting calculated by Conflict-Based Search (CBS) or dynamic window algorithms, and passing delays in narrow aisles. 

$$\eta_{traffic} \approx 1.0 - \kappa \cdot \left(\frac{N_{AMR}}{A_{aisle\_net}}\right)$$

Where $A_{aisle\_net}$ is the net drivable corridor area ($\text{m}^2$) and $\kappa$ is an empirical congestion coefficient ($\approx 0.05 - 0.12\text{ m}^2/\text{robot}$). In dense multi-robot layouts, neglecting $\eta_{traffic}$ results in severe throughput collapse, where adding more robots decreases net delivered moves due to systemic gridlock.

### 5.5 Step 5: High-Speed Sortation Kinematics

Automated sortation systems transfer parcels from a primary induction line to destination chutes at high velocity.

#### Sliding Shoe Sorter Mechanics
In a sliding shoe sorter, divert shoes travel along transverse aluminum slats. When a divert command is issued, a divert switch directs a cluster of shoes across the slat bed at an angle $\theta$, pushing the parcel laterally into a discharge spur.

```text
 Main Sorter Travel Direction (Conveyor Speed v_c) --->
 =========================================================================
     [Slat] [Slat] [Slat] [Slat] [Slat]
      ---    ---    ---    ---    ---
       \      \      \      \      \   <-- Divert Shoes Sliding
        \      \      \      \      \      at Angle theta
         v      v      v      v      v
 ---------------------------------------+
                                         \
                                          \  Discharge Chute
                                           \ (Divert Angle theta)
```

The kinematics of divert are governed by the vector decomposition:
- Longitudinal conveyor velocity: $v_c$ (typically $2.0\text{ m/s}$ to $3.5\text{ m/s}$).
- Transverse divert velocity: $v_d = v_c \cdot \tan \theta$.
- Resultant parcel divert velocity: $v_r = \sqrt{v_c^2 + v_d^2} = \frac{v_c}{\cos \theta}$.

Divert angles are mechanically standardized:
- $\theta = 20^\circ$ for fragile or unstable freight (minimizes impact acceleration).
- $\theta = 30^\circ$ for standard high-velocity parcel sortation.

The lateral acceleration imparted to the package must not exceed the friction limit, or the package will slide uncontrollably across the shoe faces:

$$a_{lateral} \approx \frac{v_d}{\Delta t_{switch}} \le \mu \cdot g$$

Where $\mu$ is the coefficient of static friction between carton corrugate and the urethane shoe face ($\mu \approx 0.35 - 0.50$).

#### Gapping Belt Physics
To ensure that divert shoes can safely reset between consecutive packages, an upstream gapping system (a series of independently driven variable-speed belts) must enforce a dynamic linear gap ($G_{min}$):

$$G_{min} = N_{shoes} \cdot P_{slat} + C_{buffer}$$

Where $N_{shoes} = \lceil \frac{L_{package}}{P_{slat}} \rceil + 2$ is the number of shoes required to stabilize the package, $P_{slat}$ is the slat pitch (typically $127\text{ mm}$ or $5\text{ inches}$), and $C_{buffer}$ is a safety clearance ($\ge 200\text{ mm}$). 

The required gapping ratio between consecutive items entering the sorter determines the maximum theoretical item sortation throughput:

$$\lambda_{sort} = \frac{3600 \cdot v_c}{\bar{L}_{package} + \bar{G}_{min}} \text{ [parcels/hour]}$$

For $v_c = 2.5\text{ m/s}$, average package length $\bar{L} = 0.45\text{ m}$, and average gap $\bar{G} = 0.55\text{ m}$:

$$\lambda_{sort} = \frac{3600 \cdot 2.5}{0.45 + 0.55} = 9,000\text{ parcels/hour}$$

### 5.6 Step 6: Civil and Structural Interface Engineering

Material handling equipment imposes extreme, highly concentrated, repetitive dynamic loads on the building's concrete slab-on-ground and structural framework.

#### Concrete Floor Flatness and Levelness (ACI 117 / ASTM E1155)
Concrete floors are classified into two distinct operational regimes:
1. **Random Traffic Floors:** Vehicles travel unpredictably in all directions (staging aprons, wide aisles). Specified using the Face Floor Profile Numbering system:
   - **Floor Flatness ($F_F$):** Evaluates surface waviness over $300\text{ mm}$ intervals. Higher $F_F$ reduces vehicle pitch and chatter.
   - **Floor Levelness ($F_L$):** Evaluates conformance to an absolute horizontal plane over $3\text{ m}$ spans.
2. **Defined Traffic Floors (VNA Aisles & AS/RS Cranes):** Vehicles are physically or optically constrained to run down identical wheel tracks. Specified using the **$F_{min}$** system (or DIN 15185 in Europe).

```text
                   TRANSLATION OF WHEEL TRACK LEVELNESS
                   INTO TALL MAST DYNAMIC OSCILLATION
                   
                            Mast Tip Displacement
                                   |<- delta ->|
                                   |           |
                                   |   [LOAD]  |
                                   |     |     |
                                   |     |     |
                                   |     |     |
                                   |     |     |
                                   |     |     |
                                   |     |     |
                                   |     |     | Height H = 16 m
                                   |     |     |
                                   |     |     |
                                   |     |     |
                                   |     |     |
                                   |     |     |
                       /===========+===========+===========\
                       |             CHASSIS               |
                       \=====O=======================O=====/
                             |                       |
                             |                       ^ Elevation Diff (delta_z)
                        Wheel Track 1           Wheel Track 2
                        <---------------------------->
                              Wheelbase Width W_w
```

The mathematical relationship governing mast tip deflection ($\delta$) as a function of wheel track transverse elevation difference ($\Delta z$) is:

$$\delta = H \cdot \left(\frac{\Delta z}{W_w}\right) \cdot K_{dyn}$$

Where $K_{dyn}$ is the dynamic magnification factor ($1.5 \le K_{dyn} \le 2.5$) caused by vehicle travel acceleration and mast structural elasticity.

| Floor Classification | Typical Application | ACI 117 $F_F / F_L$ | $F_{min}$ Tolerance | DIN 15185 Elevation Diff ($\Delta z$) |
|---|---|---|---|---|
| **Conventional** | General Bulk Storage, Counterbalance Lift Trucks | $F_F 25 / F_L 20$ | N/A | $\Delta z \le 5.0\text{ mm}$ |
| **Flat** | Reach Trucks, High-Density Drive-in Racking ($H \le 8\text{ m}$) | $F_F 35 / F_L 25$ | N/A | $\Delta z \le 3.5\text{ mm}$ |
| **Superflat Low** | VNA Turret Trucks ($8\text{ m} < H \le 12\text{ m}$) | $F_F 50 / F_L 50$ | $F_{min} 75$ | $\Delta z \le 2.0\text{ mm}$ across $1.5\text{ m}$ track |
| **Superflat High** | VNA Turret Trucks ($H > 12\text{ m}$), AS/RS Bottom Rails | $F_F 65 / F_L 60$ | $F_{min} 100$ | $\Delta z \le 1.0\text{ mm}$ across $1.5\text{ m}$ track |

#### Slab Punching Shear (ACI 318)
High-bay pallet rack uprights and heavy AS/RS crane columns transfer massive point loads directly into the concrete floor slab through steel baseplates. The governing failure mode is two-way **punching shear**, where a truncated pyramid of concrete shears around the baseplate perimeter.

Per ACI 318-19, the two-way nominal shear strength ($V_c$) of an unreinforced concrete slab without shear reinforcement is:

$$V_c = 0.33 \cdot \lambda \cdot \sqrt{f'_c} \cdot b_0 \cdot d \text{ [Newtons]}$$

Where:
- $\lambda = 1.0$ for normal-weight concrete.
- $f'_c$ is the specified 28-day concrete compressive cylinder strength ($\text{MPa}$).
- $d$ is the effective structural depth of the slab ($\text{mm}$), typically $d \approx h_{slab} - 50\text{ mm}$ (accounting for concrete cover).
- $b_0$ is the critical shear perimeter ($\text{mm}$), evaluated at a distance of $d/2$ from the perimeter of the concentrated load:

$$b_0 = 4 \cdot (c + d)$$

For a square baseplate of side dimension $c$.

The design punching shear capacity must satisfy:

$$\phi V_c \ge P_u$$

Where $\phi = 0.75$ is the ACI strength reduction factor for shear, and $P_u$ is the factored ultimate concentrated column point load ($P_u = 1.2 D + 1.6 L$). If $\phi V_c < P_u$, the engineer must increase baseplate dimensions ($c$), increase slab thickness ($h$), or integrate post-tensioning tendons.

#### Seismic Tie-In Engineering (RMI ANSI MH16.1)
Storage racks are free-standing steel structures governed by RMI ANSI MH16.1 and ASCE 7. The total horizontal seismic design base shear ($V$) acting on a storage rack frame is:

$$V = C_s \cdot W_{seismic}$$

Where:
- $C_s$ is the seismic response coefficient:

$$C_s = \frac{S_{DS}}{\left(\frac{R}{I_e}\right)}$$

- $S_{DS}$ is the short-period design spectral acceleration parameter.
- $R$ is the response modification coefficient (typically $R = 4.0$ for standard pallet rack frames braced in the cross-aisle direction).
- $I_e$ is the seismic importance factor ($I_e = 1.0$ for standard warehouses; $I_e = 1.25$ or $1.50$ for hazardous materials or essential facilities).
- $W_{seismic}$ is the effective seismic dead load:

$$W_{seismic} = W_{rack\_steel} + 0.67 \cdot W_{product\_rated}$$

RMI MH16.1 mandates an operational product load reduction factor of $0.67$ to reflect the statistical reality that warehouses are rarely 100% full of maximum-weight pallets at the instant of a seismic event.

> **Note on Building-Rack Interaction:** Never structurally tie high-bay pallet racks to the building roof purlins or building structural columns without a slip-joint interface. Pre-engineered metal buildings (PEMBs) have high lateral drift flexibility, whereas industrial steel storage racks are extremely stiff. A rigid tie will cause the swaying building roof to pull or push down on the rack frames during wind or seismic events, inducing buckling and catastrophic progressive structural collapse.

### 5.7 Step 7: Power, Charging Utility Loads, and Electrical Infrastructure

The electrification of MHE fleets requires massive low-voltage and medium-voltage electrical utility capacity.

#### Fleet Power Demand Sizing
For an automated fleet operating $N_{chargers}$ fast chargers, each rated at charging power $P_{charger}$ ($\text{kW}$), total facility connected electrical load is:

$$P_{connected} = \sum_{j=1}^{N_{chargers}} P_{charger, j}$$

Because not all vehicles charge simultaneously, the design diversified electrical load ($S_{design}$) in $\text{kVA}$ is:

$$S_{design} = \frac{P_{connected} \cdot K_{diversity}}{\text{PF} \cdot \eta_{charger}}$$

Where:
- $K_{diversity}$ is the charging coincidence factor ($0.70 \le K_{diversity} \le 0.85$ for opportunity charging).
- $\text{PF}$ is the operating power factor of the switch-mode rectifier ($\text{PF} \approx 0.95$).
- $\eta_{charger}$ is the electrical conversion efficiency ($0.90 \le \eta_{charger} \le 0.94$).

#### Battery Room Safety & Hydrogen Dilution Ventilation
In facilities utilizing flooded lead-acid battery chemistries, water electrolysis during the end-of-charge overcharge phase releases explosive hydrogen gas ($H_2$).

Per OSHA 1910.178(g) and NFPA 2, the volumetric hydrogen evolution rate ($V_{H2}$) is calculated as:

$$V_{H2} = N_{cells} \cdot I_{finish} \cdot 0.000418 \text{ [m}^3/\text{hour]}$$

Where $N_{cells}$ is the total number of battery cells on charge simultaneously, and $I_{finish}$ is the finish charging current ($\text{Amperes}$).

To ensure the hydrogen concentration in the battery charging room never exceeds $1.0\%$ (which is $25\%$ of the Lower Flammable Limit of $4.0\%$), the minimum continuous mechanical exhaust airflow rate ($Q_{vent}$) is:

$$Q_{vent} = \frac{V_{H2}}{0.01} = 100 \cdot V_{H2} \text{ [m}^3/\text{hour]}$$

The exhaust system must be interlocked with hydrogen sensor gas detection alarms and wired to spark-proof Class I, Division 2 electrical motors.

---

## 6. Worked example

### Problem Statement
A high-throughput omni-channel distribution center (80,000 $\text{m}^2$) requires the complete kinematic, structural, and utility design of an automated inbound and sortation interface. The design encompasses four coupled engineering tasks:
1. **AS/RS Unit-Load Crane Throughput:** Calculate single- and dual-command cycle times and hourly crane throughput per FEM 9.851.
2. **AMR Sortation Fleet Sizing:** Size an autonomous mobile robot fleet to transfer pallets from AS/RS decant spurs to sortation induction stations.
3. **Civil Slab Punching Shear & Seismic Base Shear:** Verify concrete slab punching shear capacity for the AS/RS rack baseplates per ACI 318, and calculate seismic base shear per RMI ANSI MH16.1.
4. **Power & Ventilation Sizing:** Determine electrical substation demand and battery room hydrogen dilution airflow.

---

### Part A: AS/RS Crane Cycle Time Formulation (FEM 9.851)

#### Given Parameters
- Rack length: $L = 90.0\text{ m}$
- Rack height: $H = 30.0\text{ m}$
- Horizontal drive speed: $v_x = 4.0\text{ m/s}$, acceleration $a_x = 0.8\text{ m/s}^2$
- Vertical hoist speed: $v_y = 1.0\text{ m/s}$, acceleration $a_y = 0.6\text{ m/s}^2$
- Pickup/Deposit telescopic fork stroke time: $t_{p/d} = 12.0\text{ s}$

#### Step A1: Acceleration Kinematics
Calculate acceleration times and distances for both axes:

$$t_{ax} = \frac{v_x}{a_x} = \frac{4.0}{0.8} = 5.0\text{ s}, \quad s_{ax} = \frac{1}{2} a_x t_{ax}^2 = 0.5 \cdot 0.8 \cdot 25.0 = 10.0\text{ m}$$

Since $L = 90.0\text{ m} > 2 \cdot s_{ax} = 20.0\text{ m}$, the horizontal profile is trapezoidal.

$$t_{ay} = \frac{v_y}{a_y} = \frac{1.0}{0.6} = 1.6667\text{ s}, \quad s_{ay} = \frac{1}{2} a_y t_{ay}^2 = 0.5 \cdot 0.6 \cdot (1.6667)^2 = 0.8333\text{ m}$$

Since $H = 30.0\text{ m} > 2 \cdot s_{ay} = 1.6667\text{ m}$, the vertical profile is trapezoidal.

#### Step A2: Full-Dimension Travel Times & Shape Factor
Calculate full-span transit times:

$$T_x = 2 \cdot 5.0 + \frac{90.0 - 20.0}{4.0} = 27.50\text{ s}, \quad T_y = 2 \cdot 1.6667 + \frac{30.0 - 1.6667}{1.0} = 31.6667\text{ s}$$

Identify maximum travel time and normalized shape parameter $b$:

$$T_{max} = T_y = 31.6667\text{ s}, \quad T_{min} = T_x = 27.50\text{ s}, \quad b = \frac{T_{min}}{T_{max}} = \frac{27.50}{31.6667} = 0.86842$$

#### Step A3: Expected Travel Times
Compute expected one-way travel time $E[T_{sc\_travel}]$:

$$E[T_{sc\_travel}] = T_{max} \left(\frac{1}{3} + \frac{b^2}{30}\right) = 31.6667 \cdot (0.33333 + 0.02514) = 11.3515\text{ s}$$

Compute expected travel time between two random points $E[T_{between}]$:

$$E[T_{between}] = T_{max} \left(\frac{1}{3} + \frac{b^2}{6} - \frac{b^3}{30}\right) = 31.6667 \cdot (0.33333 + 0.12569 - 0.02183) = 13.8444\text{ s}$$

#### Step A4: Cycle Times and Throughput
Single-Command Cycle Time ($T_{sc}$):

$$T_{sc} = 2 \cdot E[T_{sc\_travel}] + 2 \cdot t_{p/d} = 2 \cdot (11.3515) + 2 \cdot (12.0) = 22.703 + 24.0 = 46.703\text{ s}$$

$$\lambda_{sc} = \frac{3600}{46.703} = 77.08\text{ pallets/hour}$$

Dual-Command Cycle Time ($T_{dc}$):

$$T_{dc} = 2 \cdot E[T_{sc\_travel}] + E[T_{between}] + 4 \cdot t_{p/d}$$

$$T_{dc} = 22.703 + 13.8444 + 4 \cdot (12.0) = 36.5474 + 48.0 = 84.547\text{ s}$$

Dual-command cycles per hour:

$$\text{Dual Cycles/hr} = \frac{3600}{84.547} = 42.58\text{ dual cycles/hour}$$

Total dual-command pallet transactions (1 store + 1 retrieve per cycle):

$$\lambda_{dc} = 2 \cdot 42.58 = 85.16\text{ pallets/hour}$$

---

### Part B: AMR Fleet Sizing for Sortation Inbound

#### Given Parameters
The AMR fleet must transfer pallets across 3 zones to sortation induction:
- **Zone 1:** Demand $Q_1 = 600\text{ moves/hr}$, $D_{loaded, 1} = 80\text{ m}$, $D_{empty, 1} = 70\text{ m}$
- **Zone 2:** Demand $Q_2 = 700\text{ moves/hr}$, $D_{loaded, 2} = 120\text{ m}$, $D_{empty, 2} = 110\text{ m}$
- **Zone 3:** Demand $Q_3 = 500\text{ moves/hr}$, $D_{loaded, 3} = 150\text{ m}$, $D_{empty, 3} = 140\text{ m}$
- Total demand: $Q_{total} = 1,800\text{ moves/hour}$
- Loaded cruising velocity: $v_{loaded} = 1.2\text{ m/s}$
- Empty repositioning velocity: $v_{empty} = 1.5\text{ m/s}$
- Fixed dwell times per trip:
  - $t_{pick} = 15.0\text{ s}$
  - $t_{drop} = 15.0\text{ s}$
  - $t_{turns} = 12.0\text{ s}$
  - $t_{queue} = 8.0\text{ s}$
  - Total fixed dwell: $t_{fixed} = 15 + 15 + 12 + 8 = 50.0\text{ s}$
- Battery availability factor: $\eta_{batt} = 0.85$
- Traffic congestion derating: $\eta_{traffic} = 0.88$

#### Step B1: Trip Times per Zone
Calculate round-trip time for each zone:

$$T_{trip, 1} = 50.0 + \frac{80}{1.2} + \frac{70}{1.5} = 163.33\text{ s}$$
$$T_{trip, 2} = 50.0 + \frac{120}{1.2} + \frac{110}{1.5} = 223.33\text{ s}$$
$$T_{trip, 3} = 50.0 + \frac{150}{1.2} + \frac{140}{1.5} = 268.33\text{ s}$$

#### Step B2: Total Hourly Transport Work
Calculate total required transport work in robot-seconds per hour:

$$W_{total} = \sum_{i=1}^3 Q_i \cdot T_{trip, i}$$

$$W_1 = 600 \cdot 163.333 = 98,000.0\text{ s}, \quad W_2 = 700 \cdot 223.333 = 156,333.3\text{ s}, \quad W_3 = 500 \cdot 268.333 = 134,166.7\text{ s}$$
$$W_{total} = 98,000.0 + 156,333.3 + 134,166.7 = 388,500.0\text{ robot-seconds/hour}$$

#### Step B3: Effective Operating Capacity per Robot
Calculate productive seconds available per AMR per hour:

$$\text{Cap}_{eff} = 3600 \cdot \eta_{batt} \cdot \eta_{traffic} = 3600 \cdot 0.85 \cdot 0.88 = 2,692.8\text{ seconds/AMR}$$

#### Step B4: AMR Fleet Size Sizing

$$N_{AMR\_exact} = \frac{W_{total}}{\text{Cap}_{eff}} = \frac{388,500.0}{2,692.8} = 144.27\text{ AMRs}$$

Rounding up to the nearest integer:

$$N_{AMR} = \lceil 144.27 \rceil = 145\text{ AMRs}$$

---

### Part C: Civil Slab Punching Shear & RMI Seismic Engineering

#### Given Parameters
- AS/RS rack upright factored ultimate load: $P_u = 240.0\text{ kN} = 240,000\text{ N}$
- Upright baseplate dimensions: $c_1 = 250\text{ mm}, c_2 = 250\text{ mm}$
- Concrete slab thickness: $h = 250\text{ mm}$
- Effective structural depth: $d = 200.0\text{ mm}$
- 28-day compressive strength: $f'_c = 35.0\text{ MPa}$
- ACI shear strength reduction factor: $\phi = 0.75$
- Seismic parameters: $S_{DS} = 0.75\text{ g}$, $R = 4.0$, $I_e = 1.25$
- Rack self-weight: $W_{rack} = 15.0\text{ kN}$ per bay
- Maximum rated pallet load: $W_{product} = 240.0\text{ kN}$ per bay (12 tiers @ $20\text{ kN/tier}$)

#### Step C1: Punching Shear Capacity Check (ACI 318)
Critical shear perimeter $b_0$ located at $d/2 = 100\text{ mm}$ from baseplate face:

$$b_0 = 4 \cdot (c + d) = 4 \cdot (250.0 + 200.0) = 4 \cdot 450.0 = 1,800.0\text{ mm}$$

Calculate nominal concrete punching shear strength $V_c$ (with $\lambda = 1.0$):

$$V_c = 0.33 \cdot 1.0 \cdot \sqrt{35.0} \cdot 1,800.0 \cdot 200.0 = 0.33 \cdot 5.9161 \cdot 360,000 = 702,831\text{ N} = 702.83\text{ kN}$$

Apply the ACI shear reduction factor $\phi = 0.75$:

$$\phi V_c = 0.75 \cdot 702.83\text{ kN} = 527.12\text{ kN}$$

Evaluate Factor of Safety ($FS$):

$$FS = \frac{\phi V_c}{P_u} = \frac{527.12\text{ kN}}{240.00\text{ kN}} = 2.20 \ge 1.00 \quad \text{[ADEQUATE - PASSED]}$$

#### Step C2: Seismic Base Shear Calculation (RMI ANSI MH16.1)
Calculate the seismic response coefficient $C_s$:

$$C_s = \frac{S_{DS}}{\left(\frac{R}{I_e}\right)} = \frac{0.75}{\left(\frac{4.0}{1.25}\right)} = \frac{0.75}{3.20} = 0.234375$$

Calculate effective seismic weight $W_{seismic}$ per bay (incorporating the $0.67$ product load factor):

$$W_{seismic} = W_{rack} + 0.67 \cdot W_{product} = 15.0 + 0.67 \cdot 240.0 = 175.80\text{ kN}$$
$$V = C_s \cdot W_{seismic} = 0.234375 \cdot 175.80\text{ kN} = 41.20\text{ kN}$$

Each rack bay frame and anchoring system must resist $41.20\text{ kN}$ of horizontal base shear at the slab interface.

---

### Part D: Power, Charging Infrastructure, and Hydrogen Ventilation

#### Step D1: AMR Fleet Charging Utility Demand
- Fleet size: $N_{AMR} = 145$ units.
- Battery capacity: $1,200\text{ Wh}$, operating on a $1.5\text{C}$ fast charge ($1.8\text{ kW}$ DC per charger).
- Charging ratio: Fleet requires 1 charger per 5 AMRs $\to N_{chargers} = \lceil 145 / 5 \rceil = 29$ fast-charging stations.
- Charger rectifier efficiency: $\eta_{charger} = 0.92$, operating power factor $\text{PF} = 0.95$.
- Connected charging power: $P_{connected} = 29 \cdot 1.8\text{ kW} = 52.2\text{ kW}$.
- Diversity coincidence factor: $K_{diversity} = 0.80$.

Diversified charging apparent power ($S_{AMR}$):

$$S_{AMR} = \frac{P_{connected} \cdot K_{diversity}}{\text{PF} \cdot \eta_{charger}} = \frac{52.2 \cdot 0.80}{0.95 \cdot 0.92} = \frac{41.76}{0.874} = 47.78\text{ kVA}$$

#### Step D2: Auxiliary Forklift Battery Room Ventilation (OSHA 1910.178)
In addition to the AMR fleet, the cross-dock maintains 20 flooded lead-acid counterbalanced lift trucks charging in an enclosed maintenance room:
- Total battery cells charging simultaneously: $N_{cells} = 20\text{ trucks} \times 24\text{ cells/truck} = 480\text{ cells}$.
- Finish charging current: $I_{finish} = 25\text{ Amperes}$.

Volumetric rate of hydrogen generation ($V_{H2}$):

$$V_{H2} = N_{cells} \cdot I_{finish} \cdot 0.000418 = 480 \cdot 25 \cdot 0.000418 = 5.016\text{ m}^3/\text{hour}$$

Continuous dilution exhaust ventilation airflow required to maintain $H_2 \le 1.0\%$ ($25\%$ of LFL):

$$Q_{vent} = \frac{V_{H2}}{0.01} = 501.6\text{ m}^3/\text{hour} \approx 295.2\text{ CFM (Cubic Feet per Minute)}$$

The design must specify a dedicated Class I, Division 2 spark-proof exhaust fan delivering at least $350\text{ CFM}$ with differential pressure sensor interlocks to the battery chargers.

---

## 7. Code, query, or calculation

The following Python script implements the mathematical formulas for FEM 9.851 AS/RS cycle times, AMR fleet sizing with congestion/battery deratings, and ACI 318 slab punching shear verification.

```python
import math

def calculate_asrs_cycle(L, H, vx, ax, vy, ay, t_pd):
    tax, tay = vx / ax, vy / ay
    sax, say = 0.5 * ax * tax**2, 0.5 * ay * tay**2
    Tx = 2 * tax + (L - 2 * sax) / vx
    Ty = 2 * tay + (H - 2 * say) / vy
    T_max, T_min = max(Tx, Ty), min(Tx, Ty)
    b = T_min / T_max
    E_one_way = T_max * (1.0 / 3.0 + (b**2) / 30.0)
    E_between = T_max * (1.0 / 3.0 + (b**2) / 6.0 - (b**3) / 30.0)
    T_sc = 2.0 * E_one_way + 2.0 * t_pd
    T_dc = 2.0 * E_one_way + E_between + 4.0 * t_pd
    return {"Tx": Tx, "Ty": Ty, "b": b, "T_sc": T_sc, "T_dc": T_dc,
            "Thru_sc": 3600.0 / T_sc, "Thru_dc": 7200.0 / T_dc}

def size_amr_fleet(demands, d_loaded, d_empty, vl, ve, t_fixed, eta_batt, eta_traffic):
    total_work_sec = 0.0
    for q, dl, de in zip(demands, d_loaded, d_empty):
        t_trip = t_fixed + (dl / vl) + (de / ve)
        total_work_sec += q * t_trip
    sec_per_robot = 3600.0 * eta_batt * eta_traffic
    n_amr_exact = total_work_sec / sec_per_robot
    return math.ceil(n_amr_exact), n_amr_exact, total_work_sec

def check_punching_shear(Pu_kN, c_mm, d_mm, fc_MPa, phi=0.75):
    b0 = 4.0 * (c_mm + d_mm)
    Vc_N = 0.33 * 1.0 * math.sqrt(fc_MPa) * b0 * d_mm
    phi_Vc_kN = (phi * Vc_N) / 1000.0
    fs = phi_Vc_kN / Pu_kN
    return {"b0_mm": b0, "phi_Vc_kN": phi_Vc_kN, "FS": fs, "adequate": fs >= 1.0}

if __name__ == "__main__":
    asrs = calculate_asrs_cycle(90.0, 30.0, 4.0, 0.8, 1.0, 0.6, 12.0)
    print(f"AS/RS FEM 9.851: Tx={asrs['Tx']:.1f}s, Ty={asrs['Ty']:.1f}s, b={asrs['b']:.4f}")
    print(f"Single Command: {asrs['T_sc']:.2f}s -> {asrs['Thru_sc']:.1f} pallets/hr")
    print(f"Dual Command:   {asrs['T_dc']:.2f}s -> {asrs['Thru_dc']:.1f} pallets/hr (42.6 cycles/hr)")

    n_ceil, n_exact, tot_sec = size_amr_fleet(
        [600, 700, 500], [80, 120, 150], [70, 110, 140], 1.2, 1.5, 50.0, 0.85, 0.88
    )
    print(f"AMR Fleet: Required Fleet Size = {n_ceil} AMRs (Exact: {n_exact:.2f})")

    shear = check_punching_shear(240.0, 250.0, 200.0, 35.0)
    print(f"Punching Shear: phi*Vc = {shear['phi_Vc_kN']:.2f} kN vs Pu = 240.00 kN (FS={shear['FS']:.2f})")
```

### Assumptions and Limits of the Script
1. The AS/RS model assumes uniform, randomized rack storage access and constant acceleration trapezoids (ignoring jerk).
2. The AMR fleet calculation assumes steady-state arrival rates with deterministic routing; stochastic queueing peaks may require discrete-event simulation validation.
3. The punching shear check follows ACI 318 for interior square baseplates without shear reinforcement or moment transfer.

### Expected Terminal Output
```text
AS/RS FEM 9.851: Tx=27.5s, Ty=31.7s, b=0.8684
Single Command: 46.70s -> 77.1 pallets/hr
Dual Command:   84.55s -> 85.2 pallets/hr (42.6 cycles/hr)
AMR Fleet: Required Fleet Size = 145 AMRs (Exact: 144.27)
Punching Shear: phi*Vc = 527.12 kN vs Pu = 240.00 kN (FS=2.20)
```

---

## 8. Data, standards, and regulatory context

Material handling automation and civil engineering interfaces are governed by an interlocking hierarchy of international standards, building codes, and safety regulations.

### 8.1 Material Handling & Automation Standards

#### FEM 9.851 & FEM 9.831 / 9.832
The Federation Europeenne de la Manutention (FEM) establishes binding engineering methods for storage and retrieval machines:
- **FEM 9.851:** Defines the standardized calculation principles for AS/RS cycle times and throughput, establishing the 5-point representative cycle and the continuous Chebyshev integral formulation used globally by MHE suppliers.
- **FEM 9.831 / 9.832:** Governs structural tolerances, permissible deflections, and clearances for AS/RS cranes and racking interfaces. It mandates that maximum mast deflection under dynamic travel acceleration must not exceed $H / 2000$ to prevent rack collisions.

#### ISO 3691-4:2023 (Driverless Industrial Trucks)
ISO 3691-4 is the premier international safety standard for AGVs and AMRs. It mandates:
- Personnel detection safety laser scanners (PL-d / Category 3 per ISO 13849-1) configured with dynamic protective fields that automatically scale with vehicle velocity and steering angle.
- Minimum clear passage margins: A minimum physical clearance of $500\text{ mm}$ between the vehicle boundary and fixed building structures for human travel; if clearance falls below $500\text{ mm}$ (confined operating zones), the vehicle must restrict speed to $\le 0.3\text{ m/s}$ and sound continuous acoustic alarms.

### 8.2 Civil, Structural & Concrete Standards

#### ACI 117 / ASTM E1155 & ACI 360R
The American Concrete Institute (ACI) and ASTM govern floor flatness and levelness:
- **ASTM E1155:** Specifies the standard test method for determining Floor Flatness ($F_F$) and Floor Levelness ($F_L$) numbers using an inclinometer (e.g., Face Dipstick) along orthogonal measurement lines within 72 hours of concrete placement.
- **ACI 360R (Guide to Design of Slabs-on-Ground):** Details structural slab thickness design under concentrated wheel point loads, joint spacing to prevent shrinkage curling, and post-tensioning tendon layouts.
- **DIN 15185 (Part 1 & 2):** The German industrial standard governing tolerances for Very Narrow Aisle (VNA) warehouses. It specifies maximum permissible transverse elevation differences across wheel tracks ($\Delta z \le 1.0\text{ mm}$ for lift heights exceeding $12\text{ m}$) measured with a continuous profiling crawler.

#### RMI ANSI MH16.1 (Industrial Steel Storage Racks)
The Rack Manufacturers Institute (RMI) standard, incorporated by reference into the International Building Code (IBC) and NFPA 5000:
- Establishes the structural design, beam-column connection testing, and baseplate anchorage of pallet racks.
- Mandates that all rack columns be anchored to the concrete slab with anchor bolts capable of resisting prescribed base shear and uplift forces.
- Defines the seismic mass formulation ($W_{seismic} = W_{dead} + 0.67 W_{live}$), recognizing dynamic pallet sliding friction during seismic events.

### 8.3 Life Safety & Electrical Standards

#### OSHA 29 CFR 1910.178 & NFPA 13
- **OSHA 1910.178:** Federal standard governing powered industrial trucks, operator training, aisle clearance maintenance, and battery charging room requirements (eyewash stations, fire protection, and mandatory mechanical ventilation).
- **NFPA 13 (Installation of Sprinkler Systems):** Dictates mandatory vertical clearances below fire sprinklers (minimum $457\text{ mm}$ / $18\text{ inches}$ for standard spray heads; $914\text{ mm}$ / $36\text{ inches}$ for ESFR sprinklers) and transverse/longitudinal flue space alignment ($75\text{ mm}$ to $150\text{ mm}$) within automated rack structures.

#### NFPA 2 & NFPA 70 (National Electrical Code)
- **NFPA 2 (Hydrogen Technologies Code) & NFPA 70 Article 500:** Classifies areas where hydrogen concentrations can exceed $1.0\%$ as Class I, Division 2 hazardous locations, requiring explosion-proof motors, non-sparking exhaust fans, and gas-tight conduit seals.

---

## 9. Industry practice and case evidence

### 9.1 Amazon Kiva / Proteus AMR Fleets: Floor Dynamics and Wear
In the mid-2010s, Amazon deployed tens of thousands of drive units (formerly Kiva Systems) utilizing downward-facing optical cameras tracking QR matrix codes affixed to concrete floors on 1-meter grids. 

**Observed Field Evidence:** Within 18 to 24 months of continuous $24/7$ multi-tier operation, severe floor degradation emerged:
1. **Polyurethane Wheel Abrasion:** Rigid polyurethane drive wheels operating over repetitive rectilinear paths abraded the concrete surface paste, pulverizing micro-silica dust that coated optical camera lenses, causing optical localization failures.
2. **Floor Joint Faulting:** Vehicles repeatedly traversing unarmored saw-cut contraction joints caused micro-spalling of joint edges. The resulting impact shocks damaged internal IMU gyroscopes and accelerated wheel motor gearbox wear.
3. **Engineering Remediation:** Amazon transitioned new construction specifications to post-tensioned, jointless slabs treated with high-durability lithium silicate liquid hardeners and densifiers, and subsequently developed the "Proteus" AMR utilizing natural-feature 3D LiDAR SLAM, eliminating dependency on physical floor QR codes.

### 9.2 Ocado Smart Platform vs. AutoStore: Cube Storage Slabs
The Ocado Smart Platform (OSP) and AutoStore represent the pinnacle of 3D cube storage, packing hundreds of thousands of inventory totes into ultra-dense cubic aluminum grids.

**Observed Field Evidence:** 
- An AutoStore grid carrying 50,000 bins exerts a massive, highly uniform dead load across the concrete slab-on-ground, but top-grid tolerances are unforgiving. The aluminum extrusion grid relies on vertical columns resting directly on the concrete.
- If the concrete slab experiences differential subgrade settlement exceeding $1.5\text{ mm}$ over a $5\text{ m}$ span, the top aluminum track rails twist out of orthogonal alignment. Overhead robots traveling at $3.1\text{ m/s}$ experience wheel flange binding, leading to motor drive overcurrent trips and grid paralysis.
- Furthermore, operating thousands of battery-powered robots in a confined airspace generates massive convective thermal heat loads. Without dedicated under-grid ventilation air handling, ambient temperatures inside the upper cube exceed $45^\circ\text{C}$, triggering lithium-ion battery thermal throttling and premature electronic inverter failures.

### 9.3 Nike European Logistics Campus (Laakdal, Belgium): Clad-Rack AS/RS
At its European Logistics Campus in Laakdal, Belgium, Nike constructed one of the world's largest automated high-bay distribution centers, featuring clad-rack AS/RS crane aisles standing over $44\text{ meters}$ tall.

**Observed Field Evidence:**
- In a clad-rack structure, the storage racking serves as the primary structural frame supporting the external wall cladding and roof, eliminating traditional building columns.
- The civil foundation consists of a continuous, $1.5\text{-meter}$-thick post-tensioned reinforced concrete raft foundation supported by hundreds of drilled foundation piles driven into bedrock. 
- The bottom crane guide rails required laser-guided micro-shimming to achieve an elevation tolerance of $\pm 0.5\text{ mm}$ along the entire $120\text{-meter}$ aisle run. This extreme civil precision was necessary to ensure that dynamic crane mast sway remained below $25\text{ mm}$ at the 44-meter top elevation during simultaneous $v_x = 4.5\text{ m/s}$ and $v_y = 1.2\text{ m/s}$ peak travel.

### 9.4 Target Food Distribution Center: VNA Slab Curl Retrofit
At a major Target regional cold-storage facility, an existing warehouse section was repurposed for high-density storage using man-up VNA turret trucks operating at a $14\text{-meter}$ top beam elevation.

**Observed Field Evidence:**
- The existing slab had been poured as a conventional reinforced slab with $6\text{-meter}$ contraction joint spacing. Over 5 years of operation, shrinkage curling had warped the slab panels, lifting the edges at contraction joints by $4.5\text{ mm}$ relative to panel centers.
- When VNA turret trucks crossed these curled joints at $2.2\text{ m/s}$, the transverse tilt caused mast sway exceeding $75\text{ mm}$. Turret carriages struck stored pallets, sheered wire guidance antennas, and severely injured one operator due to mast whiplash.
- **Engineering Remediation:** Operations were halted for six weeks. Specialized concrete contractors performed laser-guided continuous floor grinding along the defined turret wheel tracks, cutting down curled joint shoulders and injecting structural polyurea under the slab edges to stabilize void spaces, restoring floor profile to DIN 15185 specifications.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Root Physical/Kinematic Mechanism | Operational Consequence | Engineering Mitigation |
|---|---|---|---|---|
| **VNA Mast Dynamic Collision** | Mast tip shudder; scraping marks on upper rack beams; guide-wire tracking alarms. | Concrete slab out-of-flatness ($\Delta z > 1.5\text{ mm}$) dynamically amplified across tall mast height: $\delta = H \cdot (\Delta z / W_w) K_{dyn}$. | High-speed impact with racking; structural rack collapse; equipment destruction; operator injury. | Grind wheel paths to $F_{min} 100$ / DIN 15185; install optical mast sway monitoring and dynamic velocity derating. |
| **Slab Punching Shear Punch-Through** | Radial micro-cracking and circular spalling around upright baseplates; anchor bolt tilting. | Factored concentrated column load exceeds concrete two-way shear capacity: $P_u > \phi \cdot 0.33 \lambda \sqrt{f'_c} b_0 d$. | Localized slab collapse; upright settlement; rack listing; loss of vertical plumbness. | Enlarge baseplate dimensions ($c$); increase slab depth; specify high-strength steel-fiber reinforced concrete. |
| **AutoStore Grid Thermal Binding** | Robot drive motor overcurrent trips; top grid track joints buckling or opening gaps. | Inadequate structural thermal expansion joints coupled with high robot battery heat dissipation. | Grid track binding; robot localization lost; widespread automated picking stoppage. | Install expansion sliding joints; install dedicated HVAC air distribution below the top grid deck. |
| **AMR Fleet Intersection Deadlock** | Robots clustered at corridor junctions; velocity drops to zero; WES displays pathing locks. | Inadequate clear passage width preventing passing; Conflict-Based Search algorithm deadlock under high fleet density. | Total throughput collapse; downstream picking starvation; manual operator intervention required. | Establish unidirectional traffic loops; add passing bays; enforce traffic congestion factor $\eta_{traffic}$ during fleet sizing. |
| **Sliding Shoe Package Pinching** | Crushed carton edges; torn corrugate; carton jams at divert spurs; high noise levels. | Inadequate upstream gapping ($G < G_{min}$); incorrect shoe divert angle ($\theta > 30^\circ$) exceeding carton friction. | Sorter jams; damaged customer inventory; sorter emergency stop tripping; facility shutdown. | Calibrate variable-speed gapping belts; enforce dynamic shoe assignment algorithms based on package length. |
| **AS/RS Bottom Rail Galling & Binding** | Loud screeching during crane transit; premature wheel flange wear; guide rail anchor loosening. | Differential foundation settlement along the aisle length creating horizontal and vertical rail misalignments. | Crane travel motor overloads; rail shear failure; catastrophic crane derailment risk. | Install deep pile foundations below crane rails; specify adjustable sole plates with non-shrink epoxy grout. |
| **Battery Room Hydrogen Deflagration** | Hydrogen gas alarm sensors triggering at 20% LFL; pungent chemical odors in charging area. | Exhaust fan motor failure or insufficient mechanical dilution airflow ($Q_{vent} < 100 \cdot V_{H2}$) during finish charge. | Flammable gas accumulation ($H_2 \ge 4.0\%$); catastrophic explosion hazard; immediate OSHA shutdown. | Install redundant spark-proof exhaust fans; interlock charger electrical contactors directly to airflow pressure switches. |
| **Post-Tensioned Tendon Severance** | High-energy snapping sound during anchor drilling; concrete spall blowout at slab anchor hole. | Contractor drilling rack anchor bolts into a post-tensioned slab without ground-penetrating radar (GPR). | Explosive release of prestressed tendon tension; loss of structural slab integrity; worker injury risk. | Mandate GPR scanning and certified floor mapping prior to any anchor drilling in PT slab facilities. |
| **Cross-Belt Sorter Discharge Slip** | Cartons missing chute centers; parcels landing on chute divider walls or tumbling backwards. | Dust accumulation on urethane belt surfaces reducing coefficient of friction $\mu$; belt acceleration mismatch. | Package sortation errors; misroutes; parcels falling to facility floor from elevated mezzanines. | Implement automated belt cleaning brushes; install dynamic optical package trajectory verification cameras. |
| **Seismic Rack-Roof Portal Tear-Out** | Sheared bolts at rack-to-building ties; distorted roof purlins following minor seismic tremors. | Rigid mechanical ties installed between stiff industrial pallet racks and flexible building structural framing. | Building roof structural distortion; rack upright buckling under induced roof sway loads; collapse. | Eliminate rigid building ties; design free-standing racks or install slotted slip-joints allowing differential drift. |
| **Battery Premature Fleet Depletion** | AMRs stopping mid-shift in active travel aisles with depleted batteries ("bricked" units). | Overestimating opportunity charging windows; operating in cold storage without battery thermal insulation. | Fleet capacity deficit; aisle blockage; manual towing required; missed order dispatch SLAs. | Derate battery capacity by 35% for cold environments; automate dynamic opportunistic dispatch when SoC drops below 30%. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks It | How to Test / Validate It |
|---|---|---|---|
| **Uniform Randomized Rack Access (FEM 9.851)** | Enables closed-form integration of Chebyshev travel time over the continuous unit rack area. | Velocity-based SKU slotting (Pareto / ABC zoning) concentrates travel into the lower-front rack zone. | Run Monte Carlo discrete-event simulations using empirical SKU pick frequencies from WMS order history. |
| **Independent Simultaneous Multi-Axis Travel** | AS/RS crane cycle models assume horizontal and vertical drives accelerate simultaneously without electrical or mechanical coupling. | Crane power busbar voltage drop limiting simultaneous peak motor acceleration; shared power supplies. | Measure physical crane current draw and velocity profiles using onboard inverter drive telemetry during commissioning. |
| **Constant-Rate Battery Opportunity Charging** | Sizing AMR fleets assumes linear energy replenishment during short docking intervals. | Lithium-ion constant-current / constant-voltage (CC/CV) saturation curve slows charging above 80% SoC; high battery temperature throttling. | Log real-time battery Management System (BMS) charging curves under full operational temperature conditions. |
| **Homogeneous Concrete Subgrade Modulus ($k$)** | Slab thickness and punching shear calculations assume uniform vertical soil reaction support. | Localized soil voiding, poor backfill compaction around utility trenches, or groundwater infiltration. | Conduct plate load testing (ASTM D1194) and continuous subgrade proof-rolling prior to slab casting. |
| **Rigid Baseplate Bearing Pressure** | Punching shear formulations assume uniform bearing stress distribution across the steel baseplate face. | Thin, flexible baseplates ($t_{plate} < 20\text{ mm}$) that bend under high upright axial loads, concentrating stresses at column corners. | Perform finite element analysis (FEA) of the baseplate-column connection per AISC Design Guide 1. |
| **Stationary Sorter Induction Flow** | Sorter capacity equations assume uniform parcel inter-arrival gaps ($G_{min}$) at rated conveyor speed. | Batch picking waves creating cyclic surge arrivals that swamp upstream gapping belts, causing induction starvation. | Verify sorter upstream buffer queue capacity using dynamic accumulation conveyor modeling. |
| **Constant Traffic Congestion Derating ($\eta_{traffic}$)** | Fleet sizing formulas utilize a static multiplier to account for vehicle interference. | Asymmetric traffic flows where all AMRs converge simultaneously on a single bottleneck cross-aisle or packing area. | Conduct discrete-event simulation (DES) incorporating physical collision avoidance and dynamic pathing algorithms. |
| **Zero Foundation Differential Settlement** | High-bay AS/RS tolerances assume the concrete foundation raft remains perfectly flat across decades of operation. | Uneven soil consolidation, seismic liquefaction, or seasonal groundwater table fluctuations. | Install continuous hydrostatic leveling systems or conduct annual high-precision optical laser geodetic surveys. |

---

## 12. Exercises

### Exercise 1: VNA Aisle Swept Path and Mast Sway Analysis
A warehouse is evaluating a man-up VNA turret truck operating at a top lift height of $H = 15.0\text{ m}$.
- The truck wheelbase width across load wheels is $W_w = 1.60\text{ m}$.
- The pallet load dimension across the aisle is $W_{load} = 1,200\text{ mm}$, and the truck chassis width is $W_{truck} = 1,520\text{ mm}$.
- Structural floor guide rails provide a clear side clearance of $C_{side} = 90\text{ mm}$ on each side.
- The concrete slab has an out-of-levelness variation across the wheel track of $\Delta z = 2.5\text{ mm}$. Dynamic magnification is $K_{dyn} = 1.8$.

**Deliverables:**
1. Calculate the required clear aisle width ($W_{vna}$) between rack load faces.
2. Calculate the static and dynamic mast tip lateral displacement ($\delta_{static}$ and $\delta_{dynamic}$).
3. If the mechanical clearance between the pallet load and the rack upright is $75\text{ mm}$, assess whether a rack collision will occur. State the required engineering mitigation.

*Estimated completion time:* 25 minutes.

---

### Exercise 2: AS/RS Crane FEM 9.851 Kinematic Optimization
An industrial engineer is designing a Unit-Load AS/RS storage aisle:
- Rack length: $L = 100.0\text{ m}$
- Rack height: $H = 25.0\text{ m}$
- Horizontal drive: $v_x = 3.5\text{ m/s}, a_x = 0.7\text{ m/s}^2$
- Vertical hoist: $v_y = 1.2\text{ m/s}, a_y = 0.8\text{ m/s}^2$
- Telescopic fork shuttle stroke time: $t_{p/d} = 10.0\text{ s}$

**Deliverables:**
1. Determine whether horizontal and vertical motions achieve rated velocity (trapezoidal vs. triangular profile).
2. Compute full-span transit times ($T_x, T_y$) and the normalized rack shape parameter $b$.
3. Calculate the expected Single-Command cycle time ($T_{sc}$) and Dual-Command cycle time ($T_{dc}$).
4. Compute the maximum hourly pallet throughput for both operating modes.
5. What physical rack height ($H^*$) would achieve a perfectly balanced kinematic rack ($b = 1.0$)?

*Estimated completion time:* 35 minutes.

---

### Exercise 3: AMR Sortation Inbound Fleet Sizing
An automated e-commerce sorting hub requires an AMR fleet to transfer bulk parcel containers:
- Total throughput requirement: $Q = 2,400\text{ container moves/hour}$ across two primary flow paths:
  - Path A (60% of volume): $D_{loaded} = 90\text{ m}$, $D_{empty} = 80\text{ m}$.
  - Path B (40% of volume): $D_{loaded} = 140\text{ m}$, $D_{empty} = 130\text{ m}$.
- Operating velocities: $v_{loaded} = 1.2\text{ m/s}$, $v_{empty} = 1.6\text{ m/s}$.
- Fixed dwells per trip: $t_{pick} = 12\text{ s}$, $t_{drop} = 12\text{ s}$, $t_{turns} = 10\text{ s}$, $t_{queue} = 6\text{ s}$.
- Battery regime: AMR runs for $2.0\text{ hours}$ between charges; fast charging requires $0.40\text{ hours}$ including docking overhead.
- Traffic congestion derating: Layout analysis indicates an operating congestion factor of $\eta_{traffic} = 0.86$.

**Deliverables:**
1. Compute the cycle time for Path A and Path B ($T_{trip, A}, T_{trip, B}$).
2. Determine the battery availability factor ($\eta_{batt}$).
3. Calculate the total required AMR fleet size ($N_{AMR}$).
4. If peak holiday volume increases throughput demand by 25%, calculate the additional AMRs required.

*Estimated completion time:* 30 minutes.

---

### Exercise 4: Concrete Slab Punching Shear & Baseplate Sizing
A high-bay selective pallet rack installation imposes an ultimate factored column load of $P_u = 310\text{ kN}$ per upright.
- Slab thickness: $h = 220\text{ mm}$, effective depth $d = 170\text{ mm}$.
- Concrete compressive strength: $f'_c = 30\text{ MPa}$ (normal-weight concrete, $\lambda = 1.0$).
- Current square baseplate dimension: $c = 200\text{ mm} \times 200\text{ mm}$.
- ACI shear strength reduction factor: $\phi = 0.75$.

**Deliverables:**
1. Calculate the critical punching shear perimeter ($b_0$).
2. Compute the nominal concrete punching shear strength ($V_c$) and design capacity ($\phi V_c$).
3. Determine whether the current $200\text{ mm}$ baseplate is structurally adequate.
4. If inadequate, calculate the minimum square baseplate side dimension ($c^*$) required to achieve $\phi V_c \ge P_u$.

*Estimated completion time:* 25 minutes.

---

### Exercise 5: Sliding Shoe Sorter Gapping & Divert Kinematics
A high-speed sliding shoe parcel sorter operates under the following specifications:
- Conveyor linear velocity: $v_c = 2.8\text{ m/s}$.
- Slat pitch: $P_{slat} = 127\text{ mm}$.
- Divert angle: $\theta = 30^\circ$.
- Average carton length: $L_{pkg} = 500\text{ mm}$.
- Upstream safety gap buffer: $C_{buffer} = 250\text{ mm}$.
- Divert shoe switch actuation time: $\Delta t = 0.15\text{ s}$.
- Static friction coefficient between shoe and carton: $\mu = 0.40$.

**Deliverables:**
1. Calculate the transverse divert velocity ($v_d$) and resultant divert velocity ($v_r$).
2. Calculate the transverse acceleration ($a_{lateral}$) imparted during switch actuation and verify if the carton slips.
3. Determine the minimum required carton gap ($G_{min}$) and the resulting maximum theoretical hourly sortation throughput ($\lambda_{sort}$).

*Estimated completion time:* 25 minutes.

---

### Exercise 6: Mega-Facility Charging Room & Electrical Infrastructure Design
An automated warehouse operates 120 AMRs (Lithium $\text{LiFePO}_4$) and 30 Class I electric forklifts (Flooded Lead-Acid).
- AMR fast chargers: 24 stations @ $2.2\text{ kW}$ DC each, operating power factor $\text{PF} = 0.96$, efficiency $\eta = 0.93$, diversity factor $K_{div, AMR} = 0.85$.
- Forklift chargers: 30 stations @ $12.0\text{ kW}$ DC each, operating power factor $\text{PF} = 0.90$, efficiency $\eta = 0.89$, diversity factor $K_{div, FL} = 0.75$.
- Forklift batteries: 36 cells each, finish charge current $I_{finish} = 22\text{ A}$.

**Deliverables:**
1. Calculate total diversified electrical service demand ($S_{total}$) in $\text{kVA}$.
2. Calculate total hourly volumetric hydrogen evolution ($V_{H2}$) in $\text{m}^3/\text{hr}$.
3. Calculate the required continuous mechanical exhaust ventilation rate in $\text{m}^3/\text{hr}$ and $\text{CFM}$ to maintain hydrogen $\le 1.0\%$.

*Estimated completion time:* 30 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1
1. **Clear Aisle Width ($W_{vna}$):**
   - $W_{vna} = \max(W_{truck}, W_{load}) + 2 \cdot C_{side} = \max(1.52, 1.20) + 2 \cdot (0.090) = 1.520 + 0.180 = 1.700\text{ m} = 1,700\text{ mm}$.
2. **Mast Tip Displacement:**
   - Static: $\delta_{static} = H \cdot \frac{\Delta z}{W_w} = 15.0 \cdot \frac{0.0025}{1.60} = 15.0 \cdot 0.0015625 = 0.02344\text{ m} = 23.44\text{ mm}$.
   - Dynamic: $\delta_{dynamic} = \delta_{static} \cdot K_{dyn} = 23.44 \cdot 1.8 = 42.19\text{ mm}$.
3. **Collision Risk & Mitigation:**
   - Operating clearance between pallet load and rack upright is $75\text{ mm}$. While the calculated dynamic deflection ($42.19\text{ mm}$) is less than $75\text{ mm}$, if the vehicle rolls in the opposite direction or experiences mast elasticity resonance, the margin is precariously narrow ($75 - 42.19 = 32.81\text{ mm}$). 
   - Furthermore, if rack upright installation plumbness drifts by $\pm 15\text{ mm}$, clearance is completely consumed.
   - **Mitigation:** Grind the wheel tracks to DIN 15185 / $F_{min} 100$ specifications ($\Delta z \le 1.0\text{ mm}$), which reduces dynamic sway to $15.0 \cdot (0.001/1.60) \cdot 1.8 = 16.88\text{ mm}$, preserving safe clearances.

### Answer to Exercise 2
1. **Motion Profiles:**
   - Horizontal: $s_{ax} = v_x^2 / (2 a_x) = (3.5)^2 / (2 \cdot 0.7) = 12.25 / 1.4 = 8.75\text{ m}$. Since $L = 100\text{ m} > 2 \cdot 8.75 = 17.5\text{ m}$, horizontal motion is **trapezoidal**.
   - Vertical: $s_{ay} = v_y^2 / (2 a_y) = (1.2)^2 / (2 \cdot 0.8) = 1.44 / 1.6 = 0.90\text{ m}$. Since $H = 25\text{ m} > 2 \cdot 0.90 = 1.80\text{ m}$, vertical motion is **trapezoidal**.
2. **Span Times and Shape Parameter:**
   - $t_{ax} = 3.5 / 0.7 = 5.0\text{ s}$. $T_x = 2 \cdot 5.0 + (100 - 17.5) / 3.5 = 10.0 + 82.5 / 3.5 = 10.0 + 23.571 = 33.571\text{ s}$.
   - $t_{ay} = 1.2 / 0.8 = 1.5\text{ s}$. $T_y = 2 \cdot 1.5 + (25 - 1.8) / 1.2 = 3.0 + 23.2 / 1.2 = 3.0 + 19.333 = 22.333\text{ s}$.
   - $T_{max} = T_x = 33.571\text{ s}$, $T_{min} = T_y = 22.333\text{ s}$.
   - $b = T_{min} / T_{max} = 22.333 / 33.571 = 0.66525$.
3. **Cycle Times:**
   - $E[T_{sc\_travel}] = T_{max} \cdot (1/3 + b^2 / 30) = 33.571 \cdot (0.33333 + (0.66525)^2 / 30) = 33.571 \cdot (0.33333 + 0.01475) = 33.571 \cdot 0.34808 = 11.685\text{ s}$.
   - $E[T_{between}] = T_{max} \cdot (1/3 + b^2/6 - b^3/30) = 33.571 \cdot (0.33333 + 0.07376 - 0.00982) = 33.571 \cdot 0.39727 = 13.337\text{ s}$.
   - $T_{sc} = 2 \cdot (11.685) + 2 \cdot (10.0) = 23.37 + 20.0 = 43.37\text{ s}$.
   - $T_{dc} = 2 \cdot (11.685) + 13.337 + 4 \cdot (10.0) = 23.37 + 13.337 + 40.0 = 76.71\text{ s}$.
4. **Throughput:**
   - Single-Command Throughput: $\lambda_{sc} = 3600 / 43.37 = 83.0\text{ pallets/hour}$.
   - Dual-Command Throughput: $\lambda_{dc} = 7200 / 76.71 = 93.86\text{ pallets/hour}$ ($46.93\text{ dual cycles/hr}$).
5. **Kinematically Balanced Height ($H^*$):**
   - For $b = 1.0$, $T_y = T_x = 33.571\text{ s}$.
   - $33.571 = 2 \cdot t_{ay} + (H^* - 2 \cdot s_{ay}) / v_y = 3.0 + (H^* - 1.8) / 1.2$.
   - $30.571 = (H^* - 1.8) / 1.2 \implies H^* = 30.571 \cdot 1.2 + 1.8 = 36.685 + 1.8 = 38.49\text{ m}$.

### Answer to Exercise 3
1. **Trip Times:**
   - Fixed dwell: $t_{fixed} = 12 + 12 + 10 + 6 = 40.0\text{ s}$.
   - $T_{trip, A} = 40.0 + (90 / 1.2) + (80 / 1.6) = 40.0 + 75.0 + 50.0 = 165.0\text{ s}$.
   - $T_{trip, B} = 40.0 + (140 / 1.2) + (130 / 1.6) = 40.0 + 116.67 + 81.25 = 237.92\text{ s}$.
2. **Battery Availability Factor ($\eta_{batt}$):**
   - $\eta_{batt} = t_{run} / (t_{run} + t_{charge}) = 2.0 / (2.0 + 0.40) = 2.0 / 2.40 = 0.8333$.
3. **Fleet Size ($N_{AMR}$):**
   - Demand on Path A: $Q_A = 0.60 \cdot 2400 = 1,440\text{ moves/hr}$.
   - Demand on Path B: $Q_B = 0.40 \cdot 2400 = 960\text{ moves/hr}$.
   - Total work: $W = (1440 \cdot 165.0) + (960 \cdot 237.92) = 237,600 + 228,403 = 466,003\text{ robot-seconds}$.
   - Effective capacity per robot: $\text{Cap}_{eff} = 3600 \cdot 0.8333 \cdot 0.86 = 2,579.9\text{ seconds}$.
   - $N_{AMR} = \lceil 466,003 / 2,579.9 \rceil = \lceil 180.63 \rceil = 181\text{ AMRs}$.
4. **Peak Volume (25% Surge):**
   - Surge work: $W_{peak} = 1.25 \cdot 466,003 = 582,504\text{ robot-seconds}$.
   - $N_{peak} = \lceil 582,504 / 2,579.9 \rceil = \lceil 225.79 \rceil = 226\text{ AMRs}$.
   - Additional AMRs required: $226 - 181 = 45\text{ AMRs}$.

### Answer to Exercise 4
1. **Critical Perimeter ($b_0$):**
   - $b_0 = 4 \cdot (c + d) = 4 \cdot (200.0 + 170.0) = 4 \cdot 370.0 = 1,480.0\text{ mm}$.
2. **Nominal Strength and Design Capacity:**
   - $\sqrt{f'_c} = \sqrt{30.0} \approx 5.4772\text{ MPa}$.
   - $V_c = 0.33 \cdot 1.0 \cdot 5.4772 \cdot 1480.0 \cdot 170.0 = 454,772\text{ N} = 454.77\text{ kN}$.
   - $\phi V_c = 0.75 \cdot 454.77 = 341.08\text{ kN}$.
3. **Structural Adequacy:**
   - $P_u = 310.0\text{ kN} \le \phi V_c = 341.08\text{ kN}$. The current baseplate is **adequate** ($FS = 341.08 / 310.0 = 1.10$).
4. **Required Baseplate for Exact Capacity ($c^*$):**
   - To achieve $\phi V_c = P_u = 310.0\text{ kN}$:
   - Required $V_c = 310.0 / 0.75 = 413.33\text{ kN} = 413,333\text{ N}$.
   - $b_0^* = \frac{V_c}{0.33 \cdot \sqrt{f'_c} \cdot d} = \frac{413,333}{0.33 \cdot 5.4772 \cdot 170.0} = \frac{413,333}{307.27} = 1,345.17\text{ mm}$.
   - $4 \cdot (c^* + 170) = 1,345.17 \implies c^* + 170 = 336.29 \implies c^* = 166.3\text{ mm}$.
   - Minimum standard plate size: $170\text{ mm} \times 170\text{ mm}$.

### Answer to Exercise 5
1. **Divert Velocities:**
   - $v_c = 2.8\text{ m/s}$.
   - $v_d = v_c \cdot \tan 30^\circ = 2.8 \cdot 0.57735 = 1.6166\text{ m/s}$.
   - $v_r = \sqrt{(2.8)^2 + (1.6166)^2} = \sqrt{7.84 + 2.6134} = \sqrt{10.4534} = 3.233\text{ m/s}$.
2. **Lateral Acceleration & Friction Check:**
   - $a_{lateral} = v_d / \Delta t = 1.6166 / 0.15 = 10.777\text{ m/s}^2$.
   - Maximum friction limit before slip: $a_{max} = \mu \cdot g = 0.40 \cdot 9.81 = 3.924\text{ m/s}^2$.
   - **Slip Analysis:** Since $a_{lateral} = 10.78\text{ m/s}^2 > 3.924\text{ m/s}^2$, the carton **will slip** violently if the divert switch actuates abruptly in $0.15\text{ s}$. 
   - **Remediation:** Divert switches must utilize curved guide tracks that gradually ramp lateral acceleration over a transition length ($L_{switch} \ge 1.2\text{ m}$), keeping $a_{lateral} \le 3.5\text{ m/s}^2$.
3. **Gap and Throughput:**
   - Number of shoes: $N_{shoes} = \lceil L_{pkg} / P_{slat} \rceil + 2 = \lceil 500 / 127 \rceil + 2 = 4 + 2 = 6\text{ shoes}$.
   - Minimum gap: $G_{min} = N_{shoes} \cdot P_{slat} + C_{buffer} = 6 \cdot 127 + 250 = 762 + 250 = 1,012\text{ mm} = 1.012\text{ m}$.
   - Maximum throughput: $\lambda_{sort} = \frac{3600 \cdot v_c}{L_{pkg} + G_{min}} = \frac{3600 \cdot 2.8}{0.500 + 1.012} = \frac{10,080}{1.512} = 6,666.7\text{ parcels/hour}$.

### Answer to Exercise 6
1. **Total Diversified Electrical Service ($S_{total}$):**
   - AMR connected: $24 \cdot 2.2 = 52.8\text{ kW}$.
   - AMR apparent power: $S_{AMR} = \frac{52.8 \cdot 0.85}{0.96 \cdot 0.93} = \frac{44.88}{0.8928} = 50.27\text{ kVA}$.
   - Forklift connected: $30 \cdot 12.0 = 360.0\text{ kW}$.
   - Forklift apparent power: $S_{FL} = \frac{360.0 \cdot 0.75}{0.90 \cdot 0.89} = \frac{270.0}{0.801} = 337.08\text{ kVA}$.
   - $S_{total} = S_{AMR} + S_{FL} = 50.27 + 337.08 = 387.35\text{ kVA}$. (Specify a minimum $500\text{ kVA}$ dedicated transformer).
2. **Volumetric Hydrogen Generation ($V_{H2}$):**
   - Total cells: $N_{cells} = 30 \cdot 36 = 1,080\text{ cells}$.
   - $V_{H2} = 1080 \cdot 22 \cdot 0.000418 = 9.9317\text{ m}^3/\text{hour}$.
3. **Mechanical Exhaust Airflow:**
   - $Q_{vent} = V_{H2} / 0.01 = 993.17\text{ m}^3/\text{hour}$.
   - In CFM ($1\text{ m}^3/\text{hr} = 0.588578\text{ CFM}$):
   - $Q_{vent} = 993.17 \cdot 0.588578 = 584.56\text{ CFM}$.
   - Specify a spark-proof Class I Div 2 exhaust fan rated for at least $700\text{ CFM}$.

---

## 14. Interview questions

### Question 1: AS/RS Kinematics & FEM 9.851
**Prompt:** "In an AS/RS high-bay crane design, how does the normalized shape factor $b = T_{min} / T_{max}$ influence crane cycle efficiency, and why does a physically square rack rarely yield a kinematically square system?"
- **What a strong candidate covers:** The candidate explains that the Chebyshev travel metric governs simultaneous $X-Y$ crane travel, where travel time is $\max(t_x, t_y)$. A kinematically square rack ($b = 1.0$) means full-span travel times are identical ($T_x = T_y$), maximizing the probability that travel occurs along the Chebyshev diagonal, minimizing dead idle time on either axis. The candidate notes that a physically square rack (e.g., $30\text{ m} \times 30\text{ m}$) does not equal a kinematically square rack because horizontal travel speed ($v_x \approx 3.0 - 5.0\text{ m/s}$) is typically 3 to 4 times faster than vertical hoist speed ($v_y \approx 0.8 - 1.2\text{ m/s}$), and acceleration profiles differ. Thus, a kinematically balanced rack is physically 3 to 4 times longer than it is tall.
- **Red flags in weak answers:** Confusing physical dimensions with travel time; adding travel times linearly ($t_x + t_y$, Manhattan metric); ignoring acceleration trapezoids.

### Question 2: Slab Flatness vs. Levelness for Automation
**Prompt:** "What is the difference between $F_F / F_L$ per ASTM E1155 and $F_{min}$ per ACI 360R / DIN 15185, and why can a floor with acceptable $F_F / F_L$ cause catastrophic failure in a VNA turret truck aisle?"
- **What a strong candidate covers:** $F_F / F_L$ are statistical metrics designed for random-traffic floors, sampled randomly along diagonal lines. $F_F$ measures surface curvature/waviness over $300\text{ mm}$ intervals, and $F_L$ measures pitch over $3\text{ m}$. In contrast, $F_{min}$ and DIN 15185 are continuous profile tolerances measured down the exact wheel tracks of defined-traffic vehicles. A floor can pass an overall $F_F 50 / F_L 50$ test across the room, yet possess a localized transverse elevation difference ($\Delta z$) of $3\text{ mm}$ across the $1.5\text{ m}$ wheel track. For a VNA turret truck at $16\text{ m}$ lift height, that $3\text{ mm}$ step tilts the mast, creating a dynamic tip displacement of over $50\text{ mm}$, striking the racking. Defined-traffic aisles strictly require $F_{min}$ grinding.
- **Red flags in weak answers:** Believing $F_F$ and $F_L$ are interchangeable; not knowing that VNA trucks follow fixed wheel paths; failing to derive mast sway geometry ($\delta = H \cdot \Delta z / W_w$).

### Question 3: AMR vs. AGV Architecture & Fleet Sizing
**Prompt:** "How does the mathematical fleet sizing formulation differ between an AMR utilizing dynamic SLAM navigation and a traditional AGV following fixed magnetic floor tape?"
- **What a strong candidate covers:** While the underlying work equation $N = \lceil \sum Q_i T_{trip} / (3600 \eta_b \eta_t) \rceil$ applies to both, the parameters diverge sharply. Fixed-guide AGVs operate on single-track networks where passing is impossible; therefore, the traffic congestion factor $\eta_{traffic}$ degrades rapidly ($0.70 - 0.80$) under high fleet counts due to blocking and conveyor queueing. AMRs with LiDAR SLAM can dynamically re-route around obstacles, improving $\eta_{traffic}$ ($0.85 - 0.92$), but require higher safety clearance envelopes ($C_{side} \ge 500\text{ mm}$) to comply with ISO 3691-4 personnel detection fields without continuous speed deratings. Furthermore, opportunity charging profiles ($\eta_{batt}$) must be modeled explicitly based on C-rate and constant-current/constant-voltage curves.
- **Red flags in weak answers:** Stating that AMRs don't need traffic factors because "software AI optimizes it"; ignoring empty vehicle repositioning distances ($D_{empty}$); omitting battery availability factors.

### Question 4: Structural Slab Loading & Punching Shear
**Prompt:** "When an automated pallet rack upright increases its column load from $150\text{ kN}$ to $300\text{ kN}$, why is increasing baseplate thickness alone insufficient, and what governing failure mechanism must the civil engineer check?"
- **What a strong candidate covers:** The governing failure mode in an unreinforced slab-on-ground under concentrated point loading is two-way punching shear per ACI 318. The punching shear capacity $\phi V_c = \phi \cdot 0.33 \lambda \sqrt{f'_c} b_0 d$ is a function of the concrete tensile strength, slab effective depth ($d$), and the critical shear perimeter $b_0 = 4(c + d)$. Increasing baseplate thickness prevents the plate itself from bending, ensuring uniform bearing, but does not increase $b_0$ unless the baseplate plan dimensions ($c_1, c_2$) are enlarged. To resist a $300\text{ kN}$ load, the engineer must expand baseplate width/length, increase slab structural thickness, or install structural tie-rods/post-tensioning.
- **Red flags in weak answers:** Confusing one-way beam shear with two-way punching shear; assuming soil subgrade carries the concentrated shear directly; ignoring concrete compressive strength $f'_c$.

### Question 5: Seismic Building-Rack Interaction
**Prompt:** "Why do structural engineers strictly prohibit tying tall industrial storage racks rigidly to the roof framing of a pre-engineered metal building (PEMB) in seismic zones?"
- **What a strong candidate covers:** Pre-engineered metal buildings are designed as flexible moment-resisting or braced frames that experience significant lateral drift ($\Delta_{drift} \approx H / 100$ to $H / 60$) during wind and seismic events. Conversely, steel storage racks braced in the longitudinal and transverse directions are extremely stiff structures with very short natural periods of vibration. If tied rigidly together, the flexible building will lean against the stiff rack during a seismic event, transferring the building's massive roof inertial loads directly into the rack uprights. This unmodeled axial and torsional load buckles the rack uprights, causing progressive structural collapse. Tying is only permissible via engineered slip-joints that allow free horizontal relative drift while providing only out-of-plane restraint if required.
- **Red flags in weak answers:** Claiming that tying the rack to the building makes both structures "stronger"; failing to recognize differences in structural stiffness and natural frequency.

### Question 6: 3D Cube Storage (AutoStore) Retrieval Kinetics
**Prompt:** "In an AutoStore 3D cube storage system, explain why retrieval throughput is non-linear with respect to grid depth, and how inventory slotting policies mitigate this effect."
- **What a strong candidate covers:** In a 3D grid, bins are stacked directly atop each other. To retrieve a bin at depth $d$, the overhead robot must perform $d$ preliminary "digs"—hoisting the overlying bins, traversing to adjacent open cells, depositing them, and repeating. The time to retrieve a bin at depth $d$ scales as $O(d)$, because each dig incurs a full vertical hoist, clamp, travel, and lower cycle. Under pure random inventory placement in a 16-deep grid, the expected dig depth is $7.5$, throttling throughput. Facilities mitigate this using Pareto ABC velocity stratification: when a retrieved bin is returned to the grid, the WES directs the robot to deposit it in the top 2 to 3 tiers. Fast-moving items naturally migrate to the surface, reducing average operational dig depth to between $1.2$ and $1.8$.
- **Red flags in weak answers:** Assuming bins are accessed from the bottom or sides; claiming digging can be eliminated entirely; ignoring the time penalty of temporary bin restaging.

### Question 7: High-Speed Sorter Gapping Physics
**Prompt:** "What physical factors dictate the minimum package pitch on a high-speed sliding shoe sorter, and what happens if upstream gapping belts fail to maintain this gap?"
- **What a strong candidate covers:** Package pitch is the sum of package length ($L_{pkg}$) and the dynamic gap ($G$). The minimum gap $G_{min} = N_{shoes} \cdot P_{slat} + C_{buffer}$ is dictated by the number of divert shoes required to support the package length plus the physical width of the divert switch gate. When a package is diverted at angle $\theta$, the trailing divert shoes must fully cross the conveyor bed before the leading edge of the next package arrives at that switch point. If the upstream gapping system fails and $G < G_{min}$, the trailing shoes of the diverted package will physically strike the front or side of the trailing package, causing carton rotation, jams, torn corrugate, and emergency stops.
- **Red flags in weak answers:** Believing gap is a fixed time delay independent of package length; overlooking shoe return mechanics; neglecting conveyor speed $v_c$.

### Question 8: Electrical Battery Charging Room Design
**Prompt:** "What are the primary utility and life-safety engineering requirements when designing a central battery room for a fleet of 50 flooded lead-acid forklifts?"
- **What a strong candidate covers:** The candidate covers three domains:
  1. **Electrical:** Connected load calculation incorporating charger DC output, AC-DC rectifier efficiency ($\approx 88-92\%$), power factor ($\approx 0.90$), and diversity factor ($K_{div} \approx 0.75-0.85$), sizing dedicated switchgear and transformers.
  2. **Chemical/Life Safety:** Flooded lead-acid batteries emit hydrogen ($H_2$) during finish charging. Per OSHA 1910.178 and NFPA 2, the engineer must calculate hydrogen evolution ($V_{H2} = N_{cells} \cdot I_{finish} \cdot 0.000418\text{ m}^3/\text{hr}$) and specify continuous mechanical exhaust airflow to dilute $H_2$ below $1.0\%$ (25% LFL).
  3. **Classification:** Motors and lighting must be Class I, Division 2 rated; ductwork must feature airflow pressure switches interlocked to trip charger power if exhaust ventilation fails; emergency eyewash/deluge showers must be within 25 feet.
- **Red flags in weak answers:** Omitting hydrogen ventilation entirely; assuming lead-acid batteries don't outgas; ignoring electrical power factor and diversity factors.

---

## 15. Key terms

1. **AS/RS (Automated Storage and Retrieval System):** Rail-guided, computer-controlled crane mechanism designed for high-density, high-velocity unit-load or carton storage.
2. **Chebyshev Metric ($L_\infty$):** Mathematical distance metric evaluating travel time as the maximum of independent simultaneous axis transits: $\max(t_x, t_y)$.
3. **FEM 9.851:** European standard defining calculation principles for cycle times and throughput of storage and retrieval machines.
4. **Normalized Rack Shape ($b$):** The ratio of minimum to maximum full-span axis transit times: $b = T_{min} / T_{max} \le 1.0$.
5. **Single-Command Cycle ($T_{sc}$):** An AS/RS operational cycle performing one storage or one retrieval transaction per round trip from I/O.
6. **Dual-Command Cycle ($T_{dc}$):** An AS/RS operational cycle performing one storage and one retrieval transaction within a single round trip.
7. **Floor Flatness ($F_F$):** ASTM E1155 metric evaluating concrete surface waviness and curvature over short 300 mm spans.
8. **Floor Levelness ($F_L$):** ASTM E1155 metric evaluating concrete slab deviation from a true horizontal plane over 3-meter spans.
9. **Superflat Defined-Traffic ($F_{min}$):** Absolute profile tolerance standard (ACI 360R / DIN 15185) measuring transverse and longitudinal elevation differences along fixed wheel paths.
10. **Mast Dynamic Sway ($\delta_{dynamic}$):** The amplified lateral deflection of an MHE mast tip caused by floor wheel track elevation steps and horizontal travel acceleration.
11. **Punching Shear ($V_c$):** Two-way structural shear stress failure in a concrete slab around a concentrated column baseplate or heavy wheel load.
12. **Critical Shear Perimeter ($b_0$):** ACI 318 calculation boundary located at $d/2$ from the perimeter of a concentrated point load.
13. **Subgrade Modulus ($k$):** Soil stiffness parameter measuring pressure per unit vertical soil deflection ($\text{MPa/m}$ or $\text{pci}$).
14. **Post-Tensioned (PT) Slab:** Concrete slab reinforced with prestressed high-strength steel tendons, eliminating contraction joints over wide spans.
15. **RMI ANSI MH16.1:** American national standard governing the structural design, testing, and seismic analysis of industrial steel storage racks.
16. **Seismic Base Shear ($V$):** Total horizontal lateral force exerted on a rack structure during an earthquake: $V = C_s W_{seismic}$.
17. **Tier-Captive Shuttle:** Dedicated horizontal automated vehicle restricted to a single vertical rack tier, feeding end-of-aisle vertical lifts.
18. **Roaming Shuttle:** Automated shuttle capable of traversing multiple vertical tiers via an integrated vertical lift mechanism.
19. **3D Cube Storage (AutoStore):** High-density automated storage architecture where bins are stacked directly in a vertical grid without aisles.
20. **Dig Depth ($d$):** The number of overlying non-target bins that must be removed to access a requested SKU in a 3D cube storage grid.
21. **Autonomous Mobile Robot (AMR):** Driverless vehicle utilizing onboard LiDAR, cameras, and SLAM algorithms to navigate dynamically without fixed physical infrastructure.
22. **Automated Guided Vehicle (AGV):** Driverless transport vehicle guided by physical floor infrastructure such as magnetic tape, optical lines, or embedded wires.
23. **Battery Availability Factor ($\eta_{batt}$):** The fraction of an operational shift a vehicle is productive after deducting charging and docking dwells.
24. **Traffic Congestion Factor ($\eta_{traffic}$):** Operational derating factor accounting for vehicle speed throttling, queuing, and rerouting in multi-robot layouts.
25. **Sliding Shoe Sorter:** High-speed slat conveyor utilizing diverting shoes traveling along transverse tracks to push cartons into takeaway spurs.
26. **Cross-Belt Sorter:** Continuous sortation loop conveyor equipped with bi-directional motorized belt carriages that actively discharge parcels.
27. **Tilt-Tray Sorter:** Continuous loop carrier sortation system where trays tilt mechanically or pneumatically to slide cartons into destination chutes.
28. **Gapping Belt:** Series of variable-speed conveyor belts upstream of a sorter designed to meter precise physical spacing ($G_{min}$) between items.
29. **C-Rate:** Battery charge/discharge current normalized against rated capacity; $1\text{C}$ delivers full charge/discharge in 1 hour.
30. **Class I, Division 2:** National Electrical Code classification for areas where flammable gases (such as hydrogen) may accumulate under abnormal conditions.

---

## 16. Sources

- **American Concrete Institute (ACI).** (2019). *Building Code Requirements for Structural Concrete (ACI 318-19) and Commentary*. Farmington Hills, MI: ACI.
- **American Concrete Institute (ACI).** (2010). *Guide to Design of Slabs-on-Ground (ACI 360R-10)*. Farmington Hills, MI: ACI.
- **ASTM International.** (2020). *Standard Test Method for Determining $F_F$ Floor Flatness and $F_L$ Floor Levelness Numbers (ASTM E1155-20)*. West Conshohocken, PA: ASTM.
- **Bozer, Y. A., & White, J. A.** (1984). Travel-time models for automated storage/retrieval systems. *IIE Transactions*, 16(4), 329–338.
- **de Koster, R., Le-Duc, T., & Roodbergen, K. J.** (2007). Design and control of warehouse order picking: A review. *European Journal of Operational Research*, 182(2), 481–501.
- **Deutsches Institut für Normung (DIN).** (1999). *Warehouses and storage facilities; Tolerances, deformations and clearances in the narrow aisle (DIN 15185-1 & 15185-2)*. Berlin: Beuth Verlag.
- **Federation Europeenne de la Manutention (FEM).** (2003). *Calculation principles of storage and retrieval machines — Cycle times (FEM 9.851)*. Frankfurt: VDMA.
- **Federation Europeenne de la Manutention (FEM).** (1995). *Calculation principles of storage and retrieval machines — Tolerances, deformations and clearances in the storage system (FEM 9.831 / 9.832)*. Frankfurt: VDMA.
- **Gu, J., Goetschalckx, M., & McGinnis, L. F.** (2007). Research on warehouse design and performance evaluation: A comprehensive review. *European Journal of Operational Research*, 177(1), 1–21.
- **International Organization for Standardization (ISO).** (2023). *Industrial trucks — Safety requirements and verification — Part 4: Driverless industrial trucks and their systems (ISO 3691-4:2023)*. Geneva: ISO.
- **National Fire Protection Association (NFPA).** (2022). *Standard for the Installation of Sprinkler Systems (NFPA 13)*. Quincy, MA: NFPA.
- **National Fire Protection Association (NFPA).** (2023). *National Electrical Code (NFPA 70)*. Quincy, MA: NFPA.
- **Occupational Safety and Health Administration (OSHA).** (2021). *Powered Industrial Trucks (29 CFR 1910.178)*. Washington, DC: U.S. Department of Labor.
- **Rack Manufacturers Institute (RMI).** (2021). *Specification for the Design, Testing and Utilization of Industrial Steel Storage Racks (ANSI MH16.1:2021)*. Charlotte, NC: Material Handling Industry (MHI).

---

## 17. Links

- **Parent Index:** [`_index.md`](_index.md) — Module 45: Plant Layout, Facility Design, and Detailed Engineering Interfaces
- **Sibling Topics:**
  - Topic 01: Facility Types and Flow Topologies — [`01_facility-types-flow.md`](01_facility-types-flow.md)
  - Topic 02: Systematic Layout Planning — [`02_systematic-layout-planning.md`](02_systematic-layout-planning.md)
  - Topic 03: Capacity and Space Calculation — [`03_capacity-space-calculation.md`](03_capacity-space-calculation.md)
  - Topic 04: Dock and Staging Design — [`04_dock-staging.md`](04_dock-staging.md)
  - Topic 06: Engineering Documents and P&ID — [`06_engineering-documents-pid.md`](06_engineering-documents-pid.md)
  - Topic 07: MTO and Revision Control — [`07_mto-revision-control.md`](07_mto-revision-control.md)
  - Topic 08: Safety, Fire Protection, and Ergonomics — [`08_safety-fire-ergonomics.md`](08_safety-fire-ergonomics.md)
  - Topic 09: Layout Simulation and Validation — [`09_layout-simulation.md`](09_layout-simulation.md)
  - Topic 10: Design for Supply Chain — [`10_design-for-supply-chain.md`](10_design-for-supply-chain.md)
- **Root Survey File:** [`../../../course/45_plant_layout_detailed_engineering.md`](../../../course/45_plant_layout_detailed_engineering.md)
- **Prerequisites & Methods Links:**
  - Module 10: Inventory Optimization & Multi-Echelon Planning — [`../../02_methods/10_inventory_optimization_multi_echelon/_index.md`](../../02_methods/10_inventory_optimization_multi_echelon/_index.md)
  - Module 15: Warehouse Operations & Labor Analytics — [`../../03_commerce/15_warehouse_labor_analytics/_index.md`](../../03_commerce/15_warehouse_labor_analytics/_index.md)
  - Module 20: Digital Twins and Simulation — [`../../02_methods/20_digital_twins_simulation_optimization_uncertainty/_index.md`](../../02_methods/20_digital_twins_simulation_optimization_uncertainty/_index.md)
