# Drilling, Blasting, and Load-Haul Operations

> Module 39 — Mining, Metals, and Forestry Supply Chains
> Topic 03 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 15 (Production Planning and Scheduling), Module 38 (Process Industries, Commodities, and Bulk Supply Chains), Module 39 Topic 01 (Reserves, Grade, and Confidence), and Topic 02 (Mine Planning and Cut-Off Grade)
> Version 2026.5

---

## 1. Why this topic exists

In surface and underground hard-rock mining, the physical extraction chain—drilling, blasting, loading, and hauling (D-B-L-H)—is the operational and thermodynamic gateway of the entire raw material supply chain. It consumes 60% to 75% of total open-pit operating expenditures (OPEX) and dictates the physical form, spatial distribution, and metallurgical feed quality of all material entering downstream beneficiation circuits. 

Historically, mining enterprises operated under an isolated departmental silo model:
1. **The Drill and Blast Department** was incentivized on a narrow, isolated metric: minimize drill-and-blast cost per tonne of broken rock ($\$/t_{\text{drilled}}$ or $\$/t_{\text{blasted}}$). To minimize explosive costs, drill patterns were expanded (increasing burden and spacing), stemming columns were lengthened, and powder factors were reduced to the bare minimum required to dislodge the bench.
2. **The Load and Haul Department** was incentivized on shovel loading rates ($t/\text{operating hour}$) and truck cycle times, absorbing the hidden costs of coarse, blocky muckpiles that reduced bucket fill factors, increased shovel cycle times by 30% to 50%, accelerated tooth and lip wear, and induced severe structural fatigue on truck frames and suspension struts.
3. **The Mineral Processing Plant** operated under fixed comminution parameters, absorbing run-of-mine (ROM) fragmentation variations that choked primary crushers with oversize boulders, starved semi-autogenous grinding (SAG) mills of grinding media or critical fine fractions, and caused wild oscillations in mill power draw, feed rates, and mineral flotation recovery.

```
+-----------------------------------------------------------------------------+
|              THE FRAGMENTATION COST PARADOX ACROSS THE VALUE CHAIN          |
+-----------------------------------------------------------------------------+
| Unit Cost ($/t)                                                             |
|   ^                                                                         |
|   |                      Total Extraction + Comminution Cost                |
|   |                               \     /                                   |
|   |                                \   /  <-- GLOBAL OPTIMUM                |
|   |       Crushing & Milling Cost   \_/                                     |
|   |       (Dominates Comminution)    \                                      |
|   |         \                         \                                     |
|   |          \                         \                                    |
|   |           \                         \____ Drill & Blast Cost            |
|   |            \____________________         (Chemical Energy: Cheaper)     |
|   +---------------------------------------------------------------------->  |
|   0                        Powder Factor q (kg/m3)                          |
|                     Coarse <---------------------> Fine                     |
|                   Muckpile                        Muckpile                  |
+-----------------------------------------------------------------------------+
```

This structural disconnect represents an immense economic leak due to **the comminution energy paradox**:
- Comminution (crushing and grinding) is the single most energy-intensive industrial process on the planet, consuming approximately 3% to 5% of global electrical energy and up to 50% to 70% of a mine site's electrical consumption.
- Size reduction is governed by the surface energy of rock fracture. Chemical energy delivered via high-explosive detonation in blastholes is approximately 10 to 30 times cheaper per gigajoule than mechanical electrical energy delivered to tumbling mills via electric motors and heavy steel grinding balls ($0.15\text{--}0.30\ \$/\text{GJ}$ for bulk ammonium nitrate fuel oil [ANFO] versus $25.00\text{--}45.00\ \$/\text{GJ}$ for industrial grid electricity).
- Elevating the blast powder factor by 20% to 40% adds modest chemical explosive cost (e.g., $+\$0.15/t$), but micro-fractures the rock mass, generates higher micro-crack density, shifts the fragmentation passing size ($P_{80}$) downwards, and increases the fraction of pre-crushed fines ($-12.7\text{ mm}$). This can increase downstream SAG mill throughput by 10% to 25%, dropping plant comminution costs by $\$0.60/t$ to $\$1.50/t$ while unlocking tens of millions of dollars in annualized cash flow. This holistic philosophy is known as the **Mine-to-Mill (M2M)** integration framework.

Concurrently, blasting does not fracture rock in situ; it heaves and displaces the rock mass horizontally and vertically by 5 to 30 meters. When mine geologists draw ore and waste boundary polygons on pre-blast drill hole assay maps, and the loading shovel digs those polygons without accounting for 3D blast displacement vectors, severe **contact dilution** and **ore loss (misallocation)** occur. A 5% dilution in a 50,000 t/d copper concentrator feeds barren rock into the mill, displacing valuable ore, consuming expensive reagents, overloading the flotation circuit, and filling tailings storage facilities prematurely, while valuable ore is dumped into waste dumps.

Finally, haulage accounts for 40% to 50% of open-pit OPEX. Modern ultra-class haul trucks (240 to 400 metric tonnes payload) cost $\$5\text{M}$ to $\$7\text{M}$ per unit and burn 250 to 450 liters of diesel per operating hour. When trucks operate under static assignments, stochastic variance in shovel loading, haul road grades, and crusher dump pockets causes catastrophic **truck bunching** and shovel starvation. Dynamic real-time dispatching—orchestrated via two-stage mathematical optimization (linear programming network flow coupled with dynamic assignment algorithms)—is mandatory to maximize fleet utilization, enforce grade-blending specifications, and stabilize crusher feed rates.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Formulate and compute** rock mass fragmentation distributions using the Cunningham-modified Kuz-Ram equations and the Swebrec function, evaluating median fragment size ($x_{50}$), uniformity index ($n$), and passing size ($P_{80}$).
2. **Design** open-pit blast bench geometries—including burden ($B$), spacing ($S$), bench height ($H$), sub-drilling ($J$), stemming length ($T$), and charge weight ($Q$)—to achieve target powder factors ($q$) aligned with rock mass blastability indices.
3. **Quantify** downstream comminution impacts using Bond's and Morrell's grinding laws, calculating the economic trade-off between blast chemical energy expenditure and SAG mill mechanical power draw.
4. **Reconstruct** post-blast grade control polygons using Blast Movement Monitoring (BMM) 3D displacement vectors, computing the financial mitigation of contact dilution and ore misallocation.
5. **Deconstruct** haul truck cycle times from physical first principles, synthesizing vehicle motion profiles using rimpull curves, gross vehicle weight (GVW), grade resistance ($GR$), rolling resistance ($RR$), and Ton-Kilometer Per Hour (TKPH) tire limits.
6. **Model** open-pit shovel-truck loading and hauling interactions using closed cyclic queuing theory (Gordon-Newell networks and Mean Value Analysis), deriving exact match factors and queuing delays.
7. **Formulate and execute** a two-stage dynamic dispatch optimization model that combines an LP macro-level production/blending flow network with real-time heuristic truck assignment.
8. **Diagnose** operational failure modes across the D-B-L-H interface, specifying preventive engineering controls for flyrock, hard toes, crusher bridging, truck bunching, and tire heat delamination.

---

## 3. The core mental model

The extraction-to-processing interface is a continuous physical and thermodynamic pipeline where decisions made at the drill rig propagate irreversibly through the blast bench, shovel bucket, haul road, primary crusher, and grinding mills. 

```text
=============================================================================================================
                      THE MINE-TO-MILL PHYSICAL AND INFORMATION FLOWCHAIN
=============================================================================================================

 [ GEOLOGICAL MODEL ]                [ BLAST DESIGN ]                      [ MUCKPILE DYNAMICS ]
 - Rock Mass Rating (RMD, JPS)       - Hole Diameter (D), Burden (B)       - Gas Heave & Swelling (Sw)
 - UCS, Young's Modulus (E)          - Spacing (S), Stemming (T)           - 3D Displacement Vectors
 - Geochemical Grade Polygons        - Powder Factor q (kg/m3)             - Blast Movement Transmitters
           │                                       │                                       │
           ▼                                       ▼                                       ▼
 ┌──────────────────────┐                ┌──────────────────────┐                ┌──────────────────────┐
 │ Rotary Blast Hole    │──Drill Penet.─►│ Heavy ANFO / Emulsion│──Controlled───►│ Fragmented Muckpile  │
 │ Drilling & Assay Log │   Rate Sensor  │ Detonation (VOD)     │   Blast Heave  │ (P80, Fines, Boulders│
 └──────────────────────┘                └──────────────────────┘                └──────────┬───────────┘
                                                                                            │
 ┌──────────────────────────────────────────────────────────────────────────────────────────┘
 │
 ▼
 [ EXCAVATION & DISPATCH ]           [ LOGISTICS NETWORK ]                 [ COMMINUTION CIRCUIT ]
 - Shovel Diggability Index          - Rimpull / Grade Resistance          - Primary Gyratory Crusher
 - Dynamic LP Fleet Dispatch         - Fleet Match Factor (MF)             - Coarse Ore Stockpile (COS)
 - Post-Blast Polygon Tracking       - Closed-Loop Queuing Net             - SAG & Ball Mill Comminution
           │                                       │                                       │
           ▼                                       ▼                                       ▼
 ┌──────────────────────┐                ┌──────────────────────┐                ┌──────────────────────┐
 │ Hydraulic Excavator  │──Payload Data─►│ Ultra-Class Trucks   │──Crusher Dump─►│ Semi-Autogenous Mill │
 │ / Rope Shovel Bucket │   Telemetry    │ (240t-400t Haulage)  │   Pocket Feed  │ (Throughput t/h, kWh)│
 └──────────────────────┘                └──────────────────────┘                └──────────────────────┘
```

### The Causal Loops Governing the Interface

1. **The Fragmentation-Throughput Forward Loop:** Blast design parameters ($B, S, q, \text{VOD}$) determine the stress wave propagation and gas expansion in the rock mass. Higher explosive energy concentration fractures rock along micro-structural grain boundaries, increasing the passing fraction of fines ($-12.7\text{ mm}$) and reducing $P_{80}$. As $P_{80}$ drops, primary crusher throughput increases, crusher discharge is finer, and SAG mill specific energy consumption ($E_{\text{SAG}}$ in $\text{kWh/t}$) drops non-linearly. The SAG mill grind rate surges, lifting overall mine site copper or gold metal throughput.
2. **The Blast Heave and Dilution Feedback Loop:** Explosive detonation imparts momentum to the fractured rock mass. The bench face moves outward and upward, distorting the pre-blast geological boundaries. If the mining shovel excavates according to pre-blast surveyed blocks, high-grade ore is misallocated to the waste dump (permanent revenue loss), and waste rock is hauled to the primary crusher (mill dilution). Measuring 3D displacement vectors via embedded radio transmitters (BMM) allows software to digitally translate and deform the grade polygons to match the muckpile reality.
3. **The Cyclic Queuing and Fleet Balancing Loop:** Shovels act as stochastic service stations; haul trucks act as circulating customers in a closed network. If the ratio of truck arrival rate to shovel service rate (Match Factor) deviates from the network optimum, either shovels sit idle waiting for trucks (hauler deficit, loading asset capital wasted) or trucks queue in long lines at the shovel bench (hauler surplus, diesel wasted, engine hours consumed unproductive). Dynamic dispatch continuously redistributes trucks based on real-time travel times, grade blending envelopes at the crusher, and dump pocket queue depths.

```text
=============================================================================================================
               PRE-BLAST VERSUS POST-BLAST GRADE CONTROL POLYGON WARPING
=============================================================================================================

 PRE-BLAST IN-SITU BENCH (Top View)                  POST-BLAST HEAVED MUCKPILE (Top View)
 ┌──────────────────────┬──────────────────────┐     ┌──────────────────────┬──────────────────────┐
 │                      │                      │     │      /~~~~~~~~~~~~~~~\      /~~~~~~~~~~~~~~~\
 │      WASTE BLOCK     │      ORE BLOCK       │     │     /   WASTE BLOCK   \    /   ORE BLOCK     \
 │      (0.12% Cu)      │      (0.85% Cu)      │     │    │    DISPLACED      │  │    DISPLACED      │
 │                      │                      │     │    │    BY 8.5 METERS  │  │    BY 12.0 METERS │
 │                      │                      │     │     \                 /    \                 /
 └──────────────────────┴──────────────────────┘     └──────\~~~~~~~~~~~~~~~/──────\~~~~~~~~~~~~~~~/
           │                       │                             │                       │
           ▼                       ▼                             ▼                       ▼
    [ Pre-Blast Line ]      [ Pre-Blast Line ]            [ BMM Vector ]          [ Restructured Post- ]
    Surveyed Dig Boundary   Assumed Vertical              Translation (dx,dy)     Blast Dig Boundary
```

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse / Pitfall | Unit / Standard Notation |
|---|---|---|---|
| **Powder Factor ($q$)** | Mass of explosive detonated per unit volume or unit mass of broken rock. | Confusing volumetric powder factor ($\text{kg/m}^3$) with mass powder factor ($\text{kg/t}$). | $\text{kg/m}^3$ or $\text{kg/t}$ ($q_v = \rho_r \cdot q_w$) |
| **Burden ($B$)** | Distance from a blasthole to the nearest free face, or distance between consecutive rows of holes. | Conflating drill burden (drilled spacing) with effective burden at initiation timing. | Meters ($\text{m}$) |
| **Spacing ($S$)** | Distance between adjacent blastholes within the same row, measured perpendicular to burden. | Treating spacing as independent of burden; spacing must scale with burden ($S/B$ ratio). | Meters ($\text{m}$) |
| **Sub-drilling ($J$)** | Depth drilled below the planned bench floor grade to ensure breakage to grade. | Eliminating sub-drill to save drilling costs, resulting in hard unbroken rock toes. | Meters ($\text{m}$) |
| **Stemming ($T$)** | Inert material (crushed aggregate) packed into the top of a blasthole to confine explosive gases. | Using fine drill cuttings instead of sized angular gravel, leading to stemming blowout and flyrock. | Meters ($\text{m}$) |
| **Velocity of Detonation (VOD)** | Linear speed at which the detonation reaction front traverses the explosive column. | Assuming factory VOD occurs in damp, unconfined, or small-diameter holes. | $\text{m/s}$ (typically 3,500 to 6,500) |
| **Relative Weight Strength (RWS)** | Energy output per unit mass of explosive relative to standard ANFO (ANFO = 100). | Confusing weight strength with bulk strength (RBS incorporates density). | Dimensionless index (ANFO = 100) |
| **Blastability Index ($A$)** | Empirical rock mass factor (Lilly's formula) characterizing resistance to fragmentation. | Using laboratory UCS alone without incorporating joint spacing and dip orientations. | Dimensionless (typically 2 to 14) |
| **Kuz-Ram Model** | Semi-empirical model predicting fragmentation size distribution based on blast geometry. | Applying Kuz-Ram to predict micro-fines ($-1\text{ mm}$); Kuz-Ram severely underpredicts fines. | Set of algebraic equations |
| **Swebrec Function** | Three-parameter analytical distribution modeling fragmentation across coarse and fine ranges. | Mistaking the inflection point parameter $b$ for the Kuz-Ram uniformity index $n$. | Cumulative passing $P(x) \in [0, 1]$ |
| **Passing Size ($P_{80}$)** | Sieve opening dimension through which 80% of the fragmented rock mass by weight passes. | Treating $P_{80}$ as the maximum fragment size; 20% of the rock mass is larger than $P_{80}$. | Millimeters ($\text{mm}$) or Meters ($\text{m}$) |
| **Mine-to-Mill (M2M)** | Holistic strategy optimizing drilling and blasting to minimize total comminution and extraction cost. | Viewing M2M as "just blasting harder"; it requires tuned timing, sizing, and mill control. | Operational philosophy / framework |
| **Bond Work Index ($W_i$)** | Comminution parameter expressing resistance of rock to crushing and grinding. | Assuming $W_i$ is invariant; blast micro-fracturing reduces effective operating work index. | $\text{kWh/metric tonne}$ |
| **Semi-Autogenous Grinding (SAG)** | Tumbling mill using a combination of large ore chunks and steel balls for grinding. | Overfilling SAG with coarse rocks ($>250\text{ mm}$) which acts as un-grindable tramp charge. | Processing asset archetype |
| **Blast Movement Monitoring (BMM)** | System of subsurface radio transmitters tracked post-blast to measure 3D heave vectors. | Assuming blast heave is uniform across depth; surface heave exceeds bottom bench heave. | 3D vector $\vec{d} = (dx, dy, dz)$ in meters |
| **Contact Dilution** | Waste rock excavated and processed as ore due to imprecise boundary delineation. | Confusing contact dilution with internal dilution (mineralized waste inside ore vein). | Percentage mass ($\%$) |
| **Ore Loss (Misallocation)** | Economically viable ore excavated and hauled to the waste dump due to boundary errors. | Believing ore loss can be recovered later; dumped low-grade ore oxidizes and is lost. | Percentage mass ($\%$) or Lost Net Present Value ($\$$) |
| **Swell Factor ($S_w$)** | Ratio of broken rock volume to in-situ bank rock volume ($S_w = V_{\text{loose}}/V_{\text{bank}}$). | Ignoring swell when sizing truck haul bodies, causing payload weight under-utilization. | Ratio (typically 1.30 to 1.55) |
| **Match Factor ($MF$)** | Ratio of truck fleet arrival rate to shovel loading capacity. | Operating at $MF > 1.2$ under high fuel prices, causing massive truck queuing expenses. | Dimensionless ratio ($MF = 1.0$ is balanced) |
| **Rimpull** | Tractive force exerted by the truck's drive tires on the ground surface to propel motion. | Overestimating rimpull on slippery roads where wheel slip limits usable traction. | Kilonewtons ($\text{kN}$) or Kilograms-force ($\text{kgf}$) |
| **Rolling Resistance ($RR$)** | Opposing force resulting from tire flexing and tire penetration into the haul road surface. | Assuming $RR$ is fixed at 2%; unmaintained muddy haul roads easily reach 5% to 8% $RR$. | Percent equivalent grade ($\%$) or $\text{kg/t}$ |
| **Grade Resistance ($GR$)** | Gravitational force opposing vehicle motion on an uphill slope ($GR \approx \text{Grade } \%$). | Forgetting that downhill grade provides negative resistance (assisting motion up to retarder limit). | Percent equivalent grade ($\%$) or $\text{kg/t}$ |
| **Total Resistance ($TR$)** | Algebraic sum of grade resistance and rolling resistance ($TR = GR + RR$). | Negating rolling resistance on downhill runs; rolling resistance always opposes vehicle travel. | Percent equivalent grade ($\%$) |
| **Ton-Kilometer Per Hour (TKPH)** | Measure of haul truck tire thermal workload capacity ($TKPH = \text{Mean Load} \times \text{Avg Speed}$). | Exceeding tire manufacturer TKPH rating, resulting in catastrophic tire carcass delamination. | $\text{t}\cdot\text{km/h}$ |
| **Closed Cyclic Queuing** | Queuing system where a fixed population of customers (trucks) circulate among servers. | Using open $M/M/1$ queues for haul fleets; open models ignore finite truck population feedback. | Queueing network topology |
| **Dynamic Dispatching** | Real-time mathematical assignment of trucks to shovels upon completing a dump cycle. | Static fixed-truck assignment, which induces severe stochastic truck bunching. | LP / Dynamic Programming engine |

---

## 5. The mechanics, step by step

Optimizing drilling, blasting, loading, and hauling requires an interconnected sequence of physical, geotechnical, and mathematical operations.

```text
+---------------------------------------------------------------------------------------------------+
|                           THE 8-STEP D-B-L-H OPERATIONAL OPTIMIZATION PIPELINE                     |
+---------------------------------------------------------------------------------------------------+
|  [Step 1] In-situ Rock Mass Characterization (Lilly's Blastability Index A)                       |
|     │                                                                                             |
|     ▼                                                                                             |
|  [Step 2] Geometric Blast Pattern Design (Burden B, Spacing S, Stemming T, Sub-drill J, Powder q) |
|     │                                                                                             |
|     ▼                                                                                             |
|  [Step 3] Fragmentation Modeling via Modified Kuz-Ram and Swebrec Distribution                    |
|     │                                                                                             |
|     ▼                                                                                             |
|  [Step 4] Downstream Comminution Energy Coupling (Bond & Morrell Mine-to-Mill Sizing)             |
|     │                                                                                             |
|     ▼                                                                                             |
|  [Step 5] Blast Movement Vector Measurement & Post-Blast Dig Block Translation (BMM)              |
|     │                                                                                             |
|     ▼                                                                                             |
|  [Step 6] Haul Cycle Physics Decomposition (Rimpull, Retarder, Grade, Rolling Resistance, TKPH)  |
|     │                                                                                             |
|     ▼                                                                                             |
|  [Step 7] Shovel-Truck Fleet Matching & Closed Cyclic Queuing Modeling (Gordon-Newell Network)    |
|     │                                                                                             |
|     ▼                                                                                             |
|  [Step 8] Two-Stage Dynamic Shovel-Truck Dispatch Optimization (Macro-LP + Micro-Assignment)      |
+---------------------------------------------------------------------------------------------------+
```

### Step 1: In-situ Rock Mass Characterization and Blastability Indexing

The response of rock to blasting is governed by the structural discontinuities (joints, bedding planes, faults) and mechanical strength of the intact rock. The blasting engineer must compute Lilly's **Blastability Index ($A$)**:

$$A = 0.06 \times (\text{RMD} + \text{JPS} + \text{JPO} + \text{SMR}) + 0.1 \times \text{density} + 0.05 \times \text{UCS}$$

Where the parameters are scored according to geotechnical logging:
1. **Rock Mass Description ($\text{RMD}$):**
   - Powdery / friable rock: $\text{RMD} = 10$
   - Blocky rock: $\text{RMD} = 20$
   - Totally massive rock: $\text{RMD} = 50$
2. **Joint Plane Spacing ($\text{JPS}$):**
   - Close spacing ($< 0.1\text{ m}$): $\text{JPS} = 10$
   - Intermediate spacing ($0.1\text{--}1.0\text{ m}$): $\text{JPS} = 20$
   - Wide spacing ($> 1.0\text{ m}$): $\text{JPS} = 50$
3. **Joint Plane Orientation ($\text{JPO}$):**
   - Horizontal dip: $\text{JPO} = 10$
   - Dip out of face (strike parallel to face): $\text{JPO} = 20$
   - Dip into face: $\text{JPO} = 30$
   - Strike normal to face: $\text{JPO} = 40$
4. **Specific Gravity / Density Influence ($\text{SMR}$):**
   - Based on Young's Modulus ($E$, GPa) and Unconfined Compressive Strength ($\text{UCS}$, MPa): $\text{SMR} = 0.06 \times E$.
5. **Density ($\rho_r$ in $\text{t/m}^3$) and $\text{UCS}$ (in MPa).**

Typical values for $A$ range from $2$ (extremely soft, friable sandstone/shale) to $14$ (massive, unjointed taconite or basalt).

### Step 2: Geometric Blast Pattern Design

Given blasthole diameter $D$ (in millimeters, typically $200\text{--}311\text{ mm}$ for large surface mines) and bench height $H$ (typically $10\text{--}15\text{ m}$):

1. **Burden ($B$ in meters):**
   $$B = 0.03 \times D \times \sqrt{\frac{\rho_e \cdot \text{RWS}}{\rho_r}}$$
   Where $\rho_e$ is explosive density ($\text{g/cm}^3$), $\text{RWS}$ is Relative Weight Strength, and $\rho_r$ is rock density ($\text{g/cm}^3$). As a rule of thumb, $B \approx (25\text{--}35) \times \frac{D}{1000}$.
2. **Spacing ($S$ in meters):**
   - For staggered patterns: $S = 1.15 \times B$ to $1.25 \times B$.
   - For square patterns: $S = 1.0 \times B$.
3. **Sub-drilling ($J$ in meters):**
   $$J = (0.2\text{--}0.3) \times B$$
   Sub-drilling ensures the explosive charge penetrates beneath the planned bench floor, eliminating floor toes that damage shovel undercarriages and truck tires.
4. **Stemming Length ($T$ in meters):**
   $$T = (0.7\text{--}1.0) \times B$$
   Stemming must consist of crushed angular aggregate sized to $D/10$ to ensure particle interlocking and prevent premature gas venting (cratering/rifling).
5. **Total Hole Depth ($L$) and Explosive Charge Weight ($Q$):**
   $$L = H + J$$
   $$L_{\text{charge}} = L - T$$
   $$Q = L_{\text{charge}} \times \left( \frac{\pi}{4} \left(\frac{D}{1000}\right)^2 \right) \times (\rho_e \times 1000) \quad (\text{kg/hole})$$
6. **Powder Factor ($q$):**
   - Broken rock volume per hole: $V_b = B \times S \times H \quad (\text{m}^3)$
   - Volumetric powder factor: $q_v = \frac{Q}{V_b} \quad (\text{kg/m}^3)$
   - Mass powder factor: $q_w = \frac{q_v}{\rho_r} \quad (\text{kg/t})$

### Step 3: Fragmentation Modeling via Modified Kuz-Ram and Swebrec

The **Kuz-Ram model** (Cunningham, 1983, 1987) calculates the complete particle size distribution resulting from blasting:

1. **Mean Fragment Size ($x_{50}$ in cm):**
   $$x_{50} = A \cdot q_v^{-0.8} \cdot Q^{1/6} \cdot \left(\frac{115}{\text{RWS}}\right)^{19/30}$$
   Where $A$ is the Blastability Index, $q_v$ is volumetric powder factor ($\text{kg/m}^3$), $Q$ is explosive mass per hole (kg), and $\text{RWS}$ is Relative Weight Strength (ANFO = 100).
2. **Uniformity Index ($n$):**
   $$n = \left(2.2 - 14\frac{B}{D}\right) \cdot \sqrt{\frac{1 + S/B}{2}} \cdot \left(1 - \frac{W}{B}\right) \cdot \left(\frac{L_{\text{charge}}}{H}\right)^{0.1} \cdot \left(\frac{|L_{\text{bottom}} - L_{\text{top}}|}{L_{\text{charge}}} + 0.1\right)^{0.1} \cdot \left(1 + \frac{S-B}{H}\right)$$
   Where $D$ is hole diameter in mm, and $W$ is drilling standard deviation (m). In modern practice without air-decking, Cunningham's simplified uniformity formulation is:
   $$n = \left(2.2 - 14\frac{B}{D}\right) \cdot \sqrt{\frac{1 + S/B}{2}} \cdot \left(1 - \frac{W}{B}\right) \cdot \left(1 + \frac{S - B}{H}\right)^{0.1} \cdot \left(\frac{L_{\text{charge}}}{H}\right)$$
   Higher $n$ ($>1.4$) indicates a uniform fragment size; low $n$ ($<0.8$) indicates high fines and oversized boulders.
3. **Characteristic Size ($x_c$ in cm):**
   $$x_c = \frac{x_{50}}{(\ln 2)^{1/n}}$$
4. **Cumulative Passing Distribution ($P(x)$):**
   $$P(x) = 1 - \exp\left[ -\left(\frac{x}{x_c}\right)^n \right] = 1 - \exp\left[ -\ln(2) \left(\frac{x}{x_{50}}\right)^n \right]$$
5. **The 80% Passing Size ($P_{80}$):**
   $$P_{80} = x_c \cdot \left[ -\ln(1 - 0.80) \right]^{1/n} = x_c \cdot (\ln 5)^{1/n}$$

> **Note on Model Boundaries:** The classic Kuz-Ram equation underpredicts the production of fine particles ($-1.0\text{ mm}$) because it assumes a single Rosin-Rammler distribution across all fracture mechanisms. In reality, rock fragmentation occurs via two distinct physical regimes: tensile spalling/gas expansion (governing coarse macro-blocks) and compressive shear pulverization around the borehole wall (governing micro-fines). 
>
> Modern Mine-to-Mill programs use the **Swebrec distribution** (Ouchterlony, 2005) to model fines and top-size boulders accurately:
> $$P(x) = \frac{1}{1 + \left[ \frac{\ln(x_{\max}/x)}{\ln(x_{\max}/x_{50})} \right]^b}$$
> Where $x_{\max}$ is the in-situ block size or maximum allowable fragment, and $b$ is the curve curvature parameter.

### Step 4: Downstream Comminution Energy Coupling (Mine-to-Mill)

To connect blast fragmentation to mill economics, apply **Bond's Third Theory of Comminution**:

$$W_{\text{mill}} = 10 \cdot W_i \cdot \left( \frac{1}{\sqrt{P_{80}}} - \frac{1}{\sqrt{F_{80}}} \right)$$

Where:
- $W_{\text{mill}}$ is specific grinding energy ($\text{kWh/t}$).
- $W_i$ is the Bond Ball Mill Work Index ($\text{kWh/t}$).
- $F_{80}$ is the 80% passing size of the feed to the mill ($\mu\text{m}$).
- $P_{80}$ is the 80% passing size of the final product to flotation ($\mu\text{m}$).

In Semi-Autogenous Grinding (SAG) mills, the relationship is governed by the **JKMRC / Morrell Comminution Model**, where specific energy $E_{\text{SAG}}$ depends on the drop-weight impact breakage parameter ($A \times b$) and feed size:

$$E_{\text{SAG}} = K \cdot (A \times b)^{-0.75} \cdot F_{80}^{0.3} \cdot (1 - 0.6 \cdot f_{-12.7})$$

Where $f_{-12.7}$ is the mass fraction of feed finer than $12.7\text{ mm}$ (generated directly by blasting). 

Increasing powder factor $q$ generates micro-cracks in intact ore chunks, reducing the effective resistance to impact breakage and increasing $f_{-12.7}$. This reduces $E_{\text{SAG}}$, allowing the mill operator to increase volumetric feed rate ($t/\text{h}$) up to the mill motor power limit:

$$\text{Throughput}_{\text{SAG}} \left(\frac{t}{\text{h}}\right) = \frac{P_{\text{installed}} \times \eta_{\text{motor}}}{E_{\text{SAG}}}$$

Where $P_{\text{installed}}$ is installed motor power ($\text{kW}$) and $\eta_{\text{motor}}$ is electrical drive efficiency.

### Step 5: Blast Displacement Dynamics and Dilution Control

During detonation, high-pressure gases expand into the fractured burden, ejecting the rock mass forward and upward.

```text
=============================================================================================================
                          BLAST HEAVE VECTOR RESOLUTION AND CONTACT DILUTION
=============================================================================================================

 Pre-Blast Face            Detonation Heave Trajectory
 ───┐                      .- - - - - - - - - .
    │                    .'                    `.   Post-Blast Muckpile Surface
    │                  .'                        `. 
    │                 /                            \
    │                /    Displacement Vector       \
    │               /     -->  ->                    \
    │              /      d = (dx, dy, dz)            \
    │             /                                    \
 ───┴────────────┴──────────────────────────────────────┴───────────────
 Floor           Toe                                    Muckpile Toe
```

1. **3D Displacement Vectors:** Blast Movement Monitoring (BMM) transmitters embedded in vertical holes at depths $z_1, z_2$ measure the displacement vector $\vec{d}_k = (\Delta X_k, \Delta Y_k, \Delta Z_k)$.
   - Surface rock experiences maximum horizontal displacement: $\Delta X_{\text{surf}} = 10\text{--}25\text{ m}$.
   - Deep bench rock near the floor experiences restrained displacement: $\Delta X_{\text{floor}} = 2\text{--}6\text{ m}$.
2. **Polygon Transformation:** Let a pre-blast polygon boundary be a set of vertices $V_{\text{pre}} = \{(X_i, Y_i)\}$. The post-blast mining polygon $V_{\text{post}}$ is reconstructed by applying a spatial interpolation field $\mathbf{T}(\vec{x})$ derived from BMM readings:
   $$V_{\text{post}} = \{ (X_i, Y_i) + \mathbf{T}(X_i, Y_i) \}$$
3. **Dilution and Ore Loss Formulations:**
   If excavation proceeds along uncorrected pre-blast lines $V_{\text{pre}}$ instead of $V_{\text{post}}$:
   - **Contact Dilution Mass ($M_{\text{dil}}$):** Waste material physically located inside the uncorrected dig polygon:
     $$\text{Dilution } \% = \frac{M_{\text{waste\_to\_mill}}}{M_{\text{ore\_to\_mill}} + M_{\text{waste\_to\_mill}}} \times 100\%$$
   - **Ore Loss Mass ($M_{\text{loss}}$):** Ore heaved outside the pre-blast line into the adjacent waste dig block:
     $$\text{Ore Loss } \% = \frac{M_{\text{ore\_to\_waste}}}{M_{\text{total\_in-situ\_ore}}} \times 100\%$$

### Step 6: Haul Cycle Physics and Decomposition

A haul truck's total cycle time ($T_{\text{cycle}}$) is decomposed into deterministic and stochastic components:

$$T_{\text{cycle}} = t_{\text{spot\_shovel}} + t_{\text{load}} + t_{\text{haul\_loaded}} + t_{\text{spot\_dump}} + t_{\text{dump}} + t_{\text{return\_empty}} + t_{\text{queue}}$$

The transit times ($t_{\text{haul\_loaded}}$ and $t_{\text{return\_empty}}$) are calculated by integrating the vehicle equations of motion across road segments $k \in \{1, \dots, K\}$ of length $L_k$ and slope $\theta_k$:

1. **Total Resistance ($TR_k$ in % equivalent grade):**
   $$TR_k = GR_k + RR_k = \tan(\theta_k) \times 100 + RR_k$$
   Where $GR$ is Grade Resistance and $RR$ is Rolling Resistance (typically $1.5\%\text{--}2.5\%$ on well-maintained hard-packed haul roads; $3.5\%\text{--}6.0\%$ on soft, unmaintained, or wet roads).
2. **Propulsion (Rimpull-Limited Speed):**
   The tractive force $F_{\text{rimpull}}$ delivered to the ground by a diesel-electric or mechanical drive is:
   $$F_{\text{rimpull}} = \frac{3600 \cdot P_{\text{net}} \cdot \eta_{\text{drive}}}{v} \quad (\text{N})$$
   Where $P_{\text{net}}$ is engine flywheel power ($\text{kW}$), $\eta_{\text{drive}}$ is drive-train efficiency ($0.82\text{--}0.88$), and $v$ is velocity ($\text{km/h}$).
   Vehicle terminal velocity uphill is reached when tractive force equals resisting force:
   $$F_{\text{rimpull}}(v_{\text{term}}) = \frac{TR_k}{100} \cdot \text{GVW} \cdot g$$
   Where $\text{GVW}$ is Gross Vehicle Weight (Empty Vehicle Weight $\text{EVW} + \text{Payload}$) in kg, and $g = 9.81\ \text{m/s}^2$.
3. **Braking / Retarder-Limited Speed:** On downhill loaded runs ($GR_k < 0$), if $|GR_k| > RR_k$, the truck is in retarding mode. Maximum speed $v_{\text{retard}}$ is strictly constrained by the manufacturer's continuous electric retarding or hydraulic dynamic brake thermal absorption capacity ($P_{\text{retard\_kW}}$):
   $$v_{\text{downhill}} \le \frac{3600 \cdot P_{\text{retard\_kW}}}{|TR_k|/100 \cdot \text{GVW} \cdot g}$$
4. **Tire Thermal Rating (TKPH):**
   $$\text{TKPH}_{\text{operating}} = Q_{\text{mean\_tire}} \times V_{\text{shift\_avg}}$$
   $$Q_{\text{mean\_tire}} = \frac{W_{\text{tire\_empty}} + W_{\text{tire\_loaded}}}{2} \quad (\text{tonnes})$$
   $$V_{\text{shift\_avg}} = \frac{\text{Total Distance Run in Shift (km)}}{\text{Total Operating Hours in Shift (h)}}$$
   If $\text{TKPH}_{\text{operating}} > \text{TKPH}_{\text{tire\_rating}}$, trucks must be artificially speed-governed or rested to prevent internal carcass rubber reversion and catastrophic blowouts.

### Step 7: Shovel-Truck Fleet Matching and Queuing Mechanics

In an open-pit operation with $M$ shovels and $N$ trucks, matching capacity is vital:

1. **The Classic Match Factor ($MF$):**
   $$MF = \frac{N_{\text{trucks}} \times t_{\text{load}}}{M_{\text{shovels}} \times T_{\text{cycle}}}$$
   - $MF < 1.0$: Fleet is **truck-limited**. Shovels experience idle waiting time. Shovel utilization drops.
   - $MF > 1.0$: Fleet is **shovel-limited**. Trucks experience queuing delays at shovels. Truck efficiency drops.
2. **Closed Cyclic Queuing Network (Gordon-Newell Network):**
   Because trucks circulate continuously in a closed loop, standard open Poisson queuing formulas ($M/M/1$) fail. The system is modeled as a closed network with $N$ circulating customers.
   
   Using **Mean Value Analysis (MVA)** for a system with $S$ service stations (shovels, haul routes, crushers/dumps):
   - Mean waiting time at station $i$ with $n$ trucks in the network:
     $$W_i(n) = \begin{cases} 
     \frac{1}{\mu_i} \left[ 1 + L_i(n-1) \right] & \text{for single-server queues (shovels, single-dump crushers)} \\
     \frac{1}{\mu_i} & \text{for infinite-server delay lines (haul roads)}
     \end{cases}$$
   - System throughput with $n$ trucks:
     $$\lambda(n) = \frac{n}{\sum_{i=1}^{S} V_i W_i(n)}$$
   - Average queue length at station $i$:
     $$L_i(n) = \lambda(n) \cdot V_i \cdot W_i(n)$$
   Where $\mu_i$ is service rate, $V_i$ is the visit ratio of station $i$, and $L_i(n-1)$ is the queue length from the previous recursive step.

### Step 8: Two-Stage Dynamic Shovel-Truck Dispatch Optimization

To defeat stochastic truck bunching, computerized dispatch systems (e.g., Modular Mining DISPATCH, Caterpillar MineStar, Komatsu FrontRunner) execute a two-stage hierarchical algorithm:

```text
=============================================================================================================
                    TWO-STAGE HIERARCHICAL DYNAMIC DISPATCH ARCHITECTURE
=============================================================================================================

 [ STAGE 1: MACRO-LEVEL LP OPTIMIZER ] (Runs every 10-30 min or on event)
 - Objective: Minimize Total Haul Cost OR Maximize Total Production Rate
 - Subject to: Shovel Digging Limits, Crusher Throughput, Blending Windows, Truck Fleet Conservation
           │
           ▼  Produces Target Route Flow Rates x_ij (tonnes/hour)
 ┌────────────────────────────────────────────────────────────────────────┐
 │ Route S1 -> Crusher: 2,400 t/h  │  Route S2 -> Crusher: 1,600 t/h      │
 │ Route S2 -> Waste:     800 t/h  │  Route S3 -> Waste:   3,500 t/h      │
 └────────────────────────────────┬───────────────────────────────────────┘
                                  │
                                  ▼
 [ STAGE 2: REAL-TIME DYNAMIC ASSIGNMENT ENGINE ] (Runs upon each truck arrival at dump/diverge point)
 - Truck k empties at Crusher or Dump Pocket.
 - Query Real-Time State: Current shovel queue lengths q_i, truck travel times t_ik, shovel starvation risk.
 - Objective: Select Shovel i* that minimizes deviation from LP rates while minimizing idle queuing:
   i* = argmin_i { Cost_i(t) = w1 * (Expected Arrival Time - Shovel Available Time) + w2 * Rate_Deficit_i }
```

#### Stage 1 Linear Programming Formulation

Let:
- $\mathcal{S}$ be the set of active shovels ($s \in \mathcal{S}$).
- $\mathcal{D}$ be the set of dump locations ($d \in \mathcal{D}$), including crushers, waste dumps, and low-grade stockpiles.
- $\mathcal{R} \subseteq \mathcal{S} \times \mathcal{D}$ be the set of viable haul routes $(s, d)$.
- $x_{sd}$ be the material flow rate on route $(s, d)$ in metric tonnes per hour.
- $c_{sd}$ be the haulage cost per tonne on route $(s, d)$, typically proportional to travel time $\tau_{sd}$.
- $C_s$ be the maximum digging capacity of shovel $s$ ($t/\text{h}$).
- $K_d^{\min}, K_d^{\max}$ be the lower and upper processing limits of dump/crusher $d$ ($t/\text{h}$).
- $g_{sk}$ be the grade or quality attribute $k$ of material dug by shovel $s$ (e.g., $\% \text{Cu}$, $\% \text{Fe}$, $\text{ppm As}$, moisture $\%$).
- $G_{dk}^{\min}, G_{dk}^{\max}$ be the grade blending specification limits for attribute $k$ at destination $d$.
- $P$ be the nominal truck payload (tonnes).
- $\tau_{sd}$ be the total round-trip cycle time on route $(s, d)$ excluding queue delays (hours).
- $N_{\text{trucks}}$ be the total operational haul truck fleet count.

$$\min \sum_{(s, d) \in \mathcal{R}} c_{sd} \cdot x_{sd}$$

Subject to:

1. **Shovel Production Limits:**
   $$\sum_{d : (s, d) \in \mathcal{R}} x_{sd} \le C_s \quad \forall s \in \mathcal{S}$$
2. **Dump / Crusher Throughput Capacity:**
   $$K_d^{\min} \le \sum_{s : (s, d) \in \mathcal{R}} x_{sd} \le K_d^{\max} \quad \forall d \in \mathcal{D}$$
3. **Crusher Feed Grade / Quality Blending Windows:**
   $$\sum_{s : (s, d) \in \mathcal{R}} g_{sk} \cdot x_{sd} \ge G_{dk}^{\min} \sum_{s : (s, d) \in \mathcal{R}} x_{sd} \quad \forall d \in \mathcal{D}_{\text{crusher}}, \forall k$$
   $$\sum_{s : (s, d) \in \mathcal{R}} g_{sk} \cdot x_{sd} \le G_{dk}^{\max} \sum_{s : (s, d) \in \mathcal{R}} x_{sd} \quad \forall d \in \mathcal{D}_{\text{crusher}}, \forall k$$
4. **Total Haul Fleet Availability (Truck Conservation):**
   The number of trucks required on route $(s, d)$ to sustain flow $x_{sd}$ is $\frac{x_{sd} \cdot \tau_{sd}}{P}$. Therefore:
   $$\sum_{(s, d) \in \mathcal{R}} \frac{\tau_{sd}}{P} \cdot x_{sd} \le N_{\text{trucks}}$$
5. **Non-Negativity:**
   $$x_{sd} \ge 0 \quad \forall (s, d) \in \mathcal{R}$$

#### Stage 2 Dynamic Real-Time Assignment

When truck $k$ discharges at destination $d$, the dispatch engine computes the optimal shovel $s^*$ to assign by evaluating the **Need and Timing Heuristic (Modified Beenhakker Rule)**:

$$s^* = \arg\max_{s} \left\{ \Delta \text{Need}_s - \omega \cdot \max(0, t_{\text{arrival}}(k, s) - t_{\text{ready}}(s)) \right\}$$

Where:
- $\Delta \text{Need}_s = \text{TargetLPFlow}_s - \text{DeliveredFlow}_s$ is the cumulative tonnage deficit.
- $t_{\text{arrival}}(k, s)$ is the estimated arrival time of empty truck $k$ at shovel $s$.
- $t_{\text{ready}}(s)$ is the time shovel $s$ completes loading all previously assigned trucks already en route or in queue.
- $\omega$ is a penalty parameter balancing LP route compliance against truck queuing wait times.

---

## 6. Worked example

### Problem Scenario: The Mt. Kestrel Copper-Gold Mine

The Mt. Kestrel open-pit operation extracts chalcopyrite copper-gold ore hosted in competent granodiorite. The processing plant runs a Semi-Autogenous Grinding (SAG) mill followed by ball milling and froth flotation. The mine extracts $45,000\text{ t/day}$ ($1,875\text{ t/h}$) of ore and $75,000\text{ t/day}$ ($3,125\text{ t/h}$) of waste rock.

We will carry out an end-to-end calculation across four engineering stages:
1. **Sub-problem A:** Blast design and Cunningham Kuz-Ram fragmentation prediction.
2. **Sub-problem B:** Downstream SAG mill specific energy and throughput response.
3. **Sub-problem C:** Blast movement vector resolution and dilution cost calculation.
4. **Sub-problem D:** Haul cycle motion profile and closed-loop fleet sizing.

---

### Sub-Problem A: Blast Design and Kuz-Ram Fragmentation

#### Given Parameters
- Bench Height $H = 15.0\text{ m}$
- Blasthole Diameter $D = 270\text{ mm}$
- Rock Density $\rho_r = 2.70\text{ t/m}^3$ ($2.70\text{ g/cm}^3$)
- Rock Mass Geotechnical Scores: $\text{RMD} = 30$, $\text{JPS} = 20$, $\text{JPO} = 20$, Young's Modulus $E = 50\text{ GPa}$ ($\text{SMR} = 0.06 \times 50 = 3.0$), $\text{UCS} = 120\text{ MPa}$.
- Explosive: Heavy ANFO (60% emulsion, 40% ANFO), density $\rho_e = 1.25\text{ g/cm}^3$, $\text{RWS} = 105$.
- Drilling standard deviation $W = 0.20\text{ m}$.
- Blast Geometry: Burden $B = 6.5\text{ m}$, Spacing $S = 7.8\text{ m}$ (staggered pattern, $S/B = 1.20$).
- Sub-drilling $J = 1.5\text{ m}$, Stemming Length $T = 5.0\text{ m}$ (angular crushed granite).

#### 1. Compute Blastability Index ($A$)
$$A = 0.06 \times (30 + 20 + 20 + 3.0) + 0.1 \times (2.70) + 0.05 \times (120)$$
$$A = 0.06 \times 73.0 + 0.27 + 6.00 = 4.38 + 0.27 + 6.00 = 10.65$$

#### 2. Compute Hole Geometry and Powder Factor
- Total hole depth $L = H + J = 15.0 + 1.5 = 16.5\text{ m}$.
- Charge length $L_{\text{charge}} = L - T = 16.5 - 5.0 = 11.5\text{ m}$.
- Cross-sectional area of blasthole:
  $$\text{Area} = \frac{\pi}{4} \times (0.270)^2 = 0.057256\text{ m}^2$$
- Explosive mass per hole ($Q$):
  $$Q = L_{\text{charge}} \times \text{Area} \times (\rho_e \times 1000) = 11.5 \times 0.057256 \times 1250 = 823.05\text{ kg/hole}$$
- Rock volume per hole ($V_b$):
  $$V_b = B \times S \times H = 6.5 \times 7.8 \times 15.0 = 760.5\text{ m}^3$$
- Rock mass per hole ($M_b$):
  $$M_b = V_b \times \rho_r = 760.5 \times 2.70 = 2,053.35\text{ tonnes}$$
- Volumetric Powder Factor ($q_v$):
  $$q_v = \frac{Q}{V_b} = \frac{823.05}{760.5} = 1.0822\text{ kg/m}^3$$
- Mass Powder Factor ($q_w$):
  $$q_w = \frac{q_v}{\rho_r} = \frac{1.0822}{2.70} = 0.4008\text{ kg/t}$$

#### 3. Compute Cunningham Kuz-Ram Fragmentation Distribution
- **Median Fragment Size ($x_{50}$ in cm):**
  $$x_{50} = A \cdot q_v^{-0.8} \cdot Q^{1/6} \cdot \left(\frac{115}{\text{RWS}}\right)^{19/30}$$
  - $q_v^{-0.8} = (1.0822)^{-0.8} = 0.9383$
  - $Q^{1/6} = (823.05)^{0.16667} = 3.0645$
  - $\left(\frac{115}{105}\right)^{0.63333} = (1.0952)^{0.63333} = 1.0592$
  $$x_{50} = 10.65 \times 0.9383 \times 3.0645 \times 1.0592 = 32.44\text{ cm} = 324.4\text{ mm} \approx 0.324\text{ m}$$

- **Uniformity Index ($n$):**
  Using Cunningham's standard formula:
  $$n = \left(2.2 - 14\frac{B}{D}\right) \sqrt{\frac{1 + S/B}{2}} \left(1 - \frac{W}{B}\right) \left(1 + \frac{S-B}{H}\right)^{0.1} \left(\frac{L_{\text{charge}}}{H}\right)$$
  - Term 1: $14 \times \frac{B}{D} = 14 \times \frac{6.5}{270} = 0.3370 \implies 2.2 - 0.3370 = 1.863$
  - Term 2: $\sqrt{\frac{1 + 1.20}{2}} = \sqrt{1.10} = 1.0488$
  - Term 3: $1 - \frac{W}{B} = 1 - \frac{0.20}{6.5} = 1 - 0.0308 = 0.9692$
  - Term 4: $\left(1 + \frac{7.8 - 6.5}{15.0}\right)^{0.1} = \left(1 + \frac{1.3}{15}\right)^{0.1} = (1.0867)^{0.1} = 1.0083$
  - Term 5: $\frac{L_{\text{charge}}}{H} = \frac{11.5}{15.0} = 0.7667$
  $$n = 1.863 \times 1.0488 \times 0.9692 \times 1.0083 \times 0.7667 = 1.465$$

- **Characteristic Size ($x_c$ in mm):**
  $$x_c = \frac{x_{50}}{(\ln 2)^{1/n}} = \frac{324.4}{(0.69315)^{1/1.465}} = \frac{324.4}{(0.69315)^{0.6826}} = \frac{324.4}{0.7788} = 416.5\text{ mm}$$

- **Passing Size $P_{80}$ (mm):**
  $$P_{80} = x_c \cdot (\ln 5)^{1/n} = 416.5 \cdot (1.6094)^{1/1.465} = 416.5 \cdot (1.6094)^{0.6826} = 416.5 \cdot 1.3831 = 576.1\text{ mm} \approx 0.576\text{ m}$$

- **Fine Fraction (< 12.7 mm):**
  $$P(12.7) = 1 - \exp\left[ -\left(\frac{12.7}{416.5}\right)^{1.465} \right] = 1 - \exp\left[ -(0.03049)^{1.465} \right] = 1 - \exp[-0.00598] = 0.00596 = 0.60\%$$

---

### Sub-Problem B: Mine-to-Mill Comminution Coupling

The mine decides to implement a Mine-to-Mill optimization blast. By tightening the drill pattern to $B = 5.5\text{ m}$, $S = 6.6\text{ m}$, and increasing powder factor to $q_w = 0.58\text{ kg/t}$ ($q_v = 1.566\text{ kg/m}^3$), the fragmentation distribution shifts:
- New $P_{80}$ to primary crusher: drops from $576.1\text{ mm}$ to $340.0\text{ mm}$.
- Fines fraction ($-12.7\text{ mm}$): rises from $0.6\%$ to $9.5\%$.
- Primary Crusher closed-side setting produces SAG mill feed $F_{80}$: drops from $145,000\ \mu\text{m}$ ($145\text{ mm}$) to $95,000\ \mu\text{m}$ ($95\text{ mm}$).

#### 1. SAG Mill Specific Energy Calculation
The plant's operating calibrated Morrell-type SAG equation for this granodiorite is:
$$E_{\text{SAG}} = 28.5 \cdot \left(\frac{F_{80}}{100,000}\right)^{0.35} \cdot (1 - 0.75 \cdot f_{-12.7}) \quad (\text{kWh/t})$$

- **Baseline Blast:**
  $$E_{\text{SAG, baseline}} = 28.5 \cdot \left(\frac{145,000}{100,000}\right)^{0.35} \cdot (1 - 0.75 \times 0.006) = 28.5 \cdot (1.45)^{0.35} \cdot (1 - 0.0045)$$
  $$(1.45)^{0.35} = 1.1390 \implies E_{\text{SAG, baseline}} = 28.5 \times 1.1390 \times 0.9955 = 32.32\text{ kWh/t}$$

- **Optimized Mine-to-Mill Blast:**
  $$E_{\text{SAG, opt}} = 28.5 \cdot \left(\frac{95,000}{100,000}\right)^{0.35} \cdot (1 - 0.75 \times 0.095) = 28.5 \cdot (0.95)^{0.35} \cdot (1 - 0.07125)$$
  $$(0.95)^{0.35} = 0.9822 \implies E_{\text{SAG, opt}} = 28.5 \times 0.9822 \times 0.92875 = 25.99\text{ kWh/t}$$

#### 2. Mill Throughput Gain and Site Economics
The SAG mill has a total effective motor draw of $36,000\text{ kW}$ ($36\text{ MW}$).
- Baseline SAG throughput:
  $$\text{Throughput}_{\text{baseline}} = \frac{36,000\text{ kW}}{32.32\text{ kWh/t}} = 1,113.9\text{ t/h}$$
- Optimized SAG throughput:
  $$\text{Throughput}_{\text{opt}} = \frac{36,000\text{ kW}}{25.99\text{ kWh/t}} = 1,385.1\text{ t/h}$$
- Throughput Expansion:
  $$\Delta \text{Throughput} = 1,385.1 - 1,113.9 = +271.2\text{ t/h} \quad (+24.3\%)$$

- **Economic Net Benefit:**
  - Additional drill and blast cost to increase powder factor: $+\$0.42/\text{t ore}$.
  - Electrical power savings in SAG mill: $32.32 - 25.99 = 6.33\text{ kWh/t}$. At $\$0.09/\text{kWh}$, power saving is:
    $$\text{Power Saving} = 6.33 \times \$0.09 = \$0.57/\text{t ore}$$
  - Net direct processing cost reduction: $\$0.57 - \$0.42 = +\$0.15/\text{t ore}$.
  - On $15,000,000\text{ tonnes/year}$ of processed ore, direct cost savings equal $\$2.25\text{M/year}$, while the $24.3\%$ throughput expansion brings forward copper production generating an additional $\$38.5\text{M}$ in annualized operating margin.

---

### Sub-Problem C: Blast Movement and Dilution Economics

In Bench 1240, an ore-waste contact boundary runs North-South. Pre-blast ore grade is $0.85\% \text{ Cu}$, and waste grade is $0.08\% \text{ Cu}$. The blast heaves material Eastward towards the free face.
- Blast Movement Monitoring (BMM) transmitters reveal that at mid-bench height, rock moved **$\Delta X = +8.4\text{ m}$ East** and heave swell was **$\Delta Z = +2.1\text{ m}$**.
- The boundary zone affected has a length along strike of $120\text{ m}$ and bench height $15.0\text{ m}$.
- Density $\rho_r = 2.70\text{ t/m}^3$.

```text
=============================================================================================================
                      CROSS-SECTION OF UNCORRECTED DIGGING VS BMM-CORRECTED DIGGING
=============================================================================================================
 Pre-Blast Dig Line (Surveyed Floor Mark)
          │
          │             Actual Heaved Contact
          │             (Shifted 8.4 m East)
          │                      │
          ▼                      ▼
  [ WASTE MINED AS ORE ] │ [ ACTUAL ORE ]
  Width: 8.4 m           │
  Mass: 40,824 t         │
  Grade: 0.08% Cu        │
  (DILUTION TO MILL)     │
                         │
```

#### 1. Quantify Displaced Tonnage in the Contact Zone
$$\text{Displaced Width} = 8.4\text{ m}$$
$$\text{Contact Zone Volume} = 120\text{ m} \times 8.4\text{ m} \times 15.0\text{ m} = 15,120\text{ m}^3$$
$$\text{Displaced Mass} = 15,120\text{ m}^3 \times 2.70\text{ t/m}^3 = 40,824\text{ tonnes}$$

#### 2. Case 1: Uncorrected Excavation (Ignoring Blast Movement)
If the shovel excavates to the original surveyed pre-blast boundary:
- $40,824\text{ tonnes}$ of waste rock ($0.08\% \text{ Cu}$) is excavated on the ore side and sent to the primary crusher as **Contact Dilution**.
- $40,824\text{ tonnes}$ of high-grade ore ($0.85\% \text{ Cu}$) has moved into the waste block and is excavated and hauled to the waste dump as **Ore Loss**.

#### 3. Financial Penalty of Uncorrected Digging
- **Ore Loss Metal Value:**
  Lost copper metal:
  $$\text{Cu Lost} = 40,824\text{ t} \times (0.85\% - 0.08\%) = 40,824 \times 0.0077 = 314.34\text{ tonnes of Cu}$$
  At a copper price of $\$9,500/\text{tonne}$ and concentrator recovery of $88\%$:
  $$\text{Lost Revenue} = 314.34\text{ t} \times \$9,500/\text{t} \times 0.88 = \$2,627,907$$
- **Dilution Milling Penalty:**
  Milling $40,824\text{ tonnes}$ of barren waste costs $\$14.50/\text{tonne}$ in crushing, grinding, reagent, and tailings disposal:
  $$\text{Wasted Milling OPEX} = 40,824\text{ t} \times \$14.50/\text{t} = \$591,948$$
- **Total Single-Blast Economic Penalty:**
  $$\text{Total Loss} = \$2,627,907 + \$591,948 = \$3,219,855$$
By deploying $\$3,500$ worth of BMM sensors and software vector translation, this $\$3.22\text{M}$ loss is completely recovered.

---

### Sub-Problem D: Haul Cycle Motion Profile and Fleet Sizing

The mine hauls ore from Shovel 1 (Bench 1240) to the Primary Crusher (Elevation 1360).
- Loaded haul profile:
  - Segment 1: In-pit flat bench, $300\text{ m}$, $0\%$ grade, $RR = 2.5\%$.
  - Segment 2: In-pit haul ramp uphill, $1,200\text{ m}$, $+10.0\%$ grade, $RR = 2.0\%$ ($TR = 12.0\%$).
  - Segment 3: Surface flat run to crusher, $500\text{ m}$, $0\%$ grade, $RR = 2.0\%$.
- Haul Truck: 240-tonne class (CAT 793F equivalent).
  - Empty Vehicle Weight ($\text{EVW}$) = $165.0\text{ t}$.
  - Nominal Payload = $240.0\text{ t}$.
  - Gross Vehicle Weight ($\text{GVW}$) = $405.0\text{ t}$.
  - Net Flywheel Engine Power = $1,976\text{ kW}$ ($2,650\text{ hp}$). Drive efficiency $\eta = 0.85$.
- Shovel: Electric rope shovel with $33\text{ m}^3$ dipper. Pass time = $35\text{ s}$. 4 passes to load ($140\text{ s} = 2.33\text{ min}$). Truck spotting time $t_{\text{spot}} = 0.70\text{ min}$.
- Crusher dump time $t_{\text{dump}} = 1.20\text{ min}$ (including spot and raise body).

#### 1. Uphill Loaded Travel Speed (Segment 2)
On Segment 2, total resistance is $TR = 10.0\% + 2.0\% = 12.0\%$.
- Resisting force ($F_{\text{resist}}$):
  $$F_{\text{resist}} = \frac{TR}{100} \cdot \text{GVW} \cdot g = 0.12 \times 405,000\text{ kg} \times 9.81\text{ m/s}^2 = 476,766\text{ N} = 476.8\text{ kN}$$
- Maximum terminal velocity under engine power:
  $$F_{\text{rimpull}} = \frac{3600 \cdot P_{\text{net}} \cdot \eta}{v} \implies v_{\text{term}} = \frac{3600 \times 1,976 \times 0.85}{476.8} = \frac{6,046,560}{476,766} \times 3.6 = 12.68\text{ km/h}$$
- Travel time on Segment 2 ($1.2\text{ km}$ at $12.68\text{ km/h}$):
  $$t_{\text{seg2}} = \frac{1.20\text{ km}}{12.68\text{ km/h}} \times 60 = 5.68\text{ min}$$

#### 2. Total Cycle Time Synthesis
- Segment 1 ($0.3\text{ km}$, accelerating, avg speed $25\text{ km/h}$): $0.72\text{ min}$.
- Segment 2 (Ramp uphill, terminal speed $12.68\text{ km/h}$): $5.68\text{ min}$.
- Segment 3 ($0.5\text{ km}$, avg speed $40\text{ km/h}$): $0.75\text{ min}$.
- Total Loaded Haul Time $t_{\text{haul\_loaded}} = 0.72 + 5.68 + 0.75 = 7.15\text{ min}$.
- Empty Return Travel Time (truck runs downhill at retarder limit $24\text{ km/h}$ on ramp, $50\text{ km/h}$ on flats): $t_{\text{return\_empty}} = 4.20\text{ min}$.
- Total Deterministic Cycle Time:
  $$T_{\text{cycle}} = t_{\text{spot}} + t_{\text{load}} + t_{\text{haul\_loaded}} + t_{\text{dump}} + t_{\text{return\_empty}}$$
  $$T_{\text{cycle}} = 0.70 + 2.33 + 7.15 + 1.20 + 4.20 = 15.58\text{ min}$$

#### 3. Match Factor and Fleet Allocation
To fully utilize Shovel 1 (loading time $t_{\text{load\_total}} = t_{\text{spot}} + t_{\text{load}} = 0.70 + 2.33 = 3.03\text{ min}$ per truck):
$$MF = 1.0 \implies N_{\text{trucks}} = \frac{T_{\text{cycle}}}{t_{\text{load\_total}}} = \frac{15.58\text{ min}}{3.03\text{ min}} = 5.14\text{ trucks}$$
Allocating $5$ trucks yields $MF = 5 / 5.14 = 0.973$ (shovel operates at $97.3\%$ capacity, zero truck queuing). Allocating $6$ trucks yields $MF = 6 / 5.14 = 1.167$ (shovel operates at $100\%$, but each truck spends an average of $2.6\text{ min}$ per cycle queuing).

---

## 7. Code, query, or calculation

The following self-contained Python script models and solves the **Stage 1 Open-Pit Dynamic Dispatch Linear Program**. It optimizes truck flow rates across multiple shovels, primary crushers, and waste dumps, respecting shovel digging rates, crusher blending grade envelopes, and total fleet truck hours.

```python
"""Stage 1 Open-Pit Dynamic Dispatch Linear Program.

Optimizes route flow rates (t/h) to maximize production subject to shovel,
crusher, blending, and truck fleet conservation constraints.
"""

from scipy.optimize import linprog

# 1. Mining Infrastructure Parameters
shovels = [
    {"id": "S1_HighGrade", "cap": 3000.0, "grade": 0.85, "ore": True},
    {"id": "S2_LowGrade", "cap": 3500.0, "grade": 0.45, "ore": True},
    {"id": "S3_Waste", "cap": 4500.0, "grade": 0.05, "ore": False},
]
# Routes: (shovel_idx, dest_idx, cycle_time_minutes)
# Dest 0: Primary Crusher, Dest 1: Waste Dump
routes = [
    (0, 0, 22.5),  # S1 -> Crusher
    (1, 0, 18.0),  # S2 -> Crusher
    (2, 1, 15.0),  # S3 -> Waste Dump
]
payload, n_trucks, hr_min = 240.0, 18.0, 60.0  # 18 x 240-tonne haul trucks

# 2. Objective Function: Maximize total mined tonnes/h -> Minimize (-sum(x))
c = [-1.0, -1.0, -1.0]

# 3. Constraint Matrix (A_ub * x <= b_ub)
A_ub = [
    [1.0, 0.0, 0.0],  # S1 capacity <= 3000 t/h
    [0.0, 1.0, 0.0],  # S2 capacity <= 3500 t/h
    [0.0, 0.0, 1.0],  # S3 capacity <= 4500 t/h
    [1.0, 1.0, 0.0],  # Crusher throughput <= 4000 t/h
    [-0.20, 0.20, 0.0],  # Min Cu Grade 0.65%: (0.85-0.65)x0 + (0.45-0.65)x1 >= 0
    [0.13, -0.27, 0.0],  # Max Cu Grade 0.72%: (0.85-0.72)x0 + (0.45-0.72)x1 <= 0
    [1.20, 1.20, -1.0],  # Stripping ratio balance: Waste (x2) >= 1.2 * Ore
    [
        routes[0][2] / (payload * hr_min),
        routes[1][2] / (payload * hr_min),
        routes[2][2] / (payload * hr_min),
    ],  # Fleet truck limit
]
b_ub = [3000.0, 3500.0, 4500.0, 4000.0, 0.0, 0.0, 0.0, n_trucks]
bounds = [(0, None) for _ in routes]

# 4. Solve the LP using HiGHS Interior Point / Dual Simplex Solver
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")
assert res.success, f"Linear program failed: {res.message}"

# 5. Output Results and Verification
print(f"Optimal Total Mining Rate: {-res.fun:.1f} t/h")
for i, r in enumerate(routes):
    t_rate = res.x[i]
    trucks_req = t_rate * r[2] / (payload * hr_min)
    print(
        f"Route {shovels[r[0]]['id']} -> Dest {r[1]}: {t_rate:7.1f} t/h "
        f"({trucks_req:4.1f} trucks)"
    )
crush_ore = res.x[0] + res.x[1]
blend_cu = (
    (res.x[0] * 0.85 + res.x[1] * 0.45) / crush_ore if crush_ore > 0 else 0
)
tot_trucks = sum(
    res.x[i] * routes[i][2] / (payload * hr_min) for i in range(len(routes))
)
print(
    f"Crusher Blend Feed : {crush_ore:.1f} t/h @ {blend_cu:.3f}% Cu (Spec: 0.65-0.72%)"
)
print(f"Fleet Trucks Allocated : {tot_trucks:.2f} / {n_trucks:.0f} trucks")
```

### Execution Output and Validation

```text
Optimal Total Mining Rate: 8250.0 t/h
Route S1_HighGrade -> Dest 0:  1875.0 t/h ( 2.9 trucks)
Route S2_LowGrade -> Dest 0:  1875.0 t/h ( 2.3 trucks)
Route S3_Waste -> Dest 1:  4500.0 t/h ( 4.7 trucks)
Crusher Blend Feed : 3750.0 t/h @ 0.650% Cu (Spec: 0.65-0.72%)
Fleet Trucks Allocated : 9.96 / 18 trucks
```

The LP proves that under the blending envelope ($\ge 0.65\% \text{ Cu}$), the crusher achieves $3,750\text{ t/h}$ by drawing equal tonnages ($1,875\text{ t/h}$) from High-Grade Shovel 1 ($0.85\%$) and Low-Grade Shovel 2 ($0.45\%$). The stripping constraint forces Waste Shovel 3 to run at its maximum capacity ($4,500\text{ t/h}$), satisfying the $1.2:1$ stripping ratio. The operation requires only $9.96$ active trucks, leaving $8.04$ trucks available for scheduled maintenance or longer-distance waste hauls.

---

## 8. Data, standards, and regulatory context

Drilling, blasting, loading, and hauling operations are heavily constrained by international engineering standards, occupational safety regulations, and environmental statutes.

### 1. Explosives Safety and Regulatory Compliance
- **MSHA (U.S. Mine Safety and Health Administration):** Governed under 30 CFR Part 56/57 (Subpart E - Explosives). Mandates magazine licensing, minimum separation distances (American Table of Distances), strict clearance protocols before blast firing, and post-blast misfire inspection procedures.
- **Vibration and Overpressure Limits (USBM RI 8507 / OSMRE):** Blasting generates ground vibrations measured as Peak Particle Velocity ($\text{PPV}$ in $\text{mm/s}$ or $\text{in/s}$) and airblast overpressure ($\text{dBL}$). Under USBM standards, maximum allowable $\text{PPV}$ near residential structures is frequency-dependent:
  $$\text{PPV}_{\max} = \begin{cases} 0.50\text{ in/s } (12.7\text{ mm/s}) & \text{at low frequencies } (f < 10\text{ Hz}) \\ 2.00\text{ in/s } (50.8\text{ mm/s}) & \text{at high frequencies } (f > 40\text{ Hz}) \end{cases}$$
  Exceeding these limits risks structural damage to nearby communities and revokes the mine's operating permit.
- **ISEE (International Society of Explosives Engineers):** Provides standard field guidance for blast design, electrical and electronic detonator hookups, and flyrock risk exclusion zone calculations.

### 2. Haul Road Design and Vehicle Safety Standards
- **AASHTO / MSHA 30 CFR § 56.9300 (Berms and Guardrails):** Berms on the outer edge of elevated haul roads must be at least **axle-height of the largest vehicle** operating on the road ($H_{\text{berm}} \ge r_{\text{tire}} \approx 1.8\text{--}2.0\text{ m}$ for ultra-class trucks).
- **Haul Road Geometry:** Recommended running width on straight segments must equal at least $3.5$ times the width of the widest truck for two-way traffic ($W_{\text{road}} \ge 3.5 \times W_{\text{truck}} \approx 30\text{--}35\text{ m}$). On curves, super-elevation (camber) and lane widening must be engineered to prevent truck roll-overs under lateral centrifugal acceleration.
- **Ramp Grades:** Main haul ramp gradients are strictly standardized between $8.0\%$ and $10.0\%$. Exceeding $10\%$ exponentially increases wheel slip, accelerates drive-train thermal failure, and risks runaway vehicle incidents. Runaway escape ramps with gravity roll-up or loose gravel arrester beds are legally mandatory on long descents.

### 3. Global Mining Guidelines (GMG)
- **GMG Guideline for Autonomous Haulage Operations (AHS):** Standardizes safety architectures, virtual geofencing, dynamic speed zones, vehicle-to-vehicle (V2V) communication, and human-machine interaction protocols for autonomous fleets.
- **GMG Guideline for Operational Mine-to-Mill Integration:** Standardizes telemetry data exchange between dispatch systems, blast databases, primary crushers, and mill SCADA systems to enable closed-loop comminution control.

### 4. Mineral Resource Reconciliation Standards (JORC / NI 43-101)
Reconciliation across the mining interface is codified via standard reconciliation factors:
- **$F_1$ Reconciliation:** Compares the long-term resource model to the short-term grade control blast-hole model ($F_1 = \text{Tonnage / Grade / Metal}_{\text{GradeControl}} / \text{Reserves}$).
- **$F_2$ Reconciliation:** Compares the grade control model to the material delivered to the mill feed pocket ($F_2 = \text{Received at Mill} / \text{GradeControl}$). Discrepancies in $F_2$ directly expose unmeasured blast movement dilution and ore loss.
- **$F_3$ Reconciliation:** Compares total mineral produced (metal poured + tailings) back to the original resource model ($F_3 = F_1 \times F_2$).

---

## 9. Industry practice and case evidence

Modern mining enterprises operate some of the world's most sophisticated telemetry-driven physical logistics networks.

```text
=============================================================================================================
                      BENCHMARK INDUSTRIAL EVIDENCE: MINE-TO-MILL & AHS
=============================================================================================================

 CASE 1: TECK HIGHLAND VALLEY COPPER (Canada)
 - Problem: Hard chalcopyrite-bornite ore capped SAG mill throughput at 4,000 t/h.
 - Intervention: Increased mass powder factor from 0.22 kg/t to 0.38 kg/t; reduced blast hole spacing.
 - Result: Crusher feed P80 dropped 42%; SAG mill throughput expanded +14.5% (+580 t/h);
   Net site cash flow increased by $22M/year after accounting for higher explosive expenditures.

 CASE 2: BHP MINERA ESCONDIDA (Chile - World's Largest Copper Mine)
 - Problem: Complex porphyry contact zones caused severe dilution; copper recovery dropped in mill.
 - Intervention: Deployed Blast Movement Technologies (BMM) directional RF transmitters on all blasts.
 - Result: Tracked 6 to 18 meters of lateral muckpile heave; updated dig polygons prior to shovel entry;
   Eliminated 6.8% contact dilution, recovering an audited $34M in annual copper value.

 CASE 3: RIO TINTO IRON ORE - "MINE OF THE FUTURE" (Pilbara, Western Australia)
 - Problem: Operator fatigue, shift-change delays, and truck bunching on 40 km rail-head haul cycles.
 - Intervention: Deployed Komatsu FrontRunner Autonomous Haulage System (AHS) across 140+ trucks.
 - Result: Haul cycle time variance dropped by 85%; tire life expanded 20% via TKPH thermal control;
   Zero haulage lost-time injuries (LTI) over 10+ years of 24/7 autonomous operations.
```

### 1. Teck Resources — Highland Valley Copper Mine-to-Mill Campaign
At Highland Valley Copper (HVC) in British Columbia, Canada, operations were constrained by the capacity of two large $10.4\text{ m}$ ($34\text{ ft}$) diameter SAG mills. Working with the Julius Kruttschnitt Mineral Research Centre (JKMRC), HVC initiated an integrated Mine-to-Mill project. 
- Prior to the study, drill-and-blast operated as an independent cost center, maintaining a minimal powder factor of $0.22\text{ kg/t}$ to preserve explosive budgets.
- By switching from standard ANFO to high-density water-resistant bulk emulsion, increasing blasthole sub-drilling, and raising the powder factor to $0.38\text{ kg/t}$, HVC increased blast costs by $\$0.18/\text{t}$.
- The blast-generated fines fraction ($-12.7\text{ mm}$) entering the SAG mills doubled from $11\%$ to $23\%$. Because these fines bypass the impact grinding mechanism and flush directly through the SAG mill pulp discharge grates, mill specific energy dropped from $11.2\text{ kWh/t}$ to $9.4\text{ kWh/t}$.
- SAG mill throughput expanded by $14.5\%$ without the installation of any additional grinding assets, producing over $\$20\text{M}$ in net annual value.

### 2. BHP — Minera Escondida Blast Movement Dilution Control
At Escondida in the Atacama Desert of Chile, copper mineralization occurs in steeply dipping, irregular hydrothermal breccia bodies adjacent to barren porphyry waste. 
- High-energy blasts heaved the muckpile laterally by up to $18\text{ m}$. Initial mining practice excavated blocks according to pre-blast collar coordinate surveys.
- Grade reconciliation audits ($F_2$) revealed that mill head grades were consistently $8\%$ to $12\%$ lower than grade-control predictions, caused entirely by contact dilution.
- Escondida deployed radio-frequency Blast Movement Monitors (BMMs) embedded in drill holes. Sensor detection revealed that displacement was highly non-linear: surface heave moved $14\text{ m}$, while mid-bench moved $9\text{ m}$.
- By integrating 3D vector deformation algorithms into their Hexagon MinePlan grade-control software, dig boundaries were dynamically translated to match the actual post-blast muckpile. Dilution was reduced by $7.2\%$, preserving an estimated $\$34\text{M}$ in copper metal per year.

### 3. Rio Tinto & Komatsu — Autonomous Haulage Systems (AHS) in the Pilbara
In Western Australia's Pilbara iron ore operations, Rio Tinto operates the world's largest fleet of autonomous haul trucks (over 140 Komatsu 930E and 830E trucks running FrontRunner software).
- In manual operations, truck cycle times suffer from stochastic driver variance: variable acceleration rates, excessive braking, inconsistent spotting times at shovels ($45\text{--}90\text{ s}$), and production stoppages during shift changeovers ($90\text{ minutes/day}$).
- Autonomous trucks navigate using high-precision RTK-GPS (real-time kinematic satellite navigation), obstacle detection radar, and laser LiDAR sensors.
- Spotting times at shovels collapsed from an average of $62\text{ s}$ to a deterministic $32\text{ s}$. 
- The autonomous supervisory system optimizes speed curves to ensure truck tire TKPH limits are never breached, extending tire life by $20\%$ to $25\%$. Truck utilization increased by more than $1,000$ operating hours per truck per year, reducing unit haulage OPEX by $15\%$.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Underlying Physical / Operational Mechanism | Downstream Supply Chain Consequence | Mitigation / Preventive Control |
|---|---|---|---|---|
| **Excessive Boulder Generation (Oversize > 1 m)** | High secondary breakage frequency; shovel bucket teeth rattling; excavator stall alarms. | Insufficient powder factor, stemming blowout, or joint spacing larger than burden/spacing grid. | Primary gyratory crusher bridges; shovel cycle time increases by 40%; throughput collapses. | Increase powder factor; tighten burden-spacing pattern; switch to electronic detonators with tuned inter-hole delays. |
| **Bench Floor Hard Toe Formation** | Shovel struggles to reach floor grade; teeth wear prematurely; broken shovel hoist ropes. | Insufficient sub-drilling ($J < 0.2B$), water-degraded explosive, or drill hole collar deviation. | Shovel productivity drops; uneven floor damages haul truck suspension, frames, and tires. | Increase sub-drill depth; survey hole collars with GPS; use water-resistant emulsion in wet blastholes. |
| **Stemming Blowout & Flyrock Ejection** | Loud sharp airblast report; vertical dust plume ejection; rocks projected outside exclusion zone. | Undersized stemming length ($T < 0.7B$), fine drill cuttings used as stemming, or weak cap rock. | Catastrophic safety hazard; personnel injury; equipment destruction; regulatory mine shutdown. | Use angular crushed aggregate sized to $D/10$; enforce minimum stemming length; install blast mats if near infrastructure. |
| **Uncorrected Blast Heave Dilution** | Mill head grade lower than mine plan ($F_2 < 0.90$); high reagent consumption in flotation. | In-situ grade boundaries excavated without correcting for 3D explosive heave displacement. | High-grade ore dumped into waste dumps; barren waste milled; millions in lost metal revenue. | Deploy subsurface Blast Movement Monitoring (BMM) sensors; translate dig polygons prior to excavation. |
| **Shovel-Truck Bunching (Queue Saturation)** | Long lines of haul trucks idling at shovel benches; shovel operators waiting between trucks. | Stochastic variance in haul cycle times combined with static or unoptimized truck dispatching. | Massive fuel burn during idling ($80\text{--}120\text{ L/h}$ per truck); truck fleet productivity drops by 15-25%. | Deploy two-stage dynamic LP dispatching; dynamically reroute empty trucks to starved shovels. |
| **Primary Crusher Cavity Bridging** | Crusher level sensor alarms; rock breaker deployed continuously; haul trucks queuing at dump pocket. | Oversized boulders ($>1.2\text{ m}$) dumped directly into the gyratory crusher feed throat. | Crusher shut down for clearing; upstream haul fleet paralyzed by dump pocket blockage. | Install dump pocket sizing grizzly; reject oversize at shovel muckpile; optimize blast top-size ($P_{100}$). |
| **Haul Truck Tire Heat Delamination** | Hot rubber smell; tire surface blistering; internal tire pressure telemetry alerts. | Truck operating at average speeds and payloads that exceed the manufacturer's TKPH rating. | Catastrophic tire blowout; $\$50,000$ tire replacement cost; structural rim and vehicle rollover hazard. | Enforce real-time TKPH monitoring in dispatch; speed-govern loaded trucks; extend haul road cooling runaways. |
| **Haul Road Degradation (High Rolling Resistance)** | Truck rimpull maxed out in low gear; deep tire ruts visible; engine overheating on ramp. | Poor sub-base drainage, wet clay contamination, or lack of motor grader / water cart maintenance. | Rolling resistance rises from 2% to 6%; truck fuel consumption doubles; cycle times extend 30%. | Stabilize haul road base with crushed rock; maintain crown and ditches; schedule regular grader passes. |
| **Misfires and Cut-Off Explosive Columns** | Unbroken hard ground discovered during excavation; yellow $\text{NO}_x$ fumes post-blast; live boosters in muckpile. | Ground shear cutting non-electric shock tube leads, or dynamic desensitization of emulsion by adjacent blast holes. | Lethal hazard for shovel operator encountering unexploded ordnance; work halted for bomb squad extraction. | Convert to electronic digital detonators; review inter-hole delay timing to avoid dynamic shock pre-compression. |
| **Crusher Feed Grade Excursion** | Flotation cell color changes; tailings assay spikes; concentrate grade falls below smelter spec. | Shovel excavating outside grade polygon, or dispatch LP sending unblended high-impurity ore to crusher. | Smelter penalty charges applied; mineral recovery drops; concentrator circuit instability. | Implement automated shovel bucket GPS tracking; enforce strict linear programming grade limits in dispatch. |
| **Downhill Truck Brake Overheating** | Brake retarder oil temperature alarm; smoke from wheel groups on long descents. | Truck descending loaded ramp at speeds exceeding continuous dynamic retarding capacity. | Total hydraulic brake fade; catastrophic vehicle runaway down pit ramp. | Strictly enforce downhill speed limit boards; configure retarder speed control (RSC) to lock maximum descent velocity. |
| **Drill Hole Collar Deviation** | Irregular burden along crest; tight or loose toes; backbreak behind the last row of blastholes. | Inaccurate manual drill positioning; flexible drill steel bending in dipping fractured rock. | Non-uniform explosive distribution; severe oversize boulders; unsafe unstable bench crests. | Equip drill rigs with automated GPS guidance and mast angle inclinometer positioning systems. |

---

## 11. Assumptions and limits

| Assumption / Model | Why It Is Needed | Operational Condition That Breaks It | Detection & Validation Method | Correction / Fallback Strategy |
|---|---|---|---|---|
| **Homogeneous, Isotropic Rock Mass (Kuz-Ram)** | Simplifies fragmentation calculation to algebraic equations based on intact rock properties. | Intact rock has extreme anisotropy, alternating soft shale and hard quartzite beds, or major fault zones. | Geotechnical core logging and acoustic televiewer logs show structural discontinuity clusters. | Discretize blast patterns into geotechnical domains; run numerical hydrocode models (e.g., Autodyn). |
| **Rigid-Body 3D Blast Heave Translation** | Assumes rock within a block displaces uniformly along a single interpolated BMM vector. | Internal shear planes develop; surface rock flies forward while bottom bench rock is trapped. | Multiple BMM transmitters placed at different depths in the same hole show divergent vectors. | Implement 3D non-linear polyhedral mesh deformation rather than planar 2D polygon translation. |
| **Deterministic Travel Speeds on Haul Roads** | Allows cycle time synthesis from manufacturer rimpull-speed curves. | Road congestion, grader interference, weather/rain, and variable driver behavior introduce stochastic variance. | Dispatch telemetry logs show high coefficient of variation ($CV > 0.25$) on haul route segment travel times. | Switch from deterministic rimpull synthesis to empirical historical transit time distributions ($M/G/1$). |
| **Linear Blending of Ore Grades at Crusher** | Enables linear programming formulation for grade control blending constraints. | Autogenous/SAG grinding hardness, clay mineral flotation interference, and recovery are non-linear. | Metallurgical plant reconciliations show non-linear recovery drops when blending talc or clay ores. | Formulate non-linear blending constraints or deploy Mixed-Integer Non-Linear Programming (MINLP) dispatchers. |
| **Independence of Shovel Dig Rate from Muckpile** | Assumes shovel loading rate ($t/\text{h}$) is constant regardless of blast fragmentation. | Coarse muckpiles with high oversize cause bucket stalling, teeth jamming, and double-handling. | Shovel telemetry logs show diggability index (hydraulic pressure $\times$ cycle time) spikes in coarse muck. | Couple shovel loading time directly to Kuz-Ram $x_{50}$ in the dispatch allocation model. |
| **Steady-State Comminution Circuit Operation** | Assumes mill throughput scales smoothly with feed size ($F_{80}$) without circuit disruption. | Extremely fine feed causes mill pulp pooling, slurry discharge grate blinding, and SAG motor load loss. | Mill SCADA telemetry shows surging bearing pressure and rapid power oscillations. | Blend coarse competent media with fine blasted ore to maintain autogenous grinding media charge. |
| **Infinite Dump Pocket Storage Buffer** | Simplifies Stage 1 dispatch LP by assuming trucks can always dump at the crusher pocket. | Crusher undergoes emergency mechanical tripping or bridges, halting dumping instantly. | Dispatch system receives digital crusher shut-down interlock signal. | Re-route ore trucks automatically to coarse ore surge stockpiles or alternative dump stations. |
| **Constant Haul Road Rolling Resistance** | Assumes $RR = 2.0\%$ uniformly across the entire pit network throughout the shift. | Heavy rain, water truck over-watering, or lack of grading creates deep ruts, doubling $RR$ to $4\text{--}6\%$. | On-board truck telematics show drive-motor current spikes and unexpected speed drops on flat segments. | Update segment resistance parameters in dispatch engine dynamically based on truck CAN-bus telematics. |
| **Perfect Truck Driver Compliance** | Assumes human drivers immediately obey digital dispatch assignment directives. | Drivers choose their own routes, stop for unauthorized breaks, or ignore dynamic re-assignment. | Dispatch system triggers "route deviation" alarms when truck GPS coordinates deviate from vector. | Deploy audio dispatch alarms or transition the haul fleet to Autonomous Haulage Systems (AHS). |
| **Negligible Explosive Energy Desensitization** | Assumes bulk emulsion detonates at full VOD and energy throughout the entire column. | Shock waves from adjacent early-firing holes compress and desensitize emulsion in later holes. | High-speed blast video shows sluggish heave; post-blast muckpile exhibits unbroken toe blocks. | Optimize electronic detonator initiation sequences to ensure inter-hole delay exceeds shock transit time. |

---

## 12. Exercises

### Exercise 1: Blastability Index and Kuz-Ram Parameter Synthesis
- **Deliverable:** Detailed numerical derivation of Lilly's Blastability Index ($A$), powder factor ($q_v$), mean fragment size ($x_{50}$), uniformity index ($n$), and $P_{80}$.
- **Acceptance Criteria:**
  - Given: Bench height $H = 12.0\text{ m}$, hole diameter $D = 250\text{ mm}$, burden $B = 6.0\text{ m}$, spacing $S = 7.0\text{ m}$, sub-drill $J = 1.2\text{ m}$, stemming $T = 4.5\text{ m}$.
  - Rock mass: $\text{RMD} = 40$, $\text{JPS} = 30$, $\text{JPO} = 30$, $\text{SMR} = 4.0$, density $\rho_r = 2.80\text{ t/m}^3$, $\text{UCS} = 140\text{ MPa}$.
  - Explosive: Bulk emulsion, density $\rho_e = 1.20\text{ g/cm}^3$, $\text{RWS} = 110$. Drilling error $W = 0.15\text{ m}$.
  - Compute: (a) Blastability Index $A$, (b) mass explosive charge per hole $Q$, (c) volumetric powder factor $q_v$, (d) $x_{50}$ in cm, (e) uniformity index $n$, and (f) $P_{80}$ in mm.
- **Estimated Time:** 35 minutes.

### Exercise 2: Comminution Power and Throughput Trade-Off
- **Deliverable:** Engineering calculation of SAG mill power draw, throughput expansion, and net economic balance under a Mine-to-Mill blast redesign.
- **Acceptance Criteria:**
  - Baseline: $F_{80} = 120,000\ \mu\text{m}$, fines fraction $f_{-12.7} = 0.02$, SAG specific energy $E_{\text{SAG}} = 29.0\text{ kWh/t}$. Installed SAG power = $28,000\text{ kW}$.
  - Optimized Blast: Powder factor increased by $0.15\text{ kg/t}$ (costing $+\$0.35/\text{t}$). Resulting $F_{80} = 75,000\ \mu\text{m}$, fines fraction $f_{-12.7} = 0.12$.
  - Specific energy formula: $E_{\text{SAG}} = 27.0 \cdot (F_{80}/100,000)^{0.35} \cdot (1 - 0.70 \cdot f_{-12.7})\ \text{kWh/t}$.
  - Electricity price: $\$0.10/\text{kWh}$. Annual mine ore throughput: $12,000,000\text{ t/year}$.
  - Compute: (a) Baseline and optimized SAG throughput in t/h, (b) electrical cost savings per tonne, and (c) net annual cost savings after subtracting explosive costs.
- **Estimated Time:** 30 minutes.

### Exercise 3: Blast Heave Dilution and Ore Loss Calculation
- **Deliverable:** Mathematical computation of dilution mass, ore loss mass, resulting head grade, and net revenue impact for an uncorrected versus BMM-corrected blast.
- **Acceptance Criteria:**
  - Ore block ($100\text{ m}$ strike length, $15\text{ m}$ bench height, $30\text{ m}$ width) grades $1.20\% \text{ Cu}$.
  - Adjacent waste block grades $0.05\% \text{ Cu}$. Density $\rho = 2.65\text{ t/m}^3$.
  - Detonation displaces the boundary horizontally by $9.0\text{ m}$ into the waste block.
  - Compute: (a) In-situ ore mass, (b) dilution mass and ore loss mass if mined to original uncorrected lines, (c) diluted head grade delivered to the mill, and (d) financial loss assuming copper at $\$9,000/\text{t}$, $85\%$ recovery, and milling cost of $\$12.00/\text{t}$.
- **Estimated Time:** 35 minutes.

### Exercise 4: Haul Truck Motion and Rimpull Sizing
- **Deliverable:** Kinematic calculation of uphill loaded haul velocity, downhill retarding velocity, cycle time, and hourly truck production.
- **Acceptance Criteria:**
  - Haul route: $2.0\text{ km}$ uphill on a $+9.0\%$ grade with $RR = 2.0\%$ ($TR = 11.0\%$).
  - Truck: $\text{EVW} = 150.0\text{ t}$, payload $= 220.0\text{ t}$, $\text{GVW} = 370.0\text{ t}$. Flywheel power $P = 1,800\text{ kW}$, drive efficiency $\eta = 0.85$.
  - Loading time $= 2.5\text{ min}$, spotting time $= 0.8\text{ min}$, dump time $= 1.2\text{ min}$. Empty return velocity $= 45.0\text{ km/h}$.
  - Compute: (a) Terminal velocity uphill (km/h), (b) total cycle time (min), and (c) hourly hauling capacity per truck ($t/\text{h}$).
- **Estimated Time:** 30 minutes.

### Exercise 5: Queuing Match Factor and MVA Fleet Sizing
- **Deliverable:** Closed cyclic queuing analysis of a 3-shovel, multi-truck open-pit system using Mean Value Analysis (MVA).
- **Acceptance Criteria:**
  - Shovel loading time is exponentially distributed with mean $3.0\text{ min}$.
  - Average round-trip transit time on haul/return roads is $15.0\text{ min}$. Dump time at crusher is $1.5\text{ min}$.
  - System operates with $N = 12$ haul trucks.
  - Formulate the MVA recursive equations and calculate shovel utilization and average truck waiting time in queue.
- **Estimated Time:** 45 minutes.

### Exercise 6: Dynamic Dispatch LP Formulation and Extension
- **Deliverable:** Extended mathematical formulation and Python script implementing maximum throughput dispatch with arsenic contaminant blending constraints.
- **Acceptance Criteria:**
  - Modify the Section 7 LP to include an arsenic ($\text{ppm As}$) penalty constraint: crusher feed arsenic must not exceed $250\text{ ppm}$.
  - Given: Shovel 1 ($0.90\% \text{ Cu}$, $450\text{ ppm As}$), Shovel 2 ($0.50\% \text{ Cu}$, $120\text{ ppm As}$).
  - Solve for the maximum allowable ore rate from Shovel 1 that respects the arsenic cap while satisfying a minimum copper head grade of $0.65\%$.
- **Estimated Time:** 40 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1
1. **Blastability Index ($A$):**
   $$A = 0.06 \times (40 + 30 + 30 + 4.0) + 0.1 \times (2.80) + 0.05 \times (140) = 0.06 \times 104 + 0.28 + 7.00 = 6.24 + 0.28 + 7.00 = 13.52$$
2. **Hole Geometry and Explosive Mass ($Q$):**
   - Hole depth $L = 12.0 + 1.2 = 13.2\text{ m}$. Charge length $L_{\text{charge}} = 13.2 - 4.5 = 8.7\text{ m}$.
   - Area $= \frac{\pi}{4} \times (0.250)^2 = 0.049087\text{ m}^2$.
   - $Q = 8.7 \times 0.049087 \times (1.20 \times 1000) = 512.47\text{ kg/hole}$.
3. **Powder Factor ($q_v$):**
   - $V_b = 6.0 \times 7.0 \times 12.0 = 504.0\text{ m}^3$.
   - $q_v = \frac{512.47}{504.0} = 1.0168\text{ kg/m}^3$.
   - $q_w = \frac{1.0168}{2.80} = 0.3631\text{ kg/t}$.
4. **Mean Fragment Size ($x_{50}$):**
   - $q_v^{-0.8} = (1.0168)^{-0.8} = 0.9867$.
   - $Q^{1/6} = (512.47)^{0.16667} = 2.8256$.
   - $(115 / 110)^{0.63333} = (1.04545)^{0.63333} = 1.0286$.
   - $x_{50} = 13.52 \times 0.9867 \times 2.8256 \times 1.0286 = 38.77\text{ cm} = 387.7\text{ mm}$.
5. **Uniformity Index ($n$):**
   - $14 \times \frac{B}{D} = 14 \times \frac{6.0}{250} = 0.336 \implies 2.2 - 0.336 = 1.864$.
   - $\sqrt{\frac{1 + 7/6}{2}} = \sqrt{1.0833} = 1.0408$.
   - $1 - \frac{W}{B} = 1 - \frac{0.15}{6.0} = 0.975$.
   - $\left(1 + \frac{7.0-6.0}{12.0}\right)^{0.1} = (1.0833)^{0.1} = 1.008$.
   - $\frac{L_{\text{charge}}}{H} = \frac{8.7}{12.0} = 0.725$.
   - $n = 1.864 \times 1.0408 \times 0.975 \times 1.008 \times 0.725 = 1.383$.
6. **Passing Size $P_{80}$:**
   - $x_c = \frac{x_{50}}{(\ln 2)^{1/n}} = \frac{387.7}{(0.69315)^{1/1.383}} = \frac{387.7}{(0.69315)^{0.723}} = \frac{387.7}{0.7667} = 505.7\text{ mm}$.
   - $P_{80} = x_c \cdot (\ln 5)^{1/n} = 505.7 \cdot (1.6094)^{0.723} = 505.7 \cdot 1.4111 = 713.6\text{ mm} \approx 0.714\text{ m}$.

---

### Answer to Exercise 2
1. **SAG Specific Energy:**
   - Baseline: $E_{\text{SAG}} = 29.0\text{ kWh/t}$.
   - Optimized:
     $$E_{\text{SAG, opt}} = 27.0 \cdot \left(\frac{75,000}{100,000}\right)^{0.35} \cdot (1 - 0.70 \times 0.12) = 27.0 \cdot (0.75)^{0.35} \cdot (1 - 0.084)$$
     $$(0.75)^{0.35} = 0.9042 \implies E_{\text{SAG, opt}} = 27.0 \times 0.9042 \times 0.916 = 22.36\text{ kWh/t}$$
2. **Throughput Expansion:**
   - Baseline: $\frac{28,000\text{ kW}}{29.0\text{ kWh/t}} = 965.5\text{ t/h}$.
   - Optimized: $\frac{28,000\text{ kW}}{22.36\text{ kWh/t}} = 1,252.2\text{ t/h}$.
   - Throughput gain: $+286.7\text{ t/h}$ ($+29.7\%$).
3. **Economic Balance:**
   - Power saved: $29.0 - 22.36 = 6.64\text{ kWh/t}$.
   - Power cost saving: $6.64 \times \$0.10/\text{kWh} = \$0.664/\text{t}$.
   - Added blast cost: $\$0.35/\text{t}$.
   - Net processing saving: $\$0.664 - \$0.35 = \$0.314/\text{t}$.
   - Annual net direct savings: $12,000,000\text{ t} \times \$0.314/\text{t} = \$3,768,000/\text{year}$.

---

### Answer to Exercise 3
1. **In-situ Ore Mass:**
   $$V_{\text{ore}} = 100 \times 15 \times 30 = 45,000\text{ m}^3$$
   $$M_{\text{ore}} = 45,000 \times 2.65 = 119,250\text{ tonnes}$$
2. **Dilution and Loss Tonnages:**
   $$V_{\text{displaced}} = 100 \times 15 \times 9.0 = 13,500\text{ m}^3$$
   $$M_{\text{dil}} = M_{\text{loss}} = 13,500 \times 2.65 = 35,775\text{ tonnes}$$
3. **Diluted Mill Head Grade:**
   - Ore mined to mill: $(119,250 - 35,775) = 83,475\text{ t}$ at $1.20\%$, plus $35,775\text{ t}$ waste at $0.05\%$.
   - Total mass milled: $119,250\text{ tonnes}$.
   $$\text{Head Grade} = \frac{83,475 \times 1.20 + 35,775 \times 0.05}{119,250} = \frac{100,170 + 1,788.75}{119,250} = \frac{101,958.75}{119,250} = 0.855\% \text{ Cu}$$
   Grade dropped by $0.345\% \text{ Cu}$ (a $28.8\%$ drop in feed quality).
4. **Financial Impact:**
   - Lost copper metal in unmined ore sent to waste dump:
     $$\text{Cu Lost} = 35,775\text{ t} \times (1.20\% - 0.05\%) = 35,775 \times 0.0115 = 411.41\text{ tonnes of Cu}$$
   - Revenue lost: $411.41\text{ t} \times \$9,000/\text{t} \times 0.85 = \$3,147,287$.
   - Wasted milling OPEX: $35,775\text{ t} \times \$12.00/\text{t} = \$429,300$.
   - Total financial loss: $\$3,147,287 + \$429,300 = \$3,576,587$.

---

### Answer to Exercise 4
1. **Terminal Velocity Uphill:**
   - Resisting force on $TR = 11.0\%$:
     $$F_{\text{resist}} = 0.11 \times 370,000\text{ kg} \times 9.81\text{ m/s}^2 = 399,267\text{ N} = 399.3\text{ kN}$$
   - Flywheel propulsion:
     $$v_{\text{term}} = \frac{3600 \cdot P_{\text{net}} \cdot \eta}{F_{\text{resist}}} = \frac{3600 \times 1800 \times 0.85}{399.3} = \frac{5,508,000}{399,267} \times 3.6 = 13.80\text{ km/h}$$
2. **Total Cycle Time:**
   - Loaded haul time: $\frac{2.0\text{ km}}{13.80\text{ km/h}} \times 60 = 8.70\text{ min}$.
   - Empty return time: $\frac{2.0\text{ km}}{45.0\text{ km/h}} \times 60 = 2.67\text{ min}$.
   - Fixed times: $t_{\text{spot}} + t_{\text{load}} + t_{\text{dump}} = 0.80 + 2.50 + 1.20 = 4.50\text{ min}$.
   - $T_{\text{cycle}} = 8.70 + 2.67 + 4.50 = 15.87\text{ min}$.
3. **Hourly Truck Productivity:**
   $$\text{Trips per hour} = \frac{60\text{ min}}{15.87\text{ min}} = 3.78\text{ trips/h}$$
   $$\text{Production} = 3.78\text{ trips/h} \times 220\text{ t} = 831.6\text{ t/h per truck}$$

---

### Answer to Exercise 6
In the Stage 1 LP, let $x_1$ be tonnes/h from Shovel 1 and $x_2$ be tonnes/h from Shovel 2.
- Copper lower bound constraint ($\ge 0.65\%$):
  $$(0.90 - 0.65) x_1 + (0.50 - 0.65) x_2 \ge 0 \implies 0.25 x_1 - 0.15 x_2 \ge 0 \implies x_2 \le \frac{0.25}{0.15} x_1 = 1.667 x_1$$
- Arsenic upper bound constraint ($\le 250\text{ ppm}$):
  $$(450 - 250) x_1 + (120 - 250) x_2 \le 0 \implies 200 x_1 - 130 x_2 \le 0 \implies x_2 \ge \frac{200}{130} x_1 = 1.538 x_1$$
To maximize total ore throughput ($x_1 + x_2$) up to crusher limit $4,000\text{ t/h}$, set $x_1 + x_2 = 4,000$.
- Under the tightest arsenic boundary ($x_2 = 1.5385 x_1$):
  $$x_1 + 1.5385 x_1 = 4,000 \implies 2.5385 x_1 = 4,000 \implies x_1 = 1,575.7\text{ t/h}$$
  $$x_2 = 4,000 - 1,575.7 = 2,424.3\text{ t/h}$$
- Check Copper Grade:
  $$\frac{1,575.7 \times 0.90 + 2,424.3 \times 0.50}{4000} = \frac{1,418.13 + 1,212.15}{4000} = 0.6576\% \text{ Cu} \quad (\ge 0.65\% \implies \text{Feasible})$$
- Check Arsenic:
  $$\frac{1,575.7 \times 450 + 2,424.3 \times 120}{4000} = \frac{709,065 + 290,916}{4000} = 250.0\text{ ppm As} \quad (\le 250\text{ ppm} \implies \text{Active Bound})$$
The maximum allowable feed rate from Shovel 1 is **$1,575.7\text{ t/h}$**.

---

## 14. Interview questions

### Question 1: What is the fundamental physical basis of the Mine-to-Mill concept, and why does an isolated drill-and-blast cost KPI undermine site profitability?
- **What a strong answer contains:** Explains the comminution energy paradox: comminution consumes up to 70% of a mine's electrical power. Rock breakage via chemical explosives in blastholes is 10 to 30 times cheaper per gigajoule of energy than mechanical tumbling in SAG/ball mills. Minimizing drill-and-blast costs leads to wider patterns, coarser fragmentation ($P_{80}$), lower fine generation, and suppressed micro-crack density. This chokes the SAG mill, driving up specific electrical energy consumption ($E_{\text{SAG}}$ in $\text{kWh/t}$) and capping mill throughput. A global site optimization increases powder factor to reduce total mining-plus-milling unit costs ($\$/t$).
- **What a weak answer reveals:** Focuses only on shovel loading speeds or mentions "making rocks smaller" without connecting explosive energy costs to SAG mill power draw and downstream flotation kinetics.

### Question 2: Why does the classical Kuz-Ram fragmentation model fail to accurately predict the fine fraction, and how does the Swebrec function address this?
- **What a strong answer contains:** Notes that Kuz-Ram is anchored to a single Rosin-Rammler distribution, which assumes a uniform, unimodal breakage mechanism. In reality, rock blasting involves two distinct physical fracture regimes: high-pressure compressive shear pulverization around the borehole wall (generating micro-fines $-1\text{ mm}$) and tensile stress wave reflection and gas heave (breaking macro-blocks). Kuz-Ram severely underpredicts micro-fines. The Swebrec function uses a three-parameter formulation with an explicit upper block boundary ($x_{\max}$) and curvature exponent ($b$), fitting both fine and coarse screening data across three orders of magnitude.
- **What a weak answer reveals:** Confuses $P_{80}$ with mean fragment size; cannot explain the mechanics of borehole shear pulverization versus tensile spalling.

### Question 3: How does blast movement cause grade dilution, and what is the mathematical mechanism used to mitigate it?
- **What a strong answer contains:** Details that explosive detonation imparts kinetic energy that throws and swells the broken rock mass horizontally and vertically by 5 to 20 meters. If post-blast dig blocks are excavated using pre-blast collar coordinates, the shovel digs waste as ore (dilution) and dumps ore onto the waste pile (ore loss). Mitigating this requires placing directional RF transmitters (BMM) in blastholes, locating them post-blast with magnetometers, calculating 3D displacement vectors $\vec{d} = (\Delta X, \Delta Y, \Delta Z)$, and applying spatial vector deformation to translate and warp the original grade control polygons.
- **What a weak answer reveals:** Assumes rock stays where it was drilled, or claims that blast heave can be prevented by adding more stemming.

### Question 4: How does dynamic truck dispatching differ from static assignment, and why does closed-loop queuing theory dictate fleet performance?
- **What a strong answer contains:** Static assignment permanently pairs $N$ trucks to specific shovels. Because loading, hauling, and dumping times exhibit stochastic variance (Erlang or log-normal distributions), trucks inevitably bunch together, creating shovel starvation and long dump queues. Dynamic dispatch operates a two-stage architecture: Stage 1 solves an LP network flow over rolling horizons to enforce macro targets and crusher blending envelopes; Stage 2 dynamically re-routes empty trucks using heuristics (e.g., Beenhakker rule) based on real-time shovel queue states. Explains that open queuing models ($M/M/1$) fail because the truck population is finite, requiring closed Gordon-Newell network formulations.
- **What a weak answer reveals:** Believes dynamic dispatch simply assigns trucks to the shovel with the shortest physical line, ignoring grade blending constraints and downstream crusher starvation.

### Question 5: What is the Ton-Kilometer Per Hour (TKPH) rating for haul truck tires, and what happens if it is exceeded?
- **What a strong answer contains:** Defines $\text{TKPH} = Q_{\text{mean}} \times V_{\text{shift\_avg}}$, where $Q_{\text{mean}}$ is average tire load (tonnes) and $V_{\text{shift\_avg}}$ is average operating speed (km/h) across loaded and empty cycles. Explains that rubber is a poor thermal conductor; repeated cyclic deflection under heavy loads generates severe internal hysteresis heat. If operational TKPH exceeds the manufacturer's tire compound thermal rating, internal carcass temperatures exceed $120^\circ\text{C}$, triggering rubber reversion, belt separation, and catastrophic tire blowouts costing $\$50,000$ per tire and creating fatal safety hazards.
- **What a weak answer reveals:** Thinks TKPH is a measure of tire wear/tread depth rather than an internal thermal equilibrium threshold.

### Question 6: Under what conditions would an open-pit mine operate at a fleet Match Factor significantly less than 1.0?
- **What a strong answer contains:** Explains that $MF < 1.0$ means the fleet is truck-constrained (shovels experience idle wait time). This is economically optimal when: (1) capital costs of haul trucks and fuel OPEX are exceptionally high relative to shovel capital, (2) the primary crusher is already operating at 100% capacity and cannot accept more ore, (3) diesel fuel prices surge dramatically, making truck idling and queuing costs exorbitant, or (4) the mine is stripping distant low-priority waste where shovel capital utilization is secondary to cash preservation.
- **What a weak answer reveals:** Asserts that $MF$ must always be exactly $1.0$ at all times, failing to recognize capital allocation trade-offs.

### Question 7: How does haul road rolling resistance directly affect truck diesel consumption and cycle times?
- **What a strong answer contains:** Rolling resistance ($RR$) represents energy lost to tire carcass flexing and wheel penetration into the road sub-base. A $1.0\%$ increase in $RR$ is mechanically equivalent to an additional $+1.0\%$ uphill ramp grade ($10\text{ kg/t}$ of additional resisting force). If road maintenance lapses and $RR$ rises from $2.0\%$ to $5.0\%$ on a flat haul, resisting force on a 400-tonne truck surges by $118\text{ kN}$. This forces the transmission into a lower gear, cuts travel speed from $40\text{ km/h}$ to $18\text{ km/h}$, and doubles fuel consumption, easily burning an extra $150\text{ liters/hour}$ of diesel per truck.
- **What a weak answer reveals:** Believes rolling resistance only matters when driving downhill or claims it is solely determined by tire tread design.

### Question 8: How do reconciliation factors $F_1, F_2, F_3$ isolate performance failures across the extraction chain?
- **What a strong answer contains:** Defines $F_1$ as the ratio of grade-control blast-hole model tonnes/grade to the long-term resource model (tests geological interpretation and kriging). Defines $F_2$ as the ratio of mill feed head assays to the grade-control model (tests blast movement dilution, ore loss, and shovel digging compliance). Defines $F_3 = F_1 \times F_2$ as the total resource recovery factor. A low $F_2$ with an $F_1 \approx 1.0$ proves conclusively that the geology model was correct, but mining execution failed due to uncorrected blast heave dilution or shovel operator misallocation.
- **What a weak answer reveals:** Confuses metallurgical recovery in flotation cells with mining reconciliation factors.

---

## 15. Key terms

- **Blastability Index ($A$):** Geotechnical score (Lilly's formulation) quantifying rock mass resistance to explosive fragmentation based on structural joints, density, and compressive strength.
- **Burden ($B$):** The perpendicular distance from a blasthole to the nearest free face or between consecutive blast rows.
- **Spacing ($S$):** The distance between adjacent blastholes in the same row, typically scaled to $1.15\text{--}1.25 \times B$.
- **Powder Factor ($q$):** The concentration of chemical explosive energy, expressed as mass of explosive per unit volume ($\text{kg/m}^3$) or unit mass ($\text{kg/t}$) of rock.
- **Sub-drilling ($J$):** The length drilled below planned bench grade to prevent unbroken rock toes.
- **Stemming ($T$):** Sized angular inert aggregate packed above the explosive charge to confine high-pressure detonation gases.
- **Velocity of Detonation (VOD):** The speed at which the detonation reaction wave traverses the explosive column ($\text{m/s}$).
- **Relative Weight Strength (RWS):** Explosive energy per unit mass indexed against standard ANFO ($\text{ANFO} = 100$).
- **Kuz-Ram Model:** A classical semi-empirical equation system predicting rock fragmentation distributions ($x_{50}, n, x_c$) from blast geometry and rock properties.
- **Swebrec Function:** A three-parameter analytical distribution providing superior empirical fitting of blast-induced micro-fines and top-size boulders.
- **Passing Size ($P_{80}$):** Sieve aperture size through which 80% of fragmented rock mass passes.
- **Mine-to-Mill (M2M):** A systems-engineering framework optimizing drill-and-blast chemical energy to maximize downstream comminution throughput and minimize site operating cost.
- **Semi-Autogenous Grinding (SAG):** A primary grinding tumbling mill utilizing large rock chunks and steel grinding balls to pulverize run-of-mine ore.
- **Bond Work Index ($W_i$):** An empirical parameter expressing rock resistance to comminution in crushing and grinding circuits ($\text{kWh/t}$).
- **Blast Movement Monitoring (BMM):** Subsurface radio transmitters tracked post-blast to measure 3D heave displacement vectors.
- **Contact Dilution:** Barren waste rock inadvertently mined and milled due to imprecise post-blast boundary delineation.
- **Ore Loss (Misallocation):** Economic ore incorrectly dispatched to the waste dump due to post-blast boundary distortion.
- **Rimpull:** The tractive propelling force delivered by a haul truck's drive wheels to the ground surface ($\text{kN}$).
- **Grade Resistance ($GR$):** Gravitational opposing force acting on a vehicle on an inclined slope ($\approx 10\text{ kg/t}$ per $1\%$ grade).
- **Rolling Resistance ($RR$):** Friction and tire flexing resistance opposing vehicle motion on a haul road surface ($\text{kg/t}$ or equivalent $\%$ grade).
- **Total Resistance ($TR$):** Algebraic sum of Grade Resistance and Rolling Resistance ($TR = GR + RR$).
- **Ton-Kilometer Per Hour (TKPH):** A thermodynamic rating governing the maximum safe continuous mechanical and thermal workload of haul truck tires.
- **Match Factor ($MF$):** The ratio of truck arrival rate to shovel loading capacity; $MF = 1.0$ indicates perfect deterministic balance.
- **Closed Cyclic Queuing Network:** A queuing network model where a finite population of circulating vehicles interact with servers without external arrivals.
- **Two-Stage Dynamic Dispatch:** A dispatching architecture combining a macro-level LP flow optimizer with a real-time heuristic truck assignment engine.
- **Mean Value Analysis (MVA):** An exact recursive algorithm used to solve closed product-form queuing networks without evaluating normalising constants.
- **Autonomous Haulage System (AHS):** Driverless haul truck systems navigating via satellite GPS, LiDAR, and computer dispatch algorithms.
- **$F_2$ Reconciliation Factor:** The ratio of material delivered to the processing plant relative to the pre-blast grade control model.
- **Retarder Curve:** A manufacturer-certified operating curve defining the maximum safe downhill speed of a haul truck under continuous dynamic braking.
- **Swell Factor ($S_w$):** The volumetric expansion ratio of broken in-situ rock when fragmented and loaded into haulage assets.

---

## 16. Sources

- **Bond, F. C. (1952).** "The Third Theory of Comminution." *Mining Engineering*, 4(5), 484–494. (Foundational derivation of the comminution work index equation).
- **Cunningham, C. V. B. (1983).** "The Kuz-Ram Model for Prediction of Fragmentation from Blasting." *First International Symposium on Rock Fragmentation by Blasting*, Luleå, Sweden, 439–453. (Original Kuz-Ram model formulation).
- **Cunningham, C. V. B. (1987).** "Fragmentation Estimations and the Kuz-Ram Model – Four Years On." *Second International Symposium on Rock Fragmentation by Blasting*, Keystone, Colorado, 475–487. (Uniformity index modification and timing scatter adjustments).
- **Lilly, P. A. (1986).** "An Empirical Method of Assessing Rock Mass Blastability." *Large Open Pit Mining Conference*, AusIMM, Newman, Australia, 89–92. (Derivation of the Blastability Index $A$).
- **Ouchterlony, F. (2005).** "The Swebrec Function: Linking Fragmentation by Blasting and Crushing." *Mining Technology (Trans. Inst. Min. Metall. A)*, 114(1), 29–44. DOI: 10.1179/037178405X44548. (Introduction of the Swebrec distribution for micro-fines).
- **Scott, A., Napier-Munn, T. J., Morrell, S., & Kojovic, T. (2002).** *Mine-to-Mill: Optimisation of the Blasting, Crushing and Grinding Value Chain*. JKMRC Monograph Series in Mining and Mineral Processing, University of Queensland. (The authoritative reference on industrial M2M integration).
- **Morrell, S. (2004).** "Predicting the Specific Energy of Autogenous and Semi-Autogenous Mills from Small Diameter Drill Core Samples." *Minerals Engineering*, 17(3), 447–451. DOI: 10.1016/j.mineng.2003.10.018. (The Morrell comminution equations).
- **White, J. W., & Olson, J. P. (1986).** "Computer-Based Dispatching in Open Pit Mines." *Mining Engineering*, 38(6), 421–427. (Foundational two-stage LP dynamic dispatch formulation).
- **Hustrulid, W., & Kuchta, M. (2006).** *Open Pit Mine Planning and Design*. 2nd Edition, Taylor & Francis / Balkema, Rotterdam. ISBN: 978-0415405065. (Standard textbook on haul truck rimpull, haul road design, and cycle time decomposition).
- **Thornton, D. (2009).** "Application of Blast Movement Monitoring (BMM) to Reduce Dilution and Ore Loss in Open Pit Mines." *Proceedings of the Explo Conference*, AusIMM, Melbourne, 325–334. (Empirical field data and economics of 3D blast heave tracking).
- **Global Mining Guidelines Group (GMG) (2019).** *Guideline for Implementation of Autonomous Haulage Systems (AHS)*. GMG-AHS-01-2019. (Industry safety and telemetry standards for autonomous mining).
- **MSHA (2024).** *Title 30, Code of Federal Regulations (30 CFR): Mineral Resources*. Parts 56 and 57: Safety and Health Standards for Surface and Underground Metal and Nonmetal Mines. U.S. Government Publishing Office.

---

## 17. Links

- **Parent Module:** [Module 39 — Mining, Metals, and Forestry Supply Chains](_index.md)
- **Previous Topic:** [Topic 02 — Mine Planning, Cut-Off Grade, and Stope Optimization](02_mine-planning-cutoff.md)
- **Next Topic:** [Topic 04 — Grade Blending, Stockpiling, and Ore Specification](04_grade-blending.md)
- **Downstream Comminution Link:** [Topic 05 — Processing, Recovery, and Metallurgical Smelting](05_processing-recovery.md)
- **Logistics Link:** [Topic 06 — Pit-to-Port Logistics, Rail, and Marine Bulk Terminals](06_pit-to-port-logistics.md)
- **Foundational Optimization Link:** [Module 11 — Network Design Optimization](../../02_methods/11_network_design_optimization/_index.md)
- **Production Scheduling Link:** [Module 15 — Production Planning and Scheduling](../../02_methods/15_production_planning_scheduling/_index.md)
- **Process Industry Archetypes:** [Module 38 — Process Industries, Commodities, and Bulk Supply Chains](../../06_domain_electives/38_process_industries_commodities/_index.md)
- **Root Survey File:** [Survey File: Mining, Metals, and Forestry Supply Chains](../../../course/39_mining_metals_forestry.md)
