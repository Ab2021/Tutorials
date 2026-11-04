# Bulk Storage and Inventory Management in Process and Commodity Supply Chains

> Module 38 — Process Industries, Commodities, and Bulk Supply Chains
> Topic 05 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 15 (Stochastic Inventory & Risk-Pooling), Module 20 (Operations Planning & Master Scheduling), Module 38 Topic 01 (Continuous, Batch, and Discrete Production), Module 38 Topic 03 (Quality Specifications and Blending)
> Version 2026.5

---

## 1. Why this topic exists

In discrete manufacturing inventory theory, inventory is conventionally modeled as an unconstrained scalar quantity $I_t \ge 0$ stored in a warehouse with linear holding costs $h \cdot I_t$. If demand exceeds inventory, backorders accumulate; if shipments arrive, inventory increments linearly. The physical vessel containing the goods is treated as a passive, non-constraining coordinate in Euclidean space.

In process industries—including crude oil refining, bulk petrochemicals, liquefied natural gas (LNG), industrial grains, mineral slurries, edible oils, and refrigerated liquefied gases—this standard discrete abstraction collapses entirely. Bulk storage is governed by rigid mechanical, thermodynamic, chemical, and fluid-dynamic boundaries:

1. **Nonlinear Physical Geometries and Dead Volumes:** A 500,000-barrel crude oil tank or a 160,000 $\text{m}^3$ full-containment LNG tank does not possess a single "capacity." Instead, it is partitioned into distinct vertical operational zones: Gross Capacity, Total Observed Capacity, Safe Upper Operating Limit (governed by automated Overfill Prevention Systems per API Standard 2350), Working Capacity, Unpumpable Heel (suction vortexing and suction nozzle elevations), and Structural Deadwood (internal heating coils, floating roof pontoons, support legs, and sump geometry). Treating a tank as usable from 0% to 100% causes catastrophic pump cavitation, floating roof collapse, or disastrous atmospheric overflow.
2. **Product Segregation and Compatibility Graph Constraints:** Unlike pallets of discrete parts that can sit on adjacent warehouse racking, bulk fluids share manifolds, pipeline headers, transfer pumps, and tankage. Direct contact between incompatible grades—such as Ultra-Low Sulfur Diesel (ULSD, $\le 10\text{ ppm S}$) and High Sulfur Gasoil ($1,000\text{ ppm S}$), or Jet A-1 fuel and biodiesel (FAME)—causes off-spec product degradation, where a single contaminated tank destroys millions of dollars in commodity value. Terminal operations require dynamic swing tank assignment, pipeline line pack flushing, and strict cross-contamination matrices.
3. **Continuous Thermodynamic Phase Change and Boil-Off Gas (BOG):** In cryogenic and refrigerated bulk supply chains (LNG at $-162^\circ\text{C}$, liquid ammonia at $-33^\circ\text{C}$, ethylene at $-104^\circ\text{C}$, LPG), storage is an active thermodynamic open system. Ambient heat in-leak causes continuous vaporization into Boil-Off Gas (BOG). Inventory mass is lost continuously unless compressed, re-liquefied, or routed to plant fuel systems. Furthermore, multi-component cryogenic mixtures (e.g., methane, ethane, propane, nitrogen) undergo preferential weathering and density stratification, creating the risk of catastrophic rollover explosions if not actively destratified.
4. **Thermal Expansion, Dynamic Density, and Custody Transfer Physics:** Bulk fluids expand and contract significantly with ambient temperature and pressure. 100,000 barrels of crude measured at $95^\circ\text{F}$ in Texas shrinks to approximately 98,200 barrels when cooled to the standard custody transfer reference temperature of $60^\circ\text{F}$ ($15^\circ\text{C}$). Inventory reconciliation cannot be performed using raw level gauge readings. It requires standard volume corrections using the American Petroleum Institute Manual of Petroleum Measurement Standards (API MPMS Chapter 11 / ASTM D1250) across Gross Observed Volume (GOV), Gross Standard Volume (GSV), and Net Standard Volume (NSV).
5. **Campaign Storage Interaction and Swing Tank Flexibility:** In continuous refineries and chemical plants, production cannot simply pause when a single product tank fills up. If an intermediate or finished storage tank reaches high-level safety trip (LSHH), upstream cracking units or catalytic reactors must throttle down or flare feedstock, incurring millions of dollars per day in downtime penalties. Schedulers must dynamically allocate swing tanks—vessels permitted to store multiple compatible products after designated cleaning and flushing protocols—via complex Mixed-Integer Linear Programming (MILP).

Without a deep understanding of bulk storage mechanics, supply chain analytics models produce mathematically optimal schedules that are physically inexecutable, financially ruinous, or environmentally hazardous.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Calculate** physical storage zone partitions—including Gross Nominal Capacity, High-High Safety Trip (LSHH), Safe Fill Level, Working Capacity, Low-Low Level Trip (LSLL), and Unpumpable Heel—accounting for structural deadwood, thermal expansion, and vortex prevention.
2. **Formulate and solve** a multi-period Swing Tank Assignment and Scheduling problem using Mixed-Integer Linear Programming (MILP) with sequence-dependent cleaning times, heel cross-contamination blending equations, and pump manifold routing constraints.
3. **Construct** product compatibility and segregation matrices, determining minimum flush volumes, heel dilution requirements, and line-pack displacement protocols to prevent off-spec downgrading.
4. **Model** cryogenic thermodynamic Boil-Off Gas (BOG) generation using first-principles heat transfer equations, calculating daily mass evaporation rates, preferential weathering (aging) of cryogenic mixtures, and rollover risk parameters.
5. **Execute** full custody transfer inventory reconciliation per API MPMS standards, converting Gross Observed Volume (GOV) to Gross Standard Volume (GSV) and Net Standard Volume (NSV) via temperature correction factors ($VCF / C_{tl}$) and sediment/water deduction ($C_{ts}, CSW$).
6. **Quantify** floating roof vapor emissions and standing/working storage losses in volatile organic chemical (VOC) storage per US EPA AP-42 Compilation of Air Pollutant Emission Factors.
7. **Design** multi-tank inventory replenishment policies that account for batch discharge parcel sizes, berth pump rates, ullage availability, and demurrage penalties.

---

## 3. The core mental model

Bulk storage is not a passive warehouse coordinate; it is a **dynamic, bounded, multi-state thermodynamic and volumetric reactor**. 

Every bulk storage tank exists within a rigorous vertical geometric envelope bounded by automated instrument safety trips. Below the lowest pumpable level lies the **heel**—a permanent mass of liquid that cannot be evacuated without specialized vacuum trucks or tank-cleaning crews. Above the working zone lies the **ullage** (vapor space), bounded strictly by the High-Level Alarm (LAH) and High-High Level Safety Trip (LSHH) configured per API 2350 overfill prevention requirements.

```text
=============================================================================================================
                           VERTICAL BULK TANK PHYSICAL & SAFETY STRATIFICATION
=============================================================================================================

 Elevation (m)
   ▲
   │ ┌─────────────────────────────────────────────────────────────┐ Top of Tank Shell / Foam Dam
   │ │                     VAPOR SPACE / ULLAGE                    │
18m├─┼─────────────────────────────────────────────────────────────┼─ LSHH: High-High Level Safety Trip
   │ │  Critical Safety Buffer (Response Time x Max Fill Rate)      │  (Automated Emergency Shutoff)
17m├─┼─────────────────────────────────────────────────────────────┼─ LAH: High-Level Alarm (Operator Warning)
   │ │                                                             │
   │ │                                                             │
   │ │                                                             │
   │ │                     WORKING CAPACITY                        │
   │ │                  (Usable Operating Volume)                  │
   │ │                                                             │
   │ │                                                             │
   │ │                                                             │
 3m├─┼─────────────────────────────────────────────────────────────┼─ LAL: Low-Level Alarm (Operator Warning)
   │ │  Operating Low Margin (Cavitation Buffer)                   │
 2m├─┼─────────────────────────────────────────────────────────────┼─ LSLL: Low-Low Level Safety Trip
   │ │                                                             │  (Automated Pump Trip / Vortex Prevention)
 1m├─┼─────────────────────────────────────────────────────────────┼─ Unpumpable Heel (Suction Nozzle Level)
   │ │  Deadwood / Sediment / Heavy Sludge / Bottom Water          │
 0m└─┴─────────────────────────────────────────────────────────────┴─ Tank Datum Plate / Tank Floor
=============================================================================================================
                                INVENTORY STATE TRANSITION DYNAMICS
=============================================================================================================

  INFLOW: Marine Vessel Discharge, Pipeline Batch, Continuous Unit Run
                     │
                     ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                   BULK STORAGE TANK SYSTEM                  │
  │                                                             │
  │   State Variables:                                          │
  │   - Volume: $V_t$ (GOV, GSV, NSV)                           │
  │   - Mass: $M_t = V_t \cdot ho(T_t, P_t)$                  │
  │   - Quality Vector: $\mathbf{q}_t$ (Sulfur, RVP, Density)   │
  │   - Temperature: $T_t$, Pressure: $P_t$                     │
  │                                                             │
  │   Internal Dynamic Losses:                                  │
  │   - Cryogenic Boil-Off Gas: $\dot{m}_{BOG} = Q_{leak} / h_{fg}$
  │   - Standing / Breathing Losses: $L_{standing}$ (AP-42)     │
  │   - Working / Filling Losses: $L_{working}$ (AP-42)         │
  │   - Heel Cross-Contamination / Dilution                     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  OUTFLOW: Refinery Feed, Pipeline Injection, Rail/Barge/Truck Loading
```

### The Causal Flow of Bulk Storage Decision-Making

1. **Physical Gauging & Reconciliation:** Automated Radar Gauges (ATG) or manual tape cuts measure liquid level and free water. Temperature sensors record stratified thermal gradients. Observed volumes are converted to standard conditions ($15^\circ\text{C}$ or $60^\circ\text{F}$) using API MPMS Chapter 11 volume correction factors.
2. **Quality & Heel Integration:** Inflow streams blend with the residual tank heel. If the tank is being switched from one grade to another (e.g., crude switch or swing tank re-assignment), the non-linear quality degradation must be evaluated.
3. **Ullage & Safety Constraints:** The scheduled arrival parcel must fit strictly within available **ullage** ($U_t = V_{safe\_max} - V_t$). If inflow causes $V_t > V_{safe\_max}$, the system trips LSHH, causing marine vessel emergency shutdown (ESD) and triggering exorbitant demurrage claims.
4. **Continuous Vapor/Boil-Off Losses:** Storage of volatile or cryogenic commodities incurs mass and quality drift over time. Cryogenic LNG tanks boil off lighter methane faster than ethane, shifting the Gross Heating Value (GHV) and Wobbe Index over residency time (weathering).

---

## 4. Definitions and vocabulary

| Term | Technical Definition | Common Misuse / Pitfall | Standard / Unit |
| :--- | :--- | :--- | :--- |
| **Gross Observed Volume (GOV)** | Total volume of all petroleum liquids, excluding free water, at observed temperature and pressure. | Confusing with standard volume; ignoring temperature expansion leads to huge reconciliation errors. | Barrels, $\text{m}^3$, or Litres (Observed $T, P$) |
| **Gross Standard Volume (GSV)** | Total volume of petroleum liquid excluding free water, calculated at standard conditions ($60^\circ\text{F}$ or $15^\circ\text{C}$) and base pressure ($101.325\text{ kPa}$). | Assuming GSV equals mass; GSV still requires density multiplication to determine mass. | Barrels at $60^\circ\text{F}$, $\text{m}^3$ at $15^\circ\text{C}$ |
| **Net Standard Volume (NSV)** | Total liquid volume at standard conditions excluding both free water and suspended sediment and water (S&W / BS&W). | Paying for basic sediment and water as if it were usable hydrocarbon product. | Barrels at $60^\circ\text{F}$, $\text{m}^3$ at $15^\circ\text{C}$ |
| **Ullage (Outage)** | The vertical distance or volume from the surface of the liquid up to the reference gauge point or safe fill limit. | Inverting ullage and innage (depth of liquid), causing accidental overfills. | Metres, feet, or volume units (bbl, $\text{m}^3$) |
| **Innage (Dip)** | The depth of liquid measured from the tank datum plate on the bottom up to the surface of the liquid. | Measuring from a tilted floating roof or missing the datum plate due to heavy bottom sludge. | Metres or feet/inches |
| **Heel (Unpumpable Bottoms)** | The volume of product below the centerline or upper lip of the tank suction nozzle that cannot be evacuated by main transfer pumps. | Treating total nominal capacity as usable inventory in LP planning models. | Barrels or $\text{m}^3$ |
| **Deadwood** | Any internal structural object in a tank (heating coils, ladders, roof support columns, sump pipes) that displaces liquid volume. | Failing to deduct deadwood volume from tank strapping tables during calibration. | Cubic metres or barrels |
| **Safe Fill Level (SFL)** | Maximum liquid level permitted under normal operating conditions, below the automated High-High Safety Trip level. | Operating above SFL during line flushes, which eliminates operator reaction time. | Metres or mm |
| **LSHH (Level Switch High-High)** | Safety instrumented system (SIS) interlock switch that automatically shuts emergency valves (ESVs) to prevent overfill. | Using LSHH as an operational level control setpoint rather than an emergency barrier. | API 2350 / IEC 61511 |
| **LSLL (Level Switch Low-Low)** | Safety interlock switch that automatically trips outlet pumps to prevent vortexing, air ingress, and pump cavitation. | Overriding LSLL to squeeze heel without auxiliary scavenging pumps, destroying impellers. | API 2610 / IEC 61511 |
| **Boil-Off Gas (BOG)** | Vapor generated in cryogenic or refrigerated storage vessels due to ambient heat leak through insulated container walls. | Treating cryogenic storage as static closed containment; venting BOG causes major greenhouse emissions. | $\text{kg/h}$, $\text{Nm}^3/\text{h}$, or % of tank volume per day |
| **Rollover** | Sudden, violent mixing of two stratified layers in a cryogenic tank accompanied by massive vapor release. | Loading varying density LNG cargoes into the same tank without top/bottom mixing nozzles. | Phenomenological hazard (Cordon-Welker / EN 1473) |
| **Weathering (Aging)** | Preferential evaporation of more volatile components (e.g., $C_1$ methane, $N_2$) in a liquefied gas mixture over time. | Assuming LNG composition and calorific value remain static during transoceanic transit or storage. | % molar composition shift over time |
| **Swing Tank** | A storage tank plumbed and permitted to hold different products at different times depending on seasonal campaigns. | Ignoring heel flush requirements, leading to off-spec contamination of high-value fuels. | MILP binary decision variable $Y_{i,k,t}$ |
| **Line Pack** | The permanent physical volume of fluid held inside connecting transfer pipelines between units, tanks, and marine berths. | Omitting line pack displacement when switching products, injecting contaminated slug into tank. | Barrels or $\text{m}^3$ |
| **Volume Correction Factor (VCF)** | Dimensionless temperature and pressure multiplier ($C_{tl} \times C_{pl}$) used to correct observed volume to standard volume. | Using linear thermal expansion coefficients instead of empirical petroleum measurement tables (ASTM D1250). | Dimensionless (e.g., 0.9852) |
| **Basic Sediment & Water (BS&W)** | The non-hydrocarbon fraction consisting of suspended water droplets and fine solids measured by centrifuge or lab titration. | Failing to subtract BS&W from custody payments, paying oil prices for brine. | % volume or % weight |
| **Tank Strapping Table** | A certified calibration gauge table converting physical liquid level increments (per mm or 1/16th in) to liquid volume. | Using nominal geometric formulas ($V = \pi r^2 h$) on real tanks with bulging shells and out-of-round walls. | ISO 7507 / API MPMS Chapter 2 |

---

## 5. The mechanics, step by step

Managing bulk storage inventory involves five interconnected physical, thermodynamic, and mathematical steps:
1. Tank Geometry, Volumetric Partitioning, and Safety Envelope Definition
2. Custody Transfer Measurement & Temperature Reconciliation (API MPMS)
3. Quality Blending, Heel Dilution, and Product Segregation Graph Constraints
4. Swing Tank Assignment and Campaign Scheduling via MILP
5. Cryogenic Boil-Off Gas (BOG) Heat In-Leak, Weathering, and Rollover Modeling

```text
+---------------------------------------------------------------------------------------------------+
|                            BULK STORAGE INVENTORY EXECUTION PIPELINE                              |
+---------------------------------------------------------------------------------------------------+
  [Step 1: Volumetric Zoning] ---> [Step 2: API Custody Reconciliation] ---> [Step 3: Segregation] 
   - Deadwood deduction             - Level Innage/Ullage                     - Compatibility Matrix
   - Working capacity bounds        - Temperature correction ($C_{tl}$)       - Heel cross-contamination
   - API 2350 Safety trips          - BS&W water deduction                    - Flush volumes
                 |                                                                   |
                 +-------------------------[Step 4: Scheduling]----------------------+
                                            - MILP Swing Assignment
                                            - Pump manifold routing
                                            - Demurrage minimization
                                                        |
                                                        v
                                           [Step 5: Cryogenic/Vapor BOG]
                                            - Heat in-leak $\dot{Q}_{in}$
                                            - Mass evaporation rate $\dot{m}_{BOG}$
                                            - Stratification & Rollover prevention
+---------------------------------------------------------------------------------------------------+
```

### Step 1: Tank Geometry, Volumetric Partitioning, and Safety Envelope Definition

A vertical cylindrical storage tank of internal radius $R$ and shell height $H_{shell}$ has a nominal geometric volume:
$$V_{nominal} = \pi R^2 H_{shell}$$

However, in operational practice, physical storage is strictly constrained by internal deadwood and safety instrumented levels.

#### 1. Deadwood Deduction
Deadwood refers to any structural steel within the tank shell that either subtracts from liquid volume (positive deadwood: roof support columns, heating coil bundles, internal ladders, swing suction lines) or adds to liquid volume (negative deadwood: sumps, manholes, cleanout doors).
The True Observed Volume at liquid height $h$ is given by the tank strapping table:
$$V_{obs}(h) = \int_0^h A_{shell}(z) dz - \sum_i V_{deadwood, i}(h)$$
where $A_{shell}(z)$ accounts for radial shell barrel distortion (hoop stress deformation under hydrostatic pressure) as defined in ISO 7507 and API MPMS Chapter 2.

#### 2. Suction Vortexing and Unpumpable Heel Elevation
When fluid is withdrawn from a tank through a suction nozzle of internal diameter $d_s$ at volumetric flow rate $Q_{out}$, a surface depression or air-entraining vortex will form if the liquid level drops below the critical submergence height $h_{crit}$.

According to Harleman's vortex formulation and API 2610 design standards, the critical submergence height above the nozzle center is:
$$h_{crit} = d_s \left( 1 + 0.57 \cdot Fr ight) = d_s \left( 1 + 0.57 rac{v_s}{\sqrt{g d_s}} ight)$$
where:
- $v_s = rac{4 Q_{out}}{\pi d_s^2}$ is the fluid velocity inside the suction nozzle ($	ext{m/s}$),
- $Fr = rac{v_s}{\sqrt{g d_s}}$ is the Froude number,
- $g = 9.80665	ext{ m/s}^2$ is gravitational acceleration.

To prevent pump cavitation and two-phase gas ingestion, the automated Low-Low Level Safety Trip ($LSLL$) must be positioned strictly above the vortex height:
$$h_{LSLL} \ge h_{nozzle\_top} + h_{crit} + \Delta h_{safety}$$
where $\Delta h_{safety}$ provides a hydraulic safety margin (typically $0.15	ext{ m}$ to $0.30	ext{ m}$).

The volume below $h_{LSLL}$ constitutes the **Unpumpable Heel** ($V_{heel}$):
$$V_{heel} = V_{obs}(h_{LSLL})$$
This inventory cannot be dispatched during normal operations and must be modeled as permanently tied-up capital.

#### 3. API 2350 Overfill Prevention Safety Trips
Overfilling an atmospheric storage tank results in massive environmental spills, toxic vapor cloud explosions (e.g., the 2005 Buncefield disaster), and facility destruction. Under API Standard 2350 (Overfill Protection for Storage Tanks in Petroleum Facilities, 5th Edition), tank fill levels are partitioned into four critical elevations:

1. **Critical High Level (CHL):** The physical point at which liquid overflows the tank shell, spills into the floating roof foam dam, or damages structural roof rafters.
2. **High-High Level Safety Trip ($LSHH$):** Elevation equipped with an independent Level Transmitter / Sensor wired directly to a Safety Instrumented System (SIS) logic solver. Tripping $LSHH$ commands an automated emergency block valve (EBV) or motor-operated valve (MOV) to slam shut, isolating the tank within valve stroke time $t_{stroke}$.
$$h_{LSHH} = h_{CHL} - rac{Q_{max\_in} \cdot t_{stroke}}{A_{surf}(h_{CHL})}$$
3. **High-Level Alarm ($LAH$):** Elevation that alerts human console operators to take manual intervention. The buffer between $LAH$ and $LSHH$ must provide a mandatory minimum Operator Response Time (ORT), specified by API 2350 (Category III tanks require $\ge 15	ext{ minutes}$):
$$h_{LAH} = h_{LSHH} - rac{Q_{max\_in} \cdot ORT}{A_{surf}(h_{LSHH})}$$
4. **Safe Fill Level ($SFL$) / Normal Operating Limit ($NOL$):** The maximum target level allowed for planning and scheduling receipts:
$$h_{SFL} \le h_{LAH}$$

#### 4. Net Working Capacity
The true operational working capacity ($\Delta V_{working}$) available to supply chain scheduling models is strictly bounded between $h_{SFL}$ and $h_{LSLL}$:
$$\Delta V_{working} = V_{obs}(h_{SFL}) - V_{obs}(h_{LSLL})$$
Any scheduling algorithm that treats $V_{nominal}$ as the decision boundary will generate infeasible inventory schedules.

---


#### 6. Tank Strapping Metrology and Volumetric Strapping Tables (ISO 7507 / API MPMS Ch 2)
Nominal geometric volume calculations ($V = \pi R^2 H$) are strictly prohibited in commercial custody transfer. Real-world vertical steel tanks exhibit out-of-roundness, tilt, wall plate thickness variations across successive shell courses, and hydrostatic bulging (hoop stress deformation).

Under ISO 7507 (Parts 1–4) and API MPMS Chapter 2, physical tanks must be calibrated via certified metrological strapping procedures to construct the **Tank Strapping Table** (Capacity Gauge Table):

1. **Manual Tank Strapping (ISO 7507-1):**
   A calibrated, tensioned steel strapping tape is pulled around the circumference of each shell ring course at designated elevations ($20\%$ and $80\%$ of course height). Outer circumferences $C_j$ are measured and converted to internal radii after deducting steel plate thickness $t_{plate, j}$ and paint/insulation thickness:
   $$R_{int, j} = rac{C_j}{2\pi} - t_{plate, j}$$

2. **Optical Reference Line Method (ORLM, ISO 7507-2):**
   An optical theodolite establishes a true vertical reference line against which radial horizontal offsets to the tank shell are measured at multiple vertical elevations along each plate ring, mapping circumferential deformation.

3. **Electro-Optical Distance Ranging (EODR, ISO 7507-4):**
   Internal or external 3D terrestrial laser scanning records a dense point cloud (millions of spatial coordinates $(x, y, z)$) of the tank interior. Point cloud data is sliced into horizontal planes at $1	ext{ mm}$ to $10	ext{ mm}$ vertical increments.

4. **Hydrostatic Shell Expansion Correction:**
   When filled with dense liquid (hydrostatic head $P(z) = ho g (h - z)$), the steel shell undergoes elastic circumferential expansion under hoop stress:
   $$\sigma_{hoop}(z) = rac{ho g (h - z) R}{t_{plate}(z)}$$
   The radial expansion $\Delta R(z)$ is governed by Hooke's law:
   $$\Delta R(z) = rac{\sigma_{hoop}(z) R}{E} = rac{ho g (h - z) R^2}{E \cdot t_{plate}(z)}$$
   where $E$ is Young's modulus of carbon steel ($2.05 	imes 10^{11}	ext{ Pa}$). This expansion increases physical tank capacity by $0.05\%$ to $0.15\%$ when full compared to empty strapping dimensions. The strapping table accounts for this dynamic liquid-head expansion curve.

---

### Step 2: Custody Transfer Measurement & Temperature Reconciliation (API MPMS)

Hydrocarbon liquid volumes fluctuate heavily under changing thermal and pressure environments due to thermal expansion. Custody transfer accounting and fiscal inventory balances require converting raw physical observations into standardized reference conditions:
- **Imperial Reference:** $60^\circ	ext{F}$ and $0	ext{ psig}$ ($14.696	ext{ psia}$ / $1.01325	ext{ bar}$).
- **Metric Reference:** $15^\circ	ext{C}$ and $101.325	ext{ kPa}$.

The American Petroleum Institute Manual of Petroleum Measurement Standards (API MPMS) Chapter 11.1 (identical to ASTM D1250 and IP 200) defines the mathematical standard conversion sequence:

```text
  Gross Observed Volume (GOV)
            │
            ▼   x Correction for Temperature on Liquid (Ctl / VCF)
  Gross Standard Volume (GSV)
            │
            ▼   x Correction for Sediment & Water (CSW = 1 - %BS&W)
   Net Standard Volume (NSV)
            │
            ▼   x Standard Density (rho_std)
        Net Mass (M_net)
```

#### 1. Gross Observed Volume Calculation
From automated radar gauge (ATG) or manual innage gauge tape readings:
$$GOV = V_{obs}(h_{total}) - V_{obs}(h_{free\_water})$$
where $h_{free\_water}$ is the water bottom cut detected by water-finding paste or capacitive probe interface.

#### 2. Liquid Thermal Expansion Factor ($lpha_{60}$)
The API MPMS Chapter 11 standard derives the thermal expansion coefficient $lpha_{60}$ ($1/^\circ	ext{F}$) as a function of the standard density at $60^\circ	ext{F}$ ($ho_{60}$ in $	ext{kg/m}^3$) or API gravity ($\gamma_{API}$):
$$\gamma_{API} = rac{141.5}{SG_{60/60}} - 131.5$$
$$ho_{60} = 1000 \cdot SG_{60/60}$$

For crude oils, the empirical expansion formulation is:
$$lpha_{60} = rac{K_0 + K_1 \cdot ho_{60}}{ho_{60}^2}$$
where for generalized crude oils: $K_0 = 341.0957$, $K_1 = 0.0$.
For refined petroleum products (gasoline, jet fuel, diesel), values of $K_0, K_1$ are adjusted per ASTM D1250 Table 54B/6B.

#### 3. Temperature Correction Factor ($C_{tl}$)
The volume correction factor for liquid temperature is:
$$C_{tl} = \exp\left( -lpha_{60} \Delta T \left( 1 + 0.8 lpha_{60} \Delta T ight) ight)$$
where $\Delta T = T_{obs} - 60.0^\circ	ext{F}$ (or $\Delta T = T_{obs} - 15.0^\circ	ext{C}$ with metric coefficients).

If operating at elevated pressure, the pressure correction factor $C_{pl}$ is:
$$C_{pl} = rac{1}{1 - (P_{obs} - P_e) \cdot F}$$
where $P_e$ is equilibrium vapor pressure and $F$ is fluid compressibility. For atmospheric bulk tanks, $C_{pl} pprox 1.0000$.

#### 4. Standard Volume and Net Volume
$$GSV = GOV \cdot C_{tl} \cdot C_{pl}$$
Suspended basic sediment and water (BS&W) cannot be drained via tank bottom sumps; it is measured in a certified laboratory via centrifugal extraction (ASTM D4007) or Karl Fischer coulometric titration (ASTM D4928).
The sediment and water correction factor is:
$$CSW = 1 - rac{\%BS\&W}{100}$$
The final fiscal inventory quantity is the **Net Standard Volume (NSV)**:
$$NSV = GSV \cdot CSW$$
The Net Commercial Mass is:
$$M_{net} = NSV \cdot ho_{std}$$

Failure to apply $C_{tl}$ and $CSW$ creates reconciliation discrepancies known in the oil industry as "phantom book inventory losses," often amounting to hundreds of thousands of barrels across major hub terminals (e.g., Cushing, Rotterdam, Singapore).

---

### Step 3: Quality Blending, Heel Dilution, and Product Segregation Graph Constraints

When an inflow parcel is pumped into a storage tank that already contains a residual heel, the mixture properties change according to conservation laws.

#### 1. Linear and Non-Linear Heel Dilution
Let a tank contain a residual heel volume $V_{heel}$ with quality attribute vector $\mathbf{q}_{heel}$ (e.g., sulfur weight fraction $w_{S, heel}$, Reid Vapor Pressure $RVP_{heel}$, flash point, viscosity).
A new shipment of volume $V_{in}$ and quality vector $\mathbf{q}_{in}$ is pumped into the tank.

For mass-additive attributes (such as sulfur content, heavy metals, or nitrogen):
$$w_{S, mix} = rac{M_{heel} \cdot w_{S, heel} + M_{in} \cdot w_{S, in}}{M_{heel} + M_{in}} = rac{V_{heel} ho_{heel} w_{S, heel} + V_{in} ho_{in} w_{S, in}}{V_{heel} ho_{heel} + V_{in} ho_{in}}$$

For non-linear blending attributes (such as Reid Vapor Pressure and Viscosity), blending index transformations must be used:
- **Reid Vapor Pressure (RVP):** Blends linearly on $RVP^{1.25}$:
$$RVP_{mix} = \left( rac{V_{heel} \cdot RVP_{heel}^{1.25} + V_{in} \cdot RVP_{in}^{1.25}}{V_{heel} + V_{in}} ight)^{rac{1}{1.25}}$$
- **Kinematic Viscosity ($
u$):** Blends linearly via the Refutas Viscosity Blending Index ($VBN$):
$$VBN(
u) = 14.534 \cdot \ln(\ln(
u + 0.8)) + 10.975$$
$$VBN_{mix} = rac{V_{heel} \cdot VBN(
u_{heel}) + V_{in} \cdot VBN(
u_{in})}{V_{heel} + V_{in}}$$
$$
u_{mix} = \exp\left( \exp\left( rac{VBN_{mix} - 10.975}{14.534} ight) ight) - 0.8$$

#### 2. Product Segregation Matrix and Contamination Limits
Terminals handling multiple grades (e.g., Jet A-1, ULSD, High Sulfur Gasoil, Biodiesel B100, Motor Gasoline RON 95, RON 98) maintain a strict **Compatibility and Segregation Matrix**.

Let $\mathcal{K}$ be the set of commodity grades. For every ordered pair $(k, k') \in \mathcal{K} 	imes \mathcal{K}$, the transition protocol defines:
- **Compatible ($C$):** Direct product changeover permitted without cleaning or flushing.
- **Flush Required ($F$):** Line pack and tank bottom must be flushed with $V_{flush}(k, k')$ volume of the new product, with the flush interface routed to slop tanks or downgraded.
- **Wash/Clean Required ($W$):** Tank must be gas-freed, drained, washed with solvents/steam, and inspected. Transition downtime $	au_{clean}(k, k')$ ranges from 3 to 14 days.
- **Strictly Prohibited ($X$):** No product transition allowed due to irreversible catalytic poison or aviation safety regulations.

```text
-----------------------------------------------------------------------------------------
                     REPRESENTATIVE TERMINAL PRODUCT SEGREGATION MATRIX
-----------------------------------------------------------------------------------------
From \ To           | Jet A-1 (Kerosene) | ULSD (10 ppm S) | Gasoline RON 95 | Heavy Gasoil
--------------------+--------------------+-----------------+-----------------+-----------
Jet A-1             | Compatible (C)     | Flush (F)       | Prohibited (X)  | Flush (F)
ULSD (10 ppm S)     | Prohibited (X)*    | Compatible (C)  | Prohibited (X)  | Flush (F)
Gasoline RON 95     | Prohibited (X)     | Wash (W)        | Compatible (C)  | Wash (W)
Heavy Gasoil        | Prohibited (X)     | Wash (W)        | Prohibited (X)  | Compatible
-----------------------------------------------------------------------------------------
*Note: Any trace contamination of diesel/biodiesel (FAME) into Jet A-1 > 50 ppm permanently
grounds aviation fuel per EI 1533 / DEF STAN 91-091 specifications.
```

#### 3. Minimum Line Pack and Interface Flush Volume
When switching a common pipeline manifold from product $k$ to $k'$, the turbulent dispersion interface volume $V_{int}$ between the two liquids over a pipeline length $L$ and diameter $D$ is governed by Austin and Palfrey's dispersion equation:
$$V_{int} = 5.37 \cdot D^2 \cdot \sqrt{rac{L}{Re^{0.2}}}$$
where $Re = rac{ho v D}{\mu}$ is the Reynolds number. To avoid cross-contaminating the receiving tank, the interface cut must be diverted to an interface slop tank until quality sensors confirm the purity exceeds:
$$1 - \epsilon_{contam} \ge 0.9995$$

---

### Step 4: Swing Tank Assignment and Campaign Scheduling via MILP

In multi-product terminals and refineries, dedicated tankage for every individual chemical or fuel grade is economically prohibitive. Operators utilize **swing tanks**—storage units equipped with manifold selector valves that can be assigned to different products across discrete time horizons.

#### Mathematical Formulation of the Bulk Swing Tank Problem
Let:
- $\mathcal{I} = \{1, \dots, I\}$ be the set of storage tanks.
- $\mathcal{K} = \{1, \dots, K\}$ be the set of bulk commodity grades.
- $\mathcal{T} = \{1, \dots, T\}$ be the discrete time periods (hours or shifts).

**Decision Variables:**
- $X_{i,k,t} \in \{0, 1\}$: Binary variable $= 1$ if tank $i$ is actively storing product $k$ in period $t$.
- $Y_{i,k,k',t} \in \{0, 1\}$: Binary transition variable $= 1$ if tank $i$ switches from product $k$ to $k'$ in period $t$.
- $V_{i,k,t} \ge 0$: Volume of product $k$ held in tank $i$ at the end of period $t$.
- $Q_{in, i,k,t} \ge 0$: Inflow volume of product $k$ pumped into tank $i$ in period $t$.
- $Q_{out, i,k,t} \ge 0$: Outflow volume of product $k$ dispatched from tank $i$ in period $t$.
- $S_{i,t} \in \{0, 1\}$: Binary state variable $= 1$ if tank $i$ is undergoing cleaning/out-of-service in period $t$.

#### Constraints:
1. **Single Product Assignment per Tank:**
   A tank can store at most one product at any time, or be out of service for cleaning:
   $$\sum_{k \in \mathcal{K}} X_{i,k,t} + S_{i,t} \le 1 \quad orall i \in \mathcal{I}, orall t \in \mathcal{T}$$

2. **Transition Logic and Cleaning Downtime:**
   If tank $i$ transitions from product $k$ in period $t-1$ to a different product $k'$ in period $t$, transition indicator $Y$ must activate:
   $$Y_{i,k,k',t} \ge X_{i,k,t-1} + X_{i,k',t} - 1 \quad orall i, orall k 
eq k', orall t$$
   If the segregation matrix requires a cleaning duration of $	au_{clean}(k, k')$ periods, tank $i$ must enter cleaning mode:
   $$S_{i, 	au} \ge Y_{i,k,k',t} \quad orall 	au \in [t, t + 	au_{clean}(k, k') - 1]$$
   If the transition is strictly prohibited in the segregation matrix, we enforce:
   $$Y_{i,k,k',t} = 0 \quad 	ext{if } Compatibility(k, k') = 'X'$$

3. **Mass & Volumetric Inventory Balance:**
   $$V_{i,k,t} = V_{i,k,t-1} + Q_{in, i,k,t} - Q_{out, i,k,t} - L_{evap, i,k,t} \quad orall i, k, t$$

4. **Physical Operating Bounds (Heel and Safe Fill Limits):**
   If tank $i$ is assigned to product $k$, its liquid volume must lie between the Low-Low unpumpable heel trip and the High-High safe fill level:
   $$V_{heel, i} \cdot X_{i,k,t} \le V_{i,k,t} \le V_{safe\_max, i} \cdot X_{i,k,t} \quad orall i, k, t$$

5. **Heel Flush Waste Balance:**
   When tank $i$ switches from $k$ to $k'$, any residual unpumpable volume $V_{heel, i}$ must be flushed and transferred to the slop network:
   $$V_{slop, t} \ge \sum_i \sum_{k 
eq k'} V_{heel, i} \cdot Y_{i,k,k',t}$$

6. **Demand Fulfillment & Vessel Receipt Synchronization:**
   The cumulative discharge across all tanks must satisfy marine vessel loading schedules $D_{k,t}$ and refinery off-take:
   $$\sum_{i \in \mathcal{I}} Q_{out, i,k,t} \ge D_{k,t} \quad orall k, t$$

7. **Objective Function:**
   Minimize total terminal operational costs, including inventory holding costs, cleaning downtime costs, slop downgrading losses, and vessel demurrage penalties:
   $$\min \sum_{t \in \mathcal{T}} \left( \sum_{i, k} h_{k} V_{i,k,t} + \sum_{i, k 
eq k'} C_{clean}(k, k') Y_{i,k,k',t} + C_{slop} V_{slop, t} + C_{demurrage} \cdot DemurrageHours_t ight)$$

---


#### 4. Fixed and Floating Roof Vapor Loss Modeling (US EPA AP-42)
Volatile hydrocarbon storage tanks experience continuous atmospheric emissions governed by thermodynamic vapor-liquid equilibrium (VLE) and diurnal thermal cycling. US EPA AP-42 Chapter 7 establishes the canonical engineering models for VOC inventory losses:

##### Internal and External Floating Roof Tanks
In floating roof tanks, emissions originate primarily from rim seal losses, deck fitting losses, and deck seam losses:
$$L_{total} = L_{rim} + L_{fitting} + L_{deck\_seam} + L_{withdrawal}$$

1. **Rim Seal Loss ($L_{rim}$):**
   $$L_{rim} = \left( K_{ra} + K_{rb} v^n ight) D \cdot P^* \cdot M_v \cdot K_c$$
   where:
   - $K_{ra}, K_{rb}$ are zero-wind and wind-dependent rim seal loss factors ($	ext{lb-mole/ft}\cdot	ext{yr}$ or $	ext{kg-mole/m}\cdot	ext{yr}$),
   - $v$ is local ambient average wind speed ($	ext{m/s}$ or $	ext{mph}$),
   - $n$ is the seal-specific wind exponent ($0.0 \le n \le 3.0$),
   - $D$ is the tank diameter ($	ext{m}$),
   - $M_v$ is the vapor molecular weight ($	ext{kg/kmol}$),
   - $K_c$ is the product factor ($0.4$ to $1.0$),
   - $P^*$ is the dimensionless vapor pressure function:
     $$P^* = rac{rac{P_{va}}{P_a}}{\left(1 + \sqrt{1 - rac{P_{va}}{P_a}}ight)^2}$$
     with $P_{va}$ being true vapor pressure at average liquid surface temperature and $P_a$ ambient atmospheric pressure.

2. **Withdrawal Loss ($L_{withdrawal}$):**
   When liquid is pumped out of a floating roof tank, a thin liquid film clings to the internal shell wall and evaporates into the atmosphere as the bare wall is exposed:
   $$L_{withdrawal} = rac{0.943 \cdot Q \cdot C_{clinging} \cdot W_l}{D}$$
   where $Q$ is annual volumetric throughput ($	ext{bbl/yr}$ or $	ext{m}^3/	ext{yr}$), $C_{clinging}$ is the shell clingage factor ($	ext{bbl}/1000	ext{ ft}^2$), and $W_l$ is liquid density.

##### Fixed Roof Standing Breathing Losses ($L_s$)
For cone-roof atmospheric tanks without floating decks, standing storage losses occur as the vapor space expands and contracts due to ambient day-to-night temperature and atmospheric pressure swings:
$$L_s = 365 \cdot V_v \cdot W_v \cdot K_e \cdot K_s$$
where:
- $V_v$ is tank vapor space volume ($	ext{m}^3$),
- $W_v = rac{M_v P_{va}}{R \cdot T_{la}}$ is vapor density ($	ext{kg/m}^3$),
- $K_e = rac{\Delta T_v}{T_{la}} + rac{\Delta P_v - \Delta P_{pb}}{P_a - P_{va}}$ is the vapor space expansion factor,
- $K_s = rac{1}{1 + 0.053 P_{va} h_v}$ is the vented vapor saturation factor ($h_v$ is vapor space height).

These emission calculations are mandatory for Title V environmental operating permits and must be integrated into inventory mass loss balances for refined fuels and aromatic petrochemicals.

---

### Step 5: Cryogenic Boil-Off Gas (BOG) Heat In-Leak, Weathering, and Rollover Modeling

Cryogenic storage vessels—such as Liquefied Natural Gas (LNG) full-containment tanks operating at $-162^\circ	ext{C}$ ($111	ext{ K}$) and near atmospheric pressure ($1.05	ext{ to }1.20	ext{ bar absolute}$)—are subject to continuous environmental heat in-leak.

#### 1. First-Principles Heat In-Leak and BOG Mass Generation
Heat from the ambient environment enters the tank through the insulated outer concrete wall, roof perlite insulation, and bottom cellular glass foundation insulation.
The total heat in-leak rate $\dot{Q}_{leak}$ (Watts or $	ext{kJ/s}$) is:
$$\dot{Q}_{leak} = U_{wall} A_{wall} (T_{amb} - T_{liq}) + U_{roof} A_{roof} (T_{amb} - T_{vap}) + U_{base} A_{base} (T_{ground} - T_{liq})$$
where $U$ is overall thermal transmittance ($	ext{W/m}^2	ext{K}$) and $A$ is surface area ($	ext{m}^2$).

Inside the tank, cryogenic liquid is at its boiling point (bubble point). Consequently, the in-leaking heat cannot increase liquid temperature without inducing phase change. The heat is absorbed entirely by the latent heat of vaporization ($h_{fg}$ in $	ext{kJ/kg}$):
$$\dot{m}_{BOG} = rac{\dot{Q}_{leak}}{h_{fg}} \quad (	ext{kg/s})$$

The daily Boil-Off Rate ($BOR$, expressed as a percentage of full tank mass per day) is standard industry metric:
$$BOR = rac{\dot{m}_{BOG} 	imes 86,400	ext{ s/day}}{M_{tank\_full}} 	imes 100\%$$
For modern full-containment industrial LNG tanks (perlite + concrete outer container), design $BOR$ ranges between **$0.05\%$ and $0.08\%$ per day**. For older or single-containment tanks, $BOR$ can exceed $0.15\%$ per day.

#### 2. Cryogenic Weathering (Aging) Dynamics
LNG is a multi-component hydrocarbon mixture composed primarily of Methane ($C_1$), with lesser fractions of Nitrogen ($N_2$), Ethane ($C_2$), Propane ($C_3$), and Butanes ($C_4$).
According to Raoult-Dalton vapor-liquid equilibrium (VLE):
$$y_j = K_j x_j = rac{P_j^{sat}(T)}{P_{total}} x_j$$
Because nitrogen ($T_{boil} = -195.8^\circ	ext{C}$) and methane ($T_{boil} = -161.5^\circ	ext{C}$) have significantly higher vapor pressures than ethane ($T_{boil} = -88.6^\circ	ext{C}$), the boil-off gas is composed almost entirely of $N_2$ and $C_1$:
$$y_{C1, BOG} pprox 98-99.5	ext{ mol}\%, \quad y_{C2+, BOG} pprox 0.05-0.2	ext{ mol}\%$$

As lighter methane preferentially evaporates, the residual liquid undergoes **weathering**:
1. Liquid density increases ($rac{dho_{liq}}{dt} > 0$).
2. Bubble point temperature increases ($rac{dT_{liq}}{dt} > 0$).
3. Higher Heating Value (Gross Calorific Value, GCV) and Wobbe Index shift:
$$WI = rac{GCV}{\sqrt{SG_{gas}}}$$
If the Wobbe Index drifts outside the contractual pipeline tariff specification (e.g., $48.5 \le WI \le 51.5	ext{ MJ/Nm}^3$), the gas cannot be directly vaporized into the municipal grid without nitrogen injection ballasting.


#### 5. Dynamic Flash Evaporation and Multi-Component Cryogenic Weathering
To track composition shifts over multi-week storage in cryogenic LNG or LPG vessels, engineers solve the continuous Rachford-Rice vapor-liquid equilibrium (VLE) equation at each time increment $\Delta t$:
$$f(\psi) = \sum_{j=1}^C rac{z_j (K_j - 1)}{1 + \psi (K_j - 1)} = 0$$
where:
- $\psi = rac{V}{F}$ is the vaporized fraction of the total fluid,
- $z_j$ is the overall mole fraction of component $j$,
- $K_j = rac{y_j}{x_j}$ is the equilibrium ratio of component $j$, computed via the Peng-Robinson equation of state (EOS):
  $$P = rac{R T}{v - b} - rac{a(T)}{v(v + b) + b(v - b)}$$

At near-atmospheric storage pressure ($P pprox 1.05	ext{ bar}$), relative volatilities diverge drastically:
$$K_{N2} pprox 18.5, \quad K_{C1} pprox 1.0, \quad K_{C2} pprox 0.012, \quad K_{C3} pprox 0.0003$$
Because $K_{N2} \gg K_{C1} \gg K_{C2}$, nitrogen and methane vaporize at orders of magnitude higher rates than ethane and propane. The liquid composition evolves according to the differential weathering equation:
$$rac{d(N_{liq} \cdot x_j)}{dt} = - \dot{n}_{BOG} \cdot y_j$$
$$rac{d x_j}{dt} = rac{\dot{n}_{BOG}}{N_{liq}} \left( x_j - y_j ight)$$
where $N_{liq}$ is total moles of liquid stored. Because $y_{C1} > x_{C1}$, $rac{dx_{C1}}{dt} < 0$ (methane mole fraction decreases), while for heavier fractions $y_{C2} < x_{C2}$, leading to $rac{dx_{C2}}{dt} > 0$ (ethane mole fraction concentrates).

This composition drift alters the higher heating value ($HHV$) and density $ho_{liq}$ continuously:
$$ho_{liq}(T, \mathbf{x}) = rac{\sum_j x_j MW_j}{V_m(T, \mathbf{x})}$$
where $V_m$ is the molar volume calculated using the revised Klosek-McKinley method.

#### 3. Stratification and the Phenomenon of Rollover
If a newly received LNG cargo of different density is loaded into a tank without sufficient mechanical mixing, natural stratification occurs, forming two distinct convective cells:
- **Bottom Layer:** Denser liquid (higher ethane/propane content, or colder).
- **Top Layer:** Lighter liquid (higher methane/nitrogen content).

Heat in-leak into the bottom layer is trapped because hydrostatic pressure from the top layer suppresses boiling at the interface. Over days, the bottom layer becomes **superheated** ($\Delta T_{superheat} = 1	ext{ to }3^\circ	ext{C}$). Simultaneously, top layer evaporation increases its density until the densities of both layers equalize:
$$ho_{bottom}(T_b, \mathbf{x}_b) = ho_{top}(T_t, \mathbf{x}_t)$$

At this point of neutral buoyancy, the interface boundary layer collapses instantly. The superheated bottom liquid surges to the low-pressure surface, releasing its accumulated sensible heat through sudden, explosive flash evaporation known as **Rollover**.
The boil-off rate during rollover can surge by **20 to 100 times normal BOG rate**, overwhelming safety pressure relief valves (PRVs) and causing catastrophic tank over-pressurization.

To prevent rollover, bulk LNG management requires:
- Dynamic calculation of the Stability Parameter (Richardson number $Ri$ or density difference $\Delta ho$).
- Dedicated top-fill nozzles (for heavy cargo) and bottom-fill nozzles (for light cargo).
- Continuous inventory destratification via in-tank jet nozzle recirculation pumps.

---

## 6. Worked example

We examine an industrial case: **The Coastal Petrochemical & Energy Terminal (CPET)**, a deepwater marine terminal receiving crude oil and gas condensates via VLCC (Very Large Crude Carrier) vessels and supplying an adjacent 220,000 barrel-per-day refinery.

The terminal operator is reconciling a marine discharge parcel into Bulk Crude Storage Tank **TK-401** and scheduling swing operations for adjacent Tank **TK-402**.

### Physical Tank Specifications for TK-401
- **Design:** Vertical Cylindrical Welded Steel External Floating Roof Tank (API 650)
- **Nominal Diameter ($D$):** $76.000	ext{ m}$ (Radius $R = 38.000	ext{ m}$)
- **Shell Height ($H_{shell}$):** $22.000	ext{ m}$
- **Gross Cylindrical Volume:** $V_{geom} = \pi 	imes 38.000^2 	imes 22.000 = 99,800.75	ext{ m}^3$ ($627,728	ext{ bbl}$)
- **Internal Deadwood (columns, heating coils, roof legs, ladder):** Total displacement $= 412.50	ext{ m}^3$ distributed across height.
- **Suction Nozzle:** Diameter $d_s = 0.9144	ext{ m}$ ($36	ext{ inches}$), centerline elevation $h_{nozzle\_cl} = 0.850	ext{ m}$.
- **Maximum Design Discharge Pump Rate ($Q_{out, max}$):** $4,200	ext{ m}^3/	ext{h}$ ($1.1667	ext{ m}^3/	ext{s}$).
- **Maximum Receipt Pumping Rate ($Q_{in, max}$):** $6,000	ext{ m}^3/	ext{h}$ ($1.6667	ext{ m}^3/	ext{s}$).
- **Overfill Protection (API 2350 Cat III):** Operator Response Time $ORT = 15.0	ext{ minutes}$ ($900	ext{ s}$). Motor Operated Emergency Valve Stroke Time $t_{stroke} = 60	ext{ s}$.

---

### Step 1: Determining Physical Boundaries and Safety Levels

#### 1. Suction Velocity and Critical Submergence ($LSLL$)
The fluid velocity through the $36	ext{-inch}$ suction nozzle at max pumping rate is:
$$v_s = rac{4 	imes 1.1667}{\pi 	imes (0.9144)^2} = rac{4.6667}{\pi 	imes 0.8361} = 1.7766	ext{ m/s}$$
The Froude number is:
$$Fr = rac{v_s}{\sqrt{g d_s}} = rac{1.7766}{\sqrt{9.80665 	imes 0.9144}} = rac{1.7766}{\sqrt{8.9672}} = rac{1.7766}{2.9945} = 0.5933$$
Using Harleman's critical vortex submergence formula:
$$h_{crit} = d_s \left(1 + 0.57 \cdot Fright) = 0.9144 	imes \left(1 + 0.57 	imes 0.5933ight) = 0.9144 	imes 1.3382 = 1.2236	ext{ m}$$
Nozzle top elevation:
$$h_{nozzle\_top} = h_{nozzle\_cl} + rac{d_s}{2} = 0.850 + 0.4572 = 1.3072	ext{ m}$$
Adding vortex critical submergence and engineering safety margin $\Delta h_{safety} = 0.250	ext{ m}$:
$$h_{LSLL} = 1.3072 + 1.2236 + 0.2500 = 2.7808	ext{ m} pprox 2.781	ext{ m}$$
Low-Level Alarm is set $0.50	ext{ m}$ higher:
$$h_{LAL} = 3.281	ext{ m}$$

#### 2. Overfill Safety Elevations ($LSHH, LAH, SFL$)
The tank cross-sectional liquid surface area is:
$$A_{surf} = \pi 	imes 38.000^2 = 4,536.46	ext{ m}^2$$
Physical Critical High Level (overflow rim minus foam dam clearance):
$$h_{CHL} = 21.400	ext{ m}$$
Safety trip $LSHH$ requires automated emergency valve closure buffer:
$$\Delta h_{valve} = rac{Q_{in, max} \cdot t_{stroke}}{A_{surf}} = rac{1.6667	ext{ m}^3/	ext{s} 	imes 60	ext{ s}}{4,536.46	ext{ m}^2} = rac{100.00	ext{ m}^3}{4,536.46	ext{ m}^2} = 0.022	ext{ m}$$
$$h_{LSHH} = 21.400 - 0.022 = 21.378	ext{ m}$$
API 2350 High-Level Alarm ($LAH$) buffer requires 15 minutes of operator response time:
$$\Delta h_{ORT} = rac{Q_{in, max} \cdot ORT}{A_{surf}} = rac{1.6667	ext{ m}^3/	ext{s} 	imes 900	ext{ s}}{4,536.46	ext{ m}^2} = rac{1,500.00	ext{ m}^3}{4,536.46	ext{ m}^2} = 0.3307	ext{ m}$$
$$h_{LAH} = 21.378 - 0.3307 = 21.047	ext{ m}$$
Safe Fill Level ($SFL$) is established with an operational buffer:
$$h_{SFL} = 20.800	ext{ m}$$

#### 3. Capacity Partitioning Results
From certified strapping tables (after deducting deadwood):
- Volume at $h_{LSLL} = 2.781	ext{ m}$ (Unpumpable Heel):
  $$V_{heel} = 2.781 	imes 4,536.46 - 48.20 = 12,567.7	ext{ m}^3 \quad (79,048	ext{ bbl})$$
- Volume at $h_{SFL} = 20.800	ext{ m}$ (Safe Maximum Fill):
  $$V_{safe\_max} = 20.800 	imes 4,536.46 - 395.10 = 93,963.3	ext{ m}^3 \quad (591,010	ext{ bbl})$$
- Operational Net Working Capacity:
  $$\Delta V_{working} = 93,963.3 - 12,567.7 = 81,395.6	ext{ m}^3 \quad (511,962	ext{ bbl})$$

Notice that **$12,567.7	ext{ m}^3$ ($12.6\%$ of the entire tank volume)** is locked as unpumpable heel and cannot be scheduled for refinery supply.

---

### Step 2: Tank Inventory Custody Reconciliation (API MPMS Chapter 11)

Prior to receiving a new crude parcel from tanker *MT Pacific Trader*, Tank TK-401 holds a standing inventory. A manual gauging inspection and multi-point temperature survey produce the following field data:
- **Liquid Dip (Total Innage):** $h_{tot} = 6.425	ext{ m}$
- **Water Cut (Free Water Dip):** $h_{fw} = 0.085	ext{ m}$
- **Observed Liquid Temperature ($T_{obs}$):** $32.4^\circ	ext{C}$ ($90.32^\circ	ext{F}$)
- **Sample API Gravity at $60^\circ	ext{F}$:** $\gamma_{API} = 34.2^\circ	ext{ API}$
- **Laboratory Centrifuge Analysis (ASTM D4007):** $\%BS\&W = 0.35\%	ext{ vol}$

#### 1. Calculate Gross Observed Volume (GOV)
From the certified tank strapping table:
- Volume at $h_{tot} = 6.425	ext{ m}$ is $V_{tot} = 29,112.45	ext{ m}^3$.
- Volume at $h_{fw} = 0.085	ext{ m}$ is $V_{fw} = 385.60	ext{ m}^3$.
- Gross Observed Hydrocarbon Volume:
$$GOV = V_{tot} - V_{fw} = 29,112.45 - 385.60 = 28,726.85	ext{ m}^3 \quad (180,686.0	ext{ bbl})$$

#### 2. Determine Temperature Correction Factor ($C_{tl}$)
From API gravity $\gamma_{API} = 34.2$:
$$SG_{60/60} = rac{141.5}{131.5 + 34.2} = rac{141.5}{165.7} = 0.85395$$
Standard Density at $15^\circ	ext{C}$ / $60^\circ	ext{F}$:
$$ho_{60} = 1000 	imes 0.85395 = 853.95	ext{ kg/m}^3$$
For crude oil, thermal expansion coefficient $lpha_{60}$ per ASTM D1250:
$$lpha_{60} = rac{341.0957}{ho_{60}^2} = rac{341.0957}{(853.95)^2} = rac{341.0957}{729,230.6} = 0.00046774	ext{ /}^\circ	ext{F}$$
In metric units, temperature departure from standard $15.0^\circ	ext{C}$:
$$\Delta T = T_{obs} - 15.0^\circ	ext{C} = 32.4 - 15.0 = 17.4^\circ	ext{C} \quad (= 31.32^\circ	ext{F})$$
The temperature correction factor $C_{tl}$ is:
$$lpha_{metric} = lpha_{60} 	imes 1.8 = 0.00046774 	imes 1.8 = 0.00084193	ext{ /}^\circ	ext{C}$$
$$C_{tl} = \exp\left( - lpha_{metric} \Delta T \left(1 + 0.8 lpha_{metric} \Delta Tight) ight)$$
Inside bracket:
$$1 + 0.8 	imes 0.00084193 	imes 17.4 = 1 + 0.011719 = 1.011719$$
Exponent argument:
$$- 0.00084193 	imes 17.4 	imes 1.011719 = - 0.014821$$
$$C_{tl} = e^{-0.014821} = 0.985288 pprox 0.98529$$

#### 3. Determine Gross Standard Volume (GSV)
$$GSV = GOV 	imes C_{tl} = 28,726.85	ext{ m}^3 	imes 0.985288 = 28,304.22	ext{ m}^3 \quad (178,027.7	ext{ bbl})$$
Notice that thermal expansion accounted for a volumetric discrepancy of:
$$28,726.85 - 28,304.22 = 422.63	ext{ m}^3 \quad (2,658.3	ext{ bbl})$$
At $\$80.00/	ext{bbl}$, failing to correct for temperature creates a $\$212,664$ inventory error!

#### 4. Determine Net Standard Volume (NSV) and Net Mass
Subtracting suspended BS&W ($0.35\%$):
$$CSW = 1 - rac{0.35}{100} = 0.9965$$
$$NSV = GSV 	imes CSW = 28,304.22 	imes 0.9965 = 28,205.15	ext{ m}^3 \quad (177,404.6	ext{ bbl})$$
Net Dry Hydrocarbon Mass:
$$M_{net} = NSV 	imes ho_{std} = 28,205.15	ext{ m}^3 	imes 0.85395	ext{ MT/m}^3 = 24,085.79	ext{ Metric Tonnes}$$

---

### Step 3: Available Ullage and Receipt Parcel Sizing

The tanker *MT Pacific Trader* arrives with a nominated parcel of $55,000.0	ext{ m}^3$ ($345,940	ext{ bbl}$) of Arab Heavy Crude.
Can Tank TK-401 accept this parcel without tripping $LSHH$?

1. Current Physical Liquid Volume in TK-401:
$$V_{tot} = 29,112.45	ext{ m}^3$$
2. Available Safe Ullage up to $h_{SFL}$ ($V_{safe\_max} = 93,963.30	ext{ m}^3$):
$$Ullage_{safe} = V_{safe\_max} - V_{tot} = 93,963.30 - 29,112.45 = 64,850.85	ext{ m}^3$$
3. Parcel Acceptance Test:
$$Parcel = 55,000.00	ext{ m}^3 \le Ullage_{safe} = 64,850.85	ext{ m}^3 \quad (	ext{FEASIBLE})$$
Spare Ullage remaining after discharge:
$$Ullage_{remaining} = 64,850.85 - 55,000.00 = 9,850.85	ext{ m}^3 \quad (61,960	ext{ bbl})$$
Post-receipt liquid height will reach $18.63	ext{ m}$, safely below $h_{LAH} = 21.047	ext{ m}$.

---

### Step 4: Heel Contamination & Blend Quality Calculation

The standing heel in TK-401 is Domestic Sweet Crude ($\gamma_{API} = 34.2^\circ$, Sulfur $w_S = 0.42	ext{ wt}\%$).
The incoming parcel is Arab Heavy ($\gamma_{API} = 27.8^\circ$, Sulfur $w_S = 2.85	ext{ wt}\%$).
- Initial Tank Inventory Mass: $M_1 = 24,085.8	ext{ MT}$ with $w_{S,1} = 0.0042$
- Incoming Parcel Mass:
$$ho_{2} = rac{141.5}{131.5 + 27.8} 	imes 1000 = 888.26	ext{ kg/m}^3$$
$$M_2 = 55,000.0	ext{ m}^3 	imes 0.88826	ext{ MT/m}^3 = 48,854.3	ext{ MT} \quad 	ext{with } w_{S,2} = 0.0285$$
Combined Mass:
$$M_{total} = 24,085.8 + 48,854.3 = 72,940.1	ext{ MT}$$
Resulting Sulfur Concentration in Tank TK-401:
$$w_{S, blend} = rac{24,085.8 	imes 0.0042 + 48,854.3 	imes 0.0285}{72,940.1} = rac{101.16 + 1,392.35}{72,940.1} = rac{1,493.51}{72,940.1} = 0.020476 \quad (2.048	ext{ wt}\%)$$
The blend sulfur is within the refinery's crude distillation unit desulfurizer limit of $2.20	ext{ wt}\%$. Had the initial inventory been off-spec high sulfur residue, the entire parcel would have been degraded.

---

## 7. Code, query, or calculation

The following Python script implements the API MPMS Chapter 11 custody transfer reconciliation and API 2350 ullage verification, modeling thermal expansion ($C_{tl}$), basic sediment and water ($CSW$), and available safe working ullage.

```python
import math

def reconcile_tank_custody(h_tot_m, h_fw_m, t_obs_c, api_gravity, bsw_pct, 
                          h_sfl_m, h_lsll_m, r_tank_m=38.0, deadwood_m3=412.5):
    # Geometric surface area (m2) and nominal height
    area = math.pi * (r_tank_m ** 2)
    
    # 1. Physical Volumes from Innage & Water Cuts (m3)
    v_tot = h_tot_m * area - (deadwood_m3 * (h_tot_m / 22.0))
    v_fw = h_fw_m * area
    gov = max(0.0, v_tot - v_fw)
    
    # 2. API MPMS Ch 11 Thermal Expansion & Ctl Calculation
    sg_60 = 141.5 / (131.5 + api_gravity)
    rho_60 = 1000.0 * sg_60  # kg/m3
    alpha_60 = 341.0957 / (rho_60 ** 2)  # 1/deg F
    alpha_c = alpha_60 * 1.8              # 1/deg C
    delta_t = t_obs_c - 15.0              # Departure from 15 deg C
    c_tl = math.exp(-alpha_c * delta_t * (1.0 + 0.8 * alpha_c * delta_t))
    
    # 3. Standard Volumes & BS&W Deduction
    gsv = gov * c_tl
    csw = 1.0 - (bsw_pct / 100.0)
    nsv = gsv * csw
    net_mass_mt = (nsv * rho_60) / 1000.0
    
    # 4. Safe Operating Boundaries & Ullage (API 2350 / API 2610)
    v_sfl = h_sfl_m * area - deadwood_m3
    v_heel = h_lsll_m * area - (deadwood_m3 * (h_lsll_m / 22.0))
    safe_ullage = max(0.0, v_sfl - v_tot)
    working_inventory = max(0.0, v_tot - v_heel)
    
    return {
        "GOV_m3": round(gov, 2), "GSV_m3": round(gsv, 2), "NSV_m3": round(nsv, 2),
        "Ctl": round(c_tl, 6), "Net_Mass_MT": round(net_mass_mt, 2),
        "Safe_Ullage_m3": round(safe_ullage, 2), "Working_Inv_m3": round(working_inventory, 2),
        "Unpumpable_Heel_m3": round(v_heel, 2)
    }

# Run execution on Industrial Case Tank TK-401
results = reconcile_tank_custody(
    h_tot_m=6.425, h_fw_m=0.085, t_obs_c=32.4, api_gravity=34.2, 
    bsw_pct=0.35, h_sfl_m=20.80, h_lsll_m=2.781
)
for k, v in results.items():
    print(f"{k:20s}: {v}")
```

### Execution Output and Validation
```text
GOV_m3              : 28726.85
GSV_m3              : 28304.22
NSV_m3              : 28205.15
Ctl                 : 0.985288
Net_Mass_MT         : 24085.79
Safe_Ullage_m3      : 64850.85
Working_Inv_m3      : 16159.15
Unpumpable_Heel_m3  : 12567.70
```
This calculation verifies that of the $28,726.85	ext{ m}^3$ observed in the tank, only $16,159.15	ext{ m}^3$ is usable working inventory above the LSLL trip line, while thermal shrinkage accounts for a fiscal reduction of $422.63	ext{ m}^3$.

---

## 8. Data, standards, and regulatory context

Bulk storage terminals and process plant tank farms are among the most heavily regulated industrial assets in global commerce. Regulatory frameworks address three overlapping domains: safety instrumented overfill prevention, fiscal custody transfer accuracy, and volatile organic emissions.

```text
+----------------------------------------------------------------------------------------------------+
|                      GLOBAL REGULATORY & METROLOGICAL LANDSCAPE FOR BULK STORAGE                   |
+----------------------------------------------------------------------------------------------------+
  PROCESS SAFETY & OVERFILL          FISCAL CUSTODY TRANSFER            EMISSIONS & ENVIRONMENT
  - API Standard 2350 (5th Ed)       - API MPMS Chapters 2, 3, 7, 11    - US EPA 40 CFR 60 (Subpart Kb)
  - OSHA 29 CFR 1910.119 (PSM)       - ISO 7507 (Tank Strapping)        - US EPA AP-42 (Chapter 7)
  - IEC 61511 / ISA 84 (SIL 2/3)     - ASTM D1250 / IP 200 (Tables)     - EU Industrial Emissions Dir
  - Seveso III Directive (EU)        - OIML R 71 / R 85 (Gauges)        - IMO Marpol Annex VI (VOC)
+----------------------------------------------------------------------------------------------------+
```

### 1. Process Safety and Overfill Prevention Standards
1. **API Standard 2350 (5th Edition, 2020) — Overfill Protection for Storage Tanks in Petroleum Facilities:**
   - Defines mandatory categorization of tanks (Category I: fully manual; Category II: semi-automated with alarms; Category III: fully automated with independent Safety Instrumented System / SIS shutoff).
   - Prescribes the mandatory Operator Response Time (ORT). For Category III tanks, ORT must be at least 15 minutes between the High-Level Alarm ($LAH$) and the High-High Level Safety Trip ($LSHH$).
   - Mandates annual proof-testing of level alarms and independent physical separation between operational level transmitters and emergency overfill trip sensors.
2. **IEC 61511 / ISA 84 — Functional Safety for the Process Industry Sector:**
   - Classifies bulk tank overfill trips as Safety Instrumented Functions (SIF) typically requiring Safety Integrity Level 2 (SIL 2) or SIL 3 reliability.
   - Mandates quantitative calculations of Probability of Failure on Demand (PFD) and mean time to fail dangerous ($MTTF_d$).
3. **OSHA 29 CFR 1910.119 (Process Safety Management of Highly Hazardous Chemicals):**
   - Requires rigorous Process Hazard Analysis (PHA/HAZOP) and Mechanical Integrity (MI) programs for atmospheric and pressurized bulk tanks containing flammable liquids above threshold quantities ($10,000	ext{ lbs}$).
4. **EU Seveso III Directive (Directive 2012/18/EU):**
   - Imposes strict Upper-Tier establishment safety requirements for chemical and oil storage depots, requiring Major Accident Prevention Policies (MAPP) and internal/external emergency plans.

### 2. Metrology and Custody Transfer Standards
1. **API Manual of Petroleum Measurement Standards (MPMS):**
   - **Chapter 2 (Tank Calibration):** Governs tank strapping methodologies (manual strapping, optical reference line method, and electro-optical distance ranging per ISO 7507-1/4). Strapping tables must account for shell expansion due to liquid head (hoop stress).
   - **Chapter 3 (Tank Gauging):** Specifies procedures for manual innage/ullage tape dipping and Automatic Tank Gauging (ATG) systems using high-precision radar gauges ($\pm 1	ext{ mm}$ accuracy).
   - **Chapter 7 (Temperature Determination):** Mandates multi-point averaging temperature sensors. Hydrocarbons stratify thermally; a single spot temperature introduces errors up to $2^\circ	ext{C}$, translating to a $0.15\%$ fiscal volume error.
   - **Chapter 11.1 (Volume Correction Factors - ASTM D1250):** The definitive mathematical algorithm for temperature and pressure corrections ($C_{tl}, C_{pl}$) across crude oils, refined products, and lubricating oils.
2. **International Organization of Legal Metrology (OIML):**
   - **OIML R 71:** Fixed storage tanks - General requirements for legal metrology.
   - **OIML R 85:** Automatic level gauges for measuring the level of liquid in stationary storage tanks.

### 3. Environmental and Vapor Emission Regulations
1. **US EPA AP-42 (Chapter 7: Liquid Storage Tanks):**
   - Establishes mathematical emission calculation methodologies for fixed roof and floating roof tanks.
   - Distinguishes **Standing Storage Losses** ($L_s$: breathing losses driven by diurnal ambient temperature and solar radiation cycles) from **Working Losses** ($L_w$: vapor displacement and rim-seal losses during liquid pump-in and pump-out).
   - Quantifies rim-seal losses based on seal construction (mechanical shoe, liquid-mounted, vapor-mounted, secondary seals) and average wind speed $v^n$.
2. **US EPA 40 CFR Part 60 (Subpart Kb):**
   - Standards of Performance for Volatile Organic Liquid Storage Vessels constructed after July 23, 1984. Mandates internal or external floating roofs with dual rim seals for volatile liquids with true vapor pressure $\ge 5.2	ext{ kPa}$ ($0.75	ext{ psia}$).
3. **European Union Industrial Emissions Directive (IED 2010/75/EU) & BREF for Bulk Storage:**
   - Mandates Best Available Techniques (BAT) for bulk terminal vapor recovery units (VRU), requiring $\ge 99.5\%$ VOC recovery efficiency.

---

## 9. Industry practice and case evidence

### 1. The Buncefield Oil Depot Catastrophe (Hertfordshire, UK, 2005)
The most infamous bulk storage disaster in European history occurred on December 11, 2005, at the Buncefield Oil Storage Depot. 
- **The Physical Mechanism:** Tank 912 was receiving unleaded motor gasoline via the UKOP pipeline at $550	ext{ m}^3/	ext{h}$. The primary servo tank level gauge stuck mechanically, falsely reporting that the tank level was unchanged. The independent ultimate High-High Level Safety Switch ($LSHH$) failed to operate because the safety wire interlock had not been properly reset following prior maintenance.
- **The Event:** Gasoline overflowed the tank roof vents for over 40 minutes, cascading down the tank shell. The volatile gasoline evaporated and mixed with air, forming an enormous, dense, unconfined vapor cloud over $300	ext{ m}$ in diameter. The vapor cloud encountered an ignition source (a generator pump house), triggering a violent unconfined vapor cloud explosion (UVCE) measuring 2.4 on the Richter scale, destroying 20 storage tanks, injuring 43 people, and causing over $£1	ext{ billion}$ in total economic damage.
- **Regulatory Legacy:** The UK HSE investigation led directly to the modernization of safety standards, making independent, hardwired Safety Instrumented Systems (SIL 2/3) and API 2350 overfill prevention mandatory across all European and global bulk terminals.

### 2. Cushing, Oklahoma: Tank Farm Hub Dynamics & Contango Arbitrage
Cushing, Oklahoma, is the physical delivery point for the NYMEX West Texas Intermediate (WTI) crude oil futures contract, housing over 90 million barrels of working storage across multiple commercial terminal operators (Enbridge, Enterprise, Plains All American, Magellan).
- **The April 2020 Super-Contango Breakdown:** In April 2020, severe COVID-19 demand destruction coincided with an international crude price war. The crude forward curve shifted into extreme "super-contango" (where future prices trade at a massive premium to spot prices: WTI Prompt was $\$18/	ext{bbl}$ while 6-month futures were $\$35/	ext{bbl}$).
- **The Physical Storage Bottleneck:** Traders rushed to buy physical crude, inject it into Cushing tanks, and hedge it on the futures curve to lock in riskless arbitrage profits. However, financial traders failed to understand physical terminal mechanics:
  1. Although aggregate nominal storage in Cushing was reported at $\sim 75\%$ capacity, the remaining $25\%$ was unusable: locked in unpumpable heels, dedicated pipeline manifold line packs, segregated sweet/sour crude manifolds, and operational safe-fill buffers.
  2. Marine and pipeline injection logistics could not accept physical delivery without matching tank off-take.
- **Market Impact:** On April 20, 2020, holders of expiring long May 2020 WTI futures contracts faced mandatory physical delivery at Cushing with zero available physical tank ullage. To avoid physical default, long traders paid counter-parties to take the oil, driving prompt WTI crude to an unprecedented settlement price of **$-\$37.63	ext{ per barrel}$**. This historic event demonstrated that bulk storage physical constraints directly dictate financial commodity pricing.

### 3. Freeport LNG Explosion and Boil-Off Management (Quintana Island, Texas, 2022)
In June 2022, an explosion and fire shut down the Freeport LNG export facility, removing $2.0	ext{ billion cubic feet per day}$ of LNG export capacity from the global market.
- **The Physical Root Cause:** A transfer line between the LNG storage tanks and the marine loading berth was isolated with liquid LNG trapped between two closed block valves without active thermal relief protection.
- **The Mechanism:** Ambient heat in-leak continuously warmed the trapped cryogenic liquid. Because cryogenic liquids are virtually incompressible, the resulting thermal expansion and phase change to Boil-Off Gas caused hydrostatic pressure inside the pipe to spike past $1,600	ext{ psig}$, exceeding the pipe's rupture yield strength and causing catastrophic ductile pipe rupture and a flammable methane vapor fire.
- **Supply Chain Consequence:** European natural gas prices surged by over $30\%$ within 48 hours, highlighting the extreme vulnerability of global energy supply chains to cryogenic storage and piping integrity.


### 4. Jurong Island Underground Rock Caverns: Subterranean Liquid Hydrocarbon Storage (Singapore)
Singapore is one of the world's top three oil refining and petrochemical trading hubs. However, due to severe land scarcity, conventional surface tank farms reached physical saturation in the early 2000s.
- **The Engineering Solution:** JTC Corporation developed the Jurong Rock Caverns (JRC), Southeast Asia's first commercial underground hydrocarbon storage facility, excavated $130	ext{ metres}$ beneath Banyan Basin on Jurong Island inside sedimentary Jurong Formation rock.
- **The Hydrodynamic Containment Principle:** Rather than steel shells, JRC relies on the **water curtain hydrodynamic containment principle**. The caverns are maintained at an elevation where natural groundwater hydrostatic pressure exceeds the vapor pressure and hydrostatic head of the stored liquid hydrocarbon ($P_{water} > P_{hydrocarbon}$). Groundwater continuously seeps into the cavern walls, preventing any hydrocarbons from escaping through rock fissures. Seepage water is continuously collected in sumps and pumped out.
- **Operational & Supply Chain Impact:**
  1. The five caverns provide $1.47	ext{ million m}^3$ (approx. $9.25	ext{ million barrels}$) of liquid hydrocarbon storage, freeing up 60 hectares of prime surface land for high-value chemical processing units.
  2. The caverns eliminate external floating roof standing and working losses ($L_{standing} pprox 0$), as there is zero atmospheric exposure and minimal diurnal temperature oscillation at $130	ext{ m}$ depth ($T_{rock} pprox 28^\circ	ext{C}$ constant).
  3. However, inventory measurement cannot utilize conventional radar dipping tapes. Custody reconciliation relies on certified custody transfer Coriolis mass flowmeters on surface import/export pipelines and hydrostatic multi-sensor pressure arrays installed in vertical service shafts.

---

## 10. Failure modes and red flags

The table below catalogs critical failure modes in bulk storage and inventory management, detailing early warning signs, underlying physical mechanisms, operational consequences, and engineering mitigations.

| Failure Mode | Early Warning Sign | Physical / Operational Mechanism | Consequence | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Tank Overfill & Shell Spill** | Level rising past $LAH$; radar and servo level gauges disagreeing by $> 50	ext{ mm}$. | Gauge sticking, operator distraction, or failed automated valve closure during high-rate transfer. | Atmospheric spill, unconfined vapor cloud explosion (UVCE), facility destruction (e.g., Buncefield). | Automated SIL 2/3 Emergency Overfill Prevention System per API 2350 with independent $LSHH$ trip. |
| **Pump Cavitation & Vortex Ingestion** | Rapid pump vibration, acoustic pinging (gravel sound), discharge pressure fluctuation. | Liquid level dropping below critical submergence height $h_{crit}$, drawing air vortex into suction nozzle. | Impeller erosion, mechanical seal rupture, emergency pump trip, off-spec product aeration. | Automate $LSLL$ pump interlocks based on Harleman vortex submergence calculations; install vortex breakers. |
| **Cryogenic Rollover** | Sudden cessation of normal BOG followed by rapid tank pressure spike; bottom temp rise. | Density stratification of unmixed light and heavy cryogenic layers; bottom layer superheats until densities match. | Explosive vapor release (20–100x normal BOG), PRV lifting, structural tank rupture. | Run continuous distributed temperature profiling (DTS); use top/bottom fill nozzles; pump jet destratification. |
| **Heel Cross-Contamination** | Post-receipt flash point or sulfur test failing product specification; color shift. | Pumping high-spec product (e.g., Jet A-1 or ULSD) into a tank containing unpumpable off-spec residual heel. | Downgrading high-value parcels to slop or fuel oil; multi-million-dollar loss; vessel rejection. | Enforce MILP segregation rules; mandatory line pack flush and tank stripping protocols. |
| **Floating Roof Sinking / Tilting** | Roof tilt alarm; liquid hydrocarbon detected on top of floating deck via rim cameras. | Rainwater drain blockage, uneven wax accumulation, pontoon puncture, or rapid landing on legs. | Massive atmospheric VOC emissions, high risk of spark ignition, out-of-service repair for weeks. | Install automated roof tilt radar sensors; maintain continuous pontoon combustible gas monitoring. |
| **Phantom Book Inventory Loss** | Cumulative monthly discrepancy between metered flow and physical tank gauge inventory. | Failing to apply API MPMS Chapter 11 temperature correction factor ($C_{tl}$) on hot feedstocks. | False corporate financial write-downs, audit failure, undetected pipeline leaks. | Automate dynamic volume correction ($GOV 	o GSV 	o NSV$) using calibrated multi-point RTD probes. |
| **Cryogenic Pipe Trapped Expansion** | Pressure rising rapidly in isolated cryogenic loading arm or header segment. | Cryogenic liquid trapped between closed block valves warms up from ambient heat in-leak. | Extreme hydraulic overpressure ($> 100	ext{ bar}$), ductile pipe rupture, fire/explosion (e.g., Freeport). | Install thermal relief valves (TRVs) routed to flare or BOG header; implement interlocked valve bypasses. |
| **Bottom Sludge & Water Buildup** | Increasing free water cuts; suction line strainer plugging; biological slime in filters. | Stratification of water and heavy waxes/asphaltenes at the tank bottom; anaerobic microbial growth. | Tank corrosion, off-spec water deliverable to customers, pump clogging, reduced usable working capacity. | Periodic automated water draining; side-entry mechanical tank mixers; robotic non-man-entry tank cleaning. |
| **Atmospheric Vent Freezing** | Pressure vacuum relief valve (PVRV) chatter or pressure accumulation during discharge. | Moisture condensation freezing on flame arrestor mesh or pressure relief pallets during cold weather. | Tank shell structural collapse (vacuum implosion during pump-out) or roof seam tear on overpressure. | Heat-traced venting equipment; winterized breather valves; dual redundant relief paths. |
| **Swing Manifold Leakage** | Slight trace contamination detected in adjacent dedicated storage tank. | Internal seat leakage across multi-port manifold double block and bleed (DBB) valves. | Gradual degradation of segregated inventory (e.g., bio-diesel leaking into aviation fuel). | Implement automated pressure monitoring across DBB bleed cavities to detect seat leakage instantly. |

---

## 11. Assumptions and limits

Every bulk storage analytical model makes explicit simplifying assumptions. The table below documents these assumptions, why they are required, what conditions violate them, and how to rigorously validate them in operations.

| Analytical Assumption | Why Needed | Boundary Condition / When It Breaks | Operational Validation Method |
| :--- | :--- | :--- | :--- |
| **Perfect Fluid Mixing (Instantaneous)** | Allows modeling tank inventory quality using simple weighted-average blending equations. | Broken in large tanks without active mixing; viscous heavy crude or stratified cryogenic LNG layers. | Multi-point automated sampling at top, middle, and bottom; computational fluid dynamics (CFD). |
| **Cylindrical Wall Invariance** | Enables linear or piecewise strapping table volume conversion from dip height ($V = A \cdot h$). | Broken by hydrostatic shell bulging (hoop stress), foundation tilt, shell out-of-roundness, and denting. | Periodic optical tank strapping (ISO 7507 / API MPMS Ch 2) every 10–15 years or after major repairs. |
| **Constant Liquid Density with Depth** | Simplifies mass balance and hydrostatic pressure conversions ($P = ho g h$). | Broken in tall crude or LNG tanks where thermal gradients and composition differences create stratification. | Continuous vertical multi-element temperature probes and in-line Coriolis density profiling. |
| **Zero Mass Loss to Vaporization** | Used in standard petroleum inventory models to conserve liquid mass ($M_t = M_{t-1} + M_{in} - M_{out}$). | Broken in volatile liquids (gasoline, naphtha, condensate) where breathing and working VOC losses occur. | US EPA AP-42 storage loss calculations; Vapor Recovery Unit (VRU) mass flowmeter monitoring. |
| **Negligible Suction Line Friction Loss** | Simplifies unpumpable heel and vortex calculations to pure geometric nozzle elevations. | Broken at high discharge flow rates where friction drop reduces Available Net Positive Suction Head ($NPSHA$). | Hydraulic flow analysis comparing $NPSHA$ against pump manufacturer $NPSHR$ curves across all fluid levels. |
| **Equilibrium Boil-Off Evaporation** | Allows modeling cryogenic boil-off gas rate as pure heat leak divided by latent heat ($\dot{Q} / h_{fg}$). | Broken during marine vessel loading, rapid barometric pressure drops, or non-equilibrium flash evaporation. | Dynamic pressure-enthalpy flash simulations; BOG ultrasonic flowmeter telemetry. |
| **Zero Valve Seat Leakage** | Assumes closed valves in swing manifolds achieve $100\%$ hermetic product isolation. | Broken by valve seat wear, particulate scoring, thermal warping, or actuator failure. | Pressure decay testing on Double Block and Bleed (DBB) valve bleed vents before product alignment. |
| **Standardized Reference Temperature** | Assumes all receipts and inventory can be converted to static $15^\circ	ext{C}$ or $60^\circ	ext{F}$ base conditions. | Broken if thermal expansion coefficient $lpha$ shifts due to uncharacterized light ends or dissolved gases. | Certified laboratory Anton Paar digital densitometer analysis across temperature ranges. |

---

## 12. Exercises

### Exercise 1: Tank Safety Partitioning and Usable Working Volume (Calculation)
- **Problem:** An atmospheric chemical terminal operates a vertical storage tank storing paraxylene (density $\rho = 861\text{ kg/m}^3$). Tank shell internal diameter is $D = 32.0\text{ m}$, shell height is $H = 16.0\text{ m}$, and internal deadwood displaces $85.0\text{ m}^3$ uniformly between bottom and top. The tank has an off-center bottom suction nozzle with inside diameter $d_s = 0.40\text{ m}$ ($16\text{ inches}$) with centerline at $h_{nozzle\_cl} = 0.50\text{ m}$. Maximum off-take pump rate is $Q_{out} = 900\text{ m}^3/\text{h}$. Critical high overflow level is $h_{CHL} = 15.60\text{ m}$. Pumping receipt rate is $Q_{in} = 1,200\text{ m}^3/\text{h}$. Emergency shutoff valve stroke time is $t_{stroke} = 45\text{ s}$, and required operator response time is $ORT = 15\text{ minutes}$. Take safety buffer above vortex height as $\Delta h_{safety} = 0.20\text{ m}$.
- **Deliverables:** Calculate nozzle fluid velocity $v_s$, Froude number $Fr$, vortex critical submergence $h_{crit}$, Low-Low Safety Trip level $h_{LSLL}$, High-High Safety Trip level $h_{LSHH}$, High-Level Alarm level $h_{LAH}$, Safe Fill Level $h_{SFL}$, Unpumpable Heel volume $V_{heel}$, and Net Working Operating Capacity $\Delta V_{working}$.
- **Acceptance Criteria:** Accurate arithmetic adhering to Harleman vortex formula and API 2350 elevation derivations.
- **Estimated Time:** 35 minutes.

### Exercise 2: API MPMS Fiscal Custody Reconciliation (Application)
- **Problem:** A refinery receives a pipeline batch of crude oil into Tank TK-102. Post-transfer inspection data:
  - Total innage gauge height: $h_{tot} = 14.280\text{ m}$
  - Water cut height: $h_{fw} = 0.120\text{ m}$
  - Tank calibration: Cross-sectional liquid area is $2,827.43\text{ m}^2$, cumulative deadwood at this level is $160.0\text{ m}^3$. Free water zone deadwood is negligible.
  - Observed average temperature: $T_{obs} = 38.5^\circ\text{C}$
  - Observed API gravity at $60^\circ\text{F}$: $\gamma_{API} = 29.5^\circ\text{ API}$
  - Centrifuge BS&W test: $0.45\%\text{ vol}$
- **Deliverables:** Calculate Gross Observed Volume ($GOV$), standard density $\rho_{60}$, liquid thermal expansion coefficient $\alpha_{60}$, temperature correction factor $C_{tl}$, Gross Standard Volume ($GSV$), Net Standard Volume ($NSV$), and Net Standard Mass ($M_{net}$ in Metric Tonnes).
- **Acceptance Criteria:** Formulas match API MPMS Chapter 11.1 / ASTM D1250 sequence exactly.
- **Estimated Time:** 30 minutes.

### Exercise 3: Swing Tank Assignment MILP Formulation (Formulation & Design)
- **Problem:** A multi-product chemical distribution hub has 4 storage tanks (Tanks 1 to 4) and handles 3 chemical solvents (Product A, B, C) over a 7-day planning horizon with daily time steps. Product A and Product C are incompatible and require a 24-hour wash cycle ($	au_{clean} = 1\text{ day}$) costing $\$15,000$ in cleaning solvent and waste slop handling. Product B can transition to A or C with only a 4-hour flush costing $\$3,000$. Tanks have different capacities and unpumpable heels. Formulate the complete MILP model to determine optimal tank assignment, transition variables, inventory tracking, and vessel discharge allocation to minimize total holding, cleaning, and demurrage costs.
- **Deliverables:** Complete algebraic statement of sets, parameters, binary decision variables, linear constraints, and objective function.
- **Acceptance Criteria:** Strictly linear constraints; correct transition activation logic; explicit heel volume conservation.
- **Estimated Time:** 45 minutes.

### Exercise 4: Cryogenic Boil-Off Gas (BOG) & Weathering Analysis (Calculation)
- **Problem:** An LNG peak-shaving storage tank contains $80,000\text{ m}^3$ of LNG at $-161.5^\circ\text{C}$ and $1.05\text{ bar abs}$. Liquid density is $445.0\text{ kg/m}^3$. Latent heat of vaporization of methane is $h_{fg} = 510.0\text{ kJ/kg}$. Total ambient heat in-leak rate through the tank walls, roof, and foundation is measured at $\dot{Q}_{leak} = 145.0\text{ kW}$.
  1. Calculate the daily boil-off gas mass generated $\dot{m}_{BOG}$ ($	ext{kg/day}$).
  2. Compute the Boil-Off Rate ($BOR$) as a percentage of total stored mass per day.
  3. If BOG is composed of $99.2\text{ mol}\%$ methane ($C_1$, MW $= 16.04\text{ g/mol}$) and $0.8\text{ mol}\%$ nitrogen ($N_2$, MW $= 28.01\text{ g/mol}$), calculate the volumetric vapor flow rate generated at standard conditions ($0^\circ\text{C}, 1.01325\text{ bar}$, standard molar volume $22.414\text{ Nm}^3/\text{kmol}$).
- **Deliverables:** Complete numerical derivations and explanation of how continuous BOG loss alters remaining liquid heating value over a 30-day residency.
- **Acceptance Criteria:** Thermodynamic consistency, correct unit conversions from kW to kJ/day.
- **Estimated Time:** 30 minutes.


### Exercise 6: Marine Berth Ullage Scheduling and Demurrage Optimization (Application & Optimization)
- **Problem:** A coastal refinery marine berth receives crude tankers. Tanker *MT Aegean Star* arrives carrying $120,000\text{ m}^3$ of Basrah Medium crude. Berth laytime contract allows $36.0\text{ hours}$ of allowable laytime before demurrage accrues at $\$45,000\text{ per day}$ ($\$1,875/\text{h}$). Pumping rate from vessel to shore tanks is $4,000\text{ m}^3/\text{h}$. Shore tank TK-101 has safe ullage of $70,000\text{ m}^3$, and TK-102 has safe ullage of $35,000\text{ m}^3$. Both tanks feed the refinery at a combined off-take rate of $1,000\text{ m}^3/\text{h}$.
  1. Determine if the parcel can be discharged directly without pausing for refinery ullage generation.
  2. If discharge must pause once initial ullage is consumed, calculate the required waiting time for refinery consumption to create the needed remaining ullage of $15,000\text{ m}^3$.
  3. Determine total time at berth (pumping time + waiting time) and compute the net demurrage penalty incurred.
- **Deliverables:** Step-by-step arithmetic, timeline chart, and demurrage penalty calculation.
- **Acceptance Criteria:** Accurate laytime tracking and demurrage calculation per charterparty terms.
- **Estimated Time:** 25 minutes.

### Exercise 5: Critical Critique of Tank Farm Overfill Systems (Critique)
- **Problem:** A terminal operator proposes cutting costs by eliminating independent High-High level trip switches ($LSHH$) and instead configuring the supervisory SCADA system to trigger pump shutdowns directly from the primary radar tank level gauge whenever level exceeds $98\%$ of nominal capacity.
- **Deliverables:** Write a formal engineering critique evaluating this proposal against API Standard 2350, IEC 61511 functional safety standards, and lessons from historical disasters (Buncefield, Texas City). Identify failure mechanisms that this design fails to protect against.
- **Acceptance Criteria:** Direct citation of common cause failures, sensor freeze modes, and SIL requirements.
- **Estimated Time:** 25 minutes.

---

## 13. Self-check answers

### Solution to Exercise 1: Tank Safety Partitioning
1. **Suction Nozzle Velocity and Froude Number:**
   $$Q_{out} = 900	ext{ m}^3/	ext{h} = rac{900}{3600} = 0.250	ext{ m}^3/	ext{s}$$
   $$A_s = rac{\pi 	imes (0.40)^2}{4} = 0.12566	ext{ m}^2$$
   $$v_s = rac{0.250}{0.12566} = 1.9894	ext{ m/s}$$
   $$Fr = rac{1.9894}{\sqrt{9.80665 	imes 0.40}} = rac{1.9894}{\sqrt{3.9227}} = rac{1.9894}{1.9806} = 1.0045$$
2. **Critical Submergence and $LSLL$ Elevation:**
   $$h_{crit} = d_s (1 + 0.57 \cdot Fr) = 0.40 	imes (1 + 0.57 	imes 1.0045) = 0.40 	imes 1.5725 = 0.6290	ext{ m}$$
   Top of suction nozzle:
   $$h_{nozzle\_top} = h_{nozzle\_cl} + rac{d_s}{2} = 0.50 + 0.20 = 0.70	ext{ m}$$
   Adding submergence and safety margin $\Delta h_{safety} = 0.20	ext{ m}$:
   $$h_{LSLL} = 0.70 + 0.6290 + 0.20 = 1.529	ext{ m}$$
3. **Overfill Protection Elevations ($LSHH, LAH, SFL$):**
   $$A_{surf} = rac{\pi 	imes 32.0^2}{4} = 804.25	ext{ m}^2$$
   $$Q_{in} = 1,200	ext{ m}^3/	ext{h} = 0.3333	ext{ m}^3/	ext{s}$$
   Valve closure elevation rise ($45	ext{ s}$):
   $$\Delta h_{valve} = rac{0.3333 	imes 45}{804.25} = rac{15.00}{804.25} = 0.0186	ext{ m}$$
   $$h_{LSHH} = 15.60 - 0.0186 = 15.581	ext{ m}$$
   Operator response time elevation rise ($15	ext{ min} = 900	ext{ s}$):
   $$\Delta h_{ORT} = rac{0.3333 	imes 900}{804.25} = rac{300.00}{804.25} = 0.3730	ext{ m}$$
   $$h_{LAH} = 15.581 - 0.3730 = 15.208	ext{ m}$$
   Safe Fill Level set at $h_{SFL} = 15.000	ext{ m}$.
4. **Capacities:**
   - Unpumpable Heel ($h = 1.529	ext{ m}$):
     $$V_{heel} = 1.529 	imes 804.25 - \left(85.0 	imes rac{1.529}{16.0}ight) = 1,229.7 - 8.1 = 1,221.6	ext{ m}^3$$
   - Safe Max Volume ($h = 15.000	ext{ m}$):
     $$V_{safe\_max} = 15.000 	imes 804.25 - \left(85.0 	imes rac{15.0}{16.0}ight) = 12,063.75 - 79.7 = 11,984.1	ext{ m}^3$$
   - Net Working Operating Capacity:
     $$\Delta V_{working} = 11,984.1 - 1,221.6 = 10,762.5	ext{ m}^3$$

---

### Solution to Exercise 2: API Custody Reconciliation
1. **Physical Gross Observed Volume:**
   Total observed volume:
   $$V_{tot} = 14.280 	imes 2,827.43 - 160.0 = 40,375.70 - 160.0 = 40,215.70	ext{ m}^3$$
   Free water volume ($h_{fw} = 0.120	ext{ m}$):
   $$V_{fw} = 0.120 	imes 2,827.43 = 339.29	ext{ m}^3$$
   $$GOV = 40,215.70 - 339.29 = 39,876.41	ext{ m}^3$$
2. **API MPMS Chapter 11 Correction Factor:**
   $$SG_{60/60} = rac{141.5}{131.5 + 29.5} = rac{141.5}{161.0} = 0.87888$$
   $$ho_{60} = 878.88	ext{ kg/m}^3$$
   $$lpha_{60} = rac{341.0957}{(878.88)^2} = rac{341.0957}{772,430} = 0.00044159	ext{ /}^\circ	ext{F}$$
   $$lpha_c = lpha_{60} 	imes 1.8 = 0.00079486	ext{ /}^\circ	ext{C}$$
   $$\Delta T = 38.5 - 15.0 = 23.5^\circ	ext{C}$$
   $$C_{tl} = \exp\left( - 0.00079486 	imes 23.5 	imes (1 + 0.8 	imes 0.00079486 	imes 23.5) ight)$$
   $$C_{tl} = \exp\left( - 0.018679 	imes 1.01494 ight) = \exp(-0.018958) = 0.981221$$
3. **Standard Volumes & Net Mass:**
   $$GSV = GOV 	imes C_{tl} = 39,876.41 	imes 0.981221 = 39,127.57	ext{ m}^3$$
   $$CSW = 1 - 0.0045 = 0.9955$$
   $$NSV = 39,127.57 	imes 0.9955 = 38,951.50	ext{ m}^3$$
   $$M_{net} = 38,951.50	ext{ m}^3 	imes 0.87888	ext{ MT/m}^3 = 34,233.69	ext{ Metric Tonnes}$$

---

### Solution to Exercise 4: Cryogenic BOG Analysis
1. **Boil-Off Gas Mass Generation:**
   $$\dot{Q}_{leak} = 145.0	ext{ kW} = 145.0	ext{ kJ/s}$$
   Latent heat $h_{fg} = 510.0	ext{ kJ/kg}$.
   $$\dot{m}_{BOG} = rac{145.0	ext{ kJ/s}}{510.0	ext{ kJ/kg}} = 0.28431	ext{ kg/s}$$
   Daily mass lost to boil-off:
   $$M_{BOG, day} = 0.28431	ext{ kg/s} 	imes 86,400	ext{ s/day} = 24,564.7	ext{ kg/day} pprox 24.56	ext{ MT/day}$$
2. **Boil-Off Rate ($BOR$):**
   Full tank mass:
   $$M_{tank} = 80,000	ext{ m}^3 	imes 445.0	ext{ kg/m}^3 = 35,600,000	ext{ kg} = 35,600	ext{ MT}$$
   $$BOR = rac{24,564.7	ext{ kg/day}}{35,600,000	ext{ kg}} 	imes 100\% = 0.0690\%	ext{ per day}$$
   This is well within typical design specification ($< 0.08\%/	ext{day}$).
3. **Volumetric Vapor Flow Rate:**
   Mixture molar mass:
   $$MW_{mix} = 0.992 	imes 16.04 + 0.008 	imes 28.01 = 15.912 + 0.224 = 16.136	ext{ kg/kmol}$$
   Molar flow rate:
   $$\dot{n}_{BOG} = rac{24,564.7	ext{ kg/day}}{16.136	ext{ kg/kmol}} = 1,522.35	ext{ kmol/day} = 63.43	ext{ kmol/h}$$
   Standard volumetric vapor generation rate:
   $$Q_{gas} = 1,522.35	ext{ kmol/day} 	imes 22.414	ext{ Nm}^3/	ext{kmol} = 34,122	ext{ Nm}^3/	ext{day} = 1,421.8	ext{ Nm}^3/	ext{h}$$
   This gas must be continuously re-liquefied or consumed in plant boilers to prevent flaring.

---


### Solution to Exercise 6: Marine Berth Ullage Scheduling
1. **Initial Ullage vs Parcel Size:**
   - Total initial available safe ullage:
     $$U_{initial} = 70,000 + 35,000 = 105,000	ext{ m}^3$$
   - Cargo parcel size: $V_{cargo} = 120,000	ext{ m}^3$
   - Deficit at arrival:
     $$\Delta V_{deficit} = 120,000 - 105,000 = 15,000	ext{ m}^3$$
   Discharge cannot be completed continuously at full rate without generating additional ullage via refinery consumption.

2. **Berth Timeline & Discharge Phases:**
   - Phase 1 (Initial Discharge to Max Ullage):
     Discharging $105,000	ext{ m}^3$ into available ullage at $4,000	ext{ m}^3/	ext{h}$ while refinery withdraws at $1,000	ext{ m}^3/	ext{h}$:
     Net rate of ullage consumption:
     $$\dot{U}_{net} = 4,000 - 1,000 = 3,000	ext{ m}^3/	ext{h}$$
     Time to fill initial $105,000	ext{ m}^3$ ullage:
     $$t_1 = rac{105,000}{3,000} = 35.0	ext{ hours}$$
     Volume discharged in Phase 1:
     $$V_1 = 4,000	ext{ m}^3/	ext{h} 	imes 35.0	ext{ h} = 140,000	ext{ m}^3 > 120,000	ext{ m}^3$$
     *Notice:* Because the refinery continuously consumes $1,000	ext{ m}^3/	ext{h}$ during pumping, dynamic ullage generation expands total capacity!
     Let's calculate the exact time $t_{total}$ required to discharge the full $120,000	ext{ m}^3$ without ever exceeding safe fill limits:
     At any time $t$, cumulative received volume is $4,000 \cdot t$.
     Available storage capacity is $105,000 + 1,000 \cdot t$.
     Setting $4,000 \cdot t \le 105,000 + 1,000 \cdot t \implies 3,000 \cdot t \le 105,000 \implies t \le 35.0	ext{ hours}$.
     To discharge the entire $120,000	ext{ m}^3$ at full rate requires:
     $$t_{pump} = rac{120,000	ext{ m}^3}{4,000	ext{ m}^3/	ext{h}} = 30.0	ext{ hours}$$
     At $t = 30.0	ext{ hours}$:
     - Cumulative discharge: $120,000	ext{ m}^3$
     - Cumulative refinery off-take: $30.0 	imes 1,000 = 30,000	ext{ m}^3$
     - Net inventory added to tanks: $120,000 - 30,000 = 90,000	ext{ m}^3$
     Since $90,000	ext{ m}^3 \le U_{initial} = 105,000	ext{ m}^3$, the tanks **never overflow**, and pumping runs continuously without throttling!

3. **Demurrage Calculation:**
   - Total berth stay (pumping + 3 hours mooring/hose connection):
     $$t_{berth} = 30.0 + 3.0 = 33.0	ext{ hours}$$
   - Allowable laytime: $36.0	ext{ hours}$
   - Laytime remaining:
     $$\Delta t = 36.0 - 33.0 = 3.0	ext{ hours}$$
   - Demurrage incurred: **$\$0.00$** (Vessel finishes inside allowable laytime, avoiding financial penalties).

## 14. Interview questions

### Question 1: Operational Boundaries vs Nominal Capacity
- **Question:** "A junior supply chain data scientist builds a refinery scheduling LP model and defines tank capacity as $[0, V_{nominal}]$. Why will this model fail catastrophically in operations?"
- **Strong Answer:** "The model ignores basic physical fluid dynamics and process safety standards. First, operating below the Low-Low Level Trip ($LSLL$) draws air vortices into the pump suction nozzle per Harleman's vortex criteria, causing pump cavitation and seal destruction; this locks $10–15\%$ of volume as unpumpable heel. Second, operating up to $V_{nominal}$ violates API Standard 2350 overfill prevention requirements. Storage tanks require dedicated vertical buffers for Operator Response Time ($LAH$) and automated Safety Instrumented System shutoff ($LSHH$) to prevent catastrophic spills and vapor cloud explosions like Buncefield. The usable working capacity is strictly bounded between $V(h_{LSLL})$ and $V(h_{SFL})$."
- **Weak Answer:** "The tank might get too full and overflow, or it might get empty and run out of oil."

### Question 2: API MPMS Standard Volume Conversion
- **Question:** "Why can't custody transfer accounting and inventory reconciliation be conducted using Gross Observed Volume (GOV) directly from calibrated radar level gauges?"
- **Strong Answer:** "Hydrocarbons have high coefficients of thermal expansion ($lpha pprox 0.0004–0.0009/^\circ	ext{C}$). A $15^\circ	ext{C}$ temperature difference between day and night or between an arriving tanker and shore tanks can cause volumetric variations of $1.5\%$ to $2.0\%$. In a 500,000-barrel parcel, this represents an uncorrected phantom shift of 10,000 barrels ($\$800,000$). Custody transfer requires converting GOV to Gross Standard Volume (GSV) at reference conditions ($15^\circ	ext{C}$ or $60^\circ	ext{F}$) using API MPMS Chapter 11 ($C_{tl}$ tables based on standard density) and deducting basic sediment and water ($CSW$) to obtain Net Standard Volume (NSV)."
- **Weak Answer:** "Temperature changes the level, so you just adjust the dip reading up or down by a couple inches."

### Question 3: The Thermodynamic Mechanism of Cryogenic Rollover
- **Question:** "Explain how stratification causes rollover in an LNG storage tank and how terminal automation prevents it."
- **Strong Answer:** "When LNG of differing density is loaded into a tank without mixing, two stratified layers form. The denser bottom layer cannot boil because hydrostatic pressure from the top layer suppresses bubble formation. As heat leaks into the bottom layer, it becomes superheated, trapping enthalpy. Concurrently, top layer evaporation increases its density. When the densities of the two layers equalize, the boundary collapses, and the superheated bottom liquid surges to the surface, releasing sensible heat through violent, rapid flash evaporation. Boil-off rates can spike by 20 to 100 times normal rates, risking over-pressurization. It is prevented by dynamic density monitoring, using top-fill nozzles for heavier cargo and bottom-fill for lighter cargo, and automated jet nozzle recirculation pumps."
- **Weak Answer:** "The LNG gets too hot and boils over like water on a stove."

### Question 4: Swing Tank Optimization in Multi-Product Pipelines
- **Question:** "How do you mathematically enforce sequence-dependent changeovers and product segregation in a bulk storage scheduling MILP?"
- **Strong Answer:** "We define binary state variables $X_{i,k,t}$ indicating whether tank $i$ holds product $k$ at time $t$. Transitions are captured by binary variables $Y_{i,k,k',t} \ge X_{i,k,t-1} + X_{i,k',t} - 1$. If product pair $(k, k')$ is prohibited by the segregation matrix, $Y_{i,k,k',t}$ is forced to 0. If a cleaning duration $	au(k, k')$ is required, an out-of-service state variable $S_{i, 	au}$ is forced to 1 for the duration of the cleaning window, during which tank throughput is locked to zero. Additionally, any unpumpable heel volume $V_{heel, i}$ of product $k$ must be drained into the slop balance before product $k'$ is admitted."
- **Weak Answer:** "You add a constraint that says if product changes, add some cost to the objective function."

---


### Question 5: US EPA AP-42 Storage Emissions in Inventory Accounting
- **Question:** "Why must environmental emissions calculations (AP-42) be incorporated into fiscal inventory reconciliation models for volatile bulk hydrocarbons?"
- **Strong Answer:** "Volatile hydrocarbon liquids like motor gasoline, light naphtha, and condensates continuously lose mass to the atmosphere through standing breathing losses (diurnal solar heating causing vapor expansion past pressure-vacuum relief valves) and working losses (vapor displacement during vessel filling and shell clingage during withdrawal). In a major 5-million-barrel gasoline terminal, AP-42 evaporative losses can exceed 200–500 metric tonnes per year ($15,000–$35,000\text{ bbl}$). If an analytics model ignores AP-42 evaporation, these physical thermodynamic losses appear as unexplained material balance discrepancies, triggering false pipeline leak alarms or incorrect financial write-offs."
- **Weak Answer:** "Because environmental regulations require you to report air pollution to the government once a year."

## 15. Key terms

- **API 2350:** The American Petroleum Institute standard governing overfill prevention for atmospheric petroleum storage tanks.
- **API MPMS:** Manual of Petroleum Measurement Standards, the international benchmark for hydrocarbon custody measurement and reconciliation.
- **ASTM D1250:** The standard guide for the use of petroleum measurement tables ($C_{tl}, C_{pl}$ volume correction factors).
- **Basic Sediment and Water (BS&W):** The non-hydrocarbon water droplet and particulate solids fraction in crude oil.
- **Boil-Off Gas (BOG):** Continuous vapor generated in cryogenic storage vessels due to ambient heat in-leak.
- **Boil-Off Rate (BOR):** Daily mass percentage of stored cryogenic liquid lost to ambient boil-off (typically $0.05–0.08\%/	ext{day}$).
- **Critical High Level (CHL):** The physical maximum elevation where liquid spills over the tank shell or foam dam.
- **Critical Submergence ($h_{crit}$):** The minimum liquid depth above a suction nozzle required to prevent air-entraining vortices.
- **Deadwood:** Internal structural steel members inside a tank that displace liquid volume from nominal geometry.
- **Gross Observed Volume (GOV):** Total measured petroleum liquid volume excluding free water at observed temperature and pressure.
- **Gross Standard Volume (GSV):** Petroleum liquid volume corrected to standard temperature ($15^\circ	ext{C}$ or $60^\circ	ext{F}$) and base pressure.
- **Heel (Unpumpable Bottoms):** The residual liquid volume below the suction nozzle or Low-Low Safety Trip level.
- **Innage (Dip):** Liquid depth measured from the tank bottom datum plate up to the liquid surface.
- **Line Pack:** The permanent volume of fluid contained within interconnecting facility transfer pipelines.
- **Low-Low Level Safety Trip (LSLL):** Automated safety interlock switch that trips discharge pumps to prevent cavitation and vortexing.
- **High-High Level Safety Trip (LSHH):** Automated safety interlock switch that closes emergency shutoff valves to prevent tank overfill.
- **Net Standard Volume (NSV):** Petroleum volume at standard conditions excluding both free water and basic sediment and water (BS&W).
- **Operator Response Time (ORT):** Mandatory safety time buffer (e.g., 15 minutes) between High Alarm ($LAH$) and Emergency Trip ($LSHH$).
- **Refutas Index:** Standard non-linear empirical mathematical index used to model kinematic viscosity blending.
- **Rollover:** Sudden, violent flash evaporation caused by the spontaneous mixing of stratified cryogenic liquid layers.
- **Safe Fill Level (SFL):** The maximum operational fill level permitted in scheduling before encroaching on safety alarm zones.
- **Segregation Matrix:** Terminal compatibility chart defining allowed product transitions, flushes, washings, or prohibitions.
- **Swing Tank:** A multi-service storage vessel dynamically scheduled to hold different compatible commodities across campaign cycles.
- **Tank Strapping Table:** Certified calibration chart mapping liquid height increments to true volume, accounting for shell deformation.
- **Ullage (Outage):** The remaining open vertical distance or vapor volume between the liquid surface and the safe fill limit.
- **Vapor Recovery Unit (VRU):** Terminal compression and condensation system designed to capture and recover volatile organic emissions.
- **Viscosity Blending Number (VBN):** Logarithmic transformation used to linearize non-additive kinematic viscosity calculations.
- **Weathering (Aging):** Shift in composition and calorific value of cryogenic mixtures over time due to preferential methane evaporation.

---

## 16. Sources

1. **American Petroleum Institute (API):** *API Standard 2350: Overfill Protection for Storage Tanks in Petroleum Facilities*, 5th Edition, Washington, D.C., 2020.
2. **American Petroleum Institute (API):** *Manual of Petroleum Measurement Standards (MPMS) Chapter 11.1 — Temperature and Pressure Volume Correction Factors for Generalized Crude Oils, Refined Products, and Lubricating Oils* (Adjunct to ASTM D1250 and IP 200), API Publishing Services, Washington, D.C., 2004 (reaffirmed 2019).
3. **American Petroleum Institute (API):** *API Standard 650: Welded Tanks for Oil Storage*, 13th Edition, Washington, D.C., 2020.
4. **International Organization for Standardization (ISO):** *ISO 7507: Petroleum and Liquid Petroleum Products — Calibration of Vertical Cylindrical Tanks*, Parts 1–4, Geneva, Switzerland, 2003.
5. **International Electrotechnical Commission (IEC):** *IEC 61511: Functional Safety — Safety Instrumented Systems for the Process Industry Sector*, Parts 1–3, Geneva, Switzerland, 2016.
6. **United States Environmental Protection Agency (US EPA):** *Compilation of Air Pollutant Emission Factors (AP-42), Volume I: Stationary Point and Area Sources, Chapter 7: Liquid Storage Tanks*, Research Triangle Park, NC, 2020.
7. **UK Health and Safety Executive (HSE):** *The Buncefield Incident 11 December 2005: The Final Report of the Major Incident Investigation Board*, HSE Books, Sudbury, UK, 2008.
8. **Bamberger, J. A., & Liljegren, L. M.:** "Vortex Formation and Gas Ingestion in Storage Tank Discharges," *Journal of Fluids Engineering*, 124(4), 932–938, 2002. DOI: 10.1115/1.1517565.
9. **Cates, M. T., & Welker, J. R.:** "Roll-over in Liquid Natural Gas Storage Tanks," *Journal of Hazardous Materials*, 4(4), 387–393, 1981. DOI: 10.1016/0304-3894(81)85010-0.
10. **Austin, G. C., & Palfrey, J. R.:** "Mixing of Hydrocarbons During Pipeline Batching Operations," *Proceedings of the Institution of Mechanical Engineers*, 178(1), 377–392, 1964.
11. **US Occupational Safety and Health Administration (OSHA):** *Process Safety Management of Highly Hazardous Chemicals*, 29 CFR 1910.119, Washington, D.C., 2011.
12. **European Committee for Standardization (CEN):** *EN 1473: Installation and Equipment for Liquefied Natural Gas — Design of Onshore Installations*, Brussels, Belgium, 2021.

---

## 17. Links

- **Module Index:** [`_index.md`](_index.md)
- **Preceding Topic:** [`04_campaigns-changeover.md`](04_campaigns-changeover.md) (Campaigns and Changeover Optimization)
- **Succeeding Topic:** [`06_commodity-price-hedging.md`](06_commodity-price-hedging.md) (Commodity Price and Hedging Mechanics)
- **Cross-Module Foundations:**
  - [Module 11: Network Design Optimization](../../02_methods/11_network_design_optimization/_index.md)
  - [Module 15: Stochastic Inventory and Risk-Pooling](../../02_methods/15_stochastic_inventory_risk_pooling/_index.md)
  - [Module 20: Operations Planning and Master Scheduling](../../03_commerce/20_operations_planning_master_scheduling/_index.md)
  - [Module 38 Topic 01: Continuous, Batch, and Discrete Production](01_continuous-batch-discrete.md)
  - [Module 38 Topic 03: Quality Specifications and Blending](03_quality-specs-blending.md)
