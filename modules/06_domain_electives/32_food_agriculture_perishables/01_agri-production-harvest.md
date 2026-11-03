# Agricultural Production and Harvest Planning

> Module 32 — Food, Agriculture, and Perishable Supply Chains
> Topic 01 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Module 07 (Time Series Forecasting), Module 10 (Inventory Optimization), Module 21 (Supply Chain Network Design & Optimization)
> Version 2026.5

---

## 1. Why this topic exists

In conventional discrete manufacturing and retail supply chains, production schedules are governed by master production schedules (MPS), material requirements planning (MRP), and factory line capacities. If consumer demand rises by 15%, a manufacturing plant can schedule overtime, activate a secondary shift, or order additional raw materials from suppliers operating under service level agreements. Inventory in those sectors is treated as a stable capital asset that incurs financial holding costs but does not spontaneously decompose, lose mass, or degrade in chemical composition while stored in a warehouse.

When supply chain professionals apply conventional discrete manufacturing planning logic to agricultural and perishable supply chains, the operational result is systemic failure. Millions of dollars of biological raw material spoil in fields, intake queues at processing plants back up for miles, processing assets sit idle during peak periods, and growers face catastrophic contract penalties or crop rejection.

This breakdown occurs because **agricultural production is biologically decoupled from downstream demand cycles**. 

Agricultural raw materials—including processing tomatoes, wine grapes, sugarcane, sweet corn, malting barley, citrus, and leafy greens—are governed by thermodynamics, solar radiation, soil hydrology, and plant phenology. Their growth rate, maturity timestamp, and ultimate harvest yield are driven by cumulative ambient heat (thermal time) and environmental stress rather than human scheduling desires. Once a biological crop reaches physiological maturity, a rigid and irreversible biochemical countdown begins. Respiration accelerates, water transpires across cell walls, starches convert to sugars or invert into organic acids, tissues soften through enzymatic pectin breakdown, and fungal pathogens rapidly colonize the senescing plant organs.

Consider the four structural pathologies that arise when supply chains fail to engineer the farm-to-processing interface:

1. **The Harvest Bunching Shock:** A processing tomato cannery contracts 4,000 hectares of farmland across a 100-kilometer radius, planting staggered blocks over six weeks in March and April to distribute intake evenly across an 80-day operating campaign. In late June, an anomalous 10-day heatwave accelerates crop development across all parcels simultaneously. Field phenology converges. Instead of the planned daily intake of 3,500 metric tonnes, 8,200 metric tonnes reach physiological maturity in the exact same 7-day window. Because the factory’s physical intake, fluming, sorting, and evaporation lines have a hard engineering limit of 3,800 metric tonnes per day, 30,000 metric tonnes of mature fruit are forced to dwell in high ambient temperatures. Fields left unharvested rot on the vine, while trucks waiting 24 hours at the factory weighbridge experience mold bloom and acid souring, triggering massive financial losses and breach-of-contract lawsuits between grower cooperatives and industrial processors.
2. **The Cut-to-Crush Sucrose Inversion Trap:** In the sugarcane and sugar beet industries, harvesting severs the stalk or root from its biological nutrient supply. Within hours, endogenous invertase enzymes hydrolyze sucrose into non-crystallizable reducing sugars (glucose and fructose). For every six-hour delay between mechanical cutting and industrial milling, Raízen and other major cane processors measure a recoverable sugar loss of 0.8% to 1.5%, accompanied by dextran contamination that fouls industrial evaporators and crystallization centrifuges. Without real-time, mixed-integer mathematical optimization synchronizing combine harvesters, tractor-towed infield haulers (transbordos), over-the-road transport convoys, and mill intake queues, transportation bottlenecks silently destroy hundreds of thousands of dollars of industrial sucrose per day.
3. **The Maturity vs. Mass Trade-off:** Agricultural yield (tonnes per hectare) and industrial quality (soluble solids, dry matter percentage, tenderness, acidity, or color) follow conflicting non-linear trajectories over time. Harvesting sweet corn or green peas 48 hours early yields tender, high-grade produce that commands premium canning prices but sacrifices 15% to 25% of total harvestable biomass. Harvesting 48 hours late maximizes total tonnage and satisfies grower revenue incentives under crude weight-based contracts, but converts soluble sugars into starch, causing the crop tenderometer index to exceed acceptable thresholds, forcing the factory to downgrade Grade A inventory to animal feed.
4. **Machinery and Labor Capital Traps:** Modern agricultural harvesting requires specialized, high-capital machinery (e.g., self-propelled multi-row tomato harvesters costing $650,000 each; four-row sugar beet harvesters costing $900,000 each) and highly trained crews operating under strict seasonal labor regulations. If spatial dispatching is uncoordinated, multi-million-dollar harvester fleets waste 35% of their working hours road-traveling between non-contiguous field parcels rather than operating, creating self-inflicted field delays during fleeting weather windows.

To establish an efficient, resilient agri-food supply chain, operations researchers and supply chain architects cannot treat agricultural supply as an exogenous random variable. Planners must model the biophysical mechanisms of crop growth, predict maturity dates using meteorological data, optimize harvest schedules using mixed-integer linear programming (MILP), and align commercial grower contracts with processing plant intake realities. This topic establishes the end-to-end engineering, agronomic, and mathematical foundations for agricultural production and harvest planning.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Calculate** cumulative thermal time using Growing Degree Days (GDD) with crop-specific base and ceiling temperatures to forecast crop development stages and harvest maturity windows.
2. **Formulate** biophysical crop yield response models linking reference evapotranspiration ($ET_0$), dual crop coefficients ($K_{cb}, K_e$), and soil water balance to predict biomass and marketable yield under moisture stress.
3. **Quantify** post-maturity quality degradation curves (soluble solids accumulation, acid degradation, tenderometer creep, and sucrose inversion) to evaluate the economic penalty of early or delayed harvesting.
4. **Construct** a mixed-integer linear program (MILP) for multi-field harvest scheduling that maximizes delivered net margin while enforcing daily harvesting fleet limits, plant intake capacities, non-splittable field operational constraints, and weather downtime risks.
5. **Analyze** field-to-processing transportation queues and cut-to-processing windows to eliminate raw material queuing dwell times that accelerate enzymatic spoilage.
6. **Design** grower procurement contracts—evaluating acreage-based, tonnage-based, and quality-adjusted pricing grids—that align grower incentives with processing plant throughput and grade requirements.
7. **Map** regulatory and traceability compliance architectures for commercial harvesting, integrating US FDA Food Safety Modernization Act (FSMA) Rule 204 Critical Tracking Events (CTEs), GLOBALG.A.P. pre-harvest intervals, and GS1 standards.

---

## 3. The core mental model

To engineer an agricultural production and harvest planning system, you must trace the flow of biophysical state variables through field operations and logistics to factory conversion. The system is governed by a tripartite coupling: the **Biophysical Growth Engine**, the **Harvest Operations Dispatcher**, and the **Processing Intake Bottleneck**.

```text
+---------------------------------------------------------------------------------------------------+
| 1. BIOPHYSICAL CROP GROWTH & PHENOLOGY ENGINE (Field Parcel Level)                                |
|                                                                                                   |
|  [ Solar Radiation / Temp / Precip ] ──> [ Thermal Accumulation (GDD) ]                          |
|  [ Soil Water Balance / ET_c       ] ──> [ Biomass Growth & Phenology  ]                          |
|  [ Cultivar / Sowing Date          ] ──> [ Maturity Window [ES_i, LF_i] ]                         |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 2. HARVEST OPERATIONS & DISPATCH OPTIMIZATION (Regional Fleet Level)                              |
|                                                                                                   |
|  [ Harvester Fleet Capacity (ha/day) ] ──> [ Mixed-Integer Linear Program (MILP) ]                |
|  [ Crew Scheduling & Road Transit     ] ──> [ Daily Field Selection: x_{it}      ]                |
|  [ Precipitation Downtime Risk        ] ──> [ Harvest Output: Q_{it} tonnes      ]                |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 3. INFIELD LOGISTICS & BULK TRANSPORT (Transport Corridor Level)                                  |
|                                                                                                   |
|  [ Mechanical Harvester / Bins ] ──> [ Infield Haulage (Chaser Bin) ] ──> [ Bulk Road Transport ] |
|                                                                                                   |
|  Perishability Clock: Respiration Heat + Invertase Activity + Acid Drop (Dwell Time <= Tau_max)  |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| 4. INDUSTRIAL INTAKE & CONVERSION BOTTLENECK (Processing Facility Level)                          |
|                                                                                                   |
|  [ Weighbridge & Grading Lab (Brix, Defects) ] ──> [ Flume / Dry Unloading ]                     |
|  [ Raw Material Surge Buffer (Pad / Silo)    ] ──> [ Daily Processing Line Capacity Cap_{jt} ]   |
|  [ Contract Settlement: Base Price +/- Bonus ] ──> [ Packaged Canned / Frozen / Milled Product ]  |
+---------------------------------------------------------------------------------------------------+
```

### 3.1 The Three Governing Dynamics

1. **The Thermal Time Clock:** Unlike industrial parts that progress through manufacturing operations based on conveyor speeds, crops advance through phenological stages according to thermal time. Heat accumulation dictates flowering, fruit set, and physiological ripening. Because temperature varies stochastically across seasons and microclimates, harvest maturity dates cannot be scheduled via static calendar dates; they must be continuously updated via real-time and forecasted Growing Degree Days (GDD).
2. **The Field-Plant Throughput Squeeze:** The agro-industrial interface represents a classic supply-demand synchronization problem under severe physical constraints. The agricultural supply base is spatially dispersed across hundreds of individual parcels with stochastic harvest readiness. Conversely, the industrial processing facility (sugar mill, vegetable cannery, juice concentrator, winery) is a high-capital, fixed-location asset designed for continuous, steady-state throughput. If field harvest exceeds factory intake capacity, perishable trucks queue in the sun, driving severe biochemical decay. If field harvest falls below factory intake capacity, fixed operating costs ($30,000 to $100,000 per idle shift) destroy plant margins.
3. **The Divergent Utility Frontier:** Without deliberate contractual alignment, growers and processors optimize conflicting objective functions. A grower paid purely on gross delivered metric tonnes will delay harvest to maximize water weight and biomass accumulation, heedless of declining sugar concentration or increasing skin defects. A processor requires maximum Brix, low acidity, uniform fruit size, and high solids recovery to minimize evaporation energy and maximize yield per operating hour. The planning system must harmonize these conflicting utilities through mathematical optimization and price grid incentives.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse | Unit or Coding Case |
|---|---|---|---|
| **Growing Degree Day (GDD)** | The daily accumulation of ambient thermal energy above a biological base temperature ($T_{\text{base}}$) and below an upper developmental cutoff ($T_{\text{upper}}$), governing crop development rates. | Confusing GDD with calendar days or mean daily temperature; assuming GDD accumulates linearly regardless of extreme heat stress. | Degree-days ($^\circ\text{C}\cdot\text{d}$ or $^\circ\text{F}\cdot\text{d}$); $[T \cdot \Theta]$. |
| **Base Temperature ($T_{\text{base}}$)** | The minimum developmental threshold temperature below which metabolic enzymatic activity ceases and crop phenological progress stops. | Using a single universal base temperature ($10^\circ\text{C}$) for all crops (e.g., peas grow at $4.4^\circ\text{C}$, sweet corn requires $10^\circ\text{C}$, sugarcane requires $18^\circ\text{C}$). | Degrees Celsius ($^\circ\text{C}$) or Fahrenheit ($^\circ\text{F}$). |
| **Upper Threshold ($T_{\text{upper}}$)** | The developmental ceiling temperature above which the rate of plant phenological development plateaus or declines due to thermal stress and cellular protein denaturation. | Omitting $T_{\text{upper}}$ in hot climates, which drastically overestimates heat accumulation during heatwaves. | Degrees Celsius ($^\circ\text{C}$) or Fahrenheit ($^\circ\text{F}$). |
| **Crop Evapotranspiration ($ET_c$)** | The volume of water lost through plant transpiration and soil surface evaporation from a disease-free, well-fertilized crop under optimal soil moisture conditions: $ET_c = K_c \cdot ET_0$. | Conflating reference evapotranspiration ($ET_0$) with actual crop water consumption ($ET_c$ or $ET_a$). | Millimeters per day ($\text{mm/day}$); $[L \cdot T^{-1}]$. |
| **Dual Crop Coefficient ($K_{cb} + K_e$)** | The split coefficient dividing crop water demand into basal crop transpiration ($K_{cb}$) and soil surface evaporation ($K_e$), following the FAO-56 standard. | Using a single static $K_c$ across all growth stages without accounting for fractional canopy cover ($f_c$) or wet soil surface intervals. | Dimensionless ratio ($\mathbb{R}^+$). |
| **Yield Response Factor ($K_y$)** | The empirical dimensionless coefficient relating relative crop evapotranspiration deficit to relative yield reduction: $(1 - Y_a/Y_m) = K_y(1 - ET_a/ET_m)$. | Assuming yield drops linearly across all growth stages; moisture stress during flowering/pollination exhibits far higher $K_y$ than during vegetative growth. | Dimensionless sensitivity ratio ($\mathbb{R}^+$). |
| **Physiological Maturity** | The biological developmental stage at which the harvested organ (seed, fruit, root, tuber) ceases net dry matter accumulation and reaches optimal biological composition. | Confusing physiological maturity (biological peak) with commercial maturity (market harvest readiness, which may occur earlier or later). | Categorical stage / BBCH phenology scale index. |
| **Harvest Window ($[ES_i, LF_i]$)** | The bounded chronological time interval between the Earliest Start ($ES_i$) date and the Latest Finish ($LF_i$) date during which a field can be harvested without unacceptable loss of yield or quality. | Treating the harvest window as an open-ended period without explicit yield/quality degradation penalties at the boundaries. | Discrete calendar days or planning periods ($t \in \mathcal{T}$). |
| **Soluble Solids Content ($^\circ\text{Brix}$)** | The percentage of soluble sugars and solids dissolved in plant sap or extracted juice, measured using optical refractometry. | Assuming $^\circ\text{Brix}$ measures only sucrose; it captures all dissolved solids, including glucose, fructose, salts, and amino acids. | Degrees Brix ($^\circ\text{Bx}$); percentage by weight ($\%\text{ w/w}$). |
| **Tenderometer Value (TU)** | The mechanical force required to shear a standardized volume of vegetable tissue (commonly green peas and sweet corn), indicating tenderness and starch conversion. | Assuming higher tenderometer values represent better quality; higher TU indicates tougher, starchier, lower-value product. | Tenderometer Units ($\text{TU}$ or $\text{psi}$). |
| **Cut-to-Crush / Cut-to-Process Window** | The maximum allowable elapsed time between severed field harvest and the initiation of industrial thermal processing or juice extraction before biological decay degrades economic value. | Treating transport transit time as the only delay, ignoring infield trailer queuing and factory weighbridge wait times. | Hours ($\text{h}$); $[T]$. |
| **Respiration Rate ($R_O$)** | The metabolic rate at which harvested plant tissue consumes oxygen and carbohydrates to produce carbon dioxide, water, and heat: $\text{C}_6\text{H}_{12}\text{O}_6 + 6\text{O}_2 \to 6\text{CO}_2 + 6\text{H}_2\text{O} + 2,870\text{ kJ}$. | Ignoring the self-heating effect of harvested crops piled in open-top bulk trailers under summer solar loads. | $\text{mg CO}_2\text{ / (kg}\cdot\text{h)}$ or $\text{kJ / (tonne}\cdot\text{day)}$. |
| **Ethylene Inversion / Evolution** | The biological emission of ethylene gas ($\text{C}_2\text{H}_4$) triggering accelerated ripening, chlorophyll loss, and senescence in climacteric produce. | Co-loading high ethylene-producing crops (apples, melons) with ethylene-sensitive crops (cucumbers, leafy greens) in bulk transit. | $\mu\text{L / (kg}\cdot\text{h)}$; $[L^3 \cdot M^{-1} \cdot T^{-1}]$. |
| **Intake Bottleneck Capacity ($Cap_{jt}$)** | The maximum physical receiving, washing, and processing rate of industrial facility $j$ on day $t$, constrained by flume lines, weighbridges, or primary evaporator/cooker capacities. | Sizing harvest fleet capacity strictly to average factory capacity without building buffers for factory breakdowns or clean-in-place (CIP) cycles. | Metric tonnes per day ($\text{t/day}$); $[M \cdot T^{-1}]$. |
| **Non-Splittable Field Constraint** | The operational requirement that once mechanical harvesters begin harvesting a parcel, they must complete the entire parcel without redeploying to an alternate farm. | Allowing mathematical optimization models to split a 5-hectare parcel across four non-consecutive days, causing prohibitive equipment roading costs. | Binary logic constraint ($\{0, 1\}$). |
| **Pre-Harvest Interval (PHI)** | The mandatory statutory minimum time required by agricultural chemical regulations between the final pesticide application and the physical harvest of the crop. | Scheduling a field for early harvest to meet an industrial factory shortage while a chemical PHI restriction remains active, resulting in illegal residue violations. | Elapsed days ($\text{d}$). |
| **Acreage Contract** | A procurement contract where the processor contracts all production from a specified land area, bearing yield risk while securing exclusive crop volume. | Believing acreage contracts eliminate supply risk; if regional yields collapse, the processor experiences volumetric feedstock starvation. | Surface area (hectares, $\text{ha}$). |
| **Tonnage Contract** | A procurement contract where the grower commits to delivering a fixed metric tonnage of raw product, shifting field yield risk onto the grower. | Assuming growers can guarantee delivery without price premiums; in disaster years, growers default, triggering secondary spot-market panics. | Mass (metric tonnes, $\text{t}$). |
| **Quality Price Grid** | A contractual settlement schedule specifying financial bonuses or penalties applied to the base raw material price based on laboratory-inspected attributes (Brix, defects, mold, foreign matter). | Designing price grids where the marginal bonus for higher Brix is lower than the marginal revenue from delaying harvest to gain water weight. | Currency per unit mass ($\text{\$/t}$). |

---

## 5. The mechanics, step by step

Coordinating agricultural harvesting and factory processing requires a five-stage operational procedure spanning agrometeorological modeling, phenology tracking, yield response calculation, mixed-integer optimization, and contract settlement.

```text
  Stage 1: Thermal Time & Phenology Accumulation
     │     Compute daily GDD from temperature feeds; project physiological maturity.
     ▼
  Stage 2: Water Balance & Yield Response Modeling
     │     Calculate reference ET_0, crop ET_c, and soil moisture stress yield penalties.
     ▼
  Stage 3: Harvest Window & Quality Degradation Mapping
     │     Establish [ES_i, LF_i]; calculate brix, acidity, and mass penalty vectors.
     ▼
  Stage 4: Mixed-Integer Linear Programming (MILP) Dispatch
     │     Solve field-to-day assignments under harvester, roading, and plant intake constraints.
     ▼
  Stage 5: Logistics Execution & Contract Settlement
           Monitor dwell time, perform intake grading lab tests, settle quality price grid.
```

---

### Step 1: Agrometeorological Tracking and Thermal Time (GDD) Calculation

Plants do not operate on Greenwich Mean Time; they operate on thermal time. The developmental rate of a crop between planting and maturity is an integrated function of ambient temperature.

#### 1.1 Standard GDD Formulations
The base formulation for daily Growing Degree Days ($GDD_t$) on calendar day $t$ is calculated from daily maximum temperature ($T_{\max, t}$) and daily minimum temperature ($T_{\min, t}$):

$$T_{\text{avg}, t} = \frac{T_{\max, t} + T_{\min, t}}{2}$$

$$GDD_t = \max\left(0, T_{\text{avg}, t} - T_{\text{base}}\right)$$

Where $T_{\text{base}}$ is the crop-specific physiological threshold below which vegetative and reproductive development halts.

#### 1.2 Cutoff Methods: The Modified Sine and Single-Triangle Approximations
In production agriculture, the standard averaging method produces severe errors when temperatures fluctuate across $T_{\text{base}}$ or exceed the upper developmental threshold $T_{\text{upper}}$. Under extreme heat ($T > T_{\text{upper}}$), enzymes denature, respiration outpaces photosynthesis, and developmental accumulation plateaus. 

The **Single-Triangle Method with Horizontal Cutoff** (or Modified GDD Method) enforces upper and lower biological bounds:

$$T_{\max, t}^* = \min\left(T_{\max, t}, T_{\text{upper}}\right)$$

$$T_{\min, t}^* = \max\left(T_{\min, t}, T_{\text{base}}\right)$$

$$GDD_t = \max\left(0, \frac{T_{\max, t}^* + T_{\min, t}^*}{2} - T_{\text{base}}\right)$$

*Common agronomic reference thresholds:*
- Processing Tomatoes: $T_{\text{base}} = 10.0^\circ\text{C}$, $T_{\text{upper}} = 30.0^\circ\text{C}$.
- Sweet Corn: $T_{\text{base}} = 10.0^\circ\text{C}$, $T_{\text{upper}} = 30.0^\circ\text{C}$ (Standard 50/86 system in $^{\circ}\text{F}$).
- Green Peas: $T_{\text{base}} = 4.4^\circ\text{C}$, $T_{\text{upper}} = 24.0^\circ\text{C}$.
- Sugarcane: $T_{\text{base}} = 18.0^\circ\text{C}$, $T_{\text{upper}} = 34.0^\circ\text{C}$.
- Wine Grapes (*Vitis vinifera*): $T_{\text{base}} = 10.0^\circ\text{C}$, $T_{\text{upper}} = 35.0^\circ\text{C}$ (Winkler Index).

#### 1.3 Phenological Stage Projection
Cumulative thermal time from planting day $t_0$ to current day $T$ is given by:

$$GDD_{\text{cum}}(T) = \sum_{t=t_0}^{T} GDD_t$$

Each crop cultivar requires an empirically validated thermal sum ($GDD_{\text{req}}$) to transition through developmental stages (e.g., emergence $\to$ flowering $\to$ fruit set $\to$ veraison $\to$ physiological maturity). To forecast the maturity date $t_{\text{mat}}$ under operational conditions:

$$t_{\text{mat}} = \min \left\{ \tau \;\middle|\; GDD_{\text{cum}}(t_{\text{now}}) + \sum_{s=t_{\text{now}}+1}^{\tau} \widehat{GDD}_s \ge GDD_{\text{req}} \right\}$$

Where $\widehat{GDD}_s$ is computed from a blend of 14-day numerical weather predictions (NWP) and 30-year historical climatological averages.

---

### Step 2: Biophysical Crop Yield and Soil Moisture Stress Modeling

Thermal time predicts *when* a crop matures; soil hydrology and energy balance dictate *how much* biomass and yield are harvested.

#### 2.1 The FAO-56 Penman-Monteith Evapotranspiration Architecture
To determine whether a crop experiences moisture stress, the system calculates reference evapotranspiration ($ET_0$) representing an idealized grass canopy using the FAO-56 Penman-Monteith equation:

$$ET_0 = \frac{0.408 \Delta (R_n - G) + \gamma \frac{900}{T_{\text{mean}} + 273} u_2 (e_s - e_a)}{\Delta + \gamma (1 + 0.34 u_2)}$$

Where:
- $R_n$ = Net radiation at the crop surface ($\text{MJ}/(\text{m}^2\cdot\text{day})$).
- $G$ = Soil heat flux density ($\text{MJ}/(\text{m}^2\cdot\text{day})$).
- $T_{\text{mean}}$ = Mean daily air temperature at $2\text{ m}$ height ($^\circ\text{C}$).
- $u_2$ = Wind speed at $2\text{ m}$ height ($\text{m/s}$).
- $e_s - e_a$ = Saturation vapor pressure deficit of air ($\text{kPa}$).
- $\Delta$ = Slope of the vapor pressure curve ($\text{kPa}/^\circ\text{C}$).
- $\gamma$ = Psychrometric constant ($\text{kPa}/^\circ\text{C}$).

#### 2.2 Crop Water Requirements and Soil Water Depletion
Crop evapotranspiration under unstressed conditions ($ET_c$) is determined using the dual crop coefficient:

$$ET_c = \left( K_{cb} + K_e \right) ET_0$$

Where $K_{cb}$ is the basal crop coefficient representing plant transpiration and $K_e$ is the soil evaporation coefficient. 

Daily root zone depletion ($D_{r, t}$) is tracked via a daily soil water balance:

$$D_{r, t} = D_{r, t-1} - (P_t - RO_t) - I_t - CR_t + ET_{a, t} + DP_t$$

Where $P_t$ is precipitation, $RO_t$ is runoff, $I_t$ is net irrigation depth, $CR_t$ is capillary rise from groundwater, $ET_{a, t}$ is actual evapotranspiration, and $DP_t$ is deep percolation below the root zone.

When root zone depletion exceeds the readily available water threshold ($D_{r, t} > RAW = p \cdot TAW$, where $TAW$ is total available soil water and $p$ is the soil water depletion fraction), the plant closes its stomata. Actual crop evapotranspiration drops below potential:

$$ET_{a, t} = K_s \cdot K_{cb} \cdot ET_0 + K_e \cdot ET_0$$

Where the water stress coefficient $K_s$ is formulated as:

$$K_s = \begin{cases} 1.0, & D_{r, t} \le RAW \\ \frac{TAW - D_{r, t}}{TAW - RAW} = \frac{TAW - D_{r, t}}{(1 - p) TAW}, & RAW < D_{r, t} \le TAW \\ 0.0, & D_{r, t} > TAW \end{cases}$$

#### 2.3 The Yield Response to Water Stress
Yield loss resulting from cumulative vegetative or reproductive moisture stress is evaluated using the empirical Stewart-Doorenbos-Kassam yield response model:

$$\left(1 - \frac{Y_a}{Y_m}\right) = \sum_{k=1}^{K} K_{y, k} \left(1 - \frac{ET_{a, k}}{ET_{c, k}}\right)$$

Where $Y_a$ is actual harvest yield ($\text{t/ha}$), $Y_m$ is maximum potential yield under zero water stress, $K_{y, k}$ is the yield sensitivity factor for phenological growth stage $k$, and $ET_{a, k} / ET_{c, k}$ is the relative water consumption ratio in stage $k$. For processing tomatoes, water stress during flowering and fruit development ($K_y \approx 1.1$) results in severe yield abortion, whereas deliberate moderate water stress during the ripening stage ($K_y \approx 0.4$) concentrates soluble sugars ($^\circ\text{Brix}$) without sacrificing substantial fruit mass.

---

### Step 3: Determining the Harvest Window and Quality Degradation

A field parcel does not mature on a single knife-edge hour. It presents a dynamic **Harvest Window** $[ES_i, LF_i]$, bounded by the Earliest Start date ($ES_i$) and Latest Finish date ($LF_i$).

```text
       Yield / Quality
             ▲
             │                       Optimum Maturity Day t*
             │                                  ▼
Peak Brix /  │                             ╭─────────╮
Yield Curve  │                           ╭─╯         ╰─╮
             │                         ╭─╯             ╰─╮   Overripeness,
             │                       ╭─╯                 ╰─╮ Mold, Drop,
             │    Immature Fruit   ╭─╯                     ╰─ Rot Losses
             │    Low Brix / Mass ╭╯                         ╰─╮
             │                   ╭╯                             ╰─
             0───────────────────┴───────────────┼───────────────┴────────► Time (Days)
                                ES_i            t*              LF_i
                                  ◄──────── Harvest Window ─────►
```

#### 3.1 Asymmetric Quality and Mass Penalties
Within the window $[ES_i, LF_i]$, daily yield $Y_{it}$ ($\text{t/ha}$) and quality grade $B_{it}$ ($^\circ\text{Brix}$, Tenderometer, or Sugar Content) vary as a function of harvest day $t$ relative to the optimal maturity day $t_i^*$:

$$Y_{it} = Y_i^* \cdot \left[ 1 - \alpha_i^{\text{early}} \max(0, t_i^* - t) - \alpha_i^{\text{late}} \max(0, t - t_i^*) \right]$$

$$B_{it} = B_i^* + \beta_i^{\text{early}} \min(0, t - t_i^*) - \beta_i^{\text{late}} \max(0, t - t_i^*)$$

Where:
- $Y_i^*, B_i^*$ represent peak potential harvest yield and peak quality at optimal maturity day $t_i^*$.
- $\alpha_i^{\text{early}}$ is the daily mass loss penalty for early harvest (e.g., $0.02$/day due to unpollinated, undersized fruit).
- $\alpha_i^{\text{late}}$ is the daily mass loss penalty for delayed harvest (e.g., $0.035$/day due to fruit drop, vine rot, sunburn, and pest feeding). Notice that $\alpha_i^{\text{late}} > \alpha_i^{\text{early}}$, reflecting the asymmetric biological risk of field overripeness.
- $\beta_i^{\text{early}}, \beta_i^{\text{late}}$ govern quality degradation rates.

---

### Step 4: Harvest Scheduling Optimization (MILP Formulation)

Harvest scheduling across a distributed supply base is an NP-hard combinatorial optimization problem. The operations research model must assign harvesters to fields over a discrete multi-day planning horizon $\mathcal{T} = \{1, 2, \dots, T\}$.

#### 4.1 Sets and Indices
- $i \in \mathcal{I}$: Set of contracted crop parcels / fields.
- $t \in \mathcal{T}$: Set of discrete planning days across the campaign horizon.
- $k \in \mathcal{K}$: Set of mechanical harvester crews / machines.
- $j \in \mathcal{J}$: Set of processing facilities (canneries, mills, packinghouses).

#### 4.2 Parameters
- $A_i$: Surface area of field parcel $i$ (hectares, $\text{ha}$).
- $Y_{it}$: Expected harvest yield of field $i$ on day $t$ ($\text{metric tonnes/ha}$).
- $B_{it}$: Expected quality metric (e.g., $^\circ\text{Brix}$) of field $i$ on day $t$.
- $P_0$: Contracted base purchasing price per tonne ($\text{\$/t}$).
- $\pi(B)$: Contractual price adjustment bonus/penalty per tonne as a function of quality ($\text{\$/t}$).
- $c_k^{\text{harv}}$: Operating cost of harvester crew $k$ per hectare ($\text{\$/ha}$).
- $c_{ij}^{\text{trans}}$: Freight transport cost from field $i$ to plant $j$ per tonne ($\text{\$/t}$).
- $C_{kt}^{\text{harv}}$: Maximum harvesting capacity of machine crew $k$ on day $t$ ($\text{ha/day}$).
- $Cap_{jt}$: Maximum raw material receiving and processing capacity of plant $j$ on day $t$ ($\text{t/day}$).
- $ES_i, LF_i$: Earliest start and latest finish days of harvest window for field $i$.
- $\delta_{it}^{\text{rain}} \in \{0, 1\}$: Binary weather parameter indicating precipitation downtime (1 if rain halts harvest, 0 otherwise).
- $d_{ij}$: Road distance from field $i$ to plant $j$ ($\text{km}$).

#### 4.3 Decision Variables
- $x_{it} \in [0, 1]$: Continuous decision variable representing the fraction of field $i$ harvested on day $t$.
- $y_{it} \in \{0, 1\}$: Binary variable indicating whether harvesting occurs on field $i$ on day $t$.
- $w_{ijt} \ge 0$: Mass of crop (tonnes) harvested from field $i$ and transported to processing plant $j$ on day $t$.
- $u_{ikt} \in \{0, 1\}$: Binary indicator assigning harvester crew $k$ to field $i$ on day $t$.

#### 4.4 Objective Function: Maximize Net Delivered Agro-Industrial Margin
The objective function maximizes total delivered product value net of harvesting operating expenses, road transportation logistics costs, and quality penalty deviations:

$$\max \mathcal{Z} = \sum_{t \in \mathcal{T}} \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} w_{ijt} \left[ P_0 + \pi(B_{it}) \right] - \sum_{t \in \mathcal{T}} \sum_{i \in \mathcal{I}} \sum_{k \in \mathcal{K}} A_i x_{it} c_k^{\text{harv}} - \sum_{t \in \mathcal{T}} \sum_{i \in \mathcal{I}} \sum_{j \in \mathcal{J}} w_{ijt} c_{ij}^{\text{trans}}$$

#### 4.5 System Constraints

1. **Complete Field Harvest Requirement:**
   Every contracted field parcel must be fully harvested within its viable physiological window $[ES_i, LF_i]$:

   $$\sum_{t = ES_i}^{LF_i} x_{it} = 1, \quad \forall i \in \mathcal{I}$$

   $$x_{it} = 0, \quad \forall t < ES_i \quad \text{and} \quad \forall t > LF_i, \quad \forall i \in \mathcal{I}$$

2. **Crop Mass Balance and Delivery Flow:**
   Total tonnes harvested from field $i$ on day $t$ must equal the sum of deliveries to all receiving processing plants:

   $$\sum_{j \in \mathcal{J}} w_{ijt} = A_i \cdot Y_{it} \cdot x_{it}, \quad \forall i \in \mathcal{I}, \quad \forall t \in \mathcal{T}$$

3. **Harvester Fleet Daily Work Capacity:**
   The total area harvested across all assigned fields by crew $k$ on day $t$ cannot exceed machine mechanical capacity:

   $$\sum_{i \in \mathcal{I}} A_i \cdot x_{it} \cdot u_{ikt} \le C_{kt}^{\text{harv}} \cdot \left( 1 - \delta_{it}^{\text{rain}} \right), \quad \forall k \in \mathcal{K}, \quad \forall t \in \mathcal{T}$$

   In linear form, introducing continuous variable $z_{ikt} = x_{it} u_{ikt}$ with standard Big-$M$ linking constraints:

   $$\sum_{i \in \mathcal{I}} A_i z_{ikt} \le C_{kt}^{\text{harv}} \left( 1 - \delta_{it}^{\text{rain}} \right), \quad \forall k, t$$

4. **Processing Plant Daily Intake Bottleneck:**
   The cumulative raw product arriving at processing plant $j$ on day $t$ cannot exceed the plant's fluming, washing, and processing intake bottleneck:

   $$\sum_{i \in \mathcal{I}} w_{ijt} \le Cap_{jt}, \quad \forall j \in \mathcal{J}, \quad \forall t \in \mathcal{T}$$

5. **Non-Splittable Field Operational Integrity:**
   To prevent harvesters from inefficiently splitting small parcels across non-consecutive days, enforce parcel indivisibility where required:

   $$x_{it} \le y_{it}, \quad \forall i \in \mathcal{I}, \quad \forall t \in \mathcal{T}$$

   $$\sum_{t \in \mathcal{T}} y_{it} \le \Omega_i, \quad \forall i \in \mathcal{I}$$

   Where $\Omega_i$ is the maximum permitted number of harvest days allowed for parcel $i$ (e.g., $\Omega_i = 1$ for small fields under $10\text{ ha}$; $\Omega_i = 2$ for medium fields).

6. **Spatial Machine Routing and Relocation Costs:**
   When harvester crew $k$ transitions from parcel $i$ on day $t-1$ to parcel $i'$ on day $t$, it incurs an inter-field roading distance $D_{ii'}$ and mobilization setup cost $c^{\text{mob}}_{ii'}$. Introducing binary transition variables $v_{ii'kt} \in \{0, 1\}$:

   $$v_{ii'kt} \ge u_{ik, t-1} + u_{i'kt} - 1, \quad \forall i, i' \in \mathcal{I}, \quad \forall k \in \mathcal{K}, \quad \forall t \ge 2$$

   The relocation expense $\sum_{t \ge 2} \sum_{i} \sum_{i'} \sum_{k} c^{\text{mob}}_{ii'} v_{ii'kt}$ is subtracted from the objective function $\mathcal{Z}$, penalizing spatial dispersion and naturally forming geographic harvesting clusters.

#### 4.6 Dynamic Rolling-Horizon Re-optimization under Weather Stochasticity
In industrial practice, harvest dispatchers do not execute a static 90-day campaign schedule. Instead, they operate a **Rolling-Horizon Decision Architecture** with daily re-optimization:
- **Frozen Horizon ($\tau \in [t, t+1]$):** Machine and crew allocations are physically locked. Harvesters are positioned at field headlands; transport convoys are committed.
- **Tactical Planning Horizon ($\tau \in [t+2, t+7]$):** Driven by high-resolution 7-day numerical weather predictions (NWP) and observed GDD accumulation. Field harvest windows and sequence orders are optimized daily.
- **Strategic Campaign Horizon ($\tau \in [t+8, t+30]$):** Driven by 30-year climatological GDD projections to balance regional factory weekly throughput and identify impending intake bottlenecks weeks in advance.

When incoming Doppler radar detects an imminent convective storm ($> 15\text{ mm}$ rain), the system instantly sets $\delta_{it'}^{\text{rain}} = 1$ for affected parcels $i'$, re-solves the MILP within 120 seconds, and reroutes harvester crews to well-drained sandy parcels on the dry periphery of the supply shed.

---

### Step 5: Post-Harvest Transit Logistics and Cut-to-Process Dwell Limits

The physical moment of harvest severs the crop from plant vascular support. Once harvested, the crop enters an irreversible degradation phase governed by respiration kinetics.

#### 5.1 Respiration and Self-Heating Kinetics
Plant tissue continues respiration in transport trailers:

$$R_{\text{heat}} = R_0 \cdot Q_{10}^{\frac{T_{\text{ambient}} - T_{\text{ref}}}{10}}$$

Where $R_0$ is the baseline respiration rate at reference temperature $T_{\text{ref}}$ ($0^\circ\text{C}$), and $Q_{10} \approx 2.0 - 2.5$ represents the temperature sensitivity quotient. In open-top bulk trailers carrying 25 tonnes of produce under summer conditions ($35^\circ\text{C}$), internal trailer temperatures rise at $1.5^\circ\text{C}$ to $3.0^\circ\text{C}$ per hour, accelerating enzymatic breakdown.

#### 5.2 The Cut-to-Process Constraint
For perishable crops (sugarcane, sweet corn, wine grapes, canning tomatoes), total elapsed time between field severance and factory unloading must strictly respect the critical threshold $\tau_{\max}$:

$$\tau_{\text{dwell}} = t_{\text{harvest}} + t_{\text{field\_queue}} + t_{\text{road\_transit}} + t_{\text{plant\_queue}} \le \tau_{\max}$$

*Critical industry dwell thresholds:*
- Sweet Corn: $\tau_{\max} \le 4.0\text{ hours}$ (to prevent $50\%$ sucrose conversion to starch).
- Sugarcane: $\tau_{\max} \le 18.0\text{ hours}$ (to prevent severe microbial leuconostoc and dextran formation).
- Processing Tomatoes: $\tau_{\max} \le 12.0\text{ hours}$ (to prevent mold growth and lactic acid souring).
- Wine Grapes (Premium): $\tau_{\max} \le 4.0\text{ hours}$ (or mandatory nocturnal mechanical harvest between 02:00 and 08:00 at $12^\circ\text{C}$ pulp temperature to avoid polyphenol oxidation).

#### 5.3 Infield Transfer Logistics: Chaser Bins and Cyclic Queueing
Mechanical combine harvesters operate at peak fuel efficiency and minimum crop damage when unloading continuously on-the-go into tractor-towed infield haulers (known as **chaser bins**, or *transbordos* in sugarcane).

If a harvester’s internal holding bin fills while no chaser bin is positioned alongside, the combine must halt threshing in the furrow. This dead time reduces net harvesting speed from $4.5\text{ ha/h}$ to under $2.8\text{ ha/h}$.

The infield haulage cycle forms a closed-loop cyclic queue:

$$T_{\text{cycle}} = t_{\text{load}} + t_{\text{haul\_infield}} + t_{\text{dump\_roadside}} + t_{\text{return\_infield}} + t_{\text{queue}}$$

To prevent combine starvation, the ratio of chaser bin tractors ($N_{\text{chaser}}$) to combines ($N_{\text{harv}}$) is sized according to cyclic queueing theory:

$$N_{\text{chaser}} \ge \left\lceil \frac{T_{\text{cycle}}}{t_{\text{load}}} \cdot N_{\text{harv}} \right\rceil$$

Infield haulers transfer crop into 25-tonne highway semi-trailers staged on paved field headlands, preventing heavy road trucks from entering moist furrows and causing catastrophic subsoil compaction.

---

### Step 6: Grower Contracting and Risk Sharing Alignment

A mathematically sound harvest schedule will fail in practice if the commercial contracts incentivize growers to subvert the schedule.

#### 6.1 Contract Architectures
1. **Gross Tonnage Contract:** Processor pays a flat fee per delivered tonne ($P_0 \cdot Y_i$).
   *System Failure:* Grower has zero economic incentive to allow harvesting during early high-sugar periods. The grower will actively fight harvest dispatchers to delay harvest until maximum water weight is achieved, overwhelming the plant with water and overloading evaporation boilers.
2. **Acreage Contract with Quality Grid:** Processor leases the land output at base price $R_{\text{base}}$ per hectare plus a quality bonus grid linked to laboratory Brix and defect sampling:

   $$\text{Settlement}_i = A_i \cdot \left[ R_{\text{base}} + Y_{it} \cdot \left( P_{\text{base}} + \Delta P_{\text{Brix}}(B_{it}) - \Delta P_{\text{defect}}(D_{it}) \right) \right]$$

This contractual structure removes volume hoarding incentives, allowing the processor's central optimization engine to schedule harvest dates that maximize global supply chain throughput.

#### 6.2 Weather-Index Crop Insurance and Risk-Sharing Pools
When a central processor dictates harvest dates via mathematical optimization, growers face perceived unfairness: Grower D’s field may be deliberately postponed to Day 4 to prevent factory flume overflow, incurring a $3\%$ yield drop through no fault of their own.

To eliminate grower resistance, world-class agro-industrial processors deploy **Cooperative Equalization Pools**:
- **Theoretical Optimum Benchmark:** The processor calculates the theoretical maximum unconstrained value of each field $V_i^*$ at its biological peak date $t_i^*$.
- **Intervention Adjustment:** If the central optimizer shifts field $i$ away from $t_i^*$ to resolve factory bottlenecks, the processor credits the grower with the baseline peak margin $V_i^*$, absorbing the shift penalty $\Delta V_i$ as a plant-level operating overhead.
- **Parametric Weather Derivatives:** Processors and growers co-purchase parametric rainfall insurance. If cumulative rain exceeds $35\text{ mm}$ during the contracted harvest window, satellite and rain-gauge triggers execute automated payout settlements without requiring lengthy field-loss adjustor inspections.

---

## 6. Worked example

Let us execute an end-to-end numerical calculation for an agro-industrial processing facility coordinating the harvest of four commercial processing tomato fields over a 4-day operational planning horizon ($t \in \{1, 2, 3, 4\}$).

### 6.1 Baseline Parameters and Data

**Processing Cannery Specifications:**
- Processing intake bottleneck capacity: $Cap_t = 320.0\text{ metric tonnes/day}$.
- Base raw product price: $P_0 = \$110.00\text{ per metric tonne}$.
- Quality price adjustment: Premium/penalty of $\pi = +\$12.00\text{ / (t}\cdot^\circ\text{Bx)}$ for every degree Brix above $5.0^\circ\text{Bx}$ (or penalty if below $5.0^\circ\text{Bx}$).
- Harvester fleet: 1 dedicated mechanical combine crew with a maximum harvesting capacity of $C_t^{\text{harv}} = 4.0\text{ ha/day}$.
- Harvesting operating cost: $c^{\text{harv}} = \$350.00\text{ / ha}$.
- Transportation freight cost: $c^{\text{trans}} = \$0.25\text{ per tonne-kilometer (\$/t}\cdot\text{km)}$.

**Field Parcel Register:**

| Parcel ID | Area $A_i$ (ha) | Optimal Maturity Day $t^*$ | Base Yield $Y_i^*$ (t/ha) | Base Quality $B_i^*$ ($^\circ\text{Bx}$) | Haul Distance $d_i$ (km) |
|---|---:|---:|---:|---:|---:|
| **Field A** | 3.0 ha | Day 1 | 85.0 t/ha | 5.4 $^\circ\text{Bx}$ | 15 km |
| **Field B** | 3.5 ha | Day 2 | 80.0 t/ha | 5.2 $^\circ\text{Bx}$ | 25 km |
| **Field C** | 4.0 ha | Day 3 | 75.0 t/ha | 5.5 $^\circ\text{Bx}$ | 10 km |
| **Field D** | 3.0 ha | Day 3 | 90.0 t/ha | 5.3 $^\circ\text{Bx}$ | 30 km |

Notice the operational conflict: **Both Field C and Field D reach peak maturity on Day 3.**

### 6.2 Biological Degradation Functions
If a field is harvested away from its optimal day ($t \ne t^*$):
- Early harvest ($t < t^*$): Yield drops by $2.0\%$ per day early due to immature fruit ($\alpha^{\text{early}} = 0.02$). Brix drops by $0.10^\circ\text{Bx}$ per day early ($\beta^{\text{early}} = 0.10$).
- Delayed harvest ($t > t^*$): Yield drops by $3.0\%$ per day late due to fruit softening, drop, and rot ($\alpha^{\text{late}} = 0.03$). Brix drops by $0.10^\circ\text{Bx}$ per day late ($\beta^{\text{late}} = 0.10$).

Let us construct the complete state matrix for each candidate harvest day:

#### Calculations for Field A ($A_A = 3.0\text{ ha}, t^* = 1$):
- **Day 1 (Optimal):**
  - Yield: $85.0\text{ t/ha}$. Total tonnage: $3.0 \times 85.0 = 255.0\text{ t}$.
  - Brix: $5.40^\circ\text{Bx}$. Price: $\$110.00 + 12 \times (5.40 - 5.0) = \$114.80\text{ / t}$.
  - Gross Revenue: $255.0 \times 114.80 = \$29,274.00$.
  - Harvester Cost: $3.0 \times 350 = \$1,050.00$.
  - Transport Cost: $255.0\text{ t} \times 15\text{ km} \times \$0.25 = \$956.25$.
  - **Net Margin:** $\$29,274.00 - \$1,050.00 - \$956.25 = \$27,267.75$.

- **Day 2 (1 Day Late):**
  - Yield: $85.0 \times (1 - 0.03) = 82.45\text{ t/ha}$. Tonnage: $3.0 \times 82.45 = 247.35\text{ t}$.
  - Brix: $5.40 - 0.10 = 5.30^\circ\text{Bx}$. Price: $\$110.00 + 12 \times (5.30 - 5.0) = \$113.60\text{ / t}$.
  - Gross Revenue: $247.35 \times 113.60 = \$28,098.96$.
  - Transport: $247.35 \times 15 \times 0.25 = \$927.56$. Harvester: $\$1,050.00$.
  - **Net Margin:** $\$28,098.96 - \$1,050.00 - \$927.56 = \$26,121.40$.

#### Calculations for Field B ($A_B = 3.5\text{ ha}, t^* = 2$):
- **Day 1 (1 Day Early):**
  - Yield: $80.0 \times (1 - 0.02) = 78.40\text{ t/ha}$. Tonnage: $3.5 \times 78.40 = 274.40\text{ t}$.
  - Brix: $5.20 - 0.10 = 5.10^\circ\text{Bx}$. Price: $\$110.00 + 12 \times (5.10 - 5.0) = \$111.20\text{ / t}$.
  - Gross Revenue: $274.40 \times 111.20 = \$30,513.28$.
  - Transport: $274.40 \times 25 \times 0.25 = \$1,715.00$. Harvester: $3.5 \times 350 = \$1,225.00$.
  - **Net Margin:** $\$30,513.28 - \$1,225.00 - \$1,715.00 = \$27,573.28$.

- **Day 2 (Optimal):**
  - Yield: $80.0\text{ t/ha}$. Tonnage: $3.5 \times 80.0 = 280.0\text{ t}$.
  - Brix: $5.20^\circ\text{Bx}$. Price: $\$110.00 + 12 \times (5.20 - 5.0) = \$112.40\text{ / t}$.
  - Gross Revenue: $280.0 \times 112.40 = \$31,472.00$.
  - Transport: $280.0 \times 25 \times 0.25 = \$1,750.00$. Harvester: $\$1,225.00$.
  - **Net Margin:** $\$31,472.00 - \$1,225.00 - \$1,750.00 = \$28,497.00$.

#### Calculations for Field C ($A_C = 4.0\text{ ha}, t^* = 3$):
- **Day 2 (1 Day Early):**
  - Yield: $75.0 \times (1 - 0.02) = 73.50\text{ t/ha}$. Tonnage: $4.0 \times 73.50 = 294.0\text{ t}$.
  - Brix: $5.50 - 0.10 = 5.40^\circ\text{Bx}$. Price: $\$110.00 + 12 \times (5.40 - 5.0) = \$114.80\text{ / t}$.
  - Gross Revenue: $294.0 \times 114.80 = \$33,751.20$.
  - Transport: $294.0 \times 10 \times 0.25 = \$735.00$. Harvester: $4.0 \times 350 = \$1,400.00$.
  - **Net Margin:** $\$33,751.20 - \$1,400.00 - \$735.00 = \$31,616.20$.

- **Day 3 (Optimal):**
  - Yield: $75.0\text{ t/ha}$. Tonnage: $4.0 \times 75.0 = 300.0\text{ t}$.
  - Brix: $5.50^\circ\text{Bx}$. Price: $\$110.00 + 12 \times (5.50 - 5.0) = \$116.00\text{ / t}$.
  - Gross Revenue: $300.0 \times 116.00 = \$34,800.00$.
  - Transport: $300.0 \times 10 \times 0.25 = \$750.00$. Harvester: $\$1,400.00$.
  - **Net Margin:** $\$34,800.00 - \$1,400.00 - \$750.00 = \$32,650.00$.

#### Calculations for Field D ($A_D = 3.0\text{ ha}, t^* = 3$):
- **Day 3 (Optimal):**
  - Yield: $90.0\text{ t/ha}$. Tonnage: $3.0 \times 90.0 = 270.0\text{ t}$.
  - Brix: $5.30^\circ\text{Bx}$. Price: $\$110.00 + 12 \times (5.30 - 5.0) = \$113.60\text{ / t}$.
  - Gross Revenue: $270.0 \times 113.60 = \$30,672.00$.
  - Transport: $270.0 \times 30 \times 0.25 = \$2,025.00$. Harvester: $3.0 \times 350 = \$1,050.00$.
  - **Net Margin:** $\$30,672.00 - \$1,050.00 - \$2,025.00 = \$27,597.00$.

- **Day 4 (1 Day Late):**
  - Yield: $90.0 \times (1 - 0.03) = 87.30\text{ t/ha}$. Tonnage: $3.0 \times 87.30 = 261.90\text{ t}$.
  - Brix: $5.30 - 0.10 = 5.20^\circ\text{Bx}$. Price: $\$110.00 + 12 \times (5.20 - 5.0) = \$112.40\text{ / t}$.
  - Gross Revenue: $261.90 \times 112.40 = \$29,437.56$.
  - Transport: $261.90 \times 30 \times 0.25 = \$1,964.25$. Harvester: $\$1,050.00$.
  - **Net Margin:** $\$29,437.56 - \$1,050.00 - \$1,964.25 = \$26,423.31$.

---

### 6.3 Evaluating the Operational Bottleneck and Resolving the Conflict

If every grower harvested on their optimal maturity day:
- Day 1: Harvest Field A $\to 255.0\text{ tonnes} \le 320.0\text{ tonnes}$ [Feasible].
- Day 2: Harvest Field B $\to 280.0\text{ tonnes} \le 320.0\text{ tonnes}$ [Feasible].
- Day 3: Harvest Field C and Field D simultaneously:
  - Total Area: $4.0\text{ ha} + 3.0\text{ ha} = 7.0\text{ ha} > 4.0\text{ ha}$ **[HARVESTER CAPACITY VIOLATED]**.
  - Total Tonnage: $300.0\text{ t} + 270.0\text{ t} = 570.0\text{ tonnes} > 320.0\text{ tonnes}$ **[PLANT INTAKE BOTTLENECK VIOLATED by 250 tonnes]**.

The system must reschedule either Field C or Field D to another day.

**Candidate Schedule Option 1:** Shift Field C to Day 2, Shift Field B to Day 1, Harvest Field A on Day 1?
- Day 1: Field A ($3.0\text{ ha}$) + Field B ($3.5\text{ ha}$) = $6.5\text{ ha} > 4.0\text{ ha}$ [Infeasible for 1 harvester].

**Candidate Schedule Option 2:** Harvest Field A on Day 1, Field B on Day 2, Field C on Day 3, Shift Field D to Day 4:
- Day 1: Field A ($3.0\text{ ha} \le 4.0\text{ ha}$; $255.0\text{ t} \le 320.0\text{ t}$). Net Margin: $\$27,267.75$.
- Day 2: Field B ($3.5\text{ ha} \le 4.0\text{ ha}$; $280.0\text{ t} \le 320.0\text{ t}$). Net Margin: $\$28,497.00$.
- Day 3: Field C ($4.0\text{ ha} \le 4.0\text{ ha}$; $300.0\text{ t} \le 320.0\text{ t}$). Net Margin: $\$32,650.00$.
- Day 4: Field D ($3.0\text{ ha} \le 4.0\text{ ha}$; $261.9\text{ t} \le 320.0\text{ t}$). Net Margin: $\$26,423.31$.
- **Total System Net Margin:** 
  $$\$27,267.75 + \$28,497.00 + \$32,650.00 + \$26,423.31 = \$114,838.06$$
- Penalty of shifting Field D by 1 day: $\$27,597.00 - \$26,423.31 = \$1,173.69$.

**Candidate Schedule Option 3:** Harvest Field A on Day 1, Field B on Day 2, Field D on Day 3, Shift Field C to Day 4:
- Shifting Field C ($4.0\text{ ha}$) to Day 4 yields:
  - Yield: $75.0 \times (1 - 0.03) = 72.75\text{ t/ha}$. Total: $291.0\text{ t}$.
  - Brix: $5.50 - 0.10 = 5.40^\circ\text{Bx}$. Price: $\$114.80\text{ / t}$.
  - Gross Revenue: $291.0 \times 114.80 = \$33,406.80$.
  - Transport: $291.0 \times 10 \times 0.25 = \$727.50$. Harvester: $\$1,400.00$.
  - Net Margin on Day 4: $\$33,406.80 - \$1,400.00 - \$727.50 = \$31,279.30$.
  - Penalty of shifting Field C: $\$32,650.00 - \$31,279.30 = \$1,370.70$.
- **Total System Net Margin (Option 3):**
  $$\$27,267.75 + \$28,497.00 + \$27,597.00 + \$31,279.30 = \$114,641.05$$

**Mathematical Conclusion:** 
Option 2 delivers **$\$114,838.06$**, outperforming Option 3 by **$\$197.01$**. The optimization engine correctly schedules the larger yield penalty parcel (Field C, 4.0 ha, penalty $\$1,370.70$) on its peak day and shifts the smaller parcel (Field D, 3.0 ha, penalty $\$1,173.69$) by one day.

---

## 7. Code, query, or calculation

The following production-ready Python script implements the mathematical harvest scheduling model. It takes field parameters, degradation curves, harvester capacities, and processing plant intake limits, evaluates all feasible assignments, and identifies the globally optimal schedule.

```python
"""
Multi-Field Harvest Scheduling Optimizer with Intake Bottlenecks.
Solves parcel assignment over a discrete planning horizon under machinery
capacity and factory intake limits, maximizing net agro-industrial margin.
"""
import itertools

fields = [
    {"name": "Field_A", "area_ha": 3.0, "base_yield": 85.0, "opt_day": 1, "brix_opt": 5.4, "dist_km": 15},
    {"name": "Field_B", "area_ha": 3.5, "base_yield": 80.0, "opt_day": 2, "brix_opt": 5.2, "dist_km": 25},
    {"name": "Field_C", "area_ha": 4.0, "base_yield": 75.0, "opt_day": 3, "brix_opt": 5.5, "dist_km": 10},
    {"name": "Field_D", "area_ha": 3.0, "base_yield": 90.0, "opt_day": 3, "brix_opt": 5.3, "dist_km": 30},
]
days = [1, 2, 3, 4]
plant_cap_tonnes = 320.0
harv_cap_ha = 4.0
base_price = 110.0
brix_premium = 12.0
trans_cost_per_tkm = 0.25
harv_cost_per_ha = 350.0

def evaluate_schedule(assignment):
    daily_tonnes = {d: 0.0 for d in days}
    daily_ha = {d: 0.0 for d in days}
    total_margin = 0.0

    for f_idx, d in enumerate(assignment):
        f = fields[f_idx]
        delta_d = d - f["opt_day"]
        # Apply asymmetric yield loss: 2% per day early, 3% per day late
        yield_factor = 1.0 - (0.02 * abs(delta_d) if delta_d < 0 else 0.03 * delta_d)
        actual_yield = f["base_yield"] * yield_factor
        tonnes = f["area_ha"] * actual_yield
        
        daily_tonnes[d] += tonnes
        daily_ha[d] += f["area_ha"]
        
        # Enforce machinery and factory bottleneck constraints
        if daily_tonnes[d] > plant_cap_tonnes or daily_ha[d] > harv_cap_ha:
            return -float("inf"), None, None

        # Quality price grid settlement
        actual_brix = f["brix_opt"] - 0.10 * abs(delta_d)
        price_per_t = base_price + brix_premium * (actual_brix - 5.0)
        revenue = tonnes * price_per_t
        costs = (f["area_ha"] * harv_cost_per_ha) + (tonnes * f["dist_km"] * trans_cost_per_tkm)
        total_margin += (revenue - costs)

    return total_margin, daily_tonnes, daily_ha

best_margin = -float("inf")
best_sched, best_tonnes, best_ha = None, None, None

for p in itertools.product(days, repeat=len(fields)):
    margin, d_tonnes, d_ha = evaluate_schedule(p)
    if margin > best_margin:
        best_margin, best_sched = margin, p
        best_tonnes, best_ha = d_tonnes, d_ha

print(f"Optimal Schedule (Field A, B, C, D): {best_sched}")
print(f"Delivered Net Margin: ${best_margin:,.2f}")
for d in days:
    print(f"Day {d}: Harvested {best_ha[d]:.1f} ha, Delivered {best_tonnes[d]:.1f} t (Cap: {plant_cap_tonnes:.1f} t)")
```

### 7.1 Assumptions of the Script
1. Parcel harvesting is non-splittable ($\Omega_i = 1$); each field is harvested entirely within a single operational day.
2. Machine transit time between fields on the same day is negligible, bounded by the aggregate daily area limit ($4.0\text{ ha/day}$).
3. Daily intake and transport costs are linear within capacity limits.

### 7.2 Expected Output
```text
Optimal Schedule (Field A, B, C, D): (1, 2, 3, 4)
Delivered Net Margin: $114,838.06
Day 1: Harvested 3.0 ha, Delivered 255.0 t (Cap: 320.0 t)
Day 2: Harvested 3.5 ha, Delivered 280.0 t (Cap: 320.0 t)
Day 3: Harvested 4.0 ha, Delivered 300.0 t (Cap: 320.0 t)
Day 4: Harvested 3.0 ha, Delivered 261.9 t (Cap: 320.0 t)
```

---

## 8. Data, standards, and regulatory context

Agricultural harvesting and intake planning must comply with statutory food safety regulations, digital data interchange standards, and agrometeorological monitoring architectures.

### 8.1 Statutory Food Safety: US FDA FSMA Rule 204
The Food Safety Modernization Act (FSMA) Section 204 (Final Rule on Requirements for Additional Traceability Records for Certain Foods) mandates strict traceability for items on the Food Traceability List (FTL), which includes fresh leafy greens, melons, tomatoes, cucumbers, and fresh-cut produce.

Harvest operations represent a statutory **Critical Tracking Event (CTE)** designated as **Harvesting**:
- **Key Data Elements (KDEs) Required at Harvest:**
  1. *Location Identifier and Description of Field:* Specific GPS centroid coordinates or USDA Farm Service Agency (FSA) tract/field identifier.
  2. *Commodity and Variety Description:* Taxonomic and commercial cultivar name.
  3. *Quantity and Unit of Measure:* Gross bin/tote count and total harvested mass (metric tonnes or pounds).
  4. *Date of Harvesting:* Timestamp of severance from soil/vine.
  5. *Traceability Lot Code (TLC):* Unique identifier generated at harvest establishing the parent lot.
  6. *Grower Identity:* Legal business name and Global Location Number (GLN).
  7. *Cooling / Transportation Handoff Log:* Timestamp of post-harvest temperature control initiation.

Under FSMA 204, records must be maintainable in an electronic, sortable spreadsheet format and producible to FDA investigators within **24 hours** of an official request during foodborne illness outbreak investigations (e.g., *Escherichia coli* O157:H7 or *Salmonella enterica* contamination events).

#### Table 8.1: FSMA Rule 204 Traceability Architecture for Agricultural Harvesting
| Critical Tracking Event (CTE) | Mandated Key Data Elements (KDEs) | Responsible Actor | Retention & Audit Window |
|---|---|---|---|
| **Harvesting CTE** | - Field GPS Centroid & FSA Tract ID<br>- Commodity Name & Botanical Cultivar<br>- Harvest Timestamp (Date & Hour)<br>- Gross Harvested Quantity & Unit<br>- Grower GLN & Legal Entity Name<br>- Harvester Business GLN<br>- Assigned Traceability Lot Code (TLC) | Harvesting Contractor or Grower | 24-month electronic archive; 24-hour sortable disclosure upon FDA Form 482 request. |
| **Initial Packing CTE** | - TLC linked from Harvest CTE<br>- Packing Facility GLN & Name<br>- Packing Date & Time<br>- GTIN assigned to Finished Case/Tote<br>- Container Physical Spec & Pallet SSCC<br>- Post-Harvest Washing Water Sanitizer Log | Initial Packing House or Processing Weighbridge | 24-month electronic archive; bidirectional trace-back / trace-forward query capability. |
| **First Land-Based Receiver** | - Shipper GLN & Transport BOL Number<br>- Traceability Lot Code (TLC)<br>- Receiving Date & Receiving GLN<br>- Core Flesh / Pulp Temperature at Unloading<br>- Immediate Upstream Supplier Identifier | Processing Cannery, Winery, or Sugar Mill | Verification that TLC matches grower field manifests before raw product is commingled in bulk flumes. |

### 8.2 Global Standards: GS1 in Agriculture
To ensure interoperability across ERP, farm management information systems (FMIS), and processing plant weighbridge scales, supply chain systems deploy GS1 identification keys:
- **GS1 Global Location Number (GLN):** Applied at the parcel level (identifying the specific 5-hectare plot), farm gate, and processing weighbridge intake pit.
- **GS1 Global Trade Item Number (GTIN):** Applied to bulk tote bins and processed finished goods (e.g., aseptic 300-gallon tomato paste totes).
- **GS1 Serial Shipping Container Code (SSCC):** Assigned to each field bulk trailer or palletized bin container entering the transport corridor.
- **EPCIS (Electronic Product Code Information Services) 1.2 / 2.0:** XML/JSON-LD standard for publishing CTE events (`ObjectEvent`, `TransformationEvent`) linking farm harvest KDEs to plant processing runs.

### 8.3 Quality Standards: USDA Agricultural Marketing Service (AMS)
Industrial processors purchase produce under binding grade specifications codified in USDA AMS Title 7 Code of Federal Regulations (7 CFR Part 51):
- *United States Standards for Grades of Processing Tomatoes (7 CFR § 51.3410):* Defines minimum soluble solids ($^\circ\text{Brix}$), color score (measured by Hunter Lab or Agtron optical reflectance), mold tolerance ($< 2\%$ by weight for Grade A), green/unripe tolerance ($< 2\%$), and worm/insect damage.
- Inspection certificates issued at independent state-operated weighbridge grading stations (e.g., Processing Tomato Advisory Board in California) serve as the legal basis for quality price grid adjustments.

### 8.4 Agrometeorological Data Feeds and Remote Sensing Infrastructure
Production harvest models consume multi-spectral and meteorological data feeds:
- **Copernicus ERA5-Land & NOAA RTMA:** Reanalysis feeds delivering hourly 2-meter air temperature ($T_a$), dewpoint ($T_d$), solar radiation ($R_s$), and precipitation ($P$) on $9\text{-km}$ to $2.5\text{-km}$ spatial grids for automated GDD and Penman-Monteith $ET_0$ calculation.
- **ESA Sentinel-2 & NASA Landsat 8/9:** Multi-spectral imagery providing $10\text{-meter}$ resolution Normalized Difference Vegetation Index ($\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}}$) and Normalized Difference Water Index ($\text{NDWI}$). Sudden Drops in NDWI indicate canopy desiccation and impending harvest maturity, triggering automated harvest window adjustments in the MILP solver.

### 8.5 GLOBALG.A.P. IFA v6 and Agricultural Chemical Governance
Commercial export growers and processors must comply with the GLOBALG.A.P. Integrated Farm Assurance (IFA) standard Version 6:
- **Pre-Harvest Water Testing (FV 5.1):** Irrigation and frost-protection water applied within 14 days of harvest must be tested for generic *Escherichia coli*. If microbial contamination exceeds $126\text{ CFU / 100 mL}$ geometric mean, harvest is legally prohibited until a corrective sanitary survey is completed.
- **Pre-Harvest Interval (PHI) Enforcement (CB 7.6):** The farm management system must link spray job logs directly to the harvest scheduler. If an active ingredient has a 21-day PHI, the parcel is flagged with an immutable computational lock in the ERP database preventing work order issuance prior to $t_{\text{spray}} + 21\text{ days}$.
- **Maximum Residue Limits (MRLs):** Chemical residues must comply with destination market legislation (e.g., EU Regulation EC No 396/2005), which frequently mandates tolerances 10 to 100 times stricter than domestic EPA tolerances, requiring destination-specific field allocation in the MILP solver.

---

## 9. Industry practice and case evidence

### 9.1 Sugarcane Cut-to-Crush Optimization: Raízen (Brazil)
Raízen, a joint venture between Shell and Cosan, operates 35 sugar and ethanol agro-industrial units in Brazil, processing over 80 million metric tonnes of sugarcane annually across 1.3 million hectares.

**The Operational Problem:** Once chopped by mechanical combine harvesters, sugarcane stalks lose sucrose via endogenous invertases and *Leuconostoc mesenteroides* bacteria, which consume sucrose and produce viscous dextran polymers. Dextran reduces industrial sugar boiling house recovery by up to $12\%$ and increases crystallization molasses loss.

**Engineered Solution:** Raízen deployed the "Tower of Agility" (Torre de Agilidade), a centralized, 24/7 telemetry and operations research hub in Piracicaba. The center monitors over 4,000 telemetry-enabled assets (combines, transbordo tractors, multi-trailer road trains). A real-time heuristic MILP continuously re-optimizes the cut-to-crush dispatching chain.

**Documented Impact:** Raízen reduced its mean cut-to-crush dwell time from $36\text{ hours}$ in the early 2000s to under $18\text{ hours}$ fleet-wide, recovering an estimated $1.2\text{ kg}$ to $1.8\text{ kg}$ of total recoverable sugar (ATR) per metric tonne of cane milled. Across an 80-million-tonne harvest, this biological optimization yields over **$\$75\text{ million}$** in annual recovered sucrose.

### 9.2 Central Valley Processing Tomatoes: Morning Star Packing Company
The Morning Star Packing Company processes approximately $40\%$ of California's processing tomatoes and $10\%$ of global volume, operating three continuous canneries with combined seasonal intake exceeding $1.2\text{ million metric tonnes}$.

**The Operational Problem:** The processing season runs for approximately 85 to 90 continuous days between July and October. Each factory consumes up to $1,400\text{ tonnes/hour}$. If tomato deliveries drop below capacity, steam boiler efficiency collapses; if deliveries surge, hundreds of double-trailer flatbed trucks sit in $40^\circ\text{C}$ Central Valley heat, causing rapid tomato soft-rot and mold blooms that contaminate flume lines.

**Engineered Solution:** Morning Star implements an end-to-end coordinated planting and harvesting program. Seedlings are transplanted under strict degree-day schedules spanning southern Kern County up to northern Colusa County, exploiting regional thermal latitude gradients to flatten the seasonal arrival curve. Harvesters utilize automated optical sorting systems to reject green fruit in the field. Intake dispatching is coordinated via an automated truck scheduling algorithm that staggers delivery appointments into 15-minute intake windows at the weighbridge.

**Evidence of Practice:** Through thermal-gradient planting and central intake dispatching, factory assets operate at over **$92\%$ nameplate capacity utilization** across the 85-day campaign, minimizing expensive intermediate tank storage and preventing queue-induced fruit rejection.

### 9.3 Vitis Vinifera: Nocturnal Mechanical Harvesting in Premium Viticulture
In high-temperature wine regions (e.g., Napa Valley, Central Otago, Barossa Valley), premium wineries mandate nocturnal harvesting between 02:00 and 08:00.

**The Physical Driver:** During daytime hours, grape pulp temperatures exceed $32^\circ\text{C}$. Picking warm fruit ruptures berry skins, exposing grape juice to atmospheric oxygen and activating polyphenol oxidase (PPO) enzymes. This causes immediate browning of white musts (loss of aromatic thiols and terpenes) and spontaneous wild yeast fermentation. Furthermore, chilling warm juice at the winery from $32^\circ\text{C}$ down to fermentation temperatures ($14^\circ\text{C}$) imposes massive refrigeration chiller peak electrical loads ($1.16\text{ kWh}$ per tonne per $^\circ\text{C}$ reduction).

**The Supply Chain Adaptation:** Harvesting fleets operate equipped with high-intensity LED illumination packages. Mechanically harvested grapes enter the destemmer-crusher at $10^\circ\text{C}$ to $14^\circ\text{C}$, cutting plant refrigeration energy costs by **$60\%$**, eliminating oxidative browning, and preserving critical sulfur volatile aroma compounds.

### 9.4 Quick-Freeze Vegetables: Birds Eye / Nomad Foods and the 120-Minute Freeze Dash
Birds Eye (a division of Nomad Foods) contracts over 30,000 hectares of green peas and sweet corn across the United Kingdom and continental Europe, producing frozen vegetables under strict tenderometer quality standards.

**The Biophysical Challenge:** Freshly picked green peas respire at an extraordinary rate ($\approx 150\text{ mg CO}_2\text{ / (kg}\cdot\text{h)}$ at $20^\circ\text{C}$). For every hour peas remain at ambient temperature after being shelled by mobile viner combines, endogenous sucrose converts to insoluble amylose and amylopectin starches, causing the tenderometer reading to climb rapidly. Above a tenderometer reading of 105 TU, the peas lose their prized sweet burst and turn mealy, dropping in commercial wholesale value by $35\%$.

**The Logistics Solution:** Birds Eye operates a proprietary agricultural logistics network engineered around the **"120-Minute Field-to-Freezer Guarantee"**:
1. **Thermal-Unit Sowing Grids:** Pea drilling is staggered in spring based on dynamic degree-day accumulation calculations rather than fixed weekly calendars, ensuring fields do not bunch at harvest.
2. **Field-Side Radio-Telemetry Viners:** Mobile pea viners thresh the crop in the field and unload directly into high-speed refrigerated tipping trailers.
3. **Geo-Fenced Priority Intake Corridors:** Processing factories in Hull and Lowestoft utilize dynamic queue-sensing systems. If an incoming pea trailer encounters highway traffic delays pushing transit time past 75 minutes, the central plant dispatcher elevates the trailer's priority to bypass secondary queues directly into the blanching and Individual Quick Freezing (IQF) fluidised-bed tunnel ($ -35^\circ\text{C}$).

**Documented Outcome:** Across annual harvesting campaigns, over **$98.5\%$ of all contracted volume** is washed, blanched, and frozen within 150 minutes of field severance, preserving ascorbic acid (Vitamin C) levels and maintaining a 94% Grade A pack-out yield.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign | Physical / Behavioral Mechanism | Supply Chain Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **Phenological Bunching Convergence** | Unseasonable mid-season heatwave ($+5^\circ\text{C}$ above 30-year average) occurring across multiple growing districts. | Extreme thermal accumulation ($GDD$) compresses physiological development; early-planted and late-planted cultivars mature simultaneously. | Factory intake overwhelmed by $200\text{--}300\%$ over capacity; unharvested fields rot on vine; catastrophic grower breach penalties. | Geographically diversify contracted acreage across latitude/elevation bands; stagger cultivars by distinct GDD maturities. |
| 2 | **Rainout Machinery Immobilization** | S-band weather radar showing an incoming atmospheric river $> 25\text{ mm}$ precipitation within 48 hours. | Heavy precipitation saturates topsoil; combine harvesters ($> 15\text{ tonnes}$) cause severe soil liquefaction, tire rutting, and immobilization. | Complete halt of raw material intake; factory runs out of feedstock; idle shift labor costs exceed $\$50,000/\text{shift}$. | Schedule harvest buffers ahead of rainfront; maintain contracted well-drained sandy-soil fields designated for wet-weather emergency harvest. |
| 3 | **Weighbridge Queue Sugar Inversion** | Truck turn-time at factory entrance scales exceeding 120 minutes. | Produce sits in stationary open-top trailers under solar radiation; internal pile respiration raises temperature $> 40^\circ\text{C}$; microbial invertase destroys sucrose. | Recoverable sugar drop ($> 1\%$ per 6h); high microbial dextran/lactic acid; boiler evaporator fouling; product downgrade. | Implement slot-based dynamic gate appointments; enforce hard cutoff dwell limits ($\tau_{\max}$) with divert routing to secondary plants. |
| 4 | **Harvester Roading Overhead Shock** | Telematics showing harvesters spending $> 35\%$ of engine hours in road transit gear rather than threshing gear. | Optimization model solved without spatial cluster constraints, ordering harvesters to jump between non-contiguous parcels miles apart. | Loss of effective harvesting capacity; field harvest windows missed; excessive diesel consumption and machinery track wear. | Incorporate Vehicle Routing (VRP) constraints and minimum contiguous parcel clustering rules ($\ge 20\text{ ha}$) into MILP. |
| 5 | **PHI Chemical Residue Rejection** | Farm spray log audit revealing a late fungicide spray within 7 days of scheduled harvest. | Fungicide or insecticide applied to protect late canopy has a statutory 14-day Pre-Harvest Interval (PHI). | Regulatory violation; entire shipment seized and condemned by state food safety inspectors; total loss of crop value. | Hard-code regulatory PHI lockouts into the harvest ERP; field cannot be selected by optimization engine until $t > t_{\text{spray}} + PHI$. |
| 6 | **Contract Misalignment Weight Hoarding** | Growers resisting harvester entry during peak Brix maturity window; irrigation lines actively running. | Grower contract is structured purely on delivered gross metric tonnes without an adequate soluble solids ($^\circ\text{Brix}$) bonus. | Growers deliberately irrigate prior to harvest to pump fruit with water weight, diluting sugar concentration and increasing factory boiling costs. | Implement a steeply penalized quality price grid where water-diluted low-Brix fruit incurs massive deduction penalties. |
| 7 | **Soil Compaction Compounding** | Penetrometer readings exceeding $2.0\text{ MPa}$ in upper $30\text{ cm}$ of soil following harvest. | Harvesters and heavy transport trailers operate on wet soils above the plastic limit, destroying soil pore structure. | Permanent agronomic yield reduction of $15\text{--}30\%$ for subsequent 3 years; severe grower-processor contractual conflict. | Mandate low-ground-pressure tracks, central tire inflation systems (CTIS), and enforce axle weight limits; halt harvest if soil moisture $> 85\%$ field capacity. |
| 8 | **Refractometer / Lab Calibration Drift** | Statistical divergence between field sampling Brix readings and intake laboratory refractometer readings ($> 0.4^\circ\text{Bx}$). | Intake refractometers fouled by suspended solids or uncalibrated against distilled water / standard sucrose calibration solutions. | Systematic under- or over-payment to growers; supplier disputes; erroneous blending decisions in primary evaporators. | Mandate automated dual-instrument cross-checks and twice-daily recalibration using NIST-traceable sucrose standards. |
| 9 | **Bin / Tote Deficit Field Starvation** | Empty bulk bins or transbordo trailers failing to return from factory to field staging yards. | Factory receiving dock bottleneck delays bin dumping and washing, stranding transport equipment at the processing facility. | Harvesters halt mid-field with full hoppers; harvest labor sits idle while harvest windows slip toward decay. | Model closed-loop container logistics; maintain a dynamic safety stock of clean empty bins at field staging points ($1.5 \times$ daily run). |
| 10 | **Ethylene Auto-Catalytic Cascading** | Flesh firmness dropping rapidly; premature skin yellowing observed in transit bins. | Trace ethylene from bruised or mold-infected fruit triggers auto-catalytic ripening cascade across bulk transport trailer. | Softened fruit collapses under pile weight; bottom bins turn to liquid pulp in transit; flume line screens choke on debris. | Screen field fruit for mechanical bruising; enforce gentle handling; apply pre-harvest 1-MCP (1-methylcyclopropene) where approved. |
| 11 | **Microclimate Thermal Gradient Blindness** | Upland fields maturing 10 days earlier or later than projected by central regional weather station. | Planner assumes a single regional airport weather station applies to all contracted parcels, ignoring adiabatic lapse rates ($0.65^\circ\text{C} / 100\text{ m}$). | Forecasted GDD errors compound to $\pm 120\text{ GDD}$; harvesters arrive at immature or rotting fields. | Deploy on-farm IoT microclimate weather stations and calibrate temperature surfaces using gridded lapse-rate interpolation. |

---

## 11. Assumptions and limits

| # | Assumption | Why It Is Needed | What Breaks It in Practice | Diagnostic Test / Validation Method |
|---|---|---|---|---|
| 1 | **Linear Thermal Time Accumulation** | Allows daily summation of GDD to project phenological stages using basic arithmetic. | Extreme heatwaves ($T > 35^\circ\text{C}$) induce metabolic shutdown and protein denaturation, stalling growth. | Fit non-linear beta-growth or Wang-Engel temperature response curves; test for developmental stalling. |
| 2 | **Spatial Homogeneity of Microclimates** | Simplifies data pipelines by interpolating regional weather grids across thousands of farm hectares. | Topographic elevation changes, cold-air drainage valleys, and coastal marine fog layers create localized microclimates. | Perform spatial cross-validation between regional gridded weather datasets (ERA5) and on-field IoT sensors. |
| 3 | **Deterministic Factory Intake Capacity** | Treats processing plant capacity ($Cap_{jt}$) as a known constant parameter in MILP harvest scheduling. | Unexpected boiler tube failures, centrifuge jams, flume pump motor burnouts, or unscheduled CIP cleanouts halt intake. | Conduct stochastic sensitivity analysis and maintain a dynamic plant capacity derating factor ($\approx 0.85 \times \text{Nameplate}$). |
| 4 | **Non-Splittable Field Execution** | Prevents equipment roading churn by enforcing that a parcel is harvested completely once started. | Extreme mid-field rainstorms or mechanical harvester breakdown force crews to abandon half-harvested parcels. | Review historical field completion logs; evaluate MILP cost gap between integer non-splittable and splittable LP relaxations. |
| 5 | **Linear Quality and Mass Degradation** | Enables linear pricing grid calculations and linear penalty formulations in optimization solvers. | Biological decay is non-linear; fungal rot growth (*Botrytis*, *Rhizopus*) follows exponential Gompertz kinetics once initiated. | Calibrate empirical non-linear decay curves and approximate them via piece-wise linear segments in the MILP formulation. |
| 6 | **Unconstrained Secondary Haulage Equipment** | Assumes that enough road transport trucks and bulk bins exist to haul whatever volume harvesters cut. | Regional trucking shortages, driver hours-of-service (HOS) limits, or harvest container turnaround delays choke transport. | Formulate combined Harvester-Vehicle Routing Problems (HVRP) linking field cutters explicitly to truck fleet schedules. |
| 7 | **Uniform Parcel Soil Hydrology** | Assumes a single root-zone water balance applies uniformly across an entire 20-hectare field. | Soil textural variation (clay lenses interspersed with coarse sand) creates uneven maturity patches within one parcel. | Utilize satellite Sentinel-2 NDVI/NDWI within-field variance maps; flag parcels with high coefficients of variation for multi-pass harvest. |
| 8 | **Zero Pre-Harvest Crop Condemnation** | Assumes all contracted field hectares remain viable for commercial harvest throughout the campaign. | Severe hail damage, phytoplasma disease outbreaks, or pesticide drift from neighboring properties render parcels unmarketable. | Implement weekly pre-harvest agronomist drone scouting flights; update active binary eligibility parameters ($A_i > 0$). |
| 9 | **Constant Chemical Reaction Rates in Transit** | Models cut-to-process sucrose loss as a fixed percentage loss per hour of transit dwell time. | Ambient transit temperatures fluctuate dramatically between midday sun ($40^\circ\text{C}$) and nighttime transit ($15^\circ\text{C}$), altering Arrhenius rates. | Deploy real-time telematics temperature probes inside bulk trailer piles; integrate dynamic Arrhenius kinetic models into dispatching engines. |

---

## 12. Exercises

### Exercise 1: Growing Degree Day and Maturity Window Calculation
A processing tomato cultivar requires $1,500.0\text{ GDD}^\circ\text{C}$ from transplanting to physiological maturity, evaluated with $T_{\text{base}} = 10.0^\circ\text{C}$ and $T_{\text{upper}} = 30.0^\circ\text{C}$ using the Modified Sine/Cutoff method. Field 101 was transplanted on May 1. By June 30 (60 elapsed days), it has accumulated $1,180.0\text{ GDD}$. 
The 10-day meteorological weather forecast for July 1 to July 10 predicts constant daily temperatures of $T_{\max} = 34.0^\circ\text{C}$ and $T_{\min} = 18.0^\circ\text{C}$.
- **Deliverable:** 
  1. Calculate the daily effective GDD accumulated per day during July 1–10 under the specified cutoff method.
  2. Determine the exact forecasted calendar date when Field 101 reaches physiological maturity ($1,500.0\text{ GDD}$).
  3. If an unseasonal heatwave pushed actual daily $T_{\min}$ to $22.0^\circ\text{C}$ with $T_{\max} = 38.0^\circ\text{C}$, recalculate the maturity date.
- **Acceptance Criteria:** Correct application of upper threshold cutoff; arithmetic precision to one decimal place.
- **Estimated Time:** 20 minutes.

### Exercise 2: Soil Moisture Stress and Yield Reduction (FAO-56)
A 10-hectare parcel of sweet corn has a maximum potential unstressed yield of $Y_m = 18.0\text{ metric tonnes/ha}$. During the 20-day flowering and silking stage (Stage 2), cumulative potential crop evapotranspiration is $ET_c = 140.0\text{ mm}$. Due to irrigation canal maintenance failure, actual evapotranspiration delivered is $ET_a = 95.0\text{ mm}$. The yield response factor for flowering sweet corn is $K_y = 1.15$. During all other vegetative and ripening stages, full irrigation is maintained ($ET_a = ET_c$).
- **Deliverable:**
  1. Compute the relative evapotranspiration deficit $(1 - ET_a/ET_c)$ during the flowering period.
  2. Calculate the projected actual yield per hectare ($Y_a$) and total lost harvest tonnage across the 10-hectare farm.
  3. If wholesale sweet corn is contracted at $\$220.00/\text{tonne}$, determine the financial economic damage caused by the canal maintenance failure.
- **Acceptance Criteria:** Correct implementation of the Stewart-Doorenbos-Kassam equation; exact financial computation.
- **Estimated Time:** 25 minutes.

### Exercise 3: Post-Harvest Invertase Decay Economics
A sugarcane estate delivers cane to a processing central. Severed cane loses $0.18\%$ of total recoverable sucrose per hour of dwell time for the first 12 hours, accelerating to $0.35\%$ per hour for every hour between 12 and 24 hours due to *Leuconostoc* microbial infection.
Baseline recoverable sucrose at cut time ($t=0$) is $140.0\text{ kg ATR / metric tonne}$. The mill processes $10,000\text{ tonnes of cane per day}$, purchasing cane at $\$0.32\text{ per kg of ATR}$.
- **Deliverable:**
  1. Calculate the recoverable sucrose ($\text{kg ATR/tonne}$) if cut cane is processed at $6\text{ hours}$, $12\text{ hours}$, $18\text{ hours}$, and $24\text{ hours}$ post-harvest.
  2. Compute the daily dollar loss to the processing mill if average logistics delays push cut-to-crush dwell time from $8\text{ hours}$ to $18\text{ hours}$ across the entire $10,000\text{ t/day}$ intake.
- **Acceptance Criteria:** Piece-wise decay rate accurately integrated; net financial loss correctly calculated.
- **Estimated Time:** 25 minutes.

### Exercise 4: MILP Harvest Scheduling with Machine Capacity
Consider three field parcels:
- Parcel 1: $A_1 = 5.0\text{ ha}$, $Y_1 = 60\text{ t/ha}$, optimal day $t=1$.
- Parcel 2: $A_2 = 6.0\text{ ha}$, $Y_2 = 50\text{ t/ha}$, optimal day $t=2$.
- Parcel 3: $A_3 = 4.0\text{ ha}$, $Y_3 = 70\text{ t/ha}$, optimal day $t=2$.
The planning horizon is 2 days ($t \in \{1, 2\}$). The enterprise operates 1 harvester with a maximum daily capacity of $7.0\text{ ha/day}$. The processing plant has an intake limit of $360.0\text{ tonnes/day}$.
Yield drops by $5\%$ if harvested 1 day away from optimal. Base price is $\$100/\text{t}$. Harvesting cost is $\$200/\text{ha}$. Transport cost is $\$10/\text{t}$.
- **Deliverable:**
  1. Write out the algebraic constraints for harvester capacity and plant intake capacity for Days 1 and 2.
  2. Test whether the schedule (Parcel 1 on Day 1; Parcel 2 on Day 2; Parcel 3 on Day 2) is feasible. If not, state which constraint is violated.
  3. Determine the optimal feasible schedule that maximizes net operating margin.
- **Acceptance Criteria:** Explicit linear mathematical formulation; verified feasibility check; optimal solution identified.
- **Estimated Time:** 35 minutes.

### Exercise 5: Tenderometer Creep and Grade Downgrade in Processing Peas
Processing green peas are priced according to a strict tenderometer index grid:
- Grade A (Fancy): $\text{TU} \le 105 \implies \$450.00\text{ / metric tonne}$.
- Grade B (Extra Standard): $106 \le \text{TU} \le 120 \implies \$320.00\text{ / metric tonne}$.
- Grade C (Standard / Substandard): $\text{TU} \ge 121 \implies \$180.00\text{ / metric tonne}$.
A 20-hectare field exhibits a tenderometer reading of $\text{TU} = 98$ on Monday morning at 08:00, with biomass yield at $4.0\text{ t/ha}$. Under warm conditions, the tenderometer reading increases at a steady rate of $0.55\text{ TU per hour}$, while crop biomass grows by $0.02\text{ t/ha per hour}$ as starches accumulate.
- **Deliverable:**
  1. Determine the exact hour when the crop transitions from Grade A to Grade B, and from Grade B to Grade C.
  2. Calculate total gross grower revenue if the field is harvested at:
     - $t = 12\text{ hours}$ (Monday 20:00).
     - $t = 24\text{ hours}$ (Tuesday 08:00).
     - $t = 48\text{ hours}$ (Wednesday 08:00).
  3. Critique which harvest timing optimizes grower revenue versus processor finished-product brand value.
- **Acceptance Criteria:** Accurate hourly projection of continuous state variables; economic trade-off analysis.
- **Estimated Time:** 30 minutes.

### Exercise 6: Contract Design for Quality Alignment
A citrus juice processing cooperative currently contracts oranges under a flat gross weight contract ($\$180.00\text{ per metric tonne}$). Growers regularly harvest crops during peak fruit hydration immediately after heavy rains, delivering diluted fruit averaging $9.5^\circ\text{Brix}$. The processor's primary evaporators require $11.5^\circ\text{Brix}$ fruit to produce frozen concentrated orange juice (FCOJ) economically.
- **Deliverable:**
  1. Critique the behavioral failure of the existing contract.
  2. Design a revised pound-solids pricing formula ($P_{\text{solids}} = \text{Mass} \times \text{Brix} \times \text{Base Price}$) with an explicit bonus/penalty schedule that eliminates the incentive to deliver water-bloated fruit.
  3. Define the operational acceptance criteria and audit protocols required at the factory weighbridge.
- **Acceptance Criteria:** Complete mathematical contract formula; clear explanation of economic game theory dynamics.
- **Estimated Time:** 30 minutes.

### Exercise 7: Enterprise System Design for FSMA 204 Harvest Compliance
Design the data architecture and event payload for a digital harvesting dispatch system complying with US FDA FSMA Rule 204 for fresh romaine lettuce.
- **Deliverable:**
  1. Define the relational or document schema capturing all required Key Data Elements (KDEs) for the Harvesting Critical Tracking Event (CTE).
  2. Provide a mock JSON-LD / EPCIS 2.0 compliant event object documenting a 500-carton harvest event from Ranch 14, Field B.
  3. Specify how the Traceability Lot Code (TLC) is generated and physically affixed to bulk harvest totes.
- **Acceptance Criteria:** Valid JSON syntax; inclusion of all mandatory FSMA 204 KDEs; industry-standard GS1 identifiers.
- **Estimated Time:** 35 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: GDD and Maturity Window
1. **Daily Effective GDD Calculation:**
   Given $T_{\max} = 34.0^\circ\text{C}$ and $T_{\min} = 18.0^\circ\text{C}$, apply the horizontal cutoff thresholds ($T_{\text{base}} = 10.0^\circ\text{C}, T_{\text{upper}} = 30.0^\circ\text{C}$):
   - $T_{\max}^* = \min(34.0, 30.0) = 30.0^\circ\text{C}$.
   - $T_{\min}^* = \max(18.0, 10.0) = 18.0^\circ\text{C}$.
   - Effective average temperature: $T_{\text{avg}}^* = \frac{30.0 + 18.0}{2} = 24.0^\circ\text{C}$.
   - Daily $GDD = T_{\text{avg}}^* - T_{\text{base}} = 24.0 - 10.0 = 14.0\text{ GDD}^\circ\text{C/day}$.
2. **Forecasted Maturity Date:**
   - Remaining GDD required: $1,500.0 - 1,180.0 = 320.0\text{ GDD}^\circ\text{C}$.
   - Days required at $14.0\text{ GDD/day}$: $\frac{320.0}{14.0} = 22.86\text{ days} \implies 23\text{ days}$.
   - Forecasted physiological maturity date: **July 23**.
3. **Under the Anomalous Heatwave:**
   - $T_{\max}^* = \min(38.0, 30.0) = 30.0^\circ\text{C}$.
   - $T_{\min}^* = \max(22.0, 10.0) = 22.0^\circ\text{C}$.
   - $T_{\text{avg}}^* = \frac{30.0 + 22.0}{2} = 26.0^\circ\text{C}$.
   - Daily $GDD = 26.0 - 10.0 = 16.0\text{ GDD}^\circ\text{C/day}$.
   - Days required: $\frac{320.0}{16.0} = 20.0\text{ days}$.
   - Recalculated maturity date: **July 20** (advancing harvest readiness by 3 full days).

### Answer to Exercise 2: Soil Moisture Stress and Yield Reduction
1. **Evapotranspiration Deficit:**
   - Deficit ratio: $1 - \frac{ET_a}{ET_c} = 1 - \frac{95.0}{140.0} = 1 - 0.6786 = 0.3214$ ($32.14\%$ deficit).
2. **Actual Yield Calculation:**
   - Apply Doorenbos-Kassam equation: $\left(1 - \frac{Y_a}{Y_m}\right) = K_y \left(1 - \frac{ET_a}{ET_c}\right) = 1.15 \times 0.3214 = 0.3696$.
   - Relative yield: $\frac{Y_a}{Y_m} = 1 - 0.3696 = 0.6304$.
   - Actual yield: $Y_a = 18.0 \times 0.6304 = 11.347\text{ metric tonnes/ha}$.
   - Lost yield per hectare: $18.0 - 11.347 = 6.653\text{ t/ha}$.
   - Total lost tonnage across 10 hectares: $6.653 \times 10 = 66.53\text{ metric tonnes}$.
3. **Financial Economic Damage:**
   - Lost revenue: $66.53\text{ t} \times \$220.00/\text{t} = \mathbf{\$14,636.60}$.

### Answer to Exercise 3: Post-Harvest Invertase Decay Economics
1. **Sucrose Remaining Across Dwell Times:**
   - Initial: $140.0\text{ kg ATR/t}$.
   - At $6\text{ h}$ ($0.18\%/\text{h} \times 6\text{ h} = 1.08\%$ loss): $140.0 \times (1 - 0.0108) = \mathbf{138.488\text{ kg ATR/t}}$.
   - At $12\text{ h}$ ($0.18\%/\text{h} \times 12\text{ h} = 2.16\%$ loss): $140.0 \times (1 - 0.0216) = \mathbf{136.976\text{ kg ATR/t}}$.
   - At $18\text{ h}$ (Loss = $2.16\% + 6\text{ h} \times 0.35\% = 2.16\% + 2.10\% = 4.26\%$ loss):
     $140.0 \times (1 - 0.0426) = \mathbf{134.036\text{ kg ATR/t}}$.
   - At $24\text{ h}$ (Loss = $2.16\% + 12\text{ h} \times 0.35\% = 2.16\% + 4.20\% = 6.36\%$ loss):
     $140.0 \times (1 - 0.0636) = \mathbf{131.096\text{ kg ATR/t}}$.
2. **Economic Loss of Delay from 8 Hours to 18 Hours:**
   - Loss at $8\text{ h}$: $8 \times 0.18\% = 1.44\% \implies 140.0 \times (1 - 0.0144) = 137.984\text{ kg/t}$.
   - Loss at $18\text{ h}$: $4.26\% \implies 134.036\text{ kg/t}$.
   - Lost ATR per tonne: $137.984 - 134.036 = 3.948\text{ kg ATR/t}$.
   - Total lost sugar per day across $10,000\text{ tonnes}$: $10,000 \times 3.948 = 39,480.0\text{ kg ATR}$.
   - Daily financial economic destruction: $39,480.0\text{ kg} \times \$0.32/\text{kg} = \mathbf{\$12,633.60\text{ per day}}$ (over **$\$2.27\text{ million}$** across a 180-day crushing season).

### Answer to Exercise 4: MILP Harvest Scheduling
1. **Algebraic Constraints:**
   - Harvester Area Constraint: $\sum_i A_i x_{it} \le 7.0\text{ ha}, \forall t \in \{1, 2\}$.
   - Factory Intake Constraint: $\sum_i A_i Y_{it} x_{it} \le 360.0\text{ t}, \forall t \in \{1, 2\}$.
2. **Feasibility Check of Baseline Schedule:**
   - Day 1: Parcel 1 $\implies 5.0\text{ ha} \le 7.0\text{ ha}$ (Feasible); Tonnage: $5.0 \times 60 = 300.0\text{ t} \le 360.0\text{ t}$ (Feasible).
   - Day 2: Parcel 2 ($6.0\text{ ha}$) + Parcel 3 ($4.0\text{ ha}$) $\implies 6.0 + 4.0 = 10.0\text{ ha} > 7.0\text{ ha}$ **[VIOLATION: Harvester capacity exceeded by 3.0 ha]**.
   - Day 2 Tonnage: $(6.0 \times 50) + (4.0 \times 70) = 300.0 + 280.0 = 580.0\text{ t} > 360.0\text{ t}$ **[VIOLATION: Factory capacity exceeded by 220.0 t]**.
3. **Finding Optimal Schedule:**
   Either Parcel 2 or Parcel 3 must shift to Day 1:
   - *Option A: Shift Parcel 3 to Day 1.*
     - Day 1 Area: $A_1 + A_3 = 5.0 + 4.0 = 9.0\text{ ha} > 7.0\text{ ha}$ [Infeasible].
   - *Option B: Shift Parcel 2 to Day 1.*
     - Day 1 Area: $A_1 + A_2 = 5.0 + 6.0 = 11.0\text{ ha} > 7.0\text{ ha}$ [Infeasible].
   - *Option C: Shift Parcel 1 to Day 2 and Parcel 2 to Day 1.*
     - Day 1: Parcel 2 alone ($6.0\text{ ha} \le 7.0\text{ ha}$; Tonnage = $6.0 \times (50 \times 0.95) = 285.0\text{ t} \le 360.0\text{ t}$).
     - Day 2: Parcel 1 ($5.0\text{ ha}$) + Parcel 3 ($4.0\text{ ha}$) = $9.0\text{ ha} > 7.0\text{ ha}$ [Infeasible].
   - *Option D: Partial parcel harvesting / split parcel:*
     If integer parcels are strictly non-splittable, the single machine cannot harvest all 15 hectares in 2 days because total area is $15.0\text{ ha}$ while maximum 2-day machine capacity is $2 \times 7.0 = 14.0\text{ ha}$.
     *Analytical conclusion:* The problem as posed is structurally infeasible for 1 harvester over a 2-day horizon without activating a second harvester or extending the planning horizon to 3 days.

### Answer to Exercise 5: Tenderometer Creep and Grade Downgrade
1. **Transition Timestamps:**
   - Baseline at $t=0$: $\text{TU} = 98.0$ (Grade A).
   - Rate: $0.55\text{ TU/h}$.
   - Grade A ceiling: $\text{TU} = 105.0 \implies \Delta \text{TU} = 105.0 - 98.0 = 7.0\text{ TU}$.
     Time to Grade B: $\frac{7.0}{0.55} = 12.73\text{ hours}$ ($\approx \mathbf{20:44\text{ Monday}}$).
   - Grade B ceiling: $\text{TU} = 120.0 \implies \Delta \text{TU} = 120.0 - 98.0 = 22.0\text{ TU}$.
     Time to Grade C: $\frac{22.0}{0.55} = 40.0\text{ hours}$ ($\approx \mathbf{00:00\text{ Wednesday midnight}}$).
2. **Gross Revenue Across Timestamps (20 ha):**
   - At $t = 12\text{ h}$ (Grade A, $\text{TU} = 98 + 12 \times 0.55 = 104.6 \le 105$):
     - Yield: $4.0 + 12 \times 0.02 = 4.24\text{ t/ha}$. Total tonnage: $20 \times 4.24 = 84.8\text{ t}$.
     - Revenue: $84.8\text{ t} \times \$450.00/\text{t} = \mathbf{\$38,160.00}$.
   - At $t = 24\text{ h}$ (Grade B, $\text{TU} = 98 + 24 \times 0.55 = 111.2$):
     - Yield: $4.0 + 24 \times 0.02 = 4.48\text{ t/ha}$. Total tonnage: $20 \times 4.48 = 89.6\text{ t}$.
     - Revenue: $89.6\text{ t} \times \$320.00/\text{t} = \mathbf{\$28,672.00}$.
   - At $t = 48\text{ h}$ (Grade C, $\text{TU} = 98 + 48 \times 0.55 = 124.4 > 120$):
     - Yield: $4.0 + 48 \times 0.02 = 4.96\text{ t/ha}$. Total tonnage: $20 \times 4.96 = 99.2\text{ t}$.
     - Revenue: $99.2\text{ t} \times \$180.00/\text{t} = \mathbf{\$17,856.00}$.
3. **Critique:**
   Harvesting at $t=12\text{ h}$ maximizes both grower revenue ($\$38,160.00$) and processor premium pack-out. Delaying harvest from 12h to 24h increases delivered biomass by $4.8\text{ tonnes}$ ($+5.66\%$) but destroys $\$9,488.00$ ($-24.86\%$) of crop value due to the discrete grade downgrade cliff.

### Answer to Exercise 6: Contract Design
- **Grading Rubric for Solution:**
  - *Behavioral diagnosis:* Recognizes moral hazard where growers treat water weight as free revenue under tonnage contracts, externalizing boiler evaporation costs onto the processor.
  - *Mathematical Formula:* Specifies settlement based on dry soluble solids: $\text{Payment} = \text{Gross Tonnes} \times \frac{\text{Brix}}{100} \times P_{\text{solids\_base}} \times [1 + \lambda (B - 11.5)]$.
  - *Incentive Structure:* Proves that the marginal return of adding water is negative ($\frac{\partial \text{Revenue}}{\partial \text{Water}} < 0$).
  - *Audit Control:* Specifies independent composite sampling at the intake flume using automated temperature-compensated digital refractometry.

### Answer to Exercise 7: FSMA 204 System Design
- **Grading Rubric for Solution:**
  - *Schema Validation:* Captures all 7 statutory KDEs under 21 CFR § 1.1325 (Harvester GLN, Field GPS centroid, TLC, Crop Variety, Harvest Timestamp, Quantity, Cooling Log).
  - *JSON-LD Structure:* Uses valid EPCIS 2.0 vocabulary (`@context`, `isA: "ObjectEvent"`, `action: "ADD"`, `bizStep: "urn:epcglobal:cbv:bizstep:harvesting"`).
  - *TLC Physical Binding:* Mandates synthetic water-resistant 2D DataMatrix barcode labels affixed to each macro-bin before loading onto highway flatbed trailers.

---

## 14. Interview questions

### Question 1: How do you mathematically decouple harvest scheduling from calendar dates in a multi-region agricultural supply chain?
**Strong candidate answers include:**
- Explains that calendar dates are an inadequate proxy for biological development due to annual and spatial temperature variance.
- Details the calculation of thermal time via Growing Degree Days (GDD) with crop-specific base ($T_{\text{base}}$) and ceiling ($T_{\text{upper}}$) temperatures.
- Describes the integration of numerical weather predictions (NWP) and climatological historical averages to establish dynamic, rolling maturity windows $[ES_i, LF_i]$.
- Connects GDD progression to mixed-integer linear programming (MILP) dispatch models that update daily decision variables based on observed thermal accumulation.

**Red flags / weak candidate answers reveal:**
- Relies on static historical calendar schedules (e.g., "Field 12 is always harvested on September 15").
- Fails to define or understand the concept of base temperature ($T_{\text{base}}$).
- Ignores high-temperature developmental plateaus ($T_{\text{upper}}$).

---

### Question 2: Why is the cut-to-crush window so critical in sugarcane and sugar beet processing, and how do you model it in an operations research framework?
**Strong candidate answers include:**
- Identifies the biological mechanism: mechanical cutting severs the plant vascular system, activating endogenous invertase enzymes that hydrolyze crystallizable sucrose into non-crystallizable reducing sugars (glucose and fructose).
- Explains secondary infection by *Leuconostoc mesenteroides* producing dextran polymers that severely foul industrial evaporators and reduce centrifuge recovery.
- Formulates a hard dwell-time constraint ($\tau_{\text{dwell}} \le \tau_{\max}$) or penalty function in the objective function linking haulage transit time and weighbridge queuing.
- References real industrial systems (e.g., Raízen's centralized telemetry towers in Brazil).

**Red flags / weak candidate answers reveal:**
- Believes post-harvest decay is purely cosmetic or relates only to surface moisture loss.
- Treats transportation logistics as an unconstrained, independent activity decoupled from factory chemical recovery.
- Has no concept of chemical sucrose inversion or industrial sugar crystallization kinetics.

---

### Question 3: How do you resolve the operational conflict between a processing plant's desire for high Brix and a contracted grower's desire for maximum harvest tonnage?
**Strong candidate answers include:**
- Diagnoses the structural misalignment of standard gross-weight procurement contracts.
- Explains the biophysical trade-off: late-stage irrigation increases water weight (tonnes/ha) but dilutes soluble solids ($^\circ\text{Brix}$) and increases processing energy costs (evaporation).
- Proposes a quality price grid based on dry matter solids (pound-solids contracts) with asymmetric penalty tiers.
- Evaluates the game-theoretic Nash equilibrium: structuring the formula such that the marginal revenue of diluting fruit with water is strictly less than the marginal quality penalty.

**Red flags / weak candidate answers reveal:**
- Suggests forcing growers through legal threats without changing contractual financial incentives.
- Believes higher tonnage always benefits the processing plant.
- Fails to recognize that water weight adds transport freight cost and boiler fuel expense without adding finished product yield.

---

### Question 4: In a mixed-integer harvest scheduling model, why is the "non-splittable field constraint" necessary, and what happens to solver performance when it is introduced?
**Strong candidate answers include:**
- Identifies the operational physical reality: combine harvesters and roading crews incur high setup, transit, and pack-up costs when moving between fields. Splitting a 10-hectare field across three non-consecutive days is practically unacceptable.
- Formulates the constraint using binary variables ($y_{it} \in \{0, 1\}$) and continuous fraction variables ($x_{it} \in [0, 1]$) with Big-$M$ linking: $x_{it} \le y_{it}$ and $\sum_t y_{it} \le 1$.
- Explains the computational complexity: introducing hundreds of binary variables turns a fast continuous LP into an NP-hard mixed-integer program, requiring branch-and-bound, branch-and-cut, or specialized heuristic approaches (e.g., rolling horizon heuristics).

**Red flags / weak candidate answers reveal:**
- Does not understand why an optimization solver would split a field in the first place (missing that continuous LP relaxation naturally fractions fields to fill capacity knapsacks).
- Claims that adding integer constraints makes an optimization model faster or easier to solve.

---

### Question 5: How does the FAO-56 Penman-Monteith dual crop coefficient architecture improve yield forecasting compared to simple rainfall correlation models?
**Strong candidate answers include:**
- Contrasts simple empirical statistical regression with mechanistic physics-based energy balance modeling.
- Explains the dual coefficient mechanism: splitting total evapotranspiration into basal crop transpiration ($K_{cb}$, driven by stomatal conductance and green canopy cover) and soil surface evaporation ($K_e$, driven by wet topsoil).
- Describes how soil moisture balance tracks root-zone depletion relative to Total Available Water ($TAW$) and Readily Available Water ($RAW$).
- Links soil water stress ($K_s < 1.0$) to the Stewart-Doorenbos-Kassam yield response model across distinct phenological stages (e.g., flowering vs. vegetative).

**Red flags / weak candidate answers reveal:**
- Confuses reference evapotranspiration ($ET_0$) with actual rainfall ($P$).
- Treats water demand as a static constant across the entire growing season.
- Unaware of standard FAO-56 international agronomic standards.

---

### Question 6: What are the regulatory implications of the US FDA FSMA Rule 204 on harvesting dispatch systems for fresh produce?
**Strong candidate answers include:**
- Identifies FSMA 204 as a binding statutory mandate requiring end-to-end electronic traceability for items on the Food Traceability List (FTL).
- Categorizes Harvesting as a designated Critical Tracking Event (CTE).
- Lists mandatory Key Data Elements (KDEs): field GPS coordinates/FSA lot, harvest date, grower GLN, commodity variety, quantity/unit of measure, and the generation of a Traceability Lot Code (TLC).
- Notes the strict 24-hour electronic record production mandate during outbreak investigations.

**Red flags / weak candidate answers reveal:**
- Believes FSMA 204 is an optional voluntary marketing standard like organic certification.
- Cannot name any specific Key Data Elements required at harvest.
- Unaware that field location and harvest timestamps must link to downstream distribution records.

---

### Question 7: How do you handle sudden weather downtime risks (e.g., rainouts) in an automated daily harvest dispatching pipeline?
**Strong candidate answers include:**
- Models precipitation risk as a dynamic binary parameter or stochastic chance constraint ($\delta_{it}^{\text{rain}}$) driven by high-resolution radar and quantitative precipitation forecasts (QPF).
- Explains the physical consequence: heavy soil saturation halts harvesting due to combine immobilization, track slippage, and irreversible soil compaction.
- Proposes operational mitigations: maintaining a portfolio of contracted parcels on sandy, well-drained soils designated as "wet-weather reserves"; building intermediate raw product buffer inventory at the plant prior to rain fronts; executing re-optimization rolling horizons.

**Red flags / weak candidate answers reveal:**
- Suggests operating heavy harvesters through deep mud regardless of soil damage.
- Proposes holding months of perishable raw material inventory in uncooled storage.
- Assumes weather forecasts are $100\%$ deterministic weeks in advance.

---

### Question 8: Why do premium wine producers mandate nocturnal harvesting, and what is the quantitative impact on winery cold chain logistics?
**Strong candidate answers include:**
- Highlights the biochemical drivers: picking grapes at night ($10^\circ\text{C}$ to $14^\circ\text{C}$) inhibits polyphenol oxidase (PPO) enzymes, preventing oxidative browning of white musts and preserving delicate aroma thiols.
- Prevents premature wild yeast fermentation in bulk transit bins.
- Quantifies thermodynamic energy savings: eliminates the need for emergency juice must chillers at the winery crush pad, saving approximately $1.16\text{ kWh}$ of electrical refrigeration energy per metric tonne per degree Celsius of pulp temperature reduction.
- Notes secondary labor benefits: improved worker safety and crew productivity avoiding daytime heat stress ($> 38^\circ\text{C}$).

**Red flags / weak candidate answers reveal:**
- Believes nighttime harvesting is purely traditional or aesthetic.
- Unaware of oxidative browning enzymes in fruit processing.
- Ignores the thermodynamic refrigeration energy balance of industrial food processing.

---

## 15. Key terms

- **Acreage Contract:** Sourcing contract where a processor buys all production from a specified surface area, assuming yield variance risk.
- **Base Temperature ($T_{\text{base}}$):** The lowest ambient temperature at which enzymatic plant development occurs for a given species.
- **BBCH Scale:** International standardized phenological scale (00–99) defining uniform crop developmental stages.
- **Basal Crop Coefficient ($K_{cb}$):** The ratio of crop transpiration to reference evapotranspiration under unstressed conditions with a dry soil surface.
- **Critical Tracking Event (CTE):** Statutory operational milestone under FSMA 204 (such as Harvesting or Initial Packing) requiring mandatory record generation.
- **Cut-to-Crush Window:** The maximum allowable elapsed time between field severance and industrial extraction before sucrose inversion destroys economic recovery.
- **Degree-Day Cutoff Method:** Horizontal or vertical bounds applied to temperature curves to prevent developmental over-accumulation during extreme heat.
- **Dextran:** High-molecular-weight glucose polymer synthesized by *Leuconostoc* bacteria in damaged sugarcane, causing severe industrial processing fouling.
- **Dual Crop Coefficient:** The FAO-56 framework splitting crop water demand into basal transpiration ($K_{cb}$) and soil evaporation ($K_e$).
- **Ethylene ($\text{C}_2\text{H}_4$):** Gaseous plant hormone that accelerates ripening, tissue softening, and senescence in climacteric crops.
- **Fluming:** Hydraulic conveying of bulk vegetables or fruit into industrial canneries using water channels.
- **Global Location Number (GLN):** A unique 13-digit GS1 identification number assigned to agricultural fields, farm gates, and processing plants.
- **Growing Degree Day (GDD):** Mathematical measure of daily thermal energy accumulation driving crop phenology.
- **Harvest Window ($[ES_i, LF_i]$):** The bounded chronological time window during which a crop parcel can be harvested without severe quality or yield loss.
- **Invertase:** Endogenous plant enzyme that hydrolyzes sucrose into glucose and fructose upon post-harvest cellular injury.
- **Key Data Element (KDE):** Statutory data field (e.g., TLC, GPS coordinates, date) required to be logged during a FSMA 204 CTE.
- **Modified Sine Method:** Trigonometric numerical technique approximating daily heat accumulation from diurnal temperature extrema.
- **Non-Splittable Field Constraint:** Mathematical logic constraint in MILP enforcing that a parcel must be completed in continuous operational shifts.
- **Normalized Difference Vegetation Index (NDVI):** Satellite spectral index measuring photosynthetic canopy vigor and biomass density.
- **Penman-Monteith Equation:** International biophysical standard equation calculating reference evapotranspiration ($ET_0$) from solar radiation, temperature, wind, and humidity.
- **Phenology:** The study of periodic plant life-cycle events governed by climate and thermal energy.
- **Physiological Maturity:** The developmental point at which an agricultural organ reaches peak dry matter accumulation.
- **Pre-Harvest Interval (PHI):** Mandatory statutory minimum elapsed time between agricultural chemical application and mechanical harvest.
- **Readily Available Water (RAW):** The fraction of total soil moisture that a crop can extract without suffering stomatal closure and yield stress.
- **Respiration Heat:** Thermal energy released by harvested produce tissue metabolizing carbohydrates in transit containers.
- **Soluble Solids Content ($^\circ\text{Brix}$):** Standard refractometric measure of dissolved sugar percentage in plant sap.
- **Stewart-Doorenbos-Kassam Model:** Empirical equation linking relative evapotranspiration deficit to proportional crop yield reduction.
- **Tenderometer:** Standardized mechanical penetrometer/shear press measuring firmness and starch conversion in legumes and sweet corn.
- **Tonnage Contract:** Commercial procurement contract committing a grower to deliver a fixed physical mass of raw material.
- **Traceability Lot Code (TLC):** Unique alphanumeric identifier generated at harvest establishing the primary parent lot for food safety tracing.

---

## 16. Sources

- Allen, R. G., Pereira, L. S., Raes, D., & Smith, M. (1998). *Crop evapotranspiration: Guidelines for computing crop water requirements*. FAO Irrigation and Drainage Paper 56. Food and Agriculture Organization of the United Nations, Rome.
- Arnaout, J. P., & Maatouk, M. (2010). Optimization of quality and operational costs in agricultural harvest operations. *Computers & Industrial Engineering*, 58(4), 562–569.
- Doorenbos, J., & Kassam, A. H. (1979). *Yield response to water*. FAO Irrigation and Drainage Paper 33. Food and Agriculture Organization of the United Nations, Rome.
- GS1 AISBL. (2024). *GS1 Standards in Agriculture: Traceability and Identification from Farm to Fork*. GS1 Global Architecture Release 2024.1.
- Higgins, A. J., Muchow, R. C., Rudd, A. V., & Ford, A. W. (2004). Optimising harvest date in sugar production: A case study for the Mossman mill area. *Australian Journal of Agricultural Research*, 55(4), 465–471.
- Jones, J. W., Hoogenboom, G., Porter, C. H., Boote, K. J., Batchelor, W. D., Hunt, L. A., Wilkens, P. W., Singh, U., Gijsman, A. J., & Ritchie, J. T. (2003). The DSSAT cropping system model. *European Journal of Agronomy*, 18(3–4), 235–265.
- Kader, A. A. (2002). *Postharvest Technology of Horticultural Crops* (3rd ed.). University of California, Agriculture and Natural Resources, Publication 3311.
- U.S. Food and Drug Administration. (2022). *FSMA Final Rule on Requirements for Additional Traceability Records for Certain Foods (Food Traceability Rule)*. 21 CFR Part 1, Subpart S. Federal Register, 87 FR 70910.
- United States Department of Agriculture, Agricultural Marketing Service. (2020). *United States Standards for Grades of Processing Tomatoes*. 7 CFR Part 51, Subpart 51.3410.

---

## 17. Links

- [Module 32 Index](_index.md)
- Sibling Topics:
  - [Topic 02: Forecasting Perishable Demand](02_perishable-forecasting.md)
  - [Topic 03: Shelf Life, FEFO, and Age-Based Inventory](03_shelf-life-fefo.md)
  - [Topic 05: Cold Chain and Quality](05_cold-chain-quality.md)
  - [Topic 09: Sourcing Contracts and Yield Risk](09_sourcing-contracts-yield-risk.md)
- Survey File:
  - [Course Survey File: Food, Agriculture, and Perishable Supply Chains](../../../course/32_food_agriculture_perishables.md)
- Foundational Methods:
  - [Module 07: Time Series Forecasting](../../02_methods/07_time_series_forecasting/_index.md)
  - [Module 10: Inventory Optimization](../../02_methods/10_inventory_optimization_multi_echelon/_index.md)
  - [Module 21: Supply Chain Optimization](../../04_value_governance/21_supply_chain_network_design_optimization/_index.md)
