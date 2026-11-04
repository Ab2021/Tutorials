# Intermodal and Hinterland Operations

> Module 41 — Maritime, Rail, and Air Cargo Operations Engineering
> Topic 06 of 10
> Estimated study time: 4.5 hours
> Prerequisites: Module 41, Topic 01 (`01_berth-quay-crane.md`), Topic 02 (`02_terminal-yard-gate.md`), Topic 03 (`03_rail-network-paths.md`); Module 12 (`12_transportation_routing_last_mile.md`)
> Version 2026.5

*Seaport-hinterland supply chains, directional Dry Port / Inland Container Depot (ICD) hub topologies, on-dock rail intermodal transfer economics, inland waterway push-barge convoy hydrodynamics, multinomial modal split logit models with elasticity analysis, chassis pool management regimes, empty container repositioning cycles, and corridor capacity throttling.*

---

## 1. Why this topic exists

In container logistics, a marine terminal is not a freight destination; it is a dynamic flow interface. Over 85% of containerized cargo discharged at major gateway ports—such as Rotterdam, Antwerp-Bruges, Los Angeles, Long Beach, Shanghai, Hamburg, and Savannah—terminates deep in the continental hinterland, between 150 and 2,500 kilometers inland from the coastline. 

The central physical conflict of modern maritime logistics lies in the radical mismatch between maritime arrival dynamics and hinterland transport physics:
1. **The maritime arrival pulse:** An Ultra-Large Container Vessel (ULCV) carrying 24,000 Twenty-Foot Equivalent Units (TEU) discharges and loads between 4,000 and 8,000 containers across a compressed 48-to-72-hour window. This injects a high-amplitude, discontinuous volume shock into the marine terminal's storage yard.
2. **The hinterland evacuation boundary:** Inland surface transport infrastructure—highways, mainline rail corridors, inland river locks, and urban access bridges—operates under strict physical, temporal, and legal capacity limits. Rail corridors have fixed signaling headways and train length limits; inland waterways are constrained by river draft fluctuations and lock chamber cycles; road networks are governed by urban congestion, bridge weight formulas, and statutory driver Hours of Service (HOS).

When the evacuation velocity of the hinterland network falls below the quayside discharge rate, the maritime terminal yard acts as an unintended hydraulic accumulator. Yard utilization surges beyond the 75% to 80% critical threshold, triggering non-linear stacking reshuffles, quayside crane starvation, and gate queue spillbacks that can paralyze regional logistics networks.

Hinterland logistics operations engineering exists to resolve four severe structural failures:
- **Spatial gridlock and yard dwell explosion:** Inability to clear import boxes off marine terminal docks results in container dwell times escalating from a standard 3.0–3.5 days to over 8.0–12.0 days. Container yards freeze, ocean carriers enforce blank sailings, and cargo owners incur hundreds of millions of dollars in carrier detention and terminal demurrage fees.
- **Equipment mismatch and chassis starvation:** In markets such as North America where ocean carriers divested from container chassis, motor carriers must perform non-revenue "split moves"—driving miles to off-dock depots to acquire a bare chassis before arriving at a terminal gate. Inefficiencies in chassis pool rebalancing immobilize tens of thousands of containers on terminal ground slots.
- **Directional trade imbalances and empty repo debt:** Global manufacturing imbalances generate chronic spatial deficits. For every three loaded import containers entering the North American West Coast or European gateways from Asia, only one loaded export container returns. Marine terminals and hinterland depots become choked with empty equipment that must be sorted, inspected, staged, and repositioned without generating direct freight revenue.
- **Corridor congestion and carbon penalties:** Over-reliance on long-haul road drayage burns highway capacity, generates severe diesel emissions in port-adjacent environmental justice communities, and exposes cargo owners to escalating carbon taxes and roadway tolls.

Operations engineers address these challenges by treating the hinterland not as a collection of independent transport modes, but as an integrated, synchromodal network of dedicated on-dock rail facilities, inland waterways, and directional dry port hubs.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Model** seaport-hinterland container flows using synchromodal network optimization, evaluating dynamic mode substitution between road, rail, and inland waterways under infrastructure disruptions.
2. **Dimension** Dry Port and Inland Container Depot (ICD) directional hub networks, determining optimal inland staging distance, track lengths, storage buffers, and in-bond customs clearance flows.
3. **Evaluate** the economic and operational trade-offs of dedicated on-dock rail vs. near-dock vs. off-dock drayage transfer facilities using total cost, handling time, and break-even distance equations.
4. **Formulate** inland waterway push-barge convoy operations, calculating payload derating under dynamic river draft restrictions (low-water gauges), lockage queuing delays ($M/G/1$), and transit fuel consumption.
5. **Estimate** hinterland modal split using Multinomial Logit (MNL) and Nested Logit formulations, deriving direct and cross-elasticities with respect to freight rates, transit times, and schedule reliability.
6. **Formulate** chassis pool inventory rebalancing models across neutral, gray, and carrier-dedicated pool regimes, minimizing deadhead chassis split miles and terminal chassis dwell.
7. **Optimize** closed-loop empty container repositioning (repo) cycles and triangulation ("street turns") using multi-commodity network flow formulations.
8. **Synthesize** corridor capacity throttling mechanisms (time-of-day pricing, gate appointment quotas, train path reservation windows) to prevent shockwave propagation into marine terminal storage yards.

---

## 3. The core mental model

The hinterland network behaves as a multi-stage hydrodynamic conveyance network. High-pressure volumetric surges generated at the marine quayside pass through specialized transfer valves (on-dock rail intermodal yards, barge quays, and gate complexes) into parallel transport conduits of varying speed, capacity, and impedance, discharging into regional distribution hubs.

```text
=============================================================================================================
                      SEAPORT-HINTERLAND SYNCHROMODAL TRANSPORT NETWORK
=============================================================================================================

 [ DEEP-SEA QUAYSIDE ]
         │ (ULCV Discharge: 4,000 - 8,000 TEU pulse / 48 hrs)
         ▼
 [ MARINE TERMINAL YARD ] ──── (Dwell Buffer: Storage Blocks & Transfer Aisles)
         │
         ├───────────────────────────────┬───────────────────────────────┐
         │                               │                               │
         ▼ (Direct Drayage)              ▼ (Internal Rail Transfer)      ▼ (Barge Quay Spreader)
 [ IN/OUT ROAD GATES ]           [ ON-DOCK INTERMODAL (ICTF) ]   [ INLAND BARGE BERTH ]
         │ (TAS Appointment Quotas)      │ (Reach Stacker / RMG)         │ (Mobile Harbor Crane)
         │ (OCR & Scale Inspection)      │ (Spur Tracks: 500-1200m)      │ (Convoy Loading)
         │                               │                               │
         │ Highway Drayage               │ Unit Train Corridor           │ River / Canal Nav
         │ (High Cost, High Speed,       │ (Med Cost, Med Speed,         │ (Low Cost, Low Speed,
         │  Low Batch Size: 1-2 TEU)     │  Batch: 150-500 TEU)          │  Batch: 90-400 TEU)
         │                               │                               │
         ▼                               ▼                               ▼
 [ HIGHWAY CORRIDOR ]            [ FREIGHT RAIL CORRIDOR ]       [ INLAND WATERWAY ]
   (Congestion Shockwaves)         (Signaling Headway / DFC)       (River Gauges & Locks)
         │                               │                               │
         └───────────────────────────────┼───────────────────────────────┘
                                         ▼
                   [ DRY PORT / INLAND CONTAINER DEPOT (ICD) ]
                     - In-Bond Customs Clearance (Bypass Port)
                     - Intermodal Rail / Barge Ramps
                     - Empty Container Depots & Triangulation
                     - Neutral / Gray Chassis Pools
                                         │
                                         ▼ (Local Drayage Radius: < 50 km)
                   [ REGIONAL DISTRIBUTION CENTERS / FACTORIES ]
=============================================================================================================
```

### The Three Spatial Echelons of Hinterland Transport

1. **The Terminal Interface Echelon:** The physical junction point where containers transfer between marine terminal yard tractors and inland carriers. Operational efficiency is governed by handling density (crane lifts per hour), yard layout (wheeled vs. grounded), and spatial separation (on-dock rail within terminal security boundaries vs. off-dock rail yards requiring local highway drayage).
2. **The Line-Haul Trunk Corridor Echelon:** High-capacity transport arteries connecting the coastal gateway to the continental interior. Trunk corridors achieve unit cost reductions through mass aggregation:
   - *Double-Stack Unit Trains:* Transporting 200 to 500 TEU per train over lengths up to 3,000 meters (10,000 feet) in North America, or 740-meter trains in Europe.
   - *Inland Push-Barge Convoys:* Moving 90 to 600 TEU per tow along major river systems (e.g., the Rhine, Danube, Mississippi, Yangtze).
   - *Dedicated Freight Corridors (DFC):* Grade-separated rail infrastructures that decouple heavy freight traffic from passenger networks (e.g., the Betuweroute in the Netherlands or the Alameda Corridor in California).
3. **The Inland Staging & Triangulation Echelon (Dry Ports):** An inland facility directly connected to a seaport by high-capacity scheduled shuttles, where shippers leave and collect containers as if at the seaport itself. By hosting customs clearance, container maintenance, empty storage, and chassis pooling, the dry port moves the land-side operational buffer outside high-cost metropolitan coastal real estate.

---

## 4. Definitions and vocabulary

| Term | Category | Definition | Common Misuse / Operational Trap | Standard / Unit |
| :--- | :--- | :--- | :--- | :--- |
| **Intermodal Freight** | Transport | Freight movement in an ISO container using multiple modes without handling cargo during mode changes. | Conflating intermodal (unbroken box) with multimodal (cargo handled/transloaded between modes). | TEU / FEU / Box |
| **Hinterland** | Geography | The inland market territory served by a seaport for import distribution and export cargo generation. | Viewing hinterland as a static radius; hinterland boundaries fluctuate dynamically with modal freight rates. | Kilometers ($km$) / Zone |
| **Dry Port** | Infrastructure | An inland terminal directly connected to a seaport by rail/barge, offering full customs clearance and storage. | Treating an inland warehouse park as a dry port; a true dry port requires intermodal rail/barge shuttles and customs. | UN/LOCODE / ISO depot |
| **Inland Container Depot (ICD)** | Infrastructure | A common-user inland facility with public customs bond status offering handling and temporary storage of containers. | Confusing an ICD with a private freight forwarder cross-dock facility. | TEU capacity |
| **On-Dock Rail** | Infrastructure | Intermodal rail tracks located directly inside the marine container terminal footprint, eliminating highway drayage. | Underestimating footprint cost; rail tracks consume valuable ground storage space for marine containers. | Track meters ($m$) / TEU |
| **Near-Dock Rail** | Infrastructure | Intermodal rail yard adjacent to the port boundary, requiring short off-highway or dedicated arterial drayage moves. | Assuming near-dock avoids drayage costs; still requires truck tractors, chassis, and two gate transactions. | Road distance ($km$) |
| **Off-Dock ICTF** | Infrastructure | Intermodal Container Transfer Facility located 5 to 35 km inland from the port, linked by public highway corridors. | Ignoring gate queuing and highway congestion variance in line-haul train cutoff calculations. | Kilometers ($km$) |
| **Drayage** | Operations | Short-distance transport of containerized freight by truck between ocean terminals, rail ramps, and warehouses. | Treating drayage as long-haul trucking; drayage economics depend on gate turn times, not road fuel efficiency. | USD / move / turn-time |
| **Split Move (Chassis Split)** | Operations | A drayage maneuver where a driver must pick up or return a bare chassis at a facility separate from the container depot. | Overlooking split moves in trip dispatching; adds 45–90 minutes of uncompensated driver time and deadhead miles. | Driver minutes / miles |
| **Chassis** | Equipment | A specialized skeletal semi-trailer engineered to securely mount and haul ISO containers on public highways. | Assuming container and chassis are a single asset; in North America, they are owned and billed independently. | 20ft / 40ft / 53ft unit |
| **Neutral Chassis Pool** | Asset Pooling | A chassis fleet managed by a third-party leasing company, open to motor carriers serving multiple ocean carriers. | Believing neutral pools eliminate billing disputes; box-to-chassis pairing mismatches cause audit friction. | Pool fleet count |
| **Gray Chassis Pool** | Asset Pooling | A fully interoperable cooperative pool where any chassis can be used beneath any container line's box without penalty. | Assuming total interchangeability without inter-pool balancing compensation agreements. | Interoperability index |
| **Push-Barge Convoy** | Waterway | A motorized towboat pushing rigid configurations of non-powered barges coupled tightly by steel deck cables. | Modeling push-barges as standard ships; hydrodynamics depend heavily on shallow-water canal bank suction effects. | Number of barges / TEU |
| **Kaub Gauge** | Hydrodynamics | Critical reference river gauge on the Middle Rhine that dictates max vessel draft and loading capacity for European barges. | Planning barge booking based on ocean terminal water depth rather than shallowest inland river constriction gauge. | Centimeters ($cm$) |
| **Low-Water Surcharge (KWZ)** | Economics | Tiered freight surcharge applied by barge operators when river water levels fall below published gauge depths. | Failing to budget KWZ during summer/autumn drought periods, triggering 50% to 150% freight rate spikes. | USD or EUR / TEU |
| **Lock Cycle Time** | Waterway | Total time required for a vessel to enter, fill/empty, and exit a navigation lock chamber. | Assuming lock service times are deterministic; barge convoy decoupling and re-coupling cause heavy queuing tails. | Minutes / hours |
| **Modal Split Logit Model** | Econometrics | Discrete choice model predicting market share proportions across road, rail, and water modes based on utility. | Assuming linear price response; logit models have non-linear sigmoidal response curves governed by the IIA property. | Probability $P \in [0, 1]$ |
| **Value of Time (VOT)** | Econometrics | The monetary valuation shippers place on saving one hour of cargo transit time ($\beta_{time} / \beta_{cost}$). | Using a single fleet-wide VOT; high-value electronics have 10x the VOT of bulk agricultural scrap. | USD / hour / TEU |
| **Value of Reliability (VOR)** | Econometrics | The monetary valuation shippers place on reducing transit time variance or increasing on-time arrival rate. | Focusing solely on transit speed; empirical studies prove VOR often exceeds VOT for supply chain planners. | USD / % on-time / TEU |
| **Direct Price Elasticity** | Econometrics | Percentage change in a transport mode's demand resulting from a 1% change in its own freight rate. | Assuming freight demand is elastic; captive inland shippers often face highly inelastic transport choices. | Dimensionless ratio |
| **Cross-Price Elasticity** | Econometrics | Percentage change in a transport mode's demand resulting from a 1% price change in a competing transport mode. | Ignoring modal capacity ceilings when modeling cross-elasticity demand shifts from rail to highway. | Dimensionless ratio |
| **Triangulation (Street Turn)** | Operations | Direct dispatch of an empty import container to an export shipper for reloading, bypassing the marine terminal yard. | Attempting street turns without line-carrier approval; lines reserve boxes for high-yield reefer or dry export lanes. | % matched boxes |
| **Empty Repo (Repositioning)** | Operations | Logistics of moving empty non-revenue containers from surplus deficit consumption areas back to deficit export nodes. | Treating empty boxes as zero-cost inventory; repositioning consumes up to 25% of global liner vessel slots. | TEU / slot cost |
| **Corridor Metering / Throttling**| Network Flow | Dynamic restriction of freight volume admitted to an inland corridor to prevent demand exceeding physical capacity. | Throttling without buffer storage; blocking corridor ingress without staging areas triggers upstream gridlock. | TEU / hour / lane |
| **PierPass / Off-Peak Fee** | Traffic Mitigation | Traffic mitigation fee levied on weekday peak container gate moves at Los Angeles / Long Beach to incentivize off-peak hours. | Treating the fee as revenue rather than a peak-spreading incentive mechanism. | USD / TEU |
| **UIIA** | Regulatory | Uniform Intermodal Interchange Agreement: standard contract governing equipment exchange between carriers and draymen. | Ignoring UIIA insurance and maintenance clauses when dispatching motor carriers to marine terminals. | Industry Standard Contract |
| **In-Bond Transit (CBP 7512)** | Customs | U.S. customs procedure allowing freight to move inland without paying duties or undergoing final clearance at the port of entry. | Failing to secure bond authorization before dispatching train, resulting in customs holds at rail ramps. | Customs Bond Status |

---

## 5. The mechanics, step by step

Hinterland logistics operations engineering requires a mathematically rigorous procedure spanning corridor dimensioning, on-dock rail economics, inland waterway draft calculations, discrete choice modal split modeling, chassis balancing, and corridor throttling.

```text
+---------------------------------------------------------------------------------------------------+
|                           HINTERLAND OPERATIONS ENGINEERING PIPELINE                             |
+---------------------------------------------------------------------------------------------------+
  [Step 1] Network Topology & ICD Sizing
           ├── Dry port classification (Close vs. Mid-range vs. Distant)
           └── Minimum cost location and capacity sizing
  [Step 2] Dedicated On-Dock Rail vs. Off-Dock Drayage Economics
           ├── Analytical cost curves per TEU ($C_{on-dock}$ vs. $C_{off-dock}$ vs. $C_{road}$)
           └── Break-even distance threshold ($d^*$) derivation
  [Step 3] Inland Waterway Convoy Hydrodynamics & Lock Queuing
           ├── River gauge draft derating equation ($T_{barge} = D_{channel} - UKC_{min} - \Delta$)
           └── Lock chamber queuing service model ($M/G/1$)
  [Step 4] Multinomial Logit Modal Split Formulation
           ├── Systematic utility functions ($V_{im} = \alpha_m + \beta_C C_{im} + \beta_T T_{im} + \dots$)
           ├── Direct and cross-price elasticity derivations
           └── Value of Time (VOT) and Value of Reliability (VOR) estimation
  [Step 5] Chassis Pool Inventory Balancing & Split Move Optimization
           ├── Space-time network flow formulation for chassis repositioning
           └── Carrier vs. Neutral vs. Gray pool mechanics
  [Step 6] Closed-Loop Empty Container Repositioning & Street Turns
           ├── Multi-commodity inventory balance equations
           └── Triangulation match-rate optimization
  [Step 7] Corridor Capacity Throttling & Shockwave Mitigation
           ├── Hydrodynamic flow conservation equation ($q = k \cdot v(k)$)
           └── Peak gate appointment quota and off-peak fee sizing
+---------------------------------------------------------------------------------------------------+
```

### 5.1 Step 1: Network Topology and Dry Port (ICD) Sizing

A Dry Port is an inland intermodal terminal directly connected to one or more seaports via high-capacity, scheduled rail shuttle or barge services, where customers can leave and pick up their standardized units as if directly at a seaport.

Dry ports fall into three distinct functional categories (Roso et al. typology):
1. **Close Dry Port ($< 50 \text{ km}$ from port):** Designed primarily to relieve quayside gate congestion, provide buffer storage, and mitigate urban roadway choke points. Containers are evacuated directly off the vessel via short-shuttle rail or barge convoys without customs clearance at the quay. Customs clearance and cargo staging occur at the dry port.
2. **Mid-Range Dry Port ($50 - 300 \text{ km}$ from port):** Serves as a regional consolidation and distribution hub. It combines local drayage radius operations with unit train block marshalling, hosting empty container depots and light value-added logistics (cross-docking, palletizing).
3. **Distant Dry Port ($> 300 \text{ km}$ from port):** Serves an inland industrial or agricultural hinterland where the dominant line-haul transport is heavy intermodal rail. The distant dry port acts as the primary freight gateway, capturing entire regional catchment areas and shifting freight modal share from long-haul road to rail.

#### Mathematical Formulation of Dry Port Sizing and Location

Let $J$ be the set of candidate dry port locations, $I$ the set of seaport container terminals, and $K$ the set of inland shippers/consignees. The objective is to minimize total network logistics cost:

$$\min Z = \sum_{i \in I} \sum_{j \in J} c_{ij}^{shuttle} X_{ij} + \sum_{j \in J} \sum_{k \in K} c_{jk}^{dray} Y_{jk} + \sum_{j \in J} f_j Z_j + \sum_{j \in J} h_j S_j + \sum_{i \in I} \sum_{k \in K} c_{ik}^{direct} W_{ik}$$

Subject to:
- **Flow Conservation:**
  $$\sum_{i \in I} X_{ij} = \sum_{k \in K} Y_{jk} \quad \forall j \in J$$
- **Customer Demand Satisfaction:**
  $$\sum_{j \in J} Y_{jk} + \sum_{i \in I} W_{ik} = D_k \quad \forall k \in K$$
- **Dry Port Capacity Limits:**
  $$\sum_{i \in I} X_{ij} \le Cap_j \cdot Z_j \quad \forall j \in J, \quad Z_j \in \{0, 1\}$$
  $$S_j = \bar{t}_{customs} \cdot \left( \frac{\sum_{i \in I} X_{ij}}{365} \right) \cdot \theta_{peak}$$
  Where $\bar{t}_{customs}$ is the average inland bond dwell time (days), and $\theta_{peak}$ is the design peaking factor (typically 1.25 to 1.40).

#### Dry Port Yard Buffer Sizing via Little's Law

The required static ground slot storage capacity $S_{slots}$ (TEU slots) at the dry port is dimensioned by coupling Little's Law with dwell time variance and stacking thermodynamics:

$$S_{slots} = \left\lceil \frac{\lambda_{inland} \cdot \bar{D}_{dwell}}{H_{stack} \cdot \eta_{util}} \cdot \left( 1 + z_{1-\alpha} \cdot \frac{\sigma_{dwell}}{\bar{D}_{dwell}} \right) \right\rceil$$

Where:
- $\lambda_{inland}$: Average daily container arrival throughput ($\text{TEU/day}$).
- $\bar{D}_{dwell}$: Mean container dwell time at the dry port (typically 4.0 to 7.0 days for in-bond customs clearance).
- $H_{stack}$: Mean stacking height of containers in the storage yard ($H_{stack} \approx 3.0 - 4.5 \text{ tiers}$ under RTG or RMG operation).
- $\eta_{util}$: Maximum allowable operational yard utilization factor ($\eta_{util} \le 0.75$). Exceeding 75% triggers the exponential re-handling penalty modeled in Topic 02.
- $z_{1-\alpha}$: Standard normal quantile for service reliability (e.g., $z_{0.95} = 1.645$).
- $\sigma_{dwell}$: Standard deviation of container dwell time.

#### In-Bond Customs Clearance Architecture

A primary economic driver of the dry port model is shifting customs inspection and tariff assessment away from congested coastal metropolitan land. Under regulatory frameworks such as U.S. Customs and Border Protection (CBP) 19 CFR Part 18 (Immediate Transportation - IT In-Bond) or European Union Union Customs Code (UCC) Transit Procedure (T1 Document), containers move from the quayside directly onto block trains under an electronic bond without paying duties or undergoing physical cargo inspection at the marine terminal. 

The container manifest data is transmitted electronically via EDI 315 / CBP ACE In-Bond filing before vessel discharge. Customs holds are transferred to the destination ICD, eliminating the 4.0-to-8.0-day dwell bottleneck at the maritime quay and reducing coastal terminal footprint requirements by 30% to 50%.

### 5.2 Step 2: Dedicated On-Dock Rail vs. Off-Dock Drayage Economics

The transfer of containers from a marine vessel to an intermodal train occurs across three competing physical topologies:
- **On-Dock Rail:** Intermodal working tracks lie inside the marine terminal fence. Yard hostlers transfer containers directly from the vessel stack to the rail car. No public highway drayage or external gate transactions occur.
- **Near-Dock Rail:** Intermodal yard is adjacent to the port boundary (1 to 5 km). Containers move via dedicated port access roads or short drayage shuttles.
- **Off-Dock ICTF:** Intermodal terminal is located 10 to 40 km inland. Containers must be mounted onto highway chassis, clear marine terminal out-gates, navigate urban highway corridors, and clear ICTF in-gates.

```text
+---------------------------------------------------------------------------------------------------+
|                              ON-DOCK VS. OFF-DOCK COST COMPOSITION                                |
+---------------------------------------------------------------------------------------------------+
Cost ($/TEU)
  ^
  |                                        Direct Highway Drayage: C_road(d)
  |                                      /
  |                                     /
  |                                    /      Off-Dock Rail: C_off-dock(d)
  |                                   /     /
  |                                  /    /   On-Dock Rail: C_on-dock(d)
  |                                 /   /   /
  |                                /  /   /
  |  C_fixed,off  --------------->+ /   /
  |                              / +   /
  |                             /   \ /
  |  C_fixed,on   ------------>+     X  <--- Break-even distance (d*)
  |                           /     / \
  |                          /     /   \
  |  0 ---------------------+-----+-----+--------------------------------------------> Distance (km)
  |                         0    d_truck  d*
+---------------------------------------------------------------------------------------------------+
```

#### Analytical Cost Formulation

1. **On-Dock Rail Total Cost Function ($/TEU):**
   $$C_{on\text{-}dock}(d) = C_{lift,vessel} + C_{trans,yard} + C_{lift,rail} + C_{switch} + C_{track\text{-}rent} + (c_{rail} \cdot d)$$
   Where:
   - $C_{lift,vessel}$: STS crane discharge lift cost (\$25–\$45/TEU).
   - $C_{trans,yard}$: Internal terminal tractor move from quay to rail yard (\$15–\$25/TEU).
   - $C_{lift,rail}$: Rail mounted gantry (RMG) or reach stacker lift onto rail car (\$30–\$50/TEU).
   - $C_{switch}$: Railcar switching, block classification, and locomotive makeup fee (\$40–\$75/TEU).
   - $C_{track\text{-}rent}$: Marine terminal footprint capital recovery surcharge (\$15–\$30/TEU).
   - $c_{rail}$: Line-haul rail transport cost per TEU-km (\$0.035–\$0.065/TEU-km).
   - $d$: Line-haul distance in kilometers.

2. **Off-Dock Rail Total Cost Function ($/TEU):**
   $$C_{off\text{-}dock}(d) = C_{lift,vessel} + C_{trans,yard} + C_{gate,port} + C_{dray,local} + C_{chassis,split} + C_{gate,ictf} + C_{lift,ictf} + (c_{rail} \cdot d)$$
   Where:
   - $C_{gate,port} + C_{gate,ictf}$: Gate OCR, scale, and inspection transaction fees (\$20–\$40 combined).
   - $C_{dray,local}$: Local highway drayage move to off-dock ICTF (\$120–\$250/move).
   - $C_{chassis,split}$: Expected cost of deadhead bare chassis collection/return (\$30–\$65/move).
   - $C_{lift,ictf}$: Intermodal lift at inland ICTF ramp (\$35–\$55/TEU).

3. **Direct Highway Trucking Cost Function ($/TEU):**
   $$C_{road}(d) = C_{lift,vessel} + C_{trans,yard} + C_{gate,port} + C_{chassis} + (c_{truck} \cdot d)$$
   Where $c_{truck}$ is the over-the-road trucking cost per TEU-km (\$1.20–\$2.10/TEU-km).

#### Derivation of Break-Even Distance ($d^*$)

Setting $C_{on\text{-}dock}(d^*) = C_{off\text{-}dock}(d^*)$ demonstrates that on-dock rail saves a fixed operational premium per container $\Delta C_{fixed} = C_{off\text{-}dock,fixed} - C_{on\text{-}dock,fixed} \approx \$110 - \$220 \text{ /TEU}$ on every box by eliminating highway drayage and dual gate processing.

Comparing rail against direct highway trucking yields the classic modal break-even distance:

$$d^* = \frac{C_{rail,fixed} - C_{road,fixed}}{c_{truck} - c_{rail}}$$

In North American corridors, double-stack rail achieves break-even against road drayage at $d^* \approx 650 - 850 \text{ km}$. In European corridors with single-stack electric traction, break-even occurs at $d^* \approx 250 - 400 \text{ km}$.

#### Train Resistance and Traction Dynamics

The economic viability of line-haul rail intermodal transport is governed by tractive effort, locomotive horsepower, and train resistance. Total rolling resistance $R_{total}$ (pounds of force) of an intermodal train is calculated using the empirical Canadian National / Davis equation:

$$R_{total} = \sum_{w=1}^{N_{cars}} \left( 1.5 + \frac{72.5}{W_w} + 0.015 \cdot v + \frac{C_{drag} \cdot A \cdot v^2}{10,000 \cdot W_w} \right) \cdot W_w + 20 \cdot G \cdot W_{train} + 0.8 \cdot D_c \cdot W_{train}$$

Where:
- $W_w$: Gross mass per railcar (tons).
- $v$: Operating speed (mph).
- $C_{drag}$: Aerodynamic drag coefficient ($C_{drag} \approx 0.0005$ for empty well cars, $0.00035$ for fully loaded double-stack containers).
- $G$: Track gradient (percent slope). Grade resistance imposes a severe penalty: exactly $20 \text{ lbs per ton}$ of train weight per 1.0% of ruling grade.
- $D_c$: Track curvature in degrees ($0.8 \text{ lbs per ton}$ per degree of curve).
- $W_{train}$: Total trailing gross train weight (tons).

To maintain scheduled line-haul speed across undulating topography, intermodal operations engineers specify a strict Horsepower-per-Ton ($HPT$) ratio:

$$HPT = \frac{\sum P_{locomotive}}{W_{train}}$$

Double-stack intermodal corridor trains operating on flat coastal corridors (e.g., Chicago to New York) operate at $HPT \approx 1.2 - 1.5 \text{ HP/ton}$, while transcontinental trains crossing the Rocky Mountains (e.g., BNSF Southern Transcon) require $HPT \ge 2.2 - 3.0 \text{ HP/ton}$ with distributed power (DPU) remote locomotives cut into the mid-train to mitigate extreme in-train coupler drawbar tension forces.
### 5.3 Step 3: Inland Waterway Barge Convoy Hydrodynamics and Lock Queuing

Inland navigation operates under rigid hydrodynamic physical laws. Unlike deep-sea ocean vessels, inland barges operate in shallow channels where bed clearance and bank suction dictate maximum cargo displacement.

#### Hydrodynamic River Draft Derating

The permissible navigation draft of a container barge $T_{allow}$ is strictly bounded by the water depth in the shallowest section of the fairway:

$$T_{allow} = D_{chart} + \Delta h_{gauge} - UKC_{min} - \Delta_{squat}$$

Where:
- $D_{chart}$: Charted channel depth at reference low water datum (meters).
- $\Delta h_{gauge}$: Current river water level reported at the critical reference gauge (e.g., Kaub Gauge on the Rhine, Ruhrort on the Lower Rhine, or St. Louis on the Mississippi).
- $UKC_{min}$: Mandatory minimum static Under-Keel Clearance (typically $0.30 - 0.50 \text{ m}$ in inland canals and rocky riverbeds).
- $\Delta_{squat}$: Dynamic vessel squat induced by shallow-water Bernoulli drawdown:
  $$\Delta_{squat} = C_B \cdot \frac{V_barge^2}{100} \cdot \frac{A_M}{A_C - A_M}$$
  Where $C_B$ is the barge block coefficient, $V_barge$ is navigation speed relative to the water ($\text{m/s}$), $A_M$ is vessel midship cross-sectional area, and $A_C$ is channel cross-sectional area.

When water level $\Delta h_{gauge}$ drops, barge operators cannot load containers to maximum stack height. For a standard Jowi-class Rhine container vessel ($110 \text{ m} \times 11.45 \text{ m}$, max draft $3.20 \text{ m}$, nominal capacity 398 TEU):
- At normal water ($\Delta h_{gauge} \ge 2.50 \text{ m}$): Draft is $3.00 \text{ m}$; payload capacity is $398 \text{ TEU}$ (100% capacity).
- At low water ($\Delta h_{gauge} = 1.20 \text{ m}$): Allowable draft drops to $1.90 \text{ m}$; payload capacity drops to $180 \text{ TEU}$ (45% capacity).
- At extreme drought ($\Delta h_{gauge} \le 0.60 \text{ m}$): Allowable draft drops to $1.40 \text{ m}$; payload capacity collapses to under $80 \text{ TEU}$ (20% capacity), triggering massive modal diversion to rail and road.

#### Lock Chamber Queuing Formulation

Inland waterways rely on lock chambers to navigate elevation gradients. A navigation lock acts as a batch service queuing system. 

Let barge arrivals follow a Poisson process with arrival rate $\lambda$ (convoys per hour). The lock chamber has a physical length $L_{lock}$ and width $W_{lock}$. Convoys are packed into the chamber up to area limits. The cycle time of a lock chamber includes vessel entry time, gate closure, chamber filling/emptying, gate opening, and vessel exit:

$$T_{service} = t_{enter} + t_{close} + t_{hydraulic} + t_{open} + t_{exit}$$

Assuming general independent service times with mean $\bar{T}_s = 1/\mu$ and service time variance $\sigma_s^2$, the average convoy delay in the lock approach pool is modeled via the Pollaczek-Khinchine $M/G/1$ formula:

$$W_q = \frac{\lambda (\sigma_s^2 + \bar{T}_s^2)}{2 (1 - \rho)} = \frac{\rho \bar{T}_s}{2(1 - \rho)} \left(1 + C_s^2 \right)$$

Where $\rho = \lambda \bar{T}_s$ is lock utilization and $C_s^2 = \sigma_s^2 / \bar{T}_s^2$ is the squared coefficient of variation of service time. As lock utilization $\rho$ exceeds 0.85, barge waiting times explode non-linearly, destroying barge schedule reliability.

#### Hydrodynamics of Restricted Inland Waterways (Schijf's Formulation)

When a push-barge convoy navigates a restricted inland canal or river reach, the water displaced by the moving hull must flow backward through the constricted cross-section between the hull and the canal banks. This return flow velocity $\Delta u$ creates a Bernoulli pressure drop that causes the water level around the vessel to depress by $\Delta h$, sinking the vessel closer to the riverbed.

According to Schijf's one-dimensional conservation equations:

$$V_{barge} \cdot A_M = (V_{barge} + \Delta u) \cdot (A_C - A_M - B \cdot \Delta h)$$

$$\Delta h = \frac{1}{2g} \left[ (V_{barge} + \Delta u)^2 - V_{barge}^2 \right]$$

Where:
- $V_{barge}$: Ship speed relative to channel bank ($\text{m/s}$).
- $A_M$: Midship submerged cross-sectional area ($B \cdot T$, $\text{m}^2$).
- $A_C$: Undisturbed canal cross-sectional area ($\text{m}^2$).
- $B$: Convoy beam width ($\text{m}$).
- $\Delta u$: Return current velocity ($\text{m/s}$).
- $\Delta h$: Water level depression ($\text{m}$).

If return flow velocity $\Delta u$ approaches critical shallow-water wave speed $c = \sqrt{g D_{channel}}$, resistance diverges, steering control fails, and dynamic squat forces bottom grounding. Consequently, inland waterway engineers enforce strict convoy speed ceilings ($V_{barge} \le 0.70 \sqrt{g D_{channel}}$) in shallow reaches.

| Convoy Class | Configuration | Dimensions ($L \times B$) | Max TEU Capacity | Typical Fairway |
| :--- | :--- | :--- | :--- | :--- |
| **CEMT Class IV** | Single motorized barge (Johann Welker) | $85.0 \text{ m} \times 9.5 \text{ m}$ | $81 - 108 \text{ TEU}$ | European regional canals (e.g., Albert Canal) |
| **CEMT Class Va** | Large Rhine container ship | $110.0 \text{ m} \times 11.45 \text{ m}$ | $200 - 398 \text{ TEU}$ | Middle Rhine, Upper Rhine, Waal |
| **CEMT Class VIb** | Coupled 4-barge push convoy | $185.0 \text{ m} \times 22.8 \text{ m}$ | $450 - 640 \text{ TEU}$ | Lower Rhine (Rotterdam to Duisburg) |
| **Mississippi Tow** | 15–30 barge dry tow with pusher | $300.0 \text{ m} \times 32.0 \text{ m}$ | $600 - 1,200 \text{ TEU}$ (equiv) | Lower Mississippi River (New Orleans to Memphis) |

### 5.4 Step 4: Modal Split Discrete Choice Logit Modeling

Shippers and intermodal marketing companies (IMCs) choose among competing hinterland transport modes (Road, Rail, Barge) based on Random Utility Maximization (RUM). 

The utility $U_{im}$ of transport mode $m \in M$ for shipment $i$ consists of a deterministic systematic component $V_{im}$ and an unobserved stochastic error term $\varepsilon_{im}$:

$$U_{im} = V_{im} + \varepsilon_{im}$$

Assuming $\varepsilon_{im}$ are independently and identically distributed (i.i.d.) Gumbel (Extreme Value Type I) random variables, the probability $P_{im}$ that mode $m$ is selected follows the **Multinomial Logit (MNL)** model:

$$P_{im} = \frac{\exp(V_{im})}{\sum_{k \in M} \exp(V_{ik})}$$

#### Systematic Utility Function

$$V_{im} = \alpha_m + \beta_{cost} \cdot C_{im} + \beta_{time} \cdot T_{im} + \beta_{rel} \cdot Rel_{im} + \beta_{freq} \cdot F_{im}$$

Where:
- $\alpha_m$: Alternative-Specific Constant (ASC) capturing unobserved modal preferences (typically $\alpha_{road} = 0$ as reference).
- $C_{im}$: Total freight cost per TEU including drayage, rail/barge freight, and handling fees (\$).
- $T_{im}$: Door-to-door transit time (hours).
- $Rel_{im}$: Schedule reliability metric, measured as the percentage of shipments delivered within the contractual delivery window ($0 \le Rel_{im} \le 1$).
- $F_{im}$: Service departure frequency (departures per week).
- $\beta_{cost} < 0$, $\beta_{time} < 0$, $\beta_{rel} > 0$, $\beta_{freq} > 0$: Empirically estimated marginal utility parameters.

#### Derivation of Elasticities and Valuation Ratios

1. **Direct Cost Elasticity of Mode $m$:**
   $$E_{C_m}^{P_m} = \frac{\partial P_m}{\partial C_m} \cdot \frac{C_m}{P_m} = \beta_{cost} \cdot C_m \cdot (1 - P_m)$$
2. **Cross-Cost Elasticity (change in share of mode $m$ when mode $k$'s price changes):**
   $$E_{C_k}^{P_m} = \frac{\partial P_m}{\partial C_k} \cdot \frac{C_k}{P_m} = -\beta_{cost} \cdot C_k \cdot P_k$$
3. **Value of Time (VOT, \$/hour):**
   The marginal rate of substitution between transit time and freight cost:
   $$VOT = \frac{\partial V / \partial T}{\partial V / \partial C} = \frac{\beta_{time}}{\beta_{cost}}$$
4. **Value of Reliability (VOR, \$ per unit of reliability):**
   $$VOR = -\frac{\partial V / \partial Rel}{\partial V / \partial C} = -\frac{\beta_{rel}}{\beta_{cost}}$$

> **Note:** The standard MNL model exhibits the Independence of Irrelevant Alternatives (IIA) property: the ratio of probabilities between any two modes is independent of the attributes of a third mode ($P_m / P_n = \exp(V_m) / \exp(V_n)$). To eliminate red-bus/blue-bus substitution bias between rail and barge, operations engineers employ a **Nested Logit** formulation, grouping Rail and Barge into an "Intermodal" nest with an inclusive value parameter $\mu \in (0, 1]$.

#### Overcoming IIA: The Nested Logit Formulation

To account for the shared unobserved characteristics between Rail and Barge (e.g., fixed timetable inflexibility, susceptibility to network bottlenecks, common reliance on terminal transfer spreaders), we partition the choice set into two nests:
- Nest 1: Direct Road Trucking ($m \in \{\text{Road}\}$)
- Nest 2: Intermodal Transport ($m \in \{\text{Rail}, \text{Barge}\}$)

The probability of choosing mode $m$ in nest $k$ decomposes into the product of the marginal probability of choosing nest $k$ and the conditional probability of choosing mode $m$ given nest $k$:

$$P_{im} = P(k) \cdot P(m \mid k)$$

The conditional choice probability within the Intermodal nest is:

$$P(m \mid \text{Intermodal}) = \frac{\exp(V_{im} / \mu)}{\sum_{j \in \text{Intermodal}} \exp(V_{ij} / \mu)}$$

Where $\mu \in (0, 1]$ is the **inclusive value (log-sum) parameter**, reflecting the degree of independence among unobserved error terms within the nest. When $\mu = 1.0$, the Nested Logit collapses to the standard MNL model. As $\mu \to 0$, modes within the nest become perfect substitutes.

The inclusive value (expected maximum utility) of the Intermodal nest is:

$$I_{\text{Intermodal}} = \ln \left( \sum_{j \in \text{Intermodal}} \exp(V_{ij} / \mu) \right)$$

The upper-level marginal choice probability for nest $k$ is:

$$P(k) = \frac{\exp(\mu_k \cdot I_k)}{\sum_{\ell} \exp(\mu_{\ell} \cdot I_{\ell})}$$

For the degenerated singleton Road nest, $\mu_{road} = 1.0$ and $I_{road} = V_{road}$. Under this nested structure, a price change in rail shifts disproportionately more demand to barge than to road, accurately capturing physical intermodal cross-substitution without IIA distortion.

### 5.5 Step 5: Chassis Pool Management and Split Move Optimization

In containerized surface transport, a container cannot move over public roads without an intermodal chassis. In North America, following ocean carrier chassis divestment (2009–2014), three distinct chassis management regimes emerged:

```text
+---------------------------------------------------------------------------------------------------+
|                               CHASSIS POOL OPERATING REGIMES                                      |
+---------------------------------------------------------------------------------------------------+
 1. CARRIER-DEDICATED POOL:
    [Ocean Carrier A] ──► Exclusive Chassis Fleet ──► Only Carrier A Boxes
    (High idle capital, frequent split moves, zero interoperability)

 2. NEUTRAL LEASING POOL:
    [Equipment Lessor] ──► Common Pool of Chassis ──► Multiple Shippers/Truckers
    (Market daily rental rates: $35-$55/day; box-line interchange rules apply)

 3. GRAY (INTEROPERABLE) COOPERATIVE POOL:
    [Pool of Pools (TRAC/DCLI/Flexi-Van)] ──► Universal Pool ──► Any Marine Terminal / Any Box
    (Eliminates chassis split moves; requires automated inter-pool usage credit clearing)
+---------------------------------------------------------------------------------------------------+
```

#### Split Move Mechanics

A split move occurs when a motor carrier dispatched to pick up an import container at Marine Terminal A must first travel bobtail (tractor without chassis) to off-dock Depot B to acquire an approved bare chassis, before driving to Terminal A to mount the container.

Let the probability of a chassis mismatch at the marine terminal gate be $P_{split}$. The expected drayage cycle time $E[T_{dray}]$ expands:

$$E[T_{dray}] = T_{linehaul} + T_{turn,port} + P_{split} \cdot \left( t_{travel,depot} + t_{queue,depot} + t_{inspect} \right)$$

Where depot travel and inspection adds 45 to 90 minutes of non-productive turn time per container drayage move.

#### Dynamic Chassis Pool Rebalancing Formulation

To prevent chassis deficits at export ramps and surpluses at import gates, chassis pool operators solve a dynamic multi-period network flow model. Let $N$ be nodes (ports, rail ramps, empty depots, customer sites) and $T$ planning time periods:

$$\min \sum_{t=1}^T \sum_{i \in N} \sum_{j \in N} c_{ij}^{rebal} \cdot x_{ijt} + \sum_{t=1}^T \sum_{i \in N} \left( h_i \cdot s_{it}^+ + p_i \cdot s_{it}^- \right)$$

Subject to:
- **Inventory Balance:**
  $$I_{i,t} = I_{i,t-1} + \sum_{j \in N} x_{ji, t-\tau_{ji}} - \sum_{j \in N} x_{ij,t} + \Delta_{it}^{organic} + s_{it}^- - s_{it}^+ \quad \forall i, t$$
Where $x_{ijt}$ is the number of bare chassis repositioned from node $i$ to node $j$ in period $t$, $\tau_{ij}$ is transfer transit time, $\Delta_{it}^{organic}$ is the net organic inflow/outflow of chassis driven by loaded container moves, $s_{it}^+$ is surplus chassis holding, and $s_{it}^-$ is unmet chassis demand penalized at rate $p_i$.

### 5.6 Step 6: Closed-Loop Empty Container Repositioning (Repo) & Triangulation

Global trade imbalances create severe structural container surpluses at inland consumption centers and chronic deficits at export production nodes.

```text
=============================================================================================================
                          EMPTY CONTAINER REPOSITIONING & TRIANGULATION
=============================================================================================================
 TRADITIONAL REPO CYCLE (4 Gate Moves, 2 Empty Highway Legs):
 [Seaport] ──(1) Import Loaded──► [Consignee] ──(2) Empty Dray──► [Inland Empty Depot]
                                                                        │
 [Seaport] ◄──(4) Export Loaded── [Shipper] ◄───(3) Empty Dray──────────┘

 TRIANGULATED "STREET TURN" CYCLE (2 Gate Moves, 0 Empty Highway Legs):
 [Seaport] ──(1) Import Loaded──► [Consignee]
                                       │
                                       ▼ (Direct Street Turn Drayage: < 20 km)
 [Seaport] ◄──(3) Export Loaded── [Shipper]
=============================================================================================================
```

#### Triangulation Match Feasibility Conditions

A direct container street turn between Import Consignee $A$ and Export Shipper $B$ is physically and commercially feasible if and only if all five boundary conditions are satisfied:

1. **Equipment Compatibility:**
   $$\text{Type}(Container_A) = \text{Type}(Container_B) \quad (\text{e.g., } 40\text{ft High-Cube } = 40\text{ft High-Cube})$$
2. **Carrier Line Ownership:**
   $$\text{Carrier}(Container_A) = \text{Carrier}(Container_B) \quad \text{or an authorized alliance interchange agreement exists.}$$
3. **Temporal Alignment:**
   $$t_{unpack,A} + t_{dray,AB} \le t_{cutoff,B} - t_{pack,B}$$
   $$t_{cutoff,B} - t_{discharge,A} \le \text{Max Free Time Allowed}$$
4. **Spatial Proximity:**
   $$\text{Dist}(A, B) \le \text{Dist}(A, Depot) + \text{Dist}(Depot, B) - \text{Threshold}$$
5. **Physical Condition:** The container must not require steam cleaning, structural repair, or odor remediation (foodstuff or dangerous goods export contamination rules).

When triangulation succeeds, it eliminates two empty drayage legs, two marine terminal gate transactions, and 120–250 kg of highway $\text{CO}_2$ emissions per container.

### 5.7 Step 7: Corridor Capacity Throttling and Shockwave Dynamics

Inland freight corridors behave under Lighthill-Whitham-Richards (LWR) kinematic wave theory. Traffic flow $q$ (TEU per hour) is the product of spatial density $k$ (TEU per lane-kilometer) and mean travel speed $v(k)$:

$$q = k \cdot v(k)$$

Using the Greenshields linear speed-density relationship $v(k) = v_f (1 - k/k_{jam})$:

$$q = v_f \cdot k \cdot \left(1 - \frac{k}{k_{jam}}\right)$$

Maximum corridor flow (critical capacity $q_{max}$) occurs at critical density $k_{crit} = k_{jam} / 2$:

$$q_{max} = \frac{v_f \cdot k_{jam}}{4}$$

#### Shockwave Back-Propagation into Seaport Terminals

When arrival flow into the hinterland corridor exceeds bottleneck discharge capacity ($q_{in} > q_{bottle}$), a backward-propagating congestion shockwave forms with velocity $w_{shock}$:

$$w_{shock} = \frac{q_{congested} - q_{upstream}}{k_{congested} - k_{upstream}} < 0$$

If the bottleneck is located at distance $L$ from the port gate, the shockwave reaches the marine terminal gate at time $T_{reach} = L / |w_{shock}|$. Once the wave strikes the gate, truck turn times double, in-gate queuing reservoirs spill onto municipal highways, and terminal operating systems (TOS) are forced to throttle gate appointments.

#### Sizing Mitigation Controls

1. **Gate Appointment Quotas ($N_{TAS}$):** Sizing hourly booking slots strictly below bottleneck discharge rate:
   $$N_{TAS}(t) \le \alpha \cdot q_{bottle} \quad (\text{with safety buffer } \alpha \approx 0.85 - 0.90)$$
2. **Off-Peak Traffic Mitigation Fee ($Fee_{peak}$):** To shift fraction $\Delta P$ of drayage trips from peak hours ($t_{peak}$) to night gates ($t_{off-peak}$), set the fee using discrete choice utility elasticity:
   $$Fee_{peak} \ge \frac{1}{\beta_{cost}} \ln \left( \frac{P_{peak}^{target}}{1 - P_{peak}^{target}} \right) - \frac{\beta_{time}}{\beta_{cost}} \Delta T_{congest}$$

---

## 6. Worked example

### Operational Scenario: The Northern Gateway to South-German Hinterland Corridor

An intermodal logistics engineer is optimizing container evacuation from a major North Sea container port (Antwerp-Rotterdam cluster) to an inland industrial manufacturing cluster in the Rhine-Neckar region (Mannheim-Ludwigshafen, Germany). 

#### 1. Baseline System Parameters

- **Annual Corridor Volume:** 320,000 TEU imports (consumer goods, chemicals, auto parts); 200,000 TEU exports.
- **Corridor Distance:**
  - Highway route distance: $d_{road} = 480 \text{ km}$
  - Rail corridor distance: $d_{rail} = 510 \text{ km}$
  - Inland waterway river distance (via Rhine): $d_{barge} = 560 \text{ km}$
- **Current Mode Performance Attributes:**

| Parameter | Road Drayage (Truck) | Intermodal Shuttle (Rail) | Push-Barge Convoy (Inland Waterway) |
| :--- | :--- | :--- | :--- |
| **Direct Freight Cost ($C_m$)** | $1,180 \text{ \$/TEU}$ | $760 \text{ \$/TEU}$ | $440 \text{ \$/TEU}$ |
| **Door-to-Door Transit Time ($T_m$)** | $9.5 \text{ hours}$ | $24.0 \text{ hours}$ | $60.0 \text{ hours}$ |
| **Schedule Reliability ($Rel_m$)** | $92\% \text{ (0.92)}$ | $86\% \text{ (0.86)}$ | $76\% \text{ (0.76)}$ |
| **Service Frequency ($F_m$)** | Continuous ($120 \text{/wk}$) | $18 \text{ trains/week}$ | $6 \text{ sailings/week}$ |
| **Alternative-Specific Const ($\alpha_m$)**| $0.00 \text{ (ref)}$ | $0.52$ | $-0.15$ |

- **Calibrated Multinomial Logit Utility Parameters:**
  - Cost coefficient: $\beta_{cost} = -0.0032 \text{ per \$}$
  - Time coefficient: $\beta_{time} = -0.0380 \text{ per hour}$
  - Reliability coefficient: $\beta_{rel} = 3.8000 \text{ per unit probability}$
  - Frequency coefficient: $\beta_{freq} = 0.0150 \text{ per departure/week}$

---

### Step 1: Compute Systematic Utilities ($V_m$)

Calculate systematic utility for each mode:

$$V_{road} = 0.00 + (-0.0032 \times 1180) + (-0.0380 \times 9.5) + (3.8000 \times 0.92) + (0.0150 \times 120)$$
$$V_{road} = 0.00 - 3.7760 - 0.3610 + 3.4960 + 1.8000 = +1.1590$$

$$V_{rail} = 0.52 + (-0.0032 \times 760) + (-0.0380 \times 24.0) + (3.8000 \times 0.86) + (0.0150 \times 18)$$
$$V_{rail} = 0.52 - 2.4320 - 0.9120 + 3.2680 + 0.2700 = +0.7140$$

$$V_{barge} = -0.15 + (-0.0032 \times 440) + (-0.0380 \times 60.0) + (3.8000 \times 0.76) + (0.0150 \times 6)$$
$$V_{barge} = -0.15 - 1.4080 - 2.2800 + 2.8880 + 0.0900 = -0.8600$$

---

### Step 2: Compute Baseline Modal Split ($P_m$)

Exponentiate utilities:
- $\exp(V_{road}) = \exp(1.1590) = 3.1867$
- $\exp(V_{rail}) = \exp(0.7140) = 2.0421$
- $\exp(V_{barge}) = \exp(-0.8600) = 0.4232$
- $\sum \exp(V_k) = 3.1867 + 2.0421 + 0.4232 = 5.6520$

Market Shares:
$$P_{road} = \frac{3.1867}{5.6520} = 0.5638 \quad (56.38\%)$$
$$P_{rail} = \frac{2.0421}{5.6520} = 0.3613 \quad (36.13\%)$$
$$P_{barge} = \frac{0.4232}{5.6520} = 0.0749 \quad (7.49\%)$$

Annual TEU Allocated:
- Road volume: $320,000 \times 0.5638 = 180,416 \text{ TEU}$
- Rail volume: $320,000 \times 0.3613 = 115,616 \text{ TEU}$
- Barge volume: $320,000 \times 0.0749 = 23,968 \text{ TEU}$

---

### Step 3: Compute Direct and Cross-Price Elasticities

1. **Direct Cost Elasticities ($E_{C_m}^{P_m} = \beta_{cost} \cdot C_m \cdot (1 - P_m)$):**
   - Road: $E_{C_{road}}^{P_{road}} = -0.0032 \times 1180 \times (1 - 0.5638) = -3.7760 \times 0.4362 = -1.647$
   - Rail: $E_{C_{rail}}^{P_{rail}} = -0.0032 \times 760 \times (1 - 0.3613) = -2.4320 \times 0.6387 = -1.553$
   - Barge: $E_{C_{barge}}^{P_{barge}} = -0.0032 \times 440 \times (1 - 0.0749) = -1.4080 \times 0.9251 = -1.303$

2. **Cross-Price Elasticity of Road with respect to Rail Price ($E_{C_{rail}}^{P_{road}} = -\beta_{cost} \cdot C_{rail} \cdot P_{rail}$):**
   $$E_{C_{rail}}^{P_{road}} = -(-0.0032) \times 760 \times 0.3613 = +2.4320 \times 0.3613 = +0.879$$
   *(Interpretation: A 10% increase in rail rates causes an 8.79% increase in road truck demand).*

3. **Value of Time and Value of Reliability:**
   $$VOT = \frac{\beta_{time}}{\beta_{cost}} = \frac{-0.0380}{-0.0032} = \$11.88 \text{ per hour per TEU}$$
   $$VOR = -\frac{\beta_{rel}}{\beta_{cost}} = -\frac{3.8000}{-0.0032} = \$1,187.50 \text{ per 1.0 unit (or \$11.88 per 1% on-time gain)}$$

---

### Step 4: Low-Water River Drought Disruption Analysis

During late summer, the Kaub river gauge drops from normal water depth ($2.50 \text{ m}$) down to $0.95 \text{ m}$.
- Allowable barge payload falls by 65%: max load per barge drops from 350 TEU to 120 TEU.
- Barge operators implement an emergency Low-Water Surcharge (KWZ) of $+\$280 \text{/TEU}$, increasing barge cost to $C_{barge}' = \$440 + \$280 = \$720 \text{/TEU}$.
- Heavy river traffic and shallow channel navigation increase transit time to $T_{barge}' = 75.0 \text{ hours}$.
- Schedule reliability drops to $Rel_{barge}' = 55\% \text{ (0.55)}$.

Recalculate Barge Utility:
$$V_{barge}' = -0.15 + (-0.0032 \times 720) + (-0.0380 \times 75.0) + (3.8000 \times 0.55) + (0.0150 \times 6)$$
$$V_{barge}' = -0.15 - 2.3040 - 2.8500 + 2.0900 + 0.0900 = -3.1240$$

New Exponentiated Utilities:
- $\exp(V_{road}) = 3.1867$
- $\exp(V_{rail}) = 2.0421$
- $\exp(V_{barge}') = \exp(-3.1240) = 0.0440$
- New Sum: $\sum \exp(V) = 3.1867 + 2.0421 + 0.0440 = 5.2728$

New Market Shares:
$$P_{road}' = \frac{3.1867}{5.2728} = 60.44\% \quad (+4.06\% \text{ shift, } +12,992 \text{ TEU to highway})$$
$$P_{rail}' = \frac{2.0421}{5.2728} = 38.73\% \quad (+2.60\% \text{ shift, } +8,320 \text{ TEU to rail})$$
$$P_{barge}' = \frac{0.0440}{5.2728} = 0.83\% \quad (-6.66\% \text{ collapse, } 21,312 \text{ TEU lost})$$

---

### Step 5: Triangulation (Street Turn) Operational Economics

In the baseline state, 200,000 export TEU require empty containers. Currently, all import empties are returned to port, and export empties are hauled inland from port:
- Standard empty drayage leg: $480 \text{ km} \times \$1.80/\text{km} = \$864 \text{ per empty move}$.
- Terminal gate turn and lift fee at port: $\$85 \text{ per gate transaction}$.

The logistics team establishes an automated dry-port triangulation exchange in Mannheim, achieving a 30% street turn rate on export containers ($0.30 \times 200,000 = 60,000 \text{ TEU}$ triangulated annually).
- Local street turn drayage distance between import receiver and export factory: average $35 \text{ km}$.
- Street turn transfer cost: $35 \text{ km} \times \$3.50/\text{km} + \$50 \text{ (inspection/admin)} = \$172.50 \text{ per box}$.

Net Financial and Environmental Savings:
1. **Cost per Triangulated Box Saved:**
   $$\Delta C = (2 \times \$864 + 2 \times \$85) - \$172.50 = \$1,898 - \$172.50 = \$1,725.50 \text{ saved per box}$$
2. **Total Annual Net Logistics Savings:**
   $$60,000 \text{ TEU} \times \$1,725.50 = \$103,530,000 \text{ saved per year}$$
3. **Empty Highway Kilometers Eliminated:**
   $$60,000 \text{ boxes} \times (2 \times 480 \text{ km} - 35 \text{ km}) = 60,000 \times 925 \text{ km} = 55,500,000 \text{ truck-km eliminated}$$
4. **Carbon Reduction (at $0.85 \text{ kg CO}_2\text{/km}$ for heavy diesel tractor):**
   $$55,500,000 \text{ km} \times 0.85 \text{ kg CO}_2/\text{km} = 47,175 \text{ metric tons CO}_2 \text{ avoided per year}$$

### Step 6: Navigation Lock Queuing and Capacity Throttling Analysis

To evaluate barge transit variance, the logistics team models the Iffezheim lock on the Upper Rhine. The lock has a twin chamber with an average service time $\bar{T}_s = 42.0 \text{ minutes}$ ($0.70 \text{ hours}$) and a standard deviation $\sigma_s = 18.0 \text{ minutes}$ ($0.30 \text{ hours}$). Convoy arrivals follow a Poisson process with arrival rate $\lambda = 1.15 \text{ convoys/hour}$ across both chambers (effective service capacity $\mu = 1 / 0.70 = 1.4286 \text{ convoys/hour}$).

1. **Lock Utilization ($\rho$):**
   $$\rho = \frac{\lambda}{\mu} = \frac{1.15}{1.4286} = 0.8050 \quad (80.5\% \text{ utilization})$$
2. **Squared Coefficient of Variation ($C_s^2$):**
   $$C_s^2 = \frac{\sigma_s^2}{\bar{T}_s^2} = \frac{0.30^2}{0.70^2} = \frac{0.090}{0.490} = 0.1837$$
3. **Expected Lock Queuing Delay ($W_q$, Pollaczek-Khinchine formula):**
   $$W_q = \frac{\rho \cdot \bar{T}_s}{2(1 - \rho)} \left(1 + C_s^2\right) = \frac{0.8050 \times 0.70}{2(1 - 0.8050)} \left(1 + 0.1837\right) = \frac{0.5635}{0.3900} \times 1.1837 = 1.4449 \times 1.1837 = 1.710 \text{ hours}$$
4. **Impact of a 15% Inflow Surge ($\lambda' = 1.32 \text{ convoys/hour}$):**
   $$\rho' = \frac{1.32}{1.4286} = 0.9240 \quad (92.4\% \text{ utilization})$$
   $$W_q' = \frac{0.9240 \times 0.70}{2(1 - 0.9240)} \times 1.1837 = \frac{0.6468}{0.1520} \times 1.1837 = 4.2553 \times 1.1837 = 5.037 \text{ hours}$$
   *(Interpretation: A 15% increase in arrival rate produces a 195% increase in lock delay, increasing from 1.71 to 5.04 hours, degrading barge reliability and driving mode shift).*

---

## 7. Code, query, or calculation

The following self-contained Python program implements the Multinomial Logit (MNL) Hinterland Modal Split model, calculates market shares and price elasticities, and computes the economic Value of Time and Value of Reliability.

```python
"""Multinomial Logit (MNL) Hinterland Modal Split and Elasticity Model."""
import math

def calculate_modal_split(modes, beta_cost, beta_time, beta_rel, beta_freq):
    """
    Computes MNL choice probabilities, elasticities, VOT, and VOR.
    modes: dict of mode_name -> {cost, time, rel, freq, asc}
    """
    # 1. Compute systematic utility V for each mode
    utilities = {}
    for m, d in modes.items():
        v = d['asc'] + (beta_cost * d['cost']) + (beta_time * d['time']) + \
            (beta_rel * d['rel']) + (beta_freq * d['freq'])
        utilities[m] = v

    # 2. Exponentiate with numerical stability shift
    max_v = max(utilities.values())
    exp_v = {m: math.exp(v - max_v) for m, v in utilities.items()}
    sum_exp = sum(exp_v.values())
    probs = {m: exp_v[m] / sum_exp for m in modes}

    # 3. Compute direct cost elasticity: E = beta_cost * Cost * (1 - P)
    direct_elas = {m: beta_cost * modes[m]['cost'] * (1.0 - probs[m]) for m in modes}

    # 4. Valuation ratios
    vot = beta_time / beta_cost           # $/hour
    vor = -beta_rel / beta_cost           # $/unit of reliability (1.0 = 100%)

    return utilities, probs, direct_elas, vot, vor

if __name__ == "__main__":
    hinterland_modes = {
        'Road':  {'cost': 1180.0, 'time': 9.5,  'rel': 0.92, 'freq': 120.0, 'asc': 0.00},
        'Rail':  {'cost':  760.0, 'time': 24.0, 'rel': 0.86, 'freq':  18.0, 'asc': 0.52},
        'Barge': {'cost':  440.0, 'time': 60.0, 'rel': 0.76, 'freq':   6.0, 'asc': -0.15}
    }
    b_c, b_t, b_r, b_f = -0.0032, -0.0380, 3.8000, 0.0150
    u, p, elas, vot, vor = calculate_modal_split(hinterland_modes, b_c, b_t, b_r, b_f)

    print("--- HINTERLAND MODAL SPLIT & ELASTICITY BENCHMARK ---")
    print(f"Valuation: VOT = ${vot:.2f}/hr | VOR = ${vor:.2f}/unit (${vor/100:.2f} per 1% on-time)")
    for m in hinterland_modes:
        print(f"Mode: {m:5s} | Utility: {u[m]:6.3f} | Share: {p[m]*100:5.2f}% | "
              f"Direct Cost Elas: {elas[m]:6.3f}")
```

### Assumptions and Expected Output

- **Model Assumptions:** Shippers exhibit homogeneous random utility preferences; unobserved modal disturbances follow i.i.d. Gumbel distributions; alternative-specific constants are calibrated against regional freight survey data; input attributes represent door-to-door values.
- **Expected Output:**
  ```text
  --- HINTERLAND MODAL SPLIT & ELASTICITY BENCHMARK ---
  Valuation: VOT = $11.88/hr | VOR = $1187.50/unit ($11.88 per 1% on-time)
  Mode: Road  | Utility:  1.159 | Share: 56.38% | Direct Cost Elas: -1.647
  Mode: Rail  | Utility:  0.714 | Share: 36.13% | Direct Cost Elas: -1.553
  Mode: Barge | Utility: -0.860 | Share:  7.49% | Direct Cost Elas: -1.303
  ```

---

## 8. Data, standards, and regulatory context

Hinterland and intermodal container operations operate under a complex matrix of international maritime treaties, national surface transportation statutes, rail safety regulations, and commercial intermodal interchange agreements.

```text
=============================================================================================================
                              REGULATORY AND STANDARDS HIERARCHY
=============================================================================================================
 Level 1: International Safety & Equipment Standards
   ├── IMO SOLAS Chapter VI Reg 2 (Verified Gross Mass - VGM)
   ├── ISO 668 (Series 1 Freight Containers - Dimensions and Ratings)
   ├── ISO 1496 (Specifications and Testing of ISO Containers)
   └── ADN / RID / ADR (International Carriage of Dangerous Goods by Waterway / Rail / Road)

 Level 2: Regional Intermodal Network & Corridor Frameworks
   ├── European Union: TEN-T Core Network Corridors Regulation (EU) 1315/2013 (740m trains, ERTMS)
   ├── European Union: Combined Transport Directive 92/106/EEC (Cabotage exemptions, road tax relief)
   └── CCNR: Police Regulations for the Navigation of the Rhine (RPNR)

 Level 3: National Highway, Intermodal & Environmental Laws
   ├── US Federal Highway Administration (FHWA): 23 CFR Part 658 (Bridge Gross Weight Formula)
   ├── US Federal Maritime Commission (FMC): Ocean Shipping Reform Act of 2022 (OSRA 2022)
   └── US Clean Air Act & Port Drayage Rules (California CARB Advanced Clean Fleets - Zero Emission Drayage)

 Level 4: Commercial Equipment Interchange Standards
   ├── IANA: Uniform Intermodal Interchange and Facilities Access Agreement (UIIA)
   ├── AAR: Association of American Railroads Intermodal Interchange Rules (Rule 83, 95)
   └── Electronic Data Interchange (EDI / ANSI X12: 204, 214, 304, 315, 404, 410, 417; EDIFACT: CODECO, COARRI)
=============================================================================================================
```

### Critical Regulatory and Statutory Mandates

1. **U.S. Federal Highway Administration (23 CFR Part 658) - The Bridge Gross Weight Formula:**
   Federal law limits the maximum gross weight of commercial motor vehicles hauling intermodal containers on the Interstate System to 80,000 lbs (36,287 kg), subject to the Federal Bridge Formula:
   $$W = 500 \left[ \frac{L \cdot N}{N - 1} + 12N + 36 \right]$$
   Where $W$ is maximum overall gross weight, $L$ is axle group spacing distance (feet), and $N$ is the number of axles. A standard 40ft container loaded above 44,000 lbs (20 metric tons) risks violating tandem axle limits (34,000 lbs max per tandem) unless hauled on a specialized tri-axle or extended-slider chassis.
2. **IMO SOLAS Regulation VI/2 - Verified Gross Mass (VGM):**
   Mandates that no container may be loaded onto an ocean vessel without an authenticated VGM document signed by the shipper using Method 1 (calibrated weighing of sealed container) or Method 2 (weighing all contents and dunnage plus container tare mass). Intermodal rail ramps and dry ports enforce VGM checks before releasing containers onto marine terminal-bound unit trains.
3. **Ocean Shipping Reform Act of 2022 (OSRA 2022 - 46 U.S.C. 41104):**
   Prohibits ocean carriers and marine terminal operators from assessing detention and demurrage charges when container pickup or return is prevented by terminal closures, lack of appointments, dual-transaction mandates, or carrier-imposed chassis split requirements. It shifts the burden of proof to the ocean carrier to justify that fees are reasonable incentives to promote freight fluidity.
4. **EU Trans-European Transport Network (TEN-T) Core Rail Standards:**
   Regulation (EU) 1315/2013 mandates that all Core Network freight rail corridors must accommodate:
   - Minimum train length of $740 \text{ meters}$ without splitting.
   - Minimum axle load capacity of $22.5 \text{ metric tons}$.
   - Line electrification and full European Rail Traffic Management System (ERTMS / ETCS Level 2) signaling deployment.
   - P400 loading gauge clearance allowing intermodal transport of 4.0-meter-high semi-trailers on pocket wagons.
5. **Electronic Data Interchange (EDI) Protocols for Hinterland Transfer:**
   - **EDI 404 (Bill of Lading - Rail):** Transmits shipping instructions, container numbers, seal records, and route codes from ocean carriers or IMCs to Class 1 railroads.
   - **EDI 417 (Rail Waybill):** Authorizes physical movement of railcars between interchange junctions.
   - **EDI 204 / 214 (Motor Carrier Load Tender / Shipment Status):** Automates drayage dispatching and real-time GPS arrival/departure time stamps.
   - **EDI 315 / EDIFACT CODECO (Container Gate In/Out):** Transmits instant gate crossing timestamps and equipment damage inspection flags to shipping lines and terminal operating systems.

---

## 9. Industry practice and case evidence

Modern seaport-hinterland supply chains rely on specialized infrastructure corridors and digital coordination platforms. Four landmark global implementations illustrate operational benchmarks and engineering trade-offs:

```text
+---------------------------------------------------------------------------------------------------+
|                               BENCHMARK HINTERLAND CORRIDORS                                      |
+---------------------------------------------------------------------------------------------------+
 1. ROTTERDAM - RHINE CORRIDOR & THE BETUWEROUTE (Netherlands / Germany)
    - 160 km dedicated, freight-only, double-track electrified rail corridor.
    - Operates under ETCS Level 2 signaling; bypasses all Dutch passenger bottlenecks.
    - Directly connects Rotterdam Maasvlakte deep-sea terminals to the German industrial Ruhr.
    - Integrated synchromodality: dynamic dispatch balancing Betuweroute rail with Rhine push-barges.

 2. SAN PEDRO BAY & THE ALAMEDA CORRIDOR (Los Angeles / Long Beach, California)
    - 20-mile (32 km) grade-separated, triple-track rail trench depressed 10 meters below grade.
    - Eliminates 200+ street-level highway-rail intersections; reduces vehicular delays by 90%.
    - Connects marine on-dock and near-dock ICTFs to transcontinental rail classification yards.
    - Coupled with PierPass off-peak gate appointment fee system and the tri-party Pool-of-Pools.

 3. PORT OF GOTHENBURG "RAILPORT SCANDINAVIA" (Sweden / Norway)
    - Network of 30+ scheduled dedicated shuttle block trains connecting 25 inland dry ports.
    - Over 60% of container throughput evacuated via electrified rail directly from on-dock tracks.
    - Containers clear customs and undergo pre-gate inspection at inland dry ports (e.g., Falköping).

 4. INDIA WESTERN DEDICATED FREIGHT CORRIDOR (WDFC - JNPT to NCR Delhi)
    - 1,504 km dedicated heavy-haul freight rail corridor built to double-stack container clearance.
    - Electric locomotives haul 1,500-meter-long trains with trailing loads exceeding 12,000 tons.
    - Slashes transit time from Jawaharlal Nehru Port (Nhava Sheva) to Delhi from 70 hours to 18 hours.
+---------------------------------------------------------------------------------------------------+
```

### Deep Industrial Case Studies

#### Case 1: The Alameda Corridor and the San Pedro Bay "Pool of Pools"
Prior to 2002, freight trains departing the Ports of Los Angeles and Long Beach traversed three separate surface rail lines at ground level, crossing over 200 highway intersections across metropolitan Southern California. Trains were limited to 15–20 mph (24–32 km/h), generating massive vehicular gridlock, grade-crossing collisions, and locomotive idling emissions.

The Alameda Corridor Transportation Authority (ACTA) engineered a \$2.4 billion, 20-mile consolidated rail corridor, centered on a 10-mile, 33-foot-deep, 50-foot-wide concrete trench accommodating three continuous rail tracks. Trains operate at 40 mph (64 km/h) under centralized traffic control (CTC). The project consolidated operations of BNSF and Union Pacific, handling up to 50 intermodal double-stack trains per day.

To address the parallel crisis of chassis availability, the three dominant equipment leasing companies (TRAC Intermodal, DCLI, and Flexi-Van) established the "Pool of Pools" in 2015. This interoperable gray chassis pool created a shared fleet of over 65,000 chassis across 12 marine terminals and 3 major off-dock rail ramps. Motor carriers could pick up any chassis regardless of pool owner and return it to any terminal, eliminating thousands of daily chassis split moves and cutting drayage cycle times by an estimated 32 minutes per round trip.

#### Case 2: The Rhine River Drought Disruption and Synchromodal Agility (2018 & 2022)
In the autumn of 2018 and the summer of 2022, prolonged European droughts caused the Kaub gauge on the Middle Rhine to drop to historic lows of 25–35 cm. Permissible vessel draft dropped below 1.30 meters. Over 75% of barge carrying capacity vanished.

Barge operators declared *force majeure*, applying Low-Water Surcharges (KWZ) exceeding €650 per TEU. Marine terminals at Rotterdam and Antwerp experienced container yard stack freezing as import cargo accumulated on quays. Shippers attempted to shift thousands of boxes to the rail network; however, the regional rail network was constrained by infrastructure maintenance, driver shortages, and lack of available electric locomotives.

Leading chemical manufacturers (e.g., BASF Ludwigshafen) and intermodal operators (Contargo) responded by deploying **synchromodal control towers**. Using real-time digital APIs connecting river water level forecasts, rail slot reservation engines, and drayage marketplaces, dispatchers dynamically rerouted containers daily:
- High-priority, high-value components were diverted to dedicated road drayage teams.
- Heavy chemical raw materials were shifted to guaranteed block-train slots reserved months in advance.
- Lightweight empty containers were loaded onto draft-restricted barges operating with shallow-draft pusher tugs.
This synchromodal agility prevented total plant shutdowns, demonstrating that hinterland resilience requires flexible multi-modal contracts rather than static mode choices.

---

## 10. Failure modes and red flags

The table below catalogs critical failure modes in intermodal hinterland logistics, detailing root causes, early warnings, operational consequences, and engineering mitigations.

| Failure Mode | Early Warning Sign | Root Mechanism | Commercial & Operational Consequence | Mitigation / Preventive Control |
| :--- | :--- | :--- | :--- | :--- |
| **Chassis Pool Deficit** | Bare chassis dwell at marine gates exceeds 45 mins; bobtail truck queues form outside depots. | Regional trade import/export imbalance; street dwell exceeds allowed limits; slow turn time at warehouses. | Drayage drivers cannot pull import containers off docks; marine yard dwell doubles; terminal demurrage charges surge. | Enforce dynamic street dwell detention fees; implement gray interoperable pools; deploy automated GPS chassis tracking. |
| **Chassis Split Gridlock** | Motor carrier turn-times increase by > 60 mins; high proportion of deadhead miles reported. | Disconnected equipment pools; marine terminal rejects chassis owned by competing leasing entity. | Lost driver productivity under Hours of Service (HOS); missed vessel loading cutoffs; punitive drayage surcharges. | Contractual transition to interoperable gray pools; mandate on-dock universal chassis return lanes. |
| **On-Dock Rail Track Starvation** | Working tracks empty while import stacks are at 85% capacity; RMG cranes idle on rail spurs. | Inbound switch engine delayed by mainline rail dispatching; missing customs or carrier rail bill release (EDI 404). | Vessel discharge halted due to lack of ground slots; train departures canceled; contractual demurrage billed to terminal. | Automated EDI 404 validation 24 hours prior to vessel discharge; dedicated switch engines assigned to on-dock spurs. |
| **Low-Water River Drought** | Critical river gauge (e.g., Kaub) falls below 1.20 m; low-water surcharges (KWZ) exceed base freight rate. | Extended drought reduces channel depth; vessel squat forces extreme payload derating. | Barge capacity collapses by 60–80%; freight shifts suddenly to road and rail, overwhelming corridor infrastructure. | Establish synchromodal rail-barge flexible contracts; deploy ultra-shallow-draft low-water barge designs. |
| **Lock Chamber Failure / Queuing Spike** | Convoy queue length at lock approach pool exceeds 6 vessels; lock wait time exceeds 18 hours. | Mechanical failure of lock gate hydraulics, sluice valve fracture, or debris obstruction in chamber. | Inland waterway barge transit times double; supply chains face stockouts; barge asset productivity collapses. | Preventive acoustic monitoring of lock machinery; dynamic tow sequencing heuristics; secondary bypass lock chambers. |
| **Rail Classification Yard Deadlock** | Railcars dwell in humping/switching tracks > 48 hours; inbound trains held on sidings outside terminal. | Train arrival volume exceeds sorting capacity; switch engine breakdowns; train length exceeds yard siding length. | Inability to receive export trains or dispatch import shuttles; corridor congestion spills onto mainline tracks. | Block train bypass routing (avoiding intermediate switching yards); dynamic train pacing using precision scheduled railroading (PSR). |
| **Empty Container Inventory Gluttony** | Inland dry port yard exceeds 90% utilization; stack heights reach 6-high empty boxes; high wind halts operations. | Persistent directional trade deficit; ocean carriers fail to book export sweep vessels to evacuate empty boxes. | Dry port cannot accept returning loaded import boxes; gate shutdowns; equipment maintenance costs surge. | Impose punitive empty container storage surcharges on ocean carriers; mandate ocean carrier "empty-in/empty-out" ratios. |
| **In-Bond Customs Filing Rejection** | Containers gated out of terminal held at inland rail ramp under immediate CBP seizure flags. | Discrepancy between manifest data, commercial invoice, and In-Bond Form (CBP 7512 / EU T1) data. | Severe regulatory fines; cargo immobilised for 7–21 days; bonding company revokes transit authorization. | Automated EDI validation between ocean bill of lading, customs broker entries, and rail billing engines. |
| **Corridor Shockwave Gate Spillback** | Drayage truck queues extend past port security gates onto municipal freeways; local police issue citations. | Highway bottleneck capacity throttled below terminal gate discharge rate ($q_{gate} > q_{bottle}$). | Municipal Clean Air Action Plan fines; traffic safety accidents; driver detention compensation claims ($100–$150/hr). | Implement dynamic Truck Appointment Systems (TAS) tied to downstream highway sensor feeds; open 24/7 off-peak gate hours. |
| **Overweight Axle Violations at Intermodal Gate** | Weigh-in-motion (WIM) scales trigger red lights; 40ft containers fail tandem axle bridge formula. | Uneven cargo stowage inside container; shipper stuffed dense freight without tri-axle chassis assignment. | Container rejected at gate; driver forced to dray box back to transload facility for physical cargo re-handling. | Mandatory digital VGM verification coupled with pre-gate automated axle-load predictive algorithms. |
| **Street-Turn Alliance Mismatch** | Motor carrier denied street turn authorization at consignee dock; box forced to make round-trip drayage to port. | Import box belongs to Ocean Carrier X (Alliance 1), while export booking is with Ocean Carrier Y (Alliance 2). | Waste of 400–800 km of empty drayage; unnecessary gate congestion; lost driver hours and excess diesel emissions. | Inter-alliance bilateral equipment interchange agreements; neutral third-party digital street-turn clearing platforms. |
| **Hazardous Material Rail Segregation Violation** | Train makeup rejected by rail carrier safety inspector; missing buffer cars. | Incompatible IMDG/RID hazardous material classes placed in adjacent railcars; violation of 49 CFR § 174.84. | Immediate train departure halt; $25,000/day federal fines; expensive humping/shunting maneuvers to re-sequence cars. | Automated intermodal railcar placement software with hardcoded IMDG/RID segregation matrices and 5-car mechanical buffers. |
| **Overhead Catenary / Low Bridge Strike** | Laser profiling scanner triggers alarm at rail exit portal; physical contact with overhead wire. | High-Cube (9'6") double-stack container dispatched onto rail route cleared only for standard 8'6" or single-stack equipment. | Catastrophic catenary wire tear-down; entire rail division traction power outage; corridor closure for 18–36 hours. | Laser profile measurement portals at all intermodal track exits integrated with positive train control (PTC) route enforcement. |

---

## 11. Assumptions and limits

Every intermodal operations model depends on foundational assumptions. The table below details these boundaries, operational failure points, and diagnostic verification checks.

| Assumption | Operational Reality | Boundary / Failure Condition | Diagnostic Test |
| :--- | :--- | :--- | :--- |
| **Independence of Irrelevant Alternatives (IIA) in MNL** | Red-bus/blue-bus paradox: introducing an electric rail shuttle steals equal proportional share from diesel rail and barge. | Fails when competing modes share unobserved attributes (e.g., rail and barge both suffer fixed-schedule inflexibility). | Conduct Hausman-McFadden specification test; evaluate Nested Logit or Mixed Random Parameter Logit models. |
| **Deterministic Transit Times** | Line-haul transit times and rail/gate turn times fluctuate stochastically due to weather, traffic, and inspections. | Fails during severe weather disruptions, grade-crossing accidents, or customs inspection bottlenecks. | Measure transit time coefficient of variation ($CV = \sigma / \mu$); if $CV > 0.20$, replace deterministic model with stochastic queue. |
| **Infinite Chassis Rebalancing Mobility** | Bare chassis cannot be moved costlessly; repositioning requires tractor drivers, fuel, and equipment maintenance. | Fails during driver shortages or when chassis repositioning freight rates exceed demurrage savings. | Track empty-to-loaded chassis ratio and compare against local drayage tractor availability metrics. |
| **Frictionless Street Turn Matching** | Ocean shipping lines restrict street turns due to container grade quality, reefer inspections, and export line reservations. | Fails when export shippers demand food-grade containers while incoming import boxes held industrial scrap. | Inspect empty container rejection rates at export stuffing facilities; audit carrier digital street turn approval latency. |
| **Unconstrained Inland Rail Corridor Paths** | Freight trains must share track infrastructure with passenger trains possessing statutory dispatching priority. | Fails during daytime peak passenger commute hours; freight trains held in sidings, missing terminal cutoffs. | Calculate track path occupancy ratios and analyze dispatcher siding dwell records from Class 1 / infrastructure managers. |
| **Linear Economies of Scale in Barge Convoys** | Coupling more barges increases tow capacity but degrades hydrodynamic maneuverability and lock transit speed. | Fails when lock chamber dimensions force convoy decoupling; two-stage locking consumes 3–4 hours, erasing scale economies. | Compare total transit time curves of 4-barge vs. 6-barge convoys across corridors with varying lock chamber sizes. |
| **Static Value of Time (VOT) Across Shippers** | Shippers of high-value perishable goods have a VOT 10x higher than shippers of scrap paper or raw minerals. | Fails when treating the market as a single homogeneous customer class; leads to inaccurate modal split forecasts. | Segment customer demand by commodity value density (\$/kg) and calibrate distinct segment-specific logit models. |
| **Constant Freight Elasticity across Economic Cycles** | Elasticities are highly asymmetric between boom and bust freight cycles. | During peak capacity shortages, shippers are price-inelastic; during recessions, freight rates dominate mode choice. | Perform rolling-window econometric regressions across differing macroeconomic freight rate index cycles. |
| **Homogeneous Inland Waterway Current Speed** | Models assume average transit time without separating upstream against-current velocity ($v_{up} = v_{water} - v_{stream}$) from downstream velocity ($v_{down} = v_{water} + v_{stream}$). | Fails in high-gradient river sections (Middle Rhine, Upper Mississippi) where upstream transit time is 2.5x downstream. | Compare round-trip fuel and transit telemetry partitioned by directional heading. |
| **Independent Rail Shunting and Switching Dynamics** | Rail models assume switching fees ($C_{switch}$) are static per car regardless of terminal track occupancy. | Fails when rail yards reach 85% occupancy, causing non-linear switching delay spikes and locomotive crew overtime. | Monitor classification yard car dwell time vs. yard throughput to identify exponential congestion knee points. |

---

## 12. Exercises

### Exercise 1: Dedicated On-Dock Rail vs. Off-Dock Drayage Break-Even (Calculation)
- **Prompt:** A marine container terminal is evaluating whether to construct an on-dock intermodal rail yard. The on-dock capital amortization and track maintenance cost is $C_{track} = \$22.00 \text{ /TEU}$. Terminal tractors transfer containers to the on-dock tracks at $C_{trans} = \$18.00 \text{ /TEU}$, and RMG rail loading costs $C_{lift,on} = \$38.00 \text{ /TEU}$. Railcar switching and makeup fees cost $C_{switch} = \$45.00 \text{ /TEU}$. 
Conversely, utilizing the existing off-dock ICTF located 22 km from the port incurs: marine gate processing fee $C_{gate,port} = \$15.00 \text{ /TEU}$, local highway drayage $C_{dray} = \$165.00 \text{ /TEU}$, expected chassis split fee $C_{split} = \$40.00 \text{ /TEU}$, ICTF in-gate fee $C_{gate,ictf} = \$12.00 \text{ /TEU}$, and ICTF crane lift $C_{lift,off} = \$42.00 \text{ /TEU}$. 
Both options incur identical line-haul rail transport rates of $c_{rail} = \$0.045 \text{ /TEU-km}$.
Compute the net operational cost savings per TEU achieved by on-dock rail. If constructing the on-dock facility costs \$85 million and the terminal handles 450,000 intermodal rail TEU annually, compute the simple capital payback period in years.
- **Deliverable:** Mathematical cost equations, net savings per TEU, and payback period calculation.
- **Acceptance Criteria:** Correct identification of fixed vs. variable components; exact arithmetic payback calculation.
- **Estimated Time:** 25 minutes.

### Exercise 2: Multinomial Logit Modal Split and Cross-Elasticity (Calculation)
- **Prompt:** An intermodal corridor operates three transport modes with the following attributes:
  - **Road:** Cost = \$950/TEU, Transit Time = 12.0 hours, Reliability = 94%, Frequency = 100/week, $\alpha_{road} = 0.00$.
  - **Rail:** Cost = \$620/TEU, Transit Time = 28.0 hours, Reliability = 88%, Frequency = 24/week, $\alpha_{rail} = 0.40$.
  - **Barge:** Cost = \$380/TEU, Transit Time = 56.0 hours, Reliability = 80%, Frequency = 8/week, $\alpha_{barge} = -0.10$.
  - Utility parameters: $\beta_{cost} = -0.0030 \text{ \$/TEU}$, $\beta_{time} = -0.0400 \text{ /hr}$, $\beta_{rel} = 3.5000$, $\beta_{freq} = 0.0120$.
  1. Calculate the market share ($P_m$) of each mode.
  2. Calculate the direct cost elasticity of Rail ($E_{C_{rail}}^{P_{rail}}$).
  3. Calculate the cross-cost elasticity of Road demand with respect to a change in Rail freight rates ($E_{C_{rail}}^{P_{road}}$).
  4. Compute the Value of Time (VOT) in \$/hour and the Value of Reliability (VOR) in \$/% on-time.
- **Deliverable:** Step-by-step utility calculations, choice probabilities, elasticity formulas, and valuation ratios.
- **Acceptance Criteria:** Choice probabilities summing to 1.0000 ($\pm 0.0005$); correct signs on direct and cross-elasticities.
- **Estimated Time:** 35 minutes.

### Exercise 3: Inland Waterway Low-Water Derating and Fleet Sizing (Calculation)
- **Prompt:** A chemical manufacturing cluster imports 12,000 TEU per month via an inland river corridor using push-barges. A standard barge has length $L = 76.5 \text{ m}$, beam $B = 11.4 \text{ m}$, light (empty) draft $T_{empty} = 0.60 \text{ m}$, and a displacement coefficient such that each 1.0 cm of immersion corresponds to $7.5 \text{ metric tons}$ of cargo. The average gross mass of a loaded container is $15.0 \text{ metric tons/TEU}$.
Charted river depth is $D_{chart} = 2.80 \text{ m}$, and the port authority mandates a dynamic under-keel clearance plus squat allowance of $UKC + \Delta_{squat} = 0.45 \text{ m}$.
1. During normal water, gauge height is $\Delta h = +1.10 \text{ m}$. Calculate the maximum allowable vessel draft, maximum cargo metric tons, and maximum TEU capacity per barge.
2. During an extreme summer drought, gauge height drops to $\Delta h = -0.90 \text{ m}$. Calculate the revised allowable vessel draft, cargo payload, and TEU capacity per barge.
3. If each barge makes 4 round trips per month, compute how many active barges are required to haul the 12,000 monthly TEU during normal water versus during the drought.
- **Deliverable:** Permissible draft calculations, payload equations, and fleet sizing derivations.
- **Acceptance Criteria:** Correct conversion between immersion tons, cargo mass, and TEU capacity under physical draft limits.
- **Estimated Time:** 30 minutes.

### Exercise 4: Chassis Pool Network Flow Formulation (Mathematical Modeling)
- **Prompt:** Formulate a multi-period, multi-node Mixed Integer Linear Program (MILP) to optimize the rebalancing of empty intermodal chassis across a metropolitan port region containing 3 marine terminals, 2 off-dock rail ramps, and 4 major distribution centers over a 7-day rolling horizon. Motor carriers incur deadhead tractor repositioning costs per chassis mile. The model must incorporate:
  - Daily stochastic demand for bare chassis at marine terminals.
  - Daily releases of bare chassis at distribution centers.
  - Driver availability constraints on bare chassis repositioning moves.
  - Penalties for unfulfilled chassis demand ($/chassis/day) and chassis storage holding costs.
- **Deliverable:** Complete algebraic formulation: decision variables, objective function, and full constraint set with index definitions.
- **Acceptance Criteria:** Clear definition of state vs. control variables; accurate multi-period inventory balance equations.
- **Estimated Time:** 40 minutes.

### Exercise 5: Empty Container Triangulation (Street Turn) Feasibility (Optimization)
- **Prompt:** An intermodal dispatch center receives requests for 8 import container deliveries and 6 export container pickups within a metropolitan radius on a given day. All containers are standard 40ft High-Cubes belonging to the same ocean carrier.
Formulate the matching problem as a Bipartite Matching / Assignment model to maximize total drayage cost savings from triangulation compared to traditional two-way depot round trips. Define the spatial and temporal feasibility constraints governing each candidate match $(i, j)$.
- **Deliverable:** Mathematical assignment model, cost saving objective function, and temporal compatibility constraints.
- **Acceptance Criteria:** Formulation accounts for driver travel time, customer loading/unloading windows, and depot bypass savings.
- **Estimated Time:** 30 minutes.

### Exercise 6: Corridor Shockwave Back-Propagation Analysis (Dynamic Analysis)
- **Prompt:** A dedicated port access highway corridor has a free-flow speed $v_f = 80 \text{ km/h}$ and jam density $k_{jam} = 120 \text{ vehicles/lane-km}$ across 3 outbound travel lanes. A severe bottleneck (bridge repair) reduces capacity at a point 16 km downstream from the marine terminal gate to $q_{bottle} = 2,400 \text{ vehicles/hour}$. 
During morning peak gate operations, the marine terminal gate discharges trucks and port-related traffic at a rate of $q_{in} = 4,200 \text{ vehicles/hour}$ into the 3-lane corridor.
1. Determine the free-flow density $k_{in}$ corresponding to $q_{in} = 4,200 \text{ veh/h}$ under Greenshields traffic flow theory.
2. Determine the congested traffic density $k_{congest}$ upstream of the bottleneck.
3. Calculate the shockwave propagation velocity $w_{shock}$ in km/h.
4. Calculate the time in minutes before the congestion shockwave reaches the marine terminal exit gate.
- **Deliverable:** Kinematic wave calculations, Greenshields parameter derivations, shockwave speed, and time-to-gate arrival.
- **Acceptance Criteria:** Correct application of LWR shockwave equation; proper handling of multi-lane aggregate flow values.
- **Estimated Time:** 35 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Dedicated On-Dock Rail vs. Off-Dock Drayage Break-Even
1. **On-Dock Rail Fixed Cost per TEU:**
   $$C_{on,fixed} = C_{track} + C_{trans} + C_{lift,on} + C_{switch} = \$22 + \$18 + \$38 + \$45 = \$123.00 \text{ /TEU}$$
2. **Off-Dock Rail Fixed Cost per TEU:**
   $$C_{off,fixed} = C_{gate,port} + C_{dray} + C_{split} + C_{gate,ictf} + C_{lift,off} = \$15 + \$165 + \$40 + \$12 + \$42 = \$274.00 \text{ /TEU}$$
3. **Net Operational Savings per TEU:**
   $$\Delta C = C_{off,fixed} - C_{on,fixed} = \$274.00 - \$123.00 = \$151.00 \text{ saved per TEU}$$
4. **Annual Net Savings and Payback Period:**
   $$\text{Annual Operational Savings} = 450,000 \text{ TEU} \times \$151.00/\text{TEU} = \$67,950,000 \text{ per year}$$
   $$\text{Simple Capital Payback} = \frac{\$85,000,000}{\$67,950,000/\text{year}} = 1.251 \text{ years (approx. 15.0 months)}$$

---

### Answer to Exercise 2: Multinomial Logit Modal Split and Cross-Elasticity
1. **Systematic Utilities:**
   - $V_{road} = 0.00 + (-0.0030 \times 950) + (-0.0400 \times 12.0) + (3.5000 \times 0.94) + (0.0120 \times 100) = 0 - 2.850 - 0.480 + 3.290 + 1.200 = +1.1600$
   - $V_{rail} = 0.40 + (-0.0030 \times 620) + (-0.0400 \times 28.0) + (3.5000 \times 0.88) + (0.0120 \times 24) = 0.40 - 1.860 - 1.120 + 3.080 + 0.288 = +0.7880$
   - $V_{barge} = -0.10 + (-0.0030 \times 380) + (-0.0400 \times 56.0) + (3.5000 \times 0.80) + (0.0120 \times 8) = -0.10 - 1.140 - 2.240 + 2.800 + 0.096 = -0.5840$
2. **Choice Probabilities:**
   - $\exp(V_{road}) = \exp(1.1600) = 3.1899$
   - $\exp(V_{rail}) = \exp(0.7880) = 2.1990$
   - $\exp(V_{barge}) = \exp(-0.5840) = 0.5577$
   - $\sum \exp = 3.1899 + 2.1990 + 0.5577 = 5.9466$
   - $P_{road} = 3.1899 / 5.9466 = 0.5364 \quad (53.64\%)$
   - $P_{rail} = 2.1990 / 5.9466 = 0.3698 \quad (36.98\%)$
   - $P_{barge} = 0.5577 / 5.9466 = 0.0938 \quad (9.38\%)$
3. **Direct Cost Elasticity of Rail:**
   $$E_{C_{rail}}^{P_{rail}} = \beta_{cost} \cdot C_{rail} \cdot (1 - P_{rail}) = -0.0030 \times 620 \times (1 - 0.3698) = -1.860 \times 0.6302 = -1.172$$
4. **Cross-Cost Elasticity of Road with respect to Rail:**
   $$E_{C_{rail}}^{P_{road}} = -\beta_{cost} \cdot C_{rail} \cdot P_{rail} = -(-0.0030) \times 620 \times 0.3698 = +1.860 \times 0.3698 = +0.688$$
5. **Valuation Ratios:**
   $$VOT = \frac{\beta_{time}}{\beta_{cost}} = \frac{-0.0400}{-0.0030} = \$13.33 \text{ per hour}$$
   $$VOR = -\frac{\beta_{rel}}{\beta_{cost}} = -\frac{3.5000}{-0.0030} = \$1,166.67 \text{ per unit (or \$11.67 per 1% on-time)}$$

---

### Answer to Exercise 3: Inland Waterway Low-Water Derating and Fleet Sizing
1. **Normal Water Payload and Capacity:**
   - Permissible draft: $T_{allow} = D_{chart} + \Delta h - (UKC + \Delta_{squat}) = 2.80 + 1.10 - 0.45 = 3.45 \text{ m}$.
   - Available immersion draft: $\Delta T = T_{allow} - T_{empty} = 3.45 - 0.60 = 2.85 \text{ m} = 285 \text{ cm}$.
   - Cargo mass capacity: $285 \text{ cm} \times 7.5 \text{ t/cm} = 2,137.5 \text{ metric tons}$.
   - Max TEU capacity: $2,137.5 \text{ t} / 15.0 \text{ t/TEU} = 142.5 \implies 142 \text{ TEU per barge}$.
2. **Drought Water Payload and Capacity:**
   - Permissible draft: $T_{allow}' = 2.80 + (-0.90) - 0.45 = 1.45 \text{ m}$.
   - Available immersion draft: $\Delta T' = 1.45 - 0.60 = 0.85 \text{ m} = 85 \text{ cm}$.
   - Cargo mass capacity: $85 \text{ cm} \times 7.5 \text{ t/cm} = 637.5 \text{ metric tons}$.
   - Max TEU capacity: $637.5 \text{ t} / 15.0 \text{ t/TEU} = 42.5 \implies 42 \text{ TEU per barge}$ (a 70.4% capacity reduction).
3. **Fleet Sizing Requirements:**
   - Monthly volume: $12,000 \text{ TEU}$. Trips per barge per month: 4.
   - Normal water capacity per barge/month: $142 \text{ TEU} \times 4 = 568 \text{ TEU/month}$.
     $$\text{Barges required (normal)} = \lceil 12,000 / 568 \rceil = \lceil 21.13 \rceil = 22 \text{ active barges}$$
   - Drought water capacity per barge/month: $42 \text{ TEU} \times 4 = 168 \text{ TEU/month}$.
     $$\text{Barges required (drought)} = \lceil 12,000 / 168 \rceil = \lceil 71.43 \rceil = 72 \text{ active barges}$$
   *(Interpretation: To maintain volume during the drought without modal diversion, the operator must secure 50 additional barges, which is typically physically impossible, forcing traffic onto rail and road).*

---

### Rubric for Exercise 4: Chassis Pool Network Flow Formulation
- **Decision Variables (25%):** Correctly defines $x_{ijt}$ (bare chassis repositioned from $i$ to $j$ arriving at $t + \tau_{ij}$), $I_{it}$ (on-hand chassis inventory at node $i$ at start of $t$), $s_{it}^+$ (surplus inventory holding), and $s_{it}^-$ (unmet demand shortage).
- **Objective Function (25%):** Formulates total cost minimization including deadhead transport costs $\sum c_{ij} x_{ijt}$, shortage penalties $\sum p_i s_{it}^-$, and holding costs $\sum h_i s_{it}^+$.
- **Inventory Balance Constraints (30%):** Enforces rigorous conservation: $I_{i,t} = I_{i,t-1} + \sum_j x_{ji, t-\tau_{ji}} - \sum_j x_{ij,t} + \Delta_{it}^{organic} + s_{it}^- - s_{it}^+$.
- **Driver and Asset Capacity Constraints (20%):** Correctly bounds repositioning moves by available driver-hours $\sum_j t_{ij} x_{ijt} \le Cap_{driver,t}$ and yard storage capacities $I_{it} \le Cap_{yard,i}$.

---

### Rubric for Exercise 5: Empty Container Triangulation Feasibility
- **Decision Variables (20%):** Binary assignment variable $z_{ij} \in \{0, 1\}$ equal to 1 if import container $i \in I$ is directly assigned to export shipper $j \in J$.
- **Objective Function (25%):** Maximizes total net savings: $\sum_{i \in I} \sum_{j \in J} (C_{roundtrip,i} + C_{roundtrip,j} - C_{triangulate,ij}) z_{ij}$.
- **Assignment Constraints (25%):** Ensures each import container is assigned at most once ($\sum_j z_{ij} \le 1$) and each export request is fulfilled at most once ($\sum_i z_{ij} \le 1$).
- **Temporal & Spatial Validity (30%):** Enforces time windows: $(t_{discharge,i} + t_{unpack,i} + t_{travel,ij}) \cdot z_{ij} \le t_{pack\_start,j}$; requires equipment grade parity and common ocean carrier code validation.

---

### Answer to Exercise 6: Corridor Shockwave Back-Propagation Analysis
1. **Traffic Stream Properties (3 lanes aggregate):**
   - Corridor parameters: $v_f = 80 \text{ km/h}$, $k_{jam} = 3 \text{ lanes} \times 120 = 360 \text{ veh/km}$.
   - Capacity: $q_{max} = v_f \cdot k_{jam} / 4 = 80 \times 360 / 4 = 7,200 \text{ veh/h}$.
2. **Upstream Inflow Density ($k_{in}$):**
   Greenshields equation: $q = v_f k (1 - k/k_{jam}) \implies 4,200 = 80 k - (80/360) k^2 = 80 k - 0.2222 k^2$.
   $$0.2222 k^2 - 80 k + 4,200 = 0 \implies k^2 - 360 k + 18,900 = 0$$
   Solving quadratic: $k = \frac{360 - \sqrt{360^2 - 4(18,900)}}{2} = \frac{360 - \sqrt{129,600 - 75,600}}{2} = \frac{360 - \sqrt{54,000}}{2} = \frac{360 - 232.38}{2} = 63.81 \text{ veh/km}$.
   *(We select the uncongested branch $k < k_{crit} = 180$).*
3. **Congested Bottleneck Density ($k_{congest}$):**
   Downstream bottleneck discharge: $q_{bottle} = 2,400 \text{ veh/h}$.
   $$k^2 - 360 k + \frac{2,400}{0.2222} = 0 \implies k^2 - 360 k + 10,800 = 0$$
   Solving on the congested branch ($k > 180$):
   $$k_{congest} = \frac{360 + \sqrt{360^2 - 4(10,800)}}{2} = \frac{360 + \sqrt{129,600 - 43,200}}{2} = \frac{360 + \sqrt{86,400}}{2} = \frac{360 + 293.94}{2} = 326.97 \text{ veh/km}$$
4. **Shockwave Velocity ($w_{shock}$):**
   $$w_{shock} = \frac{q_{congest} - q_{in}}{k_{congest} - k_{in}} = \frac{2,400 - 4,200}{326.97 - 63.81} = \frac{-1,800}{263.16} = -6.84 \text{ km/h}$$
   *(The negative sign confirms the congestion shockwave is propagating backward toward the port gate).*
5. **Time to Reach Marine Terminal Gate:**
   $$T = \frac{L}{|w_{shock}|} = \frac{16.0 \text{ km}}{6.84 \text{ km/h}} = 2.339 \text{ hours} \approx 140.3 \text{ minutes}$$
   *(Interpretation: Gate operations have exactly 2 hours and 20 minutes from the onset of the bottleneck before the queue strikes the marine terminal out-gates, necessitating immediate appointment throttling).*

---

## 14. Interview questions

### Question 1: Dedicated On-Dock Rail vs. Off-Dock ICTF Trade-Offs
- **Prompt:** "Under what operational, physical, and financial conditions should a port authority invest $100M+ in an on-dock intermodal rail facility instead of relying on an existing near-dock or off-dock rail ramp?"
- **Strong Answer:** A strong answer evaluates the trade-off between land opportunity cost and drayage friction. On-dock rail eliminates local highway drayage costs (\$150–\$250/TEU), double gate transactions, and off-dock chassis split moves, generating immediate operational savings that justify high capital expenditure if intermodal rail volume exceeds 250,000–300,000 TEU annually. However, on-dock tracks consume prime, deep-water quayside real estate that could otherwise be used for container stacking blocks. If terminal land is severely constrained ($< 50 \text{ hectares}$) and vessel call volume is high, converting yard storage to rail tracks can lower vessel berth productivity by restricting yard buffer capacity. Furthermore, on-dock rail requires rail working tracks long enough to assemble unit trains without excessive switching moves; if track lengths are under 600–800 meters, train classification costs will erode the drayage savings.
- **Weak Answer:** "On-dock rail is always better because it is closer to the ship and avoids truck pollution on city streets."

### Question 2: The Red-Bus/Blue-Bus Flaw in Freight Modal Choice
- **Prompt:** "Why is the standard Multinomial Logit (MNL) model vulnerable when evaluating a new inland rail shuttle against existing truck and barge services, and how do you resolve it mathematically?"
- **Strong Answer:** The standard MNL model exhibits the Independence of Irrelevant Alternatives (IIA) property, caused by the assumption that error terms $\varepsilon_{im}$ are independently and identically distributed. If a new rail shuttle is introduced that is identical to an existing rail service, MNL assumes it draws proportional market share from both road and barge, rather than solely cannibalizing rail. In freight systems, rail and barge share unobserved supply chain attributes: rigid departure timetables, batch-size requirements, and common intermodal transfer interfaces. To resolve this, we employ a **Nested Logit** model, placing Rail and Barge into an "Intermodal" nest with an inclusive value parameter $\mu \in (0, 1]$. Alternatively, a **Mixed (Random Coefficients) Logit** model can be estimated, allowing cost and time parameters $\beta_C, \beta_T$ to follow continuous probability distributions across shippers, capturing unobserved taste heterogeneity and eliminating IIA restrictions.
- **Weak Answer:** "MNL is fine as long as you have enough survey data. If there is an error, you just adjust the constant terms."

### Question 3: The Root Mechanics of the North American Chassis Dislocation
- **Prompt:** "Why do North American port complexes experience severe chassis shortages during import surges, while European ports rarely experience chassis crises?"
- **Strong Answer:** The difference stems from equipment ownership and operating models. In Europe, motor carriers own their tractor-chassis combinations (merchant haulage) or ocean carriers provide door delivery under unified responsibility (carrier haulage). The container is loaded directly onto the trucker’s semi-trailer chassis; there is no secondary equipment interchange. In North America, ocean carriers historically provided chassis under the UIIA, but divested the assets between 2009 and 2014 to private leasing pools. This created a fragmented market of carrier-specific, neutral, and gray pools. Draymen often face "split moves"—picking up a bare chassis at one depot and hauling it to another terminal for a container. During import surges, empty containers are stranded on chassis in warehouse parking lots because marine terminals restrict empty returns (due to yard congestion), locking up chassis assets off-dock and starving the port of bare chassis needed to pull new imports.
- **Weak Answer:** "North American chassis manufacturers do not make enough trailers, so ports simply run out of equipment when ships arrive."

### Question 4: Hydrodynamic Mechanics of Barge Freight Derating
- **Prompt:** "Explain how a 1.0-meter drop in water level at the Kaub river gauge impacts Rhine container barge logistics, and what operational adjustments intermodal carriers make."
- **Strong Answer:** Shallow-water navigation is governed by Under-Keel Clearance and hydrodynamic squat ($\Delta_{squat} \propto V^2 / (A_C - A_M)$). When the reference gauge drops by 1.0 meter, permissible barge operating draft drops proportionally. Because container barges are displacement vessels with flat-bottom rectangular hulls, payload capacity is linear with immersion draft ($\approx 7.5 \text{ t/cm}$). A 1.0-meter draft loss eliminates 750 metric tons of cargo capacity per barge, reducing TEU payload by 50% to 70%. In response, barge operators enforce tiered Low-Water Surcharges (KWZ) to compensate for lost revenue per vessel rotation, deploy coupled shallow-draft pushers, and initiate synchromodal diversion—shifting high-priority containers to rail corridors (such as the Betuweroute) and highway drayage.
- **Weak Answer:** "The boats move slower because they might hit the riverbed, so they charge higher shipping rates."

### Question 5: Triangulation (Street Turn) Barriers Across Shipping Alliances
- **Prompt:** "What prevents an intermodal motor carrier from executing a street turn every time an import delivery is close to an export pickup?"
- **Strong Answer:** Street turns face five barriers:
  1. *Equipment Line Ownership:* If the import container is owned or leased by Maersk (Gemini Cooperation) and the export booking is with ONE (Premier Alliance), the import box cannot be used for the export load without an explicit bilateral interchange clearing agreement.
  2. *Container Physical Quality:* Import consumer goods can leave residual debris, floor stains, or interior moisture, rendering the box unsuitable for food-grade, agricultural, or high-value export stuffing without steam cleaning at an empty depot.
  3. *Temporal Misalignment:* Export container loading cutoffs may be several days after the import box is unpacked; holding the box on a chassis incurs chassis per-diem fees that exceed the street-turn drayage savings.
  4. *Customs and Regulatory Hold:* In-bond containers cannot be reloaded until customs release confirmations are recorded in the terminal operating system.
  5. *Carrier Booking Allocation:* Ocean carriers often restrict empty equipment in specific inland zones to prioritize repositioning empty boxes back to high-yield Asian export hubs.
- **Weak Answer:** "Truck drivers prefer driving back to the port because they get paid per mile traveled."

### Question 6: High-Cube Double-Stack Rail Economics
- **Prompt:** "What is the engineering and economic impact of raising overhead clearance along a rail corridor to allow double-stacking of 9'6\" High-Cube containers?"
- **Strong Answer:** Standard intermodal well cars accommodate two 8'6\" standard containers with an overhead rail structure clearance of 18'2\" (5.54 m). However, 9'6\" High-Cube containers are the dominant standard in global trade. Double-stacking two High-Cube boxes requires a minimum vertical clearance of 20'6\" (6.25 m) above top-of-rail, necessitating bridge raising, tunnel floor lowering, and catenary modifications. Economically, double-stacking doubles the volumetric capacity of a train path without doubling train length, fuel consumption, or locomotive crew costs. Train aerodynamic drag increases by only 20–25% compared to a single-stack train, while payload capacity increases by 100%. Line-haul operating costs per TEU fall by 35% to 45%, dramatically lowering the break-even distance against long-haul road drayage from 800 km down to under 500 km.
- **Weak Answer:** "Double stacking lets you carry two boxes instead of one, which cuts rail shipping costs in half."

### Question 7: Alameda Corridor Financing and Volume Throttling
- **Prompt:** "How does the Alameda Corridor Authority finance its capital bonds, and how does that fee structure influence drayage mode choice?"
- **Strong Answer:** The Alameda Corridor Transportation Authority (ACTA) financed its \$2.4B construction cost primarily through long-term municipal revenue bonds. These bonds are debt-serviced via a mandatory statutory corridor fee levied on every loaded container moving into or out of the Ports of Los Angeles and Long Beach by rail (originating or terminating outside a 200-mile local catchment zone). The fee is indexed annually to CPI. If the ACTA rail corridor fee rises too high relative to local highway drayage rates, it can induce modal distortion—incentivizing cargo owners to dray containers by road past the 200-mile boundary to inland rail ramps in the Inland Empire (e.g., San Bernardino or Colton) to avoid the corridor fee, thereby undermining the corridor's original environmental objective of removing heavy trucks from regional highways.
- **Weak Answer:** "The state of California pays for the Alameda Corridor using taxpayer fuel taxes."

### Question 8: Empty Container Inventory Accumulation at Inland Depots
- **Prompt:** "A distant dry port experiences a chronic accumulation of empty containers, reaching 95% storage capacity. What operations engineering interventions resolve this?"
- **Strong Answer:** Resolving empty container accumulation requires a four-part intervention:
  1. *Economic Levers:* Impose progressive, tiered storage dwell surcharges on ocean carriers after free time expires (e.g., \$10/day for days 1–5, \$50/day for days 6–10, \$150/day thereafter) to incentivize carrier evacuation.
  2. *Evacuation Sweeper Trains:* Charter dedicated empty "sweeper" block trains, negotiating discounted backhaul rail freight rates with railroads to haul empty flats back to marine terminals.
  3. *Expanded Triangulation:* Deploy digital matching platforms to connect inland agricultural and industrial bulk exporters with available empty dry vans, providing chassis fee credits for street-turned boxes.
  4. *Gate Inflow Throttling:* Enforce a strict "one-for-one" gate policy: motor carriers cannot return an empty container to the dry port unless they simultaneously pull an outbound loaded or empty container.
- **Weak Answer:** "Stack the containers higher using bigger reach stackers and ask the shipping lines to pick them up."

---

## 15. Key terms

- **Intermodal Freight Transport:** Moving cargo in an unbroken ISO container across multiple modes without handling cargo during mode changes.
- **Hinterland:** The continental market territory served by a seaport for import distribution and export cargo generation.
- **Dry Port:** An inland intermodal terminal directly connected to a seaport by rail/barge, offering customs clearance and storage.
- **Inland Container Depot (ICD):** A common-user inland facility under customs bond offering handling and temporary storage of containers.
- **On-Dock Rail:** Intermodal rail tracks located directly inside a marine terminal footprint, eliminating highway drayage.
- **Near-Dock Rail:** An intermodal rail facility adjacent to port boundaries, accessed via short drayage or dedicated access roads.
- **Off-Dock ICTF:** An Intermodal Container Transfer Facility located inland, requiring public highway drayage and dual gate moves.
- **Drayage:** Short-distance road trucking of containerized freight between marine terminals, rail ramps, and inland facilities.
- **Chassis:** A specialized skeletal semi-trailer engineered to securely haul ISO containers on public roadways.
- **Split Move (Chassis Split):** A drayage maneuver requiring a driver to pick up or return a chassis at a facility separate from the container depot.
- **Neutral Chassis Pool:** A shared chassis fleet managed by a third-party equipment lessor, open to multiple motor carriers.
- **Gray Chassis Pool:** An interoperable cooperative chassis pool where any chassis can be used beneath any shipping line's container.
- **Push-Barge Convoy:** A motorized towboat pushing rigid configurations of non-powered barges coupled tightly by steel cables.
- **Kaub Gauge:** Critical reference water gauge on the Middle Rhine that dictates allowable vessel draft and cargo loading limits.
- **Low-Water Surcharge (KWZ):** Tiered freight surcharge applied by inland barge operators when river gauge depths fall below published thresholds.
- **Lock Cycle Time:** Total duration required for an inland vessel to enter, fill/empty, and exit a navigation lock chamber.
- **Modal Split Logit Model:** Discrete choice econometric formulation predicting mode share probabilities based on random utility maximization.
- **Systematic Utility ($V_{im}$):** The deterministic component of utility derived from measurable attributes such as freight cost, transit time, and reliability.
- **Value of Time (VOT):** The monetary valuation shippers place on saving one hour of transit time ($\beta_{time} / \beta_{cost}$).
- **Value of Reliability (VOR):** The monetary valuation shippers place on reducing transit variance or increasing on-time arrival rate.
- **Direct Price Elasticity:** The percentage change in a mode's demand resulting from a 1% change in its own freight rate.
- **Cross-Price Elasticity:** The percentage change in a mode's demand resulting from a 1% price change in a competing transport mode.
- **Triangulation (Street Turn):** Direct dispatch of an empty import container to an export shipper for reloading, bypassing the seaport depot.
- **Empty Container Repositioning (Repo):** Logistics of moving empty non-revenue containers from surplus consumption areas to deficit export regions.
- **Corridor Metering / Throttling:** Dynamic restriction of freight volume admitted to an inland corridor to prevent exceeding bottleneck capacity.
- **PierPass:** Operational traffic mitigation program levying off-peak gate incentives at Los Angeles and Long Beach ports.
- **UIIA:** Uniform Intermodal Interchange and Facilities Access Agreement governing equipment exchange and liability.
- **In-Bond Transit (CBP 7512):** Customs procedure allowing cargo to move inland without paying duties or undergoing clearance at the seaport of entry.
- **Synchromodality:** Real-time dynamic switching of freight consignments between road, rail, and barge based on live network conditions.
- **Double-Stack Rail:** Stacking containers two-high on specialized well cars, doubling train capacity under compatible overhead clearances.

---

## 16. Sources

- **Ben-Akiva, M., & Lerman, S. R.** (1985). *Discrete Choice Analysis: Theory and Application to Travel Demand*. MIT Press. ISBN: 978-0262022170
- **Bierwirth, C., & Meisel, F.** (2015). A follow-up survey of berth allocation and quay crane scheduling problems in container terminals. *European Journal of Operational Research*, 244(3), 675–689. https://doi.org/10.1016/j.ejor.2014.12.030
- **Central Commission for the Navigation of the Rhine (CCNR).** (2024). *Inland Navigation in Europe: Market Observation Annual Report*. CCNR Publishing. https://www.ccr-zkr.org
- **De Langen, P. W., & Chouly, A.** (2004). Hinterland access regimes in seaports. *European Journal of Transport and Infrastructure Research*, 4(4), 361–380. https://doi.org/10.18757/ejtir.2004.4.4.4363
- **Federal Highway Administration (FHWA).** (2022). *Comprehensive Truck Size and Weight Limits Study*. U.S. Department of Transportation. 23 CFR Part 658.
- **Federal Maritime Commission (FMC).** (2022). *Demurrage and Detention Billing Requirements under the Ocean Shipping Reform Act of 2022*. Docket No. 22-04. 46 U.S.C. 41104.
- **Intermodal Association of North America (IANA).** (2025). *Uniform Intermodal Interchange and Facilities Access Agreement (UIIA)*. IANA Standards Board.
- **International Maritime Organization (IMO).** (2016). *Guidelines Regarding the Verified Gross Mass of a Container Carrying Cargo*. MSC.1/Circ.1475.
- **Notteboom, T., & Rodrigue, J.-P.** (2005). Port regionalization: Towards a new phase in port development. *Maritime Policy & Management*, 32(3), 297–313. https://doi.org/10.1080/03088830500139885
- **Roso, V., Woxenius, J., & Lumsden, K.** (2009). The dry port concept: Connecting container seaports with the hinterland. *Journal of Transport Geography*, 17(5), 338–345. https://doi.org/10.1016/j.jtrangeo.2008.10.008
- **Talley, W. K.** (2009). *Port Economics*. Routledge. ISBN: 978-0415777223
- **van Riessen, B., Negenborn, R. R., & Dekker, R.** (2015). Synchromodal container transportation: An overview of current trends and research opportunities. *Transportation Research Part E: Logistics and Transportation Review*, 79, 137–152. https://doi.org/10.1016/j.tre.2015.04.004

---

## 17. Links

- **Parent Module:** [Module 41: Maritime, Rail, and Air Cargo Operations Engineering](_index.md)
- **Module Survey File:** [41_maritime_rail_air_operations.md](../../../course/41_maritime_rail_air_operations.md)
- **Preceding Quayside Operations:** [Topic 01: Berth and Quay Crane Operations](01_berth-quay-crane.md)
- **Preceding Yard and Gate Operations:** [Topic 02: Terminal Yard and Gate Operations](02_terminal-yard-gate.md)
- **Preceding Rail Line Operations:** [Topic 03: Rail Network and Path Capacity](03_rail-network-paths.md)
- **Rail Marshalling and Shunting:** [Topic 04: Rail Yards and Shunting](04_rail-yards-shunting.md)
- **Downstream Operations Recovery:** [Topic 07: Disruption and Recovery](07_disruption-recovery.md)
- **Terminal Discrete-Event Simulation:** [Topic 08: Terminal Simulation](08_terminal-simulation.md)
- **Related Elective Module:** [Module 12: Transportation, Routing, and Last Mile](../../02_methods/12_transportation_routing_last_mile/_index.md)
- **Related Elective Module:** [Module 36: Global Freight, Ports, and Trade Policy](../../06_domain_electives/36_global_freight_ports_trade_policy/_index.md)
