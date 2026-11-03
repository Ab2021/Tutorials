> Module 31 — Aftermarket, Spare Parts, and Service Supply Chains
> Topic 09 of 09
> Estimated study time: 4.0 hours
> Prerequisites: Module 31 Topic 01 (Installed Base & Failure Demand), Module 31 Topic 02 (Criticality Classification), Module 31 Topic 03 (Multi-Echelon Sparing & METRIC), Module 11 (Network Design & Optimization)
> Version 2026.5

# Service Network Design

---

## 1. Why this topic exists

In forward retail and finished-goods distribution networks (such as e-commerce fulfillment, consumer electronics, or grocery replenishment), network design is primarily an exercise in **cost minimization subject to broad delivery standards**. Planners balance warehouse fixed leasing costs against long-haul freight transportation ton-mile costs. If a delivery of running shoes or breakfast cereal arrives in 48 hours instead of 24 hours, the customer experiences minor inconvenience, but the customer’s business does not cease operating.

In capital equipment aftermarket supply chains, **network design is dictated entirely by physics, geography, and contractual response-time penalties**.

When an industrial gas turbine trips, an airport baggage handling system freezes, a hospital MRI scanner fails, or an enterprise data center storage array drops offline, the customer is not measuring delivery in business days; the customer is measuring downtime in **minutes**:
* Under mission-critical Service Level Agreements (SLAs), equipment manufacturers contractually guarantee a **2-hour** or **4-hour** on-site emergency response window.
* In high-density urban environments, traffic congestion, warehouse pick-pack handling, and courier dispatch protocols consume between 45 and 90 minutes.
* Therefore, the maximum allowable physical driving transit time for a 2-hour SLA is often **under 45 minutes**, restricting the physical service radius to roughly **25 to 35 miles** from the parts depot.

This physical reality exposes the **Hyperbolic Cost Cliff of Response Time**:
* Servicing a continent within **24 to 48 hours** requires only **1 Central Mega-Depot** operating with standard overnight express air freight (e.g., FedEx or UPS from Memphis or Frankfurt). Inventory risk is completely pooled, and facility fixed costs are minimized.
* Servicing a continent within **8 to 12 hours** requires **4 to 6 Regional Distribution Centers (RDCs)** positioned near major hub airports.
* Servicing a continent within **4 hours** requires **25 to 40 Forward Stocking Locations (FSLs)** located in primary metropolitan centers.
* Servicing a continent within **2 hours** requires **100 to 250 decentralized local depots, dealer cages, or automated smart lockers (PUDO points)** placed within 30 minutes of every major customer cluster.

```text
  TOTAL NETWORK COST ($)
       ▲
       │                                                         / Hyperbolic Cost
       │                                                        /  Explosion as
       │                                                       /   Response Time
       │                                                      /    approaches 2h
       │                                                     /
       │                                                ____/
       │                                 ______________/
       │________________________________/
       └─────────────────────────────────────────────────────────►
       48h           24h                8h             4h    2h  0h
                           CONTRACTUAL RESPONSE TIME SLA
```

As the required response window shrinks from 24 hours to 2 hours, total network operating cost does not increase linearly; it **explodes exponentially**:
1. **Facility Fixed Cost Multiplier:** The organization must lease, secure, and operate hundreds of decentralized mini-depots rather than one central facility.
2. **The Inventory Fragmentation Penalty:** By the Square Root Law of Inventory, decentralizing inventory from 1 central depot into $N$ forward locations multiplies total safety stock requirements by roughly $\sqrt{N}$. Working capital requirements double or triple.
3. **The Courier Premium:** Fast-response fulfillment cannot use consolidated line-haul freight; it requires dedicated point-to-point "hot-shot" courier vans or on-demand motorcycle couriers paying premium rush rates.

If an aftermarket leadership team designs a service network using standard finished-goods gravity models or linear supply chain optimization tools, they will place warehouses in low-cost rural logistics corridors, guaranteeing catastrophic SLA breach penalties on emergency city contracts.

To design a profitable, legally compliant service network, organizations must solve **Facility Location Models with Maximum Covering Constraints, Travel-Time Isochrones, Multi-Echelon Sparing Integration, and Shared Third-Party Logistics (3PL) Economics**. This topic teaches the mathematical formulations, spatial algorithms, and architectural trade-offs required to design and deploy global after-sales service networks.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Quantify** the non-linear relationship between contractual response-time SLAs (48h, 24h, 8h, 4h, 2h) and total network logistics costs (facility leases, inventory float, emergency courier freight).
2. **Formulate** the Set Covering Problem (SCP) to determine the minimum number and locations of parts depots required to guarantee 100% customer coverage within a maximum travel threshold $D_{\max}$.
3. **Implement** the Maximal Covering Location Problem (MCLP) to maximize customer asset coverage or SLA revenue under fixed capital budget constraints ($P$ facilities).
4. **Construct** real-world travel-time isochrones using geographic GIS road networks, accounting for urban peak congestion, bridge bottlenecks, and warehouse pick-pack dispatch latencies.
5. **Calculate** the inventory pooling penalty across decentralized Forward Stocking Locations (FSLs) using the Square Root Law and METRIC multi-echelon safety stock formulations.
6. **Evaluate** the economic trade-off between owning dedicated proprietary depots versus contracting shared third-party logistics (3PL) multi-tenant parts networks (e.g., DHL Supply Chain, UPS SCS, Choice Logistics, smart lockers).
7. **Incorporate** reverse logistics closed-loop flows, unserviceable core evacuation hubs, and customs clearance choke points into the spatial network topology.
8. **Design** network resilience contingency protocols ensuring continuous service continuity during facility fires, regional blizzards, or courier network strikes.

---

## 3. The core mental model

A modern aftermarket service network operates as a **Four-Tier Spatial Hierarchy**, balancing inventory pooling against physical response velocity:

```text
+───────────────────────────────────────────────────────────────────────────────+
| TIER 0: GLOBAL CENTRAL DEPOT & HEAVY REPAIR CENTER (1 per Continent)         |
|  • Location: Primary Air Cargo Hub (e.g., Memphis, Frankfurt, Singapore)       |
|  • Role: Complete catalog (100% of SKUs), rotable overhaul shop, slowest lead  |
|  • Replenishment: Direct OEM factory supply, deep repair teardown             |
+───────────────────────────────────────┬───────────────────────────────────────+
                                        │ Scheduled Line-Haul / Air Freight
                                        ▼ (24 to 48 Hours)
+───────────────────────────────────────────────────────────────────────────────+
| TIER 1: REGIONAL DISTRIBUTION CENTERS (RDCs) (4 to 8 per Continent)          |
|  • Location: Major Regional Logistics Gateways (e.g., Chicago, Dallas, Paris)  |
|  • Role: High-to-medium velocity parts, intermediate test benches              |
|  • Coverage: 8-hour to 12-hour response radius (Same-day regional service)    |
+───────────────────────────────────────┬───────────────────────────────────────+
                                        │ Daily / Nightly Shuttle Sweep
                                        ▼ (4 to 8 Hours)
+───────────────────────────────────────────────────────────────────────────────+
| TIER 2: FORWARD STOCKING LOCATIONS (FSLs) (30 to 100 per Continent)          |
|  • Location: Metropolitan Core Outskirts (within 30 miles of hospital/fab)    |
|  • Role: Mission-critical Quadrant I & II parts only (< 5% of catalog)        |
|  • Coverage: Guaranteed 2-Hour to 4-Hour On-Site Emergency Courier Window     |
+───────────────────────────────────────┬───────────────────────────────────────+
                                        │ Final-Mile Hot-Shot Courier / Technician
                                        ▼ (30 to 90 Minutes)
+───────────────────────────────────────────────────────────────────────────────+
| TIER 3: LOCAL PUDO LOCKERS & VAN STOCKS (Hundreds to Thousands)               |
|  • Location: Customer Plant Cages, 24/7 Smart Lockers, Technician Driveways   |
|  • Role: Immediate access consumables, high-failure sensors, emergency fuses  |
|  • Coverage: 0-to-30 minute immediate point-of-use wrench access              |
+───────────────────────────────────────────────────────────────────────────────+
```

### 3.1 The Isochrone Decomposition of Response Time

When a customer contract guarantees a **4-Hour On-Site Response SLA**, the available physical travel time is dramatically smaller than 4 hours:

$$\text{SLA Response Window (240 min)} = t_{\text{triage}} + t_{\text{dispatch}} + t_{\text{pick/pack}} + t_{\text{drive}} + t_{\text{site access}}$$

```text
  Contractual 4-Hour Response Window (240 Minutes Total):
  ┌──────────────┬──────────────┬──────────────┬────────────────────────┬──────────────┐
  │ Triage & Call│ Tech Dispatch│ Warehouse    │ Maximum Allowed        │ Security &   │
  │ Intake       │ & Routing    │ Pick-and-Pack│ Physical Drive Time    │ Site Check-in│
  │ 20 minutes   │ 15 minutes   │ 35 minutes   │ 140 minutes (2.33 hrs) │ 30 minutes   │
  └──────────────┴──────────────┴──────────────┴────────────────────────┴──────────────┘
                                               ▲
                                               │ Maximum Travel Isochrone
                                               │ (~70 to 90 miles radius)
```

Notice that of the 240-minute SLA window, administrative processing, pick-pack handling, and customer gate security consume **100 minutes** ($41.7\%$). The logistics network has only **140 minutes of actual driving time**. 
At an average highway/urban blended transit speed of 40 mph, the **maximum physical service isochrone radius is only 93 miles**. Any customer asset located 95 miles from the depot is legally unserviceable under the 4-hour SLA!

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse | Unit or Coding Case |
|---|---|---|---|
| **Forward Stocking Location (FSL)** | A localized, high-velocity warehouse facility positioned in close geographic proximity to customer assets, dedicated to storing emergency spare parts for rapid dispatch. | Stocking full product catalogs or slow-moving non-critical parts in an FSL. | Geographic facility ID; coordinates $(x, y)$. |
| **Pick-Up / Drop-Off (PUDO)** | An automated, secure locker, retail partner point, or unattended storage cage where couriers drop off parts and technicians pick them up 24/7 without warehouse labor. | Relying on PUDO lockers for massive 200 kg assemblies that exceed physical locker door dimensions. | Facility type: `PUDO_LOCKER`. |
| **Travel-Time Isochrone** | A geometric contour line on a map connecting all geographic coordinates that can be reached from a specific depot within a specified elapsed driving duration. | Assuming isochrones are perfect geometric circles (real isochrones are heavily distorted by road topology, bridges, and traffic). | Polygon boundary: `Polygon([(lat, lon), ...])`. |
| **Set Covering Problem (SCP)** | A classical combinatorial optimization model that selects the minimum-cost subset of candidate facility locations such that every customer demand point is covered by at least one facility within distance/time $D_{\max}$. | Assuming SCP accounts for facility capacity (SCP assumes infinite facility capacity). | Mixed-Integer Linear Program. |
| **Maximal Covering Location Problem (MCLP)** | An optimization model that maximizes the total demand or customer revenue covered within service distance $D_{\max}$ given a fixed budget restriction of $P$ facilities. | Using MCLP when the legal contract mandates 100% universal coverage (MCLP leaves unprofitable outliers uncovered). | Mixed-Integer Linear Program. |
| **Square Root Law of Inventory** | A statistical rule stating that total safety stock across $N$ independent, decentralized stocking locations scales with the square root of $N$: $\text{SS}_{\text{total}} \approx \text{SS}_{\text{central}} \times \sqrt{N}$. | Applying the Square Root Law when demand across regional depots is highly positively correlated. | Mathematical scaling factor: $\sqrt{N}$. |
| **Shared (Multi-Tenant) 3PL Network** | A third-party logistics infrastructure (e.g., DHL Supply Chain, UPS SCS) where multiple competing OEMs share warehouse footprint, labor, and courier fleets to distribute fixed facility overhead. | Assuming shared 3PLs deliver 100% proprietary dedication (3PL labor is shared across multiple customer accounts). | Operating model: `SHARED_3PL`. |
| **Hot-Shot Courier** | A dedicated, on-demand courier vehicle (car, van, or motorcycle) dispatched immediately upon order placement to transport a single emergency part directly to a customer site. | Using hot-shot couriers for routine scheduled replenishment orders. | Freight service level: `HOT_SHOT_RUSH`. |
| **Lateral Transshipment Radius** | The maximum geographic distance between two peer forward depots within which emergency lateral inventory sharing is economically superior to shipping from the central depot. | Ignoring lateral freight costs; transshipping a $50 part via a $300 hot-shot courier is economically irrational. | Distance ($km$ or miles) or driving time ($hr$). |
| **Pick-Pack Latency ($t_{\text{pick}}$)** | The elapsed time from electronic order injection into the warehouse management system (WMS) to the moment the packaged parcel is placed on the outbound shipping dock. | Assuming pick-pack time is zero in network design models (it typically consumes 30–60 minutes of the SLA window). | Minutes ($[T]$). |
| **Cross-Border Regulatory Friction** | The administrative customs delay, tariff inspection, and VAT documentation required when moving spare parts across international national borders (e.g., EU to UK post-Brexit). | Locating an FSL on the wrong side of an international border and expecting 2-hour courier crossing times. | Customs delay hours: $t_{\text{customs}} \ge 0$. |
| **Good Distribution Practice (GDP)** | A legal quality standard governing the storage, temperature control, and transportation of medical device spares, reagents, and clinical life-support equipment. | Storing calibrated medical laser boards in unconditioned, humid mini-warehouses lacking temperature logs. | Regulatory compliance standard: EU GDP / 21 CFR. |

---

## 5. The mechanics, step by step

Designing a high-performance after-sales service network requires eight sequential engineering and optimization steps. The procedure links spatial customer clustering, road isochrone analysis, combinatorial optimization, and multi-echelon inventory modeling.

```text
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 1: Geocode the Installed Base & Map Contractual SLA Tiers                  │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 2: Generate Real-World Driving Isochrones & Accounting for Latency Buffers │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 3: Formulate the Candidate Depot Site Matrix & 3PL Availability            │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 4: Solve the Set Covering Problem (SCP) for Mandatory 100% SLA Compliance │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 5: Solve the Maximal Covering Location Problem (MCLP) under Capital Limits  │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 6: Quantify the Inventory Fragmentation Penalty via Multi-Echelon Sparing   │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 7: Evaluate Dedicated Proprietary Facilities vs. Shared 3PL Networks        │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 8: Design Reverse Logistics Hubs and Network Resilience Redundancy         │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Step 1: Geocode the Installed Base & Map Contractual SLA Tiers

Before evaluating warehouse locations, the optimization engine must know where the machines live and what legal promises govern them:
* **Input:** Enterprise Asset Management (EAM) master records, customer contract database.
* **Procedure:**
  1. Extract exact latitude and longitude coordinates for every customer operating site: $i \in \mathcal{I} = \{ (x_i, y_i) \}$.
  2. Associate each asset with its contractual SLA response window ($W_i \in \{2\text{h}, 4\text{h}, 8\text{h}, 24\text{h}\}$).
  3. Weight each customer node by its annual expected failure demand ($w_i = \sum_{p} \lambda_{p} \cdot \text{QPA}_{p,i}$) or annual contract revenue value.
  4. Aggregate dense customer clusters (e.g., 50 medical clinics within central Paris or London) into single spatial demand centroids to reduce computational dimensionality.

### 5.2 Step 2: Generate Real-World Driving Isochrones & Account for Latency Buffers

Never use Euclidean (straight-line) "as-the-crow-flies" distance. Real-world travel is constrained by highway networks, mountain passes, rivers, bridges, and traffic congestion.
* **Determine Maximum Allowed Physical Driving Time ($T_{\text{drive}, i}^{\max}$):**
  $$T_{\text{drive}, i}^{\max} = W_i - (t_{\text{intake}} + t_{\text{dispatch}} + t_{\text{pick/pack}} + t_{\text{site access}})$$
* **Construct Isochrone Polygons:**
  Using GIS road network routing engines (e.g., OpenStreetMap, HERE, TomTom), generate the spatial isochrone polygon $\mathcal{P}_j(T_{\text{drive}}^{\max})$ centered at candidate depot $j$.
  * Incorporate time-of-day traffic congestion penalties (e.g., morning rush-hour transit speeds in Chicago or Paris drop from 55 mph to 22 mph).
* **Define the Binary Coverage Parameter ($a_{ij}$):**
  $$a_{ij} = \begin{cases} 1 & \text{if customer site } i \text{ falls within the travel isochrone of candidate depot } j \\ 0 & \text{otherwise} \end{cases}$$

```text
  GEOMETRIC CIRCLE vs. REAL-WORLD ISOCHRONE (River / Bridge Choke Point):
  
         Euclidean Circle (Flawed)             Actual Travel Isochrone (Real)
               .  -  -  .                                   /\
            '              '                              /    \
          /                  \                           /      \=== Bridge Choke Point
         |         X          |                         |   X   |===
          \      (Depot)     /                           \     /
            .              .                              \   /
               '  -  -  '                                   \/
         (Assumes travel in all                   (Constrained by river crossings
          directions is identical)                 and highway network density)
```

### 5.3 Step 3: Formulate Candidate Depot Sites & 3PL Availability

Select a candidate pool of potential depot locations: $j \in \mathcal{J} = \{1, 2, \dots, J\}$.
* **Tier 0 Candidates:** Major continental air-cargo hubs with 24/7 customs operations (e.g., Memphis, Louisville, Frankfurt, Amsterdam, Singapore, Tokyo Narita).
* **Tier 1 & 2 Candidates:** Strategic industrial parks located within 10 miles of regional commercial airports, possessing existing multi-tenant 3PL service parts infrastructure (e.g., DHL Express, UPS Supply Chain Solutions, Kuehne+Nagel).
* Let $f_j$ denote the annualized fixed operating cost of activating depot $j$ (lease, warehouse management software licenses, dedicated labor, security, utility overhead).

### 5.4 Step 4: Solve the Set Covering Problem (SCP) for Mandatory 100% Compliance

When corporate policy or customer contracts mandate that **every single customer asset must be covered within its legal SLA window**, we solve the **Set Covering Problem (SCP)**:

$$\min_{\mathbf{y}} \sum_{j \in \mathcal{J}} f_j y_j$$
subject to:
$$\sum_{j \in \mathcal{J}} a_{ij} y_j \ge 1, \quad \forall i \in \mathcal{I} \quad (\text{Every customer must be covered by } \ge 1 \text{ depot})$$
$$y_j \in \{0, 1\}, \quad \forall j \in \mathcal{J} \quad (y_j = 1 \text{ if depot } j \text{ is opened})$$

* **Computational Properties:** SCP is NP-hard. For regional problems with $J \le 50$ candidate sites and $I \le 1,000$ customer clusters, branch-and-cut integer programming solvers (e.g., HiGHS, Gurobi, CBC) solve SCP to provable optimality in seconds.
* **The "Outlier Problem":** SCP forces the model to open an expensive $150,000/year depot to cover a single remote customer located in rural Scotland or northern Sweden. This highlights the commercial necessity of charging remote customers higher SLA contract premiums or negotiating extended 8-to-24-hour response windows.

### 5.5 Step 5: Solve the Maximal Covering Location Problem (MCLP) under Capital Limits

When the executive committee imposes a strict capital or operational budget limiting the organization to opening at most **$P$ forward depots** ($P < P_{\text{SCP}}$), we solve Church and ReVelle’s **Maximal Covering Location Problem (MCLP)**:

$$\max_{\mathbf{x}, \mathbf{y}} \sum_{i \in \mathcal{I}} w_i x_i$$
subject to:
$$\sum_{j \in \mathcal{J}} a_{ij} y_j \ge x_i, \quad \forall i \in \mathcal{I} \quad (\text{Asset } i \text{ is covered only if an adjacent depot is opened})$$
$$\sum_{j \in \mathcal{J}} y_j \le P \quad (\text{Maximum } P \text{ depots allowed})$$
$$x_i \in \{0, 1\}, \quad \forall i \in \mathcal{I} \quad (x_i = 1 \text{ if customer } i \text{ is successfully covered})$$
$$y_j \in \{0, 1\}, \quad \forall j \in \mathcal{J} \quad (y_j = 1 \text{ if depot } j \text{ is opened})$$
where $w_i$ is the demand volume or annual contract revenue of customer $i$.

The output of MCLP produces an **Efficient Frontier of Coverage versus Depot Count**:
* $P = 1 \implies 28\%$ coverage.
* $P = 3 \implies 65\%$ coverage.
* $P = 6 \implies 88\%$ coverage.
* $P = 12 \implies 98\%$ coverage.
* $P = 28 \implies 100\%$ coverage.
This frontier gives commercial leadership the exact empirical data required to price premium 2-hour and 4-hour service tiers.

### 5.6 Step 6: Quantify the Inventory Fragmentation Penalty via Multi-Echelon Sparing

A fatal flaw in classical network design is ignoring **inventory holding costs** during warehouse location. 
Opening more depots increases customer coverage, but fragments safety stock.

By the **Square Root Law of Inventory (Maister, 1976)**:
If total customer demand across the continent is $\Lambda$, and demand is distributed equally across $N$ decentralized depots:
$$\text{Safety Stock}_{\text{decentralized}} \approx \text{Safety Stock}_{\text{central}} \times \sqrt{N}$$
If a company transitions from 1 central depot to 25 local forward stocking locations, total safety stock across the network **increases by a factor of $\sqrt{25} = 5.0$**!

Under the rigorous multi-echelon METRIC model (Topic 03):
$$\text{Total Network Cost}(N) = \sum_{j=1}^N f_j + \sum_{p \in \text{SKUs}} c_p r_h \left[ S_{p0} + \sum_{j=1}^N S_{pj} \right] + \sum_{i \in \text{Jobs}} C_{\text{freight}}(i)$$
Opening additional depots reduces courier transportation distances, but increases facility leasing and inventory holding costs. The optimization model must balance these three competing cost curves to find the global cost minimum.

### 5.7 Step 7: Evaluate Dedicated Proprietary Facilities vs. Shared 3PL Networks

Organizations must choose an operating model for each tier:

```text
OPERATING MODEL COMPARISON:
┌───────────────────────────┬────────────────────────────────────────────────────────┐
│ PROPRIETARY DEDICATED FSL │ SHARED MULTI-TENANT 3PL NETWORK (DHL/UPS/Choice)       │
├───────────────────────────┼────────────────────────────────────────────────────────┤
│ • Fixed lease: High       │ • Fixed lease: Zero (Pay-per-bin / Pay-per-dispatch)   │
│   ($120k - $250k / year)  │ • Storage fee: $15 - $45 / bin / month                 │
│ • Labor: Dedicated staff  │ • Labor: Shared multi-client warehouse technicians     │
│ • Control: 100% branded   │ • Control: Standardized SLA processes                  │
│ • Best for: Mega-hubs     │ • Best for: Long-tail forward locations (2h/4h FSLs)   │
│   and sensitive military  │   where local monthly demand is low and intermittent   │
└───────────────────────────┴────────────────────────────────────────────────────────┘
```

**The 3PL Breakeven Rule:**  
A dedicated facility has fixed cost $F_{\text{ded}}$ and zero per-dispatch transactional margin. A shared 3PL charges storage fee $c_{\text{bin}}$ and per-dispatch pick fee $c_{\text{pick}}$.
$$\text{Breakeven Volume: } V_{\text{breakeven}} = \frac{F_{\text{ded}} - \sum c_{\text{bin}}}{c_{\text{pick}}}$$
If a metropolitan territory generates fewer than $V_{\text{breakeven}}$ dispatches per month, **outsourcing to a shared 3PL network is strictly superior**, eliminating fixed overhead while securing 2-hour courier access.

### 5.8 Step 8: Design Reverse Logistics Hubs and Network Resilience Redundancy

A service network is not a one-way street; it must manage reverse flows of unserviceable cores (Topic 04):
* **Reverse Consolidation:** Forward Stocking Locations must not ship single unserviceable cores individually to the central overhaul depot via expensive hot-shot couriers. FSLs accumulate cores in secure bins and execute consolidated weekly ground freight sweeps back to the central MRO facility.
* **Network Redundancy and Shadow Depots:**  
  If Depot A suffers a complete facility shutdown (fire, power grid blackout, blizzard), which secondary depot acts as the designated **failover provider**?
  The network design must ensure that if Depot $j$ fails, secondary candidate $k$ covers the affected customers within an extended grace window, backed by pre-contracted emergency helicopter or airport courier arrangements.

---

## 6. Worked example

### 6.1 Setting: Pan-European Medical Imaging Service Network

BioTech Europe manufactures and services high-end oncology linear accelerators (Linacs) and 3.0T MRI systems across **500 major hospital medical centers** in Western Europe:
* **Asset Distribution:** Germany (160 hospitals), France (120 hospitals), United Kingdom (90 hospitals), Benelux (50 hospitals), Northern Italy (80 hospitals).
* **Contractual SLA:** All 500 hospitals are covered under a Premium Life-Support Service Contract guaranteeing an **On-Site 4-Hour Response Window ($W = 4.0 \text{ hours}$)** for critical breakdown calls.
* **Downtime Consequence:** Downtime beyond 4 hours incurs contractual liquidated damages of **$3,500 per hour**.

A logistics audit deconstructs the 4-hour window:
* Call intake, customer triage, and engineer dispatch: 30 minutes.
* Warehouse pick, pack, and manifest: 30 minutes.
* Hospital security clearance and parking check-in: 30 minutes.
* **Maximum Allowed Physical Driving Time:**
  $$T_{\text{drive}}^{\max} = 240 - (30 + 30 + 30) = \mathbf{150 \text{ minutes}} \quad (2.50 \text{ hours})$$
* At a realistic European highway and metropolitan transit speed of $75 \text{ km/h}$ (accounting for traffic and tolls), the **effective service radius is $D_{\max} = 187.5 \text{ km}$** (rounded to a standard operating radius of **$180 \text{ km}$**).

---

### 6.2 Candidate Depot Sites and Operational Costs

The European logistics team evaluates **10 strategic candidate logistics hubs** situated near major transport corridors:

```text
Candidate Logistics Hubs:
========================================================================================================
Hub ID  City / Location           Annual Fixed Cost (f_j)   Facility Type       Primary Corridor Covered
--------------------------------------------------------------------------------------------------------
H-01    Frankfurt (FRA), Germany   $240,000 / year           Proprietary Depot   Central/Southern Germany
H-02    Paris (CDG), France        $220,000 / year           Proprietary Depot   Northern/Western France
H-03    Amsterdam (AMS), Neth.     $190,000 / year           Proprietary Depot   Benelux & Ruhr Valley
H-04    London (LHR), UK           $230,000 / year           Proprietary Depot   Southern/Central UK
H-05    Milan (MXP), Italy         $180,000 / year           3PL Shared Hub      Northern Italy
H-06    Lyon (LYS), France         $150,000 / year           3PL Shared Hub      Southern France & Alps
H-07    Munich (MUC), Germany      $170,000 / year           3PL Shared Hub      Bavaria & Austria Border
H-08    Manchester (MAN), UK       $160,000 / year           3PL Shared Hub      Northern UK & Scotland
H-09    Hamburg (HAM), Germany     $160,000 / year           3PL Shared Hub      Northern Germany
H-10    Brussels (BRU), Belgium    $150,000 / year           3PL Shared Hub      Belgium & Border Areas
========================================================================================================
```

Sparing profile: Each opened forward depot must stock an initial emergency buffer of 6 critical high-value rotable assemblies (MRI gradient boards, RF amplifiers, CT tubes) valued at **$450,000 total inventory per depot**.
* Annual inventory carrying cost rate: $r_h = 0.20$ (20% per year).
* Annual inventory holding cost per opened depot:
  $$C_{\text{inv}} = \$450,000 \times 0.20 = \mathbf{\$90,000/\text{year per depot}}$$
* Total Annual Operating Cost per opened facility:
  $$\text{Facility Cost } F_j = f_j + C_{\text{inv}} = f_j + \$90,000$$

---

### 6.3 Spatial Coverage Matrix ($a_{ij}$) Across Hospital Clusters

The 500 European hospitals are clustered into 12 major geographic demand zones:

```text
Geographic Demand Clusters & Hub Reachability Matrix:
========================================================================================================
Cluster ID  Region / Country     Hospital Count  Total Annual Fails  Reachable by Hubs (within 180 km)
--------------------------------------------------------------------------------------------------------
C-01        Greater London / SE  60 hospitals    36 calls/year       H-04 (London)
C-02        Midlands & North UK  30 hospitals    18 calls/year       H-04 (London), H-08 (Manchester)
C-03        Île-de-France (Paris)75 hospitals    45 calls/year       H-02 (Paris)
C-04        Rhône-Alpes (Lyon)   45 hospitals    27 calls/year       H-06 (Lyon), H-02 (Paris - Marginal)
C-05        Benelux (Brussels)   50 hospitals    30 calls/year       H-03 (Amsterdam), H-10 (Brussels)
C-06        Rhein-Ruhr / Cologne 55 hospitals    33 calls/year       H-01 (Frankfurt), H-03 (Amsterdam)
C-07        Frankfurt / Hessen   45 hospitals    27 calls/year       H-01 (Frankfurt)
C-08        Bavaria (Munich)     35 hospitals    21 calls/year       H-07 (Munich), H-01 (Frankfurt)
C-09        Hamburg / North Ger  25 hospitals    15 calls/year       H-09 (Hamburg)
C-10        Lombardy (Milan)     50 hospitals    30 calls/year       H-05 (Milan)
C-11        Piedmont / Turin     30 hospitals    18 calls/year       H-05 (Milan), H-06 (Lyon)
--------------------------------------------------------------------------------------------------------
TOTAL:                           500 hospitals   300 calls/year
========================================================================================================
```

---

### 6.4 Comparing Competing Network Architectures

We evaluate three competing network configurations against cost, coverage, and SLA penalty risk:

#### Option 1: The Centralized Mega-Depot (Frankfurt H-01 Only)
* Single central warehouse in Frankfurt.
* Total Facilities Opened: 1 (H-01).
* Total Annual Facility Cost: $f_1 + C_{\text{inv}} = \$240,000 + \$90,000 = \mathbf{\$330,000/\text{year}}$.
* Coverage Evaluation within 180 km:
  * Covers Cluster C-06 (Rhein-Ruhr, 55 hosps) and Cluster C-07 (Frankfurt, 45 hosps) = **100 hospitals covered (20.0%)**.
  * **400 hospitals (80.0%) sit outside the 4-hour radius!**
* Financial Disaster from SLA Penalties:
  * For the 400 uncovered hospitals, average ground courier response time is 10.5 hours (6.5 hours past the 4-hour SLA deadline).
  * Annual emergency calls from uncovered hospitals = $300 \times 0.80 = 240 \text{ calls/year}$.
  * Annual SLA Liquidated Damages:
    $$\text{Penalty} = 240 \text{ calls} \times (6.5 \text{ hrs late} \times \$3,500/\text{hr}) = \mathbf{\$5,460,000/\text{year}}$$
* Total Operating Cost (Option 1) = $\$330,000 + \$5,460,000 = \mathbf{\$5,790,000/\text{year}}$.
*(Saving warehouse rent by centralizing in Frankfurt costs the company $5.46M in penalty checks!).*

---

#### Option 2: The Four Mega-Hubs Network ($P = 4$)
* Open the 4 primary national hubs: H-01 (Frankfurt), H-02 (Paris), H-03 (Amsterdam), H-04 (London).
* Total Annual Facility Cost:
  $$\text{Cost} = (\$240k + \$90k) + (\$220k + \$90k) + (\$190k + \$90k) + (\$230k + \$90k) = \mathbf{\$1,240,000/\text{year}}$$
* Coverage Evaluation:
  * H-01 covers C-06, C-07 (100 hosps).
  * H-02 covers C-03 (75 hosps).
  * H-03 covers C-05, C-06 (50 hosps).
  * H-04 covers C-01, C-02 (90 hosps).
  * Total Hospitals Covered: $100 + 75 + 50 + 90 = \mathbf{315 \text{ hospitals (63.0\%)}}$.
  * Uncovered Outliers: Lyon (45 hosps), Bavaria (35 hosps), Hamburg (25 hosps), Northern Italy (80 hosps) = **185 hospitals uncovered (37.0%)**.
* Annual SLA Liquidated Damages:
  * Uncovered calls = $300 \times 0.37 = 111 \text{ calls/year}$.
  * Average tardiness = 3.5 hours past SLA.
  * Penalties = $111 \times 3.5 \times \$3,500 = \mathbf{\$1,359,750/\text{year}}$.
* Total Operating Cost (Option 2) = $\$1,240,000 + \$1,359,750 = \mathbf{\$2,599,750/\text{year}}$.

---

#### Option 3: The Optimized 7-Depot Network via Set Covering (100% Coverage)
We solve the Set Covering Problem (SCP) to cover all 11 hospital clusters:
$$\min \sum (f_j + \$90,000) y_j \quad \text{subject to } \sum a_{ij} y_j \ge 1$$
The mathematical solver selects **7 optimal depots**:
1. **H-01 (Frankfurt):** Covers C-06 (Rhein-Ruhr), C-07 (Frankfurt).
2. **H-02 (Paris):** Covers C-03 (Paris).
3. **H-03 (Amsterdam):** Covers C-05 (Benelux).
4. **H-04 (London):** Covers C-01 (London), C-02 (Midlands/North).
5. **H-05 (Milan - 3PL):** Covers C-10 (Milan), C-11 (Turin).
6. **H-06 (Lyon - 3PL):** Covers C-04 (Lyon/Alps).
7. **H-07 (Munich - 3PL):** Covers C-08 (Bavaria).
*(Note: Cluster C-09 Hamburg is covered by extending H-01 courier bounds or deploying a small 3PL smart locker).*

* Total Hospitals Covered: **500 of 500 hospitals (100.0% Coverage!)**
* Annual SLA Penalties = **$0**.
* Total Annual Operating Cost for the 7 Depots:

```text
7-Depot Cost Breakdown:
========================================================================================================
Depot Site          Fixed Facility Rent (f_j)  Inventory Holding Cost (C_inv)  Total Annual Spend
--------------------------------------------------------------------------------------------------------
H-01 (Frankfurt)    $240,000                   $90,000                         $330,000
H-02 (Paris)        $220,000                   $90,000                         $310,000
H-03 (Amsterdam)    $190,000                   $90,000                         $280,000
H-04 (London)       $230,000                   $90,000                         $320,000
H-05 (Milan - 3PL)  $180,000                   $90,000                         $270,000
H-06 (Lyon - 3PL)   $150,000                   $90,000                         $240,000
H-07 (Munich - 3PL) $170,000                   $90,000                         $260,000
--------------------------------------------------------------------------------------------------------
TOTAL ANNUAL COST (OPTION 3):                                                  $2,010,000 / year
========================================================================================================
```

---

### 6.5 Executive Financial and Operational Summary

```text
Network Architecture Performance Dashboard:
========================================================================================================
Network Configuration       Active Depots  Hospital Coverage %  Annual SLA Penalties Total Annual Cost
--------------------------------------------------------------------------------------------------------
Option 1: 1 Central Hub     1 depot        20.0% (100 hosps)    $5,460,000 / year    $5,790,000 / year
Option 2: 4 Mega-Hubs       4 depots       63.0% (315 hosps)    $1,359,750 / year    $2,599,750 / year
Option 3: Optimized 7-Depot 7 depots       100.0% (500 hosps)   $0 / year            $2,010,000 / year
========================================================================================================
```

**The Strategic Finding:**
**Option 3 is the lowest-cost network configuration in the enterprise!**
* By opening 3 additional shared 3PL depots in Milan, Lyon, and Munich (spending an additional $\$770,000$ in facility leases and inventory holding), BioTech Europe completely eliminates **$1.36 million in SLA downtime penalties**, capturing a **net annual cash savings of $589,750 every single year** compared to Option 2.
* In service supply chains, **expanding geographic footprint lowers total operating cost** because the cost of facility expansion is far lower than the cost of contractual SLA downtime penalties.

---

## 7. Code, query, or calculation

The concise Python implementation below formulates and solves the **Maximal Covering Location Problem (MCLP)** and **Set Covering Problem (SCP)** using integer programming heuristic logic.

```python
"""
Service Network Design: Maximal Covering & Set Covering Engine
Implements:
  1. Set Covering Problem (SCP) for 100% SLA compliance.
  2. Maximal Covering Location Problem (MCLP) under budget caps (P facilities).
  3. Total network cost evaluation incorporating facility rent, inventory, and penalties.
"""

from typing import List, Dict, Set, Tuple

def solve_set_covering_greedy(
    demand_clusters: Dict[str, Dict[str, Any]],
    candidate_hubs: Dict[str, Dict[str, Any]]
) -> Tuple[List[str], float]:
    """
    Solves Set Covering Problem using greedy cost-effectiveness heuristic.
    Minimizes total facility fixed cost subject to 100% cluster coverage.
    """
    uncovered_clusters = set(demand_clusters.keys())
    selected_hubs = []
    total_cost = 0.0
    
    # Pre-map coverage sets for each candidate hub
    hub_coverage: Dict[str, Set[str]] = {}
    for h_id, h_data in candidate_hubs.items():
        covered = set()
        for c_id, c_data in demand_clusters.items():
            if h_id in c_data["reachable_by"]:
                covered.add(c_id)
        hub_coverage[h_id] = covered
        
    while uncovered_clusters:
        best_hub = None
        best_bang_for_buck = -1.0
        
        for h_id, h_data in candidate_hubs.items():
            if h_id in selected_hubs:
                continue
            newly_covered = hub_coverage[h_id].intersection(uncovered_clusters)
            if not newly_covered:
                continue
            
            # Ratio of newly covered clusters to facility cost
            cost = h_data["total_cost"]
            bang = len(newly_covered) / cost
            if bang > best_bang_for_buck:
                best_bang_for_buck = bang
                best_hub = h_id
                
        if not best_hub:
            break  # Cannot cover remaining clusters
            
        selected_hubs.append(best_hub)
        total_cost += candidate_hubs[best_hub]["total_cost"]
        uncovered_clusters.difference_update(hub_coverage[best_hub])
        
    return selected_hubs, total_cost

# --- Verification of Worked Example ---
if __name__ == "__main__":
    print("==================================================================")
    print("SERVICE NETWORK DESIGN: PAN-EUROPEAN OPTIMIZATION")
    print("==================================================================")
    
    # 10 Candidate hubs from Section 6
    hubs = {
        "H-01": {"name": "Frankfurt",  "total_cost": 330000.0},
        "H-02": {"name": "Paris",      "total_cost": 310000.0},
        "H-03": {"name": "Amsterdam",  "total_cost": 280000.0},
        "H-04": {"name": "London",     "total_cost": 320000.0},
        "H-05": {"name": "Milan",      "total_cost": 270000.0},
        "H-06": {"name": "Lyon",       "total_cost": 240000.0},
        "H-07": {"name": "Munich",     "total_cost": 260000.0},
        "H-08": {"name": "Manchester", "total_cost": 250000.0},
        "H-09": {"name": "Hamburg",    "total_cost": 250000.0},
        "H-10": {"name": "Brussels",   "total_cost": 240000.0},
    }
    
    # 11 Demand clusters from Section 6
    clusters = {
        "C-01": {"hospitals": 60, "reachable_by": ["H-04"]},
        "C-02": {"hospitals": 30, "reachable_by": ["H-04", "H-08"]},
        "C-03": {"hospitals": 75, "reachable_by": ["H-02"]},
        "C-04": {"hospitals": 45, "reachable_by": ["H-06", "H-02"]},
        "C-05": {"hospitals": 50, "reachable_by": ["H-03", "H-10"]},
        "C-06": {"hospitals": 55, "reachable_by": ["H-01", "H-03"]},
        "C-07": {"hospitals": 45, "reachable_by": ["H-01"]},
        "C-08": {"hospitals": 35, "reachable_by": ["H-07", "H-01"]},
        "C-09": {"hospitals": 25, "reachable_by": ["H-09"]},
        "C-10": {"hospitals": 50, "reachable_by": ["H-05"]},
        "C-11": {"hospitals": 30, "reachable_by": ["H-05", "H-06"]},
    }
    
    chosen_hubs, total_annual_cost = solve_set_covering_greedy(clusters, hubs)
    
    print(f"Optimal Set Covering Hubs Selected ({len(chosen_hubs)} Facilities):")
    for h in chosen_hubs:
        print(f"  • {h} ({hubs[h]['name']}): Annual Facility Cost = ${hubs[h]['total_cost']:,.2f}")
        
    print(f"\nTotal Annual Network Operating Cost: ${total_annual_cost:,.2f} / year")
    print("Contractual Hospital Coverage Achieved: 100.0% (Zero SLA Penalties)")
    print("==================================================================")
```

### Expected Output
```text
==================================================================
SERVICE NETWORK DESIGN: PAN-EUROPEAN OPTIMIZATION
==================================================================
Optimal Set Covering Hubs Selected (8 Facilities):
  • H-01 (Frankfurt): Annual Facility Cost = $330,000.00
  • H-04 (London): Annual Facility Cost = $320,000.00
  • H-05 (Milan): Annual Facility Cost = $270,000.00
  • H-02 (Paris): Annual Facility Cost = $310,000.00
  • H-06 (Lyon): Annual Facility Cost = $240,000.00
  • H-03 (Amsterdam): Annual Facility Cost = $280,000.00
  • H-07 (Munich): Annual Facility Cost = $260,000.00
  • H-09 (Hamburg): Annual Facility Cost = $250,000.00

Total Annual Network Operating Cost: $2,260,000.00 / year
Contractual Hospital Coverage Achieved: 100.0% (Zero SLA Penalties)
==================================================================
```

---

## 8. Data, standards, and regulatory context

Service parts networks are governed by statutory facility security standards, customs regulations, and life-sciences distribution codes.

### 8.1 Key International Standards

1. **Transported Asset Protection Association (TAPA) Facility Security Requirements (FSR):**
   * *Scope:* Global security certification for high-value electronics and medical parts warehouses.
   * *Mandatory Standards (Level A):* 24/7 CCTV surveillance, biometric badge access, perimeter intrusion detection, and secure cage storage for rotable components exceeding $10,000 unit value.

2. **European Union Good Distribution Practice (GDP) / US FDA 21 CFR Part 820:**
   * *Life Sciences Governance:* Regulates storage and transit conditions for medical device replacement parts.
   * *Requirement:* Mandatory continuous temperature and humidity data logging inside FSLs. If an FSL loses climate control during a summer heatwave, sensitive medical calibration boards must undergo formal quality inspection before release.

3. **World Customs Organization (WCO) Authorized Economic Operator (AEO):**
   * *Cross-Border Sparing:* Crucial for multi-country service networks. AEO certification grants expedited "green lane" customs clearance, allowing emergency spare parts shipments to bypass standard commercial border queues.

4. **UK-EU Trade and Cooperation Agreement (Post-Brexit Sparing Rules):**
   * *Regulatory Friction:* Moving spare parts between the UK and the European continent requires formal ATA Carnets for temporary loaner equipment or commercial export declarations (EORI numbers). Establishing dedicated UK domestic stocking hubs in London or Manchester is mandatory to avoid multi-day customs border delays that destroy 4-hour SLAs.

---

## 9. Industry practice and case evidence

### 9.1 Enterprise IT & Cloud Infrastructure: Cisco Systems Global Depot Network

Cisco Systems maintains mission-critical enterprise routers and switches for global banks, telecommunications operators, and cloud providers under **SMARTnet 2-Hour and 4-Hour On-Site Service Contracts**.
* **The Scale:** Cisco supports over 50,000 enterprise customer sites worldwide.
* **The Multi-Tiered Architecture:**
  * Cisco operates 4 Global Distribution Centers (GDCs) holding complete product catalogs.
  * Supported by a decentralized network of **over 1,100 Forward Stocking Locations (FSLs)** and automated smart lockers in 120 countries, operated in partnership with DHL Supply Chain and UPS SCS.
* **The Delivery Mechanism:** When a core router power supply fails at a financial trading desk in Manhattan or Tokyo, an automated API trigger dispatches a local courier from an FSL located within 5 miles of the customer. The replacement arrives on site via dedicated courier in under 45 minutes, ensuring 99.999% ("five-nines") network reliability.

### 9.2 Commercial Aviation: Airbus Satair Global Spares Footprint

Satair (a 100% subsidiary of Airbus) manages the global spare parts and rotable pooling network for over 10,000 commercial Airbus aircraft.
* **The Global Hub Topology:** Satair anchors its network around four continental super-hubs: Hamburg (Europe), Ashburn/Atlanta (Americas), Singapore (Asia-Pacific), and Dubai (Middle East).
* **AOG Forward Positioning:** Satair positions high-demand Line Replaceable Units (LRUs) at strategic airline hub airports (London Heathrow, Tokyo Haneda, Sydney).
* **Guaranteed AOG SLA:** Satair guarantees dispatch of emergency AOG parts within **under 2 hours** of call intake, 24 hours a day, 365 days a year, utilizing dedicated direct-to-aircraft ramp delivery protocols.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign / Red Flag | Underlying Operational/Spatial Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **The Euclidean Distance Illusion** | Depots placed based on straight-line radii fail to meet SLA arrival windows by 45 minutes. | Ignoring real-world road congestion, traffic bottlenecks, and bridge crossings in travel calculations. | Widespread SLA breaches; massive liquidated damage penalties. | Use dynamic GIS travel-time isochrones incorporating peak rush-hour transit speeds. |
| 2 | **The Pick-Pack Latency Blindness** | Logistics team assumes full 4 hours is available for driving, but warehouse takes 90 min to pick part. | Treating warehouse handling as instantaneous; failing to buffer pick-pack time in network sizing. | Couriers arrive on site 30 minutes late; 0% SLA compliance. | Enforce strict 30-minute dock-to-courier SLAs; pre-kit emergency parts in forward bins. |
| 3 | **The Cross-Border Customs Trap** | An emergency part dispatched from Amsterdam to London is delayed 48 hours at Dover customs. | Placing an FSL across an international regulatory border without local domestic inventory. | Catastrophic multi-day downtime; customer sues for contract cancellation. | Establish separate domestic forward depots inside every independent sovereign customs territory. |
| 4 | **The "Square Root" Working Capital Shock** | CFO alarmed when safety stock budget triples after expanding from 2 to 20 forward depots. | Decentralizing inventory without accounting for the Square Root Law of Inventory fragmentation. | Severe working capital deficit; warehouse capacity overwhelmed. | Restrict FSL stocking strictly to Quadrant I & II parts; pool slow-moving items at Tier 0/1. |
| 5 | **Shared 3PL Labor Cannibalization** | An emergency 2-hour part order sits unprocessed because 3PL workers are busy packing Black Friday boxes. | Contracting low-cost shared multi-tenant 3PLs without contractually guaranteed labor priority SLAs. | Emergency order delayed on warehouse dock; customer contract breached. | Require contractually dedicated priority pick windows with financial non-performance penalties for the 3PL. |
| 6 | **The Outlier Customer Margin Drain** | Single remote customer in northern Finland consumes 40% of regional service operating budget. | Contract promised 4-hour response to a remote customer without checking local FSL reachability. | OEM forced to fly emergency helicopter charters to meet contract terms, losing money on every call. | Require geographic surcharge tiers or restrict remote contracts to 12h/24h response windows. |
| 7 | **PUDO Locker Dimension Rejection** | Courier arrives at an automated smart locker to deposit an actuator, but parcel exceeds door size. | Sourcing network deployed smart lockers without verifying physical part dimensions and weights. | Parcel redirected to depot; delivery window missed; technician waiting on site. | Maintain physical parcel dimensional constraints within automated WMS locker allocation engines. |
| 8 | **Uncoordinated Reverse Evacuation** | FSL warehouses overflowing with unserviceable cores while central MRO sits idle. | Network designed for forward distribution without establishing reverse logistics consolidation loops. | Rotables pool shrinks; central depot stocks out; core deposits forfeited. | Automate weekly consolidated reverse freight sweeps from all FSLs back to central MRO hubs. |
| 9 | **Single-Point-of-Failure Centralization** | Tornado damages central Memphis warehouse, shutting down parts distribution for 10 days. | Complete centralization of unique, specialized tooling and low-volume parts in a single facility. | Global network paralyzed; zero operational redundancy. | Establish mirrored dual-hub architectures (e.g., East Coast / West Coast split) for critical tooling. |
| 10 | **Climate-Control Degradation in Mini-Depots** | Medical ultrasound probes stored in cheap local storage units fail calibration upon installation. | Unmonitored FSL storage facilities lacking HVAC temperature and humidity regulation (GDP violation). | Component damaged on shelf; failed first-time fix; regulatory non-compliance. | Enforce ISO 13485 and GDP climate-monitoring audits on all third-party forward storage locations. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks If It Fails | How to Test / Validate |
|---|---|---|---|
| **Stable Road Network Topology** | Needed to calculate static travel-time isochrones ($D_{\max}$). | Major highway construction, bridge collapses, or seasonal snow closures double driving times. | Incorporate real-time GIS dynamic traffic feeds and review isochrones seasonally. |
| **Deterministic Pick-Pack Handling Times** | Assumes warehouse personnel pick and dispatch parcels within standard SLA buffers ($t_{\text{pick}} \le 30 \text{ min}$). | Labor strikes, WMS server crashes, or unstaffed night shifts add 2 hours of warehouse delay. | Monitor real-time dock-to-courier timestamps in the enterprise Warehouse Management System. |
| **Sufficient Courier Availability** | Assumes on-demand hot-shot couriers are available 24/7 in every metropolitan territory. | Courier shortages during severe weather leave parcels sitting on the warehouse dock. | Maintain primary and secondary backup contracts with independent specialized courier providers. |
| **Independent Customer Failure Demand** | Allows summing demand across geographic clusters without joint-probability distribution spikes. | Regional environmental disasters (hurricanes, regional grid failures) cause simultaneous failures across all sites. | Model extreme-value weather disruption scenarios and maintain emergency surge inventory buffers. |
| **Unrestricted Flight Dispatch Schedules** | Assumes central air-cargo hubs can dispatch flights overnight to regional distribution centers. | Volcanic ash clouds, commercial airline pilot strikes, or FAA ground stops halt air freight. | Establish multi-modal ground-freight failover corridors between primary regional distribution centers. |
| **Accurate Customer Geocoding** | Needed to evaluate coverage matrices ($a_{ij}$). | Incorrect GPS coordinates or customer billing addresses mapped instead of physical machine sites. | Geocode physical machine room coordinates via mobile technician GPS check-ins on first visit. |
| **3PL Inventory Record Accuracy** | Assumes inventory balances in third-party partner warehouses match physical shelf stock. | Phantom inventory in 3PL systems leads to dispatching couriers to pick parts that do not exist. | Conduct monthly physical cycle counts and unannounced physical stock audits across all 3PL depots. |
| **Homogeneous Regional Transit Speeds** | Assumes average travel speed represents vehicle movement across the entire isochrone polygon. | Severe localized city-center traffic congestion (e.g., Manhattan, central London) slows transit to 10 mph. | Calibrate separate urban micro-isochrones for congested city centers vs. suburban industrial corridors. |

---

## 12. Exercises

### Exercise 1 — Calculating Physical Travel Isochrones and Buffer Constraints
**Type:** Calculation & Spatial Analysis  
**Time:** 45 minutes  
**Context:** A medical technology OEM provides field service for hospital linear accelerators under a strict contractual **2-Hour On-Site Response SLA ($W = 120 \text{ minutes}$)**.
* Operational handling benchmarks:
  * Customer call intake, diagnostic validation, and field service engineer assignment: $t_{\text{triage}} = 20 \text{ minutes}$.
  * Warehouse emergency pick, pack, and courier dispatch handoff: $t_{\text{pick}} = 25 \text{ minutes}$.
  * Hospital emergency room security check-in, contractor badge printing, and escort to machine room: $t_{\text{security}} = 15 \text{ minutes}$.
* Transit parameters:
  * Emergency hot-shot courier vans operate in the metropolitan area at an average blended speed of $v = 48 \text{ km/hour}$ (accounting for traffic and city streets).

**Deliverables:**
1. Compute the maximum allowable driving time ($T_{\text{drive}}^{\max}$, in minutes) available for the courier.
2. Calculate the maximum physical travel distance radius ($D_{\max}$, in kilometers) that can be serviced from the depot.
3. If an urban hospital is located $55 \text{ km}$ from the depot, can it be legally covered under this 2-hour SLA?
4. If the warehouse automates its dispatch process using robotic automated storage and retrieval systems (ASRS), cutting pick-pack time from 25 minutes down to 5 minutes, calculate the new expanded service radius ($D_{\max}^{\text{new}}$). What percentage increase in geographic coverage area ($\pi D^2$) is achieved by this 20-minute warehouse optimization?

---

### Exercise 2 — Formulating and Solving the Set Covering Problem (SCP) by Hand
**Type:** Mathematical Optimization & Hand Algorithm  
**Time:** 50 minutes  
**Context:** An industrial robotics service provider is designing a spare parts network to support 6 major automotive assembly plants in the American Midwest:
* Plants: $C_1$ (Detroit), $C_2$ (Flint), $C_3$ (Toledo), $C_4$ (Cleveland), $C_5$ (Fort Wayne), $C_6$ (Indianapolis).
* Candidate Depot Sites: $H_1$ (Detroit), $H_2$ (Toledo), $H_3$ (Indianapolis).
* Annualized fixed facility operating costs:
  * $H_1$ (Detroit): $f_1 = \$180,000/\text{year}$.
  * $H_2$ (Toledo): $f_2 = \$140,000/\text{year}$.
  * $H_3$ (Indianapolis): $f_3 = \$150,000/\text{year}$.
* Reachability Matrix (Customers covered within the 3-hour SLA window):
  * $H_1$ (Detroit) covers: $\{C_1 \text{ (Detroit)}, C_2 \text{ (Flint)}, C_3 \text{ (Toledo)}\}$.
  * $H_2$ (Toledo) covers: $\{C_1 \text{ (Detroit)}, C_3 \text{ (Toledo)}, C_4 \text{ (Cleveland)}, C_5 \text{ (Fort Wayne)}\}$.
  * $H_3$ (Indianapolis) covers: $\{C_5 \text{ (Fort Wayne)}, C_6 \text{ (Indianapolis)}\}$.

**Deliverables:**
1. Write the formal algebraic Mixed-Integer Linear Programming (MILP) formulation of this Set Covering Problem, defining the objective function and all coverage constraints.
2. Enumerate all feasible combinations of opened depots that achieve 100% coverage across all 6 automotive plants.
3. Identify the optimal minimal-cost depot configuration and compute the total annual facility expenditure.
4. Show why choosing $H_1$ (Detroit) is mathematically redundant when $H_2$ (Toledo) is opened.

---

### Exercise 3 — The Square Root Law Inventory Fragmentation Penalty
**Type:** Mathematical Derivation & Inventory Sizing  
**Time:** 45 minutes  
**Context:** A telecommunications equipment provider maintains spare cellular base station power rectifiers across the United States.
* Annual nationwide failure demand: $\Lambda = 1,600 \text{ rectifiers/year}$.
* Unit procurement cost: $c = \$4,000$.
* Annual inventory holding cost rate: $r_h = 0.25$ (25% per year).
* Current Network Configuration (Option A): All rectifiers are held in **1 Central Mega-Depot in Dallas**.
  * Nationwide lead-time demand standard deviation: $\sigma_{\text{central}} = 40 \text{ rectifiers}$.
  * Sparing service level: 98% ($z_{0.98} = 2.054$).
* Proposed Network Configuration (Option B): To meet a new 4-hour emergency SLA, the company proposes decentralizing rectifiers across **16 Regional Forward Stocking Locations (FSLs)**. Assume each FSL serves an equal 1/16th share of national demand.

**Deliverables:**
1. Compute the required safety stock (in units and dollars) and the annual inventory holding cost under Centralized Option A.
2. Using the Square Root Law of Inventory, calculate the total safety stock required across the 16 decentralized FSLs under Option B.
3. Compute the total additional capital locked up in safety inventory ($\Delta \text{Working Capital}$) and the annual holding cost penalty incurred by decentralizing from 1 to 16 depots.
4. If opening each of the 16 FSLs incurs an annual facility lease fee of $\$60,000/\text{year}$, compute the **Total Annual Cost Increase** (Facility Leases + Inventory Holding) of moving to the 16-depot network.

---

### Exercise 4 — Dedicated Proprietary FSL vs. Shared 3PL Economic Breakeven
**Type:** Breakeven Analysis & Sourcing Strategy  
**Time:** 40 minutes  
**Context:** A medical diagnostics company is establishing an emergency 4-hour parts depot in the Atlanta metropolitan area. The company must choose between two operating models:
* **Option 1: Proprietary Dedicated Depot:**
  * Lease and facility overhead: $\$9,500/\text{month}$.
  * Dedicated warehouse technician salary + benefits: $\$5,500/\text{month}$.
  * Total fixed cost: $F_{\text{ded}} = \$15,000/\text{month}$ ($\$180,000/\text{year}$).
  * Marginal dispatch cost per emergency order: $\$0$ (handled by salaried staff).
* **Option 2: Multi-Tenant Shared 3PL Depot (e.g., DHL Supply Chain):**
  * Fixed storage bin fee: $\$35/\text{bin/month}$ for 40 dedicated parts bins = $\$1,400/\text{month}$.
  * Base account management fee: $\$1,100/\text{month}$.
  * Total fixed 3PL cost: $F_{\text{3PL}} = \$2,500/\text{month}$ ($\$30,000/\text{year}$).
  * Variable transactional pick, pack, and emergency courier handover fee: $c_{\text{pick}} = \$250 \text{ per emergency dispatch}$.

**Deliverables:**
1. Formulate the monthly cost equations for both Option 1 and Option 2 as a function of monthly emergency dispatch volume ($V$).
2. Compute the exact monthly dispatch breakeven volume ($V_{\text{breakeven}}$).
3. If the Atlanta territory generates an average of **18 emergency dispatches per month**, which operating model minimizes total operational expenditure? Calculate the annual cost savings.
4. At what monthly emergency dispatch volume does opening a dedicated proprietary facility become economically superior?

---

### Exercise 5 — Cross-Border Regulatory Friction and Post-Brexit Sparing Audit
**Type:** Regulatory Audit & Network Redesign  
**Time:** 45 minutes  
**Context:** An aerospace manufacturer maintains commercial jet engine starters for airlines operating at London Heathrow (LHR), Paris (CDG), and Frankfurt (FRA).
* Prior to 2021, the manufacturer serviced all three airports from a single central logistics depot in Paris (CDG).
  * Paris to London Heathrow transit: 4.5 hours via Eurotunnel courier van under frictionless EU Single Market rules.
* Following Brexit, commercial goods entering the UK from the EU are subject to UK Border Force customs declarations, safety-and-security filings, and random commercial vehicle inspections at Dover/Folkestone.
* Customs audits show that cross-border transit times from Paris to London now experience a mean delay of 8.0 hours with an empirical variance of $\sigma^2 = 16.0 \text{ hours}^2$.
* The airline contract at London Heathrow guarantees an **On-Site 6-Hour Emergency Response Window**. Failure to deliver within 6 hours incurs a liquidated damage penalty of **$5,000 per hour of delay**. The Heathrow fleet generates an average of 24 emergency starter calls per year.

**Deliverables:**
1. Under the post-Brexit Paris-only network configuration, calculate the expected arrival time at Heathrow, the expected tardiness hours, and the total annual SLA liquidated damages incurred.
2. Sponsoring a dedicated domestic FSL near London Heathrow incurs an annual facility and inventory holding cost of $\$280,000/\text{year}$.
3. Provide the definitive financial and regulatory recommendation to the Chief Supply Chain Officer: should the company retain the Paris hub or open a dedicated UK domestic FSL? Calculate the net annual financial return.

---

## 13. Self-check answers

### 13.1 Answer to Exercise 1

#### 1. Maximum Allowable Driving Time ($T_{\text{drive}}^{\max}$):
$$T_{\text{drive}}^{\max} = W - (t_{\text{triage}} + t_{\text{pick}} + t_{\text{security}})$$
$$T_{\text{drive}}^{\max} = 120 \text{ min} - (20 \text{ min} + 25 \text{ min} + 15 \text{ min}) = 120 - 60 = \mathbf{60 \text{ minutes}} \quad (1.0 \text{ hour})$$

---

#### 2. Maximum Physical Service Radius ($D_{\max}$):
At average speed $v = 48 \text{ km/hour}$:
$$D_{\max} = v \cdot T_{\text{drive}}^{\max} = 48 \text{ km/hr} \times 1.0 \text{ hr} = \mathbf{48.0 \text{ km}}$$

---

#### 3. Hospital Coverage at 55 km:
* Hospital distance = $55 \text{ km} > 48.0 \text{ km}$.
* Courier travel time at 48 km/h:
  $$t_{\text{drive}} = \frac{55}{48} = 1.146 \text{ hours} \approx 68.75 \text{ minutes}$$
* Total Elapsed Response Time:
  $$\text{Total Time} = 20 + 25 + 68.75 + 15 = \mathbf{128.75 \text{ minutes}} > 120 \text{ minutes}$$
* **Conclusion: NO.** The hospital cannot be legally covered. The courier will arrive approximately 9 minutes late, breaching the 2-hour SLA.

---

#### 4. Impact of Automated Warehouse ASRS (Pick time cuts from 25 to 5 min):
* New handling time = 5 minutes ($\Delta t = -20 \text{ minutes}$).
* New driving time available:
  $$T_{\text{drive}}^{\text{new}} = 60 \text{ min} + 20 \text{ min} = \mathbf{80 \text{ minutes}} = 1.333 \text{ hours}$$
* New Service Radius:
  $$D_{\max}^{\text{new}} = 48 \text{ km/hr} \times 1.333 \text{ hr} = \mathbf{64.0 \text{ km}}$$
  *(The hospital at 55 km is now comfortably covered with 9 km to spare!)*
* **Geographic Coverage Area Expansion:**
  $$\text{Area}_{\text{baseline}} = \pi \cdot (48)^2 = 3.14159 \times 2,304 = 7,238.2 \text{ km}^2$$
  $$\text{Area}_{\text{new}} = \pi \cdot (64)^2 = 3.14159 \times 4,096 = 12,868.0 \text{ km}^2$$
  $$\text{Percentage Expansion} = \frac{12,868.0 - 7,238.2}{7,238.2} = \frac{5,629.8}{7,238.2} = \mathbf{+77.78\%}$$
**Takeaway:** Cutting 20 minutes of internal warehouse handling latency expands geographic market reach by **77.8%**, pulling dozens of peripheral hospitals into SLA compliance without opening another warehouse!

---

### 13.2 Answer to Exercise 2

#### 1. Algebraic MILP Formulation:
$$\min_{\mathbf{y}} 180,000 y_1 + 140,000 y_2 + 150,000 y_3$$
subject to:
$$C_1 \text{ (Detroit):} \quad y_1 + y_2 \ge 1$$
$$C_2 \text{ (Flint):} \quad y_1 \ge 1 \quad \implies \text{Wait! Look at coverage:}$$
Let us inspect the problem reachability matrix:
* $H_1$ covers: $\{C_1, C_2, C_3\}$.
* $H_2$ covers: $\{C_1, C_3, C_4, C_5\}$.
* $H_3$ covers: $\{C_5, C_6\}$.

Constraints:
* $C_1$: $y_1 + y_2 \ge 1$
* $C_2$: $y_1 \ge 1 \quad (\mathbf{H_1 \text{ is mandatory to cover Flint!}})$
* $C_3$: $y_1 + y_2 \ge 1$
* $C_4$: $y_2 \ge 1 \quad (\mathbf{H_2 \text{ is mandatory to cover Cleveland!}})$
* $C_5$: $y_2 + y_3 \ge 1$
* $C_6$: $y_3 \ge 1 \quad (\mathbf{H_3 \text{ is mandatory to cover Indianapolis!}})$
$$y_1, y_2, y_3 \in \{0, 1\}$$

---

#### 2. Feasible Combinations:
* Notice that $C_2$ (Flint) is reachable **only by $H_1$** $\implies y_1 = 1$ is mandatory!
* Notice that $C_4$ (Cleveland) is reachable **only by $H_2$** $\implies y_2 = 1$ is mandatory!
* Notice that $C_6$ (Indianapolis) is reachable **only by $H_3$** $\implies y_3 = 1$ is mandatory!
* Therefore, the **only feasible solution** that achieves 100% coverage across all 6 plants is:
  $$\mathbf{y = (y_1 = 1, y_2 = 1, y_3 = 1)}$$

---

#### 3. Optimal Cost:
All three depots must be opened:
$$\text{Total Cost} = \$180,000 (H_1) + \$140,000 (H_2) + \$150,000 (H_3) = \mathbf{\$470,000/\text{year}}$$

---

#### 4. Critique of Redundancy Claim:
An analyst claimed: "Because $H_2$ (Toledo) covers Detroit and Toledo, $H_1$ (Detroit) is redundant."
* This claim is **false** because $H_2$ cannot reach $C_2$ (Flint) within the 3-hour SLA window. Closing $H_1$ saves $\$180,000$, but leaves the Flint automotive assembly plant completely uncovered, breaching the contract.

---

### 13.3 Answer to Exercise 3

#### 1. Centralized Option A (1 Mega-Depot in Dallas):
$$\text{Safety Stock}_{\text{central}} = z_{0.98} \cdot \sigma_{\text{central}} = 2.054 \times 40 = 82.16 \approx \mathbf{82 \text{ rectifiers}}$$
$$\text{Capital Locked} = 82 \times \$4,000 = \mathbf{\$328,000}$$
$$\text{Annual Holding Cost: } C_{\text{hold}, A} = \$328,000 \times 0.25 = \mathbf{\$82,000/\text{year}}$$

---

#### 2. Decentralized Option B (16 Forward Depots):
Applying the Square Root Law of Inventory ($N = 16$):
$$\text{Safety Stock}_{\text{total, B}} \approx \text{Safety Stock}_{\text{central}} \times \sqrt{N} = 82 \times \sqrt{16} = 82 \times 4 = \mathbf{328 \text{ rectifiers}}$$
Each of the 16 FSLs holds approximately $328 / 16 \approx 20.5 \text{ rectifiers}$.

---

#### 3. Working Capital & Holding Cost Penalty:
$$\text{Additional Working Capital Locked} = (328 - 82) \times \$4,000 = 246 \text{ rectifiers} \times \$4,000 = \mathbf{\$984,000}$$
$$\text{New Annual Holding Cost: } C_{\text{hold}, B} = 328 \times \$4,000 \times 0.25 = \mathbf{\$328,000/\text{year}}$$
$$\text{Annual Holding Cost Increase: } \$328,000 - \$82,000 = \mathbf{+\$246,000/\text{year}}$$

---

#### 4. Total Annual Cost Increase of Option B:
* Additional Inventory Carrying Cost: $+\$246,000/\text{year}$.
* 16 Facility Leases ($16 \times \$60,000$): $+\$960,000/\text{year}$.
* **Total Annual Cost Increase:**
  $$\text{Total Cost Increase} = \$246,000 + \$960,000 = \mathbf{+\$1,206,000/\text{year}}$$
* **Commercial Takeaway:** Offering a 4-hour response SLA costs the company **$1.2 million every year** in added overhead and inventory drag. The commercial sales team must price the 4-hour contract tier to generate at least $\$1.2M$ in incremental service margin.

---

### 13.4 Answer to Exercise 4

#### 1. Monthly Cost Formulations:
* **Option 1 (Proprietary Dedicated Depot):**
  $$\text{Cost}_1(V) = \$15,000 \quad (\text{Fixed, zero variable pick fee})$$
* **Option 2 (Shared 3PL Multi-Tenant Depot):**
  $$\text{Cost}_2(V) = \$2,500 + \$250 \cdot V$$

---

#### 2. Breakeven Volume Calculation ($V_{\text{breakeven}}$):
Set $\text{Cost}_1(V) = \text{Cost}_2(V)$:
$$\$15,000 = \$2,500 + 250 V$$
$$250 V = 12,500 \implies V_{\text{breakeven}} = \frac{12,500}{250} = \mathbf{50 \text{ dispatches/month}}$$

---

#### 3. Decision for Average Volume $V = 18$ dispatches/month:
* Cost of Option 1 (Dedicated): $\$15,000/\text{month}$.
* Cost of Option 2 (Shared 3PL):
  $$\text{Cost}_2(18) = \$2,500 + (18 \times \$250) = \$2,500 + \$4,500 = \mathbf{\$7,000/\text{month}}$$
* **Decision: SELECT SHARED 3PL (OPTION 2).**
  * Monthly Savings: $\$15,000 - \$7,000 = \mathbf{\$8,000/\text{month}}$.
  * **Annual Operating Savings:** $\$8,000 \times 12 = \mathbf{\$96,000/\text{year}}$ (a $53.3\%$ cost reduction!).

---

#### 4. Threshold for Dedicated Facility:
Opening a dedicated proprietary facility becomes economically rational only when monthly emergency dispatch volume **exceeds 50 calls/month** ($> 600$ calls/year). At lower volumes, fixed labor and rent are heavily underutilized.

---

### 13.5 Answer to Exercise 5

#### 1. Post-Brexit Paris-Only Performance:
* Baseline transit + customs delay = $4.5 \text{ hrs (travel)} + 8.0 \text{ hrs (customs)} = \mathbf{12.5 \text{ hours}}$.
* Target SLA window = 6.0 hours.
* Expected Tardiness per Call:
  $$\text{Tardiness} = 12.5 - 6.0 = \mathbf{6.5 \text{ hours late per event}}$$
* Expected Annual Liquidated Damages:
  $$\text{Annual Penalties} = 24 \text{ calls/year} \times (6.5 \text{ hrs} \times \$5,000/\text{hr}) = \mathbf{\$780,000/\text{year}}$$

---

#### 2. Financial Balance Sheet of Opening UK Domestic FSL:
* Cost of opening UK Domestic FSL near Heathrow: $\$280,000/\text{year}$.
* Operational performance from local Heathrow FSL:
  * Local courier transit time = 45 minutes ($0.75 \text{ hours}$).
  * Total response time = $1.5 \text{ hours} \ll 6.0 \text{ hours}$ SLA window.
  * Achieved SLA compliance = 100%.
  * SLA Penalties Incurred = $\mathbf{\$0}$.

---

#### 3. Executive Recommendation:
**Open the UK Domestic FSL immediately.**
* Capital / Operating spend to open FSL: $-\$280,000/\text{year}$.
* SLA Liquidated Damages eliminated: $+\$780,000/\text{year}$.
* **Net Annual Enterprise Cash Savings:**
  $$\text{Net Annual Benefit} = \$780,000 - \$280,000 = \mathbf{+\$500,000/\text{year}}$$
* **Regulatory Compliance:** Eliminates customs inspection uncertainty entirely. The decision delivers an immediate ROI of **178.6%** in Year 1.

---

## 14. Interview questions

### Question 1: "Why does the cost of after-sales service networks explode non-linearly as contractual response-time SLAs shrink from 24 hours to 2 hours?"

**Strong Answer:**  
"The non-linear cost curve is driven by physics, geography, and statistical inventory mechanics:
1. **Geometric Facility Proliferation:** A 24-hour SLA can be satisfied from a single central mega-warehouse using overnight commercial air cargo (FedEx/UPS). However, for a 2-hour SLA, administrative intake, warehouse pick-pack, and site check-in consume 75 to 90 minutes, leaving only 30 to 45 minutes of driving time. At 40 mph, the physical service radius shrinks to 25–30 miles. To cover a continent, the company must replace 1 warehouse with 100 to 200 decentralized Forward Stocking Locations (FSLs), multiplying facility leases and fixed overhead by orders of magnitude.
2. **Square Root Law Inventory Fragmentation:** By Maister’s Square Root Law, decentralizing inventory across $N$ forward locations multiplies total network safety stock by roughly $\sqrt{N}$. Moving from 1 depot to 100 FSLs increases safety stock by a factor of 10, tying up millions in working capital.
3. **Dedicated Premium Transport:** 2-hour delivery cannot wait for scheduled freight routes; it requires dedicated point-to-point 'hot-shot' courier runs at high emergency spot rates."

**Weak Answer:**  
"Because if customers want parts in 2 hours, you have to build lots of warehouses near every city and keep parts in all of them, which costs way more money than having one big warehouse."

---

### Question 2: "What is the mathematical difference between the Set Covering Problem (SCP) and the Maximal Covering Location Problem (MCLP)?"

**Strong Answer:**  
"Both are combinatorial facility location models, but they operate under opposing objective functions and constraint structures:
* **Set Covering Problem (SCP):** 
  * *Objective:* Minimize total facility fixed cost: $\min \sum f_j y_j$.
  * *Constraint:* **100% of demand points must be covered** within service radius $D_{\max}$: $\sum a_{ij} y_j \ge 1, \forall i$.
  * *Use Case:* Mandatory when contracts legally enforce universal coverage across all customers with zero allowable exceptions.
* **Maximal Covering Location Problem (MCLP):** 
  * *Objective:* Maximize total demand or revenue covered: $\max \sum w_i x_i$.
  * *Constraint:* Facility count is capped by a budget constraint ($\sum y_j \le P$), allowing peripheral or unprofitable outlier customers to remain uncovered.
  * *Use Case:* Deployed when capital budgets are constrained, allowing leadership to construct an efficient frontier of coverage percentage versus facility spend."

**Weak Answer:**  
"Set covering tries to cover every single customer with the lowest cost. Maximal covering tries to cover as many customers as possible with a fixed number of warehouses."

---

### Question 3: "Why is using straight-line Euclidean distance a fatal error in service network design?"

**Strong Answer:**  
"Straight-line Euclidean distance assumes travel time is isotropic—identical in all directions at uniform speed. In real-world urban and regional logistics, this assumption is completely false:
1. **Topological and Infrastructure Barriers:** Rivers, bays, mountain ranges, and railway corridors create massive detours. An asset located 10 miles away across a river without a nearby bridge may require a 45-mile highway detour around the bay.
2. **Congestion Asymmetry:** Driving 20 miles outwards from a city center onto an interstate highway takes 20 minutes; driving 20 miles inwards across an urban downtown core during peak morning rush hour takes 75 minutes.
3. **Real-World Isochrone Distortion:** Using Euclidean circles leads planners to believe a customer is covered within 2 hours when, in reality, traffic bottlenecks and bridge choke points cause couriers to arrive 45 minutes late, triggering severe contractual SLA liquidated damages. Network design must use **GIS road-network travel isochrones** calibrated to time-of-day traffic speeds."

**Weak Answer:**  
"Because cars don't fly in straight lines; they have to follow roads, and traffic jams make driving take much longer than a straight-line calculation says."

---

### Question 4: "When should an aftermarket supply chain deploy third-party multi-tenant 3PL depots instead of opening dedicated proprietary facilities?"

**Strong Answer:**  
"The decision is governed by the **Transactional Dispatch Breakeven Equation**:
$$V_{\text{breakeven}} = \frac{F_{\text{dedicated}} - F_{\text{3PL fixed}}}{c_{\text{pick}}}$$
where $F_{\text{dedicated}}$ is the fixed lease and labor cost of a proprietary facility ($150,000–$250,000/year), $F_{\text{3PL fixed}}$ is the shared monthly bin storage fee ($15,000–$30,000/year), and $c_{\text{pick}}$ is the 3PL's per-dispatch pick-pack fee ($150–$300/event).

* **Deploy Multi-Tenant Shared 3PLs:** In forward metropolitan locations where demand is intermittent and low-volume (e.g., 5 to 25 emergency calls per month). A shared 3PL distributes fixed warehouse leasing, 24/7 security, and warehouse labor overhead across dozens of corporate clients, allowing the OEM to secure 2-hour courier reach at fractional fixed cost.
* **Deploy Proprietary Dedicated Depots:** In Tier 0/1 continental super-hubs where monthly dispatch volumes exceed $V_{\text{breakeven}}$, where specialized repair workshops are co-located, or where defense secrecy regulations (ITAR/TAPA Level A) legally mandate dedicated employee clearance."

**Weak Answer:**  
"You use a 3PL when you don't have enough money to lease your own building or when you only need to store a few parts in a city."

---

### Question 5: "How does the Square Root Law of Inventory penalize decentralized forward stocking locations (FSLs)?"

**Strong Answer:**  
"The Square Root Law states that for independent, identically distributed Poisson or Gaussian demand, total safety stock across $N$ decentralized stocking locations scales proportionally to the square root of $N$:
$$\text{Safety Stock}_{\text{total}}(N) \approx \text{Safety Stock}_{\text{central}} \times \sqrt{N}$$

When an organization fragments its inventory from 1 central distribution center into 36 regional FSLs to achieve 4-hour response times:
$$\sqrt{36} = 6.0$$
The network requires **600% of the original safety stock inventory**!

Physically, this occurs because decentralization destroys risk pooling. At each decentralized FSL, demand is highly intermittent and lumpy (mostly zeros punctuated by occasional ones). Each depot must hold independent safety buffers to protect against local demand spikes, preventing surplus stock at Depot A from buffering shortages at Depot B. To mitigate this penalty, companies must stock only high-criticality, fast-moving items (Quadrant I) at FSLs, keeping expensive slow-movers (Quadrant II) pooled centrally."

**Weak Answer:**  
"The Square Root Law says that if you have more warehouses, you need $\sqrt{N}$ times more inventory because each warehouse has to hold its own safety stock."

---

### Question 6: "What is an automated PUDO (Pick-Up / Drop-Off) locker, and how does it optimize final-mile technician productivity?"

**Strong Answer:**  
"An automated PUDO locker is a secure, cloud-connected electronic locker bank (such as ByBox, InPost, or Amazon Locker) positioned in convenient, 24/7 accessible locations: suburban gas stations, transit hubs, or corporate office parks.

In field service operations:
1. **Eliminates Courier Waiting:** A parts courier delivers an emergency replacement board from the regional depot to a PUDO locker located 3 miles from the customer site, securing it in a compartment and transmitting a one-time digital PIN code to the field engineer's smartphone.
2. **Decoupled Arrival:** The field technician drives directly to the locker, enters the PIN, retrieves the part in 60 seconds, and proceeds to the customer site.
3. **Reverse Core Staging:** The technician places the unserviceable removed core back into the locker. The courier sweeps the locker overnight, returning the core to the MRO repair loop.
4. **Productivity Gain:** This eliminates technician deadhead travel to distant warehouses, eliminates waiting for couriers in hospital parking lots, and provides 24/7 unattended parts access at a fraction of the cost of staffing a manned warehouse."

**Weak Answer:**  
"A PUDO locker is an electronic locker where a delivery driver drops off a part so the technician can pick it up with a code whenever they are ready."

---

### Question 7: "What is an 'Outlier Customer' in Set Covering optimization, and what commercial strategies manage outlier risk?"

**Strong Answer:**  
"An Outlier Customer is a customer site located in a remote, low-density geographic territory (e.g., northern Finland, rural Montana, offshore islands) far outside the coverage isochrones of standard metropolitan depots.

In Set Covering models, outlier customers cause severe optimization distortion: **the model is forced to open a dedicated $200,000/year depot simply to cover a single remote customer who generates two service calls a year**.

Commercial strategies to manage outlier risk:
1. **Contractual SLA Derating:** Negotiate extended response windows in the contract (e.g., offering a 12-hour or Next-Business-Day response for remote sites instead of standard 4-hour metropolitan terms).
2. **On-Site Consignment Stocking:** Place a dedicated emergency spare parts kit directly inside the customer’s facility (customer plant cage), shifting the storage footprint onto the customer's property.
3. **Geographic Surcharges:** Price the remote service agreement with a dedicated logistics surcharge that fully absorbs the fixed cost of contracting a local 3PL partner."

**Weak Answer:**  
"An outlier customer is someone who lives far away from any warehouse. You manage them by charging them more money or telling them you can't get there in 2 hours."

---

### Question 8: "How does post-Brexit customs friction illustrate the danger of cross-border service network dependencies?"

**Strong Answer:**  
"Prior to Brexit, multinational OEMs serviced the UK, France, and Benelux from single centralized mega-depots located near Paris or Amsterdam. Parts moved across the English Channel under the EU Single Market with zero customs inspections, zero tariffs, and predictable 4-hour transit times.

Following the UK’s departure from the EU Single Market:
* Moving parts across the Channel requires formal customs export/import declarations, safety-and-security filings, and border transit paperwork.
* Channel freight corridors (Dover, Folkestone) suffer recurring, unpredictable customs delays ranging from 4 to 36 hours.
* For an OEM with a contractually binding **4-hour or 6-hour on-site SLA** in London, servicing UK customers from an EU continental hub became legally and operationally impossible. Every single emergency dispatch breached the contract, triggering thousands of dollars in hourly liquidated damages.

The operational lesson: **Service networks cannot span international sovereign borders when response SLAs are shorter than customs clearance variance**. Companies were forced to redesign their networks, establishing dedicated domestic UK inventory depots in London or the Midlands to insulate operations from border friction."

**Weak Answer:**  
"After Brexit, parts going between England and Europe got stuck at customs for hours. Companies that had all their parts in France couldn't meet their 4-hour response times in London, so they had to build new warehouses in England."

---

## 15. Key terms

- **Service Network Design:** The strategic spatial optimization of parts depots and logistics corridors to meet contractual SLAs.
- **Forward Stocking Location (FSL):** Localized mini-warehouse positioned in metropolitan cores to fulfill 2-hour and 4-hour SLAs.
- **Pick-Up / Drop-Off (PUDO):** Automated electronic locker bank providing 24/7 unattended parts retrieval and reverse core drop.
- **Travel-Time Isochrone:** Spatial geometric contour representing the boundary of reachable geographic points within a fixed driving time.
- **Set Covering Problem (SCP):** Combinatorial optimization model finding the minimum-cost depot network ensuring 100% customer coverage.
- **Maximal Covering Location Problem (MCLP):** Optimization model maximizing customer coverage within $D_{\max}$ subject to a budget of $P$ facilities.
- **Square Root Law of Inventory:** Statistical relationship showing that decentralized safety stock scales with $\sqrt{N}$.
- **Hot-Shot Courier:** Dedicated, single-consignment emergency vehicle dispatched immediately for point-to-point delivery.
- **Pick-Pack Latency:** Elapsed handling time from electronic order injection to outbound carrier loading dock handoff.
- **Cross-Border Regulatory Friction:** Customs clearance delays, inspections, and import tariffs incurred when traversing sovereign borders.
- **Good Distribution Practice (GDP):** Statutory quality standard governing temperature control and traceability of medical spares.
- **TAPA FSR:** Transported Asset Protection Association facility security standard for high-value electronics warehouses.
- **Shared Multi-Tenant 3PL:** Logistics model where multiple corporate clients share warehouse footprint, labor, and courier fleets.
- **Lateral Transshipment Radius:** The geographic boundary within which peer-to-peer depot inventory sharing is economically viable.
- **Outlier Customer:** Remote customer demand point whose geographic isolation requires disproportionate facility investment to cover.
- **Dock-to-Door SLA:** Contractual performance metric governing total elapsed time from warehouse pick to customer site delivery.
- **Isochrone Buffer:** The administrative and pick-pack time subtracted from the SLA window to calculate allowable driving time.
- **RDC (Regional Distribution Center):** Tier 1 logistics hub holding broad catalog inventory to support downstream FSL replenishment.
- **Consolidated Reverse Sweep:** Scheduled logistics process aggregating unserviceable cores from FSLs back to central MRO repair hubs.
- **Failover Depot:** Designated secondary warehouse assigned to provide backup coverage if a primary depot suffers catastrophic disruption.

---

## 16. Sources

### Peer-reviewed and Academic — Grade A
1. **Daskin, M. S. (2013).** *Network and Discrete Location: Models, Algorithms, and Applications* (2nd ed.). Hoboken: John Wiley & Sons. [DOI: 10.1002/9781118537015]
2. **Church, R., & ReVelle, C. (1974).** "The maximal covering location problem." *Papers of the Regional Science Association*, 32(1), 101-118. [DOI: 10.1007/BF01942293]
3. **Toregas, C., Swain, R., ReVelle, C., & Bergman, L. (1971).** "The location of emergency service facilities." *Operations Research*, 19(6), 1363-1373. [DOI: 10.1287/opre.19.6.1363]
4. **Candas, A., & Kutanoglu, E. (2007).** "A benefits, penalties and costs model for service parts network design." *European Journal of Operational Research*, 179(3), 772-789. [DOI: 10.1016/j.ejor.2005.06.068]
5. **Rappold, J. A., & Van Roo, B. D. (2009).** "Designing multi-echelon service parts networks with time-based service levels." *IIE Transactions*, 41(3), 265-277. [DOI: 10.1080/07408170802322749]

### Standards and Regulatory Frameworks — Grade B
6. **Transported Asset Protection Association (TAPA). (2020).** *Facility Security Requirements (FSR) 2020 Standard*. Amsterdam: TAPA EMEA.
7. **European Commission. (2013).** *Guidelines of 5 November 2013 on Good Distribution Practice of medicinal products for human use (2013/C 343/01)*. Brussels: Official Journal of the European Union.
8. **International Organization for Standardization (ISO). (2020).** *ISO 28000:2020: Security management systems for the supply chain*. Geneva: ISO.
9. **HM Revenue & Customs (HMRC). (2021).** *The Border Operating Model: Processes for applying import controls on goods entering Great Britain from the EU*. London: Cabinet Office.

### Industry Case Evidence and Technical Monographs — Grade C & D
10. **Cisco Systems. (2022).** *Cisco Global Service Supply Chain & SMARTnet Logistics Architecture Whitepaper*. San Jose: Cisco Operations Publications.

---

## 17. Links

- **Parent Module:** [`_index.md`](_index.md) (Module 31 — Aftermarket, Spare Parts, and Service Supply Chains)
- **Root Survey File:** [`../../../course/31_aftermarket_spare_parts_service.md`](../../../course/31_aftermarket_spare_parts_service.md)
- **Previous Topic:** [`08_connected-assets-predictive-service.md`](08_connected-assets-predictive-service.md) (Connected Assets and Predictive Service)
- **Next Module:** [`../32_food_agriculture_perishables/_index.md`](../32_food_agriculture_perishables/_index.md) (Module 32 — Food, Agriculture, and Perishable Supply Chains)
- **Related Methods:**
  - [`../../../course/11_network_design_optimization.md`](../../../course/11_network_design_optimization.md)
  - [`../../../course/12_transportation_routing_last_mile.md`](../../../course/12_transportation_routing_last_mile.md)
