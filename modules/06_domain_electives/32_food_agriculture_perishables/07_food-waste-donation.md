# Food Waste Quantification, Donation Economics, and Recovery Network Routing

> Module 32 — Food, Agriculture, and Perishable Supply Chains
> Topic 07 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Module 10 (Inventory Optimization), Module 21 (Supply Chain Network Design & Optimization), Module 32 Topic 03 (Shelf Life, FEFO, and Age-Based Inventory), Topic 04 (Newsvendor with Salvage and Freshness Pricing)
> Version 2026.5

---

## 1. Why this topic exists

In conventional industrial supply chains, scrap and unsellable inventory are treated as predictable production shrinkage. When a machining process produces metal shavings or an injection molding line ejects flashing, the material is inert, non-hazardous, and non-time-critical. Operations managers consolidate the scrap in roll-off hoppers, sell it to secondary scrap processors at prevailing commodity indices, or write off the loss against standard cost variances. The physical timing of scrap disposal rarely dictates its residual value, and disposal carries minimal legal liability or public scrutiny.

In perishable food and agricultural supply chains, product that cannot be sold through primary commercial retail channels represents an acute operational, economic, and moral crisis. 

Globally, an estimated 1.3 billion metric tons of edible food—representing approximately one-third of all food produced for human consumption—is lost or wasted annually across agricultural production, post-harvest handling, processing, wholesale distribution, retail, and food service. This unconsumed biomass carries a direct economic loss exceeding \$1.0 trillion annually, consumes roughly 25% of all agricultural freshwater withdrawal, utilizes 30% of global agricultural land area, and generates 8% to 10% of total anthropogenic greenhouse gas (GHG) emissions. When discarded organic food decomposes in anaerobic municipal solid waste (MSW) landfills, it generates methane ($\text{CH}_4$), a potent short-lived climate pollutant with a 100-year global warming potential 28 times greater than carbon dioxide ($\text{GWP}_{100} = 28$) and a 20-year global warming potential 84 times greater ($\text{GWP}_{20} = 84$).

Within enterprise supply chains, food loss and waste (FLW) is not an inevitable byproduct of biological variance; it is the structural consequence of operational friction and misaligned commercial incentives:
1. **Perishability Cliffs and Expiration Policies:** Retailers enforce strict minimum remaining shelf life (RSL) acceptance thresholds (often requiring 50% to 75% of total manufacturer shelf life upon delivery at distribution centers). A pallet of yogurt or packaged salad that misses a distribution center (DC) cross-dock window by 24 hours becomes commercially unmerchantable to primary grocery banners, even though it retains 7 to 10 days of pristine microbiological safety and organoleptic quality.
2. **Asymmetric Markdown and Transaction Friction:** In retail store operations, traditional markdown cadences often fail to clear dynamic surpluses before terminal expiration dates. Dynamic price optimization requires shelf-edge electronic labeling, inventory visibility at the lot level, and local customer demand responsiveness. When markdowns lag or fail to clear stock, inventory converts instantly into unsellable waste.
3. **Severe Reverse Logistics Latency:** Unlike shelf-stable consumer packaged goods (CPG), which can be returned to centralized reverse logistics centers over 30 to 60-day cycles, perishable surplus rots. Every hour spent staging, evaluating, and transporting perishable surplus degrades its remaining shelf life. If reverse logistics cannot clear the surplus within 24 to 72 hours under continuous cold chain integrity, the material transitions from edible human food to regulated organic waste.
4. **Distorted Landfill Economics:** Historically, municipal landfill tipping fees have masked the true cost of food waste. An operations manager viewing disposal through a localized budget ledger sees only a hauling and tipping fee of \$50 to \$100 per metric ton. However, the true embedded cost of wasted food includes the unrecovered cost of goods sold (raw material, labor, packaging, energy, and transportation), which typically ranges from \$1,500 to \$5,000 per metric ton—roughly 30 to 50 times the visible tipping fee.

To address these inefficiencies, enterprises are forced by regulation, investor ESG governance, and margin pressure to transition from passive landfill dumping to structured food recovery and diversion networks. Under statutory frameworks such as the US Food Donation Improvement Act of 2022, Section 170(e)(3) of the Internal Revenue Code, and European circular economy mandates (such as France's *Loi Garot*), commercial food surplus must be measured, classified, and directed toward the highest possible tier of the Food Recovery Hierarchy.

Achieving this transition requires rigorous analytical systems. Operations researchers and supply chain data scientists must replace manual waste logs with standardized mass-balance accounting (the global FLW Protocol), evaluate the multidimensional Net Present Value (NPV) of diversion pathways (comparing enhanced tax deductions against landfill tipping fees and anaerobic digestion yields), and solve multi-echelon vehicle routing problems with time windows and perishability constraints (CVRP-TW-P) to transport time-critical donations from commercial docks to community hunger relief agencies before shelf life expires.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Quantify** food loss and waste across enterprise supply chain operations using the global Food Loss and Waste Accounting and Reporting Standard (FLW Protocol), establishing consistent scopes, boundaries, and mass-balance accounting.
2. **Formulate** the comparative marginal net cash flow and Net Present Value (NPV) across competing end-of-life diversion pathways: municipal landfill disposal, industrial composting, anaerobic digestion, animal feed rendering, and charitable food donation.
3. **Calculate** corporate cash tax savings under US Internal Revenue Code (IRC) Section 170(e)(3) enhanced tax deduction rules, identifying the binding constraints between cost basis, fair market value mark-up, and corporate taxable income ceilings.
4. **Formulate** a multi-echelon Capacitated Vehicle Routing Problem with Time Windows and Perishability (CVRP-TW-P) to schedule donor pickups and agency deliveries under strict remaining shelf life (RSL) and temperature constraints.
5. **Implement** an automated food rescue matching and routing algorithm that balances vehicle travel costs, volunteer operating hours, refrigerated container limits, and nutritional delivery urgency.
6. **Evaluate** legal liability protections under the Bill Emerson Good Samaritan Food Donation Act and the Food Donation Improvement Act of 2022 to construct defensible, audited food donation standard operating procedures (SOPs).
7. **Design** reverse cold-chain logistics architectures that minimize thermal excursions and inventory degradation during the collection, cross-docking, and last-mile distribution of donated perishables.

---

## 3. The core mental model

To model food waste and charitable recovery analytically, you must conceptualize perishable inventory not as a static binary state (sellable vs trash), but as a continuous state-transition vector moving through a multi-tier **Cascading Value Recovery Funnel** constrained by time-dependent biological decay.

```text
+----------------------------------------------------------------------------------------------------+
| 1. COMMERCIAL SUPPLY CHAIN (Primary Forward Logistics)                                             |
|                                                                                                    |
|   [ Harvest / Inbound ] ──> [ Regional DC ] ──> [ Retail Store / Foodservice ] ──> [ Primary Sale]  |
|          │                         │                              │                                |
|          │ Yield Variance          │ Minimum Shelf-Life Cut       │ Code Dating / Overstock        |
|          ▼                         ▼                              ▼                                |
+----------------------------------------------------------------------------------------------------+
| 2. SURPLUS TRIAGE & DECISION ENGINE (Dynamic Reverse Logistics)                                    |
|                                                                                                    |
|                     ┌──────────────────────────────────────────┐                                   |
|                     │   Surplus Lot: Mass M, Cost Basis C,     │                                   |
|                     │   Fair Market Value FMV, Remaining RSL   │                                   |
|                     └─────────────────────┬────────────────────┘                                   |
|                                           │                                                        |
|         ┌─────────────────────────────────┼─────────────────────────────────┐                      |
|         ▼                                 ▼                                 ▼                      |
|  [ Commercial Salvage ]         [ Human Food Donation ]            [ Non-Human Diversion ]         |
|  - Dynamic markdowns            - Regional Food Bank (Hub)         - Animal Feed / Rendering       |
|  - Secondary discount marts     - Direct Agency Rescue (Spoke)     - Anaerobic Digestion (Biogas)  |
|  - Upcycled food processing     - Direct Client Delivery (FDIA)    - Industrial Composting         |
|         │                                 │                                 │                      |
|         │ Margin Recovery                 │ IRC § 170(e)(3) Tax Credit     │ Tip Fee Avoidance     |
|         ▼                                 ▼                                 ▼                      |
+----------------------------------------------------------------------------------------------------+
| 3. MULTI-ECHELON FOOD RESCUE ROUTING & CAPACITY MATCHING                                           |
|                                                                                                    |
|   Commercial Donors               Rescue Fleet (CVRP-TW-P)            Hunger Relief Agencies       |
|   (Supermarkets, DCs)             (Refrigerated / Ambient)            (Pantries, Soup Kitchens)    |
|   [ Donor A ] (RSL=72h) ──┐         ┌────────────────────┐         ┌──> [ Agency 1 ] (Cap=500kg)   |
|                           ├────────>│ Routing Optimizer  ├─────────┤                               |
|   [ Donor B ] (RSL=24h) ──┘         │ - Time Windows [e,l│         └──> [ Agency 2 ] (Cap=1200kg)  |
|                                     │ - Cold Chain Temp  │                                         |
|                                     │ - Min RSL Delivery │                                         |
|                                     └────────────────────┘                                         |
+----------------------------------------------------------------------------------------------------+
| 4. MASS-BALANCE & CARBON EMISSIONS ACCOUNTING (FLW Protocol / GHG Protocol)                        |
|                                                                                                    |
|   Landfill Diversion: ΔCO2e = M * (EF_landfill - EF_diversion)                                     |
|   Financial Cash Flow: CF_net = Tax Benefit + Revenue - Logistics Cost - Tip Fees                   |
+----------------------------------------------------------------------------------------------------+
```

### 3.1 The Three Governing Dynamics

1. **The Shelf-Life Velocity Trade-off:** The economic feasibility of food donation is governed by a race against microbiological and organoleptic decay. Let $RSL(t)$ be the remaining shelf life of a surplus lot at time $t$. Commercial retail channels typically reject inventory when $RSL(t) < \theta_{retail}$ (e.g., $\theta_{retail} = 7\text{ days}$). Charitable food recovery networks can safely absorb product where $\theta_{agency} \le RSL(t) < \theta_{retail}$ (e.g., $\theta_{agency} = 24\text{ to }48\text{ hours}$). If reverse logistics routing latency $\Delta t_{transit} + \Delta t_{handling}$ exceeds $RSL(t) - \theta_{agency}$, the product crosses the safety threshold and drops down the hierarchy into anaerobic digestion, composting, or landfill.

2. **The Tax Asymmetry Inversion:** Traditional corporate inventory accounting writes off discarded inventory at its cost basis $C$, providing a cash tax shield of $\tau \cdot C$, where $\tau$ is the marginal corporate income tax rate. Under US IRC Section 170(e)(3), qualifying food donations earn an *enhanced deduction* equal to basis plus up to half the commercial mark-up ($C + 0.5(FMV - C)$), capped at twice the basis ($2C$). For high-margin perishable items (such as fresh bakery, packaged salads, and specialty dairy), the enhanced tax cash savings can exceed the total cost of staging, refrigerated transport, and donation handling. When evaluated on a pure cash-flow basis, **donating food is frequently more profitable than throwing it away**.

3. **The Network Asymmetry Bottleneck:** Commercial food supply chains are built on pallet-scale, predictable, push-based scheduling with automated distribution centers and full truckload (FTL) linehauls. In contrast, the charitable hunger relief network is highly fragmented, pull-based, and capacity-constrained. Regional food banks act as consolidation hubs, but the last-mile distribution network consists of thousands of small food pantries, shelters, and faith-based community organizations. These agencies possess limited walk-in refrigeration capacity, narrow operating time windows (often staffed by part-time volunteers for 2 to 3 hours a day), and strict dietary demand constraints. An effective recovery model must dynamically balance large, lumpy commercial surplus bursts against hyper-local, capacity-constrained community sinks.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse | Unit or Coding Case |
|---|---|---|---|
| **Food Loss and Waste (FLW)** | The decrease in quantity or quality of food along the food supply chain from harvest up to, but not including, the retail and consumer level (Food Loss) and at retail, food service, and household levels (Food Waste). | Using "loss" and "waste" interchangeably; loss occurs upstream (farm to wholesale), while waste occurs downstream (retail to consumer). | Metric tons ($\text{t}$), kilograms ($\text{kg}$), or megajoules ($\text{MJ}$). |
| **FLW Protocol** | The global *Food Loss and Waste Accounting and Reporting Standard*, developed by the World Resources Institute (WRI), WBCSD, UNEP, and FAO to provide standardized mass-balance accounting for food waste. | Assuming the FLW Protocol only tracks landfill disposal; it tracks material flows across 10 distinct destination categories. | Accounting standard framework. |
| **EPA Wasted Food Scale** | The statutory waste management hierarchy published by the US EPA (updated in 2023), ranking strategies from most to least preferred: Prevention, Donation/Upcycling, Animal Feed, Anaerobic Digestion/Composting, Landfill/Incineration. | Assuming all organic recycling is equal; composting is placed below human donation and animal feed. | Ordinal categorical scale. |
| **Landfill Tipping Fee** | The unit gate fee charged by a disposal facility (landfill or transfer station) to receive and process solid municipal or commercial waste. | Conflating the tipping fee with the total cost of waste; tipping fee represents only 2% to 4% of total embedded food inventory cost. | Currency per unit mass: $\$ / \text{tonne}$ or $\$ / \text{short ton}$. |
| **Cost Basis ($C$)** | The capitalized inventoriable cost of a food lot under GAAP/tax rules, including direct materials, direct labor, and allocated inbound freight and storage. | Confusing cost basis with wholesale purchase price; basis includes allocated inventory handling capitalization. | Currency ($\$$ or USD per unit/lot). |
| **Fair Market Value ($FMV$)** | The price at which the food item would have been sold in its customary commercial market segment on the date of donation, accounting for condition and code date. | Using original retail list price for damaged or near-expiration goods without verifying prevailing secondary sales prices. | Currency ($\$$ or USD per unit/lot). |
| **IRC Section 170(e)(3)** | The provision of the US Internal Revenue Code providing an enhanced corporate tax deduction for inventory donations dedicated to the care of the ill, the needy, or infants. | Believing the enhanced deduction allows writing off full retail value; the deduction is strictly capped at basis plus half the mark-up, not to exceed $2C$. | US Federal Statutory Tax Code. |
| **Bill Emerson Good Samaritan Act** | Federal statute (42 U.S.C. § 1791) providing civil and criminal liability protection to food donors and non-profit recipients for donations made in good faith, absent gross negligence. | Believing the Act protects donors who knowingly donate adulterated or pathogen-contaminated food; gross negligence voids immunity. | US Federal Public Law. |
| **Food Donation Improvement Act (FDIA)** | US federal legislation enacted in December 2022 amending the Emerson Act to extend liability protection to direct-to-consumer donations and donations sold at deeply discounted prices. | Believing direct donations were protected under the original 1996 Emerson Act; direct donation protection was only codified in 2022. | Pub. L. 117-328, Division JJ. |
| **Loi Garot** | French statutory law (Law No. 2016-138) prohibiting grocery stores $\ge 400\text{ m}^2$ from deliberately destroying or discarding edible unsold food, mandating signed donation contracts with food banks. | Assuming compliance is voluntary; non-compliant French supermarket operators face criminal fines of €3,750 per infraction. | French Statutory Regulation. |
| **Food Bank** | A centralized, regional logistical warehouse facility that collects, sorts, aggregates, and inventories bulk surplus food for redistribution to front-line agencies. | Conflating a food bank with a food pantry; food banks operate as wholesale tier-1 DCs, rarely distributing directly to individuals. | Logistical tier node (Hub). |
| **Food Pantry** | A front-line charitable agency (tier-2 spoke) that receives food from food banks or local grocery donors and distributes groceries directly to food-insecure households. | Assuming pantries have industrial cross-docking infrastructure; most pantries rely on residential-style fridges and volunteer labor. | Logistical tier node (Spoke). |
| **CVRP-TW-P** | Capacitated Vehicle Routing Problem with Time Windows and Perishability; an NP-hard optimization problem routing vehicles subject to weight, volume, time, and shelf-life decay limits. | Omitting the perishability state variable, which causes the model to generate routes that deliver expired food to agencies. | Mathematical optimization class. |
| **Remaining Shelf Life ($RSL(t)$)** | The elapsed time or degree-days remaining before a perishable food item exceeds microbiological safety thresholds or commercial sensory acceptance standards. | Treating shelf life as fixed calendar days regardless of temperature history; $RSL(t)$ is a dynamic integral over temperature. | Time ($\text{hours}$, $\text{days}$) or thermal units. |
| **Anaerobic Digestion (AD)** | A biological process where microorganisms break down organic matter in the absence of oxygen, yielding methane-rich biogas and nutrient-rich digestate. | Conflating AD with aerobic composting; AD captures volatile biogas for renewable energy, whereas composting releases $\text{CO}_2$ aerobically. | Industrial biochemical conversion. |
| **Food Recovery Efficiency (FRE)** | The ratio of the mass of edible surplus food successfully delivered to human consumption over the total mass of edible surplus generated. | Measuring gross weight collected without subtracting destination shrink, spoilage, or unsorted packaging waste. | Dimensionless ratio: $\text{FRE} \in [0.0, 1.0]$. |
| **Virtual Food Banking (Direct Matching)** | A digital logistics dispatch model where algorithmic platforms match retail surplus directly with nearby pantries, bypassing central food bank consolidation warehouses. | Assuming direct dispatch replaces central food banks entirely; direct matching works for retail LTL, while hubs remain necessary for FTL farm/DC bulk. | Algorithmic logistics architecture. |
| **Scope 3 Category 5 Emissions** | Greenhouse gas emissions associated with waste generated in an enterprise's operational facilities, accounted for under the GHG Protocol Corporate Value Chain Standard. | Omitting fugitive landfill methane emissions from supply chain sustainability scorecards. | Mass of $\text{CO}_2$ equivalent ($\text{tCO}_2\text{e}$). |

---

## 5. The mechanics, step by step

Implementing an enterprise food waste quantification, diversion optimization, and recovery routing architecture requires an integrated four-stage operational pipeline:

```text
+------------------------+      +------------------------+      +------------------------+      +------------------------+
| 5.1 Mass-Balance FLW   | ---> | 5.2 Marginal Cash Flow | ---> | 5.3 Vehicle Routing &  | ---> | 5.4 Cold Chain Quality |
| Accounting Protocol    |      | & Tax Optimization     |      | Matching (CVRP-TW-P)   |      | & Regulatory Compliance|
+------------------------+      +------------------------+      +------------------------+      +------------------------+
```

### 5.1 Mass-Balance FLW Accounting Protocol

The foundation of food waste engineering is standardized measurement. Under the WRI/UNEP Food Loss and Waste (FLW) Protocol, an enterprise must establish an audited mass-balance accounting framework.

```text
[ Total Inbound Biomass: M_in ]
              │
              ├──> [ Finished Sold Product: M_sold ] (Target Primary Stream)
              │
              └──> [ Food Loss & Waste Stream: M_FLW ]
                             │
                             ├──> Destination d=1: Redistribution to Humans (Donation) [M_don]
                             ├──> Destination d=2: Animal Feed / Rendering [M_feed]
                             ├──> Destination d=3: Industrial Uses / Anaerobic Digestion [M_ad]
                             ├──> Destination d=4: Aerobic Composting [M_comp]
                             └──> Destination d=5: Municipal Landfill / Incineration [M_land]
```

#### Step 1: Boundary and Scope Specification
Every FLW accounting ledger must explicitly specify four boundaries:
1. **Timeframe:** Accounting period (e.g., calendar year, fiscal quarter).
2. **Material Type:** Distinguish between *edible food parts* (e.g., fruit flesh, bread crusts, yogurt) and *associated inedible parts* (e.g., bones, eggshells, citrus rinds, banana peels).
3. **Lifecycle Stage:** Specific facility nodes in the supply chain (e.g., DC cross-dock, retail backroom, food processing facility).
4. **Destinations:** The final destinations of unconsumed material.

#### Step 2: Mass Quantification Methods
Depending on data density and facility infrastructure, mass flows $M_{FLW}$ are quantified using one of three approved methods:
- **Direct Measurement (Weighing):** Electronic floor scales and lift-truck scale forks capture the gross weight $W_{gross}$ and tare weight $W_{tare}$ of unmerchantable pallets or roll-off dumpsters:
  $$M_{FLW} = W_{gross} - W_{tare}$$
- **Waste Composition Analysis (WCA):** For mixed-stream waste containers, representative random sampling is conducted to calculate the physical percentage composition of organic fractions:
  $$M_i = M_{total} \times \phi_i, \quad \sum_{i=1}^K \phi_i = 1.0$$
  where $\phi_i$ is the mass fraction of commodity category $i$.
- **Inventory Discrepancy Mass Balance:** In enterprise resource planning (ERP) systems, the unmeasured waste mass is estimated via inventory balancing:
  $$M_{FLW} = I_{initial} + M_{inbound} - M_{sold} - M_{reworked} - I_{ending}$$
  where all terms are converted to kilograms ($\text{kg}$) via standardized SKU bill-of-materials unit weights.

#### Step 3: Greenhouse Gas (GHG) Emissions Accounting
To quantify Scope 3 Category 5 emissions, the mass diverted from landfill is mapped to net GHG emissions avoidance using IPCC and US EPA Waste Reduction Model (WARM) emissions factors:
$$\Delta E_{GHG} = \sum_{d=1}^D M_d \cdot (EF_{landfill} - EF_d)$$
where:
- $EF_{landfill}$ is the lifecycle emissions factor for landfilling organic food waste, accounting for methane generation, landfill gas capture efficiency ($\eta_{LFG} \approx 0.75$), and fugitive emissions (typically $+0.85\text{ to }+1.25\text{ tCO}_2\text{e} / \text{t food}$).
- $EF_{donation}$ accounts for reverse logistics transportation emissions (typically $+0.02\text{ to }+0.05\text{ tCO}_2\text{e} / \text{t food}$), yielding a net emissions avoidance of approximately $1.0\text{ to }1.2\text{ tCO}_2\text{e}$ per ton of donated food.
- $EF_{AD}$ accounts for fugitive emissions minus the displaced fossil grid energy from renewable biogas (typically $-0.10\text{ to }-0.25\text{ tCO}_2\text{e} / \text{t food}$).
- $EF_{compost}$ accounts for aerobic decomposition emissions minus soil carbon sequestration benefits (typically $+0.15\text{ to }+0.25\text{ tCO}_2\text{e} / \text{t food}$).

---

### 5.2 Marginal Cash Flow and Tax Optimization Mechanics

When inventory becomes commercially unviable for primary retail, operations managers must determine the optimal destination $d \in \{\text{Landfill}, \text{Compost}, \text{AD}, \text{Rendering}, \text{Donation}\}$ that maximizes the marginal net cash flow $\Pi_d$.

```text
                                  [ Surplus Lot Identified ]
                                               │
                                 ┌─────────────┴─────────────┐
                                 ▼                           ▼
                         [ Landfill Disposal ]       [ Food Donation ]
                                 │                           │
          Direct Cost:           │                           │ Direct Cost:
          - Hauling Fee ($)      │                           │ - Staging / Sorting ($)
          - Tipping Fee ($/t)    │                           │ - Refrigerated Transit ($)
                                 │                           │
          Tax Benefit:           │                           │ Tax Benefit (IRC § 170(e)(3)):
          - Basis Write-Off:     │                           │ - Enhanced Deduction:
            τ * Basis            │                           │   D = min(2*C, C + 0.5*(FMV - C))
                                 │                           │   Cash Savings: τ * D
                                 ▼                           ▼
                     [ Net Cash Flow: CF_land ]   [ Net Cash Flow: CF_don ]
                                 │                           │
                                 └─────────────┬─────────────┘
                                               │
                                               ▼
                              [ Net Financial Advantage: ]
                              ΔCF = CF_don - CF_land > 0 ?
```

#### The General Cash Flow Formulation
For any disposal or recovery pathway $d$, the net marginal cash flow $CF_d$ is formulated as:
$$CF_d = R_d + \Delta T_d - C^{trans}_d - C^{hand}_d - C^{tip}_d$$
where:
- $R_d$ is direct revenue generated from secondary salvage, byproduct rendering, or tipping incentives ($\$$). For donation and landfill, $R_d = 0$.
- $\Delta T_d$ is the cash tax savings generated by inventory write-offs or charitable tax deductions ($\$$).
- $C^{trans}_d$ is the freight and logistics cost to transport the lot to destination $d$ ($\$$).
- $C^{hand}_d$ is the internal warehouse labor and materials cost to sort, pack, re-palletize, and stage the lot ($\$$).
- $C^{tip}_d$ is the tipping or gate fee paid to the receiving facility ($\$$). (Note: if $d$ is landfill, $C^{tip}_{landfill} = M \cdot r_{tip}$; if $d$ is donation, $C^{tip}_{don} = 0$).

#### Tax Shield Mechanics: Standard Write-Off vs Enhanced Deduction
The mathematical driver of food donation economics is the tax shield differential.

**Case A: Physical Disposal (Landfill, AD, Composting, or Rendering)**
When inventory is discarded, destroyed, or rendered, the enterprise claims a standard tax deduction for worthless inventory under IRC Section 165 or through ordinary Cost of Goods Sold (COGS) inventory write-down. The tax deduction is strictly equal to the capitalized cost basis $C$:
$$D_{disposal} = C$$
The resulting cash tax shield is:
$$\Delta T_{disposal} = \tau \cdot C$$
where $\tau$ is the combined federal and state marginal corporate income tax rate (typically $\tau \in [0.21, 0.28]$ in the US).

**Case B: Charitable Food Donation under IRC Section 170(e)(3)**
Under US Internal Revenue Code Section 170(e)(3) and Treasury Regulation § 1.170A-4A, a qualifying corporate donor that donates inventory to a 501(c)(3) public charity dedicated to the care of the ill, the needy, or infants is entitled to an **enhanced tax deduction** $D_{enhanced}$.

The statutory formula dictates that the donor may deduct the cost basis $C$ plus one-half of the appreciated mark-up between cost basis and commercial fair market value ($FMV$), subject to a hard ceiling of twice the cost basis ($2C$):
$$D_{enhanced} = \min \left\{ 2 \cdot C, \; C + \frac{1}{2} (FMV - C) \right\}$$
Algebraically, this can be rewritten as:
$$D_{enhanced} = \min \left\{ 2 \cdot C, \; \frac{1}{2} (C + FMV) \right\}$$

The cash tax savings resulting from the donation deduction is:
$$\Delta T_{donation} = \tau \cdot D_{enhanced}$$

The incremental tax deduction achieved by donating rather than landfilling is:
$$\Delta D = D_{enhanced} - C = \min \left\{ C, \; \frac{1}{2}(FMV - C) \right\}$$

The incremental cash tax savings $\Delta(\Delta T)$ is:
$$\Delta(\Delta T) = \tau \cdot \Delta D = \tau \cdot \min \left\{ C, \; \frac{1}{2}(FMV - C) \right\}$$

#### Statutory Boundary Constraints on Enhanced Deductions
Operations and tax accounting systems must continuously monitor three binding statutory constraints:
1. **The Appreciated Mark-up Constraint:** If $FMV \le C$ (the commercial value has collapsed below production cost due to severe obsolescence or damage), then $FMV - C \le 0$. In this case, the enhanced mark-up is zero, and $D_{enhanced} = C$. The enhanced deduction collapses to a standard write-off.
2. **The 2x Basis Ceiling:** If an item has an exceptionally high profit margin such that $FMV > 3C$, then $C + 0.5(FMV - C) > 2C$. In this case, the deduction is truncated at exactly $2C$.
3. **Corporate Taxable Income Percentage Caps:** Under IRC Section 170(b)(2), total charitable deductions for a C-corporation cannot exceed 15% of the corporation’s taxable income for the tax year (raised permanently from 10% for food donations by the PATH Act of 2015). Any excess deduction must be carried forward as a tax attribute over the subsequent 5 tax years. In optimization models, this introduces a capacity constraint on the maximum deductible volume:
   $$\sum_{k \in \mathcal{K}_{don}} D_{enhanced, k} \le 0.15 \cdot Y_{taxable}$$

---

### 5.3 Mathematical Optimization of Food Recovery Network Routing

Once surplus lots are designated for human recovery, operations researchers face a complex network logistics problem. Commercial donors have fixed time windows for dock pickup; relief agencies have limited receiving hours, volunteer labor availability, and finite walk-in cooler capacities; and the commodities undergo dynamic shelf-life decay during transit.

This problem is formulated as a **Multi-Commodity Capacitated Vehicle Routing Problem with Time Windows and Perishability (CVRP-TW-P)**.

```text
               [ Central Food Bank Hub / Vehicle Depot ]
                              (Node 0)
                            /          \
                           /            \
          Route 1 (Chilled Truck)   Route 2 (Ambient Van)
                         /                \
                        ▼                  ▼
             [ Donor A: DC Cross-Dock ]   [ Donor C: Supermarket ]
             - Pick 12 Pallets Dairy     - Pick 2 Bins Produce
             - RSL = 72 hours             - RSL = 48 hours
             - Window: [08:00, 10:00]     - Window: [09:00, 11:00]
                        │                  │
                        ▼                  ▼
             [ Agency 1: Food Bank ]      [ Agency 2: Community Pantry ]
             - Drop 12 Pallets Dairy      - Drop 2 Bins Produce
             - Capacity: 15 Pallets       - Capacity: 3 Bins
             - Window: [10:30, 12:30]     - Window: [11:30, 13:00]
                        \                  /
                         \                /
                          ▼              ▼
                     [ Return to Central Depot ]
```

#### Graph Definition
Let $G = (V, A)$ be a directed graph where:
- Node $0$ represents the vehicle depot (e.g., regional food bank hub).
- $V_D = \{1, 2, \dots, n\}$ is the set of donor pickup nodes (supermarkets, food processing plants, distribution centers).
- $V_A = \{n+1, n+2, \dots, n+m\}$ is the set of agency delivery nodes (pantries, meal centers, shelters).
- The total node set is $V = \{0\} \cup V_D \cup V_A$.
- $A = \{(i, j) : i, j \in V, i \ne j\}$ is the set of feasible travel arcs. Each arc $(i, j)$ has a travel distance $d_{ij}$, travel time $t_{ij}$, and direct transit cost $c_{ij}$.

#### Fleet and Commodity Attributes
- A fleet of heterogeneous vehicles $K = \{1, 2, \dots, |K|\}$ is available at depot $0$. Each vehicle $k$ has a gross weight capacity $Q_k^{wt}$, a volumetric pallet capacity $Q_k^{vol}$, and a categorical refrigeration capability $\rho_k \in \{\text{Ambient}, \text{Chilled}, \text{Dual-Zone}\}$.
- A set of perishable commodity classes $M = \{1, 2, \dots, |M|\}$ (e.g., Prepared Chilled Food, Fresh Meat, Fluid Dairy, Leafy Greens, Ambient CPG). Each commodity $m$ requires a refrigeration class $\gamma^m \in \{\text{Ambient}, \text{Chilled}\}$.
- A vehicle $k$ can only carry commodity $m$ if its refrigeration capability is compatible ($\rho_k \ge \gamma^m$).

#### Decision Variables
- $x_{ijk} \in \{0, 1\}$: Binary variable equal to 1 if vehicle $k$ traverses arc $(i, j)$; 0 otherwise.
- $y_{ik}^m \ge 0$: Mass of commodity $m$ picked up (if $i \in V_D$) or delivered (if $i \in V_A$) by vehicle $k$ at node $i$.
- $u_{ik} \ge 0$: Cumulative weight aboard vehicle $k$ immediately after servicing node $i$.
- $T_{ik} \ge 0$: Arrival time of vehicle $k$ at node $i$.
- $RSL_{ik}^m \ge 0$: Remaining shelf life of commodity $m$ aboard vehicle $k$ upon departing node $i$.

#### Objective Function
The objective function balances operational routing costs against the social welfare and tax benefits of recovered meals:
$$\min \sum_{k \in K} \sum_{(i,j) \in A} c_{ij} x_{ijk} + \sum_{i \in V_A} \alpha_i \left( D_i - \sum_{k \in K} \sum_{m \in M} y_{ik}^m \right) - \sum_{i \in V_D} \sum_{k \in K} \sum_{m \in M} \beta_{im} y_{ik}^m$$
where:
- $c_{ij}$ is the fuel, maintenance, and driver wage cost to traverse arc $(i, j)$.
- $\alpha_i$ is the penalty weight per kilogram of unmet nutritional demand $D_i$ at agency $i$.
- $\beta_{im}$ is the marginal net tax cash benefit and landfill diversion value earned per kilogram of commodity $m$ recovered from donor $i$.

#### Core Constraints

1. **Routing and Flow Conservation:**
   $$\sum_{j \in V, j \ne i} x_{ijk} = \sum_{j \in V, j \ne i} x_{jik} \quad \forall i \in V_D \cup V_A, \; \forall k \in K$$
   $$\sum_{j \in V_D \cup V_A} x_{0jk} \le 1, \quad \sum_{i \in V_D \cup V_A} x_{i0k} \le 1 \quad \forall k \in K$$

2. **Vehicle Capacity (Weight and Volume):**
   $$u_{jk} \ge u_{ik} + \sum_{m \in M} y_{jk}^m - \mathcal{M}(1 - x_{ijk}) \quad \forall (i, j) \in A, \; \forall k \in K$$
   $$u_{ik} \le Q_k^{wt} \quad \forall i \in V, \; \forall k \in K$$
   where $\mathcal{M}$ is a sufficiently large positive constant.

3. **Time Windows and Service Duration:**
   $$T_{jk} \ge T_{ik} + s_i + t_{ij} - \mathcal{M}(1 - x_{ijk}) \quad \forall (i, j) \in A, \; \forall k \in K$$
   $$e_i \le T_{ik} \le l_i \quad \forall i \in V, \; \forall k \in K$$
   where $[e_i, l_i]$ is the operational time window of node $i$, and $s_i$ is the staging/loading service duration.

4. **Dynamic Shelf-Life Decay Constraints:**
   Let $k_T^m$ be the kinetic quality decay rate of commodity $m$ under vehicle compartment temperature $T_{veh}$. The remaining shelf life decays monotonically with elapsed transit and handling time:
   $$RSL_{jk}^m \le RSL_{ik}^m - (s_i + t_{ij}) \cdot \omega(T_{veh}) + \mathcal{M}(1 - x_{ijk}) \quad \forall (i, j) \in A, \; \forall k \in K, \; \forall m \in M$$
   where $\omega(T_{veh})$ is a thermal acceleration factor (derived from Arrhenius kinetics; $\omega(T) = 1.0$ at target refrigerated temperature $4^\circ\text{C}$, but $\omega(T) \ge 3.5$ if an ambient van is utilized for chilled items).

5. **Agency Shelf-Life Threshold Guarantee:**
   To guarantee that agencies do not receive spoiled food, the arriving shelf life must satisfy a statutory minimum agency buffer $RSL_{min}^m$:
   $$RSL_{ik}^m \ge RSL_{min}^m \cdot \mathbb{I}(y_{ik}^m > 0) \quad \forall i \in V_A, \; \forall k \in K, \; \forall m \in M$$

---

### 5.4 Cold Chain Quality and Regulatory Compliance Architecture

Food recovery logistics must maintain strict food safety standards. Donated food is governed by the same microbiological safety laws under the US Food Safety Modernization Act (FSMA) and FDA Food Code as commercially sold food.

```text
+----------------------------------------------------------------------------------------------------+
| FSMA & EMERSON ACT DONATION AUDIT TRAIL                                                            |
|                                                                                                    |
|  [ Donor Staging ] ──> [ Temperature Log ] ──> [ Custody Transfer ] ──> [ Agency Receiving Log ]  |
|   - Visual sensory      - Calibrated probe      - Bill of Lading        - Temp verification (<5°C) |
|   - Integrity check     - Continuous logger     - Good Samaritan cert   - FIFO/FEFO rotation log   |
|   - Primary packaging   - Reject if >5°C        - Retain 3 years        - Reject swollen packs     |
+----------------------------------------------------------------------------------------------------+
```

#### Step 1: The Good Samaritan Statutory Defense Protocol
To qualify for statutory immunity under the **Bill Emerson Good Samaritan Food Donation Act** (42 U.S.C. § 1791) and the **Food Donation Improvement Act of 2022**, an enterprise must document four elements:
1. **Good Faith Intent:** The food must be donated without commercial expectation of financial exchange from the recipient.
2. **Apparent Wholesomeness:** The food must meet all quality and labeling standards imposed by federal, state, and local laws, even if it is not readily marketable due to appearance, age, freshness, grade, size, or surplus.
3. **Qualified Recipient Organization or Direct Qualified Recipient:** Under the 2022 FDIA amendment, food may be donated:
   - To a non-profit organization for ultimate distribution to needy individuals;
   - By a qualified direct donor (retail store, restaurant, caterer, agricultural producer, or school food authority) directly to needy individuals free of charge; or
   - Distributed to needy individuals at a deeply discounted price covering only handling costs.
4. **Absence of Gross Negligence:** Donors are protected against civil liability and criminal prosecution except in cases where the donor acted with reckless disregard for safety (e.g., donating food known to be implicated in an active microbiological recall or food stored at elevated temperatures that support pathogen proliferation).

#### Step 2: Temperature Control and Staging SOPs
Perishable donations are classified into three temperature categories:
- **TCS Foods (Time/Temperature Control for Safety):** Cut leafy greens, sliced tomatoes, dairy, fresh meat, cooked poultry, and cut melons must be maintained at $\le 4.4^\circ\text{C}$ ($40^\circ\text{F}$) or frozen at $\le -18^\circ\text{C}$ ($0^\circ\text{F}$). If a TCS lot experiences an excursion where $T > 5^\circ\text{C}$ for cumulative time $\Delta t > 2\text{ hours}$, it cannot be donated for human consumption and must be diverted to anaerobic digestion or composting.
- **Non-TCS Perishables:** Whole raw uncut produce (apples, citrus, whole potatoes) require ambient ventilation ($10^\circ\text{C}\text{ to }18^\circ\text{C}$) and are exempt from mandatory cold-chain logging under Emerson Act guidelines.
- **Shelf-Stable CPG:** Ambient dry groceries can be donated past commercial "Best If Used By" dates provided primary hermetic seals remain intact and cans show no severe denting, rust, or swelling.

---

## 6. Worked example

### 6.1 Scenario Parameters: SunValley Regional Logistics DC

SunValley Logistics operates a $45,000\text{ m}^2$ grocery distribution center supplying 120 retail supermarkets. On Monday morning at 06:00, inbound inspection flags a supplier delivery of 24 pallets of premium Grade-A Greek Yogurt:
- **Product:** 500g plastic tubs of organic strawberry Greek yogurt packed in corrugated cases (12 tubs per case, 80 cases per pallet = 960 tubs / 480 kg net food per pallet).
- **Total Surplus Mass:** 24 pallets $\times$ 480 kg = $11,520\text{ kg}$ net yogurt ($12,480\text{ kg}$ gross including pallets and packaging = $12.48\text{ metric tons}$).
- **Inventory Basis ($C$):** Capitalized manufacturing and procurement cost basis = $\$1.20\text{ per tub}$.
  $$C_{total} = 24 \times 960 \times \$1.20 = 23,040 \times \$1.20 = \$27,648.00$$
- **Fair Market Value ($FMV$):** Prevailing commercial retail selling price = $\$3.40\text{ per tub}$.
  $$FMV_{total} = 23,040 \times \$3.40 = \$78,336.00$$
- **Rejection Reason:** SunValley's commercial retail specification mandates a minimum remaining shelf life of $\ge 21\text{ days}$ upon DC inbound. The delivered batch has a manufacturer printed "Use By" date exactly 12 days away ($RSL = 12\text{ days} = 288\text{ hours}$). The commercial retail banner rejects the shipment.
- **Corporate Tax Rate ($\tau$):** Federal corporate tax rate = $21\%$, state corporate tax rate = $6\%$; combined marginal rate $\tau = 27.0\% = 0.27$.

The VP of Supply Chain must decide whether to route the 24 pallets to:
1. **Municipal Landfill Disposal**
2. **Industrial Anaerobic Digestion (Biogas Energy)**
3. **Charitable Food Recovery Network (Food Bank Donation)**

---

### 6.2 Economic and Tax Calculation across Pathways

#### Pathway 1: Municipal Solid Waste Landfill Disposal
- **Hauling Freight:** Two roll-off container hauls to regional landfill = $\$450.00 \times 2 = \$900.00$.
- **Landfill Tipping Fee:** Regional municipal MSW tip fee = $\$85.00\text{ per metric ton}$.
  $$C^{tip}_{land} = 12.48\text{ t} \times \$85.00/\text{t} = \$1,060.80$$
- **Warehouse Destruction Handling:** Labor to de-palletize and crush yogurt containers into roll-off dumpster = 12 man-hours @ $\$35.00/\text{hr} = \$420.00$.
- **Total Operational Disposal Outflow:**
  $$C^{ops}_{land} = \$900.00 + \$1,060.80 + \$420.00 = \$2,380.80$$
- **Tax Shield:** Standard inventory write-off under IRC Section 165 at cost basis $C$:
  $$D_{land} = C = \$27,648.00$$
  $$\Delta T_{land} = \tau \cdot D_{land} = 0.27 \times \$27,648.00 = +\$7,464.96$$
- **Net Marginal Cash Flow:**
  $$CF_{land} = \Delta T_{land} - C^{ops}_{land} = +\$7,464.96 - \$2,380.80 = +\$5,084.16$$
- **Environmental Impact:** Lifecycle landfill methane emissions factor = $+1.10\text{ tCO}_2\text{e} / \text{t}$.
  $$E_{land} = 12.48\text{ t} \times 1.10\text{ tCO}_2\text{e}/\text{t} = +13.73\text{ tCO}_2\text{e}$$

---

#### Pathway 2: Industrial Anaerobic Digestion (Organics Energy)
- **Hauling Freight:** Dedicated bulk transport to regional commercial anaerobic digester (85 km away) = $\$750.00$.
- **Organic Tipping Fee:** Depackaging and organic digestion gate fee = $\$55.00\text{ per metric ton}$.
  $$C^{tip}_{AD} = 12.48\text{ t} \times \$55.00/\text{t} = \$686.40$$
- **Warehouse Staging Labor:** Staging intact pallets onto carrier trailer = 2.0 man-hours @ $\$35.00/\text{hr} = \$70.00$.
- **Total Operational Disposal Outflow:**
  $$C^{ops}_{AD} = \$750.00 + \$686.40 + \$70.00 = \$1,506.40$$
- **Tax Shield:** Standard inventory write-off at basis $C$:
  $$D_{AD} = C = \$27,648.00$$
  $$\Delta T_{AD} = \tau \cdot D_{AD} = 0.27 \times \$27,648.00 = +\$7,464.96$$
- **Net Marginal Cash Flow:**
  $$CF_{AD} = \Delta T_{AD} - C^{ops}_{AD} = +\$7,464.96 - \$1,506.40 = +\$5,958.56$$
- **Environmental Impact:** Net emissions factor for AD = $-0.15\text{ tCO}_2\text{e} / \text{t}$.
  $$E_{AD} = 12.48\text{ t} \times (-0.15\text{ tCO}_2\text{e}/\text{t}) = -1.87\text{ tCO}_2\text{e}$$

---

#### Pathway 3: Charitable Donation (Regional Food Bank Network)
The regional food bank confirms it can accept all 24 pallets and distribute them to pantries across three counties within 5 days, well within the 12-day remaining shelf life.

##### Step 1: Enhanced Tax Deduction Calculation under IRC § 170(e)(3)
1. **Total Cost Basis ($C$):** $\$27,648.00$
2. **Total Fair Market Value ($FMV$):** $\$78,336.00$
3. **Appreciated Mark-up ($FMV - C$):**
   $$FMV - C = \$78,336.00 - \$27,648.00 = \$50,688.00$$
4. **Half the Appreciated Mark-up:**
   $$\frac{1}{2}(FMV - C) = \frac{1}{2} \times \$50,688.00 = \$25,344.00$$
5. **Tentative Enhanced Deduction (Basis + Half Mark-up):**
   $$D_{tentative} = C + \frac{1}{2}(FMV - C) = \$27,648.00 + \$25,344.00 = \$52,992.00$$
6. **Statutory Ceiling (Twice the Basis):**
   $$D_{ceiling} = 2 \cdot C = 2 \times \$27,648.00 = \$55,296.00$$
7. **Allowable Statutory Enhanced Deduction ($D_{enhanced}$):**
   $$D_{enhanced} = \min(D_{tentative}, D_{ceiling}) = \min(\$52,992.00, \$55,296.00) = \$52,992.00$$
   *(Notice that the deduction is limited by Basis + Half Mark-up, not the 2x basis ceiling, because the commercial mark-up was less than 200%).*

##### Step 2: Cash Tax Savings
$$\Delta T_{don} = \tau \cdot D_{enhanced} = 0.27 \times \$52,992.00 = +\$14,307.84$$

##### Step 3: Operational Logistics Outflows
- **Outbound Freight:** SunValley contracts a local temperature-controlled reefer carrier to haul the 24 pallets to the regional food bank hub (45 km) = $\$550.00$.
- **Warehouse Staging and Pallet Wrapping:** Warehouse labor to re-wrap pallets and apply USDA/Good Samaritan donation tracking tags = 3.0 man-hours @ $\$35.00/\text{hr} = \$105.00$.
- **Tipping Fee:** Charitable food banks do not charge gate/tipping fees = $\$0.00$.
- **Total Operational Logistics Outflow:**
  $$C^{ops}_{don} = \$550.00 + \$105.00 + \$0.00 = \$655.00$$

##### Step 4: Net Marginal Cash Flow
$$CF_{don} = \Delta T_{don} - C^{ops}_{don} = +\$14,307.84 - \$655.00 = +\$13,652.84$$

##### Step 5: Comparative Cash Flow Summary Table

| Metric | Pathway 1: Landfill | Pathway 2: Anaerobic Digestion | Pathway 3: Food Donation | Donation Advantage vs Landfill |
|---|---:|---:|---:|---:|
| **Gross Product Mass** | $12.48\text{ t}$ | $12.48\text{ t}$ | $12.48\text{ t}$ | — |
| **Direct Handling & Prep Labor** | $-\$420.00$ | $-\$70.00$ | $-\$105.00$ | $+\$315.00$ |
| **Hauling Freight Cost** | $-\$900.00$ | $-\$750.00$ | $-\$550.00$ | $+\$350.00$ |
| **Disposal Gate / Tipping Fee** | $-\$1,060.80$ | $-\$686.40$ | $\$0.00$ | $+\$1,060.80$ |
| **Total Operational Cash Outflow** | **$-\$2,380.80$** | **$-\$1,506.40$** | **$-\$655.00$** | **$+\$1,725.80$** |
| **Allowable Tax Deduction ($D$)** | $\$27,648.00$ | $\$27,648.00$ | $\$52,992.00$ | $+\$25,344.00$ |
| **Cash Tax Shield ($\Delta T = 0.27 \cdot D$)** | **$+\$7,464.96$** | **$+\$7,464.96$** | **$+\$14,307.84$** | **$+\$6,842.88$** |
| **Net Marginal Cash Flow ($CF$)** | **$+\$5,084.16$** | **$+\$5,958.56$** | **$+\$13,652.84$** | **$+\$8,568.68$** |
| **Net GHG Avoidance** | $0.0\text{ tCO}_2\text{e}$ (Ref) | $+15.60\text{ tCO}_2\text{e}$ | $+13.35\text{ tCO}_2\text{e}$ | $+13.35\text{ tCO}_2\text{e}$ |
| **Nutritious Meals Delivered** | 0 meals | 0 meals | **19,200 meals** | $+19,200\text{ meals}$ |

**Result Analysis:** Donating the 24 pallets of Greek yogurt yields an incremental **$+\$8,568.68$ in net cash profit** compared to landfilling the inventory, while providing 19,200 nutrient-dense protein portions (converted at USDA standard $0.6\text{ kg} = 1\text{ meal}$) to the community and avoiding $13.35\text{ tCO}_2\text{e}$ in landfill methane emissions.

---

### 6.3 Recovery Network Routing Optimization

Following delivery of the 24 pallets ($11,520\text{ kg}$) to the Regional Food Bank Hub (Node 0), the food bank must route its two refrigerated 26-foot straight trucks ($K = \{k_1, k_2\}$) to distribute the yogurt across 4 community pantries before their afternoon distribution sessions.

#### Network Node Parameters

| Node ID | Node Name | Type | Lat/Lon Dist from Hub ($d_{0i}$) | Demand ($D_i$) | Time Window $[e_i, l_i]$ | Unload Time ($s_i$) | Walk-in Cooler Cap |
|---|---|---|---:|---:|---|---:|---:|
| **0** | **Regional Food Bank Hub** | Depot | $0.0\text{ km}$ | Origin ($11,520\text{ kg}$) | $[07:00, 16:00]$ | — | $>100\text{ t}$ |
| **1** | **Valley Community Pantry** | Agency | $18.5\text{ km}$ | $3,840\text{ kg}$ (8 plt) | $[08:30, 10:30]$ | $30\text{ min}$ | $4,500\text{ kg}$ |
| **2** | **Hope Mission Center** | Agency | $29.0\text{ km}$ | $2,880\text{ kg}$ (6 plt) | $[09:00, 11:30]$ | $25\text{ min}$ | $3,500\text{ kg}$ |
| **3** | **St. Jude Food Shelf** | Agency | $36.2\text{ km}$ | $2,880\text{ kg}$ (6 plt) | $[10:00, 12:30]$ | $25\text{ min}$ | $3,200\text{ kg}$ |
| **4** | **Southside Family Center** | Agency | $22.4\text{ km}$ | $1,920\text{ kg}$ (4 plt) | $[11:00, 13:30]$ | $20\text{ min}$ | $2,500\text{ kg}$ |

#### Fleet Parameters
- Vehicles: Two identical 26-foot straight trucks with refrigeration ($T_{reefer} = 3.5^\circ\text{C}$).
- Capacity per vehicle: $Q_k^{wt} = 6,500\text{ kg}$ (maximum payload), $Q_k^{vol} = 12\text{ pallets}$.
- Average speed: $v = 45\text{ km/h}$ in mixed urban/suburban traffic ($t_{ij} = d_{ij} / 45 \times 60\text{ min}$).
- Variable vehicle operating cost: $c_{ij} = \$1.80\text{ per km}$ (diesel + driver + maintenance).

#### Distance Matrix ($d_{ij}$ in kilometers)

| Node | 0 (Hub) | 1 (Valley) | 2 (Hope) | 3 (St. Jude) | 4 (Southside) |
|---|---:|---:|---:|---:|---:|
| **0 (Hub)** | 0.0 | 18.5 | 29.0 | 36.2 | 22.4 |
| **1 (Valley)** | 18.5 | 0.0 | 14.2 | 26.5 | 21.0 |
| **2 (Hope)** | 29.0 | 14.2 | 0.0 | 15.8 | 24.5 |
| **3 (St. Jude)** | 36.2 | 26.5 | 15.8 | 0.0 | 17.5 |
| **4 (Southside)**| 22.4 | 21.0 | 24.5 | 17.5 | 0.0 |

#### Route Solving and Verification

- **Vehicle 1 Assignment (Pallets: $8 + 4 = 12\text{ pallets} = 5,760\text{ kg} \le 6,500\text{ kg}$):**
  - Route: $0 \to 1 \to 4 \to 0$
  - Leg 1 ($0 \to 1$): Depart Hub at 07:45. Distance = $18.5\text{ km}$, transit time = $24.7\text{ min}$. Arrive Node 1 at 08:10. Time window opens at 08:30 (truck waits 20 min). Service 08:30–09:00. Unload 8 pallets ($3,840\text{ kg}$). Remaining payload: $1,920\text{ kg}$.
  - Leg 2 ($1 \to 4$): Depart Node 1 at 09:00. Distance = $21.0\text{ km}$, transit time = $28.0\text{ min}$. Arrive Node 4 at 09:28. Time window opens at 11:00 (truck must wait, or schedule adjusted to depart Hub at 09:45).
  - *Schedule Optimization for Vehicle 1:*
    - Depart Hub at 08:05 $\to$ Arrive Node 1 at 08:30 (exact window start). Unload 08:30–09:00.
    - Dwell / buffer until 10:32 $\to$ Depart Node 1 at 10:32. Transit $28.0\text{ min} \to$ Arrive Node 4 at 11:00 (exact window start).
    - Unload 4 pallets ($1,920\text{ kg}$) 11:00–11:20. Remaining payload: $0\text{ kg}$.
    - Leg 3 ($4 \to 0$): Depart Node 4 at 11:20. Distance = $22.4\text{ km}$, transit time = $29.9\text{ min}$. Arrive Hub at 11:50.
  - Total Vehicle 1 Distance: $18.5 + 21.0 + 22.4 = 61.9\text{ km}$.
  - Total Vehicle 1 Operating Cost: $61.9\text{ km} \times \$1.80/\text{km} = \$111.42$.

- **Vehicle 2 Assignment (Pallets: $6 + 6 = 12\text{ pallets} = 5,760\text{ kg} \le 6,500\text{ kg}$):**
  - Route: $0 \to 2 \to 3 \to 0$
  - Leg 1 ($0 \to 2$): Depart Hub at 08:21. Distance = $29.0\text{ km}$, transit time = $38.7\text{ min}$. Arrive Node 2 at 09:00 (exact window start). Unload 6 pallets ($2,880\text{ kg}$) 09:00–09:25. Remaining payload: $2,880\text{ kg}$.
  - Leg 2 ($2 \to 3$): Depart Node 2 at 09:35. Distance = $15.8\text{ km}$, transit time = $21.1\text{ min}$. Arrive Node 3 at 09:56. Time window opens at 10:00 (truck waits 4 min). Unload 6 pallets ($2,880\text{ kg}$) 10:00–10:25. Remaining payload: $0\text{ kg}$.
  - Leg 3 ($3 \to 0$): Depart Node 3 at 10:25. Distance = $36.2\text{ km}$, transit time = $48.3\text{ min}$. Arrive Hub at 11:13.
  - Total Vehicle 2 Distance: $29.0 + 15.8 + 36.2 = 81.0\text{ km}$.
  - Total Vehicle 2 Operating Cost: $81.0\text{ km} \times \$1.80/\text{km} = \$145.80$.

- **Combined Distribution Performance:**
  - Total fleet travel distance: $61.9 + 81.0 = 142.9\text{ km}$.
  - Total distribution transportation cost: $\$111.42 + \$145.80 = \$257.22$.
  - All 4 agency delivery time windows satisfied with zero violations.
  - Maximum transit thermal exposure: Both trucks maintained at $3.5^\circ\text{C}$ continuous reefer setpoint; zero degrees-hours of cold chain excursion.
  - Remaining Shelf Life upon Delivery: $RSL \ge 11.5\text{ days}$ across all four agencies (far exceeding the mandatory agency buffer $RSL_{min} = 2.0\text{ days}$).

---

## 7. Code, query, or calculation

The following standalone Python script implements an automated food donation economic evaluator and dynamic CVRP-TW-P heuristic routing optimizer. It computes the IRC Section 170(e)(3) enhanced tax deduction, evaluates marginal cash flows across disposal pathways, and verifies time window and remaining shelf life constraints across delivery agencies.

```python
"""Food Waste Diversion Economics & Recovery Network Routing Optimizer."""
import math

class FoodDonationOptimizer:
    def __init__(self, cost_basis: float, fmv: float, mass_tons: float, tax_rate: float):
        self.c = cost_basis
        self.fmv = fmv
        self.mass = mass_tons
        self.tau = tax_rate

    def evaluate_tax_deduction(self) -> dict:
        tentative = self.c + 0.5 * max(0.0, self.fmv - self.c)
        ceiling = 2.0 * self.c
        enhanced_deduction = min(tentative, ceiling)
        cash_shield_don = self.tau * enhanced_deduction
        cash_shield_disposal = self.tau * self.c
        return {
            "enhanced_deduction": enhanced_deduction,
            "cash_shield_donation": cash_shield_don,
            "cash_shield_disposal": cash_shield_disposal,
            "incremental_tax_shield": cash_shield_don - cash_shield_disposal
        }

    def evaluate_pathways(self, hauling_land: float, tip_fee_ton: float,
                          hauling_don: float, staging_don: float) -> dict:
        tax = self.evaluate_tax_deduction()
        land_cost = hauling_land + (self.mass * tip_fee_ton)
        land_cf = tax["cash_shield_disposal"] - land_cost

        don_cost = hauling_don + staging_don
        don_cf = tax["cash_shield_donation"] - don_cost

        return {
            "landfill_net_cf": land_cf,
            "donation_net_cf": don_cf,
            "net_donation_advantage": don_cf - land_cf
        }

def route_evaluator(dist_matrix, routes, time_windows, service_times, speed_kmh=45.0):
    total_km, total_minutes = 0.0, 0.0
    schedule = {}
    for r_id, route in enumerate(routes):
        curr_time = time_windows[0][0]  # Start at depot open
        curr_km = 0.0
        for i in range(len(route) - 1):
            u, v = route[i], route[i+1]
            d = dist_matrix[u][v]
            curr_km += d
            travel_min = (d / speed_kmh) * 60.0
            arr_time = curr_time + travel_min
            start_srv = max(arr_time, time_windows[v][0])
            dep_time = start_srv + service_times[v]
            schedule[f"Truck_{r_id+1}->Node_{v}"] = {
                "arr": round(arr_time, 1), "dep": round(dep_time, 1),
                "wait": round(max(0.0, time_windows[v][0] - arr_time), 1),
                "window_ok": arr_time <= time_windows[v][1]
            }
            curr_time = dep_time
        total_km += curr_km
    return total_km, schedule

# Simulation of SunValley 24-pallet Greek Yogurt Case
opt = FoodDonationOptimizer(cost_basis=27648.0, fmv=78336.0, mass_tons=12.48, tax_rate=0.27)
econ = opt.evaluate_pathways(hauling_land=1320.00, tip_fee_ton=85.0,
                             hauling_don=550.0, staging_don=105.0)

dist = [
    [0.0, 18.5, 29.0, 36.2, 22.4],
    [18.5, 0.0, 14.2, 26.5, 21.0],
    [29.0, 14.2, 0.0, 15.8, 24.5],
    [36.2, 26.5, 15.8, 0.0, 17.5],
    [22.4, 21.0, 24.5, 17.5, 0.0]
]
tw = [[420, 960], [510, 630], [540, 690], [600, 750], [660, 810]]  # Min from midnight
srv = [0, 30, 25, 25, 20]
tot_km, sched = route_evaluator(dist, [[0, 1, 4, 0], [0, 2, 3, 0]], tw, srv)

print(f"Enhanced Deduction: ${econ['donation_net_cf']:,.2f} vs Landfill: ${econ['landfill_net_cf']:,.2f}")
print(f"Net Cash Advantage of Donation: ${econ['net_donation_advantage']:,.2f}")
print(f"Total Fleet Traversed: {tot_km:.1f} km | All Windows Valid: {all(s['window_ok'] for s in sched.values())}")
```

### 7.1 Assumptions of the Script
1. The company has sufficient corporate taxable income under IRC § 170(b)(2) so that charitable deductions are not capped at the 15% annual threshold.
2. The product satisfies all food safety requirements under the Bill Emerson Good Samaritan Act, with no unrecorded temperature excursions.
3. Travel velocities are assumed constant ($45\text{ km/h}$) across suburban arterials; wait times are absorbed without vehicle refrigeration failure.

### 7.2 Expected Output
```text
Enhanced Deduction: $13,652.84 vs Landfill: $5,084.16
Net Cash Advantage of Donation: $8,568.68
Total Fleet Traversed: 142.9 km | All Windows Valid: True
```

---

## 8. Data, standards, and regulatory context

Enterprise food loss, waste diversion, and charitable recovery systems must operate within a complex regulatory landscape spanning federal tax law, statutory liability protections, international waste accounting frameworks, and cold chain safety mandates.

```text
+----------------------------------------------------------------------------------------------------+
| REGULATORY & STANDARDS COMPLIANCE MATRIX                                                           |
|                                                                                                    |
|  1. ACCOUNTING & REPORTING    2. TAX INCENTIVES           3. LIABILITY SHIELDS    4. FOOD SAFETY   |
|  - WRI FLW Protocol          - US IRC § 170(e)(3)        - Emerson Act (1996)    - FDA FSMA        |
|  - GHG Protocol Scope 3      - PATH Act (2015)           - FDIA (Pub L 117-328)  - FDA Food Code   |
|  - EPA Wasted Food Scale     - State Tax Credits         - EU General Food Law   - USDA FSIS       |
+----------------------------------------------------------------------------------------------------+
```

### 8.1 The Food Loss and Waste Accounting and Reporting Standard (FLW Protocol)
The FLW Protocol (established by the World Resources Institute, UNEP, FAO, and Consumer Goods Forum) represents the global metric standard for corporate waste accounting.
- **Ten Statutory Destinations:** The FLW Protocol strictly categorizes food diversion into 10 destinations:
  1. *Animal feed*
  2. *Bio-based materials/biochemical processing*
  3. *Anaerobic digestion*
  4. *Composting/aerobic process*
  5. *Controlled combustion (incineration)*
  6. *Land application*
  7. *Landfill*
  8. *Sewer/wastewater treatment*
  9. *Refuse/discards/litter*
  10. *Co-digestion/other*
- **Accounting Distinction for Human Redistribution:** Under the FLW Protocol, food rescued or donated for human consumption is classified as **redistributed food**. Because it fulfills its intended biological purpose of nourishing humans, redistributed food is accounted for as a positive reduction in total waste, directly improving corporate Food Recovery Efficiency (FRE).

### 8.2 US Tax Code: IRC Section 170(e)(3) and the PATH Act of 2015
- **Qualified Beneficiaries:** To qualify for the enhanced deduction under Treasury Regulation § 1.170A-4A, the recipient organization must be an exempt 501(c)(3) public charity that uses the food solely for the care of the ill, the needy, or infants. The charity cannot transfer the food in exchange for money, property, or services.
- **Written Representation Mandate:** The donor must receive an official written statement from the donee organization stating that:
  1. The donated inventory will be used in compliance with Section 170(e)(3);
  2. Adequate books and records will be maintained;
  3. The donee will not charge any fee or consideration to end recipients.
- **PATH Act Expansion:** Prior to 2015, the enhanced deduction was restricted to Subchapter C corporations. The Protecting Americans from Tax Hikes (PATH) Act of 2015 permanently extended Section 170(e)(3) eligibility to all business entities, including S-corporations, partnerships, LLCs, and sole proprietorships, while raising the annual deduction cap from 10% to 15% of net business income.

### 8.3 The Bill Emerson Act and the Food Donation Improvement Act of 2022
- **Bill Emerson Good Samaritan Food Donation Act (42 U.S.C. § 1791):** Enacted in 1996, the statute immunizes donors and non-profit gleaners from civil and criminal liability arising from the nature, age, condition, or packaging of donated food, provided the food is apparently wholesome and donated in good faith without gross negligence.
- **The Food Donation Improvement Act of 2022 (FDIA):** Signed into law in December 2022 (Pub. L. 117-328), the FDIA resolved two major operational limitations of the 1996 Act:
  1. *Direct Donations to Individuals:* Authorized qualified food donors (retailers, restaurants, commercial caterers, school food authorities) to donate food directly to needy individuals without routing through an intermediary 501(c)(3) organization.
  2. *Deeply Discounted Food:* Extended liability protection to non-profit organizations that sell donated food at a nominal, deeply discounted price designed purely to offset handling and logistics overhead (facilitating non-profit grocery store models).

### 8.4 International Statutory Mandates: The French *Loi Garot* and EU Circular Directives
- **France (Loi Garot, Law No. 2016-138):** The world's first national legislation criminalizing grocery food waste. Supermarkets exceeding $400\text{ m}^2$ are legally barred from deliberately destroying, bleaching, or rendering unsold edible food unconsumable. Retailers must negotiate and sign formal donation partnerships with authorized food relief charities. Infractions carry fines of up to €3,750 per occurrence, escalating to €75,000 and two years imprisonment for systemic corporate non-compliance.
- **European Union Waste Framework Directive (Directive 2008/98/EC as amended by Directive (EU) 2018/851):** Legally obligates EU member states to reduce food waste by 30% by 2025 and 50% by 2030 (aligning with UN Sustainable Development Goal 12.3), mandating member states to introduce tax incentives for food donation and enforce organic waste segregation at commercial facilities.

---

## 9. Industry practice and case evidence

Corporate supply chains have evolved sophisticated food recovery operations to bridge the gap between commercial surplus generation and charitable logistics networks.

```text
+----------------------------------------------------------------------------------------------------+
| COMMERCIAL BENCHMARKS & TECHNOLOGY PLATFORMS                                                       |
|                                                                                                    |
|  [ Feeding America: MealConnect ]  ──> Algorithmic direct dispatch; 700M+ lbs annual retail rescue |
|  [ Kroger: Zero Hunger | Zero Waste ] ──> 500M+ lbs donated; organic waste to anaerobic digesters  |
|  [ Dynamic Secondary Apps ]       ──> Flashfood / Too Good To Go: Dynamic consumer markdown sales  |
+----------------------------------------------------------------------------------------------------+
```

### 9.1 Feeding America and the MealConnect Platform
Feeding America, the largest domestic hunger relief network in the US (comprising 200 central regional food banks and 60,000 local pantries and meal programs), historically relied on central hub-and-spoke consolidation. Pallets of retail supermarket surplus were picked up by regional food bank tractor-trailers, hauled to a central warehouse, un-palletized, sorted, inventoried, and re-dispatched to community pantries. 

This centralized model introduced 48 to 96 hours of reverse logistics latency, rendering the recovery of highly perishable items (prepared hot foods, delicate fresh berries, fluid dairy) infeasible.

To eliminate this latency, Feeding America engineered **MealConnect**, an algorithmic logistics platform:
- **Direct Dispatch Architecture:** When a retail supermarket manager (e.g., Walmart, Target, Kroger) flags surplus inventory, the store associate logs the commodity type, pallet/case count, and code date via a mobile barcode scanner.
- **Dynamic Matching Engine:** MealConnect's algorithm evaluates local agency distance, refrigerated vehicle availability, walk-in cooler capacity, and historical pickup reliability to immediately match the donation with a certified local food pantry within a 10-mile radius.
- **Operational Scale:** MealConnect processes more than 700 million pounds of retail food rescues annually, reducing logistics cycle time from days to hours and eliminating central warehouse handling costs.

### 9.2 The Kroger Company: "Zero Hunger \| Zero Waste"
The Kroger Co. (operating 2,700 supermarkets and 45 distribution centers) established an integrated food diversion framework across three synchronized streams:
1. **Upstream In-Store Markdowns:** Implementing dynamic computer-assisted ordering (CAO) and zebra-scanner price markdowns (25% to 50% discount) when products reach 2 to 3 days of remaining shelf life.
2. **Midstream Charitable Donation:** If markdowns fail to clear inventory 24 hours prior to expiration, products are staged in dedicated backroom walk-in coolers for daily scheduled Feeding America partner pickups. In 2023, Kroger directed over 180 million pounds of surplus food to hunger relief.
3. **Downstream Anaerobic Digestion:** For unrecoverable organic material (liquids, broken packaging, expired produce), Kroger installed an industrial-scale wastewater and solids anaerobic digestion system at its Compton, California distribution center. The system converts 55,000 tons of commercial food waste annually into renewable biogas, powering the DC's material handling equipment and offsetting corporate grid electricity demands.

### 9.3 Dynamic Consumer Markdown Apps: Flashfood and Too Good To Go
A growing segment of retail food waste is diverted through secondary commercial marketplaces that operate between standard retail sales and charitable donation:
- **Flashfood:** Integrates with grocery inventory management systems (e.g., Meijer, Loblaw, Giant Eagle). Items nearing code date are listed on a consumer smartphone app at a 50% discount. Consumers purchase items digitally and retrieve them from a branded "Flashfood Zone" walk-in refrigerator located at the front of the retail store.
- **Too Good To Go:** Operates a "Surplus Surprise Bag" model across 85,000 restaurants, bakeries, and supermarkets globally. Businesses aggregate unsold daily surplus into bags sold at one-third of retail price during a designated 30-minute evening pickup window, capturing residual revenue while eliminating disposal costs.

---

## 10. Failure modes and red flags

Operating a commercial food recovery and diversion network carries severe financial, operational, and legal risks. The following failure matrix identifies critical breakdown mechanisms and operational mitigations.

| Failure Mode | Early Warning Sign | Root Physical / Logistical Mechanism | Operational / Financial Consequence | Engineered Mitigation |
|---|---|---|---|---|
| **1. Donation Dumping** | Relief agencies complain of high disposal burdens; sorting bins filled with putrid food. | Donors stage rotten or expired inventory as "donations" to evade municipal landfill tipping fees. | Agency cuts off donor partnership; donor loses Good Samaritan liability immunity; reputational damage. | Institute strict receiving acceptance criteria at food banks; implement donor quality grading scorecards. |
| **2. Cold Chain Rupture in Transit** | Dairy containers expand/bloat; temperature data logger indicates $T > 8^\circ\text{C}$ upon arrival. | Volunteer-driven ambient vehicles or broken truck reefer units transport TCS foods during summer heat. | Microbial proliferation (e.g., *Listeria*, *Salmonella*); product must be dumped; foodborne illness risk. | Mandate passive thermal blankets, continuous Bluetooth BLE data loggers, and reefer checks before release. |
| **3. Tax Audit Clawback** | IRS auditor issues notice of deficiency regarding Section 170(e)(3) valuation deductions. | Valuation models use full list MSRP for near-expiration goods; lack of contemporaneous donee written acknowledgments. | Retroactive disallowance of enhanced deduction; repayment of tax savings plus 20% accuracy penalties. | Retain audited donee 170(e)(3) statements; establish documented formula discounting FMV based on remaining RSL. |
| **4. Agency Capacity Choke** | Donation rejected at agency dock; truck deadheads back to DC with full payload. | Bulky surplus shipment (e.g., 20 pallets of potatoes) assigned to small pantry with 2-door commercial reach-in. | Emergency disposal costs; wasted driver hours; perishable food decays while awaiting re-routing. | Dynamic capacity verification in matching algorithms; enforce mandatory capacity reservation before dispatch. |
| **5. Perishability Cliff Delivery** | Food arrives at pantry with $<6\text{ hours}$ before printed expiration date. | Excessive staging latency at donor DC or congested multi-stop routing schedules. | Agency cannot distribute food to clients before expiration; pantry volunteers forced to discard product. | Set hard constraint in CVRP-TW-P routing: $RSL_{delivery} \ge RSL_{min}$ (e.g., minimum 48 hours for pantries). |
| **6. Unlabeled Allergen Contamination** | Bulk bakery or deli surplus packed in unlabeled clear clamshell containers. | In-store prepared foods donated without statutory ingredient and Big-9 food allergen disclosures. | Severe anaphylaxis hazard for food-insecure clients; violation of Food Code; liability under Emerson Act. | Ban donation of unlabeled open-deli items; require pre-printed standardized allergen ingredient stickers. |
| **7. Channel Cannibalization Paranoia** | Store managers deliberately padlock dumpsters or bleach food to prevent salvage. | Misguided corporate belief that donating food reduces primary retail sales to lower-income shoppers. | Increased landfill costs; employee morale degradation; violation of French *Loi Garot* or local organic bans. | Publish corporate cannibalization studies proving negligible market overlap; link manager KPIs to diversion rates. |
| **8. Deadheading Rescue Fleet** | Rescue trucks operate with low volumetric utilization ($<25\%$) on return legs. | Point-to-point dispatching without reverse consolidation of empty crates, pallets, or returnable totes. | Excessive transportation cost per pound of recovered food; unsustainable grant/operating budget burn. | Multi-tier backhauling: schedule pallet and reusable plastic container (RPC) returns on agency return legs. |
| **9. Contamination of Organic Digester Feedstock** | Anaerobic digester operator rejects truckload or assesses severe contamination surcharges. | Food waste packaged in plastic films, aluminum trays, or glass commingled into organic bulk tankers. | High de-packaging penalty fees (\$150+/ton); entire load rejected and diverted to municipal landfill. | Install automated front-end mechanical de-packagers; segregate clean bulk agricultural streams from retail packaged goods. |
| **10. ESG Scope 3 Double-Counting** | Both food donor and food bank claim the exact same $\text{CO}_2\text{e}$ landfill diversion offsets. | Lack of clear ownership boundary specifications under GHG Protocol Scope 3 Category 5 accounting. | External ESG auditing failure; greenwashing accusations from investors and environmental regulators. | Establish clear contractual attribution: donor claims Scope 3 Cat 5 operational diversion; donee claims Scope 3 Cat 1. |

---

## 11. Assumptions and limits

The mathematical models, tax formulations, and routing algorithms developed in this topic depend on foundational assumptions that bound their real-world applicability.

| Assumption | Operational Rationale | Failure Condition / Boundary Limit | Verification Test |
|---|---|---|---|
| **1. Profitable Corporate Tax Position** | Enhanced deductions under IRC § 170(e)(3) generate immediate cash savings only if the firm has positive taxable income. | If the enterprise is operating at a Net Operating Loss (NOL), deductions generate non-cash tax carryforwards with discounted NPV. | Review corporate effective tax rate and verify NOL status with corporate tax accounting before scheduling high-cost logistics. |
| **2. Active Secondary Commercial Pricing** | $FMV$ is calculated based on observable commercial transactions in the customary market segment. | If an item has never been sold commercially or is damaged beyond marketability, establishing defensible $FMV$ is legally difficult. | Audit historical price-point data; apply documented liquidation price discounts (e.g., secondary salvage index). |
| **3. Linear Kinetic Shelf-Life Decay** | The model assumes remaining shelf life $RSL(t)$ decays linearly with time under stable refrigerated setpoints. | Severe temperature abuse ($T > 15^\circ\text{C}$) triggers non-linear, exponential microbial growth, invalidating simple time-based $RSL$. | Validate cold chain integrity via continuous time-temperature indicators (TTI) or digital data logger records. |
| **4. Strict Good Samaritan Wholesomeness** | Donated items meet all applicable federal, state, and local safety standards at the moment of custody transfer. | If product is subject to an active FDA/USDA recall, Emerson Act immunity is voided; distribution is a criminal offense. | Cross-reference donor warehouse management systems (WMS) with FDA/USDA recall databases prior to release. |
| **5. Deterministic Routing Travel Times** | The CVRP-TW-P formulation assumes deterministic travel times $t_{ij} = d_{ij} / v$ between nodes. | Extreme urban traffic congestion or severe weather disrupts schedule, causing vehicles to miss strict agency time windows. | Incorporate historical time-dependent travel time distributions ($\mu_{ij}(t), \sigma_{ij}(t)$) and stochastic buffer windows. |
| **6. Non-Profit Agency Absorption Capacity** | Agencies have sufficient volunteer labor and client footfall to distribute received perishables before expiration. | Volunteer cancellations or client weather disruptions cause donated food to sit undistributed in pantry coolers. | Implement real-time agency inventory reporting via mobile apps (e.g., MealConnect capacity telemetry). |
| **7. Free Disposal at Donee Receiving** | Donee food banks and pantries accept donated inventory without assessing gate or processing fees. | In some metropolitan jurisdictions, food banks facing severe glut assess nominal handling fees (\$0.05/lb) on hard-to-move items. | Verify donee fee structures in Master Donation Service Level Agreements (SLAs). |
| **8. Homogeneous Unit Pallet Mass** | Mathematical models assume uniform weight and volume across pallet positions. | Irregular pallet configurations, mixed-SKU rainbow pallets, or lightweight bakery pallets distort vehicle weight/volume ratios. | Capture true dual-constraint pallet metrics (gross weight via scale forks, height/cube via automated dimensioners). |

---

## 12. Exercises

### Exercise 1: FLW Mass-Balance and Landfill Diversion Accounting
A regional dairy processing plant in Wisconsin processes $500,000\text{ kg}$ of raw milk weekly. During a production week, mass-balance monitoring yields the following breakdown:
- $420,000\text{ kg}$ packaged into finished commercial fluid milk (sold).
- $35,000\text{ kg}$ of sweet whey byproduct diverted to commercial animal feed.
- $15,000\text{ kg}$ of packaged whole milk rejected due to an incorrect UPC label printing error (fully pasteurized and wholesome, with 14 days of remaining shelf life).
- $18,000\text{ kg}$ of processing line wash-down flush commingled into on-site anaerobic digestion.
- $12,000\text{ kg}$ of spoiled curd and bio-solids hauled to a municipal landfill.

*Deliverable:* Construct an FLW Protocol mass-balance destination table. Calculate the corporate Food Recovery Efficiency (FRE), defined as the ratio of human-consumed food over total edible food generated. (Assume the rejected UPC-labeled milk is successfully donated to a local food bank). Compute the net $\text{tCO}_2\text{e}$ emissions avoided by donating the 15,000 kg of mislabeled milk rather than landfilling it, using $EF_{landfill} = 1.15\text{ tCO}_2\text{e}/\text{t}$ and $EF_{donation} = 0.04\text{ tCO}_2\text{e}/\text{t}$.

---

### Exercise 2: IRC Section 170(e)(3) Enhanced Tax Deduction Calculation
A national commercial bakery distribution center has 50 pallets of packaged artisan sourdough loaves that cannot be shipped to grocery retailers due to a packaging seal aesthetic defect that does not affect food safety:
- Total Units: 30,000 loaves.
- Capitalized Cost Basis ($C$): $\$0.85\text{ per loaf}$.
- Customary Fair Market Value ($FMV$): $\$3.65\text{ per loaf}$.
- Marginal Combined Corporate Income Tax Rate ($\tau$): $25.0\%$.

*Deliverable:* Compute:
1. The standard inventory tax deduction if the bread is shredded and landfilled.
2. The tentative enhanced deduction (Basis + Half Mark-up).
3. The statutory ceiling (Twice the Basis).
4. The allowable enhanced deduction under IRC § 170(e)(3).
5. The cash tax shield differential between donation and disposal.
6. The maximum logistics and handling cost the bakery can incur to transport the donation to a food bank while remaining financially superior to landfilling (assuming landfill tipping fee is $\$75/\text{t}$, product mass is 18 metric tons, and landfill hauling is $\$600$).

---

### Exercise 3: Comparative Marginal Cash Flow Analysis (Landfill vs AD vs Donation)
A produce packaging facility in California has 40 metric tons of fresh carrots rejected by an export customer due to cosmetic length specifications (carrots are sound, washed, and crisp):
- Cost Basis ($C$): $\$18,000.00$.
- Commercial Domestic Value ($FMV$): $\$38,000.00$.
- Tax Rate ($\tau$): $28.0\%$.
- Option A (Landfill): Local hauling = $\$400.00$; tip fee = $\$90.00/\text{t}$.
- Option B (Anaerobic Digestion): Long-distance hauling = $\$850.00$; organic gate fee = $\$40.00/\text{t}$.
- Option C (Food Bank Donation): Freight to regional food bank = $\$650.00$; donee receiving fee = $\$0.00$; staging labor = $\$200.00$.
- Option D (Animal Feed): Local cattle rancher picks up carrots on-site; pays $\$15.00/\text{t}$ to facility; zero hauling cost; ordinary commercial income.

*Deliverable:* Calculate the net marginal cash flow for all four options. Rank the options from highest to lowest cash return. Identify whether the highest cash return aligns with the highest tier of the EPA Wasted Food Scale.

---

### Exercise 4: Capacitated Vehicle Routing with Time Windows and Perishability
A regional food rescue organization operates a refrigerated 16-foot box truck ($Q = 3,500\text{ kg}$) to collect surplus from 3 retail supermarkets ($D_1, D_2, D_3$) and deliver to 2 pantries ($A_1, A_2$). The vehicle starts and ends at the central food bank depot (Node 0).
- Node 0 (Depot): Window $[08:00, 16:00]$.
- Donor 1 ($D_1$): Pick $1,200\text{ kg}$ chilled deli; $RSL = 48\text{ hrs}$; Window $[08:30, 10:00]$; Service = $20\text{ min}$.
- Donor 2 ($D_2$): Pick $1,500\text{ kg}$ dairy; $RSL = 72\text{ hrs}$; Window $[09:15, 10:45]$; Service = $25\text{ min}$.
- Donor 3 ($D_3$): Pick $900\text{ kg}$ produce; $RSL = 36\text{ hrs}$; Window $[10:00, 11:30]$; Service = $15\text{ min}$.
- Agency 1 ($A_1$): Deliver $1,600\text{ kg}$; Window $[11:00, 12:30]$; Service = $30\text{ min}$.
- Agency 2 ($A_2$): Deliver $2,000\text{ kg}$; Window $[12:00, 14:00]$; Service = $30\text{ min}$.

*Deliverable:* Propose a single feasible tour that visits all 5 nodes without violating vehicle capacity, time windows, or shelf-life limits. Calculate total travel time, arrival times, and waiting times assuming travel times between nodes average 20 minutes.

---

### Exercise 5: Legal Liability Analysis under the Food Donation Improvement Act (FDIA)
A regional supermarket chain donates prepared rotisserie chickens directly to a local encampment of unsheltered individuals. The chickens were held at the deli counter at $60^\circ\text{C}$ ($140^\circ\text{F}$) until 21:00 closing, then chilled in a blast chiller to $3^\circ\text{C}$ by 23:00, packed in insulated cooler totes with ice packs, and handed directly to individuals at 08:00 the following morning. Three recipients allege mild gastroenteritis and threaten a civil tort lawsuit.

*Deliverable:* Write a legal and food safety audit evaluation assessing:
1. Whether direct donation to individuals is protected under federal law.
2. Which specific statute governs the liability shield.
3. What standard of conduct the plaintiffs must prove to pierce the donor's statutory immunity.
4. Whether the documented temperature profile constitutes gross negligence.

---

### Exercise 6: Hub-and-Spoke vs Direct Agency Matching Design
A metropolitan food bank network evaluates two operating models for capturing 4 million kilograms of annual retail supermarket surplus:
- **Model A (Centralized Hub-and-Spoke):** Food bank trucks pick up all retail surplus, transport it to the central 100,000 sq ft warehouse, inspect/sort the inventory, and re-distribute it to 80 neighborhood pantries.
  - Sorting and handling cost: $\$0.12\text{ per kg}$.
  - Total fleet travel distance: $180,000\text{ km/year}$ @ $\$1.90/\text{km}$.
  - Transit and handling latency: 72 hours average.
  - Perishable spoilage shrink in warehouse: $8.5\%$.
- **Model B (Direct Digital Matching - MealConnect Style):** Neighborhood pantries pick up surplus directly from local supermarkets using their own volunteer-driven vans, directed by a cloud dispatch app.
  - App platform maintenance cost: $\$60,000\text{ per year}$.
  - Volunteer mileage reimbursement: $90,000\text{ km/year}$ @ $\$0.65/\text{km}$.
  - Food bank quality audit and training staff: $\$110,000\text{ per year}$.
  - Transit latency: 3 hours average.
  - Spoilage shrink: $1.8\%$.

*Deliverable:* Calculate total annual operating costs and total net kilograms of food delivered to clients for both models. Determine the cost per delivered kilogram. Identify the operational limitations of Model B.

---

## 13. Self-check answers

### Answer to Exercise 1: FLW Mass-Balance and Landfill Diversion Accounting
1. **Mass-Balance Destination Ledger:**
   - Total Inbound Milk: $500,000\text{ kg}$.
   - Finished Sold Product: $420,000\text{ kg}$ (Human Consumption - Primary Market).
   - Redistribution for Human Consumption (Donation): $15,000\text{ kg}$.
   - Animal Feed (Whey): $35,000\text{ kg}$.
   - Industrial Uses (Anaerobic Digestion): $18,000\text{ kg}$.
   - Landfill Disposal: $12,000\text{ kg}$.
   - Total Mass Accounted: $420,000 + 15,000 + 35,000 + 18,000 + 12,000 = 500,000\text{ kg}$.

2. **Food Recovery Efficiency (FRE):**
   - Total Edible Human Food Produced = Sold + Mislabeled Milk = $420,000 + 15,000 = 435,000\text{ kg}$.
   - Total Human Food Consumed = $420,000\text{ kg}$ (commercial) $+ 15,000\text{ kg}$ (donation) = $435,000\text{ kg}$.
   - $\text{FRE} = \frac{435,000}{435,000} = 100\%$ of potential edible human food recovered.
   - If evaluating surplus alone: Total surplus edible food = $15,000\text{ kg}$; Human recovery = $15,000\text{ kg} \implies \text{Surplus Recovery Rate} = 100\%$.

3. **Net GHG Avoidance:**
   - Mass Donated = $15,000\text{ kg} = 15.0\text{ metric tons}$.
   - $\Delta E_{GHG} = M \cdot (EF_{landfill} - EF_{donation}) = 15.0\text{ t} \times (1.15 - 0.04)\text{ tCO}_2\text{e}/\text{t} = 15.0 \times 1.11 = +16.65\text{ tCO}_2\text{e}$ avoided.

---

### Answer to Exercise 2: IRC Section 170(e)(3) Enhanced Tax Deduction
1. **Standard Disposal Deduction (Landfill):**
   - $C = 30,000 \times \$0.85 = \$25,500.00$.
   - Tax shield = $0.25 \times \$25,500.00 = \$6,375.00$.
2. **Tentative Enhanced Deduction:**
   - $FMV = 30,000 \times \$3.65 = \$109,500.00$.
   - Mark-up = $FMV - C = \$109,500.00 - \$25,500.00 = \$84,000.00$.
   - $D_{tentative} = C + \frac{1}{2}(FMV - C) = \$25,500.00 + \$42,000.00 = \$67,500.00$.
3. **Statutory Ceiling:**
   - $D_{ceiling} = 2 \cdot C = 2 \times \$25,500.00 = \$51,000.00$.
4. **Allowable Statutory Enhanced Deduction:**
   - $D_{enhanced} = \min(D_{tentative}, D_{ceiling}) = \min(\$67,500.00, \$51,000.00) = \$51,000.00$.
   *(Note: The 2x basis ceiling is binding because the product carried a high mark-up: $FMV = 4.29 \times C$).*
5. **Cash Tax Shield Differential:**
   - $\Delta T_{donation} = 0.25 \times \$51,000.00 = \$12,750.00$.
   - Incremental tax savings over disposal = $\$12,750.00 - \$6,375.00 = +\$6,375.00$.
6. **Maximum Indifference Logistics Cost:**
   - Landfill operational cost = Hauling + Tipping = $\$600.00 + (18\text{ t} \times \$75/\text{t}) = \$600.00 + \$1,350.00 = \$1,950.00$.
   - Net cash flow of landfill = $\Delta T_{disposal} - C^{ops}_{landfill} = \$6,375.00 - \$1,950.00 = +\$4,425.00$.
   - Set $CF_{donation} \ge CF_{landfill}$:
     $$\Delta T_{donation} - C^{logistics}_{donation} \ge +\$4,425.00$$
     $$\$12,750.00 - C^{logistics}_{donation} \ge \$4,425.00 \implies C^{logistics}_{donation} \le \$8,325.00$$
   - The bakery can incur up to **$\$8,325.00$** in handling and freight to donate the bread and still achieve a higher net cash return than landfilling.

---

### Answer to Exercise 3: Comparative Marginal Cash Flow Analysis
- Tax Rate $\tau = 0.28$.
- **Option A (Landfill):**
  - Outflows: Hauling $\$400.00$ + Tip $(40\text{ t} \times \$90/\text{t} = \$3,600.00) = \$4,000.00$.
  - Tax shield: $\tau \cdot C = 0.28 \times \$18,000.00 = +\$5,040.00$.
  - $CF_{landfill} = +\$5,040.00 - \$4,000.00 = +\$1,040.00$.
- **Option B (Anaerobic Digestion):**
  - Outflows: Hauling $\$850.00$ + Gate $(40\text{ t} \times \$40/\text{t} = \$1,600.00) = \$2,450.00$.
  - Tax shield: $\tau \cdot C = +\$5,040.00$.
  - $CF_{AD} = +\$5,040.00 - \$2,450.00 = +\$2,590.00$.
- **Option C (Food Bank Donation):**
  - Deduction: $C + 0.5(FMV - C) = \$18,000 + 0.5(\$38,000 - \$18,000) = \$18,000 + \$10,000 = \$28,000.00$.
  - Ceiling: $2C = \$36,000.00$. Allowable $D = \$28,000.00$.
  - Tax shield: $0.28 \times \$28,000.00 = +\$7,840.00$.
  - Outflows: Freight $\$650.00$ + Staging $\$200.00 = \$850.00$.
  - $CF_{donation} = +\$7,840.00 - \$850.00 = +\$6,990.00$.
- **Option D (Animal Feed):**
  - Revenue: $40\text{ t} \times \$15/\text{t} = +\$600.00$.
  - Incurred freight/staging: $\$0.00$.
  - Tax implications: Inventory cost basis $C = \$18,000$ was sold for $\$600$, creating an ordinary loss of $\$17,400.00$.
  - Tax shield on loss = $0.28 \times \$17,400.00 = +\$4,872.00$.
  - $CF_{feed} = \text{Revenue} + \text{Tax Shield} = \$600.00 + \$4,872.00 = +\$5,472.00$.
- **Ranking:**
  1. Option C (Donation): **$+\$6,990.00$**
  2. Option D (Animal Feed): **$+\$5,472.00$**
  3. Option B (Anaerobic Digestion): **$+\$2,590.00$**
  4. Option A (Landfill): **$+\$1,040.00$**
- **Hierarchy Alignment:** Yes. The highest financial cash return (Donation, \$6,990) corresponds perfectly with the highest tier of the EPA Wasted Food Scale (Human Consumption), followed by Tier 3 (Animal Feed), Tier 4 (Anaerobic Digestion), and Tier 5 (Landfill). The enhanced tax deduction successfully realigns financial self-interest with ecological and social welfare.

---

### Answer to Exercise 4: Capacitated Vehicle Routing
- Total load to collect: $1,200 + 1,500 + 900 = 3,600\text{ kg}$. Truck capacity is $3,500\text{ kg}$.
- *Crucial Observation:* The truck cannot pick up all donors before delivering! Total pickup ($3,600\text{ kg}$) exceeds capacity ($3,500\text{ kg}$). The tour must interleave pickups and deliveries.
- **Feasible Tour:** $0 \to D_1 \to D_2 \to A_1 \to D_3 \to A_2 \to 0$.
  - Depart Node 0 at 08:10.
  - Leg 1 ($0 \to D_1$): Travel 20 min $\to$ Arrive $D_1$ at 08:30 (window $[08:30, 10:00]$). Service 20 min. Depart 08:50. Load: $1,200\text{ kg}$.
  - Leg 2 ($D_1 \to D_2$): Travel 20 min $\to$ Arrive $D_2$ at 09:10. Window opens 09:15; wait 5 min. Service 25 min (09:15–09:40). Load aboard: $1,200 + 1,500 = 2,700\text{ kg} \le 3,500\text{ kg}$.
  - Leg 3 ($D_2 \to A_1$): Travel 20 min $\to$ Arrive $A_1$ at 10:00. Window opens 11:00; wait 60 min (or delay morning start by 60 min to depart 09:10). Service 30 min (11:00–11:30). Deliver $1,600\text{ kg}$. Remaining load aboard: $2,700 - 1,600 = 1,100\text{ kg}$.
  - Leg 4 ($A_1 \to D_3$): Travel 20 min $\to$ Arrive $D_3$ at 11:50. Window $[10:00, 11:30]$ would be missed if departing at 11:30!
  - *Corrected Schedule Adjustment:* Interleave Donor 1 and 3 first:
    Tour: $0 \to D_1 (1,200) \to D_3 (900) \to A_1 (-1,600) \to D_2 (1,500) \to A_2 (-2,000) \to 0$.
    1. Arrive $D_1$ at 08:30; depart 08:50. Load = $1,200\text{ kg}$.
    2. Travel 20 min $\to$ Arrive $D_3$ at 09:10. Wait until 10:00 (or shift start). Service 15 min $\to$ Depart 10:15. Load = $2,100\text{ kg}$.
    3. Travel 20 min $\to$ Arrive $A_1$ at 10:35. Wait until 11:00. Service 30 min $\to$ Depart 11:30. Deliver $1,600\text{ kg}$. Load = $500\text{ kg}$.
    4. Travel 20 min $\to$ Arrive $D_2$ at 11:50 (Window $[09:15, 10:45]$ missed!).
  - *Mathematical Conclusion:* A single vehicle cannot service all 5 nodes due to conflicting morning pickup windows. A two-vehicle split tour is required, or agency time windows must be negotiated for extension.

---

### Answer to Exercise 5: Legal Liability Analysis under FDIA
1. **Direct Donation Protection:** Yes. Under the Food Donation Improvement Act of 2022 (Pub. L. 117-328), Section 42 U.S.C. § 1791 was amended to explicitly protect qualified direct food donors donating directly to needy individuals.
2. **Governing Statute:** The Bill Emerson Good Samaritan Food Donation Act, as amended by the FDIA of 2022.
3. **Legal Standard to Pierce Immunity:** Plaintiffs must prove **gross negligence** or **intentional misconduct**. Under federal law, gross negligence is defined as conduct evincing a reckless disregard for the health or safety of others. Simple negligence (e.g., inadvertent minor temperature drift within acceptable culinary standards) is fully immunized.
4. **Safety Profile Assessment:** The documented chilling profile (cooling from $60^\circ\text{C}$ to $\le 4.4^\circ\text{C}$ within 2 hours) complies with FDA Food Code § 3-501.14 (which permits up to 2 hours to cool from $57^\circ\text{C}$ to $21^\circ\text{C}$ and a total of 6 hours to reach $5^\circ\text{C}$). The food was maintained under continuous refrigeration and distributed cold. This adheres to accepted professional food safety standards, conclusively defeating any claim of gross negligence. The donor's statutory immunity stands.

---

### Answer to Exercise 6: Hub-and-Spoke vs Direct Matching Design
1. **Model A (Centralized Hub-and-Spoke):**
   - Inbound volume: $4,000,000\text{ kg}$.
   - Spoilage shrink: $8.5\% \implies \text{Lost mass} = 340,000\text{ kg}$.
   - Net delivered food: $4,000,000 - 340,000 = 3,660,000\text{ kg}$.
   - Sorting/handling cost: $4,000,000\text{ kg} \times \$0.12/\text{kg} = \$480,000$.
   - Fleet transportation cost: $180,000\text{ km} \times \$1.90/\text{km} = \$342,000$.
   - Total annual operating cost: $\$480,000 + \$342,000 = \$822,000$.
   - Cost per delivered kilogram: $\frac{\$822,000}{3,660,000\text{ kg}} = \mathbf{\$0.2246 / kg}$.

2. **Model B (Direct Digital Matching):**
   - Inbound volume: $4,000,000\text{ kg}$.
   - Spoilage shrink: $1.8\% \implies \text{Lost mass} = 72,000\text{ kg}$.
   - Net delivered food: $4,000,000 - 72,000 = 3,928,000\text{ kg}$ ($+268,000\text{ kg}$ more food!).
   - App platform cost: $\$60,000$.
   - Volunteer mileage reimbursement: $90,000\text{ km} \times \$0.65/\text{km} = \$58,500$.
   - Quality and training staff: $\$110,000$.
   - Total annual operating cost: $\$60,000 + \$58,500 + \$110,000 = \$228,500$.
   - Cost per delivered kilogram: $\frac{\$228,500}{3,928,000\text{ kg}} = \mathbf{\$0.0582 / kg}$.

3. **Comparison and Operational Limits:**
   - Model B achieves a **74% cost reduction** per delivered kilogram ($\$0.0582$ vs $\$0.2246$) and delivers $268,000\text{ kg}$ of additional food by eliminating warehouse dwell time.
   - *Operational Limitations of Model B:*
     - Relies heavily on volunteer driver availability and vehicle reliability; high risk of missed pickups on weekends/holidays.
     - Pantries lack equipment to handle full pallet loads (cannot receive FTL farm/processor donations).
     - Decentralized quality control increases food safety monitoring variance across unmonitored volunteer vehicles.

---

## 14. Interview questions

### Question 1: Corporate Tax Economics of Food Recovery
**Interviewer:** *"Our CFO wants to know why we should pay a carrier \$600 to haul near-expiration packaged salads to a food bank when our local waste hauler will dump the container in a landfill for a \$200 tipping fee. Walk us through the financial cash flow mechanics."*

**Strong Answer:**  
"A CFO comparing the \$600 freight bill to a \$200 tipping fee is overlooking the tax shield differential created by US Internal Revenue Code Section 170(e)(3). 

When we landfill the salads, we take an ordinary inventory write-off of our cost basis $C$. At a 25% marginal corporate tax rate, our cash tax savings is $0.25 \times C$, minus the \$200 tipping fee and hauling.

However, under IRC Section 170(e)(3), donating inventory to a qualified 501(c)(3) food bank entitles us to an *enhanced deduction* equal to basis plus half the mark-up, up to twice the basis: $D = \min(2C, C + 0.5(FMV - C))$. For packaged salads—which typically have an $FMV$ three to four times the cost basis—the deduction reaches the statutory maximum of $2C$. 

Our cash tax savings becomes $0.25 \times 2C = 0.50 \times C$—literally doubling our tax cash shield. For a \$30,000 cost basis shipment, landfilling yields \$7,500 in tax savings minus \$200 tip fee (net $+\$7,300$). Donating yields \$15,000 in cash tax savings minus the \$600 freight bill (net $+\$14,400$). Donating produces an incremental **$+\$7,100$ in hard cash flow**, while eliminating landfill methane emissions and delivering thousands of meals."

**Weak Answer:**  
*"Donating food is great for our brand's ESG score and public relations, and nobody likes throwing food away. Even though the freight is more expensive, we can write off the full retail value of the product on our corporate taxes to make up the difference."* *(Fails because full retail value write-offs are illegal under IRS regulations, ignores the 2x basis cap, and treats the decision as PR rather than rigorous cash flow optimization).*

---

### Question 2: Algorithmic Routing Constraints in Food Rescue
**Interviewer:** *"If you are designing the routing and dispatch engine for an urban food rescue platform like MealConnect, what specific constraints must you add to a classical VRPTW formulation, and why?"*

**Strong Answer:**  
"A classical Vehicle Routing Problem with Time Windows (VRPTW) optimizes travel cost subject to vehicle capacity and customer arrival windows. For perishable food rescue, that formulation is fatally incomplete. We must extend it to a **Multi-Commodity Capacitated Vehicle Routing Problem with Time Windows and Perishability (CVRP-TW-P)** with four critical additions:

1. **Perishability and Remaining Shelf Life Constraints:** We must track commodity-specific remaining shelf life $RSL^m(t)$ as a state variable. An agency delivery is invalid unless $RSL^m(t_{delivery}) \ge RSL_{min}^m$. Otherwise, the algorithm schedules deliveries that arrive on code date, forcing the agency to dump the product.
2. **Compartment Temperature Compatibility:** Food rescue fleets are heterogeneous (ambient vans, dual-zone reefers, volunteer passenger cars). We must enforce binary temperature compatibility $\rho_k \ge \gamma^m$, ensuring TCS proteins and dairy are never assigned to ambient vehicles.
3. **Agency Capacity and Throughput Bounds:** Traditional VRP assumes drop nodes can absorb unlimited deliveries up to total route volume. Community pantries have rigid walk-in cooler capacity bounds. Delivery volume cannot exceed available refrigerated cubic volume at arrival.
4. **Interleaved Pickup-and-Delivery:** Donors and agencies are geographically interspersed. Vehicles do not simply load at a central depot and deliver; they must collect at donor nodes and drop off at agencies mid-tour without exceeding intermediate gross vehicle weight limits."

**Weak Answer:**  
*"You just take the standard Google OR-Tools VRP model, put in the GPS coordinates of the grocery stores and soup kitchens, and minimize total driving distance so drivers burn less gas."* *(Completely ignores temperature zones, perishability decay, capacity bottlenecks, and interleaved pickups).*

---

### Question 3: Legal Liability Frameworks
**Interviewer:** *"A retail store manager refuses to let volunteers pick up day-old bagged salads, claiming that if someone gets sick from Listeria, our company will be sued for millions. How do you advise senior leadership?"*

**Strong Answer:**  
"The store manager's fear is legally unfounded under both federal statutory law and historical case evidence. 

Under the federal **Bill Emerson Good Samaritan Food Donation Act** (42 U.S.C. § 1791) and the **Food Donation Improvement Act of 2022**, food donors and non-profit gleaners are explicitly shielded from civil and criminal liability arising from the condition of donated food. To pierce this immunity, a plaintiff must prove **gross negligence** or **intentional misconduct**—meaning the donor knowingly donated food that was contaminated, adulterated, or subject to an active FDA recall. 

In the entire 28-year history of the Emerson Act, there has not been a single documented, published appellate legal case in the United States holding a food donor liable for illness from food donated in good faith. 

To operationalize complete legal safety, we establish an audited standard operating procedure: verify that the salads were maintained at $\le 4^\circ\text{C}$ with continuous cold chain logging, confirm the product is not under active recall, ensure primary hermetic packaging is intact, and obtain an annual signed 170(e)(3) compliance acknowledgment from the recipient food bank."

**Weak Answer:**  
*"The manager is right to be careful because anyone can sue for anything. We should just require the food bank to sign a waiver saying they won't sue us, or only donate dry canned beans instead of fresh produce."* *(Fails to understand federal statutory preemption, ignores the Emerson Act and FDIA, and recommends unnecessary food waste).*

---

### Question 4: The French *Loi Garot* vs US Voluntary Frameworks
**Interviewer:** *"Compare the European legislative approach to food waste, such as France's Loi Garot, with the United States' framework. What are the operational implications for a multinational grocery retailer?"*

**Strong Answer:**  
"The operational divergence stems from **command-and-control statutory mandates** versus **market-based tax incentives**:

- **The French / EU Model (*Loi Garot*):** France establishes a statutory prohibition. Grocery stores $>400\text{ m}^2$ are legally barred from discarding or destroying edible food, facing fines of €3,750 per infraction. Supermarkets must sign formal donation contracts with accredited charities. Operationally, this forces retailers to treat donation as a mandatory, scheduled waste stream, shifting capital investment into in-store sorting and dedicated reverse-logistics staging.
- **The US Model:** The US framework is voluntary but driven by powerful tax incentives (IRC § 170(e)(3) enhanced deductions) and statutory liability shields (Emerson Act / FDIA). Landfilling remains legally permissible in most states, but the tax code creates a positive cash-flow incentive to divert surplus to charities.

For a multinational retailer (like Carrefour in Europe vs Walmart in the US), operations in France must be designed around **strict compliance tracking, audited donation contracts, and zero-landfill mandates**, whereas US operations are designed around **marginal tax shield optimization, donee receipt verification, and cold-chain liability risk management**."

**Weak Answer:**  
*"France bans food waste and makes stores give everything to poor people, while the US doesn't care about food waste and lets stores throw everything in the trash."* *(Grossly oversimplifies, ignores US tax incentives, Emerson Act, and state-level organic landfill bans like California SB 1383).*

---

## 15. Key terms

- **FLW Protocol:** Global standard developed by WRI, UNEP, and WBCSD for measuring and reporting food loss and waste mass flows.
- **EPA Wasted Food Scale:** The statutory waste hierarchy ranking prevention, donation, animal feed, AD, composting, and landfill.
- **IRC Section 170(e)(3):** US Internal Revenue Code provision granting an enhanced tax deduction for charitable inventory donations.
- **Enhanced Deduction:** Tax deduction equal to cost basis plus half the mark-up ($C + 0.5(FMV - C)$), capped at $2C$.
- **Cost Basis ($C$):** Capitalized inventoriable cost of goods under GAAP/tax accounting rules.
- **Fair Market Value ($FMV$):** Commercial sales price of food in its customary market segment on the donation date.
- **Bill Emerson Good Samaritan Act:** 42 U.S.C. § 1791; federal statute immunizing food donors from liability absent gross negligence.
- **Food Donation Improvement Act (FDIA):** 2022 US public law extending liability protection to direct-to-consumer food donations.
- **Loi Garot:** French statutory law criminalizing the destruction of edible unsold food by supermarkets $\ge 400\text{ m}^2$.
- **Food Bank:** Tier-1 central warehouse hub collecting and aggregating bulk food for distribution to agencies.
- **Food Pantry:** Tier-2 local distribution spoke distributing grocery baskets directly to food-insecure clients.
- **CVRP-TW-P:** Capacitated Vehicle Routing Problem with Time Windows and Perishability constraints.
- **Remaining Shelf Life ($RSL(t)$):** Dynamic time or degree-days remaining before food exceeds microbiological or quality safety.
- **Anaerobic Digestion (AD):** Biochemical process breaking down organic waste without oxygen to generate methane biogas.
- **Food Recovery Efficiency (FRE):** The mass fraction of edible surplus biomass successfully delivered for human consumption.
- **MealConnect:** Algorithmic logistics dispatch platform operated by Feeding America for direct retail food rescue.
- **Tipping Fee:** Gate fee assessed by landfills or organic processors per metric ton of solid waste received.
- **TCS Foods:** Time/Temperature Control for Safety foods requiring continuous cold chain ($\le 4.4^\circ\text{C}$).
- **Scope 3 Category 5:** Corporate GHG Protocol category tracking emissions from waste generated in enterprise operations.
- **Arrhenius Quality Kinetics:** Mathematical model describing the exponential acceleration of food spoilage as temperature rises.

---

## 16. Sources

- **[A]** World Resources Institute (WRI), WBCSD, UNEP, & FAO. (2016). *Food Loss and Waste Accounting and Reporting Standard (FLW Protocol)*. WRI Standard Architecture Specification.
- **[A]** US Internal Revenue Service (IRS). (2022). *26 U.S. Code § 170 - Charitable, etc., contributions and gifts (IRC Section 170(e)(3))*. Legal Information Institute.
- **[A]** US Congress. (1996). *Bill Emerson Good Samaritan Food Donation Act*, 42 U.S.C. § 1791, Public Law 104-210.
- **[A]** US Congress. (2023). *Food Donation Improvement Act of 2022*, Public Law 117-328, Division JJ, 136 Stat. 6089.
- **[A]** US Environmental Protection Agency (EPA). (2023). *From Field to Fork: The Wasted Food Scale and Lifecycle Greenhouse Gas Assessment of Food Recovery Pathways*. EPA Report 530-R-23-005.
- **[A]** French Republic. (2016). *Loi n° 2016-138 du 11 février 2016 relative à la lutte contre le gaspillage alimentaire (Loi Garot)*. Journal Officiel de la République Française.
- **[A]** Feeding America. (2023). *MealConnect Technology Architecture and Food Rescue Logistics Operations Manual*. Feeding America Research Series.
- **[A]** ReFED. (2023). *Roadmap to 2030: Reducing US Food Waste by 50%*. ReFED Financial & Environmental Analysis.
- **[A]** European Parliament and Council. (2018). *Directive (EU) 2018/851 amending Directive 2008/98/EC on waste*. Official Journal of the European Union, L 150/109.

---

## 17. Links

- [Module 32 Index: Food, Agriculture, and Perishable Supply Chains](_index.md)
- [Shelf Life, FEFO, and Age-Based Inventory](03_shelf-life-fefo.md)
- [Newsvendor with Salvage and Freshness Pricing](04_newsvendor-salvage.md)
- [Cold Chain and Quality Assurance](05_cold-chain-quality.md)
- [Traceability and Food Safety](06_traceability-food-safety.md)
- [Food Service and Quick Commerce Operations](08_food-service-quick-commerce.md)
- [Course Overview: Supply Chain Analytics & Data Science](../../../README.md)
- [Module 21: Supply Chain Network Design & Optimization](../../04_value_governance/21_supply_chain_network_design_optimization/_index.md)
