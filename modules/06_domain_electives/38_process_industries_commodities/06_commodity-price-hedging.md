# Commodity Price and Hedging: Transformation Margins, Basis Risk, and Forward Curve Dynamics

> Module 38 — Process Industries, Commodities, and Bulk Supply Chains
> Topic 06 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 15 (Stochastic Inventory & Risk-Pooling), Module 16 (Supply Chain Network Economics), Module 20 (Operations Planning & Master Scheduling), Module 38 Topic 01 (Continuous, Batch, and Discrete Production), Module 38 Topic 02 (Yield and Mass Balance Modeling)
> Version 2026.5

---

## 1. Why this topic exists

In discrete assembly manufacturing, procurement contracts for components such as microcontrollers, stamped metal enclosures, and electric motors are established via long-term master service agreements, negotiated bill-of-materials (BOM) cost schedules, or periodic request-for-quotation (RFQ) rounds. Once contracted, component input costs remain stable for quarters or years. Finished product wholesale pricing is similarly sticky. Supply chain volatility manifests primarily as lead-time uncertainty, component shortages, or expediting fees.

Process and commodity conversion industries—including petroleum refining, petrochemical synthesis, oilseed crushing, grain milling, base metal smelting, and thermoelectric power generation—operate in a radically different economic reality. In these industries, raw feedstocks and refined finished goods are fungible, globally traded commodities whose prices fluctuate second by second on central electronic exchanges (such as CME, ICE, CBOT, LME, and EEX). These prices are driven by macro-geopolitics, international cartels, seasonal crop weather, pipeline outages, shipping chokepoint congestions, and currency fluctuations:

1. **The Transformation Margin Dictates Solvency:** A commodity processor does not generate economic profit from the absolute level of commodity prices. A crude oil refiner does not become wealthy simply because crude oil reaches $\$120/\text{barrel}$, nor does an agribusiness crush plant fail simply because soybeans fall to $\$9/\text{bushel}$. Economic survival depends entirely on the **transformation margin** (or Gross Processing Margin, GPM): the financial spread between the aggregated market value of the output co-products and the cost of the input raw material, minus operating conversion utilities. When transformation margins collapse—such as when crude feedstock spikes faster than wholesale gasoline/diesel, or when electricity wholesale prices drop below the marginal cost of natural gas fuel—a world-scale processing asset running at 100% mechanical utilization can burn tens of millions of dollars in cash per month.
2. **The Disconnect Between Benchmark Futures and Plant-Gate Cash (Basis Risk):** A petroleum refiner in the Mid-Continent cannot deliver exchange-traded NYMEX Light Sweet Crude futures contracts directly into its distillation columns, nor can a feed mill in North Carolina feed CBOT benchmark corn futures directly into its grain elevators. Physical procurement and sales occur in localized physical cash markets. The physical price at the plant gate differs from the benchmark futures price by the **local basis**—a complex, time-varying differential driven by local pipeline tariffs, barge availability, terminal storage bottlenecks, regional supply-demand imbalances, and physical quality/assay variations (e.g., API gravity, sulfur content, moisture content, protein percentage). A hedging program that assumes futures perfectly mirror physical cash exposure suffers from **basis risk**, which can completely erase anticipated hedge protection.
3. **The Term Structure of Forward Curves (Cost of Carry vs. Convenience Yield):** Physical commodities cannot be held costlessly like equity shares. Holding physical inventory requires tankage or grain silos (warehousing costs), capital financing (interest costs), insurance, and risk of chemical deterioration or degradation. Simultaneously, physical inventory provides operational flexibility—preventing catastrophic refinery unit shutdowns or allowing immediate response to local price spikes. This operational value is the **convenience yield**. The equilibrium between carrying costs and convenience yield dictates the shape of the forward curve (contango versus backwardation). A processor rolling short-dated futures hedges across time incurs a continuous **roll yield** (or roll drag), which can either augment operating margins or inflict lethal financial hemorrhaging.
4. **Liquidity Shocks and Variation Margin Cash Traps:** Exchange-traded hedging instruments require initial margin deposits and daily cash mark-to-market variation margin settlement. When underlying commodity prices experience extreme regime shifts, a financially sound physical processor holding economically perfect hedges can face hundreds of millions of dollars in immediate cash margin calls before the underlying physical sales materialize. Without dedicated liquidity buffers, physical companies can be forced into liquidation or emergency restructuring despite holding profitable operational assets.

The catastrophic corporate failures of Metallgesellschaft ($1.3 billion roll-loss disaster), the 2022 European power utility liquidity crisis (€100+ billion emergency government credit guarantees), the negative WTI pricing event of April 20, 2020 ($-\$37.63/\text{bbl}$), and recurring agribusiness basis blowouts stem from treating commodity risk as a simple corporate finance afterthought. Mastering transformation margins, basis decomposition, forward curve term structures, and minimum-variance hedging is the prerequisite for protecting process industry balance sheets and linking master operational scheduling to physical trading execution.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Deconstruct** physical commodity cash prices into benchmark futures prices, geographic transport basis, quality assay adjustments, and terminal logistics premia.
2. **Formulate and calculate** multi-leg transformation margins across continuous conversion sectors: the 3:2:1 petroleum crack spread, the soybean crush spread, and spark/dark/clean power generation spreads.
3. **Model forward curve dynamics** using the Theory of Storage, quantifying the structural transition between contango ($F > S$) and backwardation ($F < S$) driven by storage costs, interest rates, and convenience yields.
4. **Derive and estimate** the optimal minimum-variance hedge ratio ($h^*$) and hedging effectiveness ($R^2$) using Ordinary Least Squares (OLS), error-correction models (ECM), and state-space Kalman filters on empirical time-series data.
5. **Evaluate** contract rolling mechanics, quantifying the financial impact of positive roll yield in backwardated curves versus negative roll drag in contango curves.
6. **Design** a multivariate hedging program for a continuous processing facility, allocating financial contracts across multiple input feedstocks and divergent output co-products.
7. **Size liquidity and capital reserves** required to support daily mark-to-market variation margin calls under extreme value distribution price shocks.
8. **Structure** hedging portfolios to comply with IFRS 9 / ASC 815 hedge accounting rules, distinguishing cash flow hedges from fair value hedges to prevent income statement earnings volatility.

---

## 3. The core mental model

The commodity supply chain is a coupled physical-financial transformation engine. The physical layer converts raw input streams into marketable product streams through capital-intensive thermochemical or mechanical units. The financial layer overlays exchange-traded and over-the-counter (OTC) derivative instruments to lock in conversion margins, isolate basis differentials, and finance inventory holding across time.

```text
=============================================================================================================
                    PHYSICAL-FINANCIAL COMMODITY CONVERSION & HEDGING ARCHITECTURE
=============================================================================================================

PHYSICAL TRANSFORMATION LAYER:
  Feedstock Input (t, bbl, bu)         Processing Asset (Refinery, Mill, Gen)      Co-Products Output
  [Physical Crude / Soybeans / Gas] ──> [ Catalytic Cracker / Expeller / Turbine ] ──> [ Gasoline, Diesel / Meal, Oil / MWh ]
                 │                                      │                                      │
                 ▼                                      ▼                                      ▼
PRICE FORMATION & BASIS LAYER:
  Local Feedstock Cash Price ($/unit)    Gross Processing Margin (GPM)           Local Product Cash Price ($/unit)
  S_in = F_bench_in + Basis_in           GPM_phys = Sum(w_j*S_out_j) - S_in       S_out = F_bench_out + Basis_out
                 │                                      │                                      │
                 ▼                                      ▼                                      ▼
FINANCIAL DERIVATIVE LAYER (HEDGING):
  Short/Long Input Benchmark Futures     Transformation Spread Derivatives       Short Output Benchmark Futures
  [ Long Futures / Swaps: F_in ]    ──> [ Crack / Crush / Spark Spread Lock ] <── [ Short Futures / Swaps: F_out ]
                 │                                      │                                      │
                 ▼                                      ▼                                      ▼
FORWARD CURVE TERM STRUCTURE & ROLL:
  Near-Month Contract (M1) ────────────> [ Calendar Roll: M1 -> M2 ] ──────────> Deferred Contract (M2)
  - Contango (F_M2 > F_M1): Storage full, convenience yield y low -> Negative Roll Yield (Roll Drag)
  - Backwardation (F_M2 < F_M1): Supply tight, convenience yield y high -> Positive Roll Yield (Roll Gain)
                 │
                 ▼
TREASURY & MARGIN LIQUIDITY BUFFER:
  Exchange Clearinghouse (CME / ICE) <──> Initial Margin (IM) + Daily Variation Margin (VM = Q * dF)
  [ Risk of Cash Exhaustion if Margin Spike Precedes Physical Delivery Settlement ]
=============================================================================================================
```

### The Three Interlocking Engines

1. **The Physical Conversion Engine:** Governed by stoichiometry, yield curves, and mass balance conservation ($\sum m_{in} = \sum m_{out}$). The processor purchases physical feedstock at a local delivery point (e.g., Cushing, Midland, Gulf Coast, or farm elevator) paying the local cash price $S_{in}$, consumes energy and operating catalysts, and sells multiple finished products at local cash prices $S_{out,j}$ in fixed or semi-flexible yield proportions $w_j$.
2. **The Basis Decoupling Engine:** Every physical cash price $S_t$ decomposes strictly into two orthogonal risk factors: the centralized global benchmark futures price $F_t$ and the localized physical basis $B_t$:
   $$S_t = F_t + B_t$$
   Centralized futures exchanges allow perfect hedging of the systemic flat price risk $F_t$. However, derivatives exchanges cannot clear hyper-localized physical disruptions (such as a local pipeline leak, a river freeze halting grain barges, or a regional refinery fire). The processor is therefore structurally exposed to **basis risk** ($\text{Var}(B_t)$).
3. **The Intertemporal Storage & Forward Curve Engine:** Forward prices across delivery tenors $T$ reflect the net cost of carrying physical inventory over time. The forward curve term structure is governed by the Kaldor-Working-Brennan Theory of Storage:
   $$F_{t,T} = S_t e^{(r + c - y)(T - t)}$$
   where $r$ is the cost of financing capital, $c$ is physical storage and insurance cost, and $y$ is the convenience yield. When physical supplies are abundant and inventories near tank capacity, $y \to 0$, driving the curve into **contango** ($F_{t,T} > S_t$). When immediate physical supply is acutely scarce, the marginal value of holding physical barrels or bushels spikes ($y > r + c$), driving the curve into **backwardation** ($F_{t,T} < S_t$).

---

## 4. Definitions and vocabulary

| Term | Notation / Unit | Formal Definition | Common Industry Misuse / Pitfall |
|---|---|---|---|
| **Flat Price** | $F_t$ or $S_{\text{bench}}$ ($/\text{unit}$) | The unadjusted, centralized exchange price for a standard grade at a specified delivery hub (e.g., CME WTI at Cushing, CBOT Corn at Chicago). | Confusing flat price with the all-in procurement cost delivered at the plant gate. |
| **Local Basis** | $B_t$ ($/\text{unit}$) | The arithmetic difference between the local physical cash price ($S_t$) and the reference benchmark futures price ($F_t$): $B_t = S_t - F_t$. | Assuming basis is constant over time; basis exhibits high seasonality and volatility. |
| **Location Basis** | $\Delta_{\text{loc}}$ ($/\text{unit}$) | The component of basis reflecting pipeline tariffs, marine freight, rail rates, or geographic distance between plant gate and benchmark hub. | Treating pipeline tariffs as fixed without modeling tariff proration and secondary capacity auction markets. |
| **Quality Basis** | $\Delta_{\text{qual}}$ ($/\text{unit}$) | The assay-driven price adjustment for physical differences (API gravity, sulfur, moisture, foreign material) relative to exchange par. | Assuming standard linear blending cancels out quality penalties; non-linear assays (e.g. RON, sulfur) compound discounts. |
| **Gross Processing Margin (GPM)** | $\text{GPM}$ ($/\text{unit feedstock}$) | The gross revenue generated from the yield-weighted finished co-products minus the cost of the raw feedstock. | Omitting volumetric shrinkage or operating utilities when comparing against financial spreads. |
| **3:2:1 Crack Spread** | $\text{CS}_{3:2:1}$ ($/\text{barrel}$) | Synthetic refining margin representing the conversion of 3 barrels of crude oil into 2 barrels of gasoline and 1 barrel of distillate/diesel. | Forgetting that product prices are quoted in cents/gallon while crude is quoted in dollars/barrel ($42\text{ gal/bbl}$). |
| **5:3:2 Crack Spread** | $\text{CS}_{5:3:2}$ ($/\text{barrel}$) | Refining spread representing high-conversion cracking: 5 barrels of crude yielding 3 barrels of gasoline and 2 barrels of distillate. | Applying standard 3:2:1 hedge ratios to a refinery whose actual catalytic cracking yield matches 5:3:2. |
| **Reverse Crack Spread** | Strategy ($/\text{bbl}$) | Buying crude futures and selling refined product futures; executed when refining margins collapse to force economic run cuts. | Confusing a reverse crack financial trade with the physical operating act of turning down distillation columns. |
| **Soybean Crush Spread** | $\text{Crush}$ ($/\text{bushel}$) | Gross margin from crushing 1 bushel ($60\text{ lbs}$) of soybeans into 44 lbs of soybean meal and 11 lbs of soybean oil. | Mismatching units: soybeans are priced in cents/bushel, meal in dollars/short ton, oil in cents/pound. |
| **Oil Share** | $\text{OS}$ ($\%$) | The proportion of total soybean product revenue derived from soybean oil: $\text{OS} = \frac{11 \cdot P_{\text{oil}}}{11 \cdot P_{\text{oil}} + 0.022 \cdot 2000 \cdot P_{\text{meal}}}$. | Treating meal and oil as perfectly correlated co-products; meal is driven by livestock feed, oil by biofuel mandates. |
| **Board Crush vs. Local Crush** | Spread ($/\text{bu}$) | Board crush is the margin calculated using CBOT benchmark futures; local crush incorporates local elevator bean and meal cash basis. | Relying on Board crush profitability while local bean origination basis spikes during regional harvest deficits. |
| **Spark Spread** | $\text{SS}$ ($/\text{MWh}$) | Theoretical gross margin of a gas-fired power plant: electricity price minus fuel cost adjusted for plant heat rate. | Neglecting that heat rate varies non-linearly with turbine load and ambient temperature. |
| **Clean Spark Spread** | $\text{CSS}$ ($/\text{MWh}$) | Spark spread minus the cost of carbon emissions allowances required per MWh generated: $\text{CSS} = \text{SS} - (P_{\text{carbon}} \cdot \text{EF})$. | Omitting grid carbon intensity or startup/shutdown carbon compliance obligations. |
| **Dark Spread / Clean Dark Spread** | $\text{DS / CDS}$ ($/\text{MWh}$) | Theoretical gross margin of a coal-fired power plant minus fuel cost and (for Clean Dark) carbon allowance liabilities. | Overlooking rail transport demurrage and ash disposal fees in operational dark spread dispatching. |
| **Heat Rate** | $\text{HR}$ ($\text{MMBtu/MWh}$ or $\text{kJ/kWh}$) | Measure of thermal conversion efficiency: fuel heat energy required to produce one unit of electrical energy. | Using nameplate design heat rate rather than degraded actual seasonal operating heat rate. |
| **Heat Rate Call Option (HRCO)** | Derivative | A structured contract giving the buyer the right to receive power by paying an agreed heat rate times natural gas price. | Valuing HRCO as a simple financial option without considering physical plant minimum up/down time constraints. |
| **Cost of Carry** | $r + c$ ($\%/\text{year}$) | The sum of capital financing rate ($r$) and physical warehousing/tankage/insurance costs ($c$) required to hold inventory. | Treating storage as a sunk cost when commercial merchant capacity has variable market leasing rates. |
| **Convenience Yield** | $y$ ($\%/\text{year}$) | The implicit benefit or premium accrued by the holder of physical inventory, reflecting operational flexibility and stockout avoidance. | Treating convenience yield as an explicit cash dividend; it is an unobservable, implied shadow value. |
| **Contango** | Regime: $F_{t,T} > S_t$ | Term structure where deferred futures trade at a premium to near-month cash/futures prices ($r + c > y$). | Assuming contango guarantees profits for storage operators without locking in financial calendar spreads. |
| **Backwardation** | Regime: $F_{t,T} < S_t$ | Term structure where near-month physical/futures trade at a premium to deferred contracts ($y > r + c$). | Believing backwardation implies falling prices over time; it reflects high present consumption value. |
| **Roll Yield / Roll Return** | $RY$ ($\% \text{ or } \$$) | Profit or loss generated by closing an expiring near-month derivative contract and opening a deferred contract at a different price. | Ignoring roll drag: long hedgers in contango lose capital each month as expiring futures converge downward to spot. |
| **Minimum-Variance Hedge Ratio** | $h^*$ (dimensionless) | The optimal proportion of physical exposure hedged with financial derivatives to minimize the variance of portfolio cash flows. | Defaulting to a naive $1.0$ hedge ratio ($h=1$), which amplifies variance if basis and futures are negatively correlated. |
| **Hedging Effectiveness** | $R^2$ or $\text{HE}$ ($[0, 1]$) | The percentage reduction in cash flow variance achieved by the hedging strategy compared to the unhedged position. | Assuming a high $R^2$ guarantees positive cash margins; it only guarantees variance reduction around the mean. |
| **Cross-Hedging** | Strategy | Hedging a physical commodity with a futures contract of a different commodity or specification (e.g., jet fuel hedged with heating oil). | Underestimating correlation breakdown during market stress or specification regulatory changes. |
| **Stack-and-Roll** | Strategy | Hedging a multi-year physical commitment by concentrating (stacking) the full volume in liquid front-month futures and rolling them forward. | Catastrophic vulnerability to contango roll drag and massive variation margin calls (e.g., Metallgesellschaft). |
| **Strip Hedge** | Strategy | Hedging multi-period physical exposure by executing separate futures contracts across each matching delivery month in the forward strip. | Illiquidity in deferred months causing severe bid-ask slippage and execution costs. |
| **Variation Margin** | $\Delta \text{VM}$ ($\$$) | Daily cash settlement transferred between clearinghouse and market participants reflecting daily mark-to-market price moves. | Failing to secure committed bank revolving credit facilities to survive prolonged adverse margin runs. |
| **Initial Margin** | $\text{IM}$ ($\$$) | Collateral equity deposited with the clearing broker to open and maintain a leveraged futures position. | Treating IM as a fixed cost; exchanges raise IM requirements drastically during periods of high market volatility. |
| **Credit Support Annex (CSA)** | Legal agreement | Bilateral contract under ISDA documentation establishing collateral thresholds and minimum transfer amounts for OTC derivatives. | Assuming CSA thresholds remain static when credit rating downgrades trigger immediate zero-threshold cash margining. |
| **Bona Fide Hedging** | Regulatory status | Exemption granted by CFTC / regulators allowing commercial entities to exceed speculative position limits for physical risk mitigation. | Assuming any commercial firm qualifies without auditing physical processing capacity and inventory balances. |
| **Negative Pricing** | Regime ($P < 0$) | Phenomenon where producers pay buyers to take physical delivery due to storage exhaustion and shutting down costs (e.g. WTI April 2020). | Hardcoding zero lower bounds ($P \ge 0$) into ERP pricing models and Black-76 option pricing formulas. |
| **Cash Flow Hedge** | Accounting (IFRS 9 / ASC 815) | Accounting designation where unrealized derivative gains/losses are deferred in Other Comprehensive Income (OCI) until physical sale. | Failing prospective effectiveness testing, forcing volatile derivative mark-to-market swings into current quarterly net income. |

---

## 5. The mechanics, step by step

Hedging commodity processing operations requires a sequential, seven-step engineering and econometric workflow.

```text
Step 1: Map Physical Flows & Conversion Stoichiometry
  └── Quantify Mass Balance, Yields, and Net Operational Conversion Cost
Step 2: Deconstruct Local Cash Prices into Benchmark Futures and Basis
  └── S_t = F_t + B_t; Attribute Volatility to Flat Price vs. Regional Basis
Step 3: Model Forward Curve Term Structure & Theory of Storage Dynamics
  └── Calculate Implied Convenience Yield y, Carry Equilibrium, and Brennan-Schwartz Dynamics
Step 4: Econometric Estimation of Minimum-Variance Hedge Ratio (h*)
  └── OLS, Error-Correction Models (ECM), and Dynamic Kalman Filter Updating
Step 5: Synthesize Multi-Leg Transformation Spread Hedges
  └── Formulate Crack, Crush, or Spark Spread Derivative Portfolios and Contract Rounding
Step 6: Execute Contract Rolling & Calendar Spread Management
  └── Quantify Roll Yield, Select Stack vs. Strip Architecture, Manage Contango Drag
Step 7: Size Liquidity Reserves & Establish Hedge Accounting Compliance
  └── Stress-Test Daily Variation Margin (EVT/Cornish-Fisher) and Document IFRS 9 / ASC 815
```

### Step 1: Map Physical Flows and Conversion Stoichiometry

A processor must first translate physical input-output mass balances into gross processing revenue functions.

#### 1. Petroleum Refining: Crack Spread Formulations
Refining transforms crude oil into light ends, middle distillates, and heavy residues. Depending on catalytic configuration:

- **The Standard 3:2:1 Crack Spread:** Reflects a standard fluid catalytic cracking (FCC) refinery configuration where three barrels of crude yield approximately two barrels of motor gasoline and one barrel of ultra-low sulfur diesel (ULSD) or heating oil:
  $$\text{CS}_{3:2:1} = \frac{2 \cdot P_{\text{gasoline}} + 1 \cdot P_{\text{distillate}} - 3 \cdot P_{\text{crude}}}{3}$$
  Because wholesale refined products are quoted on financial exchanges in cents per US gallon ($\text{US}\ \phi/\text{gal}$) while crude oil is quoted in US dollars per barrel ($\$/\text{bbl}$), and one standard barrel equals exactly 42 US gallons:
  $$P_{\text{gasoline}} (\$/\text{bbl}) = P_{\text{RBOB}} (\$/\text{gal}) \cdot 42$$
  $$P_{\text{distillate}} (\$/\text{bbl}) = P_{\text{ULSD}} (\$/\text{gal}) \cdot 42$$
  Thus, the 3:2:1 dollar crack spread per barrel is:
  $$\text{CS}_{3:2:1} = \frac{2 \cdot (42 \cdot P_{\text{RBOB}}) + 1 \cdot (42 \cdot P_{\text{ULSD}}) - 3 \cdot P_{\text{crude}}}{3} = 28 \cdot P_{\text{RBOB}} + 14 \cdot P_{\text{ULSD}} - P_{\text{crude}}$$

- **The High-Gasoline 5:3:2 Crack Spread:** Used for refineries with extensive hydrocracking and alkylation units maximizing gasoline yield:
  $$\text{CS}_{5:3:2} = \frac{3 \cdot (42 \cdot P_{\text{RBOB}}) + 2 \cdot (42 \cdot P_{\text{ULSD}}) - 5 \cdot P_{\text{crude}}}{5} = 25.2 \cdot P_{\text{RBOB}} + 16.8 \cdot P_{\text{ULSD}} - P_{\text{crude}}$$

- **The Reverse Crack Spread:** When the market crack spread drops below the cash variable operating cost ($\text{OPEX}_{\text{var}}$), the refiner executes a reverse crack: buying crude futures and selling refined product futures while throttling unit throughput. The economic run-cut condition is:
  $$\text{CS}_{\text{market}} < \text{OPEX}_{\text{var}} + \text{Cost}_{\text{turndown}}$$

#### 2. Agribusiness: The Soybean Crush Spread and Oil Share
Soybean processing plants crush raw soybeans into soybean meal (high-protein animal feed) and crude soybean oil (edible oil, renewable diesel feedstock). One standard bushel of soybeans weighs $60\text{ lbs}$ ($0.02722\text{ metric tons}$). The standard commercial yield assumption is:
- $44\text{ lbs}$ of soybean meal ($73.33\%$ mass yield).
- $11\text{ lbs}$ of soybean oil ($18.33\%$ mass yield).
- $5\text{ lbs}$ of moisture, hulls, and processing loss ($8.34\%$ shrinkage).

On the Chicago Board of Trade (CBOT):
- Soybeans are quoted in cents per bushel ($\phi/\text{bu}$).
- Soybean meal is quoted in dollars per short ton ($\$/\text{short ton}$, where $1\text{ short ton} = 2,000\text{ lbs}$).
- Soybean oil is quoted in cents per pound ($\phi/\text{lb}$).

The gross crush margin in dollars per bushel ($\$/\text{bu}$) is:
$$\text{Crush} (\$/\text{bu}) = \left[ P_{\text{meal}} (\$/\text{ton}) \cdot \left(\frac{44\text{ lbs}}{2,000\text{ lbs}}\right) \right] + \left[ \frac{P_{\text{oil}} (\phi/\text{lb})}{100} \cdot 11\text{ lbs} \right] - \left[ \frac{P_{\text{beans}} (\phi/\text{bu})}{100} \right]$$
$$\text{Crush} (\$/\text{bu}) = 0.022 \cdot P_{\text{meal}} + 0.11 \cdot P_{\text{oil}} - \frac{P_{\text{beans}}}{100}$$

The **Oil Share** ($\text{OS}$) quantifies the percentage contribution of oil to gross product revenue:
$$\text{OS} = \frac{0.11 \cdot P_{\text{oil}}}{0.022 \cdot P_{\text{meal}} + 0.11 \cdot P_{\text{oil}}}$$
When biofuel mandates (e.g., US EPA Renewable Fuel Standard / California LCFS) expand, oil share surges from a baseline of $30\% - 35\%$ to over $45\% - 50\%$, decoupling soybean crush dynamics from livestock feed meal demand.

#### 3. Power Generation: Spark and Dark Spreads
A combined-cycle gas turbine (CCGT) power station converts natural gas into electric power. The efficiency is measured by the plant Heat Rate ($\text{HR}$), defined as the British thermal units ($\text{Btu}$) of fuel required to generate one kilowatt-hour ($\text{kWh}$), or million Btu ($\text{MMBtu}$) per megawatt-hour ($\text{MWh}$). 

The standard **Spark Spread** ($\text{SS}$) per MWh is:
$$\text{SS} = P_{\text{power}} - (\text{HR} \cdot P_{\text{gas}})$$
where $P_{\text{power}}$ is in $\$ /\text{MWh}$, $P_{\text{gas}}$ is in $\$ /\text{MMBtu}$, and $\text{HR}$ is in $\text{MMBtu}/\text{MWh}$. A modern CCGT has an operational heat rate between $6.5$ and $7.2\ \text{MMBtu}/\text{MWh}$ (thermal efficiency $\eta \approx 47\% - 53\%$). An older peaker unit might have $\text{HR} \approx 9.5 - 11.0\ \text{MMBtu}/\text{MWh}$.

Under emissions compliance regimes (such as the EU Emissions Trading System or California Cap-and-Trade), generators must surrender carbon allowances (EUAs or CCAs) for each metric ton of $\text{CO}_2$ emitted. The **Clean Spark Spread** ($\text{CSS}$) accounts for this marginal carbon penalty:
$$\text{CSS} = P_{\text{power}} - (\text{HR} \cdot P_{\text{gas}}) - (\text{EF}_{\text{gas}} \cdot P_{\text{carbon}})$$
where $\text{EF}_{\text{gas}}$ is the emissions factor ($\approx 0.053\ \text{tCO}_2/\text{MMBtu} \times \text{HR}$, yielding $\approx 0.35 - 0.40\ \text{tCO}_2/\text{MWh}$ for a modern CCGT) and $P_{\text{carbon}}$ is the carbon price ($\$/\text{tCO}_2$).

For a coal-fired unit, the equivalent metric is the **Dark Spread** and **Clean Dark Spread**, incorporating coal heat content, freight rail tariffs, and an emissions factor approximately double that of natural gas ($\approx 0.85 - 1.05\ \text{tCO}_2/\text{MWh}$).

---

### Step 2: Deconstruct Local Cash Prices into Benchmark Futures and Basis

Physical procurement and commercial off-take never settle against abstract exchange benchmark indices. Every local cash price $S_t$ paid or received at a physical asset is decomposed as:
$$S_t = F_t + B_t$$
where $F_t$ is the front-month benchmark futures price and $B_t$ is the physical basis. The physical basis is further partitioned into structural engineering and economic components:
$$B_t = \Delta_{\text{location}} + \Delta_{\text{quality}} + \Delta_{\text{logistics}} + \epsilon_{\text{local},t}$$

1. **Location Differential ($\Delta_{\text{location}}$):** The spatial transportation cost (pipeline tariff, rail freight, marine tanker charter, barge demurrage) between the physical asset and the exchange delivery hub. If Henry Hub natural gas is $\$3.00/\text{MMBtu}$ in Louisiana, but pipeline capacity into the Northeast US is constrained during a winter blizzard, Transco Zone 6 cash gas can trade at $\$25.00/\text{MMBtu}$ (a basis of $+\$22.00$).
2. **Quality Differential ($\Delta_{\text{quality}}$):** Adjustment for physical assays diverging from exchange benchmark specifications. In crude oil, West Texas Intermediate (WTI) benchmark specifies $37^\circ - 42^\circ\ \text{API}$ gravity and $<0.42\%$ sulfur. If a refinery processes Western Canadian Select (WCS)—a heavy, sour crude ($20.5^\circ\ \text{API}$, $3.5\%$ sulfur)—WCS trades at a structural discount to WTI ($-\$12$ to $-\$25/\text{bbl}$) reflecting the downstream coking and hydrotreating capacity required to refine it.
3. **Logistics & Terminal Differential ($\Delta_{\text{logistics}}$):** Throughput fees, tank strapping charges, blending charges, and loading/unloading rack margins.
4. **Local Supply-Demand Residual ($\epsilon_{\text{local},t}$):** Transitory local imbalances, such as regional grain elevator saturation during peak harvest or local refinery turnarounds.

#### Basis Risk Variance Decomposition
Consider a hedged physical position where a firm owns physical inventory $S_t$ and sells $h$ units of futures $F_t$. The change in the value of the hedged portfolio over holding period $\Delta t$ is:
$$\Delta \Pi = \Delta S_t - h \Delta F_t = (\Delta F_t + \Delta B_t) - h \Delta F_t = (1 - h) \Delta F_t + \Delta B_t$$
The variance of this hedged portfolio is:
$$\text{Var}(\Delta \Pi) = (1 - h)^2 \sigma_F^2 + \sigma_B^2 + 2(1 - h) \text{Cov}(\Delta F, \Delta B)$$
If the processor executes a naive $1:1$ hedge ($h = 1$):
$$\text{Var}(\Delta \Pi \mid h=1) = \sigma_B^2$$
This proves a fundamental axiom of commodity risk management: **A futures hedge does not eliminate price risk; it transforms systemic flat price risk ($\sigma_F^2$) into local basis risk ($\sigma_B^2$).** If basis volatility is large relative to flat price moves (common in inland oil basins, regional power nodes, or localized grain markets), a naive hedge can destabilize cash flows.

---

### Step 3: Model Forward Curve Term Structure and Theory of Storage Dynamics

Forward and futures prices do not predict the future spot price; they reflect the intertemporal equilibrium of physical inventory holding. The relationship between spot price $S_t$ and the forward price for delivery at $T$, $F_{t,T}$, is defined by the **Theory of Storage**:
$$F_{t,T} = S_t e^{(r + c - y)(T - t)}$$
where:
- $r$: Continuous annualized risk-free interest rate / cost of capital.
- $c$: Continuous annualized cost of physical storage, tankage leasing, ullage insurance, and degradation.
- $y$: Continuous annualized marginal convenience yield.

```text
Forward Price F(T)
     │
     │      CONTANGO CURVE (r + c > y)
     │      - Abundant inventory; tanks filling
     │      - Forward prices trade at a premium
     │      - F_T = S * e^((r+c-y)*T) > S
     │                             . - ~ ~ ~ F(T)
     │                   . - ~ ~ ~
     │         . - ~ ~ ~
     │S  - - - - - - - - - - - - - - - - - - - - - - - - -
     │         ` - _ _ _
     │                   ` - _ _ _
     │                             ` - _ _ _ F(T)
     │      BACKWARDATED CURVE (y > r + c)
     │      - Acute immediate physical shortage; stockout threat
     │      - Near-term cash trades at a premium; convenience yield spikes
     └───────────────────────────────────────────────────────► Maturity (T)
```

#### Regimes and the Implied Convenience Yield
1. **Full Carry / Contango ($y \approx 0 \implies F_{t,T} > S_t$):** When commercial stocks are plentiful and storage ullage is available, holding physical inventory provides minimal operational convenience. The forward curve trades at **full cost of carry**:
   $$F_{t,T}^{\text{full carry}} = S_t e^{(r + c)(T - t)}$$
   In contango, the market pays market participants to store the commodity. Physical operators fill tanks and sell deferred futures, locking in an arbitrage profit ($F_{t,T} - S_t > \text{carrying costs}$).
2. **Backwardation ($y > r + c \implies F_{t,T} < S_t$):** When current inventories are depleted, processors face catastrophic downtime if they run out of feedstock. The marginal convenience yield $y$ surges far above carrying costs:
   $$y(I_t) \to \infty \quad \text{as} \quad I_t \to I_{\text{tank bottoms}}$$
   Forward prices trade at a deep discount to prompt cash prices. The market penalizes storage and economically forces physical inventories out of tanks and into immediate consumption.

Solving for the annualized implied convenience yield from observed market quotes:
$$y = r + c - \frac{1}{T - t} \ln\left(\frac{F_{t,T}}{S_t}\right)$$

#### The Brennan-Schwartz Two-Factor Model
To capture forward curve term dynamics stochastically, Brennan and Schwartz formulated the joint diffusion of spot price $S_t$ and convenience yield $y_t$:
$$\frac{dS_t}{S_t} = (\mu - y_t) dt + \sigma_1 dz_{1,t}$$
$$dy_t = \kappa(\alpha - y_t) dt + \sigma_2 dz_{2,t}$$
$$\mathbb{E}[dz_{1,t} dz_{2,t}] = \rho_{Sy} dt$$
where $\kappa$ is the speed of mean reversion of the convenience yield toward its long-run mean $\alpha$, $\sigma_1$ is spot volatility, and $\sigma_2$ is convenience yield volatility. In periods of structural shortage, positive shocks to spot price correlate strongly with convenience yield spikes ($\rho_{Sy} > 0.70$), inverting the front of the curve while deferred maturities remain stable.

#### The Samuelson Effect
In commodity forward curves, the volatility of futures prices is not constant across maturities. The volatility of near-month contracts is significantly higher than deferred contracts, decaying exponentially as maturity lengthens:
$$\sigma_F(T - t) = \sigma_0 e^{-\alpha (T - t)}$$
where $\sigma_0$ is spot price volatility and $\alpha > 0$ is the mean-reversion rate. Near-month contracts absorb prompt physical shocks (weather, unplanned refinery outages), while long-dated contracts are anchored by long-run marginal extraction and production costs.

#### Forward Curve Curvature and Calendar Spread Butterflies
Forward curves rarely shift in parallel. They exhibit steepening, flattening, and localized curvature changes driven by inventory congestion in specific delivery months. To decompose and hedge non-parallel term structure risk, commercial desks construct **calendar spread butterflies**:
$$\text{Fly}_t = F_{t,M_1} - 2 F_{t,M_2} + F_{t,M_3}$$
where $M_1, M_2, M_3$ represent three consecutive delivery tenors (e.g., Oct, Nov, Dec). 
- When $\text{Fly}_t > 0$, the forward curve is locally convex (the middle month is depressed relative to wings, common during temporary regional refinery turnaround pauses).
- When $\text{Fly}_t < 0$, the curve is locally concave (the middle month is inverted relative to wings, reflecting an acute prompt delivery bottleneck).
Trading or hedging the butterfly allows process operators to insulate their rolling programs against curvature distortion without taking directional flat price risk.

---

### Step 4: Econometric Estimation of the Minimum-Variance Hedge Ratio

To minimize portfolio variance rather than naively matching physical volume, the processor must compute the **minimum-variance hedge ratio** ($h^*$).

#### Analytical Derivation
Let $\Delta S_t = S_t - S_{t-1}$ be the change in physical cash price, and $\Delta F_t = F_t - F_{t-1}$ be the change in benchmark futures price. For a physical long position (e.g., holding inventory) hedged by selling $h$ units of futures:
$$\Delta \Pi_t = \Delta S_t - h \Delta F_t$$
The variance of portfolio value change is:
$$\sigma_{\Pi}^2 = \text{Var}(\Delta \Pi_t) = \sigma_{\Delta S}^2 + h^2 \sigma_{\Delta F}^2 - 2 h \text{Cov}(\Delta S, \Delta F)$$
To find the variance-minimizing hedge ratio, differentiate $\sigma_{\Pi}^2$ with respect to $h$ and set to zero:
$$\frac{\partial \sigma_{\Pi}^2}{\partial h} = 2 h \sigma_{\Delta F}^2 - 2 \text{Cov}(\Delta S, \Delta F) = 0$$
$$h^* = \frac{\text{Cov}(\Delta S, \Delta F)}{\text{Var}(\Delta F)} = \rho_{SF} \frac{\sigma_{\Delta S}}{\sigma_{\Delta F}}$$
where $\rho_{SF}$ is the Pearson correlation coefficient between cash and futures changes, $\sigma_{\Delta S}$ is the standard deviation of cash price changes, and $\sigma_{\Delta F}$ is the standard deviation of futures price changes.

#### Hedging Effectiveness ($R^2$)
Substituting $h^*$ back into the variance equation:
$$\sigma_{\Pi}^{*2} = \sigma_{\Delta S}^2 + \left(\rho_{SF} \frac{\sigma_{\Delta S}}{\sigma_{\Delta F}}\right)^2 \sigma_{\Delta F}^2 - 2 \left(\rho_{SF} \frac{\sigma_{\Delta S}}{\sigma_{\Delta F}}\right) (\rho_{SF} \sigma_{\Delta S} \sigma_{\Delta F}) = \sigma_{\Delta S}^2 (1 - \rho_{SF}^2)$$
The hedging effectiveness ($\text{HE}$), representing the proportion of cash price variance eliminated, is mathematically identical to the coefficient of determination ($R^2$) of the regression:
$$\text{HE} = \frac{\sigma_{\Delta S}^2 - \sigma_{\Pi}^{*2}}{\sigma_{\Delta S}^2} = \rho_{SF}^2 = R^2$$

#### Cointegration and the Error Correction Model (ECM)
Estimating $h^*$ by running simple OLS on first differences ($\Delta S_t = \alpha + h \Delta F_t + \epsilon_t$) assumes the relationship is memoryless. However, physical cash and futures series are typically integrated of order 1 ($I(1)$) and cointegrated in levels:
$$S_t = \beta_0 + \beta_1 F_t + z_t$$
where $z_t \sim I(0)$ is the stationary equilibrium basis. Standard OLS on first differences ignores the long-run cointegrating attractor, leading to hedge ratios that drift. Under the Engle-Granger representation theorem, the true dynamic hedge must be estimated using an **Error Correction Model (ECM)**:
$$\Delta S_t = \gamma_0 + h_{\text{ECM}} \Delta F_t + \lambda z_{t-1} + \sum_{i=1}^p \phi_i \Delta S_{t-i} + \sum_{j=1}^q \theta_j \Delta F_{t-j} + u_t$$
where $\lambda < 0$ is the error-correction speed of adjustment parameter and $z_{t-1} = S_{t-1} - \hat{\beta}_0 - \hat{\beta}_1 F_{t-1}$ is the lagged basis equilibrium error. The ECM hedge ratio $h_{\text{ECM}}$ prevents over-hedging when basis diverges temporarily from its long-run spatial equilibrium.

#### Dynamic Time-Varying Hedge Ratios via the Kalman Filter
When the underlying relationship between cash and futures prices undergoes continuous structural drift (e.g. seasonal pipeline tariff adjustments or progressive crude assay degradation), static OLS fails. The dynamic hedge ratio is modeled as an unobserved state vector via the **Kalman Filter**:
- **Measurement Equation:** $\Delta S_t = h_t \Delta F_t + \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, \sigma_{\epsilon}^2)$
- **State Transition Equation:** $h_t = h_{t-1} + \eta_t, \quad \eta_t \sim \mathcal{N}(0, \sigma_{\eta}^2)$

The updated state estimate and error variance at time $t$ are given by:
$$K_t = \frac{P_{t|t-1} \Delta F_t}{(\Delta F_t)^2 P_{t|t-1} + \sigma_{\epsilon}^2}$$
$$\hat{h}_t = \hat{h}_{t-1} + K_t \left( \Delta S_t - \hat{h}_{t-1} \Delta F_t \right)$$
$$P_{t|t} = (1 - K_t \Delta F_t) P_{t|t-1}$$
where $K_t$ is the Kalman gain, ensuring the hedge ratio dynamically adapts to real-time basis shifts without over-reacting to high-frequency observation noise.

---

### Step 5: Multi-Leg Transformation Spread Hedging

Processing plants must hedge multiple co-products simultaneously against a single feedstock (or multiple feedstocks against a single product).

#### Matrix Formulation for Multi-Asset Hedging
Let $\mathbf{S}_t = [S_{\text{in},t}, S_{\text{out},1,t}, \dots, S_{\text{out},K,t}]^T$ be the vector of physical spot prices, and $\mathbf{w} = [-1, w_1, \dots, w_K]^T$ be the physical yield vector per unit of feedstock processed. The physical gross processing margin change is:
$$\Delta M_t = \mathbf{w}^T \Delta \mathbf{S}_t$$
Let $\mathbf{F}_t = [F_{1,t}, \dots, F_{M,t}]^T$ be the vector of liquid exchange-traded futures contracts available for hedging, and $\mathbf{h} = [h_1, \dots, h_M]^T$ be the hedge position vector. The net hedged margin change is:
$$\Delta \Pi_M = \Delta M_t - \mathbf{h}^T \Delta \mathbf{F}_t = \mathbf{w}^T \Delta \mathbf{S}_t - \mathbf{h}^T \Delta \mathbf{F}_t$$
The variance of the hedged conversion margin is:
$$\text{Var}(\Delta \Pi_M) = \mathbf{w}^T \boldsymbol{\Sigma}_{SS} \mathbf{w} + \mathbf{h}^T \boldsymbol{\Sigma}_{FF} \mathbf{h} - 2 \mathbf{w}^T \boldsymbol{\Sigma}_{SF} \mathbf{h}$$
where:
- $\boldsymbol{\Sigma}_{SS}$ is the $(K+1) \times (K+1)$ covariance matrix of physical spot price changes.
- $\boldsymbol{\Sigma}_{FF}$ is the $M \times M$ covariance matrix of futures price changes.
- $\boldsymbol{\Sigma}_{SF}$ is the $(K+1) \times M$ cross-covariance matrix between physical and futures changes.

Minimizing variance with respect to vector $\mathbf{h}$ yields the optimal multivariate hedge vector:
$$\frac{\partial \text{Var}(\Delta \Pi_M)}{\partial \mathbf{h}} = 2 \boldsymbol{\Sigma}_{FF} \mathbf{h} - 2 \boldsymbol{\Sigma}_{SF}^T \mathbf{w} = \mathbf{0}$$
$$\mathbf{h}^* = \boldsymbol{\Sigma}_{FF}^{-1} \boldsymbol{\Sigma}_{SF}^T \mathbf{w}$$

#### Sizing Contract Lots: The Integer Rounding Constraint
Futures contracts trade in standardized lot sizes ($Q_{\text{contract}}$). The required integer number of contracts $N_m$ for leg $m$ is:
$$N_m = \text{round}\left( \frac{h_m^* \cdot V_{\text{physical},m}}{Q_{\text{contract},m}} \right)$$
- Crude Oil (NYMEX CL): $1,000\ \text{barrels}$.
- RBOB Gasoline (NYMEX RB): $42,000\ \text{gallons}$ ($1,000\ \text{barrels}$).
- ULSD Diesel (NYMEX HO): $42,000\ \text{gallons}$ ($1,000\ \text{barrels}$).
- Corn / Soybeans (CBOT C / S): $5,000\ \text{bushels}$.
- Soybean Meal (CBOT SM): $100\ \text{short tons}$ ($200,000\ \text{lbs}$).
- Soybean Oil (CBOT BO): $60,000\ \text{lbs}$.
- Natural Gas (NYMEX NG): $10,000\ \text{MMBtu}$.
- PJM Western Hub Electricity (ICE): $50\ \text{MWh}$ (peak or off-peak blocks).

#### Proxy Hedging of Unquoted Resid Streams
Commercial petroleum refineries and chemical plants rarely produce only clean, exchange-traded co-products. A coking refinery produces significant volumes of high-sulfur fuel oil (HSFO), asphalt, vacuum gas oil (VGO), and petroleum coke, none of which possess direct liquid futures contracts on NYMEX. 
To manage the full molecular barrel, risk desks execute **proxy cross-hedges**:
$$\Delta S_{\text{resid},t} = \beta_1 \Delta F_{\text{crude},t} + \beta_2 \Delta F_{\text{diesel},t} + \epsilon_{\text{resid},t}$$
Because the residual stream has a high sulfur penalty and declining marine demand (post-IMO 2020), $\beta_1$ and $\beta_2$ exhibit structural instability during environmental regulatory transitions. The unexplained residual $\text{Var}(\epsilon_{\text{resid}})$ represents **molecular basis risk**, which cannot be diversified away through centralized exchange derivatives and must be managed via bilateral OTC physical supply contracts with industrial off-takers.

#### Integer Allocation via Mixed-Integer Quadratic Programming (MIQP)
Standard naive rounding ($N_m = \text{round}(\dots)$) can violate stoichiometric conversion ratios, generating unintended synthetic directional exposures. When operating at industrial scale, the optimal integer contract vector $\mathbf{N} = [N_1, \dots, N_M]^T \in \mathbb{Z}^M$ is solved as a **Mixed-Integer Quadratic Program (MIQP)**:
$$\min_{\mathbf{N} \in \mathbb{Z}^M} \left( \mathbf{w}^T \Delta \mathbf{S} - \mathbf{N}^T \mathbf{Q} \Delta \mathbf{F} \right)^T \boldsymbol{\Sigma} \left( \mathbf{w}^T \Delta \mathbf{S} - \mathbf{N}^T \mathbf{Q} \Delta \mathbf{F} \right) + \sum_{m=1}^M c_m |N_m|$$
$$\text{subject to} \quad \left| \frac{N_{\text{product}}}{N_{\text{feedstock}}} - \text{Ratio}_{\text{stoich}} \right| \le \delta_{\text{tol}}$$
where $\mathbf{Q} = \text{diag}(Q_1, \dots, Q_M)$ scales standardized exchange lot sizes, $c_m$ is the exchange execution and clearing commission per contract, and $\delta_{\text{tol}}$ bounds stoichiometric imbalance.

---

### Step 6: Contract Rolling, Term Structure Arbitrage, and Roll Yield

Physical facilities operate continuously for years, but commodity futures markets concentrate liquidity in prompt months (tenors $< 3$ to $6$ months). A processor hedging twelve months of future production cannot execute illiquid one-year forward contracts without incurring massive market-impact costs. They must employ dynamic **contract rolling**.

#### The Mechanics of Rolling
Prior to first notice day or contract expiry, the hedger liquidates the maturing front-month contract ($M_1$) at price $F_{t,M_1}$ and simultaneously opens an identical position in the next liquid deferred contract ($M_2$) at price $F_{t,M_2}$.

The arithmetic price differential between the two contracts at the moment of the roll is the **calendar spread**:
$$\text{Spread}_{\text{roll}} = F_{t,M_1} - F_{t,M_2}$$

```text
=============================================================================================================
                          ROLL YIELD DYNAMICS UNDER REGIME SHIFTS
=============================================================================================================
Regime: BACKWARDATION (F_M1 > F_M2)                 Regime: CONTANGO (F_M1 < F_M2)
Curve slope: Downward sloping                        Curve slope: Upward sloping

SHORT HEDGER (Producer / Seller):                   SHORT HEDGER (Producer / Seller):
1. Buys back expiring M1 at lower price              1. Buys back expiring M1 at spot/lower price
2. Sells deferred M2 at higher price                 2. Sells deferred M2 at higher price
Result: POSITIVE ROLL YIELD (Roll Gain)              Result: NEGATIVE ROLL YIELD (Roll Drag)
  Buys low, sells high on calendar spread              Locked into constant roll attrition

LONG HEDGER (Consumer / Processor Feedstock):       LONG HEDGER (Consumer / Processor Feedstock):
1. Sells expiring M1 at higher price                 1. Sells expiring M1 at lower price
2. Buys deferred M2 at lower price                   2. Buys deferred M2 at higher price
Result: POSITIVE ROLL YIELD (Roll Gain)              Result: NEGATIVE ROLL YIELD (Roll Drag)
  Rolls into cheaper feedstock tenor                   Rolls into expensive forward premiums
=============================================================================================================
```

#### Quantifying Roll Return
The percentage roll return over the roll window is defined as:
$$RY = \frac{F_{t,M_1} - F_{t,M_2}}{F_{t,M_1}}$$
- If the market is in **backwardation** ($F_{t,M_1} > F_{t,M_2}$), $RY > 0$. The forward curve converges upward toward the spot price as expiration approaches, generating structural roll gains.
- If the market is in **contango** ($F_{t,M_1} < F_{t,M_2}$), $RY < 0$. The forward price converges downward toward the spot price, generating continuous **roll drag**.

#### Stack-and-Roll vs. Strip Hedging
- **Stack-and-Roll:** The hedger places the entire multi-period volume into the liquid prompt month. Each month, the entire remaining "stack" is rolled into the next month.
  - *Advantage:* Maximum liquidity, tight bid-ask spreads, minimal execution slippage.
  - *Catastrophic Risk:* If the market enters prolonged contango, the hedger bleeds negative roll yield on the entire remaining volume each month. If spot prices plunge, daily variation margin calls drain liquidity.
- **Strip Hedge:** The hedger buys/sells separate individual contracts across each future delivery month ($M_1, M_2, \dots, M_{12}$) matching the physical production schedule.
  - *Advantage:* Locks in the exact forward curve shape at inception; zero roll risk and zero roll drag.
  - *Disadvantage:* Deferred months exhibit low trading volumes, wide bid-ask spreads, and significant market-impact costs.

---

### Step 7: Post-Hedge Liquidity, Margin Sizing, and Accounting Compliance

Executing financial derivatives introduces credit and cash liquidity requirements governed by clearinghouse rules.

#### Margin Sizing and the Liquidity Buffer
1. **Initial Margin (IM):** Upfront cash or treasury collateral posted per contract:
   $$\text{IM}_{\text{total}} = \sum_{m=1}^M |N_m| \cdot \text{IM}_{\text{unit},m}$$
   Typically $4\% - 10\%$ of total notional value during calm markets, but exchanges can double or triple IM overnight during crises.
2. **Variation Margin (VM):** Daily cash transfer reflecting mark-to-market changes:
   $$\Delta \text{VM}_t = \sum_{m=1}^M N_m \cdot Q_{\text{contract},m} \cdot (F_{m,t} - F_{m,t-1})$$
   If the physical company has shorted futures to hedge physical output and market prices surge by $30\%$, the exchange demands instant cash settlement at 08:00 the following morning. Even though the company's physical inventory is worth $30\%$ more, that physical cash cannot be realized until the goods are processed, shipped, invoiced, and collected (a 30- to 90-day cash conversion cycle).

To prevent illiquidity default, the treasury team must size a committed standby bank liquidity facility using **Extreme Value Theory (EVT)** or the **Cornish-Fisher expansion** for fat-tailed returns:
$$\text{Liquidity Buffer} \ge \text{VaR}_{1-\alpha}(\Delta \text{VM}, \tau) = \sum_{m=1}^M |N_m| \cdot Q_{\text{contract},m} \cdot F_{m,0} \cdot \left( z_{\text{CF}} \sigma_m \sqrt{\tau} \right)$$
$$z_{\text{CF}} = z_{1-\alpha} + \frac{k - 3}{6} (z_{1-\alpha}^2 - 1)$$
where $\tau$ is the liquidation lead time (typically 5 to 10 trading days), $z_{1-\alpha}$ is the normal quantile, and $k$ is the empirical kurtosis of price returns to account for severe tail jumps.

#### Hedge Accounting: IFRS 9 / ASC 815 Qualification
Without formal hedge accounting, mark-to-market fluctuations on derivatives must be recognized immediately in current-period Profit & Loss (P&L), while the offsetting physical sales are recognized in future quarters when delivered. This creates massive artificial quarterly earnings volatility.

To qualify for **Cash Flow Hedge Accounting**:
1. **Formal Designation at Inception:** Document the exact physical exposure, hedging instrument, risk objective, and designated risk component (e.g., benchmark flat price).
2. **Prospective Effectiveness Testing:** Prove that an economic relationship exists between the hedged item and the derivative ($h^*$ supported by regression or ECM with $R^2 \ge 0.80$).
3. **Credit Risk Dominance:** Establish that credit risk does not dominate the economic relationship.
4. **Hedge Ratio Alignment:** The hedge ratio must match the physical operational volume processed.
When qualified, the effective portion of derivative mark-to-market gains/losses is booked directly to **Other Comprehensive Income (OCI)** on the balance sheet, bypassing P&L until the physical commodity is delivered and settled.

---

## 6. Worked example

### Operational Profile: The Mid-Continent Refinery Crack Spread Hedge

A merchant refinery in Kansas processes **$50,000\ \text{barrels per day (bpd)}$** of crude oil. The management committee wants to lock in the refining margin for a **$30\text{-day operating campaign}$** scheduled two months forward (Total campaign volume: $1,500,000\ \text{barrels}$ of crude oil).

#### 1. Yield and Mass Balance Specifications
The refinery's catalytic cracking configuration operates on a strict **3:2:1 volumetric yield**:
- Feedstock input: $1,500,000\ \text{barrels}$ of crude oil ($50,000\ \text{bpd} \times 30\ \text{days}$).
- Output co-products:
  - Gasoline (RBOB): $\frac{2}{3} \times 1,500,000 = 1,000,000\ \text{barrels}$ ($42,000,000\ \text{gallons}$).
  - Distillate (ULSD Diesel): $\frac{1}{3} \times 1,500,000 = 500,000\ \text{barrels}$ ($21,000,000\ \text{gallons}$).
- Internal refinery fuel gas consumption and catalytic loss are accounted for as an operating utility expense ($OPEX = \$4.50/\text{bbl}$ of crude processed).

#### 2. Market Pricing and Basis Assays at Inception ($t_0$)
- **Benchmark Futures Quotes (NYMEX Month 2 Forward):**
  - Light Sweet Crude Oil (WTI): $F_{\text{crude}} = \$78.00/\text{bbl}$.
  - RBOB Gasoline: $F_{\text{gasoline}} = \$2.5000/\text{gal} \implies \$2.5000 \times 42 = \$105.00/\text{bbl}$.
  - ULSD Distillate: $F_{\text{distillate}} = \$2.8000/\text{gal} \implies \$2.8000 \times 42 = \$117.60/\text{bbl}$.
- **Local Physical Basis Quotes (Expected at Delivery $t_1$):**
  - Local crude delivered at plant gate: Midland-quality sweet crude commanding a pipeline transport tariff basis of $B_{\text{crude}} = +\$2.20/\text{bbl}$ relative to Cushing WTI.
  - Local wholesale gasoline rack: $B_{\text{gasoline}} = -\$0.0350/\text{gal} \implies -\$1.47/\text{bbl}$ relative to NYMEX RBOB.
  - Local wholesale diesel rack: $B_{\text{distillate}} = +\$0.0450/\text{gal} \implies +\$1.89/\text{bbl}$ relative to NYMEX ULSD.

#### 3. Econometric Estimation of Hedge Ratios
Using 250 daily observations of local cash price changes versus NYMEX futures changes, the quantitative risk team estimated the covariance parameters:

| Parameter | Crude Oil (CL) | Gasoline (RB) | Distillate (HO) |
|---|---|---|---|
| Daily Futures Volatility ($\sigma_{\Delta F}$) | $\$1.40/\text{bbl}$ | $\$0.0380/\text{gal}$ ($\$1.596/\text{bbl}$) | $\$0.0420/\text{gal}$ ($\$1.764/\text{bbl}$) |
| Daily Local Cash Volatility ($\sigma_{\Delta S}$) | $\$1.48/\text{bbl}$ | $\$0.0365/\text{gal}$ ($\$1.533/\text{bbl}$) | $\$0.0410/\text{gal}$ ($\$1.722/\text{bbl}$) |
| Correlation ($\rho_{SF}$) | $0.960$ | $0.945$ | $0.955$ |
| Optimal Hedge Ratio ($h^* = \rho \frac{\sigma_S}{\sigma_F}$) | $0.960 \times \frac{1.48}{1.40} = \mathbf{1.015}$ | $0.945 \times \frac{0.0365}{0.0380} = \mathbf{0.908}$ | $0.955 \times \frac{0.0410}{0.0420} = \mathbf{0.932}$ |
| Hedging Effectiveness ($R^2 = \rho^2$) | $0.9216$ ($92.2\%$) | $0.8930$ ($89.3\%$) | $0.9120$ ($91.2\%$) |

#### 4. Baseline Margin Calculation (Theoretical Benchmark vs. Local Physical)
- **Benchmark NYMEX 3:2:1 Crack Spread:**
  $$\text{CS}_{3:2:1}^{\text{bench}} = \frac{2 \cdot 105.00 + 1 \cdot 117.60 - 3 \cdot 78.00}{3} = \frac{210.00 + 117.60 - 234.00}{3} = \frac{\$93.60}{3} = \$31.20/\text{bbl}$$
- **Expected Local Physical Gross Refining Margin (GRM):**
  - Physical crude cost delivered: $S_{\text{crude}} = \$78.00 + \$2.20 = \$80.20/\text{bbl}$.
  - Physical gasoline netback: $S_{\text{gas}} = \$105.00 - \$1.47 = \$103.53/\text{bbl}$.
  - Physical diesel netback: $S_{\text{diesel}} = \$117.60 + \$1.89 = \$119.49/\text{bbl}$.
  $$\text{GRM}_{\text{physical}} = \frac{2 \cdot 103.53 + 1 \cdot 119.49 - 3 \cdot 80.20}{3} = \frac{207.06 + 119.49 - 240.60}{3} = \frac{\$85.95}{3} = \$28.65/\text{bbl}$$
  - Net operating margin after plant OPEX ($\$4.50/\text{bbl}$):
    $$\text{Margin}_{\text{net}} = \$28.65 - \$4.50 = \mathbf{\$24.15/\text{bbl}} \implies \text{Total Campaign Profit} = \$24.15 \times 1,500,000 = \mathbf{\$36,225,000}$$

#### 5. Sizing the Derivative Hedge Portfolio
The refinery needs to hedge against feedstock price increases (buy crude futures) and product price drops (sell gasoline and diesel futures).

- **Crude Oil (Feedstock Long Futures Position):**
  $$V_{\text{target, crude}} = h_{\text{crude}}^* \times 1,500,000\ \text{bbl} = 1.015 \times 1,500,000 = 1,522,500\ \text{bbl}$$
  $$\text{Contracts to Buy} = \frac{1,522,500}{1,000\ \text{bbl/contract}} = 1,522.5 \implies \mathbf{1,523\ \text{contracts (Long CL)}}$$
- **RBOB Gasoline (Product Short Futures Position):**
  $$V_{\text{target, gas}} = h_{\text{gas}}^* \times 42,000,000\ \text{gal} = 0.908 \times 42,000,000 = 38,136,000\ \text{gal}$$
  $$\text{Contracts to Sell} = \frac{38,136,000}{42,000\ \text{gal/contract}} = 908.0 \implies \mathbf{908\ \text{contracts (Short RB)}}$$
- **ULSD Diesel (Product Short Futures Position):**
  $$V_{\text{target, diesel}} = h_{\text{diesel}}^* \times 21,000,000\ \text{gal} = 0.932 \times 21,000,000 = 19,572,000\ \text{gal}$$
  $$\text{Contracts to Sell} = \frac{19,572,000}{42,000\ \text{gal/contract}} = 466.0 \implies \mathbf{466\ \text{contracts (Short HO)}}$$

#### 6. Stress Scenario: Geopolitical Price Shock & Basis Blowout at Delivery ($t_1$)
Over the 60-day holding period, a major geopolitical supply disruption occurs, driving up crude oil prices globally. Concurrently, regional refinery turnarounds create a glut of gasoline in the Mid-Continent, widening the local basis discount:

- **New Futures Prices at Expiry ($t_1$):**
  - NYMEX WTI Crude: surges from $\$78.00$ to **$\$95.00/\text{bbl}$** ($+\$17.00/\text{bbl}$).
  - NYMEX RBOB Gasoline: rises moderately from $\$2.5000$ to **$\$2.7500/\text{gal}$** ($+\$0.2500/\text{gal} = +\$10.50/\text{bbl}$).
  - NYMEX ULSD Diesel: surges from $\$2.8000$ to **$\$3.4000/\text{gal}$** ($+\$0.6000/\text{gal} = +\$25.20/\text{bbl}$).
- **New Realized Local Basis at Delivery ($t_1$):**
  - Crude basis: stays firm at **$+\$2.30/\text{bbl}$** (slight widening from $+\$2.20$).
  - Gasoline basis: blows out due to regional pipeline proration to **$-\$0.0800/\text{gal}$** ($-\$3.36/\text{bbl}$, widened from $-\$1.47$).
  - Diesel basis: strengthens due to local agricultural harvest to **$+\$0.0600/\text{gal}$** ($+\$2.52/\text{bbl}$, strengthened from $+\$1.89$).

#### 7. Daily Variation Margin Tracking During Peak Volatility
During Day 35 to Day 39 of the campaign holding period, crude oil and distillates experienced acute upward jumps. The table below traces the daily variation margin (VM) cash flows debited or credited by the clearing broker across the three contract legs:

| Day | CL Futures Price ($\Delta \text{CL}$) | RB Futures Price ($\Delta \text{RB}$) | HO Futures Price ($\Delta \text{HO}$) | CL Daily VM ($+\text{Long}$) | RB Daily VM ($-\text{Short}$) | HO Daily VM ($-\text{Short}$) | Net Daily Cash VM Flow | Cumulative Margin Balance |
|---|---|---|---|---|---|---|---|---|
| **Day 35** | $\$82.50\ (+\$1.80)$ | $\$2.540\ (+\$0.020)$ | $\$2.910\ (+\$0.045)$ | $+\$2,741,400$ | $-\$762,720$ | $-\$880,740$ | $+\$1,097,940$ | $+\$1,097,940$ |
| **Day 36** | $\$86.20\ (+\$3.70)$ | $\$2.610\ (+\$0.070)$ | $\$3.050\ (+\$0.140)$ | $+\$5,635,100$ | $-\$2,669,520$ | $-\$2,739,800$ | $+\$225,780$ | $+\$1,323,720$ |
| **Day 37** | $\$91.00\ (+\$4.80)$ | $\$2.680\ (+\$0.070)$ | $\$3.220\ (+\$0.170)$ | $+\$7,310,400$ | $-\$2,669,520$ | $-\$3,327,240$ | $+\$1,313,640$ | $+\$2,637,360$ |
| **Day 38** | $\$89.50\ (-\$1.50)$ | $\$2.710\ (+\$0.030)$ | $\$3.260\ (+\$0.040)$ | $-\$2,284,500$ | $-\$1,144,080$ | $-\$782,880$ | $-\$4,211,460$ | $-\$1,574,100$ |
| **Day 39** | $\$95.00\ (+\$5.50)$ | $\$2.750\ (+\$0.040)$ | $\$3.400\ (+\$0.140)$ | $+\$8,376,500$ | $-\$1,525,440$ | $-\$2,739,800$ | $+\$4,111,260$ | $+\$2,537,160$ |

*Observation on Day 38:* When crude dipped $-\$1.50/\text{bbl}$ while product prices continued climbing, the long crude leg lost cash while both short product legs suffered losses, resulting in a single-day cash variation margin call of **$-\$4,211,460$**. The treasury department's committed revolving liquidity line absorbed this draw without triggering asset liquidations.

#### 8. Final Financial Settlement and Margin Realization
Let us calculate the unhedged physical margin, the financial derivative P&L, and the net realized economic result.

##### A. Unhedged Physical Performance
- Physical crude paid: $\$95.00 + \$2.30 = \$97.30/\text{bbl}$.
  - Total crude feedstock cost: $1,500,000\ \text{bbl} \times \$97.30 = \mathbf{\$145,950,000}$.
- Physical gasoline revenue: $(\$2.7500 - \$0.0800) \times 42,000,000\ \text{gal} = \$2.6700 \times 42,000,000 = \mathbf{\$112,140,000}$ ($\$112.14/\text{bbl}$).
- Physical diesel revenue: $(\$3.4000 + \$0.0600) \times 21,000,000\ \text{gal} = \$3.4600 \times 21,000,000 = \mathbf{\$72,660,000}$ ($\$145.32/\text{bbl}$).
- Gross physical revenue: $\$112,140,000 + \$72,660,000 = \mathbf{\$184,800,000}$.
- Realized Physical Gross Margin:
  $$\text{GRM}_{\text{actual}} = \frac{\$184,800,000 - \$145,950,000}{1,500,000} = \frac{\$38,850,000}{1,500,000} = \mathbf{\$25.90/\text{bbl}}$$
- Net physical margin after OPEX ($\$4.50/\text{bbl}$):
  $$\text{Margin}_{\text{unhedged}} = \$25.90 - \$4.50 = \mathbf{\$21.40/\text{bbl}} \implies \text{Total Unhedged Profit} = \mathbf{\$32,100,000}$$
  *Notice:* Without hedging, higher crude costs and the gasoline basis blowout eroded $\$4,125,000$ in expected operating profits ($\$36,225,000 - \$32,100,000$).

##### B. Financial Derivative Settlement
- **Long Crude Futures P&L (1,523 contracts):**
  $$\Delta P_{\text{crude}} = \$95.00 - \$78.00 = +\$17.00/\text{bbl}$$
  $$\text{P\&L}_{\text{crude}} = 1,523\ \text{contracts} \times 1,000\ \text{bbl} \times (+\$17.00) = \mathbf{+\$25,891,000}$$
- **Short Gasoline Futures P&L (908 contracts):**
  $$\Delta P_{\text{gas}} = \$2.7500 - \$2.5000 = +\$0.2500/\text{gal}\quad (\text{loss on short})$$
  $$\text{P\&L}_{\text{gas}} = 908\ \text{contracts} \times 42,000\ \text{gal} \times (-\$0.2500) = \mathbf{-\$9,534,000}$$
- **Short Diesel Futures P&L (466 contracts):**
  $$\Delta P_{\text{diesel}} = \$3.4000 - \$2.8000 = +\$0.6000/\text{gal}\quad (\text{loss on short})$$
  $$\text{P\&L}_{\text{diesel}} = 466\ \text{contracts} \times 42,000\ \text{gal} \times (-\$0.6000) = \mathbf{-\$11,743,200}$$
- **Net Financial Derivative P&L:**
  $$\text{P\&L}_{\text{deriv}} = +\$25,891,000 - \$9,534,000 - \$11,743,200 = \mathbf{+\$4,613,800}$$

##### C. Comparison: Naive Hedge ($h=1.0$) vs. Optimal Hedge ($h^*$)
If the refinery had implemented a naive 1:1 volumetric hedge ($N_{\text{crude}} = 1,500$ long; $N_{\text{gas}} = 1,000$ short; $N_{\text{diesel}} = 500$ short):
- Naive Crude P&L: $1,500 \times 1,000 \times (+\$17.00) = +\$25,500,000$.
- Naive Gas P&L: $1,000 \times 42,000 \times (-\$0.2500) = -\$10,500,000$.
- Naive Diesel P&L: $500 \times 42,000 \times (-\$0.6000) = -\$12,600,000$.
- Naive Net Derivative P&L: $+\$25,500,000 - \$10,500,000 - \$12,600,000 = +\$2,400,000$.
- Naive Hedged Profit: $\$32,100,000 + \$2,400,000 = \mathbf{\$34,500,000}$ ($\$23.00/\text{bbl}$).
The naive hedge fell short of the budget target by $-\$1,725,000$ because it over-hedged refined products ($h_{\text{gas}}^* = 0.908, h_{\text{diesel}}^* = 0.932$) and under-hedged crude feedstock ($h_{\text{crude}}^* = 1.015$).

```text
=============================================================================================================
                              FINAL CAMPAIGN OUTCOME SUMMARY
=============================================================================================================
Metric                                Target / Inception    Unhedged Actual    Naive Hedge (h=1)    Optimal Hedged (h*)
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Crude Cost ($/bbl)                           $80.20              $97.30              $97.30              $97.30
Gross Product Revenue ($/bbl)               $108.85             $123.20             $123.20             $123.20
Gross Refining Margin ($/bbl)                $28.65              $25.90              $25.90              $25.90
Plant Operating Costs ($/bbl)                 $4.50               $4.50               $4.50               $4.50
Net Operating Margin ($/bbl)                 $24.15              $21.40              $21.40              $21.40
Derivative Financial P&L ($/bbl)              $0.00               $0.00              +$1.60              +$3.08
Final Realized Margin ($/bbl)                $24.15              $21.40              $23.00              $24.48
Total Campaign P&L ($)                  $36,225,000         $32,100,000         $34,500,000         $36,713,800
Deviation from Target ($)                        $0         -$4,125,000         -$1,725,000           +$488,800
=============================================================================================================
```
*Synthesis:* The minimum-variance derivative hedge absorbed the massive $\$17.00/\text{bbl}$ crude oil price surge. Despite the unpredicted $\$1.89/\text{bbl}$ blowout in local gasoline basis discount, the net realized margin was $\$24.48/\text{bbl}$—within $1.3\%$ of the target budget of $\$24.15/\text{bbl}$, protecting corporate earnings.

---

## 7. Code, query, or calculation

The following self-contained Python script computes the optimal minimum-variance hedge ratios for a 3:2:1 crack spread, sizes futures contracts, and calculates variance reduction ($R^2$) from empirical price-change vectors.

```python
import numpy as np

# Synthetic historical daily price changes: [dS_crude, dS_gas, dS_diesel, dF_crude, dF_gas, dF_diesel]
# Units: Crude ($/bbl), Gas ($/gal), Diesel ($/gal)
np.random.seed(42)
n_days = 250
# Covariance parameters: [sigma_Sc, sigma_Sg, sigma_Sd, sigma_Fc, sigma_Fg, sigma_Fd]
sigmas = np.array([1.48, 0.0365, 0.0410, 1.40, 0.0380, 0.0420])
corr = np.array([
    [1.00, 0.78, 0.82, 0.96, 0.75, 0.80],
    [0.78, 1.00, 0.72, 0.76, 0.945, 0.70],
    [0.82, 0.72, 1.00, 0.81, 0.71, 0.955],
    [0.96, 0.76, 0.81, 1.00, 0.74, 0.79],
    [0.75, 0.945, 0.71, 0.74, 1.00, 0.68],
    [0.80, 0.70, 0.955, 0.79, 0.68, 1.00]
])
cov = np.outer(sigmas, sigmas) * corr
d_prices = np.random.multivariate_normal(mean=np.zeros(6), cov=cov, size=n_days)

dS_c, dS_g, dS_d = d_prices[:, 0], d_prices[:, 1], d_prices[:, 2]
dF_c, dF_g, dF_d = d_prices[:, 3], d_prices[:, 4], d_prices[:, 5]

# Step 1: Minimum-Variance Hedge Ratios (h* = Cov(dS, dF) / Var(dF))
h_crude = np.cov(dS_c, dF_c)[0, 1] / np.var(dF_c)
h_gas = np.cov(dS_g, dF_g)[0, 1] / np.var(dF_g)
h_diesel = np.cov(dS_d, dF_d)[0, 1] / np.var(dF_d)

# Step 2: Hedging Effectiveness (R^2 = rho^2)
he_crude = (np.corrcoef(dS_c, dF_c)[0, 1]) ** 2
he_gas = (np.corrcoef(dS_g, dF_g)[0, 1]) ** 2
he_diesel = (np.corrcoef(dS_d, dF_d)[0, 1]) ** 2

# Step 3: Contract Sizing for 1.5M bbl crude campaign (3:2:1 crack)
vol_crude = 1_500_000.0          # bbl
vol_gas = (2.0 / 3.0) * vol_crude * 42.0   # 42,000,000 gal
vol_diesel = (1.0 / 3.0) * vol_crude * 42.0 # 21,000,000 gal

lot_crude, lot_gas, lot_diesel = 1_000.0, 42_000.0, 42_000.0
n_crude = int(np.round((h_crude * vol_crude) / lot_crude))
n_gas = int(np.round((h_gas * vol_gas) / lot_gas))
n_diesel = int(np.round((h_diesel * vol_diesel) / lot_diesel))

# Step 4: 3:2:1 Physical vs Hedged Margin Variance
# Unhedged margin: 28*S_g + 14*S_d - S_c
d_margin_unhedged = 42.0 * ((2.0 * dS_g + 1.0 * dS_d) / 3.0) - dS_c
d_margin_hedged = d_margin_unhedged + (
    (n_crude * lot_crude * dF_c - n_gas * lot_gas * dF_g - n_diesel * lot_diesel * dF_d) / vol_crude
)

var_unhedged = np.var(d_margin_unhedged)
var_hedged = np.var(d_margin_hedged)
overall_he = 1.0 - (var_hedged / var_unhedged)

print(f"Optimal Hedge Ratios: Crude={h_crude:.3f}, Gas={h_gas:.3f}, Diesel={h_diesel:.3f}")
print(f"Contract Sizing: Crude Long={n_crude}, Gas Short={n_gas}, Diesel Short={n_diesel}")
print(f"Hedge Effectiveness: Crude={he_crude*100:.1f}%, Gas={he_gas*100:.1f}%, Diesel={he_diesel*100:.1f}%")
print(f"Overall Margin Variance: Unhedged={var_unhedged:.4f}, Hedged={var_hedged:.4f}")
print(f"Total Margin Variance Reduction: {overall_he*100:.2f}%")
```

### Assumptions and Expected Output
- **Assumptions:** Daily log-differenced or arithmetic price changes are stationary; linear correlation captures the dependence structure; standardized exchange lot sizes require integer rounding.
- **Expected Terminal Output:**
  ```text
  Optimal Hedge Ratios: Crude=0.992, Gas=0.927, Diesel=0.924
  Contract Sizing: Crude Long=1489, Gas Short=927, Diesel Short=462
  Hedge Effectiveness: Crude=90.1%, Gas=86.6%, Diesel=89.2%
  Overall Margin Variance: Unhedged=0.6816, Hedged=0.2903
  Total Margin Variance Reduction: 57.40%
  ```

---

## 8. Data, standards, and regulatory context

Commodity hedging operates under strict regulatory oversight, standardized exchange contracts, and rigorous accounting standards.

```text
=============================================================================================================
                        COMMODITY RISK REGULATORY & COMPLIANCE HIERARCHY
=============================================================================================================
REGULATORY BODY          MANDATE / STANDARD             BINDING REQUIREMENTS & COMPLIANCE ARTIFACTS
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
CFTC (United States)     Dodd-Frank Act Title VII       - Federal speculative position limits (17 CFR § 150)
                         Commodity Exchange Act (CEA)   - Bona Fide Hedging Exemptions (Form 404 / 104)
                                                        - Real-time swap data reporting (Part 43 / 45)
                                                        - Large Trader Reporting System (LTRS)

ESMA / EU Regulators     EMIR (Refit)                   - Central clearing mandate for standardized OTC swaps
                         MiFID II                       - Commodity derivative position limits & position reporting
                                                        - Ancillary Activity Exemption (NFC- status)
                                                        - Bilateral margin rules for uncleared derivatives

Central Exchanges        CME, NYMEX, CBOT, ICE,         - Contract specifications, tick sizes, delivery points
                         EEX, LME                       - Daily price fluctuation limits & circuit breakers
                                                        - Initial margin methodology (SPAN / CME CORE)
                                                        - Physical delivery & EFP (Exchange for Physical) rules

Accounting Standards     IFRS 9 (International)        - Designation of risk components (benchmark flat price)
                         ASC 815 / FAS 133 (US GAAP)    - Formal prospective & retrospective hedge documentation
                                                        - OCI balance sheet deferral for Cash Flow Hedges
                                                        - Ineffectiveness recognition in current quarterly P&L
=============================================================================================================
```

### 1. CFTC Dodd-Frank Act and Position Limits
Under Title VII of the Dodd-Frank Wall Street Reform and Consumer Protection Act (17 CFR Part 150), the US Commodity Futures Trading Commission (CFTC) enforces strict federal speculative position limits across 25 core physical commodity futures contracts (including crude oil, gasoline, heating oil, natural gas, corn, soybeans, and wheat) and economically equivalent swaps.
- **Bona Fide Hedging Exemption (17 CFR § 150.3):** Commercial process industries are exempt from speculative position caps only if hedges represent an authentic substitute for physical transactions. The hedger must demonstrate:
  1. Hedges represent raw feedstock requirements or output production not exceeding 12–24 months of physical processing capacity.
  2. The positions offset physical balance sheet inventory, fixed-price purchases, or fixed-price sales.
  3. Annual filings (CFTC Form 404) are submitted detailing physical storage capacity, milling/crushing throughput, and operational conversion volumes.
- **Large Trader Reporting:** Positions exceeding exchange reporting thresholds (e.g., 350 contracts in crude oil, 200 contracts in natural gas) must be reported daily, identifying ultimate beneficial ownership.

### 2. European Regulations: EMIR and MiFID II
- **EMIR (European Market Infrastructure Regulation):** Mandates central clearing for standardized OTC derivatives and daily bilateral exchange of initial and variation margins for uncleared OTC swaps. Non-Financial Counterparties (NFCs) that use derivatives exclusively for commercial risk mitigation qualify as **NFC-**, exempting them from mandatory clearing and capital charges.
- **MiFID II Ancillary Activity Test:** Process manufacturers must continuously audit their derivative trading volume relative to total physical group activity. If trading volume exceeds regulatory thresholds ($10\% - 50\%$ depending on asset class), the entity loses its commercial industrial status and is reclassified as a regulated financial investment firm.

### 3. Financial Accounting Standards: IFRS 9 vs. US GAAP ASC 815
The primary corporate governance objective is preventing derivative mark-to-market swings from corrupting operating EBITDA:
- **Eligible Risk Components:** IFRS 9 allows a processor to designate a separately identifiable and reliably measurable risk component of a physical contract as the hedged item (e.g., hedging only the benchmark WTI crude component of a physical crude purchase). ASC 815 historically restricted risk component hedging for non-financial assets to contractually specified benchmark rates, though updates have expanded alignment with IFRS 9.
- **Effectiveness Testing:** IFRS 9 eliminated the arbitrary $80\% - 125\%$ bright-line retrospective effectiveness corridor of IAS 39, replacing it with an objective requirement for an **economic relationship** and verified **hedge ratio alignment**. However, any hedge ineffectiveness (such as basis drift or unexpected physical volume curtailment) must be quantified and expensed immediately to current-period P&L.

#### Accounting Journal Entries: Cash Flow Hedge Workflow
The following ledger workflow illustrates how qualifying cash flow hedges shield quarterly EBITDA from derivative volatility:

| Event Date | Transaction / Valuation Trigger | Journal Debit (Dr) | Journal Credit (Cr) | Financial Statement Impact |
|---|---|---|---|---|
| **$t_0$ (Inception)** | Execute long crude futures at $F_0 = \$78.00/\text{bbl}$. | None (Fair Value = $\$0$). | None (Fair Value = $\$0$). | Off-balance sheet disclosure only; post initial margin cash to broker asset account. |
| **$t_1$ (Quarter End)** | Futures price rises to $\$90.00/\text{bbl}$ ($+\$12.00$ MtM gain). Physical crude not yet delivered. | **Dr** Derivative Clearing Asset: $\$18,000,000$ | **Cr** Other Comprehensive Income (OCI Equity): $\$18,000,000$ | Zero impact on net income (EBITDA shielded); equity reserves increase by $\$18\text{M}$. |
| **$t_2$ (Delivery)** | Futures close at $\$95.00/\text{bbl}$ ($+\$5.00$ gain). Purchase physical crude at spot $\$97.30/\text{bbl}$. | **Dr** Cash / Broker Receivable: $\$25,500,000$<br>**Dr** Physical Raw Material Inventory: $\$145,950,000$ | **Cr** Derivative Clearing Asset: $\$18,000,000$<br>**Cr** OCI Equity: $\$7,500,000$<br>**Cr** Cash (Feedstock Payment): $\$145,950,000$ | Realizes $\$25.5\text{M}$ cash gain from broker to pay crude supplier. Inventory recognized at cost. |
| **$t_3$ (Sale)** | Refined gasoline and diesel sold to commercial off-takers. | **Dr** OCI Equity (Reclassification): $\$25,500,000$<br>**Dr** Cost of Goods Sold (COGS): $\$120,450,000$ | **Cr** Raw Material Inventory (Expense): $\$145,950,000$ | Reclassifies OCI hedge gain directly into COGS, reducing net crude expense to target $\$80.20/\text{bbl}$ basis. |

---

## 9. Industry practice and case evidence

Real-world commodity conversion illustrates the operational consequences of basis volatility, forward curve shifts, and margin liquidity.

```text
=============================================================================================================
                         NOTABLE INDUSTRIAL COMMODITY HEDGING CASES
=============================================================================================================
Case Entity              Market & Sector            Strategy Architecture         Outcome & Structural Lesson
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Metallgesellschaft AG    Refined Products           Stack-and-Roll front-month   $1.3B loss; curve shifted to
(MGRM, 1993)             (Heating Oil & Gasoline)   futures to hedge 10-year     contango; catastrophic roll drag
                                                    fixed-price customer supply  and variation margin drain

Southwest Airlines       Jet Fuel & Crude Oil       Multi-year out-of-the-money  Saved >$3.5B over peers; avoided
(2000–2008)              (Air Transportation)       call option collars &        crack spread cross-hedge basis
                                                    refinery crack spreads       blowouts during oil price spike

Delta Air Lines          Jet Fuel Refining Margin   Vertical integration:        Reduced crack spread volatility;
(Trainer Refinery, 2012) (Downstream Jet Fuel)      Acquired 185k bpd physical   swapped paper financial hedges
                                                    refinery in Trainer, PA      for physical asset gross margin

European Power Utilities Power, Natural Gas, &      Selling forward clean spark  €100B+ emergency credit lines;
(Uniper, Fortum, 2022)   Carbon (EEX / TTF)         spreads backed by physical   economic hedges sound, but VM
                                                    gas-fired generation         calls triggered technical default

CME / NYMEX Crude        WTI Light Sweet Crude      Physical delivery settlement Price plunged to -$37.63/bbl;
(Negative Event, 2020)   (Cushing, OK Hub)          mandate colliding with full  storage exhaustion broke zero-lower
                                                    tank storage capacity        bound assumptions across industry

Agribusiness Giants      Soybeans, Corn, & Wheat    Dynamic crush spread locks;  Managed inland origination basis
(ADM, Bunge, Cargill)    (Agri-Processing)          Board Crush on CBOT matched  risk via owned barge, terminal,
                                                    with localized basis trading and elevator logistics assets
=============================================================================================================
```

### 1. Metallgesellschaft AG (1993): The Classic Stack-and-Roll Disaster
Metallgesellschaft Refining & Marketing (MGRM), an American subsidiary of the German industrial conglomerate, agreed to sell 160 million barrels of heating oil and gasoline to independent retailers over a 10-year period at fixed prices above the prevailing spot market. 
- *Strategy:* Because 10-year forward contracts were illiquid, MGRM implemented a **short-term stack-and-roll hedge**: they purchased massive volumes of front-month NYMEX futures and OTC swaps, rolling the entire stack forward each month.
- *The Flaw:* Throughout the late 1980s, oil markets traded in structural backwardation, allowing MGRM to earn positive roll yield. In late 1993, global oil markets flipped into deep **contango**. MGRM was forced to sell expiring front-month contracts at low spot prices and buy deferred contracts at higher prices, bleeding tens of millions of dollars in roll drag each month. Simultaneously, crude oil prices collapsed, triggering over $\$1.3\text{ billion}$ in cash variation margin calls. German parent bank supervisory boards panicked and liquidated the positions at the exact trough, locking in a $\$1.33\text{ billion}$ realized loss.
- *Lesson:* Never finance long-term fixed commitments with short-dated paper in contango without sufficient liquidity reserves and structural roll protection.

### 2. Southwest Airlines vs. Industry (2000–2008): Jet Fuel Crack Hedging
Jet fuel is not traded directly on primary futures exchanges; airlines must cross-hedge jet fuel using crude oil (WTI/Brent) or heating oil (ULSD). During the 2000–2008 commodity super-cycle:
- *Strategy:* Southwest Airlines recognized that jet fuel crack spreads (the differential between refined jet fuel and crude) could widen rapidly. Southwest hedged physical jet fuel using multi-leg option collars and synthetic crack spread call options on NYMEX heating oil and crude.
- *Result:* When crude surged toward $\$147/\text{bbl}$ in 2008, Southwest's fuel hedges generated over $\$3.5\text{ billion}$ in cash savings, keeping the airline profitable while competitors filed for Chapter 11 bankruptcy.
- *Subsequent Shift:* By 2012, Delta Air Lines took cross-hedging to its ultimate logical conclusion: instead of buying financial crack spreads, Delta purchased the idled 185,000 bpd Trainer, PA oil refinery from Phillips 66. By running the refinery to maximize jet fuel yield, Delta physically neutralized its crack spread exposure.

### 3. The 2022 European Power Utility Liquidity Crisis
European power utilities (including Germany's Uniper and Finland's Fortum) routinely hedge future electricity production by selling 1- to 3-year forward power contracts on the European Energy Exchange (EEX), simultaneously purchasing forward natural gas on the Dutch Title Transfer Facility (TTF) and EU carbon allowances (EUAs).
- *The Shock:* Following the Russian gas cutoffs in 2022, TTF natural gas surged from $€20/\text{MWh}$ to over $€300/\text{MWh}$, and wholesale German baseload electricity spiked above $€1,000/\text{MWh}$.
- *The Liquidity Trap:* The utilities held economically balanced clean spark spread hedges: the physical power plants would eventually generate power and sell it at the locked-in rates. However, EEX clearing rules required daily cash variation margin. As forward prices spiked 10x to 20x, variation margin calls exceeded **€100 billion** across the European utility sector. Utilities were on the verge of bankruptcy simply because they lacked the cash to satisfy exchange margin calls before physical power generation occurred. Germany was forced to nationalize Uniper and inject over €50 billion in emergency liquidity credit lines.
- *Lesson:* A physically hedged position with zero long-term economic risk can trigger corporate bankruptcy if short-term margin liquidity lines are undersized.

### 4. April 20, 2020: Negative WTI Pricing and Storage Containment Failure
On April 20, 2020, the expiring May 2020 NYMEX WTI crude oil futures contract collapsed from $+\$17.85/\text{bbl}$ at market open to an unprecedented settlement of **$-\$37.63/\text{bbl}$**.
- *Underlying Mechanism:* Due to COVID-19 demand destruction, physical storage tanks at the Cushing, Oklahoma delivery hub were completely saturated. Under NYMEX rulebook Article 7, any long futures contract held past the close of trading on the day prior to expiration mandates physical delivery into Cushing pipe or tank manifolds.
- *The Trap:* Non-physical financial participants (such as retail passive commodity ETFs like the United States Oil Fund, USO, and momentum retail traders) held long contracts. When pipeline operators at Cushing announced that zero uncommitted tankage or pipeline injection capacity was available to take physical delivery, long holders could not legally fulfill delivery obligations.
- *The Outcome:* Long holders were economically forced to dump contracts at any price to short physical storage operators. Commercial operators with spare tank bottoms were paid $\$37.63$ per barrel to take delivery of crude oil. The event exposed the critical failure of financial models that assumed price $S_t \ge 0$ and demonstrated that in physical commodity markets, logistics bottlenecks override financial paper theory.

### 5. Agribusiness Crush Hedging during the 2012 Midwest Drought
During the summer of 2012, the severe drought across the US Midwest decimated soybean and corn yields. 
- *Market Dynamics:* CBOT soybean futures surged above $\$17.50/\text{bushel}$. However, local crushing facilities faced an unprecedented **origination basis spike**: local grain elevators in Illinois and Iowa held scarce physical beans and demanded cash premiums of $+\$1.50$ to $+\$2.50/\text{bushel}$ above CBOT futures.
- *Processing Dilemma:* Processors who had locked in "Board Crush" margins on CBOT found their margins completely eliminated by the surging cash bean basis. Global agribusiness giants (Archer Daniels Midland, Bunge, Cargill) survived and profited because their integrated supply chain networks—comprising proprietary Mississippi barge fleets, deepwater export terminals in the Gulf, and rail logistics—allowed them to divert physical grain imports from South America into the US domestic market, capitalizing on spatial basis arbitrage while merchant crushers were forced to idle plants.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Underlying Physical/Econometric Mechanism | Operational/Financial Consequence | Mitigation Protocol |
|---|---|---|---|---|
| **Structural Basis Blowout** | Cash-futures basis diverges by $>3\sigma$ from 3-year historical corridor. | Regional transport bottleneck (pipeline outage, rail congestion, river freeze) traps local physical supply. | Futures gains fail to offset physical losses; catastrophic margin collapse despite "perfect" hedge. | Implement localized basis swaps; negotiate alternative off-take delivery points with transport optionality. |
| **Contango Roll Drag Exhaustion** | Forward curve transitions from backwardation into persistent contango ($F_2 > F_1$). | Global inventory builds; storage ullage fills; implied convenience yield collapses ($y \to 0$). | Rolling short-dated futures incurs continuous cash attrition; hedge cost exceeds physical processing margin. | Shift from stack-and-roll to forward strip hedges; lease physical commercial storage to monetize carry. |
| **Variation Margin Liquidity Squeeze** | Benchmark prices experience extreme daily jumps ($>4\sigma$); exchange increases IM by $>50\%$. | Central clearinghouse demands immediate daily cash settlement on unrealized paper losses. | Technical insolvency: inability to post margin forces broker liquidation at the market peak/trough. | Size standby credit facilities via Extreme Value Theory; utilize uncleared OTC swaps with collateral thresholds. |
| **Operational Trip Mismatch** | Unplanned shutdown of primary continuous catalytic unit or turbine trip. | Physical processing stops, but financial derivatives remain active in full nominal volume. | The hedge instantly converts into an open, naked speculative derivative position. | Establish automated SCADA/MES alert link to treasury; execute immediate derivative unwind or buyback. |
| **Asymmetric Correlation Breakdown** | Cross-commodity correlation drops below $0.60$ during market stress events. | Non-linear substitute dynamics; regulatory specification changes (e.g., IMO 2020 bunker fuel shift). | Derivative hedge moves in the same direction as the physical loss, doubling total portfolio variance. | Restrict cross-hedging to high-cointegration pairs; run rolling 30-day Kalman filter hedge ratio updates. |
| **IFRS 9 / ASC 815 Disqualification** | Ineffectiveness testing fails; $R^2$ drops below documented threshold. | Basis drift or operational timing mismatch decouples derivative and physical flows. | Reclassification of derivatives to fair-value P&L; quarterly EBITDA swings panic equity analysts and lenders. | Document designated benchmark risk components; utilize regression-based prospective effectiveness. |
| **Delivery Squeeze / Expiry Trap** | Front-month futures prices spike violently on final trading day before notice. | Financial hedger forgets to roll prior to First Notice Day; short squeeze by physical delivery accumulators. | Forced physical delivery into exchange-certified warehouses at unfavorable logistics hubs. | Enforce rigid rolling calendar: close/roll all positions at least 5 trading days prior to First Notice Day. |
| **Yield / Mass Balance Drift** | Actual plant product split deviates from standard 3:2:1 or crush ratios. | Catalyst poisoning, crude slate API gravity shifts, or soybean protein quality variations. | Hedge ratios are misaligned with actual molecular yields, leaving residual co-products unhedged. | Link APS / LP planning models (Aspen PIMS) directly to treasury hedge sizing algorithms. |
| **Rogue Trader / Speculative Creep** | Derivative trading volumes exceed physical monthly capacity ($>120\%$). | Hedging desk attempts to time market direction or "trade around" physical storage positions. | Severe speculative trading losses; violation of CFTC bona fide hedging exemptions; regulatory fines. | Implement strict volume caps pegged to physical ERP storage and scheduling limits; independent risk auditing. |
| **Counterparty Credit Default** | Credit Default Swap (CDS) spread of OTC swap counterparty widens significantly. | Insolvency of off-exchange bilateral trading partner during systemic financial crisis. | Derivative receivables are wiped out; physical processor left with naked market exposure. | Mandate central clearing; require bilateral Credit Support Annexes (CSAs) with two-way daily margining. |
| **Unit Conversion Arithmetic Error** | Internal spreadsheets mix bushel, short ton, metric ton, and barrel conversions. | Differing commercial quotation conventions across physical grain/oil markets and futures exchanges. | Hedges executed at $10\times$ or $20\times$ intended physical volume, creating massive unauthorized exposure. | Enforce standardized software platforms; eliminate manual Excel spreadsheet order generation. |
| **Negative Pricing Floor Failure** | Benchmark price trades below $\$5.00/\text{bbl}$ or approaches zero in deep contango. | Storage containment exhaustion at delivery hub collisions with mandatory physical settlement rules. | Unbounded liability on long futures positions; automatic liquidation cascades by electronic brokers. | Remove zero-floor code assumptions; transition valuation models to Bachelier normal distribution. |
| **Biofuel Mandate Regulatory Shock** | EPA Renewable Volume Obligation (RVO) or RIN price fluctuates by $>50\%$. | Regulatory shift alters statutory blending mandates for renewable diesel and biodiesel feedstocks. | Soybean oil share surges unpredictably, decoupling crush meal and oil relative margin contributions. | Implement dynamic synthetic spread hedging incorporating D4/D6 RIN derivative instruments. |
| **Pipeline Proration Capacity Lock** | Shipper allocation on primary trunkline reduced from $100\%$ to $40\%$. | Upstream drilling surge overfills pipeline capacity; pipeline operator prorates historical shippers. | Physical feedstock stranded upstream; refinery starved of crude while holding long futures positions. | Maintain diversified pipeline connectivity and contracted space on alternative rail/marine off-take. |
| **Dynamic Hedge Churn Transaction Drag** | Trading desk rebalances daily based on high-frequency Kalman filter changes. | Overfitting to transitory microstructure noise and bid-ask spread bounces. | Excessive execution commissions and bid-ask slippage consume up to $15\% - 25\%$ of gross processing margin. | Enforce minimum rebalancing threshold bands (e.g. only trade if $|h_t - h_{\text{current}}| > 0.08$). |

---

## 11. Assumptions and limits

| Assumption | Rationale in Classical Hedging Theory | Real-World Operational Breakdown | Diagnostic & Validation Protocol |
|---|---|---|---|
| **Constant Hedge Ratio ($h^*$)** | Assumes the linear regression slope between cash and futures remains stationary across time. | Correlation and relative volatilities are time-varying, shifting between macro regimes and market crises. | Test for structural breaks via Chow test; implement dynamic GARCH or rolling Kalman filter estimation. |
| **Infinite Margin Liquidity** | Assumes the firm can continuously fund daily variation margin calls without balance sheet stress. | Banks freeze credit lines during systemic shocks; commercial paper markets seize up. | Conduct extreme liquidity stress tests simulating a 5-day, $5\sigma$ price surge; monitor cash-to-margin ratio. |
| **Zero Spatial Basis Correlation with Flat Price** | Assumes local basis shocks are orthogonal to centralized benchmark price fluctuations. | Massive flat price spikes often trigger pipeline capacity bottlenecks, driving extreme basis widenings. | Compute cross-covariance between $F_t$ and $B_t$; model basis as a non-linear conditional function of flat price. |
| **Frictionless Market Liquidity** | Assumes futures can be bought and sold instantly at the quoted mid-market price with zero impact. | Large volume orders in deferred delivery months or during limit-move days incur severe slippage. | Track market depth, open interest, and bid-ask spreads; restrict lot sizes to $<5\%$ of daily average volume. |
| **Deterministic Production Volume** | Assumes physical processing throughput exactly equals scheduled capacity throughout the hedge horizon. | Turnarounds run over schedule; equipment failures force unscheduled flaring or rate reductions. | Size initial hedges to $70\% - 85\%$ of baseline capacity; layer remaining $15\% - 30\%$ dynamically as production confirms. |
| **Stationarity of Price Returns** | Assumes price changes are normally distributed with constant mean and finite variance. | Commodity price changes exhibit severe leptokurtosis (fat tails), positive skewness, and jump-diffusions. | Perform Jarque-Bera tests; utilize Student's $t$ or Generalized Error Distributions (GED) for risk modeling. |
| **Linear Co-Product Yields** | Assumes crude splitting or soybean crushing produces fixed volumetric or gravimetric product ratios. | Yield curves are non-linear functions of reactor severity, catalyst age, and feedstock chemical assay. | Ingest actual weekly laboratory assay and mass-balance reconciliations into the treasury risk ledger. |
| **Regulatory Exemption Permanence** | Assumes bona fide hedging exemptions from position limits remain static under regulatory review. | Regulators update position limit frameworks or adjust speculative threshold definitions. | Maintain quarterly legal audits of physical capacity filings (CFTC Form 404 / EMIR NFC- status). |
| **Symmetry of Risk Preferences** | Assumes company minimizes two-sided variance, treating upside margin windfalls identically to downside losses. | Corporate executives and lenders prioritize downside semi-variance and cash flow ruin avoidance. | Formulate hedge objectives using Lower Partial Moments (LPM) or Conditional Value at Risk (CVaR). |
| **Constant Convenience Yield** | Assumes convenience yield $y$ is a fixed annualized rate across the forward strip. | Convenience yield is a highly non-linear, seasonal function of physical inventory $I_t$. | Re-estimate implied $y_t$ weekly from calendar spreads across prompt, mid, and deferred tenors. |
| **Absence of Expiration Delivery Squeeze** | Assumes futures contract cash-settles or delivery can be avoided smoothly before notice. | Storage saturation at delivery hubs forces predatory cash settlements against trapped financial longs. | Require all positions to be rolled or liquidated at least 5 business days prior to First Notice Day. |
| **Spatial Basis Independence Across Nodes** | Assumes basis movements at different plant gates are independent and uncorrelated. | Regional grid congestions or river basin freezes cause multi-plant basis shocks to correlate at $\rho \approx 1.0$. | Build a network-wide spatial basis covariance matrix across all plant receipt and delivery racks. |

---

## 12. Exercises

### Exercise 1: 3:2:1 Petroleum Crack Spread Formulation and Contract Sizing
A Gulf Coast refinery processes $120,000\ \text{barrels per day}$ of light sweet crude. The commercial team plans to hedge $80\%$ of its anticipated physical production for the month of October ($31\ \text{days}$). 
- The refinery operates on a standard 3:2:1 yield (3 bbl crude $\to$ 2 bbl RBOB gasoline + 1 bbl ULSD diesel).
- Econometric estimation yields optimal minimum-variance hedge ratios: $h_{\text{crude}}^* = 0.98$, $h_{\text{gasoline}}^* = 0.92$, and $h_{\text{diesel}}^* = 0.95$.
- Contract lot sizes: Crude = $1,000\ \text{bbl}$; RBOB = $42,000\ \text{gal}$; ULSD = $42,000\ \text{gal}$.

*Deliverable:* Calculate (a) the total net physical volume to be hedged across all three commodities, and (b) the exact integer number of NYMEX contracts to buy or sell for each commodity leg.
*Acceptance Criteria:* Contract counts must be rounded to the nearest integer; physical barrels must be correctly converted to gallons where required.
*Estimated time:* 25 minutes.

### Exercise 2: Soybean Crush Margin and Revenue Decomposition
An agribusiness processing plant in Iowa crushes $5,000\ \text{metric tons}$ of soybeans per day.
- Unit conversions: $1\ \text{metric ton} = 36.7437\ \text{bushels}$.
- Standard processing yield: Each $60\ \text{lb}$ bushel yields $44\ \text{lbs}$ of meal, $11\ \text{lbs}$ of crude soybean oil, and $5\ \text{lbs}$ of moisture/shrinkage.
- Market prices:
  - CBOT Soybeans: $1,240\ \phi/\text{bushel}$.
  - CBOT Soybean Meal: $\$380.00/\text{short ton}$ ($1\ \text{short ton} = 2,000\ \text{lbs}$).
  - CBOT Soybean Oil: $54.00\ \phi/\text{lb}$.

*Deliverable:* Compute (a) the daily crushing volume in bushels, (b) the Gross Processing Margin (Crush Spread) in dollars per bushel ($\$/\text{bu}$), and (c) the total daily gross processing margin in US dollars.
*Acceptance Criteria:* Arithmetic must show all intermediate conversion steps for short tons and cents.
*Estimated time:* 20 minutes.

### Exercise 3: Minimum-Variance Hedge Ratio and Variance Reduction
A petrochemical plant purchases natural gas as feedstock. Historical analysis over 120 trading days yields:
- Standard deviation of local physical cash gas price changes: $\sigma_{\Delta S} = \$0.42/\text{MMBtu}$.
- Standard deviation of NYMEX Henry Hub futures price changes: $\sigma_{\Delta F} = \$0.36/\text{MMBtu}$.
- Correlation coefficient between cash and futures price changes: $\rho_{SF} = 0.88$.

*Deliverable:* Calculate (a) the optimal minimum-variance hedge ratio ($h^*$), (b) the hedging effectiveness ($R^2$), and (c) the variance of the unhedged position versus the hedged position.
*Acceptance Criteria:* Exact mathematical derivation and numeric substitution.
*Estimated time:* 15 minutes.

### Exercise 4: Forward Curve Implied Convenience Yield and Roll Return
The spot price of Brent crude oil is $S_0 = \$84.00/\text{bbl}$. The 6-month futures contract ($T = 0.5\ \text{years}$) trades at $F_{0,0.5} = \$79.50/\text{bbl}$.
- The continuous risk-free interest rate is $r = 4.5\%\ \text{per annum}$.
- The continuous physical cost of crude oil tank storage, ullage, and insurance is $c = 3.0\%\ \text{per annum}$.

*Deliverable:* Calculate (a) whether the market is in contango or backwardation, (b) the annualized continuous implied convenience yield ($y$), and (c) the percentage roll return realized by a short hedger who holds the position across the 6-month tenor.
*Acceptance Criteria:* Apply the Theory of Storage continuous formula; state the economic intuition of the resulting convenience yield.
*Estimated time:* 20 minutes.

### Exercise 5: Clean Spark Spread and Plant Dispatch Economics
A combined-cycle gas turbine (CCGT) power station has an operational heat rate of $\text{HR} = 6.90\ \text{MMBtu}/\text{MWh}$ and an emissions factor of $\text{EF} = 0.37\ \text{metric tons of CO}_2/\text{MWh}$.
- Day-ahead peak electricity price: $P_{\text{power}} = \$68.00/\text{MWh}$.
- Spot natural gas delivered: $P_{\text{gas}} = \$3.80/\text{MMBtu}$.
- Carbon allowance price: $P_{\text{carbon}} = \$32.00/\text{metric ton}$.
- Variable non-fuel O&M cost: $\text{VOM} = \$3.50/\text{MWh}$.

*Deliverable:* Calculate (a) the standard Spark Spread, (b) the Clean Spark Spread, and (c) the net economic contribution margin per MWh. Should the plant dispatch?
*Acceptance Criteria:* Identify the threshold dispatch condition and quantify net profit per MWh.
*Estimated time:* 15 minutes.

### Exercise 6: Variation Margin Stress-Testing and Liquidity Sizing
A grain exporter holds a short futures hedge of $10,000,000\ \text{bushels}$ of CBOT wheat ($2,000\ \text{contracts}$ at $5,000\ \text{bu/contract}$) entered at an initial price of $F_0 = \$6.50/\text{bu}$.
- The treasury team wants to size a 5-day liquidity buffer to cover potential variation margin calls at a $99\%$ confidence level ($z_{0.99} = 2.326$).
- Daily wheat futures returns exhibit a volatility of $\sigma = 2.2\%$ with an empirical kurtosis of $k = 5.2$.

*Deliverable:* Using the Cornish-Fisher expansion / fat-tailed volatility model, calculate the maximum expected 5-day variation margin cash draw that the treasury credit line must support.
*Acceptance Criteria:* Show the step-by-step calculation of the scaled 5-day return and dollar margin draw.
*Estimated time:* 25 minutes.

### Exercise 7: Dynamic Kalman Filter Hedge Updating vs. Static OLS
A refinery risk manager observes a static OLS hedge ratio of $h_{\text{OLS}} = 0.95$ with an error variance of $P_{t-1} = 0.04$. The cash price error variance is $\sigma_{\epsilon}^2 = 0.16$. On day $t$, benchmark futures change by $\Delta F_t = +\$2.00/\text{bbl}$, while local crude cash changes by $\Delta S_t = +\$2.40/\text{bbl}$.
- The measurement equation is: $\Delta S_t = h_t \Delta F_t + \epsilon_t$.

*Deliverable:* Compute (a) the prior forecast error (innovation $\nu_t$), (b) the Kalman gain $K_t$, (c) the updated dynamic hedge ratio $\hat{h}_t$, and (d) the posterior error variance $P_t$.
*Acceptance Criteria:* Apply the discrete Kalman filter update equations; interpret why the hedge ratio adjusted upward.
*Estimated time:* 20 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1
1. **Physical Volumes to Hedge (80% of 31-day production):**
   - Total crude processed: $120,000\ \text{bpd} \times 31\ \text{days} = 3,720,000\ \text{bbl}$.
   - Crude volume to hedge: $3,720,000 \times 0.80 = \mathbf{2,976,000\ \text{bbl}}$.
   - Gasoline volume produced: $\frac{2}{3} \times 3,720,000 = 2,480,000\ \text{bbl} = 104,160,000\ \text{gal}$.
   - Gasoline volume to hedge: $104,160,000 \times 0.80 = \mathbf{83,328,000\ \text{gal}}$ ($1,984,000\ \text{bbl}$).
   - Diesel volume produced: $\frac{1}{3} \times 3,720,000 = 1,240,000\ \text{bbl} = 52,080,000\ \text{gal}$.
   - Diesel volume to hedge: $52,080,000 \times 0.80 = \mathbf{41,664,000\ \text{gal}}$ ($992,000\ \text{bbl}$).
2. **Contract Sizing:**
   - Crude (Long CL):
     $$N_{\text{crude}} = \frac{h_{\text{crude}}^* \times V_{\text{crude}}}{1,000} = \frac{0.98 \times 2,976,000}{1,000} = 2,916.48 \implies \mathbf{2,916\ \text{contracts (Long)}}$$
   - Gasoline (Short RB):
     $$N_{\text{gas}} = \frac{h_{\text{gas}}^* \times V_{\text{gas}}}{42,000} = \frac{0.92 \times 83,328,000}{42,000} = 1,825.28 \implies \mathbf{1,825\ \text{contracts (Short)}}$$
   - Diesel (Short HO):
     $$N_{\text{diesel}} = \frac{h_{\text{diesel}}^* \times V_{\text{diesel}}}{42,000} = \frac{0.95 \times 41,664,000}{42,000} = 942.40 \implies \mathbf{942\ \text{contracts (Short)}}$$

### Answer to Exercise 2
1. **Daily Crushing Volume in Bushels:**
   $$V_{\text{daily}} = 5,000\ \text{MT} \times 36.7437\ \text{bu/MT} = \mathbf{183,718.5\ \text{bushels/day}}$$
2. **Gross Processing Margin (Crush Spread) per Bushel:**
   - Product revenues per bushel:
     - Meal revenue: $\$380.00/\text{ton} \times \left(\frac{44\ \text{lbs}}{2,000\ \text{lbs}}\right) = \$380.00 \times 0.022 = \$8.360/\text{bu}$.
     - Oil revenue: $\$0.5400/\text{lb} \times 11\ \text{lbs} = \$5.940/\text{bu}$.
     - Total product revenue: $\$8.360 + \$5.940 = \$14.300/\text{bu}$.
   - Raw soybean feedstock cost:
     - $1,240\ \phi/\text{bu} = \$12.400/\text{bu}$.
   - Gross Crush Spread:
     $$\text{Crush} = \$14.300 - \$12.400 = \mathbf{\$1.900/\text{bushel}}$$
3. **Total Daily Gross Processing Margin:**
   $$\text{Margin}_{\text{total}} = 183,718.5\ \text{bu} \times \$1.900/\text{bu} = \mathbf{\$349,065.15/\text{day}}$$

### Answer to Exercise 3
1. **Optimal Hedge Ratio ($h^*$):**
   $$h^* = \rho_{SF} \frac{\sigma_{\Delta S}}{\sigma_{\Delta F}} = 0.88 \times \frac{0.42}{0.36} = 0.88 \times 1.1667 = \mathbf{1.0267}$$
2. **Hedging Effectiveness ($R^2$):**
   $$R^2 = \rho_{SF}^2 = (0.88)^2 = \mathbf{0.7744}\ (77.44\%)$$
3. **Variance Comparison:**
   - Unhedged variance: $\sigma_{\Delta S}^2 = (0.42)^2 = \mathbf{0.1764}\ (\$/\text{MMBtu})^2$.
   - Hedged variance: $\sigma_{\Pi}^{*2} = \sigma_{\Delta S}^2 (1 - R^2) = 0.1764 \times (1 - 0.7744) = 0.1764 \times 0.2256 = \mathbf{0.0398}\ (\$/\text{MMBtu})^2$.
   - The hedge eliminates $77.44\%$ of the cash flow variance.

### Answer to Exercise 4
1. **Regime Identification:**
   Because $F_{0,0.5} = \$79.50 < S_0 = \$84.00$, the market is in **backwardation**.
2. **Annualized Implied Convenience Yield ($y$):**
   Using $F_{0,T} = S_0 e^{(r + c - y)T}$:
   $$\ln\left(\frac{F_{0,T}}{S_0}\right) = (r + c - y)T$$
   $$r + c - y = \frac{1}{T} \ln\left(\frac{F_{0,T}}{S_0}\right) = \frac{1}{0.5} \ln\left(\frac{79.50}{84.00}\right) = 2 \times (-0.05506) = -0.11012\ (-11.01\%)$$
   $$y = r + c - (-0.11012) = 0.045 + 0.030 + 0.11012 = \mathbf{0.18512}\ (18.51\%\ \text{per annum})$$
   *Intuition:* The convenience yield ($18.51\%$) substantially exceeds total carrying costs ($7.5\%$), reflecting acute immediate demand for prompt physical crude inventory.
3. **Roll Return for a Short Hedger:**
   $$\text{Roll Return} = \frac{S_0 - F_{0,0.5}}{S_0} = \frac{84.00 - 79.50}{84.00} = \mathbf{+5.36\%}$$
   In backwardation, the short hedger enjoys a positive roll yield as the forward price converges upward toward the spot price over time.

### Answer to Exercise 5
1. **Standard Spark Spread:**
   $$\text{SS} = P_{\text{power}} - (\text{HR} \times P_{\text{gas}}) = \$68.00 - (6.90 \times \$3.80) = \$68.00 - \$26.22 = \mathbf{\$41.78/\text{MWh}}$$
2. **Clean Spark Spread:**
   $$\text{CSS} = \text{SS} - (\text{EF} \times P_{\text{carbon}}) = \$41.78 - (0.37 \times \$32.00) = \$41.78 - \$11.84 = \mathbf{\$29.94/\text{MWh}}$$
3. **Net Contribution Margin:**
   $$\text{Margin}_{\text{net}} = \text{CSS} - \text{VOM} = \$29.94 - \$3.50 = \mathbf{+\$26.44/\text{MWh}}$$
   *Dispatch Decision:* Because the net contribution margin is positive ($+\$26.44/\text{MWh} > 0$), the plant covers all operating variable costs and contributes to fixed overhead. **The plant should dispatch at maximum baseload capacity.**

### Answer to Exercise 6
1. **Fat-Tailed 5-Day Value at Risk Scaling:**
   - 1-day standard normal quantile for $99\%$: $z_{0.99} = 2.326$.
   - Cornish-Fisher adjustment for kurtosis ($k = 5.2$):
     $$z_{\text{CF}} = z + \frac{k - 3}{6}(z^2 - 1) = 2.326 + \frac{5.2 - 3}{6}(2.326^2 - 1) = 2.326 + \frac{2.2}{6}(5.410 - 1) = 2.326 + 0.3667 \times 4.410 = 2.326 + 1.617 = 3.943$$
   - 5-day scaled percentage price jump:
     $$\Delta F_{\text{max},\%} = z_{\text{CF}} \times \sigma \times \sqrt{5} = 3.943 \times 0.022 \times 2.236 = 0.1940\ (19.40\%)$$
2. **Dollar Price Jump and Variation Margin Draw:**
   - Maximum expected 5-day price jump: $\Delta F = \$6.50 \times 0.1940 = \$1.261/\text{bu}$.
   - Total volume short: $10,000,000\ \text{bushels}$.
   - Maximum expected Variation Margin cash draw:
     $$\Delta \text{VM} = 10,000,000\ \text{bu} \times \$1.261/\text{bu} = \mathbf{\$12,610,000}$$
   *Acceptance Criteria:* The treasury team must secure a dedicated, committed revolving credit line of at least **$\$12.61\text{ million}$** specifically earmarked for broker variation margin calls.

### Answer to Exercise 7
1. **Prior Forecast Error (Innovation):**
   $$\hat{y}_{t|t-1} = \hat{h}_{t-1} \Delta F_t = 0.95 \times 2.00 = \$1.90/\text{bbl}$$
   $$\nu_t = \Delta S_t - \hat{y}_{t|t-1} = 2.40 - 1.90 = \mathbf{+\$0.50/\text{bbl}}$$
2. **Kalman Gain ($K_t$):**
   $$K_t = \frac{P_{t-1} \Delta F_t}{(\Delta F_t)^2 P_{t-1} + \sigma_{\epsilon}^2} = \frac{0.04 \times 2.00}{(2.00)^2 \times 0.04 + 0.16} = \frac{0.08}{4.00 \times 0.04 + 0.16} = \frac{0.08}{0.16 + 0.16} = \frac{0.08}{0.32} = \mathbf{0.250}$$
3. **Updated Dynamic Hedge Ratio ($\hat{h}_t$):**
   $$\hat{h}_t = \hat{h}_{t-1} + K_t \nu_t = 0.95 + 0.250 \times (+0.50) = 0.95 + 0.125 = \mathbf{1.075}$$
4. **Posterior Error Variance ($P_t$):**
   $$P_t = (1 - K_t \Delta F_t) P_{t-1} = (1 - 0.250 \times 2.00) \times 0.04 = (1 - 0.50) \times 0.04 = \mathbf{0.020}$$
   *Interpretation:* Because local cash rose by $\$2.40$ while futures rose by only $\$2.00$ ($\Delta S / \Delta F = 1.20$), the Kalman filter recognized that local price volatility exceeded benchmark volatility, systematically adjusting the hedge ratio upward from $0.95$ to $1.075$ to maintain variance minimization.

---

## 14. Interview questions

### Question 1: Flat Price vs. Basis Risk
**Question:** "If our refinery executes an economically perfect $1:1$ short futures hedge on NYMEX for our entire monthly gasoline production, can we still experience a major quarterly earnings collapse due to market price movements?"
- **Strong answer contains:** "Yes. Futures contracts settle against benchmark hub prices (e.g., NYMEX RBOB in New York Harbor). Our physical refinery sells gasoline at the local wholesale rack or pipeline origin. Our all-in cash price is $S_t = F_t + B_t$. A $1:1$ futures hedge eliminates flat price risk ($\text{Var}(F_t)$), but leaves us 100% exposed to local basis risk ($\text{Var}(B_t)$). If regional supply gluts, pipeline maintenance, or local demand drops cause the local basis discount to blow out from $-\$0.02/\text{gal}$ to $-\$0.25/\text{gal}$, our realized margin collapses by that exact differential despite the futures hedge."
- **Weak answer reveals:** Believing that exchange futures guarantee complete margin safety or confusing basis risk with execution commission fees.

### Question 2: Roll Yield and the Metallgesellschaft Collapse
**Question:** "Why did Metallgesellschaft lose over $\$1.3\text{ billion}$ hedging long-term customer fixed-price heating oil contracts, and what forward curve condition caused their failure?"
- **Strong answer contains:** "MGRM used a stack-and-roll strategy, hedging 10-year physical delivery commitments with short-dated (1- to 3-month) front-month futures. This worked during backwardation because rolling short contracts yielded positive roll gains ($F_1 > F_2$). However, when the oil forward curve shifted into persistent contango ($F_1 < F_2$), MGRM was forced to buy back maturing contracts low and buy deferred contracts high, bleeding negative roll yield (roll drag) month after month. Simultaneously, falling oil prices triggered massive daily variation margin calls from clearing brokers, exhausting their cash liquidity."
- **Weak answer reveals:** Assuming MGRM failed simply because oil prices fell, failing to identify the forward curve contango roll drag or the mismatch between paper cash margin calls and long-term physical receivables.

### Question 3: Minimum-Variance Hedge Ratio Derivation
**Question:** "Why is the optimal hedge ratio $h^*$ rarely equal to $1.0$, and what does it mean if $h^* > 1.0$?"
- **Strong answer contains:** "The hedge ratio is derived by minimizing portfolio variance: $h^* = \rho_{SF} \frac{\sigma_S}{\sigma_F} = \frac{\text{Cov}(\Delta S, \Delta F)}{\text{Var}(\Delta F)}$. If local physical spot cash prices exhibit greater absolute volatility than exchange futures ($\sigma_S > \sigma_F$), and correlation is high, $h^*$ will be strictly greater than $1.0$. An $h^* = 1.15$ means that for every 100 barrels of physical commodity, we must trade 115 barrels of futures to achieve maximum variance reduction, compensating for the higher volatility of the local cash market."
- **Weak answer reveals:** Assuming $h > 1.0$ represents speculative leverage or reckless gambling rather than a variance-minimizing mathematical scaling factor.

### Question 4: Convenience Yield and Tank Storage Economics
**Question:** "What is convenience yield, and why does an empty storage tank have zero convenience yield while a full storage tank operating near minimum operating bottoms has an extremely high convenience yield?"
- **Strong answer contains:** "Convenience yield ($y$) is the unobservable economic benefit accrued by holding physical inventory rather than a paper futures contract. It represents operational option value. When a processing plant is near its minimum operating bottoms (heel), a supply disruption would force an emergency unit shutdown, costing millions. Therefore, the marginal convenience yield of holding the next physical barrel or bushel approaches infinity. Conversely, when storage tanks are completely full, additional physical inventory provides zero incremental operational protection and merely incurs carrying costs, driving $y \to 0$ and tipping the curve into contango."
- **Weak answer reveals:** Describing convenience yield as a financial dividend or interest payment, or failing to connect it to operational stockout risk.

### Question 5: Spark Spread vs. Clean Spark Spread
**Question:** "How does the implementation of an emissions trading scheme (like EU ETS) alter the dispatch order and hedging strategy of a thermoelectric power generator?"
- **Strong answer contains:** "Under carbon pricing, generators must evaluate the Clean Spark Spread: $\text{CSS} = P_{\text{power}} - (\text{HR} \times P_{\text{gas}}) - (\text{EF} \times P_{\text{carbon}})$. Because coal has roughly double the carbon emissions intensity per MWh compared to natural gas ($\approx 0.95\ \text{tCO}_2/\text{MWh}$ vs. $\approx 0.38\ \text{tCO}_2/\text{MWh}$), high carbon allowance prices flip the merit order, making CCGT gas units run ahead of coal. In terms of hedging, the utility cannot simply hedge power and gas; it must execute a three-legged hedge incorporating carbon allowance futures (EUAs) to fully lock in the spark spread."
- **Weak answer reveals:** Treating carbon costs as a fixed plant overhead expense rather than a variable marginal operating dispatch constraint.

### Question 6: Cash Flow Hedge Accounting under IFRS 9
**Question:** "Why does corporate treasury insist on qualifying our commodity hedges under IFRS 9 / ASC 815 cash flow hedge accounting?"
- **Strong answer contains:** "Without formal hedge accounting, financial derivatives are marked-to-market daily through current-period P&L, whereas physical inventory or anticipated sales are recognized in future periods when delivered. This mismatch creates massive, artificial net income volatility. IFRS 9 cash flow hedge accounting allows the effective portion of derivative mark-to-market gains and losses to be deferred in Other Comprehensive Income (OCI) on the balance sheet, releasing to P&L only when the physical sales transaction affects earnings, perfectly matching revenues and costs."
- **Weak answer reveals:** Believing hedge accounting reduces the cash taxes paid or changes the cash flows of the derivative itself.

### Question 7: Liquidity Buffer Sizing for Futures Hedges
**Question:** "Our trading desk proposes moving our crack spread hedges from bilateral OTC swaps with an investment bank to cleared NYMEX futures to eliminate counterparty risk. What balance sheet risk are we introducing?"
- **Strong answer contains:** "We eliminate counterparty credit risk, but we introduce acute **cash liquidity and variation margin risk**. Bilateral OTC swaps often carry credit support agreements (CSAs) with non-zero collateral thresholds, meaning no cash moves unless exposure exceeds $\$20\text{M}$ or $\$50\text{M}$. Cleared exchange futures require zero-threshold, daily cash mark-to-market settlement. If energy prices surge rapidly, our short futures hedges will generate tens of millions in immediate cash margin calls at 08:00 AM. If we do not have dedicated committed revolving bank credit facilities, we could face a liquidity default despite holding profitable physical assets."
- **Weak answer reveals:** Assuming cleared futures are strictly superior in all dimensions without considering margin liquidity mechanics.

### Question 8: Cross-Hedging and the Samuelson Effect
**Question:** "Why does an airline hedging jet fuel with heating oil futures experience higher hedge tracking error when using 12-month forward contracts compared to 1-month prompt contracts?"
- **Strong answer contains:** "This involves two interacting effects: cross-hedging basis drift and the Samuelson Effect. First, heating oil and jet fuel correlations decay over longer horizons due to diverging seasonal demand cycles and refinery yield shifts. Second, under the Samuelson Effect, long-dated futures contracts exhibit lower volatility ($\sigma(T) = \sigma_0 e^{-\alpha T}$) and significantly lower market liquidity than prompt contracts. The bid-ask spread and lack of depth in 12-month heating oil futures amplify execution slippage, reducing overall hedge effectiveness."
- **Weak answer reveals:** Assuming correlation is mathematically identical across all forward curve tenors.

### Question 9: Negative Pricing Mechanics on April 20, 2020
**Question:** "Why did the May 2020 WTI contract trade down to $-\$37.63/\text{bbl}$, and why couldn't cash arbitrageurs simply buy the contracts and store the oil?"
- **Strong answer contains:** "The crash occurred on the penultimate trading day before delivery notice. Under NYMEX rules, expiring long contracts mandate physical delivery at Cushing, OK. All commercial storage tanks, pipeline manifolds, and rail off-take at Cushing were $100\%$ allocated or operationally constrained due to COVID-19 demand collapse. Financial passive longs (such as retail ETF holders) could not legally or physically take delivery of physical crude oil. Physical storage operators with spare ullage demanded up to $\$37.63/\text{bbl}$ to accept custody. The lack of available physical storage destroyed the physical-financial arbitrage link, allowing prices to drop deep into negative territory."
- **Weak answer reveals:** Asserting that negative pricing was an electronic exchange software glitch or that market participants were irrational.

### Question 10: Bona Fide Hedging under CFTC Rule 150
**Question:** "Under what statutory conditions can a soybean crushing company exceed CFTC federal speculative position limits on CBOT soybean futures?"
- **Strong answer contains:** "Under CFTC 17 CFR § 150.3, the company qualifies for a **bona fide hedging exemption** if the positions represent an authentic substitute for physical transactions. The hedger must document that the futures positions represent anticipated feedstock requirements or finished meal/oil sales not exceeding 12 to 24 months of verified physical plant crushing capacity. The company must file annual CFTC Form 404 documentation demonstrating physical elevator storage volume, daily crush capacity, and commercial off-take commitments."
- **Weak answer reveals:** Claiming that any commercial company can trade unlimited futures contracts without regulatory filing or physical asset verification.

---

## 15. Key terms

- **Flat Price:** The centralized exchange benchmark price for a standard commodity grade at a primary delivery hub.
- **Physical Basis:** The localized price differential: $B_t = S_t - F_t$, reflecting transport tariffs, quality assays, and regional balances.
- **Location Basis:** Spatial differential reflecting transportation freight (pipeline, rail, barge) between plant gate and pricing hub.
- **Quality Basis:** Price differential reflecting physical assay divergence (API gravity, sulfur, moisture, protein) from par.
- **Basis Risk:** The financial variance remaining in a hedged position due to unexpected fluctuations in local physical basis.
- **Gross Processing Margin (GPM):** The conversion spread between the aggregate market value of co-products and the cost of raw feedstock.
- **3:2:1 Crack Spread:** The synthetic petroleum refining margin representing 3 barrels of crude converted to 2 barrels of gasoline and 1 barrel of diesel.
- **5:3:2 Crack Spread:** The high-conversion refining margin representing 5 barrels of crude yielding 3 barrels of gasoline and 2 barrels of diesel.
- **Reverse Crack Spread:** Hedging strategy of buying crude and selling products executed when refining margins collapse below variable operating costs.
- **Soybean Crush Spread:** The gross margin from processing 1 bushel of soybeans into 44 lbs of meal and 11 lbs of oil.
- **Oil Share:** The percentage of total soybean crushing revenue generated by crude soybean oil relative to meal.
- **Board Crush:** Synthetic crush margin calculated strictly using CBOT benchmark futures without local elevator basis.
- **Spark Spread:** The gross generation margin of a natural gas power plant: electricity price minus fuel cost adjusted for heat rate.
- **Clean Spark Spread:** The spark spread minus the cost of carbon emissions allowances per MWh generated.
- **Heat Rate:** The measure of thermoelectric generating efficiency, expressing the fuel energy (MMBtu) required to produce 1 MWh of electricity.
- **Cost of Carry:** The total capital financing, physical storage, ullage insurance, and deterioration costs of holding inventory.
- **Convenience Yield:** The unobservable operational benefit and option value accrued by holding physical commodity inventory rather than paper contracts.
- **Contango:** The forward curve structure where deferred futures trade at a premium to prompt spot prices ($r + c > y$).
- **Backwardation:** The forward curve structure where prompt spot prices trade at a premium to deferred futures ($y > r + c$).
- **Roll Yield / Roll Drag:** The financial gain (in backwardation) or loss (in contango) generated by rolling an expiring derivative contract into a deferred contract.
- **Minimum-Variance Hedge Ratio ($h^*$):** The variance-minimizing ratio of derivative volume to physical volume: $h^* = \rho \frac{\sigma_S}{\sigma_F}$.
- **Hedging Effectiveness ($R^2$):** The proportion of physical cash flow variance eliminated by the optimal derivative hedge.
- **Stack-and-Roll:** Hedging long-term physical commitments by concentrating volume in liquid prompt-month contracts and rolling forward monthly.
- **Strip Hedge:** Hedging multi-period physical exposure using individual futures matching each corresponding delivery tenor.
- **Initial Margin:** Collateral deposited with an exchange clearinghouse to open and support a leveraged derivative position.
- **Variation Margin:** Daily cash settlement debited or credited by the clearinghouse reflecting mark-to-market price moves.
- **Samuelson Effect:** The empirical phenomenon where futures price volatility increases as contract maturity approaches.
- **Error Correction Model (ECM):** An econometric regression incorporating the cointegrating basis error to estimate dynamic hedge ratios.
- **Kalman Filter:** State-space econometric algorithm updating dynamic hedge ratios recursively in response to evolving basis innovations.
- **Bona Fide Hedging:** Regulatory classification granting commercial processors exemptions from speculative position limits.
- **Cash Flow Hedge:** Accounting classification (IFRS 9 / ASC 815) deferring derivative mark-to-market swings in OCI until physical delivery.
- **Extreme Value Theory (EVT):** Statistical framework modeling fat-tailed distribution extremes to size treasury margin liquidity buffers.

---

## 16. Sources

- **Brennan, M. J. (1958).** The supply of storage. *The American Economic Review*, 48(1), 50–72. [Foundational derivation of the convenience yield and the Theory of Storage].
- **Working, H. (1949).** The theory of price of storage. *The American Economic Review*, 39(6), 1254–1262. [Seminal empirical work on carrying charges and forward curve inversion].
- **Kaldor, N. (1939).** Speculation and economic stability. *The Review of Economic Studies*, 7(1), 1–27. [Original formulation of the convenience yield concept in commodity holding].
- **Eydeland, A., & Wolyniec, K. (2003).** *Energy and Power Risk Management: New Developments in Modeling, Pricing, and Hedging*. John Wiley & Sons. [Definitive mathematical reference for spark spreads, heat rates, and power dispatch].
- **Geman, H. (2005).** *Commodities and Commodity Derivatives: Modeling and Pricing for Agriculturals, Metals and Energy*. John Wiley & Sons. [Comprehensive treatise on forward curve term structures and transformation margins].
- **Hull, J. C. (2022).** *Options, Futures, and Other Derivatives* (11th ed.). Pearson. [Standard academic derivation of minimum-variance hedge ratios and cross-hedging mechanics].
- **Culp, C. L., & Miller, M. H. (1995).** Metallgesellschaft and the economics of synthetic storage. *Journal of Applied Corporate Finance*, 7(4), 62–76. [Rigorous academic analysis of the MGRM stack-and-roll failure].
- **Commodity Futures Trading Commission (CFTC). (2021).** Position Limits for Derivatives. *Federal Register*, 86 FR 3236 (17 CFR Parts 19, 140, and 150). [Binding US federal regulations on speculative caps and bona fide hedging exemptions].
- **International Accounting Standards Board (IASB). (2014).** *IFRS 9 Financial Instruments*. IFRS Foundation. [International accounting standard governing classification, measurement, and hedge accounting].
- **Financial Accounting Standards Board (FASB). (2017).** *Targeted Improvements to Accounting for Hedging Activities* (Accounting Standards Update No. 2017-12, Topic 815). [US GAAP framework for cash flow and fair value hedge accounting].

---

## 17. Links

- **Module Index:** [Module 38 — Process Industries, Commodities, and Bulk Supply Chains](_index.md)
- **Preceding Topics:**
  - [Topic 01 — Continuous, Batch, and Discrete Production](01_continuous-batch-discrete.md)
  - [Topic 02 — Yield and Mass Balance Modeling](02_yield-mass-balance.md)
  - [Topic 03 — Quality Specifications and Blending Optimization](03_quality-specs-blending.md)
  - [Topic 05 — Bulk Storage, Inventory Accounting, and Tank Farm Scheduling](05_bulk-storage-inventory.md)
- **Subsequent Topics:**
  - [Topic 07 — Bulk Logistics, Marine Chartering, and Laytime Management](07_bulk-logistics-laytime.md)
  - [Topic 08 — Capacity Utilization, Debottlenecking, and Throughput Optimization](08_capacity-debottlenecking.md)
- **Program Survey File:** [Course Survey: Process Industries and Commodities](../../../course/38_process_industries_commodities.md)
- **Related Elective Modules:**
  - [Module 15 — Stochastic Inventory & Risk-Pooling](../../02_methods/15_inventory_theory/_index.md)
  - [Module 16 — Supply Chain Network Economics](../../02_methods/16_network_economics/_index.md)
  - [Module 20 — Operations Planning and Master Scheduling](../../03_commerce/20_operations_planning_master_scheduling/_index.md)
  - [Module 39 — Mining, Metals, and Forestry Supply Chains](../39_mining_metals_forestry/_index.md)
