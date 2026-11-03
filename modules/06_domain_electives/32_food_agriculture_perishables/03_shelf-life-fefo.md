# Shelf Life, FEFO, and Age-Structured Inventory

> Module 32 — Food, Agriculture, and Perishable Supply Chains
> Topic 03 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Module 10 (Inventory Optimization and Multi-Echelon), Module 07 (Time Series Forecasting), Module 04 (Statistics & Probability)
> Version 2026.5

---

## 1. Why this topic exists

Standard inventory theory rests on an unspoken, dangerous premise: inventory is permanent. Classic models—from the Economic Order Quantity (EOQ) to continuous-review $(s, Q)$ and periodic-review $(R, S)$ base-stock policies—treat an unsold unit sitting in a warehouse rack or retail shelf as an asset that incurs only a capital holding charge $h$ per period. Under this classical assumption, if demand drops unexpectedly, inventory simply dwells in storage until future demand arrives to consume it. The only operational risk is carrying cost and stockout penalty.

In fresh food, dairy, processed meats, cut produce, chilled prepared meals, craft brewing, pharmaceuticals, and blood banking, that premise is physically false. Inventory is governed by a biochemical countdown: a maximum usable shelf life $m$. Unsold units do not simply wait; they undergo enzymatic degradation, microbial proliferation, lipid oxidation, desiccation, and organoleptic decay. When the countdown reaches zero, the product transforms instantaneously from a high-margin commercial asset into a hazardous waste liability that incurs direct disposal fees, haulage charges, regulatory reporting overhead, and complete write-offs.

Managing perishable inventory introduces two structural conflicts that standard enterprise resource planning (ERP) systems cannot solve:

1. **The Two-Master Depletion Conflict:** Operations, logistics, and quality assurance teams strive to enforce **FEFO** (First Expired, First Out) to minimize outdating and shrinkage. However, retail consumers and clinical staff display aggressive **LIFO** (Last In, First Out) or "cherry-picking" behavior. Shoppers reach to the back of refrigerated dairy decks or produce cases to select cartons with the latest date code, leaving older units to dwell on the front rail until they expire. This behavioral friction breaks standard inventory rotation, creating a non-linear spike in store-level write-offs.
2. **Contractual Remaining Shelf Life (RSL) Covenants:** Across consumer packaged goods (CPG) supply chains, retailers protect their commercial margins by imposing rigid **Minimum Remaining Shelf Life (MRSL)** contractual covenants. Major grocery retailers (e.g., Walmart, Tesco, Carrefour, Kroger) contractually demand that food manufacturers deliver products with at least 75% (and in some categories 80% to 85%) of total shelf life remaining upon receipt at the retailer's central distribution center (DC). If a production run has a 30-day total shelf life, a pallet arriving at the DC loading dock on Day 8 has 22 days remaining (73.3% RSL). The automated Warehouse Management System (WMS) triggers a gate rejection. The manufacturer faces three punitive choices: pay emergency reverse-logistics freight to recall the load, sell it to secondary discount liquidators at an 80% discount, or pay the retailer a disposal fee.

When supply chain teams track inventory as a scalar quantity (e.g., "10,000 units on hand") rather than as an **age-structured vector**, their safety stock calculations fail catastrophically. Two distribution centers holding identical quantities of 5,000 units have completely different financial solvency if DC 1 holds fresh stock with 25 days remaining, while DC 2 holds stock with 3 days remaining that will fail retail receipt gates tomorrow. Planners who fail to model the age distribution of inventory systematically overstate operational availability, underestimate spoilage write-offs, and trigger supply chain bullwhips driven by batch expirations.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Formulate** the state vector and transition matrices of an age-structured inventory system with fixed shelf life $m$ under periodic review.
2. **Derive and evaluate** the Nahmias-Fries recursive inventory equations to compute cohort depletion, safety stock requirements, and expected period-by-period outdating volume.
3. **Calculate** the exact outdating probability for any inbound order cohort given lead time, shelf life, and stochastic demand distributions.
4. **Model and quantify** the operational and financial impact of consumer cherry-picking behavior (mixed $\alpha$-FEFO / $(1-\alpha)$-LIFO depletion) versus strict operational FEFO.
5. **Structure and audit** contractual Remaining Shelf Life (RSL) and Minimum Remaining Shelf Life (MRSL) covenants across manufacturing, central DC, regional cross-dock, and retail store tiers.
6. **Design** multi-echelon velocity-based age allocation rules that direct aging inventory cohorts to high-turn retail nodes to prevent downstream gate rejections.
7. **Formulate** dynamic markdown and shelf rotation policies that optimize the trade-off between salvage recovery margin, cannibalization of full-price stock, and landfill disposal fees.
8. **Specify** the technical integration architecture required for automated FEFO execution across ERP, WMS, and GS1-128 / 2D Digital Link barcode data carriers.

---

## 3. The core mental model

To control perishable inventory, you must abandon scalar stock counting and adopt an **age pipeline state model**. Inventory is not a static pool; it is a discrete queue of cohorts advancing toward an expiration boundary at the speed of calendar time.

```text
                                  THE PERISHABLE PIPELINE
                                  
  Inbound Supply (Qt)             Remaining Shelf Life (Periods Remaining)             Outdates (Wt)
  ====================> [ Bucket m-1 ] -> [ Bucket m-2 ] -> ... -> [ Bucket 1 ] ======> [ Bio-Waste /
       (Age = 0)         Fresh Stock      Mid-Life Stock          Critical Stock          Disposal ]
                              |                 |                      |
                              |                 |                      |
                              +--------+--------+----------+-----------+
                                       |                   |
                                       |                   |
                                       v                   v
                             [ Customer LIFO ]     [ Customer FEFO ]
                              (Cherry-Pickers)      (Compliant Flow)
                             (Pulls from Left)     (Pulls from Right)
                                       \                   /
                                        \                 /
                                         v               v
                                   =============================
                                     Fulfilled Retail Sales (Dt)
```

### 3.1 The Causal Mechanism of Perishable Flow

The lifecycle of perishable stock unfolds across three interconnected layers:

1. **The Physical Aging Engine:** Every period (day, shift, or review cycle), calendar time advances by $\Delta t = 1$. Every unit of inventory on hand automatically transitions from remaining shelf life bucket $j$ to bucket $j-1$. This movement is deterministic and irreversible. No operational action can halt the aging clock unless the product undergoes secondary processing (e.g., flash freezing, canning, or pasteurization), which changes the stock keeping unit (SKU) identity.
2. **The Depletion Arbitrator:** Total realized customer demand $D_t$ arrives at the inventory pool. The system must decide—or the physical customer determines—which age cohort satisfies which slice of demand:
   - Under **Strict FEFO**, demand is satisfied greedily from the oldest available cohort ($j=1$, then $j=2$, up to $j=m-1$). This minimizes the probability that any unit reaches age $m$.
   - Under **Customer LIFO (Cherry-Picking)**, consumers inspect date stamps and pull units greedily from the freshest available cohort ($j=m-1$, then $j=m-2$, down to $j=1$). Old units are bypassed, stagnating on the shelf until their remaining life reaches zero.
   - Under **Mixed $\alpha$-Depletion**, a proportion $\alpha \in [0, 1]$ of demand consumes according to FEFO, while $(1-\alpha)$ consumes according to LIFO.
3. **The Multi-Echelon Gatekeeper (RSL Covenants):** Inventory does not flow freely between supply chain tiers. At each node boundary—factory gate, central distribution center (CDC), regional distribution center (RDC), and retail store backroom—contractual MRSL thresholds act as selective one-way filters:

```text
  MANUFACTURER            CENTRAL DC               RETAIL STORE             CONSUMER
  [ Total Life ] ======> [ MRSL >= 75% ] =======> [ MRSL >= 50% ] ======> [ RSL >= 10% ]
   m = 30 Days             Gate: >= 23 Days         Gate: >= 15 Days         Threshold: >= 3 Days
                           < 23 Days: REJECT        < 15 Days: REJECT        < 3 Days: Unsold / Spoiled
```

If upstream transit delays or warehouse dwelling cause a lot to cross a contractual threshold prior to arrival, the receiving node rejects the shipment. The inventory becomes stranded upstream, triggering write-downs even though the product is biologically safe and edible.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Operational Misuse | Measurement Unit / Standard |
|---|---|---|---|
| **Fixed Shelf Life ($m$)** | The maximum duration in discrete time periods from packaging/release to the mandatory expiration threshold. | Confusing total shelf life with consumer-usable shelf life after supply chain transit. | Integer periods (days, weeks); $\mathbb{Z}^+$. |
| **Remaining Shelf Life (RSL)** | The time remaining between the current timestamp $t$ and the expiration date $T_{\text{exp}}$: $\text{RSL} = T_{\text{exp}} - t$. | Calculating RSL against manufacturing date instead of harvest or packaging date. | Days, hours; $[T]$. |
| **Proportionate RSL ($\text{RSL}\%$)** | The ratio of remaining shelf life to total original shelf life: $\text{RSL}\% = \frac{T_{\text{exp}} - t}{m} \times 100\%$. | Ignoring elapsed dwell time in supplier storage when quoting RSL to commercial buyers. | Percentage ($0.0\% - 100.0\%$). |
| **Minimum RSL (MRSL)** | The contractual threshold of $\text{RSL}\%$ below which a receiving facility will issue a formal gate rejection. | Treating MRSL as a negotiable guideline rather than an automated WMS hard-stop. | Percentage covenant (e.g., $75\%$ to DC, $50\%$ to Store). |
| **Outdating / Spoilage ($W_t$)** | Units of inventory whose remaining shelf life reaches zero in period $t$ without being consumed by demand. | Categorizing outdates as unexplained inventory shrinkage or physical theft. | Physical units or currency at cost ($C \cdot W_t$). |
| **Nahmias State Vector ($\mathbf{x}_t$)** | The $(m-1)$-dimensional vector $(x_{1,t}, x_{2,t}, \dots, x_{m-1,t})$ representing units on hand by remaining shelf life. | Aggregating all ages into a single total inventory count ($I_{\text{tot}} = \sum x_i$) for replenishment. | Vector of counts in $\mathbb{R}^{m-1}_+$ or $\mathbb{N}^{m-1}$. |
| **FEFO Depletion** | An inventory issuing rule where goods with the earliest expiration date are allocated or sold first. | Assuming WMS FEFO logic guarantees physical store-shelf FEFO compliance by consumers. | Operational policy rule; boolean or categorical. |
| **LIFO Depletion (Cherry-Picking)** | A customer purchasing behavior where consumers deliberately select items with the latest expiration date. | Blaming store clerks for stock rotation failure when consumer cherry-picking drove outdating. | Behavioral pattern; parameter $(1-\alpha)$. |
| **Depletion Compliance ($\alpha$)** | The fraction of total demand satisfied under FEFO order; $(1-\alpha)$ is satisfied under LIFO order. | Assuming $\alpha = 1.0$ in retail grocery environments (empirical $\alpha$ is often $0.40 - 0.70$). | Dimensionless probability $\alpha \in [0, 1]$. |
| **Gate Rejection** | The formal physical refusal by a receiving warehouse to unload a shipment due to MRSL violation or cold chain break. | Recording a gate rejection as a delivery cancellation rather than a vendor chargeback event. | Binary event ($0/1$); EDI 861 transaction code. |
| **Stagnation Trap** | A failure mode where old inventory on the shelf rail is bypassed by cherry-picking consumers until 100% expires. | Believing that aging inventory will eventually sell without price markdowns or shelf intervention. | Operational failure state. |
| **Dynamic Markdown** | A structured price reduction triggered as a function of remaining shelf life to accelerate depletion before outdating. | Applying static end-of-day markdowns after the product has visually or organoleptically degraded. | Currency discount ($\Delta p(j)$) or markdown percentage. |
| **Virtual Depletion Horizon** | The projected time required for current forecasted demand to exhaust inventory cohort $j$ under FEFO. | Comparing stock to historical sales rather than forward-looking perishable forecast. | Time periods; $[T]$. |
| **Outdating Cost ($\theta$)** | The total economic penalty per unit outdated, including purchase cost, disposal fees, and labor: $\theta = c + c_d - s$. | Omitting reverse logistics haulage and municipal tipping fees from the cost of waste. | Currency per unit ($\$ / \text{unit}$). |
| **GS1 Application Identifier 17** | The GS1 standardized data prefix indicating an item's expiration date formatted as `YYMMDD`. | Printing human-readable dates without GS1-128 or 2D DataBar barcode encoding. | Data standard syntax `(17)YYMMDD`. |

---

## 5. The mechanics, step by step

Managing an age-structured perishable inventory system requires a rigorous sequence of mathematical updates, demand allocations, gate checks, and replenishment calculations.

```text
  STEP 1: STATE VECTOR DEFINITION & RECORDING
          Record x_t = [x_1, x_2, ..., x_{m-1}] across all age cohorts.
                                  |
                                  v
  STEP 2: INBOUND RECEPTION & MRSL GATE SCREENING
          Apply RSL covenant: If RSL_received < MRSL, REJECT. Else add Q_t to cohort m-1.
                                  |
                                  v
  STEP 3: DEMAND ARBITRATION (FEFO vs. CHERRY-PICKING)
          Partition demand D_t into FEFO fraction (alpha) and LIFO fraction (1 - alpha).
                                  |
                                  v
  STEP 4: COHORT DEPLETION DYNAMICS
          Deplete oldest buckets with alpha * D_t; deplete newest with (1 - alpha) * D_t.
                                  |
                                  v
  STEP 5: OUTDATING CALCULATION & SCRAP WRITEOFF
          Compute W_t = max(0, x_{1,t} - Fulfilled_Oldest). Record disposal cost.
                                  |
                                  v
  STEP 6: TIME ADVANCEMENT & STATE AGING
          Shift remaining stock: x_{j-1, t+1} = Remaining(x_{j,t}). Calendar clock advances.
                                  |
                                  v
  STEP 7: PERISHABLE BASE-STOCK REPLENISHMENT
          Calculate Nahmias effective inventory position; issue replenishment order Q_{t+1}.
```

### Step 1: Definition of the State Space and Cohort Tracking

Let the product have a maximum shelf life of $m$ discrete review periods. At the beginning of period $t$, prior to demand realization and order arrival, the inventory state is completely characterized by the vector:

$$\mathbf{x}_t = \begin{bmatrix} x_{1,t} \\ x_{2,t} \\ \vdots \\ x_{m-1,t} \end{bmatrix}$$

where $x_{j,t} \ge 0$ denotes the quantity of inventory on hand having exactly $j$ periods of usable life remaining before it outdates. Stock in cohort $j=1$ will expire at the end of period $t$ if not consumed. Stock in cohort $j=m-1$ is fresh inventory with maximum remaining life. 

Define cumulative inventory available with remaining shelf life up to $k$ periods as:

$$S_{k,t} = \sum_{j=1}^{k} x_{j,t} \quad \text{for } k = 1, 2, \dots, m-1$$

The total physical inventory on hand is $I_t = S_{m-1,t} = \sum_{j=1}^{m-1} x_{j,t}$.

### Step 2: Inbound Reception and MRSL Covenant Enforcement

When an inbound replenishment shipment $Q_t$ arrives from an upstream supplier or plant, the receiving node inspects the date codes. Let the total original shelf life be $m_{\text{orig}}$ days, and let the remaining days upon arrival be $R_{\text{arr}}$.

1. Compute the incoming Proportionate RSL:
   $$\text{RSL}_{\text{in}} = \frac{R_{\text{arr}}}{m_{\text{orig}}}$$
2. Compare against the contractual MRSL covenant ($\text{MRSL}_{\text{covenant}}$):
   $$\text{Decision} = \begin{cases} \text{Accept and Induct into WMS}, & \text{if } \text{RSL}_{\text{in}} \ge \text{MRSL}_{\text{covenant}} \\ \text{Issue Gate Rejection (EDI 861)}, & \text{if } \text{RSL}_{\text{in}} < \text{MRSL}_{\text{covenant}} \end{cases}$$
3. If accepted, the shipment enters the state vector at cohort $k_{\text{in}} = \lceil R_{\text{arr}} / \Delta t \rceil$, where $\Delta t$ is the duration of a review period. If inbound lead time is zero and goods are fresh, $k_{\text{in}} = m-1$.

### Step 3: Demand Arbitration under FEFO and Customer Cherry-Picking

Let total realized customer demand in period $t$ be $D_t \ge 0$, drawn from probability density function $f(d)$ and cumulative distribution $F(d)$. 

To account for realistic retail behavior, demand is partitioned using compliance parameter $\alpha \in [0, 1]$:
- Compliant demand (FEFO shoppers): $D_t^{\text{FEFO}} = \alpha D_t$
- Cherry-picking demand (LIFO shoppers): $D_t^{\text{LIFO}} = (1 - \alpha) D_t$

If $\alpha = 1.0$, the system operates under pure operational FEFO. If $\alpha = 0.0$, the system collapses into pure customer LIFO.

### Step 4: The Nahmias-Fries Cohort Depletion Dynamics

Under pure FEFO ($\alpha = 1.0$), demand consumes inventory sequentially from the oldest cohort ($j=1$) toward the newest ($j=m-1$). 

The amount of demand allocated to cohort $j$ depends on whether cumulative demand exceeds cumulative stock prior to cohort $j$:

$$\text{Depletion of cohort } j = \max\left(0, \min\left(x_{j,t}, D_t - S_{j-1,t}\right)\right)$$

with $S_{0,t} = 0$.

Under mixed depletion ($\alpha < 1.0$), the arbitration occurs simultaneously at both ends of the vector:
1. $D_t^{\text{FEFO}}$ attacks cohort 1 upwards:
   $$\text{Remaining oldest cohort stock: } \tilde{x}_{1,t} = \max\left(0, x_{1,t} - \alpha D_t\right)$$
   Unsatisfied FEFO demand cascades to cohort 2, cohort 3, and so on.
2. $D_t^{\text{LIFO}}$ attacks cohort $m-1$ downwards:
   $$\text{Remaining freshest cohort stock: } \tilde{x}_{m-1,t} = \max\left(0, x_{m-1,t} - (1-\alpha) D_t\right)$$
   Unsatisfied LIFO demand cascades down to cohort $m-2, m-3$, and so on.
3. If demand from either stream cannot be satisfied from its preferred side, it consumes from the remaining interior cohorts. If total demand $D_t > I_t$, stockout occurs:
   $$\text{Unmet Demand (Lost Sales): } U_t = \max(0, D_t - I_t)$$

### Step 5: Outdating Calculation and Waste Materialization

At the end of period $t$, any stock remaining in cohort $j=1$ that was not consumed by demand reaches the end of its life and outdates.

Under pure FEFO ($\alpha = 1.0$), outdating $W_t$ is given exactly by:

$$W_t = \max\left(0, x_{1,t} - D_t\right) = [x_{1,t} - D_t]^+$$

The expected outdating $\mathbb{E}[W_t \mid x_{1,t}]$ is:

$$\mathbb{E}[W_t \mid x_{1,t}] = \int_{0}^{x_{1,t}} (x_{1,t} - u) f(u) \, du = x_{1,t} F(x_{1,t}) - \int_{0}^{x_{1,t}} u f(u) \, du$$

Under mixed depletion ($\alpha < 1.0$), outdating increases because cohort 1 is exposed only to a fraction of total demand:

$$W_t(\alpha) = \max\left(0, x_{1,t} - D_t^{\text{allocated to cohort 1}}\right) \ge \max\left(0, x_{1,t} - D_t\right)$$

The economic waste cost realized in period $t$ is:

$$\text{Cost}_{\text{waste}} = \theta \cdot W_t = (c + c_d - s) \cdot W_t$$

where $c$ is acquisition cost, $c_d$ is disposal/handling fee per unit, and $s$ is salvage recovery value ($s=0$ for landfill).

### Step 6: Time Advancement and State Aging

After depletion and outdating removal, remaining inventory advances one step along the aging pipeline into period $t+1$.

For cohorts $j = 2, 3, \dots, m-1$:

$$x_{j-1, t+1} = \text{Surviving stock from cohort } j \text{ at end of period } t$$

Under pure FEFO, utilizing the Nahmias formulation based on cumulative inventory:

$$x_{j-1, t+1} = \max\left(0, S_{j,t} - D_t\right) - \max\left(0, S_{j-1,t} - D_t\right) \quad \text{for } j = 2, \dots, m-1$$

The newly received order $Q_t$ (assuming zero replenishment lead time $L=0$) fills the freshest cohort:

$$x_{m-1, t+1} = Q_t$$

If replenishment involves a lead time of $L$ periods, $Q_t$ ages during transit or warehouse staging, entering the active state vector at an age reduced by $L$.

### Step 7: Nahmias Perishable Base-Stock Replenishment

In standard inventory theory with non-perishable goods, an order-up-to policy sets order quantity $Q_t = S - I_t$. In perishable inventory, this policy over-orders because it ignores the fact that a fraction of the current inventory will outdate before future demand can reach it.

Nahmias (1975, 1982) demonstrated that the optimal replenishment policy under fixed shelf life requires discounting older cohorts by their probability of outdating.

Let $B_j(\mathbf{x}_t)$ represent the expected quantity of inventory currently on hand in cohort $j$ that will outdate before reaching customer demand. The **Effective Inventory Position** $IP_t^{\text{eff}}$ is:

$$IP_t^{\text{eff}} = \sum_{j=1}^{m-1} \left( x_{j,t} - \mathbb{E}[\text{Outdating from cohort } j] \right) + \text{On Order} - \text{Backorders}$$

The replenishment order is then:

$$Q_t = \max\left(0, S^* - IP_t^{\text{eff}}\right)$$

where $S^*$ is calibrated to balance holding cost $h$, lost sales margin $p - c$, and outdating penalty $\theta$.


### Step 8: Closed-Form Outdating Convolutions for Low-Order Lifetimes

For items with short lifetimes ($m=2$ and $m=3$ review periods), Steven Nahmias (1975) derived explicit closed-form convolutions relating the inbound order quantity $Q_t$ to the expected outdating volume $m$ periods later.

For a commodity with lifetime $m=2$ (two-period maximum life), stock ordered in period $t$ ($Q_t$) arrives and survives into period $t+1$ as cohort $j=1$. It outdates at the end of period $t+1$ if cumulative demand over periods $t$ and $t+1$ fails to exhaust starting stock plus $Q_t$.

Let $x_{1,t}$ be starting inventory with 1 period remaining, and let $Q_t$ be the order placed at $t$. Under pure FEFO, Nahmias proved that expected outdates at the conclusion of period $t+1$ are given by:

$$\mathbb{E}[W_{t+1}] = \int_{0}^{Q_t} F(Q_t + x_{1,t} - u) f(u) \, du$$

where $F(\cdot)$ is the cumulative distribution function and $f(\cdot)$ is the probability density function of single-period demand.

For $m=3$ periods, the convolution accounts for the two intervening demand realizations $D_t$ and $D_{t+1}$, yielding the double integral:

$$\mathbb{E}[W_{t+2}] = \int_{0}^{Q_t} F^{(2)}(Q_t + x_{1,t} + x_{2,t} - u) f(u) \, du$$

where $F^{(2)}$ is the 2-fold convolution of the demand distribution. These closed-form expressions prove that outdating is strictly convex and increasing in $Q_t$, establishing that the marginal cost of inventory over-ordering is non-linear and sharply escalating in perishable supply chains.

### Step 9: Arrhenius Kinetic Adjustment for Temperature-Abused Inbound Stock

In real cold chains, shelf life is not an invariant calendar property; it is an integrated thermal function. When cold-storage sensors detect temperature abuse, the supply chain cannot use the nominal calendar date code.

The biochemical degradation rate $k(T)$ follows the Arrhenius equation:

$$k(T) = A \cdot \exp\left(-\frac{E_a}{R \cdot T_K}\right)$$

where $E_a$ is the activation energy of the primary degradation reaction ($\text{J/mol}$), $R = 8.314 \, \text{J/(mol}\cdot\text{K)}$ is the universal gas constant, $T_K$ is absolute temperature in Kelvin, and $A$ is the pre-exponential kinetic constant.

When a shipment encounters an elevated temperature excursion $T_{\text{excurs}}$ for duration $\Delta t_{\text{excurs}}$, the effective shelf life consumed during transit is accelerated by the thermal acceleration ratio:

$$A_T = \frac{k(T_{\text{excurs}})}{k(T_{\text{standard}})} = \exp\left( \frac{E_a}{R} \left( \frac{1}{T_{\text{standard}}} - \frac{1}{T_{\text{excurs}}} \right) \right)$$

The adjusted remaining shelf life $R_{\text{arr}}^{\text{adj}}$ upon arrival at the DC gate is:

$$R_{\text{arr}}^{\text{adj}} = m_{\text{orig}} - \left( t_{\text{standard\_transit}} + A_T \cdot \Delta t_{\text{excurs}} \right)$$

If $R_{\text{arr}}^{\text{adj}} / m_{\text{orig}} < \text{MRSL}_{\text{covenant}}$, the WMS triggers a gate rejection even if the physical calendar date code on the carton label appears compliant.
---

## 6. Worked example

We trace an end-to-end numerical execution of an age-structured inventory model for a high-velocity chilled dairy SKU: **Artisan Greek Yogurt 500g**.

### 6.1 Baseline Parameters and Operating Assumptions

- **Review Period ($\Delta t$):** 3 days.
- **Maximum Usable Shelf Life:** $m = 4$ review periods ($4 \times 3 = 12$ days total shelf life).
- **Unit Financials:**
  - Purchase cost ($c$): $\$2.00$ per unit.
  - Retail selling price ($p$): $\$4.50$ per unit.
  - Salvage value ($s$): $\$0.00$ (no secondary market).
  - Landfill disposal fee ($c_d$): $\$0.30$ per unit.
  - Total outdating penalty: $\theta = c + c_d - s = \$2.00 + \$0.30 - \$0.00 = \$2.30$ per unit.
  - Holding cost ($h$): $\$0.10$ per unit per 3-day period.
  - Lost sales penalty (unmet demand): $p - c = \$4.50 - \$2.00 = \$2.50$ per unit.
- **Contractual RSL Covenant:** Retailer DC requires $\text{MRSL} \ge 75\%$ ($12 \times 0.75 = 9$ days, or 3 review periods).
- **Replenishment Policy:** Order-up-to target $S^* = 100$ units. Orders placed at the start of period $t$ arrive immediately ($L=0$) fresh from the plant with full shelf life ($m-1 = 3$ periods remaining).
- **Customer Depletion Regimes:**
  - **Scenario A:** Strict FEFO compliance ($\alpha = 1.0$).
  - **Scenario B:** Cherry-picking behavior ($\alpha = 0.50$, meaning $50\%$ of demand pulls from the freshest cohort, and $50\%$ pulls from the oldest).

### 6.2 Initial Inventory State at $t=1$

The warehouse opens Period 1 with an aging stock profile:
- Cohort 1 (1 period remaining = 3 days to expiry): $x_{1,1} = 25$ units.
- Cohort 2 (2 periods remaining = 6 days to expiry): $x_{2,1} = 30$ units.
- Cohort 3 (3 periods remaining = 9 days to expiry): $x_{3,1} = 20$ units.
- Total stock on hand: $I_1 = 25 + 30 + 20 = 75$ units.

An order arrives to reach order-up-to target $S^* = 100$:
$$Q_1 = 100 - 75 = 25 \text{ units}$$
These 25 units enter the freshest cohort (Cohort 3).
Updated Cohort 3 stock: $x_{3,1} = 20 + 25 = 45$ units.
Vector before demand: $\mathbf{x}_1 = [25, 30, 45]^T$, total $I_1 = 100$ units.

Realized demand in Period 1: $D_1 = 40$ units.

---

### 6.3 Trace of Period 1 Execution

#### Scenario A: Strict FEFO ($\alpha = 1.0$)

1. **Demand Allocation:**
   - Demand $D_1 = 40$ attacks Cohort 1 ($x_{1,1} = 25$ units).
   - Cohort 1 is completely depleted: $25 - 25 = 0$ units remain.
   - Remaining demand to satisfy: $40 - 25 = 15$ units.
   - Remaining demand attacks Cohort 2 ($x_{2,1} = 30$ units).
   - Cohort 2 depletion: $15$ units consumed; $30 - 15 = 15$ units remain.
   - Cohort 3 ($x_{3,1} = 45$ units) is untouched: $45$ units remain.
2. **Outdating at End of Period 1 ($W_1$):**
   - Cohort 1 had 0 units remaining: $W_1 = 0$ units outdated.
3. **Aging Transition to Period 2:**
   - Cohort 2 stock moves to Cohort 1: $x_{1,2} = 15$ units.
   - Cohort 3 stock moves to Cohort 2: $x_{2,2} = 45$ units.
   - Cohort 3 starts at 0 before new order: $x_{3,2} = 0$.
   - Inventory on hand before replenishment: $I_2 = 15 + 45 + 0 = 60$ units.
4. **Period 1 Financial Summary (Scenario A):**
   - Revenue: $40 \times \$4.50 = \$180.00$.
   - Cost of Goods Sold: $40 \times \$2.00 = \$80.00$.
   - Holding Cost: $60 \times \$0.10 = \$6.00$.
   - Waste Cost: $0 \times \$2.30 = \$0.00$.
   - Net Operating Profit: $\$180.00 - \$80.00 - \$6.00 - \$0.00 = \$94.00$.

---

#### Scenario B: Customer Cherry-Picking ($\alpha = 0.50$)

1. **Demand Partitioning:**
   - Total demand $D_1 = 40$ units.
   - FEFO demand stream: $D_1^{\text{FEFO}} = 0.50 \times 40 = 20$ units.
   - LIFO demand stream: $D_1^{\text{LIFO}} = 0.50 \times 40 = 20$ units.
2. **Depletion Execution:**
   - **FEFO stream (20 units)** attacks Cohort 1 ($x_{1,1} = 25$ units):
     - Cohort 1 consumed: $20$ units.
     - Cohort 1 remaining: $25 - 20 = 5$ units.
   - **LIFO stream (20 units)** attacks Cohort 3 ($x_{3,1} = 45$ units):
     - Cohort 3 consumed: $20$ units.
     - Cohort 3 remaining: $45 - 20 = 25$ units.
   - Cohort 2 ($x_{2,1} = 30$ units) is untouched: $30$ units remain.
3. **Outdating at End of Period 1 ($W_1$):**
   - Cohort 1 had 5 unconsumed units reaching age limit $m$:
     $$W_1 = 5 \text{ units outdated!}$$
4. **Aging Transition to Period 2:**
   - Surviving Cohort 2 stock moves to Cohort 1: $x_{1,2} = 30$ units.
   - Surviving Cohort 3 stock moves to Cohort 2: $x_{2,2} = 25$ units.
   - Inventory on hand before replenishment: $I_2 = 30 + 25 + 0 = 55$ units.
5. **Period 1 Financial Summary (Scenario B):**
   - Revenue: $40 \times \$4.50 = \$180.00$.
   - Cost of Goods Sold: $40 \times \$2.00 = \$80.00$.
   - Holding Cost: $55 \times \$0.10 = \$5.50$.
   - Waste Penalty: $5 \times \$2.30 = \$11.50$.
   - Net Operating Profit: $\$180.00 - \$80.00 - \$5.50 - \$11.50 = \$83.00$.
   - **Profit Penalty from Cherry-Picking:** $\$94.00 - \$83.00 = \$11.00$ ($11.7\%$ margin reduction in a single period).

---

### 6.4 Trace of Period 2 Execution under Sustained Dynamics

To observe the multi-period compounding of the "stagnation trap," let Period 2 demand soften to $D_2 = 30$ units.

#### Scenario A (FEFO Continued)

- Replenishment: $Q_2 = 100 - 60 = 40$ units.
- State vector before demand: $\mathbf{x}_2 = [15, 45, 40]^T$, total $I_2 = 100$.
- Demand $D_2 = 30$ attacks Cohort 1 (15 units) $\to 0$ remain.
- Remaining 15 demand attacks Cohort 2 (45 units) $\to 30$ remain.
- Outdating $W_2 = 0$.
- Aging transition: $x_{1,3} = 30$, $x_{2,3} = 40$, $I_3 = 70$.
- Waste Cost: $\$0.00$.

#### Scenario B (Cherry-Picking Continued, $\alpha = 0.50$)

- Replenishment: $Q_2 = 100 - 55 = 45$ units.
- State vector before demand: $\mathbf{x}_2 = [30, 25, 45]^T$, total $I_2 = 100$.
- Demand $D_2 = 30$ partitioned:
  - FEFO stream ($15$ units) attacks Cohort 1 ($30$ units) $\to 30 - 15 = 15$ units remain.
  - LIFO stream ($15$ units) attacks Cohort 3 ($45$ units) $\to 45 - 15 = 30$ units remain.
- Outdating $W_2 = 15$ units outdated!
- Waste Cost: $15 \times \$2.30 = \$34.50$.
- Aging transition: $x_{1,3} = 25$ (from Cohort 2), $x_{2,3} = 30$ (from Cohort 3), $I_3 = 55$.
- Cumulative 2-period waste under Cherry-Picking: $5 + 15 = 20$ units ($\$46.00$ lost to disposal) versus $0$ units under FEFO.

---

### 6.5 Trace of Periods 3 and 4 under Sustained Dynamics

To observe how the stagnation trap behaves under varying demand velocity, we continue the simulation across Periods 3 and 4.

#### Period 3 Execution (High Demand Surge: $D_3 = 50$ units)

1. **Scenario A (Strict FEFO):**
   - Replenishment to target $S^* = 100$: On hand is $15 + 45 = 60$ in ending P2, so starting P3 stock before order is $x_{1,3} = 30, x_{2,3} = 40, x_{3,3} = 0$. Total on hand $I_3 = 70$.
   - Order placed and received: $Q_3 = 100 - 70 = 30$ units into Cohort 3.
   - State before demand: $\mathbf{x}_3 = [30, 40, 30]^T$, total $I_3 = 100$.
   - Demand $D_3 = 50$ attacks Cohort 1 ($30$ units) $\to 0$ remain.
   - Remaining demand ($20$ units) attacks Cohort 2 ($40$ units) $\to 20$ remain.
   - Cohort 3 ($30$ units) is untouched $\to 30$ remain.
   - Outdating $W_3 = 0$. Aging transition to P4: $x_{1,4} = 20, x_{2,4} = 30, I_4 = 50$.

2. **Scenario B (Cherry-Picking, $\alpha = 0.50$):**
   - Starting stock before order: $x_{1,3} = 25, x_{2,3} = 30, x_{3,3} = 0$. Total on hand $I_3 = 55$.
   - Replenishment: $Q_3 = 100 - 55 = 45$ units into Cohort 3.
   - State before demand: $\mathbf{x}_3 = [25, 30, 45]^T$, total $I_3 = 100$.
   - Demand $D_3 = 50$ partitioned: FEFO stream = 25 units; LIFO stream = 25 units.
   - FEFO stream consumes 25 from Cohort 1 ($x_{1,3} = 25$) $\to 0$ remain.
   - LIFO stream consumes 25 from Cohort 3 ($x_{3,3} = 45$) $\to 20$ remain.
   - Cohort 2 ($30$ units) is completely untouched!
   - Outdating $W_3 = 0$.
   - Aging transition to P4: Surviving Cohort 2 ($30$ units) advances to Cohort 1; surviving Cohort 3 ($20$ units) advances to Cohort 2. Ending stock on hand $I_4 = 30 + 20 = 50$.

#### Period 4 Execution (Moderate Demand: $D_4 = 30$ units)

1. **Scenario A (Strict FEFO):**
   - Replenishment: $Q_4 = 100 - 50 = 50$ units.
   - State before demand: $\mathbf{x}_4 = [20, 30, 50]^T$, total $I_4 = 100$.
   - Demand $D_4 = 30$ consumes 20 from Cohort 1 $\to 0$ remain.
   - Remaining 10 demand consumes from Cohort 2 $\to 30 - 10 = 20$ remain. Cohort 3 untouched ($50$ remain).
   - Outdating $W_4 = 0$.
   - Aging transition: $x_{1,5} = 20, x_{2,5} = 50, I_5 = 70$.

2. **Scenario B (Cherry-Picking, $\alpha = 0.50$):**
   - Replenishment: $Q_4 = 100 - 50 = 50$ units.
   - State before demand: $\mathbf{x}_4 = [30, 20, 50]^T$, total $I_4 = 100$.
   - Notice the trap: Cohort 1 holds 30 units that bypassed demand during Period 3!
   - Demand $D_4 = 30$ partitioned: FEFO stream = 15 units; LIFO stream = 15 units.
   - FEFO stream consumes 15 from Cohort 1 ($30$ units) $\to 30 - 15 = 15$ remain.
   - LIFO stream consumes 15 from Cohort 3 ($50$ units) $\to 50 - 15 = 35$ remain.
   - Cohort 2 ($20$ units) is completely untouched.
   - Outdating at end of Period 4:
     $$W_4 = 15 \text{ units outdated!}$$
   - Waste Cost: $15 \times \$2.30 = \$34.50$.
   - Aging transition: $x_{1,5} = 20, x_{2,5} = 35, I_5 = 55$.

---

### 6.6 Four-Period Operational and Financial Scorecard

The cumulative four-period comparison demonstrates the compounding structural destruction wrought by unmanaged customer cherry-picking:

| Metric | Scenario A: Strict FEFO ($\alpha = 1.0$) | Scenario B: Cherry-Picking ($\alpha = 0.50$) | Variance ($\Delta$) | Economic Meaning |
|---|---|---|---|---|
| **Total Realized Demand** | 150 units | 150 units | 0 units | Identical customer foot traffic. |
| **Total Fulfilled Sales** | 150 units | 150 units | 0 units | 100% fill rate in both cases. |
| **Period 1 Outdates ($W_1$)** | 0 units | 5 units | +5 units | Initial cohort 1 leakage. |
| **Period 2 Outdates ($W_2$)** | 0 units | 15 units | +15 units | Stagnation trap begins. |
| **Period 3 Outdates ($W_3$)** | 0 units | 0 units | 0 units | High surge absorbed cohort 1. |
| **Period 4 Outdates ($W_4$)** | 0 units | 15 units | +15 units | Bypassed P3 cohort outdates. |
| **Cumulative Outdating** | **0 units** | **35 units** | **+35 units** | Massive landfill loss. |
| **Outdating Shrink Rate** | **0.0%** | **18.9%** | **+18.9%** | Share of procurement wasted. |
| **Gross Revenue** | $\$675.00$ | $\$675.00$ | $\$0.00$ | Fixed revenue at $\$4.50/\text{unit}$. |
| **Cost of Goods Sold** | $\$300.00$ | $\$300.00$ | $\$0.00$ | Cost of units sold at $\$2.00/\text{unit}$. |
| **Holding Cost ($h$)** | $\$25.00$ | $\$21.50$ | $-\$3.50$ | Slightly lower stock due to scrap. |
| **Waste Penalties ($\theta$)** | **$\$0.00$** | **$\$80.50$** | **$+\$80.50$** | Direct disposal and lost margin. |
| **Net Operating Profit** | **$\$350.00$** | **$\$273.00$** | **$-\$77.00$** | **22.0% profit destruction.** |

---

### 6.7 RSL Contractual Gate Rejection Calculation
Suppose the supplier attempts to deliver the $Q_2 = 45$ units to a retailer DC under an EDI 856 Advanced Shipping Notice. 

- Total product shelf life: 12 days.
- Contractual covenant: $\text{MRSL} \ge 75\%$ (minimum 9 days remaining).
- Case 1: Plant dispatches stock packaged yesterday (11 days remaining).
  $$\text{RSL} = \frac{11}{12} = 91.7\% \ge 75\% \implies \mathbf{ACCEPTED}$$
- Case 2: Due to trailer staging congestion at the supplier terminal, the load dwells for 4 days. Stock has 8 days remaining upon arrival at the DC gate:
  $$\text{RSL} = \frac{8}{12} = 66.7\% < 75\% \implies \mathbf{GATE\ REJECTION}$$
  The WMS rejects the pallet. The supplier incurs:
  - Return freight: $\$450.00$.
  - Secondary liquidator salvage at $\$0.50/\text{unit}$ (loss of $\$1.50/\text{unit} \times 45 = \$67.50$).
  - Retailer non-compliance chargeback: $\$250.00$.
  - Total incident cost: $\$767.50$ on a shipment worth $\$90.00$ in production cost.

---

## 7. Code, query, or calculation

The following production script implements the Nahmias age-structured inventory transition equations and simulates comparative performance across FEFO and customer cherry-picking regimes.

```python
"""Simulate age-structured perishable inventory under FEFO and mixed LIFO regimes."""
from dataclasses import dataclass


@dataclass(frozen=True)
class PerishableConfig:
    shelf_life: int  # m periods
    order_up_to: int  # S target
    cost: float
    price: float
    holding: float
    disposal: float
    salvage: float = 0.0

    @property
    def waste_penalty(self) -> float:
        return self.cost + self.disposal - self.salvage


def simulate_age_cohorts(
    cfg: PerishableConfig, demands: list[int], alpha: float
) -> dict[str, float]:
    """Run multi-period age-structured simulation under alpha-depletion policy."""
    m = cfg.shelf_life
    x = [0] * (m - 1)
    rev, cogs, waste, hold, outdates, lost = 0.0, 0.0, 0.0, 0.0, 0, 0

    def pull_stock(qty: int, reverse: bool) -> int:
        nonlocal rev, cogs
        indices = range(m - 2, -1, -1) if reverse else range(m - 1)
        for i in indices:
            take = min(x[i], qty)
            x[i] -= take
            qty -= take
            cogs += take * cfg.cost
            rev += take * cfg.price
            if qty == 0:
                break
        return qty

    for d in demands:
        x[m - 2] += max(0, cfg.order_up_to - sum(x))
        fefo_dem = int(round(alpha * d))
        rem = pull_stock(fefo_dem, reverse=False) + pull_stock(d - fefo_dem, reverse=True)
        if rem > 0:
            lost += pull_stock(rem, reverse=False)

        outdates += x[0]
        waste += x[0] * cfg.waste_penalty
        x = x[1:] + [0]
        hold += sum(x) * cfg.holding

    return {
        "profit": round(rev - cogs - waste - hold, 2),
        "outdates": outdates,
        "lost_sales": lost,
        "waste_cost": round(waste, 2),
    }


if __name__ == "__main__":
    cfg = PerishableConfig(4, 100, cost=2.0, price=4.5, holding=0.1, disposal=0.3)
    demands = [40, 30, 45, 35, 50, 40, 30, 45]
    fefo = simulate_age_cohorts(cfg, demands, alpha=1.0)
    cherry = simulate_age_cohorts(cfg, demands, alpha=0.5)
    print(f"Pure FEFO (alpha=1.0)   -> Profit: ${fefo['profit']}, Outdates: {fefo['outdates']}")
    print(f"Cherry-Pick (alpha=0.5) -> Profit: ${cherry['profit']}, Outdates: {cherry['outdates']}")
    assert fefo["outdates"] <= cherry["outdates"]
```

### Expected Output

```text
Pure FEFO (alpha=1.0)   -> Profit: $739.0, Outdates: 0
Cherry-Pick (alpha=0.5) -> Profit: $622.4, Outdates: 53
```
---

## 8. Data, standards, and regulatory context

Managing perishable inventory across multi-enterprise supply chains requires machine-readable identification standards, EDI transaction protocols, and compliance with statutory food safety regulations.

### 8.1 GS1 Identification Standards for Age-Structured Inventory

Traditional UPC/EAN barcodes (EAN-13, UPC-A) encode only the Global Trade Item Number (GTIN). They contain no serial number, batch identifier, or expiration date. When retail staff scan an EAN-13 barcode at checkout, the POS system records a sale but cannot determine the age cohort of the unit sold.

To execute FEFO and automated age rotation, industry relies on expanded GS1 data standards:

1. **GS1-128 and GS1 DataBar:** Linear barcodes capable of encoding secondary attributes alongside the GTIN using standardized **Application Identifiers (AIs)**:
   - `AI (01)`: Global Trade Item Number (14 digits).
   - `AI (10)`: Batch or Lot Number (up to 20 alphanumeric characters).
   - `AI (17)`: Expiration Date (`YYMMDD` format).
   - `AI (15)`: Best Before Date (`YYMMDD` format).
   - `AI (7003)`: Expiration Date and Time (for ultra-short shelf life foods, e.g., fresh sandwiches).
2. **GS1 2D DataMatrix and QR Code with GS1 Digital Link:** The global retail industry is actively migrating to 2D barcodes at point of sale (the "Sunrise 2027" initiative). A single 2D code scanned at retail POS parses the expiration date directly. If an expired unit is scanned, the POS register locks with an "Expired Product - Sale Prohibited" alert, eliminating the risk of selling out-of-date goods.

```text
  GS1-128 Barcode Syntax Example:
  (01)00012345678905(17)261115(10)LOT-B429A
  ├── AI 01: GTIN = 00012345678905
  ├── AI 17: Expiration Date = 2026-11-15 (Nov 15, 2026)
  └── AI 10: Lot Identifier = LOT-B429A
```

### 8.2 EDI Supply Chain Transaction Standards

B2B trade between food manufacturers, 3PL cold-storage providers, and grocery retailers relies on ANSI ASC X12 (North America) and UN/EDIFACT (Europe/Global) messaging to enforce RSL covenants before trucks arrive:

| EDI Standard | Message Type | Critical Segments for Perishable Inventory | Operational Function |
|---|---|---|---|
| **EDI 850** | Purchase Order | `PO1`, `SAC`, `MSG` | Transmits buyer's MRSL covenant requirement (e.g., "Must deliver $\ge 75\%$ shelf life"). |
| **EDI 856** | Advance Ship Notice (ASN) | `HL` (Hierarchy), `LIN`, `DTM` (Date/Time), `SN1` | Discloses exact pallet-by-pallet lot numbers and `DTM*036` (Expiration Date) prior to physical arrival. |
| **EDI 861** | Receiving Advice | `RCD`, `QTY`, `CDI` | Communicates receiving inspection results, accepted quantities, and formal gate rejections with reason codes. |
| **EDI 846** | Inventory Inquiry / Advice | `LIN`, `QTY`, `DTM` | Provides multi-echelon stock visibility broken down by expiration date cohorts across 3PL warehouses. |

Under best-practice WMS integration, when an EDI 856 ASN arrives, the retailer's system performs an **Automated Pre-Arrival MRSL Audit**:
$$\text{Projected Arrival RSL} = \frac{T_{\text{exp}} - T_{\text{scheduled\_dock}}}{T_{\text{total\_life}}}$$
If the projected RSL violates the purchase order terms, the system automatically flags the load for gate rejection before the carrier dispatches the tractor, eliminating unnecessary road miles and demurrage penalties.

### 8.3 Regulatory Frameworks for Date Labeling and Food Safety

Food shelf-life management is governed by overlapping national and international regulations:

1. **United States (FDA and USDA FSIS):**
   - Except for Infant Formula (mandated under 21 CFR 107.20 to display a "Use by" date backed by nutritional stability testing), federal U.S. law does not mandate open date labeling on food products.
   - USDA FSIS regulates meat, poultry, and egg products. Dates must include a month, day, and phrase explaining the meaning: "Best if Used By" indicates quality/freshness; "Use By" indicates peak quality before degradation.
   - **FSMA 204 (Food Traceability Rule):** Requires entities manufacturing, processing, packing, or holding foods on the Food Traceability List (FTL)—including fresh cheeses, leafy greens, shell eggs, and finfish—to maintain Key Data Elements (KDEs) across Critical Tracking Events (CTEs), including the lot code and expiration date linked to the receiving and shipping records.
2. **European Union (Regulation (EU) No 1169/2011):**
   - Imposes a strict legal distinction between **"Use by" (Date de péremption)** and **"Best before" (Date de durabilité minimale)**:
     - **"Use by" (Article 24):** Mandatory for foods which, from a microbiological point of view, are highly perishable and are therefore likely after a short period to constitute an immediate danger to human health. Selling or distributing food past its "Use by" date is an illegal criminal offense.
     - **"Best before":** Indicates date until which the food retains its specific properties under proper storage; product can legally be sold or donated past this date if safe.


### 8.4 GS1 EPCIS 2.0 Event Architecture for Lot-Level Shelf-Life Tracking

To achieve end-to-end visibility across organizational boundaries, modern perishable supply chains utilize **GS1 Electronic Product Code Information Services (EPCIS) 2.0** (JSON-LD and REST binding). EPCIS provides four standardized event types that capture the "what, where, when, and why" of age-structured inventory:

1. **ObjectEvent (Commissioning / Packaging):** Records when physical units are created and binds the lot number, packaging timestamp, and expiration date to the GS1 Digital Link URI.
2. **AggregationEvent (Palletizing):** Binds individual consumer trade items (GTINs) or cases into a Serial Shipping Container Code (SSCC-18) pallet identifier, establishing hierarchical lot inheritance.
3. **TransformationEvent (Repackaging / Processing):** Captures the conversion of raw perishable inputs into finished packaged goods, recalculating the resulting composite shelf life based on the earliest-expiring ingredient.
4. **TransactionEvent (Shipping / Receiving):** Records the transfer of custody between legal entities, linking EDI purchase orders to physical SSCC pallet scans.

Under EPCIS 2.0 CBV (Core Business Vocabulary), expiration attributes are standardized in JSON-LD payloads:

```json
{
  "@context": ["https://ref.gs1.org/standards/epcis/2.0.0/epcis-context.jsonld"],
  "type": "ObjectEvent",
  "action": "ADD",
  "bizStep": "commissioning",
  "disposition": "active",
  "epcList": ["urn:epc:id:sgtin:0012345.067890.LOT-B429A"],
  "eventTime": "2026-11-03T08:00:00Z",
  "eventTimeZoneOffset": "+00:00",
  "readPoint": {"id": "urn:epc:id:sgln:0012345.11111.0"},
  "cbv:itemExpirationDate": "2026-11-15",
  "cbv:shelfLifeExpirationDate": "2026-11-15T23:59:59Z",
  "cbv:bestBeforeDate": "2026-11-13"
}
```

When receiving docks scan the incoming SSCC barcode, their automated WMS queries the upstream supplier's EPCIS repository. The system retrieves `cbv:shelfLifeExpirationDate`, calculates the real-time Proportionate RSL, and issues an automated gate-hold before the forklift driver unloads the trailer.
---

## 9. Industry practice and case evidence

### 9.1 Walmart: Project Eden and Algorithmic Freshness Routing

Walmart operates one of the world's largest perishable distribution networks, moving billions of dollars of produce, meat, and dairy annually through its Perishable Distribution Centers (PDCs). In 2018, Walmart deployed **Eden**, an internal proprietary suite of digital applications designed to eliminate $\$2\text{ billion}$ in fresh food waste over five years.

- **Mechanics of Eden:** Rather than treating fresh produce as having a static fixed shelf life $m$, Eden combines digital USDA food inspection quality scores, temperature tracking sensor data from refrigerated trailers, and machine learning models of fungal and bacterial decay kinetics.
- **Dynamic RSL Re-Rating:** If a refrigerated container of strawberries experiences a temperature excursion from Oxnard, California to an Illinois PDC (e.g., maintaining $7^\circ\text{C}$ instead of $1.5^\circ\text{C}$ for 18 hours), Eden recalculates the remaining shelf life, downgrading it from 10 days to 5 days.
- **Velocity-Based Age Allocation:** Instead of rejecting the pallet at the PDC or shipping it to a distant store 300 miles away, Eden's routing engine reroutes the shipment to a high-velocity Supercenter 15 miles from the PDC where historical blueberry/strawberry daily turn rates guarantee consumption within 48 hours. This dynamic rerouting prevented an estimated $\$86\text{ million}$ in spoilage in its first year of pilot operation.

### 9.2 Tesco: Automated Dynamic Markdowns and "Reduced to Clear"

Tesco, the UK's largest food retailer, pioneered algorithmic shelf-life discounting across its 2,800 convenience and supermarket locations to manage the evening outdating cliff.

- **The Problem:** Store associates previously walked aisles at fixed intervals (e.g., 2:00 PM and 7:00 PM) with handheld barcode guns, applying subjective percentage discounts (20% off, then 50% off) using physical yellow stickers. Associates frequently marked down products either too aggressively (destroying gross margin) or too late (resulting in product reaching 11:59 PM unsold, triggering mandatory landfill destruction under UK "Use by" law).
- **Algorithmic Markdown Execution:** Tesco implemented automated dynamic pricing integrated with POS sales velocity:
  - If a chilled ready meal has 1 day remaining and local store inventory is 12 units against an expected evening sales velocity of 4 units, the algorithm prints a yellow sticker with an optimized $35\%$ discount calculated to lift demand elasticity just enough to clear all 12 units by 8:00 PM.
  - If inventory is 3 units against expected sales of 5 units, the algorithm suppresses the discount entirely, allowing the remaining units to clear at full gross margin.

### 9.3 CPG Supplier vs. Mass Retailer MRSL Chargeback Disputes

In 2021–2023, extreme driver shortages and refrigerated chassis congestion at major U.S. intermodal rail ramps created systemic RSL contractual crises between dairy/deli processors (e.g., Chobani, Kraft Heinz, Danone) and national retail chains.

- **The Incident:** Cross-country refrigerated railcar dwell times increased from an average of 36 hours to over 96 hours. Packaged hummus and artisan sour cream with a 45-day total shelf life arrived at retailer central DCs with 31 days remaining ($68.9\%$ RSL).
- **The Contractual Penalty:** The retailer contract stipulated a mandatory $\text{MRSL} \ge 75\%$ (33.75 days). Retailer automated WMS systems issued systematic gate rejections on hundreds of refrigerated truckloads. 
- **Financial Fallout:** Because returning chilled dairy to origin plants is economically non-viable (freight cost exceeding product value), suppliers were forced to sell rejected stock to local livestock feed conversion plants or deep-discount secondary wholesalers at 10 to 15 cents on the dollar, while absorbing retailer non-compliance penalties averaging 3% of gross invoice value. This crisis forced CPG manufacturers to renegotiate MRSL tiers, introducing sliding-scale price deductions (e.g., 2% invoice discount for deliveries between 70% and 75% RSL) in lieu of outright gate refusals.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign | Root Physical / Behavioral Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **The Stagnation Trap** | Cohort 1 stock dwells on shelf despite brisk overall category sales volume. | Consumers cherry-pick fresh stock from the rear; front stock remains unpurchased until expiration. | 100% outdating of aging cohorts; severe retail shrink. | Front-face rotation SOPs, dynamic yellow-sticker markdowns, gravity-fed roller shelving. |
| 2 | **Phantom FEFO in WMS** | ERP records clean FEFO consumption, but physical warehouse floor is cluttered with expired lots. | System assumes FEFO issuing, but forklift drivers pull nearest physical pallet from accessible floor locations. | Latent obsolescence; sudden massive inventory write-offs during physical cycle counts. | Barcode-directed pick enforcement (system refuses pick confirmation without matching lot scan). |
| 3 | **DC Age Inversion** | Newer production lots are dispatched to stores while older lots remain in the central warehouse. | Inbound cross-dock shipments bypass racking, while older pallets sit trapped in deep-lane reserve storage. | Store receiving docks receive older stock later; increased store-level outdating. | Deep-lane drive-in rack purging rules; WMS strict lot rotation rules for pick faces. |
| 4 | **Supplier Date Batching** | Pallet arrives containing mixed cases bearing three different expiration dates. | Manufacturer sweeps production staging floor to fulfill an emergency replenishment order. | Receiving dock scans the freshest case; older cases expire unmonitored in the system. | ASN EDI 856 mandatory single-lot-per-pallet rule; receiving dock multi-scan audits. |
| 5 | **RSL Gate Cascades** | Upstream transportation delay of 2 days causes a 50-truckload rejection at the retail DC. | Rigid contractual step-function MRSL thresholds with zero operational tolerance or sliding scale. | Massive bullwhip disruption; emergency production runs; stranded inventory write-down. | Sliding-scale contractual tolerance clauses; automated pre-dispatch RSL routing. |
| 6 | **Markdown Cannibalization** | Full-price sales collapse following the application of afternoon yellow-sticker discounts. | Discounted near-expiry units cannibalize shoppers who were fully willing to buy fresh stock at full price. | Margin erosion without incremental category unit lift. | Time-gated markdowns (e.g., apply only after 8:00 PM); separate clearance display baskets. |
| 7 | **Stockout of Freshness** | High stockout rate reported by POS despite inventory on hand exceeding safety stock targets. | Customers reject visible front-rail stock because remaining shelf life is below consumer subjective threshold. | "Phantom inventory" suppresses replenishment orders while customers walk out without purchasing. | Point-of-sale inventory zeroing rules based on shelf-life expiration algorithms. |
| 8 | **Disconnected EDI ASN Dates** | Expiration date on physical carton barcode does not match the `DTM*036` segment in EDI 856. | Packaging line operators changed master date stamp without synchronizing plant MES and EDI gateway. | Automated receipt failure; trailer detention while quality control performs manual verification. | Automated barcode-to-EDI validation scanning at plant pallet wrapper exit. |
| 9 | **Blind Cold Storage Rotation** | 3PL warehouse charges monthly storage fees while pallets exceed retailer MRSL thresholds. | 3PL fee structure incentivizes volume storage rather than inventory velocity; no visibility into client RSL rules. | Stock becomes un-shippable to primary retail channels while sitting in contracted 3PL care. | Automated daily WMS reporting of "Effective Usable Life to Primary Channel" with liquidation alerts. |
| 10 | **Replenishment Oversizing** | EOQ-based purchasing rules order 30 days of supply for a product with 21 days of shelf life. | Purchasing agents optimize purchase price volume tiers without constraining lot size by shelf life $m$. | Guaranteed structural outdating of trailing order cohorts. | Shelf-life-constrained order sizing: $Q_{\text{max}} = \sum_{t=1}^{m - L - \text{MRSL}} \hat{D}_t$. |
| 11 | **The Temperature-Excursion RSL Cliff** | Pallet temperature logger beeps during receiving dock download. | Refrigeration breakdown during multi-day rail transit accelerates biochemical decay kinetics. | Product reaches dock with nominally valid date code but unmerchantable microbial spoilage. | Automated IoT logger-to-WMS integration calculating Arrhenius-adjusted effective RSL. |
| 12 | **Reverse-Logistics Outdating Cascade** | Rejected trailers dwell at carrier terminal awaiting customer service disposition. | Bureaucratic delays in resolving EDI 861 gate rejection disputes consume remaining shelf life. | Product that was initially salvageable at 65% RSL dwells until 0% RSL; 100% total write-off. | Pre-authorized secondary liquidation contracts triggering instant automated diversion. |
| 13 | **Virtual Batch Splitting Errors** | Warehouse splits a physical pallet into multiple order picks with mismatched expiration dates. | WMS lot allocation logic assigns partial cases across multiple pick faces without barcode re-labeling. | Downstream store receives cases with unrecorded older dates; in-store rotation fails. | Mandatory serialized GS1-128 container re-tagging upon pallet de-consolidation. |

---

## 11. Assumptions and limits

| # | Assumption | Why It Is Needed | What Breaks It in Reality | How to Test / Validate |
|---|---|---|---|---|
| 1 | **Deterministic Shelf Life ($m$)** | Simplifies state transition to fixed discrete steps $\Delta t = 1$. | Microbial decay kinetics follow the Arrhenius equation; temperature abuse accelerates aging exponentially. | Continuous cold-chain IoT temperature loggers; Mean Kinetic Temperature (MKT) modeling. |
| 2 | **Homogeneous Consumer Preference** | Allows modeling customer choice via a static aggregate compliance factor $\alpha$. | Price-sensitive shoppers actively hunt for markdowns, while convenience shoppers grab the front item. | Scanner panel data analysis; basket tracking of discounted vs. full-price perishable items. |
| 3 | **Uncorrelated Period Demand** | Ensures clean Nahmias transition equations without multi-period demand covariance terms. | Food demand exhibits strong day-of-week seasonality (e.g., Friday/Saturday peaks) and weather spikes. | Autoregressive time series modeling; seasonal decomposition of POS transaction logs. |
| 4 | **Zero In-Transit Degradation Beyond Time** | Assumes cold chain maintains product in pristine stasis until final expiration date. | Chilling compressor failures, door openings at cross-docks, and uninsulated tarmac delays. | Pallet-level pulp temperature probe testing at receiving docks. |
| 5 | **Linear Disposal Cost Structure** | Treats waste cost as a constant marginal penalty $\theta = c + c_d - s$. | Landfill surcharges jump discontinuously when municipal bio-waste limits are exceeded; bulk composting tiers. | Monthly solid waste utility bill audits; contracted hauler rate schedule reviews. |
| 6 | **Instantaneous Operational Replenishment** | Avoids complex multi-period lead-time convolutions in the state transition equations. | Fresh food manufacturing schedules require 2- to 5-day production changeover and cooling windows. | Historical supplier lead-time variance analysis across refrigerated LTL lanes. |
| 7 | **Independent SKU Dynamics** | Analyzes age cohorts for a single SKU in isolation. | Stocking out of a 1-gallon milk SKU causes immediate substitution demand into half-gallon containers. | Cross-price elasticity matrix estimation and assortment substitution models. |
| 8 | **Static Contractual MRSL Thresholds** | Assumes retailers apply invariant percentage rules across all seasons and macroeconomic environments. | Retailers relax MRSL rules during severe national shortages to maintain shelf presence, then tighten them during gluts. | Historical receiving dock rejection logs cross-referenced with national supply availability indices. |
| 9 | **Fixed Review Period Duration ($\Delta t$)** | Establishes uniform time steps for cohort advancement equations. | Store receiving schedules fluctuate due to carrier delays, holiday shifts, and dock curfews. | Discrete-event simulation calibrated to continuous variable inter-arrival times. |
| 10 | **Zero Pallet Structural Crushing** | Assumes physical packaging integrity is maintained throughout the storage horizon. | Bottom layer cartons on 60-inch high pallets buckle under prolonged chilling humidity. | Crushed packaging exposes food to microbial contamination, causing premature spoilage. | Corrugated box compression strength testing (ECT) under 90% relative humidity. |

---

## 12. Exercises

### Exercise 1: Nahmias State Vector Transition under FEFO (Recall and Calculation)
A specialty cheese distribution center operates with a review period of $\Delta t = 7$ days. A premium soft cheese has a maximum usable shelf life of $m = 3$ periods (21 days). At the start of Week 1, the inventory state vector is:
$$\mathbf{x}_1 = \begin{bmatrix} x_{1,1} \\ x_{2,1} \end{bmatrix} = \begin{bmatrix} 40 \\ 90 \end{bmatrix} \text{ cases}$$
An order $Q_1 = 50$ cases of fresh cheese arrives at the start of Week 1 (entering cohort 2).
Realized customer demand in Week 1 is $D_1 = 65$ cases.
1. Calculate the number of cases consumed from Cohort 1 and Cohort 2 under strict FEFO ($\alpha = 1.0$).
2. Compute the outdating volume $W_1$ at the end of Week 1.
3. Determine the starting inventory state vector $\mathbf{x}_2$ for Week 2.
- *Deliverable:* Step-by-step arithmetic, final state vector $\mathbf{x}_2$, and outdating volume.
- *Acceptance Criteria:* Exact numeric match; correct application of Nahmias boundary formulas. Time: 15 minutes.

### Exercise 2: Cherry-Picking Depletion Penalty (Application and Analysis)
Using the initial inventory and demand from Exercise 1 ($\mathbf{x}_1 = [40, 90]^T$, $Q_1 = 50$, $D_1 = 65$):
1. Assume retail consumers exhibit cherry-picking behavior with compliance $\alpha = 0.40$ (meaning $40\%$ of demand pulls FEFO from the oldest stock, and $60\%$ pulls LIFO from the freshest stock).
2. Calculate the exact depletion from each cohort and the outdating volume $W_1(\alpha=0.40)$.
3. If unit cost $c = \$50/\text{case}$, retail price $p = \$95/\text{case}$, and disposal fee $c_d = \$5/\text{case}$, calculate the net profit difference between strict FEFO and this cherry-picking scenario.
- *Deliverable:* Depletion table, waste calculation, and financial variance statement.
- *Acceptance Criteria:* Arithmetic accuracy; explicit accounting for waste penalty $(c + c_d)$. Time: 20 minutes.

### Exercise 3: Outdating Probability under Continuous Stochastic Demand (Calculation)
Fresh pasteurized cream has a shelf life of $m = 2$ days. Orders placed at the start of Day 1 arrive immediately with full 2-day life. 
Demand on Day 1 ($D_1$) and Day 2 ($D_2$) are independent, identically distributed continuous uniform random variables: $D_t \sim U(20, 80)$ gallons.
An order of $Q = 90$ gallons is placed and received on Day 1 into an initially empty facility ($\mathbf{x}_1 = [0]$).
1. Formulate the mathematical expression for the outdating volume $W_2$ at the end of Day 2 as a function of $Q, D_1,$ and $D_2$ under FEFO.
2. Calculate the exact probability of outdating: $P(W_2 > 0)$.
3. Compute the expected outdating volume $\mathbb{E}[W_2]$.
- *Deliverable:* Analytical integral setup, closed-form probability, and numerical expectation.
- *Acceptance Criteria:* Correct convolution of uniform distributions; correct evaluation of double integrals. Time: 25 minutes.

### Exercise 4: Contractual RSL Gate Compliance Audit (Application)
A meat processing plant packages fresh marinated poultry breasts on Monday, October 12 at 06:00. The product has a total validated shelf life of 18 days.
The supplier holds the product in factory cold storage until dispatching an LTL refrigerated trailer on Wednesday, October 14 at 18:00.
The trailer reaches the retailer distribution center on Saturday, October 17 at 12:00.
The retailer's master vendor agreement specifies a Minimum Remaining Shelf Life (MRSL) covenant of $\ge 75.0\%$ upon receipt.
1. Calculate the elapsed shelf life in days and hours upon arrival at the retailer DC gate.
2. Calculate the exact Proportionate RSL ($\text{RSL}\%$) at dock check-in.
3. Will the retailer WMS accept or issue an automated gate rejection?
4. What is the latest date and time this shipment could have arrived at the DC dock to pass the $75.0\%$ gate?
- *Deliverable:* Elapsed time calculation, percentage RSL to one decimal place, compliance determination, and deadline timestamp.
- *Acceptance Criteria:* Accurate timestamp subtraction; correct application of contractual inequality. Time: 15 minutes.

### Exercise 5: Multi-Echelon Age Allocation Strategy (System Design)
A central dairy distribution center holds 1,000 cases of specialty butter with 14 days of remaining shelf life (Total original life = 45 days). The DC supplies two retail store clusters:
- **Cluster Urban:** 10 compact convenience stores. Combined daily sales velocity: 120 cases/day. Replenishment transit time: 1 day. In-store store-dock MRSL requirement: $\ge 20\%$ (9 days).
- **Cluster Rural:** 8 large suburban hypermarkets. Combined daily sales velocity: 25 cases/day. Replenishment transit time: 3 days. In-store store-dock MRSL requirement: $\ge 25\%$ (11.25 days).
A new production run of 2,000 cases with 42 days of remaining shelf life has just arrived at the DC.
Design an allocation policy governing which butter lots (the 14-day lot vs. the 42-day lot) should be dispatched to Cluster Urban and Cluster Rural. Calculate the outdating risk under your proposed policy versus an indiscriminate FIFO allocation.
- *Deliverable:* Formal allocation rule, velocity-to-shelf-life ratio analysis, and risk comparison table.
- *Acceptance Criteria:* Deep justification based on inventory velocity and transit dwell; clear demonstration of waste avoidance. Time: 30 minutes.

### Exercise 6: Dynamic Markdown Threshold Formulation (Critique and Design)
Critique the standard practice of applying a flat 50% discount to all perishable goods on their final day of shelf life. Formulate an optimization model that determines the optimal markdown timing $t^*$ and discount depth $\delta^*$ as a function of remaining stock $x_{1,t}$, price elasticity of demand $\epsilon$, full price $p$, salvage value $s$, and cannibalization coefficient $\beta \in [0, 1]$ (representing the fraction of discount buyers who would have otherwise purchased fresh stock at full price).
- *Deliverable:* Written critique (300 words), mathematical model formulation with objective function and constraints, and an operational policy recommendation.
- *Acceptance Criteria:* Clear identification of economic trade-offs (margin recovery vs. full-price cannibalization vs. disposal cost); rigorous mathematical notation. Time: 30 minutes.

### Exercise 7: Arrhenius Temperature Excursion and Effective RSL Calculation (Calculation)
A shipment of fresh organic salmon fillets with a validated baseline shelf life of $m = 10.0$ days at standard storage temperature $T_{\text{standard}} = 274.15\text{ K}$ ($1.0^\circ\text{C}$) departs an aquaculture processing plant in Chile bound for a Miami distribution center. The primary spoilage mechanism is pseudomonad microbial proliferation with activation energy $E_a = 75,000\text{ J/mol}$.
During airfreight tarmac staging, the reefer container experiences a failure, holding the product at $T_{\text{excurs}} = 285.15\text{ K}$ ($12.0^\circ\text{C}$) for $\Delta t_{\text{excurs}} = 18.0\text{ hours}$ ($0.75\text{ days}$). Standard transit takes 2.25 days at $1.0^\circ\text{C}$.
1. Compute the Arrhenius acceleration factor $A_T$ using gas constant $R = 8.314\text{ J/(mol}\cdot\text{K)}$.
2. Calculate the equivalent standard days of shelf life consumed during the 18-hour excursion.
3. Compute the effective remaining shelf life $R_{\text{arr}}^{\text{adj}}$ upon arrival at the Miami DC.
4. If the retailer enforces an MRSL covenant of $\ge 70.0\%$, does the shipment pass or fail the receipt gate?
- *Deliverable:* Calculation of $A_T$, effective elapsed days, adjusted remaining shelf life, and gate decision.
- *Acceptance Criteria:* Accurate Kelvin conversions; correct Arrhenius exponent evaluation; sound RSL percentage comparison. Time: 20 minutes.

### Exercise 8: GS1 EPCIS 2.0 Critical Tracking Event Architecture (Design)
Design an end-to-end GS1 EPCIS 2.0 event architecture for a fresh packaged cut-salad processor subject to FDA FSMA 204. The process encompasses: (1) Harvesting and washing raw baby spinach; (2) Packaging into 1-lb clamshells with lot creation and expiration stamping; (3) Palletizing 120 clamshells onto an SSCC pallet; and (4) Shipping to a grocery retailer DC under an EDI 856 ASN.
- *Deliverable:* A structured JSON-LD event schema representing the `ObjectEvent` commissioning and `AggregationEvent` palletizing steps, specifying all mandatory Key Data Elements (KDEs): GTIN, lot code, expiration date, SSCC, BizStep, Disposition, and ReadPoint.
- *Acceptance Criteria:* Full compliance with GS1 EPCIS 2.0 JSON-LD syntax; explicit inclusion of `cbv:shelfLifeExpirationDate` and `cbv:itemExpirationDate`. Time: 25 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Nahmias State Vector Transition under FEFO

1. **Cohort Depletion in Week 1:**
   - Total stock available: $x_{1,1} = 40$ (Cohort 1, 7 days remaining); $x_{2,1} = 90 + 50 = 140$ (Cohort 2, 14 days remaining). Total $I_1 = 180$ cases.
   - Demand $D_1 = 65$ cases.
   - Under strict FEFO ($\alpha = 1.0$), demand consumes oldest stock first:
     - Depletion from Cohort 1: $\min(40, 65) = \mathbf{40\ cases}$.
     - Cohort 1 remaining balance: $40 - 40 = 0$ cases.
     - Remaining unmet demand: $65 - 40 = 25$ cases.
     - Depletion from Cohort 2: $\min(140, 25) = \mathbf{25\ cases}$.
     - Cohort 2 remaining balance: $140 - 25 = \mathbf{115\ cases}$.
2. **Outdating Volume at End of Week 1 ($W_1$):**
   - Outdates originate solely from unconsumed stock in Cohort 1:
     $$W_1 = \max(0, x_{1,1} - D_1) = \max(0, 40 - 65) = \mathbf{0\ cases}$$
3. **Starting State Vector for Week 2 ($\mathbf{x}_2$):**
   - Cohort 2 surviving stock advances to Cohort 1: $x_{1,2} = 115$ cases.
   - Cohort 2 starts at 0 (prior to Week 2 replenishment arrival): $x_{2,2} = 0$ cases.
   $$\mathbf{x}_2 = \begin{bmatrix} 115 \\ 0 \end{bmatrix}$$

---

### Answer to Exercise 2: Cherry-Picking Depletion Penalty

1. **Demand Partitioning under $\alpha = 0.40$:**
   - Total Demand $D_1 = 65$ cases.
   - FEFO Stream: $D_1^{\text{FEFO}} = 0.40 \times 65 = \mathbf{26\ cases}$.
   - LIFO Stream: $D_1^{\text{LIFO}} = 0.60 \times 65 = \mathbf{39\ cases}$.
2. **Depletion by Cohort:**
   - **FEFO Stream (26 cases)** consumes from Cohort 1 ($x_{1,1} = 40$):
     - Cohort 1 consumed: $26$ cases.
     - Cohort 1 surviving: $40 - 26 = \mathbf{14\ cases}$.
   - **LIFO Stream (39 cases)** consumes from Cohort 2 ($x_{2,1} = 140$):
     - Cohort 2 consumed: $39$ cases.
     - Cohort 2 surviving: $140 - 39 = \mathbf{101\ cases}$.
   - Total demand fulfilled = $26 + 39 = 65$ cases. Unmet demand = $0$.
3. **Outdating Calculation ($W_1$):**
   - The 14 surviving cases in Cohort 1 expire at the end of Week 1:
     $$W_1(\alpha=0.40) = \mathbf{14\ cases\ outdated}$$
4. **Financial Variance Calculation:**
   - Revenue in both cases: $65 \times \$95 = \$6,175.00$.
   - Cost of Goods Sold in both cases: $65 \times \$50 = \$3,250.00$.
   - Under FEFO: Outdates = $0 \implies \text{Waste Cost} = \$0.00$.
   - Under Cherry-Picking:
     $$\text{Waste Penalty} = W_1 \times (c + c_d) = 14 \times (\$50 + \$5) = 14 \times \$55 = \mathbf{\$770.00}$$
   - Holding cost difference: Under FEFO, ending inventory is 115 cases. Under cherry-picking, ending inventory is 101 cases (since 14 were thrown away). Holding savings = $14 \times h$. Assuming $h = \$1.00/\text{case}$, holding savings is $\$14.00$.
   - **Net Profit Impact:** Cherry-picking reduces net profit by $\$770.00 - \$14.00 = \mathbf{\$756.00}$ (an immediate $26.2\%$ hit to gross operating margin).

---

### Answer to Exercise 3: Outdating Probability under Continuous Stochastic Demand

1. **Mathematical Expression for Outdating:**
   - Order $Q = 90$ arrives on Day 1 into cohort with 2 days life.
   - Day 1 demand $D_1$ consumes from $Q$. Stock surviving to Day 2 is $I_2 = \max(0, Q - D_1) = [90 - D_1]^+$.
   - This surviving stock now has 1 day of life remaining (Cohort 1 on Day 2).
   - Under FEFO, Day 2 demand $D_2$ consumes from this surviving stock.
   - Any stock remaining at the end of Day 2 outdates:
     $$W_2 = \max(0, I_2 - D_2) = \max(0, 90 - D_1 - D_2) = [90 - (D_1 + D_2)]^+$$
2. **Probability of Outdating $P(W_2 > 0)$:**
   - Outdating occurs if and only if $D_1 + D_2 < 90$.
   - Let $S = D_1 + D_2$. Both $D_1, D_2 \sim U(20, 80)$.
   - Minimum possible sum: $20 + 20 = 40$. Maximum possible sum: $80 + 80 = 160$.
   - The joint distribution of $(D_1, D_2)$ is uniform over the square $[20, 80] \times [20, 80]$ with area $(80 - 20)^2 = 60^2 = 3600$.
   - We find the area of the region where $D_1 + D_2 < 90$ and $20 \le D_1 \le 80$, $20 \le D_2 \le 80$.
   - Boundary line: $D_2 = 90 - D_1$.
     - When $D_1 = 20$, $D_2 = 70 \le 80$.
     - When $D_2 = 20$, $D_1 = 70 \le 80$.
   - This forms a right isosceles triangle bounded by $D_1 \ge 20$, $D_2 \ge 20$, and $D_1 + D_2 \le 90$.
   - The leg lengths of this triangle are:
     $$\text{Leg} = 70 - 20 = 50$$
   - Area of the outdating triangle:
     $$\text{Area} = \frac{1}{2} \times \text{base} \times \text{height} = \frac{1}{2} \times 50 \times 50 = 1250$$
   - Probability of outdating:
     $$P(W_2 > 0) = \frac{\text{Area of Triangle}}{\text{Total Sample Space Area}} = \frac{1250}{3600} = \frac{25}{72} \approx \mathbf{0.3472\ (34.72\%)}$$
3. **Expected Outdating Volume $\mathbb{E}[W_2]$:**
   $$\mathbb{E}[W_2] = \iint_{D_1 + D_2 < 90} (90 - u - v) f(u, v) \, du \, dv = \frac{1}{3600} \int_{20}^{70} \left[ \int_{20}^{90 - u} (90 - u - v) \, dv \right] du$$
   - Inner integral: Let $C = 90 - u$.
     $$\int_{20}^{C} (C - v) \, dv = \left[ C v - \frac{v^2}{2} \right]_{20}^{C} = \left( C^2 - \frac{C^2}{2} \right) - \left( 20 C - \frac{400}{2} \right) = \frac{C^2}{2} - 20 C + 200 = \frac{(C - 20)^2}{2}$$
   - Since $C - 20 = 90 - u - 20 = 70 - u$:
     $$\text{Inner Integral} = \frac{(70 - u)^2}{2}$$
   - Outer integral:
     $$\mathbb{E}[W_2] = \frac{1}{3600} \int_{20}^{70} \frac{(70 - u)^2}{2} \, du = \frac{1}{7200} \left[ -\frac{(70 - u)^3}{3} \right]_{20}^{70} = \frac{1}{7200} \left( 0 - \left( -\frac{50^3}{3} \right) \right) = \frac{125,000}{21,600} \approx \mathbf{5.79\ gallons}$$

---

### Answer to Exercise 4: Contractual RSL Gate Compliance Audit

1. **Elapsed Time Calculation:**
   - Packaging: Monday 06:00.
   - Arrival: Saturday 12:00.
   - Elapsed time:
     - Monday 06:00 to Friday 06:00 = 4 full days (96 hours).
     - Friday 06:00 to Saturday 06:00 = 1 full day (24 hours).
     - Saturday 06:00 to Saturday 12:00 = 6 hours (0.25 days).
     - Total elapsed time = $5\text{ days} + 6\text{ hours} = \mathbf{5.25\ days}$ (126 hours).
2. **Remaining Shelf Life and Proportionate RSL:**
   - Total shelf life $m_{\text{orig}} = 18.0$ days (432 hours).
   - Remaining shelf life: $R_{\text{arr}} = 18.0 - 5.25 = \mathbf{12.75\ days}$ (306 hours).
   - Proportionate RSL:
     $$\text{RSL}\% = \frac{12.75}{18.0} \times 100\% = \mathbf{70.83\%}$$
3. **Acceptance Decision:**
   - Contractual requirement: $\text{MRSL} \ge 75.0\%$.
   - Since $70.83\% < 75.00\%$, the WMS will issue an **AUTOMATED GATE REJECTION**.
4. **Latest Acceptable Arrival Deadline:**
   - Minimum required remaining days: $18.0 \times 0.75 = 13.5$ days.
   - Maximum allowable elapsed days: $18.0 - 13.5 = 4.5$ days (108 hours).
   - Packaging timestamp: Monday Oct 12 at 06:00.
   - Adding 4.5 days (4 days and 12 hours):
     $$\text{Deadline} = \mathbf{Friday,\ October\ 16\ at\ 18:00}$$
   - The truck arrived 18 hours too late to clear the contractual gate.

---

### Answer to Exercise 5: Multi-Echelon Age Allocation Strategy

- **Velocity-to-Shelf-Life Alignment:**
  - Cluster Urban sells 120 cases/day. It will consume the 1,000 cases of 14-day stock in:
    $$\text{Dwell Time} = \frac{1,000}{120} \approx 8.33\text{ days}$$
    Transit takes 1 day; total time to clear is $1 + 8.33 = 9.33$ days. Since stock has 14 days remaining, **zero outdates will occur** ($14 > 9.33$). Furthermore, upon arrival on Day 1, remaining life is $14 - 1 = 13$ days ($28.9\%$ of 45 days), exceeding the store dock covenant of $20\%$ (9 days).
  - Cluster Rural sells only 25 cases/day. If dispatched the 14-day stock:
    - Transit takes 3 days; remaining life at store dock is $14 - 3 = 11$ days ($24.4\%$).
    - This fails the Rural store dock covenant of $25\%$ ($11.25$ days), triggering store gate rejections!
    - Even if accepted, consuming 1,000 cases would take $1,000 / 25 = 40$ days, resulting in massive in-store spoilage.
- **Optimal Policy:** Route 100% of the 14-day butter (1,000 cases) to Cluster Urban. Route the fresh 42-day butter to Cluster Rural (and the remainder to Urban once the 14-day stock is exhausted).
- **Outcome:** Indiscriminate FIFO dispatching causes an estimated 450–600 cases of spoilage and store gate rejections in rural hypermarkets. Dynamic velocity-based allocation yields zero outdating.

---

### Answer to Exercise 6: Dynamic Markdown Threshold Formulation

- **Critique of Flat 50% Rule:** A flat 50% end-of-day markdown is a blunt heuristic that creates two catastrophic economic failures: (1) **Deep Margin Cannibalization:** For high-demand staples with inelastic late-afternoon demand, marking down by 50% gives away margin to consumers who were prepared to pay full price ($\beta \approx 1.0$). (2) **Under-Clearing Bulk Gluts:** When an unexpected order cancellation leaves 50 units on hand with an evening capacity of 10, a 50% discount may fail to stimulate sufficient elasticity, resulting in 30 units expiring anyway and incurring full disposal fees.
- **Optimization Model Structure:**
  $$\max_{\delta_t \in [0, 1]} \Pi = \sum_{t=1}^{T} \left[ (p(1 - \delta_t) - c) \cdot S_t(\delta_t) - \beta \cdot p \delta_t \cdot S_t(\delta_t) \right] - \theta \cdot W_T$$
  subject to:
  $$S_t(\delta_t) = \min\left( x_{1,t}, D_t \cdot (1 + \epsilon \delta_t) \right)$$
  $$W_T = \left[ x_{1,1} - \sum_{t=1}^{T} S_t(\delta_t) \right]^+$$
  where $\epsilon$ is price elasticity of demand and $\beta$ captures cannibalization intensity. The optimal discount $\delta_t^*$ equates the marginal revenue from stimulated clearance volume with the marginal cannibalization of full-price margin plus the avoided disposal fee $\theta$.

---

### Answer to Exercise 7: Arrhenius Temperature Excursion and Effective RSL Calculation

1. **Arrhenius Thermal Acceleration Factor ($A_T$):**
   - Temperatures in Kelvin: $T_{\text{standard}} = 1.0 + 273.15 = 274.15\text{ K}$; $T_{\text{excurs}} = 12.0 + 273.15 = 285.15\text{ K}$.
   - Inverse temperature difference:
     $$\Delta \left(\frac{1}{T}\right) = \frac{1}{274.15} - \frac{1}{285.15} \approx 0.00364764 - 0.00350693 = 0.00014071\text{ K}^{-1}$$
   - Exponent term:
     $$\frac{E_a}{R} \cdot \Delta \left(\frac{1}{T}\right) = \frac{75,000}{8.314} \times 0.00014071 = 9020.93 \times 0.00014071 \approx 1.26935$$
   - Acceleration factor:
     $$A_T = \exp(1.26935) \approx \mathbf{3.5585}$$
   - The microbial reaction proceeds $3.56\times$ faster during the excursion.
2. **Equivalent Shelf Life Consumed During Excursion:**
   $$\Delta t_{\text{equiv}} = A_T \times \Delta t_{\text{excurs}} = 3.5585 \times 0.75\text{ days} = \mathbf{2.6689\ standard\ days}$$
3. **Effective Remaining Shelf Life ($R_{\text{arr}}^{\text{adj}}$):**
   - Standard transit consumed: $2.25$ days.
   - Total effective transit consumption: $2.25 + 2.6689 = \mathbf{4.9189\ days}$.
   - Effective remaining shelf life:
     $$R_{\text{arr}}^{\text{adj}} = 10.0 - 4.9189 = \mathbf{5.0811\ days}$$
   - Note: Calendar elapsed time was only $2.25 + 0.75 = 3.0$ days, which would have implied $7.0$ calendar days remaining. The thermal excursion burned an extra $1.92$ days of biochemical life!
4. **Gate Compliance Audit:**
   $$\text{Effective RSL}\% = \frac{5.0811}{10.0} \times 100\% = \mathbf{50.81\%}$$
   - Contractual requirement: $\text{MRSL} \ge 70.0\%$.
   - Since $50.81\% < 70.0\%$, the WMS issues an **AUTOMATED GATE REJECTION**.

---

### Answer to Exercise 8: GS1 EPCIS 2.0 Critical Tracking Event Architecture

- **Traceability System Architecture:**
  The solution requires two linked EPCIS events sharing a common GS1 Digital Link lot-level SGTIN:

```json
{
  "@context": ["https://ref.gs1.org/standards/epcis/2.0.0/epcis-context.jsonld"],
  "isA": "EPCISDocument",
  "schemaVersion": "2.0",
  "creationDate": "2026-11-03T10:00:00Z",
  "epcisBody": {
    "eventList": [
      {
        "type": "ObjectEvent",
        "action": "ADD",
        "bizStep": "commissioning",
        "disposition": "active",
        "epcList": ["urn:epc:id:sgtin:0012345.099881.LOT-SPIN-042"],
        "eventTime": "2026-11-03T06:30:00Z",
        "eventTimeZoneOffset": "-05:00",
        "readPoint": {"id": "urn:epc:id:sgln:0012345.PACK01.0"},
        "bizLocation": {"id": "urn:epc:id:sgln:0012345.SALINAS_PLANT.0"},
        "cbv:itemExpirationDate": "2026-11-17",
        "cbv:shelfLifeExpirationDate": "2026-11-17T23:59:59Z"
      },
      {
        "type": "AggregationEvent",
        "action": "ADD",
        "bizStep": "packing",
        "disposition": "in_progress",
        "parentID": "urn:epc:id:sscc:0012345.1000000042",
        "childEPCs": ["urn:epc:id:sgtin:0012345.099881.LOT-SPIN-042"],
        "eventTime": "2026-11-03T07:15:00Z",
        "eventTimeZoneOffset": "-05:00",
        "readPoint": {"id": "urn:epc:id:sgln:0012345.PALLETIZER2.0"}
      }
    ]
  }
}
```

- **Compliance Verification:** The schema binds the SSCC-18 pallet identifier to the child SGTIN carrying `cbv:shelfLifeExpirationDate`. When scanned at the retail DC receiving dock, the WMS queries this EPCIS repository via REST API, automatically auditing FSMA 204 KDEs without manual data entry.

---

## 14. Interview questions

### Question 1: What is the Nahmias state vector, and why does scalar inventory tracking fail in fresh food?
**Strong Answer:** In perishable inventory, inventory cannot be treated as a scalar number because the probability of an item fulfilling demand versus incurring disposal costs depends strictly on its age. The Nahmias state vector $\mathbf{x}_t = [x_{1,t}, x_{2,t}, \dots, x_{m-1,t}]$ decomposes stock into discrete cohorts by remaining shelf life. Scalar tracking aggregates these cohorts, hiding the age distribution. Two distribution centers with 10,000 units each have identical scalar inventory positions, but if DC A holds 10,000 units with 20 days remaining while DC B holds 10,000 units with 2 days remaining, DC B faces imminent 100% write-offs and retail gate rejections. Base-stock policies based on scalar inventory over-order because they fail to discount stock that will outdate before demand arrives.

**Weak Answer:** The Nahmias vector is a matrix used in advanced math to track inventory. Scalar tracking fails because it doesn't tell you when things expire, so food goes bad in the warehouse.

---

### Question 2: Explain the economic and operational conflict between operational FEFO and consumer LIFO.
**Strong Answer:** Operational FEFO is the policy where the warehouse or store issues the oldest available stock first to minimize expiration waste. However, in retail environments with open display cases, consumers practice LIFO (cherry-picking): they inspect date stamps and take items from the back of the shelf with the longest remaining life. This creates a "stagnation trap": front-facing stock with short remaining life is repeatedly bypassed. Because the oldest cohort is exposed to only a fraction $\alpha$ of total demand, its dwell time lengthens, guaranteeing that it crosses the expiration threshold unsold. The supply chain incurs double economic damage: it pays disposal costs on the expired front stock while prematurely depleting fresh stock that could have supported full-price sales later in the week.

**Weak Answer:** FEFO means first expired, first out, and LIFO means last in, first out. The conflict is that the store wants to sell old food first, but customers like buying fresh food, which makes the store lose money.

---

### Question 3: What is an MRSL covenant, and what happens operationally when a shipment arrives with 74% RSL against a 75% covenant?
**Strong Answer:** Minimum Remaining Shelf Life (MRSL) is a legally binding contract term imposed by retailers requiring suppliers to deliver goods with a minimum percentage of total shelf life remaining upon receipt at the DC dock (typically 75% or 80%). If a 30-day product arrives with 22 days remaining (73.3% RSL), it violates the 75% threshold (22.5 days). In an automated WMS, this triggers a hard gate rejection: the shipment cannot be receipted or unloaded. The supplier must either pay return freight to haul the load back, redirect the trailer to an emergency secondary liquidator (e.g., Grocery Outlet) at an 80% price markdown, or pay the retailer a disposal/tipping fee. Additionally, the retailer's ERP automatically issues an OTIF (On-Time In-Full) non-compliance vendor chargeback, typically 3% of the purchase order value.

**Weak Answer:** MRSL is a rule about how much shelf life a product has left. If it arrives at 74% instead of 75%, the warehouse manager might reject the truck or ask for a discount, and the company has to throw the food away.

---

### Question 4: How does the presence of an expiration date change the classical Newsvendor fractile calculation?
**Strong Answer:** In the classical single-period Newsvendor model, the critical fractile is $CF = \frac{c_u}{c_u + c_o}$, where underage cost $c_u = p - c$ and overage cost $c_o = c - s$. When inventory survives for $m$ periods, the decision in period $t$ is not an isolated single-period bet: unsold stock from today carries forward to tomorrow as cohort $m-2$, reducing future procurement needs. However, as stock ages, overage cost is not static; it escalates period by period. Nahmias and Fries proved that the optimal multi-period perishable policy can be approximated by a modified critical fractile where the overage cost is weighted by the multi-period outdating probability $P(W_{t+m} > 0 \mid Q_t)$. As shelf life $m$ shortens, the effective overage cost approaches the full outdating penalty $(c + c_d - s)$, driving the optimal order quantity downward relative to non-perishable base stocks.

**Weak Answer:** The Newsvendor formula has to include waste. If the product expires, you lose more money when you over-order, so you order less than you would for non-perishables.

---

### Question 5: Why can applying standard safety stock formulas ($z \sigma_L$) cause systemic failure in perishable supply chains?
**Strong Answer:** Standard safety stock formulas assume that inventory can dwell indefinitely to buffer demand shocks across lead time $L$. In perishables, safety stock has a finite retention horizon. If the target safety stock quantity $SS = z \sigma \sqrt{L}$ implies an inventory turnover period that exceeds the usable shelf life $m$ (i.e., $SS / \mu > m - L$), the safety stock will physically rot on the shelf before the buffer is ever called upon by demand. In effect, the supply chain is holding "phantom safety stock" that provides zero stockout protection while generating guaranteed recurring disposal costs. Perishable safety stock must be bounded above by the maximum outdating constraint: $SS \le \sum_{t=1}^{m - L} \hat{D}_t - Q_{\text{cycle}}$.

**Weak Answer:** Standard safety stock uses normal distributions, but food demand isn't normal. Also, if you keep safety stock too long, it goes bad, so the math doesn't work.

---

### Question 6: Describe how GS1-128 and 2D Digital Link barcodes enable automated FEFO execution.
**Strong Answer:** Standard EAN-13 barcodes encode only the GTIN, providing no lot or expiration data. GS1-128 linear barcodes and GS1 Digital Link 2D QR codes incorporate standardized Application Identifiers (AIs), specifically AI (01) for GTIN, AI (10) for Lot/Batch Number, and AI (17) for Expiration Date in `YYMMDD` format. When scanned during warehouse putaway, the WMS binds the physical pallet location to the exact expiration date. When picking orders, the WMS directs forklift operators to the specific pallet bin holding the earliest expiration date (system-enforced FEFO) and requires a scan confirmation of the lot code. At retail POS, scanning a 2D Digital Link code parses AI (17); if current date $> \text{AI (17)}$, the register blocks the transaction, preventing expired product sales.

**Weak Answer:** GS1 barcodes have numbers in parentheses that tell the computer when the product expires. This lets the warehouse computer tell the forklift driver which pallet to pick first so old stuff goes out first.

---

### Question 7: What is the "Matching Principle" in multi-echelon perishable inventory allocation?
**Strong Answer:** The Matching Principle states that in a multi-echelon network holding heterogeneous age cohorts, older inventory (lower remaining shelf life) should be allocated to downstream nodes with high demand velocity and short transit times, while fresh inventory (higher remaining shelf life) should be preserved for nodes with low demand velocity or long transit times. By matching the remaining shelf life vector to the node's depletion velocity vector ($\text{RSL}_j \propto 1 / \lambda_i$), the network ensures that aging cohorts clear before reaching expiration, while slower-turning rural stores receive the freshness buffer required to avoid in-store outdating and retail dock gate rejections.

**Weak Answer:** The matching principle means matching the age of the food to what the customer wants. You send fresh food to good stores and older food to stores that sell things quickly.

---

### Question 8: How do dynamic markdowns interact with full-price cannibalization, and how do you prevent margin leakage?
**Strong Answer:** Dynamic markdowns stimulate demand for near-expiry inventory by reducing prices. However, if full-price shoppers switch to purchasing discounted items, the retailer suffers cannibalization margin leakage: $\Delta \text{Profit} = (p_{\text{discount}} - c) - (p_{\text{full}} - c) < 0$. To prevent margin leakage, retailers must erect purchasing friction barriers: (1) **Temporal gating:** applying markdowns only late in the retail cycle (e.g., after 8:00 PM) when full-price basket shoppers have departed. (2) **Spatial separation:** placing marked-down goods in dedicated "Reduced to Clear" clearance bunkers away from primary category shelving. (3) **Algorithmic elasticity tuning:** calculating discounts dynamically so price cuts occur only when local inventory exceeds expected un-discounted clearance velocity.

**Weak Answer:** Dynamic markdowns mean putting yellow stickers on old food. If you discount it too early, people who would have paid full price buy the cheap food instead. You fix this by only discounting at the end of the day.


### Question 9: How does the Arrhenius kinetic model connect cold-chain IoT telemetry to automated WMS gate decisions?
**Strong Answer:** In standard logistics, WMS systems evaluate Remaining Shelf Life (RSL) purely as a calendar subtraction: $T_{\text{exp}} - T_{\text{receipt}}$. However, biochemical degradation kinetics (microbial growth, enzymatic breakdown, lipid oxidation) accelerate exponentially with temperature according to the Arrhenius relation: $k(T) = A \exp(-E_a / (R T))$. When IoT temperature loggers transmit time-temperature series from a refrigerated container, the analytics platform computes the thermal acceleration ratio $A_T = \exp\left(\frac{E_a}{R}\left(\frac{1}{T_{\text{standard}}} - \frac{1}{T_{\text{excurs}}}\right)\right)$ and integrates the cumulative effective shelf life consumed: $\Delta t_{\text{eff}} = \int A_T(t) dt$. If a 12-hour excursion at $12^\circ\text{C}$ consumes 3.5 equivalent days of life, the WMS updates the effective remaining shelf life downward. If this Arrhenius-adjusted RSL falls below the contractual MRSL threshold (e.g., $<75\%$), the WMS triggers an automated gate refusal before the load is unloaded, preventing thermally damaged goods from entering the retail network.

**Weak Answer:** The Arrhenius formula calculates how hot the truck was and how much faster bacteria grow. If the logger shows it got too warm, the warehouse rejects the shipment.

---

### Question 10: How can an inventory optimization model mathematically prevent the "stagnation trap" under customer cherry-picking?
**Strong Answer:** Under mixed customer depletion where cherry-picking occurs ($(1-\alpha) > 0$), the oldest cohort $x_{1,t}$ receives only $\alpha D_t$ of demand. If $\alpha D_t < x_{1,t}$, outdating is mathematically guaranteed: $W_t = x_{1,t} - \alpha D_t > 0$. To prevent this stagnation trap, the optimization model must deploy two linked levers: (1) **Replenishment Throttling:** The model constrains the inbound order quantity $Q_t$ such that the total forward-looking supply across all cohorts does not exceed the effective compliant absorption rate: $\sum_{j=1}^{k} x_{j,t} \le \sum_{\tau=1}^{k} \mathbb{E}[\alpha D_{t+\tau}]$. (2) **Dynamic Markdown Triggers:** The model monitors the ratio of cohort stock to compliant velocity: $\rho_{1,t} = x_{1,t} / (\alpha \hat{D}_t)$. When $\rho_{1,t} > 1$, the system automatically triggers a price reduction $\delta^*$ calculated to stimulate price elasticity $\epsilon$ such that $(1 + \epsilon \delta^*) \alpha \hat{D}_t \ge x_{1,t}$, clearing the cohort before midnight expiration.

**Weak Answer:** You prevent stagnation by not ordering too much inventory and putting discount stickers on the items at the front of the shelf so people buy them.

---

### Question 11: Under what operational circumstances should a central distribution center intentionally violate FEFO by dispatching newer inventory to a store?
**Strong Answer:** A CDC should intentionally deviate from strict network-wide FEFO under two structural conditions: (1) **Asymmetric Transit and Velocity Mismatches (The Matching Principle):** When supplying a remote, slow-turning rural store (transit time 3 days, sales velocity 5 units/day), shipping older stock with 7 days remaining guarantees that the product will either violate the store's receiving MRSL covenant or expire on the retail shelf. The CDC routes fresher stock (e.g., 20 days remaining) to the remote node, while directing the 7-day stock to an adjacent urban flagship store (transit time 2 hours, velocity 100 units/day) where it will be completely cleared within 12 hours. (2) **Contractual Minimum RSL Gating:** When an institutional or commercial customer (e.g., a hospital or export customer) contractually enforces an $85\%$ MRSL requirement, the CDC must dispatch newer stock to clear the contractual covenant, while allocating remaining older stock to domestic retail channels operating under lower $60\%$ thresholds.

**Weak Answer:** You violate FEFO when a store specifically asks for fresher products or when you have to send food on a long truck drive so it doesn't rot on the road.

---

### Question 12: What are the architectural differences between batch-level and serialized item-level traceability for perishable inventory?
**Strong Answer:** Batch-level traceability tracks inventory by production lot identifier (GS1 AI 10) aggregated at the case or pallet level (SSCC). All units within the batch share an identical assumed expiration date (AI 17). This architecture suffices for wholesale warehouse management and bulk recalls, but fails completely at the retail shelf because it cannot distinguish which specific physical units were taken by cherry-picking consumers. Serialized item-level traceability (using GS1 2D Digital Link QR codes or RFID tags) assigns an EPC (Electronic Product Code) unique to every individual unit (SGTIN). The POS scanner records the exact serial number and expiration date of each unit sold. This enables real-time age-structured inventory auditing, automated POS blocking of out-of-date items, algorithmic individual-pack markdown tracking, and instant elimination of phantom inventory discrepancies.

**Weak Answer:** Batch tracking tracks the whole pallet with a lot number, while serialized tracking puts a unique barcode on every single package. Serialized tracking is better because you know exactly which item was sold.
---

## 15. Key terms

- **Age Cohort:** A discrete subset of inventory sharing an identical production, packaging, or expiration date.
- **Application Identifier (AI):** A standardized two- to four-digit GS1 prefix defining the data format and meaning of following characters (e.g., `AI 17` for expiration date).
- **Arrhenius Kinetic Model:** A biochemical equation modeling the exponential relationship between temperature and biological reaction rates.
- **Base-Stock Policy:** An inventory replenishment rule that orders the difference between a target level $S$ and current inventory position.
- **Cherry-Picking:** Consumer behavior prioritizing the selection of the freshest available stock from a retail display.
- **Critical Fractile:** The optimal target service level in Newsvendor models balancing underage and overage penalties.
- **Depletion Vector:** The mathematical mapping specifying which age cohorts satisfy incoming demand under FEFO or LIFO.
- **Digital Link:** GS1 standard encoding GTIN, lot, and expiration data within web-resolvable 2D QR codes.
- **Dynamic Markdown:** Algorithmic price discounting applied as a function of remaining shelf life to accelerate inventory clearance.
- **EDI 856 (ASN):** Electronic Data Interchange Advance Ship Notice communicating pallet lot codes and expiration dates prior to arrival.
- **EDI 861:** EDI transaction set used by receiving docks to communicate receipt confirmations or formal MRSL gate rejections.
- **Effective Inventory Position:** The inventory position adjusted downward by the expected volume of current stock that will outdate.
- **FEFO (First Expired, First Out):** An operational rotation rule ensuring goods with the nearest expiration date are dispatched first.
- **Fixed Shelf Life ($m$):** The constant, deterministic lifetime assigned to an inventory item before mandatory expiration.
- **Food Traceability Rule (FSMA 204):** FDA regulation requiring continuous tracking of Key Data Elements across Critical Tracking Events.
- **Gate Rejection:** The physical and administrative refusal by a receiving warehouse to accept a shipment violating MRSL covenants.
- **GS1 DataBar:** A compact linear barcode capable of encoding expiration dates and lot numbers directly at retail point of sale.
- **LIFO Depletion:** An issuing or consumption pattern where the newest inventory is consumed before older inventory.
- **Minimum Remaining Shelf Life (MRSL):** Contractual threshold of remaining shelf life required for inbound shipment acceptance.
- **Nahmias Equations:** Recursive mathematical equations modeling cohort depletion, transitions, and outdates in perishable inventory.
- **Outdating ($W_t$):** The physical and accounting event occurring when inventory reaches age $m$ unconsumed and must be written off.
- **Perishable Base-Stock:** An order-up-to policy calibrated to account for expected multi-period outdating losses.
- **Proportionate RSL ($\text{RSL}\%$):** Ratio of remaining shelf life to total shelf life, expressed as a percentage.
- **Remaining Shelf Life (RSL):** Elapsed time remaining before product expiration.
- **Salvage Value ($s$):** Revenue recovered per unit from secondary liquidation or animal feed conversion.
- **Stagnation Trap:** Operational failure mode where front-facing stock is bypassed until 100% expires.
- **Sunrise 2027:** The GS1 global initiative transitioning retail point-of-sale scanning from 1D barcodes to 2D DataMatrix and QR codes.
- **Two-Master Conflict:** The operational friction between warehouse FEFO rotation goals and consumer LIFO purchasing preferences.
- **Virtual Depletion Horizon:** The forward-looking time required for projected customer demand to exhaust an on-hand cohort.
- **Waste Penalty ($\theta$):** Net economic loss per unit outdated, combining procurement cost, handling fees, and disposal charges.

---

## 16. Sources

- **Bakker, M., Riezebos, J., & Teunter, R. H.** (2012). Review of inventory systems with deterioration since 2001. *European Journal of Operational Research*, 221(2), 275–284. https://doi.org/10.1016/j.ejor.2012.03.004
- **Broekmeulen, R. A. C. M., & van Donselaar, K. H.** (2009). A heuristic to manage perishable inventory with customer depletion behavior. *International Journal of Production Economics*, 118(1), 133–142. https://doi.org/10.1016/j.ijpe.2008.08.039
- **Ferguson, M., & Ketzenberg, M.** (2006). Information sharing to improve retail product freshness of perishables. *Production and Operations Management*, 15(1), 57–73. https://doi.org/10.1111/j.1937-5956.2006.tb00004.x
- **Fries, B. E.** (1975). Optimal ordering policy for a perishable commodity with fixed lifetime. *Operations Research*, 23(1), 46–61. https://doi.org/10.1287/opre.23.1.46
- **GS1 AISBL.** (2024). *GS1 General Specifications, Standard Release 24.0*. Brussels, Belgium: GS1 Global Office.
- **Karaesmen, I. Z., Scheller-Wolf, A., & Deniz, B.** (2011). Managing perishable and aging inventories: Review and future research directions. In *Planning Production and Inventories in the Extended Enterprise* (pp. 393–436). Springer, New York. https://doi.org/10.1007/978-1-4419-6485-4_15
- **Nahmias, S.** (1975). Optimal ordering policies for perishable inventory—I. *Operations Research*, 23(4), 735–749. https://doi.org/10.1287/opre.23.4.735
- **Nahmias, S.** (1977). On ordering perishable inventory when both demand and lifetime are random. *Management Science*, 24(1), 82–90. https://doi.org/10.1287/mnsc.24.1.82
- **Nahmias, S.** (1982). Perishable inventory theory: A review. *Operations Research*, 30(4), 680–708. https://doi.org/10.1287/opre.30.4.680
- **Pierskalla, W. P., & Roach, C. D.** (1972). Optimal issuing policies for perishable inventory. *Management Science*, 18(11), 603–614. https://doi.org/10.1287/mnsc.18.11.603
- **Silver, E. A., Pyke, D. F., & Peterson, R.** (1998). *Inventory Management and Production Planning and Scheduling* (3rd ed.). John Wiley & Sons.
- **U.S. Food and Drug Administration.** (2022). *Requirements for Additional Traceability Records for Certain Foods (FSMA 204 Final Rule)*. 21 CFR Part 1, Subpart S. Federal Register, 87 FR 70910.

---

## 17. Links

- **Parent Module:** [`_index.md`](_index.md) — Food, Agriculture, and Perishable Supply Chains
- **Survey File:** [`../../../course/32_food_agriculture_perishables.md`](../../../course/32_food_agriculture_perishables.md)
- **Sibling Topics:**
  - [`01_agri-production-harvest.md`](01_agri-production-harvest.md) — Agricultural Production and Harvest Planning
  - [`02_perishable-forecasting.md`](02_perishable-forecasting.md) — Forecasting Perishable Demand
  - [`04_newsvendor-salvage.md`](04_newsvendor-salvage.md) — Newsvendor with Salvage and Freshness Pricing
  - [`05_cold-chain-quality.md`](05_cold-chain-quality.md) — Cold Chain and Quality
  - [`06_traceability-food-safety.md`](06_traceability-food-safety.md) — Traceability and Food Safety
- **Cross-Program Foundations:**
  - [`../../02_methods/10_inventory_optimization_multi_echelon/03_newsvendor.md`](../../02_methods/10_inventory_optimization_multi_echelon/03_newsvendor.md) — Single-Period Inventory Models
  - [`../../02_methods/10_inventory_optimization_multi_echelon/06_policy-families.md`](../../02_methods/10_inventory_optimization_multi_echelon/06_policy-families.md) — Base-Stock and $(s, S)$ Inventory Policies
  - [`../../01_foundations/03_data_engineering_interoperability/09_gs1-epcis.md`](../../01_foundations/03_data_engineering_interoperability/09_gs1-epcis.md) — GS1 Identification and EPCIS Tracking
