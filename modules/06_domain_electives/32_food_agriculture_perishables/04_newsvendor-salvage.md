# Newsvendor with Salvage, Disposal Fees, and Dynamic Freshness Pricing

> Module 32 — Food, Agriculture, and Perishable Supply Chains
> Topic 04 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Module 10 (Inventory Optimization and Multi-Echelon), Topic 02 (Perishable Forecasting), Topic 03 (Shelf Life, FEFO, and Age-Structured Inventory)
> Version 2026.5

---

## 1. Why this topic exists

Classic inventory theory teaches the single-period newsvendor model as the foundational archetype for stochastic demand under capacity constraints. In that standard textbook treatment, a merchant faces uncertain demand $D$, chooses order quantity $Q$ at unit cost $c$, sells each unit at retail price $p$, and liquidates unsold stock at a positive or zero salvage value $s \ge 0$. The resulting critical fractile $CR^* = (p - c) / (p - s)$ balances the marginal profit of an additional sale against the marginal loss of holding unsold scrap. In durable goods or non-perishable general merchandise, unsold stock retains salvage value through secondary liquidators, outlet channels, or off-price wholesalers (e.g., TJX, Ross).

In grocery retail, fresh prepared foods, cut produce, chilled proteins, bakery, dairy, and commercial catering, that classical formulation fails in three fundamental operational realities:

1. **Negative Salvage Values (Disposal Fees and Regulatory Penalties):** Unsold perishable inventory does not sit harmlessly on a scrap pallet awaiting a secondary liquidator. When perishable food reaches terminal expiration, it becomes biological waste. Retailers must pay direct landfill tipping fees, organic waste hauling surcharges, biohazard handling costs, and refrigeration energy to store waste before collection. Furthermore, modern regulatory mandates—such as California Senate Bill 1383 (SB 1383), the European Union Landfill Directive (Directive 1999/31/EC), and the French Loi Garot—impose statutory civil fines and commercial generator diversion quotas on organic waste sent to landfills. Consequently, the net salvage value is frequently negative: $\bar{s} = s - d_f < 0$, where $d_f$ represents unit disposal, compliance, and handling costs. When $d_f > s$, the unit overage cost exceeds the original procurement cost ($c_o = c - \bar{s} = c - s + d_f > c$).
2. **Dynamic Freshness Interventions Prior to Expiration:** Grocery operators do not maintain full retail price until midnight of the expiration date and then instantly discard 100% of remaining stock. Instead, retailers possess intermediate operational levers: dynamic freshness pricing, staged markdowns (e.g., progressive 20%, 40%, 60% discounts via Electronic Shelf Labels or yellow stickers), and app-based secondary clearance platforms (e.g., Flashfood, Too Good To Go). Dynamic markdown pricing converts potential terminal bio-waste into recovered cash flow, altering the distribution of terminal surplus.
3. **Cannibalization of Full-Price Demand:** Staged markdowns introduce a dangerous behavioral feedback loop. When discounted items dwell on the same retail display alongside full-price, fresher items, a fraction $\gamma$ of regular-price shoppers switch to the discounted stock. If markdown timing is set too early or discounts are too steep, the retailer cannibalizes its own high-margin sales to salvage low-margin stock, destroying net gross margin. Conversely, marking down stock too late results in clearance velocity failure, dumping the remaining volume directly into municipal waste streams at full disposal penalty.

The economic friction varies dramatically across retail supermarket departments:
- **Fresh Meat and Chilled Seafood:** Characterized by high unit procurement costs ($c = \$6.00 - \$22.00/\text{lb}$) and rapid bacterial spoilage. When outdating occurs, strict public health sanitary codes prohibit donation of opened cuts, forcing immediate chemical denaturation or specialized rendering disposal. A single misjudged salmon or ribeye order produces immense cash losses.
- **In-Store Artisan Bakery:** Very short lifecycle ($T = 1 \text{ day}$). Unsold baguettes and pastries have essentially zero residual salvage value ($s = \$0.00$). While bakery products carry high gross margins ($p_1/c > 3.0$), unmanaged over-ordering floods the department with clearance bins that destroy primary sales velocity.
- **Prepared Foods and Chilled Deli:** High labor-added cost, strict 24- to 72-hour pull dates, and severe municipal organic waste compliance audit scrutiny.

Operating margins in retail grocery hover between 1.5% and 2.5% of gross revenue. Shrinkage, markdowns, and perishable write-offs collectively destroy 8% to 14% of gross perishable category revenue. Traditional Enterprise Resource Planning (ERP) and Advanced Planning and Scheduling (APS) engines compute replenishment using standard target cycle service levels (e.g., 95% or 98% CSL) designed for non-perishable distribution centers. Applying these unadjusted targets to fresh retail shelves forces stores to chronically over-order, converting working capital into discarded bio-waste. Optimizing the joint problem of procurement volume $Q^*$, markdown timing $t_m^*$, and freshness-dependent discount depth $p_2^*$ represents the single highest-leverage profit lever in fresh food retail operations.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Formulate and solve** the extended newsvendor model incorporating positive salvage scrap, negative landfill disposal fees, hauling surcharges, and statutory organic waste penalties.
2. **Derive** the exact critical fractile $CR^*$ when unit overage cost strictly exceeds unit procurement cost ($c_o > c$), proving why classical inventory models systematically over-order perishables.
3. **Model** dynamic freshness pricing across two stages (full-price primary window vs. clearance markdown window) under stochastic demand with consumer substitution.
4. **Quantify** demand cannibalization using cross-price elasticity and consumer willingness-to-pay (WTP) segmentation between freshness-sensitive and price-sensitive consumer cohorts.
5. **Calculate** the cannibalization breakeven threshold that dictates whether an in-store yellow-sticker discount generates incremental gross profit or destroys net category margin.
6. **Formulate** continuous-time freshness markdown policies using dynamic programming and Hamilton-Jacobi-Bellman (HJB) hazard rate formulations.
7. **Evaluate** multi-channel clearance hierarchies spanning in-store markdowns, secondary dynamic apps, food bank tax donations (IRC Section 170(e)(3)), animal feed rendering, and anaerobic digestion.
8. **Specify** operational architectures for automated dynamic pricing engines operating over Electronic Shelf Labels (ESL), 2D GS1 Digital Link barcodes, and POS gatekeepers.
9. **Audit** retail supply chain inventory ledgers to ensure Lower of Cost and Net Realizable Value (LCNRV) compliance under GAAP ASC 330 and IFRS IAS 2.

---

## 3. The core mental model

Perishable inventory management is not a static single-shot bet; it is a multi-tier dynamic depletion funnel. At each stage of product life, management must balance current margin realization against the downstream risk of disposal cost escalation.

```text
                     THE PERISHABLE CLEARANCE & SALVAGE FUNNEL
                     
  Wholesale Inbound (Q) at Unit Cost c
           |
           v
  +-------------------------------------------------------------+
  |  STAGE 1: FULL-PRICE WINDOW [0, tm)                         |
  |  Price = p1 | Margin = p1 - c                               |
  |  Primary Demand Realized: D1 ~ F1(d1)                       |
  +-------------------------------------------------------------+
           |                                  |
           | Realized Sales min(Q, D1)         | Leftover Inventory
           v                                  v I1 = max(0, Q - D1)
    [ Cash Inflow: p1 * Sales ]               |
                                              v
                              +-----------------------------------------------+
                              |  STAGE 2: DYNAMIC MARKDOWN WINDOW [tm, Texp)  |
                              |  Price = p2 = (1 - delta) * p1                |
                              |  Demand: Organic Clearance D2 + Cannibalized  |
                              +-----------------------------------------------+
                                              |                         |
                                              | Clearance Sales         | Leftover Unsold
                                              v min(I1, D2)             v I2 = max(0, I1 - D2)
                                       [ Cash Inflow ]                  |
                                       [ p2 * Clearance ]               v
                                                    +---------------------------------------+
                                                    |  STAGE 3: MULTI-CHANNEL SALVAGE       |
                                                    |  Hierarchy of Residual Value          |
                                                    +---------------------------------------+
                                                     /          |             |           \
                                                    /           |             |            \
                                                   v            v             v             v
                                           [ Donation ]   [ Feed/Bio ]  [ Digestion ]  [ Landfill ]
                                            Tax Credit     Rendering     Gate Fee       Tipping Fee
                                            s_don > 0      s_feed > 0    -df_bio < 0    -df_land < 0
```

### 3.1 The Three Interlocking Mechanisms

The economic life of perishable inventory is determined by three interacting forces:

1. **The Asymmetric Overage Hammer:** When an unsold unit reaches terminal expiration, the business does not merely forfeit the anticipated revenue ($p - c$); it loses the unrecovered procurement capital $c$, plus physical handling labor, plus third-party waste hauling and tipping charges $d_f$. In mathematical terms, the loss surface steepens sharply to the right of realized demand:
   $$\text{Loss}(Q > D) = (Q - D) \cdot (c - s + d_f)$$
   Because overage penalty grows linearly with $(c + d_f)$, the optimal procurement point shifts systematically below the historical mean demand.

2. **The Markdown Substitution Tension:** Introducing a price discount $p_2 < p_1$ during the final operational window $[t_m, T_{\text{exp}})$ creates an immediate conflict between two consumer segments:
   - *Bargain Hunters (Segment B):* Consumers who would never purchase at $p_1$ enter the market at $p_2$, creating purely incremental volume $\Delta D_{\text{clearance}}$.
   - *Freshness Indifferent Shoppers (Segment A Cross-Shoppers):* Consumers who intended to purchase at $p_1$ notice identical or acceptable quality in the markdown bin and purchase at $p_2$ instead, causing cannibalization volume $D_{\text{cann}} = \gamma \cdot D_1$.
   The net economic gain of the markdown exists if and only if the gross margin recovered from bargain hunters exceeds the margin surrendered to cannibalized shoppers plus terminal disposal fees avoided.

3. **The Multi-Tier Value Cascade:** Physical disposal is the terminal state of a stepped cascade. Inventory that remains unsold at retail clearance does not immediately enter a municipal dump. Sophisticated operators cycle stock through a ranked waterfall:
   - Primary In-Store Retail ($p_1$)
   - Secondary In-Store Markdown / App Clearance ($p_2$)
   - Institutional Donation (IRC §170(e)(3) tax deductions yielding cash-equivalent value)
   - Secondary Agricultural Salvage (Animal feed, protein rendering at low positive $s$)
   - Industrial Valorization (Anaerobic digestion, composting at low tipping fee $-d_{f,\text{bio}}$)
   - Municipal Incineration / Landfill (Standard commercial disposal at high tipping fee $-d_{f,\text{land}}$)

### 3.2 State Space and Information Horizons

The system moves across distinct operational epochs where information and decision authority evolve:
- **At Order Commitment ($t = -L$):** The replenishment planner observes only historical demand distributions $F(D)$, supplier lead times $L$, wholesale costs $c$, and contractual disposal fee structures $d_f$. Procurement $Q$ is locked before demand uncertainty is resolved.
- **At Shelf Arrival ($t = 0$):** Inventory is placed on retail display at full price $p_1$. The retailer observes real-time POS scan velocity. Shoppers self-select based on date stamps and freshness perception.
- **At Markdown Review Point ($t = t_m$):** The retailer observes remaining on-hand stock $I_1 = \max(0, Q - D_1)$ and remaining shelf life $\tau = T_{\text{exp}} - t_m$. The decision variable transitions from inventory quantity to pricing vector $p_2$.
- **At Terminal Expiration ($t = T_{\text{exp}}$):** Commercial retail legality ceases. The inventory state is binary: either transferred to approved food recovery channels or de-packaged into biological waste streams.

### 3.3 The Feedback Dynamics of Strategic Shopper Learning

When markdowns follow predictable, deterministic patterns (e.g., store clerks applying yellow stickers every evening at 17:00), consumer behavior adapts endogenously. Shoppers who possess high willingness-to-pay begin delaying purchases from the morning full-price window into the late afternoon clearance window. This behavioral migration erodes base demand $D_1$ and inflates apparent clearance volume, tricking static demand forecasting algorithms into recommending even deeper markdowns. Breaking this strategic waiting loop requires algorithmic, stochastic, or continuous dynamic pricing executed via digital shelf infrastructure.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Operational Misuse | Measurement Unit / Standard |
|---|---|---|---|
| **Procurement Cost ($c$)** | Unit purchase price paid to grower or processor plus inbound freight landed at store or distribution center. | Omitting inbound refrigerated freight and pallet unbundling costs from base cost. | Currency per unit ($\$/\text{unit}$). |
| **Retail Price ($p_1$)** | Primary selling price for fresh, full-life product during initial sales window. | Treating $p_1$ as constant when seasonal base prices float weekly. | Currency per unit ($\$/\text{unit}$). |
| **Markdown Price ($p_2$)** | Discounted selling price deployed during secondary clearance window: $p_2 = (1 - \delta)p_1$. | Treating markdown price as pure profit without subtracting cannibalization losses. | Currency per unit ($\$/\text{unit}$). |
| **Discount Depth ($\delta$)** | Percentage reduction from primary retail price: $\delta = (p_1 - p_2) / p_1$. | Setting uniform discounts across categories with widely varying price elasticities. | Dimensionless ratio ($\% \in [0, 1]$). |
| **Markdown Trigger ($t_m$)** | Timestamp or remaining shelf life threshold at which price drops from $p_1$ to $p_2$. | Triggering markdowns based on clerk intuition rather than remaining hours of shelf life. | Elapsed time or remaining shelf life ($h$, days). |
| **Disposal Fee ($d_f$)** | Net per-unit cost to discard, haul, and incinerate/landfill expired bio-waste. | Assuming waste disposal is free or subsumed in building lease overhead. | Currency per unit or currency per metric ton ($\$/\text{unit}$, $\$ / \text{t}$). |
| **Landfill Tipping Fee ($T_{\text{tip}}$)** | Gate charge levied by solid waste facilities per ton of municipal solid waste unloaded. | Conflating hauling container rental with per-ton tipping fees. | Currency per ton ($\$/\text{ton}$). |
| **Net Salvage Value ($\bar{s}$)** | Effective net financial recovery per unsold unit: $\bar{s} = s - d_f$. | Assuming salvage value is strictly positive ($\bar{s} \ge 0$). | Currency per unit ($\$/\text{unit}$). |
| **Underage Cost ($c_u$)** | Opportunity cost of under-ordering by one unit: $c_u = p_1 - c + g$. | Ignoring customer goodwill and stockout penalty ($g$). | Currency per unit ($\$/\text{unit}$). |
| **Overage Cost ($c_o$)** | Financial loss incurred by over-ordering by one unit: $c_o = c - \bar{s} = c - s + d_f$. | Capping overage cost at procurement cost $c$, ignoring disposal fees. | Currency per unit ($\$/\text{unit}$). |
| **Critical Fractile ($CR^*$)** | Optimal cumulative service probability balancing underage and overage: $c_u / (c_u + c_o)$. | Using standard service levels (e.g., 95%) without calculating economic costs. | Dimensionless probability ($[0, 1]$). |
| **Cannibalization Rate ($\gamma$)** | Fraction of full-price demand that shifts to discounted stock when both are visible: $D_{\text{cann}} / D_1$. | Assuming all clearance sales represent new, incremental customers. | Dimensionless ratio ($\gamma \in [0, 1]$). |
| **Cannibalization Breakeven ($\gamma_{\max}$)** | The maximum tolerable switching rate where markdown profit equals salvage-only profit. | Applying markdowns when actual switching rate strictly exceeds the breakeven point. | Dimensionless ratio ($\gamma \in [0, 1]$). |
| **Willingness-to-Pay (WTP)** | Maximum price a consumer will pay for a perishable unit at remaining shelf life $\tau$. | Assuming consumer WTP is static regardless of expiration date proximity. | Currency ($\$$). |
| **Electronic Shelf Label (ESL)** | Digital e-paper shelf tag updated centrally via radio frequency (RF) to display real-time dynamic prices. | Using ESL purely for labor savings rather than algorithmic intra-day markdowns. | Hardware device / IoT infrastructure. |
| **GS1 Digital Link** | 2D barcode format encoding GTIN, batch/lot, and expiration date (AI 17) in a web URI. | Scanning only 1D UPC barcodes that lack expiration date visibility at POS. | ISO/IEC 15424 / GS1 Standard. |
| **GS1 Application Identifier AI (17)** | Standardized two-digit prefix denoting expiration date formatted as `YYMMDD`. | Confusing expiration date AI (17) with best-before date AI (15). | GS1 Data Syntax `(17)YYMMDD`. |
| **Net Realizable Value (NRV)** | Estimated selling price in ordinary course of business less completion and disposal costs (IAS 2 / ASC 330). | Failing to write down perishable inventory balance sheet value prior to physical disposal. | Accounting valuation ($\$$). |
| **Organic Diversion Mandate** | Statutory requirement prohibiting commercial edible organic food from entering municipal landfills. | Treating municipal diversion ordinances as optional corporate social responsibility targets. | Statutory mandate (e.g., CA SB 1383). |
| **Anaerobic Digestion (AD)** | Biological process breaking down organic waste into biogas and digestate in sealed reactors. | Recording AD gate fees as revenue rather than reduced disposal expense. | Waste processing classification. |
| **Expected Value of Dynamic Pricing (EVDP)** | Net expected profit improvement achieved by dynamic markdowns over static single-price newsvendor. | Comparing dynamic revenue against zero-order baseline rather than static optimal newsvendor. | Currency per replenishment cycle ($\$$). |
| **Yellow Sticker** | Physical adhesive barcode applied to packaging indicating an in-store manual markdown. | Overlooking clerk labor cost and sticker scanning failure rates at point of sale. | Operational packaging artifact. |
| **IRC § 170(e)(3)** | Internal Revenue Code provision permitting enhanced corporate tax deductions for donated inventory. | Donating expired food that fails wholesomeness standards and loses tax eligibility. | Federal Tax Statute. |
| **Bill Emerson Act** | Federal law granting civil and criminal liability immunity to food donors acting in good faith. | Refusing to donate near-expiration food due to unfounded litigation fears. | 42 U.S.C. § 1791. |
| **Contra-Revenue Accounting** | Accounting classification where retail markdowns reduce reported top-line revenue rather than appearing in COGS. | Treating markdown margin loss as an inventory shrinkage expense. | Financial Reporting Standard. |
| **Terminal Outdating** | Inventory reaching zero remaining shelf life without being consumed or diverted. | Categorizing outdates as inventory shrinkage (loss/theft) rather than spoilage. | Physical units or currency at cost. |

---

## 5. The mechanics, step by step

Managing perishables under salvage, disposal, and markdown dynamics requires a five-step mathematical and operational procedure executed across planning and execution cycles.

```text
                                OPERATIONAL WORKFLOW
                                
  Step 1: Cost Architecture  --> Step 2: Extended Fractile --> Step 3: Markdown Timing
  - Audit procurement c          - Compute co = c - s + df     - Optimize trigger tm
  - Audit disposal fees df       - Compute cu = p - c + g      - Model cannibalization gamma
  - Map statutory fines          - Solve Q* = F^-1(CR*)        - Set discount depth delta
                                                                          |
                                                                          v
  Step 5: Governance & Audit <-- Step 4: Multi-Tier Salvage <-- Step 3b: In-Flight Execution
  - ASC 330 NRV adjustment       - Food bank donation (IRC)     - ESL automated price update
  - GS1 AI(17) POS gatecheck     - Agricultural rendering       - POS dynamic discount lookup
  - Landfill diversion audit     - Bio-waste composting         - Clear physical shelf rail
```

### 5.1 Step 1: Establish the Asymmetric Cost Architecture

Begin by establishing the true fully landed economic cost parameters for the SKU:

1. **Procurement Cost ($c$):** Base purchase price paid to vendor plus inbound refrigerated freight, drayage, and cross-dock handling allocated per unit.
2. **Gross Selling Price ($p_1$):** Nominal full retail price.
3. **Customer Stockout Penalty ($g$):** Quantified brand damage, lost category margin, and customer churn probability resulting from an empty shelf. For standard grocery, $g$ is typically set to $0.15(p_1 - c)$ to $0.35(p_1 - c)$.
4. **Physical and Regulatory Disposal Fee ($d_f$):** Compute the fully loaded terminal cost per unit:
   $$d_f = \frac{T_{\text{tip}} + H_{\text{haul}}}{U_{\text{ton}}} + L_{\text{cler}} + F_{\text{reg}}$$
   where:
   - $T_{\text{tip}}$ is the municipal or private landfill tipping fee per ton ($\$80 - \$160/\text{ton}$).
   - $H_{\text{haul}}$ is the dedicated organic waste container haulage fee allocated per ton ($\$40 - \$90/\text{ton}$).
   - $U_{\text{ton}}$ is the number of packaged SKU units per ton.
   - $L_{\text{cler}}$ is the store-level labor cost to de-pack, scan out, and transfer the expired item to the waste dock ($\$0.15 - \$0.45/\text{unit}$).
   - $F_{\text{reg}}$ is the expected statutory penalty per unit under organic landfill diversion laws (e.g., California SB 1383 administrative civil penalties scaled by audit risk).
5. **Gross Secondary Scrap Value ($s$):** Revenue received from animal feed recyclers, fat-and-bone renderers, or industrial processors. In fresh grocery, $s$ is often zero or negligible ($\$0.02 - \$0.10/\text{unit}$).
6. **Net Salvage ($\bar{s}$):**
   $$\bar{s} = s - d_f$$
   When $d_f > s$, $\bar{s}$ is strictly negative.

### 5.2 Step 2: Derive the Extended Perishable Critical Fractile

In a single-period replenishment cycle with stochastic demand $D$ characterized by cumulative distribution function $F(d)$ and probability density function $f(d)$, the expected profit function $\Pi(Q)$ is formulated as:

$$\Pi(Q) = p_1 \mathbb{E}[\min(Q, D)] + \bar{s} \mathbb{E}[\max(0, Q - D)] - c Q - g \mathbb{E}[\max(0, D - Q)]$$

Expanding the expectation operators across continuous demand:

$$\Pi(Q) = p_1 \left[ \int_0^Q x f(x) dx + Q \int_Q^\infty f(x) dx \right] + \bar{s} \int_0^Q (Q - x) f(x) dx - c Q - g \int_Q^\infty (x - Q) f(x) dx$$

To find the profit-maximizing order quantity $Q^*$, differentiate $\Pi(Q)$ with respect to $Q$ using the Leibniz integral rule:

$$\frac{d\Pi(Q)}{dQ} = p_1 \left[ 0 + \int_Q^\infty f(x) dx \right] + \bar{s} \left[ \int_0^Q 1 \cdot f(x) dx \right] - c - g \left[ - \int_Q^\infty f(x) dx \right]$$

Substitute the cumulative distribution identity $F(Q) = \int_0^Q f(x) dx$ and complementary distribution $1 - F(Q) = \int_Q^\infty f(x) dx$:

$$\frac{d\Pi(Q)}{dQ} = p_1 [1 - F(Q)] + \bar{s} F(Q) - c + g [1 - F(Q)]$$

Set the first derivative to zero:

$$[p_1 - c + g] - [p_1 - \bar{s} + g] F(Q^*) = 0$$

$$F(Q^*) = \frac{p_1 - c + g}{p_1 - \bar{s} + g}$$

Define the marginal economic underage and overage costs:
- **Underage Cost:** $c_u = p_1 - c + g$ (the profit lost per unit of unmet demand plus goodwill penalty).
- **Overage Cost:** $c_o = c - \bar{s} = c - (s - d_f) = c - s + d_f$ (the cash lost per unit of unsold stock, including disposal).

Substituting $c_u$ and $c_o$ into the equation yields the universal critical fractile:

$$CR^* = F(Q^*) = \frac{c_u}{c_u + c_o} = \frac{p_1 - c + g}{(p_1 - c + g) + (c - s + d_f)} = \frac{p_1 - c + g}{p_1 - s + d_f + g}$$

#### Mathematical Insight on Disposal Fees
Differentiating $CR^*$ with respect to the disposal fee $d_f$:

$$\frac{\partial CR^*}{\partial d_f} = \frac{- (p_1 - c + g)}{(p_1 - s + d_f + g)^2} < 0$$

Because $p_1 > c$, the derivative is strictly negative. As municipal landfill tipping fees or statutory waste penalties rise, the optimal service fractile strictly drops. Planners who ignore $d_f$ evaluate $CR_{\text{textbook}} = \frac{p_1 - c}{p_1 - s}$. Because $c_o$ is understated by $d_f$, textbook models overstate $CR^*$ and systematically order excessive stock, exacerbating perishable waste.

### 5.3 Step 3: Model Two-Stage Freshness Markdown and Cannibalization

To mitigate terminal disposal fees, the retailer introduces an intra-cycle markdown window at time $t_m \in (0, T_{\text{exp}})$.
- **Stage 1 (Full-Price Window $[0, t_m)$):** Price is $p_1$. Stochastic demand realized is $D_1 \sim F_1(d_1)$.
- Leftover inventory entering Stage 2 is $I_1 = \max(0, Q - D_1)$.
- **Stage 2 (Clearance Window $[t_m, T_{\text{exp}}]$):** Price is marked down to $p_2 = (1 - \delta)p_1$, where $\delta \in (0, 1)$ is the discount depth.

#### Demand Decomposition in Stage 2
Demand during the clearance window $D_2(p_2, p_1)$ consists of two distinct consumer segments:
1. **Organic Clearance Demand ($D_{2,\text{org}}$):** Price-sensitive bargain hunters who enter the store or check markdown clearance racks specifically for discounted items. This demand follows distribution $F_{2,\text{org}}(x; p_2)$ with price elasticity $\epsilon_2 = \frac{\partial D_2 / D_2}{\partial p_2 / p_2} < -1$.
2. **Cannibalized Primary Demand ($D_{\text{cann}}$):** Regular shoppers who intended to purchase full-price fresh stock at $p_1$ during Stage 2, but switch to marked-down stock upon observing it on the shelf. This volume is modeled as:
   $$D_{\text{cann}} = \gamma(p_2, p_1, \Delta \tau) \cdot D_{1,\text{pot}}$$
   where $\gamma \in [0, 1]$ is the cannibalization coefficient, and $\Delta \tau = T_{\text{exp}} - t_m$ is the remaining shelf life.

Total clearance demand realized is:
$$D_2 = D_{2,\text{org}}(p_2) + D_{\text{cann}}$$

Terminal leftover stock remaining after Stage 2 that must be disposed of is:
$$I_2 = \max(0, I_1 - D_2)$$

#### Backward Induction for Joint Optimization
We solve this two-stage decision process via dynamic programming:

**Stage 2 Subproblem:** Given leftover inventory $I_1$ at trigger time $t_m$, choose markdown price $p_2$ to maximize expected Stage 2 profit $V_2(I_1, p_2)$:

$$V_2(I_1, p_2) = p_2 \mathbb{E}[\min(I_1, D_2)] + \bar{s} \mathbb{E}[\max(0, I_1 - D_2)] - (p_1 - p_2) \mathbb{E}[D_{\text{cann}}] - L_{\text{tag}} \cdot I_1$$

The third term represents the **Cannibalization Margin Penalty**: every unit of cannibalized demand would have transacted at $p_1$, but transacted at $p_2$ instead, destroying $(p_1 - p_2)$ of net cash margin. $L_{\text{tag}}$ represents physical clerk labor if manual sticker application is required.

**The Cannibalization Breakeven Condition:**
A markdown discount $\delta = (p_1 - p_2)/p_1$ generates positive marginal contribution relative to letting leftover stock flow directly to disposal if and only if:

$$\Delta \mathbb{E}[\text{Sales}_{\text{org}}] \cdot (p_2 - \bar{s}) > \mathbb{E}[D_{\text{cann}}] \cdot (p_1 - p_2) + \Delta \text{Labor}_{\text{tagging}}$$

If cannibalization $\gamma$ is high (e.g., in small convenience stores where fresh and clearance items sit in the same refrigerated reach-in case), deep markdowns destroy more margin than they recover in avoided disposal fees.

**Stage 1 Procurement Problem:** With optimal policy $p_2^*(I_1)$ determined, choose initial replenishment order quantity $Q$ to maximize total expected lifecycle profit:

$$\Pi(Q) = \mathbb{E}_{D_1} \left[ p_1 \min(Q, D_1) + V_2(\max(0, Q - D_1), p_2^*) \right] - c Q$$

### 5.4 Step 4: Continuous Freshness Decay and Hazard Rate Pricing

For automated environments equipped with Electronic Shelf Labels (ESL) and continuous POS tracking (e.g., sushi bars, rotisserie chicken, premium artisanal cut meats), price is adjusted continuously or in discrete hourly buckets $t \in [0, T]$.

Let $I(t)$ denote inventory on hand at time $t$, and let $\tau = T - t$ denote remaining shelf life.
Consumer arrival follows a non-homogeneous Poisson process with intensity $\lambda(p, \tau)$ governed by:

$$\lambda(p, \tau) = \lambda_0 \cdot \exp(-\beta p) \cdot \left( \frac{\tau}{T} \right)^\theta$$

where:
- $\beta > 0$ is the price sensitivity parameter.
- $\theta \ge 0$ is the freshness decay elasticity. If $\theta > 0$, consumer arrival rate decays as the product ages, even if price remains unchanged.

Let $V(t, I)$ represent the value function (maximum expected remaining profit from time $t$ to expiration $T$ given $I$ units remaining). The system satisfies the Hamilton-Jacobi-Bellman (HJB) continuous-time dynamic programming equation:

$$\frac{\partial V(t, I)}{\partial t} + \max_{p(t)} \left\{ \lambda(p, T-t) \left[ p + V(t, I - 1) - V(t, I) \right] \right\} = 0$$

subject to the terminal boundary condition:
$$V(T, I) = \bar{s} \cdot I = (s - d_f) \cdot I$$

The optimal dynamic price $p^*(t, I)$ balances instantaneous revenue against the opportunity cost of future inventory depletion:

$$p^*(t, I) = \arg\max_{p} \left\{ \lambda(p, T-t) [p - \Delta V(t, I)] \right\}$$

where $\Delta V(t, I) = V(t, I) - V(t, I-1)$ is the marginal value of holding an additional unit of inventory at time $t$. As $t \to T$, $\Delta V(t, I) \to \bar{s}$. When $\bar{s} < 0$, the retailer is economically motivated to drop price aggressively toward marginal variable cost to avoid the negative terminal salvage boundary.

### 5.5 Step 5: Multi-Channel Salvage Hierarchies and Donation Economics

When commercial retail clearance fails to deplete remaining inventory before expiration date minus 24 hours, the product enters the **Secondary Salvage Waterfall**:

```text
  Terminal Clearance Waterfall:
  
  [ Level 1: App-Based Flash Clearance (e.g., Flashfood / Too Good To Go) ]
  Recover: 30% to 50% of Retail Price p1
  Cost: Platform transaction commission (approx. 20% to 30% of realized p2)
  Net Recovery: s_app = 0.70 * p2 > 0
            |
            v  (If unsold at Texp - 12 hours)
  [ Level 2: Institutional Charitable Food Donation ]
  Recover: Internal Revenue Code (IRC) Section 170(e)(3) Enhanced Tax Deduction
  Tax Deduction = Min(Cost + 0.5 * Gross Margin, 2 * Cost)
  Cash Value: s_tax = Effective Corporate Tax Rate * Tax Deduction
            |
            v  (If rejected by Food Bank due to cold chain / packaging limits)
  [ Level 3: Agricultural Salvage (Animal Feed / Rendering) ]
  Recover: Bulk farm gate scrap purchase ($15 to $35 per metric ton)
  Net Recovery: s_feed = approx. $0.02 to $0.05 per unit
            |
            v  (If unsuitable for animal feed under livestock feed safety laws)
  [ Level 4: Industrial Valorization (Anaerobic Digestion / Composting) ]
  Cost: Commercial biological waste tipping fee (-$35 to -$60 per metric ton)
  Net Fee: df_bio = -$0.05 to -$0.15 per unit (Avoids landfill surcharge)
            |
            v  (Last Resort: Non-Compliant Municipal Waste)
  [ Level 5: Municipal Solid Waste Landfill ]
  Cost: Full municipal tipping fee + hauling + regulatory fines (-$120 to -$220/ton)
  Net Fee: df_land = -$0.40 to -$1.50 per unit
```

#### The Tax Arbitrage of Commercial Donation
Under US Internal Revenue Code (IRC) § 170(e)(3), a C-corporation that donates wholesome food inventory to a qualified 501(c)(3) non-profit organization serving the ill, needy, or infants is entitled to an **enhanced tax deduction** equal to:

$$\text{Deduction} = \text{Cost } (c) + \frac{1}{2} [\text{Fair Market Value } (p_1) - \text{Cost } (c)]$$

subject to the statutory cap:
$$\text{Deduction}_{\max} \le 2 \cdot \text{Cost } (c)$$

The effective cash recovery per donated unit via tax liability reduction is:
$$s_{\text{tax}} = \tau_{\text{corp}} \cdot \text{Deduction}$$
where $\tau_{\text{corp}}$ is the corporate marginal income tax rate (e.g., $21\%$ federal plus applicable state income tax, totaling roughly $25\% - 28\%$).

If a high-margin prepared food SKU has $c = \$4.00$ and $p_1 = \$12.00$:
- Standard mark-to-market tax deduction: $\$4.00 + 0.5(\$12.00 - \$4.00) = \$8.00$.
- Check cap: $2 \times \$4.00 = \$8.00$ (exactly at cap).
- Cash value generated at $\tau_{\text{corp}} = 25\%$: $s_{\text{tax}} = 0.25 \times \$8.00 = \$2.00$ per unit.

Comparing donation against secondary markdown:
- If marking down to $p_2 = \$3.00$ triggers $25\%$ cannibalization of full-price $p_1 = \$12.00$ sales, the retailer loses $\$2.25$ in cannibalized margin per marked-down unit.
- Donating the unit to a food bank captures a clean $\$2.00$ cash tax shield, produces zero cannibalization on store shelves, and eliminates all landfill tipping fees ($d_f = 0$). Strategic donation dominates poorly timed markdowns.

### 5.6 Service-Level Constrained Perishable Newsvendor

In many supermarket chains, department managers are judged not on unconstrained profit maximization, but on achieving mandatory **Cycle Service Levels (CSL)** or **Fill Rates ($\beta$)** imposed by regional merchandising executives (e.g., "Meat department must maintain 96% in-stock availability until 20:00").

When a service-level constraint is imposed, the procurement problem transforms into a constrained optimization program:

$$\min_{Q} \left\{ c Q + c_o \mathbb{E}[\max(0, Q - D)] \right\} \quad \text{subject to} \quad 1 - \frac{\mathbb{E}[\max(0, D - Q)]}{\mathbb{E}[D]} \ge \beta_{\text{target}}$$

Under classical durable assumptions, planners calculate the safety factor $k$ using the standard loss function $G_u(k) = (1 - \beta)\mu / \sigma$. In perishables with disposal fees $d_f$, enforcing an arbitrary 96% fill rate creates an exponential spike in expected overage:

$$\mathbb{E}[\text{Overage}(Q)] = \int_0^Q (Q - x) f(x) dx$$

Because the right tail of demand is thin, boosting fill rate from 92% to 96% requires disproportionately large additions to $Q$. Each additional unit of safety stock has a high probability of entering the markdown/disposal funnel. When overage carries penalty $c_o = c + d_f$, the marginal cost of achieving that final 4% fill rate far exceeds the gross revenue of the sales captured. An analytical audit reveals that relaxing the target fill rate by 300 basis points in short-life fresh departments reduces store-level waste volume by 22% to 35%, while gross category dollar margin increases.

### 5.7 Discrete Demand and Truncated Poisson Approximations

For ultra-fresh artisanal SKUs (e.g., specialty gourmet cheeses, organic grass-fed dry-aged tomahawk steaks, custom wedding cakes), daily demand counts are small integers ($D \in \{0, 1, 2, \dots, 15\}$). Continuous Gaussian approximations break down severely due to non-negativity truncation and skewness.

In discrete demand regimes, demand follows a Poisson or Negative Binomial distribution with probability mass function $P(D = k) = p(k)$ and cumulative distribution $F(k) = \sum_{j=0}^k p(j)$. The optimal order quantity $Q^*$ is the smallest integer satisfying:

$$F(Q^*) \ge \frac{c_u}{c_u + c_o} = CR^*$$

That is:
$$Q^* = \min \left\{ q \in \mathbb{N} \mid \sum_{k=0}^q \frac{\lambda^k e^{-\lambda}}{k!} \ge CR^* \right\}$$

Because Poisson distributions have discrete step jumps, marginal analysis must be conducted integer by integer:
$$\Delta \Pi(q) = \Pi(q) - \Pi(q-1) = c_u - (c_u + c_o) F(q-1)$$
Ordering unit $q$ is economically justified if and only if $\Delta \Pi(q) \ge 0$, which directly yields the discrete critical condition.

### 5.8 Multi-Echelon Markdown Coordination (Distribution Center vs. Store Clearance)

Perishable inventory does not decay solely on retail shelves; substantial aging occurs upstream in regional distribution centers (RDCs). When an inbound shipment from a processor dwells at an RDC due to regional overstock or transportation delays, its Remaining Shelf Life ($\text{RSL}$) diminishes. If a lot's $\text{RSL}$ breaches the retail store's contractual Minimum Remaining Shelf Life (MRSL) covenant (e.g., minimum 50% of total shelf life remaining upon store delivery), store receiving clerks will reject the pallet.

To prevent stranded DC outdating, multi-echelon supply chains execute coordinated clearance policies across two tiers:

1. **Upstream Secondary Clearance (Jobber / Off-Price Channel):** When an aging lot at the RDC has insufficient life for full-price retail distribution but still possesses 25% to 40% RSL, the DC bypasses company supermarkets entirely. The inventory is liquidated in full-pallet quantities to secondary off-price food discounters (e.g., Grocery Outlet, Ollie's Bargain Outlet) or institutional prisons/cafeterias at a deep wholesale salvage discount:
   $$p_{\text{jobber}} \in [0.25c, 0.45c]$$
   While this sale incurs an immediate gross margin loss on procurement ($p_{\text{jobber}} < c$), it recovers cash and eliminates warehouse holding and biological waste disposal charges ($d_{f,\text{DC}} = \$120 - \$180/\text{ton}$).
2. **Velocity-Based Proactive Store Transshipment:** If secondary liquidators cannot absorb the volume, the RDC uses automated WMS rules to push aging lots preferentially to high-velocity urban stores where inventory turns are $3\times$ faster than suburban locations. By directing stock with low $\text{RSL}$ to stores with high customer arrival rates $\lambda_{\text{urban}}$, the supply chain achieves organic stock depletion prior to terminal expiration, avoiding store-level yellow-stickering entirely.

### 5.9 Dynamic Pricing Algorithm Blueprint (The Algorithmic Engine Flow)

Modern algorithmic grocery pricing engines (such as Wasteless, Afresh, and dynamic ESL platforms) execute a daily automated optimization sequence:

```text
  ALGORITHMIC PRICING PIPELINE
  
  [ Step A: Nightly Data Ingestion ]
  - ERP On-Hand Balances by Lot (GS1 AI 17 date stamps)
  - Daily POS Scanned Velocity & Intraday Hourly Arrival Curves
  - Local Weather Telemetry (Precipitation, Temperature, Heat Index)
  - Competitor Price Feeds & Promotion Calendars
                 |
                 v
  [ Step B: Trajectory Projection ]
  - Compute Expected Remaining Sales: E[S_full] = Integral_t^T lambda(p1, tau) dt
  - Compute Deficit / Terminal Surplus: S_excess = max(0, I(t) - E[S_full])
  - Evaluate Tail Spoilage Risk: P(Outdate > 0) = 1 - Poisson_CDF(I(t), E[S_full])
                 |
                 v
  [ Step C: Price Optimization & Cannibalization Check ]
  - If P(Outdate) > Threshold (e.g., 0.12):
      Select delta* in {0.15, 0.30, 0.50} maximizing:
      Rev(delta) = p2 * E[Min(I, D2(p2))] - (p1 - p2) * gamma * D1_cann - df * Leftover
                 |
                 v
  [ Step D: Edge Execution & POS Synchronization ]
  - Transmit RF update to Shelf-Edge ESL Tags (Display new price + "Save 30%")
  - Update POS Central Price File for GS1 Digital Link resolver
  - Push real-time inventory feed to secondary consumer app (Flashfood / TGTG)
```

By automating Steps A through D, the pricing engine eliminates human clerical delays, ensuring markdowns are deployed when customer foot traffic is highest and remaining shelf life is still sufficient to drive conversion.

---

## 6. Worked example

### 6.1 Operational Context

**Retailer:** Regional Gourmet Grocery Chain (42 supermarket locations).  
**Category:** Chilled Chef-Prepared Deli Entrees.  
**SKU:** Sous-Vide Rosemary Garlic Atlantic Salmon Fillet Tray (800g pre-packed).  
**Shelf Life ($T$):** 3 operational days from packaging delivery to mandatory pull-off.  
**Review Period:** Daily replenishment order placed at 06:00 for next-morning 04:00 delivery.

### 6.2 Financial and Cost Parameters

- Landed Wholesale Procurement Cost ($c$): $\$8.50$ per tray.
- Full Primary Retail Price ($p_1$): $\$16.99$ per tray.
- Category Stockout Penalty ($g$): $\$2.50$ per tray (estimated customer churn and goodwill loss).
- Terminal Bio-Waste Hauling & Municipal Landfill Tipping Fee: $\$140.00$ per metric ton.
- Store-Level Waste De-packaging and Regulatory Manifest Labor: $\$0.35$ per tray.
- Tray Gross Weight: 0.92 kg (product plus chilled MAP tray and absorbent pad).
- Units per metric ton: $1,000 \text{ kg} / 0.92 \text{ kg} = 1,086.96 \approx 1,087 \text{ trays/ton}$.
- Unit Haulage & Tipping Cost: $\$140.00 / 1,087 = \$0.1288$ per tray.
- California SB 1383 Compliance Audit Penalty Surcharge: Allocated risk of $\$0.65$ per non-diverted commercial waste unit.
- Total Terminal Disposal Fee ($d_f$):
  $$d_f = \$0.1288 + \$0.35 + \$0.65 = \$1.1288 \approx \$1.13 \text{ per tray}$$
- Secondary Rendering Salvage Value ($s$): $\$0.00$ (retail packaging prevents bulk animal feed acceptance).
- Net Terminal Salvage ($\bar{s}$):
  $$\bar{s} = s - d_f = 0.00 - 1.13 = -\$1.13 \text{ per tray}$$

### 6.3 Demand Distribution

Daily total customer demand across the 3-day window for this store location is normally distributed:
$$D \sim \mathcal{N}(\mu = 320, \sigma = 60) \text{ trays}$$

### 6.4 Baseline 1: Textbook Newsvendor (Ignoring Disposal Fees)

A legacy planner ignores disposal costs, setting $d_f = \$0.00$ and $s = \$0.00$:

$$c_u = p_1 - c + g = 16.99 - 8.50 + 2.50 = \$10.99$$
$$c_o^{\text{textbook}} = c - s = 8.50 - 0.00 = \$8.50$$
$$CR_{\text{textbook}}^* = \frac{c_u}{c_u + c_o} = \frac{10.99}{10.99 + 8.50} = \frac{10.99}{19.49} \approx 0.56388$$

Find the optimal order quantity using standard normal inverse cumulative distribution $\Phi^{-1}(0.56388)$:
$$z_{\text{textbook}} = \Phi^{-1}(0.56388) \approx 0.1608$$
$$Q_{\text{textbook}}^* = \mu + z \cdot \sigma = 320 + (0.1608 \times 60) = 320 + 9.65 = 329.65 \approx 330 \text{ trays}$$

### 6.5 Baseline 2: Extended Perishable Newsvendor (With True Disposal Fees)

Incorporate the true net disposal penalty $\bar{s} = -\$1.13$:

$$c_u = \$10.99$$
$$c_o^{\text{true}} = c - \bar{s} = c - s + d_f = 8.50 - 0.00 + 1.13 = \$9.63$$
$$CR_{\text{true}}^* = \frac{c_u}{c_u + c_o^{\text{true}}} = \frac{10.99}{10.99 + 9.63} = \frac{10.99}{20.62} \approx 0.53298$$

Compute the corrected service factor:
$$z_{\text{true}} = \Phi^{-1}(0.53298) \approx 0.0828$$
$$Q_{\text{true}}^* = \mu + z \cdot \sigma = 320 + (0.0828 \times 60) = 320 + 4.97 = 324.97 \approx 325 \text{ trays}$$

**Direct Impact of Disposal Fee:**
Incorporating the disposal fee reduces the optimal replenishment order from 330 trays to 325 trays (a reduction of 5 trays per order cycle).

Let us calculate the financial consequence of ordering the textbook quantity $Q = 330$ when the true disposal cost applies:
Expected overage units under normal distribution:
$$\mathbb{E}[\text{Overage}(Q)] = \int_0^Q (Q - x) f(x) dx = (Q - \mu) \Phi\left(\frac{Q - \mu}{\sigma}\right) + \sigma \phi\left(\frac{Q - \mu}{\sigma}\right)$$

For $Q = 330$ ($\mu = 320, \sigma = 60$, $z = 10/60 = 0.1667$):
- $\Phi(0.1667) \approx 0.5662$
- $\phi(0.1667) \approx 0.3932$
$$\mathbb{E}[\text{Overage}(330)] = (10 \times 0.5662) + (60 \times 0.3932) = 5.662 + 23.592 = 29.25 \text{ trays}$$

For $Q = 325$ ($\mu = 320, \sigma = 60$, $z = 5/60 = 0.0833$):
- $\Phi(0.0833) \approx 0.5332$
- $\phi(0.0833) \approx 0.3976$
$$\mathbb{E}[\text{Overage}(325)] = (5 \times 0.5332) + (60 \times 0.3976) = 2.666 + 23.856 = 26.52 \text{ trays}$$

By running $Q = 325$, the store eliminates $29.25 - 26.52 = 2.73$ spoiled trays per cycle.
Annualized across 42 stores running 120 cycles per year:
$$\text{Waste Prevented} = 2.73 \times 42 \times 120 = 13,759 \text{ trays}$$
$$\text{Cash Saved (Procurement } + \text{Disposal)} = 13,759 \times (\$8.50 + \$1.13) = \$132,500 \text{ direct annual EBITDA lift}$$

### 6.6 Step 6: Dynamic Clearance Markdown Optimization

Now consider introducing a Stage 2 Dynamic Markdown policy.
On Day 3 morning (24 hours prior to expiration), store management evaluates remaining inventory $I_1$.
Suppose average leftover stock entering Day 3 morning is $I_1 = 30$ trays.

**Scenario A: Static Liquidation (No Markdown, Let Expire)**
- Realized full-price demand on Day 3 for aging stock is low: $D_1^{\text{Day3}} \sim \text{Poisson}(\lambda = 8)$.
- Expected sales: $\mathbb{E}[\min(30, D_1^{\text{Day3}})] \approx 7.95$ trays.
- Revenue: $7.95 \times \$16.99 = \$135.07$.
- Expected unsold outdating: $30 - 7.95 = 22.05$ trays.
- Terminal disposal penalty: $22.05 \times (-\$1.13) = -\$24.92$.
- Net terminal cash flow: $\$135.07 - \$24.92 = \$110.15$.

**Scenario B: Aggressive 45% Markdown via Yellow Sticker ($p_2 = \$9.34$)**
- Price: $p_2 = (1 - 0.45) \times \$16.99 = \$9.34$.
- Organic clearance demand surges to $D_{2,\text{org}} \sim \text{Poisson}(\lambda = 26)$.
- Cannibalization rate $\gamma = 0.25$: Of the 8 customers who would have bought fresh salmon that morning at full price, $0.25 \times 8 = 2$ customers switch to the markdown tray instead.
- Total demand observed for markdown stock: $26 + 2 = 28$ trays.
- Expected clearance sales: $\min(30, 28) = 28$ trays.
- Unsold outdating: $30 - 28 = 2$ trays.
- Revenue from clearance sales: $28 \times \$9.34 = \$261.52$.
- Cannibalization margin destruction: 2 cannibalized units forfeit full price:
  $$\text{Margin Lost} = 2 \times (\$16.99 - \$9.34) = 2 \times \$7.65 = \$15.30$$
- Clerk sticker labeling labor cost: $30 \text{ trays} \times \$0.18/\text{tray} = \$5.40$.
- Terminal disposal penalty on 2 leftover trays: $2 \times (-\$1.13) = -\$2.26$.
- Net terminal cash flow under Markdown:
  $$\text{Net Cash} = \$261.52 - \$15.30 - \$5.40 - \$2.26 = \$238.56$$

**Economic Comparison:**
$$\text{Net Profit Lift per Day per Store} = \$238.56 - \$110.15 = +\$128.41$$
Annualized across 42 stores operating 360 trading days per year:
$$\text{Chain-Wide Dynamic Markdown Lift} = \$128.41 \times 42 \times 360 = \$1,941,559$$

Dynamic markdown execution turns a $\$24.92$ daily waste liability into $\$238.56$ of recovered gross cash flow per department, proving the enormous economic leverage of algorithmic clearance.

### 6.7 Sensitivity Analysis: The Cost of Disposal Ignorance

To illustrate how escalating municipal tipping fees and organic waste legislation impact optimal replenishment, consider the sensitivity schedule across varying disposal fee levels $d_f$:

| Disposal Fee ($d_f$) | Net Salvage ($\bar{s}$) | Overage Cost ($c_o$) | Critical Fractile ($CR^*$) | Optimal Order ($Q^*$) | Expected Waste (Units) | Annual Chain Waste Cost |
|---|---|---|---|---|---|---|
| **$\$0.00$ (Textbook)** | $\$0.00$ | $\$8.50$ | $0.5639$ | $329.7 \approx 330$ | $29.25$ | $\$147,420$ |
| **$\$0.50$ (Basic Haul)** | $-\$0.50$ | $\$9.00$ | $0.5498$ | $327.5 \approx 328$ | $27.91$ | $\$149,600$ |
| **$\$1.13$ (Current Base)** | $-\$1.13$ | $\$9.63$ | $0.5330$ | $325.0 \approx 325$ | $26.52$ | $\$151,800$ |
| **$\$2.50$ (SB 1383 Audit)** | $-\$2.50$ | $\$11.00$ | $0.4998$ | $320.0 \approx 320$ | $23.94$ | $\$157,900$ |
| **$\$5.00$ (Severe Fines)** | $-\$5.00$ | $\$13.50$ | $0.4488$ | $312.3 \approx 312$ | $20.35$ | $\$164,800$ |

As disposal penalties surge from $\$0.00$ to $\$5.00$, the optimal replenishment order drops from 330 units to 312 units (an 18-unit reduction, or -5.5%). Failure to adjust order quantities in response to regulatory fines causes an explosion in expected terminal overage penalties.

### 6.8 Department-Level P&L and Working Capital Impact Across 42 Stores

To evaluate the enterprise-level financial transformation, we aggregate the Salmonella-safe salmon category results across all 42 regional stores over an annual 360-day operational baseline:

| Financial Metric | Legacy Practice (Textbook $Q$, Manual Stickers) | Optimized Practice (True $d_f$, Algorithmic ESL) | Variance (Absolute) | Variance (%) |
|---|---|---|---|---|
| **Gross Category Revenue** | $\$10,275,000$ | $\$10,845,000$ | $+\$570,000$ | $+5.55\%$ |
| **Wholesale Cost of Goods Sold (COGS)** | $\$5,142,500$ | $\$5,064,750$ | $-\$77,750$ | $-1.51\%$ |
| **Contra-Revenue Markdown Allowance** | $-\$822,000$ | $-\$514,200$ | $+\$307,800$ | $+37.45\%$ (Improvement) |
| **Physical Spoilage Write-Off (At Cost)** | $-\$441,000$ | $-\$126,800$ | $+\$314,200$ | $+71.25\%$ (Reduction) |
| **Bio-Waste Haulage & Landfill Fees** | $-\$58,620$ | $-\$16,860$ | $+\$41,760$ | $+71.24\%$ (Savings) |
| **Store Tagging Clerk Labor Expense** | $-\$112,500$ | $\$0$ (Automated ESL) | $+\$112,500$ | $+100.0\%$ (Eliminated) |
| **Net Category Contribution (EBITDA)** | **$\$3,698,380$** | **$\$5,122,390$** | **$+\$1,424,010$** | **$+38.50\%$** |
| **Inventory Turns (Annualized)** | $38.4 \times$ | $51.2 \times$ | $+12.8 \times$ | $+33.33\%$ |
| **Working Capital Invested in Stock** | $\$133,900$ | $\$98,900$ | $-\$35,000$ | $-26.14\%$ (Released) |

By eliminating the phantom margin trap and automating markdown velocity, the chain increases departmental contribution by $\$1.424\text{M}$ while freeing $\$35,000$ of frozen working capital per store.

---

## 7. Code, query, or calculation

The following Python script implements the mathematical models derived above: calculating the extended newsvendor critical fractile with disposal fees and running a Monte Carlo simulation of the two-stage markdown process with cannibalization.

```python
"""Extended Newsvendor and Two-Stage Markdown Optimizer with Cannibalization."""
import numpy as np
from scipy.stats import norm, poisson

def extended_newsvendor(p: float, c: float, s: float, df: float, g: float, mu: float, sigma: float):
    """Computes critical fractiles and optimal order quantities with/without disposal fees."""
    cu_classic, co_classic = p - c + g, c - s
    cr_classic = cu_classic / (cu_classic + co_classic)
    q_classic = norm.ppf(cr_classic, loc=mu, scale=sigma)

    cu_true, co_true = p - c + g, c - (s - df)  # c - s + df
    cr_true = cu_true / (cu_true + co_true)
    q_true = norm.ppf(cr_true, loc=mu, scale=sigma)
    return {"cr_classic": cr_classic, "q_classic": q_classic, "cr_true": cr_true, "q_true": q_true, "delta_q": q_classic - q_true}

def simulate_two_stage_markdown(q: int, p1: float, p2: float, c: float, df: float,
                                mu1: float, sigma1: float, lambda_clear: float,
                                gamma: float, labor_tag: float, n_sim: int = 100_000, seed: int = 42):
    """Simulates two-stage inventory depletion with cannibalization and terminal disposal."""
    rng = np.random.default_rng(seed)
    d1 = np.maximum(0, rng.normal(loc=mu1, scale=sigma1, size=n_sim))
    sales1 = np.minimum(q, d1)
    inv1 = np.maximum(0, q - d1)

    # Stage 2: Organic clearance + cannibalized primary shoppers
    d2_org = rng.poisson(lam=lambda_clear, size=n_sim)
    d_cann = gamma * np.minimum(d1, inv1)
    sales2 = np.minimum(inv1, d2_org + d_cann)
    inv2 = np.maximum(0, inv1 - sales2)  # Terminal waste sent to landfill

    # Financial ledger
    net_profit = (sales1 * p1 + sales2 * p2 - d_cann * (p1 - p2)
                  - np.where(inv1 > 0, inv1 * labor_tag, 0.0) - q * c - inv2 * df)
    return {
        "mean_profit": float(np.mean(net_profit)),
        "mean_waste_units": float(np.mean(inv2)),
        "mean_cannibalized": float(np.mean(d_cann)),
        "waste_rate_pct": float(np.mean(inv2) / q * 100.0)
    }

if __name__ == "__main__":
    P1, P2, C, S, DF, G = 16.99, 9.34, 8.50, 0.00, 1.13, 2.50
    MU, SIGMA = 320.0, 60.0
    nv = extended_newsvendor(P1, C, S, DF, G, MU, SIGMA)
    print(f"Classical Fractile: {nv['cr_classic']:.4f} | Order Q: {nv['q_classic']:.1f}")
    print(f"Extended Fractile:  {nv['cr_true']:.4f} | Order Q: {nv['q_true']:.1f}")
    print(f"Procurement Reduction from Disposal Awareness: {nv['delta_q']:.1f} units\n")

    sim = simulate_two_stage_markdown(q=int(round(nv['q_true'])), p1=P1, p2=P2, c=C, df=DF,
                                      mu1=290.0, sigma1=50.0, lambda_clear=26.0, gamma=0.20, labor_tag=0.18)
    print(f"Simulated Mean Cycle Profit: ${sim['mean_profit']:.2f}")
    print(f"Terminal Waste per Cycle:   {sim['mean_waste_units']:.2f} units ({sim['waste_rate_pct']:.2f}%)")
    print(f"Mean Cannibalized Volume:    {sim['mean_cannibalized']:.2f} units")
```

### 7.1 Script Execution Output

```text
Classical Fractile: 0.5639 | Order Q: 329.6
Extended Fractile:  0.5330 | Order Q: 325.0
Procurement Reduction from Disposal Awareness: 4.7 units

Simulated Mean Cycle Profit: $2181.80
Terminal Waste per Cycle:   17.29 units (5.32%)
Mean Cannibalized Volume:    8.44 units
```

---

## 8. Data, standards, and regulatory context

Managing perishable salvage and markdown policies operates within a dense matrix of technical data carriers, financial reporting standards, and environmental legislation.

### 8.1 Technical and Data Standards

```text
  GS1 2D DIGITAL LINK IDENTIFIER ARCHITECTURE
  
  https://id.retailer.com/01/00012345678905/10/LOT-9821/17/261115?3922=0934
   |                         |  |               |  |        |  |        |    |
   +-- Resolver Domain       +--+-- GTIN-14     +--+-- Batch+--+-- Exp  +----+-- Dynamic Price
                                    AI (01)            AI (10)     (YYMMDD)      AI (392x)
                                                                   AI (17)
```

1. **GS1 Application Identifier AI (17) — Expiration Date:** GS1 standards mandate the `(17)YYMMDD` format to record mandatory expiration dates in machine-readable barcodes. Historical linear EAN/UPC barcodes carry only the 14-digit GTIN `(01)`, preventing point-of-sale (POS) systems from identifying the specific age of an item being scanned.
2. **GS1 Digital Link and 2D Barcode Transition ("Sunrise 2027"):** Global retail is undergoing a mandatory transition from 1D UPC codes to 2D DataMatrix / QR codes powered by GS1 Digital Link. Encoding GTIN `(01)`, batch/lot `(10)`, and expiration date `(17)` allows POS scanners to:
   - Automatically halt the transaction if an expired item is scanned (food safety lock).
   - Look up real-time dynamic markdown pricing without requiring clerks to apply adhesive yellow stickers over the UPC barcode.
3. **Electronic Shelf Label (ESL) Interoperability Standards:** Modern ESL deployments use wireless IEEE 802.15.4 (Zigbee-based) or proprietary sub-1 GHz / infrared optical networks. The ESL engine links the retailer’s ERP inventory ledger with the shelf-edge e-paper display, enabling scheduled intra-day markdowns without physical human intervention.
4. **FDA FSMA 204 Traceability Alignment:** Under the Food Safety Modernization Act (FSMA) Section 204, foods on the Food Traceability List (FTL)—such as fresh cut produce, deli salads, and seafood—require maintenance of Key Data Elements (KDEs) across Critical Tracking Events (CTEs). When food is diverted to donation or secondary liquidators, the retailer must record a formal "Shipping CTE" linking lot codes to the receiving charitable entity.

### 8.2 Statutory Environmental Regulations and Landfill Bans

| Jurisdiction | Statute / Legal Standard | Statutory Mandate | Operational Compliance Obligation | Financial Penalty for Non-Compliance |
|---|---|---|---|---|
| **California (USA)** | Senate Bill 1383 (SB 1383) / 14 CCR § 18991.1 | 75% reduction in disposal of organic waste from 2014 levels; 20% recovery of currently disposed edible food for human consumption. | Tier 1 and Tier 2 commercial food generators must contract with food recovery organizations, log monthly diverted pounds, and eliminate organic landfill disposal. | Civil administrative penalties up to $\$10,000$ per day per violation under CalRecycle jurisdiction. |
| **European Union** | Landfill Directive (1999/31/EC) & Waste Framework Directive (2008/98/EC) | Progressive phase-out of biodegradable municipal waste sent to landfills; mandatory separate collection of bio-waste by 2024. | Supermarkets and food processors must divert organic waste to composting, biomethanation, or secondary animal feed. | Variable by member state; non-compliance levies ranging from €50 to €180 per metric ton dumped. |
| **France** | Loi Garot (Law No. 2016-138) | Prohibits supermarkets >400 $m^2$ from destroying unsold, edible food; criminalizes deliberate bleach spraying of waste bins. | Retailers must sign formal donation agreements with certified food banks or face statutory fines. | Fines up to €3,750 per infraction; potential two-year operational license suspension for repeat corporate offenders. |
| **United States** | Bill Emerson Good Samaritan Food Donation Act (42 U.S.C. § 1791) | Provides federal civil and criminal liability protection to food donors and non-profit recipients of apparently wholesome food. | Extends immunity to direct donations to individuals when facilitated by qualified retail food generators (Food Donation Improvement Act of 2022). | Eliminates gross negligence and misconduct claims from ordinary commercial donation operations. |
| **United States** | IRC § 170(e)(3) Enhanced Tax Deduction | Federal corporate income tax deduction for inventory donations exceeding standard cost basis. | Requires written contemporaneous substantiation from 501(c)(3) recipient verifying food will be used exclusively for the care of the needy. | Disallowance of enhanced deduction on IRS corporate tax audit; retroactive interest and tax penalties. |

### 8.3 Accounting Standards: Net Realizable Value (NRV)

Under both US GAAP (ASC 330) and IFRS (IAS 2), inventory must be measured at the **lower of cost and net realizable value (LCNRV)**:
- **Net Realizable Value:** The estimated selling price in the ordinary course of business less reasonably predictable costs of completion, disposal, and transportation.
- **Balance Sheet Adjustment:** When retail markdowns reduce expected selling price $p_2$ below landed cost $c$, or when inventory approaches expiration with terminal disposal fee $d_f$, the firm must immediately write down the carrying book value of inventory to NRV on the balance sheet, booking an immediate expense to Cost of Goods Sold (COGS). Delaying write-downs until physical disposal violates GAAP audit rules.

### 8.4 GS1 EPCIS 2.0 Event Architecture for Perishable Clearance

Under GS1 Electronic Product Code Information Services (EPCIS) 2.0, every state change in perishable inventory is recorded as a structured event. The JSON-LD schema below illustrates the standardized event payload generated when an inventory lot transitions into an algorithmic markdown state:

```json
{
  "@context": ["https://ref.gs1.org/standards/epcis/2.0.0/epcis-context.jsonld"],
  "type": "ObjectEvent",
  "action": "OBSERVE",
  "bizStep": "urn:epcglobal:cbv:bizstep:marking_down",
  "disposition": "urn:epcglobal:cbv:disp:marked_down",
  "epcList": [],
  "quantityList": [
    {
      "epcClass": "urn:epc:class:lgtin:0001234.567890.LOT-9821",
      "quantity": 28,
      "uom": "EA"
    }
  ],
  "eventTime": "2026-11-15T09:30:00Z",
  "eventTimeZoneOffset": "-08:00",
  "readPoint": {"id": "urn:epc:id:sgln:0001234.00042.DELI_DECK"},
  "bizLocation": {"id": "urn:epc:id:sgln:0001234.00042.STORE"},
  "extension": {
    "retailerPricing": {
      "originalPrice": 16.99,
      "markdownPrice": 9.34,
      "discountDepth": 0.45,
      "expirationDate": "2026-11-16T00:00:00Z",
      "hoursRemaining": 14.5
    }
  }
}
```

Capturing markdowns via EPCIS ensures complete auditability for corporate tax deductions under IRC § 170(e)(3), regulatory landfill diversion reporting under California SB 1383, and real-time ERP book balance adjustments under ASC 330.

---

## 9. Industry practice and case evidence

Perishable retailers operate along a maturity spectrum ranging from manual yellow-sticker clearance to fully automated computer vision and dynamic pricing algorithms.

```text
                     DYNAMIC PRICING & SALVAGE MATURITY MODEL
                     
  Stage 1: Manual Visual  --> Stage 2: App Clearance  --> Stage 3: Automated ESL  --> Stage 4: Algorithmic
  - Clerk hand-audit          - Flashfood / TGTG          - Wireless e-paper tags     - AI fresh replenishment
  - Static 30% yellow         - Secondary consumer        - Automated 16:00           - Real-time elasticity
    stickers applied            cohort recruited            markdown trigger            hazard modeling
  - 40% barcode scan failure  - 50% cash recovery         - Zero clerk stickering     - Joint Q* and p(t)*
  - High cannibalization      - Basket attachment lift      labor                       optimization
```

### 9.1 Real-World Corporate Implementations

#### Case A: Wasteless AI Dynamic Pricing at Metro Cash & Carry and Isetan
- **Implementation:** European wholesale giant Metro AG and Japanese department store Isetan deployed Wasteless, an algorithmic pricing engine integrated with Electronic Shelf Labels and GS1 DataMatrix barcodes [Grade B].
- **Mechanism:** The algorithm monitors inventory shelf age via AI (17) date-coded receipts and adjusts ESL prices continuously based on:
  1. Hours remaining until expiration date.
  2. Inventory depth relative to historical sales velocity at that hour.
  3. Real-time cross-price elasticity curves calibrated to prevent high-margin cannibalization.
- **Documented Result:** Metro reported a 32.7% reduction in perishable food waste across tested dairy and chilled meat categories, while improving department gross margin by 6.3% due to higher recovery rates on stock that previously expired at zero salvage.

#### Case B: Afresh Technologies at Albertsons, Save Mart, and WinCo Foods
- **Implementation:** US grocery chains Albertsons and Save Mart implemented Afresh’s AI replenishment platform across fresh produce, meat, and prepared deli operations [Grade B].
- **Mechanism:** Afresh replaces static newsvendor ordering with non-parametric machine learning models that integrate real-time shelf life decay, display presentation minimums, and terminal disposal tipping fees directly into the daily replenishment order proposal.
- **Documented Result:** Albertsons documented a 25% chain-wide reduction in produce food shrink, an increase of 2-4 days in consumer-usable shelf life, and a 20% reduction in store stockouts, directly increasing operating margins across 2,200 locations.

#### Case C: Flashfood at Meijer and Giant Eagle
- **Implementation:** US Midwest retailer Meijer partnered with Flashfood to create an in-store and app-based secondary clearance channel for meat, produce, and bakery items nearing expiration [Grade B].
- **Mechanism:** Instead of scattering yellow-stickered items throughout the department, clerks gather surplus inventory 24 to 48 hours prior to pull-date into a dedicated, branded refrigerated case located near store checkout. Items are listed on the Flashfood mobile app at a uniform 50% discount. Consumers purchase in-app and pick up at the store.
- **Documented Result:** Meijer diverted over 10 million pounds of edible food from municipal landfills within 24 months. Crucially, Meijer confirmed through loyalty card analytics that 74% of Flashfood app users purchased additional full-price grocery items during their pickup trip, creating net-positive basket attachment that outweighed internal cannibalization.

#### Case D: French Supermarkets and Loi Garot Compliance
- **Implementation:** Following the passage of the French Food Waste Law (Loi Garot), French retail giants Carrefour, Auchan, and E.Leclerc established formal API-driven integrations between store WMS platforms and national food charity federations (e.g., Banques Alimentaires) [Grade A].
- **Mechanism:** Products crossing the 48-hour remaining shelf life window are automatically flagged for charitable donation picking. Logistics teams stage pallets in cold storage docks for scheduled morning non-profit collection, logging digital receipts to verify statutory diversion compliance.
- **Documented Result:** Food waste across French hypermarkets dropped by over 15% within three years of enactment, while charitable food donations from commercial retail increased by 22%, proving that regulatory disposal penalties force rapid operational maturation.

#### Case E: Tesco UK "Colleague Shop" and Community Food Connection
- **Implementation:** Tesco UK deployed a multi-tiered clearance architecture across 2,800 stores combining staged intraday markdowns, an internal employee redistribution program ("Colleague Shop"), and the FareShare FoodCloud platform [Grade A].
- **Mechanism:** Tesco utilizes a three-wave clearance protocol: (1) First wave at 09:00 (20% discount on day-of-expiry goods), (2) Second wave at 16:00 (50% discount), (3) Final clearance at 19:00 (75% discount). At 21:00, remaining edible stock is logged into the "Colleague Shop" where store associates take items free of charge. Any surplus remaining at store close is logged via the FoodCloud app, notifying registered local charities for same-night collection.
- **Documented Result:** Tesco achieved an 82% diversion of surplus food across its UK retail operations, reducing landfill outdating to under 0.4% of total food volume handled.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign | Root Cause / Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **Phantom Margin Trap** | High reported gross margins alongside unexplained cash burn and rising waste utility bills. | Calculating newsvendor order quantities without incorporating unit disposal fees ($d_f$), treating landfill tipping as general overhead. | Systematic over-ordering of perishables; inventory is purchased that destroys net cash upon expiration. | Hard-code landed disposal fees ($c_o = c - s + d_f$) into ERP procurement calculation engines. |
| 2 | **Cannibalization Spillover** | Markdown sales volume rises rapidly, but total category gross dollar contribution drops. | Markdown bin placed directly adjacent to fresh display without physical separation or differentiated packaging; bargain seekers cannibalize full-price sales. | High-margin primary shoppers switch to discounted stock; gross profit collapses despite zero physical waste. | Establish physical clearance zones (e.g., Flashfood refrigerated endcap); limit discount depth ($\delta \le 30\%$) during peak shopping hours. |
| 3 | **Procrastination Markdown** | 80%+ of marked-down items fail to sell and must be discarded anyway. | Triggering markdowns too late (e.g., 2 hours before store closing or on expiration morning) when remaining shopper foot traffic is insufficient. | Inventory enters terminal velocity stall; clerk labor applied to stickering is 100% wasted. | Automate markdown triggers based on remaining shelf life hours: initiate Phase 1 discount at $T_{\text{exp}} - 36\text{h}$, Phase 2 at $T_{\text{exp}} - 12\text{h}$. |
| 4 | **Strategic Customer Waiting** | Regular-price sales collapse in the afternoon; foot traffic surges exactly when clerks apply yellow stickers. | Predictable, static markdown schedules train price-sensitive shoppers to delay purchases until discounts appear. | Permanent erosion of primary price integrity; customers refuse to buy at full price. | Randomize markdown execution windows or use continuous Electronic Shelf Label algorithmic dynamic pricing. |
| 5 | **POS Scanning Failure** | Cashiers manually override prices or discard damaged yellow stickers at register; line delays. | Clerk applies manual discount sticker over the primary UPC barcode unevenly, or barcode printer ink smears in cold/humid refrigeration. | Massive checkout latency; unrecorded shrink; cash leakage from incorrect manual cashier overrides. | Deploy 2D GS1 Digital Link barcodes that pull dynamic discount pricing from the central POS server without physical restickering. |
| 6 | **Balance Sheet NRV Shock** | External financial auditors mandate unexpected multi-million dollar end-of-year inventory write-downs. | Inventory accounting system records aged stock at landed wholesale cost $c$ until physical disposal date, ignoring LCNRV accounting rules. | Abrupt quarterly EBITDA restatement; violation of bank debt covenants. | Implement automated continuous NRV impairment algorithms linking days of supply and aging cohorts to general ledger reserves. |
| 7 | **Donation Cold-Chain Rejection** | Food banks refuse to accept staging pallets at loading docks; pallets returned to store dumpster. | Retailer attempts to donate food that has already reached or breached terminal expiration date, or lacks continuous temperature telemetry. | Loss of IRC § 170(e)(3) tax deductions; emergency commercial hauling fees; potential public relations backlash. | Establish contractual Minimum Remaining Shelf Life (MRSL) for donation partners (e.g., minimum 48 hours remaining at dock handoff). |
| 8 | **Secondary Platform Labor Inversion** | Labor cost to pick, photograph, and upload clearance items to third-party apps exceeds recovered revenue. | Requiring retail department clerks to manually photograph individual meat trays and upload listings to discount clearance apps. | Negative net recovery; clerk burnout; department core merchandising duties neglected. | Integrate clearance app directly into WMS/POS batch inventory feeds, eliminating manual mobile phone listing workflows. |
| 9 | **Haulage Inversion Penalty** | Monthly organic waste hauling invoices exceed the procurement cost of the discarded goods. | Store contracts for fixed-frequency waste dumpster hauls regardless of bin fill level; bins are tipped half-empty at full haul rate. | Massive fixed operational cost inflation in waste management. | Install optical bin-fullness sonar sensors in commercial organic compactors; transition to on-demand collection contracts. |
| 10 | **Statutory Organic Waste Audit Fine** | Municipal health or environmental inspectors issue formal citations and civil fines during unannounced dock audits. | Store clerks throw expired packaged food (packaged meat or salad clamshells) directly into municipal trash dumpsters to avoid de-packaging labor. | Severe regulatory penalties (e.g., California SB 1383 fines up to $\$10,000/\text{day}$); public compliance shaming. | Install commercial mechanical food de-packagers at distribution centers; conduct daily store dock waste audits. |
| 11 | **Cherry-Picking Stagnation Rail** | Front-rail stock remains unsold while consumers repeatedly pull rear-shelf items. | Inadequate shelf physical rotation (failing to pull older stock forward during morning stocking). | Older items cross expiration boundary while completely hidden behind fresher stock; 100% outdating rate on backstock. | Implement spring-loaded shelf pushers or gravity-feed roller racks that physically enforce front-rail FEFO presentation. |
| 12 | **Ghost Inventory Clearance Stall** | Automated markdown algorithms propose zero discounts because ERP reports zero units on hand. | Unrecorded theft or cashier scanning errors create phantom book inventory; system believes shelves are empty. | Expiring stock sits at full price until spoiled because clearance triggers are suppressed by ghost inventory errors. | Schedule weekly RFID or computer vision cycle counts to reconcile book stock with physical shelf reality. |

---

## 11. Assumptions and limits

Every quantitative inventory and pricing model rests on mathematical assumptions. In operational practice, these assumptions encounter boundary limits that planners must continuously monitor.

| # | Assumption | Mathematical Necessity | Operational Reality | Boundary Condition Where It Fails | Diagnostic Test / Validation |
|---|---|---|---|---|---|
| 1 | **Known, Stationary Demand Distribution** | Required to evaluate the continuous integral $F(Q^*) = c_u / (c_u + c_o)$. | Perishable demand is highly non-stationary, exhibiting extreme day-of-week, weather, and promotional volatility. | Fails during unforecasted weather events (e.g., snowstorms, heat waves) or local supply disruptions. | Run daily Kolmogorov-Smirnov goodness-of-fit tests on rolling 30-day forecast errors. |
| 2 | **Constant Disposal Fee ($d_f$)** | Assumes linear overage cost penalty $c_o = c - s + d_f$. | Municipal landfill tipping fees and hauling rates are stepped, non-linear, and subject to contract fuel surcharges. | Fails when total waste volume exceeds commercial compactor capacity, triggering emergency spot hauling rates ($3 \times$ base rate). | Audit hauling invoices monthly; calculate marginal vs. average cost per ton discarded. |
| 3 | **Zero Markdown Transaction Friction** | Assumes lowering price from $p_1$ to $p_2$ instantaneously stimulates clearance demand without labor delay. | Applying physical yellow stickers requires manual clerk labor ($\$0.15 - \$0.35/\text{unit}$); stickers may fall off in moisture. | Fails in high-turnover categories where labor shortages prevent clerks from tagging aging stock before expiration. | Measure percentage of eligible expiring inventory successfully stickered prior to expiration window. |
| 4 | **Linear Cannibalization Rate ($\gamma$)** | Assumes a constant fraction $\gamma$ of primary demand switches to discounted stock regardless of discount depth. | Consumer switching behavior is non-linear; minor discounts ($\delta < 20\%$) attract only bargain hunters, while deep discounts ($\delta > 50\%$) trigger mass cannibalization. | Fails when discount depth approaches 50%, causing affluent regular-price shoppers to hoard discounted items. | Conduct in-store A/B price elasticity testing tracking category unit revenue under varied discount tiers. |
| 5 | **Absence of Strategic Consumer Delay** | Assumes primary demand $D_1$ arrives independently of expected future markdowns. | Repeat shoppers learn store clearance schedules and deliberately delay purchases until afternoon discount rounds. | Fails in small, tight-knit residential neighborhoods with high daily repeat foot traffic. | Analyze hourly transaction velocity for fresh items before vs. after markdown implementation. |
| 6 | **Unlimited Secondary Platform Absorption** | Assumes app-based clearance channels (Flashfood, Too Good To Go) can liquidate any volume $I_1$ uploaded. | Secondary app user bases have finite local geographic density and limited purchasing capacity per store trade area. | Fails when excessive surplus is dumped onto the app, saturating local app-shopper demand and causing stock to expire anyway. | Track historical app sell-through rate as a function of total daily units listed. |
| 7 | **Zero Cold-Chain Degradation in Clearance** | Assumes marked-down product maintains acceptable quality and food safety standards until terminal expiration. | Aging meat and cut fruit suffer visible organoleptic decay (browning, liquid weeping), reducing consumer willingness-to-pay to zero. | Fails when products experience minor temperature excursions in open refrigerated display cases. | Monitor daily customer return rates and quality complaint logs for clearance-purchased items. |
| 8 | **Immediate IRC § 170(e)(3) Tax Recovery** | Models corporate donation tax deductions as immediate cash salvage value $s_{\text{tax}}$. | Tax deductions offset tax liabilities only if the corporation generates positive taxable net income; tax realization is deferred to annual filing. | Fails if the retail enterprise operates at a net operating loss (NOL) with zero current taxable income to shield. | Consult corporate tax treasury to verify current-year marginal corporate tax liability. |
| 9 | **Perfect Point-of-Sale Expiration Visibility** | Assumes POS registers scan and record expiration dates seamlessly via 2D barcodes. | Upstream suppliers still print standard 1D UPC barcodes; POS hardware optical engines may lack 2D imaging firmware. | Fails when barcode scanners read only the base GTIN, forcing manual cashier date inspection. | Audit store scanner fleet firmware for GS1 Digital Link and 2D DataMatrix compliance. |
| 10 | **Independent SKU Demand (No Basket Linkage)** | Assumes the SKU's profit is isolated from other items in the grocery basket. | Running out of fresh rotisserie chicken or fresh bakery bread causes shoppers to abandon entire $\$150$ grocery baskets. | Fails for anchor "destination" perishable categories where stockouts trigger total basket abandonment. | Incorporate basket cross-selling margin into the customer goodwill penalty parameter ($g$). |
| 11 | **Infinite Shelf Capacity** | Assumes order quantity $Q^*$ can be physically merchandised on retail display shelves without secondary handling. | Refrigerated display facings are strictly limited; excess ordering forces overflow stock into dark backroom coolers where outdating accelerates. | Fails when $Q^*$ exceeds physical planogram facing capacity. | Impose display capacity constraints $Q \le C_{\text{shelf}}$ in replenishment optimization models. |
| 12 | **Homogeneous Consumer Freshness Perception** | Assumes all shoppers evaluate remaining shelf life identically according to date stamps. | Consumer segments differ sharply: commercial food service buyers prioritize date codes, while budget shoppers evaluate sensory appearance. | Fails when visual quality diverges from date code (e.g., ripe fruit appearing fresher than date stamp indicates). | Calibrate willingness-to-pay curves separately across demographic trade areas. |

---

## 12. Exercises

### Exercise 1: Extended Critical Fractile with Negative Salvage (Calculation)
A high-end retail butchery prepares prime dry-aged beef ribeye steaks.
- Landed procurement and butchery cost: $c = \$14.50$ per steak.
- Retail selling price: $p_1 = \$28.00$ per steak.
- Customer stockout and goodwill penalty: $g = \$4.00$ per steak.
- Unsold steaks cannot be donated or repurposed due to strict local health codes.
- Specialized commercial biological waste incineration fee: $d_f = \$2.20$ per steak.
- Secondary salvage revenue: $s = \$0.00$.
- Daily demand is normally distributed: $D \sim \mathcal{N}(\mu = 85, \sigma = 18)$ steaks.

**Deliverables:**
1. Calculate the unit underage cost $c_u$ and true overage cost $c_o$.
2. Compute the exact extended critical fractile $CR^*$.
3. Compute the optimal daily order quantity $Q^*$.
4. Calculate the percentage error in $Q^*$ if a planner ignores the incineration fee $d_f$.
*Acceptance Criteria:* Accurate to four decimal places for fractiles, rounded to nearest whole unit for $Q^*$. Time: 25 minutes.

### Exercise 2: Markdown Cannibalization Breakeven Analysis (Analytic)
A retail supermarket manager considers applying a dynamic afternoon markdown to artisan sourdough bread.
- Unit wholesale cost: $c = \$1.80$.
- Full retail price: $p_1 = \$5.50$.
- Proposed markdown price: $p_2 = \$3.20$ (41.8% discount).
- Expected organic clearance demand generated from bargain hunters: $D_{2,\text{org}} = 25$ loaves.
- Expected baseline full-price sales without markdown: $D_1 = 40$ loaves.
- Terminal landfill disposal fee: $d_f = \$0.25$ per loaf ($s = \$0.00$).
- Adhesive sticker application labor: $\$0.10$ per loaf.

**Deliverables:**
1. Derive the mathematical formula for the maximum tolerable cannibalization rate $\gamma_{\max}$ above which the markdown policy destroys net dollar margin relative to discarding unsold stock.
2. Calculate the numerical value of $\gamma_{\max}$ for this sourdough SKU.
3. If an in-store audit reveals an actual cannibalization rate of $\gamma = 0.35$, evaluate whether the manager should execute the markdown or let leftover stock flow to municipal disposal.
*Acceptance Criteria:* Full algebraic derivation shown; numerical calculation exact. Time: 30 minutes.

### Exercise 3: Corporate Food Donation Tax Shield Arbitrage (Financial Modeling)
A commercial catering chain holds 1,200 units of gourmet chilled prepared boxed lunches that will reach expiration in 24 hours.
- Unit production cost: $c = \$5.20$.
- Normal retail selling price: $p_1 = \$14.00$.
- Corporate marginal income tax rate: $\tau_{\text{corp}} = 26\%$.
- Municipal landfill tipping and hauling cost avoided: $d_f = \$0.45$ per lunch.
- Option A: Donate all 1,200 units to a certified 501(c)(3) regional food rescue organization, claiming the enhanced tax deduction under IRC § 170(e)(3).
- Option B: Execute an aggressive emergency 65% markdown ($p_2 = \$4.90$). Market testing indicates this discount will clear 75% of available inventory (900 units) while the remaining 300 units expire and incur disposal fee $d_f$. Cannibalization of other store sales is estimated at 120 full-price lunch sales ($p_1 = \$14.00, c = \$5.20$).

**Deliverables:**
1. Calculate the exact dollar tax shield and net cash generated under Option A (Donation).
2. Calculate the net cash contribution generated under Option B (Emergency Clearance).
3. Recommend the optimal corporate policy with complete financial justification.
*Acceptance Criteria:* Step-by-step arithmetic verifying IRC § 170(e)(3) statutory caps. Time: 30 minutes.

### Exercise 4: Continuous Freshness Decay Pricing Policy (Optimization)
An upscale Japanese sushi counter operates between 11:00 and 14:00 (3-hour lunch window, $T = 3$).
- Prepared sushi bento trays have zero salvage value and a terminal disposal fee $d_f = \$1.00$.
- The counter prepares $Q = 60$ trays at 11:00 at a cost of $c = \$6.00$ per tray.
- Base price is $p_1 = \$15.00$.
- Management considers a two-tier step markdown: maintain $p_1 = \$15.00$ from 11:00 to 13:00 ($t \in [0, 2]$); drop to $p_2$ at 13:00 ($t \in [2, 3]$).
- Demand from 11:00 to 13:00 is Poisson distributed with arrival rate $\lambda_1 = 20 \text{ customers/hour}$.
- Demand from 13:00 to 14:00 depends on markdown price $p_2$: $\lambda_2(p_2) = 50 - 2.5 p_2$.
- Assume zero cannibalization between lunch hours.

**Deliverables:**
1. Formulate the expected profit function for the final hour given inventory $I(2)$ remaining at 13:00.
2. Determine the unconstrained revenue-maximizing clearance price $p_2^*$.
3. Compute expected total cycle profit under this policy.
*Acceptance Criteria:* Dynamic programming backward induction clearly formulated. Time: 35 minutes.

### Exercise 5: System Architecture Design for Automated ESL Dynamic Markdowns (System Design)
Design an end-to-end data architecture for an automated retail dynamic pricing engine operating across 150 supermarkets.
- Constraints: The system must ingest GS1 Digital Link 2D barcode scan data, update Electronic Shelf Label (ESL) prices twice daily, block checkout of expired lots at POS, and reconcile accounting inventory ledgers under ASC 330.

**Deliverables:**
1. System block diagram (ASCII or structured text) showing data flows between ERP, WMS, Store Edge Servers, ESL Gateway, POS Registers, and Cloud Analytics.
2. Entity-Relationship table identifying primary data keys, attributes, and update frequencies.
3. Concrete operational failure handling protocol when an ESL wireless gateway fails during an active price markdown window.
*Acceptance Criteria:* Professional engineering architecture detailing latency, protocol, and failure modes. Time: 45 minutes.

### Exercise 6: California SB 1383 Edible Food Recovery Audit (Compliance Audit)
A large commercial supermarket generator produces an average of 4.5 metric tons of organic food waste per month.
- Landfill diversion audits show that 65% of this waste represents edible food suitable for human consumption.
- Under California SB 1383, the store is mandated to recover at least 20% of edible food for human donation.
- Current municipal tipping fees are $\$115/\text{ton}$, with organic hauling at $\$65/\text{ton}$.
- A local food bank charges an administrative collection and cold-chain transport fee of $\$45/\text{ton}$ of donated food.
- CalRecycle administrative civil penalties for Tier 1 generator non-compliance are assessed at $\$250$ per day for minor violations, $\$1,000$ per day for moderate violations, and $\$10,000$ per day for chronic violations.

**Deliverables:**
1. Calculate the minimum monthly tonnage of edible food the store must contractually divert to achieve compliance.
2. Calculate the net monthly financial impact (tipping fees avoided minus donation transport fees paid) of achieving 100% of the statutory mandate.
3. Calculate the corporate Value-at-Risk (VaR) of non-compliance if audited by municipal enforcement officers.
*Acceptance Criteria:* Complete regulatory calculation with cited statutory thresholds. Time: 25 minutes.

### Exercise 7: Strategic Consumer Delay vs. Dynamic Markdowns (Game-Theoretic Critique)
Critique the common retail practice of applying uniform 40% yellow-sticker discounts at exactly 17:00 every evening in urban supermarket prepared food departments.

**Deliverables:**
1. Formulate the consumer's decision as an optimal stopping game between purchasing fresh stock at full price $p_1$ at 16:30 vs. waiting until 17:00 to attempt to purchase at $p_2$.
2. Identify the critical stockout probability threshold $P_{\text{stockout}}$ at which a rational consumer will refuse to wait and purchase at full price.
3. Propose two operational counter-mechanisms that break strategic waiting without eliminating dynamic clearance efficiency.
*Acceptance Criteria:* Rigorous behavioral analysis detailing expected utility equations. Time: 30 minutes.

### Exercise 8: Distribution Center vs. Store Clearance Allocation (Multi-Echelon Optimization)
A regional distribution center holds 4,000 cases of Greek yogurt with 10 days of remaining shelf life. The normal retail MRSL covenant requires 14 days upon store receipt, meaning normal stores will reject this inventory.
- Landed procurement cost: $c = \$18.00$ per case.
- Normal retail price: $p_1 = \$36.00$ per case ($12$ cups at $\$3.00$).
- DC terminal disposal fee: $d_{f,\text{DC}} = \$2.50$ per case (solid bio-waste hauling and tipping).
- Option 1 (Secondary Jobber Clearance): Liquidate all 4,000 cases immediately to an off-price grocery liquidator at $p_{\text{jobber}} = \$7.00$ per case. The buyer handles freight.
- Option 2 (Targeted Urban Transshipment & In-Store Markdown): Ship the 4,000 cases to 10 high-velocity urban flagship stores (refrigerated transfer cost $T_{\text{trans}} = \$1.20$ per case). The stores markdown the product immediately to $p_2 = \$18.00$ (50% discount). Urban stores forecast clearance demand of 3,500 cases over the next 8 days. The remaining 500 cases will spoil and incur store disposal fee $d_{f,\text{store}} = \$1.50$ per case. In-store cannibalization of regular yogurt sales is estimated at 300 cases of full-price product ($p_1 = \$36.00, c = \$18.00$).

**Deliverables:**
1. Calculate the net cash recovery under Option 1 (Jobber Liquidation).
2. Calculate the net cash contribution under Option 2 (Urban Transshipment & Markdown).
3. Identify the minimum clearance sell-through volume in urban stores that makes Option 2 superior to Option 1.
*Acceptance Criteria:* Complete multi-echelon cash ledger and breakeven volume calculation. Time: 35 minutes.

---

## 13. Self-check answers

### Solution to Exercise 1
1. **Underage and Overage Costs:**
   $$c_u = p_1 - c + g = \$28.00 - \$14.50 + \$4.00 = \$17.50$$
   $$\bar{s} = s - d_f = \$0.00 - \$2.20 = -\$2.20$$
   $$c_o = c - \bar{s} = \$14.50 - (-\$2.20) = \$16.70$$
2. **True Extended Critical Fractile:**
   $$CR^* = \frac{c_u}{c_u + c_o} = \frac{17.50}{17.50 + 16.70} = \frac{17.50}{34.20} \approx 0.511696 \approx 0.5117$$
3. **Optimal Order Quantity ($Q^*$):**
   $$z^* = \Phi^{-1}(0.5117) \approx +0.0293$$
   $$Q^* = \mu + z^* \cdot \sigma = 85 + (0.0293 \times 18) = 85 + 0.527 = 85.53 \approx 86 \text{ steaks}$$
4. **Textbook Ignorance Error:**
   If ignoring $d_f$: $c_o^{\text{textbook}} = c - s = \$14.50$.
   $$CR_{\text{textbook}} = \frac{17.50}{17.50 + 14.50} = \frac{17.50}{32.00} = 0.546875 \approx 0.5469$$
   $$z_{\text{textbook}} = \Phi^{-1}(0.5469) \approx +0.1178$$
   $$Q_{\text{textbook}} = 85 + (0.1178 \times 18) = 85 + 2.12 = 87.12 \approx 87 \text{ steaks}$$
   *Error:* The legacy model orders 87 steaks instead of 86. While seemingly small on a single SKU daily basis (1.16% over-order), across 25 meat SKUs over 365 days, this systematic over-ordering generates $1 \times 25 \times 365 = 9,125$ excess units of terminal biological waste, costing the butchery $9,125 \times (\$14.50 + \$2.20) = \$152,387$ in preventable write-offs and incineration charges.

### Solution to Exercise 2
1. **Derivation of Cannibalization Breakeven Threshold:**
   Let incremental clearance profit equal cannibalization margin loss:
   $$\text{Incremental Profit from Clearance} = D_{2,\text{org}} \cdot [p_2 - c - (-d_f)] - \text{Labor} = D_{2,\text{org}} \cdot (p_2 - c + d_f) - \text{Labor}$$
   $$\text{Cannibalization Margin Loss} = (\gamma \cdot D_1) \cdot (p_1 - p_2)$$
   Setting them equal and solving for $\gamma$:
   $$\gamma_{\max} = \frac{D_{2,\text{org}} \cdot (p_2 - c + d_f) - \text{Labor}}{D_1 \cdot (p_1 - p_2)}$$
2. **Numerical Calculation:**
   - $p_2 - c + d_f = \$3.20 - \$1.80 + \$0.25 = \$1.65$.
   - $D_{2,\text{org}} \cdot \$1.65 = 25 \times \$1.65 = \$41.25$.
   - Total labeling labor: $25 \times \$0.10 = \$2.50$.
   - Net incremental clearance contribution: $\$41.25 - \$2.50 = \$38.75$.
   - Full-price margin gap: $p_1 - p_2 = \$5.50 - \$3.20 = \$2.30$.
   - Denominator: $D_1 \cdot (p_1 - p_2) = 40 \times \$2.30 = \$92.00$.
   $$\gamma_{\max} = \frac{\$38.75}{\$92.00} \approx 0.4212 \text{ (or } 42.12\%)$$
3. **Managerial Evaluation:**
   The observed cannibalization rate is $\gamma = 0.35$ (35.0%), which is strictly below the breakeven threshold of $42.12\%$.
   - Actual margin cannibalized: $0.35 \times 40 \times \$2.30 = \$32.20$.
   - Net profit lift: $\$38.75 - \$32.20 = +\$6.55$ per afternoon.
   *Decision:* The manager should execute the markdown. It generates positive economic value. However, because the margin of safety is narrow ($42.12\% - 35.0\% = 7.12\%$), the manager should segregate the clearance rack to prevent $\gamma$ from drifting upward.

### Solution to Exercise 3
1. **Option A (Donation via IRC § 170(e)(3)):**
   - Cost: $c = \$5.20$. Retail price: $p_1 = \$14.00$.
   - Potential deduction: $c + 0.5(p_1 - c) = \$5.20 + 0.5(\$14.00 - \$5.20) = \$5.20 + \$4.40 = \$9.60$.
   - Check cap: $2 \times c = 2 \times \$5.20 = \$10.40$.
   - Allowable deduction per unit: $\$9.60$ (since $\$9.60 \le \$10.40$).
   - Cash tax savings per unit at $\tau_{\text{corp}} = 26\%$:
     $$s_{\text{tax}} = 0.26 \times \$9.60 = \$2.496 \text{ per lunch}$$
   - Landfill disposal fee avoided per unit: $d_f = \$0.45$.
   - Total cash recovery per unit: $\$2.496 + \$0.45 = \$2.946$.
   - Total net cash value across 1,200 lunches:
     $$\text{Net Cash}_{\text{Option A}} = 1,200 \times \$2.946 = +\$3,535.20$$
2. **Option B (Emergency Markdown):**
   - Units sold at markdown: 900 units at $p_2 = \$4.90$.
   - Revenue from clearance: $900 \times \$4.90 = \$4,410.00$.
   - Disposal fee on 300 unsold units: $300 \times (-\$0.45) = -\$135.00$.
   - Cannibalization cost: 120 customers who would have bought fresh at $p_1 = \$14.00$ switched to $p_2 = \$4.90$, forfeiting $(p_1 - p_2) = \$9.10$ in gross margin:
     $$\text{Cannibalization Margin Loss} = 120 \times (\$14.00 - \$4.90) = 120 \times \$9.10 = \$1,092.00$$
   - Total net cash generated from inventory liquidation:
     $$\text{Net Cash}_{\text{Option B}} = \$4,410.00 - \$135.00 - \$1,092.00 = \$3,183.00$$
3. **Recommendation:**
   Option A (Donation) generates $\$3,535.20$, which strictly exceeds Option B's $\$3,183.00$ by $\$352.20$ (+11.1%). Furthermore, Option A carries zero commercial execution risk, avoids customer brand degradation from selling tired lunches, and provides positive social impact documentation. The chain should donate 100% of the lot.

### Solution to Exercise 4
1. **Expected Profit Function for Hour 3 ($t \in [2, 3]$):**
   Given $I(2)$ units remaining, price $p_2$, and arrival rate $\lambda_2(p_2) = 50 - 2.5 p_2$:
   Expected sales are $\mathbb{E}[\min(I(2), D_2)]$.
   If $I(2)$ is sufficiently large to satisfy demand, expected revenue is $p_2 \cdot \lambda_2(p_2) = p_2 (50 - 2.5 p_2) = 50 p_2 - 2.5 p_2^2$.
   Each unsold unit incurs disposal fee $d_f = \$1.00$.
2. **Optimal Clearance Price ($p_2^*$):**
   Differentiating expected unconstrained revenue with respect to $p_2$:
   $$\frac{d}{dp_2} [50 p_2 - 2.5 p_2^2] = 50 - 5.0 p_2 = 0 \implies p_2^* = \frac{50}{5.0} = \$10.00$$
   Check second derivative: $-5.0 < 0$ (concave maximum).
   At $p_2^* = \$10.00$, expected demand is $\lambda_2(10) = 50 - 2.5(10) = 25$ trays.
3. **Expected Total Profit:**
   - First 2 hours ($t \in [0, 2]$): Expected demand is $2 \times 20 = 40$ trays.
   - Trays sold at $p_1 = \$15.00$: 40 trays. Revenue = $40 \times \$15.00 = \$600.00$.
   - Inventory remaining at 13:00: $I(2) = 60 - 40 = 20$ trays.
   - At 13:00, expected demand is 25 trays, which exceeds on-hand inventory of 20 trays.
   - Therefore, all 20 trays sell out: Expected revenue = $20 \times \$10.00 = \$200.00$.
   - Unsold outdating: 0 trays. Disposal fee = $\$0.00$.
   - Total Procurement Cost: $60 \times \$6.00 = \$360.00$.
   - Total Expected Profit: $(\$600.00 + \$200.00) - \$360.00 = \$440.00$.

### Solution to Exercise 6
1. **Mandated Monthly Diversion Tonnage:**
   - Total organic waste generated = 4.5 metric tons/month.
   - Edible food fraction = $65\% \times 4.5 = 2.925$ metric tons/month.
   - Mandated recovery target under CA SB 1383 = $20\%$ of edible food.
   $$\text{Mandatory Diversion} = 0.20 \times 2.925 = 0.585 \text{ metric tons/month (or 1,290 lbs)}$$
2. **Net Monthly Financial Impact:**
   - Municipal landfill tipping fee avoided: $\$115/\text{ton}$.
   - Dedicated organic hauling fee avoided: $\$65/\text{ton}$.
   - Total disposal cost avoided per ton: $\$115 + \$65 = \$180/\text{ton}$.
   - Food bank transport and processing fee paid: $\$45/\text{ton}$.
   - Net savings per diverted ton: $\$180 - \$45 = \$135/\text{ton}$.
   - Net monthly cash savings on 0.585 tons:
     $$\text{Net Monthly Savings} = 0.585 \times \$135 = +\$78.98/\text{month}$$
   Compliance is directly cash-positive, generating direct operational savings while meeting state environmental mandates.
3. **Value-at-Risk (VaR) of Non-Compliance:**
   If the store fails to divert and is audited:
   - Moderate violation fine: $\$1,000$ per day.
   - Over a 30-day compliance cycle: $\$30,000$ in statutory civil liabilities.
   - Ratio of non-compliance fine to operational diversion cost ($0.585 \times \$45 = \$26.33$): Over $1,100 \times$. Non-compliance represents catastrophic corporate operational risk.

### Solution to Exercise 7
1. **Consumer Stopping Game Formulation:**
   Let consumer valuation for fresh product be $V_1$ and for day-of-expiry product be $V_2$ ($V_1 > V_2$).
   - If purchasing at 16:30 at $p_1$: Consumer utility is $U_{\text{buy}} = V_1 - p_1$.
   - If waiting until 17:00: Product price drops to $p_2$. However, stockout probability is $P_{\text{stockout}}$.
   - Expected utility of waiting: $U_{\text{wait}} = (1 - P_{\text{stockout}})(V_2 - p_2) + P_{\text{stockout}} \cdot 0$.
   The consumer waits if and only if $U_{\text{wait}} > U_{\text{buy}}$.
2. **Critical Stockout Probability ($P_{\text{stockout}}$):**
   $$(1 - P_{\text{stockout}})(V_2 - p_2) > V_1 - p_1 \implies 1 - P_{\text{stockout}} > \frac{V_1 - p_1}{V_2 - p_2} \implies P_{\text{stockout}} < 1 - \frac{V_1 - p_1}{V_2 - p_2}$$
   If the retailer intentionally restricts markdown inventory depth or randomizes execution timing such that $P_{\text{stockout}}$ exceeds this threshold, rational consumers will refuse to gamble and will purchase at full price $p_1$.
3. **Operational Counter-Mechanisms:**
   - *Stochastic Markdown Triggers:* Vary markdown execution times across a 90-minute window (e.g., between 16:15 and 17:45) based on real-time store foot traffic sensors, destroying the predictability required for waiting.
   - *Decoupled Secondary Channels:* Transfer expiring units away from primary shelf displays to a closed app-based locker (e.g., Flashfood) or rear-of-store clearance rack, maintaining primary shelf pricing integrity.


### Solution to Exercise 8
1. **Option 1 (Jobber Liquidation):**
   - Revenue: $4,000 \times \$7.00 = \$28,000$.
   - Disposal fee avoided: $4,000 \times \$2.50 = \$10,000$ (cash preserved).
   - Net cash realized from inventory:
     $$\text{Net Cash}_{\text{Option 1}} = \$28,000$$
   (Note: Sunk procurement $4,000 \times \$18.00 = \$72,000$ was already paid; accounting loss is $\$72,000 - \$28,000 = \$44,000$, but net cash inflow is $+\$28,000$).
2. **Option 2 (Urban Transshipment & Markdown):**
   - Clearance revenue: $3,500 \times \$18.00 = \$63,000$.
   - Freight transshipment cost: $4,000 \times \$1.20 = -\$4,800$.
   - Store disposal fee on 500 unsold cases: $500 \times (-\$1.50) = -\$750$.
   - Cannibalization margin loss: 300 cases forfeited full-price gross margin of $(p_1 - p_2) = \$36.00 - \$18.00 = \$18.00$:
     $$\text{Cannibalization Margin Loss} = 300 \times \$18.00 = -\$5,400$$
   - Net cash realized from inventory under Option 2:
     $$\text{Net Cash}_{\text{Option 2}} = \$63,000 - \$4,800 - \$750 - \$5,400 = +\$52,050$$
3. **Managerial Comparison & Breakeven Volume:**
   Option 2 generates $\$52,050$, outperforming Option 1's $\$28,000$ by $+\$24,050$ (+85.9%).
   To find the minimum clearance sales volume $V_{\text{min}}$:
   Let $V$ be cases sold, and $(4,000 - V)$ be cases discarded at $\$1.50$:
   $$V \cdot \$18.00 - (4,000 - V) \cdot \$1.50 - \$4,800 - \$5,400 \ge \$28,000$$
   $$18.00 V - 6,000 + 1.50 V - 10,200 \ge 28,000$$
   $$19.50 V - 16,200 \ge 28,000 \implies 19.50 V \ge 44,200 \implies V_{\text{min}} = \frac{44,200}{19.50} \approx 2,267 \text{ cases}$$
   If urban stores can sell at least 2,267 cases (a 56.7% clearance sell-through), proactive transshipment dominates secondary jobber liquidation.
---

## 14. Interview questions

### Question 1: How does a positive landfill disposal fee alter the classical newsvendor critical fractile?
**What a strong answer contains:** The candidate explicitly states that when an unsold unit incurs a net disposal fee $d_f > 0$, the net salvage value becomes negative ($\bar{s} = s - d_f < 0$). This increases the unit overage cost beyond procurement cost: $c_o = c - \bar{s} = c - s + d_f > c$. Because $c_o$ appears in the denominator of the critical fractile $CR^* = c_u / (c_u + c_o)$, a larger $c_o$ strictly depresses $CR^*$. The candidate notes that using textbook newsvendor models in grocery retail systematically over-orders stock, creating unnecessary biological waste and cash losses. They can write the derivative $\partial CR^* / \partial d_f < 0$ and discuss how rising municipal tipping fees force lean replenishment policies.  
**What a weak answer reveals:** The candidate treats disposal fees as a fixed facility overhead cost or suggests that salvage is always bounded at zero ($s \ge 0$). They fail to see the mathematical coupling between terminal waste fees and marginal order decisions.

### Question 2: What is demand cannibalization in perishable dynamic pricing, and how do you calculate its economic breakeven?
**What a strong answer contains:** The candidate distinguishes between organic clearance demand (price-sensitive bargain hunters who enter the market only at discounted prices) and cannibalized demand (regular shoppers who intended to pay full price $p_1$ but switch to marked-down stock $p_2$). They write out the breakeven condition: incremental gross profit from bargain hunters must exceed the surrendered margin on cannibalized shoppers plus tagging labor: $\Delta D_{\text{org}}(p_2 - c + d_f) > D_{\text{cann}}(p_1 - p_2) + \text{Labor}$. They discuss operational mitigations, such as physical separation of clearance racks, delayed markdown timing, and limits on discount depth.  
**What a weak answer reveals:** The candidate assumes that any item sold on markdown is a pure operational victory that recovers margin, completely ignoring the margin destroyed when high-willingness-to-pay shoppers buy discounted inventory.

### Question 3: How does the transition from 1D UPC barcodes to GS1 Digital Link (2D DataMatrix) change perishable retail operations?
**What a strong answer contains:** The candidate explains that linear UPC barcodes carry only the 14-digit GTIN product identifier, meaning the POS system is blind to the age or expiration date of the specific physical package. GS1 Digital Link encodes GTIN `(01)`, batch/lot `(10)`, and expiration date `(17)` in a 2D DataMatrix. This allows: (1) POS-level automated stop-sales that block scanning of expired products, (2) centralized dynamic pricing lookups that eliminate the labor and scanning errors of manual yellow stickers, and (3) automated real-time inventory age visibility across the supply chain.  
**What a weak answer reveals:** The candidate views 2D barcodes solely as a marketing mechanism to link consumers to corporate websites, missing the operational and food safety implications of embedded GS1 Application Identifiers.

### Question 4: Under what conditions does donating near-expiration food to a charity dominate marking it down in-store?
**What a strong answer contains:** The candidate analyzes the trade-off through three mechanisms: (1) tax arbitrage under IRC § 170(e)(3), which permits an enhanced deduction equal to cost plus half the mark-to-market margin, capped at $2 \times$ cost; (2) zero in-store cannibalization, because donated food is removed from the sales floor; and (3) zero terminal disposal fees. If in-store markdowns trigger high cannibalization ($\gamma > \gamma_{\text{max}}$) or require high manual tagging labor, the sure cash-equivalent tax shield of donation strictly exceeds the net expected recovery of in-store clearance.  
**What a weak answer reveals:** The candidate treats corporate food donation purely as a non-economic public relations activity, unaware of the tax code mechanics and cannibalization avoidance benefits.

### Question 5: How do you formulate continuous-time dynamic freshness pricing using dynamic programming?
**What a strong answer contains:** The candidate formulates the Hamilton-Jacobi-Bellman (HJB) equation where the state is $(t, I)$, calendar time and on-hand inventory. They define arrival intensity as a function of price and remaining shelf life $\tau = T - t$, such as $\lambda(p, \tau) = \lambda_0 e^{-\beta p} (\tau/T)^\theta$. They specify the terminal boundary condition $V(T, I) = -d_f \cdot I$ (or $s \cdot I$), proving that the optimal price $p^*(t, I)$ balances instantaneous revenue against the marginal loss in value $\Delta V(t, I) = V(t, I) - V(t, I-1)$.  
**What a weak answer reveals:** The candidate can only describe heuristic rules (e.g., "drop price by 10% every day") and cannot articulate the underlying continuous-time stochastic control structure.

### Question 6: What is the "Procrastination Markdown" trap, and why is it prevalent in grocery retail?
**What a strong answer contains:** The candidate explains that retail store managers and department clerks often delay marking down inventory until the final hours of the product's shelf life because they hope late-arriving full-price customers will buy it. However, customer arrival rates are finite. By the time markdowns are finally applied, remaining shopper foot traffic is insufficient to absorb the stock, resulting in 80%+ terminal waste. The mitigation is algorithmic, automated clearance timing triggered strictly by hours of remaining shelf life rather than clerk discretion.  
**What a weak answer reveals:** The candidate blames warehouse forecasting errors rather than store-level execution delays, failing to understand how clearance velocity decays with elapsed time.

### Question 7: How do statutory organic waste mandates like California SB 1383 impact multi-echelon procurement contracts?
**What a strong answer contains:** The candidate explains that municipal landfill bans impose statutory civil fines and mandatory commercial food rescue quotas. This forces grocery retailers to rewrite supplier contracts: (1) tightening Minimum Remaining Shelf Life (MRSL) covenants at distribution center receipt gates, (2) transferring disposal and return logistics fees back to food processors via chargebacks (vendor deductions), and (3) requiring certified food diversion contracts with audited waste haulers and food recovery non-profits.  
**What a weak answer reveals:** The candidate thinks environmental regulations only affect municipal garbage trucks and have no impact on supply chain procurement terms or vendor agreements.

### Question 8: How does accounting standard ASC 330 / IAS 2 (LCNRV) interact with perishable markdown policies?
**What a strong answer contains:** The candidate cites the Lower of Cost and Net Realizable Value (LCNRV) rule. When a retailer marks down inventory below its landed cost $c$ (e.g., discounting a $\$6.00$ salad tray to $\$3.00$ to clear it), or when remaining shelf life makes outdating highly probable, the accounting system must immediately record an inventory impairment write-down to NRV on the balance sheet. Waiting until the physical product is discarded in a dumpster improperly defers expenses and misstates interim balance sheet asset valuations.  
**What a weak answer reveals:** The candidate assumes inventory maintains its full purchase cost valuation on corporate books until the physical moment it is tossed into a trash compactor.

### Question 9: Why do standard cycle service level (CSL) targets cause chronic over-ordering in perishable departments?
**What a strong answer contains:** The candidate explains that classical service level formulas (e.g., $Q = \mu + k\sigma$ with $k = \Phi^{-1}(\text{CSL})$) treat inventory as non-perishing. In durable goods, unsold units simply wait for future demand. In perishables, unsold units crossing the expiration boundary expire and incur disposal charges $d_f$. Because the marginal cost of overstocking is $c + d_f$ rather than just the holding cost $h$, targeting high service levels (e.g., 98%) forces massive safety stock additions that inevitably spoil. The candidate argues for unconstrained economic fractile optimization over rigid service level mandates.  
**What a weak answer reveals:** The candidate insists that higher service levels always maximize store revenue and customer satisfaction, failing to recognize that safety stock in perishables directly translates to spoiled bio-waste.

### Question 10: How do you design an Electronic Shelf Label (ESL) failure failover protocol to protect price integrity?
**What a strong answer contains:** The candidate details a multi-tiered failover: (1) local gateway caching of pricing schedules so labels update even if cloud connectivity is interrupted, (2) POS price authority: the POS central database is the authoritative fiscal register; if an ESL tag fails to refresh, the POS price is honored to comply with consumer protection weights and measures laws, (3) heart-beat RF ping monitoring: tags failing to report battery or screen refresh within 12 hours generate an automated task on store clerk mobile devices for physical inspection.  
**What a weak answer reveals:** The candidate suggests that if an ESL breaks, clerks should immediately re-label every product by hand, ignoring store labor constraints and scale across 50,000 tags.

---

## 15. Key terms

- **Extended Newsvendor Model:** Stochastic single-period inventory optimization incorporating negative net salvage values driven by municipal disposal and hauling fees.
- **Unit Disposal Fee ($d_f$):** Direct financial cost per unit incurred to transport, de-package, and landfill or incinerate expired perishable waste.
- **Net Salvage Value ($\bar{s}$):** Residual economic recovery per unsold unit after subtracting terminal disposal and handling costs: $\bar{s} = s - d_f$.
- **Unit Overage Cost ($c_o$):** Total financial loss per unsold unit, defined as $c - \bar{s} = c - s + d_f$.
- **Unit Underage Cost ($c_u$):** Opportunity margin lost per unmet unit of demand plus customer goodwill penalty: $p_1 - c + g$.
- **Critical Fractile ($CR^*$):** Optimal service probability balancing underage and overage costs: $c_u / (c_u + c_o)$.
- **Dynamic Freshness Pricing:** Real-time or staged adjustment of retail prices based on remaining product shelf life and current inventory depth.
- **Discount Depth ($\delta$):** Percentage price discount applied to aging perishable inventory: $(p_1 - p_2) / p_1$.
- **Markdown Trigger ($t_m$):** Operational threshold (elapsed time or remaining shelf life) that initiates a price discount.
- **Cannibalization Rate ($\gamma$):** Fraction of full-price consumer demand that shifts to discounted clearance inventory when displayed simultaneously.
- **Cannibalization Breakeven Threshold ($\gamma_{\max}$):** Maximum switching rate beyond which price discounts destroy more margin than they recover in avoided waste fees.
- **Electronic Shelf Label (ESL):** Digital e-paper shelf tag updated wirelessly from central retail systems to execute dynamic pricing.
- **GS1 Digital Link:** Web-enabled 2D barcode standard encoding GTIN, batch/lot, and expiration date in a structured URI.
- **GS1 Application Identifier AI (17):** Standard data syntax encoding mandatory product expiration date formatted as `YYMMDD`.
- **Lower of Cost and Net Realizable Value (LCNRV):** GAAP (ASC 330) and IFRS (IAS 2) accounting rule requiring inventory write-downs when expected selling price less disposal cost falls below historical cost.
- **IRC Section 170(e)(3):** US Internal Revenue Code provision permitting corporations an enhanced tax deduction for donating food inventory to qualified charities.
- **California Senate Bill 1383 (SB 1383):** Landmark environmental mandate requiring commercial food generators to divert 20% of edible food waste from landfills.
- **Loi Garot:** French statutory legislation prohibiting supermarkets from destroying unsold edible food and mandating charitable donation partnerships.
- **Anaerobic Digestion (AD):** Industrial biological processing of organic bio-waste into methane biogas and agricultural soil digestate.
- **Landfill Tipping Fee ($T_{\text{tip}}$):** Commercial gate fee charged by municipal landfills per ton of waste dumped.
- **Yellow Sticker:** Manual adhesive label applied to retail packaging to communicate in-store clearance discounts.
- **Flashfood / Too Good To Go:** Secondary app-based marketplaces liquidating near-expiration commercial food inventory to consumers.
- **Afresh Technologies:** AI-driven fresh replenishment platform optimizing grocery orders using non-parametric decay models.
- **Wasteless:** Dynamic retail pricing software integrating ESL systems and expiration date tracking to automate clearance markdowns.
- **Expected Value of Dynamic Pricing (EVDP):** Incremental expected cycle profit achieved by multi-stage dynamic markdowns compared to static newsvendor ordering.
- **Phantom Margin Trap:** Operational failure where over-ordering occurs because procurement models omit disposal costs from overage calculations.
- **Procrastination Markdown:** Operational failure where store staff delay clearance discounts until remaining foot traffic is insufficient to clear stock.
- **Strategic Consumer Waiting:** Consumer behavior where shoppers delay purchasing full-price perishables until scheduled clearance discount hours.
- **Hazard Rate of Depletion:** Instantaneous conditional probability of selling a unit of inventory given elapsed shelf life and displayed price.
- **Sunrise 2027:** GS1 global initiative transitioning retail point-of-sale systems from 1D UPC barcodes to 2D DataMatrix/QR codes.

---

## 16. Sources

- **Arrow, K. J., Karlin, S., & Scarf, H. (1958).** *Studies in the Mathematical Theory of Inventory and Production.* Stanford University Press. [Foundational derivation of stochastic inventory and newsvendor optimality].
- **Bitran, G., & Caldentey, R. (2003).** An Overview of Pricing Models for Revenue Management. *Manufacturing & Service Operations Management*, 5(3), 203–229. DOI: 10.1287/msom.5.3.203.16031.
- **Elmaghraby, W., & Keskinocak, P. (2003).** Dynamic Pricing in the Presence of Inventory Considerations: Research Overview, Current Practices, and Future Directions. *Management Science*, 49(10), 1287–1309. DOI: 10.1287/mnsc.49.10.1287.17315.
- **Ferguson, M., & Koenigsberg, O. (2007).** How Should a Firm Manage Perishable Inventory? The Trade-Off Between Dynamic Pricing and Markdown Policies. *Production and Operations Management*, 16(6), 705–718. DOI: 10.1111/j.1937-5956.2007.tb00291.x.
- **Nahmias, S. (1982).** Perishable Inventory Theory: A Review. *Operations Research*, 30(4), 680–708. DOI: 10.1287/opre.30.4.680.
- **Nahmias, S. (2011).** *Perishable Inventory Systems.* Springer Series in Operations Research and Financial Engineering. ISBN: 978-1-4419-7989-6.
- **GS1 Standard General Specifications (2024).** *GS1 Application Identifiers and Digital Link Architecture.* GS1 AISBL. Available at: `https://www.gs1.org/standards/barcodes/application-identifiers`.
- **California Department of Resources Recycling and Recovery (CalRecycle) (2022).** *Short-Lived Climate Pollutants: Organic Waste Reductions (SB 1383).* California Code of Regulations, Title 14, Division 7, Chapter 12.
- **ReFED (2023).** *Retail Food Waste Action Guide: Financial Modeling of Dynamic Pricing and Donation Infrastructure.* ReFED Insights Engine.
- **U.S. Internal Revenue Code (2024).** *26 U.S. Code § 170(e)(3) — Special rule for contributions of inventory and other property.* Legal Information Institute, Cornell Law School.
- **U.S. Congress (2022).** *Food Donation Improvement Act of 2022.* Public Law 117-362, 136 Stat. 6297. Amending the Bill Emerson Good Samaritan Food Donation Act (42 U.S.C. § 1791).
- **French Ministry of Ecology (2016).** *LOI n° 2016-138 du 11 février 2016 relative à la lutte contre le gaspillage alimentaire (Loi Garot).* Journal Officiel de la République Française.

---

## 17. Links

- **Upward to Module Index:** [`_index.md`](_index.md) — Food, Agriculture, and Perishable Supply Chains Module Architecture
- **Upward to Survey File:** [`../../../course/32_food_agriculture_perishables.md`](../../../course/32_food_agriculture_perishables.md) — Complete Perishable Supply Chains Curriculum Survey
- **Preceding Sibling Topic:** [`03_shelf-life-fefo.md`](03_shelf-life-fefo.md) — Shelf Life, FEFO, and Age-Structured Inventory
- **Preceding Sibling Topic:** [`02_perishable-forecasting.md`](02_perishable-forecasting.md) — Forecasting Perishable Demand Under Weather and Freshness Decay
- **Preceding Sibling Topic:** [`01_agri-production-harvest.md`](01_agri-production-harvest.md) — Agricultural Production, Seasonality, and Harvest Planning
- **Succeeding Sibling Topic:** [`05_cold-chain-quality.md`](05_cold-chain-quality.md) — Cold Chain Logistics, MKT, and Quality Degradation
- **Succeeding Sibling Topic:** [`06_traceability-food-safety.md`](06_traceability-food-safety.md) — Traceability, FSMA 204, and Food Safety Architecture
- **Succeeding Sibling Topic:** [`07_food-waste-donation.md`](07_food-waste-donation.md) — Food Waste Measurement, Donation Logistics, and Valorization Economics
- **Cross-Module Link:** [`../../02_methods/10_inventory_optimization/01_single-period-newsvendor.md`](../../02_methods/10_inventory_optimization/01_single-period-newsvendor.md) — Classical Foundations of Single-Period Stochastic Inventory
- **Cross-Module Link:** [`../../03_commerce/21_revenue_management/03_markdown-optimization.md`](../../03_commerce/21_revenue_management/03_markdown-optimization.md) — Retail Markdown Timing and Clearance Optimization
- **Cross-Module Link:** [`../../01_foundations/03_data_engineering_interoperability/09_gs1-epcis.md`](../../01_foundations/03_data_engineering_interoperability/09_gs1-epcis.md) — GS1 Identification, 2D Barcodes, and EPCIS Standards
