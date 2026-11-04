# Component Allocation under Shortage: Game-Theoretic Rationing, Truth-Inducing Mechanisms, and Multi-Objective Optimization

> Module 40 — Telecom, Technology Hardware, and Aerospace Supply Chains
> Topic 02 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 09 (Inventory Theory and Stochastic Replenishment), Module 10 (Multi-Echelon Inventory Optimization), Module 14 (Supply Chain Game Theory and Contract Design), Module 31 (High-Tech, Electronics, and Semiconductor Operations), Module 37 (Supply Chain Risk, Resilience, and Disruption Management)
> Version 2026.5

---

## 1. Why this topic exists

In standard discrete manufacturing, procurement operates under the assumption that suppliers can scale output to match purchase orders given sufficient lead time, or that price mechanisms clear discrepancies between supply and demand. In the semiconductor and technology hardware ecosystem, this assumption fails catastrophically during supply contractions.

Front-end semiconductor fabrication takes place in multi-billion-dollar cleanrooms where wafer cycle times range from 14 to 26 weeks. Back-end outsourced semiconductor assembly and test (OSAT) adds another 4 to 8 weeks, resulting in total production lead times of 20 to 52 weeks. Wafer fabrication capacity is strictly inelastic in the short to medium term. Adding a new advanced fabrication line requires $10 billion to $20 billion in capital expenditure and 24 to 36 months of facility construction, tool installation (such as ASML extreme ultraviolet lithography scanners with multi-year order backlogs), and process qualification. 

When aggregate downstream demand exceeds manufacturing capacity, silicon vendors, integrated device manufacturers (IDMs), and authorized component distributors cannot clear the market by dynamically raising prices on long-term enterprise customers without breaching Master Supply Agreements (MSAs) or destroying strategic commercial partnerships. Instead, supply is *administered* through administrative rationing rules, known in the electronics industry as **component allocation**.

Administrative rationing unleashes severe behavioral distortions. When downstream buyers—including telecom equipment manufacturers (e.g., Ericsson, Nokia, Cisco), consumer electronics original equipment manufacturers (OEMs like Apple, Samsung, Dell), automotive tier-1 integrators (e.g., Bosch, Continental), and electronics manufacturing services (EMS providers such as Foxconn, Jabil, Flex)—discover that an integrated circuit (IC) is placed on allocation, they face a severe operational threat: the **golden screw problem**. A $500,000 5G cellular base station or a $100,000 enterprise networking core switch cannot be shipped, invoiced, or deployed if a single $1.50 power management IC (PMIC), Ethernet physical layer transceiver (PHY), or micro-controller unit (MCU) is missing. 

To protect assembly lines from catastrophic "line-down" stoppages, procurement managers respond rationally to allocation by inflating forward purchase orders. If a buyer requires 100,000 units per quarter and anticipates that the chipmaker will allocate only 50% of ordered quantities, the buyer places an order for 200,000 units. Furthermore, the buyer often duplicates identical orders across multiple authorized distributors (e.g., Arrow, Avnet, WPG) and direct manufacturer sales channels. This phenomenon, known as **double ordering** or **order gaming**, creates massive **phantom demand** across the order book.

The consequences of unconstrained order gaming are economically disastrous:
1. **Order Book Decoupling:** The semiconductor manufacturer's order backlog ceases to reflect real consumption, expanding to three to five times total fab capacity.
2. **Artificial Lead-Time Inversion:** As the nominal order backlog swells, published component lead times explode from 16 weeks to 52, 78, or even 99 weeks. This triggers additional panic ordering from secondary and tertiary tiers of the supply chain, amplifying the bullwhip effect.
3. **Misallocated Physical Inventory:** Under naive rationing schemes (such as proportional allocation), the buyers who inflate their orders the most aggressively secure the highest share of physical output, leaving disciplined buyers starved of critical parts.
4. **Catastrophic Bullwhip Bull-Whip Whiplash and Write-Downs:** The moment end-customer demand softens or additional fabrication capacity comes online, downstream buyers abruptly cancel their duplicate, inflated orders. The artificial backlog evaporates overnight, stranding chipmakers and distributors with billions of dollars of unsellable, obsolete work-in-progress (WIP) and finished goods inventory. 

This boom-and-bust cycle was the root cause of the historic Cisco Systems $2.2 billion inventory write-down in 2001, and it recurred at global scale during the 2020–2023 semiconductor crisis, which idled automotive assembly plants worldwide and delayed 5G telecommunication infrastructure rollouts.

Solving component allocation requires moving beyond ad-hoc spreadsheet rationing and political escalations. It demands rigorous mathematical modeling: understanding the game-theoretic incentives of downstream buyers, designing **truth-inducing allocation mechanisms** (such as turn-and-earn and capacity reservation contracts) that penalize order inflation, formulating **multi-objective mixed-integer linear programs (MILP)** that optimize revenue and contractual Service Level Agreements (SLAs) while enforcing baseline customer equity, and deploying statistical **order de-biasing filters** that extract genuine consumption signals from distorted order books.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Map** the structural semiconductor lead-time continuum from wafer start, photolithography, wafer sort, packaging, and final burn-in test to quantify rigid capacity boundaries ($K$) across planning horizons.
2. **Formulate** the mathematical payoff functions for downstream buyers under proportional allocation and prove why proportional rationing induces a Nash equilibrium characterized by infinite order inflation (double ordering) when cancellation penalties are zero.
3. **Design** truth-inducing allocation mechanisms, including lexicographic priority rules and turn-and-earn policies, that set the marginal allocation derivative with respect to order volume ($\partial A_i / \partial O_i$) to zero, thereby disincentivizing gaming.
4. **Construct** a multi-objective Mixed-Integer Linear Program (MILP) that allocates scarce components across competing product lines, customer tiers, and time periods, balancing gross margin contribution, customer strategic weights, contractual SLA liquidated damages, and minimum equity floors.
5. **Implement** algorithmic de-biasing techniques on inflated order books using historical sell-through reconciliation, Point-of-Sale (POS) telemetry, device activation data, and customer inventory days-on-hand (DOI) audits.
6. **Evaluate** legal, regulatory, and commercial compliance constraints governing component rationing, including Uniform Commercial Code (UCC) § 2-615 fair allocation mandates, Robinson-Patman Act price and service discrimination limits, and non-cancellable, non-returnable (NCNR) contract structures.
7. **Solve** an end-to-end component allocation problem using bounded Python optimization tooling (`scipy.optimize.linprog`), extracting optimal shipment schedules under capacity constraints and contractual shortage penalty structures.
8. **Audit** supply chain warning indicators to distinguish genuine demand surges from speculative phantom order backlogs before component allocation collapses into inventory write-downs.

---

## 3. The core mental model

Component allocation is an administrative control problem operating over an asynchronous, multi-echelon network subject to strict physical production delays and strategic human agent behavior. 

When silicon fabrication capacity $K$ is less than aggregate unconstrained demand $\sum D_i$, the physical market clearing mechanism shifts from price adjustments to administrative rationing. The system is governed by three interacting, closed-loop feedback subsystems:
1. **The Physical Capacity Constraint Subsystem:** Wafer starts, yield degradation curves, and back-end assembly bottlenecks determine the strictly finite volume of net shippable die ($K$).
2. **The Order Book Gaming Subsystem (The Unstable Bullwhip Loop):** When downstream customers observe allocation factors $\alpha < 1$, their rational response is to artificially inflate submitted orders $O_i > D_i$ to capture a larger slice of $K$, triggering lead-time inflation and secondary panic orders.
3. **The Governance and De-Biasing Subsystem (The Stabilizing Control Loop):** The component vendor implements truth-inducing allocation rules (turn-and-earn, capacity deposits, lexicographic SLAs) and filters the order book using external telemetry to reconcile submitted orders with genuine consumption.

The following ASCII architectural diagram illustrates the structural feedback mechanisms, information flows, and gaming dynamics connecting wafer fabs, chipmakers, distributors, EMS assemblers, and technology OEMs:

```text
========================================================================================================================
                      SEMICONDUCTOR ALLOCATION ARCHITECTURE & ORDER GAMING DYNAMICS
========================================================================================================================

   UPSTREAM SUPPLY LAYER                    MANUFACTURER / ALLOCATION ENGINE                 DOWNSTREAM CUSTOMER LAYER
 (Inelastic Physics: 20-52 Wks)                (Administrative Rationing)                     (Strategic Game-Theoretic Agents)

 ┌───────────────────────────┐                                                       ┌───────────────────────────────┐
 │   Wafer Fab Operations    │                                                       │   Customer A (Telecom OEM)    │
 │  - Cleanroom cycle: 20 wks│                                                       │  - True Demand: D_A           │
 │  - Wafer Starts: W_t      │                                                       │  - Contract SLA: Min Volume   │
 │  - Defect Density: D_0    │                                                       │  - Shortage Penalty: \pi_A    │
 └─────────────┬─────────────┘                                                       └───────────────┬───────────────┘
               │                                                                                     │
               ▼ Net Die Output                                                                      │ Inflates Order:
 ┌───────────────────────────┐               ┌─────────────────────────────────┐                     │ O_A = D_A / \alpha
 │   OSAT Assembly & Test    │               │     ORDER BOOK AUDIT ENGINE     │                     │
 │  - Packaging: 4-8 wks     │               │  - Run-rate validation (EDI 867)│◄────────────────────┤
 │  - Final Test Yield: Y_t  │               │  - Activation telemetry tracking│                     │
 └─────────────┬─────────────┘               │  - Days-on-Hand (DOH) auditing  │                     │
               │                             └────────────────┬────────────────┘                     │
               ▼ Net Usable Supply (K)                        │ Sanitized Demand (\hat{D})           │
 ┌────────────────────────────────────────────────────────────┴───────────────────────────────┐      │
 │                     MULTI-OBJECTIVE ALLOCATION OPTIMIZATION ENGINE                         │      │
 │                                                                                            │      │
 │   Objective: Maximize Margin + Strategic Value - SLA Shortfall Penalties                   │      │
 │   Subject to:                                                                              │      │
 │     1. Physical Capacity:  \sum x_i <= K                                                   │      │
 │     2. Contractual SLAs:   x_i + s_i >= SLA_min_i                                          │      │
 │     3. Baseline Equity:    x_i >= \alpha_floor * \hat{D}_i                                 │      │
 │     4. Packaging Quotas:   x_i \in {Integer Multiples of Standard Pack Quantity}           │      │
 └────────────────────────────────────────────┬───────────────────────────────────────────────┘      │
                                              │                                                      │
                                              ├──────────────────────────────┐                       │
                         Allocated Flow A_A   │         Allocated Flow A_B   │                       │
                                              ▼                              ▼                       │
                               ┌─────────────────────────────┐┌──────────────────────────────┐       │
                               │ Customer A Physical Inflow  ││ Customer B Physical Inflow   │       │
                               │  - Received: A_A <= K       ││  - Received: A_B <= K        │       │
                               └──────────────┬──────────────┘└──────────────┬───────────────┘       │
                                              │                              │                       │
                                              ▼                              ▼                       │
                               ┌─────────────────────────────┐┌──────────────────────────────┐       │
                               │ Downstream Hardware Build   ││ Downstream Hardware Build    │       │
                               │  - Golden Screw Constraint: ││  - Missing IC Halts Factory  │       │
                               │    PCB Assembly Blocked     ││  - Idle SMT Lines            │       │
                               └──────────────┬──────────────┘└──────────────┬───────────────┘       │
                                              │                              │                       │
                                              ▼                              ▼                       │
 ┌───────────────────────────────────────────────────────────────────────────────────────────┐       │
 │                       BEHAVIORAL FEEDBACK & GAMING DYNAMICS                               │       │
 │                                                                                           │       │
 │   - If Allocation Rule is Proportional: A_i = K * (O_i / \sum O_j)                         │       │
 │   - Buyer Incentive: \partial A_i / \partial O_i > 0  ==> Rational strategy is O_i -> \infty│───────┘
 │   - Phantom Orders Cascade into Distributor Networks (Arrow, Avnet) & Direct Channels     │
 │   - Bullwhip Collapse: Sudden order cancellations trigger massive inventory write-downs   │
 └───────────────────────────────────────────────────────────────────────────────────────────┘
========================================================================================================================
```

### The Three Interlocking Operational Loops

1. **The Capacity Inelasticity Constraint (Supply Engine):**
   The physical output of silicon devices is governed by irreversible chemical, thermodynamic, and mechanical operations. Silicon wafers pass through hundreds of photolithography, ion implantation, chemical vapor deposition (CVD), and chemical mechanical planarization (CMP) steps. Output cannot be accelerated by running equipment faster; doing so introduces micro-defects that collapse wafer sort yield. Because capacity $K$ is invariant over the planning horizon, the allocation engine must treat $K$ as an absolute upper bound.

2. **The Proportional Allocation Gaming Trap (The Destructive Loop):**
   When the chipmaker rations capacity based on the ratio of a customer's submitted order to the total order book ($A_i = K \cdot \frac{O_i}{\sum O_j}$), every buyer has a direct mathematical incentive to inflate their order $O_i$. By placing an order twice as large as actual requirements, a buyer increases their fractional allocation. When all buyers adopt this dominant strategy, aggregate orders expand exponentially while physical supply remains flat. The allocation factor drops further, reinforcing the buyers' conviction that shortages are worsening, prompting yet another wave of phantom orders.

3. **The Truth-Inducing Stabilization Loop (The Control Engine):**
   To break the gaming trap, the allocation architecture decouples physical allocations from the forward order book $O_i$. Under **turn-and-earn**, allocations are strictly indexed to historical verified consumption $S_i$ over past baseline quarters. Because $\partial A_i / \partial O_i = 0$, order inflation yields zero additional silicon. Concurrently, the manufacturer binds forward commitments with non-cancellable, non-returnable (NCNR) legal agreements backed by cash deposits, forcing buyers to absorb financial liability for phantom orders.

---

## 4. Definitions and vocabulary

The following terminology governs component allocation, semiconductor operations, and game-theoretic rationing mechanisms:

| Term | Precise Technical Definition | Common Industry Misconception | Unit / Standard |
|---|---|---|---|
| **Component Allocation** | The administrative rationing of manufactured inventory when aggregate unconstrained demand strictly exceeds shippable capacity ($\sum D_i > K$). | Confused with standard production scheduling or order prioritization. | Dimensionless ratio ($\alpha \le 1.0$) or discrete units |
| **Golden Screw Problem** | A system-level production stoppage where an entire finished product cannot be completed or shipped due to the absence of a single low-cost, sole-sourced component. | Believed to be caused only by expensive ASICs; most frequently triggered by standard commoditized passives or PMICs. | Binary state (Line-Down vs Operational) |
| **Double Ordering (Order Gaming)** | The deliberate submission of duplicate or inflated purchase orders across multiple suppliers or distribution channels to secure a larger share of rationed capacity. | Mistaken for genuine sudden surges in end-customer macroeconomic demand. | Quantity units ($O_i - D_i$) |
| **Phantom Demand** | The portion of an open order book that represents speculative or defensive over-ordering and will be cancelled the moment lead times normalize or supply improves. | Treated as firm contractual backlog by corporate revenue forecasters. | Units or currency ($) |
| **Proportional Allocation** | A rationing policy where each participant receives an allocation strictly proportional to their submitted order: $A_i = K \cdot (O_i / \sum O_j)$. | Assumed to be "fair" and unbiased; in practice, it mathematically guarantees maximum order book gaming. | Ratio / Percentage |
| **Uniform Allocation** | A rationing policy that caps every claimant at a uniform maximum threshold $\bar{q}$, fulfilling smaller orders in full: $A_i = \min(O_i, \bar{q})$. | Confused with proportional rationing; uniform allocation protects small buyers while capping mega-accounts. | Units per customer |
| **Lexicographic Minimax Fairness** | An allocation mechanism that maximizes the minimum allocation ratio across all claimants, subject to hierarchical priority constraints. | Confused with simple unweighted equal sharing. | Mathematical vector ordering |
| **Turn-and-Earn Allocation** | An allocation rule where current supply is rationed strictly based on historical verified sales velocity or consumption run-rate over a prior baseline window. | Believed to freeze growth; it eliminates gaming by making current order size irrelevant to current allocation. | Ratio based on trailing historical units |
| **NCNR Agreement** | Non-Cancellable, Non-Returnable commercial purchase contract binding the buyer to take physical delivery and pay 100% invoice value regardless of market changes. | Assumed to be legally bulletproof; frequently challenged under commercial impracticability or insolvency. | Legal contract classification |
| **Wafer Sort Yield ($Y_{\text{sort}}$)** | The percentage of functional, defect-free raw integrated circuit dies on a processed silicon wafer prior to dicing and packaging. | Confused with back-end assembly yield or overall fab line yield. | Percentage ($0 \le Y \le 100\%$) |
| **DPW (Dies Per Wafer)** | The total gross geometric count of silicon dies that fit physically onto a circular semiconductor wafer of diameter $d$ (e.g., 200mm or 300mm). | Assumed to equal shippable units; gross DPW must be multiplied by sort yield to determine net usable die. | Gross units per wafer |
| **OSAT** | Outsourced Semiconductor Assembly and Test; third-party packaging, ball grid array (BGA) soldering, wire bonding, and final functional test facilities. | Assumed to have negligible lead times compared to wafer fabs; OSAT substrate shortages often bind overall capacity. | Manufacturing tier designation |
| **Lead Time Inversion** | A structural distortion where component lead time exceeds the target end product's commercial market life cycle or customer contract horizon. | Viewed as an operational inconvenience; actually breaks conventional MRP scheduling logic completely. | Time (Weeks) |
| **Point-of-Sale (POS) Sell-Through** | Verified consumption reporting (EDI 867) detailing inventory transferred from distributor warehouses to end customers. | Confused with sell-in (shipments from chipmaker to distributor shelves). | Units / Period |
| **Days of Inventory on Hand (DOI / DOH)** | The physical inventory currently held in stock divided by average daily consumption rate: $\text{DOI} = I / \bar{c}$. | Assumed to be uniform across tiers; distributors often hoard high DOI while OEM assembly lines starve. | Days |
| **Liquidated Damages SLA** | Contractual clauses establishing predetermined financial penalties payable per day or per unit of unfulfilled delivery commitments. | Confused with standard breach of contract lawsuits; liquidated damages are automatic and formulaic. | Currency per unit (\$ / unit) |
| **UCC § 2-615** | Section of the Uniform Commercial Code governing commercial excuse by failure of presupposed conditions, mandating "fair and reasonable" customer allocation. | Believed to allow arbitrary vendor favoritism; legally requires objective, non-discriminatory allocation schemas. | US Statutory Standard |
| **Bayesian Order De-biasing** | Statistical shrinkage filtering that discounts forward purchase orders based on historical customer forecast error and cancellation variance. | Confused with standard exponential smoothing; explicitly models the strategic gaming distribution. | Probability / Discount Factor |
| **Reel / Tray Packing Constraint (SPQ)** | Standard Pack Quantity; the physical packaging increment (e.g., tape-and-reel of 3,000 ICs, waffle tray of 250 ICs) below which components cannot be shipped. | Overlooked in continuous LP models; introduces discrete mixed-integer combinatorial constraints. | Integer multiples |

---

## 5. The mechanics, step by step

Component allocation requires a disciplined, multi-stage analytical process. The workflow transforms raw fabrication capacity and noisy customer demand signals into an legally compliant, profit-maximizing, and game-resistant shipment schedule.

```text
+---------------------------------------------------------------------------------------------------+
|                              COMPONENT ALLOCATION EXECUTION PIPELINE                              |
+---------------------------------------------------------------------------------------------------+
  [Step 1: Capacity & Yield Reconciliation]
    |--> Compute Gross DPW, apply defect density models (Murphy/Poisson), factor assembly & test yield
    |--> Establish definitive binding shippable capacity bound: K_t
    |
  [Step 2: Demand Signal Sanitization & De-Biasing]
    |--> Audit open order book against POS sell-through, device activations, and customer inventory DOI
    |--> Apply Bayesian shrinkage filters: discount orders exhibiting historical gaming patterns
    |--> Generate sanitized baseline demand vector: \hat{D}_t
    |
  [Step 3: Game-Theoretic Incentive Audit]
    |--> Evaluate customer payoff functions under prospective allocation mechanisms
    |--> Verify whether proposed mechanism satisfies truth-inducing property (\partial A_i / \partial O_i = 0)
    |
  [Step 4: Lexicographic Priority & Contractual Segmentation]
    |--> Segment demand into Tier 1 (Life-Safety & Binding SLAs), Tier 2 (Strategic), Tier 3 (Spot)
    |--> Compute baseline equity floors to avoid complete assembly line starvation
    |
  [Step 5: Multi-Objective LP/MILP Optimization]
    |--> Formulate and solve optimization model: Maximize Contribution Margin minus SLA Shortfalls
    |--> Enforce component capacity bounds, minimum contract commitments, and SPQ packaging integrality
    |
  [Step 6: Commercial Binding & NCNR Enforcement]
    |--> Convert optimized allocations into non-cancellable, non-returnable (NCNR) purchase agreements
    |--> Require cash deposits or irrevocable letters of credit to lock downstream commitment
    |
  [Step 7: Order Book Clearing & Post-Allocation De-escalation]
    |--> Monitor lead-time convergence and gracefully unwind allocation rules as capacity expands
+---------------------------------------------------------------------------------------------------+
```

### Step 1: Baseline Capacity and Usable Yield Calculation

The allocation engine must determine the physical upper bound $K$ of shippable units over planning horizon $T$. In semiconductor manufacturing, shippable inventory is not equal to wafer starts; it is heavily attenuated by wafer geometry and multi-stage yields.

1. **Calculate Gross Dies Per Wafer (DPW):**
   For a circular silicon wafer of diameter $d$ (typically 300 mm) and a rectangular die of dimensions length $L$ and width $W$ (area $A_{\text{die}} = L \times W$):
   $$\text{DPW} = \frac{\pi \left(\frac{d}{2}\right)^2}{A_{\text{die}}} - \frac{\pi d}{\sqrt{2 A_{\text{die}}}}$$
   The first term represents wafer area divided by die area; the second term approximates edge-loss dies intersected by the circular wafer perimeter.

2. **Model Wafer Sort Yield ($Y_{\text{sort}}$):**
   Using Murphy's yield model, which accounts for non-uniform spatial defect clustering across the wafer:
   $$Y_{\text{sort}} = \left(\frac{1 - e^{-A_{\text{die}} D_0}}{A_{\text{die}} D_0}\right)^2$$
   where $D_0$ is the defect density per unit area (defects/$\text{cm}^2$).

3. **Incorporate Downstream Assembly and Test Yields:**
   Processed wafers are thinned, diced, mounted onto leadframes or organic substrates, wire-bonded or flip-chip bumped, encapsulated in epoxy mold compound, and subjected to electrical and thermal burn-in testing. Net shippable capacity $K$ across $W_{\text{starts}}$ wafer starts is:
   $$K = W_{\text{starts}} \times \text{DPW} \times Y_{\text{sort}} \times Y_{\text{assembly}} \times Y_{\text{final\_test}}$$
   Because $K$ is strictly bounded by available lithography capacity and packaging lines, it serves as the hard constraint for all downstream allocation policies:
   $$\sum_{i=1}^N x_i \le K$$

### Step 2: Demand Signal Sanitization and De-Biasing

Before applying any allocation formula, the raw order book must be sanitized. Raw purchase orders during a shortage reflect defensive gaming rather than genuine end-system consumption.

1. **Calculate Customer Inventory Days-on-Hand (DOI):**
   Using Electronic Data Interchange (EDI 852 / 867) feeds, query the customer's on-hand component inventory $I_i$ and their average daily production consumption rate $c_i$:
   $$\text{DOI}_i = \frac{I_i}{c_i}$$
   If $\text{DOI}_i > 45 \text{ days}$ during an acute industry-wide shortage, the customer is hoarding inventory. Their submitted order must be flagged for de-biasing.

2. **Reconcile Against End-Device Activation Telemetry:**
   For telecom infrastructure and technology hardware, genuine demand is bounded by downstream deployment velocity:
   - Telecom: Radio units cannot be deployed faster than base station tower site acquisition, civil zoning permits, and structural engineering approvals.
   - Enterprise Hardware: Cloud servers cannot be commissioned faster than data center power rack allocations and cooling facility build-outs.
   - Consumer Electronics: Final assembly is bounded by point-of-sale sell-through and cellular network IMEI / MAC address activations.

3. **Apply Bayesian Shrinkage De-Biasing:**
   Let $O_{i,t}$ be the unconstrained order submitted by customer $i$ for period $t$. Let $\mu_{i,t}$ be the historical baseline consumption rate estimated from trailing actuals, and let $\sigma_{i,\text{game}}^2$ represent historical order volatility. The sanitized demand $\hat{D}_{i,t}$ is computed via shrinkage:
   $$\hat{D}_{i,t} = \lambda_i O_{i,t} + (1 - \lambda_i) \mu_{i,t}$$
   where the shrinkage weight $\lambda_i \in [0, 1]$ is parameterized by customer forecast credibility:
   $$\lambda_i = \frac{\sigma_0^2}{\sigma_0^2 + \sigma_{i,\text{game}}^2}$$
   When a customer has a documented history of placing volatile orders followed by cancellations, $\sigma_{i,\text{game}}^2 \to \infty$, driving $\lambda_i \to 0$. The allocation model ignores their submitted order $O_{i,t}$ and plans strictly against verified baseline consumption $\mu_{i,t}$.

### Step 3: Game-Theoretic Formalization of the Allocation Game

To understand why simple allocation rules fail, we formalize the buyer's decision problem using the classic Cachon-Lariviere rationing framework.

Let $N$ downstream buyers compete for fixed capacity $K$. Each buyer $i$ faces true end-customer demand $D_i$. Buyer $i$ submits an order $O_i$ to the chipmaker. The total submitted order book is:
$$O_{\text{total}} = \sum_{j=1}^N O_j$$
If $O_{\text{total}} > K$, capacity is binding, and the chipmaker applies an allocation rule $A_i(O_i, O_{-i})$ to determine physical delivery.

#### Proportional Allocation Rule Mechanics
Under proportional allocation, each buyer receives a fraction of capacity equal to their share of total orders:
$$A_i(O_i, O_{-i}) = K \cdot \frac{O_i}{\sum_{j=1}^N O_j}$$
Buyer $i$'s financial payoff $\Pi_i(O_i, O_{-i})$ is governed by the revenue generated from satisfied demand minus the inventory holding or disposal cost of overage, and the penalty for unfulfilled demand:
$$\Pi_i(O_i, O_{-i}) = p_i \min\left(D_i, A_i\right) - w_i A_i - h_i \max\left(0, A_i - D_i\right) - \pi_i \max\left(0, D_i - A_i\right)$$
where $p_i$ is unit selling price, $w_i$ is wholesale purchase cost, $h_i$ is unit overage/holding cost, and $\pi_i$ is the unit shortage penalty (lost goodwill or SLA liquidated damages).

#### Proof of the Double Ordering Nash Equilibrium
To find buyer $i$'s optimal order $O_i^*$, differentiate allocation $A_i$ with respect to $O_i$:
$$\frac{\partial A_i}{\partial O_i} = K \cdot \frac{\left(\sum_{j=1}^N O_j\right) - O_i}{\left(\sum_{j=1}^N O_j\right)^2} = K \cdot \frac{\sum_{j \neq i} O_j}{\left(O_i + \sum_{j \neq i} O_j\right)^2} > 0$$
Notice that $\frac{\partial A_i}{\partial O_i}$ is strictly positive for all finite order volumes as long as competitors submit non-zero orders ($\sum_{j \neq i} O_j > 0$). An increase in $O_i$ directly increases buyer $i$'s allocation by diluting the allocation share of all other competitors.

Now consider the buyer's cost of ordering excess units. If the chipmaker allows free order cancellations or does not enforce binding take-or-pay penalties, the overage penalty is negligible ($h_i \to 0$). 
Whenever $A_i < D_i$, the marginal profit of receiving an additional unit is:
$$\frac{\partial \Pi_i}{\partial A_i} = (p_i - w_i) + \pi_i > 0$$
Applying the chain rule:
$$\frac{\partial \Pi_i}{\partial O_i} = \frac{\partial \Pi_i}{\partial A_i} \cdot \frac{\partial A_i}{\partial O_i} > 0$$
Because the derivative is strictly positive, buyer $i$'s payoff is strictly increasing in $O_i$ until $A_i = D_i$. 

To secure $A_i = D_i$ under proportional rationing when competitors submit orders $O_{-i}$, buyer $i$ must set:
$$K \cdot \frac{O_i}{O_i + \sum_{j \neq i} O_j} = D_i \implies O_i^* = \frac{D_i \sum_{j \neq i} O_j}{K - D_i}$$
If all symmetric buyers attempt to secure their full true demand $D$ when total demand exceeds capacity ($\sum D_i > K$), the denominator approaches zero or becomes negative. The resulting unique symmetric Nash equilibrium is:
$$O_i^* \to \infty$$
In commercial practice, buyers do not order infinite units; their orders expand until they hit artificial constraints: their maximum corporate credit line, ERP system order validation caps, or vendor-imposed order limits. Thus, **proportional allocation mathematically guarantees order inflation and destroys the informational integrity of the order book.**

```text
+---------------------------------------------------------------------------------------------------+
|                        THE INCENTIVE STRUCTURE OF ALLOCATION RULES                                |
+---------------------------------------------------------------------------------------------------+
| Rule Type       | Allocation Formula             | Derivative: \partial A_i / \partial O_i | Gaming Incentive     |
+-----------------+--------------------------------+----------------------------------------+----------------------+
| Proportional    | A_i = K * (O_i / \sum O_j)     | > 0 (Strictly Positive)                | Extreme (Order -> \infty) |
| Uniform         | A_i = min(O_i, \bar{q})        | = 0 for O_i > \bar{q}                  | Weak (Order to \bar{q})  |
| Turn-and-Earn   | A_i = K * (S_i / \sum S_j)     | = 0 (Identically Zero)                 | None (Truth-Inducing)|
| Capacity Dep.   | A_i = f(Deposits, Contracts)   | Bound by Financial Escrow              | None (Risk-Bearing)  |
+---------------------------------------------------------------------------------------------------+
```

### Step 4: Formulation of Truth-Inducing Allocation Mechanisms

To eliminate double ordering, the chipmaker must implement allocation rules that eliminate the marginal gain from order inflation.

#### 1. The Turn-and-Earn Mechanism
Under turn-and-earn, allocation is decoupled from current forward orders $O_i$ and indexed strictly to verified historical consumption velocity $S_i$ over a defined trailing baseline window (e.g., the four quarters preceding the shortage):
$$A_i = K \cdot \frac{S_i}{\sum_{j=1}^N S_j}$$
Evaluate the gaming derivative under turn-and-earn:
$$\frac{\partial A_i}{\partial O_i} = 0$$
Because submitted order volume $O_i$ does not appear in the allocation formula, inflating forward orders yields zero additional components. Buyers submit orders that reflect their realistic operating requirements, restoring transparency to the order backlog.

#### 2. Uniform Rationing (Sprumont Mechanism)
Under uniform rationing, capacity is distributed equally among all claimants, subject to the condition that no claimant receives more than their requested order:
$$A_i = \min(O_i, \bar{q})$$
where the rationing threshold $\bar{q}$ is the unique scalar satisfying:
$$\sum_{i=1}^N \min(O_i, \bar{q}) = K$$
Uniform rationing is strategy-proof against upward order inflation for claimants whose demand exceeds $\bar{q}$. Submitting an order larger than $\bar{q}$ does not increase allocation beyond $\bar{q}$.

#### 3. Capacity Reservation with Financial Escrow
The chipmaker requires buyers to purchase capacity options before wafer starts are committed:
- **Reservation Fee ($r$):** Non-refundable upfront fee per unit of fab capacity reserved, payable at $t - 26 \text{ weeks}$.
- **Execution Price ($w$):** Purchase price per physical unit delivered upon completion.
If a buyer inflates their reservation, they incur a guaranteed cash loss of $r \cdot (O_i - D_i)$. This shifts financial risk back to the buyer, inducing truthful revelation of demand distributions.

### Step 5: Lexicographic Priority and Tiered Fair-Share Allocation

In high-reliability hardware supply chains (telecommunications carrier networks, aerospace avionics, medical life-support systems), treating all demand homogeneously is legally and commercially unacceptable. The allocation engine must execute **lexicographic priority sorting**:

1. **Tier 1: Life-Safety, Critical Infrastructure, and Binding Legal SLAs**
   Components required for emergency communications (E-911), air traffic management avionics, and contracts carrying severe liquidated damages clauses. These allocations are satisfied first up to their minimum legal threshold:
   $$x_{i,\text{Tier 1}} = \min(D_i, \text{SLA}_i^{\min})$$

2. **Tier 2: Core Strategic Partnerships**
   Key strategic accounts governed by multi-year joint technology development agreements. These accounts receive allocations based on a weighted fair-share policy:
   $$x_{i,\text{Tier 2}} = \min\left(D_i - x_{i,\text{Tier 1}}, (K - K_{\text{Tier 1}}) \cdot \frac{w_i S_i}{\sum_{j \in \text{Tier 2}} w_j S_j}\right)$$

3. **Tier 3: Standard Transactional and Run-Rate Accounts**
   Transactional buyers receiving residual capacity:
   $$x_{i,\text{Tier 3}} = \max\left(0, \min\left(D_i, K_{\text{residual}} \cdot \frac{S_i}{\sum_{j \in \text{Tier 3}} S_j}\right)\right)$$

4. **Tier 4: Spot Market and Broker Arbitrage Demands**
   Uncontracted or speculative orders. Under binding shortage, Tier 4 allocation is set strictly to zero ($x_{\text{Tier 4}} = 0$) to prevent broker hoarding.

```text
+---------------------------------------------------------------------------------------------------+
|                               LEXICOGRAPHIC ALLOCATION WATERFALL                                  |
+---------------------------------------------------------------------------------------------------+
  Total Shippable Capacity: K
    │
    ├─► [Priority 1: Mandatory Contractual SLAs & Life-Safety Commitments]
    │     ├── Allocation: x_i = min(D_i, SLA_min_i)
    │     └── Residual Capacity: K_rem1 = max(0, K - \sum x_i,Tier1)
    │
    └─► [Priority 2: Strategic Long-Term Core Accounts]
          ├── Allocation: x_i = min(D_i - x_i,Tier1, K_rem1 * (w_i S_i / \sum w_j S_j))
          └── Residual Capacity: K_rem2 = max(0, K_rem1 - \sum x_i,Tier2)
                │
                └─► [Priority 3: Standard Run-Rate Accounts (Equity Floor Constrained)]
                      ├── Allocation: x_i = min(D_i, K_rem2 * (S_i / \sum S_j))
                      └── Residual Capacity: K_rem3 = max(0, K_rem2 - \sum x_i,Tier3)
                            │
                            └─► [Priority 4: Spot / Speculative (Rationed to Zero)]
                                  └── Allocation = 0
+---------------------------------------------------------------------------------------------------+
```

### Step 6: Multi-Objective LP/MILP Mathematical Formulation

When multiple constrained components must be allocated across diverse product lines assembled for multiple customer tiers across rolling time periods, the problem must be solved using Mixed-Integer Linear Programming (MILP).

#### Indices and Sets
- $c \in \mathcal{C}$: Constrained component part numbers (e.g., Baseband ASIC, PMIC, RF Transceiver).
- $p \in \mathcal{P}$: Finished hardware products (e.g., 5G Massive MIMO Radio Unit, Enterprise Core Router, Satellite Gateway).
- $i \in \mathcal{I}$: Downstream customers / accounts.
- $t \in \mathcal{T}$: Planning time buckets (weeks or months).

#### Parameters
- $K_{c,t}$: Usable shippable capacity of component $c$ in period $t$.
- $B_{c,p}$: Bill of Materials (BOM) usage coefficient (units of component $c$ required per unit of product $p$).
- $D_{p,i,t}$: Sanitized demand for product $p$ from customer $i$ in period $t$.
- $m_{p,i,t}$: Net unit contribution margin of product $p$ sold to customer $i$ in period $t$.
- $w_i$: Strategic commercial weight of customer $i$.
- $\text{SLA}_{p,i,t}^{\min}$: Minimum contractual volume commitment under legal Master Service Agreement.
- $\pi_{p,i,t}$: Contractual liquidated damages penalty per unit of SLA shortfall.
- $\alpha_i^{\text{floor}}$: Minimum equity fill-rate fraction required to prevent catastrophic line-down at customer $i$.
- $\text{SPQ}_p$: Standard pack quantity / minimum manufacturing batch size for product $p$.

#### Decision Variables
- $x_{p,i,t} \ge 0$: Units of product $p$ allocated and shipped to customer $i$ in period $t$.
- $s_{p,i,t} \ge 0$: Contractual SLA shortfall units for product $p$ and customer $i$ in period $t$.
- $y_{p,i,t} \in \mathbb{Z}_{\ge 0}$: Integer number of standard packaging batches produced.

#### Objective Function
Maximize total strategic contribution margin net of contractual SLA liquidated damages penalties:
$$\max Z = \sum_{t \in \mathcal{T}} \sum_{i \in \mathcal{I}} \sum_{p \in \mathcal{P}} \left( w_i \cdot m_{p,i,t} \cdot x_{p,i,t} - \pi_{p,i,t} \cdot s_{p,i,t} \right)$$

#### Constraints
1. **Component Capacity Bounds:** Total consumption of scarce component $c$ across all products and customers cannot exceed physical supply:
   $$\sum_{i \in \mathcal{I}} \sum_{p \in \mathcal{P}} B_{c,p} \cdot x_{p,i,t} \le K_{c,t} \quad \forall c \in \mathcal{C}, \forall t \in \mathcal{T}$$

2. **Demand Upper Bounds:** Allocations cannot exceed de-biased demand:
   $$x_{p,i,t} \le \hat{D}_{p,i,t} \quad \forall p \in \mathcal{P}, \forall i \in \mathcal{I}, \forall t \in \mathcal{T}$$

3. **Contractual SLA Shortfall Tracking:** Shortfall $s_{p,i,t}$ measures unfulfilled SLA commitments:
   $$x_{p,i,t} + s_{p,i,t} \ge \text{SLA}_{p,i,t}^{\min} \quad \forall p \in \mathcal{P}, \forall i \in \mathcal{I}, \forall t \in \mathcal{T}$$

4. **Customer Equity Floors (Anti-Starvation Constraints):** Every active customer receives at least a minimal baseline allocation to preserve assembly tooling and prevent complete shutdown:
   $$x_{p,i,t} \ge \alpha_i^{\text{floor}} \cdot \hat{D}_{p,i,t} \quad \forall p \in \mathcal{P}, \forall i \in \mathcal{I}, \forall t \in \mathcal{T}$$

5. **Packaging Integrality Constraints:** Finished goods must be produced in discrete packaging increments:
   $$x_{p,i,t} = y_{p,i,t} \cdot \text{SPQ}_p, \quad y_{p,i,t} \in \mathbb{Z}_{\ge 0} \quad \forall p \in \mathcal{P}, \forall i \in \mathcal{I}, \forall t \in \mathcal{T}$$

### Step 7: Commercial Binding and NCNR Enforcement

The operationalization of optimized allocations requires immediate contractual lock-in:
1. **Transmit Binding Allocation Notices:** Within 48 hours of model execution, issue formal EDI 855 (Purchase Order Acknowledgment) transactions establishing firm scheduled delivery dates.
2. **Execute NCNR Addenda:** All allocated units must be converted into Non-Cancellable, Non-Returnable agreements. If a customer attempts to cancel an allocated order, they remain legally obligated to pay 100% of the invoice value.
3. **Escrow Cash Deposits:** Require tier-3 and tier-4 customers to post 20% to 50% non-refundable cash deposits or open confirmed, irrevocable letters of credit (LCs). This eliminates speculative phantom buyers immediately.

---

## 6. Worked example

To understand the mathematical and operational differences between allocation mechanisms, we examine a complete end-to-end allocation decision for an enterprise technology chipmaker facing a severe capacity shortfall on an advanced 7nm baseband processing ASIC.

### Problem Context and Operational Parameters
- **Component:** 7nm Telecom Baseband Processing IC (Part Number: `BB-7090`).
- **Planning Horizon:** Q3 Production Window (13 Weeks).
- **Physical Capacity:** Front-end wafer starts and back-end packaging yield a total net shippable volume of:
  $$K = 500{,}000 \text{ units}$$
- **Customer Landscape:** Three major technology hardware OEMs submit purchase orders.

The customer demand, contract terms, unit economics, and historical consumption data are detailed in the following table:

| Customer Profile | Market Sub-Domain | True Demand ($D_i$) | Contractual SLA Floor ($\text{SLA}_i^{\min}$) | SLA Shortfall Penalty ($\pi_i$) | Unit Contribution Margin ($m_i$) | Strategic Weight ($w_i$) | Historic Baseline Run-Rate ($S_i$) |
|---|---|---:|---:|---:|---:|---:|---:|
| **Customer Alpha** | Tier-1 Hyperscale Telecom OEM | 300,000 units | 150,000 units | \$35 / unit | \$85 / unit | 1.50 | 220,000 units |
| **Customer Beta** | Global High-End Smartphone OEM | 400,000 units | 100,000 units | \$20 / unit | \$60 / unit | 1.20 | 200,000 units |
| **Customer Gamma** | Industrial IoT & Grid Automation | 300,000 units | 0 units | \$0 / unit | \$45 / unit | 0.80 | 80,000 units |
| **Total** | | **1,000,000 units** | **250,000 units** | — | — | — | **500,000 units** |

### Market Condition
$$\sum_{i=1}^3 D_i = 1{,}000{,}000 \text{ units} > K = 500{,}000 \text{ units}$$
Total demand exceeds available shippable capacity by 2:1 (aggregate allocation factor $\alpha = 0.50$).

We evaluate the system performance under four explicit allocation policies:
1. **Policy 1: Proportional to Order (Naive Linear Rationing)**
2. **Policy 2: Turn-and-Earn (Indexed to Historical Consumption)**
3. **Policy 3: Lexicographic Priority Tiering (SLA First, Then Equal Sharing)**
4. **Policy 4: Multi-Objective LP Optimization (Margin, Strategic Weights, SLA Penalties, 15% Equity Floor)**

---

### Policy 1: Proportional to Order (Naive Linear Rationing)

Assuming customers submit un-gamed orders equal to their initial demand ($O_i = D_i$):
$$\alpha = \frac{K}{\sum D_i} = \frac{500{,}000}{1{,}000{,}000} = 0.50$$
Each customer receives exactly 50% of their submitted order:
- **Alpha:** $A_{\text{Alpha}} = 300{,}000 \times 0.50 = 150{,}000 \text{ units}$
- **Beta:** $A_{\text{Beta}} = 400{,}000 \times 0.50 = 200{,}000 \text{ units}$
- **Gamma:** $A_{\text{Gamma}} = 300{,}000 \times 0.50 = 150{,}000 \text{ units}$

#### Financial and Operational Audit:
- **Contractual SLA Performance:**
  - Alpha SLA Shortfall: $\max(0, 150{,}000 - 150{,}000) = 0 \text{ units}$. Penalty: \$0.
  - Beta SLA Shortfall: $\max(0, 100{,}000 - 200{,}000) = 0 \text{ units}$. Penalty: \$0.
  - Gamma SLA Shortfall: 0 units. Penalty: \$0.
- **Gross Contribution Margin:**
  $$\text{Margin} = (150{,}000 \times \$85) + (200{,}000 \times \$60) + (150{,}000 \times \$45) = \$12{,}750{,}000 + \$12{,}000{,}000 + \$6{,}750{,}000 = \$31{,}500{,}000$$
- **Net Profit Realized:**
  $$\text{Net Profit} = \$31{,}500{,}000 - \$0 = \$31{,}500{,}000$$

#### The Failure Mode (Gaming Reaction):
Customer Gamma, an industrial IoT maker with low margins and zero SLA leverage, receives 150,000 units—vastly exceeding their historical quarterly run-rate of 80,000 units. Meanwhile, Customer Alpha, a critical national telecom infrastructure builder, receives only 150,000 units against a demand of 300,000 units, delaying nationwide cellular deployments. 

Observing this rule, Customer Beta doubles their order next period to $O_{\text{Beta}} = 800{,}000$, and Gamma inflates to $O_{\text{Gamma}} = 600{,}000$. The order book expands to 1.7 million units, crashing the allocation factor to $\alpha = 0.29$ and diluting Customer Alpha's allocation down to 88,235 units, which severely breaches Alpha's SLA and triggers massive liquidated damages penalties.

---

### Policy 2: Turn-and-Earn (Historical Run-Rate Baseline)

To neutralize gaming, the chipmaker allocates capacity based on verified historical consumption $S_i$ over the prior four quarters.
Total historical baseline:
$$\sum_{j=1}^3 S_j = 220{,}000 + 200{,}000 + 80{,}000 = 500{,}000 \text{ units}$$
Because total capacity $K = 500{,}000$ exactly matches total baseline consumption $\sum S_j$, the turn-and-earn ratio is:
$$\beta = \frac{K}{\sum S_j} = \frac{500{,}000}{500{,}000} = 1.00$$
Each customer receives 100% of their historical baseline:
- **Alpha:** $A_{\text{Alpha}} = 220{,}000 \times 1.00 = 220{,}000 \text{ units}$
- **Beta:** $A_{\text{Beta}} = 200{,}000 \times 1.00 = 200{,}000 \text{ units}$
- **Gamma:** $A_{\text{Gamma}} = 80{,}000 \times 1.00 = 80{,}000 \text{ units}$

#### Financial and Operational Audit:
- **Contractual SLA Performance:**
  - Alpha SLA Shortfall: $\max(0, 150{,}000 - 220{,}000) = 0 \text{ units}$. Penalty: \$0.
  - Beta SLA Shortfall: $\max(0, 100{,}000 - 200{,}000) = 0 \text{ units}$. Penalty: \$0.
  - Gamma SLA Shortfall: 0 units. Penalty: \$0.
- **Gross Contribution Margin:**
  $$\text{Margin} = (220{,}000 \times \$85) + (200{,}000 \times \$60) + (80{,}000 \times \$45) = \$18{,}700{,}000 + \$12{,}000{,}000 + \$3{,}600{,}000 = \$34{,}300{,}000$$
- **Net Profit Realized:**
  $$\text{Net Profit} = \$34{,}300{,}000 - \$0 = \$34{,}300{,}000$$

#### Strategic Assessment:
Gross margin increases by **\$2,800,000** (+8.9%) compared to naive proportional allocation. Customer Alpha receives 220,000 units, supporting their major telecom infrastructure rollouts. Customer Gamma receives their true historical run-rate of 80,000 units, completely eliminating speculative inventory hoarding. Because $\partial A_i / \partial O_i = 0$, no customer has any incentive to inflate their forward purchase orders.

---

### Policy 3: Lexicographic Priority Tiering (SLA First, Then Equal Sharing)

The chipmaker fulfills binding legal SLA contracts first, and then distributes residual capacity equally among claimants.

1. **Step 1: Fulfill Minimum SLA Commitments:**
   - Alpha SLA: 150,000 units
   - Beta SLA: 100,000 units
   - Gamma SLA: 0 units
   - Total SLA Capacity Consumed: $150{,}000 + 100{,}000 + 0 = 250{,}000 \text{ units}$
   - Residual Shippable Capacity:
     $$K_{\text{residual}} = 500{,}000 - 250{,}000 = 250{,}000 \text{ units}$$

2. **Step 2: Allocate Residual Capacity Equally Across Remaining Unfulfilled Demand:**
   Remaining unfulfilled demand:
   - Alpha: $D_{\text{Alpha}} - 150{,}000 = 300{,}000 - 150{,}000 = 150{,}000 \text{ units}$
   - Beta: $D_{\text{Beta}} - 100{,}000 = 400{,}000 - 100{,}000 = 300{,}000 \text{ units}$
   - Gamma: $D_{\text{Gamma}} - 0 = 300{,}000 \text{ units}$
   - Total Residual Demand: $150{,}000 + 300{,}000 + 300{,}000 = 750{,}000 \text{ units}$

   Equal share of residual capacity:
   $$\text{Residual Allocation Ratio} = \frac{250{,}000}{750{,}000} = \frac{1}{3}$$
   - Alpha Residual: $150{,}000 \times \frac{1}{3} = 50{,}000 \text{ units}$
   - Beta Residual: $300{,}000 \times \frac{1}{3} = 100{,}000 \text{ units}$
   - Gamma Residual: $300{,}000 \times \frac{1}{3} = 100{,}000 \text{ units}$

3. **Step 3: Total Allocated Quantities:**
   - Alpha: $150{,}000 + 50{,}000 = 200{,}000 \text{ units}$
   - Beta: $100{,}000 + 100{,}000 = 200{,}000 \text{ units}$
   - Gamma: $0 + 100{,}000 = 100{,}000 \text{ units}$
   - Total Allocated: $200{,}000 + 200{,}000 + 100{,}000 = 500{,}000 \text{ units}$

#### Financial and Operational Audit:
- **Gross Contribution Margin:**
  $$\text{Margin} = (200{,}000 \times \$85) + (200{,}000 \times \$60) + (100{,}000 \times \$45) = \$17{,}000{,}000 + \$12{,}000{,}000 + \$4{,}500{,}000 = \$33{,}500{,}000$$
- **Net Profit Realized:**
  $$\text{Net Profit} = \$33{,}500{,}000 - \$0 = \$33{,}500{,}000$$

---

### Policy 4: Multi-Objective LP Optimization (With 15% Anti-Starvation Equity Floor)

The chipmaker solves the formal mathematical optimization model.
- **Objective:** Maximize $\sum (w_i \cdot m_i \cdot x_i) - \sum (\pi_i \cdot s_i)$
- **Strategic Margin Values ($w_i \cdot m_i$):**
  - Alpha: $1.50 \times \$85 = \$127.50 / \text{unit}$
  - Beta: $1.20 \times \$60 = \$72.00 / \text{unit}$
  - Gamma: $0.80 \times \$45 = \$36.00 / \text{unit}$
- **Shortfall Penalties:** $\pi_{\text{Alpha}} = \$35$, $\pi_{\text{Beta}} = \$20$, $\pi_{\text{Gamma}} = \$0$
- **Anti-Starvation Equity Floor:** Minimum 15% fill rate for all active accounts to prevent complete factory shutdown ($\alpha^{\text{floor}} = 0.15$):
  - Alpha minimum: $300{,}000 \times 0.15 = 45{,}000 \text{ units}$
  - Beta minimum: $400{,}000 \times 0.15 = 60{,}000 \text{ units}$
  - Gamma minimum: $300{,}000 \times 0.15 = 45{,}000 \text{ units}$

#### Optimization Solution Walkthrough:
Because Customer Alpha provides the highest weighted margin (\$127.50/unit) and carries the highest SLA penalty (\$35/unit), the linear programming solver prioritizes Alpha up to their maximum demand limit:
1. **Allocate to Alpha:** $x_{\text{Alpha}} = 300{,}000 \text{ units}$ (100% fill rate).
   - Capacity consumed: 300,000 units.
   - Remaining capacity: $500{,}000 - 300{,}000 = 200{,}000 \text{ units}$.
2. **Satisfy Gamma's Minimum Equity Floor:**
   - To prevent line-down at Gamma: $x_{\text{Gamma}} = 45{,}000 \text{ units}$ (15% fill rate).
   - Remaining capacity: $200{,}000 - 45{,}000 = 155{,}000 \text{ units}$.
3. **Allocate Residual Capacity to Beta:**
   - Customer Beta has the second highest weighted margin (\$72.00/unit) and an SLA commitment of 100,000 units.
   - Remaining capacity (155,000 units) strictly exceeds Beta's SLA floor (100,000 units) and Beta's equity floor (60,000 units).
   - Allocate all residual capacity to Beta: $x_{\text{Beta}} = 155{,}000 \text{ units}$ ($155{,}000 / 400{,}000 = 38.75\%$ fill rate).

#### Financial and Operational Audit:
- **Contractual SLA Shortfalls:**
  - Alpha SLA Shortfall: $\max(0, 150{,}000 - 300{,}000) = 0 \text{ units}$. Penalty: \$0.
  - Beta SLA Shortfall: $\max(0, 100{,}000 - 155{,}000) = 0 \text{ units}$. Penalty: \$0.
  - Gamma SLA Shortfall: 0 units. Penalty: \$0.
- **Gross Contribution Margin:**
  $$\text{Margin} = (300{,}000 \times \$85) + (155{,}000 \times \$60) + (45{,}000 \times \$45) = \$25{,}500{,}000 + \$9{,}300{,}000 + \$2{,}025{,}000 = \$36{,}825{,}000$$
- **Net Profit Realized:**
  $$\text{Net Profit} = \$36{,}825{,}000 - \$0 = \$36{,}825{,}000$$

---

### Comparative Policy Summary and Trade-Off Ledger

The following comparative table summarizes the financial, strategic, and operational outcomes across all four allocation rules:

| Metric | Policy 1: Proportional | Policy 2: Turn-and-Earn | Policy 3: Lexicographic | Policy 4: Multi-Objective LP |
|---|---:|---:|---:|---:|
| **Alpha Allocation** (Demand: 300k) | 150,000 units (50.0%) | 220,000 units (73.3%) | 200,000 units (66.7%) | **300,000 units (100.0%)** |
| **Beta Allocation** (Demand: 400k) | 200,000 units (50.0%) | 200,000 units (50.0%) | 200,000 units (50.0%) | **155,000 units (38.8%)** |
| **Gamma Allocation** (Demand: 300k) | 150,000 units (50.0%) | 80,000 units (26.7%) | 100,000 units (33.3%) | **45,000 units (15.0%)** |
| **SLA Violations / Penalties** | \$0 | \$0 | \$0 | **\$0** |
| **Realized Gross Contribution Margin** | \$31,500,000 | \$34,300,000 | \$33,500,000 | **\$36,825,000** |
| **Margin Advantage vs Naive Policy** | Baseline (\$0) | +\$2,800,000 (+8.9%) | +\$2,000,000 (+6.3%) | **+\$5,325,000 (+16.9%)** |
| **Susceptibility to Order Gaming** | Extreme ($\partial A/\partial O > 0$) | Zero ($\partial A/\partial O = 0$) | Moderate | Low (Constrained by $\hat{D}$) |
| **Anti-Starvation Line-Down Protection**| High (50% fill) | Moderate (Run-rate) | Moderate (33% fill) | Controlled (15% floor) |

Policy 4 captures an additional **\$5,325,000** in contribution margin compared to naive proportional allocation while fully respecting contractual SLAs and preserving Customer Gamma's operational viability with a 15% anti-starvation baseline.

---

## 7. Code, query, or calculation

The following self-contained Python script implements the Multi-Objective Component Allocation Linear Program (Policy 4) using `scipy.optimize.linprog`. The script solves the exact allocation problem formulated in Section 6, enforcing component capacity bounds, contractual SLA shortfall penalties, and minimum customer fairness floors.

```python
"""Multi-Objective Component Allocation Linear Program under Binding Capacity."""
import numpy as np
from scipy.optimize import linprog

# Problem Parameters
customers = ["Alpha (Telecom)", "Beta (Smartphone)", "Gamma (IoT)"]
demand = np.array([300000.0, 400000.0, 300000.0])  # Units
margin = np.array([85.0, 60.0, 45.0])              # $/unit contribution margin
sla_min = np.array([150000.0, 100000.0, 0.0])       # Contractual minimum SLA units
sla_penalty = np.array([35.0, 20.0, 0.0])          # Shortfall liquidated damages $/unit
strategic_weight = np.array([1.5, 1.2, 0.8])       # Strategic account multiplier
capacity = 500000.0                                # Net shippable capacity bound K
fairness_pct = 0.15                                # Minimum fill-rate anti-starvation floor

# Decision Vector: z = [x0, x1, x2, s0, s1, s2]
# x_i: units allocated to customer i; s_i: SLA shortfall for customer i
# Objective: Maximize sum(weight_i * margin_i * x_i) - sum(penalty_i * s_i)
# linprog minimizes c^T * z ==> invert signs for maximization
c = np.concatenate([-(margin * strategic_weight), sla_penalty])

# Inequality Constraints: A_ub * z <= b_ub
A_ub = []
b_ub = []

# 1. Physical Capacity Constraint: x0 + x1 + x2 <= capacity
A_ub.append([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
b_ub.append(capacity)

# 2. SLA Shortfall Tracking: sla_min_i - x_i <= s_i  ==>  -x_i - s_i <= -sla_min_i
for i in range(3):
    row = [0.0] * 6
    row[i] = -1.0
    row[3 + i] = -1.0
    A_ub.append(row)
    b_ub.append(-sla_min[i])

# Variable Bounds:
# Allocation x_i: [fairness_pct * demand_i, demand_i]
# Shortfall s_i:  [0.0, sla_min_i]
bounds = [(fairness_pct * demand[i], demand[i]) for i in range(3)] + [
    (0.0, sla_min[i]) for i in range(3)
]

# Solve using HiGHS interior point / simplex solver
res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

if not res.success:
    raise RuntimeError(f"Optimization failed: {res.message}")

allocations = res.x[:3]
shortfalls = res.x[3:]
realized_margin = np.sum(margin * allocations)
total_penalties = np.sum(sla_penalty * shortfalls)
net_profit = realized_margin - total_penalties

print("=== OPTIMAL MULTI-OBJECTIVE COMPONENT ALLOCATION SCHEDULE ===")
for i, name in enumerate(customers):
    fill = (allocations[i] / demand[i]) * 100.0
    print(f"{name:18s} | Allocated: {allocations[i]:>10,.0f} u | Fill: {fill:>5.1f}% | "
          f"SLA Shortfall: {shortfalls[i]:>7,.0f} u")
print("-" * 65)
print(f"Total Allocated Quantity: {np.sum(allocations):>10,.0f} / {capacity:,.0f} units")
print(f"Gross Realized Margin:    ${realized_margin:>13,.2f}")
print(f"Total SLA Penalties:      ${total_penalties:>13,.2f}")
print(f"Net Economic Profit:      ${net_profit:>13,.2f}")
```

### Execution Output and Validation
Running this script yields the exact optimal solution derived analytically in Section 6:
```text
=== OPTIMAL MULTI-OBJECTIVE COMPONENT ALLOCATION SCHEDULE ===
Alpha (Telecom)    | Allocated:    300,000 u | Fill: 100.0% | SLA Shortfall:       0 u
Beta (Smartphone)  | Allocated:    155,000 u | Fill:  38.8% | SLA Shortfall:       0 u
Gamma (IoT)        | Allocated:     45,000 u | Fill:  15.0% | SLA Shortfall:       0 u
-----------------------------------------------------------------
Total Allocated Quantity:    500,000 / 500,000 units
Gross Realized Margin:    $36,825,000.00
Total SLA Penalties:      $         0.00
Net Economic Profit:      $36,825,000.00
```
The solver utilizes 100% of available capacity (500,000 units), fulfills Customer Alpha's high-margin demand in full, avoids all contractual SLA penalties, enforces Customer Gamma's 15% anti-starvation baseline, and delivers an exact net profit of \$36,825,000.00.

---

## 8. Data, standards, and regulatory context

Component allocation is not merely an internal engineering or mathematical exercise; it is strictly constrained by commercial law, antitrust regulations, and global supply chain data interchange standards.

### Commercial Law: Uniform Commercial Code (UCC) § 2-615
In the United States and jurisdictions adopting the UCC, when unforeseen supervening events (such as raw material embargoes, cleanroom chemical contamination, or natural disasters) render full performance commercially impracticable, the seller's obligation is governed by **UCC Section 2-615 (Excuse by Failure of Presupposed Conditions)**.

Key statutory mandates include:
1. **The Allocation Requirement (UCC § 2-615(b)):** The seller must allocate production and deliveries among its customers. Crucially, the statute dictates that the allocation must be **fair and reasonable**.
2. **Inclusion of Non-Contracted Regular Customers:** The seller may include regular customers not currently under firm contract, as well as its own internal downstream manufacturing requirements, in the allocation pool.
3. **Prohibition of Arbitrary Favoritism:** A supplier cannot allocate 100% of supply to an affiliated subsidiary while completely shutting off an independent buyer without exposing itself to massive breach-of-contract litigation.
4. **Mandatory Formal Notice (UCC § 2-615(c)):** The seller must seasonably notify all buyers that there will be delay or non-delivery, providing an estimated quota made available for each customer.

### Antitrust and Competition Law Constraints
- **The Robinson-Patman Act (15 U.S.C. § 13):** Prohibits sellers from discriminating in price or promotional services between different purchasers of commodities of like grade and quality where the effect may be substantially to lessen competition. While the statute primarily targets price discrimination, predatory allocation schemes designed to drive rival device manufacturers out of business invite federal antitrust scrutiny.
- **Article 102 of the Treaty on the Functioning of the European Union (TFEU):** Dominant semiconductor fabricators or IDMs with significant market power (e.g., controlling a proprietary cellular modem or GPU standard) face strict liability for "abuse of dominance." A refusal to supply or a discriminatory allocation policy that starves European industrial OEMs while favoring domestic champions can trigger multi-billion-euro antitrust penalties from the European Commission.

### Data Interchange Standards: EDI and RosettaNet
Component allocation workflows rely on standardized machine-to-machine data communication protocols:

```text
+---------------------------------------------------------------------------------------------------+
|                           EDI & ROSETTANET ALLOCATION MESSAGE FLOW                                |
+---------------------------------------------------------------------------------------------------+
  Customer ERP / MRP System                                            Supplier / Allocation Engine
       │                                                                            │
       ├──► EDI 830 / RosettaNet PIP 4A5: Planning Schedule / Forecast Data ────────┼──► (Raw Forecast)
       │                                                                            │
       ├──► EDI 850 / RosettaNet PIP 3A4: Formal Purchase Order Submission ─────────┼──► (Order Book)
       │                                                                            │
       │                                  [Optimization & Rationing Solved]          │
       │                                                                            │
       │◄── EDI 855 / RosettaNet PIP 3A4: PO Acknowledgment (Allocated Quota) ──────┤
       │                                                                            │
       │◄── EDI 865: Purchase Order Change Request / Allocation Schedule Update ────┤
       │                                                                            │
       ├──► EDI 867 / RosettaNet PIP 2A9: Product Transfer & Resale (POS Data) ─────┼──► (Audit Trail)
       │                                                                            │
       │◄── EDI 856: Advanced Shipping Notice (Physical Dispatch) ──────────────────┤
+---------------------------------------------------------------------------------------------------+
```

- **EDI 830 (Planning Schedule with Release Capability):** Used by technology OEMs to share rolling 12-month forward forecasts. During allocation, chipmakers compare EDI 830 forecasts against historical actuals to flag speculative gaming.
- **EDI 850 (Purchase Order) & EDI 855 (PO Acknowledgment):** Downstream buyers submit orders via EDI 850. The chipmaker's allocation engine responds via EDI 855, formally adjusting line-item quantities and scheduled ship dates to reflect the approved allocation quota.
- **EDI 867 (Product Transfer and Resale Report):** Authorized distributors transmit point-of-sale sell-through data to chipmakers, verifying that allocated parts were sold to legitimate end users rather than hoarded.
- **RosettaNet Partner Interface Processes (PIPs):** Standardized XML-based B2B transactions widely deployed across the semiconductor industry:
  - **PIP 3A4:** Request Purchase Order / Confirm Purchase Order.
  - **PIP 4A5:** Plan Schedule Notification (Collaborative forecasting).
  - **PIP 2A9:** Query Product Transfer Status.

### JEDEC and SEMI Standards
- **JEDEC JESD46 / JESD48:** Define standard protocols for Product Change Notification (PCN) and Product Discontinuance (End-of-Life / Last-Time Buy). When a fab line transition forces component allocation, chipmakers must provide 6 months minimum notice before discontinuance.
- **SEMI Standards (SEMI E10 / E79):** Specify equipment reliability, availability, and overall equipment effectiveness (OEE) metrics, governing the calculation of baseline fab capacity availability.

---

## 9. Industry practice and case evidence

Component allocation failures and gaming dynamics have driven some of the most significant balance sheet restructurings and supply chain disruptions in industrial history.

```text
+---------------------------------------------------------------------------------------------------+
|                            HISTORICAL COMPONENT ALLOCATION CRISES                                 |
+---------------------------------------------------------------------------------------------------+
  Year        Event                             Mechanism                         Consequence
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  2001        Cisco Systems Dot-Com Collapse    Proportional allocation gaming    $2.2B inventory
                                                by EMS partners (Celestica,       write-down; 40%
                                                Solectron); double ordering       workforce layoff.
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  2020-2023   Global Semiconductor Shortage     Automotive JIT cancellations;     $210B lost auto
              ("The Chip Crunch")               foundries reallocated to 5G/PC;   revenue; telecom
                                                severe golden screw stoppages.    rollouts delayed.
  ─────────────────────────────────────────────────────────────────────────────────────────────────
  2023-2025   Generative AI GPU Allocation      Advanced packaging (CoWoS) bind;  Strict escrow &
              (H100 / Blackwell Nodes)          cloud providers pay 100% upfront; non-refundable
                                                priority to tier-1 hyperscalers.  prepayments.
+---------------------------------------------------------------------------------------------------+
```

### Case Evidence 1: The 2001 Cisco Systems $2.2 Billion Inventory Write-Down
- **The Context:** In 1999–2000, during the peak of the telecommunications and internet infrastructure boom, Cisco experienced annual revenue growth exceeding 40%. Lead times for optical transceivers, network processors, and high-speed memory modules extended from 6 weeks to over 40 weeks.
- **The Gaming Mechanism:** Cisco outsourced manufacturing to contract EMS providers (Solectron, Celestica, Flextronics). To secure scarce IC allocations from component manufacturers, Cisco and its EMS partners placed duplicate orders across multiple channels. Because chipmakers rationed output proportionally, Cisco's procurement teams inflated order quantities by 200% to 300%. The order backlog swelled to over \$10 billion, leading corporate management to believe that future demand was virtually limitless.
- **The Collapse:** In early 2001, telecommunication service providers abruptly cut capital expenditures. Cisco's actual end-customer orders dropped by 30%. Because the order book consisted primarily of phantom double orders, cancellations cascaded upstream simultaneously. In May 2001, Cisco announced a **\$2.25 billion inventory write-off**—the largest in electronics history at the time—and cut 8,500 jobs. Component suppliers who built fab capacity to satisfy Cisco's phantom backlog suffered multi-year balance sheet distress.

### Case Evidence 2: The 2020–2023 Global Automotive and Telecom "Chip Crunch"
- **The Context:** In Q1 2020, at the onset of the COVID-19 pandemic, global automotive OEMs (e.g., Ford, GM, Volkswagen) cancelled forward wafer orders based on traditional Just-In-Time (JIT) replenishment principles. Pure-play semiconductor foundries (TSMC, UMC, GlobalFoundries) immediately reallocated idle fabrication capacity to consumer electronics, cloud computing, and 5G telecommunication infrastructure chipmakers experiencing surging demand.
- **The Golden Screw Crisis:** When automotive demand rebounded sharply in late 2020, automakers discovered that foundry lead times exceeded 52 weeks. Automakers were placed on strict component allocation. Automotive assembly lines worldwide were idled because they lacked \$1 to \$5 microcontrollers (e.g., Renesas, NXP, Infineon electronic stability control ICs) and power management chips. 
- **The Economic Toll:** The global automotive industry lost an estimated **\$210 billion in revenue** in 2021 alone, producing 9.5 million fewer vehicles than planned. In telecommunications, lead times for optical transport switches and 5G baseband processing modules reached 70 to 90 weeks, forcing network operators to delay nationwide cellular rollouts.
- **What Transferred vs. What Did Not:** Technology companies with sophisticated, truth-inducing allocation systems (e.g., Apple, which financed dedicated fab lines and enforced multi-year NCNR agreements) navigated the crisis with record revenues. Companies that relied on transactional purchasing, spot brokers, and naive proportional allocation suffered massive line-down stoppages.

### Case Evidence 3: Apple's Advanced Node Capacity Reservation Architecture
- **Industry Practice:** Apple maintains a structural advantage in semiconductor allocation by completely bypassing transactional allocation games. For leading-edge process nodes (e.g., TSMC 3nm and 2nm nodes), Apple signs multi-billion-dollar advance **Capacity Reservation Contracts**.
- **Financial Architecture:** Apple prepays billions of dollars to finance TSMC's capital equipment expenditures (purchasing ASML EUV lithography tools on TSMC's behalf) in exchange for exclusive, first-priority rights to 100% of the initial wafer output of new fabrication lines for the first 12 to 18 months of production.
- **Yield Risk Absorption:** In exchange for dedicated capacity, Apple historically absorbed the cost of defective dies during initial fab yield ramps, paying per-wafer rather than per-good-die. This structural alignment eliminates double ordering entirely: capacity is fully funded, fully transparent, and legally locked.

---

## 10. Failure modes and red flags

Operating under component allocation is inherently high-risk. The following risk matrix details the primary failure modes, operational early warning signs, underlying mechanisms, and engineering mitigations:

| Failure Mode | Early Warning Sign | Root Cause / Mechanism | Operational & Financial Consequence | Engineering Mitigation |
|---|---|---|---|---|
| **Phantom Order Backlog Meltdown** | Order backlog expanding at $>3\times$ historical consumption run-rate while end-market growth is flat. | Downstream buyers game proportional allocation by submitting duplicate orders across distributors. | Sudden mass cancellations; massive inventory write-offs (\$100M+); fab overcapacity. | Implement turn-and-earn allocation; require 30% non-refundable cash deposits on forward orders. |
| **Golden Screw Assembly Line Paralysis** | High inventory of 99% of BOM components ($100M+ WIP) accompanied by zero finished goods output. | Failure to coordinate multi-component allocation across complementary BOM parts. | Assembly lines idled; operating cash flow frozen; massive finished goods delivery delays. | Formulate multi-component kitting constraints in allocation LP; synchronize allocations across BOM. |
| **Gray Market Counterfeit Infiltration** | Procurement teams purchasing components from non-authorized independent brokers at $20\times$ MSRP. | Desperate attempts to bypass vendor allocation to avoid line-down stoppages. | Defective, remarked, or cloned chips enter production; catastrophic field reliability failures. | Enforce strict authorized distribution tracing (AS6081 standard); mandate X-ray and decapsulation test. |
| **Legal Challenge Under UCC § 2-615** | Cease-and-desist letters from tier-2 customers alleging discriminatory allocation favoring tier-1 accounts. | Allocating 100% of scarce capacity to high-margin accounts without objective allocation criteria. | Costly commercial litigation; preliminary injunctions freezing warehouse shipments; damages awards. | Document mathematical allocation policies; enforce minimum anti-starvation equity floors ($\ge 15\%$). |
| **Distributor Hoarding and Arbitrage** | Distributor inventory (DOH) surging above 90 days while end-OEM factories are starving. | Authorized distributors withhold allocated inventory from OEM contracts to sell at spot markups. | Artificial worsening of shortage; destruction of end-customer trust; price gouging accusations. | Audit EDI 867 POS data weekly; contractually mandate pass-through pricing with spot-sale audit penalties. |
| **Lead Time Inversion Panic** | Quoted component lead times exceed 52 weeks while customer contract commitments are 12 weeks. | ERP/MRP systems automatically push out purchase orders to match published lead times, creating circular panic. | Circular bullwhip amplification; complete paralysis of standard MRP replenishment engines. | Decouple ERP order generation from published lead times; freeze planning fences; mandate manual allocation. |
| **Customer Engineering Redesign Cascade** | Customers initiating pin-compatible redesigns using secondary silicon vendors. | Protracted component allocation with poor visibility forces customers to abandon platform. | Permanent, irreversible loss of socket design-win; multi-year revenue destruction after shortage ends. | Provide transparent, legally binding allocation schedules; offer pin-compatible drop-in alternative dies. |
| **Contractual SLA Liquidated Damages Spiral** | Monthly penalty accruals from key telecom/aerospace accounts exceeding component gross margins. | Allocation optimization ignores contractual shortfall penalties, allocating purely on nominal margin. | Millions of dollars in direct contractual penalty payouts; cash flow destruction. | Incorporate explicit penalty terms ($\pi_i s_i$) in allocation MILP objective function. |
| **Packaging Substrate OSAT Bottleneck** | Plentiful front-end raw wafer inventory accompanied by zero finished packaged IC availability. | Focusing allocation monitoring purely on wafer fabs while ignoring packaging substrate / wire shortages. | Capital tied up in unfinished raw silicon wafers; inability to deliver completed components. | Model multi-echelon constraints covering wafer fabrication, substrate supply, and OSAT test lines. |
| **Premature Allocation De-escalation** | Resuming standard open ordering immediately upon first signs of inventory stabilization. | Unwinding allocation rules before phantom backlog is purged; secondary bullwhip wave strikes. | Severe second-wave whiplash; inventory whipsaw between acute shortage and extreme overstock. | Enforce phased allocation de-escalation: maintain NCNR terms until order backlog normalizes to 1.2x. |

---

## 11. Assumptions and limits

Mathematical allocation models rely on foundational assumptions. When these assumptions are violated in live industrial operations, optimization models produce suboptimal or disastrous outcomes.

| Assumption | Operational Rationale | Boundary Condition / What Breaks It | Diagnostic Test / Validation Method |
|---|---|---|---|
| **Fixed Short-Term Capacity ($K$)** | Wafer fabrication lead times prevent dynamic capacity expansion within the planning horizon. | Emergency foundry reallocations, yield breakthroughs, or tool crashes alter $K$ mid-cycle. | Weekly reconciliation of fab wafer starts, defect density charts, and OSAT line yields. |
| **Observability of Consumption ($\hat{D}_i$)** | Historical sales run-rates and EDI 867 POS data accurately reflect true customer demand. | Disruptive technological shifts (e.g., sudden transition to a new 5G standard) make past run-rates obsolete. | Cross-validate run-rate data against downstream end-system deployment and activation telemetry. |
| **Independence of Customer Demands** | Customers operate independently without strategic coordination or collusive behavior. | Tier-1 OEMs place duplicate orders through multiple EMS partners (e.g., Foxconn and Jabil for same device). | Cross-reference OEM end-product BOM allocations across all contract manufacturing partners. |
| **Linear Margin and Penalty Structures** | Profit margins and SLA penalties scale linearly with allocated physical unit volumes. | Step-function penalty clauses (e.g., binary \$10M fine if quarterly volume falls below 80% threshold). | Formulate non-linear or piecewise-linear penalties using binary indicator variables in MILP. |
| **Enforceability of NCNR Contracts** | Customers will honor non-cancellable commitments when market conditions deteriorate. | Commercial bankruptcy, extreme market collapse, or legal challenges under commercial impracticability. | Require upfront cash deposits, escrow funds, or irrevocable letters of credit for high-risk accounts. |
| **Absence of Secondary Market Leakage** | Allocated components are consumed directly in production and not resold on spot markets. | Disgruntled or opportunistic customers resell allocated inventory to independent brokers at 10x price. | Implement serial number tracking and trace field failures back to original allocated customer accounts. |
| **Homogeneous Component Quality** | All manufactured dies meeting minimum electrical specifications are equally usable by all customers. | High-performance telecom applications require tight silicon speed-binning or automotive-grade temp range. | Model multi-bin allocation: partition capacity $K$ into discrete parametric quality bins ($K_{\text{bin1}}, K_{\text{bin2}}$). |
| **Static Customer Solvency** | Allocated customers will remain financially solvent throughout the 20-52 week production lead time. | High interest rates and cash burn drive tier-2/3 hardware startups into insolvency before delivery. | Weekly monitoring of customer Altman Z-scores, credit default swap spreads, and payment aging. |

---

## 12. Exercises

### Exercise 1: Recall — Mathematical Properties of Proportional Allocation
State the proportional allocation rule formula for buyer $i$ facing capacity $K$ and order vector $\mathbf{O}$. Compute the partial derivative $\frac{\partial A_i}{\partial O_i}$ and explain mathematically why this derivative proves that proportional allocation induces order inflation when order cancellations are costless.
- **Deliverable:** Mathematical derivation and 1-paragraph proof.
- **Acceptance Criteria:** Correctly evaluate quotient rule; show $\frac{\partial A_i}{\partial O_i} > 0$ for all finite orders; state the resulting Nash equilibrium.
- **Estimated Time:** 20 minutes.

### Exercise 2: Recall — Turn-and-Earn Strategy-Proofness
Define the turn-and-earn allocation mechanism. Formulate the allocation equation for customer $i$ using historical consumption $S_i$. Prove why turn-and-earn is strategy-proof against forward order book gaming.
- **Deliverable:** Mathematical formulation and written explanation.
- **Acceptance Criteria:** Show that $\frac{\partial A_i}{\partial O_i} \equiv 0$; identify one practical operational limitation of pure turn-and-earn during a new product launch.
- **Estimated Time:** 20 minutes.

### Exercise 3: Calculation — Usable Die Capacity from Wafer Starts
A semiconductor foundry processes 1,200 wafer starts per month on a 300 mm diameter line producing a 5G network processor.
- Die dimensions: $L = 12 \text{ mm}$, $W = 10 \text{ mm}$ ($A_{\text{die}} = 1.20 \text{ cm}^2$).
- Defect density: $D_0 = 0.08 \text{ defects/cm}^2$. Murphy's yield model applies: $Y_{\text{sort}} = \left(\frac{1 - e^{-A D_0}}{A D_0}\right)^2$.
- Assembly packaging yield: $Y_{\text{assy}} = 98.5\%$.
- Final electrical burn-in test yield: $Y_{\text{test}} = 97.0\%$.
Calculate:
1. Gross Dies Per Wafer (DPW) using the perimeter edge-loss formula.
2. Wafer Sort Yield ($Y_{\text{sort}}$).
3. Net shippable component capacity ($K$) per month.
- **Deliverable:** Step-by-step arithmetic and final integer capacity.
- **Acceptance Criteria:** Correct DPW arithmetic ($\pm 2$ dies); correct yield percentages; final capacity rounded to nearest integer.
- **Estimated Time:** 30 minutes.

### Exercise 4: Calculation — Nash Equilibrium Order Inflation
Two symmetric hardware manufacturers compete for a scarce FPGA component with fixed capacity $K = 100{,}000$ units. Each firm faces true end-customer demand $D_1 = D_2 = 80{,}000$ units. The chipmaker applies proportional allocation. Each unit sold yields a net profit of $\$50$ ($p - w = \$50$). Unit shortage penalty is $\pi = \$30$. Cost of overage (holding or discarding an unneeded component) is $h = \$10$.
1. Formulate the expected payoff function for Firm 1 as a function of $O_1$ given $O_2$.
2. Assume Firm 2 orders $O_2 = 120{,}000$ units. Calculate Firm 1's optimal best response order $O_1^*$ to secure their true demand $D_1 = 80{,}000$ units.
3. If the chipmaker eliminates all cancellation fees ($h = \$0$), find the symmetric Nash equilibrium order volume $O^*$.
- **Deliverable:** Payoff equations, best response calculation, and equilibrium analysis.
- **Acceptance Criteria:** Exact numerical answer for part 2; rigorous game-theoretic justification for part 3.
- **Estimated Time:** 35 minutes.

### Exercise 5: Calculation — Lexicographic Minimax Fairness
Four customers submit demands for a rationed automotive microcontroller: $D_A = 20{,}000$, $D_B = 45{,}000$, $D_C = 60{,}000$, $D_D = 75{,}000$. Total available supply is $K = 120{,}000$ units.
Compute the exact physical allocation vector $(x_A, x_B, x_C, x_D)$ under the **Uniform Rationing (Sprumont)** mechanism: $x_i = \min(D_i, \bar{q})$.
- **Deliverable:** Iterative calculation of the uniform threshold $\bar{q}$ and allocated unit quantities.
- **Acceptance Criteria:** Identify which customers are fully satisfied; solve for exact integer $\bar{q}$; confirm $\sum x_i = 120{,}000$.
- **Estimated Time:** 25 minutes.

### Exercise 6: Application — Multi-Objective Allocation with Integer Packaging Batches
A technology hardware manufacturer allocates 180,000 units of a power management IC across two customer product lines:
- **Product A (5G Base Station):** Unit margin = \$120, Demand = 120,000 units, SLA floor = 60,000 units, SLA shortfall penalty = \$50/unit, Standard Pack Quantity ($\text{SPQ}$) = 5,000 units/reel.
- **Product B (Enterprise Switch):** Unit margin = \$70, Demand = 140,000 units, SLA floor = 40,000 units, SLA shortfall penalty = \$25/unit, Standard Pack Quantity ($\text{SPQ}$) = 10,000 units/tray.
Determine the allocation schedule that maximizes net economic contribution margin while satisfying capacity and packaging integrality constraints.
- **Deliverable:** Mathematical formulation and optimal integer allocation table.
- **Acceptance Criteria:** Ensure all allocated quantities are exact multiples of their respective SPQs; verify capacity constraint is not breached; compute exact total net dollar margin.
- **Estimated Time:** 35 minutes.

### Exercise 7: Critique — Evaluating an OEM's Dual-Sourcing and Double-Ordering Strategy
A consumer electronics procurement director discovers that their primary supplier of display driver ICs is entering allocation with a 50% quota. The director immediately:
1. Places an order for 200% of required volume with the primary supplier.
2. Places an identical order for 100% of required volume with an authorized secondary distributor.
3. Issues a non-binding letter of intent (LOI) to an unverified gray-market broker for emergency stock.
Critique this procurement strategy. Identify three severe operational risks, evaluate the legality under commercial contracts, and propose an alternative, risk-mitigated strategy.
- **Deliverable:** 1-page executive critique and alternative strategic roadmap.
- **Acceptance Criteria:** Identify inventory liability exposure, counterfeit risks, and bullwhip feedback; articulate NCNR deposit alternatives.
- **Estimated Time:** 30 minutes.

### Exercise 8: Design — End-to-End Truth-Inducing Allocation Governance Framework
Design an institutional allocation governance framework for an enterprise semiconductor manufacturer facing an industry-wide shortage. Your framework must specify:
1. The mathematical allocation algorithm across customer tiers.
2. The de-biasing telemetry inputs and audit verification mechanisms.
3. The commercial contract structure (NCNR terms, cancellation penalties, escrow requirements).
4. The post-shortage de-escalation protocol to unwind allocation rules without triggering an inventory collapse.
- **Deliverable:** Comprehensive governance policy document with flowcharts and mathematical formulas.
- **Acceptance Criteria:** Follows the 7-step pipeline from Section 5; satisfies UCC § 2-615 legal standards; includes quantitative de-escalation trigger metrics.
- **Estimated Time:** 45 minutes.

---

## 13. Self-check answers

### Answer 1: Recall — Mathematical Properties of Proportional Allocation
The proportional allocation rule is:
$$A_i(O_i, O_{-i}) = K \cdot \frac{O_i}{\sum_{j=1}^N O_j}$$
Using the quotient rule, the partial derivative with respect to $O_i$ is:
$$\frac{\partial A_i}{\partial O_i} = K \cdot \frac{1 \cdot \left(\sum_{j=1}^N O_j\right) - O_i \cdot (1)}{\left(\sum_{j=1}^N O_j\right)^2} = K \cdot \frac{\sum_{j \neq i} O_j}{\left(\sum_{j=1}^N O_j\right)^2}$$
Because capacity $K > 0$ and competitors' orders $\sum_{j \neq i} O_j > 0$, the numerator and denominator are strictly positive, so:
$$\frac{\partial A_i}{\partial O_i} > 0 \quad \forall O_i \ge 0$$
This proves that a buyer can always strictly increase their physical allocation by submitting a larger order $O_i$. When cancellation fees are zero ($h_i \to 0$), the marginal cost of ordering unneeded units is zero, while the marginal benefit of receiving scarce units is strictly positive ($(p - w) + \pi > 0$). Therefore, the optimal strategy for every buyer is to inflate $O_i$ until constrained by credit limits or external caps, establishing an infinite order inflation Nash equilibrium ($O^* \to \infty$).

### Answer 2: Recall — Turn-and-Earn Strategy-Proofness
Turn-and-earn allocates capacity based on historical consumption velocity $S_i$ across a prior baseline window:
$$A_i = K \cdot \frac{S_i}{\sum_{j=1}^N S_j}$$
Differentiating with respect to the forward order book submission $O_i$:
$$\frac{\partial A_i}{\partial O_i} \equiv 0$$
Because the submitted forward order $O_i$ does not appear in the mathematical allocation formula, the derivative is identically zero. A buyer gains zero additional components by inflating $O_i$. The mechanism is strategy-proof against order book gaming. 
*Operational Limitation:* Turn-and-earn severely penalizes fast-growing new accounts or new product introduction (NPI) platforms that have zero or low historical consumption ($S_i \approx 0$), freezing market share and locking in historical incumbents.

### Answer 3: Calculation — Usable Die Capacity from Wafer Starts
1. **Gross Dies Per Wafer (DPW):**
   - Wafer diameter $d = 300 \text{ mm} = 30 \text{ cm}$. Wafer radius $r = 15 \text{ cm}$.
   - Wafer area: $\pi \times 15^2 = 225\pi \approx 706.858 \text{ cm}^2$.
   - Die area: $A_{\text{die}} = 1.20 \times 1.00 = 1.20 \text{ cm}^2$.
   - Gross DPW formula:
     $$\text{DPW} = \frac{706.858}{1.20} - \frac{\pi \times 30}{\sqrt{2 \times 1.20}} = 589.048 - \frac{94.248}{\sqrt{2.40}} = 589.048 - \frac{94.248}{1.5492} = 589.048 - 60.837 = 528.211 \approx 528 \text{ dies}$$

2. **Wafer Sort Yield ($Y_{\text{sort}}$):**
   - Exponent: $A_{\text{die}} D_0 = 1.20 \times 0.08 = 0.096$.
   - Murphy's formula:
     $$Y_{\text{sort}} = \left(\frac{1 - e^{-0.096}}{0.096}\right)^2 = \left(\frac{1 - 0.908464}{0.096}\right)^2 = \left(\frac{0.091536}{0.096}\right)^2 = (0.9535)^2 \approx 0.9092 \quad (90.92\%)$$

3. **Net Shippable Capacity ($K$) per Month:**
   $$K = W_{\text{starts}} \times \text{DPW} \times Y_{\text{sort}} \times Y_{\text{assy}} \times Y_{\text{test}}$$
   $$K = 1{,}200 \times 528 \times 0.9092 \times 0.985 \times 0.970$$
   $$K = 633{,}600 \times 0.9092 \times 0.985 \times 0.970 \approx 576{,}069 \times 0.985 \times 0.970 \approx 567{,}428 \times 0.970 \approx 550{,}405 \text{ units}$$
   Net shippable capacity is **550,405 units/month**.

### Answer 4: Calculation — Nash Equilibrium Order Inflation
1. Under proportional allocation, Firm 1 receives $A_1 = K \frac{O_1}{O_1 + O_2}$.
   Firm 1's true demand is $D_1 = 80{,}000$. Net margin is $m = \$50$, shortage penalty $\pi = \$30$, holding cost $h = \$10$.
2. To secure exactly $A_1 = D_1 = 80{,}000$ units when $K = 100{,}000$ and $O_2 = 120{,}000$:
   $$100{,}000 \cdot \frac{O_1^*}{O_1^* + 120{,}000} = 80{,}000$$
   $$100{,}000 O_1^* = 80{,}000 O_1^* + 80{,}000 \times 120{,}000$$
   $$20{,}000 O_1^* = 9{,}600{,}000{,}000 \implies O_1^* = 480{,}000 \text{ units}$$
   Firm 1 must inflate their order by $6\times$ (ordering 480,000 units to receive 80,000 units).
3. If $h = \$0$, there is zero financial penalty for receiving surplus units. As shown in Exercise 1, $\frac{\partial \Pi_1}{\partial O_1} > 0$ for all finite $O_1$ whenever $A_1 < D_1$. Since $D_1 + D_2 = 160{,}000 > K = 100{,}000$, both firms cannot be simultaneously satisfied. Each firm continuously increases its order to capture a larger share. The unique symmetric Nash equilibrium is $O_1^* = O_2^* \to \infty$ (orders expand to credit limits).

### Answer 5: Calculation — Lexicographic Minimax Fairness
Demands: $D_A = 20{,}000$, $D_B = 45{,}000$, $D_C = 60{,}000$, $D_D = 75{,}000$. Total demand = 200,000. Supply $K = 120{,}000$.
Uniform rule: $x_i = \min(D_i, \bar{q})$. We solve for threshold $\bar{q}$:
- **Iteration 1:** Assume $\bar{q} \ge D_A = 20{,}000$. 
  Customer A is fully satisfied: $x_A = 20{,}000$.
  Remaining capacity: $K_1 = 120{,}000 - 20{,}000 = 100{,}000$.
  Remaining claimants: 3 (B, C, D).
  Trial uniform cap: $\bar{q}_1 = \frac{100{,}000}{3} \approx 33{,}333.33$.
- **Check Threshold:**
  $33{,}333.33 < D_B (45{,}000)$, $D_C (60{,}000)$, and $D_D (75{,}000)$.
  Thus, no additional customer is fully satisfied below this threshold.
- **Final Threshold:** $\bar{q} = 33{,}333.33$ units.
- **Final Allocations:**
  - $x_A = 20{,}000$ units (100% fill rate)
  - $x_B = 33{,}333$ units (74.1% fill rate)
  - $x_C = 33{,}333$ units (55.6% fill rate)
  - $x_D = 33{,}334$ units (44.4% fill rate)
  Sum of allocations: $20{,}000 + 33{,}333 + 33{,}333 + 33{,}334 = 120{,}000$ units.

### Answer 6: Application — Multi-Objective Allocation with Integer Packaging Batches
- Available Capacity: $K = 180{,}000$ units.
- Product A: Margin = \$120, SLA = 60,000, Shortfall Penalty = \$50, $\text{SPQ} = 5{,}000$. Effective value = \$120/unit.
- Product B: Margin = \$70, SLA = 40,000, Shortfall Penalty = \$25, $\text{SPQ} = 10{,}000$. Effective value = \$70/unit.

**Step 1: Satisfy Minimum Contractual SLAs:**
- Product A: $x_A \ge 60{,}000$ units ($60{,}000 / 5{,}000 = 12 \text{ reels}$). Capacity used: 60,000.
- Product B: $x_B \ge 40{,}000$ units ($40{,}000 / 10{,}000 = 4 \text{ trays}$). Capacity used: 40,000.
- Total SLA Capacity Consumed: $60{,}000 + 40{,}000 = 100{,}000 \text{ units}$.
- Residual Capacity: $180{,}000 - 100{,}000 = 80{,}000 \text{ units}$.

**Step 2: Allocate Residual Capacity to Highest-Margin Product:**
- Product A has higher margin (\$120 vs \$70). Max additional demand for Product A:
  $$D_A - 60{,}000 = 120{,}000 - 60{,}000 = 60{,}000 \text{ units}$$
- Allocate 60,000 units to Product A ($60{,}000 / 5{,}000 = 12 \text{ reels}$).
- Total Product A allocation: $x_A = 60{,}000 + 60{,}000 = 120{,}000 \text{ units}$ (24 reels).
- Remaining residual capacity: $80{,}000 - 60{,}000 = 20{,}000 \text{ units}$.

**Step 3: Allocate Final Residual Capacity to Product B:**
- Product B can take an additional $140{,}000 - 40{,}000 = 100{,}000$ units.
- Allocate remaining 20,000 units to Product B ($20{,}000 / 10{,}000 = 2 \text{ trays}$).
- Total Product B allocation: $x_B = 40{,}000 + 20{,}000 = 60{,}000 \text{ units}$ (6 trays).

**Total Allocation Summary:**
- Product A: $x_A = 120{,}000 \text{ units}$ (100% fill rate; 24 reels).
- Product B: $x_B = 60{,}000 \text{ units}$ (42.9% fill rate; 6 trays).
- Capacity Check: $120{,}000 + 60{,}000 = 180{,}000 \text{ units}$ (100% utilization).
- Net Economic Margin:
  $$\text{Margin} = (120{,}000 \times \$120) + (60{,}000 \times \$70) = \$14{,}400{,}000 + \$4{,}200{,}000 = \$18{,}600{,}000$$

### Rubric 7: Critique — Dual-Sourcing and Double-Ordering Strategy
A proficient answer must evaluate:
- **Financial Liability:** The primary supplier will likely convert allocated orders into Non-Cancellable, Non-Returnable (NCNR) agreements. If the OEM orders 200% from the primary and 100% from the secondary, they face a 300% inventory overhang when supply normalizes, risking multi-million-dollar write-offs.
- **Counterfeit / Reliability Exposure:** Issuing an LOI to an unverified gray-market broker exposes the hardware to cloned, remarked, or used components, threatening field reliability and invalidating carrier telecommunication certifications.
- **Recommended Action:** Execute a formal capacity reservation agreement with the primary supplier backed by a 20% deposit; establish real-time EDI 867 visibility with the authorized distributor; ban unverified broker purchases; redesign assembly schedules to match verified component arrival dates.

### Rubric 8: Design — Allocation Governance Framework
A proficient framework must contain:
1. **Mathematical Architecture:** Clear multi-tier waterfall (Tier 1: SLA contracts with liquidated damages; Tier 2: Turn-and-earn strategic accounts; Tier 3: Proportional with anti-starvation floor).
2. **De-Biasing Protocol:** Specific data integration steps linking EDI 830/850 orders to EDI 867 POS data, customer warehouse DOI audits, and end-device cellular activation rates.
3. **Commercial Policy:** Mandatory NCNR execution within 5 business days of allocation notice; minimum 25% cash deposit required for all uncontracted accounts.
4. **De-escalation Criteria:** Explicit numerical thresholds to exit allocation: (a) Published component lead time drops below 20 weeks; (b) Customer order-to-consumption ratio stabilizes between 0.95 and 1.10 for two consecutive quarters; (c) Fab capacity utilization drops below 92%.

---

## 14. Interview questions

### Question 1: The Allocation Gaming Mechanism
**Prompt:** "During a component shortage, why does proportional allocation incentivize customers to inflate their purchase orders, and what mathematical property of the allocation rule drives this behavior?"
- **What a strong answer contains:** Formulates the proportional rule $A_i = K \frac{O_i}{\sum O_j}$. Demonstrates that the first derivative $\frac{\partial A_i}{\partial O_i} = K \frac{\sum_{j \neq i} O_j}{(\sum O_j)^2}$ is strictly positive. Explains that when order cancellations are free, the marginal cost of inflating orders is zero, creating a dominant strategy to order infinite units. Mentions the resulting phantom backlog and bullwhip collapse.
- **What a weak answer reveals:** Claims that customers inflate orders simply due to "panic" or "bad communication," failing to recognize the rigorous game-theoretic mathematical incentives created by the rationing mechanism.

### Question 2: Truth-Inducing Allocation Design
**Prompt:** "How would you design a component allocation rule that eliminates order book gaming without requiring complex real-time auditing of customer factories?"
- **What a strong answer contains:** Proposes a **turn-and-earn** mechanism where allocations are indexed strictly to verified historical consumption ($S_i$) over trailing baseline periods. Notes that because $\frac{\partial A_i}{\partial O_i} \equiv 0$, forward order inflation yields zero additional silicon. Alternatively proposes capacity reservation contracts with non-refundable deposits ($r$) where customers internalize the financial risk of speculative ordering.
- **What a weak answer reveals:** Suggests "calling the customers to negotiate" or applying across-the-board percentage cuts to forward orders, which directly perpetuates proportional gaming.

### Question 3: The Golden Screw Dilemma
**Prompt:** "You are the VP of Supply Chain for an enterprise networking OEM. Your \$50,000 core router cannot be shipped because a \$0.80 voltage regulator is on 65-week allocation. How do you resolve this operational crisis?"
- **What a strong answer contains:** Outlines a multi-pronged mitigation hierarchy: (1) Formulate multi-component kitting constraints in MRP to prevent building uncompletable subassemblies; (2) Initiate immediate electrical engineering qualification of drop-in pin-compatible alternative regulators; (3) Engage directly with the executive leadership of the semiconductor IDM to offer NCNR multi-year commitments in exchange for emergency wafer reallocation; (4) Implement strict authorized distributor tracing (avoiding unvetted gray-market brokers to prevent counterfeit insertion).
- **What a weak answer reveals:** Suggests halting all manufacturing lines indefinitely, or immediately buying unverified chips from eBay/brokers without qualification or counterfeit screening.

### Question 4: Multi-Objective LP Formulation
**Prompt:** "Walk me through the formulation of an optimization model that allocates scarce microcontrollers across multiple automotive customer accounts. What belong in the objective function, and what are the critical constraints?"
- **What a strong answer contains:** Articulates an LP/MILP model. Objective: Maximize weighted contribution margin minus contractual SLA shortfall liquidated damages. Constraints: Component capacity ($\sum x_i \le K$), demand bounds ($x_i \le \hat{D}_i$), contractual minimum SLAs ($x_i + s_i \ge \text{SLA}_i$), minimum anti-starvation equity floors ($x_i \ge \alpha^{\text{floor}} \hat{D}_i$), and packaging integrality (multiples of standard pack quantity).
- **What a weak answer reveals:** Omits contractual SLA shortfall penalties, ignores packaging integrality, or suggests a single-objective model that purely maximizes revenue, which starves smaller customers and triggers antitrust / UCC § 2-615 legal challenges.

### Question 5: Order De-Biasing Techniques
**Prompt:** "How do you distinguish genuine structural demand growth from phantom double ordering when evaluating a customer's purchase order history during an industry-wide shortage?"
- **What a strong answer contains:** Identifies empirical de-biasing signals: (1) Reconciling sell-in orders against EDI 867 / POS sell-through reports; (2) Monitoring customer inventory Days-on-Hand (DOI = $I / c$), where DOI $> 45$ days indicates hoarding; (3) Cross-referencing against physical deployment telemetry (e.g., cell tower zoning permits, cloud data center power provisioning, MAC/IMEI activations); (4) Applying Bayesian shrinkage filters that weight orders by historical forecast credibility.
- **What a weak answer reveals:** Assumes customer purchase orders are always truthful or suggests relying purely on executive verbal assurances.

### Question 6: Legal Mandates Under Shortage (UCC § 2-615)
**Prompt:** "If our factory experiences a catastrophic fire and our component supply is cut by 60%, can we legally allocate 100% of remaining output to our highest-margin customer while giving zero components to our long-standing lower-margin accounts?"
- **What a strong answer contains:** Cites **UCC § 2-615(b)**. Explains that the law excuses non-performance only if the seller allocates available production among its customers in a manner that is **fair and reasonable**. Notes that allocating 100% to one preferred customer while completely cutting off others breaches commercial good faith, exposes the company to massive breach-of-contract lawsuits, and risks antitrust claims under the Robinson-Patman Act. Mentions the requirement to issue formal written notice under UCC § 2-615(c).
- **What a weak answer reveals:** Believes commercial contracts allow complete supplier discretion or that "profit maximization" serves as a valid legal defense against contract repudiation.

### Question 7: NCNR Contract Execution
**Prompt:** "What is an NCNR contract, why is it implemented during component allocation, and what is its primary failure mode?"
- **What a strong answer contains:** Defines NCNR (Non-Cancellable, Non-Returnable). Explains that it forces buyers to internalize inventory holding and obsolescence risk, eliminating costless order cancellations. Identifies the primary failure mode: customer insolvency or commercial repudiation when end-markets collapse. If a buyer goes bankrupt or faces total business failure, an NCNR clause becomes an unsecured creditor claim in bankruptcy court, leaving the supplier holding worthless inventory.
- **What a weak answer reveals:** Assumes NCNR guarantees 100% cash collection under all economic conditions without requiring upfront deposits or letters of credit.

### Question 8: Unwinding Allocation (De-escalation)
**Prompt:** "What operational indicators would you monitor to decide when it is safe to lift component allocation rules and return to standard unconstrained order replenishment?"
- **What a strong answer contains:** Names quantitative criteria: (1) Component published lead times compressing to normal historical ranges (e.g., $<16$ weeks); (2) Fab book-to-bill ratio dropping to approximately 1.0; (3) Customer order-to-consumption ratio stabilizing between 0.95 and 1.05; (4) Channel inventory (distributor DOI) stabilizing within target corridors (e.g., 30–45 days); (5) Foundry capacity utilization softening to $<92\%$. Emphasizes phased de-escalation rather than sudden unconstrained ordering to prevent a secondary bullwhip whipsaw.
- **What a weak answer reveals:** Suggests lifting allocation the moment total monthly production output matches total monthly incoming orders, ignoring the accumulated phantom backlog.

---

## 15. Key terms

- **Allocation Factor ($\alpha$):** The dimensionless scalar ($0 \le \alpha \le 1$) representing total available capacity divided by aggregate unconstrained demand.
- **Anti-Starvation Floor:** A mandatory minimum allocation percentage enforced across all active accounts to prevent customer assembly line shutdown.
- **Bayesian Shrinkage Filter:** A statistical smoothing technique that discounts volatile purchase orders toward historical baseline consumption.
- **Book-to-Bill Ratio:** The ratio of incoming purchase order value to shipped and billed revenue; values $>1.2$ during shortages signal aggressive order gaming.
- **Capacity Reservation Contract:** An advance commercial agreement where a buyer pays an upfront fee to guarantee dedicated manufacturing capacity.
- **Dies Per Wafer (DPW):** The gross geometric count of silicon integrated circuit dies fitting onto a single circular semiconductor wafer.
- **Double Ordering:** The strategic placement of duplicate purchase orders across multiple suppliers or distributors to capture a larger share of rationed supply.
- **EDI 855:** Electronic Data Interchange Purchase Order Acknowledgment, used to communicate formal allocated quantities and ship schedules.
- **EDI 867:** Electronic Data Interchange Product Transfer and Resale Report, providing verified point-of-sale sell-through data from distributors.
- **Golden Screw Problem:** A manufacturing stoppage where completed multi-thousand-dollar systems cannot ship due to the absence of a single low-cost component.
- **Lead Time Inversion:** A supply chain failure where component procurement lead time exceeds the commercial life cycle of the final finished product.
- **Lexicographic Minimax Fairness:** An allocation algorithm that hierarchically maximizes the minimum fill rate across competing customer accounts.
- **Line-Down Situation:** The complete stoppage of a downstream assembly facility caused by a component stockout, incurring severe hourly downtime penalties.
- **Liquidated Damages:** Contractually predetermined financial penalties assessed per unit or per day for failing to meet delivery commitments.
- **Murphy's Yield Model:** A mathematical defect clustering model used to predict functional wafer sort yield as a function of die area and defect density.
- **NCNR Agreement:** Non-Cancellable, Non-Returnable procurement contract locking in buyer financial liability.
- **OSAT:** Outsourced Semiconductor Assembly and Test facilities responsible for packaging, lead bonding, and final functional burn-in testing.
- **Phantom Demand:** Speculative or duplicate order backlog created by downstream buyers that will evaporate upon shortage normalization.
- **Point-of-Sale (POS) Sell-Through:** The verified volume of components transferred from distributors to genuine end-customer production facilities.
- **Proportional Allocation:** A naive rationing policy that distributes scarce capacity proportionally to submitted order sizes, incentivizing double ordering.
- **Robinson-Patman Act:** US federal antitrust statute prohibiting discriminatory pricing and promotional practices that injure competition.
- **Standard Pack Quantity (SPQ):** The discrete physical packaging increment (reels, trays, tubes) required for automated surface-mount assembly.
- **Turn-and-Earn Allocation:** A strategy-proof rationing mechanism that indexes current component supply strictly to verified historical consumption velocity.
- **UCC § 2-615:** Uniform Commercial Code provision governing commercial impracticability and mandating fair and reasonable allocation among customers.
- **Uniform Rationing (Sprumont):** A strategy-proof allocation rule capping all claimants at a common ceiling $\bar{q}$, fulfilling smaller orders in full.
- **Wafer Sort Yield:** The percentage of functional, defect-free silicon dies identified during electrical probe testing prior to wafer dicing.

---

## 16. Sources

- **Cachon, G. P., & Lariviere, M. A. (1999a).** Capacity allocation using past sales: When to turn-and-earn. *Management Science*, 45(5), 685–703. https://doi.org/10.1287/mnsc.45.5.685
- **Cachon, G. P., & Lariviere, M. A. (1999b).** Capacity choice and allocation: Strategic behavior and supply chain performance. *Management Science*, 45(8), 1091–1108. https://doi.org/10.1287/mnsc.45.8.1091
- **Lee, H. L., Padmanabhan, V., & Whang, S. (1997).** Information distortion in a supply chain: The bullwhip effect. *Management Science*, 43(4), 546–558. https://doi.org/10.1287/mnsc.43.4.546
- **Sprumont, Y. (1991).** The division problem with single-peaked preferences: A characterization of the uniform allocation rule. *Econometrica*, 59(2), 509–519. https://doi.org/10.2307/2938266
- **Uniform Commercial Code (UCC) § 2-615.** *Excuse by Failure of Presupposed Conditions.* American Law Institute and National Conference of Commissioners on Uniform State Laws.
- **Federal Trade Commission.** *The Robinson-Patman Act: Price Discrimination and Competitive Injury.* 15 U.S.C. § 13.
- **JEDEC Solid State Technology Association.** *JESD46D: Customer Notification of Product/Process Changes by Semiconductor Suppliers.* Standard Specification.
- **JEDEC Solid State Technology Association.** *JESD48C: Product Discontinuance.* Standard Specification.
- **Semiconductor Equipment and Materials International (SEMI).** *SEMI E10: Specification for Definition and Measurement of Equipment Reliability, Availability, and Maintainability.*
- **U.S. Department of Commerce. (2022).** *Results from Semiconductor Supply Chain Request for Information (RFI).* Office of Technology Evaluation, Bureau of Industry and Security.
- **Cisco Systems, Inc. (2001).** *Form 10-K Annual Report for the Fiscal Year Ended July 28, 2001.* United States Securities and Exchange Commission (Detailing the \$2.25 billion inventory write-down).

---

## 17. Links

- Module Index: [Module 40 — Telecom, Technology Hardware, and Aerospace Supply Chains](_index.md)
- Prior Topic: [Configuration, BOM, and Life Cycle](01_configuration-bom-lifecycle.md)
- Next Topic: [EMS and Contract Manufacturing](03_ems-contract-manufacturing.md)
- Sibling Topic: [NPI and Ramp](04_npi-ramp.md)
- Root Survey: [Module 40 Survey File](../../../course/40_telecom_technology_hardware.md)
- Prerequisite Foundations:
  - [Module 09 — Inventory Theory and Stochastic Replenishment](../../02_methods/09_inventory_theory_stochastic_replenishment/_index.md)
  - [Module 10 — Multi-Echelon Inventory Optimization](../../02_methods/10_multi_echelon_inventory/_index.md)
  - [Module 14 — Supply Chain Game Theory and Contract Design](../../02_methods/14_game_theory_contracts/_index.md)
  - [Module 31 — High-Tech, Electronics, and Semiconductor Operations](../../06_domain_electives/31_high_tech_electronics_semiconductors/_index.md)
  - [Module 37 — Supply Chain Risk, Resilience, and Disruption Management](../../06_domain_electives/37_risk_resilience_disruption/_index.md)
