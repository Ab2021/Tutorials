# Refurbishment and Remanufacturing

> Module 37 — Circular Economy and Reverse Logistics Networks
> Topic 05 of 10
> Estimated study time: 4.5 hours
> Prerequisites: Topic 01 (Reverse Flows and the Disposition Hierarchy), Topic 02 (Return Unit Economics), Topic 03 (Grading and Recovery Yield), Module 10 (Inventory Fundamentals), Module 11 (Safety Stock & Service Levels), Module 12 (Multi-Echelon Inventory Optimization)
> Version 2026.5

---

## 1. Why this topic exists

In conventional forward supply chains, manufacturing operations run on predictable inputs. Procurement teams issue purchase orders against rigid engineering drawings; suppliers deliver virgin materials accompanied by Advanced Shipping Notices (ASNs) and certificates of analysis; production lines consume homogeneous parts under deterministic bills of materials (BOMs); and assembly cycle times exhibit minimal variance. Production planning operates in a structured environment where lead times, component yields, and quality tolerances are tightly bounded.

Closed-loop supply chains and reverse manufacturing environments invert every one of these stabilizing assumptions. When an enterprise recovers used products—whether heavy-duty diesel engines, commercial aircraft avionics, enterprise data center servers, wind turbine gearboxes, or high-end consumer electronics—it confronts what reverse logistics engineering terms the **triple uncertainty**:
1. **Timing uncertainty:** Cores return at the whim of end-user replacement cycles, lease expirations, maintenance outages, and macroeconomic trade-in incentives, decoupled from current production schedules.
2. **Quantity uncertainty:** Aggregate return volumes fluctuate based on historical installed base sales, geographic dispersion, seasonal replacement surges, and the effectiveness of buyback incentives.
3. **Quality uncertainty:** Inbound assets arrive in radically heterogeneous physical, thermal, mechanical, and electrical condition. Two assets of the same model year may have vastly different remaining useful life (RUL), corrosion, thermal degradation, fatigue, and cosmetic damage.

The economic imperative to master remanufacturing and refurbishment is immense. In capital-intensive industrial equipment and complex electronic systems, between 70% and 85% of the total embedded energy, greenhouse gas footprint, raw material mass, and precision machining cost resides in the heavy structural casting, forging, or chassis (such as an engine block, turbine casing, or server chassis). Disposing of or shredding these assets into raw scrap metals destroys nearly all of this accumulated intellectual and mechanical value. Recycling recovers only the base commodity melt value—fractions of pennies per dollar of original value—while consuming substantial thermal energy.

**Remanufacturing** captures this embedded value by restoring a worn or failed asset to its original original equipment manufacturer (OEM) performance specifications, or surpassing them with modern engineering upgrades, backed by a warranty equivalent to or exceeding that of a newly manufactured unit. **Refurbishment**, while operating at a lower recovery tier, restores functionality and cosmetic appearance to defined secondary market standards without necessarily returning the product to pristine OEM tolerances.

Yet enterprises routinely fail to realize these economic and environmental gains because they treat remanufacturing as an undisciplined repair shop rather than an advanced analytical production system. Five foundational operational bottlenecks consistently undermine profitability:
- **The Core Acquisition Dilemma:** Setting core buyback prices too low starves the remanufacturing facility of feedstock, starving capital-intensive facilities of throughput. Setting buyback prices too high attracts severely degraded, economically unviable units or causes an adverse selection "lemons" spiral where holders of high-quality cores hoard them while holders of junk cores flood the depot.
- **The Reverse BOM Disassembly Problem:** Unlike forward assembly, which follows a linear or converging tree structure, disassembly is governed by non-linear AND/OR directed hypergraphs. Technicians must decide how deep to disassemble an asset: does one completely dismantle the core into individual bolts and washers, or stop at modular sub-assemblies? Complete disassembly destroys structural joins and escalates labor costs; shallow disassembly misses hidden internal cracks and leaves valuable high-grade alloys unrecovered.
- **Stochastic Yield and Demand Imbalance:** Disassembling a batch of 1,000 cores yields a fixed bundle of parts—say, 900 engine blocks, 750 crankshafts, and only 400 electronic control units (ECUs). But to build a remanufactured engine, these parts must be consumed in strict 1:1:1 ratios. This yields the classic "unbalanced bundle" trap: the plant suffers simultaneous inventory stockouts of bottleneck components and catastrophic inventory accumulation of surplus components.
- **Capacity Planning and Line Balancing Instability:** Forward assembly lines are engineered for smooth cycle times with low coefficients of variation ($c_s = \sigma / \mu \approx 0.05 - 0.15$). In remanufacturing, cycle times for cleaning, non-destructive evaluation (NDE), and re-machining are heavy-tailed random variables ($c_s \in [1.0, 2.5]$). A heavily rusted or coked core takes four times longer to process than a lightly used core. This extreme variance triggers catastrophic queue explosions, work-in-process (WIP) accumulation, and dynamic bottleneck migration that destabilizes traditional manufacturing execution systems (MES).
- **Market Cannibalization and Channel Conflict:** Offering remanufactured products at a discount (typically 20% to 40% below new product retail) risks eroding high-margin primary product sales. If existing brand-loyal customers switch to buying discounted remanufactured units, the enterprise cannibalizes its own core revenue stream. Conversely, failing to offer certified remanufactured goods leaves secondary recommerce markets open to third-party gray-market rebuilders who capture aftermarket margin and damage OEM brand reputation through inferior quality.

Solving these interconnected operational and economic challenges requires a unified quantitative framework integrating microeconomic contract design, graph theory, stochastic dynamic programming, heavy-traffic queueing models, and game-theoretic dual-market pricing. This topic provides the mathematical foundations, operational mechanics, and computational tools required to design and manage industrial-grade refurbishment and remanufacturing networks.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Formulate and solve** quality-differentiated core acquisition pricing models $P(q)$ that balance core holder reservation price distributions against downstream remanufacturing yields, eliminating adverse selection and minimizing feedstock acquisition cost.
2. **Construct and evaluate** reverse Bill of Materials (rBOM) AND/OR disassembly graphs, applying topological sorting and mixed-integer linear programming (MILP) to identify the profit-maximizing selective disassembly depth and stopping boundary under precedence and joint recovery constraints.
3. **Model and calibrate** multi-component stochastic recovery yields using Beta and Dirichlet-multinomial probability distributions to capture component wear heterogeneity and vintage degradation across incoming core lots.
4. **Compare and optimize** push versus pull disassembly policies, mathematically proving the exact inventory holding cost, lead time, and obsolescence trade-offs that dictate when an enterprise should store intact cores versus disassembling into part-level stock.
5. **Formulate a multi-period joint production planning model** that synchronizes remanufactured component streams with virgin component procurement to satisfy forward finished goods demand at minimal total holding, production, and shortage cost.
6. **Quantify remanufacturing line congestion, queue length, and lead time** under high processing time variance ($c_s > 1.0$) using Kingman's heavy-traffic $G/G/m$ queueing approximation, predicting dynamic bottleneck migration across cleaning, NDE, and re-machining workcenters.
7. **Evaluate dedicated versus hybrid manufacturing/remanufacturing facility architectures**, quantifying the economic and operational break-even points between economies of scale and cross-entropy disruptions.
8. **Derive closed-loop dual-market equilibrium prices** $(p_n^*, p_r^*)$ and customer segmentation boundaries $(\theta_{nr}, \theta_{r0})$ under customer willingness-to-pay discounting ($\delta$), computing the marginal cannibalization rate and Karush-Kuhn-Tucker (KKT) shadow prices when remanufacturing volume is core-constrained.

---

## 3. The core mental model

The operational and economic architecture of remanufacturing can be understood as an interconnected closed-loop system of five coupled feedback mechanisms spanning market demand, core collection, physical transformation, inventory balancing, and secondary sales.

```text
       +-------------------------------------------------------------+
       |                  PRIMARY FORWARD MARKET                     |
       |  New Product Demand: Dn(pn, pr) | Selling Price: pn         |
       +------------------------------+------------------------------+
                                      |
                         (Sales to Installed Base N)
                                      v
       +-------------------------------------------------------------+
       |               INSTALLED BASE & CORE GENERATION              |
       |  Assets in Service | End-of-Life (EOL) | Fleet Retirement   |
       +------------------------------+------------------------------+
                                      |
                      (Core Returns driven by Buyback)
                                      v
       +-------------------------------------------------------------+
       |               CORE ACQUISITION & GRADING ENGINE             |
       |  Grading Schedule P(q) | Reservation Price F(r|q)           |
       |  Sort into Tiers: Grade A (Light), B (Med), C (Heavy), Scrap|
       +------------------------------+------------------------------+
                                      |
                          (Acquired Raw Cores: QC)
                                      v
       +-------------------------------------------------------------+
       |             REVERSE BOM DISASSEMBLY AND/OR GRAPH            |
       |  Selective Disassembly Optimization (MILP Stopping Rule)    |
       |  Non-Destructive Unbolting vs Destructive Fastener Removal  |
       +------------------+-----------------------+------------------+
                          |                       |
            (Harvested High-Value Subassemblies)  | (Scrap / Metal Melt)
                          v                       v
       +------------------------------------+   +--------------------+
       |   COMPONENT RECOVERY & TESTING     |   | MATERIAL RECYCLING |
       |  Stochastic Yield: Yk ~ Beta(a, b) |   | Smelting & Slag    |
       +------------------+-----------------+   +--------------------+
                          |
             (Recovered Good Components)
                          v
       +-------------------------------------------------------------+
       |            DUAL-SOURCE INVENTORY BALANCING ENGINE           |
       |  Matching Stochastic Yield Vector Y to BOM Requirements     |
       |  Shortage Injection: Virgin Parts Procurement Q_virgin      |
       +------------------------------+------------------------------+
                                      |
                     (Balanced Component Kits: Q_kit)
                                      v
       +-------------------------------------------------------------+
       |         VARIABLE-PACED RE-ASSEMBLY & TESTING LINE           |
       |  High Variance Cycle Times (cs > 1.2) | Kingman WIP Buffers |
       |  Precision CNC Machining, Balancing, Dynamometer Testing    |
       +------------------------------+------------------------------+
                                      |
                 (Finished Remanufactured Units: Q_reman)
                                      v
       +-------------------------------------------------------------+
       |                SECONDARY RECOMMERCE MARKET                  |
       |  Reman Demand: Dr(pn, pr) | Price: pr = delta * pn          |
       |  Certified OEM Warranty | Distinct Packaging & Branding     |
       +------------------------------+------------------------------+
                                      |
       (Market Cannibalization Feedback: delta * dDr steals from Dn)
                                      +-------------------------------> [Back to Primary Market]
```

### The Five Coupled System Loops

1. **The Core Acquisition Supply Loop:** The primary forward market injects new physical units into the global installed base $N$. After an operational lifespan governed by Weibull failure mechanics or economic obsolescence, units reach end-of-life (EOL). Core holders evaluate the OEM's quality-differentiated buyback offer $P(q)$ against their internal reservation price $r_i$ (which reflects alternative scrap value, secondary market resale, or the hassle cost of packaging and return). Higher acquisition prices expand the acquired core volume $Q_C$, but compress the unit gross margin of downstream remanufacturing.
2. **The Reverse BOM Disassembly and Yield Loop:** Acquired cores pass into the disassembly triage engine. Unlike forward assembly where parts are joined additively, disassembly is a subtractive branching process modeled as an AND/OR directed hypergraph. Disassembling a sub-assembly yields joint component outputs governed by stochastic recovery yield distributions $Y_k$. The optimization engine evaluates the marginal recovery value of every liberated component against the marginal labor, tooling, and core damage risk of the disassembly operation, executing an optimal stopping rule that halts disassembly when marginal net value becomes negative.
3. **The Dual-Source Inventory Balancing Loop:** Disassembly releases components in fixed physical ratios, but physical wear creates asymmetric component survival rates. Durable structural components (engine blocks, machine beds) exhibit 85–95% yields, while dynamic friction components (bearings, pistons, seals, valves) exhibit zero to 20% yields. To satisfy the strict 1:1 bill of materials required for finished good assembly, the inventory engine operates a dual-sourcing model: it matches stochastic harvested part flows with top-up procurement of virgin components, balancing the holding cost of surplus parts against the premium cost and lead time of virgin parts.
4. **The Variable-Paced Line Balancing Loop:** Components enter the re-machining, reconditioning, and assembly line. Forward lines maintain deterministic cycle times with low variance. Remanufacturing lines face high cycle-time variance ($c_s > 1.0$) caused by varying core contamination, thermal distortion, metal galling, and multiple rework loops. This processing variance triggers queue explosions under high utilization. The system dynamics require dynamic buffer sizing via Kingman's heavy-traffic approximation and dynamic line pacing to prevent station starvation and WIP gridlock.
5. **The Cannibalization and Market Clearing Loop:** Finished remanufactured goods enter the secondary market at price $p_r$, competing alongside new products priced at $p_n$. Consumers possess heterogeneous willingness-to-pay valuations $\theta \sim U[0, 1]$ and discount remanufactured goods by a perceived quality factor $\delta \in (0, 1)$. The pricing engine solves a joint optimization problem: lowering $p_r$ expands market share into price-sensitive customer segments who could never afford a new unit, but simultaneously cannibalizes high-margin new product sales at a marginal rate equal to $\delta$. When inbound core supply is constrained, the firm raises $p_r$, capturing scarcity rents and driving consumers back toward new product purchases.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse or Confusion | Standard Units / Metrics |
|---|---|---|---|
| **Core** | A previously sold, used, worn, or failed asset or module containing high-value castings, forgings, or electronic chassis retained for remanufacturing. | Confused with scrap, garbage, or raw waste. A core possesses substantial embedded engineering value. | Units (count), kg, or Core ID |
| **Remanufacturing** | An industrial process that restores an end-of-life core to original OEM performance specifications (or better) with an equivalent warranty. | Confused with simple repair or reconditioning. Remanufacturing requires complete teardown, inspection, re-machining, and testing. | ANSI/RIC001.1-2016 compliance |
| **Refurbishment** | The functional and cosmetic reconditioning of a used product to acceptable secondary working standards, not necessarily OEM tolerances. | Used interchangeably with remanufacturing; refurbishment does not involve full dimensional re-machining. | Cosmetic grades (A/B/C) |
| **Repair** | The localized corrective action taken on a specific failed component to return an asset to operational state without a complete overhaul. | Mistaken for remanufacturing; repair does not reset the product's overall wear clock or life expectancy. | Mean Time To Repair (MTTR) |
| **Reverse BOM (rBOM)** | An AND/OR directed hypergraph representing the non-linear disassembly relationships, joint outputs, and material extraction paths of a product. | Assumed to be simply the inverse of a forward assembly tree. | Hypergraph $G = (V, E)$, Node Depth |
| **Selective Disassembly** | The operational strategy of halting product teardown at an intermediate sub-assembly level to maximize net recovery value. | Believing disassembly must always proceed to the smallest fastener. | Net Recovery Margin ($/unit) |
| **Destructive Disassembly** | Teardown operations that intentionally cut, drill, or sever low-value fasteners or housings to rapidly access high-value internal components. | Viewed as an error; destructive methods are often optimal when labor cost exceeds fastener scrap value. | Disassembly Cycle Time (sec) |
| **Core Acquisition Price $P(q)$** | The financial buyback incentive offered to a core holder as a function of verified or nominal condition grade $q$. | Setting a single flat price regardless of quality, inducing adverse selection. | USD ($), EUR (€) per unit |
| **Core Reservation Price $r_i$** | The minimum monetary amount a core holder requires to relinquish and ship an end-of-life asset to the remanufacturer. | Assuming core holders return assets freely due to environmental sentiment. | USD ($) per unit |
| **Adverse Selection (Lemons Effect)** | The economic phenomenon where a uniform buyback price causes holders of high-quality cores to withhold them while low-quality cores flood in. | Attributing low incoming quality purely to bad customer behavior rather than pricing structure. | Mean quality delta: $E[q \mid P] - \bar{q}$ |
| **Stochastic Recovery Yield $Y_k$** | The probability distribution governing the proportion of usable, defect-free components recovered from a disassembled batch of cores. | Assuming recovery yields are deterministic constants in MRP/ERP systems. | Beta parameters $(\alpha, \beta)$, Yield % |
| **Push Disassembly** | An operational policy that immediately disassembles all incoming cores upon receipt and stores harvested components in inventory. | Equating push disassembly with lean manufacturing; push can cause severe component inventory holding bloat. | Inventory Holding Cost ($/week) |
| **Pull Disassembly** | An operational policy that holds intact cores in raw storage and disassembles them only when specific downstream component demand arises. | Believing pull is always superior; pull can fail when disassembly lead times exceed customer lead-time tolerance. | Lead Time (days), Stockout Rate |
| **Cycle Time Coeff. of Var. ($c_s$)** | The ratio of processing time standard deviation to mean processing time ($c_s = \sigma / \mu$) across remanufacturing operations. | Assuming normal distribution with low variance; remanufacturing processing times are heavily right-skewed. | Dimensionless ratio ($c_s$) |
| **Dynamic Bottleneck Migration** | The phenomenon where line bottlenecks shift unpredictably between cleaning, NDE, and machining workcenters as core condition varies. | Assuming the bottleneck station is permanently fixed at the same machine tool. | Station Utilization ($\rho_j$) |
| **Willingness-to-Pay Discount ($\delta$)** | The fractional valuation factor ($\delta \in (0, 1)$) that consumers apply to remanufactured goods relative to pristine new goods. | Assuming remanufactured goods can be sold at parity price with new products without demand collapse. | Dimensionless factor $\delta$ |
| **Cannibalization Rate ($\gamma$)** | The proportion of remanufactured product sales that directly replace sales of new products rather than capturing new market demand. | Believing all remanufactured sales are 100% additive to corporate revenue. | Marginal substitution ratio: $\partial D_n / \partial D_r$ |
| **Indifference Valuation ($\theta_{nr}$)** | The consumer valuation threshold at which a buyer is indifferent between purchasing a new product and a remanufactured product. | Confusing market boundary thresholds with mean customer willingness to pay. | Normalized consumer valuation $\theta \in [0, 1]$ |
| **Core Availability Shadow Price ($\lambda$)** | The marginal increase in dual-market profit resulting from relaxing the core availability constraint by one unit. | Setting remanufacturing targets without accounting for physical core feedstock limits. | USD ($) per core unit |
| **Digital Product Passport (DPP)** | An interoperable digital record capturing asset origin, material composition, repair history, and disassembly instructions. | Treating DPP as a simple static barcode rather than an active lifecycle data exchange. | JSON-LD schema / ISO 14040 |
| **Core Deposit / Surcharge** | An upfront fee charged to new buyers that is refunded only when an equivalent worn core is returned to the OEM. | Confusing deposits with non-refundable environmental disposal taxes. | USD ($) per unit sold |
| **Absorbing Terminal State** | In an absorbing Markov chain of reverse routing, a final disposition state (such as scrap/melt or certified resale) that cannot be exited. | Allowing infinite rework loops without economic or cycle-count stopping limits. | Absorption Probability Matrix |

---

## 5. The mechanics, step by step

Remanufacturing execution requires a coordinated sequence of five analytical mechanisms.

```text
  +-------------------------------------------------------------------------------------------------+
  | Step 5.1: Calibrate Quality-Differentiated Core Acquisition Pricing Curves P(q)                 |
  | Inputs: Reservation price distribution F(r|q), installed base N, downstream salvage value       |
  | Outputs: Optimal tier pricing P*, target intake volume QC, expected feedstock quality E[q]      |
  +------------------------------------------------+------------------------------------------------+
                                                   |
                                                   v
  +-------------------------------------------------------------------------------------------------+
  | Step 5.2: Construct and Solve Reverse BOM Disassembly AND/OR Precedence Graph                   |
  | Inputs: CAD disassembly hypergraph G=(V,E), labor rates, tool costs, component values Vc        |
  | Outputs: Optimal selective disassembly depth, stopping boundary, non-destructive vs shear paths |
  +------------------------------------------------+------------------------------------------------+
                                                   |
                                                   v
  +-------------------------------------------------------------------------------------------------+
  | Step 5.3: Model Stochastic Recovery Yields and Solve Dual-Source Inventory Balancing           |
  | Inputs: Component Beta distributions Yk~Beta(a,b), BOM multipliers, virgin part lead times      |
  | Outputs: Balanced production batches, push vs pull boundary, virgin top-up procurement plan     |
  +------------------------------------------------+------------------------------------------------+
                                                   |
                                                   v
  +-------------------------------------------------------------------------------------------------+
  | Step 5.4: Plan Remanufacturing Capacity and Buffer Sizing under Processing Time Variance        |
  | Inputs: Arrival rates lambda, station mean cycle times mu, coefficient of variation cs > 1.0    |
  | Outputs: Workstation buffer allocations, Kingman queue waiting times, dedicated vs hybrid lines |
  +------------------------------------------------+------------------------------------------------+
                                                   |
                                                   v
  +-------------------------------------------------------------------------------------------------+
  | Step 5.5: Solve Closed-Loop Dual-Market Pricing under Cannibalization and Core Limits           |
  | Inputs: WTP discount delta, production costs cn and cr, core supply limit Q_core_max            |
  | Outputs: Equilibrium prices (pn*, pr*), demand allocations (Dn*, Dr*), KKT shadow prices lambda|
  +-------------------------------------------------------------------------------------------------+
```

### 5.1 Core Acquisition Pricing and Quality Grading Curves

An enterprise cannot remanufacture cores it does not possess. Setting core acquisition pricing requires modeling the economic behavior of decentralized core holders (individual consumers, fleet operators, or independent maintenance repair facilities).

Let the physical quality of a core be represented by a continuous scalar $q \in [0, 1]$, where $q = 1$ denotes a pristine asset with zero wear, and $q = 0$ denotes a catastrophically failed, thermally cracked unit. In practice, quality is partitioned into $K$ discrete inspection condition tiers: $\{k = 1, 2, \dots, K\}$, where Tier 1 represents Grade A (light cosmetic wear, fully functioning core casting), Tier 2 represents Grade B (moderate wear, minor scoring), Tier 3 represents Grade C (heavy wear, weld repair required), and Tier 4 represents scrap/unrecoverable.

Every core holder $i$ possesses an unobserved internal reservation price $r_i$ for relinquishing the core. The reservation price represents the holder's opportunity cost: the value of holding the core as an on-site emergency spare, selling it to a local scrap merchant, or the friction cost of packaging and shipping. We model $r_i$ as a conditional continuous random variable with cumulative distribution function $F(r \mid q)$ and probability density function $f(r \mid q)$. Higher-quality cores have higher reservation prices: $\partial F(r \mid q) / \partial q < 0$.

If the remanufacturer offers an acquisition buyback price $P(q)$ for a core of verified quality $q$, a rational core holder returns the unit if and only if:
$$P(q) \ge r_i$$

For a total addressable installed base of $N(q)$ units in quality condition $q$, the aggregate core return supply function $S(P, q)$ is:
$$S(P, q) = N(q) \cdot F(P(q) \mid q)$$

#### The Lemons Problem and Adverse Selection

Suppose the enterprise implements a flat acquisition price $\bar{P}$ across all incoming units because it lacks pre-inspection screening capabilities. The probability of acquisition across quality levels becomes:
$$\Pr(\text{Return} \mid q) = F(\bar{P} \mid q)$$

Because $F(\bar{P} \mid q)$ is strictly decreasing in $q$, the conditional expected quality of cores acquired under a flat price $\bar{P}$ is:
$$E[q \mid \text{Return}, \bar{P}] = \frac{\int_0^1 q \cdot N(q) \cdot F(\bar{P} \mid q) \, dq}{\int_0^1 N(q) \cdot F(\bar{P} \mid q) \, dq} < E[q \mid \text{Installed Base}]$$

This is Akerlof's classic adverse selection spiral in reverse logistics: owners of high-quality cores find $\bar{P}$ insulting and retain or sell them through secondary channels, while owners of junk cores flood the remanufacturing facility with severely degraded units. The plant's average remanufacturing cost escalates, destroying profitability.

#### Optimal Multi-Tier Buyback Formulation

To defeat adverse selection, the enterprise implements a quality-differentiated pricing schedule $\mathbf{P} = [P_1, P_2, \dots, P_K]^T$. Let $R_k$ be the expected net downstream revenue generated by remanufacturing a Tier $k$ core (accounting for disassembly yield and resale value), and let $c_k^{proc}$ be the diagnostic inspection, handling, and reconditioning cost for Tier $k$. The acquisition profit maximization problem is:
$$\max_{P_1, \dots, P_K} \Pi_{acq} = \sum_{k=1}^K N_k \cdot F_k(P_k) \cdot \left[ R_k - c_k^{proc} - P_k \right]$$

Assuming reservation prices in Tier $k$ follow a uniform distribution over $[a_k, b_k]$, the supply function is $F_k(P_k) = \frac{P_k - a_k}{b_k - a_k}$. Taking the first-order condition with respect to $P_k$:
$$\frac{\partial \Pi_{acq}}{\partial P_k} = \frac{N_k}{b_k - a_k} \left[ (R_k - c_k^{proc} - P_k) - (P_k - a_k) \right] = 0$$
$$P_k^* = \frac{R_k - c_k^{proc} + a_k}{2}$$
subject to the economic participation constraint $a_k \le P_k^* \le b_k$ and the non-negative marginal margin condition $R_k - c_k^{proc} \ge a_k$. If $R_k - c_k^{proc} < a_k$, Tier $k$ is economically non-viable and the optimal policy sets $P_k^* = 0$ (reject or scrap at gate).

#### Core Deposit and Surcharge Contract Mechanics

In capital equipment and commercial aftermarket ecosystems, OEMs rarely rely solely on voluntary buybacks. Instead, they deploy a **Core Deposit / Surcharge Contract**. When a commercial buyer purchases a replacement unit (whether new or remanufactured), the invoice lists two separate line items: the net purchase price $P_{net}$ and a mandatory core deposit $D_{core}$.

The total cash outflow at transaction time is:
$$P_{gross} = P_{net} + D_{core}$$

The core deposit $D_{core}$ is fully refunded if and only if the customer returns an equivalent spent core meeting minimum acceptance criteria within a contractual return window $T_{horizon}$ (typically 30 to 60 days). Let $c_{trans}$ be the customer's internal logistics and handling friction cost to return the core. A rational fleet operator returns the core if:
$$D_{core} - c_{trans} \ge V_{scrap}$$
where $V_{scrap}$ is the local scrap metal melt value.

If the OEM sets $D_{core} \le V_{scrap} + c_{trans}$, core holders abandon the deposit, sell the core to local scrap merchants, and the remanufacturer starves of feedstock. If the OEM sets $D_{core}$ excessively high, buyers face cash flow friction and seek unbundled third-party alternatives. The unrefunded core deposit creates an **unearned core deposit liability** on the OEM balance sheet. If unreturned after $T_{horizon}$, the deposit is recognized as forfeited breakage income, but at the cost of permanent loss of that physical core asset.

### 5.2 Reverse BOM Disassembly Graph Analytics and Sequencing

Forward assembly is a converging, tree-structured operation: multiple child parts merge into a single parent subassembly. Disassembly is a diverging, non-linear directed hypergraph known as an **AND/OR Disassembly Graph** $G = (V, E)$.

```text
                            [Complete Core Asset: Root V0]
                                          |
                        +-----------------+-----------------+
                        | (Operation O1: Unbolt Housing)    | (Operation O2: Shear Shell)
                        v                                   v
             [AND Node: Joint Split]                 [AND Node: Joint Split]
             /                     \                 /                         [Sub-Assembly A1]       [Sub-Assembly A2]  [Comp C1: Damaged]    [Sub-Assembly A2]
          |                         |
    (Op O3: Teardown)         (Op O4: Press Pin)
          v                         v
     [Comp C2] [Comp C3]       [Comp C4] [Comp C5]
```

- **Vertices ($V$):** Represent physical states of the asset, ranging from the complete intact product (the root node $V_0$), down through intermediate sub-assemblies, to individual components and fasteners (terminal leaf nodes).
- **Hyper-edges / Operations ($E$ or $\mathcal{O}$):** Represent physical disassembly tasks $o \in \mathcal{O}$. Each operation $o$ requires a specific labor time $t_o$, a tool setup cost, and carries an operational risk of damaging the components $\pi_o \in [0, 1]$.
- **AND Nodes:** Represent joint disassembly operations where executing a single task $o$ simultaneously yields two or more disjoint sub-components: $\text{Tail}(o) \to \{\text{Head}_1(o), \text{Head}_2(o), \dots\}$.
- **OR Nodes:** Represent alternative operational methods to disassemble the same sub-assembly. For example, a frozen fastener can be removed via non-destructive induction heating and ultrasonic unscrewing ($o_{nd}$), OR via destructive plasma cutting / EDM drilling ($o_d$). Destructive operations have lower labor time ($t_{o,d} < t_{o,nd}$) but destroy the fastener and risk thermal damage to the adjacent housing.

#### Selective Disassembly Optimization Formulation

In high-throughput remanufacturing, disassembling an asset down to its smallest washer is rarely profit-maximizing. The labor cost of unbolting corroded, low-value brackets frequently exceeds their scrap or replacement value. The enterprise must execute **selective disassembly**: determining the optimal depth of teardown.

Let $\mathcal{C}$ be the set of all recoverable components. For each component $c \in \mathcal{C}$, let $V_c$ be its net market value (either resale value or avoided virgin procurement cost), and let $x_c \in \{0, 1\}$ be a binary decision variable indicating whether component $c$ is recovered. For each disassembly operation $o \in \mathcal{O}$, let $c_o = t_o \cdot w_{labor} + c_{tool}$ be the direct cost of executing the operation, and let $y_o \in \{0, 1\}$ be a binary decision variable indicating whether operation $o$ is performed.

The Selective Disassembly Optimization problem is formulated as a Mixed-Integer Linear Program (MILP):
$$\max_{\mathbf{x}, \mathbf{y}} \Pi_{dis} = \sum_{c \in \mathcal{C}} V_c \cdot y_{release}(c) \cdot x_c - \sum_{o \in \mathcal{O}} c_o \cdot y_o$$

Subject to:
1. **Precedence Constraints:** An operation $o$ can only be performed if all preceding operations $\mathcal{P}(o)$ required to physically expose the component have been completed:
   $$y_o \le y_p \quad \forall p \in \mathcal{P}(o)$$
2. **Component Release Constraints:** A component $c$ can only be harvested if at least one operation that separates it from its parent assembly has been executed:
   $$x_c \le \sum_{o \in \mathcal{R}(c)} y_o \quad \forall c \in \mathcal{C}$$
   where $\mathcal{R}(c)$ is the set of operations that release component $c$.
3. **Alternative Path Mutex (OR Nodes):** If an assembly can be separated by operation $o_1$ OR operation $o_2$, at most one method may be selected:
   $$y_{o_1} + y_{o_2} \le 1$$

#### Optimal Stopping Rule

In dynamic manual or robotic teardown, technicians apply an optimal stopping rule derived from the marginal Bellman equation. Disassembly must halt along a branch when the expected marginal value of the next operation $\Delta V_o$ is strictly less than the operational cost plus the risk-adjusted damage to the parent assembly:
$$\Delta V_o = \sum_{k \in \text{Children}(o)} V_k \cdot E[Y_k] - c_o - \pi_o \cdot V_{parent} \le 0$$
If $\Delta V_o \le 0$, the subassembly is preserved intact and routed directly to subassembly-level testing or sold as an integrated module.

#### Branch-and-Bound and State-Space Graph Traversal

Because the AND/OR disassembly graph contains both alternative paths (OR) and joint dependencies (AND), finding the globally optimal selective disassembly sequence is an NP-hard combinatorial problem. Let $S_0$ denote the initial root state (fully assembled product), and let $S_t$ denote the set of liberated parts and intact subassemblies at stage $t$.

The state-space transition is governed by:
$$S_{t+1} = (S_t \setminus \{\text{Parent}(o)\}) \cup \text{Children}(o)$$

To avoid evaluating all $O(2^{|\mathcal{O}|})$ disassembly combinations, practical disassembly planning algorithms deploy **Branch-and-Bound search with topological sorting**. A node in the search tree is pruned whenever its upper bound on remaining recoverable value $UB(S_t)$ is strictly less than the current best complete solution (Incumbent):
$$UB(S_t) = \sum_{c \in S_t} V_c + \sum_{k \in \mathcal{U}(S_t)} \max(0, V_k - c_k^{min}) < \Pi_{incumbent}$$
where $\mathcal{U}(S_t)$ is the set of unharvested components physically reachable from state $S_t$, and $c_k^{min}$ is the minimum cumulative operation cost to liberate component $k$.

### 5.3 Stochastic Yield Modeling and Core-to-Demand Matching

Once components are liberated, their operational integrity is uncertain. Unlike virgin manufacturing where part quality is binomially distributed with very low defect rates ($p < 0.001$), remanufactured component yields follow continuous, wide probability distributions.

Let the recovery yield of component type $k$ across disassembled cores be a random variable $Y_k \in [0, 1]$. We model $Y_k$ using a Beta distribution parameterized by shape parameters $(\alpha_k, \beta_k)$:
$$f(y_k) = \frac{1}{B(\alpha_k, \beta_k)} y_k^{\alpha_k - 1} (1 - y_k)^{\beta_k - 1}$$
$$E[Y_k] = \frac{\alpha_k}{\alpha_k + \beta_k}, \quad \text{Var}(Y_k) = \frac{\alpha_k \beta_k}{(\alpha_k + \beta_k)^2 (\alpha_k + \beta_k + 1)}$$

When incoming core vintages vary, a Dirichlet-multinomial distribution is used to model the multi-class condition vector (e.g., % Grade A, % Grade B, % Scrap).

#### The Unbalanced Bundle Dilemma

Consider a product whose BOM requires $m_k$ units of component $k$ per finished good. Disassembling $Q_C$ raw cores yields an available component supply vector $\mathbf{S}^{comp}$:
$$S_k^{comp} = Q_C \cdot Y_k \cdot m_k$$

Because components experience different physical stresses during life, their yields are asymmetric: $E[Y_{\text{block}}] \approx 0.90$, while $E[Y_{\text{crankshaft}}] \approx 0.60$, and $E[Y_{\text{camshaft}}] \approx 0.45$. If the remanufacturer attempts to assemble finished goods using *only* harvested components, the maximum achievable production volume $Q_R^{pure}$ is constrained by the **minimum yield bottleneck**:
$$Q_R^{pure} = Q_C \cdot \min_k \left( \frac{Y_k}{m_k} \right)$$

For all components $j$ where $Y_j > \min_k Y_k$, a physical surplus is generated:
$$\text{Surplus}_j = Q_C \cdot \left[ Y_j - \min_k Y_k \right]$$

This surplus triggers catastrophic inventory holding costs, capital lockup, and warehouse bloat.

#### Push Disassembly vs. Pull Disassembly

To resolve this imbalance, enterprises choose between two operational policies:

```text
+---------------------------------------------------------------------------------------------------+
| PUSH DISASSEMBLY POLICY:                                                                          |
| Inbound Cores (QC) ===> [Immediate 100% Disassembly] ===> Harvested Component Inventory (Sk)       |
|                                                                    |                              |
|                                                     Component Demand Match                        |
|                                                                    v                              |
|                                                     Assembly Line + Massive Surplus Inventory     |
+---------------------------------------------------------------------------------------------------+

+---------------------------------------------------------------------------------------------------+
| PULL DISASSEMBLY POLICY:                                                                          |
| Inbound Cores (QC) ===> [Raw Core Storage Buffer]                                                 |
|                                     |                                                             |
|                      Pull Signal: Triggered by Demand                                             |
|                                     v                                                             |
|            [Selective Disassembly] ===> Exact Assembly Match + Virgin Part Top-Up                 |
+---------------------------------------------------------------------------------------------------+
```

- **Push Disassembly:** Cores are stripped immediately upon receipt. Harvested parts are cleaned, inspected, and placed into component inventory.
  - *Advantage:* Component availability is immediate; lead time to assemble finished goods is minimized.
  - *Disadvantage:* High inventory holding costs; high warehouse space consumption; components risk oxidation, corrosion, and technical obsolescence while sitting on shelves.
- **Pull Disassembly:** Cores remain fully assembled in raw storage gaylords or yard racking. Cores are scheduled for teardown only when downstream demand for specific finished assemblies or critical spare parts emerges.
  - *Advantage:* Minimal component warehouse space; cores act as compact, protected storage for parts; preserves optionality to sell intact cores.
  - *Disadvantage:* Longer order fulfillment lead times; if teardown reveals defective components, immediate stockouts occur unless virgin parts are air-freighted.

**Mathematical Decision Boundary:** Let $h_c$ be the unit holding cost of an intact core per period, let $h_k$ be the holding cost of harvested component $k$, and let $c_{hold}^{push} = \sum_k h_k$ be the aggregate component holding cost. Pull disassembly dominates Push disassembly whenever:
$$\sum_k h_k - h_c > \frac{E[\text{Stockout Penalty}] \cdot \Pr(Y_k < \text{Threshold})}{\text{Lead Time Flexibility Factor}}$$
In heavy capital goods where $\sum h_k / h_c \approx 2.5 - 4.0$ due to binning, rust prevention, and packaging of individual parts, Pull disassembly is structurally superior.

#### Comparative Evaluation of Disassembly Control Policies

| Evaluation Dimension | Push Disassembly | Pull Disassembly | Hybrid Disassembly-to-Order (DTO) |
|---|---|---|---|
| **Trigger Mechanism** | Inbound core arrival | Customer finished good demand | Safety stock reorder point on bottleneck parts |
| **Storage Medium** | Harvested individual components | Intact core gaylords / yard racks | Core buffer + high-turn parts supermarket |
| **Inventory Holding Cost** | Highest ($h_{comp} \approx 3 \times h_{core}$) | Lowest (compact core footprint) | Intermediate (controlled buffer sizing) |
| **Component Obsolescence** | Extreme; unneeded parts sit on shelves | Negligible; cores preserve parts | Low; only bottleneck parts harvested early |
| **Fulfillment Lead Time** | Shortest (parts immediately available) | Longest (teardown on critical path) | Fast for standard BOMs; moderate for customs |
| **Material Handling Labor** | Highest (binning, bagging, shelf tagging) | Lowest (single core movement) | Balanced cellular routing |
| **Yield Uncertainty Buffer** | Excess safety stock in piece parts | Virgin part expediting buffer | Dynamic virgin top-up trigger limits |

#### Multi-Period Joint Production Planning Formulation

To bridge the gap between stochastic recovery yield and deterministic customer demand $D_t$, the enterprise procures virgin top-up components $Q_{kt}^{vir}$ at unit cost $c_k^{vir}$. Let $Q_t^R$ be remanufactured finished goods produced in period $t$, $Q_t^C$ be cores disassembled, and $I_{kt}^{comp}$ be component inventory. The dynamic optimization model is:
$$\min \sum_{t=1}^T \left[ c_C Q_t^C + c_{dis} Q_t^C + c_{reman} Q_t^R + \sum_k c_k^{vir} Q_{kt}^{vir} + h_c I_t^C + \sum_k h_k I_{kt}^{comp} + p_{loss} (D_t - Q_t^R)^+ \right]$$

Subject to:
$$I_t^C = I_{t-1}^C + S_t^{acq} - Q_t^C$$
$$I_{kt}^{comp} = I_{k,t-1}^{comp} + E[Y_k] m_k Q_t^C + Q_{kt}^{vir} - m_k Q_t^R \quad \forall k$$
$$Q_t^R \le D_t$$
$$Q_t^C, Q_t^R, Q_{kt}^{vir}, I_t^C, I_{kt}^{comp} \ge 0$$

### 5.4 Remanufacturing Capacity Planning and Line Balancing

Manufacturing engineers trained in forward assembly design lines assuming normal cycle time distributions with low coefficients of variation ($c_s = \sigma / \mu \approx 0.05 - 0.15$). In remanufacturing, cycle times exhibit extreme variance ($c_s \in [1.0, 2.5]$) due to three physical factors:
1. **Contamination Heterogeneity:** Cleaning an engine block with light surface dust requires a 10-minute aqueous ultrasonic bath. Cleaning an engine block with heavy carbon coking and baked sludge requires a 90-minute cycle of thermal pyrolysis burn-off followed by steel-shot blasting.
2. **Wear and Dimensional Distortion:** Re-machining a cylinder bore with minimal taper requires a single fast CNC honing pass. A severely warped or scored cylinder bore requires installing cast iron cylinder liners, boring, micro-grinding, and three verification passes.
3. **Multi-Pass Diagnostic Rework Loops:** If a remanufactured printed circuit board or hydraulic valve fails initial pressure testing, it does not leave the system; it loops back to manual diagnosis and re-soldering.

#### Line Congestion and Kingman's Heavy-Traffic Approximation

Consider a remanufacturing workstation (such as a precision CNC bore grinder or dynamic rotor balancer) operating as a $G/G/m$ queueing system with $m$ identical machines, arrival rate $\lambda$, mean service time $\tau = 1/\mu$, arrival coefficient of variation $c_a$, and service time coefficient of variation $c_s$. Workstation utilization is $\rho = \frac{\lambda}{m \mu}$.

Under heavy-traffic conditions, the expected waiting time in queue $W_q$ is given by Kingman's generalized approximation:
$$W_q \approx \left( \frac{c_a^2 + c_s^2}{2} \right) \left( \frac{\rho^{\sqrt{2(m+1)} - 1}}{m(1 - \rho)} \right) \tau$$

The expected work-in-process inventory in queue is given by Little's Law:
$$L_q = \lambda \cdot W_q$$

Notice the critical variance multiplier:
$$V_{mult} = \frac{c_a^2 + c_s^2}{2}$$

In a forward line where $c_a = 0.2$ and $c_s = 0.1$:
$$V_{mult}^{forward} = \frac{0.04 + 0.01}{2} = 0.025$$

In a remanufacturing cell where $c_a = 0.8$ (pulsed arrivals from batch disassembly) and $c_s = 1.5$ (heavy wear variance):
$$V_{mult}^{reman} = \frac{0.64 + 2.25}{2} = 1.445$$

The variance multiplier is **57.8 times larger** in remanufacturing. If an operations manager runs the remanufacturing line at forward manufacturing utilization targets (say, $\rho = 0.90$), the queue explodes. The line collapses into WIP congestion, parts block aisleways, cycle lead times stretch from days to months, and priority expediting wrecks production schedules.

> **Note:** Remanufacturing facilities must intentionally plan operating utilization targets 15% to 25% lower than forward manufacturing lines (targeting $\rho \in [0.65, 0.75]$) or deploy decoupling WIP buffer supermarkets sized specifically for $c_s > 1.2$.

#### Dynamic Bottleneck Migration

Because core wear is stochastic, workstation workloads shift dynamically:
- When incoming core lots are heavily corroded (e.g., winter salt exposure on commercial vehicle axles), the chemical wash and grit-blasting cell runs at $\rho = 0.98$ and becomes the system bottleneck, while downstream precision CNC machines sit starved.
- When incoming core lots have severe internal mechanical wear but clean exteriors, the cleaning cell drops to $\rho = 0.40$, while the precision re-machining and sleeve-insertion cell surges to $\rho = 0.95$, becoming the new bottleneck.

To maintain line balance, remanufacturers must deploy **cross-trained flexible labor** capable of moving between cleaning, machining, and assembly, combined with **dynamic pull buffers (CONWIP - Constant Work In Process)** that throttle line input based on the current bottleneck station's queue.

#### CONWIP Control and Buffer Sizing for Remanufacturing Lines

To prevent the catastrophic queue explosions predicted by Kingman's formula ($W_q \propto \frac{c_a^2 + c_s^2}{2}$), high-performing remanufacturing plants reject traditional push dispatching and implement **Constant Work-in-Process (CONWIP)** pull loops.

Under CONWIP, a fixed number of physical production authorization cards $W$ circulate between the final test stand and the initial core disassembly cell. A core cannot be stripped until a finished unit passes final testing and releases a CONWIP card.

The optimal card count $W^*$ is derived from Little's Law evaluated at the critical workcenter bottleneck:
$$W^* = \text{TH}^* \times (T_0 + W_q^*)$$
where $\text{TH}^*$ is the target line throughput (units/hour), $T_0 = \sum_j \tau_j$ is the raw processing time across all workstations, and $W_q^*$ is the allowable queue waiting time under target buffer space. By capping total system WIP at $W^*$, CONWIP transforms cycle time variance into queueing delays outside the plant gate rather than gridlock on the production floor, maintaining line velocity and preventing tool-starvation deadlocks.

#### Dedicated vs. Hybrid Manufacturing Lines

Enterprises producing both new and remanufactured products face an architectural choice:

| Dimension | Dedicated Remanufacturing Line | Hybrid Co-Located Line (New + Reman) |
|---|---|---|
| **Capital Investment** | High; requires duplicate machine tools, test stands, and cleaning tanks. | Low; shares high-capital CNC mills, coordinate measuring machines (CMMs), and test dynamometers. |
| **Operational Complexity** | Low; remanufacturing variance is isolated from new production. | Extreme; high reman cycle-time variance can starve or block high-speed new production. |
| **Contamination Risk** | Zero; dirty cores never enter pristine new assembly cleanrooms. | Severe; grit, carbon, and metal chips from core teardown risk contaminating new tolerances. |
| **Asset Utilization** | Lower; subject to core intake volume seasonality and shortages. | High; line balances shifts by running new production when cores are scarce. |
| **Workforce Skill** | Craft / diagnostic specialists capable of adaptive salvage machining. | Standardized assembly technicians operating rigid standard operating procedures (SOPs). |

The empirical rule of thumb: If remanufacturing processing variance $c_s > 0.8$, dedicated lines dominate. If $c_s < 0.4$ (as in modular electronics or cell phone screen swaps) and capital cost is high, hybrid lines are economically optimal provided physical cleaning is strictly quarantined.

### 5.5 Market Cannibalization and Closed-Loop Dual-Market Pricing

Selling remanufactured products alongside new products creates market cannibalization. To analyze this trade-off, we deploy the standard closed-loop supply chain microeconomic model (Ferguson & Toktay 2006; Atasu, Sarvary & Van Wassenhove 2008).

#### Consumer Utility and Market Segmentation

Consider a market of heterogeneous consumers with population size normalized to $M = 1$. Each consumer is characterized by a willingness-to-pay parameter for a brand-new product, $\theta$, uniformly distributed on the unit interval:
$$\theta \sim U[0, 1]$$

Consumers evaluate two product offerings:
- **New Product:** Priced at $p_n$, unit manufacturing cost $c_n$. Delivers perceived quality normalized to $1.0$.
- **Remanufactured Product:** Priced at $p_r$, unit remanufacturing cost $c_r$ (where $c_r < c_n$). Delivers perceived quality $\delta \in (0, 1)$, where $\delta$ represents the market-wide willingness-to-pay discount factor for remanufactured goods.

A consumer of type $\theta$ derives surplus utility:
$$U_n(\theta) = \theta - p_n \quad \text{(if buying new)}$$
$$U_r(\theta) = \delta \theta - p_r \quad \text{(if buying remanufactured)}$$
$$U_0 = 0 \quad \text{(if buying neither)}$$

Consumers maximize utility. A consumer prefers buying new over remanufactured if $U_n(\theta) \ge U_r(\theta)$:
$$\theta - p_n \ge \delta \theta - p_r \iff (1 - \delta)\theta \ge p_n - p_r \iff \theta \ge \frac{p_n - p_r}{1 - \delta} \equiv \theta_{nr}$$

A consumer prefers buying remanufactured over buying nothing if $U_r(\theta) \ge 0$:
$$\delta \theta - p_r \ge 0 \iff \theta \ge \frac{p_r}{\delta} \equiv \theta_{r0}$$

For both products to have positive demand in the market ($D_n > 0$ and $D_r > 0$), we must have $0 < \theta_{r0} < \theta_{nr} < 1$. Under uniform distribution of $\theta$, the demand functions are:
$$D_n(p_n, p_r) = 1 - \theta_{nr} = 1 - \frac{p_n - p_r}{1 - \delta}$$
$$D_r(p_n, p_r) = \theta_{nr} - \theta_{r0} = \frac{p_n - p_r}{1 - \delta} - \frac{p_r}{\delta}$$

```text
  0                    theta_r0                     theta_nr                   1
  +-----------------------+----------------------------+-----------------------+
  |    Buys Nothing       |    Buys Remanufactured     |       Buys New        |
  |    (Surplus < 0)      |    (Ur > Un and Ur >= 0)   |       (Un >= Ur)      |
  +-----------------------+----------------------------+-----------------------+
```

Notice the aggregate market size expansion:
$$D_{total} = D_n + D_r = 1 - \theta_{r0} = 1 - \frac{p_r}{\delta}$$
Introducing the remanufactured product expands total customer reach by selling to lower-valuation consumers in the segment $[	heta_{r0}, \theta_{nr}]$ who would never pay $p_n$ for a new product.

#### The Cannibalization Rate

The marginal cannibalization rate $\gamma$ measures how many new unit sales are destroyed for every additional remanufactured unit sold:
$$\gamma = \frac{\partial D_n / \partial p_r}{|\partial D_r / \partial p_r|}$$
Computing the partial derivatives:
$$\frac{\partial D_n}{\partial p_r} = \frac{1}{1 - \delta}$$
$$\frac{\partial D_r}{\partial p_r} = -\left( \frac{1}{1 - \delta} + \frac{1}{\delta} \right) = -\frac{1}{\delta(1 - \delta)}$$
$$\gamma = \frac{\frac{1}{1 - \delta}}{\frac{1}{\delta(1 - \delta)}} = \delta$$

This is a clean analytical result: **the marginal cannibalization rate equals exactly the consumer discount factor $\delta$**. If consumers perceive remanufactured quality to be 70% of new ($\delta = 0.70$), then 70% of remanufactured sales are stolen directly from primary new product sales, while only 30% represent genuine market expansion!

#### Joint Profit Maximization and Unconstrained Equilibrium

The OEM maximizes total corporate profit across both channels:
$$\max_{p_n, p_r} \Pi(p_n, p_r) = (p_n - c_n) D_n(p_n, p_r) + (p_r - c_r) D_r(p_n, p_r)$$

Taking partial derivatives with respect to $p_n$ and $p_r$:
$$\frac{\partial \Pi}{\partial p_n} = D_n - \frac{p_n - c_n}{1 - \delta} + \frac{p_r - c_r}{1 - \delta} = 1 - \frac{2p_n - 2p_r - c_n + c_r}{1 - \delta} = 0$$
$$\frac{\partial \Pi}{\partial p_r} = D_r + \frac{p_n - c_n}{1 - \delta} - (p_r - c_r)\left( \frac{1}{1 - \delta} + \frac{1}{\delta} \right) = 0$$

Solving this linear system of first-order conditions simultaneously yields the unconstrained optimal prices:
$$p_n^* = \frac{1 + c_n}{2}$$
$$p_r^* = \frac{\delta + c_r}{2}$$

Remarkably, in the unconstrained dual-market monopoly, **the price of the new product $p_n^*$ is completely independent of the remanufacturing cost $c_r$ and the quality discount $\delta$**. The firm prices the new product exactly as if it were a single-product monopolist, and sets the remanufactured price $p_r^*$ proportionally to its discounted value and recovery cost.

#### Core-Constrained Equilibrium (Kuhn-Tucker Resolution)

In reality, remanufacturing volume is strictly constrained by the volume of available, recoverable cores:
$$D_r(p_n, p_r) \le Q_{core}^{cap}$$

When $D_r(p_n^*, p_r^*) > Q_{core}^{cap}$, the constraint binds. We formulate the Lagrangian:
$$\mathcal{L}(p_n, p_r, \lambda) = (p_n - c_n) D_n(p_n, p_r) + (p_r - c_r) D_r(p_n, p_r) - \lambda \left( D_r(p_n, p_r) - Q_{core}^{cap} \right)$$

Under binding core supply, the remanufacturer cannot satisfy unconstrained demand. To clear the market at exactly $Q_{core}^{cap}$, the firm must raise $p_r$ above its unconstrained price. Setting $D_r(p_n, p_r) = Q_{core}^{cap}$ and solving the KKT system yields:
$$p_n^{**} = \frac{1 + c_n}{2}$$
$$p_r^{**} = \delta \left[ \frac{1 + c_n}{2} - (1 - \delta) Q_{core}^{cap} \right]$$
$$D_n^{**} = 1 - Q_{core}^{cap} - \left[ \frac{1 + c_n}{2} - (1 - \delta) Q_{core}^{cap} \right] = \frac{1 - c_n}{2} - \delta Q_{core}^{cap}$$

The shadow price of core availability $\lambda$ is:
$$\lambda^* = \frac{p_r^{**} - p_r^*}{\delta(1 - \delta)}$$

**Economic Implication:** When core feedstock is scarce, the OEM raises the remanufactured price $p_r^{**}$. This price increase has a dual benefit: it extracts scarcity rents from secondary buyers and pushes marginal buyers back into the primary new product market ($D_n^{**} > D_n^*$). Scarcity in reverse logistics acts as an organic protective shield against cannibalization.

---

## 6. Worked example

### Company Profile and Operational Baseline

**Apex Powertrain Solutions** remanufactures heavy-duty commercial vehicle turbochargers for Class 8 freight trucks. Apex operates a primary OEM manufacturing line producing new turbochargers, alongside an industrial remanufacturing facility that processes returned turbocharger cores.

### Baseline Parameters

- **Market Size:** Normalized quarterly fleet replacement demand $M = 10,000$ turbochargers.
- **Maximum Fleet Reservation Price:** $V_{max} = \$1,000.00$ per turbocharger.
- **New Turbocharger Manufacturing Cost:** $c_n = \$450.00$ per unit (normalized $c_n = 0.45$).
- **Perceived Quality Discount Factor:** Fleet managers discount remanufactured turbochargers by $\delta = 0.65$ relative to new units.
- **Quarterly Core Inbound Volume:** Apex collects $Q_C = 1,429$ raw returned cores per quarter through its national commercial dealership network.

```text
                         [Raw Inbound Cores: QC = 1,429 units]
                                           |
                 +-------------------------+-------------------------+
                 |                         |                         |
       Tier 1: 40% (572 units)   Tier 2: 45% (643 units)   Tier 3: 15% (214 units)
       Buyback: $120.00/core     Buyback: $80.00/core      Buyback: $30.00/core
       Yield: 90% (515 good)     Yield: 65% (418 good)     Yield: 30% (64 good)
                 |                         |                         |
                 +-------------------------+-------------------------+
                                           |
                           Total Usable Cores = 997 ~ 1,000 units
                                           |
                             (Reverse BOM Teardown & Re-machining)
                                           v
                             Unit Reman Cost: cr = $220.00
```

### Step 1: Core Acquisition Pricing and Recovery Yield Accounting

Apex segments inbound cores into three verified inspection tiers:
- **Tier 1 (Fleet Preventative Pulls, <150k miles):** $P_1 = \$120.00$ buyback payout; accounts for 40% of inbound volume (572 cores). Inspection yield into certified remanufacturing is 90% ($572 \times 0.90 = 515$ usable units).
- **Tier 2 (Operational Failure / Oil Leak, 150k–300k miles):** $P_2 = \$80.00$ buyback payout; accounts for 45% of inbound volume (643 cores). Inspection yield is 65% ($643 \times 0.65 = 418$ usable units).
- **Tier 3 (Catastrophic Thermal Seizure, >300k miles):** $P_3 = \$30.00$ buyback payout; accounts for 15% of inbound volume (214 cores). Inspection yield is 30% ($214 \times 0.30 = 64$ usable units).

**Arithmetic:**
- **Blended Acquisition Payout per Inbound Core:**
  $$\bar{P} = (0.40 \times 120.00) + (0.45 \times 80.00) + (0.15 \times 30.00) = 48.00 + 36.00 + 4.50 = \$88.50 \text{ per core}$$
- **Blended Usable Recovery Yield:**
  $$\bar{Y} = (0.40 \times 0.90) + (0.45 \times 0.65) + (0.15 \times 0.30) = 0.36 + 0.2925 + 0.045 = 0.6975 \approx 70.0\%$$
- **Total Net Usable Remanufactured Units Available:**
  $$Q_{core}^{cap} = 1,429 \times 0.6975 = 996.7 \approx 1,000 \text{ units per quarter}$$
- **Effective Inbound Core Cost per Recovered Usable Unit:**
  $$c_{core}^{eff} = \frac{\$88.50}{0.70} = \$126.43 \text{ per recovered core}$$

### Step 2: Reverse BOM Disassembly and Remanufacturing Unit Cost

Every usable core undergoes selective disassembly and precision reconditioning across four primary sub-assemblies:
1. **Turbine Housing (Ductile Ni-Resist Iron):** Chemical stripping, ultrasonic de-coking, precision CNC flange face-milling: $\$38.57$ per unit.
2. **Center Housing Rotating Assembly (CHRA):** High-pressure blast cleaning, bore honing, bronze sleeve insert installation: $\$35.00$ per unit.
3. **Compressor Wheel & Shaft:** Re-machining journal surfaces, micro-grinding: Included in CHRA.
4. **Virgin Replacement Kit:** Mandatory replacement of dynamic wear parts (hydrodynamic journal bearings, piston ring oil seals, thrust collars, stainless fasteners, O-rings): $\$20.00$ per unit.

**Total Effective Remanufacturing Unit Cost ($c_r$):**
$$c_r = c_{core}^{eff} + c_{clean/dis} + c_{machining} + c_{virgin\_parts}$$
$$c_r = \$126.43 + \$38.57 + \$35.00 + \$20.00 = \$220.00 \text{ per unit}$$
Normalized cost: $c_r = \frac{\$220.00}{\$1,000.00} = 0.22$.

### Step 3: Unconstrained Dual-Market Pricing Benchmark

First, calculate the unconstrained pricing policy if Apex had unlimited core supply:
$$p_n^* = \frac{1 + c_n}{2} \times V_{max} = \frac{1 + 0.45}{2} \times 1,000 = 0.725 \times 1,000 = \$725.00$$
$$p_r^* = \frac{\delta + c_r}{2} \times V_{max} = \frac{0.65 + 0.22}{2} \times 1,000 = \frac{0.87}{2} \times 1,000 = 0.435 \times 1,000 = \$435.00$$

**Market Demands under Unconstrained Pricing:**
$$\theta_{nr} = \frac{p_n - p_r}{(1 - \delta) V_{max}} = \frac{725.00 - 435.00}{(1 - 0.65) \times 1,000} = \frac{290.00}{350.00} = 0.82857$$
$$\theta_{r0} = \frac{p_r}{\delta V_{max}} = \frac{435.00}{0.65 \times 1,000} = \frac{435.00}{650.00} = 0.66923$$
$$D_n^* = M \times (1 - \theta_{nr}) = 10,000 \times (1 - 0.82857) = 1,714.3 \text{ units}$$
$$D_r^* = M \times (\theta_{nr} - \theta_{r0}) = 10,000 \times (0.82857 - 0.66923) = 1,593.4 \text{ units}$$

**Profit Evaluation:**
$$\Pi_{new}^* = (725.00 - 450.00) \times 1,714.3 = \$275.00 \times 1,714.3 = \$471,432.50$$
$$\Pi_{reman}^* = (435.00 - 220.00) \times 1,593.4 = \$215.00 \times 1,593.4 = \$342,581.00$$
$$\Pi_{total}^* = \$471,432.50 + \$342,581.00 = \$814,013.50$$

**Deficit Diagnosis:** Unconstrained remanufacturing demand is $1,593.4$ units, but available core supply yields only $Q_{core}^{cap} = 1,000.0$ units. Unconstrained pricing creates a **shortage of 593.4 cores**. If Apex attempts to hold prices at $\$435.00$, it will stock out, alienate fleet buyers, and forfeit pricing power.

### Step 4: Karush-Kuhn-Tucker (KKT) Core-Constrained Pricing

To clear the market at exactly $Q_{core}^{cap} = 1,000$ units (normalized $q_{core} = 1,000 / 10,000 = 0.10$), Apex applies the constrained equilibrium:
$$p_n^{**} = \frac{1 + c_n}{2} \times V_{max} = \$725.00 \quad \text{(New price remains unchanged)}$$
$$x^* = \frac{1 + c_n}{2} - (1 - \delta) q_{core} = 0.725 - (1 - 0.65) \times 0.10 = 0.725 - 0.035 = 0.690$$
$$p_r^{**} = (\delta \times x^*) \times V_{max} = (0.65 \times 0.690) \times 1,000 = 0.4485 \times 1,000 = \$448.50$$

Apex increases the remanufactured price by $\$13.50$ (from $\$435.00$ to $\$448.50$).

**Recalculate Constrained Market Demands:**
$$\theta_{nr}^{**} = \frac{725.00 - 448.50}{350.00} = \frac{276.50}{350.00} = 0.7900$$
$$\theta_{r0}^{**} = \frac{448.50}{650.00} = 0.6900$$
$$D_n^{**} = 10,000 \times (1 - 0.7900) = 2,100.0 \text{ units}$$
$$D_r^{**} = 10,000 \times (0.7900 - 0.6900) = 1,000.0 \text{ units} \quad \text{(Matches core capacity exactly)}$$

**Financial Results under Constrained Pricing:**
$$\Pi_{new}^{**} = (725.00 - 450.00) \times 2,100.0 = \$275.00 \times 2,100.0 = \$577,500.00$$
$$\Pi_{reman}^{**} = (448.50 - 220.00) \times 1,000.0 = \$228.50 \times 1,000.0 = \$228,500.00$$
$$\Pi_{total}^{**} = \$577,500.00 + \$228,500.00 = \$806,000.00$$

**Strategic Takeaway:** By raising the remanufactured price to match physical core supply, Apex drives 385.7 fleet managers who were priced out of remanufactured back into buying new turbochargers ($D_n$ surges from $1,714.3$ to $2,100.0$). Because the margin on new turbochargers ($\$275.00$) exceeds the margin on remanufactured turbochargers ($\$228.50$), total company profit drops by only $\$8,013.50$ (less than 1%) despite a 37% core supply deficit.

### Step 5: Capacity Planning and Line Congestion at Balancing Workcenter

Finished CHRA assemblies pass through a high-speed dynamic balancing and vibration testing cell prior to final casing packaging.
- Quarterly operating hours: 500 hours (60 operational days $\times$ 8.33 hours/day).
- Total remanufacturing throughput required: 1,000 units / 500 hours = arrival rate $\lambda = 2.0$ units per hour.
- The cell has 1 dedicated Schenck dynamic balancing machine tool ($m = 1$).
- Mean service time: $\tau = 24.0$ minutes = $0.40$ hours (service rate $\mu = 2.5$ units/hour).
- Operating utilization: $\rho = \frac{\lambda}{\mu} = \frac{2.0}{2.5} = 0.80$ (80.0%).
- Inbound arrival coefficient of variation: $c_a = 1.0$ (Poisson-like arrivals from batch cleaning).

**Contrast Forward vs. Remanufacturing Variance:**
1. **Forward Line:** Dynamic balancing of pristine new rotors has minimal shaft runout. Processing time is nearly deterministic: $c_s^{fwd} = 0.10$.
   $$W_q^{fwd} \approx \left( \frac{1.0^2 + 0.10^2}{2} \right) \left( \frac{0.80}{1 - 0.80} \right) 24.0 = \left( \frac{1.01}{2} \right) (4.0) (24.0) = 0.505 \times 96.0 = 48.48 \text{ minutes}$$
   $$L_q^{fwd} = \lambda \times W_q = 2.0 \times (48.48 / 60) = 1.62 \text{ units in buffer}$$
2. **Remanufacturing Line:** Due to shaft thermal warping and weld-overlay grinding variations, balancing requires between 1 and 4 grinding/weight-removal passes: $c_s^{reman} = 1.40$.
   $$W_q^{reman} \approx \left( \frac{1.0^2 + 1.40^2}{2} \right) \left( \frac{0.80}{1 - 0.80} \right) 24.0 = \left( \frac{1.0 + 1.96}{2} \right) (4.0) (24.0) = 1.48 \times 96.0 = 142.08 \text{ minutes (2.37 hours)}$$
   $$L_q^{reman} = \lambda \times W_q = 2.0 \times (142.08 / 60) = 4.74 \text{ units in buffer}$$

**Operational Conclusion:** Remanufacturing processing variance increases queue waiting time from $48.5$ minutes to **$142.1$ minutes (a 2.93x increase)**. The plant layout must allocate staging buffer space for at least 5 heavy CHRAs at the balancing machine to prevent upstream starvation, and line planners cannot push utilization above 80% without triggering queue explosions.

---
### Step 6: Strategic Regime Comparison and Financial P&L

To assess the macro-economic value of circular remanufacturing, Apex evaluates three operating regimes:
1. **Regime 1: Single-Product Monopoly (Forward Only):** No core recovery. Prices new units at $p_n^{mono} = \frac{1 + c_n}{2} \times V_{max} = \$725.00$. Demand is $D_n^{mono} = M(1 - p_n^{mono}/V_{max}) = 10,000(1 - 0.725) = 2,750.0$ units.
2. **Regime 2: Unconstrained Dual-Market Equilibrium:** Abundant core feedstock ($Q_C \to \infty$). $p_n^* = \$725.00, p_r^* = \$435.00$. Demands: $D_n^* = 1,714.3, D_r^* = 1,593.4$ units.
3. **Regime 3: Core-Constrained Dual-Market Equilibrium (Apex Actual):** Feedstock capped at 1,000 usable cores. $p_n^{**} = \$725.00, p_r^{**} = \$448.50$. Demands: $D_n^{**} = 2,100.0, D_r^{**} = 1,000.0$ units.

```text
+---------------------------------------------------------------------------------------------------+
| STRATEGIC REGIME COMPARISON (QUARTERLY P&L IN USD)                                                |
+------------------------------------+--------------------+--------------------+--------------------+
| Financial / Operational Metric     | Regime 1: Forward  | Regime 2: Unconstr | Regime 3: Constrain|
+------------------------------------+--------------------+--------------------+--------------------+
| New Selling Price (pn)             | $725.00            | $725.00            | $725.00            |
| Reman Selling Price (pr)           | N/A                | $435.00            | $448.50            |
| New Unit Demand (Dn)               | 2,750.0 units      | 1,714.3 units      | 2,100.0 units      |
| Reman Unit Demand (Dr)             | 0.0 units          | 1,593.4 units      | 1,000.0 units      |
| Total Market Volume (D_total)      | 2,750.0 units      | 3,307.7 units      | 3,100.0 units      |
| Primary Revenue                    | $1,993,750.00      | $1,242,867.50      | $1,522,500.00      |
| Secondary Recommerce Revenue       | $0.00              | $693,129.00        | $448,500.00        |
| Total Gross Revenue                | $1,993,750.00      | $1,935,996.50      | $1,971,000.00      |
| Cost of Goods Sold (COGS)          | $1,237,500.00      | $1,121,983.00      | $1,165,000.00      |
| Total Gross Profit                 | $756,250.00        | $814,013.50        | $806,000.00        |
| Gross Profit Margin %              | 37.93%             | 42.05%             | 40.89%             |
| Net Circular Profit Delta vs Fwd   | Base ($0.00)       | +$57,763.50 (+7.6%)| +$49,750.00 (+6.6%)|
+------------------------------------+--------------------+--------------------+--------------------+
```

Even under a binding 37% core deficit (Regime 3), closed-loop remanufacturing expands total units delivered from 2,750 to 3,100, lifts corporate gross margin from 37.93% to 40.89%, and adds nearly $\$50,000.00$ in quarterly gross profit compared to refusing to enter the circular economy.

---

## 7. Code, query, or calculation

The following self-contained Python script implements the closed-loop dual-market pricing optimization model under cannibalization and binding core availability constraints. It solves for unconstrained versus Karush-Kuhn-Tucker constrained equilibrium prices, demand allocations, cannibalization metrics, and core shadow values.

```python
import numpy as np

def optimize_closed_loop_pricing(
    v_max: float, cn: float, cr: float, delta: float, 
    m_market: float, q_core_cap: float
) -> dict:
    """Solves closed-loop dual-market pricing under cannibalization and core limits."""
    # Normalized economic parameters
    cn_norm, cr_norm = cn / v_max, cr / v_max
    q_norm = q_core_cap / m_market
    
    # Unconstrained analytical optimal prices (Ferguson & Toktay 2006)
    pn_unc_norm = (1.0 + cn_norm) / 2.0
    pr_unc_norm = (delta + cr_norm) / 2.0
    
    dn_unc_norm = 1.0 - (pn_unc_norm - pr_unc_norm) / (1.0 - delta)
    dr_unc_norm = (pn_unc_norm - pr_unc_norm) / (1.0 - delta) - pr_unc_norm / delta
    
    dr_unc = dr_unc_norm * m_market
    dn_unc = dn_unc_norm * m_market
    
    if dr_unc <= q_core_cap:
        # Core supply is abundant; unconstrained pricing clears
        is_constrained = False
        pn_opt = pn_unc_norm * v_max
        pr_opt = pr_unc_norm * v_max
        dn_opt, dr_opt = dn_unc, dr_unc
        lam_shadow = 0.0
    else:
        # KKT core availability constraint binds: Dr = q_core_cap
        is_constrained = True
        pn_opt = ((1.0 + cn_norm) / 2.0) * v_max
        x_val = (1.0 + cn_norm) / 2.0 - (1.0 - delta) * q_norm
        pr_opt = (delta * x_val) * v_max
        dn_opt = (1.0 - q_norm - x_val) * m_market
        dr_opt = q_core_cap
        lam_shadow = (pr_opt - pr_unc_norm * v_max) / (delta * (1.0 - delta) * v_max)

    profit_new = (pn_opt - cn) * dn_opt
    profit_reman = (pr_opt - cr) * dr_opt
    total_profit = profit_new + profit_reman
    cannibalization_rate = delta  # Marginal substitution rate dDn / |dDr|

    return {
        "constrained": is_constrained,
        "price_new": round(pn_opt, 2), "price_reman": round(pr_opt, 2),
        "demand_new": round(dn_opt, 1), "demand_reman": round(dr_opt, 1),
        "profit_new": round(profit_new, 2), "profit_reman": round(profit_reman, 2),
        "profit_total": round(total_profit, 2),
        "cannibalization_rate": round(cannibalization_rate, 4),
        "shadow_price_core": round(lam_shadow, 4)
    }

# Execute industrial case parameters:
results = optimize_closed_loop_pricing(
    v_max=1000.0, cn=450.0, cr=220.0, delta=0.65, 
    m_market=10000.0, q_core_cap=1000.0
)
for k, v in results.items():
    print(f"  {k}: {v}")
```

### Verification and Execution Output

When executed in Python 3.10+, the script outputs the following verified console metrics:

```text
  constrained: True
  price_new: 725.0
  price_reman: 448.5
  demand_new: 2100.0
  demand_reman: 1000.0
  profit_new: 577500.0
  profit_reman: 228500.0
  profit_total: 806000.0
  cannibalization_rate: 0.65
  shadow_price_core: 0.0593
```

The script verifies that the core constraint binds, sets the optimal remanufactured price at $\$448.50$ (matching the KKT derivation), confirms demand reaches the physical 1,000-unit ceiling, and shows a positive shadow price of $\lambda = 0.0593$, indicating each additional core acquired expands profit by $\$59.30$.

---

## 8. Data, standards, and regulatory context

Remanufacturing operates under a strict matrix of technical standards, environmental protection directives, consumer disclosure rules, and customs treaties.

### Technical and Environmental Standards

- **ANSI/RIC001.1-2016 (Standard for Information Required for Remanufacturing):** Issued by the Remanufacturing Industries Council (RIC) and accredited by ANSI. It establishes the benchmark definition of remanufacturing: an industrial process that restores an end-of-life core to "like-new" operating condition and performance. It mandates that a remanufactured asset must be completely disassembled, cleaned of all biological, chemical, and particulate contaminants, subjected to 100% inspection against OEM engineering tolerances, re-machined or replaced with genuine or approved parts, and tested to meet or exceed OEM performance specifications.
- **ISO 8887-1:2017 (Technical Product Documentation — Design for Manufacture, Assembling, Disassembling and End-of-Life Processing):** Defines formal geometric and mechanical standards for reverse engineering, modularity, and disassembly notation on engineering drawings. It standardizes the graphic language for non-destructive versus destructive joining methods.
- **ISO 14040/14044 (Life Cycle Assessment - LCA):** Establishes the carbon, water, and material accounting rules required to validate environmental claims. Remanufacturers must use attributional LCA models with cut-off or system-expansion rules to allocate primary material extraction credits between the first lifecycle and subsequent remanufacturing cycles.

### Binding Regulatory Mandates

```text
+---------------------------------------------------------------------------------------------------+
| REGULATORY COMPLIANCE MATRIX FOR CLOSED-LOOP REMANUFACTURING                                      |
+--------------------------+-----------------------+------------------------------------------------+
| Regulation               | Jurisdiction          | Operational Mandate                            |
+--------------------------+-----------------------+------------------------------------------------+
| EU ESPR (2024/1781)      | European Union        | Digital Product Passport (DPP), DfD standards  |
| EU Right to Repair (2024)| European Union        | 10-year spare parts & repair manual access     |
| US FTC Green Guides      | United States (FTC)   | 16 CFR § 260.11 "Remanufactured" disclosures   |
| Magnuson-Moss Act        | United States (15 USC)| Warranty parity; prohibits tie-in sales        |
| Basel Convention         | Global (UNEP)         | Transboundary hazardous waste export control   |
+--------------------------+-----------------------+------------------------------------------------+
```

1. **EU Ecodesign for Sustainable Products Regulation (ESPR, Regulation EU 2024/1781):** Enacted in 2024, ESPR replaces the 2009 Ecodesign Directive, expanding requirements beyond energy-using products to virtually all physical goods entering the EU market (commercial electronics, industrial machinery, automotive assemblies, textiles). ESPR introduces mandatory **Digital Product Passports (DPP)**: every product must carry a machine-readable data carrier (QR code / RFID) linking to a decentralized data record containing material composition, recycled content, disassembly sequence videos, expected durability, and remanufacturing history. Products failing to meet reparability and disassembly scores face market ban.
2. **EU Right to Repair Directive (Directive EU 2024/1799):** Mandates that manufacturers must provide independent remanufacturers and consumers access to original spare parts, diagnostic software, and repair manuals at reasonable, non-discriminatory prices for a minimum of 5 to 10 years after product discontinuation. Crucially, it prohibits software locking or serial number cryptographic pairing (parts pairing) that disables a remanufactured component when installed into a host chassis.
3. **U.S. Federal Trade Commission (FTC) Green Guides (16 CFR § 260.11):** Enforces truth-in-advertising standards for circular claims. An enterprise cannot market a product as "Remanufactured" or "Factory Rebuilt" unless the asset has been completely disassembled, cleaned, inspected with all worn parts replaced, and tested to meet original OEM performance standards. Marketing a repaired or cosmetically cleaned product as "Remanufactured" constitutes deceptive advertising subject to civil monetary penalties.
4. **Basel Convention on the Control of Transboundary Movements of Hazardous Wastes (1989 / 2019 Ban Amendment):** Severely restricts the international movement of used assets, e-waste, and lithium-ion batteries. Developing nations frequently classify uncharacterized electronic or automotive cores as hazardous waste. To import cores across borders for remanufacturing without customs seizure, enterprises must obtain formal Prior Informed Consent (PIC) or prove that cores are certified, functional feedstocks with commercial buyback contracts under Annex IX (B1110 classification).

---

## 9. Industry practice and case evidence

### Industrial Case Evidence

```text
+---------------------------------------------------------------------------------------------------+
| INDUSTRIAL BENCHMARKS IN CIRCULAR REMANUFACTURING                                                 |
+------------------+------------------------+-------------------------------------------------------+
| Enterprise       | Sector                 | Core Operating Mechanism                              |
+------------------+------------------------+-------------------------------------------------------+
| Caterpillar      | Heavy Earthmoving      | Cat Reman: Core deposit surcharge & laser cladding    |
| Renault Group    | Automotive             | Flins Re-Factory: Plant conversion to circular hub    |
| Apple Inc.       | Consumer Electronics   | Daisy/Dave robots: Automated selective disassembly    |
| Cummins Inc.     | Commercial Diesel      | Cummins ReCon: Global core banking & salvage machining|
| Xerox Corp.      | Office Systems         | Asset Recovery Management: 5-7 design lifecycles      |
+------------------+------------------------+-------------------------------------------------------+
```

- **Caterpillar (Cat Reman):** Caterpillar operates one of the world's largest industrial remanufacturing systems, processing over 90 million pounds of end-of-life material annually across nine global facilities. Caterpillar enforces core returns using a **Core Deposit / Surcharge business model**: when a customer purchases a remanufactured engine, cylinder head, or hydraulic pump, they are billed a deposit (often 30% to 50% of the replacement price) that is refunded only when the customer's used core is returned to a certified Cat dealer. The dealer performs standardized visual inspection using Cat's proprietary Core Acceptance Criteria (CAC). Metallurgically, Cat utilizes advanced salvage technology—including high-velocity oxygen-fuel (HVOF) thermal spraying, robotic laser cladding, and precision electroplating—to re-deposit metal onto worn crankshaft journals and cylinder bores, restoring them to original factory dimensions using less than 15% of the raw material mass of a new casting. *(Documented corporate reporting).*
- **Renault Group (Flins Re-Factory):** In 2021, Renault initiated the conversion of its historic Flins vehicle assembly plant in France into Europe's first dedicated circular economy industrial site, known as the **Re-Factory**. The facility combines four dedicated streams: *Re-Trofit* (converting commercial combustion vans to electric powertrains), *Re-Energy* (dismantling EV battery packs for second-life stationary energy storage), *Re-Cycle* (selective parts harvesting and closed-loop plastic compounding), and *Re-Start* (mechatronic prototyping). In 2024, the site processed over 30,000 used vehicles and 15,000 mechanical units annually, cutting reconditioning cycle times from 21 days to 6 days through synchronized lean cellular layouts. *(Documented operational case).*
- **Apple Inc. (Certified Refurbished & Automated Disassembly):** Apple separates its primary and secondary channels to prevent brand erosion while capturing recommerce value. Products sold via *Apple Certified Refurbished* receive a new outer enclosure, a pristine replacement battery, genuine OEM accessories, a standard 1-year limited warranty, and AppleCare eligibility, sold at a fixed 15% discount in distinct plain white packaging. In reverse operations, Apple engineered proprietary custom robotic disassembly lines named **Daisy** and **Dave**. Daisy selectively disassembles 15 different iPhone models at a rate of 200 phones per hour (1.2 million per year), using chilled air (-80°C) to shatter adhesive bonds, followed by robotic punch-and-shear heads to liberate logic boards, cameras, and haptic engines intact without shredding or cross-metal contamination. *(Documented corporate environmental progress reports).*
- **Cummins Inc. (Cummins ReCon):** Cummins remanufactures diesel and natural gas engines, turbochargers, and fuel injectors across facilities in North America, Europe, and Asia. Cummins maintains a computerized **Core Banking System** that tracks dealer core return accounts in real time. If a dealership returns a core with a cracked block that fails magnetic particle inspection, the system automatically assigns partial core credit based on recoverable sub-assemblies (crankshaft and cylinder heads), avoiding binary dispute deadlocks. Cummins maintains an engineering salvage library containing over 20,000 validated salvage repair procedures, utilizing friction welding and induction heat-treating to certify re-machined parts to current EPA emission standards. *(Documented case study).*

### Supply Chain Inference vs. Documented Fact

> **Note:** While OEMs publish aggregate circularity metrics (e.g., total tonnage remanufactured or energy saved), detailed internal yield distributions, dynamic pricing algorithms, and cannibalization rates are guarded as trade secrets. Published models in peer-reviewed literature represent validated empirical approximations of actual corporate practice.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign | Physical / Economic Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **Core Hoarding & Reverse Bullwhip** | Sudden collapse in core return volume followed by violent supply spikes. | Dealers or fleet operators stockpile cores on yard pallets waiting for annual buyback price promotions or volume rebates. | Remanufacturing line swings from idle starvation to severe warehouse WIP gridlock; holding costs soar. | Institute rolling core deposit expiration windows (e.g., 60-day return deadline) and eliminate volume-rebate spikes. |
| 2 | **The "Lemons" Death Spiral** | Average yield of usable castings drops month-over-month; scrap rate surges. | A uniform acquisition price under-compensates high-quality cores, driving good cores to third-party rebuilders while junk cores flood in. | Processing costs escalate; machine tools waste time on unrecoverable cores; reman margins collapse. | Implement multi-tier condition grading schedules with calibrated pre-acceptance inspection rubrics. |
| 3 | **Destructive Disassembly Over-Reach** | High replacement cost for damaged structural castings during teardown. | Technicians use pneumatic chisels or plasma torches to speed up teardown, cracking underlying aluminum housings. | Destroys the primary economic asset (the core casting); forces expensive virgin part replacements. | Establish strict SOPs for non-destructive induction heating; install torque-limited automated unbolting stations. |
| 4 | **Uncontrolled Cannibalization** | Primary new product sales drop sharply upon launch of remanufactured SKU. | Remanufactured product is priced too low ($\delta p_n \ll p_r$), or marketing fails to differentiate target customer segments. | Enterprise cannibalizes high-margin corporate sales; overall gross margin dollars decline. | Calibrate dual pricing via KKT model; restrict reman sales to secondary geographic or price-sensitive channels. |
| 5 | **Component Imbalance WIP Explosion** | Warehouse shelves fill with thousands of surplus engine blocks while assembly lines sit halted. | Unbalanced recovery yields ($Y_{\text{block}} \gg Y_{\text{crank}}$) under a push disassembly policy. | Millions of dollars in working capital locked in obsolete components; warehouse aisles blocked. | Transition from Push to Pull disassembly; dynamically top up bottleneck parts with scheduled virgin procurement. |
| 6 | **Dynamic Bottleneck Migration** | Workstations alternate between massive queue backups and complete starvation. | High cycle-time variance ($c_s > 1.2$) combined with seasonal shifts in core contamination and corrosion. | Production lead times stretch unpredictably; customer order fulfillment dates missed. | Deploy cross-trained flexible float workers; install decoupling CONWIP buffer supermarkets; de-rate utilization to $\le 75\%$. |
| 7 | **Sub-Surface Fatigue Escapes** | Remanufactured units experience early field failures under warranty (<90 days). | Inadequate non-destructive evaluation (NDE); visual inspection fails to detect internal subsurface fatigue micro-cracks. | Catastrophic warranty claims, double reverse logistics costs, and severe OEM brand equity destruction. | Mandate automated Eddy-current, magnetic particle, or ultrasonic resonance testing for all dynamic structural cores. |
| 8 | **Virgin Supplier Price Squeeze** | Escalating unit costs for the 10% to 20% virgin parts needed to complete reman assemblies. | Tier 1 component suppliers view remanufacturing as a competitor to their new part sales and price-gouge on spare parts. | Reman unit cost $c_r$ surges, eliminating the profit spread against new product manufacturing cost $c_n$. | Negotiate dual-use volume master contracts covering both new production and aftermarket reman spare parts. |
| 9 | **Transboundary Customs Seizure** | Core shipments impounded at international border ports for weeks. | Customs authorities classify incoming core containers as illegal hazardous e-waste or scrap under the Basel Convention. | Supply line starvation; demurrage fees; potential environmental regulatory fines. | Secure formal Prior Informed Consent (PIC); mark shipments with commercial invoices proving non-waste status under Basel B1110. |
| 10 | **Greenwashing False Advertising** | Regulatory inquiry or class-action lawsuit alleging deceptive consumer advertising. | Marketing department labels simple cleaned or repaired products as "Certified Remanufactured." | FTC enforcement penalties (16 CFR § 260.11), mandatory consumer refunds, and brand damage. | Align corporate marketing definitions strictly with ANSI/RIC001.1-2016 and implement third-party process audits. |
| 11 | **Reverse BOM Version Drift** | Harvested components cannot be assembled into current-generation core housings. | Forward engineering issued an Engineering Change Order (ECO) modifying mounting bolt patterns across model years. | High scrap rates on harvested components; assembly technicians forced to stop lines to resolve fitment errors. | Tag cores with manufacturing serial numbers and maintain historical rBOM backward-compatibility matrices in PLM. |
| 12 | **Core Deposit Abandonment** | Core return rates remain persistently low despite mandatory customer deposits. | The core deposit surcharge was set lower than the hassle/shipping cost or third-party scrap value. | Remanufacturing plant is starved of feedstock; capital-intensive machine tools sit idle. | Recalibrate core deposit value to exceed the market scrap price plus shipping friction by at least 25%. |

---

## 11. Assumptions and limits

| # | Assumption | Why Needed | Boundary Condition / Failure State | Diagnostic Test / Empirical Validation |
|---|---|---|---|---|
| 1 | **Core Condition Observable at Gate** | Enables multi-tier acquisition pricing $P(q)$ without incurring adverse selection. | Broken if core internal wear is latent and invisible until full chemical teardown and NDE. | Measure pre-disassembly diagnostic error matrix against post-teardown dimensional yield. |
| 2 | **Reversible Disassembly Graph** | Allows modeling teardown using directed hypergraphs without irreversible structural destruction. | Fails when components are permanently welded, riveted, or potted in epoxy (anti-tamper design). | Audit product design against ISO 8887-1 Design for Disassembly (DfD) standards. |
| 3 | **Uniform Customer WTP Distribution** | Enables analytical closed-form derivation of dual-market equilibrium prices $(p_n^*, p_r^*)$. | Real consumer willingness-to-pay is often bimodal (ultra-premium brand loyalists vs extreme discount hunters). | Fit empirical price-elasticity survey data using discrete choice logit or mixed-logit models. |
| 4 | **Linear Reservation Prices** | Provides a tractable linear core supply function $S(P) = N \cdot \frac{P - a}{b - a}$. | Fails if core holders exhibit extreme loss aversion or threshold behavior (e.g., minimum payout heuristics). | Run empirical econometric regressions on historical core buyback promotions. |
| 5 | **Stationary Recovery Yields** | Allows modeling component availability using static Beta parameters $(\alpha_k, \beta_k)$. | Fails when changes in fleet operating duty cycles or regional weather cause sudden yield shifts. | Track monthly control charts (CUSUM / EWMA) on component scrap and recovery rates. |
| 6 | **Independent Station Cycle Times** | Required to apply Kingman's heavy-traffic $G/G/m$ queueing approximation. | Fails when rework loops create autocorrelated feedback cycles between test stands and rework benches. | Compute lag-1 autocorrelation coefficients $\rho_k$ on consecutive workstation service times. |
| 7 | **Reman-Virgin Component Interchangeability** | Assumes remanufactured and virgin components are functionally identical on the re-assembly line. | Fails if reman parts have undersized dimensions (e.g., ground crankshaft journals) requiring selective mating. | Review engineering tolerances; check if selective-fit bearing shells are required. |
| 8 | **Zero Secondary Warranty Arbitrage** | Assumes consumers do not purchase remanufactured units and fraudulently return them as new for full refund. | Fails if outer packaging and physical cosmetic appearance are indistinguishable. | Inspect tamper-evident security labels, unique serial numbers, and laser-etched "REMAN" chassis marks. |
| 9 | **Unconstrained Virgin Supply** | Assumes virgin top-up components can be procured with deterministic lead time to balance shortages. | Fails during global supply chain crises when virgin suppliers prioritize new vehicle assembly lines. | Monitor supplier On-Time In-Full (OTIF) and dual-source critical bottleneck components. |
| 10 | **Static Regulatory Status of Cores** | Assumes environmental and customs classifications of cores remain constant across borders. | Fails when national governments update hazardous waste lists under the Basel Convention. | Conduct quarterly trade compliance audits on cross-border Harmonized System (HS) tariff codes. |

---

## 12. Exercises

### Exercise 1: Distinguishing Circular Transformation Tiers (Recall)
- **Deliverable:** A structured comparative matrix contrasting Remanufacturing, Refurbishment, Repair, and Recycling across four technical dimensions: (a) Depth of disassembly, (b) Warranty equivalence to new product, (c) Dimensional tolerance reset, and (d) Embedded energy preservation.
- **Acceptance Criteria:** Definitions must strictly conform to ANSI/RIC001.1-2016 and ISO 8887-1.
- **Estimated Time:** 20 minutes.

### Exercise 2: Optimal Core Buyback Tariff Formulation (Application)
- **Deliverable:** Calculate the profit-maximizing buyback price $P_k^*$ for three condition tiers of commercial vehicle brake calipers:
  - Tier 1 (Light wear): Expected downstream recovery value $R_1 = \$160.00$, diagnostic/handling cost $c_1 = \$20.00$, reservation prices $r_1 \sim U[30, 90]$.
  - Tier 2 (Medium wear): $R_2 = \$110.00$, $c_2 = \$25.00$, $r_2 \sim U[20, 70]$.
  - Tier 3 (Severe rust): $R_3 = \$55.00$, $c_3 = \$30.00$, $r_3 \sim U[15, 50]$.
  - State whether any tier should be rejected at the gate ($P^* = 0$).
- **Acceptance Criteria:** Full arithmetic derivation showing first-order condition solutions and verification of boundary constraints.
- **Estimated Time:** 30 minutes.

### Exercise 3: Reverse BOM Disassembly Stopping Boundary (Application)
- **Deliverable:** Given a subassembly with root housing containing three harvested components:
  - Component 1: Value $V_1 = \$80.00$, removal operation $O_1$ requires labor cost $c_1 = \$25.00$, recovery yield $Y_1 = 0.90$.
  - Component 2: Accessible only after $O_1$. Value $V_2 = \$45.00$, removal operation $O_2$ requires labor cost $c_2 = \$30.00$, recovery yield $Y_2 = 0.70$.
  - Component 3: Accessible only after $O_2$. Value $V_3 = \$20.00$, removal operation $O_3$ requires labor cost $c_3 = \$12.00$, recovery yield $Y_3 = 0.50$, but carries an 8% risk of cracking the root housing (housing salvage value $V_{\text{house}} = \$60.00$).
  - Calculate the net marginal value of each operation and state the exact optimal selective disassembly stopping point.
- **Acceptance Criteria:** Correctly evaluate marginal Bellman equations $\Delta V_o$ and identify which operations must be executed.
- **Estimated Time:** 35 minutes.

### Exercise 4: Stochastic Yield and the Unbalanced Bundle (Application)
- **Deliverable:** A remanufacturer processes 500 hydraulic motor cores. The motor BOM requires 1 Main Shaft, 1 Cylinder Block, and 7 Piston Slippers.
  - Recovery yields are independent random variables: $Y_{\text{shaft}} \sim \text{Beta}(8, 2)$ ($E[Y] = 0.80$), $Y_{\text{block}} \sim \text{Beta}(6, 4)$ ($E[Y] = 0.60$), and individual piston slippers have an independent survival rate of $p = 0.70$.
  - Calculate: (a) The expected maximum remanufactured motor volume achievable purely from harvested parts without virgin procurement, (b) The expected surplus inventory of Main Shafts, and (c) The number of virgin piston slippers required to achieve a production run of 300 complete motors.
- **Acceptance Criteria:** Correct application of binomial expectations and minimum-yield bottleneck logic.
- **Estimated Time:** 30 minutes.

### Exercise 5: Remanufacturing Queue Congestion and Buffer Sizing (Application)
- **Deliverable:** A CNC cylinder head skimming station receives 4.0 cylinder heads per hour ($\lambda = 4.0$). The station has $m = 2$ identical CNC machines. Each machine has a mean processing time $\tau = 24.0$ minutes ($\mu = 2.5$ units/hour/machine). Arrival coefficient of variation is $c_a = 0.90$.
  - Calculate the expected queue waiting time $W_q$ and buffer queue length $L_q$ under two conditions:
    - Case A: Forward manufacturing line where skimming cycle time is tightly controlled ($c_s = 0.15$).
    - Case B: Remanufacturing line where deep warpage requires variable multi-pass skimming ($c_s = 1.60$).
  - Calculate the ratio of remanufacturing WIP to forward WIP.
- **Acceptance Criteria:** Correct execution of Kingman's heavy-traffic $G/G/m$ queueing approximation.
- **Estimated Time:** 35 minutes.

### Exercise 6: Dual-Market Pricing under Core Constraints (Application)
- **Deliverable:** An OEM sells a new commercial drone and a remanufactured version. Market size $M = 5,000$ units. $V_{max} = \$2,000.00$. New drone manufacturing cost $c_n = \$800.00$. Remanufacturing cost $c_r = \$360.00$. Willingness-to-pay discount $\delta = 0.70$.
  - Calculate: (a) Unconstrained optimal prices $(p_n^*, p_r^*)$ and demands $(D_n^*, D_r^*)$, (b) Marginal cannibalization rate $\gamma$, and (c) Constrained optimal prices $(p_n^{**}, p_r^{**})$ and demands $(D_n^{**}, D_r^{**})$ when available core supply is capped at $Q_{core}^{cap} = 800$ units.
- **Acceptance Criteria:** Complete mathematical derivation with normalized variables, KKT conditions, and financial totals.
- **Estimated Time:** 40 minutes.

### Exercise 7: Critique of a Failed Hybrid Line (Critique)
- **Deliverable:** Review a real-world scenario where a heavy industrial pump manufacturer merged its new pump assembly and remanufacturing lines into a single hybrid flexible manufacturing cell to "maximize capital asset utilization." Within four months, on-time delivery of new pumps dropped from 96% to 68%, WIP inventory quadrupled, and customer warranty claims for particulate contamination doubled.
  - Identify the three fundamental structural root causes of this operational failure using queueing theory, cycle time variance ($c_s$), and contamination mechanics.
  - Prescribe the operational restructuring plan required to restore stability.
- **Acceptance Criteria:** Critique must address Kingman variance multipliers, contamination quarantines, and line architecture trade-offs.
- **Estimated Time:** 45 minutes.

### Exercise 8: Enterprise Closed-Loop Architecture Design (Design)
- **Deliverable:** Design a complete closed-loop remanufacturing program for an electric commercial van manufacturer recovering 100 kWh battery packs. Your proposal must specify:
  1. Core acquisition incentive tariff (deposit surcharge vs buyback) based on battery State of Health (SOH).
  2. Selective disassembly decision tree (module harvesting vs direct stationary second-life vs raw hydrometallurgical recycling).
  3. Digital Product Passport (DPP) data schema compliant with EU ESPR (2024/1781).
  4. Dual-market warranty and recommerce branding policy to prevent primary vehicle sales cannibalization.
- **Acceptance Criteria:** Concrete, operational proposal covering financial mechanisms, technical metrics (SOH thresholds), and regulatory compliance.
- **Estimated Time:** 60 minutes.

---

## 13. Self-check answers

### Answer 1: Distinguishing Circular Transformation Tiers
A complete response must provide the following structured comparison:
- **Remanufacturing:** Full teardown to raw parts; 100% inspection against OEM blueprints; dimensional re-machining to restore original tolerances; equivalent-to-new warranty (ANSI/RIC001.1-2016); preserves 70–85% of embedded energy and material.
- **Refurbishment:** Partial teardown; cosmetic reconditioning and functional testing to secondary working standards; shorter warranty (e.g., 90 days); does not re-machine castings; preserves 50–70% of embedded energy.
- **Repair:** Localized replacement or fixing of a single failed component without full overhaul; no dimensional reset; limited warranty covering only repaired part; preserves 80–90% of current asset state but does not reset product wear clock.
- **Recycling:** Destructive shredding, smelting, or chemical dissolution to recover base raw material melt; destroys 100% of geometric and machining value; preserves only 15–30% of original embedded energy.

### Answer 2: Optimal Core Buyback Tariff Formulation
Using the optimal unconstrained pricing formula $P_k^* = \frac{R_k - c_k^{proc} + a_k}{2}$:
- **Tier 1:** Net margin before buyback: $R_1 - c_1 = 160.00 - 20.00 = \$140.00$.
  Reservation range $[a_1, b_1] = [30, 90]$.
  $$P_1^* = \frac{140.00 + 30.00}{2} = \frac{170.00}{2} = \$85.00$$
  Check bounds: $30 \le 85.00 \le 90$ (Valid).
- **Tier 2:** Net margin before buyback: $R_2 - c_2 = 110.00 - 25.00 = \$85.00$.
  Reservation range $[a_2, b_2] = [20, 70]$.
  $$P_2^* = \frac{85.00 + 20.00}{2} = \frac{105.00}{2} = \$52.50$$
  Check bounds: $20 \le 52.50 \le 70$ (Valid).
- **Tier 3:** Net margin before buyback: $R_3 - c_3 = 55.00 - 30.00 = \$25.00$.
  Reservation range $[a_3, b_3] = [15, 50]$.
  $$P_3^* = \frac{25.00 + 15.00}{2} = \frac{40.00}{2} = \$20.00$$
  Check bounds: $15 \le 20.00 \le 50$ (Valid).
  Economic check: $R_3 - c_3 = 25.00 > a_3 = 15.00$. Tier 3 produces a positive net margin of $\$5.00$ per unit acquired ($25.00 - 20.00$). Therefore, **no tier is rejected at the gate**.

### Answer 3: Reverse BOM Disassembly Stopping Boundary
Evaluate the net marginal value $\Delta V_o$ for each operation:
- **Operation 1 (Harvests Component 1):**
  $$\Delta V_1 = (V_1 \times Y_1) - c_1 = (80.00 \times 0.90) - 25.00 = 72.00 - 25.00 = +\$47.00$$
  Since $\Delta V_1 > 0$, execute Operation 1.
- **Operation 2 (Harvests Component 2):**
  $$\Delta V_2 = (V_2 \times Y_2) - c_2 = (45.00 \times 0.70) - 30.00 = 31.50 - 30.00 = +\$1.50$$
  Since $\Delta V_2 > 0$, execute Operation 2.
- **Operation 3 (Harvests Component 3):**
  $$\Delta V_3 = (V_3 \times Y_3) - c_3 - (\pi_3 \times V_{\text{house}}) = (20.00 \times 0.50) - 12.00 - (0.08 \times 60.00)$$
  $$\Delta V_3 = 10.00 - 12.00 - 4.80 = -\$6.80$$
  Since $\Delta V_3 < 0$, Operation 3 destroys value!
- **Stopping Boundary:** Halt disassembly after Operation 2. Do NOT perform Operation 3. Preserve Component 3 inside the root housing and route the remaining subassembly to scrap or modular resale.

### Answer 4: Stochastic Yield and the Unbalanced Bundle
- **Expected Yields from 500 Cores:**
  - Expected Main Shafts: $500 \times 0.80 = 400$ shafts.
  - Expected Cylinder Blocks: $500 \times 0.60 = 300$ blocks.
  - Expected Piston Slippers: $500 \times (7 \text{ per core}) \times 0.70 = 3,500 \times 0.70 = 2,450$ slippers.
- **Part (a): Pure Harvest Production Volume:**
  Finished motors require 1 Shaft, 1 Block, and 7 Slippers:
  $$\text{Max from Shafts} = 400 / 1 = 400$$
  $$\text{Max from Blocks} = 300 / 1 = 300$$
  $$\text{Max from Slippers} = 2,450 / 7 = 350$$
  Bottleneck component is the Cylinder Block ($300$ units). Pure remanufactured volume achievable is **300 complete motors**.
- **Part (b): Surplus Main Shaft Inventory:**
  $$\text{Surplus Shafts} = 400 - 300 = 100 \text{ shafts}$$
- **Part (c): Virgin Piston Slippers for 300 Motors:**
  To produce 300 motors, required slippers $= 300 \times 7 = 2,100$ slippers.
  Harvested slippers available $= 2,450$.
  Since $2,450 \ge 2,100$, **zero virgin piston slippers are required**; in fact, there is a surplus of 350 harvested slippers.

### Answer 5: Remanufacturing Queue Congestion and Buffer Sizing
Given: $\lambda = 4.0$ units/hour, $m = 2$ machines, $\tau = 24.0$ min ($0.40$ hr), $\mu = 2.5$ units/hr/machine.
Utilization: $\rho = \frac{\lambda}{m \mu} = \frac{4.0}{2 \times 2.5} = \frac{4.0}{5.0} = 0.80$.
Kingman utilization factor for $m = 2$:
$$\text{Factor}_{util} = \frac{\rho^{\sqrt{2(2+1)} - 1}}{2(1 - \rho)} = \frac{0.80^{\sqrt{6} - 1}}{2(1 - 0.80)} = \frac{0.80^{2.4495 - 1}}{2(0.20)} = \frac{0.80^{1.4495}}{0.40} = \frac{0.7235}{0.40} = 1.80875$$
- **Case A (Forward Line, $c_a = 0.90, c_s = 0.15$):**
  $$V_{mult} = \frac{0.90^2 + 0.15^2}{2} = \frac{0.81 + 0.0225}{2} = \frac{0.8325}{2} = 0.41625$$
  $$W_q^{fwd} \approx 0.41625 \times 1.80875 \times 24.0 = 18.07 \text{ minutes (0.301 hours)}$$
  $$L_q^{fwd} = \lambda \times W_q = 4.0 \times 0.301 = 1.20 \text{ units in buffer}$$
- **Case B (Remanufacturing Line, $c_a = 0.90, c_s = 1.60$):**
  $$V_{mult} = \frac{0.90^2 + 1.60^2}{2} = \frac{0.81 + 2.56}{2} = \frac{3.37}{2} = 1.685$$
  $$W_q^{reman} \approx 1.685 \times 1.80875 \times 24.0 = 73.15 \text{ minutes (1.219 hours)}$$
  $$L_q^{reman} = \lambda \times W_q = 4.0 \times 1.219 = 4.88 \text{ units in buffer}$$
- **WIP Ratio:** $\frac{L_q^{reman}}{L_q^{fwd}} = \frac{4.88}{1.20} = \mathbf{4.07\times}$. Remanufacturing requires more than quadruple the buffer staging space.

### Answer 6: Dual-Market Pricing under Core Constraints
Normalized parameters: $c_n = 800 / 2,000 = 0.40$, $c_r = 360 / 2,000 = 0.18$, $\delta = 0.70$.
- **Part (a): Unconstrained Solution:**
  $$p_n^* = \frac{1 + 0.40}{2} \times 2,000 = 0.70 \times 2,000 = \$1,400.00$$
  $$p_r^* = \frac{0.70 + 0.18}{2} \times 2,000 = \frac{0.88}{2} \times 2,000 = 0.44 \times 2,000 = \$880.00$$
  $$\theta_{nr} = \frac{1,400 - 880}{(1 - 0.70) \times 2,000} = \frac{520}{600} = 0.8667$$
  $$\theta_{r0} = \frac{880}{0.70 \times 2,000} = \frac{880}{1,400} = 0.6286$$
  $$D_n^* = 5,000 \times (1 - 0.8667) = 666.7 \text{ units}$$
  $$D_r^* = 5,000 \times (0.8667 - 0.6286) = 1,190.5 \text{ units}$$
- **Part (b): Marginal Cannibalization Rate:**
  $$\gamma = \delta = 0.70 \quad \text{(70% of reman sales substitute for new sales)}$$
- **Part (c): Core-Constrained Solution ($Q_{core}^{cap} = 800$, normalized $q = 800 / 5,000 = 0.16$):**
  $$p_n^{**} = \$1,400.00 \quad \text{(Unchanged)}$$
  $$x^* = 0.70 - (1 - 0.70) \times 0.16 = 0.70 - 0.048 = 0.652$$
  $$p_r^{**} = (0.70 \times 0.652) \times 2,000 = 0.4564 \times 2,000 = \$912.80$$
  $$\theta_{nr}^{**} = \frac{1,400 - 912.80}{600} = \frac{487.20}{600} = 0.8120$$
  $$\theta_{r0}^{**} = \frac{912.80}{1,400} = 0.6520$$
  $$D_n^{**} = 5,000 \times (1 - 0.8120) = 940.0 \text{ units}$$
  $$D_r^{**} = 5,000 \times (0.8120 - 0.6520) = 800.0 \text{ units} \quad \text{(Matches cap exactly)}$$
  New unit sales expand from $666.7$ to $940.0$ units (+273.3 units).

### Rubric for Exercise 7: Critique of a Failed Hybrid Line
- **Root Cause 1 (Queueing Variance Contagion):** Reman processing variance ($c_s > 1.2$) injected into a shared line causes WIP explosions under Kingman's law, starving downstream stations and destroying forward line takt time.
- **Root Cause 2 (Physical Particulate Contamination):** Dirty cores introduced into cleanroom assembly areas release particulate grit and carbon, contaminating pristine hydraulic seals and causing the doubling of 90-day field warranty failures.
- **Root Cause 3 (Workforce Cognitive Friction):** Assembly technicians accustomed to standardized assembly struggle with adaptive diagnostic re-machining, causing human errors and routing bottlenecks.
- **Remediation Plan:** Physically decouple core teardown and cleaning into a sealed preprocessing cell; establish dedicated decoupling CONWIP buffers; operate reman at $\le 75\%$ utilization.

### Rubric for Exercise 8: Enterprise Closed-Loop Architecture Design
- **Core Incentive Tariff (25%):** SOH-tiered deposit surcharge refunded via battery telemetry data ($>80\%$ SOH receives 100% deposit; 60–80% receives 70%; $<60\%$ receives scrap value).
- **Disassembly Tree & Stopping Rule (25%):** High-level battery pack opening; individual module SOH grading; modules $>70\%$ SOH routed to stationary second-life storage; failed modules routed to hydrometallurgical black mass recycling.
- **Digital Product Passport Schema (25%):** JSON-LD schema capturing chemistry, carbon footprint, cycle history, and certified repair logs compliant with EU ESPR (2024/1781).
- **Dual-Market Recommerce Policy (25%):** Second-life storage sold exclusively in commercial B2B energy markets, shielding primary passenger van sales from cannibalization.

---

## 14. Interview questions

### Question 1: How do you mathematically establish whether an OEM should enter the remanufacturing business if it risks cannibalizing its primary product sales?
- **What a strong answer contains:** A strong answer outlines the consumer utility framework ($U_n = \theta - p_n$, $U_r = \delta \theta - p_r$), proves that the marginal cannibalization rate equals $\delta$, and demonstrates that remanufacturing expands total market reach by serving low-valuation customers $[	heta_{r0}, \theta_{nr}]$. It highlights the cost spread condition: entry is strictly profitable whenever the cost savings $(c_n - c_r)$ exceed the revenue erosion caused by cannibalization, and notes that if the OEM abstains, third-party remanufacturers will enter and capture secondary margins without internalizing OEM brand protection.
- **What a weak answer reveals:** A weak answer treats cannibalization purely as a marketing issue, asserts without proof that "remanufacturing always damages high-end luxury brands," or fails to recognize market size expansion.

### Question 2: Why can you not balance a remanufacturing line using the same line-balancing algorithms (e.g., Ranked Positional Weight) used in automotive forward assembly?
- **What a strong answer contains:** Cites Kingman's heavy-traffic queueing formula, identifying that remanufacturing cycle times exhibit extreme right-skewed variance ($c_s \in [1.0, 2.5]$) due to variable rust, coking, and diagnostic rework loops. Explains that deterministic takt times do not hold, causing dynamic bottleneck migration where the bottleneck station shifts based on the condition mix of incoming core batches. Prescribes dynamic CONWIP buffers, de-rating utilization to $\le 75\%$, and cross-trained floating labor.
- **What a weak answer reveals:** Suggests simply "timing the workers with a stopwatch and taking the average," failing to grasp how variance multiplies queue wait times.

### Question 3: What is the "unbalanced bundle" problem in reverse logistics, and what operational policies resolve it?
- **What a strong answer contains:** Explains that disassembling a fixed core releases components in fixed stoichiometric ratios, but stochastic wear produces unequal recovery yields across parts ($Y_{\text{block}} \gg Y_{\text{crank}}$). Discusses the trade-off between Push Disassembly (which creates surplus part inventory holding costs) and Pull Disassembly (which protects holding costs but risks stockouts). Outlines the role of dual-source linear programming to dynamically procure virgin parts for bottleneck components.
- **What a weak answer reveals:** Confuses the unbalanced bundle with standard supplier delivery delays or assumes all parts inside a core wear out at the exact same rate.

### Question 4: Under what conditions does an OEM's core supply constraint actually protect its primary product profit?
- **What a strong answer contains:** Formulates the Karush-Kuhn-Tucker (KKT) constrained dual-market pricing model. Explains that when core supply $Q_{core}^{cap}$ is less than unconstrained reman demand $D_r^*$, the OEM must raise the remanufactured price $p_r^{**}$ to clear the market. This price hike extracts scarcity rents from secondary buyers while simultaneously pushing marginal buyers back into purchasing higher-margin new products ($D_n^{**} > D_n^*$).
- **What a weak answer reveals:** Assumes core shortages are always purely negative for corporate profits, missing the price-discrimination mechanism.

### Question 5: How does Akerlof's "Lemons Problem" manifest in core acquisition, and how can an analytics team design a buyback mechanism to defeat it?
- **What a strong answer contains:** Shows that offering a uniform buyback price $\bar{P}$ creates adverse selection: core holders compare $\bar{P}$ to their condition-dependent reservation price $r(q)$, causing high-quality cores to be withheld while degraded cores flood the depot ($E[q \mid \text{Return}] < \bar{q}$). Prescribes multi-tier inspection pricing $P_k^*$, digital pre-grading apps using computer vision, and core deposit surcharges tied to verified physical condition.
- **What a weak answer reveals:** Blames return quality on "unethical customers" rather than misaligned economic contract design.

### Question 6: What is the difference between an AND/OR disassembly graph and a standard assembly tree, and how is the optimal selective disassembly stopping point determined?
- **What a strong answer contains:** Explains that reverse BOMs are directed hypergraphs where AND nodes represent joint component releases and OR nodes represent alternative removal methods (e.g., non-destructive unbolting vs destructive cutting). States the marginal Bellman stopping equation: halt teardown when the expected marginal value of liberated parts minus operation cost and core damage risk drops below zero.
- **What a weak answer reveals:** Believes disassembly is simply running the forward assembly CAD tree in reverse order or that products must always be 100% disassembled.

### Question 7: How do the EU Ecodesign for Sustainable Products Regulation (ESPR) and Digital Product Passports (DPP) impact remanufacturing supply chains?
- **What a strong answer contains:** Details ESPR (Regulation EU 2024/1781) requirements: mandatory machine-readable data carriers linking to decentralized DPP records containing material composition, disassembly sequence documentation, and repair histories. Mentions the ban on software anti-repair locks (parts pairing) and explains how DPP reduces inspection uncertainty at core gate arrival, directly mitigating adverse selection.
- **What a weak answer reveals:** Vaguely mentions "sustainability laws in Europe" without naming ESPR, DPP, or the technical data requirements.

### Question 8: How would you decide whether to configure a facility as a dedicated remanufacturing plant versus a hybrid manufacturing line?
- **What a strong answer contains:** Evaluates the trade-off across processing variance ($c_s$), contamination risks, asset utilization, and capital expenditure. Recommends dedicated lines when $c_s > 0.8$ or when cores introduce severe particulate contamination (heavy engines, hydraulic equipment). Recommends hybrid lines when capital equipment is extremely expensive (e.g., specialized surface-mount electronic pick-and-place), provided teardown and chemical cleaning are quarantined.
- **What a weak answer reveals:** Recommends hybrid lines purely to "save money on buildings" without evaluating queue variance or contamination risks.

---
### Question 9: How does an unrefunded core deposit affect an enterprise's balance sheet and working capital?
- **What a strong answer contains:** A strong answer explains that unrefunded core deposits are recorded as a current liability ("Accrued Core Deposit Liability" or "Unearned Customer Refund Liability") rather than immediate revenue. It explains that cash flow is initially positive, but working capital is restricted. Once the contractual return window ($T_{horizon}$) passes, the unreturned core is reclassified as breakage revenue, but the asset itself is lost, requiring a write-off against future core inventory pipelines.
- **What a weak answer reveals:** A weak answer believes core deposits are recorded directly as retained earnings or sales margin on day one.

### Question 10: How do you design an automated inspection and grading station to minimize total error costs under asymmetric under-grading and over-grading penalties?
- **What a strong answer contains:** Formulates the inspection decision problem under an asymmetric cost loss function: $C(\text{decision}, \text{truth})$. Recognizes that over-grading ($C_{over}$) leads to field warranty returns, customer churn, and double shipping, while under-grading ($C_{under}$) causes salvage margin loss. Shows how the optimal classification probability threshold $\tau^*$ shifts away from 0.50 to $\tau^* = \frac{C_{under}}{C_{under} + C_{over}}$. If $C_{over} \gg C_{under}$, the classification threshold is set high (conservative grading).
- **What a weak answer reveals:** Focuses entirely on classification accuracy or ROC-AUC without connecting model thresholds to dollar-denominated operational error costs.

---

## 15. Key terms

- **Core:** A worn, retired, or failed asset retained for high-value remanufacturing.
- **Remanufacturing:** Industrial restoration of a core to original OEM specifications with an equivalent warranty.
- **Refurbishment:** Cosmetic and functional reconditioning to acceptable secondary standards.
- **Reverse Bill of Materials (rBOM):** An AND/OR directed hypergraph modeling disassembly paths.
- **Selective Disassembly:** Halting product teardown at an intermediate subassembly depth to maximize net margin.
- **Destructive Disassembly:** Rapid teardown intentionally shearing low-value fasteners to liberate high-value cores.
- **Core Acquisition Price $P(q)$:** The quality-tiered financial incentive offered to core holders.
- **Core Reservation Price $r_i$:** The minimum threshold payment required for a core holder to return an asset.
- **Adverse Selection (Lemons Effect):** A uniform buyback tariff attracting disproportionately degraded cores.
- **Stochastic Recovery Yield $Y_k$:** The random variable governing usable components recovered per core.
- **Push Disassembly:** An operational policy stripping all incoming cores immediately into part inventory.
- **Pull Disassembly:** An operational policy retaining cores intact until downstream assembly demand triggers teardown.
- **Unbalanced Bundle:** The simultaneous surplus of high-yield parts and shortage of low-yield parts from teardown.
- **Cycle Time Coefficient of Variation ($c_s$):** The ratio $\sigma / \mu$ capturing remanufacturing task variability.
- **Kingman's Heavy-Traffic Approximation:** Formula quantifying queue waiting time explosions under high $c_s$.
- **Dynamic Bottleneck Migration:** The shifting of line constraints between cleaning, machining, and testing.
- **Willingness-to-Pay Discount ($\delta$):** The fractional consumer valuation discount for remanufactured goods.
- **Cannibalization Rate ($\gamma$):** The marginal substitution rate showing new product sales displaced by reman sales.
- **Indifference Valuation ($\theta_{nr}$):** The customer valuation threshold dividing new buyers from reman buyers.
- **Core Availability Shadow Price ($\lambda$):** The marginal profit value of relaxing the physical core supply constraint.
- **Digital Product Passport (DPP):** Standardized digital identity tracking product lifecycle and disassembly data.
- **Core Deposit / Surcharge:** An upfront fee refunded only upon returning an acceptable worn core.
- **ANSI/RIC001.1-2016:** The national standard defining formal remanufacturing process criteria.
- **ISO 8887-1:2017:** The technical product documentation standard for design for disassembly (DfD).
- **EU ESPR (2024/1781):** EU regulation mandating ecodesign requirements and digital product passports.
- **EU Right to Repair (2024/1799):** EU directive mandating 10-year parts access and banning parts-pairing locks.
- **FTC Green Guides (16 CFR § 260.11):** U.S. disclosure rules governing truth in "remanufactured" marketing claims.
- **Basel Convention:** International treaty restricting transboundary shipments of uncertified e-waste cores.
- **CONWIP (Constant Work in Process):** A pull queue control mechanism pacing line input to bottleneck capacity.
- **Absorbing Markov Chain:** Mathematical model evaluating multi-pass inspection, rework loops, and terminal scrap.

---

## 16. Sources

1. **Guide, V. D. R., & Van Wassenhove, L. N. (2009).** *The Evolution of Closed-Loop Supply Chain Research.* Operations Research, 57(1), 10–18. https://doi.org/10.1287/opre.1080.0628
2. **Ferguson, M. E., & Toktay, L. B. (2006).** *The Effect of Competition on Recovery Strategies.* Production and Operations Management, 15(3), 351–368. https://doi.org/10.1111/j.1937-5956.2006.tb00250.x
3. **Atasu, A., Sarvary, M., & Van Wassenhove, L. N. (2008).** *Remanufacturing as a Marketing Strategy.* Management Science, 54(10), 1731–1746. https://doi.org/10.1287/mnsc.1080.0893
4. **Savaskan, R. C., Bhattacharya, S., & Van Wassenhove, L. N. (2004).** *Closed-Loop Supply Chain Models with Product Remanufacturing.* Management Science, 50(2), 239–252. https://doi.org/10.1287/mnsc.1030.0186
5. **Debo, L. G., Savaskan, R. C., & Van Wassenhove, L. N. (2005).** *Market Segmentation and Product Technology Selection for Remanufacturable Products.* Management Science, 51(8), 1193–1205. https://doi.org/10.1287/mnsc.1050.0396
6. **Lambert, A. J. D. (2003).** *Disassembly Sequencing: A Survey.* International Journal of Production Research, 41(16), 3721–3759. https://doi.org/10.1080/0020754031000120078
7. **Gungor, A., & Gupta, S. M. (1999).** *Issues in Environmentally Conscious Manufacturing and Product Recovery: A Survey.* Computers & Industrial Engineering, 36(4), 811–853. https://doi.org/10.1016/S0360-8352(99)00167-9
8. **Guide, V. D. R., Jayaraman, V., & Srivastava, R. (1999).** *Production Planning and Control for Remanufacturing: A State-of-the-Art Survey.* Robotics and Computer-Integrated Manufacturing, 15(3), 221–230. https://doi.org/10.1016/S0736-5845(99)00020-4
9. **American National Standards Institute / Remanufacturing Industries Council. (2016).** *Standard for Information Required for Remanufacturing (ANSI/RIC001.1-2016).* ANSI/RIC.
10. **International Organization for Standardization. (2017).** *Technical product documentation — Design for manufacture, assembling, disassembling and end-of-life processing (ISO Standard No. 8887-1:2017).* ISO.
11. **European Parliament and Council of the European Union. (2024).** *Regulation (EU) 2024/1781 establishing a framework for setting ecodesign requirements for sustainable products (ESPR).* Official Journal of the European Union, L 1781.
12. **European Parliament and Council of the European Union. (2024).** *Directive (EU) 2024/1799 on common rules promoting the repair of goods.* Official Journal of the European Union, L 1799.
13. **United States Federal Trade Commission. (2012).** *Guides for the Use of Environmental Marketing Claims (16 CFR Part 260).* Federal Register, 77(197), 62122–62132.
14. **United Nations Environment Programme. (2019).** *Basel Convention on the Control of Transboundary Movements of Hazardous Wastes and Their Disposal (Adopted 1989, amended 2019).* UNEP.
15. **Caterpillar Inc. (2024).** *Caterpillar Sustainability Report 2023: Circular Economy and Cat Reman Performance.* Caterpillar Corporate Communications.

---

## 17. Links

- **Parent Module Index:** [`_index.md`](_index.md) — Circular Economy and Reverse Logistics Networks
- **Preceding Sibling Topic:** [`04_reverse-network-design.md`](04_reverse-network-design.md) — Reverse Network Design and Collection Consolidation
- **Foundational Grading Precursor:** [`03_grading-yield.md`](03_grading-yield.md) — Grading and Recovery Yield Optimization
- **Economics Precursor:** [`02_returns-unit-economics.md`](02_returns-unit-economics.md) — Return Unit Economics and Net Recovery
- **Disposition Hierarchy Precursor:** [`01_reverse-flows-disposition.md`](01_reverse-flows-disposition.md) — Reverse Flows and the Disposition Hierarchy
- **Downstream Sibling Topic:** [`06_returns-fraud.md`](06_returns-fraud.md) — Returns Fraud and Abuse Mitigation
- **Second-Life Battery Topic:** [`08_batteries-ewaste-second-life.md`](08_batteries-ewaste-second-life.md) — Batteries, E-Waste, and Second-Life Systems
- **Recommerce Commercialization Topic:** [`10_recommerce-product-as-service.md`](10_recommerce-product-as-service.md) — Recommerce and Product-as-a-Service Business Models
- **Course Root Survey File:** [`../../../course/37_circular_economy_reverse_networks.md`](../../../course/37_circular_economy_reverse_networks.md)
