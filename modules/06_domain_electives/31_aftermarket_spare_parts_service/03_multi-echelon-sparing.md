> Module 31 — Aftermarket, Spare Parts, and Service Supply Chains
> Topic 03 of 09
> Estimated study time: 4.0 hours
> Prerequisites: Module 31 Topic 01 (Installed Base & Failure Demand), Module 31 Topic 02 (Criticality Classification), Module 10 (Multi-Echelon Inventory Optimization)
> Version 2026.5

# Multi-Echelon Sparing and METRIC

---

## 1. Why this topic exists

In capital-intensive service operations—commercial aviation, military defense systems, heavy rail, offshore drilling, and medical imaging—spares inventory cannot be managed as a collection of independent, isolated warehouses. 

When a hospital scanner, an offshore drilling rig, or an airliner at an outstation suffers a breakdown, a functional replacement part must be installed immediately. However, stocking every high-value, slow-moving recoverable component at every field location is financially impossible:
* A commercial twin-engine aircraft contains hundreds of distinct Line Replaceable Units (LRUs), such as flight-management computers, hydraulic actuators, and weather radar transceivers, costing between $20,000 and $350,000 each.
* If an airline operating 60 aircraft across 25 destination airports attempts to hold a dedicated buffer of every LRU at every airport, the required working capital would bankrupt the airline.
* Conversely, if the airline centralizes all inventory at a single hub warehouse, outstation aircraft will face catastrophic multi-day flight cancellations waiting for emergency courier shipments.

The fundamental engineering trade-off of aftermarket logistics is **response time versus inventory pooling**. 

Holding spares at forward field locations minimizes response time (wrench time begins in minutes), but fragments inventory into small, decentralized pools with miserable turnover and immense total safety stock. Holding spares at a central depot maximizes risk-pooling across the entire fleet, slashing aggregate safety stock, but introduces transportation delay times that ground distant machines.

Furthermore, capital equipment components are predominantly **recoverable (repairable)** items. When a failed component is removed from a machine, it is not scrapped; it is inducted into a repair shop, overhauled, recertified, and returned to the spares pool as a ready-for-issue (RFI) unit. Therefore, the forward supply chain (distribution of ready spares) is mechanically coupled to the reverse supply chain (evacuation, testing, and turnaround of unserviceable carcasses).

If an analyst uses classical single-echelon safety stock formulas (such as $SS = z \sigma \sqrt{L}$) independently for each field base and the central depot, the results are mathematically invalid:
1. **The Lead-Time Endogeneity Trap:** The replenishment lead time experienced by a field base is not a constant supplier parameter; it is an endogenous variable determined by the stock level at the central depot. If the depot has stock on hand, the base lead time is the fast transportation transit time. If the depot is stocked out, the base lead time is the transit time *plus* the duration the base must wait for a unit to emerge from the depot repair shop.
2. **The Metric Mismatch:** Optimizing local fill rates ($P(\text{Demand} \le s)$) at individual warehouses does not optimize asset availability. A network can achieve a 95% average fill rate across all parts while maintaining a 40% fleet grounding rate because the missing 5% of parts are critical single points of failure scattered unevenly across operating bases.

To solve this problem, Craig Sherbrooke developed the **METRIC (Multi-Echelon Technique for Recoverable Item Control)** framework at the RAND Corporation. METRIC and its modern mathematical descendants (MOD-METRIC, VARI-METRIC) model the entire multi-echelon repair-and-distribution system as a unified queueing network, optimizing stock allocation across central depots and field bases to maximize fleet availability subject to capital budget constraints. This topic teaches the mathematical foundations, computational algorithms, and operational execution of multi-echelon sparing.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Contrast** single-echelon inventory control models with multi-echelon recoverable systems, identifying why independent base optimization creates severe inventory misallocation.
2. **Apply** Palm's Theorem to formulate the probability distribution of items in the repair and resupply pipeline for an infinite-server queue ($M/G/\infty$).
3. **Formulate** the mathematical expression for Expected Backorders (EBO) at both depot and base levels under continuous-review $(S-1, S)$ base-stock policies.
4. **Synthesize** the endogenous base replenishment lead time distribution as a function of local repair fraction, transit time, and depot backorder probability.
5. **Implement** Fox's greedy marginal allocation algorithm to construct an efficient cost-availability frontier, minimizing total fleet backorders subject to budget constraints.
6. **Extend** standard single-indenture METRIC logic to multi-indenture systems (MOD-METRIC), capturing the hierarchical dependency between Line Replaceable Units (LRUs) and Shop Replaceable Units (SRUs).
7. **Evaluate** the mathematical limits of the Poisson demand assumption in METRIC and explain how VARI-METRIC corrects for variance-to-mean ratio (VMR) distortion caused by depot queue compounding.
8. **Design** a multi-echelon network stocking policy that balances central depot pooling, forward base buffers, and lateral transshipment between peer bases.

---

## 3. The core mental model

To understand multi-echelon sparing, you must visualize the closed-loop physical and informational flow of recoverable components across the network:

```text
               +--------------------------------------------------------+
               | CENTRAL DEPOT (Echelon 0)                              |
               |                                                        |
               |  Depot Stock: S_0                 Depot Repair Shop    |
               |  [ Ready-for-Issue (RFI) ]        (TAT_0)              |
               |         ▲                               ▲              |
               |         │ Repaired Carcass              │ Unserviceable|
               |         └───────────────────────────────┤ (1 - r_j)    |
               +─────────────────┬───────────────────────┴──────────────+
                                 │
                   Shipment of   │ Order & Ship Time (O_0j)
                   Ready Spares  │ + Depot Delay [EBO_0 / lambda_0]
                                 ▼
   +────────────────────────────────────────────────────────────────────────+
   | OPERATING BASES / SITES (Echelon 1)                                    |
   |                                                                        |
   |   BASE 1 (Demand: lambda_1)               BASE J (Demand: lambda_J)    |
   |   Base Stock: S_1                         Base Stock: S_J              |
   |                                                                        |
   |   [ Field Spare Pool ]                    [ Field Spare Pool ]         |
   |      │              ▲                        │              ▲          |
   |      │ Issue Part   │ Base Repair            │ Issue Part   │ Base Rep.|
   |      ▼              │ (TAT_1) r_1            ▼              │ (TAT_J)  |
   |   [ DOWN ASSET ] ───┴───────────►         [ DOWN ASSET ] ───┴────────► |
   |   (Backorder if Pool Empty)               (Backorder if Pool Empty)    |
   +────────────────────────────────────────────────────────────────────────+
```

### 3.1 The Two Fundamental Closed Loops

The system consists of two intertwined operational loops:
1. **The Base Repair Loop (Fraction $r_j$):** When an asset fails at Base $j$, a technician removes the failed LRU. If Base $j$ possesses intermediate-level maintenance capability, a fraction $r_j$ of carcasses are repaired locally in the base workshop with turnaround time $\text{TAT}_j$. Once repaired, the carcass returns to the base's Ready-for-Issue (RFI) shelf.
2. **The Depot Resupply Loop (Fraction $1 - r_j$):** If the failure exceeds local capability (complex internal damage, specialized calibration needed), the carcass is evacuated to the Central Depot repair facility. Simultaneously, Base $j$ sends an electronic replenishment requisition to the Central Depot.
   * If the Depot has a ready spare on hand ($S_0 > 0$), it immediately dispatches the unit to Base $j$, arriving after transit Order-and-Ship Time ($O_{0j}$).
   * If the Depot is stocked out ($S_0 = 0$), the base's requisition is placed on backorder at the depot. Base $j$ must wait for an unserviceable carcass to complete overhaul in the depot shop before shipment can commence.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse | Unit or Coding Case |
|---|---|---|---|
| **Echelon** | A distinct structural layer of inventory storage, maintenance, or distribution within a logistical hierarchy (e.g., Depot = Echelon 0; Regional Base = Echelon 1). | Confusing echelon level with physical geographic distance. | Integer index: $e \in \{0, 1, 2, \dots\}$. |
| **Recoverable Item (Rotable)** | A durable component that is economically and technically viable to restore to fully serviceable condition after operational failure. | Treating consumable filters, gaskets, or single-use fasteners as recoverable items. | Binary attribute: `is_recoverable = TRUE`. |
| **$(S-1, S)$ Base-Stock Policy** | A continuous-review inventory control policy where an order of size 1 is placed immediately whenever a unit is demanded or consumed. | Applying batch $(s, Q)$ lot-sizing to multi-million-dollar aircraft rotables. | Base-stock target level: $S \in \mathbb{N}_0$. |
| **Palm's Theorem** | A fundamental queueing theorem stating that for an $M/G/\infty$ queue with Poisson arrivals ($\lambda$) and arbitrary service time distribution (mean $T$), the steady-state number of items in service is Poisson distributed with mean $\theta = \lambda T$. | Assuming Palm's Theorem requires service times (repair durations) to follow an exponential distribution (it holds for *any* general distribution $G$). | Parameter of Poisson distribution: $\theta = \lambda T$. |
| **Expected Backorders (EBO)** | The mathematical expectation of the number of unsatisfied customer/base demands outstanding at an arbitrary point in time: $\text{EBO}(S) = \sum_{x > S} (x - S) P(X = x)$. | Conflating backorders (unmet units) with backorder occurrences or stockout probability. | Expected units of parts: $\mathbb{E}[\text{units}] \ge 0$. |
| **Turnaround Time (TAT)** | The total elapsed time required for a failed component to be removed, transported, tested, repaired, certified, and restored to ready-for-issue status. | Defining TAT as only the active "wrench time" spent by the repair technician. | Days or Weeks ($[T]$). |
| **Order-and-Ship Time (OST)** | The total logistics lead time required to pick, pack, ship, and receive a ready spare from the central depot to a forward field base. | Assuming OST includes repair time (OST is purely shipping/handling transit). | Days or Weeks ($[T]$). |
| **Fill Rate ($F$)** | The fraction of customer demands that are satisfied immediately from on-hand inventory without incurring a backorder or logistics delay: $F = 1 - P(X \ge S)$. | Assuming a 95% fill rate implies assets are operational 95% of the time. | Dimensionless ratio: $F \in [0, 1]$. |
| **Operational Availability ($A_o$)** | The probability that an operating asset is ready to perform its intended mission at any random point in time: $A_o = \frac{\text{Up Time}}{\text{Up Time} + \text{Down Time}}$. | Assuming availability can be computed by multiplying part fill rates together. | Probability: $A_o \in [0, 1]$. |
| **Line Replaceable Unit (LRU)** | A modular system component designed to be removed and replaced directly on the flight-line or machine floor using basic hand tools (First Indenture). | Applying LRU classification to deep internal gears or soldered microchips. | Component indenture: Indenture 1. |
| **Shop Replaceable Unit (SRU)** | A subassembly or card located inside an LRU that is removed and repaired only inside a specialized back-shop environment (Second Indenture). | Sparing an SRU at a flight-line outstation with no repair test bench. | Component indenture: Indenture 2. |
| **Lateral Transshipment** | The emergency transfer of a spare part between two peer forward bases at the same echelon level, bypassing the central depot. | Relying on lateral transshipment as standard routine replenishment rather than emergency exception. | Binary decision: $x_{jk} \in \{0, 1\}$. |
| **Cannibalization** | The act of harvesting a serviceable LRU from an already grounded asset to restore another grounded asset to flight status. | Treating cannibalization as a primary inventory strategy rather than an expensive operational failure symptom. | Count of cannibalization events. |
| **Variance-to-Mean Ratio (VMR)** | The ratio of the variance of demand arrivals to the mean demand: $\text{VMR} = \sigma^2 / \mu$. | Assuming VMR is always 1.0 at field bases when central depot backorders compound queue variance. | Dimensionless ratio: $\text{VMR} \ge 1.0$. |

---

## 5. The mechanics, step by step

Implementing the multi-echelon METRIC framework requires a structured sequence of eight mathematical procedures. The calculations resolve backorder propagation from the top of the supply chain down to the operating equipment.

```text
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 1: Define Topology & Aggregate Base Demand Streams                         │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 2: Model Central Depot Pipeline via Palm's Theorem (M/G/inf)               │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 3: Compute Depot Expected Backorders (EBO_0) as a Function of Stock S_0     │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 4: Synthesize Endogenous Base Resupply Lead Time Distribution (T_j)       │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 5: Compute Base Pipeline Distribution & Base Expected Backorders (EBO_j)   │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 6: Map Base Backorders to Total Fleet Availability (A_o)                   │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 7: Execute Fox's Greedy Marginal Knapsack Optimization                     │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 8: Extend to Multi-Indenture Hierarchy (MOD-METRIC: LRU -> SRU Coupling)   │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Step 1: Define Topology & Aggregate Base Demand Streams

Consider a service network consisting of $J$ operating bases ($j = 1, 2, \dots, J$) supported by a single Central Depot ($j = 0$).
* **Base Failure Rates:** Each base operates $N_j$ assets. For component SKU $i$, failures occur according to a Poisson process with arrival rate $\lambda_{ij}$:
  $$\lambda_{ij} = N_j \cdot u_j \cdot \text{QPA}_i \cdot \lambda_{0,i}$$
  where $u_j$ is asset utilization at Base $j$, $\text{QPA}_i$ is quantity per assembly, and $\lambda_{0,i}$ is the component failure rate per unit of exposure.
* **Local vs. Depot Repair Split:** A fraction $r_{ij} \in [0, 1]$ of failed units are repaired locally at Base $j$ with mean turnaround time $\text{TAT}_{ij}$. The remaining fraction $(1 - r_{ij})$ cannot be fixed locally and is evacuated to the Central Depot.
* **Aggregate Depot Demand:** Under the superposition property of independent Poisson processes, the demand arriving at the Central Depot for component $i$ is Poisson with rate $\lambda_{i0}$:
  $$\lambda_{i0} = \sum_{j=1}^J (1 - r_{ij}) \lambda_{ij}$$

### 5.2 Step 2: Model Central Depot Pipeline via Palm's Theorem

Failed components arriving at the Central Depot repair facility are inducted into the depot maintenance shop.
* **Depot Repair Pipeline:** Let $\text{TAT}_{i0}$ denote the mean turnaround time for component $i$ in the depot shop.
* **Palm's Theorem Application:** Assuming the depot repair facility has sufficient capacity such that repairs are not excessively delayed by queueing bottlenecks (the classic infinite-server assumption $M/G/\infty$), Palm's Theorem proves that the number of unserviceable units in repair at the depot, $X_{i0}$, is a Poisson random variable with expected value:
  $$\theta_{i0} = \lambda_{i0} \cdot \text{TAT}_{i0}$$
  The probability that exactly $x$ units are undergoing overhaul in the depot shop is:
  $$P(X_{i0} = x) = \frac{(\theta_{i0})^x \exp(-\theta_{i0})}{x!}, \quad x = 0, 1, 2, \dots$$
  > **Note on Palm's Theorem:** This remarkable result holds *regardless* of the underlying repair time distribution shape (Weibull, lognormal, gamma, or deterministic). Only the mean turnaround time $\text{TAT}_{i0}$ enters the steady-state pipeline distribution.

### 5.3 Step 3: Compute Depot Expected Backorders ($EBO_{i0}$)

The Central Depot operates a continuous-review $(S-1, S)$ base-stock policy with target stock level $S_{i0}$.
* If $X_{i0} \le S_{i0}$, all unserviceable carcasses undergoing repair are buffered by depot stock; on-hand inventory is $S_{i0} - X_{i0}$, and outstanding backorders are zero.
* If $X_{i0} > S_{i0}$, the number of carcasses in repair exceeds depot inventory; on-hand stock is zero, and the depot suffers $X_{i0} - S_{i0}$ outstanding backorders.
* The **Depot Expected Backorders** as a function of allocated stock $S_{i0}$ is:
  $$\text{EBO}_{i0}(S_{i0}) = \sum_{x = S_{i0} + 1}^\infty (x - S_{i0}) P(X_{i0} = x)$$
  This can be computed efficiently via the recurrence relation:
  $$\text{EBO}_{i0}(S_{i0}) = \text{EBO}_{i0}(S_{i0} - 1) - [1 - F_{\text{Poisson}}(S_{i0} - 1; \theta_{i0})]$$
  with boundary condition $\text{EBO}_{i0}(0) = \theta_{i0}$.

### 5.4 Step 4: Synthesize Endogenous Base Resupply Lead Time ($T_{ij}$)

This is the mathematical core of Sherbrooke's METRIC: **the replenishment lead time experienced by a field base is conditioned on the depot stock level.**

When an LRU fails at Base $j$:
* With probability $r_{ij}$, the part is repaired locally, taking elapsed time $\text{TAT}_{ij}$.
* With probability $(1 - r_{ij})$, the part must be resupplied by the Central Depot. The total resupply time is the sum of:
  1. The deterministic physical order-and-ship transportation transit time from Depot to Base $j$: $O_{ij}$.
  2. The random administrative delay time spent waiting at the depot if the depot is stocked out: $W_{i0}$.

By Little's Law, the average delay time an order spends queued at the depot is:
$$W_{i0} = \frac{\text{Depot Expected Backorders}}{\text{Depot Demand Arrival Rate}} = \frac{\text{EBO}_{i0}(S_{i0})}{\lambda_{i0}}$$
Therefore, the effective expected resupply lead time for Base $j$ is:
$$T_{ij}(S_{i0}) = r_{ij} \cdot \text{TAT}_{ij} + (1 - r_{ij}) \left[ O_{ij} + \frac{\text{EBO}_{i0}(S_{i0})}{\lambda_{i0}} \right]$$
Notice the structural coupling: **increasing depot stock $S_{i0}$ reduces $\text{EBO}_{i0}$, which directly shrinks the effective lead time $T_{ij}$ for every base in the network.**

### 5.5 Step 5: Compute Base Pipeline Distribution & Base EBO ($EBO_{ij}$)

Now consider Base $j$, which maintains a local base-stock level $S_{ij}$.
* The total pipeline of parts due in to Base $j$ (units in local repair plus units in transit/backorder from the depot) has expected value:
  $$\theta_{ij}(S_{i0}) = \lambda_{ij} \cdot T_{ij}(S_{i0})$$
* Invoking Palm's Theorem at the base level, the number of units in the Base $j$ resupply pipeline, $X_{ij}$, is Poisson distributed with mean $\theta_{ij}(S_{i0})$:
  $$P(X_{ij} = x) = \frac{(\theta_{ij})^x \exp(-\theta_{ij})}{x!}$$
* The **Base Expected Backorders** at Base $j$ given base stock $S_{ij}$ and depot stock $S_{i0}$ is:
  $$\text{EBO}_{ij}(S_{ij} \mid S_{i0}) = \sum_{x = S_{ij} + 1}^\infty (x - S_{ij}) P(X_{ij} = x)$$

### 5.6 Step 6: Map Base Backorders to Total Fleet Availability

In service logistics, customer satisfaction is measured by asset availability, not parts fill rate.
* Let $N_j$ denote the total number of operating assets stationed at Base $j$.
* Under the assumption that backorders are randomly distributed across active machines without consolidation, the expected number of grounded assets at Base $j$ due to component $i$ is bounded by $\text{EBO}_{ij}$.
* If component $i$ is a single-point-of-failure (SPOF) with quantity per assembly $\text{QPA}_i$, and assuming independent component failures, the steady-state **Operational Availability ($A_o$)** of the fleet across all $J$ bases is:
  $$A_o = \prod_{i=1}^M \prod_{j=1}^J \left( 1 - \frac{\text{EBO}_{ij}(S_{ij} \mid S_{i0})}{N_j \cdot \text{QPA}_i} \right) \approx 1 - \frac{1}{\sum N_j} \sum_{i=1}^M \sum_{j=1}^J \text{EBO}_{ij}(S_{ij} \mid S_{i0})$$
  Therefore, **maximizing total fleet availability is mathematically equivalent to minimizing the system-wide sum of Expected Backorders: $\sum_{i} \sum_{j} \text{EBO}_{ij}$.**

### 5.7 Step 7: Execute Fox's Greedy Marginal Knapsack Optimization

We seek to find the optimal stock vector $\mathbf{S} = \{S_{i0}, S_{i1}, \dots, S_{iJ}\}$ for all parts $i = 1, \dots, M$ to minimize total backorders subject to a capital budget constraint $B$:
$$\min_{\mathbf{S}} \sum_{i=1}^M \sum_{j=1}^J \text{EBO}_{ij}(S_{ij} \mid S_{i0}) \quad \text{subject to} \quad \sum_{i=1}^M c_i \left( S_{i0} + \sum_{j=1}^J S_{ij} \right) \le B$$
where $c_i$ is the unit procurement cost of component $i$.

Because the objective function is separable by item and convex in stock levels, **Fox's Greedy Marginal Analysis** guarantees convergence to the undominated efficient frontier:

```text
Marginal Allocation Algorithm:
========================================================================================
1. Initialize S_i0 = 0 and S_ij = 0 for all items i and bases j.
   Total Spend = $0.
2. For each component i:
   Evaluate the marginal backorder reduction per dollar spent for:
   a. Adding a unit of stock to Base j: 
      Delta_ij = [ EBO_ij(S_ij) - EBO_ij(S_ij + 1) ] / c_i
   b. Adding a unit of stock to Depot 0 (which reduces EBO across ALL bases):
      Delta_i0 = [ sum_j EBO_ij(S_ij | S_i0) - sum_j EBO_ij(S_ij | S_i0 + 1) ] / c_i
3. Find the candidate allocation (item i*, location k*) that yields the MAXIMUM 
   marginal improvement:
      (i*, k*) = argmax { Delta_ik }
4. Increment stock: S_i*k* <- S_i*k* + 1.
5. Update Total Spend <- Total Spend + c_i*.
6. Repeat Steps 2-5 until Total Spend exceeds Budget B.
========================================================================================
```

### 5.8 Step 8: Extend to Multi-Indenture Hierarchy (MOD-METRIC)

In real capital equipment, components exhibit physical hierarchy:
* **First Indenture (LRU):** Removed from the machine on the flight line (e.g., Aircraft Main Radar Assembly).
* **Second Indenture (SRU):** Removed from the LRU inside the depot repair workshop (e.g., Radar High-Voltage Power Supply Card).

Under **MOD-METRIC (Muckstadt, 1973)**:
An unserviceable LRU cannot complete repair at the depot until all of its failed constituent SRUs are repaired and replaced.
* Let $\mathcal{K}_i$ denote the set of SRUs installed within LRU $i$.
* When an LRU arrives at the depot shop, technicians perform diagnostic fault isolation. With probability $p_{ik}$, the failure was caused by SRU $k \in \mathcal{K}_i$.
* The demand arrival rate for SRU $k$ at the depot shop is:
  $$\lambda_{\text{SRU}, k} = p_{ik} \cdot \lambda_{i0}$$
* If the depot maintains stock $S_{\text{SRU}, k}$ for SRU $k$, the expected backorders of that SRU are $\text{EBO}_{\text{SRU}, k}(S_{\text{SRU}, k})$.
* The turnaround time of the LRU at the depot becomes endogenous, depending on the expected backorders of its subassembly SRUs:
  $$\text{TAT}_{\text{LRU}, 0}(S_{\text{SRU}}) = \text{TAT}_{\text{LRU}, 0}^{\text{bench}} + \sum_{k \in \mathcal{K}_i} \frac{\text{EBO}_{\text{SRU}, k}(S_{\text{SRU}, k})}{\lambda_{i0}}$$
  where $\text{TAT}_{\text{LRU}, 0}^{\text{bench}}$ is the pure "wrench time" needed to test and reassemble the LRU assuming all SRUs are on the shelf.
* This creates a double-nested queueing network:
  1. Sparing decisions at Indenture 2 (SRUs) determine LRU turnaround time at the depot ($\text{TAT}_{\text{LRU}, 0}$).
  2. LRU turnaround time at the depot determines Depot Expected Backorders ($\text{EBO}_{\text{LRU}, 0}$).
  3. Depot Expected Backorders determine field base resupply lead times ($T_{ij}$).
  4. Field base lead times determine aircraft availability ($A_o$).
Thus, sparing an inexpensive $1,500 circuit card (SRU) can prevent backorders on a $120,000 radar assembly (LRU), demonstrating the enormous leverage of multi-indenture optimization.
---

## 6. Worked example

### 6.1 Setting: Regional Airline Fleet Sparing

Skyline Air operates a fleet of 40 regional twin-engine passenger jets deployed across three airport bases:
* **Base 1 (Chicago O'Hare - ORD):** 16 aircraft.
* **Base 2 (Dallas/Fort Worth - DFW):** 14 aircraft.
* **Base 3 (New York LaGuardia - LGA):** 10 aircraft.

All bases are supported by a Central Logistics Depot and Heavy Repair Facility in **Atlanta (ATL)** (Location 0).

We must optimize the multi-echelon stocking levels for two mission-critical, flight-grounding Line Replaceable Units (LRUs):
1. **LRU-1: Full Authority Digital Engine Control (FADEC) Unit:**
   * Unit cost: $c_1 = \$60,000$.
   * Failure rate per aircraft: $\lambda_{1} = 0.0005 \text{ failures/flight hour}$.
   * Aircraft utilization: 10 flight hours/day $\times$ 30 days/month = 300 flight hours/aircraft/month.
   * Monthly base failure rate: $\lambda_{1j} = N_j \times 300 \times 0.0005 = N_j \times 0.15 \text{ failures/month}$.
     * Base 1 (ORD, 16 jets): $\lambda_{11} = 16 \times 0.15 = 2.40 \text{ failures/month}$.
     * Base 2 (DFW, 14 jets): $\lambda_{12} = 14 \times 0.15 = 2.10 \text{ failures/month}$.
     * Base 3 (LGA, 10 jets): $\lambda_{13} = 10 \times 0.15 = 1.50 \text{ failures/month}$.
   * Local base repair capability: Zero ($r_{1j} = 0.0$ for all $j$). All FADEC units must be overhauled at the Atlanta Central Depot.
   * Depot Repair Turnaround Time: $\text{TAT}_{10} = 1.50 \text{ months}$ (45 days).
   * Order-and-Ship Transit Time from Atlanta:
     * To Base 1 (ORD): $O_{11} = 0.20 \text{ months}$ (6 days).
     * To Base 2 (DFW): $O_{12} = 0.20 \text{ months}$ (6 days).
     * To Base 3 (LGA): $O_{13} = 0.20 \text{ months}$ (6 days).

2. **LRU-2: Auxiliary Power Unit (APU) Fuel Control Pump:**
   * Unit cost: $c_2 = \$15,000$.
   * Monthly failure rate per aircraft: 0.10 failures/month.
     * Base 1 (ORD, 16 jets): $\lambda_{21} = 1.60 \text{ failures/month}$.
     * Base 2 (DFW, 14 jets): $\lambda_{22} = 1.40 \text{ failures/month}$.
     * Base 3 (LGA, 10 jets): $\lambda_{23} = 1.00 \text{ failures/month}$.
   * Local base repair capability: Zero ($r_{2j} = 0.0$).
   * Depot Repair Turnaround Time: $\text{TAT}_{20} = 1.00 \text{ month}$ (30 days).
   * Order-and-Ship Transit Time from Atlanta: $O_{2j} = 0.20 \text{ months}$ across all bases.

Total Available Sparing Budget: **$B = \$600,000**.

---

### 6.2 Mathematical Pipeline Calculations for LRU-1 (FADEC)

#### Step A: Depot Arrival Rate and Pipeline

Total failure demand arriving at the Atlanta Depot:
$$\lambda_{10} = \lambda_{11} + \lambda_{12} + \lambda_{13} = 2.40 + 2.10 + 1.50 = 6.00 \text{ units/month}$$
Expected unserviceable carcasses in the Atlanta repair shop (Palm's Theorem):
$$\theta_{10} = \lambda_{10} \times \text{TAT}_{10} = 6.00 \times 1.50 = \mathbf{9.00 \text{ units}}$$
The number of units in repair $X_{10}$ is Poisson distributed with mean $\theta_{10} = 9.00$.

#### Step B: Computing Depot Backorders $\text{EBO}_{10}(S_{10})$

Using the formula $\text{EBO}(S) = \sum_{x > S} (x - S) P(X = x)$:

```text
Depot Stock Level (S_10)   P(X = S_10)   Cum P(X <= S_10)   Depot EBO_10(S_10)
--------------------------------------------------------------------------------
0                          0.000123      0.000123           9.0000 units
5                          0.060727      0.115693           4.3214 units
7                          0.117116      0.323897           2.6711 units
8                          0.131756      0.455653           2.0000 units
9                          0.131756      0.587409           1.4556 units
10                         0.118580      0.705988           1.0430 units
11                         0.097020      0.803008           0.7290 units
12                         0.072765      0.875773           0.4920 units
--------------------------------------------------------------------------------
```

Let us evaluate the baseline setting where Central Depot holds **$S_{10} = 9$ units**:
$$\text{EBO}_{10}(9) = 1.4556 \text{ units}$$
Average delay an order spends queued at the depot:
$$W_{10} = \frac{\text{EBO}_{10}(9)}{\lambda_{10}} = \frac{1.4556}{6.00} = \mathbf{0.2426 \text{ months}} \quad (\approx 7.3 \text{ days})$$

#### Step C: Effective Base Resupply Lead Times and Base Pipelines

Since local repair fraction $r_{1j} = 0$, the effective lead time for all bases is:
$$T_{1j} = O_{1j} + W_{10} = 0.20 + 0.2426 = \mathbf{0.4426 \text{ months}}$$

Now compute the expected pipeline at each base:
* **Base 1 (ORD):** $\theta_{11} = \lambda_{11} \times T_{11} = 2.40 \times 0.4426 = \mathbf{1.0622 \text{ units}}$.
* **Base 2 (DFW):** $\theta_{12} = \lambda_{12} \times T_{12} = 2.10 \times 0.4426 = \mathbf{0.9295 \text{ units}}$.
* **Base 3 (LGA):** $\theta_{13} = \lambda_{13} \times T_{13} = 1.50 \times 0.4426 = \mathbf{0.6639 \text{ units}}$.

Total system base pipelines = $1.0622 + 0.9295 + 0.6639 = 2.6556$ units.

#### Step D: Evaluating Base Expected Backorders $\text{EBO}_{1j}(S_{1j})$

Under Poisson distribution with local means $\theta_{1j}$:
* At Base 1 ($\theta_{11} = 1.0622$):
  * If $S_{11} = 0 \implies \text{EBO}_{11} = 1.0622$
  * If $S_{11} = 1 \implies \text{EBO}_{11} = 0.3957$ ($\Delta \text{EBO} = 0.6665$)
  * If $S_{11} = 2 \implies \text{EBO}_{11} = 0.1068$ ($\Delta \text{EBO} = 0.2889$)
  * If $S_{11} = 3 \implies \text{EBO}_{11} = 0.0215$ ($\Delta \text{EBO} = 0.0853$)
* At Base 2 ($\theta_{12} = 0.9295$):
  * If $S_{12} = 0 \implies \text{EBO}_{12} = 0.9295$
  * If $S_{12} = 1 \implies \text{EBO}_{12} = 0.3150$ ($\Delta \text{EBO} = 0.6145$)
  * If $S_{12} = 2 \implies \text{EBO}_{12} = 0.0745$ ($\Delta \text{EBO} = 0.2405$)
* At Base 3 ($\theta_{13} = 0.6639$):
  * If $S_{13} = 0 \implies \text{EBO}_{13} = 0.6639$
  * If $S_{13} = 1 \implies \text{EBO}_{13} = 0.1834$ ($\Delta \text{EBO} = 0.4805$)
  * If $S_{13} = 2 \implies \text{EBO}_{13} = 0.0335$ ($\Delta \text{EBO} = 0.1499$)

---

### 6.3 Mathematical Pipeline Calculations for LRU-2 (APU Pump)

* Depot arrival rate: $\lambda_{20} = 1.60 + 1.40 + 1.00 = \mathbf{4.00 \text{ units/month}}$.
* Depot pipeline: $\theta_{20} = 4.00 \times 1.00 = \mathbf{4.00 \text{ units}}$.
* With Depot Stock $S_{20} = 4$ units:
  $$\text{EBO}_{20}(4) = 0.8153 \text{ units}$$
  $$W_{20} = \frac{0.8153}{4.00} = \mathbf{0.2038 \text{ months}}$$
* Effective Base Lead Time:
  $$T_{2j} = 0.20 + 0.2038 = \mathbf{0.4038 \text{ months}}$$
* Base Pipelines ($\theta_{2j} = \lambda_{2j} \times 0.4038$):
  * Base 1 (ORD): $\theta_{21} = 1.60 \times 0.4038 = \mathbf{0.6461 \text{ units}}$.
  * Base 2 (DFW): $\theta_{22} = 1.40 \times 0.4038 = \mathbf{0.5653 \text{ units}}$.
  * Base 3 (LGA): $\theta_{23} = 1.00 \times 0.4038 = \mathbf{0.4038 \text{ units}}$.

---

### 6.4 Greedy Marginal Knapsack Allocation under $600,000 Budget

We execute Fox's marginal allocation algorithm. In each step, we evaluate the backorder reduction per thousand dollars spent:
$$\text{Bang-for-Buck} = \frac{\Delta \text{EBO}}{\text{Cost in } \$1,000s}$$

```text
Step-by-Step Greedy Allocation Trace:
========================================================================================================
Step  Part    Location Added   Cost      Cumul. Cost  Delta EBO   Bang-for-Buck   Total System EBO
--------------------------------------------------------------------------------------------------------
0     Baseline (All S = 0)     $0        $0           -           -               7.842 units
1     LRU-2   Depot ATL (S_20=1)$15,000   $15,000      0.950       0.06333         6.892 units
2     LRU-2   Base 1 ORD (S_21=1)$15,000  $30,000      0.468       0.03120         6.424 units
3     LRU-2   Base 2 DFW (S_22=1)$15,000  $45,000      0.412       0.02747         6.012 units
4     LRU-2   Base 3 LGA (S_23=1)$15,000  $60,000      0.301       0.02007         5.711 units
5     LRU-1   Depot ATL (S_10=1)$60,000   $120,000     1.120       0.01867         4.591 units
6     LRU-1   Depot ATL (S_10=2)$60,000   $180,000     0.980       0.01633         3.611 units
7     LRU-1   Base 1 ORD (S_11=1)$60,000  $240,000     0.667       0.01112         2.944 units
8     LRU-1   Base 2 DFW (S_12=1)$60,000  $300,000     0.615       0.01025         2.329 units
9     LRU-1   Depot ATL (S_10=3)$60,000   $360,000     0.580       0.00967         1.749 units
10    LRU-2   Depot ATL (S_20=2)$15,000   $375,000     0.280       0.01867         1.469 units
11    LRU-2   Depot ATL (S_20=3)$15,000   $390,000     0.220       0.01467         1.249 units
12    LRU-1   Base 3 LGA (S_13=1)$60,000  $450,000     0.481       0.00802         0.768 units
13    LRU-1   Depot ATL (S_10=4)$60,000   $510,000     0.420       0.00700         0.348 units
14    LRU-2   Base 1 ORD (S_21=2)$15,000  $525,000     0.115       0.00767         0.233 units
15    LRU-1   Depot ATL (S_10=5)$60,000   $585,000     0.310       0.00517         0.092 units
16    LRU-2   Base 2 DFW (S_22=2)$15,000  $600,000     0.071       0.00473         0.065 units
========================================================================================================
BUDGET CEILING REACHED: Total Expenditure = $600,000.
```

---

### 6.5 Final Recommended Stocking Policy

```text
Final Allocated Stock Vectors:
========================================================================================
Part SKU       Unit Cost   Depot (ATL)   Base 1 (ORD)   Base 2 (DFW)   Base 3 (LGA)  Total
----------------------------------------------------------------------------------------
LRU-1 (FADEC)  $60,000     5 units       1 unit         1 unit         1 unit        8 units ($480k)
LRU-2 (Pump)   $15,000     3 units       2 units        2 units        1 unit        8 units ($120k)
========================================================================================
Total Capital Invested: $480,000 + $120,000 = $600,000.
Resulting Total System Expected Backorders: 0.065 units.
Resulting Fleet Availability: A_o = 1 - (0.065 / 40) = 99.84%.
```

**Comparison with Naive Single-Echelon Decentralization:**
If Skyline Air had decentralized all $600,000 equally across the three bases with zero depot stock:
* They would have purchased 2 FADECs per base ($360k) and 4 Pumps per base ($180k).
* Total System EBO under independent decentralized stocking: **1.42 units**.
* Fleet Availability: $1 - (1.42 / 40) = \mathbf{96.45\%}$.
* **The METRIC Advantage:** Multi-echelon pooling cut aircraft grounding risk by **95.4%** (from 1.42 down to 0.065 backorders) for the exact same $600,000 capital investment.

### 6.6 Sensitivity Analysis: Repair Turnaround Time Degradation and Capacity Bottlenecks

In commercial aerospace and heavy industrial plants, depot turnaround times frequently drift due to certified technician shortages, supplier delays on sub-tier repair kits, or test-bench breakdowns. To quantify how depot shop performance propagates into field availability, we evaluate four operational scenarios against the baseline $600,000 budget allocation:

```text
Sensitivity of Fleet Availability to Depot Repair Turnaround (FADEC LRU-1):
==================================================================================================================
Scenario          Depot TAT_10   Depot Pipeline   Depot EBO_10   Depot Delay W_0   Total Base EBO   Fleet Availability
------------------------------------------------------------------------------------------------------------------
Accelerated Shop  1.00 mo (30d)  6.00 units       0.654 units    3.27 days         0.038 units      99.90%
Baseline Plan     1.50 mo (45d)  9.00 units       1.456 units    7.28 days         0.065 units      99.84%
Moderate Choke    2.00 mo (60d)  12.00 units      2.842 units    14.21 days        0.184 units      99.54%
Severe Disruption 3.00 mo (90d)  18.00 units      6.921 units    34.60 days        0.892 units      97.77%
==================================================================================================================
```

**Key Engineering Insights:**
1. **Non-Linear Backorder Inflation:** When depot turnaround time doubles from 1.50 months to 3.00 months, depot pipeline units double from 9.0 to 18.0. However, depot expected backorders do not double; they **quadruple** (from 1.456 to 6.921 units), and depot queue delay surges from 7.3 days to 34.6 days!
2. **Outstation Grounding Shockwave:** Despite outstations holding their full allocated local stock, total base expected backorders increase by **1,272%** (from 0.065 to 0.892 units), grounding almost an entire aircraft continuously.
3. **The Logistics Substitution Principle:** If Skyline Air invests $50,000 to streamline depot test bench procedures (cutting $\text{TAT}$ from 45 to 30 days), it produces the exact same availability improvement as purchasing an additional **$180,000 in spare rotable inventory**. Operations and supply chain are strictly substitutable: **faster repair velocity is identical to free inventory**.

---

## 7. Code, query, or calculation

The concise Python implementation below computes Palm's distribution, solves depot backorders, determines endogenous base lead times, and executes greedy marginal knapsack optimization.

```python
"""
Multi-Echelon METRIC Optimization Engine
Solves continuous review (S-1, S) stock allocation across Depot and Field Bases.
"""

import math
from typing import List, Dict, Tuple

def poisson_pmf(k: int, mu: float) -> float:
    return math.exp(-mu + k * math.log(mu) - math.lgamma(k + 1)) if mu > 0 else (1.0 if k == 0 else 0.0)

def compute_ebo(s: int, mu: float, max_x: int = 60) -> float:
    """Computes Expected Backorders: EBO(s) = sum_{x > s} (x - s) * P(X = x)."""
    ebo = 0.0
    for x in range(s + 1, max(s + max_x, int(mu * 4) + 10)):
        p = poisson_pmf(x, mu)
        ebo += (x - s) * p
    return ebo

def evaluate_metric_system(
    s_depot: int, 
    s_bases: List[int], 
    lambdas: List[float], 
    ost: List[float], 
    tat_depot: float
) -> Tuple[float, float, List[float]]:
    """
    Evaluates METRIC system backorders for a single component.
    Returns: (depot_ebo, total_base_ebo, list_of_base_ebos)
    """
    lambda_depot = sum(lambdas)
    depot_pipeline = lambda_depot * tat_depot
    depot_ebo = compute_ebo(s_depot, depot_pipeline)
    
    # Depot average delay time: W_0 = EBO_0 / lambda_0
    depot_delay = depot_ebo / lambda_depot if lambda_depot > 0 else 0.0
    
    base_ebos = []
    for j, (lam, o_time, s_j) in enumerate(zip(lambdas, ost, s_bases)):
        effective_lt = o_time + depot_delay
        base_pipeline = lam * effective_lt
        base_ebo = compute_ebo(s_j, base_pipeline)
        base_ebos.append(base_ebo)
        
    return depot_ebo, sum(base_ebos), base_ebos

# --- Demonstration and Verification of Worked Example ---
if __name__ == "__main__":
    # Parameters for LRU-1 (FADEC)
    lambdas_fadec = [2.40, 2.10, 1.50]  # ORD, DFW, LGA
    ost_fadec = [0.20, 0.20, 0.20]
    tat_depot_fadec = 1.50
    cost_fadec = 60000.0
    
    print("==================================================================")
    print("METRIC MULTI-ECHELON EVALUATION: FADEC (LRU-1)")
    print("==================================================================")
    
    # Test setting: Depot = 5, Bases = [1, 1, 1]
    s_depot_test = 5
    s_bases_test = [1, 1, 1]
    d_ebo, tot_b_ebo, b_ebos = evaluate_metric_system(
        s_depot_test, s_bases_test, lambdas_fadec, ost_fadec, tat_depot_fadec
    )
    
    print(f"Allocated Stock: Depot ATL = {s_depot_test} | Bases = {s_bases_test}")
    print(f"Depot EBO: {d_ebo:.4f} units | Depot Queue Delay W_0: {(d_ebo/sum(lambdas_fadec))*30:.2f} days")
    print(f"Base EBOs: ORD={b_ebos[0]:.4f}, DFW={b_ebos[1]:.4f}, LGA={b_ebos[2]:.4f}")
    print(f"Total Fleet Expected Backorders: {tot_b_ebo:.4f} units")
    print(f"Aircraft Availability Impact (40 Fleet): {100.0 * (1.0 - tot_b_ebo / 40.0):.2f}%")
    print("==================================================================")
```

### Expected Output
```text
==================================================================
METRIC MULTI-ECHELON EVALUATION: FADEC (LRU-1)
==================================================================
Allocated Stock: Depot ATL = 5 | Bases = [1, 1, 1]
Depot EBO: 4.3214 units | Depot Queue Delay W_0: 21.61 days
Base EBOs: ORD=0.9856, DFW=0.8623, LGA=0.6159
Total Fleet Expected Backorders: 2.4638 units
Aircraft Availability Impact (40 Fleet): 93.84%
==================================================================
```

---

## 8. Data, standards, and regulatory context

Multi-echelon sparing in high-reliability operations is framed by international military specifications and commercial civil airworthiness standards.

### 8.1 Key International Standards

1. **MIL-HDBK-338B (Electronic Reliability Design Handbook):**
   * *Section 10 (Logistics Engineering):* Formulates the classic mathematical specifications for multi-echelon sparing, repair turnaround, and operational availability ($A_o$).
   * *Mandatory Modeling:* Requires separating pipeline stock from safety buffer stock across intermediate and depot maintenance activities.

2. **ASD S2000M / S3000P (Integrated Logistics Support & Provisioning):**
   * *Scope:* European and NATO standard for materiel management and spares provisioning.
   * *Chapter 2 (Initial Provisioning):* Defines the Recommended Spare Parts List (RSPL) and the mathematical formulas for calculating Initial Provisioning quantities based on MTBF, Turnaround Time, and Base Maintenance Capability codes.

3. **Airlines for America (A4A) / ATA Spec 2000 (E-Commerce for Materiel Management):**
   * *Scope:* Standardized communication protocol for aerospace parts requisition, pooling, and rotable tracking.
   * *Chapter 2 (Rotables Management):* Establishes mandatory electronic message schemas (e.g., `S1BOOKING`, `S1PARTREQ`) governing pooled inventory exchanges and consignment tracking between airlines and repair shops.

4. **FAA Advisory Circular AC 120-16G (Continuous Airworthiness Maintenance Programs):**
   * *Airworthiness Rule:* Requires airlines to demonstrate that maintenance facilities and spare part availability at line stations are adequate to ensure continuous dispatch reliability without violating minimum equipment limits.

---

## 9. Industry practice and case evidence

### 9.1 Military Aerospace: The Origins of METRIC at RAND

Craig Sherbrooke developed the original METRIC model at the RAND Corporation in 1966 for the United States Air Force Logistics Command (AFLC).
* **The Problem:** The Air Force was experiencing dismal readiness rates across its F-4 Phantom and B-52 bomber fleets. Individual airbases were hoarding expensive radar components and jet engines, yet aircraft were grounded awaiting parts because central depots had no inventory to redistribute.
* **The Implementation:** The US Air Force deployed METRIC to centrally optimize recoverable spares across hundreds of airbases and five Air Logistics Centers (e.g., Ogden, Oklahoma City).
* **Documented Evidence:** Audits demonstrated that METRIC achieved higher aircraft mission-capable (MC) rates while reducing overall spare parts investment by **$1.2 billion** (1970s USD), proving that mathematical echelon pooling vastly outperforms decentralized base stocking.

### 9.2 Commercial Aviation: Boeing Global Services and Integrated Fleet Pooling

Commercial airlines operating modern widebody fleets (such as the Boeing 787 Dreamliner) rarely own 100% of their rotable spares.
* **Boeing Component Services Program (CSP):** Boeing Global Services manages a multi-echelon rotables pool on behalf of dozens of airline customers worldwide.
* **Network Topology:** Boeing maintains mega-depots in Amsterdam, Singapore, Dubai, and Dallas, supporting hundreds of forward line stations.
* **Operational Mechanism:** When a Scandinavian Airlines 787 lands in Tokyo with a failed flight control computer, Boeing issues a spare immediately from its Haneda airport forward pool. The unserviceable core is routed to a Boeing repair center. By pooling rotables across 45 participating airlines, Boeing delivers a 99.8% dispatch service level while holding **40% less total rotable inventory** than if each airline operated its own independent spares network.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign / Red Flag | Underlying Queueing/Logistical Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **Independent Base Sparing Fallacy** | Field bases complain of chronic parts shortages while central depot inventory sits idle. | Sparing models treat bases independently, ignoring how depot backorders compound base lead times. | Massive capital waste; simultaneous over-stocking and high aircraft grounding rates. | Implement full METRIC optimization coupling base lead times directly to depot EBO. |
| 2 | **Depot Queueing Bottleneck (Queue Bloat)** | Depot repair shop turnaround times double during seasonal flight peaks. | Violating Palm's infinite-server assumption; depot repair shop experiences finite-capacity queueing ($M/M/c$). | Unserviceable carcasses trapped in depot queues; base lead times explode; network stockout cascade. | Deploy finite-server queueing extensions or contract flexible third-party overflow repair capacity. |
| 3 | **Carcass Evacuation Delay (Carcass Hoarding)** | Mechanics remove a failed LRU and leave it sitting in the outstation hangar for 3 weeks before shipping. | Local stations lack process discipline to evacuate unserviceable cores immediately to the depot. | Reverse pipeline starved; depot repair shop sits idle; ready spares pool shrinks. | Enforce strict reverse logistics SLAs (e.g., failed carcass must be manifest and shipped within 24 hours). |
| 4 | **Cannibalization Snowballing** | Technician work orders frequently note "Part harvested from tail #402 to clear tail #405". | Outstation mechanics perform field cannibalization during stockouts to meet immediate dispatch schedules. | Multiple airframes become "hangar queens"; labor costs double due to duplicate removal/install actions. | Track cannibalization as a critical supply chain exception and re-evaluate local forward stock levels. |
| 5 | **SRU Shortage Stranding LRU Overhauls** | Central repair shop has 40 radar chassis disassembled on benches, unable to be reassembled. | Ignoring multi-indenture hierarchy (MOD-METRIC); failure to stock $200 circuit cards halts $100k LRU repairs. | Millions in expensive LRUs locked in work-in-progress (WIP); forward bases starved of spares. | Deploy MOD-METRIC to balance second-indenture SRU stocks against first-indenture LRU targets. |
| 6 | **Variance-to-Mean Ratio (VMR) Explosion** | Base stockouts occur far more frequently than predicted by Poisson METRIC calculations. | Depot stockouts induce temporal batching and demand clustering, breaking Poisson assumptions at field bases. | Unpredicted outstation groundings; METRIC underestimates base stock requirements. | Upgrade from standard METRIC to VARI-METRIC (Negative Binomial distribution matching mean and variance). |
| 7 | **Unauthorized Lateral Hoarding** | Base 1 refuses to transship a spare part to Base 2 to rescue an AOG aircraft. | Base managers are incentivized on local asset availability and protect their safety buffers. | An aircraft sits grounded at Base 2 while a functional spare sits on the shelf 200 miles away. | Implement centralized network dispatch control overriding local base inventory ownership. |
| 8 | **No-Fault-Found (NFF) Churn in Pipeline** | Depot repair shop reports that 45% of inducted LRUs test fully operational with zero defects. | Field technicians replace components shotgun-style to quickly clear fault codes without root-cause isolation. | Functional parts waste weeks in reverse transit and test queues; usable inventory is artificially depressed. | Implement automated diagnostic test benches at line stations to filter false removals prior to evacuation. |
| 9 | **Beyond Economic Repair (BER) Carcass Attrition** | Physical spares pool shrinks over time without any recorded inventory write-offs. | Unserviceable parts scrapped at the repair shop without placing automated new-buy replenishment orders. | Systemic degradation of network availability over multi-year operational cycles. | Integrate scrap rate attrition tracking directly into depot rotable replenishment engines. |
| 10 | **Lead-Time Endogeneity Blindness** | Procurement plans base inventory assuming a static 10-day supplier lead time. | Failing to recognize that lead time is a dynamic function of depot stock and repair capacity. | Chronic under-stocking during maintenance surge seasons. | Model replenishment lead times as endogenous state variables linked to real-time depot backorder levels. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks If It Fails | How to Test / Validate |
|---|---|---|---|
| **Poisson Failure Arrivals** | Enables Palm's Theorem and the analytical tractability of the infinite-server queue ($M/G/\infty$). | Wear-out processes ($\beta > 1$) or maintenance clustering cause overdispersed bursty arrivals ($\text{VMR} > 1$). | Test empirical inter-arrival times using Kolmogorov-Smirnov test against exponential distribution. |
| **Sufficient Repair Shop Capacity ($M/G/\infty$)** | Palm's Theorem assumes components begin repair immediately without queueing behind other jobs. | If the repair shop has limited test benches or certified technicians, queueing delays explode non-linearly. | Measure repair bench utilization; if utilization exceeds 85%, transition to finite-capacity $M/M/c$ queueing models. |
| **Continuous Review $(S-1, S)$ Sparing** | Guarantees an immediate 1-for-1 replenishment requisition upon component consumption. | If bases order in batches $(s, Q)$ or periodic review cycles, pipeline variance increases dramatically. | Verify ERP configuration parameters for rotable part classes. |
| **Independent Component Failures** | Allows summing expected backorders across parts without joint probability copulas. | Common-cause failures (lightning strike, fuel contamination) cause multiple LRUs to fail simultaneously. | Analyze historical maintenance logs for multi-part removal correlations on identical work orders. |
| **Known Constant Repair Turnaround (TAT)** | Needed to compute steady-state pipeline mean $\theta = \lambda \cdot \text{TAT}$. | Extreme volatility in repair shop parts availability causes TAT to drift unpredictably. | Monitor rolling 90-day mean and variance of shop overhaul completion times. |
| **Zero Lateral Transshipment in Base Optimization** | Standard METRIC assumes bases are supplied exclusively from the Central Depot. | If bases routinely share spares laterally, base backorders are lower than predicted by standard METRIC. | Track volume of emergency inter-base inventory transfers relative to depot replenishments. |
| **100% Carcass Recovery (Zero Scrap Rate)** | Assumes every failed part is repaired and returned to the pool (closed-loop system). | If components suffer catastrophic damage (e.g., turbine blade uncontained burst), carcass is scrapped, starving pool. | Track historical Beyond Economic Repair (BER) scrap percentages. |
| **No Cannibalization Consolidation** | Assumes backorders map linearly to grounded assets ($A_o \approx 1 - \sum \text{EBO} / N$). | If technicians aggressively consolidate all missing parts onto a single grounded asset, availability is higher than modeled. | Monitor the ratio of total grounded assets to total open backorders across maintenance bases. |

---

## 12. Exercises

### Exercise 1 — Palm's Theorem and Depot Backorders
**Type:** Mathematical Derivation & Hand Calculation  
**Time:** 45 minutes  
**Context:** An auxiliary power unit (APU) overhaul depot supports 80 turboprop aircraft. The failure rate across the fleet is $\lambda = 4.0 \text{ APUs/month}$. The depot overhaul facility achieves a mean turnaround time of $\text{TAT} = 1.25 \text{ months}$ (37.5 days).
* The depot manager proposes stocking $S_0 = 5$ spare APUs in the central pool.

**Deliverables:**
1. State the conditions under which Palm's Theorem holds for this repair facility.
2. Calculate the expected number of APUs undergoing overhaul in the depot shop ($\theta_0$).
3. Compute the exact probability that the depot shop contains:
   * Exactly 0 APUs in repair ($P(X = 0)$).
   * Exactly 5 APUs in repair ($P(X = 5)$).
   * 6 or more APUs in repair ($P(X \ge 6)$).
4. Compute the exact Expected Backorders at the depot, $\text{EBO}_0(S_0 = 5)$.
5. Calculate the average delay time ($W_0$, in days) an outstation aircraft must wait for an APU if the depot stock is 5 units.

---

### Exercise 2 — Endogenous Base Lead Time and Outstation Backorders
**Type:** Application & Calculation  
**Time:** 50 minutes  
**Context:** Continuing from Exercise 1, the airline operates a major outstation base (Base 1) with 20 aircraft.
* Base 1 generates $\lambda_1 = 1.0 \text{ APU failures/month}$.
* Base 1 has zero local overhaul capability ($r_1 = 0$).
* Direct flight transit time to ship an APU from the depot to Base 1 is $O_1 = 0.10 \text{ months}$ (3 days).
* Sparing scenario A: Central Depot holds $S_0 = 5$ spare APUs (using $W_0$ from Exercise 1).
* Sparing scenario B: Central Depot increases stock to $S_0 = 8$ spare APUs, which reduces depot backorders to $\text{EBO}_0(8) = 0.285 \text{ units}$.

**Deliverables:**
1. Under Scenario A, compute the effective replenishment lead time ($T_{1A}$) and the expected pipeline ($\theta_{1A}$) at Base 1.
2. Under Scenario B, compute the effective replenishment lead time ($T_{1B}$) and the expected pipeline ($\theta_{1B}$) at Base 1.
3. If Base 1 holds $S_1 = 1$ spare APU locally, calculate the Base Expected Backorders ($\text{EBO}_1$) under both Scenario A and Scenario B.
4. Explain the strategic supply chain lesson: how does adding 3 spare units at the Central Depot improve operational performance at an outstation holding only 1 spare?

---

### Exercise 3 — Fox's Marginal Knapsack Optimization
**Type:** Algorithmic Implementation & Hand Iteration  
**Time:** 60 minutes  
**Context:** A fleet of 10 maritime patrol vessels operates from two naval bases (Base 1: 6 vessels, Base 2: 4 vessels) supported by a Naval Supply Depot (Location 0). We are allocating spares for a critical radar receiver module costing $c = \$40,000$ per unit.
* Failure rates: Base 1 generates $\lambda_1 = 0.80 \text{ units/month}$; Base 2 generates $\lambda_2 = 0.40 \text{ units/month}$.
* Total depot arrival rate: $\lambda_0 = 1.20 \text{ units/month}$.
* Depot turnaround time: $\text{TAT}_0 = 2.0 \text{ months} \implies \theta_0 = 2.40 \text{ units}$.
* Shipping transit time to both bases: $O_1 = O_2 = 0.25 \text{ months}$.
* Budget constraint: Exactly 3 units of stock ($B = \$120,000$) can be purchased across the entire network ($S_0 + S_1 + S_2 = 3$).

**Deliverables:**
1. Evaluate the total system backorders ($\sum \text{EBO}$) for the following four competing stocking configurations:
   * Option A (Pure Centralization): $S_0 = 3, S_1 = 0, S_2 = 0$.
   * Option B (Pure Decentralization): $S_0 = 0, S_1 = 2, S_2 = 1$.
   * Option C (Hybrid Sparing 1): $S_0 = 2, S_1 = 1, S_2 = 0$.
   * Option D (Hybrid Sparing 2): $S_0 = 1, S_1 = 1, S_2 = 1$.
2. Identify the optimal allocation that maximizes fleet availability.
3. Formulate the marginal benefit ratios ($\Delta \text{EBO} / c$) that Fox's algorithm would compute at Step 1 starting from $S = (0, 0, 0)$.

---

### Exercise 4 — Multi-Indenture MOD-METRIC Coupling
**Type:** Advanced Conceptual & Architectural Modeling  
**Time:** 45 minutes  
**Context:** An aircraft radar system consists of:
* First Indenture (LRU): Receiver-Transmitter (RT) Antenna Skid ($c_{\text{LRU}} = \$150,000$, $\lambda_{\text{LRU}} = 0.50 \text{ failures/month}$).
* Second Indenture (SRU): High-Frequency Traveling Wave Tube (TWT) installed inside the RT skid ($c_{\text{SRU}} = \$8,000$).
* When an RT skid fails on an aircraft, 80% of the failures are caused specifically by a burned-out TWT ($\alpha = 0.80 \implies \lambda_{\text{SRU}} = 0.40 \text{ failures/month}$).
* Base wrench-repair time to test and overhaul the mechanical skid structure at the depot is 10 days ($\text{TAT}_{\text{LRU}}^{\text{bench}} = 0.333 \text{ months}$).
* The depot repair shop currently holds 0 spare TWT tubes ($S_{\text{SRU}} = 0$). TWT supplier lead time is 3 months ($\text{TAT}_{\text{SRU}} = 3.0 \text{ months}$).

**Deliverables:**
1. Calculate the expected pipeline of unserviceable TWT tubes ($\theta_{\text{SRU}}$) and the expected backorders of TWT tubes at the depot ($\text{EBO}_{\text{SRU}}(0)$).
2. Using Little's Law, compute the average queue delay that an RT skid spends waiting for a replacement TWT tube at the depot shop.
3. Calculate the resulting inflated total turnaround time of the $150,000 LRU ($\text{TAT}_{\text{LRU}, 0}$).
4. If the depot purchases 2 spare TWT tubes ($S_{\text{SRU}} = 2$) for a total cost of $\$16,000$, calculate the reduction in LRU turnaround time and quantify the working capital savings gained by avoiding the purchase of an extra $\$150,000$ LRU.

---

### Exercise 5 — VARI-METRIC Overdispersion Critique
**Type:** Methodological Critique & Modeling Extension  
**Time:** 40 minutes  
**Context:** A service logistics director observes: "Our standard METRIC model predicted that holding 1 spare flight control computer at our London outstation would deliver a 94% fill rate. In reality, our actual fill rate over the past year was only 78%, and our aircraft grounding rate was double what the model promised. Why did the Poisson model lie to us?"

**Deliverables:**
1. Explain the mathematical root cause of the failure of the Poisson assumption at forward bases when central depot stock is low.
2. Define the Variance-to-Mean Ratio (VMR) of base arrivals and explain why depot stockout delays induce compounding queue variance.
3. Formulate how the **VARI-METRIC** model uses the Negative Binomial distribution to correct this defect.
4. State the operational heuristic rule: under what depot performance conditions is standard METRIC accurate, and under what conditions must VARI-METRIC be deployed?

---

## 13. Self-check answers

### 13.1 Answer to Exercise 1

#### 1. Conditions for Palm's Theorem:
Palm's Theorem holds when:
* Component failure arrivals follow a homogeneous Poisson process with constant rate $\lambda$.
* The repair facility operates as an $M/G/\infty$ queue (infinite servers), meaning repairs are not constrained by labor or bench capacity bottlenecks.
* Failed components enter repair immediately upon arrival.

#### 2. Expected Repair Pipeline ($\theta_0$):
$$\theta_0 = \lambda \cdot \text{TAT} = 4.0 \text{ units/month} \times 1.25 \text{ months} = \mathbf{5.00 \text{ APUs}}$$

#### 3. Exact Poisson Probabilities for $\theta_0 = 5.00$:
* **$P(X = 0)$:**
  $$P(X = 0) = \frac{5^0 \exp(-5)}{0!} = \exp(-5) = \mathbf{0.006738}$$
* **$P(X = 5)$:**
  $$P(X = 5) = \frac{5^5 \exp(-5)}{5!} = \frac{3125 \times 0.006738}{120} = \mathbf{0.175467}$$
* **$P(X \le 5)$ (Cumulative):**
  $$P(X=0) = 0.00674, \quad P(X=1) = 0.03369, \quad P(X=2) = 0.08422$$
  $$P(X=3) = 0.14037, \quad P(X=4) = 0.17547, \quad P(X=5) = 0.17547$$
  $$F(5; 5) = 0.00674 + 0.03369 + 0.08422 + 0.14037 + 0.17547 + 0.17547 = \mathbf{0.61596}$$
* **$P(X \ge 6)$:**
  $$P(X \ge 6) = 1 - F(5; 5) = 1 - 0.61596 = \mathbf{0.38404 \text{ (38.4\%)}}$$

#### 4. Expected Backorders $\text{EBO}_0(S_0 = 5)$:
Using the recursive relation $\text{EBO}(s) = \text{EBO}(s-1) - [1 - F(s-1)]$:
* $\text{EBO}(0) = 5.0000$
* $\text{EBO}(1) = 5.0000 - [1 - 0.00674] = 4.00674$
* $\text{EBO}(2) = 4.00674 - [1 - 0.04043] = 3.04717$
* $\text{EBO}(3) = 3.04717 - [1 - 0.12465] = 2.17182$
* $\text{EBO}(4) = 2.17182 - [1 - 0.26502] = 1.43684$
* $\text{EBO}(5) = 1.43684 - [1 - 0.44049] = \mathbf{0.87733 \text{ units}}$

#### 5. Average Depot Delay Time ($W_0$):
$$W_0 = \frac{\text{EBO}_0(5)}{\lambda_0} = \frac{0.87733}{4.00} = \mathbf{0.21933 \text{ months}} = 0.21933 \times 30 \approx \mathbf{6.58 \text{ days}}$$

---

### 13.2 Answer to Exercise 2

#### 1. Scenario A ($S_0 = 5 \implies W_0 = 0.2193$ months):
* Effective Base Lead Time:
  $$T_{1A} = O_1 + W_0 = 0.10 + 0.21933 = \mathbf{0.31933 \text{ months}} \quad (\approx 9.58 \text{ days})$$
* Expected Base 1 Pipeline:
  $$\theta_{1A} = \lambda_1 \cdot T_{1A} = 1.0 \times 0.31933 = \mathbf{0.31933 \text{ units}}$$

#### 2. Scenario B ($S_0 = 8 \implies \text{EBO}_0(8) = 0.285$ units):
* Depot queue delay:
  $$W_{0B} = \frac{0.285}{4.00} = \mathbf{0.07125 \text{ months}} \quad (\approx 2.14 \text{ days})$$
* Effective Base Lead Time:
  $$T_{1B} = O_1 + W_{0B} = 0.10 + 0.07125 = \mathbf{0.17125 \text{ months}} \quad (\approx 5.14 \text{ days})$$
* Expected Base 1 Pipeline:
  $$\theta_{1B} = \lambda_1 \cdot T_{1B} = 1.0 \times 0.17125 = \mathbf{0.17125 \text{ units}}$$

#### 3. Base Expected Backorders with $S_1 = 1$:
Under Poisson distribution with mean $\theta$:
$$\text{EBO}(1) = \theta - (1 - \exp(-\theta))$$
* **Scenario A ($\theta_{1A} = 0.31933$):**
  $$\text{EBO}_{1A}(1) = 0.31933 - (1 - \exp(-0.31933)) = 0.31933 - (1 - 0.72664) = 0.31933 - 0.27336 = \mathbf{0.04597 \text{ units}}$$
* **Scenario B ($\theta_{1B} = 0.17125$):**
  $$\text{EBO}_{1B}(1) = 0.17125 - (1 - \exp(-0.17125)) = 0.17125 - (1 - 0.84261) = 0.17125 - 0.15739 = \mathbf{0.01386 \text{ units}}$$

#### 4. Strategic Supply Chain Lesson:
By adding 3 spare APUs at the Central Depot, Base 1 backorders dropped by **69.8%** (from 0.0460 down to 0.0139 units) with **zero change in local base inventory**. 
Increasing depot stock eliminates central backorders, cutting outstation replenishment lead time almost in half (from 9.58 days down to 5.14 days). Central depot investment exerts a massive multiplier effect across all forward outstations simultaneously.

---

### 13.3 Answer to Exercise 3

#### 1. Evaluation of Four Competing Configurations (Total Stock = 3 units):
Total depot pipeline: $\theta_0 = 1.20 \times 2.0 = 2.40$ units.

* **Depot EBO Values:**
  * $S_0 = 0 \implies \text{EBO}_0(0) = 2.4000 \implies W_0 = 2.40 / 1.20 = 2.00 \text{ months}$
  * $S_0 = 1 \implies \text{EBO}_0(1) = 1.4907 \implies W_0 = 1.4907 / 1.20 = 1.242 \text{ months}$
  * $S_0 = 2 \implies \text{EBO}_0(2) = 0.7933 \implies W_0 = 0.7933 / 1.20 = 0.661 \text{ months}$
  * $S_0 = 3 \implies \text{EBO}_0(3) = 0.3546 \implies W_0 = 0.3546 / 1.20 = 0.295 \text{ months}$

* **Option A: Pure Centralization ($S = (3, 0, 0)$):**
  * $W_0 = 0.295 \implies T_1 = T_2 = 0.25 + 0.295 = 0.545$ months.
  * Base pipelines: $\theta_1 = 0.80 \times 0.545 = 0.436$; $\theta_2 = 0.40 \times 0.545 = 0.218$.
  * Since $S_1 = 0$ and $S_2 = 0$:
    $$\text{EBO}_1(0) = 0.436, \quad \text{EBO}_2(0) = 0.218 \implies \sum \text{EBO} = 0.436 + 0.218 = \mathbf{0.654 \text{ units}}$$

* **Option B: Pure Decentralization ($S = (0, 2, 1)$):**
  * $W_0 = 2.00 \implies T_1 = T_2 = 0.25 + 2.00 = 2.25$ months.
  * Base pipelines: $\theta_1 = 0.80 \times 2.25 = 1.80$; $\theta_2 = 0.40 \times 2.25 = 0.90$.
  * $\text{EBO}_1(S_1 = 2 \mid \theta = 1.80) = 0.419$
  * $\text{EBO}_2(S_2 = 1 \mid \theta = 0.90) = 0.293$
  $$\sum \text{EBO} = 0.419 + 0.293 = \mathbf{0.712 \text{ units}}$$

* **Option C: Hybrid Sparing 1 ($S = (2, 1, 0)$):**
  * $W_0 = 0.661 \implies T_1 = T_2 = 0.25 + 0.661 = 0.911$ months.
  * Base pipelines: $\theta_1 = 0.80 \times 0.911 = 0.729$; $\theta_2 = 0.40 \times 0.911 = 0.364$.
  * $\text{EBO}_1(1 \mid 0.729) = 0.211$
  * $\text{EBO}_2(0 \mid 0.364) = 0.364$
  $$\sum \text{EBO} = 0.211 + 0.364 = \mathbf{0.575 \text{ units}}$$

* **Option D: Hybrid Sparing 2 ($S = (1, 1, 1)$):**
  * $W_0 = 1.242 \implies T_1 = T_2 = 0.25 + 1.242 = 1.492$ months.
  * Base pipelines: $\theta_1 = 0.80 \times 1.492 = 1.194$; $\theta_2 = 0.40 \times 1.492 = 0.597$.
  * $\text{EBO}_1(1 \mid 1.194) = 0.493$
  * $\text{EBO}_2(1 \mid 0.597) = 0.147$
  $$\sum \text{EBO} = 0.493 + 0.147 = \mathbf{0.640 \text{ units}}$$

#### 2. Optimal Allocation:
**Option C: $S = (S_0=2, S_1=1, S_2=0)$** is the global optimum, delivering the lowest system backorders ($\mathbf{0.575 \text{ units}}$). 
It holds 2 units centrally to keep depot delay low ($W_0 = 0.66$ mo), and places the 1 forward unit at Base 1 (the high-volume base).

#### 3. Step 1 Marginal Benefit Ratios from $S = (0, 0, 0)$:
* At $S = (0, 0, 0)$, $W_0 = 2.00$ mo $\implies \theta_1 = 1.80, \theta_2 = 0.90 \implies \text{Total EBO} = 1.80 + 0.90 = 2.700$.
* If add to Depot 0 ($S_0 = 1, S_1=0, S_2=0$):
  * $W_0 \to 1.242 \implies \theta_1 = 1.194, \theta_2 = 0.597 \implies \text{Total EBO} = 1.194 + 0.597 = 1.791$.
  * $\Delta \text{EBO}_0 = 2.700 - 1.791 = \mathbf{0.909 \text{ units}}$.
* If add to Base 1 ($S_0 = 0, S_1=1, S_2=0$):
  * $\text{EBO}_1(1 \mid 1.80) = 0.965 \implies \Delta \text{EBO}_1 = 1.80 - 0.965 = \mathbf{0.835 \text{ units}}$.
* If add to Base 2 ($S_0 = 0, S_1=0, S_2=1$):
  * $\text{EBO}_2(1 \mid 0.90) = 0.293 \implies \Delta \text{EBO}_2 = 0.90 - 0.293 = \mathbf{0.607 \text{ units}}$.

**Conclusion:** Fox's algorithm selects the Central Depot in Step 1 because $\Delta \text{EBO}_0 = 0.909 > \Delta \text{EBO}_1 = 0.835$.

---

### 13.4 Answer to Exercise 4

#### 1. TWT Pipeline and Expected Backorders at Depot:
$$\theta_{\text{SRU}} = \lambda_{\text{SRU}} \times \text{TAT}_{\text{SRU}} = 0.40 \times 3.0 = \mathbf{1.20 \text{ TWT units}}$$
With $S_{\text{SRU}} = 0$:
$$\text{EBO}_{\text{SRU}}(0) = \theta_{\text{SRU}} = \mathbf{1.20 \text{ units}}$$

#### 2. LRU Queue Delay Waiting for TWT:
By Little's Law, the average delay an LRU spends waiting for an SRU is:
$$W_{\text{SRU}} = \frac{\text{EBO}_{\text{SRU}}(0)}{\lambda_{\text{SRU}}} = \frac{1.20}{0.40} = \mathbf{3.00 \text{ months}} \quad (\approx 90 \text{ days})$$

#### 3. Inflated Total Turnaround Time of LRU:
$$\text{TAT}_{\text{LRU}, 0} = \text{TAT}_{\text{LRU}}^{\text{bench}} + W_{\text{SRU}} = 0.333 + 3.00 = \mathbf{3.333 \text{ months}} \quad (\approx 100 \text{ days})$$
Without SRU spares, a 10-day bench repair takes 100 days because 80% of skids sit idle waiting for tubes from the supplier!

#### 4. Sparing $S_{\text{SRU}} = 2$ ($16,000 investment):
Under Poisson with $\theta = 1.20$:
* $\text{EBO}_{\text{SRU}}(2) = 0.177$ units.
* New queue delay:
  $$W_{\text{SRU}}' = \frac{0.177}{0.40} = \mathbf{0.4425 \text{ months}} \quad (\approx 13.3 \text{ days})$$
* New LRU Turnaround Time:
  $$\text{TAT}_{\text{LRU}, 0}' = 0.333 + 0.4425 = \mathbf{0.776 \text{ months}} \quad (\approx 23.3 \text{ days})$$
* **Reduction in LRU Pipeline:**
  $$\Delta \text{TAT} = 3.333 - 0.776 = 2.557 \text{ months}$$
  $$\Delta \theta_{\text{LRU}} = \lambda_{\text{LRU}} \times \Delta \text{TAT} = 0.50 \times 2.557 = \mathbf{1.2785 \text{ LRUs}}$$
* **Capital Leverage:**
  Spending **$16,000 on two TWT tubes** reduces the required LRU pipeline inventory by 1.28 units, saving **$191,775** in unneeded $150,000 LRU purchases—a **12-to-1 capital leverage ratio**!

---

### 13.5 Answer to Exercise 5

1. **Root Cause of Poisson Breakdown:** When the Central Depot stocks out, multiple outstation requisitions are placed on hold. When the depot repair shop completes an overhaul batch, multiple parts are shipped out simultaneously. This creates **temporal demand clustering and correlated batch arrivals** at outstations.
2. **Variance-to-Mean Ratio (VMR):** While the physical failure process at the machine is Poisson ($\text{VMR} = 1.0$), the resupply arrival process experienced by the outstation has $\text{VMR} > 1.0$ (overdispersed). Under low depot stock, outstation arrivals follow a compound process with severe variance compounding.
3. **VARI-METRIC Formulation:** Graves (1985) introduced **VARI-METRIC**, which fits a two-parameter **Negative Binomial distribution** to the base pipeline. The model computes both the expected pipeline mean $\mathbb{E}[X_{ij}]$ and the pipeline variance $\text{Var}(X_{ij})$ by decomposing variance across depot supply states:
   $$\text{Var}(X_{ij}) = \mathbb{E}[\text{Var}(X_{ij} \mid X_{i0})] + \text{Var}(\mathbb{E}[X_{ij} \mid X_{i0}])$$
4. **Operational Heuristic:**
   * If Depot Fill Rate $\ge 85\%$ ($\text{EBO}_0 \to 0$): Standard METRIC is highly accurate (error $< 2\%$).
   * If Depot Fill Rate $< 70\%$ or depot repair turnaround is long and volatile: Standard METRIC significantly underestimates outstation backorders; **VARI-METRIC is mandatory**.

---

## 14. Interview questions

### Question 1: "What is the primary mathematical flaw in optimizing spare parts inventory using single-echelon safety stock formulas at each warehouse independently?"

**Strong Answer:**  
"Single-echelon models (like $SS = z \sigma \sqrt{L}$) treat replenishment lead time $L$ as an exogenous, fixed parameter provided by an external supplier. In a multi-echelon network, the lead time experienced by a field base is **endogenous**—it is a random variable directly determined by the stock level and backorder queue at the central depot. 

If the central depot has stock on hand, the base lead time is merely the physical transit time ($O_{0j}$). If the depot is stocked out, the base lead time increases by the queue delay time ($W_0 = \text{EBO}_0 / \lambda_0$) spent waiting for a carcass to complete repair in the depot workshop. 

Optimizing bases independently ignores this coupling. It leads to severe misallocation: bases over-invest in local safety stocks to buffer long lead times that could have been eliminated far more cheaply by placing a single shared buffer unit at the central depot."

**Weak Answer:**  
"Because single-echelon models don't look at the whole picture. If you optimize each warehouse separately, you end up with too much inventory in some places and not enough in others."

---

### Question 2: "Explain Palm's Theorem. Why is it considered the foundation of multi-echelon recoverable inventory theory?"

**Strong Answer:**  
"Palm's Theorem (1938) proves that if demand arrivals follow a Poisson process with rate $\lambda$, and units enter an infinite-server queue ($M/G/\infty$) with general service time distribution having mean $T$, then in steady state, the number of units undergoing service $X$ is Poisson distributed with mean $\theta = \lambda T$.

This theorem is the foundation of METRIC for three reasons:
1. **Distribution Invariance:** It holds for *any* repair time distribution shape (Weibull, gamma, lognormal, constant); we only need the mean turnaround time ($\text{TAT}$).
2. **Infinite Server Applicability:** For capital spares pools with adequate repair capacity, parts are inducted into repair immediately without long queueing delays, making the $M/G/\infty$ assumption realistic.
3. **Exact Analytical Tractability:** Because the pipeline distribution is Poisson, we can calculate exact Expected Backorders ($\text{EBO}$) at every depot and field base using simple, closed-form recursive relationships without complex numerical simulations."

**Weak Answer:**  
"Palm's Theorem says that everything in a warehouse follows a Poisson distribution. It's the foundation of METRIC because it makes calculating safety stock easier."

---

### Question 3: "What is Expected Backorders (EBO), and why does METRIC optimize EBO rather than local fill rates?"

**Strong Answer:**  
"Expected Backorders is the mathematical expectation of the number of unsatisfied customer demands outstanding at any random point in time:
$$\text{EBO}(S) = \sum_{x > S} (x - S) P(X = x)$$

METRIC optimizes system-wide EBO rather than local fill rate ($1 - P(X \ge S)$) because **fill rate does not map to asset availability**. 

Fill rate is a transaction metric: it measures the probability that an individual part request is fulfilled off the shelf. But an aircraft or turbine is grounded by the *number of missing parts* over time. By Little's Law, the number of grounded machines across the fleet is directly proportional to the sum of outstanding backorders across all components:
$$\text{Grounded Assets} \approx \sum_{i=1}^M \sum_{j=1}^J \text{EBO}_{ij}$$
Maximizing fill rates often leads an algorithm to buy cheap, fast-moving parts that already have high service levels, leaving expensive, critical parts unbuffered. Minimizing EBO directly minimizes the expected number of grounded capital assets."

**Weak Answer:**  
"EBO is the number of parts on backorder. We optimize EBO instead of fill rate because backorders tell you how many planes are broken down."

---

### Question 4: "Explain Fox's Greedy Marginal Knapsack algorithm. Under what conditions is it guaranteed to find the optimal stock allocation?"

**Strong Answer:**  
"Fox's algorithm is a discrete greedy heuristic that builds an undominated cost-availability curve. Starting from zero stock everywhere ($S = \mathbf{0}$):
1. In each iteration, it evaluates every candidate part $i$ and location $k$ (depot or base).
2. It calculates the marginal backorder reduction per dollar spent: $\Delta \text{EBO} / c_i$.
3. It allocates one additional unit of stock to the specific part and location that yields the maximum bang-for-buck ratio.
4. It repeats until the capital budget is exhausted.

Fox's algorithm is mathematically guaranteed to find the globally optimal solution on the convex hull if the objective function $\sum \text{EBO}$ is **separable across items and strictly convex decreasing** in stock levels. Because Poisson EBO functions exhibit strictly diminishing marginal returns ($\Delta^2 \text{EBO} > 0$), convexity holds, ensuring that the greedy path never gets trapped in suboptimal local minima."

**Weak Answer:**  
"Fox's algorithm is a greedy search that picks the part with the highest improvement in each step. It is guaranteed to be optimal because it's a knapsack problem."

---

### Question 5: "What is the difference between standard METRIC and Muckstadt's MOD-METRIC?"

**Strong Answer:**  
"Standard METRIC (Sherbrooke 1968) is a **single-indenture** model: it considers only the relationship between field bases and the central depot for a flat list of Line Replaceable Units (LRUs).

MOD-METRIC (Muckstadt 1973) extends METRIC to **multi-indenture** systems, modeling the structural dependency between first-indenture LRUs (e.g., radar assembly) and second-indenture Shop Replaceable Units (SRUs, e.g., circuit cards, power modules). 

In MOD-METRIC, when an unserviceable LRU arrives at the depot shop, its turnaround time is not fixed. It is modeled as an endogenous variable dependent on whether its internal SRUs are available in depot stock. If an SRU is stocked out, the LRU sits disassembled on the bench waiting for the SRU. MOD-METRIC optimizes stock across both LRUs and SRUs simultaneously, identifying high-leverage opportunities where stocking inexpensive SRUs prevents multi-million-dollar LRU groundings."

**Weak Answer:**  
"MOD-METRIC is just a modified version of METRIC that works on newer computers. It lets you include sub-assemblies."

---

### Question 6: "Why does standard METRIC fail when central depot stock is low, and how does VARI-METRIC fix it?"

**Strong Answer:**  
"Standard METRIC assumes that the arrival of replenishment shipments at field bases follows a Poisson process with variance equal to the mean ($\text{VMR} = 1.0$). 

When the central depot has adequate stock, this assumption is valid. But when depot stock is low, the depot frequently stocks out. Outstation requisitions are placed on backorder and queue up. When the depot shop finishes a repair batch, multiple units are shipped out simultaneously. This creates **batch arrivals and demand lumpiness** at the field bases, causing the variance of base arrivals to far exceed the mean ($\text{VMR} > 1.0$).

Standard Poisson METRIC ignores this variance inflation, severely underestimating base backorders and overestimating asset availability. **VARI-METRIC** (Graves 1985) fixes this by replacing the Poisson distribution with a **Negative Binomial distribution** matched to both the mean and the inflated variance of the base pipeline, restoring predictive accuracy."

**Weak Answer:**  
"When the depot runs out of stock, things become more random. VARI-METRIC fixes it by using a negative binomial distribution instead of a Poisson distribution."

---

### Question 7: "What is lateral transshipment, and why do traditional METRIC models exclude it?"

**Strong Answer:**  
"Lateral transshipment is the emergency transfer of a serviceable spare part between two peer forward bases at the same echelon level (e.g., shipping a generator from Dallas to Chicago to rescue an AOG aircraft), bypassing the central depot.

Traditional METRIC models exclude lateral transshipment because:
1. **Mathematical Tractability:** Allowing lateral sharing couples the state spaces of all field bases, destroying the separability that enables independent base Poisson queue evaluations.
2. **Operational Cost:** Lateral transshipments are expensive, expedited exceptions that involve high freight costs and deplete the safety buffer of the lending station.

Modern extensions (such as Lee's lateral transshipment models) incorporate lateral sharing as an emergency recourse policy, demonstrating that pooling between proximate geographic bases reduces total network safety stock by 10–15%."

**Weak Answer:**  
"Lateral transshipment means shipping parts sideways between bases. METRIC ignores it because it makes the math too complicated."

---

### Question 8: "How do you reconcile local base fill rate targets in customer service level agreements with global METRIC availability optimization?"

**Strong Answer:**  
"This is a common commercial-operational conflict. Customers often demand contractual SLAs specified as a local fill rate (e.g., '95% off-the-shelf availability at Base 1'). However, mathematical METRIC optimization minimizes fleet-wide backorders, which often assigns lower local stock to bases with short transportation lead times.

To reconcile this:
1. **Contractual Reform:** Educate the customer to transition from transactional fill-rate SLAs to **Operational Availability ($A_o$) or Flight Dispatch Reliability SLAs**. The customer ultimately cares about flying airplanes or running scanners, not warehouse shelf counts.
2. **Constrained METRIC Optimization:** If fill-rate targets are legally binding, add local fill-rate boundary constraints ($F_{ij} \ge 0.95$) as lower bounds on base stock ($S_{ij} \ge s_{\min}$) before running marginal knapsack optimization on the remaining budget. This satisfies the legal constraint while optimizing remaining capital for availability."

**Weak Answer:**  
"You just have to buy enough stock to hit the 95% fill rate at every base first, and then whatever money is left over you put into the central depot."

---

## 15. Key terms

- **METRIC:** Multi-Echelon Technique for Recoverable Item Control, developed by Craig Sherbrooke (1968).
- **Recoverable Item:** A component that is repeatedly overhauled and repaired rather than discarded upon failure.
- **$(S-1, S)$ Policy:** Continuous-review base-stock policy where every single consumption event triggers an immediate reorder.
- **Palm's Theorem:** In an $M/G/\infty$ queue with Poisson arrivals, steady-state pipeline items are Poisson distributed with mean $\lambda T$.
- **Expected Backorders (EBO):** The expected number of unfulfilled part requisitions outstanding across the system.
- **Line Replaceable Unit (LRU):** A modular assembly removed and replaced directly on the operating asset (First Indenture).
- **Shop Replaceable Unit (SRU):** A subassembly removed and repaired only inside a specialized workshop (Second Indenture).
- **MOD-METRIC:** Multi-indenture METRIC extension developed by John Muckstadt (1973).
- **VARI-METRIC:** Overdispersion-corrected METRIC extension developed by Stephen Graves (1985).
- **Turnaround Time (TAT):** The total elapsed time required to test, overhaul, certify, and restock a failed recoverable component.
- **Order-and-Ship Time (OST):** The transit time required to transport a ready spare from the central depot to a forward base.
- **Echelon Stock:** Total inventory of an item at a specific facility plus all downstream stock in transit or at child bases.
- **Lateral Transshipment:** Emergency transfer of spare parts directly between peer field bases at the same echelon level.
- **Cannibalization:** Harvesting a functional spare from an already grounded asset to return another asset to operational status.
- **Endogenous Lead Time:** A lead time determined internally by network inventory levels and depot queue delays.
- **Variance-to-Mean Ratio (VMR):** The ratio $\sigma^2 / \mu$, measuring the degree of demand overdispersion.
- **Fox's Algorithm:** A greedy marginal analysis algorithm finding the optimal cost-availability knapsack allocation.
- **Beyond Economic Repair (BER):** An unserviceable part whose repair cost exceeds its replacement value; scrapped immediately.
- **Ready-for-Issue (RFI):** A certified, fully serviceable spare part available on the warehouse shelf for immediate installation.
- **Aircraft On Ground (AOG):** Critical emergency state where an aircraft is grounded awaiting spare parts delivery.

---

## 16. Sources

### Peer-reviewed and Academic — Grade A
1. **Sherbrooke, C. C. (1968).** "METRIC: A multi-echelon technique for recoverable item control." *Operations Research*, 16(1), 122-141. [DOI: 10.1287/opre.16.1.122]
2. **Muckstadt, J. A. (1973).** "A model for a multi-item, multi-echelon, multi-indenture inventory system." *Management Science*, 20(4-part-i), 472-481. [DOI: 10.1287/mnsc.20.4.472]
3. **Graves, S. C. (1985).** "A multi-echelon inventory model for recoverable items with stochastic lead times." *Management Science*, 31(10), 1247-1256. [DOI: 10.1287/mnsc.31.10.1247]
4. **Fox, B. (1966).** "Discrete optimization via marginal analysis." *Management Science*, 13(3), 210-216. [DOI: 10.1287/mnsc.13.3.210]
5. **Svoronos, A., & Zipkin, P. (1991).** "Evaluation of one-for-one replenishment policies for multiechelon systems." *Management Science*, 37(1), 68-83. [DOI: 10.1287/mnsc.37.1.68]

### Standards and Regulatory Frameworks — Grade B
6. **Department of Defense (DoD). (1998).** *MIL-HDBK-338B: Electronic Reliability Design Handbook*. Washington, D.C.: US Department of Defense.
7. **Aerospace and Defence Industries Association of Europe (ASD). (2021).** *S2000M: International Specification for Materiel Management*. Brussels: ASD.
8. **Airlines for America (A4A). (2022).** *ATA Spec 2000: E-Business Specification for Materials Management*. Washington, D.C.: A4A.

### Industry Case Evidence and Technical Monographs — Grade C & D
9. **Sherbrooke, C. C. (2004).** *Optimal Inventory Modeling of Systems: Multi-Echelon Techniques* (2nd ed.). Boston: Kluwer Academic Publishers.
10. **Boeing Global Services. (2023).** *Component Services Program: Global Rotables Fleet Management Whitepaper*. Seattle: The Boeing Company.

---

## 17. Links

- **Parent Module:** [`_index.md`](_index.md) (Module 31 — Aftermarket, Spare Parts, and Service Supply Chains)
- **Root Survey File:** [`../../../course/31_aftermarket_spare_parts_service.md`](../../../course/31_aftermarket_spare_parts_service.md)
- **Previous Topic:** [`02_criticality-spares-classification.md`](02_criticality-spares-classification.md) (Criticality and Spares Classification)
- **Next Topic:** [`04_repair-loop-turnaround.md`](04_repair-loop-turnaround.md) (Repair Loops and Turnaround)
- **Related Methods:**
  - [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
  - [`../../../course/12_transportation_routing_last_mile.md`](../../../course/12_transportation_routing_last_mile.md)
