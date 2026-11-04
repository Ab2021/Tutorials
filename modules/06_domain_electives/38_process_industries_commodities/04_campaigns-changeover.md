# Campaigns and Changeover

> Module 38 — Process Industries, Commodities, and Bulk Supply Chains
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 15 (Production Planning and Scheduling), Module 16 (Inventory Theory), Topic 01 (Continuous, Batch, and Discrete Production)
> Version 2026.5

---

## 1. Why this topic exists

In discrete manufacturing, changeovers are typically modeled as sequence-independent or mildly sequence-dependent tool swaps. The Single-Minute Exchange of Die (SMED) methodology treats setups as mechanical tasks to be externalized, compressed, or eliminated. In an automotive stamping plant or an electronics surface-mount assembly line, a machine stops, an operator replaces a fixture or component reel in eight minutes, and the line resumes production. Work-in-process (WIP) sitting on an adjacent transfer conveyor remains chemically inert, structurally stable, and fully conforming to quality specifications.

Process industries operate under fundamentally different physical, thermodynamic, and fluid dynamic constraints. In a continuous polyethylene polymerization plant, a dairy spray drying tower, a continuous pharmaceutical fluid-bed granulator, an oil refinery fluid catalytic cracking unit, or an industrial paint mixing tank:

1. **Changeovers are physically destructive, thermodynamically prolonged, and material-intensive:** Switching from one product grade to another does not simply mean changing a software recipe or tool head. It requires flushing thousands of liters of piping, purging extruders, draining residual viscous fluids from vessel bottoms (heels), and decontaminating heat exchangers. In thermal processes like glass furnaces or cement kilns, changing chemistry alters operating temperatures and flame dynamics; thermal re-equilibration takes hours or days, during which refractories undergo thermal expansion stresses.
2. **Transition scrap represents a massive downgrade loss:** When a polyolefin extrusion train transitions from a low-melt-index blow-molding grade (MFI = 0.3 g/10 min) to a high-melt-index injection-molding grade (MFI = 20.0 g/10 min), the continuous reaction cannot be halted. For several hours, the extruder outputs intermediate, "wide-spec" transition polymer. This off-spec material cannot be sold as prime virgin polymer ($1,500/MT); it must be downgraded to generic regrind or wax blending material ($600/MT) or incinerated for fuel value. A single transition can generate 50 to 200 metric tons of transition scrap, costing $50,000 to $180,000 in immediate downgrade penalties.
3. **Contamination pathways are irreversible (asymmetric changeovers):** In food, dairy, specialty chemicals, and pharmaceuticals, cross-contamination risks are strictly asymmetric. Moving from a clear beverage to a dark berry beverage requires only a 15-minute warm water flush. Moving from a dark berry beverage back to a clear beverage requires an aggressive 4-hour Clean-in-Place (CIP) caustic wash, acid rinse, and spectroscopic validation. Moving from an allergen-free nutritional powder to a peanut- or soy-containing formulation requires minimal cleaning; moving from peanut-containing powder back to allergen-free powder requires a complete mechanical disassembly, wet-wash sanitization, ATP bioluminescence swab testing, and microbiological validation lasting 24 to 48 hours.
4. **The "Long Campaign" operational pathology:** Confronted with massive setup downtime and painful transition scrap penalties, plant managers instinctively schedule gigantic production campaigns ("run Grade A continuously for eight weeks"). This eliminates changeover penalties on the plant floor. However, it creates severe downstream supply chain crises:
   - **Bulk Storage Hydraulic Lock-up:** Long runs produce hundreds or thousands of tons of a single grade, completely filling dedicated finished-goods silos or tank farms. When storage reaches 100% capacity, the continuous plant must undergo an emergency shut-down or flare feedstock.
   - **Campaign Starvation:** While the asset is monopolized by Grade A for two months, customer inventories for Grades B, C, and D are exhausted. The business suffers severe stockouts on high-margin specialty grades, losing key customer contracts.
   - **Trapped Working Capital:** Millions of dollars of bulk intermediate inventory sit idle in silos, incurring capital carrying charges, tank leasing fees, and degradation risks (e.g., caking, oxidation, moisture absorption).

Optimizing campaigns and changeovers is therefore not an isolated operational scheduling problem. It is the core mathematical nexus where physical equipment mechanics, chemical fluid dynamics, downstream storage capacities, and commercial working capital trade-offs intersect.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Formulate** sequence-dependent changeover scheduling as an Asymmetric Traveling Salesperson Problem (ATSP) with Miller-Tucker-Zemlin (MTZ) and Desrochers-Laporte subtour elimination constraints.
2. **Quantify** total changeover cost matrices by calculating lost contribution margins, thermal and chemical CIP utility costs, and off-spec transition scrap downgrade penalties.
3. **Solve** campaign length optimization using an extended Economic Lot Scheduling Problem (ELSP) that explicitly incorporates continuous production/consumption rates, transition scrap mass balance, and downstream bulk storage capacity bounds.
4. **Design** and size a multi-grade Product Wheel, categorizing high-runner (A), medium-runner (B), and low-runner (C) products into fixed, integer-multiple cyclic production rhythms ($k_i \in \{1, 2, 4\}$).
5. **Model** Clean-in-Place (CIP) turnaround thermodynamics and fluid mechanics using Sinner’s Circle (TACT parameters: Time, Action/Shear, Chemical Concentration, Temperature) to optimize wash regimes between grades.
6. **Diagnose** and resolve triangle inequality violations in transition cost matrices arising from irreversible physical, microbiological, or chemical contamination rules.
7. **Implement** an executable, bounded Python algorithm to optimize sequence-dependent campaign schedules and evaluate storage-induced economic penalties.
8. **Audit** enterprise planning architectures against ISA-88 and ISA-95 standards to establish operational guardrails that prevent schedule breakdown, silo hydraulic lock-up, and campaign starvation.

---

## 3. The core mental model

Campaign and changeover planning balances two orthogonal structural forces: **Transition Topology** (the sequence dimension) and **Run-Length Economics** (the lot-sizing dimension), bounded by equipment capacity and physical storage limits.

```text
                        THE CAMPAIGN DYNAMICS SYSTEM
                        
         SEQUENCE TOPOLOGY (ATSP)            LOT-SIZING ECONOMICS (ELSP)
      +-----------------------------+     +-------------------------------+
      | Grade A (Clear / Low-MFI)   |     | Changeover & Scrap Cost/yr    |
      |   |                         |     |   C_setup(T) = sum(C_ij) / T   |
      |   v (Low penalty: flush)    |     |   (Decreases hyperbolically)  |
      | Grade B (White / Med-MFI)   |     |                               |
      |   |                         |     | Inventory Holding Cost/yr     |
      |   v (Low penalty: purge)    |     |   C_hold(T) = 0.5 * H_sum * T |
      | Grade C (Black / High-MFI)  |     |   (Increases linearly)        |
      |   |                         |     +-------------------------------+
      |   +-------------------------+                     |
      |   Aggressive CIP / Boil-out                       v
      +-----------------------------+     +-------------------------------+
                                          | Optimal Cycle Time: T*        |
                                          | Balance Point: C_setup=C_hold |
                                          +-------------------------------+
                                                          |
                                                          v
                                          +-------------------------------+
                                          | OPERATIONAL FEASIBILITY BOUNDS|
                                          |                               |
                                          | 1. Machine Capacity Floor:    |
                                          |    T >= T_min (Downtime cap)  |
                                          |                               |
                                          | 2. Silo Storage Ceiling:      |
                                          |    T <= T_max (Tank capacity) |
                                          +-------------------------------+
```

### The Product Wheel Architecture

The operational bridge between sequence optimization and campaign length is the **Product Wheel**. A Product Wheel is a fully repetitive, visual scheduling cadence designed to produce every required grade in a sequence that minimizes changeover costs, while cycling frequently enough to maintain customer service levels and respect storage limits.

```text
                     THE 4-GRADE PRODUCT WHEEL CADENCE
                     
                              12:00 (Start Cycle)
                                 Grade 1 (Natural)
                              +--------------------+
                             /                      \
                            /   Run Time: t_1        \
                           /    Inventory: Peak       \
       Changeover 4->1    /                            \  Changeover 1->2
       Full Strip & CIP  /                              \ Minor Purge
       (16h, 80 MT scrap)                                \ (1.5h, 8 MT scrap)
                        |                                |
       Grade 4 (Black)  |          CYCLE TIME            |  Grade 2 (White)
       Run Time: t_4    |               T                |  Run Time: t_2
                        |          (e.g., 8 Weeks)       |
                         \                              /
       Changeover 3->4    \                            /  Changeover 2->3
       Black Color Wash    \                          /   Color Flush
       (2.0h, 10 MT scrap)  \   Run Time: t_3        /    (2.5h, 12 MT scrap)
                             \                      /
                              +--------------------+
                                Grade 3 (Yellow)
                                      06:00
```

### Inventory Sawtooth with Off-Spec Transition Phase

In discrete manufacturing, an inventory sawtooth profile transitions instantaneously from zero to production slope $(P - D)$. In process industries, every campaign transition contains an intermediate dead-band or transition zone:

```text
Inventory
Level
  ^
  |        Peak Inventory I_max = D * T * (1 - D/P)
  |                  /\
  |                 /  \                          /-- Tank Physical Capacity
  |                /    \                        /
  |               /      \                      /
  |              /        \                    /
  |  Prime Run  /          \  Depletion Rate  /
  |  Slope P-D /            \  Slope: -D     /
  |           /              \              /
  |          /                \            /
  |         /                  \          /
  |        /                    \        /
  |       /                      \      /
  |  +---+                        \    /
  |  |   |                         \  /
  +--+---+--------------------------\/--------------------> Time
     <---> <----------------------->
    Setup/       Prime Run Phase        Depletion Phase
  Transition       Duration: t_i         Duration: T - t_i
     tau_ij
```

During the setup window $\tau_{ij}$, the asset is off-line undergoing CIP, purging, or thermal stabilization, while downstream demand draws down inventory at rate $D_i$. Furthermore, during the transition phase, off-spec scrap volume $V_{ij}^{trans}$ is rejected or diverted to downgrade bins before prime product begins filling the product silo.

---

## 4. Definitions and vocabulary

| Term | Definition | Common Misuse or Pitfall | Standard Units |
|---|---|---|---|
| **Campaign** | A continuous, uninterrupted production run of a single product grade or family on a dedicated or multi-purpose asset before a changeover occurs. | Confused with a production batch; a campaign often comprises dozens of identical batches or days of continuous flow. | Days, Hours, or Metric Tons (MT) |
| **Changeover Time ($\tau_{ij}$)** | The total non-productive elapsed duration required to transition an asset from completing grade $i$ to yielding prime-spec grade $j$. | Omitting thermal stabilization or quality validation time; counting only physical cleaning time. | Hours (h) |
| **Sequence-Dependent Setup Cost ($S_{ij}$)** | The direct cash and utility expense incurred by transitioning from grade $i$ to grade $j$, independent of lost production volume. | Conflating lost opportunity cost with direct utility costs (water, caustic, steam, purge resin). | Currency ($ or €) |
| **Asymmetric Transition Matrix ($C$)** | A square matrix of dimensions $N \times N$ where $C_{ij} \ne C_{ji}$, reflecting irreversible physical/chemical contamination paths. | Assuming symmetric costs ($C_{ij} = C_{ji}$); running reverse sequences that trigger 10x cleaning penalties. | Currency ($) or Hours (h) |
| **Transition Scrap ($V_{ij}^{trans}$)** | The mass of off-spec or intermediate-grade material produced while physical/chemical parameters shift from grade $i$ to $j$. | Treating scrap as zero-value waste; it usually has salvage/regrind value ($p_{downgrade} > 0$). | Metric Tons (MT) or kg |
| **Downgrade Penalty ($\Delta p_{ij}$)** | The economic value loss per unit mass of transition material: $\Delta p = p_{prime} - p_{downgrade}$. | Calculating penalty as prime sales price rather than net realizable loss against feed cost. | $/MT or $/kg |
| **Economic Lot Scheduling Problem (ELSP)** | The mathematical problem of scheduling $N$ products with known demand rates on a single shared facility on a repetitive cyclic basis. | Applying standard single-product EOQ formulas independently, causing overlapping production conflicts. | Dimensionless / Optimization Model |
| **Product Wheel** | A structured, repetitive visual schedule that cycles through a set of products in an optimized sequence at a fixed or semi-fixed frequency. | Treating the wheel as rigid; failing to leave dynamic insertion slots or buffer hours for demand surges. | Weeks or Days per cycle |
| **Common Cycle Time ($T$)** | The duration required for a product wheel to complete one full rotation through all scheduled grades. | Believing all products must run in every cycle; low-runners can run every $2T$ or $4T$. | Days or Weeks |
| **Basic Period ($T_B$)** | The fundamental time increment in an extended ELSP formulation, where product cycle times are integer multiples: $T_i = k_i T_B$. | Picking a $T_B$ so small that setup downtime consumes all capacity, or so large that inventory explodes. | Days or Weeks |
| **Clean-in-Place (CIP)** | Automated cleaning of interior surfaces of pipes, vessels, process equipment, and fittings without disassembling the plant. | Assuming CIP is fixed-duration; CIP duration varies widely based on soil load and chemical recipes. | Minutes or Hours |
| **Sinner's Circle (TACT)** | The thermodynamic/physical cleaning principle balancing Time, Action (shear), Chemistry (concentration), and Temperature. | Compensating for poor fluid shear (Action) purely by extending wash time, wasting thermal energy. | Dimensionless balance |
| **Clean-out-of-Place (COP)** | Manual or semi-automated cleaning requiring the physical disassembly and removal of parts to a dedicated washing station. | Underestimating labor costs and equipment re-assembly misalignment during changeovers. | Hours |
| **Triangle Inequality Violation** | A condition where transitioning directly from $i$ to $k$ costs more than transitioning through an intermediate grade $j$ ($C_{ik} > C_{ij} + C_{jk}$). | Forcing direct transitions in MIP models when an intentional intermediate sacrificial run is cheaper. | Currency ($) or Hours |
| **Heel / Dead Volume** | Residual material remaining at the bottom of a storage tank, reactor, or sump that cannot be removed by normal gravity discharge. | Ignoring heel dilution; fresh grade pumped into an unstripped tank gets contaminated by the heel. | $m^3$, Liters, or MT |
| **Campaign Starvation** | The operational failure where downstream customer orders or packaging lines run out of a grade while the upstream asset is locked in a long campaign. | Blaming the commercial forecast when the root cause was an oversized campaign length. | Stockout incidents or Lost Sales ($) |
| **Silo Hydraulic Lock-up** | The forced shutdown of an upstream continuous production asset because downstream finished-goods storage silos are 100% full. | Blaming plant reliability when the root cause was scheduling a campaign longer than silo capacity. | Unplanned downtime hours |
| **Hydrodynamic Purge** | The displacement of a preceding fluid or polymer melt by pumping the subsequent fluid through the system under laminar or turbulent flow. | Assuming plug-flow displacement; parabolic velocity profiles cause long transition tails. | Volumes or Residence Times |
| **Residence Time Distribution (RTD)** | The probability distribution describing the amount of time a fluid element spends inside a continuous reactor or extruder. | Assuming narrow RTD; wide RTD creates persistent contamination tails requiring extended purges. | Probability Density Function |
| **Allergen Washout Matrix** | A binary or scalar matrix defining validated cleaning requirements when transitioning between specific allergen profiles. | Treating all allergens identically; some proteins (e.g., gluten) denature easily, others (peanut) persist. | Cleaning Category / Hours |
| **Melt Flow Index (MFI)** | A measure of the ease of flow of the melt of a thermoplastic polymer, inversely related to molecular weight. | Scheduling high-MFI before low-MFI; flushing viscous resin with thin resin takes 5x longer. | $g/10\text{ min}$ |

---

## 5. The mechanics, step by step

Optimizing campaign schedules, changeover paths, and product wheels requires a rigorous, multi-stage engineering procedure:

```text
  Step 1: State Space & Cost Matrix Construction
          [Calculate C_ij = Lost Margin + Scrap Downgrade + Direct CIP]
                             |
                             v
  Step 2: Sequence Optimization via ATSP Formulations
          [Solve ATSP with MTZ / Desrochers-Laporte Subtour Elimination]
                             |
                             v
  Step 3: Campaign Length Optimization (Extended ELSP)
          [Calculate T*, Machine Capacity Floor, and Silo Capacity Ceiling]
                             |
                             v
  Step 4: Product Wheel Sizing & Multi-Period ABC Cadence
          [Assign Integer Multipliers k_i in {1, 2, 4} & Natural Progressions]
                             |
                             v
  Step 5: Clean-in-Place (CIP) Optimization via Sinner's Circle
          [Balance TACT: Time, Mechanical Shear, Chemistry, Temperature]
```

### 5.1 Step 1: Transition State Space Characterization & Tri-Partite Cost Matrix Construction

Every grade transition $(i \to j)$ incurs costs across three distinct operational mechanisms:
1. **Downtime Opportunity Cost ($C_{ij}^{downtime}$):** The lost contribution margin from halting or slowing the production line during cleaning, thermal re-equilibration, or mechanical re-tooling.
2. **Material Scrap Downgrade Penalty ($C_{ij}^{scrap}$):** The physical mass of off-spec or contaminated product produced during fluid displacement, multiplied by the net price differential between prime product and downgrade/regrind value.
3. **Direct Utility & Cleaning Expense ($C_{ij}^{utility}$):** The direct out-of-pocket costs for cleaning chemicals (caustic, acid), thermal steam, demineralized water, purge resins, mechanical filters, and effluent wastewater treatment surcharges.

The total sequence-dependent changeover cost $C_{ij}$ is formulated as:

$$C_{ij} = \tau_{ij} \cdot \left(P \cdot m\right) + V_{ij}^{trans} \cdot \left(p_i^{prime} - p_{ij}^{down}\right) + S_{ij}^{utility}$$

Where:
- $\tau_{ij}$ is the total downtime (hours).
- $P$ is the nominal operating production rate (metric tons per hour, MT/h).
- $m$ is the marginal contribution margin of the asset ($\$/MT$), defined as revenue minus variable raw material and utility costs ($p - v$). If the asset is capacity-constrained and sold out, $P \cdot m$ represents the true opportunity cost of lost capacity. If the asset is market-constrained (unutilized capacity exists), $P \cdot m$ is replaced by the marginal hourly carrying cost of unabsorbed fixed overhead.
- $V_{ij}^{trans}$ is the transition scrap volume (MT).
- $p_i^{prime}$ is the market value of prime grade $i$ ($\$/MT$).
- $p_{ij}^{down}$ is the salvage value of the transition scrap ($\$/MT$).
- $S_{ij}^{utility}$ is the direct utility and chemical cost ($\$).

#### Hydrodynamic Purge and Transition Volume Mechanics
In continuous flow systems and polymer extrusion, fluid adheres to pipe and vessel walls due to the no-slip boundary condition ($v(R) = 0$). In laminar flow, the velocity profile is parabolic:

$$v(r) = 2 v_{avg} \left(1 - \frac{r^2}{R^2}\right)$$

Fluid in the center moves at twice the average velocity, while fluid near the wall moves infinitesimally slowly. Consequently, displacing Grade $i$ with Grade $j$ results in a prolonged "transition tail." The concentration $C(t)$ of residual Grade $i$ exiting an equipment volume $V_{eq}$ under continuous stirred-tank dynamics follows:

$$C(t) = C_0 \exp\left(-\frac{t}{\theta}\right) = C_0 \exp\left(-\frac{\dot{V} t}{V_{eq}}\right)$$

Where $\theta = V_{eq} / \dot{V}$ is the mean residence time. To achieve a specification threshold of $99.9\%$ purity ($C(t)/C_0 \le 0.001$), the system must be purged for:

$$t_{purge} = -\theta \ln(0.001) \approx 6.91 \cdot \theta$$

Thus, clearing a vessel requires approximately 7 vessel volumes of material throughput. In high-viscosity non-Newtonian polymer melts, purge resins containing abrasive scrubbing agents (e.g., calcium carbonate or glass fiber concentrates) are used to mechanically scour the barrel walls, trading direct chemical costs for reduced purge volumes.

---

### 5.2 Step 2: The Asymmetric Traveling Salesperson Problem (ATSP) Formulation

When scheduling $N$ distinct products in a cyclic campaign, determining the optimal production sequence that minimizes total changeover cost (or total setup downtime) is mathematically equivalent to the **Asymmetric Traveling Salesperson Problem (ATSP)** defined on a complete directed graph $G = (V, A)$.

Let $V = \{1, 2, \dots, N\}$ represent the set of product grades. Let $x_{ij} \in \{0, 1\}$ be a binary decision variable equal to 1 if grade $j$ immediately follows grade $i$ in the cyclic schedule, and 0 otherwise.

#### Mathematical Formulation

$$\min \sum_{i=1}^N \sum_{j=1, j \ne i}^N C_{ij} x_{ij}$$

Subject to:

$$\sum_{j=1, j \ne i}^N x_{ij} = 1 \quad \forall i \in V \quad \text{(Each grade has exactly one successor)}$$

$$\sum_{i=1, i \ne j}^N x_{ij} = 1 \quad \forall j \in V \quad \text{(Each grade has exactly one predecessor)}$$

#### Subtour Elimination Constraints
The assignment constraints above ensure every node has an in-degree of 1 and an out-degree of 1. However, they allow disconnected sub-cycles (e.g., $1 \to 2 \to 1$ and $3 \to 4 \to 3$). Subtours must be explicitly eliminated.

**1. Miller-Tucker-Zemlin (MTZ) Formulation:**
Introduce continuous auxiliary variables $u_i \in \mathbb{R}$ representing the sequencing position of grade $i$:

$$u_i - u_j + N x_{ij} \le N - 1 \quad \forall i, j \in \{2, \dots, N\}, i \ne j$$

$$1 \le u_i \le N - 1 \quad \forall i \in \{2, \dots, N\}$$

If $x_{ij} = 1$, the constraint forces $u_j \ge u_i + 1$, preventing any closed loop that does not include the base node 1. While compact ($O(N^2)$ constraints), the MTZ linear programming relaxation is notoriously weak.

**2. Desrochers-Laporte Lifted MTZ Formulation:**
Desrochers and Laporte (1991) strengthened MTZ by incorporating reverse arc variables and lifting the coefficients:

$$u_i - u_j + (N - 1) x_{ij} + (N - 3) x_{ji} \le N - 2 \quad \forall i, j \in \{2, \dots, N\}, i \ne j$$

This significantly tightens the polytope and accelerates branch-and-cut solve times by an order of magnitude for $N \le 50$.

#### Physical Causes of Triangle Inequality Violations
In standard Euclidean routing, the triangle inequality always holds: $C_{ik} \le C_{ij} + C_{jk}$. In process changeovers, **the triangle inequality is routinely violated**:

$$C_{ik} > C_{ij} + C_{jk}$$

*Industrial Example:* Transitioning directly from a Clear Cosmetic Resin (Grade $i$) to a Flame-Retardant Black Resin (Grade $k$) containing brominated additives causes severe chemical pitting if executed directly, requiring a 24-hour acid decontamination ($C_{ik} = \$80,000$). 
However, if the plant schedules an intermediate Commodity White Resin (Grade $j$) between them, the transition from Clear to White takes 2 hours ($C_{ij} = \$6,000$), and White to Black takes 2 hours ($C_{jk} = \$8,000$). The sum $C_{ij} + C_{jk} = \$14,000$, saving $\$66,000$.

When the triangle inequality is violated, standard TSP formulations that forbid node repetition will produce sub-optimal schedules. Planners must either:
1. Pre-process the cost matrix using the Floyd-Warshall all-pairs shortest path algorithm: $C_{ik} \leftarrow \min(C_{ik}, C_{ij} + C_{jk})$.
2. Allow "dummy" or sacrificial wash campaigns in the production schedule.

---

### 5.3 Step 3: Extended Economic Lot Scheduling Problem (ELSP) with Transition Waste & Storage

Once the optimal sequence $\pi = (\pi(1), \pi(2), \dots, \pi(N))$ is established via ATSP, the planner must determine the **optimal cycle time ($T$)** and individual **campaign run lengths ($Q_i$)**.

The classical Economic Lot Scheduling Problem (ELSP) balances inventory holding costs against setup costs. In process industries, the classical formulation must be extended to incorporate:
- Non-zero sequence-dependent transition scrap $V_{ij}^{trans}$.
- Asset capacity feasibility floors.
- Finite downstream bulk silo/tank storage ceilings.

#### Formulation of the Extended ELSP Objective Function
Let:
- $D_i$ = Continuous demand rate for grade $i$ (MT/year).
- $P_i$ = Production rate for grade $i$ (MT/year), where $P_i > D_i$.
- $\rho_i = D_i / P_i$ = Capacity utilization of grade $i$.
- $h_i$ = Inventory holding cost rate ($\$/MT/\text{year}$).
- $C_{setup}^{cycle} = \sum_{i=1}^N C_{i, \pi(i)}$ = Total sequence-dependent changeover and scrap cost per cycle ($\$).
- $\tau^{cycle} = \sum_{i=1}^N \tau_{i, \pi(i)}$ = Total downtime per cycle (hours).
- $T$ = Common cycle time of the product wheel (years).

Under a common cycle time $T$, the production batch size for grade $i$ is:

$$Q_i = D_i T$$

The time spent producing prime grade $i$ during each cycle is:

$$t_i = \frac{Q_i}{P_i} = \frac{D_i T}{P_i} = \rho_i T$$

The inventory of grade $i$ follows a sawtooth profile. Inventory accumulates during production at rate $(P_i - D_i)$ for duration $t_i$. The maximum inventory level $I_i^{max}$ attained at the end of the production run is:

$$I_i^{max} = (P_i - D_i) t_i = (P_i - D_i) \left(\frac{D_i T}{P_i}\right) = D_i T \left(1 - \frac{D_i}{P_i}\right) = D_i T (1 - \rho_i)$$

The average inventory held throughout the year is $\frac{1}{2} I_i^{max}$. Therefore, the total annual holding cost across all $N$ products is:

$$\text{Holding Cost}(T) = \sum_{i=1}^N \frac{1}{2} h_i D_i (1 - \rho_i) T = \frac{1}{2} H_{sum} T$$

Where $H_{sum} = \sum_{i=1}^N h_i D_i (1 - \rho_i)$.

The number of cycles run per year is $1/T$. The annual setup and transition scrap cost is:

$$\text{Setup Cost}(T) = \frac{C_{setup}^{cycle}}{T}$$

The Total Annual Cost $TC(T)$ is the sum of setup, scrap, and holding costs:

$$TC(T) = \frac{C_{setup}^{cycle}}{T} + \frac{1}{2} H_{sum} T$$

Setting the first derivative with respect to $T$ to zero:

$$\frac{d TC(T)}{dT} = -\frac{C_{setup}^{cycle}}{T^2} + \frac{1}{2} H_{sum} = 0$$

Yields the unconstrained optimal cycle time $T^*$:

$$T^* = \sqrt{\frac{2 C_{setup}^{cycle}}{H_{sum}}} = \sqrt{\frac{2 \sum_{i=1}^N C_{i, \pi(i)}}{\sum_{i=1}^N h_i D_i \left(1 - \frac{D_i}{P_i}\right)}}$$

#### Operational Boundary Constraints

In reality, $T^*$ cannot be arbitrarily selected; it is strictly bounded by equipment availability and storage capacity.

**1. Machine Capacity Feasibility Floor ($T_{min}^{capacity}$):**
In every cycle $T$, the asset must run production for all products plus absorb all changeover downtime. Let $H_{year}$ be the total operating hours available per year (e.g., 8,400 h/year). Total time allocated in cycle $T$ cannot exceed $T$:

$$\sum_{i=1}^N t_i + \frac{\tau^{cycle}}{H_{year}} \le T$$

$$\sum_{i=1}^N \rho_i T + \frac{\tau^{cycle}}{H_{year}} \le T \implies T \left(1 - \sum_{i=1}^N \rho_i\right) \ge \frac{\tau^{cycle}}{H_{year}}$$

$$T \ge T_{min}^{capacity} = \frac{\tau^{cycle} / H_{year}}{1 - \sum_{i=1}^N \rho_i}$$

If the total machine utilization $\sum \rho_i \ge 1.0$, the facility is mathematically over-subscribed; no feasible cycle exists. As utilization approaches $100\%$, $T_{min}^{capacity} \to \infty$, forcing massive campaigns to avoid starvation.

**2. Finite Bulk Silo Storage Ceiling ($T_{max}^{silo}$):**
Each product grade $i$ is stored in a dedicated tank or silo of capacity $C_i^{silo}$. Accounting for safety stock ($I_i^{safe}$) and unpumpable heel volume ($I_i^{heel}$), the effective available working capacity is:

$$C_i^{eff} = C_i^{silo} - I_i^{safe} - I_i^{heel}$$

Peak inventory $I_i^{max}$ cannot exceed $C_i^{eff}$:

$$D_i T (1 - \rho_i) \le C_i^{eff} \implies T \le \frac{C_i^{eff}}{D_i (1 - \rho_i)} \quad \forall i \in \{1, \dots, N\}$$

The global storage ceiling is governed by the most restrictive silo:

$$T_{max}^{silo} = \min_{i \in \{1, \dots, N\}} \left[ \frac{C_i^{eff}}{D_i (1 - \rho_i)} \right]$$

**3. Feasibility Window and Economic Shadow Price:**
A mathematically feasible, storage-stable campaign cycle exists if and only if:

$$T_{min}^{capacity} \le T_{max}^{silo}$$

If $T^* \in [T_{min}^{capacity}, T_{max}^{silo}]$, the unconstrained economic optimum is feasible. 
If $T^* > T_{max}^{silo}$, the plant is storage-constrained. The planner must clamp the cycle time to $T_{constrained} = T_{max}^{silo}$. The economic cost penalty imposed by the storage constraint is:

$$\Delta TC = TC(T_{max}^{silo}) - TC(T^*)$$

This delta $\Delta TC$ represents the annual economic shadow price of bulk storage capacity, providing the direct capital justification for installing additional storage silos.

---

### 5.4 Step 4: Multi-Period Product Wheel Architecture & Operational Sizing

In complex process plants producing 20 to 100 SKUs, running every product in every single cycle is economically disastrous. Low-volume specialty grades ("C-items") would have tiny run times ($t_i = 1\text{ hour}$) dwarfed by 8-hour changeovers, destroying line efficiency.

To solve this, the **Multi-Period Product Wheel** employs the **Basic Period ($T_B$)** approach (Dobson, 1987). Products are classified using an ABC volume-margin matrix and assigned integer frequency multipliers $k_i \in \{1, 2, 4, 8\}$:

```text
               MULTI-PERIOD PRODUCT WHEEL CADENCE (T_B = 2 Weeks)
               
  Week:     1      2      3      4      5      6      7      8
  Cycle:  [--- Cycle 1 ---]  [--- Cycle 2 ---]  [--- Cycle 3 ---]  [--- Cycle 4 ---]
  
  High-Runners (A-Items, k_i = 1):
  Grade A1:  [== Run ==]      [== Run ==]      [== Run ==]      [== Run ==]
  Grade A2:  [== Run ==]      [== Run ==]      [== Run ==]      [== Run ==]
  
  Medium-Runners (B-Items, k_i = 2):
  Grade B1:  [==== Run ====]                    [==== Run ====]
  Grade B2:                 [==== Run ====]                    [==== Run ====]
  
  Low-Runners (C-Items, k_i = 4):
  Grade C1:  [======== Run ========]
  Grade C2:                                    [======== Run ========]
```

#### Product Wheel Design Protocol
1. **Pareto Categorization:**
   - **A-Grades (Top 70% volume):** $k_i = 1$. Produced in every basic period $T_B$. Cycle length $T_i = T_B$.
   - **B-Grades (Next 20% volume):** $k_i = 2$. Produced every alternate cycle ($2 T_B$).
   - **C-Grades (Bottom 10% volume):** $k_i = 4$ or $8$. Produced once every 4 or 8 cycles, or manufactured strictly Make-to-Order (MTO) as campaign campaigns.
2. **Spoke Alignment & Natural Progression:**
   Within each period, products are ordered to follow natural physical gradients:
   - *Color:* Natural $\to$ White $\to$ Yellow $\to$ Red $\to$ Blue $\to$ Black.
   - *Viscosity / Molecular Weight:* High Viscosity (low MFI) $\to$ Low Viscosity (high MFI). Purging a high-viscosity fluid with a low-viscosity fluid is hydrodynamically efficient, whereas low-viscosity fluid channels through the center of high-viscosity fluid without displacing the boundary layer.
   - *Allergen Profile:* Allergen-Free $\to$ Dairy $\to$ Soy $\to$ Peanut.
   - *Melt Temperature:* Low Temperature $\to$ High Temperature.
3. **Breathing Wheel Buffer Allocation:**
   Real demand fluctuates. A rigid product wheel will collapse during the first demand spike. Robust wheels incorporate a 10% to 15% **breathing buffer** (unassigned capacity hours) at the end of each rotation. If demand for Grade A1 surges, its run duration expands into the buffer without displacing the start dates of subsequent grades. If demand is soft, the buffer is used for preventative maintenance or idle energy conservation.

---

### 5.5 Step 5: Clean-in-Place (CIP) Mechanics & Sinner's Circle Turnaround Optimization

In food, dairy, consumer packaged goods, and pharmaceuticals, changeover duration and cost are dictated by Clean-in-Place (CIP) turnaround physics. CIP effectiveness is governed by **Sinner’s Circle**, which establishes that cleaning is a thermodynamic-mechanical balance of four interdependent variables:

$$\text{Cleaning Efficacy} = f\left(\text{Time } t, \text{ Mechanical Action } \tau_w, \text{ Chemical Concentration } C_{chem}, \text{ Temperature } T_{wash}\right)$$

```text
                           SINNER'S CIRCLE OF CIP
                           
                    +-----------------------------------+
                    |         TEMPERATURE (T)           |
                    |  Arrhenius Reaction Kinetics:     |
                    |  k = A * exp(-E_a / R*T)          |
                    |  Hydrolyzes fats & caramelizes    |
                    +-----------------+-----------------+
                                      |
       +-----------------------+      |      +-----------------------+
       |     ACTION (tau_w)    |      |      |     CHEMISTRY (C)     |
       | Fluid Shear Stress:   |<-----+----->| Chemical Dissolution: |
       | tau_w = 0.5*f*rho*v^2 |      |      | Caustic (NaOH: 1-2%)  |
       | Turbulent Re >= 10^5  |      |      | Acid (HNO3: 0.5-1.5%) |
       +-----------------------+      |      +-----------------------+
                                      v
                    +-----------------------------------+
                    |             TIME (t)              |
                    |  Downtime Opportunity Cost        |
                    |  t = Total CIP Cycle Duration     |
                    +-----------------------------------+
```

If any one parameter is reduced, the other three must be increased to maintain equal microbiological and chemical cleanliness:
- Decreasing wash temperature from $80^\circ\text{C}$ to $60^\circ\text{C}$ saves thermal steam energy, but requires doubling chemical concentration or increasing wash circulation time by 75%.
- Increasing flow velocity to elevate turbulent wall shear stress ($\tau_w$) allows significant reductions in cleaning chemical concentration and wash duration.

#### Fluid Mechanics of Mechanical Action
To achieve mechanical cleaning of pipe and vessel internal walls, flow must be fully turbulent, breaking the laminar boundary sub-layer. The Reynolds number must exceed $10^5$:

$$Re = \frac{\rho v D}{\mu} \ge 100,000$$

The wall shear stress exerted by the circulating fluid is:

$$\tau_w = \frac{1}{2} f \rho v^2$$

Where $f$ is the Fanning friction factor, calculated via the Colebrook-White equation. Industrial CIP standards mandate a minimum linear fluid velocity:

$$v_{CIP} \ge 1.5 \text{ to } 2.1 \text{ m/s}$$

Flow velocities below $1.5\text{ m/s}$ allow soil particles to settle in low-velocity pockets and dead legs ($L/D > 1.5$), leading to CIP validation failure.

#### Standard 5-Step CIP Regime Dynamics
1. **Pre-Rinse (Water at $15-25^\circ\text{C}$):** Displaces residual bulk product (heels) to recovery or drain. Must use ambient water; hot water (> $45^\circ\text{C}$) denatures proteins and bakes them onto stainless steel surfaces.
2. **Caustic Detergent Wash ($1.0-2.5\%\text{ NaOH at } 70-85^\circ\text{C}$):** Saponifies fats into soluble soaps, peptizes proteins, and disperses organic soils.
3. **Intermediate Rinse (Clean Water):** Flushes residual caustic detergent until pH probes at the return header detect neutral water ($pH \le 7.5$).
4. **Acid Wash ($0.5-1.5\%\text{ } HNO_3 \text{ or } H_3PO_4 \text{ at } 55-65^\circ\text{C}$):** Dissolves inorganic mineral scales (calcium oxalate, "milkstone") and neutralizes residual caustic films.
5. **Final Sanitizing Rinse (Water for Injection or Microbiocidal Rinse at $20^\circ\text{C}$):** Circulates peracetic acid (PAA, 150-250 ppm) or sterile water to achieve a $6-\log$ reduction in viable vegetative pathogens.

#### Turnaround Economics and Caustic Recovery
Single-use CIP systems dump chemical wash solutions directly to the Effluent Treatment Plant (ETP) after one pass, incurring massive chemical replenishment costs and severe ETP neutralization surcharges ($pH < 6$ or $pH > 9$ triggers regulatory fines). 

Modern plants employ **Multi-Tank CIP Skids** with automated conductivity-controlled return routing. Solutions with conductivity $\ge 40\text{ mS/cm}$ are diverted back to a heated Caustic Recovery Tank, reclaiming 75% to 85% of caustic chemistry and thermal energy.

---

## 6. Worked example

An industrial polyethylene compounding and extrusion plant operates a single high-capacity compounding train producing four commercial polymer grades. We will determine the optimal cyclic production sequence, evaluate triangle inequality violations, and compute the optimal campaign cycle time under production capacity and downstream silo storage constraints.

### 1. Plant Operating Parameters
- Production rate: $P = 10.0\text{ MT/h}$.
- Annual operational availability: 350 days/year ($H_{year} = 350 \times 24 = 8,400\text{ h/year}$).
- Nominal annual plant capacity: $P_{annual} = 10.0 \times 8,400 = 84,000\text{ MT/year}$.
- Marginal contribution margin: $m = \$400\text{ / MT}$ of prime resin.
  - Hourly lost opportunity cost: $P \cdot m = 10.0 \times \$400 = \$4,000\text{ / hour}$ of downtime.
- Prime resin market value: $p_{prime} = \$1,500\text{ / MT}$.
- Transition off-spec scrap salvage value: $p_{down} = \$600\text{ / MT}$.
  - Downgrade scrap penalty: $\Delta p = p_{prime} - p_{down} = \$1,500 - \$600 = \$900\text{ / MT}$.

### 2. Product Demand, Holding Costs, and Storage Capacities
The facility produces four distinct grades with dedicated finished product silos:

| Grade Index ($i$) | Grade Description | Annual Demand $D_i$ (MT/yr) | Utilization $\rho_i = D_i / P_{ann}$ | Holding Cost $h_i$ (\$/MT/yr) | Silo Capacity $C_i^{silo}$ (MT) | Effective Silo Cap $C_i^{eff}$ (MT) |
|:---:|---|:---:|:---:|:---:|:---:|:---:|
| **0** | Natural Clear Blow-Molding | 25,000 | 0.2976 | $120 | 3,300 | 3,000 |
| **1** | White Injection-Molding ($TiO_2$) | 20,000 | 0.2381 | $130 | 2,800 | 2,500 |
| **2** | Yellow Pipe-Extrusion | 15,000 | 0.1786 | $140 | 2,250 | 2,000 |
| **3** | Black UV-Stabilized Pipe | 10,000 | 0.1190 | $150 | 1,700 | 1,500 |
| **Total** | | **70,000** | **0.8333** | | | |

*Storage note:* Effective capacity accounts for a 200 MT unpumpable cone heel and 100 MT minimum buffer stock. Total asset utilization is $\sum \rho_i = 70,000 / 84,000 = 0.8333$ (83.33% load; 16.67% unallocated capacity).

### 3. Asymmetric Transition Matrices
Due to color progression and carbon black cross-contamination, the changeover parameters are highly asymmetric:

#### Table A: Changeover Downtime Matrix $\tau_{ij}$ (Hours)
$$\tau = \begin{pmatrix} 0.0 & 1.5 & 3.0 & 4.0 \\ 6.0 & 0.0 & 2.5 & 3.5 \\ 8.0 & 5.0 & 0.0 & 2.0 \\ 16.0 & 12.0 & 9.0 & 0.0 \end{pmatrix}$$

#### Table B: Off-Spec Transition Scrap Volume Matrix $V_{ij}^{trans}$ (Metric Tons)
$$V^{trans} = \begin{pmatrix} 0.0 & 8.0 & 15.0 & 20.0 \\ 30.0 & 0.0 & 12.0 & 18.0 \\ 45.0 & 25.0 & 0.0 & 10.0 \\ 80.0 & 60.0 & 45.0 & 0.0 \end{pmatrix}$$

#### Table C: Direct Cleaning & Purge Chemical Expense Matrix $S_{ij}^{utility}$ (USD)
$$S^{utility} = \begin{pmatrix} \$0 & \$1,200 & \$1,800 & \$2,200 \\ \$3,500 & \$0 & \$1,500 & \$2,000 \\ \$4,800 & \$3,200 & \$0 & \$1,400 \\ \$9,500 & \$7,500 & \$5,500 & \$0 \end{pmatrix}$$

### 4. Step 1: Calculating the Total Changeover Cost Matrix $C_{ij}$
Using the cost formula $C_{ij} = \tau_{ij} \times \$4,000 + V_{ij}^{trans} \times \$900 + S_{ij}^{utility}$:

For transition $0 \to 1$ (Natural $\to$ White):
$$C_{01} = (1.5 \times 4,000) + (8.0 \times 900) + 1,200 = 6,000 + 7,200 + 1,200 = \$14,400$$

For reverse transition $1 \to 0$ (White $\to$ Natural - requiring extensive purging of white $TiO_2$ pigment):
$$C_{10} = (6.0 \times 4,000) + (30.0 \times 900) + 3,500 = 24,000 + 27,000 + 3,500 = \$54,500$$

For reverse transition $3 \to 0$ (Black $\to$ Natural - requiring complete extruder teardown and barrel burn-out):
$$C_{30} = (16.0 \times 4,000) + (80.0 \times 900) + 9,500 = 64,000 + 72,000 + 9,500 = \$145,500$$

The full total changeover cost matrix $C$ (in USD) is:

$$C = \begin{pmatrix} 
\$0 & \$14,400 & \$27,300 & \$36,200 \\ 
\$54,500 & \$0 & \$22,300 & \$32,200 \\ 
\$77,300 & \$45,700 & \$0 & \$18,400 \\ 
\$145,500 & \$109,500 & \$82,000 & \$0 
\end{pmatrix}$$

### 5. Step 2: Sequence Optimization via ATSP
We evaluate the set of cyclic tours traversing all 4 nodes. Since $N=4$, there are $(4-1)! = 6$ possible directed cyclic tours:

1. **Tour A (Natural Progression):** $0 \to 1 \to 2 \to 3 \to 0$
   - Arc $0 \to 1$: $\$14,400$, $\tau = 1.5\text{ h}$, $V = 8\text{ MT}$
   - Arc $1 \to 2$: $\$22,300$, $\tau = 2.5\text{ h}$, $V = 12\text{ MT}$
   - Arc $2 \to 3$: $\$18,400$, $\tau = 2.0\text{ h}$, $V = 10\text{ MT}$
   - Arc $3 \to 0$: $\$145,500$, $\tau = 16.0\text{ h}$, $V = 80\text{ MT}$
   - **Total Cycle Cost:** $C_A = 14,400 + 22,300 + 18,400 + 145,500 = \mathbf{\$200,600}$
   - **Total Cycle Downtime:** $\tau_A = 1.5 + 2.5 + 2.0 + 16.0 = \mathbf{22.0\text{ hours}}$
   - **Total Cycle Scrap:** $V_A = 8 + 12 + 10 + 80 = \mathbf{110.0\text{ MT}}$

2. **Tour B:** $0 \to 1 \to 3 \to 2 \to 0$
   - $C_B = C_{01} + C_{13} + C_{32} + C_{20} = 14,400 + 32,200 + 82,000 + 77,300 = \mathbf{\$205,900}$
   - $\tau_B = 1.5 + 3.5 + 9.0 + 8.0 = 22.0\text{ hours}$

3. **Tour C:** $0 \to 2 \to 1 \to 3 \to 0$
   - $C_C = C_{02} + C_{21} + C_{13} + C_{30} = 27,300 + 45,700 + 32,200 + 145,500 = \mathbf{\$250,700}$

4. **Tour D:** $0 \to 2 \to 3 \to 1 \to 0$
   - $C_D = C_{02} + C_{23} + C_{31} + C_{10} = 27,300 + 18,400 + 109,500 + 54,500 = \mathbf{\$209,700}$

5. **Tour E:** $0 \to 3 \to 1 \to 2 \to 0$
   - $C_E = C_{03} + C_{31} + C_{12} + C_{20} = 36,200 + 109,500 + 22,300 + 77,300 = \mathbf{\$245,300}$

6. **Tour F (Reverse / Worst Sequence):** $0 \to 3 \to 2 \to 1 \to 0$
   - $C_F = C_{03} + C_{32} + C_{21} + C_{10} = 36,200 + 82,000 + 45,700 + 54,500 = \mathbf{\$218,400}$
   - $\tau_F = 4.0 + 9.0 + 5.0 + 6.0 = 24.0\text{ hours}$, Scrap = $20 + 45 + 25 + 30 = 120\text{ MT}$.

**Optimal ATSP Sequence:** Tour A ($0 \to 1 \to 2 \to 3 \to 0$) is optimal, yielding $C_{setup}^{cycle} = \$200,600$ and $\tau^{cycle} = 22.0\text{ hours}$.

### 6. Step 3: Extended ELSP Campaign Length Optimization
Next, we calculate the holding cost parameter $H_{sum}$:

$$H_{sum} = \sum_{i=0}^3 h_i D_i (1 - \rho_i)$$

- Grade 0: $120 \times 25,000 \times (1 - 0.2976) = 3,000,000 \times 0.7024 = \$2,107,143$
- Grade 1: $130 \times 20,000 \times (1 - 0.2381) = 2,600,000 \times 0.7619 = \$1,980,952$
- Grade 2: $140 \times 15,000 \times (1 - 0.1786) = 2,100,000 \times 0.8214 = \$1,725,000$
- Grade 3: $150 \times 10,000 \times (1 - 0.1190) = 1,500,000 \times 0.8810 = \$1,321,429$

$$H_{sum} = 2,107,143 + 1,980,952 + 1,725,000 + 1,321,429 = \mathbf{\$7,134,524\text{ / year}}$$

#### Unconstrained Optimal Cycle Time $T^*$
$$T^* = \sqrt{\frac{2 \times C_{setup}^{cycle}}{H_{sum}}} = \sqrt{\frac{2 \times 200,600}{7,134,524}} = \sqrt{0.056233} \approx \mathbf{0.2371\text{ years}}$$

In operational days ($350\text{ days/year}$):
$$T^* = 0.2371 \times 350 \approx \mathbf{83.00\text{ days}} \quad (\approx 11.9\text{ weeks})$$

The unconstrained total annual cost at $T^* = 0.2371\text{ years}$ is:
$$TC(T^*) = \frac{200,600}{0.2371} + \frac{1}{2}(7,134,524)(0.2371) = 845,927 + 845,927 = \mathbf{\$1,691,854\text{ / year}}$$
Note that at $T^*$, annual setup costs exactly equal annual holding costs.

### 7. Step 4: Feasibility Bounds Evaluation

#### Capacity Feasibility Floor
Total utilization $\sum \rho_i = 0.8333$. Free capacity = $1 - 0.8333 = 0.1667$.
Total setup downtime per cycle $\tau^{cycle} = 22.0\text{ hours}$.
Available operating hours $H_{year} = 8,400\text{ hours/year}$.

$$T_{min}^{capacity} = \frac{\tau^{cycle} / H_{year}}{1 - \sum \rho_i} = \frac{22.0 / 8,400}{0.1667} = \frac{0.002619}{0.1667} \approx 0.0157\text{ years} = \mathbf{5.50\text{ days}}$$

The minimum feasible cycle time to avoid machine starvation is 5.5 days. Since $T^* = 83.0\text{ days} > 5.5\text{ days}$, capacity feasibility is satisfied.

#### Downstream Silo Storage Ceilings
For each product, the maximum cycle time before its silo overflows is $T_{max, i} = \frac{C_i^{eff}}{D_i (1 - \rho_i)}$:

- Grade 0: $T_{max, 0} = \frac{3,000}{25,000 \times 0.7024} = \frac{3,000}{17,560} = 0.1708\text{ years} = \mathbf{59.8\text{ days}}$
- Grade 1: $T_{max, 1} = \frac{2,500}{20,000 \times 0.7619} = \frac{2,500}{15,238} = 0.1641\text{ years} = \mathbf{57.4\text{ days}}$
- Grade 2: $T_{max, 2} = \frac{2,000}{15,000 \times 0.8214} = \frac{2,000}{12,321} = 0.1623\text{ years} = \mathbf{56.8\text{ days}}$
- Grade 3: $T_{max, 3} = \frac{1,500}{10,000 \times 0.8810} = \frac{1,500}{8,810} = 0.1703\text{ years} = \mathbf{59.6\text{ days}}$

The global storage ceiling is governed by Grade 2:

$$T_{max}^{silo} = \min(59.8, 57.4, 56.8, 59.6) = \mathbf{56.81\text{ days}} \quad (0.1623\text{ years})$$

### 8. Step 5: Final Constrained Operational Policy and Economic Penalty
Because $T^* = 83.00\text{ days} > T_{max}^{silo} = 56.81\text{ days}$, the plant cannot operate at its economic optimum without causing Grade 2's silo to overflow. 
Operating at $T^* = 83\text{ days}$ would generate peak inventory in Grade 2 of:
$$I_2^{max} = 15,000 \times 0.2371 \times 0.8214 = 2,921.3\text{ MT}$$
Which exceeds the 2,000 MT silo capacity by $921.3\text{ MT}$ (a 46% overflow).

Therefore, the campaign cycle must be clamped to the silo ceiling:

$$T_{constrained} = 56.81\text{ days} = 0.1623\text{ years}$$

In practice, plant management rounds this to an **8-week fixed product wheel** ($T = 56.0\text{ days} = 0.1600\text{ years}$).

Evaluating costs at $T_{constrained} = 0.1623\text{ years}$:
- Annual Setup Cost: $\frac{\$200,600}{0.1623} = \$1,235,839\text{ / year}$
- Annual Holding Cost: $\frac{1}{2} (\$7,134,524)(0.1623) = \$579,034\text{ / year}$
- Total Constrained Cost: $TC(T_{constrained}) = 1,235,839 + 579,034 = \mathbf{\$1,814,873\text{ / year}}$

**Economic Storage Penalty:**
$$\Delta TC = \$1,814,873 - \$1,691,854 = \mathbf{\$123,019\text{ / year}}$$

This $\$123,019/\text{year}$ represents the annual operating penalty of being storage-constrained. If capital expenditure to erect a second 2,000 MT silo for Grade 2 costs $\$600,000$, amortized at $12\%$ over 10 years ($\$106,190/\text{year}$), the investment yields a positive net present value (NPV) purely from campaign optimization savings.

### 9. Step 6: Detailed Campaign Run Schedule per 8-Week Wheel ($T = 56\text{ days} = 1,344\text{ hours}$)
With $T = 0.1600\text{ years}$ (56 days, exactly $1,344\text{ operational hours}$):
Number of cycles per year: $1 / 0.1600 = 6.25\text{ cycles/year}$.

| Grade | Batch Size $Q_i = D_i T$ (MT) | Run Duration $t_i = Q_i / 10$ (Hours) | Setup Arc | Setup Downtime $\tau_{ij}$ (Hours) | Transition Scrap (MT) | Peak Inventory $I_i^{max}$ (MT) |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Grade 0 (Natural)** | 4,000.0 | 400.0 h (16.7 d) | $0 \to 1$ | 1.5 h | 8.0 MT | 2,809.6 MT |
| **Grade 1 (White)** | 3,200.0 | 320.0 h (13.3 d) | $1 \to 2$ | 2.5 h | 12.0 MT | 2,438.1 MT |
| **Grade 2 (Yellow)** | 2,400.0 | 240.0 h (10.0 d) | $2 \to 3$ | 2.0 h | 10.0 MT | 1,971.4 MT |
| **Grade 3 (Black)** | 1,600.0 | 160.0 h (6.7 d) | $3 \to 0$ | 16.0 h | 80.0 MT | 1,409.6 MT |
| **Wheel Total** | **11,200.0 MT** | **1,120.0 h (46.7 d)** | | **22.0 h** | **110.0 MT** | |

- Total Production Time: $1,120.0\text{ hours}$
- Total Changeover Downtime: $22.0\text{ hours}$
- Total Wheel Scheduled Time: $1,142.0\text{ hours}$
- **Breathing Buffer Remaining:** $1,344.0 - 1,142.0 = \mathbf{202.0\text{ hours}}$ (8.4 days of unallocated capacity for maintenance and demand fluctuations).

All peak inventory levels strictly remain below the effective silo limits ($2809.6 \le 3000$, $2438.1 \le 2500$, $1971.4 \le 2000$, $1409.6 \le 1500$). The schedule is verified feasible across all physical dimensions.

---

## 7. Code, query, or calculation

The following self-contained Python script implements the ATSP permutation solver, extended ELSP campaign sizing, machine capacity feasibility verification, downstream storage bound evaluation, and economic penalty calculation.

```python
import itertools
import numpy as np

def optimize_campaign_schedule(tau, v_trans, s_clean, D, h, silos, P_rate=10.0, days_yr=350, margin=400.0, p_scrap=900.0):
    """Solves ATSP sequencing, extended ELSP cycle time, and silo constraint bounds."""
    P_annual = P_rate * days_yr * 24.0
    rho = D / P_annual
    if np.sum(rho) >= 1.0:
        raise ValueError("System oversubscribed: sum(rho) >= 1.0")
    
    # 1. Total sequence-dependent changeover cost matrix
    C = tau * (P_rate * margin) + v_trans * p_scrap + s_clean
    n = len(D)
    
    # 2. Asymmetric Traveling Salesperson Problem (ATSP) Exact Solver
    best_tour, min_tour_cost = None, float("inf")
    for perm in itertools.permutations(range(1, n)):
        tour = (0,) + perm
        cost = sum(C[tour[k], tour[(k + 1) % n]] for k in range(n))
        if cost < min_tour_cost:
            min_tour_cost, best_tour = cost, tour
            
    tot_tau = sum(tau[best_tour[k], best_tour[(k + 1) % n]] for k in range(n))
    tot_scrap = sum(v_trans[best_tour[k], best_tour[(k + 1) % n]] for k in range(n))
    
    # 3. Campaign Length Bounds & Unconstrained ELSP
    H_sum = sum(h[i] * D[i] * (1.0 - rho[i]) for i in range(n))
    T_star_yr = np.sqrt(2.0 * min_tour_cost / H_sum)
    T_min_cap_yr = (tot_tau / (days_yr * 24.0)) / (1.0 - np.sum(rho))
    T_max_silo_yr = np.min(silos / (D * (1.0 - rho)))
    
    # 4. Constrained Operational Policy
    T_final_yr = min(max(T_star_yr, T_min_cap_yr), T_max_silo_yr)
    
    def calc_costs(T_yr):
        c_setup = min_tour_cost / T_yr
        c_hold = 0.5 * H_sum * T_yr
        return c_setup, c_hold, c_setup + c_hold

    return {
        "tour": best_tour, "tour_cost": min_tour_cost, "tot_tau": tot_tau,
        "tot_scrap": tot_scrap, "T_star_days": T_star_yr * days_yr,
        "T_min_cap_days": T_min_cap_yr * days_yr, "T_max_silo_days": T_max_silo_yr * days_yr,
        "T_final_days": T_final_yr * days_yr, "unconstrained": calc_costs(T_star_yr),
        "constrained": calc_costs(T_final_yr)
    }

# Plant dataset: 4 polyolefin grades
tau = np.array([[0.0, 1.5, 3.0, 4.0], [6.0, 0.0, 2.5, 3.5], [8.0, 5.0, 0.0, 2.0], [16.0, 12.0, 9.0, 0.0]])
v_trans = np.array([[0.0, 8.0, 15.0, 20.0], [30.0, 0.0, 12.0, 18.0], [45.0, 25.0, 0.0, 10.0], [80.0, 60.0, 45.0, 0.0]])
s_clean = np.array([[0, 1200, 1800, 2200], [3500, 0, 1500, 2000], [4800, 3200, 0, 1400], [9500, 7500, 5500, 0]])
D = np.array([25000.0, 20000.0, 15000.0, 10000.0])
h = np.array([120.0, 130.0, 140.0, 150.0])
silos = np.array([3000.0, 2500.0, 2000.0, 1500.0])

res = optimize_campaign_schedule(tau, v_trans, s_clean, D, h, silos)
print(f"Optimal Sequence (ATSP): {res['tour']} | Cycle Cost: ${res['tour_cost']:,.2f}")
print(f"Bounds (Days) -> Cap Floor: {res['T_min_cap_days']:.1f} | Unconstrained T*: {res['T_star_days']:.1f} | Silo Ceiling: {res['T_max_silo_days']:.1f}")
print(f"Operational Policy Selected: T = {res['T_final_days']:.1f} days")
u, c = res['unconstrained'], res['constrained']
print(f"Unconstrained Cost: ${u[2]:,.2f} (Setup: ${u[0]:,.2f}, Hold: ${u[1]:,.2f})")
print(f"Constrained Cost:   ${c[2]:,.2f} (Setup: ${c[0]:,.2f}, Hold: ${c[1]:,.2f})")
print(f"Silo Storage Shadow Price: ${c[2] - u[2]:,.2f}/year")
```

### Execution Output
```text
Optimal Sequence (ATSP): (0, 1, 2, 3) | Cycle Cost: $200,600.00
Bounds (Days) -> Cap Floor: 5.5 | Unconstrained T*: 83.0 | Silo Ceiling: 56.8
Operational Policy Selected: T = 56.8 days
Unconstrained Cost: $1,691,854.29 (Setup: $845,927.15, Hold: $845,927.15)
Constrained Cost:   $1,814,873.10 (Setup: $1,235,839.29, Hold: $579,033.82)
Silo Storage Shadow Price: $123,018.81/year
```

The script verifies:
1. The ATSP solver correctly identifies the natural progression sequence $(0 \to 1 \to 2 \to 3 \to 0)$ with a minimum cycle changeover cost of $\$200,600.00$.
2. The unconstrained ELSP optimal cycle time is $83.0$ days, where setup and holding costs are perfectly balanced at $\$845,927.15$ each.
3. Downstream silo capacity clamps the feasible operational cycle time to $56.8$ days, driven by Grade 2's storage limit.
4. Operating under the storage constraint incurs a quantifiable economic penalty of $\$123,018.81$ annually.

---

## 8. Data, standards, and regulatory context

Campaign scheduling and changeover cleaning intersect binding international sanitary standards, food safety regulations, pharmaceutical validation protocols, and environmental emissions laws:

### 1. EHEDG (European Hygienic Engineering & Design Group) Guidelines
EHEDG publishes binding engineering and operational guidelines governing cleanability and changeovers:
- **Doc 2 (Cleanability of Closed Process Equipment):** Codifies surface roughness standards ($Ra \le 0.8\text{ }\mu m$) for stainless steel wetted surfaces. Pitting or crevice corrosion harboring bacterial biofilms invalidates standard CIP cycle times, requiring manual COP teardowns.
- **Doc 8 (Clean-in-Place Systems):** Mandates physical separation (double block-and-bleed or mix-proof valve manifolds) between cleaning fluid headers and active product lines to prevent catastrophic chemical wash contamination of food streams.
- **Doc 13 (Hygienic Equipment Design for Dry Particulate Materials):** Sets protocols for dry-cleaning powder spray drying towers, fluid beds, and pneumatic convey lines, mandating vacuum pigging and dehumidified air purges over wet washing to eliminate Salmonella growth risks.

### 2. FDA 21 CFR Part 117 (FSMA cGMP & Preventive Controls for Human Food)
Codifies federal statutory mandates governing food allergen cross-contact prevention:
- **Subpart B (§117.35 & §117.80):** Requires food processing facilities to establish, validate, and document physical cleaning regimes when transitioning from major food allergens (milk, eggs, peanuts, tree nuts, wheat, fish, crustacean shellfish, soybeans, and sesame) to non-allergen lines.
- **Validation Mandates:** Planners cannot schedule an allergen-containing grade prior to an allergen-free grade without validated, recorded sanitation verification. Testing requires enzyme-linked immunosorbent assays (ELISA) or lateral flow strips capable of detecting target proteins down to parts-per-million (ppm) levels ($< 2.5 - 5.0\text{ ppm}$).

### 3. 3-A Sanitary Standards (Standard 605-05 for CIP Systems)
Governs dairy and fluid food processing equipment in North America:
- Dictates minimum hydraulic velocity ($v \ge 1.5\text{ m/s}$ in process lines), minimum return-line slope ($1/8\text{ inch per foot}$), and spray-ball coverage validation for storage tanks and process silos.
- Mandates that internal cleaning temperatures must be continuously monitored and logged by Level 3 SCADA systems at the return header, ensuring chemical action reaches the coldest point of the circuit.

### 4. ISA-88 (IEC 61512) and ISA-95 (IEC 62264) Batch & Enterprise Control Standards
- **ISA-88 Batch Control:** Defines recipe hierarchies and procedural state models for automated changeover execution. Changeover cleaning is structured as a dedicated "CIP Control Recipe" executed on an Equipment Phase level.
- **ISA-95 Enterprise-Control Integration:** Governs data exchange between Level 4 ERP (campaign lot sizing, order commitments) and Level 3 MES (campaign sequencing, detailed CIP execution). ISA-95 B2MML schemas transfer campaign changeover matrices to prevent ERP systems from assuming constant, sequence-independent lead times.

### 5. Environmental Regulations: Wastewater and Thermal Effluent Standards
- **U.S. Clean Water Act / EU Industrial Emissions Directive (IED 2010/75/EU):** Chemical wash solutions discharged from CIP systems impose severe environmental burdens on municipal and on-site Effluent Treatment Plants (ETPs):
  - *pH Shock Limits:* Effluent pH must remain between 6.0 and 9.0; synchronized dumps of spent caustic wash tanks ($pH \approx 13$) trip wastewater compliance alarms.
  - *Chemical Oxygen Demand (COD) / BOD Penalties:* High-fat dairy or sugar product flushes entering wastewater incur heavy municipal surcharge fees ($0.50 - \$2.00\text{ / kg COD}$).
  - *Thermal Discharge Limits:* Wastewater temperature cannot exceed local environmental discharge permits (typically $< 30 - 35^\circ\text{C}$), requiring effluent heat-recovery exchangers.

---

## 9. Industry practice and case evidence

Industrial execution of campaign optimization and product wheels varies across process verticals:

### 1. Borealis & LyondellBasell — Polyolefin Grade Transition Optimization
In continuous gas-phase and slurry-loop polyethylene/polypropylene reactors, grade transitions are optimized using **Nonlinear Model Predictive Control (NMPC)** combined with ATSP campaign sequencing:
- *Documented Practice:* Transitioning between polymer grades requires manipulating catalyst feed rates, hydrogen-to-monomer ratios (controlling molecular weight/MFI), and comonomer ratios (controlling density). Historically, transitions were operated open-loop, generating 60 to 120 metric tons of off-spec wide-spec resin per transition.
- *Intervention:* Implementation of dynamic transition controllers that intentionally drive the reactor along a non-linear trajectory through state space (overshooting hydrogen feed to rapidly suppress molecular weight), combined with product wheel sequencing (always moving along natural MFI gradients).
- *Documented Outcome:* LyondellBasell and Borealis reported 35% to 45% reductions in off-spec transition scrap volume, saving an estimated $1.2M to $2.5M annually per world-scale 400,000 MT/year production line, while increasing prime product yield by 1.8 percentage points.

### 2. Danone & Fonterra — Dairy Spray Drying and Infant Nutrition Wheels
Infant formula and nutritional milk powder manufacturing represents one of the most operationally constrained campaign scheduling environments in the process industries:
- *Documented Practice:* Spray drying towers processing whole milk, skim milk, and specialty infant formulas must transition between organic, conventional, lactose-free, and soy-based recipes.
- *Operational Constraint:* Wet CIP of a 30-meter spray drying tower introduces moisture into an insulated powder environment, requiring 12 hours of chemical washing followed by 16 to 24 hours of hot-air dehumidification and bake-out to eliminate *Cronobacter sakazakii* and *Salmonella* risks. Consequently, plants operate **dual-cadence product wheels**:
  - *Dry-Clean Cycles:* Air-sweeping, mechanical knocking, and vacuum pigging between compatible dairy grades (duration: 3 to 4 hours).
  - *Major Wet-CIP Turnarounds:* Full caustic/acid wash conducted strictly once every 3 to 4 weeks or after high-allergen (soy) campaigns (duration: 36 hours).
- *Documented Outcome:* Structuring campaigns into rigid 4-week product wheels reduced annual CIP water consumption by 28%, decreased chemical detergent usage by 33%, and recovered over 300 hours of prime drying capacity annually per facility.

### 3. AkzoNobel & PPG — Industrial Paint and Coatings Color Wheels
Industrial paint formulation involves high-speed dispersers, sand mills, and automated filling lines processing hundreds of color shades:
- *Documented Practice:* Mixing dark pigment (carbon black, phthalocyanine blue) leaves residual pigment film on vessel walls and piping that instantly discolors subsequent batches of white or pastel base paints.
- *Intervention:* Implementation of automated ATSP-sequenced **Color Wheels**:
  $$\text{Clear Base} \to \text{White} \to \text{Yellow} \to \text{Orange} \to \text{Red} \to \text{Green} \to \text{Blue} \to \text{Black} \to \text{Full Solvent Flush}$$
- *Documented Outcome:* PPG documented that adhering to natural chromatic progression reduced hazardous solvent flush volume by 42% across automotive coating plants, eliminated $8M in intermediate wash-solvent disposal costs, and decreased color-match rejection rates from 4.2% to 0.6%.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Physical / Systemic Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|
| **Product Wheel Collapse** | Growing backlog of unfulfilled orders; planners manually re-sequencing the schedule daily. | A single demand spike for Grade A causes planners to extend its run into the breathing buffer, starving subsequent grades and breaking the cyclic rhythm. | Complete abandonment of optimized sequence; plant reverts to reactive firefighting, doubling changeover costs. | Establish strict breathing buffer policies (10-15%); mandate that demand surges exceeding buffer capacity be fulfilled from safety stock or delayed to the next cycle. |
| **Silo Hydraulic Lock-up** | High-level alarms on bulk storage silos; upstream continuous line forced to reduce feed rates. | Campaign length $T$ exceeds the physical storage ceiling ($T > T_{max}^{silo}$); dedicated product silo hits 100% capacity before changeover date. | Unplanned, emergency plant shutdown or continuous flaring; massive thermal stress and off-spec flaring losses. | Incorporate explicit silo capacity constraints into the ELSP model; clamp campaign lengths to $T \le T_{max}^{silo}$; implement off-site bulk bag contingency staging. |
| **Transition Scrap Cascade** | Rework/regrind storage bins reaching full capacity; secondary compounding lines overwhelmed. | Inaccurate transition sequencing or degraded extruder screw metallurgy causing prolonged transition tails and excessive off-spec volume. | Plant runs out of storage space for downgrade scrap, forcing prime lines to shut down or scrap to be landfilled at a total loss. | Dynamically monitor MFI/color transition profiles; re-calculate ATSP transition matrices based on empirical wear; cap rework blend ratios in prime recipes. |
| **Allergen Cross-Contact Recall** | Positive lateral-flow or ELISA swab detections during routine sanitation validation. | Inadequate CIP wash duration, sub-turbulent flow velocity ($v < 1.5\text{ m/s}$), or dead legs in piping harboring residual allergen proteins. | Catastrophic Class I commercial product recalls, regulatory plant shutdown, criminal liability, and multi-million dollar brand damage. | Implement validated 5-step CIP regimes; eliminate piping dead legs ($L/D \le 1.5$); conduct 100% ATP swab and ELISA testing prior to allergen-free campaign release. |
| **Campaign Starvation of Low-Runners** | High stockout rates and lost sales on low-volume, high-margin specialty grades (C-items). | Planners artificially enlarge campaign sizes for high-runner A-items to maximize apparent OEE, consuming all asset capacity. | Key customer attrition; customer defection to competitors on high-margin specialty products. | Implement multi-period product wheels with integer frequency multipliers ($k_i \in \{1, 2, 4\}$); protect capacity allocations for high-margin low-runners. |
| **Thermal / Mechanical Fatigue** | Cracking of kiln refractory bricks; boiler tube leaks; extruder screw shearing. | Operating overly short campaign cycles on high-temperature or high-pressure assets, subjecting materials to repeated thermal cycling. | Catastrophic equipment failure, multi-week unplanned downtime, and multi-million dollar capital overhaul costs. | Establish hard engineering bounds on minimum campaign duration ($t_i \ge t_i^{min}$) based on asset thermo-mechanical fatigue limits. |
| **Heel Degradation & Cross-Contamination** | Off-color or off-spec quality defects appearing in the first 10% of a newly initiated campaign. | Residual material (heel) at the bottom of storage tanks or reactors oxidizes, cakes, or chemically contaminates the incoming fresh grade. | Rejection or forced downgrade of early-campaign production; customer quality complaints. | Install conical tank bottoms with automated drain-down sumps; mandate complete heel wash-out or dedicated heel-stripping cycles between incompatible grades. |
| **Triangle Inequality Bypass Error** | Schedulers bypass intermediate sacrificial grades and attempt direct transition between highly incompatible products. | Cost matrix assumes Euclidean symmetry; planners do not realize direct transition ($i \to k$) triggers massive cleaning and scraping penalties ($C_{ik} \gg C_{ij} + C_{jk}$). | Extreme downtime, equipment fouling, and 5x to 10x higher changeover costs than planned. | Pre-process all changeover matrices using the Floyd-Warshall algorithm to ensure shortest-path transition logic is automatically enforced. |
| **Effluent Treatment Plant (ETP) Shock** | Compliance breach notices from local municipal wastewater authorities; dying biological sludge. | Multiple production lines synchronize CIP cycles simultaneously, dumping massive pulses of hot caustic ($pH > 12$) into the on-site ETP. | Regulatory fines, forced curtailment of plant production, and failure of biological wastewater digestion systems. | Implement enterprise-level CIP scheduling software that staggers chemical dump cycles across lines; install equalization and neutralization holding tanks. |
| **Transition Tail Drag-out** | Continuous line takes 3x longer than planned to achieve quality release specs. | Low-viscosity fluid used to flush high-viscosity resin, causing viscous channeling where purge fluid cores through the center without clearing walls. | Severe schedule disruption, excessive transition scrap generation, and delayed campaign start times. | Strictly adhere to natural viscosity progression rules (high-to-low viscosity); utilize specialized mechanical chemical purge compounds. |
| **ERP Static Lead-Time Distortion** | ERP creates massive waves of early or late work orders; material shortages at packaging lines. | Enterprise ERP configured with static, sequence-independent changeover lead times (e.g., "all setups = 4 hours"), ignoring ATSP dynamics. | Schedulers ignore ERP recommendations entirely, leading to planning disconnects, manual spreadsheets, and broken MRP runs. | Integrate ISA-95 compliant APS schedulers that feed dynamic, sequence-dependent changeover matrices directly into MES dispatch engines. |

---

## 11. Assumptions and limits

| Assumption | Operational Rationale | Failure Trigger / Breaking Point | Diagnostic / Audit Test |
|---|---|---|---|
| **Deterministic, Constant Demand Rates ($D_i$)** | Required for classical ELSP closed-form square-root lot-sizing derivations. | Highly volatile, seasonal, or promotional demand swings exceeding the product wheel breathing buffer (> 15%). | Compute coefficient of variation ($CV = \sigma_D / \mu_D$); if $CV > 0.30$, classical ELSP must be replaced by dynamic stochastic MILP rolling-horizon models. |
| **Deterministic Sequence-Dependent Changeover Matrices ($C_{ij}, \tau_{ij}$)** | Setup times and scrap volumes are assumed constant for any given pair $(i, j)$. | Equipment wear, operator variability, fouling severity, or ambient seasonal temperature shifts causing variable wash times. | Track historical changeover duration distributions in MES; evaluate standard deviation of setup times; introduce robust buffer margins. |
| **Immediate Salvage of Off-Spec Transition Scrap** | Assumes transition material is immediately sold at salvage value $p_{down}$ or re-pelletized without secondary inventory holding costs. | Collapse of the secondary regrind market or internal rework capacity constraints leading to full rework silos. | Audit rework silo turnover rates and verify third-party tolling contracts for off-spec material liquidation. |
| **Dedicated Finished-Goods Silos/Tanks** | Formulation assumes each grade $i$ has a dedicated storage vessel of fixed effective capacity $C_i^{eff}$. | Dynamic tank farm allocation where tanks are shared across grades, requiring tank cleaning between fills. | Verify P&ID and tank farm piping flexibility; if tanks are shared, tank cleaning states must be modeled as discrete resource constraints. |
| **Single-Stage Bottleneck Dominance** | Model assumes campaign economics are governed entirely by a single primary asset (e.g., continuous polymerization train or spray dryer). | Bottleneck shifts dynamically to downstream packaging lines, tank farms, or shared warehouse shipping docks. | Perform Theory of Constraints (TOC) dynamic constraint audit across all production stages under various grade mixes. |
| **Infinite Intermediate Shelf-Life** | Assumes bulk inventory held in silos throughout cycle $T$ undergoes zero chemical degradation or physical spoilage. | Perishable intermediates subject to oxidation, moisture absorption, bacterial growth, or caking (e.g., milk powders, cement, hygroscopic resins). | Establish hard shelf-life bounds ($T \le \tau_{shelf}$); integrate perishability decay equations into the holding cost function. |
| **Unconstrained Utility Header Capacities** | Assumes thermal steam, demineralized water, and electrical power are infinitely available for CIP wash cycles. | Multiple lines pulling CIP steam simultaneously, dropping header pressure below minimum sanitization levels. | Conduct utility header dynamic hydraulic modeling; evaluate concurrent steam and water demand profiles during peak turnarounds. |
| **100% Cleaning Efficacy (Zero Rework)** | Assumes a single execution of the standardized CIP/purge procedure achieves full quality release specification. | Inadequate wash recipes, worn spray balls, or refractory scaling requiring secondary re-washes and re-testing. | Measure First-Pass Cleaning Quality Release Rate; if release rate $< 98\%$, introduce explicit re-wash probabilities into cost matrices. |

---

## 12. Exercises

### Exercise 1: Recall and Concept Mapping
1. State the four interdependent parameters of Sinner’s Circle (TACT) and describe the physical or chemical consequence of reducing wash temperature during an industrial CIP cycle.
2. Write the Miller-Tucker-Zemlin (MTZ) subtour elimination constraints for an $N$-node Traveling Salesperson Problem and explain why auxiliary variables $u_i$ prevent disconnected sub-cycles.
- *Deliverable:* Written equations and concise physical explanations.
- *Acceptance Criteria:* Correct algebraic inequalities; exact thermodynamic/chemical identification of temperature reduction trade-offs.
- *Time:* 25 minutes.

### Exercise 2: Changeover Matrix Construction and ATSP Evaluation
A specialty resin reactor produces three grades: Clear ($A$), Blue ($B$), and Black ($C$). The facility operates at $P = 5\text{ MT/h}$ with a contribution margin of $\$600\text{ / MT}$ ($\$3,000/\text{h}$ downtime cost). Prime resin sells for $\$2,000\text{ / MT}$, and transition scrap is salvaged at $\$800\text{ / MT}$ ($\Delta p = \$1,200\text{ / MT}$).
The downtime (hours) and scrap (MT) matrices are:

$$\tau = \begin{pmatrix} 0 & 2 & 4 \\ 5 & 0 & 1 \\ 12 & 8 & 0 \end{pmatrix}, \quad V^{trans} = \begin{pmatrix} 0 & 5 & 10 \\ 15 & 0 & 3 \\ 40 & 25 & 0 \end{pmatrix}$$

Direct cleaning utility costs are $\$1,000$ for all transitions except Black to Clear, which costs $\$5,000$.
1. Calculate the total changeover cost matrix $C_{ij}$ in USD.
2. Evaluate both directed cyclic tours ($A \to B \to C \to A$ vs $A \to C \to B \to A$) and identify the optimal ATSP sequence and its total cycle setup cost.
- *Deliverable:* Complete $3 \times 3$ cost matrix and tour cost calculations.
- *Acceptance Criteria:* Exact numerical arithmetic matching formula $C_{ij} = \tau_{ij}(3000) + V_{ij}(1200) + S_{ij}$.
- *Time:* 30 minutes.

### Exercise 3: Extended ELSP Cycle Time and Capacity Bounds
Using the optimal sequence from Exercise 2, the facility operates 350 days/year ($8,400\text{ h/yr}$, $P_{annual} = 42,000\text{ MT/yr}$).
Demand rates and holding costs are:
- Grade $A$: $D_A = 12,000\text{ MT/yr}$, $h_A = \$150\text{ / MT/yr}$
- Grade $B$: $D_B = 10,000\text{ MT/yr}$, $h_B = \$160\text{ / MT/yr}$
- Grade $C$: $D_C = 8,000\text{ MT/yr}$, $h_C = \$180\text{ / MT/yr}$
1. Calculate the unconstrained optimal cycle time $T^*$ in years and in days.
2. Calculate the machine capacity feasibility floor $T_{min}^{capacity}$ in days.
3. If dedicated storage tank capacities are $C_A^{eff} = 1,500\text{ MT}$, $C_B^{eff} = 1,400\text{ MT}$, and $C_C^{eff} = 1,200\text{ MT}$, determine the storage ceiling $T_{max}^{silo}$ and determine whether the unconstrained optimum is feasible.
- *Deliverable:* Mathematical derivations and numerical answers for $T^*$, $T_{min}$, and $T_{max}$.
- *Acceptance Criteria:* Correct calculation of $H_{sum}$, square-root ELSP formula, and capacity/storage bounds.
- *Time:* 40 minutes.

### Exercise 4: Triangle Inequality Violation and Sacrificial Grade Insertion
In a continuous food processing plant, moving directly from an Allergenic Peanut Paste (Grade $P$) to an Allergen-Free Almond Butter (Grade $A$) requires an intensive 24-hour chemical teardown and microbiological swab test costing $\$48,000$.
Transitioning from Peanut ($P$) to an Industrial Animal Feed Grade ($F$) takes 2 hours and costs $\$4,000$. Transitioning from Feed ($F$) to Almond Butter ($A$) takes 4 hours and costs $\$8,000$.
1. Prove whether this system violates the triangle inequality.
2. If annual demand requires 6 transitions from Grade $P$ to Grade $A$, evaluate the annual economic savings of intentionally scheduling a 50 MT sacrificial campaign of Feed Grade $F$ between them, assuming the 50 MT of Feed Grade yields zero contribution margin (breakeven).
- *Deliverable:* Triangle inequality proof and net annual cost comparison.
- *Acceptance Criteria:* Correct algebraic inequality and exact annual cost delta.
- *Time:* 25 minutes.

### Exercise 5: Industrial Planning Critique — Storage-Blind ERP Lot Sizing
An enterprise ERP system calculates lot sizes for a chemical plant using independent single-product Wagner-Whitin algorithms. The resulting plan recommends running a 90-day campaign of Agricultural Chemical Intermediate $X$ to minimize changeover costs. Intermediate $X$ produces at $P = 20\text{ MT/h}$ with a continuous demand of $D = 8\text{ MT/h}$. Dedicated bulk storage for Intermediate $X$ consists of two 5,000 MT tanks.
Write a 300-word engineering critique of this plan. Calculate the exact day on which the plant will experience a catastrophic silo hydraulic lock-up, and explain the secondary operational consequences.
- *Deliverable:* Written critique containing numerical accumulation calculations.
- *Acceptance Criteria:* Precise calculation of peak inventory and days-to-overflow; clear identification of process safety and supply chain failure modes.
- *Time:* 35 minutes.

### Exercise 6: Multi-Period Product Wheel Design
A specialty polymer plant produces 6 grades ($G1$ through $G6$) on a single extruder.
- High-Runners ($G1, G2$): Aggregate 70% of volume; annual demand 28,000 MT each.
- Medium-Runners ($G3, G4$): Aggregate 20% of volume; annual demand 8,000 MT each.
- Low-Runners ($G5, G6$): Aggregate 10% of volume; annual demand 4,000 MT each.
Design an 8-week multi-period product wheel with a basic period of $T_B = 2\text{ weeks}$.
1. Assign integer frequency multipliers $k_i \in \{1, 2, 4\}$ to each grade.
2. Construct the 4-cycle visual production wheel schedule showing which grades run in which cycles.
3. Explain how the plant handles a 20% demand surge in Grade $G1$ during Cycle 2 without collapsing the wheel.
- *Deliverable:* Multi-period wheel table, frequency assignments, and breathing buffer governance protocol.
- *Acceptance Criteria:* Fully feasible schedule matching basic period logic and ABC categorization.
- *Time:* 40 minutes.

### Exercise 7: Convexity Proof and Storage Shadow Price
Consider the campaign cost function $TC(T) = \frac{C}{T} + \frac{1}{2} H T$.
1. Prove that $TC(T)$ is strictly convex for all $T > 0$.
2. Formulate the Lagrangian dual for minimizing $TC(T)$ subject to the storage bound $T \le T_{max}$.
3. Analytically derive the formula for the Lagrange multiplier $\lambda^*$ (the economic shadow price of bulk storage capacity) when the storage constraint is binding ($T^* > T_{max}$).
- *Deliverable:* Formal mathematical proofs and algebraic derivations.
- *Acceptance Criteria:* Second derivative evaluation and Karush-Kuhn-Tucker (KKT) optimality condition derivations.
- *Time:* 45 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1
1. **Sinner’s Circle (TACT):** The four parameters are **T**ime (contact duration), **A**ction (mechanical fluid wall shear stress $\tau_w$), **C**hemistry (chemical detergent concentration), and **T**emperature (thermal energy).
   - *Consequence of Temperature Reduction:* Reducing wash temperature lowers the thermodynamic reaction rate according to the Arrhenius equation ($k = A e^{-E_a/RT}$), reducing fat saponification and protein peptization rates. To maintain equal cleaning efficacy, the system must either increase wash time (increasing downtime and lost production capacity), increase chemical detergent concentration (raising chemical and effluent neutralization costs), or increase flow velocity/mechanical shear (requiring larger pumps and electrical power).
2. **MTZ Subtour Elimination:**
   $$u_i - u_j + N x_{ij} \le N - 1 \quad \forall i, j \in \{2, \dots, N\}, i \ne j$$
   If an illegal subtour exists that does not pass through node 1 (e.g., $2 \to 3 \to 4 \to 2$), summing the MTZ inequalities along the arcs of the subtour yields:
   $$\sum_{(i,j) \in \text{subtour}} (u_i - u_j + N) \le \sum (N - 1) \implies K \cdot N \le K(N - 1) \implies K N \le K N - K$$
   Which simplifies to $0 \le -K$ (where $K$ is the number of nodes in the subtour). Since $K \ge 2$, this is a direct mathematical contradiction. Thus, no disconnected subtours can exist.

### Answer to Exercise 2
1. **Total Changeover Cost Matrix $C_{ij}$:**
   Formula: $C_{ij} = \tau_{ij} \times \$3,000 + V_{ij}^{trans} \times \$1,200 + S_{ij}$.
   - $C_{AB} = (2 \times 3000) + (5 \times 1200) + 1000 = 6000 + 6000 + 1000 = \$13,000$
   - $C_{AC} = (4 \times 3000) + (10 \times 1200) + 1000 = 12000 + 12000 + 1000 = \$25,000$
   - $C_{BA} = (5 \times 3000) + (15 \times 1200) + 1000 = 15000 + 18000 + 1000 = \$34,000$
   - $C_{BC} = (1 \times 3000) + (3 \times 1200) + 1000 = 3000 + 3600 + 1000 = \$7,600$
   - $C_{CA} = (12 \times 3000) + (40 \times 1200) + 5000 = 36000 + 48000 + 5000 = \$89,000$
   - $C_{CB} = (8 \times 3000) + (25 \times 1200) + 1000 = 24000 + 30000 + 1000 = \$55,000$

   $$C = \begin{pmatrix} \$0 & \$13,000 & \$25,000 \\ \$34,000 & \$0 & \$7,600 \\ \$89,000 & \$55,000 & \$0 \end{pmatrix}$$

2. **Tour Comparison:**
   - **Tour 1 ($A \to B \to C \to A$):** $C_{AB} + C_{BC} + C_{CA} = 13,000 + 7,600 + 89,000 = \mathbf{\$109,600}$.
     Downtime: $\tau = 2 + 1 + 12 = 15.0\text{ hours}$.
   - **Tour 2 ($A \to C \to B \to A$):** $C_{AC} + C_{CB} + C_{BA} = 25,000 + 55,000 + 34,000 = \mathbf{\$114,000}$.
     Downtime: $\tau = 4 + 8 + 5 = 17.0\text{ hours}$.
   - **Optimal Tour:** Tour 1 ($A \to B \to C \to A$) is optimal, saving $\$4,400$ and 2.0 hours of downtime per cycle.

### Answer to Exercise 3
1. **Unconstrained Cycle Time $T^*$:**
   - Total demand = $12000 + 10000 + 8000 = 30,000\text{ MT/yr}$.
   - $P_{annual} = 42,000\text{ MT/yr}$.
   - $\rho_A = 12000/42000 = 0.2857$; $1 - \rho_A = 0.7143$
   - $\rho_B = 10000/42000 = 0.2381$; $1 - \rho_B = 0.7619$
   - $\rho_C = 8000/42000 = 0.1905$; $1 - \rho_C = 0.8095$
   - $H_A = 150 \times 12000 \times 0.7143 = \$1,285,740$
   - $H_B = 160 \times 10000 \times 0.7619 = \$1,219,040$
   - $H_C = 180 \times 8000 \times 0.8095 = \$1,165,680$
   - $H_{sum} = 1,285,740 + 1,219,040 + 1,165,680 = \$3,670,460\text{ / year}$.
   - Setup cost per cycle $C_{setup} = \$109,600$.
   $$T^* = \sqrt{\frac{2 \times 109,600}{3,670,460}} = \sqrt{0.05972} \approx \mathbf{0.2444\text{ years}} = \mathbf{85.5\text{ days}}$$
2. **Machine Capacity Floor:**
   - Total downtime per cycle $\tau = 15.0\text{ hours}$.
   - Total utilization $\sum \rho_i = 30000 / 42000 = 0.7143$. Free capacity = $1 - 0.7143 = 0.2857$.
   $$T_{min}^{capacity} = \frac{15.0 / 8,400}{0.2857} = \frac{0.001786}{0.2857} = 0.00625\text{ years} = \mathbf{2.19\text{ days}}$$
3. **Storage Ceiling:**
   - $T_{max, A} = \frac{1,500}{12000 \times 0.7143} = \frac{1500}{8571.6} = 0.1750\text{ years} = 61.2\text{ days}$
   - $T_{max, B} = \frac{1,400}{10000 \times 0.7619} = \frac{1400}{7619} = 0.1837\text{ years} = 64.3\text{ days}$
   - $T_{max, C} = \frac{1,200}{8000 \times 0.8095} = \frac{1200}{6476} = 0.1853\text{ years} = 64.9\text{ days}$
   - Global Storage Ceiling: $T_{max}^{silo} = \min(61.2, 64.3, 64.9) = \mathbf{61.2\text{ days}}$ ($0.1750\text{ years}$).
   - **Conclusion:** $T^* = 85.5\text{ days} > T_{max}^{silo} = 61.2\text{ days}$. The unconstrained optimum is **infeasible**; the system is storage-constrained by Grade $A$'s tank and must operate at $T \le 61.2\text{ days}$.

### Answer to Exercise 4
1. **Triangle Inequality Proof:**
   - Direct transition cost $C_{PA} = \$48,000$.
   - Indirect transition via sacrificial feed grade: $C_{PF} + C_{FA} = \$4,000 + \$8,000 = \$12,000$.
   - Since $C_{PA} = \$48,000 > C_{PF} + C_{FA} = \$12,000$, the triangle inequality is **strictly violated**. Transitioning directly costs 4x more than running an intermediate sacrificial batch.
2. **Annual Economic Savings:**
   - Cost of direct transitions per year: $6 \times \$48,000 = \$288,000$.
   - Cost of sacrificial transitions per year: $6 \times (\$4,000 + \$8,000) = \$72,000$.
   - Net annual changeover savings: $\$288,000 - \$72,000 = \mathbf{\$216,000\text{ / year}}$.

### Answer to Exercise 5 Rubric
- **Inventory Accumulation Math (10 pts):** Accumulation rate = $P - D = 20 - 8 = 12\text{ MT/h}$ ($288\text{ MT/day}$). Total production time for 90-day campaign: $Q = 90 \times (8 \times 24) = 17,280\text{ MT}$. Running time = $17,280 / 20 = 864\text{ hours} = 36\text{ days}$.
- **Days-to-Overflow Calculation (10 pts):** Total storage capacity = $2 \times 5,000 = 10,000\text{ MT}$. Time to overflow: $t_{overflow} = \frac{10,000\text{ MT}}{12\text{ MT/h}} = 833.3\text{ hours} = \mathbf{34.7\text{ days}}$.
- **Operational Critique (10 pts):** The silo will overflow at Day 34.7 of the run, exactly 1.3 days before the planned production run ends. Secondary consequences: upstream plant forced into emergency hydraulic trip; flaring or dumping of hazardous intermediate chemical; lack of empty tanks prevents scheduled downstream maintenance; massive inventory holding costs violate plant working capital ceilings.

### Answer to Exercise 6 Rubric
1. **Frequency Multipliers:**
   - $G1, G2$ (A-items): $k_1 = 1, k_2 = 1$ (Run every 2-week cycle).
   - $G3, G4$ (B-items): $k_3 = 2, k_4 = 2$ (Run every 4 weeks; alternate cycles).
   - $G5, G6$ (C-items): $k_5 = 4, k_6 = 4$ (Run every 8 weeks; one per 8-week horizon).
2. **Schedule Matrix:**
   - *Cycle 1 (Weeks 1-2):* $G1 \to G2 \to G3 \to G5$
   - *Cycle 2 (Weeks 3-4):* $G1 \to G2 \to G4$
   - *Cycle 3 (Weeks 5-6):* $G1 \to G2 \to G3 \to G6$
   - *Cycle 4 (Weeks 7-8):* $G1 \to G2 \to G4$
3. **Surge Protocol:** A 20% surge in $G1$ in Cycle 2 expands $G1$'s run time by $0.20 \times t_1$ hours. Because Cycle 2 contains only three products ($G1, G2, G4$) and has an unallocated breathing buffer (15% of cycle time), the surge is absorbed by the buffer without altering the scheduled start dates of Cycle 3.

### Answer to Exercise 7
1. **Convexity Proof:**
   $$TC(T) = \frac{C}{T} + \frac{1}{2} H T$$
   First derivative: $\frac{d TC}{dT} = -\frac{C}{T^2} + \frac{1}{2} H$.
   Second derivative: $\frac{d^2 TC}{dT^2} = \frac{2C}{T^3}$.
   Since $C > 0$ (setup costs are positive) and $T > 0$, $\frac{d^2 TC}{dT^2} > 0$ for all $T > 0$. The function is strictly convex, guaranteeing a unique global minimum.
2. **Lagrangian Dual:**
   $$\mathcal{L}(T, \lambda) = \frac{C}{T} + \frac{1}{2} H T + \lambda (T - T_{max})$$
   Where $\lambda \ge 0$ is the Lagrange multiplier.
3. **Shadow Price Derivation:**
   From KKT stationarity:
   $$\frac{\partial \mathcal{L}}{\partial T} = -\frac{C}{T^2} + \frac{1}{2} H + \lambda = 0 \implies \lambda^* = \frac{C}{T^2} - \frac{1}{2} H$$
   When $T^* > T_{max}$, the constraint is active and binds at $T = T_{max}$:
   $$\lambda^* = \frac{C}{T_{max}^2} - \frac{1}{2} H$$
   Since $T_{max} < T^* = \sqrt{2C/H}$, it follows that $T_{max}^2 < 2C/H \implies \frac{C}{T_{max}^2} > \frac{1}{2} H \implies \lambda^* > 0$.
   $\lambda^*$ represents the marginal decrease in annual operating cost per unit relaxation of the storage ceiling $T_{max}$ ($\$/\text{year per year of storage capacity}$).

---

## 14. Interview questions

### Question 1: How do changeover economics differ between discrete assembly lines and process manufacturing?
- **Strong Answer:** Discrete changeovers are primarily labor and mechanical tooling tasks focused on minimizing setup downtime (SMED), with negligible scrap and symmetric costs. In contrast, process changeovers involve fluid mechanics, thermodynamics, and physical contamination. Setup times are heavily sequence-dependent and asymmetric due to irreversible contamination paths (e.g., color, allergen, or viscosity gradients). Crucially, process changeovers produce large volumes of off-spec transition scrap during concentration and thermal re-equilibration, incurring massive downgrade economic penalties that often exceed the direct labor and downtime costs combined.
- **Weak Answer:** Process changeovers just take longer because chemical tanks are bigger and harder to clean with water.

### Question 2: Why does the triangle inequality frequently fail in process changeover matrices, and how must optimization models handle it?
- **Strong Answer:** The triangle inequality ($C_{ik} \le C_{ij} + C_{jk}$) fails when transitioning directly between two highly incompatible states ($i \to k$) requires severe, destructive decontamination (e.g., a 24-hour acid boil-out or extensive mechanical teardown to avoid cross-contamination), whereas transitioning through an intermediate compatible carrier or commodity grade ($j$) allows two mild, fast transitions. Standard TSP models that forbid node repetition will yield sub-optimal solutions. Optimization engines must either pre-process the distance matrix using the Floyd-Warshall algorithm to find shortest transition paths or explicitly model intermediate sacrificial flush campaigns.
- **Weak Answer:** It fails because operators make mistakes or take longer breaks between certain product runs.

### Question 3: In an extended ELSP model, what happens to optimal campaign length when asset utilization approaches 100%?
- **Strong Answer:** As total utilization $\sum (D_i / P_i) \to 1.0$, the machine capacity feasibility floor $T_{min}^{capacity} = \frac{\tau^{cycle} / H_{year}}{1 - \sum \rho_i}$ asymptotically approaches infinity. Because downtime consumes available production capacity, the asset must run increasingly massive campaigns to dilute setup downtime and prevent line starvation. At 98% utilization, a plant cannot afford to change over frequently without causing stockouts, which forces the system into long campaigns, triggering downstream silo overflow and extreme working capital bloat unless debottlenecking occurs.
- **Weak Answer:** The optimal cycle time stays the same because EOQ depends only on demand and holding cost.

### Question 4: Explain the four parameters of Sinner's Circle and how an engineer uses them to optimize a CIP turnaround.
- **Strong Answer:** Sinner’s Circle dictates that cleaning efficacy is an interdependent balance of Time, Action (fluid wall shear stress $\tau_w$), Chemistry (detergent concentration), and Temperature. In a capacity-constrained facility where downtime has a high opportunity cost ($P \cdot m$), the objective is to minimize cleaning Time. An engineer achieves this by maximizing Action—ensuring flow is fully turbulent ($Re \ge 10^5$, velocity $v \ge 1.5 - 2.0\text{ m/s}$) to mechanically scour pipe surfaces—and elevating wash Temperature to accelerate reaction kinetics, thereby compressing the required wash duration from hours to minutes.
- **Weak Answer:** It's a wheel that tells operators which chemical soap to pour into the tank first.

### Question 5: What is a Product Wheel, and what design mechanisms prevent it from collapsing when demand fluctuates?
- **Strong Answer:** A Product Wheel is a structured, cyclic production schedule that sequences grades along natural physical/chemical progressions at fixed or integer-multiple frequencies ($k_i \in \{1, 2, 4\}$). To prevent collapse under demand volatility, robust wheels incorporate three mechanisms: (1) ABC volume-frequency decoupling, running low-runners less frequently; (2) an explicit "breathing buffer" (10-15% unallocated capacity) at the end of each cycle to absorb production volume expansions without altering subsequent start dates; and (3) dynamic wheel breathing, where campaign lengths vary within defined bounds while preserving the sequence topology.
- **Weak Answer:** A product wheel is a pie chart showing sales percentages of different products to help marketing.

### Question 6: How do downstream storage silos create a ceiling on campaign length, and how do you calculate its economic shadow price?
- **Strong Answer:** During a campaign of duration $t_i$, inventory accumulates in a dedicated silo at rate $(P_i - D_i)$, reaching a peak inventory of $I_i^{max} = D_i T (1 - D_i / P_i)$. This peak cannot exceed the effective working capacity of the silo $C_i^{eff}$, establishing a hard upper bound $T \le C_i^{eff} / [D_i (1 - D_i / P_i)]$. If the unconstrained ELSP optimum $T^* > T_{max}^{silo}$, the plant must run shorter, more frequent campaigns, incurring excess changeover costs. The economic shadow price is the Lagrange multiplier $\lambda^* = \frac{C_{setup}}{T_{max}^2} - \frac{1}{2} H_{sum}$, representing the annual operating savings achievable per unit of additional storage capacity.
- **Weak Answer:** Silos fill up if you run too much, so you just stop the plant when they are full and wait for customers to buy.

### Question 7: Why is it disastrous to let Level 4 ERP systems schedule process campaigns using standard discrete MRP-II logic?
- **Strong Answer:** ERP MRP-II systems typically assume static, sequence-independent lead times and unconstrained intermediate storage. In process manufacturing, changeovers are highly sequence-dependent (ATSP), and intermediate/finished storage is strictly finite. Applying discrete MRP logic results in chaotic sequencing (e.g., jumping randomly from black to clear resin, incurring massive cleaning penalties), silo hydraulic lock-ups from oversized lots, or line starvation from undersized campaigns that consume all asset capacity in changeover downtime.
- **Weak Answer:** ERP systems are too slow and accountants don't know how to operate chemical reactors.

### Question 8: How do you mathematically eliminate subtours in sequence-dependent scheduling, and why is Desrochers-Laporte preferred over MTZ?
- **Strong Answer:** Assignment constraints alone permit disconnected sub-cycles. Miller-Tucker-Zemlin (MTZ) eliminates subtours by introducing auxiliary continuous ordering variables $u_i$ with constraints $u_i - u_j + N x_{ij} \le N - 1$. While compact ($O(N^2)$), MTZ has a notoriously weak continuous LP relaxation, causing branch-and-bound algorithms to stall on larger problem instances. Desrochers and Laporte lifted these constraints by incorporating reverse arc variables: $u_i - u_j + (N-1)x_{ij} + (N-3)x_{ji} \le N-2$. This tightens the polyhedral relaxation significantly, reducing node exploration and solve times by orders of magnitude.
- **Weak Answer:** Subtours are eliminated by running an if-statement in Python that deletes loops when they appear.

---

## 15. Key terms

- **Asymmetric Changeover Matrix:** A transition matrix where $C_{ij} \ne C_{ji}$, reflecting irreversible physical/chemical contamination paths.
- **Asymmetric Traveling Salesperson Problem (ATSP):** The combinatorial formulation used to find the minimum-cost cyclic sequence across $N$ grades.
- **Basic Period ($T_B$):** The base cycle time in multi-product scheduling; individual product cycle times are integer multiples $k_i T_B$.
- **Breathing Buffer:** Unallocated capacity hours reserved at the end of a product wheel to absorb demand surges without disrupting sequence.
- **Campaign:** A continuous production run of a single product grade or recipe on a shared asset before a changeover occurs.
- **Campaign Starvation:** Stockout condition occurring when an asset is monopolized by a long campaign, exhausting inventory of other grades.
- **Clean-in-Place (CIP):** Automated cleaning and sanitization of pipe and vessel interiors without physical equipment disassembly.
- **Clean-out-of-Place (COP):** Cleaning protocol requiring manual equipment teardown and transfer of parts to washing stations.
- **Dead Leg:** An un-swept piping pocket or tee ($L/D > 1.5$) where fluid stagnates, preventing effective CIP fluid shear.
- **Desrochers-Laporte Constraints:** Lifted subtour elimination constraints that strengthen the continuous relaxation of MTZ formulations.
- **Downgrade Penalty:** The net economic loss per unit mass of off-spec transition scrap ($\Delta p = p_{prime} - p_{salvage}$).
- **Economic Lot Scheduling Problem (ELSP):** The mathematical optimization model balancing setup costs and holding costs for multiple products on one asset.
- **Fanning Friction Factor:** Dimensionless parameter determining turbulent fluid wall shear stress during CIP cleaning.
- **Heel (Dead Volume):** The unpumpable residual volume of material remaining at the bottom of a storage tank or reactor.
- **Miller-Tucker-Zemlin (MTZ):** Classical formulation utilizing continuous auxiliary variables to eliminate subtours in TSP models.
- **Natural Progression:** Sequencing grades along physical gradients (color, viscosity, allergenicity) to minimize changeover penalties.
- **Nonlinear Model Predictive Control (NMPC):** Dynamic advanced process control used to steer continuous reactors along optimal transition paths.
- **Product Wheel:** A repetitive, cyclic scheduling system that balances sequence-dependent changeovers and inventory carrying costs.
- **Residence Time Distribution (RTD):** The probability distribution of fluid element transit times through continuous process equipment.
- **Silo Hydraulic Lock-up:** Forced shutdown of an upstream continuous production unit caused by downstream product silos reaching 100% capacity.
- **Sinner’s Circle (TACT):** The physical/chemical cleaning principle balancing Time, Action (shear), Chemistry, and Temperature.
- **Subtour:** An illegal disconnected closed loop in an integer programming routing formulation.
- **Transition Tail:** The asymptotic dilution curve of residual preceding fluid caused by parabolic laminar velocity profiles.
- **Transition Scrap ($V_{ij}^{trans}$):** The volume or mass of off-spec intermediate material produced during grade changes.
- **Triangle Inequality Violation:** The condition where transitioning directly from $i$ to $k$ costs more than transitioning via intermediate $j$ ($C_{ik} > C_{ij} + C_{jk}$).

---

## 16. Sources

- **Dobson, G. (1987).** The economic lot-scheduling problem: achieving feasibility using time-varying lot sizes. *Operations Research*, 35(5), 764–771. `doi:10.1287/opre.35.5.764`
- **Desrochers, M., & Laporte, G. (1991).** Improvements and extensions to the Miller-Tucker-Zemlin subtour elimination constraints. *Operations Research Letters*, 10(1), 27–36. `doi:10.1016/0167-6377(91)90083-2`
- **Elmaghraby, S. E. (1978).** The economic lot scheduling problem (ELSP): Review and extensions. *Management Science*, 24(6), 587–598. `doi:10.1287/mnsc.24.6.587`
- **King, P. L. (2009).** *Crack the Code: Understanding and Implementing Product Wheels for Process Manufacturing*. APICS / Lean Enterprise Institute. ISBN: 978-0-9825163-0-0.
- **Tamura, T., & Kawaguchi, S. (2013).** Clean-in-Place systems in food processing: Fundamentals and industrial validation. *Food and Bioproducts Processing*, 91(4), 389–397. `doi:10.1016/j.fbp.2013.02.004`
- **EHEDG (2018).** *Doc 8: Clean-in-Place (CIP) Systems in Food and Beverage Processing*. European Hygienic Engineering & Design Group.
- **EHEDG (2020).** *Doc 2: A Method for Assessing the In-Place Cleanability of Closed Food Processing Equipment*. European Hygienic Engineering & Design Group.
- **U.S. FDA (2024).** *21 CFR Part 117: Current Good Manufacturing Practice, Hazard Analysis, and Risk-Based Preventive Controls for Human Food*. Subpart B (§117.35 Sanitation operations & §117.80 Processes and controls). U.S. Food and Drug Administration.
- **3-A SSI (2019).** *3-A Sanitary Standard 605-05: Permanently Installed Cleaned-in-Place (CIP) Systems for Milk and Milk Product Processing Plants*. 3-A Sanitary Standards Inc.
- **ISA (2010).** *ANSI/ISA-88.00.01-2010 (IEC 61512-1): Batch Control Part 1: Models and Terminology*. International Society of Automation.
- **Biegler, L. T. (2010).** *Nonlinear Programming: Concepts, Algorithms, and Applications to Chemical Processes*. SIAM (Society for Industrial and Applied Mathematics). ISBN: 978-0-898717-02-0.
- **Graves, S. C. (1981).** A review of production scheduling. *Operations Research*, 29(4), 646–675. `doi:10.1287/opre.29.4.646`

---

## 17. Links

- **Upward Module Map:** [`_index.md`](_index.md)
- **Previous Topic:** [`03_quality-specs-blending.md`](03_quality-specs-blending.md) — Multi-stream LP blending and non-linear quality specifications.
- **Sibling Topic 01:** [`01_continuous-batch-discrete.md`](01_continuous-batch-discrete.md) — The three production modes and state-task network formulations.
- **Sibling Topic 02:** [`02_yield-mass-balance.md`](02_yield-mass-balance.md) — Mass balance reconciliation, split fractions, and co-product accounting.
- **Next Topic:** [`05_bulk-storage-inventory.md`](05_bulk-storage-inventory.md) — Tank farms, silos, segregation, heel dynamics, and inventory physics.
- **Downstream Optimization:** [`08_capacity-debottlenecking.md`](08_capacity-debottlenecking.md) — Asset capacity, downtime mitigation, and constraint economics.
- **Root Survey Map:** [`../../../course/38_process_industries_commodities.md`](../../../course/38_process_industries_commodities.md)
- **Practical Application Lab:** [`../../../course/38_process_industries_commodities.md#7-labs`](../../../course/38_process_industries_commodities.md#7-labs) — Lab 3: Campaign and Changeover Optimization under Storage Constraints.
