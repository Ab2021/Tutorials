# Mine Planning and Cut-Off Grade Optimization

> Module 39 — Mining, Metals, and Forestry Supply Chains
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 16 (Supply Chain Network Economics), Module 20 (Operations Planning and Master Scheduling), Module 39 Topic 01 (Reserves, Grade, and Confidence)
> Version 2026.5

---

## 1. Why this topic exists

In mineral supply chains, raw material extraction is physically irreversible and capital-intensive. Unlike discrete manufacturing assembly lines where bills of materials are deterministic and component quality is standardized, an open-pit or underground mine exploits a spatially fixed, heterogeneous orebody. The spatial distribution of rock cannot be relocated; its chemical grade, mineralogy, structural hardness, and deleterious contaminants vary continuously throughout three dimensions.

Crucially, **ore is not a geological fact; it is an economic classification**. Every cubic meter of rock excavated from an open pit must be assigned to one of three destination streams:
1. **Direct Processing (Concentrator/Heap Leach):** High-grade or standard-grade material whose revenue exceeds processing, refining, and opportunity costs.
2. **Strategic Stockpiling:** Marginal or sub-economic material whose revenue covers processing costs but whose immediate treatment would displace higher-margin feed during periods of milling capacity constraints.
3. **Waste Dump:** Rock whose recoverable metal revenue fails to cover incremental milling and realization expenses, or rock that must be extracted solely to satisfy geotechnical wall stability constraints.

Assigning this classification incorrectly destroys enterprise value at an alarming scale:
- If the cut-off grade is set too low (over-mining), the concentrator is flooded with low-grade material. Hard, low-yield rock consumes scarce comminution energy (crushing and grinding mills), displaces higher-grade ore, and generates voluminous tailings without generating sufficient cash flow to cover the time value of invested capital.
- If the cut-off grade is set too high (under-mining or unmanaged high-grading), mine life is artificially truncated, sunk capital in concentrator infrastructure is amortized over fewer total ounces or tonnes of metal, unit overhead costs soar, and marginal ore is dumped irreversibly onto waste dumps where it oxidizes or is permanently buried.
- If pit sequencing ignores stripping ratio dynamics and pushback staging, a mine faces either crippling front-loaded waste pre-stripping capital expenditures or, conversely, sudden ore starvation when mining phases bottleneck and shovel fleets cannot access deep benches.

For decades, mineral planning relied on static break-even rules dating to the 19th century, calculating cut-off grade merely as the cost of processing divided by net metal price. This naive approach ignores processing capacity bottlenecks, refinery quotas, and the fundamental opportunity cost of time. Modern resource economics relies on the synthesis of two foundational algorithmic frameworks:
- **The Lerchs-Grossmann (LG) 3D Maximum Closure Algorithm**, which determines the Ultimate Pit Limit (UPL) and optimal nested pushback phases on directed block graphs.
- **Kenneth F. Lane's Dynamic Cut-Off Grade Theory**, which optimizes cut-off grades over time by simultaneously balancing mining, concentrator, and refining capacity constraints against the financial opportunity cost of delaying future reserves.

Mastering these algorithms, their mathematical duality, and their operational dynamics is indispensable for resource supply chain analysts, mine planning engineers, and bulk commodity strategists.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Formulate** economic block values ($V_b$) from geological block models by incorporating grade, metallurgical recovery, commodity prices, smelting and refining deductions, and differential mining costs.
2. **Construct** the directed block precedence network and prove the equivalence between the 3D Ultimate Pit Limit (UPL) problem and the Maximum Weight Closure / Minimum Cut problem on bipartite graphs.
3. **Generate** nested pit shells using revenue factor parameterization ($\lambda$) and evaluate pushback sequencing against operational Minimum Mining Width (MMW) constraints.
4. **Quantify** stripping ratio dynamics by calculating instantaneous ($SR_t$), cumulative ($SR_{\text{cum}}$), and break-even ($BESR$) stripping ratios across different waste pre-stripping schedules.
5. **Derive** Kenneth F. Lane's three limiting cut-off grades ($g_m, g_c, g_r$) corresponding to mining, concentrator, and refinery capacity bottlenecks.
6. **Calculate** Lane's three balancing cut-off grades ($g_{mc}, g_{cr}, g_{mr}$) and determine the global profit-maximizing cut-off grade $g^*$ incorporating the opportunity cost of time ($d = \delta W$).
7. **Design** a life-of-mine (LOM) dynamic declining cut-off grade schedule coupled with strategic low-grade stockpile reclamation to maximize Net Present Value (NPV).
8. **Critique** mine schedules against international reserve disclosure standards (JORC 2012, NI 43-101, SEC S-K 1300) and the Global Industry Standard on Tailings Management (GISTM).

---

## 3. The core mental model

Open-pit mine planning is an integrated geometric and economic optimization loop. The physical mine consists of discrete three-dimensional blocks constrained by geotechnical slope physics. The mineral processing supply chain consists of three sequential, capacity-constrained transformation stages: Mining ($M$), Concentrating ($C$), and Refining/Marketing ($R$).

```text
=============================================================================================================
                      INTEGRATED MINE PLANNING AND CUT-OFF GRADE ARCHITECTURE
=============================================================================================================

  GEOLOGICAL BLOCK MODEL
  [3D Array of Blocks: x, y, z, Grade (g), Density (ρ), Lithology, Geotech Zone]
                         │
                         ▼
  ECONOMIC BLOCK VALUATION
  V_b = max( V_ore - V_waste, 0 )  -->  Weight assignment w_u
                         │
                         ▼
  PIT SHELL OPTIMIZATION (LERCHS-GROSSMANN / PICARD MAX-CLOSURE)
  Directed Graph G = (V, A) with slope precedence arcs (u, v)
  Parametric Revenue Factors: λ_1 < λ_2 < ... < λ_K  -->  Nested Pit Shells
                         │
                         ▼
  PUSHBACK STAGING & BENCH SEQUENCING
  Incorporate Minimum Mining Width (MMW), haul road ramps (8-10% grade), and bench sink rates
  Stripping Ratio Profile: Declining vs. Constant vs. Increasing (JIT)
                         │
                         ▼
  LANE'S DYNAMIC CUT-OFF GRADE OPTIMIZATION (Life-of-Mine / Annual Loop)
  Opportunity Cost of Capital: d_t = δ · W_t  (where W_t = remaining deposit NPV)
  Fixed Costs: F_t = f + d_t
       │
       ├─────────────────────────────────┬─────────────────────────────────┐
       ▼                                 ▼                                 ▼
  MINING BOTTLENECK           CONCENTRATOR BOTTLENECK             REFINERY BOTTLENECK
  Capacity M (t rock/yr)      Capacity C (t ore/yr)               Capacity R (t metal/yr)
  g_m = c / [(s - r)·y]       g_c = [c + (F/C)] / [(s - r)·y]     g_r = c / [(s - r - (F/R))·y]
       │                                 │                                 │
       └─────────────────────────────────┼─────────────────────────────────┘
                                         ▼
                     BALANCING CUT-OFF GRADES: g_mc, g_cr, g_mr
                                         │
                                         ▼
                     OPTIMAL EFFECTIVE CUT-OFF GRADE: g*_t
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 ▼                                               ▼
         Ore: Grade ≥ g*_t                                Waste: Grade < g*_t
                 │                                               │
      ┌──────────┴──────────┐                                    ▼
      ▼                     ▼                             WASTE DUMPS
  HIGH-GRADE FEED    LOW-GRADE ORE STOCKPILE             Non-acid forming (NAF) vs
  (Direct to Mill)   (Stored for future LOM reclaim)     Potentially acid generating (PAG)
      │                     │
      ▼                     ▼
  CONCENTRATOR / BENEFICIATION PLANT (Flotation / Leaching / Magnetic Separation)
      │
      ▼
  CONCENTRATE LOGISTICS (Pipeline / Rail / Bulk Bulk-Haul)
      │
      ▼
  SMELTER / REFINERY / OFFTAKE (Final London Metal Exchange / COMEX cathode or bullion)

=============================================================================================================
```

The system operates across three coupled planning horizons:

| Horizon | Time Scale | Key Decision Variables | Governing Mathematical Framework |
|---|---|---|---|
| **Strategic (LOM)** | 10–30 Years | Ultimate pit limit, processing plant capacity sizing, dynamic cut-off trajectory. | Lerchs-Grossmann max closure, Lane dynamic programming. |
| **Tactical (Medium-Term)** | 1–3 Years | Pushback staging, stripping ratio targets, stockpile footprint, equipment capital. | Mixed-integer linear programming (MILP), nested shell parametrization. |
| **Operational (Short-Term)** | Daily–Monthly | Blast muckpile polygon delineation (diglines), truck-shovel dispatch, crusher blending. | Spatial grade control kriging, non-linear blending heuristics. |

The operational feedback loop connects the physical state of the mine with financial asset valuation:
- The spatial pushback schedule determines which benches are active and sets the instantaneous stripping ratio ($SR_t$).
- The active bench grade distribution feeds into Lane's equations, which determine the active bottleneck among mining, comminution, and smelting.
- The resulting operating cash flow updates the remaining project net present value ($W_{t+1}$), which recalibrates the opportunity cost ($d_{t+1} = \delta W_{t+1}$) for the subsequent planning period.

---

## 4. Definitions and vocabulary

| Term | Definition | Misuse / Common Pitfall | Unit / Standard |
|---|---|---|---|
| **Geological Block Model** | A 3D spatial discretization of an orebody into regular orthogonal cells containing rock mass, grade, density, and metallurgy. | Confusing interpolated block grades (kriged) with actual blasted muckpile grades. | Length (m), Mass (t), Grade (%, g/t) |
| **Economic Block Value ($V_b$)** | The net profit generated by extracting, processing, and selling a block, minus all mining and processing costs. | Ignoring differential waste mining costs ($m_o \neq m_w$) or environmental rehabilitation liabilities. | USD ($) per block |
| **Ultimate Pit Limit (UPL)** | The boundary of the open pit that maximizes undiscounted total operating profit while obeying slope constraints. | Assuming UPL represents an actionable extraction schedule; it lacks temporal staging. | Spatial polygon / solid (m³) |
| **Lerchs-Grossmann (LG)** | A graph-theoretic algorithm that finds the maximum-weight closure on a directed graph representing mining blocks and slope dependencies. | Believing LG maximizes NPV; LG strictly maximizes undiscounted cash flow. | Algorithmic method (1965) |
| **Maximum Closure Problem** | In graph theory, finding a subset of vertices $C \subseteq V$ of maximal weight such that all outgoing arcs from $C$ terminate in $C$. | Conflating maximum closure with shortest path or minimum spanning tree problems. | Graph theory ($G = (V, A)$) |
| **Precedence Constraint** | The physical requirement that overlying blocks must be removed before an underlying block can be excavated without slope failure. | Modeling slope angles uniformly in all directions when geological faulting imposes anisotropic limits. | Degrees (°), search patterns |
| **Pushback (Phase)** | A practical, operational mining slice designed inside the UPL that provides an operable working width for mining equipment. | Designing pushbacks narrower than the Minimum Mining Width (MMW) of shovels and haul trucks. | Horizontal width (m), bench count |
| **Revenue Factor ($\lambda$)** | A scaling multiplier applied to commodity price ($P' = \lambda \cdot P$) to generate a sequence of nested pit shells. | Treating $\lambda$ as a real-world price forecast rather than a parameter to identify high-margin cores. | Dimensionless multiplier |
| **Nested Pit Shells** | A series of concentric pit shells generated by incrementing $\lambda$ from low to high values ($0.3 \le \lambda \le 1.5$). | Scheduling pit shells directly without smoothing into engineered road-access phases. | Shell index ($k \in \{1, \dots, K\}$) |
| **Instantaneous Stripping Ratio ($SR_t$)** | The ratio of waste tonnes mined to ore tonnes mined within a specific operating period $t$. | Confusing periodic stripping ratio with the life-of-mine average ratio. | Tonnes waste / tonnes ore ($t_w / t_o$) |
| **Break-Even Stripping Ratio ($BESR$)** | The maximum allowable stripping ratio at which the profit of mining ore open-pit equals zero (or equals underground mining margin). | Using unrecovered metal grade or gross price instead of net recovered value minus processing costs. | Dimensionless ($t_w / t_o$) |
| **Break-Even Cut-Off ($g_{be}$)** | The grade where net revenue from processing a mined block exactly equals the incremental cost of processing and marketing. | Applying break-even cut-off when the processing plant is the operational bottleneck. | Mass fraction (% or g/t) |
| **Internal / Marginal Cut-Off ($g_{\text{marg}}$)** | The cut-off grade applied to rock already blasted and at the pit floor, where mining cost is sunk and only milling cost is tested. | Counting sunk drill-and-blast expenditures in the marginal diversion decision. | Mass fraction (% or g/t) |
| **Kenneth Lane's Theory** | An analytical optimization method that derives cut-off grades to maximize enterprise NPV under capacity and opportunity cost limits. | Treating cut-off grade as a static parameter invariant over the life of the mine. | Economic theory (1988) |
| **Opportunity Cost ($d = \delta W$)** | The cost of capital incurred by delaying the realization of all remaining future deposit reserves by one time period. | Neglecting $d$, which understates the economic penalty of processing low-grade rock in early years. | USD per year ($/yr) |
| **Mining-Limited Cut-Off ($g_m$)** | The cut-off grade that maximizes cash flow per unit time when the mining fleet is fully utilized and the mill is starved. | Assuming $g_m$ changes with the discount rate; it depends purely on operating costs and recovery. | Mass fraction (% or g/t) |
| **Concentrator-Limited Cut-Off ($g_c$)** | The cut-off grade when the processing plant is at capacity, balancing processing costs and corporate opportunity costs. | Setting mill feed based on mine extraction rate rather than milling bottleneck economics. | Mass fraction (% or g/t) |
| **Refinery-Limited Cut-Off ($g_r$)** | The cut-off grade when downstream metal smelting, refining, or marketing contracts impose the binding throughput constraint. | Using $g_r$ when downstream refinery quotas are unbounded. | Mass fraction (% or g/t) |
| **Balancing Cut-Off ($g_{ij}$)** | The unique cut-off grade at which the throughput of stage $i$ exactly matches the throughput capacity of stage $j$. | Forgetting that balancing grades depend on the empirical grade-tonnage distribution. | Mass fraction (% or g/t) |
| **Dynamic Declining Cut-Off** | A policy of feeding high cut-off grades in early years and progressively lowering cut-offs as remaining mine life shrinks. | Failing to allocate sufficient CAPEX or land for low-grade stockpile footprint and rehandle. | Schedule over time ($g^*_t$) |
| **Minimum Mining Width (MMW)** | The minimum horizontal distance required for a primary loading shovel and two haul trucks to operate safely. | Designing theoretical pit shells that cannot accommodate physical machinery turning radiuses. | Meters (m) |
| **Bench Sink Rate** | The maximum vertical depth (number of benches) that can be excavated in a pushback per calendar year. | Over-scheduling production beyond physical drilling, blasting, and vertical dewatering limits. | Benches per year (b/yr) or m/yr |
| **Net Smelter Return (NSR)** | The net revenue realized per tonne of ore or concentrate after deducting smelting charges, refining deductions, and penalties. | Calculating revenue directly from metal quotation without deducting deductions and treatment charges. | USD per tonne ($/t) |
| **Ore Loss and Dilution** | Ore loss is valuable mineral misplaced to waste; dilution is barren waste excavated and processed as ore. | Assuming block model boundaries represent knife-edge selective mining cuts in blast muckpiles. | Percentage (%) |
| **Geometallurgy** | The integration of geological, geochemical, and mineralogical data into spatial models to predict plant throughput and recovery. | Assuming constant metallurgical recovery across varying mineralogical alteration zones. | Domain-specific model |
| **Comminution Energy** | The electrical energy consumed by crushing and grinding circuits (kWh/t) to reduce rock size to mineral liberation dimensions. | Assuming fixed comminution cost per tonne regardless of rock Bond Work Index ($BWI$). | Kilowatt-hours per tonne ($kWh/t$) |
| **Digline Delineation** | The physical surveyed contour flagged on blasted muckpiles separating ore from waste for shovel operators. | Assuming perfect excavator selectivity along diglines without bucket edge carry-over. | Spatial polygon boundary |
| **In-Pit Crushing & Conveying (IPCC)** | Sizing rock inside the pit and transporting it via continuous conveyors rather than diesel haul trucks. | Ignoring the inflexibility of fixed conveyor corridors during active pushback wall retreats. | Technology classification |

---

## 5. The mechanics, step by step

Mine planning and cut-off grade optimization execute across five distinct mathematical and operational phases:

```text
+---------------------------------------------------------------------------------------------------+
| Phase 1: Block Valuation and Geotechnical Precedence Graph Formulation                             |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| Phase 2: Ultimate Pit Limit Optimization via Lerchs-Grossmann and Max-Flow Min-Cut                 |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| Phase 3: Pushback Staging and Nested Pit Shell Generation                                         |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| Phase 4: Stripping Ratio Dynamics and Waste Pre-Stripping Schedules                               |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
+---------------------------------------------------------------------------------------------------+
| Phase 5: Kenneth F. Lane's Optimal Cut-Off Grade Determination                                    |
+---------------------------------------------------------------------------------------------------+
```

### Phase 1: Block Valuation and Geotechnical Precedence Graph Formulation

Every block $b$ in the block model with rock mass $M_b$ (tonnes), mineral grade $g_b$ (mass fraction), metallurgical recovery $R(g_b)$, commodity price $P$ ($/tonne metal), refining/selling cost $r$ ($/tonne metal), processing cost $c$ ($/tonne rock), and mining cost $m$ ($/tonne rock) has two mutually exclusive operational states:

1. **If processed as ore:**
   $$V_{b, \text{ore}} = M_b \cdot \left[ (P - r) \cdot g_b \cdot R(g_b) - c - m_o \right]$$
2. **If excavated as waste:**
   $$V_{b, \text{waste}} = - M_b \cdot m_w$$
   where $m_o$ is the unit mining cost of ore and $m_w$ is the unit mining cost of waste.

The decision to treat a mined block as ore rather than waste is governed by its marginal benefit:
$$\Delta V_b = V_{b, \text{ore}} - V_{b, \text{waste}} = M_b \cdot \left[ (P - r) \cdot g_b \cdot R(g_b) - c - (m_o - m_w) \right]$$

If $\Delta V_b > 0$, the block yields higher cash flow as ore; otherwise, it is waste. The economic weight $w_b$ assigned to the block for the ultimate pit limit calculation is:
$$w_b = \max\left( V_{b, \text{ore}}, V_{b, \text{waste}} \right)$$

For polymetallic deposits containing multiple payable metals (e.g., Copper $g_{\text{Cu}}$, Gold $g_{\text{Au}}$, Molybdenum $g_{\text{Mo}}$) along with penalty elements (e.g., Arsenic $g_{\text{As}}$), economic block valuation uses **Net Smelter Return (NSR)**:
$$\text{NSR}_b = \sum_{k} \left[ P_k \cdot (1 - \text{deduction}_k) - r_k \right] \cdot g_{b,k} \cdot R_k(g_{b,k}) - \sum_{j} \text{Penalty}_j(g_{b,j})$$
$$V_{b, \text{ore}} = M_b \cdot \left[ \text{NSR}_b - c - m_o \right]$$

#### Spatial Support and the Information Effect
In long-term planning, block values are assigned to large planning blocks $V$ (e.g., $20\text{ m} \times 20\text{ m} \times 15\text{ m}$). In daily operations, mining selectivity is executed on smaller Selective Mining Units (SMU) $v$ (e.g., $5\text{ m} \times 5\text{ m} \times 15\text{ m}$). By Krige's dispersion variance relation:
$$\sigma^2(v \mid D) = \sigma^2(V \mid D) + \sigma^2(v \mid V)$$
Because $\sigma^2(v \mid D) > \sigma^2(V \mid D)$, large kriged blocks smooth out local grade extremes. Applying an operational cut-off grade directly to smoothed planning blocks artificially underestimates metal recovered above high cut-offs and overestimates metal above low cut-offs. Mine planners correct for this information effect using lognormal affine volume-variance transformations before running Lerchs-Grossmann optimizations.

#### Geotechnical Precedence Graph
To extract block $u$ at depth level $z$, all overlying blocks within a geotechnical inverted cone defined by safe wall slope angle $\theta$ (typically $38^\circ$ to $55^\circ$) must be removed. 

We construct a directed graph $G = (V, A)$, where each block $u$ is a vertex $u \in V$ with weight $w_u$. A directed arc $(u, v) \in A$ exists if extracting block $u$ requires the prior extraction of block $v$ (i.e., $v \in \text{Pred}(u)$). In a regular 3D grid, slope angles are approximated using search patterns (e.g., 1-5-9 or 1-9-25 block templates) mapping vertical bench heights to horizontal block offsets.

---

### Phase 2: Ultimate Pit Limit Optimization via Lerchs-Grossmann and Max-Flow Min-Cut

The Ultimate Pit Limit (UPL) identifies the subset of blocks $C \subseteq V$ that maximizes total undiscounted cash flow without violating slope precedence constraints. 

A subset $C \subseteq V$ is a **closure** of directed graph $G = (V, A)$ if:
$$u \in C \quad \text{and} \quad (u, v) \in A \implies v \in C$$

The UPL is the **Maximum-Weight Closure Problem**:
$$\max_{C \subseteq V} \sum_{u \in C} w_u \quad \text{subject to } (u \in C \wedge (u, v) \in A) \implies v \in C$$

#### The Original Lerchs-Grossmann Graph Tree Algorithm (1965)
The classic 1965 Lerchs-Grossmann formulation constructs a directed rooted tree $T = (V \cup \{u_0\}, E)$, where $u_0$ is an artificial root node representing the surface:
1. Every tree branch connected to the root is classified as either **strong** or **weak**:
   - A tree branch $T_x$ rooted at node $x$ is **strong** if the cumulative weight of all nodes in the sub-tree is strictly positive: $W(T_x) = \sum_{v \in T_x} w_v > 0$.
   - A tree branch $T_x$ is **weak** if its cumulative sub-tree weight is non-positive: $W(T_x) \le 0$.
2. The algorithm performs systematic tree normalizations:
   - If an edge connects a strong branch to a weak branch, it attempts to pivot tree edges.
   - A tree is **normalized** when all branches directly attached to root $u_0$ are strong, and within every strong branch, all sub-branches have non-positive weight.
   - The set of nodes contained in all strong branches attached to the root in a normalized tree constitutes the exact maximum closure.

#### Picard's Max-Flow / Min-Cut Equivalence
Jean-Claude Picard (1976) proved that the Maximum-Weight Closure Problem is mathematically equivalent to finding the Minimum Cut in an augmented bipartite-like network $G' = (V \cup \{s, t\}, A')$. This breakthrough allowed modern operations research to replace the original tree-pivoting heuristic with high-performance network max-flow solvers:

1. Introduce a fictitious source node $s$ and a sink node $t$.
2. For every block $u \in V$ with positive weight ($w_u > 0$):
   - Add a directed arc $(s, u)$ with capacity:
     $$c(s, u) = w_u$$
3. For every block $v \in V$ with negative weight ($w_v < 0$):
   - Add a directed arc $(v, t)$ with capacity:
     $$c(v, t) = -w_v$$
4. For every geological precedence constraint $(u, v) \in A$ (meaning $u$ requires $v$):
   - Add a directed arc $(u, v)$ with infinite capacity:
     $$c(u, v) = \infty$$

Let $(S, T)$ be an $(s, t)$-cut partitioning the node set $V \cup \{s, t\}$ such that $s \in S$ and $t \in T$. The capacity of the cut is:
$$\text{Cap}(S, T) = \sum_{u \in S, v \in T} c(u, v)$$

If the cut cuts any infinite-capacity precedence arc $(u, v)$, $\text{Cap}(S, T) = \infty$. Thus, any finite minimum cut strictly enforces geotechnical precedence ($u \in S \implies v \in S$). For a valid closure $C = S \setminus \{s\}$:
$$\text{Cap}(S, T) = \sum_{u \in T, w_u > 0} w_u + \sum_{v \in S, w_v < 0} (-w_v)$$

Adding and subtracting $\sum_{u \in S, w_u > 0} w_u$:
$$\text{Cap}(S, T) = \sum_{u \in V, w_u > 0} w_u - \sum_{u \in C} w_u$$

Since $\sum_{u \in V, w_u > 0} w_u$ is a fixed constant, **minimizing cut capacity $\text{Cap}(S, T)$ strictly maximizes closure weight $\sum_{u \in C} w_u$**. The optimal set of blocks to extract is precisely $C^* = S^* \setminus \{s\}$, solvable in polynomial time via the push-relabel algorithm or the Hochbaum's Pseudoflow (HPF) algorithm.

---

### Phase 3: Pushback Staging and Nested Pit Shell Generation

Extracting the entire ultimate pit limit in a single bench-by-bench operation across the entire deposit surface (worst-case scheduling) delays access to deep, high-grade ore, producing low or negative early cash flows and a depressed Net Present Value (NPV). 

To maximize NPV, mining is phased into **nested pushbacks (phases)**:
1. Scale the base commodity price $P$ by a revenue factor $\lambda \in [\lambda_{\min}, \lambda_{\max}]$, where typically $0.3 \le \lambda \le 1.5$:
   $$w_u(\lambda) = \max\left( M_u [ (\lambda P - r) g_u R - c - m_o ], -M_u m_w \right)$$
2. Solve the maximum closure problem for an ascending sequence of factors $\lambda_1 < \lambda_2 < \dots < \lambda_K = 1.0$.
3. Because higher revenue factors monotonically relax the threshold for extracting waste to reach ore, the resulting pit shells form a strictly nested sequence:
   $$C(\lambda_1) \subseteq C(\lambda_2) \subseteq \dots \subseteq C(\lambda_K)$$

```text
  Bench Level
    0 ───┬───────────────────────────────────────────┬─── Surface
       ┌─┴─────────────────────────────────────────┴─┐
   -50 │          Pushback 1 (λ = 0.5, High Grade)    │
       ├──────────────┬─────────────────┬────────────┤
  -100 │              │ Pushback 2      │            │
       ├──────────────┤ (λ = 0.8)       ├────────────┤
  -150 │              │                 │            │
       ├──────────────┼─────────────────┼────────────┤
  -200 │              │ Pushback 3      │            │
       └──────────────┤ (λ = 1.0, UPL)  ├────────────┘
                      └─────────────────┘
```

#### The Monotonicity of Parametric Minimum Cuts
The strict nesting property $C(\lambda_1) \subseteq C(\lambda_2)$ follows from the monotonicity of parametric network flows (Picard & Ratliff, 1975). Let $w_u(\lambda) = \lambda \cdot R_u - K_u$. For any two multipliers $\lambda_1 < \lambda_2$:
$$\Delta w_u = w_u(\lambda_2) - w_u(\lambda_1) = (\lambda_2 - \lambda_1) \cdot R_u \ge 0$$
Since every block's economic valuation non-decreases as $\lambda$ increases, the marginal benefit of adding overlying waste blocks to access deep positive blocks increases monotonically. By submodularity of cut capacity functions, the source set $S^*(\lambda)$ expands monotonically with $\lambda$, guaranteeing that pushback boundaries never cross or invert in three dimensions.

#### Best-Case vs. Worst-Case DCF Sequencing Gap
Jeff Whittle established the classic bounds for evaluating nested pit shells:
- **Best-Case Schedule:** Pushback 1 is mined out completely from top to bottom before Pushback 2 commences, and so forth. Cash flows are aggressively front-loaded:
  $$\text{NPV}_{\text{best}} = \sum_{k=1}^K \sum_{t \in T_k} \frac{CF_t}{(1 + \delta)^t}$$
- **Worst-Case Schedule:** The entire deposit envelope defined by the ultimate shell $K$ is mined flat, bench-by-bench horizontally across all phases simultaneously. Waste stripping is heavily front-loaded, and ore delivery is deferred:
  $$\text{NPV}_{\text{worst}} = \sum_{b=1}^B \frac{CF_b}{(1 + \delta)^{t_b}}$$
- The difference $\Delta \text{NPV}_{\text{gap}} = \text{NPV}_{\text{best}} - \text{NPV}_{\text{worst}}$ represents the theoretical maximum financial prize accessible through phase optimization. Realistic operational schedules (the "Milawa" algorithm) lie between these bounds.

#### Pushback Design Constraints
Raw nested shells cannot be mined directly. Mining engineers smooth them into engineered pushbacks that respect:
- **Minimum Mining Width (MMW):** A physical operating floor width (typically 40–80 m) allowing two-way 240-tonne haul truck traffic and shovel swing clearances:
  $$MMW \ge 2 \cdot W_{\text{truck}} + W_{\text{shovel}} + 2 \cdot C_{\text{clearance}}$$
- **Ramp Haulage Design:** Continuous haul roads with 8% to 10% road gradient and 25–35 m road width including safety berms.
- **Vertical Sink Rate:** Geotechnical limit on vertical bench advance (typically no more than 6 to 10 benches per year per pushback).

---

### Phase 4: Stripping Ratio Dynamics and Waste Pre-Stripping Schedules

The **Instantaneous Stripping Ratio ($SR_t$)** in production period $t$ is the ratio of waste mass excavated to ore mass delivered:
$$SR_t = \frac{W_t}{O_t} = \frac{\text{Waste Tonnes Mined in Year } t}{\text{Ore Tonnes Delivered to Mill in Year } t}$$

The **Cumulative Stripping Ratio ($SR_{\text{cum}}$)** over the total life of mine (LOM) is:
$$SR_{\text{cum}} = \frac{\sum_{t=1}^T W_t}{\sum_{t=1}^T O_t}$$

The economic upper bound is the **Break-Even Stripping Ratio ($BESR$)**, beyond which open-pit extraction destroys more value than it creates:
$$BESR = \frac{(P - r) \cdot g \cdot R - c - m_o}{m_w}$$

```text
Stripping Ratio (W:O)
    │
    │  \   Declining Stripping Ratio (High early CAPEX, low operational risk)
    │   \
    │────\────────────────── Constant Stripping Ratio (Balanced fleet, stockpile buffer)
    │     \
    │      \_________        Increasing / Just-in-Time Stripping (High risk, max early NPV)
    │                \
    └─────────────────┴─────────────────────► Time (Years)
```

#### Comparison of Operational Stripping Ratio Policies

| Operational Metric | Declining Stripping Ratio | Constant Stripping Ratio | Increasing / JIT Stripping |
|---|---|---|---|
| **Fleet Sizing ($N_{\text{trucks}}$)** | Peak fleet required in Years 1–3; excess trucks idled in later life. | Constant fleet maintained across entire project lifecycle. | Small initial fleet; massive capital re-investment required mid-life. |
| **Working Bench Width** | Maximum width available ($>100\text{ m}$); optimal shovel swing efficiency. | Stable operating widths ($60\text{--}80\text{ m}$); balanced traffic. | Pinched benches ($40\text{--}50\text{ m}$); frequent traffic bottlenecks. |
| **Early Capital Expenditure** | Extremely heavy front-loaded waste pre-stripping cash sink. | Predictable, smoothed annual operational expenditures. | Deferred to maximum extent permissible by slope geometry. |
| **Discounted Enterprise NPV** | Sub-optimal; heavy initial cash outlays severely penalize discounted cash flows. | Balanced; provides robust compromise between DCF and execution risk. | Highest theoretical NPV if zero geotechnical or operational delays occur. |
| **Operational & Wall Risk** | Negligible; deep benches are uncovered years in advance of mill feed. | Moderate; buffer stockpiles absorb minor blast delays or wall slumps. | Extreme; localized wall failures or shovel breakdowns cause immediate mill starvation. |

---

### Phase 5: Kenneth F. Lane's Optimal Cut-Off Grade Theory

Static break-even cut-off grade calculations assume infinite time and unconstrained processing capacity:
$$g_{be} = \frac{c}{(P - r) \cdot R}$$

Kenneth F. Lane (1988) proved that when extraction involves sequential, capacity-constrained facilities operating under a discount rate $\delta$, **cut-off grade must maximize the Net Present Value of the remaining deposit reserves**.

#### 1. Mathematical Formulation of the Objective Function
Consider a mineral operation defined by three capacity limits:
- Mining stage capacity: $M$ (tonnes of total rock per year).
- Concentrator stage capacity: $C$ (tonnes of ore processed per year).
- Smelter / Refinery stage capacity: $R_{\text{ref}}$ (tonnes of pure metal produced per year).

Financial parameters:
- $s$: gross selling price per unit of refined metal ($/t metal).
- $r$: realization and refining cost per unit of refined metal ($/t metal).
- $c$: unit processing cost per tonne of ore ($/t ore).
- $m$: unit mining cost per tonne of rock ($/t rock).
- $f$: fixed corporate and site overhead cost per unit time ($/year).
- $W$: present value of all remaining unmined orebody reserves ($).
- $\delta$: corporate discount rate (%/year, continuous).
- $d = \delta \cdot W$: **Opportunity cost of delaying remaining reserves** by one unit of time ($/year).

Total time cost rate:
$$F = f + d = f + \delta W$$

Let $Q_m$ be the total material mined. For an arbitrary cut-off grade $g$:
- $p(u)$ is the probability density function of mineral grade $u$ in the mined material.
- $x(g) = \int_g^\infty p(u) \, du$ is the mass proportion of mined rock classified as ore.
- $\bar{g}(g) = \frac{1}{x(g)} \int_g^\infty u \cdot p(u) \, du$ is the average grade of the selected ore.
- $y$ is the metallurgical recovery fraction.
- $k(g) = \bar{g}(g) \cdot y$ is the finished metal yield per tonne of ore.

Per tonne of total rock mined ($Q_m = 1$):
- Ore tonnes milled: $x(g)$
- Metal tonnes produced: $x(g) \cdot k(g)$

The net profit per unit of mined material is:
$$P(g) = (s - r) \cdot x(g) \cdot k(g) - c \cdot x(g) - m$$

The time $T(g)$ required to process this unit of mined material depends on which stage is binding:
$$T(g) = \max\left( \frac{1}{M}, \frac{x(g)}{C}, \frac{x(g) \cdot k(g)}{R_{\text{ref}}} \right)$$

Lane's objective is to choose $g$ to maximize the present value increment:
$$\max_g V(g) = \max_g \left[ P(g) - (f + \delta W) \cdot T(g) \right] = \max_g \left[ P(g) - F \cdot T(g) \right]$$

---

#### 2. Derivation of the Three Limiting Cut-Off Grades

##### Case A: Mining Stage is the Sole Bottleneck ($T = 1/M$)
Here, the concentrator and refinery have excess capacity. The time to mine one unit of material is fixed at $T = 1/M$, independent of cut-off grade $g$.
$$V_m(g) = (s - r) \cdot x(g) \cdot k(g) - c \cdot x(g) - m - \frac{F}{M}$$

To find the cut-off grade $g_m$ that maximizes $V_m(g)$, take the derivative with respect to $g$ and set to zero. Note that:
$$\frac{d}{dg} x(g) = -p(g), \quad \frac{d}{dg} \left[ x(g) \bar{g}(g) \right] = -g \cdot p(g)$$

Therefore:
$$\frac{d V_m}{dg} = (s - r) \cdot (-g \cdot p(g) \cdot y) - c \cdot (-p(g)) = 0$$
$$- (s - r) \cdot g \cdot y + c = 0$$

$$g_m = \frac{c}{(s - r) \cdot y}$$

*Crucial Insight:* The mining-limited cut-off $g_m$ equals the classical marginal break-even cut-off. Because processing capacity is unconstrained, mining extra material takes no additional time at the plant, so the opportunity cost $F$ drops out of the marginal grade equation.

---

##### Case B: Concentrator Stage is the Sole Bottleneck ($T = x(g) / C$)
Here, the mine can supply rock faster than the mill can grind it. Time required per unit mined rock is $T = x(g) / C$.
$$V_c(g) = (s - r) \cdot x(g) \cdot k(g) - c \cdot x(g) - m - F \cdot \frac{x(g)}{C}$$
$$V_c(g) = (s - r) \cdot x(g) \cdot k(g) - \left( c + \frac{F}{C} \right) \cdot x(g) - m$$

Differentiating with respect to $g$ and setting to zero:
$$\frac{d V_c}{dg} = -(s - r) \cdot g \cdot p(g) \cdot y + \left( c + \frac{F}{C} \right) \cdot p(g) = 0$$

$$g_c = \frac{c + \frac{F}{C}}{(s - r) \cdot y} = \frac{c + \frac{f + \delta W}{C}}{(s - r) \cdot y}$$

*Crucial Insight:* When the mill is the bottleneck, the effective cost of treating a tonne of ore is not just the operating milling cost $c$, but $c$ **plus the fixed overhead and opportunity cost per unit of concentrator capacity** ($F/C$). Because $d = \delta W$ is highest in the early years of a mine's life, $g_c$ is significantly elevated during early operations, rejecting marginal ore to mill only high-margin rock.

---

##### Case C: Smelter / Refinery Stage is the Sole Bottleneck ($T = x(g) k(g) / R_{\text{ref}}$)
Here, the downstream refinery, leaching circuit, or offtake contract cannot absorb more metal. Time required per unit of mined rock is $T = x(g) k(g) / R_{\text{ref}}$.
$$V_r(g) = (s - r) \cdot x(g) \cdot k(g) - c \cdot x(g) - m - F \cdot \frac{x(g) \cdot k(g)}{R_{\text{ref}}}$$
$$V_r(g) = \left( s - r - \frac{F}{R_{\text{ref}}} \right) \cdot x(g) \cdot k(g) - c \cdot x(g) - m$$

Differentiating with respect to $g$ and setting to zero:
$$\frac{d V_r}{dg} = -\left( s - r - \frac{F}{R_{\text{ref}}} \right) \cdot g \cdot p(g) \cdot y + c \cdot p(g) = 0$$

$$g_r = \frac{c}{\left( s - r - \frac{F}{R_{\text{ref}}} \right) \cdot y} = \frac{c}{\left( s - r - \frac{f + \delta W}{R_{\text{ref}}} \right) \cdot y}$$

*Crucial Insight:* When downstream marketing or refining is constrained, time cost is charged per unit of finished metal ($F/R_{\text{ref}}$), effectively reducing net metal price and shifting the cut-off grade upward.

---

#### 3. Derivation of the Three Balancing Cut-Off Grades

The limiting cut-offs $g_m, g_c, g_r$ assume that exactly one component is at capacity while the other two have infinite slack. In practice, stages interact. The **balancing cut-off grades** define the exact cut-off where two sequential stages are simultaneously operating at 100% capacity:

1. **Mining-Concentrator Balancing Cut-Off ($g_{mc}$):**
   Mined ore rate matches mill capacity:
   $$x(g_{mc}) \cdot M = C \implies x(g_{mc}) = \frac{C}{M}$$
2. **Concentrator-Refinery Balancing Cut-Off ($g_{cr}$):**
   Concentrator metal output matches refinery capacity:
   $$C \cdot k(g_{cr}) = R_{\text{ref}} \implies k(g_{cr}) = \bar{g}(g_{cr}) \cdot y = \frac{R_{\text{ref}}}{C}$$
3. **Mining-Refinery Balancing Cut-Off ($g_{mr}$):**
   Total metal mined matches refinery capacity:
   $$M \cdot x(g_{mr}) \cdot k(g_{mr}) = R_{\text{ref}} \implies x(g_{mr}) \cdot k(g_{mr}) = \frac{R_{\text{ref}}}{M}$$

---

#### 4. Analytical Solutions for Parametric Grade Distributions

In mineral economics, grade distributions are commonly modeled using continuous parametric probability density functions. For a **Lognormal Grade Distribution** $g \sim \text{Lognormal}(\mu, \sigma^2)$, with probability density:
$$p(g) = \frac{1}{g \sigma \sqrt{2\pi}} \exp\left( -\frac{(\ln g - \mu)^2}{2\sigma^2} \right), \quad g > 0$$

The proportion of material mined exceeding cut-off grade $g$ is given by the standard normal complementary cumulative distribution:
$$x(g) = 1 - \Phi\left( \frac{\ln g - \mu}{\sigma} \right) = \Phi\left( \frac{\mu - \ln g}{\sigma} \right)$$

The quantity of contained metal per tonne of mined material above cut-off $g$ is derived via partial expectation:
$$\int_g^\infty u \cdot p(u) \, du = \exp\left(\mu + \frac{\sigma^2}{2}\right) \cdot \Phi\left( \frac{\mu + \sigma^2 - \ln g}{\sigma} \right)$$

The average grade of the selected ore $\bar{g}(g)$ is therefore:
$$\bar{g}(g) = \frac{\exp\left(\mu + \frac{\sigma^2}{2}\right) \cdot \Phi\left( \frac{\mu + \sigma^2 - \ln g}{\sigma} \right)}{\Phi\left( \frac{\mu - \ln g}{\sigma} \right)}$$

Under this analytical lognormal model, the mining-concentrator balancing cut-off $g_{mc}$ has an exact closed-form expression:
$$x(g_{mc}) = \frac{C}{M} \implies \Phi\left( \frac{\mu - \ln g_{mc}}{\sigma} \right) = \frac{C}{M}$$
$$g_{mc} = \exp\left( \mu - \sigma \Phi^{-1}\left( \frac{C}{M} \right) \right)$$

The concentrator-refinery balancing cut-off $g_{cr}$ satisfies:
$$\bar{g}(g_{cr}) = \frac{R_{\text{ref}}}{C \cdot y}$$
which is solved rapidly using single-variable Newton-Raphson iteration. For bounded bulk commodities (e.g., iron ore, where total Fe cannot physically exceed hematite's stoichiometric limit of $69.94\% \text{ Fe}$, or bauxite's $\text{Al}_2\text{O}_3$ limits), standard practice adopts a four-parameter **Beta Distribution** on interval $[a, b]$, ensuring cut-off calculations respect physical chemical bounds.

---

#### 5. Lane's Golden Rule: Finding the Global Optimum $g^*$

To determine the true profit-maximizing cut-off grade $g^*$ for any operating period:
1. Compare each pair of interacting stages:
   - Between Mine and Concentrator: Take the median of $\{g_m, g_{mc}, g_c\}$. Let this be $g_{M-C}$.
   - Between Concentrator and Refinery: Take the median of $\{g_c, g_{cr}, g_r\}$. Let this be $g_{C-R}$.
   - Between Mine and Refinery: Take the median of $\{g_m, g_{mr}, g_r\}$. Let this be $g_{M-R}$.
2. Evaluate total net present value increment $V(g)$ across the resulting candidates. The global optimum $g^*$ is the cut-off that maximizes $V(g)$ subject to physical throughput feasibility.

#### 6. Polymetallic and Multi-Element Extensions (Lane with By-Products & Penalties)
When ores contain multiple payable metals (e.g., Cu, Au, Mo) and penalty contaminants (e.g., As, Bi), the single-grade distribution $p(u)$ is replaced by a multivariate Net Smelter Return ($\text{NSR}$) distribution. The concentrator-limited cut-off transforms into an **NSR Cut-Off ($\text{NSR}_c$)**:
$$\text{NSR}_c = c + \frac{f + \delta W}{C}$$
Any block with $\text{NSR}_b \ge \text{NSR}_c$ qualifies as ore. If deleterious elements threaten smelter contract rejection ceilings ($Q_{\text{penalty}}^{\max}$), an additional downstream environmental bottleneck is formulated:
$$T_{\text{penalty}}(g) = \frac{x(g) \cdot \bar{g}_{\text{penalty}}(g) \cdot y_{\text{penalty}}}{Q_{\text{penalty}}^{\max}}$$
This generates a penalty-limited cut-off that truncates high-arsenic zones or enforces strategic blending with clean concentrates.

#### 7. Dynamic Programming Multi-Period Cut-Off Optimization
Over the complete Life of Mine, the decision problem is solved via Bellman backward induction. Let $S_t$ be the state of remaining mineral reserves in active pushbacks at period $t$:
$$W_t(S_t) = \max_{g_t \in \mathcal{G}} \left\{ \frac{\Pi(g_t, S_t) + W_{t+1}(S_{t+1}(g_t))}{1 + \delta \cdot \Delta t(g_t, S_t)} \right\}$$
where:
$$\Pi(g_t, S_t) = (s - r) \cdot Q_m(t) \cdot x(g_t) \cdot \bar{g}(g_t) \cdot y - c \cdot Q_m(t) \cdot x(g_t) - m \cdot Q_m(t) - f \cdot \Delta t(g_t)$$
$$\Delta t(g_t, S_t) = \max\left( \frac{Q_m(t)}{M}, \frac{Q_m(t) \cdot x(g_t)}{C}, \frac{Q_m(t) \cdot x(g_t) \cdot \bar{g}(g_t) \cdot y}{R_{\text{ref}}} \right)$$

As $t \to T_{\text{LOM}}$, remaining deposit reserves deplete, so $W_{t+1} \to 0$ and the opportunity cost $d_t = \delta W_t \to 0$. This establishes the mathematical necessity of the **Lane Declining Cut-Off Schedule**:
- **Years 1–5:** High $W_t \implies$ high opportunity cost $d_t \implies$ high cut-off grade ($g^* \gg g_{be}$). High-grade ore is fed to the mill; marginal ore ($g_{be} \le g < g^*$) is diverted to low-grade stockpiles.
- **Years 6–12:** Moderate $W_t \implies$ cut-off grade declines toward concentrator balancing grade.
- **Years 13–End of Life:** Depleted pit $\implies W_t \approx 0 \implies d_t \approx 0 \implies g^* \to g_{be}$. The pit reaches exhaustion, and the concentrator is fed entirely by reclaiming the accumulated low-grade stockpiles.

---

## 6. Worked example

We examine a realistic copper porphyry open-pit operation, "Atacama Copper Phase 3", evaluating cut-off grades and pushback economics.

### Operational and Economic Parameters
- **Mining Fleet Capacity ($M$):** $36,000,000 \text{ t rock / year}$ ($36.0 \text{ Mt/y}$)
- **Concentrator Capacity ($C$):** $15,000,000 \text{ t ore / year}$ ($15.0 \text{ Mt/y}$)
- **Refinery / Smelter Quota ($R_{\text{ref}}$):** $90,000 \text{ t Cu metal / year}$ ($90.0 \text{ kt/y}$)
- **Copper Market Price ($s$):** $\$8,500 / \text{t Cu metal}$ ($\$3.85 / \text{lb}$)
- **Refining and Realization Cost ($r$):** $\$1,000 / \text{t Cu metal}$ (Net price $s - r = \$7,500 / \text{t Cu}$)
- **Mining Cost ($m$):** $\$2.60 / \text{t rock mined}$
- **Concentrator Milling Cost ($c$):** $\$11.40 / \text{t ore milled}$
- **Fixed Site Overhead ($f$):** $\$36,000,000 / \text{year}$
- **Metallurgical Recovery ($y$):** Constant $88\% = 0.88$
- **Discount Rate ($\delta$):** $10\%$ per annum ($0.10$)
- **Remaining Net Present Value ($W$):** $\$720,000,000$

### Orebody Grade Distribution
The mined rock within the current pushback follows a parameterized distribution where grade $g$ is expressed in percent copper ($\% \text{ Cu}$):
- Grade range: $0.00\% \le g \le 1.80\%$
- For this pushback, empirical tonnage and grade functions are:
  - Proportion of ore: $x(g) = 1.0 - 1.25 \cdot g$ (for $g \in [0.15\%, 0.60\%]$)
  - Average grade of ore: $\bar{g}(g) = 0.22\% + 0.70 \cdot g$

---

### Step 1: Compute Time Cost Rate ($F$) and Break-Even Cut-Off ($g_{be}$)
Calculate corporate opportunity cost:
$$d = \delta \cdot W = 0.10 \times \$720,000,000 = \$72,000,000 / \text{year}$$

Total annual time cost:
$$F = f + d = \$36,000,000 + \$72,000,000 = \$108,000,000 / \text{year}$$

Calculate static break-even cut-off grade ($g_{be}$):
$$g_{be} = \frac{c}{(s - r) \cdot y} = \frac{\$11.40}{\$7,500 / \text{t} \times 0.88} = \frac{11.40}{6,600} = 0.001727 = 0.173\% \text{ Cu}$$

Any rock with grade below $0.173\% \text{ Cu}$ cannot cover milling operating costs and is designated as waste.

---

### Step 2: Calculate Lane's Three Limiting Cut-Off Grades

#### 1. Mining-Limited Cut-Off ($g_m$):
$$g_m = \frac{c}{(s - r) \cdot y} = g_{be} = 0.173\% \text{ Cu}$$

#### 2. Concentrator-Limited Cut-Off ($g_c$):
Fixed cost per tonne of mill capacity:
$$\frac{F}{C} = \frac{\$108,000,000}{15,000,000 \text{ t}} = \$7.20 / \text{t ore}$$

Effective processing cost:
$$c_{\text{eff}} = c + \frac{F}{C} = \$11.40 + \$7.20 = \$18.60 / \text{t ore}$$

$$g_c = \frac{c_{\text{eff}}}{(s - r) \cdot y} = \frac{\$18.60}{\$7,500 \times 0.88} = \frac{18.60}{6,600} = 0.002818 = 0.282\% \text{ Cu}$$

#### 3. Refinery-Limited Cut-Off ($g_r$):
Fixed cost per tonne of refinery metal capacity:
$$\frac{F}{R_{\text{ref}}} = \frac{\$108,000,000}{90,000 \text{ t}} = \$1,200 / \text{t Cu metal}$$

Net metal margin after refinery time cost:
$$(s - r)_{\text{eff}} = (s - r) - \frac{F}{R_{\text{ref}}} = \$7,500 - \$1,200 = \$6,300 / \text{t Cu}$$

$$g_r = \frac{c}{(s - r)_{\text{eff}} \cdot y} = \frac{\$11.40}{\$6,300 \times 0.88} = \frac{11.40}{5,544} = 0.002056 = 0.206\% \text{ Cu}$$

Summary of limiting cut-offs:
$$g_m = 0.173\% \text{ Cu}, \quad g_r = 0.206\% \text{ Cu}, \quad g_c = 0.282\% \text{ Cu}$$

---

### Step 3: Calculate Balancing Cut-Off Grades

#### 1. Mining-Concentrator Balancing Cut-Off ($g_{mc}$):
Mined ore fraction must equal capacity ratio:
$$x(g_{mc}) = \frac{C}{M} = \frac{15.0 \text{ Mt}}{36.0 \text{ Mt}} = 0.4167$$

Using the deposit ore fraction formula $x(g) = 1.0 - 1.25 \cdot g$:
$$1.0 - 1.25 \cdot g_{mc} = 0.4167 \implies 1.25 \cdot g_{mc} = 0.5833 \implies g_{mc} = \frac{0.5833}{1.25} = 0.4666\% \text{ Cu}$$

#### 2. Concentrator-Refinery Balancing Cut-Off ($g_{cr}$):
Metal yield must balance concentrator and refinery:
$$k(g_{cr}) = \bar{g}(g_{cr}) \cdot y = \frac{R_{\text{ref}}}{C} = \frac{90,000 \text{ t}}{15,000,000 \text{ t}} = 0.00600 = 0.600\% \text{ recovered Cu}$$

Since $y = 0.88$:
$$\bar{g}(g_{cr}) = \frac{0.00600}{0.88} = 0.006818 = 0.682\% \text{ Cu average grade}$$

Using $\bar{g}(g) = 0.22\% + 0.70 \cdot g$:
$$0.22\% + 0.70 \cdot g_{cr} = 0.682\% \implies 0.70 \cdot g_{cr} = 0.462\% \implies g_{cr} = 0.660\% \text{ Cu}$$

---

### Step 4: Determine the Governing Optimum Cut-Off Grade $g^*$

We evaluate the pair-wise median choices:
1. **Mine vs. Mill:** Median of $\{g_m, g_{mc}, g_c\} = \text{median}(0.173\%, 0.467\%, 0.282\%) = 0.282\% \text{ Cu}$ ($g_c$).
2. **Mill vs. Refinery:** Median of $\{g_c, g_{cr}, g_r\} = \text{median}(0.282\%, 0.660\%, 0.206\%) = 0.282\% \text{ Cu}$ ($g_c$).

Both pair-wise comparisons converge on $g_c = 0.282\% \text{ Cu}$. 
Because $g_c < g_{mc}$ ($0.282\% < 0.467\%$), the ore fraction is:
$$x(0.282\%) = 1.0 - 1.25 \cdot (0.282) = 1.0 - 0.3525 = 0.6475$$

If the mining fleet operates at capacity ($36 \text{ Mt}$), ore supply would be $36 \times 0.6475 = 23.31 \text{ Mt}$, far exceeding mill capacity ($15 \text{ Mt}$). Thus, the concentrator is the strictly binding bottleneck, confirming $g^* = 0.282\% \text{ Cu}$.

### Step 5: Cash Flow Comparison: Lane Dynamic Cut-Off vs. Static Break-Even

| Parameter | Policy A: Static Break-Even ($g = 0.173\%$) | Policy B: Lane Dynamic Optimum ($g = 0.282\%$) | Variance / Benefit |
|---|---|---|---|
| **Cut-Off Grade ($g$)** | $0.173\% \text{ Cu}$ | $0.282\% \text{ Cu}$ | $+0.109\% \text{ Cu}$ |
| **Ore Fraction ($x$)** | $1.0 - 1.25(0.173) = 78.4\%$ | $1.0 - 1.25(0.282) = 64.8\%$ | $-13.6\%$ (Higher selectivity) |
| **Average Mill Feed Grade ($\bar{g}$)** | $0.22\% + 0.70(0.173) = 0.341\% \text{ Cu}$ | $0.22\% + 0.70(0.282) = 0.417\% \text{ Cu}$ | $+0.076\% \text{ Cu}$ (+22.3% head grade) |
| **Tonnes Milled ($O_t$)** | $15,000,000 \text{ t}$ (Full capacity) | $15,000,000 \text{ t}$ (Full capacity) | Mill fully utilized |
| **Tonnes Mined Required** | $15.0 / 0.784 = 19.13 \text{ Mt}$ | $15.0 / 0.648 = 23.15 \text{ Mt}$ | $+4.02 \text{ Mt}$ (Within $36 \text{ Mt}$ fleet) |
| **Marginal Ore Stockpiled** | $0 \text{ t}$ | $23.15 \times (0.784 - 0.648) = 3.15 \text{ Mt}$ | Stored for LOM tail |
| **Recovered Metal Produced** | $15 \text{ Mt} \times 0.341\% \times 0.88 = 45,012 \text{ t Cu}$ | $15 \text{ Mt} \times 0.417\% \times 0.88 = 55,044 \text{ t Cu}$ | **$+10,032 \text{ t Cu metal}$ (+22.3%)** |
| **Gross Revenue (Net of Ref.)** | $45,012 \text{ t} \times \$7,500 = \$337,590,000$ | $55,044 \text{ t} \times \$7,500 = \$412,830,000$ | $+\$75,240,000$ |
| **Milling Cost ($c \cdot O_t$)** | $15 \text{ Mt} \times \$11.40 = \$171,000,000$ | $15 \text{ Mt} \times \$11.40 = \$171,000,000$ | Identical comminution spend |
| **Mining Cost ($m \cdot M_t$)** | $19.13 \text{ Mt} \times \$2.60 = \$49,738,000$ | $23.15 \text{ Mt} \times \$2.60 = \$60,190,000$ | $+\$10,452,000$ extra mining |
| **Fixed Cost ($f$)** | $\$36,000,000$ | $\$36,000,000$ | Invariant |
| **Annual Operating Cash Flow** | **$\$80,852,000 / \text{year}$** | **$\$145,640,000 / \text{year}$** | **$+\$64,788,000 / \text{year}$ (+80.1%)** |

By lifting the cut-off grade from $0.173\%$ to $0.282\%$, Atacama Copper increases annual operating cash flow by **$\$64.79\text{ million}$ (+80.1%)** in Year 1. The additional mining cost of $\$10.45\text{ million}$ is dwarfed by $\$75.24\text{ million}$ in extra copper revenues achieved by not diluting mill grinding circuits with sub-economic rock. The $3.15\text{ Mt}$ of marginal material between $0.173\%$ and $0.282\%$ is safely stockpiled to be milled at project maturity when $d \approx 0$.

---

### Step 6: Multi-Year Pushback Stripping and Declining Cut-Off Schedule

To demonstrate the full Life-of-Mine trajectory, Atacama Copper's three-stage pushback sequence is evaluated over 12 operating years:
- **Phase 1 (Pushback 1, Years 1–4):** High-grade starter pit, $SR_1 = 1.2$, $W_1 = \$720\text{M}$. Concentrator-limited cut-off governs at $g^* = 0.282\% \text{ Cu}$. Cumulative stockpile accumulation: $12.6\text{ Mt}$ of marginal ore averaging $0.228\% \text{ Cu}$.
- **Phase 2 (Pushback 2, Years 5–8):** Intermediate cutback, $SR_2 = 2.4$, $W_5 = \$410\text{M}$. Opportunity cost declines to $d_5 = 0.10 \times \$410\text{M} = \$41\text{M/yr}$. Time cost rate $F = \$77\text{M/yr}$. Limiting concentrator cut-off shifts downward:
  $$g_c = \frac{11.40 + (77 / 15)}{6,600} = \frac{11.40 + 5.13}{6,600} = 0.250\% \text{ Cu}$$
  The mine feeds $g^* = 0.250\% \text{ Cu}$ directly to the mill, while continuing to divert $0.173\% \le g < 0.250\%$ to low-grade stockpiles.
- **Phase 3 & Stockpile Reclamation (Years 9–12):** Pit finishes mining in Year 10 ($W_{10} = \$65\text{M}$, $d_{10} = \$6.5\text{M/yr}$). In Years 11 and 12, open-pit mining ceases ($M = 0$). Remaining project value $W \approx 0 \implies d \approx 0$. Cut-off drops to static break-even $g^* = g_{be} = 0.173\% \text{ Cu}$. Front-end loaders reclaim the $21.5\text{ Mt}$ accumulated stockpile at $\$1.20 / \text{t}$ rehandle cost, keeping the concentrator at 100% utilization until final site closure.

#### 12-Year Life-of-Mine Discounted Cash Flow Ledger (Lane Dynamic Policy)

| Year | Mining Phase / Source | Ore Milled (Mt) | Feed Grade (% Cu) | Metal Cu (kt) | Revenue ($M) | OPEX ($M) | Net Cash Flow ($M) | DF (10%) | Discounted CF ($M) |
|---|---|---|---|---|---|---|---|---|---|
| **Yr 1** | Pushback 1 (Starter) | 15.00 | 0.417% | 55.04 | $412.83 | $267.19 | $145.64 | 0.9091 | $132.40 |
| **Yr 2** | Pushback 1 | 15.00 | 0.417% | 55.04 | $412.83 | $267.19 | $145.64 | 0.8264 | $120.36 |
| **Yr 3** | Pushback 1 | 15.00 | 0.410% | 54.12 | $405.90 | $267.19 | $138.71 | 0.7513 | $104.21 |
| **Yr 4** | Pushback 1 / PB 2 Transition | 15.00 | 0.405% | 53.46 | $400.95 | $271.30 | $129.65 | 0.6830 | $88.55 |
| **Yr 5** | Pushback 2 | 15.00 | 0.395% | 52.14 | $391.05 | $278.40 | $112.65 | 0.6209 | $69.94 |
| **Yr 6** | Pushback 2 | 15.00 | 0.390% | 51.48 | $386.10 | $278.40 | $107.70 | 0.5645 | $60.80 |
| **Yr 7** | Pushback 2 | 15.00 | 0.380% | 50.16 | $376.20 | $278.40 | $97.80 | 0.5132 | $50.19 |
| **Yr 8** | Pushback 2 / PB 3 Transition | 15.00 | 0.370% | 48.84 | $366.30 | $282.10 | $84.20 | 0.4665 | $39.28 |
| **Yr 9** | Pushback 3 (Final Pit Wall) | 15.00 | 0.355% | 46.86 | $351.45 | $285.50 | $65.95 | 0.4241 | $27.97 |
| **Yr 10** | Pushback 3 (Pit Floor Exhaustion) | 15.00 | 0.345% | 45.54 | $341.55 | $285.50 | $56.05 | 0.3855 | $21.61 |
| **Yr 11** | Low-Grade Stockpile Reclaim | 15.00 | 0.235% | 31.02 | $232.65 | $189.00 | $43.65 | 0.3505 | $15.30 |
| **Yr 12** | Low-Grade Stockpile Reclaim | 6.50 | 0.225% | 12.87 | $96.53 | $81.90 | $14.63 | 0.3186 | $4.66 |
| **Total** | **LOM Full Operations** | **171.50 Mt** | **0.377% avg** | **556.57 kt** | **$4,174.34M** | **$2,932.07M** | **$1,142.27M** | — | **$735.27M** |

Under the static break-even policy without stockpiling, the deposit is processed at an average head grade of $0.341\% \text{ Cu}$ across 11.4 years without front-loaded revenue, yielding an undiscounted cash flow of $\$921.7\text{ million}$ and a discounted NPV of **$\$482.1\text{ million}$**. The Lane Dynamic Strategy with strategic stockpile management delivers an undiscounted cash flow of $\$1,142.3\text{ million}$ and an NPV of **$\$735.3\text{ million}$**, demonstrating an absolute financial uplift of **$+\$253.2\text{ million}$ (+52.5% NPV increase)**.

---

## 7. Code, query, or calculation

The following self-contained Python script calculates Lane's limiting cut-off grades ($g_m, g_c, g_r$), evaluates balancing cut-offs, and outputs the optimal policy. The script adheres strictly to operational bounds and runs with standard library dependencies.

```python
"""Lane's Cut-Off Grade Optimizer for Open-Pit Mining Operations."""

def solve_lane_cutoffs(params: dict) -> dict:
    s, r = params["price_metal"], params["refining_cost"]
    c, m = params["milling_cost"], params["mining_cost"]
    f, y = params["fixed_cost"], params["recovery"]
    M, C, R = params["cap_mine"], params["cap_mill"], params["cap_ref"]
    W, delta = params["npv_future"], params["discount_rate"]

    d = delta * W
    F = f + d
    net_price = (s - r) * y

    # Limiting Cut-Offs
    g_m = c / net_price
    g_c = (c + (F / C)) / net_price
    denom_r = (s - r - (F / R)) * y
    g_r = (c / denom_r) if denom_r > 0 else float("inf")

    # Deposit Grade Model in decimals: x(g) = 1.0 - 125.0*g; g_bar(g) = 0.0022 + 0.70*g
    g_mc = (1.0 - (C / M)) / 125.0 if M > 0 else 0.0
    # k(g) = g_bar * y = R / C  =>  g_bar = R / (C * y)
    target_gbar = R / (C * y)
    g_cr = (target_gbar - 0.0022) / 0.70

    # Lane pairwise median rule
    def median3(a, b, c_val):
        return sorted([a, b, c_val])[1]

    pair_mc = median3(g_m, g_mc, g_c)
    pair_cr = median3(g_c, g_cr, g_r)
    g_star = pair_mc if pair_mc == pair_cr else min(pair_mc, pair_cr)

    # Performance at g_star
    x_star = max(0.0, 1.0 - 125.0 * g_star)
    gbar_star = 0.0022 + 0.70 * g_star
    milled = min(C, M * x_star)
    metal = milled * gbar_star * y
    mining_req = milled / x_star
    cash_flow = (s - r) * metal - c * milled - m * mining_req - f

    return {
        "g_m_pct": g_m * 100, "g_c_pct": g_c * 100, "g_r_pct": g_r * 100,
        "g_mc_pct": g_mc * 100, "g_cr_pct": g_cr * 100, "g_opt_pct": g_star * 100,
        "cash_flow_m": cash_flow / 1e6, "metal_kt": metal / 1e3
    }

if __name__ == "__main__":
    p = {
        "price_metal": 8500.0, "refining_cost": 1000.0, "milling_cost": 11.40,
        "mining_cost": 2.60, "fixed_cost": 36e6, "recovery": 0.88,
        "cap_mine": 36e6, "cap_mill": 15e6, "cap_ref": 90e3,
        "npv_future": 720e6, "discount_rate": 0.10
    }
    res = solve_lane_cutoffs(p)
    print(f"Limiting Cut-Offs: gm={res['g_m_pct']:.3f}%, gc={res['g_c_pct']:.3f}%, gr={res['g_r_pct']:.3f}%")
    print(f"Balancing Cut-Offs: g_mc={res['g_mc_pct']:.3f}%, g_cr={res['g_cr_pct']:.3f}%")
    print(f"Optimal Cut-Off: g*={res['g_opt_pct']:.3f}% | Cash Flow: ${res['cash_flow_m']:.2f}M/yr | Cu: {res['metal_kt']:.2f} kt/yr")
```

### Execution Output and Validation
```text
Limiting Cut-Offs: gm=0.173%, gc=0.282%, gr=0.206%
Balancing Cut-Offs: g_mc=0.467%, g_cr=0.660%
Optimal Cut-Off: g*=0.282% | Cash Flow: $145.89M/yr | Cu: 55.08 kt/yr
```
The program confirms the hand-calculated cut-off of $0.282\% \text{ Cu}$ and reproduces the $\$145.89\text{ million}$ annual cash flow, demonstrating mathematical consistency.

---

## 8. Data, standards, and regulatory context

Mine planning and cut-off grade policies are governed by statutory public reporting codes and environmental standards. Publicly listed mining corporations are legally prohibited from publishing arbitrary cut-off grades or inflated reserve statements.

### 1. International Reserve Reporting Standards
Resource-to-reserve conversion requires establishing "Modifying Factors" (mining, processing, metallurgical, infrastructure, economic, marketing, legal, environmental, social, and governmental factors):
- **JORC Code (2012 Edition, Australasia):** Clause 29 mandates that Mineral Reserves must be based on a Life of Mine Plan demonstrating economic viability at the time of reporting. Clause 20 requires reporting of the cut-off grades used, the rationale for their selection, and the commodity price assumptions adopted. JORC Table 1 Section 4 requires explicit disclosure of mining dilution, mining recovery, minimum mining widths, and infrastructure availability.
- **NI 43-101 / CIM Standards (Canada):** Requires a Canadian Institute of Mining, Metallurgy and Petroleum (CIM) compliant Technical Report. Form 43-101F1 (Item 15) mandates full disclosure of the cut-off grade equation, marginal cut-off grade, dilution, mining recovery, and sensitivity analyses.
- **SEC Regulation S-K 1300 (United States, 17 CFR § 229.1300):** Subpart 1300 requires disclosure of Mineral Reserves based on a Pre-Feasibility or Feasibility Study prepared by a Qualified Person (QP). It explicitly requires the QP to prove **"Reasonable Prospects for Eventual Economic Extraction" (RPEEE)** using trailing commodity price ceilings (e.g., 24- to 36-month historical averages) rather than speculative spot prices. Item 601(b)(96) mandates filing a comprehensive Technical Report Summary detailing capital costs, operating costs, and discounted cash flow sensitivity tables.

### 2. Operational Cut-Off vs. Statutory Reserve Cut-Off
A critical regulatory distinction exists between:
- **The Statutory Reserve Cut-Off ($g_{\text{reserve}}$):** Calculated using conservative, long-term consensus pricing and static break-even economics ($g_{be}$). This defines the total legal reserve inventory published in annual 10-K, AIF, or ASX filings.
- **The Operational Cut-Off ($g^*_t$):** Derived dynamically via Lane's theory to maximize operational NPV. Mining companies may operate at $g^*_t > g_{\text{reserve}}$ in early years (diverting marginal reserve ore to stockpiles) without violating securities laws, provided that stockpiled material is accounted for as mineralized inventory with verified economic recoverability.

### 3. Tailings and Environmental Regulation (GISTM)
The **Global Industry Standard on Tailings Management (GISTM, 2020)** directly constrains cut-off grade strategies:
- Raising the cut-off grade increases the total volume of rock processed per unit of metal produced over time, increasing slurry tailings deposition rates.
- Diverting high volumes of sub-grade material to low-grade stockpiles creates massive surface footprints susceptible to **Acid Mine Drainage (AMD)**. If the waste contains reactive sulfide minerals (pyrite, pyrrhotite), environmental regulators (e.g., US EPA Clean Water Act, Chilean SMA, Australian EPA) require lined containment pads, active water treatment plants, and multi-million-dollar closure bonds.
- Under Scope 1 and Scope 2 decarbonization reporting, open-pit haulage accounts for 30–50% of direct mine diesel consumption. High stripping ratios carry embedded carbon penalties under the European Union Carbon Border Adjustment Mechanism (CBAM) and corporate ESG covenants.

---

## 9. Industry practice and case evidence

### 1. Minera Escondida (BHP / Rio Tinto, Chile)
Minera Escondida, the world's largest copper mine, extracts over $350\text{ million tonnes}$ of total material annually across the Escondida and Spence pits. 
- *Dynamic Cut-Off and Dual Processing Circuits:* Escondida pioneered multi-stream cut-off management. Extracted material is classified into three distinct processing destinations based on dynamic cut-offs:
  1. High-grade sulfide ore ($>0.70\% \text{ Cu}$) is routed to the Laguna Seca and Los Colorados flotation concentrators.
  2. Intermediate low-grade oxide and run-of-mine (ROM) sulfide ore ($0.30\% \le g < 0.70\%$) is routed to massive bio-leaching heap pads using sulfuric acid and seawater.
  3. Rock with $g < 0.30\%$ is dumped as waste.
- *Outcome:* This dynamic multi-stream cut-off strategy expanded Escondida's effective plant capacity without requiring proportional flotation grinding capital, maintaining annual production above $1.0\text{ million tonnes}$ of copper cathode and concentrate.

### 2. Kennecott Bingham Canyon (Rio Tinto, Utah, USA)
Bingham Canyon is one of the deepest open pits in the world ($>1.2\text{ km}$ depth, $>4\text{ km}$ wide).
- *The 2013 Man Engine Slide:* On April 10, 2013, Bingham Canyon experienced the largest non-volcanic landslide in North American history when $145\text{ million tonnes}$ of rock collapsed from the north-eastern pit wall.
- *Mechanism:* To maximize short-term NPV and defer stripping costs, pushback sequencing had steepened the working walls along a geotechnical fault zone. 
- *Consequence and Resilience:* Because Rio Tinto had installed advanced radar interferometry monitoring, the pit floor was evacuated 7 hours prior to collapse, resulting in zero fatalities. However, the haulage road was severed, burying a primary shovel and starving the Copperton concentrator of high-grade bottom-pit ore for months. Rio Tinto accelerated waste stripping in the upper pushbacks and reclaimed low-grade stockpiles to sustain mill operations while re-establishing haulage ramps.

### 3. Freeport-McMoRan Grasberg (Papua, Indonesia)
Grasberg represents the classic open-pit to underground transition:
- *Stripping Ratio Dynamics:* As the open pit deepened past 1,000 meters in the high-rainfall tropical terrain of Papua, the instantaneous stripping ratio escalated beyond $8:1 \text{ waste-to-ore}$. The geometric requirement to push back the mountain walls to access deeper ore became physically and economically unsustainable.
- *Strategic Transition:* Freeport initiated development of the Grasberg Block Cave (GBC) and Deep Mill Level Zone (DMLZ) underground operations beneath the open-pit floor. By transitioning to block caving, the operation bypassed open-pit surface stripping entirely, utilizing gravity-induced rock fracturing to deliver high-tonnage ore directly to underground gyratory crushers.

### 4. Kalgoorlie Super Pit (Northern Star Resources, Western Australia)
Consolidating hundreds of historical underground leases across the famous Golden Mile into a single mega open pit ($3.5\text{ km}$ long, $1.5\text{ km}$ wide):
- *Pushback Management:* The operation relies on sequential pushback cutbacks (e.g., Golden Pike, Morrison, Fimiston South). 
- *Failure Mode:* In 2018, rockfalls on the east wall compromised access ramps, halving mining rates. Northern Star restructured the life-of-mine plan by optimizing cut-off grades across underground portal developments inside the pit floor, substituting high-grade underground stoping feed to preserve mill utilization while cutback stripping stabilized the outer open-pit slopes.

### 5. Olympic Dam (BHP, South Australia)
Olympic Dam exploits one of the world's largest polymetallic iron oxide copper gold (IOCG) deposits, producing Copper, Uranium oxide ($\text{U}_3\text{O}_8$), Gold, and Silver.
- *Coupled Multi-Stage Bottlenecks:* The underground sub-level open stoping operation feeds an integrated hydrometallurgical flowsheet. Extracted ore passes through grinding, copper flotation, acid leaching of flotation tailings to extract uranium, solvent extraction (SX), precipitation of ammonium diuranate (yellowcake), and an on-site copper smelter and electro-refinery.
- *Lane Bottleneck Interaction:* Cut-off grade optimization cannot be solved on a single metal basis. BHP applies multi-variable Net Smelter Return (NSR) cut-offs. If global uranium market quotas are binding, the refinery stage constraint $R_{\text{ref}}$ becomes active on uranium rather than copper, shifting the optimal stope cut-off grade toward copper-rich, uranium-moderate ore zones. Conversely, when copper smelting capacity is constrained by sulfur gas handling limits, the plant balances feed between high-sulfur bornite-chalcopyrite stopes and low-sulfur chalcocite stopes.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Root Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|
| **Static Cut-Off Fallacy** | Constant cut-off grade reported in LOM plan despite large swings in remaining reserve value ($W_t$). | Mine planning software configured for single-period break-even analysis without Lane opportunity cost loops. | Sub-optimal cash flows; low-grade rock floods mill in early years, depressing enterprise NPV by 15–30%. | Implement dynamic cut-off scheduling with automated $d = \delta W$ iteration in LOM scheduling engines. |
| **Premature High-Grading** | Pit floor drops rapidly while upper pushbacks remain unstripped; stripping ratio drops below $50\%$ of LOM average. | Management incentivized on short-term quarterly production bonuses, deferring waste stripping. | Concentrator starves when high-grade benches are exhausted; subsequent multi-year pre-stripping cash sink. | Enforce minimum pushback vertical advance limits (sink rates) and audit stripping ratio compliance against LOM plan. |
| **Geotechnical Over-Steepening** | Cracking on upper bench berms; micro-seismic event frequency increases on radar monitoring. | Over-steepening precedence arcs in Lerchs-Grossmann to artificially lower waste stripping requirements. | Catastrophic pit wall collapse; loss of haul roads, buried equipment, extended site shut-down. | Calibrate LG search patterns to empirical geotechnical domains; incorporate berm catchment widths and ramp geometry. |
| **Pushback Bottlenecking (Pinching)** | Loading shovels waiting on haul trucks; single-lane haulage traffic; high tire wear. | Pushback width designed narrower than Minimum Mining Width (MMW) to chase high NPV shells. | Shovel productivity drops by 30–50%; haulage collision hazards; unachievable production targets. | Impose strict MMW cutoffs ($>60\text{ m}$) during phase engineering; reject non-compliant pit shells. |
| **Stockpile Mineral Degradation** | Declining copper/gold flotation recovery from reclaimed stockpile material; acid leachate runoff. | Oxidation of reactive sulfide minerals (chalcopyrite, pyrrhotite) during atmospheric exposure on stockpile pads. | Metallurgical recoveries drop by 10–25%; heavy reagent consumption; environmental fines for acidic runoff. | Segregate stockpiles by oxidation kinetics; cap long-term stockpiles with clay; prioritize rapid reclaim schedules. |
| **Smelter Penalty Blindness** | Offtake deductions increase; smelter rejects concentrate lots or imposes punitive penalty charges. | Cut-off optimization focused solely on primary metal ($g_{\text{Cu}}$) while ignoring co-associated arsenic (As), bismuth (Bi), or mercury (Hg). | Severe revenue loss; smelter rejection forces emergency blending or distress sales at steep discounts. | Formulate multi-element Net Smelter Return (NSR) block models; incorporate non-linear penalty curves. |
| **Recovery Discontinuity at Cut-Off** | Reconciled metal yield is systematically 10–20% below feasibility study plant projections. | Assuming constant recovery fraction ($y = 0.88$) down to zero grade instead of grade-dependent recovery ($R(g)$). | Processing low-grade material generates negative operating cash flow; plant energy wasted on barren gangue. | Model recovery as non-linear kinetic flotation curves: $R(g) = R_{\max} [1 - \exp(-k \cdot g)]$. |
| **Fleet Capacity Mismatch** | Haul trucks queuing at crusher while stripping shovels idle, or vice versa. | Adjusting cut-off grades dynamically without re-optimizing the load-and-haul mobile equipment fleet. | Capital misallocation; mobile fleet operates at poor utilization, driving up unit mining costs ($m$). | Couple Lane cut-off optimization with discrete-event truck-shovel fleet simulation models. |
| **Sunk Cost Fallacy in Waste Stripping** | Continuing to strip high-cost pushbacks during severe commodity price downturns. | Treating planned waste pre-stripping as an unavoidable commitment rather than an optional call option. | Severe corporate cash burn; operations drain treasury stripping waste for reserves that cannot be sold profitably. | Recalibrate pushback viability dynamically against trailing spot prices; temporarily mothball outer pushbacks. |
| **Information Smoothing Bias** | Mined ore tonnage is higher than planned, but actual head grade is significantly lower than kriged model. | Ordinary Kriging smoothing creates artificial dispersion of grades, misestimating tonnage above high cut-offs. | Concentrator fails to hit metal production targets; mill throughput restricted by harder-than-expected gangue. | Utilize Conditional Geostatistical Simulation (e.g., Sequential Gaussian Simulation) to model grade variability. |
| **Tailings Dam Bottlenecking** | Concentrator forced to throttle throughput below nameplate capacity due to tailings crest height delays. | Cut-off optimization assumed unconstrained processing without linking slurry volume to tailings lift schedules. | Plant downtime; capital stranded; regulatory intervention halting discharge permits under GISTM rules. | Couple Lane concentrator capacity constraint $C$ directly with annual engineered tailings deposition volume limits. |
| **Haul Road Gradient Creep** | Haul truck transmission overheating; cycle times increase by 25%; diesel burn spikes. | Pit slope steepening forces internal ramps to exceed design gradient (creeping from 8% to 12% or 14%). | Equipment premature failures; catastrophic haul truck runaway risks; severe haulage cost escalation. | Enforce rigid haul road gradient ceilings in mine planning software and verify physical surveys monthly. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks It | How to Test / Validate |
|---|---|---|---|
| **Deterministic Commodity Prices** | Required to calculate single economic block values ($w_b$) and Lerchs-Grossmann maximum closures. | High commodity price volatility, macroeconomic super-cycles, and structural currency shifts. | Run parametric sensitivity analysis across revenue factors ($\lambda \in [0.4, 1.4]$) and Monte Carlo price simulations. |
| **Homogeneous Geotechnical Slopes** | Simplifies precedence arc generation in graph algorithms using uniform geometric cones. | Structural geological discontinuities, anisotropic fault planes, bedding planes, and groundwater pore pressure. | Divide block model into structural geotechnical domains with azimuth-dependent slope precedence templates. |
| **Instantaneous Concentrator Adaptation** | Assumes mill can process variable head grades at nameplate capacity without throughput throttling. | SAG mill grindability variations (Bond Work Index, $A \times b$), semi-autogenous grinding hardness, and reagent kinetics. | Incorporate geometallurgical hardness models; constrain mill throughput as a non-linear function of rock hardness. |
| **Linear Mining and Milling Costs** | Assumes unit costs ($m, c$) are independent of excavation depth, haul distance, and production volume. | Deep haulage cycle times increase diesel consumption; economies of scale alter plant unit overhead. | Apply bench-specific mining cost escalators ($m_z = m_0 + \Delta m \cdot z$) and variable power/reagent cost models. |
| **Zero Stockpile Degradation** | Assumes stockpiled marginal ore retains 100% of its metallurgical recovery over decades of storage. | Pyrite oxidation, clay weathering, mineral dissolution, and secondary copper leaching during wet seasons. | Conduct long-term column leach tests and metallurgical kinetic oxidation trials on weathered core samples. |
| **Infinite Waste Dump Capacity** | Assumes waste dumps can absorb unlimited rock volumes immediately adjacent to the pit rim. | Topographical boundaries, environmental land permits, indigenous heritage claims, and haul distance limits. | Integrate waste dump landform design directly into life-of-mine haulage and cycle-time optimization models. |
| **Stationary Discount Rate ($\delta$)** | Required for constant opportunity cost calculation ($d = \delta W$) throughout the mine life. | Changing corporate capital structure, debt covenants, political risk premiums, and interest rate cycles. | Recalibrate weighted average cost of capital (WACC) periodically; conduct discount rate sensitivity sweeps (8% to 15%). |
| **Perfect Selective Mining Unit (SMU)** | Assumes mining equipment can separate ore from waste cleanly at block boundary resolutions (e.g., $10\text{ m} \times 10\text{ m}$). | Blast-induced movement, excavator bucket sizing, bench face slumping, and grade control delineation errors. | Apply mining dilution (5%–15%) and ore loss (2%–8%) factors to raw block model inventories prior to optimization. |
| **Static Orebody Geometry** | Assumes resource boundaries are completely defined prior to life-of-mine planning. | Infill and brownfield exploration drilling identifying major orebody depth or strike extensions. | Perform rolling dynamic life-of-mine schedule updates as new drilling data is integrated into block models. |
| **Independent Stage Bottlenecks** | Treats mining, milling, and refining capacities as independent scalar constants ($M, C, R$). | Integrated energy grid curtailments, shared water supply shortages, or port conveyor outages. | Model interconnected system constraints using linear programming network flow formulations with shared resources. |

---

## 12. Exercises

### Exercise 1: Break-Even vs. Marginal Cut-Off under Differential Mining Costs
A polymetallic open pit mines copper ore at $\$2.80 / \text{t rock}$ and waste at $\$2.30 / \text{t rock}$. Milling cost is $\$12.50 / \text{t ore}$, and administration is $\$1.50 / \text{t ore}$. Net copper price after refining deductions is $\$7,200 / \text{t Cu}$, and metallurgical recovery is $86\%$.
- **Deliverable:** Calculate (a) the break-even cut-off grade ($g_{be}$) for unmined in-situ rock, and (b) the internal marginal cut-off grade ($g_{\text{marg}}$) for rock already excavated and sitting on the pit floor.
- **Acceptance Criteria:** Correctly incorporate the differential mining cost ($m_o - m_w$) into $g_{be}$ and treat mining cost as sunk for $g_{\text{marg}}$.
- **Estimated Time:** 20 minutes.

### Exercise 2: Analytical Derivation of Lane's Concentrator Cut-Off ($g_c$)
Starting from Lane's net value equation $V_c(g) = (s - r) x(g) k(g) - c x(g) - m - F [x(g)/C]$, carry out the step-by-step differentiation with respect to cut-off grade $g$ to prove that $g_c = [c + (F/C)] / [(s - r) y]$.
- **Deliverable:** Full mathematical derivation detailing the application of Leibniz rule / calculus on integrals of the probability density function $p(u)$.
- **Acceptance Criteria:** Clearly show how $-g \cdot p(g)$ arises from the derivative of the recovered metal function.
- **Estimated Time:** 25 minutes.

### Exercise 3: 2D Lerchs-Grossmann Graph Closure Hand Calculation
Consider a 2D cross-section with 5 surface blocks (Bench 1) and 3 underlying blocks (Bench 2). Geotechnical slope angle is $45^\circ$, requiring that to extract Bench 2 Block $(2, j)$, the overlying blocks $(1, j-1), (1, j), (1, j+1)$ must be extracted. Block economic values are:
- Bench 1: Block $(1,1) = -\$2$; Block $(1,2) = -\$3$; Block $(1,3) = -\$4$; Block $(1,4) = -\$3$; Block $(1,5) = -\$2$.
- Bench 2: Block $(2,2) = +\$8$; Block $(2,3) = +\$14$; Block $(2,4) = +\$5$.
- **Deliverable:** Draw the directed block precedence graph, identify all feasible pit closures, and calculate the maximum closure weight.
- **Acceptance Criteria:** State which blocks are in the optimal pit and prove that no alternative feasible closure yields higher cash flow.
- **Estimated Time:** 30 minutes.

### Exercise 4: Pushback Sequencing and Discounted Cash Flow
A copper deposit can be extracted via two strategies:
- Strategy 1 (Worst-Case / Horizontal Benches): Mines $10\text{ Mt}$ ore and $20\text{ Mt}$ waste uniformly over 3 years ($SR = 2.0$ each year). Cash flows are: Year 1 = $\$30\text{M}$, Year 2 = $\$40\text{M}$, Year 3 = $\$50\text{M}$.
- Strategy 2 (Best-Case / Pushbacks): Pre-strips in Year 1 to mine high-grade core ($SR = 3.0$ in Year 1; $1.5$ in Years 2–3). Cash flows are: Year 1 = $\$65\text{M}$, Year 2 = $\$40\text{M}$, Year 3 = $\$15\text{M}$.
- **Deliverable:** Compute the Net Present Value (NPV) of both strategies at a $10\%$ discount rate. State the dollar value added by pushback staging.
- **Acceptance Criteria:** Show exact discounting arithmetic for each year; state the percentage NPV uplift.
- **Estimated Time:** 20 minutes.

### Exercise 5: Dynamic Cut-Off Schedule and Stockpile Sizing
An iron ore mine has a 3-year pushback with fixed mill capacity of $10\text{ Mt/yr}$. Lane's optimization indicates optimal cut-off grades of $g^*_1 = 58\% \text{ Fe}$, $g^*_2 = 55\% \text{ Fe}$, and $g^*_3 = 52\% \text{ Fe}$. The statutory break-even cut-off is $50\% \text{ Fe}$. In Years 1 and 2, $2.5\text{ Mt/yr}$ of marginal ore ($50\% \le g < g^*$) is mined and diverted to a low-grade stockpile.
- **Deliverable:** Design the stockpile reclamation schedule for Year 4 assuming pit mining terminates in Year 3. Specify required stockpile pad capacity (tonnes) and footprint (m²) assuming a loose bulk density of $2.2 \text{ t/m}^3$ and an angle of repose of $35^\circ$ with max stack height of $15\text{ m}$.
- **Acceptance Criteria:** Accurate volume, area, and reclaim sequencing calculations.
- **Estimated Time:** 35 minutes.

### Exercise 6: Break-Even Stripping Ratio Sensitivity to Carbon Pricing
A gold mine operates at $P = \$2,000 / \text{oz Au}$ with realization cost $r = \$50 / \text{oz}$. Recovered grade is $g \cdot R = 1.8 \text{ g/t} = 0.05787 \text{ oz/t ore}$. Milling cost is $c = \$18.00 / \text{t ore}$. Base mining costs are $m_o = \$3.00 / \text{t}$ and $m_w = \$2.50 / \text{t}$. A government carbon tax of $\$80 / \text{t CO}_2\text{e}$ is introduced. Diesel haulage produces $0.015 \text{ t CO}_2\text{e} / \text{t rock mined}$, and grid electricity produces $0.030 \text{ t CO}_2\text{e} / \text{t ore milled}$.
- **Deliverable:** Calculate the Break-Even Stripping Ratio ($BESR$) (a) before the carbon tax, and (b) after the carbon tax.
- **Acceptance Criteria:** Incorporate carbon emissions costs into both mining and milling unit cost terms.
- **Estimated Time:** 25 minutes.

### Exercise 7: Regulatory Reserve Audit Critique (NI 43-101 / SEC S-K 1300)
A junior mining company announces a Feasibility Study claiming a $40\%$ increase in Mineral Reserves by adopting a "dynamic spot-price cut-off grade" tied to a recent 3-month spike in lithium prices.
- **Deliverable:** Write an audit critique evaluating the compliance of this statement against SEC S-K 1300 and NI 43-101 standards.
- **Acceptance Criteria:** Cite specific regulatory clauses regarding pricing horizons, Qualified Person responsibilities, and reserve conversion.
- **Estimated Time:** 30 minutes.

### Exercise 8: Mine-to-Mill Comminution Energy Trade-Off
A mine spends $\$0.80 / \text{t rock}$ on drilling and blasting, producing an average fragment size ($P_{80}$) of $250\text{ mm}$, resulting in semi-autogenous grinding (SAG) mill throughput of $1,200 \text{ t/h}$ at $\$9.50 / \text{t}$ power cost. By increasing blast explosive powder factor by $40\%$, drilling and blasting cost rises to $\$1.15 / \text{t}$, but $P_{80}$ drops to $120\text{ mm}$, increasing SAG throughput to $1,450 \text{ t/h}$ and reducing grinding power to $\$7.80 / \text{t}$.
- **Deliverable:** Calculate the net cost change per tonne of ore and determine the impact on Lane's concentrator-limited cut-off grade $g_c$.
- **Acceptance Criteria:** Quantify both unit operating cost savings and the capacity expansion effect on $g_c$.
- **Estimated Time:** 30 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1
1. **Break-Even Cut-Off ($g_{be}$):**
   $$c_{\text{total}} = c + m_o - m_w = \$12.50 + \$1.50 + (\$2.80 - \$2.30) = \$14.00 + \$0.50 = \$14.50 / \text{t ore}$$
   $$g_{be} = \frac{\$14.50}{\$7,200 \times 0.86} = \frac{14.50}{6,192} = 0.002342 = 0.234\% \text{ Cu}$$
2. **Marginal Internal Cut-Off ($g_{\text{marg}}$):**
   Mining cost is sunk. Rock is already broken on the pit floor.
   $$c_{\text{marg}} = c_{\text{mill}} + c_{\text{admin}} = \$12.50 + \$1.50 = \$14.00 / \text{t ore}$$
   $$g_{\text{marg}} = \frac{\$14.00}{\$7,200 \times 0.86} = \frac{14.00}{6,192} = 0.002261 = 0.226\% \text{ Cu}$$
   *Decision Rule:* Material between $0.226\%$ and $0.234\%$ must be mined if it sits in the pit path, but once blasted, it is sent to the mill or stockpile rather than the waste dump.

### Answer to Exercise 2
Start with:
$$V_c(g) = (s - r) \int_g^\infty u \cdot p(u) \, du \cdot y - \left( c + \frac{F}{C} \right) \int_g^\infty p(u) \, du - m$$
Applying Leibniz integral rule:
$$\frac{d}{dg} \left[ \int_g^\infty u \cdot p(u) \, du \right] = -g \cdot p(g), \quad \frac{d}{dg} \left[ \int_g^\infty p(u) \, du \right] = -p(g)$$
Differentiating $V_c(g)$ with respect to $g$:
$$\frac{d V_c}{dg} = -(s - r) \cdot g \cdot p(g) \cdot y + \left( c + \frac{F}{C} \right) \cdot p(g)$$
Setting $\frac{d V_c}{dg} = 0$ and dividing by $p(g) \neq 0$:
$$-(s - r) \cdot g \cdot y + \left( c + \frac{F}{C} \right) = 0 \implies g_c = \frac{c + \frac{F}{C}}{(s - r) \cdot y}$$
Q.E.D.

### Answer to Exercise 3
- To extract Block $(2,3) = +\$14$, slope constraints require extracting $(1,2), (1,3), (1,4)$.
  Weight of this closure $C_1$:
  $$w(C_1) = w(2,3) + w(1,2) + w(1,3) + w(1,4) = +14 + (-3) + (-4) + (-3) = +14 - 10 = +\$4$$
- To extract Block $(2,2) = +\$8$, we must extract $(1,1), (1,2), (1,3)$.
  Weight $w(C_2) = +8 + (-2) + (-3) + (-4) = +8 - 9 = -\$1$ (Negative, sub-economic on its own).
- Combining $(2,2)$ and $(2,3)$ requires extracting blocks $(1,1), (1,2), (1,3), (1,4)$:
  Weight $w(C_{1+2}) = (+14 + 8) + (-2 - 3 - 4 - 3) = +22 - 12 = +\$10$.
- Combining $(2,2), (2,3), (2,4)$ requires extracting all 5 surface blocks:
  Weight $w(C_{\text{all}}) = (+14 + 8 + 5) + (-2 - 3 - 4 - 3 - 2) = +27 - 14 = +\$13$.
- Comparing all valid closures:
  - Empty pit: $\$0$
  - Only $(2,3)$ closure: $+\$4$
  - $(2,2) + (2,3)$ closure: $+\$10$
  - All blocks closure: $+\$13$
- **Optimal Pit Limit:** Extract all 8 blocks. The maximum closure weight is **$+\$13$**.

### Answer to Exercise 4
Calculate NPV at $\delta = 10\%$ ($DF_1 = 0.9091, DF_2 = 0.8264, DF_3 = 0.7513$):
- **Strategy 1 (Worst-Case):**
  $$NPV_1 = 30(0.9091) + 40(0.8264) + 50(0.7513) = 27.27 + 33.06 + 37.57 = \$97.90\text{ million}$$
- **Strategy 2 (Best-Case Pushbacks):**
  $$NPV_2 = 65(0.9091) + 40(0.8264) + 15(0.7513) = 59.09 + 33.06 + 11.27 = \$103.42\text{ million}$$
- **Value Added:** $\$103.42\text{M} - \$97.90\text{M} = +\$5.52\text{ million}$ (+5.64% increase) achieved purely through temporal extraction sequencing without altering total recovered mineral mass.

### Answer to Exercise 5
1. **Stockpile Volume and Pad Dimensions:**
   - Accumulated marginal ore: $2 \times 2.5\text{ Mt} = 5.0\text{ Mt}$.
   - Stockpile bulk volume: $V = \frac{5,000,000\text{ t}}{2.2\text{ t/m}^3} = 2,272,727\text{ m}^3$.
   - For a flat-topped stockpile with height $h = 15\text{ m}$ and angle of repose $\theta = 35^\circ$, the horizontal setback per side is $\Delta x = \frac{15}{\tan(35^\circ)} = \frac{15}{0.7002} \approx 21.4\text{ m}$.
   - Modeling as a square frustum with base side $B$ and top side $b = B - 2(21.4) = B - 42.8\text{ m}$:
     $$V = \frac{h}{3} \cdot [B^2 + B \cdot b + b^2] = 5 \cdot [3 B^2 - 128.4 B + 1831.8] = 2,272,727\text{ m}^3$$
     $$3 B^2 - 128.4 B + 1831.8 = 454,545 \implies 3 B^2 - 128.4 B - 452,713 = 0$$
     $$B = \frac{128.4 + \sqrt{(-128.4)^2 - 4(3)(-452,713)}}{6} = \frac{128.4 + \sqrt{16,486 + 5,432,556}}{6} = \frac{128.4 + 2334.3}{6} \approx 410.5\text{ m}$$
   - Base footprint area: $A = (410.5\text{ m})^2 \approx 168,510\text{ m}^2 \approx 16.85\text{ hectares}$.
2. **Year 4 Reclamation Schedule:**
   - Total ore available in Year 4: $5.0\text{ Mt}$ at average grade $\approx 53.5\% \text{ Fe}$.
   - Mill capacity: $10.0\text{ Mt/yr}$.
   - The stockpile feeds the mill for exactly $5.0 / 10.0 = 0.5\text{ years}$ (6 months) at $100\%$ capacity.

### Answer to Exercise 6
1. **Base BESR (Pre-Tax):**
   - Net revenue per tonne ore: $(P - r) \cdot (g \cdot R) = (\$2,000 - \$50) \times 0.05787 = \$1,950 \times 0.05787 = \$112.85 / \text{t ore}$.
   - Production cost per tonne ore: $c + m_o = \$18.00 + \$3.00 = \$21.00 / \text{t ore}$.
   - Waste mining cost: $m_w = \$2.50 / \text{t waste}$.
   $$BESR_{\text{base}} = \frac{\$112.85 - \$21.00}{\$2.50} = \frac{\$91.85}{\$2.50} = 36.74 : 1$$
2. **Post-Carbon Tax BESR:**
   - Carbon cost per tonne rock mined: $0.015 \text{ t CO}_2\text{e} \times \$80 = \$1.20 / \text{t}$.
   - Carbon cost per tonne ore milled: $0.030 \text{ t CO}_2\text{e} \times \$80 = \$2.40 / \text{t}$.
   - New costs:
     - $m_w' = \$2.50 + \$1.20 = \$3.70 / \text{t waste}$.
     - $m_o' = \$3.00 + \$1.20 = \$4.20 / \text{t ore}$.
     - $c' = \$18.00 + \$2.40 = \$20.40 / \text{t ore}$.
   - Net margin per tonne ore: $\$112.85 - (\$20.40 + \$4.20) = \$112.85 - \$24.60 = \$88.25 / \text{t ore}$.
   $$BESR_{\text{tax}} = \frac{\$88.25}{\$3.70} = 23.85 : 1$$
   *Impact:* The carbon tax slashes the allowable break-even stripping ratio by **$35.1\%$** (from $36.74$ to $23.85$), contracting the physical ultimate pit shell and stranding deep reserves.

### Answer to Exercise 7
- **Audit Findings:** The disclosure violates SEC S-K 1300 (17 CFR § 229.1300) and NI 43-101.
- **Specific Non-Compliances:**
  1. *Price Deck Violation:* SEC S-K 1300 mandates trailing multi-year average prices (typically 24 to 36 months) or documented consensus long-term forecasts. Using a 3-month spot price spike is explicitly illegal for statutory reserve estimation.
  2. *Reasonable Prospects Violation (RPEEE):* Mineral Reserves require proven economic extractability over the entire multi-decade project life. Spot spikes do not guarantee economic viability over the mine life.
  3. *Qualified Person (QP) Liability:* The QP must sign off under liability of statutory securities misrepresentation. Declaring reserves on short-term spot prices subjects the issuer to mandatory retraction, trading halts, and shareholder class-action litigation.

### Answer to Exercise 8
1. **Operating Cost Impact:**
   - Extra drill & blast spend: $\$1.15 - \$0.80 = +\$0.35 / \text{t ore}$.
   - Grinding power savings: $\$9.50 - \$7.80 = -\$1.70 / \text{t ore}$.
   - Net direct operating cost change: $\Delta c = +\$0.35 - \$1.70 = -\$1.35 / \text{t ore}$ ($c$ drops from $\$9.50$ to $\$8.15 / \text{t}$).
2. **Capacity and Cut-Off Impact:**
   - Mill throughput increases: $1,200 \text{ t/h} \to 1,450 \text{ t/h}$ (+20.8% capacity expansion in $C$).
   - Recall $g_c = \frac{c + (F/C)}{(s - r) y}$.
   - Because $c$ decreases by $\$1.35 / \text{t}$ AND $C$ increases by $20.8\%$ (which reduces $F/C$), the numerator $c + F/C$ drops significantly on both counts.
   - *Conclusion:* Better fragmentation lowers the concentrator-limited cut-off grade $g_c$, enabling the plant to process lower-grade material profitably and expanding total recoverable reserve metal.

---

## 14. Interview questions

### Question 1: What is the fundamental flaw of using a static break-even cut-off grade over the entire life of an open-pit mine?
- **Strong Answer:** A static break-even cut-off grade ($g_{be} = c / [(s-r)y]$) assumes that processing capacity is infinite and capital has zero opportunity cost over time. In reality, concentrator throughput is bounded. Processing a tonne of marginal rock whose grade barely covers operating costs takes exactly the same comminution time in a ball mill as a tonne of high-grade rock. By occupying the bottleneck, marginal ore delays the extraction and cash flows of all higher-grade ore remaining in the deposit. Kenneth Lane proved that the true effective processing cost is $c + (f + \delta W)/C$, where $\delta W$ is the opportunity cost of capital. Applying static break-even early in mine life depresses NPV by 10% to 30%.
- **Weak Answer:** Static cut-offs are bad because commodity prices change every day on the market, so you should adjust your cut-off to follow the spot price.

### Question 2: How does Picard's theorem relate the Lerchs-Grossmann pit optimization problem to network flow algorithms?
- **Strong Answer:** Jean-Claude Picard (1976) proved that the Maximum Weight Closure Problem on a directed graph—which is the exact mathematical formulation of the Lerchs-Grossmann Ultimate Pit Limit—is isomorphic to the Minimum Cut problem on an augmented $(s, t)$ network. By constructing a source $s$ connected to positive blocks with capacities equal to block profit ($w_b > 0$), connecting negative blocks ($w_b < 0$) to sink $t$ with capacities $-w_b$, and setting all slope precedence arcs to infinite capacity, any finite $(s, t)$-cut enforces slope feasibility. Minimizing cut capacity maximizes closure weight. This established that pit optimization can be solved in polynomial time via max-flow algorithms (e.g., Push-Relabel, Hochbaum's Pseudoflow) rather than relying on heuristic graph traversal.
- **Weak Answer:** Picard proved that you can use linear programming to find pit limits, which makes it faster than doing it by hand.

### Question 3: Under Lane's theory, why does the mining-limited cut-off grade ($g_m$) NOT depend on the corporate discount rate or remaining deposit value?
- **Strong Answer:** The mining-limited cut-off $g_m$ represents the scenario where the mining fleet is operating at 100% capacity while the concentrator and refinery have excess idle capacity. The time required to mine unit material is $T = 1/M$, which is a constant invariant to cut-off grade $g$. When taking the first derivative of the net value equation $V_m(g) = P(g) - (f + \delta W)(1/M)$ with respect to $g$, the term $(f + \delta W)/M$ vanishes completely because it is independent of $g$. Therefore, $g_m$ simplifies strictly to the operating marginal cost divided by net price, completely unaffected by the opportunity cost of capital.
- **Weak Answer:** Because mining costs are paid to truck drivers and diesel suppliers, which doesn't depend on Wall Street interest rates.

### Question 4: How does a mining engineer determine pushbacks from nested pit shells?
- **Strong Answer:** Nested pit shells are generated by scaling commodity price through parametric revenue factors $\lambda \in [0.3, 1.2]$. While these shells identify the concentric, highest-value zones of the deposit, raw shells cannot be mined directly because they have jagged boundaries and lack haulage ramps. An engineer selects a subset of shells (e.g., 4 to 6 phases) that provide sufficient contained ore to feed the mill for 2–4 years per phase. The engineer then applies smoothing algorithms to enforce the Minimum Mining Width (MMW) for shovel-truck turning circles and designs operational haul road ramps at an 8–10% gradient with geotechnical berms.
- **Weak Answer:** You just pick every fifth shell from the computer output and tell the excavators to dig them in order.

### Question 5: What is the operational risk of adopting an "Increasing / Just-in-Time" stripping ratio strategy?
- **Strong Answer:** An increasing stripping strategy defers waste mining to the latest possible geometric period, maximizing early cash flow and NPV. However, it completely eliminates operational buffering. In a JIT schedule, waste benches sit immediately adjacent to active ore faces. Any operational disruption—such as a localized pit wall failure, blast pattern misfires, heavy rainfall, or shovel mechanical breakdowns—immediately halts waste removal, which immediately starves the concentrator of ore feed. It trades off long-term operational resilience for short-term financial discounting.
- **Weak Answer:** The risk is that you have to spend more money on trucks at the end of the mine instead of at the beginning.

### Question 6: Under what conditions should a mine reclaim its low-grade stockpiles?
- **Strong Answer:** Low-grade stockpiles containing material above the marginal break-even cut-off ($g_{be}$) but below the dynamic operational cut-off ($g^*_t$) should be reclaimed under two conditions: (1) During the terminal phase of the mine life when the open pit is exhausted and remaining deposit value $W \approx 0$, driving opportunity cost $d \approx 0$; or (2) During operational emergencies where open-pit mining is temporarily restricted (e.g., major wall slide, severe wet season) to keep the capital-intensive concentrator operating at full capacity. Stockpile reclaim economics must verify that rehandle and milling costs are lower than current recovered metal revenues.
- **Weak Answer:** You reclaim stockpiles whenever commodity prices go up so you can sell more metal.

### Question 7: Explain the difference between statutory reserve reporting cut-offs and internal operational cut-offs under SEC S-K 1300.
- **Strong Answer:** Under SEC S-K 1300 and NI 43-101, public mineral reserves must establish "Reasonable Prospects for Eventual Economic Extraction" (RPEEE) using trailing commodity price ceilings and static break-even cut-offs that demonstrate economic viability across the entire declared reserve life. The statutory cut-off is legally binding for external corporate disclosure. Conversely, the internal operational cut-off is an active optimization parameter derived via Lane's theory to maximize operational NPV. Operating at a higher operational cut-off than the published reserve cut-off is standard practice, provided the un-milled marginal ore is stockpiled and accounted for as mineralized inventory.
- **Weak Answer:** SEC cut-offs are for the government, while operational cut-offs are what the mine manager writes on the whiteboard.

### Question 8: How does an increase in the corporate discount rate ($\delta$) alter the optimal cut-off grade and the life-of-mine schedule?
- **Strong Answer:** An increase in the discount rate $\delta$ directly inflates the corporate opportunity cost of time ($d = \delta W$). In Lane's formulation for concentrator and refinery bottlenecks, $g_c$ and $g_r$ are monotonically increasing functions of $F = f + \delta W$. Therefore, a higher discount rate forces the optimal cut-off grade upward during early production years. The mine high-grades aggressively, increasing head grade and metal production to generate front-loaded cash flows. More material is diverted to waste dumps or low-grade stockpiles, effectively shortening the active mining life of the open pit.
- **Weak Answer:** A higher discount rate makes everything more expensive, so you have to mine more tonnes to make up for the lost profit.

### Question 9: Why does high-grading without an engineered stockpiling plan permanently destroy mineral endowment?
- **Strong Answer:** When cut-off grade is raised to accelerate short-term cash flows, material between the break-even cut-off ($g_{be}$) and the elevated operational cut-off ($g^*$) is extracted. If this rock is dumped onto unsegregated waste dumps alongside barren country rock, it becomes physically and economically unrecoverable. Once co-mingled with waste and buried under hundreds of millions of tonnes of overburden, re-excavation costs exceed mineral value, and sulfide mineral exposure triggers acid generation. Strategic high-grading requires dedicated, surveyed, lined low-grade stockpile pads that preserve the material as a capital asset for late-life reclamation.
- **Weak Answer:** It's bad because geologists don't like seeing rock thrown away after they spent time drilling it.

### Question 10: How do geometallurgical rock hardness variations impact Lane's concentrator-limited cut-off grade?
- **Strong Answer:** In classical Lane formulations, concentrator capacity $C$ is treated as a constant scalar tonnage. In reality, rock comminution hardness (e.g., Semi-Autogenous Grinding Power Index, Bond Work Index) varies spatially across mineral alteration domains. If a high-grade block is exceptionally hard, treating it throttles mill throughput from $C$ down to $C' < C$. This drives up effective fixed and opportunity cost per tonne milled ($F / C'$), raising the concentrator cut-off grade $g_c$. Conversely, soft ore expands mill throughput, lowering $F/C$ and expanding the economic envelope to process lower-grade material profitably. Modern geometallurgical planning replaces scalar capacities with throughput-hardness response surfaces.
- **Weak Answer:** Hard rock breaks excavator teeth, which increases maintenance costs and changes the mine plan.

### Question 11: How does the Information Effect impact cut-off grade reconciliation between long-term planning and short-term blast-hole grade control?
- **Strong Answer:** The Information Effect stems from the spatial support difference between large exploration drilling blocks (where grades are smoothed by kriging) and actual blast muckpile selective mining units. Long-term planning models often over-predict tonnage and under-predict grade at high cut-offs due to dispersion smoothing ($\sigma^2(V \mid D) < \sigma^2(v \mid D)$). In short-term grade control, high-density blast-hole assays reveal localized high-grade pods and barren inclusions that the long-term model smeared together. If long-term plans fail to apply volume-variance corrections (e.g., affine or lognormal shortcut transformations), the mine reconciliation will exhibit systematic grade under-performance and unexpected tonnage surpluses at the primary crusher.
- **Weak Answer:** The information effect just means you know more about the rock when you dig it than when you drilled it years ago.

### Question 12: Under what conditions does In-Pit Crushing and Conveying (IPCC) constrain pushback flexibility compared to conventional shovel-truck fleets?
- **Strong Answer:** Fully mobile or semi-mobile IPCC systems replace diesel haul truck cycles with continuous high-capacity belt conveyors, slashing unit haulage operating costs ($m$). However, IPCC imposes severe geometric rigidity on pushback staging. Conventional truck haulage can easily re-route around steep ramps, unstable benches, or multiple simultaneous pushback phases. In contrast, an IPCC system requires dedicated linear bench conveyor corridors and transfer points that cannot be blasted or undercut while operating. This locks the mine into wider, flatter bench sinking geometries, restricting the planner's ability to execute opportunistic high-grading or sudden phase sequence adjustments during commodity price shifts.
- **Weak Answer:** Conveyors are bad because if the belt snaps, all work stops completely until someone fixes it.

---

## 15. Key terms

- **Block Valuation ($V_b$):** The net operating margin of a discretized 3D geological cell, assigned as ore or waste.
- **Ultimate Pit Limit (UPL):** The spatial boundary that maximizes undiscounted operating profit subject to geotechnical wall slopes.
- **Maximum Weight Closure:** A graph-theoretic optimization identifying a node subset with maximum sum of weights respecting directed precedence.
- **Picard's Theorem:** Mathematical equivalence proving the maximum closure problem can be solved as a minimum cut in an augmented $(s,t)$ network.
- **Revenue Factor ($\lambda$):** A parametric price multiplier used to generate nested pit shells for pushback sequencing.
- **Pushback (Phase):** An operational extraction slice within a pit satisfying equipment minimum mining widths and haul roads.
- **Minimum Mining Width (MMW):** The spatial clearance required for loading shovels and haul trucks to maneuver safely ($>40\text{ m}$).
- **Instantaneous Stripping Ratio ($SR_t$):** Periodic ratio of waste mass mined to ore mass delivered ($W_t / O_t$).
- **Break-Even Stripping Ratio ($BESR$):** The maximum economic limit of waste per unit ore before open-pit mining becomes uneconomic.
- **Declining Stripping Ratio:** Waste scheduling policy extracting waste immediately, carrying high early CAPEX but zero ore starvation risk.
- **Just-in-Time (JIT) Stripping:** Deferring waste stripping to maximize early NPV, introducing high operational vulnerability to wall slides.
- **Kenneth F. Lane's Theory:** Dynamic cut-off optimization balancing mining, milling, and refining bottlenecks against capital opportunity cost.
- **Opportunity Cost of Capital ($d = \delta W$):** The financial penalty of delaying remaining deposit cash flows by processing marginal ore.
- **Mining-Limited Cut-Off ($g_m$):** Cut-off grade maximizing profit when the mining fleet is the sole bottleneck; equals marginal break-even.
- **Concentrator-Limited Cut-Off ($g_c$):** Cut-off grade maximizing profit when the mill is constrained, carrying fixed and opportunity time costs.
- **Refinery-Limited Cut-Off ($g_r$):** Cut-off grade maximizing profit when downstream metal sales or refining quotas are binding.
- **Balancing Cut-Off ($g_{ij}$):** Cut-off grade where stage $i$ and stage $j$ throughputs operate simultaneously at 100% capacity.
- **Dynamic Declining Cut-Off:** Strategic schedule applying high cut-offs early in mine life and lowering them as reserves deplete.
- **Stockpile Rehandle:** Reclaiming stockpiled marginal ore during late-stage LOM when capital opportunity costs approach zero.
- **Break-Even Cut-Off ($g_{be}$):** Grade where recovered metal revenue equals operating milling and realization costs.
- **Internal Marginal Cut-Off ($g_{\text{marg}}$):** Grade threshold where mining cost is sunk and only incremental milling is tested.
- **Reasonable Prospects (RPEEE):** Statutory reporting criterion requiring proof of economic extraction for declared mineral resources.
- **JORC Code (2012):** Australasian code for public reporting of exploration results, mineral resources, and ore reserves.
- **SEC Regulation S-K 1300:** United States federal rules governing disclosure of mining properties and reserve estimates.
- **GISTM (2020):** Global Industry Standard on Tailings Management regulating structural stability and environmental monitoring.

---

## 16. Sources

- **Lane, K. F. (1988).** *The Economic Definition of Ore: Cut-Off Grades in Theory and Practice.* Mining Journal Books Ltd, London. ISBN: 978-0955283901. [Foundational text for dynamic cut-off theory].
- **Lerchs, H., & Grossmann, I. F. (1965).** Optimum design of open-pit mines. *CIM Bulletin*, 58(633), 47–54. [Original publication of the 3D graph closure algorithm].
- **Picard, J.-C. (1976).** Maximal closure of a graph and applications to combinatorial problems. *Management Science*, 22(11), 1268–1272. DOI: 10.1287/mnsc.22.11.1268. [Proof of equivalence between max closure and min cut].
- **Hustrulid, W., Kuchta, M., & Martin, R. (2013).** *Open Pit Mine Planning and Design* (3rd ed.). CRC Press / Balkema, Leiden. ISBN: 978-1466575127. [Standard reference for pushback design, slope templates, and MMW].
- **Whittle, J. (1988).** The Four-D Open Pit Optimization Package. In *Proceedings of the 21st APCOM Symposium*, Society for Mining, Metallurgy & Exploration (SME), Littleton, CO. [Parametric revenue factors and nested shell scheduling].
- **Hochbaum, D. S. (2008).** The Pseudoflow algorithm: A new algorithm for the maximum-flow problem. *Operations Research*, 56(4), 992–1009. DOI: 10.1287/opre.1080.0525. [Fast modern solver for ultimate pit limit closures].
- **Joint Ore Reserves Committee (JORC). (2012).** *Australasian Code for Reporting of Exploration Results, Mineral Resources and Ore Reserves (The JORC Code).* The JORC Committee, Melbourne. [Clauses 20 and 29].
- **Canadian Institute of Mining, Metallurgy and Petroleum (CIM). (2019).** *CIM Estimation of Mineral Resources & Mineral Reserves Best Practice Guidelines.* CIM Council, Montreal. [Standards for NI 43-101 technical reports].
- **U.S. Securities and Exchange Commission (SEC). (2018).** *Modernization of Property Disclosures for Mining Registrants.* Final Rule, 17 CFR Parts 229, 230, 239, and 249 (Regulation S-K Subpart 1300).
- **International Council on Mining and Metals (ICMM), UNEP, & PRI. (2020).** *Global Industry Standard on Tailings Management (GISTM).* UN Environment Programme.
- **Dagdelen, K. (2001).** Open pit optimization—who needs it? In *Proceedings of the 30th APCOM Symposium*, SME, Littleton, CO, pp. 705–715. [Critique of undiscounted ultimate pit limits vs. discounted dynamic sequencing].
- **King, B. (1999).** Optimal mining cut-off grades: Lane's methodology applied to complex multi-element deposits. *CIM Bulletin*, 92(1031), 89–94. [Multi-element extension of Lane's formulation].
- **BHP Group Limited. (2024).** *Minera Escondida Operational and Technical Review.* BHP Operational Reports, Melbourne. [Real-world multi-stream cut-off evidence].
- **Rio Tinto plc. (2014).** *The Bingham Canyon Slide: Geotechnical Investigation and Operational Recovery.* Rio Tinto Technical Disclosures, London.
- **BHP Group Limited. (2023).** *Olympic Dam Technical Briefing and Metallurgical Processing Review.* BHP Investor Presentations, Adelaide.

---

## 17. Links

- **Module Directory Index:** [`_index.md`](_index.md)
- **Previous Sibling Topic:** [`01_reserves-grade-confidence.md`](01_reserves-grade-confidence.md) — Reserves, Grade, and Confidence
- **Next Sibling Topic:** [`03_drilling-blasting-loadhaul.md`](03_drilling-blasting-loadhaul.md) — Drilling, Blasting, and Load-Haul
- **Downstream Processing Topic:** [`04_grade-blending.md`](04_grade-blending.md) — Grade Blending and Specification
- **Root Survey File:** [`../../../course/39_mining_metals_forestry.md`](../../../course/39_mining_metals_forestry.md)
- **Related Methods Module:** [Module 11: Network Design Optimization](../../02_methods/11_network_design_optimization/_index.md)
- **Related Operations Module:** [Module 20: Operations Planning and Master Scheduling](../../03_commerce/20_operations_planning_master_scheduling/_index.md)
