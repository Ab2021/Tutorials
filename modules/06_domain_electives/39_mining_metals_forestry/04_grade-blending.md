# Grade Blending and Specification: Multi-Element Optimization, Penalty Mechanics, and Bed Blending Theory

> Module 39 — Mining, Metals, and Forestry Supply Chains
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 (Reserves, Grade, and Confidence), Topic 02 (Mine Planning and Cut-Off Grade Optimization), Topic 03 (Drilling, Blasting, and Load-Haul Operations)
> Version 2026.5

---

## 1. Why this topic exists

In discrete assembly manufacturing, raw materials arrive from certified vendors according to standardized bills of materials with predictable engineering tolerances measured in parts per thousand. In petrochemistry, crude oil feedstocks arrive via pipeline with well-characterized assay curves, and refining hydrotreaters operate in steady-state regimes over months. Mining supply chains face the opposite operating condition: their primary input is an unmanufactured, spatially variable, geological formation excavated in discrete multi-thousand-tonne blast increments whose elemental grades, moisture contents, and metallurgical hardness fluctuate hourly by orders of magnitude.

Concentrators, hydrometallurgical leaching circuits, blast furnaces, and base metal smelters are continuous chemical reactors designed to operate within narrow thermodynamic, physical, and chemical windows. When feed grades deviate from design specifications:
1. **Flotation circuits lose recovery:** Low-grade feed starves the froth phase and collapses concentrate quality, while grade surges cause metal "slipping" into tailings because collector reagent dosing and cell residence times cannot adjust instantaneously to high-frequency spikes.
2. **Pyrometallurgical furnaces suffer refractory failure and slag freezing:** Blast furnaces require precise basicity ratios ($(\text{CaO} + \text{MgO}) / (\text{SiO}_2 + \text{Al}_2\text{O}_3)$) to control slag viscosity and melting point. Excess alumina ($\text{Al}_2\text{O}_3 > 2.5\%$) increases slag viscosity, trapping payable metal droplets and elevating tapping temperatures, while excess silica ($\text{SiO}_2$) consumes expensive limestone fluxes and accelerates furnace refractory erosion.
3. **Smelter contracts enforce steep financial penalties and rejection rights:** Commercial base metal concentrate contracts (copper, lead, zinc) do not evaluate deliveries on payable metal alone. Smelter contracts impose tiered penalty step functions on deleterious elements such as arsenic ($\text{As}$), bismuth ($\text{Bi}$), antimony ($\text{Sb}$), mercury ($\text{Hg}$), fluorine ($\text{F}$), and chlorine ($\text{Cl}$). Exceeding threshold limits triggers punitive per-tonne deductions that destroy operating margins; crossing rejection limits permits the smelter to refuse the shipment at the port of discharge, leaving the mining company liable for dead freight, ocean demurrage, and toxic waste disposal.
4. **Physical moisture triggers ship liquefaction and freight penalties:** Ore concentrates shipped with moisture contents exceeding the Transportable Moisture Limit (TML) liquefy under cyclic ocean swell, shifting cargo and capsizing dry bulk carriers. Conversely, under-drying concentrate creates airborne dusting during ship loading, triggering environmental fines. Furthermore, freight tariffs are billed on Wet Metric Tonnes (WMT), while commercial payable metal settlements are strictly priced on Dry Metric Tonnes (DMT). Transporting unblended high-moisture ore burns millions of dollars annually hauling non-payable water across rail and maritime corridors.

Grade blending is the primary operational mechanism that decouples heterogeneous upstream extraction from rigid downstream processing and market specifications. Blending is not a simple arithmetic averaging exercise. It is a multi-dimensional, constrained mathematical optimization problem executed across dynamic stockpile inventories, non-linear moisture and bulk density physics, physical bed stacking geometries, and real-time sensor feedback. Mastering multi-element blending, penalty linearization, bed blending theory, and online closed-loop trimming is essential for maximizing net realized metal value across the bulk supply chain.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Formulate and solve** multi-element grade blending Linear Programs (LP) and Mixed-Integer Linear Programs (MILP) that balance primary payable metals against deleterious impurity thresholds, processing throughput limits, and stockpile balances.
2. **Linearize** non-linear, discontinuous smelter contract penalty step functions and rejection thresholds using binary indicator variables and Special Ordered Sets of Type 2 (SOS2).
3. **Execute** mass and volumetric reconciliations between Wet Metric Tonnes (WMT) and Dry Metric Tonnes (DMT), proving why linear moisture blending assumptions introduce systematic bias in grade optimization.
4. **Evaluate and design** longitudinal bed blending stockpiles, comparing Chevron and Windrow stacking methods against material segregation mechanics, angle of repose dynamics, and reclaimer geometry.
5. **Calculate** stockpile variance reduction factors using Pierre Gy's geostatistical sampling and blending theory, deriving the minimum number of layers ($N_L$) required to suppress feed grade volatility below downstream tolerance limits.
6. **Design** closed-loop quality control architectures that integrate real-time Prompt Gamma Neutron Activation Analysis (PGNAA) cross-belt analyzers with dynamic trim feeder hoppers to eliminate high-frequency grade disturbances.
7. **Quantify** the financial trade-off between the capital expenditure of automated bed blending yards and the operating value captured through penalty avoidance, flux minimization, and premium pricing.

---

## 3. The core mental model

The grade blending supply chain operates as a continuous quality filter that transforms high-variance, discrete ore extractions from open-pit benches and underground stopes into an ultra-low-variance, tightly constrained feed stream for mineral processing plants, rail loadouts, and export vessels.

### 3.1 The Multi-Stage Homogenization Architecture

Grade variance reduction occurs across four sequential stages, each targeting a specific frequency band of geological uncertainty:

```text
=============================================================================================================
                       CLOSED-LOOP GRADE BLENDING AND HOMOGENIZATION ARCHITECTURE
=============================================================================================================

 [ MINE BENCHES / STOPES ] ──> Discrete Blasts, High Spatial Variance (Hours to Days)
            │
            ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 1: RUN-OF-MINE (ROM) FINGER STOCKPILES & MULTI-FEEDER DISPATCH                                    │
 │ - Multi-element Linear Programming (LP) optimization                                                    │
 │ - Tonnage allocation across high-grade, medium-grade, low-grade, and penalty fingers                    │
 └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
            │ Primary Crushing (P80 < 150 mm)
            ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 2: LONGITUDINAL BED BLENDING YARD (MACRO-HOMOGENIZATION)                                          │
 │ - Stacking: Chevron (triangular apex) or Windrow (multi-ridge segregation suppression)                  │
 │ - NL layers (100 to 500 layers) deposited longitudinally                                                │
 │ - Variance reduction: Var(out) = Var(in) / (2 * NL) * Gy_autocorrelation_factor                        │
 │ - Reclaiming: Transversal face slices via bridge bucket-wheel or barrel reclaimer                       │
 └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
            │ Reclaimed Stream (Hours frequency variance dampened)
            ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ STAGE 3: REAL-TIME SENSING & DYNAMIC TRIM HOPPERS (MICRO-HOMOGENIZATION)                                │
 │ - Cross-Belt PGNAA Analyzer: Thermal neutron capture measuring full elemental spectrum (1-minute scan)  │
 │ - Closed-Loop Control: Feedforward / Feedback PID & Model Predictive Control (MPC)                      │
 │ - Dynamic Trimming: High-grade/low-impurity surge hoppers adjust variable-speed feeders (t/h)          │
 └─────────────────────────────────────────────────────────────────────────────────────────────────────────┘
            │
            ├───────────────────────────────────────────────┬─────────────────────────────────────────────┐
            ▼                                               ▼                                             ▼
  [ BENEFICIATION PLANT ]                         [ TRAIN LOADOUT ]                             [ PORT TERMINAL ]
  - Stable froth kinetics                         - Contract spec compliant                     - Vessel blend composite
  - Minimum flux consumption                      - Zero rail dead-freight                      - Demurrage & penalty-free
=============================================================================================================
```

### 3.2 The Fundamental Trade-off

The operational objective of blending is to maximize the Net Realized Value (NRV) per dry metric tonne of ore mined:

$$\max \quad \text{NRV} = R_{\text{payable}} - C_{\text{mining}} - C_{\text{haul}} - C_{\text{process}} - C_{\text{freight}} - P_{\text{penalties}} - C_{\text{blending}}$$

Where:
- $R_{\text{payable}}$ is the gross commercial revenue from payable metals recovered.
- $P_{\text{penalties}}$ is the non-linear penalty deduction incurred from deleterious elements.
- $C_{\text{blending}}$ is the sum of rehandling costs, stockpile pad depreciation, stacker-reclaimer power consumption, and working capital inventory interest.

If an enterprise under-invests in blending ($C_{\text{blending}} \to 0$), incoming grade variance surges. The downstream processing plant suffers recovery losses, consumption of corrective reagents escalates, and finished products breach contract limits, triggering massive penalties ($P_{\text{penalties}} \gg 0$) or rejection. Conversely, over-investing in blending through infinite stockpile capacity incurs excessive rehandling costs, inventory degradation, and working capital lockup. Optimal blending aligns stockpile mechanics and optimization algorithms to minimize the sum of variance penalties and handling costs.

---

## 4. Definitions and vocabulary

| Term | Definition | Common Misuse | Unit / Code |
|---|---|---|---|
| **Dry Metric Tonne (DMT)** | Mass of mineral ore or concentrate excluding all uncombined surface and interstitial water after drying at $105^\circ\text{C}$ to constant weight. | Confusing with Wet Metric Tonnes (WMT) when calculating metal pricing, leading to phantom revenue projections. | Metric Tonne ($\text{t}_{\text{dry}}$) |
| **Wet Metric Tonne (WMT)** | As-received physical mass of ore including free moisture; the physical mass weighed on belt scales and rail scales. | Using WMT to calculate chemical grade percentages without moisture adjustments. | Metric Tonne ($\text{t}_{\text{wet}}$) |
| **Transportable Moisture Limit (TML)** | The maximum gross moisture content at which a granular bulk cargo may be safely loaded into a bulk carrier without risk of liquefying under voyage vibration. | Treating TML as a commercial guideline rather than a legally binding maritime safety limit under the IMO IMSBC Code. | Weight Percentage ($\% \text{ w/w}$) |
| **Deleterious Element** | Any non-payable chemical element (e.g., $\text{As}, \text{Sb}, \text{Bi}, \text{Hg}, \text{Cd}, \text{F}, \text{Cl}$) that impairs downstream smelting, corrodes equipment, or creates environmental hazards. | Assuming low concentrations of impurities are harmless; many elements trigger penalties at parts per million (ppm). | $\%$, $\text{ppm}$, or $\text{g/t}$ |
| **Treatment & Refining Charges (TC/RC)** | Fees paid by the miner to a smelter/refiner: TC is charged per DMT of concentrate smelted; RC is charged per pound or ounce of payable metal refined. | Treating TC/RC as a fixed flat fee rather than a rate indexed to price benchmarks and impurity escalation clauses. | $\text{USD/DMT}$ (TC), $\text{US\text{c}/lb}$ or $\text{USD/oz}$ (RC) |
| **Penalty Step Function** | A non-linear, discontinuous pricing structure where smelters levy incremental dollar penalties per unit of impurity above predefined threshold tiers. | Modeling penalty structures as smooth linear functions in LP models, which underestimates threshold tipping points. | $\text{USD/DMT}$ per unit impurity |
| **Chevron Stacking** | A stockpile bedding method where a traveling stacker deposits mineral layers along a fixed central longitudinal axis, forming a triangular prism cross-section. | Applying Chevron stacking to materials with extreme size distributions without accounting for coarse particle toe roll. | Geometric method |
| **Windrow Stacking** | A bedding method where the stacker deposits multiple parallel, small triangular rows across the width of the yard before building higher tiers. | Assuming Windrow stacking has the same capital cost as Chevron; Windrow requires complex boom luffing and slewing kinematics. | Geometric method |
| **Bed Blending Ratio ($r$)** | The ratio of the variance of the reclaimed output grade to the variance of the stacked input grade ($\sigma_{\text{out}}^2 / \sigma_{\text{in}}^2$). | Assuming the variance reduction factor is strictly $1/N_L$; real beds are governed by spatial autocorrelation and Gy's theory. | Dimensionless ($0 < r < 1$) |
| **Prompt Gamma Neutron Activation Analysis (PGNAA)** | An online sensor technique that bombards cross-belt material with thermal neutrons and detects characteristic prompt gamma rays emitted upon neutron capture. | Confusing PGNAA (which penetrates the entire rock cross-section) with surface-only X-ray Fluorescence (XRF). | Sensor technology |
| **Special Ordered Set of Type 2 (SOS2)** | An ordered set of variables where at most two adjacent variables can be strictly positive; used in MILP to linearize piecewise continuous functions. | Using standard SOS1 sets for continuous piecewise functions, which enforces point discontinuities. | Optimization construct |
| **Basicity Ratio ($B_2$ or $B_4$)** | Thermodynamic ratio of basic oxides to acidic oxides in metallurgical slag ($B_2 = \text{CaO} / \text{SiO}_2$; $B_4 = (\text{CaO} + \text{MgO}) / (\text{SiO}_2 + \text{Al}_2\text{O}_3)$). | Blending for iron grade while ignoring basicity, resulting in furnace slag freezing or high flux reagent costs. | Dimensionless ratio |
| **Angle of Repose** | The steepest angle of descent relative to the horizontal plane to which bulk granular material can be piled without slumping (typically $34^\circ - 38^\circ$). | Assuming angle of repose is independent of moisture and particle size distribution. | Degrees ($^\circ$) |
| **Reclaimer Rake** | An oscillating harrow mechanism mounted on a bridge reclaimer that sweeps across the triangular face of a stockpile bed to induce continuous, uniform avalanching. | Operating a reclaimer without a rake on segregated Chevron beds, resulting in cyclical high-frequency grade waves. | Mechanical component |

---

## 5. The mechanics, step by step

Implementing an industrial-grade multi-element blending and specification system requires seven rigorous engineering steps, progressing from physical characterization to dynamic closed-loop control.

```text
=============================================================================================================
                       SEVEN-STEP GRADE BLENDING & SPECIFICATION METHODOLOGY
=============================================================================================================
 [Step 1] Multi-Element Feedstock Characterization (DMT, WMT, SG, Deleterious Assays, Hardness)
    │
    ▼
 [Step 2] Non-Linear Moisture & Volume Reconciliations (Dry-Wet Mass Conservation & Swell Mechanics)
    │
    ▼
 [Step 3] Multi-Element Blending LP Formulation (Mass Balance, Grade Bounds, Basicity Ratios)
    │
    ▼
 [Step 4] Smelter Penalty Linearization via MILP (SOS2 Formulation of Escalating Step Penalties)
    │
    ▼
 [Step 5] Stockpile Bed Blending Geometry Design (Chevron vs Windrow Kinematics & Segregation Control)
    │
    ▼
 [Step 6] Geostatistical Variance Reduction Sizing (Pierre Gy Variogram Integral & NL Determination)
    │
    ▼
 [Step 7] PGNAA Online Closed-Loop Trimming Control (Real-Time Radiometric Sensing & Feedforward Trimming)
=============================================================================================================
```

---

### Step 1: Multi-Element Feedstock Characterization

Every potential ore parcel available for blending must be profiled across physical, chemical, and metallurgical dimensions. Let $\mathcal{I} = \{1, 2, \dots, I\}$ represent the set of available feed sources (open-pit blast benches, underground stope drawpoints, low-grade ROM surge fingers, or port storage pads).

For each source $i \in \mathcal{I}$, capture:
1. **Available Wet Tonnage:** $W_i$ (WMT).
2. **Moisture Content:** $m_i \in [0, 1)$, defined as:
   $$m_i = \frac{M_{\text{water}, i}}{M_{\text{total\_wet}, i}} = \frac{W_i - D_i}{W_i}$$
   where $D_i$ is the Dry Metric Tonnage.
3. **Primary Payable Grades:** $g_{i, p}$ for $p \in \mathcal{P}$ (e.g., $\% \text{Cu}, \% \text{Fe}, \text{g/t Au}, \text{g/t Ag}$).
4. **Deleterious Impurity Concentrations:** $g_{i, d}$ for $d \in \mathcal{D}$ (e.g., $\text{ppm As}, \text{ppm Bi}, \% \text{SiO}_2, \% \text{Al}_2\text{O}_3, \% \text{P}, \% \text{S}$).
5. **Comminution & Metallurgical Parameters:** Bond Work Index ($\text{BWi}_i$, measured in $\text{kWh/t}$), SAG Power Index ($\text{SPI}_i$, in minutes), and kinetic flotation recovery potential ($R_{i, p}$, in $\%$).
6. **In-situ and Piled Bulk Densities:** In-situ dry density $\rho_{i, \text{insitu}}$ ($\text{t/m}^3$) and loose bulk density $\rho_{i, \text{bulk}}$ ($\text{t/m}^3$).

---

### Step 2: Non-Linear Moisture and Volume Reconciliations

A pervasive error in mining logistics is blending grades linearly on a Wet Metric Tonne basis. Because metal content is fixed within the dry mineral matrix, blending ores with unequal moisture contents creates an intrinsic non-linearity.

#### 2.1 Dry Mass Transformation
Given a decision variable $x_i^{\text{wet}}$ representing the wet mass (WMT) pulled from source $i$, the corresponding dry mass $x_i^{\text{dry}}$ (DMT) is:

$$x_i^{\text{dry}} = x_i^{\text{wet}} \cdot (1 - m_i)$$

The total blended dry mass $X^{\text{dry}}$ and wet mass $X^{\text{wet}}$ satisfy:

$$X^{\text{dry}} = \sum_{i \in \mathcal{I}} x_i^{\text{wet}} (1 - m_i), \qquad X^{\text{wet}} = \sum_{i \in \mathcal{I}} x_i^{\text{wet}}$$

The composite blended moisture fraction $M_{\text{blend}}$ is non-linear with respect to wet mass ratios:

$$M_{\text{blend}} = \frac{\sum_{i \in \mathcal{I}} x_i^{\text{wet}} m_i}{\sum_{i \in \mathcal{I}} x_i^{\text{wet}}} = 1 - \frac{\sum_{i \in \mathcal{I}} x_i^{\text{dry}}}{\sum_{i \in \mathcal{I}} \frac{x_i^{\text{dry}}}{1 - m_i}}$$

#### 2.2 Volumetric Stockpile Swell Mechanics
When ore is excavated and stacked, its volume expands by the Swell Factor ($S_i \ge 1.0$), reducing bulk density:

$$V_i = \frac{x_i^{\text{wet}}}{\rho_{i, \text{bulk}}} = \frac{x_i^{\text{dry}}}{(1 - m_i) \cdot \rho_{i, \text{bulk}}}$$

If a stockyard blending bed has a maximum physical volumetric capacity $V_{\max}$ ($\text{m}^3$), the volumetric constraint is:

$$\sum_{i \in \mathcal{I}} \frac{x_i^{\text{dry}}}{(1 - m_i) \cdot \rho_{i, \text{bulk}}} \le V_{\max}$$

Ignoring moisture and bulk density causes stockpile overfilling, bench toe impingement, and equipment collisions.

---

### Step 3: Formulating the Multi-Element Blending LP

The core blending model determines the optimal dry mass allocation $x_i^{\text{dry}} \ge 0$ drawn from each source $i \in \mathcal{I}$ over a planning horizon $t$.

#### 3.1 Objective Function
Maximize the net commercial margin across payable metals, mining costs, haulage costs, processing costs, and penalty deductions:

$$\max_{x^{\text{dry}}} \quad Z = \sum_{p \in \mathcal{P}} \text{Price}_p \cdot \sum_{i \in \mathcal{I}} g_{i, p} x_i^{\text{dry}} \cdot \text{Rec}_p - \sum_{i \in \mathcal{I}} (c_{i, \text{mine}} + c_{i, \text{haul}} + c_{\text{proc}}) x_i^{\text{dry}} - \text{Penalties}(x^{\text{dry}})$$

#### 3.2 Mass Conservation and Source Availability
For each source $i$, the allocated dry tonnage cannot exceed the available inventory $D_i^{\max}$:

$$0 \le x_i^{\text{dry}} \le D_i^{\max} \quad \forall i \in \mathcal{I}$$

The blended campaign must hit a targeted total dry throughput $T^{\text{target}}$:

$$\sum_{i \in \mathcal{I}} x_i^{\text{dry}} = T^{\text{target}}$$

#### 3.3 Multi-Element Grade Constraints
The blended grade for any element $e \in \mathcal{P} \cup \mathcal{D}$ is the dry-mass weighted average:

$$G_e(x^{\text{dry}}) = \frac{\sum_{i \in \mathcal{I}} g_{i, e} x_i^{\text{dry}}}{\sum_{i \in \mathcal{I}} x_i^{\text{dry}}} = \frac{\sum_{i \in \mathcal{I}} g_{i, e} x_i^{\text{dry}}}{T^{\text{target}}}$$

To enforce lower bounds $G_e^{\min}$ and upper bounds $G_e^{\max}$:

$$G_e^{\min} \le \frac{\sum_{i \in \mathcal{I}} g_{i, e} x_i^{\text{dry}}}{T^{\text{target}}} \le G_e^{\max}$$

Multiplying through by $T^{\text{target}} = \sum_{i \in \mathcal{I}} x_i^{\text{dry}}$ yields exact linear inequalities:

$$\sum_{i \in \mathcal{I}} (g_{i, e} - G_e^{\min}) x_i^{\text{dry}} \ge 0 \quad \forall e \in \mathcal{P} \cup \mathcal{D}$$

$$\sum_{i \in \mathcal{I}} (g_{i, e} - G_e^{\max}) x_i^{\text{dry}} \le 0 \quad \forall e \in \mathcal{P} \cup \mathcal{D}$$

#### 3.4 Slag Basicity and Ratio Constraints
Pyrometallurgical smelters and blast furnaces require strict chemical ratios between gangue oxides. For example, maintaining a silica-to-alumina ratio between $R_{\min}$ and $R_{\max}$:

$$R_{\min} \le \frac{\sum_{i \in \mathcal{I}} g_{i, \text{SiO}_2} x_i^{\text{dry}}}{\sum_{i \in \mathcal{I}} g_{i, \text{Al}_2\text{O}_3} x_i^{\text{dry}}} \le R_{\max}$$

Linearized as:

$$\sum_{i \in \mathcal{I}} (g_{i, \text{SiO}_2} - R_{\min} g_{i, \text{Al}_2\text{O}_3}) x_i^{\text{dry}} \ge 0$$

$$\sum_{i \in \mathcal{I}} (g_{i, \text{SiO}_2} - R_{\max} g_{i, \text{Al}_2\text{O}_3}) x_i^{\text{dry}} \le 0$$

Similarly, for the quaternary basicity index $B_4$:

$$\frac{\sum_{i} (g_{i, \text{CaO}} + g_{i, \text{MgO}}) x_i^{\text{dry}}}{\sum_{i} (g_{i, \text{SiO}_2} + g_{i, \text{Al}_2\text{O}_3}) x_i^{\text{dry}}} \ge B_4^{\min} \implies \sum_{i \in \mathcal{I}} \left[ (g_{i, \text{CaO}} + g_{i, \text{MgO}}) - B_4^{\min} (g_{i, \text{SiO}_2} + g_{i, \text{Al}_2\text{O}_3}) \right] x_i^{\text{dry}} \ge 0$$

---

### Step 4: Smelter Contract Penalty Structures and Mathematical Linearization

Smelter contracts do not charge linear penalty rates. They utilize tiered, escalating step functions with rejection cutoffs.

```text
=============================================================================================================
                       SMELTER PENALTY STEP FUNCTION WITH REJECTION THRESHOLD
=============================================================================================================
 Penalty ($/DMT)
        ▲
        │                                                     X [REJECTION REGIME: Grade > Treject]
        │                                                 ┌───  Shipment refused, full demurrage & return
        │                                                 │     costs charged to miner.
 Tier 2 │                                   ┌─────────────┘
  Rate  │                                   │ (Slope = p2, e.g., $5.00 per 100 ppm)
        │                      ┌────────────┘
 Tier 1 │                      │ (Slope = p1, e.g., $2.50 per 100 ppm)
  Rate  │         ─────────────┘
        │         (Penalty Free: Grade <= T0)
     $0 └─────────┴────────────┴────────────┴─────────────────► Impurity Grade (e.g., As ppm)
                  0           T0           T1                Treject
=============================================================================================================
```

#### 4.1 Formal Definition of Piecewise Penalty Function
Let $g_d$ be the final blended concentration of deleterious element $d$. The penalty per dry tonne $\Phi_d(g_d)$ is defined across $S$ tiers:

$$\Phi_d(g_d) = \begin{cases} 
0 & \text{if } 0 \le g_d \le T_0 \\
p_1 (g_d - T_0) & \text{if } T_0 < g_d \le T_1 \\
p_1 (T_1 - T_0) + p_2 (g_d - T_1) & \text{if } T_1 < g_d \le T_2 \\
\infty \text{ (Hard Infeasibility)} & \text{if } g_d > T_{\text{reject}}
\end{cases}$$

Where $p_s$ is the marginal penalty rate in tier $s$, with $p_{s+1} > p_s$ (convex escalation).

#### 4.2 Linearization via Special Ordered Sets of Type 2 (SOS2)
Because the penalty function is continuous and convex (increasing marginal penalty rates), it can be linearized using SOS2 formulations without introducing binary variables, or via explicit piecewise linear segments.

Define grid points $k \in \{0, 1, 2, \dots, K\}$ corresponding to thresholds $T_0, T_1, \dots, T_K$, where $T_K = T_{\text{reject}}$. At each grid point, let the cumulative unit penalty be $C_k = \Phi_d(T_k)$.

Introduce continuous weighting variables $\lambda_k \ge 0$ satisfying:

$$\sum_{k=0}^K \lambda_k = 1$$

$$\sum_{i \in \mathcal{I}} g_{i, d} x_i^{\text{dry}} = T^{\text{target}} \sum_{k=0}^K \lambda_k T_k$$

The total penalty cost incurred across the campaign is:

$$\text{Penalty}_d = T^{\text{target}} \sum_{k=0}^K \lambda_k C_k$$

Where $\{\lambda_0, \lambda_1, \dots, \lambda_K\}$ is declared as an **SOS2 set**, restricting at most two adjacent $\lambda_k$ variables to be non-zero. If the penalty curve is strictly convex ($p_{s+1} \ge p_s$), the LP relaxation automatically enforces adjacency, solving in polynomial time without branch-and-bound overhead.

If the contract contains non-convex penalties (e.g., a flat administrative fee triggered the moment $g_d > T_0$), binary indicator variables $z_s \in \{0, 1\}$ are mandatory:

$$g_d - T_0 \le M z_1, \qquad \text{AdminFee} \ge F \cdot z_1$$

Where $M$ is a big-M upper bound ($T_{\text{reject}} - T_0$).

---

### Step 5: Stockpile Bed Blending Theory and Stacking Geometry

Once the LP determines the multi-element blend, the ore is routed to an automated bed blending stockyard. Homogenization is achieved by stacking ore in hundreds of thin longitudinal layers and reclaiming it perpendicularly across all layers simultaneously.

#### 5.1 Chevron vs. Windrow Stacking Kinematics

```text
=============================================================================================================
                       CHEVRON VS. WINDROW STACKING GEOMETRY
=============================================================================================================

          CHEVRON STACKING METHOD                         WINDROW STACKING METHOD
       (Single Central Stacker Axis)               (Multi-Pass Luffing & Slewing Stacker)

                    /\                                          /\      /\      /\
                   /  \                                        /  \    /  \    /  \
                  / /\ \                                      / /\ \  / /\ \  / /\ \
                 / /  \ \                                    / /  \ \/ /  \ \/ /  \ \
                / / /\ \ \                                  / / /\ \  / /\ \  / /\ \ \
               /_ /_  _\ _\                                /_ /_  _\_/_/  \_\/_ /_  _\_\
              ─────────────                               ─────────────────────────────
              Toe        Toe                              Toe                       Toe
  [ Severe Coarse Radial Segregation ]            [ Suppressed Segregation; High Mechanical Cost ]
=============================================================================================================
```

#### 5.2 Particle Segregation Mechanics
Granular materials stacked on piles segregate according to particle mass and friction coefficients:
- **Kinetic Sieving:** As bulk ore cascades down the pile's slope, smaller particles fall through voids between larger rocks, settling near the apex.
- **Trajectory and Momentum:** Large, dense boulders ($+100\text{ mm}$) possess higher kinetic energy and roll down to the outer toes of the bed.

In **Chevron Stacking**:
- A single boom travels continuously back and forth along the pile length, discharging ore at the apex.
- Cross-section consists of concentric triangular layers.
- **Advantage:** Simplest stacker kinematics (single linear travel axis, no slewing required); lowest capital cost.
- **Disadvantage:** Severe radial segregation. The core is ultra-fine; the outer toes are coarse. If reclaimed with a front-end loader or bucket-wheel excavator without a full-face harrow, severe periodic grade and hardness swings occur.

In **Windrow Stacking**:
- The stacker utilizes a slewing and luffing boom to lay down dozens of parallel triangular rows across the width of the yard before stepping up to higher tiers.
- Cross-section consists of a cellular honeycomb matrix of small windrows.
- **Advantage:** Minimizes segregation roll-down distances to small local sub-piles. Fines and coarse fractions remain uniformly dispersed across the bed cross-section.
- **Disadvantage:** Stacker requires complex multi-axis servo positioning, continuous boom slewing/luffing, and higher electrical and mechanical maintenance.

---

### Step 6: Geostatistical Variance Reduction Sizing (Pierre Gy's Theory)

The performance of a bed blending yard is quantified by the **Variance Reduction Factor** ($r$):

$$r = \frac{\sigma_{\text{out}}^2}{\sigma_{\text{in}}^2} \in (0, 1)$$

Where $\sigma_{\text{in}}^2$ is the grade variance of incoming ore increments (time scale $\Delta t$), and $\sigma_{\text{out}}^2$ is the grade variance of the reclaimed ore stream.

#### 6.1 Ideal Random Superposition
If incoming ore increments were independent and identically distributed (white noise, zero spatial autocorrelation), stacking $N_L$ layers of equal thickness and reclaiming them in a single slice would yield:

$$\sigma_{\text{out}}^2 = \frac{\sigma_{\text{in}}^2}{N_L} \implies r_{\text{ideal}} = \frac{1}{N_L}$$

For a two-way Chevron pile (where a complete pass involves stacking from South to North, then North to South), each physical layer contains two passes:

$$r_{\text{ideal, 2-way}} = \frac{1}{2 N_L}$$

#### 6.2 Autocorrelation and Gy's Variogram Integral
Geological ore is never independent white noise; it exhibits spatial autocorrelation governed by the semi-variogram $\gamma(h)$. Pierre Gy demonstrated that autocorrelation impairs bed blending efficiency. Long-range grade trends pass directly through the bed if the stacker cycle time synchronizes with periodic mining cycles.

Gy's variance reduction factor accounting for spatial autocorrelation along the pile length $L$ is:

$$r = \frac{1}{N_L} + \left(1 - \frac{1}{N_L}\right) \cdot \frac{1}{L^2} \int_0^L \int_0^L \rho(|u - v|) \, du \, dv$$

Where $\rho(h) = 1 - \frac{\gamma(h)}{C_0 + C}$ is the autocorrelation function of the incoming ore stream at lag distance $h$, and $C_0 + C$ is the total geostatistical sill.

For a spherical variogram with range $a$ and zero nugget ($C_0 = 0$):

$$\gamma(h) = \begin{cases} 
C \left( \frac{3 h}{2 a} - \frac{h^3}{2 a^3} \right) & \text{for } h \le a \\
C & \text{for } h > a 
\end{cases}$$

When the pile length $L$ exceeds the variogram range $a$ ($L \ge a$), the correlation integral collapses to:

$$\frac{1}{L^2} \int_0^L \int_0^L \rho(|u - v|) \, du \, dv \approx \frac{3 a}{8 L}$$

Thus, the realistic variance reduction factor for a longitudinal bed is:

$$r \approx \frac{1}{N_L} + \left(1 - \frac{1}{N_L}\right) \frac{3 a}{8 L}$$

#### 6.3 Calculating Required Number of Layers ($N_L$)
Given a process engineering requirement that outgoing feed grade standard deviation cannot exceed $\sigma_{\text{spec}}$, the required number of layers $N_L$ is solved directly:

$$\sigma_{\text{out}}^2 = r \cdot \sigma_{\text{in}}^2 \le \sigma_{\text{spec}}^2$$

$$\frac{1}{N_L} \left( 1 - \frac{3 a}{8 L} \right) + \frac{3 a}{8 L} \le \frac{\sigma_{\text{spec}}^2}{\sigma_{\text{in}}^2}$$

$$N_L \ge \frac{1 - \frac{3 a}{8 L}}{\frac{\sigma_{\text{spec}}^2}{\sigma_{\text{in}}^2} - \frac{3 a}{8 L}}$$

If $\frac{\sigma_{\text{spec}}^2}{\sigma_{\text{in}}^2} \le \frac{3 a}{8 L}$, no amount of physical layering can achieve the target specification without increasing bed length $L$ or pre-blending upstream.

#### 6.4 Transversal Face Slice Kinematics
A bridge-type bucket wheel reclaimer or barrel reclaimer advances along the longitudinal track at speed $v_{\text{adv}}$ ($\text{m/min}$). The reclaimer's oscillating rake sweeps across the triangular face at angle $\theta$ (equal to the material angle of repose, $\sim 36^\circ$).

The volumetric reclaim rate $Q$ ($\text{m}^3/\text{h}$) is:

$$Q = 60 \cdot v_{\text{adv}} \cdot A_{\text{face}}$$

Where $A_{\text{face}} = \frac{1}{2} B_{\text{bed}} H_{\text{bed}}$ is the cross-sectional area of the triangular prism. The bucket wheel traverses back and forth across the base, collecting the uniformly avalanched blend of all $N_L$ layers simultaneously.

---

### Step 7: Online PGNAA Sensing and Dynamic Closed-Loop Trimming

Even a 400-layer blending bed leaves high-frequency residual grade variance. Furthermore, the final 5–10% of a bed (the "end cones") cannot achieve full layered superposition due to pile slope geometry. To eliminate these residual disturbances, industrial systems employ **Prompt Gamma Neutron Activation Analysis (PGNAA)** coupled with **Dynamic Trim Surge Bins**.

```text
=============================================================================================================
                       PGNAA CLOSED-LOOP FEEDFORWARD DYNAMIC TRIMMING CIRCUIT
=============================================================================================================
   Bulk Reclaimed Ore Stream (e.g., 2,500 t/h nominal, fluctuating grade)
  ───────────────────────────────────────┬───────────────────────────────────────────► To Concentrator
                                         │
                                   ┌─────▼─────┐
                                   │   PGNAA   │ Real-time multi-element assay
                                   │  ANALYZER │ (Cu, Fe, As, SiO2, Al2O3, Moisture)
                                   └─────┬─────┘ Every 60 seconds
                                         │
                                         ▼
                                   ┌───────────┐
                                   │  PROCESS  │ Model Predictive Controller (MPC)
                                   │ CONTROL / │ Error = Target_Grade - Measured_Grade
                                   │    PLC    │ Computes Trim Feed Rate (0 - 300 t/h)
                                   └─────┬─────┘
                                         │
                    Variable Frequency   │ Drive (VFD) Control Signal
                                         ▼
                                ┌─────────────────┐
                                │ TRIM SURGE BIN  │ Low-impurity or ultra-high-grade
                                │ & FEEDER HOPPER │ sweetening agent
                                └────────┬────────┘
                                         │
                                         ▼ Controlled Trim Stream (u_trim t/h)
  ───────────────────────────────────────┴───────────────────────────────────────────► Final Target Blend
=============================================================================================================
```

#### 7.1 PGNAA Radiometric Physics
The cross-belt analyzer mounts beneath and above the primary transfer conveyor:
1. **Neutron Source:** A radioactive isotope ($^{252}\text{Cf}$, half-life 2.64 years) or electronic Deuterium-Tritium (D-T) neutron generator produces thermalized neutrons ($E_n \approx 0.025\text{ eV}$).
2. **Neutron Capture:** Neutrons penetrate through the entire ore profile on the moving belt (depths up to $300\text{ mm}$), colliding with atomic nuclei. Nuclei capture neutrons and instantly emit high-energy prompt gamma rays ($\gamma$, energy 0.1 to 10 MeV).
3. **Scintillation Spectroscopy:** Bismuth Germanate (BGO) or Sodium Iodide (NaI) detectors record gamma photon counts per energy channel. Each element has an unmistakable spectral signature:
   - Iron ($\text{Fe}$): Dominant prompt gamma peaks at $7.63\text{ MeV}$ and $7.64\text{ MeV}$.
   - Copper ($\text{Cu}$): Characteristic gamma lines at $7.91\text{ MeV}$ and $7.64\text{ MeV}$.
   - Silicon ($\text{Si}$): Prompt gamma emission at $4.93\text{ MeV}$ and $3.54\text{ MeV}$.
   - Chlorine ($\text{Cl}$): Powerful gamma response at $6.11\text{ MeV}$.
   - Arsenic ($\text{As}$): Prompt capture gamma ray at $7.01\text{ MeV}$.
4. **Volume Penetration Advantage:** Unlike X-ray Fluorescence (XRF), which only measures a $20\text{-micrometer}$ skin depth and fails when dust or moisture coats particles, PGNAA samples the complete cross-sectional volume of the conveyor belt at tonnages up to $10,000\text{ t/h}$.

#### 7.2 Dynamic Trimming Closed-Loop Control Algorithm
Let $Q_{\text{main}}(t)$ be the mass flow rate ($\text{t/h}$) of ore reclaimed from the main stockpile, and let $g_{\text{main}, e}(t)$ be its real-time elemental grade measured by the PGNAA analyzer at time $t$.

An auxiliary trim surge hopper contains a known "corrective sweetening ore" with grade $g_{\text{trim}, e}$. A Variable Frequency Drive (VFD) controls the trim feeder discharge rate $u_{\text{trim}}(t) \in [0, u_{\max}]$.

The combined mass flow rate $Q_{\text{final}}(t)$ and combined grade $g_{\text{final}, e}(t)$ entering the plant are:

$$Q_{\text{final}}(t) = Q_{\text{main}}(t) + u_{\text{trim}}(t)$$

$$g_{\text{final}, e}(t) = \frac{Q_{\text{main}}(t) g_{\text{main}, e}(t) + u_{\text{trim}}(t) g_{\text{trim}, e}}{Q_{\text{main}}(t) + u_{\text{trim}}(t)}$$

To hit a target setpoint $g_e^*$, solve for the required instantaneous trim rate $u_{\text{trim}}^*(t)$:

$$g_e^* (Q_{\text{main}}(t) + u_{\text{trim}}^*(t)) = Q_{\text{main}}(t) g_{\text{main}, e}(t) + u_{\text{trim}}^*(t) g_{\text{trim}, e}$$

$$u_{\text{trim}}^*(t) = Q_{\text{main}}(t) \cdot \frac{g_e^* - g_{\text{main}, e}(t)}{g_{\text{trim}, e} - g_e^*}$$

Because physical feeder actuators experience transport lag $\tau_{\text{delay}}$ and rate-of-change constraints, a Model Predictive Controller (MPC) executes this calculation over a receding 15-minute horizon, damping residual out-of-bed variance by an additional $60\% - 80\%$.

---

## 6. Worked example

### 6.1 Industrial Problem Setting: The Andina-Pacific Copper Complex

The Andina-Pacific mining complex produces copper concentrate for export to East Asian smelters. The operation extracts ore from three distinct open-pit sectors and maintains one legacy oxidized surface stockpile:

| Source $i$ | Sector Description | Available Wet Mass $W_i$ (WMT) | Moisture $m_i$ ($\%$) | Copper Grade $g_{\text{Cu}}$ ($\%$) | Arsenic Grade $g_{\text{As}}$ ($\text{ppm}$) | $\text{SiO}_2$ Grade ($\%$) | $\text{Al}_2\text{O}_3$ Grade ($\%$) | Mining & Haul Cost $c_i$ ($\text{USD/DMT}$) | Loose Bulk Density $\rho_i$ ($\text{t/m}^3$) |
|---|---|---|---|---|---|---|---|---|---|
| **S1** | Deep Primary Skarn (Enargite rich) | $120,000$ | $7.0\%$ | $2.80\%$ | $2,400$ | $52.0\%$ | $12.0\%$ | $\$18.50$ | $1.75$ |
| **S2** | East Pit Porphyry (Clean chalcopyrite) | $200,000$ | $9.5\%$ | $1.65\%$ | $450$ | $64.0\%$ | $14.5\%$ | $\$14.20$ | $1.60$ |
| **S3** | North Footwall (High-silica low-grade) | $150,000$ | $5.0\%$ | $0.85\%$ | $200$ | $75.0\%$ | $8.0\%$ | $\$9.80$ | $1.65$ |
| **S4** | Reclaim Finger 4 (High-arsenic transitional) | $60,000$ | $12.0\%$ | $2.10\%$ | $3,200$ | $48.0\%$ | $16.0\%$ | $\$6.50$ | $1.45$ |

#### Campaign Requirements:
1. **Target Blend Tonnage:** Exactly $150,000\text{ DMT}$ of blended feed for the mill campaign.
2. **Minimum Head Grade:** $G_{\text{Cu}} \ge 1.80\%$.
3. **Maximum Arsenic Concentration:** $G_{\text{As}} \le 1,200\text{ ppm}$ (Contract hard rejection limit at smelter is $2,000\text{ ppm}$; internal operating specification is $1,200\text{ ppm}$).
4. **Flotation Flux Constraint:** Silica-to-Alumina ratio must satisfy:
   $$3.8 \le \frac{\% \text{SiO}_2}{\% \text{Al}_2\text{O}_3} \le 5.2$$
5. **Stockyard Volumetric Limit:** The bed blending yard can hold at most $105,000\text{ m}^3$ of loose uncompacted ore.

#### Commercial Smelter Terms (Normalized per DMT of Blended Feed):
- **Payable Metal Value:** Net revenue factor of $\$48.00$ per percentage point of Cu per DMT.
- **Arsenic Penalty Structure:**
  - $0 \le G_{\text{As}} \le 800\text{ ppm}$: Penalty free ($\$0.00/\text{DMT}$).
  - $800 < G_{\text{As}} \le 1,200\text{ ppm}$: Penalty of $\$2.50/\text{DMT}$ per $100\text{ ppm}$ above $800\text{ ppm}$.
  - $1,200 < G_{\text{As}} \le 1,800\text{ ppm}$: Tier 2 penalty of $\$6.00/\text{DMT}$ per $100\text{ ppm}$ above $1,200\text{ ppm}$.
  - $G_{\text{As}} > 1,800\text{ ppm}$: Rejection risk.

---

### 6.2 Step-by-Step Mathematical Solution

#### Step 1: Convert Available Inventories to Dry Metric Tonnes (DMT)
$$D_i = W_i \cdot (1 - m_i)$$
- Source 1: $D_1 = 120,000 \cdot (1 - 0.070) = 111,600\text{ DMT}$
- Source 2: $D_2 = 200,000 \cdot (1 - 0.095) = 181,000\text{ DMT}$
- Source 3: $D_3 = 150,000 \cdot (1 - 0.050) = 142,500\text{ DMT}$
- Source 4: $D_4 = 60,000 \cdot (1 - 0.120) = 52,800\text{ DMT}$

Total available dry ore = $487,900\text{ DMT}$. We require an optimal selection of $150,000\text{ DMT}$.

#### Step 2: Establish the Linear Optimization Constraints
Let $x_1, x_2, x_3, x_4$ denote the dry metric tonnes drawn from each source ($x_i \ge 0$).

1. **Total Mass Balance:**
   $$x_1 + x_2 + x_3 + x_4 = 150,000\text{ DMT}$$

2. **Copper Grade Requirement ($G_{\text{Cu}} \ge 1.80\%$):**
   $$\sum_{i=1}^4 (g_{i, \text{Cu}} - 1.80) x_i \ge 0$$
   $$(2.80 - 1.80) x_1 + (1.65 - 1.80) x_2 + (0.85 - 1.80) x_3 + (2.10 - 1.80) x_4 \ge 0$$
   $$1.00 x_1 - 0.15 x_2 - 0.95 x_3 + 0.30 x_4 \ge 0$$

3. **Arsenic Impurity Limit ($G_{\text{As}} \le 1,200\text{ ppm}$):**
   $$\sum_{i=1}^4 (g_{i, \text{As}} - 1,200) x_i \le 0$$
   $$(2,400 - 1,200) x_1 + (450 - 1,200) x_2 + (200 - 1,200) x_3 + (3,200 - 1,200) x_4 \le 0$$
   $$1,200 x_1 - 750 x_2 - 1,000 x_3 + 2,000 x_4 \le 0$$

4. **Silica-to-Alumina Ratio Bounds:**
   - Lower bound ($R \ge 3.8$):
     $$\sum_{i=1}^4 (g_{i, \text{SiO}_2} - 3.8 \cdot g_{i, \text{Al}_2\text{O}_3}) x_i \ge 0$$
     - S1: $52.0 - 3.8 \cdot 12.0 = 52.0 - 45.6 = +6.4$
     - S2: $64.0 - 3.8 \cdot 14.5 = 64.0 - 55.1 = +8.9$
     - S3: $75.0 - 3.8 \cdot 8.0 = 75.0 - 30.4 = +44.6$
     - S4: $48.0 - 3.8 \cdot 16.0 = 48.0 - 60.8 = -12.8$
     $$6.4 x_1 + 8.9 x_2 + 44.6 x_3 - 12.8 x_4 \ge 0$$
   - Upper bound ($R \le 5.2$):
     $$\sum_{i=1}^4 (g_{i, \text{SiO}_2} - 5.2 \cdot g_{i, \text{Al}_2\text{O}_3}) x_i \le 0$$
     - S1: $52.0 - 5.2 \cdot 12.0 = 52.0 - 62.4 = -10.4$
     - S2: $64.0 - 5.2 \cdot 14.5 = 64.0 - 75.4 = -11.4$
     - S3: $75.0 - 5.2 \cdot 8.0 = 75.0 - 41.6 = +33.4$
     - S4: $48.0 - 5.2 \cdot 16.0 = 48.0 - 83.2 = -35.2$
     $$-10.4 x_1 - 11.4 x_2 + 33.4 x_3 - 35.2 x_4 \le 0$$

5. **Volumetric Stockyard Constraint ($V \le 105,000\text{ m}^3$):**
   The volume occupied per dry tonne of source $i$ is $v_i = \frac{1}{(1 - m_i) \rho_i}$:
   - $v_1 = \frac{1}{(1 - 0.070) \cdot 1.75} = \frac{1}{0.93 \cdot 1.75} = \frac{1}{1.6275} \approx 0.6144\text{ m}^3/\text{DMT}$
   - $v_2 = \frac{1}{(1 - 0.095) \cdot 1.60} = \frac{1}{0.905 \cdot 1.60} = \frac{1}{1.4480} \approx 0.6906\text{ m}^3/\text{DMT}$
   - $v_3 = \frac{1}{(1 - 0.050) \cdot 1.65} = \frac{1}{0.95 \cdot 1.65} = \frac{1}{1.5675} \approx 0.6380\text{ m}^3/\text{DMT}$
   - $v_4 = \frac{1}{(1 - 0.120) \cdot 1.45} = \frac{1}{0.88 \cdot 1.45} = \frac{1}{1.2760} \approx 0.7837\text{ m}^3/\text{DMT}$

   $$0.6144 x_1 + 0.6906 x_2 + 0.6380 x_3 + 0.7837 x_4 \le 105,000\text{ m}^3$$

6. **Inventory Availability Bounds:**
   $$x_1 \le 111,600, \quad x_2 \le 181,000, \quad x_3 \le 142,500, \quad x_4 \le 52,800$$

---

#### Step 3: Solve the Linear Program
We evaluate the net profit contribution per dry tonne for each source before penalties:
- Revenue coefficient: $r_i = 48.00 \cdot g_{i, \text{Cu}}$
- Margin coefficient: $c_i^{\text{net}} = r_i - c_i$
  - S1: $48.00 \cdot 2.80 - 18.50 = 134.40 - 18.50 = +\$115.90/\text{DMT}$
  - S2: $48.00 \cdot 1.65 - 14.20 = 79.20 - 14.20 = +\$65.00/\text{DMT}$
  - S3: $48.00 \cdot 0.85 - 9.80 = 40.80 - 9.80 = +\$31.00/\text{DMT}$
  - S4: $48.00 \cdot 2.10 - 6.50 = 100.80 - 6.50 = +\$94.30/\text{DMT}$

S1 is the highest-value ore ($\$115.90$), followed by S4 ($\$94.30$), S2 ($\$65.00$), and S3 ($\$31.00$). However, S1 and S4 carry toxic levels of arsenic ($2,400\text{ ppm}$ and $3,200\text{ ppm}$), while S2 and S3 provide clean dilution capacity ($450\text{ ppm}$ and $200\text{ ppm}$).

Solving this LP yields the optimal blend allocation:
- **$x_1^* = 53,250\text{ DMT}$** (Source 1: Primary Skarn)
- **$x_2^* = 82,750\text{ DMT}$** (Source 2: Clean Porphyry)
- **$x_3^* = 0\text{ DMT}$** (Source 3: Low-grade not needed)
- **$x_4^* = 14,000\text{ DMT}$** (Source 4: High-As Reclaim)
- **Total Blended DMT:** $53,250 + 82,750 + 0 + 14,000 = 150,000\text{ DMT}$.

---

#### Step 4: Verify Blended Quality Attributes and Constraints

1. **Blended Copper Grade:**
   $$G_{\text{Cu}} = \frac{53,250 \cdot 2.80 + 82,750 \cdot 1.65 + 0 \cdot 0.85 + 14,000 \cdot 2.10}{150,000}$$
   $$G_{\text{Cu}} = \frac{149,100 + 136,537.5 + 0 + 29,400}{150,000} = \frac{315,037.5}{150,000} = 2.10025\% \approx 2.100\%$$
   Constraint check: $2.100\% \ge 1.80\%$ (**Feasible, +0.30% margin**).

2. **Blended Arsenic Concentration:**
   $$G_{\text{As}} = \frac{53,250 \cdot 2,400 + 82,750 \cdot 450 + 0 \cdot 200 + 14,000 \cdot 3,200}{150,000}$$
   $$G_{\text{As}} = \frac{127,800,000 + 37,237,500 + 0 + 44,800,000}{150,000} = \frac{209,837,500}{150,000} = 1,398.92\text{ ppm}$$
   Wait! The basic LP upper bound was set to $1,200\text{ ppm}$ in requirement 3! If the LP has an upper bound of $1,200\text{ ppm}$, let us re-solve to strictly honor the $1,200\text{ ppm}$ upper bound.

Let us re-calculate the exact binding frontier at $G_{\text{As}} = 1,200\text{ ppm}$:
To suppress As from $1,398.92\text{ ppm}$ down to $1,200\text{ ppm}$, we must reduce the high-As ores (S1 and S4) and increase the clean diluent S2.
Let us solve the system of equations with $x_3 = 0, x_1 + x_2 + x_4 = 150,000$:
- Grade target: $G_{\text{As}} = 1,200\text{ ppm} \implies 2,400 x_1 + 450 x_2 + 3,200 x_4 = 150,000 \cdot 1,200 = 180,000,000$
- Substitute $x_2 = 150,000 - x_1 - x_4$:
  $$2,400 x_1 + 450(150,000 - x_1 - x_4) + 3,200 x_4 = 180,000,000$$
  $$1,950 x_1 + 2,750 x_4 + 67,500,000 = 180,000,000$$
  $$1,950 x_1 + 2,750 x_4 = 112,500,000$$
  Divide by 50: $39 x_1 + 55 x_4 = 2,250,000$.

To maximize profit, compare the net profit per unit of arsenic penalty consumption:
- S1 brings $\$115.90 - \$65.00 = \$50.90/\text{DMT}$ advantage over S2, generating $2,400 - 450 = 1,950\text{ ppm}$ of As delta. Ratio = $50.90 / 1,950 = \$0.0261/\text{ppm}$.
- S4 brings $\$94.30 - \$65.00 = \$29.30/\text{DMT}$ advantage over S2, generating $3,200 - 450 = 2,750\text{ ppm}$ of As delta. Ratio = $29.30 / 2,750 = \$0.0106/\text{ppm}$.

S1 is vastly more profitable per unit of arsenic budget consumed than S4! Therefore, the LP sets $x_4^* = 0$, maximizing $x_1$:
$$39 x_1 + 0 = 2,250,000 \implies x_1^* = \frac{2,250,000}{39} = 57,692.3\text{ DMT}$$
$$x_2^* = 150,000 - 57,692.3 = 92,307.7\text{ DMT}$$
$$x_3^* = 0\text{ DMT}, \qquad x_4^* = 0\text{ DMT}$$

Let us verify all constraints under this exact solution:
1. **Total Tonnage:** $57,692.3 + 92,307.7 = 150,000.0\text{ DMT}$.
2. **Blended Copper Grade:**
   $$G_{\text{Cu}} = \frac{57,692.3 \cdot 2.80 + 92,307.7 \cdot 1.65}{150,000} = \frac{161,538.44 + 152,307.71}{150,000} = \frac{313,846.15}{150,000} = 2.0923\%$$
   $2.092\% \ge 1.80\%$ (**Satisfied**).
3. **Blended Arsenic Grade:**
   $$G_{\text{As}} = \frac{57,692.3 \cdot 2,400 + 92,307.7 \cdot 450}{150,000} = \frac{138,461,520 + 41,538,465}{150,000} = \frac{179,999,985}{150,000} = 1,200.0\text{ ppm}$$
   $1,200.0\text{ ppm} \le 1,200.0\text{ ppm}$ (**Exact Binding**).
4. **Silica and Alumina Grades & Ratio:**
   - $\text{SiO}_2$: $\frac{57,692.3 \cdot 52.0 + 92,307.7 \cdot 64.0}{150,000} = \frac{3,000,000 + 5,907,693}{150,000} = 59.385\%$
   - $\text{Al}_2\text{O}_3$: $\frac{57,692.3 \cdot 12.0 + 92,307.7 \cdot 14.5}{150,000} = \frac{692,308 + 1,338,462}{150,000} = 13.538\%$
   - Ratio: $\frac{\% \text{SiO}_2}{\% \text{Al}_2\text{O}_3} = \frac{59.385}{13.538} = 4.386$
   $3.8 \le 4.386 \le 5.2$ (**Satisfied, perfectly centered in target window**).
5. **Physical Volume Occupied in Yard:**
   $$V = 57,692.3 \cdot 0.6144 + 92,307.7 \cdot 0.6906 = 35,446.1 + 63,747.7 = 99,193.8\text{ m}^3$$
   $99,193.8\text{ m}^3 \le 105,000.0\text{ m}^3$ (**Feasible, consumes 94.5% of stockyard volume**).
6. **Wet Metric Tonnes (WMT) Hauled to Crusher:**
   - Source 1: $W_1 = \frac{57,692.3}{1 - 0.070} = 62,034.7\text{ WMT}$
   - Source 2: $W_2 = \frac{92,307.7}{1 - 0.095} = 101,997.5\text{ WMT}$
   - Total wet ore to haul = $164,032.2\text{ WMT}$.
   - Blended moisture = $\frac{164,032.2 - 150,000.0}{164,032.2} = \frac{14,032.2}{164,032.2} = 8.555\%$.

---

#### Step 5: Commercial Penalty Accounting
Under the smelter contract:
- The arsenic concentration is $1,200.0\text{ ppm}$.
- The penalty applies to arsenic in excess of $800\text{ ppm}$:
  $$\Delta_{\text{As}} = 1,200 - 800 = 400\text{ ppm}$$
- Penalty charge rate is $\$2.50$ per $100\text{ ppm}$ above $800\text{ ppm}$:
  $$\text{Unit Penalty} = \frac{400}{100} \cdot \$2.50 = 4 \cdot \$2.50 = \$10.00/\text{DMT}$$
- Total campaign penalty cost:
  $$\text{Total Penalty} = 150,000\text{ DMT} \cdot \$10.00/\text{DMT} = \$1,500,000$$

#### Net Commercial Profitability Calculation:
- Gross Metal Revenue: $150,000\text{ DMT} \cdot (2.092308 \cdot \$48.00) = 150,000 \cdot \$100.4308 = \$15,064,615$
- Mining & Haulage Costs:
  - S1: $57,692.31 \cdot \$18.50 = \$1,067,308$
  - S2: $92,307.69 \cdot \$14.20 = \$1,310,769$
  - Total extraction cost = $\$2,378,077$
- Smelter Penalty Deductions: $\$1,500,000$
- **Net Realized Margin:** $\$15,064,615 - \$2,378,077 - \$1,500,000 = \$11,186,538$ ($\$74.58/\text{DMT}$).

---

#### Step 6: Stockpile Bed Sizing via Pierre Gy Variance Theory
The blended ore is fed to a Chevron bed blending yard.
- Incoming copper grade variance from pit dispatch: $\sigma_{\text{in}}^2 = 0.320 \, (\% \text{Cu})^2$ ($\sigma_{\text{in}} = 0.566\%$).
- Maximum allowable feed variance entering the flotation plant: $\sigma_{\text{spec}}^2 = 0.004 \, (\% \text{Cu})^2$ ($\sigma_{\text{spec}} = 0.063\%$).
- Autocorrelation parameters along the $250\text{-meter}$ pile length ($L = 250\text{ m}$):
  - Spatial variogram range: $a = 40\text{ m}$.
  - Variogram correlation factor:
    $$\frac{3 a}{8 L} = \frac{3 \cdot 40}{8 \cdot 250} = \frac{120}{2,000} = 0.060$$
- Target variance ratio:
  $$r = \frac{\sigma_{\text{spec}}^2}{\sigma_{\text{in}}^2} = \frac{0.004}{0.320} = 0.0125$$

Calculate the required number of Chevron layers $N_L$:
$$r \approx \frac{1}{N_L} \left(1 - \frac{3 a}{8 L}\right) + \frac{3 a}{8 L}$$
Wait! Notice that the baseline correlation factor $\frac{3 a}{8 L} = 0.060 > r = 0.0125$.
This illustrates an essential operational insight: **Physical layering alone cannot reduce variance below $0.060$ because long-range spatial autocorrelation penetrates the bed!**
Even with $N_L \to \infty$, the asymptotic minimum variance reduction achievable by the bed is:
$$r_{\min} = \frac{3 a}{8 L} = 0.060 \implies \sigma_{\text{bed\_min}}^2 = 0.060 \cdot 0.320 = 0.0192 \, (\% \text{Cu})^2$$
The bed achieves $\sigma_{\text{bed}} = \sqrt{0.0192} = 0.1386\% \text{ Cu}$.

To hit the strict plant tolerance of $\sigma_{\text{spec}} = 0.063\%$, the stockyard must operate in two stages:
1. Build a **$N_L = 250$ layer** Chevron bed to achieve $r_{\text{bed}} \approx \frac{1}{250}(1 - 0.060) + 0.060 = 0.00376 + 0.060 = 0.06376 \implies \sigma_{\text{bed}} = 0.143\%$.
2. Deploy the **Online PGNAA closed-loop trimming system** (Step 7) on the reclaimed stream to trim residual spikes, reducing variance from $0.0192$ down to $0.0038$ ($\sigma_{\text{spec}} = 0.0617\%$), meeting the plant specification.

---

## 7. Code, query, or calculation

The following self-contained Python script implements the multi-element blending Linear Program formulated in Section 6 using `scipy.optimize.linprog`. It handles dry-to-wet conversions, mass balance, grade bounds, silica-to-alumina flux constraints, and yard volume limits.

```python
import numpy as np
from scipy.optimize import linprog

# 1. Source definitions: [Cu %, As ppm, SiO2 %, Al2O3 %, Moisture %, Cost $/DMT, Density t/m3, Avail DMT]
sources = {
    'S1': [2.80, 2400.0, 52.0, 12.0, 0.070, 18.50, 1.75, 111600.0],
    'S2': [1.65,  450.0, 64.0, 14.5, 0.095, 14.20, 1.60, 181000.0],
    'S3': [0.85,  200.0, 75.0,  8.0, 0.050,  9.80, 1.65, 142500.0],
    'S4': [2.10, 3200.0, 48.0, 16.0, 0.120,  6.50, 1.45,  52800.0]
}
names = list(sources.keys())
n = len(names)
target_dmt = 150000.0
vol_limit = 105000.0

# Objective: Minimize cost - revenue (Net Profit Maximization: Rev = $48.00/t per %Cu)
c = np.array([sources[k][5] - 48.0 * sources[k][0] for k in names])

# Inequality Constraints: A_ub * x <= b_ub
A_ub, b_ub = [], []
# Cu >= 1.80% -> -sum((Cu_i - 1.80)*x_i) <= 0
A_ub.append([-(sources[k][0] - 1.80) for k in names])
b_ub.append(0.0)
# As <= 1200 ppm -> sum((As_i - 1200)*x_i) <= 0
A_ub.append([(sources[k][1] - 1200.0) for k in names])
b_ub.append(0.0)
# SiO2 / Al2O3 >= 3.8 -> -sum((SiO2 - 3.8*Al2O3)*x_i) <= 0
A_ub.append([-(sources[k][2] - 3.8 * sources[k][3]) for k in names])
b_ub.append(0.0)
# SiO2 / Al2O3 <= 5.2 -> sum((SiO2 - 5.2*Al2O3)*x_i) <= 0
A_ub.append([(sources[k][2] - 5.2 * sources[k][3]) for k in names])
b_ub.append(0.0)
# Stockyard Volume <= 105,000 m3: v_i = 1 / ((1 - m_i) * rho_bulk)
v_i = [1.0 / ((1.0 - sources[k][4]) * sources[k][6]) for k in names]
A_ub.append(v_i)
b_ub.append(vol_limit)

# Equality Constraint: Total Dry Tonnage == 150,000
A_eq = [[1.0] * n]
b_eq = [target_dmt]

# Bounds: 0 <= x_i <= Available DMT
bounds = [(0.0, sources[k][7]) for k in names]

# Solve LP
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')

# Display Results
print(f"Status: {res.message}")
for k, x in zip(names, res.x):
    wmt = x / (1.0 - sources[k][4])
    print(f"  {k}: {x:10.1f} DMT ({wmt:10.1f} WMT)")
blend_cu = sum(res.x[i] * sources[names[i]][0] for i in range(n)) / target_dmt
blend_as = sum(res.x[i] * sources[names[i]][1] for i in range(n)) / target_dmt
blend_sio2 = sum(res.x[i] * sources[names[i]][2] for i in range(n)) / target_dmt
blend_al2o3 = sum(res.x[i] * sources[names[i]][3] for i in range(n)) / target_dmt
pen_as = max(0.0, (blend_as - 800.0) / 100.0 * 2.50) if blend_as <= 1200.0 else 10.0
net_margin = (-res.fun / target_dmt) - pen_as
print(f"Blended Specs: Cu={blend_cu:.3f}%, As={blend_as:.1f} ppm, SiO2/Al2O3={blend_sio2/blend_al2o3:.2f}")
print(f"Net Margin: ${net_margin:.2f} / DMT (Total Margin: ${net_margin * target_dmt:,.2f})")
```

### Expected Output
```text
Status: Optimization terminated successfully. (HiGHS Status 7: Optimal)
  S1:    57692.3 DMT (   62034.7 WMT)
  S2:    92307.7 DMT (  101997.5 WMT)
  S3:        0.0 DMT (       0.0 WMT)
  S4:        0.0 DMT (       0.0 WMT)
Blended Specs: Cu=2.092%, As=1200.0 ppm, SiO2/Al2O3=4.39
Net Margin: $74.58 / DMT (Total Margin: $11,186,538.46)
```

---

## 8. Data, standards, and regulatory context

Grade blending decisions intersect international commercial law, statutory environmental mandates, and maritime safety regulations. Operating outside these standards exposes the mining enterprise to vessel seizures, contract defaults, and severe financial penalties.

### 8.1 International Maritime Organization (IMO) IMSBC Code
Bulk mineral concentrates shipped by sea are governed by the **International Maritime Solid Bulk Cargoes (IMSBC) Code** (Resolution MSC.268(85), updated biennially).
1. **Group A Cargoes (Liquefaction Risk):** Fine mineral concentrates (copper, nickel, lead, zinc, and iron ore fines) can liquefy if shipped with moisture exceeding the **Transportable Moisture Limit (TML)**. Liquefaction shifts the cargo's center of gravity during vessel rolling, inducing sudden catastrophic capsizing.
2. **Mandatory Flow Table / Proctor-Fagerberg Testing:** Under ISO 12742, shippers must certify the Flow Moisture Point (FMP) and TML within 7 days prior to vessel loading.
3. **Binding Rule:** The maximum permissible moisture for loading is strictly $90\%$ of the FMP:
   $$\text{TML} = 0.90 \times \text{FMP}$$
   Blending wet ROM stockpiles into export concentrate beds must be constrained to prevent composite moisture from crossing the statutory TML.

### 8.2 ISO Sampling and Quality Standards
Commercial settlements and grade reconciliation require compliance with international sampling standards:
- **ISO 12743:** *Copper, lead, zinc and nickel concentrates — Sampling procedures for determination of metal and moisture content.* Mandates cross-stream mechanical cutter speeds ($< 0.6\text{ m/s}$), sample increment mass formulas based on Pierre Gy's Fundamental Sampling Error (FSE), and preparation protocols to eliminate bias.
- **ISO 3082:** *Iron ores — Sampling and sample preparation procedures.* Establishes mass-basis and time-basis mechanical increment extraction rules from moving conveyor belts.
- **ISO 10251:** *Copper, lead, zinc and nickel concentrates — Determination of mass loss on drying.* Specifies oven drying at $105^\circ\text{C} \pm 5^\circ\text{C}$ to constant weight for DMT settlement.

### 8.3 Chinese Customs Decree 112 (Import Inspection Quality Thresholds)
China operates the world's largest base metal smelting network and enforces statutory import thresholds for copper and lead concentrates through the **General Administration of Customs of the People's Republic of China (GACC)**.
- Under National Standard **GB 20424-2006** (*Limits of deleterious elements in copper concentrates*), any import cargo exceeding:
  - $\text{Arsenic (As)} > 0.50\%$ ($5,000\text{ ppm}$)
  - $\text{Lead (Pb)} > 6.0\%$
  - $\text{Cadmium (Cd)} > 0.05\%$
  - $\text{Mercury (Hg)} > 0.01\%$ ($100\text{ ppm}$)
  - $\text{Bismuth (Bi)} > 0.20\%$
  is classified as solid hazardous industrial waste and denied customs entry. Vessels carrying non-compliant blends are quarantined, turned back at the discharge berth, and subject to mandatory repatriation at the shipper's expense.

---

## 9. Industry practice and case evidence

### 9.1 Case Evidence 1: Arsenic Management at Codelco (Chuquicamata & Radomiro Tomic)
- **Context:** The Chuquicamata underground and open-pit deposits in northern Chile contain massive zones of enargite ($\text{Cu}_3\text{AsS}_4$) and tennantite ($\text{Cu}_{12}\text{As}_4\text{S}_{13}$), driving mine feed arsenic levels up to $4,000 - 8,000\text{ ppm}$.
- **Mechanism:** Codelco implemented an integrated multi-tier blending supply chain. Underground high-arsenic copper ore is routed through segregated underground surge passes and blended with ultra-clean chalcopyrite ore from Radomiro Tomic. At the primary concentrator, real-time PGNAA analyzers monitor conveyor streams feeding the semi-autogenous grinding (SAG) mills.
- **Outcome:** Through linear programming blending across three mine sectors, Codelco maintains final concentrate arsenic at precisely $0.28\% - 0.35\%$ ($2,800 - 3,500\text{ ppm}$), avoiding the punitive $\$30.00/\text{DMT}$ penalty tier while remaining safely below the Chinese import rejection threshold of $0.50\%$.

### 9.2 Case Evidence 2: Pilbara Iron Ore Bed Blending Yards (BHP & Rio Tinto)
- **Context:** Western Australia's Pilbara iron ore operations mine distinct channel iron deposits (CID, such as Yandi), high-phosphorus Brockman ores, and micro-platy Marra Mamba ores. Asian blast furnaces demand consistent chemical specs: $\text{Fe} \ge 62.0\%$, $\text{SiO}_2 \le 4.5\%$, $\text{Al}_2\text{O}_3 \le 1.8\%$, and $\text{Phosphorus (P)} \le 0.075\%$.
- **Mechanism:** At Newman, Area C, and Tom Price, automated stockyards utilize twin automated traveling stackers operating in **Windrow mode** to construct $300,000\text{-tonne}$ blending beds consisting of $350 - 500$ discrete longitudinal layers. Bridge-mounted barrel reclaimers equipped with full-face reciprocating harrows slice the bed perpendicularly.
- **Outcome:** incoming trainload grade variances of $\sigma_{\text{in}} = 1.25\% \text{ Fe}$ and $\sigma_{\text{in}} = 0.35\% \text{ Al}_2\text{O}_3$ are suppressed to out-of-bed delivery variances of $\sigma_{\text{out}} < 0.12\% \text{ Fe}$ and $\sigma_{\text{out}} < 0.04\% \text{ Al}_2\text{O}_3$. This ultra-consistent blend commands premium pricing on the Platts Iron Ore Index (IODEX), generating tens of millions of dollars in net annual margin premiums over spot lump products.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Physical / Statistical Mechanism | Supply Chain Consequence | Mitigation / Control |
|---|---|---|---|---|
| **Stockyard Toe Slumping & Segregation** | Cyclic sinusoidal grade oscillations recorded during reclaiming. | Coarse boulders roll down Chevron slopes to the pile toes; fines stay concentrated at apex. | Downstream flotation circuits experience cyclical grade and grindability shocks every 45 minutes. | Transition to Windrow stacking; install reciprocating reclaimer face rakes to enforce uniform avalanching. |
| **Moisture Dilution Bias** | Head grade reconciliation consistently reports $3\% - 8\%$ lower metal than the LP predicted. | Blending calculations formulated on WMT instead of DMT; high-moisture parcels under-deliver dry mass. | Flotation plant feed under-grades; quarterly metal production targets missed; contractual shipment delays. | Enforce strict dry mass transformation ($x_i^{\text{dry}} = x_i^{\text{wet}}(1 - m_i)$) in all LP optimization constraints. |
| **Smelter Penalty Cliff Overshoot** | Commercial settlements arrive with surprise multi-million dollar penalty deductions. | Average composite grade was within limits, but high-frequency unblended parcels triggered tier escalation. | Profit margins wiped out; cash flow deficits; credit facility covenant breaches. | Formulate LP with buffer safety margins ($3\sigma$ below penalty tiers); install online PGNAA trimming. |
| **Slag Freezing & Taphole Failure** | Smelter furnace differential pressure rises; tapping temperature escalates. | Alumina-to-silica ratio exceeds furnace design limit ($> 2.5$), dramatically increasing slag viscosity. | Furnace shutdown; skulling of slag bath; millions of dollars in refractory relining and lost production. | Incorporate strict slag basicity and gangue ratio constraints directly into the upstream blending LP. |
| **Vessel Liquefaction at Sea** | Cargo moisture certificates issued near the statutory TML; port stockpile rainfall. | High-moisture fine ore loaded into bulk carriers undergoes cyclic shear compaction, developing pore water pressure. | Catastrophic cargo shift, vessel instability, hull cracking, and potential loss of ship and crew. | Enforce IMO IMSBC compliance; install covered stockpile sheds; maintain automated moisture analyzers at ship loaders. |
| **End-Cone Dilution Shock** | Flotation recovery drops sharply during the final 8 hours of reclaiming each stockpile bed. | The triangular sloped ends of a longitudinal bed cannot achieve layered superposition; layer count $N_L \to 1$. | Plant feed grade collapses or spikes; unblended material enters circuit without homogenization. | Re-route end-cone material to secondary surge bins; recirculate end-cone ore into next bedding campaign. |
| **PGNAA Calibration Drift** | Systematic bias between real-time analyzer readings and subsequent laboratory XRF assays. | Detector temperature drift, scintillator aging, or variations in belt load profile and bulk density. | Dynamic trimming feeders dose wrong ratios, corrupting feed specification rather than correcting it. | Daily automated background checks; monthly dynamic calibration loops against certified physical belt cuts (ISO 12743). |
| **Reclaimer Rake Mechanical Snag** | Oscillating harrow motor trips on high torque; uneven furrow marks across pile face. | Compaction of sticky, high-clay ore at the pile base causing excessive drag on reclaimer teeth. | Stockyard reclaim halts; mill feed starved; emergency bulldozers re-introduce extreme grade variance. | Monitor rake drive hydraulic pressure; blend sticky clay ore with abrasive quartz skarn; maintain tooth clearances. |
| **Volumetric Stockpile Overfilling** | Material overflows retaining berms; reclaimer tracks buried in spilled ore. | Failure to account for rock swell factors ($S \ge 1.4$) and loose bulk density when scheduling deliveries. | Equipment derailment; emergency cleanup costs; mechanical structural damage to stacker rails. | Enforce volumetric yard capacity constraints ($\sum x_i / \rho_{\text{bulk}} \le V_{\max}$) in the daily LP model. |
| **Rejection Threshold Breach at Discharge** | Customer issues legal Notice of Rejection; cargo arrested at import port. | A single toxic parcel (e.g., $5,500\text{ ppm As}$) bypassed blending through bypass chutes during equipment maintenance. | Shipper liable for vessel demurrage, dead freight, return ocean voyage, and hazardous material storage fines. | Implement automated physical interlocks preventing conveyor bypass when deleterious assays exceed thresholds. |

---

## 11. Assumptions and limits

| Assumption / Simplification | Why It Is Needed | Boundary Condition / Failure State | How to Audit / Test in Practice |
|---|---|---|---|
| **Linear Mass Blending of Grades** | Permits formulation of multi-element blending as a solvable Linear Program (LP). | Fails for non-additive metallurgical properties (e.g., pH, flotation kinetics, froth viscosity, work index). | Conduct bench-scale flotation tests on composite physical blends to verify kinetic recovery additivity. |
| **Constant Bulk Density Across Sources** | Simplifies volumetric inventory tracking and capacity constraints. | Fails when blending high-density massive sulfides ($\rho = 3.2$) with porous oxidized saprolite ($\rho = 1.4$). | Measure loose bulk density of individual source lithologies across moisture ranges using certified pycnometers. |
| **Uniform Layer Thickness in Stockpiles** | Essential for Pierre Gy's ideal $1/N_L$ variance reduction derivation. | Fails if stacker travel speed varies, belt feed surges, or stacker reverses mid-travel. | Audit stacker PLC logs for travel speed consistency and belt mass-scale integration linearity. |
| **Zero Chemical Leaching on Stockpile Pads** | Assumes elemental chemistry remains static while ore sits on blending beds. | Fails in high-rainfall zones where sulfide ores oxidize, generating acid mine drainage and leaching metals. | Install lysimeters and water collection sumps beneath stockpile pads; assay runoff water for dissolved copper. |
| **Perfect Transversal Face Reclaiming** | Assumes the reclaimer cuts across all $N_L$ layers simultaneously. | Fails if front-end loaders or excavators dig selectively into the pile toes or face. | Enforce bridge-type barrel or bucket-wheel reclaimers with full-face rakes; ban selective wheel loader reclaim. |
| **Instantaneous Assay Availability** | Assumes blast-hole and stockpile assays are known before planning blend cuts. | Fails when laboratory turnaround lags mining dispatch by $> 24$ hours. | Deploy fast online bench analyzers, MWD (Measurement While Drilling) sensors, or portable XRF instruments. |
| **Uniform Particle Size Distribution (PSD)** | Assumes particle sizing does not affect chemical assay distribution. | Fails in mineral systems where metals preferentially concentrate in fine fractions (e.g., copper in fines). | Screen size-fraction assays ($-75\mu\text{m}, +150\mu\text{m}, +2\text{mm}$) to detect preferential deportment. |
| **Unconstrained Reclaim Rate** | Assumes dynamic trim surge feeders can ramp instantaneously to match setpoints. | Fails when feeder hoppers bridge, hang up on sticky ore, or hit maximum motor RPM limits. | Equip trim hoppers with air cannons and vibratory bin dischargers; monitor feeder motor load percentages. |

---

## 12. Exercises

### Exercise 1: Multi-Element LP Formulation (Iron Ore Sinter Feed)
- **Context:** An iron ore export terminal in South Africa blends three stockpiles to prepare a $200,000\text{-DMT}$ shipment of Premium Sinter Fines for a European steelmaker.
  - Stockpile A: $64.5\% \text{ Fe}$, $2.8\% \text{ SiO}_2$, $1.4\% \text{ Al}_2\text{O}_3$, $0.08\% \text{ P}$, cost $\$42/\text{DMT}$, available $120,000\text{ DMT}$.
  - Stockpile B: $61.0\% \text{ Fe}$, $5.5\% \text{ SiO}_2$, $2.2\% \text{ Al}_2\text{O}_3$, $0.04\% \text{ P}$, cost $\$28/\text{DMT}$, available $150,000\text{ DMT}$.
  - Stockpile C: $58.0\% \text{ Fe}$, $7.2\% \text{ SiO}_2$, $1.1\% \text{ Al}_2\text{O}_3$, $0.03\% \text{ P}$, cost $\$18/\text{DMT}$, available $100,000\text{ DMT}$.
- **Contract Specifications:**
  - $\text{Fe} \ge 62.0\%$
  - $\text{Al}_2\text{O}_3 \le 1.70\%$
  - $\text{P} \le 0.055\%$
  - $\text{Alumina-to-Silica Ratio} \le 0.35$
- **Deliverable:** Formulate the objective function and all linear constraints to minimize total acquisition cost while fulfilling the $200,000\text{-DMT}$ contract.
- **Acceptance Criteria:** State decision variables, algebraic cost function, and all boundary inequalities.

### Exercise 2: Moisture and Wet-to-Dry Reconciliation
- **Context:** A bauxite mining operation transports wet ore from tropical mine pits to a coastal port.
  - Pit Alpha delivers $40,000\text{ WMT}$ at $18.0\%$ moisture, available alumina ($\text{Al}_2\text{O}_3$) grade $48.5\%$ (dry basis).
  - Pit Beta delivers $60,000\text{ WMT}$ at $8.5\%$ moisture, available alumina grade $44.0\%$ (dry basis).
- **Deliverable:** Calculate:
  1. The total dry metric tonnes (DMT) delivered.
  2. The blended moisture percentage of the combined wet shipment.
  3. The blended dry alumina grade ($\% \text{Al}_2\text{O}_3$).
  4. The error in alumina grade if an operator erroneously calculated grade using wet mass weights.
- **Acceptance Criteria:** Show all intermediate mass balance equations; state grade error to 4 decimal places.

### Exercise 3: Smelter Penalty Step Function Linearization
- **Context:** A zinc concentrate contract specifies a base treatment charge of $\$220.00/\text{DMT}$ for concentrate containing $\le 0.10\% \text{ Cadmium (Cd)}$. Penalties escalate as follows:
  - $0.10\% < \text{Cd} \le 0.20\%$: $\$15.00/\text{DMT}$ per $0.01\% \text{ Cd}$ above $0.10\%$.
  - $0.20\% < \text{Cd} \le 0.35\%$: $\$35.00/\text{DMT}$ per $0.01\% \text{ Cd}$ above $0.20\%$.
  - $\text{Cd} > 0.35\%$: Outright rejection.
- **Deliverable:** Formulate the Mixed-Integer Linear Programming (MILP) equations using Special Ordered Sets of Type 2 (SOS2) to represent this penalty structure for a blending optimization model producing $T = 50,000\text{ DMT}$.
- **Acceptance Criteria:** Define grid points $T_k$, cumulative penalty coefficients $C_k$, SOS2 variables $\lambda_k$, and linking constraints.

### Exercise 4: Pierre Gy Stockpile Bed Sizing
- **Context:** A copper-gold concentrator requires feed copper grade standard deviation entering the rod mill to not exceed $\sigma_{\text{spec}} = 0.08\% \text{ Cu}$.
  - Mine blast-hole grade dispatch standard deviation is $\sigma_{\text{in}} = 0.48\% \text{ Cu}$.
  - A proposed longitudinal Chevron bedding stockpile has a length of $L = 300\text{ meters}$.
  - Spatial geostatistical analysis yields a spherical variogram along the stacking direction with range $a = 60\text{ meters}$.
- **Deliverable:**
  1. Compute the theoretical minimum variance reduction factor $r_{\min}$ achievable by an infinite number of layers in this bed.
  2. Determine whether the specification $\sigma_{\text{spec}} = 0.08\%$ is physically achievable in this yard.
  3. If achievable, calculate the minimum number of Chevron layers $N_L$ required. If unachievable, calculate the minimum bed length $L$ required to make it feasible.
- **Acceptance Criteria:** Provide step-by-step arithmetic proofs using Gy's variance integral formula.

### Exercise 5: Chevron vs. Windrow Stacking Selection
- **Context:** A lead-zinc operation processes friable galena-sphalerite ore with a high top size ($P_{95} = 120\text{ mm}$) and high clay content ($14\% -2\mu\text{m}$ fines). Heavy rainfall occurs four months per year. The plant operates a bridge reclaimer.
- **Deliverable:** Write a 200-word engineering critique recommending either Chevron or Windrow stacking.
- **Acceptance Criteria:** Explicitly evaluate segregation mechanics, particle roll-down, moisture absorption, equipment mechanical complexity, and reclaimer rake dynamics.

### Exercise 6: Online PGNAA Dynamic Trim Sizing
- **Context:** A conveyor belt transports reclaimed iron ore at a nominal rate of $4,000\text{ t/h}$. An online PGNAA analyzer detects an unexpected spike in silica ($\text{SiO}_2$), measuring $6.8\% \text{ SiO}_2$ over a 2-minute rolling average. The target plant feed is $4.5\% \text{ SiO}_2$. A dynamic trim hopper contains low-silica concentrate ($1.2\% \text{ SiO}_2$).
- **Deliverable:**
  1. Calculate the required instantaneous trim mass flow rate $u_{\text{trim}}^*$ ($\text{t/h}$) to restore the combined plant feed to exactly $4.5\% \text{ SiO}_2$.
  2. If the trim feeder has a maximum mechanical discharge limit of $800\text{ t/h}$, calculate the resulting blended silica grade entering the plant.
- **Acceptance Criteria:** Show closed-loop mass balance equations and resulting percentages.

---

## 13. Self-check answers

### Self-Check for Exercise 1: Multi-Element LP Formulation
1. **Decision Variables:** Let $x_A, x_B, x_C$ represent the dry metric tonnes drawn from Stockpiles A, B, and C.
2. **Objective Function:**
   $$\min \quad Z = 42 x_A + 28 x_B + 18 x_C$$
3. **Mass Balance Constraint:**
   $$x_A + x_B + x_C = 200,000$$
4. **Fe Grade Constraint ($\text{Fe} \ge 62.0\%$):**
   $$(64.5 - 62.0) x_A + (61.0 - 62.0) x_B + (58.0 - 62.0) x_C \ge 0 \implies 2.5 x_A - 1.0 x_B - 4.0 x_C \ge 0$$
5. **Alumina Constraint ($\text{Al}_2\text{O}_3 \le 1.70\%$):**
   $$(1.4 - 1.7) x_A + (2.2 - 1.7) x_B + (1.1 - 1.7) x_C \le 0 \implies -0.3 x_A + 0.5 x_B - 0.6 x_C \le 0$$
6. **Phosphorus Constraint ($\text{P} \le 0.055\%$):**
   $$(0.08 - 0.055) x_A + (0.04 - 0.055) x_B + (0.03 - 0.055) x_C \le 0 \implies 0.025 x_A - 0.015 x_B - 0.025 x_C \le 0$$
7. **Alumina-to-Silica Ratio Constraint ($\text{Al}_2\text{O}_3 \le 0.35 \cdot \text{SiO}_2$):**
   $$\sum_{i} (g_{i, \text{Al}_2\text{O}_3} - 0.35 \cdot g_{i, \text{SiO}_2}) x_i \le 0$$
   - Stockpile A: $1.4 - 0.35(2.8) = 1.4 - 0.98 = +0.42$
   - Stockpile B: $2.2 - 0.35(5.5) = 2.2 - 1.925 = +0.275$
   - Stockpile C: $1.1 - 0.35(7.2) = 1.1 - 2.52 = -1.42$
   $$0.42 x_A + 0.275 x_B - 1.42 x_C \le 0$$
8. **Inventory Bounds:**
   $$0 \le x_A \le 120,000, \quad 0 \le x_B \le 150,000, \quad 0 \le x_C \le 100,000$$

---

### Self-Check for Exercise 2: Moisture and Wet-to-Dry Reconciliation
1. **Dry Metric Tonnes Delivered:**
   - Pit Alpha: $D_\alpha = 40,000 \times (1 - 0.180) = 40,000 \times 0.820 = 32,800\text{ DMT}$
   - Pit Beta: $D_\beta = 60,000 \times (1 - 0.085) = 60,000 \times 0.915 = 54,900\text{ DMT}$
   - Total DMT = $32,800 + 54,900 = 87,700\text{ DMT}$.
2. **Blended Moisture Percentage:**
   - Total Wet Mass = $40,000 + 60,000 = 100,000\text{ WMT}$.
   - Total Water Mass = $(40,000 \times 0.18) + (60,000 \times 0.085) = 7,200 + 5,100 = 12,300\text{ tonnes water}$.
   - Blended Moisture = $\frac{12,300}{100,000} \times 100\% = 12.30\%$.
3. **True Blended Dry Alumina Grade:**
   $$G_{\text{dry}} = \frac{32,800 \times 48.5\% + 54,900 \times 44.0\%}{87,700} = \frac{1,590,800 + 2,415,600}{87,700} = \frac{4,006,400}{87,700} = 45.6830\%$$
4. **Erroneous Wet-Basis Alumina Calculation:**
   $$G_{\text{wet\_err}} = \frac{40,000 \times 48.5\% + 60,000 \times 44.0\%}{100,000} = \frac{1,940,000 + 2,640,000}{100,000} = 45.8000\%$$
   $$\text{Error} = G_{\text{wet\_err}} - G_{\text{dry}} = 45.8000\% - 45.6830\% = +0.1170\% \text{ (Overestimation)}$$
   Overestimating alumina by $0.117\%$ over $87,700\text{ DMT}$ falsely implies $102.6$ additional tonnes of pure alumina in the refinery digestors.

---

### Self-Check for Exercise 3: Smelter Penalty Step Function Linearization
1. **Grid Points ($T_k$) and Cumulative Penalties ($C_k$):**
   - $k=0$: $T_0 = 0.00\%$, $C_0 = \$0.00/\text{DMT}$
   - $k=1$: $T_1 = 0.10\%$, $C_1 = \$0.00/\text{DMT}$ (Threshold for penalties)
   - $k=2$: $T_2 = 0.20\%$, $C_2 = (0.20 - 0.10) / 0.01 \times \$15.00 = 10 \times \$15.00 = \$150.00/\text{DMT}$
   - $k=3$: $T_3 = 0.35\%$, $C_3 = \$150.00 + (0.35 - 0.20) / 0.01 \times \$35.00 = \$150.00 + 15 \times \$35.00 = \$150.00 + \$525.00 = \$675.00/\text{DMT}$
2. **SOS2 Formulation Equations:**
   - Define weighting variables $\lambda_0, \lambda_1, \lambda_2, \lambda_3 \ge 0$, declared as **SOS2**.
   - Convex Combination Constraint:
     $$\lambda_0 + \lambda_1 + \lambda_2 + \lambda_3 = 1$$
   - Grade Coupling Constraint ($T = 50,000\text{ DMT}$):
     $$\sum_{i \in \mathcal{I}} g_{i, \text{Cd}} x_i = 50,000 \cdot (0.00 \lambda_0 + 0.10 \lambda_1 + 0.20 \lambda_2 + 0.35 \lambda_3)$$
   - Total Campaign Penalty Charge Added to Objective:
     $$\text{Penalty}_{\text{Total}} = 50,000 \cdot (0.00 \lambda_0 + 0.00 \lambda_1 + 150.00 \lambda_2 + 675.00 \lambda_3)$$
   - Rejection Limit Enforcement: The grid terminates strictly at $T_3 = 0.35\%$. Because $\sum \lambda_k = 1$, the model cannot select any blend with cadmium exceeding $0.35\%$, guaranteeing compliance with smelter rejection rules.

---

### Self-Check for Exercise 4: Pierre Gy Stockpile Bed Sizing
1. **Theoretical Minimum Variance Reduction Factor ($r_{\min}$):**
   $$r_{\min} = \frac{3 a}{8 L} = \frac{3 \times 60}{8 \times 300} = \frac{180}{2,400} = 0.0750$$
2. **Feasibility Evaluation:**
   - Minimum achievable variance with infinite layers ($N_L \to \infty$):
     $$\sigma_{\text{bed\_min}}^2 = r_{\min} \times \sigma_{\text{in}}^2 = 0.0750 \times (0.48)^2 = 0.0750 \times 0.2304 = 0.01728 \, (\% \text{Cu})^2$$
     $$\sigma_{\text{bed\_min}} = \sqrt{0.01728} = 0.1315\% \text{ Cu}$$
   - The required specification is $\sigma_{\text{spec}} = 0.08\% \implies \sigma_{\text{spec}}^2 = 0.0064 \, (\% \text{Cu})^2$.
   - Because $\sigma_{\text{spec}}^2 = 0.0064 < \sigma_{\text{bed\_min}}^2 = 0.01728$, **the specification is physically unachievable in a 300-meter bed regardless of how many layers are stacked!**
3. **Minimum Required Bed Length ($L_{\min}$):**
   To make $\sigma_{\text{spec}} = 0.08\%$ feasible, $r_{\min}$ must satisfy:
   $$r_{\min} \le \frac{\sigma_{\text{spec}}^2}{\sigma_{\text{in}}^2} = \frac{0.0064}{0.2304} = 0.02778$$
   $$r_{\min} = \frac{3 a}{8 L_{\min}} \le 0.02778 \implies L_{\min} \ge \frac{3 \times 60}{8 \times 0.02778} = \frac{180}{0.2222} = 810\text{ meters}$$
   The stockyard length must be extended to at least $810\text{ meters}$ to break the spatial autocorrelation of the blast-hole grades. Alternatively, install an upstream surge bin pre-blending system or online PGNAA trimming.

---

### Self-Check for Exercise 5: Chevron vs. Windrow Stacking Selection
**Recommendation: Windrow Stacking.**
- **Physical Segregation Control:** With a large top size ($P_{95} = 120\text{ mm}$), Chevron stacking would cause severe coarse boulder roll-down to the pile toes. When the bridge reclaimer cuts across the bed, the outer toes will deliver slugs of coarse, hard rock, followed by slugs of fine clay from the apex, causing extreme comminution and froth stability upsets. Windrow stacking traps coarse particles within dozens of localized sub-windrows, suppressing cross-pile segregation.
- **Moisture and Clay Dynamics:** The high clay content ($14\%$) under tropical rainfall risks pile slope slumping and face gouging. In Chevron stacking, rainwater pools in the central core; Windrow stacking provides multi-channel drainage.
- **Reclaimer Operation:** The bridge reclaimer's reciprocating rake functions optimally when particle sizing is homogeneously distributed across the triangular face. Windrow stacking prevents heavy toe avalanches that trip rake drive overload alarms. Despite higher stacker mechanical complexity, Windrow stacking is necessary to protect plant recovery.

---

### Self-Check for Exercise 6: Online PGNAA Dynamic Trim Sizing
1. **Target Trim Calculation:**
   - Main flow $Q_{\text{main}} = 4,000\text{ t/h}$, $g_{\text{main}} = 6.8\% \text{ SiO}_2$.
   - Trim grade $g_{\text{trim}} = 1.2\% \text{ SiO}_2$, Target grade $g^* = 4.5\% \text{ SiO}_2$.
   $$u_{\text{trim}}^* = Q_{\text{main}} \cdot \frac{g_{\text{main}} - g^*}{g^* - g_{\text{trim}}} = 4,000 \cdot \frac{6.8 - 4.5}{4.5 - 1.2} = 4,000 \cdot \frac{2.3}{3.3} = 2,787.88\text{ t/h}$$
2. **Evaluation with Feeder Mechanical Limit ($u_{\max} = 800\text{ t/h}$):**
   - The required trim ($2,787.9\text{ t/h}$) exceeds the feeder limit ($800\text{ t/h}$). The controller clamps trim rate to $u = 800\text{ t/h}$.
   - Blended mass flow entering plant = $4,000 + 800 = 4,800\text{ t/h}$.
   - Resulting blended silica grade:
     $$g_{\text{final}} = \frac{(4,000 \times 6.8) + (800 \times 1.2)}{4,800} = \frac{27,200 + 960}{4,800} = \frac{28,160}{4,800} = 5.8667\% \approx 5.87\% \text{ SiO}_2$$
   - The feeder saturation prevents hitting the $4.5\%$ setpoint, but still dampens the silica disturbance from $6.80\%$ down to $5.87\%$. The control system must throttle main line reclaim rate $Q_{\text{main}}$ to $1,148\text{ t/h}$ to achieve full compliance if $800\text{ t/h}$ is binding.

---

## 14. Interview questions

### Question 1: Why does linear blending optimization based on Wet Metric Tonnes (WMT) produce systematic head grade reconciliation shortfalls in mineral concentrators?
- **What a strong answer contains:** Explains that payable and deleterious metal contents are physically bound to the dry crystalline rock matrix, not the pore water. Formulating grade balance constraints using WMT weights implicitly assigns metal grades to moisture. When blending an ore with $12\%$ moisture with an ore at $4\%$ moisture, the dry mass delivered from the wet parcel is substantially lower than its wet weight implies ($D = W(1 - m)$). Consequently, the actual dry mass-weighted grade entering the plant deviates from the LP solution, causing predictable grade and metal shortfalls in reconciliation ($F_2$ factor shortfalls).
- **What a weak answer reveals:** Believes moisture is merely a freight cost concern and assumes metal percentages average linearly across total mass regardless of water content.

### Question 2: In bed blending theory, why can a 400-layer longitudinal Chevron stockpile fail to achieve the theoretical $1/\sqrt{N_L}$ standard deviation reduction?
- **What a strong answer contains:** Identifies two primary physical/statistical mechanisms: (1) **Spatial Autocorrelation:** Ore grades are not independent random variables; they follow spatial variograms. If the variogram range $a$ is significant relative to bed length $L$, long-range grade cycles pass directly through the bed, imposing an asymptotic variance reduction floor governed by Gy's integral ($r_{\min} \approx \frac{3a}{8L}$). (2) **Radial Segregation:** Heavy boulders roll to the pile toes while fines stay at the apex. Unless the reclaimer employs a full-face reciprocating rake and continuous multi-depth cutting, cross-sectional segregation converts spatial variance into periodic time-series variance.
- **What a weak answer reveals:** Attributes the shortfall solely to mechanical breakdowns or assumes more layers always guarantee proportional variance reduction.

### Question 3: How do you linearize a smelter contract penalty step function with escalating tiers and a rejection threshold in a Mixed-Integer Linear Program?
- **What a strong answer contains:** Details two mathematical formulations: (1) **Special Ordered Sets of Type 2 (SOS2):** Define grid points at the penalty threshold boundaries, compute cumulative dollar penalties at each point, and link grades through convex combination weights $\lambda_k$ where at most two adjacent weights can be non-zero. Notes that if the penalty curve is convex ($p_{s+1} \ge p_s$), standard LP solvers find the global optimum without integer branching. (2) **Rejection Limit Handling:** Truncates the grid strictly at the rejection threshold ($T_K = T_{\text{reject}}$) to treat rejection as a hard constraint.
- **What a weak answer reveals:** Suggests using crude average linear penalty slopes or attempts to solve the model using non-linear heuristic solvers that fail to guarantee optimality.

### Question 4: Compare Chevron stacking to Windrow stacking. What mechanical, metallurgical, and economic factors dictate the choice?
- **What a strong answer contains:** 
  - *Chevron:* Requires only a traveling stacker on a fixed linear axis; lowest capital cost. However, it induces extreme radial particle segregation (fines at core, coarse at toes). Suitable for homogeneous ores with narrow particle size distributions or fine concentrates.
  - *Windrow:* Requires a slewing and luffing stacker boom to deposit multiple parallel longitudinal rows. Higher capital and maintenance cost, but dramatically suppresses segregation by limiting coarse roll-down. Mandatory for heterogeneous ores with wide particle size distributions, sticky clay ores, or operations lacking bridge face reclaimers.
- **What a weak answer reveals:** Assumes Chevron and Windrow are interchangeable layout preferences with no impact on comminution or plant metallurgy.

### Question 5: What is the physical operational advantage of an online PGNAA analyzer over a high-throughput X-ray Fluorescence (XRF) conveyor scanner?
- **What a strong answer contains:** Explains penetration physics. XRF utilizes low-energy X-rays that penetrate only $10 - 50\text{ micrometers}$ into rock surfaces. Consequently, XRF measurements are easily corrupted by dust coatings, surface moisture films, and coarse boulder shielding. PGNAA bombards the material with thermal neutrons that penetrate the entire $200 - 300\text{ mm}$ bed profile on a moving conveyor belt, measuring full bulk volumetric capture gamma spectra. This makes PGNAA robust against moisture, dust, and particle segregation at industrial belt tonnages ($> 5,000\text{ t/h}$).
- **What a weak answer reveals:** Confuses radiometric neutron capture with surface X-ray reflection, or believes both instruments have identical measurement capabilities.

### Question 6: Under the IMO IMSBC Code, what is the Transportable Moisture Limit (TML), and how does it constrain coastal concentrate blending?
- **What a strong answer contains:** Identifies TML as the maximum moisture content ($90\%$ of the Flow Moisture Point, FMP) at which granular bulk mineral cargoes can be safely shipped without risk of dynamic liquefaction under maritime cyclic wave shear. Explains that coastal port blending cannot treat moisture merely as an economic mass deduction. If high-moisture concentrate parcels are blended into an export vessel cargo and the composite moisture exceeds the certified TML, the maritime port authority will prohibit vessel departure.
- **What a weak answer reveals:** Views moisture purely as a shipping penalty charge and is unaware of the structural liquefaction risk to bulk cargo carriers.

### Question 7: Why is the Silica-to-Alumina ratio ($\text{SiO}_2 / \text{Al}_2\text{O}_3$) critical in pyrometallurgical blending, and how do you enforce it as a linear inequality in an LP?
- **What a strong answer contains:** Explains that blast furnaces and copper smelters require precise slag basicity and viscosity control. Excess alumina elevates slag viscosity and liquidus temperature, causing metal entrapment and tapping freezing, while silica acts as an acidic network former requiring basic limestone flux ($\text{CaO}$) additions. Derives the linear transformation:
  $$\frac{\sum g_{i, \text{SiO}_2} x_i}{\sum g_{i, \text{Al}_2\text{O}_3} x_i} \ge R_{\min} \implies \sum (g_{i, \text{SiO}_2} - R_{\min} g_{i, \text{Al}_2\text{O}_3}) x_i \ge 0$$
- **What a weak answer reveals:** States that impurities should simply be minimized, without understanding slag thermodynamics or the mathematical cross-multiplication required for linear programming.

### Question 8: How does an auxiliary dynamic trim hopper operate in conjunction with a Model Predictive Controller (MPC) on a reclaimed stockyard stream?
- **What a strong answer contains:** Describes a feedforward/feedback closed-loop architecture. An online PGNAA analyzer measures elemental grades on the main reclaim belt. When grade deviations occur (e.g., an unexpected arsenic or silica spike exiting the stockpile), the MPC predicts downstream arrival times and modulates a variable frequency drive on the trim surge hopper. The trim hopper discharges a high-grade or low-impurity corrective ore to neutralize the spike in real time before the stream enters the grinding circuit.
- **What a weak answer reveals:** Suggests stopping the entire conveyor belt whenever an off-spec grade is detected.

---

## 15. Key terms

- **Dry Metric Tonne (DMT):** Commercial moisture-free mass baseline for mineral commodities.
- **Wet Metric Tonne (WMT):** As-weighed physical mass containing free moisture.
- **Transportable Moisture Limit (TML):** Statutory maximum moisture ($90\%$ of FMP) to prevent vessel liquefaction.
- **Flow Moisture Point (FMP):** Moisture percentage at which bulk ore begins to exhibit fluid plastic behavior under cyclic shear.
- **Deleterious Elements:** Non-payable impurities ($\text{As}, \text{Sb}, \text{Bi}, \text{Hg}, \text{Cl}, \text{F}$) penalized under commercial contracts.
- **Treatment & Refining Charges (TC/RC):** Smelter processing fees charged per DMT of concentrate and pound/ounce of refined metal.
- **Penalty Step Function:** Discontinuous, escalating financial deductions for impurities exceeding baseline thresholds.
- **Special Ordered Set of Type 2 (SOS2):** MILP mathematical construct linearizing continuous piecewise functions.
- **Chevron Stacking:** Longitudinal bedding method depositing layers along a fixed central apex.
- **Windrow Stacking:** Multi-ridge bedding method suppressing cross-sectional coarse particle segregation.
- **Bed Blending Ratio ($r$):** Stockpile variance reduction metric ($\sigma_{\text{out}}^2 / \sigma_{\text{in}}^2$).
- **Pierre Gy Variance Integral:** Geostatistical formula linking bed variance reduction to spatial autocorrelation.
- **Autocorrelation Function ($\rho(h)$):** Spatial continuity parameter defining grade correlation over lag distance $h$.
- **Transversal Reclaiming:** Slicing perpendicular to longitudinal stockpile layers to combine all bedding passes.
- **Reclaimer Rake / Harrow:** Oscillating triangular lattice driving uniform face avalanching.
- **Prompt Gamma Neutron Activation Analysis (PGNAA):** Volumetric online cross-belt elemental sensing technology.
- **Thermal Neutron Capture:** Nuclear reaction where nuclei absorb low-energy neutrons and emit prompt gammas.
- **Dynamic Trimming:** Real-time auxiliary feeder injection to neutralize high-frequency grade spikes.
- **Model Predictive Control (MPC):** Advanced process control algorithm optimizing actuator trajectories over receding horizons.
- **Basicity Ratio ($B_2, B_4$):** Thermodynamic ratio of basic oxides to acidic oxides in furnace slag.
- **Kinetic Sieving:** Particle segregation mechanism where fines drop through voids during bulk cascading.
- **Angle of Repose:** Maximum stable slope angle of granular bulk materials ($\sim 34^\circ - 38^\circ$).
- **End-Cone Dilution:** Variance reduction collapse at the sloped extremities of longitudinal stockpiles.
- **IMSBC Code:** International maritime safety regulations governing dry bulk cargo shipping.
- **ISO 12743:** International standard governing mechanical sampling of base metal concentrates.
- **ISO 3082:** International standard for mechanical sampling and preparation of iron ores.
- **GB 20424-2006:** Mandatory Chinese national import threshold standard for base metal concentrates.
- **Swell Factor:** Volumetric expansion ratio of solid rock upon blasting and excavation ($S \ge 1.0$).
- **Loose Bulk Density ($\rho_{\text{bulk}}$):** Mass per unit volume of uncompacted ore in stockpiles ($\text{t/m}^3$).
- **Net Realized Value (NRV):** Commercial revenue remaining after mining, logistics, processing, and penalty deductions.

---

## 16. Sources

- **Gy, P. (1982).** *Sampling of Particulate Materials: Theory and Practice.* Elsevier Scientific Publishing, Amsterdam. ISBN: 978-0-444-41826-5. [Foundational derivation of sampling theory, fundamental error, and bed blending variance reduction integrals].
- **Pitard, F. F. (1993).** *Pierre Gy's Sampling Theory and Sampling Practice: Heterogeneity, Sampling Correctness, and Statistical Process Control.* 2nd Edition, CRC Press. ISBN: 978-0-8493-8653-4. [Detailed mathematical formulation of stockpile layering, segregation error, and variance reduction factors].
- **Schofield, C. G. (1980).** *Homogenisation/Blending Systems Design and Control for Minerals Processing.* Trans Tech Publications, Clausthal-Zellerfeld. ISBN: 978-0-87849-030-1. [Engineering mechanics of Chevron vs Windrow stacking geometry and bridge reclaimer design].
- **Robinson, P. C. (2006).** *Smelter Contracts and Impurity Penalties.* In: *Mineral Processing Plant Design, Practice, and Control*, Mular, A. L., Halbe, D. N., and Barratt, D. J. (Eds.), Society for Mining, Metallurgy, and Exploration (SME), Littleton, CO. ISBN: 978-0-87335-223-9. [Commercial penalty structures, TC/RC mechanics, and rejection thresholds].
- **International Maritime Organization (2020).** *IMSBC Code: International Maritime Solid Bulk Cargoes Code and Supplement.* IMO Publishing, London. Resolution MSC.268(85). ISBN: 978-92-801-1698-4. [Statutory standards for Transportable Moisture Limit (TML) and liquefaction risk management].
- **International Organization for Standardization (2021).** *ISO 12743:2021 — Copper, lead, zinc and nickel concentrates — Sampling procedures for determination of metal and moisture content.* ISO, Geneva. [Mechanical cross-stream cutter design and moisture determination rules].
- **Foster, R., & Williams, S. (2014).** *Online Elemental Analysis in Mineral Processing Using PGNAA Technology.* In: *Proceedings of the 12th AusIMM Mill Operators' Conference*, Townsville, Australia, pp. 245–254. [Cross-belt radiometric physics and closed-loop dynamic trim control implementations].
- **General Administration of Quality Supervision, Inspection and Quarantine of the People's Republic of China (2006).** *GB 20424-2006: Limits of deleterious elements in copper concentrates.* Standards Press of China, Beijing. [Binding statutory import rejection thresholds].

---

## 17. Links

- **Upward Index:** [Module 39 — Mining, Metals, and Forestry Supply Chains](_index.md)
- **Preceding Sibling Topic 01:** [Reserves, Grade, and Confidence](01_reserves-grade-confidence.md)
- **Preceding Sibling Topic 02:** [Mine Planning and Cut-Off Grade Optimization](02_mine-planning-cutoff.md)
- **Preceding Sibling Topic 03:** [Drilling, Blasting, and Load-Haul Operations](03_drilling-blasting-loadhaul.md)
- **Subsequent Sibling Topic 05:** [Processing and Recovery](05_processing-recovery.md)
- **Subsequent Sibling Topic 06:** [Pit-to-Port Logistics](06_pit-to-port-logistics.md)
- **Subsequent Sibling Topic 07:** [Stockpiles and Bulk Inventory](07_stockpiles-inventory.md)
- **Subsequent Sibling Topic 08:** [Commodity Price and Offtake](08_commodity-offtake.md)
- **Root Survey File:** [Module 39 Survey](../../../modules/39_mining_metals_forestry.md)
