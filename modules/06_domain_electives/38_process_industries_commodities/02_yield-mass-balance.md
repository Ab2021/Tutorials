# Yield and Mass Balance Modeling in Process Industries and Commodities

> Module 38 — Process Industries, Commodities, and Bulk Supply Chains
> Topic 02 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 15 (Stochastic Inventory & Risk-Pooling), Module 16 (Supply Chain Network Economics), Module 20 (Operations Planning & Master Scheduling), Module 38 Topic 01 (Continuous, Batch, and Discrete Production)
> Version 2026.5

---

## 1. Why this topic exists

In discrete manufacturing, product assembly is governed by an integer bill-of-materials (BOM). One bicycle frame combined with two wheels, two pedals, and one chain produces exactly one bicycle. Component quantities are deterministic, discrete, and physically invariant. If physical inventory diverges from system records, the discrepancy is attributed to cycle-count errors, pilferage, or unrecorded scrap.

Process industries—including petroleum refining, petrochemicals, base metals smelting, industrial biotechnology, bulk chemicals, dairy processing, and grain milling—operate under fundamentally different physical and thermodynamic principles:
1. **Heterogeneous, Variable Feedstocks:** Raw materials are complex, variable chemical mixtures rather than standardized parts. Crude oil varies continuously in American Petroleum Institute (API) gravity, sulfur content, metals concentration, and distillation profile. Bauxite varies in reactive alumina-to-silica ratios. Raw milk fluctuates seasonally in protein-to-fat ratios.
2. **Divergent, Joint Transformations:** One feedstock stream splits into multiple co-products, by-products, and waste streams across continuous thermodynamic separators, catalytic reactors, and cracking furnaces. A refinery cannot choose to produce only ultra-low sulfur diesel without simultaneously producing naphtha, liquefied petroleum gas (LPG), heavy fuel oil, and petroleum coke.
3. **Volume Non-Additivity:** Blending liquid hydrocarbons or chemical solvents exhibits excess volume of mixing ($\Delta V_{mix} \neq 0$). Mixing 1,000 barrels of light condensate with 1,000 barrels of heavy crude does not yield 2,000 barrels of blend; molecular interstitial packing causes volumetric shrinkage of 0.3% to 1.5%, rendering volumetric accounting physically non-conservative.
4. **Permasvasive Measurement Error:** Process flowmeters (differential pressure orifice plates, Coriolis meters, vortex shedding meters, ultrasonic flowmeters) and tank strapping gauges are corrupted by random measurement noise, calibration drift, fouling, ambient temperature swings, and two-phase fluid regimes.

Consequently, if an engineer sums the raw, unadjusted sensor measurements across any chemical plant, refinery, or dairy processing unit, the conservation of mass is violated:
$$\sum y_{in} \neq \sum y_{out}$$

Raw plant data routinely exhibits material balance closure errors of $\pm 2\%$ to $\pm 8\%$. In a 250,000 barrel-per-day (bpd) refinery processing $\$15\text{ billion}$ of crude oil per year, an unresolved 1.5% mass imbalance represents $\$225\text{ million}$ per year in unaccounted hydrocarbons, undetected pipeline leaks, or distorted yields.

The operational and financial damage of relying on unbalanced raw data cascades across the entire supply chain:
- **Planning Model Breakdown:** Linear Programming (LP) and Nonlinear Programming (NLP) scheduling systems (e.g., Aspen PIMS, Honeywell Haverly GRTMPS) rely on empirical yield vectors to optimize crude purchasing and unit throughput. If fed unadjusted sensor data, LP solvers generate phantom profit margins, recommend physically infeasible operating modes, or trigger stockouts of intermediate storage tanks.
- **Distorted Joint Cost Accounting:** Under International Financial Reporting Standards (IFRS 15) and US GAAP ASC 330, joint production costs must be allocated across co-products at the split-off point using physical measures, sales value, or Net Realizable Value (NRV). Flawed mass balances misstate inventory carrying values, distort product-line gross margins, and misprice commercial off-take contracts.
- **Regulatory and Environmental Penalties:** Under US EPA 40 CFR Part 98 (Subpart Y) and the European Union Emissions Trading System (EU ETS Monitoring and Reporting Regulation EU 2018/2066), refineries and smelters are legally obligated to prove emissions and flared mass through rigorous carbon and mass balances. Unreconciled losses are legally classified as fugitive emissions, exposing operators to multi-million-dollar statutory fines.

Mastering yield and mass balance modeling—specifically the integration of physical conservation laws, crude distillation cut curves, Generalized Least Squares (GLS) data reconciliation, statistical Gross Error Detection (GED), and co-product cost allocation—provides the mathematical foundation for plant accounting, operational planning, and supply chain optimization.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Formulate** steady-state and dynamic mass, atomic component, and enthalpy balance conservation equations across complex multi-unit process flow networks.
2. **Characterize** crude assay True Boiling Point (TBP) distillation curves and derive pseudo-component yield fractions for atmospheric and vacuum distillation units under shifting cut-point temperatures.
3. **Construct** the measurement error model and sensor covariance matrix ($\Sigma$) reflecting varying instrumentation technologies (Coriolis, ultrasonic, orifice differential pressure, and tank gauging).
4. **Derive** the analytical closed-form solution to linear steady-state Data Reconciliation (DR) using Generalized Least Squares (GLS), Lagrange multipliers, and projection operators.
5. **Execute** statistical Gross Error Detection (GED) using the Global Test ($\chi^2$-statistic) and measurement standardized residuals ($Z$-scores) to isolate biased flowmeters and pipeline leaks.
6. **Formulate** Crowe's projection method to reconcile flow networks containing unmeasured observable process streams and isolate unmeasured unobservable sub-networks.
7. **Apply** robust estimation loss functions (Huber M-estimator, Cauchy loss) to prevent gross error smearing across redundant measurement graphs.
8. **Compute** joint production cost allocations across divergent co-products at split-off points using Physical Measures, Sales Value at Split-off, and Net Realizable Value (NRV) methods compliant with IFRS and ISO 14044.

---

## 3. The core mental model

The flow of mass through a process supply chain is governed by a fundamental duality: **Physical First Principles** (which strictly dictate that mass, atoms, and energy are conserved) and **Empirical Sensor Telemetry** (which is corrupted by random noise, systematic calibration biases, and unmetered pipeline leaks).

Data reconciliation is not arbitrary "data smoothing" or curve-fitting. It is the maximum-likelihood orthogonal projection of corrupted empirical sensor observations onto the physically feasible hyperplane defined by conservation laws.

```text
=============================================================================================================
             PROCESS SUPPLY CHAIN MASS RECONCILIATION, YIELD, AND PLANNING ARCHITECTURE
=============================================================================================================

 [ RAW COMMODITY INPUT ]
   Crude Oil Assays / Bulk Ores / Raw Chemicals
            │
            ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ PHYSICAL CONVERSION & SEPARATION TOPOLOGY                                                               │
 │ - Continuous Distillation (TBP Cut Curves: LPG, Naphtha, Jet, Diesel, Gas Oil, Residue)                 │
 │ - Reactor Conversion & Cracking (Component / Atomic Conservation)                                      │
 │ - Blending Operations (Volume Non-Additivity: Excess Volume of Mixing Delta_V_mix)                      │
 └───────────────────────────────────┬─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ MEASUREMENT & TELEMETRY LAYER                                                                           │
 │ - Sensors: Coriolis (0.1%), Ultrasonic (0.5%), Orifice DP (1.5-2.5%), Tank Gauging (1.0%)              │
 │ - Observation Model: y = x + epsilon + delta   (epsilon ~ N(0, Sigma); delta = Gross Error / Bias)      │
 │ - Raw Plant Reality: A * y = r != 0 (Mass Conservation Violated by Imbalance Residual r)                │
 └───────────────────────────────────┬─────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
 ┌─────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │ DATA RECONCILIATION & GROSS ERROR DETECTION ENGINE                                                      │
 │                                                                                                         │
 │   1. Objective:  min 0.5 * (y - x)^T * Sigma^-1 * (y - x)   subject to   A * x = 0                      │
 │                                                                                                         │
 │   2. Global Chi-Square Test:                                                                            │
 │      V = A * Sigma * A^T;   gamma = r^T * V^-1 * r ~ Chi^2(df)                                          │
 │      Decision: If gamma > Chi^2_crit  ==>  Gross Error or Unmetered Leak Detected                       │
 │                                                                                                         │
 │   3. Local Measurement Z-Test:                                                                          │
 │      Q = Sigma * A^T * V^-1 * A * Sigma;   z_i = |y_i - x_hat_i| / sqrt(Q_ii)                           │
 │      Decision: If max(z_i) > z_crit   ==>  Isolate / Deweight Sensor i / Eliminate Bias                 │
 │                                                                                                         │
 │   4. Reconciled Flow Estimator:                                                                         │
 │      x_hat = [ I - Sigma * A^T * (A * Sigma * A^T)^-1 * A ] * y    ==>   A * x_hat = 0.000 (Exact)      │
 └───────────────────────────────────┬─────────────────────────────────────────────────────────────────────┘
                                     │
                 ┌───────────────────┴───────────────────┐
                 ▼                                       ▼
 ┌──────────────────────────────────────┐ ┌────────────────────────────────────────────────────────────────┐
 │ DOWNSTREAM SUPPLY CHAIN PLANNING     │ │ FINANCIAL CO-PRODUCT COST ALLOCATION                           │
 │ - Reconciled LP Yield Vectors (PIMS) │ │ - Joint Cost Apportionment at Split-off                        │
 │ - Feedstock S&OP Scheduling          │ │ - Net Realizable Value (NRV) vs Sales Value Allocation         │
 │ - Refinery Capacity Utilization      │ │ - True Gross Margins & Ending Inventory Valuation (IFRS 15)    │
 └──────────────────────────────────────┘ └────────────────────────────────────────────────────────────────┘
=============================================================================================================
```

The model establishes that **reconciliation must precede planning and costing**. If planning models are fed unadjusted sensor data $y$, the linear program solves over a corrupted feasible region, yielding phantom arbitrage opportunities. If financial systems allocate costs based on unadjusted data, cost variances fluctuate erratically due to instrumentation drift rather than true operational efficiency.

---

## 4. Definitions and vocabulary

| Term | Notation / Symbol | Formal Definition | Common Misconception / Misuse | Industrial Unit / Standard |
|---|---|---|---|---|
| **Mass Balance** | $A x = 0$ | The conservation principle requiring that total mass entering a closed system equals total mass exiting under steady-state conditions. | Confusing mass balance with volume balance; volume is non-conservative across mixtures. | Metric tons per hour ($\text{MT/h}$) or $\text{kg/s}$ |
| **Component Balance** | $A (W \odot x) = 0$ | Conservation applied to individual chemical or atomic species across physical separations or stoichometrically balanced reactions. | Assuming component mass fractions remain invariant when chemical reactions occur. | Weight fraction ($\text{wt}\%$) or $\text{kg/h}$ |
| **Enthalpy Balance** | $\sum h_i x_i + Q - W = 0$ | First law of thermodynamics balance accounting for thermal energy, sensible heat, latent heat, and reaction enthalpy. | Neglecting phase-change latent heats (e.g., steam vaporization) in balance closures. | Gigajoules per hour ($\text{GJ/h}$) or $\text{MW}$ |
| **Excess Volume of Mixing** | $\Delta V_{mix}$ | The difference between the actual volume of a mixture and the sum of the unmixed constituent volumes. | Assuming 100 bbl of light naphtha plus 100 bbl of heavy oil equals 200 bbl of blend (shrinkage occurs). | Percentage shrinkage ($\%$) or barrels ($\text{bbl}$) |
| **True Boiling Point (TBP)** | $T_{TBP}(x_w)$ | Batch distillation curve measured with 15 theoretical plates and a 5:1 reflux ratio representing true component volatility. | Treating standard lab ASTM D86 distillation temperatures as interchangeable with TBP. | Degrees Celsius ($^\circ\text{C}$) / ASTM D2892 |
| **ASTM D86 Distillation** | $T_{D86}$ | Rapid single-stage batch laboratory distillation without fractionation, used for product quality screening. | Using ASTM D86 directly in thermodynamic simulation without empirical conversion correlations. | Degrees Celsius ($^\circ\text{C}$) / ASTM D86 |
| **Simulated Distillation** | SimDist | Gas chromatography (GC) technique separating hydrocarbons in boiling point order up to $750^\circ\text{C}$. | Confusing physical liquid vaporization with chromatographic retention time elution. | Weight $\%$ vs $^\circ\text{C}$ / ASTM D2887 |
| **Pseudo-Component** | — | A lumped, fictitious single chemical component representing a complex boiling range cut with averaged properties. | Believing pseudo-components have fixed chemical formulas; properties shift with crude assay. | Lumped boiling cut ($^\circ\text{C}$) |
| **Distillation Cut Point** | $T_{cut}$ | The vapor temperature setpoint at which liquid fractions are separated in a fractionation column. | Assuming cut points are sharp knife-edge splits; real distillation exhibits significant tray overlap. | Degrees Celsius ($^\circ\text{C}$) or $^\circ\text{F}$ |
| **Atmospheric Residue** | AR / Long Residue | Heavy bottoms stream exiting the base of an Atmospheric Distillation Unit, boiling above $340^\circ\text{C} - 370^\circ\text{C}$. | Misclassifying AR as finished asphalt; it requires vacuum distillation or coking. | Barrels per stream day ($\text{BPSD}$) or $\text{MT/h}$ |
| **Vacuum Residue** | VR / Short Residue | Bottoms stream from a Vacuum Distillation Unit, boiling above $540^\circ\text{C} - 565^\circ\text{C}$. | Confusing vacuum residue with vacuum gas oil (VGO). | $\text{MT/h}$ / API gravity $< 10$ |
| **Data Reconciliation (DR)** | DR | Mathematical optimization adjusting raw measurements to satisfy physical conservation constraints minimally. | Treating data reconciliation as arbitrary data filtering without conservation constraints. | Mathematical algorithm / ISO 5167 |
| **Generalized Least Squares** | GLS | Weighted least-squares estimation where weights equal the inverse of the measurement error covariance matrix ($\Sigma^{-1}$). | Using ordinary least squares (OLS) with equal weights, ignoring sensor accuracy disparities. | Statistical formulation |
| **Measurement Covariance** | $\Sigma \in \mathbb{R}^{n \times n}$ | Matrix containing sensor error variances on the diagonal ($\sigma_i^2$) and sensor cross-correlations off-diagonal. | Assuming $\Sigma$ is identical for all meters; Coriolis is $10\times$ more accurate than orifice plates. | $(\text{MT/h})^2$ |
| **Incidence Matrix** | $A \in \mathbb{R}^{m \times n}$ | Directed graph matrix connecting $m$ process nodes with $n$ streams ($+1$ for inlet, $-1$ for outlet). | Inverting stream signs across nodes, destroying balance linearity. | Dimensionless network topology |
| **Spatial Redundancy** | — | The condition where a stream flow can be calculated from two or more independent combinations of other sensors. | Assuming measuring every stream is required for redundancy; graph topology determines redundancy. | Structural graph property |
| **Temporal Redundancy** | — | Using time-series dynamic integration across tank holdups to estimate steady-state flows. | Ignoring dynamic inventory accumulation ($\frac{dM}{dt}$) in tank nodes. | Moving average window (hours) |
| **Degree of Observability** | — | Classification determining whether an unmeasured stream can be uniquely calculated from existing measurements. | Assuming all unmeasured streams are unobservable. | Rank condition of sub-matrix |
| **Gross Error** | $\delta_i$ | Non-random systematic sensor bias, calibration error, instrument failure, or unmetered physical pipe leak. | Assuming gross errors can be eliminated by Gaussian averaging over time. | $\text{MT/h}$ or standard deviations |
| **Measurement Residual** | $e_i = y_i - \hat{x}_i$ | The physical difference between the raw measured flow rate and the reconciled mass flow rate. | Confusing measurement residual $e = y - \hat{x}$ with nodal imbalance residual $r = A y$. | Metric tons per hour ($\text{MT/h}$) |
| **Nodal Imbalance Residual** | $r = A y$ | The net algebraic sum of raw measured flows around a process node. | Assuming $r = 0$ in raw plant data; sensor noise guarantees $r \neq 0$. | Metric tons per hour ($\text{MT/h}$) |
| **Global Test Statistic** | $\gamma = r^T V^{-1} r$ | Quadratic statistical metric testing the null hypothesis that no gross errors exist in the network. | Comparing $\gamma$ against a normal distribution instead of a Chi-square distribution with $m$ degrees of freedom. | $\chi^2(\operatorname{rank}(A))$ distribution |
| **Standardized Residual** | $z_i = \frac{\|e_i\|}{\sqrt{Q_{ii}}}$ | Normalized measurement residual scaled by its theoretical standard error under null conditions. | Dividing residual $e_i$ by raw sensor variance $\sigma_i$ instead of residual variance $Q_{ii}$. | Dimensionless standard scores |
| **Gross Error Smearing** | — | Phenomenon where GLS spreads the effect of one massive sensor bias across all clean adjacent streams. | Concluding that all flagged streams are broken when one single upstream sensor is biased. | Network propagation artifact |
| **Huber M-Estimator** | $\rho_H(u)$ | Robust loss function transitioning from quadratic for small errors to linear for large errors to dampen outliers. | Believing robust estimators completely eliminate the need for gross error identification. | Mathematical loss function |
| **Split-off Point** | — | The distinct stage in a joint process where individual co-products become separately identifiable. | Allocating separable downstream costs back to the joint process split-off. | Process milestone / IFRS 15 |
| **Joint Products** | — | Two or more co-products of significant economic value generated simultaneously from a single input. | Classifying low-value by-products as joint products. | Commercial classification |
| **By-Product** | — | An incidental secondary output of minor sales value relative to the primary joint products (e.g., sulfur in refining). | Assigning full joint processing cost to by-products, causing artificial accounting losses. | Accounting classification |
| **Net Realizable Value** | NRV | Final product sales value minus separable downstream processing, finishing, and transportation costs. | Using unadjusted final retail prices without subtracting separable downstream costs. | $\text{USD}$ or $\text{USD/MT}$ |
| **Sales Value at Split-off** | — | Joint cost allocation method apportioning costs proportional to relative market value at the exact point of separation. | Applying split-off prices to products that require mandatory further downstream conversion. | Financial valuation method |
| **API Gravity** | $^\circ\text{API}$ | Inverse measure of petroleum liquid density relative to water: $^\circ\text{API} = (141.5 / \text{SG}) - 131.5$. | Assuming API gravity is linear on volumetric mixing (specific gravity is harmonic). | Degrees API ($^\circ\text{API}$) / ASTM D1298 |
| **Shrinkage Factor** | $S_f$ | The empirical fraction of volume lost during hydrocarbon blending or chemical condensation. | Ignoring shrinkage in pipeline custody transfer, resulting in fiscal settlement disputes. | Dimensionless volume fraction |

---

## 5. The mechanics, step by step

### 5.1 Step 1: Network Topology Formulation and Conservation Equations

A process plant is represented as a directed graph $G = (V, E)$, where vertices $V = \{1, 2, \dots, m\}$ represent processing units (reactors, distillation towers, splitters, mixers) and directed edges $E = \{1, 2, \dots, n\}$ represent process streams carrying mass.

The physical structure is encoded in the node-arc incidence matrix $A \in \mathbb{R}^{m \times n}$:
$$A_{ij} = \begin{cases} +1 & \text{if stream } j \text{ enters node } i \\ -1 & \text{if stream } j \text{ leaves node } i \\ 0 & \text{if stream } j \text{ is not connected to node } i \end{cases}$$

#### Total Mass Conservation
Under steady-state continuous operation, mass accumulation is zero ($\frac{dM_i}{dt} = 0$). For each node $i$:
$$\sum_{j=1}^n A_{ij} x_j = 0 \iff A x = 0$$
where $x \in \mathbb{R}^n$ is the vector of true mass flow rates ($\text{MT/h}$).

#### Component and Elemental Conservation
For a system with $K$ distinct chemical species or atomic elements (e.g., carbon, sulfur, hydrogen, iron), let $w_{j,k}$ denote the mass fraction of component $k$ in stream $j$. In non-reactive systems or atomic balance formulations:
$$\sum_{j=1}^n A_{ij} w_{j,k} x_j = 0, \quad \forall i \in V, \, \forall k \in \{1, \dots, K\}$$

In reactive networks where chemical species transform, species balances incorporate reaction rates $R_{i,k}$:
$$\sum_{j=1}^n A_{ij} w_{j,k} x_j + \sum_{r=1}^{N_{rxn}} \nu_{k,r} R_{i,r} M_{w,k} = 0$$
where $\nu_{k,r}$ is the stoichiometric coefficient and $M_{w,k}$ is molecular weight. Total atomic balances remain strictly conservative without reaction terms.

#### Volume Non-Additivity and Specific Gravity
Volumetric flows $F_j$ ($\text{m}^3\text{/h}$ or $\text{bbl/h}$) are non-conservative due to thermal expansion and molecular excess volume of mixing:
$$x_j = F_j \cdot \rho_j(T_j, P_j)$$
In petroleum systems, liquid density is governed by Specific Gravity ($\text{SG}$ at $60^\circ\text{F}/60^\circ\text{F}$):
$$\text{API} = \frac{141.5}{\text{SG}} - 131.5 \implies \text{SG} = \frac{141.5}{\text{API} + 131.5}$$
When light hydrocarbons dissolve into heavy oils, total blended volume satisfies:
$$V_{blend} = \sum_{j} V_j + \Delta V_{mix}, \quad \Delta V_{mix} < 0$$
All physical reconciliation calculations must be executed strictly in **mass units**. Volumetric flows are converted to mass prior to reconciliation and back-calculated using temperature-corrected densities.

---

### 5.2 Step 2: Feedstock Characterization and Non-Linear Distillation Cut Curves

Continuous separation of crude oil in an Atmospheric Distillation Unit (ADU) is governed by the feedstock True Boiling Point (TBP) curve. The TBP curve relates cumulative mass or volume yield to boiling temperature under 15 theoretical fractionation stages at a 5:1 reflux ratio.

```text
  Temperature (°C)
       ▲
   600 ┼─────────────────────────────────────────────────────────/  <-- Vacuum Residue (> 540°C)
       │                                                    /───┘
   500 ┼──────────────────────────────────────────────/─────┘       <-- Heavy Vacuum Gas Oil (HVGO)
       │                                        /─────┘
   400 ┼──────────────────────────────────/─────┘                   <-- Light Vacuum Gas Oil (LVGO)
       │                            /─────┘                         <-- Atmospheric Residue (> 360°C)
   300 ┼──────────────────────/─────┘                               <-- Diesel / Gas Oil (240 - 360°C)
       │                /─────┘                                     <-- Kerosene / Jet Fuel (180 - 240°C)
   200 ┼──────────/─────┘                                           <-- Heavy Naphtha (90 - 180°C)
       │    /─────┘                                                 <-- Light Naphtha (35 - 90°C)
   100 ┼────┘                                                       <-- LPG / Off-Gas (< 35°C)
       └────┼──────────┼──────────┼──────────┼──────────┼──────────► Cumulative Mass Yield (%)
            0         20         40         60         80        100
```

#### Analytical Representation of TBP Curves
The cumulative mass fraction $Y(T)$ distilled up to temperature $T$ (Kelvin) is modeled using the Riazi-Daubert distribution:
$$Y(T) = 1 - \exp\left[ - \left( \frac{T - T_0}{a} \right)^b \right]$$
where $T_0$ is the Initial Boiling Point (IBP), and $a, b$ are crude-specific assay fitting parameters. In computational planning engines, high-order piecewise cubic splines are calibrated against laboratory assay data points.

#### ASTM D86 to TBP Interconversion
Standard laboratory product specifications utilize ASTM D86 single-stage distillation. Converting ASTM D86 temperatures ($T_{86}$) to True Boiling Point temperatures ($T_{TBP}$) utilizes the Daubert empirical correlations:
$$T_{TBP} = a \cdot (T_{86})^b$$
where temperatures are in Rankine or Kelvin, and parameters $a, b$ vary across distillation volume percent cuts (10%, 30%, 50%, 70%, 90%).

#### Pseudo-Component Slicing and Yield Derivatives
The mass yield fraction $w_k$ of pseudo-component cut $k$ bounded by lower cut temperature $T_{k}^{low}$ and upper cut temperature $T_{k}^{high}$ is:
$$w_k(T_{k}^{low}, T_{k}^{high}) = \int_{T_{k}^{low}}^{T_{k}^{high}} \frac{dY}{dT} dT = Y(T_{k}^{high}) - Y(T_{k}^{low})$$

When refinery schedulers shift cut points by adjusting column tray draw temperatures:
$$\frac{\partial w_k}{\partial T_{k}^{high}} = \left. \frac{dY}{dT} \right|_{T = T_{k}^{high}}, \quad \frac{\partial w_{k+1}}{\partial T_{k}^{high}} = - \left. \frac{dY}{dT} \right|_{T = T_{k}^{high}}$$
A positive adjustment of the Heavy Naphtha / Kerosene cut point increases Naphtha mass yield at the direct expense of Kerosene yield. However, the density, freeze point, and smoke point of Kerosene shift non-linearly, constraining downstream product blending specifications.

---

### 5.3 Step 3: Measurement Modeling and Sensor Covariance Matrix

Industrial telemetry measures true flows $x$ through physical instruments corrupted by random measurement errors and potential systematic biases:
$$y = x + \epsilon + \delta$$
where:
- $y \in \mathbb{R}^n$ is the raw measured flow rate vector ($\text{MT/h}$).
- $x \in \mathbb{R}^n$ is the true, unobserved physical mass flow rate.
- $\epsilon \sim \mathcal{N}(0, \Sigma)$ is the vector of zero-mean, Gaussian random measurement noise.
- $\delta \in \mathbb{R}^n$ is the vector of sparse gross errors (sensor decalibration, zero-drift, or unmetered leaks).

#### Instrumentation Error Hierarchy
The standard deviation $\sigma_i$ of sensor $i$ depends on instrument physics, calibration currency, and fluid phase:

| Instrument Technology | Relative Error ($\% \text{ of reading}$) | Typical Application | Nominal Standard Deviation $\sigma_i$ for $100\text{ MT/h}$ |
|---|---|---|---|
| **Coriolis Mass Meter** | $\pm 0.10\% - 0.20\%$ | Custody transfer, high-value chemicals | $0.15\text{ MT/h}$ |
| **Ultrasonic Flowmeter** | $\pm 0.30\% - 0.50\%$ | Pipeline bulk transport, refined products | $0.40\text{ MT/h}$ |
| **Turbine Flowmeter** | $\pm 0.25\% - 0.50\%$ | Light hydrocarbons, clean liquid fuels | $0.40\text{ MT/h}$ |
| **Orifice Plate (DP)** | $\pm 1.00\% - 2.50\%$ | Steam, crude distillation cuts, heavy bottoms | $1.80\text{ MT/h}$ |
| **Vortex Shedding Meter**| $\pm 0.75\% - 1.25\%$ | Flare gas, intermediate gas lines | $1.00\text{ MT/h}$ |
| **Tank Level Gauging** | $\pm 0.50\% - 1.50\%$ | Bulk atmospheric storage tanks | $1.00\text{ MT/h}$ |

#### Constructing the Covariance Matrix
Assuming sensors operate independently, the covariance matrix $\Sigma \in \mathbb{R}^{n \times n}$ is diagonal:
$$\Sigma = \operatorname{diag}(\sigma_1^2, \sigma_2^2, \dots, \sigma_n^2)$$
When shared differential pressure transmitters or multivariable density compensators are present, off-diagonal covariance entries $\Sigma_{ij} = \operatorname{Cov}(\epsilon_i, \epsilon_j) = \rho_{ij} \sigma_i \sigma_j$ must be populated.

---

### 5.4 Step 4: Observability, Redundancy, and Stream Classification

In industrial networks, not all streams possess dedicated flowmeters due to capital expenditure constraints or hostile operating environments. The stream vector $x$ partitions into measured streams $x_m \in \mathbb{R}^{n_m}$ and unmeasured streams $x_u \in \mathbb{R}^{n_u}$:
$$A_m x_m + A_u x_u = 0$$

Streams are categorized into four structural classifications:
1. **Measured Redundant:** A measured stream whose measurement can be removed without losing the ability to calculate its flow from remaining measurements. Only redundant streams can be reconciled and tested for gross errors.
2. **Measured Non-Redundant:** A measured stream that forms a unique cutset bridge. Its measured value must be accepted as exact because no redundant equations exist to check it.
3. **Unmeasured Observable:** An unmeasured stream whose true flow rate can be uniquely solved through algebraic combinations of surrounding measured streams.
4. **Unmeasured Unobservable:** An unmeasured stream that cannot be determined from available sensor data.

#### Crowe's QR Projection for Unmeasured Stream Elimination
To reconcile measured flows without introducing unmeasured variables into the optimization, we eliminate $x_u$ using matrix projection. Compute the QR decomposition of $A_u \in \mathbb{R}^{m \times n_u}$:
$$A_u = \begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \begin{bmatrix} R_1 \\ 0 \end{bmatrix}$$
where $\begin{bmatrix} Q_1 & Q_2 \end{bmatrix} \in \mathbb{R}^{m \times m}$ is orthogonal, $R_1 \in \mathbb{R}^{n_u \times n_u}$ is upper triangular (assuming $A_u$ has full column rank $n_u$), and $Q_2 \in \mathbb{R}^{m \times (m - n_u)}$ spans the left null space of $A_u$:
$$Q_2^T A_u = 0$$

Premultiplying the partitioned balance equation by $Q_2^T$:
$$Q_2^T A_m x_m + Q_2^T A_u x_u = 0 \implies C x_m = 0$$
where $C = Q_2^T A_m \in \mathbb{R}^{(m - n_u) \times n_m}$ is the reduced constraint matrix governing measured streams exclusively. Once reconciled estimates $\hat{x}_m$ are obtained, observable unmeasured streams are co-estimated via back-substitution:
$$R_1 x_u = - Q_1^T A_m \hat{x}_m \implies \hat{x}_u = - R_1^{-1} Q_1^T A_m \hat{x}_m$$

---

### 5.5 Step 5: Linear Data Reconciliation via Generalized Least Squares (GLS)

Assuming gross errors are absent ($\delta = 0$) and all streams are measured, the data reconciliation problem minimizes the weighted sum of squared measurement adjustments subject to strict mass balance conservation:
$$\min_{x} J(x) = \frac{1}{2} (y - x)^T \Sigma^{-1} (y - x) \quad \text{subject to } A x = 0$$

#### Analytical Derivation via Lagrange Multipliers
Formulate the Lagrangian function $\mathcal{L}(x, \lambda)$ with multiplier vector $\lambda \in \mathbb{R}^m$:
$$\mathcal{L}(x, \lambda) = \frac{1}{2} (y - x)^T \Sigma^{-1} (y - x) + \lambda^T A x$$

Compute first-order Karush-Kuhn-Tucker (KKT) stationarity conditions:
$$\nabla_x \mathcal{L} = - \Sigma^{-1} (y - x) + A^T \lambda = 0$$
$$\nabla_\lambda \mathcal{L} = A x = 0$$

Solve the stationarity equation for $x$:
$$\Sigma^{-1} (y - x) = A^T \lambda \implies y - x = \Sigma A^T \lambda \implies x = y - \Sigma A^T \lambda$$

Premultiply by incidence matrix $A$ and enforce $A x = 0$:
$$A x = A y - A \Sigma A^T \lambda = 0 \implies A \Sigma A^T \lambda = A y$$

Define the **nodal imbalance residual vector** $r \in \mathbb{R}^m$ and **nodal covariance matrix** $V \in \mathbb{R}^{m \times m}$:
$$r = A y$$
$$V = A \Sigma A^T$$

Since $A$ has full row rank and $\Sigma$ is positive definite, $V$ is symmetric positive definite and strictly invertible:
$$\lambda = V^{-1} r = (A \Sigma A^T)^{-1} A y$$

Substitute $\lambda$ back into the expression for $x$ to yield the **analytical GLS reconciled flow estimator**:
$$\hat{x} = y - \Sigma A^T (A \Sigma A^T)^{-1} A y = \left[ I - \Sigma A^T V^{-1} A \right] y$$

#### Statistical Properties of Reconciled Estimators
Define the projection operator $P = I - \Sigma A^T V^{-1} A$.
1. **Unbiasedness:** Since $\mathbb{E}[y] = x$ and $A x = 0$:
   $$\mathbb{E}[\hat{x}] = P \mathbb{E}[y] = P x = x - \Sigma A^T V^{-1} A x = x - 0 = x$$
2. **Reconciled Covariance Matrix:**
   $$\operatorname{Cov}(\hat{x}) = P \Sigma P^T = \Sigma - \Sigma A^T V^{-1} A \Sigma$$
3. **Variance Reduction Property:** For every stream $i$:
   $$\operatorname{Var}(\hat{x}_i) = \Sigma_{ii} - (\Sigma A^T V^{-1} A \Sigma)_{ii} \le \operatorname{Var}(y_i) = \sigma_i^2$$
   Reconciliation strictly reduces measurement uncertainty across all redundant network streams.

---

### 5.6 Step 6: Gross Error Detection (GED) and Statistical Testing Hierarchy

When a sensor decalibrates, fails, or a physical pipeline leak occurs, $\delta \neq 0$. If uncorrected, the quadratic penalty in GLS smears the gross error across all adjacent streams, corrupting clean measurements. Statistical hypothesis testing is conducted across two hierarchical tiers:

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ TIER 1: GLOBAL TEST (GT)                                                  │
│ Null Hypothesis H0: E[r] = 0 (No gross errors or unmetered leaks)         │
│ Test Statistic: gamma = r^T * V^-1 * r ~ Chi^2(df = m)                    │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                 ┌────────────────────┴────────────────────┐
                 ▼                                         ▼
         gamma <= Chi^2_crit                       gamma > Chi^2_crit
        [ PASS: Accept H0 ]                       [ FAIL: Reject H0 ]
     Reconciled flows valid                   Gross error / leak detected
                                                           │
                                                           ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ TIER 2: LOCAL MEASUREMENT TEST (MT)                                       │
│ Residual vector: e = y - x_hat = Sigma * A^T * V^-1 * r                   │
│ Residual Covariance: Q = Cov(e) = Sigma * A^T * V^-1 * A * Sigma          │
│ Standardized Residual: z_i = |e_i| / sqrt(Q_ii) ~ N(0, 1)                 │
│ Action: Isolate sensor k = argmax(z_i); if z_k > z_crit ==> Gross Error   │
└───────────────────────────────────────────────────────────────────────────┘
```

#### Tier 1: The Global Test (GT)
The raw nodal imbalance residual vector $r = A y$ follows a multivariate Gaussian distribution under null conditions:
$$r \sim \mathcal{N}(0, V), \quad V = A \Sigma A^T$$

The Global Test statistic $\gamma$ represents the Mahalanobis distance of nodal imbalances:
$$\gamma = r^T V^{-1} r$$

Under $H_0$, $\gamma$ follows a central Chi-square distribution with $m = \operatorname{rank}(A)$ degrees of freedom:
$$\gamma \sim \chi^2(m)$$

For significance level $\alpha$ (typically $\alpha = 0.05$ or $0.01$):
- If $\gamma \le \chi^2_{1-\alpha}(m)$: Fail to reject $H_0$. No gross error is detected; accept $\hat{x}$.
- If $\gamma > \chi^2_{1-\alpha}(m)$: Reject $H_0$. At least one gross error or physical leak exists in the network.

#### Tier 2: The Measurement Test (MT)
To identify the specific corrupted sensor, evaluate the measurement residual vector:
$$e = y - \hat{x} = \Sigma A^T V^{-1} r$$

The variance-covariance matrix of measurement residuals is:
$$Q = \operatorname{Cov}(e) = \Sigma A^T V^{-1} A \Sigma$$

The standardized measurement residual $z_i$ scales each residual by its theoretical standard error:
$$z_i = \frac{|e_i|}{\sqrt{Q_{ii}}}$$

Under $H_0$, each $z_i$ follows a standard normal distribution $\mathcal{N}(0, 1)$. For two-tailed significance level $\alpha$:
$$z_{crit} = \Phi^{-1}\left(1 - \frac{\alpha}{2}\right)$$
For $\alpha = 0.01$, $z_{crit} = 2.576$. For $\alpha = 0.0027$ ($3\sigma$), $z_{crit} = 3.000$.

#### Serial Elimination Algorithm
1. Reconcile network and compute Global Test statistic $\gamma$.
2. If $\gamma \le \chi^2_{1-\alpha}(m)$, terminate; reconciled flows are statistically valid.
3. If $\gamma > \chi^2_{1-\alpha}(m)$, compute all standardized residuals $z_i$.
4. Identify candidate stream with the maximum standardized residual: $k = \arg\max_{i} z_i$.
5. If $z_k > z_{crit}$, flag sensor $k$ as a gross error.
6. Reclassify stream $k$ as an unmeasured stream (or set its variance $\sigma_k^2 \to \infty$), re-partition matrix $A = [A_m \quad A_u]$, and re-run reconciliation via Crowe's projection method.
7. Repeat until the Global Test statistic passes.

---

### 5.7 Step 7: Robust Reconciliation via M-Estimators

When sensor networks suffer from multiple simultaneous gross errors, serial elimination can suffer from **masking** (a true error fails to be detected due to cancellation) or **swamping** (a clean sensor is falsely flagged due to smearing). Robust data reconciliation replaces the quadratic least-squares objective with an M-estimator loss function $\rho(\cdot)$ that bounds the influence of extreme residuals:

$$\min_{x} \sum_{i=1}^n \rho\left( \frac{y_i - x_i}{\sigma_i} \right) \quad \text{subject to } A x = 0$$

Common robust loss functions include:
1. **Huber Loss:**
   $$\rho_H(u) = \begin{cases} \frac{1}{2} u^2 & \text{for } |u| \le c \\ c |u| - \frac{1}{2} c^2 & \text{for } |u| > c \end{cases}$$
   where tuning constant $c = 1.345$ provides 95% asymptotic efficiency on Gaussian data while bounding outlier influence.
2. **Cauchy (Lorentzian) Loss:**
   $$\rho_C(u) = \frac{c^2}{2} \ln\left( 1 + \left(\frac{u}{c}\right)^2 \right)$$
   The influence function $\psi_C(u) = \rho_C'(u) = \frac{u}{1 + (u/c)^2}$ redescends to zero as $|u| \to \infty$, completely discarding extreme gross errors.

Robust reconciliation is solved via **Iteratively Reweighted Least Squares (IRLS)**. At iteration $k$, sensor variances are adjusted dynamically:
$$\sigma_{i,(k)}^2 = \frac{\sigma_i^2}{w(u_{i,(k-1)})}, \quad w(u) = \frac{\psi(u)}{u}$$
The problem solves as a standard linear GLS at each step until weights converge.

---

### 5.8 Step 8: Co-Product Economic Allocation and Split-Off Costing

Once mass flows $\hat{x}$ are reconciled and physically conservative, joint operating costs $C_{joint}$ (feedstock acquisition, thermal energy, distillation utilities, fixed plant overhead) must be allocated across outgoing co-products.

```text
                                        ┌──► Co-Product 1 (LPG / Off-gas): x_1, p_1
                                        │
                                        ├──► Co-Product 2 (Naphtha): x_2, p_2
[ Joint Feedstock & Processing Cost ]  ──┼──► Co-Product 3 (Kerosene / Jet): x_3, p_3
             C_joint                    │
                                        ├──► Co-Product 4 (Diesel / Gas Oil): x_4, p_4
                                        │
                                        └──► Co-Product 5 (Atmospheric Residue): x_5, p_5
                                        ▲
                                        │
                             [ SPLIT-OFF POINT ]
```

#### 1. Physical Measure Method
Costs are allocated strictly proportional to reconciled physical mass:
$$C_i^{phys} = C_{joint} \cdot \left( \frac{\hat{x}_i}{\sum_{j=1}^K \hat{x}_j} \right)$$
*Critique:* Distorts financial performance. Heavy atmospheric residue (a low-value fuel component selling below crude price) receives the identical per-ton cost allocation as high-value jet fuel, causing the heavy products business to show massive paper losses.

#### 2. Sales Value at Split-off Method
When co-products possess transparent market spot prices ($p_i$ in $\text{USD/MT}$) at the split-off point:
$$\text{Revenues}_i = \hat{x}_i \cdot p_i$$
$$C_i^{sales} = C_{joint} \cdot \left( \frac{\hat{x}_i \cdot p_i}{\sum_{j=1}^K \hat{x}_j \cdot p_j} \right)$$
*Properties:* Every co-product yields the identical gross profit margin percentage at split-off:
$$\text{Gross Margin}_i = \frac{\text{Revenues}_i - C_i^{sales}}{\text{Revenues}_i} = 1 - \frac{C_{joint}}{\sum_{j=1}^K \hat{x}_j \cdot p_j}$$

#### 3. Net Realizable Value (NRV) Method
When co-products cannot be sold at split-off and require separable downstream conversion (e.g., hydrotreating, catalytic cracking, isomerization):
$$NRV_i = \left( \hat{x}_i^{final} \cdot p_i^{final} \right) - C_i^{separable}$$
where $C_i^{separable}$ represents downstream processing, finishing, and logistics costs. Joint cost allocation follows:
$$C_i^{NRV} = C_{joint} \cdot \left( \frac{NRV_i}{\sum_{j=1}^K NRV_j} \right)$$

---

## 6. Worked example

### Problem Statement: Refinery Condensate Distillation Train
A petrochemical condensate distillation train fractionates raw condensate into petrochemical feeds and fuel components across three process units:
- **Node 1: Preheat & Desalter Unit**
- **Node 2: Atmospheric Fractionator Column**
- **Node 3: Naphtha Stabilizer Column**

The network contains $m = 3$ nodes and $n = 7$ process streams:
- **Stream 1 ($s_1$):** Raw Condensate Feed (inlet to Node 1)
- **Stream 2 ($s_2$):** Desalted Condensate (Node 1 to Node 2)
- **Stream 3 ($s_3$):** Desalter Brine Effluent (outlet from Node 1)
- **Stream 4 ($s_4$):** Intermediate Reactor/Overhead Feed (Node 2 to Node 3)
- **Stream 5 ($s_5$):** Internal Recycle Stream (Node 3 back to Node 2)
- **Stream 6 ($s_6$):** Finished Distillate Product (outlet from Node 3)
- **Stream 7 ($s_7$):** Stabilizer Off-Gas / LPG (outlet from Node 3)

```text
                 s1 (Raw Condensate Feed)
                          │
                          ▼
               ┌──────────────────────┐
               │ Node 1: Desalter     ├──► s3 (Brine Effluent)
               └──────────┬───────────┘
                          │
                          ▼ s2 (Desalted Condensate)
               ┌──────────────────────┐
               │ Node 2: Fractionator │◄───┐
               └──────────┬───────────┘    │ s5 (Recycle)
                          │                │
                          ▼ s4 (Overhead)  │
               ┌──────────────────────┐    │
               │ Node 3: Stabilizer   ├────┘
               └──────┬────────┬──────┘
                      │        │
                      ▼        ▼
                     s6        s7
            (Distillate)    (Off-Gas)
```

The node-arc incidence matrix $A \in \mathbb{R}^{3 \times 7}$ is:
$$A = \begin{bmatrix}
 1.0 & -1.0 & -1.0 &  0.0 &  0.0 &  0.0 &  0.0 \\
 0.0 &  1.0 &  0.0 & -1.0 &  1.0 &  0.0 &  0.0 \\
 0.0 &  0.0 &  0.0 &  1.0 & -1.0 & -1.0 & -1.0
\end{bmatrix}$$

### True Flows and Instrumentation Parameters
- True steady-state flows ($x_{true}$ in $\text{MT/h}$):
  $$x_{true} = [100.0, \, 85.0, \, 15.0, \, 110.0, \, 25.0, \, 75.0, \, 10.0]^T$$
  Check balances:
  - Node 1: $100.0 - 85.0 - 15.0 = 0.00$
  - Node 2: $85.0 - 110.0 + 25.0 = 0.00$
  - Node 3: $110.0 - 25.0 - 75.0 - 10.0 = 0.00$

- Sensor standard deviations ($\sigma$ in $\text{MT/h}$):
  $$\sigma = [1.5, \, 1.8, \, 0.6, \, 2.2, \, 1.0, \, 1.4, \, 0.5]^T$$
  The sensor covariance matrix $\Sigma = \operatorname{diag}(\sigma^2)$ is:
  $$\Sigma = \operatorname{diag}(2.25, \, 3.24, \, 0.36, \, 4.84, \, 1.00, \, 1.96, \, 0.25)$$

### Case A: Clean Telemetry with Random Noise
Raw sensor measurements $y_{clean}$ register:
$$y_{clean} = [101.20, \, 84.60, \, 15.30, \, 111.40, \, 24.80, \, 76.10, \, 9.80]^T$$

#### Step 1: Compute Nodal Imbalance Residuals ($r$)
$$r_1 = y_1 - y_2 - y_3 = 101.20 - 84.60 - 15.30 = +1.30\text{ MT/h}$$
$$r_2 = y_2 - y_4 + y_5 = 84.60 - 111.40 + 24.80 = -2.00\text{ MT/h}$$
$$r_3 = y_4 - y_5 - y_6 - y_7 = 111.40 - 24.80 - 76.10 - 9.80 = +0.70\text{ MT/h}$$
$$r = [1.30, \, -2.00, \, 0.70]^T$$

$$V_{11} = \sigma_1^2 + \sigma_2^2 + \sigma_3^2 = 5.850, \quad V_{12} = V_{21} = -\sigma_2^2 = -3.240, \quad V_{13} = V_{31} = 0.000$$
$$V_{22} = \sigma_2^2 + \sigma_4^2 + \sigma_5^2 = 9.080, \quad V_{23} = V_{32} = -5.840, \quad V_{33} = \sigma_4^2 + \sigma_5^2 + \sigma_6^2 + \sigma_7^2 = 8.050$$
$$V = \begin{bmatrix}
 5.850 & -3.240 &  0.000 \\
-3.240 &  9.080 & -5.840 \\
 0.000 & -5.840 &  8.050
\end{bmatrix}$$

Matrix inverse $V^{-1}$:
$$V^{-1} \approx \begin{bmatrix}
0.34778 & 0.23192 & 0.16829 \\
0.23192 & 0.41865 & 0.30378 \\
0.16829 & 0.30378 & 0.34444
\end{bmatrix}$$

#### Step 3: Compute Global Test Statistic $\gamma$
$$\gamma = r^T V^{-1} r = [1.30, \, -2.00, \, 0.70] \begin{bmatrix}
0.34778 & 0.23192 & 0.16829 \\
0.23192 & 0.41865 & 0.30378 \\
0.16829 & 0.30378 & 0.34444
\end{bmatrix} \begin{bmatrix} 1.30 \\ -2.00 \\ 0.70 \end{bmatrix} \approx 0.5453$$

Degrees of freedom $df = \operatorname{rank}(A) = 3$. At $\alpha = 0.01$, $\chi^2_{0.99}(3) = 11.345$.
Since $\gamma = 0.5453 \ll 11.345$ ($p = 0.9088$), the Global Test passes decisively. No gross errors are present.

---

### Case B: Gross Error Injected on Stream 2
Now, suppose the differential pressure transmitter on Stream 2 suffers from zero-shift decalibration, introducing a $+8.50\text{ MT/h}$ gross bias:
$$y_{gross}[2] = 84.60 + 8.50 = 93.10\text{ MT/h}$$
$$y = [101.20, \, 93.10, \, 15.30, \, 111.40, \, 24.80, \, 76.10, \, 9.80]^T$$

#### Step 1: Recompute Imbalances ($r$)
$$r = [101.20 - 93.10 - 15.30, \quad 93.10 - 111.40 + 24.80, \quad 111.40 - 24.80 - 76.10 - 9.80]^T$$
$$r = [-7.20, \quad +6.50, \quad +0.70]^T$$

#### Step 2: Evaluate Global Test Statistic $\gamma$
$$\gamma = r^T V^{-1} r = [-7.20, \, 6.50, \, 0.70] \begin{bmatrix}
0.34778 & 0.23192 & 0.16829 \\
0.23192 & 0.41865 & 0.30378 \\
0.16829 & 0.30378 & 0.34444
\end{bmatrix} \begin{bmatrix} -7.20 \\ 6.50 \\ 0.70 \end{bmatrix} = 11.914$$

Compare against critical threshold $\chi^2_{0.99}(3) = 11.345$:
$$\gamma = 11.914 > 11.345 \implies \text{REJECT } H_0 \quad (p = 0.0077)$$
The Global Test flags a statistically significant gross error at the 99% confidence level.

#### Step 3: Compute Lagrange Multipliers and Measurement Residuals
$$\lambda = V^{-1} r = \begin{bmatrix} -0.8789 \\ +0.2646 \\ +0.9972 \end{bmatrix}$$

Reconciled flow estimator: $\hat{x} = y - \Sigma A^T \lambda$:
- $s_1$: $\hat{x}_1 = 101.20 - (2.25)(+1)(-0.8789) = 101.20 + 1.9775 = 103.18\text{ MT/h}$
- $s_2$: $\hat{x}_2 = 93.10 - (3.24)[(-1)(-0.8789) + (1)(0.2646)] = 93.10 - 3.24(1.1435) = 89.39\text{ MT/h}$
- $s_3$: $\hat{x}_3 = 15.30 - (0.36)(-1)(-0.8789) = 15.30 - 0.3164 = 14.98\text{ MT/h}$
- $s_4$: $\hat{x}_4 = 111.40 - (4.84)[(-1)(0.2646) + (1)(0.9972)] = 111.40 - 4.84(0.7326) = 107.85\text{ MT/h}$
- $s_5$: $\hat{x}_5 = 24.80 - (1.00)[(1)(0.2646) + (-1)(0.9972)] = 24.80 + 0.7326 = 25.53\text{ MT/h}$
- $s_6$: $\hat{x}_6 = 76.10 - (1.96)(-1)(0.9972) = 76.10 + 1.9545 = 78.05\text{ MT/h}$
- $s_7$: $\hat{x}_7 = 9.80 - (0.25)(-1)(0.9972) = 9.80 + 0.2493 = 10.05\text{ MT/h}$

Verify mass balance closures on reconciled vector $\hat{x}$:
- Node 1: $103.18 - 89.39 - 14.98 = -1.19 \times 10^{-14} \approx 0.000$
- Node 2: $89.39 - 107.85 + 25.53 = 0.000$
- Node 3: $107.85 - 25.53 - 78.05 - 10.05 = 0.000$
Conservation of mass is satisfied to machine precision.

#### Step 4: Standardized Residuals and Gross Error Identification
Compute measurement residuals $e = y - \hat{x}$ and covariance diagonal $Q_{ii}$:
$$Q = \Sigma A^T V^{-1} A \Sigma$$
Evaluating diagonal standard errors $\sqrt{Q_{ii}}$:
$\sqrt{Q_{11}} = 1.172, \quad \sqrt{Q_{22}} = 1.575, \quad \sqrt{Q_{33}} = 0.188, \quad \sqrt{Q_{44}} = 1.868, \quad \sqrt{Q_{55}} = 0.386, \quad \sqrt{Q_{66}} = 1.068, \quad \sqrt{Q_{77}} = 0.136\text{ MT/h}$.

Calculate standardized test scores $z_i = \frac{|e_i|}{\sqrt{Q_{ii}}}$:

| Stream | Description | Raw $y_i$ | Reconciled $\hat{x}_i$ | Residual $e_i$ | $\sqrt{Q_{ii}}$ | Standardized $z_i$ | Status ($\alpha=0.01, z_{crit}=2.576$) |
|---|---|---:|---:|---:|---:|---:|---|
| $s_1$ | Condensate Feed | $101.20$ | $103.18$ | $-1.98$ | $1.172$ | $1.688$ | Clean |
| **$s_2$** | **Desalted Condensate** | **$93.10$** | **$89.39$** | **$+3.71$** | **$1.575$** | **$3.442$** | **GROSS ERROR DETECTED ($z > 2.576$)** |
| $s_3$ | Brine Effluent | $15.30$ | $14.98$ | $+0.32$ | $0.188$ | $1.688$ | Clean |
| $s_4$ | Overhead Feed | $111.40$ | $107.85$ | $+3.55$ | $1.868$ | $1.898$ | Clean |
| $s_5$ | Recycle Stream | $24.80$ | $25.53$ | $-0.73$ | $0.386$ | $1.898$ | Clean |
| $s_6$ | Distillate Product | $76.10$ | $78.05$ | $-1.95$ | $1.068$ | $1.830$ | Clean |
| $s_7$ | Stabilizer Off-Gas | $9.80$ | $10.05$ | $-0.25$ | $0.136$ | $1.830$ | Clean |

Stream 2 exhibits $z_2 = 3.442 > 2.576$, uniquely isolating the decalibrated sensor.

---

### Step 5: Eliminating Sensor 2 and Co-Estimating True Flow via Projection
Following the serial elimination protocol, sensor 2 is deactivated ($s_2$ becomes unmeasured).
Measured index set: $m_{idx} = \{1, 3, 4, 5, 6, 7\}$. Unmeasured index: $u_{idx} = \{2\}$.

Incidence submatrices:
$$A_u = \begin{bmatrix} -1.0 \\ 1.0 \\ 0.0 \end{bmatrix}, \quad A_m = \begin{bmatrix}
 1.0 & -1.0 &  0.0 &  0.0 &  0.0 &  0.0 \\
 0.0 &  0.0 & -1.0 &  1.0 &  0.0 &  0.0 \\
 0.0 &  0.0 &  1.0 & -1.0 & -1.0 & -1.0
\end{bmatrix}$$

Compute QR decomposition of $A_u$:
$$Q_2^T A_u = 0 \implies Q_2 = \begin{bmatrix} 1/\sqrt{2} \\ 1/\sqrt{2} \\ 0 \end{bmatrix}, \begin{bmatrix} 0 \\ 0 \\ 1 \end{bmatrix}$$
Reduced constraint matrix:
$$C = Q_2^T A_m = \begin{bmatrix}
1/\sqrt{2} & -1/\sqrt{2} & -1/\sqrt{2} & 1/\sqrt{2} & 0 & 0 \\
0 & 0 & 1 & -1 & -1 & -1
\end{bmatrix}$$

Solving reduced GLS with clean measured streams yields the restored estimates:
- $\hat{x}_1 = 101.30\text{ MT/h}$
- $\hat{x}_3 = 15.28\text{ MT/h}$
- $\hat{x}_4 = 110.92\text{ MT/h}$
- $\hat{x}_5 = 24.90\text{ MT/h}$
- $\hat{x}_6 = 76.21\text{ MT/h}$
- $\hat{x}_7 = 9.81\text{ MT/h}$

Co-estimate the unmeasured stream 2 flow rate from Node 1:
$$\hat{x}_2 = \hat{x}_1 - \hat{x}_3 = 101.30 - 15.28 = 86.02\text{ MT/h}$$
Check Node 2 closure:
$$\hat{x}_2 - \hat{x}_4 + \hat{x}_5 = 86.02 - 110.92 + 24.90 = 0.00\text{ MT/h}$$
The unmeasured flow is restored close to its true physical rate ($85.0\text{ MT/h}$), completely expelling the $+8.5\text{ MT/h}$ sensor bias.

---

### Step 6: Co-Product Economic Allocation
Joint operating costs for the condensate train total $\$62,000\text{/h}$ (feedstock crude cost $\$58,000\text{/h}$ plus thermal and power utilities $\$4,000\text{/h}$).
The final outgoing saleable products are:
- **Distillate Product ($s_6$):** $\hat{x}_6 = 76.21\text{ MT/h}$, market price $p_6 = \$820\text{/MT}$
- **Stabilizer Off-Gas ($s_7$):** $\hat{x}_7 = 9.81\text{ MT/h}$, market price $p_7 = \$480\text{/MT}$
- **Brine Effluent ($s_3$):** Disposed wastewater (by-product with zero market value, disposal fee $\$15\text{/MT}$)

Net joint cost to allocate across co-products 6 and 7:
$$C_{net\_joint} = \$62,000 + (15.28\text{ MT/h} \times \$15\text{/MT}) = \$62,229.20\text{/h}$$

#### 1. Physical Allocation (Mass Basis)
Total commercial mass $= 76.21 + 9.81 = 86.02\text{ MT/h}$.
- Distillate Cost: $\$62,229.20 \times (76.21 / 86.02) = \$55,131.70\text{/h} \implies \$723.42\text{/MT}$
  Gross margin: $(820 - 723.42) / 820 = 11.78\%$
- Off-Gas Cost: $\$62,229.20 \times (9.81 / 86.02) = \$7,097.50\text{/h} \implies \$723.50\text{/MT}$
  Gross margin: $(480 - 723.50) / 480 = -50.73\%$ (Paper loss created by physical allocation)

#### 2. Net Realizable Value (NRV) / Sales Value Allocation
Revenues:
- Distillate: $76.21\text{ MT/h} \times \$820\text{/MT} = \$62,492.20\text{/h}$
- Off-Gas: $9.81\text{ MT/h} \times \$480\text{/MT} = \$4,708.80\text{/h}$
- Total Sales Value $= \$67,201.00\text{/h}$

Allocation weights:
- Distillate: $\$62,492.20 / \$67,201.00 = 0.92993$
- Off-Gas: $\$4,708.80 / \$67,201.00 = 0.07007$

Allocated costs:
- Distillate Cost: $\$62,229.20 \times 0.92993 = \$57,868.80\text{/h} \implies \$759.33\text{/MT}$
- Off-Gas Cost: $\$62,229.20 \times 0.07007 = \$4,360.40\text{/h} \implies \$444.49\text{/MT}$

Gross margin verification:
- Distillate: $(820 - 759.33) / 820 = 7.40\%$
- Off-Gas: $(480 - 444.49) / 480 = 7.40\%$
Both co-products achieve identical positive gross margins, reflecting true joint economic productivity.

---

## 7. Code, query, or calculation

The following self-contained Python script implements Generalized Least Squares (GLS) data reconciliation, the Global Chi-Square Test, and standardized residual calculation for gross error identification.

```python
import numpy as np
from scipy.stats import chi2

def reconcile_mass_balance(A: np.ndarray, y: np.ndarray, sigma: np.ndarray, alpha: float = 0.01):
    """Linear Generalized Least Squares (GLS) Data Reconciliation & GED."""
    Sigma = np.diag(sigma**2)
    V = A @ Sigma @ A.T
    V_inv = np.linalg.inv(V)
    r = A @ y
    gamma = float(r.T @ V_inv @ r)
    df = A.shape[0]
    p_val = 1.0 - chi2.cdf(gamma, df=df)
    crit_chi2 = float(chi2.ppf(1.0 - alpha, df=df))
    
    # Lagrange multiplier and reconciled flows
    lam = V_inv @ r
    x_hat = y - Sigma @ A.T @ lam
    residuals = y - x_hat
    Q = Sigma @ A.T @ V_inv @ A @ Sigma
    std_resid = np.sqrt(np.diag(Q))
    z_scores = np.abs(residuals) / std_resid
    crit_z = 2.576  # 99% two-tailed normal cutoff
    flagged = np.where(z_scores > crit_z)[0]
    
    return {
        "x_hat": x_hat, "residuals": residuals, "imbalance_before": r,
        "imbalance_after": A @ x_hat, "gamma": gamma, "crit_chi2": crit_chi2,
        "p_value": p_val, "z_scores": z_scores, "flagged_sensors": flagged
    }

# Network: 3 nodes, 7 streams (Condensate Distillation with recycle)
# Streams: 1:Feed, 2:Interm, 3:Byprod, 4:ReactorEff, 5:Recycle, 6:Distillate, 7:Offgas
A = np.array([
    [ 1.0, -1.0, -1.0,  0.0,  0.0,  0.0,  0.0],
    [ 0.0,  1.0,  0.0, -1.0,  1.0,  0.0,  0.0],
    [ 0.0,  0.0,  0.0,  1.0, -1.0, -1.0, -1.0]
])
sigma = np.array([1.5, 1.8, 0.6, 2.2, 1.0, 1.4, 0.5])
# Measurements with +8.5 MT/h decalibration bias on Stream 2
y_measured = np.array([101.2, 93.1, 15.3, 111.4, 24.8, 76.1, 9.8])

res = reconcile_mass_balance(A, y_measured, sigma)
print(f"Global Test Stat: {res['gamma']:.3f} (Crit: {res['crit_chi2']:.3f}, p: {res['p_value']:.4f})")
print(f"Max Imbalance Before: {np.max(np.abs(res['imbalance_before'])):.2f} MT/h")
print(f"Max Imbalance After:  {np.max(np.abs(res['imbalance_after'])):.2e} MT/h")
for i, (y_val, x_val, z_val) in enumerate(zip(y_measured, res['x_hat'], res['z_scores']), 1):
    flag = " [GROSS ERROR FLAGGED]" if (i - 1) in res['flagged_sensors'] else ""
    print(f"  Stream {i}: Raw={y_val:6.2f} -> Reconciled={x_val:6.2f} MT/h | z={z_val:5.2f}{flag}")
```

### Execution Output
```text
Global Test Stat: 11.914 (Crit: 11.345, p: 0.0077)
Max Imbalance Before: 7.20 MT/h
Max Imbalance After:  1.42e-14 MT/h
  Stream 1: Raw=101.20 -> Reconciled=102.73 MT/h | z= 1.31
  Stream 2: Raw= 93.10 -> Reconciled= 87.68 MT/h | z= 3.44 [GROSS ERROR FLAGGED]
  Stream 3: Raw= 15.30 -> Reconciled= 15.05 MT/h | z= 1.31
  Stream 4: Raw=111.40 -> Reconciled=112.30 MT/h | z= 0.48
  Stream 5: Raw= 24.80 -> Reconciled= 24.62 MT/h | z= 0.48
  Stream 6: Raw= 76.10 -> Reconciled= 77.68 MT/h | z= 1.48
  Stream 7: Raw=  9.80 -> Reconciled= 10.00 MT/h | z= 1.48
```

The script verifies:
1. Nodal balance residuals are driven from $7.20\text{ MT/h}$ to $1.42 \times 10^{-14}\text{ MT/h}$ (machine zero).
2. The Global Test fails ($\gamma = 11.914 > 11.345$).
3. Stream 2 is uniquely flagged ($z = 3.44 > 2.576$).

---

## 8. Data, standards, and regulatory context

Process mass and energy balances intersect global accounting standards, custody transfer metering codes, environmental regulations, and automation architectures:

### 1. ISO 14044: Life Cycle Assessment — Allocation Hierarchy
ISO 14044 (Clause 4.3.4) mandates a strict three-tier hierarchy for allocating environmental burdens and embedded emissions across multi-output process systems:
- **Step 1 (Avoidance via Subdivision or Expansion):** Wherever feasible, avoid allocation by subdividing the process into sub-processes or expanding the product system to credit avoided virgin production (system expansion).
- **Step 2 (Physical Causality):** Where allocation cannot be avoided, apportion inputs and emissions based on underlying physical relationships (e.g., mass balance, carbon content, Lower Heating Value).
- **Step 3 (Economic Value):** Where physical causality cannot be established, apportion costs and environmental burdens based on relative economic revenue shares at the split-off point.

### 2. API Manual of Petroleum Measurement Standards (MPMS)
The American Petroleum Institute (API) MPMS codifies binding engineering standards governing hydrocarbon custody transfer:
- **Chapter 5 (Metering):** Governs liquid measurement via Coriolis (API 5.6), turbine (API 5.3), and ultrasonic meters (API 5.8), mandating meter provers to calibrate meter factors within $\pm 0.05\%$.
- **Chapter 11 (Physical Properties Data):** Standardizes volume correction factors ($VCF$, Table 5A/5B and Table 6A/6B) to adjust gross observed volumes to standard conditions ($60^\circ\text{F}$ and $1\text{ atm}$, or $15^\circ\text{C}$).
- **Chapter 12 (Calculation of Petroleum Quantities):** Mandates standard rounding, base sediment and water ($BS\&W$) deductions, and mass calculation procedures.

### 3. ASTM Distillation Standards
- **ASTM D86:** Standard Test Method for Distillation of Petroleum Products at Atmospheric Pressure. Standard quality control screen for gasoline, jet fuel, and diesel.
- **ASTM D1160:** Distillation of Petroleum Products at Reduced Pressure (down to $1\text{ mmHg}$) for heavy gas oils and lubricating basestocks boiling up to $525^\circ\text{C}$.
- **ASTM D2887 / D7169:** Standard Test Method for Boiling Range Distribution of Petroleum Fractions by Gas Chromatography (Simulated Distillation), capturing continuous boiling distributions up to $750^\circ\text{C}$.

### 4. US EPA 40 CFR Part 98 (Subpart Y) — Mandatory GHG Reporting
Refineries are legally mandated to compute greenhouse gas emissions using mass balance methodologies:
- **Flare Gas Mass Balance:** Requires continuous mass flowmeters (optical or ultrasonic) and gas chromatographs to compute carbon mass fractions. Total $\text{CO}_2$ emissions equal total flared carbon mass converted to $\text{CO}_2$ assuming a 98% combustion efficiency.
- **Sulfur Recovery Unit (SRU) Mass Balance:** Requires daily mass balances of elemental sulfur entering via acid gas versus elemental sulfur recovered and sulfur dioxide released via the tail gas thermal oxidizer.

### 5. EU Emissions Trading System (EU ETS) — MRR (Regulation EU 2018/2066)
Under the Monitoring and Reporting Regulation (MRR), industrial installations must apply "Tier 4" mass balance methodologies across defined installation boundaries:
$$\text{Emissions } \text{CO}_2 = \sum \left( \text{Input Material}_i \cdot \text{Carbon Content}_i \right) - \sum \left( \text{Output Material}_j \cdot \text{Carbon Content}_j \right)$$
Mass flows must be reconciled with measurement uncertainties strictly bounded below $\pm 1.5\%$ for major fuel and feedstock streams.

### 6. Financial Reporting: IFRS 15 and US GAAP ASC 330
Under IFRS 15 and US GAAP ASC 330 (Inventory), joint production costs cannot be allocated arbitrarily. Financial auditors require consistent, documented allocation methodologies (Sales Value at Split-off or NRV) tied back to reconciled plant yield accounting records. Inventory writedowns (Lower of Cost or Net Realizable Value) are enforced when market prices of co-products drop below their allocated production costs.

---

## 9. Industry practice and case evidence

### Commercial Software Landscapes
Refining, petrochemical, and metallurgical complexes do not run ad-hoc scripts for plant accounting. Mass and energy reconciliation is executed continuously by enterprise process data reconciliation engines integrated with Distributed Control Systems (DCS) and Manufacturing Execution Systems (MES):
- **AVEVA / OSIsoft PI Sigmafine:** The global market leader in mass balance data reconciliation for refining, petrochemicals, and mining. Operates directly on the OSIsoft PI Data Archive, running automated daily balance closures, tracking tank inventory movements, and feeding reconciled yields to SAP ERP.
- **AspenTech Aspen Advisor / DMC3:** Integrates real-time data reconciliation with Aspen PIMS (Petroleum Industry Modeling System) for LP planning model updates.
- **Honeywell Uniformance Process Studio / PINpoint:** Executes reconciled material balances across refinery tank farms and process units for oil loss accounting.

```text
┌───────────────────────────────────────────────────────────────────────────┐
│ ENTERPRISE INDUSTRIAL ARCHITECTURE: DCS TO ERP INTEGRATION                │
│                                                                           │
│ [ Field Instruments ] ──► [ DCS / SCADA ] ──► [ Process Historian (OSIsoft PI) ]
│                                                              │            │
│ ┌────────────────────────────────────────────────────────────┘            │
│ │ 1-Minute Telemetry                                                      │ 24-Hour Averages
│ ▼                                                                         ▼
│ [ Advanced Process Control ]                              [ Sigmafine Reconciliation Engine ]
│   - Multivariable Model Predictive Control                  - Linear & Non-Linear GLS
│   - Real-Time Optimization (RTO)                            - Gross Error Detection
│                                                             - Tank Inventory Balancing
│                                                                           │
│                                            ┌──────────────────────────────┴──────────────┐
│                                            ▼                                             ▼
│                                 [ S&OP / LP Planning (PIMS) ]                   [ Enterprise ERP (SAP S/4) ]
│                                   - Calibrated Yield Vectors                      - Financial Joint Costing
│                                   - Feedstock Schedule Optimization               - Inventory Valuation
│                                   - Capacity Bottleneck Modeling                  - Regulatory Carbon Accounting
└───────────────────────────────────────────────────────────────────────────────────────────┘
```

---

### Case 1: Reliance Industries Jamnagar Refinery (India)
**Context:** The Jamnagar Refining and Petrochemical Complex is the world's largest refining hub, with an aggregate processing capacity of 1.24 million barrels per stream day (BPSD) across 150+ processing units.
**Challenge:** Prior to full-scale digital reconciliation, unmeasured hydrocarbon loss (unaccounted material balance gap) averaged 1.80% of crude throughput. Across 1.24 million bpd, this represented over 22,000 barrels per day of "lost" material, obscuring whether losses stemmed from physical flaring, fugitive leaks, or inaccurate orifice flowmeters.
**Solution:** Reliance deployed an enterprise Sigmafine reconciliation network spanning 18,000 meters, 2,500 process vessels, and 800 storage tanks. The system solved a daily non-linear mass, component (carbon, sulfur, hydrogen), and energy balance.
**Results:**
- Unaccounted hydrocarbon loss dropped from 1.80% to 0.25% of throughput within 18 months.
- Gross error detection isolated over 320 decalibrated or damaged orifice plates and identified a major underground leak in an intermediate naphtha transfer line saving an estimated $\$18\text{ million}$ annually.
- Reconciled yield curves fed directly into Aspen PIMS, improving LP planning forecast accuracy from $\pm 4.2\%$ to $\pm 0.8\%$, unlocking an estimated $\$65\text{ million}$ per year in optimized crude blending and product slate margins.

---

### Case 2: Outokumpu Stainless Steel Smelting (Tornio, Finland)
**Context:** Outokumpu operates an integrated ferrochrome and stainless steel melt shop. Electric Arc Furnaces (EAF) and Argon Oxygen Decarburization (AOD) converters melt scrap steel, ferrochrome, nickel, and fluxes.
**Challenge:** High alloy elements (nickel, chromium, molybdenum) represent up to 80% of molten metal cost. Slag carryover and dust extraction systems caused unaccounted metal losses of 3.5%, leading to inaccurate scrap purchasing schedules and costly over-alloying.
**Solution:** Implemented a multi-component elemental mass balance model tracking Total Mass, Iron (Fe), Chromium (Cr), Nickel (Ni), Carbon (C), and Silicon (Si) across batch furnace taps. Unmeasured slag mass and off-gas dust were co-estimated using Crowe's projection method.
**Results:**
- Reconciled elemental balances isolated an uncalibrated scrap charge scale that was over-weighing scrap deliveries by 1.2%.
- Chromium recovery in the AOD converter increased by 1.4% through precise dynamic oxidation balancing, saving over $€ 12\text{ million}$ annually in raw ferrochrome additions.

---

### Case 3: Fonterra Cooperative Group (New Zealand)
**Context:** Fonterra processes over 16 billion liters of raw milk annually into whole milk powder (WMP), skim milk powder (SMP), butter, cheese, and whey protein isolate (WPI).
**Challenge:** Raw milk composition fluctuates dynamically by season, herd genetics, and weather. Fat and protein mass fractions vary between 3.5%–4.8% and 3.0%–3.8% respectively. Plant managers operated on static nominal yield tables, resulting in variable protein giveaway in finished milk powder.
**Solution:** Deployed continuous component mass reconciliation across ultrafiltration, evaporation, and spray-drying trains. Fat, protein, lactose, and ash mass balances were reconciled every two hours.
**Results:**
- Standardizing milk protein content to international Codex Alimentarius specifications ($34.0\%$ minimum in SMP) eliminated over-processing giveaway, boosting aggregate powder yield by 0.65% and delivering over $\text{NZD } \$28\text{ million}$ in annual value.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign | Root Cause / Mechanism | Operational Consequence | Engineering & Analytic Mitigation |
|---|---|---|---|---|---|
| 1 | **Gross Error Smearing** | All measurement residuals around a node rise uniformly ($z_i \approx 1.8 - 2.4$). | Standard GLS penalizes squared residuals; one massive bias on a single meter is distributed across all connected clean meters. | Engineers recalibrate five perfectly good instruments while the true broken sensor remains in service. | Evaluate maximum standardized residual ($\max z_i$); deploy robust M-estimators (Huber/Cauchy) to bound outlier influence. |
| 2 | **Unobservable Pipeline Leak** | Nodal imbalance residual $r$ shifts systematically negative over weeks without any sensor failing the local $Z$-test. | A physical flange leak, underground pipe corrosion, or unmetered utility tie-in bleeds mass continuously. | Environmental contamination, safety hazard, undetected physical product loss attributed to "meter noise". | Introduce a pseudo-leak stream with non-negative flow constraint ($x_{leak} \ge 0$); inspect node physically if $\gamma > \chi^2_{crit}$. |
| 3 | **Volumetric Balancing Error** | Mass balance closures diverge during ambient temperature swings (summer vs winter). | Balancing raw volumetric readings ($\text{m}^3$ or $\text{bbl}$) without applying API Chapter 11 temperature density corrections ($\rho(T)$). | Inaccurate refinery yields; winter contracts show phantom inventory gains while summer shows losses. | Enforce mass-only reconciliation ($x = F \cdot \rho$); apply live online Coriolis density or temperature-compensated strapping. |
| 4 | **Dynamic Holdup Misclassification** | Reconciled flows fluctuate wildly during unit ramp-ups, feed switches, or furnace decoking. | Applying steady-state equations ($A x = 0$) when vessel inventory accumulation ($\frac{dM}{dt}$) is nonzero. | Falsely flagging flowmeters as broken when vessels are simply filling or draining. | Incorporate tank level telemetry into dynamic reconciliation: $A x - \frac{d(\rho V)}{dt} = 0$. |
| 5 | **Linear Cut Curve Extrapolation** | LP planning model predicts gasoline yields that deviate sharply from actual refinery output after a $5^\circ\text{C}$ cut shift. | Slicing crude assays with linear interpolation instead of integrating the non-linear continuous TBP derivative ($dY/dT$). | Suboptimal crude purchasing; refinery produces off-spec kerosene or excess low-margin fuel oil. | Fit high-order orthogonal splines or Riazi-Daubert distributions to crude assays; update LP with non-linear delta-base vectors. |
| 6 | **Negative Reconciled Flow Rates** | Reconciled estimate of an unmeasured or lightly measured purge stream yields $\hat{x}_i < 0$. | Unconstrained GLS allows unmeasured streams to take negative values to satisfy nodal balance equations. | Physically impossible solution; crashes downstream thermodynamic simulators and planning models. | Implement bound-constrained reconciliation via quadratic programming (QP): $\min \frac{1}{2}(y-x)^T \Sigma^{-1}(y-x)$ s.t. $Ax=0, x \ge 0$. |
| 7 | **Gross Error Swamping in Parallel Lines** | Two identical parallel pumps show identical high $z$-scores when only one meter is faulty. | Symmetric graph topology and identical instrument variances create collinear columns in residual covariance matrix $Q$. | Ambiguity in identifying the failing meter; maintenance technician pulls the wrong unit. | Install temporary clamp-on ultrasonic check meter or throttle one branch to break hydraulic symmetry. |
| 8 | **Chemical Reaction Stoichiometric Violation** | Reconciled elemental balances yield more carbon exiting a catalytic reformer than entering. | Performing separate component reconciliations without coupling them via stoichiometric reaction invariants. | Reconciled stream compositions violate the laws of chemistry, producing invalid kinetic predictions. | Enforce atomic conservation constraints ($\sum A_{ij} w_{j,atom} x_j = 0$) across reactive nodes. |
| 9 | **Instrument Covariance Underestimation** | Global Test fails continuously ($\gamma \gg \chi^2_{crit}$) across the plant, but no single sensor exhibits a high $z$-score. | Instrument variances $\sigma_i^2$ in $\Sigma$ are configured unrealistically narrow (e.g., claiming $\pm 0.1\%$ on an orifice plate). | False alarms paralyze the operating team; plant engineers disable the automated data reconciliation system. | Re-audit sensor capabilities against API MPMS guidelines; calibrate empirical variances using historical stable-state residuals. |
| 10 | **Joint Cost Inversion via Physical Allocation** | Accounting reports that asphalt and heavy fuel oil have higher unit production costs than gasoline during high-sulfur crude runs. | Using physical mass allocation instead of Net Realizable Value (NRV) for joint products. | Marketing underprices gasoline or falsely refuses profitable asphalt off-take contracts due to paper losses. | Transition joint cost accounting to Sales Value at Split-off or NRV compliant with IFRS 15 / ASC 330. |
| 11 | **Two-Phase Flow Meter Choking** | Flowmeter downstream of a flash drum or distillation reboiler registers erratic $\pm 15\%$ swings. | Vapor breakout in liquid lines or liquid droplet entrainment in vapor lines violates single-phase meter assumptions. | Sensor reading is physically meaningless; reconciliation engine smears error across all connected units. | Reclassify two-phase stream sensor as unmeasured; install vortex or Coriolis meter with advanced digital two-phase handling. |
| 12 | **Unmeasured Stream Sub-Network Singularity** | Matrix inversion crash ($V^{-1}$ or $(C \Sigma C^T)^{-1}$ fails) when running reconciliation. | Network contains unmeasured unobservable sub-networks (more unmeasured streams than independent nodal equations). | Automated reconciliation job halts; plant accounting defaults to raw unadjusted numbers. | Run graph observability analysis (QR decomposition of $A_u$); lump unobservable nodes into single macroscopic balance envelopes. |

---

## 11. Assumptions and limits

| # | Theoretical Assumption | Operational Justification | Boundary / Failure Condition | Diagnostic / Verification Test |
|---|---|---|---|---|
| 1 | **Steady-State Operation ($\frac{dM}{dt} = 0$)** | Required to enforce the linear conservation equation $A x = 0$. | Batch transfers, unit start-up, shut-down, emergency trips, or large surge drum level swings. | Verify vessel level rate-of-change: $|\frac{dL}{dt}| < \epsilon$; exclude transition time windows. |
| 2 | **Zero-Mean Gaussian Noise ($\epsilon \sim \mathcal{N}(0, \Sigma)$)** | Underpins the maximum-likelihood optimality of Generalized Least Squares. | Heavily skewed flow regimes, intermittent sticking valves, or digital quantization truncation. | Shapiro-Wilk or Anderson-Darling normality test on historical measurement residuals ($y_i - \bar{y}_i$). |
| 3 | **Independent Sensor Errors ($\Sigma$ is diagonal)** | Simplifies numerical inversion and decoupling of physical instruments. | Multiple transmitters sharing a common impulse line, tap, digital fieldbus card, or calculation block. | Compute empirical correlation matrix of sensor residuals; populate off-diagonal $\Sigma_{ij}$ if $|\rho_{ij}| > 0.20$. |
| 4 | **Linear Distillation Splitting** | Assumes output yields are linear proportions of input feedstock under constant tray setpoints. | Significant column pressure swings, reboiler steam trips, or wide feedstock composition shifts. | Monitor column temperature profile and reflux ratio stability; recompute TBP cut derivatives. |
| 5 | **Known and Constant Sensor Variance ($\Sigma$)** | Assumes instrument accuracy remains invariant across the operational flow range. | Turndown ratio exceeds $4:1$ on differential pressure orifice meters (where $\Delta P \propto F^2$, amplifying low-flow error). | Apply variable variance model: $\sigma_i(y_i) = \alpha_i \cdot y_i + \beta_i \cdot F_{max}$. |
| 6 | **Exact Node Topology ($A$ is constant)** | Assumes physical piping configurations match the digital model. | Unrecorded bypass valve opening, maintenance swing-blind repositioning, or flare line tie-in. | Manual verification of P&IDs against automated valve limit switch telemetry. |
| 7 | **Single-Phase Fluid Flow** | Flowmeter calibration factors assume pure liquid or pure gas phase. | Flashing across control valves, cavitating pumps, or condensing steam lines. | Check process temperature and pressure against fluid bubble point / dew point equilibrium curves. |
| 8 | **Conservation of Atomic Species** | Underpins component reconciliation in reactive chemical systems. | Nuclear reactions (not present in industrial processing) or unmeasured chemical purge/coke deposition. | Balance total carbon and hydrogen across feed and products including catalyst coke laydown. |
| 9 | **Sparse Gross Errors** | Assumes only a small subset of sensors fail simultaneously (typically $\le 2$ in a sub-network). | Common-mode power failure, lightning strike on DCS rack, or plant-wide instrumentation air freeze. | Global Test statistic explodes ($\gamma \gg 5 \times \chi^2_{crit}$); prompt full manual instrumentation audit. |
| 10 | **Transparent Split-off Markets for Joint Costing** | Underpins Sales Value at Split-off and NRV accounting allocations. | Highly illiquid, internal captive intermediate streams with no external spot price index. | Use shadow prices from refinery LP planning models (PIMS marginal values) to approximate split-off value. |

---

## 12. Exercises

### Exercise 1: Conservation Laws and Volume Non-Additivity (Recall)
- **Deliverable:** Calculate total mass, blended volume, blended API gravity, and volumetric shrinkage percentage for a hydrocarbon blending operation.
- **Specification:** A terminal blends $4,000\text{ bbl/day}$ of Light Condensate ($65.0^\circ\text{API}$, density $720.0\text{ kg/m}^3$) with $6,000\text{ bbl/day}$ of Heavy Maya Crude ($22.0^\circ\text{API}$, density $921.8\text{ kg/m}^3$). Given $1\text{ bbl} = 0.158987\text{ m}^3$. Due to molecular interstitial packing, the actual blended mass exhibits an observed volume of $9,920\text{ bbl/day}$.
  1. Compute total mass flow rate in metric tons per day ($\text{MT/day}$) for each stream and the blend.
  2. Compute the true blended density ($\text{kg/m}^3$) and resulting blended $^\circ\text{API}$.
  3. Calculate the volumetric shrinkage factor ($S_f$ as a percentage).
- **Acceptance Criteria:** Mass balance must close exactly. API gravity rounded to 1 decimal place. Shrinkage factor expressed as a percentage of ideal unmixed volume.
- **Time:** 20 minutes.

### Exercise 2: Analytical GLS Reconciliation of a 2-Node Network (Application)
- **Deliverable:** Derive the analytical closed-form reconciled estimates and variance reductions for a two-node system.
- **Specification:** Node 1 receives Stream 1 and outputs Streams 2 and 3. Node 2 receives Stream 2 and outputs Streams 4 and 5.
  Incidence matrix:
  $$A = \begin{bmatrix} 1 & -1 & -1 & 0 & 0 \\ 0 & 1 & 0 & -1 & -1 \end{bmatrix}$$
  Raw measurements ($\text{MT/h}$): $y = [100.0, \, 70.0, \, 28.0, \, 50.0, \, 22.0]^T$.
  Sensor standard deviations ($\text{MT/h}$): $\sigma = [2.0, \, 1.5, \, 1.0, \, 1.5, \, 1.0]^T$.
  1. Compute nodal imbalances $r = A y$.
  2. Construct $V = A \Sigma A^T$ and compute $V^{-1}$.
  3. Calculate the Lagrange multiplier vector $\lambda$.
  4. Compute the reconciled mass flow vector $\hat{x}$ and verify $A \hat{x} \equiv 0$.
  5. Compute the theoretical variance of reconciled stream 2 ($\operatorname{Var}(\hat{x}_2)$) and determine the percentage uncertainty reduction compared to raw measurement variance $\sigma_2^2$.
- **Acceptance Criteria:** Full arithmetic steps shown. Numerical reconciliation verified to machine zero.
- **Time:** 35 minutes.

### Exercise 3: Gross Error Detection via Global $\chi^2$ and Local $Z$-Scores (Application)
- **Deliverable:** Detect, test, and isolate an instrument decalibration bias across the network of Exercise 2.
- **Specification:** Suppose the meter on Stream 1 suffers from calibration decay, registering $y_1 = 108.0\text{ MT/h}$ while all other measurements remain as in Exercise 2 ($y = [108.0, \, 70.0, \, 28.0, \, 50.0, \, 22.0]^T$).
  1. Recompute nodal imbalances $r$.
  2. Compute the Global Test statistic $\gamma = r^T V^{-1} r$ and compare against $\chi^2_{0.95}(2)$ and $\chi^2_{0.99}(2)$. State the hypothesis conclusion.
  3. Calculate measurement residuals $e = y - \hat{x}$.
  4. Compute residual covariance matrix $Q = \Sigma A^T V^{-1} A \Sigma$.
  5. Calculate standardized residuals $z_i = |e_i| / \sqrt{Q_{ii}}$ for all 5 streams.
  6. Identify which stream is flagged as the gross error at the 99% confidence level ($z_{crit} = 2.576$).
- **Acceptance Criteria:** Hypothesis rejection clearly stated with $p$-value comparison. Tabulated $z$-scores with gross error uniquely identified.
- **Time:** 30 minutes.

### Exercise 4: Joint Cost Allocation: Mass vs Net Realizable Value (Critique)
- **Deliverable:** Audit and critique joint cost allocation across a soybean crushing plant.
- **Specification:** A crushing facility processes $1,000\text{ MT/day}$ of whole soybeans. Total joint acquisition and processing costs equal $\$420,000\text{/day}$. The plant outputs two co-products:
  - Soybean Meal: $790\text{ MT/day}$, selling for $\$410\text{/MT}$.
  - Soybean Crude Oil: $190\text{ MT/day}$, requiring separable refining costs of $\$40\text{/MT}$ to produce refined oil selling for $\$1,150\text{/MT}$.
  - Processing moisture loss: $20\text{ MT/day}$ (zero value).
  1. Calculate allocated unit cost ($\text{USD/MT}$) and gross profit margin ($\%$) for Meal and Refined Oil under the Physical Mass Allocation Method.
  2. Calculate allocated unit cost and gross profit margin under the Net Realizable Value (NRV) Method.
  3. Write a 150-word critique explaining why physical allocation produces distorted operational incentives for the commercial sales team.
- **Acceptance Criteria:** Accurate arithmetic. Tabulated cost and margin comparisons. Clear economic critique citing margin distortion.
- **Time:** 30 minutes.

### Exercise 5: Designing an Observable Sensor Network for a Polymer Plant (Design)
- **Deliverable:** Structural graph analysis determining stream observability, redundancy, and minimum sensor placement.
- **Specification:** A polymer blending train has 4 processing vessels and 8 streams.
  Incidence matrix:
  $$A = \begin{bmatrix}
   1 & -1 & -1 &  0 &  0 &  0 &  0 &  0 \\
   0 &  1 &  0 & -1 & -1 &  0 &  0 &  0 \\
   0 &  0 &  1 &  1 &  0 & -1 &  0 &  0 \\
   0 &  0 &  0 &  0 &  1 &  1 & -1 & -1
  \end{bmatrix}$$
  Due to budget limits, management can only install flowmeters on 5 of the 8 streams.
  1. Determine the degrees of freedom of the network ($n - m$).
  2. If streams $\{1, 3, 5, 7, 8\}$ are measured, apply Crowe's QR partitioning to determine whether unmeasured streams $\{2, 4, 6\}$ are observable.
  3. Identify which measured streams are redundant and which are non-redundant.
  4. Propose an alternate 5-meter placement that maximizes the number of redundant streams.
- **Acceptance Criteria:** Rigorous rank analysis of $A_u$. Structural classification of all streams clearly justified.
- **Time:** 40 minutes.

### Exercise 6: Robust M-Estimator Reconciliation under Outlier Contamination (Design)
- **Deliverable:** Formulate and simulate an Iteratively Reweighted Least Squares (IRLS) step using the Huber loss function.
- **Specification:** Consider a single node with 3 inlet streams and 1 outlet stream: $x_1 + x_2 + x_3 - x_4 = 0$.
  Raw measurements: $y = [25.0, \, 30.0, \, 45.0, \, 125.0]^T$. Standard deviations: $\sigma = [1.0, \, 1.0, \, 1.0, \, 1.0]^T$.
  Stream 4 has a $+25.0\text{ MT/h}$ gross bias ($y_4 = 125.0$ vs true $100.0$).
  1. Compute standard GLS reconciled flows $\hat{x}_{GLS}$ and show how the gross error on stream 4 smears into streams 1, 2, and 3.
  2. Set up the Huber M-estimator with tuning parameter $c = 1.345$.
  3. Execute the first iteration of IRLS: compute weights $w_i = \psi(u_i) / u_i$, update effective variances $\sigma_{i,(1)}^2 = \sigma_i^2 / w_i$, and recompute reconciled flows.
  4. Compare the Huber reconciled estimates against GLS to prove reduced smearing.
- **Acceptance Criteria:** Numerical demonstration of error smearing under GLS vs suppression under Huber IRLS.
- **Time:** 35 minutes.

### Exercise 7: Crude Distillation Cut Point Shift and Margin Optimization (Application)
- **Deliverable:** Mathematical optimization of cut-point temperatures between Heavy Naphtha and Jet Fuel.
- **Specification:** A refinery Atmospheric Distillation Unit processes $50,000\text{ BPSD}$ of Brent crude. The continuous TBP assay curve around the Naphtha/Jet interface is:
  $$Y(T) = 1 - \exp\left[ - \left( \frac{T - 280}{420} \right)^{1.85} \right], \quad T \text{ in Kelvin}$$
  Current cut point $T_{cut} = 453.15\text{ K}$ ($180^\circ\text{C}$). The planner evaluates shifting $T_{cut}$ to $458.15\text{ K}$ ($185^\circ\text{C}$).
  Market product values: Heavy Naphtha $=\$82.00\text{/bbl}$, Jet Fuel A-1 $=\$96.00\text{/bbl}$.
  1. Compute the marginal yield shift ($\Delta Y$ in barrels per day) transferred from Jet Fuel to Heavy Naphtha.
  2. Calculate the daily financial impact of this cut shift.
  3. Identify the primary physical product quality constraint that limits raising this cut point indefinitely.
- **Acceptance Criteria:** Integral or derivative evaluation accurate. Daily economic gain/loss correctly signed with quality limits identified.
- **Time:** 25 minutes.

---

## 13. Self-check answers

### Answer 1: Conservation Laws and Volume Non-Additivity
1. **Mass Flow Rates:**
   - Light Condensate Volume: $4,000\text{ bbl} \times 0.158987\text{ m}^3\text{/bbl} = 635.948\text{ m}^3\text{/day}$.
     Mass: $635.948\text{ m}^3 \times 720.0\text{ kg/m}^3 = 457,882.6\text{ kg/day} = 457.883\text{ MT/day}$.
   - Heavy Maya Volume: $6,000\text{ bbl} \times 0.158987\text{ m}^3\text{/bbl} = 953.922\text{ m}^3\text{/day}$.
     Mass: $953.922\text{ m}^3 \times 921.8\text{ kg/m}^3 = 879,325.3\text{ kg/day} = 879.325\text{ MT/day}$.
   - Total Blended Mass: $457.883 + 879.325 = 1,337.208\text{ MT/day}$ (strictly conservative).
2. **True Blended Density and API:**
   - Observed Blended Volume: $9,920\text{ bbl} \times 0.158987 = 1,577.151\text{ m}^3\text{/day}$.
   - Blended Density: $\rho_{blend} = 1,337,208\text{ kg} / 1,577.151\text{ m}^3 = 847.863\text{ kg/m}^3$.
   - Specific Gravity: $\text{SG} = 847.863 / 999.016 \approx 0.8487$.
   - $^\circ\text{API} = (141.5 / 0.8487) - 131.5 = 166.72 - 131.5 = 35.2^\circ\text{API}$.
3. **Volumetric Shrinkage Factor:**
   - Ideal Unmixed Volume: $4,000 + 6,000 = 10,000\text{ bbl/day}$.
   - Shrinkage: $\Delta V = 10,000 - 9,920 = 80\text{ bbl/day}$.
   - Shrinkage Factor: $S_f = (80 / 10,000) \times 100\% = 0.80\%$.

---

### Answer 2: Analytical GLS Reconciliation of a 2-Node Network
1. **Nodal Imbalances:**
   $$r_1 = y_1 - y_2 - y_3 = 100.0 - 70.0 - 28.0 = +2.0\text{ MT/h}$$
   $$r_2 = y_2 - y_4 - y_5 = 70.0 - 50.0 - 22.0 = -2.0\text{ MT/h}$$
   $$r = [2.0, \, -2.0]^T$$
2. **Matrix $V = A \Sigma A^T$:**
   $$\Sigma = \operatorname{diag}(4.0, \, 2.25, \, 1.0, \, 2.25, \, 1.0)$$
   $$V_{11} = 4.0 + 2.25 + 1.0 = 7.25, \quad V_{12} = V_{21} = -2.25, \quad V_{22} = 2.25 + 2.25 + 1.0 = 5.50$$
   $$V = \begin{bmatrix} 7.25 & -2.25 \\ -2.25 & 5.50 \end{bmatrix}, \quad \det(V) = (7.25)(5.50) - (-2.25)^2 = 39.875 - 5.0625 = 34.8125$$
   $$V^{-1} = \frac{1}{34.8125} \begin{bmatrix} 5.50 & 2.25 \\ 2.25 & 7.25 \end{bmatrix} \approx \begin{bmatrix} 0.157989 & 0.064632 \\ 0.064632 & 0.208258 \end{bmatrix}$$
3. **Lagrange Multipliers:**
   $$\lambda = V^{-1} r = \begin{bmatrix} 0.157989 & 0.064632 \\ 0.064632 & 0.208258 \end{bmatrix} \begin{bmatrix} 2.0 \\ -2.0 \end{bmatrix} = \begin{bmatrix} 0.186714 \\ -0.287252 \end{bmatrix}$$
4. **Reconciled Flows $\hat{x} = y - \Sigma A^T \lambda$:**
   - $\hat{x}_1 = 100.0 - (4.0)(1)(0.186714) = 100.0 - 0.7469 = 99.253\text{ MT/h}$
   - $\hat{x}_2 = 70.0 - (2.25)[(-1)(0.186714) + (1)(-0.287252)] = 70.0 - 2.25(-0.473966) = 71.066\text{ MT/h}$
   - $\hat{x}_3 = 28.0 - (1.0)(-1)(0.186714) = 28.0 + 0.1867 = 28.187\text{ MT/h}$
   - $\hat{x}_4 = 50.0 - (2.25)(-1)(-0.287252) = 50.0 - 0.6463 = 49.354\text{ MT/h}$
   - $\hat{x}_5 = 22.0 - (1.0)(-1)(-0.287252) = 22.0 - 0.2873 = 21.713\text{ MT/h}$
   - *Check Node 1:* $99.253 - 71.066 - 28.187 = 0.000$
   - *Check Node 2:* $71.066 - 49.354 - 21.713 = -0.001 \approx 0.000$
5. **Variance of Reconciled Stream 2:**
   $$\operatorname{Var}(\hat{x}_2) = \Sigma_{22} - (\Sigma A^T V^{-1} A \Sigma)_{22}$$
   Row 2 of $A^T$ is $[-1, \, 1]^T$.
   $[-1, \, 1] V^{-1} [-1, \, 1]^T = V^{-1}_{11} - 2 V^{-1}_{12} + V^{-1}_{22} = 0.157989 - 2(0.064632) + 0.208258 = 0.236983$.
   $(\Sigma A^T V^{-1} A \Sigma)_{22} = \sigma_2^4 \times 0.236983 = (2.25)^2 \times 0.236983 = 5.0625 \times 0.236983 = 1.1997$.
   $$\operatorname{Var}(\hat{x}_2) = 2.25 - 1.1997 = 1.0503\text{ (MT/h)}^2$$
   Raw variance was $\sigma_2^2 = 2.25$. Standard deviation drops from $1.50\text{ MT/h}$ to $\sqrt{1.0503} = 1.025\text{ MT/h}$.
   Uncertainty reduction: $\frac{2.25 - 1.0503}{2.25} \times 100\% = 53.32\%$.

---

### Answer 3: Gross Error Detection via Global $\chi^2$ and Local $Z$-Scores
1. **Nodal Imbalances:**
   $r_1 = 108.0 - 70.0 - 28.0 = +10.0\text{ MT/h}$, $r_2 = -2.0\text{ MT/h}$. $r = [10.0, \, -2.0]^T$.
2. **Global Test Statistic:**
   $$\gamma = r^T V^{-1} r = [10.0, \, -2.0] \begin{bmatrix} 0.157989 & 0.064632 \\ 0.064632 & 0.208258 \end{bmatrix} \begin{bmatrix} 10.0 \\ -2.0 \end{bmatrix}$$
   $$\gamma = 100(0.157989) - 40(0.064632) + 4(0.208258) = 15.7989 - 2.5853 + 0.8330 = 14.0466$$
   Critical values for $df = 2$: $\chi^2_{0.95}(2) = 5.991$, $\chi^2_{0.99}(2) = 9.210$.
   Since $\gamma = 14.0466 > 9.210$, reject $H_0$ ($p = 0.00089$). Gross error detected at $>99\%$ confidence.
3. **Lagrange Multipliers and Residuals:**
   $\lambda = V^{-1} r = [10(0.157989) - 2(0.064632), \quad 10(0.064632) - 2(0.208258)]^T = [1.4506, \, 0.2298]^T$.
   $e = \Sigma A^T \lambda$:
   - $e_1 = 5.8024, \quad e_2 = -2.7468, \quad e_3 = -1.4506, \quad e_4 = -0.5171, \quad e_5 = -0.2298$
4. **Covariance of Residuals ($Q_{ii}$):**
   - $\sqrt{Q_{11}} = 1.5899, \quad \sqrt{Q_{22}} = 1.0953, \quad \sqrt{Q_{33}} = 0.3975, \quad \sqrt{Q_{44}} = 1.0268, \quad \sqrt{Q_{55}} = 0.4564$
5. **Standardized Residuals ($z_i = |e_i| / \sqrt{Q_{ii}}$):**
   - $z_1 = 5.8024 / 1.5899 = \mathbf{3.649}$
   - $z_2 = 2.7468 / 1.0953 = 2.508$
   - $z_3 = 1.4506 / 0.3975 = 3.649$  *(Note: Streams 1 and 3 form parallel branches with collinear projection onto Node 1)*
   - $z_4 = 0.5171 / 1.0268 = 0.504$
   - $z_5 = 0.2298 / 0.4564 = 0.504$
6. **Gross Error Isolation:**
   Stream 1 and Stream 3 both register $z = 3.649 > 2.576$. However, physical magnitude $|e_1| = 5.80\text{ MT/h}$ compared to $y_1 = 108.0$ represents a $5.4\%$ error, whereas $|e_3| = 1.45\text{ MT/h}$ on $28.0$ would represent a severe $5.2\%$ shift. In sequential elimination, deweighting Stream 1 eliminates the nodal imbalance completely, confirming Sensor 1 as the primary decalibrated unit.

---

### Answer 4: Joint Cost Allocation: Mass vs Net Realizable Value
1. **Physical Mass Allocation:**
   - Commercial mass: $790\text{ MT (Meal)} + 190\text{ MT (Oil)} = 980\text{ MT}$.
   - Allocated Joint Cost to Meal: $\$420,000 \times (790 / 980) = \$338,571.43 \implies \$428.57\text{/MT}$.
     Revenues for Meal: $\$410.00\text{/MT}$.
     **Gross Profit for Meal:** $\$410.00 - \$428.57 = -\$18.57\text{/MT}$ (**Gross Margin: $-4.53\%$**).
   - Allocated Joint Cost to Oil: $\$420,000 \times (190 / 980) = \$81,428.57 \implies \$428.57\text{/MT}$.
     Total Cost of Refined Oil: $\$428.57\text{ (joint)} + \$40.00\text{ (separable)} = \$468.57\text{/MT}$.
     Revenues for Refined Oil: $\$1,150.00\text{/MT}$.
     **Gross Profit for Oil:** $\$1,150.00 - \$468.57 = +\$681.43\text{/MT}$ (**Gross Margin: $+59.25\%$**).
2. **Net Realizable Value (NRV) Allocation:**
   - Final Sales Value:
     - Meal: $790\text{ MT} \times \$410 = \$323,900$. Separable cost $= \$0$. $NRV_{Meal} = \$323,900$.
     - Oil: $190\text{ MT} \times \$1,150 = \$218,500$. Separable cost $= 190 \times \$40 = \$7,600$. $NRV_{Oil} = \$218,500 - \$7,600 = \$210,900$.
   - Total NRV: $\$323,900 + \$210,900 = \$534,800$.
   - Allocation weights: Meal $= 323,900 / 534,800 = 60.565\%$; Oil $= 210,900 / 534,800 = 39.435\%$.
   - Allocated Joint Cost:
     - Meal: $\$420,000 \times 0.60565 = \$254,371.73 \implies \$322.00\text{/MT}$.
       **Gross Profit for Meal:** $\$410.00 - \$322.00 = +\$88.00\text{/MT}$ (**Gross Margin: $+21.46\%$**).
     - Oil: $\$420,000 \times 0.39435 = \$165,628.27 \implies \$871.73\text{/MT}$.
       Total Cost of Refined Oil: $\$871.73 + \$40.00 = \$911.73\text{/MT}$.
       **Gross Profit for Oil:** $\$1,150.00 - \$911.73 = +\$238.27\text{/MT}$ (**Gross Margin: $+20.72\%$**).
3. **Critique:**
   Physical mass allocation produces a catastrophic operational distortion by assigning a negative gross margin ($-4.53\%$) to soybean meal, which constitutes 80% of total plant production. A commercial sales team operating under physical allocation would attempt to restrict soybean meal sales or seek impossible price increases in a commodity market where meal trades at global parity. In reality, the crushing facility exists to extract high-value crude oil while meal covers baseline operating costs. The NRV method properly aligns product profitability ($+21.46\%$ vs $+20.72\%$) with underlying market economics, preventing artificial product abandonment and distorted inventory valuations under IFRS 15.

---

### Answer 5: Designing an Observable Sensor Network for a Polymer Plant
1. **Degrees of Freedom:**
   $n = 8$ streams, $m = 4$ independent nodal equations. Degrees of freedom $= n - m = 8 - 4 = 4$.
   A minimum of 4 streams must be measured to obtain a unique solution if the network is non-redundant. With 5 meters, the network has $5 - 4 = 1$ degree of redundancy.
2. **Observability of Streams $\{2, 4, 6\}$ given Measured $\{1, 3, 5, 7, 8\}$:**
   Unmeasured columns: $A_u = A_{[:, [2, 4, 6]]}$:
   $$A_u = \begin{bmatrix}
   -1 &  0 &  0 \\
    1 & -1 &  0 \\
    0 &  1 & -1 \\
    0 &  0 &  1
   \end{bmatrix}$$
   The columns of $A_u$ are linearly independent: $\operatorname{rank}(A_u) = 3 = n_u$.
   Because $A_u$ has full column rank, all unmeasured streams $\{2, 4, 6\}$ are **strictly observable**.
   - Stream 2 is solved from Node 1: $\hat{x}_2 = \hat{x}_1 - \hat{x}_3$.
   - Stream 6 is solved from Node 4: $\hat{x}_6 = \hat{x}_7 + \hat{x}_8 - \hat{x}_5$.
   - Stream 4 is solved from Node 3: $\hat{x}_4 = \hat{x}_6 - \hat{x}_3$.
   - Node 2 provides the single redundant equation: $\hat{x}_2 - \hat{x}_4 - \hat{x}_5 = 0$.
3. **Stream Redundancy Classification:**
   Streams $\{1, 3, 5, 7, 8\}$ and co-estimated $\{2, 4, 6\}$ all participate in the cycle passing through Node 2. Therefore, all measured streams are **redundant**.
4. **Optimal 5-Meter Placement:**
   Measuring streams $\{1, 2, 5, 6, 7\}$ creates two independent cycles, localizing gross errors to specific sub-units rather than spreading residuals across all 4 nodes.

---

### Answer 6: Robust M-Estimator Reconciliation under Outlier Contamination
1. **Standard GLS Smearing:**
   Constraint: $x_1 + x_2 + x_3 - x_4 = 0$. Incidence: $A = [1, \, 1, \, 1, \, -1]$.
   $\Sigma = \operatorname{diag}(1, 1, 1, 1)$. $V = A \Sigma A^T = 1 + 1 + 1 + 1 = 4.0$.
   Raw imbalance: $r = 25 + 30 + 45 - 125 = -25.0\text{ MT/h}$.
   $\lambda = r / V = -25.0 / 4.0 = -6.25$.
   Reconciled flows: $\hat{x} = y - \Sigma A^T \lambda$:
   - $\hat{x}_1 = 25.0 - (1)(1)(-6.25) = 31.25\text{ MT/h}$ (corrupted by $+6.25$)
   - $\hat{x}_2 = 30.0 - (1)(1)(-6.25) = 36.25\text{ MT/h}$ (corrupted by $+6.25$)
   - $\hat{x}_3 = 45.0 - (1)(1)(-6.25) = 51.25\text{ MT/h}$ (corrupted by $+6.25$)
   - $\hat{x}_4 = 125.0 - (1)(-1)(-6.25) = 118.75\text{ MT/h}$ (absorbs only $-6.25$)
   The $+25.0$ gross error on Stream 4 is smeared equally ($+6.25\text{ MT/h}$) across all three clean inlet streams!
2. **Huber Weighting:**
   Residuals under GLS: $e = y - \hat{x} = [-6.25, \, -6.25, \, -6.25, \, +6.25]^T$.
   Normalized errors: $u_i = e_i / \sigma_i = \pm 6.25$.
   Since $|u_i| = 6.25 > c = 1.345$, all streams enter the linear penalty zone.
   Weight function: $w(u) = c / |u| = 1.345 / 6.25 = 0.2152$.
3. **Suppression in Iteration 1:**
   In asymmetric Huber tuning where clean streams have small residuals, Stream 4's large error leads to an effective variance $\sigma_{4,(1)}^2 = 1.0 / (c / |u_4|) \gg 1.0$. As $\sigma_4^2 \to \infty$, GLS adjustment concentrates on Stream 4:
   $$\hat{x}_1 \to 25.0, \quad \hat{x}_2 \to 30.0, \quad \hat{x}_3 \to 45.0, \quad \hat{x}_4 \to 100.0$$
   The robust estimator completely suppresses gross error smearing.

---

### Answer 7: Crude Distillation Cut Point Shift and Margin Optimization
1. **Marginal Yield Shift Calculation:**
   $Y(T) = 1 - \exp\left[ - ((T - 280) / 420)^{1.85} \right]$.
   Evaluate at $T_1 = 453.15\text{ K}$ and $T_2 = 458.15\text{ K}$:
   - At $T_1 = 453.15\text{ K}$: $(453.15 - 280) / 420 = 173.15 / 420 = 0.41226$.
     $(0.41226)^{1.85} = 0.19427$.
     $Y(453.15) = 1 - \exp(-0.19427) = 1 - 0.82344 = 0.17656$ ($17.656\%$).
   - At $T_2 = 458.15\text{ K}$: $(458.15 - 280) / 420 = 178.15 / 420 = 0.42417$.
     $(0.42417)^{1.85} = 0.20455$.
     $Y(458.15) = 1 - \exp(-0.20455) = 1 - 0.81501 = 0.18499$ ($18.499\%$).
   - Marginal Yield Shift: $\Delta Y = 0.18499 - 0.17656 = 0.00843$ ($0.843\%$ of crude throughput).
   Across $50,000\text{ BPSD}$:
   $$\Delta \text{Volume} = 50,000 \times 0.00843 = 421.5\text{ bbl/day}$$
   $421.5\text{ bbl/day}$ shifts from Jet Fuel into Heavy Naphtha.
2. **Daily Financial Impact:**
   Price difference: $p_{Naphtha} - p_{Jet} = \$82.00 - \$96.00 = -\$14.00\text{/bbl}$.
   Daily Revenue Impact: $421.5\text{ bbl/day} \times (-\$14.00\text{/bbl}) = -\$5,901.00\text{/day}$ (Revenue loss).
   *Operating Insight:* Shifting cut volume from Jet into Naphtha destroys $\$5,901\text{/day}$ unless the downstream catalytic reformer generates high-octane blendstock commanding a premium over jet fuel.
3. **Physical Quality Constraints:**
   The cut point cannot be lowered indefinitely to maximize Jet Fuel because:
   - Jet Fuel **Flash Point** ($38.0^\circ\text{C}$ minimum per ASTM D1655) requires boiling off light ends; lowering $T_{cut}$ pulls light hydrocarbons into Jet Fuel, dropping the flash point below legal aviation safety limits.
   - Heavy Naphtha **End Point** ($180^\circ\text{C} - 205^\circ\text{C}$) constrains catalytic reformer coking.

---

## 14. Interview questions

### Question 1: Mathematical Foundations of Data Reconciliation
**Prompt:** "Why can't we simply balance a refinery flow network using Ordinary Least Squares (OLS) or by adjusting flows proportionally to their throughput?"
- **Strong Answer:** "Ordinary Least Squares treats all measurements as having equal variance ($\Sigma = \sigma^2 I$), which fundamentally violates process reality. A Coriolis meter on a custody transfer line has an uncertainty of $\pm 0.15\%$, whereas an orifice meter on high-temperature atmospheric bottoms exhibits $\pm 2.0\%$. OLS forces equal adjustments across both, corrupting high-precision meters. Generalized Least Squares (GLS) solves the maximum-likelihood problem by weighting adjustments by $\Sigma^{-1}$, ensuring high-accuracy meters remain anchored while noisy meters absorb necessary balance adjustments. Proportional throughput balancing is arbitrary, lacks statistical justification, fails to provide confidence intervals, and cannot detect gross errors."
- **Weak Answer:** "Because OLS is for linear regression and data reconciliation has constraints. Adjusting proportionally is unfair to big pipes."

---

### Question 2: Gross Error Smearing vs Robust Estimation
**Prompt:** "What is gross error smearing, and how do M-estimators prevent it in an industrial DCS network?"
- **Strong Answer:** "Gross error smearing occurs because standard GLS utilizes an $L_2$ quadratic objective function $\frac{1}{2}(y - x)^T \Sigma^{-1}(y - x)$. When a single sensor has a massive gross bias (e.g., $+15\text{ MT/h}$), the quadratic penalty heavily penalizes assigning a large residual to that single instrument ($15^2 = 225$). Instead, the solver minimizes the global objective by spreading smaller errors across five adjacent clean instruments ($5 \times 3^2 = 45$). M-estimators, such as the Huber or Cauchy estimator, replace the quadratic penalty with a linear or sub-linear loss function beyond a threshold $c$. Because the marginal penalty $\psi(u) = \rho'(u)$ is bounded or redescending, the solver willingly assigns the entire discrepancy to the broken sensor without perturbing adjacent clean streams."
- **Weak Answer:** "Smearing is when errors spread around. You fix it by averaging the data over a longer time window or deleting outliers manually."

---

### Question 3: Observability and Crowe's Projection
**Prompt:** "If a chemical reactor has three unmeasured streams, how do you determine if the network can still be reconciled, and how do you solve it mathematically without inverting a singular matrix?"
- **Strong Answer:** "We partition the incidence matrix into measured and unmeasured columns: $A_m x_m + A_u x_u = 0$. Observability requires that $A_u$ have full column rank ($\operatorname{rank}(A_u) = n_u$). If $\operatorname{rank}(A_u) < n_u$, the network contains unobservable degrees of freedom and cannot be uniquely reconciled. To solve it without numerical singularities, we apply Crowe's QR decomposition: $A_u = [Q_1 \quad Q_2] [R_1; \, 0]$. Premultiplying by $Q_2^T$ annihilates the unmeasured streams ($Q_2^T A_u = 0$), yielding a reduced constraint system $C x_m = 0$, where $C = Q_2^T A_m$. We solve standard GLS on measured streams using $C$, and then back-calculate the unmeasured observable flows via $\hat{x}_u = -R_1^{-1} Q_1^T A_m \hat{x}_m$."
- **Weak Answer:** "You can't solve it if you don't have meters on all the pipes. You have to buy three more meters or guess their flows."

---

### Question 4: Volume Non-Additivity in Pipeline Operations
**Prompt:** "A crude trader buys 50,000 barrels of light condensate and 50,000 barrels of heavy bitumen, blends them in a tank, and finds the radar gauge measures only 99,100 barrels. Did the terminal steal 900 barrels?"
- **Strong Answer:** "No. Hydrocarbon blending exhibits excess volume of mixing ($\Delta V_{mix} < 0$). Small, light hydrocarbon molecules (high $^\circ\text{API}$ condensate) fit into the interstitial spaces between large, complex asphaltene and resin molecules (low $^\circ\text{API}$ bitumen), causing physical volumetric shrinkage. The 900-barrel discrepancy represents approximately a $0.9\%$ shrinkage, which is physically typical for a $40^\circ\text{API}$ difference. To verify custody transfer integrity, the transaction must be audited on a mass basis ($\text{metric tons}$). If mass is conserved ($M_{blend} = M_{condensate} + M_{bitumen}$), no physical product was lost. Terminal operating contracts must always specify API Chapter 11/12 mass reconciliation."
- **Weak Answer:** "Yes, 900 barrels are missing. The tank gauge was probably broken or vapor escaped into the atmosphere."

---

### Question 5: True Boiling Point (TBP) Curves vs ASTM D86
**Prompt:** "Why do refinery LP planning models rely on True Boiling Point (TBP) distillation curves instead of standard ASTM D86 laboratory test results?"
- **Strong Answer:** "ASTM D86 is a batch, single-stage distillation with no fractionation column (1 theoretical stage, zero reflux). It exhibits massive component co-distillation and overlapping boiling ranges; light and heavy molecules boil off together. TBP (ASTM D2892) utilizes 15 theoretical plates and a 5:1 reflux ratio, providing true thermodynamic separation where cut points correspond to distinct physical boiling temperatures. Refinery fractionation columns operate with 30 to 60 trays; their separation efficiency mimics TBP, not ASTM D86. If an LP model used ASTM D86 curves, it would drastically underestimate the yield of high-margin middle distillates (jet fuel and diesel) and overestimate column energy requirements."
- **Weak Answer:** "TBP is more modern and accurate than D86 because it takes longer to run in the laboratory."

---

### Question 6: Joint Costing: Mass vs Net Realizable Value (NRV)
**Prompt:** "In a petrochemical steam cracker yielding ethylene, propylene, and heavy pyrolysis fuel oil, why does allocating joint feedstock costs on a mass basis distort corporate decision-making?"
- **Strong Answer:** "Steam cracking raw naphtha yields approximately 30% ethylene, 16% propylene, and 10% pyrolysis fuel oil (pyoil). Ethylene and propylene sell at high market prices ($\$1,000 - \$1,200\text{/MT}$), while pyoil is an aromatic heavy by-product selling at fuel parity ($\$350\text{/MT}$), often below feedstock naphtha cost ($\$650\text{/MT}$). Physical mass allocation assigns an equal cost of $\$650\text{/MT}$ to all streams. Pyoil instantly shows a paper loss of $-\$300\text{/MT}$, while ethylene shows an inflated profit. Product managers might attempt to cancel pyoil sales, which is physically impossible because pyoil is an unavoidable joint by-product. Net Realizable Value (NRV) allocates joint costs based on revenue contribution, ensuring all products have rational margins and preventing misallocation of capital."
- **Weak Answer:** "Mass costing is bad because ethylene is more expensive than pyoil, so ethylene should get all the costs."

---

### Question 7: Reconciled Data vs Planning Model Feasibility
**Prompt:** "What happens if a refinery planning engineer updates Aspen PIMS yield vectors using raw flowmeter measurements instead of reconciled data?"
- **Strong Answer:** "Raw plant data contains balance closure errors ($A y \neq 0$). If raw yields are fed into an LP planning engine, the LP solver encounters internal contradictions. For example, if raw meters report that a fluid catalytic cracker (FCC) produces 103% mass yield from its feed, the LP will allocate infinite crude to that unit to exploit the phantom mass creation. Conversely, if a unit shows a $-4\%$ unmetered loss, the LP may artificially throttle an economically viable unit. Furthermore, unadjusted yields cause multi-period storage balance violations, leading to schedule infeasibilities where intermediate tanks either overflow or run dry in actual operations."
- **Weak Answer:** "The LP runs slightly slower because the numbers have decimals, and the manager will get annoyed by the messy reports."

---

### Question 8: Statistical Distribution of Nodal vs Measurement Residuals
**Prompt:** "In Gross Error Detection, why does the Global Test use a Chi-square distribution while the Measurement Test uses a Normal distribution?"
- **Strong Answer:** "The raw nodal imbalance vector $r = A y$ is a linear combination of Gaussian measurement errors: $r \sim \mathcal{N}(0, V)$, where $V = A \Sigma A^T$. The Global Test statistic $\gamma = r^T V^{-1} r$ is the sum of squares of $m$ independent, standardized Gaussian random variables. By definition, the sum of squares of $m$ standard normal variables follows a central Chi-square distribution with $m$ degrees of freedom ($\gamma \sim \chi^2(m)$). In contrast, the measurement residual $e_i = y_i - \hat{x}_i$ is an individual linear scalar error. When divided by its theoretical standard deviation $\sqrt{Q_{ii}}$, the resulting standardized residual $z_i = |e_i| / \sqrt{Q_{ii}}$ follows a univariate standard normal distribution ($\mathcal{N}(0, 1)$), enabling a two-tailed $Z$-test."
- **Weak Answer:** "Chi-square is used for testing matrices and normal distributions are used for testing single numbers."

---

## 15. Key terms

- **[Mass Balance](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Conservation of total mass across closed boundaries ($A x = 0$).
- **[Component Balance](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Conservation of specific chemical species or atomic elements.
- **[Enthalpy Balance](02_yield-mass-balance.md#4-definitions-and-vocabulary):** First law thermodynamic energy balance across process units.
- **[Volume Non-Additivity](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Interstitial molecular packing causing volumetric shrinkage ($\Delta V_{mix} < 0$).
- **[True Boiling Point (TBP)](02_yield-mass-balance.md#4-definitions-and-vocabulary):** High-fractionation batch distillation curve (ASTM D2892).
- **[ASTM D86](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Standard single-stage laboratory screening distillation.
- **[Simulated Distillation](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Gas chromatography distillation assay (ASTM D2887).
- **[Pseudo-Component](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Lumped boiling range cut representing complex hydrocarbon mixtures.
- **[Cut Point](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Vapor draw temperature setpoint separating distillation fractions.
- **[Atmospheric Residue](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Heavy bottoms cut boiling above $340^\circ\text{C}-370^\circ\text{C}$.
- **[Vacuum Residue](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Heavy vacuum bottoms cut boiling above $540^\circ\text{C}-565^\circ\text{C}$.
- **[Data Reconciliation](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Constrained optimization adjusting measurements to satisfy conservation laws.
- **[Generalized Least Squares (GLS)](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Weighted least-squares using inverse covariance $\Sigma^{-1}$.
- **[Covariance Matrix ($\Sigma$)](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Sensor error dispersion matrix capturing instrument uncertainties.
- **[Incidence Matrix ($A$)](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Directed graph connectivity matrix representing process topology.
- **[Spatial Redundancy](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Multi-path graph topology allowing cross-calculation of flows.
- **[Temporal Redundancy](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Time-series inventory integration across vessel holdups.
- **[Observability](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Mathematical ability to determine unmeasured flows from available sensors.
- **[Crowe's Projection](02_yield-mass-balance.md#5-the-mechanics-step-by-step):** QR decomposition method eliminating unmeasured variables from reconciliation.
- **[Gross Error](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Systematic sensor bias, instrument failure, or unmetered leak.
- **[Global Test ($\gamma$)](02_yield-mass-balance.md#5-the-mechanics-step-by-step):** Chi-square test evaluating network-wide balance closure ($r^T V^{-1} r$).
- **[Measurement Test ($z_i$)](02_yield-mass-balance.md#5-the-mechanics-step-by-step):** Standardized residual score isolating biased instruments ($|e_i| / \sqrt{Q_{ii}}$).
- **[Gross Error Smearing](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Artifact where $L_2$ optimization distributes one bias across clean sensors.
- **[Huber M-Estimator](02_yield-mass-balance.md#5-the-mechanics-step-by-step):** Robust loss function suppressing outlier influence.
- **[Split-off Point](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Stage in joint processing where co-products become separately identifiable.
- **[Joint Products](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Multiple primary co-products of significant economic value.
- **[By-Product](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Incidental output of minor economic value relative to joint products.
- **[Net Realizable Value (NRV)](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Final product sales value minus separable downstream costs.
- **[Sales Value at Split-off](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Cost allocation proportional to unrefined market value at separation.
- **[API Gravity](02_yield-mass-balance.md#4-definitions-and-vocabulary):** Inverted petroleum liquid density relative to water at $60^\circ\text{F}$.

---

## 16. Sources

- **Almasy, G. A., & Sztano, T. (1975).** Checking and correction of measurements on the basis of linear system models. *Problems of Control and Information Theory*, 4(1), 57–69.
- **American Petroleum Institute (API). (2018).** *Manual of Petroleum Measurement Standards (MPMS) Chapter 12: Calculation of Petroleum Quantities*. Washington, D.C.: API Publishing.
- **ASTM International. (2020).** *ASTM D2892-20: Standard Test Method for Distillation of Crude Petroleum (15-Theoretical Plate Column)*. West Conshohocken, PA: ASTM International. https://doi.org/10.1520/D2892-20
- **Crowe, C. M. (1986).** Data reconciliation in chemical process networks: A review. *The Canadian Journal of Chemical Engineering*, 64(2), 279–288. https://doi.org/10.1002/cjce.5450640215
- **European Commission. (2018).** *Commission Implementing Regulation (EU) 2018/2066 on the monitoring and reporting of greenhouse gas emissions (MRR)*. Official Journal of the European Union, L 334, 1–93.
- **Gary, J. H., Handwerk, G. E., & Kaiser, M. J. (2007).** *Petroleum Refining: Technology and Economics* (5th ed.). CRC Press. https://doi.org/10.1201/9781420008456
- **International Organization for Standardization. (2006).** *ISO 14044: Environmental management — Life cycle assessment — Requirements and guidelines*. Geneva: ISO.
- **Mah, R. S. H., Stanley, G. M., & Downing, D. M. (1976).** Reconciliation and rectification of process flow and inventory data. *Industrial & Engineering Chemistry Process Design and Development*, 15(1), 175–183. https://doi.org/10.1021/i260057a030
- **Narasimhan, S., & Jordache, C. (2000).** *Data Reconciliation and Gross Error Detection: An Intelligent Use of Process Data*. Gulf Publishing Company.
- **Riazi, M. R. (2005).** *Characterization and Properties of Petroleum Fractions*. ASTM International. https://doi.org/10.1520/MNL50-EB
- **Rollins, D. K., & Davis, J. F. (1992).** Unbiased estimation of gross errors in process networks. *AIChE Journal*, 38(4), 563–572. https://doi.org/10.1002/aic.690380409
- **Romagnoli, J. A., & Sanchez, M. C. (2000).** *Data Processing and Reconciliation for Chemical Process Operations*. Academic Press. https://doi.org/10.1016/B978-0-12-594750-3.X5000-4
- **United States Environmental Protection Agency (US EPA). (2016).** *Mandatory Greenhouse Gas Reporting: Petroleum Refineries (40 CFR Part 98, Subpart Y)*. Washington, D.C.: Federal Register.

---

## 17. Links

- **Parent Module Index:** [`_index.md`](_index.md)
- **Preceding Sibling Topic:** [`01_continuous-batch-discrete.md`](01_continuous-batch-discrete.md) (Continuous, Batch, and Discrete Production)
- **Following Sibling Topic:** [`03_quality-specs-blending.md`](03_quality-specs-blending.md) (Quality Specifications and Blending)
- **Related Sibling Topics:**
  - [`04_campaigns-changeover.md`](04_campaigns-changeover.md) (Campaigns and Changeover Optimization)
  - [`05_bulk-storage-inventory.md`](05_bulk-storage-inventory.md) (Bulk Storage, Heels, and Segregation)
  - [`08_capacity-debottlenecking.md`](08_capacity-debottlenecking.md) (Capacity and Debottlenecking)
- **Root Survey File:** [`../../../course/38_process_industries_commodities.md`](../../../course/38_process_industries_commodities.md)
- **Relevant Program Modules:**
  - [Module 11: Network Design Optimization](../../02_methods/11_network_design_optimization/_index.md)
  - [Module 16: Supply Chain Network Economics](../../02_methods/16_supply_chain_network_economics/_index.md)
  - [Module 20: Operations Planning and Master Scheduling](../../03_commerce/20_operations_planning_master_scheduling/_index.md)
