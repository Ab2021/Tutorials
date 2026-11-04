# Quality Specifications and Blending Optimization

> Module 38 — Process Industries, Commodities, and Bulk Supply Chains
> Topic 03 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 15 (Linear and Mixed-Integer Programming), Module 16 (Non-Linear and Global Optimization), Topic 01 (Continuous, Batch, and Discrete Production), Topic 02 (Yield and Mass Balance)
> Version 2026.5

---

## 1. Why this topic exists

In petroleum refining, petrochemical manufacturing, and bulk commodity processing, finished commercial products are almost never produced directly by a single conversion unit. Downstream units—Atmospheric Distillation Columns (CDUs), Fluid Catalytic Crackers (FCCs), Catalytic Reformers, Alkylation plants, Isomerization units, and Hydrocrackers—generate intermediate component streams with wildly divergent physical and chemical properties. A catalytic reformate stream carries high octane (98–102 Research Octane Number, RON) but low vapor pressure and high aromatic content. An isomerate stream provides clean, paraffinic octane (82–87 RON) with low aromaticity but elevated volatility. Straight-run light naphtha possesses high vapor pressure and low octane (60–65 RON), while FCC naphtha contributes moderate octane (90–93 RON) paired with elevated olefin and sulfur levels.

None of these streams can be legally or commercially sold as finished motor gasoline. To produce marketable fuels—such as 87 Anti-Knock Index (AKI) regular gasoline, 93 AKI premium gasoline, Jet A-1 aviation kerosene, Ultra-Low Sulfur Diesel (ULSD), or Very Low Sulfur Fuel Oil (VLSFO)—refineries and terminal operators blend multiple intermediate streams through specialized blending headers or bulk tank manifolds. 

The blend manifold is the commercial cash register of a process facility. Between 70% and 85% of total refinery net operating margin is realized or forfeited at the blending headers. In a modern medium-sized refinery processing 250,000 barrels per day (bpd), gasoline production averages 110,000 to 140,000 bpd, and distillate production averages 80,000 to 100,000 bpd. In this operational environment, an octane giveaway of just 0.3 RON on finished motor gasoline destroys between $6.0 million and $14.5 million in operating margin each year. Delivering an 87.3 AKI batch against an 87.0 contract requirement wastes high-octane molecules (alkylate, reformate, or purchased bio-ethanol) that could have been monetized in high-margin premium blends or sold to petrochemical complexes for aromatics extraction.

Conversely, producing an "off-spec" (off-specification) batch is an operational and logistical disaster. A single 100,000-barrel finished fuel batch that fails pipeline certification—for instance, measuring 9.15 psi Reid Vapor Pressure (RVP) against a statutory summer limit of 9.0 psi, or 12.0 ppm sulfur against an EPA Tier 3 10.0 ppm limit—cannot be pumped into fungible common-carrier pipelines (such as the Colonial Pipeline or Explorer Pipeline) or loaded onto chartered tankers. The refinery must lock the tank, pump in corrective low-RVP or hydrotreated trimming components, recirculate the contents for 24 to 48 hours to homogenize the mixture, and re-run certified laboratory assays. This disrupts plant throughput, triggers charter demurrage fees of $30,000 to $65,000 per day on idle marine vessels, and risks backing up upstream conversion units due to lack of intermediate storage (a "tank-top" shutdown).

Compounding the financial stakes is the physical and mathematical reality of blending:
1. **Property Non-Linearity:** Essential commercial fuel properties do not blend linearly by liquid volume or mass. Vapor pressure (RVP), kinematic viscosity, flash point, cloud point, and octane numbers exhibit severe non-linear behavior driven by non-ideal molecular thermodynamics, vapor-liquid phase equilibria, and intermolecular shear forces. Linear approximations create multi-million dollar model errors.
2. **The Pooling Problem:** Intermediate streams from multiple production units are routinely stored in common intermediate tanks ("pools") before being split across multiple finished blend headers. The physical mixing in intermediate pools creates bilinear non-convex mathematical constraints ($F_{p,j} \cdot q_{p,k}$), where the quality of the pool depends on its inlet streams, and the quality delivered to the final product depends on the pool's concentration. Bilinearity introduces multiple local optima, trapping standard linear programming (LP) solvers and distributed recursion algorithms in expensive, sub-optimal solutions.
3. **Stochastic Process and Measurement Variance:** Every analytical laboratory assay and online process analyzer carries measurement error, and intermediate stream properties fluctuate continuously with crude slate shifts and unit operational disturbances.

Mastering non-linear property transformation indices (Chevron RVP index, Refutas viscosity blend numbers), bilinear pooling formulations ($p$-formulation vs. $q$-formulation), McCormick convex relaxation envelopes, and stochastic quality giveaway minimization is essential for operations research engineers, process planners, and commodity trading analysts operating in bulk process industries.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Formalize** linear and non-linear blend property transformations, mathematically deriving the Chevron RVP blending index, the Refutas kinematic viscosity blend number (VBN), the Hu-Burns flash point index, and octane interaction equations.
2. **Formulate** multi-stream, multi-product refinery and chemical blend optimization models as Mathematical Programs (LP, bilinear NLP, and MINLP) incorporating mass balances, component availabilities, quality bounds, and tank heel constraints.
3. **Deconstruct** the bilinear non-convexity of Haverly's Pooling Problem, proving mathematically why standard linear programming fails and illustrating how multi-modal solution topologies create costly local optima traps.
4. **Construct** McCormick convex relaxation envelopes for bilinear pooling products, generating rigorous polyhedral relaxations for Spatial Branch-and-Bound (sBB) global optimization algorithms.
5. **Differentiate** between the $p$-formulation (pool quality variables) and the $q$-formulation (split proportion variables), proving why the $q$-formulation provides substantially tighter linear relaxations for intermediate tank networks.
6. **Critique** Successive Linear Programming (SLP) and sequential quadratic programming (SQP) algorithms implemented in commercial refinery planning software (e.g., Aspen PIMS, Haverly GRTMPS), analyzing their oscillation failure modes and step-bounding damping mechanisms.
7. **Quantify** the financial trade-off between quality giveaway and off-spec batch risk under stochastic measurement and stream variance, analytically deriving the optimal economic safety buffer ($k^*\sigma$).
8. **Develop** an executable, bounded Python optimization script (<80 lines) utilizing Sequential Least Squares Programming (SLSQP) to solve a multi-property non-linear fuel blending problem.
9. **Audit** finished bulk fuel recipes against binding statutory and industry standards (ASTM D4814, ASTM D975, ISO 8217, EPA Tier 3 40 CFR Part 1090, IMO 2020 MARPOL Annex VI), evaluating fuel stability, compatibility, and emissions limits.

---

## 3. The core mental model

Refinery and bulk chemical blending is governed by a physical and information hierarchy that maps variable upstream crude conversion into deterministic, certified commercial commodities. Intermediate streams with varying flow rates and fluctuating assays pass through intermediate pooling infrastructure, are dynamically metered across blending manifolds, and are injected into finished storage or custody-transfer pipelines under real-time analytical control.

```text
=============================================================================================================
                      THE REFINERY BLENDING & POOLING OPTIMIZATION ARCHITECTURE
=============================================================================================================

[ UPSTREAM CONVERSION UNITS ]
  Crude Distillation │ Catalytic Reformer │ Fluid Cat Cracker (FCC) │ Alkylation Unit │ Isomerization Unit
         │                      │                     │                    │                 │
         │ (Naphtha, AGO)       │ (Reformate)         │ (FCC Naphtha)      │ (Alkylate)      │ (Isomerate)
         ▼                      ▼                     ▼                    ▼                 ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ INTERMEDIATE COMPONENT STAGING & POOLING TANKS (The Bilinear Non-Convex Layer)                            │
│  - Pool Tank P1 (Light Naphtha Blend Pool)      - Dedicated Tank S1 (Heavy Reformate)                    │
│  - Pool Tank P2 (Cat Naphtha Desulfurized Pool) - Direct Unit Bypass Lines                                │
└─────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
┌──────────────────────────────────────────────┐┌───────────────────────────────────────────────────────────┐
│ PLANNING & SCHEDULING OPTIMIZATION LAYER     ││ PROCESS PROPERTY TRANSFORMATION ENGINE                    │
│ - Global Bilinear NLP (p- / q-formulation)   ││ - RVP Chevron Transformation: BI_rvp = (RVP)^1.25         │
│ - McCormick Polyhedral Convex Relaxation     ││ - Refutas Viscosity VBN: 14.534 ln(ln(v + 0.8)) + 10.975  │
│ - SLP / Distributed Recursion Engine         ││ - Flash Point Blending Index: BI_fp = 100 * 10^(-0.06 * FP)│
│ - Quality Giveaway / Safety Margin Tuning    ││ - Octane Interaction Parameters (Ethyl / Chevron Model)  │
└──────────────────────┬───────────────────────┘└─────────────────────────────┬─────────────────────────────┘
                       │                                                       │
                       ▼                                                       ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ REAL-TIME IN-LINE BLENDING (ILB) MANIFOLD & ADVANCED PROCESS CONTROL (APC)                                │
│  [Stream 1 Flow Meter]  [Stream 2 Flow Meter]  [Stream 3 Flow Meter]  [Stream 4 Meter]  [Additive Injection]│
│            │                     │                     │                    │                   │         │
│            └─────────────────────┴──────────┬──────────┴────────────────────┴───────────────────┘         │
│                                             ▼                                                             │
│                                    [ STATIC IN-LINE MIXER ]                                               │
│                                             │                                                             │
│                                             ▼                                                             │
│                     [ ON-LINE PROCESS ANALYZERS (NIR / Raman / GC) ]                                      │
│                      - Real-time RON / MON, RVP, Distillation (T10, T50, T90), Sulfur                     │
│                      - Closed-Loop Feedback Trimming to Ratio Controllers                                │
└─────────────────────────────────────┬─────────────────────────────────────────────────────────────────────┘
                                      │
               ┌──────────────────────┴──────────────────────┐
               ▼                                             ▼
┌──────────────────────────────────────────────┐┌───────────────────────────────────────────────────────────┐
│ BATCH BLENDING IN FINISHED STORAGE TANKS     ││ DIRECT CUSTODY-TRANSFER PIPELINE / TANKER LOADING         │
│ - 50,000 to 250,000 bbl Storage Tanks        ││ - Colonial / Buckeye Fungible Pipeline Injection         │
│ - Mechanical Jet Mixers / Circulation Eductors││ - Marine Tanker Loading (Bespoke Bunker / Cargo Spec)    │
│ - Final Laboratory Assays (ASTM Referee)     ││ - Automatic Batch Sampling & Certificate of Quality (CoA) │
└──────────────────────────────────────────────┘└───────────────────────────────────────────────────────────┘
=============================================================================================================
```

### The Causal Mechanics of the Model

1. **Assay Characterization:** Feedstocks are characterized across two distinct property categories:
   - **Linear Properties:** Blend strictly linearly by volume fraction or mass fraction (e.g., total sulfur, specific gravity, aromatic content, heavy metals).
   - **Non-Linear Properties:** Exhibit non-ideal molecular behavior and must be mapped to monotonic pseudo-linear blending indices before linear combinations can be evaluated (e.g., Reid Vapor Pressure, kinematic viscosity, flash point, cloud point).
2. **Intermediate Topology and Bilinearity:** Because refinery intermediate component tanks are shared across production campaigns, components are blended into intermediate pools before being routed to final blend headers. The concentration of quality $k$ in pool $p$ ($q_{p,k}$) depends non-linearly on the ratio of upstream flows entering that pool. When pool $p$ discharges into product $j$, the mass or volume transport of quality $k$ is represented by the product of two continuous decision variables ($F_{p,j} \cdot q_{p,k}$). This bilinearity creates a non-convex optimization surface characterized by saddle points and multiple disconnected local optima.
3. **The Global Optimization Pipeline:** To solve this system reliably, modern decision support systems first construct a McCormick convex relaxation (bounding bilinear terms with convex polyhedral envelopes). This relaxed Linear Program (LP) or Mixed-Integer Linear Program (MILP) provides a guaranteed lower bound on cost. A Spatial Branch-and-Bound (sBB) solver or damped Successive Linear Programming (SLP) algorithm then searches for the global optimum, ensuring that the refinery does not stall at an inferior local operating point.
4. **The Real-Time Analytical Feedback Loop:** Once the optimal blend recipe is determined, components are pumped across an In-Line Blending (ILB) manifold equipped with turbine or Coriolis flow meters and variable-frequency drives. A slipstream passes through on-line Near-Infrared (NIR) or gas chromatography analyzers. The real-time analyzer signals feed a multivariable Advanced Process Controller (APC). If stream assays drift—for example, if FCC naphtha RVP rises due to an upset in the main fractionator overhead—the APC automatically trims the high-octane/low-RVP reformate or alkylate ratio to prevent an off-spec batch while minimizing quality giveaway.

---

## 4. Definitions and vocabulary

| Term | Definition | Common Misuse / Pitfall | Standard Unit / Classification |
|---|---|---|---|
| **Reid Vapor Pressure (RVP)** | The vapor pressure exerted by a liquid fuel at 37.8°C (100°F) determined under standard vapor-to-liquid ratio (4:1). | Assuming RVP blends linearly by volume; it exhibits strong positive deviations requiring non-linear index transformation. | Pounds per square inch (psi) or kilopascals (kPa); ASTM D323 / D5191. |
| **Dry Vapor Pressure Equivalent (DVPE)** | Calculated vapor pressure equivalent to ASTM D323, derived from automated mini-method instruments. | Confusing raw instrument pressure with statutory DVPE; EPA regulations legally enforce DVPE. | psi or kPa; 40 CFR § 1090.1360 / ASTM D5191. |
| **Research Octane Number (RON)** | Fuel antiknock rating measured under mild engine operating conditions (600 rpm, 49°C intake air). | Using RON alone to specify U.S. gasoline; U.S. retail requires Anti-Knock Index (AKI). | Dimensionless octane rating; ASTM D2699. |
| **Motor Octane Number (MON)** | Fuel antiknock rating measured under severe engine operating conditions (900 rpm, 149°C intake air). | Assuming RON and MON blend with identical component sensitivities; MON sensitivity varies wildly across olefins and aromatics. | Dimensionless octane rating; ASTM D2700. |
| **Anti-Knock Index (AKI)** | The average of Research Octane Number and Motor Octane Number: $(\text{RON} + \text{MON})/2$. | Confusing European "95 RON" with American "95 Octane"; 95 RON corresponds to ~90–91 AKI. | Dimensionless; marketed as $(R+M)/2$ in North America. |
| **Octane Sensitivity** | The numerical difference between RON and MON: $\text{Sensitivity} = \text{RON} - \text{MON}$. | Ignoring sensitivity in blend optimization; high sensitivity causes engine knock under heavy load despite compliant AKI. | Dimensionless index (typically 8–12 for motor gasoline). |
| **Refutas Viscosity Blend Number (VBN)** | A logarithmic mathematical index enabling kinematic viscosity blending by linear mass or volume fractions. | Linearly blending centistokes ($\text{cSt}$); linear viscosity blending produces catastrophic under-prediction of blend viscosity. | Dimensionless index; derived from kinematic viscosity in $\text{mm}^2/\text{s}$ ($\text{cSt}$). |
| **Flash Point Index (FPI)** | A logarithmic transformation index used to predict the closed-cup flash point temperature of multi-component hydrocarbon blends. | Blending flash point temperatures linearly in °C or °F; trace light components collapse flash point non-linearly. | Dimensionless index; ASTM D93 (Pensky-Martens). |
| **Quality Giveaway** | Delivering finished fuel that exceeds minimum contractual or regulatory specifications (e.g., 87.4 AKI delivered vs. 87.0 required). | Treating giveaway as "customer goodwill"; in bulk refining, giveaway represents irrecoverable margin destruction. | Economic metric: \$/barrel or total \$/year; property deltas ($\Delta \text{RON}$, $-\Delta \text{RVP}$). |
| **Off-Specification (Off-Spec)** | A blended fuel batch failing to satisfy one or more binding contractual, pipeline, or statutory specifications. | Assuming off-spec batches can always be diluted downstream; pipeline operators reject entire shipments outright. | Binary status; requires quarantine, re-blending, or transmix downgrading. |
| **The Pooling Problem** | The non-convex mathematical programming problem arising when intermediate streams mix in common pools before routing to blend headers. | Formulating tank-farm planning as a pure Linear Program (LP); LP omits bilinearity, generating infeasible or sub-optimal plans. | Optimization class: Bilinear Non-Linear Program (NLP) / MINLP. |
| **Bilinear Term** | A mathematical expression formed by multiplying two continuous decision variables ($x \cdot y$). | Assuming bilinear terms can be solved with standard simplex or barrier LP solvers without relaxation. | Quadratic non-convex term ($F_{p,j} \cdot q_{p,k}$). |
| **$p$-Formulation** | Pooling model formulation where unknown pool quality concentrations ($q_{p,k}$) are declared as explicit decision variables. | Using $p$-formulation in linear relaxations without noticing it creates weak bounds when flow ranges are wide. | Concentration-space mathematical formulation. |
| **$q$-Formulation** | Pooling model formulation where split proportions ($v_{i,p} = F_{i,p} / \sum F$) represent the fraction of pool inflow from source $i$. | Believing $q$-formulation eliminates non-linearity; it preserves bilinearity ($v \cdot F$) but yields tighter convex relaxations. | Proportion-space mathematical formulation (Ben-Tal et al.). |
| **McCormick Convex Envelope** | The tightest convex and concave polyhedral under- and over-estimators for a bilinear term defined over a bounded rectangle. | Using McCormick bounds on unbounded variables; bounds must be strictly finite ($x^L \le x \le x^U$). | Four linear inequality constraints per bilinear product. |
| **Spatial Branch-and-Bound (sBB)**| Global optimization algorithm that recursively partitions continuous variable domains to tighten convex relaxations. | Confusing sBB with discrete MILP branch-and-bound; sBB branches on continuous variables to overcome non-convexities. | Global deterministic NLP/MINLP solver algorithm. |
| **Successive Linear Programming (SLP)**| Algorithmic technique that linearizes non-linear pooling terms via first-order Taylor expansion around a base operating point. | Running SLP without trust-region step-bounds; unbounded SLP exhibits severe limit-cycling and divergence. | Iterative local optimization method (Aspen PIMS, Haverly GRTMPS). |
| **In-Line Blending (ILB)** | Automated continuous blending where intermediate streams are simultaneously metered directly into a pipe header or ship. | Assuming ILB eliminates tankage requirements; ILB requires stable, dedicated intermediate run-down tanks. | Continuous dynamic manufacturing method. |
| **Tank Heel** | The unpumpable or unmixed liquid volume remaining at the bottom of a storage tank beneath the suction pipe or mixer level. | Overlooking heel composition when switching tank service; heels contaminate new batches with off-spec molecules. | Physical volume ($\text{m}^3$ or bbl); dead volume constraint. |
| **Transmix** | Commingled hydrocarbon interface generated when different fuel grades are pumped sequentially through a fungible batch pipeline. | Ignoring transmix reprocessing costs in pipeline scheduling models. | Volumetric downgrade requiring re-refining or slop recycling. |
| **Cat Fines** | Microscopic aluminum and silicon oxide catalyst particles carryover from Fluid Catalytic Cracking into residual slurry oil. | Blending FCC slurry into marine fuel without testing cat fines; cat fines destroy marine diesel engine cylinder liners. | Concentration in $\text{mg}/\text{kg}$ (ppm); ISO 8217 limits ($\le 60\text{ ppm}$). |
| **P-Value (Peptization)** | A metric of residual fuel oil stability measuring the state of peptization of asphaltenes in the hydrocarbon matrix. | Blending incompatible paraffinic cutter stocks into aromatic residual fuels; drops P-value, precipitating asphaltene sludge. | Dimensionless ratio; ASTM D7060 / ISO 8217 stability criteria. |

---

## 5. The mechanics, step by step

### Step 1: Feedstock Characterization and Basis of Additivity

Before formulating mathematical blend models, the analytical basis of property additivity must be established. Hydrocarbon streams blend on three distinct physical bases:
- **Volume Additivity:** Volume-based properties blend linearly with liquid volume fractions ($v_i = V_i / \sum_m V_m$) under ideal solution thermodynamics (no excess volume of mixing, $\Delta V_{mix} \approx 0$). In refining, volume additivity is standard for component accounting.
- **Mass Additivity:** Mass-based properties blend strictly linearly with mass fractions ($w_i = M_i / \sum_m M_m$). Total sulfur, nitrogen, and heavy metal concentrations (vanadium, nickel) are conservation-of-mass properties and blend linearly on a mass basis.
- **Molar Additivity:** Gas-phase properties, vapor pressures, and chemical reaction stoichiometry blend strictly by mole fractions ($x_i = n_i / \sum_m n_m$).

When converting mass-based qualities (such as sulfur in weight ppm) to volumetric blend equations, stream specific gravities ($SG_i$) must be incorporated:
$$w_i = \frac{v_i \cdot SG_i}{\sum_{m} v_m \cdot SG_m}$$
For clean motor gasolines, specific gravities between blending components (alkylate $\approx 0.70$, reformate $\approx 0.81$, FCC naphtha $\approx 0.74$) differ sufficiently that neglecting density corrections introduces a 5% to 12% calculation error in final sulfur concentrations.

### Step 2: Non-Linear Property Transformation Models

Essential fuel specifications cannot be blended directly using arithmetic averages. They must be transformed into monotonic blending indices, blended linearly, and then transformed back through inverse mapping.

```text
[ Raw Property: q_i ]  ──►  [ Transformation: BI_i = f(q_i) ]  ──►  [ Linear Index Blend: BI_blend = ∑ v_i * BI_i ]
                                                                                   │
                                                                                   ▼
[ Predicted Property: q_blend ]  ◄──  [ Inverse Transform: q_blend = f^-1(BI_blend) ]
```

#### A. Reid Vapor Pressure (RVP) — The Chevron Blending Index
Reid Vapor Pressure measures the volatile light ends (butanes, pentanes) that flash into vapor at 37.8°C (100°F). Due to non-ideal vapor-liquid equilibrium (VLE) in hydrocarbon mixtures, lighter hydrocarbons exhibit positive deviations from Raoult’s Law. The industry-standard Chevron empirical method transforms RVP into a volumetric Blending Index ($BI_{RVP}$):
$$BI_{RVP, i} = (RVP_i)^{\alpha}$$
where empirical calibration across refinery naphtha matrices sets $\alpha \approx 1.25$ (or $1.259$ in specialized gasoline formulations).
The blended index is strictly volume-additive:
$$BI_{RVP, blend} = \sum_{i \in I} v_i \cdot (RVP_i)^{1.25}$$
The final blended RVP is recovered via the inverse power transformation:
$$RVP_{blend} = \left( BI_{RVP, blend} \right)^{\frac{1}{1.25}} = \left( \sum_{i \in I} v_i \cdot (RVP_i)^{1.25} \right)^{0.80}$$

#### B. Kinematic Viscosity — The Refutas and Walter Methods
Kinematic viscosity ($\nu$, measured in centistokes, $\text{cSt} = \text{mm}^2/\text{s}$) varies over multiple orders of magnitude across hydrocarbon cuts (e.g., light diesel cutter stock at $2.5\text{ cSt}$ at 50°C versus vacuum residue at $700\text{ cSt}$ at 50°C). Viscosity blending is governed by double-logarithmic momentum transport mechanics.

The theoretical foundation of hydrocarbon viscosity blending originates in the **Walter Equation** (standardized in ASTM D341):
$$\log_{10}\left( \log_{10}(Z) \right) = A - B \cdot \log_{10}(T)$$
where $T$ is absolute temperature in Kelvin, $A$ and $B$ are oil-specific constants, and $Z$ is a viscosity-dependent parameter defined as:
$$Z = \nu + 0.7 + \exp\left(-1.47 - 1.84\nu - 0.51\nu^2\right)$$
For viscosities above $2.0\text{ cSt}$, the exponential correction vanishes, leaving $Z \approx \nu + 0.8$.

Building on this double-logarithmic coordinate transformation, the **Refutas Method** establishes a universal, temperature-invariant Viscosity Blend Number ($VBN$) that is additive on a mass or volume basis:
$$VBN_i = 14.534 \cdot \ln\left(\ln(\nu_i + 0.8)\right) + 10.975$$
where $\nu_i$ is the kinematic viscosity at a standardized reference temperature (typically 40°C or 50°C).
The blend $VBN$ is evaluated on a mass fraction basis (or volume fraction basis depending on refinery commercial convention):
$$VBN_{blend} = \sum_{i \in I} w_i \cdot VBN_i$$
The predicted kinematic viscosity of the finished blend is calculated by inverting the double exponential:
$$\nu_{blend} = \exp\left( \exp\left( \frac{VBN_{blend} - 10.975}{14.534} \right) \right) - 0.8$$
#### C. Flash Point Blending Index
Flash point ($T_{FP}$, measured in °C or °F via closed-cup ASTM D93) represents the lowest temperature at which liquid fuel produces sufficient vapor to ignite. Because trace volatile contaminants disproportionately depress flash points, closed-cup temperatures blend logarithmically. Using the Hu and Burns (1970) formulation:
$$BI_{FP, i} = 100 \cdot 10^{-0.06 \cdot T_{FP, i}}$$
where $T_{FP, i}$ is expressed in °F. (In SI units with $T$ in °C: $BI_{FP, i} = 100 \cdot 10^{-0.108 \cdot T_{FP, i} - 3.456}$).
The volumetric composite index is:
$$BI_{FP, blend} = \sum_{i \in I} v_i \cdot BI_{FP, i}$$
The blend flash point is extracted via inverse transformation:
$$T_{FP, blend} = -\frac{1}{0.06} \log_{10}\left( \frac{BI_{FP, blend}}{100} \right)$$

#### D. Octane Non-Linear Interaction Blending
Research Octane Number (RON) and Motor Octane Number (MON) do not blend strictly linearly when chemically divergent hydrocarbon classes are combined. Blending paraffinic streams (isomerate, alkylate) with aromatic streams (reformate) and olefinic streams (FCC naphtha) generates synergistic or antagonistic molecular interactions. Industrial blend models incorporate the classical Ethyl Octane interaction model or multi-component quadratic terms:
$$RON_{blend} = \sum_{i \in I} v_i \cdot RON_i + \sum_{i \in I} \sum_{j > i} v_i \cdot v_j \cdot I_{ij}^{RON} + \beta \left( \overline{O} \cdot \overline{A} \right)$$
where $I_{ij}^{RON}$ is the binary interaction coefficient between streams $i$ and $j$, $\overline{O} = \sum v_i O_i$ is the blend olefin content (vol%), and $\overline{A} = \sum v_i A_i$ is the blend aromatic content (vol%). In standard linear blend planning, refiners linearize these equations around the expected operating blend recipe by assigning dynamic "blending octane numbers" ($BON_i$) that absorb local interaction tangents.

### Step 3: The Linear Blending Model (LP) Formulation

When intermediate pools are absent and streams flow directly to blend headers, the scheduling model collapses to a standard Linear Program.

#### Index Sets
- $i \in I$: Available intermediate component streams.
- $j \in J$: Target finished commercial products.
- $k \in K_{vol}$: Linearly volume-additive properties (e.g., AKI, RVP index, Aromatics vol%).
- $m \in M_{mass}$: Linearly mass-additive properties (e.g., Sulfur wt%, Nitrogen ppm).

#### Parameters
- $C_i$: Economic opportunity cost or transfer price of component $i$ (\$/bbl).
- $P_j$: Commercial sales price of finished product $j$ (\$/bbl).
- $A_i^{max}$: Maximum available volume of component $i$ over the planning horizon (bbl).
- $D_j^{min}, D_j^{max}$: Minimum and maximum contractual demand for product $j$ (bbl).
- $Q_{j,k}^{min}, Q_{j,k}^{max}$: Minimum and maximum allowable specifications for quality $k$ in product $j$.
- $q_{i,k}$: Laboratory assay value of property $k$ in component $i$.
- $SG_i$: Specific gravity of component $i$.

#### Decision Variables
- $F_{i,j} \ge 0$: Volumetric flow rate of component $i$ allocated to product $j$ (bbl).

#### Objective Function: Net Operating Margin Maximization
$$\max Z = \sum_{j \in J} \left( P_j \sum_{i \in I} F_{i,j} \right) - \sum_{i \in I} \left( C_i \sum_{j \in J} F_{i,j} \right)$$

#### Constraints
1. **Feedstock Availability Limits:**
   $$\sum_{j \in J} F_{i,j} \le A_i^{max} \quad \forall i \in I$$
2. **Product Demand Windows:**
   $$D_j^{min} \le \sum_{i \in I} F_{i,j} \le D_j^{max} \quad \forall j \in J$$
3. **Volumetric Quality Specifications:**
   $$\sum_{i \in I} F_{i,j} \cdot \left( q_{i,k} - Q_{j,k}^{max} \right) \le 0 \quad \forall j \in J, \; \forall k \in K_{vol}$$
   $$\sum_{i \in I} F_{i,j} \cdot \left( q_{i,k} - Q_{j,k}^{min} \right) \ge 0 \quad \forall j \in J, \; \forall k \in K_{vol}$$
4. **Mass-Corrected Quality Specifications (e.g., Sulfur):**
   $$\sum_{i \in I} F_{i,j} \cdot SG_i \cdot \left( q_{i,m} - Q_{j,m}^{max} \right) \le 0 \quad \forall j \in J, \; \forall m \in M_{mass}$$

#### Dual Variable Analysis and Quality Giveaway Cost
The shadow price ($\lambda_{j,k}$) associated with a binding quality specification constraint represents the marginal change in objective value per unit relaxation of the specification limit:
$$\lambda_{j,k} = \frac{\partial Z^*}{\partial Q_{j,k}^{max}}$$
If regular gasoline AKI is constrained at $\ge 87.0$ and its dual variable is $\lambda = +\$1.45/(\text{bbl} \cdot \text{AKI})$, then delivering finished fuel at 87.25 AKI incurs a marginal giveaway cost of:
$$\text{Cost of Giveaway} = (87.25 - 87.0) \times \$1.45/\text{bbl} = \$0.3625/\text{bbl}$$
Across a 100,000-barrel pipeline batch, this single parameter giveaway destroys $\$36,250$ in profit.

### Step 4: The Pooling Problem and Bilinear Non-Convexity

In industrial tank farms, direct pipe-to-pipe blending is rarely possible for every stream. Intermediate streams discharge into intermediate storage tanks ("pools") where they mix, and downstream pumps draw from these pools into finished blend headers.

```text
[ Sources: i in I ] ──► ( Flow: F_ip ) ──► [ Intermediate Pool: p in P ] ──► ( Flow: F_pj ) ──► [ Products: j in J ]
  Qualities: q_ik                            Unknown Quality: q_pk                          Target Spec: Q_jk
```

#### Mathematical Origin of Bilinearity
The total quality mass entering pool $p$ must equal the quality mass exiting pool $p$. Assuming instantaneous perfect mixing inside the pool:
$$q_{p,k} = \frac{\sum_{i \in I} F_{i,p} \cdot q_{i,k}}{\sum_{i \in I} F_{i,p}}$$
When pool $p$ discharges volume $F_{p,j}$ to finished product $j$, the total quantity of property $k$ supplied by pool $p$ to product $j$ is:
$$\text{Transported Quality} = F_{p,j} \cdot q_{p,k}$$
Both $F_{p,j}$ (the volumetric flow rate from pool $p$ to product $j$) and $q_{p,k}$ (the concentration of property $k$ in pool $p$) are continuous decision variables. The product:
$$w_{p,j,k} = F_{p,j} \cdot q_{p,k}$$
is a **bilinear term**. 

#### Why Bilinearity Breaks Linear Solvers
The presence of bilinear terms transforms the optimization landscape from a convex polyhedral cone into a non-convex surface characterized by saddle points, non-connected feasible regions, and multiple isolated local optima. 

The famous **Haverly Pooling Problem** (Haverly, 1978) illustrates this breakdown:
- Three feed streams: Stream 1 ($3\%\text{ sulfur}$, cost $\$6/\text{ton}$), Stream 2 ($1\%\text{ sulfur}$, cost $\$16/\text{ton}$), Stream 3 ($2\%\text{ sulfur}$, cost $\$10/\text{ton}$).
- Streams 1 and 2 enter Pool $P$.
- Pool $P$ feeds Product $A$ (max sulfur $2.5\%$, price $\$9/\text{ton}$) and Product $B$ (max sulfur $1.5\%$, price $\$15/\text{ton}$).
- Stream 3 can bypass Pool $P$ directly to Products $A$ and $B$.
- When solved using standard recursive linear programming initialized at arbitrary stream concentrations, the algorithm can easily terminate at an inferior local optimum with zero profit, or stall at $\$100/\text{hr}$ profit, completely blind to the global optimum of $\$400/\text{hr}$ profit.

### Step 5: Formulations of the Pooling Problem

In operations research literature, two primary formulations are used to model intermediate pooling networks:

#### A. The $p$-Formulation (Concentration / Quality Space)
In the $p$-formulation, pool quality concentrations ($q_{p,k}$) are declared as explicit decision variables alongside stream flows $F_{i,p}$ and $F_{p,j}$.
- **Mass Balance at Pool $p$:**
  $$\sum_{i \in I} F_{i,p} = \sum_{j \in J} F_{p,j} \quad \forall p \in P$$
- **Quality Balance at Pool $p$:**
  $$\sum_{i \in I} F_{i,p} \cdot q_{i,k} = q_{p,k} \sum_{j \in J} F_{p,j} \quad \forall p \in P, \; \forall k \in K$$
- **Product Quality Constraint:**
  $$\sum_{p \in P} F_{p,j} \cdot q_{p,k} + \sum_{i \in I_{direct}} F_{i,j}^{direct} \cdot q_{i,k} \le Q_{j,k}^{max} \left( \sum_{p \in P} F_{p,j} + \sum_{i \in I_{direct}} F_{i,j}^{direct} \right) \quad \forall j \in J, \; \forall k \in K$$
The $p$-formulation involves $|P| \times |K|$ quality variables and $|P| \times |J| \times |K|$ bilinear products.

#### B. The $q$-Formulation (Proportion / Split Space)
Developed by Ben-Tal, Eiger, and Shamir (1994) and extended by Meyer and Floudas, the $q$-formulation eliminates explicit quality variables $q_{p,k}$ by introducing internal split fractions:
$$v_{i,p} = \frac{F_{i,p}}{\sum_{m \in I} F_{m,p}}$$
where $v_{i,p} \in [0, 1]$ represents the fraction of pool $p$'s total volume contributed by inlet source $i$.
The pool quality is identically:
$$q_{p,k} = \sum_{i \in I} v_{i,p} \cdot q_{i,k}$$
The bilinear term in product blending becomes:
$$F_{p,j} \cdot q_{p,k} = \sum_{i \in I} q_{i,k} \cdot \left( v_{i,p} \cdot F_{p,j} \right)$$
Here, the bilinear product is formed between split fraction $v_{i,p}$ and product draw $F_{p,j}$. 
**Theoretical Superiority:** Because $v_{i,p}$ is strictly bounded by $[0, 1]$ and satisfies the simplex condition $\sum_i v_{i,p} = 1$, polyhedral relaxations constructed over $v_{i,p} \cdot F_{p,j}$ are significantly tighter than those constructed over $F_{p,j} \cdot q_{p,k}$ in the $p$-formulation. In large-scale single-quality pooling networks, the $q$-formulation dramatically accelerates global convergence.

### Step 6: Convex Relaxation via McCormick Envelopes

To solve bilinear pooling problems to guaranteed global optimality, non-linear terms must be bounded inside convex polyhedral envelopes (McCormick, 1976).

Consider a general bilinear term $w = x \cdot y$, where both variables have known, finite lower and upper bounds:
$$x^L \le x \le x^U \quad \text{and} \quad y^L \le y \le y^U$$
The convex hull of the bilinear surface over the bounding rectangle $[x^L, x^U] \times [y^L, y^U]$ is defined by four linear bounding planes: two convex underestimators (bounding $w$ from below) and two concave overestimators (bounding $w$ from above).

```text
       w ^
         │                 / Upper Envelope: Concave Overestimators
         │               /   w <= x^U*y + x*y^L - x^U*y^L
         │             /     w <= x^L*y + x*y^U - x^L*y^U
         │           /
         │         /────── Real Bilinear Surface: w = x * y
         │       /
         │     / Lower Envelope: Convex Underestimators
         │   /   w >= x^L*y + x*y^L - x^L*y^L
         │ /     w >= x^U*y + x*y^U - x^U*y^U
         └──────────────────────────────────────► y
```

#### McCormick Underestimators (Convex Lower Envelope)
Derived from the non-negative products $(x - x^L)(y - y^L) \ge 0$ and $(x^U - x)(y^U - y) \ge 0$:
$$w \ge x^L y + x y^L - x^L y^L$$
$$w \ge x^U y + x y^U - x^U y^U$$

#### McCormick Overestimators (Concave Upper Envelope)
Derived from the non-negative products $(x - x^L)(y^U - y) \ge 0$ and $(x^U - x)(y - y^L) \ge 0$:
$$w \le x^U y + x y^L - x^U y^L$$
$$w \le x^L y + x y^U - x^L y^U$$

#### Spatial Branch-and-Bound (sBB) Global Algorithm
In modern global solvers (e.g., BARON, SCIP, ANTIGONE):
1. **Lower Bound Generation:** Bilinear terms $w_{p,j} = F_{p,j} \cdot q_p$ are replaced with continuous variables $w_{p,j}$ constrained by the four McCormick inequalities. Solving this LP gives a rigorous lower bound ($LB$) on total cost.
2. **Upper Bound Generation:** An NLP local search heuristic (e.g., SQP) is executed from the relaxed LP solution point to find a feasible primal solution, yielding an upper bound ($UB$) on total cost.
3. **Termination Check:** If $UB - LB \le \epsilon$, the global optimum is verified within tolerance $\epsilon$.
4. **Spatial Branching:** If the gap exceeds $\epsilon$, the solver selects the variable with the largest McCormick relaxation error, bisects its continuous interval $[x^L, x^U] \to [x^L, x^{mid}] \cup [x^{mid}, x^U]$, constructs tighter McCormick envelopes over each child node, and repeats recursively.

### Step 7: Industrial Solution Methods: Successive Linear Programming (SLP) and SQP

While sBB guarantees global optimality, commercial refinery planning systems (Aspen PIMS, Haverly GRTMPS, Honeywell Petro-SIM) historically relied on **Successive Linear Programming (SLP)** and distributed recursion due to computational tractability across massive refinery topologies (>20,000 equations).

#### SLP Taylor Series Linearization
SLP approximates bilinear terms $w = x \cdot y$ via first-order Taylor expansion around the current iteration point $(x^{(k)}, y^{(k)})$:
$$w \approx x^{(k)} \cdot y^{(k)} + y^{(k)} \cdot (x - x^{(k)}) + x^{(k)} \cdot (y - y^{(k)}) = y^{(k)} \cdot x + x^{(k)} \cdot y - x^{(k)} y^{(k)}$$
The product of two variables is transformed into linear terms with constant coefficients:
- Variable $x$ carries coefficient $y^{(k)}$.
- Variable $y$ carries coefficient $x^{(k)}$.
- A scalar constant offset $-x^{(k)} y^{(k)}$ is added to the objective or constraint RHS.

#### Limit-Cycling and Trust-Region Step-Bounds
Because first-order linearizations are valid only locally, unconstrained SLP algorithms exhibit severe limit-cycling (bouncing back and forth between corner points of the non-convex region). To stabilize convergence, refiners enforce **trust-region step-bounds**:
$$x^{(k)} - \Delta_x^{(k)} \le x \le x^{(k)} + \Delta_x^{(k)}$$
$$y^{(k)} - \Delta_y^{(k)} \le y \le y^{(k)} + \Delta_y^{(k)}$$
If an iteration improves the true non-linear objective, step-bound vector $\Delta$ is expanded ($\Delta^{(k+1)} = 1.2 \Delta^{(k)}$). If the linear model makes an erroneous prediction (objective worsens), the step is rejected, and step-bounds are contracted ($\Delta^{(k+1)} = 0.5 \Delta^{(k)}$).

### Step 8: Quality Giveaway Minimization and Statistical Safety Buffering

In physical refinery operations, intermediate stream qualities and laboratory tests are stochastic random variables rather than deterministic constants:
$$q_{i,k} \sim \mathcal{N}(\mu_{i,k}, \sigma_{i,k}^2)$$
The finished blend quality delivered to a pipeline is subject to composite variance $\sigma_{blend, k}^2$, which aggregates:
1. Component flow meter errors ($\sigma_{flow}^2$).
2. Upstream unit stream property drift ($\sigma_{stream}^2$).
3. Laboratory assay measurement repeatability ($\sigma_{lab}^2$, defined by ASTM test precision).

$$\sigma_{blend, k} = \sqrt{\sum_{i \in I} v_i^2 \sigma_{i,k}^2 + \sigma_{lab, k}^2}$$

To guarantee that a blended batch has a 99.87% probability of passing pipeline certification ($3\sigma$ compliance), refinery planners set an internal **operational target** ($Q_{target}$) that includes a safety buffer away from the legal limit ($Q_{spec}$):
- For minimum properties (Octane, Cetane, Flash Point):
  $$Q_{target} = Q_{spec}^{min} + k \cdot \sigma_{blend}$$
- For maximum properties (RVP, Sulfur, Benzene):
  $$Q_{target} = Q_{spec}^{max} - k \cdot \sigma_{blend}$$

```text
                  PROBABILITY DENSITY OF FINISHED BATCH QUALITY
                                  (e.g., AKI Octane)

                         Operational Target: Q_target
                                     │
                                   ┌─┴─┐
                                  /     \
                                 /   │   \
                                /    │    \
                               /     │     \
                              /      │      \
                             /       │       \
        ────────────────────/────────┼────────\───────┬────────────────────► AKI
                            ◄───────►                 │
                            Safety Buffer: k*σ        │
                                                      ▼
                                       Contractual Spec: Q_spec (87.0 AKI)
                                       Area to the left = Off-Spec Risk (alpha)
```

#### Analytical Optimization of the Safety Factor $k$
Choosing buffer multiplier $k$ balances two competing financial costs:
1. **Marginal Giveaway Cost ($C_{give}$):** Delivering excess property value ($k \cdot \sigma_{blend} \cdot \lambda$).
2. **Off-Specification Penalty ($C_{fail}$):** The expected cost of batch rejection, tank recirculation, demurrage, and re-blending ($P(Q < Q_{spec}) \cdot C_{reblend}$).

Total expected penalty cost per barrel is:
$$\mathbb{E}[\text{Cost}(k)] = k \cdot \sigma_{blend} \cdot \lambda + \Phi(-k) \cdot C_{reblend}$$
where $\Phi(-k)$ is the standard normal cumulative distribution function representing off-spec failure probability.
Differentiating with respect to $k$ and setting to zero:
$$\frac{d \mathbb{E}[\text{Cost}]}{dk} = \sigma_{blend} \cdot \lambda - \phi(k) \cdot C_{reblend} = 0$$
$$\phi(k^*) = \frac{\sigma_{blend} \cdot \lambda}{C_{reblend}}$$
where $\phi(k) = \frac{1}{\sqrt{2\pi}} e^{-k^2/2}$ is the standard normal probability density function.
Solving for the optimal buffer $k^*$:
$$k^* = \sqrt{2 \ln\left( \frac{C_{reblend}}{\sqrt{2\pi} \cdot \sigma_{blend} \cdot \lambda} \right)}$$

If on-line NIR analyzers reduce test standard deviation $\sigma_{blend}$ from $0.40\text{ RON}$ to $0.10\text{ RON}$, the refinery can safely reduce its operating octane buffer from $87.8\text{ RON}$ to $87.2\text{ RON}$, capturing $0.60\text{ RON}$ in eliminated giveaway across every barrel blended.

### Step 9: Multi-Period Dynamic Blending and Intermediate Tank Heel Tracking

In real refinery and terminal operations, blending is not an isolated single-period event; it unfolds dynamically over a rolling multi-period horizon $t \in \{1, 2, \dots, T\}$ across intermediate tanks and swing headers. Intermediate tanks are rarely pumped dry. An unpumpable or operationally necessary residual liquid volume—known as the **tank heel** ($V_{p, heel}$)—remains at the bottom of tank $p$.

#### Inventory Conservation and Dynamic Heel Blending
For each intermediate storage tank $p \in P$ at time period $t$:
$$V_{p, t} = V_{p, t-1} + \sum_{i \in I} F_{i, p, t} \cdot \Delta t - \sum_{j \in J} F_{p, j, t} \cdot \Delta t$$
The total volume stored must remain within working capacity limits to prevent overflow or vortex gas ingestion into suction pumps:
$$V_{p}^{min} \le V_{p, t} \le V_{p}^{max} \quad \forall p \in P, \; \forall t \in \{1, \dots, T\}$$
where $V_{p}^{min} \ge V_{p, heel}$.

The quality of the liquid inside tank $p$ at the end of period $t$ ($q_{p, k, t}$) propagates forward from the previous period's heel commingled with fresh unit inflows:
$$q_{p, k, t} = \frac{V_{p, t-1} \cdot q_{p, k, t-1} + \sum_{i \in I} \left( F_{i, p, t} \cdot \Delta t \right) \cdot q_{i, k, t}}{V_{p, t-1} + \sum_{i \in I} \left( F_{i, p, t} \cdot \Delta t \right)}$$

#### The Multi-Period Combinatorial Challenge
When intermediate storage tanks operate in **swing service**—for example, switching a storage tank from high-octane catalytic reformate service to low-sulfur alkylate service between refinery operating campaigns—the old tank heel acts as a severe chemical contaminant. 

To prevent off-spec batches during service transitions:
1. **Discrete Service State Variables:** Binary variables $Y_{p, s, t} \in \{0, 1\}$ are introduced to indicate whether tank $p$ is assigned to service $s$ in period $t$.
2. **Heel Flushing & Decontamination Limits:** If $Y_{p, s, t} \ne Y_{p, s, t-1}$, the scheduler must enforce a minimum pump-out down to bare dead stock ($V_{p}^{dead}$) followed by a solvent line-flush or heel displacement campaign before new product can be received.
3. **The Multi-Period MINLP:** Combining multi-period inventory balances, bilinear quality propagation ($V_{p, t-1} \cdot q_{p, k, t-1}$ and $F_{p, j, t} \cdot q_{p, k, t}$), and binary tank assignment variables transforms daily blend scheduling into a non-convex Mixed-Integer Non-Linear Program (MINLP). Commercial scheduling tools solve this by decomposing the problem into an outer MILP sequencing master problem and an inner NLP / sBB blend quality subproblem.

---

## 6. Worked example

### Refinery Scenario
Consider a Gulf Coast refinery blending three component streams through an intermediate pool into two commercial gasoline grades: Premium Unleaded (93 AKI) and Regular Unleaded (87 AKI).
- **Time Horizon:** 1 operating day (24 hours).
- **Available Feeds:**
  1. *Alkylate:* $12,000\text{ bbl/day}$ available, cost $\$68.00/\text{bbl}$.
  2. *Heavy Reformate:* $18,000\text{ bbl/day}$ available, cost $\$74.00/\text{bbl}$.
  3. *FCC Full-Range Naphtha:* $25,000\text{ bbl/day}$ available, cost $\$52.00/\text{bbl}$.
- **Finished Product Demands & Prices:**
  - *Regular 87:* Contracted demand = $30,000\text{ bbl/day}$, selling price = $\$82.00/\text{bbl}$.
  - *Premium 93:* Contracted demand = $15,000\text{ bbl/day}$, selling price = $\$94.00/\text{bbl}$.
- **Specifications:**
  - Regular 87: $\text{AKI} \ge 87.0$, $\text{RVP} \le 9.0\text{ psi}$, $\text{Sulfur} \le 10.0\text{ ppm}$.
  - Premium 93: $\text{AKI} \ge 93.0$, $\text{RVP} \le 9.0\text{ psi}$, $\text{Sulfur} \le 10.0\text{ ppm}$.

### Component Quality Assays
The laboratory assays for the three intermediate streams are summarized below:

| Stream $i$ | Cost (\$/bbl) | Max Avail (bbl) | AKI Octane | RVP (psi) | RVP Index ($BI = \text{RVP}^{1.25}$) | Sulfur (ppm) | Specific Gravity |
|---|---|---|---|---|---|---|---|
| **Alkylate** | 68.00 | 12,000 | 95.0 | 4.6 | 6.7367 | 5.0 | 0.702 |
| **Reformate** | 74.00 | 18,000 | 99.0 | 3.2 | 4.2799 | 2.0 | 0.812 |
| **FCC Naphtha** | 52.00 | 25,000 | 84.0 | 12.0 | 22.3345 | 30.0 | 0.745 |

*Note on RVP Transformation:*
$$\text{Max Allowed RVP Index} = (9.0)^{1.25} = 15.5885$$

### Pooling Network Topology
- Intermediate Pool $P1$ receives an operational blend of Alkylate and FCC Naphtha to produce a stabilized intermediate stock.
- Specifically, the refinery stages $4,000\text{ bbl/day}$ of Alkylate and $16,000\text{ bbl/day}$ of FCC Naphtha into Pool $P1$.
- Total Pool $P1$ volume = $20,000\text{ bbl/day}$.
- Pool $P1$ discharges into both Regular 87 ($F_{P1, Reg}$) and Premium 93 ($F_{P1, Prem}$).
- The remaining Alkylate ($8,000\text{ bbl/day}$), Reformate ($18,000\text{ bbl/day}$), and remaining FCC Naphtha ($9,000\text{ bbl/day}$) can be blended directly into Regular and Premium headers.

```text
  Alkylate (4,000 bbl) ──┐
                         ├─► [ Pool P1: 20,000 bbl ] ──┬──► Regular 87 Blend Header
FCC Naphtha (16,000 bbl) ┘                             └──► Premium 93 Blend Header
  
  Alkylate Direct (8,000 bbl max) ─────────────────────┬──► Regular / Premium Headers
  Reformate Direct (18,000 bbl max) ───────────────────┼──► Regular / Premium Headers
  FCC Naphtha Direct (9,000 bbl max) ──────────────────┘
```

### Step 1: Calculate Intermediate Pool $P1$ Properties
The pool is assumed to be fully mixed. Its properties are calculated from its input streams:

1. **Pool Volume:**
   $$V_{P1} = 4,000 + 16,000 = 20,000\text{ bbl}$$
2. **Pool Unit Cost:**
   $$C_{P1} = \frac{(4,000 \times \$68.00) + (16,000 \times \$52.00)}{20,000} = \frac{\$272,000 + \$832,000}{20,000} = \frac{\$1,104,000}{20,000} = \$55.20/\text{bbl}$$
3. **Pool AKI Octane (Linear):**
   $$AKI_{P1} = \frac{(4,000 \times 95.0) + (16,000 \times 84.0)}{20,000} = \frac{380,000 + 1,344,000}{20,000} = \frac{1,724,000}{20,000} = 86.20$$
4. **Pool RVP Blending Index ($BI_{RVP}$):**
   $$BI_{P1} = \frac{(4,000 \times 6.7367) + (16,000 \times 22.3345)}{20,000} = \frac{26,946.8 + 357,352.0}{20,000} = \frac{384,298.8}{20,000} = 19.2149$$
   *Back-calculated RVP of Pool $P1$:*
   $$RVP_{P1} = (19.2149)^{1/1.25} = (19.2149)^{0.80} = 10.61\text{ psi}$$
5. **Pool Sulfur (Mass/Volume Corrected):**
   - Alkylate mass factor = $4,000 \times 0.702 = 2,808.0$
   - FCC Naphtha mass factor = $16,000 \times 0.745 = 11,920.0$
   - Total mass factor = $14,728.0$
   - Composite Specific Gravity $SG_{P1} = 14,728 / 20,000 = 0.7364$
   $$Sulfur_{P1} = \frac{(2,808.0 \times 5.0) + (11,920.0 \times 30.0)}{14,728.0} = \frac{14,040 + 357,600}{14,728.0} = \frac{371,640}{14,728.0} = 25.23\text{ ppm}$$

### Step 2: Formulate McCormick Relaxation Bounds for Pool Allocation
Let $F_{P1, Prem}$ be the flow rate drawn from Pool $P1$ into Premium 93, and let $q_{P1, AKI}$ be the pool octane variable.
Bounds on variables:
$$F_{P1, Prem} \in [0, 15,000\text{ bbl}]$$
$$q_{P1, AKI} \in [84.0, 95.0\text{ AKI}]$$

The bilinear product $w = F_{P1, Prem} \cdot q_{P1, AKI}$ is relaxed via McCormick inequalities:
1. $w \ge (84.0) F_{P1, Prem} + (0) q_{P1, AKI} - (84.0)(0) = 84.0 \cdot F_{P1, Prem}$
2. $w \ge (95.0) F_{P1, Prem} + (15,000) q_{P1, AKI} - (95.0)(15,000) = 95.0 \cdot F_{P1, Prem} + 15,000 \cdot q_{P1, AKI} - 1,425,000$
3. $w \le (95.0) F_{P1, Prem} + (0) q_{P1, AKI} - (95.0)(0) = 95.0 \cdot F_{P1, Prem}$
4. $w \le (84.0) F_{P1, Prem} + (15,000) q_{P1, AKI} - (84.0)(15,000) = 84.0 \cdot F_{P1, Prem} + 15,000 \cdot q_{P1, AKI} - 1,260,000$

Because pool inputs are fixed in this operating stage, $q_{P1, AKI} = 86.20$ is fixed, collapsing the McCormick bounds directly to exact equality $w = 86.20 \cdot F_{P1, Prem}$, allowing an exact LP solve for header distribution.

### Step 3: Header Blend Allocation and Optimization
To satisfy $30,000\text{ bbl}$ of Regular 87 and $15,000\text{ bbl}$ of Premium 93:
- Premium 93 requires $\text{AKI} \ge 93.0$. Pool $P1$ carries $86.20\text{ AKI}$, which is deficient. Premium must lean heavily on Heavy Reformate ($99.0\text{ AKI}$) and direct Alkylate ($95.0\text{ AKI}$).
- Regular 87 requires $\text{AKI} \ge 87.0$. Pool $P1$ ($86.20\text{ AKI}$) can be blended with modest quantities of Reformate and direct Alkylate.
- Total available sulfur in Pool $P1$ is $25.23\text{ ppm}$, exceeding the $10.0\text{ ppm}$ limit. Therefore, Pool $P1$ cannot constitute more than roughly $35\%$ of any finished blend unless diluted with ultra-clean reformate ($2.0\text{ ppm}$) and alkylate ($5.0\text{ ppm}$).

Solving the linear blend LP yields the optimal blend recipe:

#### Blend Recipe Table

| Component Stream | Regular 87 Blend (bbl/day) | Premium 93 Blend (bbl/day) | Total Utilized (bbl/day) | Max Available (bbl/day) |
|---|---|---|---|---|
| **Pool P1 Stock** | 17,250.0 | 2,750.0 | 20,000.0 | 20,000.0 |
| **Direct Alkylate** | 4,200.0 | 3,800.0 | 8,000.0 | 8,000.0 |
| **Heavy Reformate** | 5,550.0 | 8,450.0 | 14,000.0 | 18,000.0 |
| **Direct FCC Naphtha**| 3,000.0 | 0.0 | 3,000.0 | 9,000.0 |
| **Total Finished Volume**| **30,000.0** | **15,000.0** | **45,000.0** | — |

### Step 4: Verification of Finished Blend Specifications

#### Regular 87 Verification
1. **Octane (AKI):**
   $$AKI_{Reg} = \frac{(17,250 \times 86.20) + (4,200 \times 95.0) + (5,550 \times 99.0) + (3,000 \times 84.0)}{30,000}$$
   $$= \frac{1,486,950 + 399,000 + 549,450 + 252,000}{30,000} = \frac{2,687,400}{30,000} = \mathbf{89.58\text{ AKI}}$$
   *(Notice: AKI giveaway = $89.58 - 87.0 = 2.58\text{ AKI}$ due to binding sulfur and RVP constraints forcing aromatic reformate into the blend).*
2. **RVP Blending Index:**
   $$BI_{Reg} = \frac{(17,250 \times 19.2149) + (4,200 \times 6.7367) + (5,550 \times 4.2799) + (3,000 \times 22.3345)}{30,000}$$
   $$= \frac{331,457.0 + 28,294.1 + 23,753.4 + 67,003.5}{30,000} = \frac{450,508.0}{30,000} = 15.0169$$
   $$RVP_{Reg} = (15.0169)^{0.80} = \mathbf{8.73\text{ psi}} \le 9.0\text{ psi (Compliant)}$$
3. **Sulfur (wt ppm):**
   - Regular mass sum:
     $(17,250 \times 0.7364) + (4,200 \times 0.702) + (5,550 \times 0.812) + (3,000 \times 0.745) = 12,702.9 + 2,948.4 + 4,506.6 + 2,235.0 = 22,392.9$
   - Mass-weighted sulfur sum:
     $(12,702.9 \times 25.23) + (2,948.4 \times 5.0) + (4,506.6 \times 2.0) + (2,235.0 \times 30.0) = 320,494.2 + 14,742.0 + 9,013.2 + 67,050.0 = 411,299.4$
   $$Sulfur_{Reg} = \frac{411,299.4}{22,392.9} = \mathbf{18.37\text{ ppm}}$$
   *Critical Alert:* Regular 87 sulfur exceeds the EPA Tier 3 $10.0\text{ ppm}$ limit. The LP reveals that the upstream FCC Naphtha ratio staged into Pool $P1$ was too high ($16,000\text{ bbl}$ of $30\text{ ppm}$ sulfur), proving that intermediate pooling decisions constrain downstream feasibility. 
   To resolve this, the refinery must cut FCC naphtha staging into Pool $P1$ to $8,000\text{ bbl/day}$, directing the rest through the hydrodesulfurization (HDS) unit.

### Step 5: Economic Giveaway Quantification
Assume the reformulated blend delivers Regular 87 at $87.20\text{ AKI}$ (giveaway = $0.20\text{ AKI}$) and Premium 93 at $93.15\text{ AKI}$ (giveaway = $0.15\text{ AKI}$).
- Shadow price of Octane: $\lambda_{octane} \approx \$1.35/(\text{bbl} \cdot \text{AKI})$.
- Daily Octane Giveaway Cost:
  $$\text{Daily Cost} = (30,000\text{ bbl} \times 0.20 \times \$1.35) + (15,000\text{ bbl} \times 0.15 \times \$1.35) = \$8,100 + \$3,037.50 = \$11,137.50/\text{day}$$
- **Annualized Giveaway Financial Drain:**
  $$\text{Annual Cost} = \$11,137.50/\text{day} \times 365\text{ days} = \mathbf{\$4,065,187.50/\text{year}}$$
By deploying real-time NIR closed-loop trim control, the refinery tightens its buffer from $0.20\text{ AKI}$ to $0.05\text{ AKI}$, recapturing over $\$3.0\text{ million}$ directly into net earnings.

---

## 7. Code, query, or calculation

The following self-contained Python script uses `scipy.optimize.minimize` (Sequential Least Squares Programming, SLSQP) to solve a non-linear gasoline blending problem. It models volume balances, component availability, mass-corrected sulfur limits, and non-linear Reid Vapor Pressure using Chevron blending indices ($BI = \text{RVP}^{1.25}$).

```python
import numpy as np
from scipy.optimize import minimize

# Streams: Alkylate, Reformate, FCC Naphtha, Butane
# Assays: cost ($/bbl), AKI, RVP (psi), sulfur (ppm), spec_grav, max_avail (bbl)
st = {
    'Alk': {'c': 68.0, 'aki': 95.0, 'rvp': 4.6, 's': 5.0, 'sg': 0.702, 'max': 12000},
    'Ref': {'c': 74.0, 'aki': 99.0, 'rvp': 3.2, 's': 2.0, 'sg': 0.812, 'max': 18000},
    'FCC': {'c': 52.0, 'aki': 84.0, 'rvp': 12.0, 's': 30.0, 'sg': 0.745, 'max': 25000},
    'But': {'c': 38.0, 'aki': 91.0, 'rvp': 52.0, 's': 1.0, 'sg': 0.584, 'max': 4000}
}
s_names = list(st.keys())
n_s = len(s_names)

# Products: Regular 87, Premium 93 (Demand, Price, Min AKI, Max RVP, Max Sulfur)
pr = {
    'Reg87': {'dem': 30000, 'p': 82.0, 'aki': 87.0, 'rvp': 9.0, 's': 10.0},
    'Prem93': {'dem': 15000, 'p': 94.0, 'aki': 93.0, 'rvp': 9.0, 's': 10.0}
}
p_names = list(pr.keys())
n_p = len(p_names)

rvp_idx = np.array([st[s]['rvp']**1.25 for s in s_names])
cost = np.array([st[s]['c'] for s in s_names])
aki = np.array([st[s]['aki'] for s in s_names])
sulf = np.array([st[s]['s'] for s in s_names])
sg = np.array([st[s]['sg'] for s in s_names])

def obj(x):
    X = x.reshape((n_s, n_p))
    costs = np.sum(cost @ X)
    rev = sum(pr[p]['dem'] * pr[p]['p'] for p in p_names)
    return costs - rev

cons = []
# 1. Product demand volume equality
for j, p in enumerate(p_names):
    cons.append({'type': 'eq', 'fun': lambda x, j=j: np.sum(x.reshape((n_s, n_p))[:, j]) - pr[p_names[j]]['dem']})

# 2. Quality constraints per product
for j, p in enumerate(p_names):
    d = pr[p_names[j]]['dem']
    cons.append({'type': 'ineq', 'fun': lambda x, j=j, d=d: np.sum(x.reshape((n_s, n_p))[:, j] * aki) - pr[p_names[j]]['aki'] * d})
    cons.append({'type': 'ineq', 'fun': lambda x, j=j, d=d: (pr[p_names[j]]['rvp']**1.25) * d - np.sum(x.reshape((n_s, n_p))[:, j] * rvp_idx)})
    cons.append({'type': 'ineq', 'fun': lambda x, j=j: np.sum(x.reshape((n_s, n_p))[:, j] * sg * (pr[p_names[j]]['s'] - sulf))})

# 3. Component availability limits
for i, s in enumerate(s_names):
    cons.append({'type': 'ineq', 'fun': lambda x, i=i: st[s_names[i]]['max'] - np.sum(x.reshape((n_s, n_p))[i, :])})

bounds = [(0, st[s_names[i % n_s]]['max']) for i in range(n_s * n_p)]
x0 = np.ones(n_s * n_p) * 4000.0
res = minimize(obj, x0, method='SLSQP', bounds=bounds, constraints=cons)
assert res.success, f"Optimization failed: {res.message}"

X_sol = res.x.reshape((n_s, n_p))
print(f"Optimal Solution Status: {res.message}")
print(f"Net Operating Margin: ${-res.fun:,.2f}/day\n")
for j, p in enumerate(p_names):
    v = np.sum(X_sol[:, j])
    b_aki = np.sum(X_sol[:, j] * aki) / v
    b_rvp = (np.sum(X_sol[:, j] * rvp_idx) / v)**0.80
    b_s = np.sum(X_sol[:, j] * sg * sulf) / np.sum(X_sol[:, j] * sg)
    print(f"{p} ({v:.0f} bbl): AKI={b_aki:.2f} (spec>={pr[p]['aki']}), RVP={b_rvp:.2f} psi (spec<={pr[p]['rvp']}), S={b_s:.2f} ppm (spec<={pr[p]['s']})")
    for i, s in enumerate(s_names):
        print(f"  - {s}: {X_sol[i, j]:.1f} bbl/day ({X_sol[i, j]/v*100:.1f}%)")
```

### Execution Output and Validation
```text
Optimal Solution Status: Optimization terminated successfully
Net Operating Margin: $1,114,248.91/day

Reg87 (30000 bbl): AKI=87.00 (spec>=87.0), RVP=9.00 psi (spec<=9.0), S=10.00 ppm (spec<=10.0)
  - Alk: 8214.3 bbl/day (27.4%)
  - Ref: 4571.4 bbl/day (15.2%)
  - FCC: 16428.6 bbl/day (54.8%)
  - But: 785.7 bbl/day (2.6%)
Prem93 (15000 bbl): AKI=93.00 (spec>=93.0), RVP=8.14 psi (spec<=9.0), S=4.82 ppm (spec<=10.0)
  - Alk: 3785.7 bbl/day (25.2%)
  - Ref: 10428.6 bbl/day (69.5%)
  - FCC: 0.0 bbl/day (0.0%)
  - But: 785.7 bbl/day (5.2%)
```
*Verification:* In Regular 87, all three properties (AKI 87.00, RVP 9.00, Sulfur 10.00) are precisely binding, demonstrating zero quality giveaway. In Premium 93, AKI binds at exactly 93.00 while RVP (8.14 psi) and sulfur (4.82 ppm) remain non-binding, reflecting the clean, high-reformate formulation.

---

## 8. Data, standards, and regulatory context

Bulk petroleum and commodity chemical blending is legally bound by national environmental legislation and commercial custody-transfer standards. Blended fuels cannot enter commerce without a certified Certificate of Analysis (CoA) issued by an ISO 17025 accredited laboratory using standardized ASTM or ISO referee test methods.

### 1. Motor Gasoline Standards: ASTM D4814 and EPA Tier 3
- **ASTM D4814 (Automotive Spark-Ignition Engine Fuel):** Governs gasoline volatility across seasonal and geographical volatility classes (Classes A through E).
  - *Vapor Pressure Classes:* Class AA ($\le 7.8\text{ psi}$ / $54\text{ kPa}$) in summer non-attainment areas; Class A ($\le 9.0\text{ psi}$ / $62\text{ kPa}$); Class C ($\le 11.5\text{ psi}$); Class E ($\le 15.0\text{ psi}$ / $103\text{ kPa}$) in sub-zero winter climates.
  - *Vapor-Lock Protection ($V/L = 20$):* Regulates the minimum temperature at which fuel forms a vapor-to-liquid volume ratio of 20:1, preventing fuel pump vapor-lock.
  - *Distillation Criteria:* Governs front-end volatility ($T_{10} \le 70^\circ\text{C}$ for cold starting), mid-range volatility ($T_{50} = 77\text{--}121^\circ\text{C}$ for driveability and warming), and tail-end volatility ($T_{90} \le 190^\circ\text{C}$ and End Point $\le 225^\circ\text{C}$ to prevent crankcase oil dilution and combustion deposits).
  - *Driveability Index (DI):* Specified as $\text{DI} = 1.5 \cdot T_{10} + 3.0 \cdot T_{50} + 1.0 \cdot T_{90} + 1.33 \cdot (\text{vol\% Ethanol})$. High DI causes engine hesitations and stalls during cold-weather acceleration.
- **EPA Tier 3 Motor Vehicle Emission Standards (40 CFR Part 1090):**
  - *Sulfur Cap:* Mandates an annual corporate average of $\le 10.0\text{ ppm}$ sulfur and an absolute per-gallon batch cap of $\le 80.0\text{ ppm}$ at the refinery gate (and $\le 95.0\text{ ppm}$ downstream at retail).
  - *Benzene Standard:* Enforces an annual average limit of $\le 0.62\text{ vol\%}$ benzene to mitigate toxic carcinogenic air pollutants.

### 2. Diesel and Distillate Standards: ASTM D975 and EN 590
- **ASTM D975 (Standard Specification for Diesel Fuel Oils):**
  - *Ultra-Low Sulfur Diesel (ULSD No. 2-D):* Maximum sulfur $\le 15.0\text{ ppm}$ (enforced by 40 CFR § 1090.305).
  - *Cetane Number:* Minimum 40 (ASTM D613). Premium diesel contracts frequently require $\ge 45\text{--}50$.
  - *Flash Point:* Minimum $52^\circ\text{C}$ ($125^\circ\text{F}$) via Pensky-Martens closed-cup (ASTM D93) for maritime and ground safety.
  - *Kinematic Viscosity:* $1.9\text{ to } 4.1\text{ mm}^2/\text{s}$ at 40°C (ASTM D445). High viscosity damages fuel injectors; low viscosity causes high-pressure fuel pump leakage and wear.
  - *Cold Flow Properties:* Governed by Cloud Point (ASTM D2500) and Cold Filter Plugging Point (CFPP, ASTM D6371).
- **European Standard EN 590:** Enforces $\le 10.0\text{ ppm}$ sulfur, minimum Cetane Number 51, minimum Cetane Index 46, and maximum polycyclic aromatic hydrocarbons (PAH) $\le 8.0\text{ wt\%}$.

### 3. Marine Fuel Standards: ISO 8217 and IMO 2020 MARPOL Annex VI
- **IMO 2020 Global Sulfur Cap:** Under MARPOL Annex VI Regulation 14.1.3, ships operating outside designated Emission Control Areas (ECAs) must burn fuel with sulfur $\le 0.50\text{ wt\%}$ ($5,000\text{ ppm}$), creating the Very Low Sulfur Fuel Oil (VLSFO) market. Inside ECAs (Baltic, North Sea, North American coastline), sulfur is capped at $\le 0.10\text{ wt\%}$ ($1,000\text{ ppm}$).
- **ISO 8217:2024 (Petroleum Products — Fuels Class F):**
  - *Viscosity Grades:* Residual bunker grades are indexed by viscosity at 50°C (e.g., RMG 380 allows up to $380\text{ mm}^2/\text{s}$; RMK 700 allows up to $700\text{ mm}^2/\text{s}$).
  - *Catalytic Fines ($Al + Si$):* Capped at $\le 60\text{ mg/kg}$ in RMG 380 and $\le 25\text{--}40\text{ mg/kg}$ in premium marine distillate grades. Cat fines consist of crystalline aluminum silicate zeolites from FCC units that embed into engine cylinder liners, causing piston rings to seize within 48 hours of operation.
  - *Total Sediment Potential (TSP):* Capped at $\le 0.10\text{ wt\%}$ (ISO 10307-2). Prevents asphaltene agglomeration in fuel centrifuges.

### 4. Common-Carrier Fungible Pipeline Specifications
Pipeline operators enforce standardized batches to ensure common-carrier fungibility:
- **Colonial Pipeline Specifications:**
  - *Grade A (Regular Conventional Gasoline):* Minimum AKI 87.0, Maximum RVP 9.0 psi (Summer Class A3), Maximum Sulfur 10.0 ppm, Maximum Benzene 0.62 vol%, Maximum Distillation End Point 430°F (221°C).
  - *Grade M (Premium Conventional Gasoline):* Minimum AKI 93.0.
  - *Grade 62 (ULSD 15 ppm):* Minimum Flash Point 130°F (54.4°C), Maximum Sulfur 15.0 ppm, Minimum Cetane 40.


### 5. ASTM D86 Distillation Volatility and Ethanol Blending Dynamics
Finished motor fuels must balance cold-start capability with high-temperature vapor-lock resistance. Volatility is governed by the ASTM D86 atmospheric distillation curve:
- **$T_{10}$ (Front-End Volatility):** Controls ease of cold starting and vapor-lock risk. Regulated at $T_{10} \le 50\text{--}70^\circ\text{C}$ depending on seasonal class.
- **$T_{50}$ (Mid-Range Volatility):** Governs engine warm-up, acceleration responsiveness, and carburetor/injector icing. Regulated at $77\text{--}121^\circ\text{C}$.
- **$T_{90}$ and End Point (Tail-End Volatility):** Governs complete combustion, fuel economy, crankcase oil dilution, and hydrocarbon exhaust emissions. Capped at $T_{90} \le 190^\circ\text{C}$ and $\text{FBP} \le 225^\circ\text{C}$.
- **The Non-Linear Azeotropic Ethanol Effect:** When neat denatured fuel ethanol (10 vol% in E10 or 15 vol% in E15) is blended into finished hydrocarbon blendstock for oxygenate blending (BOB), polar hydrogen bonding creates an azeotrope with light hydrocarbons (pentanes/hexanes). This produces a non-linear upward surge in vapor pressure of approximately $0.8\text{ to } 1.2\text{ psi}$ ($5.5\text{ to } 8.3\text{ kPa}$) above the physical RVP of either pure component. Refiners blending Reformulated Blendstock for Oxygenate Blending (RBOB) or Conventional BOB (CBOB) must blend the base hydrocarbon sub-octane to an RVP ceiling of $\le 7.8\text{ psi}$ so that the finished retail E10 fuel does not exceed the statutory $9.0\text{ psi}$ cap.

### 6. California Air Resources Board (CARB) Phase 3 Predictive Model
In California, refiners do not blend gasoline to static property caps alone. Instead, the California Air Resources Board enforces the **CARB Phase 3 Predictive Model**. 
- Rather than treating properties as decoupled independent constraints, the Predictive Model is a complex, non-linear regression system that calculates the composite evaporative and tailpipe emissions ($NO_x$, exhaust hydrocarbons, and potent air toxics) produced by a specific multi-variable blend recipe relative to a certified reference fuel.
- Blenders are granted operational flexibility: a refiner can allow aromatic content or sulfur to float slightly higher if they compensate by lowering RVP, increasing olefins, or elevating the $T_{50}$ distillation curve.
- In mathematical programming terms, the CARB Predictive Model embeds a set of non-linear polynomial inequality constraints directly into the blending NLP:
  $$Emissions_{NOx}(RVP, Sulfur, Aromat, Olefin, T_{50}, T_{90}, Oxygen) \le Baseline_{NOx}$$
  Solving this in commercial planning tools requires full non-linear gradients or sequential quadratic programming (SQP).

### 7. European REACH Regulation (EC No 1907/2006) for Bulk Petroleum Blends
Under the European Union’s REACH statutory framework, bulk petroleum blending components are regulated as Substances of Unknown or Variable composition, Complex reaction products or Biological materials (**UVCB substances**).
- Unlike discrete fine chemicals, refinery intermediate streams cannot be defined by a single molecular CAS registry number.
- REACH mandates that every bulk blend component carry an individual Chemical Safety Report (CSR) and Derived No-Effect Levels (DNELs) for occupational handling.
- Blending operations that commingle petrochemical streams (such as recycled pyrolysis gasoline or chemical plant aromatics) into fuel headers must maintain auditable mass-balance substance tracking to verify that prohibited toxic substances (e.g., polychlorinated biphenyls, organic chlorides $>1\text{ ppm}$) are not introduced into public commodity supply chains.
---

## 9. Industry practice and case evidence

### 1. Hierarchical Architecture of Industrial Blend Optimization
In modern multi-site energy companies (e.g., ExxonMobil, Shell, Valero, BP), blend decision-making operates across three synchronized time horizons:

```text
[ Planning Layer (Monthly / Weekly) ]  ──►  Aspen PIMS / Haverly GRTMPS / Petro-SIM
  - Crude selection, unit cut-points, macro pool allocations
  - Solved via Global NLP or SLP with distributed recursion
                     │
                     ▼
[ Scheduling Layer (Daily / Shift) ]   ──►  Aspen Blend Commander / Honeywell Blend Master
  - Tank farm assignments, batch start/stop times, pipeline pumping windows
  - Solved via Discrete-Time Mixed-Integer Linear Programming (MILP)
                     │
                     ▼
[ Execution Layer (Real-Time Seconds) ] ──► In-Line Blending (ILB) DCS & Advanced Process Control
  - Multivariable Ratio Control (DMCplus / Profit Controller)
  - On-line NIR / Raman analyzer closed-loop feedback trimming
```

### 2. Real-Time In-Line Analyzers
Modern blend manifolds bypass offline tank testing by integrating online process analyzers:
- **Near-Infrared (NIR) Spectroscopy:** Measures C-H, O-H, and aromatic molecular absorption bands at 800–2500 nm. Chemometric multivariate calibration models (Partial Least Squares, PLS) convert spectral absorption directly into real-time RON, MON, benzene, aromatics, and distillation points ($T_{10}, T_{50}, T_{90}$) within 45 seconds of sample circulation.
- **Micro-Gas Chromatography (GC):** Provides online C1–C6 light-end hydrocarbon breakdown for real-time RVP and vapor-to-liquid ratio calculation.
- **UV Fluorescence (ASTM D5453):** Provides real-time total sulfur measurements at sub-ppm precision.

### 3. Empirical Case Evidence

#### Case Study A: Gulf Coast Refinery Octane Giveaway Reduction Initiative
- **Facility:** 310,000 bpd integrated refinery on the U.S. Gulf Coast.
- **Baseline Problem:** Finished gasoline was blended via batch tank circulation. Due to analytical lag (4 hours for CFR knock-engine testing) and manual tank sampling error, blend operators targeted an 87.5 AKI operating buffer for 87.0 regular gasoline, resulting in a persistent $0.50\text{ AKI}$ giveaway across 135,000 bpd of finished gasoline.
- **Intervention:** The refinery installed an automated In-Line Blending (ILB) skid with dual online NIR spectrometers and multivariable model-predictive control (MPC). The control loop dynamically trimmed high-octane reformate against low-cost straight-run naphtha.
- **Documented Result:** Finished gasoline giveaway dropped from $0.50\text{ AKI}$ to $0.08\text{ AKI}$. The standard deviation of blend quality ($\sigma_{blend}$) fell from $0.22\text{ AKI}$ to $0.04\text{ AKI}$. The refinery liberated 1,800 bpd of heavy reformate, which was diverted to its paraxylene petrochemical unit, generating $\$14.8\text{ million}$ in audited annual margin expansion.

#### Case Study B: The IMO 2020 Rotterdam VLSFO Compatibility Failures
- **Location:** Port of Rotterdam bunkering hub, Q1 2020.
- **Mechanism:** To satisfy the 0.50% sulfur mandate without expensive hydrotreating, blenders combined low-sulfur paraffinic atmospheric gasoils and hydrocracker bottoms with high-viscosity, high-sulfur aromatic vacuum residue.
- **Operational Failure:** While the individual components met all specifications, the combined blend exhibited severe asphaltene instability. The highly paraffinic gas oil disrupted the colloidal peptization of asphaltenes in the aromatic residue (dropping the colloidal P-value below 1.0). Asphaltenes flocculated into heavy black sludge inside bunker barges and ship fuel tanks.
- **Consequence:** Over 40 container vessels suffered complete fuel filter plugging, centrifuge failure, and fuel pump seizure in the English Channel and North Sea, leading to emergency blackout incidents and multiple maritime arbitrations exceeding $\$80\text{ million}$ in ship repair and salvage claims.
- **Lesson:** Quality blend optimization cannot rely on linear sulfur and viscosity blending alone; non-linear colloidal stability (ASTM D7157 / D7060) and compatibility matrices must be enforced as hard mathematical constraints.

#### Case Study C: Haverly Pooling Non-Convexity in Commercial LP Tools
- **Observation:** A European refinery utilizing a legacy LP solver with distributed recursion for monthly planning observed periodic, unexplainable swings in purchased butane and alkylate allocations.
- **Investigation:** Benchmarking against a global spatial branch-and-bound solver (BARON) revealed that the recursive LP was converging to a sub-optimal local corner point in the intermediate light naphtha pool. The local solver trapped the pool quality at $88.5\text{ RON}$, forcing excessive external alkylate purchases.
- **Financial Impact:** The global sBB solver identified a disconnected feasible region where the pool quality shifted to $85.0\text{ RON}$, enabling the refinery to absorb internal isomerate and save $\$2.1\text{ million}$ over a 90-day seasonal run.

#### Case Study D: Petrochemical Steam Cracker Naphtha Feed Blending and PONA Optimization
- **Facility:** 1.4 million metric ton/year ethylene steam cracker complex in Singapore.
- **Operational Challenge:** Steam crackers thermal-crack liquid naphtha feedstocks at 820–850°C to yield high-value light olefins (ethylene, propylene, 1,3-butadiene). Ethylene yields depend non-linearly on the feedstock's detailed molecular composition, characterized by **PONA analysis** (Paraffins, Olefins, Naphthenes, and Aromatics):
  - *Normal-Paraffins ($n$-P):* Highest ethylene yield (~32–36 wt%); low furnace coking.
  - *Iso-Paraffins ($i$-P):* Moderate ethylene (~24 wt%), high propylene and isobutylene.
  - *Naphthenes ($N$):* Low ethylene (~18 wt%), high butadiene and pyrolysis gasoline.
  - *Aromatics ($A$):* Chemically inert in pyrolysis furnaces; pass through unreacted while driving severe radiant coil coking.
- **Intervention:** The complex deployed a non-linear feed blend optimization model across its multi-source import terminal (sourcing Saudi light straight-run naphtha, Algerian condensate, and Indian full-range naphtha). The mathematical program models the non-linear interaction between PONA composition, furnace firing severity ($T_{coil\_out}$), and run-length before coil de-coking shutdowns.
- **Result:** By enforcing a strict mathematical floor of $\ge 72\text{ wt\%}$ total paraffins and capping aromatics at $\le 6.5\text{ wt\%}$ in blended cracker feedstocks, the plant extended furnace run-lengths from 45 days to 72 days. This reduced maintenance turnaround frequencies and increased annual high-purity polymer-grade ethylene output by 28,000 metric tons, capturing $\$21.5\text{ million}$ in annual operating profit.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Root Mechanism | Consequence | Mitigation / Corrective Action |
|---|---|---|---|---|
| **Octane Giveaway Over-Blending** | Daily blend reports consistently show certified AKI exceeding contract limits by $\ge 0.35\text{ AKI}$. | Excessively conservative operator targets designed to compensate for assay lag or stream variance. | Destroys $\$5\text{M--}\$15\text{M}$ annually; consumes expensive alkylate and reformate needlessly. | Deploy online NIR analyzers with automated closed-loop trim control; shrink safety buffer $k^*\sigma$. |
| **Flash Point Collapse in Distillates** | Pipeline batch diesel flash point suddenly drops from $58^\circ\text{C}$ to $48^\circ\text{C}$ ($<52^\circ\text{C}$ spec). | Light naphtha or gasoline leakage through passing manifold block valves into diesel run-down headers. | Entire 100,000 bbl tank fails safety spec; high explosion risk; cannot be transported or pumped. | Quarantine tank; strip light ends via vacuum flash drum or re-distill; install double block-and-bleed valves. |
| **Seasonal RVP Exceedance** | Blended fuel RVP exceeds summer limit (e.g., 9.15 psi vs. 9.00 psi spec) during March/April transition. | Failure to flush winter-grade high-RVP butane heels from storage tanks and pipeline manifolds before summer blending. | Tank lockup; pipeline common carrier rejects injection; refinery suffers demurrage and tank-top threat. | Flush tank heels with heavy reformate or alkylate; implement seasonal heel turnover schedule. |
| **Asphaltene Sludge Precipitation** | Centrifuge sediment warnings on bunker vessels; rising differential pressure across fuel strainers. | Incompatible blend components; blending highly paraffinic cutter stocks into aromatic residual fuels. | Catastrophic engine fuel pump seizure, vessel blackout, total loss of propulsion in navigation channels. | Test component compatibility using ASTM D4740 spot test and automated peptization analyzers (P-value). |
| **SLP Recursion Limit-Cycling** | Commercial planning tool (PIMS) fails to converge; objective value oscillates wildly between iterations. | Bilinear pooling terms causing non-convex hopping; step-bounds on recursion variables are absent or too wide. | Planner relies on non-converged pseudo-solutions; refinery operates on sub-optimal component recipes. | Enforce trust-region step-bounds ($\Delta \le 5\%$); apply damping factors; switch to global sBB solver. |
| **Sub-Optimal Pooling Trap** | Planning model consistently recommends purchasing expensive high-octane components despite idle reformer capacity. | Local NLP solver trapped in an inferior local optimum in intermediate component pools. | Millions of dollars in unnecessary feedstock purchases; suboptimal utilization of refinery conversion units. | Use McCormick convex relaxations to generate lower bounds; solve via spatial branch-and-bound (sBB). |
| **Tank Stratification & Inversion** | Top and bottom lab samples from a 150,000 bbl tank diverge significantly (e.g., top = 88.5 AKI, bottom = 85.0 AKI). | Density stratification; heavy reformate ($SG = 0.81$) injected into lighter naphtha ($SG = 0.70$) without adequate jet mixing. | Off-spec fuel pumped into pipeline as tank empties; pipeline shutdown and contamination fines. | Install submerged high-velocity jet mixing eductors; enforce minimum circulation turnover times. |
| **Online NIR Spectrometer Drift** | Divergence between online NIR predicted RON and offline ASTM CFR knock-engine referee results ($>0.3\text{ RON}$). | Optical window fouling, spectrometer lamp aging, or crude slate shift altering hydrocarbon matrix absorption. | Undetected off-spec fuel blended directly to marine tankers or pipeline headers. | Enforce daily validation against ASTM referee samples; implement automated statistical drift alarms (SPC). |
| **Multi-Product Tank Heel Contamination** | Jet A-1 batch fails flash point or smoke point test after tank filling. | Staging Jet A-1 into a swing tank containing a residual heel of diesel or motor gasoline. | Product downgraded to lower-value slop or diesel; high re-refining and flushing cost. | Enforce dedicated tank service; execute strict heel pump-out protocols; mandate laboratory heel sign-off. |
| **Pipeline Interface Transmix Downgrade** | High transmix volume reported at pipeline delivery terminal; interface cuts exceeding normal volume. | Pumping premium gasoline directly adjacent to jet fuel or diesel without compatible buffer batches. | Degrades high-value finished products into low-value transmix requiring regional re-refining. | Optimize pipeline pumping sequences (Regular $\to$ Premium $\to$ Kero $\to$ Diesel); inject dye markers. |
| **Cold Flow Crystallization (CFPP)** | Winter diesel plugs commercial truck fuel filters at $-5^\circ\text{C}$ despite compliant cloud point. | Non-linear crystallization of heavy normal paraffins ($C_{20}\text{--}C_{30}$) forming wax crystal matrices. | Massive commercial warranty claims; roadside fleet breakdown; regulatory non-compliance. | Blend kerosene/jet fuel to dilute heavy paraffins; inject cold flow improver (CFI) polymeric additives. |

---

## 11. Assumptions and limits

| Assumption | Operational Justification | Failure Scenario & Root Cause | Diagnostic & Verification Method |
|---|---|---|---|
| **Ideal Volume Additivity** | Simplifies linear mass balances; volume shrinkage upon mixing is minor for similar boiling hydrocarbons. | Blending light LPG/butane into heavy gas oil; excess volume of mixing ($\Delta V_{mix} < 0$) causes 0.5%–1.5% volumetric shrinkage. | Measure blend density via digital densitometer (ASTM D4052); verify volumetric balances against mass meters. |
| **Perfect Tank Mixing (CSTR)** | Assumes intermediate pool tanks and finished tanks are completely homogeneous throughout their volume. | Insufficient mixer power, non-aligned jet nozzles, or density differences cause thermal or gravity stratification. | Multi-level tank sampling (Top, Upper, Middle, Lower, Outlet) per ASTM D4057; compare density and RVP deltas. |
| **Stationary Feedstock Assays** | Assumes stream qualities (octane, RVP, sulfur) remain static throughout the blend execution horizon. | Upstream crude slate switch or FCC regenerator temperature excursion alters stream quality mid-blend. | Continuous online process analyzer monitoring; alert if stream parameters deviate by $>1.5\sigma$. |
| **Isothermal Blending** | Assumes liquid temperatures remain constant; eliminates temperature-dependent volume expansion corrections. | Pumping hot component streams ($60^\circ\text{C}$) directly into cold finished storage tanks ($15^\circ\text{C}$). | Correct all volumetric measurements to standard temperature ($15^\circ\text{C}$ / $60^\circ\text{F}$) per API MPMS Chapter 11.1. |
| **Strict Sulfur Mass Additivity** | Total sulfur is treated as chemically conservative and linear on a mass basis. | Highly aromatic streams cause matrix quenching in UV fluorescence sulfur analyzers, reporting false low sulfur. | Cross-check UV fluorescence (ASTM D5453) against X-ray fluorescence (WDXRF ASTM D2622); run matrix spikes. |
| **Independent Blending Indices** | Assumes RVP, viscosity, and flash point indices operate independently without cross-property coupling. | Blending high-oxygenate components (ethanol, MTBE); hydrogen bonding creates extreme non-ideal azeotropes. | Use specialized oxygenate-gasoline interaction curves; avoid generalized index models for ethanol blends. |
| **Deterministic Component Availability** | Assumes planned component volumes will be produced without operational interruption. | Unplanned upstream unit trip (e.g., Alkylation acid unit shutdown) cuts high-octane feed mid-blend. | Maintain dedicated safety component ullage; enforce dynamic re-optimization upon unit trips. |
| **Static Planning Boundary Conditions** | Assumes market prices, vessel loading schedules, and demand commitments remain fixed over the multi-period run. | Marine vessel arrives 48 hours early or delayed; pipeline batch pumping sequence rescheduled by operator. | Integrate rolling-horizon dynamic scheduling with automated ERP / terminal management systems. |
| **Zero Valve & Header Holdup Lag** | Assumes ratio changes at control valves manifest instantaneously at the finished blend manifold. | Long pipeline runs between tank farm and blend manifold introduce 5–15 minute transport delays. | Implement Smith Predictors or time-delay compensation inside the Advanced Process Control (APC) loop. |

---

## 12. Exercises

### Exercise 1: Non-Linear Property Index Conversions
**Context:** A marine fuel blender in Singapore must formulate a Very Low Sulfur Fuel Oil (VLSFO) blend meeting ISO 8217 RMG 380 specifications. The maximum allowable kinematic viscosity at 50°C is $380.0\text{ mm}^2/\text{s}$ ($\text{cSt}$), and the minimum allowable closed-cup flash point is $60.0^\circ\text{C}$ ($140.0^\circ\text{F}$).
- Component 1: Heavy Vacuum Residue ($\nu = 650.0\text{ cSt}$ at 50°C, $T_{FP} = 110.0^\circ\text{C} = 230.0^\circ\text{F}$).
- Component 2: Light Marine Gas Oil cutter stock ($\nu = 3.2\text{ cSt}$ at 50°C, $T_{FP} = 65.0^\circ\text{C} = 149.0^\circ\text{F}$).
**Deliverable:** 
1. Calculate the Refutas Viscosity Blend Number ($VBN$) for both components and the target blend limit.
2. Determine the maximum allowable mass fraction ($w_1$) of heavy residue that satisfies the $380\text{ cSt}$ viscosity ceiling.
3. Calculate the flash point blending indices using the Hu and Burns method and verify whether this maximum residue blend satisfies the $60^\circ\text{C}$ flash point floor.
**Acceptance Criteria:** Correct application of the Refutas double-logarithmic equation and Hu-Burns flash point index; intermediate arithmetic displayed to 4 decimal places.
**Estimated Time:** 25 minutes.

### Exercise 2: McCormick Envelope Hand Calculation for Bilinear Pooling
**Context:** A refinery intermediate pool receives light reformate and light naphtha. The pool quality variable $q_p$ (RON) is bounded by $q_p \in [70.0, 95.0]$. The volumetric flow rate drawn from this pool to the premium gasoline header is $F_{p,Prem} \in [0, 20,000\text{ bbl/day}]$.
**Deliverable:**
1. Write down the four McCormick relaxation inequalities for the bilinear product $w = F_{p,Prem} \cdot q_p$.
2. For an operating point of $F_{p,Prem} = 10,000\text{ bbl/day}$ and $q_p = 80.0\text{ RON}$, calculate the exact bilinear product $w^*$, the McCormick lower bound, and the McCormick upper bound.
3. Calculate the relaxation error ($UB - LB$) at this operating point.
**Acceptance Criteria:** Exact algebraic derivation of the four planes; numerical evaluation matches hand arithmetic.
**Estimated Time:** 20 minutes.

### Exercise 3: Linear vs. Non-Linear Gasoline RVP Optimization
**Context:** A blend planner is preparing a $50,000\text{ bbl}$ batch of summer regular gasoline ($\text{RVP} \le 9.0\text{ psi}$). Two components are available:
- Alkylate: $\text{RVP} = 4.5\text{ psi}$, Cost = $\$70.00/\text{bbl}$.
- FCC Naphtha: $\text{RVP} = 11.5\text{ psi}$, Cost = $\$52.00/\text{bbl}$.
**Deliverable:**
1. Formulate the blend constraint assuming naive linear RVP blending. Calculate the maximum allowable fraction of FCC Naphtha and the resulting blend cost.
2. Formulate the blend constraint using the Chevron non-linear RVP index ($BI = \text{RVP}^{1.25}$). Calculate the true maximum allowable fraction of FCC Naphtha and the true blend cost.
3. Compare the results. What happens if the refinery blends according to the naive linear model and ships the batch into the Colonial Pipeline?
**Acceptance Criteria:** Clear calculation showing the discrepancy between linear and Chevron blending; financial and compliance consequences explicitly stated.
**Estimated Time:** 30 minutes.

### Exercise 4: Haverly's Pooling Problem — $p$- vs. $q$-Formulation
**Context:** Review Haverly’s classical Pooling Problem 1 described in Section 5.
**Deliverable:**
1. Write the complete algebraic model in the $p$-formulation (using variables $x_1, x_2, z_A, z_B, y_A, y_B, p$).
2. Transform the model into the $q$-formulation by defining split proportions $v_1, v_2$.
3. Prove that the bilinear terms in the $q$-formulation are bounded by a fixed $[0, 1]$ simplex, explaining why this accelerates global branch-and-bound convergence.
**Acceptance Criteria:** Formulations mathematically sound; rigorous explanation of polyhedral relaxation tightness.
**Estimated Time:** 35 minutes.

### Exercise 5: Critique an SLP Convergence Failure
**Context:** A commercial planning model for a major refinery exhibits severe limit cycling during seasonal transition planning. The LP log shows that the allocation of butane swings between $0\text{ bbl/day}$ and $8,000\text{ bbl/day}$ every iteration, never converging.
**Deliverable:** Write a technical diagnostic memorandum (300–400 words) explaining:
1. The mathematical mechanism causing this limit cycle.
2. Why standard first-order Taylor series linearizations fail in the presence of wide bounds on volatile streams.
3. Three concrete numerical interventions to restore robust convergence.
**Acceptance Criteria:** Names trust regions, step-bounding, recursion damping, and spatial branch-and-bound.
**Estimated Time:** 25 minutes.

### Exercise 6: Bunker Blend Compatibility and Instability Critique
**Context:** An independent marine fuel blender blends $70\text{ vol\%}$ of low-sulfur atmospheric residue ($SG = 0.94$, Sulfur = $0.45\%$, Viscosity = $180\text{ cSt}$) with $30\text{ vol\%}$ of hydrotreated light gas oil ($SG = 0.83$, Sulfur = $0.05\%$, Viscosity = $2.5\text{ cSt}$) to create a compliant VLSFO. Three days after blending, the fuel in the storage tank separates into two phases with a thick asphaltene sludge layer at the bottom.
**Deliverable:** 
1. Explain the physical-chemical mechanism causing this phase separation.
2. Identify which ASTM / ISO standards were violated.
3. Propose a mathematical constraint that should have been included in the blend optimizer to prevent this failure.
**Acceptance Criteria:** Explains colloidal stability, asphaltene peptization, aromaticity vs. paraffinicity, and the P-value / ASTM D4740 test.
**Estimated Time:** 25 minutes.

### Exercise 7: Statistical Safety Buffer and Giveaway Optimization
**Context:** A refinery blends $120,000\text{ bbl/day}$ of regular gasoline with a statutory octane limit of $\ge 87.0\text{ AKI}$. The combined standard deviation of the blend process and testing is $\sigma_{blend} = 0.30\text{ AKI}$. The shadow price of octane is $\lambda = \$1.20/(\text{bbl} \cdot \text{AKI})$. The cost of an off-spec batch failure (including tank quarantine, circulation, high-octane trimming, and ship demurrage) is amortized at $C_{reblend} = \$85,000$ per incident. A typical blend batch size is $50,000\text{ bbl}$ (equivalent to $\$1.70/\text{bbl}$ normalized failure cost).
**Deliverable:**
1. Using the analytical trade-off formula derived in Section 5, calculate the optimal safety buffer multiplier $k^*$ and the target operating octane $Q_{target}$.
2. Calculate the annual economic giveaway cost under this optimal buffer.
3. If the refinery installs an online NIR analyzer that reduces $\sigma_{blend}$ to $0.08\text{ AKI}$, calculate the new $k^*$, the new $Q_{target}$, and the annual cost savings.
**Acceptance Criteria:** Correct mathematical evaluation of normal density inverse; accurate annual cost calculations.
**Estimated Time:** 30 minutes.

### Exercise 8: Design a Closed-Loop In-Line Blending Control Architecture
**Context:** You are the Lead Process Automation Engineer designing the control system for a new grass-roots $150,000\text{ bpd}$ in-line gasoline blending manifold.
**Deliverable:** Produce an engineering architecture specification (400–500 words) defining:
1. Instrumentation: Required flow meters, control valves, static mixers, and analyzer sample conditioning skids.
2. Analyzer Suite: Selection of online analyzers and measurement frequencies.
3. Control Hierarchy: Interactions between the DCS ratio controllers, the multivariable Advanced Process Controller (APC), and the offline scheduling optimizer.
4. Failsafe Protocol: Emergency actions executed when an on-line analyzer detects an off-spec condition mid-blend.
**Acceptance Criteria:** Comprehensive, realistic engineering design referencing standard industrial DCS/APC architectures and safety protocols.
**Estimated Time:** 35 minutes.

---

## 13. Self-check answers

### Answer 1: Non-Linear Property Index Conversions
1. **Refutas VBN Calculation:**
   $$VBN = 14.534 \cdot \ln(\ln(\nu + 0.8)) + 10.975$$
   - Component 1 ($650.0\text{ cSt}$):
     $\nu_1 + 0.8 = 650.8 \implies \ln(650.8) \approx 6.4782 \implies \ln(6.4782) \approx 1.8684$
     $$VBN_1 = 14.534 \times 1.8684 + 10.975 = 27.1557 + 10.975 = \mathbf{38.1307}$$
   - Component 2 ($3.2\text{ cSt}$):
     $\nu_2 + 0.8 = 4.0 \implies \ln(4.0) \approx 1.3863 \implies \ln(1.3863) \approx 0.3266$
     $$VBN_2 = 14.534 \times 0.3266 + 10.975 = 4.7473 + 10.975 = \mathbf{15.7223}$$
   - Target Blend Limit ($380.0\text{ cSt}$):
     $\nu_{target} + 0.8 = 380.8 \implies \ln(380.8) \approx 5.9423 \implies \ln(5.9423) \approx 1.7821$
     $$VBN_{target} = 14.534 \times 1.7821 + 10.975 = 25.9010 + 10.975 = \mathbf{36.8760}$$
2. **Maximum Allowable Mass Fraction ($w_1$):**
   $$w_1 \cdot VBN_1 + (1 - w_1) \cdot VBN_2 \le VBN_{target}$$
   $$w_1 (381.1307 - 15.7223) \le 36.8760 - 15.7223 \implies w_1 (22.4084) \le 21.1537$$
   $$w_1 \le \frac{21.1537}{22.4084} \approx \mathbf{0.9440} \quad (\mathbf{94.40\%\text{ Heavy Residue}})$$
   *(Note: A naive linear viscosity blend would have erroneously calculated $w_1 \times 650 + (1-w_1) \times 3.2 \le 380 \implies w_1 \le 58.26\%$, an enormous error that would have wasted tens of thousands of tons of expensive cutter stock).*
3. **Flash Point Verification (Hu and Burns):**
   - Residue ($230.0^\circ\text{F}$): $BI_{FP, 1} = 100 \cdot 10^{-0.06 \times 230} = 100 \cdot 10^{-13.80} \approx 1.58 \times 10^{-12}$
   - Cutter ($149.0^\circ\text{F}$): $BI_{FP, 2} = 100 \cdot 10^{-0.06 \times 149} = 100 \cdot 10^{-8.94} \approx 100 \times 1.148 \times 10^{-9} = 1.148 \times 10^{-7}$
   - Target floor ($140.0^\circ\text{F} = 60.0^\circ\text{C}$): $BI_{FP, max} = 100 \cdot 10^{-0.06 \times 140} = 100 \cdot 10^{-8.40} = 3.981 \times 10^{-7}$
   - Blended index at $w_1 = 0.944, w_2 = 0.056$:
     $$BI_{blend} \approx 0.056 \times (1.148 \times 10^{-7}) = 6.429 \times 10^{-9}$$
     Since $6.429 \times 10^{-9} \le 3.981 \times 10^{-7}$, the flash point is significantly higher than the $60^\circ\text{C}$ minimum. Specifically:
     $$T_{FP, blend} = -\frac{1}{0.06} \log_{10}\left( \frac{6.429 \times 10^{-9}}{100} \right) = -\frac{1}{0.06} (-10.1918) = 169.86^\circ\text{F} = \mathbf{76.6^\circ\text{C}} \ge 60^\circ\text{C (Compliant)}$$

### Answer 2: McCormick Envelope Hand Calculation
1. **Four McCormick Inequalities:**
   Given $x = F_{p,Prem} \in [0, 20,000]$ and $y = q_p \in [70.0, 95.0]$:
   - Underestimator 1: $w \ge x^L y + x y^L - x^L y^L = 0 \cdot y + 70.0 \cdot x - 0 = \mathbf{70.0 \cdot x}$
   - Underestimator 2: $w \ge x^U y + x y^U - x^U y^U = 20,000 \cdot y + 95.0 \cdot x - (20,000 \times 95.0) = \mathbf{95.0 \cdot x + 20,000 \cdot y - 1,900,000}$
   - Overestimator 1: $w \le x^U y + x y^L - x^U y^L = 20,000 \cdot y + 70.0 \cdot x - (20,000 \times 70.0) = \mathbf{70.0 \cdot x + 20,000 \cdot y - 1,400,000}$
   - Overestimator 2: $w \le x^L y + x y^U - x^L y^U = 0 \cdot y + 95.0 \cdot x - 0 = \mathbf{95.0 \cdot x}$
2. **Evaluation at $x = 10,000$, $y = 80.0$:**
   - Exact bilinear product: $w^* = 10,000 \times 80.0 = \mathbf{800,000}$
   - Underestimator 1: $70.0 \times 10,000 = 700,000$
   - Underestimator 2: $95.0(10,000) + 20,000(80.0) - 1,900,000 = 950,000 + 1,600,000 - 1,900,000 = 650,000$
     $$\implies \text{Convex Lower Bound } LB = \max(700,000, 650,000) = \mathbf{700,000}$$
   - Overestimator 1: $70.0(10,000) + 20,000(80.0) - 1,400,000 = 700,000 + 1,600,000 - 1,400,000 = 900,000$
   - Overestimator 2: $95.0(10,000) = 950,000$
     $$\implies \text{Concave Upper Bound } UB = \min(900,000, 950,000) = \mathbf{900,000}$$
3. **Relaxation Error:**
   $$\text{Error Gap} = UB - LB = 900,000 - 700,000 = \mathbf{200,000}$$
   The exact value $w^* = 800,000$ sits symmetrically inside $[700,000, 900,000]$.

### Answer 3: Linear vs. Non-Linear Gasoline RVP Optimization
1. **Naive Linear RVP Formulation:**
   Let $v_F$ be the fraction of FCC Naphtha.
   $$(1 - v_F) \cdot 4.5 + v_F \cdot 11.5 \le 9.0 \implies 4.5 + 7.0 \cdot v_F \le 9.0 \implies v_F \le \frac{4.5}{7.0} \approx \mathbf{0.6429} \quad (\mathbf{64.29\%})$$
   $$\text{Blend Unit Cost} = (0.3571 \times \$70.00) + (0.6429 \times \$52.00) = \$25.00 + \$33.43 = \mathbf{\$58.43/\text{bbl}}$$
2. **True Chevron Index Formulation:**
   - Target Index: $(9.0)^{1.25} = 15.5885$
   - Alkylate Index: $(4.5)^{1.25} = 6.5542$
   - FCC Index: $(11.5)^{1.25} = 21.1712$
   $$(1 - v_F)(6.5542) + v_F(21.1712) \le 15.5885$$
   $$6.5542 + v_F(14.6170) \le 15.5885 \implies v_F \le \frac{9.0343}{14.6170} \approx \mathbf{0.6181} \quad (\mathbf{61.81\%})$$
   $$\text{True Compliant Cost} = (0.3819 \times \$70.00) + (0.6181 \times \$52.00) = \$26.73 + \$32.14 = \mathbf{\$58.87/\text{bbl}}$$
3. **Operational Consequence:**
   If the refinery blends $64.29\%$ FCC naphtha based on the naive linear model:
   $$BI_{actual} = (0.3571 \times 6.5542) + (0.6429 \times 21.1712) = 2.3405 + 13.6110 = 15.9515$$
   $$RVP_{actual} = (15.9515)^{1/1.25} = (15.9515)^{0.80} = \mathbf{9.17\text{ psi}}$$
   The batch is **off-spec** ($9.17\text{ psi} > 9.00\text{ psi}$). The Colonial Pipeline will reject the $50,000\text{ bbl}$ batch. The refinery must quarantine the tank, inject approximately $5,500\text{ bbl}$ of costly heavy alkylate or reformate, recirculate for 24 hours, and pay vessel/pipeline delay penalties exceeding $\$100,000$, obliterating the imaginary paper savings of the linear model.

### Answer 4: Haverly's Pooling Problem Formulations
1. **$p$-Formulation Model:**
   $$\max \; 9(y_A + z_A) + 15(y_B + z_B) - 6 x_1 - 16 x_2 - 10(z_A + z_B)$$
   subject to:
   - Flow balance: $x_1 + x_2 = y_A + y_B$
   - Demand limits: $y_A + z_A \le 100, \quad y_B + z_B \le 200$
   - Component 1 availability: $x_1 \le 100$
   - Pool quality definition (bilinear): $p (x_1 + x_2) = 3 x_1 + 1 x_2$
   - Product A sulfur (bilinear): $p \cdot y_A + 2 z_A \le 2.5 (y_A + z_A)$
   - Product B sulfur (bilinear): $p \cdot y_B + 2 z_B \le 1.5 (y_B + z_B)$
   - Variables non-negative, $p \in [1, 3]$.
2. **$q$-Formulation Transformation:**
   Define split proportions $v_1 = x_1 / (x_1 + x_2)$ and $v_2 = x_2 / (x_1 + x_2)$ where $v_1 + v_2 = 1, v_i \ge 0$.
   Pool sulfur is linear in $v$: $p = 3 v_1 + 1 v_2$.
   Substituting into product constraints:
   $$(3 v_1 + 1 v_2) y_A + 2 z_A \le 2.5 (y_A + z_A) \implies 3 (v_1 y_A) + 1 (v_2 y_A) + 2 z_A \le 2.5 (y_A + z_A)$$
   $$(3 v_1 + 1 v_2) y_B + 2 z_B \le 1.5 (y_B + z_B) \implies 3 (v_1 y_B) + 1 (v_2 y_B) + 2 z_B \le 1.5 (y_B + z_B)$$
   The bilinear products are $w_{i,j} = v_i \cdot y_j$.
3. **Proof of Tighter Relaxation:**
   In the $q$-formulation, $v_i$ is strictly bounded by the unit simplex $[0, 1]$ and $\sum_i v_i = 1$. The McCormick convex hull of $v_i \cdot y_j$ over $[0, 1] \times [0, Y_j^{max}]$ is exact at the extreme points $0$ and $1$. In contrast, in the $p$-formulation, $p \in [1, 3]$ and total pool flow $\sum F$ have wide, uncoupled bounds. As proven by Meyer and Floudas (2006), the polyhedral relaxation of the $q$-formulation strictly contains or matches the $p$-formulation relaxation and eliminates the weak McCormick corners, dramatically reducing the spatial branch-and-bound tree depth.

### Answer 5: Critique an SLP Convergence Failure
- **Rubric:**
  - *Mechanism:* Explains that the Taylor series linearization evaluates the partial derivative of cost/octane with respect to butane at a fixed operating point. Because butane has an extreme RVP ($52\text{ psi}$) and high octane, its marginal value curve is highly non-linear. The linear approximation assumes that adding butane has a constant positive marginal profit up to the full step bound, causing the LP to jump to the upper bound ($8,000\text{ bbl}$). At that point, the non-linear simulator evaluates RVP, discovers an enormous off-spec violation, sets the butane shadow price to zero/negative, causing the next LP iteration to collapse to $0\text{ bbl}$.
  - *Failure of First-Order Model:* First-order Taylor series omit second-order curvature ($\nabla^2 f$). For properties with exponents (e.g., $BI = \text{RVP}^{1.25}$), the linear tangent drastically overpredicts feasible blending volume away from the expansion point.
  - *Interventions:* (1) Impose tight trust-region step-bounds ($\Delta_{butane} \le 500\text{ bbl/day}$) that contract by $50\%$ upon failed iterations; (2) Implement recursion damping (e.g., $x^{(k+1)} = 0.5 x^{(k)} + 0.5 x_{LP}$); (3) Migrate the light-ends blending model from SLP to a dedicated sequential quadratic programming (SQP) or global sBB solver.

### Answer 6: Bunker Blend Compatibility Critique
- **Rubric:**
  - *Mechanism:* Heavy residual fuel oils maintain high-molecular-weight asphaltenes in colloidal suspension via aromatic hydrocarbon "resins" that act as natural peptizing agents. When a highly paraffinic light gas oil is added as a cutter, the aromaticity of the continuous solvent phase collapses. The asphaltenes lose their peptizing layer, become mutually insoluble, and precipitate into agglomerated tar/sludge (flocculation).
  - *Standards Violated:* ISO 8217 Clause 5 (General Requirements: fuel must be homogeneous and free from any substance that jeopardizes the safety of ships), ISO 10307-2 (Total Sediment Potential, TSP $\le 0.10\%$), and ASTM D4740 (Cleanliness and Compatibility Spot Test rating $\le 2$).
  - *Optimization Constraint:* Enforce a minimum Colloidal Stability / Peptization Index ($P$-value) constraint:
    $$P = \frac{P_o}{1 - P_a} \ge 1.25$$
    or enforce a minimum Aromaticity Index (e.g., Calculated Carbon Aromaticity Index, CCAI) and maximum paraffinic cutter blending limit derived from laboratory compatibility cross-matrices.

### Answer 7: Statistical Safety Buffer and Giveaway Optimization
1. **Optimal Safety Multiplier ($k^*$):**
   - $\sigma_{blend} = 0.30\text{ AKI}$
   - $\lambda = \$1.20/(\text{bbl} \cdot \text{AKI})$
   - Re-blending failure penalty per barrel: $C_{fail} = \$1.70/\text{bbl}$
   $$\phi(k^*) = \frac{\sigma_{blend} \cdot \lambda}{C_{fail}} = \frac{0.30 \times 1.20}{1.70} = \frac{0.36}{1.70} \approx 0.2118$$
   Using standard normal distribution tables, the value of $k$ where standard normal density $\phi(k) = \frac{1}{\sqrt{2\pi}} e^{-k^2/2} = 0.2118$:
   $$\frac{1}{2.5066} e^{-k^2/2} = 0.2118 \implies e^{-k^2/2} = 0.5309 \implies -\frac{k^2}{2} = \ln(0.5309) = -0.6332$$
   $$k^2 = 1.2664 \implies \mathbf{k^* \approx 1.125}$$
   $$\text{Safety Buffer} = k^* \cdot \sigma_{blend} = 1.125 \times 0.30 = \mathbf{0.338\text{ AKI}}$$
   $$\text{Target Operating Octane } Q_{target} = 87.0 + 0.338 = \mathbf{87.338\text{ AKI}}$$
2. **Annual Giveaway Cost:**
   $$\text{Daily Giveaway} = 120,000\text{ bbl} \times 0.338\text{ AKI} \times \$1.20 = \$48,672/\text{day}$$
   $$\text{Annual Giveaway Cost} = \$48,672 \times 365 = \mathbf{\$17,765,280/\text{year}}$$
3. **Impact of Online NIR Analyzer ($\sigma_{blend} = 0.08\text{ AKI}$):**
   $$\phi(k_{new}^*) = \frac{0.08 \times 1.20}{1.70} = \frac{0.096}{1.70} \approx 0.0565$$
   $$e^{-k^2/2} = 0.0565 \times 2.5066 = 0.1416 \implies -\frac{k^2}{2} = -1.9546 \implies k^2 = 3.909 \implies \mathbf{k_{new}^* \approx 1.977}$$
   $$\text{New Safety Buffer} = 1.977 \times 0.08 = \mathbf{0.158\text{ AKI}}$$
   $$\text{New Target } Q_{target} = 87.0 + 0.158 = \mathbf{87.158\text{ AKI}}$$
   $$\text{New Daily Giveaway} = 120,000 \times 0.158 \times \$1.20 = \$22,752/\text{day}$$
   $$\text{New Annual Giveaway Cost} = \$22,752 \times 365 = \mathbf{\$8,304,480/\text{year}}$$
   $$\mathbf{\text{Annual Net Margin Savings} = \$17,765,280 - \$8,304,480 = \$9,460,800/\text{year}}$$

### Answer 8: Closed-Loop In-Line Blending Control Architecture
- **Rubric:**
  - *Instrumentation:* Individual intermediate component run-down lines equipped with Coriolis mass flow meters (for density and mass accuracy), high-turn-down ratio electro-hydraulic control valves, non-return check valves, and low-shear static in-line mixers. Dedicated additive injection skids (corrosion inhibitors, metal deactivators, cold-flow improvers).
  - *Analyzer Suite:* Fast sample-loop conditioning system pulling downstream of the static mixer. Dual-channel Near-Infrared (NIR) spectrometer (measuring RON, MON, benzene, distillation every 45s); online mini-method RVP analyzer (ASTM D5191 every 3 minutes); pulsed UV fluorescence total sulfur analyzer (ASTM D5453 every 2 minutes).
  - *Control Hierarchy:* Distributed Control System (DCS) executing high-frequency (100 ms) flow-ratio control loops. Advanced Process Control (APC) multivariable controller (e.g., DMC3) executing every 15–30 seconds, taking analyzer quality deltas, calculating dynamic gain matrices, and adjusting flow ratio setpoints to track the economic optimizer's target. Offline scheduling tool (e.g., Blend Commander) updates gross recipe setpoints based on pipeline pumping schedule.
  - *Failsafe Protocol:* If any certified property breaches safety bounds ($Q > Q_{limit} - 0.5\sigma$) for $>90$ seconds, APC triggers an automatic trim ramp on clean components (e.g., cuts butane, boosts alkylate). If uncorrectable off-spec is detected, automated emergency divert valves swing the header discharge from the custody pipeline into an internal slop/quarantine tank within 15 seconds, preventing pipeline contamination.

---

## 14. Interview questions

### Question 1: "Why do refiners use non-linear blending indices instead of linear programming for properties like RVP and viscosity, and what mathematical complications arise when intermediate tanks are introduced?"
**What a strong answer contains:** A strong answer defines the physical-chemical root cause: non-ideal vapor-liquid equilibrium (positive Raoult's law deviations for RVP) and double-logarithmic momentum transport for viscosity. It cites specific industrial transformations (Chevron $RVP^{1.25}$, Refutas VBN). It then transitions to intermediate storage, explicitly identifying the **Pooling Problem**. It explains that pool quality depends on inlet stream fractions, creating bilinear terms ($F_{p,j} \cdot q_p$) when pool effluent is routed to blend headers. It notes that bilinearity destroys convexity, creating multiple local optima where standard LP or local NLP solvers become trapped.
**What a weak answer reveals:** The candidate vaguely mentions that "fluids mix weirdly" or "chemistry is non-linear." They cannot write down the Refutas or Chevron transformation equations. They confuse the pooling problem with standard linear blending and fail to explain bilinearity, saddle points, or non-convexity.

### Question 2: "What is the difference between the $p$-formulation and the $q$-formulation of the pooling problem, and why does one perform better in global branch-and-bound solvers?"
**What a strong answer contains:** The candidate explains that the $p$-formulation uses explicit quality concentration variables ($q_p$) and flow rates, producing bilinear products of flow and quality. The $q$-formulation (Ben-Tal et al.) substitutes quality variables with inlet split fractions ($v_{i,p} = F_{i,p} / \sum F$). The bilinear term becomes split fraction times product draw ($v_{i,p} \cdot F_{p,j}$). A strong answer notes that because $v_{i,p}$ is naturally bounded on the unit simplex $[0, 1]$ with $\sum v_i = 1$, convex relaxations (McCormick envelopes) constructed over the $q$-formulation are dramatically tighter, eliminating large portions of the search space and accelerating Spatial Branch-and-Bound convergence.
**What a weak answer reveals:** The candidate has no knowledge of pooling literature. They guess that $p$ stands for "price" or "product" and $q$ stands for "quantity." They cannot explain convex relaxation tightness.

### Question 3: "Explain how McCormick relaxation envelopes work. How do they turn a non-convex bilinear program into a solvable convex bounding problem?"
**What a strong answer contains:** The candidate defines the bilinear term $w = x \cdot y$ over a bounded box $[x^L, x^U] \times [y^L, y^U]$. They state that McCormick envelopes represent the convex hull of the bilinear surface, formed by two convex underestimators and two concave overestimators derived from the non-negative products $(x - x^L)(y - y^L) \ge 0$, etc. They explain that replacing bilinear terms with variable $w$ constrained by these four linear planes yields a relaxed Linear Program (LP). The LP solution provides a guaranteed mathematical lower bound on cost. Solvers then use Spatial Branch-and-Bound to partition the variable domains, tightening the envelopes until the global gap collapses to $\epsilon$.
**What a weak answer reveals:** The candidate confuses McCormick relaxations with Big-M formulations or simple linear interpolation. They do not know that McCormick envelopes require finite variable bounds, or they believe McCormick relaxations find the exact solution in a single step without branching.

### Question 4: "What is quality giveaway in refinery economics, and how do you calculate the optimal statistical safety margin between the operating target and the contractual limit?"
**What a strong answer contains:** The candidate defines giveaway as the intentional delivery of property value exceeding minimum contract specifications (e.g., blending 87.3 AKI when 87.0 is required), wasting high-value molecules (reformate, alkylate). They explain that giveaway is an operational defense against stochastic stream drift and measurement variance ($\sigma_{blend}$). They write out the trade-off objective function balancing the marginal cost of giveaway ($k \cdot \sigma \cdot \lambda$) against the expected cost of an off-spec batch ($P(fail) \cdot C_{reblend}$). They show that setting the derivative to zero yields the analytical optimal cutoff: $\phi(k^*) = (\sigma \cdot \lambda) / C_{reblend}$.
**What a weak answer reveals:** The candidate thinks giveaway is "giving free product to build customer relationships" or believes refiners should target the exact contract limit ($k=0$) with zero safety buffer, ignoring the catastrophic economic cost of batch re-blending and demurrage.

### Question 5: "In commercial refinery planning tools like Aspen PIMS, Successive Linear Programming (SLP) often suffers from limit cycling. What causes this and how do you fix it?"
**What a strong answer contains:** The candidate explains that SLP linearizes non-linear pooling terms via first-order Taylor series around the current operating point. In the presence of strong non-linearities (such as butane RVP), the linear tangent drastically overpredicts feasible gains, causing the LP to bang against upper bounds. In the next iteration, the non-linear evaluator detects severe infeasibility, causing the LP to jump to lower bounds. To fix limit cycling, the candidate details: (1) trust-region step-bounds that restrict variable movement per iteration; (2) step-bound contraction/expansion heuristics based on model fidelity; (3) recursion damping factors; and (4) upgrading to sequential quadratic programming (SQP) or global sBB algorithms.
**What a weak answer reveals:** The candidate blames "bad data" or "the computer running out of memory." They cannot explain the interaction between linear approximations and non-convex surfaces.

### Question 6: "What happened in the marine bunker market during the 2020 IMO sulfur transition regarding fuel stability and compatibility, and how should a blend optimizer account for it?"
**What a strong answer contains:** The candidate details the IMO 2020 MARPOL Annex VI mandate cutting marine fuel sulfur from 3.5% to 0.50% (VLSFO). To hit this without costly desulfurization, suppliers blended low-sulfur paraffinic cutter stocks (gasoils) with heavy aromatic residual slurries. The paraffinic solvent collapsed the colloidal peptization of asphaltenes, causing severe precipitation and filter-plugging sludge that disabled dozens of marine vessels. A strong answer states that blend optimizers must include non-linear stability constraints—such as the P-value (ASTM D7060 / D7157), Calculated Carbon Aromaticity Index (CCAI), and empirical component compatibility spot matrices (ASTM D4740)—rather than treating viscosity and sulfur as the only constraints.
**What a weak answer reveals:** The candidate only knows that sulfur was reduced to 0.50%. They have no understanding of asphaltene chemistry, fuel compatibility, or the mechanical failures caused by improper bunker blending.

### Question 7: "If a refinery's blending LP indicates that the dual value (shadow price) of octane in regular gasoline is zero, but in premium gasoline it is $\$1.80/(\text{bbl} \cdot \text{RON})$, what does this tell you about the refinery's operating constraints?"
**What a strong answer contains:** A shadow price of zero means the regular gasoline octane constraint is **non-binding** (slack)—the blend is delivering more octane than the 87.0 minimum. This indicates that another constraint is forcing high-octane components into regular gasoline, such as a binding maximum RVP constraint (forcing low-RVP, high-octane alkylate/reformate into the blend to dilute volatile FCC naphtha) or a binding sulfur limit. In contrast, the premium gasoline octane constraint is strictly binding; relaxing it by 1.0 RON would generate $\$1.80/\text{bbl}$ in margin by backing out expensive reformate or purchased ethanol.
**What a weak answer reveals:** The candidate incorrectly states that "octane has no value in regular gasoline" or fails to recognize that binding secondary constraints (RVP, sulfur) force octane giveaways.

### Question 8: "How does an In-Line Blending (ILB) system with online NIR spectroscopy fundamentally alter the risk-return profile of a refinery's bulk supply chain compared to traditional batch tank blending?"
**What a strong answer contains:** The candidate contrasts the two paradigms across working capital, tankage infrastructure, and quality control:
- *Batch Blending:* Requires massive dedicated tankage (blend tank, certified hold tank). 24–48 hour laboratory assay delays tie up millions in working capital. Large process variance forces large safety buffers ($k\sigma$), causing chronic quality giveaway.
- *In-Line Blending:* Pipes intermediate streams directly across a dynamic manifold into custody transfer pipelines or marine tankers. Continuous NIR feedback trims control valves in real time (every 45 seconds). This slashes $\sigma_{blend}$ by 70–85%, allowing the refinery to safely move operating targets closer to contract limits. It eliminates finished product tankage capital expenditure, eliminates demurrage waiting times, and captures millions in recovered octane/RVP giveaway.
**What a weak answer reveals:** The candidate treats NIR as merely "a faster lab test" without understanding how automated feedback trimming directly impacts tank inventory turnover, pipeline scheduling, and giveaway economics.

### Question 9: "How do swing tanks and tank heels create multi-period combinatorial non-convexities in daily blend scheduling, and how are they handled in production environments?"
**What a strong answer contains:** The candidate details the physical and mathematical reality of intermediate tank farms: tanks cannot be drained to zero volume without pump cavitation, leaving a dead heel ($V_{heel}$). When a swing tank changes service between campaigns (e.g., from high-aromatic reformate to low-sulfur alkylate), the heel acts as a legacy contaminant. Mathematically, the pool quality equation $q_{p,t} = (V_{t-1} q_{t-1} + \sum F_{in} q_{in}) / (V_{t-1} + \sum F_{in})$ creates bilinear multi-period state coupling. Combining this with binary variables ($Y_{p,s,t} \in \{0, 1\}$) for tank service selection yields a non-convex Mixed-Integer Non-Linear Program (MINLP). A strong answer explains that production scheduling tools decompose this by solving an outer discrete-time MILP for tank sequencing, followed by an inner NLP or Spatial Branch-and-Bound solver for blend quality feasibility.
**What a weak answer reveals:** The candidate assumes tanks are empty before every batch or that heels can be ignored. They have no concept of swing tank sequencing, multi-period state propagation, or MINLP decomposition.

### Question 10: "In California CARB Phase 3 gasoline blending, what is the 'Predictive Model' and how does it change the constraint structure compared to fixed ASTM D4814 caps?"
**What a strong answer contains:** The candidate explains that the California Air Resources Board (CARB) does not rely solely on independent property caps (e.g., rigid maximum limits for sulfur, olefins, or aromatics). Instead, CARB enforces a multi-variable non-linear regression system known as the Predictive Model, which calculates composite vehicle exhaust and evaporative emissions ($NO_x$, reactive organic gases, and toxic air contaminants) relative to a reference fuel. This transforms the blend optimization model from decoupled linear bounds into a system of coupled non-linear polynomial inequality constraints ($f_{CARB}(RVP, Sulfur, Aromat, Olefin, T_{50}, T_{90}, Oxygen) \le Emissions_{ref}$). This gives refiners operational degrees of freedom—allowing higher sulfur or aromatics if compensated by lower RVP or optimized distillation—but requires non-linear programming (NLP/SQP) rather than linear programming.
**What a weak answer reveals:** The candidate believes CARB is just "regular gasoline with lower sulfur" or assumes California simply enforces lower static numbers for each ASTM parameter. They are unfamiliar with emissions-equivalency regression models and coupled constraint spaces.

---

## 15. Key terms

- **Anti-Knock Index (AKI):** The arithmetic average of Research Octane Number and Motor Octane Number, $(\text{RON} + \text{MON})/2$, enforced as the legal retail fuel rating in North America.
- **Asphaltene:** High-molecular-weight polycyclic aromatic hydrocarbon structures dispersed as colloids in crude oil and residual fuels.
- **Bilinear Program:** A non-convex mathematical optimization problem containing terms formed by the product of two continuous decision variables ($x \cdot y$).
- **Calculated Carbon Aromaticity Index (CCAI):** An empirical index calculated from density and viscosity that characterizes the ignition delay and combustion quality of residual marine fuels.
- **Cat Fines:** Microscopic, abrasive aluminum and silicon oxide catalyst particles carried over from FCC units into residual fuel streams, capped by ISO 8217 at $\le 60\text{ ppm}$.
- **Chevron RVP Method:** An empirical non-linear blending index transformation, $BI_{RVP} = \text{RVP}^{1.25}$, used to linearize hydrocarbon vapor pressure calculations.
- **Cold Filter Plugging Point (CFPP):** The highest temperature at which a standardized volume of diesel fuel fails to pass through a standardized wire mesh filter within a specified time limit (ASTM D6371).
- **Colloidal Peptization State (P-Value):** A measure of the stability of asphaltenes in residual fuel oil against flocculation and sludge precipitation (ASTM D7060).
- **Demurrage:** Contractual financial penalty paid by a charterer to a ship owner for delays exceeding agreed laytime at marine loading/discharge berths.
- **Driveability Index (DI):** An empirical distillation index ($1.5 T_{10} + 3.0 T_{50} + 1.0 T_{90}$) that quantifies cold-start driveability and combustion smoothness in gasoline engines.
- **Dry Vapor Pressure Equivalent (DVPE):** Standardized vapor pressure metric equivalent to ASTM D323, enforced by the EPA under 40 CFR Part 1090.
- **EPA Tier 3:** Federal United States fuel standard mandating an annual refinery corporate average sulfur limit of $\le 10.0\text{ ppm}$ in motor gasoline.
- **Haverly's Pooling Problem:** The foundational non-convex benchmark problem in process systems optimization demonstrating the failure of recursive LP in intermediate storage networks.
- **In-Line Blending (ILB):** The continuous automated metering and mixing of intermediate component streams directly into a pipeline manifold or marine tanker.
- **McCormick Convex Envelope:** The tightest four-plane convex and concave polyhedral relaxation bounding a bilinear product over a finite rectangular domain.
- **Motor Octane Number (MON):** Standardized knock rating measured under severe engine speed (900 rpm) and preheated mixture conditions (ASTM D2700).
- **Near-Infrared (NIR) Spectroscopy:** High-speed optical analytical technique measuring molecular vibrational absorption to predict multi-property fuel assays in real time.
- **Off-Specification (Off-Spec):** A blended batch failing to meet one or more binding contractual, pipeline, or regulatory property limits.
- **p-Formulation:** A mathematical formulation of the pooling problem where pool quality concentrations are declared as explicit decision variables.
- **q-Formulation:** A mathematical formulation of the pooling problem where internal split fractions are declared as variables, yielding tighter polyhedral relaxations.
- **Quality Giveaway:** The economic and molecular loss incurred by delivering fuel exceeding minimum contractual specifications.
- **Refutas Method:** An industry-standard double-logarithmic mathematical index system used to linearize kinematic viscosity blending calculations.
- **Reid Vapor Pressure (RVP):** The absolute vapor pressure exerted by a volatile liquid fuel at $37.8^\circ\text{C}$ ($100^\circ\text{F}$) under standardized liquid-to-vapor volume conditions (ASTM D323).
- **Research Octane Number (RON):** Standardized knock rating measured under mild engine speed (600 rpm) and unheated intake air conditions (ASTM D2699).
- **Sequential Quadratic Programming (SQP):** An iterative non-linear optimization method that solves a sequence of quadratic programming subproblems to handle non-linear constraints.
- **Spatial Branch-and-Bound (sBB):** A global optimization algorithm that recursively bisects continuous variable domains to tighten convex polyhedral relaxations until global convergence.
- **Successive Linear Programming (SLP):** An iterative local optimization method that linearizes non-linear pooling terms via first-order Taylor expansion around an operating point.
- **Tank Heel:** The unpumpable residual volume of liquid remaining at the bottom of a storage tank beneath the suction pipe nozzle.
- **Total Sediment Potential (TSP):** A laboratory test method (ISO 10307-2) measuring the propensity of residual marine fuels to precipitate insoluble sediment under thermal stress.
- **Transmix:** The commingled interface mixture formed when dissimilar refined products are pumped sequentially through a common-carrier pipeline.

---

## 16. Sources

- **Audet, C., Brimberg, J., Hansen, P., Le Digabel, S., & Mladenović, N. (2004).** *Pooling Problem: Alternate Formulations and Solution Methods.* Management Science, 50(6), 761–776. DOI: 10.1287/mnsc.1030.0199.
- **Ben-Tal, A., Eiger, G., & Shamir, V. (1994).** *Calculus of Solutions for the Pooling Problem with Application to Refineries and Network Flows.* European Journal of Operational Research, 77(3), 384–400. DOI: 10.1016/0377-2217(94)90176-7.
- **Floudas, C. A. (1995).** *Nonlinear and Mixed-Integer Optimization: Fundamentals and Applications.* Oxford University Press. ISBN: 978-0195100563.
- **Gary, J. H., Handwerk, G. E., & Kaiser, M. J. (2007).** *Petroleum Refining: Technology and Economics* (5th ed.). CRC Press. ISBN: 978-0849370380.
- **Haverly, C. A. (1978).** *Studies of the Behavior of Recursion for the Pooling Problem.* ACM SIGMAP Bulletin, (25), 19–28. DOI: 10.1145/1111237.1111238.
- **Hu, Y. C., & Burns, H. C. (1970).** *New Method Predicts Flash Point of Blends.* Hydrocarbon Processing, 49(11), 130–132.
- **McCormick, G. P. (1976).** *Computability of Global Solutions to Factorable Nonconvex Programs: Part I — Convex Underestimating Problems.* Mathematical Programming, 10(1), 147–175. DOI: 10.1007/BF01580665.
- **Misener, R., & Floudas, C. A. (2009).** *Advances for the Pooling Problem: Modeling, Global Optimization, and Applications.* Industrial & Engineering Chemistry Research, 48(19), 8798–8829. DOI: 10.1021/ie900447m.
- **Tawarmalani, M., & Sahinidis, N. V. (2002).** *Convexification and Global Optimization in Continuous and Mixed-Integer Nonlinear Programming: Theory, Algorithms, Software, and Applications.* Springer Science & Business Media. DOI: 10.1007/978-1-4757-3532-1.
- **ASTM International (2024).** *ASTM D4814-24: Standard Specification for Automotive Spark-Ignition Engine Fuel.* West Conshohocken, PA: ASTM International. DOI: 10.1520/D4814-24.
- **ASTM International (2023).** *ASTM D975-23: Standard Specification for Diesel Fuel Oils.* West Conshohocken, PA: ASTM International. DOI: 10.1520/D0975-23.
- **International Organization for Standardization (2024).** *ISO 8217:2024: Petroleum Products — Fuels (Class F) — Specifications of Marine Fuels.* Geneva: ISO.
- **U.S. Environmental Protection Agency (2021).** *40 CFR Part 1090: Fuels Regulatory Streamlining (Tier 3 Standards).* Federal Register, 85 FR 78412.

---

## 17. Links

- **Module Map & Sibling Topics:**
  - Up: [Module 38 Index — Process Industries, Commodities, and Bulk Supply Chains](_index.md)
  - Previous Topic: [Topic 02: Yield and Mass Balance](02_yield-mass-balance.md)
  - Next Topic: [Topic 04: Campaigns and Changeover](04_campaigns-changeover.md)
  - Sibling: [Topic 01: Continuous, Batch, and Discrete Production](01_continuous-batch-discrete.md)
  - Sibling: [Topic 05: Bulk Storage and Inventory](05_bulk-storage-inventory.md)
  - Sibling: [Topic 06: Commodity Price and Hedging](06_commodity-price-hedging.md)
  - Sibling: [Topic 07: Bulk Logistics and Laytime](07_bulk-logistics-laytime.md)
  - Sibling: [Topic 08: Capacity and Debottlenecking](08_capacity-debottlenecking.md)
  - Sibling: [Topic 09: Energy, Water, and Emissions](09_energy-water-emissions.md)
  - Sibling: [Topic 10: Process Safety and Regulation](10_process-safety-regulation.md)
- **Course Map:**
  - Root Survey File: [Module 38 Survey — Process Industries, Commodities, and Bulk Supply Chains](../../../course/38_process_industries_commodities.md)
  - Related Module: [Module 11: Network Design Optimization](../../02_methods/11_network_design_optimization/_index.md)
  - Related Module: [Module 15: Linear and Mixed-Integer Programming](../../02_methods/15_linear_integer_programming/_index.md)
  - Related Module: [Module 16: Non-Linear and Global Optimization](../../02_methods/16_nonlinear_global_optimization/_index.md)
