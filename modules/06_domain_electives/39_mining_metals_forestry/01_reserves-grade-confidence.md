# Reserves, Grade, and Confidence: Mineral Resource Estimation and Supply Chain Reconciliation

> Module 39 — Mining, Metals, and Forestry Supply Chains
> Topic 01 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 12 (Inventory Theory), Module 38 (Process Industries, Commodities, and Bulk Supply Chains)
> Version 2026.5

---

## 1. Why this topic exists

In discrete assembly manufacturing, the supply chain begins with qualified suppliers who deliver parts fabricated to rigid engineering drawings, certified by inspection sheets, and held to tolerances of micrometers. In chemical processing, feedstocks arrive via dedicated pipelines or tanker vessels with certified purity levels, known boiling-point curves, and stable thermodynamic properties. Production planners in those industries schedule plants under the working assumption that incoming raw materials match their specification sheets within small, controlled variances.

Mining supply chains operate under the inverse reality. A mining enterprise does not purchase its primary raw material from an external vendor; its primary raw material is a heterogeneous, unseen geological deposit formed millions of years ago by complex hydrothermal, magmatic, or sedimentary processes and buried tens to hundreds of meters beneath the earth's surface. 

The mining supply chain begins with an acute information deficit:
1. **Extreme spatial subsampling:** Even after an intensive multi-million-dollar exploration campaign, the cumulative physical volume of diamond drill core pulled from an orebody represents between $1 \times 10^{-8}$ and $1 \times 10^{-6}$ (one hundred-millionth to one millionth) of the total rock mass that will ultimately be excavated, hauled, milled, and smelted. Every long-term capital allocation decision—from mill sizing and fleet selection to multi-decade customer offtake agreements—rests on an interpolation of that infinitesimal physical sample.
2. **Upstream variance propagation (The Geological Bullwhip):** In mining, upstream variance does not dampen down the value chain; it amplifies. A $10\%$ shortfall in head grade at the primary crusher does not merely reduce concentrate output by $10\%$. Because mineral processing plants (flotation cells, leaching circuits, autoclaves) operate according to non-linear chemical kinetics, lower feed grades reduce flotation recovery rates, alter grinding energy consumption per unit of payable metal, dilute concentrate quality, elevate penalty elements (such as arsenic, bismuth, or antimony), delay vessel loading at the export terminal, and trigger financial demurrage penalties and smelter rejection clauses.
3. **Irreversibility of extraction capital:** Once an open pit is stripped or an underground decline developed, misestimated geological boundaries cannot be re-engineered without catastrophic capital destruction. If a mineralized domain was modeled as continuous when it is in fact structurally faulted or compartmentalized, processing equipment runs starved of ore, unit operating costs escalate exponentially, and capital write-downs follow. The mining industry has written off tens of billions of dollars across major projects over the past two decades due entirely to flawed resource models, inadequate grade control, and uncalibrated modifying factors.

To bridge the gap between subsurface geological uncertainty and commercial supply chain commitments, the global extractive industry relies on two foundational disciplines:
- **International Resource Reporting Codes:** Standardized classification systems (CRIRSCO, JORC, NI 43-101, SEC S-K 1300) that enforce rigorous boundaries between undiscovered mineralization, **Mineral Resources** (geological confidence: Inferred, Indicated, Measured), and **Mineral Reserves** (technically mineable and economically viable: Probable, Proved). These codes mandate that supply chain commitments, capital budgets, and public representations be anchored in demonstrable geostatistical confidence and audited **Modifying Factors**.
- **Mine-to-Mill Reconciliation:** Quantitative mathematical tracking frameworks—formalized by Kenneth Parker (1987) and expanded by modern geostatisticians—that continuously measure mass and metal flow across the mining value chain. By calculating multi-stage reconciliation factors ($F_1$ from long-term block model to blast-hole grade control; $F_2$ from grade control to plant delivery; $F_3$ from plant delivery to final metal accounted), supply chain analysts isolate whether shortfalls stem from geological over-smoothing, blast displacement, haulage misrouting, or plant metallurgical losses.

Supply chain optimization in mining is impossible without mastering the mathematics of spatial continuity (semivariograms), spatial interpolation (ordinary kriging), confidence classification, and variance propagation across the reconciliation chain. This topic establishes that foundation.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Differentiate** between Mineral Resources (Inferred, Indicated, Measured) and Mineral Reserves (Probable, Proved) under the CRIRSCO international reporting framework (JORC 2012, Canadian NI 43-101, and US SEC S-K 1300).
2. **Formulate and fit** experimental spatial semivariograms, extracting nugget effect ($C_0$), partial sill ($C$), total sill ($C_0 + C$), and spatial range ($a$) across spherical, exponential, and Gaussian theoretical models.
3. **Derive and solve** the system of Ordinary Kriging (OK) linear equations to compute the Best Linear Unbiased Estimator (BLUE) weights and Kriging estimation variance for an unsampled Selective Mining Unit (SMU).
4. **Quantify** the Support Effect and Information Effect, using Krige's dispersion variance relation to explain why drillhole composite variances exceed block model variances and why uncorrected selective mining generates severe grade shortfalls.
5. **Apply** the full spectrum of Modifying Factors—operational dilution, contact dilution, mining recovery, metallurgical recovery curves, cut-off grade economics, and ESG constraints—to convert mineral resources into recoverable mineral reserves.
6. **Calculate and diagnose** multi-stage mine-to-mill reconciliation factors ($F_1$, $F_2$, $F_3$, and $F_{\text{total}}$), isolating the physical root causes of tonnage, grade, and contained metal discrepancies between planning models and physical output.
7. **Model** the operational propagation of upstream grade volatility into downstream supply chain buffers, quantifying required Run-of-Mine (ROM) buffer stockpile volumes, rail freight batching flexibility, and export vessel laytime risks.
8. **Audit** technical disclosure reports and mining production budgets to detect over-smoothed block models, unearned resource-to-reserve conversions, and hidden offtake delivery vulnerabilities.

---

## 3. The core mental model

The extraction, transformation, and distribution of mineral commodities follow a continuous pipeline that converts spatial geological uncertainty into commercial contract performance. This transformation requires two coupled models: the **Geological Confidence and Conversion Engine** (which converts raw drillhole assays into marketable reserves via geostatistics and modifying factors) and the **Closed-Loop Reconciliation Architecture** (which tracks physical mass and metal from the pit face to the final customer).

### 3.1 The CRIRSCO Classification Matrix

The Committee for Mineral Reserves International Reporting Standards (CRIRSCO) established the universal standard governing how physical rocks become recognized financial and operational assets. Mineralization is categorized along two orthogonal axes: **increasing geological confidence** (horizontal) and the application of **modifying factors** (vertical).

```text
=============================================================================================================
                  THE CRIRSCO RESOURCE-TO-RESERVE CONVERSION MATRIX
=============================================================================================================

                         EXPLORATION RESULTS
                                  │
                                  ▼
 ┌────────────────────────────────────────────────────────────────────────┐
 │                           MINERAL RESOURCES                            │
 │            (Reasonable Prospects for Eventual Economic Extraction)     │
 │                                                                        │
 │     Increasing Geological Knowledge and Spatial Confidence             │
 │   ────────────────────────────────────────────────────────►            │
 │                                                                        │
 │     ┌──────────────────┐    ┌──────────────────┐    ┌────────────────┐ │
 │     │     INFERRED     │    │    INDICATED     │    │    MEASURED    │ │
 │     │  Low confidence  │    │ Medium confidence│    │ High confidence│ │
 │     │  Cannot convert! │    │  Spatial continuity│  │Confirmed continuity│
 │     └────────┬─────────┘    └────────┬─────────┘    └────────┬───────┘ │
 └──────────────┼───────────────────────┼───────────────────────┼─────────┘
                │                       │                       │
                │ NO CONVERSION         │  CONSIDERATION OF     │  CONSIDERATION OF
                │ PERMITTED!            │  MODIFYING FACTORS:   │  MODIFYING FACTORS:
                │ (Confidence too low   │  - Mining methods     │  - Mining methods
                │  for reserve planning)│  - Processing/metall. │  - Processing/metall.
                │                       │  - Economic/prices    │  - Economic/prices
                │                       │  - Marketing/offtake  │  - Marketing/offtake
                │                       │  - Legal/permitting   │  - Legal/permitting
                │                       │  - Infrastructure     │  - Infrastructure
                │                       │  - Social & ESG       │  - Social & ESG
                │                       │                       │
                │                       ▼                       ▼
 ┌──────────────┴─────────────────────────────────────────────────────────┐
 │                            MINERAL RESERVES                            │
 │                   (Economically Mineable & Extractable)                │
 │                                                                        │
 │                             ┌──────────────────┐    ┌────────────────┐ │
 │                             │     PROBABLE     │    │     PROVED     │ │
 │                             │  MINERAL RESERVE │    │ MINERAL RESERVE│ │
 │                             │                  │    │ (Highest level │ │
 │                             │ (From Indicated  │    │  of confidence;│ │
 │                             │  or lower-conf.  │    │  from Measured)│ │
 │                             │  Measured)       │    │                │ │
 │                             └──────────────────┘    └────────────────┘ │
 └────────────────────────────────────────────────────────────────────────┘
=============================================================================================================
```

> **Critical Rule of Mineral Supply Chains:** Under CRIRSCO, JORC, NI 43-101, and SEC S-K 1300, **Inferred Resources can NEVER be converted directly into Mineral Reserves**. Inferred resources carry an estimation variance so high that they cannot form the basis of a definitive mine plan, capital expenditure commitment, or commercial supply contract.

### 3.2 Spatial Continuity and the Semivariogram

Geological grade does not vary randomly like white noise; nor is it uniform like manufactured sheet metal. It exhibits **spatial auto-correlation**: two rock samples taken 2 meters apart are far more likely to carry identical grades than two samples taken 100 meters apart. 

The mathematical tool used to model this spatial correlation is the **semivariogram** $\gamma(h)$, which plots the average squared difference between sample grades as a function of the spatial separation vector (lag distance $h$):

```text
=============================================================================================================
                              ANATOMY OF THE EXPERIMENTAL SEMIVARIOGRAM
=============================================================================================================
 Semivariance γ(h)
    ^
    │
    │                                                   Total Sill (C0 + C) = Population Variance σ²
    │                                              ─────────────────────────────────────────────────
    │                                        *   *   *   *   *   *   *   *   *   *   *   *
    │                                 *   *
    │                             *
    │                        *
    │                   *   ◄── Partial Sill (C)
    │              *
    │          *
    │      *
    │  *
 C0 ┼ ◄── Nugget Effect (Sampling error, micro-scale heterogeneity at h -> 0)
    │
    └───────────────────────┼───────────────────────────────────────────────────────────────►
   0                        a (Spatial Range of Correlation)                           Lag Distance (h, m)

=============================================================================================================
```

- **Nugget Effect ($C_0$):** Semivariance as lag distance approaches zero ($h \to 0$). Represents measurement error, sample preparation noise, and micro-scale mineralization variability occurring at distances smaller than the closest drill spacing.
- **Partial Sill ($C$):** The spatially structured variance component.
- **Total Sill ($C_0 + C$):** The plateau where semivariance levels off. At distances beyond the range, spatial correlation drops to zero, and the semivariance equals the total population variance $\sigma^2$.
- **Range ($a$):** The separation distance beyond which sample pairs become statistically independent. Samples separated by $h > a$ provide zero spatial correlation for estimation.

### 3.3 The Parker Mine-to-Mill-to-Market Reconciliation Chain

Once the long-term resource block model is estimated via Ordinary Kriging, mining operations begin. Material flows through four sequential evaluation gates. Each gate introduces physical mass losses, unplanned dilution, and measurement errors. 

The Parker (1987) reconciliation framework tracks performance across three multiplicative ratios:

```text
=============================================================================================================
                        THE PARKER MINE-TO-MILL RECONCILIATION CHAIN
=============================================================================================================

 ┌────────────────────────────────┐
 │     LONG-TERM RESERVE MODEL    │  Planned tonnes, grade, contained metal
 │   (Ordinary Kriging, Block)    │  Basis of Life-of-Mine (LOM) production schedule
 └────────────────┬───────────────┘
                  │
                  ▼  [ Reconciliation Factor F1 ]  =  (Grade Control Model / Reserve Model)
 ┌────────────────────────────────┐
 │   GRADE CONTROL BLAST-HOLE     │  Short-term infill drilling (6m x 6m pattern)
 │             MODEL              │  Ore/Waste dig-line delineation polygons
 └────────────────┬───────────────┘
                  │
                  ▼  [ Reconciliation Factor F2 ]  =  (Mill Received / Grade Control Dispatched)
 ┌────────────────────────────────┐
 │     MINED & DELIVERED ORE      │  Excavator loading, haul truck dispatch telemetry
 │     (To Primary Crusher/ROM)   │  Stockpile blending buffers, contact dilution
 └────────────────┬───────────────┘
                  │
                  ▼  [ Reconciliation Factor F3 ]  =  (Accounted Metal / Mill Head Feed Metal)
 ┌────────────────────────────────┐
 │    MILL METALLURGICAL BALANCE  │  Primary crusher weightometers, online cross-belt analyzers
 │    (Concentrator / Leaching)   │  Slurry density, auto-samplers, flotation recovery
 └────────────────┬───────────────┘
                  │
                  ▼  [ Total Reconciliation F_total ]  =  F1 x F2 x F3
 ┌────────────────────────────────┐
 │   COMMERCIAL PRODUCT OFFTAKE   │  Clean concentrate / refined cathode delivered to port
 │   (Customer Deliveries / Port) │  Offtake specification compliance, zero demurrage
 └────────────────────────────────┘
=============================================================================================================
```

- **$F_1$ Factor (Geological and Estimation Bias):** Compares the short-term grade control model (derived from dense blast-hole drilling) to the long-term reserve block model (derived from sparse exploration diamond drilling). Evaluates whether the resource geologist over-smoothed the deposit or misidentified structural boundaries.
- **$F_2$ Factor (Mining Execution and Dilution):** Compares the ore received at the processing plant to the ore predicted by grade control. Evaluates excavator operator digging accuracy, blast movement displacement, truck misrouting to waste dumps, and stockpile inventory tracking errors.
- **$F_3$ Factor (Metallurgical and Measurement Performance):** Compares the final metal recovered in concentrate plus metal lost in tailings to the head metal claimed to have entered the plant. Evaluates weightometer calibration, slurry auto-sampling bias, and processing circuit recovery kinetics.
- **$F_{\text{total}} = F_1 \times F_2 \times F_3$:** The net reconciliation factor. If $F_{\text{total}} < 1.0$, the supply chain experiences an unbudgeted metal shortfall, threatening commercial delivery schedules.

---

## 4. Definitions and vocabulary

| Term | Definition | Common Misuse | Unit / Code |
|---|---|---|---|
| **Mineral Resource** | A concentration or occurrence of solid material of economic interest in such form, grade, and quantity that there are reasonable prospects for eventual economic extraction. | Conflating with "Reserve"; treating resources as immediately extractable ore. | Tonnes ($t$) and Grade ($\%$, $\text{g/t}$) |
| **Mineral Reserve (Ore Reserve)** | The economically mineable part of a Measured and/or Indicated Mineral Resource, inclusive of dilution and allowances for losses incurred during mining. | Reporting reserves without demonstrating modifying factors or economic feasibility. | Tonnes ($t$) and Grade ($\%$, $\text{g/t}$) |
| **Competent Person (CP) / Qualified Person (QP)** | A minerals industry professional who is a member of an authorized professional organization with enforceable disciplinary powers and has at least five years of relevant experience in the style of mineralization and activity undertaken. | Assuming any registered engineer or geologist qualifies regardless of commodity experience. | Statutory designation (JORC / NI 43-101 / S-K 1300) |
| **CRIRSCO** | Committee for Mineral Reserves International Reporting Standards; the global umbrella body aligning national reporting codes (JORC, CIM/NI 43-101, SEC S-K 1300, SAMREC, PERC). | Treating CRIRSCO as a regulatory body rather than a standards harmonizer. | International Template |
| **Inferred Mineral Resource** | Part of a resource for which quantity and grade are estimated from limited geological evidence and sampling; geological and grade continuity are implied but not verified. | Including Inferred resources in mine production schedules or reserve statements. | Category 3 Confidence |
| **Indicated Mineral Resource** | Part of a resource for which quantity, grade, and physical characteristics are estimated with sufficient confidence to allow application of modifying factors in sufficient detail to support mine planning. | Assuming Indicated implies guaranteed profitability without modifying factor evaluation. | Category 2 Confidence |
| **Measured Mineral Resource** | Part of a resource for which quantity, grade, and physical characteristics are estimated with confidence high enough to allow application of modifying factors to support detailed mine planning and final economic evaluation. | Assuming Measured implies zero spatial error; estimation variance is minimized but non-zero. | Category 1 Confidence |
| **Probable Mineral Reserve** | The economically mineable part of an Indicated (and in some circumstances Measured) Mineral Resource. | Lowering confidence to Inferred when modifying factors are uncertain. | Category 2 Reserve |
| **Proved (Proven) Mineral Reserve** | The economically mineable part of a Measured Mineral Resource, representing the highest confidence category of reserve. | Labeling reserves as Proved when modifying factors carry unresolved permitting or legal risks. | Category 1 Reserve |
| **Modifying Factors** | Considerations used to convert Mineral Resources to Mineral Reserves: mining, processing, metallurgical, infrastructure, economic, marketing, legal, environmental, social, and governmental factors. | Treating modifying factors as a simple percentage haircut rather than an engineered audit. | Multi-disciplinary criteria |
| **Cut-Off Grade (COG)** | The lowest grade of mineralized material that qualifies as economically viable ore; material above COG is processed, material below COG is sent to waste or low-grade stockpiles. | Assuming COG is constant; COG fluctuates dynamically with metal prices, recovery, and operating costs. | $\%$ (base metals), $\text{g/t}$ (precious metals) |
| **Semivariogram $\gamma(h)$** | A mathematical function measuring half the average squared difference between sample values separated by spatial distance vector $h$. | Confusing semivariogram with the covariance function or standard variance. | $(\text{grade units})^2$, e.g., $\%^2$, $(\text{g/t})^2$ |
| **Nugget Effect ($C_0$)** | The vertical discontinuity of the semivariogram at $h \to 0$, representing sampling error and micro-scale geological noise. | Assuming $C_0 = 0$; ignoring nugget effect leads to severe over-confidence in local estimates. | $(\text{grade units})^2$ |
| **Partial Sill ($C$)** | The variance increment attributable to spatial structure above the nugget effect. | Confusing partial sill with total sill ($C_0 + C$). | $(\text{grade units})^2$ |
| **Total Sill ($C_0 + C$)** | The asymptotic plateau of the semivariogram, equal to the a priori variance of the spatial domain. | Fitting a sill that exceeds or falls far short of the declustered sample variance. | $(\text{grade units})^2$ |
| **Spatial Range ($a$)** | The separation distance beyond which samples are no longer spatially correlated. | Using search radii larger than the range in ordinary kriging without geostatistical justification. | Meters ($\text{m}$) |
| **Ordinary Kriging (OK)** | A linear spatial estimation method that calculates the best linear unbiased estimate (BLUE) of an unsampled point or block using a weighted linear combination of surrounding samples. | Claiming kriging "predicts the exact grade"; kriging calculates an expected mean and estimation variance. | Spatial algorithm |
| **Kriging Estimation Variance ($\sigma_{OK}^2$)** | The calculated minimum variance of estimation error produced by the kriging system; depends solely on spatial geometry and variogram, not sample values. | Equating kriging variance with true error; it measures configuration geometry, not data outlier spread. | $(\text{grade units})^2$ |
| **Support Effect** | The mathematical and physical phenomenon where the variance of grades measured on large volumes (blocks) is strictly smaller than the variance of grades measured on small volumes (drill cores). | Using raw drill core histograms directly to forecast block-scale cut-off grade recoveries. | Volume-variance law |
| **Information Effect** | The inevitable misclassification of ore and waste during mining because selection is based on imperfect, sample-estimated grades rather than true block grades. | Assuming zero operational dilution and perfect ore-waste discrimination at the pit face. | Classification error |
| **Selective Mining Unit (SMU)** | The smallest physical volume of rock that mining equipment (excavator, haul truck) can separate into ore or waste during production. | Setting block model dimensions smaller than equipment capabilities (e.g., $2\text{m}$ blocks for a $40\text{m}^3$ shovel). | Dimensions ($L \times W \times H, \text{ m}$) |
| **Run-of-Mine (ROM)** | Raw, uncrushed ore excavated from the pit or underground workings and delivered directly to the primary crusher or surface stockpiles. | Treating ROM as uniform feedstock; ROM exhibits severe short-term grade and hardness fluctuations. | Bulk tonnes ($t$) |
| **$F_1$ Reconciliation Factor** | The ratio of ore tonnes, grade, or metal estimated by short-term grade control blast-hole models to that predicted by the long-term resource block model for the same excavated volume. | Blaming mining equipment operators for $F_1$ failures; $F_1$ measures geological modeling bias. | Dimensionless ratio |
| **$F_2$ Reconciliation Factor** | The ratio of ore tonnes, grade, or metal received at the processing plant to that predicted by the short-term grade control model. | Assuming $F_2$ is purely haulage loss; $F_2$ captures dig-line boundary errors and blast heave. | Dimensionless ratio |
| **$F_3$ Reconciliation Factor** | The ratio of final accounted metal (metal recovered in concentrate plus metal measured in tailings) to the head metal claimed to have entered the plant. | Forcing plant accounting to balance by adjusting head grade artificially; $F_3$ measures plant assay integrity. | Dimensionless ratio |
| **Total Reconciliation ($F_{\text{total}}$)** | The overall product $F_{\text{total}} = F_1 \times F_2 \times F_3$, comparing final accounted commercial metal to the original long-term reserve block model. | Analyzing $F_{\text{total}}$ without decomposing into $F_1, F_2, F_3$; conceals compensating errors. | Dimensionless ratio |
| **Unplanned Dilution** | Waste or sub-economic material below the cut-off grade that is unintentionally mined, hauled, and fed to the plant alongside economic ore. | Believing dilution contains zero metal; wall rock dilution often carries low background grades. | $\%$ of total feed mass |
| **Mining Recovery** | The percentage of economic ore tonnes and metal within a designated reserve block that is successfully extracted and delivered to the processing facility. | Assuming mining recovery is $100\%$; ore left in pit floors, hanging walls, or misrouted reduces recovery. | $\%$ |

---

## 5. The mechanics, step by step

Developing a compliant mineral reserve estimate, establishing geostatistical confidence, and executing mine-to-mill reconciliation follows an unbroken eight-step engineering procedure:

```text
=============================================================================================================
                          THE EIGHT-STEP RESERVES-TO-RECONCILIATION WORKFLOW
=============================================================================================================

 [Step 1] Exploration Drilling & QA/QC  ──► [Step 2] Spatial Semivariogram Modeling
                   │                                              │
                   ▼                                              ▼
 [Step 3] Ordinary Kriging Estimation   ──► [Step 4] Support Effect & Change of Support
                   │                                              │
                   ▼                                              ▼
 [Step 5] Resource Classification       ──► [Step 6] Modifying Factors & Reserve Conversion
                   │                                              │
                   ▼                                              ▼
 [Step 7] Grade Control & Dig-Lines     ──► [Step 8] Multi-Stage Reconciliation (F1, F2, F3)
=============================================================================================================
```

### Step 1: Exploration Drilling, Sampling QA/QC, and Compositing

Resource estimation begins with drilling. Geologists drill diamond core (DD) or reverse circulation (RC) holes into the subsurface. 

1. **Core Logging and Splitting:** Diamond drill core is extracted, photographed, geologically logged (lithology, alteration, structural fabric, mineralogy), and sawn longitudinally in half. One half is archived for audit; the other half is crushed and pulverized for chemical assay.
2. **Quality Assurance / Quality Control (QA/QC):** To comply with JORC and NI 43-101 standards, sampling campaigns must insert blind quality control samples at a frequency of at least $10\%$:
   - **Certified Reference Materials (CRMs / Standards):** Pre-packaged matrix-matched pulverized samples of certified grade inserted to detect assay laboratory bias. Bias exceeding $\pm 2$ standard deviations triggers investigation; exceeding $\pm 3$ standard deviations requires re-assaying the batch.
   - **Coarse Blanks:** Unmineralized barren rock inserted immediately following high-grade intercepts to detect cross-contamination in crushers and pulverizers. Contamination $>3 \times$ detection limit requires cleaning protocol audits.
   - **Field and Pulp Duplicates:** Duplicate core halves and duplicate pulp splits sent to primary and secondary umpire laboratories to quantify sampling precision and fundamental sampling error (FSE).
3. **Assay Compositing:** Raw drill intervals vary in length based on geological boundaries. Because geostatistical analysis requires equal sample support (constant volume), raw assay intervals are regularized into fixed-length **composites** (typically $1.0\text{ m}$, $2.0\text{ m}$, or bench height):
   $$Z_{\text{comp}} = \frac{\sum_{i=1}^k l_i \cdot \rho_i \cdot Z_i}{\sum_{i=1}^k l_i \cdot \rho_i}$$
   where $l_i$ is interval length, $\rho_i$ is rock bulk density, and $Z_i$ is raw assay grade. If density is constant, length-weighting suffices.

### Step 2: Spatial Continuity Analysis and Semivariogram Modeling

To quantify how grade correlates across space, geologists calculate the **experimental semivariogram** $\hat{\gamma}(h)$ along major geological axes (strike, dip, plunge):

$$\hat{\gamma}(h) = \frac{1}{2 N(h)} \sum_{i=1}^{N(h)} \left[ Z(x_i) - Z(x_i + h) \right]^2$$

where:
- $h$ is the spatial lag distance (with directional tolerance angle $\theta$ and lag tolerance $\Delta h$).
- $N(h)$ is the number of paired sample points separated by distance $h$.
- $Z(x_i)$ and $Z(x_i + h)$ are the composite grades at spatial locations $x_i$ and $x_i + h$.

Because experimental values represent discrete points subject to sampling scatter, a continuous, conditionally negative definite theoretical mathematical model must be fitted to the experimental points. The three standard authorized models are:

1. **Spherical Model (Most common in mineral deposits):**
   $$\gamma(h) = \begin{cases} 0 & \text{for } h = 0 \\ C_0 + C \left[ \frac{3}{2} \left(\frac{h}{a}\right) - \frac{1}{2} \left(\frac{h}{a}\right)^3 \right] & \text{for } 0 < h \le a \\ C_0 + C & \text{for } h > a \end{cases}$$
2. **Exponential Model (Steep gradient at origin, asymptotic reach):**
   $$\gamma(h) = C_0 + C \left[ 1 - \exp\left(-\frac{3h}{a}\right) \right]$$
   (where $a$ represents the practical range reaching $95\%$ of the sill).
3. **Gaussian Model (Extremely smooth spatial continuity; used for porphyry core zones or coal seam thicknesses):**
   $$\gamma(h) = C_0 + C \left[ 1 - \exp\left(-\frac{3h^2}{a^2}\right) \right]$$

**Directional Anisotropy:** Deposits rarely display identical continuity in all directions. Hydrothermal veins or sedimentary layers exhibit longer ranges along strike ($a_{\text{strike}}$) and dip ($a_{\text{dip}}$) than across thickness ($a_{\text{thickness}}$). An **anisotropy ellipsoid** is established via coordinate transformation using rotation matrices.

### Step 3: Block Model Construction and Ordinary Kriging Estimation

The 3D deposit volume is discretized into a regular grid of **Selective Mining Units (SMUs)** (e.g., $10\text{ m} \times 10\text{ m} \times 10\text{ m}$ blocks). For each block centroid $x_0$, the grade $Z^*(x_0)$ is estimated as a linear weighted combination of $n$ neighboring composite samples:

$$Z^*(x_0) = \sum_{i=1}^n \lambda_i Z(x_i)$$

To guarantee that the estimator is the **Best Linear Unbiased Estimator (BLUE)**:
1. **Unbiasedness Condition:** The expected error must be zero: $E[Z^*(x_0) - Z(x_0)] = 0$, requiring the weights to sum strictly to unity:
   $$\sum_{i=1}^n \lambda_i = 1$$
2. **Minimum Variance Condition:** The estimation error variance $\sigma_E^2 = \text{Var}(Z^*(x_0) - Z(x_0))$ must be minimized. Applying Lagrange multipliers ($\mu$) to minimize $\sigma_E^2$ subject to $\sum \lambda_i = 1$ yields the **Ordinary Kriging System of Linear Equations**:

$$\sum_{j=1}^n \lambda_j \gamma(x_i, x_j) + \mu = \gamma(x_i, x_0) \quad \forall i \in \{1, 2, \dots, n\}$$
$$\sum_{i=1}^n \lambda_i = 1$$

In matrix notation:

$$\begin{bmatrix}
\gamma(x_1, x_1) & \gamma(x_1, x_2) & \dots & \gamma(x_1, x_n) & 1 \\
\gamma(x_2, x_1) & \gamma(x_2, x_2) & \dots & \gamma(x_2, x_n) & 1 \\
\vdots & \vdots & \ddots & \vdots & \vdots \\
\gamma(x_n, x_1) & \gamma(x_n, x_2) & \dots & \gamma(x_n, x_n) & 1 \\
1 & 1 & \dots & 1 & 0
\end{bmatrix}
\begin{bmatrix}
\lambda_1 \\ \lambda_2 \\ \vdots \\ \lambda_n \\ \mu
\end{bmatrix}
=
\begin{bmatrix}
\gamma(x_1, x_0) \\ \gamma(x_2, x_0) \\ \vdots \\ \gamma(x_n, x_0) \\ 1
\end{bmatrix}$$

Solving $\mathbf{K} \boldsymbol{\lambda} = \mathbf{k}$ yields the optimal kriging weights $\lambda_1, \dots, \lambda_n$ and Lagrange multiplier $\mu$.

The **Kriging Estimation Variance** $\sigma_{OK}^2(x_0)$ is then:

$$\sigma_{OK}^2(x_0) = \sum_{i=1}^n \lambda_i \gamma(x_i, x_0) + \mu$$

> **Key Insight:** Notice that $\sigma_{OK}^2$ depends exclusively on the spatial geometry of the samples relative to the target block ($\gamma(x_i, x_j)$ and $\gamma(x_i, x_0)$) and the semivariogram parameters ($C_0, C, a$). It does not depend on the actual grade values $Z(x_i)$. Kriging variance measures geometric data configuration and spatial continuity, not local grade volatility.

### Step 4: The Support Effect and Change of Support

A diamond drill core is a cylinder $5\text{ cm}$ in diameter; an SMU block is $1,000\text{ m}^3$ ($2,700\text{ tonnes}$); an open pit bench is $500,000\text{ tonnes}$. 

Under Krige's relation, the dispersion variance of small units $v$ within a larger volume $V$ satisfies:

$$D^2(v/V) = D^2(v/D) - D^2(V/D)$$

Because larger volumes average out extreme peaks and troughs:
- The variance of block grades $\sigma_{\text{block}}^2$ is strictly smaller than the variance of sample grades $\sigma_{\text{sample}}^2$.
- The distribution of block grades is narrower and more symmetrical than the highly skewed sample distribution.

```text
=============================================================================================================
                  THE SUPPORT EFFECT: DRILL CORES VS. SELECTIVE MINING UNITS
=============================================================================================================
 Relative Frequency
    ^
    │          Sample Grade Distribution (Drill core composites: High variance, extreme tails)
    │          ─────────────────────────────────────────────────────────────────────────────
    │          Block Grade Distribution (SMU blocks: Averaged, lower variance, narrower)
    │          - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    │
    │                           - - -
    │                         -   |   -
    │                       -     |     -
    │                     -       |       -
    │                   -         |         -
    │                 /           |           \
    │               /  -          |          -  \
    │             /   -           |           -   \
    │           /    -            |            -    \
    │         /     -             |             -     \
    │       /      -              |              -      \
    │     /       -               |               -       \
    └────┴───────┴────────────────┼────────────────┴───────┴───────────────────────────────►
        0                        Cut-Off Grade (COG)                               Grade (% Cu)
         ◄────── Waste ──────────►◄────────────────── Ore ────────────────────────►
=============================================================================================================
```

If a mine planning engineer evaluates recoverable tonnes and grade above a cut-off grade by applying the cut-off directly to raw drillhole sample distributions, they will severely overestimate recovered grade and underestimate ore tonnage. Geostatisticians must apply **Change of Support** corrections (such as the Affine Correction, Indirect Lognormal Correction, or Discrete Gaussian Model) to estimate the true distribution of SMU block grades prior to scheduling.

### Step 5: Resource Classification Workflow

Once all blocks are estimated and their kriging variances computed, the deposit is classified into CRIRSCO resource categories:

1. **Measured Mineral Resource:**
   - Samples originate from at least 3 or 4 independent drillholes within a search radius well inside the semivariogram range ($r \le \frac{1}{2}a$ or $\frac{1}{3}a$).
   - High data density (e.g., $25\text{ m} \times 25\text{ m}$ drilling pattern).
   - Kriging Efficiency $KE > 0.70$ and Slope of Regression $ZZ^* > 0.80$, where:
     $$KE = \frac{\sigma_{\text{block}}^2 - \sigma_{OK}^2}{\sigma_{\text{block}}^2}, \quad \text{Slope } ZZ^* \approx \frac{\sigma_{\text{block}}^2 - \sigma_{OK}^2 + |\mu|}{\sigma_{\text{block}}^2 - \sigma_{OK}^2 + 2|\mu|}$$
   - Geological continuity of structure and grade confirmed beyond reasonable doubt.
2. **Indicated Mineral Resource:**
   - Samples originate from at least 2 or 3 drillholes within the semivariogram range ($r \le a$).
   - Intermediate drill density (e.g., $50\text{ m} \times 50\text{ m}$ pattern).
   - Kriging Efficiency $0.40 \le KE \le 0.70$; geological continuity assumed with realistic confidence.
   - Sufficient confidence to enable mine planning and modifying factor application.
3. **Inferred Mineral Resource:**
   - Samples from isolated or widely spaced holes ($r > a$ or single-drillhole interpolation).
   - Exploration drill density ($100\text{ m} \times 100\text{ m}$ or greater).
   - Geological continuity implied but not confirmed; cannot convert to reserves.

### Step 6: Modifying Factors and Reserve Conversion

Mineral Resources become Mineral Reserves only through a formal engineering and economic feasibility study that applies **Modifying Factors**:

```text
Mineral Resource (In-Situ In-Pit Block)
    │
    ├── Apply Mining Dilution (Contact Dilution + Blast Heave Dilution)
    ├── Apply Mining Recovery (Ore Loss in Stopes, Pit Floors)
    ├── Apply Metallurgical Recovery Models (Flotation/Leach recovery as f(Grade, Alteration))
    ├── Apply Economic Cut-Off Grade (Mining Cost, Processing Cost, G&A, Realized Metal Price)
    ├── Apply Infrastructure Constraints (Haul Road Geotechnics, Crusher Cavity Throughput)
    └── Apply Environmental, Social, and Legal Permitting (Water extraction, FPIC, Tailings)
    │
    ▼
Mineral Reserve (Proved or Probable) -> Feedstock to Long-Term Supply Chain
```

1. **Mining Dilution Calculation:**
   $$\text{Diluted Tonnes } T_{\text{dil}} = T_{\text{insitu}} \times (1 + \delta_{\text{dil}})$$
   $$\text{Diluted Grade } g_{\text{dil}} = \frac{T_{\text{insitu}} \cdot g_{\text{insitu}} + (\delta_{\text{dil}} \cdot T_{\text{insitu}}) \cdot g_{\text{waste}}}{T_{\text{dil}}}$$
   where $\delta_{\text{dil}}$ is the dilution percentage and $g_{\text{waste}}$ is the grade of adjacent wall rock entering the muckpile.
2. **Mining Recovery Factor:**
   $$\text{Recovered Ore Tonnes } T_{\text{rec}} = T_{\text{dil}} \times \eta_{\text{mine}}$$
   where $\eta_{\text{mine}}$ (typically $92\% - 98\%$) accounts for unrecoverable ore left on pit benches or underground stope walls.
3. **Economic Cut-Off Grade (Breakeven):**
   $$COG_{\text{breakeven}} = \frac{C_{\text{process}} + C_{\text{G\&A}} + C_{\text{transport}}}{(P_{\text{metal}} - C_{\text{refine}}) \times R_{\text{metallurgical}}}$$
   where $C$ denotes unit operating costs per tonne, $P_{\text{metal}}$ is realized commodity price, and $R_{\text{metallurgical}}$ is the recovery fraction.

### Step 7: Grade Control and Blast-Hole Delineation

During production, the mine excavates benches in 10-meter to 15-meter lifts. Long-term exploration drillholes ($50\text{m}$ apart) are replaced by high-density **blast-hole drilling** ($5\text{m} \times 5\text{m}$ to $7\text{m} \times 7\text{m}$ pattern).

1. **Blast-Hole Sampling:** Cuttings from rotary blast-hole drills are sampled using riffle splitters or automated sampling cones.
2. **Short-Term Grade Control Model:** Blast-hole assays are modeled on high-resolution grids ($2\text{m} \times 2\text{m}$).
3. **Dig-Line Generation:** Mine geologists draw geometric polygons ("dig-lines") separating:
   - High-grade ore (direct feed to crusher).
   - Medium-grade ore (feed to blending stockpiles).
   - Low-grade ore (feed to long-term sub-grade stockpiles).
   - Waste rock (routed to external waste rock dumps).
4. **Blast Movement Tracking:** Detonation displaces rock horizontally by $2\text{ to } 15\text{ meters}$. If dig-lines are not dynamically translated using blast-movement monitors (BMM transponders), excavators will shovel high-grade ore into the waste dump and send barren wall rock to the primary crusher.

### Step 8: Multi-Stage Mine-to-Mill Reconciliation (Parker Framework)

Reconciliation operates as an audit loop across monthly, quarterly, and annual horizons. The Parker framework defines three distinct reconciliation ratios across tonnage ($T$), grade ($g$), and contained metal ($M = T \times g$):

1. **$F_1$ Reconciliation (Grade Control vs. Long-Term Reserve Block Model):**
   $$F_{1, T} = \frac{T_{\text{GC}}}{T_{\text{Model}}}, \quad F_{1, g} = \frac{g_{\text{GC}}}{g_{\text{Model}}}, \quad F_{1, M} = \frac{M_{\text{GC}}}{M_{\text{Model}}} = F_{1, T} \times F_{1, g}$$
   - Evaluates: Resource modeling accuracy, interpolation bias, and long-term geological interpretation.
   - Diagnostic Threshold: $F_{1, M} \in [0.95, 1.05]$ indicates a calibrated reserve block model. $F_{1, M} < 0.90$ signals severe geological over-estimation or unmodeled structural discontinuities.
2. **$F_2$ Reconciliation (Ore Delivered to Plant vs. Grade Control Model):**
   $$F_{2, T} = \frac{T_{\text{Delivered}}}{T_{\text{GC}}}, \quad F_{2, g} = \frac{g_{\text{Delivered}}}{g_{\text{GC}}}, \quad F_{2, M} = \frac{M_{\text{Delivered}}}{M_{\text{GC}}} = F_{2, T} \times F_{2, g}$$
   - Evaluates: Mining execution, blast movement control, excavator digging precision, haul truck dispatch routing, and ROM stockpile inventory tracking.
   - Diagnostic Threshold: $F_{2, T} > 1.05$ accompanied by $F_{2, g} < 0.92$ is the universal signature of **excessive operational dilution** (mining past dig-line boundaries).
3. **$F_3$ Reconciliation (Metallurgical Accounted Metal vs. Received Mill Head Feed):**
   $$F_{3, M} = \frac{M_{\text{Product Recovered}} + M_{\text{Tailings Measured}}}{M_{\text{Delivered Claimed}}}$$
   - Evaluates: Primary crusher belt-scale calibration, slurry auto-sampling integrity, online metallurgical balancing, and plant assay bias.
   - Diagnostic Threshold: $F_{3, M} \in [0.98, 1.02]$. Persistent values $<0.97$ indicate systematic over-reporting of incoming head grade or unmeasured plant slurry leaks.
4. **Total Net Reconciliation ($F_{\text{total}}$):**
   $$F_{\text{total}, M} = F_{1, M} \times F_{2, M} \times F_{3, M} = \frac{\text{Final Accounted Metal}}{\text{Original Reserve Model Metal}}$$
   - If $F_{\text{total}, M} < 1.0$, the enterprise produces less marketable commodity than promised in long-term mine plans, directly undermining debt covenants and customer offtake schedules.

---

## 6. Worked example

### 6.1 The Problem Setting: Kopano Copper Mine

The Kopano Copper Mine is an open-pit porphyry operation in the Zambian Copperbelt. We examine a specific production block on Bench 1420 (Block B-12). 

A new Selective Mining Unit (SMU) block centroid is located at coordinates $x_0 = (50.0\text{ m}, 50.0\text{ m})$. The block has dimensions $25.0\text{ m} \times 25.0\text{ m} \times 15.0\text{ m}$ (bench height). Average rock density is $\rho = 2.70\text{ t/m}^3$.

Four exploration diamond drill core composites surround block centroid $x_0$:

| Sample ID | Easting $x$ (m) | Northing $y$ (m) | Copper Grade $Z(x)$ (% Cu) |
|---|---|---|---|
| **$S_1$** | 30.0 | 40.0 | 0.850 |
| **$S_2$** | 70.0 | 30.0 | 0.620 |
| **$S_3$** | 75.0 | 65.0 | 0.540 |
| **$S_4$** | 25.0 | 75.0 | 0.780 |

The deposit-wide variogram in this structural domain follows an isotropic **Spherical Model**:
- Nugget effect: $C_0 = 0.040\%^2$
- Partial sill: $C = 0.160\%^2$
- Total sill: $C_0 + C = 0.200\%^2$
- Range of spatial correlation: $a = 120.0\text{ m}$

$$\gamma(h) = \begin{cases} 0 & h = 0 \\ 0.040 + 0.160 \left[ 1.5 \left(\frac{h}{120}\right) - 0.5 \left(\frac{h}{120}\right)^3 \right] & 0 < h \le 120 \\ 0.200 & h > 120 \end{cases}$$

### 6.2 Part 1: Calculating Distances and Semivariances

First, we calculate the Euclidean distance $D_{ij} = \sqrt{(x_i - x_j)^2 + (y_i - y_j)^2}$ between each pair of sample points:
- $D_{12} = \sqrt{(70-30)^2 + (30-40)^2} = \sqrt{1600 + 100} = \sqrt{1700} \approx 41.23\text{ m}$
- $D_{13} = \sqrt{(75-30)^2 + (65-40)^2} = \sqrt{2025 + 625} = \sqrt{2650} \approx 51.48\text{ m}$
- $D_{14} = \sqrt{(25-30)^2 + (75-40)^2} = \sqrt{25 + 1225} = \sqrt{1250} \approx 35.36\text{ m}$
- $D_{23} = \sqrt{(75-70)^2 + (65-30)^2} = \sqrt{25 + 1225} = \sqrt{1250} \approx 35.36\text{ m}$
- $D_{24} = \sqrt{(25-70)^2 + (75-30)^2} = \sqrt{2025 + 2025} = \sqrt{4050} \approx 63.64\text{ m}$
- $D_{34} = \sqrt{(25-75)^2 + (75-65)^2} = \sqrt{2500 + 100} = \sqrt{2600} \approx 50.99\text{ m}$

Next, we calculate the distance $d_{i0}$ from each sample point to target centroid $x_0 = (50.0, 50.0)$:
- $d_{10} = \sqrt{(50-30)^2 + (50-40)^2} = \sqrt{400 + 100} = \sqrt{500} \approx 22.36\text{ m}$
- $d_{20} = \sqrt{(50-70)^2 + (50-30)^2} = \sqrt{400 + 400} = \sqrt{800} \approx 28.28\text{ m}$
- $d_{30} = \sqrt{(50-75)^2 + (50-65)^2} = \sqrt{625 + 225} = \sqrt{850} \approx 29.15\text{ m}$
- $d_{40} = \sqrt{(50-25)^2 + (50-75)^2} = \sqrt{625 + 625} = \sqrt{1250} \approx 35.36\text{ m}$

Applying the spherical variogram formula $\gamma(h) = 0.040 + 0.160 \left[ 1.5\left(\frac{h}{120}\right) - 0.5\left(\frac{h}{120}\right)^3 \right]$:
- $\gamma(D_{11}) = \gamma(D_{22}) = \gamma(D_{33}) = \gamma(D_{44}) = 0.0000$
- $\gamma(D_{12}) = 0.040 + 0.160 [1.5(41.23/120) - 0.5(41.23/120)^3] = 0.040 + 0.160 [0.5154 - 0.0202] = 0.1192$
- $\gamma(D_{13}) = 0.040 + 0.160 [1.5(51.48/120) - 0.5(51.48/120)^3] = 0.040 + 0.160 [0.6435 - 0.0394] = 0.1366$
- $\gamma(D_{14}) = 0.040 + 0.160 [1.5(35.36/120) - 0.5(35.36/120)^3] = 0.040 + 0.160 [0.4420 - 0.0128] = 0.1087$
- $\gamma(D_{23}) = 0.040 + 0.160 [1.5(35.36/120) - 0.5(35.36/120)^3] = 0.1087$
- $\gamma(D_{24}) = 0.040 + 0.160 [1.5(63.64/120) - 0.5(63.64/120)^3] = 0.040 + 0.160 [0.7955 - 0.0746] = 0.1553$
- $\gamma(D_{34}) = 0.040 + 0.160 [1.5(50.99/120) - 0.5(50.99/120)^3] = 0.040 + 0.160 [0.6374 - 0.0383] = 0.1358$

For the right-hand vector $\gamma(x_i, x_0)$:
- $\gamma(d_{10}) = 0.040 + 0.160 [1.5(22.36/120) - 0.5(22.36/120)^3] = 0.040 + 0.160 [0.2795 - 0.0032] = 0.0842$
- $\gamma(d_{20}) = 0.040 + 0.160 [1.5(28.28/120) - 0.5(28.28/120)^3] = 0.040 + 0.160 [0.3535 - 0.0065] = 0.0955$
- $\gamma(d_{30}) = 0.040 + 0.160 [1.5(29.15/120) - 0.5(29.15/120)^3] = 0.040 + 0.160 [0.3644 - 0.0072] = 0.0972$
- $\gamma(d_{40}) = 0.040 + 0.160 [1.5(35.36/120) - 0.5(35.36/120)^3] = 0.1087$

### 6.3 Part 2: Ordinary Kriging Matrix Formulation and Solution

The Ordinary Kriging system $\mathbf{A} \mathbf{w} = \mathbf{b}$ is expressed as:

$$\begin{bmatrix}
0.0000 & 0.1192 & 0.1366 & 0.1087 & 1.0000 \\
0.1192 & 0.0000 & 0.1087 & 0.1553 & 1.0000 \\
0.1366 & 0.1087 & 0.0000 & 0.1358 & 1.0000 \\
0.1087 & 0.1553 & 0.1358 & 0.0000 & 1.0000 \\
1.0000 & 1.0000 & 1.0000 & 1.0000 & 0.0000
\end{bmatrix}
\begin{bmatrix}
\lambda_1 \\ \lambda_2 \\ \lambda_3 \\ \lambda_4 \\ \mu
\end{bmatrix}
=
\begin{bmatrix}
0.0842 \\ 0.0955 \\ 0.0972 \\ 0.1087 \\ 1.0000
\end{bmatrix}$$

Solving this linear system yields:
- $\lambda_1 = 0.3265$ (Sample $S_1$, nearest sample at $22.36\text{ m}$, receives largest weight)
- $\lambda_2 = 0.2440$ (Sample $S_2$, distance $28.28\text{ m}$)
- $\lambda_3 = 0.2488$ (Sample $S_3$, distance $29.15\text{ m}$)
- $\lambda_4 = 0.1807$ (Sample $S_4$, furthest sample at $35.36\text{ m}$)
- Lagrange multiplier $\mu = 0.0015$

Verification of unbiasedness:
$$\sum_{i=1}^4 \lambda_i = 0.3265 + 0.2440 + 0.2488 + 0.1807 = 1.0000$$

**Estimated Grade at $x_0$:**
$$Z^*(x_0) = \sum_{i=1}^4 \lambda_i Z(x_i)$$
$$Z^*(x_0) = (0.3265 \times 0.850) + (0.2440 \times 0.620) + (0.2488 \times 0.540) + (0.1807 \times 0.780)$$
$$Z^*(x_0) = 0.2775 + 0.1513 + 0.1344 + 0.1409 = 0.7041\% \text{ Cu}$$

**Kriging Estimation Variance:**
$$\sigma_{OK}^2(x_0) = \sum_{i=1}^4 \lambda_i \gamma(x_i, x_0) + \mu$$
$$\sigma_{OK}^2(x_0) = (0.3265 \times 0.0842) + (0.2440 \times 0.0955) + (0.2488 \times 0.0972) + (0.1807 \times 0.1087) + 0.0015$$
$$\sigma_{OK}^2(x_0) = 0.02749 + 0.02330 + 0.02418 + 0.01964 + 0.00150 = 0.09611\%^2$$
$$\sigma_{OK}(x_0) = \sqrt{0.09611} \approx 0.3100\% \text{ Cu}$$

### 6.4 Part 3: Applying Modifying Factors to the Block

The in-situ volume of Block B-12 is:
$$V = 25.0\text{ m} \times 25.0\text{ m} \times 15.0\text{ m} = 9,375\text{ m}^3$$
$$\text{In-Situ Tonnage } T_{\text{insitu}} = 9,375\text{ m}^3 \times 2.70\text{ t/m}^3 = 25,312.5\text{ tonnes}$$
$$\text{Contained Copper Metal} = 25,312.5\text{ t} \times 0.007041 = 178.23\text{ tonnes Cu}$$

Engineering audit establishes the following modifying factors for Bench 1420:
- Operational contact dilution: $\delta_{\text{dil}} = 8.0\%$ with hanging wall waste rock carrying $g_{\text{waste}} = 0.080\%$ Cu.
- Mining recovery: $\eta_{\text{mine}} = 95.0\%$.

**Diluted Run-of-Mine (ROM) Reserve Tonnage:**
$$T_{\text{dil}} = 25,312.5 \times (1 + 0.08) = 27,337.5\text{ tonnes}$$

**Diluted Run-of-Mine Grade:**
$$\text{Dilution Waste Tonnes} = 25,312.5 \times 0.08 = 2,025.0\text{ tonnes}$$
$$\text{Metal in Dilution Waste} = 2,025.0\text{ t} \times 0.0008 = 1.62\text{ tonnes Cu}$$
$$\text{Total Contained Metal} = 178.23 + 1.62 = 179.85\text{ tonnes Cu}$$
$$g_{\text{dil}} = \frac{179.85\text{ t}}{27,337.5\text{ t}} = 0.6579\% \text{ Cu}$$

**Recovered Ore Delivered to Surface:**
$$T_{\text{delivered}} = 27,337.5\text{ t} \times 0.95 = 25,970.6\text{ tonnes}$$
$$\text{Contained Metal Delivered} = 25,970.6\text{ t} \times 0.006579 = 170.86\text{ tonnes Cu}$$

### 6.5 Part 4: Monthly Reconciliation Run Across Bench 1420

Over the subsequent production month, an entire mining sector comprising Block B-12 and surrounding blocks (totaling 500,000 planned tonnes) was excavated and processed through the Kopano concentrator. 

The physical measurements across the four evaluation gates were:
1. **Long-Term Reserve Block Model (Plan):**
   - Tonnes: $T_{\text{Model}} = 500,000.0\text{ t}$
   - Grade: $g_{\text{Model}} = 0.7200\%$ Cu
   - Contained Metal: $M_{\text{Model}} = 500,000.0 \times 0.0072 = 3,600.0\text{ t Cu}$
2. **Grade Control In-Pit Blast-Hole Model (Actual Delineated):**
   - Blast-hole drilling at $6\text{m} \times 6\text{m}$ pattern delineated:
   - Tonnes: $T_{\text{GC}} = 520,000.0\text{ t}$
   - Grade: $g_{\text{GC}} = 0.6800\%$ Cu
   - Contained Metal: $M_{\text{GC}} = 520,000.0 \times 0.0068 = 3,536.0\text{ t Cu}$
3. **Mill Feed Received (Primary Crusher Weightometer & Cross-Belt Analyzer):**
   - Ore received from pit haulage trucks:
   - Tonnes: $T_{\text{Mill}} = 540,000.0\text{ t}$
   - Grade: $g_{\text{Mill}} = 0.6300\%$ Cu
   - Contained Metal: $M_{\text{Mill}} = 540,000.0 \times 0.0063 = 3,402.0\text{ t Cu}$
4. **Mill Final Accounted Metal (Product Concentrates + Measured Tailings):**
   - Clean copper concentrate produced: $9,639.0\text{ t}$ concentrate at $30.00\%$ Cu $= 2,891.7\text{ t Cu}$ ($85.0\%$ plant recovery).
   - Tailings slurries measured: $530,361.0\text{ t}$ tailings at $0.0900\%$ Cu $= 476.3\text{ t Cu}$.
   - Total Final Accounted Metal: $M_{\text{Accounted}} = 2,891.7 + 476.3 = 3,368.0\text{ t Cu}$.

### 6.6 Calculating the Parker Multi-Stage Reconciliation Ratios

**1. $F_1$ Reconciliation (Grade Control / Long-Term Model):**
$$F_{1, T} = \frac{520,000}{500,000} = 1.0400 \quad (+4.0\% \text{ ore tonnes})$$
$$F_{1, g} = \frac{0.6800}{0.7200} = 0.9444 \quad (-5.56\% \text{ grade})$$
$$F_{1, M} = \frac{3,536.0}{3,600.0} = 0.9822 \quad (-1.78\% \text{ contained metal})$$
*Interpretation:* The long-term resource model over-smoothed the grade boundary, slightly over-estimating in-situ grade and under-estimating tonnage. Contained metal is within acceptable geostatistical calibration ($\pm 2\%$).

**2. $F_2$ Reconciliation (Mill Received / Grade Control Dispatched):**
$$F_{2, T} = \frac{540,000}{520,000} = 1.0385 \quad (+3.85\% \text{ delivered tonnes})$$
$$F_{2, g} = \frac{0.6300}{0.6800} = 0.9265 \quad (-7.35\% \text{ delivered grade})$$
$$F_{2, M} = \frac{3,402.0}{3,536.0} = 0.9621 \quad (-3.79\% \text{ delivered metal})$$
*Interpretation:* Severe mining execution failure. Tonnage increased by $3.85\%$ while grade dropped by $7.35\%$. Shovels excavated beyond dig-line boundaries into low-grade contact rock (excess operational dilution), and blast heave displaced the ore boundary by $4.5\text{ meters}$, which was uncorrected in pit dispatch.

**3. $F_3$ Reconciliation (Accounted Metallurgical Metal / Claimed Mill Feed):**
$$F_{3, M} = \frac{3,368.0}{3,402.0} = 0.9900 \quad (-1.00\% \text{ metal balance deficit})$$
*Interpretation:* The primary crusher cross-belt analyzer had a small positive calibration bias (over-reporting head grade by $1.0\%$), or minor unmeasured pipe scaling hold-up occurred in the flotation launders.

**4. Total Net Reconciliation ($F_{\text{total}}$):**
$$F_{\text{total}, T} = \frac{540,000}{500,000} = 1.0800 \quad (+8.00\% \text{ tonnes})$$
$$F_{\text{total}, g} = \frac{0.6300}{0.7200} = 0.8750 \quad (-12.50\% \text{ grade})$$
$$F_{\text{total}, M} = F_{1, M} \times F_{2, M} \times F_{3, M} = 0.9822 \times 0.9621 \times 0.9900 = 0.9355 \quad (-6.45\% \text{ net metal})$$

### 6.7 Supply Chain and Financial Impact

The net reconciliation deficit $F_{\text{total}, M} = 0.9355$ generated severe supply chain bottlenecks:
1. **Mill Capacity Bottleneck:** The concentrator processed $40,000$ extra tonnes of barren rock to produce less copper. Grinding power costs escalated by $\$184,000$ ($40,000\text{ t} \times 23\text{ kWh/t} \times \$0.20/\text{kWh}$).
2. **Product Deficit:** Planned metal production was $3,600 \times 0.85 = 3,060.0\text{ t Cu}$. Actual concentrate production was only $2,891.7\text{ t Cu}$—a monthly shortfall of $168.3\text{ tonnes of payable copper}$.
3. **Demurrage and Port Delay:** At the Port of Dar es Salaam, Kopano had chartered a Handymax bulk carrier with a committed laycan window to load $10,000\text{ tonnes}$ of concentrate. Because concentrate production trailed plan by $561\text{ tonnes}$, the vessel sat at anchor for 4 days awaiting cargo completion. At a charter demurrage rate of $\$35,000/\text{day}$, the mine incurred $\$140,000$ in cash demurrage penalties.

---

## 7. Code, query, or calculation

The following self-contained Python script implements the Ordinary Kriging estimation and the multi-stage Parker reconciliation calculations derived in Section 6.

### Script Assumptions and Requirements
- Standard Python 3.9+ with `numpy` installed (`pip install numpy`).
- Implements an isotropic Spherical Semivariogram model.
- Solves the $(n+1) \times (n+1)$ Ordinary Kriging linear system using `numpy.linalg.solve`.
- Computes point estimation, kriging variance, and multi-stage $F_1, F_2, F_3, F_{\text{total}}$ reconciliation metrics.

```python
"""Ordinary Kriging spatial estimation and multi-stage reconciliation factors."""
import numpy as np

def spherical_gamma(h, c0: float, c: float, a: float):
    """Compute semivariance using an isotropic spherical variogram model."""
    h = np.asarray(h, dtype=float)
    h_a = np.clip(h / a, 0.0, 1.0)
    structured = c * (1.5 * h_a - 0.5 * (h_a ** 3))
    gamma = np.where(h == 0.0, 0.0, np.where(h <= a, c0 + structured, c0 + c))
    return float(gamma) if gamma.ndim == 0 else gamma

# 1. Inputs: 4 sample composites [x, y, grade %Cu] and target block centroid x0
samples = np.array([[30.0, 40.0, 0.85], [70.0, 30.0, 0.62],
                    [75.0, 65.0, 0.54], [25.0, 75.0, 0.78]])
x0 = np.array([50.0, 50.0])
c0, c, a = 0.04, 0.16, 120.0
n = len(samples)

# 2. Distance matrices and Kriging setup
coords, grades = samples[:, :2], samples[:, 2]
D = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=-1)
d0 = np.linalg.norm(coords - x0, axis=1)

A = np.zeros((n + 1, n + 1))
b = np.zeros(n + 1)
A[:n, :n] = spherical_gamma(D, c0, c, a)
A[:n, n] = A[n, :n] = 1.0
b[:n] = spherical_gamma(d0, c0, c, a)
b[n] = 1.0

# 3. Solve Ordinary Kriging linear system
solution = np.linalg.solve(A, b)
weights, mu = solution[:n], solution[n]
z_est = float(np.dot(weights, grades))
s2_ok = float(np.dot(weights, b[:n]) + mu)

# 4. Multi-Stage Mine-to-Mill Reconciliation (Tonnes, Grade %Cu, Contained Metal t)
m_plan = (500000.0, 0.7200, 500000.0 * 0.0072)
m_gc   = (520000.0, 0.6800, 520000.0 * 0.0068)
m_mill = (540000.0, 0.6300, 540000.0 * 0.0063)
metal_accounted = 3368.0

f1 = (m_gc[0] / m_plan[0], m_gc[1] / m_plan[1], m_gc[2] / m_plan[2])
f2 = (m_mill[0] / m_gc[0], m_mill[1] / m_gc[1], m_mill[2] / m_gc[2])
f3_metal = metal_accounted / m_mill[2]
f_tot = (m_mill[0] / m_plan[0], m_mill[1] / m_plan[1], f1[2] * f2[2] * f3_metal)

# 5. Formatted execution output
print(f"--- Ordinary Kriging Results ---")
print(f"Weights [w1, w2, w3, w4]: {np.round(weights, 4)} (Sum = {np.sum(weights):.4f})")
print(f"Lagrange Multiplier mu   : {mu:.4f}")
print(f"Estimated Block Grade    : {z_est:.4f} % Cu")
print(f"Kriging Variance / StdDev: {s2_ok:.5f} %^2 / {np.sqrt(s2_ok):.4f} % Cu")
print(f"\n--- Mine-to-Mill Reconciliation Factors ---")
print(f"F1 (Grade Control / Plan): T={f1[0]:.4f}, G={f1[1]:.4f}, M={f1[2]:.4f}")
print(f"F2 (Delivered / GC)      : T={f2[0]:.4f}, G={f2[1]:.4f}, M={f2[2]:.4f}")
print(f"F3 (Accounted / Mill)    : Metal={f3_metal:.4f}")
print(f"Total Reconciliation     : T={f_tot[0]:.4f}, G={f_tot[1]:.4f}, M={f_tot[2]:.4f}")
```

### Expected Output
```text
--- Ordinary Kriging Results ---
Weights [w1, w2, w3, w4]: [0.3265 0.244  0.2488 0.1807] (Sum = 1.0000)
Lagrange Multiplier mu   : 0.0015
Estimated Block Grade    : 0.7041 % Cu
Kriging Variance / StdDev: 0.09611 %^2 / 0.3100 % Cu

--- Mine-to-Mill Reconciliation Factors ---
F1 (Grade Control / Plan): T=1.0400, G=0.9444, M=0.9822
F2 (Delivered / GC)      : T=1.0385, G=0.9265, M=0.9621
F3 (Accounted / Mill)    : Metal=0.9900
Total Reconciliation     : T=1.0800, G=0.8750, M=0.9356
```

---

## 8. Data, standards, and regulatory context

Mineral resource and reserve reporting is strictly governed by statutory capital market securities legislation. A mining company cannot publish resource estimates or reserve statements without formal regulatory compliance.

### 8.1 The CRIRSCO International Family

The Committee for Mineral Reserves International Reporting Standards (CRIRSCO) maintains the International Reporting Template, which serves as the core framework for national reporting codes:

1. **JORC Code (Australasia):**
   - Governing Body: Joint Ore Reserves Committee of the Australasian Institute of Mining and Metallurgy (AusIMM), Australian Institute of Geoscientists (AIG), and Minerals Council of Australia (MCA).
   - Jurisdiction: Australian Securities Exchange (ASX) and New Zealand Stock Exchange (NZX).
   - Core Standard: JORC (2012 Edition, with comprehensive 2024–2026 revisions). Mandates Table 1 disclosures (Sampling Techniques, Reporting of Exploration Results, Estimation and Reporting of Mineral Resources, Estimation and Reporting of Ore Reserves).
2. **National Instrument 43-101 (Canada):**
   - Governing Body: Canadian Securities Administrators (CSA) adhering to Canadian Institute of Mining, Metallurgy and Petroleum (CIM) Definition Standards (2014).
   - Jurisdiction: Toronto Stock Exchange (TSX) and TSX Venture Exchange (TSX-V).
   - Core Standard: Binding statutory law. Requires filing a Form 43-101F1 Technical Report within 45 days of publicly disclosing material resource changes.
3. **SEC Regulation S-K Subpart 1300 (United States):**
   - Governing Body: U.S. Securities and Exchange Commission (SEC).
   - Promulgated: 2018; mandatory compliance for all SEC registrants effective January 1, 2021 (rescinding the antiquated 1982 SEC Industry Guide 7).
   - Alignment: Formally aligned US reporting with CRIRSCO definitions. Requires filing an **Initial Assessment** (with discounted cash flows) to support Mineral Resource disclosure and a **Preliminary Feasibility Study (PFS)** or **Feasibility Study (FS)** signed by a Qualified Person to declare Mineral Reserves.
4. **SAMREC Code (South Africa):**
   - Governing Body: South African Mineral Resource Committee; mandatory for Johannesburg Stock Exchange (JSE).
5. **PERC Standard (Europe):**
   - Governing Body: Pan-European Reserves & Resources Reporting Committee; accepted across European bourses (LSE, Euronext).

### 8.2 The Qualified Person (QP) / Competent Person (CP) Framework

Under all CRIRSCO codes, legal responsibility rests personally upon the Qualified Person (QP) or Competent Person (CP). A QP/CP must satisfy three non-negotiable statutory tests:
- **Professional Registration:** Must hold active membership or fellow status in an approved Recognized Professional Organization (RPO) with statutory disciplinary authority (e.g., AusIMM, AIG, PGO, EGBC, SME Registered Member).
- **Relevant Experience:** Must possess a minimum of five (5) years of direct, relevant experience in the specific geological deposit style, mineralization type, and mining method under evaluation. (A gold vein specialist cannot act as CP for a brine lithium project).
- **Personal and Criminal Liability:** In jurisdictions like Canada (OSC) and the US (SEC), signing off on a fraudulent or grossly negligent technical disclosure report carries personal civil liability, director disqualification, and potential criminal sanctions for securities fraud.

### 8.3 Prohibited Disclosures and Public Compliance Rules

Regulatory codes strictly prohibit specific misleading reporting practices commonly attempted by junior promoters:
1. **No Inferred Resources in Economic Models:** Inferred resources may not be included in financial models supporting Preliminary Feasibility Studies or Feasibility Studies. They cannot be converted to reserves or used to demonstrate debt repayment capacity.
2. **Prohibition of "Gross Metal in Ground":** Reporting in-situ contained metal multiplied by current spot price (e.g., "$10 Billion of In-Situ Gold!") without subtracting mining recovery, dilution, metallurgical recovery, operating costs, and royalties is explicitly illegal under NI 43-101 and SEC S-K 1300.
3. **Transparent Price Assumptions:** Commodity price decks used to calculate cut-off grades must reflect trailing benchmark averages (e.g., 3-year historical trailing averages) or defensible consensus forecasts, not speculative peak price spikes.

### 8.4 Physical Sampling and Custody Standards

Geostatistical estimates are invalid if the underlying physical samples are unrepresentative. Sampling must conform to rigorous ISO standards:
- **ISO 3082 (Iron Ores — Sampling and Sample Preparation Procedures):** Establishes mechanical falling-stream cutter dimensions, increment masses, and precision criteria to eliminate fundamental sampling error (FSE).
- **ISO 12743 (Copper, Lead, Zinc, and Nickel Concentrates — Sampling Procedures):** Specifies mechanical auger and cross-belt sampling rules for slurries and dry concentrates, governing moisture determination and commercial offtake settlements.
- **Pierre Gy’s Sampling Theory:** The universal mathematical framework governing particulate sampling:
  $$\sigma_{\text{FSE}}^2 = C \cdot d^3 \cdot \left(\frac{1}{M_s} - \frac{1}{M_L}\right)$$
  where $M_s$ is sample mass, $M_L$ is lot mass, $d$ is maximum particle diameter, and $C$ is the mineralogical constitutional parameter. Crushing rock to smaller particle sizes ($d$) prior to mass splitting is mathematically mandatory to prevent catastrophic sample variance.

---

## 9. Industry practice and case evidence

### 9.1 Case Evidence 1: The Bre-X Busang Gold Scandal (1997)

The most infamous failure of resource integrity in mining history occurred at the Busang project in East Kalimantan, Indonesia, promoted by Canadian junior exploration company Bre-X Minerals Ltd.

```text
=============================================================================================================
                    CHRONOLOGY OF THE BRE-X BUSANG FRAUD AND NI 43-101 GENESIS
=============================================================================================================
 1993–1996: Aggressive Core Salting       Early 1997: Due Diligence Crisis       May 1997: Total Collapse
 ┌──────────────────────────────────┐     ┌──────────────────────────────────┐   ┌───────────────────────────┐
 │ Bre-X geologists manually salt   │     │ Partner Freeport-McMoRan drills  │   │ Independent audit proves  │
 │ barren drill core with alluvial  │ ──► │ 7 twin verification holes;       │──►│ Busang contains zero gold.│
 │ and crushed jewelry gold.        │     │ finds zero economic gold.        │   │ Bre-X stock -> $0.00.     │
 │ Resources hyped: 200M oz gold.   │     │ Chief Geologist Michael de Guzman│   │ $6 Billion market cap     │
 │ Stock rallies from $0.50 to $280.│     │ mysteriously falls from helicopter│   │ completely destroyed.     │
 └──────────────────────────────────┘     └──────────────────────────────────┘   └─────────────┬─────────────┘
                                                                                               │
                                                                                               ▼
                                          2001: Regulatory Consequence
                                          ┌──────────────────────────────────────────────────────────────────┐
                                          │ Canadian Securities Administrators enact National Instrument     │
                                          │ 43-101. Mandates Qualified Persons, independent QA/QC, certified │
                                          │ reference materials, umpire assays, and Form 43-101F1 reports.  │
                                          └──────────────────────────────────────────────────────────────────┘
=============================================================================================================
```

- **Mechanism:** Bre-X geologists took barren volcanic drill core, crushed it on site, and physically "salted" the samples using panned alluvial gold dust and shave gold from local jewelry. Because no independent umpire assays, blind blanks, or certified reference materials (CRMs) were inserted by independent third parties, the assay laboratories reported extraordinary bonanza gold intercepts.
- **Consequence:** Market capitalization reached CAD $6 billion on the Toronto Stock Exchange. When prospective joint-venture partner Freeport-McMoRan drilled seven twin verification holes directly adjacent to the reported Bre-X discovery holes, assay results returned essentially zero gold. Bre-X declared bankruptcy, wiping out institutional pension funds and retail investors.
- **Regulatory Result:** The Canadian securities regulators, together with the CIM, created **National Instrument 43-101**. NI 43-101 ended the era of informal resource claims by codifying statutory Qualified Person accountability, mandatory public technical reports, and non-negotiable QA/QC protocols across global capital markets.

### 9.2 Case Evidence 2: Ravensthorpe Nickel Hydrometallurgical Failure (BHP Billiton, 2008–2009)

In 2008, BHP commissioned the $\$2.2\text{ billion}$ Ravensthorpe nickel laterite mine in Western Australia. The project was designed around a massive resource of lateritic ore feeding an atmospheric leach and High-Pressure Acid Leach (HPAL) hydrometallurgical circuit.

- **Modifying Factor Breakdown:** The geological block model correctly predicted total nickel grade in the ground. However, the conversion from resource to reserve failed to model the **mineralogical domining** and **metallurgical modifying factors**:
  1. The deposit contained high concentrations of nontronite and smectite swelling clays that had not been differentiated from limonitic ores in the block model.
  2. In the beneficiation and leaching circuits, these clays exhibited catastrophic rheological properties, blinding counter-current decantation (CCD) wash thickeners and preventing slurry solid-liquid separation.
  3. Acid consumption surged to un-economic levels, while nickel recovery dropped far below the feasibility study design threshold.
- **Supply Chain and Financial Collapse:** Just eight months after commissioning, BHP suspended operations in January 2009, terminating 1,800 workers and writing off the entire $\$2.2\text{ billion}$ capital investment (later selling the asset for $\$340\text{ million}$).
- **Core Lesson:** A mineral resource block model that interpolates only head metal grade ($\% \text{ Ni}$) while ignoring mineralogical processing domains, clay rheology, and acid consumption curves violates basic modifying factor principles and guarantees downstream operational failure.

### 9.3 Case Evidence 3: Operational Mine-to-Mill Reconciliation at Escondida and Grasberg

World-scale open pit operations—such as BHP's Minera Escondida in Chile (copper) and Freeport-McMoRan's Grasberg in Indonesia (copper-gold)—process between 100,000 and 250,000 tonnes of ore per day. At this scale, a $2\%$ uncorrected error in $F_1$ or $F_2$ represents over $\$100\text{ million}$ in annual lost cash flow.

- **Advanced Grade Control Implementation:**
  - **Blast Movement Monitoring:** Both operations utilize embedded radio-frequency Blast Movement Monitors (BMMs) placed in dedicated vertical drill holes throughout the blast pattern. Post-detonation, sensors locate the new 3D coordinates of the transponders. Vector translation software dynamically warps the pre-blast grade control dig-lines to match the actual post-blast muckpile heave, reducing contact dilution by up to $40\%$.
  - **Fleet Telemetry Integration:** Haul trucks are equipped with high-precision GPS (HP-GPS) integrated with fleet management systems (e.g., Modular Mining DISPATCH or Komatsu FrontRunner). When an excavator bucket dips into an ore polygon, the truck dispatch system automatically locks the haul destination to the primary crusher or a specific grade-blending stockpile. If an operator attempts to dump ore at a waste dump (or waste at the crusher), the system triggers an automated geofence alarm and halts the truck.
  - **SmartTag Tracking:** Radio-frequency identification (RFID) tags encased in impact-resistant polymer spheres are placed into blast-holes. When the ore passes through the primary crusher and travels across conveyor belts into the SAG mill feed stockpile, RFID antenna arrays detect the tags, recording exact transit times and verifying whether material delivered to the mill originated from the modeled blast block.
- **Reconciliation Impact:** Deploying closed-loop Parker reconciliation backed by BMMs and RFID tracking consistently shifts monthly $F_2$ reconciliation factors from volatile swings ($0.88 - 1.15$) to a stable band of $0.98 - 1.02$, stabilizing concentrator feed grades and eliminating unexpected smelter offtake shortfalls.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Physical / Statistical Mechanism | Supply Chain Consequence | Mitigation / Control |
|---|---|---|---|---|
| **1. Over-smoothed Kriged Block Grades** | Grade control consistently encounters higher peak grades and deeper barren zones than predicted by block model. | Using an excessively large search radius or fitting an artificial zero nugget effect, smoothing out natural geological variability. | Mill receives unpredictable grade spikes and plunges; flotation circuits experience severe recovery drops due to unstable kinetics. | Re-estimate using localized search ellipses; implement Conditional Simulation (e.g., Turning Bands or Sequential Gaussian Simulation). |
| **2. Uncorrected Support Effect** | Block model predicts high tonnage above cut-off, but open-pit mining discovers ore boundaries are narrow and discontinuous. | Applying cut-off grade directly to point-support drill core composite distributions without change-of-support variance reduction. | Severe over-estimation of recoverable ore tonnes and under-estimation of unit mining costs; premature mill starvation. | Apply Discrete Gaussian Change of Support or Affine Variance Reduction prior to cut-off evaluation. |
| **3. Inferred Resource Smuggling** | Mine planning schedule relies on "Inferred" blocks to fulfill Years 3–5 production commitments. | Incorporating low-confidence exploration blocks into the production schedule to artificially extend Life of Mine (LOM). | Inevitable production collapse when infill drilling reveals mineralization is discontinuous or sub-economic; debt default. | Strictly audit mine plans against CRIRSCO guidelines; zero Inferred resources allowed in base-case financial models. |
| **4. Unmonitored Blast Movement Displacement** | Visual ore boundaries inside the muckpile match blast-hole drill collars, but delivered plant head grade is severely depressed ($F_{2, g} \ll 1.0$). | Explosive detonation heaves and shifts rock horizontally by $3\text{ to }12\text{ meters}$; loading unadjusted pre-blast polygons digs adjacent waste. | Massive operational dilution; thousands of tonnes of economic ore dumped on waste piles; barren rock fed to mill. | Deploy Blast Movement Monitors (BMM); translate dig-line polygons dynamically based on 3D heave vectors. |
| **5. Ignoring Mineralogical Domaining** | Total metal assays match the block model, but concentrator recovery drops by $15\% - 30\%$. | Estimating total metal (e.g., Total Cu) while ignoring metallurgical species (e.g., copper oxides vs. secondary chalcocite vs. refractory chalcopyrite). | Massive drop in concentrate production; reagent consumption spikes; severe acid or collector cost overruns. | Mandate multi-element sequential leach assays (oxide, transition, sulfide); construct explicit metallurgical domain wireframes. |
| **6. Belt Weightometer Calibration Drift** | Monthly $F_3$ factor steadily climbs above $1.05$ or drops below $0.93$. | Mechanical belt scale idlers on the primary crusher feed conveyor accumulate dust or lose tare calibration. | Erroneous dry tonnage reporting corrupts entire metallurgical mass balance; distorted inventory tracking. | Mandate weekly physical chain tests and zero-point calibrations on all belt weightometers; verify against survey pit volumes. |
| **7. Cross-Contamination in Sample Preparation** | Exploration blanks consistently return assay values significantly above the analytical lower detection limit. | Inadequate compressed-air cleaning or barren quartz flushes between core samples in laboratory crushers and pulverizers. | Artificial low-grade halos modeled around narrow veins; false expansion of economic resource wireframes. | Implement strict laboratory QA/QC audits; enforce mandatory barren quartz flushes between all high-grade sample preps. |
| **8. Unmeasured Stockpile Inventory Oxidation** | Concentrate recovery from ROM stockpile ore is $10\% - 20\%$ lower than fresh pit ore of identical head grade. | Base metal sulfide minerals (chalcopyrite, sphalerite, pentlandite) oxidize when exposed to air and rain on surface stockpiles. | Altered mineral surface chemistry prevents xanthate collector attachment during flotation; metal lost to tailings. | Implement strict First-In, First-Out (FIFO) stockpile dispatch policies; cap long-term sulfide stockpile residence times to $<45\text{ days}$. |
| **9. Truck Misrouting and Dump Bleed** | Pit excavators mine economic ore, but crusher receives lower tonnage than truck dispatch logs claim ($F_{2, T} \ll 1.0$). | Haul truck drivers misread physical bench stakes or dispatch GPS loses connectivity, dumping ore on waste rock dumps. | Permanent, irrecoverable loss of high-grade metal; waste dumps contaminated with acid-generating sulfides. | Implement automated geofenced truck dispatch interlocks; install RFID SmartTags in haul truck beds. |
| **10. Slurry Auto-Sampler Bias** | Shift composite head grades fail to balance with concentrate and tailings assays ($F_3$ fails to close). | Primary slurry sampler cutter speed is non-uniform, or cutter blades are damaged/worn, causing preferential sampling of fines or coarse particles. | Fraudulent metallurgical reconciliation; plant metallurgical recovery metrics become legally and operationally un-auditable. | Install ISO-compliant linear cross-stream slurry samplers; audit cutter velocity profiles and wear tolerances monthly. |
| **11. Static Economic Cut-Off Grade** | Low-grade ore is dumped on waste dumps during structural commodity bull markets, destroying enterprise value. | Using an antiquated, static cut-off grade established during the initial feasibility study regardless of prevailing metal prices. | Sub-optimal exploitation of resource endowment; lost operating margin and shortened mine life. | Transition to Lane's dynamic cut-off grade algorithm, optimizing cut-off grade based on mill throughput and market price. |
| **12. Arbitrary Resource Downgrading for ESG** | Permitting delays or community water rights disputes halt operations despite high geostatistical confidence. | Evaluating resources purely on geological drill spacing while treating legal, water, and social licence factors as external afterthoughts. | Complete asset freeze; inability to declare reserves under SEC S-K 1300 and JORC; multi-billion-dollar impairment. | Formally integrate ESG, tailings management (GISTM), and water stewardship into the modifying factor conversion audit. |

---

## 11. Assumptions and limits

| Assumption / Simplification | Why It Is Needed | Boundary Condition / Failure State | How to Audit / Test in Practice |
|---|---|---|---|
| **1. Second-Order Stationarity of Grade** | Allows estimation of covariance and semivariogram across the entire spatial domain from localized sample pairs. | Fails in the presence of strong spatial trends, regional zoning, or abrupt fault displacements. | Test for spatial trend surfaces using moving-window drift analysis; domain the deposit into structurally homogeneous fault blocks. |
| **2. Linear Additivity of Grade Variables** | Essential for Ordinary Kriging; assumes sample grades average linearly based on mass weighting. | Fails for non-additive variables like pH, specific gravity, metallurgical recovery $\%$, or rock quality designation (RQD). | Accumulate metal content (Grade $\times$ Density $\times$ Volume) rather than interpolating non-linear variables directly. |
| **3. Constant Specific Gravity (Density)** | Allows length-weighted composite assays to be treated as equivalent to mass-weighted sample inputs. | Fails in high-grade base metal deposits where heavy sulfides (e.g., galena $\rho=7.5\text{ t/m}^3$) double rock density. | Measure immersion specific gravity on all core intervals; perform mass-weighted compositing: $l_i \cdot \rho_i \cdot Z_i$. |
| **4. Zero Spatial Displacement During Blasting** | Simplifies short-term grade control by allowing dig-lines to be drawn directly on pre-blast drillhole collars. | Fails in open pit bench blasting where explosive confinement heaves rock horizontally by $3\text{ to }15\text{ meters}$. | Measure horizontal and vertical displacement vectors using electronic 3D Blast Movement Monitors (BMMs). |
| **5. Deterministic Modifying Factors** | Simplifies reserve estimation by applying single fixed percentages for dilution ($8\%$) and recovery ($95\%$). | Fails under varying geotechnical rock mass ratings (RMR) where bad ground causes stope over-break of $25\%+$. | Build geotechnical domain models that scale dilution and recovery dynamically based on local rock quality. |
| **6. Ergodicity of Spatial Random Fields** | Assumes the single realization of the deposit observed in drillholes is representative of the underlying stochastic process. | Fails in highly nuggety, extreme-grade deposits (e.g., coarse free-gold veins) with limited sample counts. | Conduct extreme value capping (top-cutting) and use Spatial Indicator Kriging or Multi-Gaussian simulations. |
| **7. Perfect Particulate Mixing on Stockpiles** | Assumes ROM buffer stockpiles blend out grade swings into a uniform, homogenized mill feed. | Fails due to particle size segregation (coarse rocks roll to base of pile; fine clay particles concentrate at top). | Audit stockpile stacking and reclaiming methods; replace end-dump cone piles with chevron-stacked blending beds. |
| **8. Unbiased Blast-Hole Cuttings Sampling** | Assumes cone splitters or dust collectors on rotary blast-hole drill rigs capture a representative assay sample. | Fails when heavy sulfide minerals preferentially segregate into fine drill dust or wet cuttings smear along the drill stem. | Regularly twin blast-hole samples with dedicated reverse circulation (RC) grade control drilling to test sampling bias. |
| **9. Instantaneous Steady-State Mill Inventory** | Assumes material entering the primary crusher in a given calendar month matches the material exiting the flotation cells. | Fails in large concentrators with massive SAG mill scats stockpiles, high slurry residence volumes, or thickener surge capacity. | Perform comprehensive physical plant inventory surveys (slurry tank levels, bin soundings) at month-end cutoff. |
| **10. Rigid Search Ellipsoid Dimensions** | Ensures Ordinary Kriging algorithms select an adequate number of sample points across all deposit blocks. | Can cause extreme local over-smoothing if search ellipses span across unmineralized barren dykes or fault offsets. | Enforce hard boundary constraints (wireframe domains); samples outside a domain cannot inform blocks inside. |

---

## 12. Exercises

### Exercise 1: Semivariogram Fitting and Model Evaluation
- **Context:** An exploration geologist calculates experimental semivariance values $\hat{\gamma}(h)$ from copper drillhole composites at $20\text{-meter}$ lag intervals along the strike of a skarn deposit:

| Lag Distance $h$ (m) | 0.0 | 20.0 | 40.0 | 60.0 | 80.0 | 100.0 | 120.0 |
|---|---|---|---|---|---|---|---|
| Experimental $\hat{\gamma}(h)$ ($\%^2$) | 0.000 | 0.075 | 0.125 | 0.160 | 0.180 | 0.182 | 0.179 |

- **Task:** 
  1. Estimate the nugget effect $C_0$, partial sill $C$, total sill $C_0 + C$, and range $a$ by visual inspection.
  2. Compute theoretical semivariance $\gamma(h)$ at $h = 50.0\text{ m}$ using a Spherical Model.
  3. Compute theoretical semivariance $\gamma(h)$ at $h = 50.0\text{ m}$ using an Exponential Model (practical range $a$).
- **Deliverable:** Mathematical derivation and comparative error analysis.
- **Acceptance Criteria:** Numerical values match analytical formulas; parameter definitions clearly stated.
- **Estimated Time:** 25 minutes.

### Exercise 2: Change of Support and Dispersion Variance
- **Context:** Raw $1.0\text{-meter}$ diamond drill core composites in an open-pit gold deposit exhibit a sample variance of $\sigma_{\text{sample}}^2 = 4.50\text{ (g/t)}^2$ with a mean grade of $\bar{Z} = 1.80\text{ g/t}$. Geostatistical dispersion calculations determine that the variance of samples within an SMU block ($v \in V$) is $D^2(v/V) = 2.70\text{ (g/t)}^2$.
- **Task:**
  1. Calculate the theoretical variance of SMU block grades $\sigma_{\text{block}}^2$ within the deposit using Krige's relation.
  2. Explain why an analyst who uses the raw drillhole composite distribution to predict the proportion of ore above a high cut-off grade ($COG = 3.0\text{ g/t}$) will severely overstate the recoverable gold grade.
- **Deliverable:** Exact numerical variance calculation and a 150-word written technical critique.
- **Acceptance Criteria:** Krige's relation correctly applied; support effect mechanisms accurately described.
- **Estimated Time:** 20 minutes.

### Exercise 3: Three-Point Ordinary Kriging by Hand
- **Context:** An unsampled point $x_0$ is surrounded by three sample composites:
  - Sample $A$: Distance $d_{A0} = 10.0\text{ m}$, Grade $Z_A = 1.20\%$ Ni
  - Sample $B$: Distance $d_{B0} = 20.0\text{ m}$, Grade $Z_B = 0.90\%$ Ni
  - Sample $C$: Distance $d_{C0} = 25.0\text{ m}$, Grade $Z_C = 0.60\%$ Ni
  The pairwise sample distances are $D_{AB} = 15.0\text{ m}$, $D_{AC} = 30.0\text{ m}$, and $D_{BC} = 20.0\text{ m}$.
  The semivariogram is linear: $\gamma(h) = 0.05 \cdot h$ (zero nugget, slope $0.05\%^2/\text{m}$).
- **Task:**
  1. Construct the $4 \times 4$ Ordinary Kriging matrix system $\mathbf{K} \boldsymbol{\lambda} = \mathbf{k}$ including the Lagrange multiplier $\mu$.
  2. Set up the matrix equations with exact numerical values.
  3. Given the solution weights $\lambda_A = 0.5833$, $\lambda_B = 0.3333$, $\lambda_C = 0.0834$, and $\mu = 0.0833$, verify unbiasedness.
  4. Calculate the estimated grade $Z^*(x_0)$ and the kriging estimation variance $\sigma_{OK}^2(x_0)$.
- **Deliverable:** Step-by-step arithmetic equations and final numerical estimates.
- **Acceptance Criteria:** Unbiasedness verified ($\sum \lambda_i = 1.0$); grade estimate and kriging variance mathematically correct.
- **Estimated Time:** 35 minutes.

### Exercise 4: Multi-Stage Mine-to-Mill Reconciliation Diagnostic
- **Context:** An open-pit gold mine reports the following production data for Q2:
  - **Reserve Plan:** $1,200,000\text{ tonnes}$ @ $2.40\text{ g/t Au}$ ($2,880.0\text{ kg Au}$).
  - **Grade Control In-Pit Model:** $1,260,000\text{ tonnes}$ @ $2.25\text{ g/t Au}$ ($2,835.0\text{ kg Au}$).
  - **Delivered to Crusher (Dispatched):** $1,350,000\text{ tonnes}$ @ $1.95\text{ g/t Au}$ ($2,632.5\text{ kg Au}$).
  - **Plant Accounted (Bullion + Tails):** $1,340,000\text{ tonnes}$ @ $1.96\text{ g/t Au}$ ($2,626.4\text{ kg Au}$).
- **Task:**
  1. Compute $F_{1, T}$, $F_{1, g}$, and $F_{1, M}$.
  2. Compute $F_{2, T}$, $F_{2, g}$, and $F_{2, M}$.
  3. Compute $F_{3, M}$.
  4. Compute $F_{\text{total}, T}$, $F_{\text{total}, g}$, and $F_{\text{total}, M}$.
  5. Diagnose the primary operational failure in the supply chain: Did it occur in geological estimation, pit mining execution, or plant metallurgical accounting?
- **Deliverable:** Full reconciliation metrics table and a concise 4-sentence diagnostic summary.
- **Acceptance Criteria:** All ratios correct to 4 decimal places; diagnostic matches physical mechanism.
- **Estimated Time:** 30 minutes.

### Exercise 5: Critical Audit of an NI 43-101 Technical Report
- **Context:** A junior mining company releases a Preliminary Economic Assessment (PEA) for a prospective open-pit copper-gold deposit. The report claims:
  - "Total Mineral Resource: 80 million tonnes @ $0.80\%$ Cu equivalent, comprising 20 Mt Indicated and 60 Mt Inferred."
  - "Life of Mine Production Schedule: 8.0 Mt per annum over a 10-year mine life, processing 80 Mt of ore."
  - "Project Net Present Value (NPV): $\$650\text{ million}$ at an $8\%$ discount rate."
- **Task:** 
  1. Identify the fatal regulatory and technical non-compliance flaw under NI 43-101 and CRIRSCO standards.
  2. Critique the supply chain viability of building a capital processing plant based on this production schedule.
  3. Detail what infill drilling and engineering studies are mandatory before an institutional debt lender would finance the project.
- **Deliverable:** Technical audit memorandum (maximum 300 words).
- **Acceptance Criteria:** Names statutory rules regarding Inferred resources; outlines financial and operational risks.
- **Estimated Time:** 30 minutes.

### Exercise 6: Upstream Grade Volatility and Downstream Port Demurrage Risk
- **Context:** A zinc-lead mining operation transports bulk zinc concentrate via rail to a coastal bulk export terminal. The concentrator's budgeted monthly concentrate output is $20,000\text{ tonnes}$ of concentrate carrying $55.0\%$ Zn, requiring a feed of $200,000\text{ tonnes}$ of ore at $6.0\%$ Zn with a steady $91.67\%$ flotation recovery. 
  During a month with poor grade control reconciliation, delivered head grade drops to $4.8\%$ Zn ($F_{2, g} = 0.80$). Flotation plant recovery drops non-linearly to $86.0\%$ due to lower feed kinetics. Crusher tonnage throughput cannot be increased above $200,000\text{ tonnes/month}$ due to SAG mill motor constraints.
- **Task:**
  1. Calculate the actual concentrate tonnes produced during the month (assuming concentrate grade is maintained at $55.0\%$ Zn).
  2. Determine the concentrate production deficit in tonnes relative to budget.
  3. If a 20,000-tonne bulk carrier arrives at the port under a contract with a 3-day laytime allowance and daily demurrage of $\$25,000/\text{day}$, and concentrate is railed at a fixed rate of $600\text{ tonnes/day}$, calculate the number of excess days the vessel sits at anchor and the total demurrage penalty incurred.
- **Deliverable:** Full step-by-step mathematical calculations and financial penalty quantification.
- **Acceptance Criteria:** Tonnage and laytime arithmetic correct; logistics constraints honored.
- **Estimated Time:** 35 minutes.

### Exercise 7: Designing an Automated Grade Control and Reconciliation Architecture
- **Context:** You are appointed Director of Supply Chain & Mine Analytics for a copper producer experiencing chronic $-8\%$ monthly metal reconciliation shortfalls.
- **Task:** Design an end-to-end operational architecture integrating sensors, digital tracking, and reconciliation software from pit face to export berth.
- **Deliverable:** A structured systems architecture plan detailing:
  - In-pit sensing and blast movement tracking.
  - Haul truck telemetry and geofencing dispatch controls.
  - Primary crusher and conveyor analytical instrumentation.
  - Daily/weekly reconciliation software trigger points and management escalation thresholds.
- **Acceptance Criteria:** Concrete technologies named; addresses $F_1$, $F_2$, and $F_3$ failure points directly.
- **Estimated Time:** 45 minutes.

---

## 13. Self-check answers

### Self-Check for Exercise 1
1. **Semivariogram Parameter Identification:**
   - At $h = 0$, $\hat{\gamma}(0) = 0$. As $h \to 0$ from positive lags, values extrapolate to approximately $0.040\%^2 \implies \text{Nugget Effect } C_0 = 0.040\%^2$.
   - Plateau levels off around $h = 80 - 100\text{ m}$ at $\approx 0.180\%^2 \implies \text{Total Sill } C_0 + C = 0.180\%^2$.
   - Partial Sill $C = (C_0 + C) - C_0 = 0.180 - 0.040 = 0.140\%^2$.
   - Range $a = 80.0\text{ m}$ (distance where curve reaches the total sill plateau).
2. **Spherical Model Calculation at $h = 50.0\text{ m}$:**
   $$\frac{h}{a} = \frac{50}{80} = 0.625$$
   $$\gamma(50) = 0.040 + 0.140 \left[ 1.5(0.625) - 0.5(0.625)^3 \right] = 0.040 + 0.140 [0.9375 - 0.12207] = 0.040 + 0.140(0.81543) = 0.1542\%^2$$
3. **Exponential Model Calculation at $h = 50.0\text{ m}$:**
   $$\gamma(50) = 0.040 + 0.140 \left[ 1 - \exp\left(-\frac{3 \times 50}{80}\right) \right] = 0.040 + 0.140 [1 - \exp(-1.875)]$$
   $$\gamma(50) = 0.040 + 0.140 [1 - 0.15335] = 0.040 + 0.140(0.84665) = 0.1585\%^2$$

### Self-Check for Exercise 2
1. **Variance of SMU Block Grades:**
   Under Krige's relation:
   $$D^2(v/D) = D^2(v/V) + D^2(V/D)$$
   $$\sigma_{\text{sample}}^2 = D^2(v/V) + \sigma_{\text{block}}^2$$
   $$\sigma_{\text{block}}^2 = \sigma_{\text{sample}}^2 - D^2(v/V) = 4.50 - 2.70 = 1.80\text{ (g/t)}^2$$
2. **Technical Critique of Support Effect Error:**
   Raw $1\text{m}$ drill core samples exhibit high dispersion variance ($\sigma^2 = 4.50$) with extreme outlier tails. In reality, excavators mine $1,000\text{ m}^3$ SMU blocks where local internal averaging dampens variability ($\sigma^2 = 1.80$). If an analyst applies a high cut-off grade ($3.0\text{ g/t}$) to the sample histogram, the heavy upper tail of the sample distribution suggests that substantial high-grade ore exists. At block support, however, the highest grades are mixed with adjacent lower grades inside the block volume, pulling block grades back toward the mean ($1.80\text{ g/t}$). Consequently, far fewer blocks will exceed the $3.0\text{ g/t}$ cut-off than the sample histogram predicts, resulting in severe shortfalls in actual recovered metal and catastrophic financial underperformance.

### Self-Check for Exercise 3
1. **Semivariances:**
   With $\gamma(h) = 0.05 \cdot h$:
   - $\gamma(D_{AA}) = \gamma(D_{BB}) = \gamma(D_{CC}) = 0$
   - $\gamma(D_{AB}) = 0.05 \times 15 = 0.75$, $\gamma(D_{AC}) = 0.05 \times 30 = 1.50$, $\gamma(D_{BC}) = 0.05 \times 20 = 1.00$
   - $\gamma(d_{A0}) = 0.05 \times 10 = 0.50$, $\gamma(d_{B0}) = 0.05 \times 20 = 1.00$, $\gamma(d_{C0}) = 0.05 \times 25 = 1.25$
2. **Kriging System Matrix:**
   $$\begin{bmatrix}
   0.00 & 0.75 & 1.50 & 1.00 \\
   0.75 & 0.00 & 1.00 & 1.00 \\
   1.50 & 1.00 & 0.00 & 1.00 \\
   1.00 & 1.00 & 1.00 & 0.00
   \end{bmatrix}
   \begin{bmatrix}
   \lambda_A \\ \lambda_B \\ \lambda_C \\ \mu
   \end{bmatrix}
   =
   \begin{bmatrix}
   0.50 \\ 1.00 \\ 1.25 \\ 1.00
   \end{bmatrix}$$
3. **Verification of Weights:**
   $$\sum \lambda_i = 0.5833 + 0.3333 + 0.0834 = 1.0000 \quad (\text{Unbiased})$$
4. **Estimated Grade and Kriging Variance:**
   $$Z^*(x_0) = (0.5833 \times 1.20) + (0.3333 \times 0.90) + (0.0834 \times 0.60) = 0.7000 + 0.3000 + 0.0500 = 1.0500\% \text{ Ni}$$
   $$\sigma_{OK}^2(x_0) = \sum \lambda_i \gamma(x_i, x_0) + \mu = (0.5833 \times 0.50) + (0.3333 \times 1.00) + (0.0834 \times 1.25) + 0.0833$$
   $$\sigma_{OK}^2(x_0) = 0.29165 + 0.33330 + 0.10425 + 0.08330 = 0.8125\%^2$$

### Self-Check for Exercise 4
1. **$F_1$ Reconciliation:**
   - $F_{1, T} = 1,260,000 / 1,200,000 = 1.0500 \quad (+5.0\%)$
   - $F_{1, g} = 2.25 / 2.40 = 0.9375 \quad (-6.25\%)$
   - $F_{1, M} = 2,835.0 / 2,880.0 = 0.9844 \quad (-1.56\%)$
2. **$F_2$ Reconciliation:**
   - $F_{2, T} = 1,350,000 / 1,260,000 = 1.0714 \quad (+7.14\%)$
   - $F_{2, g} = 1.95 / 2.25 = 0.8667 \quad (-13.33\%)$
   - $F_{2, M} = 2,632.5 / 2,835.0 = 0.9286 \quad (-7.14\%)$
3. **$F_3$ Reconciliation:**
   - $F_{3, M} = 2,626.4 / 2,632.5 = 0.9977 \quad (-0.23\%)$
4. **Total Reconciliation:**
   - $F_{\text{total}, T} = 1,350,000 / 1,200,000 = 1.1250 \quad (+12.50\%)$
   - $F_{\text{total}, g} = 1.95 / 2.40 = 0.8125 \quad (-18.75\%)$
   - $F_{\text{total}, M} = 2,626.4 / 2,880.0 = 0.9119 \quad (-8.81\%)$
5. **Root-Cause Diagnostic:**
   The primary failure occurred in **$F_2$ (pit mining execution)**. $F_1$ metal was within $\pm 1.6\%$ (good geological model), and $F_3$ was $0.9977$ (excellent plant metallurgical balance). However, $F_2$ showed a $7.14\%$ tonnage surge accompanied by a catastrophic $13.33\%$ grade collapse, resulting in a $7.14\%$ metal loss. This signature indicates severe operational contact dilution and unmonitored blast movement displacement at the excavator pit face.

### Self-Check for Exercise 5: NI 43-101 Audit Rubric
- **Fatal Regulatory Flaw:** Including $60\text{ Mt}$ of Inferred Resources in a 10-year production schedule. Under NI 43-101 and CIM standards, Inferred resources lack sufficient geological confidence to have modifying factors applied and cannot be used in production scheduling or economic evaluations in a Pre-Feasibility or Feasibility Study.
- **Supply Chain Risk:** Building an $8.0\text{ Mtpa}$ concentrator on a reserve base of only $20\text{ Mt}$ (2.5 years of mill feed) guarantees capital destruction. Once the 20 Mt of Indicated material is exhausted, the mill will face starvation if Inferred mineralization proves discontinuous.
- **Mandatory Actions:** Infill diamond drilling on $50\text{m} \times 50\text{m}$ spacing to upgrade Inferred to Indicated resources; metallurgical core testing to establish flotation recovery equations; and a formal Pre-Feasibility Study demonstrating economic viability.

### Self-Check for Exercise 6
1. **Actual Concentrate Produced:**
   - Metal entering plant $= 200,000\text{ t} \times 0.048 = 9,600\text{ t Zn}$.
   - Recovered metal $= 9,600\text{ t} \times 0.86 = 8,256\text{ t Zn}$.
   - Concentrate mass produced $= 8,256\text{ t} / 0.55 = 15,010.9\text{ tonnes concentrate}$.
2. **Concentrate Production Deficit:**
   - Budgeted concentrate $= 20,000.0\text{ t}$.
   - Deficit $= 20,000.0 - 15,010.9 = 4,989.1\text{ tonnes of concentrate}$.
3. **Laytime and Demurrage Calculation:**
   - Time required to rail $20,000\text{ t}$ at $600\text{ t/day} = 33.33\text{ days}$.
   - At month end, available concentrate is only $15,010.9\text{ t}$. Railing the missing $4,989.1\text{ t}$ at $600\text{ t/day}$ requires $\frac{4,989.1}{600} = 8.32\text{ days}$.
   - Vessel allowable laytime $= 3.0\text{ days}$.
   - Net excess demurrage days $= 8.32 - 3.00 = 5.32\text{ days}$ (or if vessel arrived expecting full cargo staged, delay is $8.32\text{ days}$).
   - Total Demurrage Penalty $= 5.32\text{ days} \times \$25,000/\text{day} = \$133,000$.

---

## 14. Interview questions

### Question 1: What is the fundamental difference between an Indicated Mineral Resource and a Probable Mineral Reserve?
- **What a strong answer contains:** Explains that an Indicated Resource reflects geological confidence (continuity of geology and grade demonstrated by adequate drill spacing), but remains an in-situ geological estimate with "reasonable prospects for eventual economic extraction." A Probable Reserve represents the economically mineable portion of an Indicated Resource *after* the rigorous, audited application of Modifying Factors (mining dilution, mining recovery, metallurgical recovery curves, cut-off grade economics, processing costs, environmental, social, and legal permitting).
- **What a weak answer reveals:** Believes the difference is simply drill spacing, or assumes reserves are just resources multiplied by a generic recovery factor without naming modifying factors.

### Question 2: Why can Inferred Mineral Resources never be converted into Mineral Reserves under CRIRSCO reporting codes?
- **What a strong answer contains:** Cites JORC / NI 43-101 / SEC S-K 1300 rules. Explains that Inferred resources have implied, unverified geological and grade continuity based on widely spaced, isolated samples. The estimation variance is too high to allow the application of modifying factors in sufficient detail to support mine scheduling or economic viability assessments. Smuggling Inferred resources into bankable financial models creates severe debt default risk.
- **What a weak answer reveals:** Thinks Inferred resources can become "Inferred Reserves," demonstrating complete ignorance of international securities regulations.

### Question 3: How does the Support Effect impact the relationship between exploration diamond drill assays and actual production block grades?
- **What a strong answer contains:** Quotes Krige’s relation: $D^2(v/V) = D^2(v/D) - D^2(V/D)$. Demonstrates that small physical volumes (drill cores) have higher dispersion variance and heavy distribution tails compared to large volumes (Selective Mining Units or benches). Explains that applying a cut-off grade directly to point-support composite distributions overestimates recovered grade and underestimates tonnage, requiring Change of Support corrections (Discrete Gaussian or Affine transformations).
- **What a weak answer reveals:** Confuses support effect with physical ground support (rock bolts/shotcrete) or assumes block grades have the same variance as drill core samples.

### Question 4: In the Parker Mine-to-Mill Reconciliation framework, how do you diagnose an operation where $F_{1, M} \approx 1.00$, but $F_{2, T} = 1.10$ and $F_{2, g} = 0.88$?
- **What a strong answer contains:** Notes that $F_{1, M} \approx 1.00$ proves the long-term resource block model is well-calibrated against grade control blast-hole drilling. The failure is entirely localized in $F_2$ (pit mining execution). A $10\%$ increase in tonnage coupled with a $12\%$ drop in grade ($F_{2, M} = 1.10 \times 0.88 = 0.968$, a $3.2\%$ metal loss) is the textbook signature of **uncontrolled operational contact dilution**. Shovel operators are digging outside marked dig-lines, or blast heave displaced the muckpile boundaries without dynamic polygon adjustment.
- **What a weak answer reveals:** Blames the exploration geologist or claims the processing plant is losing metal in the tailings.

### Question 5: Why does the Ordinary Kriging estimation variance depend solely on the spatial geometry and the variogram, rather than the actual composite assay grades?
- **What a strong answer contains:** Derives the kriging system: minimizing the error variance $\sigma_E^2$ subject to $\sum \lambda_i = 1$ leads to $\sigma_{OK}^2 = \sum \lambda_i \gamma(x_i, x_0) + \mu$. The matrix $\mathbf{K}$ and vector $\mathbf{k}$ contain only semivariance values $\gamma(h)$, which are functions of Euclidean distance vectors and the modeled spatial continuity parameters ($C_0, C, a$). Consequently, kriging variance measures spatial data configuration and drill density, not whether local rock is high-grade or low-grade.
- **What a weak answer reveals:** Believes kriging variance is higher simply because sample grades are high, confusing estimation geometry with heteroscedastic data scatter.

### Question 6: What are Blast Movement Monitors (BMMs) and how do they protect downstream supply chain integrity?
- **What a strong answer contains:** Details that explosive detonation heaves rock horizontally and vertically by several meters. BMM transponders placed in blast holes are tracked post-blast to measure 3D displacement vectors. Translating pre-blast grade control dig-lines to match the post-blast muckpile prevents excavators from routing high-grade ore to waste dumps and waste rock to the primary crusher. This preserves feed grade, prevents mill starvation, and ensures contracted concentrate delivery volumes.
- **What a weak answer reveals:** Thinks BMMs are vibration sensors used only for community noise compliance.

### Question 7: How does an unpredicted drop in mill head grade cascade through the downstream bulk supply chain to the export port?
- **What a strong answer contains:** Traces the non-linear operational cascade: lower feed grade reduces flotation recovery kinetics, requiring higher reagent dosages and more grinding energy per unit of recovered metal. Total concentrate mass produced drops. Rail freight trains run under-utilized or must be canceled. At the port, bulk concentrate stockpiles fall short of the required vessel cargo parcel, triggering ship loading delays, berth congestion, and expensive daily demurrage penalties ($>\$30,000/\text{day}$).
- **What a weak answer reveals:** Assumes supply chains simply wait for more ore without financial or logistical penalties.

### Question 8: How do changing macroeconomic metal prices alter the definition of what constitutes an Ore Reserve?
- **What a strong answer contains:** Explains that reserves are defined by economic extractability. Cut-off grade is inversely proportional to realized metal price: $COG \approx C / (P \cdot R)$. When commodity prices decline, the cut-off grade rises; blocks previously classified as Probable Reserves fall below the cut-off and are reclassified as sub-economic Mineral Resources or waste, shrinking reported reserves and altering the optimal Life-of-Mine schedule.
- **What a weak answer reveals:** Believes reserves are fixed geological quantities that never change once drilled.

---

## 15. Key terms

- **CRIRSCO:** Committee for Mineral Reserves International Reporting Standards; global body harmonizing national public reporting codes.
- **JORC Code:** Australasian reporting code mandating standards for public disclosure of Exploration Results, Mineral Resources, and Ore Reserves.
- **NI 43-101:** Canadian National Instrument 43-101; statutory standard of disclosure for mineral projects on Canadian exchanges.
- **SEC Regulation S-K 1300:** U.S. SEC rules modernizing property disclosures for mining registrants, aligned with CRIRSCO.
- **Qualified Person (QP):** Professional engineer or geoscientist with statutory accountability for signing public mineral disclosure reports.
- **Mineral Resource:** In-situ mineral concentration with reasonable prospects for eventual economic extraction, categorized by geological confidence.
- **Inferred Resource:** Lowest-confidence resource category; spatial continuity implied but unverified; cannot convert to reserves.
- **Indicated Resource:** Moderate-confidence resource category; sufficient continuity to support mine planning and modifying factor application.
- **Measured Resource:** Highest-confidence resource category; spatial continuity confirmed by dense drilling; supports detailed mine design.
- **Mineral Reserve:** Economically mineable part of Measured or Indicated resources after applying all Modifying Factors.
- **Probable Reserve:** Economically viable reserve derived from Indicated resources (or lower-confidence Measured resources).
- **Proved Reserve:** Highest-confidence reserve category, derived exclusively from Measured resources with fully resolved modifying factors.
- **Modifying Factors:** Technical, economic, processing, metallurgical, legal, environmental, and social criteria used to convert resources to reserves.
- **Semivariogram:** Spatial statistics function measuring half the average squared difference between sample grades as a function of separation distance.
- **Nugget Effect ($C_0$):** Semivariance at zero lag distance representing micro-scale geological variability and measurement error.
- **Partial Sill ($C$):** Spatially structured variance component between nugget effect and total sill.
- **Total Sill ($C_0 + C$):** Plateau of the semivariogram representing total population variance.
- **Spatial Range ($a$):** Distance beyond which spatial correlation between samples drops to zero.
- **Ordinary Kriging:** Linear spatial interpolation technique providing the Best Linear Unbiased Estimator (BLUE) and estimation variance.
- **Kriging Variance ($\sigma_{OK}^2$):** Calculated estimation error variance based on sample geometry and spatial variogram parameters.
- **Support Effect:** Principle that dispersion variance decreases as the physical volume (support) of the evaluated unit increases.
- **Information Effect:** Misclassification of ore and waste during mining caused by selecting blocks using imperfect sample estimates.
- **Selective Mining Unit (SMU):** Smallest volume of rock that mining equipment can physically discriminate into ore or waste.
- **Run-of-Mine (ROM):** Raw, unprocessed ore excavated from the mine and delivered to stockpiles or the primary crusher.
- **$F_1$ Reconciliation:** Ratio comparing short-term blast-hole grade control models to the long-term resource block model.
- **$F_2$ Reconciliation:** Ratio comparing ore delivered to the processing plant to the short-term grade control model.
- **$F_3$ Reconciliation:** Metallurgical balance ratio comparing final accounted metal (product + tails) to claimed plant feed head metal.
- **Total Reconciliation ($F_{\text{total}}$):** Product of $F_1 \times F_2 \times F_3$, measuring net metal performance against the original reserve model.
- **Operational Dilution:** Unplanned waste rock excavated outside planned ore boundaries and fed to the processing plant.
- **Mining Recovery:** Percentage of ore tonnes and metal within a designed stope or pit block successfully delivered to the surface.

---

## 16. Sources

- **CRIRSCO (2019):** *International Reporting Template for the Public Reporting of Exploration Results, Mineral Resources and Mineral Reserves.* Committee for Mineral Reserves International Reporting Standards.
- **Joint Ore Reserves Committee (2012):** *The Australasian Code for Reporting of Exploration Results, Mineral Resources and Ore Reserves (The JORC Code).* AusIMM, AIG, MCA.
- **Canadian Institute of Mining, Metallurgy and Petroleum (2014):** *CIM Definition Standards — For Mineral Resources and Mineral Reserves.* CIM Standing Committee on Reserve Definitions.
- **U.S. Securities and Exchange Commission (2018):** *Modernization of Property Disclosures for Mining Registrants.* 17 CFR Parts 229, 230, 239, and 249 (Regulation S-K Subpart 1300).
- **David, M. (1977):** *Geostatistical Ore Reserve Estimation.* Elsevier Scientific Publishing Company, Amsterdam.
- **Isaaks, E. H., & Srivastava, R. M. (1989):** *An Introduction to Applied Geostatistics.* Oxford University Press, New York.
- **Journel, A. G., & Huijbregts, C. J. (1978):** *Mining Geostatistics.* Academic Press, London.
- **Matheron, G. (1963):** *Principles of geostatistics.* Economic Geology, 58(8), 1246–1266. DOI: 10.2113/gsecongeo.58.8.1246.
- **Parker, H. M. (1987):** *The volume-variance relationship: A review.* Geostatistics for Natural Resources Characterization, Part 1, Reidel, Dordrecht.
- **Rossi, M. E., & Deutsch, C. V. (2014):** *Mineral Resource Estimation.* Springer, Dordrecht. DOI: 10.1007/978-94-007-6776-8.
- **Schofield, N. A. (2001):** *The misuse of kriging in mine planning.* In: *Mine to Mill 2001 Conference Proceedings*, Australasian Institute of Mining and Metallurgy, Melbourne.
- **Morley, C. (2014):** *Beyond reconciliation — The development of an integrated mine-to-mill production tracking system.* In: *Orebody Modelling and Strategic Mine Planning Conference*, AusIMM, Perth.

---

## 17. Links

- **Upward Index:** [Module 39 — Mining, Metals, and Forestry Supply Chains](_index.md)
- **Sibling Topic 02:** [Mine Planning and Cut-Off Grade Optimization](02_mine-planning-cutoff.md)
- **Sibling Topic 03:** [Drilling, Blasting, and Load-Haul Operations](03_drilling-blasting-loadhaul.md)
- **Sibling Topic 04:** [Grade Blending and Specification Management](04_grade-blending.md)
- **Sibling Topic 05:** [Processing, Beneficiation, and Metallurgical Recovery](05_processing-recovery.md)
- **Domain Survey File:** [Course Module 39 Survey File](../../../course/39_mining_metals_forestry.md)
- **Prerequisite Module:** [Module 38 — Process Industries, Commodities, and Bulk Supply Chains](../38_process_industries_commodities/_index.md)
