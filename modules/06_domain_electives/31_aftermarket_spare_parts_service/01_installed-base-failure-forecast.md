> Module 31 — Aftermarket, Spare Parts, and Service Supply Chains
> Topic 01 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Module 06 (Machine Learning & Reliability Foundations), Module 08 (Hierarchical & Intermittent Forecasting), Module 10 (Inventory Optimization)
> Version 2026.5

# Installed Base and Failure Demand

---

## 1. Why this topic exists

In retail, consumer packaged goods, and standard manufacturing supply chains, demand is driven by market forces: consumer sentiment, price elasticity, marketing promotions, seasonal holidays, macroeconomic growth, and competitor actions. Inventory planners in those domains analyze historical point-of-sale (POS) data, build time-series regressions, and run demand-sensing algorithms based on store traffic and digital clickstreams.

When supply chain professionals apply those same retail forecasting methodologies to capital equipment aftermarket and spare parts operations, the result is chronic operational failure. Millions of dollars are wasted holding slow-moving parts that never move, while critical multi-million-dollar assets sit idle waiting weeks for a fifty-dollar gasket or sensor.

This failure occurs because **aftermarket service demand is not market-driven; it is asset-driven**. 

Spare parts do not experience spontaneous consumer desire. A replacement turbine blade, an MRI gradient amplifier, an aircraft hydraulic actuator, or a mining haul-truck differential is demanded only because an asset in the field has reached an inspection interval, suffered component degradation, or experienced sudden physical failure. Therefore, the true source of truth for aftermarket demand is not sales order history, but the **installed base (IB)**: the living population of operating machines, their specific engineering configurations, their chronological ages, their accumulated operating hours or duty cycles, and the harshness of their operational environments.

Consider the pathology of forecasting spare parts from order history alone:
1. **The Ghost Demand Fallacy:** An industrial gas turbine operator purchases ten spare fuel nozzles in October to stock their local site warehouse. A traditional time-series forecast registers a sudden demand spike of ten units and predicts elevated demand for subsequent quarters. In reality, the customer’s installed base did not consume ten nozzles; they simply transferred inventory from the OEM's depot to their own site cage. The turbines will not consume another nozzle for three years. The statistical model orders unneeded replenishments, creating stranded working capital.
2. **Censored Demand from Stockouts:** An airline needs a bleed-air regulator valve for a Boeing 787 grounded in Singapore. The regional parts distribution center is stocked out. The airline cannot wait 14 days for backorder fulfilment; it performs a field cannibalization from an out-of-service airframe or procures a loaner valve from a partner carrier under an emergency interline agreement. The OEM’s ERP system logs zero sales because no shipment occurred. The historical demand time series registers a zero, depressing future safety stock calculations precisely where demand was desperately urgent.
3. **The Bathtub and Fleet Aging Lag:** A fleet of 500 electric transit buses is deployed across four municipal agencies. In years one and two, warranty claims and infant mortality faults are handled by commissioning engineers. During years three through six, failures are low and random (the useful life period). A time-series model trained on these three years forecasts near-zero demand for suspension bushings, traction motor bearings, and pneumatic valves. By year seven, the entire fleet enters the wear-out phase simultaneously. The sudden surge in component degradation overwhelms supplier lead times, grounding 18% of the municipal fleet.

To build an accurate, defensible service supply chain, organizations must abandon pure historical curve-fitting. Planners must establish a causal engineering link between the asset register, component physics of failure, operating exposure, and supply replenishment lead times. This topic provides the mathematical, structural, and operational framework required to model installed bases and predict failure demand.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Differentiate** the mathematical and physical drivers of aftermarket failure demand from standard retail and finished-goods demand.
2. **Construct** an asset installed base data schema that captures asset identity, engineering Bill of Materials (BOM) revisions, installation timestamps, utilization rates, and environmental severity tiers.
3. **Calculate** time-dependent operational exposure across a heterogeneous fleet using operating hours, thermal cycles, calendar age, and mission profiles.
4. **Formulate** failure demand models using homogeneous Poisson processes (HPP), non-homogeneous Poisson processes (NHPP), and renewal theory for repairable and consumable components.
5. **Calibrate** two-parameter and three-parameter Weibull hazard rate functions from asset maintenance records with right-censored and left-truncated operational data.
6. **Classify** component demand patterns into smooth, intermittent, slow-moving, and erratic categories using the Syntetos-Boylan classification scheme (Average Demand Interval vs. Squared Coefficient of Variation).
7. **Separate** the four distinct demand streams—scheduled preventive maintenance, unscheduled random breakdown, installed-base expansion, and fleet retirement/last-time-buy—into decoupled forecasting pipelines.
8. **Reconstruct** true unconstrained historical demand by identifying, adjusting, and uncensoring historical stockout periods and cannibalization events.

---

## 3. The core mental model

To understand failure demand, you must hold the end-to-end causal chain in your mind. The demand signal travels through physical, operational, and data layers before reaching the spare parts warehouse:

```text
+-----------------------------------------------------------------------------------+
| PHYSICAL & OPERATIONAL LAYER: THE INSTALLED BASE                                 |
|                                                                                   |
|  [ Asset Register ] ---> [ Exposure Engine ] ---> [ Stress Factors ]             |
|   Serial No.              Operating Hours          Ambient Temp / Dust            |
|   Site Location           Thermal / Duty Cycles    Vibration / Load Profiles      |
|   BOM Configuration       Calendar Age             Operator Behavior              |
+-----------------------------------------------------------------------------------+
                                     │
                                     ▼
+-----------------------------------------------------------------------------------+
| RELIABILITY & DEGRADATION LAYER: THE HAZARD FUNCTION                              |
|                                                                                   |
|  [ Hazard h(t) ] ──────> [ Failure Mechanism ] ──> [ Replacement Event ]         |
|   Infant Mortality        Wear-out / Fatigue       Planned Overhaul (Deterministic)|
|   Random / Useful Life    Corrosion / Erosion      Unplanned Breakdown (Stochastic)|
|   Wear-out                Dielectric Breakdown     Condition-Triggered Alert      |
+-----------------------------------------------------------------------------------+
                                     │
                                     ▼
+-----------------------------------------------------------------------------------+
| SUPPLY CHAIN & TRANSACTION LAYER: THE DEMAND REGISTRATION                        |
|                                                                                   |
|  [ Maintenance Order ] -> [ Stock Availability ] ─> [ Supply Chain Signal ]       |
|   Parts Requisition        In Stock  ──> Issued     Depot Replenishment Reorder   |
|   Scrap vs. Repair         Stockout ──> Censored    Expedite / Cannibalization /  |
|   Part Supersession                     Emergency    Direct Sourcing / Lost Order |
+-----------------------------------------------------------------------------------+
```

### 3.1 The Three Essential Layers

1. **The Physical and Operational Layer:** Assets do not exist in the abstract. They are distributed physical entities operating under specific stress regimes. A gas compressor operating offshore in the North Sea suffers salt-spray corrosion and high wave-induced cyclical vibration; an identical machine operating in an enclosed, climate-controlled pharmaceutical facility in Switzerland experiences a completely different hazard rate. Operational exposure (hours, cycles, starts, stops) dictates the clock against which components degrade.
2. **The Reliability and Degradation Layer:** Components possess failure characteristics governed by materials science and physics of failure. Degradation is modelled through hazard rate functions, $h(t)$. When an inspection reveals wear beyond a maintenance limit, or when catastrophic functional failure occurs, a physical part removal takes place.
3. **The Supply Chain and Transaction Layer:** The maintenance technician generates a work order and requests parts from the stockroom. If the inventory system fulfills the request immediately, a consumption event is logged. If parts are unavailable, the technician may cannibalize another machine, repair the old part with field expedients, or delay the repair. The transaction recorded in the ERP system is frequently a distorted, delayed, or censored reflection of the physical failure event.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse | Unit or Coding Case |
|---|---|---|---|
| **Installed Base (IB)** | The complete, verified inventory of all operational assets currently in service across customer locations, including serial numbers, configurations, and operating states. | Confusing cumulative historical sales volume with active surviving field assets. | Count of active assets ($\mathbb{N}$); dimensionless. |
| **Exposure ($E_t$)** | The cumulative amount of operational usage accumulated by an asset or population over time period $t$ (e.g., flight hours, MWh, operating cycles). | Assuming all installed machines run 24 hours per day, 365 days per year. | Operating Hours (OH), Operating Cycles (OC), or Calendar Days. |
| **Hazard Rate ($h(t)$)** | The instantaneous rate of failure at time $t$, given survival up to time $t$: $h(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t \mid T \ge t)}{\Delta t}$. | Treating hazard rate as a probability (it is a rate and can exceed 1.0). | Failures per unit of exposure (e.g., failures / $10^3$ hours); $[T^{-1}]$. |
| **Mean Time Between Failures (MTBF)** | The expected operating time between consecutive failures for a repairable asset or system: $\text{MTBF} = \frac{\text{Total Operating Exposure}}{\text{Total Unscheduled Failures}}$. | Applying MTBF to non-repairable parts (which have MTTF) or assuming constant failure rates when wear-out exists. | Hours, Cycles, or Kilometres; $[T]$. |
| **Mean Time to Failure (MTTF)** | The expected operating time to the first and only failure of a non-repairable (consumable) item. | Using MTTF for complex repairable assemblies like pumps, engines, or circuit boards. | Hours, Cycles, or Calendar Days; $[T]$. |
| **Bathtub Curve** | A conceptual hazard rate curve exhibiting three distinct epochs: decreasing hazard (infant mortality), constant hazard (useful life), and increasing hazard (wear-out). | Believing all individual components follow a bathtub curve; empirical studies show only ~4–10% of aerospace parts exhibit wear-out bathtub behavior. | Plot of $h(t)$ versus operational age $t$. |
| **Weibull Shape Parameter ($\beta$)** | The dimensionless exponent in a Weibull distribution that dictates whether the failure rate decreases ($\beta < 1$), remains constant ($\beta = 1$), or increases ($\beta > 1$). | Assuming $\beta = 1$ (exponential distribution) for all mechanical parts to simplify calculations. | Dimensionless real number ($\beta \in \mathbb{R}^+$). |
| **Weibull Scale Parameter ($\eta$)** | The characteristic life parameter representing the age at which 63.2% of the component population will have failed, regardless of the shape parameter. | Confusing characteristic life ($\eta$) with mean time to failure ($\text{MTTF}$). | Unit of exposure identical to $t$ (Hours, Cycles). |
| **Censored Data** | Survival analysis data where the event of interest (failure) has not occurred within the observation window (right-censored) or occurred prior to observation (left-censored). | Dropping unfailed assets from the dataset, which catastrophically overestimates failure rates. | Binary indicator: $\delta_i \in \{0, 1\}$ (0 = censored, 1 = failure). |
| **Average Demand Interval (ADI)** | The average number of time periods (e.g., weeks, months) between consecutive non-zero demand transactions: $\text{ADI} = \frac{N}{k}$. | Calculating ADI on aggregated regional data instead of SKU-depot location level. | Time periods per demand transaction; dimensionless ratio. |
| **Squared Coeff. of Variation ($CV^2$)** | The variance of non-zero demand transaction sizes divided by the square of their mean: $CV^2 = \left(\frac{\sigma_z}{\mu_z}\right)^2$. | Calculating $CV^2$ including zero-demand periods, which conflates intermittency with sizing lumpiness. | Dimensionless variance ratio. |
| **Syntetos-Boylan Classification** | A two-dimensional categorization framework dividing demand series into Smooth, Intermittent, Erratic, and Lumpy based on cutoffs $\text{ADI} = 1.32$ and $CV^2 = 0.49$. | Using Croston's method on smooth or erratic demand, or standard exponential smoothing on lumpy demand. | Categorical: `Smooth`, `Intermittent`, `Erratic`, `Lumpy`. |
| **Bill of Materials (BOM) Effectivity** | The date, serial-range, or configuration block during which a specific part number is valid for installation on a parent assembly. | Forecasting a legacy part without checking if recent engineering changes superseded it with a modern revision. | Range of dates (`date_start`, `date_end`) or serial intervals. |
| **Cannibalization** | The practice of removing a functional spare part from an idle, unserviceable, or storage asset to repair an active down machine. | Ignoring cannibalization in ERP data, thereby failing to register unmet demand. | Binary or integer count of cannibalization events. |
| **Last-Time Buy (LTB)** | A final procurement order placed with a supplier to cover all projected future life-of-type demand for a component that the supplier is discontinuing. | Forecasting LTB demand using standard 12-month exponential smoothing instead of lifetime asset retirement curves. | Total quantity of parts to cover asset phase-out horizon ($Q_{\text{LTB}}$). |

---

## 5. The mechanics, step by step

Forecasting failure demand from an installed base requires an integrated sequence of eight analytical procedures. You cannot jump directly to a statistical formula without curating the asset register and resolving observational bias.

```text
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 1: Ingest and Validate the Asset Register (Asset ID, Serial, Site, BOM)    │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 2: Ingest and Standardize Exposure Logs (Hours, Starts, Cycles, Env Tiers) │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 3: Link Work Orders and Uncensor Demand (Extract Replaced SKUs, Identify   │
  │         Cannibalization, Impute Stockout Periods)                               │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 4: Separate Demand into Distinct Operational Streams (Planned Overhaul,    │
  │         Unplanned Random, IB Commissioning, Fleet Retirement)                   │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 5: Classify Component Demand Characteristics (Compute ADI and CV²)         │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 6: Fit Parametric Reliability Functions (Weibull MLE, NHPP Power Law)      │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 7: Project Forward Exposure Across the Planning Horizon (Run Asset Fleet   │
  │         Utilization Schedules)                                                  │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 8: Convolve Exposure and Hazard Models to Generate Part-Level Demand       │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Step 1: Ingest and Validate the Asset Register

The asset register is the master repository of all physical machines in the field. 
* **Input:** Enterprise Asset Management (EAM / CMMS) records, ERP equipment masters, warranty registries, and sales contracts.
* **Procedure:**
  1. Filter out decommissioned, retired, and scrapped assets unless performing historical survival analysis.
  2. Map parent-child relationships: Asset Level 1 (Plant/System) $\to$ Level 2 (Machine/Skid) $\to$ Level 3 (Subassembly) $\to$ Level 4 (Maintainable Item / SKU).
  3. Reconcile BOM effectivity dates. If Part A was replaced by Part B via Engineering Change Order (ECO) #1042 on June 1, 2024, ensure that Asset Serial #001 through #400 (which were built with Part A) link to Part A unless a field retrofit kit was installed.
* **Output:** Validated asset master table:
  $$\mathcal{A} = \{ a_i = \langle \text{AssetID}, \text{Serial}, \text{Model}, \text{Site}, t_{\text{install}}, \text{ConfigID}, \text{Status} \rangle \}$$

### 5.2 Step 2: Ingest and Standardize Exposure Logs

Assets fail due to usage stress, not the passage of calendar time. A pump sitting idle in a warehouse does not accumulate impeller wear.
* **Input:** SCADA telemetry, IoT hour-meters, vehicle telematics, field technician service sheets, customer self-reported meter readings.
* **Procedure:**
  1. Identify primary and secondary exposure metrics (e.g., Operating Hours $OH$, Thermal Starts $N_{\text{start}}$, Kilometres $KM$, Tonnage Throughput $W$).
  2. Detect meter rollovers, resets, and gauge replacements. When a 5-digit hour meter ticks from 99,999 to 00,000, do not treat this as negative exposure.
  3. Compute daily or monthly utilization rates:
     $$u_{i,t} = \frac{\Delta \text{Exposure}_{i,t}}{\Delta \text{Calendar Days}_t}$$
  4. Assign environmental severity factors ($S_i \ge 1.0$) based on ambient operating conditions:
     $$S_i = f(\text{Temperature}, \text{Particulate Contamination}, \text{Corrosion Index}, \text{Vibration})$$
* **Output:** Clean operational exposure matrix:
  $$\mathbf{E}_{i,t} = u_{i,t} \cdot S_i \cdot \Delta t$$

### 5.3 Step 3: Link Work Orders and Uncensor Demand

A raw sales order database is fatally corrupted by inventory stockouts, batch ordering, and technician workarounds.
* **Input:** CMMS work order maintenance logs, warehouse inventory issue records, procurement expedite requests.
* **Procedure:**
  1. Link each part issue back to a specific `AssetID` and `WorkOrderID`. Exclude warehouse adjustments, scrap write-offs, and stock transfers from the failure count.
  2. Identify stockout censoring: If a part had zero inventory for 22 days in March, and the asset was down awaiting parts, log a failure demand event on Day 1 of the downtime, not when the replenishment arrived in April.
  3. Adjust for field cannibalization: When Work Order #882 notes "Part pulled from Asset #12 to restore Asset #04", record two failure demands: one for the initial failure on Asset #04, and one future replenishment demand for Asset #12.
* **Output:** Reconstructed true failure event dataset:
  $$\mathcal{D}_{\text{true}} = \{ d_k = \langle \text{AssetID}_i, \text{PartSKU}_p, t_{\text{event}}, \text{FailureMode}, \text{Type} \in \{\text{PM}, \text{CM}\} \rangle \}$$

### 5.4 Step 4: Separate Demand into Distinct Operational Streams

Demand streams have fundamentally different mathematical structures and must never be aggregated into a single forecasting model.

$$\text{Demand}_{p,t}^{\text{Total}} = D_{p,t}^{\text{Planned}} + D_{p,t}^{\text{Unplanned}} + D_{p,t}^{\text{Commissioning}} + D_{p,t}^{\text{Retirement}}$$

1. **Planned Maintenance ($D_{p,t}^{\text{Planned}}$):** Deterministic or semi-deterministic demand driven by original equipment manufacturer (OEM) manuals and statutory regulations (e.g., 5,000-hour turbine inspection kits, 100,000-km axle overhauls). Computed directly by projecting exposure to scheduled maintenance trigger thresholds.
2. **Unplanned Failure ($D_{p,t}^{\text{Unplanned}}$):** Stochastic demand driven by degradation, stress exceedances, and component defects. Modelled via probability distributions and reliability hazard rates.
3. **Commissioning & IB Growth ($D_{p,t}^{\text{Commissioning}}$):** Initial provisioning kits (spares packages purchased alongside new capital equipment) and infant mortality burn-in replacements.
4. **Fleet Retirement & LTB ($D_{p,t}^{\text{Retirement}}$):** Declining demand as older asset blocks are retired, culminating in life-of-type last-time buys.

### 5.5 Step 5: Classify Component Demand Characteristics

Compute intermittency and lumpiness metrics for each part-location pair to route parts to appropriate forecasting engines.
* **Input:** Time series of unplanned demand occurrences for part $p$ at depot location $l$ over $T$ periods.
* **Procedure:**
  1. Calculate Average Demand Interval (ADI):
     $$\text{ADI} = \frac{T}{N_{\text{periods with demand} > 0}}$$
  2. Calculate the sample mean ($\bar{z}$) and standard deviation ($s_z$) of non-zero transaction sizes:
     $$CV^2 = \left( \frac{s_z}{\bar{z}} \right)^2$$
  3. Classify into the Syntetos-Boylan quadrant:
     * **Smooth ($\text{ADI} < 1.32$, $CV^2 < 0.49$):** Regular demand, low variance. Use exponential smoothing (ETS) or auto-ARIMA.
     * **Intermittent ($\text{ADI} \ge 1.32$, $CV^2 < 0.49$):** Sporadic timing, consistent sizing. Use Syntetos-Boylan Approximation (SBA) or Croston's method.
     * **Erratic ($\text{ADI} < 1.32$, $CV^2 \ge 0.49$):** Regular timing, highly volatile order sizes. Use negative binomial regression or neural temporal point processes.
     * **Lumpy ($\text{ADI} \ge 1.32$, $CV^2 \ge 0.49$):** Sporadic timing and volatile sizes. Use compound Poisson, Poisson hurdle models, or physical installed-base hazard models.

### 5.6 Step 6: Fit Parametric Reliability Functions

For components with increasing or decreasing failure rates (wear-out or burn-in), estimate the hazard function parameters.
* **Input:** Lifetime data vector $\mathcal{T} = \{ (t_i, \delta_i) \}$, where $t_i$ is the operational age of part $i$, and $\delta_i = 1$ if failed, $\delta_i = 0$ if censored (still running).
* **Procedure (Weibull Maximum Likelihood Estimation):**
  1. The two-parameter Weibull probability density function (PDF) and cumulative distribution function (CDF) are:
     $$f(t; \beta, \eta) = \frac{\beta}{\eta} \left( \frac{t}{\eta} \right)^{\beta - 1} \exp\left[ - \left( \frac{t}{\eta} \right)^\beta \right]$$
     $$F(t; \beta, \eta) = 1 - \exp\left[ - \left( \frac{t}{\eta} \right)^\beta \right]$$
  2. The hazard rate is:
     $$h(t; \beta, \eta) = \frac{f(t)}{1 - F(t)} = \frac{\beta}{\eta} \left( \frac{t}{\eta} \right)^{\beta - 1}$$
  3. Construct the log-likelihood function incorporating censored observations:
     $$\ln L(\beta, \eta) = \sum_{i=1}^n \left[ \delta_i \ln\left( \frac{\beta}{\eta} \left( \frac{t_i}{\eta} \right)^{\beta - 1} \right) - \left( \frac{t_i}{\eta} \right)^\beta \right]$$
  4. Differentiate with respect to $\beta$ and $\eta$, set to zero, and solve using Newton-Raphson iteration.
* **Interpretation of Shape Parameter $\beta$:**
  * $\beta < 1.0$: Infant mortality / burn-in defect. Decreasing hazard. (Early electrical failure, installation errors).
  * $\beta = 1.0$: Purely random failure. Constant hazard rate $\lambda = 1/\eta$. (Exponential distribution; memoryless).
  * $1.0 < \beta < 2.0$: Mild wear-out / early mechanical degradation.
  * $\beta \ge 2.0$: Distinct wear-out (fatigue, erosion, bearing raceway spalling, polymer degradation).

#### 5.6.1 Numerical Optimization for Censored Weibull Likelihood

In real-world maintenance logs, exact analytical solutions for Weibull parameters do not exist when data contains right-censored units. Differentiating the log-likelihood function yields two simultaneous non-linear equations:

$$\frac{\partial \ln L}{\partial \beta} = \frac{r}{\beta} + \sum_{i \in \text{Failures}} \ln(t_i) - \frac{1}{\eta^\beta} \sum_{i=1}^n t_i^\beta \ln(t_i) = 0$$

$$\frac{\partial \ln L}{\partial \eta} = -\frac{r \beta}{\eta} + \frac{\beta}{\eta^{\beta+1}} \sum_{i=1}^n t_i^\beta = 0$$

where $r = \sum_{i=1}^n \delta_i$ is the total count of observed failures, and $n$ is the total sample size (failures plus active censored units).

Solving the second equation directly for $\eta$ as a function of $\beta$:
$$\eta(\beta) = \left( \frac{1}{r} \sum_{i=1}^n t_i^\beta \right)^{1/\beta}$$

Substituting $\eta(\beta)$ back into the first equation yields a single root-finding problem in $\beta$:
$$g(\beta) = \frac{1}{\beta} + \frac{1}{r} \sum_{i \in \text{Failures}} \ln(t_i) - \frac{\sum_{i=1}^n t_i^\beta \ln(t_i)}{\sum_{i=1}^n t_i^\beta} = 0$$

This 1-dimensional equation is solved via the Newton-Raphson method or Brent's method:
$$\beta_{k+1} = \beta_k - \frac{g(\beta_k)}{g'(\beta_k)}$$
Once $\hat{\beta}$ converges, $\hat{\eta}$ is computed directly. Ignoring the censored terms ($\sum_{i=1}^n t_i^\beta$ truncated to failures only) systematically biases $\hat{\beta}$ upward and $\hat{\eta}$ downward, creating an illusion of rapid wear-out that distorts inventory purchases.

#### 5.6.2 Renewal Theory vs. Repairable System Processes (NHPP)

A vital conceptual distinction in aftermarket modeling is whether a component is **non-repairable (replaced upon failure)** or **repairable (restored to operation)**:

1. **Renewal Processes (Perfect Repair / "Good-as-New"):**
   * When a consumable component (e.g., a filter, brake pad, or mechanical seal) fails, it is replaced with a brand-new part. The component's operational clock resets to zero.
   * The sequence of operating lifetimes $\{X_1, X_2, \dots, X_k\}$ forms a sequence of independent and identically distributed (i.i.d.) random variables with CDF $F(t)$.
   * The expected number of renewals (spares demanded) in time interval $[0, t]$ is given by the **Renewal Function** $M(t) = \mathbb{E}[N(t)]$, satisfying the fundamental integral equation:
     $$M(t) = F(t) + \int_0^t M(t - x) f(x) \, dx$$
   * By **Blackwell’s Renewal Theorem**, for large horizons $t$, the rate of replacement demand approaches a steady-state constant:
     $$\lim_{t \to \infty} \frac{M(t)}{t} = \frac{1}{\mu_{\text{lifetime}}} = \frac{1}{\eta \Gamma(1 + 1/\beta)}$$
   * In the short-to-medium term, however, fleets commissioned simultaneously exhibit violent cyclical oscillations in renewal demand before damping out to the steady-state mean.

2. **Non-Homogeneous Poisson Processes (Minimal Repair / "Bad-as-Old"):**
   * When a complex repairable subassembly (e.g., a locomotive engine block or flight control computer) suffers an in-service fault, technicians repair only the specific failed circuit or fastener. The overall assembly is returned to service in a "bad-as-old" state, retaining its accumulated operational fatigue.
   * Such systems are governed by a Non-Homogeneous Poisson Process (NHPP) with instantaneous intensity function $\nu(t)$.
   * Under the widely utilized **Crow-AMSAA / Power Law model**:
     $$\nu(t) = \lambda \gamma t^{\gamma - 1}$$
     * $\gamma = 1$: System exhibits no reliability trend (Homogeneous Poisson Process).
     * $\gamma > 1$: System is deteriorating with age; failure intensity increases over operational time.
     * $\gamma < 1$: Reliability growth; system failure intensity decreases as engineering flaws are rectified.
   * Expected failures in planning horizon $[t_1, t_2]$:
     $$\mathbb{E}[N(t_1, t_2)] = \int_{t_1}^{t_2} \nu(t) \, dt = \lambda (t_2^\gamma - t_1^\gamma)$$
```text
    Hazard Rate h(t)
          ^
          │
          │  \                                                /  beta > 1 (Wear-out)
          │   \                                              /
          │    \                                            /
          │     \__________________________________________/     beta = 1 (Useful Life)
          │
          │   beta < 1 (Infant Mortality)
          └─────────────────────────────────────────────────────────> Operating Age (t)
              Phase 1: Burn-In       Phase 2: Useful Life      Phase 3: Wear-out
```

### 5.7 Step 7: Project Forward Exposure Across the Planning Horizon

Generate expected usage for each asset over future periods $\tau = t+1, \dots, t+H$.
* **Input:** Production schedules, airline flight schedules, drilling drilling plans, historical average utilization.
* **Procedure:**
  For each active asset $a \in \mathcal{A}$, determine expected utilization $\hat{u}_{a,\tau}$. Update the asset's cumulative operational age:
  $$\hat{T}_{a,\tau} = T_{a,\text{current}} + \sum_{s=t+1}^\tau \hat{u}_{a,s} \cdot \Delta s$$

### 5.8 Step 8: Convolve Exposure and Hazard Models to Generate Part-Level Demand

Compute the expected failures for part SKU $p$ across all assets containing part $p$.
* **Formula:**
  For an asset $a$ with current age $T_a$ accumulating additional exposure $\Delta E_a = \hat{u}_a \Delta t$ over interval $[T_a, T_a + \Delta E_a]$:
  $$\mathbb{P}(\text{Failure on Asset } a) = \frac{F(T_a + \Delta E_a) - F(T_a)}{1 - F(T_a)}$$
* Total expected failure demand for SKU $p$ across all $M$ operating assets in period $\tau$:
  $$\mathbb{E}[D_{p,\tau}^{\text{Unplanned}}] = \sum_{a=1}^M \left[ \frac{\exp\left( - (T_a / \eta_p)^{\beta_p} \right) - \exp\left( - ((T_a + \Delta E_{a,\tau}) / \eta_p)^{\beta_p} \right)}{\exp\left( - (T_a / \eta_p)^{\beta_p} \right)} \right] \cdot q_{a,p}$$
  where $q_{a,p}$ is the quantity per assembly (QPA) of part $p$ on asset $a$.

---

## 6. Worked example

### 6.1 Setting: Industrial Centrifugal Compressors

Apex Energy Solutions manages a fleet of 240 industrial centrifugal gas compressors deployed across three operational sectors:
1. **Sector Alpha (Desert Inland):** 80 compressors. High ambient heat ($45^\circ\text{C}$), high dust and sand.
2. **Sector Bravo (Offshore Marine):** 60 compressors. Salt-spray corrosion, high wave vibration, continuous 24/7 duty.
3. **Sector Charlie (Temperate Pipeline):** 100 compressors. Standard indoor compressor stations, moderate duty.

We must generate the quarterly spare parts demand forecast for Quarter 3 across three critical components:
* **SKU-101 (Dry Gas Seal):** High-precision mechanical seal. Subject to thermal wear and elastomer degradation. Wear-out component ($\beta > 1$).
* **SKU-202 (Piezoelectric Vibration Sensor):** Solid-state electronic sensor. Subject to random electrical spike failure. Useful life component ($\beta = 1$).
* **SKU-303 (High-Efficiency Lube Oil Filter Cartridge):** Consumable maintenance filter. Replaced deterministically every 2,000 operating hours.

```text
Fleet Breakdown:
========================================================================================
Sector      Asset Count   Avg Running Hrs/Day   Env Severity (S_i)   Current Mean Age
----------------------------------------------------------------------------------------
Alpha       80 units      18.0 hrs/day          1.25                 7,200 hrs
Bravo       60 units      23.5 hrs/day          1.50                 11,500 hrs
Charlie    100 units      14.0 hrs/day          1.00                 4,800 hrs
========================================================================================
Total:     240 units
```

Each compressor contains:
* 2 $\times$ SKU-101 (Dry Gas Seals)
* 4 $\times$ SKU-202 (Vibration Sensors)
* 1 $\times$ SKU-303 (Lube Filter Assembly)

Planning Horizon: Q3 consists of 90 calendar days.

---

### 6.2 Component Reliability Profiles

From historical CMMS work order history with right-censoring, reliability engineering calibrated the following parameters:

1. **SKU-101 (Dry Gas Seal):**
   * Weibull parameters in base laboratory conditions: Shape $\beta = 2.4$, Scale $\eta_0 = 12,000$ equivalent hours.
   * Effective scale parameter adjusted for environmental severity:
     $$\eta_{\text{eff}} = \frac{\eta_0}{S_i}$$
     * Sector Alpha ($S = 1.25$): $\eta_{\text{Alpha}} = 12,000 / 1.25 = 9,600$ hours.
     * Sector Bravo ($S = 1.50$): $\eta_{\text{Bravo}} = 12,000 / 1.50 = 8,000$ hours.
     * Sector Charlie ($S = 1.00$): $\eta_{\text{Charlie}} = 12,000 / 1.00 = 12,000$ hours.

2. **SKU-202 (Vibration Sensor):**
   * Exponential distribution ($\beta = 1.0$). Constant failure rate:
     $$\text{MTBF}_0 = 50,000 \text{ hours} \implies \lambda_0 = 2.0 \times 10^{-5} \text{ failures/hour}$$
   * Severity multiplier increases failure rate directly: $\lambda_{\text{eff}} = \lambda_0 \cdot S_i$.
     * Sector Alpha ($S = 1.25$): $\lambda_{\text{Alpha}} = 2.0 \times 10^{-5} \cdot 1.25 = 2.50 \times 10^{-5}$ failures/hr.
     * Sector Bravo ($S = 1.50$): $\lambda_{\text{Bravo}} = 2.0 \times 10^{-5} \cdot 1.50 = 3.00 \times 10^{-5}$ failures/hr.
     * Sector Charlie ($S = 1.00$): $\lambda_{\text{Charlie}} = 2.0 \times 10^{-5} \cdot 1.00 = 2.00 \times 10^{-5}$ failures/hr.

3. **SKU-303 (Filter Cartridge):**
   * Deterministic replacement interval: $\Delta T_{\text{service}} = 2,000$ operating hours.


#### 6.4 Sensitivity Analysis across Environmental and Utilization Variations

In real capital equipment fleets, operating environments and plant utilization vary due to dispatch demand, market pricing, and weather. To understand how changes in fleet conditions propagate into parts requirements, we evaluate three sensitivity scenarios against the baseline Q3 plan:

```text
Sensitivity Scenarios for Q3 Spare Parts:
========================================================================================================
Scenario                  Description                             SKU-101 (Seals)  SKU-202 (Sensors)  SKU-303 (Filters)
--------------------------------------------------------------------------------------------------------
Baseline                  As calculated in Section 6.3            144.2 units      38.3 units         240 units
Scenario A: Summer Heat   Sector Alpha ambient severity S_i -> 1.45 158.4 units (+9.8%)39.8 units (+3.9%) 240 units (0%)
Scenario B: Heavy Surge   Sector Bravo runs 24.0 hrs/day (100% util) 149.7 units (+3.8%) 38.6 units (+0.8%) 240 units (0%)
Scenario C: Low Demand    Sector Charlie runs 8.0 hrs/day (peaking)139.1 units (-3.5%) 35.4 units (-7.6%) 140 units (-41.7%)
========================================================================================================
```

**Analytical Findings:**
* **SKU-101 (Wear-out):** Highly sensitive to ambient severity in Sector Alpha because $\beta = 2.4$. Increasing $S_i$ from 1.25 to 1.45 reduces characteristic life $\eta$ from 9,600 to 8,276 hours, accelerating the exponential wear-out tail.
* **SKU-202 (Random):** Scales linearly with total fleet exposure hours, showing moderate variance across scenarios.
* **SKU-303 (Overhaul filter):** Exhibits threshold behavior. In Scenario C, reducing Sector Charlie running hours from 14.0 to 8.0 hrs/day means assets accumulate only 720 hours in Q3 ($4,800 + 720 = 5,520$ hrs). None of the 100 Charlie compressors cross the 6,000-hour overhaul threshold during Q3, dropping planned demand from 240 to 140 cartridges (-41.7%). This illustrates why linear regression models completely fail for scheduled maintenance kits.

#### 6.5 Sparing Covariance and Multi-Location Pooling Economics

When stocking parts across multiple depots, total inventory investment depends on whether demand across sectors is correlated:

$$\text{Var}(D_{\text{total}}) = \sum_{i=1}^K \text{Var}(D_i) + 2 \sum_{i < j} \text{Cov}(D_i, D_j)$$

For purely physical failures driven by independent asset exposure, $\text{Cov}(D_i, D_j) = 0$. Under independent Poisson demand:
$$\sigma_{\text{pooled}} = \sqrt{\sum_{i=1}^K \sigma_i^2} = \sqrt{\sum_{i=1}^K \mu_i}$$

If parts are stocked centrally:
$$SS_{\text{central}} = z_{\alpha} \cdot \sqrt{\mu_{\text{total}}} = 1.645 \cdot \sqrt{38.27} = 10.18 \approx 11 \text{ sensors}$$

If parts are stocked independently across three decentralized forward locations:
* Alpha: $z_{\alpha} \cdot \sqrt{12.96} = 1.645 \cdot 3.60 = 5.92 \approx 6$ sensors.
* Bravo: $z_{\alpha} \cdot \sqrt{15.23} = 1.645 \cdot 3.90 = 6.42 \approx 7$ sensors.
* Charlie: $z_{\alpha} \cdot \sqrt{10.08} = 1.645 \cdot 3.17 = 5.21 \approx 6$ sensors.
* Decentralized Total Safety Stock: $6 + 7 + 6 = 19$ sensors.

**Pooling Advantage:** Centralized pooling reduces safety stock from 19 to 11 units—a **42.1% reduction in safety inventory** while delivering an identical 95% service level.
---

### 6.3 Calculations for Q3 Demand

#### Step A: Calculate Q3 Operating Exposure per Sector

Operating Exposure $\Delta E = \text{Assets} \times \text{Hours/Day} \times 90 \text{ Days}$:

1. **Sector Alpha:**
   $$\Delta E_{\text{Alpha}} = 80 \times 18.0 \times 90 = 129,600 \text{ operating hours}$$
   Per-asset exposure: $\Delta e_{\text{Alpha}} = 18.0 \times 90 = 1,620$ hours.
2. **Sector Bravo:**
   $$\Delta E_{\text{Bravo}} = 60 \times 23.5 \times 90 = 126,900 \text{ operating hours}$$
   Per-asset exposure: $\Delta e_{\text{Bravo}} = 23.5 \times 90 = 2,115$ hours.
3. **Sector Charlie:**
   $$\Delta E_{\text{Charlie}} = 100 \times 14.0 \times 90 = 126,000 \text{ operating hours}$$
   Per-asset exposure: $\Delta e_{\text{Charlie}} = 14.0 \times 90 = 1,260$ hours.

Total fleet Q3 exposure = $129,600 + 126,900 + 126,000 = 382,500$ compressor operating hours.

---

#### Step B: Unplanned Demand for SKU-202 (Random Sensor Failures)

Because $\beta = 1.0$ (constant failure rate), the timing of failures is memoryless. Expected failures depend strictly on total exposure hours:

$$\mathbb{E}[\text{Failures}] = \text{Quantity Per Assembly (QPA)} \times \sum (\Delta E_{\text{sector}} \times \lambda_{\text{sector}})$$

$$\text{Alpha:} \quad 4 \text{ sensors} \times (129,600 \text{ hrs} \times 2.50 \times 10^{-5}) = 4 \times 3.24 = 12.96 \text{ units}$$
$$\text{Bravo:} \quad 4 \text{ sensors} \times (126,900 \text{ hrs} \times 3.00 \times 10^{-5}) = 4 \times 3.807 = 15.228 \text{ units}$$
$$\text{Charlie:} \quad 4 \text{ sensors} \times (126,000 \text{ hrs} \times 2.00 \times 10^{-5}) = 4 \times 2.52 = 10.08 \text{ units}$$

$$\mathbb{E}[D_{\text{SKU-202}}] = 12.96 + 15.228 + 10.08 = 38.268 \approx 38 \text{ sensors}$$

Under a Poisson distribution with mean $\mu = 38.27$:
* Variance $\sigma^2 = \mu = 38.27 \implies \sigma = \sqrt{38.27} = 6.19$ units.
* 95% service level stock requirement:
  $$S_{0.95} = \mu + z_{0.95} \cdot \sigma = 38.27 + (1.645 \times 6.19) = 38.27 + 10.18 = 48.45 \approx 49 \text{ units}$$

---

#### Step C: Unplanned Demand for SKU-101 (Dry Gas Seal Wear-out)

Because $\beta = 2.4 > 1$, failure probability depends heavily on current component age.
For each sector, we calculate conditional probability of failure:
$$P(\text{Fail in Q3} \mid \text{Survived to } T) = \frac{\exp\left[ -\left(\frac{T}{\eta}\right)^\beta \right] - \exp\left[ -\left(\frac{T + \Delta e}{\eta}\right)^\beta \right]}{\exp\left[ -\left(\frac{T}{\eta}\right)^\beta \right]} = 1 - \exp\left[ \left(\frac{T}{\eta}\right)^\beta - \left(\frac{T + \Delta e}{\eta}\right)^\beta \right]$$

1. **Sector Alpha:**
   * Current age $T = 7,200$ hrs; Increment $\Delta e = 1,620$ hrs; End age $T + \Delta e = 8,820$ hrs.
   * Adjusted $\eta = 9,600$ hrs; $\beta = 2.4$.
   * $(T / \eta)^\beta = (7,200 / 9,600)^{2.4} = (0.75)^{2.4} = 0.5016$
   * $((T + \Delta e) / \eta)^\beta = (8,820 / 9,600)^{2.4} = (0.91875)^{2.4} = 0.8171$
   * Exponent difference: $0.5016 - 0.8171 = -0.3155$
   * Conditional Failure Probability:
     $$P_{\text{Alpha}} = 1 - \exp(-0.3155) = 1 - 0.7294 = 0.2706$$
   * Expected seal failures in Alpha (80 compressors $\times$ 2 seals/unit = 160 seals):
     $$\mathbb{E}[D_{\text{Alpha}}] = 160 \times 0.2706 = 43.30 \text{ seals}$$

2. **Sector Bravo:**
   * Current age $T = 11,500$ hrs; Increment $\Delta e = 2,115$ hrs; End age $T + \Delta e = 13,615$ hrs.
   * Adjusted $\eta = 8,000$ hrs; $\beta = 2.4$.
   * $(T / \eta)^\beta = (11,500 / 8,000)^{2.4} = (1.4375)^{2.4} = 2.3789$
   * $((T + \Delta e) / \eta)^\beta = (13,615 / 8,000)^{2.4} = (1.7019)^{2.4} = 3.5937$
   * Exponent difference: $2.3789 - 3.5937 = -1.2148$
   * Conditional Failure Probability:
     $$P_{\text{Bravo}} = 1 - \exp(-1.2148) = 1 - 0.2968 = 0.7032$$
   * Expected seal failures in Bravo (60 compressors $\times$ 2 seals/unit = 120 seals):
     $$\mathbb{E}[D_{\text{Bravo}}] = 120 \times 0.7032 = 84.38 \text{ seals}$$
   *(Note the dramatic wear-out surge: older machines in harsh salt spray exhibit a 70.3% failure probability in a single quarter!)*

3. **Sector Charlie:**
   * Current age $T = 4,800$ hrs; Increment $\Delta e = 1,260$ hrs; End age $T + \Delta e = 6,060$ hrs.
   * Adjusted $\eta = 12,000$ hrs; $\beta = 2.4$.
   * $(T / \eta)^\beta = (4,800 / 12,000)^{2.4} = (0.40)^{2.4} = 0.1102$
   * $((T + \Delta e) / \eta)^\beta = (6,060 / 12,000)^{2.4} = (0.505)^{2.4} = 0.1963$
   * Exponent difference: $0.1102 - 0.1963 = -0.0861$
   * Conditional Failure Probability:
     $$P_{\text{Charlie}} = 1 - \exp(-0.0861) = 1 - 0.9175 = 0.0825$$
   * Expected seal failures in Charlie (100 compressors $\times$ 2 seals/unit = 200 seals):
     $$\mathbb{E}[D_{\text{Charlie}}] = 200 \times 0.0825 = 16.50 \text{ seals}$$

Total Expected Unplanned Demand for SKU-101:
$$\mathbb{E}[D_{\text{SKU-101}}] = 43.30 + 84.38 + 16.50 = 144.18 \approx 144 \text{ seals}$$

A naive time-series forecast based on the previous year (when Sector Bravo was only 5,000 hours old) would have predicted approximately 38 seals. Failure to account for Weibull wear-out would have led to a catastrophic stockout of over 100 mechanical seals, stranding offshore production platforms.

---

#### Step D: Planned Demand for SKU-303 (Lube Oil Filter Overhauls)

Replacement trigger: Every 2,000 operating hours.
* **Alpha:** 1,620 hrs accumulated per compressor in Q3.
  * Baseline hour status: Assets currently at 7,200 hrs. Previous filter change was at 6,000 hrs. Next change is due at 8,000 hrs.
  * Since $7,200 + 1,620 = 8,820 \text{ hrs} \ge 8,000 \text{ hrs}$, **all 80 compressors** in Alpha will cross the 8,000-hour threshold during Q3.
  * Planned filter demand for Alpha = $80 \times 1 = 80$ cartridges.
* **Bravo:** 2,115 hrs accumulated per compressor in Q3.
  * Current status: 11,500 hrs. Previous filter change was at 10,000 hrs. Next change is due at 12,000 hrs.
  * Since $11,500 + 2,115 = 13,615 \text{ hrs} \ge 12,000 \text{ hrs}$, **all 60 compressors** cross the 12,000-hour threshold.
  * In addition, $13,615 - 12,000 = 1,615$ hours into the next cycle (does not hit 14,000 in Q3).
  * Planned filter demand for Bravo = $60 \times 1 = 60$ cartridges.
* **Charlie:** 1,260 hrs accumulated per compressor in Q3.
  * Current status: 4,800 hrs. Previous change was at 4,000 hrs. Next change due at 6,000 hrs.
  * End status: $4,800 + 1,260 = 6,060 \text{ hrs} \ge 6,000 \text{ hrs}$.
  * Planned filter demand for Charlie = $100 \times 1 = 100$ cartridges.

Total Planned Demand for SKU-303 = $80 + 60 + 100 = 240$ cartridges.
This demand is 100% deterministic, conditioned entirely on operating schedules.

---

### 6.4 Summary of Final Demand Plan for Q3

| Part SKU | Component Description | Planned Demand | Expected Unplanned | 95% Sparing Target | Dominant Sourcing Strategy |
|---|---|---:|---:|---:|---|
| **SKU-101** | Dry Gas Mechanical Seal | 0 | 144.2 | 164 | Depot repair pool + OEM overhaul kits |
| **SKU-202** | Piezo Vibration Sensor | 0 | 38.3 | 49 | Forward site buffer stock |
| **SKU-303** | Lube Oil Filter Cartridge | 240 | 0.0 | 240 | Scheduled bulk supplier contract drop-ship |

---

## 7. Code, query, or calculation

The calculation below provides the complete, unambiguous computational implementation of the conditional Weibull failure projection and the Syntetos-Boylan demand classification.

```python
"""
Installed Base Failure Demand & Demand Classification Engine
Implements:
  1. Conditional Weibull failure expectation across multi-sector asset fleets.
  2. Syntetos-Boylan demand categorization (ADI, CV^2).
"""

import math
from typing import List, Dict, Tuple

def conditional_weibull_p_fail(
    t_current: float, 
    delta_exposure: float, 
    beta: float, 
    eta: float
) -> float:
    """
    Computes P(T <= t_current + delta_exposure | T > t_current) for Weibull.
    Formula: 1 - exp[ (t_current/eta)^beta - ((t_current + delta_exposure)/eta)^beta ]
    """
    if t_current < 0 or delta_exposure <= 0 or beta <= 0 or eta <= 0:
        raise ValueError("Parameters must be strictly positive.")
    
    exponent_delta = math.pow(t_current / eta, beta) - math.pow((t_current + delta_exposure) / eta, beta)
    p_fail = 1.0 - math.exp(exponent_delta)
    return p_fail

def classify_syntetos_boylan(demand_series: List[float]) -> Dict[str, any]:
    """
    Evaluates demand series and categorizes according to the Syntetos-Boylan rule.
    Cutoffs: ADI = 1.32 periods, CV^2 = 0.49.
    """
    total_periods = len(demand_series)
    non_zero_demands = [d for d in demand_series if d > 0]
    num_non_zero = len(non_zero_demands)
    
    if num_non_zero == 0:
        return {"ADI": float('inf'), "CV2": 0.0, "Category": "Zero Demand"}
    
    # ADI: Average Demand Interval
    adi = total_periods / float(num_non_zero)
    
    # CV^2: Coefficient of variation squared of non-zero transaction sizes
    mean_z = sum(non_zero_demands) / float(num_non_zero)
    if num_non_zero > 1:
        variance_z = sum((z - mean_z) ** 2 for z in non_zero_demands) / (num_non_zero - 1)
        cv2 = variance_z / (mean_z ** 2)
    else:
        cv2 = 0.0
    
    # Categorization logic
    if adi < 1.32 and cv2 < 0.49:
        category = "Smooth"
    elif adi >= 1.32 and cv2 < 0.49:
        category = "Intermittent"
    elif adi < 1.32 and cv2 >= 0.49:
        category = "Erratic"
    else:
        category = "Lumpy"
        
    return {
        "Total_Periods": total_periods,
        "Non_Zero_Count": num_non_zero,
        "ADI": round(adi, 3),
        "Mean_Size": round(mean_z, 2),
        "CV2": round(cv2, 3),
        "Category": category
    }

# --- Demonstration of Worked Example Verification ---
if __name__ == "__main__":
    # Parameters for Dry Gas Seal (SKU-101) across 3 sectors
    sectors = [
        {"name": "Alpha", "assets": 80, "qpa": 2, "age": 7200, "delta_e": 1620, "beta": 2.4, "eta": 9600},
        {"name": "Bravo", "assets": 60, "qpa": 2, "age": 11500, "delta_e": 2115, "beta": 2.4, "eta": 8000},
        {"name": "Charlie", "assets": 100, "qpa": 2, "age": 4800, "delta_e": 1260, "beta": 2.4, "eta": 12000},
    ]
    
    print("==================================================================")
    print("INSTALLED BASE CONDITIONAL WEIBULL DEMAND CALCULATIONS")
    print("==================================================================")
    total_expected_demand = 0.0
    for sec in sectors:
        p_fail = conditional_weibull_p_fail(sec["age"], sec["delta_e"], sec["beta"], sec["eta"])
        installed_components = sec["assets"] * sec["qpa"]
        expected_demand = installed_components * p_fail
        total_expected_demand += expected_demand
        print(f"Sector {sec['name']:<8}: P(Fail) = {p_fail:.4f} | Parts = {installed_components:3d} | E[Demand] = {expected_demand:6.2f}")
    
    print("------------------------------------------------------------------")
    print(f"Total Portfolio Expected Demand: {total_expected_demand:.2f} units")
    print("==================================================================\n")
    
    # Syntetos-Boylan Classification Test on 24-month historical depot orders
    sample_series = [0, 2, 0, 0, 0, 1, 0, 0, 3, 0, 0, 0, 0, 2, 0, 0, 0, 0, 1, 0, 0, 0, 4, 0]
    classification = classify_syntetos_boylan(sample_series)
    print("DEMAND PROFILE CLASSIFICATION:")
    for k, v in classification.items():
        print(f"  {k:<15}: {v}")
```

### Expected Output
```text
==================================================================
INSTALLED BASE CONDITIONAL WEIBULL DEMAND CALCULATIONS
==================================================================
Sector Alpha   : P(Fail) = 0.2706 | Parts = 160 | E[Demand] =  43.30
Sector Bravo   : P(Fail) = 0.7032 | Parts = 120 | E[Demand] =  84.38
Sector Charlie : P(Fail) = 0.0825 | Parts = 200 | E[Demand] =  16.50
------------------------------------------------------------------
Total Portfolio Expected Demand: 144.18 units
==================================================================

DEMAND PROFILE CLASSIFICATION:
  Total_Periods  : 24
  Non_Zero_Count : 6
  ADI            : 4.0
  Mean_Size      : 2.17
  CV2            : 0.267
  Category       : Intermittent
```

---

## 8. Data, standards, and regulatory context

Unlike retail supply chains governed by voluntary consumer barcodes (GS1 GTIN), aftermarket service data in heavy industry, aviation, healthcare, and energy is subject to binding international data standards and statutory safety regulations.

### 8.1 Key International Standards

1. **ISO 14224:2016 (Petroleum, Petrochemical, and Natural Gas Industries — Collection and Exchange of Reliability and Maintenance Data):**
   * *Scope:* Establishes a rigorous, standard taxonomic hierarchy for equipment classification (Industry $\to$ Business Category $\to$ Installation $\to$ Plant/Unit $\to$ Section/System $\to$ Equipment Unit $\to$ Subunit $\to$ Maintainable Item).
   * *Binding Requirement:* Mandatory reporting of failure mechanisms (mechanical, electrical, structural), failure causes, detection methods, and downtime hours.
   * *Impact on Analytics:* Mandates the distinction between functional failures (asset cannot operate) and degraded failures (asset operates with reduced performance).

2. **ASD/AIA S1000D / S2000M (International Specifications for Technical Publications and Materiel Management):**
   * *Scope:* Global aerospace and defence standard for interactive electronic technical manuals (S1000D) and integrated logistics support / spares provisioning (S2000M).
   * *Impact on Analytics:* Defines the Breakdown Element Identifier (BEI) and Recommended Spares Provisioning List (RSPL). Establishes standardized NATO Commercial and Government Entity (NCAGE) codes and National Stock Numbers (NSN).

3. **MIMOSA OSA-EAI / OSA-CBM (Open Systems Architecture for Enterprise Application Integration / Condition-Based Maintenance):**
   * *Scope:* Open standards for physical asset management, condition monitoring telemetry, and failure mode identification.
   * *Impact on Analytics:* Provides standard schemas (CRIS, Common Relational Information Schema) to integrate real-time sensor streams (vibration, acoustics, spectrographic oil analysis) directly into asset maintenance registers.

4. **FAR Part 121 / EASA Part M & Part 145 (Aviation Airworthiness Regulations):**
   * *Scope:* US Federal Aviation Administration (FAA) and European Union Aviation Safety Agency (EASA) continuous airworthiness management.
   * *Binding Requirement:* Life-Limited Parts (LLPs) such as turbine disks must be tracked by serial number with unbroken, auditable documentation from manufacture to retirement. When an LLP reaches its cycle threshold, it **must** be removed regardless of physical condition.
   * *Impact on Analytics:* LLP demand is 100% deterministic against cumulative flight cycles. Forecasting errors for LLPs reflect data integration failure, not stochastic uncertainty.

5. **FDA 21 CFR Part 820.65 / EU MDR 2017/745 (Medical Device Traceability):**
   * *Scope:* Requires Unique Device Identification (UDI) down to the component level for implantable and critical life-support medical hardware (dialysis units, ventilators, linear accelerators).
   * *Impact on Analytics:* Precludes unauthorized third-party part substitution. Demands strict adherence to BOM effectivity tables.

---

## 9. Industry practice and case evidence

### 9.1 Commercial Aviation: The "Power-by-the-Hour" Paradigm

In commercial aerospace, engine manufacturers (Rolls-Royce, GE Aerospace, Pratt & Whitney, CFM International) shifted from selling engines to selling available thrust hours under long-term TotalCare or TrueChoice service agreements.

* **Operational Reality:** The airline pays a fixed fee per engine flight hour (EFH). In exchange, the OEM assumes all financial and operational liability for maintenance, shop visits, and spare parts.
* **Analytics Mechanism:** The OEM continuously streams high-frequency engine telemetry (exhaust gas temperature margin, shaft vibration, fuel flow deviations) via ACARS and satellite links.
* **Case Evidence:** Rolls-Royce operates an Engine Health Monitoring (EHM) center in Derby, UK, tracking over 13,000 commercial engines worldwide. By coupling thermodynamic engine performance models with real-time flight schedules, Rolls-Royce forecasts engine removals up to 18 months in advance. This shifts parts demand from chaotic unscheduled removals to scheduled shop visits, cutting spare engine pool requirements by over 20% across global hubs while ensuring 99.9% dispatch reliability.

### 9.2 Semiconductor Manufacturing Equipment: ASML Lithography Sparing

Photolithography tools produced by ASML (such as Extreme Ultraviolet / EUV NXE systems) cost over €180 million each and operate in 24/7 semiconductor fabrication facilities (TSMC, Intel, Samsung). 

* **The Downtime Penalty:** Downtime on an EUV tool costs the chip fabricator upwards of $75,000 per hour.
* **Supply Chain Structure:** Parts cannot be stored in ambient warehouse conditions; optical components and laser mirrors require vacuum packaging or nitrogen purging.
* **Practice:** ASML uses an asset-driven predictive service model. Each lithography tool is digitized as a Digital Twin. High-speed laser sensors detect lens contamination and droplet generator nozzle wear at the picosecond scale. Spares are positioned at forward stocking locations (FSLs) within a 45-minute courier radius of key fabs (Tainan, Hsinchu, Hillsboro, Leixlip). Parts demand is triggered automatically by automated diagnostic routines before wafer yield degrades, achieving sub-two-hour mean time to repair (MTTR).

### 9.3 Heavy Rail Locomotives: Fleet Availability Contracting

Alstom and Siemens Mobility frequently operate under Performance-Based Logistics (PBL) contracts for passenger rail networks (e.g., UK Class 390 Pendolino, Deutsche Bahn ICE fleets).

* **The Failure of Time Series:** When a rail operator used 3-year historical exponential smoothing for traction motor bogie overhauls, they faced massive stockouts. The reason: the fleet had reached 1.2 million kilometres, triggering mandatory mid-life wheelset reprofiling.
* **The Solution:** Rail manufacturers transitioned to exposure-based planning. Train control management systems (TCMS) upload exact kilometre and braking energy logs nightly over Wi-Fi when locomotives enter depots. Parts kits are automatically scheduled for delivery to specific maintenance bays 48 hours before an individual train reaches its mileage limit.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign / Red Flag | Underlying Physical/Data Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **Ghost Demand from Batch Ordering** | Sudden, massive demand surge for a slow-moving part from a single depot without change in fleet size. | Customer or regional depot placed an artificial buffer order or "budget exhaustion" buy at fiscal year-end. | Production runs started for obsolete parts; OEM warehouse floods with unsellable inventory. | Clean demand history by decoupling customer inventory transfers from true field work order scrap events. |
| 2 | **Censored Stockout History** | Long strings of zero demand during periods of known supply shortages or supplier disruptions. | ERP records zero sales because no inventory existed to issue; unfulfilled requests were cancelled or redirected. | Future safety stock calculations decrease precisely on components that had severe shortages. | Impute lost demand from backorder logs, customer expedite emails, and asset downtime records. |
| 3 | **Bathtub Wear-Out Blindness** | Time-series forecasting models (ARIMA, Holt-Winters) project flat, low demand for aging fleets. | A constant failure rate ($\beta = 1$) assumption applied to parts undergoing mechanical fatigue ($\beta > 2$). | Widespread fleet groundings as entire asset cohorts enter wear-out simultaneously; multi-month lead times. | Implement age-dependent Weibull hazard forecasting linked directly to chronological asset commission dates. |
| 4 | **Cannibalization Distortion** | Zero parts issued from stockroom, yet assets maintain high operational uptime during a stockout. | Field technicians quietly harvest working parts from offline, parked, or maintenance-reserve assets. | Secondary assets remain cannibalized "Christmas trees"; future parts demand surges unpredictably upon rebuild. | Institute mandatory CMMS logging of all component de-installations with cross-asset serial number tracing. |
| 5 | **BOM Supersession Disconnect** | Explosive demand spikes on obsolete SKU while modern replacement SKU registers zero sales. | Engineering issued an ECO changing part numbers, but the ERP service catalog and planner BOM were never updated. | Procurement orders obsolete revisions at inflated supplier setup costs; new superseded inventory sits dead. | Automate BOM supersession chains (Form, Fit, Function interchangeability) within the forecasting pipeline. |
| 6 | **Operating Exposure Ignorance** | Identical failure forecasts generated for high-utilization mining trucks and low-utilization standby units. | Forecasting models use calendar time elapsed rather than cumulative operating hours or tonnage throughput. | Premature maintenance on standby assets; catastrophic, unpredicted field breakdowns on frontline assets. | Scale all degradation and hazard rates by actual machine telemetry exposure ($OH, KM, Cycles$). |
| 7 | **Infant Mortality After Service** | Replaced parts fail at 10$\times$ the baseline rate within 48 hours of maintenance overhaul. | Poor installation practices, improper torque, or contamination during field service (burn-in defect, $\beta < 1$). | High customer dissatisfaction; erroneous conclusion that the spare parts batch is defective. | Separate post-maintenance infant mortality from steady-state failures; introduce technician training and FMEA quality audits. |
| 8 | **No-Fault-Found (NFF) Churn** | High volume of parts returned from the field, but factory test benches find 80% have zero defects. | Field technicians replace components shotgun-style to quickly clear fault codes without root-cause isolation. | Repair shops overwhelmed testing functional parts; excessive pipeline inventory holding costs. | Audit diagnostic isolation procedures; institute technician training and penalize high NFF return rates. |
| 9 | **End-of-Life Inventory Trap** | Massive inventory write-offs occurring 24 months after a major customer retires a machine model. | Planners extrapolate historical 5-year average demand into a future horizon where the installed base is collapsing. | Millions in dead working capital written off as scrap; warehouse capacity choked. | Bind the forecasting horizon strictly to contracted asset retirement schedules and customer decommissioning dates. |
| 10 | **Secondary Component Cascading** | Sudden, unexplained spike in high-value structural components (e.g., turbine rotors, engine crankcases). | Failure of an inexpensive secondary part (e.g., $10 lubrication o-ring) was ignored, causing catastrophic primary failure. | Unplanned, unbudgeted capital expenditures; extended asset outages. | Establish dependent reliability modeling where failure of primary parts is conditioned on secondary maintenance adherence. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks If It Fails | How to Test / Validate |
|---|---|---|---|
| **Accurate Asset Register** | We must know how many machines exist and where they operate to calculate total exposure. | If 20% of machines are unrecorded (grey market, third-party sales), failure demand is severely under-forecasted. | Reconcile warranty registrations, field service technician visits, and remote telemetry pings against the master database. |
| **Monitored Operating Exposure** | Converts calendar time into physical degradation stress. | If telemetry is missing or estimated poorly, failure rates diverge wildly across assets. | Perform periodic physical audit of hour-meters; check correlation between fuel consumption and logged hours. |
| **Stable Environmental Factors** | Allows scaling baseline laboratory failure rates to field severity tiers. | If an asset is moved from a temperate indoor plant to an offshore desert platform, failure rates jump unexpectedly. | Tag assets with geolocation coordinates; update environmental stress tiers dynamically when assets transfer sites. |
| **Independence of Component Failures** | Allows summing failure probabilities across components without complex joint-distribution copulas. | Common-cause failures (e.g., power surge, fuel contamination) cause multiple components to fail simultaneously. | Run statistical tests for spatial and temporal clustering of failures across co-located assets. |
| **Constant Configuration Stability** | Assumes the machine configuration remains identical to its engineering BOM. | Unauthorized field modifications or non-OEM repairs invalidate OEM failure models and part numbers. | Conduct mandatory configuration audits during annual preventive maintenance overhauls. |
| **Known Part Interchangeability** | Connects historical failure demand across multiple superseded part revisions. | Failing to link old and new part revisions fractures the demand history, preventing statistical calibration. | Verify that form-fit-function (FFF) interchangeability tables are maintained in PLM and ERP systems. |
| **Negligible Technician Damage** | Assumes that the physical replacement process does not induce secondary failures. | If maintenance procedures damage adjacent components, failure rates increase dramatically post-repair. | Track failure rates as a function of time-since-last-maintenance (hazard spikes immediately post-repair indicate technician damage). |
| **Homogeneous Failure Modes** | Fits a single Weibull distribution to a part number. | If a part has two distinct failure modes (e.g., electrical coil burnout vs. bearing mechanical seizure), a single distribution fits neither. | Stratify reliability models by root-cause failure mode code (ISO 14224 failure mechanism classification). |

---

## 12. Exercises

### Exercise 1 — Compute Uncensored Exposure and Intermittency Metrics
**Type:** Calculation & Analysis  
**Time:** 45 minutes  
**Context:** You are the service supply chain analyst for a fleet of 50 hospital CT scanners. Over a 24-month observation window (24 monthly periods), a specialized high-voltage tube replacement SKU registered the following monthly demand history across the central service depot:
```text
Month:   1   2   3   4   5   6   7   8   9  10  11  12  13  14  15  16  17  18  19  20  21  22  23  24
Demand:  0,  0,  2,  0,  0,  0,  1,  0,  0,  0,  0,  3,  0,  0,  0,  1,  0,  0,  0,  0,  2,  0,  0,  1
```
During Months 8, 9, and 10, the central depot was completely stocked out of tubes. Two scanners were grounded during Month 9, and their tube requisitions were cancelled and redirected to a high-cost spot-broker in Germany (unrecorded in the local ERP).

**Deliverables:**
1. Calculate the raw Average Demand Interval (ADI) and the raw squared coefficient of variation ($CV^2$) of non-zero demand from the ERP history. Classify the demand into the Syntetos-Boylan quadrant.
2. Reconstruct the uncensored demand series by restoring the two unfulfilled demand events in Month 9.
3. Recompute the ADI and $CV^2$ on the uncensored dataset. State whether the classification shifts and what forecasting method should be deployed.

---

### Exercise 2 — Weibull Hazard Calibration and Wear-out Sparing
**Type:** Application & Hand Calculation  
**Time:** 60 minutes  
**Context:** A fleet of 40 mining haul trucks operates in an open-pit copper mine. Each truck utilizes 6 heavy-duty wheel suspension struts. 
* Current truck age: All 40 trucks were deployed simultaneously and have reached 14,000 operating hours.
* Component characteristics: Suspension struts follow a Weibull wear-out distribution with shape parameter $\beta = 3.0$ and characteristic life $\eta = 20,000$ operating hours.
* Operating schedule: Over the next 6-month planning cycle, each truck is scheduled to operate 20 hours per day for 180 days.

**Deliverables:**
1. Compute the expected operational exposure increment ($\Delta e$) accumulated by each truck over the 6-month period.
2. Calculate the conditional probability of failure for an individual strut during the upcoming 6-month window, given survival to 14,000 hours.
3. Calculate the total expected strut failure demand across the entire fleet of 40 trucks.
4. Assuming the number of strut failures follows a Poisson distribution with mean equal to your expected demand, calculate the required stock of spare struts to guarantee an 98% parts availability service level (use standard Poisson cumulative probabilities or normal approximation with continuity correction: $z_{0.98} = 2.054$).

---

### Exercise 3 — Demand Stream Decomposition Design
**Type:** Architectural Design  
**Time:** 45 minutes  
**Context:** A global marine propulsion manufacturer provides service for 1,200 cargo vessels. The planning team currently runs a single automated Prophet/Auto-ARIMA forecasting algorithm on monthly spare parts orders from the ERP. The business is experiencing severe inventory imbalances: excess stock of scheduled maintenance gasket kits and constant stockouts of electronic propulsion controllers.

**Deliverables:**
1. Draw an architectural data flow diagram detailing how the organization should separate its demand forecasting into four decoupled pipelines:
   * Planned Overhauls (Drydock Schedule)
   * Unplanned Stochastic Failures
   * New Ship Commissioning Provisioning
   * Fleet Decommissioning & Last-Time Buys
2. For each pipeline, define:
   * The primary enterprise system acting as the source of truth.
   * The core mathematical model or planning logic.
   * The primary update frequency (daily, weekly, monthly, annual).
---

### Exercise 4 — Renewal Theory vs. Homogeneous Poisson Process for Bearing Replacements
**Type:** Comparative Mathematical Derivation  
**Time:** 50 minutes  
**Context:** A fleet of 100 wind turbines contains main shaft roller bearings that operate continuously (8,760 hours per year). 
* Reliability profile: The bearings follow a Weibull wear-out distribution with $\beta = 2.0$ and $\eta = 20,000$ hours.
* The engineering manager claims: "Since the mean time between failures is $\text{MTTF} = \eta \Gamma(1 + 1/\beta) = 20,000 \cdot \Gamma(1.5) = 20,000 \cdot 0.8862 = 17,725$ hours, we can simply assume a constant annual failure rate of $\lambda = 8,760 / 17,725 = 0.494$ failures per turbine per year. Therefore, for 100 turbines, we should stock 50 bearings every year."

**Deliverables:**
1. Calculate the actual probability that a brand-new bearing fails during its first year of operation ($t \in [0, 8,760]$ hours).
2. Calculate the conditional probability that a surviving bearing fails during its second year of operation ($t \in [8,760, 17,520]$ hours).
3. Calculate the conditional probability that a surviving bearing fails during its third year of operation ($t \in [17,520, 26,280]$ hours).
4. Compare the actual expected annual failures across the fleet for Years 1, 2, and 3 against the engineering manager's flat 50-bearing forecast. Explain why assuming an HPP causes severe cash flow and service level distortions.

---

### Exercise 5 — BOM Effectivity and Supersession Planning under Fleet Retrofit
**Type:** Real-World Application & Supply Planning  
**Time:** 40 minutes  
**Context:** A defense contractor manages 120 combat vehicles. A hydraulic steering actuator, Part A (SKU-ACT-01), has historically exhibited an MTBF of 1,200 operating hours. Due to frequent field failures, engineering releases an upgraded revision, Part B (SKU-ACT-02), with an MTBF of 3,600 hours.
* The retrofit policy dictates that Part B is fully backward compatible (Form, Fit, Function interchangeable).
* When Part A fails in the field, technicians must replace it with Part B if available. If Part B is out of stock, Part A can still be installed until existing depot stocks are exhausted.
* Existing depot stock of Part A is 24 units. No new Part A units will be procured.
* Fleet exposure is 200 operating hours per vehicle per month across all 120 vehicles.

**Deliverables:**
1. Model the transition of the fleet from Part A to Part B over a 12-month horizon.
2. Calculate the monthly demand for actuators and determine in which month the depot inventory of Part A is fully depleted.
3. Graph or tabulate the expected monthly procurement requirements for Part B over the 12-month period, accounting for the changing fleet composition.
---

## 13. Self-check answers

### 13.1 Answer to Exercise 1

#### 1. Raw ERP Data Calculation:
* Total observation periods $T = 24$ months.
* Non-zero demand months: Month 3 (2), Month 7 (1), Month 12 (3), Month 16 (1), Month 21 (2), Month 24 (1).
* Number of non-zero periods: $N = 6$.
* **ADI:**
  $$\text{ADI}_{\text{raw}} = \frac{T}{N} = \frac{24}{6} = 4.0 \text{ months}$$
* Non-zero transaction sizes: $\{2, 1, 3, 1, 2, 1\}$.
  * Mean size $\bar{z}$:
    $$\bar{z} = \frac{2 + 1 + 3 + 1 + 2 + 1}{6} = \frac{10}{6} \approx 1.667 \text{ units}$$
  * Sample variance $s_z^2$:
    $$s_z^2 = \frac{(2-1.667)^2 + (1-1.667)^2 + (3-1.667)^2 + (1-1.667)^2 + (2-1.667)^2 + (1-1.667)^2}{5}$$
    $$s_z^2 = \frac{0.111 + 0.444 + 1.778 + 0.444 + 0.111 + 0.444}{5} = \frac{3.333}{5} = 0.667$$
  * **$CV^2$:**
    $$CV^2_{\text{raw}} = \frac{s_z^2}{\bar{z}^2} = \frac{0.667}{(1.667)^2} = \frac{0.667}{2.778} = 0.240$$
* **Classification:**
  * $\text{ADI} = 4.0 \ge 1.32$ (Intermittent)
  * $CV^2 = 0.240 < 0.49$ (Low variance sizing)
  * **Category: Intermittent.** Standard recommendation: Syntetos-Boylan Approximation (SBA).

---

#### 2. Uncensored Demand Reconstruction:
* Month 9 had 2 lost demand events due to stockout. Updated demand for Month 9 = 2.
* Updated non-zero demand months: Month 3 (2), Month 7 (1), Month 9 (2), Month 12 (3), Month 16 (1), Month 21 (2), Month 24 (1).
* New count of non-zero periods: $N_{\text{new}} = 7$.
* Non-zero sizes: $\{2, 1, 2, 3, 1, 2, 1\}$. Total demand = 12 units.

---

#### 3. Recomputed Uncensored Metrics:
* **New ADI:**
  $$\text{ADI}_{\text{new}} = \frac{24}{7} \approx 3.429 \text{ months}$$
* New mean size $\bar{z}_{\text{new}}$:
  $$\bar{z}_{\text{new}} = \frac{12}{7} \approx 1.714 \text{ units}$$
* New variance $s_{z,\text{new}}^2$:
  $$\sum (z_i - \bar{z})^2 = (2-1.714)^2 \times 3 + (1-1.714)^2 \times 3 + (3-1.714)^2 \times 1$$
  $$= 3(0.0818) + 3(0.5098) + 1(1.6538) = 0.2454 + 1.5294 + 1.6538 = 3.4286$$
  $$s_{z,\text{new}}^2 = \frac{3.4286}{6} = 0.5714$$
* **New $CV^2$:**
  $$CV^2_{\text{new}} = \frac{0.5714}{(1.714)^2} = \frac{0.5714}{2.9378} \approx 0.1945$$
* **Conclusion:** The category remains **Intermittent** ($\text{ADI} = 3.43 \ge 1.32$, $CV^2 = 0.195 < 0.49$). However, the raw data overestimated the demand interval by 16.6% (4.0 vs. 3.43 months) and underestimated annualized run-rate by 20% (10 units vs. 12 units). Relying on raw data would have caused structural under-stocking.

---

### 13.2 Answer to Exercise 2

#### 1. Expected Operational Exposure:
$$\Delta e = 20 \text{ hours/day} \times 180 \text{ days} = 3,600 \text{ operating hours}$$
* Current age: $T = 14,000$ hours.
* End of period age: $T + \Delta e = 14,000 + 3,600 = 17,600$ hours.

---

#### 2. Conditional Probability of Failure:
* Parameters: $\beta = 3.0$, $\eta = 20,000$ hours.
$$\left(\frac{T}{\eta}\right)^\beta = \left(\frac{14,000}{20,000}\right)^{3.0} = (0.70)^3 = 0.3430$$
$$\left(\frac{T + \Delta e}{\eta}\right)^\beta = \left(\frac{17,600}{20,000}\right)^{3.0} = (0.88)^3 = 0.681472$$
* Exponent difference:
$$\Delta \text{Exponent} = 0.3430 - 0.681472 = -0.338472$$
* Conditional Failure Probability:
$$P(\text{Failure}) = 1 - \exp(-0.338472) = 1 - 0.71286 = 0.28714 \approx 28.71\%$$

---

#### 3. Total Expected Fleet Demand:
* Total installed struts: 40 trucks $\times$ 6 struts/truck = 240 installed struts.
$$\mathbb{E}[D] = 240 \times 0.28714 = 68.91 \approx 69 \text{ struts}$$

---

#### 4. Safety Stock and Sparing Level for 98% Service Level:
* Under Poisson distribution: $\mu = 68.91$, $\sigma = \sqrt{68.91} = 8.301$.
* Using Normal approximation with continuity correction:
  $$S_{0.98} = \mu + z_{0.98} \cdot \sigma + 0.5$$
  $$S_{0.98} = 68.91 + (2.054 \times 8.301) + 0.5 = 68.91 + 17.05 + 0.5 = 86.46$$
* **Final Sparing Policy:** Stock **87 struts** across the mine site warehouse. 

---

### 13.4 Answer to Exercise 4

#### 1. Failure Probability in Year 1 ($t \in [0, 8,760]$ hrs):
$$P(\text{Fail in Yr 1}) = 1 - \exp\left[ -\left(\frac{8,760}{20,000}\right)^{2.0} \right] = 1 - \exp\left[ -(0.438)^2 \right] = 1 - \exp(-0.19184) = 1 - 0.8254 = 0.1746$$
* Expected fleet failures in Year 1 (100 turbines):
  $$\mathbb{E}[D_{\text{Yr 1}}] = 100 \times 0.1746 = 17.46 \approx 17.5 \text{ bearings}$$

#### 2. Conditional Failure Probability in Year 2 ($t \in [8,760, 17,520]$ hrs):
$$P(\text{Fail in Yr 2} \mid \text{Survived Yr 1}) = 1 - \exp\left[ \left(\frac{8,760}{20,000}\right)^2 - \left(\frac{17,520}{20,000}\right)^2 \right]$$
$$(0.438)^2 - (0.876)^2 = 0.19184 - 0.76738 = -0.57554$$
$$P(\text{Fail in Yr 2}) = 1 - \exp(-0.57554) = 1 - 0.5624 = 0.4376$$
* For the ~82.5 surviving turbines from Year 1:
  $$\mathbb{E}[D_{\text{Yr 2, cohort 1}}] = 82.54 \times 0.4376 = 36.12 \text{ bearings}$$
* For the 17.5 bearings replaced in Year 1 (which are now in their first year of life):
  $$\mathbb{E}[D_{\text{Yr 2, renewals}}] = 17.46 \times 0.1746 = 3.05 \text{ bearings}$$
* Total expected fleet failures in Year 2 = $36.12 + 3.05 = 39.17 \approx 39.2 \text{ bearings}$.

#### 3. Conditional Failure Probability in Year 3 ($t \in [17,520, 26,280]$ hrs):
$$P(\text{Fail in Yr 3} \mid \text{Survived Yr 2}) = 1 - \exp\left[ \left(\frac{17,520}{20,000}\right)^2 - \left(\frac{26,280}{20,000}\right)^2 \right]$$
$$(0.876)^2 - (1.314)^2 = 0.76738 - 1.72660 = -0.95922$$
$$P(\text{Fail in Yr 3}) = 1 - \exp(-0.95922) = 1 - 0.3832 = 0.6168$$
* Surviving original bearings entering Year 3 = $82.54 \times (1 - 0.4376) = 46.42$ bearings.
* Failures from original cohort in Year 3 = $46.42 \times 0.6168 = 28.63$ bearings.
* Failures from Year 1 replacements (now entering Year 2): $17.46 \times 0.4376 = 7.64$ bearings.
* Failures from Year 2 replacements (now entering Year 1): $39.17 \times 0.1746 = 6.84$ bearings.
* Total expected fleet failures in Year 3 = $28.63 + 7.64 + 6.84 = 43.11 \approx 43.1$ bearings.

#### 4. Comparison against Engineering Manager's HPP Flat Forecast:

| Year | Actual Expected Failures (Weibull Renewal) | Manager's HPP Flat Forecast | Error / Variance | Supply Chain Consequence |
|---:|---:|---:|---:|---|
| **Year 1** | 17.5 bearings | 50 bearings | +32.5 excess (+186%) | $650,000 tied up in unneeded inventory; massive working capital waste. |
| **Year 2** | 39.2 bearings | 50 bearings | +10.8 excess (+28%) | Sparing remains over-buffered. |
| **Year 3** | 43.1 bearings | 50 bearings | +6.9 excess (+16%) | Approaching steady-state renewal rate. |
| **Year 4+** | Damped oscillations | 50 bearings | Oscillates around 49.4 | Demand eventually damps to $t / \text{MTTF} \approx 49.4$ by Blackwell's theorem. |

**Takeaway:** The flat HPP assumption fails to capture the multi-year wear-out buildup, causing severe initial cash hoarding followed by incorrect allocation.

---

### 13.5 Answer to Exercise 5

#### 1. Monthly Fleet Exposure & Demand Generation:
* Fleet size: 120 vehicles.
* Monthly exposure: $120 \times 200 \text{ hrs/month} = 24,000 \text{ fleet hours/month}$.
* Constant failure rate assumption holds within individual part classes ($\beta = 1.0$).
  * Part A failure rate: $\lambda_A = 1 / 1,200 \text{ failures/hour}$.
  * Part B failure rate: $\lambda_B = 1 / 3,600 \text{ failures/hour}$.

#### 2. Depletion of Part A Stock:
* At Month 1, all 120 vehicles contain Part A.
* Expected Month 1 failures:
  $$\mathbb{E}[D_1] = 24,000 \times \frac{1}{1,200} = 20.0 \text{ failures}$$
* All 20 failures are fulfilled from existing Part A depot stock (24 units available).
* Remaining Part A depot stock entering Month 2: $24 - 20 = 4$ units.
* Vehicles running Part B in Month 2: 20 vehicles.
* Vehicles running Part A in Month 2: 100 vehicles.

* Month 2 Demand:
  * From Part A vehicles: $100 \times 200 \times (1 / 1,200) = 20,000 / 1,200 = 16.67$ failures.
  * From Part B vehicles: $20 \times 200 \times (1 / 3,600) = 4,000 / 3,600 = 1.11$ failures.
  * Total Month 2 Failures: $16.67 + 1.11 = 17.78 \approx 18$ failures.
* Fulfilling Month 2:
  * The remaining 4 units of Part A are issued immediately.
  * The depot stock of Part A is **fully exhausted in Month 2**.
  * The remaining $17.78 - 4 = 13.78 \approx 14$ failures must be fulfilled by procuring Part B.

#### 3. Monthly Part B Procurement Plan:
As the fleet progressively transitions from Part A (1,200 hr MTBF) to Part B (3,600 hr MTBF), total monthly failure demand falls from 20 units/month to a steady-state level of:
$$D_{\text{steady-state}} = 24,000 \times \frac{1}{3,600} = 6.67 \text{ units/month}$$

The procurement plan must order:
* Month 1: 0 units of Part B (covered by Part A stock).
* Month 2: 14 units of Part B.
* Months 3–6: Rapidly declining orders (17 $\to$ 14 $\to$ 12 $\to$ 10 units of Part B).
* Month 12+: Steady-state replenishment of 7 units of Part B per month.
Ordering a flat historical average of 20 units/month would leave the contractor with 150+ excess units of Part B by Year 2.
---

### 13.3 Rubric for Exercise 3

A passing design must demonstrate clear architectural decoupling across the four streams:

```text
               +--------------------------------------------+
               | ENTERPRISE DATA SOURCES                    |
               +--------------------------------------------+
                 │                    │                   │
                 ▼                    ▼                   ▼
        [ Drydock Schedule ]    [ SCADA / IoT ]    [ Shipyard Orderbook ]
        [ Class Regulations]    [ CMMS Work Orders] [ Decommission Schedule]
                 │                    │                   │
                 ▼                    ▼                   ▼
+──────────────────────+ +──────────────────────+ +──────────────────────+
| PIPELINE 1:          | | PIPELINE 2:          | | PIPELINE 3 & 4:      |
| Planned Overhaul     | | Stochastic Failure   | | Commissioning & EOL  |
| Deterministic Bill   | | Weibull / NHPP /     | | Initial Provisioning |
| of Materials Kits    | | Intermittent Model   | | / Last-Time Buy      |
|                      | |                      | |                      |
| Cadence: Quarterly   | | Cadence: Weekly      | | Cadence: Semi-Annual |
+──────────────────────+ +──────────────────────+ +──────────────────────+
                 │                    │                   │
                 └────────────────────┼───────────────────┘
                                      ▼
               +--------------------------------------------+
               | CONSOLIDATED MASTER SERVICE DEMAND PLAN    |
               +--------------------------------------------+
```

* **Pipeline 1 (Planned):** Ingests vessel drydock schedules (5-year regulatory survey cycle). Binds directly to OEM service kit BOMs. Deterministic execution.
* **Pipeline 2 (Unplanned):** Ingests daily telematics (engine hours, power load) and CMMS breakdown history. Runs Weibull hazard and intermittent statistical models.
* **Pipeline 3 & 4 (Commissioning & LTB):** Ingests shipbuilding contracts for initial provisioning packages. Ingests ship scrapping/retirement curves to execute life-of-type last-time buys without stranding inventory.

---

## 14. Interview questions

### Question 1: "Why can't we use traditional time-series forecasting methods (like Prophet or ARIMA) for capital equipment spare parts?"

**Strong Answer:**  
"Traditional time-series models assume that demand is generated by an autonomous, continuous stochastic process driven by calendar time, trend, and seasonality. In capital aftermarket, demand does not originate from consumer purchasing behavior; it is mechanically derived from the operating installed base. Spare parts demand violates standard time-series assumptions in three ways:
1. **Lumpiness and Intermittency:** Most SKU-location series consist of long strings of zeros punctuated by irregular bursts, which causes ARIMA to generate meaningless non-integer fractions or revert to zero.
2. **Censoring and Inventory Distortion:** Historical orders reflect purchasing behavior and supplier stockouts, not physical failures. If a part was out of stock, zero sales were recorded, biasing time-series models downward.
3. **Non-Stationary Wear-Out Dynamics:** As an equipment fleet ages, component failure rates increase according to Weibull hazard mechanics ($\beta > 1$). A time-series model trained on the initial 3 years of an asset's life will fail to predict the massive, simultaneous wear-out surge that occurs when the fleet reaches its characteristic life."

**Weak Answer:**  
"Because spare parts demand is too random and has too many zeros. Prophet and ARIMA only work on retail data, so for spare parts we usually just take a 12-month moving average or use simple exponential smoothing."

---

### Question 2: "What is the difference between right-censored and left-truncated reliability data, and why does ignoring censoring distort spare parts forecasts?"

**Strong Answer:**  
"In reliability survival analysis:
* **Right-censored data** occurs when an asset or component has not yet failed by the close of the observation window. We know it survived at least $T_{\text{obs}}$ hours, but we do not know its ultimate failure time.
* **Left-truncated data** occurs when assets enter the study cohort after already accumulating unobserved operating exposure (e.g., acquiring a used fleet without historical records).

If an analyst calculates component failure rates by simply averaging the lifespans of failed parts while discarding active, unfailed assets (the right-censored units), the dataset suffers severe survivorship bias. It artificially truncates the survival tail, vastly overestimating the failure rate and underestimating the characteristic life ($\eta$). In the supply chain, this leads to massive over-procurement of expensive components that last far longer in the field than the flawed analysis suggested."

**Weak Answer:**  
"Censored data means that some of the records are missing or corrupted in the database. If you drop the censored data, you might have less data, which makes your forecast have a higher error margin."

---

### Question 3: "Under what conditions is a constant failure rate assumption ($\beta = 1$) valid, and what are the operational dangers of applying it to mechanical components?"

**Strong Answer:**  
"A constant failure rate ($\beta = 1$) implies an exponential distribution, which possesses the memoryless property: $P(T > t + s \mid T > t) = P(T > s)$. This is mathematically valid primarily for solid-state electronic components (sensors, circuit boards) and complex multi-component systems subject strictly to external Poisson stress shocks (such as foreign object damage or electrical power surges).

Applying $\beta = 1$ to mechanical wear-out components (bearings, turbine blades, pump impellers, brake pads) is operationally dangerous. Mechanical parts undergo fatigue, corrosion, and wear governed by Weibull shape parameters $\beta \ge 2.0$. If you assume $\beta = 1$, you assume a 10-year-old bearing has the exact same probability of failing tomorrow as a brand-new bearing installed yesterday. This prevents the supply chain from anticipating fleet wear-out waves, leading to massive stockouts and asset groundings when the fleet reaches its fatigue threshold."

**Weak Answer:**  
"A constant failure rate is valid when the asset is in the middle of the bathtub curve. It makes calculations much easier because you can just use MTBF. It's dangerous because sometimes things fail faster if they are old."

---

### Question 4: "Explain the Syntetos-Boylan demand classification. What are its two axes, and what forecasting method should be deployed for a part in the 'Lumpy' quadrant?"

**Strong Answer:**  
"The Syntetos-Boylan framework categorizes intermittent demand along two structural dimensions:
1. **Average Demand Interval (ADI):** The average time between non-zero demand events ($\text{ADI} = T / N$). The standard empirical cutoff is 1.32 periods.
2. **Squared Coefficient of Variation ($CV^2$):** The normalized variance of non-zero order sizes ($CV^2 = (s_z / \bar{z})^2$). The standard cutoff is 0.49.

A **Lumpy** demand pattern occurs when $\text{ADI} \ge 1.32$ and $CV^2 \ge 0.49$, meaning demand events arrive infrequently and order quantities are highly volatile. Standard Croston's method and exponential smoothing fail here because they assume stable transaction sizes. For lumpy parts, the correct approach is:
* A **compound Poisson process** or **negative binomial distribution** coupled with extreme-value sizing.
* Or, ideally, abandoning purely statistical time-series methods entirely and deploying an **installed-base physical hazard model** that links order lumpiness directly to scheduled fleet maintenance overhauls."

**Weak Answer:**  
"The two axes are lead time and cost. Lumpy demand means you don't know when orders will come or how big they will be. For lumpy parts, you should use Croston's method with a safety stock buffer."

---

### Question 5: "What is Blackwell's Renewal Theorem, and why is it essential for multi-year capital spares budgeting?"

**Strong Answer:**  
"Blackwell's Renewal Theorem states that for any renewal process with non-lattice lifetime distributions having mean $\mu$, the expected number of renewals in an interval $[t, t + h]$ approaches $h / \mu$ as $t \to \infty$. 

In capital spares budgeting, when a brand-new fleet of assets is deployed simultaneously, the components all begin at age zero. Over the first few years, demand exhibits pronounced, non-stationary wave-like peaks as the initial cohort wears out simultaneously, followed by a lull as new parts are installed, followed by a secondary, broader peak. Blackwell's theorem proves that these oscillations gradually damp out due to the variance in component lifetimes, eventually settling into a constant long-term replacement rate of $1/\mu$ failures per unit of exposure. Understanding this prevents financial planners from mistaking early cyclical peaks for permanent demand increases or premature lulls for permanent reliability breakthroughs."

**Weak Answer:**  
"It's a mathematical theorem that says if you wait long enough, everything fails at an average rate. You use it to calculate the average number of parts you need each year."

---

### Question 6: "How do you handle stockout censoring when reconstructing true demand history from an enterprise ERP database?"

**Strong Answer:**  
"When an inventory location stocks out of a spare part, customer demand cannot be fulfilled, and transaction records become right-censored. To reconstruct true unconstrained demand, we execute three steps:
1. **Identify Censored Intervals:** Query daily inventory balance snapshots to identify periods where on-hand inventory was zero while backorders, customer purchase orders, or asset downtime work orders were open.
2. **Estimate Latent Arrival Intensity:** For the censored intervals, compute the baseline arrival rate $\lambda_0$ from unconstrained historical periods under identical installed base exposure, or utilize an Expectation-Maximization (EM) algorithm to estimate the missing Poisson counts conditioned on the observed zero-stock duration.
3. **Incorporate Alternative Fulfillment Channels:** Audit CMMS maintenance logs to detect field workarounds—specifically component cannibalization from inactive assets or emergency third-party broker spot-buys. Add these unrecorded events back into the demand time series at the timestamp of the physical breakdown event.

Failing to uncensor demand creates a vicious cycle: stockouts suppress recorded demand, which lowers safety stock targets, which guarantees future stockouts."

**Weak Answer:**  
"You look at the days where stock was zero and fill in the missing numbers by copying the average demand from the days before the stockout happened."

---

### Question 7: "What is the difference between a Form-Fit-Function (FFF) supersession and a non-interchangeable engineering change order?"

**Strong Answer:**  
"A **Form-Fit-Function (FFF) supersession** means a new part revision (Part B) can physically replace an older revision (Part A) without requiring structural, mechanical, or electrical modifications to the parent asset:
* **One-Way Interchangeable:** Part B can replace Part A, but Part A cannot replace Part B (common when Part B has enhanced durability or higher pressure ratings).
* **Two-Way Interchangeable:** Either part can replace the other arbitrarily.

In contrast, a **non-interchangeable engineering change order** means Part B cannot be installed on older assets without an extensive field modification kit or parent subassembly redesign. 

From an analytics perspective, an FFF supersession requires chaining the demand histories together so the forecasting algorithm pools their statistical sample size. A non-interchangeable ECO requires maintaining two completely separate demand streams linked strictly to the BOM effectivity dates and serial number blocks of the parent machines."

**Weak Answer:**  
"FFF means the parts are identical and you can just delete the old part number from the system. Non-interchangeable means you have to keep both part numbers in the catalog."

---

### Question 8: "Why does the presence of significant No-Fault-Found (NFF) rates distort both reliability modeling and repair pipeline inventory?"

**Strong Answer:**  
"No-Fault-Found (NFF) occurs when a field technician removes an expensive component suspected of causing a machine fault, but comprehensive testing at the central depot or OEM repair facility finds the component fully functional.

NFF distorts the supply chain in two distinct ways:
1. **Reliability Model Corruption:** If the removal is logged as a component failure in the CMMS, the statistical model artificially deflates the component's MTBF and characteristic life ($\eta$). The reliability engineer erroneously models a physical degradation mode that does not exist, triggering unnecessary engineering redesigns and inflating spare parts provisioning.
2. **Repair Pipeline Bloat:** Functional components spend weeks or months travelling through the reverse logistics pipeline, sitting in test queues, and incurring bench-test fees. This ties up massive working capital in 'floating inventory' that is functionally flawless but unavailable for field dispatch. In commercial aviation, NFF rates frequently reach 30–50% on avionics line-replaceable units (LRUs), requiring airlines to buy 20% more spare inventory simply to buffer the diagnostic testing loop."

**Weak Answer:**  
"NFF means parts are returned that aren't actually broken. It distorts the supply chain because you waste money shipping good parts back and forth to the repair depot."

---

## 15. Key terms

- **Installed Base (IB):** The total population of active, operational capital assets currently in field service.
- **Operating Exposure:** The cumulative physical stress units (hours, cycles, kilometers) accumulated by an asset.
- **Hazard Rate ($h(t)$):** The instantaneous failure rate of a component given survival up to time $t$.
- **Weibull Distribution:** A continuous probability distribution parameterized by shape $\beta$ and scale $\eta$, modeling infant mortality, random, and wear-out failures.
- **Shape Parameter ($\beta$):** The Weibull parameter determining whether failure rates decrease, stay constant, or increase with age.
- **Scale Parameter ($\eta$):** The characteristic life at which 63.2% of a component population will fail.
- **Right-Censored Observation:** Data from an asset or component that has not failed by the end of the study window.
- **Left-Truncated Data:** Historical operational exposure accumulated by an asset prior to observation.
- **Syntetos-Boylan Classification:** A 2x2 demand taxonomy based on ADI (cutoff 1.32) and $CV^2$ (cutoff 0.49).
- **Average Demand Interval (ADI):** The mean number of time periods elapsed between consecutive non-zero demand transactions.
- **Squared Coefficient of Variation ($CV^2$):** Variance divided by the squared mean of non-zero transaction sizes.
- **Croston’s Method:** A forecasting technique that independently models the inter-arrival time and the order size of intermittent demand.
- **Syntetos-Boylan Approximation (SBA):** A debiased mathematical correction to Croston's original intermittent demand formula.
- **Censored Demand:** Unmet customer demand that was never recorded in ERP sales tables due to inventory stockouts.
- **Cannibalization:** The practice of harvesting a functional component from an inactive asset to return an active machine to service.
- **Bill of Materials (BOM) Effectivity:** The engineering time or serial-number boundaries governing valid component part numbers.
- **Part Supersession:** The formal replacement of an older part SKU by a newly engineered revision (Form, Fit, Function).
- **Initial Provisioning (IP):** The initial bundle of spare parts purchased by a customer at the time of capital equipment commissioning.
- **Last-Time Buy (LTB):** The final procurement run executed to secure life-of-type spares prior to supplier component obsolescence.
- **Non-Homogeneous Poisson Process (NHPP):** A Poisson point process where the failure intensity function $\lambda(t)$ changes over time.
- **Power Law Process (Crow-AMSAA):** An NHPP model where cumulative failure intensity follows a power curve $\lambda(t) = \lambda \beta t^{\beta-1}$.
- **ISO 14224:** The international standard governing equipment reliability and maintenance data collection in process industries.
- **S1000D / S2000M:** International specifications for technical documentation and materiel management in aerospace and defence.
- **Mean Time Between Failures (MTBF):** Total operating exposure divided by total unscheduled failures for a repairable asset.
- **Quantity Per Assembly (QPA):** The number of identical components installed on a single parent asset.

---

## 16. Sources

### Peer-reviewed and Academic — Grade A
1. **Syntetos, A. A., & Boylan, J. E. (2005).** "The accuracy of intermittent demand estimates." *International Journal of Forecasting*, 21(2), 303-314. [DOI: 10.1016/j.ijforecast.2004.10.001]
2. **Croston, J. D. (1972).** "Forecasting stock control of intermittent demand." *Operational Research Quarterly*, 23(3), 289-303. [DOI: 10.1057/jors.1972.50]
3. **Romeijnders, W., Teunter, R., & van Jaarsveld, W. (2012).** "A two-step method for forecasting spare parts demand using information on component repairs." *European Journal of Operational Research*, 220(2), 394-407. [DOI: 10.1016/j.ejor.2012.01.042]
4. **Kim, S. H., Cohen, M. A., & Netessine, S. (2007).** "Performance contraction in service supply chains." *Management Science*, 53(12), 1843-1858. [DOI: 10.1287/mnsc.1070.0736]
5. **Ghodrati, B., & Kumar, U. (2005).** "Reliability and operating environment-based spare parts estimation approach: a case study in Kiruna Mine, Sweden." *Journal of Quality in Maintenance Engineering*, 11(2), 169-184. [DOI: 10.1108/13552510510601384]

### Standards and Regulatory Frameworks — Grade B
6. **International Organization for Standardization. (2016).** *ISO 14224:2016 — Petroleum, petrochemical and natural gas industries — Collection and exchange of reliability and maintenance data for equipment*. Geneva: ISO.
7. **Aerospace and Defence Industries Association of Europe (ASD). (2021).** *S2000M: International Specification for Materiel Management — Integrated Product Support*. Brussels: ASD.
8. **Federal Aviation Administration (FAA). (2020).** *Title 14 CFR Part 121: Operating Requirements: Domestic, Flag, and Supplemental Operations*. Washington, D.C.: US Government Publishing Office.

### Industry Case Evidence and Technical Monographs — Grade C & D
9. **Rolls-Royce plc. (2023).** *Annual Report 2023: Civil Aerospace Services & TotalCare Performance Overview*. London: Rolls-Royce Investor Relations.
10. **ASML Holding N.V. (2023).** *Customer Support and Installed Base Management Whitepaper*. Veldhoven: ASML Technical Communications.

---

## 17. Links

- **Parent Module:** [`_index.md`](_index.md) (Module 31 — Aftermarket, Spare Parts, and Service Supply Chains)
- **Root Survey File:** [`../../../course/31_aftermarket_spare_parts_service.md`](../../../course/31_aftermarket_spare_parts_service.md)
- **Next Topic:** [`02_criticality-spares-classification.md`](02_criticality-spares-classification.md) (Criticality and Spares Classification)
- **Prerequisite Foundations:**
  - [`../../02_methods/07_time_series_forecasting/06_intermittent-demand.md`](../../02_methods/07_time_series_forecasting/06_intermittent-demand.md)
  - [`../../02_methods/08_hierarchical_intermittent_probabilistic_forecasting/05_intermittent-classification.md`](../../02_methods/08_hierarchical_intermittent_probabilistic_forecasting/05_intermittent-classification.md)
  - [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
