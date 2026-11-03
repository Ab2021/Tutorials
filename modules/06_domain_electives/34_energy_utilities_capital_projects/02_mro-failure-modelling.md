> Module 34 — Energy, Utilities, and Capital Project Supply Chains
> Topic 02 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Module 10 (Inventory Optimization), Module 11 (Operations Research), Module 31 (Aftermarket & Spare Parts)
> Version 2026.5

# MRO Demand and Failure Modelling

---

## 1. Why this topic exists

In asset-intensive capital infrastructure—such as thermal and nuclear power stations, high-voltage transmission and distribution (T&D) grids, offshore hydrocarbon production platforms, natural gas pipeline compressor stations, and chemical process complexes—maintenance, repair, and operating (MRO) spare parts represent an acute operational paradox. Between 60% and 85% of distinct active stock-keeping units (SKUs) in a utility or refinery central warehouse experience zero demand in any given calendar quarter. Traditional supply chain systems categorize these parts as "slow-moving," "lumpy," or "intermittent." 

When standard enterprise resource planning (ERP) and supply chain planning (SCP) suites encounter intermittent demand, their default behavior is to apply classical Gaussian time-series algorithms: standard single exponential smoothing, moving averages, or Winters' models, paired with safety stock formulas based on normal distribution assumptions ($SS = z \cdot \sigma_L$). In an energy or utility operating environment, this practice produces two catastrophic failure modes:

1. **Systemic Over-Stocking of Non-Critical Slow-Movers:** For intermittent consumables (e.g., specialized gaskets, control valves, excitation fuses), standard models interpret long stretches of zero demand followed by a discrete withdrawal as high variance. The calculated standard deviation balloons, resulting in safety stock recommendations that lock up tens of millions of dollars in stagnant working capital for items whose actual failure probability is minuscule.
2. **Critical Stockouts of Insurance Spares During Catastrophic Plant Events:** For major capital rotable assemblies (e.g., boiler feed pump mechanical seals, gas turbine transition pieces, 345 kV step-up transformer bushings), historical consumption in the enterprise transaction log may be zero for five consecutive years. Gaussian models calculate a mean demand of zero and a standard deviation of zero, recommending an inventory level of zero. When an asset reaches the wear-out phase or suffers an unpredicted electrical surge, the lack of an on-hand replacement forces an extended unplanned outage. In power generation and petrochemical processing, the downtime cost of a critical asset routinely ranges from $100,000 to $1,500,000 per day in lost gross margin and expected unserved energy (EENS) regulatory penalties.

The root cause of this failure is an ontological error: **MRO spare parts demand is not an autonomous market-driven economic time series; it is the physical consequence of mechanical, thermal, electrical, and chemical asset degradation.** 

Transactional warehouse logs do not reflect true demand. Instead, they record a filtered, distorted artifact influenced by maintenance planning lead times, batch requisitioning, budget cycles, and stockout censoring. When a plant technician requires five mechanical seals but the warehouse only holds two, the ERP records an issue of two; the unsatisfied demand of three units is either cannibalized from an idle train, sourced via an emergency spot buy, or deferred, remaining invisible to historical demand algorithms.

Furthermore, capital equipment does not fail at a constant, memoryless rate over its multi-decade operational lifecycle. Power plant turbines, high-pressure steam headers, and transmission switchgear undergo severe operational stress, cyclic thermal fatigue, corrosion, and dielectric breakdown. Assuming a constant failure rate (Homogeneous Poisson Process) ignores the physics of asset aging. Conversely, relying solely on run-hours ignores real-time condition signals—such as vibration spectra, lubricant metal particle counts, thermographic hot spots, and dissolved gas concentrations in transformer insulating oil.

To solve this problem, MRO demand forecasting must integrate three technical disciplines:
- **Intermittent Statistical Point Forecasting:** Mathematical methods (Croston, Syntetos-Boylan Approximation [SBA], and Teunter-Syntetos-Babai [TSB]) that mathematically separate demand occurrence probability from demand transaction size, eliminating the severe positive bias inherent in legacy estimators and dynamically adjusting for obsolescence.
- **Non-Homogeneous Point Process Reliability Modelling:** Stochastic intensity formulations—specifically the Non-Homogeneous Poisson Process (NHPP) Power Law Process (Crow-AMSAA model)—that quantify wear-out aging ($\beta > 1$) and reliability improvement ($\beta < 1$) across aging capital infrastructure.
- **Condition-Based Degradation & Discrete Stochastic Inventory Sizing:** Integration of condition-based maintenance (CBM) signals via Proportional Hazards Models (PHM) and P-F intervals, coupled with exact discrete Poisson and Compound Poisson distribution safety stock formulas and Palm's theorem for rotable closed loops.

Implementing this unified architecture transforms MRO materials management from a reactive, inventory-bloated guessing game into a mathematically rigorous, asset-synchronized supply chain capability.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Classify** complex MRO spare part demand series into smooth, intermittent, slow-moving, and erratic regimes using the Syntetos-Boylan / Kosten categorization thresholds based on Average Demand Interval ($ADI$) and Squared Coefficient of Variation ($CV^2$).
2. **Formulate and compute** point demand forecasts using Croston's method, the Syntetos-Boylan Approximation (SBA), and the Teunter-Syntetos-Babai (TSB) method, explaining parameter update dynamics for obsolescence detection.
3. **Formulate** the Non-Homogeneous Poisson Process (NHPP) Power Law Process (Crow-AMSAA model) for repairable systems, estimating shape ($\beta$) and scale ($\lambda$) parameters to model asset aging versus reliability growth.
4. **Derive** the cumulative intensity function $\Lambda(t)$ and instantaneous failure intensity $\lambda(t)$ under minimal repair (As-Bad-As-Old, ABAO) versus perfect renewal (As-Good-As-New, AGAN) assumptions.
5. **Integrate** condition monitoring indicators (vibration velocity RMS, dissolved gas analysis DGA, oil debris particle counts) into Proportional Hazards Models (Cox PHM) to derive dynamic Remaining Useful Life (RUL) and spare part requisition triggers.
6. **Compute** discrete slow-moving safety stock levels and reorder points under Poisson and Compound Poisson distributed lead-time demand using exact cumulative distribution functions for a target Cycle Service Level ($CSL$) or Fill Rate ($FR$).
7. **Apply** Palm's Theorem and the $(S-1, S)$ continuous review base-stock policy to closed-loop repairable capital rotable spares, determining optimal depot spare holdings under turnaround time (TAT) constraints.
8. **Quantify** the financial trade-off between holding-cost-intensive slow-movers and unserved energy / forced plant outage penalty costs using marginal reliability economics.

---

## 3. The core mental model

The core mental model for MRO failure and demand forecasting connects the physical degradation of industrial machinery to stochastic materials replenishment. Rather than treating warehouse withdrawals as independent random variables, the framework separates demand into two complementary streams: a **Stochastic Intermittent Transactional Stream** for generic and semi-critical consumables, and an **Asset-Synchronized Degradation Stream** for critical capital spares and rotable equipment. These two streams converge in a **Stochastic Discrete Inventory Sizing Engine**.

```text
+---------------------------------------------------------------------------------------------------+
| PHYSICAL ASSET BASE & OPERATIONAL EXPOSURE (CCGT, Nuclear, T&D Grid, Hydrocarbon Pipeline)        |
|  - Operating Hours (t)    - Thermal Cycling / Starts (N)    - Environmental Severity / Grid Load  |
+---------------------------------------------------------------------------------------------------+
                                                  |
                         +------------------------+------------------------+
                         |                                                 |
                         v                                                 v
        [ STREAM 1: CONSUMABLE / INTERMITTENT ]           [ STREAM 2: CRITICAL CAPITAL ROTABLE ]
                         |                                                 |
                         v                                                 v
             Historical Warehouse Logs                        Physical Condition Monitoring
         (Discharges, Scraps, Requisitions)                  (Vibration, DGA Oil, Thermography)
                         |                                                 |
                         v                                                 v
             Demand Profile Classification                         P-F Interval Dynamics
           - Calculate ADI and CV^2                          - Detect Inception Point (P)
           - Map to Syntetos-Boylan Matrix                   - Track Degradation toward Failure (F)
                         |                                                 |
                         v                                                 v
             Intermittent Demand Engine                       Reliability Point Process Engine
           - Croston Method (Size vs Interval)               - NHPP Power Law Process: lambda(t), beta
           - SBA (De-biasing Correction)                     - Cox Proportional Hazards Model (PHM)
           - TSB (Obsolescence Tracking)                     - Remaining Useful Life (RUL) Forecast
                         |                                                 |
                         +------------------------+------------------------+
                                                  |
                                                  v
                  +---------------------------------------------------------------+
                  | DUAL-STREAM FUSION & LEAD-TIME DEMAND SYNTHESIS               |
                  |  - Nominal Lead Time L (Supplier / Repair Turnaround TAT)     |
                  |  - Discrete Demand Distribution: Poisson / Compound Poisson   |
                  |  - Dynamic CBM Override: P-F window < L triggers order        |
                  +---------------------------------------------------------------+
                                                  |
                                                  v
                  +---------------------------------------------------------------+
                  | DISCRETE INVENTORY OPTIMIZATION ENGINE                        |
                  |  - Exact Poisson Reorder Point: s = min {s | F(s) >= CSL}     |
                  |  - Palm's Theorem Rotable Base Stock: S = Poisson(lambda*TAT) |
                  |  - Cost Minimization: Holding Cost vs Forced Outage Risk Cost |
                  +---------------------------------------------------------------+
                                                  |
                                                  v
                  +---------------------------------------------------------------+
                  | ENTERPRISE WORK ORDER & PROCUREMENT REQUISITION               |
                  |  - Purchase Orders (Consumables, Vendor Lead Time L)          |
                  |  - Workshop Rotable Repair Induction (TAT Closed Loop)        |
                  |  - Outage Staging Kitting (Synchronized to Planned Shutdown)  |
                  +---------------------------------------------------------------+
```

### The Information and Causal Mechanics

1. **Separation of Rate and Scale:** In Stream 1, transaction histories are parsed into non-zero demand events $z_t$ and elapsed time intervals between events $p_t$. By updating these two components independently via exponential smoothing, the intermittent engine prevents the classic Gaussian distortion where zero periods artificially depress the estimated order quantity, or a large single order artificially inflates the perceived event frequency.
2. **Aging and Reliability Physics:** In Stream 2, the asset's operating age $t$ (measured in fired hours, duty cycles, or calendar time) feeds into a Non-Homogeneous Poisson Process. Unlike the memoryless exponential distribution where the hazard rate is constant, the Power Law process explicitly incorporates the aging parameter $\beta$. When $\beta > 1$, the instantaneous failure intensity $\lambda(t)$ accelerates over time, signaling that historical replenishment rates are insufficient to protect upcoming operational periods.
3. **CBM Lead-Time Window Override:** Condition monitoring does not merely refine a statistical distribution; it provides a deterministic countdown. The P-F interval defines the operational window between the earliest detectable symptom of potential functional failure (point $P$) and actual functional breakdown (point $F$). If the estimated Remaining Useful Life ($RUL$) falls below the replenishment lead time $L$ plus a safety buffer, the system overrides stochastic stock models and initiates an immediate deterministic procurement or rebuild work order.
4. **Discrete Tail Risk Sizing:** The outputs of both streams converge into an inventory policy engine that rejects Gaussian normality. Because lead-time demand $\mu_L$ is typically a small number (often $\mu_L < 1.0$), the engine uses discrete cumulative Poisson or Compound Poisson formulations. This captures the true skewness and zero-inflation of MRO spares, avoiding negative safety stocks and accurately pricing the tail risk of catastrophic power plant outages.

---

## 4. Definitions and vocabulary

| Term | Symbol / Code | Formal Definition | Common Operational Misuse | Standard Unit / Dimension |
| :--- | :--- | :--- | :--- | :--- |
| **MRO Spares** | MRO | Materials, spare parts, consumables, and tooling maintained to support asset availability and integrity. | Treating MRO parts with finished-goods retail forecasting algorithms. | SKU / Item ID |
| **Intermittent Demand** | — | A demand pattern characterized by frequent periods of zero demand interspersed with sporadic non-zero transactions. | Conflating zero demand in warehouse logs with zero component failures (ignoring stockouts). | Discrete count per period |
| **Average Demand Interval** | $ADI$ | The average number of time buckets between two successive non-zero demand transactions: $ADI = N / K$. | Calculating $ADI$ across calendar days rather than standardized planning periods. | Periods / transaction |
| **Coefficient of Variation** | $CV$ | Ratio of the standard deviation of non-zero transaction sizes to their mean: $CV = \sigma_z / \mu_z$. | Calculating $CV$ across the entire time series including zeros, which inflates variance. | Dimensionless ratio |
| **Croston's Method** | — | Intermittent forecasting algorithm that separately updates non-zero demand size $z_t$ and inter-arrival interval $p_t$. | Believing Croston provides an unbiased forecast; it exhibits known positive bias. | Units / period |
| **Syntetos-Boylan Approx.** | SBA | An analytical correction to Croston's method that multiplies the point forecast by $(1 - \alpha/2)$ to eliminate positive bias. | Applying SBA when demand is continuous and non-intermittent ($ADI < 1.32$). | Units / period |
| **Teunter-Syntetos-Babai** | TSB | Intermittent forecasting algorithm that updates demand probability $p_t$ in every period, enabling rapid obsolescence decay. | Setting the probability smoothing parameter $\beta_{tsb}$ identical to size smoothing $\alpha$. | Units / period |
| **Homogeneous Poisson** | HPP | Point process where failure events arrive at a constant instantaneous rate $\lambda$, with exponential inter-arrival times. | Applying HPP to aging capital assets undergoing mechanical wear-out or fatigue. | Failures / unit time ($\text{hours}^{-1}$) |
| **Non-Homogeneous Poisson** | NHPP | Point process where the instantaneous arrival rate $\lambda(t)$ varies with time or accumulated operational exposure. | Assuming NHPP resets to zero failure intensity following a minor component repair (violating ABAO). | Failures / unit time |
| **Crow-AMSAA Power Law** | — | Parametric NHPP model where cumulative failure intensity is $\Lambda(t) = \lambda t^\beta$ and hazard is $\lambda(t) = \lambda \beta t^{\beta-1}$. | Confusing Crow-AMSAA with a two-parameter renewal Weibull model. | Dimensionless $\beta$, scale $\lambda$ |
| **Shape Parameter** | $\beta$ | The elasticity of failure intensity with respect to operating time in a Power Law process. | Interpreting $\beta > 1$ as infant mortality (it indicates wear-out aging). | Dimensionless |
| **Minimal Repair** | ABAO | Maintenance action that restores an asset to operating condition without altering its failure rate ("As-Bad-As-Old"). | Assuming a component replacement resets the entire complex multi-component machine to new. | System state condition |
| **Perfect Renewal** | AGAN | Maintenance action that restores an asset to pristine, like-new condition ("As-Good-As-New"), resetting the point process. | Assuming minor field servicing qualifies as perfect renewal. | System state condition |
| **P-F Interval** | — | The operational duration between the detection of a potential failure ($P$) and functional failure ($F$). | Treating the P-F interval as a constant rather than an operational stress-dependent distribution. | Operating hours or days |
| **Proportional Hazards** | PHM | Cox regression model where baseline hazard $h_0(t)$ is scaled exponentially by diagnostic condition covariates $\mathbf{Z}(t)$. | Using static one-time oil sample readings rather than time-series covariate trajectories. | Hazard multiplier |
| **Remaining Useful Life** | RUL | Expected operational time remaining before an operating asset reaches its functional failure threshold. | Treating RUL as a deterministic deadline rather than a stochastic distribution. | Operating hours |
| **Dissolved Gas Analysis** | DGA | Diagnostic technique measuring combustible gas concentrations (ppm) dissolved in transformer mineral oil. | Treating total dissolved combustible gas (TDCG) without decomposing specific hydrocarbon ratios. | Parts per million (ppm) |
| **Cycle Service Level** | $CSL$ ($\alpha$) | The probability that lead-time demand does not exceed the reorder point: $P(D_L \le s)$. | Confusing $CSL$ with fill rate ($FR$), leading to over-investment in high-volume spares. | Percentage (%) |
| **Fill Rate** | $FR$ ($\beta_s$) | The fraction of total demand volume met directly from shelf stock without backordering. | Assuming 95% CSL implies 95% Fill Rate (Fill Rate is almost always higher). | Percentage (%) |
| **Palm's Theorem** | — | Theorem stating that the steady-state distribution of items in an $M/G/\infty$ repair loop is Poisson with mean $\theta = \lambda \cdot TAT$. | Applying Palm's theorem when repair shop capacity is severely constrained (queuing bottlenecks). | Mean repair loop units |
| **Turnaround Time** | $TAT$ | Total elapsed time required to remove, transport, inspect, remanufacture, test, and restock a repairable rotable. | Using nominal bench-repair time while omitting logistics, customs, and inspection delays. | Days or Weeks |
| **Expected Unserved Energy** | EENS | Probabilistic metric quantifying the volume of electrical energy expected to be unsupplied due to asset outages. | Omitting EENS penalties from spare part inventory business case justifications. | MWh or Currency ($) |

---

## 5. The mechanics, step by step

### Step 1: Demand Classification and Data Conditioning

Before applying any mathematical forecasting algorithm, you must categorize historical demand transactions using the Syntetos-Boylan classification framework. The framework relies on two metrics:
1. **Average Demand Interval ($ADI$):** The average inter-arrival time between non-zero demand occurrences, computed across $N$ total discrete time buckets (typically months or 4-week periods):
   $$ADI = \frac{N}{\sum_{t=1}^N \mathbb{I}(y_t > 0)}$$
   where $\mathbb{I}(\cdot)$ is the indicator function and $y_t$ is the demand observed in period $t$.
2. **Squared Coefficient of Variation of Non-Zero Demand ($CV^2$):** The relative variance of the demand transaction sizes:
   $$CV^2 = \left(\frac{\sigma_z}{\mu_z}\right)^2 = \frac{\frac{1}{K-1} \sum_{k=1}^K (z_k - \mu_z)^2}{\left(\frac{1}{K} \sum_{k=1}^K z_k\right)^2}$$
   where $z_k$ represents the $k$-th non-zero demand size, and $K$ is the total count of non-zero transactions.

Based on empirical validation across industrial inventory datasets (Syntetos, Boylan, and Croston, 2005), demand is partitioned into four quadrants using the critical cut-off thresholds $ADI = 1.32$ and $CV^2 = 0.49$:

```text
       CV^2 ^
            |
            |      ERRATIC       |       LUMPY
       0.49 +--------------------+--------------------
            |       SMOOTH       |    INTERMITTENT
            |                    |
            +--------------------+--------------------> ADI
            0                   1.32
```

- **Smooth Demand ($ADI < 1.32, CV^2 < 0.49$):** Regular occurrence and low size variability. Standard Simple Exponential Smoothing (SES) or Holt-Winters models are mathematically appropriate.
- **Intermittent Demand ($ADI \ge 1.32, CV^2 < 0.49$):** Sporadic occurrence with relatively consistent transaction sizes. Syntetos-Boylan Approximation (SBA) dominates.
- **Erratic Demand ($ADI < 1.32, CV^2 \ge 0.49$):** Frequent occurrence with extreme transaction size variability. Croston or SBA with compound distribution modeling.
- **Lumpy Demand ($ADI \ge 1.32, CV^2 \ge 0.49$):** Highly intermittent and highly variable in size. This is the classic profile of high-cost utility spares. Requires TSB or compound Poisson/negative binomial modeling.

#### Data Sanitization and Uncensoring
Prior to classification, filter the transaction history to eliminate artificial distortions:
- **Censored Stockout Periods:** If a part was at zero stock for 4 months, do not record zero demand. Impute demand using the historical arrival rate or operational run-hours to avoid depressing future forecast levels.
- **Requisition Batching:** If a maintenance campaign created 1 requisition for 12 valves across 12 distinct pipeline blocks, disaggregate the transaction into its physical failure components.
- **Outage Withdrawals:** Separate planned turnaround withdrawals (which are deterministic project demand) from unplanned breakdown work orders (stochastic failure demand).

---

### Step 2: Intermittent Demand Forecasting Formulation

#### The Croston Method
Croston (1972) decoupled the demand time series into two distinct latent variables:
- $z_t$: The estimated non-zero demand size.
- $p_t$: The estimated inter-arrival interval (time elapsed between positive demands).

Let $q$ be the number of periods elapsed since the last observed positive demand. In every period $t$:
$$\text{If } y_t = 0: \quad z_t = z_{t-1}, \quad p_t = p_{t-1}, \quad q = q + 1$$
$$\text{If } y_t > 0: \quad z_t = \alpha y_t + (1 - \alpha) z_{t-1}$$
$$p_t = \alpha q + (1 - \alpha) p_{t-1}$$
$$q = 1$$
where $\alpha \in (0, 1)$ is the exponential smoothing parameter (typically $0.05 \le \alpha \le 0.20$ for industrial MRO).

The point forecast for demand in period $t+1$ is the expected size divided by the expected inter-arrival interval:
$$\hat{y}_{t+1}^{\text{Croston}} = \frac{z_t}{p_t}$$

#### The Syntetos-Boylan Approximation (SBA)
Syntetos and Boylan (2001, 2005) demonstrated mathematically that Croston's estimator is fundamentally biased. Because $z_t$ and $p_t$ are treated as independent random variables, the expected value of their ratio does not equal the ratio of their expected values:
$$\mathbb{E}\left[\frac{z_t}{p_t}\right] = \mathbb{E}[z_t] \cdot \mathbb{E}\left[\frac{1}{p_t}\right]$$
By Jensen's inequality, since $g(x) = 1/x$ is strictly convex for $x > 0$, $\mathbb{E}[1/p_t] > 1/\mathbb{E}[p_t]$. Consequently, Croston's method systematically over-forecasts demand by a factor approximately equal to $\frac{\alpha}{2}$.

The Syntetos-Boylan Approximation corrects for this mathematical bias:
$$\hat{y}_{t+1}^{\text{SBA}} = \left(1 - \frac{\alpha}{2}\right) \frac{z_t}{p_t}$$
In utility inventory environments with large populations of intermittent parts, applying the SBA de-biasing multiplier $(1 - \alpha/2)$ reduces total inventory holding costs across the portfolio by 8% to 14% without degrading fill rates.

#### The Teunter-Syntetos-Babai (TSB) Method
Croston and SBA suffer from a fatal operational flaw when applied to aging assets or equipment nearing decommissioning: **the obsolescence trap**. Because $p_t$ is updated *only when a positive demand occurs*, if an asset is retired and never fails again, $y_t = 0$ indefinitely. Croston and SBA will keep $z_t$ and $p_t$ frozen at their last updated values forever, forecasting perpetual positive demand for a dead asset.

Teunter, Syntetos, and Babai (2011) resolved this by updating the **probability of demand occurrence** ($p_t^{\text{TSB}}$) in *every single period*, regardless of whether demand occurred:
$$\text{Demand Probability:} \quad p_t^{\text{TSB}} = \beta_{\text{tsb}} \mathbb{I}(y_t > 0) + (1 - \beta_{\text{tsb}}) p_{t-1}^{\text{TSB}}$$
$$\text{Demand Size (if } y_t > 0): \quad z_t = \alpha y_t + (1 - \alpha) z_{t-1}$$
$$\text{Demand Size (if } y_t = 0): \quad z_t = z_{t-1}$$
where $\beta_{\text{tsb}} \in (0, 1)$ is the probability smoothing parameter and $\alpha \in (0, 1)$ is the size smoothing parameter.

The TSB forecast for period $t+1$ is:
$$\hat{y}_{t+1}^{\text{TSB}} = p_t^{\text{TSB}} \cdot z_t$$

If a part experiences no demand for consecutive periods, $\mathbb{I}(y_t > 0) = 0$, causing $p_t^{\text{TSB}}$ to decay geometrically at rate $(1 - \beta_{\text{tsb}})^k$. This allows automated detection of part obsolescence and systematic phase-out of safety stock.

---

### Step 3: Reliability and Failure Point Process Modeling (NHPP Power Law)

For critical equipment, demand is dictated by the physical failure rate of repairable systems. A repairable system is one that, after failing, can be restored to operational status by replacing a sub-assembly without replacing the entire asset.

#### Homogeneous vs. Non-Homogeneous Poisson Processes
- **Homogeneous Poisson Process (HPP):** Assumes constant failure intensity $\lambda(t) = \lambda_0$. Inter-arrival times between failures are independent and identically distributed (i.i.d.) exponential random variables. Valid only during the flat "useful life" section of the bathtub curve.
- **Non-Homogeneous Poisson Process (NHPP):** Allows failure intensity $\lambda(t)$ to vary with operational exposure $t$. Appropriate for assets subject to cyclic thermal fatigue, mechanical wear, corrosion, and aging infrastructure.

#### The Crow-AMSAA Power Law Process Formulation
Under the Power Law Process (Ascher and Feingold, 1984; Crow, 1974), the cumulative failure intensity function $\Lambda(t)$ is defined as:
$$\Lambda(t) = \mathbb{E}[N(t)] = \lambda t^\beta \quad (\lambda > 0, \; \beta > 0)$$
where $t$ represents cumulative operational exposure (fired hours, megawatt-hours, or starts).

The instantaneous failure intensity $\lambda(t)$ is the first derivative of $\Lambda(t)$:
$$\lambda(t) = \frac{d\Lambda(t)}{dt} = \lambda \beta t^{\beta - 1}$$

The shape parameter $\beta$ provides unambiguous engineering diagnostic insight:
- **$\beta < 1$ (Reliability Growth / Infant Mortality):** The instantaneous failure rate decreases over time. Common during the commissioning and post-turnaround shake-down phase of a newly built power block.
- **$\beta = 1$ (Stationary HPP):** The failure rate is strictly constant. Failures are purely random, driven by external operational transients rather than physical degradation.
- **$\beta > 1$ (Wear-Out / Aging):** The failure rate accelerates with operating time. This is the operational reality of utility assets operating beyond their initial design life.

```text
Hazard
Intensity
  ^
  |        \
  |         \   beta < 1 (Infant Mortality)
  |          \
  |---------------------------- beta = 1 (Useful Life / Random)
  |                  /
  |                 /   beta > 1 (Wear-Out / Aging Infrastructure)
  |                /
  +------------------------------------------------------------>
  0                                             Operating Time (t)
```

#### Maximum Likelihood Estimation (MLE) of $\beta$ and $\lambda$
Consider a capital asset observed over cumulative operating exposure up to time $T$. During this period, $n$ failure events occur at operational timestamps $0 < t_1 < t_2 < \dots < t_n \le T$.

Under a **time-truncated (Type I censoring)** observation window where the observation period ends at a fixed exposure time $T$:
$$\hat{\beta} = \frac{n}{n \ln(T) - \sum_{i=1}^n \ln(t_i)} = \frac{n}{\sum_{i=1}^n \ln(T / t_i)}$$
$$\hat{\lambda} = \frac{n}{T^{\hat{\beta}}}$$

Under a **failure-truncated (Type II censoring)** observation window where monitoring ceases immediately upon the occurrence of the $n$-th failure at timestamp $t_n$:
$$\hat{\beta} = \frac{n}{\sum_{i=1}^{n-1} \ln(t_n / t_i)}$$
$$\hat{\lambda} = \frac{n}{t_n^{\hat{\beta}}}$$

#### Expected Spares Demand over Lead-Time Horizons
To determine the expected spares required during an upcoming procurement or turnaround lead time window $[T_{\text{start}}, T_{\text{end}}]$, evaluate the cumulative intensity function over the future horizon:
$$\mathbb{E}[N(T_{\text{start}}, T_{\text{end}})] = \Lambda(T_{\text{end}}) - \Lambda(T_{\text{start}}) = \hat{\lambda} \left(T_{\text{end}}^{\hat{\beta}} - T_{\text{start}}^{\hat{\beta}}\right)$$

If an asset has accumulated $T = 30,000$ operating hours and the procurement lead time for a replacement impeller assembly is equivalent to $4,000$ operating hours, the expected failure demand during that procurement lead time is:
$$\mathbb{E}[D_L] = \hat{\lambda} \left[(34,000)^{\hat{\beta}} - (30,000)^{\hat{\beta}}\right]$$
When $\hat{\beta} > 1$, this expected demand is substantially larger than historical averages would indicate, directly protecting the asset against wear-out induced stockouts.

---

### Step 4: Condition-Based Degradation & Physical Signal Integration

Statistical models forecast baseline failure tendencies across a population. However, when an individual operating machine begins to physically fail, online and offline Condition-Based Maintenance (CBM) sensors provide real-time evidence of mechanical degradation.

#### The P-F Interval Architecture
The P-F curve models the progression from pristine operation to functional breakdown:
- **Point P (Potential Failure):** The earliest timestamp at which an impending failure can be detected using specialized diagnostic instrumentation (e.g., high-frequency acoustic emission, oil spectrographic analysis, DGA, ultrasonic inspection, vibration acceleration enveloping).
- **Point F (Functional Failure):** The timestamp at which the machine can no longer perform its intended design function safely or within operational tolerances.
- **P-F Interval ($\Delta t_{PF}$):** The available lead time between detection ($P$) and total loss of function ($F$).

```text
Asset
Health ^
       |  Point P (Potential Failure: Earliest physical symptom detected)
 100%  +---------\
       |           \
       |             \   P-F Interval: Available Procurement / Kitting Window
       |               \
       |                 \
       |                   \  Point F (Functional Failure: Trip / Forced Outage)
   0%  +--------------------+--------------------------------------------->
       0                    t_P                                          t_F  Time
```

#### Proportional Hazards Modeling (Cox PHM)
To link continuous condition monitoring telemetry directly to failure hazard rates, formulate a Proportional Hazards Model (Jardine, Lin, and Banjevic, 2006):
$$h(t, \mathbf{Z}(t)) = h_0(t) \cdot \exp\left(\sum_{j=1}^m \gamma_j Z_j(t)\right) = h_0(t) \cdot \exp\left(\boldsymbol{\gamma}^T \mathbf{Z}(t)\right)$$
where:
- $h_0(t)$ is the baseline hazard function, typically modeled via a two-parameter Weibull rate: $h_0(t) = \frac{\beta}{\eta} \left(\frac{t}{\eta}\right)^{\beta - 1}$.
- $\mathbf{Z}(t) = [Z_1(t), Z_2(t), \dots, Z_m(t)]^T$ is a vector of dynamic, time-dependent diagnostic covariates.
- $\boldsymbol{\gamma} = [\gamma_1, \gamma_2, \dots, \gamma_m]^T$ is a vector of regression coefficients estimated via partial likelihood maximization.

In a high-voltage power transformer, diagnostic covariates include:
- $Z_1(t)$: Acetylene ($C_2H_2$) concentration in ppm from Dissolved Gas Analysis (indicates high-energy electrical arcing).
- $Z_2(t)$: Hydrogen ($H_2$) concentration in ppm (indicates corona discharge or partial discharge).
- $Z_3(t)$: Dielectric breakdown voltage in kV.
- $Z_4(t)$: Moisture content in oil in ppm.

#### Dynamic Remaining Useful Life (RUL) Calculation
Given the current diagnostic covariate vector $\mathbf{Z}(t)$ at time $t$, the conditional survival probability $S(t + \Delta t \mid t, \mathbf{Z}(t))$ for an additional operating horizon $\Delta t$ is:
$$S(t + \Delta t \mid t, \mathbf{Z}(t)) = \exp\left( - \int_t^{t + \Delta t} h_0(u) \exp(\boldsymbol{\gamma}^T \mathbf{Z}(u)) \, du \right)$$
The stochastic Remaining Useful Life is the expected duration until survival probability drops below a critical threshold $\delta_{\text{crit}}$ (e.g., $\delta_{\text{crit}} = 0.05$):
$$\text{RUL}(t) = \inf \left\{ \Delta t \ge 0 : S(t + \Delta t \mid t, \mathbf{Z}(t)) \le \delta_{\text{crit}} \right\}$$

#### Supply Chain Replenishment Trigger Logic
The inventory ordering decision rule bridges maintenance engineering and procurement:
$$\text{If } \text{RUL}(t) \le L_{\text{supplier}} + T_{\text{staging}} + SS_{\text{time}}:$$
$$\text{Trigger Immediate Deterministic Purchase / Repair Order for Replacement Assembly}$$
$$\text{Else:}$$
$$\text{Maintain Standard Stochastic Poisson Inventory Buffer}$$
where $L_{\text{supplier}}$ is vendor procurement lead time, $T_{\text{staging}}$ is pre-outage staging time, and $SS_{\text{time}}$ is a temporal safety margin. This logic converts an unexpected breakdown into a planned, just-in-time replenishment.

---

### Step 5: Slow-Moving Inventory Optimization and Safety Stock Sizing

#### Why Gaussian Normality Fails for Intermittent MRO
The standard textbook safety stock formula is:
$$SS = z_{\alpha} \cdot \sqrt{L \cdot \sigma_D^2 + \mu_D^2 \cdot \sigma_L^2}$$
where $z_{\alpha}$ is the standard normal inverse cumulative distribution function at service level $\alpha$.

When applied to slow-moving spares where average demand during replenishment lead time ($\mu_L = \hat{y} \cdot L$) is low (e.g., $\mu_L = 0.25$ units):
1. **Symmetry Assumption:** The normal distribution is symmetric around the mean. For $\mu_L = 0.25$, a normal model assigns positive probability mass to negative demand values (e.g., $P(D_L < 0) \approx 30.8\%$), which is physically impossible.
2. **Tail Probability Distortion:** Discrete failure events occur in non-negative integer steps ($0, 1, 2, \dots$). Continuous Gaussian approximations severely underestimate the probability of experiencing zero demand, while understating the relative likelihood of experiencing 1 or 2 discrete failure events.

#### Exact Discrete Poisson Lead-Time Demand Sizing
When individual failure events occur independently, lead-time demand $D_L$ follows a discrete Poisson distribution with parameter $\mu_L = \hat{y} \cdot L$:
$$P(D_L = k) = \frac{(\mu_L)^k e^{-\mu_L}}{k!}, \quad k \in \{0, 1, 2, \dots\}$$
The cumulative probability distribution $F(s)$ represents the probability that total lead-time demand does not exceed reorder point $s$:
$$F(s) = P(D_L \le s) = \sum_{k=0}^s \frac{(\mu_L)^k e^{-\mu_L}}{k!}$$

To achieve a target **Cycle Service Level** ($CSL = \alpha_{\text{target}}$):
$$s^* = \min \left\{ s \in \mathbb{N}_0 : F(s) \ge \alpha_{\text{target}} \right\}$$
The discrete safety stock is then:
$$SS = s^* - \mu_L$$

#### Compound Poisson Lead-Time Demand (Batch Sizing)
When failure events arrive according to a Poisson process with rate $\lambda_{\text{arr}}$, but each failure event requires a random transaction batch size $W$ (e.g., replacing all 4 or 8 valve gaskets simultaneously):
$$D_L = \sum_{i=1}^{N_L} W_i$$
where $N_L \sim \text{Poisson}(\lambda_{\text{arr}} \cdot L)$ and $W_i$ are i.i.d. positive discrete random variables with probability mass function $g(w) = P(W = w)$.

The compound distribution is evaluated using the **Panjer Recursion** (Panjer, 1981):
$$P(D_L = 0) = \exp\left(-\lambda_{\text{arr}} L (1 - g(0))\right)$$
$$P(D_L = x) = \frac{\lambda_{\text{arr}} L}{x} \sum_{j=1}^x j \cdot g(j) \cdot P(D_L = x - j), \quad x \ge 1$$
The reorder point $s^*$ is calculated by summing the Panjer mass probabilities until the cumulative target service level is satisfied.

#### Rotable Spares Optimization (Palm's Theorem)
For expensive, repairable assemblies (e.g., gas turbine nozzles, boiler feed pumps, circuit breaker operating mechanisms), assets are not scrapped upon failure; they are removed, replaced with a spare from depot inventory, and sent to an internal workshop or OEM service facility for refurbishment.

This forms a closed-loop repairable system governed by **Palm's Theorem** ($M/G/\infty$ queueing model):
> If demand arrivals follow a Poisson process with rate $\lambda$, and repair turnaround times ($TAT$) are independent random variables with arbitrary distribution and mean $\tau = \mathbb{E}[TAT]$, then the steady-state number of items undergoing repair in the closed loop follows a Poisson distribution with mean:
> $$\theta = \lambda \cdot \mathbb{E}[TAT]$$

Under a continuous-review base-stock policy $(S-1, S)$, whenever an item fails, a replacement is immediately issued from stock, and the failed core enters the repair loop. The stock level on hand plus on order equals $S$.

The probability of having $k$ backorders (assets waiting for spares) is given by the Poisson tail:
$$P(\text{Backorders} > 0) = P(\text{Items in Repair} \ge S) = 1 - \sum_{j=0}^{S-1} \frac{\theta^j e^{-\theta}}{j!}$$

The **Expected Backorders ($EBO$)** for a base-stock level $S$ is:
$$EBO(S) = \sum_{j=S}^\infty (j - S) \frac{\theta^j e^{-\theta}}{j!} = \theta \left(1 - F(S-1; \theta)\right) - S \left(1 - F(S; \theta)\right)$$

To determine the optimal base-stock level $S^*$, solve the marginal cost-benefit trade-off:
$$\Delta EBO(S) = EBO(S-1) - EBO(S) = 1 - F(S-1; \theta)$$
$$S^* = \min \left\{ S \in \mathbb{N}_0 : C_{\text{holding}} \ge C_{\text{downtime}} \cdot \Delta EBO(S) \right\}$$
where $C_{\text{holding}}$ is the annual carrying cost of the spare, and $C_{\text{downtime}}$ is the unserved energy or asset downtime penalty incurred per unit of backorder time.

---

## 6. Worked example

### The Asset and Context
Consider a 650 MW Combined Cycle Gas Turbine (CCGT) power generating facility operating in a deregulated wholesale electricity market. The plant utilizes four high-pressure boiler feed water pumps (HP BFPs): two active base-load pumps and two hot standby pumps. 

Each pump utilizes a specialized silicon-carbide mechanical seal cartridge assembly. If an active pump seal suffers catastrophic blowout and the standby pump cannot start due to a missing seal, the entire power block must de-rate by 325 MW. The market penalty for forced outage capacity deficit is \$65,000 per hour.

We will execute an end-to-end failure modeling, intermittent forecasting, and inventory optimization calculation for this critical mechanical seal inventory.

---

### Step 1: Historical Failure Process Calibration (Crow-AMSAA)
The fleet of four pumps has accumulated a total of $T = 36,000$ operating hours over the past five calendar years. Maintenance logs indicate that mechanical seal failures occurred at the following cumulative fleet run-hour timestamps:
- $t_1 = 4,200$ hours
- $t_2 = 11,800$ hours
- $t_3 = 18,600$ hours
- $t_4 = 24,900$ hours
- $t_5 = 30,100$ hours
- $t_6 = 34,500$ hours

Total observed failures: $n = 6$. The observation period ended at $T = 36,000$ hours (time-truncated Type I censoring).

#### Parameter Estimation:
Compute $\ln(T / t_i)$ for each failure event:
- $\ln(36,000 / 4,200) = \ln(8.5714) = 2.1484$
- $\ln(36,000 / 11,800) = \ln(3.0508) = 1.1154$
- $\ln(36,000 / 18,600) = \ln(1.9355) = 0.6604$
- $\ln(36,000 / 24,900) = \ln(1.4458) = 0.3686$
- $\ln(36,000 / 30,100) = \ln(1.1960) = 0.1790$
- $\ln(36,000 / 34,500) = \ln(1.0435) = 0.0426$

Sum of logarithmic ratios:
$$\sum_{i=1}^6 \ln\left(\frac{T}{t_i}\right) = 2.1484 + 1.1154 + 0.6604 + 0.3686 + 0.1790 + 0.0426 = 4.5144$$

Calculate the Crow-AMSAA shape parameter $\hat{\beta}$:
$$\hat{\beta} = \frac{n}{\sum_{i=1}^n \ln(T / t_i)} = \frac{6}{4.5144} = 1.3291$$

Engineering Interpretation: $\hat{\beta} = 1.3291 > 1.0$. The pump seals are exhibiting pronounced wear-out aging due to thermal cyclic fatigue and erosive particle wear.

Calculate the scale parameter $\hat{\lambda}$:
$$\hat{\lambda} = \frac{n}{T^{\hat{\beta}}} = \frac{6}{(36,000)^{1.3291}} = \frac{6}{1,136,765} = 5.2781 \times 10^{-6}$$

#### Lead-Time Demand Forecast:
The procurement lead time for a replacement OEM seal cartridge is 6 months, during which the pump fleet is scheduled to operate an additional $4,000$ cumulative hours (taking total exposure from $36,000$ to $40,000$ hours).

Expected failures over the replenishment lead-time window:
$$\Lambda(36,000) = 6.0000$$
$$\Lambda(40,000) = \hat{\lambda} \cdot (40,000)^{\hat{\beta}} = (5.2781 \times 10^{-6}) \cdot (40,000)^{1.3291}$$
$$(40,000)^{1.3291} = \exp(1.3291 \times \ln(40,000)) = \exp(1.3291 \times 10.5966) = \exp(14.0839) = 1,307,632$$
$$\Lambda(40,000) = (5.2781 \times 10^{-6}) \times 1,307,632 = 6.9019$$

$$\mathbb{E}[N(36,000, 40,000)] = \Lambda(40,000) - \Lambda(36,000) = 6.9019 - 6.0000 = 0.9019 \text{ failures}$$

---

### Step 2: Warehouse Transaction Intermittent Forecasting
In the central stores warehouse, consumable seal elastomer rebuild kits for these pumps have recorded the following monthly transaction history over the last 12 months:
$$y = [0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2]$$

#### Classification:
- Total periods: $N = 12$ months
- Non-zero periods: Month 3 ($y_3 = 2$), Month 7 ($y_7 = 1$), Month 12 ($y_{12} = 2$). Total non-zero count: $K = 3$.
- $ADI = N / K = 12 / 3 = 4.00$ months.
- Non-zero transaction sizes: $z = [2, 1, 2]$.
  $$\mu_z = \frac{2 + 1 + 2}{3} = \frac{5}{3} \approx 1.6667$$
  $$\sigma_z^2 = \frac{(2 - 1.6667)^2 + (1 - 1.6667)^2 + (2 - 1.6667)^2}{3 - 1} = \frac{0.1111 + 0.4444 + 0.1111}{2} = 0.3333$$
  $$CV^2 = \frac{\sigma_z^2}{\mu_z^2} = \frac{0.3333}{(1.6667)^2} = \frac{0.3333}{2.7778} = 0.1200$$

Since $ADI = 4.00 \ge 1.32$ and $CV^2 = 0.1200 < 0.49$, this part falls strictly into the **Intermittent Demand** category. The Syntetos-Boylan Approximation is the required statistical model.

#### Forecast Computation:
Set smoothing parameter $\alpha = 0.15$.
Initial conditions at Month 0: $z_0 = 1.6667, p_0 = 4.00$.
- **Months 1–2:** $y = 0$. $z$ and $p$ remain unchanged. $q$ increments to 3.
- **Month 3:** $y_3 = 2, q = 3$.
  $$z_3 = 0.15(2) + 0.85(1.6667) = 0.30 + 1.4167 = 1.7167$$
  $$p_3 = 0.15(3) + 0.85(4.00) = 0.45 + 3.40 = 3.8500$$
  $$q = 1$$
- **Months 4–6:** $y = 0$. $z_6 = 1.7167, p_6 = 3.8500, q$ increments to 4.
- **Month 7:** $y_7 = 1, q = 4$.
  $$z_7 = 0.15(1) + 0.85(1.7167) = 0.15 + 1.4592 = 1.6092$$
  $$p_7 = 0.15(4) + 0.85(3.8500) = 0.60 + 3.2725 = 3.8725$$
  $$q = 1$$
- **Months 8–11:** $y = 0$. $z_{11} = 1.6092, p_{11} = 3.8725, q$ increments to 5.
- **Month 12:** $y_{12} = 2, q = 5$.
  $$z_{12} = 0.15(2) + 0.85(1.6092) = 0.30 + 1.3678 = 1.6678$$
  $$p_{12} = 0.15(5) + 0.85(3.8725) = 0.75 + 3.2916 = 4.0416$$

Point Forecasts for Month 13:
$$\hat{y}_{13}^{\text{Croston}} = \frac{z_{12}}{p_{12}} = \frac{1.6678}{4.0416} = 0.4127 \text{ units/month}$$
$$\hat{y}_{13}^{\text{SBA}} = \left(1 - \frac{\alpha}{2}\right) \frac{z_{12}}{p_{12}} = \left(1 - \frac{0.15}{2}\right) \times 0.4127 = (0.925) \times 0.4127 = 0.3817 \text{ units/month}$$

Applying SBA eliminates a 7.5% systematic positive demand bias.

---

### Step 3: Discrete Poisson Safety Stock and Reorder Point Sizing
Now, calculate the safety stock and reorder point for the complete mechanical seal cartridge.
- Supplier replenishment lead time: $L = 6$ months.
- Cumulative failure demand expected during lead time (from NHPP wear-out):
  $$\mu_L = 0.9019 \text{ units}$$
- Target Cycle Service Level: $CSL = 95.0\%$ ($\alpha = 0.95$).

#### Exact Poisson Cumulative Probabilities:
Calculate $P(D_L = k) = \frac{(0.9019)^k e^{-0.9019}}{k!}$:
$$e^{-0.9019} = 0.40581$$
- $k = 0$:
  $$P(D_L = 0) = \frac{(0.9019)^0 \times 0.40581}{1} = 0.40581 \quad \implies F(0) = 0.40581$$
- $k = 1$:
  $$P(D_L = 1) = \frac{(0.9019)^1 \times 0.40581}{1} = 0.36599 \quad \implies F(1) = 0.40581 + 0.36599 = 0.77180$$
- $k = 2$:
  $$P(D_L = 2) = \frac{(0.9019)^2 \times 0.40581}{2} = \frac{0.8134 \times 0.40581}{2} = 0.16504 \quad \implies F(2) = 0.77180 + 0.16504 = 0.93684$$
- $k = 3$:
  $$P(D_L = 3) = \frac{(0.9019)^3 \times 0.40581}{6} = \frac{0.7336 \times 0.40581}{6} = 0.04961 \quad \implies F(3) = 0.93684 + 0.04961 = 0.98645$$

#### Finding Reorder Point $s^*$:
- $F(0) = 40.58\% < 95\%$
- $F(1) = 77.18\% < 95\%$
- $F(2) = 93.68\% < 95\%$
- $F(3) = 98.65\% \ge 95\%$

Therefore, the exact discrete reorder point is:
$$s^* = 3 \text{ units}$$
Safety Stock:
$$SS = s^* - \mu_L = 3 - 0.9019 = 2.0981 \text{ units}$$

#### Critique of Naive Gaussian Safety Stock:
If a planner had applied the naive normal distribution formula with $z_{0.95} = 1.6449$:
$$SS_{\text{Gauss}} = z \cdot \sqrt{\mu_L} = 1.6449 \cdot \sqrt{0.9019} = 1.6449 \cdot 0.9497 = 1.5621$$
$$s_{\text{Gauss}} = \mu_L + SS_{\text{Gauss}} = 0.9019 + 1.5621 = 2.4640$$
If rounded down to 2 units (a common ERP default truncation), actual achieved service level is only $F(2) = 93.68\%$, leaving the power plant exposed to a 6.32% stockout risk during the lead time. In our CCGT plant, that 6.32% risk translates to an expected downtime risk cost of:
$$\text{Risk Cost} = (1 - 0.9368) \times (325 \text{ MW} \times 48 \text{ hours forced outage} \times \$65/\text{MWh}) = 0.0632 \times \$1,014,000 = \$64,085$$
Carrying the third seal cartridge incurs an annual holding cost of:
$$C_{\text{holding}} = 1 \text{ unit} \times \$28,000 \text{ unit cost} \times 18\% = \$5,040/\text{year}$$
Spending \$5,040 annually to avoid an expected \$64,085 downtime exposure produces an annual net expected return of \$59,045. The discrete Poisson calculation provides the exact mathematical justification.

---

## 7. Code, query, or calculation

The following self-contained Python script implements:
1. Intermittent demand forecasting comparing Croston and SBA.
2. Crow-AMSAA parameter calibration and expected failure forecasting over replenishment lead times.
3. Exact discrete Poisson reorder point sizing for a specified Cycle Service Level.

```python
import math

def forecast_intermittent_sba(demand_series, alpha=0.15):
    non_zeros = [y for y in demand_series if y > 0]
    z = sum(non_zeros) / len(non_zeros) if non_zeros else 1.0
    p = len(demand_series) / len(non_zeros) if non_zeros else 1.0
    q = 1
    for y in demand_series:
        if y > 0:
            z = alpha * y + (1.0 - alpha) * z
            p = alpha * q + (1.0 - alpha) * p
            q = 1
        else:
            q += 1
    croston_f = z / p
    sba_f = (1.0 - (alpha / 2.0)) * croston_f
    return {"croston": croston_f, "sba": sba_f, "z": z, "p": p}

def fit_crow_amsaa(event_times, end_time):
    n = len(event_times)
    log_sum = sum(math.log(end_time / t) for t in event_times)
    beta = n / log_sum
    scale = n / (end_time ** beta)
    return beta, scale

def nhpp_expected_failures(beta, scale, t_start, t_end):
    return scale * (t_end ** beta - t_start ** beta)

def poisson_reorder_point(lead_time_demand, target_csl=0.95):
    s = 0
    prob = math.exp(-lead_time_demand)
    cum_prob = prob
    while cum_prob < target_csl:
        s += 1
        prob = (prob * lead_time_demand) / s
        cum_prob += prob
    return s, cum_prob

# Verification execution with worked example parameters
history = [0, 0, 2, 0, 0, 0, 1, 0, 0, 0, 0, 2]
fcst = forecast_intermittent_sba(history, alpha=0.15)
print(f"SBA Monthly Rate: {fcst['sba']:.4f} (Croston: {fcst['croston']:.4f})")

events = [4200, 11800, 18600, 24900, 30100, 34500]
beta, scale = fit_crow_amsaa(events, end_time=36000)
mu_L = nhpp_expected_failures(beta, scale, 36000, 40000)
print(f"Crow-AMSAA Beta: {beta:.4f}, Lead-Time Demand (4k hrs): {mu_L:.4f}")

s_point, achieved_csl = poisson_reorder_point(mu_L, target_csl=0.95)
print(f"Optimal Reorder Point (s*): {s_point} units (Achieved CSL: {achieved_csl:.4f})")
```

### Script Execution Output
```text
SBA Monthly Rate: 0.3817 (Croston: 0.4127)
Crow-AMSAA Beta: 1.3291, Lead-Time Demand (4k hrs): 0.9019
Optimal Reorder Point (s*): 3 units (Achieved CSL: 0.9864)
```

---

## 8. Data, standards, and regulatory context

MRO failure modeling and inventory planning operate under strict regulatory, safety, and data governance frameworks. Because asset failure in energy systems can precipitate human casualty, environmental contamination, or regional grid collapse, reliability datasets and stocking policies are subject to external audit.

### International Reliability Standards

#### ISO 14224:2016
*Petroleum, petrochemical and natural gas industries — Collection and exchange of reliability and maintenance data for equipment.*
- **Mandate:** Establishes a standardized taxonomy for asset hierarchy (Industry $\to$ Business Category $\to$ Installation $\to$ Plant/Unit $\to$ Section/System $\to$ Equipment Unit $\to$ Subunit $\to$ Component/Maintainable Item).
- **Failure Mode Taxonomy:** Requires failure records to explicitly distinguish "Failure Mode" (observable effect, e.g., external seal leakage), "Failure Mechanism" (physical degradation, e.g., abrasive wear, cavitation), and "Failure Cause" (root etiology, e.g., lubrication starvation, operator error).
- **Impact on Forecasting:** Enforces the separation of operating exposure (run hours, starts) from idle standby time, providing the foundational exposure metric $t$ necessary for NHPP calibration.

#### IEEE Std 493-2007 (IEEE Gold Book)
*Design of Reliable Industrial and Commercial Power Systems.*
- **Scope:** Provides comprehensive empirical reliability data for electrical distribution components: transformers, switchgear, vacuum circuit breakers, disconnect switches, and protective relays.
- **Application:** Supplies default baseline failure rates ($\lambda_0$) and mean time to repair ($MTTR$) for new installations lacking historical operational telemetry.

#### ISO 55000 / 55001:2014
*Asset Management — Overview, principles and terminology / Management systems — Requirements.*
- **Requirements:** Section 6.2 mandates that organizations establish risk-based decision-making processes for asset lifecycle management. It explicitly binds spare parts inventory strategy to asset criticality and residual operational risk.

### Regulatory and Reliability Compliance

#### NERC GADS (Generating Availability Data System)
- **Applicability:** Mandatory reporting for North American generation assets $\ge 20$ MW connected to the bulk electric system.
- **Mandatory Reporting:** Utilities must report every forced outage event, unserved megawatt capacity, component cause code (IEEE/NERC standard codes), and the duration of the outage.
- **Supply Chain Linkage:** When a forced outage is prolonged due to the unavailability of spare parts, it must be reported with Cause Code 9999 ("Lack of Spare Parts"). Extended outages attract severe regulatory scrutiny and directly degrade the generator's Equivalent Forced Outage Rate (EFORd), lowering capacity payment revenues in regional transmission organization (RTO) capacity auctions.

#### NRC 10 CFR 50, Appendix B (Nuclear Quality Assurance)
- **Criterion XIII (Handling, Storage and Shipping):** Dictates rigorous environmental and shelf-life controls for safety-related MRO items (elastomers, lubricants, instrumentation relays).
- **Commercial Grade Dedication (CGD):** If an MRO item is procured from a non-nuclear certified supplier, it must undergo extensive destructive and non-destructive physical testing before being accepted into safety-related inventory. This extends replenishment lead times from weeks to months, requiring substantial Poisson buffer increases.

#### EPA Clean Air Act & Clean Water Act (Leak Detection and Repair - LDAR)
- Petrochemical and refining facilities operate under mandatory LDAR regulations. If an uncontained mechanical seal leak exceeds 500 ppm volatile organic compounds (VOCs), the plant faces statutory fines of up to \$45,000 per day until repaired. This environmental penalty dramatically increases the stockout penalty cost $C_{\text{downtime}}$, shifting the optimal base-stock point upward.

---

## 9. Industry practice and case evidence

### Case Evidence 1: High-Voltage Transmission Transformer Bushing Sparing (European TSO)

```text
Transformer Fleet (400 kV / 220 kV)
  |
  +---> Online DGA & Bushing Power Factor Monitoring
  |      - Detect C2H2 trace arcing & Tan Delta dielectric rise
  |      - Inception of Potential Failure (P)
  |
  +---> Asset Aging Mechanism: NHPP Beta = 1.48 (Aging Fleet)
  |      - High infant mortality post-replacement, then severe wear-out > 25 yrs
  |
  +---> Supply Chain Intervention:
         - Displaced Static 1-for-1 Replenishment with Palm's Theorem Pooling
         - Lead Time L = 14 months (Specialized porcelain supply bottleneck)
         - Reduced System EENS by 42% while trimming fleet spare capital by €11.4M
```

A major Western European Transmission System Operator (TSO) operating over 1,200 high-voltage substation power transformers faced an aging asset demographic: more than 40% of the transformer fleet exceeded 35 years of service life. High-voltage oil-impregnated paper (OIP) bushings represented the single largest contributor to catastrophic transformer tank fires and explosive ruptures.

Historically, the utility maintained an ad-hoc sparing policy: regional depots held arbitrary numbers of replacement bushings based on local depot manager discretion. When supply chain lead times for specialized 400 kV condenser bushings spiked from 18 weeks to 56 weeks due to ceramic porcelain raw material shortages, multiple substations operated without on-hand insurance spares.

The utility re-engineered its sparing architecture:
1. **NHPP Fleet Modeling:** Fitting Crow-AMSAA models to fleet failure records revealed $\hat{\beta} = 1.48$, indicating substantial acceleration in failure intensity across transformers older than 25 years.
2. **CBM Telemetry Integration:** Online automated Dissolved Gas Analysis (DGA) monitors and continuous dielectric loss factor ($\tan \delta$) monitoring were linked to ERP materials planning. If $\tan \delta$ increased by more than $0.2\%$ per month or hydrogen/acetylene ratios indicated electrical partial discharge, an automated procurement reservation was triggered against central depot inventory.
3. **Palm's Theorem Depot Pooling:** Instead of maintaining localized regional inventories, the TSO centralized all emergency insurance bushings into two strategic regional logistics hubs. By applying Palm's theorem ($M/G/\infty$ queueing model) across the pooled failure rate of the aggregated grid network, the required system-wide bushing stock was reduced from 84 units to 48 units, liberating €11.4 million in tied-up working capital while simultaneously cutting expected unserved energy (EENS) risk by 42%.

---

### Case Evidence 2: Offshore Gas Compression Choke Valve Sparing (North Sea Operator)

An upstream oil and gas operator in the Norwegian sector of the North Sea operated three deepwater production platforms. Production choke valves on subsea manifold tie-backs were subject to severe erosive wear from high-velocity multiphase hydrocarbons containing entrained formation sand.

Under standard ERP Gaussian safety stock rules, the platform logistics system continuously experienced stockouts of high-erosion tungsten-carbide trim sets. Each unplanned choke valve change-out required mobilizing a specialized light well intervention vessel (LWIV) at a day rate exceeding \$180,000. When a trim set was missing, the well was throttled, causing daily deferred production losses of \$450,000.

The operator executed the following analytics transformation:
- Replaced standard Gaussian safety stock calculations with the **Syntetos-Boylan Approximation** for consumable gaskets and **Compound Poisson lead-time demand modeling** for valve trim sets.
- Installed ultrasonic clamp-on sand monitors and choke acoustic emission sensors to quantify real-time sand production rates ($g/\text{sec}$) and valve vibration amplitude.
- Integrated sand exposure rates as time-varying covariates into a Cox Proportional Hazards Model, allowing the engineering team to predict choke trim Remaining Useful Life within a $\pm 10$-day confidence interval.
- **Documented Result:** Unplanned emergency choke valve mobilizations decreased from 14 events per year to 2 events per year over a 36-month tracking window. Total annual MRO inventory carrying cost dropped by 22% due to the elimination of panic spot-buying and redundant offshore cage hoarding.

---

### Case Evidence 3: Thermal Fleet Gas Turbine Combustor Basket Failures (North American Independent Power Producer)

A North American independent power producer (IPP) operating a fleet of 18 aeroderivative gas turbines experienced recurrent forced outages due to combustor basket thermal barrier coating (TBC) spallation and flame nozzle crack propagation. 

The maintenance organization had applied simple 12-month rolling moving average forecasts to order hot-gas-path (HGP) rotables. During years 1 to 4 following commercial commissioning, no combustor baskets failed, prompting the ERP system to automatically reduce reorder points to zero. In year 5, as the fleet reached 32,000 equivalent operating hours (EOH), seven combustor baskets suffered severe cracking within a 90-day window. The OEM lead time for replacement rotable baskets was 40 weeks. The IPP was forced to de-commit from its RTO capacity obligations, incurring \$3.8 million in non-performance penalties.

The company abandoned moving averages and implemented:
- **Teunter-Syntetos-Babai (TSB)** forecasting across all gas turbine rotable assemblies to handle cyclic maintenance campaigns and detect component obsolescence following turbine upgrades.
- **Crow-AMSAA NHPP Modeling** calibrated directly against equivalent operating hours and thermal starts. The model demonstrated that the combustor baskets entered an acute wear-out regime ($\hat{\beta} = 2.15$) immediately upon crossing 28,000 EOH.
- The inventory strategy was shifted to a continuous-review base-stock $(S-1, S)$ policy governed by Palm's theorem, incorporating an empirical turnaround time ($TAT$) of 24 weeks at the authorized OEM repair depot. The fleet has since maintained a 100% parts availability record during major hot-gas-path turnaround windows.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Root Mechanism | Operational Consequence | Mitigation Strategy |
| :--- | :--- | :--- | :--- | :--- |
| **Croston Positive Bias** | Systemic creeping inventory buildup across slow-moving parts. | Applying Croston without the $(1 - \alpha/2)$ SBA de-biasing multiplier; Jensen's inequality inflates demand rate. | Working capital locked in low-turnover consumables. | Replace Croston with SBA across all SKUs where $ADI \ge 1.32$ and $CV^2 < 0.49$. |
| **Obsolescence Trap** | Inactive parts retain positive reorder points years after asset retirement. | Croston/SBA update inter-arrival time $p_t$ only upon positive demand; zero demand freezes the forecast. | Obsolete inventory purchased for decommissioned equipment. | Deploy TSB algorithm; update demand probability $p_t$ every period to decay zero-demand parts. |
| **HPP Aging Assumption** | Spares stockouts surge as an asset fleet passes 15–20 years of operation. | Assuming constant failure rate ($\beta = 1$) for mechanical/electrical assets undergoing thermal/fatigue aging. | Severe unserved energy and forced outage downtime as wear-out accelerates. | Fit NHPP Power Law models; calibrate $\beta$; adjust lead-time demand by $\Lambda(t+L) - \Lambda(t)$. |
| **Gaussian Truncation** | ERP safety stock calculated as zero or negative for critical parts. | Normal distribution assumption applied to discrete low-mean Poisson demand ($\mu_L < 1.0$). | Under-protection of critical single-point-of-failure assets. | Enforce discrete Poisson or Compound Poisson distribution safety stock formulas. |
| **Stockout Censoring** | Forecast drops following an extended plant forced outage. | Zero warehouse issues during stockouts logged as zero demand in ERP history. | Chronic vicious cycle: stockout depresses forecast, leading to further stocking cuts. | Uncensor demand: impute unconstrained demand during stockout periods using operating hours. |
| **Batch Artifact Distortion** | Single large turnaround order misclassified as high-frequency erratic demand. | Combining planned outage project requisitions with stochastic daily breakdown issues. | Inflated safety stocks for items whose demand is actually scheduled and project-driven. | Decouple planned turnaround work packages from unplanned stochastic breakdown demand. |
| **P-F Interval Misalignment** | CBM vibration alarm triggers, but replacement part lead time exceeds P-F window. | Inception-to-failure window ($P \to F$) shorter than supplier procurement lead time $L$. | Asset trips before replacement arrives; condition monitoring fails to prevent outage. | Establish strategic buffer stocks for all long-lead items ($L > \Delta t_{PF}$) regardless of CBM health. |
| **Rotable TAT Drift** | Depot stockouts occur despite stable asset failure rates. | Repair vendor turnaround time ($TAT$) increases without an adjustment in base stock $S$. | Backorders accumulate; assets sit idle waiting for overhauled rotables. | Monitor depot repair TAT as a dynamic parameter; adjust Palm base-stock level $S = \lambda \cdot TAT$. |
| **Severity Tier Neglect** | Peaking turbines experience far higher failure rates than base-load sister units. | Modeling failure strictly by calendar time rather than thermal starts and firing hours. | Inadequate spares allocated to cyclic, thermally stressed operating units. | Normalize exposure metric $t$ to Equivalent Operating Hours ($EOH = \text{hours} + a \cdot \text{starts}$). |
| **Field Cannibalization** | Machine operational, but identical sister unit in laydown cage stripped of components. | Maintenance technicians take parts from idle trains to bypass procurement delays. | Ghost availability; double the parts required when idle asset is returned to service. | Mandate ERP tracking of all component cannibalizations as explicit emergency demand events. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks It | How to Test / Validate It |
| :--- | :--- | :--- | :--- |
| **Minimal Repair (ABAO)** | Allows NHPP Power Law formulation; assumes a failed component repair restores system to pre-failure state. | Comprehensive major overhaul that replaces all primary wear components, effectively resetting asset age to like-new. | Test failure inter-arrival trends pre- and post-overhaul using the Laplace Trend Test. |
| **Poisson Independence** | Simplifies lead-time demand convolution into tractable discrete probability mass functions. | Common-cause cascading failures (e.g., severe grid frequency surge blowing multiple substation breakers). | Evaluate dispersion index ($Var / Mean$); if variance significantly exceeds mean, use Negative Binomial. |
| **Constant Supplier Lead Time** | Decouples demand rate forecasting from replenishment logistics modeling. | Supply chain disruption, raw material allocation, port congestion, or custom component forging backlogs. | Track vendor lead-time distributions; convolve stochastic lead time with stochastic demand using Panjer recursion. |
| **Stationary Inter-Arrivals** | Croston and SBA require underlying failure mechanism parameters to remain stationary between updates. | Sudden shifts in plant operating regime (e.g., base-load coal plant converted to flexible cycling). | Monitor Cumulative Sum (CUSUM) of forecast errors or track asset duty cycle changes. |
| **Deterministic Run-Hour Link** | Links cumulative operational exposure $t$ directly to calendar planning periods. | Volatile wholesale electricity prices causing dispatchers to cycle plants on and off unpredictably. | Re-estimate forward run-hours monthly using market dispatch simulation models (e.g., PLEXOS). |
| **Accurate P-F Detection** | Assumes diagnostic sensor instrumentation reliably detects incipient faults at point $P$. | Sensor drift, calibration error, blind spots (e.g., vibration sensor placed outside the vibration transmission path). | Conduct periodic bump-tests, oil lab cross-checks, and false-alarm / missed-detection audits. |
| **Infinite Repair Capacity** | Palm's Theorem ($M/G/\infty$) assumes no queuing delays when components enter repair depots. | OEM repair facility backlogs, specialized technician strikes, or machining center bottlenecks. | Monitor repair queue depths; if queuing occurs, model the repair facility as a finite-capacity $M/M/c$ queue. |
| **Non-Perishable Inventory** | Assumes MRO spare parts can remain in warehouse storage indefinitely without degradation. | Elastomeric seal embrittlement, bearing false brinelling from vibration, lubricant oxidation, capacitor leakage. | Implement shelf-life management, periodic motor shaft rotation, and climate-controlled storage policies. |

---

## 12. Exercises

### Exercise 1: Demand Classification and Regimes
A natural gas compressor station tracks warehouse demand for four critical SKUs over a 24-month observation window. The aggregated summary statistics are provided below:

| SKU ID | Description | Total Periods ($N$) | Non-Zero Periods ($K$) | Mean Non-Zero Size ($\mu_z$) | Std Dev Non-Zero Size ($\sigma_z$) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **SKU-A** | Filter Coalescer Element | 24 | 18 | 4.2 | 1.1 |
| **SKU-B** | High-Pressure Ball Valve Seat | 24 | 5 | 1.0 | 0.0 |
| **SKU-C** | Dry Gas Seal O-Ring Kit | 24 | 4 | 2.5 | 1.8 |
| **SKU-D** | Thermocouple Sensor Probe | 24 | 20 | 1.2 | 1.0 |

**Deliverable:**
1. Calculate the Average Demand Interval ($ADI$) and Squared Coefficient of Variation ($CV^2$) for each SKU.
2. Classify each SKU into its respective Syntetos-Boylan quadrant (Smooth, Intermittent, Erratic, or Lumpy).
3. Recommend the optimal statistical forecasting technique for each SKU.

*Acceptance Criteria:* Exact numerical calculation of $ADI$ and $CV^2$ to two decimal places; correct quadrant assignment based on the $(1.32, 0.49)$ threshold matrix.  
*Estimated Time:* 20 minutes.

---

### Exercise 2: Croston vs. SBA vs. TSB Forecast Evolution
An industrial boiler feed pump lubrication pressure switch recorded the following quarterly issue history over the last 8 quarters:
$$y = [0, 1, 0, 0, 2, 0, 0, 0]$$

**Deliverable:**
1. Trace the step-by-step forecast calculations for Quarters 1 through 8 using Croston's method with $\alpha = 0.20$. Assume initial states $z_0 = 1.0, p_0 = 2.0, q = 1$.
2. Compute the de-biased Syntetos-Boylan Approximation (SBA) forecast for Quarter 9.
3. Compute the Teunter-Syntetos-Babai (TSB) forecast for Quarter 9 using $\alpha = 0.20$ and $\beta_{\text{tsb}} = 0.15$. Assume initial probability $p_0^{\text{TSB}} = 0.50$.
4. Critique why the TSB forecast exhibits superior operational behavior given the three consecutive zero-demand quarters at the end of the time series.

*Acceptance Criteria:* Full arithmetic trace table showing $t, y_t, q, z_t, p_t, \hat{y}_{t+1}$ across all 8 periods; explicit numerical values for Quarter 9 forecasts.  
*Estimated Time:* 30 minutes.

---

### Exercise 3: Crow-AMSAA Parameter Estimation and Reliability Growth Analysis
A fleet of six newly commissioned 2.5 MW wind turbines is monitored over an initial operational observation period of $T = 12,000$ cumulative operating hours. Planetary gearbox planetary bearing spallation failures occur at the following cumulative fleet timestamps:
$$t_1 = 1,450 \text{ hrs}, \quad t_2 = 3,200 \text{ hrs}, \quad t_3 = 5,800 \text{ hrs}, \quad t_4 = 8,900 \text{ hrs}, \quad t_5 = 11,400 \text{ hrs}$$

**Deliverable:**
1. State whether the observation window represents Type I (time-truncated) or Type II (failure-truncated) censoring.
2. Calculate the Crow-AMSAA Maximum Likelihood Estimates for $\hat{\beta}$ and $\hat{\lambda}$.
3. Interpret the engineering meaning of $\hat{\beta}$. Does the gearbox fleet exhibit infant mortality, random useful life, or wear-out aging?
4. Calculate the expected number of gearbox bearing failures during the upcoming maintenance operating window from $T_1 = 12,000$ hours to $T_2 = 16,000$ hours.

*Acceptance Criteria:* Correct formula application for Type I censoring; logarithmic summation rounded to four decimal places; accurate derivation of $\mathbb{E}[N(12k, 16k)]$.  
*Estimated Time:* 25 minutes.

---

### Exercise 4: Exact Poisson vs. Gaussian Safety Stock Sizing
A critical step-down transformer auxiliary cooling oil pump has an expected failure rate of $\lambda = 0.05$ failures per month. The procurement lead time for a replacement hermetic pump assembly is $L = 6$ months. 

**Deliverable:**
1. Compute the expected lead-time demand $\mu_L$.
2. Calculate the exact discrete Poisson probability distribution $P(D_L = k)$ and cumulative distribution $F(s)$ for $k \in \{0, 1, 2, 3\}$.
3. Determine the required discrete reorder point $s^*$ to satisfy a $95\%$ Cycle Service Level ($CSL$).
4. Compute the naive continuous Gaussian reorder point $s_{\text{Gauss}}$ using $z_{0.95} = 1.6449$.
5. Evaluate the operational risk and explain the catastrophic inventory consequence if the Gaussian result is truncated or rounded down by an automated ERP planning run.

*Acceptance Criteria:* Explicit calculation of Poisson terms to five decimal places; exact integer reorder point identified; clear quantitative comparison of achieved service levels.  
*Estimated Time:* 25 minutes.

---

### Exercise 5: Rotable Closed-Loop Base-Stock Sizing (Palm's Theorem)
A nuclear power plant utility operates a fleet of 12 identical control rod drive mechanisms (CRDMs). CRDMs fail according to a Poisson process with an aggregate fleet failure rate of $\lambda = 3$ units per year. When a CRDM fails, it is removed, replaced from depot spares, and sent to an offsite OEM licensed hot-shop for full overhaul.

The logistics and overhaul turnaround time ($TAT$) follows a lognormal distribution with a mean of $\mathbb{E}[TAT] = 0.50$ years (6 months) and a standard deviation of 0.25 years.

**Deliverable:**
1. Explain why Palm's Theorem applies to this system despite the fact that turnaround time is lognormally distributed rather than exponentially distributed.
2. Calculate the mean number of units in the repair loop $\theta$.
3. Compute the probability of having $k$ units in repair for $k \in \{0, 1, 2, 3, 4, 5\}$.
4. Determine the minimum rotable base-stock level $S^*$ required to ensure that the probability of having no spares on the shelf (stockout probability $P(\text{Backorders} > 0)$) is less than $5\%$.
5. If carrying one spare CRDM costs \$120,000 per year, and an unserved CRDM backorder forces a reactor de-rate costing \$800,000 per month, justify whether the optimal base-stock level should be increased to $S^* + 1$.

*Acceptance Criteria:* Correct calculation of $\theta$; exact Poisson probabilities; clear mathematical and economic justification for the optimal base-stock level.  
*Estimated Time:* 30 minutes.

---

### Exercise 6: Condition-Based CBM Work Order Integration
A 345 kV high-voltage circuit breaker's hydraulic operating mechanism exhibits nitrogen gas accumulator pressure decay. Continuous sensor telemetry tracks the pressure drop rate. The baseline hazard is modeled via a Weibull rate with shape $\beta = 2.0$ and scale $\eta = 5,000$ operating cycles.

A Cox Proportional Hazards Model incorporates the pressure decay rate covariate $Z_1(t)$ (measured in $\text{bar}/\text{month}$):
$$h(t, Z_1(t)) = h_0(t) \cdot \exp(\gamma_1 \cdot Z_1(t))$$
where the calibrated regression coefficient is $\gamma_1 = 1.45$.

The circuit breaker has currently accumulated $t = 3,500$ cycles. Sensor telemetry indicates that the pressure decay rate has accelerated to $Z_1(t) = 2.4 \text{ bar}/\text{month}$.

**Deliverable:**
1. Calculate the baseline hazard rate $h_0(3500)$ at the current cycle count.
2. Calculate the adjusted hazard rate $h(3500, Z_1=2.4)$.
3. Determine the hazard multiplication factor caused by the hydraulic pressure degradation.
4. If supplier lead time for a replacement hydraulic manifold is 45 days, and the breaker averages 15 operating cycles per month, formulate the procurement decision logic. Should the utility trigger an immediate purchase order or wait for the next annual scheduled maintenance overhaul?

*Acceptance Criteria:* Step-by-step arithmetic evaluation of the Weibull baseline hazard and the exponential covariate multiplier; clear operational procurement decision.  
*Estimated Time:* 25 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Demand Classification and Regimes
1. Compute metrics:
   - **SKU-A:** $ADI = 24 / 18 = 1.33$. $CV^2 = (1.1 / 4.2)^2 = (0.2619)^2 = 0.0686$.
   - **SKU-B:** $ADI = 24 / 5 = 4.80$. $CV^2 = (0.0 / 1.0)^2 = 0.0000$.
   - **SKU-C:** $ADI = 24 / 4 = 6.00$. $CV^2 = (1.8 / 2.5)^2 = (0.7200)^2 = 0.5184$.
   - **SKU-D:** $ADI = 24 / 20 = 1.20$. $CV^2 = (1.0 / 1.2)^2 = (0.8333)^2 = 0.6944$.
2. Categorization:
   - **SKU-A:** $ADI = 1.33 \ge 1.32$, $CV^2 = 0.07 < 0.49 \implies$ **Intermittent Demand**.
   - **SKU-B:** $ADI = 4.80 \ge 1.32$, $CV^2 = 0.00 < 0.49 \implies$ **Intermittent Demand**.
   - **SKU-C:** $ADI = 6.00 \ge 1.32$, $CV^2 = 0.52 \ge 0.49 \implies$ **Lumpy Demand**.
   - **SKU-D:** $ADI = 1.20 < 1.32$, $CV^2 = 0.69 \ge 0.49 \implies$ **Erratic Demand**.
3. Recommended Forecasting Methods:
   - **SKU-A & SKU-B:** Syntetos-Boylan Approximation (SBA). Eliminates Croston positive bias while handling long zero periods.
   - **SKU-C:** Teunter-Syntetos-Babai (TSB) with compound negative binomial distribution for inventory sizing.
   - **SKU-D:** Simple Exponential Smoothing (SES) or Croston with variance adjustment for erratic sizing.

---

### Answer to Exercise 2: Croston vs. SBA vs. TSB Forecast Evolution
1. Croston calculation trace ($\alpha = 0.20, z_0 = 1.0, p_0 = 2.0, q = 1$):
   - **Q1 ($y_1 = 0$):** $z_1 = 1.0, p_1 = 2.0, q = 2$.
   - **Q2 ($y_2 = 1$):** $y_2 > 0$.
     $$z_2 = 0.20(1) + 0.80(1.0) = 1.0$$
     $$p_2 = 0.20(2) + 0.80(2.0) = 0.40 + 1.60 = 2.0$$
     $$q = 1$$
   - **Q3 ($y_3 = 0$):** $z_3 = 1.0, p_3 = 2.0, q = 2$.
   - **Q4 ($y_4 = 0$):** $z_4 = 1.0, p_4 = 2.0, q = 3$.
   - **Q5 ($y_5 = 2$):** $y_5 > 0$.
     $$z_5 = 0.20(2) + 0.80(1.0) = 0.40 + 0.80 = 1.20$$
     $$p_5 = 0.20(3) + 0.80(2.0) = 0.60 + 1.60 = 2.20$$
     $$q = 1$$
   - **Q6 ($y_6 = 0$):** $z_6 = 1.20, p_6 = 2.20, q = 2$.
   - **Q7 ($y_7 = 0$):** $z_7 = 1.20, p_7 = 2.20, q = 3$.
   - **Q8 ($y_8 = 0$):** $z_8 = 1.20, p_8 = 2.20, q = 4$.
2. Quarter 9 Forecasts:
   $$\hat{y}_9^{\text{Croston}} = \frac{z_8}{p_8} = \frac{1.20}{2.20} = 0.5455 \text{ units/quarter}$$
   $$\hat{y}_9^{\text{SBA}} = \left(1 - \frac{0.20}{2}\right) \times 0.5455 = (0.90) \times 0.5455 = 0.4909 \text{ units/quarter}$$
3. TSB Calculation ($\alpha = 0.20, \beta_{\text{tsb}} = 0.15, p_0^{\text{TSB}} = 0.50$):
   - Q1 ($y=0$): $p_1 = 0.85(0.50) = 0.4250, z_1 = 1.0$
   - Q2 ($y=1$): $p_2 = 0.15(1) + 0.85(0.4250) = 0.15 + 0.3613 = 0.5113, z_2 = 1.0$
   - Q3 ($y=0$): $p_3 = 0.85(0.5113) = 0.4346, z_3 = 1.0$
   - Q4 ($y=0$): $p_4 = 0.85(0.4346) = 0.3694, z_4 = 1.0$
   - Q5 ($y=2$): $p_5 = 0.15(1) + 0.85(0.3694) = 0.15 + 0.3140 = 0.4640, z_5 = 1.20$
   - Q6 ($y=0$): $p_6 = 0.85(0.4640) = 0.3944, z_6 = 1.20$
   - Q7 ($y=0$): $p_7 = 0.85(0.3944) = 0.3352, z_7 = 1.20$
   - Q8 ($y=0$): $p_8 = 0.85(0.3352) = 0.2849, z_8 = 1.20$
   $$\hat{y}_9^{\text{TSB}} = p_8 \cdot z_8 = 0.2849 \times 1.20 = 0.3419 \text{ units/quarter}$$
4. Critique: Croston and SBA kept the inter-arrival interval frozen at $p = 2.20$ throughout Quarters 6, 7, and 8 because no demand occurred. Consequently, they predict a high demand rate ($0.49 - 0.55$). TSB actively degraded the demand probability from $0.4640$ down to $0.2849$, lowering the forecast by $30\%$ to reflect potential obsolescence.

---

### Answer to Exercise 3: Crow-AMSAA Parameter Estimation
1. Observation window ending at a fixed operational exposure time $T = 12,000$ hours is **Type I (time-truncated) censoring**.
2. Parameter Estimation:
   Compute $\ln(T / t_i)$:
   - $\ln(12,000 / 1,450) = \ln(8.2759) = 2.1134$
   - $\ln(12,000 / 3,200) = \ln(3.7500) = 1.3218$
   - $\ln(12,000 / 5,800) = \ln(2.0690) = 0.7271$
   - $\ln(12,000 / 8,900) = \ln(1.3483) = 0.2988$
   - $\ln(12,000 / 11,400) = \ln(1.0526) = 0.0513$
   $$\sum_{i=1}^5 \ln\left(\frac{12,000}{t_i}\right) = 2.1134 + 1.3218 + 0.7271 + 0.2988 + 0.0513 = 4.5124$$
   $$\hat{\beta} = \frac{n}{\sum \ln(T / t_i)} = \frac{5}{4.5124} = 1.1081$$
   $$\hat{\lambda} = \frac{n}{T^{\hat{\beta}}} = \frac{5}{(12,000)^{1.1081}} = \frac{5}{34,228} = 1.4608 \times 10^{-4}$$
3. Interpretation: $\hat{\beta} = 1.1081 > 1.0$. The gearbox fleet is beginning to exhibit early wear-out aging and fatigue spallation.
4. Expected failures in next window ($12,000$ to $16,000$ hours):
   $$\Lambda(12,000) = 5.000$$
   $$\Lambda(16,000) = (1.4608 \times 10^{-4}) \cdot (16,000)^{1.1081} = (1.4608 \times 10^{-4}) \cdot 47,078 = 6.8771$$
   $$\mathbb{E}[N(12k, 16k)] = 6.8771 - 5.0000 = 1.8771 \text{ failures}$$

---

### Answer to Exercise 4: Exact Poisson vs. Gaussian Safety Stock Sizing
1. Expected lead-time demand:
   $$\mu_L = \lambda \cdot L = 0.05 \text{ failures/month} \times 6 \text{ months} = 0.30 \text{ units}$$
2. Discrete Poisson Probabilities ($e^{-0.30} = 0.74082$):
   - $k=0$: $P(D_L = 0) = 0.74082 \implies F(0) = 0.74082$
   - $k=1$: $P(D_L = 1) = 0.30 \times 0.74082 = 0.22225 \implies F(1) = 0.74082 + 0.22225 = 0.96307$
   - $k=2$: $P(D_L = 2) = \frac{(0.30)^2 \times 0.74082}{2} = 0.03334 \implies F(2) = 0.96307 + 0.03334 = 0.99641$
   - $k=3$: $P(D_L = 3) = \frac{(0.30)^3 \times 0.74082}{6} = 0.00333 \implies F(3) = 0.99641 + 0.00333 = 0.99974$
3. Exact Reorder Point for $CSL = 95\%$:
   $F(0) = 74.08\% < 95\%$; $F(1) = 96.31\% \ge 95\%$.
   Therefore, exact reorder point $s^* = 1 \text{ unit}$.
4. Continuous Gaussian Reorder Point:
   $$s_{\text{Gauss}} = \mu_L + z \sqrt{\mu_L} = 0.30 + 1.6449 \sqrt{0.30} = 0.30 + 1.6449(0.5477) = 0.30 + 0.9009 = 1.2009$$
5. Operational Risk: If an automated ERP system truncates $1.2009$ down to an integer reorder point of $s = 0$ (interpreting $< 1.5$ as round down, or assuming continuous replenishment), the achieved service level collapses to $F(0) = 74.08\%$. The transformer is exposed to a $25.92\%$ probability of an uncontained outage during replenishment lead time.

---

### Answer to Exercise 5: Rotable Closed-Loop Base-Stock Sizing
1. **Palm's Theorem Invariance:** Palm's Theorem states that the steady-state distribution of an $M/G/\infty$ queue depends *solely on the mean* of the service time distribution ($\mathbb{E}[TAT]$), and is completely invariant to the functional form or variance of the distribution (whether exponential, lognormal, or deterministic).
2. Mean units in repair loop:
   $$\theta = \lambda \cdot \mathbb{E}[TAT] = 3 \text{ units/year} \times 0.50 \text{ years} = 1.50 \text{ units}$$
3. Poisson Probabilities for $\theta = 1.50$ ($e^{-1.50} = 0.22313$):
   - $P(0) = 0.22313 \implies F(0) = 0.22313$
   - $P(1) = 1.5 \times 0.22313 = 0.33470 \implies F(1) = 0.55783$
   - $P(2) = \frac{(1.5)^2 \times 0.22313}{2} = 0.25102 \implies F(2) = 0.80885$
   - $P(3) = \frac{(1.5)^3 \times 0.22313}{6} = 0.12551 \implies F(3) = 0.93436$
   - $P(4) = \frac{(1.5)^4 \times 0.22313}{24} = 0.04707 \implies F(4) = 0.98143$
4. Minimum Base Stock for Stockout Probability $< 5\%$:
   We require $P(\text{Backorders} > 0) = 1 - F(S-1) \le 0.05 \iff F(S-1) \ge 0.95$.
   Looking at cumulative distribution:
   - For $S = 4 \implies F(3) = 0.9344 < 0.95$ (Stockout risk $= 6.56\%$).
   - For $S = 5 \implies F(4) = 0.9814 \ge 0.95$ (Stockout risk $= 1.86\%$).
   Therefore, the minimum required base stock is $S^* = 5 \text{ units}$.
5. Economic Justification:
   Marginal benefit of increasing from $S=4$ to $S=5$:
   Reduction in backorder probability $= 1 - F(3) - (1 - F(4)) = P(4) = 0.04707$.
   Avoided monthly downtime penalty risk $= 0.04707 \times \$800,000 \times 12 = \$451,872/\text{year}$.
   Annual carrying cost of 5th spare $= \$120,000/\text{year}$.
   Since net annual benefit ($\$451,872 - \$120,000 = +\$331,872$) is strongly positive, carrying $S = 5$ units is financially mandatory.

---

### Answer to Exercise 6: Condition-Based CBM Work Order Integration
1. Baseline hazard at $t = 3,500$ cycles ($\beta = 2.0, \eta = 5,000$):
   $$h_0(t) = \frac{\beta}{\eta} \left(\frac{t}{\eta}\right)^{\beta - 1} = \frac{2}{5,000} \left(\frac{3,500}{5,000}\right)^{2 - 1} = (0.0004) \times (0.70) = 0.00028 \text{ failures/cycle}$$
2. Adjusted hazard rate with covariate $Z_1 = 2.4 \text{ bar/month}$:
   $$\exp(\gamma_1 \cdot Z_1) = \exp(1.45 \times 2.4) = \exp(3.48) = 32.4593$$
   $$h(3500, Z_1=2.4) = 0.00028 \times 32.4593 = 0.009089 \text{ failures/cycle}$$
3. Hazard Multiplication Factor:
   $$\frac{h(t, Z_1)}{h_0(t)} = \exp(3.48) = 32.46$$
   The hydraulic degradation accelerates the instantaneous failure hazard by over 32 times above baseline.
4. Procurement Decision:
   Monthly failure hazard $= 15 \text{ cycles/month} \times 0.009089 = 0.1363 \text{ failures/month}$.
   Expected failure time $= 1 / 0.1363 \approx 7.3 \text{ months}$, but cumulative failure probability over supplier lead time (45 days = 1.5 months = 22.5 cycles) is:
   $$P(\text{Fail in 45 days}) = 1 - \exp(-0.009089 \times 22.5) = 1 - \exp(-0.2045) = 18.5\%$$
   Waiting for the annual scheduled overhaul exposes the substation to an unacceptable $18.5\%$ risk of forced explosive breaker failure. **Decision: Issue an immediate emergency purchase order for the replacement hydraulic manifold and schedule a localized weekend outage within 30 days.**

---

## 14. Interview questions

### Question 1: Why does standard Gaussian safety stock fail catastrophically when applied to slow-moving MRO spare parts?
**Strong Answer:**  
A strong answer explains that the standard normal safety stock formula ($SS = z \cdot \sigma_L$) makes two assumptions that are violated by slow-moving MRO demand: continuity and symmetry. 

First, for slow-moving parts, the mean demand during replenishment lead time ($\mu_L = \lambda \cdot L$) is typically small (often $\mu_L < 1.0$). A continuous Gaussian distribution assigns substantial probability mass to negative demand, which is physically impossible. Second, physical failure demand arrives in discrete non-negative integer steps ($0, 1, 2, \dots$). Continuous normal distributions distort the probability of the zero-demand state and underestimate tail risk. 

When planners round or truncate continuous Gaussian results, automated ERP systems frequently set safety stocks to zero for critical single-point-of-failure assets. The correct mathematical approach is to use discrete Poisson or Compound Poisson lead-time demand distributions and solve for the reorder point $s^*$ using cumulative discrete mass probabilities.

**Weak Answer:**  
"Because MRO demand is hard to predict, has a lot of zeros, and normal distribution is meant for high-volume retail like Walmart, so the formula gets confused by zeros." (Fails to state the mathematical mechanisms of discrete probability, symmetry violations, or negative demand truncation).

---

### Question 2: Explain the mathematical mechanism of Croston’s method and why the Syntetos-Boylan Approximation (SBA) is required.
**Strong Answer:**  
Croston's method overcomes the limitation of simple exponential smoothing by decomposing an intermittent demand time series into two separate latent components: the non-zero demand size $z_t$ and the inter-arrival interval $p_t$ between positive demands. Both components are updated via exponential smoothing only when positive demand occurs. The point forecast is then formulated as the ratio $\hat{y}_{t+1} = z_t / p_t$.

However, as proven by Syntetos and Boylan, Croston's estimator is mathematically biased. Because $z_t$ and $p_t$ are independent random variables, $\mathbb{E}[z_t / p_t] = \mathbb{E}[z_t] \cdot \mathbb{E}[1/p_t]$. By Jensen's inequality, since $f(x) = 1/x$ is strictly convex for $x > 0$, $\mathbb{E}[1/p_t] > 1/\mathbb{E}[p_t]$. Consequently, Croston's method systematically over-forecasts demand. 

Syntetos and Boylan derived an analytical correction factor, $(1 - \alpha/2)$, which de-biases the ratio: $\hat{y}_{t+1}^{\text{SBA}} = (1 - \frac{\alpha}{2}) \frac{z_t}{p_t}$. In empirical utility inventory portfolios, applying SBA eliminates between 8% and 14% of phantom inventory holding costs without reducing fill rates.

**Weak Answer:**  
"Croston updates size and time separately. SBA is just a newer version of Croston that uses a different formula to make the forecast smaller because Croston is known to be too high." (Lacks Jensen's inequality derivation, mathematical definition of the correction factor, and structural understanding of why the bias arises).

---

### Question 3: How does the Teunter-Syntetos-Babai (TSB) method solve the "obsolescence trap" in capital asset spare parts?
**Strong Answer:**  
The obsolescence trap in Croston and SBA occurs because the inter-arrival interval $p_t$ is updated *exclusively when a positive demand transaction occurs*. If an industrial asset is retired, upgraded, or decommissioned, no further demand occurs ($y_t = 0$ indefinitely). Under Croston and SBA, because $y_t = 0$, $p_t$ and $z_t$ never update; they remain frozen at their last calculated values. The system continues to generate a positive replenishment forecast indefinitely for an asset that no longer exists.

Teunter, Syntetos, and Babai (2011) resolved this by altering the state space: instead of tracking the inter-arrival interval $p_t$, TSB tracks the **probability of demand occurrence in period $t$**, denoted $p_t^{\text{TSB}}$. Crucially, this probability is updated in *every single period*:
$$p_t^{\text{TSB}} = \beta_{\text{tsb}} \mathbb{I}(y_t > 0) + (1 - \beta_{\text{tsb}}) p_{t-1}^{\text{TSB}}$$
When an asset enters obsolescence and demand ceases, $\mathbb{I}(y_t > 0) = 0$ in every subsequent period. The demand probability decays geometrically at rate $(1 - \beta_{\text{tsb}})^k$, driving the overall forecast $\hat{y}_{t+1} = p_t^{\text{TSB}} \cdot z_t$ toward zero and allowing automated phase-out of safety stocks.

**Weak Answer:**  
"TSB uses a different parameter to track obsolescence so that when parts stop being used, the inventory system notices and stops ordering them." (Fails to articulate that TSB updates demand probability in zero periods, whereas Croston only updates during positive transactions).

---

### Question 4: Under what conditions is a Non-Homogeneous Poisson Process (NHPP) required instead of a standard Homogeneous Poisson Process (HPP), and what does $\beta > 1$ indicate?
**Strong Answer:**  
A Homogeneous Poisson Process (HPP) assumes that failure events arrive at a constant instantaneous rate $\lambda(t) = \lambda_0$, meaning the asset is memoryless and its failure inter-arrival times follow an exponential distribution. This is valid only during the middle "useful life" phase of the bathtub curve.

In capital infrastructure (power plants, refineries, T&D grids), assets operate for decades under cyclic thermal, mechanical, and electrical stresses. When physical components experience wear, erosion, corrosion, or insulation breakdown, their failure rate is not constant; it accelerates with operating time or duty cycles. An NHPP Power Law Process (Crow-AMSAA) models this time-dependent intensity: $\lambda(t) = \lambda \beta t^{\beta - 1}$.

The shape parameter $\beta$ is the elasticity of failure intensity with respect to operating time:
- $\beta = 1$: HPP (constant failure rate, memoryless).
- $\beta < 1$: Reliability growth or infant mortality (decreasing failure intensity, common during plant commissioning).
- $\beta > 1$: Wear-out aging (accelerating failure intensity).
For an aging utility asset with $\beta > 1$, using an HPP based on historical average failure rates will severely under-estimate future lead-time failures, resulting in stockouts during the wear-out phase.

**Weak Answer:**  
"NHPP is used when failures are not constant. Beta greater than one means the machine is getting old and breaking down more often." (Lacks the formal mathematical formulation of the intensity function, the connection to the Crow-AMSAA model, and the distinction between HPP and NHPP).

---

### Question 5: What is Palm’s Theorem, and how does it govern rotable spare parts inventory?
**Strong Answer:**  
Palm's Theorem (1938) governs repairable, closed-loop rotable inventory systems. It states that if failure demand arrivals follow a Poisson process with rate $\lambda$, and the repair turnaround times ($TAT$) are independent, identically distributed random variables with arbitrary distribution and mean $\mathbb{E}[TAT]$, then the steady-state number of items undergoing repair in the repair facility follows a Poisson distribution with mean:
$$\theta = \lambda \cdot \mathbb{E}[TAT]$$

The operational significance is twofold:
1. **Distributional Invariance:** The steady-state distribution of items in repair depends *exclusively on the mean* of the turnaround time, not its variance or shape (whether normal, lognormal, Weibull, or deterministic).
2. **Base-Stock Sizing:** Under a continuous review $(S-1, S)$ policy, the probability of experiencing a stockout (having all $S$ spares exhausted and an asset waiting on backorder) is exactly equal to the probability that the number of items in repair exceeds $S$. Therefore, base stock $S^*$ is determined by solving the cumulative Poisson distribution $F(S; \theta) \ge 1 - \alpha_{\text{risk}}$ or by balancing marginal holding costs against unserved energy downtime penalties.

**Weak Answer:**  
"Palm's theorem is used for repairable parts. It calculates how many spares you need in the workshop loop based on how fast the repair shop works." (Fails to cite the $M/G/\infty$ queueing origin, the Poisson mean $\lambda \cdot TAT$, or the invariance property).

---

### Question 6: How do you mathematically uncensor historical MRO demand when a warehouse has experienced prolonged stockouts?
**Strong Answer:**  
In transactional ERP systems, warehouse logs record issues, not true demand. If an SKU was completely out of stock for 4 months, the ERP records zero issues. Treating those zeros as true demand artificially depresses the estimated arrival rate $\lambda$ or inflates the inter-arrival interval $p_t$, leading to lower safety stock recommendations precisely when the item has demonstrated high demand risk.

To uncensor the time series:
1. **Identify Censored Periods:** Cross-reference inventory stock level ledgers to identify time buckets where on-hand inventory was zero ($I_t = 0$) and backorders were logged, or where maintenance work orders were deferred.
2. **Imputation via Run-Hour Exposure:** For asset-linked spares, impute demand during the stockout window by multiplying the operating exposure accumulated during that window ($H_{\text{stockout}}$) by the baseline failure rate calibrated during unconstrained operating periods: $\hat{y}_{\text{censored}} = \lambda_{\text{unconstrained}} \cdot H_{\text{stockout}}$.
3. **Statistical Imputation:** For generic consumables, treat the stockout period as missing data and apply Expectation-Maximization (EM) or substitute the conditional expected value $\mathbb{E}[Y \mid Y > 0]$ derived from non-censored intervals.

**Weak Answer:**  
"You look at the purchase orders or ask the maintenance guys what they wanted to order while it was out of stock and add it back into Excel." (Fails to provide a rigorous mathematical framework, operational exposure rate linkage, or statistical imputation methodology).

---

### Question 7: How does Condition-Based Maintenance (CBM) telemetry alter the traditional stochastic safety stock calculation?
**Strong Answer:**  
Traditional safety stock calculations treat failure as a purely stochastic, memoryless event requiring a permanent statistical buffer. CBM telemetry—such as vibration RMS spectra, oil debris particle counts, and dissolved gas analysis (DGA)—identifies the physical onset of failure (Point P on the P-F curve).

When a diagnostic indicator breaches a potential failure threshold, the time to functional failure ($F$) transitions from an unconditioned stochastic distribution to a deterministic, bounded window: the **Remaining Useful Life (RUL)**. 

If the calculated $\text{RUL} > L_{\text{procurement}} + T_{\text{safety}}$, the supply chain does not need to hold an on-hand safety stock; it can rely on just-in-time procurement triggered by the CBM alarm. If $\text{RUL} < L_{\text{procurement}}$, the CBM alarm provides an early warning to expedite emergency procurement, activate mutual aid pools, or schedule a controlled asset de-rate to stretch the P-F interval. CBM thus converts stochastic inventory holding into synchronized, deterministic project kitting.

**Weak Answer:**  
"CBM tells you when the machine is going to break so you don't need safety stock because you just order the part when the sensor turns red." (Ignores the relationship between the P-F interval and procurement lead time, and fails to recognize that if $L > \Delta t_{PF}$, safety stock is still mandatory).

---

### Question 8: How do you balance holding costs against unserved energy penalties for multi-million-dollar insurance spares?
**Strong Answer:**  
Insurance spares (e.g., large generator step-up transformers, spare turbine rotors) exhibit ultra-low annual failure rates ($\lambda \approx 0.01 - 0.05$) and long replacement lead times ($L \approx 12 - 24$ months), with unit capital costs ranging from \$1M to \$8M. Holding costs ($C_h = h \cdot C_u$) are substantial, but the cost of an unmitigated stockout ($C_s$) involves months of plant downtime, lost gross margin, and Expected Unserved Energy (EENS) regulatory penalties.

The decision is formulated as an expected total cost minimization problem under discrete Poisson lead-time demand:
$$\min_{S \in \{0, 1, 2\}} \mathbb{E}[TC(S)] = C_h \cdot S + C_s \cdot EBO(S)$$
where $EBO(S)$ is the expected backorders given stock level $S$. 
Because $S$ is discrete, evaluate the marginal economic benefit:
$$\Delta \mathbb{E}[Cost] = C_s \cdot [EBO(S-1) - EBO(S)] - C_h$$
For an insurance spare, holding $S=1$ is mathematically justified if:
$$C_s \cdot P(D_L \ge 1) \ge C_h \iff P(D_L \ge 1) \ge \frac{C_h}{C_s}$$
Because the critical ratio $C_h / C_s$ is often extremely small (e.g., $\$100\text{k/yr} / \$15\text{M} = 0.0067$), an annual failure probability of just 1% is sufficient to mandate stocking the spare.

**Weak Answer:**  
"You calculate the ROI by looking at how much downtime costs per day versus how much the spare part costs to buy." (Fails to formulate the marginal discrete cost equation, expected backorder calculation, or the critical ratio rule).

---

## 15. Key terms

- **Average Demand Interval ($ADI$):** The average number of time periods elapsed between two consecutive non-zero demand events.
- **As-Bad-As-Old (ABAO):** The assumption that a minimal maintenance repair restores a failed system to operating condition without improving its underlying failure intensity.
- **As-Good-As-New (AGAN):** The assumption that a maintenance overhaul completely renews the asset, resetting its age and failure intensity to zero.
- **Base-Stock Policy $(S-1, S)$:** A continuous review inventory replenishment policy where every inventory withdrawal immediately triggers a replenishment order of equal size.
- **Compound Poisson Process:** A stochastic process where event arrivals follow a Poisson distribution, but each event carries a random discrete transaction size.
- **Condition-Based Maintenance (CBM):** Maintenance triggered by real-time diagnostic sensor measurements indicating physical degradation.
- **Cox Proportional Hazards Model (PHM):** A semi-parametric survival regression model that scales a baseline hazard function by an exponential combination of diagnostic covariates.
- **Croston’s Method:** A forecasting technique for intermittent demand that independently smoothes non-zero transaction sizes and inter-arrival intervals.
- **Crow-AMSAA Model:** A parametric Non-Homogeneous Poisson Process formulation with a Power Law intensity function used to model reliability growth and aging.
- **Cycle Service Level ($CSL$):** The probability that total demand during replenishment lead time does not exceed the inventory reorder point.
- **Dissolved Gas Analysis (DGA):** A diagnostic technique that measures concentrations of combustible hydrocarbon gases in transformer oil to detect internal electrical and thermal faults.
- **Expected Backorders ($EBO$):** The expected number of unfilled customer or maintenance demands outstanding at any random point in time under a given stocking policy.
- **Expected Unserved Energy (EENS):** A power grid reliability metric quantifying the expected megawatt-hours of electricity that cannot be supplied due to generation or transmission outages.
- **Fill Rate ($FR$):** The proportion of total customer or work-order demand volume that is satisfied immediately from available shelf inventory.
- **Homogeneous Poisson Process (HPP):** A memoryless stochastic point process characterized by a strictly constant instantaneous arrival rate.
- **Intermittent Demand:** A demand time series characterized by frequent periods of zero demand interspersed with occasional positive transactions.
- **Lumpy Demand:** A demand pattern characterized by both long inter-arrival intervals ($ADI \ge 1.32$) and high transaction size variability ($CV^2 \ge 0.49$).
- **Non-Homogeneous Poisson Process (NHPP):** A stochastic point process where the arrival intensity function varies with time or operational exposure.
- **Palm’s Theorem:** A queueing theorem stating that the steady-state number of items in an $M/G/\infty$ service system follows a Poisson distribution with mean $\lambda \cdot \mathbb{E}[TAT]$.
- **Panjer Recursion:** An efficient recursive algorithm for calculating the exact probability distribution of compound Poisson and compound renewal processes.
- **P-F Interval:** The operational duration between the detection of a potential failure symptom ($P$) and actual functional breakdown ($F$).
- **Remaining Useful Life (RUL):** The expected operating duration remaining before an operating asset breaches its functional failure threshold.
- **Rotable Spare:** A high-value, serialized capital component that is refurbished and reused across multiple operating cycles rather than scrapped.
- **Shape Parameter ($\beta$):** The elasticity parameter in Weibull and Crow-AMSAA models indicating infant mortality ($\beta < 1$), random failure ($\beta = 1$), or wear-out aging ($\beta > 1$).
- **Syntetos-Boylan Approximation (SBA):** A de-biasing modification of Croston's forecasting method using the multiplier $(1 - \alpha/2)$.
- **Teunter-Syntetos-Babai (TSB) Method:** An intermittent demand forecasting method that updates demand occurrence probability in every period to detect obsolescence.
- **Turnaround Time ($TAT$):** The total elapsed operational time required to transport, overhaul, test, and restock a repairable rotable component.

---

## 16. Sources

- **Ascher, H., & Feingold, H.** (1984). *Repairable Systems Reliability: Modeling, Inference, Misconceptions and Their Causes*. Marcel Dekker, New York. ISBN: 0-8247-7276-8.
- **Croston, J. D.** (1972). Forecasting non-normal demand for slow-moving items. *Operational Research Quarterly*, 23(3), 289–303. DOI: 10.2307/3007885.
- **Crow, L. H.** (1974). Reliability analysis for complex, repairable systems. In F. Proschan & R. J. Serfling (Eds.), *Reliability and Biometry: Statistical Analysis of Lifelength* (pp. 379–410). SIAM, Philadelphia.
- **IEEE Std 493-2007.** *IEEE Recommended Practice for the Design of Reliable Industrial and Commercial Power Systems (IEEE Gold Book)*. IEEE Industry Applications Society. ISBN: 978-0-7381-5677-4.
- **ISO 14224:2016.** *Petroleum, petrochemical and natural gas industries — Collection and exchange of reliability and maintenance data for equipment*. International Organization for Standardization, Geneva.
- **ISO 55000:2014.** *Asset management — Overview, principles and terminology*. International Organization for Standardization, Geneva.
- **Jardine, A. K., Lin, D., & Banjevic, D.** (2006). A review on machinery diagnostics and prognostics implementing condition-based maintenance. *Mechanical Systems and Signal Processing*, 20(7), 1483–1510. DOI: 10.1016/j.ymssp.2005.09.012.
- **NERC.** (2024). *Generating Availability Data System (GADS) Data Reporting Instructions*. North American Electric Reliability Corporation, Atlanta, GA.
- **Palm, C.** (1938). Analysis of the Erlang traffic formulae for busy-signal systems. *Ericsson Technics*, 6, 39–58.
- **Panjer, H. H.** (1981). Recursive evaluation of a family of compound distributions. *ASTIN Bulletin: The Journal of the IAA*, 12(1), 22–26. DOI: 10.2143/AST.12.1.2005118.
- **Silver, E. A., Pyke, D. F., & Peterson, R.** (1998). *Inventory Management and Production Planning and Scheduling* (3rd ed.). John Wiley & Sons, New York. ISBN: 978-0471119470.
- **Syntetos, A. A., & Boylan, J. E.** (2001). On the bias of intermittent demand estimates. *International Journal of Production Economics*, 71(1–3), 457–466. DOI: 10.1016/S0925-5273(00)00143-2.
- **Syntetos, A. A., & Boylan, J. E.** (2005). The accuracy of intermittent demand estimates. *International Journal of Forecasting*, 21(2), 303–314. DOI: 10.1016/j.ijforecast.2004.10.001.
- **Teunter, R. H., Syntetos, A. A., & Babai, M. Z.** (2011). Intermittent demand: Linking forecasting to inventory obsolescence. *European Journal of Operational Research*, 214(3), 606–615. DOI: 10.1016/j.ejor.2011.05.018.

---

## 17. Links

- **Up to Module Index:** [`_index.md`](_index.md)
- **Preceding Sibling Topic:** [`01_asset-register-criticality.md`](01_asset-register-criticality.md)
- **Succeeding Sibling Topic:** [`03_outage-turnaround-materials.md`](03_outage-turnaround-materials.md)
- **Related Project Procurement Topic:** [`04_long-lead-procurement.md`](04_long-lead-procurement.md)
- **Related Grid Equipment Topic:** [`08_renewables-grid-spares.md`](08_renewables-grid-spares.md)
- **Module Survey File:** [`../../../course/34_energy_utilities_capital_projects.md`](../../../course/34_energy_utilities_capital_projects.md)
- **Foundational Inventory Optimization:** [`../../02_methods/10_inventory_optimization_multi_echelon.md`](../../02_methods/10_inventory_optimization_multi_echelon.md)
- **Aftermarket & Spare Parts Service Supply Chains:** [`../31_aftermarket_spare_parts_service/01_installed-base-failure-forecast.md`](../31_aftermarket_spare_parts_service/01_installed-base-failure-forecast.md)
