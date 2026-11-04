# Terminal Simulation and Discrete-Event Modeling

> Module 41 — Maritime, Rail, and Air Cargo Operations Engineering
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 12 (Transportation, Routing, and Last Mile), Module 20 (Operations Research & Mathematical Programming), Module 41 Topic 01 (Berth and Quay Crane Operations), Module 41 Topic 02 (Terminal Yard and Gate Operations)
> Version 2026.5

---

## 1. Why this topic exists

Modern intermodal freight terminals—whether maritime container terminals handling 24,000 TEU Ultra-Large Container Vessels (ULCVs), rail intermodal classification yards marshalling 15,000-foot double-stack trains, or air cargo hubs processing thousands of Unit Load Devices (ULDs) within tight flight connection banks—represent some of the most capital-intensive, operationally coupled queueing networks in global industry. A state-of-the-art automated maritime container terminal demands capital expenditures between $1.0 billion and $2.5 billion. Individual Ship-to-Shore (STS) quay cranes cost $12 million to $16 million each, automated stacking cranes (ASCs) or rail-mounted gantries (RMGs) cost $4 million to $6 million each, and automated guided vehicles (AGVs) or autonomous container transporters cost $350,000 to $600,000 per unit, excluding guideway, charging, and navigation infrastructure.

At this scale, terminal operators and design engineering consortia cannot evaluate alternative berth layouts, equipment fleet sizing, automated dispatching algorithms, or operational disruption responses using static spreadsheet formulas or trial-and-error experimentation on live infrastructure. Static capacity models based on average cycle times systematically overestimate system throughput because they neglect variance, dynamic queue propagation, spatial blocking, and cyclic resource interdependencies. If an analytical model predicts that 4 AGVs per quay crane will achieve 35 moves per hour based on mean travel and hoist times, physical deployment in a live terminal frequently yields fewer than 26 moves per hour. The lost productivity stems from dynamic vehicle headway conflicts, quay crane starvation while waiting for transfer vehicles, yard gantry clashing, and buffer saturation at block hand-off areas.

The financial consequences of such miscalculations are severe:
1. **Quayside asset starvation:** When a $15 million quay crane idles for lack of a transfer vehicle, the idle crane time degrades Berth Moves Per Hour (BMPH). A 12-hour turnaround extension on a 20,000 TEU vessel imposes direct vessel idle costs of $45,000 to $90,000 per day in time-charter equivalent rates, while carrier partners burn additional metric tons of bunker fuel sprinting across the ocean leg to recover schedule reliability.
2. **Capital misallocation:** Purchasing excess mobile equipment to buffer against operational uncertainty creates physical gridlock on guideways and inflates capital charges. Conversely, under-sizing an automated vehicle fleet creates a persistent quayside bottleneck that permanently caps terminal capacity below designed berth throughput.
3. **Operational deadlock:** In automated terminals with dedicated guideways and narrow transfer aprons, dispatching vehicles without spatial-temporal conflict management induces physical and algorithmic deadlocks (circular wait conditions), freezing operations across entire berth sectors.

Discrete-Event Simulation (DES) exists as the core operational engineering discipline to solve these challenges. By modeling the state of the terminal strictly at discrete chronological points where events occur—such as crane hoist completions, vehicle dispatch assignments, optical character recognition (OCR) gate scans, or rail car positionings—DES enables rigorous, risk-free analysis of stochastic, non-linear, and constrained logistics systems. In contemporary operations, DES has evolved from an offline capital-planning tool into the algorithmic core of operational digital twins. Synchronized with live Terminal Operating Systems (TOS) via real-time telemetry and Electronic Data Interchange (EDI) milestone streams (such as EDI 315 and EDI 214), simulation engines project operational trajectories hours into the future, enabling dynamic resource re-allocation before quayside starvation or gate congestion materializes.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Architect** a discrete-event simulation model for coupled intermodal terminal systems, incorporating quay cranes, horizontal transfer vehicles (AGVs, terminal tractors), yard cranes (ASCs, RTGs), rail intermodal tracks, and gate complexes.
2. **Characterize and fit** empirical operational delay distributions—including shifted lognormal, three-parameter Weibull, and shifted gamma distributions—accounting for skewness, physical minimum durations, and heavy-tailed operational disruptions.
3. **Formulate** priority-queue-driven event-scheduling algorithms and closed queueing network abstractions that capture server starvation, downstream buffer blocking, and spatial vehicle interference.
4. **Determine** simulation warm-up periods and eliminate initialization bias using Welch's moving average procedure and marginal confidence interval heuristics.
5. **Execute** steady-state output analysis using the method of batch means and independent replications, demonstrating that batch sizes achieve statistical independence via lag-$k$ autocorrelation testing.
6. **Apply** variance reduction techniques (VRT), specifically Common Random Numbers (CRN), to evaluate competing terminal equipment configurations with higher statistical power and smaller sample sizes.
7. **Validate** terminal simulation outputs against historical TOS event logs and EDI 315/214 transaction milestones using goodness-of-fit hypothesis testing and Theil's inequality coefficient.
8. **Design** a real-time operational digital twin architecture that assimilates TOS state telemetry, resolves model divergence through state re-anchoring, and evaluates predictive lookahead dispatch policies.

---

## 3. The core mental model

At the heart of terminal operations engineering is a **closed, multi-tier, coupled cyclic queueing system** with finite intermediate buffers and spatial constraints. Entities (containers, railcars, air cargo ULDs) flow through a hierarchy of physical servers (quay cranes, internal transfer vehicles, yard stacking cranes, external trucks) whose operations are tightly synchronized.

```text
=============================================================================================================
                      DISCRETE-EVENT TERMINAL SIMULATION ARCHITECTURE
=============================================================================================================

 [ VESSEL SUBSYSTEM ]        [ HORIZONTAL TRANSPORT ]         [ YARD STACK SUBSYSTEM ]    [ LANDSIDE SUBSYSTEM ]
+---------------------+     +-----------------------+     +--------------------------+    +--------------------+
| Ship Bay Stowage    |     | AGV / ITV Fleet       |     | Yard Block & ASC / RTG   |    | Gate Complex / OCR |
| - Cell Guides       |     | - Guideway Graph      |     | - Landside Transfer Zone |    | - Ingate Portals   |
| - Hatch Covers      |     | - Battery Charging    |     | - Stacking Rules / Dwell |    | - Inspection / MT  |
+----------+----------+     +-----------+-----------+     +------------+-------------+    +---------+----------+
           │                            │                              │                            │
           ▼                            ▼                              ▼                            ▼
+---------------------+     +-----------------------+     +--------------------------+    +--------------------+
| Quay Crane (QC)     |◄───►| Quay Transfer Zone    |◄───►| Yard Transfer Point (TP) |◄───►| Over-the-Road (OTR)|
| - Hoist & Trolley   |     | - Buffer Cassette     |     | - ASC Gantry & Trolley   |    | Trucks & Rail Ramps|
| - Lashing Platform  |     | - Conflict Tiles      |     | - Rehandle Sequences     |    | - Train Shunting   |
+----------+----------+     +-----------+-----------+     +------------+-------------+    +---------+----------+
           ▲                            ▲                              ▲                            ▲
           │                            │                              │                            │
═══════════╪════════════════════════════╪══════════════════════════════╪════════════════════════════╪════════════
           │           SIMULATION KERNEL & SYNCHRONIZATION BUS         │                            │
═══════════╪════════════════════════════╪══════════════════════════════╪════════════════════════════╪════════════
           │                            │                              │                            │
+----------┴----------------------------┴------------------------------┴----------------------------┴----------+
| FUTURE EVENT LIST (FEL): Min-Heap Priority Queue (Timestamp t, Event Type, Entity ID, Target Resource ID)    |
| - Clock Advance Mechanism: t <- t_next (Variable time increments between discrete state transitions)         |
+--------------------------------------------------------------------------------------------------------------+
                                                │
                                                ▼
+--------------------------------------------------------------------------------------------------------------+
| SYSTEM STATE VECTOR S(t) = { QC_state(t), Vehicle_pos(t), Block_inventory(t), Gate_queue(t), Buffers(t) }  |
| - Physical Constraints: Non-crossing cranes, collision-free guideway tiles, finite buffer capacities         |
| - Operational Rules: Starvation policies, blocking protocols, dynamic vehicle dispatch heuristics           |
+--------------------------------------------------------------------------------------------------------------+
                                                │
                                                ▼
+--------------------------------------------------------------------------------------------------------------+
| TELEMETRY DIGITAL TWIN INTERFACE (SHADOW MODE)                                                                |
| - TOS Ingestion: EDI 315 / 214, CAM logs, PLC position feeds, Crane spreader lock/unlock events              |
| - State Estimator: Compares simulated S_sim(t) with physical S_phys(t); triggers state re-anchoring on drift|
| - Lookahead Engine: Fast-forwards simulation N hours into the future to predict congestion and bottlenecks   |
+--------------------------------------------------------------------------------------------------------------+
=============================================================================================================
```

### The Three World-Views of Simulation

Discrete-event simulation models can be formulated under three classical paradigms:
1. **Event-Scheduling:** The system dynamics are modeled strictly as discrete events that alter state variables and schedule future events on a global priority queue (the Future Event List, or FEL). This worldview offers the highest computational efficiency and minimal memory overhead, making it the preferred architecture for large-scale terminal digital twins running thousands of entities.
2. **Process-Interaction:** The model tracks the lifecycle of individual entities (such as a container or an AGV) as they flow through a sequence of processes, requesting, seizing, holding, and releasing shared resources. This worldview is intuitive and aligns naturally with visual simulation packages (e.g., AnyLogic, FlexSim, Simio).
3. **Activity-Scanning:** The simulation checks at each clock update whether state conditions enable activities to begin. While flexible for complex logic, its $O(N)$ scanning cost per time step renders it computationally impractical for large-scale container terminals without three-phase enhancements.

### The Cyclic Starvation-Blocking Dynamics

The quayside transfer zone exemplifies the operational coupling between adjacent handling stages:
- **Quay Crane Starvation:** The quay crane has unlashed and hoisted a container from the ship's cell guide, moved its trolley to the portal beam, but cannot lower the spreader because no horizontal transport vehicle (AGV or terminal tractor) is present at the quay transfer point. The crane enters an involuntary idle state ($S_{\text{QC}} = \text{STARVED}$), degrading net berth moves per hour.
- **Vehicle Blocking:** An AGV arrives at the yard block transfer point carrying an import container, but the Automated Stacking Crane (ASC) is busy performing a housekeeping rehandle or servicing a landside external truck. If the physical buffer rack (or ground transfer slot) is full, the AGV cannot discharge its load. It remains immobilized in the transfer zone ($S_{\text{AGV}} = \text{BLOCKED}$), preventing subsequent vehicles from accessing the block and reducing the effective transport fleet size.

Understanding this coupled queueing dynamic reveals why isolated optimizations fail: accelerating quay crane trolley speeds without expanding horizontal vehicle fleet capacity or yard transfer buffers merely shifts the bottleneck, transforming crane motion time into crane starvation time.

---

## 4. Definitions and vocabulary

| Term | Definition | Common Misuse | Standard Unit / Format |
|---|---|---|---|
| **Future Event List (FEL)** | A priority queue sorted chronologically containing all scheduled future system events $(t_e, \text{type}_e, \text{entity}_e)$. | Confusing the FEL with an active job queue or dispatch order list. | Min-heap ordered by epoch $t \in \mathbb{R}^+$ (seconds). |
| **Quay Crane (QC / STS)** | Ship-to-Shore rail-mounted gantry crane dedicated to hoisting containers between vessel cell guides and the quay apron. | Assuming QC cycle times are deterministic constants. | Moves per hour (GMPH, NMPH) / Seconds per lift cycle. |
| **Internal Transfer Vehicle (ITV)** | Generic classification for terminal tractors (bomb carts), AGVs, or autonomous straddle carriers cycling between quay and yard. | Conflating street-legal drayage trucks with internal terminal tractors. | Vehicle units ($N_{\text{veh}}$); speed in meters/second (m/s). |
| **Automated Stacking Crane (ASC)** | Rail-mounted gantry crane operating over a container yard block, typically dedicated to automated stacking and hand-off. | Equating ASCs with mobile rubber-tired gantry cranes (RTGs). | Moves per hour / Seconds per container pick/place cycle. |
| **Transfer Point (TP)** | A designated spatial interchange buffer where containers are exchanged between horizontal transfer vehicles and yard cranes. | Treating TPs as infinite-capacity queueing sinks. | Number of ground slots or physical buffer cassettes. |
| **Starvation** | A condition where a processing server (e.g., QC) is available to work but is idled because no input entity or transfer vehicle is present. | Confusing starvation with machine breakdown or planned maintenance. | Percentage of total operational time ($\%$) or hours ($h$). |
| **Blocking** | A condition where a server or vehicle has completed its operation but cannot release the entity because downstream buffers are full. | Treating blocked time as active travel or service time. | Percentage of operational time ($\%$) or seconds ($s$). |
| **Warm-Up Period ($T_0$)** | The initial transient interval of a simulation run during which statistics are contaminated by artificial starting conditions (e.g., empty queues). | Recording performance metrics from $t = 0$ without discarding warm-up data. | Time units (hours) or number of completed entity cycles. |
| **Initialization Bias** | Systematic error in steady-state parameter estimates caused by sampling from the transient start-up distribution rather than the limiting distribution. | Assuming that running a single long simulation completely eliminates initialization error. | Dimensionless bias $\mathbb{E}[\bar{Y}] - \mu_Y$. |
| **Welch's Moving Average** | A smoothing procedure applied across multiple independent replications to identify the point where the moving average stabilizes. | Applying smoothing across time within a single run rather than across replications. | Moving average window parameter $w \in \mathbb{Z}^+$. |
| **Method of Batch Means** | An output analysis technique that divides a single long stationary simulation run into contiguous, non-overlapping batches to compute confidence intervals. | Forming batches that are too small, violating the assumption of uncorrelated batch means. | Batch size $b \in \mathbb{Z}^+$, number of batches $k \in \mathbb{Z}^+$. |
| **Autocorrelation (Lag-$k$)** | The correlation between observations within the same time-series separated by $k$ time intervals: $\rho_k = \text{Cov}(X_t, X_{t+k}) / \text{Var}(X_t)$. | Assuming standard sample variance $s^2 / n$ applies to autocorrelated DES output data. | Dimensionless coefficient $\rho_k \in [-1, 1]$. |
| **Common Random Numbers (CRN)** | A variance reduction technique that synchronizes identical pseudo-random number streams across competing system designs to minimize comparison variance. | Using different random number seeds across alternatives and claiming observed differences are statistically significant. | Dimensionless variance reduction percentage ($\%$). |
| **Antithetic Variates** | A variance reduction technique generating pairs of simulation runs using complementary random numbers ($U_i$ and $1 - U_i$) to induce negative covariance. | Applying antithetic sampling to non-monotonic response surfaces, which inflates variance. | Dimensionless correlation coefficient. |
| **Terminal Operating System (TOS)** | Central database and transactional software platform managing container inventory, work orders, vessel stowage, and equipment dispatch. | Assuming the TOS performs continuous physical dynamic simulation. | Proprietary enterprise software (e.g., Navis N4, Tideworks). |
| **EDI 315** | ANSI ASC X12 Electronic Data Interchange transaction set for Ocean Shipment Status Details, providing standardized container milestone events. | Confusing shipment status EDI 315 with booking (EDI 301) or manifest (EDI 309). | EDI standard transaction text segment. |
| **EDI 214** | ANSI ASC X12 Transportation Carrier Shipment Status Message used by motor carriers and drayage truckers for gate and milestone tracking. | Assuming EDI 214 is used for vessel-side container operations. | EDI standard transaction text segment. |
| **Digital Twin (Shadow Mode)** | A simulation model coupled to real-time physical telemetry that continuously mirrors terminal state and tests forward-looking dispatch policies. | Calling an offline, uncalibrated CAD or DES model a "digital twin." | Latency of state synchronization (milliseconds to minutes). |
| **Shifted Lognormal Distribution** | A three-parameter continuous distribution ($\theta, \mu, \sigma$) where $X - \theta \sim \text{Lognormal}(\mu, \sigma^2)$, with location parameter $\theta$ setting the physical minimum time. | Using a two-parameter lognormal distribution that permits zero-duration crane cycles. | Parameters: $\theta \in \mathbb{R}^+$ (s), $\mu \in \mathbb{R}$ (log-s), $\sigma \in \mathbb{R}^+$ (log-s). |
| **Three-Parameter Weibull** | A continuous distribution characterized by location $\gamma$, scale $\alpha$, and shape $\beta$, frequently used for equipment failure and repair modeling. | Setting location $\gamma = 0$ when modeling operations with rigid physical minimum durations. | Parameters: $\gamma \in \mathbb{R}^+$ (s), $\alpha \in \mathbb{R}^+$ (s), $\beta \in \mathbb{R}^+$ (dimensionless). |
| **Deadlock / Circular Wait** | A state where two or more vehicles or cranes cannot proceed because each is waiting for a spatial tile or resource held by another. | Confusing temporary queue congestion with irreversible topological deadlock. | Discrete boolean state condition ($0$ or $1$). |
| **Spatial Reservation Tile** | A discretized polygonal zone of the terminal guideway network locked by a vehicle control system to guarantee collision-free travel. | Allowing multiple automated vehicles into a single unmanaged intersection tile. | Tile identifier; coordinate boundary $(x_1, y_1, x_2, y_2)$ in meters. |
| **Gross Moves Per Hour (GMPH)** | Total crane container lifts divided by total elapsed crane hours between first lift and last lift, including all operational and delay intervals. | Equating GMPH with net operational speed (NMPH), which excludes non-operational delays. | Lifts / Crane-Hour ($\text{moves}/h$). |
| **Net Moves Per Hour (NMPH)** | Total crane container lifts divided strictly by net operating hours, excluding recorded external delays (e.g., vessel list, twistlock jams). | Reporting NMPH as contractual terminal billing throughput without clarifying exclusions. | Lifts / Net Crane-Hour ($\text{moves}/h$). |
| **Berth Moves Per Hour (BMPH)** | Total container lifts handled across all assigned cranes divided by the total vessel berth working time (hours). | Multiplying single-crane GMPH by crane count without deducting inter-crane interference losses. | Total lifts / Berth-Hour ($\text{moves}/h$). |
| **Theil's Inequality Coefficient ($U$)** | A normalized metric bounding model simulation prediction error against historical empirical measurements between 0 (perfect fit) and 1 (worst fit). | Relying solely on $R^2$, which can mask severe mean bias and variance scaling errors. | Dimensionless coefficient $U \in [0, 1]$, decomposed into $U^M, U^S, U^C$. |

---

## 5. The mechanics, step by step

Developing and analyzing a high-fidelity terminal discrete-event simulation model requires a strict, eight-stage engineering procedure:

```text
+-----------------------+     +-----------------------+     +-----------------------+     +-----------------------+
| 1. System             |     | 2. Empirical Delay    |     | 3. Simulation Engine  |     | 4. Guideway & Dispatch|
| Decomposition & State |───► | Distribution Fitting  |───► | Core & Future Event   |───► | Routing Architecture  |
| Formulation           |     | & Goodness-of-Fit     |     | List (FEL) Mechanics  |     | (Deadlock-Free Logic) |
+-----------------------+     +-----------------------+     +-----------------------+     +-----------------------+
                                                                                                      │
                                                                                                      ▼
+-----------------------+     +-----------------------+     +-----------------------+     +-----------------------+
| 8. Real-Time Digital  |     | 7. Telemetry & EDI    |     | 6. Steady-State Output|     | 5. Warm-Up Period     |
| Twin Integration &    |◄─── | Historical Model      |◄─── | Analysis & Variance   |◄─── | Determination         |
| State Re-Anchoring    |     | Validation            |     | Reduction (Batch/CRN) |     | (Welch's Method)      |
+-----------------------+     +-----------------------+     +-----------------------+     +-----------------------+
```

---

### Step 1: System Decomposition and State Vector Formulation

Define the boundaries of the terminal system and construct the mathematical state vector $S(t)$. In an intermodal container terminal, the state vector at continuous time $t$ is partitioned into equipment, entity, and spatial sub-states:

$$S(t) = \Big( \mathbf{QC}(t), \mathbf{ITV}(t), \mathbf{ASC}(t), \mathbf{BUF}(t), \mathbf{GATE}(t), \mathbf{GRID}(t) \Big)$$

Where:
- $\mathbf{QC}(t) = \{QC_i(t)\}_{i=1}^{N_{qc}}$: For each crane $i$, $QC_i(t) \in \{\text{IDLE}, \text{HOISTING}, \text{TROLLEYING}, \text{LASHING}, \text{STARVED}, \text{BLOCKED}, \text{BREAKDOWN}\}$.
- $\mathbf{ITV}(t) = \{ITV_j(t)\}_{j=1}^{N_{itv}}$: For each vehicle $j$, $ITV_j(t) = \big( x_j(t), y_j(t), v_j(t), \text{status}_j, \text{container\_id}_j \big)$, where status includes $\text{TRAVEL\_EMPTY}, \text{WAIT\_QC}, \text{RECEIVING}, \text{TRAVEL\_LOADED}, \text{WAIT\_ASC}, \text{DISCHARGING}$.
- $\mathbf{ASC}(t) = \{ASC_k(t)\}_{k=1}^{N_{asc}}$: Stacking crane state, gantry location, trolley coordinate, and current move classification (import pick, export place, rehandle housekeeping).
- $\mathbf{BUF}(t) = \{b_m(t)\}_{m=1}^{M}$: Current container count occupying physical interchange buffer slots. Buffer capacity constraints require $0 \le b_m(t) \le B_m^{\max}$.
- $\mathbf{GRID}(t) = \{g_{u,v}(t)\}$: Binary reservation state of guideway spatial tiles, where $g_{u,v}(t) = 1$ if tile $(u,v)$ is occupied or reserved by vehicle $j$, preventing collisions.

---

### Step 2: Empirical Operational Delay Distribution Fitting

Terminal operational durations are never normally distributed. Physical handling times exhibit rigid lower bounds (dictated by maximum hoist and trolley mechanical acceleration limits) and pronounced right-side skewness (caused by lashing cage delays, twistlock jams, flipper alignment difficulties, and hatch cover unseating).

To prevent simulation artifacts such as zero or negative duration times, fit three-parameter continuous distributions with an explicit location parameter (shift) representing the physical minimum duration $\theta > 0$.

#### 1. Shifted Lognormal Distribution
If $X$ represents crane hoist-and-trolley cycle time, let $Y = X - \theta$ follow a lognormal distribution with parameters $\mu$ and $\sigma^2$:

$$f(x; \theta, \mu, \sigma) = \frac{1}{(x - \theta)\sigma \sqrt{2\pi}} \exp\left( -\frac{(\ln(x - \theta) - \mu)^2}{2\sigma^2} \right), \quad x > \theta$$

The theoretical mean and variance are:

$$\mathbb{E}[X] = \theta + \exp\left( \mu + \frac{\sigma^2}{2} \right)$$

$$\operatorname{Var}(X) = \exp(2\mu + \sigma^2)\left( \exp(\sigma^2) - 1 \right)$$

#### 2. Three-Parameter Weibull Distribution
Widely used for mechanical equipment repair intervals, horizontal vehicle transit times, and gate inspection delays:

$$f(x; \gamma, \alpha, \beta) = \frac{\beta}{\alpha} \left( \frac{x - \gamma}{\alpha} \right)^{\beta - 1} \exp\left( -\left( \frac{x - \gamma}{\alpha} \right)^\beta \right), \quad x \ge \gamma$$

Where $\gamma$ is the location parameter (minimum physical transit or inspection time), $\alpha > 0$ is the scale parameter, and $\beta > 0$ is the shape parameter. The mean is:

$$\mathbb{E}[X] = \gamma + \alpha \Gamma\left( 1 + \frac{1}{\beta} \right)$$

#### 3. Shifted Gamma Distribution
Used for multi-stage sequential processes such as gate OCR processing, physical driver inspection, and radiation scanning:

$$f(x; \gamma, k, \lambda) = \frac{\lambda^k (x - \gamma)^{k - 1} e^{-\lambda (x - \gamma)}}{\Gamma(k)}, \quad x > \gamma$$

#### Parameter Estimation via Profile Maximum Likelihood (MLE)
For the Shifted Lognormal distribution, given empirical observations $\{x_1, x_2, \dots, x_n\}$ and an unknown physical minimum lower bound $\theta < \min(x_i)$, the log-likelihood function is:

$$\ln L(\theta, \mu, \sigma^2) = -n \ln(\sigma \sqrt{2\pi}) - \sum_{i=1}^n \ln(x_i - \theta) - \frac{1}{2\sigma^2} \sum_{i=1}^n \left( \ln(x_i - \theta) - \mu \right)^2$$

Because standard gradient optimization exhibits a singularity as $\theta \to \min(x_i)$, estimate parameters via a one-dimensional profile log-likelihood search over $\theta \in [0, \min(x_i) - \epsilon]$. For any fixed candidate $\theta$, closed-form conditional MLE estimators exist:

$$\hat{\mu}(\theta) = \frac{1}{n} \sum_{i=1}^n \ln(x_i - \theta)$$

$$\hat{\sigma}^2(\theta) = \frac{1}{n} \sum_{i=1}^n \left( \ln(x_i - \theta) - \hat{\mu}(\theta) \right)^2$$

Substitute $\hat{\mu}(\theta)$ and $\hat{\sigma}^2(\theta)$ back into $\ln L$ to obtain the concentrated profile log-likelihood $\ln L_p(\theta)$, maximizing it numerically using golden section search.

Similarly, for the Three-Parameter Weibull distribution, the profile likelihood over location $\gamma < \min(x_i)$ yields the conditional shape equation solved via Newton-Raphson iteration:

$$\frac{1}{\beta} + \frac{1}{n} \sum_{i=1}^n \ln(x_i - \gamma) - \frac{\sum_{i=1}^n (x_i - \gamma)^\beta \ln(x_i - \gamma)}{\sum_{i=1}^n (x_i - \gamma)^\beta} = 0$$

#### Truncated Operational Distributions
Real-world handling equipment operates under hard physical and automated safety boundaries. Spreader cycle times cannot exceed the emergency timeout threshold $U_{\max}$ (typically 300 to 600 seconds, after which TOS alarms trigger manual supervisor intervention). Define the doubly truncated density function over support $[\theta, U_{\max}]$:

$$f_{\text{trunc}}(x; \boldsymbol{\psi}) = \frac{f(x; \boldsymbol{\psi})}{F(U_{\max}; \boldsymbol{\psi}) - F(\theta; \boldsymbol{\psi})}, \quad \theta \le x \le U_{\max}$$

Sampling from $f_{\text{trunc}}(x)$ via inverse transform sampling requires evaluating:

$$X = F^{-1}\left( F(\theta) + U \cdot \left[ F(U_{\max}) - F(\theta) \right] \right), \quad U \sim \mathcal{U}(0, 1)$$

#### Goodness-of-Fit Verification
Validate fitted parameters using the **Kolmogorov-Smirnov (K-S)** test and **Anderson-Darling (A-D)** test. Because terminal congestion is disproportionately driven by extreme operational delays, the Anderson-Darling test statistic $A^2$ is preferred over K-S because it gives heavier weight to distribution tails:

$$A^2 = -n - \sum_{i=1}^n \frac{2i - 1}{n} \left[ \ln F(X_{(i)}) + \ln(1 - F(X_{(n+1-i)})) \right]$$

Reject candidate distributions where the test $p$-value falls below $\alpha = 0.05$.

---

### Step 3: Simulation Engine Core and Future Event List (FEL) Mechanics

The discrete-event simulation engine advances through time using a variable-step clock governed by a priority queue (min-heap). 

1. **Initialization:** Set clock $t \leftarrow 0$. Initialize state vector $S(0)$. Populate the FEL with exogenous arrival events (e.g., vessel berthing, train arrival, external gate truck arrivals) and initial equipment ready states.
2. **Event Selection:** Extract the event $e^* = (t^*, \text{type}^*, \text{entity}^*, \text{resource}^*)$ from the top of the min-heap where $t^* = \min \{t_e \in \text{FEL}\}$.
3. **Clock Advance:** Advance simulation clock strictly to $t \leftarrow t^*$.
4. **State Transition:** Execute the operational logic associated with $\text{type}^*$:
   - Update equipment state (e.g., transition QC from $\text{HOISTING}$ to $\text{WAIT\_VEHICLE}$).
   - Evaluate conditional queue transitions: If an AGV is waiting in queue, bind the AGV to the QC, sample a handover duration $\Delta t_{\text{handover}}$ from the fitted shifted lognormal distribution, and schedule a future event:
     $$\text{FEL} \leftarrow \text{FEL} \cup \big( t + \Delta t_{\text{handover}}, \text{QC\_TRANSFER\_COMPLETED}, \text{entity}^*, \text{resource}^* \big)$$
   - If no AGV is available, increment the QC starvation counter and set $S_{\text{QC}} \leftarrow \text{STARVED}$.
5. **Statistics Collection:** Accumulate time-weighted state integrals (e.g., $\int_0^t N_{\text{queue}}(\tau) d\tau$) and discrete tallies.
6. **Termination Check:** If $t < T_{\text{end}}$ and the FEL is non-empty, return to Step 2. Otherwise, initiate output analysis.

---

### Step 4: Guideway Modeling, Vehicle Routing, and Deadlock Avoidance

In automated terminals, horizontal transport vehicles share intersecting bi-directional or uni-directional guideway networks. Modeling AGVs simply as speed-distance travel delays produces inaccurate throughput estimates because it ignores headway following limits, intersection turn delays, and spatial gridlock.

1. **Spatial Representation:** Represent the terminal apron and yard roadways as a directed graph $G = (V, E)$, where vertices $V$ represent lane intersections and transfer points, and edges $E$ represent travel lanes discretized into reservation tiles of length $L_{\text{tile}} = 15 \text{ to } 20 \text{ meters}$.
2. **Time-Space Routing (A\* Algorithm with Reservation):** When vehicle $j$ is dispatched from quay coordinate $(x_1, y_1)$ to yard block coordinate $(x_2, y_2)$, the vehicle path planner computes a time-space trajectory:
   $$\Pi_j = \big\{ (v_1, [t_{1,\text{enter}}, t_{1,\text{exit}}]), (v_2, [t_{2,\text{enter}}, t_{2,\text{exit}}]), \dots, (v_m, [t_{m,\text{enter}}, t_{m,\text{exit}}]) \big\}$$
   A tile $(v, t)$ can only be reserved if no conflicting vehicle trajectory overlaps that spatial envelope plus a safety headway buffer $\delta_{\text{safe}}$:
   $$[t_{k,\text{enter}} - \delta_{\text{safe}}, t_{k,\text{exit}} + \delta_{\text{safe}}] \cap [t'_{k,\text{enter}}, t'_{k,\text{exit}}] = \emptyset \quad \forall j' \neq j$$
3. **Kinematic Motion Profiles and Cornering Limits:**
   Vehicle transit is governed by trapezoidal velocity curves accounting for container mass and cornering centripetal acceleration limits:
   $$v(t) = \min\left( v_{\max}, \; \sqrt{2 a_{\max} d_{\text{remain}}}, \; \sqrt{\mu_{\text{friction}} \cdot g \cdot R_{\text{turn}}} \right)$$
   Where $R_{\text{turn}}$ is the curve radius, $g = 9.81 \text{ m/s}^2$, and $\mu_{\text{friction}} \approx 0.35$ is the dry pavement friction coefficient preventing container tipping or chassis pin shear.
4. **Deadlock Prevention (Banker's Algorithm & Graph Cycles):**
   A circular wait condition occurs when vehicle $A$ holds tile 1 and requests tile 2, while vehicle $B$ holds tile 2 and requests tile 1. The simulation kernel must enforce a deadlock avoidance policy:
   - *Resource Allocation Graph (RAG) Monitoring:* Maintain a directed graph of vehicle claims. Before granting a reservation tile, run Tarjan's strongly connected components algorithm to verify that granting the tile will not induce a directed cycle.
   - *Multi-Agent Conflict-Based Search (CBS):* For large AGV fleets ($N_{\text{veh}} \ge 50$), CBS decomposes the routing problem into a two-level search: a high-level search over a Conflict Tree (resolving vertex and edge collisions through spatial-temporal constraints) and low-level single-agent A* searches.
   - *Buffer Reservation Rule:* A vehicle cannot depart the quayside until a physical transfer slot or holding tile at the target yard block has been provisionally allocated, preventing gridlocks at block entry gates.
---

### Step 5: Warm-Up Period Determination (Welch's Method)

Intermodal terminal simulations are non-terminating, steady-state systems when evaluating long-term infrastructure capacity. Starting the simulation with an empty yard, empty berths, and idle cranes introduces severe initialization bias: early container cycles experience artificially zero queueing delays.

To determine the warm-up truncation point $T_0$, apply **Welch's Moving Average Procedure**:

1. Perform $R$ independent replications of the simulation, each running for length $M$ time steps (or move counts), using independent pseudo-random number seeds ($R \ge 10$).
2. Let $Y_{r, t}$ be the observed metric of interest (e.g., quay crane cycle time or AGV queue length) at discrete time step $t$ on replication $r$, where $t = 1, 2, \dots, M$ and $r = 1, 2, \dots, R$.
3. Compute the cross-replication average for each time step $t$:
   $$\bar{Y}_t = \frac{1}{R} \sum_{r=1}^R Y_{r, t}, \quad t = 1, 2, \dots, M$$
   The cross-replication average $\bar{Y}_t$ possesses expectation $\mathbb{E}[\bar{Y}_t] = \mathbb{E}[Y_{r, t}]$, eliminating random replication noise while preserving the underlying transient trend.
4. Calculate the moving average $\bar{Y}_t(w)$ with smoothing window parameter $w$ (where $w$ is an integer satisfying $w \le \lfloor M/4 \rfloor$):
   $$\bar{Y}_t(w) = \begin{cases}
   \frac{1}{2w + 1} \sum_{s = -w}^w \bar{Y}_{t + s}, & \text{for } t = w + 1, \dots, M - w \\
   \frac{1}{2t - 1} \sum_{s = -(t-1)}^{t-1} \bar{Y}_{t + s}, & \text{for } t = 1, \dots, w
   \end{cases}$$
5. Plot $\bar{Y}_t(w)$ against time $t$. Identify the truncation point $T_0$ where the smoothed curve ceases its initial directional ascent or descent and fluctuates randomly around a stable horizontal line.

#### Variance Properties and Window Parameter Selection
The theoretical variance of the smoothed moving average is:

$$\operatorname{Var}\left( \bar{Y}_t(w) \right) \approx \frac{\sigma_Y^2}{R(2w + 1)}$$

Selecting window parameter $w$ involves an engineering trade-off:
- If $w$ is too small, high-frequency stochastic noise obscures the underlying trend, preventing unambiguous identification of $T_0$.
- If $w$ is too large, the moving average filter smooths out the genuine knee of the curve, artificially shifting the apparent stabilization point to the right and causing unnecessary discarding of valid steady-state data.

#### Statistical Verification: Schruben's Initialization Bias Test
Complement visual inspection with Schruben's hypothesis test based on standardized time-series cumulative sums:

$$S_k = \sum_{i=1}^k (X_i - \bar{X}_N), \quad k = 1, 2, \dots, N$$

Under the null hypothesis $H_0$ of stationarity (no initialization bias), the standardized sequence $T_k = \frac{S_k}{\sigma \sqrt{N}}$ converges to a standard Brownian bridge on $[0, 1]$. If the test statistic exceeds critical thresholds, reject $H_0$ and increase truncation point $T_0$.

6. Delete all simulated observations occurring prior to $T_0$. Retain only data from $t \in [T_0 + 1, M]$ for steady-state analysis.

```text
Throughput / Cycle Metric Y_t(w)
      ^
      |                 Transient Warm-Up Phase         |        Steady-State Operational Regime
      |               (Severe Initialization Bias)      |     (Fluctuates stably around true mean mu)
      |                                                 |
      |                                         .-.     |      .-.           .-.         .-.
      |                                       .'   '.   |    .'   '.   .-. .'   '. .-. .'   '.
      |                                     .'       '.-+---'       '-'   '       ' '-'       '-.
      |                                   .'            |
      |                                 .'              |   Mean Level mu = 42.1 moves/hour
      |                             _.-'                |---------------------------------------
      |                         _.-'                    |
      |                     _.-'                        |
      |                 _.-'                            |
      |             _.-'                                |
      |         _.-'                                    |
      |     _.-'                                        |
  0.0 +----+--------------------------------------------+--------------------------------------->
     t = 0                                           t = T_0                                   t = M
                                               (Truncation Point)
```

---

### Step 6: Steady-State Output Analysis and Variance Reduction

Simulated terminal metrics (e.g., container transfer cycle times) are strongly **autocorrelated**: a long cycle time caused by a twistlock failure increases queue lengths, directly inflating the cycle times of subsequent moves. Standard classical statistics ($s^2 / \sqrt{n}$) assume independent and identically distributed (i.i.d.) observations; applying them to raw simulation output severely underestimates true standard errors, generating falsely narrow confidence intervals.

#### 1. Method of Batch Means
To compute statistically valid confidence intervals from a single long stationary simulation run of length $N$ moves (after discarding the warm-up period $T_0$):

1. Divide the $N$ observations $X_1, X_2, \dots, X_N$ into $k$ contiguous, non-overlapping batches, each containing $b$ observations, such that $N = k \cdot b$.
2. Compute the sample mean for each batch $j \in \{1, 2, \dots, k\}$:
   $$\bar{X}_j = \frac{1}{b} \sum_{i=1}^b X_{(j-1)b + i}$$
3. Test for independence between adjacent batch means by calculating the sample lag-1 autocorrelation $r_1$:
   $$r_1 = \frac{\sum_{j=1}^{k-1} (\bar{X}_j - \bar{X})(\bar{X}_{j+1} - \bar{X})}{\sum_{j=1}^k (\bar{X}_j - \bar{X})^2}, \quad \text{where } \bar{X} = \frac{1}{k} \sum_{j=1}^k \bar{X}_j$$
   If $|r_1| > z_{1 - \alpha/2} / \sqrt{k}$ (or if von Neumann's ratio test rejects randomness), batch size $b$ is insufficient. Double the batch size $b \leftarrow 2b$ and repeat until batch means are uncorrelated.
4. When batch means $\bar{X}_1, \dots, \bar{X}_k$ are approximately i.i.d. Gaussian, calculate the batch variance:
   $$S_{\bar{X}}^2 = \frac{1}{k - 1} \sum_{j=1}^k (\bar{X}_j - \bar{X})^2$$
5. Construct the $(1 - \alpha) \cdot 100\%$ confidence interval for the true operational parameter $\mu$:
   $$\bar{X} \pm t_{k - 1, 1 - \alpha/2} \frac{S_{\bar{X}}}{\sqrt{k}}$$

#### 2. Variance Reduction via Common Random Numbers (CRN)
When comparing two terminal design alternatives (e.g., Design $A$: 6 AGVs vs. Design $B$: 8 AGVs), do not draw independent random streams. Using **Common Random Numbers (CRN)** synchronizes identical pseudo-random number seeds across both runs for vessel arrival times, stowage bay profiles, crane hoist distributions, and equipment breakdown schedules.

Let $\bar{X}_A$ and $\bar{X}_B$ be the estimated average throughput under configurations $A$ and $B$. The variance of the estimated difference $D = \bar{X}_A - \bar{X}_B$ is:

$$\operatorname{Var}(D) = \operatorname{Var}(\bar{X}_A - \bar{X}_B) = \operatorname{Var}(\bar{X}_A) + \operatorname{Var}(\bar{X}_B) - 2 \operatorname{Cov}(\bar{X}_A, \bar{X}_B)$$

By subjecting both terminal models to the exact same sequence of synthetic operational challenges, high delays in Design $A$ correlate positively with high delays in Design $B$, ensuring $\operatorname{Cov}(\bar{X}_A, \bar{X}_B) > 0$. This positive covariance subtracts from total variance, yielding statistical significance with substantially fewer simulation runs.

Strict enforcement requires **Dedicated Random Streams**: dedicate Stream 1 strictly to vessel arrivals, Stream 2 to QC hoist times, Stream 3 to AGV travel times, and Stream 4 to ASC block cycles. Without dedicated streams, adding an extra vehicle desynchronizes the random draws, destroying the positive covariance.

#### 3. Control Variates (CV)
Let $X$ be the complex simulated terminal throughput metric and let $Z$ be an ancillary simulation output whose true expected value $\mathbb{E}[Z] = \mu_Z$ is known analytically (e.g., the theoretical unconstrained hoist duration of discharged boxes). The control variate estimator is:

$$\bar{X}_{\text{CV}} = \bar{X} - c^* (Z - \mu_Z), \quad \text{where } c^* = \frac{\operatorname{Cov}(X, Z)}{\operatorname{Var}(Z)}$$

The resulting variance reduction is governed by the squared correlation coefficient:

$$\operatorname{Var}(\bar{X}_{\text{CV}}) = \operatorname{Var}(\bar{X})(1 - \rho_{XZ}^2)$$

If $\rho_{XZ} \ge 0.80$, the variance of the throughput estimator is reduced by at least $64\%$.

#### 4. Regenerative Simulation Method
When an intermodal terminal exhibits recurrent empty-and-idle epochs (such as a multi-user river barge terminal or feeder berth where all berths and queues empty completely between scheduled ship windows), the simulation can be analyzed using the **Regenerative Method**:
- The timeline is divided into $n$ independent, identically distributed regenerative cycles of duration $\tau_j = T_j - T_{j-1}$.
- The steady-state throughput parameter is estimated as a ratio of cycle rewards: $\hat{\mu} = \frac{\bar{Y}}{\bar{\tau}} = \frac{\sum_{j=1}^n Y_j}{\sum_{j=1}^n \tau_j}$, where $Y_j$ is total moves completed in cycle $j$.
- Confidence intervals are constructed without batching or warm-up truncation using Taylor series linearization of ratio estimators.
---

### Step 7: Historical Telemetry and Model Validation

A terminal simulation model must be quantitatively validated against historical operational data extracted from the Terminal Operating System (TOS). Standard telemetry streams include:
- **EDI 315 Milestone Statuses:** Vessel arrival (`VA`), vessel discharge (`VD`), loading on railcar (`RC`), gate departure (`DL`).
- **Crane Automation / PLC Logs:** Crane optical positioning timestamps, spreader latch/unlatch sensor triggers, and container gross weight scale measurements.
- **Vehicle Telemetry (VMT / DGPS):** Coordinates sampled at 1 Hz, battery state of charge (SoC), and velocity profiles.

#### TOS Database Schema Telemetry Mapping
In enterprise Terminal Operating Systems (e.g., Navis N4, CyberLogitec OPUS, Tideworks Mainsail), telemetry data resides in relational tables mapped directly into discrete-event simulation entities and state transitions:

| TOS Relational Table | Captured Operational Attributes | Simulation Entity & State Mapping |
|---|---|---|
| `inv_unit` | Container ID, ISO size/type (20/40/45ft), gross weight (VGM), IMDG hazard class. | Instantiates entity attributes and physical dimensions in the simulation engine. |
| `inv_unit_move` | Move type (`DISCH`, `LOAD`, `YARD_SHIFT`), start/end timestamps, CHE equipment ID. | Calibrates empirical duration distributions for QC hoist, AGV travel, and ASC stacking. |
| `vsl_vessel_visit_details` | Inbound voyage, ETA, ETB (berthing), ETD, working berth meter coordinates. | Sets exogenous boundary events: vessel arrival, berth allocation, and departure deadlines. |
| `che_equipment` | Crane/AGV ID, operational status (`WORKING`, `MAINT`, `BREAKDOWN`), telemetry coordinates. | Calibrates machine failure rates (MTBF) and repair duration distributions (MTTR). |
| `road_truck_visit` | In-gate timestamp, trouble-ticket flags, out-gate timestamp, turn time. | Calibrates non-homogeneous Poisson arrival processes ($\lambda(t)$) for landside gate complexes. |
#### Validation Metrics and Theil's Inequality Coefficient
Compare simulated time-series outputs ($S_t$) against actual empirical measurements ($A_t$) over $T$ historical shifts. Compute **Theil's Inequality Coefficient ($U$)**:

$$U = \frac{\sqrt{\frac{1}{T} \sum_{t=1}^T (S_t - A_t)^2}}{\sqrt{\frac{1}{T} \sum_{t=1}^T S_t^2} + \sqrt{\frac{1}{T} \sum_{t=1}^T A_t^2}}$$

Where $U$ is bounded between $0$ (perfect forecast) and $1$ (maximum discrepancy). Decompose $U$ into three interpretable economic components:

$$U^M = \frac{(\bar{S} - \bar{A})^2}{\frac{1}{T} \sum_{t=1}^T (S_t - A_t)^2} \quad \text{(Bias Proportion: systematic under/over-estimation)}$$

$$U^S = \frac{(s_S - s_A)^2}{\frac{1}{T} \sum_{t=1}^T (S_t - A_t)^2} \quad \text{(Variance Proportion: unequal variation magnitude)}$$

$$U^C = \frac{2(1 - r) s_S s_A}{\frac{1}{T} \sum_{t=1}^T (S_t - A_t)^2} \quad \text{(Covariance Proportion: non-systematic tracking error)}$$

Where $U^M + U^S + U^C = 1$. An operationally valid simulation model must achieve $U < 0.15$, with the bias proportion $U^M < 0.05$ and variance proportion $U^S < 0.10$. A high bias proportion ($U^M$) indicates incorrect equipment baseline speeds, while a high variance proportion ($U^S$) indicates poorly fitted stochastic operational delay distributions.

---

### Step 8: Operational Digital Twin and Real-Time State Re-Anchoring

An operational digital twin couples the validated offline DES engine to the physical terminal via high-frequency telemetry buses (e.g., Apache Kafka or MQTT), operating in **Shadow Mode**.

```text
+-----------------------+                         +-----------------------+
| Physical Terminal     |                         | Operational DES       |
| - QCs, AGVs, ASCs     |                         | Digital Twin          |
| - TOS Event Bus (N4)  |                         | - State Vector S_sim  |
+-----------+-----------+                         +-----------+-----------+
            │                                                 ▲
            │ Telemetry Stream (1 Hz)                         │ Re-Anchoring Pulse
            │ (GPS, Spreader, EDI)                            │ (Adjust S_sim <- S_phys)
            ▼                                                 │
+-----------------------+                         +-----------+-----------+
| Event Ingestion &     |─── Filtered State ────► | State Divergence &    |
| Schema Normalizer     |    S_phys(t)            | Drift Estimator       |
+-----------------------+                         +-----------+-----------+
                                                              │
                                                  Divergence  ▼ Threshold Exceeded
                                                  +-----------------------+
                                                  | Fast-Forward Lookahead|
                                                  | Predictive Simulation |
                                                  | (t + 1h to t + 4h)    |
                                                  +-----------------------+
```

1. **State Ingestion & Synchronization:** At intervals $\Delta t_{\text{sync}} = 30 \text{ to } 60 \text{ seconds}$, the twin assimilates live container positions, crane status flags, and truck queue lengths from the TOS database.
2. **Drift Detection & State Re-Anchoring:** Because physical operations experience human interventions, micro-delays, and unmodeled sensor faults, the simulated state $S_{\text{sim}}(t)$ inevitably drifts from physical state $S_{\text{phys}}(t)$. If the Euclidean spatial drift or inventory mismatch exceeds threshold $\epsilon_{\text{drift}}$, the simulation executes a **state re-anchoring pulse**:
   - Force-update simulated container locations to physical coordinates reported by TOS work orders.
   - Re-initialize equipment state vectors.
   - Synchronize the Future Event List by cancelling superseded dispatch orders.
3. **Predictive Lookahead Execution:** Upon re-anchoring, the digital twin spawns $K$ parallel fast-forward simulation runs projecting terminal performance 2 to 4 hours into the future. By testing alternative dispatch policies (e.g., dynamic vehicle pooling versus dedicated quay crane assignment) under simulated forward uncertainty, the twin identifies emerging yard congestion and recommends preemptive re-allocation work orders to the live TOS dispatch controller.

---

## 6. Worked example

### Problem Statement: Quayside AGV Fleet Sizing Analysis

A marine container terminal operator is expanding an automated berth to service Ultra-Large Container Vessels (ULCVs) of 14,000 TEU capacity. The operations engineering team must determine the required Automated Guided Vehicle (AGV) fleet size to serve **2 Super-Post-Panamax Quay Cranes (QCs)** working concurrently on adjacent ship bays.

The terminal management is evaluating two capital configurations:
- **Configuration A:** 6 AGVs total (3.0 AGVs per QC).
- **Configuration B:** 8 AGVs total (4.0 AGVs per QC).

The vessel call requires discharging **2,400 import containers** (1,200 moves per crane). The engineering target is to achieve a combined berth throughput of at least **80 Berth Moves Per Hour (BMPH)** (40 Gross Moves Per Hour per crane) while keeping quay crane starvation below **5.0%**.

### Input Parameters and Operational Distributions

1. **Quay Crane Cycle Time ($T_{\text{QC}}$):** Unconstrained hoist, trolley, and cell guide transit time follows a **Shifted Lognormal Distribution** with shift parameter $\theta = 0$, location parameter $\mu = 4.40$ log-seconds, and shape parameter $\sigma = 0.15$ log-seconds.
   $$\mathbb{E}[T_{\text{QC}}] = \exp\left( 4.40 + \frac{0.15^2}{2} \right) = \exp(4.41125) = 82.373 \text{ seconds}$$
   The theoretical maximum speed per crane (with zero vehicle starvation) is:
   $$\text{Productivity}_{\max} = \frac{3,600 \text{ s/h}}{82.373 \text{ s/move}} = 43.704 \text{ GMPH}$$
   Combined theoretical maximum for 2 QCs = $87.408 \text{ BMPH}$.

2. **Terminal Apron and Yard Geometry:**
   - One-way travel distance between berth apron and assigned automated yard block follows a continuous uniform distribution: $D \sim \mathcal{U}(200, 500) \text{ meters}$, with mean distance $\bar{D} = 350.0 \text{ meters}$.
   - Vehicle kinematic parameters:
     - Loaded AGV velocity: $v_L = 5.0 \text{ m/s}$; Acceleration/deceleration: $a = 0.5 \text{ m/s}^2$.
     - Empty AGV velocity: $v_E = 6.0 \text{ m/s}$; Acceleration/deceleration: $a = 0.5 \text{ m/s}^2$.

3. **Kinematic Travel Time Calculations:**
   - *Loaded Transit:* 
     Acceleration distance: $d_{\text{acc}} = \frac{v_L^2}{2a} = \frac{5.0^2}{2(0.5)} = 25.0 \text{ m}$.
     Deceleration distance: $d_{\text{dec}} = 25.0 \text{ m}$.
     Constant speed cruise distance: $\bar{D} - (d_{\text{acc}} + d_{\text{dec}}) = 350.0 - 50.0 = 300.0 \text{ m}$.
     Acceleration time: $t_{\text{acc}} = \frac{v_L}{a} = \frac{5.0}{0.5} = 10.0 \text{ s}$; Deceleration time: $t_{\text{dec}} = 10.0 \text{ s}$.
     Cruise time: $t_{\text{cruise}} = \frac{300.0}{5.0} = 60.0 \text{ s}$.
     Total mean loaded travel time:
     $$\mathbb{E}[T_{\text{travel, loaded}}] = 10.0 + 60.0 + 10.0 = 80.0 \text{ seconds}$$
   - *Empty Transit:*
     Acceleration distance: $d_{\text{acc}} = \frac{6.0^2}{2(0.5)} = 36.0 \text{ m}$; Deceleration distance: $d_{\text{dec}} = 36.0 \text{ m}$.
     Cruise distance: $350.0 - 72.0 = 278.0 \text{ m}$.
     Accel/decel times: $t_{\text{acc}} = t_{\text{dec}} = \frac{6.0}{0.5} = 12.0 \text{ s}$.
     Cruise time: $t_{\text{cruise}} = \frac{278.0}{6.0} = 46.333 \text{ s}$.
     Total mean empty travel time:
     $$\mathbb{E}[T_{\text{travel, empty}}] = 12.0 + 46.333 + 12.0 = 70.333 \text{ seconds}$$

4. **Yard Block ASC Hand-Off Service Time ($T_{\text{ASC}}$):**
   Follows a **Three-Parameter Weibull Distribution** with location $\gamma = 10.0 \text{ s}$, scale $\alpha = 42.0 \text{ s}$, and shape $\beta = 2.5$.
   $$\mathbb{E}[T_{\text{ASC}}] = \gamma + \alpha \Gamma\left( 1 + \frac{1}{\beta} \right) = 10.0 + 42.0 \cdot \Gamma(1.4) = 10.0 + 42.0(0.88726) = 47.265 \text{ seconds}$$

5. **Spreader Alignment and Handshake Latency ($T_{\text{align}}$):** Constant $15.0 \text{ seconds}$.

6. **Total Unconstrained AGV Round-Trip Cycle Time:**
   $$T_{\text{loop}} = \mathbb{E}[T_{\text{travel, loaded}}] + \mathbb{E}[T_{\text{ASC}}] + \mathbb{E}[T_{\text{travel, empty}}] + T_{\text{align}} = 80.0 + 47.265 + 70.333 + 15.0 = 212.598 \text{ seconds}$$

### Analytical Queueing Lower Bound
Using deterministic closed queueing network approximations, the theoretical minimum number of AGVs per crane required to eliminate starvation is:

$$R^* = \frac{T_{\text{loop}}}{\mathbb{E}[T_{\text{QC}}]} = \frac{212.598 \text{ s}}{82.373 \text{ s}} = 2.581 \text{ AGVs per QC}$$

For 2 QCs, the baseline requirement is $2 \times 2.581 = 5.162 \text{ AGVs}$.
- Configuration A (6 AGVs) provides $6 / 2 = 3.0 \text{ AGVs/QC}$ (a $+16.2\%$ safety buffer).
- Configuration B (8 AGVs) provides $8 / 2 = 4.0 \text{ AGVs/QC}$ (a $+55.0\%$ safety buffer).

Because analytical models assume smooth flow and neglect stochastic variance, queueing at block transfer points, and vehicle dispatch delays, we execute a rigorous DES experiment.

---

### Simulation Execution: Step-by-Step Numerical Analysis

#### 1. Warm-Up Determination via Welch's Procedure
Ten pilot replications ($R = 10$) of 48 simulated hours were conducted from an empty-and-idle initial state. The matrix of hourly berth throughput $Y_{r, t}$ across the initial 12 operational hours is recorded below:

| Replication ($r$) | $t=1$ | $t=2$ | $t=3$ | $t=4$ | $t=5$ | $t=6$ | $t=7$ | $t=8$ | $t=9$ | $t=10$ | $t=11$ | $t=12$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rep 1** | 28 | 41 | 52 | 61 | 67 | 70 | 72 | 73 | 73 | 74 | 73 | 73 |
| **Rep 2** | 27 | 44 | 55 | 63 | 69 | 72 | 73 | 74 | 74 | 73 | 74 | 74 |
| **Rep 3** | 29 | 42 | 54 | 62 | 68 | 71 | 72 | 73 | 73 | 74 | 73 | 73 |
| **Rep 4** | 30 | 40 | 51 | 60 | 66 | 69 | 71 | 72 | 74 | 73 | 74 | 73 |
| **Rep 5** | 26 | 43 | 56 | 64 | 70 | 73 | 74 | 74 | 73 | 75 | 73 | 74 |
| **Rep 6** | 28 | 45 | 53 | 61 | 68 | 71 | 73 | 73 | 74 | 73 | 74 | 73 |
| **Rep 7** | 31 | 42 | 52 | 62 | 67 | 70 | 72 | 72 | 73 | 74 | 73 | 74 |
| **Rep 8** | 27 | 41 | 54 | 63 | 69 | 72 | 73 | 74 | 74 | 73 | 74 | 73 |
| **Rep 9** | 29 | 44 | 55 | 64 | 68 | 71 | 72 | 73 | 73 | 73 | 74 | 74 |
| **Rep 10** | 29 | 43 | 53 | 61 | 69 | 72 | 73 | 73 | 74 | 73 | 73 | 73 |
| **Mean $\bar{Y}_t$** | **28.4** | **42.5** | **53.5** | **62.1** | **68.1** | **71.1** | **72.5** | **73.1** | **73.5** | **73.5** | **73.6** | **73.4** |
| **Smoothed $\bar{Y}_t(2)$** | 28.4 | 38.1 | 46.7 | 59.5 | 65.5 | 69.4 | 71.7 | 72.8 | 73.2 | 73.4 | 73.5 | 73.5 |

Between $t = 1 \text{ h}$ and $t = 8 \text{ h}$, the smoothed moving average ascends steeply from $28.4 \text{ BMPH}$ to $72.8 \text{ BMPH}$ as the empty quayside fills with inbound container flows. From $t = 10 \text{ h}$ onward, $\bar{Y}_t(2)$ stabilizes completely, fluctuating within $[73.2, 73.5] \text{ BMPH}$.

**Conclusion:** The warm-up truncation point is conservatively set to $T_0 = 12 \text{ hours}$. All statistics recorded prior to $t = 12 \text{ h}$ are excised from steady-state evaluations.
#### 2. Steady-State Batch Means Analysis (Configuration A: 6 AGVs)
A long steady-state simulation of $1,280 \text{ post-warmup operational hours}$ was executed. The output was partitioned into $k = 40 \text{ batches}$ of size $b = 32 \text{ hours}$ ($N = 1,280 \text{ h}$).

- Lag-1 autocorrelation test on batch means:
  $$r_1 = \frac{\sum_{j=1}^{39} (\bar{X}_j - \bar{X})(\bar{X}_{j+1} - \bar{X})}{\sum_{j=1}^{40} (\bar{X}_j - \bar{X})^2} = 0.0412$$
  Critical threshold at $\alpha = 0.05$: $\frac{z_{0.975}}{\sqrt{k}} = \frac{1.960}{\sqrt{40}} = 0.3099$.
  Because $|r_1| = 0.0412 < 0.3099$, the batch means are statistically independent.

- Sample batch statistics for Berth Throughput ($X$ in BMPH):
  $$\bar{X}_A = \frac{1}{40} \sum_{j=1}^{40} \bar{X}_j = 73.420 \text{ BMPH}$$
  $$S_A^2 = \frac{1}{39} \sum_{j=1}^{40} (\bar{X}_j - 73.420)^2 = 8.162 \quad \implies \quad S_A = 2.8569 \text{ BMPH}$$
  $$\text{Standard Error } \operatorname{SE}(\bar{X}_A) = \frac{S_A}{\sqrt{40}} = \frac{2.8569}{6.3245} = 0.4517 \text{ BMPH}$$

- 95% Confidence Interval ($t_{39, 0.975} = 2.0227$):
  $$\text{CI}_{0.95}(\bar{X}_A) = 73.420 \pm (2.0227)(0.4517) = 73.420 \pm 0.914 \text{ BMPH} = [72.506, 74.334] \text{ BMPH}$$
- Average QC Starvation: **$16.08\%$ of total crane time**.

#### 3. Steady-State Batch Means Analysis (Configuration B: 8 AGVs)
Under Configuration B (8 AGVs), the simulation was executed over identical batch intervals:
- Sample batch statistics for Berth Throughput:
  $$\bar{X}_B = 84.180 \text{ BMPH}$$
  $$S_B^2 = 4.321 \quad \implies \quad S_B = 2.0787 \text{ BMPH}$$
  $$\operatorname{SE}(\bar{X}_B) = \frac{2.0787}{\sqrt{40}} = 0.3287 \text{ BMPH}$$

- 95% Confidence Interval:
  $$\text{CI}_{0.95}(\bar{X}_B) = 84.180 \pm (2.0227)(0.3287) = 84.180 \pm 0.665 \text{ BMPH} = [83.515, 84.845] \text{ BMPH}$$
- Average QC Starvation: **$3.79\%$ of total crane time**.

#### 4. Hypothesis Testing with Common Random Numbers (CRN)
We compare Configuration B against Configuration A. 

- **Case 1: Independent Replications (Without CRN):**
  $$\operatorname{Var}(\bar{X}_B - \bar{X}_A) = \frac{S_A^2}{40} + \frac{S_B^2}{40} = \frac{8.162 + 4.321}{40} = \frac{12.483}{40} = 0.3121$$
  $$\operatorname{SE}_{\text{indep}} = \sqrt{0.3121} = 0.5587 \text{ BMPH}$$

- **Case 2: Synchronized Common Random Numbers (With CRN):**
  By synchronizing identical container bay pickup sequences and yard block drop locations, the sample covariance between batch means is positive:
  $$\operatorname{Cov}(\bar{X}_A, \bar{X}_B) = 5.241$$
  $$\operatorname{Var}(\bar{X}_B - \bar{X}_A)_{\text{CRN}} = \frac{S_A^2 + S_B^2 - 2 \operatorname{Cov}(\bar{X}_A, \bar{X}_B)}{40} = \frac{8.162 + 4.321 - 2(5.241)}{40} = \frac{2.001}{40} = 0.0500$$
  $$\operatorname{SE}_{\text{CRN}} = \sqrt{0.0500} = 0.2236 \text{ BMPH}$$

- **Variance Reduction Achieved:**
  $$\text{Variance Reduction} = 1 - \frac{\operatorname{Var}_{\text{CRN}}}{\operatorname{Var}_{\text{indep}}} = 1 - \frac{0.0500}{0.3121} = 1 - 0.1602 = 83.98\%$$

- **Difference Confidence Interval:**
  $$\Delta = \bar{X}_B - \bar{X}_A = 84.180 - 73.420 = +10.760 \text{ BMPH}$$
  $$\text{CI}_{0.95}(\Delta) = 10.760 \pm (2.0227)(0.2236) = 10.760 \pm 0.452 \text{ BMPH} = [+10.308, +11.212] \text{ BMPH}$$

### Operational and Economic Assessment
1. **Target Compliance:**
   - Configuration A (6 AGVs) achieves $73.42 \text{ BMPH}$, failing the $80.0 \text{ BMPH}$ operational threshold, with a $16.08\%$ crane starvation rate that breaches the $5.0\%$ ceiling.
   - Configuration B (8 AGVs) achieves $84.18 \text{ BMPH}$ ($42.09 \text{ GMPH}$ per crane), exceeding the target while reducing crane starvation to $3.79\%$.
2. **Turnaround Time Impact:**
   - To complete 2,400 moves:
     - Config A: $\frac{2,400 \text{ moves}}{73.42 \text{ BMPH}} = 32.689 \text{ hours}$.
     - Config B: $\frac{2,400 \text{ moves}}{84.18 \text{ BMPH}} = 28.510 \text{ hours}$.
   - Port turnaround time reduction: $32.689 - 28.510 = 4.179 \text{ hours saved per vessel call}$.
3. **Multi-Stakeholder Economic Valuation:**
   Over 150 scheduled ULCV vessel calls per year, saving $4.179 \text{ hours}$ per call creates financial value across three distinct stakeholders:
   - **Terminal Berth Capacity Value:** $150 \text{ calls} \times 4.179 \text{ h} = 626.85 \text{ berth-hours saved}$. At a marginal berth commercial opportunity value of $\$3,500/\text{hour}$, annual terminal capacity value equals:
     $$\text{Value}_{\text{berth}} = 626.85 \times \$3,500 = \$2,193,975/\text{year}$$
   - **Carrier Bunker Speed-Up Avoidance:** To recover a 4.18-hour port delay on a 3,500 nm sea leg, an ocean carrier must accelerate from 17.0 knots to 18.8 knots. Applying the cubic fuel power law ($P \propto v^3$), this sprint burns an extra 18.0 metric tons of Very Low Sulfur Fuel Oil (VLSFO) per voyage. At $\$620/\text{metric ton}$:
     $$\text{Value}_{\text{fuel}} = 150 \times 18.0 \times \$620 = \$1,674,000/\text{year}$$
   - **Stevedoring Labor Overtime Savings:** Eliminating 4.18 hours of crane gang operating hours across 2 assigned crane gangs at an overtime rate of $\$450/\text{gang-hour}$:
     $$\text{Value}_{\text{labor}} = 150 \times 4.179 \times 2 \times \$450 = \$564,165/\text{year}$$
   - **Total Annual Economic Benefit:**
     $$\text{Total Value} = \$2,193,975 + \$1,674,000 + \$564,165 = \$4,432,140/\text{year}$$
   - **Capital Investment and Payback:**
     Purchasing 2 additional battery AGVs at $\$450,000/\text{unit}$ requires $\$900,000 \text{ CAPEX}$.
     $$\text{Simple Payback Period} = \frac{\$900,000}{\$4,432,140/\text{year}} = 0.203 \text{ years} \quad (\approx 2.44 \text{ months})$$
     The simulation definitively demonstrates that Configuration B represents the economically superior capital allocation.
---

## 7. Code, query, or calculation

The following self-contained Python program implements an event-scheduling discrete-event simulation model of a coupled Quay Crane, AGV transport, and Automated Stacking Crane yard system. It calculates crane productivity, starvation percentage, and prints operational statistics.

```python
import heapq, random, math

class TerminalDES:
    def __init__(self, n_agvs=6, seed=42):
        random.seed(seed)
        self.n_agvs, self.fel, self.time = n_agvs, [], 0.0
        self.qc_moves, self.starve_time, self.last_starve = 0, 0.0, 0.0
        self.agv_idle, self.qc_waiting, self.asc_busy = n_agvs, False, False
        self.yard_queue = []

    def sched(self, delay, event):
        heapq.heappush(self.fel, (self.time + delay, event))

    def run(self, max_time=86400.0):
        self.sched(0.0, "QC_READY")
        while self.fel and self.time < max_time:
            self.time, event = heapq.heappop(self.fel)
            if event == "QC_READY":
                if self.agv_idle > 0:
                    self.agv_idle -= 1
                    lift = random.lognormvariate(4.40, 0.15)  # E[X] ~ 82.4s
                    self.sched(lift, "QC_TRANSFER_DONE")
                else:
                    self.qc_waiting, self.last_starve = True, self.time
            elif event == "QC_TRANSFER_DONE":
                self.qc_moves += 1
                travel_out = random.weibullvariate(80.0, 3.5) # Loaded travel
                self.sched(travel_out, "AGV_AT_YARD")
                self.sched(0.0, "QC_READY")
            elif event == "AGV_AT_YARD":
                if not self.asc_busy:
                    self.asc_busy = True
                    self.sched(random.expovariate(1.0 / 47.0), "ASC_DONE")
                else:
                    self.yard_queue.append(self.time)
            elif event == "ASC_DONE":
                self.asc_busy = False
                if self.yard_queue:
                    self.yard_queue.pop(0)
                    self.asc_busy = True
                    self.sched(random.expovariate(1.0 / 47.0), "ASC_DONE")
                travel_back = random.weibullvariate(70.0, 4.0) # Empty return
                self.sched(travel_back, "AGV_AT_QUAY")
            elif event == "AGV_AT_QUAY":
                if self.qc_waiting:
                    self.qc_waiting = False
                    self.starve_time += self.time - self.last_starve
                    lift = random.lognormvariate(4.40, 0.15)
                    self.sched(lift, "QC_TRANSFER_DONE")
                else:
                    self.agv_idle += 1
        gmph = (self.qc_moves / self.time) * 3600.0
        starve_pct = (self.starve_time / self.time) * 100.0
        return self.qc_moves, gmph, starve_pct

print(f"{'AGVs':<6}{'Moves':<8}{'GMPH':<10}{'Starvation %':<14}")
for n in [3, 4, 5, 6, 8]:
    moves, gmph, st = TerminalDES(n_agvs=n, seed=101).run(max_time=86400.0)
    print(f"{n:<6}{moves:<8}{gmph:<10.2f}{st:<14.2f}")
```

### Assumptions and Expected Output
- **Assumptions:** Single QC coupled to a dedicated AGV fleet and one ASC block buffer; non-preemptive queueing; 24-hour simulation horizon (86,400 seconds); pseudorandom seed synchronized.
- **Expected Console Output:**
  ```text
  AGVs  Moves   GMPH      Starvation %  
  3     884     36.83     15.71         
  4     1002    41.75     4.23          
  5     1028    42.83     1.24          
  6     1035    43.12     0.48          
  8     1039    43.29     0.11          
  ```

---

## 8. Data, standards, and regulatory context

Simulation modeling of terminal facilities operates within strict telecommunications, transport data interchange, equipment safety, and engineering standardization frameworks:

### Electronic Data Interchange (EDI) and Telemetry Standards
1. **ANSI ASC X12 Standards (North American Maritime & Intermodal):**
   - **Transaction Set 315 (Ocean Shipment Status Details):** Standardized format reporting container milestone event codes:
     - `VA`: Vessel Arrival at Port Berth.
     - `VD`: Vessel Discharge Completed (Spreader release at quay apron).
     - `AL`: Container Loaded onto Vessel.
     - `AG`: Container In-Gated at Terminal Gate Portal.
     - `AR`: Container Out-Gated from Terminal Facility.
     - `RC`: Container Loaded onto Rail Intermodal Well-Car.
     - `DL`: Final Container Delivery to Consignee.
   - **Transaction Set 214 (Transportation Carrier Shipment Status):** Standardized format used by motor drayage carriers to report gate arrival, driver queue check-in, and chassis interchange.

#### Raw EDI Telemetry Stream Decomposition
To calibrate empirical distributions and drive the digital twin, raw EDI transmission streams are parsed into timestamped discrete event tuples:

```text
ISA*00*          *00*          *ZZ*TERMINALOPERATOR*ZZ*CARRIERLINE    *260514*0830*U*00401*000004128*0*P*>~
GS*QO*TERMINALOPERATOR*CARRIERLINE*20260514*0830*4128*X*004010~
ST*315*0001~
B4*VD*20260514*0832*MSKU9182734*UN*L*22G1*APMT*USLAX~
N9*BM*987654321~
Q2*9431808*US*EVER GIVEN*041E*98234*1~
R4*D*UN*USLAX*PORT OF LOS ANGELES*US*CA~
SE*7*0001~
GE*1*4128~
IEA*1*000004128~
```

The ingestion parser extracts:
1. `B4-01`: Event Code (`VD` = Vessel Discharge).
2. `B4-02/03`: Timestamp (`2026-05-14 08:32:00 UTC`).
3. `B4-04`: Equipment Identifier (`MSKU9182734`).
4. `B4-07`: ISO Equipment Size/Type (`22G1` = 20ft standard dry box).
5. `Q2-03`: Lloyds Vessel IMO / Name (`9431808`, `EVER GIVEN`).

This tuple updates the simulated container entity from `STOWED_ON_VESSEL` to `TRANSFERRED_TO_APRON`, logging the physical handling interval to update the running shifted lognormal hoist distribution.
   - `BAPLIE` (Bayplan / Stowage Plan Message): Transmits ship stowage coordinates (Bay-Row-Tier) used to seed the initial entity state in quayside simulations.
   - `COPRAR` (Container Discharge/Loading Order): Carrier work order specifying move lists.
   - `COARRI` (Container Discharge/Loading Report): Confirmation message acknowledging completed container lifts, providing telemetry for empirical delay distribution calibration.
   - `CODECO` (Gate-In / Gate-Out Report): Generates the time-series boundary data required to calibrate gate complex arrival processes.

### Simulation and Interoperability Frameworks
- **IEEE 1516 (High Level Architecture - HLA):** Standard governing distributed simulation architecture. Used when coupling independent sub-simulations—such as linking a quayside hydrodynamic vessel motion simulation to a terminal logistics DES engine via a Run-Time Infrastructure (RTI) software bus.
- **ISO 23247 (Digital Twin Framework for Manufacturing and Logistics):** Defines reference architectures, data modeling primitives, and information synchronization protocols between physical assets, edge devices, and digital twin simulation kernels.

### Safety, Labor, and Spatial Regulations
- **OSHA 29 CFR Part 1917 (Marine Terminals):** Mandates minimum physical clearances between moving mechanical handling equipment and personnel walkways. In automated terminal simulation, OSHA regulations dictate the minimum dimensions of spatial safety zones surrounding automated cranes and AGV travel corridors.
- **IMO MSC/Circ. 1353 (Revised Guidelines for Cargo Securing Manuals):** Regulates lashing platform ergonomics and safety sequences on container ships. Constrains simulation precedence graphs by mandating that twistlocks must be removed or attached at designated lashing platforms rather than suspended above active vehicle travel lanes.

---

## 9. Industry practice and case evidence

### Industrial Deployments and Evidence Ledger

| Port / Facility | Technology & Tools | Operational Scale | Documented Outcome | Evidence Grade |
|---|---|---|---|:---:|
| **Port of Rotterdam** (APM Terminals MVII & ECT Delta) | Discrete-event simulation (TBA PROTEUS / Chesscon) | Automated berth, 100+ battery AGVs, dual-trolley STS cranes, Lift-AGV cassettes. | Validated Lift-AGV decoupling: decoupling ground cassettes reduced QC starvation by 18% compared to direct-drop AGVs. | A |
| **Port of Long Beach** (Middle Harbor / LBCT) | Multi-tier terminal simulation and TOS digital twin | $1.5B investment; 3.3M TEU annual capacity; 70+ automated ASC blocks. | Simulation identified yard gantry clashing under dual-carrier train arrivals, shifting dispatch from fixed-crane blocks to dynamic dual-ASC handover zones. Achieved >32 BMPH. | A |
| **Port of Singapore** (PSA Tuas Mega Port) | Real-time multi-agent digital twin and predictive DES | 65M TEU target capacity; 200+ Autonomous Container Transporters (ACTs). | Implemented spatial tile reservation to eliminate AGV gridlock at high densities; dynamic predictive lookahead reduced vehicle travel delays by 14%. | B |
| **BNSF Logistics Park Chicago** (Elwood, IL) | Rail intermodal terminal simulation (FlexSim / RailTrack) | Multi-track wide-span RMG intermodal railyard; >1.0M annual rail lifts. | Simulation demonstrated that doubling hostler tractor fleet without expanding trackside staging buffers caused yard road gridlock, reducing train turnaround by 8%. | B |
| **Cathay Pacific Cargo Terminal** (Hong Kong HKIA) | Material handling system DES (Siemens Dematic) | 2.6M tonnes annual air cargo capacity; multi-level Elevating Transfer Vehicles (ETVs). | Simulated ULD workstation dwell distributions, determining optimal ETV buffer replenishment cutoffs to prevent missed air cargo flight departures. | C |

### Real-World Engineering Insights

1. **The Decoupling Power of Lift-AGVs:**
   At Rotterdam Maasvlakte II, initial simulation studies evaluated standard flatbed AGVs against **Lift-AGVs** equipped with hydraulic lifting platforms capable of depositing containers onto passive steel cassettes in the yard buffer zone. Simulation proved that while Lift-AGVs cost 22% more per vehicle, they completely decoupled the horizontal transport vehicle from the yard gantry crane. The AGV drops its container onto the cassette in 12 seconds and immediately departs, eliminating the 40–80 second delay spent waiting for an ASC to arrive. The simulation demonstrated that the terminal achieved identical throughput with 28% fewer total vehicles.

2. **The Pitfall of Dispatching to "Nearest Available Vehicle":**
   At Long Beach Container Terminal, early dispatch rules assigned import transfer orders to the nearest idle AGV. When tested under heavy simulation workloads, this myopic greedy heuristic induced spatial vehicle clustering: dozens of vehicles accumulated at the central berth, while distant quay cranes starved for empty chassis. Simulation enabled engineers to develop and validate **Lookahead Hungarian Bipartite Matching**, which balances current travel distance against anticipated future crane release timestamps over a 15-minute moving horizon.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Underlying Physical Mechanism | Operational Consequence | Engineering Mitigation |
|---|---|---|---|---|
| **Circular Guideway Deadlock** | Average AGV speed drops to 0.0 m/s; vehicle queue lengths grow without work progress. | Vehicles on intersecting guideway loops simultaneously hold the tile requested by the adjacent vehicle (circular wait). | Complete freeze of horizontal transport; quayside cranes starve; manual human override required. | Implement Banker's Algorithm or Tarjan cycle detection within the tile reservation engine; enforce one-way loop topologies. |
| **Buffer Cassette Saturation** | Yard transfer queue spikes; AGVs remain loaded at yard block entrance lanes. | ASC handling rate drops below quay crane delivery rate; interchange buffer capacity $B_{\max}$ is exhausted. | Upstream vehicle blocking; AGVs cannot discharge loads, backing up onto main terminal travel thoroughfares. | Dynamic throttling: TOS slows QC dispatch or temporarily diverts import flows to alternative yard blocks with lower dwell. |
| **Fictitious Throughput Overestimation** | Simulation projects 42 GMPH; physical terminal field deployment achieves only 28 GMPH. | Model uses deterministic or symmetric normal distributions, ignoring heavy-tailed operational delays and breakdown distributions. | Capital project fails to meet design throughput targets; bondholder covenants and carrier SLAs breached. | Fit empirical shifted lognormal or Weibull distributions calibrated against historical TOS EDI 315/214 telemetry logs. |
| **Severe Initialization Bias** | Simulation metrics show high sensitivity to total run length; short runs report unrealistically high capacity. | Statistics are gathered from $t = 0$ while queues and buffers are empty, under-representing steady-state congestion. | Mis-sizing equipment fleet; underestimating necessary yard storage capacity and vehicle counts. | Apply Welch's moving average procedure; discard all observations prior to truncation point $T_0$. |
| **Underestimated Confidence Intervals** | Simulation runs with identical parameters report non-overlapping confidence bounds across different seeds. | Standard errors computed using i.i.d. assumptions ($s^2/n$) on highly autocorrelated output data. | Erroneous engineering design decisions based on false statistical confidence. | Employ the Method of Batch Means; verify batch independence via lag-1 autocorrelation testing ($r_1 \approx 0$). |
| **Spurious Optimization via Unsynchronized Seeds** | Alternative $B$ appears superior to Alternative $A$, but advantage evaporates when re-tested under different random seeds. | Comparing system configurations using independent random streams rather than synchronizing random variations. | Capitalizing on random noise rather than genuine algorithmic or physical architectural superiority. | Enforce Common Random Numbers (CRN); isolate dedicated pseudo-random streams for each operational process. |
| **Digital Twin State Drift** | Simulated container inventory diverges from physical yard inventory within 2 hours of synchronization. | Unmodeled physical events (e.g., manual re-handles, lost RFID reads, unrecorded mechanical slow-downs) compound over time. | Forward predictive lookahead simulations recommend invalid or physically impossible dispatch orders. | Implement periodic state re-anchoring pulses; filter incoming telemetry via discrete-state Bayesian estimation. |
| **Crane Interference Clashing** | Simulated gantry crane moves occur simultaneously at adjacent bays without delay. | Simulation model omits physical non-crossing and minimum safety distance constraints ($d_{\min} \ge 25 \text{ m}$) between cranes on shared rails. | Overestimating multi-crane productivity; real cranes clash and block each other along the vessel hull. | Embed spatial rail non-crossing equations and inter-crane safety buffers directly into the crane scheduling engine. |
| **Rehandle Explosion in High-Density Yard** | ASC moves per container hand-off exceed 2.8; landside truck turn times exceed 90 minutes. | Yard storage occupancy exceeds 80% of nominal capacity; random stacking forces deep reshuffling to retrieve targeted boxes. | Landside gate failure; severe drayage driver queues extending onto municipal highways; demurrage penalties. | Model dynamic yard segregation rules; cap simulated yard density at 75% utilization; incorporate pre-marshalling logic. |
| **Battery Depletion Cascades** | AGVs stall on active guideways; charging station queues exceed physical connector slots. | Rapid continuous duty cycles deplete vehicle batteries faster than opportunistic charging intervals allow. | Cascading vehicle fleet attrition; quay cranes starve as available fleet shrinks below critical threshold $R^*$. | Embed state-of-charge (SoC) thermodynamic battery decay equations and priority charging dispatch rules into the DES kernel. |
| **Tandem-Lift Spreader Centerline Jam** | Spreader micro-adjustment alarms trigger repeatedly; crane hoist speed drops by 60%. | Tandem spreader picking dual 20ft containers encounters severe weight asymmetry (>10 tonnes differential), inducing transverse list in cell guides. | Extreme hoist delays; risk of structural guide deformation; manual stevedore intervention required. | Formulate dynamic weight-balancing heuristics in the QCSP engine; restrict tandem lifting to matched weight-bracket pairs. |
| **TOS Telemetry Wireless Blackout** | Digital twin flags missing heartbeat; vehicle positions freeze in simulation visualization. | High-density 5-high container stacks cause 5.8 GHz RF multipath reflection and packet dropout to vehicle on-board units. | Digital twin state divergence; predictive dispatch orders issued based on stale physical coordinates. | Implement edge-buffer replay on vehicle PLCs; switch to dead-reckoning state estimation during RF dropouts. |
---

## 11. Assumptions and limits

| Assumption | Engineering Rationale | Vulnerability / Failure Boundary | Verification & Stress Testing Protocol |
|---|---|---|---|
| **Stationary Arrival Processes** | Vessel calls and external drayage truck arrivals modeled as homogeneous Poisson or renewal processes. | Severe weather, typhoon vessel bunching, or gate labor strikes produce non-stationary, clustered burst arrivals. | Test model using non-homogeneous Poisson processes (NHPP) with time-varying arrival rates $\lambda(t)$; run storm recovery scenarios. |
| **Constant Equipment Kinematics** | Constant acceleration $a$ and velocity $v$ parameters applied across all vehicle moves. | Roadway surface water, tire wear, wind resistance against high-cube empty containers, and battery degradation alter kinematics. | Calibrate acceleration parameters against 1 Hz DGPS vehicle telemetry logs under wet and windy operational conditions. |
| **Infinite Component Reliability** | Mean Time Between Failures (MTBF) assumed sufficiently large to neglect mechanical breakdown during short runs. | Unplanned spreader twistlock failures or hydraulic line ruptures instantly paralyze crane bays. | Incorporate explicit Weibull failure and lognormal repair distributions into long-horizon capacity simulations. |
| **Zero Human Operator Variance** | Semi-automated equipment cycle times modeled without behavioral differences between human operators. | Gang shift changes, fatigue during night shifts, and operator skill disparities create up to 25% throughput variance. | Segment historical cycle data by operator shift (e.g., Day Shift vs. Night Shift); fit stratified operational distributions. |
| **Static Guideway Topologies** | Network graph vertices and edges remain fixed throughout the entire simulation run. | Physical terminal maintenance, surface repaving, or crane boom breakdowns force lane closures and re-routing. | Implement dynamic graph edge weight modification within the A* routing engine to simulate unexpected lane blockages. |
| **Unconstrained Vessel Stability** | Quay cranes are assumed free to discharge any assigned container without altering ship trim, heel, or hull bending. | Improper discharge sequences induce excessive vessel heel (list), causing containers to jam in vertical hull cell guides. | Couple the discrete-event simulation scheduler to a naval architectural hydrostatic stability calculator. |
| **Perfect TOS Telemetry State Knowledge** | Digital twin assumes sensor data (RFID, OCR, DGPS) is 100% complete, uncorrupted, and latency-free. | Missing sensor packets, multipath GPS drift under crane booms, and delayed EDI messages introduce state ambiguity. | Introduce synthetic noise, dropout probabilities ($p_{\text{loss}} = 0.02$), and timestamp jitter into the digital twin ingestion bus. |
| **Homogeneous Container Dimensions** | Containers treated as uniform 40-foot equivalent units (FEU) for simplified spatial reservation. | Handling 20-foot twin-lifts, 45-foot pallet-wides, over-height flat racks, and ISO tank containers alters cycle mechanics. | Model explicit container type attributes (ISO 6346 size/type codes); apply conditional twin-lift and out-of-gauge cycle penalties. |
| **Linear Scalability of Equipment Fleets** | Throughput assumed to increase monotonically as additional AGVs or tractors are added to the fleet. | Guideway congestion, intersection gridlock, and transfer buffer blocking induce diminishing marginal returns and negative returns. | Sweep fleet size $N_{\text{veh}}$ across a wide range ($2 \text{ to } 12 \text{ per QC}$) to identify the congestion inflection point. |
| **Twistlock Fatigue Immunity** | Spreader mechanical twistlocks assumed to suffer zero fatigue-induced pin lock failures during standard operations. | Cyclic dynamic fatigue loads cause twistlock shear pin jamming, requiring manual torch cutting on lashing platforms. | Incorporate explicit Weibull failure distributions ($MTBF \approx 12,000 \text{ cycles}$) into spreader component reliability logic. |
| **Uncorrelated Wind and Tidal Hydrology Dynamics** | Wind shear velocity and coastal tidal surge water levels modeled as independent stochastic variates. | Storm surge depressions and high onshore wind fronts are hydrodynamically coupled, simultaneously degrading crane stability and under-keel clearance. | Model meteorological boundaries via joint copula distributions (Gumbel copula for tail-dependent surge and wind extremes). |
---

## 12. Exercises

### Exercise 1: Statistical Distribution Fitting and Parameter Estimation
A container terminal extracts 1,200 observations of quay crane spreader positioning times during discharge operations. Field measurements indicate that the absolute minimum physical cycle duration is $\theta = 35.0 \text{ seconds}$. The sample mean of the raw observations is $\bar{X} = 78.50 \text{ seconds}$, and the sample standard deviation is $S = 22.10 \text{ seconds}$.
- **Deliverable:** Fit a **Shifted Lognormal Distribution** to these empirical observations. Calculate the exact numerical values for the underlying parameters $\mu$ and $\sigma^2$ of the shifted variable $Y = X - \theta$.
- **Acceptance Criteria:** Show all intermediate equations for log-transformed moments; express $\mu$ and $\sigma$ to four decimal places.
- **Estimated Time:** 25 minutes.

### Exercise 2: Welch's Moving Average and Truncation Point Identification
A simulation model of an intermodal rail terminal is executed across $R = 5$ independent replications. The observed number of railcars shunted per hour over the first 8 hours of each run is recorded in the matrix below:

| Replication | $t = 1$ | $t = 2$ | $t = 3$ | $t = 4$ | $t = 5$ | $t = 6$ | $t = 7$ | $t = 8$ |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Rep 1** | 12 | 18 | 24 | 31 | 35 | 36 | 35 | 36 |
| **Rep 2** | 10 | 16 | 26 | 29 | 33 | 34 | 36 | 35 |
| **Rep 3** | 14 | 19 | 23 | 30 | 36 | 37 | 34 | 37 |
| **Rep 4** | 11 | 17 | 25 | 32 | 34 | 35 | 35 | 36 |
| **Rep 5** | 13 | 20 | 27 | 28 | 37 | 38 | 35 | 36 |

- **Deliverable:** 
  1. Compute the cross-replication mean $\bar{Y}_t$ for each time step $t \in \{1, 2, \dots, 8\}$.
  2. Calculate Welch's moving average $\bar{Y}_t(w)$ using a window parameter $w = 1$.
  3. Identify the warm-up truncation point $T_0$ where the system achieves steady-state shunting throughput.
- **Acceptance Criteria:** Tabulate all intermediate cross-replication and smoothed moving average values; state the explicit engineering rationale for selecting $T_0$.
- **Estimated Time:** 30 minutes.

### Exercise 3: Batch Means Output Analysis and Autocorrelation Testing
A steady-state container terminal simulation generates 1,000 hours of post-warmup berth throughput data. The engineer divides the sequence into $k = 10 \text{ batches}$ of $b = 100 \text{ hours}$ each. The resulting batch means (in Berth Moves Per Hour) are:
$$\mathbf{\bar{X}} = [78.2, 81.4, 79.6, 80.8, 82.1, 79.9, 81.2, 80.5, 78.9, 81.5]$$
- **Deliverable:**
  1. Compute the grand sample mean $\bar{X}$ and the sample batch variance $S_{\bar{X}}^2$.
  2. Calculate the sample lag-1 autocorrelation coefficient $r_1$ across the batch means.
  3. Determine whether the batches are statistically independent at the $\alpha = 0.05$ significance level ($z_{0.975} = 1.960$).
  4. Construct the $95\%$ confidence interval for the true mean berth moves per hour using the Student's $t$-distribution ($t_{9, 0.975} = 2.262$).
- **Acceptance Criteria:** Full arithmetic steps shown for $r_1$ and confidence interval bounds; state whether batch size $b = 100$ is adequate.
- **Estimated Time:** 35 minutes.

### Exercise 4: Common Random Numbers (CRN) Variance Reduction Calculation
Two alternative dispatching algorithms (Heuristic A vs. Heuristic B) for automated guided vehicles are compared across 5 simulation runs. 
- Under independent random numbers, the sample variance of the difference was $\operatorname{Var}(\bar{X}_A - \bar{X}_B) = 14.80$.
- Under Common Random Numbers (CRN), the sample statistics across the 5 paired runs yielded:
  - $\bar{X}_A = 36.4 \text{ GMPH}$, $S_A^2 = 8.50$.
  - $\bar{X}_B = 41.2 \text{ GMPH}$, $S_B^2 = 9.20$.
  - Sample covariance between paired runs: $\operatorname{Cov}(\bar{X}_A, \bar{X}_B) = 7.60$.
- **Deliverable:**
  1. Calculate the variance of the difference $\operatorname{Var}(\bar{X}_A - \bar{X}_B)_{\text{CRN}}$ under Common Random Numbers.
  2. Compute the exact percentage variance reduction achieved by synchronizing the random number streams.
  3. Calculate the $95\%$ confidence interval on the throughput gain $\Delta = \bar{X}_B - \bar{X}_A$ using $t_{4, 0.975} = 2.776$.
- **Acceptance Criteria:** Show variance covariance substitution; confirm whether the throughput advantage of Heuristic B is statistically significant.
- **Estimated Time:** 25 minutes.

### Exercise 5: Closed Queueing Network vs. Discrete-Event Simulation Capacity Sizing
An air cargo terminal operates 4 Elevating Transfer Vehicles (ETVs) transferring Unit Load Devices (ULDs) between an automated high-bay storage rack and 12 build-up/breakdown workstations.
- Workstation service time: $\mathbb{E}[S_{\text{work}}] = 24.0 \text{ minutes}$ (exponentially distributed).
- ETV travel and placement cycle time: $\mathbb{E}[S_{\text{ETV}}] = 3.5 \text{ minutes}$ (deterministic).
- **Deliverable:** 
  1. Formulate the closed queueing network equations (Buzen's convolution algorithm or Mean Value Analysis) to calculate maximum theoretical workstation throughput.
  2. Critique why this closed queueing formulation systematically overestimates actual terminal throughput compared to a discrete-event simulation model that incorporates physical ETV rail interference, acceleration limits, and workstation buffer blocking.
- **Acceptance Criteria:** Provide analytical throughput upper bound; write a 250-word engineering critique detailing three specific spatial/stochastic failure modes captured only by DES.
- **Estimated Time:** 40 minutes.

### Exercise 6: Real-Time Digital Twin State Re-Anchoring Architecture
Design the software and algorithmic architecture for an operational digital twin of a 4-berth container terminal.
- **Deliverable:** 
  1. Draw an architectural diagram illustrating the interfaces between the Terminal Operating System (TOS database / Kafka event bus), the telemetry ingestion pipeline, the state divergence estimator, and the fast-forward predictive DES engine.
  2. Formulate the mathematical objective function and decision criteria used to trigger an automated state re-anchoring pulse when physical and simulated container inventories drift.
  3. Specify the fallback protocol when physical sensor telemetry is lost for $> 15 \text{ minutes}$.
- **Acceptance Criteria:** Architectural diagram showing data flows, protocols (MQTT/Kafka/gRPC), and specific mathematical drift metrics (e.g., Euclidean position error, discrete inventory mismatch proportion).
- **Estimated Time:** 45 minutes.

### Exercise 7: Guideway Deadlock Detection and Avoidance Formulation
Consider a terminal intersection where two automated guided vehicle travel lanes cross at right angles.
- **Deliverable:**
  1. Formulate a graph-theoretic Banker's Algorithm or Petri Net representation that guarantees mutual exclusion and prevents circular wait deadlocks across $N$ vehicles requesting access to the 4 spatial reservation tiles forming the intersection.
  2. Write pseudo-code for the tile reservation request function `RequestTiles(vehicle_id, trajectory_tiles)`.
- **Acceptance Criteria:** Pseudo-code must execute in polynomial time $O(|V| + |E|)$; mathematically demonstrate that no circular wait state can materialize.
- **Estimated Time:** 35 minutes.

---

## 13. Self-check answers

### Solution to Exercise 1: Statistical Distribution Fitting
1. The empirical variable is $X$ with minimum bound $\theta = 35.0 \text{ s}$. The shifted variable is $Y = X - \theta = X - 35.0$.
2. Compute the mean and variance of $Y$:
   $$\mathbb{E}[Y] = \mathbb{E}[X] - \theta = 78.50 - 35.00 = 43.50 \text{ seconds}$$
   $$\operatorname{Var}(Y) = \operatorname{Var}(X) = S^2 = 22.10^2 = 488.41 \text{ seconds}^2$$
3. For a lognormal distribution $Y \sim \operatorname{LN}(\mu, \sigma^2)$, the theoretical moments are:
   $$\mathbb{E}[Y] = \exp\left( \mu + \frac{\sigma^2}{2} \right) = 43.50$$
   $$\operatorname{Var}(Y) = \exp(2\mu + \sigma^2) \left( \exp(\sigma^2) - 1 \right) = (\mathbb{E}[Y])^2 \left( \exp(\sigma^2) - 1 \right) = 488.41$$
4. Solve for $\sigma^2$:
   $$\exp(\sigma^2) - 1 = \frac{\operatorname{Var}(Y)}{(\mathbb{E}[Y])^2} = \frac{488.41}{43.50^2} = \frac{488.41}{1892.25} = 0.25811$$
   $$\exp(\sigma^2) = 1.25811 \quad \implies \quad \sigma^2 = \ln(1.25811) = 0.22961$$
   $$\sigma = \sqrt{0.22961} = 0.4792$$
5. Solve for $\mu$:
   $$\mu = \ln(\mathbb{E}[Y]) - \frac{\sigma^2}{2} = \ln(43.50) - \frac{0.22961}{2} = 3.7728 - 0.1148 = 3.6580$$
6. **Final Fitted Parameters:** $\theta = 35.0000 \text{ s}$, $\mu = 3.6580 \text{ log-seconds}$, $\sigma = 0.4792 \text{ log-seconds}$ ($\sigma^2 = 0.2296$).

---

### Solution to Exercise 2: Welch's Moving Average
1. **Cross-Replication Averages ($\bar{Y}_t = \frac{1}{5} \sum_{r=1}^5 Y_{r, t}$):**
   - $\bar{Y}_1 = \frac{12 + 10 + 14 + 11 + 13}{5} = \frac{60}{5} = 12.00$
   - $\bar{Y}_2 = \frac{18 + 16 + 19 + 17 + 20}{5} = \frac{90}{5} = 18.00$
   - $\bar{Y}_3 = \frac{24 + 26 + 23 + 25 + 27}{5} = \frac{125}{5} = 25.00$
   - $\bar{Y}_4 = \frac{31 + 29 + 30 + 32 + 28}{5} = \frac{150}{5} = 30.00$
   - $\bar{Y}_5 = \frac{35 + 33 + 36 + 34 + 37}{5} = \frac{175}{5} = 35.00$
   - $\bar{Y}_6 = \frac{36 + 34 + 37 + 35 + 38}{5} = \frac{180}{5} = 36.00$
   - $\bar{Y}_7 = \frac{35 + 36 + 34 + 35 + 35}{5} = \frac{175}{5} = 35.00$
   - $\bar{Y}_8 = \frac{36 + 35 + 37 + 36 + 36}{5} = \frac{180}{5} = 36.00$
2. **Welch's Moving Average ($w = 1$):**
   - For $t = 1$: $\bar{Y}_1(1) = \bar{Y}_1 = 12.00$.
   - For $t = 2$: $\bar{Y}_2(1) = \frac{\bar{Y}_1 + \bar{Y}_2 + \bar{Y}_3}{3} = \frac{12.0 + 18.0 + 25.0}{3} = \frac{55.0}{3} = 18.33$.
   - For $t = 3$: $\bar{Y}_3(1) = \frac{18.0 + 25.0 + 30.0}{3} = \frac{73.0}{3} = 24.33$.
   - For $t = 4$: $\bar{Y}_4(1) = \frac{25.0 + 30.0 + 35.0}{3} = \frac{90.0}{3} = 30.00$.
   - For $t = 5$: $\bar{Y}_5(1) = \frac{30.0 + 35.0 + 36.0}{3} = \frac{101.0}{3} = 33.67$.
   - For $t = 6$: $\bar{Y}_6(1) = \frac{35.0 + 36.0 + 35.0}{3} = \frac{106.0}{3} = 35.33$.
   - For $t = 7$: $\bar{Y}_7(1) = \frac{36.0 + 35.0 + 36.0}{3} = \frac{107.0}{3} = 35.67$.
   - For $t = 8$: $\bar{Y}_8(1) = \frac{\bar{Y}_7 + \bar{Y}_8}{2} = \frac{35.0 + 36.0}{2} = 35.50$.
3. **Truncation Point $T_0$ Selection:**
   Between $t = 1$ and $t = 5$, the moving average increases steeply from $12.00$ to $33.67$. At $t = 6$, the curve stabilizes at $35.33$ and remains essentially flat through $t = 8$ ($35.67, 35.50$).
   Therefore, the warm-up period is **$T_0 = 5 \text{ hours}$**. Steady-state analysis begins at $t = 6 \text{ hours}$, with an estimated steady-state mean throughput of $\mu \approx 35.6 \text{ railcars/hour}$.

---

### Solution to Exercise 3: Batch Means Output Analysis
1. **Grand Mean ($\bar{X}$):**
   $$\sum_{j=1}^{10} \bar{X}_j = 78.2 + 81.4 + 79.6 + 80.8 + 82.1 + 79.9 + 81.2 + 80.5 + 78.9 + 81.5 = 804.10$$
   $$\bar{X} = \frac{804.10}{10} = 80.410 \text{ BMPH}$$
2. **Sample Variance of Batch Means ($S_{\bar{X}}^2$):**
   Deviations $(\bar{X}_j - \bar{X})$:
   $[-2.21, +0.99, -0.81, +0.39, +1.69, -0.51, +0.79, +0.09, -1.51, +1.09]$.
   Squared deviations:
   $[4.8841, 0.9801, 0.6561, 0.1521, 2.8561, 0.2601, 0.6241, 0.0081, 2.2801, 1.1881]$.
   Sum of squared deviations $= 13.8890$.
   $$S_{\bar{X}}^2 = \frac{13.8890}{10 - 1} = \frac{13.8890}{9} = 1.5432 \quad \implies \quad S_{\bar{X}} = \sqrt{1.5432} = 1.2423 \text{ BMPH}$$
3. **Lag-1 Autocorrelation ($r_1$):**
   Numerator $\sum_{j=1}^9 (\bar{X}_j - \bar{X})(\bar{X}_{j+1} - \bar{X})$:
   $= (-2.21)(0.99) + (0.99)(-0.81) + (-0.81)(0.39) + (0.39)(1.69) + (1.69)(-0.51) + (-0.51)(0.79) + (0.79)(0.09) + (0.09)(-1.51) + (-1.51)(1.09)$
   $= -2.1879 - 0.8019 - 0.3159 + 0.6591 - 0.8619 - 0.4029 + 0.0711 - 0.1359 - 1.6459 = -5.6223$.
   Denominator $= \sum_{j=1}^{10} (\bar{X}_j - \bar{X})^2 = 13.8890$.
   $$r_1 = \frac{-5.6223}{13.8890} = -0.4048$$
   Critical threshold at $\alpha = 0.05$: $\frac{1.960}{\sqrt{10}} = \frac{1.960}{3.1623} = 0.6198$.
   Because $|r_1| = 0.4048 < 0.6198$, we cannot reject the null hypothesis of independence. The batch means are sufficiently uncorrelated.
4. **95% Confidence Interval ($t_{9, 0.975} = 2.262$):**
   $$\operatorname{SE}(\bar{X}) = \frac{S_{\bar{X}}}{\sqrt{10}} = \frac{1.2423}{3.1623} = 0.3928 \text{ BMPH}$$
   $$\text{CI}_{0.95} = 80.410 \pm (2.262)(0.3928) = 80.410 \pm 0.889 \text{ BMPH} = [79.521, 81.299] \text{ BMPH}$$

---

### Solution to Exercise 4: Common Random Numbers (CRN)
1. **Variance of the Difference under CRN:**
   $$\operatorname{Var}(\bar{X}_A - \bar{X}_B)_{\text{CRN}} = \operatorname{Var}(\bar{X}_A) + \operatorname{Var}(\bar{X}_B) - 2 \operatorname{Cov}(\bar{X}_A, \bar{X}_B) = 8.50 + 9.20 - 2(7.60) = 17.70 - 15.20 = 2.50$$
2. **Percentage Variance Reduction:**
   $$\text{Variance Reduction} = 1 - \frac{\operatorname{Var}_{\text{CRN}}}{\operatorname{Var}_{\text{indep}}} = 1 - \frac{2.50}{14.80} = 1 - 0.1689 = 83.11\%$$
3. **95% Confidence Interval for $\Delta = \bar{X}_B - \bar{X}_A$ ($n = 5$, $t_{4, 0.975} = 2.776$):**
   Point estimate: $\hat{\Delta} = 41.2 - 36.4 = +4.80 \text{ GMPH}$.
   Sample standard error of difference: $\operatorname{SE}(\hat{\Delta}) = \sqrt{\frac{\operatorname{Var}_{\text{CRN}}}{n}} = \sqrt{\frac{2.50}{5}} = \sqrt{0.50} = 0.7071 \text{ GMPH}$.
   $$\text{CI}_{0.95}(\Delta) = 4.80 \pm (2.776)(0.7071) = 4.80 \pm 1.963 \text{ GMPH} = [+2.837, +6.763] \text{ GMPH}$$
   Because the entire $95\%$ confidence interval lies strictly above zero ($[+2.84, +6.76]$), Heuristic B produces a statistically significant throughput improvement over Heuristic A.

---

### Rubric for Design Exercises (Exercises 5, 6, and 7)

- **Exercise 5 (Closed Queueing vs. DES Capacity):**
  - *Full Credit:* Correct mathematical setup of Buzen's algorithm / MVA yielding theoretical upper bound; identifies three distinct structural breakdown mechanisms: (1) physical spatial interference on shared tracks, (2) acceleration/deceleration kinematics, (3) buffer blocking causing upstream workstation starvation.
  - *Partial Credit:* Calculates MVA but provides generic critique of queuing theory without naming terminal-specific mechanisms.
  - *No Credit:* Confuses open and closed queueing formulations.

- **Exercise 6 (Real-Time Digital Twin Architecture):**
  - *Full Credit:* Clean architectural diagram showing end-to-end telemetry flow (TOS $\rightarrow$ Kafka $\rightarrow$ State Divergence Estimator $\rightarrow$ DES Kernel); explicit mathematical drift metric (e.g., $L_2$ spatial norm $+ \sum |\Delta \text{inventory}| > \epsilon$); specifies robust fallback protocol (freezing dispatch recommendations and falling back to reactive FIFO dispatch upon telemetry loss).
  - *Partial Credit:* Diagram lacks specific communication protocols or omits mathematical formulation of the drift trigger.
  - *No Credit:* Submits a generic simulation flow chart without real-time state re-anchoring mechanics.

- **Exercise 7 (Guideway Deadlock Avoidance):**
  - *Full Credit:* Mathematically sound Banker's algorithm or Petri Net formulation proving absence of circular wait states; polynomial-time pseudo-code enforcing atomic multi-tile reservation or strict monotonic tile numbering.
  - *Partial Credit:* Algorithm detects deadlocks reactively (after vehicles freeze) rather than preventing deadlocks proactively.
  - *No Credit:* Proposes simple traffic-light heuristics that fail under saturated traffic conditions.

---

## 14. Interview questions

### Question 1: Handling Initialization Bias in Terminal Simulation
**Interviewer:** *"We are commissioning a new automated container terminal simulation. A junior engineer ran the model for 24 simulated hours starting from an empty terminal and reported an average crane productivity of 38 gross moves per hour. Do you accept this figure? How would you validate or correct it?"*

- **What a strong answer contains:** 
  - Immediately rejects the metric due to **initialization bias**. Starting with an empty yard and empty quay buffers means early crane cycles encounter zero queueing, zero vehicle contention, and zero yard gantry delays, producing an invalid estimate.
  - Explains the execution of **Welch's Moving Average Procedure**: run 10+ independent replications, compute cross-replication means for each hourly time step, apply a smoothing window $w$, and plot the resulting trajectory.
  - Defines the truncation point $T_0$ where the transient response stabilizes into steady-state operation.
  - Explains that all data collected prior to $T_0$ must be excised from the statistical accounting, computing steady-state metrics exclusively over $[T_0, T_{\text{end}}]$.
- **What a weak answer reveals:** 
  - Suggests simply running the simulation for 48 hours instead of 24 without discarding starting data.
  - Believes that averaging multiple independent runs eliminates initialization bias (it does not; it merely yields a more precise estimate of an inherently biased transient curve).

---

### Question 2: Batch Means vs. Independent Replications
**Interviewer:** *"When conducting steady-state output analysis for a port digital twin, what governs your decision between running the Method of Batch Means versus running multiple Independent Replications?"*

- **What a strong answer contains:**
  - Evaluates the fundamental engineering trade-off: **Warm-up Waste versus Autocorrelation Risk**.
  - In *Independent Replications (IR)*, each replication must discard the warm-up period $T_0$. If $T_0 = 12 \text{ hours}$ and 20 replications are performed, $240 \text{ hours}$ of simulated computation are completely discarded. However, IR guarantees that replication averages are strictly independent and identically distributed (i.i.d.).
  - In the *Method of Batch Means*, the warm-up period is paid exactly once at the beginning of a single very long simulation run. However, the engineer must rigorously prove that batch size $b$ is sufficiently large to extinguish internal autocorrelation between adjacent batches, typically verified using lag-1 autocorrelation testing ($r_1 \approx 0$) or von Neumann's ratio test.
  - Recommends Batch Means when simulation execution is computationally expensive and warm-up truncation consumes a large percentage of available compute.
- **What a weak answer reveals:**
  - Unaware that simulation observations are autocorrelated; claims that standard sample variance formulas apply directly to raw cycle time outputs.
  - Confuses batch size with replication count.

---

### Question 3: Common Random Numbers (CRN) Mechanics
**Interviewer:** *"You are comparing two dynamic dispatching algorithms for an automated guided vehicle fleet. How do you design your simulation experiment to prove Algorithm B is truly superior to Algorithm A using the minimum number of computer runs?"*

- **What a strong answer contains:**
  - Specifies the application of **Common Random Numbers (CRN)** as a variance reduction technique.
  - Formulates the variance of the difference: $\operatorname{Var}(\bar{X}_B - \bar{X}_A) = \operatorname{Var}(\bar{X}_B) + \operatorname{Var}(\bar{X}_A) - 2\operatorname{Cov}(\bar{X}_A, \bar{X}_B)$.
  - Explains that synchronizing pseudo-random number streams across both experimental configurations ensures that both algorithms face the exact same vessel bay stowage profiles, container dwell times, crane micro-delays, and truck gate arrival bursts.
  - Notes that this synchrony induces a strong positive covariance ($\operatorname{Cov} > 0$), dramatically reducing the standard error of the estimated difference and enabling statistical significance with far fewer runs.
  - Highlights the operational necessity of **stream synchronization**: ensuring that random numbers assigned to crane hoist delays do not cross over into vehicle travel distributions.
- **What a weak answer reveals:**
  - Suggests running Monte Carlo trials with completely different random seeds for each algorithm and taking the difference of the means.
  - Lacks understanding of the covariance term in variance reduction.

---

### Question 4: Deadlock Prevention in Automated Terminals
**Interviewer:** *"In our simulation of an automated container terminal with 80 AGVs, the model periodically freezes: the simulation clock advances, but zero containers are moved. What is happening mathematically, and how do you resolve it in the simulation architecture?"*

- **What a strong answer contains:**
  - Identifies the phenomenon as **circular wait deadlock** on the terminal guideway network or transfer buffers.
  - Formulates the four Coffman conditions: (1) Mutual exclusion, (2) Hold and wait, (3) No preemption, and (4) Circular wait.
  - Explains how deadlock arises on discretized spatial networks: Vehicle 1 occupies Tile A and requests Tile B; Vehicle 2 occupies Tile B and requests Tile A.
  - Proposes proactive avoidance algorithms:
    - *Time-Space Tile Reservation:* Vehicles reserve a time-indexed sequence of spatial tiles $[t_{\text{enter}}, t_{\text{exit}}]$ prior to departure; conflicting reservations are rejected during path planning.
    - *Banker's Algorithm / Cycle Detection:* Check the Resource Allocation Graph (RAG) using Tarjan's strongly connected components algorithm before granting a tile.
    - *Topological Design:* Enforce strict one-way loop circulation patterns on terminal aprons.
- **What a weak answer reveals:**
  - Attributes the freeze to a software bug, memory leak, or infinite `while` loop in the simulation code.
  - Suggests "teleporting" or randomly deleting vehicles when they stop moving.

---

### Question 5: Digital Twin Synchronization and State Divergence
**Interviewer:** *"What is the difference between an offline discrete-event simulation model and an operational real-time digital twin operating in shadow mode at a container terminal?"*

- **What a strong answer contains:**
  - Differentiates by **telemetry coupling, synchronization frequency, and state re-anchoring**.
  - An offline DES model evaluates static infrastructure designs or policy rules over long hypothetical time horizons using synthetic stochastic generators.
  - An operational digital twin is continuously coupled to live Terminal Operating System (TOS) event buses (via Kafka, MQTT, or EDI 315/214 streams). It initializes its state vector $S_{\text{sim}}(t)$ from physical reality $S_{\text{phys}}(t)$.
  - Addresses **state divergence**: because real-world terminals experience unmodeled micro-delays, sensor noise, and human interventions, simulated state drifts from physical state. The twin must implement periodic **state re-anchoring pulses** to force-align container positions, crane statuses, and queue counts.
  - Explains the twin's predictive function: executing parallel fast-forward lookahead simulations (e.g., projecting 2–4 hours into the future) to detect emerging quayside starvation or gate congestion before it occurs in reality.
- **What a weak answer reveals:**
  - Describes a digital twin merely as a 3D visual CAD animation or an offline simulation model connected to an operational dashboard.
  - Does not understand how the simulation engine handles drift between physical reality and model state.

---

### Question 6: Statistical Delay Fitting for Operational Disruptions
**Interviewer:** *"Why is it dangerous to model container terminal crane hoist and travel times using normal or exponential distributions? What distributions do you use, and why?"*

- **What a strong answer contains:**
  - Explains that the normal distribution allows non-zero probabilities for negative or zero durations, which violates mechanical physics, and is symmetric, whereas logistics delays are heavily right-skewed.
  - Explains that the exponential distribution has a coefficient of variation $c_v = 1$ and a mode at zero, implying that the most probable crane cycle time is zero seconds, which is physically absurd.
  - Recommends three-parameter distributions with an explicit **location parameter (shift) $\theta > 0$** representing physical kinematic limits (minimum hoist/travel time):
    - **Shifted Lognormal:** Excellent representation of manual and semi-automated crane cycles where human micro-delays create a long right-side tail.
    - **Three-Parameter Weibull:** Well-suited for equipment mechanical breakdown and repair intervals.
  - Stresses goodness-of-fit testing using the **Anderson-Darling ($A^2$)** test over Kolmogorov-Smirnov, because Anderson-Darling places higher mathematical weight on tail behavior where extreme operational disruptions occur.
- **What a weak answer reveals:**
  - States that normal distributions are acceptable because of the Central Limit Theorem.
  - Does not know how to handle minimum physical process durations.

---

### Question 7: Buffer Decoupling Economics
**Interviewer:** *"In an automated terminal, what is the operational purpose of installing an elevated cassette or ground buffer in the interchange zone between quay cranes and automated horizontal vehicles?"*

- **What a strong answer contains:**
  - Identifies buffers as **variance decoupling mechanisms** in a tightly coupled cyclic queueing system.
  - Without a buffer, the transfer requires simultaneous presence of both servers: the quay crane spreader and the AGV flatbed. If either server is delayed, the other server is immediately starved or blocked.
  - With a physical buffer cassette (e.g., Lift-AGV systems), the quay crane deposits the container onto the cassette and immediately returns to the vessel hold. The Lift-AGV retrieves the container independently.
  - In queueing terms, buffers convert a synchronous hand-off into an asynchronous queueing system, absorbing stochastic variance in travel times and reducing quay crane starvation by 15% to 25%.
- **What a weak answer reveals:**
  - Believes buffers are installed merely for temporary storage when the yard is full.
  - Cannot explain the mathematical relationship between buffer capacity and server decoupling.

---

### Question 8: Validating Simulation against TOS Telemetry
**Interviewer:** *"You have developed a simulation model of a container terminal. You have 6 months of historical EDI 315 messages and crane PLC logs. How do you rigorously prove your model matches historical reality?"*

- **What a strong answer contains:**
  - Specifies quantitative time-series validation using **Theil's Inequality Coefficient ($U$)**:
    $$U = \frac{\sqrt{\frac{1}{T}\sum (S_t - A_t)^2}}{\sqrt{\frac{1}{T}\sum S_t^2} + \sqrt{\frac{1}{T}\sum A_t^2}}$$
  - Explains the economic interpretation of decomposing $U$ into **Bias Proportion ($U^M$)**, **Variance Proportion ($U^S$)**, and **Covariance Proportion ($U^C$)**.
  - A model with high bias ($U^M$) indicates systematic under-estimation of equipment speeds. A model with high variance ($U^S$) indicates poorly modeled stochastic distributions.
  - Explains the use of two-sample Kolmogorov-Smirnov and Wilcoxon rank-sum tests to compare empirical crane cycle distributions against simulated cycle distributions.
  - Emphasizes testing the model under historical shock events (e.g., a 4-hour crane breakdown) to verify that the simulated recovery trajectory mirrors historical TOS recovery patterns.
- **What a weak answer reveals:**
  - Relies solely on plotting a visual line chart or reporting a simple coefficient of determination ($R^2$), which can mask massive mean biases.
  - Suggests comparing only the aggregate monthly throughput totals.

---

## 15. Key terms

- **Activity Scanning:** A simulation execution worldview that advances time in uniform increments and evaluates conditional activity triggers.
- **Anderson-Darling Test:** A statistical goodness-of-fit test that applies heavy weighting to distribution tails, critical for operational delay fitting.
- **Antithetic Variates:** A variance reduction technique that pairs complementary random uniform numbers ($U$ and $1 - U$) to induce negative covariance.
- **Automated Guided Vehicle (AGV):** Unmanned horizontal transport vehicle navigated via transponders, magnets, or LIDAR.
- **Automated Stacking Crane (ASC):** Rail-mounted gantry crane dedicated to automated container block stacking and interchange.
- **Banker's Algorithm:** A resource allocation algorithm used to avoid deadlocks by testing for safe system states before granting resources.
- **Batch Means:** An output analysis method that partitions a single stationary simulation run into non-overlapping batches to estimate confidence intervals.
- **Berth Moves Per Hour (BMPH):** Total container moves completed across all assigned quay cranes divided by vessel berth working hours.
- **Blocking:** An operational condition where a server completes processing but cannot discharge the entity because downstream buffers are full.
- **Circular Wait:** A deadlock state where each entity holds a resource required by the next entity in a closed dependency loop.
- **Common Random Numbers (CRN):** A variance reduction technique synchronizing identical random streams across competing configurations.
- **Discrete-Event Simulation (DES):** A modeling paradigm where system state changes occur instantaneously at discrete chronological event timestamps.
- **EDI 214:** ANSI ASC X12 transaction standard for transportation carrier shipment status, used in motor carrier gate operations.
- **EDI 315:** ANSI ASC X12 transaction standard for ocean shipment status details, providing terminal milestone events (`VD`, `VA`, `RC`).
- **Event-Scheduling:** A simulation architecture that advances time directly between events managed on a Future Event List.
- **Future Event List (FEL):** A min-heap priority queue storing all scheduled future simulation events ordered by timestamp.
- **Gross Moves Per Hour (GMPH):** Total container lifts divided by total elapsed crane hours, including operational and external delay intervals.
- **Initialization Bias:** Error introduced into steady-state performance estimates by sampling from transient warm-up distributions.
- **Kolmogorov-Smirnov Test:** A non-parametric goodness-of-fit test quantifying the maximum distance between empirical and theoretical cumulative distributions.
- **Lag-1 Autocorrelation ($r_1$):** The correlation between consecutive observations in a time-series, used to confirm batch independence.
- **Net Moves Per Hour (NMPH):** Crane container lifts divided strictly by net operating hours, excluding recorded external delays.
- **Process-Interaction:** A simulation worldview that tracks the lifecycle of individual entities as they navigate through shared resources.
- **Quay Crane (STS):** Ship-to-Shore rail-mounted gantry crane used to lift containers between vessel cell guides and the quayside apron.
- **Shifted Lognormal Distribution:** A three-parameter distribution incorporating an explicit lower bound location parameter for modeling handling times.
- **Starvation:** An operational state where a server is idle and available to work but lacks incoming entities or transfer vehicles.
- **State Re-Anchoring:** An operational digital twin synchronization mechanism that resets simulated state vectors to match physical sensor telemetry.
- **Theil's Inequality Coefficient ($U$):** A normalized metric bounding simulation forecasting error between 0 and 1, decomposed into bias, variance, and covariance.
- **Three-Parameter Weibull:** A continuous distribution characterized by location, scale, and shape, used for equipment failure and repair modeling.
- **Time-Space Tile Reservation:** A collision avoidance method that grants exclusive access to spatial grid tiles over discrete time intervals.
- **Welch's Moving Average:** A cross-replication smoothing technique used to visually and statistically determine simulation warm-up periods.

---

## 16. Sources

1. **Law, A. M.** (2015). *Simulation Modeling and Analysis* (5th ed.). McGraw-Hill Education. ISBN: 978-0073401324. [Seminal text on discrete-event simulation mechanics, Welch's method, batch means, and variance reduction].
2. **Banks, J., Carson, J. S., Nelson, B. L., & Nicol, D. M.** (2010). *Discrete-Event System Simulation* (5th ed.). Prentice Hall. ISBN: 978-0136062127. [Covers Future Event List architectures, pseudorandom generation, and output analysis].
3. **Steenken, D., Voß, S., & Stahlbock, R.** (2004). Container terminal operation and operations research - a classification and literature review. *OR Spectrum*, 26(1), 3–49. DOI: [10.1007/s00291-003-0157-z](https://doi.org/10.1007/s00291-003-0157-z).
4. **Vis, I. F. A., & de Koster, R.** (2003). Transshipment of containers at a container terminal: An overview. *European Journal of Operational Research*, 147(1), 1–16. DOI: [10.1016/S0377-2217(02)00293-X](https://doi.org/10.1016/S0377-2217(02)00293-X).
5. **Stahlbock, R., & Voß, S.** (2008). Operations research at container terminals: a follow-up to the state-of-the-art review. *OR Spectrum*, 30(1), 1–52. DOI: [10.1007/s00291-007-0100-9](https://doi.org/10.1007/s00291-007-0100-9).
6. **Welch, P. D.** (1983). The statistical analysis of simulation results. In P. C. Lavenberg (Ed.), *Computer Performance Modeling Handbook* (pp. 268–328). Academic Press. ISBN: 978-0124387201. [Original mathematical formulation of Welch's moving average procedure].
7. **Schmeiser, B.** (1982). Batch size effects in the analysis of simulation output. *Operations Research*, 30(3), 556–568. DOI: [10.1287/opre.30.3.556](https://doi.org/10.1287/opre.30.3.556).
8. **Buzen, J. P.** (1973). Computational algorithms for closed queueing networks with exponential servers. *Communications of the ACM*, 16(9), 527–531. DOI: [10.1145/362342.362345](https://doi.org/10.1145/362342.362345).
9. **Angeloudis, P., & Bell, M. G. H.** (2011). A review of container terminal simulation models. *Maritime Policy & Management*, 38(5), 523–542. DOI: [10.1080/03088839.2011.597448](https://doi.org/10.1080/03088839.2011.597448).
10. **Theil, H.** (1966). *Applied Economic Forecasting*. North-Holland Publishing Company. [Original mathematical formulation of Theil's inequality coefficient and error decomposition].
11. **ANSI ASC X12 Standard.** (2024). *EDI Transaction Set 315: Ocean Shipment Status Details*. American National Standards Institute.
12. **ANSI ASC X12 Standard.** (2024). *EDI Transaction Set 214: Transportation Carrier Shipment Status Message*. American National Standards Institute.
13. **United Nations Economic Commission for Europe (UNECE).** (2023). *UN/EDIFACT Standard Directory: BAPLIE, COPRAR, COARRI, CODECO*. UN/CEFACT.
14. **IEEE Computer Society.** (2010). *IEEE Std 1516-2010: Standard for Modeling and Simulation High Level Architecture (HLA)*. IEEE. DOI: [10.1109/IEEESTD.2010.5557723](https://doi.org/10.1109/IEEESTD.2010.5557723).
15. **International Organization for Standardization.** (2021). *ISO 23247: Automation systems and integration — Digital twin framework for manufacturing*. ISO.
16. **Occupational Safety and Health Administration.** (2022). *29 CFR Part 1917: Marine Terminals*. U.S. Department of Labor.

---

## 17. Links

- **Parent Module Index:** [`_index.md`](_index.md) — Module 41: Maritime, Rail, and Air Cargo Operations Engineering
- **Sibling Topics:**
  - Topic 01: [`01_berth-quay-crane.md`](01_berth-quay-crane.md) — Berth Allocation and Quay Crane Operations
  - Topic 02: [`02_terminal-yard-gate.md`](02_terminal-yard-gate.md) — Terminal Yard Stacking and Gate Operations
  - Topic 03: [`03_rail-network-paths.md`](03_rail-network-paths.md) — Rail Network and Path Capacity
  - Topic 04: [`04_rail-yards-shunting.md`](04_rail-yards-shunting.md) — Rail Yards and Shunting
  - Topic 05: [`05_air-cargo-uld.md`](05_air-cargo-uld.md) — Air Cargo Build-Up and ULD Operations
  - Topic 06: [`06_intermodal-hinterland.md`](06_intermodal-hinterland.md) — Intermodal Hinterland Evacuation
  - Topic 07: [`07_disruption-recovery.md`](07_disruption-recovery.md) — Disruption Management and Schedule Recovery
  - Topic 09: [`09_dg-security-operations.md`](09_dg-security-operations.md) — Dangerous Goods and Security Operations
  - Topic 10: [`10_automation-capital.md`](10_automation-capital.md) — Terminal Automation and Capital Allocation Decisions
- **Survey Course File:** [`../../../course/41_maritime_rail_air_operations.md`](../../../course/41_maritime_rail_air_operations.md)
- **Relevant Foundational Modules:**
  - [Module 11: Network Design Optimization](../../02_methods/11_network_design_optimization/_index.md)
  - [Module 12: Transportation, Routing, and Last Mile](../../02_methods/12_transportation_routing_last_mile/_index.md)
  - [Module 20: Operations Research & Mathematical Programming](../../02_methods/20_operations_research_mathematical_programming/_index.md)
  - [Module 36: Global Freight, Ports, and Trade Policy](../../06_domain_electives/36_global_freight_ports_trade_policy/_index.md)
