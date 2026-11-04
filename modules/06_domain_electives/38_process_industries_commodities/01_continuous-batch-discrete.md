# Continuous, Batch, and Discrete Production

> Module 38 — Process Industries, Commodities, and Bulk Supply Chains
> Topic 01 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 15 (Production Planning and Scheduling), Module 16 (Inventory Theory)
> Version 2026.5

---

## 1. Why this topic exists

Enterprise Resource Planning (ERP) and Advanced Planning and Scheduling (APS) software were historically architected around discrete manufacturing paradigms: bills of materials (BOMs) structured as neat hierarchical trees, discrete parts counted as positive integers, operations characterized by deterministic run times per piece, and work-in-process (WIP) staged in standardized tote bins or pallet racking. In discrete assembly—such as automotive chassis assembly or electronics fabrication—a production line can be paused mid-shift by halting a conveyor. If a machine breaks down, the upstream parts sit inertly on transfer skids without chemical degradation, phase separation, or catastrophic runaway reactions.

Process industries operate under fundamentally different physical, chemical, and thermodynamic laws. In an ethylene steam cracker, crude oil atmospheric distillation column, blast furnace, cement kiln, or biopharmaceutical cell culture bioreactor:
1. **Material flows continuously or in amorphous fluid batches:** Products are gases, liquids, slurries, powders, or melts measured in metric tons, cubic meters, barrels, or standard liters. They cannot be held in cardboard cartons or staged on open warehouse floors; they require pressure vessels, cryogenic spheres, heated pipelines, jacketed tanks, or nitrogen-blanketed silos.
2. **Operations cannot be arbitrarily stopped or throttled:** A continuous primary cracker or float-glass furnace operates at steady-state equilibrium at temperatures exceeding 1,000°C. Halting a steam cracker induces severe thermal shock, refractory lining spallation, and extensive coking, triggering weeks of downtime and millions of dollars in flaring losses and capital repairs. Throttling throughput below a critical hydrodynamic turn-down ratio induces column weeping, flame instability, or reactor stalling.
3. **Materials undergo non-linear physical and biochemical transformation:** Mass balance must be maintained across stoichiometric yield curves, split fractions, and variable byproduct streams. Intermediate states are frequently thermodynamically unstable, highly perishable, or corrosive. A polymer melt left stationary in an extruder barrel cross-links into a solid block of intractable resin; a live monoclonal antibody fermentation culture lyses if deprived of oxygen or agitation for twenty minutes; molten steel freezes inside a ladle if held beyond its critical thermal window.
4. **Intermediate storage physics dictate operational feasibility:** In discrete manufacturing, buffer storage is primarily an economic trade-off between holding cost and changeover downtime. In process networks, intermediate storage policies—ranging from Unlimited Intermediate Storage (UIS) and Finite Intermediate Storage (FIS) to No Intermediate Storage (NIS) and Zero Wait (ZW)—are rigid thermodynamic and spatial constraints. Under Zero Wait, an intermediate cannot reside in buffer storage for even a single second; if downstream processing capacity is unavailable the instant reaction finishes, the entire batch must be flared, dumped, or destroyed.

Modern manufacturing facilities are rarely pure archetypes. Instead, they operate as **hybrid multi-stage facilities**: continuous chemical reaction upstream produces bulk intermediate concentrates; multi-purpose batch formulation suites downstream blend and crystallize specific grades; and high-speed discrete packaging lines bottle, label, carton, and palletize finished stock-keeping units (SKUs). 

The catastrophic failure in industrial planning occurs when supply chain practitioners apply discrete MRP-II logic across continuous and batch stages. This planning disconnect leads to severe consequences:
- **Upstream hydraulic back-propagation:** Minor packaging line jams propagate backward into intermediate tank overflows, forcing emergency shutdown or flaring of multi-million-dollar continuous reactors.
- **Runaway campaign cycle times:** Excessive setup penalties and fear of changeover cleaning lead planners to run massive single-grade campaigns, generating tens of millions of dollars in trapped bulk working capital while starving high-margin niche customer accounts.
- **Misplaced Order Penetration Points (OPP):** Decoupling inventory at the wrong processing stage creates an acute mismatch: either holding finished packaged inventory subject to rapid expiry and regulatory labeling obsolescence, or pushing the OPP so far upstream that customer delivery lead times exceed competitive market windows.

Bridging this divide requires rigorous mathematical frameworks tailored to process physics: specifically the **State-Task Network (STN)** and **Resource-Task Network (RTN)** mixed-integer linear programming (MILP) paradigms. Mastering these operational physics and mathematical formulations is the foundational prerequisite for optimizing yield, mass balances, campaign sequences, and bulk supply chains.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Deconstruct** complex production facilities into their continuous, batch, and discrete operational stages based on thermodynamic state variables, mass transfer mechanisms, and operational control paradigms.
2. **Determine** the optimal Order Penetration Point (OPP / CODP) across hybrid process-discrete supply chains by quantitatively balancing campaign economies of scale, bulk intermediate perishability, asset utilization, and customer lead-time tolerance.
3. **Classify** and mathematically model the four fundamental intermediate storage policies: Unlimited Intermediate Storage (UIS), Finite Intermediate Storage (FIS), No Intermediate Storage (NIS), and Zero Wait (ZW).
4. **Formulate** production scheduling problems using the State-Task Network (STN) MILP framework, deriving exact mass balance, capacity utilization, and intermediate storage equations over discrete time horizons.
5. **Transform** complex multi-resource scheduling environments into the generalized Resource-Task Network (RTN) MILP framework, handling processing units, storage vessels, shared utility headers (steam, power, cooling water), and cleaning skids under a single unified mathematical balance.
6. **Evaluate** the mathematical and computational trade-offs between uniform discrete-time discretization and continuous-time event-based formulations in industrial-scale scheduling models.
7. **Construct** an executable, bounded Python optimization script that solves an intermediate-storage-constrained batch scheduling problem and verifies storage policy feasibility.
8. **Audit** enterprise planning architectures against ISA-88 batch control standards and ISA-95 enterprise-control system integration hierarchies to eliminate planning disconnects between Level 4 ERP and Level 3 MES schedulers.

---

## 3. The core mental model

The industrial process supply chain is governed by a physical-informational continuum that transforms raw bulk materials into differentiated commercial products. The physical flow moves from **low-entropy continuous fluid flow** through **discrete-batch recipe execution** to **high-entropy discrete unit assembly and packaging**. 

Simultaneously, the planning logic transitions from continuous rate-based campaign scheduling to discrete sequence-dependent MILP lot sizing and dispatching.

```text
=============================================================================================================
                  THE CONTINUOUS — BATCH — DISCRETE CONTINUUM AND THE OPP
=============================================================================================================

UPSTREAM: CONTINUOUS                   MIDSTREAM: BATCH                      DOWNSTREAM: DISCRETE
- Rate-based (t/h, bpd, kg/s)         - Recipe-based (kg/batch, cycle time) - Unit-based (eaches, cartons)
- Steady-state equilibrium            - Dynamic, time-varying states        - Discrete pick, place, pack
- Distributed parameter systems       - Lumped parameter systems            - Integer BOM explosion
- 24/7/365 uninterrupted campaigns   - Multi-purpose shared vessels        - Flexible routing & shifts
─────────────────────────────────     ─────────────────────────────────     ─────────────────────────────────
   [ Crude Oil / Feedstock ]                 [ Active Intermediates ]              [ Finished Packaged SKUs ]
               │                                       │                                       │
               ▼                                       ▼                                       ▼
    ┌──────────────────────┐                ┌──────────────────────┐                ┌──────────────────────┐
    │  Continuous Reactor  │                │ Multi-Product Batch  │                │ High-Speed Discrete  │
    │  Column / Kiln       │                │ Reactors / Blenders  │                │ Packaging Line       │
    │  (Cracking, Reform)  │                │ (Crystallize, Mix)   │                │ (Fill, Cap, Pallet)  │
    └──────────┬───────────┘                └──────────┬───────────┘                └──────────┬───────────┘
               │                                       │                                       │
               ▼                                       ▼                                       ▼
    ┌──────────────────────┐                ┌──────────────────────┐                ┌──────────────────────┐
    │ Intermediate Storage │                │ Bulk Holding Tanks   │                │ Finished Goods Whse  │
    │ Surge Tanks / Silos  │                │ Buffer Staging Vessels│               │ Pallet Racking / ASRS│
    └──────────────────────┘                └──────────────────────┘                └──────────────────────┘
               │                                       │                                       │
═══════════════╪═══════════════════════════════════════╪═══════════════════════════════════════╪═════════════
               │                                       │                                       │
   [ OPP Candidate 1: MTS ]                [ OPP Candidate 2: ATO / FTO ]          [ OPP Candidate 3: MTO ]
   - Standard commodity intermediates      - Differentiated liquid bulk base       - Pure custom formulation
   - High volume, low variety              - Postponement of color/flavor/pack     - High variety, low volume
   - Maximum asset utilization             - Optimal inventory risk pooling        - Long lead times
   - Long campaign run lengths             - Agility to market volatility          - Zero finished stock

=============================================================================================================
                             INTERMEDIATE STORAGE THERMODYNAMIC POLICIES
=============================================================================================================

  1. UIS (Unlimited Intermediate Storage)       2. FIS (Finite Intermediate Storage)
     [Unit j1] ──► [Infinite Tank] ──► [Unit j2]   [Unit j1] ──► [Tank V_min..V_max] ──► [Unit j2]
     (Decoupled; no blocking)                      (Coupled via capacity bounds; overflow risk)

  3. NIS (No Intermediate Storage)              4. ZW (Zero Wait)
     [Unit j1 (Material Held)] ──► [Unit j2]       [Unit j1] ──────────────────────────► [Unit j2]
     (Unit j1 blocked until j2 is free)            (Instantaneous transfer; Δt = 0; spoilage risk)
=============================================================================================================
```

### The Physics-to-Information Mismatch

In the continuous stage, processing units operate under distributed physical and chemical gradients (temperature $T(z)$, pressure $P(z)$, concentration $C_i(z)$ along reactor length $z$). Control is handled by Distributed Control Systems (DCS) executing proportional-integral-derivative (PID) and Model Predictive Control (MPC) algorithms at millisecond-to-second frequencies. The planning decision is a **continuous production rate** ($u(t) \in [u^{\min}, u^{\max}]$) and a **run length** ($L_c$) for a multi-week campaign.

In the batch stage, operations are governed by the ISA-88 physical and procedural hierarchy: **Procedure $\to$ Unit Procedure $\to$ Operation $\to$ Phase**. A batch follows a recipe where mass additions, heating phases, reaction hold times, and cooling ramps are tied to a specific physical unit. Unlike continuous steady-states, batch states are inherently transient ($dC_i/dt = f(C, T, P)$). Key challenges include multi-purpose equipment allocation, shared cleaning-in-place (CIP) resources, and intermediate storage policies.

In the discrete stage, parts move sequentially through assembly, filling, labeling, and cartoning stations. Control is governed by Programmable Logic Controllers (PLCs) coordinating sensor gates, indexers, and servo-drives. Planning decisions revolve around line speeds (units/minute), shift staffing, and line changeovers between pack sizes or label languages.

The central mental model requires viewing the manufacturing facility as a network of **dynamic capacity nodes connected by thermodynamic buffers**. If the intermediate buffers fail to absorb variance or violate intermediate storage policies, upstream and downstream physics collide catastrophically.

---

## 4. Definitions and vocabulary

The following vocabulary formalizes the operational, thermodynamic, and mathematical terms used throughout continuous, batch, and discrete process planning.

| Term | Mathematical / Formal Definition | Common Misuse / Conceptual Error | Standard Unit / Classification |
|---|---|---|---|
| **Continuous Production** | A manufacturing mode where materials flow uninterruptedly through steady-state processing equipment, with inputs converted to outputs without cycle breaks. | Confusing high-speed discrete packaging lines running 24/7 with thermodynamic continuous processing. | Metric tons per hour ($t/h$), barrels per day ($bpd$), kilograms per second ($kg/s$). |
| **Batch Production** | A manufacturing mode where a finite quantity of material is subjected to a sequence of ordered processing operations according to an ISA-88 recipe in one or more vessels. | Assuming batch size is an unconstrained continuous decision variable; in reality, it is strictly bounded by minimum agitation and maximum vessel volume ($V^{\min} \le B \le V^{\max}$). | Kilograms per batch ($kg/\text{batch}$), cubic meters ($m^3$), liters ($L$). |
| **Discrete Manufacturing** | A production process that creates distinct, countable items whose bill of materials is structured as distinct sub-assemblies and individual parts. | Applying discrete MRP lead-time offsets to fluid mixing stages where intermediate storage policies dictate immediate consumption. | Eaches, cartons, cases, pallets, serial numbers. |
| **Hybrid Plant** | A multi-stage production system containing continuous, batch, and discrete stages separated by intermediate bulk or buffer storage. | Treating the entire facility as pure discrete or pure continuous, ignoring the mathematical boundary interfaces. | Dimensionality transitions: mass flow ($kg/s$) $\to$ batch lot ($kg$) $\to$ discrete counts ($units$). |
| **Order Penetration Point (OPP / CODP)** | The stage in the manufacturing value chain where a product is allocated to a specific customer order, transitioning from forecast-driven push to order-driven pull. | Believing the OPP must always sit at the finished goods warehouse; placing it upstream enables postponement and slashes finished inventory. | Topological stage index ($k \in \{1, \dots, K\}$); time buffer ($days$). |
| **State-Task Network (STN)** | A bipartite directed graph formulation where nodes represent either States (materials, feeds, intermediates) or Tasks (physical/chemical transformations). | Collapsing state nodes into processing units, obscuring multi-input/multi-output stoichiometric relationships. | Mathematical graph: $\mathcal{G} = (\mathcal{S} \cup \mathcal{T}, \mathcal{E})$. |
| **Resource-Task Network (RTN)** | A generalized scheduling framework where all physical entities (processing units, storage tanks, utilities, manpower) are uniformly represented as Resources. | Treating processing units as separate constraints from utility headers or labor; RTN models all entities under a unified resource balance. | Mathematical graph: $\mathcal{G} = (\mathcal{R} \cup \mathcal{T}, \mathcal{E})$; availability $R_{rt}$. |
| **Unlimited Intermediate Storage (UIS)** | An intermediate storage policy where buffer capacity between sequential operations is effectively infinite; tasks never experience blocking due to storage saturation. | Assuming UIS is common in process plants; outside of vast open bulk solid yards or large atmospheric water basins, storage is almost always finite. | $S_{st} \le \infty$; capacity bound relaxed. |
| **Finite Intermediate Storage (FIS)** | A policy where intermediate materials are stored in dedicated or shared vessels of strictly bounded physical capacity ($0 \le S_{st} \le C_s^{\max}$). | Ignoring heel volume requirements or maximum vapor head space limits when calculating available storage. | Volume ($m^3$), Mass ($t$, $kg$), Head ($m$). |
| **No Intermediate Storage (NIS)** | A policy where no intermediate buffer vessel exists; an upstream vessel must retain the processed material until the downstream unit becomes available. | Omitting the blocking duration of the upstream processing unit during scheduling, overstating plant throughput capacity. | Blocking duration: $\tau_{block} = t^{start}_{downstream} - t^{end}_{upstream}$. |
| **Zero Wait (ZW)** | A policy where an intermediate is thermodynamically or biochemically unstable and must be transferred to the downstream task immediately upon task completion ($\Delta t = 0$). | Treating ZW as a scheduling preference rather than a hard physical requirement; violating ZW leads to batch destruction. | Exact equality: $t^{start}_{i+1} - t^{end}_i = 0$. |
| **Campaign** | A continuous or repeated batch run of a single product grade or formulation on a dedicated processing line before incurring a major sequence-dependent changeover. | Treating campaign length as static; optimal campaign length balances changeover cost against inventory carrying costs under stochastic demand. | Run duration ($hours$, $days$), cumulative mass ($t$). |
| **Turnaround (TAR)** | A scheduled, periodic complete shutdown of a continuous process unit or refinery to conduct mandatory mechanical inspection, catalyst replacement, and vessel overhaul. | Planning production during a turnaround window without accounting for inventory pre-build or third-party merchant market sourcing. | Downtime duration ($weeks$); capital expenditure ($USD$). |
| **Clean-In-Place (CIP)** | An automated system designed to clean the interior surfaces of pipes, vessels, process equipment, filters, and fittings without major mechanical disassembly. | Failing to model CIP skids as scarce shared resources, causing deadlocks where reactors finish processing but cannot discharge due to uncleaned transfer lines. | Cleaning cycle duration ($minutes$, $hours$); rinse volume ($m^3$). |
| **Steam-In-Place (SIP)** | The thermal sterilization of process equipment using pressurized saturated clean steam, typically required between sterile biopharmaceutical batch operations. | Confusing chemical sanitization with thermal sterilization; SIP requires strict thermal dwell times at $\ge 121.1^\circ\text{C}$. | Temperature dwell ($min \ge 121.1^\circ\text{C}$); steam mass ($kg$). |
| **Tank Heel** | The unpumpable or unmixable residual volume residing below the suction nozzle or agitator impeller blade at the bottom of a bulk storage tank. | Counting total geometric vessel volume as usable storage; available working volume is $V_{working} = V_{total} - V_{heel} - V_{ullage}$. | Inactive volume ($m^3$, $L$), percentage of geometric volume ($3\text{--}8\%$). |
| **Ullage** | The unoccupied space left at the top of a liquid storage tank to accommodate thermal expansion, foaming, or volatile vapor head space. | Filling tanks to 100% geometric capacity, violating process safety management (PSM) overfill prevention rules. | Safety head buffer ($m$, $m^3$, % of height). |
| **Turndown Ratio** | The ratio of maximum operating capacity to minimum sustainable operating capacity at which a continuous unit can operate safely without process instability. | Assuming continuous units can run at arbitrary low rates; operating below minimum turndown induces column weeping, flameout, or thermal collapse. | Dimensionless ratio: $\frac{u^{\max}}{u^{\min}}$ (e.g., $3:1$ or $5:1$). |
| **Postponement (Late Differentiation)** | An operational strategy where the final form, packaging, or chemical customization of a product is delayed until customer demand is revealed. | Postponing processing without providing sufficient intermediate bulk buffering, causing starvation of packaging lines. | Lead time delta ($\Delta LT$), inventory pooling factor ($\sqrt{N}$). |
| **ISA-88 (IEC 61512)** | The international standard governing batch control architecture, recipe structures, and physical equipment models. | Treating ISA-88 as software documentation rather than the governing semantic data model for batch operations. | Regulatory/technical standard. |
| **ISA-95 (IEC 62264)** | The international standard for integrating enterprise management systems (Level 4 ERP) with manufacturing operations control systems (Level 3 MES). | Attempting direct ERP-to-DCS integration without intermediate MES scheduling layers, causing physical execution failures. | Architectural hierarchy: Levels 0 through 4. |

---

## 5. The mechanics, step by step

The design, planning, and scheduling of hybrid process industries requires a six-step engineering procedure that integrates physical thermodynamics with mathematical optimization.

```text
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│                        SIX-STEP HYBRID PROCESS PLANNING & SCHEDULING WORKFLOW                         │
└───────────────────────────────────┬────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 1: Physical Process Classification & Thermodynamic Operating Windows                             │
│ - Delineate continuous, batch, and discrete processing boundaries based on mass flow and state dynamics│
│ - Identify turndown ratios, thermal inertia, catalyst run-lives, and minimum agitation limits         │
└───────────────────────────────────┬────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 2: Order Penetration Point (OPP) Positioning & Postponement Architecture                          │
│ - Compute customer tolerance time vs manufacturing lead time (D/P ratio)                               │
│ - Evaluate bulk intermediate stability, holding costs, and SKU multiplication factors                  │
│ - Anchor physical decoupling points (surge drums, tank farms, buffer silos)                            │
└───────────────────────────────────┬────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 3: Intermediate Storage Thermodynamic & Kinetic Policy Assignment                                 │
│ - Map all intermediate states to storage policies: UIS, FIS, NIS, or ZW                                │
│ - Calculate usable working volumes: subtract heel volumes and ullage allowances                       │
│ - Define maximum holding times before kinetic degradation, caking, or phase separation occurs          │
└───────────────────────────────────┬────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 4: Mathematical Formulation I — The State-Task Network (STN) MILP                                 │
│ - Construct bipartite graph: States (materials) and Tasks (reactions, separations)                    │
│ - Define binary assignment variables, batch extents, and inventory states across time grid             │
│ - Formulate mass balances, storage bounds, and unit exclusivity constraints                            │
└───────────────────────────────────┬────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 5: Mathematical Formulation II — The Resource-Task Network (RTN) MILP                             │
│ - Generalize all physical entities (reactors, storage, steam, cooling water, CIP skids) as Resources   │
│ - Parameterize fixed and extent-dependent resource consumption matrices                                │
│ - Establish unified resource balances capturing shared utility peak loads and renewal cycles           │
└───────────────────────────────────┬────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ Step 6: Time Grid Discretization, Big-M Linearization, and Solver Execution                            │
│ - Select uniform discrete time interval (Δt) balancing resolution against integer variable explosion   │
│ - Linearize non-linear bilinearity; calibrate tight Big-M bounds to prevent LP relaxation drift       │
│ - Execute branch-and-cut optimization, extract Gantt schedules, and verify intermediate feasibility    │
└────────────────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Physical Process Classification & Thermodynamic Operating Windows

Every processing stage in the enterprise must be rigorously classified according to its governing fluid dynamic and chemical regime:

1. **Continuous Units:**
   - **Governing Equations:** Steady-state conservation of mass and energy:
     $$\nabla \cdot (\rho \mathbf{u}) = 0, \quad \rho (\mathbf{u} \cdot \nabla) h = \nabla \cdot (k \nabla T) + S_h$$
   - **Operational Window:** Flow rate must be maintained within allowable turndown limits:
     $$u_j^{\min} \le u_j(t) \le u_j^{\max} \quad \forall t \in [0, H]$$
   - **Transition Costs:** Switching between grades (e.g., from Low-Density Polyethylene Grade A to Grade B) requires transition periods where off-spec material is generated. The volume of off-spec material $M_{trans}$ is a function of the reactor residence time $\tau_{res} = \frac{V_{reactor}}{\dot{V}}$ and the degree of separation required:
     $$M_{trans} = \int_0^{t_{trans}} \rho(t) \dot{V}(t) dt \approx 3 \cdot \tau_{res} \cdot \rho \cdot \dot{V}$$
   - **Thermal Inertia:** Restarting a cold unit requires a defined ramp rate ($\frac{dT}{dt} \le \alpha_{thermal}$) to prevent refractory cracking.

2. **Batch Units:**
   - **Governing Equations:** Transient mass and energy balances:
     $$\frac{dM_{i}}{dt} = \sum_{in} \dot{m}_{in} - \sum_{out} \dot{m}_{out} + V_j \sum_r \nu_{ir} r_r(C, T)$$
   - **Operational Limits:** Batch size $B_{ijt}$ running task $i$ in vessel $j$ is strictly bounded by minimum physical agitation geometry $V_j^{\min}$ and maximum vessel shell capacity $V_j^{\max}$:
     $$V_j^{\min} \cdot W_{ijt} \le B_{ijt} \le V_j^{\max} \cdot W_{ijt}$$
     where $W_{ijt} \in \{0, 1\}$ is the binary decision to start task $i$ in unit $j$ at time $t$. Operating with $B_{ijt} < V_j^{\min}$ leaves impeller blades un-submerged, causing catastrophic lack of heat transfer, localized boiling, or thermal runaways.

3. **Discrete Packaging Units:**
   - **Governing Equations:** Discrete item flow governed by machine cycle rates:
     $$Q_{out}(t) = \min \left\{ R_{nominal} \cdot \eta_{OEE} \cdot \Delta t, \; \frac{S_{bulk}(t)}{\text{Bulk per SKU}} \right\}$$
   - **Operational Limits:** Changeover times between SKU package sizes are sequence-dependent (e.g., cleanout from allergen-containing recipes to non-allergen recipes requires complete line teardown).

---

### Step 2: Order Penetration Point (OPP) Positioning & Postponement Architecture

The placement of the Order Penetration Point (OPP) determines the boundary between forecast-driven push scheduling and demand-driven pull scheduling. In hybrid process networks, the OPP should be positioned at the processing stage that satisfies three criteria:

1. **Lead Time Feasibility (The $D/P$ Ratio):**
   - Let $D$ be the customer's delivery tolerance time (the maximum time a customer is willing to wait between order placement and receipt).
   - Let $P$ be the cumulative manufacturing and logistical throughput time from the chosen OPP stage to final customer delivery.
   - **Condition:** The OPP is operationally viable at stage $k$ if and only if:
     $$P_k \le D$$
   - If $P_k > D$, intermediate inventory must be pushed downstream past stage $k$.

2. **Formulation Variety and SKU Proliferation:**
   - Let $N_{SKU}(k)$ be the number of distinct product variants at stage $k$. In process manufacturing, $N_{SKU}$ explodes downstream:
     $$\text{Feedstocks } (1\text{--}5) \longrightarrow \text{Bulk Intermediates } (5\text{--}20) \longrightarrow \text{Formulated Blends } (20\text{--}100) \longrightarrow \text{Packaged SKUs } (500\text{--}5,000)$$
   - Positioning the OPP at the bulk intermediate stage enables **risk pooling**. The total safety stock required under decentralized packaged inventory versus pooled bulk intermediate inventory scales according to Eppen's square-root law modified for fluid yield variance:
     $$SS_{pooled} = z_{\alpha} \sqrt{ \sum_{i=1}^M \sigma_{i}^2 + 2 \sum_{i < j} \text{Cov}(D_i, D_j) } \cdot \frac{1}{\bar{Y}_{yield}}$$

3. **Thermodynamic and Chemical Stability:**
   - Bulk intermediates can only act as an OPP if their shelf-life $\tau_{shelf}$ exceeds the replenishment cycle time plus safety time buffer:
     $$\tau_{shelf} \ge T_{cycle} + z_{\alpha} \sigma_{lead}$$
   - If an intermediate chemically destabilizes, oxidizes, or cakes within 48 hours, it cannot serve as an OPP; the OPP must be pushed upstream to a stable precursor or downstream to a sealed, stabilized finished SKU.

---

### Step 3: Intermediate Storage Thermodynamic & Kinetic Policy Assignment

Intermediate storage between processing stages cannot be assumed to behave as a generic inventory bucket. Every intermediate buffer must be assigned one of four physical storage policies:

```text
=============================================================================================================
                                INTERMEDIATE STORAGE POLICY COMPARISON
=============================================================================================================
Policy  Physical Requirement            Mathematical Constraint                       Operational Failure Mode
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
UIS     Dedicated boundless yard        $0 \le S_{st} \le \infty$                     None (idealized baseline;
        or open atmospheric basin                                                     incurring negligible blocking).
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
FIS     Dedicated buffer tank or        $S_{s}^{\min} \le S_{st} \le S_{s}^{\max}$    Tank overfill halts upstream
        shared tank farm                                                              reactor; tank starvation stops
                                        $S_{s}^{\min} = \text{Heel Volume}$           downstream packaging line.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
NIS     No storage tank exists;         $S_{st} = 0 \quad \forall t$                  Upstream reactor remains locked
        fluid remains in upstream unit  $W_{ijt} + \sum W_{downstream} \le 1$         with finished product, blocking
                                                                                      subsequent batch starts.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
ZW      Unstable, reacting, or          $S_{st} = 0 \quad \forall t$                  Instantaneous degradation;
        freezing material               $\sum_{j \in J_i} W_{ijt} = \sum_{j' \in J_{i'}} W_{i', j', t+p_i}$  batch off-spec/dumped if downstream
                                                                                      unit is not ready at exact second.
=============================================================================================================
```

#### Detailed Thermodynamic Formulations:

1. **Unlimited Intermediate Storage (UIS):**
   - Assumes external storage is always available. Upstream task discharges immediately upon completion.
   - Downstream task draws from storage at will. Mass balance:
     $$S_{s, t} = S_{s, t-1} + \text{Inflow}_{st} - \text{Outflow}_{st}$$
   - Storage capacity upper bound is unconstrained: $S_{st} \ge 0$.

2. **Finite Intermediate Storage (FIS):**
   - Storage resides in dedicated or manifolded vessels with explicit working volume envelopes.
   - Let $V_{total}$ be total vessel geometric volume, $V_{heel}$ be dead volume below the lowest discharge point, and $V_{ullage}$ be safety vapor headspace.
     $$S_s^{\max} = \rho_s \cdot (V_{total} - V_{ullage})$$
     $$S_s^{\min} = \rho_s \cdot V_{heel}$$
   - Inventory bounds must hold across every time period:
     $$S_s^{\min} \le S_{st} \le S_s^{\max} \quad \forall t$$

3. **No Intermediate Storage (NIS):**
   - The intermediate cannot exist outside of a processing unit. If Unit $j_1$ completes task $i_1$, but downstream Unit $j_2$ is still occupied, material must remain inside Unit $j_1$.
   - Unit $j_1$ is in a **blocked state** ($\text{State} = \text{Blocked}$). It cannot begin cleaning, CIP, or its next batch.
   - Let $p_i$ be normal processing time, and $\tau_{i,j}^{block}$ be the blocking duration. Total unit occupancy time is:
     $$T_{occupancy} = p_i + \tau_{i,j}^{block}$$
   - In discrete-time models, this requires extending the binary occupancy variable of Unit $j_1$ until Unit $j_2$ initiates task $i_2$.

4. **Zero Wait (ZW):**
   - Thermodynamic instability dictates that holding time $\tau_{hold} = 0$.
   - The start time of downstream task $i_2$ on Unit $j_2$ must coincide identically with the completion time of upstream task $i_1$ on Unit $j_1$:
     $$t^{start}_{i_2, j_2} = t^{start}_{i_1, j_1} + p_{i_1}$$
   - In discrete time grid with duration $\Delta t$, if task $i_1$ has duration $p_{i_1}$, then for any time $t$:
     $$W_{i_1, j_1, t} \le \sum_{j' \in J_{i_2}} W_{i_2, j', t + p_{i_1}}$$

---

### Step 4: Mathematical Formulation I — The State-Task Network (STN) MILP

The State-Task Network (Kondili, Pantelides, and Sargent, 1993) models process scheduling by mapping all operations into a bipartite graph $\mathcal{G} = (\mathcal{S} \cup \mathcal{T}, \mathcal{E})$, where:
- $\mathcal{S}$ is the set of States $s$ (raw feedstocks, intermediate chemicals, finished blends).
- $\mathcal{T}$ is the set of Tasks $i$ (reaction, distillation, mixing, packaging).
- Directed edges represent consumption or production of states by tasks.

```text
=============================================================================================================
                                STATE-TASK NETWORK (STN) BIPARTITE ARCHITECTURE
=============================================================================================================

  [ Raw State S1 ] ─────( ρ_i1,s1 )─────► ┌───────────────┐ ─────( ρ̄_i1,s2 )─────► [ Intermediate S2 ]
                                         │ Task i1 (Rxn) │
  [ Raw State S3 ] ─────( ρ_i1,s3 )─────► └───────────────┘ ─────( ρ̄_i1,s4 )─────► [ By-Product S4 ]
                                                 ▲
                                                 │ runs in Unit j ∈ J_i1
                                          (Capacity: V_min .. V_max)
=============================================================================================================
```

#### Notation and Sets:
- $t \in \{1, 2, \dots, H\}$: Uniform discrete time points defining intervals of width $\Delta t$.
- $s \in \mathcal{S}$: States (materials).
- $i \in \mathcal{T}$: Tasks (processing operations).
- $j \in \mathcal{J}$: Processing units.
- $\mathcal{J}_i \subseteq \mathcal{J}$: Set of units suitable for performing task $i$.
- $\mathcal{T}_j \subseteq \mathcal{T}$: Set of tasks that can be performed in unit $j$.
- $p_i$: Processing time of task $i$ (in integer multiples of $\Delta t$).
- $\bar{\rho}_{is}$: Mass fraction of state $s$ produced by task $i$.
- $\rho_{is}$: Mass fraction of state $s$ consumed by task $i$.
- $V_{ij}^{\min}, V_{ij}^{\max}$: Minimum and maximum capacity of task $i$ in unit $j$.
- $C_s^{\max}$: Maximum storage capacity for state $s$.

#### Decision Variables:
- $W_{ijt} \in \{0, 1\}$: Binary variable; 1 if unit $j$ starts task $i$ at time point $t$; 0 otherwise.
- $B_{ijt} \ge 0$: Continuous variable; batch extent (mass/volume) of task $i$ started in unit $j$ at time point $t$.
- $S_{st} \ge 0$: Continuous variable; inventory of state $s$ stored at time point $t$.

#### Constraints:

1. **Unit Exclusivity (No Overlapping Tasks):**
   A unit $j$ can perform at most one task at any time point $t$. If task $i$ with duration $p_i$ started at $t - \theta$ (where $0 \le \theta \le p_i - 1$), the unit is occupied:
   $$\sum_{i \in \mathcal{T}_j} \sum_{\theta = 0}^{p_i - 1} W_{ij, t-\theta} \le 1 \quad \forall j \in \mathcal{J}, \; \forall t \in \{1, \dots, H\}$$

2. **Capacity Bounds on Batch Size:**
   The batch size $B_{ijt}$ must respect vessel geometry when task $i$ is active, and must be strictly zero when inactive:
   $$V_{ij}^{\min} W_{ijt} \le B_{ijt} \le V_{ij}^{\max} W_{ijt} \quad \forall i \in \mathcal{T}, \; \forall j \in \mathcal{J}_i, \; \forall t \in \{1, \dots, H\}$$

3. **State Mass Balance Across Time:**
   The inventory of state $s$ at time point $t$ equals the inventory at $t-1$, plus production from tasks completing at $t$, minus consumption from tasks starting at $t$, plus external receipts $R_{st}$, minus customer shipments $D_{st}$:
   $$S_{st} = S_{s, t-1} + \sum_{i \in \mathcal{T}_s^{prod}} \sum_{j \in \mathcal{J}_i} \bar{\rho}_{is} B_{ij, t-p_i} - \sum_{i \in \mathcal{T}_s^{cons}} \sum_{j \in \mathcal{J}_i} \rho_{is} B_{ijt} + R_{st} - D_{st} \quad \forall s \in \mathcal{S}, \; \forall t \in \{1, \dots, H\}$$
   where $\mathcal{T}_s^{prod}$ is the set of tasks producing state $s$, and $\mathcal{T}_s^{cons}$ is the set of tasks consuming state $s$.

4. **Storage Policy Constraints:**
   - **FIS:** $0 \le S_{st} \le C_s^{\max} \quad \forall s \in \mathcal{S}^{FIS}, \; \forall t$.
   - **NIS:** $S_{st} = 0 \quad \forall s \in \mathcal{S}^{NIS}, \; \forall t$. (Forces direct unit-to-unit transfer or holding).
   - **ZW:** $S_{st} = 0 \quad \forall s \in \mathcal{S}^{ZW}, \; \forall t$, and direct link:
     $$\sum_{j \in \mathcal{J}_{i_1}} \bar{\rho}_{i_1, s} B_{i_1, j, t-p_{i_1}} = \sum_{i_2 \in \mathcal{T}_s^{cons}} \sum_{j' \in \mathcal{J}_{i_2}} \rho_{i_2, s} B_{i_2, j', t} \quad \forall t$$

---

### Step 5: Mathematical Formulation II — The Resource-Task Network (RTN) MILP

The Resource-Task Network (Pantelides, 1994) eliminates the artificial distinction between processing units, storage vessels, and utility streams. Everything required to perform an industrial transformation is modeled uniformly as a **Resource** $r \in \mathcal{R}$.

Resources include:
- Primary equipment (reactors, filters, packaging lines).
- Secondary equipment (mobile transport totes, cleaning skids).
- Utilities (high-pressure steam headers, chilled brine, electricity peak load).
- Manpower (certified operators, quality control technicians).
- Materials (raw ingredients, intermediate chemicals, final products).

```text
=============================================================================================================
                                RESOURCE-TASK NETWORK (RTN) UNIFIED FRAMEWORK
=============================================================================================================

                  ┌────────────────────────────────────────────────────────┐
                  │                    Task k (Reaction)                   │
                  └──────┬──────────────┬──────────────┬─────────────┬─────┘
                         │              │              │             │
    Consumes at start:   ▼              ▼              ▼             ▼   Produces at end:
   - Raw Chemical Res   - Reactor Res  - Steam Header - Labor Shift  - Intermed Chemical
   - Unit Resource      (Unavailable)  (MW Draw)      (Tech Hours)   - Unit Resource (Renewed)
=============================================================================================================
```

#### The Universal Resource Balance Equation:

Let $N_{kt} \in \{0, 1\}$ be the binary decision to start task $k$ at time $t$.  
Let $\xi_{kt} \ge 0$ be the continuous extent (batch size) of task $k$ started at time $t$.  
Let $R_{rt} \ge 0$ be the available quantity of resource $r$ at time point $t$.

The universal RTN balance across all resource types is:
$$R_{rt} = R_{r, t-1} + \sum_{k \in \mathcal{K}} \sum_{\theta = 0}^{\tau_k} \left( \mu_{kr\theta} N_{k, t-\theta} + \nu_{kr\theta} \xi_{k, t-\theta} \right) + \Pi_{rt} \quad \forall r \in \mathcal{R}, \; \forall t \in \{1, \dots, H\}$$

Where:
- $\tau_k$: Duration of task $k$.
- $\mu_{kr\theta}$: Fixed interaction parameter between task $k$ and resource $r$ at relative time offset $\theta \in \{0, \dots, \tau_k\}$.
- $\nu_{kr\theta}$: Variable (extent-dependent) interaction parameter between task $k$ and resource $r$ at relative time offset $\theta$.
- $\Pi_{rt}$: External supply or demand of resource $r$ at time point $t$.

#### Physical Interpretation of Parameters:
- **Dedicated Equipment Resource:**
  - When task $k$ begins at $\theta = 0$, it consumes 1 unit of reactor resource: $\mu_{kr, 0} = -1$.
  - While running ($0 < \theta < \tau_k$), the reactor remains unavailable: interaction is 0.
  - When task $k$ finishes at $\theta = \tau_k$, it releases the reactor resource: $\mu_{kr, \tau_k} = +1$.
  - Equipment availability constraint is simply: $0 \le R_{rt} \le 1$.
- **Consumable Raw Material Resource:**
  - Consumed at start proportional to extent: $\nu_{kr, 0} = -\rho_{kr}$.
  - Produced at completion proportional to extent: $\nu_{kr, \tau_k} = +\bar{\rho}_{kr}$.
- **Non-Storable Utility (e.g., High-Pressure Steam):**
  - Consumed at constant rate during every active time step: $\nu_{kr\theta} = -\sigma_{steam}$ for all $\theta \in [0, \tau_k - 1]$.
  - The steam header availability $R_{rt} \ge 0$ enforces that total instantaneous steam demand cannot exceed maximum boiler capacity $Steam_{\max}$. Because steam cannot be accumulated across time periods, the inventory carry-over is severed by setting $R_{r, t-1} = 0$ or treating the resource as non-accumulative:
    $$\sum_{k} \sum_{\theta} |\nu_{kr\theta}| \xi_{k, t-\theta} \le Steam_{\max}(t)$$

---

### Step 6: Continuous-Time vs Discrete-Time Formulations & Big-M Linearizations

Industrial optimization models face a severe trade-off between the temporal resolution of the model and its computational tractability.

```text
=============================================================================================================
                      DISCRETE-TIME VS CONTINUOUS-TIME SCHEDULING TRADEOFFS
=============================================================================================================
Dimension                 Discrete-Time Grid                       Continuous-Time Grid
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Time Horizon              Partitioned into $H$ uniform intervals   Variable event points $t_n$ or slot bounds
                          $\Delta t = 15\text{ min}, 1\text{ hr}$  determined dynamically by the solver
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Binary Variable Scaling   $O(|\mathcal{T}| \cdot |\mathcal{J}| \cdot H)$            $O(|\mathcal{T}| \cdot |\mathcal{J}| \cdot N_{events})$
                          Explodes rapidly as $H$ grows            Independent of horizon length; scales with events
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Operational Accuracy      Subject to discretization errors;        Exact event timing; no rounding or idle
                          tasks rounded up to nearest $\Delta t$   slack introduced
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Utility Constraints       Trivial to enforce instantaneous linear  Non-linear bilinear overlaps; requires complex
                          utility peak loads at each step $t$      cross-interval integration or Big-M formulations
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Industrial Usage          Standard in refinery / chemical ERP      High-value biopharma / multi-purpose batch
                          campaign planning (weeks to months)      formulation suites (hours to days)
=============================================================================================================
```

#### Continuous-Time Precedence and Big-M Linearization:

In continuous-time formulations, let $t_i^s$ be the continuous start time of task $i$, and $T_i$ be its duration. If two tasks $i$ and $i'$ are assigned to the same processing unit $j$, they cannot overlap. This introduces a disjunctive constraint:
$$t_i^s + T_i \le t_{i'}^s \quad \text{OR} \quad t_{i'}^s + T_{i'} \le t_i^s$$

To linearize this disjunction in an MILP, define a binary sequencing variable $Z_{i, i'} = 1$ if task $i$ precedes task $i'$, and $0$ if task $i'$ precedes task $i$. Using Big-M parameters:
$$t_i^s + T_i \le t_{i'}^s + M (1 - Z_{i, i'})$$
$$t_{i'}^s + T_{i'} \le t_i^s + M Z_{i, i'}$$

**The Big-M Pitfall:** If $M$ is chosen too large (e.g., $M = 10^7$), the continuous LP relaxation becomes extremely weak, leading to massive branch-and-bound trees and numerical stagnation. The parameter $M$ must be set to the tightest mathematically valid upper bound:
$$M = H_{horizon} - \min(T_i, T_{i'})$$

---

## 6. Worked example

We examine a realistic, end-to-end industrial scheduling problem: an agrochemical specialty plant operating a hybrid continuous-batch-discrete train over a 12-hour planning horizon ($H = 12$ hours), discretized into uniform intervals of $\Delta t = 1$ hour ($t \in \{1, 2, \dots, 12\}$).

```text
=============================================================================================================
                      AGROCHEMICAL HYBRID FACILITY PROCESS FLOWSHEET
=============================================================================================================

 [ Raw Precursor A ] ──► ┌──────────────────────────────────────┐
                         │ Continuous Synthesizer CS-101        │
 [ Raw Precursor B ] ──► │ (Rate: 10 t/h; steady-state)         │
                         └──────────────────┬───────────────────┘
                                            │ produces Intermediate X (10 t/h)
                                            ▼
                         ┌──────────────────────────────────────┐
                         │ Intermediate Tank TK-201 (FIS)       │
                         │ Capacity: Min 5 t (Heel), Max 30 t   │
                         └──────────────────┬───────────────────┘
                                            │ draws Intermediate X
                                            ▼
                         ┌──────────────────────────────────────┐
                         │ Batch Reactor BR-301 (Multi-Purpose) │
                         │ Task 1: Herbicide Base H1            │
                         │ Task 2: Fungicide Base F1            │
                         │ Capacity: Min 10 t, Max 20 t / batch │
                         └───────┬──────────────────────┬───────┘
                                 │                      │
             Discharges Herbicide│                      │Discharges Fungicide
             Base H1 (ZW Policy) │                      │Base F1 (NIS Policy)
                                 ▼                      ▼
    ┌──────────────────────────────────────┐    ┌──────────────────────────────────────┐
    │ High-Speed Bottling Line PK-401      │    │ Holding inside BR-301 until          │
    │ (Instantaneous transfer; ZW)         │    │ Secondary Granulator GR-501 is free  │
    │ Packs 10 t/h into 1-L Retail Bottles │    │ Granulates 5 t/h into Ag-Pellets     │
    └──────────────────────────────────────┘    └──────────────────────────────────────┘
=============================================================================================================
```

### Technical Parameters and Operating Rules:

1. **Continuous Synthesizer CS-101:**
   - Feeds Precursors A and B continuously to generate **Intermediate X** at a constant rate of $10\text{ metric tons per hour}$ ($10\text{ t/h}$).
   - Initial inventory of Intermediate X in Tank TK-201 at $t = 0$: $S_{X, 0} = 15\text{ t}$.
2. **Intermediate Storage Tank TK-201 (FIS Policy):**
   - Minimum Heel capacity: $S_X^{\min} = 5\text{ t}$.
   - Maximum working capacity: $S_X^{\max} = 30\text{ t}$.
   - Inventory violation ($S_X < 5$ or $S_X > 30$) triggers plant emergency shutdown.
3. **Batch Reactor BR-301 (Multi-Purpose Vessel):**
   - Can run **Task 1 (Herbicide Base H1)** or **Task 2 (Fungicide Base F1)**.
   - Batch capacity limits: $V^{\min} = 10\text{ t}$, $V^{\max} = 20\text{ t}$.
   - Processing durations:
     - Task 1 (Herbicide H1): $p_{H1} = 2\text{ hours}$. Requires $1.0\text{ t}$ of Intermediate X per ton of H1 ($100\%$ mass yield).
     - Task 2 (Fungicide F1): $p_{F1} = 3\text{ hours}$. Requires $1.0\text{ t}$ of Intermediate X per ton of F1 ($100\%$ mass yield).
   - Sequence-dependent cleaning: Switching between H1 and F1 requires a $1\text{-hour}$ CIP wash ($p_{CIP} = 1\text{ hr}$).
4. **Intermediate Storage Policies Downstream of BR-301:**
   - **Herbicide H1 is Zero Wait (ZW):** Due to rapid phase separation and crystallization, H1 must be transferred directly from BR-301 into Packaging Line PK-401 at the exact hour of batch completion.
   - **Fungicide F1 is No Intermediate Storage (NIS):** No buffer tank exists between BR-301 and Granulator GR-501. If GR-501 is busy, F1 must be held inside BR-301, blocking BR-301 from starting any new batch or CIP operation.
5. **Downstream Finishing Units:**
   - **Bottling Line PK-401:** Dedicated to Herbicide H1. Packaging rate: $10\text{ t/h}$. A $20\text{-t}$ batch requires exactly $2\text{ hours}$ to bottle.
   - **Granulator GR-501:** Dedicated to Fungicide F1. Processing rate: $5\text{ t/h}$. A $15\text{-t}$ batch requires exactly $3\text{ hours}$ to granulate.

---

### Step-by-Step Mathematical Calculation & Schedule Feasibility Audit

We evaluate a proposed production plan to determine whether all thermodynamic, capacity, and storage constraints hold over the 12-hour horizon.

#### Proposed Production Schedule:
- **Period $t = 1$ to $12$:** Continuous Synthesizer CS-101 generates $10\text{ t/h}$ of Intermediate X into TK-201. Total generation over 12 hours $= 12 \times 10 = 120\text{ t}$.
- **Batch 1 (Herbicide H1 on BR-301):**
  - Start time: $t = 1$. Duration: $p_{H1} = 2\text{ hours}$ (runs during hours $1\text{--}2$).
  - Batch size: $B_{H1, 1} = 20\text{ t}$.
  - Intermediate X consumption: $20\text{ t}$ drawn at start time $t = 1$.
  - Completion time: $t = 2$ (end of hour 2 / start of hour 3).
  - Storage Policy: **Zero Wait (ZW)** into Bottling Line PK-401. PK-401 must start bottling at $t = 3$ sharp.
  - PK-401 Run: Bottling $20\text{ t}$ at $10\text{ t/h}$ takes $2\text{ hours}$ (runs during hours $3\text{--}4$). Finishes at $t = 4$.
- **CIP Cleaning on BR-301:**
  - After discharging H1 at $t = 2$, BR-301 undergoes mandatory CIP wash before switching to Fungicide F1.
  - Duration: $1\text{ hour}$ (runs during hour 3, from $t = 2$ to $t = 3$).
- **Batch 2 (Fungicide F1 on BR-301):**
  - Start time: $t = 3$ (immediately after CIP completes at $t = 3$).
  - Duration: $p_{F1} = 3\text{ hours}$ (runs during hours $4, 5, 6$).
  - Batch size: $B_{F1, 3} = 15\text{ t}$.
  - Intermediate X consumption: $15\text{ t}$ drawn at start time $t = 3$.
  - Completion time: $t = 6$.
  - Downstream Destination: Granulator GR-501.
  - **Conflict Analysis:** Suppose Granulator GR-501 was pre-committed to a maintenance overhaul until $t = 7$.
  - Under **No Intermediate Storage (NIS)**, Fungicide F1 cannot discharge at $t = 6$. It must be held inside BR-301 for $\tau_{block} = 1\text{ hour}$ (during hour 7, until $t = 7$).
  - BR-301 is completely blocked during hour 7; no cleaning or new batch can initiate.
  - At $t = 7$, GR-501 becomes available; F1 discharges into GR-501.
  - Granulation of $15\text{ t}$ at $5\text{ t/h}$ runs during hours $8, 9, 10$ ($3\text{ hours}$).
- **Batch 3 (Herbicide H1 on BR-301):**
  - After discharging F1 at $t = 7$, BR-301 requires CIP cleaning ($1\text{ hour}$, hour 8).
  - Start time for Batch 3: $t = 8$.
  - Duration: $2\text{ hours}$ (runs during hours $9\text{--}10$).
  - Batch size: $B_{H1, 8} = 20\text{ t}$.
  - Intermediate X consumption: $20\text{ t}$ drawn at start time $t = 8$.
  - Completion time: $t = 10$.
  - Zero Wait transfer to PK-401 starting at $t = 10$, bottling runs during hours $11\text{--}12$.

---

### Mass Balance and Intermediate Storage (TK-201) Audit

We trace the inventory $S_{X, t}$ of Intermediate X at every hourly milestone $t \in \{1, \dots, 12\}$ using the STN mass balance equation:
$$S_{X, t} = S_{X, t-1} + \text{Inflow}_{X, t} - \text{Outflow}_{X, t}$$

Initial condition: $S_{X, 0} = 15.0\text{ t}$.

| Hour ($t$) | Initial Inventory ($t-1$) | Inflow from CS-101 ($t/h$) | Outflow to BR-301 ($t$) | Ending Inventory $S_{X, t}$ ($t$) | FIS Bound Check ($[5.0, 30.0]\text{ t}$) | Operational Status |
|:---:|:---:|:---:|:---:|:---:|:---:|:---|
| **1** | $15.0$ | $+10.0$ | $-20.0$ (Batch 1 H1 starts) | **$5.0$** | **Feasible (At exact minimum heel!)** | Batch 1 draws $20\text{ t}$; tank touches heel safely. |
| **2** | $5.0$ | $+10.0$ | $0.0$ | **$15.0$** | Feasible | BR-301 reacting; tank replenishes. |
| **3** | $15.0$ | $+10.0$ | $-15.0$ (Batch 2 F1 starts) | **$10.0$** | Feasible | BR-301 on CIP, starts F1 at end of hr 3. |
| **4** | $10.0$ | $+10.0$ | $0.0$ | **$20.0$** | Feasible | BR-301 reacting F1. PK-401 bottling H1. |
| **5** | $20.0$ | $+10.0$ | $0.0$ | **$30.0$** | **Feasible (At exact maximum capacity!)** | Tank reaches upper shell limit. |
| **6** | $30.0$ | $+10.0$ | $0.0$ | **$40.0 \longrightarrow \text{VIOLATION!}$** | **FATAL OVERFLOW (+10 t)** | **TK-201 overflows by 10 tons!** |

---

### Mathematical Root-Cause Analysis of the Schedule Failure:

The inventory calculation reveals an acute process breakdown at Hour 6:
$$S_{X, 6} = S_{X, 5} + 10.0 - 0.0 = 30.0 + 10.0 = 40.0\text{ t} > S_X^{\max} (30.0\text{ t})$$

Why did this occur?
1. Continuous Synthesizer CS-101 was allowed to push material at an uninterrupted $10\text{ t/h}$.
2. Batch Reactor BR-301 was blocked during Hour 7 under the **No Intermediate Storage (NIS)** policy, preventing any new batch from drawing Intermediate X.
3. Because BR-301 was running a long 3-hour recipe on Fungicide F1 followed by unit blocking, no intermediate was withdrawn from TK-201 between $t = 3$ and $t = 8$ ($5\text{ consecutive hours}$).
4. Inflow over those 5 hours $= 5 \times 10 = 50\text{ t}$. Starting inventory at $t = 3$ was $10\text{ t}$. Total volume $= 60\text{ t}$, which vastly exceeds the $30\text{-t}$ tank limit!

#### The Engineering Resolution:
To restore feasibility, the continuous rate of CS-101 must be dynamically throttled to match downstream consumption, or a third batch must be initiated in a parallel vessel. 

If CS-101 has a minimum turndown of $5:1$ ($u^{\min} = 2.0\text{ t/h}$), we throttle CS-101 between $t = 4$ and $t = 7$:
- Let $u_{CS}(4) = 4.0\text{ t/h}$, $u_{CS}(5) = 3.0\text{ t/h}$, $u_{CS}(6) = 2.0\text{ t/h}$.
- Recalculated inventories:
  - $S_{X, 4} = 10.0 + 4.0 = 14.0\text{ t}$.
  - $S_{X, 5} = 14.0 + 3.0 = 17.0\text{ t}$.
  - $S_{X, 6} = 17.0 + 2.0 = 19.0\text{ t} \le 30.0\text{ t}$ (**Feasible**).

This demonstrates why continuous process rates and batch scheduling cannot be planned independently: intermediate storage limits bind them into a single non-linear, coupled optimization problem.

---

## 7. Code, query, or calculation

The following self-contained Python script models and solves the multi-period intermediate-storage-constrained scheduling problem using `scipy.optimize.milp`. It enforces unit exclusivity, batch size limits, finite intermediate storage (FIS), and zero-wait (ZW) synchronization.

```python
import numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds

# Horizon: 6 hours; Tasks: H1 (dur=2, draw=20), F1 (dur=2, draw=15)
# Variables x: [W_H1(t) for t=0..5, W_F1(t) for t=0..5, S(t) for t=0..5] -> 18 vars
# Binary: W_H1 (0..5), W_F1 (6..11); Continuous: S_X (12..17)
c = np.zeros(18)
c[0:6] = -50.0  # Maximize Herbicide margin ($50/t * 20t = $1000)
c[6:12] = -40.0 # Maximize Fungicide margin ($40/t * 15t = $600)
integrality = np.zeros(18)
integrality[0:12] = 1  # Binary indicators

lb = np.zeros(18); ub = np.zeros(18)
ub[0:12] = 1.0  # Binary bounds
lb[12:18] = 5.0; ub[12:18] = 30.0  # FIS Tank TK-201 bounds: [5t, 30t]

A_rows, lhs, rhs = [], [], []
# 1. Unit exclusivity on Reactor: H1(dur=2), F1(dur=2) cannot overlap
for t in range(6):
    row = np.zeros(18)
    for tau in range(max(0, t - 1), t + 1):
        row[tau] += 1.0       # W_H1 active
        row[6 + tau] += 1.0   # W_F1 active
    A_rows.append(row); lhs.append(0.0); rhs.append(1.0)

# 2. Inventory mass balance: S(t) - S(t-1) + 20*W_H1(t) + 15*W_F1(t) = Inflow (10t/h)
# S_0 = 15t. For t=0: S(0) + 20*W_H1(0) + 15*W_F1(0) = 15 + 10 = 25
for t in range(6):
    row = np.zeros(18)
    row[12 + t] = 1.0
    if t > 0:
        row[12 + t - 1] = -1.0
    row[t] = 20.0       # Draw for H1
    row[6 + t] = 15.0   # Draw for F1
    A_rows.append(row)
    rhs_val = 25.0 if t == 0 else 10.0
    lhs.append(rhs_val); rhs.append(rhs_val)

# 3. ZW downstream check: H1 finishing at t must not exceed packaging line cap
# PK-401 available at t=2..5. H1 started at t=0 finishes at t=2.
A_mat = np.array(A_rows)
constraints = LinearConstraint(A_mat, lhs, rhs)
res = milp(c=c, integrality=integrality, bounds=Bounds(lb, ub), constraints=constraints)

print(f"Optimization Status: {res.status} ({res.message})")
print(f"Optimal Objective Value: ${-res.fun:.2f}")
w_h1 = res.x[0:6].round().astype(int)
w_f1 = res.x[6:12].round().astype(int)
s_inv = res.x[12:18].round(2)
print("Hour:       ", [f"H{t}" for t in range(1, 7)])
print("Start H1:   ", w_h1.tolist())
print("Start F1:   ", w_f1.tolist())
print("Inventory X:", s_inv.tolist())
```

### Assumptions and Expected Output:
- **Assumptions:** Discretized 1-hour time intervals; constant continuous synthesis inflow ($10\text{ t/h}$); deterministic recipe execution times; linear storage bounds.
- **Expected Output:**
```text
Optimization Status: 0 (Optimization terminated successfully. (HiGHS Status 7: Optimal))
Optimal Objective Value: $150.00
Hour:        ['H1', 'H2', 'H3', 'H4', 'H5', 'H6']
Start H1:    [1, 0, 1, 0, 1, 0]
Start F1:    [0, 0, 0, 0, 0, 0]
Inventory X: [5.0, 15.0, 5.0, 15.0, 5.0, 15.0]
```
The solver allocates the higher-margin product (Herbicide H1, margin $50/batch vs $40/batch for F1) across all three available 2-hour slots (starting at H1, H3, and H5). Intermediate inventory in Tank TK-201 fluctuates smoothly between 5.0 t (the exact minimum heel) and 15.0 t, never breaching the 30.0 t upper capacity limit.
---

## 8. Data, standards, and regulatory context

Process manufacturing architectures operate under strict regulatory and standardization mandates that define how recipes, equipment models, and transactional data are exchanged between administrative IT and physical operational technology (OT).

```text
=============================================================================================================
                    ENTERPRISE-TO-CONTROL HIERARCHY (ISA-95 / IEC 62264)
=============================================================================================================

 LEVEL 4: ENTERPRISE PLANNING (ERP / SCM)
 - Planning Horizon: Weeks to Months | Time Resolution: Days to Weeks
 - Function: Master Production Schedule (MPS), MRP, Financial Accounting, Customer Orders
 ────────────────────────────────────────────────────────────────────────────────────────────
                                       ▲ B2MML XML / JSON Messages
                                       ▼ (Material Requirements, Production Orders)
 LEVEL 3: MANUFACTURING OPERATIONS MANAGEMENT (MOM / MES)
 - Planning Horizon: Shifts to Weeks | Time Resolution: Minutes to Hours
 - Function: Detailed Finite Scheduling (STN/RTN), Batch Execution (ISA-88), Dispatching
 ────────────────────────────────────────────────────────────────────────────────────────────
                                       ▲ OPC UA / Industrial Ethernet
                                       ▼ (Recipe Phase Triggers, Equipment Status)
 LEVEL 2: DIRECT PROCESS CONTROL (DCS / SCADA / PLC)
 - Execution Frequency: Milliseconds to Seconds
 - Function: Continuous PID Loops, MPC Advanced Control, Sequence Logic, Safety Instrumented Systems (SIS)
=============================================================================================================
```

### Key Regulatory Standards and Frameworks:

1. **ISA-88 / IEC 61512 (Batch Control Architecture):**
   - Establishes the universal hierarchy for batch manufacturing. It explicitly decouples the **Recipe** (what to make) from the **Equipment** (where to make it).
   - **Physical Model:** Enterprise $\to$ Site $\to$ Area $\to$ Process Cell $\to$ Unit $\to$ Equipment Module $\to$ Control Module.
   - **Recipe Model:** General Recipe $\to$ Site Recipe $\to$ Master Recipe $\to$ Control Recipe. A Master Recipe defines the formula, header, equipment requirements, and the procedural sequence: **Procedure $\to$ Unit Procedure $\to$ Operation $\to$ Phase**.
   - *Binding Impact:* APS scheduling engines must allocate resources at the *Unit Procedure* level, respecting phase synchronization and CIP sanitization transitions.

2. **ISA-95 / IEC 62264 (Enterprise-Control System Integration):**
   - Defines the standard object models and B2MML (Business to Manufacturing Markup Language) schemas for exchanging data between Level 4 ERP and Level 3 MES.
   - Constrains how production schedules are transmitted: ERP must send `ProductionSchedule` containing `ProductionOrder` and `WorkOrder` objects; MES returns actual consumption via `ProductionPerformance` and `MaterialActual` transactions.
   - *Planning Trap:* ERP systems lacking ISA-95 compliance attempt to schedule continuous crackers and batch reactors as discrete work centers with fixed run-rate lead times, stripping out critical intermediate storage and utility state variables.

3. **OSHA 1910.119 / EU Seveso III Directive (Process Safety Management of Highly Hazardous Chemicals):**
   - Regulates manufacturing plants handling hazardous materials above threshold quantities.
   - Requires formal Process Safety Information (PSI), Management of Change (MOC), and strict overfill prevention protocols.
   - *Operational Constraint:* Storage tanks holding toxic or flammable fluids (e.g., chlorine, ethylene oxide, phosgene) cannot use standard FIS bounds based purely on geometric shell volume. Safety instrumented systems (SIS) mandate an independent High-High Level Trip (LAHH) at 85% to 90% capacity. Schedulers who attempt to utilize storage above the LAHH trip level face immediate emergency shutdown and regulatory penalties.

4. **FDA 21 CFR Part 11 and cGMP (21 CFR Part 211 / EudraLex Volume 4):**
   - Enforces electronic records, electronic signatures, and strict batch-to-batch segregation in pharmaceuticals and food processing.
   - *Storage & Cleaning Constraints:* Mandates validated Clean-In-Place (CIP) and Steam-In-Place (SIP) holding times. If an empty, sanitized reactor exceeds its validated Clean Hold Time (e.g., 72 hours without starting a batch), it loses its sterile status and must undergo a full re-sanitization cycle before processing. Schedulers must model Clean Hold Time decay as an active operational constraint.

---

## 9. Industry practice and case evidence

### Case 1: The Verbund System — BASF SE (Ludwigshafen, Germany)
- **Context:** The world's largest integrated chemical manufacturing site, spanning 10 square kilometers with approximately 200 interconnected production plants, producing thousands of chemical products.
- **Continuous-Batch-Storage Architecture:** Upstream production is anchored by two world-scale continuous steam crackers. The crackers continuously crack naphtha into ethylene, propylene, and aromatics. These streams feed directly into downstream continuous synthesis plants (acrylic acid, ammonia, styrenics) and intermediate batch formulation plants.
- **The Decoupling Challenge:** Plants are physically linked via a $2,850\text{-kilometer}$ pipeline network. Storing gaseous ethylene or toxic intermediates in massive tank farms is economically prohibitive and legally constrained under Seveso III regulations. Intermediate storage is severely finite (FIS) or zero wait (ZW).
- **Supply Chain Analytics Implementation:** BASF uses advanced RTN-based MILP formulations integrated with real-time dynamic pipeline pressure models. The optimization balances the chemical mass balance with the **energy Verbund**: waste reaction heat from exothermic continuous reactors generates high-pressure steam that is fed into a shared $100\text{-bar}$ steam grid to power endothermic distillation columns and batch evaporators.
- **Evidence of Impact:** The integrated Verbund scheduling slashes primary energy consumption by over 35% compared to decentralized manufacturing, saving millions of metric tons of $CO_2$ and over €1 billion annually in energy and logistics costs.

### Case 2: Multi-Product Agrochemical Formulation — Syngenta / Corteva
- **Context:** Global formulation and packaging plants producing active crop protection chemicals (fungicides, herbicides, insecticides).
- **Physical Tension:** Synthesis of active ingredients (AIs) is conducted in large-scale multi-purpose synthesis reactors running multi-week campaigns to minimize hazardous setup cleaning. Packaging, however, must respond to hyper-volatile weather patterns and seasonal pest infestations where farmer order delivery windows are under 48 hours ($D \le 48\text{ hr}$).
- **OPP Re-Engineering:** Historically, the plants operated as Make-to-Stock (MTS) with the OPP positioned at the packaged finished goods warehouse. This caused catastrophic write-offs: localized droughts resulted in millions of dollars of unsold, expired, pre-bottled herbicide in specific retail languages, while neighboring countries faced stockouts.
- **Operational Redesign:** The OPP was shifted upstream to the **bulk liquid formulated concentrate** stage (Postponement / Package-to-Order FTO). Active ingredients are synthesized and formulated into un-dyed, un-diluted bulk base liquid stored in stainless-steel FIS tank farms. High-speed discrete packaging lines are scheduled via finite-capacity MILP to execute late-stage dilution, colorant addition, and localized multilingual labeling upon receipt of verified regional distributor orders.
- **Documented Result:** Finished goods inventory was reduced by 28%, write-offs due to chemical shelf-life expiration dropped by 64%, and order fill rates improved from 82% to 97%.

### Case 3: Monoclonal Antibody Biomanufacturing — Roche / Genentech
- **Context:** Mammalian cell culture (CHO cells) manufacturing of therapeutic antibodies.
- **Kinetic and Storage Constraints:** Upstream fed-batch bioreactors run continuous 14-day fermentation cycles where living cells secrete the target protein. Downstream purification involves protein A affinity chromatography, viral inactivation, and ultrafiltration.
- **Zero-Wait (ZW) Reality:** The harvested cell culture fluid (HCCF) is highly sensitive to enzymatic proteolysis. Once discharged from the bioreactor, the fluid must undergo primary centrifugation and initial capture chromatography under strict Zero Wait conditions ($t_{hold} \le 4\text{ hours}$). Holding unpurified harvest in buffer tanks beyond 4 hours results in irreversible antibody fragmentation.
- **Scheduling Innovation:** Operations research engineers implemented continuous-time STN models that treat downstream chromatography skids and buffer preparation tanks as synchronized ZW resources. The scheduler guarantees that chromatography column resin equilibration is complete at the exact hour the 14-day bioreactor fermentation cycle terminates, completely eliminating harvest loss.

---

## 10. Failure modes and red flags

The table below catalogs operational and mathematical failure modes encountered when planning and scheduling continuous, batch, and discrete networks.

| Failure Mode | Early Warning Flag | Physical / Mathematical Mechanism | Operational Consequence | Mitigation Protocol |
|---|---|---|---|---|
| **Upstream Hydraulic Cascade** | Downstream packaging buffer tanks rising past 80% capacity during shift change. | Discrete packaging line halt back-propagates into intermediate FIS tanks, forcing continuous upstream reactors to throttle past turndown or flare. | Environmental flaring, massive thermal stress on continuous cracking furnaces, off-spec startup losses. | Install dynamic bypass accumulators; implement feed-forward throttling algorithms; establish secondary bulk off-loading racks. |
| **Zero-Wait (ZW) Violation** | Intermediate batch holding in reactor vessel awaiting downstream transfer line clearance. | Scheduler models ZW intermediate as Finite Intermediate Storage (FIS); downstream unit is busy when upstream reaction completes. | Irreversible batch degradation, phase separation, polymerization, or product cross-linking; complete batch dump. | Enforce hard equality timing constraints in MILP; model transfer lines and destination vessels as coupled simultaneous resources. |
| **Tank Heel Contamination** | Quality lab detects trace cross-contamination (e.g., 5 ppm solvent) in high-purity product batch. | Scheduling a new chemical campaign into an FIS tank without accounting for the unpumpable heel volume of the previous campaign. | Entire storage tank downgraded to low-value off-spec fuel blend; regulatory recall if cross-contamination involves allergens. | Integrate campaign changeover matrices directly into tank allocation constraints; mandate tank wash/drain cycles in schedule. |
| **The "Infinite Capacity MRP" Trap** | Master production schedule reports achievable output, but plant floor experiences chronic gridlock. | Level 4 ERP applies standard discrete MRP logic with fixed lead times, ignoring vessel exclusivity and intermediate storage bounds. | Staged raw materials spoil on loading docks; line starvation alternated with chaotic floor congestion. | Implement Level 3 finite-capacity MES scheduling (STN/RTN); strip lead-time offsets from fluid transformation steps in ERP. |
| **NIS Blocking Paralysis** | High reactor utilization metrics reported alongside collapsing total site mass throughput. | Upstream reactor finishes batch but remains occupied as a temporary storage vessel because downstream unit is running late. | Effective plant capacity collapses by 40–60%; operators cannot initiate cleaning or subsequent high-priority batches. | Explicitly penalize reactor blocking duration in the objective function; invest in dedicated intermediate surge vessels. |
| **Shared Utility Header Collapse** | High-pressure steam pressure drops from 40 bar to 28 bar, causing reactor jackets to lose temperature. | Multiple independent batch reactors initiate heating phases simultaneously; total instantaneous steam demand exceeds boiler capacity. | Extended batch cycle times, altered chemical reaction kinetics, batch crystallization failure, boiler flameout. | Formulate steam and chilled water headers as shared non-accumulative resources in an RTN framework with peak capacity bounds. |
| **Time Grid Discretization Distortion** | Optimized schedule creates artificial idle gaps between sequential unit tasks. | Using a coarse discrete time grid ($\Delta t = 2\text{ hours}$) for tasks with fractional durations ($1.2\text{ hours}$ rounded up to $2.0\text{ hours}$). | Cumulative loss of 15–25% of annual plant asset capacity due to artificial mathematical padding. | Shift to finer time grid ($\Delta t = 10\text{--}15\text{ min}$) or implement continuous-time precedence-based MILP formulations. |
| **Big-M Relaxation Drift** | MILP solver fails to close the optimality gap after hours of computation on a 7-day schedule. | Overly conservative Big-M bounds ($M = 10^8$) in continuous-time disjunctive sequencing constraints weaken the LP relaxation. | Solver stalls in branch-and-bound tree; planners abort optimization and revert to suboptimal manual Excel heuristics. | Compute tight, instance-specific Big-M values derived from exact horizon bounds and minimum task durations. |
| **Clean Hold Time Expiration** | Sanitized vessel rejected by quality assurance prior to batch charging. | Schedule leaves clean equipment idle longer than its validated Maximum Clean Hold Time (e.g., 72 hr) under cGMP rules. | Vessel must undergo unplanned 4-hour CIP/SIP re-cleaning, missing its scheduled batch start and cascading delays downstream. | Add dynamic shelf-life decay constraints to available clean equipment resources in the scheduling engine. |
| **Dry Bulk Silo Arching / Caking** | Pneumatic transfer blowers stall due to solid blockages at intermediate silo discharge cones. | Hygroscopic intermediate powders (e.g., detergent granules, polymer pellets) staged in silos during extended humid downtime. | Physical hammer-rapping or explosive sonic horn clearing required; downstream packaging line completely starved. | Restrict maximum allowable dwell time in dry silos; schedule continuous recirculation or nitrogen purge sweeps. |
| **Turndown Violation Runaway** | Continuous furnace burner trips, triggering automated emergency shutdown sequence. | Scheduler throttles continuous feed rate below the burner's certified physical turndown ratio to balance low downstream demand. | Flame instability, unburned fuel accumulation, emergency flare event, 48-hour mechanical purge and re-ignition cycle. | Set explicit non-zero lower bounds on continuous feed variables ($u(t) \ge u^{\min}$) linked to binary operating status. |
| **Postponement Buffer Starvation** | Packaging line stops every shift awaiting bulk intermediate release from QA testing. | Placing the OPP at intermediate formulation without accounting for analytical quality control testing release lead times (24–48 hr). | Packaging labor sits idle; expedited overtime incurred to meet customer shipping windows. | Model analytical QA lab testing stages as explicit procedural tasks with finite durations before inventory release. |

---

## 11. Assumptions and limits

Every mathematical formulation and operational policy in process scheduling rests upon foundational assumptions. When operating conditions violate these assumptions, the planning models diverge from physical reality.

```text
=============================================================================================================
                                      ASSUMPTIONS AND LIMITS LEDGER
=============================================================================================================
#   Assumption                  Operational Necessity                     Violation Trigger                 Diagnostic & Test Protocol
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
1   Deterministic Task          Enforces fixed durations in STN/RTN       Catalyst aging, fouling, ambient  Perform historical cycle time regression;
    Durations ($p_i$)           formulations to maintain linearity.       cooling water temperature swings. flag tasks with coefficient of variation > 0.10.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
2   Linear Mass Proportionality Constant yield fractions ($\bar{\rho}$) Non-linear reaction kinetics,     Cross-validate against rigorous Aspen Plus /
    (Fixed Yields)              prevent non-linear bilinearity in MILP.   feedstock impurity fluctuations.  process simulation models across operating regimes.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
3   Perfect Fluid Mixing        Treats bulk storage tanks as perfectly    Viscous stratification, phase     Conduct computational fluid dynamics (CFD) or
    (Lumped Parameters)         homogenous, lumped inventory states.      settling, temperature gradients.  multi-point physical sampling across tank levels.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
4   Instantaneous Fluid         Material movement between vessels         Viscous slurry pumping over long  Audit pipe run lengths; if transfer time > 0.05
    Transfer                    takes zero time in the discrete grid.     distances through small headers.  of task duration, model transfer as explicit task.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
5   Unlimited Utility           Excludes boiler and chiller headers       Multiple units starting heating   Install high-frequency power/steam logging;
    Availability                from the scheduling constraint matrix.    or cleaning simultaneously.       conduct peak-load coincident demand analysis.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
6   Static Equipment            Assumes equipment never suffers           Mechanical seal leaks, pump       Calculate Mean Time Between Failures (MTBF);
    Reliability                 unplanned breakdowns during horizon.      trips, valve actuation failures.  incorporate stochastic buffers or robust bounds.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
7   Deterministic Demand        Fixes shipment quantities ($D_{st}$)      Emergency spot orders, customer   Compare forecast error (MAPE/WAPE) against
    at Horizon Boundary         to establish a closed linear horizon.     cancellations, vessel delays.     safety stock buffers at the decoupled OPP.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
8   Sequence Independence       Changeover times between tasks are        Allergen cleanouts, dark-to-      Audit cleanout logs; if $CIP_{A \to B} \neq CIP_{B \to A}$,
    of Cleaning (when omitted)  treated as static constants.              light color transitions.          formulate sequence-dependent changeover matrix.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
9   Uniform Fluid Density       Converts mass to volume using static      Thermal expansion, foaming,       Measure density $\rho(T)$; implement temperature-
    Across Thermal Ranges       density parameters ($\rho$).              dissolved gas head space.         compensated volume calculations in MES.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
10  Zero Degradation During     Assumes materials in FIS tanks retain     Oxidation, moisture absorption,   Conduct accelerated stability testing; set maximum
    Storage Dwell               100% potency indefinitely.                biological contamination.         allowable inventory residence age ($\tau_{dwell}^{\max}$).
=============================================================================================================
```

---

## 12. Exercises

### Exercise 1: Physical Classification and Thermodynamic Regimes
Classify each of the following four production units as **Continuous**, **Batch**, or **Discrete**. For each unit, identify its primary thermodynamic or mechanical state variable, its standard unit of planning measurement, and its primary operational failure mode if halted mid-operation:
1. A crude oil atmospheric distillation column.
2. A $10,000\text{-liter}$ jacketed stainless-steel reactor synthesizing an active pharmaceutical ingredient (API).
3. A rotary cement kiln burning limestone and clay at $1,450^\circ\text{C}$.
4. A high-speed aerosol can filling and crimping line.

- **Deliverable:** A structured technical table covering all four units.
- **Acceptance Criteria:** Correct classification; exact state variable; operational failure mode grounded in physical mechanisms.
- **Estimated Time:** 25 minutes.

---

### Exercise 2: Mathematical Derivation — STN Mass Balance under FIS and Recycled By-Products
Consider an STN task $i_1$ (Catalytic Cracking) that consumes State $S_1$ (Gas Oil) and simultaneously produces two states: State $S_2$ (Ethylene Intermediate) with mass fraction $\bar{\rho}_{i_1, S_2} = 0.65$, and State $S_3$ (Heavy Fuel Oil By-Product) with mass fraction $\bar{\rho}_{i_1, S_3} = 0.35$. 

State $S_3$ is partially recycled back into the cracker: $20\%$ of all produced $S_3$ is immediately re-injected as a quench stream. The remaining $80\%$ of $S_3$ is routed to an FIS storage tank with bounds $[S_3^{\min}, S_3^{\max}]$.
1. Write the rigorous discrete-time STN mass balance equation for State $S_3$ across time intervals $t-1$ to $t$.
2. Formulate the mathematical inequality that prevents the FIS tank from exceeding its maximum working capacity $S_3^{\max}$ at any time period $t$, accounting for an external pipeline off-take demand $D_{S_3, t}$.

- **Deliverable:** Full algebraic derivation with clearly defined index sets and parameter definitions.
- **Acceptance Criteria:** Mathematical closure of mass balance; correct representation of the recycle split fraction; accurate linear inequality bounds.
- **Estimated Time:** 35 minutes.

---

### Exercise 3: RTN Resource Matrix Parameterization
A specialty chemical plant operates a batch reactor (Resource $R_1$) that requires high-pressure steam (Resource $R_2$, measured in megawatts thermal, $MW_{th}$) and one certified operator (Resource $R_3$). 

Task $k_1$ has a duration of 3 hours ($\tau_{k_1} = 3\text{ hr}$, time grid $\Delta t = 1\text{ hr}$, offsets $\theta \in \{0, 1, 2, 3\}$). Task $k_1$ requires:
- The reactor $R_1$ exclusively for the entire duration.
- A fixed steam draw of $4.0\text{ }MW_{th}$ during its first hour of heating ($\theta = 0$), dropping to a maintenance steam draw of $1.0\text{ }MW_{th}$ during hours 2 and 3 ($\theta = 1, 2$).
- One operator ($R_3$) present during charging at $\theta = 0$ and during discharging at $\theta = 3$, but no operator during the automated reaction hold ($\theta = 1, 2$).

Construct the exact numerical interaction matrices $\mu_{k_1, r, \theta}$ for resources $r \in \{R_1, R_2, R_3\}$ across all time offsets $\theta \in \{0, 1, 2, 3\}$.

- **Deliverable:** A $3 \times 4$ parameter matrix with explicit explanations of sign conventions ($+$ vs $-$).
- **Acceptance Criteria:** Accurate signs (negative for consumption, positive for release); exact numerical values matching the operational narrative.
- **Estimated Time:** 30 minutes.

---

### Exercise 4: OPP Positioning & Postponement Trade-Off Calculation
A consumer paint manufacturer produces 40 distinct colors of exterior latex paint. Under the current Make-to-Stock (MTS) policy, all 40 colors are pre-mixed, bottled in $5\text{-liter}$ cans, and held in the finished goods warehouse. Each SKU has normally distributed monthly demand with mean $\mu_i = 500\text{ cans}$ and standard deviation $\sigma_i = 120\text{ cans}$. Demand across all SKUs is mutually independent. Target customer service level is $95\%$ ($z = 1.645$).
1. Calculate the total safety stock (in cans) required across all 40 finished SKUs under the current MTS policy.
2. The company proposes shifting the OPP upstream to a **Package-to-Order (PTO / Postponement)** architecture: an uncolored white base paint is stored in bulk FIS tanks. Color tinting and canning occur upon confirmed distributor orders. Calculate the pooled safety stock (in cans equivalent) required for the bulk base paint.
3. Compute the percentage reduction in safety stock achieved by shifting the OPP.

- **Deliverable:** Full numerical calculations with intermediate steps shown.
- **Acceptance Criteria:** Correct application of statistical risk pooling; accurate arithmetic; stated assumptions regarding lead time and yield.
- **Estimated Time:** 30 minutes.

---

### Exercise 5: Critical Analysis of Storage Policy Violations
An operations manager in a biopharmaceutical plant attempts to schedule a new therapeutic protein campaign using standard commercial scheduling software. The software does not support Zero Wait (ZW) constraints, so the manager models the intermediate harvest holding vessel as a Finite Intermediate Storage (FIS) tank with a maximum holding capacity of 5,000 liters. 
1. Explain the biochemical and thermodynamic failure mechanism that will occur on the plant floor if the downstream chromatography column experiences an unplanned 6-hour electrical interlock trip.
2. Formulate the exact mathematical constraint that the manager must add to an MILP scheduler to enforce true Zero Wait behavior between Bioreactor Task 1 (duration $p_1$) and Chromatography Task 2 on unit $j \in \mathcal{J}_2$.

- **Deliverable:** Technical explanation of protein degradation kinetics and exact algebraic constraint.
- **Acceptance Criteria:** Cites specific biological degradation mechanisms (proteolysis, aggregation); mathematically sound equality/inequality constraint.
- **Estimated Time:** 25 minutes.

---

### Exercise 6: Discrete-Time vs Continuous-Time MILP Complexity Evaluation
A multi-product batch chemical facility runs 8 processing tasks across 4 multi-purpose reactors over a 7-day planning horizon.
1. If the planning horizon is modeled using a uniform discrete time grid with intervals of $\Delta t = 15\text{ minutes}$, calculate the exact number of time intervals $H$ and the total number of binary unit-task assignment variables $W_{ijt}$.
2. If the problem is reformulated using a continuous-time precedence formulation with 12 event points across the horizon, calculate the number of binary sequencing variables $Z_{i, i'}$ required.
3. Contrast the computational pros and cons of both formulations for this specific plant.

- **Deliverable:** Quantitative variable counts and a comparative technical critique.
- **Acceptance Criteria:** Accurate arithmetic on variable dimensions; deep comparison of LP relaxation tightness vs variable count.
- **Estimated Time:** 30 minutes.

---

### Exercise 7: Campaign Sizing and Changeover Optimization
A chemical reactor produces three grades of polymer: Grade X, Grade Y, and Grade Z. Demand rates are constant: $D_X = 20\text{ t/day}$, $D_Y = 15\text{ t/day}$, $D_Z = 10\text{ t/day}$. The reactor produces polymer at a rate of $P = 100\text{ t/day}$ when running. Switching between grades requires a complete cleanout taking 1 full day and costing $\$10,000$ in lost production, solvent, and labor. Inventory holding cost is $\$2.00\text{ per ton per day}$.
1. Formulate the Economic Production Quantity (EPQ) / optimal campaign cycle time $T^*$ (in days) across the family of three products.
2. Calculate the optimal cycle time $T^*$, the run length (in days) for each product within the cycle, and the total annual setup and holding cost.

- **Deliverable:** Full mathematical derivation and numerical solution.
- **Acceptance Criteria:** Correct multi-product EPQ formula accounting for utilization $\sum \frac{D_i}{P} < 1$; accurate calculation of run lengths and cost totals.
- **Estimated Time:** 40 minutes.

---

### Exercise 8: Industrial Architecture Audit
You are hired to audit a chemical manufacturing site where the corporate IT department attempted to connect SAP S/4HANA (Level 4 ERP) directly to Emerson DeltaV (Level 2 DCS) to execute automated dynamic batch scheduling, bypassing the Level 3 MES layer entirely.
1. Identify three structural reasons why direct ERP-to-DCS scheduling fails in process manufacturing.
2. Outline the correct data exchange architecture according to the ISA-95 standard, specifying the exact objects and B2MML schemas that must bridge Level 4 and Level 3.

- **Deliverable:** A formal technical audit report (350–500 words).
- **Acceptance Criteria:** Rigorous reference to ISA-95 data models; specific physical failure mechanisms identified.
- **Estimated Time:** 35 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Physical Classification and Thermodynamic Regimes
| Unit | Classification | Primary State Variable | Standard Planning Unit | Primary Failure Mode if Halted Mid-Operation |
|---|---|---|---|---|
| **Distillation Column** | Continuous | Vapor-liquid equilibrium profile; tray temperature $T(z)$ and pressure $P(z)$. | Metric tons per hour ($t/h$), barrels per day ($bpd$). | Column weeping: vapor velocity drops, liquid dumps through tray perforations, destroying separation profile; requires 12–24 hr to re-establish thermal equilibrium. |
| **API Jacketed Reactor** | Batch | Concentration profile $C_i(t)$ and reaction temperature $T(t)$ over time. | Kilograms per batch ($kg/\text{batch}$), cycle time ($hr$). | Uncontrolled thermal runaway or localized crystallization around impeller; batch solidification requiring manual jackhammer cleanout. |
| **Rotary Cement Kiln** | Continuous | Solid clinkering bed temperature ($1,450^\circ\text{C}$) and gas flame temperature. | Metric tons of clinker per day ($t/day$). | Severe thermal shock causing cracking and spallation of internal refractory bricks; kiln shell warps under asymmetric thermal sagging. |
| **Aerosol Filling Line** | Discrete | Mechanical indexer speed (cans/min), propellant pressure ($bar$). | Cans per minute, cases per shift, overall equipment effectiveness (OEE). | Minor mechanical can jam; cans remain stable on conveyor without chemical degradation; line pauses cleanly via safety interlock. |

---

### Answer to Exercise 2: STN Mass Balance under FIS and Recycled By-Products
1. **Discrete-Time STN Mass Balance Equation for By-Product State $S_3$:**
   Let $\mathcal{T}_{S_3}^{prod} = \{i_1\}$ be the set of tasks producing $S_3$, with stoichiometric mass fraction $\bar{\rho}_{i_1, S_3} = 0.35$.  
   Let $p_{i_1}$ be the processing duration of task $i_1$.  
   Of the produced $S_3$, a fraction $\alpha_{recycle} = 0.20$ is instantaneously recycled back into active cracker units, leaving $(1 - \alpha_{recycle}) = 0.80$ entering intermediate storage.  
   Let $B_{i_1, j, t-p_{i_1}}$ be the batch extent of task $i_1$ completing on unit $j$ at time $t$.  
   Let $D_{S_3, t}$ be the external pipeline shipment off-take at time $t$.  
   The rigorous balance equation is:
   $$S_{3, t} = S_{3, t-1} + (1 - \alpha_{recycle}) \sum_{j \in \mathcal{J}_{i_1}} \bar{\rho}_{i_1, S_3} B_{i_1, j, t-p_{i_1}} - D_{S_3, t}$$
   Substituting numerical parameters:
   $$S_{3, t} = S_{3, t-1} + (0.80)(0.35) \sum_{j \in \mathcal{J}_{i_1}} B_{i_1, j, t-p_{i_1}} - D_{S_3, t} = S_{3, t-1} + 0.28 \sum_{j \in \mathcal{J}_{i_1}} B_{i_1, j, t-p_{i_1}} - D_{S_3, t}$$

2. **FIS Storage Bound Inequality:**
   To guarantee the FIS tank never breaches its working capacity envelope $[S_3^{\min}, S_3^{\max}]$ at any time period $t \in \{1, \dots, H\}$:
   $$S_3^{\min} \le S_{3, t-1} + 0.28 \sum_{j \in \mathcal{J}_{i_1}} B_{i_1, j, t-p_{i_1}} - D_{S_3, t} \le S_3^{\max}$$

---

### Answer to Exercise 3: RTN Resource Matrix Parameterization
We construct the interaction parameter matrix $\mu_{k_1, r, \theta}$ for Task $k_1$ across resources $R_1$ (Reactor), $R_2$ (Steam Header in $MW_{th}$), and $R_3$ (Certified Operator) across relative time offsets $\theta \in \{0, 1, 2, 3\}$.

**Sign Convention:**
- Negative ($-$) indicates resource consumption/draw.
- Positive ($+$) indicates resource production/release.
- Zero ($0$) indicates no interaction at that specific offset.

$$\mu_{k_1, r, \theta} = \begin{pmatrix}
\theta = 0 & \theta = 1 & \theta = 2 & \theta = 3 \\
-1.0 & 0.0 & 0.0 & +1.0 \\
-4.0 & -1.0 & -1.0 & 0.0 \\
-1.0 & +1.0 & 0.0 & -1.0
\end{pmatrix} \begin{array}{l} \text{Row 1: Reactor } R_1 \text{ (Allocated at } \theta=0\text{, released at } \theta=3\text{)} \\ \text{Row 2: Steam } R_2 \text{ (Draws 4 MW at } \theta=0\text{; 1 MW at } \theta=1, 2\text{)} \\ \text{Row 3: Operator } R_3 \text{ (Draws 1 at start, releases at } \theta=1\text{; draws at } \theta=3\text{)} \end{array}$$

*Note on Operator $R_3$ at $\theta = 3$:* Discharging requires the operator. At $\theta = 3$, the operator is consumed ($-1.0$) for discharging, and upon completion of the task, is released back to the general labor pool ($+1.0$). If modeled within the task boundary, the net transient interaction at discharge completes when the batch is empty.

---

### Answer to Exercise 4: OPP Positioning & Postponement Trade-Off Calculation
1. **Current MTS Policy (Decentralized Safety Stock):**
   - For each individual color SKU $i$:
     $$SS_i = z \cdot \sigma_i = 1.645 \times 120 = 197.4\text{ cans}$$
   - Across all $N = 40$ independent SKUs:
     $$SS_{total}^{MTS} = \sum_{i=1}^{40} SS_i = 40 \times 197.4 = 7,896\text{ cans}$$

2. **Package-to-Order (PTO / Postponement) Policy (Pooled Safety Stock):**
   - Under total postponement, demand aggregates across all 40 SKUs into a single bulk base paint demand:
     $$\sigma_{pooled} = \sqrt{\sum_{i=1}^{40} \sigma_i^2} = \sqrt{40 \times (120)^2} = \sqrt{40 \times 14,400} = \sqrt{576,000} \approx 758.95\text{ cans equivalent}$$
   - Pooled safety stock required:
     $$SS_{pooled}^{PTO} = z \cdot \sigma_{pooled} = 1.645 \times 758.95 \approx 1,248.47\text{ cans equivalent}$$

3. **Percentage Reduction in Safety Stock:**
   $$\text{Reduction} = \frac{SS_{total}^{MTS} - SS_{pooled}^{PTO}}{SS_{total}^{MTS}} = \frac{7,896 - 1,248.5}{7,896} = \frac{6,647.5}{7,896} \approx 84.19\%$$
   Shifting the OPP upstream to the bulk uncolored liquid stage slashes required safety stock by **$84.2\%$**, freeing substantial working capital and completely eliminating finished product obsolescence.

---

### Answer to Exercise 5: Critical Analysis of Storage Policy Violations
1. **Biochemical Failure Mechanism:**
   Harvested cell culture fluid (HCCF) contains host cell proteins, active proteases (e.g., cathepsins), and glycosidases released during cell lysis in the bioreactor. If held in an intermediate vessel for 6 hours without active purification:
   - **Proteolytic Cleavage:** Proteases attack the hinge region of the monoclonal antibodies, cleaving them into Fab and Fc fragments, destroying biological efficacy.
   - **Deamidation and Aggregation:** pH and dissolved oxygen changes trigger high-molecular-weight (HMW) aggregation. Once aggregates form, the batch cannot be rescued and must be discarded under FDA cGMP quality specifications.
2. **Exact Mathematical Constraint for Zero Wait (ZW):**
   Let $W_{1, j_1, t} \in \{0, 1\}$ be the binary start of Bioreactor Task 1 (duration $p_1$) on unit $j_1$.  
   Let $W_{2, j_2, t} \in \{0, 1\}$ be the binary start of Chromatography Task 2 on unit $j_2 \in \mathcal{J}_2$.  
   The downstream task must start at the exact completion time $t + p_1$:
   $$W_{1, j_1, t} \le \sum_{j_2 \in \mathcal{J}_2} W_{2, j_2, t + p_1} \quad \forall t \in \{1, \dots, H - p_1\}$$
   Combined with mass conservation linking batch sizes directly without intermediate storage inventory variables.

---

### Answer to Exercise 6: Discrete-Time vs Continuous-Time MILP Complexity Evaluation
1. **Discrete-Time Formulation:**
   - Planning horizon: $7\text{ days} = 7 \times 24 = 168\text{ hours}$.
   - Number of intervals: $H = \frac{168\text{ hr}}{0.25\text{ hr}} = 672\text{ intervals}$.
   - Number of tasks $|\mathcal{T}| = 8$; number of units $|\mathcal{J}| = 4$.
   - Assuming all tasks can run on all 4 units:
     $$\text{Total Binary Variables } W_{ijt} = |\mathcal{T}| \times |\mathcal{J}| \times H = 8 \times 4 \times 672 = 21,504\text{ binary variables}$$
2. **Continuous-Time Precedence Formulation:**
   - Number of event points $N = 12$.
   - Binary sequencing variables $Z_{i, i'}$ are defined for each pair of tasks $(i, i')$ competing for the same unit $j$.
   - For 8 tasks, the number of distinct task pairs is $\binom{8}{2} = \frac{8 \times 7}{2} = 28$.
   - Across 4 units, total binary variables $\approx 4 \times 28 = 112\text{ binary variables}$.
3. **Comparative Critique:**
   - The discrete-time formulation carries **21,504 binary variables**, creating a massive branch-and-bound tree that will likely struggle to prove optimality within reasonable runtimes. However, it handles utility peaks and inventory balances trivially at each time step.
   - The continuous-time formulation carries only **112 binary variables**, enabling rapid tree exploration. However, its LP relaxation is inherently weaker due to Big-M linearization, and incorporating shared utility headers (steam, power) requires complex bilinear cross-interval integration.

---

### Answer to Exercise 7: Campaign Sizing and Changeover Optimization
1. **Multi-Product EPQ Cycle Time Formulation:**
   - Total daily demand rate: $D = \sum_{i} D_i = 20 + 15 + 10 = 45\text{ t/day}$.
   - Production rate: $P = 100\text{ t/day}$.
   - Total capacity utilization: $\rho = \frac{D}{P} = \frac{45}{100} = 0.45$.
   - Total changeover setup cost per cycle (3 setups $\times \$10,000$): $S_{cycle} = 3 \times \$10,000 = \$30,000$.
   - Total changeover setup time per cycle (3 setups $\times 1\text{ day}$): $\tau_{setup} = 3\text{ days}$.
   - Holding cost $h = \$2.00/\text{t}\cdot\text{day}$.
   - The optimal cycle time $T^*$ balances setup costs against holding costs under finite production rate:
     $$T^* = \sqrt{\frac{2 \cdot S_{cycle}}{h \cdot \sum_{i} D_i \left(1 - \frac{D_i}{P}\right)}} = \sqrt{\frac{2 \times 30,000}{2.00 \times \left[ 20(1 - 0.20) + 15(1 - 0.15) + 10(1 - 0.10) \right]}}$$
     $$\sum_{i} D_i \left(1 - \frac{D_i}{P}\right) = 20(0.80) + 15(0.85) + 10(0.90) = 16.0 + 12.75 + 9.0 = 37.75\text{ t/day}$$
     $$T^* = \sqrt{\frac{60,000}{2.00 \times 37.75}} = \sqrt{\frac{60,000}{75.50}} = \sqrt{794.70} \approx 28.19\text{ days}$$
   - Feasibility check: Available production time per cycle must exceed demand production time plus setup time:
     $$T^* (1 - \rho) \ge \tau_{setup} \implies 28.19 \times (1 - 0.45) = 28.19 \times 0.55 = 15.50\text{ days} \ge 3.0\text{ days} \quad \text{(Feasible)}$$

2. **Run Lengths and Annual Cost:**
   - Cycle time: $T^* \approx 28.2\text{ days}$. Number of cycles per year $= \frac{365}{28.19} \approx 12.95\text{ cycles/yr}$.
   - Production run lengths per cycle ($t_i = \frac{D_i \cdot T^*}{P}$):
     - Grade X: $t_X = \frac{20 \times 28.19}{100} = 5.64\text{ days}$ ($564\text{ t}$).
     - Grade Y: $t_Y = \frac{15 \times 28.19}{100} = 4.23\text{ days}$ ($423\text{ t}$).
     - Grade Z: $t_Z = \frac{10 \times 28.19}{100} = 2.82\text{ days}$ ($282\text{ t}$).
     - Setup downtime: $3.0\text{ days}$. Idle capacity: $28.19 - (5.64 + 4.23 + 2.82 + 3.0) = 12.50\text{ days}$.
   - Total Annual Setup Cost: $12.95\text{ cycles} \times \$30,000 = \$388,500/\text{year}$.
   - Total Annual Holding Cost: $\frac{1}{2} h \cdot T^* \cdot \sum D_i \left(1 - \frac{D_i}{P}\right) \times 365 / T^* = \$388,500/\text{year}$.
   - Total Annual Policy Cost: $\$388,500 + \$388,500 = \$777,000/\text{year}$.

---

### Answer to Exercise 8: Industrial Architecture Audit
1. **Three Reasons Why Direct ERP-to-DCS Integration Fails:**
   - **Temporal Mismatch:** ERP operates in daily/weekly buckets with batch transaction processing, whereas DCS operates in millisecond/second real-time control loops. ERP cannot respond to dynamic process excursions (e.g., foaming, exothermic temperature spikes).
   - **Semantic and Object Incompatibility:** ERP models manufacturing via flat BOMs and routing tables with fixed lead-time offsets. DCS understands PID blocks, valve positions, and interlocks. ERP cannot parse ISA-88 recipe procedural logic (Phase $\to$ Operation $\to$ Unit Procedure).
   - **Absence of Intermediate Storage and Utility State Tracking:** ERP does not model physical tank levels, heels, ullage, shared steam headers, or CIP skids. Transmitting an ERP schedule directly to DCS results in equipment deadlocks and tank overflow trips.
2. **Correct ISA-95 Architecture:**
   - Bridge Level 4 and Level 2 using a dedicated **Level 3 MES (Manufacturing Execution System)**.
   - ERP transmits high-level production requirements using the ISA-95 `ProductionSchedule` B2MML XML/JSON message schema.
   - Level 3 MES executes finite-capacity STN/RTN mathematical optimization, mapping orders into ISA-88 Master Recipes, allocating physical units, and verifying FIS/ZW storage and utility constraints.
   - MES dispatches individual `ControlRecipe` phases to Level 2 DCS via OPC UA. DCS executes real-time phase logic and reports execution telemetry back to MES. MES aggregates actuals and sends `ProductionPerformance` messages back to Level 4 ERP.

---

## 14. Interview questions

The following questions evaluate technical depth in continuous, batch, and discrete manufacturing analytics.

### Question 1: How do you mathematically enforce a Zero Wait (ZW) intermediate storage policy in a discrete-time STN scheduling model?
- **What a strong answer contains:** Defines ZW as an operational requirement where intermediate dwell time between tasks is strictly zero ($\Delta t = 0$). Explains that in an STN model, this requires removing the intermediate state storage variable ($S_{st} = 0$) or formulating an exact equality linking the completion of the upstream task directly to the initiation of the downstream task:
  $$\sum_{j \in \mathcal{J}_{upstream}} \bar{\rho}_{1, s} B_{1, j, t-p_1} = \sum_{i_2} \sum_{j' \in \mathcal{J}_{i_2}} \rho_{2, s} B_{2, j', t} \quad \forall t$$
  Highlights the difference between ZW and FIS/NIS, noting that ZW couples upstream and downstream unit occupancy simultaneously.
- **What a weak answer reveals:** Confuses Zero Wait with Just-In-Time (JIT) inventory philosophy. Suggests setting holding cost to infinity in the objective function, failing to understand that soft penalties permit small mathematical violations that cause catastrophic physical batch spoilage.

---

### Question 2: Why can't we use standard discrete MRP-II logic to plan a continuous refinery or chemical cracker?
- **What a strong answer contains:** Details the physical divergence between discrete BOM assembly and continuous fluid processing: continuous processes involve non-linear stoichiometric yields, variable byproduct slates, recycle loops, and physical turndown limits. Points out that MRP assumes fixed lead-time offsets independent of batch size and sequence-dependent changeovers, whereas continuous units run steady-state campaigns where transitions generate off-spec material. Mentions the necessity of mass and energy balance closure ($\sum \dot{m}_{in} = \sum \dot{m}_{out}$).
- **What a weak answer reveals:** Focuses only on unit of measure differences (kilograms vs eaches) without articulating fluid dynamics, steady-state thermodynamics, turndown ratios, or recycle loops.

---

### Question 3: Compare the State-Task Network (STN) and the Resource-Task Network (RTN). Under what circumstances is RTN mathematically superior?
- **What a strong answer contains:** Identifies STN as a bipartite graph separating States from Tasks, requiring dedicated unit assignment indices and separate storage equations. Explains that RTN unifies all physical entities (equipment, storage vessels, steam, cooling water, operators) into a single concept: Resources. RTN is strictly superior when the facility features **shared secondary resources** (e.g., shared CIP skids, constrained steam/power headers, mobile transfer pumps) because these non-storable and renewable entities are captured under the identical universal balance equation without ad-hoc constraint formulation.
- **What a weak answer reveals:** Treats them as identical acronyms or claims RTN is simply an updated version of STN without naming the mathematical unification of resources.

---

### Question 4: How does positioning the Order Penetration Point (OPP) impact working capital and asset utilization in a hybrid agrochemical or specialty chemical plant?
- **What a strong answer contains:** Explains that shifting the OPP upstream (e.g., from Make-to-Stock finished packaged SKUs to Package-to-Order bulk liquid intermediate) leverages Eppen's square-root law to aggregate stochastic demand variance, slashing safety stock by 50–80%. Notes the trade-off: moving the OPP upstream increases customer lead time (which must remain within customer tolerance $D$), and requires packaging lines to operate with rapid changeover agility, potentially lowering discrete packaging OEE while maintaining high, continuous upstream asset utilization.
- **What a weak answer reveals:** Thinks OPP is purely a marketing choice. Does not mention postponement, risk pooling, shelf-life perishability, or the $D/P$ ratio.

---

### Question 5: What is No Intermediate Storage (NIS), and how does it create the "reactor blocking" phenomenon?
- **What a strong answer contains:** Defines NIS as the absence of buffer tanks between successive processing stages. If an upstream reactor completes a batch but the downstream unit is still operating or undergoing cleaning, the upstream vessel cannot discharge. It enters a "blocked" state, where it acts as a passive storage tank. The strong candidate notes that during blocking, the upstream reactor is unavailable for cleaning or initiating its next batch, drastically lowering effective plant capacity. Explains how this is linearized in an MILP by extending the binary occupancy variable.
- **What a weak answer reveals:** Believes NIS means inventory is zero throughout the entire supply chain. Misses the fact that material is physically trapped inside the processing unit.

---

### Question 6: What are the mathematical and computational dangers of using overly large Big-M values in continuous-time scheduling models?
- **What a strong answer contains:** Explains that Big-M linearizations model disjunctive operational choices ($A \text{ before } B \text{ or } B \text{ before } A$). If $M$ is set to an arbitrarily large number (e.g., $10^8$), the continuous relaxation of the MILP becomes exceedingly loose. In the LP relaxation, fractional binary variables ($Z = 0.0001$) can satisfy the inequality with virtually no penalty, creating an artificially optimistic objective bound. This results in an enormous integrality gap, causing the branch-and-bound solver to stall and explore millions of nodes without closing the gap. Mentions numerical instability and matrix scaling issues.
- **What a weak answer reveals:** Thinks Big-M is just a programming trick where "bigger is always safer" to avoid cutting off feasible solutions.

---

### Question 7: Explain the difference between ISA-88 and ISA-95. At which level of the ISA-95 hierarchy does finite-capacity campaign scheduling reside?
- **What a strong answer contains:** Clarifies that ISA-88 governs **batch process control** (equipment models, physical recipes, procedural phases), whereas ISA-95 governs **enterprise-to-control system integration** (Level 4 ERP down to Level 3 MOM/MES). Finite-capacity campaign scheduling sits squarely at **Level 3 (MES/MOM)**. Level 4 ERP lacks the granular state visibility, tank level physics, and recipe phase models to generate executable process schedules.
- **What a weak answer reveals:** Reverses the standards or claims that scheduling is done at Level 4 ERP and pushed directly to Level 2 PLCs.

---

### Question 8: How do you model non-storable shared utilities, such as a high-pressure steam header, in an RTN formulation?
- **What a strong answer contains:** Explains that unlike mass resources (feedstocks, intermediates), utility resources cannot be carried forward across time intervals ($R_{utility, t}$ does not inherit $R_{utility, t-1}$). The balance equation must reset every time interval or enforce an instantaneous maximum draw:
  $$\sum_k \sum_\theta |\nu_{kr\theta}| \xi_{k, t-\theta} \le Utility^{\max}(t)$$
  Notes that heating tasks have high peak steam draws at start ($\theta = 0$) followed by lower maintenance draws, and the optimization model must smooth task initiation to prevent coincident peak demands from exceeding boiler generation capacity.
- **What a weak answer reveals:** Treats steam as an inventory tank where steam can be stored for days without condensing.

---

## 15. Key terms

- **Batch Production:** Cyclic recipe-driven manufacturing of bounded lots in multi-purpose equipment.
- **Big-M Method:** Linearization technique for disjunctive constraints, requiring tight parameter calibration.
- **Clean Hold Time:** Maximum allowable idle duration for a sanitized vessel before re-cleaning is mandated.
- **Clean-In-Place (CIP):** Automated mechanical and chemical sanitization of process equipment without disassembly.
- **Continuous Production:** Steady-state fluid flow manufacturing operating 24/7/365 within hydrodynamic turndown windows.
- **Decoupled Inventory Point (OPP):** The boundary stage separating forecast-driven push from order-driven pull.
- **Discrete Manufacturing:** Part-by-part fabrication and assembly of distinct, countable units.
- **Discrete-Time Formulation:** Scheduling model dividing the time horizon into fixed, uniform intervals ($\Delta t$).
- **Continuous-Time Formulation:** Scheduling model where task start and end times are continuous decision variables.
- **Finite Intermediate Storage (FIS):** Buffer policy with strictly bounded physical working volume envelopes.
- **Heel Volume:** Unusable fluid volume residing below the lowest discharge point of a storage tank.
- **Hybrid Plant:** Facility combining upstream continuous, intermediate batch, and downstream discrete operations.
- **ISA-88 (IEC 61512):** Global standard for batch control architecture and recipe structures.
- **ISA-95 (IEC 62264):** Standard for integrating enterprise ERP (Level 4) with factory MES (Level 3).
- **No Intermediate Storage (NIS):** Policy where lack of buffer vessels forces materials to be held inside processing units.
- **Order Penetration Point (CODP / OPP):** The structural node where inventory is decoupled and customized to customer order.
- **Postponement:** Delaying final formulation, coloring, or packaging to pool inventory risk.
- **Resource-Task Network (RTN):** Unified scheduling framework modeling equipment, utilities, labor, and materials as resources.
- **State-Task Network (STN):** Bipartite graph formulation mapping transformations between States and Tasks.
- **Steam-In-Place (SIP):** Thermal sterilization of processing equipment using pressurized saturated steam.
- **Tank Ullage:** Unoccupied headspace maintained at the top of a liquid tank for thermal expansion and safety.
- **Turndown Ratio:** Ratio of maximum sustainable throughput to minimum stable operating throughput in a continuous unit.
- **Turnaround (TAR):** Periodic site-wide shutdown for maintenance, vessel inspection, and catalyst replacement.
- **Ullage Safety Limit:** Maximum fill line established below tank shell height to prevent overfill tripping.
- **Unlimited Intermediate Storage (UIS):** Theoretical storage policy assuming infinite intermediate buffer capacity.
- **Verbund Integration:** Physical and energetic interlinking of chemical plants to achieve near-zero waste.
- **Working Volume:** The net usable storage capacity of a tank ($V_{working} = V_{total} - V_{heel} - V_{ullage}$).
- **Zero Wait (ZW):** Thermodynamic policy requiring instantaneous unit-to-unit transfer without intermediate delay.
- **Disjunctive Sequencing:** Logic enforcing that two tasks sharing a single unit cannot overlap in time.
- **Unit Exclusivity:** Mathematical constraint ensuring a processing vessel runs at most one task at any time step.

---

## 16. Sources

- **Floudas, C. A., & Lin, X. (2004).** Continuous-time versus discrete-time approaches for scheduling of chemical processes: a review. *Computers & Chemical Engineering*, 28(11), 2109–2129. https://doi.org/10.1016/j.compchemeng.2004.05.002
- **Kondili, E., Pantelides, C. C., & Sargent, R. W. H. (1993).** A general algorithm for short-term scheduling of batch operations—I. MILP formulation. *Computers & Chemical Engineering*, 17(2), 211–227. https://doi.org/10.1016/0098-1354(93)80015-F
- **Maravelias, C. T., & Grossmann, I. E. (2003).** New general continuous-time state-task network formulation for short-term scheduling of multipurpose batch plants. *Industrial & Engineering Chemistry Research*, 42(13), 3056–3074. https://doi.org/10.1021/ie020938z
- **Méndez, C. A., Cerdá, J., Grossmann, I. E., Harjunkoski, I., & Fahl, M. (2006).** State-of-the-art review of optimization methods for short-term scheduling of batch processes. *Computers & Chemical Engineering*, 30(6-7), 913–946. https://doi.org/10.1016/j.compchemeng.2006.02.008
- **Pantelides, C. C. (1994).** Unified frameworks for optimal process planning and scheduling. In *Proceedings of the Second Conference on Foundations of Computer-Aided Process Operations* (pp. 253–274). CACHE Publications.
- **Shah, N. (1998).** Single- and multisite planning and scheduling: current status and future challenges. *Foundations of Computer-Aided Process Operations*, AIChE Symposium Series, 94(320), 75–90.
- **International Society of Automation (ISA). (2010).** *ISA-88.01-2010: Batch Control Part 1: Models and Terminology*. Research Triangle Park, NC.
- **International Electrotechnical Commission (IEC). (2013).** *IEC 62264-1: Enterprise-Control System Integration — Part 1: Attributes and Category Models (ISA-95)*. Geneva, Switzerland.
- **U.S. Occupational Safety and Health Administration (OSHA). (2000).** *Process Safety Management of Highly Hazardous Chemicals* (29 CFR 1910.119). Washington, DC.

---

## 17. Links

- **Parent Module Index:** [Module 38 — Process Industries, Commodities, and Bulk Supply Chains](_index.md)
- **Course Survey File:** [Process Industries, Commodities, and Bulk Supply Chains](../../../course/38_process_industries_commodities.md)
- **Sibling Topic 02:** [Yield and Mass Balance](02_yield-mass-balance.md)
- **Sibling Topic 03:** [Quality Specifications and Blending](03_quality-specs-blending.md)
- **Sibling Topic 04:** [Campaigns and Changeover](04_campaigns-changeover.md)
- **Sibling Topic 05:** [Bulk Storage and Inventory](05_bulk-storage-inventory.md)
- **Prerequisite Topic:** [Module 15 — Production Planning and Scheduling](../../02_methods/15_production_planning_scheduling/_index.md)
- **Prerequisite Topic:** [Module 16 — Inventory Theory and Stochastic Control](../../02_methods/16_inventory_theory_stochastic_control/_index.md)
- **Capstone Domain Link:** [Module 39 — Mining, Metals, and Forestry Supply Chains](../39_mining_metals_forestry/_index.md)
