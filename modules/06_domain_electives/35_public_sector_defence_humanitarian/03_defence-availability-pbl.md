# Defence Availability and Performance-Based Logistics

> Module 35 — Public Sector, Defence, and Humanitarian Logistics
> Topic 03 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 10 (Inventory Optimization), Module 11 (Safety Stock & Service Levels), Module 15 (MRO & Spare Parts), Module 31 Topic 03 (Multi-Echelon Sparing), Module 35 Topic 01 (Public Procurement & Frameworks), Module 35 Topic 02 (Whole-Life Cost & MEAT Evaluation)
> Version 2026.5

---

## 1. Why this topic exists

Modern defence supply chains operate under an operational mandate fundamentally distinct from commercial enterprise logistics: mission assurance under catastrophic operational penalty. In commercial supply chains, inventory stockouts incur lost sales, customer dissatisfaction, or expedited freight surcharges. In defence operations, an unavailable Line Replaceable Unit (LRU) for an air superiority fighter, guided-missile destroyer, or main battle tank grounds the weapon system, forfeiting operational deterrence, compromising tactical readiness, or endangering warfighter survival during high-tempo combat engagements.

Three systemic structural realities govern defence logistics and separate it from commercial distribution:

```text
+----------------------------------------------------------------------------------------------------+
|                      THE TRILEMMA OF MODERN DEFENCE SUSTAINMENT & AVAILABILITY                     |
+----------------------------------------------------------------------------------------------------+
| 1. MULTI-DECADE ASSET LIFECYCLES VERSUS 18-MONTH COMMERCIAL TECHNOLOGY CYCLES                     |
|    - Front-line weapon systems (F-35, Eurofighter Typhoon, Virginia-class submarines, M1A2 Abrams) |
|      remain operational for 40 to 60+ years across complex mid-life upgrade spirals.               |
|    - Commercial Off-The-Shelf (COTS) microelectronics, semiconductors, and software operating     |
|      systems have commercial production lifecycles of 18 to 36 months, creating permanent         |
|      Diminishing Manufacturing Sources and Material Shortages (DMSMS) obsolescence cliffs.          |
+----------------------------------------------------------------------------------------------------+
| 2. CLOSED-LOOP ROTABLE REPAIR PIPELINES ACROSS MULTI-ECHELON NETWORKS                             |
|    - High-value subsystems (radar processors, flight control actuators, turbofan modules) are not |
|      consumed; they are removed at the flight line, evacuated through repair pipelines,            |
|      refurbished at intermediate facilities or industrial depots, and returned to inventory.      |
|    - Pipeline dynamics are characterized by long turnaround times (60 to 180 days), complex        |
|      retrograde transit logistics, and non-linear backorder propagation across base and depot tiers.|
+----------------------------------------------------------------------------------------------------+
| 3. CONTRACTUAL ALIGNMENT AND THE PERFORMANCE-BASED LOGISTICS (PBL) REVOLUTION                      |
|    - Legacy transactional procurement (FAR Part 15 / spares buying) established perverse           |
|      economic incentives: prime contractors maximized revenue when parts failed frequently,        |
|      repair queues swelled, and emergency engineering modifications were requisitioned.            |
|    - Performance-Based Logistics (PBL) shifts procurement from transactional piece-parts to buying |
|      operational readiness: contracting for platform availability, mission-capable hours, and      |
|      mean time between removals under shared-risk, fixed-price incentive structures.               |
+----------------------------------------------------------------------------------------------------+
```

When a major military platform experiences degraded readiness—such as modern fifth-generation combat aircraft achieving mission-capable rates below 60% due to depot maintenance backlogs and parts shortages—the root causes are rarely manufacturing defects on the assembly line. Instead, failures stem from misaligned sustainment contracts, uncoordinated multi-echelon rotable inventories, unmitigated electronics obsolescence, and administrative delay times that dominate physical repair times.

Addressing these operational failures requires rigorous analytical methods. Asset managers and logistics engineers must replace naive single-echelon safety stock formulas with multi-echelon stochastic repairable models (METRIC and VARI-METRIC), apply hazard rate and renewal mathematics to forecast DMSMS obsolescence events, and structure performance-based contracting curves that align prime contractor profitability directly with weapon system operational availability ($A_o$).

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. Deconstruct and calculate Operational Availability ($A_o$), Inherent Availability ($A_i$), and Achieved Availability ($A_a$), quantitatively isolating the impacts of Mean Time Between Maintenance (MTBM), Mean Time to Repair (MTTR), and Administrative and Logistics Delay Time (ALDT).
2. Formulate and solve multi-echelon rotable spares inventory requirements across organizational (O-level), intermediate (I-level), and depot (D-level) maintenance nodes using Palm's Theorem, the METRIC framework, and VARI-METRIC variance-to-mean adjustments.
3. Model Diminishing Manufacturing Sources and Material Shortages (DMSMS) component obsolescence dynamics using Weibull hazard distributions and non-homogeneous Poisson processes (NHPP) to identify optimal life-of-type buy quantities versus engineering redesigns.
4. Structure Performance-Based Logistics (PBL) contracting mechanisms, deriving target fees, gain-share and penalty slopes, availability deadbands, and liquidated damage thresholds under Firm-Fixed-Price per Flight Hour (FFP-PBTH) and Cost-Plus-Incentive-Fee (CPIF) models.
5. Identify, diagnose, and remediate systemic failure modes in defence sustainment networks, including unauthorized cannibalization death spirals, No-Fault-Found (NFF) component recycling, and contractor moral hazard under misaligned metrics.
6. Evaluate compliance and operational alignment with defence acquisition standards, including DoD Instruction 5000.91, Federal Acquisition Regulation (FAR) Part 12 versus Part 15, DEF-STAN 00-600, S3000L Integrated Product Support specifications, and DFARS technical data rights clauses.
7. Execute marginal allocation heuristics (Sherbrooke's greedy algorithm) to minimize expected weapon system backorders across forward operating bases subject to hard capital budget constraints.

---

## 3. The core mental model

The defence availability ecosystem is a closed-loop cybernetic system operating across operational, industrial, and contractual tiers. Unlike commercial supply chains that follow linear procure-move-consume patterns, military readiness depends on the continuous circular rotation of high-value rotable assemblies supported by public depots and commercial industrial partners.

```text
====================================================================================================
                        OPERATIONAL FLEET & MISSION ENVIRONMENT (FLIGHT LINE / BASE)
====================================================================================================
   +-------------------------------------------------------------------------------------------+
   | Fleet Sortie Generation & Flying Hour Program (FHP): N Aircraft accumulating flight hours  |
   +-------------------------------------------------------------------------------------------+
                                                │
                                                ▼  Component Failures (λ_ij = N_j * H_j * θ_i)
   +-------------------------------------------------------------------------------------------+
   | Organizational Maintenance (O-Level / Flight Line):                                       |
   | - Built-In Test (BIT) Fault Isolation & Diagnostic Triage                                |
   | - Unserviceable LRU removed from aircraft (Aircraft down: Non-Mission Capable Supply)     |
   | - Serviceable LRU pulled from Base Rotable Float Stock (s_ij)                             |
   +-------------------------------------------------------------------------------------------+
         │                                                             ▲
         │ Unserviceable LRU Carcass                                   │ Serviceable Replenishment
         ▼                                                             │
   +───────────────────────────────────+                               │
   | Base Intermediate Maintenance     |──(Repairable: Fraction r_ij)──┤
   | (I-Level / AIS Test Benches):     |   Turnaround Time T_Bij       │
   | Diagnostic verification & repair  |                               │
   +───────────────────────────────────+                               │
         │                                                             │
         │ Unrepairable at Base (Fraction 1 - r_ij): Not Repaired This Station (NRTS)
         ▼                                                             │
   +───────────────────────────────────+                               │
   | Retrograde Logistics Pipeline:    |                               │
   | Secure transit (Lead Time O_Rij)  |                               │
   +───────────────────────────────────+                               │
         │                                                             │
         ▼                                                             │
=======================================================================│============================
                        INDUSTRIAL SUSTAINMENT DEPOT & OEM REPAIR TIER │
=======================================================================│============================
   +───────────────────────────────────+                               │
   | Central Sustainment Depot         |                               │
   | (D-Level / OEM Specialized Hub):  |                               │
   | - Complete teardown and overhaul  |                               │
   | - SRU fault isolation & repair    |                               │
   | - Turnaround Time T_Di            |                               │
   | - Depot Buffer Stock (s_Di)       |                               │
   | - Expected Depot Backorders EBO_D |                               │
   +───────────────────────────────────+                               │
         │                             │                               │
         │ Piece-Part / SRU Demand     │ Serviceable Assembly Released │
         ▼                             ▼                               │
   +─────────────────────+     +───────────────────────────────────+   │
   | DMSMS & Subtier     |     | Forward Logistics Pipeline:       |───┘
   | Component Supply:   |     | Priority transit (Lead Time O_Fij)|
   | - Silicon Foundries |     | Base replenishment: O_Fij + W_Di  |
   | - LOT Buy Reserves  |     +───────────────────────────────────+
   | - Form-Fit-Function |
   |   Redesign Pipeline |
   +─────────────────────+
         ▲
         │ Contractual Governance & Availability Telemetry
====================================================================================================
                        PERFORMANCE-BASED LOGISTICS (PBL) CONTRACT ENGINE
====================================================================================================
   +───────────────────────────────────────────────────────────────────────────────────────────+
   | Telemetry Ingestion: Flight Hours Flown, Fully Mission Capable (FMC) Hours, Depot TAT    |
   | Incentive Settlement: Base PBTH Rate +/- Availability Gain-Share Bonus / Penalty Curve    |
   +───────────────────────────────────────────────────────────────────────────────────────────+
```

### The Causal Feedback Loops

The dynamics of this closed-loop architecture are governed by three tightly coupled feedback loops:

1. **The Readiness Regeneration Loop (Balancing):** As aircraft fly operational sorties, rotable subsystems accumulate stress cycles and degrade. Component failures transition aircraft from Fully Mission Capable (FMC) to Non-Mission Capable Supply (NMCS). If the base rotable float contains a serviceable asset ($s_{ij} > 0$), the unserviceable carcass is swapped immediately, minimizing Administrative and Logistics Delay Time (ALDT), restoring the aircraft to FMC status, and maintaining fleet operational availability ($A_o$).
2. **The Cannibalization Death Spiral (Reinforcing):** When rotable float stock is depleted ($s_{ij} = 0$) and depot replenishment pipelines experience delays, base maintenance commanders face intense operational pressure to meet daily flying schedules. Maintenance crews resort to "cannibalization"—removing a functioning LRU from an already grounded aircraft to install on an aircraft scheduled for flight. This doubles the maintenance labor hours per flight hour, exposes delicate connectors and wiring harnesses to mechanical fatigue, induces No-Fault-Found (NFF) diagnostic errors, and grounds the cannibalized airframe indefinitely as a "hangar queen," driving fleet availability down in an accelerating collapse.
3. **The PBL Incentive Alignment Loop (Balancing):** Under a Firm-Fixed-Price per Flight Hour (FFP-PBTH) contract, the contractor receives a fixed dollar amount for every hour the fleet flies, but must fund all component repairs, transportation, and depot operations out of that revenue. If component reliability is low, high removal rates consume the contractor's operational margin. This creates an economic incentive for the contractor to invest internal research and development capital into component reliability enhancements, root-cause failure elimination, and test bench diagnostic precision, directly improving Mean Time Between Removals (MTBR) and elevating weapon system availability.

---

## 4. Definitions and vocabulary

The following technical terms, operational metrics, and military standards form the analytical foundation of defence availability modeling:

| Term / Acronym | Formal Definition | Operational Context & Formulation | Common Misuse / Pitfall |
| :--- | :--- | :--- | :--- |
| **Operational Availability ($A_o$)** | The probability that an item or weapon system will operate satisfactorily at a random point in time under stated operating conditions in an actual operational environment. | $A_o = \frac{\text{Uptime}}{\text{Uptime} + \text{Downtime}} = \frac{\text{MTBM}}{\text{MTBM} + \text{MDT}}$ (dimensionless ratio, typically 0.70 to 0.90). | Confusing $A_o$ with design reliability or inherent availability; $A_o$ includes supply chain and logistics delay times. |
| **Inherent Availability ($A_i$)** | The probability that an item will operate satisfactorily under ideal operational conditions, accounting only for corrective maintenance downtime. | $A_i = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}}$ (dimensionless ratio, typically > 0.95). | Assuming high $A_i$ guarantees high fleet readiness; ignores preventive maintenance, spares stockouts, and transit delays. |
| **Achieved Availability ($A_a$)** | The probability that an item will operate satisfactorily when accounting for both corrective and preventive maintenance, but excluding logistics delays. | $A_a = \frac{\text{MTBM}}{\text{MTBM} + \bar{M}}$, where $\bar{M}$ is the mean active maintenance downtime. | Believing $A_a$ reflects flight line reality; omits parts requisition lead times and administrative queues. |
| **Mean Down Time (MDT)** | The total elapsed time required to restore an unserviceable system to fully operational status. | $\text{MDT} = \bar{M} + \text{ALDT} = \text{MTTR} + \text{LDT} + \text{ADT}$ (expressed in clock hours or operational days). | Treating MDT as identical to wrench-turning repair time (MTTR); ALDT typically accounts for 60%–85% of total MDT. |
| **Administrative & Logistics Delay Time (ALDT)** | The portion of downtime during which maintenance is delayed awaiting spare parts, specialized tooling, test benches, or administrative releases. | $\text{ALDT} = \text{Awaiting Parts (AWP)} + \text{Transit Time} + \text{Work Order Queue Time}$ (clock hours). | Classifying logistics delays as maintenance inefficiencies rather than supply chain structural deficits. |
| **Mission Capable (MC) Rate** | The percentage of total possessed aircraft or systems capable of performing at least one primary military mission. | $\text{MC} = \frac{\text{FMC Hours} + \text{PMC Hours}}{\text{Total Possessed Hours}} = 1 - (\text{NMCS} + \text{NMCM})$ (percentage). | Equating MC rate directly to $A_o$; MC rate measures state duration across possessed inventory, while $A_o$ reflects stochastic readiness probability. |
| **Non-Mission Capable Supply (NMCS)** | The condition or time duration where an asset cannot perform its mission because it is waiting for spare parts. | Expressed as an NMCS rate: $\frac{\text{NMCS Hours}}{\text{Total Possessed Hours}} \times 100\%$ (target typically < 5% to 8%). | Treating NMCS as purely an airbase warehouse issue; NMCS is driven by depot turnaround queues and supplier lead times. |
| **Non-Mission Capable Maintenance (NMCM)** | The condition or time duration where an asset cannot perform its mission due to active scheduled or unscheduled maintenance. | Expressed as an NMCM rate: $\frac{\text{NMCM Hours}}{\text{Total Possessed Hours}} \times 100\%$ (target typically < 10% to 15%). | Conflating active wrench time with total maintenance duration, ignoring technician staffing constraints. |
| **Line Replaceable Unit (LRU)** | A modular assembly designed for rapid removal and replacement directly at the flight line or operational unit (O-Level). | Characterized by quick-release fasteners, blind-mate connectors, and built-in test isolation (e.g., radar transmitter, APU). | Using LRU interchangeably with SRU; an LRU is repaired by swapping internal subassemblies (SRUs). |
| **Shop Replaceable Unit (SRU)** | A sub-tier electronic card, hydraulic valve, or mechanical assembly located inside an LRU, repaired at I-Level or D-Level. | Multi-indenture component: failure of an SRU causes failure of the parent LRU (e.g., circuit card assembly, power supply). | Allocating SRU stock at the flight line; flight line maintainers lack the specialized test benches to replace SRUs. |
| **Performance-Based Logistics (PBL)** | An outcomes-based acquisition strategy where the customer contracts for system capability rather than individual parts and repair services. | Contracts specify availability targets ($A_o$), flight hour rates, and logistics footprint limits under risk-sharing structures. | Viewing PBL as outsourcing military maintenance; PBL aligns commercial supply pipelines while maintenance execution remains military or hybrid. |
| **Power by the Hour (PBTH)** | A contracting structure where the customer pays a fixed fee per operational operating hour (e.g., flight hour, engine hour). | Payment = $\text{Operating Hours Flown} \times \text{Agreed Rate per Hour}$ ($/flight hour). | Confusing PBTH with time-and-materials; PBTH requires the vendor to fund all spares and repairs within the hourly rate. |
| **DMSMS** | Diminishing Manufacturing Sources and Material Shortages: the loss or impending loss of manufacturers or suppliers of critical items. | Driven by semiconductor obsolescence, REACH environmental chemical bans, or sole-source foundry closures. | Treating DMSMS as an unexpected crisis; DMSMS is an inevitable engineering lifecycle reality requiring predictive renewal models. |
| **Life-of-Type (LOT) Buy** | A single final procurement order of sufficient component inventory to support a weapon system through its planned retirement date. | Sized via hazard models to cover remaining fleet operating life plus safety stock buffer for demand volatility. | Underestimating component degradation and holding costs over 15–20 year storage horizons in nitrogen dry-boxes. |
| **Form, Fit, Function (FFF)** | A replacement component that matches the physical dimensions, electrical interfaces, and operational performance of an obsolete part. | Requires no modifications to the parent assembly mating surfaces or host software interfaces. | Assuming FFF replacements require zero qualification; flight certification and environmental testing are mandatory. |
| **Palm's Theorem** | A stochastic theorem stating that if demand follows a Poisson process with rate $\lambda$ and lead times $T$ are i.i.d. with mean $E[T]$, the steady-state pipeline follows a Poisson distribution with mean $\Lambda = \lambda E[T]$. | Foundation of repairable inventory theory; pipeline distribution depends only on mean lead time, independent of lead time distribution shape. | Applying Palm's Theorem without verifying Poisson demand arrivals, such as during batched deployment retrograde shipments. |
| **METRIC** | Multi-Echelon Technique for Recoverable Item Control: a mathematical model for optimizing multi-echelon repairable spares. | Minimizes expected backorders (EBO) across bases subject to inventory investment budget using marginal analysis. | Ignoring variance in depot repair queues; classic METRIC underestimates base stock requirements when depot backorders fluctuate. |
| **VARI-METRIC** | An extension of METRIC that calculates the variance-to-mean ratio (VMR) of depot backorders and fits a negative binomial distribution. | Accounts for stochastic compound demand at bases caused by depot repair bottlenecks. | Assuming Poisson distributions apply equally to volatile depot supply pipelines with high service time variance. |
| **No-Fault-Found (NFF)** | A condition where a removed LRU passes all bench tests at the intermediate or depot repair facility without exhibiting defects. | Also termed Cannot Duplicate (CND) or Retest OK (RTOK); typically accounts for 20% to 45% of avionics removals. | Categorizing NFF as a false removal; NFF is frequently caused by intermittent thermal faults, cable harness wear, or environmental vibration. |
| **Carcass** | An unserviceable rotable assembly or subassembly removed from an operating system, awaiting induction into the repair cycle. | The physical repairable core; legally controlled property under military inventory tracking (Condition Code F). | Treating carcasses as scrap; a rotable supply chain cannot function without rapid retrograde transit of carcasses back to depots. |

---

## 5. The mechanics, step by step

Achieving weapon system operational availability requires four interconnected analytical workflows: availability decomposition, multi-echelon rotable inventory optimization, predictive obsolescence management, and incentive-aligned PBL contract design.

### Step 1: Fleet operational availability decomposition

Operational readiness begins with the formal mathematical decomposition of time states. A military fleet of $N$ weapon systems possesses a total calendar time pool across a planning horizon $T_{\text{cal}}$:

$$\text{Total Possessed Hours} = N \times T_{\text{cal}} = \text{Operating Time} + \text{Standby Time} + \text{Downtime}$$

Operational Availability ($A_o$) measures the probability that a system is either operating or capable of operating when called upon:

$$A_o = \frac{\text{Uptime}}{\text{Uptime} + \text{Downtime}} = \frac{\text{MTBM}}{\text{MTBM} + \text{MDT}}$$

Where:
- $\text{MTBM}$ is the Mean Time Between Maintenance events, encompassing both unscheduled corrective maintenance and scheduled preventive maintenance:

$$\text{MTBM} = \frac{1}{\lambda_u + \lambda_s}$$

Here, $\lambda_u$ is the unscheduled failure rate (events per operating hour) and $\lambda_s$ is the scheduled maintenance induction rate.

- $\text{MDT}$ is the Mean Down Time per maintenance event. MDT is decomposed into active maintenance time and supply delay time:

$$\text{MDT} = \bar{M} + \text{ALDT}$$

Where $\bar{M}$ is the active wrench-turning maintenance downtime:

$$\bar{M} = \frac{\lambda_u \cdot \text{MTTR}_u + \lambda_s \cdot \text{MTTR}_s}{\lambda_u + \lambda_s}$$

And $\text{ALDT}$ is Administrative and Logistics Delay Time:

$$\text{ALDT} = \text{ALDT}_{\text{supply}} + \text{ALDT}_{\text{transport}} + \text{ALDT}_{\text{admin}} + \text{ALDT}_{\text{queue}}$$

To connect $A_o$ to flight line operations, we evaluate its relationship to Inherent Availability ($A_i$) and Achieved Availability ($A_a$):

```text
+----------------------------------------------------------------------------------------------------+
|                                    AVAILABILITY SPECTRUM COMPARISON                                |
+----------------------------------------------------------------------------------------------------+
| Inherent Availability (Ai):    MTBF / (MTBF + MTTR_u)                                              |
| - Pure engineering design metric.                                                                  |
| - Excludes preventive maintenance, spare parts stockouts, transportation, and technician delays.   |
| - Typical value: 0.960 to 0.990.                                                                   |
+----------------------------------------------------------------------------------------------------+
| Achieved Availability (Aa):    MTBM / (MTBM + M_bar)                                               |
| - Operational maintenance design metric.                                                           |
| - Includes preventive and corrective wrench-turning maintenance under ideal logistics support.     |
| - Excludes spares shortages, supply chain pipeline delays, and administrative queue delays.        |
| - Typical value: 0.880 to 0.940.                                                                   |
+----------------------------------------------------------------------------------------------------+
| Operational Availability (Ao): MTBM / (MTBM + MDT) = MTBM / (MTBM + M_bar + ALDT)                  |
| - Total operational reality metric.                                                                |
| - Captures the full impact of parts shortages (NMCS), depot repair queues, and transit lead times.  |
| - Typical military value: 0.650 to 0.850.                                                          |
+----------------------------------------------------------------------------------------------------+
```

The Operational Availability equation proves mathematically why engineering design reliability alone cannot guarantee military readiness: if $\text{ALDT}$ swells due to a broken rotable repair pipeline, $A_o$ collapses even if $A_i$ exceeds 99%.

### Step 2: Multi-echelon rotable spares optimization (METRIC & VARI-METRIC)

Rotable items cycle through a closed-loop network comprising $J$ forward air bases ($j = 1, \dots, J$) and a single centralized industrial sustainment depot ($D$). Let:
- $N_j$ be the number of weapon systems stationed at Base $j$.
- $H_j$ be the annual flight hours programmed per weapon system at Base $j$.
- $Z_i$ be the Quantity Per Aircraft (QPA) of Line Replaceable Unit $i$.
- $\theta_i$ be the failure rate per flight hour of LRU $i$.

The annual demand rate generated at Base $j$ for LRU $i$ is:

$$\lambda_{ij} = N_j \cdot H_j \cdot Z_i \cdot \theta_i$$

#### Base-level maintenance triage and routing

When an LRU fails, flight line maintainers remove the unserviceable carcass and induct it into the Base Intermediate Maintenance facility (I-Level):
- With probability $r_{ij}$, the fault can be repaired locally at Base $j$ with an average repair turnaround time $T_{Bij}$ (days).
- With probability $(1 - r_{ij})$, the item cannot be repaired locally and is declared Not Repaired This Station (NRTS). The unserviceable carcass is evacuated via retrograde logistics to the Central Depot, requiring an order-and-retrograde shipping lead time $O_{Rij}$ (days).

#### Central depot pipeline dynamics

The central depot experiences an aggregate demand arrival rate from all operational bases:

$$\lambda_{iD} = \sum_{j=1}^J (1 - r_{ij}) \lambda_{ij}$$

The depot overhauls unserviceable carcasses with an average industrial repair turnaround time $T_{Di}$ (days). Under Palm's Theorem, assuming repair resources act as an ample service capacity ($M/G/\infty$) queue, the steady-state number of items in the depot repair pipeline, $X_{iD}$, follows a Poisson distribution with mean pipeline parameter $\Lambda_{iD}$:

$$\Lambda_{iD} = \left(\frac{\lambda_{iD}}{365}\right) \cdot T_{Di}$$

If the depot holds a rotable float stock of $s_{iD}$ serviceable units, the expected backorders at the depot, $EBO_D(s_{iD})$, is given by:

$$EBO_D(s_{iD}) = \sum_{x = s_{iD} + 1}^\infty (x - s_{iD}) P(X_{iD} = x) = \sum_{x = s_{iD} + 1}^\infty (x - s_{iD}) \frac{\Lambda_{iD}^x e^{-\Lambda_{iD}}}{x!}$$

Using the identity for Poisson expectations:

$$EBO_D(s_{iD}) = \Lambda_{iD} \cdot \left[1 - \sum_{k=0}^{s_{iD}-1} \frac{\Lambda_{iD}^k e^{-\Lambda_{iD}}}{k!}\right] - s_{iD} \cdot \left[1 - \sum_{k=0}^{s_{iD}} \frac{\Lambda_{iD}^k e^{-\Lambda_{iD}}}{k!}\right]$$

The average depot delay experienced by a base requisition, $W_{iD}$ (in days), is obtained via Little's Law:

$$W_{iD} = \frac{EBO_D(s_{iD})}{\lambda_{iD} / 365}$$

#### Base-level pipeline and expected backorders

When Base $j$ declares an LRU as NRTS, it submits an emergency requisition to the central depot. The effective replenishment lead time for an NRTS event is the forward transit time $O_{Fij}$ plus any queuing delay $W_{iD}$ at the depot awaiting stock release.

The total effective replenishment lead time for Base $j$, $T_{ij}^{\text{eff}}$, is the weighted expectation across local repair and depot resupply:

$$T_{ij}^{\text{eff}}(s_{iD}) = r_{ij} \cdot T_{Bij} + (1 - r_{ij}) \cdot \left[O_{Fij} + W_{iD}\right]$$

Under the classic METRIC approximation, the steady-state pipeline of unserviceable/replenishing units at Base $j$, $X_{ij}$, is assumed to be Poisson-distributed with mean parameter $\Lambda_{ij}$:

$$\Lambda_{ij}(s_{iD}) = \left(\frac{\lambda_{ij}}{365}\right) \cdot T_{ij}^{\text{eff}}(s_{iD})$$

With base stock $s_{ij}$, the expected base backorders is:

$$EBO_j(s_{ij} \mid s_{iD}) = \sum_{y = s_{ij} + 1}^\infty (y - s_{ij}) \frac{\Lambda_{ij}^y e^{-\Lambda_{ij}}}{y!}$$

#### The VARI-METRIC refinement

In operational reality, depot backorders fluctuate stochastically. When the depot stockouts occur, multiple bases experience simultaneous stockouts, violating the Poisson assumption of independent single arrivals. This induces compound Poisson arrivals at the bases, where the variance of the base pipeline exceeds its mean.

VARI-METRIC calculates the Variance-to-Mean Ratio (VMR) of the base pipeline:

$$\operatorname{Var}(X_{ij}) = \Lambda_{ij}(s_{iD}) + (1 - r_{ij})^2 \cdot \left(\frac{\lambda_{ij}}{\lambda_{iD}}\right)^2 \cdot \operatorname{Var}(EBO_D(s_{iD}))$$

When $\operatorname{Var}(X_{ij}) > \Lambda_{ij}$, VARI-METRIC fits a Negative Binomial distribution to $X_{ij}$, defining parameters $r$ (shape) and $p$ (success probability):

$$p = \frac{\Lambda_{ij}}{\operatorname{Var}(X_{ij})}, \quad r = \frac{\Lambda_{ij}^2}{\operatorname{Var}(X_{ij}) - \Lambda_{ij}}$$

Base backorders are then computed over the Negative Binomial distribution, preventing the severe understocking that occurs under naive Poisson models during depot bottlenecks.

#### Marginal allocation algorithm (Sherbrooke's heuristic)

To allocate a rotable float capital budget $B$ across multiple LRUs $i = 1, \dots, I$, depot stock levels $s_{iD}$, and base stock levels $s_{ij}$, we solve the optimization problem:

$$\min \sum_{i=1}^I \sum_{j=1}^J EBO_{ij}(s_{ij} \mid s_{iD}) \quad \text{subject to} \quad \sum_{i=1}^I c_i \cdot \left( s_{iD} + \sum_{j=1}^J s_{ij} \right) \le B$$

The procedure executes a two-tier marginal allocation:
1. For each item $i$, evaluate the marginal reduction in system expected backorders per dollar:

$$\Delta(s) = \frac{EBO(s) - EBO(s+1)}{c_i}$$

2. For a fixed depot stock $s_{iD}$, allocate base stock greedily across bases until marginal gains diminish.
3. Search over feasible depot stock levels $s_{iD} \in \{0, 1, \dots, s_D^{\max}\}$ to find the global minimum backorder frontier for each expenditure level.

System operational availability across all independent critical LRUs is derived from base backorders:

$$A_o \approx \prod_{i=1}^I \prod_{j=1}^J \left( 1 - \frac{EBO_{ij}}{N_j \cdot Z_i} \right)^{Z_i}$$

### Step 3: DMSMS obsolescence risk modeling & Life-of-Type sizing

Diminishing Manufacturing Sources and Material Shortages (DMSMS) occurs when commercial component manufacturers discontinue silicon chips, microprocessors, or materials critical to defence electronics.

```text
+----------------------------------------------------------------------------------------------------+
|                         DMSMS COMPONENT LIFECYCLE & RESOLUTION PATHWAYS                            |
+----------------------------------------------------------------------------------------------------+
| Active Production  ──>  Declining Market  ──>  End-of-Life (EOL)  ──>  Discontinued / Obsolete     |
| (Full Commercial        (Single Source /       (Last-Time-Buy         (Zero OEM Production;        |
|  Availability)           Lead Times Spike)      Notice Issued)         Scrap / Grey Market Only)   |
+----------------------------------------------------------------------------------------------------+
                                                        │
                      ┌─────────────────────────────────┴─────────────────────────────────┐
                      ▼                                                                   ▼
       RESOLUTION OPTION A: LIFE-OF-TYPE (LOT) BUY                        RESOLUTION OPTION B: FFF REDESIGN
       - Procure 100% of remaining lifecycle demand.                     - Engineering redesign of circuit card.
       - Sunk capital upfront + ongoing storage costs.                   - Non-recurring engineering (NRE) cost.
       - Risk: Demand underestimation (premature stockout)                - Flight recertification & testing.
               or fleet early retirement (excess scrap).                  - Risk: Schedule slip & firmware bugs.
```

#### Weibull hazard rate modeling for electronics obsolescence

The probability of a commercial component facing an End-of-Life (EOL) notice by operating age $t$ is governed by a Weibull hazard function $h(t)$:

$$h(t) = \frac{\beta}{\eta} \left( \frac{t}{\eta} \right)^{\beta - 1}$$

Where:
- $\beta$ is the shape parameter. For electronics components undergoing commercial obsolescence driven by technological displacement, $\beta > 1$ (typically $2.5 \le \beta \le 4.0$, reflecting an increasing failure/discontinuation hazard rate over calendar time).
- $\eta$ is the characteristic life or scale parameter (typically 7 to 12 years for industrial-grade semiconductors).

The cumulative probability of an obsolescence event occurring prior to time $t$ is:

$$F(t) = 1 - \exp\left[ -\left(\frac{t}{\eta}\right)^\beta \right]$$

The conditional probability of obsolescence within a planning horizon $[t_1, t_2]$ given survival to $t_1$ is:

$$P(\text{EOL} \in [t_1, t_2] \mid t > t_1) = 1 - \exp\left[ \left(\frac{t_1}{\eta}\right)^\beta - \left(\frac{t_2}{\eta}\right)^\beta \right]$$

#### Economic optimization of Life-of-Type (LOT) Bridge Buys

When an OEM issues a Last-Time-Buy (LTB) notification, the program office must select a procurement quantity $Q_{\text{LOT}}$ to support remaining weapon system service life $T_{\text{rem}}$ (years).

Let annual demand for the component across all depot and base repairs be stochastic, modeled as a Poisson process with parameter $\lambda_{\text{part}}$. Total lifecycle demand $D$ across the remaining operational horizon $T_{\text{rem}}$ follows:

$$D \sim \operatorname{Poisson}(\mu_D = \lambda_{\text{part}} \cdot T_{\text{rem}})$$

Economic parameters:
- $c$: Unit acquisition cost during the LTB window ($/unit).
- $h$: Annual physical holding and nitrogen-purged dry-box storage cost rate (expressed as a fraction of unit cost, typically 0.08 to 0.15 $/$/year).
- $d$: Continuous discount rate (weighted average cost of capital or government real discount rate, e.g., 0.03 to 0.05).
- $C_{\text{redesign}}$: Non-Recurring Engineering (NRE) cost, flight qualification, and software re-baselining required if the component runs out of stock prior to fleet retirement (typically $1,500,000 to $10,000,000).
- $v_{\text{salvage}}$: Net salvage value of surplus components at platform retirement (often near $0 due to specialized military packaging and lack of external commercial market).

The total expected discounted lifecycle cost function $E[\text{LCC}(Q)]$ is:

$$E[\text{LCC}(Q)] = c \cdot Q + \text{Cost}_{\text{holding}}(Q) + C_{\text{redesign}} \cdot P(D > Q) - v_{\text{salvage}} \cdot E[\max(0, Q - D)]$$

Where the discounted storage cost across the horizon is:

$$\text{Cost}_{\text{holding}}(Q) \approx \sum_{t=1}^{T_{\text{rem}}} \frac{h \cdot c}{(1 + d)^t} \cdot E[I_t(Q)]$$

Here $E[I_t(Q)]$ is the expected inventory on hand at year $t$.

Because the penalty of understocking ($C_{\text{redesign}}$) dwarfs the unit holding cost ($c$), the optimal LOT buy quantity $Q^*$ corresponds to an asymmetric critical fractal. Treating the decision under the newsvendor formulation:
- Underage Cost: $C_u = C_{\text{redesign}}$
- Overage Cost: $C_o = c + \sum_{t=1}^{T_{\text{rem}}} \frac{h \cdot c}{(1+d)^t} - v_{\text{salvage}}$

The target cumulative non-stockout probability is:

$$P(D \le Q^*) \ge \frac{C_u}{C_u + C_o} = \frac{C_{\text{redesign}}}{C_{\text{redesign}} + C_o}$$

In practice, $\frac{C_u}{C_u + C_o}$ frequently exceeds 0.98 or 0.99, demonstrating why defence programs systematically procure substantial buffer quantities during last-time-buy windows to avoid catastrophic engineering redesign cycles.

### Step 4: PBL contract mechanism design & incentive curves

Performance-Based Logistics structures contract economics to incentivize prime contractors to invest in reliability and supply chain velocity.

```text
+----------------------------------------------------------------------------------------------------+
|                         PERFORMANCE-BASED LOGISTICS (PBL) INCENTIVE CURVE                          |
+----------------------------------------------------------------------------------------------------+
| Contractor                                                                                         |
| Margin / Fee                                                                                       |
|     ▲                                                                                              |
|     │                                                +---------------- Maximum Incentive Cap (Fee_max)
|     │                                               /                                              |
|     │                                              / Gain-Share Slope (alpha)                      |
| F_T │─────────────────────────────────────────────+ Target Availability (Ao_target, Fee_target)   |
|     │                                            /                                                 |
|     │                                           /                                                  |
|     │                       Deadband           /                                                   |
|     │               [Ao_target - delta, Ao_target]                                                 |
|     │                     +───────────────────+                                                    |
|     │                    /                                                                         |
|     │                   / Penalty Slope (beta)                                                     |
|     │                  /                                                                           |
| F_min ────────────────+ Availability Floor (Ao_floor)                                             |
|     │                 │                                                                            |
|     │                 │ Liquidated Damages / Contract Default                                      |
|     └─────────────────┼──────────────────────────┼───────────────────────┼──────────────────────►  |
|                      Ao_floor                 Ao_target               Ao_cap            Operational|
|                                                                                         Availability|
+----------------------------------------------------------------------------------------------------+
```

#### Contracting structure taxonomy

1. **Firm-Fixed-Price per Flight Hour (FFP-PBTH):**
   The military pays a negotiated rate $R_{\text{FH}}$ ($/flight hour) based on audited flying hour logs:

   $$\text{Baseline Payment} = \text{Flight Hours Flown} \times R_{\text{FH}}$$

   The contractor is legally obligated to provide all rotable spares, maintenance personnel, depot overhauls, and transportation necessary to maintain the fleet above an availability floor. If an LRU experiences high failure rates, the contractor absorbs 100% of the cost overrun. If the contractor re-engineers an LRU to double its MTBF, the contractor retains 100% of the maintenance cost savings.

2. **Cost-Plus-Incentive-Fee (CPIF):**
   Used when operational tempo or baseline system reliability is highly uncertain. The contract establishes a Target Cost $C_T$, a Target Fee $F_T$, a Government Share Ratio $s_B$, and a Contractor Share Ratio $s_S$ (where $s_B + s_S = 1.0$):

   $$\text{Contractor Fee} = F_T + s_S \cdot (C_T - C_{\text{actual}}) \pm \text{Incentive Bonus/Penalty}(A_o)$$

   Clamped between minimum fee $F_{\min}$ and maximum fee $F_{\max}$.

#### Piecewise linear availability incentive formulation

Under an FFP-PBTH contract with availability incentives, the final fee adjustment $\phi(A_o)$ is determined relative to four thresholds:
- $A_{\text{floor}}$: Minimum allowable operational availability.
- $A_{\text{target}}$: Contract baseline performance target.
- $A_{\text{cap}}$: Maximum rewarded availability.
- $\delta$: Deadband parameter around $A_{\text{target}}$ to absorb minor operational noise.

The availability incentive function $\psi(A_o)$ modifies the contract payment:

$$\psi(A_o) = \begin{cases}
-P_{\max}, & A_o < A_{\text{floor}} \\
-\beta \cdot (A_{\text{target}} - \delta - A_o), & A_{\text{floor}} \le A_o < A_{\text{target}} - \delta \\
0, & A_{\text{target}} - \delta \le A_o \le A_{\text{target}} \\
+\alpha \cdot (A_o - A_{\text{target}}), & A_{\text{target}} < A_o \le A_{\text{cap}} \\
+B_{\max}, & A_o > A_{\text{cap}}
\end{cases}$$

Where:
- $\alpha$ is the gain-share slope ($/percentage point improvement).
- $\beta$ is the penalty slope ($/percentage point degradation). In defence contracts, $\beta > \alpha$ (asymmetric disincentive) to penalize readiness deficits more severely than marginal over-performance is rewarded.
- $P_{\max}$ is the maximum liquidated penalty.
- $B_{\max}$ is the capped maximum incentive bonus.

Total contractor compensation for the settlement period is:

$$\text{Total Remittance} = (\text{Flight Hours Flown} \times R_{\text{FH}}) + \psi(A_o)$$

---

## 6. Worked example

### Operational scenario

Consider a deployed wing of 48 multi-role fighter aircraft distributed across two forward operating bases supported by a single centralized industrial sustainment depot:
- **Base Alpha (Front-Line Quick Reaction Alert):** $N_A = 28$ aircraft, flying an average of $H_A = 300$ flight hours per airframe per year. Total annual flight hours: $28 \times 300 = 8,400\text{ FH/year}$.
- **Base Bravo (Training and Operational Reserve):** $N_B = 20$ aircraft, flying an average of $H_B = 250$ flight hours per airframe per year. Total annual flight hours: $20 \times 250 = 5,000\text{ FH/year}$.
- Total fleet utilization: $13,400\text{ flight hours/year}$.

We analyze a mission-critical LRU: the **Modular Low-Power Radio Frequency (MLPRF) Processor** within the active electronically scanned array (AESA) radar suite.
- Quantity per aircraft: $Z = 1$.
- Unit acquisition/replacement cost: $c = \$220,000$.
- Mean Time Between Failures: $\text{MTBF} = 400\text{ flight hours}$.
- Unscheduled removal rate: $\theta = 1 / 400 = 0.0025\text{ failures/flight hour}$.

### Maintenance and logistics parameters

```text
+--------------------------------------------------------+-----------------+-----------------+
| Parameter                                              | Base Alpha      | Base Bravo      |
+--------------------------------------------------------+-----------------+-----------------+
| Aircraft Possessed (N_j)                               | 28 aircraft     | 20 aircraft     |
| Annual Flight Hours per Airframe (H_j)                 | 300 FH/yr       | 250 FH/yr       |
| Total Annual Base Flight Hours                         | 8,400 FH/yr     | 5,000 FH/yr     |
| Base Failure Rate (lambda_ij = Base FH / MTBF)         | 21.0 items/yr   | 12.5 items/yr   |
| Daily Failure Rate (lambda_daily = lambda_ij / 365)    | 0.057534 /day   | 0.034247 /day   |
| Base Repair Capability Fraction (r_ij)                 | 0.60 (60%)      | 0.50 (50%)      |
| Base Repair Turnaround Time (T_Bij)                    | 14 days         | 18 days         |
| Retrograde Transit Time to Depot (O_Rij)               | 10 days         | 12 days         |
| Forward Transit Time from Depot (O_Fij)                | 8 days          | 10 days         |
+--------------------------------------------------------+-----------------+-----------------+
| Central Sustainment Depot Overhaul Turnaround Time (T_D): 60 days                          |
+--------------------------------------------------------------------------------------------+
```

### Step 1: Central depot demand and pipeline computation

The depot experiences demand only for carcasses that cannot be repaired at the bases (NRTS items):
- Base Alpha NRTS rate: $(1 - r_A) \times \lambda_A = (1 - 0.60) \times 21.0 = 8.40\text{ items/year}$.
- Base Bravo NRTS rate: $(1 - r_B) \times \lambda_B = (1 - 0.50) \times 12.5 = 6.25\text{ items/year}$.
- Total depot demand arrival rate:

$$\lambda_D = 8.40 + 6.25 = 14.65\text{ items/year}$$

Daily depot demand:

$$\lambda_{D,\text{daily}} = \frac{14.65}{365} = 0.040137\text{ items/day}$$

Under Palm's Theorem, with depot repair turnaround time $T_D = 60\text{ days}$, the expected number of units in the depot repair pipeline $\Lambda_D$ is:

$$\Lambda_D = \lambda_{D,\text{daily}} \times T_D = 0.040137 \times 60 = 2.4082\text{ units}$$

### Step 2: Depot stock level and expected backorders

We evaluate a depot rotable stock allocation of $s_D = 3\text{ units}$.

We compute the Poisson probabilities $P(X = k)$ and cumulative values $F(k)$ for $\Lambda_D = 2.4082$:

| $k$ | Probability $P(X = k)$ | Cumulative $F(k) = P(X \le k)$ |
| :---: | :---: | :---: |
| 0 | $e^{-2.4082} = 0.089977$ | 0.089977 |
| 1 | $\frac{2.4082^1}{1!} \times 0.089977 = 0.216682$ | 0.306659 |
| 2 | $\frac{2.4082^2}{2!} \times 0.089977 = 0.260907$ | 0.567566 |
| 3 | $\frac{2.4082^3}{3!} \times 0.089977 = 0.209425$ | 0.776991 |
| 4 | $\frac{2.4082^4}{4!} \times 0.089977 = 0.126084$ | 0.903075 |
| 5 | $\frac{2.4082^5}{5!} \times 0.089977 = 0.060727$ | 0.963802 |
| 6 | $\frac{2.4082^6}{6!} \times 0.089977 = 0.024374$ | 0.988176 |
| 7 | $\frac{2.4082^7}{7!} \times 0.089977 = 0.008385$ | 0.996561 |
Depot Expected Backorders $EBO_D(s_D = 3)$:

$$EBO_D(3) = \Lambda_D [1 - F(2)] - 3 [1 - F(3)] = 2.4082(1 - 0.567566) - 3(1 - 0.776991) = 1.041388 - 0.669027 = 0.372361\text{ units}$$

Average delay per requisition at depot ($W_D$):

$$W_D = \frac{EBO_D(3)}{\lambda_{D,\text{daily}}} = \frac{0.372361}{0.040137} = 9.277\text{ days}$$

### Step 3: Base replenishment lead times and pipeline sizing

With depot average wait time $W_D = 9.277\text{ days}$:

#### Base Alpha
Effective replenishment lead time:

$$T_A^{\text{eff}} = r_A \cdot T_{BA} + (1 - r_A) \cdot [O_{FA} + W_D]$$
$$T_A^{\text{eff}} = 0.60 \times 14 + 0.40 \times [8 + 9.277] = 8.40 + 0.40 \times 17.277 = 8.40 + 6.911 = 15.311\text{ days}$$

Base Alpha expected pipeline ($\Lambda_A$):

$$\Lambda_A = \lambda_{A,\text{daily}} \times T_A^{\text{eff}} = 0.057534 \times 15.311 = 0.8809\text{ units}$$

#### Base Bravo
Effective replenishment lead time:

$$T_B^{\text{eff}} = r_B \cdot T_{BB} + (1 - r_B) \cdot [O_{FB} + W_D]$$
$$T_B^{\text{eff}} = 0.50 \times 18 + 0.50 \times [10 + 9.277] = 9.00 + 0.50 \times 19.277 = 9.00 + 9.639 = 18.639\text{ days}$$

Base Bravo expected pipeline ($\Lambda_B$):

$$\Lambda_B = \lambda_{B,\text{daily}} \times T_B^{\text{eff}} = 0.034247 \times 18.639 = 0.6383\text{ units}$$

### Step 4: Base stock allocation and backorder determination

Assume the inventory budget allows allocating $s_A = 2\text{ units}$ at Base Alpha and $s_B = 1\text{ unit}$ at Base Bravo.

For Base Alpha ($\Lambda_A = 0.8809$): $P(0) = 0.414408, P(1) = 0.365052, P(2) = 0.160787$, yielding cumulative values $F_A(1) = 0.779460$ and $F_A(2) = 0.940247$.

$$EBO_A(2) = \Lambda_A [1 - F_A(1)] - 2 [1 - F_A(2)] = 0.8809(1 - 0.779460) - 2(1 - 0.940247) = 0.194274 - 0.119506 = 0.074768\text{ units}$$
For Base Bravo ($\Lambda_B = 0.6383$): $P(0) = 0.528191, P(1) = 0.337144$, yielding cumulative values $F_B(0) = 0.528191$ and $F_B(1) = 0.865335$.

$$EBO_B(1) = \Lambda_B [1 - F_B(0)] - 1 [1 - F_B(1)] = 0.6383(1 - 0.528191) - 1(1 - 0.865335) = 0.301156 - 0.134665 = 0.166491\text{ units}$$
#### Fleet Operational Availability ($A_o$):
Total expected fleet backorders for the MLPRF LRU:

$$EBO_{\text{fleet}} = EBO_A(2) + EBO_B(1) = 0.074768 + 0.166491 = 0.241259\text{ aircraft grounded}$$

Total fleet size: $N = 48\text{ aircraft}$.

Assuming the MLPRF processor is the availability-limiting subsystem:

$$A_o = 1 - \frac{EBO_{\text{fleet}}}{N} = 1 - \frac{0.241259}{48} = 1 - 0.005026 = 0.994974 \approx 99.50\%$$

Total spares float capital invested:

$$\text{Capital} = (s_D + s_A + s_B) \times c = (3 + 2 + 1) \times \$220,000 = 6 \times \$220,000 = \$1,320,000$$

### Step 5: DMSMS Life-of-Type (LOT) obsolescence buy calculation

During year 4 of operation, the commercial semiconductor manufacturer of the radar's Field Programmable Gate Array (FPGA) announces immediate discontinuation (Last-Time-Buy notice).
- Remaining fleet operational service life: $T_{\text{rem}} = 8\text{ years}$.
- Each MLPRF processor contains 2 FPGAs ($Z_{\text{FPGA}} = 2$).
- Annual component replacement demand across base and depot repairs: $\lambda_{\text{FPGA}} = 28\text{ chips/year}$.
- Expected total 8-year demand: $\mu_D = 28 \times 8 = 224\text{ chips}$.
- Unit cost during LTB: $c = \$1,400\text{ / chip}$.
- Storage and nitrogen dry-box holding cost: $h = 10\%\text{ per year}$ ($0.10 \times \$1,400 = \$140\text{ / chip / year}$).
- Discount rate: $d = 4.0\%\text{ per year}$.
- Cost of stockout (redesigning the circuit board, flight recertification, and software refactoring): $C_{\text{redesign}} = \$2,800,000$.

#### Critical ratio calculation:
Over the 8-year horizon, the present value factor of annual holding cost is:

$$\text{PVA}_{8, 4\%} = \frac{1 - (1 + 0.04)^{-8}}{0.04} = 6.7327 \implies H_8 = 0.10 \times \$1,400 \times 6.7327 = \$942.58$$

Overage and underage costs:
- $C_o = c + H_8 = \$1,400 + \$942.58 = \$2,342.58$
- $C_u = C_{\text{redesign}} = \$2,800,000$

Critical fractile:

$$\text{CR} = \frac{C_u}{C_u + C_o} = \frac{2,800,000}{2,800,000 + 2,342.58} = \frac{2,800,000}{2,802,342.58} = 0.999164$$

To achieve a cumulative non-stockout probability of $99.9164\%$ under a Poisson demand distribution with mean $\mu_D = 224$:
Using the Gaussian approximation to the Poisson distribution ($\sigma = \sqrt{\mu_D} = \sqrt{224} = 14.9666$):
The standard normal quantile for $z_{0.999164}$ is $z \approx 3.143$.

$$Q^* \approx \mu_D + z \cdot \sigma = 224 + (3.143 \times 14.9666) = 224 + 47.04 = 271.04 \rightarrow 272\text{ units}$$

The optimal economic decision is to execute a Life-of-Type buy of **272 FPGAs** at a total purchase cost of $272 \times \$1,400 = \$380,800$, spending an extra $\$67,200$ on 48 buffer units to insure the fleet against a $\$2.8\text{M}$ redesign penalty.

### Step 6: PBL contract fee and payment settlement

The wing operates under a multi-year Firm-Fixed-Price per Flight Hour (FFP-PBTH) contract with the radar OEM:
- Base flight hour rate: $R_{\text{FH}} = \$680\text{ / flight hour}$.
- Annual planned flying hours: $13,400\text{ FH}$.
- Contract baseline remittance: $13,400 \times \$680 = \$9,112,000$.
- Target Operational Availability: $A_{\text{target}} = 85.0\%$.
- Deadband: $\delta = 1.0\%$ (Range: $84.0\%$ to $85.0\%$).
- Availability Floor: $A_{\text{floor}} = 78.0\%$.
- Availability Cap: $A_{\text{cap}} = 92.0\%$.
- Gain-share slope: $\alpha = \$150,000\text{ per 1.0\% (0.01) above target}$.
- Penalty slope: $\beta = \$250,000\text{ per 1.0\% (0.01) below deadband floor}$.

#### Annual settlement calculation
Across the contract year, fleet telemetry records an achieved radar operational availability of $A_o = 89.2\%$.

Since $A_o > A_{\text{target}}$:
- Percentage points above target: $89.2\% - 85.0\% = 4.2\% = 0.042$.
- Incentive bonus earned:

$$\psi(A_o) = \alpha \times 100 \times (A_o - A_{\text{target}}) = \$150,000 \times 4.2 = \$630,000$$

Net annual remittance to contractor:

$$\text{Total Payment} = \$9,112,000 + \$630,000 = \$9,742,000$$

The contractor generated $\$630,000$ in pure profit expansion by maintaining high reliability and rapid depot turnaround times, which kept rotable pipelines full and eliminated flight line backorders.

---

## 7. Code, query, or calculation

The following self-contained Python script implements the METRIC multi-echelon optimization algorithm. It evaluates depot and base rotable stock levels, computes expected backorders via Palm's Theorem, and outputs the fleet operational availability ($A_o$) matching the worked scenario.

```python
"""Multi-Echelon Rotable Spares Optimization (METRIC Algorithm)."""
import math

def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0.0: return 1.0 if k == 0 else 0.0
    return (lam**k) * math.exp(-lam) / math.factorial(k)

def ebo_poisson(stock: int, lam: float) -> float:
    if lam <= 0.0: return 0.0
    k_max = int(lam + 10.0 * math.sqrt(max(lam, 1.0)) + stock + 25)
    return sum((k - stock) * poisson_pmf(k, lam) for k in range(stock + 1, k_max))

def solve_metric_network(fleet_size: int, bases: list, lru_cost: float,
                         budget: float, depot_tat_days: float):
    # Aggregate depot pipeline from base Not-Repaired-This-Station (NRTS) demands
    lam_depot_yr = sum((1.0 - b["r_base"]) * b["lam_yr"] for b in bases)
    lam_depot_daily = lam_depot_yr / 365.0
    depot_pipeline = lam_depot_daily * depot_tat_days

    best_config, min_fleet_ebo = None, float("inf")
    max_spares = int(budget // lru_cost)

    for s_depot in range(min(max_spares + 1, 10)):
        ebo_depot = ebo_poisson(s_depot, depot_pipeline)
        w_depot_days = (ebo_depot / lam_depot_daily) if lam_depot_daily > 0 else 0.0
        rem_spares = max_spares - s_depot

        # Compute effective lead times and pipelines for each base
        base_pipelines = []
        for b in bases:
            lam_d = b["lam_yr"] / 365.0
            eff_lt = b["r_base"] * b["t_base_days"] + (1.0 - b["r_base"]) * (b["o_fwd_days"] + w_depot_days)
            base_pipelines.append(lam_d * eff_lt)

        # Greedy marginal allocation across forward bases
        s_bases = [0] * len(bases)
        for _ in range(rem_spares):
            best_base, max_reduction = -1, -1.0
            for i in range(len(bases)):
                red = ebo_poisson(s_bases[i], base_pipelines[i]) - ebo_poisson(s_bases[i] + 1, base_pipelines[i])
                if red > max_reduction:
                    max_reduction, best_base = red, i
            if best_base >= 0 and max_reduction > 1e-6:
                s_bases[best_base] += 1

        tot_ebo = sum(ebo_poisson(s_bases[i], base_pipelines[i]) for i in range(len(bases)))
        if tot_ebo < min_fleet_ebo:
            min_fleet_ebo = tot_ebo
            ao = max(0.0, 1.0 - (tot_ebo / fleet_size))
            best_config = (s_depot, s_bases, tot_ebo, ao, (s_depot + sum(s_bases)) * lru_cost)

    return best_config

if __name__ == "__main__":
    bases_data = [
        {"name": "Alpha", "lam_yr": 21.0, "r_base": 0.60, "t_base_days": 14.0, "o_fwd_days": 8.0},
        {"name": "Bravo", "lam_yr": 12.5, "r_base": 0.50, "t_base_days": 18.0, "o_fwd_days": 10.0}
    ]
    s_d, s_b, tot_ebo, ao_val, spent = solve_metric_network(
        fleet_size=48, bases=bases_data, lru_cost=220000, budget=1500000, depot_tat_days=60.0
    )
    print(f"Optimal Allocation: Depot Stock = {s_d}, Base Stocks = {s_b}")
    print(f"Total Fleet EBO: {tot_ebo:.4f} aircraft grounded")
    print(f"System Operational Availability (Ao): {ao_val * 100:.2f}%")
    print(f"Total Float Capital Invested: ${spent:,.2f}")
```

### Expected output

```text
Optimal Allocation: Depot Stock = 2, Base Stocks = [2, 2]
Total Fleet EBO: 0.2033 aircraft grounded
System Operational Availability (Ao): 99.58%
Total Float Capital Invested: $1,320,000.00
```

> **Note:** The algorithmic marginal analysis discovers that reallocating one rotable asset from Depot ($s_D = 2$) to Base Bravo ($s_B = 2$) reduces total fleet expected backorders from $0.2413$ to $0.2033$, raising operational availability from $99.50\%$ to $99.58\%$ at the exact same $\$1,320,000$ capital expenditure.

---

## 8. Data, standards, and regulatory context

Defence availability analytics and performance-based contracting operate under strict statutory, regulatory, and engineering standards. Failure to follow these requirements risks contract cancellation, False Claims Act litigation, or severe audit sanctions by government accountability bodies.

```text
+----------------------------------------------------------------------------------------------------+
|                         DEFENCE REGULATORY & GOVERNANCE HIERARCHY                                  |
+----------------------------------------------------------------------------------------------------+
| STATUTORY LAW & ACQUISITION REGULATIONS                                                            |
| - 10 U.S.C. Chapter 271 / Truth in Negotiations Act (TINA): Cost and pricing certification.        |
| - FAR Part 12 (Commercial Items) vs FAR Part 15 (Contracting by Negotiation).                      |
| - DFARS 252.227-7013 / 7014: Rights in Technical Data and Computer Software.                      |
+----------------------------------------------------------------------------------------------------+
| POLICY & INSTRUCTION MANDATES                                                                      |
| - DoD Instruction 5000.91: Product Support Management for the Adaptive Acquisition Framework.      |
| - DoD PBL Guidebook: Mandatory methodology for structuring availability business case analyses.   |
| - UK MoD DEF-STAN 00-600: Integrated Logistics Support Requirements for MOD Projects.             |
+----------------------------------------------------------------------------------------------------+
| TECHNICAL & ENGINEERING SPECIFICATIONS                                                             |
| - MIL-HDBK-338B: Electronic Reliability Design Handbook (MTBF, MTTR, Ai, Ao formulations).        |
| - GEIA-STD-0007 / SAE S3000L: International Procedure Handbook for Integrated Logistics Support.  |
| - DSPPO SD-22: Diminishing Manufacturing Sources and Material Shortages (DMSMS) Guidebook.        |
+----------------------------------------------------------------------------------------------------+
```

### 1. Acquisition policy and the PBL mandate

- **DoD Instruction 5000.91 (Product Support Management for the Adaptive Acquisition Framework, 2021):**
  Establishes mandatory policy requiring Program Managers (PMs) and Product Support Managers (PSMs) to evaluate Performance-Based Logistics (PBL) product support arrangements for all covered weapon systems. The instruction mandates that life cycle sustainment plans explicitly establish quantifiable availability metrics ($A_o$), materiel availability ($A_m$), materiel reliability ($R_m$), and operating and support (O&S) cost baselines.
- **FAR Part 12 versus FAR Part 15 Regulatory Tension:**
  A core legal challenge in PBL contracting is the interface between commercial contracting mechanisms and military oversight:
  - *FAR Part 12 (Commercial Items):* Allows Firm-Fixed-Price contracts exempt from certified cost or pricing data requirements. Prime contractors strongly prefer FAR Part 12 for PBL contracts because it permits commercial margins when aggressive reliability investments reduce sustainment costs.
  - *FAR Part 15 (Contracting by Negotiation):* Mandates submission of certified cost or pricing data under 10 U.S.C. Chapter 271 (formerly the Truth in Negotiations Act - TINA) for non-commercial sole-source acquisitions exceeding $2,000,000. The Department of Defense Inspector General (DoD IG) frequently challenges FAR Part 12 determinations for military-unique combat aircraft subsystems, forcing hybrid contracting arrangements.

### 2. Technical data rights and intellectual property (DFARS)

The primary barrier to competition and long-term cost containment in defence availability contracts is the allocation of intellectual property rights:
- **DFARS 252.227-7013 (Rights in Technical Data - Noncommercial Items):**
  - *Unlimited Rights:* The government can provide engineering drawings, maintenance data, and software source code to third-party competitors or organic military depots.
  - *Government Purpose Rights (GPR):* The government can use and disclose data for government purposes, including competitive reprocurement, but cannot release data for commercial use. GPR typically transitions to Unlimited Rights after five years.
  - *Limited / Proprietary Rights:* The contractor retains full commercial ownership because the item was developed entirely at private expense. The government cannot share technical data with third parties or organic military depots without explicit contractor licensing.
- When prime contractors maintain proprietary control over diagnostic fault trees, automated test equipment (ATE) interface test adapters, and LRU manufacturing source data, the military faces "vendor hold-up." In this scenario, organic maintenance depots cannot repair carcasses, leaving the military with no viable alternative to sole-source commercial availability contracts.

### 3. Engineering and logistics standards

- **MIL-HDBK-338B (Electronic Reliability Design Handbook):**
  The authoritative military technical standard defining failure physics, MTBF estimation, Arrhenius temperature acceleration modeling, and the mathematical derivations of $A_i, A_a,$ and $A_o$. It formalizes the standard assumptions governing corrective maintenance lognormal distributions and Poisson arrival processes.
- **GEIA-STD-0007 / SAE S3000L / S1000D:**
  The international integrated product support specification suite. GEIA-STD-0007 defines Logistics Product Data (LPD) schemas, standardizing failure modes, effects, and criticality analysis (FMECA), level of repair analysis (LORA), and maintenance allocation charts across military ERP systems.
- **DSPPO SD-22 (DMSMS Guidebook, Defense Standardization Program Office):**
  Governs proactive obsolescence management across five operational maturity levels: Level 1 (reactive fire-fighting upon supplier notice) to Level 5 (proactive design-refresh planning integrated into system operational flight software blocks).

---

## 9. Industry practice and case evidence

The operational record of performance-based logistics and weapon system availability models across three decades reveals both dramatic readiness gains and structural governance vulnerabilities.

```text
+----------------------------------------------------------------------------------------------------+
|                         COMPARATIVE DEFENCE SUSTAINMENT CASE EVIDENCE                              |
+----------------------------------------------------------------------------------------------------+
| Case Study / Platform    | Contract Structure       | Observed Readiness Impact | Primary Governance Finding       |
+--------------------------+--------------------------+---------------------------+----------------------------------+
| UK MoD Tornado (ATTAC)   | Firm-Fixed-Price         | Aircraft availability up  | Alignment achieved via joint     |
| & Typhoon (TAS)          | availability per flying  | > 40%; support costs      | MoD-BAE co-located flight line   |
| (BAE Systems)            | hour with gain-share     | cut by £1.4B over 10 yrs  | maintenance integration.         |
+--------------------------+--------------------------+---------------------------+----------------------------------+
| US Navy F/A-18 F414      | FFP-PBTH 10-year engine  | Engine availability rose  | Aggressive component reliability |
| Turbofan Engine          | availability and time-   | from 65% to > 88%; bare   | investments funded directly by   |
| (GE Aviation)            | on-wing (TOW) metrics    | engine removals fell 30%  | contractor margin expansion.     |
+--------------------------+--------------------------+---------------------------+----------------------------------+
| US Air Force C-17        | Global Sustainment       | High mission-capable      | Virtual international inventory  |
| Globemaster III (GSP)    | Partnership (GSP) pooled | rates (> 80%) maintained  | pooling reduced aggregate spares |
| (Boeing)                 | rotable inventory        | across global fleet       | capital requirements by > $1.2B. |
+--------------------------+--------------------------+---------------------------+----------------------------------+
| Joint Strike Fighter     | Hybrid Product Support   | Mission Capable rates     | Severe friction from proprietary |
| F-35 Lightning II        | Integrator; transition   | frequently < 60%; chronic | data lock-in, ALIS/ODIN delays,  |
| (Lockheed Martin)        | toward 5-year PBL        | depot repair backlogs     | and organic depot starvation.    |
+--------------------------+--------------------------+---------------------------+----------------------------------+
```

### 1. UK Ministry of Defence: Tornado ATTAC and Typhoon Availability Service (TAS)

In the early 2000s, the UK Ministry of Defence faced escalating operating costs and declining availability across its Panavia Tornado GR4 fleet under traditional transactional maintenance. In 2006, the MoD signed the **Availability Transformation: Tornado Aircraft Contract (ATTAC)** with BAE Systems, a £945 million, long-term performance-based contract guaranteeing flight line availability and guaranteed flying hours.

- **Mechanics:** BAE Systems assumed responsibility for base spares management, technical publications, second-line intermediate maintenance, and rotable overhaul pipelines. Royal Air Force (RAF) uniformed maintainers worked alongside BAE civilian engineers in combined maintenance hangars at RAF Marham and RAF Lossiemouth.
- **Results:** Over the contract's ten-year execution, fleet availability increased by more than 40%, scheduled maintenance turnaround times were cut in half, and the MoD documented cash-releasing savings exceeding £1.4 billion. The model was subsequently expanded into the **Typhoon Availability Service (TAS)** and the **Centurion** weapon system integration program, proving that co-located integrated teams eliminate the administrative delays that typically inflate ALDT.

### 2. US Navy & GE Aviation: F414 Turbofan Engine PBL

Powering the Boeing F/A-18E/F Super Hornet and EA-18G Growler, the General Electric F414-GE-400 engine represents one of the longest-running and most successful subsystem PBL implementations in US naval aviation.

- **Mechanics:** Under successive multi-year Firm-Fixed-Price per Flight Hour contracts, GE Aviation is paid based on engine flight hours logged across fleet carrier air wings. GE is responsible for rotable float sizing, depot module overhaul at Fleet Readiness Center Southeast (FRCSE), technical data updates, and obsolescence mitigation for all engine components.
- **Results:** Naval Air Systems Command (NAVAIR) reported that engine availability improved from ~65% to consistently above 88%, while bare-engine unscheduled removal rates dropped by more than 30%. Because GE retained the savings from extended Time-On-Wing (TOW), the company invested tens of millions of dollars in advanced turbine blade thermal barrier coatings and combustor metallurgy improvements—investments that would have reduced vendor revenue under traditional cost-plus repair contracting.

### 3. F-35 Lightning II: The Challenges of Proprietary Data Lock-In

The Lockheed Martin F-35 Lightning II program represents the largest defence acquisition in history, with lifecycle sustainment costs projected to exceed $1.3 trillion over 66 years. The program's sustainment struggles provide a case study in the risks of misaligned availability architecture:

- **Empirical Findings (GAO Reports GAO-21-439 and GAO-23-105341):** Throughout 2020–2024, the F-35 fleet consistently failed to meet military availability targets, with Mission Capable (MC) rates hovering between 51% and 58% (against an 80% target) and Full Mission Capable (FMC) rates often below 35%.
- **Root Causes:**
  - *Data Rights and Lock-In:* The original acquisition contract left critical technical data rights, diagnostic algorithms, and proprietary software interfaces under prime contractor ownership (Limited Rights). Organic military depots at Ogden Air Logistics Complex and Fleet Readiness Center East could not repair rotable carcasses without proprietary contractor licenses, creating depot repair queues exceeding 180 days.
  - *Telemetry and IT Infrastructure:* The Autonomic Logistics Information System (ALIS)—designed to automatically order replacement spares upon landing—suffered from severe data integrity defects, frequently generating erroneous fault codes that triggered unnecessary removals and inflated No-Fault-Found (NFF) rates. The Department of Defense subsequently initiated a multi-year effort to replace ALIS with the Operational Data Integrated Network (ODIN).
  - *Spares Pooling Frictions:* International partner nations participating in the global rotable pool experienced localized stockouts because sovereign customs regulations, ITAR export restrictions, and retrograde logistics pipelines delayed carcass evacuation to centralized European and North American repair depots.

---

## 10. Failure modes and red flags

The table below outlines twelve structural failure modes, operational early warning signs, underlying causal mechanisms, readiness consequences, and mandatory engineering/contractual mitigations in defence availability systems:

| # | Failure Mode | Early Warning Sign | Underlying Causal Mechanism | Operational Consequence | Engineering / Contractual Mitigation |
| :- | :--- | :--- | :--- | :--- | :--- |
| 1 | **Cannibalization Death Spiral** | Ratio of cannibalization actions per 100 flight hours exceeds 10.0; presence of stripped "hangar queen" airframes. | Flight line commanders harvest parts from grounded aircraft to cover depot pipeline stockouts ($s_j = 0$). | Labor hours per flight hour double; connectors, wire bundles, and brackets fatigue; multiple airframes rendered permanently non-flyable. | Establish strict regulatory caps on authorized cannibalizations; enforce minimum rotable float safety stock ($s_j \ge 1$); mandate METRIC replenishment alerts. |
| 2 | **No-Fault-Found (NFF) Churn** | Intermediate bench Retest OK (RTOK) rate exceeds 25% for high-value avionics LRUs. | Intermittent thermal failures, airframe vibration, or loose backplane wiring that cannot be duplicated on static bench test fixtures. | Serviceable stock depleted; logistics pipelines clogged with functioning parts cycling through depots; true systemic faults remain uncorrected. | Deploy advanced environmental stress screening (ESS), synthetic bench test harnesses with dynamic shaking/thermal profiling, and flight line bus analyzers. |
| 3 | **Proprietary Data Hold-Up** | Organic military depots cannot obtain test program sets (TPS) or manufacturing drawings; direct-award PBL extensions. | Government failed to acquire Government Purpose Rights (GPR) or Level 3 Technical Data Packages during initial platform acquisition. | Sole-source contractor maintains monopoly pricing leverage; organic depot overhaul capacity starved; military cannot compete sustainment. | Implement modular open systems architecture (MOSA); contractually enforce deferred data rights licensing (DFARS 252.227-7013); negotiate escrow source codes. |
| 4 | **Sortie Cherry-Picking / Gaming** | Surge in short, benign daytime training sorties near the end of the contract quarter; reluctance to fly combat weapons profiles. | Contractor contractually rewarded on aggregate $A_o$ or basic flying hours, with penalties tied to catastrophic component wear. | Fleet operational readiness degrades for severe tactical profiles; airframes accumulate uneven structural fatigue; warfighter training compromised. | Structure availability metrics to require Full Mission Capability (FMC) across specified combat mission profiles; audit mission telemetry. |
| 5 | **DMSMS Obsolescence Cliff** | Subtier component supplier issues Last-Time-Buy (LTB) notice; broker spot-market prices spike by 500%–2,000%. | Commercial silicon foundries decommission older microelectronics fabrication lines (e.g., planar CMOS, 180nm nodes) without military coordination. | Entire platform fleet grounded due to stockout of a single $50 semiconductor; forced emergency redesign costing millions in unbudgeted funds. | Mandate proactive DMSMS monitoring (SD-22); execute automated BOM scanning against commercial EOL databases; budget proactive LOT buys. |
| 6 | **Depot Repair Queue Starvation** | Depot work-in-progress (WIP) collapses, followed 60 days later by a massive wave of unrepairable carcasses. | Base maintainers batch unserviceable carcasses, delaying retrograde shipment until a sea container or bulk pallet is filled. | Bullwhip effect in depot repair lines; technician idle time followed by extreme overtime; base pipeline stockouts spike dramatically. | Enforce single-piece flow or maximum 48-hour retrograde evacuation SLAs under Transportation Priority 1 (TP-1) mandates. |
| 7 | **Operational Surge Disconnect** | Peacetime availability reaches 90%, but collapses below 40% within 30 days of combat deployment. | PBL contract sized strictly for steady-state peacetime flying hour programs (FHP) without wartime surge rotable float provisions. | Deployed combat units exhaust local rotable floats immediately; long-distance retrograde pipelines sever during combat logistics friction. | Incorporate War Reserve Materiel (WRM) and Deployable Spares Packages (DSP) into contract baselines; include formal surge pricing schedules. |
| 8 | **Cost-Plus Moral Hazard** | Contractor maintenance headcounts expand steadily while MTBR remains flat or degrades over a multi-year period. | Sustainment managed under Cost-Plus-Fixed-Fee (CPFF) or Cost-Plus-Award-Fee (CPAF) with subjective award-fee scoring boards. | Government pays inflated operating costs; contractor lacks economic incentive to engineer root-cause reliability improvements. | Transition contract to Firm-Fixed-Price per Flight Hour (FFP-PBTH) with explicit availability gain-share / penalty curves. |
| 9 | **Unrealistic Carcass Condemnation** | Physical rotable float shrinks continuously over time; inventory reconciliation reveals missing cores. | Supply models assume zero scrap rate ($r_i + \text{NRTS} = 1.0$), but field handling, catastrophic crashes, or battle damage condemn 2%–5% of carcasses. | Systemic shortfall in total rotable pipeline inventory; expected backorders rise permanently; bases face structural stockouts. | Incorporate positive condemnation (washout) rates into METRIC replenishment equations; budget annual procurement of attrition carcasses. |
| 10 | **Nondelivery of Piece-Parts to Depot** | LRU depot overhaul turnaround times swell from 45 days to 180+ days; depots report high Awaiting Parts (AWP) status. | Prime contractor optimizes top-level LRU float, but fails to manage tier-2/3 Shop Replaceable Unit (SRU) and piece-part suppliers. | Industrial depot test benches sit idle awaiting low-cost hardware; LRU repair lines jam; bases run out of serviceable modules. | Extend PBL metrics to intermediate repair nodes; establish multi-indenture MOD-METRIC inventory controls covering SRUs and consumable hardware. |
| 11 | **Cyber and Software Baselines** | Field-serviceable LRUs rejected by aircraft avionics bus immediately upon installation on the flight line. | Hardware rotable repaired to correct physical tolerances, but contains out-of-date Operational Flight Program (OFP) firmware. | Serviceable rotable float rendered unusable; flight line delays escalate; aircraft marked Non-Mission Capable Maintenance for software. | Enforce integrated hardware-software configuration management; mandate bench-level OFP flashing prior to return-to-stock packaging. |
| 12 | **Counterfeit Electronic Penetration** | Premature electronic component burn-in failures during flight line pre-flight engine runs; MTBF drops 70%. | Broker sourcing during an unmitigated DMSMS crisis introduces substandard or remarketed silicon into the rotable supply loop. | Catastrophic flight safety risks; destructive inflight radar or engine controller failures; emergency grounding of entire fleet. | Mandate strict compliance with SAE AS5553 and DFARS 252.246-7007 (Contractor Counterfeit Electronic Part Detection and Avoidance System). |

---

## 11. Assumptions and limits

Every mathematical model of defence availability and performance-based logistics rests on foundational simplifying assumptions. Understanding when these assumptions break down in operational environments is essential for realistic sustainment planning:

| # | Theoretical Assumption | Operational Purpose | Vulnerability / Failure Boundary | Empirical Verification Test |
| :- | :--- | :--- | :--- | :--- |
| 1 | **Stationary Poisson Demand Arrivals** | Allows use of Palm's Theorem to model rotable repair pipelines as $M/G/\infty$ queues with parameter $\Lambda = \lambda T$. | Fails during flight surges, wartime deployments, weapons training exercises, or weather-induced fleet groundings. | Perform dispersion test ($D = s^2 / \bar{x}$); if $D > 1.25$, Poisson is invalid; transition to Negative Binomial or Markov-modulated models. |
| 2 | **Ample Service / Repair Capacity ($M/G/\infty$)** | Assumes units inducted into base or depot repair never wait in queue for an available test bench or qualified technician. | Breaks down when depot repair benches experience capacity bottlenecks, technician shortages, or piece-part starvation (AWP). | Measure queue delay fraction: if $\text{Queue Time} / \text{Active Repair Time} > 0.15$, model as a finite-capacity $M/M/c$ or $G/G/c$ network. |
| 3 | **Independence of Component Failures** | Enables system availability to be computed as the product of individual subsystem availability probabilities: $A_o = \prod A_{o,i}$. | Violates reality during common-mode environmental stress events (lightning strike, combat EMP, contaminated JP-8 fuel, hydraulic pressure surges). | Calculate cross-correlation matrix of component removals across time buckets; identify co-failure clusters. |
| 4 | **Deterministic / Static Logistics Lead Times** | Assumes retrograde shipping, depot turnaround, and forward resupply lead times are constant or follow fixed distributions. | Logistics lead times vary dramatically based on airlift availability, naval replenishment-at-sea cycles, and customs hold-ups. | Track coefficient of variation ($CV_{\text{LT}} = \sigma_{\text{LT}} / \mu_{\text{LT}}$); if $CV > 0.30$, apply stochastic lead-time models. |
| 5 | **100% Carcass Recovery (Zero Condemnation)** | Classic METRIC assumes every removed unserviceable carcass is repaired and returned to the inventory pool without loss. | Aircraft crashes, fire damage, extreme corrosion, or maintenance handling accidents destroy 1% to 5% of carcasses permanently. | Measure annual condemnation (washout) fraction; inject carcass attrition replenishment terms into depot induction models. |
| 6 | **Zero Cannibalization Cross-Contamination** | Models assume cannibalization actions do not degrade component reliability or induce subsequent secondary failures. | Physical extraction and re-installation stresses connectors, degrades electromagnetic gaskets, and induces wiring harness wear. | Track MTBF of previously cannibalized LRUs versus virgin factory-installed LRUs; quantify cannibalization reliability penalty. |
| 7 | **Uniform Aircraft Operating Tempo** | Assumes all airframes across a wing accumulate flight hours, carrier landings, and thermal cycles at an identical rate. | Operational commanders systematically assign specific "preferred" airframes to high-intensity sorties, creating fleet fatigue skew. | Calculate Gini coefficient or variance of flight hours across tail numbers; if skew is severe, model airframe-specific degradation. |
| 8 | **Perfect Diagnostic Fault Isolation** | Assumes flight line Built-In Test (BIT) correctly identifies the failed LRU on the initial maintenance action. | High No-Fault-Found (NFF) rates cause maintainers to remove multiple functioning LRUs in sequence before locating the root defect. | Measure ratio of total LRU removals to confirmed bench repairs; adjust effective demand rates by the false-removal multiplier. |

---

## 12. Exercises

### Exercise 1: Availability metric hierarchy deconstruction
An active-duty squadron of 24 strike fighters operates under the following maintenance parameters over an audited 365-day calendar year:
- Total fleet operating time logged: 7,200 flight hours.
- Unscheduled corrective maintenance: 360 events, requiring an average active repair time ($\text{MTTR}_u$) of 4.5 clock hours.
- Scheduled preventive maintenance inspections: 180 events, requiring an average active servicing time ($\text{MTTR}_s$) of 8.0 clock hours.
- Supply delay: maintainers logged a cumulative 28,800 clock hours waiting for parts from the base warehouse or central depot.
- Administrative delay: work-order sign-offs and maintenance shift handovers consumed a cumulative 7,200 clock hours.

*Deliverable:* Compute (a) Inherent Availability ($A_i$), (b) Achieved Availability ($A_a$), and (c) Operational Availability ($A_o$) for the squadron. State which parameter provides the highest leverage for improving fleet readiness.
*Acceptance criteria:* Complete mathematical derivations with explicit units; correct formulas applied; answers rounded to four decimal places.
*Estimated time:* 25 minutes.

### Exercise 2: Palm's Theorem & rotable pipeline sizing
A forward naval air station supports a squadron of maritime patrol aircraft generating demands for a mission computer LRU at a Poisson rate of $\lambda = 18\text{ failures/year}$. Local base intermediate maintenance repairs 70% of the failures ($r = 0.70$) with an average turnaround time of $T_B = 10\text{ days}$. The remaining 30% are evacuated to a central depot with an average retrograde transit time of 5 days, a depot repair time of 45 days, and a forward return transit time of 5 days.

*Deliverable:*
1. Calculate the daily demand rate and the effective replenishment lead time for the base.
2. Determine the expected pipeline parameter $\Lambda_{\text{base}}$ under Palm's Theorem.
3. Compute the expected backorders $EBO(s)$ for base stock levels $s = 0, 1, 2,$ and $3\text{ units}$.
4. What minimum base stock level is required to achieve a fill rate of at least 90%?
*Acceptance criteria:* Correct step-by-step Poisson probability calculations; explicit Little's Law formulations; correct interpretation of fill rate.
*Estimated time:* 35 minutes.

### Exercise 3: Multi-echelon METRIC allocation across two forward bases
A military helicopter fleet is stationed across two forward operating locations:
- Base 1: Demand rate $\lambda_1 = 20\text{ units/year}$; Base repair fraction $r_1 = 0.50$; Base repair turnaround time $T_{B1} = 12\text{ days}$; Forward shipping time $O_{F1} = 6\text{ days}$.
- Base 2: Demand rate $\lambda_2 = 15\text{ units/year}$; Base repair fraction $r_2 = 0.40$; Base repair turnaround time $T_{B2} = 15\text{ days}$; Forward shipping time $O_{F2} = 8\text{ days}$.
- Central Sustainment Depot: Overhaul turnaround time $T_D = 50\text{ days}$.
- LRU unit cost: $c = \$150,000$.

*Deliverable:*
1. Compute the central depot demand arrival rate and expected depot repair pipeline $\Lambda_D$.
2. Calculate the depot expected backorders $EBO_D(s_D)$ for depot stock levels $s_D = 1$ and $s_D = 2\text{ units}$.
3. For $s_D = 2$, compute the average depot delay $W_D$, the resulting base pipelines $\Lambda_1$ and $\Lambda_2$, and find the optimal base stock allocation $(s_1, s_2)$ under a total system inventory limit of 5 units (Depot + Base 1 + Base 2 = 5).
*Acceptance criteria:* Full numerical calculations showing intermediate Poisson terms; explicit identification of the allocation that minimizes total base expected backorders.
*Estimated time:* 45 minutes.

### Exercise 4: DMSMS Life-of-Type buy vs. engineering redesign financial analysis
The commercial foundry producing an application-specific integrated circuit (ASIC) used in a naval sonar transducer issues an immediate Last-Time-Buy (LTB) notice. The sonar program faces the following decision parameters:
- Remaining weapon system lifecycle: 10 years.
- Annual replacement demand: 40 ASICs/year (Poisson distributed; total 10-year demand $\mu_D = 400\text{ units}$).
- LTB purchase price: $c = \$2,500\text{ per ASIC}$.
- Annual physical storage and nitrogen-purged holding cost rate: $h = 12\%\text{ per year}$.
- Government real discount rate: $d = 3.5\%\text{ per year}$.
- Cost of premature stockout: If the ASIC supply is exhausted before Year 10, the program must fund a Form-Fit-Function redesign of the processing card at a non-recurring engineering and qualification cost of $C_{\text{redesign}} = \$4,500,000$.
- Salvage value of excess chips at platform retirement: $v_{\text{salvage}} = \$0$.

*Deliverable:*
1. Calculate the present value factor and total discounted lifecycle holding cost per chip over the 10-year horizon.
2. Determine the critical fractile ratio under newsvendor economics.
3. Compute the optimal Life-of-Type procurement quantity $Q^*$ using the normal approximation to the Poisson distribution.
4. Calculate the financial risk premium (insurance cost) paid above the expected demand quantity.
*Acceptance criteria:* Complete discounting formulas; correct application of newsvendor underage/overage costs; clear justification of buffer stock sizing.
*Estimated time:* 35 minutes.

### Exercise 5: Performance-Based Logistics contract incentive curve design
Design a Firm-Fixed-Price per Flight Hour (FFP-PBTH) incentive contract for an air force training wing of 30 jet trainers flying an aggregate 9,000 flight hours per year:
- Negotiated baseline hourly rate: $R_{\text{FH}} = \$1,200\text{ / flight hour}$.
- Target Operational Availability: $A_{\text{target}} = 82.0\%$.
- Availability deadband: $[81.0\%, 82.0\%]$.
- Availability Floor: $A_{\text{floor}} = 74.0\%$.
- Availability Cap: $A_{\text{cap}} = 88.0\%$.
- Incentive bonus rate: Contractor earns $\$200,000$ for each full 1.0% (0.01) improvement above $A_{\text{target}}$.
- Disincentive penalty rate: Contractor is penalized $\$350,000$ for each full 1.0% (0.01) drop below the deadband floor of $81.0\%$.
- Liquidated damages: If $A_o < 74.0\%$, contractor forfeits a flat $\$3,000,000$ penalty and is placed in contractual cure status.

*Deliverable:*
1. Formulate the piecewise linear payment equation $\text{Total Remittance}(A_o)$.
2. Calculate total annual contractor remittance under three operational performance scenarios:
   - Scenario A (Exemplary support): Achieved $A_o = 86.5\%$.
   - Scenario B (Nominal noise): Achieved $A_o = 81.4\%$.
   - Scenario C (Severe pipeline collapse): Achieved $A_o = 76.0\%$.
3. Explain why the penalty slope exceeds the gain-share slope from the perspectives of both military mission assurance and contractor risk pricing.
*Acceptance criteria:* Explicit piecewise mathematical formulations; accurate monetary settlements; rigorous contractual risk critique.
*Estimated time:* 30 minutes.

### Exercise 6: Systemic failure critique: The cannibalization death spiral
A newly deployed armored brigade combat team experiences an unexpected drop in main battle tank operational readiness from 85% to 54% over a 90-day field exercise. Investigation reveals that maintenance crews conducted 142 unauthorized parts cannibalizations to keep lead combat vehicles operational during daily gunnery tables.

*Deliverable:* Prepare an analytical failure-investigation briefing (400–600 words) detailing:
1. The structural root causes in the supply chain that initially prompted maintainers to initiate cannibalization.
2. The mechanical, diagnostic, and human-capital feedback mechanisms that accelerated the readiness collapse.
3. A five-point corrective action plan incorporating multi-echelon inventory policies, logistics tracking, and command incentives to restore fleet readiness.
*Acceptance criteria:* Must reference METRIC pipeline mechanics, No-Fault-Found dynamics, technician labor fatigue, and formal command governance.
*Estimated time:* 40 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Availability metric hierarchy deconstruction

1. **Operating parameters:**
   - Fleet operating time: $T_o = 7,200\text{ flight hours}$.
   - Unscheduled failures: $N_u = 360$ events.
   - Mean Time Between Failures: $\text{MTBF} = \frac{7,200}{360} = 20.0\text{ flight hours}$.
   - Corrective repair wrench time: $\text{MTTR}_u = 4.5\text{ hours}$.
   - Total corrective wrench downtime: $360 \times 4.5 = 1,620\text{ hours}$.

2. **Scheduled maintenance parameters:**
   - Scheduled events: $N_s = 180$ events.
   - Total maintenance events: $N_{\text{tot}} = 360 + 180 = 540$ events.
   - Mean Time Between Maintenance: $\text{MTBM} = \frac{7,200}{540} = 13.3333\text{ flight hours}$.
   - Preventive servicing wrench time: $\text{MTTR}_s = 8.0\text{ hours}$.
   - Total preventive wrench downtime: $180 \times 8.0 = 1,440\text{ hours}$.

3. **Active maintenance time ($\bar{M}$):**
   - Total active maintenance downtime: $1,620 + 1,440 = 3,060\text{ hours}$.
   - Mean active maintenance downtime per event: $\bar{M} = \frac{3,060}{540} = 5.6667\text{ hours}$.

4. **Logistics and administrative delay time (ALDT):**
   - Supply delay: $28,800\text{ hours}$.
   - Administrative delay: $7,200\text{ hours}$.
   - Total delay downtime: $28,800 + 7,200 = 36,000\text{ hours}$.
   - Mean delay per maintenance event: $\text{ALDT} = \frac{36,000}{540} = 66.6667\text{ hours}$.
   - Total Mean Down Time: $\text{MDT} = \bar{M} + \text{ALDT} = 5.6667 + 66.6667 = 72.3333\text{ hours}$.

5. **Availability metrics:**
   - **Inherent Availability ($A_i$):**

     $$A_i = \frac{\text{MTBF}}{\text{MTBF} + \text{MTTR}_u} = \frac{20.0}{20.0 + 4.5} = \frac{20.0}{24.5} = 0.816327 \approx 81.63\%$$

   - **Achieved Availability ($A_a$):**

     $$A_a = \frac{\text{MTBM}}{\text{MTBM} + \bar{M}} = \frac{13.3333}{13.3333 + 5.6667} = \frac{13.3333}{19.0000} = 0.701754 \approx 70.18\%$$

   - **Operational Availability ($A_o$):**

     $$A_o = \frac{\text{MTBM}}{\text{MTBM} + \text{MDT}} = \frac{13.3333}{13.3333 + 72.3333} = \frac{13.3333}{85.6667} = 0.155642 \approx 15.56\%$$

6. **Readiness leverage evaluation:**
   Active wrench-turning maintenance ($\bar{M} = 5.67\text{ hours}$) accounts for only $7.8\%$ of total downtime, whereas administrative and logistics delay ($\text{ALDT} = 66.67\text{ hours}$) accounts for $92.2\%$ of downtime. Halving wrench time ($\text{MTTR}$) would raise $A_o$ by less than two percentage points; reducing parts wait times (ALDT) provides overwhelming operational leverage to restore fleet readiness.

---

### Answer to Exercise 2: Palm's Theorem & rotable pipeline sizing

1. **Daily demand and replenishment lead time:**
   - Annual demand: $\lambda = 18\text{ units/year}$.
   - Daily demand: $\lambda_{\text{daily}} = \frac{18}{365} = 0.049315\text{ units/day}$.
   - Depot cycle time: $T_{\text{depot}} = 5\text{ (retrograde)} + 45\text{ (repair)} + 5\text{ (forward)} = 55\text{ days}$.
   - Effective replenishment lead time:

     $$T_{\text{eff}} = r \cdot T_B + (1 - r) \cdot T_{\text{depot}} = 0.70 \times 10 + 0.30 \times 55 = 7.0 + 16.5 = 23.5\text{ days}$$

2. **Expected pipeline parameter ($\Lambda$):**

   $$\Lambda = \lambda_{\text{daily}} \times T_{\text{eff}} = 0.049315 \times 23.5 = 1.1589\text{ units}$$

3. **Poisson probabilities and Expected Backorders ($EBO(s)$):**
   For $\Lambda = 1.1589$, individual Poisson probabilities are $P(0) = 0.313833, P(1) = 0.363701, P(2) = 0.210747, P(3) = 0.081442, P(4) = 0.023596$, with cumulative values $F(0) = 0.313833, F(1) = 0.677534, F(2) = 0.888281, F(3) = 0.969723$.
   Expected backorders:
   - For $s = 0$: $EBO(0) = \Lambda = 1.1589\text{ units}$.
   - For $s = 1$: $EBO(1) = \Lambda [1 - F(0)] - 1 [1 - F(1)] = 1.1589(1 - 0.313833) - 1(1 - 0.677534) = 0.7952 - 0.3225 = 0.4727\text{ units}$.
   - For $s = 2$: $EBO(2) = \Lambda [1 - F(1)] - 2 [1 - F(2)] = 1.1589(1 - 0.677534) - 2(1 - 0.888281) = 0.3737 - 0.2234 = 0.1503\text{ units}$.
   - For $s = 3$: $EBO(3) = \Lambda [1 - F(2)] - 3 [1 - F(3)] = 1.1589(1 - 0.888281) - 3(1 - 0.969723) = 0.1295 - 0.0908 = 0.0387\text{ units}$.

4. **Fill rate evaluation:**
   Under a Poisson base stock policy, the requisition fill rate equals the cumulative distribution function at the stock level: $\text{Fill Rate}(s) = P(X \le s - 1)$ or $F(s-1)$ for immediate off-the-shelf fulfillment upon arrival:
   - At $s = 2$: $F(1) = 67.75\%$ (fails 90% target).
   - At $s = 3$: $F(2) = 88.83\%$ (near 90%).
   - At $s = 4$: $F(3) = 96.97\% \ge 90\%$.
   Therefore, a base stock of **$s = 4\text{ units}$** is required to guarantee a fill rate exceeding 90%.

---

### Answer to Exercise 3: Multi-echelon METRIC allocation across two forward bases

1. **Depot demand and pipeline:**
   - Base 1 NRTS: $(1 - 0.50) \times 20 = 10.0\text{ units/year}$.
   - Base 2 NRTS: $(1 - 0.40) \times 15 = 9.0\text{ units/year}$.
   - Aggregate depot demand: $\lambda_D = 10.0 + 9.0 = 19.0\text{ units/year}$.
   - Daily depot demand: $\lambda_{D,\text{daily}} = \frac{19.0}{365} = 0.052055\text{ units/day}$.
   - Depot repair pipeline ($T_D = 50\text{ days}$):

     $$\Lambda_D = 0.052055 \times 50 = 2.6027\text{ units}$$

2. **Depot expected backorders:**
   For $\Lambda_D = 2.6027$:
   - $P(0) = 0.074074, P(1) = 0.192793, P(2) = 0.250893, P(3) = 0.217666, P(4) = 0.141630$.
   - $F_D(1) = 0.266867, F_D(2) = 0.517760, F_D(3) = 0.735426$.
   - For $s_D = 1$: $EBO_D(1) = 2.6027(1 - 0.074074) - 1(1 - 0.266867) = 2.4100 - 0.7331 = 1.6769\text{ units}$.
   - For $s_D = 2$: $EBO_D(2) = 2.6027(1 - 0.266867) - 2(1 - 0.517760) = 1.9081 - 0.9645 = 0.9436\text{ units}$.

3. **Base allocations for $s_D = 2$ ($EBO_D = 0.9436$):**
   - Depot wait time: $W_D = \frac{0.9436}{0.052055} = 18.127\text{ days}$.
   - Base 1 lead time: $T_1^{\text{eff}} = 0.50(12) + 0.50(6 + 18.127) = 6.0 + 12.064 = 18.064\text{ days}$.
   - Base 1 pipeline: $\Lambda_1 = \left(\frac{20}{365}\right) \times 18.064 = 0.054795 \times 18.064 = 0.9898\text{ units}$.
   - Base 2 lead time: $T_2^{\text{eff}} = 0.40(15) + 0.60(8 + 18.127) = 6.0 + 15.676 = 21.676\text{ days}$.
   - Base 2 pipeline: $\Lambda_2 = \left(\frac{15}{365}\right) \times 21.676 = 0.041096 \times 21.676 = 0.8908\text{ units}$.

   Remaining inventory for bases: $5 - s_D = 5 - 2 = 3\text{ units}$.
   We evaluate possible splits $(s_1, s_2)$ summing to 3:
   - *Allocation A: $s_1 = 2, s_2 = 1$*
     - $\Lambda_1 = 0.9898$: $EBO_1(2) = 0.9898(1 - 0.7320) - 2(1 - 0.9168) = 0.2653 - 0.1664 = 0.0989$.
     - $\Lambda_2 = 0.8908$: $EBO_2(1) = 0.8908(1 - 0.4103) - 1(1 - 0.7758) = 0.5253 - 0.2242 = 0.3011$.
     - Total base EBO: $0.0989 + 0.3011 = 0.4000\text{ units}$.
   - *Allocation B: $s_1 = 1, s_2 = 2$*
     - $EBO_1(1) = 0.9898(1 - 0.3717) - 1(1 - 0.7320) = 0.6219 - 0.2680 = 0.3539$.
     - $EBO_2(2) = 0.8908(1 - 0.7758) - 2(1 - 0.9385) = 0.1997 - 0.1230 = 0.0767$.
     - Total base EBO: $0.3539 + 0.0767 = 0.4306\text{ units}$.

   **Conclusion:** The optimal allocation under a 5-unit budget is **$s_D = 2, s_1 = 2, s_2 = 1$**, achieving a minimized fleet expected backorder of **0.4000 units**.

---

### Answer to Exercise 4: DMSMS Life-of-Type buy vs. engineering redesign financial analysis

1. **Holding cost discounting:**
   - Present value factor for a 10-year annuity at $d = 3.5\%$:

     $$\text{PVA}_{10, 3.5\%} = \frac{1 - (1 + 0.035)^{-10}}{0.035} = \frac{1 - 0.708919}{0.035} = \frac{0.291081}{0.035} = 8.3166$$

   - Discounted holding cost per chip: $H_{10} = h \cdot c \cdot \text{PVA}_{10, 3.5\%} = 0.12 \times \$2,500 \times 8.3166 = \$2,494.98$.

2. **Critical fractile:**
   - Overage cost: $C_o = c + H_{10} = \$2,500 + \$2,494.98 = \$4,994.98$.
   - Underage cost: $C_u = C_{\text{redesign}} = \$4,500,000$.

     $$\text{CR} = \frac{C_u}{C_u + C_o} = \frac{4,500,000}{4,500,000 + 4,994.98} = \frac{4,500,000}{4,504,994.98} = 0.998891$$

3. **Optimal LOT buy quantity ($Q^*$):**
   - Demand distribution: $\mu_D = 400$, standard deviation $\sigma_D = \sqrt{400} = 20.0$.
   - Quantile of standard normal distribution for $p = 0.998891$: $z_{0.998891} \approx 3.06$.

     $$Q^* = \mu_D + z \cdot \sigma_D = 400 + (3.06 \times 20.0) = 400 + 61.2 \rightarrow 462\text{ chips}$$

4. **Risk premium / insurance analysis:**
   - Baseline expected demand: 400 chips (cost: $\$1,000,000$).
   - Optimal purchase quantity: 462 chips (cost: $\$1,155,000$).
   - Buffer inventory procured: 62 chips.
   - Financial risk premium paid for obsolescence insurance: $62 \times \$2,500 = \$155,000$ (with discounted holding cost: $62 \times \$4,994.98 = \$309,689$). Spending $\$155,000$ in upfront capital provides a $99.89\%$ statistical guarantee of avoiding a $\$4,500,000$ engineering redesign program.

---

### Answer to Exercise 5: Performance-Based Logistics contract incentive curve design

1. **Piecewise linear contract equation:**
   - Baseline annual hours: $9,000\text{ FH}$. Baseline payment: $9,000 \times \$1,200 = \$10,800,000$.
   - Payment piecewise formula:

     $$\text{Remittance}(A_o) = \$10,800,000 + \begin{cases}
     -\$3,000,000, & A_o < 0.740 \\
     -\$35,000,000 \times (0.810 - A_o), & 0.740 \le A_o < 0.810 \\
     \$0, & 0.810 \le A_o \le 0.820 \\
     +\$20,000,000 \times (A_o - 0.820), & 0.820 < A_o \le 0.880 \\
     +\$1,200,000, & A_o > 0.880
     \end{cases}$$

2. **Scenario calculations:**
   - **Scenario A ($A_o = 86.5\%$):**
     - $A_o > 0.820$; performance improvement: $0.865 - 0.820 = 0.045$ (4.5 percentage points).
     - Bonus earned: $4.5 \times \$200,000 = \$900,000$.
     - Total Remittance: $\$10,800,000 + \$900,000 = \mathbf{\$11,700,000}$.
   - **Scenario B ($A_o = 81.4\%$):**
     - $A_o$ lies inside the deadband $[0.810, 0.820]$.
     - Bonus / Penalty: $\$0$.
     - Total Remittance: $\mathbf{\$10,800,000}$.
   - **Scenario C ($A_o = 76.0\%$):**
     - $A_o$ lies in penalty zone: $0.740 \le 0.760 < 0.810$.
     - Shortfall below deadband floor: $0.810 - 0.760 = 0.050$ (5.0 percentage points).
     - Penalty assessed: $5.0 \times \$350,000 = \$1,750,000$.
     - Total Remittance: $\$10,800,000 - \$1,750,000 = \mathbf{\$9,050,000}$.

3. **Asymmetric slope justification:**
   The penalty slope ($\$35,000,000$) exceeds the gain-share slope ($\$20,000,000$) by 75%. From the military customer's perspective, availability falling below the operational threshold compromises pilot training pipelines and national defense readiness, creating non-linear mission damages. From the contractor's perspective, the asymmetric penalty enforces organizational focus on risk mitigation, while the deadband protects against financial penalties arising from minor stochastic noise.

---

### Rubric for Exercise 6: Systemic failure critique (Cannibalization death spiral)

A complete analytical briefing must satisfy the following technical evaluation rubric:

| Dimension | Unsatisfactory (0–1 pts) | Competent (2–3 pts) | Masterful (4–5 pts) |
| :--- | :--- | :--- | :--- |
| **Root Cause Diagnosis** | Blames maintenance discipline or mechanic impatience without examining supply metrics. | Identifies stockouts at base warehouse ($s=0$) and depot replenishment delays as drivers. | Deconstructs the exact trigger: depot queue expansion (AWP), retrograde shipping batching, and lack of visibility into pipeline lead times forcing command triage. |
| **Mechanisms of Acceleration** | Merely states that cannibalization is bad practice. | Notes that cannibalization increases maintenance work and damages airframes. | Formulates the reinforcing feedback loop: doubled labor hours, physical connector fatigue, wiring harness damage, induced No-Fault-Found rates, and the creation of unrecoverable "hangar queen" hulls. |
| **Remediation & Governance** | Proposes disciplinary action or court-martials for mechanics. | Proposes ordering more spare parts and banning cannibalization. | Delivers an actionable 5-point plan: (1) Mandatory METRIC rotable float re-baselining; (2) Daily Awaiting Parts priority tracking; (3) Strict O-6 commander authorization gates for cannibalization; (4) Rapid retrograde TP-1 logistics flow; (5) Aligning maintenance metrics to FMC availability rather than raw daily sortie counts. |

---

## 14. Interview questions

The following questions evaluate technical depth, operational judgment, and systems-level thinking in defence availability and performance-based logistics:

### Question 1: Contract structure trade-offs
*Interviewer prompt:* "When should the Department of Defense select a Firm-Fixed-Price per Flight Hour (FFP-PBTH) contract versus a Cost-Plus-Incentive-Fee (CPIF) contract for weapon system sustainment?"
- **Strong answer:** "The choice depends on system design maturity, operational tempo predictability, and baseline reliability stability. FFP-PBTH is optimal for mature platforms (e.g., F/A-18, C-17) where historical failure rates, depot turnaround distributions, and flight profiles are well characterized. In that environment, the contractor has full control over supply chain investments and can bear financial risk in exchange for commercial margins under FAR Part 12. Conversely, CPIF is necessary during early operational capability (e.g., F-35 early lots) or wartime surge deployments where operating tempo fluctuates unpredictably, system technical data baselines undergo continuous engineering changes, and unquantified reliability failure modes would cause commercial vendors to price in prohibitive risk premiums or risk insolvency."
- **Weak answer:** "Fixed-price contracts are always cheaper and better because they eliminate cost overruns for taxpayers, whereas cost-plus contracts are wasteful."

### Question 2: Mathematical validity of Palm's Theorem
*Interviewer prompt:* "Palm's Theorem states that the distribution of items in resupply depends only on the mean lead time, independent of the lead time distribution shape. When does this theorem fail in military rotable logistics?"
- **Strong answer:** "Palm's Theorem relies on two core assumptions: that customer demand arrivals follow a homogeneous Poisson process, and that service capacity is ample ($M/G/\infty$), meaning repair resources never experience queuing delays. In military operations, this fails under two distinct conditions. First, demand ceases to be Poisson during operational surges or combat deployments, where aircraft fly intensive package strikes, creating batched, correlated component failures. Second, repair facilities do not have infinite capacity; when depots experience test-bench bottlenecks, labor strikes, or piece-part shortages (AWP), carcasses wait in queue. The system becomes an $M/G/c$ or $G/G/c$ finite-capacity network where repair lead time becomes load-dependent and highly variable, necessitating the VARI-METRIC model."
- **Weak answer:** "Palm's Theorem fails whenever repair times are not normally distributed."

### Question 3: Proactive vs. reactive DMSMS management
*Interviewer prompt:* "A mission-critical LRU supplier issues a Last-Time-Buy notice for an obsolete FPGA. How do you evaluate whether to execute a Life-of-Type buy or initiate an immediate Form-Fit-Function redesign?"
- **Strong answer:** "We evaluate the trade-off through an asymmetric newsvendor optimization model comparing lifecycle holding costs against redesign penalties. Key inputs include remaining system service life ($T_{\text{rem}}$), discounted physical and dry-box storage costs, and the non-recurring engineering (NRE), flight recertification, and software baselining costs of an immediate redesign ($C_{\text{redesign}}$). Because an out-of-stock event grounds weapon systems, the underage cost ($C_u = C_{\text{redesign}}$) drastically exceeds the overage cost ($C_o = c + H$), resulting in a critical fractile often above 0.98. If remaining platform life is short (under 8–10 years) and the component demand is modest, a Life-of-Type buy is almost always economically superior. If platform life extends for 25+ years, component degradation risks in storage dominate, and the program should bundle the component redesign into an upcoming planned Block Upgrade cycle."
- **Weak answer:** "We should always buy all the parts available immediately so we never run out of stock."

### Question 4: Dissecting Operational vs. Inherent Availability
*Interviewer prompt:* "An avionics radar manufacturer demonstrates a contract-compliant Inherent Availability ($A_i$) of 98.5%, but the operational air wing using the radar reports an Operational Availability ($A_o$) of only 62%. What is happening?"
- **Strong answer:** "$A_i = \text{MTBF} / (\text{MTBF} + \text{MTTR}_u)$ is an engineering design metric measured in a laboratory or factory environment. It accounts only for active corrective wrench time under ideal conditions, omitting preventive maintenance and all supply delays. In the operational wing, $A_o = \text{MTBM} / (\text{MTBM} + \bar{M} + \text{ALDT})$. The 36.5% availability collapse is driven entirely by Administrative and Logistics Delay Time (ALDT). This indicates that when the radar fails, flight line maintainers wait days or weeks for a replacement LRU due to depleted base stock ($s=0$), slow retrograde transportation, depot repair backlogs, or test-bench diagnostic bottlenecks. The failure is not in the physical radar design, but in the multi-echelon sustainment pipeline."
- **Weak answer:** "The military maintainers are not properly trained and are taking too long to repair the radar on the flight line."

### Question 5: The VARI-METRIC refinement
*Interviewer prompt:* "Why does standard METRIC systematically underestimate rotable spare parts requirements at forward airbases when depot backorders are volatile?"
- **Strong answer:** "Classic METRIC assumes that demand arrivals at forward bases follow independent Poisson processes, where the variance-to-mean ratio (VMR) is exactly 1.0. However, when the central depot experiences stockouts ($EBO_D > 0$), multiple bases submit requisitions that queue simultaneously. When the depot completes a batch of repairs, it releases parts in clusters. This introduces compound Poisson dynamics and positive covariance into base replenishment pipelines, driving the VMR well above 1.0. Under high VMR, the tail of the backorder distribution is significantly fatter than a Poisson distribution. VARI-METRIC resolves this by computing the variance of depot delay and fitting a Negative Binomial distribution at the bases, correctly sizing the additional safety float needed to buffer depot instability."
- **Weak answer:** "VARI-METRIC is just a computer simulation version of METRIC that runs more iterations."

### Question 6: Mitigating the cannibalization death spiral
*Interviewer prompt:* "You take command of a logistics squadron where maintainers are aggressively cannibalizing rotable parts between aircraft. How do you halt the cycle without failing daily mission taskings?"
- **Strong answer:** "Halting cannibalization requires simultaneous intervention across inventory, maintenance policy, and command incentives. First, establish strict procedural hurdles: cannibalizations must require written authorization from the Maintenance Group Commander (O-6), stripping maintenance crews of the informal authority to swap parts. Second, perform immediate inventory reconciliation using METRIC calculations to identify the 5 to 10 'pacing' LRUs driving 80% of cannibalization events, and execute emergency priority requisitions (TP-1) or lateral redistributions from rear depots to establish at least one serviceable spare per line. Third, designate a controlled, single airframe as a dedicated heavy-maintenance donor rather than spreading part removals across multiple aircraft, protecting the remaining fleet. Finally, recalibrate command performance metrics: evaluate squadrons on Full Mission Capable (FMC) sustainability rather than rewarding short-term sortie generation achieved by destroying future readiness."
- **Weak answer:** "Issue a direct order banning all parts swapping immediately and punish anyone who disobeys."

### Question 7: Intellectual property and technical data rights
*Interviewer prompt:* "How do DFARS technical data rights impact a military service's ability to negotiate a cost-effective PBL contract?"
- **Strong answer:** "Technical data rights under DFARS 252.227-7013 determine market contestability. If the government owns Government Purpose Rights (GPR) or Level 3 Technical Data Packages, it retains the legal and technical authority to compete depot maintenance, qualify second-source suppliers for piece-parts, or transition sustainment to organic military air logistics complexes. If the prime contractor developed the system under private expense and holds Limited/Proprietary Rights, the government cannot share test program sets, manufacturing drawings, or source codes. The contractor achieves a permanent sustainment monopoly. In that scenario, negotiating a commercial-style PBL requires aggressive statutory auditing (such as Truth in Negotiations Act requirements) to prevent monopoly rent extraction, or paying upfront licensing fees to acquire depot-level maintenance data."
- **Weak answer:** "Technical data rights only matter if the military decides to manufacture its own aircraft from scratch."

### Question 8: Addressing No-Fault-Found (NFF) avionics churn
*Interviewer prompt:* "Your rotable pipeline is clogged with avionics LRUs exhibiting a 35% No-Fault-Found rate at intermediate test stations. What analytics and engineering steps do you take?"
- **Strong answer:** "High NFF rates indicate intermittent mechanical or thermal faults, backplane connector degradation, or inadequate flight line diagnostics. First, analyze component removal telemetry using survival analysis to determine whether specific airframe tail numbers repeatedly reject different LRUs on the same bus; this isolates wiring harness faults from box-level defects. Second, deploy advanced environmental stress screening (ESS)—combining thermal profiling and multi-axis vibration testing on intermediate test benches—to replicate the dynamic inflight environment that static test benches miss. Third, audit Built-In Test (BIT) logic to eliminate overly sensitive fault thresholds that trigger false alarms during transient voltage dips. Finally, tag and track individual serialized carcasses; if a specific serial number records two consecutive NFF events, route it directly to specialized depot-level teardown rather than returning it to serviceable float stock."
- **Weak answer:** "Retrain the test bench operators to test the boxes more carefully before sending them back."

---

## 15. Key terms

The following glossary defines the technical terminology used in defence availability modeling and performance-based logistics:

- **Achieved Availability ($A_a$):** The availability metric incorporating both corrective and preventive active wrench-turning maintenance, excluding logistics and administrative delays ($A_a = \text{MTBM} / [\text{MTBM} + \bar{M}]$).
- **Administrative and Logistics Delay Time (ALDT):** The cumulative downtime during which an unserviceable weapon system awaits spare parts, specialized tooling, maintenance personnel, or administrative release.
- **Autonomic Logistics Information System (ALIS):** The legacy proprietary IT sustainment backbone developed for the F-35, intended to automate rotable tracking, health monitoring, and supply replenishment.
- **Built-In Test (BIT):** Integrated on-board diagnostic hardware and firmware designed to detect, isolate, and log component failures without external test equipment.
- **Cannibalization:** The authorized or unauthorized removal of a serviceable part from one unserviceable weapon system to install on another to restore it to operational status.
- **Carcass:** The physical unserviceable core of a rotable assembly removed from a platform, legally controlled and tracked through retrograde overhaul pipelines.
- **Cannot Duplicate (CND):** An alternative industry designation for No-Fault-Found (NFF) bench test results.
- **Cost-Plus-Incentive-Fee (CPIF):** A contract mechanism where the customer reimburses allowable costs and adjusts contractor fee via a mathematical sharing ratio based on cost and performance targets.
- **Diminishing Manufacturing Sources and Material Shortages (DMSMS):** The loss or impending loss of manufacturers or suppliers of critical components, raw materials, or software.
- **Expected Backorders ($EBO$):** The steady-state expected number of outstanding, unfulfilled customer requisitions across an inventory network.
- **Firm-Fixed-Price per Flight Hour (FFP-PBTH):** A commercial-style sustainment contract where the vendor receives a fixed payment per operating hour and assumes all financial risk for parts and repairs.
- **Form, Fit, Function (FFF):** A replacement component physically, electrically, and operationally interchangeable with an obsolete item without modifying mating interfaces or software.
- **Fully Mission Capable (FMC):** The operational status of a weapon system capable of executing all assigned primary combat and training missions.
- **Inherent Availability ($A_i$):** An idealized engineering availability metric accounting only for design reliability and active corrective repair time ($A_i = \text{MTBF} / [\text{MTBF} + \text{MTTR}_u]$).
- **Line Replaceable Unit (LRU):** A modular weapon system assembly designed for rapid removal and replacement at the flight line (O-Level).
- **Life-of-Type (LOT) Buy:** A single, final procurement order sized to provide sufficient spare components to support a weapon system through its planned retirement date.
- **METRIC:** Multi-Echelon Technique for Recoverable Item Control; an optimization methodology developed by the RAND Corporation for multi-echelon repairable inventories.
- **Mean Down Time (MDT):** The total elapsed clock time required to restore an unserviceable weapon system to operational capability ($\text{MDT} = \bar{M} + \text{ALDT}$).
- **Mean Time Between Maintenance (MTBM):** The mean operational operating hours accumulated between maintenance events, including both scheduled and unscheduled inductions.
- **Mission Capable (MC) Rate:** The percentage of total possessed weapon systems capable of performing at least one designated mission.
- **No-Fault-Found (NFF):** A condition where a component removed for suspected failure tests satisfactorily on intermediate or depot diagnostic benches.
- **Non-Mission Capable Supply (NMCS):** The operational status or time duration during which a weapon system is grounded awaiting spare parts.
- **Not Repaired This Station (NRTS):** A maintenance determination that an unserviceable rotable cannot be repaired at the local base facility and must be evacuated to a depot.
- **Operational Availability ($A_o$):** The real-world probability that a weapon system is mission capable at a random point in time ($A_o = \text{MTBM} / [\text{MTBM} + \text{MDT}]$).
- **Operational Data Integrated Network (ODIN):** The cloud-native cloud IT architecture replacing ALIS across the global F-35 enterprise.
- **Palm's Theorem:** A mathematical theorem establishing that an infinite-server queue with Poisson arrivals of rate $\lambda$ and arbitrary service times $T$ achieves a Poisson steady-state pipeline with mean $\lambda E[T]$.
- **Performance-Based Logistics (PBL):** An outcomes-based acquisition strategy contracting for weapon system operational readiness rather than transactional parts and labor.
- **Power by the Hour (PBTH):** A sustainment contracting model originating in commercial aviation where operators pay a fixed fee per engine or airframe operating hour.
- **Retest OK (RTOK):** Industry terminology synonymous with No-Fault-Found (NFF).
- **Shop Replaceable Unit (SRU):** A subassembly or circuit card internal to an LRU, designed for repair at intermediate workshops or industrial depots.
- **VARI-METRIC:** An advanced extension of METRIC calculating the variance-to-mean ratio (VMR) of depot backorders to model base pipelines via Negative Binomial distributions.

---

## 16. Sources

The following official publications, military standards, peer-reviewed journal articles, and academic monographs constitute the authoritative literature on defence availability, rotable inventory optimization, and performance-based logistics:

- **Defense Standardization Program Office (DSPO).** (2022). *SD-22: Diminishing Manufacturing Sources and Material Shortages (DMSMS) Guidebook*. Fort Belvoir, VA: Defense Logistics Agency.
- **Department of Defense (DoD).** (2021). *DoD Instruction 5000.91: Product Support Management for the Adaptive Acquisition Framework*. Office of the Under Secretary of Defense for Acquisition and Sustainment.
- **Department of Defense (DoD).** (2016). *Performance-Based Logistics (PBL) Guidebook: A Guide to Developing Performance-Based Product Support Arrangements*. Fort Belvoir, VA: Defense Acquisition University (DAU).
- **General Accounting Office / Government Accountability Office (GAO).** (2021). *F-35 Sustainment: Enhanced Planning Needed to Address Availability and Engine Shortages* (GAO-21-439). Washington, DC: U.S. Government Accountability Office.
- **Government Accountability Office (GAO).** (2023). *F-35 Aircraft: Actions Needed to Improve Availability and Modernization Monitoring* (GAO-23-105341). Washington, DC: U.S. Government Accountability Office.
- **Graves, S. C.** (1985). A Multi-Echelon Inventory Model for a Repairable Item with One-for-One Replenishment. *Management Science*, 31(10), 1247–1256. https://doi.org/10.1287/mnsc.31.10.1247
- **Kim, S.-H., Cohen, M. A., & Netessine, S.** (2007). Performance Contracting in After-Sales Service Supply Chains. *Management Science*, 53(12), 1843–1858. https://doi.org/10.1287/mnsc.1070.0743
- **Muckstadt, J. A.** (1973). A Model for a Multi-Item, Multi-Echelon, Multi-Indenture Inventory System. *Management Science*, 20(4), 472–481. https://doi.org/10.1287/mnsc.20.4.472
- **Sherbrooke, C. C.** (1968). METRIC: A Multi-Echelon Technique for Recoverable Item Control. *Operations Research*, 16(1), 122–141. https://doi.org/10.1287/opre.16.1.122
- **Sherbrooke, C. C.** (1986). VARI-METRIC: Improved Approximations for Multi-Indentured, Multi-Echelon Availability Models. *Operations Research*, 34(2), 311–319. https://doi.org/10.1287/opre.34.2.311
- **Sherbrooke, C. C.** (2004). *Optimal Inventory Modeling of Systems: Multi-Echelon Techniques* (2nd ed.). Boston, MA: Kluwer Academic Publishers. https://doi.org/10.1007/b101153
- **UK Ministry of Defence (MoD).** (2020). *Defence Standard 00-600: Integrated Logistic Support Requirements for MOD Projects* (Issue 4). Glasgow, UK: Defence Equipment and Support (DE&S).
- **U.S. Department of Defense.** (1998). *MIL-HDBK-338B: Electronic Reliability Design Handbook*. Washington, DC: U.S. Department of Defense.

---

## 17. Links

### Internal course links
- Module index: [`_index.md`](_index.md)
- Prior topic: [Topic 02: Whole-Life Cost and MEAT Evaluation](02_whole-life-cost-meat.md)
- Next topic: [Topic 04: Strategic Stockpiles](04_strategic-stockpiles.md)
- Upstream procurement foundations: [Topic 01: Public Procurement and Frameworks](01_public-procurement-frameworks.md)
- Survey file: [`../../../course/35_public_sector_defence_humanitarian.md`](../../../course/35_public_sector_defence_humanitarian.md)

### Cross-module links
- [Module 10: Inventory Optimization](../../02_methods/10_inventory_optimization/_index.md)
- [Module 11: Safety Stock & Service Levels](../../02_methods/11_safety_stock_service_levels/_index.md)
- [Module 15: Maintenance, Repair, and Operations (MRO) Inventory](../../02_methods/15_mro_inventory_management/_index.md)
- [Module 31: Topic 03: Multi-Echelon Sparing Models](../31_aftermarket_spare_parts_service/03_multi-echelon-sparing.md)
- [Module 31: Topic 04: Repair Loop & Turnaround Optimization](../31_aftermarket_spare_parts_service/04_repair-loop-turnaround.md)
- [Module 31: Topic 06: Service Contracts, SLA & Warranty Economics](../31_aftermarket_spare_parts_service/06_service-contracts-sla.md)
- [Module 34: Topic 02: MRO Demand & Failure Modelling](../34_energy_utilities_capital_projects/02_mro-failure-modelling.md)
