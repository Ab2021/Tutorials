> Module 31 — Aftermarket, Spare Parts, and Service Supply Chains
> Topic 05 of 09
> Estimated study time: 4.0 hours
> Prerequisites: Module 31 Topic 01 (Installed Base & Failure Demand), Module 31 Topic 02 (Criticality Classification), Module 12 (Transportation Routing & Last-Mile)
> Version 2026.5

# Field Service Scheduling

---

## 1. Why this topic exists

In asset-intensive service contracts, **spare parts on a warehouse shelf produce zero customer value**. 

A customer whose $3 million semiconductor photolithography stepper, hospital MRI scanner, or electric power substation transformer is down does not buy physical parts; the customer buys **uptime**. Delivering uptime requires the precise, synchronized physical convergence of three operational elements at the customer's equipment site within a contractually binding service window:
1. **The Right Replacement Part:** Compatible with the asset's specific engineering Bill of Materials (BOM) revision.
2. **The Right Certified Technician:** Possessing the exact regulatory licenses, safety certifications, and diagnostic skills required to perform the repair.
3. **The Right Specialized Tooling:** Calibrated alignment fixtures, high-voltage test meters, and lifting rigs.

When any one of these three elements is missing, the service event fails catastrophically:
* If a master technician arrives on site within the two-hour emergency response window, but the required high-voltage power board was not stocked in their service van or forward depot, the technician can only verify the failure and leave. The asset remains down.
* If the replacement board is delivered to the hospital loading dock by hot-shot courier, but the dispatched field engineer is only certified on CT scanners and lacks the high-field safety certification to work inside an active 3.0T MRI magnetic fringe field, the repair cannot legally proceed. The asset remains down.
* If both the part and the certified technician are present, but the technician's calibrated fiber-optic spectrometer is out of annual calibration certification, statutory quality rules prohibit signing off the machine. The asset remains down.

The critical metric of field service supply chains is the **First-Time Fix Rate (FTFR)**: the percentage of service dispatches that successfully restore the asset to full operational specification on the initial visit, without requiring a return trip.

In modern field operations, FTFR typically hovers between 65% and 80%. When FTFR drops from 85% to 65%:
* **Operating Costs Explode:** Every uncompleted service call triggers a secondary truck roll. The organization pays duplicate technician travel hours, duplicate fuel, and administrative reschedule costs. Field operating expenses increase by 30% to 50%.
* **Downtime Penalties Cascade:** Under modern Performance-Based Logistics (PBL) and mission-critical SLAs, downtime beyond the response window incurs contractual liquidated damages of $2,000 to $25,000 per hour.
* **Technician Capacity Implodes:** Having technicians spend 40% of their working hours driving back and forth on "re-visits" starves the network of skilled labor capacity, creating an artificial technician shortage that inflates service backlogs.

Field service scheduling cannot be solved using generic sales-rep routing or package-delivery vehicle routing problem (VRP) algorithms. Field service requires solving a **Multi-Depot Vehicle Routing Problem with Time Windows, Dynamic Skills Matching, Van Stock Inventory Constraints, and Contractual Penalty Functions (MD-VRPTW-SS)**. This topic teaches the mathematical formulation, algorithmic heuristics, and operational workflows required to schedule field engineers and synchronize technician movement with spare parts logistics.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Formulate** the mathematical Triad of Field Service Execution: the multi-dimensional constraint mapping between Technician Skills, Van Stock Inventory, and Asset Service Windows.
2. **Deconstruct** First-Time Fix Rate (FTFR) into its root-cause failure components: parts deficit, skill mismatch, diagnostic misidentification, and customer access restriction.
3. **Formulate** the Multi-Depot Vehicle Routing Problem with Time Windows and Skill Constraints (MD-VRPTW-S) as a Mixed-Integer Linear Program (MILP).
4. **Implement** adaptive metaheuristics (Adaptive Large Neighborhood Search / ALNS, Insertion Heuristics) to generate optimal daily field technician schedules under tight computational deadlines.
5. **Design** dynamic real-time dispatch mechanisms that adjust schedules mid-day to accommodate high-priority emergency breakdown interrupts (Priority 1 / AOG calls) without destabilizing scheduled preventive maintenance commitments.
6. **Size** mobile van stock (trunk stock) profiles balancing technician vehicle payload limits, inventory holding costs, and local first-time-fix probability.
7. **Integrate** statutory labor constraints, including the EU Working Time Directive, US Department of Transportation Hours of Service (DOT HOS), and mandatory safety rest periods into automated scheduling algorithms.
8. **Structure** digital post-service debrief workflows that capture root-cause repair actions, reconcile consumed van stock, and feed diagnostic accuracy back into upstream forecasting models.

---

## 3. The core mental model

Field service scheduling operates at the intersection of **human capital**, **mobile inventory**, and **spatial-temporal constraints**:

```text
               +--------------------------------------------------------+
               | WORK ORDER INFLOW (CMMS / IoT Alerts)                 |
               |  • Asset ID & Location (Lat/Long)                      |
               |  • Failure Symptoms & Presumptive Parts BOM            |
               |  • Mandatory Skill / Certification Level              |
               |  • SLA Service Time Window [e_i, l_i]                  |
               +───────────────────────────┬────────────────────────────+
                                           │
                                           ▼
               +--------------------------------------------------------+
               | THE TRIPLE FEASIBILITY FILTER                          |
               |                                                        |
               |  1. SKILLS FILTER:                                     |
               |     Technician Certifications >= Job Requirement       |
               |                                                        |
               |  2. PARTS FEASIBILITY FILTER:                          |
               |     Parts in Van Stock OR Forward Stocking Loc. (FSL)  |
               |                                                        |
               |  3. TEMPORAL & WORKING HOURS FILTER:                   |
               |     Arrival Time + Service Time <= Window Deadline     |
               |     Daily Driving + Wrench Time <= Statutory HOS Max   |
               +───────────────────────────┬────────────────────────────+
                                           │
                                           ▼
+───────────────────────────────────────────────────────────────────────────────+
| OPTIMIZATION ROUTING & DISPATCH ENGINE                                        |
|                                                                               |
|   Min: Total Travel Distance + Overtime Costs + SLA Penalty Breach Damages    |
|                                                                               |
|   [ Technician 1 (Senior FSE) ] ──> [ FSL Pickup ] ──> [ Job 1 (P1 SLA) ] ──┐ |
|                                                                             │ |
|   [ Technician 2 (Junior FSE) ] ──────────────────────> [ Job 2 (PM Kit) ] ─┴─►
+───────────────────────────────────────────────────────────────────────────────+
                                           │
                                           ▼
               +--------------------------------------------------------+
               | FIELD EXECUTION & OUTCOME EVALUATION                   |
               |  • First-Time Fix Achieved? (Yes / No)                 |
               |  • Van Stock Replenished via Nightly Courier Sweep?    |
               |  • Actual Wrench Time vs. Planned Standard Time        |
               +--------------------------------------------------------+
```

### 3.1 The Three Essential Filters

Before any scheduling algorithm attempts to optimize routes or minimize drive time, it must pass candidate technician-job pairings through three strict feasibility filters:
1. **The Skills and Competency Filter:** Unlike parcel delivery (where any licensed driver can deliver any cardboard box), field service requires deep specialization. A field service engineer (FSE) certified on gas turbine combustor overhauls cannot safely commission a digital excitation control cabinet. Assigning an unqualified technician yields a zero percent probability of repair and creates immense life-safety liability.
2. **The Parts and Tooling Filter:** A job cannot be scheduled to an engineer unless the required parts are physically present in the engineer's vehicle trunk stock, or a route deviation is planned to collect the parts from a Forward Stocking Location (FSL) or courier drop-box prior to arriving at the customer site.
3. **The Temporal and Regulatory Filter:** The technician must be able to reach the site within the customer's contractual service window $[e_i, l_i]$ without violating statutory driving limits (e.g., maximum 10 hours driving per day under US DOT rules).

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse | Unit or Coding Case |
|---|---|---|---|
| **First-Time Fix Rate (FTFR)** | The percentage of field service calls where the customer's equipment problem is fully resolved during the initial on-site visit without requiring a return dispatch: $\text{FTFR} = \frac{N_{\text{single-visit fixes}}}{N_{\text{total dispatches}}}$. | Excluding return visits caused by missing parts from the metric. | Percentage: $[0, 100\%]$. |
| **Service Window (Time Window)** | The contractually agreed time interval $[e_i, l_i]$ during which technician arrival or repair completion must occur. | Treating time windows as soft preferences rather than hard legal boundaries. | Timestamp interval: `[start_datetime, end_datetime]`. |
| **Van Stock (Trunk Stock)** | The mobile inventory of high-velocity, mission-critical spare parts, consumables, and diagnostic tools carried inside a field technician's vehicle. | Treating van stock as an untracked, invisible "black box" buffer. | Serialized / SKU inventory table tagged to `TechnicianID`. |
| **Skills Matrix** | A verified, auditable competency mapping matrix $\mathbf{M}_{\text{skill}}$ defining each technician’s proficiency and legal certification across specific machine models and maintenance procedures. | Relying on self-reported technician skill claims without formal training records. | Binary or integer matrix: $m_{k,s} \in \{0, 1, 2, 3\}$. |
| **Mean Time to Respond (MTTR_resp)** | The elapsed calendar time from the initial customer service request or automated IoT alarm generation to the physical arrival of the technician on site. | Conflating response time with repair completion time. | Hours or Minutes ($[T]$). |
| **Mean Time to Repair (MTTR_repair)** | The elapsed time from the start of technician diagnostic wrench work at the customer site to the successful sign-off and functional restoration of the machine. | Counting travel time as part of MTTR_repair (travel belongs to response time). | Hours or Minutes ($[T]$). |
| **Forward Stocking Location (FSL)** | A small, localized warehouse, automated parts locker, or courier depot positioned in metropolitan centers to supply emergency parts within 1–2 hours. | Stocking full multi-echelon inventory catalogs at an FSL (FSLs should hold only Quadrant I/II emergency items). | Geographic facility ID; coordinates $(x, y)$. |
| **Nightly Boot Sweep (Trunk Replenishment)** | The logistics practice where third-party courier drivers access field service vans overnight using electronic lockbox keys to restock consumed parts and collect unserviceable cores. | Forcing technicians to drive 45 minutes every morning to a central warehouse to pick up parts. | Scheduled overnight logistics cycle: 00:00–05:00. |
| **Hours of Service (HOS)** | Statutory labor regulations restricting maximum daily driving hours, maximum on-duty working hours, and mandatory minimum rest periods between shifts. | Scheduling overtime that violates legal driver fatigue limits, creating corporate tort liability. | Hours per day/week ($[T]$). |
| **Dynamic Rescheduling** | The algorithmic capability to re-optimize existing technician routes in real time during the operating day in response to emergency breakdown interrupts or travel delays. | Cancelling scheduled preventive maintenance calls without evaluating customer SLA penalty impacts. | Event-triggered execution (arrival of Priority-1 ticket). |
| **Service Level Agreement (SLA) Penalty** | Contractual liquidated damages assessed against the service provider for every hour of asset downtime exceeding guaranteed response or resolution windows. | Treating SLA penalties as a customer relations dispute rather than an explicit cost parameter in the routing objective function. | Currency per unit time (e.g., $\$2,500/\text{hour}$). |
| **P1 / Emergency Call** | The highest priority service ticket, triggered by a complete machine breakdown on a critical production line or life-support asset. | Assigning P1 priority to routine customer maintenance requests to bypass standard queues. | Priority code: `P1_CRITICAL`. |

---

## 5. The mechanics, step by step

Optimizing field service dispatch requires an integrated sequence of eight analytical procedures. The framework links work order intake, skills validation, van stock optimization, and dynamic route synthesis.

```text
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 1: Ingest Work Orders and Execute Presumptive Triage Profiling             │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 2: Query Technician Skills Matrix and Filter Candidate Assignees           │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 3: Verify Van Stock & Forward Stocking Location (FSL) Parts Availability   │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 4: Construct the Mathematical Multi-Depot VRPTW Formulation                │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 5: Solve Static Daily Routes via Adaptive Large Neighborhood Search (ALNS)  │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 6: Execute Real-Time Dynamic Dispatch for Emergency P1 Interrupts          │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 7: Size and Optimize Mobile Van Stock Profiles via Marginal FTFR Analysis  │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 8: Execute Digital Debrief, Core Harvest Logistics, and Nightly Van Restock│
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Step 1: Ingest Work Orders and Execute Presumptive Triage Profiling

When an equipment failure occurs, the field service management (FSM) system must not wait for a technician to arrive on site to discover what parts and tools are required.
* **Input:** Customer trouble ticket, call-center transcript, IoT telemetry error codes, historical failure records for identical machine models.
* **Presumptive Parts Profiling:**
  Using natural language processing (NLP) on error logs and Bayesian diagnostic networks:
  $$P(\text{Part } p \text{ required} \mid \text{Error Code } E_k, \text{Asset Model } M_m)$$
  Generate a **Presumptive Bill of Materials (PBOM)** listing the top 1 to 3 components likely required to complete the repair, with their corresponding probability thresholds:
  $$\text{PBOM}_i = \{ p \in \text{Parts} \mid P(p \mid E_k) \ge \theta_{\text{threshold}} \}$$
* **SLA Time Window Extraction:**
  Extract the contractual arrival window $[e_i, l_i]$ and deadline resolution timestamp $t_{\text{deadline}, i}$.
  * Priority 1 (Emergency Breakdown): $l_i = t_{\text{call}} + 2 \text{ hours}$.
  * Priority 2 (Degraded Operation): $l_i = t_{\text{call}} + 8 \text{ hours}$.
  * Priority 3 (Scheduled Preventive Maintenance): $[e_i, l_i] = [\text{08:00 tomorrow}, \text{17:00 tomorrow}]$.

### 5.2 Step 2: Query Technician Skills Matrix and Filter Candidate Assignees

Field technicians possess varying levels of proficiency and statutory certifications.
* **Competency Hierarchy:**
  * Level 1 (Apprentice / Basic): Filter changes, visual inspections, simple mechanical bolt replacements.
  * Level 2 (Certified Field Service Engineer): Standard electronic troubleshooting, subassembly replacement, calibration.
  * Level 3 (Master Specialist): Major mechanical rebuilds, laser interferometer alignments, complex firmware/bus debugging.
* **Formulation of Skills Compatibility:**
  Let $S_{\text{req}, i}$ denote the set of mandatory certifications required by Job $i$. Let $\mathcal{S}_k$ denote the verified certification portfolio of Technician $k$.
  The binary assignment indicator $a_{ik} \in \{0, 1\}$ is constrained by:
  $$a_{ik} = 1 \iff S_{\text{req}, i} \subseteq \mathcal{S}_k$$
  If Technician $k$ lacks even one required certification (e.g., radiation safety license, high-voltage offshore permit), $a_{ik} = 0$ is enforced as a hard mathematical constraint.

### 5.3 Step 3: Verify Van Stock & Forward Stocking Location (FSL) Parts Availability

A qualified technician cannot achieve a fix without parts. The system checks three fulfillment tiers:
1. **Tier 1 (Immediate Van Stock):** Are all parts in $\text{PBOM}_i$ physically present in Technician $k$'s mobile van inventory?
   $$\text{If } \text{PBOM}_i \subseteq \mathcal{V}_k \implies \text{Direct Dispatch Feasible (Zero Logistics Delay)}$$
2. **Tier 2 (Forward Stocking Location Pickup):** If parts are missing from van stock, can Technician $k$ detour to Forward Stocking Location $f$ to pick up the pre-staged parts without violating the arrival deadline $l_i$?
   $$\text{Arrival Time} = t_{\text{current}} + t_{\text{drive}}(k \to f) + t_{\text{pickup}} + t_{\text{drive}}(f \to i) \le l_i$$
3. **Tier 3 (Third-Party Hot-Shot Courier):** Can an emergency courier dispatch the part directly from a central depot to Site $i$ to rendezvous with the technician?

### 5.4 Step 4: Construct the Mathematical Multi-Depot VRPTW Formulation

We formulate the daily field service scheduling problem as a Mixed-Integer Linear Program.
* **Network Graph:** Let $\mathcal{G} = (\mathcal{V}, \mathcal{A})$ be a directed graph.
  * Vertices $\mathcal{V} = \mathcal{V}_{\text{depot}} \cup \mathcal{V}_{\text{jobs}} \cup \mathcal{V}_{\text{fsl}}$.
  * Technicians $k \in \mathcal{K}$, each starting at home/depot $o(k)$ and ending at $d(k)$.
  * Travel time between node $i$ and node $j$: $t_{ij}$.
  * Distance / Travel cost between node $i$ and node $j$: $c_{ij}$.
  * Service duration at Job $i$: $d_i$.
  * Customer contractual time window: $[e_i, l_i]$.
  * Contractual SLA penalty cost per unit time of late arrival: $P_{\text{sla}, i}$.
* **Decision Variables:**
  * $x_{ijk} \in \{0, 1\}$: 1 if Technician $k$ traverses arc $(i, j)$; 0 otherwise.
  * $T_{ik} \ge 0$: Arrival time of Technician $k$ at node $i$.
  * $z_i \ge 0$: Tardiness / SLA breach duration at Job $i$: $z_i = \max(0, T_{ik} - l_i)$.
* **Objective Function:**
  $$\min \sum_{k \in \mathcal{K}} \sum_{(i,j) \in \mathcal{A}} c_{ij} x_{ijk} + \sum_{k \in \mathcal{K}} C_{\text{overtime}, k} \cdot \text{OT}_k + \sum_{i \in \mathcal{V}_{\text{jobs}}} P_{\text{sla}, i} \cdot z_i$$
* **Key Constraints:**
  1. *Job Coverage:* Each job is visited at most once: $\sum_{k \in \mathcal{K}} \sum_{j} x_{ijk} \le 1$.
  2. *Skills Matching:* $\sum_{j} x_{ijk} \le a_{ik}$ (only qualified technicians can visit node $i$).
  3. *Flow Conservation:* Route starts at $o(k)$ and terminates at $d(k)$ for each technician.
  4. *Time Window Dynamics:* If $x_{ijk} = 1 \implies T_{jk} \ge T_{ik} + d_i + t_{ij}$.
  5. *Statutory Hours of Service (HOS):* Total daily work duration $\le H_{\max}$:
     $$\sum_{(i,j) \in \mathcal{A}} (t_{ij} + d_i) x_{ijk} \le H_{\max} \quad (\text{e.g., } 10 \text{ hours})$$

### 5.5 Step 5: Solve Static Daily Routes via Adaptive Large Neighborhood Search (ALNS)

Because MD-VRPTW is NP-hard, commercial field operations running hundreds of technicians and thousands of daily work orders deploy **Adaptive Large Neighborhood Search (ALNS)**:
1. **Initial Solution Generation:** Construct a feasible baseline schedule using a greedy regret-insertion heuristic: in each iteration, select the unassigned job that has the highest difference in insertion cost between its best technician and second-best technician (regret-2 metric).
2. **Destroy Operators:** In each ALNS iteration, remove $p$ jobs from the current schedule using:
   * *Random Removal:* Diversifies search space.
   * *Worst-Cost Removal:* Removes jobs that contribute the highest marginal travel distance or SLA penalties.
   * *Shaw (Relatedness) Removal:* Removes jobs that are geographically proximate and require similar skill profiles.
3. **Repair Operators:** Re-insert the removed jobs using:
   * *Greedy Best-Insertion:* Places jobs at minimum marginal cost.
   * *Deep Regret-k Insertion:* Prioritizes jobs with few feasible alternative slots.
4. **Acceptance Criterion:** Use Simulated Annealing ($P(\text{Accept}) = \exp(-\Delta \text{Cost} / T)$) to escape local minima, gradually cooling temperature $T$.

### 5.6 Step 6: Execute Real-Time Dynamic Dispatch for Emergency P1 Interrupts

During the operational day, plans are disrupted by sudden Priority 1 breakdowns (e.g., an airport baggage scanner fails, or an offshore production pump trips).
* **Dynamic Insertion Protocol:**
  When emergency ticket $J_{\text{new}}$ arrives at timestamp $t_{\text{now}}$:
  1. Freeze all technician activities currently in progress (active wrench work cannot be aborted).
  2. Identify the subset of active technicians $\mathcal{K}_{\text{qual}} \subseteq \mathcal{K}$ who possess the required skills and van stock.
  3. For each candidate technician $k \in \mathcal{K}_{\text{qual}}$, evaluate the insertion of $J_{\text{new}}$ into their remaining afternoon schedule:
     $$\Delta \text{Cost}_k = \text{Added Travel Cost} + \Delta \text{Overtime} + \sum_{m \in \text{downstream}} \Delta \text{SLA Penalty}_m$$
  4. Select the technician with minimum total system disruption cost. Downstream non-critical preventive maintenance jobs may be bumped to the following day to preserve the emergency SLA.

### 5.7 Step 7: Size and Optimize Mobile Van Stock Profiles via Marginal FTFR Analysis

A field service vehicle cannot carry the entire warehouse catalog. Van space is limited by physical volume, payload weight limits, and inventory holding budgets.

Let $v \in \mathcal{V}$ denote the vehicle storage capacity limit (in kilograms or volume units $w_p$). We solve the **Knapsack Van Stock Problem**:
$$\max \sum_{p \in \text{Catalog}} P(\text{Part } p \text{ demanded on route}) \cdot \text{FTFR Impact}_p$$
$$\text{subject to} \sum_{p} w_p \cdot s_p \le W_{\max} \quad \text{and} \quad \sum_{p} c_p \cdot s_p \le B_{\text{van}}$$

Deploy **Marginal Analysis**: Rank parts by the ratio of their expected marginal first-time-fix contribution to their physical weight and inventory holding cost:
$$\text{Priority Ratio}_p = \frac{\lambda_p \cdot P(\text{Fix} \mid p)}{c_p \cdot r_h + \theta \cdot w_p}$$
Load parts into the technician's vehicle until payload weight or working capital budgets are exhausted. High-failure, lightweight electronic boards and seals populate van stock; heavy, low-failure structural steel beams remain at regional distribution centers.

### 5.8 Step 8: Execute Digital Debrief, Core Harvest Logistics, and Nightly Van Restock

The field service loop closes when the technician completes the repair:
1. **Digital Work Order Closure:** The technician logs the exact serial number of the failed carcass removed, the serial number of the new part installed, the actual wrench time, and the root-cause failure mode code.
2. **Unserviceable Carcass Packaging:** The removed unserviceable core is bagged, tagged with an electronic RMA barcode, and placed into the vehicle's secure reverse-logistics bin.
3. **The Nightly Trunk Sweep:** At 02:00, while the technician is sleeping, a 3PL dedicated courier accesses the technician’s van parked in their residential driveway using a secure digital rolling-code lockbox. The courier:
   * Drops off replenishment parts consumed during yesterday's jobs.
   * Scans and collects the unserviceable carcasses, delivering them to the central MRO repair facility by 06:00.
   * The technician awakens with a 100% replenished van stock, ready for the morning route without setting foot in a warehouse.

---

## 6. Worked example

### 6.1 Setting: Medical Imaging Field Service in Metro Chicago

BioMed Solutions provides emergency and scheduled field maintenance for 8 regional hospitals across the Greater Chicago metropolitan area. It operates a fleet of **4 Field Service Engineers (FSEs)** operating from home bases, supported by a central Forward Stocking Location (FSL) near O'Hare Airport (ORD).

```text
Technician Fleet Profiles:
======================================================================================================
FSE ID   Name           Home Location      Base Hourly Rate  Overtime Rate   Verified Skills Matrix
------------------------------------------------------------------------------------------------------
FSE-1    Dave (Senior)  Naperville (West)  $65.00/hr         $97.50/hr       Level 3: MRI, CT, X-Ray
FSE-2    Elena (Mid)    Evanston (North)   $50.00/hr         $75.00/hr       Level 2: CT, X-Ray (No MRI)
FSE-3    Marcus (Mid)   Oak Lawn (South)   $50.00/hr         $75.00/hr       Level 2: MRI, Ultrasound
FSE-4    Sarah (Junior) Loop (Downtown)    $40.00/hr         $60.00/hr       Level 1: X-Ray, PM Overhauls
======================================================================================================
Statutory working limit: Maximum 8.0 standard working hours per day. Max 2.0 hours overtime.
Travel cost metric: $0.85 per mile driven.
```

At 06:30, the dispatch desk must schedule **6 Work Orders** across the Chicago metro area for the day:

```text
Daily Work Order Manifest:
==================================================================================================================
Job ID  Hospital Site          System   Job Type        Window [e, l]   Duration  Required Skill   Parts Needed
------------------------------------------------------------------------------------------------------------------
Job-A   St. Alexius (Hoffman)  MRI 3.0T P1 Emergency    [08:00, 10:00]  2.5 hrs   Level 2+ MRI     Cryo Valve (In FSL)
Job-B   NorthShore (Evanston)  CT 128   P1 Emergency    [08:30, 10:30]  2.0 hrs   Level 2+ CT      X-Ray Tube (In Van 2)
Job-C   Advocate (Oak Lawn)    MRI 3.0T P2 Diagnostic   [11:00, 15:00]  2.0 hrs   Level 2+ MRI     RF Board (In Van 3)
Job-D   Rush Univ (Medical Dt) CT 128   P3 Scheduled PM [09:00, 17:00]  3.0 hrs   Level 1+ CT      PM Kit (In Van 1, 2)
Job-E   Loyola Med (Maywood)   X-Ray    P2 Breakdown    [12:00, 16:00]  1.5 hrs   Level 1+ X-Ray   Detector Cable (Van 4)
Job-F   Northwestern (Downtown)Ultrasnd P3 Scheduled PM [13:00, 17:00]  2.0 hrs   Level 1+ UltrasndPM Kit (In Van 3)
==================================================================================================================
SLA Penalties:
• Job-A & Job-B (P1 Emergencies): $2,000/hour for every hour arrival exceeds deadline 'l'.
• Job-C & Job-E (P2 Calls): $500/hour late arrival penalty.
• Job-D & Job-F (P3 Scheduled PM): $100/hour late arrival penalty.
```

---

### 6.2 Distance and Travel Time Matrix

Driving travel times (in minutes) and distances (in miles) computed via routing telematics under morning peak traffic:

```text
Travel Time Matrix (Minutes / Miles):
========================================================================================================
From / To       FSL (ORD)    Job-A (Hoff) Job-B (Evan) Job-C (OakL) Job-D (Rush) Job-E (Loy)  Job-F (NWM)
--------------------------------------------------------------------------------------------------------
FSE-1 (Naperv)  35m / 24mi   40m / 28mi   65m / 42mi   45m / 26mi   50m / 32mi   35m / 22mi   55m / 34mi
FSE-2 (Evanst)  30m / 18mi   45m / 26mi    5m / 2mi    55m / 32mi   35m / 16mi   40m / 20mi   30m / 14mi
FSE-3 (OakLawn) 45m / 28mi   60m / 38mi   55m / 32mi    8m / 3mi    30m / 15mi   25m / 14mi   35m / 18mi
FSE-4 (Loop)    30m / 16mi   45m / 30mi   30m / 14mi   35m / 18mi    8m / 2mi    20m / 10mi    5m / 1mi
FSL (ORD)        -           25m / 16mi   30m / 18mi   45m / 28mi   35m / 20mi   25m / 14mi   35m / 20mi
Job-A (Hoff)    25m / 16mi    -           45m / 28mi   65m / 42mi   50m / 32mi   40m / 25mi   50m / 33mi
Job-B (Evan)    30m / 18mi   45m / 28mi    -           55m / 32mi   30m / 15mi   35m / 18mi   25m / 12mi
========================================================================================================
```

---

### 6.3 Executing the Triple Feasibility Filter

1. **Job-A (St. Alexius — MRI P1 Emergency):**
   * *Skill Required:* Level 2+ MRI.
   * *Qualified Technicians:* Dave (FSE-1, Level 3) and Marcus (FSE-3, Level 2). Elena and Sarah are disqualified.
   * *Parts Requirement:* Cryogenic Valve. **Not in any van stock; stocked at FSL (ORD).**
   * *Feasibility Comparison:*
     * Marcus (FSE-3): Drive Oak Lawn $\to$ FSL = 45 min. Pickup part = 15 min. Drive FSL $\to$ Job-A = 25 min. Total time = 85 min. Departure at 07:00 $\implies$ Arrival at 08:25 (Well within [08:00, 10:00] window).
     * Dave (FSE-1): Drive Naperville $\to$ FSL = 35 min. Pickup part = 15 min. Drive FSL $\to$ Job-A = 25 min. Total time = 75 min. Departure at 07:00 $\implies$ Arrival at 08:15 (Within window).
2. **Job-B (NorthShore — CT P1 Emergency):**
   * *Skill Required:* Level 2+ CT.
   * *Qualified Technicians:* Dave (FSE-1) and Elena (FSE-2). Marcus and Sarah are disqualified.
   * *Parts Requirement:* CT X-Ray Tube. **Present in Elena’s (FSE-2) Van Stock.**
   * *Assignment Logic:* Elena lives in Evanston, only 5 minutes (2 miles) from NorthShore Hospital!
     * Departure at 08:00 $\implies$ Arrival at 08:05. Direct hit, zero logistics delay. Elena is mandatory for Job-B.
3. **Job-C (Advocate Oak Lawn — MRI P2):**
   * *Skill Required:* Level 2+ MRI.
   * *Parts Requirement:* RF Board. **Present in Marcus’s (FSE-3) Van Stock.**
   * *Assignment Logic:* Marcus lives in Oak Lawn (8 minutes away). Assigning Job-C to Marcus guarantees zero travel delay and 100% FTFR.

---

### 6.4 The Global Daily Schedule Solution

Applying the ALNS optimization algorithm to minimize travel cost and balance workloads yields the following coordinated schedule:

```text
Optimized Daily Technician Schedules:
==================================================================================================================
Technician  Assigned Route Sequence                                 Timeline & Activities              Hours / Pay
------------------------------------------------------------------------------------------------------------------
FSE-1 (Dave)   Home (07:15) -> FSL ORD (Pick Cryo Valve, 07:50-08:05)   07:15-08:30: Drive & FSL Pick    Drive: 2.1h
               -> Job-A St. Alexius (08:30-11:00, 2.5h repair)          08:30-11:00: Job-A MRI Repair    Wrench: 5.5h
               -> Job-D Rush Univ (11:50-14:50, 3.0h CT PM)             11:50-14:50: Job-D CT PM OverhaulTotal: 7.6h
               -> Home Naperville (Arrive 15:40)                        14:50-15:40: Return Drive        Pay: $494.00
                                                                                                         Dist: 88 mi

FSE-2 (Elena)  Home (08:15) -> Job-B NorthShore (08:20-10:20, 2.0h rep) 08:20-10:20: Job-B CT Repair    Drive: 1.6h
               -> Job-E Loyola Maywood (11:00-12:30, 1.5h X-Ray)        11:00-12:30: Job-E X-Ray Repair  Wrench: 3.5h
               -> Home Evanston (Arrive 13:10)                          12:30-13:10: Return Drive        Total: 5.1h
                                                                                                         Pay: $255.00
                                                                                                         Dist: 40 mi

FSE-3 (Marcus) Home (10:45) -> Job-C Advocate (10:55-12:55, 2.0h rep)   10:55-12:55: Job-C MRI Repair    Drive: 1.8h
               -> Job-F Northwestern (13:30-15:30, 2.0h Ultrasnd PM)    13:30-15:30: Job-F Ultrasound PM Wrench: 4.0h
               -> Home Oak Lawn (Arrive 16:10)                          15:30-16:10: Return Drive        Total: 5.8h
                                                                                                         Pay: $290.00
                                                                                                         Dist: 39 mi

FSE-4 (Sarah)  Reserve Standby / Local Chicago Outstation Support       08:00-16:30: Shop Prep & Kitting Total: 8.0h
               (Available as immediate buffer for P1 emergency pop-up)                                   Pay: $320.00
==================================================================================================================
```

---

### 6.5 Financial and Operational Performance Metrics

```text
Performance Evaluation Dashboard:
========================================================================================================
Operational Metric                      Achieved Value          Target Standard       Status
--------------------------------------------------------------------------------------------------------
First-Time Fix Rate (FTFR)              100.0% (6 of 6 jobs)    >= 85.0%              Exceeded
Contractual SLA Penalty Breaches        $0 (0 minutes late)     $0                    Perfect Compliance
Total Fleet Drive Time                  5.5 hours               < 8.0 hours           Optimized
Total Miles Driven                      167 miles               < 250 miles           Minimized ($141.95)
Technician Overtime Hours               0.0 hours               < 2.0 hours           Zero Overtime Incurred
Technician Labor Payroll Cost           $1,359.00               < $1,600.00           Under Budget
Direct Transportation Freight Savings   $850.00                 -                     Eliminated 3rd party couriers
========================================================================================================
```

**Comparison with Naive Unsynchronized Dispatch:**
If dispatch had sent the nearest technician without checking van stock:
* Elena (FSE-2) would have been sent to Job-A (18 minutes away). Upon arrival at 08:00, she would have discovered she lacked MRI high-field certification.
* A second engineer (Dave) would have had to be dispatched, arriving at 12:30—**2.5 hours past the P1 deadline**, incurring a **$5,000 SLA cash penalty**.
* Sparing and scheduling synchronization eliminated $5,000 in liquidated damages on a single hospital call.

---

## 7. Code, query, or calculation

The concise Python implementation below models skills filtering, checks van stock parts feasibility, and executes an insertion routing solver.

```python
"""
Field Service Skills-Inventory-Routing Evaluator
Validates skills feasibility, parts availability, time windows, and computes insertion cost.
"""

from typing import List, Dict, Any

def is_assignment_feasible(tech: Dict[str, Any], job: Dict[str, Any]) -> Tuple[bool, str]:
    """Checks skills match and van stock parts availability."""
    # 1. Skills verification
    if job["required_skill"] not in tech["skills"]:
        return False, f"Missing Skill: requires {job['required_skill']}"
    
    # 2. Parts verification (In Van or requires FSL pickup)
    has_part = job["required_part"] in tech["van_stock"]
    if not has_part and not job["allow_fsl_pickup"]:
        return False, f"Missing Part: {job['required_part']} not in van"
        
    return True, "Feasible"

def evaluate_route_timeline(
    tech: Dict[str, Any], 
    jobs: List[Dict[str, Any]], 
    distance_matrix: Dict[str, Dict[str, float]]
) -> Dict[str, Any]:
    """Calculates total drive time, wrench time, overtime, and SLA tardiness penalties."""
    current_time = tech["shift_start"]
    current_loc = tech["home_loc"]
    total_drive_min = 0.0
    total_wrench_min = 0.0
    total_sla_penalties = 0.0
    
    for job in jobs:
        drive_time = distance_matrix[current_loc][job["site_id"]]
        arrival_time = current_time + (drive_time / 60.0)
        total_drive_min += drive_time
        
        # Check SLA Time Window [e, l]
        tardiness_hours = max(0.0, arrival_time - job["window_late"])
        total_sla_penalties += tardiness_hours * job["penalty_hourly"]
        
        # Service execution
        start_service = max(arrival_time, job["window_early"])
        departure_time = start_service + job["duration_hours"]
        total_wrench_min += job["duration_hours"] * 60.0
        
        current_time = departure_time
        current_loc = job["site_id"]
        
    # Return to home base
    return_drive = distance_matrix[current_loc][tech["home_loc"]]
    total_drive_min += return_drive
    finish_time = current_time + (return_drive / 60.0)
    
    total_hours = (total_drive_min + total_wrench_min) / 60.0
    overtime_hours = max(0.0, total_hours - 8.0)
    
    return {
        "Total_Hours": round(total_hours, 2),
        "Drive_Hours": round(total_drive_min / 60.0, 2),
        "Wrench_Hours": round(total_wrench_min / 60.0, 2),
        "Overtime_Hours": round(overtime_hours, 2),
        "SLA_Penalties": round(total_sla_penalties, 2),
        "Finish_Time": round(finish_time, 2)
    }

# --- Demonstration of Feasibility and Evaluation ---
if __name__ == "__main__":
    tech_elena = {
        "id": "FSE-2", "name": "Elena", "home_loc": "Evanston",
        "skills": ["Level 1: X-Ray", "Level 2: CT"],
        "van_stock": ["CT X-Ray Tube", "PM Kit"],
        "shift_start": 8.0
    }
    
    job_b = {
        "id": "Job-B", "site_id": "NorthShore", "required_skill": "Level 2: CT",
        "required_part": "CT X-Ray Tube", "allow_fsl_pickup": False,
        "window_early": 8.5, "window_late": 10.5, "duration_hours": 2.0, "penalty_hourly": 2000.0
    }
    
    feasible, reason = is_assignment_feasible(tech_elena, job_b)
    print("==================================================================")
    print("FIELD SERVICE DISPATCH FEASIBILITY TEST")
    print("==================================================================")
    print(f"Assigning {job_b['id']} to {tech_elena['name']}: {feasible} ({reason})")
    
    # Distance lookup mock (minutes)
    dist_map = {
        "Evanston": {"NorthShore": 5.0, "Evanston": 0.0},
        "NorthShore": {"Evanston": 5.0}
    }
    
    timeline = evaluate_route_timeline(tech_elena, [job_b], dist_map)
    print("\nROUTE TIMELINE EVALUATION:")
    for k, v in timeline.items():
        print(f"  {k:<16}: {v}")
    print("==================================================================")
```

### Expected Output
```text
==================================================================
FIELD SERVICE DISPATCH FEASIBILITY TEST
==================================================================
Assigning Job-B to Elena: True (Feasible)

ROUTE TIMELINE EVALUATION:
  Total_Hours     : 2.17
  Drive_Hours     : 0.17
  Wrench_Hours    : 2.0
  Overtime_Hours  : 0.0
  SLA_Penalties   : 0.0
  Finish_Time     : 10.58
==================================================================
```

---

## 8. Data, standards, and regulatory context

Field service scheduling operates under strict statutory labor safety laws and transport regulations that legally bound scheduling optimization algorithms.

### 8.1 Key Regulatory Standards

1. **EU Working Time Directive (Directive 2003/88/EC):**
   * *Mandatory Rest Periods:* Minimum of 11 consecutive hours of daily rest per 24-hour period.
   * *Working Limits:* Maximum average working week of 48 hours, including overtime. Mandatory break every 6 hours of work.
   * *Algorithmic Impact:* Dispatch algorithms that attempt to schedule early-morning emergency calls following a late-night shift breach statutory EU law, exposing the enterprise to labor criminal liability.

2. **US DOT Federal Motor Carrier Safety Administration (FMCSA) Hours of Service (49 CFR Part 395):**
   * *Scope:* Applies to field service engineers operating commercial service vehicles exceeding 10,000 lbs GVWR (e.g., heavy utility crane trucks, mobile oil-field test labs).
   * *Mandatory Boundaries:* Maximum 11 hours driving following 10 consecutive hours off-duty; mandatory 30-minute break after 8 hours of driving.

3. **OSHA 29 CFR 1910.331–335 / NFPA 70E (Electrical Safety in the Workplace):**
   * *Qualification Mandate:* Prohibits uncertified personnel from working on energized electrical systems exceeding 50 volts.
   * *Systemic Requirement:* Scheduling engines must track arc-flash safety certifications as an absolute prerequisite constraint prior to dispatching technicians to industrial breaker panels or medical linear accelerators.

4. **ISO 55001:2014 (Asset Management — Management Systems):**
   * *Standard:* Requires that technical maintenance actions be auditable, verifying that personnel executing service possess certified competencies conforming to manufacturer technical specifications.

---

## 9. Industry practice and case evidence

### 9.1 Elevator & Escalator Maintenance: Otis ONE IoT & Dynamic Dispatch

Otis Elevator maintains over 2.2 million elevators worldwide under long-term service contracts.
* **The Traditional Pathology:** Historically, elevators broke down, customers phoned the call center, and dispatchers paged the nearest local mechanic. Mechanics spent 30% of their time driving back to regional depots to pick up parts they didn't have in their vans, resulting in an FTFR of only 68%.
* **The Connected IoT Revolution (Otis ONE):** Modern Otis elevators stream real-time vibration, door-cycle telemetry, and motor temperature over cellular IoT to the Otis Cloud.
* **Predictive Dispatch Coupling:** Machine learning models predict door lock mechanism wear weeks before failure. The dispatch engine automatically pushes a work order to the technician's mobile app, bundles the replacement door contact roller into the technician’s nightly trunk delivery, and routes the technician to the building during scheduled off-peak office hours. FTFR increased to **over 88%**, while emergency passenger entrapments fell by 40%.

### 9.2 Semiconductor Capital Equipment: ASML Global Field Operations

ASML field service engineers maintain extreme ultraviolet (EUV) lithography scanners in 24/7 semiconductor fabrication plants (TSMC, Samsung, Intel).
* **The Consequence of Failure:** Downtime costs over $75,000 per hour.
* **Operational Discipline:** ASML deploys a "Pair-and-Spare" dispatch architecture:
  * Master Level 3 engineers are paired with junior engineers for skill transfer.
  * Parts are not stocked in vans; they are stored in climate-controlled Cleanroom Forward Stocking Locations directly inside the customer's fab perimeter.
  * The scheduling system tracks technician fatigue levels down to the minute: engineers are legally barred from logging into fab tool control benches after 10 consecutive hours of duty, triggering automated handover protocols to incoming shift engineers to eliminate technician fatigue-induced wafer contamination.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign / Red Flag | Underlying Operational/Systemic Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **The Blind Geo-Proximity Dispatch** | Nearest technician dispatched to an emergency job arrives on site and discovers they lack required skills. | Dispatch system optimizes purely for minimum driving distance without checking the skills matrix. | 0% First-Time Fix; duplicate travel cost; severe SLA breach penalty damages. | Implement hard skills matrix filtering as an unbypassable gate prior to route distance calculation. |
| 2 | **The Phantom Van Inventory Trap** | Dispatcher assumes part is in van stock, but technician finds an empty bin upon arrival. | Disconnect between physical van stock and ERP inventory records (unrecorded usage, theft, damaged parts). | Failed service call; technician forced to drive to depot; customer furious. | Mandate mobile barcode scanning for every part consumption; perform weekly cycle counts on van stock. |
| 3 | **Clandestine Van Hoarding (Squirrel Stocking)** | Technicians hoard high-value critical boards inside their vans "just in case" without logging them. | Technicians fear being caught without parts and bypass central inventory return rules. | Millions in unrecorded floating inventory; central planning orders unneeded replacements. | Conduct surprise physical audits ("boot audits"); penalize unrecorded trunk inventory. |
| 4 | **Overtime Burnout Spiral** | Technician sick leave surges; master engineers quit due to exhaustion. | Scheduling engine systematically overloads high-skill engineers while junior technicians sit idle. | High employee turnover; severe regional service capability vacuum. | Implement workload-balancing equity constraints across technician skill tiers. |
| 5 | **Static Planning in a Dynamic World** | Entire day's schedule collapses by 10:00 AM due to morning traffic and delayed jobs. | Dispatch engine generates rigid static routes that cannot adapt dynamically to real-time execution variance. | Cascading late arrivals; end-of-day jobs cancelled; massive customer dissatisfaction. | Deploy dynamic real-time rolling-horizon re-optimization algorithms. |
| 6 | **Statutory HOS Violations** | Transport authority audits find technicians logging 13 hours of driving per day. | Automated schedulers optimize drive times without enforcing legal daily driving and rest limits. | Massive regulatory fines; company impoundment of service vehicles; catastrophic corporate liability. | Hard-code statutory DOT/EU driving limits into mathematical MILP constraints. |
| 7 | **The "Shotgun Troubleshooting" Drain** | Technicians replace three expensive boards sequentially until the machine boots up. | Inadequate diagnostic training leads technicians to use parts replacement as their diagnostic tool. | Good parts destroyed or returned as NFF; soaring parts consumption rates. | Require automated diagnostic fault-tree sign-off on mobile apps prior to part authorization. |
| 8 | **Uncoordinated Parts Rendezvous Failure** | Technician arrives at site at 09:00, but the courier delivering the emergency part arrives at 14:00. | Dispatch failed to synchronize technician travel routing with third-party courier delivery tracking. | Technician wastes 5 hours idle on site waiting for courier; customer billed for dead time. | Implement synchronized joint-arrival scheduling linking courier telematics directly to FSM dispatch. |
| 9 | **Customer Access Denial Stoppage** | Technician arrives on time but is refused entry at hospital security for 2 hours. | Dispatch failed to verify customer site access badges, contractor safety orientations, or immunization records. | Service window missed; technician idle; job aborted. | Maintain customer site credential profiles in the FSM system and verify technician clearance prior to dispatch. |
| 10 | **Unclosed Core Reverse Logistics Leak** | Technicians accumulate 15 unserviceable cores in their vehicle trunk over three months. | Absence of an automated reverse logistics tracking loop for demounted carcasses. | Central MRO starved of cores; core deposits forfeited; pool shrinks. | Mandate that technicians cannot receive new van stock until unserviceable cores are surrendered to couriers. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks If It Fails | How to Test / Validate |
|---|---|---|---|
| **Deterministic Travel Times ($t_{ij}$)** | Needed to solve static MD-VRPTW schedules before the operating day begins. | Traffic jams, severe weather, or road closures cause technicians to miss time windows. | Integrate real-time predictive traffic engines (e.g., Google Maps API, HERE) into dynamic dispatch models. |
| **Accurate Presumptive Parts Profiling** | Assumes the remote error code accurately predicts which physical part is broken. | If diagnostic accuracy is poor, technician arrives with the wrong part, failing the call. | Monitor historical correlation between initial logged trouble code and final consumed part SKU. |
| **Stationary Service Durations ($d_i$)** | Schedulers allocate fixed time slots (e.g., 2.0 hours) for specific job types. | Seized bolts, wiring corrosion, or unexpected secondary failures extend job to 6 hours, wrecking downstream routes. | Model service durations as lognormal random variables and introduce schedule buffer slack. |
| **Complete Van Stock Visibility** | Assumes ERP inventory balances match the physical parts inside the vehicle. | Parts stolen, dropped, or unrecorded during night shifts cause unexpected stockouts on site. | Reconcile daily mobile app consumption logs against physical inventory scans. |
| **Independent Job Service Requirements** | Assumes each service call can be completed by a single individual technician. | Heavy assemblies (e.g., MRI magnet gradients) require two-person lifting crews (tandem routing). | Flag tandem-service jobs in CMMS; enforce synchronized multi-technician arrival constraints. |
| **Full Cellular Connectivity** | Needed for real-time dynamic re-optimization and mobile work order updates. | Underground hospital basements or remote rural substations have zero cellular signal. | Ensure mobile FSM software supports full offline operation with automated data sync upon reconnection. |
| **Customer Site Availability** | Assumes customer equipment is powered down and accessible upon technician arrival. | Production managers refuse to release the machine to the technician because they are running a rush order. | Require automated customer pre-arrival SMS confirmation 2 hours prior to dispatch. |
| **Compliance with Skills Matrix** | Assumes documented certifications reflect true physical troubleshooting capability. | A certified engineer who hasn't worked on a machine in 5 years struggles, vastly exceeding standard wrench time. | Track competency recency (date of last service execution on specific model) alongside formal certification. |

---

## 12. Exercises

### Exercise 1 — Calculating First-Time Fix Rate and Economic Impact
**Type:** Calculation & Performance Analysis  
**Time:** 45 minutes  
**Context:** An industrial compressed-air service provider completed 1,200 field service work orders across a quarterly period. A forensic review of CMMS work orders reveals:
* 816 jobs were successfully diagnosed, repaired, and signed off during the initial visit.
* 192 jobs required a second visit because the technician did not have the required spare part in van stock.
* 120 jobs required a second visit because the initial technician lacked the specialized electrical variable-speed-drive (VSD) certification and had to be replaced by a Level 3 specialist.
* 48 jobs required a second visit due to incorrect initial fault diagnosis by the call center (wrong parts ordered).
* 24 jobs required a second visit because the customer factory was locked or safety orientation was unavailable upon arrival.
* Operating costs: Each dispatch (truck roll) costs an average of $\$280$ in vehicle travel, fuel, and technician driving time. Each hour of uncompleted downtime on contract assets incurs an average customer SLA penalty of $\$150/\text{hour}$ (average re-visit delay = 24 hours).

**Deliverables:**
1. Compute the baseline First-Time Fix Rate (FTFR) for the quarter.
2. Construct a Pareto table categorizing the root causes of first-time fix failure.
3. Calculate the total financial waste (duplicate truck rolls plus SLA penalties) generated by failed first visits over the quarter.
4. If an automated mobile van-stock kitting and skills-filtering initiative eliminates 75% of the parts-related failures and 80% of the skills-related failures, calculate the new FTFR and the net quarterly financial savings.

---

### Exercise 2 — Skills Matrix and Time Window Feasibility Screening
**Type:** Algorithmic Screening & Mathematical Logic  
**Time:** 50 minutes  
**Context:** A medical equipment dispatch desk must assign 3 incoming emergency service tickets to 3 available Field Service Engineers.

```text
Technician Profiles:
========================================================================================================
Tech ID  Location       Current Status    Skills Portfolio        Van Stock Parts Available
--------------------------------------------------------------------------------------------------------
T-1      Plano          Free at 09:00     MRI-Level 2, CT-Level 2 RF Coil, Pre-Amp Board, Gasket Kit
T-2      Irving         Free at 09:30     CT-Level 3, X-Ray-L2    CT Detector, X-Ray Tube, PM Kit
T-3      Fort Worth     Free at 08:45     MRI-Level 3, CT-Level 1 Cryo Valve, Magnet Compressor, RF Coil
========================================================================================================
```

```text
Incoming Work Orders:
========================================================================================================
Job ID   Hospital Site    System   Window [e, l]   Required Skill   Required Part
--------------------------------------------------------------------------------------------------------
Job-1    Dallas Medical   MRI      [09:30, 11:00]  MRI-Level 2+     RF Coil
Job-2    Arlington Mem.   CT       [10:00, 11:30]  CT-Level 2+      CT Detector
Job-3    Denton Regional  MRI      [10:30, 12:30]  MRI-Level 3      Cryo Valve
========================================================================================================
```

```text
Driving Time Matrix (Minutes):
========================================================================================================
From / To        Dallas Medical (Job-1)   Arlington Mem. (Job-2)   Denton Regional (Job-3)
--------------------------------------------------------------------------------------------------------
T-1 (Plano)      30 min                   55 min                   45 min
T-2 (Irving)     20 min                   25 min                   40 min
T-3 (Fort Worth) 45 min                   25 min                   60 min
========================================================================================================
```

**Deliverables:**
1. Construct the $3 \times 3$ Binary Feasibility Matrix evaluating Skills, Parts, and Time Window constraints for all technician-job pairs.
2. Identify the unique feasible matching assignment that satisfies 100% of the constraints without breaching any SLA windows.
3. Calculate the total fleet driving time (in minutes) for this feasible assignment.

---

### Exercise 3 — MILP Formulation of Van Stock Optimization
**Type:** Mathematical Formulation & Optimization  
**Time:** 60 minutes  
**Context:** A service manager is configuring the mobile van stock for a senior wind turbine technician. The technician's van has a strict maximum payload weight capacity of $W_{\max} = 150 \text{ kg}$ and a total inventory working capital budget limit of $B = \$15,000$.
* The technician services a regional cluster of 40 identical turbines.
* Historical telemetry indicates 8 candidate component SKUs frequently cause emergency turbine stoppages.

```text
Candidate Spare Parts Portfolio:
========================================================================================================
SKU ID  Component Description            Weight (w_p)  Unit Cost (c_p)  Monthly Failure Prob P(p)  FTFR Gain
--------------------------------------------------------------------------------------------------------
SKU-1   Pitch Motor Control Drive Relay  4.0 kg        $450             0.45                       0.45
SKU-2   Hydraulic Brake Solenoid Valve   12.0 kg       $1,200           0.30                       0.30
SKU-3   Anemometer Wind Speed Sensor     2.5 kg        $600             0.25                       0.25
SKU-4   Yaw Gearbox Planetary Pinion     65.0 kg       $4,800           0.08                       0.08
SKU-5   Main Generator Carbon Brush Kit  6.0 kg        $250             0.55                       0.55
SKU-6   Vibration Accelerometer Probe    1.5 kg        $850             0.20                       0.20
SKU-7   High-Pressure Hydraulic Filter   3.0 kg        $120             0.60                       0.60
SKU-8   Electronic Yaw Control Inverter  45.0 kg       $6,500           0.15                       0.15
========================================================================================================
```

**Deliverables:**
1. Formulate the binary 0-1 Knapsack Optimization problem to maximize expected First-Time Fix contributions subject to weight and budget constraints.
2. Compute the "Bang-for-Buck" heuristic index for each SKU:
   $$\text{Index}_p = \frac{\text{Monthly Failure Prob } P(p)}{w_p \cdot c_p}$$
3. Determine the optimal set of spare parts to load into the van stock.
4. Calculate the total weight, total capital cost, and the overall expected probability that an emergency call can be resolved directly from van stock.

---

### Exercise 4 — Dynamic Dispatch and Emergency P1 Interruption
**Type:** Scenario Modeling & Real-Time Decision Analysis  
**Time:** 45 minutes  
**Context:** It is 11:30 AM. Technician Bob is currently servicing a scheduled, non-critical preventive maintenance job (Job-PM) on an industrial air handler at Site A (scheduled finish time: 13:00).
* Bob’s afternoon schedule consists of Job-B (scheduled PM at Site B, window [13:30, 16:30], duration 2.0 hours, distance from Site A = 30 minutes).
* At 11:35 AM, an emergency Priority 1 breakdown (Job-P1) occurs at Site C (a pharmaceutical sterile packaging plant).
  * Equipment: Critical cleanroom chiller failure.
  * Contractual SLA deadline: Technician must arrive by 13:00 ($l_{\text{P1}} = 13:00$).
  * SLA Breach Penalty: $\$3,000/\text{hour}$ of late arrival.
  * Service duration: 2.5 hours.
  * Driving times: Site A $\to$ Site C = 40 minutes. Site C $\to$ Site B = 35 minutes.
* Bob is the only certified technician within a 50-mile radius who has the required chiller replacement control board in his van stock.

**Deliverables:**
1. Evaluate Option 1: Bob finishes Job-PM at 13:00, then drives directly to Site C. Calculate arrival time at Site C and the resulting SLA cash penalty.
2. Evaluate Option 2: Dispatch orders Bob to abort/pause Job-PM immediately at 11:45 (requiring 15 minutes to safely pack up tools), drive directly to Site C, complete Job-P1, and then proceed to Site B. Evaluate arrival times, SLA compliance, and whether Job-B can still be completed before 17:00 without exceeding daily statutory working limits (8-hour shift ending at 17:00).
3. Provide the dispatch command decision and write the communication script to both customers.

---

### Exercise 5 — Statutory Hours of Service (HOS) Compliance Audit
**Type:** Legal & Operational Audit  
**Time:** 40 minutes  
**Context:** A utility services contractor is being audited by labor authorities following a highway collision involving a field service crane truck. The dispatch record shows the following schedule executed by Technician Frank:
* 06:00: Left home depot. Drove 2.5 hours to Job 1.
* 08:30–12:30: Performed active high-voltage transformer repair (4.0 hours wrench time).
* 12:30–14:00: Drove 1.5 hours to Job 2.
* 14:00–18:00: Performed emergency switchgear repair (4.0 hours wrench time).
* 18:00–20:00: Drove 2.0 hours back to home depot.
* Total day: 14.0 elapsed hours (6.0 hours driving, 8.0 hours wrench time).
* The governing labor regulation states: "No driver may drive more than 11 hours or remain on duty for more than 14 consecutive hours. A mandatory 30-minute non-working rest break must be logged prior to accumulating 8 consecutive hours of on-duty time."

**Deliverables:**
1. Audit Frank’s timeline against the statutory on-duty and rest break regulations. Did a violation occur? If so, at what exact timestamp?
2. Explain the difference between "driving time" and "on-duty non-driving wrench time" in transport safety law.
3. Redesign the scheduling algorithm's mathematical constraint set to guarantee that automated dispatch engines automatically insert mandatory 30-minute rest buffers into technician routes.

---

## 13. Self-check answers

### 13.1 Answer to Exercise 1

#### 1. Baseline First-Time Fix Rate (FTFR):
$$\text{FTFR} = \frac{N_{\text{single-visit fixes}}}{N_{\text{total dispatches}}} = \frac{816}{1,200} = \mathbf{0.680} \text{ (68.0\%)}$$

#### 2. Pareto Root-Cause Breakdown of Failures:
Total failed first visits = $1,200 - 816 = 384$ failed visits.

```text
Root Cause Category           Failed Calls   Percentage of Failures   Cumulative %
----------------------------------------------------------------------------------
Missing Parts in Van Stock    192 calls      50.0%                    50.0%
Missing Skills / License      120 calls      31.25%                   81.25%
Call Center Diagnostic Error   48 calls      12.5%                    93.75%
Customer Site Access Locked    24 calls       6.25%                  100.0%
----------------------------------------------------------------------------------
Total Failed Visits           384 calls     100.0%
```

#### 3. Total Quarterly Financial Waste:
* **Duplicate Truck Rolls:** 384 return visits $\times$ $\$280/\text{trip} = \mathbf{\$107,520}$.
* **Contractual SLA Downtime Penalties:**
  $$\text{Downtime Penalties} = 384 \text{ failures} \times 24 \text{ hrs delay} \times \$150/\text{hr} = \mathbf{\$1,382,400}$$
* **Total Quarterly Financial Waste:**
  $$\text{Total Waste} = \$107,520 + \$1,382,400 = \mathbf{\$1,489,920/\text{quarter}} \quad (\approx \$5.96 \text{M / year})$$

#### 4. Impact of Van-Stock & Skills Initiative:
* Parts failures eliminated: $192 \times 0.75 = 144$ failures saved.
* Skills failures eliminated: $120 \times 0.80 = 96$ failures saved.
* Total failures avoided = $144 + 96 = 240$ failures converted to first-time fixes.
* New single-visit fixes = $816 + 240 = \mathbf{1,056 \text{ jobs}}$.
* **New FTFR:**
  $$\text{FTFR}_{\text{new}} = \frac{1,056}{1,200} = \mathbf{0.880} \text{ (88.0\%) - Smashed the 85% target!}$$
* **Quarterly Cash Savings:**
  $$\text{Savings} = 240 \text{ avoided return visits} \times [\$280 + (24 \times \$150)] = 240 \times [\$280 + \$3,600] = 240 \times \$3,880 = \mathbf{\$931,200/\text{quarter}}$$

---

### 13.2 Answer to Exercise 2

#### 1. Evaluation of $3 \times 3$ Feasibility:

* **Technician T-1 (Plano):**
  * *For Job-1 (Dallas Med, MRI-L2, RF Coil):*
    * Skill: MRI-L2 (Match). Part: RF Coil in Van (Match).
    * Time: Depart 09:00 + 30 min drive = Arrive 09:30. Window is [09:30, 11:00]. **FEASIBLE.**
  * *For Job-2 (Arlington, CT-L2, CT Detector):*
    * Part: CT Detector **NOT in van stock**. **INFEASIBLE.**
  * *For Job-3 (Denton, MRI-L3, Cryo Valve):*
    * Skill: T-1 is only MRI-L2; Job-3 requires MRI-L3. Part: Cryo Valve **NOT in van**. **INFEASIBLE.**

* **Technician T-2 (Irving):**
  * *For Job-1 (Dallas Med, MRI-L2, RF Coil):*
    * Skill: T-2 has zero MRI skills. Part: No RF coil. **INFEASIBLE.**
  * *For Job-2 (Arlington, CT-L2, CT Detector):*
    * Skill: CT-L3 (Exceeds L2). Part: CT Detector in Van (Match).
    * Time: Depart 09:30 + 25 min drive = Arrive 09:55. Window is [10:00, 11:30]. **FEASIBLE** (Arrives 5 min early, waits until 10:00).
  * *For Job-3 (Denton, MRI-L3, Cryo Valve):*
    * Skill: Zero MRI skills. **INFEASIBLE.**

* **Technician T-3 (Fort Worth):**
  * *For Job-1 (Dallas Med, MRI-L2, RF Coil):*
    * Skill: MRI-L3. Part: RF Coil in Van. Feasible on skill/part, but leaves Job-3 unassigned!
  * *For Job-2 (Arlington, CT-L2, CT Detector):*
    * Skill: T-3 is only CT-L1; Job requires CT-L2. **INFEASIBLE.**
  * *For Job-3 (Denton, MRI-L3, Cryo Valve):*
    * Skill: MRI-L3 (Match). Part: Cryo Valve in Van (Match).
    * Time: Depart 08:45 + 60 min drive = Arrive 09:45. Window is [10:30, 12:30]. **FEASIBLE** (Arrives early, ready for service).

#### 2. Unique Feasible Assignment:
$$\mathbf{T-1 \to Job-1 \quad | \quad T-2 \to Job-2 \quad | \quad T-3 \to Job-3}$$
Any other permutation breaches either skills, van stock, or time windows.

#### 3. Total Fleet Drive Time:
$$\text{Total Drive Time} = 30 \text{ min (T-1)} + 25 \text{ min (T-2)} + 60 \text{ min (T-3)} = \mathbf{115 \text{ minutes}} \quad (\approx 1.92 \text{ hours})$$

---

### 13.3 Answer to Exercise 3

#### 1. Binary Knapsack Formulation:
$$\max \sum_{p=1}^8 P(p) \cdot x_p \quad \text{subject to} \quad \sum_{p=1}^8 w_p x_p \le 150 \quad \text{and} \quad \sum_{p=1}^8 c_p x_p \le 15,000, \quad x_p \in \{0, 1\}$$

#### 2. Bang-for-Buck Index Calculation:

```text
SKU     Weight (w)  Cost (c)   w * c       Prob P(p)   Index = P / (w * c)   Rank
----------------------------------------------------------------------------------
SKU-7   3.0 kg      $120       360         0.60        1.667 x 10^-3         1
SKU-5   6.0 kg      $250       1,500       0.55        0.367 x 10^-3         2
SKU-1   4.0 kg      $450       1,800       0.45        0.250 x 10^-3         3
SKU-3   2.5 kg      $600       1,500       0.25        0.167 x 10^-3         4
SKU-6   1.5 kg      $850       1,275       0.20        0.157 x 10^-3         5
SKU-2   12.0 kg     $1,200     14,400      0.30        0.021 x 10^-3         6
SKU-8   45.0 kg     $6,500     292,500     0.15        0.0005 x 10^-3        7
SKU-4   65.0 kg     $4,800     312,000     0.08        0.00026 x 10^-3       8
```

#### 3. Optimal Selection (Greedy Insertion by Rank):
* Select SKU-7: Cumul W = 3.0 kg, Cumul Cost = $120.
* Select SKU-5: Cumul W = 9.0 kg, Cumul Cost = $370.
* Select SKU-1: Cumul W = 13.0 kg, Cumul Cost = $820.
* Select SKU-3: Cumul W = 15.5 kg, Cumul Cost = $1,420.
* Select SKU-6: Cumul W = 17.0 kg, Cumul Cost = $2,270.
* Select SKU-2: Cumul W = 29.0 kg, Cumul Cost = $3,470.
* Select SKU-8: Cumul W = 74.0 kg, Cumul Cost = $9,970.
* Next is SKU-4 (Weight = 65 kg, Cost = $4,800):
  * Total Weight = $74.0 + 65.0 = 139.0 \text{ kg} \le 150 \text{ kg}$ (Feasible!).
  * Total Cost = $\$9,970 + \$4,800 = \mathbf{\$14,770} \le \$15,000$ (Feasible!).
  * Select SKU-4!

#### 4. Final Van Stock Performance:
* **Selected SKUs:** **All 8 SKUs can be loaded!**
* Total Payload Weight: $139.0 \text{ kg}$ (under $150 \text{ kg}$ limit).
* Total Capital Invested: $\$14,770$ (under $\$15,000$ budget).
* **Expected First-Time Fix Coverage:** The probability of having the part on any random emergency call is over **95%**!

---

### 13.4 Answer to Exercise 4

#### 1. Option 1 Evaluation (Complete PM at Site A first):
* Finish Site A at 13:00.
* Drive Site A $\to$ Site C = 40 minutes.
* Arrival at Site C = 13:40.
* SLA Deadline = 13:00.
* **Tardiness:** $13:40 - 13:00 = 40 \text{ minutes} = 0.667 \text{ hours}$.
* **Contractual SLA Penalty:**
  $$\text{Penalty} = 0.667 \text{ hours} \times \$3,000/\text{hr} = \mathbf{\$2,000 \text{ penalty cash loss}}.$$

#### 2. Option 2 Evaluation (Abort PM, Intercept Emergency P1):
* Abort Site A at 11:45. Drive 40 min to Site C.
* Arrival at Site C = 12:25.
* **SLA Compliance:** Arrives at 12:25, which is **35 minutes ahead of the 13:00 deadline**!
* **SLA Penalty:** $\mathbf{\$0}$.
* Complete Job-P1: 12:25 + 2.5 hours wrench time = Finish at 14:55.
* Proceed to Site B: Drive Site C $\to$ Site B = 35 minutes $\implies$ Arrival at Site B at 15:30.
* Customer B Window: [13:30, 16:30]. Arrival at 15:30 is within the legal window!
* However, Job-B requires 2.0 hours wrench time ($15:30 + 2.0 \text{h} = 17:30$).
* Bob's working hours: Started at 08:30 $\implies 17:30$ finish = 9.0 hours total on-duty (within the legal 10-hour limit, incurs 1.0 hour overtime at $\$75/\text{hr}$).

#### 3. Command Decision:
**Execute Option 2 immediately.**
* Financial trade-off: Paying $\$75$ in technician overtime avoids a **$2,000 SLA penalty** on the cleanroom chiller.
* Communication script:
  * *To Customer A:* "Due to an emergency life-safety/cleanroom failure, your non-critical air handler PM was temporarily paused. Technician will return tomorrow at 08:00 to complete the remaining 1.25 hours."
  * *To Customer C:* "Emergency technician Bob has been diverted and is en route with the replacement board; arrival confirmed for 12:25."

---

### 13.5 Answer to Exercise 5

1. **Audit Finding & Timestamp:** **A severe statutory violation occurred at 14:00.**
   * Frank commenced duty at 06:00.
   * By 14:00, Frank accumulated 8.0 consecutive hours of on-duty time (2.5h drive + 4.0h wrench + 1.5h drive = 8.0 hours) **without logging a 30-minute non-working rest break**.
   * Continuing to work on Job 2 at 14:00 without a rest break constitutes an explicit federal/statutory Hours-of-Service violation.

2. **Legal Distinction:** Under FMCSA and labor transport law, "On-Duty Time" includes **both driving time and active wrench repair time**. Transport law regulates *driver fatigue*. Performing 4 hours of heavy physical transformer repair in high heat induces physical and mental exhaustion identical to driving. Therefore, wrench time does *not* reset the driving clock; it accelerates fatigue.

3. **MILP Constraint Redesign:**
   Insert a mandatory rest-break dummy node $r_k$ with duration $d_{r} = 0.50 \text{ hours}$ (30 minutes) into every technician route, constrained such that:
   $$T_{r, k} \le T_{\text{start}, k} + 8.0 \text{ hours}$$
   The algorithm is forced to route the technician to a rest stop or pause work before the 8-hour on-duty threshold is breached.

---

## 14. Interview questions

### Question 1: "Why does maximizing technician route density or minimizing driving mileage often lead to lower First-Time Fix Rates (FTFR)?"

**Strong Answer:**  
"Standard vehicle routing algorithms (VRP) minimize total travel distance ($\min \sum c_{ij} x_{ij}$). When applied naively to field service, the algorithm always assigns the geographically closest technician to an incoming job.

However, field service is constrained by skills and parts feasibility. If the nearest technician is only 5 miles away but lacks the specific certification for that machine model or does not have the required replacement board in their van stock, dispatching them achieves a zero percent first-time fix. The technician arrives, discovers they cannot complete the repair, and leaves. 

A return visit must be scheduled, requiring a second technician to drive 30 miles the following day. By minimizing travel distance in the short term, the algorithm induced duplicate travel in the medium term, destroyed FTFR, and inflicted severe customer downtime. True field service optimization must minimize **Total System Cost**—including SLA penalties and the cost of return visits—forcing skills and inventory feasibility as hard constraints before evaluating route distances."

**Weak Answer:**  
"Because if technicians drive less, they might not have the right tools in their car, so they have to go back to the warehouse."

---

### Question 2: "What is the difference between a static routing schedule and dynamic real-time dispatch, and how do you handle an emergency P1 ticket?"

**Strong Answer:**  
"A **static routing schedule** is generated overnight (typically using ALNS or genetic heuristics) before the operating day starts. It builds an optimized sequence of scheduled preventive maintenance calls, minor diagnostics, and known backlog repairs based on expected travel times.

**Dynamic real-time dispatch** operates during the working day to absorb stochastic disruptions: traffic gridlock, jobs running past standard wrench time, and, most critically, incoming **Priority 1 emergency breakdowns**.

When a P1 breakdown occurs:
1. We cannot abort a technician who is mid-way through a hazardous or delicate teardown.
2. The dynamic engine queries the fleet's real-time GPS coordinates and van inventories.
3. It filters for technicians possessing the mandatory skills and parts.
4. It evaluates candidate insertions into the afternoon schedules of qualified technicians, calculating the **Marginal System Disruption Cost**:
   $$\Delta \text{Cost} = \Delta \text{Drive Cost} + \Delta \text{Overtime} + \sum \Delta \text{Downstream SLA Penalties}$$
5. It assigns the P1 ticket to the technician with the minimum disruption cost, dynamically bumping non-critical PM jobs to the following day's backlog."

**Weak Answer:**  
"Static routing is what you do the night before. Dynamic dispatch is when an emergency happens, and you call whichever technician is closest on their cell phone."

---

### Question 3: "How do you mathematically optimize mobile van stock (trunk stock) under physical vehicle payload and capital budget constraints?"

**Strong Answer:**  
"We model van stocking as a **Multi-Constraint Binary Knapsack Problem**. 
Let each candidate spare part SKU $p$ have unit purchase price $c_p$, physical weight $w_p$, and physical volume $v_p$. Let $\lambda_p$ denote the historical probability that component $p$ is required on an emergency field call in the technician’s assigned territory.

The objective is to maximize the technician's autonomous First-Time Fix probability:
$$\max \sum_{p \in \text{Catalog}} \lambda_p \cdot P(\text{Fix} \mid p) \cdot x_p$$
subject to:
$$\sum_{p} w_p x_p \le W_{\max} \quad (\text{Vehicle Payload Limit})$$
$$\sum_{p} v_p x_p \le V_{\max} \quad (\text{Vehicle Cubic Shelving Volume})$$
$$\sum_{p} c_p x_p \le B_{\text{van}} \quad (\text{Working Capital Budget})$$

We solve this using discrete dynamic programming or greedy marginal analysis ranking parts by their **bang-for-buck ratio**:
$$\text{Index}_p = \frac{\lambda_p \cdot P(\text{Fix} \mid p)}{c_p \cdot r_h + \alpha w_p + \beta v_p}$$
This ensures high-failure, lightweight electronic boards and seals populate the van, while heavy, rarely failing structural castings remain at central distribution centers."

**Weak Answer:**  
"You look at what parts the technician used over the last three months and tell them to keep two of each in the back of the van."

---

### Question 4: "What is a 'Nightly Boot Sweep' in field service logistics, and why does it deliver immense productivity gains?"

**Strong Answer:**  
"A Nightly Boot Sweep (or Trunk Delivery Service) is a specialized reverse-and-forward logistics process operated by dedicated 3PL couriers between 00:00 and 05:00:
1. While the field service technician is off-duty, the courier accesses the technician’s locked service van parked in their residential driveway using a secure electronic rolling-code lockbox.
2. The courier deposits the replenishment spare parts consumed during the previous day’s work orders, directly restocking the van bins.
3. Simultaneously, the courier removes the unserviceable cores (carcasses) that the technician demounted and placed into the return bin, transporting them to the central MRO repair facility by 06:00.

This delivers immense productivity gains:
* **Eliminates Deadhead Driving:** Technicians do not waste 60–90 minutes every morning driving to a central warehouse to pick up parts.
* **Accelerates Reverse Logistics:** Unserviceable carcasses enter the repair loop within 24 hours of removal rather than sitting in a van for weeks.
* **Increases Wrench Time:** Converting 90 minutes of morning warehouse driving into billable field wrench time increases technician productive capacity by 15% to 20% across the entire enterprise."

**Weak Answer:**  
"It's when a delivery driver puts parts into the technician's van at night so the technician doesn't have to go to the warehouse in the morning."

---

### Question 5: "Under transport safety regulations (like US DOT HOS or EU Working Time), how does 'wrench time' interact with 'driving time'?"

**Strong Answer:**  
"In transport safety jurisprudence, regulatory limits govern **total driver fatigue**, not merely the physical act of steering a vehicle.

Under both US DOT (49 CFR Part 395) and EU Directive 2003/88/EC:
* A driver's duty day is categorized into: (1) Off-Duty, (2) Sleeper Berth, (3) Driving, and (4) **On-Duty Non-Driving**.
* Technician 'wrench time' spent diagnosing machinery, climbing wind turbine towers, or pulling heavy cables is legally classified as **On-Duty Non-Driving Time**.

Critically, on-duty wrench time **does not pause or reset the fatigue clock**; it counts directly toward the daily statutory maximum on-duty window (e.g., the 14-hour consecutive rule under US DOT). 
Furthermore, regulations mandate that a driver cannot drive after accumulating 8 consecutive hours of on-duty time without logging a 30-minute non-working rest break. An automated scheduling algorithm that fails to count wrench time toward the driver's duty limits will generate illegal schedules, exposing the company to massive tort liability and commercial vehicle impoundment."

**Weak Answer:**  
"Driving time is when you're behind the wheel. Wrench time doesn't count as driving, so you can work as many hours as you want as long as you don't drive more than 11 hours."

---

### Question 6: "Explain the difference between Mean Time to Respond (MTTR_resp) and Mean Time to Repair (MTTR_repair), and why conflating them distorts SLA pricing."

**Strong Answer:**  
"In service level agreements:
* **Mean Time to Respond ($\text{MTTR}_{\text{resp}}$):** The elapsed calendar time from customer ticket creation to the physical arrival of the certified technician at the asset site. It is governed primarily by **geography, dispatch velocity, traffic, and technician availability**.
* **Mean Time to Repair ($\text{MTTR}_{\text{repair}}$):** The elapsed time from the technician’s arrival to the successful functional restoration and sign-off of the machine. It is governed by **equipment maintainability, diagnostic accuracy, parts availability, and technician wrench skill**.

Conflating them distorts SLA pricing because they require completely different operational investments:
* Improving $\text{MTTR}_{\text{resp}}$ requires opening more Forward Stocking Locations (FSLs), hiring more geographically dispersed technicians, and paying for emergency hot-shot couriers.
* Improving $\text{MTTR}_{\text{repair}}$ requires engineering investments: pre-kitting standard overhaul assemblies, training technicians, and deploying advanced automated diagnostic software.
If an organization prices a contract assuming a '4-hour MTTR' without specifying whether it means response or resolution, it risks paying massive downtime penalties when a 2-hour response call requires 6 hours of complex mechanical overhaul."

**Weak Answer:**  
"Response time is how long it takes to get there. Repair time is how long it takes to fix it. If you mix them up, you will charge the customer the wrong price."

---

### Question 7: "What is a 'Tandem Service Call', and why does it represent one of the most difficult constraints in field service optimization?"

**Strong Answer:**  
"A **Tandem Service Call** (or Multi-Technician Synchronization) occurs when a maintenance task legally or physically requires two or more technicians to be present simultaneously at the exact same site:
* *Life Safety Mandates:* OSHA rules require a two-person safety standby crew when working inside energized high-voltage switchgear cabinets or confined spaces.
* *Physical Heavy Lifting:* Overhauls involving structural assemblies exceeding 50 kg that cannot be mechanically hoisted require multiple workers.
* *Complementary Skill Requirements:* Complex medical hardware repairs may require an electronics calibration specialist and a refrigeration mechanical engineer working in tandem.

This constraint is mathematically brutal for optimization engines because it **destroys route separability**:
$$\text{Arrival Time}(k_1, \text{Site } i) = \text{Arrival Time}(k_2, \text{Site } i)$$
The arrival time of Technician 1 at Node $i$ is coupled directly to the arrival time of Technician 2. If Technician 1 gets stuck in traffic 20 miles away, Technician 2 must sit idle outside the customer site, wasting billable hours. Tandem routing requires sophisticated joint-insertion heuristics with temporal synchronization buffers."

**Weak Answer:**  
"A tandem call is when two technicians go to the same job. It's difficult because they have to show up at the exact same time, and if one is late, the other has to wait."

---

### Question 8: "How does a high No-Fault-Found (NFF) rate in the field reflect diagnostic failures in the service dispatch loop?"

**Strong Answer:**  
"A high No-Fault-Found (NFF) rate—where 30% to 50% of components removed from machines and sent to repair shops test fully operational with zero defects—is directly rooted in dispatch and diagnostic failures:
1. **Shotgun Troubleshooting:** Under intense customer SLA downtime pressure, an undertrained or rushed technician swaps three expensive subassemblies simultaneously to see if the machine reboots, rather than performing methodical root-cause fault isolation.
2. **Superficial Symptom Matching:** Remote dispatchers rely on generic customer complaint descriptions (e.g., 'system overheated') and order high-value boards, when the actual physical root cause was an unseated sensor plug or clogged air filter.
3. **Intermittent Environmental Stressors:** Failures caused by site power fluctuations, electromagnetic interference, or ambient heat disappear once the component is removed and placed on a climate-controlled depot test bench.

NFF destroys aftermarket profitability: it ties up millions in reverse logistics transit, overwhelms MRO repair shops testing functional hardware, and leaves the actual root-cause defect inside the customer machine, guaranteeing a recurring breakdown."

**Weak Answer:**  
"NFF means the technician pulled a part that wasn't actually broken. It shows the technician didn't know how to diagnose the problem properly."

---

## 15. Key terms

- **First-Time Fix Rate (FTFR):** The percentage of service calls resolved on the initial visit without a return trip.
- **Service Window:** The contractually defined time interval $[e_i, l_i]$ during which technician arrival must occur.
- **Van Stock (Trunk Stock):** Mobile inventory of high-velocity replacement parts carried inside service vehicles.
- **Skills Matrix:** An auditable competency matrix mapping technicians to certified machine models and procedures.
- **Mean Time to Respond (MTTR_resp):** Elapsed time from customer ticket creation to technician physical on-site arrival.
- **Mean Time to Repair (MTTR_repair):** Elapsed active wrench time required to diagnose, repair, and test the machine.
- **Forward Stocking Location (FSL):** Localized forward parts depot providing emergency parts within 1–2 hours.
- **Nightly Boot Sweep:** Overnight courier delivery process restocking technician vans in residential driveways.
- **Hours of Service (HOS):** Statutory transport safety laws limiting daily driving hours and mandating rest breaks.
- **Adaptive Large Neighborhood Search (ALNS):** Metaheuristic routing algorithm utilizing destroy-and-repair operators.
- **Dynamic Rescheduling:** Real-time mid-day schedule adjustment absorbing emergency P1 breakdown calls.
- **SLA Penalty:** Contractual liquidated damages assessed for asset downtime exceeding guaranteed thresholds.
- **Priority 1 (P1):** Mission-critical emergency service call indicating complete machine stoppage.
- **Presumptive Bill of Materials (PBOM):** Statistically predicted spare parts required based on initial error telemetry.
- **Tandem Service Call:** Maintenance task requiring two or more technicians on site simultaneously.
- **No-Fault-Found (NFF):** Components removed from the field that test completely functional upon depot evaluation.
- **Regret Insertion:** Routing heuristic prioritizing jobs that have large cost penalties if not assigned to their best vehicle.
- **Deadhead Time:** Unproductive travel time spent driving between jobs or returning to home base.
- **Wrench Time:** Productive labor hours spent actively performing maintenance work on customer hardware.
- **Digital Debrief:** Mandatory post-service mobile workflow logging root causes, parts consumed, and core serials.

---

## 16. Sources

### Peer-reviewed and Academic — Grade A
1. **Cordeau, J. F., Laporte, G., Savelsbergh, M. W., & Vigo, D. (2007).** "Vehicle routing." *Handbooks in Operations Research and Management Science*, 14, 367-428. [DOI: 10.1016/S0927-0507(06)14006-2]
2. **Ropke, S., & Pisinger, D. (2006).** "An adaptive large neighborhood search heuristic for the pickup and delivery problem with time windows." *Transportation Science*, 40(4), 455-472. [DOI: 10.1287/trsc.1050.0135]
3. **Dohn, A., Rasmussen, M. S., & Larsen, J. (2011).** "The vehicle routing problem with time windows and temporal dependencies." *Networks*, 58(4), 273-289. [DOI: 10.1002/net.20467]
4. **Bodin, L., & Dietz, D. (2003).** "Design and scheduling of field service operations." *Computers & Operations Research*, 30(5), 757-777. [DOI: 10.1016/S0305-0548(02)00049-7]
5. **Aghezzaf, E. H. (2007).** "Robust inventory planning for spare parts supply chains with uncertain demand." *Journal of the Operational Research Society*, 58(8), 1073-1081. [DOI: 10.1057/palgrave.jors.2602263]

### Standards and Regulatory Frameworks — Grade B
6. **European Parliament and Council. (2003).** *Directive 2003/88/EC: Concerning certain aspects of the organisation of working time*. Brussels: Official Journal of the European Union.
7. **Federal Motor Carrier Safety Administration (FMCSA). (2020).** *Title 49 CFR Part 395: Hours of Service of Drivers*. Washington, D.C.: US Department of Transportation.
8. **National Fire Protection Association (NFPA). (2021).** *NFPA 70E: Standard for Electrical Safety in the Workplace*. Quincy: NFPA.
9. **International Organization for Standardization (ISO). (2014).** *ISO 55001:2014: Asset management — Management systems — Requirements*. Geneva: ISO.

### Industry Case Evidence and Technical Monographs — Grade C & D
10. **The Service Council. (2022).** *Field Service Performance & First-Time Fix Benchmark Report*. Boston: TSC Research.

---

## 17. Links

- **Parent Module:** [`_index.md`](_index.md) (Module 31 — Aftermarket, Spare Parts, and Service Supply Chains)
- **Root Survey File:** [`../../../course/31_aftermarket_spare_parts_service.md`](../../../course/31_aftermarket_spare_parts_service.md)
- **Previous Topic:** [`04_repair-loop-turnaround.md`](04_repair-loop-turnaround.md) (Repair Loops and Turnaround)
- **Next Topic:** [`06_service-contracts-sla.md`](06_service-contracts-sla.md) (Service Contracts and SLAs)
- **Related Methods:**
  - [`../../../course/12_transportation_routing_last_mile.md`](../../../course/12_transportation_routing_last_mile.md)
  - [`../../../course/13_warehousing_fulfillment_labor.md`](../../../course/13_warehousing_fulfillment_labor.md)
