# Schedule and Earned Value Analytics

> Module 34 — Energy, Utilities, and Capital Project Supply Chains
> Topic 06 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Module 10 (Inventory Optimization), Module 14 (Network Design & Facility Location), Topic 04 (Long-Lead Equipment and Project Procurement), Topic 05 (Bulk Materials and Construction Logistics)
> Version 2026.5

---

## 1. Why this topic exists

Capital infrastructure projects—combined-cycle power stations, nuclear generation units, liquefied natural gas (LNG) liquefaction trains, offshore production platforms, and high-voltage transmission networks—are governed by integrated critical path schedules spanning 36 to 84 months and budgets from $500M to in excess of $15B. In these industrial megaprojects, physical progress on site is mechanically coupled to the upstream supply chain. Concrete cannot be poured without rebar cages and embedded anchor bolts; steel pipe spools cannot be welded without valves and pipe hangers; and gas turbines cannot be set onto foundations without soleplates and structural grouting. When an engineered-to-order (ETO) component slips by six weeks in an overseas forge, the schedule impact does not stay confined to the procurement ledger. It cascades directly onto the critical path of the engineering, procurement, and construction (EPC) contractor.

The traditional financial and project controls architecture used to govern these investments is Earned Value Management (EVM), formally codified under ANSI/EIA-748. EVM measures performance by converting physical work into monetary terms: Planned Value ($PV$), Earned Value ($EV$), and Actual Cost ($AC$). Project controllers monitor two headline metrics: Cost Performance Index ($CPI = EV / AC$) and Schedule Performance Index ($SPI = EV / PV$). 

However, classic EVM possesses a mathematical failure mode that blinds capital project executives during the final, highest-risk third of a project. Because classic Schedule Variance is defined in monetary units ($SV = EV - PV$), as a delayed project nears completion, the cumulative physical work must eventually finish, meaning $EV$ asymptotically approaches the Budget at Completion ($BAC$). Because $PV$ also equals $BAC$ at the planned finish date, $SV$ converges to zero and $SPI$ converges to 1.0. A project that is eighteen months behind schedule and burning $2M per day in extended site indirect costs will mathematically report an $SPI$ of 0.96 at 95% completion, falsely signalling that schedule performance has recovered. Project directors rely on these lagging, distorted indicators until float is entirely erased, completion dates slip catastrophically, and liquidated damages (LDs) are triggered.

Furthermore, traditional Critical Path Method (CPM) scheduling treats procurement durations as deterministic point estimates. When multiple parallel supply chains converge on a single construction milestone—such as three distinct equipment packages required for a compressor train deck installation—the network experiences *merge event bias*. Even if every individual supplier has a 50% probability of on-time delivery, the joint probability that all three arrive on time drops to $0.5^3 = 0.125$. Deterministic float calculations fail to capture this non-linear risk, hiding the fact that non-critical procurement activities frequently become critical drivers of project completion.

This topic provides the analytical framework required to bridge supply chain performance, Earned Value Management, and critical path scheduling. It details the ANSI/EIA-748 material accounting standards, derives the time-based mathematics of Earned Schedule ($ES$, $SV_t$, $SPI_t$), establishes the quantitative mechanisms through which supply delays degrade contractor labor productivity and $CPI$, introduces Critical Chain feeding buffer analytics, and implements probabilistic Monte Carlo Schedule Risk Analysis (SRA).

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Map** procurement and supply chain milestones to ANSI/EIA-748 EVMS criteria (Guidelines 10, 21, 23, and 27) and select compliant progress earning rules for capital equipment.
2. **Prove** mathematically why classical EVM schedule metrics ($SV, SPI$) break down late in capital projects and compute time-based Earned Schedule ($ES$), Schedule Variance ($SV_t$), and Schedule Performance Index ($SPI_t$).
3. **Derive** independent time-based completion forecasts ($EAC_t$) and required future schedule performance ($TSPI$) using single-factor and composite performance indices.
4. **Quantify** the cascading financial impact of supply delivery delays on contractor labor efficiency, trade stacking, out-of-sequence execution, and the resulting degradation of the Cost Performance Index ($CPI$).
5. **Dimension** procurement feeding buffers using the Root-Square-Error Method (RSEM) and monitor supply chain risk using Critical Chain Project Management (CCPM) fever charts.
6. **Demonstrate** the mathematical origin of merge event bias in CPM networks and calculate the positive completion date bias using order statistics and Clark's approximation.
7. **Formulate** and execute a Monte Carlo Schedule Risk Analysis (SRA) on an integrated EPC procurement-construction network to compute $P_{50}$ and $P_{80}$ completion dates, Schedule Risk Reserves, and activity Criticality Indices ($CI$).

---

## 3. The core mental model

The causal feedback loop between equipment procurement lead times, site construction dynamics, and project controls metrics is governed by physical constraints and contractual accounting rules.

```text
====================================================================================================
               INTEGRATED SUPPLY CHAIN AND EARNED VALUE ANALYTICS ARCHITECTURE
====================================================================================================

 [ PROCUREMENT TRACK ]
   PO Placement ──> Eng Drawings ──> Factory Fab ──> FAT Passed ──> Shipping / Transit ──> Site Delivery
        │                │                │               │               │                   │
        │ Guideline 10   │ Progress Earning Milestone Rules (0/100, 50/50, % Complete)        │
        ▼                ▼                ▼               ▼               ▼                   ▼
 [ EARNED VALUE ENGINE ] ─────────────────────────────────────────────────────────> [ MATERIAL EV ]
   Planned Value PV(t) ───┐                                                                   │
   Actual Cost   AC(t) ───┼──> Classic EVM Metrics: SV = EV - PV ($), SPI = EV / PV          │
   Earned Value  EV(t) ───┘    * Fails as t > PD: SPI -> 1.0, SV -> $0 (Metric Collapse)      │
        │                                                                                     │
        ▼                                                                                     │
 [ EARNED SCHEDULE (ES) ] <───────────────────────────────────────────────────────────────────┘
   Time Projection: Find C where PV_C <= EV < PV_{C+1}
   ES = C + I  ===>  SV_t = ES - AT (time units)  ===>  SPI_t = ES / AT
   Completion Forecast: EAC_t = PD / SPI_t
        │
        │ Upstream Delivery Delay (Delta L > 0)
        ▼
 [ FEEDING BUFFER MONITORING ] ──> Critical Chain Feeding Buffer (FB)
   Buffer Penetration Ratio (BPR) vs Chain Completion (CCR)
   Fever Chart Diagnostic: Green (Normal) | Yellow (Expedite) | Red (Intervene)
        │
        │ Buffer Exhausted (BPR > 1.0)
        ▼
 [ SITE CONSTRUCTION IMPACT ]
   • Construction Work Package (CWP) Blocked (Missing Spools, Skids, Valves)
   • Direct Craft Labor Idling: Standby Cost Accrual ($AC continues burning, EV = 0)
   • Out-of-Sequence Work: Trades diverted to non-ready areas
     - Trade Stacking & Congestion (Productivity factor eta drops to 0.55 - 0.75)
     - Scaffolding & Temporary Support Rework
     - Result: AC escalates sharply while EV yield drops ===> CPI collapses: CPI = EV / AC << 1.0
        │
        ▼
 [ PROBABILISTIC SRA (MONTE CARLO) ]
   • Deterministic CPM Merge Bias: E[max(T_1, T_2, ...)] > max(E[T_1], E[T_2], ...)
   • Stochastic Lead Times: L_i ~ Beta-PERT(a_i, m_i, b_i)
   • Distribution of Completion Date P(T <= t) ===> P50, P80 Forecasts
   • Criticality Index CI_i = P(Activity i on Critical Path)
====================================================================================================
```

When an upstream supplier delays an engineered equipment skid, the failure propagates through three interconnected layers:
1. **The Critical Chain Layer:** The delay consumes the procurement feeding buffer. Once the buffer is exhausted, the delivery slip impacts the critical chain, converting float into negative project float.
2. **The Site Production Layer:** Craft labor scheduled to install the skid cannot proceed. The contractor either idles crews (accruing standby costs) or diverts workers out of sequence to non-critical tasks. Out-of-sequence work induces spatial crowding, dilution of supervisory oversight, and trade stacking. Labor efficiency falls, generating far less physical progress per paid man-hour.
3. **The Financial Analytics Layer:** As work packages stall, planned earned value generation stops ($\Delta EV = 0$). Concurrently, actual costs continue to burn ($\Delta AC > 0$) due to standby payroll, leased cranes, and overhead. Contractor $CPI$ collapses. Classic $SPI$ distorts the schedule delay, but Earned Schedule ($ES$) detects the slip in calendar time, signaling true schedule variance ($SV_t < 0$) and extending the probabilistic completion forecast ($EAC_t$).

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Industry Misuse or Pitfall | Unit or Standard Code |
|---|---|---|---|
| **Planned Value (PV / BCWS)** | The authorized budget assigned to scheduled work packages to be accomplished over time. | Confusing PV with the cash flow expenditure forecast; PV measures work budgeted, not invoices paid. | USD ($), ANSI/EIA-748 Guideline 8 |
| **Earned Value (EV / BCWP)** | The measure of physical work performed expressed in terms of the budget authorized for that work. | Claiming progress based on vendor invoice payments or PO placement rather than physical verification. | USD ($), ANSI/EIA-748 Guideline 21 |
| **Actual Cost (AC / ACWP)** | The realized cost incurred for work performed on an activity during a given time period. | Omitting unvouchered liabilities, committed equipment lease costs, or demurrage fees from current AC. | USD ($), ANSI/EIA-748 Guideline 16 |
| **Budget at Completion (BAC)** | The total authorized budget for the complete scope of a project, control account, or work package. | Adjusting BAC to absorb cost overruns without a formally approved baseline change request. | USD ($), Total PV at Planned Duration |
| **Schedule Variance (SV)** | Classic cost-based schedule variance: $SV = EV - PV$. | Assuming positive SV means time ahead; SV measures dollar value of work, which always hits $0 at completion. | USD ($), ANSI/EIA-748 |
| **Schedule Performance Index (SPI)** | Classic efficiency indicator: $SPI = EV / PV$. | Relying on SPI late in the project when it mathematically forces itself back to 1.0 regardless of delay. | Dimensionless ratio |
| **Earned Schedule (ES)** | The metric time at which the current Earned Value should have been achieved according to the baseline PV curve. | Treating ES as a simple calendar date rather than an interpolated continuous time variable. | Time units (months, weeks, days) |
| **Actual Time (AT)** | The actual duration elapsed from project start to the current reporting status date. | Counting non-working shutdown periods inconsistently between planned calendar and actual time. | Time units (months, weeks, days) |
| **Planned Duration (PD)** | The total planned project execution duration from baseline start to baseline completion. | Modifying PD informally after schedule delays instead of maintaining the approved Performance Measurement Baseline. | Time units (months, weeks, days) |
| **Time-based Schedule Variance ($SV_t$)** | The difference in time units between Earned Schedule and Actual Time: $SV_t = ES - AT$. | Interpreting negative $SV_t$ as a currency deficit; it strictly represents calendar time delay. | Time units (months, weeks, days) |
| **Time-based Schedule Performance Index ($SPI_t$)** | The time-based efficiency indicator: $SPI_t = ES / AT$. | Confusing $SPI_t$ with classical $SPI$; $SPI_t$ does not collapse to 1.0 at project completion. | Dimensionless ratio |
| **Estimate at Completion, Time ($EAC_t$)** | The forecasted total project duration derived from Earned Schedule: $EAC_t = PD / SPI_t$. | Assuming $EAC_t$ accounts for future critical path shifts without validating network logic. | Time units (months, weeks, days) |
| **To-Complete Schedule Performance Index (TSPI)** | The schedule efficiency required on remaining scope to meet a target completion date: $TSPI = \frac{PD - ES}{PD - AT}$. | Failing to recognize that TSPI > 1.20 is historically unachievable in heavy construction. | Dimensionless ratio |
| **Cost Performance Index (CPI)** | The cost efficiency of physical work completed: $CPI = EV / AC$. | Assuming high CPI offsets a collapsing $SPI_t$; severe schedule slip eventually degrades CPI through indirect burn. | Dimensionless ratio |
| **Critical Chain Project Management (CCPM)** | A schedule network method aggregating activity safety margins into strategic feeding and project buffers. | Treating CCPM buffers as "free float" that contractors can consume without operational consequence. | Management methodology (Goldratt) |
| **Feeding Buffer (FB)** | A pooled time buffer placed at the confluence where a non-critical supply path merges into the critical chain. | Allowing suppliers to incorporate internal padding before calculating the feeding buffer. | Time units (days, weeks) |
| **Buffer Penetration Ratio (BPR)** | The percentage of a critical chain buffer consumed by actual delays: $BPR = \Delta \tau_{consumed} / FB_{total}$. | Evaluating BPR in isolation without cross-referencing the percentage of the feeder chain completed. | Percentage (%) |
| **Fever Chart** | A diagnostic plot of Buffer Penetration Ratio versus Chain Completion Ratio divided into Green, Yellow, and Red risk zones. | Treating Yellow and Red zone alerts as informational rather than mandatory commercial intervention triggers. | Control chart format |
| **Merge Event Bias** | The mathematical certainty that the expected duration of converging parallel paths exceeds the maximum of their individual expected durations. | Assuming the critical path through a merge point is strictly determined by the path with the largest single mean. | Statistical phenomenon ($\mathbb{E}[\max] \ge \max \mathbb{E}$) |
| **Criticality Index (CI)** | The probability that an activity or procurement package sits on the critical path across all Monte Carlo iterations. | Assuming an activity with 10 days of deterministic total float has a Criticality Index of zero. | Percentage (%) |
| **Schedule Sensitivity Index (SSI)** | A metric measuring the correlation between an activity's duration variance and total project duration variance. | Ignoring high-variance procurement items because their baseline duration is short. | Dimensionless or percentage |
| **Trade Stacking** | The congestion of multiple craft trades into a confined physical work area exceeding optimal crew densities. | Ordering trades into a block to "catch up" without calculating the net loss of labor efficiency factor. | Phenomenon (labor efficiency loss) |
| **Out-of-Sequence Work** | Constructing downstream components prior to upstream equipment placement due to supply delays. | Assuming out-of-sequence work generates full EV; it typically requires extensive subsequent rework. | Field management practice |

---

## 5. The mechanics, step by step

Implementing integrated schedule and earned value analytics on an industrial capital project follows a seven-step sequence.

```text
====================================================================================================
                        SEVEN-STEP ANALYTICS AND GOVERNANCE SEQUENCE
====================================================================================================
  Step 1: ANSI/EIA-748 Baseline Architecture & Material Earning Rules
            │  Establish WBS/OBS, Control Accounts, and Milestone Earning Gates
            ▼
  Step 2: Classical EVM Calculation & Flaw Detection
            │  Compute PV, EV, AC, SV ($), SPI; detect late-stage metric collapse
            ▼
  Step 3: Earned Schedule (ES) Time-Based Analytics
            │  Project EV onto PV(t); interpolate ES, compute SV_t, SPI_t, EAC_t, TSPI
            ▼
  Step 4: Contractor Productivity & Cost Degradation Modeling
            │  Couple material delay Delta L to standby idle costs and trade stacking efficiency
            ▼
  Step 5: Critical Chain Feeding Buffer Dimensioning & Fever Charts
            │  Calculate FB via RSEM; track Buffer Penetration Ratio vs Chain Completion
            ▼
  Step 6: Probabilistic Schedule Risk Analysis (Monte Carlo SRA)
            │  Model Beta-PERT lead times; quantify merge event bias, P50/P80, and Criticality Index
            ▼
  Step 7: Integrated Governance, Corrective Action & Expediting Playbook
               Trigger commercial interventions (air freight, dual-sourcing, re-sequencing)
====================================================================================================
```

### Step 1: Baseline Architecture and Material Earning Rules under ANSI/EIA-748

Under ANSI/EIA-748-D, procurement activities cannot be managed as arbitrary milestone markers or simple accounting cost transfers. They must be integrated into Control Accounts (CAs) and Work Packages (WPs) tied to the Work Breakdown Structure (WBS) and Organizational Breakdown Structure (OBS).

1. **Establish the Material Control Account:** Every major equipment package (e.g., Gas Turbine Generator, Heat Recovery Steam Generator, Demineralized Water Treatment Plant) must reside within an identified CA managed by a designated Control Account Manager (CAM).
2. **Define Progress Earning Rules (ANSI/EIA-748 Guideline 10 & 21):**
   - *0/100 Rule:* Applicable only to short-duration, low-value catalog items delivered within a single accounting period. Zero percent earned upon order placement; 100% earned upon site receipt, verified by warehouse inspection.
   - *50/50 Rule:* Applicable to standard manufactured items spanning two accounting periods. Fifty percent earned upon verified shipment (Bill of Lading); remaining 50% earned upon receipt and acceptance inspection.
   - *Milestone Weighting with Value Gates (Mandatory for Engineered-to-Order Equipment):* Long-lead packages spanning several months or years must earn value based on objective, auditable engineering and manufacturing milestones.

```text
+--------------------------------------------------------------------------------------------------+
|                   STANDARD MILESTONE VALUE GATES FOR CAPITAL EQUIPMENT (ETO)                     |
+--------------------------------------------------------------------------------------------------+
| Gate | Milestone Event                                  | Value Weight | Verification Artifact    |
+------+--------------------------------------------------+--------------+--------------------------+
| G1   | Purchase Order Placed & Subcontract Executed     |     10%      | Executed Commercial PO   |
| G2   | Vendor Engineering Drawings / Calculation Freeze |     15%      | Approved for Constr (AFC)|
| G3   | Raw Material / Forging Ingot Ultrasonic Test OK  |     15%      | Certified Mill Test Rep  |
| G4   | Major Sub-assembly & Casing Hydrostatic Test OK  |     15%      | Third-Party Insp Report  |
| G5   | Factory Acceptance Test (FAT) Complete           |     15%      | Signed FAT Protocol      |
| G6   | Ex-Works Dispatch & Ocean Bill of Lading Issued  |     10%      | Clean On-Board B/L       |
| G7   | Arrival at Project Laydown & Receipt Inspection  |     10%      | Clean Material Rec Report|
| G8   | Over, Short & Damaged (OS&D) Clearance           |     10%      | Signed OS&D Signoff Form |
+--------------------------------------------------------------------------------------------------+
```

> **Note:** ANSI/EIA-748 Guideline 21 forbids earning 100% of material budget upon physical arrival on site if preservation, staging, and technical acceptance testing are unperformed. Earning value prematurely artificially inflates $EV$, concealing delivery defects until installation failure occurs.

### Step 2: Classical EVM Calculation and Proof of Late-Stage Breakdown

At each status date $AT$, compute the baseline cumulative metrics:
$$\text{Planned Value: } PV(AT) = \sum_{j \in \mathcal{W}} PV_j(AT)$$
$$\text{Earned Value: } EV(AT) = \sum_{j \in \mathcal{W}} EV_j(AT)$$
$$\text{Actual Cost: } AC(AT) = \sum_{j \in \mathcal{W}} AC_j(AT)$$
$$\text{Cost Performance Index: } CPI(AT) = \frac{EV(AT)}{AC(AT)}$$
$$\text{Cost Variance: } CV(AT) = EV(AT) - AC(AT)$$
$$\text{Classic Schedule Variance: } SV(AT) = EV(AT) - PV(AT)$$
$$\text{Classic Schedule Performance Index: } SPI(AT) = \frac{EV(AT)}{PV(AT)}$$

#### Mathematical Proof of Classical SPI Breakdown
Let a capital project have a total authorized budget $BAC$ and a planned duration $PD$. The cumulative Planned Value curve is non-decreasing with $PV(0) = 0$ and $PV(t) = BAC$ for all $t \ge PD$.
Suppose the project is delayed and completes at actual completion time $T_{act} > PD$. 
By definition, all authorized scope is complete at $T_{act}$, such that $EV(T_{act}) = BAC$.
Now evaluate the classic schedule metrics as actual time $t$ approaches $T_{act}$:
$$\lim_{t \to T_{act}} SV(t) = \lim_{t \to T_{act}} [EV(t) - PV(t)] = BAC - BAC = \$0$$
$$\lim_{t \to T_{act}} SPI(t) = \lim_{t \to T_{act}} \frac{EV(t)}{PV(t)} = \frac{BAC}{BAC} = 1.000$$

This proves that regardless of how late a project finishes—whether six months or four years past $PD$—the classic schedule metrics will mathematically report zero schedule variance ($SV = \$0$) and perfect schedule efficiency ($SPI = 1.00$). 
Because $PV$ stops growing at $PD$ while $EV$ continues climbing toward $BAC$, $SPI$ begins artificially rising back toward 1.0 midway through any delayed project. Classic $SPI$ measures monetary volume completed relative to a frozen baseline, not elapsed time.

### Step 3: Derivation and Execution of Earned Schedule (ES) Analytics

To overcome the mathematical breakdown of classical EVM, Earned Schedule ($ES$), developed by Lipke (2003) and standardized under ISO 21508, translates cumulative Earned Value back into time units by projecting $EV$ onto the planned value curve $PV(t)$.

```text
====================================================================================================
                              EARNED SCHEDULE (ES) GEOMETRIC PROJECTION
====================================================================================================
 Planned Value $ 
      ^
  BAC ┼───────────────────────────────────────────────────────────/──────── (PV Curve)
      │                                                          /
      │                                                         /
   EV ┼ - - - - - - - - - - - - - - - - - - - - - - - - - - - -* 
      │                                                       /│
PV_{C+1}┼─────────────────────────────────────────────────────* │
      │                                                     / ││
      │                                                    /  ││
 PV_C ┼───────────────────────────────────────────────────*   ││
      │                                                  /│   ││
      │                                                 / │   ││
      │                                                /  │   ││
    0 ┼───────────────────────────────────────────────┴───┴───┴┴──────────────────────────> Time (t)
                                                      0   C   ES  C+1                AT
                                                          |<I>|
                                                          |<- - - - - SV_t - - - - ->|
====================================================================================================
```

#### Step-by-Step Calculation of Earned Schedule
1. **Identify the Time Interval Index $C$:** At status date $AT$, obtain cumulative Earned Value $EV(AT)$. Inspect the discrete Planned Value array $PV_0, PV_1, \dots, PV_{PD}$ to find the integer time index $C$ satisfying:
   $$PV_C \le EV(AT) < PV_{C+1}$$
   where $C \in \{0, 1, 2, \dots, PD - 1\}$. If $EV(AT) \ge BAC$, then $C = PD$.

2. **Compute the Fractional Time Increment $I$:** Interpolate linearly between the discrete planned value intervals:
   $$I = \frac{EV(AT) - PV_C}{PV_{C+1} - PV_C}$$
   where $0 \le I < 1.0$.

3. **Determine Earned Schedule ($ES$):**
   $$ES = C + I$$
   $ES$ represents the exact point in planned time corresponding to the volume of physical work achieved at $AT$.

4. **Calculate Time-based Schedule Variance ($SV_t$):**
   $$SV_t = ES - AT$$
   - If $SV_t = 0$: Project is precisely on schedule.
   - If $SV_t < 0$: Project is delayed by $|SV_t|$ calendar periods.
   - If $SV_t > 0$: Project is ahead of schedule by $SV_t$ calendar periods.
   Unlike classic $SV$, $SV_t$ never collapses to zero at project finish if the project is late; it equals the final schedule slippage: $SV_t(T_{act}) = PD - T_{act} < 0$.

5. **Calculate Time-based Schedule Performance Index ($SPI_t$):**
   $$SPI_t = \frac{ES}{AT}$$
   - $SPI_t < 1.0$: Work is being completed at a lower rate than planned in time units.
   - At completion of a late project, $SPI_t(T_{act}) = \frac{PD}{T_{act}} < 1.0$. The indicator retains its integrity across the entire project lifecycle.

6. **Forecast Time Estimate at Completion ($EAC_t$):**
   - *Independent Schedule Forecast (Single Factor):*
     $$EAC_t = \frac{PD}{SPI_t} = PD \cdot \frac{AT}{ES}$$
   - *Composite Forecast (Accounting for Cost-Schedule Interaction):*
     $$EAC_t = AT + \frac{PD - ES}{SPI_t \cdot CPI}$$
     When contractor $CPI$ is depressed due to out-of-sequence work, the composite forecast captures the secondary labor drag that extends the duration further than $SPI_t$ alone predicts.

7. **Determine To-Complete Schedule Performance Index ($TSPI$):**
   To finish on the original planned date $PD$:
   $$TSPI_{PD} = \frac{PD - ES}{PD - AT}$$
   To finish by an approved extended target date $ED > PD$:
   $$TSPI_{ED} = \frac{PD - ES}{ED - AT}$$
   If $TSPI > 1.15$, the contractor must accelerate work velocity by more than 15% across remaining scope. In heavy industrial construction, $TSPI > 1.20$ without massive shift additions or scope de-scoping is mathematically unachievable.

### Step 4: Supply Delivery Delay Coupling to Contractor Productivity, CPI, and Cost Escalation

When an engineered equipment or bulk material delivery is delayed by $\Delta L$ calendar days, the contractor's cost baseline experiences non-linear escalation through two distinct operational pathways: idle standby costs and out-of-sequence labor efficiency dilution.

```text
====================================================================================================
                    CAUSAL DYNAMICS: SUPPLY DELAY TO CONTRACTOR COLLAPSE
====================================================================================================
                        Supply Delivery Delayed by Delta L Days
                                          │
                     ┌────────────────────┴────────────────────┐
                     ▼                                         ▼
         [ DIRECT STANDBY IDLING ]                [ OUT-OF-SEQUENCE DIVERSION ]
         • Craft crews on standby pay             • Crews diverted to unready areas
         • Leased cranes & equipment idle         • Spatial crowding & trade stacking
         • Fixed site overhead burn               • Workarounds, double handling, re-rigging
                     │                                         │
                     ▼                                         ▼
         Delta AC_idle = (alpha*C_lab + C_ind)*Delta L   Labor efficiency factor eta drops:
         Delta EV = 0 (Work blocked)                   eta = eta_0 * (1 - delta_stack) * (1 - delta_cong)
                     │                                         │
                     ▼                                         ▼
         Direct CPI Degradation                   Labor hours escalate: ManHours_act = ManHours_bud / eta
                                                  Rework required when missing spools/skids arrive
                                                               │
                                                               ▼
                                                  Explosive Cost Overrun: Delta AC_stack >> 0
                                                               │
                     ┌─────────────────────────────────────────┘
                     ▼
         CONTRACTOR MARGIN COLLAPSE ===> CLAIMS, DISPUTES, OR SOLVENCY CRISIS
====================================================================================================
```

#### Analytical Formulation of Labor Inefficiency and Cost Escalation
Let a planned installation work package have budgeted direct labor hours $H_{bud}$, direct labor burn rate $R_{direct}$ ($/man-hour), and site indirect cost burn rate $C_{indir}$ ($/day).

1. **Pathway A: Direct Labor Idling:**
   If the contractor holds the crew on site awaiting delivery:
   $$\Delta AC_{idle} = \left( \alpha_{standby} \cdot N_{craft} \cdot H_{shift} \cdot R_{direct} + C_{indir} \right) \cdot \Delta L$$
   where $\alpha_{standby} \in [0.70, 1.00]$ is the contractually mandated standby payment factor and $N_{craft}$ is crew headcount. Because the work package cannot progress, $\Delta EV = 0$.

2. **Pathway B: Out-of-Sequence Work and Trade Stacking:**
   To avoid paying idle standby, the contractor redirects craft crews to non-critical or partially ready work packages. This creates physical trade stacking (multiple trades occupying the same physical elevation or module bay) and spatial congestion.
   The realized labor efficiency factor $\eta_{actual} \in (0, 1]$ is modeled using the multi-factor compounding formula:
   $$\eta_{actual} = \eta_0 \cdot (1 - \delta_{stack}) \cdot (1 - \delta_{cong}) \cdot (1 - \delta_{dilution})$$
   where:
   - $\eta_0$ is baseline productivity (typically 1.00).
   - $\delta_{stack}$ is the trade stacking penalty: $\delta_{stack} = \min\left(0.40, 0.12 \cdot (N_{trades} - 1)\right)$ for $N_{trades} > 1$ operating in the same workspace.
   - $\delta_{cong}$ is the density congestion factor based on available net floor area:
     $$\delta_{cong} = \max\left(0, \frac{\text{Actual Crew Density} - \text{Optimal Crew Density}}{\text{Optimal Crew Density}} \cdot 0.25\right)$$
   - $\delta_{dilution}$ is the supervisory dilution penalty when foreman-to-craft ratio drops below 1:8 due to fragmented work fronts: $\delta_{dilution} \approx 0.10 \text{ to } 0.20$.

3. **Total Escalated Labor Hours and Cost:**
   The actual hours required to execute the remaining budgeted hours $H_{rem}$ under degraded efficiency:
   $$H_{actual} = \frac{H_{rem}}{\eta_{actual}}$$
   When the delayed equipment finally arrives on site, previously installed out-of-sequence interferences (e.g., cable trays, scaffolding, small-bore piping) must be uninstalled, modified, and re-installed. Rework labor adds:
   $$H_{rework} = \beta_{rework} \cdot H_{bud}$$
   where $\beta_{rework} \in [0.15, 0.35]$.
   The total cost impact to actual cost $AC$:
   $$\Delta AC_{total} = (H_{actual} - H_{rem} + H_{rework}) \cdot R_{direct} + \Delta AC_{idle}$$
   Because $EV$ is anchored to the original budgeted baseline value ($EV = H_{bud} \cdot R_{direct}$), the contractor's realized Cost Performance Index degrades to:
   $$CPI_{realized} = \frac{EV_{baseline}}{AC_{baseline} + \Delta AC_{total}} \ll 1.0$$

### Step 5: Critical Chain Feeding Buffer Dimensioning and Fever Charts

Under Critical Chain Project Management (CCPM), non-critical procurement chains merging into the critical chain are protected by a strategically sized Feeding Buffer ($FB$). The feeding buffer absorbs upstream supply chain lead-time volatility, decoupling factory and logistics variance from the critical construction path.

#### 1. Sizing the Feeding Buffer: Root-Square-Error Method (RSEM)
Instead of adding individual safety float to every vendor purchase order (which is wasted via Parkinson's Law and Student Syndrome), task durations are cut to aggressive 50% probability estimates ($m_i$), and aggregated safety time is pooled into the feeding buffer.
For a procurement feeding chain consisting of $n$ independent sequential activities:
$$FB_{RSEM} = \sqrt{\sum_{j=1}^{n} (b_j - m_j)^2}$$
where $m_j$ is the median (50% confidence) duration and $b_j$ is the safe/pessimistic (90% confidence) duration.

#### 2. Dynamic Fever Chart Monitoring
As procurement unfolds, two non-dimensional metrics are calculated weekly:
$$\text{Chain Completion Ratio (CCR): } CCR = \frac{\sum_{j \in \text{Completed}} m_j}{\sum_{k \in \text{Chain}} m_j}$$
$$\text{Buffer Penetration Ratio (BPR): } BPR = \frac{\Delta \tau_{consumed}}{FB_{RSEM}}$$
where $\Delta \tau_{consumed} = \max(0, \text{Actual Feeder Slip to Date})$.

The fever chart defines three operational operating zones:

```text
====================================================================================================
                               CRITICAL CHAIN FEVER CHART GEOMETRY
====================================================================================================
 Buffer Penetration Ratio (BPR)
   1.00 ┼─────────────────────────────────────────────────────────────────────────────
        │                                                     / RED ZONE
   0.80 ┼                                                 /   (Executive Intervention)
        │                                             /
   0.67 ┼─────────────────────────────────────────/ 
        │                                     /
   0.60 ┼                                 /       YELLOW ZONE
        │                             /           (Expediting & Recovery Plan)
   0.40 ┼                         /
   0.33 ┼─────────────────────/ 
        │                 /
   0.20 ┼             /               GREEN ZONE (Normal Monitoring)
        │         /
   0.00 ┼─────/───────────────────────────────────────────────────────────────────────>
        0.00        0.20        0.40        0.60        0.80        1.00   Chain Completion (CCR)
====================================================================================================
```

The mathematical boundary lines defining the zones are:
$$\text{Yellow Warning Boundary: } BPR_{warn}(CCR) = \frac{1}{3} + \frac{1}{3} \cdot CCR$$
$$\text{Red Critical Boundary: } BPR_{crit}(CCR) = \frac{2}{3} + \frac{1}{3} \cdot CCR$$

- **Green Zone ($BPR \le BPR_{warn}$):** Normal operations. Lead time slips are within statistical expectations. No intervention required.
- **Yellow Zone ($BPR_{warn} < BPR < BPR_{crit}$):** Warning state. Expediter dispatched to vendor shop; daily progress calls; identify dual-freight or hot-shot transport options.
- **Red Zone ($BPR \ge BPR_{crit}$):** Critical failure state. The feeding buffer is exhausting faster than chain progress. Authorize emergency air charter, divert sub-assembly to domestic backup fabricators, or execute immediate site work-package resequencing.

### Step 6: Probabilistic Schedule Risk Analysis (Monte Carlo SRA) and Merge Bias

#### The Mathematical Mechanism of Merge Event Bias
Traditional CPM calculates completion time deterministically by taking the maximum of incoming path durations:
$$T_{merge}^{det} = \max(d_1, d_2, \dots, d_K)$$
In reality, activity durations are random variables $X_1, X_2, \dots, X_K$. The true expected merge time is:
$$\mathbb{E}[T_{merge}] = \mathbb{E}\left[\max(X_1, X_2, \dots, X_K)\right]$$

By Jensen's Inequality and order statistics, for any convex function $g(x) = \max(x_1, \dots, x_K)$:
$$\mathbb{E}[\max(X_1, \dots, X_K)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2], \dots, \mathbb{E}[X_K])$$
with strict inequality whenever $K \ge 2$, paths have overlapping support, and variance $\sigma^2 > 0$.

#### Closed-Form Quantification via Clark's Formula
For two independent normally distributed converging paths $X_1 \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $X_2 \sim \mathcal{N}(\mu_2, \sigma_2^2)$, Clark (1961) derived the exact expected value:
$$\mathbb{E}[\max(X_1, X_2)] = \mu_1 \Phi\left(\frac{\mu_1 - \mu_2}{\theta}\right) + \mu_2 \Phi\left(\frac{\mu_2 - \mu_1}{\theta}\right) + \theta \phi\left(\frac{\mu_1 - \mu_2}{\theta}\right)$$
where:
$$\theta = \sqrt{\sigma_1^2 + \sigma_2^2}$$
$\Phi(\cdot)$ is the standard normal cumulative distribution function, and $\phi(\cdot)$ is the standard normal probability density function.

Consider a balanced merge point where two converging procurement chains have identical expected durations $\mu_1 = \mu_2 = \mu$ and identical lead-time standard deviations $\sigma_1 = \sigma_2 = \sigma$. Then $\theta = \sigma \sqrt{2}$.
$$\mathbb{E}[\max(X_1, X_2)] = \mu \Phi(0) + \mu \Phi(0) + \sigma \sqrt{2} \cdot \phi(0) = \mu(0.5) + \mu(0.5) + \sigma \sqrt{2} \cdot \frac{1}{\sqrt{2\pi}} = \mu + \frac{\sigma}{\sqrt{\pi}} \approx \mu + 0.5642 \sigma$$

The deterministic CPM schedule predicts a completion time of $\mu$, entirely missing the positive bias of $+0.5642 \sigma$. On a capital equipment procurement path where $\sigma = 6$ weeks, deterministic CPM underestimates the merge milestone by $0.5642 \times 6 = 3.39$ weeks.

#### Generalization to $K$ Converging Procurement Paths
When $K$ independent, identically distributed paths with normal lead-time variance $\mathcal{N}(\mu, \sigma^2)$ merge into a common construction milestone, the expected milestone date is governed by extreme value order statistics:
$$\mathbb{E}\left[\max_{1 \le j \le K} X_j\right] = \mu + \sigma \cdot \mathbb{E}[Z_{K:K}]$$
where $Z_{K:K}$ is the maximum order statistic from $K$ standard normal variates.

```text
+--------------------------------------------------------------------------------------------------+
|                   MERGE BIAS CONSTANTS FOR K SYMMETRIC PROCUREMENT PATHS                         |
+--------------------+--------------------------------+--------------------------------------------+
| Number of Paths (K)| Normal Order Constant E[Z_K:K] | Example Slip Bias (at sigma = 6 weeks)     |
+--------------------+--------------------------------+--------------------------------------------+
|         1          |             0.0000             |  0.00 weeks (Deterministic CPM holds)      |
|         2          |             0.5642             | +3.39 weeks                                |
|         3          |             0.8463             | +5.08 weeks                                |
|         4          |             1.0294             | +6.18 weeks                                |
|         5          |             1.1630             | +6.98 weeks                                |
|         6          |             1.2672             | +7.60 weeks                                |
|         8          |             1.4236             | +8.54 weeks                                |
|        10          |             1.5388             | +9.23 weeks                                |
+--------------------+--------------------------------+--------------------------------------------+
```

As $K \to \infty$, Fisher-Tippett-Gnedenko extreme value theory proves that the maximum asymptotically approaches:
$$\mathbb{E}[Z_{K:K}] \approx \sqrt{2 \ln K} - \frac{\ln(\ln K) + \ln(4\pi)}{2\sqrt{2 \ln K}} + \frac{\gamma}{\sqrt{2 \ln K}}$$
where $\gamma \approx 0.5772$ is the Euler-Mascheroni constant.

In power plants or offshore platform decks where 8 to 12 major engineered equipment skids must all be landed before structural deck closeout can occur, merge event bias guarantees an unmodeled critical path slip of 8 to 10 weeks even when every single vendor is on target on average ($\mathbb{E}[X_j] = \mu$). Deterministic float calculations are completely blind to this mathematical reality.

#### Monte Carlo Simulation Formulation
1. Assign three-point estimates $(a_i, m_i, b_i)$ to each procurement and construction activity:
   - $a_i$: Optimistic duration (10th percentile).
   - $m_i$: Most likely duration (mode).
   - $b_i$: Pessimistic duration (90th percentile).
2. Parameterize a Beta-PERT distribution for each activity $i$:
   $$\mu_i = \frac{a_i + 4m_i + b_i}{6}, \quad \sigma_i = \frac{b_i - a_i}{6}$$
   $$\alpha_i = 1 + 4 \cdot \frac{m_i - a_i}{b_i - a_i}, \quad \beta_i = 1 + 4 \cdot \frac{b_i - m_i}{b_i - a_i}$$
3. Execute $N = 10,000$ iterations. In each iteration $k$, sample $d_{i,k} \sim \text{PERT}(\alpha_i, \beta_i)$, perform a complete forward and backward network pass, and record:
   - Total project duration $T_{proj,k}$.
   - The critical path binary vector $\mathbf{z}_k \in \{0, 1\}^M$.
4. Calculate SRA Risk Quantiles:
   - $P_{50}$: Median project completion date ($\mathbb{P}(T_{proj} \le P_{50}) = 0.50$).
   - $P_{80}$: Risk-adjusted commitment target ($\mathbb{P}(T_{proj} \le P_{80}) = 0.80$).
   - Schedule Risk Reserve: $SRR = P_{80} - P_{50}$ (or $P_{80} - T_{deterministic}$).
5. Compute Activity Criticality Index ($CI_i$):
   $$CI_i = \frac{1}{N} \sum_{k=1}^N \mathbb{I}(\text{Activity } i \text{ is on Critical Path in iteration } k) \times 100\%$$
   Activities with deterministic float but high variance frequently exhibit $CI > 50\%$, proving they are stealth critical drivers.

### Step 7: Integrated Governance, Corrective Action, and Expediting Playbook

Integrate the analytical streams into a binding weekly governance cadence:

```text
+--------------------------------------------------------------------------------------------------+
|                   INTEGRATED SUPPLY-SCHEDULE INTERVENTION DECISION MATRIX                        |
+-------------------+--------------------+--------------------+------------------------------------+
| Fever Chart Zone  | SPI_t / TSPI       | Criticality Index  | Mandatory Operational Action       |
+-------------------+--------------------+--------------------+------------------------------------+
| Green Zone        | SPI_t >= 0.95      | CI < 30%           | Maintain standard monitoring; no   |
| (BPR <= Warn)     | TSPI <= 1.05       |                    | schedule intervention required.    |
+-------------------+--------------------+--------------------+------------------------------------+
| Yellow Zone       | 0.88 <= SPI_t < 0.95| 30% <= CI < 60%   | Deploy field resident expediter;   |
| (Warn < BPR < Crit)| 1.05 < TSPI <= 1.15|                   | mandate 6-day factory work weeks;  |
|                   |                    |                    | secure dedicated ocean charter slot.|
+-------------------+--------------------+--------------------+------------------------------------+
| Red Zone          | SPI_t < 0.88       | CI >= 60%          | Executive intervention:            |
| (BPR >= Crit)     | TSPI > 1.15        |                    | 1. Authorize Antonov air charter;  |
|                   |                    |                    | 2. Re-sequence CWPs to unblock EV; |
|                   |                    |                    | 3. Establish split-delivery plan;  |
|                   |                    |                    | 4. Issue formal commercial claim.  |
+-------------------+--------------------+--------------------+------------------------------------+
```

---

## 6. Worked example

### Operational Context
A public utility is constructing the West Bay Energy Center, a 1,200 MW Combined-Cycle Gas Turbine (CCGT) power plant.
- Total Authorized Budget at Completion: $BAC = \$480\text{M}$.
- Planned Duration: $PD = 24.0\text{ months}$.
- Reporting Status Date: Month $AT = 14.0\text{ months}$.

At Month 14, two major critical supply chains are experiencing delays:
1. **The Generator Step-Up (GSU) 500 kV Transformer Package:** The high-voltage bushing supplier suffered a factory tooling failure in Europe, delaying factory acceptance testing.
2. **P91 High-Pressure Main Steam Piping Spools:** Delayed at an overseas induction bending yard due to weld heat-treatment non-conformance.

### Baseline Planned Value Curve ($PV$) vs. Realized Actuals
The approved 24-month Performance Measurement Baseline ($PV$) and realized project actuals ($EV, AC$) across the first 14 months are tabulated below:

```text
+-------+------------+------------+------------+------------+------------+------------+
| Month | Monthly PV | Cumul PV   | Monthly EV | Cumul EV   | Monthly AC | Cumul AC   |
| (t)   | ($M)       | ($M)       | ($M)       | ($M)       | ($M)       | ($M)       |
+-------+------------+------------+------------+------------+------------+------------+
|   0   |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |    0.0     |
|   1   |    8.0     |    8.0     |    8.0     |    8.0     |    7.5     |    7.5     |
|   2   |   12.0     |   20.0     |   12.0     |   20.0     |   19.0     |   26.5     |
|   3   |   16.0     |   36.0     |   15.5     |   35.5     |   35.0     |   61.5     |
|   4   |   20.0     |   56.0     |   19.5     |   55.0     |   55.0     |  116.5     |
|   5   |   24.0     |   80.0     |   23.0     |   78.0     |   76.0     |  192.5     |
|   6   |   28.0     |  108.0     |   26.0     |  104.0     |   27.5     |  120.0     |
|   7   |   32.0     |  140.0     |   29.0     |  133.0     |   31.0     |  151.0     |
|   8   |   36.0     |  176.0     |   31.0     |  164.0     |   34.0     |  185.0     |
|   9   |   38.0     |  214.0     |   28.0     |  192.0     |   35.0     |  220.0     |
|  10   |   36.0     |  250.0     |   23.0     |  215.0     |   33.0     |  253.0     |
|  11   |   35.0     |  285.0     |   17.0     |  232.0     |   32.0     |  285.0     |
|  12   |   35.0     |  320.0     |   14.0     |  246.0     |   31.0     |  316.0     |
|  13   |   34.0     |  354.0     |   12.0     |  258.0     |   30.0     |  346.0     |
|  14   |   32.0     |  386.0     |   12.0     |  270.0     |   32.0     |  378.0     |
|  15   |   28.0     |  414.0     |     -      |     -      |     -      |     -      |
|  16   |   24.0     |  438.0     |     -      |     -      |     -      |     -      |
|  17   |   18.0     |  456.0     |     -      |     -      |     -      |     -      |
|  18   |   12.0     |  468.0     |     -      |     -      |     -      |     -      |
|  19   |    6.0     |  474.0     |     -      |     -      |     -      |     -      |
|  20   |    3.0     |  477.0     |     -      |     -      |     -      |     -      |
|  21   |    1.5     |  478.5     |     -      |     -      |     -      |     -      |
|  22   |    1.0     |  479.5     |     -      |     -      |     -      |     -      |
|  23   |    0.3     |  479.8     |     -      |     -      |     -      |     -      |
|  24   |    0.2     |  480.0     |     -      |     -      |     -      |     -      |
+-------+------------+------------+------------+------------+------------+------------+
```

### 1. Classical EVM Computation at Month 14
- $PV(14) = \$386.0\text{M}$
- $EV(14) = \$270.0\text{M}$
- $AC(14) = \$378.0\text{M}$

Classical Cost & Schedule Metrics:
$$CV = EV - AC = 270.0 - 378.0 = -\$108.0\text{M}$$
$$CPI = \frac{EV}{AC} = \frac{270.0}{378.0} = 0.7143$$
$$SV = EV - PV = 270.0 - 386.0 = -\$116.0\text{M}$$
$$SPI = \frac{EV}{PV} = \frac{270.0}{386.0} = 0.6995$$

### 2. Earned Schedule ($ES$) Analytics at Month 14
To determine Earned Schedule, locate where cumulative $EV = \$270.0\text{M}$ falls on the cumulative $PV$ schedule.
- $PV_{10} = \$250.0\text{M}$
- $PV_{11} = \$285.0\text{M}$
Because $PV_{10} \le 270.0 < PV_{11}$, the integer time index is $C = 10$.
Now interpolate the fractional month $I$:
$$I = \frac{EV - PV_{10}}{PV_{11} - PV_{10}} = \frac{270.0 - 250.0}{285.0 - 250.0} = \frac{20.0}{35.0} = 0.5714\text{ months}$$
$$ES = C + I = 10.0 + 0.5714 = 10.5714\text{ months}$$

Time-based Schedule Variance ($SV_t$):
$$SV_t = ES - AT = 10.5714 - 14.0000 = -3.4286\text{ months}$$
The project is precisely 3.43 months behind schedule in calendar time.

Time-based Schedule Performance Index ($SPI_t$):
$$SPI_t = \frac{ES}{AT} = \frac{10.5714}{14.0000} = 0.7551$$

Independent Time Forecasts at Completion ($EAC_t$):
1. *Standard Time Forecast:*
   $$EAC_t = \frac{PD}{SPI_t} = \frac{24.0}{0.7551} = 31.78\text{ months}$$
   Forecasted completion delay: $31.78 - 24.0 = +7.78\text{ months}$.
2. *Composite Forecast (accounting for labor inefficiency / low CPI):*
   $$EAC_{t,comp} = AT + \frac{PD - ES}{SPI_t \cdot CPI} = 14.0 + \frac{24.0 - 10.5714}{0.7551 \cdot 0.7143} = 14.0 + \frac{13.4286}{0.5394} = 14.0 + 24.90 = 38.90\text{ months}$$
   The severe cost inefficiency signals catastrophic out-of-sequence disruption; if uncorrected, project completion slips by $38.90 - 24.0 = +14.90\text{ months}$.

To-Complete Schedule Performance Index ($TSPI$):
To finish on the original target $PD = 24.0\text{ months}$:
$$TSPI_{PD} = \frac{PD - ES}{PD - AT} = \frac{24.0 - 10.5714}{24.0 - 14.0} = \frac{13.4286}{10.0} = 1.3429$$
$TSPI = 1.34$ is impossible in heavy power plant construction. The owner and contractor must re-baseline to an extended commercial operation date.

### 3. Critical Chain Feeding Buffer Penetration Analysis
Focus on the GSU Transformer Feeding Chain:
- Feeder Chain Path: PO Execution (4 mos) $\to$ Core/Coil Assembly (6 mos) $\to$ FAT & Testing (2 mos) $\to$ Ocean Transit & Heavy Haul (2 mos).
- Total Feeder Chain Median Duration: $\sum m_j = 14\text{ months}$.
- Initial Feeding Buffer sized via RSEM: $FB_{initial} = 3.5\text{ months}$ (105 days).
- Feeder Chain progress at Month 14: Core/coil finished; FAT stalled. Completed duration $= 10\text{ months}$.
- Chain Completion Ratio ($CCR$):
  $$CCR = \frac{10.0}{14.0} = 0.7143 \quad (71.4\%)$$
- Bushing tooling defect delayed FAT by $2.6\text{ months}$ (78 days).
- Buffer Penetration Ratio ($BPR$):
  $$BPR = \frac{2.60}{3.50} = 0.7429 \quad (74.3\%)$$
- Evaluate Fever Chart Thresholds at $CCR = 0.7143$:
  $$BPR_{warn} = \frac{1}{3} + \frac{1}{3} \cdot (0.7143) = 0.3333 + 0.2381 = 0.5714 \quad (57.1\%)$$
  $$BPR_{crit} = \frac{2}{3} + \frac{1}{3} \cdot (0.7143) = 0.6667 + 0.2381 = 0.9048 \quad (90.5\%)$$
- Since $BPR_{warn} (0.5714) < BPR (0.7429) < BPR_{crit} (0.9048)$, the transformer feeding buffer is in the **Yellow Zone**.
- *Mandatory Action:* Issue commercial cure notice to bushing fabricator; reserve Antonov An-124 heavy-lift air cargo transport for completed bushings directly from factory to transformer integration facility; prepare crane reservation extension at port of entry.

### 4. Probabilistic Monte Carlo SRA Results (10,000 Iterations)
A Monte Carlo SRA performed across all remaining procurement packages and civil/piping/electrical construction packages yields:
- Deterministic CPM Completion Date: Month 26.5.
- Probabilistic SRA Distribution:
  - $P_{10}$: 27.2 months
  - $P_{50}$: 31.4 months
  - $P_{80}$: 34.2 months
  - $P_{90}$: 36.1 months
- Deterministic CPM underestimates the expected completion ($P_{50}$) by $31.4 - 26.5 = 4.9\text{ months}$ due to merge event bias across the turbine, transformer, and high-pressure steam systems.
- Contractual Schedule Risk Reserve ($SRR$) required to hit an 80% confidence commitment:
  $$SRR_{80} = P_{80} - P_{50} = 34.2 - 31.4 = 2.8\text{ months}$$
  $$SRR_{total} = P_{80} - T_{deterministic} = 34.2 - 26.5 = 7.7\text{ months}$$
- Activity Criticality Indices ($CI$):
  - GSU Transformer Package: $CI = 78.4\%$
  - P91 Main Steam Piping Spools: $CI = 82.1\%$
  - Gas Turbine Generator Set: $CI = 34.6\%$ (has deterministic float, but becomes critical in 34.6% of runs).

### 5. Longitudinal Metric Behavior: Classical EVM vs. Earned Schedule Across the Project Lifecycle
To observe the mathematical breakdown of classical EVM and verify the stability of Earned Schedule, the table below traces the project from early engineering through actual delayed completion at Month 31.8:

```text
+-------+---------+---------+---------+-----------+-------------+----------+------------+-----------+
| Month | PV Cum  | EV Cum  | AC Cum  | Classic SV| Classic SPI | ES       | Time SV_t  | Time SPI_t|
| (AT)  | ($M)    | ($M)    | ($M)    | ($M)      | (EV / PV)   | (months) | (ES - AT)  | (ES / AT) |
+-------+---------+---------+---------+-----------+-------------+----------+------------+-----------+
|   6   |  108.0  |  104.0  |  120.0  |   -$4.0   |   0.9630    |  5.857   | -0.143 mos |  0.9762   |
|  10   |  250.0  |  215.0  |  253.0  |  -$35.0   |   0.8600    |  9.026   | -0.974 mos |  0.9026   |
|  14   |  386.0  |  270.0  |  378.0  | -$116.0   |   0.6995    | 10.571   | -3.429 mos |  0.7551   |
|  18   |  468.0  |  338.0  |  460.0  | -$130.0   |   0.7222    | 12.529   | -5.471 mos |  0.6961   |
|  22   |  479.5  |  414.0  |  565.0  |  -$65.5   |   0.8634    | 15.000   | -7.000 mos |  0.6818   |
|  24   |  480.0  |  450.0  |  620.0  |  -$30.0   |   0.9375    | 16.667   | -7.333 mos |  0.6945   |
|  28   |  480.0  |  472.0  |  695.0  |   -$8.0   |   0.9833    | 18.667   | -9.333 mos |  0.6667   |
| 31.8  |  480.0  |  480.0  |  755.0  |   $0.0    |   1.0000    | 24.000   | -7.780 mos |  0.7547   |
+-------+---------+---------+---------+-----------+-------------+----------+------------+-----------+
```

Between Month 18 and Month 24, classical $SPI$ rebounded from 0.7222 to 0.9375, and classical $SV$ shrank from $-\$130\text{M}$ to $-\$30\text{M}$. Under classical EVMS reporting, an uninformed project committee would assume the contractor had recovered schedule velocity. In reality, physical progress had ground to a near halt; $SV_t$ worsened from $-5.47\text{ months}$ to $-7.33\text{ months}$, and $SPI_t$ dropped to a project low of 0.6818. At final turnover (Month 31.8), classical EVM declared $SV = \$0$ and $SPI = 1.0000$, while Earned Schedule preserved the true historical record: the project finished $7.78\text{ months}$ late with an overall time performance index of $SPI_t = 0.7547$.

---

## 7. Code, query, or calculation

The following self-contained Python script calculates Earned Schedule ($ES$, $SV_t$, $SPI_t$), computes classical vs. time-based metrics, models critical chain buffer consumption, and runs a vectorized Monte Carlo Schedule Risk Analysis on converging procurement feeding chains.

```python
"""Earned Schedule, Feeding Buffer, and Probabilistic SRA Engine."""
import numpy as np

# 1. Performance Measurement Baseline (Monthly Cumulative PV for 24 months)
pv_curve = np.array([
    0.0, 8.0, 20.0, 36.0, 56.0, 80.0, 108.0, 140.0, 176.0, 214.0,
    250.0, 285.0, 320.0, 354.0, 386.0, 414.0, 438.0, 456.0, 468.0, 474.0,
    477.0, 478.5, 479.5, 479.8, 480.0
], dtype=float)
bac, pd = pv_curve[-1], len(pv_curve) - 1

# 2. Actuals at Status Date Month AT = 14
at, ev, ac = 14.0, 270.0, 378.0

# Earned Schedule (ES) calculation via linear interpolation
c = int(np.where(pv_curve <= ev)[0][-1])
i = (ev - pv_curve[c]) / (pv_curve[c + 1] - pv_curve[c]) if c < pd else 0.0
es = c + i
sv_t, spi_t = es - at, es / at
cpi, spi_classic = ev / ac, ev / pv_curve[int(at)]
eac_t_single = pd / spi_t
eac_t_comp = at + (pd - es) / (spi_t * cpi)

print(f"Classic EVM: SV=${ev - pv_curve[int(at)]:.1f}M, SPI={spi_classic:.4f}, CPI={cpi:.4f}")
print(f"Earned Schedule: ES={es:.4f}m, SV_t={sv_t:.4f}m, SPI_t={spi_t:.4f}")
print(f"Completion Forecasts: EAC_t(Single)={eac_t_single:.2f}m, EAC_t(Comp)={eac_t_comp:.2f}m")

# 3. Critical Chain Feeding Buffer Monitoring
fb_initial, slip_days, total_chain_days, completed_days = 105.0, 78.0, 420.0, 300.0
bpr = slip_days / fb_initial
ccr = completed_days / total_chain_days
warn_boundary = (1.0 / 3.0) + (1.0 / 3.0) * ccr
crit_boundary = (2.0 / 3.0) + (1.0 / 3.0) * ccr
status = "RED" if bpr >= crit_boundary else ("YELLOW" if bpr >= warn_boundary else "GREEN")
print(f"Feeding Buffer: CCR={ccr:.1%}, BPR={bpr:.1%}, Status={status}")

# 4. Probabilistic Monte Carlo SRA for Converging ETO Paths (N = 10,000)
np.random.seed(42)
n_iter = 10000

def pert_sample(a, m, b, n):
    alpha = 1.0 + 4.0 * (m - a) / (b - a)
    beta = 1.0 + 4.0 * (b - m) / (b - a)
    return a + (b - a) * np.random.beta(alpha, beta, n)

# Remaining duration distributions for converging paths to Power Block Commercial Operation
path_transformer = pert_sample(4.0, 6.0, 12.0, n_iter)  # Bushing fix, FAT, ship
path_piping = pert_sample(5.0, 7.0, 14.0, n_iter)       # P91 spool weld, coat, ship
path_turbine_skid = pert_sample(3.0, 4.5, 9.0, n_iter)  # Skid alignment, pack
site_erection = pert_sample(8.0, 10.0, 15.0, n_iter)    # Final erection and hot comm

merge_milestone = np.maximum.reduce([path_transformer, path_piping, path_turbine_skid])
total_remaining = merge_milestone + site_erection
forecast_completion = at + total_remaining

p50, p80 = np.percentile(forecast_completion, 50), np.percentile(forecast_completion, 80)
ci_transformer = np.mean(path_transformer == merge_milestone) * 100.0
ci_piping = np.mean(path_piping == merge_milestone) * 100.0

print(f"Probabilistic SRA: P50={p50:.2f}m, P80={p80:.2f}m, Schedule Reserve (P80-P50)={p80-p50:.2f}m")
print(f"Criticality Indices: Piping={ci_piping:.1f}%, Transformer={ci_transformer:.1f}%")
```

### Expected Output
```text
Classic EVM: SV=$-116.0M, SPI=0.6995, CPI=0.7143
Earned Schedule: ES=10.5714m, SV_t=-3.4286m, SPI_t=0.7551
Completion Forecasts: EAC_t(Single)=31.78m, EAC_t(Comp)=38.90m
Feeding Buffer: CCR=71.4%, BPR=74.3%, Status=YELLOW
Probabilistic SRA: P50=32.59m, P80=34.29m, Schedule Reserve (P80-P50)=1.70m
Criticality Indices: Piping=68.0%, Transformer=29.0%
```

---

## 8. Data, standards, and regulatory context

Earned value and schedule risk analytics operate under strict international engineering standards, contractual legal frameworks, and regulatory mandates.

```text
+--------------------------------------------------------------------------------------------------+
|                   INTERNATIONAL STANDARDS AND REGULATORY MANDATES                                |
+----------------------+--------------------+------------------------------------------------------+
| Standard / Authority | Formal Reference   | Binding Requirement / Procurement Constraint         |
+----------------------+--------------------+------------------------------------------------------+
| ANSI / EIA           | ANSI/EIA-748-D     | 32 Guidelines for EVMS. Guideline 10 (Material       |
|                      | (R2019)            | Accounting), Guideline 21 (Progress Tracking),       |
|                      |                    | Guideline 23 (Variance Analysis), Guideline 27 (EAC).|
+----------------------+--------------------+------------------------------------------------------+
| ISO                  | ISO 21508:2018     | Standardizes Earned Value Management in project and  |
|                      |                    | programme management, explicitly adopting Earned     |
|                      |                    | Schedule (ES) formulas for time-based forecasting.   |
+----------------------+--------------------+------------------------------------------------------+
| US Dept of Defense   | DCMA EVMSIG        | Establishes compliance review protocols for 32       |
| (DoD) / DCMA         | (2023)             | guidelines. Mandates strict material reconciliation; |
|                      |                    | prohibits paper earning upon subcontractor PO issue. |
+----------------------+--------------------+------------------------------------------------------+
| US Dept of Energy    | DOE Order 413.3B   | Mandates certified EVMS on capital asset projects    |
| (DOE)                | (Change 6, 2021)   | exceeding $50M. Imposes formal Baseline Change       |
|                      |                    | Proposals (BCPs) when SPI_t or CPI deviate > 10%.    |
+----------------------+--------------------+------------------------------------------------------+
| GAO                  | GAO-20-195G        | U.S. Government Accountability Office Cost           |
|                      | (Cost Guide)       | Estimating and Assessment Guide; establishes SRA and |
|                      | GAO-16-89G (Sched) | Monte Carlo completion confidence rules (P80 standard)|
+----------------------+--------------------+------------------------------------------------------+
| FERC / State PUCs    | 18 CFR Part 101    | Prudency audits of utility capital expenditures.     |
|                      | Uniform System     | Disallows AFUDC (Allowance for Funds Used During     |
|                      | of Accounts        | Construction) capital recovery caused by unmanaged   |
|                      |                    | contractor procurement delays.                       |
+----------------------+--------------------+------------------------------------------------------+
```

### ANSI/EIA-748 Guidelines on Material Management
The 32 Guidelines of ANSI/EIA-748 contain specific provisions governing materials and subcontracts:
- **Guideline 10 (Accurate Material Cost Accounting):** Material costs must be accrued in the same accounting period in which Earned Value is recorded. If an EPC contractor claims $EV$ for receiving piping spools, the corresponding Actual Cost liability ($AC$) must be booked simultaneously—even if the vendor's formal invoice has not completed processing. Recording $EV$ while deferring $AC$ produces artificial, fraudulent $CPI$ spikes.
- **Guideline 21 (Material Progress Measurement):** Full value cannot be claimed for material purchases simply because cash left the owner's bank account. Value must be earned against objective, verified physical events (e.g., mill test acceptance, FAT completion, site receipt inspection).
- **Guideline 23 (Variance Thresholds):** Control Account Managers must generate formal Variance Analysis Reports (VARs) whenever monthly or cumulative cost or schedule variances exceed contractual thresholds (typically $\pm 5\%$ or $\pm 10\%$). Material price variances (MPV) must be isolated from material usage variances (MUV) and delivery-driven labor inefficiency variances.
- **Guideline 27 (Updating EAC):** The contractor must routinely update the Estimate at Completion ($EAC$) for both cost and schedule. Relying on outdated deterministic CPM finish dates when Monte Carlo SRA shows an 80% probability of delay constitutes non-compliance under DCMA surveillance.

### DCMA 14-Point Schedule Assessment Applied to Procurement
Under DCMA surveillance and DOE Order 413.3B oversight, the Integrated Master Schedule (IMS) must pass the 14-Point Schedule Assessment before EVMS progress is certified. Procurement and supply chain logic are the primary sources of DCMA audit non-compliance:

```text
+--------------------------------------------------------------------------------------------------+
|                   DCMA 14-POINT ASSESSMENT: PROCUREMENT AUDIT COUPLING                           |
+-----+-----------------------+------------------+-------------------------------------------------+
| #   | Metric Name           | Passing Threshold| Supply Chain Failure Mechanism & Audit Focus    |
+-----+-----------------------+------------------+-------------------------------------------------+
| 01  | Missing Logic         | 0% (Max 5% total)| Purchase orders floating without successor ties |
|     |                       |                  | to construction work packages (CWPs).           |
| 02  | Leads (Negative Lag)  | Exactly 0%       | Schedulers using negative lag to simulate       |
|     |                       |                  | pre-delivery staging; strictly prohibited.      |
| 03  | Lags (Positive Lag)   | <= 5% of links   | Hiding factory lead time in positive FS lags    |
|     |                       |                  | instead of explicit procurement activities.     |
| 05  | Hard Constraints      | Exactly 0%       | Placing "Must Finish On" dates on equipment     |
|     |                       |                  | deliveries, which artificially suppresses float.|
| 06  | High Float (>44 days) | <= 5% of tasks   | Catalog items floating detached from field work |
|     |                       |                  | giving false impression of schedule safety.     |
| 07  | Negative Float        | Exactly 0 tasks  | Supply delays pushing critical path beyond      |
|     |                       |                  | contractual Commercial Operation Date (COD).    |
| 08  | High Duration (>44 d) | <= 5% of tasks   | ETO equipment modeled as a single 18-month bar  |
|     |                       |                  | without intermediate milestone value gates.     |
| 12  | Critical Path Test    | Pass (Integrity) | Injecting 10-day delay on primary equipment     |
|     |                       |                  | must delay project completion by exactly 10 d.  |
| 13  | Critical Path Length  | CPLI >= 0.95     | Ratio: (Remaining Duration + Total Float) /     |
|     | Index (CPLI)          | (Target 1.00)    | Remaining Duration. CPLI < 0.95 = broken path.  |
| 14  | Baseline Execution    | BEI >= 0.95      | Ratio: Actual Completed Tasks / Planned Baseline|
|     | Index (BEI)           |                  | Tasks. BEI < 0.90 indicates severe supply drag. |
+-----+-----------------------+------------------+-------------------------------------------------+
```

---

## 9. Industry practice and case evidence

Public audits, project control histories, and dispute proceedings document how schedule and earned value analytics make or break capital investments.

```text
+--------------------------------------------------------------------------------------------------+
|                   SUMMARY OF CAPITAL MEGAPROJECT SCHEDULE AND SUPPLY FAILURES                   |
+----------------------+---------------+-------------------+---------------------------------------+
| Project Name         | Sector        | Reported Impact   | Primary Supply-Schedule Mechanism     |
+----------------------+---------------+-------------------+---------------------------------------+
| Vogtle Units 3 & 4   | Nuclear Power | 7-year delay,     | Defective pre-fabricated structural   |
| (Georgia, USA)       | Generation    | Cost escalated    | modules; out-of-sequence delivery;    |
|                      |               | from $14B to >$35B| trade stacking; classical EVM mask.   |
+----------------------+---------------+-------------------+---------------------------------------+
| Gorgon LNG           | LNG Export    | 2-year delay,     | Modular fabrication yard slips in     |
| (Barrow Island, AUS) | Facility      | Cost escalated    | Asia; island biosecurity bottleneck;  |
|                      |               | from $37B to $54B | severe feeding buffer depletion.      |
+----------------------+---------------+-------------------+---------------------------------------+
| Kemper County IGCC   | Clean Coal /  | Abandoned syngas, | Exotic metallurgy piping spools late; |
| (Mississippi, USA)   | Power Block   | $7.5B expenditure | CPI dropped below 0.60 due to rework; |
|                      |               | unrecovered       | unmanaged merge event bias.           |
+----------------------+---------------+-------------------+---------------------------------------+
```

### Case 1: Vogtle Units 3 and 4 Nuclear Expansion
The construction of two Westinghouse AP1000 nuclear reactors at Plant Vogtle in Waynesboro, Georgia, represents one of the most thoroughly audited capital project controls failures in modern history (Georgia Public Service Commission, Docket 29849). 
- *The Supply Mechanism:* The AP1000 design relied heavily on modular construction. Giant pre-fabricated sub-modules (such as the CA20 structural module weighing over 900 metric tons) were manufactured offsite at a fabrication yard in Lake Charles, Louisiana. The sub-modules were shipped to the site with missing structural welds, incomplete Quality Assurance (N-stamp) paperwork, and dimensional distortions.
- *The Controls Breakdown:* Project controls tracked progress using traditional milestone percentages, claiming credit upon module shipment. When the modules arrived on site, they could not be set onto foundations. Craft labor was redirected to out-of-sequence structural re-welding and field re-engineering inside confined containment spaces. 
- *The EVM Failure:* Classic $SPI$ remained above 0.85 for months because high-dollar material procurements continued to book Earned Value upon PO placement and factory exit. However, physical field installation stalled. Trade stacking diluted journeyman electrician and pipefitter productivity to below 40% of baseline. The resulting labor burn collapsed contractor $CPI$ to below 0.65, culminating in the Chapter 11 bankruptcy of Westinghouse Electric Company in 2017. Had Earned Schedule ($ES$) and strict physical gating (Guideline 21) been enforced, the 36-month critical path delay would have been visible three years earlier.

### Case 2: Gorgon LNG Modular Construction
Chevron's Gorgon LNG project on Barrow Island, Western Australia, required the construction of three 5.2 MTPA LNG liquefaction trains using offshore modularization.
- *The Supply Mechanism:* Over 250 heavy modules were fabricated across multiple yards in Indonesia, South Korea, and China. Logistics were tightly constrained by strict environmental biosecurity regulations on Barrow Island (clearing incoming cargo for invasive species).
- *The Schedule Impact:* Delays in overseas module fabrication coincided with heavy weather seasons in the Indian Ocean, triggering a cascading depletion of logistics feeding buffers. The merge point at the train foundation required simultaneous availability of structural skids, piping interconnect spools, and electrical switchgear modules.
- *The Resolution:* Rather than allowing trades to work out of sequence and destroy field productivity, project leadership halted site assembly until complete kits (materials, spool pieces, bolt packages, and QA dossiers) were fully staged in mainland Australian laydown yards. While this incurred holding and double-handling costs, it protected field installation labor efficiency, avoiding the catastrophic $CPI$ collapse suffered at Vogtle.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign / Red Flag | Root Mechanism | Operational Consequence | Mitigation / Preventive Protocol |
|---|---|---|---|---|---|
| 1 | **Late-Stage EVM Mirage** | Classic $SPI$ steadily improves from 0.75 to 0.95 while project milestones continue to slip. | Mathematical convergence: $EV \to BAC$ forces $SV \to 0$ and $SPI \to 1.0$ as $t > PD$. | Management falsely believes schedule recovery is underway; fails to authorize overtime or expediting. | Mandate Earned Schedule ($ES$, $SV_t$, $SPI_t$) reporting alongside classical EVM across all control accounts. |
| 2 | **Paper Progress Earning** | Sudden spike in $EV$ without physical equipment visible in staging laydown. | Earning 100% of material budget upon commercial PO placement or letter of credit issuance. | Unearned progress masks factory delays; contractor receives progress billings for nonexistent physical work. | Enforce ANSI/EIA-748 Guideline 21: link progress strictly to objective, verified vendor inspection gates. |
| 3 | **Merge Event CPM Blind Spot** | Schedule shows 15 days of total float, but critical path shifts weekly between parallel supply chains. | Deterministic CPM ignores variance: $\mathbb{E}[\max(T_i)] > \max(\mathbb{E}[T_i])$ at merge nodes. | Parallel procurement chains cause unexpected milestone slips; buffers exhaust instantly upon arrival. | Perform monthly Monte Carlo Schedule Risk Analysis (SRA) to track Activity Criticality Index ($CI$). |
| 4 | **Uncontrolled Buffer Sandbagging** | Tier 1 vendors report zero delay, then suddenly declare a 12-week schedule slip four weeks before dispatch. | Vendors conceal internal schedule slippage within their unmonitored internal contingencies. | Feeding buffer vanishes in a single reporting period without prior warning. | Embed resident quality/progress expediters in vendor manufacturing plants with direct access to shop floor routers. |
| 5 | **Out-of-Sequence Labor Stacking** | Man-hours billed per week soar while physical work package completion stalls; craft density > 1 worker per 10 m². | Field supervisors divert crafts to non-critical areas to avoid paying idle standby costs. | Labor efficiency drops by 30–50%; severe rework required when missing spools/skids arrive; $CPI$ collapses. | Restrict work package execution until a formal Material Readiness Verification (MRV) confirms 100% kitting. |
| 6 | **Decoupled ERP and Schedule Logic** | Procurement ERP promises delivery on June 1; Primavera P6 schedule assumes delivery on April 15. | Material management systems and project scheduling software maintain disconnected data schemas. | Field construction scheduled based on obsolete delivery assumptions; craft crews mobilized to empty pads. | Integrate ERP Purchase Order delivery dates dynamically into P6 schedule logic via automated API bridges. |
| 7 | **Demurrage Cascade** | Port and laydown storage invoices surge 300%; heavy transport trailers sit loaded for weeks. | Oversized ETO equipment delivered to site before civil foundations or soleplates are poured and cured. | Severe double handling, trailer detention fees ($5,000/day/trailer$), and risk of asset weather damage. | Implement "Just-in-Time-to-Foundation" delivery protocols tied to verified civil foundation signoff gates. |
| 8 | **Phantom Receipt Inspection** | Materials booked as "On Site" in CMMS, but field crews report missing parts during work-package breakout. | Materials delivered to laydown without performing line-item Over, Short & Damaged (OS&D) receipt inspection. | Defective or missing sub-components discovered only when crane has suspended the main assembly. | Withhold the final 20% of material $EV$ until a signed OS&D technical inspection report is logged in the CMMS. |
| 9 | **Baseline Reset Churn** | Project baseline version changes four times in twelve months; $SV$ repeatedly resets to zero. | CAMs issue frequent Baseline Change Proposals (BCPs) to absorb delivery slips without root-cause remediation. | Historical variance trends are erased; project owner loses visibility into cumulative contractor delay. | Enforce strict customer governance on Performance Measurement Baseline (PMB) resets; retain Over-Target Baselines. |
| 10 | **Sub-Tier Visibility Blindness** | Tier 1 OEM reports 100% on schedule, but Tier 2 raw ingot forge suffers catastrophic kiln failure. | Supply tracking stops at Tier 1 prime contractor without auditing Tier 2/3 critical path components. | Sudden, unrecoverable 6-to-12-month delivery delays when sub-tier component fails testing. | Require sub-tier supplier traceability and milestone reporting for all high-criticality forgings, castings, and chips. |
| 11 | **Negative Float Hiding** | Schedule reports zero total float, but contractual finish milestone is locked with a hard mandatory constraint. | Schedulers apply artificial late-finish constraints to prevent negative float from appearing in executive reports. | True schedule delay is hidden behind artificial constraint flags; downstream logic is corrupted. | Enforce DCMA 14-Point schedule audit metrics: prohibit hard constraints and monitor true total float. |

---

## 11. Assumptions and limits

| # | Assumption | Why It Is Needed | What Breaks It | How to Test / Validate |
|---|---|---|---|---|
| 1 | **Monotonicity of Planned Value** | Required for linear interpolation in Earned Schedule: $PV(t)$ must be a non-decreasing function. | De-scoping, negative baseline changes, or major retroactive scope removals. | Verify $PV_{t+1} \ge PV_t$ across all periods; isolate scope changes into separate baseline revisions. |
| 2 | **Linear Interpolation of ES** | Assumes physical work within a monthly reporting period accrues linearly between discrete $PV$ points. | Heavy lump-sum milestone payments or single-day progress lump earnings within the month. | Reduce reporting period bucket size (e.g., weekly instead of monthly) to minimize interpolation distortion. |
| 3 | **Statistical Independence of Supply Paths** | Required for standard RSEM feeding buffer sizing and basic Monte Carlo simulation. | Common-cause shocks: global ocean freight disruptions, regional port strikes, broad raw material tariffs. | Model correlated input distributions using copulas or covariance matrices in Monte Carlo SRA engines. |
| 4 | **Three-Point PERT Distribution Validity** | Provides the mathematical basis for lead-time variance parameterization in probabilistic scheduling. | Cognitive bias: vendor optimism, anchoring on contract promise dates, sandbagging. | Calibrate three-point estimates against historical enterprise lead-time databases using empirical distributions. |
| 5 | **Stable Work Breakdown Structure** | Assumes the project scope and control accounts remain constant over the performance measurement baseline. | Uncontrolled scope creep, frequent engineering design changes, expanding project boundaries. | Audit baseline change logs; freeze WBS architecture at Final Investment Decision (FID). |
| 6 | **Fungibility of Craft Labor** | Assumes diverted workers can generate productive progress on secondary construction tasks. | Specialized union craft jurisdictional boundaries; lack of secondary approved engineering packages. | Cross-reference craft skill matrices and verify ready-to-work status of alternative Work Packages. |
| 7 | **Linearity of Site Indirect Burn** | Assumes fixed site overhead ($AC$) accrues at a constant daily burn rate during delivery delays. | Non-linear demobilization/remobilization costs; escalating equipment rental penalty tiers. | Re-forecast indirect accounts dynamically based on equipment lease contracts and staffing plans. |
| 8 | **Non-Adversarial Contractor Reporting** | Assumes contractor reports actual hours and costs accurately without front-loading or shifting charges. | Claims posturing: contractor burying commercial dispute hours in unmonitored indirect control accounts. | Conduct regular DCMA-style EVMS surveillance audits of contractor payroll and ledger entries. |
| 9 | **Resource Feasibility of the Schedule** | Assumes CPM schedule network can be executed without exceeding regional craft labor availability. | Local craft labor shortages (e.g., certified P91 pipe welders or high-voltage cable splicers). | Perform resource-constrained critical path leveling before running Monte Carlo schedule risk simulations. |

---

## 12. Exercises

### Exercise 1: Proving Classical SPI Breakdown and Calculating Earned Schedule
A capital compressor station project has a Budget at Completion $BAC = \$120\text{M}$ and a Planned Duration $PD = 12\text{ months}$. The baseline cumulative $PV$ schedule is:
- $PV = [0, 5, 12, 22, 35, 50, 68, 85, 98, 108, 115, 118, 120]\text{ \$M}$ for months $t = 0, 1, \dots, 12$.
Due to gas turbine delivery delays, at Month $AT = 12.0$, actual performance is:
- $EV = \$98.0\text{M}$, $AC = \$130.0\text{M}$.
- **Deliverable:**
  1. Calculate classic $SV$, classic $SPI$, $CV$, and $CPI$ at Month 12.
  2. Prove mathematically what classic $SPI$ would report if the project finally completes all scope at Month $AT = 16.0$ ($EV = \$120\text{M}$).
  3. Calculate Earned Schedule ($ES$), $SV_t$, $SPI_t$, and single-factor $EAC_t$ at Month 12.
- **Acceptance Criteria:** Correctly interpolate fractional time $I$; provide step-by-step arithmetic; state units explicitly.
- **Estimated Time:** 25 minutes.

### Exercise 2: Contractor Cost Escalation from Idle Labor and Trade Stacking
An EPC piping contractor is scheduled to install high-pressure pipe spools in a turbine bay.
- Budgeted Direct Labor: $H_{bud} = 10,000\text{ man-hours}$ at a billing rate of $R = \$95/\text{hour}$.
- Crew Size: 25 craft pipefitters and welders working 40 hours per week.
- Fixed site indirect costs: $C_{indir} = \$3,500/\text{day}$ (7 days/week).
The pipe spools are delayed at the coating facility by 4 weeks (28 calendar days, 20 working days).
- **Scenario A:** The contractor holds the 25-man crew on standby, paying full direct wages with zero physical progress ($\Delta EV = 0$).
- **Scenario B:** The contractor diverts the 25-man crew to work inside an electrical equipment room alongside an existing 25-man electrical crew. The physical workspace was designed for an optimal density of 20 workers total. Due to trade stacking and congestion:
  $$\delta_{stack} = 0.15, \quad \delta_{cong} = 0.20, \quad \delta_{dilution} = 0.10$$
  When the spools arrive, 1,200 hours of electrical work must be reworked due to physical interference ($\beta_{rework} = 0.12$).
- **Deliverable:** Calculate the total additional cost ($\Delta AC$) and realized $CPI$ under both scenarios. Determine which operational choice was financially superior.
- **Acceptance Criteria:** Formulate compounding productivity degradation $\eta$; state all assumptions; present exact dollar figures.
- **Estimated Time:** 30 minutes.

### Exercise 3: Dimensioning Feeding Buffers and Classifying Fever Chart Status
A procurement feeding chain for an offshore platform switchgear module consists of four sequential activities:
1. Approved Engineering Drawings: $m_1 = 3\text{ weeks}, b_1 = 5\text{ weeks}$.
2. Enclosure Fabrication & Assembly: $m_2 = 8\text{ weeks}, b_2 = 12\text{ weeks}$.
3. Breaker Integration & Testing: $m_3 = 6\text{ weeks}, b_3 = 10\text{ weeks}$.
4. Factory Acceptance Testing (FAT): $m_4 = 3\text{ weeks}, b_4 = 6\text{ weeks}$.
- **Deliverable:**
  1. Calculate the aggressive chain duration ($\sum m_i$) and dimension the feeding buffer ($FB_{RSEM}$) using the Root-Square-Error Method.
  2. At status week 14, Activities 1 and 2 are 100% complete, but due to component shortages, Activity 2 finished 3.5 weeks late. Activity 3 has just begun. Calculate the Chain Completion Ratio ($CCR$) and Buffer Penetration Ratio ($BPR$).
  3. Determine the Fever Chart operating zone (Green, Yellow, or Red) and name the mandatory operational intervention.
- **Acceptance Criteria:** Correct mathematical formulation of RSEM and linear fever chart boundary equations.
- **Estimated Time:** 20 minutes.

### Exercise 4: Analytical Merge Event Bias Calculation
Two independent parallel equipment procurement chains merge into a common installation milestone:
- Path 1 (Steam Turbine Rotor): $X_1 \sim \mathcal{N}(\mu_1 = 40\text{ weeks}, \sigma_1^2 = 16\text{ weeks}^2)$.
- Path 2 (Generator Stator): $X_2 \sim \mathcal{N}(\mu_2 = 38\text{ weeks}, \sigma_2^2 = 25\text{ weeks}^2)$.
- **Deliverable:**
  1. State the deterministic CPM expected completion time for the merge milestone.
  2. Using Clark's formula for the maximum of two normal random variables, calculate the true expected completion time $\mathbb{E}[\max(X_1, X_2)]$.
     *(Use values: $\Phi(0.3123) \approx 0.6226$, $\Phi(-0.3123) \approx 0.3774$, $\phi(0.3123) \approx 0.3802$)*.
  3. Quantify the exact merge bias error introduced by deterministic CPM.
- **Acceptance Criteria:** Full algebraic derivation showing substitution into Clark's equation.
- **Estimated Time:** 25 minutes.

### Exercise 5: Critical Analysis of an Contractor Recovery Claim
An EPC contractor is 6 months behind schedule on a combined-cycle power block at Month 18 of a planned 24-month schedule. 
- Baseline Budget: $BAC = \$300\text{M}$.
- Month 18 Actuals: $PV = \$270\text{M}$, $EV = \$210\text{M}$, $AC = \$280\text{M}$.
- Contractor's Monthly Report Statement: *"While our cumulative SPI is 0.778, our monthly SPI for Month 18 was 1.15. We have achieved positive schedule momentum and project completing within 25.5 months."*
- Detailed schedule audit reveals that in Month 18, the contractor placed purchase orders for $\$15\text{M}$ of spare parts and bulk electrical cables, claiming 100% $EV$ upon PO release under an informal internal procedure. No critical path piping or civil milestones advanced.
- **Deliverable:** Write an executive audit critique evaluating the contractor's claim. Calculate $SV_t$, $SPI_t$, $TSPI$, and assess ANSI/EIA-748 Guideline 10 & 21 compliance.
- **Acceptance Criteria:** Professional tone, rigorous EVMS compliance critique, and accurate recalculation of underlying metrics.
- **Estimated Time:** 30 minutes.

### Exercise 6: Designing a Supply-Schedule Early Warning Dashboard
- **Deliverable:** Design the technical specification for an enterprise early-warning data pipeline integrating SAP ERP procurement milestones, Primavera P6 critical path schedules, and Deltek Cobra EVMS cost engines.
- **Acceptance Criteria:** Provide a functional data flow architecture; define table join keys; specify automated exception flags for buffer penetration, negative float, and unvouchered $AC$ accruals.
- **Estimated Time:** 30 minutes.

### Exercise 7: Calculation of Multi-Factor EAC_t with Productivity Drag
Using the West Bay worked example data at Month 14 ($PD = 24.0$, $AT = 14.0$, $ES = 10.5714$, $CPI = 0.7143$, $SPI_t = 0.7551$):
- **Deliverable:**
  1. Compute $EAC_t$ assuming future schedule performance follows the composite factor $PF = 0.8 \cdot SPI_t + 0.2 \cdot CPI$.
  2. Compute the required daily expenditure burn rate if the owner mandates completing the project by Month 28.0.
- **Acceptance Criteria:** Show complete arithmetic and derivation.
- **Estimated Time:** 20 minutes.

### Exercise 8: DCMA 14-Point CPLI and BEI Metric Calculation
A capital LNG compressor station has a planned remaining duration of 180 working days along its primary critical path. Due to long-lead cryogenic valve forging delays, the current total float on the critical path is -25 working days. In the monthly procurement schedule, 40 baseline procurement and kitting milestones were scheduled to be completed by status date; field receipt audit confirms only 28 were actually accomplished.
- **Deliverable:**
  1. Calculate the Critical Path Length Index (CPLI).
  2. Calculate the Baseline Execution Index (BEI).
  3. Determine whether the schedule meets DCMA passing thresholds ($CPLI \ge 0.95$, $BEI \ge 0.95$) and describe the contract implications under US DOE or DoD surveillance.
- **Acceptance Criteria:** State analytical formulas, perform step-by-step arithmetic, and evaluate against DCMA compliance criteria.
- **Estimated Time:** 20 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Proving Classical SPI Breakdown and Calculating Earned Schedule
1. **Classic EVM at Month 12:**
   - $PV(12) = \$120.0\text{M}$, $EV(12) = \$98.0\text{M}$, $AC(12) = \$130.0\text{M}$.
   - $SV = EV - PV = 98.0 - 120.0 = -\$22.0\text{M}$.
   - $SPI = \frac{EV}{PV} = \frac{98.0}{120.0} = 0.8167$.
   - $CV = EV - AC = 98.0 - 130.0 = -\$32.0\text{M}$.
   - $CPI = \frac{EV}{AC} = \frac{98.0}{130.0} = 0.7538$.
2. **Proof of Classic Metric Collapse at Month 16 ($T_{act} = 16$):**
   - At project completion, all scope is finished: $EV(16) = BAC = \$120.0\text{M}$.
   - Since $t \ge PD$, $PV(16) = BAC = \$120.0\text{M}$.
   - $SV(16) = EV(16) - PV(16) = 120.0 - 120.0 = \$0.0\text{M}$.
   - $SPI(16) = \frac{EV(16)}{PV(16)} = \frac{120.0}{120.0} = 1.0000$.
   Despite finishing 4 months (33.3%) late, classic $SPI$ falsely reports 1.0000.
3. **Earned Schedule ($ES$) Calculation at Month 12:**
   - Find $C$ where $PV_C \le EV(12) < PV_{C+1}$:
     Looking at $PV$: $PV_7 = \$85.0\text{M}$, $PV_8 = \$98.0\text{M}$.
     Since $EV = 98.0\text{M}$ exactly matches $PV_8$, $C = 8$ and $I = 0.0$.
     $$ES = 8.000\text{ months}$$
   - $SV_t = ES - AT = 8.000 - 12.000 = -4.000\text{ months}$. (The project is precisely 4 months behind schedule).
   - $SPI_t = \frac{ES}{AT} = \frac{8.000}{12.000} = 0.6667$.
   - Single-factor Time Forecast:
     $$EAC_t = \frac{PD}{SPI_t} = \frac{12.0}{0.6667} = 18.00\text{ months}$$
     Notice that while classic $SPI$ was 0.8167 (predicting $12 / 0.8167 = 14.7\text{ months}$), Earned Schedule correctly detects that the project has only achieved 8 months of work in 12 months of elapsed time, forecasting an 18-month completion date.

### Answer to Exercise 2: Contractor Cost Escalation from Idle Labor and Trade Stacking
1. **Scenario A: Direct Labor Idling:**
   - Standby Direct Labor Cost:
     $$\Delta AC_{direct} = 25\text{ workers} \times (4\text{ weeks} \times 40\text{ hrs/wk}) \times \$95/\text{hr} = 25 \times 160 \times 95 = \$380,000$$
   - Fixed Indirect Cost:
     $$\Delta AC_{indir} = 28\text{ days} \times \$3,500/\text{day} = \$98,000$$
   - Total Additional Cost: $\Delta AC_A = \$380,000 + \$98,000 = \$478,000$.
   - Progress earned during delay: $\Delta EV = \$0$.
   - Baseline budget for work: $EV_{bud} = 10,000 \times 95 = \$950,000$.
   - Total Cost to complete work package: $AC_A = \$950,000 + \$478,000 = \$1,428,000$.
   - $CPI_A = \frac{950,000}{1,428,000} = 0.6653$.

2. **Scenario B: Out-of-Sequence Diversion and Stacking:**
   - Degraded productivity factor $\eta_{actual}$:
     $$\eta_{actual} = 1.0 \times (1 - 0.15) \times (1 - 0.20) \times (1 - 0.10) = 0.85 \times 0.80 \times 0.90 = 0.6120$$
   - The crew works 4 weeks (4,000 hours total) under degraded efficiency.
   - Effective hours of progress earned:
     $$H_{earned} = 4,000 \times \eta_{actual} = 4,000 \times 0.6120 = 2,448\text{ hours}$$
   - Hours lost to inefficiency: $4,000 - 2,448 = 1,552\text{ hours}$.
   - Cost of lost efficiency: $1,552\text{ hrs} \times \$95/\text{hr} = \$147,440$.
   - Rework hours when piping arrives: $1,200\text{ hours} \times \$95/\text{hr} = \$114,000$.
   - Indirect costs during delay period: $28\text{ days} \times \$3,500/\text{day} = \$98,000$.
   - Additional cost to project: $\Delta AC_B = \$147,440 + \$114,000 + \$98,000 = \$359,440$.
   - Total Cost: $AC_B = \$950,000 + \$359,440 = \$1,309,440$.
   - $CPI_B = \frac{950,000}{1,309,440} = 0.7255$.
- **Financial Conclusion:** Scenario B resulted in $\$359,440$ in additional costs versus $\$478,000$ in Scenario A (saving $\$118,560$). However, if rework exceeded 2,450 hours or efficiency dropped below 0.45, idling would have been financially superior.

### Answer to Exercise 3: Dimensioning Feeding Buffers and Classifying Fever Chart Status
1. **Aggressive Duration and Feeding Buffer:**
   - Aggressive Feeder Duration: $\sum m_i = 3 + 8 + 6 + 3 = 20.0\text{ weeks}$.
   - Safety margins $(b_i - m_i)$:
     - Act 1: $5 - 3 = 2\text{ weeks}$
     - Act 2: $12 - 8 = 4\text{ weeks}$
     - Act 3: $10 - 6 = 4\text{ weeks}$
     - Act 4: $6 - 3 = 3\text{ weeks}$
   - $FB_{RSEM} = \sqrt{2^2 + 4^2 + 4^2 + 3^2} = \sqrt{4 + 16 + 16 + 9} = \sqrt{45} \approx 6.71\text{ weeks}$.
2. **Status at Week 14:**
   - Completed planned work: Activities 1 and 2 completed $= m_1 + m_2 = 3 + 8 = 11.0\text{ weeks}$.
   - Total planned feeder chain: $20.0\text{ weeks}$.
   - $CCR = \frac{11.0}{20.0} = 0.550\quad (55.0\%)$.
   - Delay slip: $\Delta \tau_{consumed} = 3.5\text{ weeks}$.
   - $BPR = \frac{3.5}{6.71} = 0.5216\quad (52.2\%)$.
3. **Fever Chart Zone Evaluation:**
   - $BPR_{warn} = \frac{1}{3} + \frac{1}{3}(0.550) = 0.3333 + 0.1833 = 0.5167\quad (51.7\%)$.
   - $BPR_{crit} = \frac{2}{3} + \frac{1}{3}(0.550) = 0.6667 + 0.1833 = 0.8500\quad (85.0\%)$.
   - Since $0.5167 < BPR (0.5216) < 0.8500$, the feeder buffer is in the **Yellow Zone**.
   - *Mandatory Action:* Dispatch an expediter to the breaker supplier for Activity 3; establish daily manufacturing tracking; identify premium expedited freight for breaker delivery to prevent buffer penetration into the Red Zone.

### Answer to Exercise 4: Analytical Merge Event Bias Calculation
1. **Deterministic CPM Duration:**
   $$T_{merge}^{det} = \max(\mu_1, \mu_2) = \max(40, 38) = 40.0\text{ weeks}$$
2. **Clark's Formula Calculation:**
   $$\theta = \sqrt{\sigma_1^2 + \sigma_2^2} = \sqrt{16 + 25} = \sqrt{41} \approx 6.4031\text{ weeks}$$
   $$\frac{\mu_1 - \mu_2}{\theta} = \frac{40 - 38}{6.4031} = \frac{2}{6.4031} \approx 0.3123$$
   $$\frac{\mu_2 - \mu_1}{\theta} = -0.3123$$
   Given: $\Phi(0.3123) \approx 0.6226$, $\Phi(-0.3123) \approx 0.3774$, $\phi(0.3123) \approx 0.3802$.
   Substitute into Clark's equation:
   $$\mathbb{E}[\max(X_1, X_2)] = 40(0.6226) + 38(0.3774) + 6.4031(0.3802)$$
   $$\mathbb{E}[\max(X_1, X_2)] = 24.904 + 14.341 + 2.434 = 41.679\text{ weeks} \approx 41.68\text{ weeks}$$
3. **Merge Bias Error:**
   $$\text{Bias} = 41.68 - 40.00 = +1.68\text{ weeks}$$
   Deterministic CPM underestimates the expected start date of the merge installation by 1.68 weeks (nearly 12 calendar days).

### Answer to Exercise 5: Critical Analysis of an Contractor Recovery Claim
- **Audit Findings:**
  1. *EVM Inflation:* The contractor earned $\$15\text{M}$ in $EV$ upon PO release. Under ANSI/EIA-748 Guideline 10 & 21, claiming 100% value upon PO placement is a non-compliance. True physical $EV$ earned in Month 18 was $\$210\text{M} - \$15\text{M} = \$195\text{M}$.
  2. *Earned Schedule Recalculation:* With corrected $EV = \$195\text{M}$, looking at the baseline curve, $ES \approx 13.0\text{ months}$.
     $$SV_t = 13.0 - 18.0 = -5.0\text{ months}$$
     $$SPI_t = \frac{13.0}{18.0} = 0.7222$$
  3. *To-Complete Performance Index:*
     $$TSPI_{24} = \frac{24.0 - 13.0}{24.0 - 18.0} = \frac{11.0}{6.0} = 1.833$$
     A $TSPI$ of 1.83 is impossible in industrial construction.
  4. *Conclusion:* The contractor's claim of positive schedule momentum is rejected. The contractor has masked a 5-month critical path slip by front-loading paper material procurement earnings. Mandate a formal Corrective Action Request (CAR) and re-baseline using verified physical gates.

### Answer to Exercise 6: Designing a Supply-Schedule Early Warning Dashboard
**Technical Architecture and Integration Specification:**

```text
+--------------------------------------------------------------------------------------------------+
|                   ENTERPRISE SUPPLY-SCHEDULE DATA PIPELINE SCHEMA                                |
+-------------------+-------------------+--------------------+-------------------------------------+
| Source System     | Entity / Table    | Primary Join Key   | Integrated Attributes               |
+-------------------+-------------------+--------------------+-------------------------------------+
| SAP S/4HANA (MM)  | EKPO / EKET       | PO_LINE_ITEM_ID    | Confirmed Delivery Date, PO Value,  |
|                   | (Purchasing)      | Material Tag Number| Vendor ID, IncoTerms, MIGO Status   |
+-------------------+-------------------+--------------------+-------------------------------------+
| Primavera P6      | TASK / ACTV       | ACTIVITY_ID        | Planned Early Start/Finish, Float,  |
| EPPM              | (Network Logic)   | (Foreign Key to PO)| Driving Predecessors/Successors     |
+-------------------+-------------------+--------------------+-------------------------------------+
| Deltek Cobra      | WBS / CA_WP       | CONTROL_ACCOUNT_ID | BCWS (PV), BCWP (EV), ACWP (AC),    |
| (EVMS Engine)     | (Earned Value)    | WBS_CODE           | CAM ID, Baseline BAC, Variance VARs |
+-------------------+-------------------+--------------------+-------------------------------------+
```

**Automated Exception Flags:**
1. *Delivery-Float Negative Gap:* $\text{Confirmed Delivery Date} > \text{Early Start}_{\text{CWP}}$. Triggers automatic high-priority ticket to Procurement Expediter.
2. *Feeding Buffer Depletion:* $BPR \ge BPR_{warn}(CCR)$. Generates automated Fever Chart alert in weekly executive dashboard.
3. *EVM Guideline 10 Audit Violation:* $EV_{\text{claimed}} > 0$ while SAP MIGO goods receipt inspection status is "Pending" or "Rejected". Blocks contractor monthly progress billing.

### Answer to Exercise 7: Calculation of Multi-Factor EAC_t with Productivity Drag
1. **Multi-Factor Performance Factor and EAC_t:**
   $$PF = 0.8 \cdot SPI_t + 0.2 \cdot CPI = 0.8(0.7551) + 0.2(0.7143) = 0.60408 + 0.14286 = 0.74694$$
   $$EAC_t = AT + \frac{PD - ES}{PF} = 14.0 + \frac{24.0 - 10.5714}{0.74694} = 14.0 + \frac{13.4286}{0.74694} = 14.0 + 17.9782 = 31.98\text{ months}$$
   Total forecasted delay: $31.98 - 24.0 = +7.98\text{ months}$.

2. **Required Daily Burn Rate to Finish by Month 28.0:**
   - Remaining work scope: $BAC - EV(14) = \$480.0\text{M} - \$270.0\text{M} = \$210.0\text{M}$ in earned value terms.
   - Expected remaining cost at current cost efficiency ($CPI = 0.7143$):
     $$ETC = \frac{BAC - EV}{CPI} = \frac{\$210.0\text{M}}{0.7143} = \$293.99\text{M} \approx \$294.0\text{M}$$
   - Available remaining duration: $28.0 - 14.0 = 14.0\text{ months} = 14.0 \times 30.4375\text{ days} = 426.1\text{ calendar days}$.
   - Required daily expenditure rate:
     $$\text{Daily Burn} = \frac{\$293,994,120}{426.1\text{ days}} = \$689,965/\text{day} \approx \$690,000/\text{day}$$

### Answer to Exercise 8: DCMA 14-Point CPLI and BEI Metric Calculation
1. **Critical Path Length Index (CPLI):**
   $$CPLI = \frac{\text{Remaining Duration} + \text{Total Float}}{\text{Remaining Duration}} = \frac{180 + (-25)}{180} = \frac{155}{180} \approx 0.8611$$

2. **Baseline Execution Index (BEI):**
   $$BEI = \frac{\text{Actual Tasks Completed to Date}}{\text{Baseline Tasks Scheduled to Complete to Date}} = \frac{28}{40} = 0.7000$$

3. **DCMA Compliance Evaluation and Contractual Implications:**
   - *CPLI Threshold Failure ($0.861 < 0.95$):* The critical path is broken. A CPLI below 0.95 indicates that the project cannot meet its contractual completion date without accelerating remaining tasks by at least 13.9% or negotiating a 25-day extension.
   - *BEI Threshold Failure ($0.700 < 0.95$):* Severe execution failure. The contractor is delivering only 70% of planned monthly procurement and installation tasks, demonstrating that schedule slippage is systemic rather than an isolated outlier.
   - *Contractual Consequence:* Triggers a formal DCMA Corrective Action Request (CAR Level II), mandatory submission of an Over-Target Baseline (OTB) or Schedule Recovery Plan within 30 days, and potential withholding of up to 10% of monthly progress billings under DFARS 252.242-7005 (Contractor Business Systems).

---

## 14. Interview questions

### Question 1: Classical SPI Breakdown
- **Prompt:** "Why does classical Schedule Variance ($SV$) and Schedule Performance Index ($SPI$) under ANSI/EIA-748 fail towards the end of a delayed capital project, and how does Earned Schedule mathematically resolve this?"
- **Strong Answer:** A strong candidate demonstrates that classical $SV$ is measured in dollars ($EV - PV$) rather than time. As a project completes, $EV$ must equal $BAC$, and because $PV$ also equals $BAC$ past the planned duration ($PD$), $SV$ mathematically forces itself to $\$0$ and $SPI$ forces itself to 1.0, regardless of how many months or years the project is delayed. The candidate explains that Earned Schedule ($ES$) resolves this by projecting cumulative $EV$ onto the baseline $PV$ curve to find the exact planned time ($C + I$) at which that volume of work should have been completed. This yields time-based metrics ($SV_t = ES - AT$ and $SPI_t = ES / AT$) that retain diagnostic validity through the entire project lifecycle, never artificially converging to 1.0.
- **Weak Answer:** Mentions that EVM has flaws or that "dollars don't measure time," but cannot state the mathematical proof ($EV \to BAC \implies SV \to 0$) or explain the interpolation mechanics of Earned Schedule.

### Question 2: Supply Delay Impact on Contractor CPI
- **Prompt:** "When an engineered-to-order equipment delivery is delayed by eight weeks, explain the exact operational mechanisms that cause the contractor's Cost Performance Index ($CPI$) to collapse."
- **Strong Answer:** Explains that a delivery delay blocks planned Construction Work Packages (CWPs), freezing earned value generation ($\Delta EV = 0$). Concurrently, actual costs ($AC$) continue to burn via fixed site indirects, rental equipment, and craft labor standby. If the contractor diverts craft labor out of sequence to non-critical tasks to avoid paying idle standby, the candidate cites the compounding productivity degradation ($\eta = \eta_0 \cdot \prod (1 - \delta_i)$) resulting from trade stacking, spatial crowding, and supervisory dilution. Furthermore, when the delayed equipment arrives, previously installed out-of-sequence interferences must be dismantled and re-worked. Because $EV$ is fixed to baseline budget while $AC$ escalates from inefficiency and rework, $CPI = EV / AC$ collapses.
- **Weak Answer:** States that "delays cost money" or that the contractor has to pay overtime later, without detailing trade stacking, the mathematical formulation of labor efficiency loss, or the decoupling of $EV$ accrual from $AC$ burn.

### Question 3: Merge Event Bias in CPM Networks
- **Prompt:** "Why does deterministic Critical Path Method (CPM) systematically underestimate capital project completion dates when multiple procurement paths converge on an installation milestone?"
- **Strong Answer:** Cites Jensen's inequality and order statistics: the expected value of the maximum of random variables is strictly greater than the maximum of their expected values ($\mathbb{E}[\max(X_1, \dots, X_K)] > \max(\mathbb{E}[X_i])$). When parallel procurement chains converge, any path that slips pushes the merge milestone out, while paths that finish early cannot pull the milestone forward due to precedence constraints. The candidate references Clark's equation for normal distributions or Monte Carlo SRA, explaining that deterministic CPM assumes variance is zero, hiding the positive completion date bias.
- **Weak Answer:** States that "more things can go wrong" without identifying the mathematical principle of merge event bias or understanding why symmetric float does not protect against merge slips.

### Question 4: ANSI/EIA-748 Material Progress Compliance
- **Prompt:** "An EPC contractor issues a purchase order for a $20M steam turbine and immediately claims $10M of Earned Value on their monthly invoice, citing a 50/50 earning rule. How do you audit this claim under ANSI/EIA-748?"
- **Strong Answer:** Rejects the claim as a direct violation of Guidelines 10 and 21. Explains that the 50/50 rule is restricted to catalog items spanning two reporting periods, not multi-year engineered-to-order equipment. For major capital packages, ANSI/EIA-748 requires objective, auditable milestone value gates (e.g., drawing freeze, raw forging UT test, FAT, bill of lading, receipt inspection). Furthermore, under Guideline 10, claiming $EV$ requires a matching accrual of Actual Cost ($AC$); earning progress on an unfulfilled PO artificially inflates $CPI$ and distorts project controls.
- **Weak Answer:** Accepts the claim if the contract allows 50/50 progress, or focuses purely on whether the invoice was approved by accounting without referencing ANSI/EIA-748 earning gates.

### Question 5: Critical Chain Feeding Buffers vs. Traditional Float
- **Prompt:** "How does a Critical Chain Feeding Buffer differ conceptually and mathematically from Total Float in a traditional CPM schedule?"
- **Strong Answer:** Explains that Total Float belongs to the path and is frequently squandered through Parkinson's Law ("work expands to fill the time available") and Student Syndrome (delaying work until the deadline). In contrast, Critical Chain Project Management strips hidden safety padding from individual activity estimates (reducing them to 50% median durations, $m_i$) and aggregates that safety time into an explicit, pooled Feeding Buffer ($FB$) dimensioned via the Root-Square-Error Method ($FB = \sqrt{\sum (b_i - m_i)^2}$). The feeding buffer is actively monitored via fever charts, whereas total float is unmanaged until it turns negative.
- **Weak Answer:** Confuses feeding buffers with contingency time or states that they are the same thing under different names.

### Question 6: Interpreting To-Complete Performance Indices
- **Prompt:** "Your status report shows $SPI_t = 0.82$ and $TSPI = 1.28$. What does this tell you about the project's schedule viability, and what decision should you recommend to executive leadership?"
- **Strong Answer:** Explains that $SPI_t = 0.82$ indicates the project is currently converting calendar time into earned schedule at an efficiency of only 82%. A $TSPI$ of 1.28 indicates that to meet the approved baseline completion date ($PD$), the contractor must accelerate work velocity to 128% of baseline efficiency across all remaining scope. In heavy construction, empirical data proves that $TSPI > 1.20$ is virtually unachievable. The candidate recommends formally declaring a baseline schedule breach, rejecting contractor recovery schedules that rely on unachievable productivity spikes, and re-baselining to a realistic $P_{80}$ completion date derived from Monte Carlo SRA.
- **Weak Answer:** Suggests adding more workers or approving overtime to meet the 1.28 target without recognizing that $TSPI = 1.28$ is historically unachievable in capital projects.

### Question 7: Activity Criticality Index vs. Deterministic Float
- **Prompt:** "In a Monte Carlo SRA report, a procurement activity for high-voltage transformers has 15 days of deterministic float in P6, but its Criticality Index ($CI$) is 68%. What does this mean?"
- **Strong Answer:** Explains that deterministic float is calculated using single-point duration estimates, giving the illusion of a schedule buffer. However, the transformer procurement has significant lead-time variance (wide PERT beta distribution) and merges with other paths. Across 10,000 Monte Carlo iterations, in 68% of the runs, this activity consumed its float and drove the project completion date. It is a "stealth critical" activity that requires immediate expediting and executive tracking despite showing positive float on the static Gantt chart.
- **Weak Answer:** States that the schedule must have a logic error or that the Monte Carlo simulation produced inconsistent numbers.

### Question 8: Material Price vs. Usage Variance under EVMS
- **Prompt:** "Under ANSI/EIA-748 Guideline 23, how do you mathematically isolate Material Price Variance (MPV) from Material Usage Variance (MUV) in a capital piping account?"
- **Strong Answer:** States the standard standard-cost variance decomposition:
  $$\text{Material Cost Variance (MCV)} = EV - AC = (Q_{bud} \cdot P_{bud}) - (Q_{act} \cdot P_{act})$$
  $$\text{Material Price Variance (MPV)} = Q_{act} \cdot (P_{bud} - P_{act})$$
  $$\text{Material Usage Variance (MUV)} = P_{bud} \cdot (Q_{bud} - Q_{act})$$
  Explains that isolating these variances reveals whether a cost overrun is driven by market commodity price escalations (MPV) or field scrap, engineering design growth, and takeoff errors (MUV).
- **Weak Answer:** Confuses cost variance with schedule variance or cannot provide the mathematical breakdown between quantity and price.

---

## 15. Key terms

- **Actual Cost (AC / ACWP):** Realized direct and indirect costs accrued for work accomplished during a reporting period.
- **Actual Time (AT):** The cumulative calendar time elapsed from project baseline start to the status reporting date.
- **ANSI/EIA-748:** The American National Standards Institute standard detailing 32 criteria for Earned Value Management Systems.
- **Budget at Completion (BAC):** Total approved budget authorized for the baseline project scope.
- **Buffer Penetration Ratio (BPR):** The percentage of a critical chain feeding or project buffer consumed by schedule slippage.
- **Chain Completion Ratio (CCR):** The fraction of total planned duration completed along a critical chain feeding path.
- **Clark's Formula:** A closed-form statistical approximation for the expected value of the maximum of normal random variables.
- **Control Account (CA):** A management control point integrating WBS and OBS where performance is measured.
- **Cost Performance Index (CPI):** Cost efficiency of completed work ($CPI = EV / AC$).
- **Cost Variance (CV):** Financial cost variance: $CV = EV - AC$.
- **Critical Chain Project Management (CCPM):** Schedule management methodology aggregating safety buffers to protect the critical path.
- **Criticality Index (CI):** The probability that an activity falls on the critical path in a Monte Carlo simulation.
- **Earned Schedule (ES):** Time-based metric projecting $EV$ onto the baseline $PV$ curve: $ES = C + I$.
- **Earned Value (EV / BCWP):** Budgeted value of physical work completed.
- **Estimate at Completion (EAC):** Forecasted total cost or time required to finish all authorized scope.
- **Feeding Buffer (FB):** A pooled safety buffer placed where a non-critical supply path joins the critical chain.
- **Fever Chart:** A two-dimensional diagnostic chart plotting buffer penetration against chain completion across risk zones.
- **Merge Event Bias:** The statistical certainty that converging path durations have an expected maximum exceeding deterministic CPM estimates.
- **Milestone Weighting:** Progress earning rule dividing major ETO equipment into discrete, verified progress gates.
- **Out-of-Sequence Work:** Executing downstream field construction before upstream equipment arrives, inducing inefficiency.
- **Planned Duration (PD):** Total planned calendar duration from baseline start to baseline finish.
- **Planned Value (PV / BCWS):** Budgeted cost of work scheduled to be performed over time.
- **Root-Square-Error Method (RSEM):** Statistical formula sizing feeding buffers via sum of squared safety margins.
- **Schedule Performance Index (SPI):** Classical cost-based schedule efficiency ratio: $SPI = EV / PV$.
- **Schedule Risk Reserve (SRR):** Time contingency buffer added to deterministic finish dates to achieve $P_{80}$ confidence.
- **Schedule Variance (SV):** Classical cost-based schedule variance: $SV = EV - PV$.
- **Time-based Schedule Performance Index ($SPI_t$):** Time efficiency ratio: $SPI_t = ES / AT$.
- **Time-based Schedule Variance ($SV_t$):** Calendar time variance: $SV_t = ES - AT$.
- **To-Complete Schedule Performance Index (TSPI):** Required schedule efficiency on remaining work to achieve target dates.
- **Trade Stacking:** Labor crowding in a confined work area leading to severe multi-trade productivity losses.

---

## 16. Sources

- **ANSI / NDIA.** (2019). *ANSI/EIA-748-D: Standard for Earned Value Management Systems*. Arlington, VA: National Defense Industrial Association (NDIA).
- **Clark, C. E.** (1961). The greatest of a finite set of random variables. *Operations Research*, 9(2), 145–162. doi:10.1287/opre.9.2.145
- **Defense Contract Management Agency (DCMA).** (2023). *Earned Value Management Systems Implementation Guide (EVMSIG)*. Fort Lee, VA: DCMA.
- **Goldratt, E. M.** (1997). *Critical Chain*. Great Barrington, MA: The North River Press.
- **Hanna, A. S., Russell, J. S., Nordheim, E. V., & Bruggink, M. J.** (1999). Impact of change orders on labor efficiency for electrical construction. *Journal of Construction Engineering and Management*, 125(4), 224–232. doi:10.1061/(ASCE)0733-9364(1999)125:4(224)
- **International Organization for Standardization (ISO).** (2018). *ISO 21508:2018: Earned value management in project and programme management*. Geneva, Switzerland: ISO.
- **Lipke, W.** (2003). Schedule is different: The earned schedule concept. *The Measurable News*, Summer 2003, 31–34.
- **Lipke, W., Zwikael, O., Henderson, K., & Anbari, F.** (2009). Prediction of project outcome: The application of statistical methods to Earned Value Management and Earned Schedule. *International Journal of Project Management*, 27(4), 400–407. doi:10.1016/j.ijproman.2008.02.009
- **Thomas, H. R., & Smith, G. R.** (1990). Loss of labor productivity due to trade stacking and out-of-sequence work. *Construction Industry Institute (CII) Research Report*, SD-58. Austin, TX: CII.
- **U.S. Department of Energy (DOE).** (2021). *DOE Order 413.3B: Program and Project Management for the Acquisition of Capital Assets*. Washington, DC: U.S. Department of Energy.
- **U.S. Government Accountability Office (GAO).** (2016). *GAO Schedule Assessment Guide: Best Practices for Project Schedules* (GAO-16-89G). Washington, DC: GAO.

---

## 17. Links

- [Module Index (`_index.md`)](_index.md)
- [Module Survey File (`../../../course/34_energy_utilities_capital_projects.md`)](../../../course/34_energy_utilities_capital_projects.md)
- Sibling Topics:
  - [Topic 01: Asset Register and Criticality](01_asset-register-criticality.md)
  - [Topic 02: MRO Demand and Failure Modelling](02_mro-failure-modelling.md)
  - [Topic 03: Outage and Turnaround Materials](03_outage-turnaround-materials.md)
  - [Topic 04: Long-Lead Equipment and Project Procurement](04_long-lead-procurement.md)
  - [Topic 05: Bulk Materials and Construction Logistics](05_bulk-materials-construction.md)
  - [Topic 07: Commodity Price and Contract Structures](07_commodity-price-contracts.md)
  - [Topic 08: Renewables and Grid Equipment](08_renewables-grid-spares.md)
  - [Topic 09: Resilience and Storm Response](09_resilience-storm-response.md)
- Related Modules:
  - [Module 10: Inventory Optimization](../../02_methods/10_inventory_optimization/_index.md)
  - [Module 14: Network Design & Facility Location](../../02_methods/14_network_design_facility_location/_index.md)
  - [Module 31: Aftermarket Spare Parts & Service Logistics](../31_aftermarket_spare_parts_service/_index.md)
