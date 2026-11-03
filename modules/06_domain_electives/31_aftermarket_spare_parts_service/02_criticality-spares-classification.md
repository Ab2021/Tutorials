> Module 31 — Aftermarket, Spare Parts, and Service Supply Chains
> Topic 02 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Module 31 Topic 01 (Installed Base and Failure Demand), Module 10 (Inventory Optimization)
> Version 2026.5

# Criticality and Spares Classification

---

## 1. Why this topic exists

In finished goods and retail supply chains, inventory classification is dominated by Pareto's Law: standard **ABC analysis** sorts stock-keeping units (SKUs) by annual monetary sales volume (annual demand multiplied by unit price). The top 10–20% of SKUs generating 70–80% of revenue receive "Class A" designation and rigorous replenishment control, while the long tail of low-spend items is relegated to "Class C" with minimal review.

When an aftermarket supply chain applies traditional ABC classification to spare parts, it guarantees operational and financial disaster.

The fundamental pathology of spare parts management is the conflation of **commercial expense** with **operational criticality**:
* A $120,000 spare structural casing for an offshore gas compressor carries enormous balance-sheet value (Class A in ABC analysis). Yet if the operating compressor's casing suffers minor surface scratching, the machine continues running without interruption. The casing is **expensive, but non-critical**.
* A $45 elastomeric fluoroelastomer (Viton) o-ring seal on the compressor's primary dry gas fuel line carries negligible spend (Class C in ABC analysis). Yet if that $45 seal ruptures and no replacement is on hand in the local site warehouse, the high-pressure gas system trips an emergency shutdown. The offshore production platform flares gas, losing $750,000 per day in unrecoverable hydrocarbons. The seal is **cheap, but ultra-critical**.

Under standard ABC rules, the procurement system obsessively monitors the $120,000 non-critical casing, holding multi-million-dollar safety buffers that sit untouched on wooden pallets for twenty years. Simultaneously, the inventory system minimizes working capital on the $45 "Class C" seal, purchasing it via periodic bulk reorders with long lead times, oblivious to the fact that a stockout halts the entire operating facility.

Furthermore, traditional classification schemes fail to account for:
1. **System Redundancy:** A failure of a pump bearing does not stop the plant if an identical standby pump is piped in parallel with an automatic failover check valve ($1+1$ redundancy).
2. **Replacement Lead Time:** A critical part with a 2-hour local courier availability requires a vastly different stocking strategy than an equally critical part with a 26-week custom forging lead time from a German foundry.
3. **Multi-Attribute Failure Modes:** Criticality is not a scalar score; it is a vector of consequences spanning personnel life-safety, catastrophic environmental releases, statutory non-compliance fines, production throughput losses, and contractual service level agreement (SLA) penalties.

To run a reliable, cost-efficient aftermarket network, organizations must abandon one-dimensional spend ranking. Supply chain leaders must implement multidimensional **criticality classification** grounded in Reliability-Centered Maintenance (RCM), Failure Mode, Effects, and Criticality Analysis (FMECA), and consequence-based decision frameworks. This topic provides the mathematical, taxonomic, and operational mechanics required to categorize spare parts and map each segment to a defensible inventory policy.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Demonstrate** why traditional Pareto/ABC inventory classification fails in capital equipment aftermarket environments using empirical cost and downtime metrics.
2. **Construct** a comprehensive criticality taxonomy that integrates human safety, environmental hazard, production downtime cost, system redundancy, and replenishment lead time.
3. **Evaluate** the mathematical differences between ordinal Risk Priority Numbers (RPN) from standard FMEA and continuous quantitative Criticality Numbers ($C_r$) from FMECA (MIL-STD-1629A).
4. **Implement** the VED (Vital, Essential, Desirable) classification scheme and establish auditable, non-subjective boundary criteria for equipment parts catalogs.
5. **Formulate** a multi-criteria Analytic Hierarchy Process (AHP) scoring model for spare parts prioritization, including pairwise comparison matrices and consistency ratio ($CR$) verification.
6. **Design** a two-dimensional Criticality-Value Policy Matrix (e.g., Critical/Expensive, Critical/Inexpensive, Non-Critical/Expensive, Non-Critical/Inexpensive) and assign explicit stocking strategies to each quadrant.
7. **Calculate** the economic breakeven stocking point for insurance spares using expected downtime loss, component failure intensity, and inventory holding cost.
8. **Establish** an engineering change governance protocol to dynamically adjust spare parts criticality ratings based on operating asset age, field reliability drift, and contract SLA revisions.

---

## 3. The core mental model

To make rational stocking decisions, you must map every spare part across two orthogonal axes: **Operational Criticality (Consequence of Stockout)** and **Economic Profile (Procurement Cost & Velocity)**.

```text
       OPERATIONAL CRITICALITY (Consequence of Missing Part)
       High ▲
            │ ┌──────────────────────────────┬──────────────────────────────┐
            │ │ QUADRANT I: INSURANCE SPARES │ QUADRANT II: STRATEGIC CRIT. │
            │ │ Vital / Low Spend / Long LT  │ Vital / High Spend / Long LT │
            │ │                              │                              │
            │ │ • Policy: Local Forward Stock│ • Policy: Central Pooling    │
            │ │ • Target Availability: 99.5% │ • Target Availability: 95.0% │
            │ │ • Mechanism: Generous Buffer │ • Mechanism: METRIC / Repair │
            │ │ • Example: High-temp O-ring, │ • Example: MRI Gradient Amp, │
            │ │   pressure sensor, burst-disk│   turbine rotor, avionics LRU│
            │ ├──────────────────────────────┼──────────────────────────────┤
            │ │ QUADRANT IV: BULK HARDWARE   │ QUADRANT III: COMMERCIAL EXP.│
            │ │ Desirable / Cheap / Common   │ Desirable / Expensive / Rare │
            │ │                              │                              │
            │ │ • Policy: Vendor-Managed Inv.│ • Policy: Build-to-Order     │
            │ │ • Target Availability: 90.0% │ • Target Availability: N/A   │
            │ │ • Mechanism: Two-bin Kanban  │ • Mechanism: Supplier SLA    │
            │ │ • Example: Standard bolts,   │ • Example: Decorative fairing│
            │ │   filters, wiring zip-ties   │   sound insulation, non-OEM  │
            │ └──────────────────────────────┴──────────────────────────────┘
            └─────────────────────────────────────────────────────────────►
            Low                                                         High
                             COMMERCIAL VALUE / UNIT COST
```

### 3.1 Deconstructing the Four Quadrants

1. **Quadrant I (Insurance Spares / Critical-Inexpensive):**
   * *The Blind Spot:* The most dangerous quadrant in supply chain operations. These parts cost under $100, but their absence causes immediate, catastrophic asset stoppage.
   * *Stocking Policy:* Place generous safety stock directly at forward stocking locations (FSLs) or machine site cages. The annual holding cost of an extra five $50 gaskets is $50/year (assuming a 20% holding cost rate). A single hour of asset downtime costs $10,000+. Sparing here is cheap insurance; running out is managerial malpractice.
2. **Quadrant II (Strategic Critical / Critical-Expensive):**
   * *The Working Capital Driver:* High-value assemblies ($10,000 to $500,000+) whose absence stops the asset, but whose high cost precludes stocking multiple units at every site.
   * *Stocking Policy:* Centralized or regional pooling. Use multi-echelon inventory optimization (METRIC models), rotables repair loops, and fast air-courier distribution networks.
3. **Quadrant III (Commercial Expensive / Non-Critical):**
   * *The Capital Trap:* High-cost components that do not affect the machine’s primary functional output or safety (e.g., cosmetic panels, auxiliary operator convenience lighting, comfort seating).
   * *Stocking Policy:* Zero stocking or build-to-order. Negotiate supplier lead-time contracts or repair-and-return agreements. Never tie up working capital buffering non-critical luxury components.
4. **Quadrant IV (Bulk Hardware / Non-Critical Inexpensive):**
   * *The Administrative Waste:* Standard commercial off-the-shelf (COTS) fasteners, brackets, and standard hydraulic fittings.
   * *Stocking Policy:* Consignment stock or Vendor-Managed Inventory (VMI) utilizing physical two-bin visual Kanban systems. Eliminate analytical forecasting overhead entirely.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse | Unit or Coding Case |
|---|---|---|---|
| **Criticality** | The measure of the severity of operational, economic, safety, or environmental consequences resulting from an item's functional failure and subsequent unavailability. | Treating part unit purchase price as a surrogate for criticality. | Categorical (Vital / Essential / Desirable) or Continuous Index $[0, 100]$. |
| **VED Classification** | A specialized aftermarket classification sorting parts into Vital (halts operation), Essential (degrades performance), and Desirable (negligible impact). | Allowing technicians to subjectively classify 90% of parts as "Vital" without auditable criteria. | Categorical: `V`, `E`, `D`. |
| **FMEA / FMECA** | Failure Mode, Effects, and (Criticality) Analysis: A systematic inductive engineering technique to identify all potential failure modes, causes, and consequences across a system. | Running FMEA solely for design quality audits and never transferring the results to ERP stocking policies. | Engineering report conforming to IEC 60812 or MIL-STD-1629A. |
| **Risk Priority Number (RPN)** | The mathematical product of ordinal rankings: $\text{RPN} = \text{Severity} \times \text{Occurrence} \times \text{Detection}$. | Treating RPN as a ratio-scale continuous metric or using RPN thresholds directly to size safety stocks. | Integer scale: typically $[1, 1000]$; dimensionless ordinal. |
| **Criticality Number ($C_r$)** | The expected number of failures of a specific component mode resulting in system loss per operating period: $C_r = \lambda \cdot t \cdot \alpha \cdot \beta_f$. | Conflating the qualitative FMEA RPN with the quantitative FMECA criticality number. | Expected critical failures per million hours or cycles ($[T^{-1}]$). |
| **Active Redundancy** | A system architecture where two or more parallel redundant units operate simultaneously; failure of one unit causes zero interruption to system function. | Assuming redundant equipment eliminates the need for spare parts (redundancy merely converts a crash-stop into a repair window). | System topology: e.g., $1+1$ active, 2-out-of-3 ($2oo3$). |
| **Standby Redundancy** | A configuration where a secondary backup unit remains unpowered (cold standby) or idling (warm standby) until an automated or manual switch engages it upon primary failure. | Ignoring switchover reliability; if the transfer switch fails, the standby unit is useless. | Topology: $1+1$ standby; switchover probability $P_{\text{switch}} \in [0, 1]$. |
| **Mean Down Time (MDT)** | The total elapsed calendar time an asset is non-operational, including logistics delay time (LDT), administrative delay, and active Mean Time to Repair (MTTR). | Conflating MTTR (wrench time) with MDT (which is dominated by parts waiting time). | Hours ($[T]$). |
| **Logistics Delay Time (LDT)** | The portion of asset downtime spent waiting for spare parts, specialized tooling, or technician dispatch. | Assuming LDT is zero by default in reliability availability models. | Hours ($[T]$). |
| **Insurance Spare** | A high-criticality, low-probability-of-failure part held solely to insure against extended catastrophic downtime caused by long replenishment lead times. | Applying standard Economic Order Quantity (EOQ) or Wilson lot-sizing to insurance spares. | Binary stocking policy: $S \in \{0, 1, 2\}$. |
| **Analytic Hierarchy Process (AHP)** | A structured multi-criteria decision technique developed by Thomas Saaty utilizing pairwise comparisons and eigenvalue decomposition to generate criteria weights. | Calculating AHP weights without verifying the Consistency Ratio ($CR < 0.10$). | Normalized priority vector $\mathbf{w} \in \mathbb{R}^m$, $\sum w_i = 1.0$. |
| **Master Minimum Equipment List (MMEL)** | An aviation/marine regulatory document specifying which instruments, equipment, or functions may be temporarily inoperative while still allowing safe operation. | Sparing non-MMEL items with the same urgency as "No-Go" items required for vehicle dispatch. | Regulatory status: `No-Go`, `Go-if`, `Go`. |
| **Downtime Cost Rate ($C_{\text{down}}$)** | The total marginal economic loss incurred per unit of asset downtime, including lost production margin, flaring penalties, idle labor, and contractual SLA damages. | Using direct accounting depreciation instead of lost marginal cash contribution. | Currency per unit time (e.g., $\$ / \text{hour}$). |
| **Holding Cost Rate ($r_h$)** | The annualized percentage cost of capital, physical warehousing, insurance, obsolescence risk, and tax associated with carrying spare inventory. | Using a generic 10% corporate cost of capital without factoring in spares obsolescence risks. | Percentage per year (typically $0.18 - 0.30 \text{ yr}^{-1}$). |

---

## 5. The mechanics, step by step

Implementing an auditable, multidimensional spares classification engine requires eight systematic steps. The procedure integrates plant engineering, reliability physics, and procurement operations.

```text
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 1: Establish Asset-Level Functional Criticality (Safety, Environment, Cost)│
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 2: Conduct FMECA and Identify Subsystem Redundancy Topologies              │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 3: Compute Quantitative Component Criticality Indices (Cr & VED Status)    │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 4: Calibrate Multi-Criteria Weights via Analytic Hierarchy Process (AHP)   │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 5: Score Component Composite Criticality and Define Boundary Thresholds     │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 6: Map Components into the Two-Dimensional Policy Matrix (Criticality/Value)│
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 7: Calculate Economic Stocking Thresholds for Long Lead-Time Spares        │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 8: Establish Dynamic Governance, Obsolescence Tracking, and Review Triggers │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Step 1: Establish Asset-Level Functional Criticality

Before evaluating a spare part, you must evaluate the machine into which it is installed. A bearing inside a tertiary backup sump pump does not carry the same consequence as the identical bearing inside the plant’s primary distillation compressor.
* **Input:** Plant layout, Process Flow Diagrams (PFD), Piping and Instrumentation Diagrams (P&ID), operating profit models, customer contracts.
* **Procedure:**
  Evaluate each parent asset across four statutory and operational impact categories:
  1. **Safety Consequence ($S_{\text{safe}} \in [1, 5]$):** 1 = Negligible first-aid; 5 = Severe risk of human fatality or major toxic release.
  2. **Environmental Consequence ($S_{\text{env}} \in [1, 5]$):** 1 = Fully contained inside facility; 5 = Off-site catastrophic environmental breach with statutory criminal liability.
  3. **Production Downtime Impact ($S_{\text{prod}} \in [1, 5]$):**
     * Tier 1: Zero production loss (standby capacity absorbs load).
     * Tier 2: Deferred production (can be caught up on weekend shifts without customer impact).
     * Tier 3: Partial derating of facility throughput ($< 25\%$).
     * Tier 4: Major unit shutdown ($25\% - 75\%$ loss).
     * Tier 5: Total plant trip ($> 75\%$ output halted; $C_{\text{down}} > \$50,000/\text{hr}$).
  4. **Contractual SLA Penalty ($S_{\text{sla}} \in [1, 5]$):** Contractual liquidated damages payable to end customers for asset unavailability.
* **Output:** Asset Functional Criticality Category: $\mathcal{A}_{\text{tier}} \in \{\text{Critical-1}, \text{Essential-2}, \text{General-3}\}$.

### 5.2 Step 2: Conduct FMECA and Identify Subsystem Redundancy Topologies

Determine how component failure propagates to the parent asset.
* **Input:** Engineering FMECA sheets, reliability block diagrams (RBD).
* **Procedure:**
  1. Map the engineering component to its system failure modes (e.g., seizure, fatigue cracking, leakage, electrical short).
  2. Identify the active hardware redundancy configuration:
     * **No Redundancy ($1oo1$):** Single Point of Failure (SPOF). Component failure directly causes asset shutdown.
     * **Parallel Active Redundancy ($1oo2$ or $2oo3$):** Component failure triggers an alarm; load is transferred automatically. Asset continues running.
     * **Standby Redundancy ($1+1$ Cold/Warm):** Asset requires manual or automated switchover to secondary line.
  3. Determine the conditional probability of system loss given component failure ($\beta_f$):
     $$\beta_f = P(\text{Asset functional failure} \mid \text{Component failure mode occurs})$$
     * For a single point of failure: $\beta_f = 1.0$.
     * For a fully redundant $1+1$ active system with independent failures: $\beta_f \approx 0.0$ (assuming mean time to repair the primary is far less than MTBF of the secondary).

### 5.3 Step 3: Compute Quantitative Component Criticality ($C_r$) vs. Qualitative RPN

Avoid using ordinal Risk Priority Numbers (RPN) to determine inventory stocking. Standard RPN multiplies three ordinal scores:
$$\text{RPN} = S \times O \times D$$
Because $S$, $O$, and $D$ are ordinal ranking scales (where the distance between 3 and 4 is not mathematically equal to the distance between 8 and 9), multiplying them produces a heavily skewed, non-continuous scale with 120 unique outcomes out of 1000 combinations, yielding mathematical nonsense (e.g., an event with $S=10, O=1, D=1 \implies \text{RPN}=10$ is treated as lower priority than $S=2, O=3, D=2 \implies \text{RPN}=12$, even though the first event represents catastrophic fatality risk).

Instead, utilize the quantitative **Criticality Number ($C_r$)** defined in **MIL-STD-1629A**:
$$C_r = \sum_{j=1}^m (\lambda \cdot t \cdot \alpha_j \cdot \beta_{f,j})$$
where:
* $\lambda$ = Component total failure rate (failures per hour).
* $t$ = Operating duration or mission exposure (hours).
* $\alpha_j$ = Failure mode ratio ($\sum \alpha_j = 1.0$; fraction of component failures attributable to failure mode $j$).
* $\beta_{f,j}$ = Conditional failure effect probability (probability that mode $j$ results in asset functional loss).

Map the resulting engineering analysis into the standardized **VED (Vital, Essential, Desirable)** taxonomy:
* **Vital (V):** $C_r > 0$ on a Critical-1 asset with no installed redundancy ($\beta_f = 1.0$). Absence of the part causes instantaneous plant trip or violates safety/environmental statutory licenses.
* **Essential (E):** Component failure causes asset derating, reduces efficiency, or compromises secondary systems. Asset can continue operating for a limited grace period (e.g., 24–72 hours per MMEL limits).
* **Desirable (D):** Non-critical auxiliary parts. Absence does not impact production, safety, or legal compliance.

### 5.4 Step 4: Calibrate Multi-Criteria Weights via Analytic Hierarchy Process (AHP)

Because spare parts decision-making balances conflicting objectives (safety, downtime, lead time, unit cost), deploy Saaty’s Analytic Hierarchy Process to calculate rigorous, mathematically consistent criteria weights.

1. Construct an $m \times m$ pairwise comparison matrix $\mathbf{A}$, where entry $a_{jk}$ represents the relative importance of criterion $j$ versus criterion $k$ using Saaty’s 1-to-9 fundamental scale:
   * $1$: Equal importance.
   * $3$: Moderate importance.
   * $5$: Strong importance.
   * $7$: Very strong / demonstrated importance.
   * $9$: Extreme importance.
   * Reciprocal property: $a_{kj} = 1 / a_{jk}$.
2. Normalize the columns of matrix $\mathbf{A}$:
   $$\bar{a}_{jk} = \frac{a_{jk}}{\sum_{i=1}^m a_{ik}}$$
3. Compute the criteria priority weight vector $\mathbf{w}$ as the row averages of the normalized matrix:
   $$w_j = \frac{1}{m} \sum_{k=1}^m \bar{a}_{jk}$$
4. Calculate the maximum eigenvalue $\lambda_{\max}$:
   $$\mathbf{A} \mathbf{w} = \lambda_{\max} \mathbf{w} \implies \lambda_{\max} = \frac{1}{m} \sum_{j=1}^m \frac{(\mathbf{A} \mathbf{w})_j}{w_j}$$
5. Compute the Consistency Index ($CI$) and Consistency Ratio ($CR$):
   $$CI = \frac{\lambda_{\max} - m}{m - 1}$$
   $$CR = \frac{CI}{RI}$$
   where $RI$ is the empirical Random Consistency Index for an $m \times m$ matrix (for $m=4$, $RI=0.90$; for $m=5$, $RI=1.12$).
   * **Rule of Acceptance:** If $CR < 0.10$, the judgments are mathematically consistent. If $CR \ge 0.10$, the pairwise comparisons are inconsistent and must be re-evaluated.

### 5.5 Step 5: Score Component Composite Criticality

Compute the Composite Criticality Score ($\text{CCS}_p \in [0, 100]$) for each part SKU $p$:
$$\text{CCS}_p = 100 \times \sum_{j=1}^m w_j \cdot s_{p,j}$$
where $s_{p,j} \in [0, 1]$ is the normalized score of part $p$ on criterion $j$.

Define objective classification bands:
* **Tier 1 (High Criticality / Vital):** $\text{CCS}_p \ge 70$ (or any part with a statutory Safety score of 1.0 regardless of composite average).
* **Tier 2 (Medium Criticality / Essential):** $40 \le \text{CCS}_p < 70$.
* **Tier 3 (Low Criticality / Desirable):** $\text{CCS}_p < 40$.

### 5.6 Step 6: Map Components into the Two-Dimensional Policy Matrix

Cross the Composite Criticality Tier with the Commercial Spend / Unit Cost tier to place the SKU into its operational quadrant (as illustrated in Section 3):

```text
Criticality Score (CCS)
      ▲
  100 ┼──────────────────────────────┬──────────────────────────────┐
      │ Q-I: INSURANCE SPARES        │ Q-II: STRATEGIC CRITICAL     │
      │ • Max availability (99.5%)   │ • Multi-echelon pooling      │
      │ • Decentralized stock        │ • Repair loop / rotables     │
   70 ┼──────────────────────────────┼──────────────────────────────┤
      │ Q-IV (A): ESSENTIAL BUFFER   │ Q-III (A): MONITORED REORDER │
      │ • Standard safety stock      │ • Central regional hub       │
      │ • Target availability: 95%   │ • Target availability: 90%   │
   40 ┼──────────────────────────────┼──────────────────────────────┤
      │ Q-IV (B): BULK / TWO-BIN VMI │ Q-III (B): ORDER-ON-DEMAND   │
      │ • Min-max visual Kanban      │ • Zero stock / Vendor SLA    │
    0 ┴──────────────────────────────┴──────────────────────────────►
      $0                           $1,000                     $100,000+
                                UNIT COST
```

### 5.7 Step 7: Calculate Economic Stocking Thresholds for Insurance Spares

For high-criticality parts with extremely low demand and long lead times (where statistical Poisson models often suggest stocking 0 units due to low average failure rates), use the **Marginal Economic Risk Model**.

Stocking an extra spare unit ($S=1$ vs. $S=0$) trades off holding cost against expected downtime catastrophe:
* Annual holding cost of stocking 1 spare:
  $$C_{\text{hold}} = r_h \cdot P_{\text{unit}}$$
  where $r_h$ is the annual holding rate (e.g., $0.20$) and $P_{\text{unit}}$ is the part purchase price.
* Expected annual stockout downtime cost if $S=0$:
  $$\mathbb{E}[C_{\text{downout}}] = \lambda \cdot L \cdot C_{\text{down}}$$
  where:
  * $\lambda$ = Annual failure rate of the component across the fleet (failures/year).
  * $L$ = Replenishment lead time (years or fraction of a year).
  * $C_{\text{down}}$ = Cost per unit time of asset downtime.
* **Economic Sparing Rule:** Stock at least 1 spare if the expected downtime loss exceeds the holding cost:
  $$\mathbb{E}[C_{\text{downout}}] > C_{\text{hold}} \implies \lambda \cdot L \cdot C_{\text{down}} > r_h \cdot P_{\text{unit}}$$
  Solving for the breakeven failure rate:
  $$\lambda_{\text{breakeven}} = \frac{r_h \cdot P_{\text{unit}}}{L \cdot C_{\text{down}}}$$

If the physical component failure rate $\lambda$ exceeds $\lambda_{\text{breakeven}}$, stocking the spare is economically positive for the enterprise, even if historical demand over the past three years was zero!

### 5.8 Step 8: Dynamic Governance and Obsolescence Triggers

Criticality is not static. Establish mandatory review triggers:
1. **Asset Life-Cycle Stage:** When an operating fleet approaches retirement, downgrade criticality from Vital to Desirable, permitting parts harvesting (cannibalization) to avoid end-of-life inventory write-offs.
2. **Supplier Lead-Time Shifts:** If a supplier announces that a forging lead time has extended from 8 weeks to 42 weeks, immediately re-run Step 7; the part may cross the threshold into mandatory insurance sparing.
3. **Engineering Retrofits:** When an engineering change order installs a secondary bypass loop or redundant backup sensor, recalculate the conditional failure probability $\beta_f$, immediately demoting the component from Quadrant I to Quadrant IV.

---

## 6. Worked example

### 6.1 Setting: Regional Healthcare Imaging Network

MetroHealth Systems operates a network of 12 regional hospitals running a capital fleet of:
* **50 Magnetic Resonance Imaging (MRI) 3.0T Systems:** High-field scanners. Operational downtime cost rate: $C_{\text{down}}^{\text{MRI}} = \$15,000/\text{hour}$ (lost patient billing, outpatient rescheduling penalties, emergency trauma diversion).
* **100 Computed Tomography (CT) 128-Slice Scanners:** Emergency department workhorses. Operational downtime cost rate: $C_{\text{down}}^{\text{CT}} = \$6,000/\text{hour}$.

The network supply chain team currently manages inventory using standard ABC dollar-volume ranking. They are experiencing severe service breakdowns: an MRI in the primary trauma hospital was down for 9 days awaiting an $850 vacuum valve, while the central medical warehouse holds $1.4 million worth of spare cosmetic gantry fairings and unused patient couches.

We must implement a multi-criteria criticality classification across six representative SKUs and establish an auditable sparing policy.

```text
Candidate Spare Parts Portfolio:
======================================================================================================
SKU ID   Description                 Asset System   Unit Cost  Lead Time  Fleet MTBF  Redundancy (RBD)
------------------------------------------------------------------------------------------------------
SKU-A1   Cryogenic Pressure Valve    MRI Magnet     $850       14 days    80,000 hrs  None (SPOF, beta=1.0)
SKU-B2   Gradient RF Amplifier       MRI Imaging    $95,000    60 days    35,000 hrs  None (SPOF, beta=1.0)
SKU-C3   CT High-Speed X-Ray Tube    CT Scanner     $45,000    3 days     4,000 hrs   None (SPOF, beta=1.0)
SKU-D4   Patient Couch Actuator      CT Scanner     $4,200     7 days     25,000 hrs  Degraded (Manual crank)
SKU-E5   Optical Fiber Transceiver   MRI / CT Data  $120       2 days     50,000 hrs  Active 1+1 Dual Loop
SKU-F6   Cosmetic Gantry Cover Shell CT Scanner     $12,000    45 days    500,000 hrs Non-functional panel
======================================================================================================
Annual inventory holding cost rate: r_h = 0.20 (20% per annum).
Operating schedule: 16 hours/day, 365 days/year = 5,840 operating hours/asset/year.
```

---

### 6.2 Step-by-Step Multi-Criteria AHP Weighting

We establish four criteria for spare parts criticality:
1. **$C_1$: Downtime Cost Impact ($\$/\text{hr}$):** Direct financial and clinical loss.
2. **$C_2$: Clinical Safety & Regulatory Risk:** Patient safety hazard or radiation compliance breach.
3. **$C_3$: System Redundancy & Failure Effect ($\beta_f$):** Degree of backup protection.
4. **$C_4$: Replenishment Lead Time Vulnerability:** Supplier manufacturing and logistics delay.

The clinical engineering and supply chain leadership committee establishes the pairwise comparison matrix $\mathbf{A}$:

$$\mathbf{A} = \begin{pmatrix} 
1 & 2 & 3 & 5 \\
1/2 & 1 & 2 & 4 \\
1/3 & 1/2 & 1 & 3 \\
1/5 & 1/4 & 1/3 & 1 
\end{pmatrix}$$

#### Step A: Normalize Matrix and Compute Weights

1. Sum of each column in $\mathbf{A}$:
   * Column 1: $1 + 0.50 + 0.333 + 0.20 = 2.0333$
   * Column 2: $2 + 1 + 0.50 + 0.25 = 3.7500$
   * Column 3: $3 + 2 + 1 + 0.333 = 6.3333$
   * Column 4: $5 + 4 + 3 + 1 = 13.0000$
2. Divide each entry by its column sum to get $\bar{\mathbf{A}}$:
   $$\bar{\mathbf{A}} = \begin{pmatrix}
   0.4918 & 0.5333 & 0.4737 & 0.3846 \\
   0.2459 & 0.2667 & 0.3158 & 0.3077 \\
   0.1639 & 0.1333 & 0.1579 & 0.2308 \\
   0.0984 & 0.0667 & 0.0526 & 0.0769
   \end{pmatrix}$$
3. Compute Priority Weights vector $\mathbf{w}$ (row means):
   * $w_1 (\text{Downtime Cost}) = \frac{0.4918 + 0.5333 + 0.4737 + 0.3846}{4} = \frac{1.8834}{4} = \mathbf{0.4709}$
   * $w_2 (\text{Clinical Safety}) = \frac{0.2459 + 0.2667 + 0.3158 + 0.3077}{4} = \frac{1.1361}{4} = \mathbf{0.2840}$
   * $w_3 (\text{Redundancy}) = \frac{0.1639 + 0.1333 + 0.1579 + 0.2308}{4} = \frac{0.6859}{4} = \mathbf{0.1715}$
   * $w_4 (\text{Lead Time}) = \frac{0.0984 + 0.0667 + 0.0526 + 0.0769}{4} = \frac{0.2946}{4} = \mathbf{0.0736}$
   $$\sum w_j = 0.4709 + 0.2840 + 0.1715 + 0.0736 = 1.0000$$

#### Step B: Check Consistency Ratio ($CR$)

1. Compute product vector $\mathbf{A} \mathbf{w}$:
   $$\mathbf{A} \mathbf{w} = \begin{pmatrix} 1(0.4709) + 2(0.2840) + 3(0.1715) + 5(0.0736) \\ 0.5(0.4709) + 1(0.2840) + 2(0.1715) + 4(0.0736) \\ 0.3333(0.4709) + 0.5(0.2840) + 1(0.1715) + 3(0.0736) \\ 0.2(0.4709) + 0.25(0.2840) + 0.3333(0.1715) + 1(0.0736) \end{pmatrix} = \begin{pmatrix} 1.9214 \\ 1.1568 \\ 0.6913 \\ 0.2960 \end{pmatrix}$$
2. Divide elements of $\mathbf{A} \mathbf{w}$ by $\mathbf{w}$:
   * $\lambda_1 = 1.9214 / 0.4709 = 4.0803$
   * $\lambda_2 = 1.1568 / 0.2840 = 4.0732$
   * $\lambda_3 = 0.6913 / 0.1715 = 4.0309$
   * $\lambda_4 = 0.2960 / 0.0736 = 4.0217$
3. Compute $\lambda_{\max}$:
   $$\lambda_{\max} = \frac{4.0803 + 4.0732 + 4.0309 + 4.0217}{4} = 4.0515$$
4. Compute Consistency Index ($CI$):
   $$CI = \frac{\lambda_{\max} - m}{m - 1} = \frac{4.0515 - 4}{3} = \frac{0.0515}{3} = 0.01717$$
5. Compute Consistency Ratio ($CR$) using $RI = 0.90$ (for $m=4$):
   $$CR = \frac{CI}{RI} = \frac{0.01717}{0.90} = \mathbf{0.0191}$$
   **Result:** Since $CR = 0.0191 \ll 0.10$, the committee’s pairwise judgments are rigorously consistent and approved.

---

### 6.3 Scoring the Six Component SKUs

Each SKU is rated by clinical engineering on an objective $[0, 1.0]$ normalized scale across the four weighted criteria:

```text
Raw Engineering Attribute Scoring Matrix:
========================================================================================================
SKU ID   Description               C1: Downtime   C2: Safety/Reg  C3: Redundancy  C4: Lead Time   Unit Cost
                                   (Weight 0.471) (Weight 0.284)  (Weight 0.171)  (Weight 0.074)
--------------------------------------------------------------------------------------------------------
SKU-A1   Cryogenic Valve (MRI)     1.00 ($15k/hr) 0.90 (Helium)   1.00 (SPOF)     0.60 (14 days)  $850
SKU-B2   Gradient RF Amp (MRI)     1.00 ($15k/hr) 0.50 (Diagnostic)1.00 (SPOF)    1.00 (60 days)  $95,000
SKU-C3   CT X-Ray Tube (CT)        0.70 ($6k/hr)  0.60 (Trauma)   1.00 (SPOF)     0.20 (3 days)   $45,000
SKU-D4   Couch Actuator (CT)       0.30 ($2k/hr)  0.20 (Ergonomic)0.30 (Manual)   0.40 (7 days)   $4,200
SKU-E5   Fiber Transceiver (Dual)  0.10 (Bypass)  0.10 (Telecom)  0.05 (1+1 Act)  0.10 (2 days)   $120
SKU-F6   Gantry Cover (CT)         0.00 ($0/hr)   0.05 (Cosmetic) 0.00 (No effect)0.80 (45 days)  $12,000
========================================================================================================
```

#### Computing Composite Criticality Scores ($\text{CCS}_p = 100 \times \sum w_j s_{pj}$):

1. **SKU-A1 (Cryogenic Valve):**
   $$\text{CCS} = 100 \times [0.4709(1.00) + 0.2840(0.90) + 0.1715(1.00) + 0.0736(0.60)]$$
   $$\text{CCS} = 100 \times [0.4709 + 0.2556 + 0.1715 + 0.0442] = 100 \times 0.9422 = \mathbf{94.22} \implies \text{Vital (Tier 1)}$$

2. **SKU-B2 (Gradient RF Amplifier):**
   $$\text{CCS} = 100 \times [0.4709(1.00) + 0.2840(0.50) + 0.1715(1.00) + 0.0736(1.00)]$$
   $$\text{CCS} = 100 \times [0.4709 + 0.1420 + 0.1715 + 0.0736] = 100 \times 0.8580 = \mathbf{85.80} \implies \text{Vital (Tier 1)}$$

3. **SKU-C3 (CT X-Ray Tube):**
   $$\text{CCS} = 100 \times [0.4709(0.70) + 0.2840(0.60) + 0.1715(1.00) + 0.0736(0.20)]$$
   $$\text{CCS} = 100 \times [0.3296 + 0.1704 + 0.1715 + 0.0147] = 100 \times 0.6862 = \mathbf{68.62} \implies \text{Essential (Tier 2)}$$
   *(Note: Because CT X-ray tubes fail frequently, MTBF=4,000 hrs, clinical operations upgrades it via statutory override to Vital).*

4. **SKU-D4 (Patient Couch Actuator):**
   $$\text{CCS} = 100 \times [0.4709(0.30) + 0.2840(0.20) + 0.1715(0.30) + 0.0736(0.40)]$$
   $$\text{CCS} = 100 \times [0.1413 + 0.0568 + 0.0515 + 0.0294] = 100 \times 0.2790 = \mathbf{27.90} \implies \text{Desirable (Tier 3)}$$

5. **SKU-E5 (Optical Fiber Transceiver):**
   $$\text{CCS} = 100 \times [0.4709(0.10) + 0.2840(0.10) + 0.1715(0.05) + 0.0736(0.10)]$$
   $$\text{CCS} = 100 \times [0.0471 + 0.0284 + 0.0086 + 0.0074] = 100 \times 0.0915 = \mathbf{9.15} \implies \text{Desirable (Tier 3)}$$

6. **SKU-F6 (Gantry Cover Shell):**
   $$\text{CCS} = 100 \times [0.4709(0.00) + 0.2840(0.05) + 0.1715(0.00) + 0.0736(0.80)]$$
   $$\text{CCS} = 100 \times [0.0000 + 0.0142 + 0.0000 + 0.0589] = 100 \times 0.0731 = \mathbf{7.31} \implies \text{Desirable (Tier 3)}$$

---

### 6.4 Economic Breakeven Sparing Analysis for SKU-A1 vs. Traditional ABC

Let us examine the economic reality of **SKU-A1 (Cryogenic Valve)**:
* Traditional ABC Classification:
  * Annual fleet exposure = 50 MRIs $\times$ 5,840 hrs/yr = 292,000 operating hours.
  * Valve MTBF = 80,000 hours.
  * Expected annual fleet demand $\lambda_{\text{fleet}} = 292,000 / 80,000 = 3.65 \text{ valves/year}$.
  * Annual Spend = $3.65 \times \$850 = \mathbf{\$3,102.50/\text{year}}$.
  * In a hospital network spending $48 million annually on parts, $3,102 puts this part in the bottom 5% of spend (**Class C**). Under traditional ABC, zero safety stock is held at regional hospitals; parts are ordered on-demand when failure occurs.
* The Cost of Stockout:
  * Supplier replenishment lead time: $L = 14 \text{ days} = 336 \text{ hours}$.
  * If the valve fails and stock is zero, the MRI is grounded for 14 days awaiting shipment.
  * Downtime Loss = $336 \text{ hours} \times \$15,000/\text{hour} = \mathbf{\$5,040,000}$.
* The Breakeven Sparing Equation:
  $$C_{\text{hold}} = r_h \cdot P_{\text{unit}} = 0.20 \times \$850 = \$170/\text{year}$$
  $$\mathbb{E}[C_{\text{downout}}] = \lambda_{\text{fleet}} \cdot L \cdot C_{\text{down}} = 3.65 \times \left(\frac{14}{365}\right) \times (336 \times \$15,000) = 3.65 \times 0.038356 \times \$5,040,000 = \mathbf{\$705,600/\text{year}}$$
  $$\lambda_{\text{breakeven}} = \frac{r_h \cdot P_{\text{unit}}}{L \cdot C_{\text{down}}} = \frac{\$170}{(14/365) \cdot (336 \cdot \$15,000)} = \frac{170}{193,315} = \mathbf{0.000879 \text{ failures/year}}$$

Since actual fleet failure rate ($\lambda = 3.65$) is **4,152 times higher** than $\lambda_{\text{breakeven}}$, holding stock is an overwhelming economic necessity. Holding an extra 3 spare valves across the network costs $\$510/\text{year}$ in capital holding costs while eliminating $\$705,600$ in expected annual downtime catastrophe.

---

### 6.5 Final Policy Mapping Matrix

| SKU ID | Description | CCS Score | VED Class | Unit Cost | Quadrant | Sparing Policy & Stocking Location |
|---|---|---:|:---:|---:|:---:|---|
| **SKU-A1** | Cryogenic Valve | 94.2 | **V** | $850 | **Q-I (Insurance)** | Stock 1 unit at each of the 4 major hospital hub depots. 99.5% target availability. |
| **SKU-B2** | Gradient RF Amplifier | 85.8 | **V** | $95,000 | **Q-II (Strategic)** | Pool 2 units at central logistics hub; OEM 4-hour hot-standby courier contract. |
| **SKU-C3** | CT X-Ray Tube | 68.6 | **V\*** | $45,000 | **Q-II (Strategic)** | Continuous rotables pipeline; forward buffer at central depot; fast 24-hr vendor swap. |
| **SKU-D4** | Patient Couch Actuator | 27.9 | **D** | $4,200 | **Q-III (Commercial)** | Zero network inventory. Order-on-demand from domestic supplier (7-day lead time). |
| **SKU-E5** | Optical Fiber Transceiver | 9.2 | **D** | $120 | **Q-IV (Bulk)** | Two-bin Kanban bin at all 12 local hospital technician cages. |
| **SKU-F6** | Gantry Cover Shell | 7.3 | **D** | $12,000 | **Q-III (Commercial)** | Purge from warehouse! Return to OEM or scrap. Zero stocking authorized. |

*\*Upgraded to Vital via clinical trauma protocol override.*

---

## 7. Code, query, or calculation

The code below implements the complete Analytic Hierarchy Process (AHP) matrix solver, validates the Consistency Ratio ($CR$), and executes the multi-attribute policy classification engine.

```python
"""
Multi-Criteria Spares Criticality & AHP Sparing Policy Engine
Implements:
  1. Saaty's Analytic Hierarchy Process (AHP) pairwise matrix solver.
  2. Eigenvalue max estimation and Consistency Ratio (CR) validation.
  3. Multi-attribute scoring and Quadrant Sparing Policy routing.
"""

import numpy as np
from typing import Dict, List, Tuple

# Random Consistency Index (RI) lookup table for matrix dimensions 1 to 10
RI_TABLE = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24, 7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49}

def solve_ahp_weights(pairwise_matrix: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    Computes priority weights, lambda_max, and Consistency Ratio from a pairwise matrix.
    """
    n = pairwise_matrix.shape[0]
    if n != pairwise_matrix.shape[1]:
        raise ValueError("Pairwise matrix must be square.")
    
    # Column normalization method
    col_sums = pairwise_matrix.sum(axis=0)
    norm_matrix = pairwise_matrix / col_sums
    weights = norm_matrix.mean(axis=1)
    
    # Calculate lambda_max
    weighted_sum_vector = np.dot(pairwise_matrix, weights)
    lambda_ratios = weighted_sum_vector / weights
    lambda_max = float(np.mean(lambda_ratios))
    
    # Consistency Index & Consistency Ratio
    ci = (lambda_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI_TABLE.get(n, 1.49)
    cr = ci / ri if ri > 0 else 0.0
    
    return weights, lambda_max, cr

def route_sparing_policy(ccs: float, unit_cost: float, is_safety_override: bool = False) -> Dict[str, str]:
    """
    Routes component to inventory policy quadrant based on CCS and Unit Cost.
    """
    effective_ccs = 100.0 if is_safety_override else ccs
    
    # Criticality Threshold: 70 = Vital/Critical, 40 = Essential, <40 = Desirable
    # Value Threshold: $5,000
    if effective_ccs >= 70.0:
        if unit_cost < 5000.0:
            quadrant = "Q-I: Insurance Spare"
            policy = "Forward Local Stock (99.5% Service Level)"
        else:
            quadrant = "Q-II: Strategic Critical"
            policy = "Centralized Pooling / Rotables Repair Loop (95.0% SL)"
    elif effective_ccs >= 40.0:
        if unit_cost < 5000.0:
            quadrant = "Q-IV (A): Essential Buffer"
            policy = "Regional Warehouse Stock (92.0% SL)"
        else:
            quadrant = "Q-III (A): Monitored Reorder"
            policy = "Central Pool / Fast Air Expedite (90.0% SL)"
    else:
        if unit_cost < 500.0:
            quadrant = "Q-IV (B): Bulk Hardware"
            policy = "Two-Bin Visual Kanban / VMI (Consignment)"
        else:
            quadrant = "Q-III (B): Non-Critical Commercial"
            policy = "Zero Stock / Order-on-Demand / Supplier SLA"
            
    return {"Quadrant": quadrant, "Policy": policy, "Effective_CCS": round(effective_ccs, 2)}

# --- Verification of Worked Example ---
if __name__ == "__main__":
    # Pairwise comparison matrix for [Downtime, Safety, Redundancy, LeadTime]
    A = np.array([
        [1.000, 2.000, 3.000, 5.000],
        [0.500, 1.000, 2.000, 4.000],
        [0.333, 0.500, 1.000, 3.000],
        [0.200, 0.250, 0.333, 1.000]
    ])
    
    weights, l_max, cr = solve_ahp_weights(A)
    print("==================================================================")
    print("AHP CRITICALITY WEIGHTING ENGINE")
    print("==================================================================")
    criteria_names = ["Downtime Cost", "Safety/Regulatory", "Redundancy Effect", "Lead Time Vulnerability"]
    for name, w in zip(criteria_names, weights):
        print(f"  {name:<26}: Weight = {w:.4f} ({w*100:.1f}%)")
    print(f"\n  Lambda Max = {l_max:.4f} | Consistency Ratio (CR) = {cr:.4f}")
    assert cr < 0.10, "AHP matrix is inconsistent!"
    print("  Status: AHP Matrix Mathematically Consistent (CR < 0.10)")
    print("==================================================================\n")
    
    # Portfolio of 6 test SKUs from Section 6
    portfolio = [
        {"sku": "SKU-A1", "name": "Cryogenic Valve",    "scores": [1.00, 0.90, 1.00, 0.60], "cost": 850.0,   "override": False},
        {"sku": "SKU-B2", "name": "Gradient RF Amp",    "scores": [1.00, 0.50, 1.00, 1.00], "cost": 95000.0, "override": False},
        {"sku": "SKU-C3", "name": "CT X-Ray Tube",      "scores": [0.70, 0.60, 1.00, 0.20], "cost": 45000.0, "override": True},
        {"sku": "SKU-D4", "name": "Couch Actuator",     "scores": [0.30, 0.20, 0.30, 0.40], "cost": 4200.0,  "override": False},
        {"sku": "SKU-E5", "name": "Fiber Transceiver",   "scores": [0.10, 0.10, 0.05, 0.10], "cost": 120.0,   "override": False},
        {"sku": "SKU-F6", "name": "Gantry Cover Shell", "scores": [0.00, 0.05, 0.00, 0.80], "cost": 12000.0, "override": False},
    ]
    
    print("SPARES CLASSIFICATION & POLICY ROUTING RESULTS:")
    for item in portfolio:
        raw_ccs = 100.0 * float(np.dot(weights, item["scores"]))
        routing = route_sparing_policy(raw_ccs, item["cost"], item["override"])
        print(f"\n{item['sku']} - {item['name']} (Cost: ${item['cost']:,.0f}):")
        print(f"  Raw CCS: {raw_ccs:.2f} | Override: {item['override']} | Effective CCS: {routing['Effective_CCS']}")
        print(f"  Routed Quadrant : {routing['Quadrant']}")
        print(f"  Assigned Policy : {routing['Policy']}")
```

### Expected Output
```text
==================================================================
AHP CRITICALITY WEIGHTING ENGINE
==================================================================
  Downtime Cost             : Weight = 0.4709 (47.1%)
  Safety/Regulatory         : Weight = 0.2840 (28.4%)
  Redundancy Effect         : Weight = 0.1715 (17.1%)
  Lead Time Vulnerability   : Weight = 0.0736 (7.4%)

  Lambda Max = 4.0515 | Consistency Ratio (CR) = 0.0191
  Status: AHP Matrix Mathematically Consistent (CR < 0.10)
==================================================================

SPARES CLASSIFICATION & POLICY ROUTING RESULTS:

SKU-A1 - Cryogenic Valve (Cost: $850):
  Raw CCS: 94.22 | Override: False | Effective CCS: 94.22
  Routed Quadrant : Q-I: Insurance Spare
  Assigned Policy : Forward Local Stock (99.5% Service Level)

SKU-B2 - Gradient RF Amp (Cost: $95,000):
  Raw CCS: 85.80 | Override: False | Effective CCS: 85.80
  Routed Quadrant : Q-II: Strategic Critical
  Assigned Policy : Centralized Pooling / Rotables Repair Loop (95.0% SL)

SKU-C3 - CT X-Ray Tube (Cost: $45,000):
  Raw CCS: 68.62 | Override: True | Effective CCS: 100.0
  Routed Quadrant : Q-II: Strategic Critical
  Assigned Policy : Centralized Pooling / Rotables Repair Loop (95.0% SL)

SKU-D4 - Couch Actuator (Cost: $4,200):
  Raw CCS: 27.90 | Override: False | Effective CCS: 27.90
  Routed Quadrant : Q-III (B): Non-Critical Commercial
  Assigned Policy : Zero Stock / Order-on-Demand / Supplier SLA

SKU-E5 - Fiber Transceiver (Cost: $120):
  Raw CCS: 9.15 | Override: False | Effective CCS: 9.15
  Routed Quadrant : Q-IV (B): Bulk Hardware
  Assigned Policy : Two-Bin Visual Kanban / VMI (Consignment)

SKU-F6 - Gantry Cover Shell (Cost: $12,000):
  Raw CCS: 7.31 | Override: False | Effective CCS: 7.31
  Routed Quadrant : Q-III (B): Non-Critical Commercial
  Assigned Policy : Zero Stock / Order-on-Demand / Supplier SLA
```

---

## 8. Data, standards, and regulatory context

In capital-intensive industries, spare parts classification is not merely an internal inventory optimization exercise; it is tightly regulated by international statutory safety codes, process safety management (PSM) regulations, and defense standards.

### 8.1 International Standards Landscape

1. **MIL-STD-1629A (Procedures for Performing a Failure Mode, Effects, and Criticality Analysis):**
   * *Origin:* United States Department of Defense.
   * *Core Mechanism:* Defines the formal mathematical methodology for Task 102 (Criticality Analysis), establishing the formula for Criticality Number $C_r$ and the Criticality Matrix ($C_r$ vs. Severity Category I to IV).
   * *Regulatory Impact:* Mandatory for defense materiel provisioning and government aerospace contract fulfillment.

2. **SAE JA1011 / JA1012 (Evaluation Criteria for Reliability-Centered Maintenance Processes):**
   * *Scope:* Global aerospace, automotive, and industrial standard defining what an organization must do to claim its maintenance program conforms to classical RCM (Nowlan and Heap lineage).
   * *Mandatory Principles:* RCM mandates that failure consequences must be classified into four strictly partitioned categories before considering maintenance tasks or spares:
     1. Hidden failure consequences (failures with no direct indication to operating crew).
     2. Safety and environmental consequences.
     3. Operational consequences (production losses).
     4. Non-operational consequences (direct repair cost only).
   * *Supply Chain Implication:* Sparing policies must directly mirror these four consequence categories; stocking a part to protect against an unverified non-operational consequence is non-compliant.

3. **IEC 60812:2018 (Failure Modes and Effects Analysis - FMEA and FMECA):**
   * *Scope:* International Electrotechnical Commission standard governing industrial machinery and electronics.
   * *Guidance on Criticality:* Explicitly warns against multiplying ordinal scales (RPN) for quantitative decision-making; endorses qualitative matrix mapping or probabilistic risk integration.

4. **OSHA 29 CFR 1910.119 (Process Safety Management of Highly Hazardous Chemicals):**
   * *Statutory Authority:* US Occupational Safety and Health Administration.
   * *Section (j) Mechanical Integrity:* Mandates that operators of chemical plants and refineries maintain an auditable Critical Equipment List (CEL). Replacement parts for equipment on the CEL must meet strict quality assurance standards and be verified for compatibility prior to installation.
   * *Legal Implication:* Stocking non-OEM or uncertified counterfeit commercial parts for a CEL-designated valve or pump is a federal violation punishable by criminal fines and facility shutdown.

5. **FAA / EASA Master Minimum Equipment List (MMEL) & Configuration Maintenance Procedures (CMP):**
   * *Aviation Governance:* Regulates civil passenger aircraft dispatch. An airline may only dispatch an aircraft with inoperative equipment if that equipment is explicitly listed in the MMEL under specific relief categories (Category A = specific time limit; Category B = 3 consecutive calendar days; Category C = 10 consecutive calendar days; Category D = 120 calendar days).
   * *Supply Chain Coupling:* Parts supporting "No-Go" systems (not on the MMEL) must be positioned at every line-maintenance outstation; Category C and D parts are pooled centrally at the main maintenance base.

---

## 9. Industry practice and case evidence

### 9.1 Offshore Oil & Gas: The NORSOK Z-008 Classification Standard

In the North Sea, operating under extreme marine weather conditions, Norwegian offshore operators (Equinor, Aker BP) operate under the **NORSOK Z-008 standard (Risk-Based Maintenance and Maintenance Consequences)**.

* **The Problem:** Historical warehousing on offshore oil rigs was physically constrained by platform deck space and structural crane load limits. Operators were flying parts out via expensive helicopter charters while rig warehouses were overflowing with redundant Class A and B parts.
* **The NORSOK Solution:** NORSOK Z-008 mandates a rigorous consequence classification dividing equipment into High, Medium, and Low Risk based on a 5$\times$5 matrix crossing Probability with HSE (Health, Safety, Environment) and Production Consequences.
* **Sparing Cutover:** Equinor reclassified over 250,000 spares across its offshore platforms. Parts with zero HSE impact and available standby redundancy ($1+1$) were purged from platform stockrooms and consolidated into onshore supply bases (Stavanger, Bergen). Platform inventory weight dropped by 34%, while helicopter emergency hot-shots were reduced by 45% because platform bins were dedicated exclusively to Quadrant I insurance spares (seals, sensors, relief valves).

### 9.2 Commercial Nuclear Power: Safety-Related vs. Balance-of-Plant Provisioning

Nuclear generating stations (operated by utilities such as Constellation Energy or EDF) operate under the strictest inventory classification rules in the world.

* **Regulatory Bifurcation:** The spare parts catalog is legally divided into **Safety-Related (SR)** and **Balance-of-Plant (BOP)**.
* **Safety-Related Requirements:** A safety-related spare (e.g., an emergency core cooling pump seal or backup diesel generator relay) requires an unbroken Certificate of Conformance (CoC), nuclear pedigree tracing, and 10 CFR 50 Appendix B quality assurance audits.
* **Economic Reality:** A standard commercial ball bearing costs $60; the identical bearing procured with nuclear Safety-Related qualification and destructive batch testing costs $4,800.
* **Supply Chain Strategy:** Utilities use multi-utility sparing consortia (such as the Utilities Service Alliance / Pooled Inventory Management program). Member nuclear plants share ownership of ultra-high-cost, long-lead safety spares (e.g., spare main transformer, reactor coolant pump motors), holding them in a climate-controlled bunker in Memphis, Tennessee, with dedicated multi-modal transport rights, cutting individual utility inventory holding costs by over $30 million per reactor.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign / Red Flag | Underlying Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **The Pareto Spend Illusion** | Warehouse holds millions in slow-moving structural parts while assets sit idle waiting for $50 seals. | Applying standard ABC inventory classification based on annual dollar spend instead of stockout downtime cost. | Crippling asset downtime despite high balance-sheet inventory value. | Transition immediately to 2D Criticality-Value matrix segmentation. |
| 2 | **Subjective VED Inflation ("Cry Wolf")** | Maintenance technicians classify 85% of all parts in the ERP catalog as "Vital". | Absence of objective, auditable operational and safety criteria; fear of being blamed for downtime. | Budget exhaustion; inability to prioritize true critical parts; planners ignore the rating entirely. | Establish algorithmic AHP scoring with mandatory FMECA redundancy and downtime rate inputs. |
| 3 | **Ordinal RPN Multiplication Error** | Safety stocks sized directly from FMEA Risk Priority Numbers ($S \times O \times D$). | Multiplying ordinal scales yields arbitrary, non-monotonic values that ignore replenishment lead time and unit cost. | Misallocated safety stocks; extreme over-stocking of low-severity frequent events. | Replace RPN with quantitative FMECA Criticality Numbers ($C_r$) and economic breakeven models. |
| 4 | **Redundancy Ignorance** | Identical high safety stock held for primary and standby pumps. | Catalog criticality is assigned at the component SKU level rather than the asset functional location level. | Unnecessary capital tie-up buffering components protected by fully automatic failover redundancy. | Link ERP spare parts records to functional equipment tags in the CMMS asset register. |
| 5 | **Lead-Time Vulnerability Blindness** | A non-critical part with a 52-week specialized casting lead time is stocked at zero units. | Criticality analysis evaluated only downtime consequence and ignored catastrophic supplier lead-time exposure. | When the part eventually fails, the asset remains derated or idle for an entire year awaiting supplier casting. | Incorporate Lead Time Vulnerability ($L$) as an independent axis in multi-criteria scoring. |
| 6 | **Static Criticality Decay** | Decoupled equipment fleets nearing retirement retain high "Vital" stock targets. | Criticality ratings are treated as "set-and-forget" engineering masters that are never reviewed during asset sunset. | Massive write-offs of expensive spares when the parent asset is decommissioned. | Institute dynamic lifecycle governance that automatically downgrades criticality 24 months prior to asset retirement. |
| 7 | **Safety-Override Bypass** | Sparing algorithm cancels reorder of a toxic gas alarm sensor because annual spend is under $200. | Optimization algorithms minimizing inventory holding costs without hard safety constraints. | Statutory safety violations; catastrophic worker exposure; regulatory facility shutdown. | Implement absolute "Safety Hard Overrides" that force 99.5% service levels regardless of cost. |
| 8 | **Single-Point-of-Failure (SPOF) Masking** | An asset thought to have $1+1$ redundancy trips when an unrecorded shared power rail or cooling line fails. | Flawed Reliability Block Diagram (RBD) failing to identify common-cause dependency. | Simultaneous failure of both primary and standby units, catching the supply chain completely unbuffered. | Conduct rigorous common-cause failure analysis during FMECA audits. |
| 9 | **Cannibalization Distortion of Criticality** | A part marked "Desirable" is frequently harvested from offline assets by field mechanics. | ERP logs zero emergency orders because mechanics bypass the supply chain via field cannibalization. | Latent damage accumulates across the fleet; asset rebuild times explode unpredictably. | Audit CMMS work order teardown records to detect clandestine cannibalization patterns. |
| 10 | **Consortium Free-Rider Failure** | An operator relies on a shared joint-industry spares pool, but finds the pool empty when emergency strikes. | Multiple operators experience correlated environmental failures (e.g., regional hurricane) simultaneously. | Extended asset outage because the pooled spare was already claimed by a competitor. | Model joint-probability pooling demand and establish contractually guaranteed pool replenishment lead times. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks If It Fails | How to Test / Validate |
|---|---|---|---|
| **Quantifiable Downtime Hourly Cost ($C_{\text{down}}$)** | Establishes the economic cost of stockout against which holding costs are balanced. | If downtime cost is unknown or wildly overstated, every part appears to require immediate local stocking. | Reconcile downtime rate models with corporate financial accounting and marginal contribution margin statements. |
| **Accurate Reliability Block Diagrams (RBD)** | Determines whether system redundancy ($\beta_f$) protects against component failure. | If hidden common-cause failures exist, redundancy is illusory, and stockout halts the system unexpectedly. | Review engineering P&IDs for shared utilities (power, lube oil, cooling water, instrumentation air). |
| **Monotonic Component Degradation** | Assumes failure modes identified in FMECA reflect actual operational wear. | If unpredicted environmental or operational stresses occur, unmodeled failure modes emerge in the field. | Compare CMMS work order failure codes against the baseline FMECA failure mode library. |
| **Independence of Pairwise AHP Criteria** | Saaty’s AHP assumes criteria are mutually independent (no feedback between levels). | If strong inner dependencies exist (e.g., safety risk dictates lead time), standard AHP generates distorted weights. | Check correlation between criteria scores; if significant dependencies exist, transition to Analytic Network Process (ANP). |
| **Stable Supplier Replenishment Lead Times** | Sparing breakeven models assume lead time $L$ is a known, bounded parameter. | In macroeconomic supply crunches, lead times double, converting an adequately buffered part into a stockout crisis. | Track supplier promised vs. actual delivery lead times dynamically in the procurement ERP. |
| **Form-Fit-Function (FFF) Substitution Validity** | Assumes non-critical parts cannot be substituted for critical parts. | If technicians can safely modify or substitute commercial hardware in emergencies, criticality is lower than modeled. | Maintain auditable engineering interchangeability and substitution tables in the PLM/ERP system. |
| **Auditable Maintenance Logging** | Sparing models require knowing whether a part replacement was scheduled or emergency. | If maintenance orders do not distinguish PM from emergency breakdown, $C_r$ calculations are corrupted. | Enforce mandatory CMMS work order closure classification rules for maintenance technicians. |
| **Non-Zero Component Salvage Value** | Multi-echelon pooling models assume high-value critical parts can be refurbished in a repair loop. | If failed parts are damaged beyond repair (scrap rate = 100%), the rotables pooling model collapses into consumable reorder. | Track historical scrap rates ($\text{BER} = \text{Beyond Economic Repair}$) for all high-value components. |

---

## 12. Exercises

### Exercise 1 — The Flaw of Ordinal RPN vs. Quantitative Criticality
**Type:** Calculation & Methodological Critique  
**Time:** 45 minutes  
**Context:** An engineering team conducted an FMEA on a deepwater offshore drilling blowout preventer (BOP). They evaluated three components using standard 1-to-10 ordinal scales ($S, O, D \in [1, 10]$):
* **Component 1 (Hydraulic Pilot Control Solenoid):**
  * Severity $S = 10$ (Catastrophic: loss of well control).
  * Occurrence $O = 2$ (Very low: $\lambda = 1.0 \times 10^{-6} \text{ failures/hr}$).
  * Detection $D = 2$ (High: continuous electrical telemetry).
* **Component 2 (Position Feedback Potentiometer):**
  * Severity $S = 4$ (Moderate: asset continues, minor diagnostic alarm).
  * Occurrence $O = 5$ (Moderate: $\lambda = 5.0 \times 10^{-5} \text{ failures/hr}$).
  * Detection $D = 4$ (Moderate: verified during monthly test).
* **Component 3 (Hydraulic Accumulator Bladder):**
  * Severity $S = 7$ (Major: loss of secondary emergency pressure reserve).
  * Occurrence $O = 3$ (Low: $\lambda = 5.0 \times 10^{-6} \text{ failures/hr}$).
  * Detection $D = 2$ (High: automated pressure drop monitoring).

**Deliverables:**
1. Compute the traditional Risk Priority Number ($\text{RPN} = S \times O \times D$) for all three components and rank them from highest to lowest priority.
2. The plant inventory manager enforces a rule: "Only parts with $\text{RPN} \ge 50$ receive safety stock." Identify the catastrophic operational defect in this rule.
3. Compute the quantitative Criticality Number ($C_r$) for Component 1 and Component 2 over a 1-year mission ($t = 8,760 \text{ hours}$), assuming failure mode ratios $\alpha = 1.0$ and conditional system failure probabilities $\beta_{f,1} = 1.0$ (SPOF) and $\beta_{f,2} = 0.10$ (redundant channel).
4. Contrast the RPN ranking against the $C_r$ ranking and write a 1-paragraph policy critique for the supply chain director.

---

### Exercise 2 — Multi-Criteria AHP Spares Matrix Construction
**Type:** Mathematical Application & Matrix Analysis  
**Time:** 60 minutes  
**Context:** A freight rail locomotive operator is designing an AHP decision model to prioritize rolling stock spare parts across three criteria:
* $C_1$: **Dispatch Impact (Downtime Penalty)**
* $C_2$: **Safety & Track Derailment Risk**
* $C_3$: **Supplier Lead Time Exposure**

The chief mechanical officer and procurement director provide the following pairwise comparison assessments:
* Dispatch Impact is considered moderately more important than Safety ($a_{12} = 3$).
* Dispatch Impact is considered strongly more important than Lead Time ($a_{13} = 5$).
* Safety is considered moderately more important than Lead Time ($a_{23} = 2$).

**Deliverables:**
1. Construct the complete $3 \times 3$ pairwise comparison matrix $\mathbf{A}$, ensuring reciprocity ($a_{kj} = 1 / a_{jk}$).
2. Normalize the matrix using the column-normalization method and calculate the priority weights vector $\mathbf{w} = [w_1, w_2, w_3]^T$.
3. Compute the maximum eigenvalue $\lambda_{\max}$.
4. Compute the Consistency Index ($CI$) and Consistency Ratio ($CR$) using $RI = 0.58$ for $m=3$. State whether the comparison matrix is accepted under the $CR < 0.10$ threshold.

---

### Exercise 3 — Economic Breakeven Sparing for Industrial Gas Turbine Blades
**Type:** Financial & Sparing Optimization  
**Time:** 45 minutes  
**Context:** A combined-cycle power plant operates four 250 MW gas turbines. A specialized stage-1 turbine nozzle segment set costs $P_{\text{unit}} = \$180,000$.
* The plant operates in base-load power generation earning a marginal contribution of $\$60/\text{MWh}$. Total plant downtime cost when a turbine is offline is:
  $$C_{\text{down}} = 250 \text{ MW} \times \$60/\text{MWh} = \$15,000/\text{hour}$$
* The nozzle supplier manufactures these sets to order with a specialized forging lead time of $L = 26 \text{ weeks} = 4,368 \text{ hours}$.
* Fleet historical reliability data indicates that nozzle segment sets suffer foreign object damage and thermal crack failures across the four turbines at an aggregate fleet rate of $\lambda_{\text{fleet}} = 0.15 \text{ failures/year}$ (an average of one failure every 6.67 years).
* The power utility’s annual inventory holding cost rate is $r_h = 0.18$ (18% per year).

**Deliverables:**
1. Compute the annual inventory holding cost ($C_{\text{hold}}$) of stocking 1 spare nozzle set.
2. Compute the expected annual downtime catastrophe loss ($\mathbb{E}[C_{\text{downout}}]$) if the utility holds zero spare nozzles in stock ($S=0$).
3. Calculate the economic breakeven failure rate ($\lambda_{\text{breakeven}}$) above which holding a spare becomes profitable.
4. Provide a definitive stocking recommendation with return-on-investment (ROI) justification for the executive investment committee.

---

### Exercise 4 — Designing a Criticality Transition Rule for Asset Sunsetting
**Type:** Policy & Architectural Design  
**Time:** 45 minutes  
**Context:** A defense logistics command is managing the phase-out of an armored personnel carrier fleet over a 5-year decommissioning horizon. The fleet size will decrease from 500 vehicles in Year 1 to 350 in Year 2, 200 in Year 3, 50 in Year 4, and 0 in Year 5.
* Currently, 140 parts in the catalog are classified as "Vital / Quadrant I (Insurance Spares)", commanding automatic buffer replenishment.
* Historically, when the command retired previous vehicle classes, over $42 million in brand-new spare parts were scrapped upon vehicle retirement because stocking policies were never changed.

**Deliverables:**
1. Design a multi-stage **Criticality Transition Framework** that programmatically demotes parts through VED and Policy Quadrants as the fleet size drops.
2. Define the exact quantitative trigger rules based on Remaining Fleet Life (RFL) and Component Lead Time ($L$).
3. Formulate the mathematical conditions under which **field cannibalization (controlled parts harvesting)** from decommissioned hulls is officially authorized as a formal substitute for purchasing new factory spares.

---

### Exercise 5 — Diagnostic Failure Mode and Redundancy Audit
**Type:** Applied Case Critique  
**Time:** 40 minutes  
**Context:** An offshore crude oil export pumping skid contains two 100% capacity centrifugal pumps arranged in a parallel $1+1$ active redundancy configuration (Pump Alpha and Pump Bravo). The planning analyst reviewed the CMMS and noted: "Since the pumps are 100% redundant, neither pump is a single point of failure. Therefore, all 48 internal spare parts (bearings, seals, impellers, couplings) are classified as Non-Critical (Quadrant IV) with zero safety stock."
Three weeks later, an electrical fire on the main skid wiped out both pumps simultaneously, halting crude export for 11 days.

**Deliverables:**
1. Identify the fatal analytical error in the analyst's redundancy evaluation.
2. Categorize the failure using reliability engineering taxonomy (independent failure vs. common-cause failure).
3. Redesign the spares classification audit checklist to prevent this defect from recurring in future plant reviews.

---

## 13. Self-check answers

### 13.1 Answer to Exercise 1

#### 1. RPN Calculation and Ranking:
* **Component 1 (Pilot Solenoid):** $\text{RPN}_1 = 10 \times 2 \times 2 = \mathbf{40}$
* **Component 2 (Potentiometer):** $\text{RPN}_2 = 4 \times 5 \times 4 = \mathbf{80}$
* **Component 3 (Accumulator Bladder):** $\text{RPN}_3 = 7 \times 3 \times 2 = \mathbf{42}$

**RPN Priority Ranking:**
$$\text{Rank 1: Component 2 (RPN = 80)} > \text{Rank 2: Component 3 (RPN = 42)} > \text{Rank 3: Component 1 (RPN = 40)}$$

#### 2. Critique of the RPN $\ge 50$ Sparing Rule:
Under the $\text{RPN} \ge 50$ threshold, **Component 1 is denied safety stock** because its RPN is 40. Yet Component 1 has a Severity of 10 (catastrophic well blowout risk!). Conversely, Component 2 receives safety stock because its RPN is 80, despite having a minor Severity of 4 (non-critical alarm). 
The rule creates a lethal vulnerability: it prioritizes frequent minor nuisances over catastrophic life-safety events.

#### 3. Quantitative Criticality Number ($C_r$) Calculation:
Over $t = 8,760 \text{ hours}$:
* **Component 1:**
  $$C_{r,1} = \lambda_1 \cdot t \cdot \alpha_1 \cdot \beta_{f,1} = (1.0 \times 10^{-6}) \times 8,760 \times 1.0 \times 1.0 = \mathbf{0.00876 \text{ critical failures/year}}$$
* **Component 2:**
  $$C_{r,2} = \lambda_2 \cdot t \cdot \alpha_2 \cdot \beta_{f,2} = (5.0 \times 10^{-5}) \times 8,760 \times 1.0 \times 0.10 = \mathbf{0.0438 \text{ failures/year}}$$
* However, weighting by the financial/safety loss severity ($S_1 = \$50,000,000$ well blowout vs. $S_2 = \$5,000$ diagnostic repair):
  $$\mathbb{E}[\text{Loss}_1] = 0.00876 \times \$50,000,000 = \mathbf{\$438,000/\text{year}}$$
  $$\mathbb{E}[\text{Loss}_2] = 0.0438 \times \$5,000 = \mathbf{\$219/\text{year}}$$

#### 4. Policy Critique:
RPN is mathematically invalid for inventory prioritization because it multiplies ordinal scales without economic or severity weighting. Component 1 presents an expected annual catastrophe exposure of $\$438,000$, whereas Component 2 presents an exposure of only $\$219$. Sparing policies must be governed by expected consequence loss and hard safety constraints, not ordinal RPN products.

---

### 13.2 Answer to Exercise 2

#### 1. Pairwise Comparison Matrix $\mathbf{A}$:
$$\mathbf{A} = \begin{pmatrix}
1 & 3 & 5 \\
1/3 & 1 & 2 \\
1/5 & 1/2 & 1
\end{pmatrix} = \begin{pmatrix}
1.000 & 3.000 & 5.000 \\
0.3333 & 1.000 & 2.000 \\
0.2000 & 0.500 & 1.000
\end{pmatrix}$$

#### 2. Normalization and Priority Weights Vector $\mathbf{w}$:
* Column Sums:
  * Col 1: $1.000 + 0.3333 + 0.2000 = 1.5333$
  * Col 2: $3.000 + 1.000 + 0.500 = 4.5000$
  * Col 3: $5.000 + 2.000 + 1.000 = 8.0000$
* Normalized Matrix $\bar{\mathbf{A}}$:
  $$\bar{\mathbf{A}} = \begin{pmatrix}
  1.000/1.5333 & 3.000/4.5000 & 5.000/8.0000 \\
  0.3333/1.5333 & 1.000/4.5000 & 2.000/8.0000 \\
  0.2000/1.5333 & 0.500/4.5000 & 1.000/8.0000
  \end{pmatrix} = \begin{pmatrix}
  0.6522 & 0.6667 & 0.6250 \\
  0.2174 & 0.2222 & 0.2500 \\
  0.1304 & 0.1111 & 0.1250
  \end{pmatrix}$$
* Priority Weights (Row Means):
  * $w_1 (\text{Dispatch Impact}) = \frac{0.6522 + 0.6667 + 0.6250}{3} = \frac{1.9439}{3} = \mathbf{0.6480} \text{ (64.8\%)}$
  * $w_2 (\text{Safety Risk}) = \frac{0.2174 + 0.2222 + 0.2500}{3} = \frac{0.6896}{3} = \mathbf{0.2299} \text{ (23.0\%)}$
  * $w_3 (\text{Lead Time}) = \frac{0.1304 + 0.1111 + 0.1250}{3} = \frac{0.3665}{3} = \mathbf{0.1221} \text{ (12.2\%)}$
  * Check: $\sum w_j = 0.6480 + 0.2299 + 0.1221 = 1.0000$.

#### 3. Compute $\lambda_{\max}$:
$$\mathbf{A} \mathbf{w} = \begin{pmatrix}
1.0(0.6480) + 3.0(0.2299) + 5.0(0.1221) \\
0.3333(0.6480) + 1.0(0.2299) + 2.0(0.1221) \\
0.2000(0.6480) + 0.5(0.2299) + 1.0(0.1221)
\end{pmatrix} = \begin{pmatrix}
0.6480 + 0.6897 + 0.6105 \\
0.2160 + 0.2299 + 0.2442 \\
0.1296 + 0.1150 + 0.1221
\end{pmatrix} = \begin{pmatrix}
1.9482 \\
0.6901 \\
0.3667
\end{pmatrix}$$
* Eigenvalue ratios:
  * $\lambda_1 = 1.9482 / 0.6480 = 3.0065$
  * $\lambda_2 = 0.6901 / 0.2299 = 3.0017$
  * $\lambda_3 = 0.3667 / 0.1221 = 3.0033$
$$\lambda_{\max} = \frac{3.0065 + 3.0017 + 3.0033}{3} = \mathbf{3.0038}$$

#### 4. Consistency Index ($CI$) and Consistency Ratio ($CR$):
$$CI = \frac{\lambda_{\max} - m}{m - 1} = \frac{3.0038 - 3}{2} = \frac{0.0038}{2} = 0.0019$$
$$CR = \frac{CI}{RI} = \frac{0.0019}{0.58} = \mathbf{0.00328}$$
**Conclusion:** Since $CR = 0.0033 \ll 0.10$, the pairwise comparison judgments exhibit outstanding mathematical consistency and are formally accepted.

---

### 13.3 Answer to Exercise 3

#### 1. Annual Inventory Holding Cost ($C_{\text{hold}}$):
$$C_{\text{hold}} = r_h \cdot P_{\text{unit}} = 0.18 \times \$180,000 = \mathbf{\$32,400/\text{year}}$$

#### 2. Expected Annual Downtime Loss with Zero Spares ($S=0$):
* Downtime per stockout event = Lead time $L = 26 \text{ weeks} = 4,368 \text{ hours}$.
* Downtime cost per failure event:
  $$C_{\text{event}} = 4,368 \text{ hrs} \times \$15,000/\text{hr} = \$65,520,000$$
* Expected annual catastrophic downtime loss:
  $$\mathbb{E}[C_{\text{downout}}] = \lambda_{\text{fleet}} \times C_{\text{event}} = 0.15 \text{ failures/yr} \times \$65,520,000 = \mathbf{\$9,828,000/\text{year}}$$

#### 3. Economic Breakeven Failure Rate ($\lambda_{\text{breakeven}}$):
$$\lambda_{\text{breakeven}} = \frac{C_{\text{hold}}}{C_{\text{event}}} = \frac{\$32,400}{\$65,520,000} = \mathbf{0.0004945 \text{ failures/year}}$$
*(Equates to 1 failure every 2,022 years!)*

#### 4. Executive Recommendation & ROI:
The actual fleet failure rate ($\lambda = 0.15$) is **303 times higher** than the breakeven rate. 
* Stocking 1 spare nozzle segment set costs $\$32,400/\text{year}$.
* It avoids an expected annual outage loss of $\$9,828,000$.
* Net Annual Financial Benefit = $\$9,828,000 - \$32,400 = \mathbf{\$9,795,600/\text{year}}$.
* **ROI:** $> 30,000\%$. The plant must stock 1 insurance spare set immediately in Quadrant I/II.

---

### 13.4 Rubric for Exercise 4 (Asset Sunsetting Framework)

A comprehensive answer must define a dynamic multi-stage transition protocol:

```text
Fleet Sunsetting Timeline:
================================================================================================
Phase     Time to EOL     Fleet Size   Criticality Sparing Policy Transition Rule
------------------------------------------------------------------------------------------------
Phase 1   > 36 months     500 -> 350   Normal Quadrant I/II Sparing; Full Reorder Authorized.
Phase 2   24 - 36 months  350 -> 200   Freeze all Last-Time Buys; Re-evaluate Lead-Time Risk.
Phase 3   12 - 24 months  200 -> 50    Demote Vital to Essential; Demote Essential to Desirable.
                                       Zero new POs for parts with Lead Time L > Remaining Life.
Phase 4   < 12 months     50 -> 0      Formal Cannibalization Protocol Activated. 
                                       Zero procurement. Decommissioned hulls become spares pool.
================================================================================================
```

**Quantitative Trigger Condition for Cannibalization Authorization:**
Field cannibalization is formally triggered for Part $p$ when:
$$t_{\text{lead}, p} > \text{Remaining Operating Life of Target Hull} \quad \text{OR} \quad Q_{\text{on-hand}} + Q_{\text{pipeline}} = 0 \text{ and decommissioned hull available}$$
Every harvesting action must generate a formal CMMS negative-issue transaction to maintain configuration control.

---

### 13.5 Answer to Exercise 5

1. **Analytical Defect:** The analyst assumed that because two physical pumps exist in parallel ($1+1$), the system possesses zero Single Points of Failure (SPOF). The analyst failed to evaluate the **Common-Cause Failure (CCF)** modes: shared electrical switchgear, common skid structural baseplates, shared lubrication skids, and common discharge header valves.
2. **Reliability Taxonomy:** This was a classic **Common-Cause Failure (CCF)**. Both pumps were rendered non-operational by a single root-cause event (electrical switchgear cabinet fire), completely bypassing component-level parallel redundancy.
3. **Audit Redesign Checklist:**
   * Does any common electrical feeder, MCC, or control transformer feed both trains?
   * Do both redundant lines share common suction/discharge piping, bypass manifolds, or manual isolation valves?
   * Are both redundant units located in the exact same physical fire/blast zone?
   * If any common-cause link exists, components required to restore the common infrastructure are designated **Vital (Quadrant I)** with dedicated on-site spares.

---

## 14. Interview questions

### Question 1: "Why does traditional ABC analysis fail when applied to spare parts, and how do you replace it?"

**Strong Answer:**  
"Traditional ABC analysis segments inventory solely on annual dollar velocity—annual demand multiplied by unit price. This fails catastrophically in spare parts because monetary volume does not correlate with operational consequence. A $50 Viton o-ring has near-zero annual spend, making it Class C in ABC analysis, yet its failure can trip an entire chemical plant, incurring $500,000 in downtime. Conversely, a $100,000 turbine casing is Class A, yet it rarely fails and may be non-critical if cosmetic or non-functional.

We replace ABC analysis with a two-dimensional **Criticality-Value Policy Matrix**. The vertical axis measures operational consequence (downtime cost per hour, safety risk, environmental hazard, system redundancy, and lead-time vulnerability, synthesized via AHP or FMECA). The horizontal axis measures unit cost and velocity. This isolates **Quadrant I Insurance Spares** (inexpensive but vital parts) where high service levels (99%+) are maintained via forward buffers, and **Quadrant II Strategic Spares** (expensive and vital parts) managed via centralized pooling and rotables repair loops."

**Weak Answer:**  
"ABC analysis is only for retail because spare parts have too many zeros. To fix it, you just do an ABC-XYZ analysis where XYZ measures how random the demand is."

---

### Question 2: "What is the difference between ordinal RPN in FMEA and the quantitative Criticality Number ($C_r$) in MIL-STD-1629A?"

**Strong Answer:**  
"In standard FMEA, the Risk Priority Number is calculated as $\text{RPN} = S \times O \times D$, where Severity, Occurrence, and Detection are qualitative ordinal numbers on a 1-to-10 scale. Multiplying ordinal numbers violates basic measurement theory because the intervals between ranks are not equal; an event with $S=10, O=1, D=1$ gets an RPN of 10, ranking lower than $S=2, O=5, D=2$ ($\text{RPN}=20$), which dangerously prioritizes minor frequent defects over catastrophic life-safety hazards.

In contrast, MIL-STD-1629A FMECA defines the quantitative Criticality Number as:
$$C_r = \sum (\lambda \cdot t \cdot \alpha \cdot \beta_f)$$
where $\lambda$ is the actual physical component failure rate, $t$ is operating mission duration, $\alpha$ is the failure mode ratio, and $\beta_f$ is the conditional probability of system loss given component failure. $C_r$ is a continuous, ratio-scale metric representing the expected number of catastrophic system failures per unit of operating exposure. It directly feeds probabilistic risk assessment and inventory stocking models."

**Weak Answer:**  
"RPN is what you calculate in design meetings by multiplying severity times occurrence times detection. Criticality number is just a different formula used by the military that uses failure rates."

---

### Question 3: "How does active hardware redundancy ($1+1$ or $2oo3$) impact component-level inventory stocking policy?"

**Strong Answer:**  
"Active hardware redundancy directly modifies the component's conditional system failure effect probability, $\beta_f$. 
* In a non-redundant system ($1oo1$), component failure causes instantaneous system functional loss ($\beta_f = 1.0$), forcing the part into **Vital / Quadrant I or II** status requiring immediate local or rapid-courier availability.
* In an active $1+1$ redundant configuration with automated failover, component failure causes an alarm, but the standby unit assumes full operational load without interrupting the plant ($\beta_f \approx 0.0$). 

This converts an emergency crash-stop into a planned repair grace period. The stocking policy shifts: instead of holding dedicated forward buffer stock at every machine site, the spare can be held centrally at a regional distribution center with a 24-to-48-hour delivery commitment, provided that the logistics delivery time (LDT) plus repair time is significantly less than the MTBF of the surviving redundant unit."

**Weak Answer:**  
"If you have redundant equipment, you don't need spare parts at all because if one breaks, the other one just keeps running."

---

### Question 4: "Under what conditions is it economically rational to stock an 'insurance spare' whose expected demand over the next three years is zero?"

**Strong Answer:**  
"It is rational when the expected annual catastrophic downtime loss exceeds the annual inventory holding cost:
$$\lambda_{\text{fleet}} \cdot L \cdot C_{\text{down}} > r_h \cdot P_{\text{unit}}$$
where $\lambda_{\text{fleet}}$ is the component failure rate, $L$ is the supplier replenishment lead time, $C_{\text{down}}$ is the hourly cost of asset downtime, $r_h$ is the holding rate, and $P_{\text{unit}}$ is the part cost.

Consider a customized turbine rotor forging costing $P_{\text{unit}} = \$200,000$ with a 1-year lead time ($L = 8,760 \text{ hrs}$) on a power plant where downtime costs $C_{\text{down}} = \$10,000/\text{hr}$. The annual holding cost at 20% is $\$40,000/\text{year}$. If a failure occurs and stock is zero, the plant remains down for an entire year, losing $\$87.6 \text{ million}$. 
The breakeven failure rate is:
$$\lambda_{\text{breakeven}} = \frac{\$40,000}{\$87,600,000} = 0.000456 \text{ failures/year}$$
Even if the rotor only fails once every 1,000 years ($\lambda = 0.001$), holding the spare yields an expected net benefit of $\$47,600$ per year. In insurance spares, you are stocking against lead-time consequence exposure, not historical demand frequency."

**Weak Answer:**  
"You stock it if the boss tells you it's critical. If a part has zero demand, normal inventory models say stock zero, but for insurance spares you just buy one and keep it on the shelf forever."

---

### Question 5: "What is the Analytic Hierarchy Process (AHP), and why must you verify the Consistency Ratio ($CR < 0.10$)?"

**Strong Answer:**  
"The Analytic Hierarchy Process (AHP) is a multi-criteria decision-making method developed by Thomas Saaty. It decomposes a complex decision into a hierarchy of criteria, utilizes pairwise comparison matrices on a 1-to-9 scale, and extracts priority weights via the principal eigenvector of the matrix ($\mathbf{A} \mathbf{w} = \lambda_{\max} \mathbf{w}$).

Verifying the Consistency Ratio ($CR = CI / RI$) is mandatory because human subjective judgments are prone to transitivity violations. For example, an expert might judge Criterion A twice as important as B, B twice as important as C, but then inconsistently state that C is more important than A. If $CR \ge 0.10$, the degree of randomness and contradiction in the pairwise matrix is unacceptably high, invalidating the derived weights. Checking that $CR < 0.10$ guarantees that the resulting spares prioritization is mathematically defensible and logically coherent."

**Weak Answer:**  
"AHP is a way to weight different goals. The Consistency Ratio just checks if the math is correct, and if it's over 10% it means someone made a typo in the matrix."

---

### Question 6: "Explain the VED classification scheme. How do you prevent 'VED creep' where every part is classified as Vital?"

**Strong Answer:**  
"VED classifies spare parts into:
* **Vital (V):** Parts whose failure causes immediate, unrecoverable stoppage of the primary asset, violates statutory safety or environmental licenses, or creates an immediate life hazard.
* **Essential (E):** Parts whose failure causes performance derating, loss of secondary systems, or operational inefficiency, but where the machine can operate safely for a defined grace period.
* **Desirable (D):** Parts whose failure has negligible impact on production or safety (auxiliary, cosmetic, ergonomic).

'VED creep' occurs when field technicians mark 80–90% of parts as Vital out of fear of downtime recrimination. To prevent this, an organization must:
1. **Enforce Objective Gate Criteria:** A part cannot be Vital unless linked to an active Single Point of Failure (SPOF) on an asset designated Critical-1 in the master asset register.
2. **Require Regulatory/Safety Cross-Referencing:** Safety-vital status requires citing the specific statutory code (e.g., OSHA PSM, FAA MMEL, NORSOK).
3. **Decouple Criticality from Planner Performance:** Audit CMMS downtime work orders; if an asset continued running for 48 hours after a part failed, that part is programmatically stripped of its 'Vital' status."

**Weak Answer:**  
"VED stands for Vital, Essential, Desirable. You prevent creep by setting a rule that only 20% of the catalog can be Vital, so planners have to rank them and cut off the bottom 80%."

---

### Question 7: "What is an MMEL in aviation or maritime, and how does it directly govern forward stocking location (FSL) positioning?"

**Strong Answer:**  
"A Master Minimum Equipment List (MMEL) is an airworthiness authority approved document that establishes which instruments, systems, or structural subassemblies can be temporarily inoperative at flight dispatch, and under what operational restrictions.
* **Non-MMEL Items ('No-Go'):** Components not listed in the MMEL must be 100% operational for dispatch (e.g., primary flight controls, engine fuel pumps). If a No-Go part fails at an outstation, the aircraft is grounded (Aircraft On Ground / AOG).
* **MMEL Relief Items ('Go-If' / 'Go'):** Systems classified under relief categories (Category A, B, C, D) provide operating grace periods ranging from 3 to 120 calendar days.

This governs spares positioning: No-Go items must be placed in Forward Stocking Locations (FSLs) at all line-maintenance line stations to prevent AOG cancellations. Category C and D items, which permit 10 to 120 days of flight with the system inoperative, are pooled centrally at the main maintenance base and flown in during scheduled weekend overhauls."

**Weak Answer:**  
"MMEL is a list of all parts on a plane. If a part is on the list, you have to stock it at the airport. If it's not on the list, you don't need it."

---

### Question 8: "Why does the economic breakeven failure rate ($\lambda_{\text{breakeven}}$) scale inversely with replenishment lead time ($L$)?"

**Strong Answer:**  
"The breakeven formula is:
$$\lambda_{\text{breakeven}} = \frac{r_h \cdot P_{\text{unit}}}{L \cdot C_{\text{down}}}$$
Because lead time $L$ sits in the denominator, as lead time lengthens, the breakeven failure rate approaches zero.

Physically, lead time represents the **duration of vulnerability**. If a supplier can deliver an emergency replacement in 2 hours ($L = 2/8760 = 0.000228 \text{ years}$), the maximum downtime loss during a stockout is only 2 hours of lost production. But if the supplier requires a 40-week specialized manufacturing run ($L = 40/52 = 0.769 \text{ years}$), a stockout strands the capital asset for nine months. 

Because the potential catastrophe increases linearly with $L$, the failure frequency required to economically justify holding a spare drops dramatically. For ultra-long lead-time parts, an event that occurs once every 500 years still economically justifies holding a dedicated insurance spare."

**Weak Answer:**  
"Because if lead time is long, you have to wait longer to get the part, so you should buy more spares to be safe."

---

## 15. Key terms

- **Criticality:** The magnitude of operational, financial, safety, or environmental damage caused by a part stockout.
- **VED Classification:** A three-tier aftermarket taxonomy: Vital, Essential, Desirable.
- **ABC Analysis:** A traditional finished-goods inventory technique sorting items by annual monetary expenditure ($D \times P$).
- **Insurance Spare:** A high-consequence, low-probability part stocked to protect against catastrophic lead-time downtime.
- **Risk Priority Number (RPN):** The ordinal product of Severity, Occurrence, and Detection in classical FMEA.
- **Criticality Number ($C_r$):** The quantitative expected number of failure events causing system loss in FMECA (MIL-STD-1629A).
- **Single Point of Failure (SPOF):** A component whose independent failure causes immediate, complete system functional collapse.
- **Active Redundancy:** A system architecture where parallel units operate concurrently ($1oo2$, $2oo3$), absorbing component failure with zero downtime.
- **Standby Redundancy:** A configuration where a backup unit remains unpowered until engaged by an automated or manual switch.
- **Common-Cause Failure (CCF):** A single root-cause event (fire, flood, power loss) that simultaneously disables redundant units.
- **Analytic Hierarchy Process (AHP):** A multi-criteria decision method using pairwise comparisons and eigenvector weighting.
- **Consistency Ratio ($CR$):** The ratio measuring mathematical transitivity in an AHP matrix; must be $< 0.10$.
- **Mean Down Time (MDT):** Total elapsed non-operational time, including logistics delay time and active repair.
- **Logistics Delay Time (LDT):** The component of downtime spent awaiting spare parts delivery and technician arrival.
- **Master Minimum Equipment List (MMEL):** An aviation/marine regulatory specification of allowed inoperative equipment at dispatch.
- **Form-Fit-Function (FFF):** The physical, mounting, and performance interchangeability of an engineered replacement part.
- **SAE JA1011:** The global standard defining the mandatory criteria for Reliability-Centered Maintenance (RCM) processes.
- **OSHA 1910.119:** The US Process Safety Management standard governing mechanical integrity on critical chemical assets.
- **NORSOK Z-008:** The offshore oil and gas standard governing risk-based maintenance and consequence classification.
- **Breakeven Failure Rate ($\lambda_{\text{breakeven}}$):** The threshold component failure rate where holding cost equals expected stockout downtime loss.
- **Holding Cost Rate ($r_h$):** The annualized percentage carrying cost of inventory capital, warehousing, and obsolescence.
- **Downtime Cost Rate ($C_{\text{down}}$):** The marginal economic cash contribution lost per unit time of asset stoppage.
- **Rotables Pool:** A centralized inventory of high-value repairable assemblies cycled through a continuous repair loop.
- **Two-Bin Kanban:** A physical, visual replenishment mechanism deployed for low-cost, non-critical hardware.
- **Vendor-Managed Inventory (VMI):** An arrangement where the supplier monitors and maintains on-site stockroom inventory.

---

## 16. Sources

### Peer-reviewed and Academic — Grade A
1. **Saaty, T. L. (1990).** "How to make a decision: the analytic hierarchy process." *European Journal of Operational Research*, 48(1), 9-26. [DOI: 10.1016/0377-2217(90)90057-I]
2. **Huiskonen, J. (2001).** "Maintenance spare parts logistics: Special characteristics and strategic choices." *International Journal of Production Economics*, 71(1-3), 125-133. [DOI: 10.1016/S0925-5273(00)00112-2]
3. **Braglia, M., Grassi, A., & Montanari, R. (2004).** "Multi-attribute classification method for spare parts inventory management." *Journal of Quality in Maintenance Engineering*, 10(1), 55-65. [DOI: 10.1108/13552510410526875]
4. **Gajpal, P. P., Ganesh, L. S., & Rajendran, C. (1994).** "Criticality analysis of spare parts using the analytic hierarchy process." *International Journal of Production Economics*, 35(1-3), 293-297. [DOI: 10.1016/0925-5273(94)90095-4]
5. **Cavalieri, S., Garetti, M., Macchi, M., & Pinto, R. (2008).** "A decision-making framework for managing maintenance spare parts." *Production Planning & Control*, 19(4), 379-396. [DOI: 10.1080/09537280802034471]

### Standards and Regulatory Frameworks — Grade B
6. **Department of Defense (DoD). (1980).** *MIL-STD-1629A: Procedures for Performing a Failure Mode, Effects and Criticality Analysis*. Washington, D.C.: US Government Printing Office.
7. **Society of Automotive Engineers (SAE). (2009).** *SAE JA1011: Evaluation Criteria for Reliability-Centered Maintenance (RCM) Processes*. Warrendale: SAE International.
8. **International Electrotechnical Commission (IEC). (2018).** *IEC 60812:2018: Failure modes and effects analysis (FMEA and FMECA)*. Geneva: IEC.
9. **Standards Norway. (2017).** *NORSOK Z-008:2017: Risk-based maintenance and consequence classification*. Lysaker: Standards Norway.

### Industry Case Evidence and Technical Monographs — Grade C & D
10. **Nowlan, F. S., & Heap, H. F. (1978).** *Reliability-Centered Maintenance*. Report AD-A066579. San Francisco: United Airlines for Office of Assistant Secretary of Defense.

---

## 17. Links

- **Parent Module:** [`_index.md`](_index.md) (Module 31 — Aftermarket, Spare Parts, and Service Supply Chains)
- **Root Survey File:** [`../../../course/31_aftermarket_spare_parts_service.md`](../../../course/31_aftermarket_spare_parts_service.md)
- **Previous Topic:** [`01_installed-base-failure-forecast.md`](01_installed-base-failure-forecast.md) (Installed Base and Failure Demand)
- **Next Topic:** [`03_multi-echelon-sparing.md`](03_multi-echelon-sparing.md) (Multi-Echelon Sparing and METRIC)
- **Related Methods:**
  - [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
  - [`../../../course/14_procurement_supplier_risk_spend.md`](../../../course/14_procurement_supplier_risk_spend.md)
