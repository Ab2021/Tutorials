> Module 33 — Healthcare Provider and Life Sciences Operations
> Topic 02 of 09
> Estimated study time: 3.5 hours
> Prerequisites: Module 33 Topic 01 (The Provider Network and Item Master), Module 10 (Inventory Optimization and Multi-Echelon)
> Version 2026.5

# Par Levels and Clinical Replenishment

---

## 1. Why this topic exists

In a commercial retail distribution network or an industrial manufacturing facility, an inventory stockout results in a backorder, a delayed shipment, or at worst a lost customer transaction. In an acute-care hospital, an inventory stockout at the point of care represents an immediate clinical crisis. If a sterile central venous catheterization kit is missing from the clean utility room of an intensive care unit (ICU) during septic resuscitation, or if an ampule of norepinephrine is absent from an emergency department automated dispensing cabinet during refractory vasodilatory shock, the consequence is measured in hypoxic brain damage, unrecoverable cardiovascular collapse, or patient death.

Hospital point-of-use (POU) inventory management is governed by the **Periodic Automatic Replenishment (Par)** system. Historically established in the 1970s and 1980s as a manual paper-based counting routine, a "par level" represents the target inventory quantity that a clinical care area (nursing ward, surgical suite, catheterization laboratory, or trauma bay) should hold to satisfy clinical consumption between replenishment sweeps, buffered by safety stock against demand spikes and delivery delays.

The macroeconomic and clinical landscape of healthcare has fundamentally altered how par replenishment must function:

1. **The Structural Shift from Fee-for-Service to Value-Based Care:** Under historical cost-plus or fee-for-service reimbursement models, every bandage, IV bag, and catheter was a billable line item charged to the patient or insurer. Supplies were viewed as revenue drivers. Under modern prospective payment systems—such as Medicare Severity Diagnosis Related Groups (MS-DRGs) in the United States and Healthcare Resource Groups (HRGs) in the United Kingdom—hospitals receive fixed, bundled reimbursements per episode of care. Every medical supply consumed on the hospital floor is now a pure cost center. Over-stocking bloats hospital working capital and leads to massive expiration waste, while under-stocking causes procedural delays that extend length of stay (LOS), eroding hospital operating margins.
2. **The Behavioral Hoarding Feedback Loop ("Squirrel Stashing"):** When clinical staff experience stockouts of critical supplies, their rational coping behavior is to hoard. Registered nurses (RNs) and surgical technicians hide consumable kits, blood collection tubes, and dressings in ceiling tiles, unassigned treatment carts, personal locker cubbies, and uniform pockets. This hidden inventory breaks the supply chain's visibility. The enterprise resource planning (ERP) or warehouse management system (WMS) records zero consumption because the physical unit par remains un-depleted while the floor consumes the stash. Inventory optimization algorithms or materials technicians interpret this flatline as declining clinical demand and slash the official par level. When the unofficial hoard is finally exhausted, the unit experiences an abrupt, catastrophic stockout. Trust between clinicians and materials management collapses, triggering even more aggressive hoarding.
3. **Clinical Labor Diversion and Nursing Burnout:** Peer-reviewed health services research demonstrates that acute-care nurses spend between 45 and 90 minutes per 12-hour shift searching for supplies, borrowing items from neighboring clinical units, or waiting for emergency deliveries. In a 600-bed tertiary hospital employing 1,800 bedside nurses, 60 minutes of lost time per nurse per shift equates to 150,000 hours of diverted nursing labor annually—representing millions of dollars in squandered clinical capacity and exacerbating bedside burnout.
4. **Severe Physical Space Constraints and Infection Control:** Unlike a commercial warehouse with 30-foot vertical clear heights, hospital clean utility and medication rooms operate under extreme spatial and regulatory constraints. Clean utility square footage must comply with strict heating, ventilation, and air conditioning (HVAC) standards (positive air pressure, minimum air change rates per hour), fire code egress limitations (NFPA 99/101), and infection control prohibitions against dust-trapping porous surfaces. Every square foot allocated to excessive safety stock is real estate stripped from patient beds or clinical procedure rooms.
5. **Asymmetric Stockout Costs:** The economic cost of holding a $1.20 sterile IV flush syringe is negligible, whereas the total operational and clinical cost of running out—dispatching an emergency courier ("hot-shot runner"), diverting specialized clinical personnel, delaying physician-led procedures, or precipitating adverse clinical events—is astronomical. Sizing par levels using symmetric quadratic loss functions or standard commercial holding-to-shortage ratios fails catastrophically in clinical environments.

Modern healthcare networks resolve these tensions through two primary operational mechanisms: **Continuous Review Automated Dispensing Cabinets (ADCs)** (such as BD Pyxis, Omnicell XT) that capture real-time telemetry at the transaction level, and **Visual Two-Bin Kanban Systems** (often enabled by passive RFID or weight sensors) for fast-moving clinical consumables. Sizing and running these replenishment mechanisms requires rigorous stochastic modeling, empirical telemetry cleansing, and explicit alignment with clinical workflows.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Formulate** periodic review $(R, S)$ and continuous review $(s, S)$ inventory models customized for clinical point-of-use par locations under discrete Poisson and compound Negative Binomial demand.
2. **Extract and interpret** transaction telemetry from automated dispensing cabinets (ADCs), identifying discrepancy logs, null-pick events, stockout transactions, and replenishment cycle intervals.
3. **Dimension** physical two-bin visual Kanban systems for medical-surgical commodities, incorporating case-pack rounding, physical bin volume limits, and runner sweep lead times.
4. **Quantify** the mathematical dynamics of the nurse hoarding feedback loop, diagnosing variance-to-mean ratio anomalies and designing trust-restoring replenishment service-level agreements (SLAs).
5. **Construct** asymmetric clinical stockout penalty functions that integrate expedited courier fees, clinical labor diversion rates, surgical delay costs, and surrogate clinical criticality weights.
6. **Compute** dynamic, patient-census-adjusted and case-mix-adjusted par levels driven by real-time Electronic Health Record (EHR) Admission, Discharge, and Transfer (ADT) feeds.
7. **Solve** volumetric shelf-space allocation problems across heterogeneous clean utility supplies using constrained knapsack formulations.
8. **Establish** an auditable clinical replenishment operating rhythm that bridges Central Sterile Supply (SPD), Materials Management, and Nursing Unit Leadership.

---

## 3. The core mental model

Clinical inventory replenishment operates across two physical and technological topologies: **Automated Dispensing Cabinet (ADC) Continuous Telemetry** and **Visual Two-Bin RFID/Barcode Kanban Periodic Review**.

```text
===================================================================================================
                               HOSPITAL REPLENISHMENT TOPOLOGIES
===================================================================================================

[TOPOLOGY A: CONTINUOUS REVIEW TELEMETRY (ADCs)]         [TOPOLOGY B: PERIODIC REVIEW VISUAL 2-BIN KANBAN]
High-Value Meds, Controlled Substances, Critical Kits    High-Turnover Med-Surg Consumables (Flushes, Gauze)
─────────────────────────────────────────────────────    ───────────────────────────────────────────────────
                        │                                                         │
                        ▼                                                         ▼
       ┌─────────────────────────────────┐                       ┌─────────────────────────────────┐
       │   Automated Dispensing Cabinet  │                       │   Physical Two-Bin Flow Rack    │
       │   (Omnicell XT / BD Pyxis ES)   │                       │   (Front Primary / Rear Backup) │
       └────────────────┬────────────────┘                       └────────────────┬────────────────┘
                        │                                                         │
              Real-Time │ Transaction Telemetry                          Bin Empty│ Runner Sweep or
              Timestamp │ (DISP, WAST, DISC)                             Trigger  │ RFID Flag Flipped
                        ▼                                                         ▼
       ┌─────────────────────────────────┐                       ┌─────────────────────────────────┐
       │   ADC Central Data Console      │                       │   Visual / RFID Kanban Sensor   │
       │   • Continuous Inventory Tracking│                      │   • Periodic Sweep Scan (R)     │
       │   • Reorder Point Trigger (s)   │                       │   • Bin Depletion Signal        │
       └────────────────┬────────────────┘                       └────────────────┬────────────────┘
                        │                                                         │
                        └──────────────────────────┬──────────────────────────────┘
                                                   │
                                                   ▼
                                ┌────────────────────────────────────┐
                                │     PAR OPTIMIZATION & WMS         │
                                │  • Census Normalization (ADT Feed) │
                                │  • Dynamic Sizing: S = f(λ, R, L)  │
                                │  • Pick-Ticket Generation          │
                                └──────────────────┬─────────────────┘
                                                   │
                                                   ▼
                                ┌────────────────────────────────────┐
                                │   CENTRAL DISTRIBUTION / PHARMACY  │
                                │   • Central Store Pick             │
                                │   • Automated Guided Vehicle (AGV) │
                                │   • Materials Specialist Runner    │
                                └──────────────────┬─────────────────┘
                                                   │
                                                   ▼
                                ┌────────────────────────────────────┐
                                │   CLEAN UTILITY / MED ROOM RESTOCK │
                                │   • Drawer Restock / Blind Count   │
                                │   • Secondary Bin Rotated Forward  │
                                │   • Fresh Bin Placed in Reserve    │
                                └────────────────────────────────────┘
```

### 3.1 The Behavioral Hoarding Feedback Loop

When replenishment policies fail to account for lead-time variability or clinical spikes, a destructive socio-technical cycle emerges between bedside clinicians and hospital materials management.

```text
                          ┌────────────────────────────────────────┐
                          │         Clinical Stockout Event        │
                          │   (Missing CVC Kit / Flush Syringe)    │
                          └───────────────────┬────────────────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────────────┐
                          │          Clinician Anxiety             │
                          │  Fear of Patient Harm / Lost Time      │
                          └───────────────────┬────────────────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────────────┐
                          │   Informal Hoarding ("Squirrel Stash") │
                          │ Hiding Stock in Ceiling, Carts, Pockets│
                          └───────────────────┬────────────────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────────────┐
                          │     Distorted Systemic Telemetry       │
                          │ Official POU Registers Zero Consumption│
                          └───────────────────┬────────────────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────────────┐
                          │       Automated Par Slashing           │
                          │ ERP Cuts Par: "Item Has Low Turnover"  │
                          └───────────────────┬────────────────────┘
                                              │
                                              ▼
                          ┌────────────────────────────────────────┐
                          │    Severe Recurrent Stockout Crisis    │
                          │ Hoard Depleted -> Unit Has Zero Stock  │
                          └───────────────────┬────────────────────┘
                                              │
                                              └───────► (Repeats with Higher Intensity)
```

Breaking this cycle requires replacing manual guessing with mathematically defensible safety buffers, guaranteeing transparent replenishment service-level agreements (SLAs), and engineering physical visibility into clinical storage rooms.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse | Unit or System |
|---|---|---|---|
| **Periodic Automatic Replenishment (Par)** | An order-up-to inventory control policy ($S$) where stock levels at a point-of-use location are inspected periodically and replenished to a predetermined ceiling. | Confusing the par level ($S$) with the average on-hand inventory balance. | Discrete count ($\text{eaches}$). |
| **Reorder Point ($s$)** | The inventory threshold that triggers an unscheduled, immediate replenishment order in a continuous-review policy. | Assuming periodic review systems have a static reorder point. | Discrete count ($\text{eaches}$). |
| **Review Period ($R$)** | The fixed time interval between successive inventory reviews, physical counts, or replenishment sweeps by supply chain staff. | Confusing the replenishment review schedule with order delivery lead time. | Hours or days ($[T]$). |
| **Replenishment Lead Time ($L$)** | Total elapsed duration from the moment a reorder condition is triggered until goods are physically placed into the point-of-use bin or cabinet drawer. | Measuring lead time only to central dock delivery rather than unit put-away. | Hours or days ($[T]$). |
| **Automated Dispensing Cabinet (ADC)** | A computerized, electronically locked storage cabinet (e.g., BD Pyxis, Omnicell) providing secure item-level tracking, user authentication, and transaction logging at the clinical point of use. | Treating ADCs as passive safes rather than real-time supply chain sensors. | Hardware platform (drawers, pockets). |
| **Visual Two-Bin Kanban** | A lean replenishment mechanism where an item is stored in two identical physical bins (primary and reserve); depletion of the primary bin serves as the visual signal to reorder. | Allowing nurses to pick from both bins simultaneously, defeating the replenishment trigger. | Physical bins, color-coded. |
| **Clean Utility Room** | A controlled-environment hospital room dedicated to storing sterile and clean medical-surgical supplies, subject to positive air pressure and strict cleanliness standards. | Storing external corrugated cardboard shipping containers inside the room. | Facility location identifier. |
| **Unit Census ($N_t$)** | The number of active, occupied patient beds in a clinical nursing unit on day or shift $t$. | Assuming bed capacity (licensed beds) equals active clinical demand. | Integer count of patients. |
| **Case Mix Index (CMI)** | A relative clinical severity score reflecting the clinical complexity and resource consumption intensity of a hospital's patient population. | Assuming two nursing units with identical patient counts consume identical supply volumes. | Dimensionless ratio (base 1.00). |
| **Null-Pick / Zero-Pick** | An event where a clinician opens a designated ADC pocket or searches a par bin, finds zero stock, and closes it without obtaining the requested item. | Failing to log null-picks, thereby recording zero demand instead of a clinical stockout. | Discrete logged event. |
| **Discrepancy Count** | A condition where the physical count inside an ADC drawer or bin does not match the perpetual inventory balance recorded by the management software. | Dismissing discrepancies as administrative quirks rather than indicators of diversion or hoarding. | Integer deviation ($\Delta = \text{Actual} - \text{System}$). |
| **Blind Count** | An ADC software workflow requiring the clinician or supply technician to input the physical count without displaying the expected system balance. | Displaying the system balance on screen, which induces "lazy counting" (confirming without verifying). | Binary workflow flag. |
| **Squirrel Stash** | Informal, unauthorized physical inventory hoarded by clinical staff outside official par locations to guard against perceived stockouts. | Treating hoarding as malicious behavior rather than a rational response to supply chain unreliability. | Hidden inventory count ($\text{eaches}$). |
| **STAT / Hot-Shot Courier** | An emergency, unscheduled point-to-point delivery of a missing critical medication or surgical supply, executed by dedicated transport runners or external couriers. | Treating STAT couriers as standard operating procedure rather than an expensive exception. | Discrete run; cost $(\$ / \text{trip})$. |
| **Case-Pack Rounding** | The operational constraint of rounding clinical replenishment quantities up to the manufacturer's inner-pack or carton multiple. | Replenishing loose, unsealed eaches that contaminate sterile packaging or clutter bins. | Integer multiple ($Q_{\text{pack}}$). |
| **Cycle Service Level ($\alpha$)** | The probability that no stockout occurs during an entire replenishment cycle (review period plus lead time): $\alpha = P(D \le S)$. | Conflating cycle service level with volumetric item fill rate. | Probability $[0, 1]$. |
| **Item Fill Rate ($\beta$)** | The fraction of total clinical demand that is satisfied immediately from on-hand point-of-use inventory: $\beta = 1 - \frac{E[\text{Shortage}]}{E[\text{Demand}]}$. | Assuming a 95% cycle service level equates to a 95% fill rate (fill rate is almost always higher). | Percentage $[0, 1]$. |
| **Par Creep** | The gradual, uncontrolled upward drift of par levels over time driven by clinical complaints without formal demand-data review. | Sizing par levels permanently based on single-event historical peak demand. | Count drift rate ($\Delta S / \Delta t$). |
| **Volumetric Shelf Footprint** | The physical three-dimensional cubic volume occupied by an item's par allocation relative to available clean utility bin dimensions. | Optimizing inventory strictly in financial units while ignoring shelf-space saturation. | Cubic centimeters ($\text{cm}^3$) or shelf inches. |
| **Stockout Penalty Function** | A mathematical cost model penalizing inventory shortages based on clinical criticality, procedure delay, expediting expense, and patient risk. | Using a generic corporate shortage penalty for life-critical clinical items. | Currency per shortage event $(\$ / \text{event})$. |
| **Return to Stock (RTS)** | An ADC or supply transaction logging an item that was removed for a patient but returned unused, requiring integrity verification. | Allowing opened or temperature-sensitive items to be returned without quality inspection. | Transaction log record. |
| **Central Sterile Processing / Central Supply (SPD)** | The centralized hospital department responsible for decontaminating, inspecting, assembling, sterilizing, and distributing surgical instruments and clinical kits. | Treating SPD as an external vendor rather than an internal closed-loop supply node. | Hospital cost center. |
| **Medication Override Matrix** | An approved institutional list of emergency medications that clinicians may remove from an ADC prior to pharmacist order verification. | Overriding routine maintenance medications during non-emergencies. | Clinical policy table. |
| **Low-Unit-of-Measure (LUM)** | A distribution paradigm where medical supplies are picked and packed at the central warehouse in the exact "each" or inner-box quantities needed for immediate shelf put-away. | Shipping bulk corrugated master cartons directly onto clinical floors. | Logistics distribution protocol. |
| **Gravity-Fed Flow Rack** | A sloped shelving system where cartons or Kanban bins are loaded from the rear and slide forward by gravity, physically enforcing First-In, First-Out (FIFO) consumption. | Loading new supplies from the front of the rack, burying older inventory behind. | Physical storage fixture. |
| **Look-Alike / Sound-Alike (LASA)** | Medications or clinical supplies with confusingly similar names, packaging, or colors requiring physical and visual separation in par storage. | Storing LASA medications in adjacent ADC pockets or side-by-side Kanban bins. | Patient safety designation. |
| **Consolidated Service Center (CSC)** | An off-campus, centralized logistics and warehousing hub operated by a health system to aggregate purchasing, sterilization, kitting, and LUM distribution. | Treating a CSC as a simple remote warehouse without clinical kitting integration. | Enterprise logistics node. |

---

## 5. The mechanics, step by step

Implementing high-reliability clinical par replenishment requires ten sequential, interlocked steps integrating clinical demand forecasting, spatial engineering, and behavioral control.

```text
  ┌────────────────────────────────────────────────────────────────────────┐
  │ Step 1: Baseline Demand Characterization and Census Normalization      │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼────────────────────────────────────┐
  │ Step 2: Replenishment Topology and Technology Selection                │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼────────────────────────────────────┐
  │ Step 3: Lead Time Decomposition and Exposure Window Modeling           │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼────────────────────────────────────┐
  │ Step 4: Discrete Stochastic Par Sizing Formulas                        │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼────────────────────────────────────┐
  │ Step 5: Visual Two-Bin Kanban Sizing Mechanics                         │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼────────────────────────────────────┐
  │ Step 6: Clinical Stockout Penalty Function Formulation                 │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼────────────────────────────────────┐
  │ Step 7: ADC Telemetry Ingestion and Data Cleansing                     │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼────────────────────────────────────┐
  │ Step 8: Behavioral Hoarding Diagnostics and Systematic Remediation     │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼────────────────────────────────────┐
  │ Step 9: Volumetric and Geometric Shelf Constraints                     │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────▼────────────────────────────────────┐
  │ Step 10: Dynamic Par Adjustment Engine and Governance Rhythm           │
  └────────────────────────────────────────────────────────────────────────┘
```

### Step 1: Baseline Demand Characterization and Census Normalization

Hospital demand at the point of care is driven by two underlying variables: the number of active patients (unit census) and the clinical consumption intensity per patient. Raw historical issue quantities are distorted by fluctuating bed occupancy.

1. **Extract Historical Consumption and Census:** For each SKU $i$ on nursing unit $u$, extract daily transaction quantities $D_{i,u,t}$ and the midnight patient census $N_{u,t}$ for $t \in \{1, 2, \dots, T\}$.
2. **Compute Normalized Patient-Day Usage Rate:**
   $$\lambda_{ppd, i, u} = \frac{\sum_{t=1}^T D_{i, u, t}}{\sum_{t=1}^T N_{u, t}}$$
   where $\lambda_{ppd}$ represents the average consumption per patient-day.
3. **Characterize Demand Dispersion:** Calculate the sample mean $\bar{D}$ and sample variance $s^2$ of daily consumption. Evaluate the **Variance-to-Mean Ratio (VMR)**:
   $$\text{VMR} = \frac{s^2}{\bar{D}}$$
   * If $\text{VMR} \approx 1.0$, demand conforms to a standard **Poisson process** (typical for rare medications, specialized catheter kits, and emergency trays).
   * If $\text{VMR} > 1.5$, demand exhibits significant **overdispersion** (typical for routine consumables like IV flushes, exam gloves, and dressings consumed in bursts during shift changes). Overdispersed items must be modeled via a **Negative Binomial distribution** or compound Poisson process to prevent under-buffering safety stock.

### Step 2: Replenishment Topology and Technology Selection

Not all items belong in automated cabinets, and not all items belong in open bins. Match each SKU to an appropriate replenishment topology based on value, regulatory control, shelf life, and clinical urgency:

```text
┌──────────────────────────────────────────────────────────────────────────────────────────┐
│                                 TOPOLOGY SELECTION MATRIX                                │
├─────────────────────────┬───────────────────────────┬────────────────────────────────────┤
│ Clinical Attribute      │ Automated Dispensing (ADC)│ Visual Two-Bin Kanban              │
├─────────────────────────┼───────────────────────────┼────────────────────────────────────┤
│ Regulatory Status       │ DEA Schedule II-V, Toxins │ Non-controlled Med-Surg Supplies   │
│ Unit Purchase Cost      │ High (> $50.00 / each)    │ Low-to-Moderate (< $20.00 / each)  │
│ Traceability Need       │ Serialized, Lot-Tracked   │ Commodity SKU, Lot-Batch           │
│ Physical Dimensions     │ Small to Medium vials/kits│ Bulk cartons, tubing, flushes      │
│ Consumption Velocity    │ Low to Moderate           │ High to Very High                  │
│ Stockout Risk Consequence│ Critical / Life-Safety    │ Moderate (substitutable)           │
└─────────────────────────┴───────────────────────────┴────────────────────────────────────┘
```

### Step 3: Lead Time Decomposition and Exposure Window Modeling

The inventory risk exposure window $T_e$ depends on whether replenishment operates under periodic or continuous review.

1. **Periodic Review Policy $(R, S)$:** In a periodic par cart or scheduled runner sweep, the replenishment exposure window is the sum of the review period $R$ and the replenishment lead time $L$:
   $$T_e = R + L$$
   If a nursing unit is reviewed every 24 hours ($R = 1.0\text{ day}$) and internal warehouse picking, staging, and transport take 12 hours ($L = 0.5\text{ days}$), the par level $S$ must protect clinical operations over an exposure window of $T_e = 1.5\text{ days}$ (36 hours).
2. **Continuous Review Policy $(s, S)$:** In an ADC with real-time network telemetry, an order is triggered the moment on-hand inventory drops to or below the reorder point $s$. The exposure window is strictly the replenishment lead time:
   $$T_e = L$$
3. **Deconstruct Lead Time Components:** Total lead time $L$ is rarely deterministic. It represents the convolution of four discrete operational segments:
   $$L = t_{\text{batch}} + t_{\text{pick}} + t_{\text{transit}} + t_{\text{putaway}}$$
   * $t_{\text{batch}}$: Delay until the central warehouse or pharmacy releases the replenishment wave.
   * $t_{\text{pick}}$: Time required to pick and pack items in the Central Service Center (CSC).
   * $t_{\text{transit}}$: Transport duration across the hospital campus (pneumatic tube, cart runner, or AGV).
   * $t_{\text{putaway}}$: Time elapsed between arrival in the unit's clean room and physical drawer stocking.

### Step 4: Discrete Stochastic Par Sizing Formulas

Because medical supplies are dispensed in discrete integer units and daily demand is often low, continuous Gaussian approximations break down. Discrete distributions must be applied.

#### 4.1 Poisson Par Sizing (Periodic Review)

For items exhibiting stationary Poisson arrival rates with daily mean $\lambda$:
The total demand $X$ across exposure window $T_e$ follows a Poisson distribution with parameter:
$$\mu_e = \lambda \cdot T_e$$
The probability of observing exactly $k$ demands during the exposure window is:
$$P(X = k) = \frac{\mu_e^k e^{-\mu_e}}{k!}$$
To achieve a target cycle service level $\alpha^*$ (e.g., $\alpha^* = 0.995$ for critical supplies), the optimal par level $S^*$ is the smallest integer satisfying the cumulative distribution function (CDF):
$$S^* = \min \left\{ S \in \mathbb{Z}_{\ge 0} \;\middle|\; \sum_{k=0}^S \frac{\mu_e^k e^{-\mu_e}}{k!} \ge \alpha^* \right\}$$
The safety stock component is the difference between the par level and the expected exposure demand:
$$SS = S^* - \mu_e$$

#### 4.2 Negative Binomial Par Sizing (Overdispersed Demand)

When sample variance significantly exceeds the mean ($\sigma_e^2 > \mu_e$), fit a Negative Binomial distribution parameterized by failure count $r$ and success probability $p$:
$$p = \frac{\mu_e}{\sigma_e^2}, \quad r = \frac{\mu_e^2}{\sigma_e^2 - \mu_e}$$
The cumulative probability is evaluated using the regularized incomplete beta function:
$$P(X \le S) = I_p(r, S + 1) \ge \alpha^*$$
Solving for the discrete integer $S^*$ guarantees that safety stock accounts for burst consumption.

### Step 5: Visual Two-Bin Kanban Sizing Mechanics

A physical two-bin Kanban operates as an elegant, fail-safe visual continuous-review system.

```text
┌───────────────────────────────────────┬───────────────────────────────────────┐
│          PRIMARY BIN (FRONT)          │          RESERVE BIN (REAR)           │
│ Holds active clinical consumption     │ Holds safety buffer + lead-time stock │
│ Capacity = B units                    │ Capacity = B units                    │
│ Status: In Use by Nursing Staff       │ Status: Untouched until Primary Empty │
└───────────────────────────────────────┴───────────────────────────────────────┘
```

1. **Trigger Condition:** Nurses pick exclusively from the primary bin. When the primary bin is completely exhausted, the clinician places the empty bin on an upper return shelf or flips an integrated RFID card. This physical action constitutes the reorder trigger.
2. **Protection Requirement:** The reserve bin must sustain clinical consumption during the entire replenishment lead time $L$ required for materials management to collect the empty bin, pick replacement stock in the central store, deliver it to the unit, and restock.
3. **Minimum Single Bin Capacity ($B_{\text{min}}$):**
   $$B_{\text{min}} = \mu_L + z_{\alpha} \cdot \sigma_L$$
   where $\mu_L = \lambda \cdot L$ and $\sigma_L = \sqrt{L \cdot \sigma_d^2 + \lambda^2 \cdot \sigma_L^2}$ accounting for lead-time variability.
4. **Case-Pack Rounding Constraint:** To eliminate loose eaches and damaged packaging, bin capacity must be rounded up to the nearest integer multiple of the manufacturer's inner case pack $Q_{\text{pack}}$:
   $$B = \left\lceil \frac{B_{\text{min}}}{Q_{\text{pack}}} \right\rceil \times Q_{\text{pack}}$$
5. **Total Two-Bin Footprint:** The total inventory authorized for the unit is:
   $$\text{Total System Stock} = 2 \times B$$
   When the runner returns with replenishment, they restock the empty bin, place it in the *reserve* (rear) position, and slide the partially consumed reserve bin forward into the *primary* position. This enforces strict **First-In, First-Out (FIFO)** physical rotation.

### Step 6: Clinical Stockout Penalty Function Formulation

Setting par levels based solely on arbitrary management service-level targets (e.g., "everything gets 95%") leads to severe misallocation. High-cost, non-critical items are over-buffered while low-cost, life-critical items run out.

Define the **Total Economic and Clinical Cost of Shortage ($c_s$)**:
$$c_s = c_{\text{courier}} + c_{\text{labor}} + c_{\text{delay}} + \psi_{\text{clinical}}$$

* $c_{\text{courier}}$: Direct financial cost of an emergency STAT courier or dedicated runner from a regional distribution center (typically $\$75\text{--}\$250$ per run).
* $c_{\text{labor}}$: Cost of clinical personnel diverted to hunt for stock:
  $$c_{\text{labor}} = \tau_{\text{hunt}} \times w_{\text{RN}}$$
  where $\tau_{\text{hunt}}$ is nurse search time (0.5 to 1.5 hours) and $w_{\text{RN}}$ is the fully loaded hourly nurse wage rate ($\$65.00/\text{hr}$).
* $c_{\text{delay}}$: Cost of surgical theater or procedure room idle time:
  $$c_{\text{delay}} = \Delta t_{\text{delay}} \times C_{\text{OR\_minute}}$$
  Operating room downtime is benchmarked at $\$60.00\text{--}\$120.00$ per minute. A 15-minute delay for a missing sterile tray costs over $\$1,200.00$.
* $\psi_{\text{clinical}}$: Explicit clinical risk penalty scalar, categorized by patient hazard:
  * Category 1 (Non-critical convenience item): $\psi = \$0$.
  * Category 2 (Therapeutic delay, minimal clinical harm): $\psi = \$500$.
  * Category 3 (Severe physiological compromise, life-support impact): $\psi = \$5,000\text{--}\$25,000$.

Using the newsvendor formulation, balance the expected cost of holding excess inventory against the expected cost of shortage. The unit holding cost over the exposure window is:
$$h_e = c_{\text{unit}} \times r_h \times \frac{T_e}{365.25}$$
where $c_{\text{unit}}$ is unit acquisition cost and $r_h$ is the annual holding rate (typically $20\%\text{--}30\%$, incorporating capital cost, spatial cost, obsolescence, and sterile expiry).

The optimal economic-clinical service level is given by the **Critical Ratio ($CR$)**:
$$CR = \frac{c_s}{c_s + h_e}$$
For an inexpensive, life-critical item ($c_{\text{unit}} = \$5.00$, $c_s = \$2,500.00$), $CR > 0.9999$. Sizing par levels to this critical ratio guarantees mathematical alignment with patient safety.

### Step 7: ADC Telemetry Ingestion and Data Cleansing

Automated Dispensing Cabinets generate massive event streams. To construct accurate par models, raw transaction tables must be parsed and filtered:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                              TYPICAL ADC TRANSACTION LOG SCHEMA                        │
├───────────┬─────────────┬──────────┬──────────┬─────────┬───────────┬──────────────────┤
│ Tx_ID     │ Timestamp   │ Unit_ID  │ SKU_ID   │ Tx_Type │ Quantity  │ User_Role        │
├───────────┼─────────────┼──────────┼──────────┼─────────┼───────────┼──────────────────┤
│ 9081241   │ 07:14:22    │ ICU_03   │ MED_4091 │ DISP    │ 1         │ RN_Bedside       │
│ 9081242   │ 07:42:10    │ ICU_03   │ MED_1022 │ DISC    │ -2        │ Pharmacy_Tech    │
│ 9081243   │ 08:15:00    │ ICU_03   │ MED_4091 │ WAST    │ 0.5       │ RN_Bedside       │
│ 9081244   │ 09:30:11    │ ICU_03   │ MED_8812 │ NULL_PK │ 0         │ RN_Bedside       │
│ 9081245   │ 11:00:45    │ ICU_03   │ MED_4091 │ RESTK   │ 10        │ Materials_Runner │
└───────────┴─────────────┴──────────┴──────────┴─────────┴───────────┴──────────────────┘
```

#### Data Cleansing Rules

1. **Reconstruct True Clinical Demand:** True clinical consumption is not simply `DISP`.
   $$D_{\text{clinical}} = \text{DISP} - \text{RTS} - \text{WAST}$$
   If an RN dispenses 2 ampules, administers 1, and wastes or returns 1, true demand is 1.
2. **Convert Null-Picks to Censored Demand:** A `NULL_PK` transaction represents a stockout event. If an RN opens an empty pocket, actual demand was $\ge 1$. Impute unobserved lost demand using survival analysis or local historical averages.
3. **Filter Administrative Batch Adjustments:** Bulk inventory transfers or end-of-month cycle counts logged as transactions must be flagged and excluded from daily consumption time series.

### Step 8: Behavioral Hoarding Diagnostics and Systematic Remediation

To detect informal "squirrel stashes" before they cause stockouts:

1. **Calculate Rolling Coefficient of Variation ($CV$):**
   $$CV_t = \frac{\sigma_{D, [t-30, t]}}{\bar{D}_{[t-30, t]}}$$
   An erratic $CV$ with prolonged zero-consumption stretches followed by sudden massive withdrawals indicates an informal ward stash being periodically replenished.
2. **Discrepancy Signature Analysis:** Frequent negative discrepancies (`DISC < 0`) during shift transitions (07:00 and 19:00) signal unauthorized off-cabinet hoarding.
3. **Remediation Protocol:**
   * Establish **Supply SLA Guarantees:** Materials management commits to a maximum 15-minute response time for emergency replenishment requests.
   * Hold **"No-Fault Hoard Amnesty" Days:** Periodically invite clinical staff to return hoarded supplies from breakrooms and personal lockers without disciplinary penalty.
   * Install **Clear Acrylic Storage Bins:** Replace opaque bins with transparent acrylic containers so nurses can physically see reserve stock levels from across the room.

### Step 9: Volumetric and Geometric Shelf Constraints

Clean utility rooms have rigid geometric limits. Expanding the par level of one SKU consumes physical space required by another.

Formulate the **Multi-Item Knapsack Shelf Allocation Problem**:
$$\max \sum_{i=1}^M \beta_i(S_i) \cdot w_i$$
subject to:
$$\sum_{i=1}^M v_i \cdot S_i \le V_{\text{total}}$$
$$S_i \ge S_{i, \text{min}} \quad \forall i \in \{1, \dots, M\}$$
where:
* $\beta_i(S_i)$ is the item fill rate achieved under par level $S_i$.
* $w_i$ is the clinical criticality weight.
* $v_i$ is the physical unit volume ($\text{cm}^3$) of SKU $i$.
* $V_{\text{total}}$ is the total usable volumetric capacity of the clean utility shelving.
* $S_{i, \text{min}}$ is the mandatory minimum safety stock dictated by clinical policy.

Solve using integer linear programming (ILP) or a greedy heuristic sorting items by marginal fill-rate benefit per unit volume: $\frac{\Delta \beta_i / \Delta S_i}{v_i}$.

### Step 10: Dynamic Par Adjustment Engine and Governance Rhythm

Clinical par levels must not remain static. Establish an automated weekly re-tuning loop:

```text
       ┌────────────────────────────────────────────────────────┐
       │             Weekly Automated Tuning Job                │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │ Ingest 90-Day Rolling Telemetry & 14-Day ADT Forecast   │
       │ Compute: λ_ppd, VMR, Lead-Time Distributions           │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
       ┌────────────────────────────────────────────────────────┐
       │ Evaluate S_new vs S_current                            │
       │ Check Volumetric Constraints & Packaging Multiples     │
       └───────────────────────────┬────────────────────────────┘
                                   │
                                   ▼
                 ┌───────────────────────────────────┐
                 │ Is |S_new - S_current| > Threshold?│
                 └───────┬───────────────────┬───────┘
                         │                   │
                     YES │                   │ NO
                         ▼                   ▼
       ┌────────────────────────┐      ┌────────────────────────┐
       │ Route to Joint Supply/ │      │ Maintain Existing Par  │
       │ Nursing Governance     │      │ Log "No Change" Reason │
       └──────────┬─────────────┘      └────────────────────────┘
                  │
                  ▼
       ┌────────────────────────┐
       │ Sign-off by Nurse Unit │
       │ Manager & Supply Lead  │
       └──────────┬─────────────┘
                  │
                  ▼
       ┌────────────────────────┐
       │ Push Update to ADC and │
       │ WMS/ERP Master Tables  │
       └────────────────────────┘
```

---

## 6. Worked example

### 6.1 Clinical Setting and Parameters

Consider a 32-bed Medical-Surgical Intensive Care Unit (MSICU) at a tertiary academic medical center.
* **Operating Census:** Average occupied beds $\bar{N} = 28$ patients (87.5% occupancy rate).
* **Review Schedules:**
  * **Automated Dispensing Cabinet (Pharmacy):** Daily review ($R = 1.0\text{ day} = 24\text{ hours}$); hospital central pharmacy delivery lead time $L = 0.5\text{ days} = 12\text{ hours}$. Total exposure window $T_e = R + L = 1.5\text{ days}$.
  * **Central Supply Consumables (Two-Bin Kanban):** Twice-daily runner sweep ($R = 0.5\text{ days} = 12\text{ hours}$); replenishment lead time from Central Distribution Center $L = 0.5\text{ days} = 12\text{ hours}$. Total exposure window $T_e = 1.0\text{ day}$. Lead time while reserve bin covers demand: $L = 0.5\text{ days}$.

We analyze three representative SKUs spanning the clinical spectrum:

1. **Item A (CVC Triple-Lumen Kit, 7 Fr 20 cm):** Stored in ADC supply tower. Vital sterile kit for vascular access in shock.
2. **Item B (Normal Saline 0.9% IV Flush 10 mL):** Stored in Visual Two-Bin Kanban rack in the clean utility room. High-turnover routine consumable.
3. **Item C (Norepinephrine Bitartrate 4 mg/4 mL Ampule):** Stored in ADC locked matrix drawer. Ultra-critical first-line vasopressor for hemodynamic stabilization. Continuous review ($s, S$).

```text
┌─────────────────────────────────┬───────────────────┬───────────────────┬───────────────────┐
│ Parameter                       │ Item A (CVC Kit)  │ Item B (Saline)   │ Item C (Norepi)   │
├─────────────────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Replenishment Mode              │ Periodic ADC Par  │ Two-Bin Kanban    │ Continuous ADC    │
│ Unit Purchase Cost ($c$)        │ $95.00            │ $1.20             │ $18.50            │
│ Annual Holding Cost Rate ($r_h$)│ 25.0%             │ 20.0%             │ 22.0%             │
│ Mean Daily Unit Demand ($\lambda$)│ 4.0 kits/day      │ 160 syringes/day  │ 2.5 ampules/day   │
│ Daily Demand Variance ($\sigma^2$)│ 4.2 (Poisson)     │ 320.0 (NegBin)    │ 2.6 (Poisson)     │
│ Review Period ($R$)             │ 1.0 day           │ 0.5 days          │ 0.0 days (Cont.)  │
│ Replenishment Lead Time ($L$)   │ 0.5 days          │ 0.5 days          │ 0.1667 days (4 hr)│
│ Exposure Window ($T_e$)         │ 1.5 days          │ 1.0 day           │ 0.1667 days       │
│ Case-Pack Size ($Q_{\text{pack}}$)│ 1 kit             │ 30 syringes       │ 1 ampule          │
│ Target Service Level ($\alpha^*$)│ 99.9%             │ 99.0%             │ 99.99%            │
│ Stockout Penalty Cost ($c_s$)   │ $650.00           │ $35.00            │ $1,200.00         │
└─────────────────────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

---

### 6.2 Calculations Step by Step

#### Calculation 1: Item A (CVC Kit) — Periodic Review Poisson Sizing

1. **Expected Exposure Demand:**
   $$\mu_e = \lambda \times T_e = 4.0 \times 1.5 = 6.0\text{ kits}$$
2. **Economic Critical Ratio Verification:**
   Holding cost over the 1.5-day exposure window:
   $$h_e = c \times r_h \times \frac{T_e}{365.25} = 95.00 \times 0.25 \times \frac{1.5}{365.25} = \$0.0975$$
   Critical ratio:
   $$CR = \frac{c_s}{c_s + h_e} = \frac{650.00}{650.00 + 0.0975} = \frac{650.00}{650.0975} \approx 0.99985$$
   The clinical critical ratio ($99.985\%$) exceeds the baseline target ($99.9\%$), dictating an effective target $\alpha^* = 0.99985$.
3. **Evaluate Cumulative Poisson Probabilities for $\mu_e = 6.0$:**
   $$P(X = k) = \frac{6.0^k e^{-6.0}}{k!}$$

```text
┌────┬──────────────┬──────────────┬────┬──────────────┬──────────────┐
│ k  │ P(X = k)     │ CDF: F(k)    │ k  │ P(X = k)     │ CDF: F(k)    │
├────┼──────────────┼──────────────┼────┼──────────────┼──────────────┤
│ 6  │ 0.16062      │ 0.60630      │ 12 │ 0.01126      │ 0.99117      │
│ 7  │ 0.13768      │ 0.74398      │ 13 │ 0.00520      │ 0.99637      │
│ 8  │ 0.10326      │ 0.84724      │ 14 │ 0.00223      │ 0.99860      │
│ 9  │ 0.06884      │ 0.91608      │ 15 │ 0.00089      │ 0.99949      │
│ 10 │ 0.04130      │ 0.95738      │ 16 │ 0.00033      │ 0.99982      │
│ 11 │ 0.02253      │ 0.97991      │ 17 │ 0.00012      │ 0.99994      │
└────┴──────────────┴──────────────┴────┴──────────────┴──────────────┘
```

4. **Determine Par Level $S$:**
   * To satisfy $\alpha^* \ge 0.99985$, we require $F(S) \ge 0.99985$.
   * $F(16) = 0.99982 < 0.99985$.
   * $F(17) = 0.99994 \ge 0.99985$.
   * Therefore, **Par Level $S = 17\text{ kits}$**.
5. **Safety Stock:**
   $$SS = S - \mu_e = 17 - 6.0 = 11.0\text{ kits}$$

---

#### Calculation 2: Item B (Normal Saline Flush) — Visual Two-Bin Kanban Sizing

1. **Lead Time Demand Characteristics:**
   The reserve bin must sustain demand during the replenishment turnaround lead time ($L = 0.5\text{ days}$):
   $$\mu_L = \lambda \times L = 160 \times 0.5 = 80\text{ syringes}$$
   Since daily variance is overdispersed ($\sigma_d^2 = 320$):
   $$\sigma_L = \sqrt{L \times \sigma_d^2} = \sqrt{0.5 \times 320} = \sqrt{160} \approx 12.649\text{ syringes}$$
2. **Buffer Sizing for 99.0% Service Level:**
   For $\alpha = 0.990$, the standard normal quantile is $z_{0.99} = 2.326$.
   $$SS_{\text{raw}} = z_{0.99} \times \sigma_L = 2.326 \times 12.649 = 29.42\text{ syringes}$$
3. **Unconstrained Single Bin Capacity:**
   $$B_{\text{raw}} = \mu_L + SS_{\text{raw}} = 80 + 29.42 = 109.42\text{ syringes}$$
4. **Apply Case-Pack Rounding ($Q_{\text{pack}} = 30$):**
   $$B = \left\lceil \frac{109.42}{30} \right\rceil \times 30 = 4 \times 30 = 120\text{ syringes}$$
5. **Total System Authorized Stock:**
   $$\text{Total Stock} = 2 \times B = 2 \times 120 = 240\text{ syringes (8 boxes of 30)}$$
6. **Effective Service Protection:**
   With $B = 120$, the actual buffer in the reserve bin is:
   $$SS_{\text{actual}} = 120 - 80 = 40\text{ syringes}$$
   Effective $z$-score:
   $$z_{\text{actual}} = \frac{40}{12.649} \approx 3.16$$
   This delivers an actual cycle service level of $\Phi(3.16) = 99.92\%$, completely absorbing shift-change surges.

---

#### Calculation 3: Item C (Norepinephrine Ampules) — Continuous Review ADC Sizing

1. **Lead Time Exposure Demand:**
   Continuous telemetry triggers restocking immediately when on-hand inventory reaches $s$. Lead time from central pharmacy is 4 hours ($L = \frac{4}{24} = 0.1667\text{ days}$).
   $$\mu_L = \lambda \times L = 2.5 \times 0.1667 = 0.4167\text{ ampules}$$
2. **Evaluate Poisson CDF for $\mu_L = 0.4167$:**
   $$P(X = k) = \frac{0.4167^k e^{-0.4167}}{k!}$$

```text
┌────┬──────────────┬──────────────┬───────────────────────────────────────────┐
│ k  │ P(X = k)     │ CDF: F(k)    │ Clinical Service Coverage                 │
├────┼──────────────┼──────────────┼───────────────────────────────────────────┤
│ 0  │ 0.65924      │ 0.65924      │ 65.92% (Unacceptable)                    │
│ 1  │ 0.27468      │ 0.93392      │ 93.39% (High clinical stockout risk)      │
│ 2  │ 0.05723      │ 0.99115      │ 99.12% (Does not meet 99.99% target)      │
│ 3  │ 0.00795      │ 0.99910      │ 99.91% (Marginal)                         │
│ 4  │ 0.00083      │ 0.99993      │ 99.993% (Meets 99.99% target)             │
│ 5  │ 0.00007      │ 1.00000      │ 99.999%                                   │
└────┴──────────────┴──────────────┴───────────────────────────────────────────┘
```

3. **Determine Reorder Point $s$:**
   * Reorder point requires $F(s) \ge 0.9999$.
   * $F(3) = 0.99910 < 0.99990$.
   * $F(4) = 0.99993 \ge 0.99990$.
   * Therefore, **Reorder Point $s = 4\text{ ampules}$**.
4. **Determine Order-Up-To Level $S$ (Cabinet Capacity):**
   The ADC mini-drawer has fixed physical pocket partitions of 12 units. Central pharmacy replenishment waves run efficiently in 8-pack batches ($Q = 8$).
   $$S = s + Q = 4 + 8 = 12\text{ ampules}$$
   Whenever the cabinet balance drops to 4 ampules, an automated queue alert dispatches an 8-pack from pharmacy to restore inventory to 12.

---

### 6.3 Consolidated Policy Summary

```text
┌─────────────────────────┬───────────────────┬───────────────────┬───────────────────┐
│ Metric                  │ Item A (CVC Kit)  │ Item B (Saline)   │ Item C (Norepi)   │
├─────────────────────────┼───────────────────┼───────────────────┼───────────────────┤
│ Replenishment Policy    │ Periodic (R, S)   │ Two-Bin Kanban    │ Continuous (s, S) │
│ Reorder Point ($s$)     │ N/A (Daily Sweep) │ Bin Empty Trigger │ 4 ampules         │
│ Par Level ($S$)         │ 17 kits           │ 240 syringes      │ 12 ampules        │
│ Safety Stock ($SS$)     │ 11.0 kits         │ 40 syringes (res) │ 3.58 ampules (ROP)│
│ Achieved Service Level  │ 99.99%            │ 99.92%            │ 99.99%            │
│ Total Unit Stock Value  │ $1,615.00         │ $288.00           │ $222.00           │
│ Annual Holding Cost     │ $403.75           │ $57.60            │ $48.84            │
└─────────────────────────┴───────────────────┴───────────────────┴───────────────────┘
```

---

## 7. Code, query, or calculation

The following self-contained Python script implements discrete stochastic par level sizing, continuous-review reorder point calculation, and two-bin Kanban dimensioning. It uses `scipy.stats` for exact Poisson CDF evaluation and enforces case-pack constraints.

### 7.1 Python Implementation

```python
import math
from typing import Dict, Any
from scipy import stats

def compute_clinical_par_policy(
    item_name: str,
    daily_demand: float,
    review_period_days: float,
    lead_time_days: float,
    unit_cost: float,
    annual_holding_rate: float,
    stockout_penalty: float,
    target_csl: float = 0.999,
    case_pack: int = 1,
    kanban_mode: bool = False
) -> Dict[str, Any]:
    """Calculates clinical replenishment par levels, reorder points, and Kanban sizing."""
    exposure_days = review_period_days + lead_time_days
    mu_exposure = daily_demand * exposure_days
    mu_lead = daily_demand * lead_time_days
    
    # Critical ratio from shortage and holding economics
    h_period = unit_cost * annual_holding_rate * (exposure_days / 365.25)
    cr = stockout_penalty / (stockout_penalty + h_period)
    effective_alpha = max(cr, target_csl)
    
    # Discrete Poisson model for periodic review par level (S)
    dist_exposure = stats.poisson(mu_exposure)
    par_level = int(dist_exposure.ppf(effective_alpha))
    achieved_csl = float(dist_exposure.cdf(par_level))
    safety_stock = par_level - mu_exposure
    
    # Continuous review reorder point (s) across lead time
    dist_lead = stats.poisson(mu_lead)
    reorder_point = int(dist_lead.ppf(target_csl))
    
    result = {
        "item": item_name,
        "mu_exposure": round(mu_exposure, 2),
        "par_S": par_level,
        "reorder_s": reorder_point,
        "safety_stock": round(safety_stock, 2),
        "achieved_csl": round(achieved_csl, 4)
    }
    
    if kanban_mode:
        # Two-bin sizing: Single bin must cover lead-time demand + buffer, rounded to pack
        raw_bin = dist_lead.ppf(target_csl)
        bin_capacity = int(math.ceil(raw_bin / case_pack) * case_pack)
        result["single_bin_capacity"] = bin_capacity
        result["total_kanban_system"] = 2 * bin_capacity
        
    return result

if __name__ == "__main__":
    items = [
        ("CVC Triple-Lumen Kit", 4.0, 1.0, 0.5, 95.0, 0.25, 650.0, 0.999, 1, False),
        ("Saline Flush 10mL", 160.0, 0.5, 0.5, 1.20, 0.20, 35.0, 0.990, 30, True),
        ("Norepinephrine 4mg", 2.5, 0.0, 0.1667, 18.5, 0.22, 1200.0, 0.9999, 1, False),
    ]
    print(f"{'Item':<22} | {'Exp Dem':<7} | {'Par S':<5} | {'ROP s':<5} | {'SafeStk':<7} | {'CSL':<6} | {'Kanban'}")
    print("-" * 75)
    for row in items:
        res = compute_clinical_par_policy(row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9])
        kanban_str = f"2x{res['single_bin_capacity']}" if "single_bin_capacity" in res else "N/A"
        print(f"{res['item']:<22} | {res['mu_exposure']:<7.2f} | {res['par_S']:<5} | {res['reorder_s']:<5} | {res['safety_stock']:<7.2f} | {res['achieved_csl']:<6.4f} | {kanban_str}")
```

### 7.2 Program Output

```text
Item                   | Exp Dem | Par S | ROP s | SafeStk | CSL    | Kanban
---------------------------------------------------------------------------
CVC Triple-Lumen Kit   | 6.00    | 17    | 8     | 11.00   | 0.9999 | N/A
Saline Flush 10mL      | 160.00  | 215   | 102   | 55.00   | 1.0000 | 2x120
Norepinephrine 4mg     | 0.42    | 6     | 4     | 5.58    | 1.0000 | N/A
```

---

## 8. Data, standards, and regulatory context

Hospital clinical replenishment intersects with rigorous federal safety mandates, accreditation standards, and building codes. Sizing and managing clinical storage is never purely an inventory problem; it is a clinical compliance regime.

### 8.1 The Joint Commission (TJC) Accreditation Standards

The Joint Commission enforces explicit environmental and medication management standards during unannounced hospital surveys:

1. **Standard MM.03.01.01 (Medication Storage and Security):** Medications must be stored in locked areas or within automated dispensing cabinets to prevent diversion. Controlled substances require perpetual inventory verification. High-alert medications (e.g., concentrated potassium chloride, neuromuscular blocking agents) must be physically segregated in designated, warning-labeled ADC drawers to prevent mis-selection.
2. **Standard MM.03.01.03 (Safe Medication Dispensing):** Emergency medications must be available in immediately accessible, tamper-evident emergency carts or crash boxes. The replenishment policy must define explicit restock turnaround protocols following cart seal rupture.
3. **Standard IC.02.02.01 (Infection Control & Sterile Storage):** Clean and sterile supplies must be protected from contamination. Supplies must be stored at least **8 inches above the finished floor**, **18 inches below fire sprinkler heads** (or 36 inches if dense storage racks obstruct spray patterns), and at least **2 inches from exterior walls** to prevent moisture condensation.
4. **The Corrugated Cardboard Ban:** The Joint Commission and the Association of periOperative Registered Nurses (AORN) strictly prohibit external shipping cartons constructed of porous corrugated cardboard in clean utility rooms, surgical cores, or medication preparation areas. Corrugated cardboard is a documented reservoir for fungal spores (particularly *Aspergillus* species) and insect debris accumulated during freight transport. Supplies must be de-boxed in an outer breakdown area and transferred into wipeable plastic bins before entering the clean utility envelope.

### 8.2 Drug Enforcement Administration (DEA) Mandates

Replenishment of Schedule II–V controlled substances stored in ADCs is governed by **Title 21 Code of Federal Regulations Part 1301 and 1304**:

* **Perpetual Electronic Ledger:** Every access, dispense, waste, and restock transaction must be logged with user credentials, witness verification (for waste), and timestamp.
* **Blind Discrepancy Audits:** Supply chain and pharmacy technicians restocking controlled substances must perform blind counts. The ADC software must not disclose expected inventory balances. Discrepancies must be investigated and resolved prior to shift closure. Any unresolved theft or significant loss must be reported to the DEA via **Form 106** within one business day.

### 8.3 Drug Supply Chain Security Act (DSCSA) & GS1 Point-of-Use Standards

Under the US DSCSA (fully enforceable under FDA Title II of the DQSA):
* **Transaction Information (TI), Transaction History (TH), and Transaction Statement (TS)** must trace prescription pharmaceuticals to the unit package level.
* Point-of-use replenishment systems are transitioning to **GS1 DataMatrix** 2D barcode scanning at the bedside and ADC restocking console, verifying the Global Trade Item Number (GTIN), lot number, serial number, and expiration date in a single scan. Scanning catches expired stock before it enters the active par bin.

### 8.4 USP <800> Hazardous Drug Handling and Storage

United States Pharmacopeia (USP) General Chapter <800> governs the handling of hazardous drugs (HDs)—including antineoplastic chemotherapy agents, immunosuppressants, and certain hormonal therapies—in healthcare facilities:
* **Storage Environment:** Hazardous medications stored on patient care units must be segregated from non-hazardous inventory. If un-manipulated dosage forms (e.g., intact unit-dose capsules or blistered tablets) are stored in an ADC, they must reside in dedicated, clearly designated drawers with spill containment.
* **Personal Protective Equipment (PPE) Co-Location:** Par levels for hazardous drug administration must be physically co-located with specialized chemotherapy chemotherapy-rated gloves, gowns, and hazardous waste disposal receptacles.

### 8.5 CDC and VFC Vaccine Cold-Chain Requirements

Point-of-use pharmaceutical refrigerators storing vaccines or temperature-sensitive biologics must comply with the Centers for Disease Control and Prevention (CDC) Vaccine Storage and Handling Toolkit:
* **Continuous Telemetry Logging:** Refrigerators must be equipped with calibrated digital data loggers (DDLs) using buffered probes (e.g., glycol-encased temperature sensors) that log readings at least every 30 minutes.
* **Alert Escalation:** The par replenishment system must integrate with automated building telemetry. If temperatures breach the $2^\circ\text{C}\text{--}8^\circ\text{C}$ range for $> 15\text{ minutes}$, automated alerts must escalate to pharmacy on-call personnel to prevent thermal compromise.

### 8.6 Fire Protection and Building Egress (NFPA 99 / NFPA 101)

Clean utility rooms must comply with National Fire Protection Association (NFPA) codes:
* **NFPA 101 (Life Safety Code):** Minimum clear corridor widths (typically 8 feet in hospital egress routes) must not be compromised by staged par carts or overflow replenishment bins.
* **NFPA 99 (Health Care Facilities Code):** Combustible loading limitations restrict the density of plastic storage containers and alcohol-based preparatory wipes permitted per square foot in clean storage zones.

---

## 9. Industry practice and case evidence

Hospital systems across North America and Europe have transformed point-of-use replenishment from legacy manual guesswork into high-precision, data-driven supply networks.

### 9.1 Mayo Clinic: Conversion to Two-Bin RFID Kanban

* **Setting:** Mayo Clinic Saint Marys and Rochester Methodist hospitals (over 1,200 licensed acute-care beds).
* **The Challenge:** Historical replenishment relied on daily manual barcode scanning of open wire par carts by materials specialists. Stockouts of common consumables (IV tubing, catheters, dressing kits) caused significant nurse dissatisfaction, widespread hoarding in breakroom cabinets, and approximately 70 minutes per nurse shift spent searching for missing items. Par levels were routinely inflated by 30–50% to appease clinical complaints, swelling hospital working capital and overcrowding clean rooms.
* **Operational Intervention:** Materials management transitioned over 3,000 consumable medical-surgical SKUs across 120 inpatient nursing units to a standardized **Two-Bin Visual Kanban System** equipped with passive UHF RFID sensor shelves. When a nurse depletes a primary bin, they place it on an RFID-enabled replenishment shelf. The RFID antenna automatically detects the bin's embedded transponder, instantly transmitting a pick-requisition to the centralized Consolidated Service Center (CSC) without requiring manual barcode scanning.
* **Empirical Outcomes:**
  * Clean utility stockouts decreased by **82%** across the acute-care wards.
  * Direct bedside nursing time recovered was measured at **42 minutes per nurse per 12-hour shift**, redirected to direct patient care.
  * Point-of-use inventory valuation dropped by **28%**, eliminating redundant safety stock and expired consumables.
  * Emergency STAT delivery calls from nursing units to central distribution declined from an average of 45 calls per day to fewer than 4 per day.

### 9.2 Cleveland Clinic: ADC Optimization and Telemetry Tuning

* **Setting:** Cleveland Clinic health system, encompassing the main campus and regional hospitals operating over 1,400 automated dispensing cabinets.
* **The Challenge:** Decades of uncoordinated par updates resulted in severe "par creep." Fast-moving emergency medications and routine analgesics frequently suffered "null-picks" (stockout on drawer access), while slow-moving specialty medications sat in high-capacity matrix drawers, expiring before use. Discrepancy logs were overwhelmed by thousands of false alerts triggered by bad drawer slotting.
* **Operational Intervention:** The enterprise pharmacy analytics team implemented a centralized **ADC Optimization Engine**. The software continuously extracts nightly transaction logs (`DISP`, `RESTK`, `DISC`, `NULL_PK`), normalizes demand against unit patient census and surgical volume, and recalculates continuous-review $(s, S)$ thresholds using empirical Poisson and compound negative binomial distributions. Fast-moving items were migrated to high-speed carousel towers with expanded pocket capacities, while items with less than 1 dispense per quarter were consolidated into centralized regional code carts.
* **Empirical Outcomes:**
  * Overall ADC stockout rate declined by **64%**, while total ADC on-hand inventory valuation dropped by **$4.2 million**.
  * Pharmacy restocking labor was reduced by **22%** due to optimized refill batch sizes, cutting unnecessary top-off sweeps.
  * Null-pick clinical incidents fell below 0.05% of total dispense transactions.

### 9.3 Intermountain Health: Census-Driven Dynamic Par Replenishment

* **Setting:** Intermountain Health (Utah and Idaho), serving 33 hospitals supplied by a centralized Supply Chain Center (SCC).
* **The Challenge:** Pediatric and adult medical units experienced extreme seasonal demand volatility driven by respiratory syncytial virus (RSV) and winter influenza surges. Fixed par levels designed for average occupancy failed completely during peak winter months, triggering severe stockouts, and generated excessive excess inventory during summer lulls.
* **Operational Intervention:** Intermountain integrated real-time **Admission, Discharge, and Transfer (ADT)** data from the electronic health record directly into its warehouse management and replenishment engine. Daily par levels for seasonal respiratory supplies (pediatric nasal cannulas, nebulizer kits, suction catheters) are dynamically scaled each morning based on active patient census and forward 48-hour scheduled surgical caseload.
* **Empirical Outcomes:** During the peak 2022–2023 pediatric RSV surge, units operating under dynamic census-adjusted pars maintained a **99.4% clinical fill rate** without requiring emergency air couriers, while avoiding post-surge inventory write-offs.

### 9.4 Mercy Health / ROi: The Consolidated Service Center (CSC) LUM Revolution

* **Setting:** Mercy Health (St. Louis, MO) and its provider-owned supply chain organization, Resource Optimization & Innovation (ROi).
* **The Challenge:** Multiple regional community hospitals were each managing their own central store warehouses, receiving hundreds of partial bulk pallet shipments daily from commercial medical-surgical distributors. Ward par rooms were flooded with oversized brown shipping boxes, floor space was saturated, and nurses were breaking down master cartons to find individual boxes of syringes.
* **Operational Intervention:** ROi constructed a 100,000-square-foot Consolidated Service Center (CSC) that assumed all distributor deliveries. The CSC disassembles master cartons, breaks products down into **Low-Unit-of-Measure (LUM)** totes, and builds customized department-specific replenishment carts sequenced to match the physical shelf layout of each ward's clean utility room. Carts are dispatched daily via dedicated internal fleet routes.
* **Empirical Outcomes:**
  * Hospital clean utility inventory holding dropped from **18 days of supply to 4.5 days of supply**.
  * Clinical clean utility space requirements across all hospital facilities decreased by **35%**, freeing floor area for patient beds.
  * In-hospital supply chain labor costs declined by **$12 million over three years**, while packaging waste in clinical wings was virtually eliminated.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Root Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|
| **Phantom Zero-Demand** | Zero recorded consumption for 3+ consecutive weeks on high-turnover item. | Nurses hoarding supplies in an unmonitored "squirrel stash," bypassing official par bins. | Par optimization algorithm slashes par to zero; abrupt massive stockout when stash empties. | Conduct rolling coefficient of variation audit; hold no-fault hoarding amnesty; inspect clean rooms. |
| **Null-Pick Drawer Lockout** | Rapid succession of `NULL_PK` logs in ADC console without recorded `DISP`. | Reorder point ($s$) set too low or pharmacy restocking lead time exceeded. | Immediate bedside treatment delay; nurse leaves patient to borrow medication from adjacent floor. | Recalculate $(s, S)$ using empirical lead-time tail quantiles ($99.9\text{th}$ percentile); set automated alerts. |
| **Over-Par Stuffing** | Items wedged tightly into plastic bins; torn paper-poly pouches. | Well-meaning supply runners forcing entire case packs into undersized bins to avoid returning partial boxes to warehouse. | Compromised sterile barrier; packaging pinholes allow bacterial contamination; inventory loss. | Align physical bin volume with case-pack dimensions; mandate strict "no-stuffing" put-away protocol. |
| **Corrugated Cardboard Contamination** | External brown cardboard shipping cartons found on clean utility room shelves. | Delivery runners skipping outer de-boxing step to save time during restocking rush. | Immediate Joint Commission citation; fungal spore (*Aspergillus*) contamination risk to immunocompromised patients. | Establish strict outer breakdown staging area; install physical barriers; audit clean room entries. |
| **STAT Courier Cascade** | Exponential spike in emergency delivery requests to central supply between 14:00 and 18:00. | Inadequate par sizing during shift-change peak; morning consumption exhausts daily par before scheduled evening restock. | Extreme transport costs; central runners overwhelmed; routine restocking delayed, amplifying cycle. | Move from 24-hour periodic review to twice-daily sweeps ($R = 0.5\text{ days}$) or convert to continuous 2-bin Kanban. |
| **Ghost Inventory via Emergency Overrides** | Perpetual ADC inventory balance shows positive stock, but physical drawer is completely empty. | Clinicians utilizing emergency override functions to bypass patient-specific profiling during resuscitations without counting. | Stockout during emergency; misstated balance sheet; severe delay in automatic replenishment trigger. | Implement automated post-override blind reconciliation audits within 2 hours; enforce override reviews. |
| **Batch-Return Expiration Shock** | Sudden write-off of large batch of expired sterile kits in central store. | Clinical unit returning aged, hoarded stock to central store all at once just prior to expiry. | Financial loss; false perception of central store quality failure; unbudgeted replacement cost. | Enforce strict return-to-stock physical inspection; implement RFID tracking of kit manufacturing dates. |
| **Two-Bin Cross-Contamination** | Both front and rear Kanban bins partially consumed simultaneously. | Clinicians unfamiliar with Kanban pulling from rear bin because front bin was low; failure to rotate. | Depletion of both bins before reorder trigger is tripped, causing a total stockout of both bins. | Affix clear visual training placards; use gravity-fed flow racks where rear bin is physically inaccessible until front bin is removed. |
| **Par Creep** | Uninterrupted quarter-over-quarter growth in par inventory valuation across all units. | Supply chain supervisors granting permanent par increases whenever a nurse complains, without data review. | Clean room space exhaustion; severe inventory holding costs; high expiration write-offs. | Mandate bi-annual clinical governance committee approval for any permanent par increase $> 10\%$. |
| **Discrepancy Fatigue** | Supply technicians and nurses routinely clearing ADC discrepancy alerts with generic reason codes ("count error"). | Excessive false discrepancy alerts caused by bad drawer configurations or complicated packaging. | Complete loss of inventory control; theft and diversion of controlled substances go undetected. | Redesign drawer pocket layout; implement single-dose blister packaging; monitor discrepancy resolution times. |
| **Lead-Time Drift from Central Sterile** | Clean room par levels empty despite normal clinical demand; carts staged in basement corridors. | Unplanned backlog or machine breakdown in Central Sterile Processing (SPD), inflating $L$. | Unit runs out of procedural trays; scheduled surgeries delayed; clinical friction. | Model $L$ as a stochastic variable; trigger automated safety stock buffers when SPD queue depth exceeds threshold. |
| **Shelf-Life Starvation in High-Par Drawers** | Specialized medications expiring in ADC pockets at rates exceeding 25% of purchases. | Par level set to historical maximum surge demand for an item with a short stability window (e.g., reconstituted vials). | Massive drug waste; balance sheet write-downs; risk of administering expired drug during code. | Consolidate low-velocity, high-urgency drugs into regional emergency kits; reduce par to minimum safe clinical coverage. |
| **Look-Alike / Sound-Alike (LASA) Adjacency** | Near-miss administration reports logged in hospital incident management system. | Storing look-alike vials (e.g., Ephedrine and Epinephrine) in adjacent drawer pockets or side-by-side bins. | Critical medication administration error; severe patient harm or mortality; regulatory investigation. | Enforce mandatory physical separation in non-adjacent drawers; apply tall-man lettering; mandate barcode verification. |
| **Refrigerated Cold-Chain Telemetry Failure** | Refrigerator door left unlatched overnight; high temperature alarm ignored. | Inadequate latching mechanism on point-of-use drug refrigerator combined with unmonitored local alarm buzzer. | Complete thermal denaturation of unit insulin and vaccine supply; financial write-off exceeding $20,000. | Retrofit continuous cloud-connected IoT temperature probes with multi-tiered SMS and email escalation alerts. |

---

## 11. Assumptions and limits

| Assumption | Why Needed | Failure Conditions | How to Test / Boundary Verification |
|---|---|---|---|
| **Demand Independence Across Beds** | Allows modeling unit demand as the sum of independent Poisson processes across occupied beds. | Infectious disease outbreaks (e.g., Norovirus, COVID-19) where all patients require identical isolation kits simultaneously. | Calculate spatial autocorrelation and dispersion index; use Negative Binomial when VMR $> 1.5$. |
| **Stationary Demand Intensity Within Review Window** | Assumes the mean daily demand rate $\lambda$ remains constant over the exposure window $(R + L)$. | Abrupt surge in trauma admissions, mass casualty events, or seasonal viral admission waves. | Track rolling 7-day vs 30-day demand averages; integrate EHR forward admission scheduling feeds. |
| **Unbiased Telemetry Capture** | Relies on ADC logs and Kanban card scans to represent true clinical consumption timing. | Clinicians pulling supplies under emergency override without logging, or hoarding supplies in personal pockets. | Cross-reference ADC dispense records with EHR electronic Medication Administration Records (eMAR). |
| **Constant Replenishment Lead Time ($L$)** | Simplifies safety stock formulas by treating delivery duration from central store to clean room as deterministic. | Central warehouse staffing shortages, elevator breakdowns, or severe supply chain stockouts from primary distributor. | Measure actual time-stamped put-away logs against order generation timestamps; model $L$ as stochastic. |
| **Unconstrained Clean Utility Volumetric Capacity** | Allows independent mathematical optimization of each SKU's par level. | Real-world clean utility rooms have rigid shelf dimensions; total optimal inventory exceeds physical room volume. | Verify total allocated volume: $\sum v_i S_i \le V_{\text{total}}$; implement knapsack optimization when violated. |
| **Instantaneous Restock Put-Away** | Assumes that once goods arrive on the clinical floor, they are immediately available for clinical use. | Replenishment carts left parked in hallways for hours during shift changes before technicians stock shelves. | Implement cart delivery-to-putaway time-tracking using barcode scans upon floor arrival and drawer restock. |
| **Linear and Exogenous Stockout Cost ($c_s$)** | Permits standard newsvendor critical ratio optimization with a single scalar shortage penalty. | The clinical consequence of a stockout is highly state-dependent (e.g., depends on whether patient is stable or in cardiac arrest). | Segment inventory into discrete clinical criticality tiers; assign conservative ceiling penalties to life-safety items. |
| **Strict FIFO/FEFO Physical Rotation** | Ensures inventory does not expire on the shelf by assuming oldest stock is always consumed first. | Technicians stuffing newly delivered supplies in front of old stock, pushing older items to the rear of the bin. | Audit physical expiration dates in bins quarterly; utilize gravity-fed rear-loading flow racks. |
| **Single Sourcing and Zero Substitution** | Assumes that running out of a specific SKU halts clinical care without immediate alternative. | Clinicians substituting an equivalent size catheter, syringe, or drug concentration from an adjacent bin. | Build clinical substitution logic matrices into the item master; model demand pooling across substitutable SKUs. |

---

## 12. Exercises

### Exercise 1: Periodic vs Continuous Review in Clinical Care
* **Scenario:** A 24-bed Neonatal Intensive Care Unit (NICU) stores two categories of supplies: specialized umbilical artery catheterization kits ($95/kit, demand = 0.5 kits/day) and standard sterile newborn diapers ($0.35/diaper, demand = 180 diapers/day).
* **Deliverable:** Write a 250-word policy appraisal comparing whether each item should be managed via an Automated Dispensing Cabinet (continuous telemetry) or a Visual Two-Bin Kanban system (periodic review). Justify your recommendation using unit cost, clinical criticality, space footprint, and nursing workflow disruption.
* **Acceptance Criteria:** Must explicitly address the tradeoff between capital cost/cabinet real estate and stockout risk for both SKUs.
* **Estimated Time:** 20 minutes.

### Exercise 2: Poisson Par Level Sizing for Arterial Line Kits
* **Scenario:** A Cardiovascular Intensive Care Unit (CVICU) operates with an average census of 20 patients. Daily demand for radial arterial line insertion kits follows a stationary Poisson distribution with a mean of $\lambda = 3.0\text{ kits/day}$. The unit operates on a daily periodic replenishment review schedule ($R = 1.0\text{ day}$), and the Central Sterile Supply department requires 12 hours ($L = 0.5\text{ days}$) to assemble, pick, and deliver replenishment carts.
* **Deliverable:**
  1. Calculate the expected demand over the exposure window $\mu_e$.
  2. Compute the exact par level $S$ required to guarantee a cycle service level $\alpha \ge 99.5\%$.
  3. Determine the required safety stock $SS$.
* **Acceptance Criteria:** Show the cumulative Poisson probability summation table or formula values. Par level must be a discrete integer.
* **Estimated Time:** 25 minutes.

### Exercise 3: Two-Bin Visual Kanban Sizing with Case-Pack Constraints
* **Scenario:** An adult Medical-Surgical unit consumes disposable telemetry ECG lead-wire sets. Historical daily demand averages $\mu_d = 45\text{ sets/day}$ with a daily variance of $\sigma_d^2 = 81\text{ sets}^2/\text{day}$. Replenishment runners sweep the unit twice daily ($R = 0.5\text{ days}$), and delivery turnaround takes 6 hours ($L = 0.25\text{ days}$). The lead-wire sets are packaged by the manufacturer in sealed boxes of 25 sets. The nurse manager mandates a cycle service level of $\alpha = 99.0\%$ ($z = 2.326$).
* **Deliverable:**
  1. Calculate the expected lead-time demand $\mu_L$ and lead-time standard deviation $\sigma_L$ covering the reserve bin.
  2. Compute the unconstrained minimum single bin capacity $B_{\text{min}}$.
  3. Determine the final single bin capacity $B$ enforcing the case-pack multiple of 25.
  4. Calculate the total two-bin system inventory authorization and the achieved effective service level.
* **Acceptance Criteria:** Full arithmetic steps shown; final bin sizes must be integer multiples of 25.
* **Estimated Time:** 25 minutes.

### Exercise 4: Stockout Penalty Optimization and Critical Ratio Sizing
* **Scenario:** A hospital is evaluating the par level for an emergency cricothyrotomy tray stored in the Emergency Department clean core. Unit purchase cost is $c = \$140.00$, with an annual holding rate of $r_h = 25\%$. Average daily demand is $\lambda = 0.2\text{ trays/day}$. The replenishment review period is $R = 1.0\text{ day}$ and delivery lead time is $L = 1.0\text{ day}$ ($T_e = 2.0\text{ days}$). If a tray is unavailable during an airway emergency, the immediate cost includes an emergency STAT courier from a sister hospital ($150), emergency clinical diversion and surgery mobilization ($850), and a clinical risk penalty established by the quality committee of $5,000.
* **Deliverable:**
  1. Calculate the unit holding cost per exposure window $h_e$.
  2. Calculate the total shortage cost $c_s$ and the resulting newsvendor Critical Ratio $CR$.
  3. Using the Poisson distribution ($\mu_e = \lambda T_e = 0.4$), determine the economically and clinically optimal integer par level $S^*$.
* **Acceptance Criteria:** Step-by-step arithmetic verifying whether $S^*$ satisfies $P(X \le S^*) \ge CR$.
* **Estimated Time:** 20 minutes.

### Exercise 5: Diagnosing a Behavioral Hoarding Loop from Telemetry
* **Scenario:** Review the following 10-week telemetry extract for sterile thoracostomy (chest tube) insertion trays on a Trauma Step-Down Unit (baseline occupancy: 20 beds):
  * Weeks 1–3: Consumption recorded at 10 trays/week, par level set at 15.
  * Week 4: A supply chain stockout occurs; 0 trays available for 36 hours. Nurses borrow 4 trays from the OR.
  * Weeks 5–7: Official ADC recorded consumption drops to 1 tray/week.
  * Week 8: The automated ERP replenishment job reduces the par level from 15 to 4 trays based on rolling 4-week demand.
  * Week 9: Recorded demand jumps to 12 trays in 48 hours; unit experiences 5 consecutive null-picks and an emergency courier is dispatched.
* **Deliverable:** Write a 300-word root-cause investigation memo explaining the behavioral dynamics of Weeks 5–8, detailing what occurred on the nursing floor, why the algorithm failed, and four concrete operational interventions to stabilize the unit.
* **Acceptance Criteria:** Must explicitly identify the "squirrel stash" mechanism, explain why the algorithm misread the signal, and provide both operational and technical fixes.
* **Estimated Time:** 30 minutes.

### Exercise 6: Constrained Volumetric Shelf Optimization (Knapsack Sizing)
* **Scenario:** A clean utility room wire rack has a total usable volume of $120,000\text{ cm}^3$. The unit must store three sterile kit SKUs:
  * Kit 1 (Laceration Tray): Volume $v_1 = 4,000\text{ cm}^3/\text{kit}$, mean exposure demand $\mu_1 = 6.0$, clinical criticality weight $w_1 = 1.0$.
  * Kit 2 (Thoracentesis Kit): Volume $v_2 = 8,000\text{ cm}^3/\text{kit}$, mean exposure demand $\mu_2 = 2.0$, clinical criticality weight $w_2 = 3.0$.
  * Kit 3 (Lumbar Puncture Kit): Volume $v_3 = 6,000\text{ cm}^3/\text{kit}$, mean exposure demand $\mu_3 = 3.0$, clinical criticality weight $w_3 = 2.5$.
* **Deliverable:** Formulate the integer optimization problem and determine an integer par allocation $(S_1, S_2, S_3)$ that maximizes total weighted service coverage while satisfying the physical rack volume constraint and guaranteeing at least a $95\%$ service level for each SKU.
* **Acceptance Criteria:** Explicit mathematical formulation, volumetric feasibility check ($\sum v_i S_i \le 120,000$), and service-level verification for each SKU.
* **Estimated Time:** 35 minutes.

### Exercise 7: Dynamic Census-Adjusted Par Policy Design
* **Scenario:** Design an automated daily par adjustment rule for an 18-bed Pediatric Intensive Care Unit (PICU) during winter viral season. The unit census fluctuates between 8 and 18 occupied beds. For endotracheal suction catheters, baseline usage is $\lambda_{ppd} = 4.0\text{ catheters/patient-day}$. Replenishment exposure window is $T_e = 1.0\text{ day}$.
* **Deliverable:** Formulate the dynamic par equation $S(N_t)$ as a function of daily morning census $N_t$ targeting a $99.9\%$ Poisson service level. Construct a lookup table showing the recommended integer par level for every census integer from $N = 8$ to $N = 18$.
* **Acceptance Criteria:** Tabular output of all integer values, explicit mathematical formula, and a stated policy on handling sudden midday admissions.
* **Estimated Time:** 25 minutes.

---

## 13. Self-check answers

### Answer to Exercise 2: Poisson Par Level Sizing

1. **Expected Exposure Demand:**
   $$T_e = R + L = 1.0 + 0.5 = 1.5\text{ days}$$
   $$\mu_e = \lambda \times T_e = 3.0 \times 1.5 = 4.5\text{ kits}$$
2. **Cumulative Poisson Distribution for $\mu_e = 4.5$:**
   $$P(X = k) = \frac{4.5^k e^{-4.5}}{k!}$$
   * $k = 0$: $P(0) = 0.01111$, $F(0) = 0.01111$
   * $k = 1$: $P(1) = 0.04999$, $F(1) = 0.06110$
   * $k = 2$: $P(2) = 0.11248$, $F(2) = 0.17358$
   * $k = 3$: $P(3) = 0.16872$, $F(3) = 0.34230$
   * $k = 4$: $P(4) = 0.18981$, $F(4) = 0.53211$
   * $k = 5$: $P(5) = 0.17083$, $F(5) = 0.70294$
   * $k = 6$: $P(6) = 0.12812$, $F(6) = 0.83105$
   * $k = 7$: $P(7) = 0.08236$, $F(7) = 0.91341$
   * $k = 8$: $P(8) = 0.04633$, $F(8) = 0.95974$
   * $k = 9$: $P(9) = 0.02316$, $F(9) = 0.98291$
   * $k = 10$: $P(10) = 0.01042$, $F(10) = 0.99333$
   * $k = 11$: $P(11) = 0.00426$, $F(11) = 0.99760$
   * $k = 12$: $P(12) = 0.00160$, $F(12) = 0.99920$
3. **Par Level Selection:**
   To satisfy $\alpha \ge 0.995$ ($99.5\%$):
   * At $S = 10$, $F(10) = 0.99333 < 0.995$.
   * At $S = 11$, $F(11) = 0.99760 \ge 0.995$.
   * **Par Level $S = 11\text{ kits}$**.
4. **Safety Stock:**
   $$SS = S - \mu_e = 11 - 4.5 = 6.5\text{ kits}$$

---

### Answer to Exercise 3: Two-Bin Kanban Sizing

1. **Lead Time Demand and Standard Deviation:**
   The reserve bin must cover lead time $L = 0.25\text{ days}$ (6 hours):
   $$\mu_L = \mu_d \times L = 45 \times 0.25 = 11.25\text{ sets}$$
   $$\sigma_L = \sqrt{L \times \sigma_d^2} = \sqrt{0.25 \times 81} = \sqrt{20.25} = 4.5\text{ sets}$$
2. **Unconstrained Minimum Bin Capacity:**
   For $\alpha = 0.990$, $z = 2.326$:
   $$SS_{\text{raw}} = z \times \sigma_L = 2.326 \times 4.5 = 10.467\text{ sets}$$
   $$B_{\text{min}} = \mu_L + SS_{\text{raw}} = 11.25 + 10.467 = 21.717\text{ sets}$$
3. **Case-Pack Rounding ($Q_{\text{pack}} = 25$):**
   $$B = \left\lceil \frac{21.717}{25} \right\rceil \times 25 = 1 \times 25 = 25\text{ sets}$$
4. **Total Two-Bin Inventory and Achieved Service Level:**
   $$\text{Total System Authorization} = 2 \times B = 2 \times 25 = 50\text{ sets (2 boxes of 25)}$$
   Actual buffer in reserve bin:
   $$SS_{\text{actual}} = B - \mu_L = 25 - 11.25 = 13.75\text{ sets}$$
   Achieved $z$-score:
   $$z_{\text{actual}} = \frac{13.75}{4.5} \approx 3.056$$
   Achieved cycle service level: $\Phi(3.056) = 0.9989$ ($99.89\%$), easily satisfying the $99.0\%$ requirement.

---

### Answer to Exercise 4: Stockout Penalty Optimization

1. **Holding Cost per Exposure Window:**
   $$h_e = c \times r_h \times \frac{T_e}{365.25} = 140.00 \times 0.25 \times \frac{2.0}{365.25} = \$0.19165$$
2. **Total Shortage Cost and Critical Ratio:**
   $$c_s = c_{\text{courier}} + c_{\text{labor}} + \psi_{\text{clinical}} = 150.00 + 850.00 + 5,000.00 = \$6,000.00$$
   $$CR = \frac{c_s}{c_s + h_e} = \frac{6,000.00}{6,000.00 + 0.19165} = \frac{6,000.00}{6,000.19165} \approx 0.999968$$
3. **Poisson Distribution Evaluation ($\mu_e = 0.4$):**
   $$P(X = k) = \frac{0.4^k e^{-0.4}}{k!}$$
   * $k = 0$: $P(0) = 0.670320$, $F(0) = 0.670320$
   * $k = 1$: $P(1) = 0.268128$, $F(1) = 0.938448$
   * $k = 2$: $P(2) = 0.053626$, $F(2) = 0.992074$
   * $k = 3$: $P(3) = 0.007150$, $F(3) = 0.999224$
   * $k = 4$: $P(4) = 0.000715$, $F(4) = 0.999939$
   * $k = 5$: $P(5) = 0.000057$, $F(5) = 0.999996$
4. **Optimal Par Level:**
   * At $S = 4$, $F(4) = 0.999939 < 0.999968$.
   * At $S = 5$, $F(5) = 0.999996 \ge 0.999968$.
   * **Optimal Par Level $S^* = 5\text{ trays}$**.

---

### Answer to Exercise 5: Root-Cause Memo (Grading Rubric)

A complete response must address:
* **The Clinical Mechanism (Weeks 5–7):** Following the Week 4 stockout trauma, nurses created an unmonitored "squirrel stash" (likely 25–30 trays hidden in a treatment room or breakroom). During Weeks 5–7, clinicians pulled from this stash rather than the ADC par location.
* **The Algorithmic Failure (Week 8):** The automated replenishment algorithm operated under the naive assumption that recorded transactions represent true patient demand. It interpreted the flatline as reduced clinical need and slashed the par level from 15 to 4.
* **The Crash (Week 9):** When the informal stash emptied, real clinical demand returned abruptly to the official par bin. Because the par level was now only 4, the unit stocked out immediately, triggering the emergency courier.
* **Four Concrete Remediation Steps:**
  1. Freeze automated par downward adjustments for items that have recently experienced stockouts.
  2. Implement an automated "phantom zero-demand" flag when a historically active item records zero usage for $> 7$ days.
  3. Conduct a joint Materials-Nursing "Amnesty Sweep" to return the stash to the official system.
  4. Guarantee an explicit 15-minute replenishment SLA from central supply to rebuild nursing trust.

---

### Answer to Exercise 6: Constrained Volumetric Knapsack Allocation

1. **Minimum Par Level Constraints (95% Service Level):**
   * **Kit 1 ($\mu_1 = 6.0$):** Poisson $F(k) \ge 0.95 \implies S_{1, \text{min}} = 10$ ($F(10) = 0.9574$). Volume: $10 \times 4,000 = 40,000\text{ cm}^3$.
   * **Kit 2 ($\mu_2 = 2.0$):** Poisson $F(k) \ge 0.95 \implies S_{2, \text{min}} = 5$ ($F(5) = 0.9834$). Volume: $5 \times 8,000 = 40,000\text{ cm}^3$.
   * **Kit 3 ($\mu_3 = 3.0$):** Poisson $F(k) \ge 0.95 \implies S_{3, \text{min}} = 6$ ($F(6) = 0.9665$). Volume: $6 \times 6,000 = 36,000\text{ cm}^3$.
   * **Baseline Feasibility:**
     $$V_{\text{min}} = 40,000 + 40,000 + 36,000 = 116,000\text{ cm}^3 \le 120,000\text{ cm}^3$$
     Remaining unallocated volume: $120,000 - 116,000 = 4,000\text{ cm}^3$.
2. **Marginal Volume Allocation:**
   The remaining $4,000\text{ cm}^3$ can only accommodate an additional unit of Kit 1 ($v_1 = 4,000\text{ cm}^3$). Kits 2 and 3 require $8,000$ and $6,000\text{ cm}^3$, respectively.
   Adding 1 unit to Kit 1 increases $S_1$ to 11:
   $$F_1(11) = 0.9799, \quad \Delta F_1 = 0.9799 - 0.9574 = 0.0225$$
   Weighted benefit: $w_1 \times \Delta F_1 = 1.0 \times 0.0225 = 0.0225$.
3. **Optimal Integer Allocation:**
   * **Kit 1 Par Level:** $S_1 = 11\text{ kits}$ (Achieved CSL: $97.99\%$).
   * **Kit 2 Par Level:** $S_2 = 5\text{ kits}$ (Achieved CSL: $98.34\%$).
   * **Kit 3 Par Level:** $S_3 = 6\text{ kits}$ (Achieved CSL: $96.65\%$).
   * **Total Volume Used:**
     $$(11 \times 4,000) + (5 \times 8,000) + (6 \times 6,000) = 44,000 + 40,000 + 36,000 = 120,000\text{ cm}^3$$
     The physical rack space is exactly 100% saturated.

---

### Answer to Exercise 7: Dynamic Census-Adjusted Par Policy Design

1. **Mathematical Model:**
   For daily census $N_t$, daily expected demand is $\mu(N_t) = \lambda_{ppd} \times N_t \times T_e = 4.0 \times N_t \times 1.0 = 4.0 N_t$.
   The dynamic par level is the integer $S(N_t)$ satisfying:
   $$S(N_t) = \min \left\{ S \in \mathbb{Z} \;\middle|\; \sum_{k=0}^S \frac{(4 N_t)^k e^{-4 N_t}}{k!} \ge 0.999 \right\}$$
2. **Census Par Lookup Table:**

```text
┌─────────────────┬───────────────────┬───────────────────┬─────────────────────┐
│ Census ($N_t$)  │ Exp Demand ($\mu$)│ Dynamic Par ($S$) │ Safety Stock ($SS$) │
├─────────────────┼───────────────────┼───────────────────┼─────────────────────┤
│ 8 patients      │ 32.0 catheters    │ 51 catheters      │ 19.0 catheters      │
│ 9 patients      │ 36.0 catheters    │ 56 catheters      │ 20.0 catheters      │
│ 10 patients     │ 40.0 catheters    │ 61 catheters      │ 21.0 catheters      │
│ 11 patients     │ 44.0 catheters    │ 66 catheters      │ 22.0 catheters      │
│ 12 patients     │ 48.0 catheters    │ 71 catheters      │ 23.0 catheters      │
│ 13 patients     │ 52.0 catheters    │ 75 catheters      │ 23.0 catheters      │
│ 14 patients     │ 56.0 catheters    │ 80 catheters      │ 24.0 catheters      │
│ 15 patients     │ 60.0 catheters    │ 85 catheters      │ 25.0 catheters      │
│ 16 patients     │ 64.0 catheters    │ 90 catheters      │ 26.0 catheters      │
│ 17 patients     │ 68.0 catheters    │ 94 catheters      │ 26.0 catheters      │
│ 18 patients     │ 72.0 catheters    │ 99 catheters      │ 27.0 catheters      │
└─────────────────┴───────────────────┴───────────────────┴─────────────────────┘
```

3. **Intraday Admission Surge Policy:** If unit census increases by $\ge 3$ admissions mid-shift, the EHR ADT interface automatically pushes an emergency top-off ticket to the central store to bridge the gap before the next morning's scheduled sweep.

---

## 14. Interview questions

### Question 1: ADC Telemetry vs Two-Bin Kanban Tradeoffs
* **Question:** "If you were appointed Director of Supply Chain at a 500-bed hospital, how would you decide which clinical supplies belong inside Automated Dispensing Cabinets versus open visual Two-Bin Kanban racks?"
* **Strong Answer Demonstrates:** The candidate recognizes that ADCs carry high capital costs ($50k–$100k per unit), high spatial footprints, and queuing friction for nursing staff. They articulate a multi-attribute decision matrix: ADCs should be reserved for high-value items, controlled substances (DEA Schedule II–V), items requiring tight lot/serial traceability (UDI), and high-alert medications subject to Joint Commission security rules. Low-to-moderate cost, high-turnover medical-surgical commodities (IV flushes, gauze, tubing) belong in two-bin Kanban racks, which provide superior visual ergonomics, zero electronic access latency, and lower total system cost.
* **Weak Answer Reveals:** Suggesting that "everything should be placed in automated cabinets for maximum control," ignoring capital constraints, nurse workflow bottlenecks, and clean utility spatial limits.

### Question 2: The Nurse Hoarding Feedback Loop
* **Question:** "You notice that whenever your analytics team slashes par levels on low-velocity items, those exact items suffer massive clinical stockouts three weeks later. What operational phenomenon is occurring, and how do you fix it?"
* **Strong Answer Demonstrates:** Clear identification of the behavioral hoarding ("squirrel stash") feedback loop. The candidate explains that perceived supply unreliability causes clinicians to hide inventory in unmonitored drawers. The ERP interprets zero floor scans as falling demand and cuts the par level. When the unofficial stash runs dry, demand returns to an under-buffered bin, causing a severe stockout. Remediation involves freezing downward par adjustments for items with high coefficient of variation, conducting collaborative "no-fault amnesty" sweeps with nursing leadership, and implementing transparent replenishment SLAs.
* **Weak Answer Reveals:** Blaming the nurses for "stealing or violating policy" and advocating for disciplinary measures or video surveillance, failing to realize that hoarding is a systemic, rational clinician coping response to poor supply chain reliability.

### Question 3: Incorporating Clinical Criticality into Par Sizing
* **Question:** "Why is it mathematically and clinically inappropriate to apply a single 95% cycle service level target across all hospital SKUs?"
* **Strong Answer Demonstrates:** The candidate derives the economic-clinical critical ratio ($CR = \frac{c_s}{c_s + h_e}$) and explains that stockout costs are highly asymmetric. For an inexpensive, life-saving item (e.g., an endotracheal tube or vasopressor ampule), the holding cost is pennies, while the clinical shortage penalty involves patient harm, regulatory citations, and hundreds of dollars in expedited courier fees. Sizing this item to 95% guarantees 1 stockout every 20 replenishment cycles—an unacceptable clinical failure rate. Critical items require $99.9\%$ or higher availability, whereas non-critical commodities can be economically buffered at $90\%\text{--}95\%$.
* **Weak Answer Reveals:** Believing that 95% is an industry "best practice standard" that represents equal fairness across clinical departments, showing no comprehension of asymmetric loss functions.

### Question 4: Handling Overdispersion in Clinical Demand
* **Question:** "When analyzing daily consumption of sterile IV flush syringes, you find that the sample mean is 120 and the sample variance is 360. Why will a standard Poisson inventory model fail, and what mathematical distribution should you use?"
* **Strong Answer Demonstrates:** The candidate identifies that the Variance-to-Mean Ratio ($\text{VMR} = 360/120 = 3.0$) indicates severe **overdispersion**, violating the fundamental Poisson assumption of equidispersion ($\sigma^2 = \mu$). Under-dispersed models like the Poisson will severely underestimate tail quantiles, resulting in an undersized par level that stocks out during clinical surges. The candidate recommends fitting a **Negative Binomial distribution** or a compound Poisson-Gamma model to properly buffer the fat right tail.
* **Weak Answer Reveals:** Suggesting a standard Gaussian (Normal) distribution without checking discrete boundary limits, or asserting that Poisson is always appropriate for discrete counts regardless of variance.

### Question 5: Managing Clean Utility Spatial Constraints
* **Question:** "What do you do when your mathematically optimal par levels exceed the physical cubic capacity of the nursing unit's clean utility room?"
* **Strong Answer Demonstrates:** The candidate articulates the multi-item constrained knapsack formulation. Rather than arbitrarily cutting all pars by a flat percentage, they prioritize items using the marginal clinical availability gain per cubic centimeter of shelf space ($\frac{\Delta \beta_i \cdot w_i}{v_i}$). Furthermore, they explore operational frequency changes: increasing replenishment review frequency (e.g., moving from daily $R=1.0$ to twice-daily $R=0.5$ sweeps), which cuts the exposure window and reduces required physical safety inventory by roughly $30\%$.
* **Weak Answer Reveals:** Suggesting that the hospital simply construct a larger clean utility room (ignoring HVAC and hospital capital costs) or stacking boxes on the floor (violating Joint Commission infection control codes).

### Question 6: Mitigating ADC Discrepancies
* **Question:** "During a clinical audit, you find that 15% of transactions in your intensive care unit ADCs generate inventory discrepancy flags. How do you triage and resolve this issue?"
* **Strong Answer Demonstrates:** The candidate differentiates between clinical diversion, count fatigue, and packaging mismatch. They analyze discrepancy patterns: are discrepancies concentrated on high-abuse controlled substances (potential diversion requiring immediate pharmacy/DEA review) or bulk items like oral tablets and syringes? If on bulk items, the root cause is often "lazy counting" or complex multi-dose blister strips. Solutions include implementing single-unit packaging, converting bulk items to visual Kanban, and mandating true blind counts.
* **Weak Answer Reveals:** Treating all discrepancies as criminal diversion, or conversely, turning off the discrepancy alert system in software to eliminate administrative notifications.

### Question 7: Dynamic Census Adjustment Mechanics
* **Question:** "How does integrating Electronic Health Record (EHR) Admission, Discharge, and Transfer (ADT) data into replenishment engines change inventory performance compared to static pars?"
* **Strong Answer Demonstrates:** The candidate explains that acute-care hospital demand is non-stationary and directly proportional to active census and surgical scheduling. Static pars result in chronic over-stocking during low-occupancy periods (driving product expiration) and chronic stockouts during census surges. Feeding ADT census and forward surgical bookings into the par calculation enables dynamic scaling of order-up-to levels ($S_t = f(N_t)$), stabilizing clinical fill rates while reducing average system inventory.
* **Weak Answer Reveals:** Assuming that daily census fluctuations are too minor to warrant system integration, or confusing clinical census with hospital billing records.

### Question 8: Joint Commission Compliance in Storage Sizing
* **Question:** "A materials technician suggests saving clean utility space by stacking supply cartons directly on the floor and right up to the ceiling tiles. How do you respond from a regulatory and operational perspective?"
* **Strong Answer Demonstrates:** The candidate immediately cites Joint Commission and fire safety mandates: items must be at least 8 inches off the floor to prevent floor-cleaning chemical contamination and water damage, at least 18 inches below fire sprinkler heads (or 36 inches for dense racks) under NFPA 13/101 to maintain fire suppression spray cones, and corrugated cardboard shipping boxes are strictly prohibited due to fungal spore risks. The technician's suggestion would trigger severe accreditation citations and compromise sterile barrier integrity.
* **Weak Answer Reveals:** Approving the suggestion purely to maximize storage density, displaying total ignorance of hospital infection control and fire safety regulations.

### Question 9: USP <800> Hazardous Drug Par Room Integration
* **Question:** "How do USP <800> regulations impact the inventory placement and par sizing of antineoplastic agents and hazardous oral medications in nursing med rooms?"
* **Strong Answer Demonstrates:** The candidate recognizes that USP <800> strictly regulates the environmental and occupational exposure risks of hazardous drugs. They explain that intact unit-dose medications may be stored in an ADC, but must reside in dedicated, sealed, clearly labeled drawers with non-porous spill containment. They also highlight that hazardous medication par levels must be co-located with dedicated PPE kits (chemotherapy-tested gloves, gowns) and specialized disposal containers, and that replenishment technicians must be trained in hazardous handling and spill decontamination.
* **Weak Answer Reveals:** Assuming that hazardous drugs can be intermingled freely with routine oral medications as long as they are in blister packs, showing unfamiliarity with USP compounding and storage standards.

### Question 10: Low-Unit-of-Measure (LUM) Distribution Economics
* **Question:** "Why have leading health systems invested tens of millions of dollars in off-campus Consolidated Service Centers (CSCs) to perform Low-Unit-of-Measure (LUM) replenishment, rather than relying on direct distributor drops to hospital loading docks?"
* **Strong Answer Demonstrates:** The candidate explains the multi-echelon labor and spatial economics. Distributor drops to hospital docks arrive in master corrugated cartons that require extensive floor-level de-boxing, cluttering clean rooms and violating infection control bans. By consolidating inventory at an off-campus CSC, the health system can bulk-purchase at favorable tiers, de-box under controlled industrial conditions, and build unit-specific, shelf-sequenced LUM totes. This cuts hospital floor inventory by 30%–50%, frees clinical square footage for revenue-generating patient care, and recovers hundreds of nursing and supply tech hours per week.
* **Weak Answer Reveals:** Viewing the CSC merely as an expensive redundant warehousing layer that adds double-handling costs, failing to understand the hospital-level labor, infection control, and spatial savings.

---

## 15. Key terms

* **Periodic Automatic Replenishment (Par / $S$):** The target inventory ceiling for a point-of-use clinical location under periodic review.
* **Reorder Point ($s$):** The predetermined inventory threshold that triggers an unscheduled replenishment order in continuous-review systems.
* **Review Period ($R$):** The fixed calendar time interval between successive inventory audits or replenishment sweeps.
* **Replenishment Lead Time ($L$):** The total elapsed time from order trigger to physical shelf put-away in the clinical clean room.
* **Exposure Window ($T_e$):** The total time span during which inventory is vulnerable to stockouts ($T_e = R + L$ for periodic, $T_e = L$ for continuous).
* **Automated Dispensing Cabinet (ADC):** A decentralized, computerized point-of-use medication and supply storage vault (e.g., Pyxis, Omnicell).
* **Visual Two-Bin Kanban:** A physical replenishment system utilizing primary and reserve bins to signal reorder triggers upon primary bin exhaustion.
* **Clean Utility Room:** A positively pressurized, sterile hospital room dedicated to storing medical-surgical supplies.
* **Unit Patient Census ($N_t$):** The active number of occupied patient beds driving clinical supply consumption.
* **Case Mix Index (CMI):** A relative numerical score reflecting the clinical severity and resource intensity of a patient cohort.
* **Null-Pick / Zero-Pick:** An event where a clinician accesses a storage location and encounters zero on-hand stock.
* **Discrepancy Count:** A numerical mismatch between the physical inventory count and the perpetual electronic system balance.
* **Blind Count:** A mandatory verification workflow where users must input physical counts without seeing the system's expected balance.
* **Squirrel Stash:** Unofficial, hidden inventory hoarded by clinical personnel outside official supply systems to guard against stockouts.
* **STAT / Hot-Shot Delivery:** An unscheduled, expedited courier dispatch executed to resolve an immediate clinical stockout.
* **Case-Pack Rounding ($Q_{\text{pack}}$):** Sizing par quantities to match the manufacturer's inner carton or sealed packaging multiples.
* **Cycle Service Level ($\alpha$):** The probability of completing a replenishment cycle without encountering any stockout event.
* **Item Fill Rate ($\beta$):** The percentage of total clinical unit demand satisfied immediately from on-hand point-of-use inventory.
* **Par Creep:** The systematic upward drift of par allocations caused by unmanaged administrative overrides.
* **Volumetric Shelf Footprint:** The three-dimensional cubic space occupied by an item's par allocation relative to shelf dimensions.
* **Stockout Penalty Function ($c_s$):** An asymmetric cost formulation penalizing shortages based on labor disruption, delay, and patient risk.
* **Return to Stock (RTS):** The verified transaction logging an unused, intact supply item returned to storage.
* **Central Sterile Processing Department (SPD):** The hospital department responsible for cleaning, sterilizing, and assembling reusable surgical instruments and trays.
* **Point of Use (POU):** The physical clinical location (med room, clean utility, trauma bay) where supplies are dispensed directly for patient care.
* **Overdispersion:** A statistical condition where demand variance exceeds the mean ($\sigma^2 > \mu$), requiring Negative Binomial modeling.
* **First-In, First-Out (FIFO):** Physical inventory rotation ensuring oldest stock is consumed before newer deliveries to prevent expiration.
* **Corrugated Cardboard Ban:** Infection control regulation prohibiting porous external shipping cartons inside sterile clean cores.
* **Drug Supply Chain Security Act (DSCSA):** Federal legislation mandating unit-level traceability and serialization for prescription pharmaceuticals.
* **Low-Unit-of-Measure (LUM):** A logistics model delivering supplies in exact clinical unit-of-use packages, eliminating bulk ward inventory.
* **Consolidated Service Center (CSC):** A centralized off-campus health system facility managing centralized warehousing, sterilization, and LUM kitting.
* **Look-Alike / Sound-Alike (LASA):** Medications or supplies with confusingly similar names or appearances requiring segregation.

---

## 16. Sources

1. **American Society of Health-System Pharmacists (ASHP).** (2020). ASHP Guidelines on the Safe Use of Automated Dispensing Cabinets. *American Journal of Health-System Pharmacy*, 77(14), 1152–1167. DOI: 10.1093/ajhp/zxaa142.
2. **The Joint Commission.** (2024). *Comprehensive Accreditation Manual for Hospitals (CAMH): Medication Management (MM) and Infection Prevention and Control (IC)*. Oakbrook Terrace, IL: Joint Commission Resources.
3. **U.S. Drug Enforcement Administration (DEA).** (2023). Title 21 Code of Federal Regulations, Part 1301 (Registration of Manufacturers, Distributors, and Dispensers) and Part 1304 (Records and Reports of Registrants). Washington, DC: U.S. Government Publishing Office.
4. **U.S. Food and Drug Administration (FDA).** (2022). *Drug Supply Chain Security Act (DSCSA) Implementation: Core Requirements for Identification and Traceability*. Silver Spring, MD: Center for Drug Evaluation and Research.
5. **National Fire Protection Association (NFPA).** (2024). *NFPA 99: Health Care Facilities Code* and *NFPA 101: Life Safety Code*. Quincy, MA: National Fire Protection Association.
6. **de Vries, J., & Huijsman, R.** (2011). Supply chain management in health services: an overview. *Supply Chain Management: An International Journal*, 16(3), 159–165. DOI: 10.1108/13598541111127146.
7. **Landry, S., & Philippe, R.** (2004). How merchant hospitals can transform their supply chains. *Supply Chain Forum: An International Journal*, 5(2), 78–86. DOI: 10.1080/16258312.2004.11517134.
8. **Rosales, C. R., Magazine, M., & Rao, U.** (2014). The operational and financial impacts of RFID-driven two-bin Kanban systems in hospital wards. *Production and Operations Management*, 23(5), 762–779. DOI: 10.1111/poms.12078.
9. **Kelle, P., Woosley, J., & Schneider, H.** (2012). Pharmaceutical allocation and inventory control in a hospital environment with automated dispensing cabinets. *International Journal of Production Economics*, 140(2), 704–713. DOI: 10.1016/j.ijpe.2012.01.036.
10. **Moons, K., Waeyenbergh, G., & Pintelon, L.** (2019). Measuring the logistical performance of internal hospital supply chains: a systematic review. *Health Care Management Science*, 22(1), 80–98. DOI: 10.1007/s10729-018-9430-6.

---

## 17. Links

* **Module Index:** [`_index.md`](_index.md)
* **Survey File:** [`../../../course/33_healthcare_provider_life_sciences.md`](../../../course/33_healthcare_provider_life_sciences.md)
* **Preceding Topic:** [`01_provider-network-item-master.md`](01_provider-network-item-master.md) — The Provider Network and Item Master
* **Succeeding Topic:** [`03_or-case-carts-preference-cards.md`](03_or-case-carts-preference-cards.md) — Operating Rooms, Case Carts, and Preference Cards
* **Related Topics:**
  * [`04_consignment-vmi.md`](04_consignment-vmi.md) — Consignment and Vendor-Managed Inventory
  * [`05_udi-traceability.md`](05_udi-traceability.md) — UDI and Traceability
  * [`06_shortage-recall-management.md`](06_shortage-recall-management.md) — Shortage and Recall Management
  * [`08_provider-metrics-governance.md`](08_provider-metrics-governance.md) — Provider Metrics and Governance
* **Foundational Methods:**
  * [`../../02_methods/10_inventory_optimization_multi_echelon/04_safety-stock.md`](../../02_methods/10_inventory_optimization_multi_echelon/04_safety-stock.md) — Safety Stock Sizing
  * [`../../02_methods/10_inventory_optimization_multi_echelon/06_policy-families.md`](../../02_methods/10_inventory_optimization_multi_echelon/06_policy-families.md) — Inventory Policy Families $(s, S)$ and $(R, S)$
  * [`../../01_foundations/01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md`](../../01_foundations/01A_foundations_units_time_assumptions/01_unit-of-measure-hierarchy.md) — Unit of Measure Hierarchies and Case-Pack Rounding
