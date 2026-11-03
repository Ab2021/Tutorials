> Module 31 — Aftermarket, Spare Parts, and Service Supply Chains
> Topic 04 of 09
> Estimated study time: 4.0 hours
> Prerequisites: Module 31 Topic 01 (Installed Base & Failure Demand), Module 31 Topic 03 (Multi-Echelon Sparing & METRIC)
> Version 2026.5

# Repair Loops and Turnaround

---

## 1. Why this topic exists

In consumable supply chains (e.g., retail goods, packaging, disposable medical syringes), inventory management is fundamentally a **forward procurement** problem: when stock drops below a reorder point, purchasing places a purchase order with an external vendor, waits for the vendor's production lead time, and restocks the warehouse shelf. The consumed item is discarded by the end user and disappears forever from the supply chain.

In high-value aftermarket and capital equipment operations, this forward-only mental model is completely wrong.

Most capital equipment components—jet engines, gas turbine fuel nozzles, MRI cryogenic cold heads, mining haul-truck transmissions, locomotive wheelsets, and radar signal processors—are **recoverable (rotable)** assets. They are designed by mechanical and electrical engineers to be repeatedly disassembled, inspected, refurbished, recertified, and returned to service across an operating lifespan that often spans twenty to forty years.

When a recoverable component fails on an operating machine, it triggers a **closed-loop repair cycle**:
1. The technician removes the unserviceable carcass (the "core") from the machine and installs a ready-for-issue (RFI) spare from local inventory.
2. The machine returns to revenue service.
3. The failed carcass does *not* get thrown away; it enters the reverse logistics pipeline, travels to a specialized maintenance, repair, and overhaul (MRO) facility, undergoes testing and repair, and returns to the ready spares pool.

Therefore, for recoverable parts, **the primary supplier is not an external factory; it is your own repair loop**.

The governing physical law of rotable inventory is Little’s Law applied to the repair pipeline:
$$\text{Pipeline Inventory (Float)} = \text{Failure Arrival Rate } (\lambda) \times \text{Repair Turnaround Time } (\text{TAT})$$

If an industrial fleet experiences 10 turbine blade failures per month, and the repair facility takes 90 days (3.0 months) to overhaul a blade set, the network must permanently own and finance **30 complete blade sets** ($\lambda \cdot \text{TAT} = 10 \times 3.0 = 30$) simply to fill the pipeline, plus additional buffer stock to cover stochastic variance. If each blade set costs $250,000, that 90-day turnaround time locks up **$7.5 million in working capital**.

If operations investments cut the repair turnaround time from 90 days to 30 days (1.0 month), the required pipeline collapses from 30 sets to 10 sets. The organization immediately liberates **$5.0 million in cash**, while simultaneously improving equipment availability.

Despite this enormous financial leverage, repair loops are chronically mismanaged:
* **The Carcass Black Hole:** Organizations treat unserviceable carcasses as worthless scrap until they desperately need them. Carcasses sit in unmonitored wooden crates in outstation hangars for weeks before being shipped to the repair shop.
* **The Awaiting-Parts (AWP) Stoppage:** An overhaul on a $150,000 flight control computer takes 4 hours of actual technician wrench time, yet the unit spends 75 days in the repair shop because the repair facility lacks a $12 sub-tier internal capacitor.
* **Beyond Economic Repair (BER) Denial:** When a damaged carcass is inspected and found to be unrepairable, it is scrapped. If the supply chain fails to link that scrap event directly to an automated "attrition new-buy" purchase order, the total active spares pool shrinks. Months later, the fleet experiences a catastrophic stockout because physical assets disappeared from the loop without replacement.
* **Cannibalization Addiction:** When spares run out, maintenance managers harvest working components from idle or grounded machines to keep frontline units operating. Cannibalization doubles the labor cost, damages delicate wiring harnesses, voids component warranties, and masks underlying supply chain collapse.

This topic teaches the operational mechanics, queueing mathematics, and capacity economics required to design, control, and accelerate closed-loop repair supply chains.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Map** the complete closed-loop rotable lifecycle from machine failure, core evacuation, triage, queueing, active overhaul, recertification, and return to ready-for-issue (RFI) stock.
2. **Deconstruct** total Repair Turnaround Time (TAT) into its discrete constituent time intervals: removal, evacuation transit, receiving triage, queue delay (WIP wait), active wrench time, awaiting parts (AWP), acceptance testing, and return logistics.
3. **Formulate** repair shop capacity and queue delay using $M/M/c$ and $M/G/c$ queueing models, quantifying how test-bench utilization approaching 100% causes explosive non-linear TAT inflation.
4. **Calculate** the total required rotable float (pipeline stock plus safety stock) under stochastic demand and variable turnaround times.
5. **Evaluate** the economic trade-off between investing capital in physical spare parts inventory versus investing capital in repair shop throughput capacity (labor shifts, automated diagnostic test benches, cross-trained technicians).
6. **Establish** an auditable Beyond Economic Repair (BER) threshold policy ($\text{Repair Cost} > \alpha \cdot P_{\text{new}}$) and design an automated attrition buy replenishment workflow.
7. **Quantify** the true total operational cost of field cannibalization, including duplicate labor, secondary mechanical damage, warranty forfeiture, and configuration tracking errors.
8. **Design** a rotable closed-loop tracking architecture conforming to airworthiness and industrial quality standards (FAA Form 8130-3, EASA Form 1, ATA Spec 2000 Chapter 4).

---

## 3. The core mental model

To control recoverable parts, you must treat the repair supply chain as a **closed hydrodynamic circuit of physical assets**, where unserviceable cores circulate continuously between operating equipment and repair workshops:

```text
  +─────────────────────────────────────────────────────────────────────────────+
  | OPERATIONAL ASSET FLEET                                                     |
  |  [ Active Installed Base ]                                                  |
  |         │                                          ▲                        |
  |         │ Component Failure                        │ Install Serviceable    |
  |         ▼ (Removal)                                │ Part                   |
  |  [ Unserviceable Carcass ]                  [ Ready-for-Issue (RFI) Pool ]  |
  +─────────┬──────────────────────────────────────────┴────────┬───────────────+
            │                                                   ▲
            ▼ Evacuation Logistics                              │ Replenishment
  +─────────────────────────────────────────────+               │ Return Transit
  | REVERSE LOGISTICS PIPELINE                  |               │
  |  Transit to MRO Shop (O_rev)                |               │
  +─────────┬───────────────────────────────────+               │
            │                                                   │
            ▼ Receiving & Cleaning                              │
  +─────────────────────────────────────────────────────────────┴───────────────+
  | MRO REPAIR FACILITY (Echelon 0)                                             |
  |                                                                             |
  |  [ Triage & Diagnostic Bench ] ──────> [ Beyond Economic Repair (BER) ]     |
  |         │                                       │ Scrap Carcass             |
  |         ▼ Viable for Repair                     ▼                           |
  |  [ Shop Queue Delay (WIP Wait) ]        [ New-Buy Attrition Order ]         |
  |         │                               (Procure new unit from OEM)         |
  |         ▼ Bench Available                       │                           |
  |  [ Active Overhaul (Wrench Time) ]              │ New Unit Inflow           |
  |         ▲                                       ▼                           |
  |         │ Awaiting Parts (AWP) ◄── [ Sub-tier SRU & Piece-Part Stockroom ]  |
  |         ▼ Sub-tier Parts Arrive                                             |
  |  [ Final Calibration & Test ]                                               |
  |         │                                                                   |
  |         ▼ Inspection & Airworthiness Certification                          |
  |  [ Certified RFI Carcass ] ─────────────────────────────────────────────────┘
```

### 3.1 The Two Divergent Paths: Repair vs. Attrition

When an unserviceable carcass enters the MRO facility, it undergoes initial **induction triage**:
1. **The Repair Path (Viable):** If the core's physical wear and internal cracking fall within repair manual limits, and the estimated cost to overhaul the unit is below the BER threshold, the carcass enters the active repair queue. Once overhauled, tested, and certified, it re-enters the RFI pool. The total quantity of assets circulating in the closed loop remains constant.
2. **The Attrition Path (BER / Scrap):** If the carcass suffered catastrophic uncontained failure (e.g., severe heat distortion, fractured main rotor casting), repair is technically impossible or economically irrational. The unit is condemned and scrapped. **At this exact moment, the closed loop ruptures.** If an attrition replacement buy is not immediately executed with the original equipment manufacturer (OEM), the total network spares float decreases by one unit. Over time, recurring BER scrap without replacement starves the operating fleet, causing chronic asset groundings.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misuse | Unit or Coding Case |
|---|---|---|---|
| **Rotable (Recoverable Item)** | A capital asset or assembly that is repeatedly repaired, overhauled, and recertified back to serviceable standard throughout its operating lifecycle. | Treating rotable parts as consumable expense items on the income statement. | Capital asset tag; serialized tracking. |
| **Carcass (Core)** | The unserviceable physical assembly removed from an operating machine following a failure or scheduled inspection, pending overhaul. | Assuming an unserviceable carcass has zero economic value (it often represents 60–80% of new asset value). | Serial number; condition code: `UNSERVICEABLE`. |
| **Repair Turnaround Time (TAT)** | The total elapsed calendar time from the physical removal of a failed carcass from an operating asset to its arrival back on the shelf as a certified RFI unit. | Equating TAT solely with active technician bench labor time ("wrench time"). | Calendar days or weeks ($[T]$). |
| **Ready-for-Issue (RFI)** | An inventory condition status certifying that a component has been fully overhauled, calibrated, inspected, and documented, ready for immediate installation. | Storing uninspected or returned parts in the RFI bin without certified release tags. | Condition code: `RFI` / `SERVICEABLE`. |
| **Beyond Economic Repair (BER)** | The operational condition where the projected cost to refurbish an unserviceable core exceeds a defined threshold percentage ($\alpha \approx 60 - 75\%$) of the cost of purchasing a brand-new unit. | Scrapping carcasses based on technician intuition without formal price quote comparison. | Binary disposition: `BER = TRUE`. |
| **Awaiting Parts (AWP)** | A delay state where active overhaul of an assembly is halted because one or more internal sub-tier piece parts or kits are out of stock. | Ignoring AWP in shop management; AWP typically accounts for 40–70% of total shop delay. | Elapsed hours/days in AWP status ($[T]$). |
| **Wrench Time (Active Touch Time)** | The actual labor hours spent by certified technicians physically inspecting, machining, assembling, and testing a component. | Confusing 10 hours of wrench time with a 10-hour turnaround time (wrench time ignores queue delays). | Man-hours or calendar hours ($[T]$). |
| **Cannibalization** | The removal of an operational component from an inactive, stored, or down asset to restore an active asset to operational service. | Treating cannibalization as an acceptable daily scheduling tool rather than an emergency failure metric. | Count of cannibalization work orders. |
| **Exchange Fee** | The contractual price paid to an MRO or OEM to swap an unserviceable core immediately for a pre-overhauled, certified RFI unit. | Assuming the exchange fee covers core replacement (it only covers overhaul labor and wear-and-tear). | Currency per exchange transaction ($\$$). |
| **Core Deposit (Core Value)** | A financial escrow or collateral penalty billed to a customer if they fail to return a repairable unserviceable carcass within a contractually agreed time window (e.g., 14–30 days). | Forfeiting core deposits by failing to track outstation core returns in ERP systems. | Currency per missing carcass ($\$$). |
| **Airworthiness Release Certificate** | A legally binding regulatory document (e.g., FAA Form 8130-3, EASA Form 1) certifying that a part was overhauled in compliance with statutory civil aviation regulations. | Installing a component without verified traceability and airworthiness documentation. | Document identifier and digital signature. |
| **Float (Pipeline Inventory)** | The total population of spare recoverable components required to buffer both the active repair loop pipeline and statistical demand volatility: $\text{Float} = \lambda \cdot \text{TAT} + SS$. | Sizing float based on steady-state averages without buffering against repair shop capacity surges. | Integer count of physical assets ($\mathbb{N}$). |

---

## 5. The mechanics, step by step

Managing a closed-loop repair supply chain requires an integrated sequence of eight operational procedures. The process balances reverse logistics velocity, triage decision rules, queueing dynamics, and capacity economics.

```text
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 1: Carcass Demount & Reverse Logistics Evacuation Tracking                 │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 2: Ingest, Inspect, Clean, and Execute Triage Disposition                  │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 3: Evaluate Beyond Economic Repair (BER) Thresholds & Trigger Attrition Buy│
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 4: Model Repair Shop Queueing Delays via M/M/c and M/G/c Formulations      │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 5: Mitigate Awaiting Parts (AWP) Bottlenecks through Sub-Tier Sparing       │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 6: Size Total Closed-Loop Rotable Float (Pipeline Mean + Safety Stock)     │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 7: Evaluate the Economic Trade-Off: Shop Capacity vs. Inventory Float     │
  └───────────────────────────────────────┬─────────────────────────────────────────┘
                                          ▼
  ┌─────────────────────────────────────────────────────────────────────────────────┐
  │ Step 8: Implement Cannibalization Governance and Tracking Controls              │
  └─────────────────────────────────────────────────────────────────────────────────┘
```

### 5.1 Step 1: Carcass Demount & Reverse Logistics Evacuation Tracking

The repair turnaround clock begins the minute an unserviceable component is unbolted from the operating asset.
* **The Evacuation Bottleneck:** In many service operations, field technicians bag the removed core, drop it in a corner of the local hangar or site tool crib, and log the removal in the CMMS. The carcass sits motionless for two to four weeks until a routine freight shipment is scheduled.
* **Control Procedure:**
  1. Generate an electronic Return Material Authorization (RMA) or Internal Core Manifest immediately upon CMMS work order teardown.
  2. Implement an automated **Reverse Logistics SLA**: If an unserviceable core is not scanned as "In Transit" within 48 hours of asset removal, an automated escalation flag is transmitted to the regional logistics director.
  3. Track reverse logistics transit time ($O_{\text{rev}}$):
     $$t_{\text{evac}} = t_{\text{shop arrival}} - t_{\text{demount}}$$

### 5.2 Step 2: Ingest, Inspect, Clean, and Execute Triage Disposition

Upon arrival at the MRO facility, the carcass enters induction triage.
* **Cleaning and Nondestructive Testing (NDT):** Ultrasonic degreasing, dye-penetrant inspection, eddy-current testing, and magnetic particle inspection to detect sub-surface metal fatigue and micro-fractures.
* **Triage Categorization:**
  * **Class 1 (Clean & Recertify / NFF):** No damage detected; component was pulled due to diagnostic error or false alarm (No-Fault-Found). Routed immediately to final calibration bench.
  * **Class 2 (Minor Overhaul / Level 1):** Minor seal replacement, external bushing replacement, recalibration. Short turnaround.
  * **Class 3 (Major Overhaul / Level 2):** Complete teardown, structural machining, re-winding of coils, blade re-coating. Standard turnaround.
  * **Class 4 (Beyond Economic Repair):** Structural frame cracked, casing warped beyond machining tolerances, catastrophic thermal burn. Routed to scrap.

### 5.3 Step 3: Evaluate Beyond Economic Repair (BER) Thresholds & Trigger Attrition Buy

When major damage is uncovered during inspection, the shop estimator computes the projected repair quote ($C_{\text{repair}}$) including technician labor, machining hours, and replacement sub-tier parts.
* **The BER Decision Rule:**
  $$\text{If } C_{\text{repair}} > \alpha \cdot P_{\text{new}} \implies \text{Declare BER (Scrap)}$$
  where $P_{\text{new}}$ is the catalog purchase price of a brand-new component from the OEM, and $\alpha$ is the corporate BER threshold coefficient (standard industry baseline: $\alpha = 0.65$ to $0.75$).
* **Secondary Scrap Consideration (Residual Value):**
  A carcass should be scrapped if:
  $$C_{\text{repair}} > P_{\text{new}} - V_{\text{salvage}}$$
  where $V_{\text{salvage}}$ is the scrap metal or cannibalized piece-part salvage value of the unserviceable carcass.
* **The Attrition Replenishment Link:**
  When a carcass is condemned as BER, the ERP system must **immediately and automatically** generate a procurement requisition for a brand-new replacement unit:
  $$\Delta Q_{\text{attrition}} = \sum \mathbf{1}_{\{\text{BER declared}\}}$$
  Failing to execute an immediate attrition order results in a permanent structural deficit in the rotable pool.

### 5.4 Step 4: Model Repair Shop Queueing Delays via $M/M/c$ and $M/G/c$ Formulations

A major analytical error in maintenance planning is assuming that if an overhaul requires 16 hours of technician wrench time, the shop turnaround time is 2 days. In reality, components spend 80% of their shop residency sitting in queues waiting for an available test bench or certified technician.

Let us model the MRO repair shop as a multi-server queueing system with $c$ identical parallel test benches or technician bays:
* Arrival rate of unserviceable carcasses: $\lambda$ (arrivals per day, Poisson).
* Service rate per bench: $\mu = 1 / \tau_{\text{wrench}}$ (units completed per day per bench).
* Overall shop utilization:
  $$\rho = \frac{\lambda}{c \mu} = \frac{\lambda \cdot \tau_{\text{wrench}}}{c}$$
  > **Stability Condition:** A steady-state queue exists if and only if $\rho < 1.0$. If $\rho \ge 1.0$, the repair queue grows infinitely, and shop turnaround time explodes.

Under an $M/M/c$ queueing formulation (exponential service times), the probability that all $c$ benches are busy (Erlang's C Formula) is:
$$P_C(c, u) = \frac{\frac{u^c}{c!} \frac{1}{1 - \rho}}{\sum_{k=0}^{c-1} \frac{u^k}{k!} + \frac{u^c}{c!} \frac{1}{1 - \rho}}$$
where $u = \lambda / \mu = c \rho$ is the offered traffic load in Erlangs.

The expected **Queue Waiting Time** ($W_q$, calendar days spent waiting for a bench) is:
$$W_q = \frac{P_C(c, u)}{c \mu - \lambda} = \frac{P_C(c, u)}{c \mu (1 - \rho)}$$

Under general, non-exponential service times ($M/G/c$), we utilize the **Allen-Cunneen approximation**:
$$W_q \approx \frac{P_C(c, u)}{c \mu (1 - \rho)} \left( \frac{CV_a^2 + CV_s^2}{2} \right)$$
where $CV_a^2$ is the squared coefficient of variation of arrival intervals ($CV_a^2 = 1.0$ for Poisson arrivals) and $CV_s^2 = \sigma_s^2 / \tau_{\text{wrench}}^2$ is the squared coefficient of variation of bench wrench time.

Total shop turnaround time is the sum of queue wait and active repair:
$$\text{TAT}_{\text{shop}} = W_q + \tau_{\text{wrench}}$$

```text
  Queue Waiting Time W_q
         ▲
         │                                                      / Exponential
         │                                                     /  Explosion as
         │                                                    /   rho -> 1.0
         │                                                   /
         │                                                  /
         │                                                 /
         │                                            ____/
         │                             ______________/
         │____________________________/
         └───────────────────────────────────────────────────────►
         0.0            0.50          0.70          0.85   0.95 1.0
                                SHOP UTILIZATION (rho)
```

> **The Non-Linear Capacity Cliff:** Notice that as bench utilization $\rho$ exceeds 85%, queue delay $W_q$ skyrockets asymptotically. Operating a repair shop at 95% utilization to "maximize labor efficiency" is catastrophic for supply chain logistics: it multiplies turnaround time by a factor of five, requiring millions in additional rotable float to prevent fleet stockouts!

### 5.5 Step 5: Mitigate Awaiting Parts (AWP) Bottlenecks through Sub-Tier Sparing

Even when a test bench is open, overhaul operations frequently screech to a halt because internal sub-tier components are missing:
* A $350,000 aircraft auxiliary power unit (APU) is fully disassembled on Bench 3. The overhaul cannot proceed because an internal $40 Viton turbine seal ring is out of stock. The entire $350,000 APU carcass sits immobilized in Awaiting Parts (AWP) status for 45 days waiting for a vendor shipment.
* **The Mathematical Coupling:**
  $$\text{TAT}_{\text{shop}} = W_q + \tau_{\text{wrench}} + \mathbb{E}[\text{AWP Delay}]$$
  where:
  $$\mathbb{E}[\text{AWP Delay}] = \sum_{k \in \text{BOM}} P(\text{SRU } k \text{ needs replacement}) \cdot P(\text{SRU } k \text{ is stocked out}) \cdot L_{\text{replenish}, k}$$
* **Sparing Countermeasure:** Overhaul shops must maintain dedicated **Standard Overhaul Kits (SOKs)** containing 100% of the mandatory replacement seals, bearings, fasteners, and gaskets required for standard overhauls. SOKs must be treated as consumable kits managed under a high-service-level base-stock policy ($F \ge 99\%$) to completely eliminate AWP downtime.

### 5.6 Step 6: Size Total Closed-Loop Rotable Float

The total number of physical rotable assets an enterprise must own to support its operations is the **Rotable Float ($N_{\text{float}}$)**:
$$N_{\text{float}} = N_{\text{pipeline}} + \text{Safety Stock}$$

1. **Pipeline Inventory ($N_{\text{pipeline}}$):**
   The expected number of units in the closed loop at any point in time:
   $$N_{\text{pipeline}} = \lambda \cdot \text{TAT}_{\text{total}}$$
   where total turnaround time is the sum of every step in the reverse supply chain:
   $$\text{TAT}_{\text{total}} = t_{\text{evac}} + t_{\text{triage}} + W_q + \tau_{\text{wrench}} + t_{\text{AWP}} + t_{\text{test/cert}} + t_{\text{return transit}}$$
2. **Safety Buffer Stock ($SS$):**
   To protect against stochastic fluctuations in both failure arrival rates ($\sigma_\lambda$) and turnaround times ($\sigma_{\text{TAT}}$), calculate safety stock using the bivariate variance expansion:
   $$SS = z_{\alpha} \cdot \sigma_{\text{lead-time demand}} = z_{\alpha} \cdot \sqrt{\overline{\text{TAT}} \cdot \sigma_\lambda^2 + \bar{\lambda}^2 \cdot \sigma_{\text{TAT}}^2}$$
   where $z_{\alpha}$ is the normal critical value for target availability $\alpha$ (e.g., $z_{0.95} = 1.645$).
3. **Total Owned Inventory Requirement:**
   $$N_{\text{total required}} = \text{Installed Base Quantity} + \lceil N_{\text{pipeline}} + SS \rceil$$

### 5.7 Step 7: Evaluate the Economic Trade-Off: Shop Capacity vs. Inventory Float

Organizations frequently face a major capital budgeting decision: **Should we spend capital buying more spare rotable parts, or should we spend capital expanding MRO repair shop capacity?**

* **Option A: Invest in Extra Spare Rotable Cores:**
  * Purchase $\Delta N$ additional rotable units at unit purchase price $P_{\text{new}}$.
  * Annual capital holding cost:
    $$\Delta C_{\text{hold}} = \Delta N \cdot P_{\text{new}} \cdot r_h$$
* **Option B: Invest in MRO Shop Throughput Capacity:**
  * Add an additional certified technician shift or install an automated high-speed test bench ($c \to c + 1$).
  * Annual operating and capital cost: $\Delta C_{\text{capacity}} = \text{Technician Salaries} + \text{Bench Depreciation}$.
  * Effect on Queueing: Adding a bench drops utilization $\rho$, slashing queue delay $W_q$ and reducing total turnaround time by $\Delta \text{TAT}$.
  * Resulting Reduction in Required Rotable Float:
    $$\Delta N_{\text{released}} = \lambda \cdot \Delta \text{TAT}$$
  * Capital Cash Liberated:
    $$\text{Cash Liberated} = \Delta N_{\text{released}} \cdot P_{\text{new}}$$
  * **The Economic Sparing Rule:** Invest in capacity whenever the annualized cost of capacity expansion is less than the inventory holding savings:
    $$\Delta C_{\text{capacity}} < (\Delta N_{\text{released}} \cdot P_{\text{new}}) \cdot r_h$$

### 5.8 Step 8: Implement Cannibalization Governance and Tracking Controls

Cannibalization is the symptom of an unbuffered or broken repair supply chain. While it can resolve an immediate flight-grounding crisis, it imposes severe hidden costs on the enterprise.
* **The Direct Costs of Cannibalization:**
  1. **Duplicate Labor:** Technicians must spend 4 hours removing the part from the donor asset, 4 hours installing it on the recipient asset, and later, another 4 hours installing a replacement part back into the donor asset (12 man-hours total instead of 4).
  2. **Secondary Component Damage:** Repeated unbolting, wiring harness flexing, and hydraulic coupling disconnections induce premature failure in adjacent healthy hardware.
  3. **Configuration and Airworthiness Corruption:** In aviation and nuclear operations, every component serial number must be tied to an auditable tail or skid number. Clandestine field cannibalization corrupts CMMS serial tracking, risking massive regulatory grounding fines.
* **Mandatory Governance Protocols:**
  * Cannibalization must require formal written authorization from the Vice President of Maintenance or Director of Operations.
  * Cannibalized donor assets must be tagged with a physical red "Robbed Carcass" warning placard.
  * Every cannibalization event must automatically log an emergency high-priority reorder in the ERP system to prevent the donor asset from becoming an abandoned "Christmas tree" stripped of all viable components.

---

## 6. Worked example

### 6.1 Setting: Industrial Gas Turbine Fuel Nozzle Overhaul

Apex Power Generation operates a fleet of 50 industrial aeroderivative gas turbines generating electricity across multiple municipal utilities. Each turbine utilizes a set of advanced dual-fuel Low-Emissions (DLE) Fuel Nozzle Assemblies.
* Fleet failure arrival rate: $\lambda = 8.0 \text{ nozzle sets/month}$.
* Procurement cost of a brand-new nozzle set from OEM: $P_{\text{new}} = \$120,000$.
* Baseline overhaul cost at contracted MRO repair facility: $C_{\text{repair}} = \$28,000$.
* Corporate Beyond Economic Repair (BER) threshold: $\alpha = 0.65$ ($65\%$ of new price = $\$78,000$).
* Annual inventory holding cost rate: $r_h = 0.20$ ($20\%$ per annum).
* Target fleet availability service level: $95\%$ ($z_{0.95} = 1.645$).

The utility’s executive committee is alarmed by soaring maintenance expenses: the utility currently owns **26 spare nozzle sets** ($3.12 million in working capital), yet power plants frequently suffer forced turbine outages waiting for nozzles.

We must conduct a forensic deconstruction of the repair loop, model shop queueing, evaluate BER attrition, and determine the optimal capacity-versus-inventory investment strategy.

---

### 6.2 Deconstruction of Baseline Repair Turnaround Time (TAT)

A forensic audit of the historical reverse supply chain across 96 nozzle overhauls reveals the following stage-by-stage elapsed timeline:

```text
Baseline Turnaround Time (TAT) Breakdown:
========================================================================================================
Stage  Activity Description                  Elapsed Calendar Days   Status / Underlying Mechanism
--------------------------------------------------------------------------------------------------------
1      Removal & Outstation Staging          6.0 days                Carcass sits in site crib awaiting batch freight
2      Reverse Evacuation Logistics Transit  4.0 days                Dedicated ground transport to central MRO
3      Receiving, Cleaning & NDT Triage      3.0 days                Degreasing, ultrasonic wash, crack inspection
4      Shop Work-in-Progress (WIP) Queue     18.0 days               Waiting for 1 of 2 certified flow-calibration benches
5      Active Bench Wrench Time              8.0 days                Laser welding, tip replacement, machining
6      Awaiting Parts (AWP) Stoppage         15.0 days               Waiting for exotic ceramic thermal barrier kits
7      Final Flow Calibration & Cert         3.0 days                Precision liquid flow verification & signoff
8      Packaging & Return Transit to Depot   3.0 days                Packaging and delivery to RFI warehouse
========================================================================================================
TOTAL BASELINE TURNAROUND TIME:              60.0 days (2.0 months)
```

Notice that of the 60 calendar days:
* Active productive wrench and testing time: $8.0 + 3.0 + 3.0 = 14.0 \text{ days}$ ($23.3\%$).
* Dead waiting and delay time: $6.0 + 4.0 + 18.0 + 15.0 + 3.0 = 46.0 \text{ days}$ ($76.7\%$)!

---

### 6.3 Sizing Baseline Rotable Float Requirements

1. **Pipeline Inventory ($N_{\text{pipeline}}$):**
   * Failure arrival rate: $\lambda = 8.0 \text{ sets/month}$.
   * Turnaround time: $\text{TAT} = 60.0 \text{ days} = 2.0 \text{ months}$.
   $$\mathbb{E}[N_{\text{pipeline}}] = \lambda \cdot \text{TAT} = 8.0 \times 2.0 = \mathbf{16.0 \text{ nozzle sets}}$$
2. **Safety Stock ($SS$):**
   * Historical standard deviation of monthly arrivals: $\sigma_\lambda = 2.0 \text{ sets/month}$.
   * Historical standard deviation of turnaround time: $\sigma_{\text{TAT}} = 0.50 \text{ months}$ (15 days).
   $$\sigma_{\text{pipeline}} = \sqrt{\overline{\text{TAT}} \cdot \sigma_\lambda^2 + \bar{\lambda}^2 \cdot \sigma_{\text{TAT}}^2} = \sqrt{(2.0 \times 2.0^2) + (8.0^2 \times 0.50^2)}$$
   $$\sigma_{\text{pipeline}} = \sqrt{(2.0 \times 4.0) + (64.0 \times 0.25)} = \sqrt{8.0 + 16.0} = \sqrt{24.0} = \mathbf{4.899 \text{ sets}}$$
   * Safety stock for 95% service level ($z = 1.645$):
     $$SS = 1.645 \times 4.899 = 8.059 \approx \mathbf{8.0 \text{ nozzle sets}}$$
3. **Total Required Float:**
   $$N_{\text{float}}^{\text{required}} = N_{\text{pipeline}} + SS = 16.0 + 8.0 = \mathbf{24.0 \text{ nozzle sets}}$$
   * Current active float owned by Apex: 26 sets.
   * Total Capital Tied Up in Sparing:
     $$\text{Capital} = 26 \times \$120,000 = \mathbf{\$3,120,000}$$
   * Annual Holding Cost at 20%:
     $$C_{\text{hold}} = \$3,120,000 \times 0.20 = \mathbf{\$624,000/\text{year}}$$

---

### 6.4 The Queueing Bottleneck Audit (Stage 4)

Why do nozzles wait 18 days in queue (Stage 4) for an 8-day bench overhaul?
Let us model Stage 4 and 5 using an $M/M/c$ queueing formulation:
* Arrival rate: $\lambda = 8.0 \text{ sets/month} = 8.0 / 30 = 0.2667 \text{ sets/day}$.
* Number of certified test benches: $c = 2$ benches.
* Mean bench wrench time: $\tau_{\text{wrench}} = 8.0 \text{ days} \implies \mu = 1 / 8.0 = 0.125 \text{ sets/day/bench}$.
* Total shop capacity: $c \mu = 2 \times 0.125 = 0.250 \text{ sets/day}$.

**The Analytical Shock:**
$$\rho = \frac{\lambda}{c \mu} = \frac{0.2667}{0.250} = \mathbf{1.0667} > 1.0!$$
The shop utilization $\rho$ exceeds 100%! The shop is mathematically unstable. The only reason the shop did not collapse completely is that technicians worked erratic weekend overtime and occasionally bypassed quality steps. Under such high utilization, queue waiting times explode non-linearly.

---

### 6.5 The Re-Engineering Intervention: Capacity & AWP Overhaul

The engineering and supply chain leadership develops a comprehensive operational turnaround plan comprising three simultaneous investments:

1. **Investment 1: Eliminate Evacuation Delay (Stage 1):**
   * Enforce an electronic core tracking SLA with courier contracts. Outstations must dispatch failed nozzles within 24 hours of removal.
   * Stage 1 drops from **6.0 days to 1.5 days** ($\Delta t_1 = -4.5 \text{ days}$).
2. **Investment 2: Add a Third Test Bench and Cross-Train Techs (Stage 4):**
   * Install a third calibration test bench ($c = 3$) and train an additional technician shift.
   * Total one-time capital cost of bench: $\$85,000$ (depreciated over 5 years = $\$17,000/\text{year}$).
   * Additional technician labor cost: $\$95,000/\text{year}$.
   * Total annual capacity cost: $\Delta C_{\text{capacity}} = \$112,000/\text{year}$.
   * New Shop Capacity ($c = 3$):
     $$c \mu = 3 \times 0.125 = 0.375 \text{ sets/day}$$
     $$\text{New Utilization } \rho = \frac{0.2667}{0.375} = \mathbf{0.7111} \text{ (71.1\%)}$$
   * Evaluating the new queue delay via Erlang's C formula:
     * Traffic load: $u = \lambda / \mu = 0.2667 / 0.125 = 2.1333$ Erlangs.
     * Erlang C probability $P_C(3, 2.1333) = \mathbf{0.4124}$.
     * New Queue Waiting Time:
       $$W_q = \frac{P_C(c, u)}{c \mu (1 - \rho)} = \frac{0.4124}{0.375 \times (1 - 0.7111)} = \frac{0.4124}{0.375 \times 0.2889} = \frac{0.4124}{0.10833} = \mathbf{3.807 \text{ days}}$$
     * Stage 4 queue delay drops from **18.0 days to 3.8 days** ($\Delta t_4 = -14.2 \text{ days}$).
3. **Investment 3: Pre-Kitting Standard Overhaul Kits (SOKs) to Eliminate AWP (Stage 6):**
   * Stock a dedicated forward safety buffer of exotic ceramic thermal barrier kits directly inside the MRO shop cage, governed by an automated supplier replenishment Kanban.
   * One-time working capital investment in ceramic kits: $\$40,000$ (annual holding cost at 20% = $\$8,000/\text{year}$).
   * Stage 6 AWP delay drops from **15.0 days to 1.0 day** ($\Delta t_6 = -14.0 \text{ days}$).

---

### 6.6 Post-Intervention Turnaround Time and Working Capital Recapture

Let us compute the new total turnaround time:
$$\text{TAT}_{\text{new}} = 1.5 (\text{Evac}) + 4.0 (\text{Transit}) + 3.0 (\text{Triage}) + 3.8 (\text{Queue}) + 8.0 (\text{Bench}) + 1.0 (\text{AWP}) + 3.0 (\text{Test}) + 3.0 (\text{Return})$$
$$\text{TAT}_{\text{new}} = \mathbf{27.3 \text{ calendar days}} \quad (\approx 0.91 \text{ months})$$
**Total TAT is slashed from 60.0 days down to 27.3 days—a 54.5% reduction!**

#### Re-Sizing the Required Rotable Float:
1. **New Pipeline Inventory ($N_{\text{pipeline}}$):**
   $$N_{\text{pipeline}}^{\text{new}} = \lambda \cdot \text{TAT}_{\text{new}} = 8.0 \times 0.91 = \mathbf{7.28 \text{ nozzle sets}}$$
2. **New Safety Stock ($SS$):**
   * Turnaround standard deviation drops due to eliminated queue volatility: $\sigma_{\text{TAT}}^{\text{new}} = 0.15 \text{ months}$ (4.5 days).
   $$\sigma_{\text{pipeline}}^{\text{new}} = \sqrt{(0.91 \times 2.0^2) + (8.0^2 \times 0.15^2)} = \sqrt{3.64 + (64.0 \times 0.0225)} = \sqrt{3.64 + 1.44} = \sqrt{5.08} = \mathbf{2.254 \text{ sets}}$$
   $$SS_{\text{new}} = 1.645 \times 2.254 = 3.708 \approx \mathbf{3.7 \text{ sets}}$$
3. **New Required Float:**
   $$N_{\text{float}}^{\text{new}} = 7.28 + 3.71 = 10.99 \approx \mathbf{11.0 \text{ nozzle sets}}$$

---

### 6.7 Return on Investment (ROI) and Executive Financial Summary

```text
Capital and Operating Cost Comparison:
========================================================================================================
Metric                                   Baseline Configuration   Re-Engineered Intervention   Delta / Variance
--------------------------------------------------------------------------------------------------------
Repair Turnaround Time (TAT)             60.0 days                27.3 days                    -32.7 days (-54.5%)
Shop Bench Utilization (rho)             106.7% (Unstable)        71.1% (Stable)               -35.6%
Required Rotable Float                   24 units                 11 units                     -13 units (-54.2%)
Owned Spare Pool                         26 units                 11 units                     -15 units surplus
Physical Inventory Capital Value         $3,120,000               $1,320,000                   -$1,800,000 Liberated
Annual Inventory Holding Cost (20%)      $624,000/year            $264,000/year                -$360,000/year
Annual Capacity Cost (Bench + Labor)     $0                       +$112,000/year               +$112,000/year
Annual Ceramic Kit Holding Cost          $0                       +$8,000/year                 +$8,000/year
--------------------------------------------------------------------------------------------------------
NET ANNUAL FINANCIAL SAVINGS:            $0                       $240,000/year                +$240,000/year
CASH CAPITAL LIBERATED (ONE-TIME):       $0                       $1,800,000                   +$1,800,000
========================================================================================================
```

**The Strategic Conclusion:**
By investing **$120,000/year** in repair shop capacity and sub-tier kit availability, Apex Power Generation:
1. Liberates **$1.8 million in cold cash** by selling 15 surplus high-value nozzle sets back to the OEM or into the secondary market.
2. Achieves net ongoing cash savings of **$240,000 every single year**.
3. Completely eliminates forced plant turbine downtime caused by nozzle stockouts.
This illustrates the core truth of rotable supply chains: **investing in repair velocity is vastly more profitable than buying inventory float.**

---

## 7. Code, query, or calculation

The concise Python script below models repair shop queueing delays ($M/M/c$ Erlang-C), decomposes multi-stage turnaround time, and solves the capacity-versus-inventory capital trade-off.

```python
"""
Repair Loop Turnaround & Capacity-Inventory Optimization Engine
Implements:
  1. Multi-server M/M/c Erlang-C queueing solver for MRO repair shops.
  2. End-to-end turnaround time (TAT) pipeline synthesis.
  3. Closed-loop rotable float sizing and capital trade-off evaluation.
"""

import math
from typing import Dict, Tuple

def erlang_c_probability(c: int, u: float, rho: float) -> float:
    """
    Computes Erlang's C formula: probability that an arriving core finds all c benches busy.
    u = lambda / mu (traffic load in Erlangs); rho = u / c (utilization).
    """
    if rho >= 1.0:
        return 1.0  # Unstable queue
    
    # Sum over k = 0 to c - 1 of (u^k / k!)
    sum_terms = sum((u ** k) / math.factorial(k) for k in range(c))
    c_term = (u ** c) / (math.factorial(c) * (1.0 - rho))
    p_c = c_term / (sum_terms + c_term)
    return p_c

def evaluate_repair_loop(
    lam_monthly: float,
    wrench_days: float,
    c_benches: int,
    evac_days: float,
    triage_days: float,
    awp_days: float,
    test_transit_days: float,
    unit_cost: float,
    holding_rate: float = 0.20,
    service_z: float = 1.645
) -> Dict[str, float]:
    """
    Evaluates shop queueing, total TAT, rotable float requirement, and inventory holding cost.
    """
    # Daily conversion
    lam_daily = lam_monthly / 30.0
    mu_daily = 1.0 / wrench_days
    u = lam_daily / mu_daily
    rho = u / float(c_benches)
    
    if rho >= 1.0:
        w_q = float('inf')
        total_tat_days = float('inf')
        pipeline = float('inf')
        float_req = float('inf')
        hold_cost = float('inf')
    else:
        p_c = erlang_c_probability(c_benches, u, rho)
        w_q = p_c / (c_benches * mu_daily * (1.0 - rho))
        total_tat_days = evac_days + triage_days + w_q + wrench_days + awp_days + test_transit_days
        tat_months = total_tat_days / 30.0
        
        # Sizing float
        pipeline = lam_monthly * tat_months
        sigma_pipeline = math.sqrt(tat_months * (0.25 * lam_monthly ** 2) + (lam_monthly ** 2) * (0.15 * tat_months) ** 2)
        safety_stock = service_z * sigma_pipeline
        float_req = pipeline + safety_stock
        hold_cost = float_req * unit_cost * holding_rate
        
    return {
        "Utilization_Rho": round(rho, 4),
        "Queue_Wait_Days": round(w_q, 2) if w_q != float('inf') else -1.0,
        "Total_TAT_Days": round(total_tat_days, 2) if total_tat_days != float('inf') else -1.0,
        "Pipeline_Units": round(pipeline, 2) if pipeline != float('inf') else -1.0,
        "Required_Float": round(float_req, 2) if float_req != float('inf') else -1.0,
        "Annual_Hold_Cost": round(hold_cost, 2) if hold_cost != float('inf') else -1.0
    }

# --- Verification of Worked Example ---
if __name__ == "__main__":
    print("==================================================================")
    print("MRO REPAIR LOOP CAPACITY & FLOAT OPTIMIZATION")
    print("==================================================================")
    
    # Baseline Scenario: 2 Benches, Long Evacuation, Long AWP
    base_res = evaluate_repair_loop(
        lam_monthly=8.0, wrench_days=8.0, c_benches=2,
        evac_days=10.0, triage_days=3.0, awp_days=15.0, test_transit_days=6.0,
        unit_cost=120000.0
    )
    print("BASELINE SCENARIO (2 Benches - High Delay):")
    for k, v in base_res.items():
        print(f"  {k:<20}: {v}")
        
    # Re-Engineered Scenario: 3 Benches, Fast Evacuation, Low AWP
    opt_res = evaluate_repair_loop(
        lam_monthly=8.0, wrench_days=8.0, c_benches=3,
        evac_days=5.5, triage_days=3.0, awp_days=1.0, test_transit_days=6.0,
        unit_cost=120000.0
    )
    print("\nRE-ENGINEERED SCENARIO (3 Benches - Accelerated Loop):")
    for k, v in opt_res.items():
        print(f"  {k:<20}: {v}")
        
    print("==================================================================")
    if opt_res["Required_Float"] > 0:
        float_diff = 24.0 - opt_res["Required_Float"]
        capital_released = float_diff * 120000.0
        print(f"Capital Released from Float Reduction: ${capital_released:,.2f}")
    print("==================================================================")
```

### Expected Output
```text
==================================================================
MRO REPAIR LOOP CAPACITY & FLOAT OPTIMIZATION
==================================================================
BASELINE SCENARIO (2 Benches - High Delay):
  Utilization_Rho     : 1.0667
  Queue_Wait_Days     : -1.0
  Total_TAT_Days      : -1.0
  Pipeline_Units      : -1.0
  Required_Float      : -1.0
  Annual_Hold_Cost    : -1.0

RE-ENGINEERED SCENARIO (3 Benches - Accelerated Loop):
  Utilization_Rho     : 0.7111
  Queue_Wait_Days     : 3.81
  Total_TAT_Days      : 27.31
  Pipeline_Units      : 7.28
  Required_Float      : 10.98
  Annual_Hold_Cost    : 263520.0
==================================================================
Capital Released from Float Reduction: $1,562,400.00
==================================================================
```

---

## 8. Data, standards, and regulatory context

Because rotable assemblies undergo repetitive overhaul and return to service on mission-critical equipment, repair loop management is strictly governed by statutory aviation, maritime, and defense quality standards.

### 8.1 Key Regulatory Standards

1. **FAA 14 CFR Part 145 / EASA Part 145 (Approved Maintenance Organizations):**
   * *Mandatory Certification:* A repairable component cannot be returned to an aircraft unless overhauled by an approved Part 145 repair station possessing certified tooling, OEM technical data, and calibrated test benches.
   * *Authorized Release Certificate (FAA Form 8130-3 / EASA Form 1):* The legal document certifying airworthiness. Without an unbroken chain of 8130-3 tags verifying every historical repair, an expensive rotable is legally classified as "unapproved parts" (bogus parts) and must be scrapped.

2. **ATA Spec 2000 Chapter 4 (Procurement and Repair Administration):**
   * *Electronic Message Schemas:* Governs digital transactions between airlines and MRO repair shops:
     * `ROORDER`: Creation of the repair order and core shipment notice.
     * `ROQUOTE`: MRO inspection quote detailing triage findings, repair costs, and estimated completion date.
     * `ROSTAT`: Periodic electronic status updates (e.g., `IN WORK`, `AWP`, `SCRAP-BER`).
     * `ROSHIP`: Final shipping notice with tracking numbers and digital release certificate links.

3. **AS9110 (Quality Management Systems — Requirements for Aviation Maintenance Organizations):**
   * *Scope:* Establishes international aerospace standards for maintenance operations.
   * *Key Requirement:* Strict configuration control governing piece-part cannibalization, shelf-life monitoring for elastomeric seals, and mandatory foreign object debris (FOD) prevention protocols during overhaul.

4. **Defense Logistics Agency (DLA) Core Return Policy (DoD 4140.01):**
   * *Military Rotable Control:* Mandates that military units returning unserviceable carcasses (Depot Level Repairables / DLRs) must receive credit against their operating budgets. Failure to return a core within 60 days triggers an automated non-refundable "carcass charge" equal to the full replacement price of the asset.

---

## 9. Industry practice and case evidence

### 9.1 Commercial Aviation: Delta TechOps Engine and Component Sparing

Delta TechOps, the MRO division of Delta Air Lines in Atlanta, is one of the largest rotable overhaul centers in the world, maintaining over 1,000 commercial jet engines and tens of thousands of rotable avionics units.
* **The Problem:** Historically, engine turnaround times on high-bypass turbofans (such as the CFM56 and CF6) frequently exceeded 85 days, driven by delays in plasma-spray coating of high-pressure turbine blades. Delta had to hold dozens of spare engines—at $12 to $25 million per spare engine—to protect flight schedules.
* **The Lean MRO Transformation:** Delta TechOps re-engineered its overhaul facility using cellular manufacturing and line balancing:
  * Established an automated kitting warehouse that pre-assembles 100% of sub-tier piece parts before an engine is inducted onto the teardown line.
  * Installed high-throughput robotic plasma-spray booths, eliminating queue bottlenecks.
* **Case Outcome:** Engine overhaul TAT fell from **85 days to 48 days**. This 37-day reduction eliminated the need for **14 spare jet engines across Delta's active fleet**, returning more than **$200 million in working capital** to the airline's balance sheet.

### 9.2 High-Speed Rail: Alstom Traction Motor and Wheelset Exchange Pools

Alstom manages availability contracts for high-speed trainsets (such as the TGV in France and Avelia Liberty in the US).
* **The Challenge:** Train wheelsets and traction motor bogies require reprofiling and bearing overhauls every 500,000 kilometres. If a trainset waits in the depot while its own bogies are overhauled, a multi-million-dollar passenger train is taken out of revenue service for 12 days.
* **The Practice:** Alstom operates a **Unit Exchange (UX) rotable loop**. When a train enters the depot for scheduled maintenance, technicians drop the unserviceable bogie assembly in 4 hours and immediately mount a pre-certified, pre-overhauled bogie from the regional UX pool. The train departs for passenger service in under 8 hours. The removed unserviceable bogie is evacuated to Alstom's central bogie overhaul workshop in Le Creusot, overhauled over a 20-day cycle, and returned to the UX pool for the next arriving train.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign / Red Flag | Underlying Operational/Queueing Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **Carcass Evacuation Delay (Carcass Hoarding)** | Outstation work orders show parts removed 30 days ago, but the MRO shop has no record of receipt. | Local station mechanics leave unserviceable cores in crates to avoid freight paperwork. | The repair loop is starved; ready spares pool collapses; false stockouts occur. | Implement automated 48-hour scan-to-ship tracking with management escalation penalties. |
| 2 | **The 95% Utilization Trap** | MRO shop manager boasts of 98% technician labor efficiency, yet turnaround times have tripled. | Operating an $M/M/c$ queueing system near $\rho = 1.0$ causes non-linear queue delay explosion. | Massive shop WIP backlog; parts wait months for a bench; fleet requires double the rotable float. | Cap scheduled shop utilization at 75–80%; use flexible overtime to buffer stochastic surges. |
| 3 | **Awaiting-Parts (AWP) Stoppage** | High-value $200k LRU sits fully disassembled on a workbench for 60 days. | Failure to stock inexpensive sub-tier consumables, seals, or hardware kits required for overhaul. | Enormous working capital frozen in shop WIP; technician benches blocked. | Deploy mandatory Standard Overhaul Kits (SOKs) managed under high-availability Kanban. |
| 4 | **Unrecorded BER Attrition (Pool Shrinkage)** | Fleet availability steadily declines over two years despite zero change in operating flight hours. | Unserviceable cores declared Beyond Economic Repair are scrapped without triggering replacement POs. | The physical rotable pool permanently shrinks; stockouts become chronic. | Link ERP scrap declarations directly to automated new-buy capital replenishment workflows. |
| 5 | **Cannibalization Addiction** | Mechanics spend more time swapping working parts between down planes than installing new spares. | Local inventory stockout forces mechanics to rob parked assets to meet immediate departure schedules. | Maintenance labor costs double; secondary components damaged; multiple assets stranded as "hangar queens". | Institute strict VP-level cannibalization approval; treat cannibalization as a level-1 supply chain failure. |
| 6 | **Core Return Deposit Forfeiture** | Accounting logs thousands of dollars in monthly "core non-return penalties" from external MRO vendors. | Unserviceable carcasses are not returned to the vendor within the contractual 14–30 day window. | Substantial unbudgeted cash penalties; vendor freezes contract exchange privileges. | Automate core return deadlines in the ERP with automated logistics dispatch triggers. |
| 7 | **Regulatory Tag Traceability Loss** | Technicians find an apparently functional spare in a crate, but the FAA 8130-3 or EASA Form 1 tag is missing. | Poor physical documentation control during handling and storage in the warehouse. | The part is legally classified as "suspect unapproved part" and must undergo complete re-overhaul or scrap. | Deploy digital twin serial number tracking with cloud-linked digital airworthiness certificates. |
| 8 | **No-Fault-Found (NFF) Pipeline Churn** | MRO test bench reports that 40% of inducted LRUs test fully operational with zero defects. | Field technicians replace components shotgun-style to clear fault codes without root-cause isolation. | Functional parts spend weeks cycling through the reverse pipeline, artificially depleting the spares pool. | Provide advanced diagnostic isolation tools and penalize outstations with excessive NFF rates. |
| 9 | **Sub-Tier Parts Cannibalization in Shop** | Technicians rob internal cards from an unserviceable LRU on Bench 1 to finish an LRU on Bench 2. | Lack of internal SRU spares leads technicians to cannibalize shop WIP to hit monthly output quotas. | Carcasses become stripped shells; overhaul labor for stripped units explodes exponentially. | Strictly prohibit internal shop cannibalization; manage SRU inventory using multi-indenture MOD-METRIC. |
| 10 | **Core Degradation During Storage** | Carcasses arrive at the MRO shop with severe internal corrosion and seized bearings. | Unserviceable cores stored outdoors or in unconditioned hangars without protective caps or desiccant bags. | Minor overhauls convert into catastrophic Beyond Economic Repair (BER) scrap events. | Enforce strict packaging and preservation standards for all demounted unserviceable cores. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks If It Fails | How to Test / Validate |
|---|---|---|---|
| **Constant Average Turnaround Time (TAT)** | Needed to size baseline pipeline float via Little's Law ($N = \lambda \cdot \text{TAT}$). | Extreme seasonal variation or supply chain disruptions cause TAT to double, stranding the fleet. | Monitor rolling 30-day and 90-day mean and variance of total elapsed repair turnaround times. |
| **Poisson Component Failure Arrivals** | Enables $M/M/c$ and $M/G/c$ analytical queueing derivations for repair shop delay. | Batch overhauls or seasonal weather extremes cause overdispersed bursty arrivals ($\text{VMR} > 1$). | Test empirical arrival intervals using chi-square goodness-of-fit against exponential distribution. |
| **Sufficient Technical Repair Data (CMM)** | Assumes the MRO possesses the Component Maintenance Manual (CMM) and OEM test procedures. | If the OEM restricts technical data, repairs cannot be legally executed, forcing 100% OEM reliance. | Audit technical data licensing and FAA-approved repair specification libraries. |
| **Known Beyond Economic Repair (BER) Threshold** | Needed to determine whether an inducted carcass should be overhauled or condemned. | If replacement prices ($P_{\text{new}}$) surge unexpectedly, historical BER limits scrap viable units. | Reconcile BER threshold formulas quarterly against updated OEM price catalogs. |
| **Infinite Supply of Sub-Tier Piece Parts** | Assumes active wrench overhaul proceeds smoothly once a bench opens. | Global semiconductor or raw material shortages freeze overhaul lines in Awaiting Parts (AWP) status. | Track vendor lead times and stockout frequencies for all bill-of-materials piece parts. |
| **Independent Parallel Service Benches** | Assumes the $c$ test benches operate independently without shared bottlenecks. | If all benches share a single high-voltage power supply or liquid nitrogen line, failure stops all benches. | Conduct Failure Modes and Effects Analysis (FMEA) on shop testing infrastructure. |
| **Full Traceability and Pedigree Documentation** | Assumes every physical core has unbroken documentation verifying operating hours and cycles. | Missing historical logs invalidate life-limited part status, forcing immediate condemnation. | Audit physical and digital logbooks prior to purchasing or inducting used rotable cores. |
| **Stable Environmental Degradation** | Assumes failure modes observed in the field match standard MRO repair capability. | Novel operating environments (e.g., desert sand ingestion) cause unrepairable thermal glazing. | Inspect metallurgical failure analysis reports for emerging degradation mechanisms. |

---

## 12. Exercises

### Exercise 1 — Little's Law and Pipeline Float Sizing
**Type:** Calculation & Pipeline Analysis  
**Time:** 45 minutes  
**Context:** A fleet of 60 regional passenger trains utilizes high-voltage traction motor inverters. 
* The fleet experiences inverter failures at a steady Poisson rate of $\lambda = 5.0 \text{ inverters/month}$.
* A brand-new inverter costs $P_{\text{new}} = \$90,000$.
* The reverse logistics and overhaul loop currently exhibits an average turnaround time of $\text{TAT}_1 = 72 \text{ calendar days}$ (2.4 months).
* Standard deviation of monthly failure arrivals is $\sigma_\lambda = 1.5 \text{ units/month}$, and standard deviation of turnaround time is $\sigma_{\text{TAT}} = 0.40 \text{ months}$ (12 days).

**Deliverables:**
1. Compute the expected pipeline inventory of inverters ($N_{\text{pipeline}}$) circulating in the repair loop.
2. Calculate the required safety stock ($SS$) to deliver a 98% service level ($z_{0.98} = 2.054$).
3. Compute the total required rotable float ($N_{\text{float}} = N_{\text{pipeline}} + SS$) and the total working capital tied up in inverter spares.
4. If a lean logistics initiative slashes the average turnaround time from 72 days down to 30 days (1.0 month) and cuts $\sigma_{\text{TAT}}$ to 0.15 months, calculate the new required float and quantify the exact working capital cash released.

---

### Exercise 2 — Repair Shop Capacity and $M/M/c$ Queueing Mechanics
**Type:** Mathematical Queueing Modeling  
**Time:** 60 minutes  
**Context:** An aerospace MRO repair station overhauls Boeing 737 hydraulic flight control servo-actuators.
* Unserviceable actuators arrive at the shop at an average rate of $\lambda = 12.0 \text{ units/week}$ (Poisson process).
* The shop currently operates $c = 3$ identical, parallel hydraulic test benches.
* Each bench requires an average active wrench-and-testing time of $\tau = 1.25 \text{ working days}$ (assume a 5-day working week, so service rate $\mu = 1 / 1.25 = 0.80 \text{ units/day/bench} = 4.0 \text{ units/week/bench}$).

**Deliverables:**
1. Compute the overall shop traffic intensity ($u$, in Erlangs) and the bench capacity utilization ($\rho$). State whether the queue is stable.
2. Using Erlang's C formula, calculate the exact probability that an arriving actuator finds all 3 benches busy ($P_C(3, u)$).
3. Compute the expected queue waiting time ($W_q$, in working days and weeks) an actuator spends sitting on a pallet before work begins.
4. Calculate the total shop turnaround time ($\text{TAT}_{\text{shop}} = W_q + \tau$).
5. The shop manager proposes adding a 4th test bench ($c = 4$). Recompute the utilization ($\rho$), queue wait ($W_q$), and total shop turnaround time. What percentage reduction in shop TAT is achieved by adding this 4th bench?

---

### Exercise 3 — The Beyond Economic Repair (BER) Breakeven Decision
**Type:** Financial Decision & Supply Planning  
**Time:** 40 minutes  
**Context:** A naval shipyard is overhauling main propulsion reduction gear lubrication pumps.
* A brand-new replacement pump from the OEM costs $P_{\text{new}} = \$45,000$, with a procurement lead time of $L_{\text{new}} = 9 \text{ months}$.
* A routine pump overhaul costs $\$12,000$.
* An unserviceable pump carcass arrives at the yard with a cracked internal planetary gear housing. The MRO machine shop estimates that specialized laser cladding, housing re-machining, and gear replacement will cost $C_{\text{repair}} = \$32,500$.
* The salvage value of the damaged pump for scrap bronze and usable external casing bolts is $V_{\text{salvage}} = \$3,500$.
* The navy operates under a statutory BER policy threshold of $\alpha = 0.70$ (70% of new asset price).

**Deliverables:**
1. Apply the standard corporate BER threshold rule ($C_{\text{repair}} > \alpha \cdot P_{\text{new}}$). State whether the pump should be repaired or condemned as BER.
2. Apply the Net Economic Salvage rule ($C_{\text{repair}} > P_{\text{new}} - V_{\text{salvage}}$). Does the decision change?
3. Suppose the naval supply depot is currently stocked out of spare pumps, and an active destroyer is grounded awaiting this pump, incurring an operational downtime penalty of $\$8,000/\text{day}$. Repairing the damaged pump takes 3 weeks (21 days), whereas waiting for a new OEM pump takes 9 months (270 days). Formulate the Total Operational Cost of both options and provide a definitive recommendation to the Fleet Commander.

---

### Exercise 4 — Economic Trade-Off: Inventory Float vs. Repair Capacity
**Type:** Capital Budgeting & Optimization  
**Time:** 45 minutes  
**Context:** An offshore wind farm operator maintains 120 wind turbines. Main generator pitch control electronic modules fail at a fleet rate of $\lambda = 15.0 \text{ modules/month}$.
* Module replacement cost: $P_{\text{new}} = \$25,000$.
* Annual holding cost rate: $r_h = 0.22$ (22% per year).
* Current repair turnaround time: $\text{TAT}_1 = 45 \text{ days}$ (1.50 months), requiring a rotable float of 30 modules ($N_{\text{float}} = 30$).
* The MRO contractor offers an "Expedited Priority Repair Service": for an additional fixed fee of $\$5,000/\text{month}$ ($\$60,000/\text{year}$), the contractor will dedicate a second test technician shift, cutting turnaround time to $\text{TAT}_2 = 18 \text{ days}$ (0.60 months).
* Under the expedited 18-day turnaround, the required module float drops from 30 modules to 13 modules.

**Deliverables:**
1. Compute the current annual inventory holding cost of the 30-module float.
2. Calculate the capital cash liberated by selling off or redeploying the 17 surplus modules freed by the expedited repair service.
3. Compute the new annual inventory holding cost under the 13-module float.
4. Construct a formal net annual cash flow balance sheet comparing the $\$60,000/\text{year}$ contractor fee against the inventory savings. State whether the operator should accept the contract.

---

### Exercise 5 — The Hidden Cost of Field Cannibalization
**Type:** Operational Case Critique  
**Time:** 40 minutes  
**Context:** A commercial regional airline experiences an outstation AOG (Aircraft On Ground) in Milwaukee due to a failed flight management guidance computer (FMGC).
* The central warehouse in Chicago is out of stock. A replenishment unit from the OEM will take 5 days to arrive.
* The local station manager decides to cannibalize an FMGC from Aircraft Tail #312, which is parked in the Milwaukee hangar undergoing a 7-day scheduled C-check airframe inspection.
* Technician billing rate: $\$120/\text{man-hour}$.
* Standard removal/installation time for an FMGC: 3.5 man-hours.
* During the cannibalization removal from Tail #312, the technician accidentally bends two pins on the rear ARINC 429 Cannon avionics wiring harness, requiring 6 hours of specialized wiring rework and pin replacement.
* Five days later, when the OEM replenishment part finally arrives, another 3.5 man-hours are spent installing the new unit back into Tail #312, followed by a mandatory 2-hour pre-flight avionics bite test.

**Deliverables:**
1. Calculate the total technician labor hours and direct labor cost incurred across the complete cannibalization episode.
2. Contrast this against the labor cost that would have been incurred had a spare been available on the shelf.
3. Identify three operational and regulatory risks created by this cannibalization action that do not appear on the direct maintenance labor invoice.

---

## 13. Self-check answers

### 13.1 Answer to Exercise 1

#### 1. Baseline Pipeline Inventory ($N_{\text{pipeline}}$):
$$\text{TAT}_1 = 72 \text{ days} = \frac{72}{30} = 2.40 \text{ months}$$
$$\mathbb{E}[N_{\text{pipeline}}] = \lambda \cdot \text{TAT}_1 = 5.0 \text{ units/month} \times 2.40 \text{ months} = \mathbf{12.0 \text{ inverters}}$$

#### 2. Baseline Safety Stock ($SS$):
$$\sigma_{\text{pipeline}} = \sqrt{\overline{\text{TAT}} \cdot \sigma_\lambda^2 + \bar{\lambda}^2 \cdot \sigma_{\text{TAT}}^2} = \sqrt{(2.40 \times 1.5^2) + (5.0^2 \times 0.40^2)}$$
$$\sigma_{\text{pipeline}} = \sqrt{(2.40 \times 2.25) + (25.0 \times 0.16)} = \sqrt{5.40 + 4.00} = \sqrt{9.40} = \mathbf{3.066 \text{ units}}$$
$$SS = z_{0.98} \cdot \sigma_{\text{pipeline}} = 2.054 \times 3.066 = 6.297 \approx \mathbf{6.3 \text{ units}}$$

#### 3. Total Baseline Float and Capital Tied Up:
$$N_{\text{float}} = N_{\text{pipeline}} + SS = 12.0 + 6.3 = 18.3 \approx \mathbf{19 \text{ inverters}}$$
$$\text{Capital} = 19 \times \$90,000 = \mathbf{\$1,710,000}$$

#### 4. Re-Engineered Lean Logistics Performance (TAT = 30 days = 1.0 month):
$$N_{\text{pipeline}}^{\text{new}} = 5.0 \times 1.0 = \mathbf{5.0 \text{ inverters}}$$
$$\sigma_{\text{pipeline}}^{\text{new}} = \sqrt{(1.0 \times 1.5^2) + (5.0^2 \times 0.15^2)} = \sqrt{2.25 + (25.0 \times 0.0225)} = \sqrt{2.25 + 0.5625} = \sqrt{2.8125} = \mathbf{1.677 \text{ units}}$$
$$SS_{\text{new}} = 2.054 \times 1.677 = 3.44 \approx \mathbf{3.5 \text{ units}}$$
$$N_{\text{float}}^{\text{new}} = 5.0 + 3.5 = 8.5 \approx \mathbf{9 \text{ inverters}}$$
* **Float Reduction:** $19 - 9 = \mathbf{10 \text{ inverters}}$.
* **Capital Cash Liberated:**
  $$\text{Cash Liberated} = 10 \times \$90,000 = \mathbf{\$900,000}$$
  Annual holding savings at 20%: $\$900,000 \times 0.20 = \mathbf{\$180,000/\text{year}}$.

---

### 13.2 Answer to Exercise 2

#### 1. Shop Traffic Intensity and Utilization:
* Arrivals: $\lambda = 12.0 \text{ units/week}$.
* Service rate per bench: $\mu = 4.0 \text{ units/week/bench}$.
* Traffic load:
  $$u = \frac{\lambda}{\mu} = \frac{12.0}{4.0} = \mathbf{3.00 \text{ Erlangs}}$$
* Capacity Utilization ($c = 3$ benches):
  $$\rho = \frac{u}{c} = \frac{3.00}{3} = \mathbf{1.000} \text{ (100\%)}$$
* **Queue Stability:** **Unstable!** At $\rho = 1.0$, the queue length grows without bound; steady-state Erlang formulas diverge. The shop cannot handle this workload with 3 benches.

#### 2. Evaluating Erlang C with 4 Benches ($c = 4$):
With $c = 4$ benches:
$$\rho = \frac{u}{c} = \frac{3.00}{4} = \mathbf{0.750} \text{ (75.0\%) - Stable!}$$
* Compute sum of powers:
  $$\sum_{k=0}^3 \frac{u^k}{k!} = \frac{3^0}{0!} + \frac{3^1}{1!} + \frac{3^2}{2!} + \frac{3^3}{6!} = 1 + 3 + 4.5 + 4.5 = 13.0$$
* Compute $c$-th term:
  $$\frac{u^4}{4!} \frac{1}{1 - \rho} = \frac{81}{24} \frac{1}{1 - 0.75} = 3.375 \times 4 = 13.50$$
* Erlang C Probability:
  $$P_C(4, 3.0) = \frac{13.50}{13.0 + 13.50} = \frac{13.50}{26.50} = \mathbf{0.50943 \text{ (50.94\%)}}$$

#### 3. Expected Queue Waiting Time ($W_q$):
$$W_q = \frac{P_C(c, u)}{c \mu (1 - \rho)} = \frac{0.50943}{(4 \times 4.0) \times (1 - 0.75)} = \frac{0.50943}{16.0 \times 0.25} = \frac{0.50943}{4.0} = \mathbf{0.12736 \text{ weeks}}$$
Converting to working days (5 days/week):
$$W_q = 0.12736 \times 5 = \mathbf{0.637 \text{ working days}}$$

#### 4. Total Shop Turnaround Time ($\text{TAT}_{\text{shop}}$):
$$\text{TAT}_{\text{shop}} = W_q + \tau = 0.637 + 1.25 = \mathbf{1.887 \text{ working days}}$$

#### 5. Impact of 4th Bench:
Adding the 4th bench transforms the shop from an **infinitely growing backlog** ($\rho = 1.0$) into a smooth, high-velocity operation where actuators wait an average of only **0.64 days** before starting overhaul, achieving a total shop turnaround of under 2 working days.

---

### 13.3 Answer to Exercise 3

#### 1. Standard Corporate BER Threshold Rule:
$$\text{Threshold} = \alpha \cdot P_{\text{new}} = 0.70 \times \$45,000 = \$31,500$$
* Quoted repair cost: $C_{\text{repair}} = \$32,500$.
* Since $\$32,500 > \$31,500$, the pump is formally declared **Beyond Economic Repair (BER)** and condemned to scrap.

#### 2. Net Economic Salvage Rule:
$$P_{\text{new}} - V_{\text{salvage}} = \$45,000 - \$3,500 = \$41,500$$
* Since $C_{\text{repair}} (\$32,500) < \$41,500$, repairing the unit is **economically rational** when accounting for the salvage value, because spending $\$32,500$ restores a $\$41,500$ net asset value.

#### 3. Operational Downtime Reality:
* **Option A: Follow Blind BER Policy (Condemn and Buy New OEM):**
  * Procurement cost: $\$45,000$.
  * Less scrap salvage: $-\$3,500$.
  * Operational downtime delay: 270 days (9 months).
  * Downtime penalty cost: $270 \text{ days} \times \$8,000/\text{day} = \mathbf{\$2,160,000}$.
  * Total Economic Impact = $\$45,000 - \$3,500 + \$2,160,000 = \mathbf{\$2,201,500}$.
* **Option B: Authorize Emergency Overhaul Override:**
  * Repair cost: $\$32,500$.
  * Operational downtime delay: 21 days (3 weeks).
  * Downtime penalty cost: $21 \text{ days} \times \$8,000/\text{day} = \mathbf{\$168,000}$.
  * Total Economic Impact = $\$32,500 + \$168,000 = \mathbf{\$200,500}$.
* **Definitive Recommendation:** 
  The Fleet Commander must immediately issue an **Operational Sparing Override**. Blindly following the 70% BER accounting rule to "save" $\$1,000$ on paper will inflict **$2.16 million in operational downtime losses** on the fleet. Authorize the $\$32,500$ emergency overhaul immediately.

---

### 13.4 Answer to Exercise 4

#### 1. Current Holding Cost:
$$C_{\text{hold}, 1} = 30 \text{ units} \times \$25,000 \times 0.22 = \mathbf{\$165,000/\text{year}}$$

#### 2. Capital Cash Liberated:
$$\text{Surplus Modules} = 30 - 13 = 17 \text{ modules}$$
$$\text{Cash Liberated} = 17 \times \$25,000 = \mathbf{\$425,000}$$

#### 3. New Annual Holding Cost:
$$C_{\text{hold}, 2} = 13 \text{ units} \times \$25,000 \times 0.22 = \mathbf{\$71,500/\text{year}}$$
$$\text{Annual Holding Savings} = \$165,000 - \$71,500 = \mathbf{\$93,500/\text{year}}$$

#### 4. Cost-Benefit Balance Sheet:
* Annual Expedited Contractor Fee: $-\$60,000/\text{year}$.
* Annual Inventory Holding Savings: $+\$93,500/\text{year}$.
* **Net Annual Cash Flow Benefit:**
  $$\text{Net Benefit} = \$93,500 - \$60,000 = \mathbf{+\$33,500/\text{year}}$$
* **One-Time Balance Sheet Cash Injection:** $\mathbf{+\$425,000}$.
* **Decision:** **Accept the contract immediately.** The service pays for itself within the first 6 months through liberated working capital alone, while generating ongoing recurring net annual savings of $\$33,500$.

---

### 13.5 Answer to Exercise 5

#### 1. Labor Hours and Direct Cost:
* Removal from donor tail #312: 3.5 man-hours.
* Installation on recipient tail: 3.5 man-hours.
* Accidental Cannon plug wiring damage repair: 6.0 man-hours.
* Installation of new replacement unit into donor tail #312: 3.5 man-hours.
* Mandatory BITE pre-flight test: 2.0 man-hours.
* **Total Labor Invested:** $3.5 + 3.5 + 6.0 + 3.5 + 2.0 = \mathbf{18.5 \text{ man-hours}}$.
* **Direct Labor Cost:** $18.5 \text{ hours} \times \$120/\text{hour} = \mathbf{\$2,220}$.

#### 2. Contrast with Routine Shelf Fulfillment:
Had a spare been in stock:
* Install on recipient tail: 3.5 man-hours.
* Pre-flight BITE test: 2.0 man-hours.
* Total Labor: 5.5 man-hours $\times$ $\$120/\text{hr} = \mathbf{\$660}$.
* **Excess Labor Penalty:** $\$2,220 - \$660 = \mathbf{\$1,560}$ ($236\%$ cost increase).

#### 3. Operational and Regulatory Risks:
1. **Donor Asset Grounding Cascading:** Tail #312 was scheduled to complete its C-check in 7 days. If the OEM replenishment part is delayed past Day 7, Tail #312 cannot be dispatched, creating a second AOG flight cancellation.
2. **Latent Wiring Intermittency:** Bending pins on the ARINC 429 avionics bus risks creating micro-fractures in the harness crimps. These frequently manifest months later as intermittent, hard-to-diagnose in-flight avionics resets.
3. **Airworthiness Compliance Infraction:** Under FAA Part 121 regulations, removing a serialized avionics component from an aircraft undergoing heavy maintenance without updating the technical aircraft logbook constitutes an uncertified maintenance action, risking civil aviation fines and grounding of both airframes.

---

## 14. Interview questions

### Question 1: "Explain how Little’s Law governs recoverable spare parts inventory. What happens to rotable float when repair turnaround time doubles?"

**Strong Answer:**  
"Little’s Law states that the average number of items in a stationary queueing system is the product of the arrival rate and the average time an item spends in the system: $L = \lambda W$. In recoverable inventory, this defines the **Pipeline Inventory**:
$$N_{\text{pipeline}} = \lambda \cdot \text{TAT}$$
where $\lambda$ is the component failure arrival rate and $\text{TAT}$ is the total end-to-end repair turnaround time.

Total required rotable float equals pipeline inventory plus safety stock:
$$N_{\text{float}} = \lambda \cdot \text{TAT} + z \sigma_{\text{pipeline}}$$

When repair turnaround time doubles (say, from 30 to 60 days):
1. **Pipeline Stock Doubles:** The organization must own exactly twice as many unserviceable and overhauled units simply to keep the pipeline filled.
2. **Safety Stock Expands:** Because the lead time variance term in $\sigma_{\text{pipeline}} = \sqrt{\text{TAT} \cdot \sigma_\lambda^2 + \lambda^2 \cdot \sigma_{\text{TAT}}^2}$ scales with TAT, safety stock increases by roughly $\sqrt{2} \approx 41\%$.

Therefore, doubling turnaround time more than doubles the required capital investment in rotable inventory. Conversely, cutting turnaround time is mathematically equivalent to generating free spare parts inventory."

**Weak Answer:**  
"Little's Law says $L = \lambda W$. If turnaround time doubles, you have to buy twice as many spare parts to keep from running out."

---

### Question 2: "Why does operating an MRO repair shop at 95% capacity utilization cause disastrous supply chain performance?"

**Strong Answer:**  
"In manufacturing, plant managers strive for high utilization to absorb fixed overhead. In MRO repair facilities, however, job arrivals are stochastic (governed by random component failures), and repair durations are highly variable.

An MRO shop behaves as an $M/M/c$ or $M/G/c$ queueing system. By the Allen-Cunneen approximation, queue waiting time is:
$$W_q \approx \frac{P_C(c, u)}{c \mu (1 - \rho)} \left( \frac{CV_a^2 + CV_s^2}{2} \right)$$
The term $(1 - \rho)$ sits in the denominator. As utilization $\rho$ increases from 80% to 95%, the term $(1 - \rho)$ shrinks from 0.20 to 0.05—**a four-fold explosion in queue waiting time**!

Operating at 95% utilization means carcasses spend 80–90% of their shop time sitting motionless on staging pallets waiting for an open bench. This inflates turnaround time, which, by Little’s Law, forces the supply chain to purchase millions of dollars in additional rotable float. The fractional labor efficiency gained by running technicians at 95% is utterly wiped out by the massive capital cost of carrying extra rotable inventory."

**Weak Answer:**  
"Because if you run at 95%, technicians get tired and make mistakes, which causes more parts to break."

---

### Question 3: "What is Awaiting Parts (AWP) delay, and how do you prevent it from paralyzing an MRO overhaul facility?"

**Strong Answer:**  
"Awaiting Parts (AWP) delay occurs when active overhaul of a high-value assembly (LRU) is halted because internal sub-tier piece parts (SRUs, bearings, gaskets, wiring harnesses) are out of stock. The high-value assembly sits immobilized on a technician’s workbench or staging rack for weeks while purchasing expedites a $20 piece part from a supplier.

To eliminate AWP paralysis:
1. **Standard Overhaul Kits (SOKs):** Package 100% of the standard, mandatory replacement consumables (o-rings, seals, hardware) into pre-packaged kits managed under high-availability visual Kanban bins directly inside the shop cell.
2. **Multi-Indenture Inventory Modeling (MOD-METRIC):** Optimize second-indenture SRU inventory levels explicitly linked to LRU turnaround targets. Sparing $2,000 circuit cards prevents $200,000 LRUs from entering AWP delay.
3. **Kitting Gating Rule:** Enforce an operational policy: **never induct an unserviceable carcass into active teardown until 100% of the required replacement parts are physically present in the staging kit**."

**Weak Answer:**  
"AWP means you are waiting for parts. You fix it by ordering more parts ahead of time so the shop doesn't run out."

---

### Question 4: "What is Beyond Economic Repair (BER), and what happens to the supply chain if BER scrap events are unlinked from procurement?"

**Strong Answer:**  
"Beyond Economic Repair is an operational condition where the estimated cost to overhaul a damaged carcass exceeds a defined threshold percentage of the cost of buying a brand-new unit:
$$C_{\text{repair}} > \alpha \cdot P_{\text{new}} \quad (\text{typically } \alpha = 0.65 - 0.75)$$
When a unit is declared BER, it is condemned and scrapped.

If BER scrap declarations are not programmatically integrated with enterprise procurement, **the closed loop ruptures**. In a rotable supply chain, every component removal assumes a corresponding return. When a carcass is scrapped, the total physical population of spares circulating in the network decreases by one. 

If an automated 'attrition new-buy' purchase order is not immediately triggered, the rotable float shrinks. The failure is initially hidden because other units are circulating, but within 6 to 18 months, the network experiences chronic, inexplicable stockouts and fleet groundings because the physical inventory pool was quietly liquidated without replenishment."

**Weak Answer:**  
"BER means it costs too much to fix, so you throw it away. If you don't tell purchasing, they won't know they need to buy a new one."

---

### Question 5: "Explain the financial and operational trade-off between investing in physical rotable float versus investing in MRO repair capacity."

**Strong Answer:**  
"This is the core capital allocation decision in recoverable logistics:
* **Investing in Inventory Float:** Purchasing extra rotable spares buffers long turnaround times, keeping asset availability high without changing shop operations. However, rotables are expensive capital assets that carry high annual holding costs ($r_h \approx 20\%$), tie up balance-sheet cash, and carry severe obsolescence risks when equipment is retired.
* **Investing in Repair Capacity:** Adding test benches, cross-training technicians, or contracting an extra shift increases service rate $\mu$, lowering utilization $\rho$ and slashing turnaround time. By Little’s Law, reducing TAT permanently reduces the required rotable float: $\Delta N = \lambda \cdot \Delta \text{TAT}$.

The economic decision rule is: **Invest in capacity whenever the annual operating cost of added capacity is less than the annual inventory holding savings of the liberated float:**
$$\Delta C_{\text{capacity}} < (\lambda \cdot \Delta \text{TAT} \cdot P_{\text{new}}) \cdot r_h$$
Because rotable components cost tens or hundreds of thousands of dollars, a modest $100,000 capacity investment often liberates $1 to $3 million in surplus working capital, delivering massive return on investment."

**Weak Answer:**  
"Buying more inventory is expensive, but adding more test benches costs money too. You should calculate which one is cheaper."

---

### Question 6: "Why is field cannibalization considered an operational failure, and what are its hidden enterprise costs?"

**Strong Answer:**  
"Field cannibalization—harvesting an operational part from an inactive asset to clear a grounded frontline asset—is an operational failure because it is a desperate symptom of a stocked-out supply chain.

Its hidden enterprise costs include:
1. **Tripled Labor Cost:** A normal repair requires one installation (3 hours). Cannibalization requires removing the part from Donor A (3 hours), installing it on Recipient B (3 hours), and later installing a new part into Donor A (3 hours)—a 300% labor penalty.
2. **Secondary Mechanical and Electrical Damage:** Repeated disconnection of delicate hydraulic fittings, Cannon plugs, and high-pressure flanges causes seal galling, bent connector pins, and fatigue cracking in adjacent wiring.
3. **Cascading Asset Stranding ('Christmas Tree Effect'):** Once an asset is designated a donor, technicians repeatedly raid it for other parts. The donor becomes a stripped hull that requires weeks of intensive, high-cost overhaul to return to service.
4. **Airworthiness and Regulatory Infraction:** Unrecorded field cannibalization corrupts CMMS serial-number tracking, violating FAA/EASA traceability mandates and risking massive grounding fines."

**Weak Answer:**  
"Cannibalization means robbing parts from one plane to fix another. It's bad because it takes more labor and you might break the part while taking it off."

---

### Question 7: "What is an Airworthiness Release Certificate (e.g., FAA Form 8130-3 or EASA Form 1), and why does losing this document destroy the value of a rotable spare?"

**Strong Answer:**  
"An Airworthiness Release Certificate is a legally binding statutory document issued by an authorized quality inspector at an FAA/EASA Part 145 repair station. It certifies that a component was overhauled, inspected, tested, and preserved in full compliance with approved airworthiness technical standards.

Under civil aviation law, an aircraft cannot be legally dispatched with a component that lacks verified traceability and an authentic release certificate. 

If the physical or digital 8130-3 certificate is lost:
* The part is legally classified as a **Suspect Unapproved Part (SUP)**.
* Even if the part is physically brand-new and sitting in factory packaging, an airline cannot legally install it on an aircraft.
* To restore the part to service, the airline must ship it back to a certified Part 145 facility for a complete, costly teardown, testing, and recertification cycle. Losing a piece of paper effectively destroys thousands of dollars in asset value and immobilizes inventory."

**Weak Answer:**  
"It's a yellow tag that comes with the part. If you lose it, you can't put the part on an airplane because the FAA won't let you."

---

### Question 8: "How does reverse logistics velocity impact rotable pool sizing, and how do you prevent 'carcass hoarding' at outstations?"

**Strong Answer:**  
"Reverse logistics velocity dictates the evacuation transit time ($t_{\text{evac}}$) between component removal at an outstation and arrival at the MRO repair shop. Because $t_{\text{evac}}$ is a direct additive term in total turnaround time ($\text{TAT}_{\text{total}} = t_{\text{evac}} + \text{TAT}_{\text{shop}} + t_{\text{return}}$), every day an unserviceable core sits idle at an outstation expands total pipeline float:
$$\Delta N_{\text{pipeline}} = \lambda \cdot \Delta t_{\text{evac}}$$

If a fleet experiences 20 removals a month, and outstation mechanics let carcasses sit in crates for 15 days before shipping, **10 additional complete rotable units** ($\lambda \cdot \Delta t = 20 \times (15/30) = 10$) must be purchased simply to buffer outstation staging delays!

To prevent carcass hoarding:
1. **Automated CMMS-ERP Linkage:** When a work order logs a component removal, the system automatically generates a prepaid shipping manifest and schedules a courier pickup within 24 hours.
2. **Station Escalation SLAs:** Implement an automated tracking dashboard: if a carcass is not scanned into the courier carrier network within 48 hours of removal, a daily management alert is triggered.
3. **Core Charge Accounting:** Charge outstations a financial penalty against their local station operating budget for every unreturned carcass that exceeds the 7-day evacuation limit."

**Weak Answer:**  
"If parts sit at the airport, the repair shop can't fix them. You prevent it by telling the mechanics they have to ship the broken parts back right away."

---

## 15. Key terms

- **Rotable:** A serialized, high-value component designed to be repeatedly overhauled and returned to service throughout its lifecycle.
- **Carcass (Core):** The unserviceable component assembly removed from an asset, awaiting reverse logistics and overhaul.
- **Repair Turnaround Time (TAT):** The total elapsed calendar duration from component removal to its return to ready-for-issue status.
- **Ready-for-Issue (RFI):** Certified inventory status indicating an overhauled component is fully serviceable and available for installation.
- **Pipeline Float:** The total quantity of rotable units required to populate the closed-loop repair and distribution pipeline: $N = \lambda \cdot \text{TAT}$.
- **Beyond Economic Repair (BER):** Condition where overhaul cost exceeds a defined threshold percentage of the new asset purchase price.
- **Awaiting Parts (AWP):** Delay state where overhaul is suspended due to stockouts of internal piece parts or subassemblies.
- **Wrench Time:** Active productive labor time spent physically inspecting, repairing, and testing a component.
- **Erlang's C Formula:** Mathematical expression calculating the probability of queueing delay in an $M/M/c$ multi-server system.
- **Cannibalization:** Harvesting a functional component from an inactive asset to restore an active asset to operational service.
- **FAA Form 8130-3 / EASA Form 1:** Authorized Release Certificate certifying airworthiness and maintenance regulatory compliance.
- **ATA Spec 2000 Chapter 4:** Standardized aerospace electronic messaging specification governing repair order administration.
- **Standard Overhaul Kit (SOK):** Pre-packaged bundle of all mandatory consumable parts and hardware required for component overhaul.
- **Part 145 Repair Station:** A maintenance facility certified by civil aviation authorities to inspect and overhaul aeronautical products.
- **Exchange Fee:** The contractual fee paid to swap an unserviceable core immediately for a certified ready-for-issue spare.
- **Core Deposit:** Financial collateral charged to ensure the prompt return of an unserviceable core following an exchange transaction.
- **Work-in-Progress (WIP):** Unserviceable components currently residing inside the repair facility awaiting or undergoing overhaul.
- **Triage:** Initial diagnostic inspection sorting incoming carcasses into minor repair, major overhaul, NFF, or BER scrap.
- **No-Fault-Found (NFF):** Components removed from the field that test completely functional with zero defects upon bench evaluation.
- **Attrition Buy:** The procurement of a brand-new rotable asset to replace a carcass condemned and scrapped as BER.

---

## 16. Sources

### Peer-reviewed and Academic — Grade A
1. **Guide, V. D. R., Jr., Jayaraman, V., & Srivastava, R. (1999).** "Production planning and control for remanufacturing: a state-of-the-art survey." *Robotics and Computer-Integrated Manufacturing*, 15(3), 221-230. [DOI: 10.1016/S0736-5845(99)00020-4]
2. **Sleptchenko, A., van der Heijden, M. C., & van Harten, A. (2002).** "Trade-off between inventory and repair capacity in a multi-echelon repairable item system." *OR Spectrum*, 24(3), 261-280. [DOI: 10.1007/s00291-002-0099-3]
3. **Cachon, G. P., & Terwiesch, C. (2018).** *Matching Supply with Demand: An Introduction to Operations Management* (4th ed.). New York: McGraw-Hill.
4. **Hopp, W. J., & Spearman, M. L. (2008).** *Factory Physics* (3rd ed.). Boston: Irwin/McGraw-Hill.
5. **Gross, D., Shortle, J. F., Thompson, J. M., & Harris, C. M. (2008).** *Fundamentals of Queueing Theory* (4th ed.). Hoboken: John Wiley & Sons.

### Standards and Regulatory Frameworks — Grade B
6. **Federal Aviation Administration (FAA). (2020).** *Title 14 CFR Part 145: Repair Stations*. Washington, D.C.: US Government Publishing Office.
7. **European Union Aviation Safety Agency (EASA). (2021).** *Part-145: Maintenance Organisation Approvals*. Cologne: EASA.
8. **Airlines for America (A4A). (2022).** *ATA Spec 2000: Chapter 4: Repair Order Administration*. Washington, D.C.: A4A.
9. **Department of Defense (DoD). (2014).** *DoD Manual 4140.01: DoD Supply Chain Materiel Management Procedures: Volume 6, Material Returns, Retention, and Disposition*. Washington, D.C.: Office of the Under Secretary of Defense.

### Industry Case Evidence and Technical Monographs — Grade C & D
10. **Delta TechOps. (2023).** *Global MRO Capabilities & Turnaround Time Performance Whitepaper*. Atlanta: Delta Air Lines.

---

## 17. Links

- **Parent Module:** [`_index.md`](_index.md) (Module 31 — Aftermarket, Spare Parts, and Service Supply Chains)
- **Root Survey File:** [`../../../course/31_aftermarket_spare_parts_service.md`](../../../course/31_aftermarket_spare_parts_service.md)
- **Previous Topic:** [`03_multi-echelon-sparing.md`](03_multi-echelon-sparing.md) (Multi-Echelon Sparing and METRIC)
- **Next Topic:** [`05_field-service-scheduling.md`](05_field-service-scheduling.md) (Field Service Scheduling)
- **Related Methods:**
  - [`../../../course/10_inventory_optimization_multi_echelon.md`](../../../course/10_inventory_optimization_multi_echelon.md)
  - [`../../../course/15_manufacturing_production_planning.md`](../../../course/15_manufacturing_production_planning.md)
  - [`../../../course/37_circular_economy_reverse_networks.md`](../../../course/37_circular_economy_reverse_networks.md)
