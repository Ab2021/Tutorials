# Demurrage and Detention

> Module 36 — Global Freight, Ports, and Trade Policy
> Topic 06 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Module 11 (Safety Stock & Service Levels), Module 14 (Network Design), Topic 01 (Incoterms and Trade Documents), Topic 02 (Total Landed Cost), Topic 05 (Port Operations and Congestion)
> Version 2026.5

---

## 1. Why this topic exists

In international containerized shipping, marine container equipment and port terminal real estate represent capital-intensive assets governed by velocity and asset turnover. Ocean common carriers invest tens of billions of dollars in standard twenty-foot equivalent units (TEUs), forty-foot dry containers (40DC), forty-foot high-cube containers (40HC), and temperature-controlled refrigerated containers (reefers). Marine terminal operators (MTOs) manage high-density container yards where spatial storage capacity is strictly bounded and vessel discharge schedules require rapid cargo evacuation. To prevent marine terminals from functioning as low-cost commercial warehouses and to ensure container equipment cycles rapidly back into maritime circulation, ocean carriers and terminal operators established two historical accessorial charge mechanisms:

1. **Demurrage:** A daily financial charge assessed on containers remaining inside the marine terminal gate, off-dock container yard (CY), or inland intermodal rail ramp beyond a contractually stipulated window of "free time" following vessel discharge.
2. **Detention (historically termed Per Diem in North American intermodal transport):** A daily financial charge assessed on container equipment remaining outside the terminal gate in the physical custody of the importer, drayage motor carrier, or cargo owner beyond an agreed free-time grace period following terminal outgate, ending only when the empty container is safely returned to an authorized ocean carrier facility.

In theoretical transport economics, demurrage and detention (D&D) charges operate as liquidated damages and behavioral incentives—what maritime law and regulatory authorities codify as the **Incentive Principle**. They exist to incentivize cargo owners (beneficial cargo owners / BCOs), non-vessel-operating common carriers (NVOCCs), and freight forwarders to clear import cargo promptly and cycle equipment back into maritime trade lanes.

In practical supply chain execution, however, systemic disruptions routinely decouple D&D charges from operational incentives, transforming them into punitive, uncontrollable financial liabilities. Between 2020 and 2022, the nine largest global container liner alliances collected more than $30 billion in demurrage and detention fees globally. U.S. import corridors bore a disproportionate share of these assessments. Shippers, drayage truckers, and intermodal intermediaries were invoiced hundreds of millions of dollars for container assets they were physically, operationally, or legally prevented from retrieving or returning.

The core economic failure stems from systemic bottlenecks entirely outside the cargo owner's operational control:
- **Terminal Congestion and Closed Stacks:** Marine terminals operating at 95%+ yard utilization routinely close gates, cancel drayage appointment slots, or declare entire container stacks inaccessible ("closed blocks"), while ocean carrier automated billing systems continue to run the demurrage clock.
- **Mandatory Dual Transactions (Matchbacks):** MTOs and steamship lines frequently enforce "dual-transaction" policies, prohibiting a drayage trucker from picking up an import container unless they simultaneously deliver an empty container of the exact same carrier line. When appointment slots for empty returns are capped or shut down, the trucker is locked out from picking up released import freight.
- **The North American Chassis Deficit and Split:** Unlike European and Asian logistics where containers move on carrier-supplied trailers or integrated truck chassis, U.S. ocean carriers divested their chassis fleets to independent Intermodal Equipment Providers (IEPs) between 2009 and 2014. If an IEP chassis pool at a port terminal is exhausted, or if a drayman must perform an uncompensated 35-mile "chassis split" to a secondary depot, demurrage accrues rapidly while the container sits grounded on the terminal pavement.
- **Regulatory and Customs Exam Holds:** Containers flagged by U.S. Customs and Border Protection (CBP) or Partner Government Agencies (PGAs)—such as the Food and Drug Administration (FDA) or USDA Animal and Plant Health Inspection Service (APHIS)—for non-intrusive imaging (VACIS) or intensive devanning at an off-dock Centralized Examination Station (CES) are placed in federal custody. Shippers cannot touch the freight, yet ocean carriers historically refused to toll (pause) the demurrage clock.
- **Progressive Stepped Tariff Schedules:** D&D tariffs are aggressively non-linear. Daily penalty rates escalate across progressive stepped tiers—for example, jumping from $150 per day in Tier 1 to $280 in Tier 2 and $420+ in Tier 3. A multi-container consignment caught in a terminal IT outage or customs audit can accrue accessorial fees exceeding the commercial value of the underlying cargo within two weeks.

Recognizing that D&D charges had degenerated into a perverse profit center that rewarded ocean carriers for port congestion, the United States Congress enacted the **Ocean Shipping Reform Act of 2022 (OSRA 2022)**, amending the Shipping Act of 1984. OSRA 2022 codified statutory standards prohibiting unjust and unreasonable practices under 46 U.S.C. § 41102(c), shifted the burden of proof in fee disputes onto ocean carriers, mandated 13 statutory data elements on every D&D invoice under 46 U.S.C. § 41104(d), and empowered the Federal Maritime Commission (FMC) to issue binding regulations governing billing privity (Docket No. 22-05) and reasonable D&D practices.

For supply chain analysts, freight procurement executives, and enterprise data scientists, managing demurrage and detention is no longer a clerical freight-audit task. It requires mathematical modeling of dwell probability distributions, automated ingestion and reconciliation of Electronic Data Interchange (EDI) milestone streams, integration of terminal appointment logs, and structured legal dispute workflows. Treating D&D as unpreventable operating friction permanently degrades corporate operating income.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Classify and demarcate** demurrage, detention, per diem, and terminal storage across North American split-billing regimes and European/Asian combined free-time regimes.
2. **Formulate** piecewise non-linear stepped tariff functions for standard dry, high-cube, refrigerated, and dangerous goods containers across multi-tiered dwell windows.
3. **Decompose** container dwell variance into root causes: customs exams (NII/VACIS vs intensive CES), terminal gate appointment rationing, chassis pool dislocation, and carrier-mandated dual transactions.
4. **Evaluate** the legal validity of D&D invoices under the FMC "Incentive Principle" (46 CFR § 545.5) and OSRA 2022 statutory standards (46 U.S.C. §§ 41102(c) and 41104).
5. **Audit** ocean carrier and MTO billing records against the mandatory statutory data elements codified in 46 U.S.C. § 41104(d) and FMC Docket No. 22-05.
6. **Quantify** the bullwhip interaction between container dwell delays, accessorial fees, landed cost variances, and inventory safety stock requirements at destination distribution centers.
7. **Construct** an automated audit and reconciliation engine in Python that cross-checks EDI 315/214 event logs against billing records to isolate disputable charges.
8. **Design** an enterprise dispute management protocol supported by evidentiary audit trails (terminal operating system snapshots, appointment rejection timestamps, and gate interchange receipts).

---

## 3. The core mental model

Demurrage and detention govern the custody lifecycle of a shipping container across physical, operational, and legal boundaries. The system operates on two parallel clocks: the **Terminal Storage / Demurrage Clock** (inside the gate) and the **Equipment Detention / Per Diem Clock** (outside the gate).

### The Container Custody and Free Time Lifecycle

```text
========================================================================================================================
                                     CONTAINER CUSTODY AND D&D ACCRUAL LIFECYCLE
========================================================================================================================

 [ OCEAN TRANSIT ]        [ IMPORT TERMINAL: INSIDE GATE ]                [ INLAND TRANSIT: OUTSIDE GATE ]
 Vessel Berths &        Container Discharged    Container Available        Drayage Trucker     Container Unloaded     Empty Returned
 Discharges Cargo       to Terminal Yard        for Pickup (LFD Starts)    Pulls Full Box      at Consignee Facility  to Terminal / Depot
        │                      │                       │                          │                   │                      │
        ▼                      ▼                       ▼                          ▼                   ▼                      ▼
  EDI 315: 'VA'          EDI 315: 'VD'           EDI 315: 'AV'              EDI 315: 'OA'       EDI 214: 'CD'          EDI 315: 'RD'
 (Vessel Arrival)      (Vessel Discharge)      (Available for Dray)        (Outgate Loaded)    (Cargo Delivered)      (Empty Ingate)

        ├──────────────────────┼───────────────────────┤                          ├───────────────────┼──────────────────────┤
        │  DISCHARGE BUFFER    │  DEMURRAGE FREE TIME  │ DEMURRAGE ACCRUAL        │ DETENTION FREE    │ DETENTION ACCRUAL    │
        │  (Vessel Unloading & │  (Typically 3 to 5    │ (Tiered Stepped Tariff:  │ TIME (Typically 4 │ (Tiered Per Diem:    │
        │   Yard Stacking)     │   Business Days)      │  Tier 1 -> Tier 2 -> 3)  │  to 7 Days)       │  Tier 1 -> Tier 2)   │
        └──────────────────────┴───────────────────────┴──────────────────────────┴───────────────────┴──────────────────────┘
        ▲                                              ▲                          ▲                                          ▲
        │                                              │                          │                                          │
   [DISCHARGE]                                [LAST FREE DAY: LFD]           [GATE-OUT]                             [EMPTY RETURN]
                                              Clock expires; charges         Demurrage ends;                         Detention ends;
                                              begin midnight                 Detention begins                        Equipment cleared
========================================================================================================================
```

### The Jurisdictional Split: North America vs. Europe and Asia

A primary source of analytical confusion in global supply chain planning is that the operational boundaries of D&D differ fundamentally between North America and the rest of the world:

1. **The North American Split Regime:**
   - **Demurrage** applies strictly to containers located *inside* the marine terminal gate or rail ramp after the expiration of demurrage free time. In U.S. ports, demurrage is frequently bifurcated into two separate charges:
     - *Carrier Demurrage:* The ocean carrier's charge for container equipment utilization while sitting on port property.
     - *Terminal Storage / Wharf Storage:* The MTO's charge for physical real estate occupancy in the yard.
   - **Detention (Per Diem):** Applies strictly *outside* the terminal gate. The clock starts the moment the container gates out of the port on a chassis and stops only when the empty container is returned to the carrier's designated empty depot or marine terminal.
   - **Chassis Per Diem:** A third independent fee assessed by Intermodal Equipment Providers (IEPs) for the daily rental of the roadable chassis carrying the container.

2. **The European and Asian Combined Free-Time Regime:**
   - Many European (e.g., Rotterdam, Antwerp, Hamburg) and Asian (e.g., Singapore, Shanghai, Busan) ports operate under a **Combined Demurrage & Detention (Merged Free Time)** tariff.
   - The carrier grants a single unified bucket of free days (e.g., 10 calendar days).
   - The unified clock starts upon vessel discharge (`VD`) and stops only when the empty container is returned to the carrier (`RD`). As long as the container is cleared, drayed, unloaded, and returned within 10 days, zero accessorial fees accrue, regardless of how many days were spent inside versus outside the port.
   - If the container exceeds the combined window, the carrier assesses a single combined stepped rate. Terminal storage, however, is billed separately by the port authority or terminal operator directly to the cargo receiver or barging operator.

```text
========================================================================================================================
                           COMPARATIVE JURISDICTIONAL ARCHITECTURE: SPLIT VS. COMBINED
========================================================================================================================
FEATURE                         NORTH AMERICAN SPLIT REGIME             EUROPEAN / ASIAN COMBINED REGIME
------------------------------------------------------------------------------------------------------------------------
Inside Terminal Gate            Demurrage (Carrier) + Storage (MTO)     Combined D&D (Carrier) + Quay Rent (Port/MTO)
Outside Terminal Gate           Detention / Per Diem (Carrier)          Combined D&D (Carrier)
Chassis Provisioning            Independent IEP Rental ($35-$50/day)     Integrated Trucker Trailer / Carrier Haulage
Free Time Allocation            Separated: 4-5 days Port, 5-7 days Street Merged: 10-14 Calendar Days Total
Regulatory Body                 Federal Maritime Commission (FMC)       DG COMP (EU) / National Maritime Courts
Statutory Protection            OSRA 2022 (46 U.S.C. §§ 41102, 41104)   Civil / Commercial Contract Law
========================================================================================================================
```

### The State Machine and Custody Transitions

The physical lifecycle of a container can be modeled mathematically as a formal finite state machine (FSM). Let $S(t)$ represent the container status at time $t$:

```text
                     [DISCHARGED (VD)]
                            │
                            ▼
                     [PENDING HOLDS] ──(CBP / FDA Hold)──► [GOV_HOLD (Tolled)]
                            │                                     │
                     (All Holds Cleared)                          │ (Release)
                            │                                     │
                            ▼                                     ▼
                     [AVAILABLE (AV)] ◄───────────────────────────┘
                            │
                     (Free Time Starts)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
      [FREE_TIME_PORT]           [PORT_DEMURRAGE (Tiered)]
              │                           │
              └─────────────┬─────────────┘
                            │
                    (Outgate Loaded: OA)
                            │
                            ▼
                   [TRANSIT_TO_CONSIGNEE]
                            │
                   (Arrive Consignee: CD)
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
     [FREE_TIME_STREET]         [STREET_DETENTION (Tiered)]
              │                           │
              └─────────────┬─────────────┘
                            │
                    (Ingate Empty: RD)
                            │
                            ▼
                    [TERMINATED / SETTLED]
```

### The Four Operational Friction Gates

Demurrage and detention accumulate when physical movement is blocked at one of four critical operational friction gates:

```text
[FRICTION GATE 1: ADMINISTRATIVE & CUSTOMS HOLDS]
Container grounded in terminal -> CBP/FDA/USDA hold placed -> Terminal blocks gate release ->
Carrier demurrage clock continues -> Contested: Container is physically unavailable.

[FRICTION GATE 2: TERMINAL APPOINTMENT RATIONING]
Importer ready to dray -> Terminal Operating System (TOS) has 0 available appointment slots ->
Last Free Day passes -> Demurrage accrues -> Contested: Lack of operational access.

[FRICTION GATE 3: THE CHASSIS SPLIT]
Container available at Terminal A -> Chassis pool empty at Terminal A -> Trucker must bobtail
15 miles to Depot B for chassis -> Extra transit time & cost -> Delays pull beyond LFD.

[FRICTION GATE 4: EMPTY RETURN LOCKOUT & DUAL TRANSACTIONS]
Consignee unloads box -> Trucker attempts return to Port -> Terminal announces "Empties Refused:
Yard Full" or "Dual Transaction Required" -> Trucker cannot ingate box -> Detention accrues daily.
```

---

## 4. Definitions and vocabulary

| Term | Operational Definition | Common Misconception | Measurement Unit / Coding |
|---|---|---|---|
| **Demurrage** | Daily charge assessed by ocean carriers or terminal operators on loaded or empty containers remaining on terminal property past free time. | Confusing demurrage with detention; assuming demurrage includes inland drayage delays. | USD (\$) or local currency per TEU/FEU per day; EDI 310 accessorial code `DEM`. |
| **Detention** | Daily charge assessed by the equipment owner (ocean carrier or lessor) for holding container equipment outside the terminal past free time. | Believing detention only applies to warehouse unloading delays; it covers the entire gate-out to empty-return window. | USD (\$) per container per day; EDI 310 accessorial code `DET`. |
| **Per Diem** | Daily rental charge for equipment usage; in U.S. drayage, used interchangeably with container detention or intermodal chassis rental. | Believing per diem is a government tax; it is a private commercial accessorial charge. | USD (\$) per day per asset (container or chassis). |
| **Free Time** | Contractually or tariff-stipulated grace period during which no demurrage or detention charges are assessed. | Assuming free time is standard across all carriers; it varies drastically by contract tier, equipment type, and lane. | Elapsed days: calendar days or working/business days. |
| **Last Free Day (LFD)** | The final calendar or business day of the free-time window before daily accessorial charges commence at 00:01 hours the following day. | Assuming LFD is static; LFD changes dynamically if customs holds or terminal closures extend the window. | Date (YYYY-MM-DD); EDI 315 code `LFD`. |
| **Terminal Storage** | Fee assessed by a Marine Terminal Operator (MTO) for physical land occupancy and container stacking in the container yard. | Assuming terminal storage is identical to carrier demurrage; in North America, they can be billed separately. | USD (\$) per TEU/FEU per day. |
| **Physical Availability** | Condition where a container is physically unstacked, located in an accessible transfer zone, cleared of all customs/carrier holds, and open for drayage. | Conflating constructive arrival (`VD`) with physical readiness; a container buried in a closed stack is unavailable. | Binary flag: True/False; EDI 315 status `AV`. |
| **Constructive Placement** | Legal doctrine where a carrier notifies an importer that cargo has arrived, shifting the custodial clock even if physical delivery cannot occur immediately. | Believing constructive notice justifies charging demurrage when the terminal gate is physically shut. | Legal status / notification timestamp. |
| **Stepped Tariff** | Non-linear progressive pricing structure where daily D&D rates increase across successive dwell time intervals. | Assuming D&D is a flat linear daily rate; rates can double or triple in higher tiers. | Piecewise step function: \$/day per tier bracket. |
| **Dual Transaction (Matchback)** | Terminal operating policy requiring a drayage truck to bring an empty container into the port in order to be allowed to pick up an import full container. | Assuming truckers can always pick up containers on single-leg runs; dual requirements cause appointment gridlock. | Operational rule / constraint. |
| **Chassis Split** | Operational movement where a motor carrier must pick up an intermodal chassis at a different geographic location from where the container is staged. | Assuming chassis are always mounted to containers at marine terminals; marine containers are grounded in stacks. | Geographic split: miles / drayage surcharge. |
| **Intermodal Equipment Provider (IEP)** | Commercial entity that owns, leases, and maintains intermodal chassis fleets used to move maritime containers over public highways. | Assuming ocean carriers own all chassis; U.S. carriers divested chassis fleets to independent IEPs between 2009 and 2014. | Entity identifier / SCAC code. |
| **Centralized Examination Station (CES)** | Privately operated, CBP-bonded facility where imported containers are transferred for physical, intensive customs inspections. | Believing customs exams occur at the terminal berth; containers must be drayed off-dock to a CES. | Facility code / FIRMS code. |
| **Non-Intrusive Inspection (NII)** | Radiographic scanning (e.g., VACIS / high-energy X-ray) of containers at terminal gates or railheads without devanning cargo. | Assuming NII causes multi-week delays; NII typically takes 24–48 hours unless secondary anomalies appear. | Exam type / CBP hold status. |
| **Incentive Principle** | FMC doctrine establishing that D&D charges are legally permissible under 46 U.S.C. § 41102(c) only when they incentivize cargo movement. | Assuming carriers have an absolute right to collect D&D regardless of operational impediments. | FMC regulatory standard (46 CFR § 545.5). |
| **OSRA 2022** | U.S. federal statute (Ocean Shipping Reform Act of 2022) reforming maritime regulation, shifting D&D dispute burdens, and mandating invoice transparency. | Believing OSRA eliminated D&D; it eliminated unreasonable practices and non-compliant invoicing. | Public Law 117-146; 46 U.S.C. § 41101 et seq. |
| **Mandatory Invoice Elements** | Statutory checklist of 13+ data fields required on any D&D invoice under 46 U.S.C. § 41104(d); omission voids the invoice. | Treating invoice errors as minor clerical details; failure to include statutory fields extinguishes payment obligations. | Statutory data schema (46 U.S.C. § 41104(d)). |
| **Billing Privity** | FMC rule (Docket No. 22-05) mandating that ocean carriers and MTOs may only invoice D&D to the party with whom they have a direct contractual relationship. | Billing drayage motor carriers who have no contractual relationship with the ocean carrier for ocean demurrage. | Legal doctrine (46 CFR Part 541). |
| **Terminal Appointment System (TAS)** | Digital scheduling platform (e.g., eModal, Advent e-Gate) used by MTOs to meter gate traffic and allocate truck entry time slots. | Assuming appointments are guaranteed; TAS platforms frequently run out of slots days before LFD. | Time-window token / appointment ID. |
| **Empty Return Restriction** | Carrier or terminal directive rejecting empty container returns due to yard saturation, or redirecting returns to distant off-dock depots. | Assuming empty containers can always be returned to the discharge port; restrictions cause per diem spikes. | Operational status: open, closed, redirected. |
| **Merchant Haulage** | Transportation arrangement where the cargo owner (BCO) contracts directly with the drayage trucker and manages inland movement. | Believing carrier is responsible for inland delays under merchant haulage; BCO bears detention risk. | Commercial routing code. |
| **Carrier Haulage** | Transportation arrangement where the ocean carrier contracts the inland drayage or rail move under a through bill of lading (intermodal door move). | Assuming demurrage and detention apply normally; under carrier haulage, carrier cannot charge detention for its own trucker delays. | Commercial routing code. |
| **Street Turn (Triangulation)** | The immediate reuse of an empty import container for an export booking without returning the equipment to the marine terminal. | Assuming street turns can be executed unilaterally; carrier electronic authorization via eModal or Avantida is mandatory. | Process milestone / authorization code. |
| **Tolling (Clock Freeze)** | The pausing or suspension of the free-time clock due to government holds, terminal closures, or appointment shortages. | Assuming clock runs continuously without exception; tolling extends LFD day-for-day. | State variable / day increment. |

---

## 5. The mechanics, step by step

Managing demurrage and detention requires a closed-loop procedural framework spanning five operational phases. Each phase represents a shift in physical custody, legal accountability, and clock status.

```text
========================================================================================================================
                                     FIVE-PHASE D&D OPERATIONAL MECHANICS
========================================================================================================================

  PHASE 1: DISCHARGE & AVAILABILITY
  Vessel Discharge -> EDI 315 ('VD') -> Customs & Carrier Releases -> Physical Unstacking -> EDI 315 ('AV')
         │
         ▼
  PHASE 2: CLOCK INITIATION & TARIFF APPLICATION
  Establish Start Date -> Apply Calendar/Business Rules -> Calculate LFD -> Map Stepped Tier Thresholds
         │
         ▼
  PHASE 3: DRAYAGE & TERMINAL INTERACTION
  TAS Slot Booking -> Chassis Allocation -> Dual Transaction Matching -> Gate In/Out (EDI 315 'OA')
         │
         ▼
  PHASE 4: INLAND DWELL & EMPTY RETURN
  Transit to Consignee -> Unloading / Devanning -> Empty Depot Assignment -> Empty Ingate (EDI 315 'RD')
         │
         ▼
  PHASE 5: AUDIT, RECONCILIATION & DISPUTE
  Invoice Ingestion -> OSRA 2022 Statutory Audit -> Milestone Reconciliation -> Dispute Dossier Submission
========================================================================================================================
```

### Phase 1: Vessel Discharge and Physical Availability Determination

1. **Vessel Discharge Milestone (`VD`):**
   - The container is lifted from the vessel cell guide by a ship-to-shore (STS) gantry crane and placed onto a terminal yard tractor or automated guided vehicle (AGV).
   - The terminal operating system (TOS) generates an EDI 315 status message with event code `VD` (Vessel Discharged).
   - *Critical Legal Distinction:* Vessel discharge does **not** equal physical availability. The container may be placed into a multi-container stack 5 containers high and 6 rows deep.

2. **Hold Verification and Clearance:**
   - The TOS cross-references electronic releases across three mandatory gates:
     a. *Customs Clearance:* CBP electronic entry release (CBP Form 3461 / 7501 or ACE electronic 1C release).
     b. *Carrier Freight Release:* Ocean carrier confirms that ocean freight charges, bill of lading endorsements, and any origin collect charges are satisfied.
     c. *Terminal Release / Fee Clearance:* Payment of local terminal handling charges (THC) or PierPass / clean truck fund fees where applicable.
   - If CBP flags the container for a 1H (Intensive Exam), 2H (VACIS scan), or PGA hold (e.g., FDA hold for food products), the container is placed on terminal hold.

3. **Physical Availability Milestone (`AV`):**
   - Under OSRA 2022 and modern FMC doctrine, a container is legally and operationally available only when:
     - All administrative and government agency holds are lifted.
     - The container is mounted on a chassis (in wheeled operations) or positioned in an open, actively worked transfer block (in grounded operations).
     - The terminal gate appointment system has bookable appointments available for the container during regular gate hours.
   - The TOS transmits EDI 315 event code `AV` (Available for Pickup) with a verifiable timestamp.

### Phase 2: Free Time Clock Initiation, Stepped Tariffs, and Lead Time Variance

1. **Determining the Clock Start:**
   - Marine terminal tariffs define the demurrage clock start. Common contractual triggers include:
     - *Trigger A:* 00:01 hours on the calendar day immediately following the availability timestamp (`AV`).
     - *Trigger B:* 07:00 hours on the first business day following discharge (`VD`), provided the container is available.
   - *Business Days vs. Calendar Days:* Under standard FMC guidelines and port tariffs, free time traditionally counts business days (Monday through Friday, excluding official public and longshore union holidays). However, once free time expires, many ocean carrier tariffs shift to **calendar days**, meaning Saturdays, Sundays, and holidays are billed at full stepped rates.

2. **Mathematical Formulation of Stepped Tariffs:**
   Ocean carriers and MTOs employ piecewise non-linear stepped tariffs to penalize long-dwell containers. Let $\tau$ represent total elapsed dwell days, and let $\tau_{\text{free}}$ denote contractual free time. The chargeable dwell days $\tau_{\text{bill}}$ are:

   $$\tau_{\text{bill}} = \max(0, \tau - \tau_{\text{free}})$$

   Tariff schedules define $K$ discrete tiers. Each tier $k \in \{1, 2, \dots, K\}$ is characterized by:
   - A day duration interval $[t_{k-1}, t_k)$, where $t_0 = \tau_{\text{free}}$.
   - A daily charging rate $r_k$ (in \$/day), with $r_1 < r_2 < \dots < r_K$.

   The cumulative demurrage or detention charge $C(\tau)$ is computed using the piecewise formulation:

   $$C(\tau) = \sum_{k=1}^K r_k \cdot \max\left(0, \min(\tau, t_k) - \max(\tau_{\text{free}}, t_{k-1})\right)$$

   For an operational model with 3 stepped tiers following free time:
   - Tier 1: Days 1 to 4 beyond free time at rate $r_1$
   - Tier 2: Days 5 to 8 beyond free time at rate $r_2$
   - Tier 3: Days 9+ beyond free time at rate $r_3$

   The cumulative cost function expands explicitly to:

   $$C(\tau) = \begin{cases} 
   0 & \text{if } \tau \le \tau_{\text{free}} \\
   r_1(\tau - \tau_{\text{free}}) & \text{if } \tau_{\text{free}} < \tau \le t_1 \\
   r_1(t_1 - \tau_{\text{free}}) + r_2(\tau - t_1) & \text{if } t_1 < \tau \le t_2 \\
   r_1(t_1 - \tau_{\text{free}}) + r_2(t_2 - t_1) + r_3(\tau - t_2) & \text{if } \tau > t_2 
   \end{cases}$$

   The marginal cost of an additional day of dwell $\frac{dC}{d\tau}$ is a step function jumping from $0$ to $r_1$, $r_2$, and $r_3$.

3. **Lead Time Variance and Safety Stock Bullwhip:**
   Container port dwell is not merely an accessorial expense; it is a primary driver of supply chain lead time variance. Let total international replenishment lead time $L$ be the sum of ocean voyage time $L_{\text{ocean}}$, port dwell time $L_{\text{dwell}}$, and inland drayage transit time $L_{\text{inland}}$:

   $$L = L_{\text{ocean}} + L_{\text{dwell}} + L_{\text{inland}}$$

   Assuming voyage and inland transit are independent of port dwell:

   $$\sigma_L^2 = \sigma_{\text{ocean}}^2 + \sigma_{\text{dwell}}^2 + \sigma_{\text{inland}}^2$$

   When port dwell distribution develops a heavy tail (due to customs holds, chassis shortages, or terminal gridlock), $\sigma_{\text{dwell}}^2$ surges by an order of magnitude. Under classic inventory theory, safety stock $SS$ required to maintain service level $z$ at a destination distribution center with daily demand mean $\bar{D}$ and demand variance $\sigma_D^2$ scales with lead time variance:

   $$SS = z \cdot \sqrt{\bar{L} \sigma_D^2 + \bar{D}^2 \sigma_L^2}$$

   The total financial cost of container dwell combines accessorial fees with the capital cost of delayed inventory:

   $$TC_{\text{dwell}}(\tau) = C_{\text{dem}}(\tau) + C_{\text{det}}(\tau) + \tau \cdot v \cdot \left(\frac{r_{\text{wacc}} + r_{\text{spoil}}}{365}\right)$$

   where $v$ is the shipment value, $r_{\text{wacc}}$ is the weighted average cost of capital, and $r_{\text{spoil}}$ is the obsolescence or spoilage rate.

### Phase 3: Drayage Execution, Gate Appointments, and Chassis Mechanics

1. **Terminal Appointment Booking and Slot Rationing:**
   - The drayage motor carrier accesses the terminal appointment system (TAS) to secure an import pickup slot.
   - *The Phantom Appointment Phenomenon:* When gate appointments are scarce, drayage carriers deploy automated scripts or dispatchers to hoard open slots across multiple containers. Terminals respond by capping daily bookings and assessing $50 to $100 "no-show" penalties.
   - If the TAS shows zero availability prior to the container's LFD, a commercial conflict arises. Under the FMC Incentive Principle, if the shipper or motor carrier makes diligent, documented attempts to secure an appointment but none are offered by the MTO, demurrage cannot be lawfully assessed for those locked-out days.

2. **Chassis Provisioning and Pool Dynamics:**
   - Intermodal container drayage requires an intermodal chassis meeting Federal Motor Carrier Safety Administration (FMCSA) roadability standards.
   - Three operational chassis models exist:
     a. *Intermodal Equipment Provider (IEP) Pools:* Truckers rent chassis from interoperable pools (e.g., DCLI, TRAC Intermodal, Flexi-Van) at daily per diem rates ($35–$50/day).
     b. *BCO-Dedicated Chassis Fleets:* High-volume shippers lease dedicated chassis to eliminate terminal pool shortages.
     c. *Carrier Haulage Wheels:* Ocean carrier provides the chassis as part of an integrated through rate.
   - *The Chassis Split Failure Mode:* If the marine terminal has containers grounded on the pavement but has no bare chassis on site, the drayman must travel to an off-dock chassis depot, inspect and hook a chassis, and drive to the marine terminal. If the off-dock depot has no chassis, the container sits immobilized, and demurrage accrues.
   - *The Tri-Axle Highway Limit:* Standard tandem-axle chassis carry gross vehicle weights up to 80,000 lbs under the U.S. Federal Bridge Formula. A 20-foot container carrying heavy dense cargo (e.g., tile, metal ingots, wine) exceeding 38,000 lbs cargo weight requires a specialized tri-axle chassis. A shortage of tri-axle chassis frequently grounds heavy containers in port yards.

3. **The Dual-Transaction Bottleneck:**
   - To manage yard congestion, MTOs enforce "dual-transaction" rules: a motor carrier cannot enter to pick up an import container unless it simultaneously delivers an export container or an empty container.
   - If the ocean carrier restricts empty returns to specific off-dock yards or closes empty receiving lines, the drayman cannot execute the inbound leg of the dual transaction, blocking import container retrieval.

4. **Terminal Outgate Milestone (`OA`):**
   - The container successfully passes terminal optical character recognition (OCR) portal scanners and physical interchange inspection.
   - The MTO issues an Equipment Interchange Receipt (EIR) or electronic gate ticket.
   - The TOS transmits EDI 315 status `OA` (Outgate Loaded).
   - **Crucial Clock Transition:** At the instant of `OA`, the **demurrage clock terminates**, and the **detention (per diem) clock commences**.

### Phase 4: Consignee Facility Dwell, Unloading, and Empty Return Friction

1. **Inland Transit and Consignee Facility Dwell:**
   - The container transits to the consignee's warehouse or distribution center (DC).
   - *Live Unload vs. Drop-and-Hook:*
     - *Live Unload:* The drayage driver waits at the facility dock while the warehouse devanns the freight (standard free driver waiting time is 2 hours; excess wait incurs trucker driver detention charges). The container is emptied immediately and drayed back, minimizing equipment dwell.
     - *Drop-and-Hook:* The driver uncouples the chassis and container at the DC yard and departs. The warehouse devanns the cargo over several days. The container remains on the equipment detention clock.
   - If DC warehouse doors are congested, drop-and-hook containers sit in the yard, consuming detention free days.

2. **Empty Return Routing Instructions:**
   - Once devanned, the container must be returned to the ocean carrier.
   - The ocean carrier issues electronic return directives through portals (e.g., ReturnStates, eModal, or carrier APIs).
   - Carriers frequently designate specific return locations:
     - Return to original marine terminal.
     - Return to off-dock inland container depot (ICD) or rail ramp.
     - *Redirection:* Carrier redirects the return to a depot 30 miles in the opposite direction.
     - *Empty Freeze:* Carrier issues a complete stop on empty returns for that specific equipment type and size due to port surplus.

3. **Empty Ingate Milestone (`RD`):**
   - The empty container is inspected for physical structural damage at the depot gate and accepted into yard inventory.
   - The depot issues an empty interchange receipt.
   - The carrier or depot transmits EDI 315 event code `RD` (Empty Container Returned) or EDI 214 event code `E` (Empty Ingate).
   - **The detention clock terminates.**

### Phase 5: Automated Audit, Regulatory Challenge, and Dispute Adjudication

1. **Invoice Ingestion and Data Parsing:**
   - D&D invoices are received electronically via EDI 310, PDF scrapers, or freight audit platforms.
   - Every invoice line item is decomposed into:
     - Equipment ID (Container Number and Size/Type).
     - Bill of Lading (B/L) number.
     - Voyage and vessel name.
     - Free time granted vs. free time billed.
     - Start and end timestamps for demurrage and detention.
     - Applicable tariff rule and stepped tier breakdown.

2. **The OSRA 2022 Statutory Audit Checklist (46 U.S.C. § 41104(d)):**
   Under U.S. federal law, an ocean carrier or MTO is legally barred from collecting D&D if the invoice fails to include all mandatory statutory elements:

```text
========================================================================================================================
                               OSRA 2022 STATUTORY INVOICE COMPLIANCE CHECKLIST
========================================================================================================================
 [ ] 1. Bill of Lading (B/L) Number
 [ ] 2. Container Number(s)
 [ ] 3. Port of Discharge (POD)
 [ ] 4. Vessel Name and Voyage Number
 [ ] 5. Date Container Made Physically Available
 [ ] 6. Demurrage / Detention Free Time Start Date
 [ ] 7. Demurrage / Detention Free Time End Date (Last Free Day - LFD)
 [ ] 8. Applicable Tariff Rule / Service Contract Number
 [ ] 9. Allowed Free Time (in integer days)
 [ ] 10. Start Date of Invoiced Charges
 [ ] 11. End Date of Invoiced Charges
 [ ] 12. Daily Stepped Tariff Rate Breakdown
 [ ] 13. Total Amount Due
 [ ] 14. FMC Dispute Point of Contact (Email, Phone, Postal Address)
 [ ] 15. Certification Statement: "Charges Comply with FMC Regulations Under 46 CFR Part 541 and 46 U.S.C. § 41104"
========================================================================================================================
```

   *Statutory Rule:* Under 46 U.S.C. § 41104(f), if an invoice fails to contain all required information, the billed party is **under no statutory obligation to pay the fee**.

3. **Milestone Cross-Examination and Discrepancy Flagging:**
   The automated audit engine joins three disparate data streams:
   - Stream A: Carrier/MTO Invoices.
   - Stream B: Carrier EDI 315 / 214 milestone timestamps.
   - Stream C: Independent physical evidence: TAS appointment reservation logs, CBP ACE entry status history, terminal closure bulletins, and truck telematics.

   The engine checks five core violation rules:
   - *Rule 1 (Premature Clock):* Did the free time clock start before the container was physically available (`AV`)?
   - *Rule 2 (Customs Toll Failure):* Did charges accrue during active government holds (VACIS/CES/FDA)?
   - *Rule 3 (Gate Closure Toll Failure):* Were charges assessed on days the terminal was closed (strikes, weather, holidays, IT outages)?
   - *Rule 4 (No-Appointment Window):* Did demurrage accrue when the motor carrier documented zero available TAS slots?
   - *Rule 5 (Empty Turn-Away):* Did detention accrue when the carrier or terminal refused empty container ingate?

4. **Dispute Dossier Assembly and Submission:**
   - If discrepancies exist, an automated dispute dossier is generated.
   - The dossier compiles:
     - Disputed invoice copy.
     - Timeline visualization comparing billed days vs. lawful days.
     - Evidentiary attachments (screenshots of TAS appointment rejection screens with date/time watermarks, CBP hold release notices, terminal closure alerts).
     - Formal dispute citation under 46 U.S.C. § 41102(c) and 46 CFR § 545.5.
   - Under FMC rules, the carrier must respond and resolve or justify the charge within 30 days. Unresolved improper charges are escalated to the FMC's Office of Consumer Affairs and Dispute Resolution Services (CADRS) or filed as a formal Charge Complaint under 46 U.S.C. § 41310.

---

## 6. Worked example

### Shipment Scenario: Transpacific Inbound Consignment

A medical instrumentation and precision optics manufacturer imports a high-value consignment from Yantian, China, to Los Angeles, California. The shipment comprises **10 containers**:
- **8 Standard 40' Dry High-Cube Containers (40HC):** Diagnostics and electronic testing hardware.
- **2 Refrigerated 40' Containers (40RF Reefers):** Temperature-controlled liquid assay reagents requiring strict $+4^\circ\text{C}$ climate control.

The ocean carrier issues an ocean through bill of lading to the Port of Los Angeles (POLA). The shipment is executed under Merchant Haulage.

### Ocean Carrier Contractual Free Time and Stepped Tariff Schedule

```text
TARIFF SCHEDULE: POLA DISCHARGE (USD)
------------------------------------------------------------------------------------------------------
EQUIPMENT TYPE: 40' DRY HIGH-CUBE (40HC)
- Free Time: 4 Working Days (Monday through Friday, excluding terminal holidays)
- Tier 1 (Days 1 to 4 past free time):    $180 per container per calendar day
- Tier 2 (Days 5 to 8 past free time):    $280 per container per calendar day
- Tier 3 (Days 9+ past free time):        $420 per container per calendar day

EQUIPMENT TYPE: 40' REEFER (40RF)
- Free Time: 2 Working Days
- Tier 1 (Days 1 to 3 past free time):    $350 per container per calendar day
- Tier 2 (Days 4+ past free time):        $550 per container per calendar day
- Mandatory Terminal Reefer Power & Daily Monitoring Fee: $95 per reefer per calendar day (Accrues from
  day of discharge, not subject to free time).

DETENTION (PER DIEM) TARIFF (BOTH TYPES - OUTSIDE GATE)
- Free Time: 5 Calendar Days starting the day of gate-out
- Tier 1 (Days 1 to 4 past free time):    $175 per container per calendar day
- Tier 2 (Days 5+ past free time):        $300 per container per calendar day
------------------------------------------------------------------------------------------------------
```

### Operational Timeline and Milestone Event Log

The consignment experiences the following operational milestones across a 16-day window:

- **Day 1 (Monday):** Vessel berths. All 10 containers discharged (`VD`) between 08:00 and 14:00.
  - CBP places a 2H Non-Intrusive Inspection (VACIS) hold on **3 dry containers** (Containers C-01, C-02, C-03).
  - FDA places an intensive physical sampling hold on **1 reefer container** (Reefer R-01), requiring transfer to an off-dock CES.
  - The remaining 5 dry containers (C-04 through C-08) and 1 reefer (R-02) have clean customs and carrier releases, but are placed in a closed yard block pending terminal stack reorganization.
- **Day 2 (Tuesday):** Terminal completes yard moves. Containers C-04 through C-08 and Reefer R-02 are officially declared **Available (`AV`)** at 10:00.
  - Free time clock for C-04..C-08 starts Day 3 (Wednesday). Nominal LFD is Day 6 (Monday), spanning 4 working days (Wed, Thu, Fri, Mon).
  - Free time clock for Reefer R-02 starts Day 3 (Wednesday). Nominal LFD is Day 4 (Thursday), spanning 2 working days (Wed, Thu).
- **Day 3 (Wednesday):** Major terminal IT failure. All terminal gates close at 11:00 for the remainder of Wednesday and all of **Day 4 (Thursday)**. Zero truck appointments are honored or available.
- **Day 5 (Friday):** Terminal gates reopen. Massive truck queue. Drayage carrier attempts to book TAS appointments for C-04..C-08 and R-02. All appointment slots for Friday are fully booked. Earliest available TAS appointment is Day 8 (Monday).
  - CBP completes VACIS scan on C-01..C-03; customs hold released at 16:00. Available milestone (`AV`) set for Day 8 (Monday).
- **Day 6 (Saturday) & Day 7 (Sunday):** Weekend. Terminal closed for standard gate operations.
- **Day 8 (Monday):**
  - Motor carrier picks up Reefer R-02 and dry containers C-04, C-05, C-06 (Outgate `OA`).
  - Motor carrier attempts to pick up C-07 and C-08. Terminal gate rejects entry: ocean carrier has placed an emergency "Empty Ingate Freeze" on that terminal, preventing the trucker from executing the mandatory dual transaction. Pickup delayed to Day 10.
  - FDA inspects Reefer R-01 at CES; release granted at 17:00. Container returned to terminal yard. Designated Available (`AV`) Day 9 (Tuesday).
- **Day 9 (Tuesday):** Free time clock starts for C-01..C-03 (LFD = Day 12, Friday). Free time clock starts for Reefer R-01 (LFD = Day 10, Wednesday).
- **Day 10 (Wednesday):**
  - Motor carrier secures single-transaction override. Picks up C-07 and C-08 (`OA`).
  - Motor carrier picks up Reefer R-01 (`OA`).
- **Day 11 (Thursday):** Motor carrier picks up C-01, C-02, C-03 (`OA`).
- **Inland Operations and Empty Returns:**
  - Containers C-04..C-06 and Reefer R-02 devanned at DC on Day 11. Empty containers returned (`RD`) on Day 12 (Friday). Total days outgate = 4 days (within 5-day detention free time $\implies \$0$ detention).
  - Containers C-07, C-08 and Reefer R-01 devanned on Day 14. Returned (`RD`) on Day 16. Total days outgate = 6 days (1 day past 5-day free time $\implies$ Tier 1 detention).
  - Containers C-01, C-02, C-03 devanned on Day 13. Returned (`RD`) on Day 15. Total days outgate = 4 days ($\implies \$0$ detention).

### Comprehensive Day-by-Day Operational Ledger

```text
========================================================================================================================
                                    DAY-BY-DAY OPERATIONAL LEDGER (DAYS 1 TO 16)
========================================================================================================================
DAY  DATE / NAME  C-01..C-03 (VACIS)   C-04..C-06 (CLEAN)   C-07..C-08 (DUAL-TX) R-01 (FDA CES)      R-02 (CLEAN REEFER)
------------------------------------------------------------------------------------------------------------------------
D01  Mon (VD)     Discharged / CBP 2H  Discharged / Blocked Discharged / Blocked Discharged / FDA 1H Discharged / Blocked
D02  Tue          CBP Exam Pending     Declared AV (10:00)  Declared AV (10:00)  CES Dray Pending     Declared AV (10:00)
D03  Wed          CBP Exam Pending     GATE CRASH (Tolled)  GATE CRASH (Tolled)  At Off-Dock CES      GATE CRASH (Tolled)
D04  Thu          CBP Exam Pending     GATE CRASH (Tolled)  GATE CRASH (Tolled)  At Off-Dock CES      GATE CRASH (Tolled)
D05  Fri          CBP Rel (16:00)      TAS FULL (Tolled)    TAS FULL (Tolled)    At Off-Dock CES      TAS FULL (Tolled)
D06  Sat          Weekend              Weekend              Weekend              Weekend              Weekend
D07  Sun          Weekend              Weekend              Weekend              Weekend              Weekend
D08  Mon          Declared AV          PULLED (OA)          DUAL-TX LOCKOUT      CES Released (17:00) PULLED (OA)
D09  Tue (Free 1) Port Dwell           Inland Transit       DUAL-TX LOCKOUT      Declared AV          Inland Transit
D10  Wed (Free 2) Port Dwell           DC Devanning         PULLED (OA)          PULLED (OA)          DC Devanning
D11  Thu (Free 3) PULLED (OA)          DC Devanning         Inland Transit       Inland Transit       DC Devanning
D12  Fri          Inland Transit       EMPTY RETURNED (RD)  DC Devanning         DC Devanning         EMPTY RETURNED (RD)
D13  Sat          DC Devanning         Settled              DC Devanning         DC Devanning         Settled
D14  Sun          DC Devanning         Settled              DC Devanning         DC Devanning         Settled
D15  Mon          EMPTY RETURNED (RD)  Settled              DC Devanning         DC Devanning         Settled
D16  Tue          Settled              Settled              EMPTY RETURNED (RD)  EMPTY RETURNED (RD)  Settled
========================================================================================================================
```

### The Invoiced Bill vs. The Lawful OSRA 2022 Audit

The ocean carrier issues an invoice totaling **$15,945.00**. The carrier's billing system computed demurrage starting strictly from the day after vessel discharge (Day 2), ignoring terminal gate closures, customs holds, and dual-transaction lockouts.

Let us perform the container-by-container audit:

#### 1. Dry Containers C-04, C-05, C-06 (Clean Cargo)
- *Carrier Billing:*
  - Discharge Day 1. Free time: 4 days (Days 2, 3, 4, 5).
  - Chargeable days begin Day 6 (Saturday). Outgate on Day 8 (Monday).
  - Charged Days 6, 7, 8 = 3 calendar days @ Tier 1 ($180) = $540 per container $\times 3 = \$1,620.00$.
- *Lawful Audit:*
  - Availability was Day 2 (Tuesday). Working day free time clock: Wed Day 3 (closed), Thu Day 4 (closed).
  - Under FMC Docket No. 20-04, days when the terminal gate is inoperable **cannot count against free time**. Days 3 and 4 are tolled.
  - Adjusted free days: Day 5 (Friday), Day 8 (Monday - weekend excluded).
  - On Day 5, zero TAS appointments were available. Free time tolled.
  - LFD extends to Day 8 (Monday). Outgate occurred on Day 8.
  - **Lawful Demurrage: 0 days $\implies \$0.00$.**
  - **Disputed Overcharge: $\$1,620.00$.**

#### 2. Dry Containers C-07, C-08 (Dual-Transaction Lockout)
- *Carrier Billing:*
  - Charged from Day 6 through Day 10 = 5 calendar days.
  - Tier 1 (4 days @ $180) + Tier 2 (1 day @ $280) = $\$720 + \$280 = \$1,000$ per container $\times 2 = \$2,000.00$.
- *Lawful Audit:*
  - Adjusted LFD was Day 8 (Monday).
  - On Day 8, trucker arrived to pull C-07 and C-08 but was turned away due to the carrier's dual-transaction lockout / empty freeze.
  - Under 46 U.S.C. § 41102(c), a carrier cannot charge demurrage when its own operational policies prevent cargo retrieval. Days 8, 9, and 10 are tolled.
  - Outgate occurred Day 10.
  - **Lawful Demurrage: 0 days $\implies \$0.00$.**
  - **Disputed Overcharge: $\$2,000.00$.**

#### 3. Dry Containers C-01, C-02, C-03 (CBP VACIS Hold)
- *Carrier Billing:*
  - Carrier billed from Day 6 through Day 11 (Outgate) = 6 calendar days.
  - Tier 1 (4 days @ $180) + Tier 2 (2 days @ $280) = $\$720 + \$560 = \$1,280$ per container $\times 3 = \$3,840.00$.
- *Lawful Audit:*
  - Containers were subject to an active government hold and were physically inaccessible until released Friday Day 5 at 16:00.
  - Physical availability (`AV`) declared Day 8 (Monday).
  - Free time: 4 working days = Day 8 (Mon), Day 9 (Tue), Day 10 (Wed), Day 11 (Thu).
  - Outgate occurred on Day 11 (Thursday). Pickup occurred on or before LFD.
  - **Lawful Demurrage: 0 days $\implies \$0.00$.**
  - **Disputed Overcharge: $\$3,840.00$.**

#### 4. Reefer R-02 (Clean High-Cube Reefer)
- *Carrier Billing:*
  - Free time: 2 days (Days 2 and 3). Billed Days 4 through 8 = 5 calendar days.
  - Tier 1: 3 days @ $350 = $1,050.
  - Tier 2: 2 days @ $550 = $1,100.
  - Demurrage charged: $\$2,150.00$.
  - Reefer power/monitoring: 8 days (Days 1 to 8) @ $95/day = $\$760.00$.
  - Total billed for R-02: $\$2,910.00$.
- *Lawful Audit:*
  - Available Day 2. Days 3 and 4 gates were closed (tolled).
  - Free time runs Day 5 (Friday) and Day 8 (Monday).
  - Outgate occurred Day 8.
  - **Lawful Demurrage: 0 days $\implies \$0.00$.**
  - Reefer power/monitoring is a physical utility service rendered by the terminal: 8 days @ $95 = \$760.00$ is legally valid.
  - **Lawful Total: $\$760.00$.**
  - **Disputed Overcharge: $\$2,150.00$.**

#### 5. Reefer R-01 (FDA Intensive Exam at CES)
- *Carrier Billing:*
  - Carrier billed demurrage from Day 4 through Day 10 = 7 calendar days.
  - Tier 1: 3 days @ $350 = $1,050.
  - Tier 2: 4 days @ $550 = $2,200.
  - Demurrage charged: $\$3,250.00$.
  - Reefer power/monitoring: 10 days @ $95 = $\$950.00$.
  - Total billed for R-01: $\$4,200.00$.
- *Lawful Audit:*
  - Container was under FDA hold and drayed off-dock to a CES. The container was not physically available at the marine terminal until Day 9.
  - Free time: 2 working days = Day 9 (Tue) and Day 10 (Wed).
  - Outgate occurred Day 10. Within free time.
  - **Lawful Demurrage: 0 days $\implies \$0.00$.**
  - Reefer power fee: Billed for 10 days. However, for 4 days (Days 5 through 8), the container was at the off-dock CES, where the importer paid the CES facility direct plug-in charges. Terminal cannot double-bill for power when the box is off-dock.
  - Lawful terminal power days: 6 days (Days 1..4 and Days 9..10) @ $95 = \$570.00$.
  - **Lawful Total: $\$570.00$.**
  - **Disputed Overcharge: $\$3,630.00$.**

#### 6. Inland Detention (Per Diem) Audit
- *Carrier Billing:*
  - C-04..C-06, R-02: 4 days outgate $\implies 0$ days charged.
  - C-01..C-03: 4 days outgate $\implies 0$ days charged.
  - C-07, C-08 (2 boxes): 6 days outgate. Free time 5 days $\implies 1$ day past free time @ Tier 1 ($175) $\times 2 = \$350.00$.
  - R-01 (1 box): 6 days outgate. Free time 5 days $\implies 1$ day past free time @ Tier 1 ($175) $\times 1 = \$175.00$.
  - Detention billed: $\$525.00$.
- *Lawful Audit:*
  - Consignee experienced standard warehouse devanning delays. Empty containers returned 1 day beyond free time. No terminal return restrictions were in effect on Day 16.
  - **Lawful Detention: 3 containers $\times 1$ day $\times \$175 = \$525.00$.**
  - **Disputed Overcharge: $\$0.00$.**

### Reconciliation Summary Table

```text
======================================================================================================
                              FINANCIAL AUDIT AND DISPUTE RECONCILIATION
======================================================================================================
CONTAINER ID        TYPE   CARRIER BILLED   LAWFUL AMOUNT   DISPUTED AMOUNT   PRIMARY OSRA/FMC CITATION
------------------------------------------------------------------------------------------------------
C-01                40HC        $1,280.00           $0.00         $1,280.00   CBP VACIS Hold / No Avail
C-02                40HC        $1,280.00           $0.00         $1,280.00   CBP VACIS Hold / No Avail
C-03                40HC        $1,280.00           $0.00         $1,280.00   CBP VACIS Hold / No Avail
C-04                40HC          $540.00           $0.00           $540.00   Gate Closure / TAS Lockout
C-05                40HC          $540.00           $0.00           $540.00   Gate Closure / TAS Lockout
C-06                40HC          $540.00           $0.00           $540.00   Gate Closure / TAS Lockout
C-07                40HC        $1,000.00           $0.00         $1,000.00   Dual-Transaction Lockout
C-08                40HC        $1,000.00           $0.00         $1,000.00   Dual-Transaction Lockout
R-01 (Demurrage)    40RF        $3,250.00           $0.00         $3,250.00   FDA CES Hold / No Avail
R-01 (Power Fee)    40RF          $950.00         $570.00           $380.00   Duplicate Off-Dock Power
R-02 (Demurrage)    40RF        $2,150.00           $0.00         $2,150.00   Gate Closure / Tolled Free
R-02 (Power Fee)    40RF          $760.00         $760.00             $0.00   Valid Utility Service
Detention (3 boxes) MIX           $525.00         $525.00             $0.00   Lawful Consignee Dwell
------------------------------------------------------------------------------------------------------
TOTAL CONSIGNMENT              $14,895.00       $1,855.00        $13,040.00   87.5% DISPUTE RECOVERY
======================================================================================================
```

The enterprise logistics audit team recovers **$13,040.00** of the $14,895.00 invoiced charges (an 87.5% reduction), paying only the lawful power monitoring fees and valid inland equipment detention.

---

## 7. Code, query, or calculation

The following self-contained Python script implements a programmatic demurrage and detention audit engine. It ingests container milestone records, applies progressive stepped tariff tiers, evaluates physical availability exceptions under OSRA 2022 rules, and calculates lawful versus disputed charges.

```python
"""Automated Demurrage and Detention Audit Reconciliation Engine (OSRA 2022)."""
from dataclasses import dataclass
from typing import List, Tuple

@dataclass
class ContainerAudit:
    cid: str
    ctype: str  # 'DRY' or 'REEFER'
    qty: int
    billed_dem: float
    billed_det: float
    billed_pwr: float
    dwell_term: int
    dwell_inland: int
    tolled_term: int
    offdock_ces: int
    osra_certified: bool

def stepped_cost(days: int, tiers: List[Tuple[int, float]]) -> float:
    cost, acc = 0.0, 0
    for max_d, rate in tiers:
        if days <= acc: break
        span = (max_d - acc) if max_d else (days - acc)
        t_days = min(days - acc, span)
        cost += t_days * rate
        acc += t_days
    return cost

def audit(rec: ContainerAudit) -> dict:
    billed = rec.billed_dem + rec.billed_det + rec.billed_pwr
    if not rec.osra_certified:
        return {"id": rec.cid, "lawful": 0.0, "disputed": billed, "status": "VOID_OSRA"}
    free_dem = 4 if rec.ctype == "DRY" else 2
    tiers_dem = [(4, 180.0), (8, 280.0), (None, 420.0)] if rec.ctype == "DRY" else [(3, 350.0), (None, 550.0)]
    tiers_det = [(4, 175.0), (None, 300.0)]
    
    net_term = max(0, rec.dwell_term - rec.tolled_term)
    law_dem = stepped_cost(max(0, net_term - free_dem), tiers_dem) * rec.qty
    law_det = stepped_cost(max(0, rec.dwell_inland - 5), tiers_det) * rec.qty
    pwr_rate = 95.0 if rec.ctype == "REEFER" else 0.0
    law_pwr = max(0, rec.dwell_term - rec.offdock_ces) * pwr_rate * rec.qty
    
    lawful = law_dem + law_det + law_pwr
    return {"id": rec.cid, "lawful": lawful, "disputed": max(0.0, billed - lawful), "status": "AUDITED"}

records = [
    ContainerAudit("C-01..03 (3x40HC)", "DRY",    3, 3840.0,   0.0,   0.0, 11, 4, 7, 0, True),
    ContainerAudit("C-04..06 (3x40HC)", "DRY",    3, 1620.0,   0.0,   0.0,  8, 4, 4, 0, True),
    ContainerAudit("C-07..08 (2x40HC)", "DRY",    2, 2000.0, 350.0,   0.0, 10, 6, 6, 0, True),
    ContainerAudit("R-01     (1x40RF)", "REEFER", 1, 3250.0, 175.0, 950.0, 10, 6, 8, 4, True),
    ContainerAudit("R-02     (1x40RF)", "REEFER", 1, 2150.0,   0.0, 760.0,  8, 4, 6, 0, True),
    ContainerAudit("C-VOID   (1x40HC)", "DRY",    1,  850.0,   0.0,   0.0,  7, 2, 0, 0, False),
]

tot_b = sum(r.billed_dem + r.billed_det + r.billed_pwr for r in records)
tot_l, tot_d = 0.0, 0.0
for r in records:
    res = audit(r)
    tot_l += res["lawful"]
    tot_d += res["disputed"]
    print(f"{res['id']:18} | Lawful: ${res['lawful']:8.2f} | Disputed: ${res['disputed']:8.2f} | Status: {res['status']}")
print("-" * 65)
print(f"Total Billed:   ${tot_b:9.2f}")
print(f"Total Lawful:   ${tot_l:9.2f}")
print(f"Total Disputed: ${tot_d:9.2f} ({(tot_d/tot_b)*100:.1f}% Recovery)")
```

### Expected Output

```text
C-01..03 (3x40HC)  | Lawful: $    0.00 | Disputed: $ 3840.00 | Status: AUDITED
C-04..06 (3x40HC)  | Lawful: $    0.00 | Disputed: $ 1620.00 | Status: AUDITED
C-07..08 (2x40HC)  | Lawful: $  350.00 | Disputed: $ 2000.00 | Status: AUDITED
R-01     (1x40RF)  | Lawful: $  745.00 | Disputed: $ 3630.00 | Status: AUDITED
R-02     (1x40RF)  | Lawful: $  760.00 | Disputed: $ 2150.00 | Status: AUDITED
C-VOID   (1x40HC)  | Lawful: $    0.00 | Disputed: $  850.00 | Status: VOID_OSRA
-----------------------------------------------------------------
Total Billed:   $ 15945.00
Total Lawful:   $  1855.00
Total Disputed: $ 14090.00 (88.4% Recovery)
```

---

## 8. Data, standards, and regulatory context

Demurrage and detention operate at the intersection of international private contract law, national transport statutes, and administrative agency enforcement. The regulatory framework experienced an unprecedented transformation between 2020 and 2026.

```text
========================================================================================================================
                                REGULATORY AND STANDARDS ARCHITECTURE FOR D&D
========================================================================================================================

  [ U.S. STATUTORY BASE ]             [ FMC ADMINISTRATIVE RULES ]              [ DIGITAL EDI STANDARDS ]
  Ocean Shipping Reform Act of 2022   46 CFR § 545.5: Incentive Principle      ANSI X12 / UN/EDIFACT
  46 U.S.C. § 41102(c): Reasonableness 46 CFR Part 541: Invoicing & Privity     EDI 315 / IFTSTA: Ocean Milestones
  46 U.S.C. § 41104(d): Mandatory Data Docket No. 22-05: Direct Billing Only    EDI 214 / IFTMIN: Drayage Status
  46 U.S.C. § 41310: Charge Complaints Docket No. 22-04: Burden of Proof        EDI 310 / INVOIC: Freight Invoice
========================================================================================================================
```

### 1. United States Regulatory Framework: OSRA 2022 and the Shipping Act

The primary regulatory body governing international maritime ocean shipping to and from the United States is the **Federal Maritime Commission (FMC)**. The statutory foundation rests on the Shipping Act of 1984, as amended by the Ocean Shipping Reform Act of 1998 and the **Ocean Shipping Reform Act of 2022 (Public Law 117-146)**.

#### A. 46 U.S.C. § 41102(c) — The "Just and Reasonable" Standard
Section 41102(c) prohibits common carriers, marine terminal operators, and ocean transportation intermediaries (OTIs) from failing to "establish, observe, and enforce just and reasonable regulations and practices relating to or connected with receiving, handling, storing, or delivering property."
- In **FMC Docket No. 20-04 (Interpretive Rule on Demurrage and Detention, 46 CFR § 545.5)**, the Commission codified the **Incentive Principle**:
  - D&D charges are reasonable only insofar as they serve their fundamental purpose as financial incentives to promote freight fluidity.
  - If a shipper, consignee, or motor carrier is prevented from retrieving cargo or returning equipment due to circumstances outside their operational control (e.g., severe port congestion, gate appointment lockouts, unannounced terminal closures, government agency holds, or lack of empty return authorization), the charges no longer incentivize movement. They become punitive revenue-generation mechanisms that violate § 41102(c).

#### B. 46 U.S.C. § 41104(a)(15) and (d) — Mandatory Invoicing Rules
OSRA 2022 enacted rigorous transparency requirements. Under § 41104(d), any invoice assessing demurrage or detention must include 13 accurate, verifiable data elements:
1. Date that container is made physically available.
2. Port of discharge.
3. Container number(s).
4. Earliest return date (for exports).
5. Allowed free time in days.
6. Start date of free time.
7. End date of free time (Last Free Day).
8. Applicable rules tariff number or service contract reference.
9. Applicable daily stepped rate structure.
10. Total amount due.
11. Email address and telephone number for dispute inquiries.
12. Carrier certification stating that the assessed charges comply with FMC rules.
13. Indication of whether the invoice is based on carrier haulage or merchant haulage.

*Statutory Elimination of Obligation:* Under 46 U.S.C. § 41104(f), if an invoice omits any of these required data fields, **the billed party has zero statutory obligation to pay the fee**.

#### C. FMC Final Rule on Billing Requirements (Docket No. 22-05, 46 CFR Part 541)
Issued in February 2024 with full legal effect:
- **Contractual Privity Requirement:** Demurrage and detention invoices may only be issued to:
  1. The person for whose account the ocean transportation is provided (the contracting shipper or consignee named on the bill of lading), or
  2. The "consignee" (the person entitled to delivery of the cargo).
- Ocean carriers and MTOs are explicitly **prohibited from billing drayage motor carriers** for demurrage or detention unless the trucker has entered into a direct, independent bilateral contract with the ocean carrier for equipment provisioning. This ended decades of carriers holding draymen financially hostage by blocking entire motor carrier fleets at terminal gates over disputed BCO demurrage bills.
- **Time Limits:** The billing party must issue a D&D invoice within **30 calendar days** from the date the charges stop accruing. Shippers have at least **30 calendar days** to submit dispute requests, and the carrier must resolve or formally answer the dispute within **30 calendar days**.

#### D. Formal Charge Complaints (46 U.S.C. § 41310)
OSRA 2022 established an expedited, low-barrier administrative dispute mechanism:
- A shipper or motor carrier submits a Charge Complaint through the FMC online portal.
- The Commission reviews the complaint for facial plausibility. If accepted, the FMC issues an **Order to Show Cause** directly to the common carrier.
- **Burden of Proof:** Under 46 U.S.C. § 41104(d), the statutory burden of proof rests entirely on the billing ocean carrier to establish that the charge complies with FMC regulations, was calculated accurately, and was just and reasonable. If the carrier fails to meet this burden, the FMC orders an immediate refund or cancellation of the charge, and may initiate civil penalty proceedings under 46 U.S.C. § 41107 (penalties up to $10,000+ per violation).

### 2. International and Jurisprudential Comparison

Outside the United States, D&D regulation relies primarily on commercial contract law, standard maritime terms (e.g., BIMCO standard contracts), and national competition frameworks:
- **United Kingdom and English Maritime Law:**
  - English common law views demurrage as liquidated damages agreed in advance for breach of the charterparty or transport contract (*The Laura Prima* [1982] 1 Lloyd's Rep 1; *The Eternal Bliss* [2021] UKSC 51).
  - English courts distinguish between legitimate liquidated damages and unenforceable "penalty clauses" under the doctrine established in *Cavendish Square Holding BV v. Talal El Makdessi* [2015] UKSC 67. If a demurrage rate is held to be extravagant, exorbitant, or unconscionable relative to the carrier's legitimate interest in equipment turn, it is legally void.
- **Canada:**
  - The Canadian Transportation Agency (CTA) exercises statutory oversight over rail demurrage under the *Canada Transportation Act* (S.C. 1996, c. 10, § 169.31). The CTA applies a "commercial fairness" standard analogous to the FMC Incentive Principle, disallowing rail storage charges when railway service failures prevent container interchange at port rail ramps in Vancouver and Montreal.
- **Australia:**
  - The Australian Competition and Consumer Commission (ACCC) publishes annual *Container Stevedoring Monitoring Reports* evaluating terminal access charges and vehicle booking system (VBS) pricing. Part X of the *Competition and Consumer Act 2010* grants conditional exemptions to liner shipping conferences, but stevedore gate fees and detention charges face increasing regulatory scrutiny regarding unconscionable conduct.
- **FIATA (International Federation of Freight Forwarders Associations):**
  - Published the *FIATA Best Practice Guide on Demurrage and Detention in Container Shipping*, establishing standard international definitions, advocating for 10-day combined free time on deep-sea lanes, and recommending that customs inspection dwell be excluded from commercial detention calculations.

### 3. Electronic Data Interchange (EDI) and Digital Standards

Programmatic audit and enforcement of D&D require ingestion and parsing of standard EDI and API protocols:

```text
======================================================================================================
                                 STANDARDIZED D&D DATA INTERCHANGE
======================================================================================================
STANDARD / MESSAGE   NAME                           FUNCTION IN D&D AUDIT ENGINE
------------------------------------------------------------------------------------------------------
ANSI X12 EDI 315     Ocean Status Details           Establishes foundational timestamps: 'VD' (discharge),
                                                    'AV' (available), 'OA' (outgate), 'RD' (empty return).
ANSI X12 EDI 214     Transportation Carrier Status  Motor carrier drayage milestones: 'X1' (dock arrival),
                                                    'CD' (cargo delivered), 'E' (empty equipment returned).
ANSI X12 EDI 310     Ocean Freight Invoice          Carries accessorial line items: accessorial code `DEM`
                                                    (Demurrage), `DET` (Detention), daily rate, days.
UN/EDIFACT IFTSTA    Transport Status Message       International equivalent of EDI 315; reports container
                                                    yard custody transfers and customs release events.
UN/EDIFACT INVOIC    Commercial Invoice             International standard for freight and accessorial billing.
DCSA Track & Trace   Digital Container Shipping     RESTful API standardizing 43 physical container events
(v2.2 / v3.0)        Association Interface          with sub-minute ISO 8601 UTC timestamp precision.
======================================================================================================
```

---

## 9. Industry practice and case evidence

### 1. The 2020–2022 Transpacific Congestion Shock

During the peak of the pandemic supply chain disruption, the San Pedro Bay ports of Los Angeles and Long Beach experienced unprecedented vessel queues, with over 100 container vessels waiting at anchor simultaneously in late 2021. 

```text
======================================================================================================
            TRANSPACIFIC SUPPLY CHAIN GRIDLOCK: THE ACCESSORIAL CASCADE (2021-2022)
======================================================================================================
1. RECORD IMPORT SURGE -> Terminals exceed 95% yard capacity; container stacks exceed 6-high.
2. TOS CLOSED BLOCKS  -> Terminals seal entire container stacks; containers physically inaccessible.
3. CHASSIS DEPLETION  -> Street dwell of chassis jumps from 3.5 days to 9.8 days; 0 chassis in port.
4. TAS APPOINTMENT CRUSH -> Truckers log into TAS at midnight; 100% of appointments booked in 45 seconds.
5. DUAL-TX LOCKOUT    -> Carriers block empty returns; truckers unable to drop empties to pull fulls.
6. FINANCIAL DAMAGE   -> Top 9 ocean carriers bill $30B+ in D&D globally; small BCOs forced into insolvency.
======================================================================================================
```

According to the FMC Fact Finding 29 Final Report (*Effects of the COVID-19 Pandemic on the U.S. Ocean Transportation System*, May 2022), D&D collections by the top nine ocean carriers operating in U.S. lanes surged from **$2.1 billion in 2020** to **$6.9 billion in 2021**, and exceeded **$10 billion in 2022**. A major fraction of these charges was assessed on containers that cargo owners were physically unable to move.

### 2. Landmark FMC Enforcement Precedents

The establishment of the FMC's Bureau of Enforcement, Investigations, and Compliance (BEIC) following OSRA 2022 produced major enforcement actions against global liner companies:

#### Case A: FMC v. Ocean Network Express (ONE) — Docket No. 21-17
- *Factual Context:* ONE assessed detention charges on drayage carriers returning empty containers at the Port of Houston during days when the designated terminal gate was closed for business.
- *Legal Holding:* The FMC established that charging detention when the carrier's designated facility is physically closed violates 46 U.S.C. § 41102(c). A carrier cannot penalize an equipment user for failing to perform an impossible act. ONE settled the civil penalty proceeding, paid an administrative fine, and agreed to credit or refund all improper fees.

#### Case B: FMC Investigation of Wan Hai Lines — Docket No. 21-16
- *Factual Context:* Wan Hai Lines invoiced motor carriers per diem detention charges across 21 shipments discharged at the ports of Los Angeles and Long Beach. The drayage truckers documented that they attempted to schedule empty return appointments through the terminal appointment portal on 13 separate occasions, but the terminal displayed zero available return slots.
- *Carrier Defense:* Wan Hai argued that the motor carriers could have returned the containers to alternative yards or waited for appointments to open.
- *Holding:* The FMC Administrative Law Judge (ALJ) ruled that an ocean carrier's failure to provide a realistic, bookable return location renders detention charges unreasonable per se. Wan Hai settled the enforcement proceeding for **$850,000 in civil penalties** and implemented comprehensive operational tolling rules.

#### Case C: Bed Bath & Beyond v. Orient Overseas Container Line (OOCL) and Yang Ming
- *Factual Context:* In 2023, the liquidating estate of Bed Bath & Beyond filed formal reparations complaints under 46 U.S.C. § 41301 against major ocean carriers, seeking over $31 million in damages.
- *Allegations:* The shipper demonstrated that ocean carriers systematically billed demurrage on containers placed in terminal "closed areas" where terminal cranes were prohibited from digging out boxes, and billed detention when port terminals refused empty container ingates. This case established the standard for historical forensic audits of enterprise freight billing.

#### Case D: Mediterranean Shipping Company (MSC) / SOFIGEL Case
- *Factual Context:* A French food importer shipped frozen goods via MSC to the Port of Le Havre. The shipment was placed under an intensive veterinary health inspection by French customs authorities (Border Inspection Post / BIP), delaying container outgate by 18 days.
- *Holding:* MSC sought €24,000 in demurrage. The commercial court held that while the carrier was entitled to compensation for actual shore power utility consumption, demurrage penalty rates could not be applied during statutory sovereign health inspections where the importer had committed no procedural fault.

### 3. Enterprise BCO Procurement and Audit Strategies

Sophisticated enterprise importers (e.g., Walmart, Target, Home Depot, Nike) manage D&D through procurement contracts and automated analytics:

1. **Contractual Free Time Optimization:**
   - Standard carrier tariffs grant only 3 to 4 days of demurrage free time and 5 days of detention free time.
   - Enterprise volume contracts negotiate **extended free time tiers**:
     - 10 to 14 days of demurrage free time at primary import gateway ports.
     - 14 to 21 days of detention free time for inland distribution centers.
     - "Toll-the-clock" contractual clauses specifying that any day with a CBP hold, gate closure, or TAS appointment unavailability automatically stops the free time clock.
2. **Automated Freight Audit and Payment (FAP) Firewalls:**
   - Enterprises deploy automated audit engines (e.g., Trax, Blume Global, project44, Envecon) directly between carrier EDI 310 invoicing feeds and enterprise ERP accounts payable modules.
   - Any D&D invoice lacking the 13 mandatory OSRA 2022 fields is automatically rejected by the system without human intervention.
   - The engine automatically cross-references the carrier's invoiced dwell against independently scraped TOS availability timestamps and terminal gate closure bulletins. Disputed line items are placed into escrow, and formal dispute packages are programmatically generated and emailed to the carrier's FMC dispute point of contact.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Root Physical / Legal Mechanism | Consequence | Analytical & Operational Mitigation |
|---|---|---|---|---|
| **1. Billing Before Physical Availability** | Invoiced demurrage start date matches vessel discharge timestamp (`VD`). | Carrier billing engine automates clock start upon discharge without verifying terminal stack status or physical accessibility (`AV`). | Shipper pays demurrage on containers buried under 5 tiers of steel in a closed yard stack. | Programmatically join EDI 315 `VD` against `AV` timestamps; reject invoices where billing start precedes `AV`. |
| **2. Invoicing Motor Carriers (Privity Breach)** | Ocean carrier bills drayage trucker directly for container demurrage. | Ocean carrier attempts to enforce historical tariff rules, bypassing FMC Docket 22-05 (46 CFR Part 541). | Trucker credit lines frozen; illegal liability transfer; terminal gate lockouts. | Cite 46 CFR Part 541; mandate that carriers bill only the BCO contracting party; file FMC Charge Complaint if trucker is coerced. |
| **3. Failure to Toll During Customs Holds** | Demurrage accrues while container status shows active CBP 1H, 2H, or FDA hold. | Carrier claims customs delay is a "shipper compliance issue," refusing to toll free time during non-intrusive scanning or CES drayage. | Tens of thousands of dollars in accessorial debt on containers under armed federal custody. | Pull CBP ACE entry logs showing exact hold release timestamps; submit dispute under FMC Incentive Principle doctrine. |
| **4. Dual-Transaction Appointment Lockout** | Trucker logs show 10+ consecutive TAS appointment rejections citing "Empty Required." | Terminal operating policy bans single-leg import pickups during high yard congestion, rationing gate capacity. | Shipper cannot retrieve cargo despite ready trucks; demurrage escalates to Tier 3 ($400+/day). | Script automated daily screenshot captures of TAS portal displaying appointment denials; submit as evidentiary proof of unavailability. |
| **5. Stepped Tariff Tier Miscalculation** | Invoice applies Tier 2 or Tier 3 rates immediately upon free time expiration. | Carrier billing software fails to reset day counters or misapplies calendar-day increments to business-day free time. | Systematic overcharging of 40% to 100% on early dwell days. | Implement algorithmic tier calculation; recalculate daily marginal cost based on published service contract schedules. |
| **6. Weekend / Holiday Demurrage Billing** | Charges assessed on Saturday or Sunday following a Friday LFD under a business-day tariff. | Carrier tariff specifies business-day free time, but switches to calendar days immediately upon expiration without contractual basis. | Improper weekend accessorial charges assessed when terminal gates are closed. | Audit tariff language; verify terminal operating hours; dispute charges assessed on dates when terminal gate was closed. |
| **7. Empty Return Turn-Away and Redirection** | Drayman arrives at terminal with empty container and is turned away by gate security. | Terminal empty yard reaches maximum capacity; carrier fails to arrange alternative empty depot receiving capacity. | Consignee incurs mounting per diem detention while equipment is stranded on chassis. | Require drayman to obtain a stamped "Turn-Away Slip" or gate rejection log; submit to carrier to freeze detention clock instantly. |
| **8. Omission of Mandatory OSRA Invoice Elements** | D&D invoice lacks FMC dispute email, rate tier breakdown, or certification statement. | Carrier billing systems fail to update electronic invoice templates to comply with 46 U.S.C. § 41104(d). | Invoice is legally void; carrier has no statutory right to enforce payment. | Execute automated regex validation against 13 OSRA fields; issue immediate statutory rejection notice under 46 U.S.C. § 41104(f). |
| **9. Chassis Split Cost Leakage** | Drayage trucker bills $150 "chassis split" surcharge plus 3 days of chassis per diem. | Terminal container yard operates as a grounded facility without mounted chassis; IEP pool exhausted on-dock. | Unbudgeted inland accessorial costs erode landed gross margins. | Analyze lane historical chassis availability; contract with motor carriers operating private chassis or negotiate carrier haulage. |
| **10. Discrepant EDI 315 Milestones** | Carrier invoice dates contradict carrier's own EDI 315 tracking milestone data. | Asynchronous data pipelines between carrier operational dispatch systems and centralized corporate billing ERPs. | Fraudulent or erroneous accessorial invoices accepted by accounts payable. | Build automated milestone diff engine; flag any invoice where billed dwell differs from EDI `AV` to `OA` duration by $>0$ hours. |
| **11. Double-Billing Reefer Utility Services** | Carrier bills terminal daily reefer monitoring fee while container is at off-dock CES facility. | Marine terminal automated billing continues recurring daily plug charge without checking gate-out to CES. | Shipper pays terminal $95/day and CES facility $120/day for identical electrical power monitoring. | Cross-reference CES gate-in/gate-out logs against terminal power fee invoices; claw back overlapping service days. |
| **12. Dispute Filing Window Expiration** | Carrier rejects dispute citing expiration of 30-day tariff dispute window. | Importer routes D&D invoices through slow manual accounts payable approvals, missing statutory dispute deadlines. | Complete forfeiture of legal right to recover improper D&D assessments. | Implement 72-hour automated invoice ingestion SLA; trigger automated dispute generation within 5 business days of receipt. |

---

## 11. Assumptions and limits

| Analytical Assumption | Why It Is Needed | Boundary Condition Where It Fails | Method to Test / Validate Assumption |
|---|---|---|---|
| **1. Milestone Timestamp Integrity** | Assumes carrier and terminal EDI 315 / 214 event logs represent true physical timestamps. | Carrier IT systems experience batch processing lag, backdating `AV` status to vessel discharge date. | Compare EDI 315 transmission header timestamps against body event timestamps; cross-reference with GPS gate logs. |
| **2. Legal Enforceability of OSRA 2022** | Assumes 46 U.S.C. §§ 41102(c) and 41104 apply universally to all maritime shipments. | Fails on purely domestic inland moves, shipments under foreign sovereign jurisdictions, or non-U.S. discharge ports. | Verify that bill of lading designates a U.S. port of discharge or U.S. inland intermodal point under FMC jurisdiction. |
| **3. Strict Independence of Demurrage and Detention Clocks** | Assumes demurrage terminates cleanly upon outgate (`OA`) and detention begins. | Fails in combined demurrage/detention free time regimes (Europe/Asia) where a single merged clock governs total dwell. | Inspect carrier service contract to verify whether lane operates under split North American rules or merged global rules. |
| **4. Motor Carrier Diligence in TAS Booking** | Assumes motor carrier made genuine, documented attempts to secure gate appointments before LFD. | Motor carrier failed to log into TAS, booked appointments late, or missed scheduled appointment windows ("no-shows"). | Demand motor carrier TAS system audit logs showing specific login timestamps, query filters, and system rejection notices. |
| **5. Exclusivity of Merchant Haulage Risk** | Assumes BCO bears full responsibility for inland transit and equipment turn under merchant haulage. | Carrier haulage moves where the ocean carrier arranges the trucker; carrier cannot assess detention for trucker delays. | Examine ocean bill of lading box: verify whether freight is booked as Merchant Haulage (MH) or Carrier Haulage (CH). |
| **6. Additivity of Piecewise Stepped Tariffs** | Assumes stepped tiers operate strictly on marginal elapsed days within each defined threshold. | Carrier tariff contains "retroactive step" clauses (e.g., if dwell exceeds 10 days, Tier 3 rate applies retroactively to Day 1). | Audit published carrier ocean tariff rules; test invoice math for retroactive application vs. marginal step application. |
| **7. Physical Availability Equivalence to Berth Productivity** | Assumes rapid vessel discharge guarantees prompt container staging in accessible yard blocks. | Mega-vessel discharge (15,000+ TEU) overwhelms yard equipment; containers sit in inaccessible stacks for 5+ days post-discharge. | Track terminal average dwell and yard density metrics; verify physical container location code within terminal yard. |
| **8. Separability of Terminal Storage and Demurrage** | Assumes carrier demurrage and MTO terminal storage follow identical clock tolling rules. | MTO is an independent private terminal with a separate public tariff not bound by ocean carrier service contract concessions. | Cross-examine ocean carrier service contract against independent MTO marine terminal schedule filed with the FMC. |
| **9. Uniformity of Chassis Roadability** | Assumes chassis retrieved from IEP pools are mechanically roadable and pass FMCSA pre-trip inspection. | Chassis pool contains high percentage of bad-order chassis (flat tires, broken twistlocks, failed brakes), causing dray delays. | Review Driver Vehicle Inspection Reports (DVIR); track gate turn-around times and bad-order chassis repair logs. |
| **10. Solvency and Standing of Contracting Parties** | Assumes disputed funds can be recovered from the ocean carrier following formal dispute adjudication. | NVOCC or freight forwarder bankruptcy; carrier administrative delays stretching dispute resolution beyond 12 months. | Escrow disputed funds; utilize FMC CADRS mediation; file formal Charge Complaints under 46 U.S.C. § 41310. |

---

## 12. Exercises

### Exercise 1: Demurrage vs. Detention Boundary and Classification (Recall)
- **Deliverable:** A technical briefing document (maximum 2 pages) classifying the operational, legal, and financial differences between carrier demurrage, terminal storage, container detention, and chassis per diem.
- **Acceptance Criteria:**
  1. Correctly maps the physical custody transitions (Discharge $\to$ Available $\to$ Outgate $\to$ Unload $\to$ Ingate) to each fee.
  2. Delineates the jurisdictional differences between the North American split regime and the European/Asian combined free-time regime.
  3. Identifies the distinct billing entities (Carrier, MTO, IEP) and applicable regulatory oversight bodies.
- **Estimated Time:** 45 minutes.

### Exercise 2: Stepped Tariff Dwell Modeling (Application)
- **Deliverable:** A mathematical model and manual calculation spreadsheet computing cumulative accessorial charges for a consignment of 6 standard 40' high-cube containers and 2 refrigerated containers experiencing varying dwell durations.
- **Acceptance Criteria:**
  1. Correctly applies a 3-tier stepped tariff schedule ($0/$175/$300/$450 for dry; $0/$350/$600 for reefer).
  2. Accurately handles business-day free time vs. calendar-day stepped accrual.
  3. Includes daily reefer monitoring utility fees ($110/day) applied from discharge date.
  4. Produces zero arithmetic errors in marginal versus cumulative cost breakdowns.
- **Estimated Time:** 60 minutes.

### Exercise 3: Root-Cause Dwell Decomposition (Critique)
- **Deliverable:** A diagnostic root-cause analysis report evaluating a real-world terminal dwell crisis involving 25 containers delayed across customs holds, gate computer crashes, and dual-transaction lockouts.
- **Acceptance Criteria:**
  1. Deconstructs total container-days into four distinct buckets: administrative delay, physical yard congestion, carrier operational restrictions, and shipper delay.
  2. Applies the FMC Incentive Principle (46 CFR § 545.5) to categorize which container-days are legally chargeable versus non-chargeable.
  3. Recommends specific operational mitigations for each identified failure mode.
- **Estimated Time:** 60 minutes.

### Exercise 4: OSRA 2022 Statutory Invoice Audit (Audit)
- **Deliverable:** A formal audit memorandum reviewing a sample ocean carrier demurrage invoice against the statutory requirements of 46 U.S.C. § 41104(d).
- **Acceptance Criteria:**
  1. Checks all 13 mandatory statutory data elements.
  2. Identifies at least 3 missing or defective statutory fields (e.g., lack of availability date, omitted certification statement, incorrect dispute contact).
  3. Formulates the precise statutory legal consequence under 46 U.S.C. § 41104(f).
- **Estimated Time:** 45 minutes.

### Exercise 5: Programmatic Reconciliation Engine Extension (Design & Code)
- **Deliverable:** An extended Python script modifying the code in Section 7 to handle dynamic calendar-day vs. business-day conversions and generate automated dispute notification text.
- **Acceptance Criteria:**
  1. Implements a date-aware calendar utility skipping weekends and official U.S. federal/longshore holidays for free-time calculation.
  2. Automatically formats a formal legal dispute letter referencing specific B/L numbers, container IDs, and FMC statutory citations.
  3. Executes cleanly without external dependencies beyond the Python standard library.
- **Estimated Time:** 90 minutes.

### Exercise 6: Terminal Appointment and Dual-Transaction Strategy (Design)
- **Deliverable:** An operational standard operating procedure (SOP) for a drayage dispatch operation navigating terminal appointment systems (TAS) and mandatory dual-transaction constraints at a congested container port.
- **Acceptance Criteria:**
  1. Defines an automated appointment booking protocol (booking windows, bot usage policies, cancellation rules).
  2. Details an evidentiary logging procedure (screenshot archives, timestamped API logs) to prove lack of appointment availability.
  3. Establishes a matchback optimization workflow pairing inbound import fulls with outbound export loads to satisfy dual-transaction mandates.
- **Estimated Time:** 60 minutes.

### Exercise 7: Service Contract Free Time Negotiation Framework (Design)
- **Deliverable:** A commercial negotiation term sheet and playbook for an enterprise shipper importing 12,000 TEUs annually, focusing on D&D mitigation.
- **Acceptance Criteria:**
  1. Formulates tiered free-time clauses (demurrage, detention, combined) across primary and secondary discharge ports.
  2. Incorporates contractual "clock-stopping" language for government inspections, terminal IT failures, and equipment shortages.
  3. Establishes a bilateral dispute resolution timeline and liquidated damage caps.
- **Estimated Time:** 60 minutes.

---

## 13. Self-check answers

### Self-Check for Exercise 2: Stepped Tariff Dwell Modeling
*Problem Statement:* A shipment of four 40' dry containers arrives at a port with 4 business days of free time. Containers are discharged on Monday, Day 1, and designated Available (`AV`) on Tuesday, Day 2.
Tariff:
- Free Time: 4 Business Days.
- Tier 1 (Days 1–4 past free time): $160/day (calendar days).
- Tier 2 (Days 5–8 past free time): $260/day (calendar days).
- Tier 3 (Days 9+ past free time): $400/day (calendar days).
Containers gate out on Day 16 (Wednesday). There are no customs holds or terminal closures.

*Calculation Step-by-Step:*
1. **Clock Start:** Available Day 2 (Tuesday). Free time starts Day 3 (Wednesday).
2. **Free Time Window:** 4 business days: Day 3 (Wed), Day 4 (Thu), Day 5 (Fri), Day 8 (Mon) [skipping weekend Days 6 and 7].
   - **Last Free Day (LFD):** Day 8 (Monday).
3. **Chargeable Dwell Days:** Containers gate out on Day 16.
   - Chargeable days commence Day 9 (Tuesday) through Day 16 (Wednesday).
   - Total chargeable days = $16 - 8 = 8$ calendar days.
4. **Applying Stepped Tiers:**
   - Total chargeable days = 8 days.
   - Tier 1: Covers first 4 days past free time (Days 9, 10, 11, 12) $\implies 4 \text{ days} \times \$160 = \$640$.
   - Tier 2: Covers next 4 days past free time (Days 13, 14, 15, 16) $\implies 4 \text{ days} \times \$260 = \$1,040$.
   - Tier 3: 0 days.
   - Total Demurrage per Container = $\$640 + \$1,040 = \$1,680$.
5. **Consignment Total:**
   - 4 containers $\times \$1,680 = \mathbf{\$6,720.00}$.

### Self-Check for Exercise 4: OSRA 2022 Statutory Invoice Audit Rubric
A passing evaluation of an ocean carrier invoice under 46 U.S.C. § 41104(d) must identify the following components:
1. **Field Completeness Audit:**
   - Did the auditor examine all 13 statutory elements? (Full marks require explicit verification of: container number, B/L number, port of discharge, vessel/voyage, availability date, free time start/end dates, allowed free time integer, billing start/end dates, stepped rate schedule, total due, dispute email/phone, FMC compliance certification, haulage type).
2. **Defect Identification:**
   - If the carrier provides a general corporate contact number instead of an explicit, designated D&D dispute contact, it is a statutory defect under § 41104(d)(2)(K).
   - If the invoice omits the exact date when the container was made physically available (`AV`), citing only the vessel discharge date (`VD`), it fails § 41104(d)(2)(D).
   - If the mandatory certification statement ("the charges comply with any rules or regulations promulgated by the FMC under 46 U.S.C. § 41104") is missing, it fails § 41104(d)(2)(L).
3. **Legal Conclusion:**
   - The evaluation must conclude under 46 U.S.C. § 41104(f) that the invoice is legally invalid and the billed party has no statutory obligation to pay until a fully compliant invoice is reissued.

---

## 14. Interview questions

### Question 1: Operational vs. Legal Definitions of D&D
**Prompt:** "Walk me through the exact operational and legal boundary between demurrage, detention, and terminal storage in a U.S. port versus a European port."
- **Strong Answer:** Clearly articulates that in U.S. ports (split regime), demurrage applies inside the terminal gate for container equipment dwell, terminal storage applies for land occupancy (often billed directly by the MTO), and detention (per diem) applies outside the gate from outgate to empty return. Explains that European/Asian ports frequently utilize a combined demurrage/detention regime where a single free-time clock runs from discharge to empty return regardless of gate passage, while quay storage is billed separately by terminal operators.
- **Weak Answer:** Confuses demurrage and detention; claims demurrage is for truckers and detention is for ships; unaware of the European combined free-time model.

### Question 2: The FMC Incentive Principle
**Prompt:** "What is the FMC's 'Incentive Principle' established in Docket No. 20-04, and how does it apply when a container is held for an intensive CBP customs inspection?"
- **Strong Answer:** Explains that under 46 CFR § 545.5 and 46 U.S.C. § 41102(c), D&D charges are legally permissible only when they act as an incentive to promote freight fluidity. If a container is under an active CBP or FDA hold, or if terminal gates are shut or TAS appointment slots are zero, the cargo owner cannot legally or physically move the container. Assessing D&D under these conditions fails to provide a behavioral incentive, converting the charge into an unreasonable practice that violates federal law.
- **Weak Answer:** States that customs holds are the shipper's fault so the shipper must always pay the carrier; unaware of FMC Docket 20-04 or the legal definition of physical availability.

### Question 3: OSRA 2022 Invoice Deficiencies
**Prompt:** "If an ocean carrier sends you a demurrage invoice for $45,000 that lists the correct B/L, container numbers, and stepped rates, but fails to list the container physical availability date and omits the FMC compliance certification statement, what is your legal and operational response?"
- **Strong Answer:** Cites 46 U.S.C. § 41104(d) and (f). Explains that under federal statute, the inclusion of all 13 mandatory invoice elements is a statutory prerequisite for payment. Because the invoice omits the physical availability date and the compliance certification, the billed party is under no statutory obligation to pay the fee. The operational response is to issue an immediate formal rejection citing § 41104(f), placing the charge into legal dispute, and demanding a compliant reissue while auditing the true availability date against EDI 315 logs.
- **Weak Answer:** Advises paying the invoice immediately to avoid credit holds; suggests calling the sales rep to ask for a small courtesy discount.

### Question 4: The Economics of Stepped Tariffs
**Prompt:** "Why do ocean carriers structure D&D tariffs as non-linear stepped functions rather than flat daily rates, and how does this affect supply chain safety stock and dwell modeling?"
- **Strong Answer:** Explains that stepped tariffs are mathematically structured as convex penalty functions to impose escalating marginal costs on chronic dwell, forcing cargo owners to prioritize evacuating aging containers. In supply chain modeling, this creates extreme financial tail risk: lead time delays do not generate linear cost increases, but exponential cost spikes. Consequently, safety stock models must account for the high financial penalty of port dwell variability by buffering inventory upstream or securing contractual free-time extensions.
- **Weak Answer:** Believes stepped tariffs are designed to give shippers a discount the longer they stay; fails to connect tariff non-linearity to supply chain risk.

### Question 5: Direct Billing Privity (FMC Docket 22-05)
**Prompt:** "Historically, ocean carriers frequently billed drayage motor carriers for ocean demurrage and placed gate locks on entire trucking fleets when bills were disputed. What did the FMC's Final Rule on Demurrage and Detention Billing Requirements change?"
- **Strong Answer:** References FMC Docket No. 22-05 (46 CFR Part 541). Explains that the rule established strict contractual privity: carriers and MTOs may only bill the party with whom they have a direct contractual relationship (the BCO or consignee named on the shipping contract). Carriers are prohibited from billing drayage truckers who are merely performing third-party haulage, eliminating the coercive practice of locking out motor carrier fleets over disputed BCO accessorial invoices.
- **Weak Answer:** Assumes truckers are still liable for all port demurrage; unfamiliar with 46 CFR Part 541.

### Question 6: Mitigating Dual-Transaction Lockouts
**Prompt:** "Your drayage dispatch team reports that they cannot pull 15 import containers before their Last Free Day because the marine terminal requires a dual transaction, and all empty return slots are blocked. How do you resolve this operationally and financially?"
- **Strong Answer:** Operationally: Direct dispatch to document the lockout with timestamped video/screenshots of the TAS portal showing zero appointment availability and terminal bulletins announcing empty return restrictions. Request motor carrier dispatch to log attempts every morning and afternoon. Financially: Submit an immediate advance notice of operational unavailability to the ocean carrier, demanding a toll on the LFD clock. If the carrier assesses demurrage, submit a formal dispute backed by the evidentiary dossier under 46 U.S.C. § 41102(c).
- **Weak Answer:** Tells dispatch to keep trying and pay the demurrage when billed; does not collect contemporaneous evidentiary records.

### Question 7: Chassis Dislocation and Split Economics
**Prompt:** "Explain what a 'chassis split' is, why it occurs predominantly in the United States, and how it impacts landed freight costs."
- **Strong Answer:** Explains that in the U.S., unlike Europe or Asia where containers are moved on carrier-supplied trailers or integrated equipment, ocean carriers divested their chassis fleets between 2009 and 2014 to independent IEPs. Containers at marine terminals are stacked grounded on the pavement. If a terminal has no chassis pool on-dock, the trucker must bobtail to a separate off-dock chassis depot, hook a chassis, and drive to the port. This creates additional drayage transit time, driver wait time, chassis split surcharges ($100–$200), and delays that push containers past their LFD.
- **Weak Answer:** Thinks a chassis split means physically cutting a chassis in half; does not understand the historical U.S. chassis divestment model.

### Question 8: Integrating EDI 315 in Freight Audit
**Prompt:** "Which specific ANSI X12 EDI 315 status codes are critical for auditing a demurrage invoice, and what logic would you write to detect an improper billing clock?"
- **Strong Answer:** Identifies `VD` (Vessel Discharged), `AV` (Available for Delivery), `OA` (Outgate Loaded), and `RD` (Empty Returned). Explains the audit logic: (1) Free time clock start date must equal the business day following `AV`, not `VD`. (2) Any days between `AV` and `OA` with active customs hold codes (e.g., `1H`, `2H`) or documented terminal closures must be subtracted from total dwell. (3) If Net Chargeable Days $= \max(0, \text{Dwell} - \text{Tolled Days} - \text{Free Time}) \le 0$, lawful demurrage is $0.
- **Weak Answer:** Mentions EDI 214 only; cannot name specific 315 event codes or articulate the mathematical difference between discharge and availability.

### Question 9: Combined Free Time vs. Split Free Time in Global Procurement
**Prompt:** "You are negotiating an ocean contract for 5,000 TEUs from Shanghai to Rotterdam and 5,000 TEUs from Shanghai to Los Angeles. How do you structure your D&D contract terms differently for each lane?"
- **Strong Answer:** For Rotterdam: Negotiate a single **Combined Demurrage & Detention** bucket of 14 calendar days, allowing maximum flexibility for barge or rail transit into inland Europe without triggering separate equipment detention. Ensure clarity on whether terminal quay storage is paid directly to the port or through the carrier. For Los Angeles: Negotiate separate split buckets—at least 7 to 10 working days of demurrage (to protect against gate congestion and railhead delays) and 10 to 14 calendar days of equipment detention (to allow inland DC devanning and empty return). Include explicit contractual clauses incorporating OSRA 2022 tolling standards for customs and TAS appointment shortages.
- **Weak Answer:** Requests 10 days of flat free time for both lanes without recognizing the structural difference between combined European and split North American regimes.

---

## 15. Key terms

1. **Demurrage:** Financial assessment for container equipment and land occupancy inside a terminal beyond free time.
2. **Detention (Per Diem):** Financial assessment for holding container equipment outside the terminal beyond free time.
3. **Free Time:** Grace period during which cargo or equipment may dwell without incurring accessorial charges.
4. **Last Free Day (LFD):** The final date of the free-time window before daily accessorial charges commence.
5. **Physical Availability:** Operational state where cargo is unstacked, customs-cleared, and accessible for immediate drayage.
6. **Constructive Placement:** Carrier notification of cargo arrival that historically triggered clocks prior to physical availability reforms.
7. **Incentive Principle:** FMC doctrine requiring that D&D charges serve solely to incentivize freight fluidity under 46 CFR § 545.5.
8. **OSRA 2022:** Ocean Shipping Reform Act of 2022, modernizing U.S. maritime regulation and invoice requirements.
9. **46 U.S.C. § 41102(c):** Federal statute prohibiting unjust and unreasonable practices regarding cargo handling and delivery.
10. **46 U.S.C. § 41104(d):** Statutory mandate listing the 13 required data fields on all demurrage and detention invoices.
11. **Billing Privity:** FMC rule (46 CFR Part 541) restricting D&D billing strictly to direct contractual counterparties.
12. **Stepped Tariff:** Piecewise non-linear pricing schedule where daily accessorial rates escalate across dwell brackets.
13. **Dual Transaction:** Terminal gate policy requiring an empty container ingate to authorize an import full outgate.
14. **Chassis Split:** Secondary drayage leg required to pick up an intermodal chassis at an off-dock facility.
15. **Intermodal Equipment Provider (IEP):** Commercial entity owning and leasing roadable chassis fleets to motor carriers.
16. **Terminal Appointment System (TAS):** Mandatory digital platform used by marine terminals to schedule gate transactions.
17. **Empty Return Restriction:** Terminal or carrier directive refusing or redirecting empty container returns.
18. **Centralized Examination Station (CES):** Customs-bonded off-dock facility for physical intensive cargo inspections.
19. **Non-Intrusive Inspection (NII):** Radiographic imaging (VACIS) of shipping containers conducted without devanning cargo.
20. **EDI 315:** Ocean Status Details transaction set communicating container custody milestones (`VD`, `AV`, `OA`, `RD`).
21. **EDI 214:** Transportation Carrier Status message communicating motor carrier drayage and delivery milestones.
22. **EDI 310:** Freight Invoice transaction set carrying freight charges and accessorial line items (`DEM`, `DET`).
23. **Merchant Haulage:** Routing where the cargo owner contracts inland drayage directly with motor carriers.
24. **Carrier Haulage:** Routing where the ocean carrier manages through intermodal transport under a through B/L.
25. **Combined Free Time:** Unified free-time window common in Europe/Asia covering both inside-terminal and outside-terminal dwell.
26. **Terminal Storage:** MTO fee for real estate occupancy in the container yard, distinct from carrier equipment demurrage.
27. **Reefer Monitoring Fee:** Daily utility surcharge for electrical power connection and physical inspection of refrigerated boxes.
28. **Charge Complaint:** Formal dispute mechanism under 46 U.S.C. § 41310 enabling shippers to challenge unlawful charges at the FMC.
29. **Matchback:** Strategic pairing of an import container discharge with an export container reload to minimize empty miles.
30. **Tolling:** Operational or legal pausing of the free-time clock due to external disruptions, closures, or holds.
31. **Street Turn:** The direct reloading of an empty import container for export carriage without intervening depot return.
32. **Bad-Order Chassis:** An intermodal chassis flagged for mechanical defects (lights, brakes, tires) failing FMCSA roadability.

---

## 16. Sources

- **Federal Maritime Commission (2020).** *Interpretive Rule on Demurrage and Detention Under the Shipping Act.* 46 CFR Part 545, Docket No. 20-04. Federal Register, Vol. 85, No. 96, pp. 29638–29666.
- **Federal Maritime Commission (2024).** *Demurrage and Detention Billing Requirements.* Final Rule, 46 CFR Part 541, Docket No. 22-05. Federal Register, Vol. 89, No. 38, pp. 14330–14365.
- **United States Congress (2022).** *Ocean Shipping Reform Act of 2022 (OSRA 2022).* Public Law 117-146, 136 Stat. 1272; codified across 46 U.S.C. §§ 41101–41310.
- **Federal Maritime Commission (2022).** *Fact Finding Investigation 29: Final Report on the Effects of the COVID-19 Pandemic on the U.S. Ocean Transportation System.* Washington, D.C.: FMC Bureau of Enforcement.
- **FIATA (2021).** *Best Practice Guide on Demurrage and Detention in Container Shipping.* Geneva: International Federation of Freight Forwarders Associations.
- **Digital Container Shipping Association (DCSA) (2023).** *DCSA Standard for Track & Trace Interface (v3.0.0).* Amsterdam: DCSA Standards Committee.
- **Notteboom, T., Pallis, A., & Rodrigue, J.-P. (2022).** *Port Economics, Management and Policy.* New York: Routledge. (Chapters 5 and 6: Terminal Operations, Equipment Velocity, and Accessorial Pricing).
- **World Shipping Council (WSC) (2023).** *The Ocean Carrier Role in Intermodal Container Velocity and Supply Chain Fluidity.* Washington, D.C.: WSC Industry Whitepaper.

---

## 17. Links

- **Survey Module Map:** [`../../../course/36_global_freight_ports_trade_policy.md`](../../../course/36_global_freight_ports_trade_policy.md)
- **Module Index:** [`_index.md`](_index.md)
- **Sibling Topics:**
  - [`01_incoterms-documents.md`](01_incoterms-documents.md) — Incoterms and Trade Documents
  - [`02_landed-cost.md`](02_landed-cost.md) — Total Landed Cost
  - [`03_classification-valuation-origin.md`](03_classification-valuation-origin.md) — Classification, Valuation, and Origin
  - [`04_modal-choice-emissions.md`](04_modal-choice-emissions.md) — Modal Choice and Emissions
  - [`05_port-operations-congestion.md`](05_port-operations-congestion.md) — Port Operations and Congestion
  - [`07_freight-procurement-rate-risk.md`](07_freight-procurement-rate-risk.md) — Freight Procurement and Rate Risk
  - [`08_chokepoints-resilience.md`](08_chokepoints-resilience.md) — Chokepoints and Network Resilience
  - [`09_trade-policy-sanctions-carbon.md`](09_trade-policy-sanctions-carbon.md) — Trade Policy, Sanctions, and Carbon Rules
  - [`10_customs-broker-management.md`](10_customs-broker-management.md) — Customs and Broker Management
- **Prerequisite Modules:**
  - Module 11: Safety Stock and Service Levels
  - Module 14: Network Design and Facility Location
  - Module 19: International Logistics and Customs Operations
