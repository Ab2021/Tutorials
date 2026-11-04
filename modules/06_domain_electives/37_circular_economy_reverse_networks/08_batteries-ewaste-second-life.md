# Batteries, E-Waste, and Second Life

> Module 37 — Circular Economy and Reverse Logistics Networks
> Topic 08 of 10
> Estimated study time: 4.0 hours
> Prerequisites: Module 10 (Inventory Optimization), Module 11 (Network Design Optimization), Module 12 (Transportation & Routing), Module 37 Topic 01 (Reverse Flows and Disposition Hierarchy), Topic 03 (Grading and Recovery Yield)
> Version 2026.5

---

## 1. Why this topic exists

High-voltage lithium-ion battery packs and complex electronic waste (e-waste) represent the single largest economic, safety, and regulatory liability in modern reverse supply chains. In an internal combustion engine (ICE) vehicle, the powertrain depreciates gradually, contains non-hazardous solid metals, and can be salvaged through well-established automotive dismantling yards with minimal specialized equipment. In an electric vehicle (EV), the traction battery pack accounts for 30% to 45% of the total bill-of-materials (BOM) cost, weighs between 300 kg and 700 kg, operates at lethal direct-current potentials (400 V to 800 V), and stores gigajoules of electrochemical energy capable of uncontained thermal runaway if damaged, mechanically breached, or thermally abused.

A retirement cliff is rapidly approaching global supply chains. As the first and second waves of mass-market electric vehicles reach their initial design lifespans (typically 8 to 12 years or 160,000 to 250,000 km), millions of high-capacity battery packs are exiting mobile service annually. When an EV battery degrades to roughly 70% to 80% of its initial certified capacity—termed its State of Health ($\text{SoH} = 0.70\text{--}0.80$)—it is typically deemed retired from automotive use due to restricted driving range, power fade during peak acceleration, and sluggish regenerative braking. However, a retired 75 kWh automotive pack at 75% SoH still retains over 56 kWh of usable, high-grade electrochemical storage. Discarding or immediately shredding such assets destroys millions of dollars in residual embodied energy, carbon investment, and refined precursor materials.

Conversely, routing end-of-first-life batteries into secondary stationary energy storage systems (BESS)—such as commercial peak shaving, renewable solar/wind buffer storage, and telecom backup—creates immense economic value while amortizing initial manufacturing emissions across an extended operational horizon. Yet realizing this second-life potential presents severe logistical, operational, and physical bottlenecks:

1. **Extreme Degradation Heterogeneity:** Identical battery packs manufactured in the same batch diverge radically across their first life. A battery operated in a hot climate under continuous ultra-fast DC charging exhibits severe loss of lithium inventory, mechanical particle micro-cracking, and advanced solid electrolyte interphase (SEI) growth. An identical pack driven mildly in a temperate zone with slow AC overnight charging retains pristine crystal structures. Reverse supply chains receive non-standardized, highly heterogeneous cores whose internal state cannot be discerned by external visual grading alone.
2. **The Knee-Point Catastrophe:** Battery degradation is non-linear. After a prolonged period of predictable, linear capacity loss, lithium-ion cells often reach a sudden inflection point—the "knee point"—where internal degradation mechanisms shift into rapid, exponential capacity decay and internal resistance spikes. Redeploying a pack that is hovering on the brink of its knee point into a 10-year stationary grid contract triggers premature system failure, catastrophic warranty claims, and field replacement costs that dwarf the asset's initial acquisition savings.
3. **Hazardous Materials Freight Barriers:** Lithium-ion batteries are classified under international dangerous goods regulations (ADR, US DOT 49 CFR, IMDG, ICAO) as Class 9 Dangerous Goods (UN 3480 and UN 3481). If a returned battery is classified as damaged or defective (under ADR Special Provision 376), it cannot be moved via standard freight. It requires specialized explosion-proof, flame-arresting, heat-insulating packaging (Packaging Instruction P908 / LP904) costing up to $1,000 per packaging shell, transport by certified hazmat carriers, and strict routing restrictions, turning a potentially profitable recovery flow into a crushing freight deficit.
4. **Tightening International Mandates:** The European Union Battery Regulation (Regulation (EU) 2023/1542) completely reorganizes battery lifecycle accountability. It mandates serialized Digital Battery Passports (DBP) starting in 2027, establishes aggressive recycling efficiency thresholds (e.g., 80% lithium recovery by 2031), imposes strict minimum levels of mandatory recycled content in new cells (cobalt, lithium, nickel), and legally defines repurposing criteria and liability transfers. Similar extended producer responsibility (EPR) regulations are advancing in North America and East Asia.
5. **Commodity and Black Mass Arbitrage:** The decision to send a battery to a second-life application versus direct recycling depends on a volatile commodity arbitrage. When battery-grade lithium carbonate, nickel sulfate, and cobalt command record premiums, the immediate Net Recovery Value ($NRV$) from crushing the pack into "black mass"—the powdered mixture of nickel, manganese, cobalt oxides, and graphite—and processing it through hydrometallurgical refining may exceed the discounted cash flows of a repurposed stationary storage system once re-certification, warranty reserves, and module disassembly labor are factored in.

Supply chain engineers and circular economy analysts must master the physical mechanics of battery degradation, electrochemical testing protocols, hazardous logistics constraints, regulatory compliance frameworks, and commodity extraction economics. This topic provides the quantitative methodologies required to optimize battery screening, triage, reverse routing, second-life integration, and recycling yield capture.

---
### Cathode Chemistry Divergence and the Circular Split: NMC vs. LFP vs. LCO

A critical operational challenge in battery reverse logistics is that circular disposition pathways are strictly dictated by cathode mineralogy. The commercial vehicle, passenger automotive, and consumer electronics sectors deploy divergent chemistries with opposing residual value profiles:

1. **Nickel-Manganese-Cobalt (NMC) and Nickel-Cobalt-Aluminum (NCA):**
   - *Physical Characteristics:* High gravimetric energy density ($200	ext{--}280\,	ext{Wh/kg}$); sensitive to high-temperature calendar aging; lower thermal runaway onset temperature ($150	ext{--}180^\circ	ext{C}$).
   - *Circular Profile:* High material intrinsic value. Nickel and cobalt command substantial commercial payability in black mass trading. When battery-grade nickel and cobalt prices spike, the high Net Recovery Value of direct hydrometallurgical recycling creates intense competition against second-life stationary storage deployments.
2. **Lithium Iron Phosphate (LFP, $\text{LiFePO}_4$):**
   - *Physical Characteristics:* Moderate energy density ($140	ext{--}180\,	ext{Wh/kg}$); exceptional cycle life ($3,500	ext{--}6,000$ cycles); high thermal stability with runaway onset exceeding $270^\circ	ext{C}$; extremely flat open-circuit voltage (OCV) profile across $20\%	ext{--}80\%$ SoC.
   - *Circular Profile:* Negligible contained metal value. Iron and phosphate have zero commodity payability in black mass markets. Reclaiming lithium alone from LFP scrap often yields an $NRV_{\text{recycle}}$ near zero or negative after factoring in inert shredding and chemical leaching costs. Conversely, LFP's massive cycle life and intrinsic thermal safety make it the ideal candidate for Tier 1 and Tier 2 Second-Life Battery Energy Storage Systems (BESS).
3. **Lithium Cobalt Oxide (LCO, $\text{LiCoO}_2$):**
   - *Physical Characteristics:* Dominated consumer electronics (smartphones, laptops, power tools) for decades; high volumetric energy density; cobalt mass fraction exceeds $60\%$ of cathode weight.
   - *Circular Profile:* Prone to physical pouch swelling and micro-dendrite formation after years of unmanaged consumer charging. Due to extreme fire risk, irregular form factors, and absent BMS standardization, e-waste LCO packs are virtually never repurposed into second-life systems; they are routed directly to mechanical and hydrometallurgical recycling to capture high-value cobalt salts.

Understanding this cathode taxonomy prevents reverse logistics planners from applying uniform disposition rules across heterogeneous incoming streams.

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Deconstruct** the multi-scale physical degradation mechanisms of lithium-ion chemistries (SEI layer expansion, lithium plating, Loss of Lithium Inventory, Loss of Active Material) and model non-linear calendar and cycle aging trajectories.
2. **Formulate** the mathematical conditions governing the "knee point" in battery aging curves to prevent deploying near-failure packs into high-liability second-life energy storage systems.
3. **Evaluate** State of Health ($\text{SoH}$) testing protocols—including Coulomb counting, Electrochemical Impedance Spectroscopy (EIS), Incremental Capacity Analysis (ICA), and Hybrid Pulse Power Characterization (HPPC)—balancing diagnostic precision against throughput time and capital expenditure.
4. **Design** a multi-tier second-life triage decision algorithm that classifies incoming battery returns into direct pack repurposing, module harvesting, cell-level extraction, or chemical recycling based on electrical, mechanical, and thermal criteria.
5. **Audit** reverse logistics packaging and transport protocols against ADR and US DOT 49 CFR Class 9 hazardous goods requirements (UN 3480/3481, Special Provisions 376 and 377, Packaging Instructions P908/LP904 and P909).
6. **Implement** compliance architectures satisfying the EU Battery Regulation (Regulation (EU) 2023/1542), including Digital Battery Passport (DBP) data models, recycled content tracking, and safety liability transfers.
7. **Calculate** the Net Recovery Value ($NRV$) of hydrometallurgical and pyrometallurgical recycling streams by modeling black mass yields, metal mass fractions, commercial payability factors, and treatment/refining charges (TC/RC).
8. **Develop** an executable, bounded Python decision engine that ingests pack-level diagnostic telemetry, executes multi-tier gating rules, and selects the recovery pathway that maximizes economic net recovery.

---

## 3. The core mental model

The reverse lifecycle of batteries and complex e-waste operates as a **Coupled Physical-Chemical Cascading Funnel**. Unlike standard consumer returns where grading is primarily cosmetic and functional, battery reverse flows are governed by invisible electrochemical states, thermal stability boundaries, and stringent legal classifications that dictate transportation safety and processing feasibility.

The system balances three competing vectors:
- **Degradation State:** How far is the electrochemical system from its catastrophic knee point and internal short-circuit risk?
- **Hazard Level:** Is the pack intact, degraded, mechanically compromised, or chemically unstable (thermal runaway hazard)?
- **Arbitrage Economics:** Does the discounted margin from a second-life warranty-backed asset exceed the immediate net metallurgical recovery value of black mass minus reverse transit and processing fees?

```text
=============================================================================================================
                          BATTERY REVERSE CASCADING & RECOVERY ARCHITECTURE
=============================================================================================================

[ RETIRED ELECTRIC VEHICLE / INDUSTRIAL ASSET / E-WASTE COLLECTION NODE ]
  - In-vehicle On-Board Diagnostics (OBD-II) extraction
  - BMS diagnostic history: State of Health (SoH), cycle count, cumulative throughput, max temp, DTCs
                                       │
                                       ▼
[ HAZARDOUS GOODS LOGISTICS TRIAGE (ADR / DOT 49 CFR) ]
 ├───────────────────────────────────────────────┬───────────────────────────────────────────────┐
 │ INTACT / NON-CRITICAL (ADR SP 377 / P909)     │ DAMAGED / DEFECTIVE / THERMAL RISK (ADR SP 376)
 │ Standard Class 9 freight; $150–$250/pack      │ Critical safety hazard; P908 / LP904 containers
 │ Standard palletization & venting              │ Hermetic, flame-arresting, pyro-insulating boxes
 │                                               │ Freight surcharge $700–$1,200/pack
 └───────────────────────┬───────────────────────┴───────────────────────┬───────────────────────┘
                         │                                               │
                         ▼                                               ▼
         [ CENTRAL GATEWAY DIAGNOSTIC CENTER ]               [ DIRECT EMERGENCY SHREDDER ]
         - Visual & hermetic seal inspection                 - Rapid brine / cryogenic immersion
         - Fast pulse screening (HPPC: 10–30s)               - Inert atmosphere ($N_2$/$CO_2$) shred
         - High-precision EIS (0.01 Hz – 1 kHz)              - Direct black mass extraction
         - Incremental Capacity Analysis (ICA)                           │
                         │                                               │
 ┌───────────────────────┴───────────────────────────────────────┐       │
 │                                                               │       │
 ▼ (SoH >= 80%, dV <= 20mV, R_ratio <= 1.35)                     │       │
[ TIER 1: PACK-LEVEL BESS REPURPOSE ]                            │       │
 - Minimal teardown; preserve factory enclosure and OEM busbars  │       │
 - Software flash: replace OEM automotive BMS with stationary BMS│       │
 - Applications: C&I peak shaving, utility solar buffering       │       │
                                                                 │       │
 ▼ (70% <= SoH < 80% OR cell imbalance dV > 20mV)                │       │
[ TIER 2: MODULE HARVESTING & RE-BINNING ]                       │       │
 - Pack enclosure de-lidded; busbars cut/unbolted                │       │
 - Module screening; defective modules culled                    │       │
 - Matched modules (SoH ±2%, dR ±5%) re-racked into 48V/100V BESS│       │
 - Applications: Telecom towers, residential ESS, low-speed EVs  │       │
                                                                 │       │
 ▼ (SoH < 70% OR internal short risk OR knee point reached)       ▼       ▼
[ TIER 3: HYDROMETALLURGICAL / PYROMETALLURGICAL RECYCLING ] ◄───────────┘
 - Deep discharge to 0.0V (salt brine bath or regenerative load bank)
 - Inert mechanical shredding (rotary shear under $N_2$ blanket)
 - Air classification (plastic separator) + magnetic separation (ferrous steel)
 - Eddy current separation (aluminum / copper foils)
 - Output: Concentrated "Black Mass" (Ni, Co, Li, Mn, Graphite)
 - Chemical extraction: Acid leaching ($H_2SO_4 + H_2O_2$) & solvent extraction
 - Final product: Battery-grade precursor salts ($Li_2CO_3$, $LiOH$, $NiSO_4$, $CoSO_4$)
=============================================================================================================
```

The fundamental law of battery circular economics is **Hierarchical Degradation Matching**: never commit high-labor teardown to an asset whose residual electrochemical life cannot amortize the re-manufacturing overhead, and never recycle an asset whose pack-level integrity can support a profitable stationary storage deployment.

---
### The Tri-Vector Disposition Boundary Formulation

Mathematically, the disposition decision engine solves a dynamic boundary surface optimization at the arrival of each battery core $k$. Let $\vec{x}_k = [\text{SoH}_k, \Delta V_k, R_{0,k}, \kappa_k]^T$ represent the physical state vector of pack $k$, where $\kappa_k = \frac{d^2 Q_k}{d N^2}$ denotes the knee curvature. Let $\vec{P}(t)$ represent current market commodity prices and secondary storage demand:

$$NRV_{\text{BESS}}(k) = E_{\text{nom}} \cdot \text{SoH}_k \cdot P_{\text{BESS}}(t) - C_{\text{refurb}}(k) - C_{\text{warranty}}(k) - C_{\text{freight}}(\text{haz})$$

$$NRV_{\text{recycle}}(k) = M_{\text{pack}} \cdot \left[ \sum_{m \in \mathcal{M}} w_m \cdot P_m(t) \cdot \alpha_m - C_{\text{process}} \right] - C_{\text{freight}}(\text{haz})$$

The optimal policy $\pi^*(k)$ chooses the maximum non-negative recovery path:

$$\pi^*(k) = \begin{cases} 
\text{TIER 1 (Pack BESS)}, & \text{if } \vec{x}_k \in \Omega_{\text{safe}} \text{ and } NRV_{\text{BESS}} \ge \max(NRV_{\text{mod}}, NRV_{\text{recycle}}) \\
\text{TIER 2 (Module Harvest)}, & \text{if } \vec{x}_k \in \Omega_{\text{mod}} \text{ and } NRV_{\text{mod}} \ge \max(NRV_{\text{BESS}}, NRV_{\text{recycle}}) \\
\text{TIER 3 (Recycle)}, & \text{otherwise.}
\end{cases}$$

Here, $\Omega_{\text{safe}}$ defines the safe operating envelope: $\text{SoH} \ge 0.80$, $\Delta V \le 20\,\text{mV}$, $R/R_0 \le 1.35$, and $\kappa_k < \epsilon_{\text{knee}}$. If an asset violates $\Omega_{\text{safe}}$, second-life deployment is legally and physically blocked, regardless of prevailing BESS resale premiums.

## 4. Definitions and vocabulary

| Term | Definition | Common Misuse | Unit / Standard |
|---|---|---|---|
| **State of Health (SoH)** | Ratio of present usable discharge capacity to original factory rated nominal capacity: $\text{SoH} = Q_{\text{actual}} / Q_{\text{nominal}}$. | Confusing SoH with State of Charge (SoC). SoC is the fuel gauge (0–100%); SoH is the size of the tank. | Dimensionless ratio or % (0.0 to 1.0) |
| **State of Charge (SoC)** | Ratio of available charge in the battery at a given instant to the maximum charge it can currently hold. | Assuming 100% SoC means 100% SoH. A degraded battery can be at 100% SoC while having only 60% SoH. | % (0% to 100%) |
| **C-Rate** | Normalized measure of charging or discharging current relative to nominal capacity. A 1C rate discharges the battery in 1 hour; 0.5C in 2 hours; 2C in 30 minutes. | Treating C-rate as constant amperage across different pack sizes. | $\text{h}^{-1}$ or normalized current |
| **Solid Electrolyte Interphase (SEI)** | Passivation layer formed on the carbon/graphite anode during initial formation cycles, resulting from electrolyte reduction. | Viewing SEI growth as purely beneficial; continuous calendar growth consumes active lithium inventory. | Nanometers ($\text{nm}$) thickness |
| **Loss of Lithium Inventory (LLI)** | Irreversible consumption of cyclable lithium ions due to continuous SEI growth, lithium plating, or electrolyte decomposition. | Believing LLI damages the electrode crystal structure; LLI means cyclable charge carriers are trapped. | Ampere-hours ($\text{Ah}$) or % |
| **Loss of Active Material (LAM)** | Degradation of cathode or anode host matrices through particle cracking, delamination, or phase transition, reducing lithium storage sites. | Conflating LAM with LLI. LAM reduces physical storage sites; LLI reduces available carriers. | $\text{Ah}$ or % |
| **Knee Point (Aging Knee)** | The sharp inflection point in a battery's capacity fade curve where degradation transitions from linear/sublinear to rapid exponential decay. | Assuming battery fade is always linear throughout second life. Operating past the knee causes rapid failure. | Cycle count or equivalent full cycles (EFC) |
| **Electrochemical Impedance Spectroscopy (EIS)** | Non-destructive characterization technique applying small AC sinusoidal voltage/current signals across a frequency spectrum to measure complex impedance. | Believing a standard DC multi-meter resistance reading provides the same insight as multi-frequency EIS. | Ohms ($\Omega$), m$\Omega$, Nyquist plot |
| **Nyquist Plot** | Parametric plot of the imaginary component of impedance (reactance, $-Z''$) versus the real component (resistance, $Z'$) across swept frequencies. | Treating high-frequency resistance as total battery impedance. | Ohms ($\Omega$) across Hz to kHz |
| **Incremental Capacity Analysis (ICA)** | Differential analysis plotting $dQ/dV$ versus voltage $V$ during low-rate charging or discharging to reveal phase transition peaks. | Attempting ICA on high C-rate cycling data; ICA requires quasi-equilibrium rates ($\le \text{C}/20$) or filtering. | $\text{Ah}/\text{V}$ vs. $\text{V}$ |
| **Differential Voltage Analysis (DVA)** | Complementary differential curve plotting $dV/dQ$ versus transferred charge $Q$, isolating individual electrode degradation. | Confusing DVA peaks with ICA peaks; DVA inverts the derivative axes. | $\text{V}/\text{Ah}$ vs. $\text{Ah}$ |
| **Hybrid Pulse Power Characterization (HPPC)** | Test protocol applying high-rate charge and discharge pulses (typically 10s to 30s) to measure dynamic voltage response and internal resistance. | Confusing pure ohmic resistance with total overpotential polarization resistance. | $\text{m}\Omega$, defined in USABC manual |
| **Black Mass** | Heavy black powder produced from shredding and mechanically processing spent battery cells, containing concentrated cathode and anode active materials. | Calling shredded whole battery scrap black mass before mechanical separation of casings and foils. | Dry weight metric ton or kg; % assay |
| **Hydrometallurgy** | Chemical recovery route utilizing aqueous acid leaching (e.g., $H_2SO_4 + H_2O_2$) followed by solvent extraction and precipitation to recover high-purity salts. | Believing hydrometallurgy burns the battery; hydrometallurgy is a low-temperature chemical dissolution process. | Recovery yield % ($>95\%$ target) |
| **Pyrometallurgy** | Thermal smelting process where batteries are fed into a high-temperature shaft furnace ($>1,200^\circ\text{C}$) to produce an alloy of Ni, Co, and Cu, slagging Li and Al. | Assuming pyrometallurgy easily recovers lithium; lithium enters the slag phase and requires complex secondary recovery. | Metric tons per batch; furnace MWh |
| **Payability** | Percentage of the prevailing London Metal Exchange (LME) or Fastmarkets metal spot price paid by a refiner for contained metals in black mass. | Assuming refiners pay 100% of commodity spot prices for the contained metal in raw scrap. | % of spot price (e.g., 65%–75%) |
| **Treatment & Refining Charges (TC/RC)** | Fees deducted by smelters or hydrometallurgical refiners to process crude concentrate/black mass into refined, battery-grade metal salts. | Assuming the seller receives gross metal value without processing deductions. | $\$ / \text{metric ton}$ or $\$ / \text{kg}$ processed |
| **UN 3480** | United Nations hazardous materials classification for stand-alone Lithium Ion Batteries (including lithium polymer). | Using UN 3481 when shipping bare, extracted battery packs without equipment. | Class 9 Dangerous Goods |
| **UN 3481** | UN hazardous classification for Lithium Ion Batteries contained in equipment or packed with equipment. | Shipping loose replacement packs under UN 3481. | Class 9 Dangerous Goods |
| **ADR Special Provision 376 (SP 376)** | International road transport regulation governing damaged, defective, or leaking lithium batteries prone to thermal runaway. | Shipping physically damaged, swollen, or thermal-event packs under standard transport rules. | Critical vs. Non-critical distinction |
| **Packaging Instruction P908 / LP904** | Prescribed ADR/UN packaging standard for damaged/defective batteries, requiring fireproof, non-combustible cushioning and thermal insulation. | Using standard corrugated cardboard or wooden crates for compromised packs. | UN certified packaging specification |
| **ADR Special Provision 377 (SP 377)** | Specific provision permitting carriage of lithium batteries destined for disposal or recycling under packaging instruction P909. | Misclassifying batteries intended for second-life repurposing as waste batteries under SP 377. | Regulatory shipping designation |
| **Digital Battery Passport (DBP)** | Legally mandated electronic record for industrial and EV batteries containing serialized provenance, chemistry, carbon footprint, and SoH telemetry. | Treating DBP as a static marketing QR code; DBP is a dynamic, cryptographically verifiable lifecycle ledger. | EU Regulation 2023/1542, Art. 77 |
| **Knee-On-Set Point** | The cycle number or time at which the derivative of capacity loss with respect to cycle count ($d^2Q/dN^2$) exhibits its maximum negative curvature. | Waiting until capacity hits 50% to declare the knee point; the onset occurs earlier, between 70% and 80% SoH. | Cycles or cumulative equivalent throughput |

---

## 5. The mechanics, step by step

Evaluating end-of-life battery packs and routing them through hazardous logistics, gateway diagnostic testing, second-life remanufacturing, or hydrometallurgical recycling requires a rigorous seven-stage operational protocol.

```text
  +---------------------------------------------------------------------------------+
  | STAGE 1: In-Situ Vehicle & Fleet Telemetry Extraction                           |
  | Read BMS error codes, cumulative Ah, cell delta-V, and thermal excursions       |
  +---------------------------------------+-----------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE 2: Hazardous Goods Classification & Packaging Protocol                    |
  | Apply ADR SP 376 vs. SP 377; select P908/LP904 explosion-containment vs. P909   |
  +---------------------------------------+-----------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE 3: Gateway Electrical Diagnostic & Degradation Screening                  |
  | Rapid pulse test (HPPC), EIS impedance spectroscopy, and knee-point detection   |
  +---------------------------------------+-----------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE 4: Mechanical & Thermal Integrity Inspection                              |
  | Casing deformation, gas vent integrity, seal air-tightness, busbar weld audit   |
  +---------------------------------------+-----------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE 5: Multi-Tier Disposition Decision Matrix                                 |
  | Algorithmic branch: Pack BESS vs. Module Harvesting vs. Hydrometallurgy         |
  +---------------------------------------+-----------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE 6: Remanufacturing, Re-Binning, & Re-Certification (Tiers 1 & 2)          |
  | Cell/module capacity grouping (+/-2%), BMS reprogramming, UL 1974 compliance   |
  +---------------------------------------+-----------------------------------------+
                                          |
                                          v
  +---------------------------------------------------------------------------------+
  | STAGE 7: Inert Shredding & Hydrometallurgical Extraction (Tier 3)               |
  | Deep discharge, N2 atmosphere milling, black mass extraction, metal recovery    |
  +---------------------------------------------------------------------------------+
```

### Stage 1: In-situ vehicle and fleet telemetry extraction

Before physical decoupling, the field technician connects an authorized diagnostic interface to the vehicle's On-Board Diagnostics (OBD-II) port or directly to the Battery Management System (BMS) controller area network (CAN) bus.

1. **Extract High-Voltage Diagnostic Trouble Codes (DTCs):** Check for persistent critical DTCs:
   - Isolation fault resistance ($R_{\text{iso}} < 500\,\Omega/\text{V}$ per ISO 6469-1).
   - Thermal runaway / over-temperature event log ($T_{\text{max}} > 60^\circ\text{C}$).
   - Internal short-circuit flag or extreme cell voltage divergence ($\Delta V_{\text{cell}} = V_{\text{max}} - V_{\text{min}} > 100\,\text{mV}$).
   - Crash pyrotechnic fuse deployment.
2. **Log Lifecycle Utilization Metrics:**
   - Cumulative total charge throughput ($Ah_{\text{throughput}}$) and Equivalent Full Cycles (EFC):
     $$\text{EFC} = \frac{\int |I(t)| \, dt}{2 \times Q_{\text{nominal}}}$$
   - Ratio of fast DC charging energy ($>50\,\text{kW}$) to total energy throughput.
   - Temperature dwell histogram: time spent operating above $45^\circ\text{C}$ or below $-10^\circ\text{C}$.
3. **Record Static Open-Circuit Voltage (OCV):** Record individual series cell voltages at rest. Calculate the cell voltage standard deviation ($\sigma_V$). If $\sigma_V > 30\,\text{mV}$, significant cell imbalance or localized accelerated degradation is present.

### Stage 2: Hazardous goods classification and packaging protocol

Lithium-ion batteries are regulated under international dangerous goods frameworks: ADR (Europe), US DOT 49 CFR (United States), and IMDG (Maritime). Misclassification carries severe criminal liability and safety risks.

```text
Does pack exhibit physical damage, swelling, electrolyte leakage, 
thermal event history, or abnormal voltage (< 1.5V/cell)?
                   │
         ┌─────────┴─────────┐
        YES                  NO
         │                   │
         ▼                   ▼
Is there imminent risk of      Is the battery destined for 
thermal runaway / explosion?   disposal or recycling?
   │            │                   │            │
  YES           NO                 YES           NO
   │            │                   │            │
   ▼            ▼                   ▼            ▼
[ADR SP 376]  [ADR SP 376]       [ADR SP 377]  [Standard Class 9]
Critical      Defective/Non-Crit Packaging     UN 3480
P908 / LP904  P908 / LP904       P909          P903
Dedicated     Standard Hazmat    Bulk Bin      Commercial LTL
Specialized   Transport          Permitted     Transport
Container     $400–$600          $150–$250     $120–$180
$800–$1,500
```

1. **Classification Under Special Provision 376 (SP 376):**
   - **Critical Damaged/Defective:** Batteries that under normal conditions of transport are prone to lead to a rapid thermal runaway, produce toxic gases, or ignite. Must be packaged under Packaging Instruction **P908** or Large Packaging Instruction **LP904**.
   - Packaging specifications: Outer packaging must be rigid steel, aluminum, or plywood meeting Packaging Group I performance. The container must incorporate non-combustible, pyro-absorbent cushioning material (vermiculite, expanded perlite, or engineered glass bubbles) capable of absorbing all released electrolyte and dampening an internal explosion. It must feature pressure-relief vents with flame arrestors.
   - Economic impact: Certified single-use LP904 boxes cost between $700 and $1,200 each, plus dangerous goods freight surcharges ($2.5\times$ to $4\times$ standard freight).
2. **Classification Under Special Provision 377 (SP 377):**
   - Applies to batteries transported for disposal or recycling. May be packaged under **P909**, allowing non-critical used batteries to be packed in sturdy outer packagings or steel drums with vermiculite cushioning at significantly lower operational cost ($150 to $250 per pack).
3. **State of Charge Depletion for Transport:**
   - To mitigate thermal hazard during transit, packs intended for second-life or recycling are discharged to $\le 30\%$ SoC prior to carrier pick-up, matching ICAO/IATA safety standards.

### Stage 3: Gateway electrical diagnostic and degradation screening

Once received at the central consolidation or triage facility, the battery undergoes non-destructive electrical characterization to decouple its degradation mechanisms: Loss of Lithium Inventory (LLI) versus Loss of Active Material (LAM).

#### A. Rapid Hybrid Pulse Power Characterization (HPPC) Screening
To avoid holding an asset in a test bay for a 10-hour full reference discharge cycle, gateway facilities execute a 30-second HPPC pulse protocol:
- Apply a discharge pulse current $I_{\text{pulse}}$ (e.g., 1C rate) for duration $\Delta t = 10\,\text{s}$.
- Measure the immediate ohmic voltage drop ($\Delta V_0$ at $t = 10\,\text{ms}$) and the polarization voltage drop ($\Delta V_p$ at $t = 10\,\text{s}$).
- Compute the Ohmic Resistance ($R_0$) and Dynamic Polarization Resistance ($R_p$):
  $$R_0 = \frac{\Delta V_0}{I_{\text{pulse}}}, \quad R_{\text{total}} = \frac{\Delta V_{10\text{s}}}{I_{\text{pulse}}}, \quad R_p = R_{\text{total}} - R_0$$
- If the pack's internal resistance has increased by more than $100\%$ relative to its beginning-of-life (BoL) specification ($R_{\text{total}} / R_{\text{nominal}} > 2.0$), the pack is severely degraded and disqualified from second-life applications.

#### B. Electrochemical Impedance Spectroscopy (EIS)
EIS applies a low-amplitude AC sinusoidal current $I(t) = I_0 \sin(\omega t)$ across frequencies from $10\,\text{kHz}$ down to $0.01\,\text{Hz}$, recording the complex voltage response $V(t) = V_0 \sin(\omega t + \phi)$. The impedance is:
$$Z(\omega) = \frac{V(\omega)}{I(\omega)} = Z'(\omega) + j Z''(\omega)$$

```text
   -Z'' (Imaginary Impedance / Reactance)
      ^
      |                 Mid-frequency Semicircle:
      |            Charge Transfer Resistance (R_ct) &
      |            Double-Layer Capacitance (C_dl)
      |                       ***
      |                   *         *
      |                 *             *      Low-frequency Tail (45°):
      |                *               *     Warburg Impedance (Z_w)
      |     High-freq *                 *      Solid-State Li+ Diffusion
      |    Intercept: |                  *             /
      |        R_s    |                   *          /
      +--------|------+--------------------*-------/-------------> Z' (Real Resistance)
             (Ohmic)                      (R_s + R_sei + R_ct)
```

- **High-Frequency Intercept ($Z'' = 0$ at $\sim 1\,\text{kHz}$):** Yields the bulk electrolyte, separator, and contact ohmic resistance ($R_s$).
- **First Semicircle (High to Mid Frequency):** Quantifies resistance across the Solid Electrolyte Interphase layer ($R_{\text{sei}}$). Significant broadening indicates severe SEI thickening.
- **Second Semicircle (Mid to Low Frequency):** Quantifies charge transfer resistance ($R_{\text{ct}}$) at the electrode/electrolyte interface.
- **Low-Frequency Tail ($<0.1\,\text{Hz}$):** 45-degree slope represents Warburg impedance ($Z_w$), reflecting the chemical diffusion coefficient of lithium ions into the active host particles ($D_{\text{Li}}$).

#### C. Incremental Capacity Analysis (ICA) and Knee-Point Detection
By differentiating charge throughput with respect to voltage ($dQ/dV$) during a controlled C/10 or C/20 recharge cycle, characteristic peaks appear corresponding to electrochemical phase transitions:
- A downward shift in peak area indicates **Loss of Lithium Inventory (LLI)** (active lithium is trapped in the SEI layer or plated as metallic lithium).
- A lateral shift in peak positions indicates internal resistance growth.
- Proportional shrinkage of specific peaks indicates **Loss of Active Material (LAM)** at the positive cathode ($LAM_{\text{PE}}$) or negative anode ($LAM_{\text{NE}}$).

**The Mathematical Knee-Point Formulation:**
Normal battery aging follows a sub-linear calendar decay governed by diffusion-limited SEI growth:
$$Q_{\text{loss}}(t) \approx k_{\text{cal}} \cdot t^{0.5} + k_{\text{cyc}} \cdot N^z \quad (z \approx 0.8\text{--}1.0)$$
The knee point occurs when cyclic degradation mechanisms (particle cracking exposing fresh graphite, followed by localized lithium plating) accelerate. The onset of the knee point is identified when the second derivative of capacity fade transitions past a critical threshold:
$$\frac{d^2 Q_{\text{loss}}}{d N^2} > \epsilon_{\text{knee}} \quad \text{and} \quad \frac{d R_0}{d N} > \lambda_{\text{resistance}}$$
Any battery whose operational trajectory places it within $15\%$ of this inflection boundary must be rejected from Tier 1 or Tier 2 second-life deployment.
#### Mathematical Modeling of Pulse Relaxation (Randles Equivalent Circuit)
To deconvolve internal resistance during the Stage 3 rapid screening pulse without waiting for full EIS frequency sweeps, test benches fit the dynamic voltage relaxation curve to a second-order Randles equivalent circuit model (Thevenin dual-RC model):

$$V(t) = V_{\text{ocv}}(\text{SoC}) - I_{\text{pulse}} R_0 - I_{\text{pulse}} R_1 \left( 1 - e^{-\frac{t}{\tau_1}} \right) - I_{\text{pulse}} R_2 \left( 1 - e^{-\frac{t}{\tau_2}} \right)$$

where:
- $R_0$ is the pure high-frequency bulk ohmic resistance of current collectors, electrolyte, and tabs.
- $R_1, C_1$ represent the rapid charge-transfer resistance and electrochemical double-layer capacitance at the particle-electrolyte interface (time constant $\tau_1 = R_1 C_1 \approx 0.5\text{--}2.0\,\text{s}$).
- $R_2, C_2$ represent the slower Solid Electrolyte Interphase (SEI) diffusion relaxation overpotential (time constant $\tau_2 = R_2 C_2 \approx 10\text{--}30\,\text{s}$).

By capturing high-frequency voltage samples ($1\,\text{kHz}$ sampling rate) during a 10-second discharge pulse followed by a 10-second open-circuit relaxation, non-linear least squares regression identifies $[R_0, R_1, \tau_1, R_2, \tau_2]$ in under 200 milliseconds of computational time. An elevation in $R_1$ flags cathode particle cracking, while an elevation in $R_2$ flags advanced SEI passivation layer growth and lithium inventory depletion.

### Stage 4: Mechanical and thermal integrity inspection

Non-electrical degradation can cause structural failure or fire hazards in second-life BESS installations:
1. **Pouch/Prismatic Swelling Assessment:** Lithium-ion cells generate gases ($CO_2$, $CO$, $CH_4$, $C_2H_4$) during overcharge, electrolyte oxidation, or high-temperature storage. Cell thickness is measured using laser displacement sensors. Swelling exceeding $8\%$ of nominal thickness indicates internal delamination and gas generation; such units are routed directly to recycling.
2. **Enclosure Seal & Environmental Audit:** Check pack casing for corrosion, physical puncture, or water ingress. Pressure decay leak testing (applying $20\text{--}50\,\text{mbar}$ positive air pressure and monitoring $\Delta P$ over 60 seconds) confirms IP67 enclosure seal integrity.
3. **High-Voltage Pyrotechnic Fuse & Contactor Inspection:** Cycle high-voltage DC contactors and inspect economizer coils. Measure contact resistance ($R_{\text{contact}} < 2\,\text{m}\Omega$). Blown pyrofuses or pitted contactors are tagged for replacement.

### Stage 5: Multi-tier disposition decision matrix

Incoming retired assets are routed according to a deterministic decision hierarchy that balances technical safety, performance yield, and net economic value.

```text
=============================================================================================================
                    GATEWAY DISPOSITION DECISION LOGIC
=============================================================================================================

IF (Physical Damage == TRUE) OR (Thermal Event DTC == TRUE) OR (Casing Leak == TRUE):
    --> DISPOSITION: TIER 3 - RECYCLE CRITICAL (ADR SP 376 / Packaging P908)
    
ELSE IF (SoH >= 0.80) AND (Cell Delta-V <= 20 mV) AND (R_int / R_nominal <= 1.35) AND (Knee Margin > 20%):
    --> DISPOSITION: TIER 1 - PACK-LEVEL BESS REPURPOSE
        (Preserve OEM structural pack; flash stationary BMS; C&I deployment)

ELSE IF (0.70 <= SoH < 0.80) OR (Cell Delta-V > 20 mV):
    IF (Disassembly Feasibility Score == HIGH) AND (R_int / R_nominal <= 1.65):
        --> DISPOSITION: TIER 2 - MODULE-LEVEL HARVESTING & RE-BINNING
            (Teardown pack to modules; cull bad modules to recycling; re-bin good modules)
    ELSE:
        --> DISPOSITION: TIER 3 - RECYCLE STANDARD (ADR SP 377 / Packaging P909)

ELSE:
    --> DISPOSITION: TIER 3 - RECYCLE STANDARD (Direct to Black Mass Processing)
=============================================================================================================
```

### Stage 6: Remanufacturing, re-binning, and re-certification (Tiers 1 & 2)

When an asset qualifies for Tier 1 or Tier 2 second-life utilization, remanufacturing engineering protocols govern the assembly:

1. **Pack-Level Conversion (Tier 1):**
   - Retain the automotive battery casing, liquid cooling plate, and internal high-voltage cabling.
   - Remove the proprietary OEM automotive BMS controller and install an open-architecture stationary BMS (supporting Modbus TCP, DNP3, or IEC 61850).
   - Re-certify the complete pack enclosure under stationary energy storage safety standards: **UL 1974** (Standard for Evaluation for Repurposing Batteries) and **IEC 62619**.
2. **Module-Level Disassembly and Re-Binning (Tier 2):**
   - De-lid pack casing using automated robotic milling or high-voltage insulated tools.
   - Sever laser-welded busbars using fiber laser cutting or mechanical shear nibblers, avoiding cell terminal short-circuiting.
   - Screen harvested modules on automated test benches. Measure individual module capacity ($Q_{\text{mod}}$) and internal resistance ($R_{\text{mod}}$).
   - **Re-Binning Algorithm:** Bin modules into strict compatibility bins:
     $$\Delta \text{SoH}_{\text{bin}} \le \pm 2.0\%, \quad \Delta R_{\text{int, bin}} \le \pm 5.0\%$$
   - Modules from different production batches or operating histories must never be wired in series without active cell balancing circuitry. In a series string, the overall string capacity is bounded by the weakest cell:
     $$Q_{\text{string}} = \min_{i} \{ Q_{\text{cell}, i} \}$$
   - Weak cells reach their lower cut-off voltage prematurely, triggering BMS string disconnect and stranding usable energy in healthier series partners.

### Stage 7: Inert shredding and hydrometallurgical extraction (Tier 3)

When a battery fails second-life screening, or reaches the end of its second life, it enters material recovery to reclaim critical battery minerals.

```text
[ SPENT PACK / MODULE ]
       │
       ▼
[ DEEP ELECTRICAL DISCHARGE ] ──► Regenerative electronic load bank to 0.5V/cell,
       │                          followed by brine immersion (5% NaCl solution)
       ▼
[ MECHANICAL SHREDDING ] ──────► Primary rotary shredder operated under inert nitrogen (N2)
       │                          or carbon dioxide (CO2) atmosphere to prevent VOC ignition
       ▼
[ THERMAL DRYING / VACUUM ] ───► Evaporate and condense volatile electrolyte solvents (DMC, EMC, EC)
       │
       ▼
[ MECHANICAL SEPARATION ]
 ├─► Magnetic Drum Separator ───────► Ferrous steel casing scrap (Fe)
 ├─► Zig-Zag Air Classifier ────────► Polymer separator films (PP/PE)
 ├─► Eddy Current Separator ────────► Coarse aluminum (Al) and copper (Cu) foils
 └─► Vibratory Fine Mesh Screen ───► CRUDE BLACK MASS (D50 < 150 um)
                                              │
                                              ▼
                                 [ HYDROMETALLURGICAL REFINING ]
                                  ├─ Acid Leaching: H2SO4 + H2O2 (reductant) at 60–80°C
                                  │   Dissolves metal oxides: Li, Ni, Co, Mn into solution
                                  ├─ Solid-Liquid Filtration: Unreacted graphite cake recovered
                                  ├─ Solvent Extraction (SX): Sequential organic extractants
                                  │   Separates Fe/Al impurities, then Co, Ni, and Mn salts
                                  └─ Chemical Precipitation: Add Na2CO3 to precipitate Li2CO3
                                      Yields battery-grade Li2CO3 (>99.5%), NiSO4, CoSO4
```

1. **Inert Shredding Protocols:** Crushing charged cells in ambient air releases volatile organic carbonates (dimethyl carbonate, ethyl methyl carbonate) that react with oxygen and residual electrical sparks to trigger violent explosions. Industrial recyclers execute shredding inside sealed chambers flooded with liquid nitrogen (cryogenic process) or gaseous $N_2 / CO_2$ maintained below $4\%$ oxygen concentration.
2. **Black Mass Metallurgical Mass Balance:** The yield of black mass typically ranges from $40\%$ to $45\%$ of the total incoming pack weight. In an NMC622 chemistry ($LiNi_{0.6}Mn_{0.2}Co_{0.2}O_2$), black mass typically grades at:
   - Nickel ($\text{Ni}$): $18\%\text{--}22\%$
   - Cobalt ($\text{Co}$): $5\%\text{--}8\%$
   - Lithium ($\text{Li}$): $3\%\text{--}4.5\%$ (equivalent to $16\%\text{--}24\%$ lithium carbonate equivalent, LCE)
   - Manganese ($\text{Mn}$): $5\%\text{--}7\%$
   - Carbon / Graphite: $35\%\text{--}42\%$
   - Residual impurities: Fluorine ($\text{F} < 1.0\%$), Aluminum ($\text{Al} < 1.5\%$), Copper ($\text{Cu} < 0.8\%$).

---
#### Chemical Leaching Stoichiometry and Precipitation Reactions
In Stage 7, hydrometallurgical refining converts crude black mass powder into ultra-pure precursor metal salts. The chemical reactions governing acid leaching and fractional precipitation are strictly governed by stoichiometry:

1. **Reductive Acid Leaching:**
   Crude black mass containing mixed cathode oxides is leached in sulfuric acid ($H_2SO_4$) at $60\text{--}80^\circ\text{C}$. Hydrogen peroxide ($H_2O_2$) acts as a critical reducing agent, converting insoluble trivalent cobalt ($\text{Co}^{3+}$) and tetravalent manganese ($\text{Mn}^{4+}$) into soluble divalent ions ($\text{Co}^{2+}, \text{Mn}^{2+}$):
   $$2 \text{Li}(Ni_{1-x-y}\text{Mn}_x \text{Co}_y)O_2 + 3 H_2SO_4 + H_2O_2 \longrightarrow 2(1-x-y)\text{NiSO}_4 + 2x \text{MnSO}_4 + 2y \text{CoSO}_4 + \text{Li}_2SO_4 + 4 H_2O + O_2 \uparrow$$
   Insoluble graphite remains as a solid filter cake and is recovered via vacuum filtration ($>98\%$ graphite recovery).

2. **Sequential Solvent Extraction (SX):**
   Organic extractants (such as Cyanex 272 or D2EHPA) selectively extract residual iron, copper, and aluminum impurities at pH $2.0\text{--}3.0$, followed by extraction of cobalt sulfate ($\text{CoSO}_4$) and manganese sulfate ($\text{MnSO}_4$). Nickel sulfate ($\text{NiSO}_4$) remains in the raffinate aqueous stream and is crystallized into battery-grade hexahydrate crystals ($\text{NiSO}_4 \cdot 6H_2O$).

3. **Lithium Carbonate Precipitation:**
   The concentrated lithium-bearing pregnant leach solution is heated to $80\text{--}95^\circ\text{C}$ and treated with saturated sodium carbonate ($\text{Na}_2CO_3$). Because lithium carbonate exhibits retrograde solubility (less soluble at elevated temperatures), it precipitates as battery-grade crystal salts:
   $$\text{Li}_2SO_4 + \text{Na}_2CO_3 \longrightarrow \text{Li}_2CO_3 \downarrow + \text{Na}_2SO_4$$
   The precipitated lithium carbonate is washed with deionized hot water, dried, and micronized to $>99.5\%$ battery-grade purity.

## 6. Worked example

An automotive OEM reverse logistics division receives a decommissioned fleet of **$N = 500$ electric commercial delivery van battery packs**. 

### Fleet Baseline Specifications
- **Pack Chemistry:** NMC622 cathode / graphite anode.
- **Original Nominal Capacity:** $E_{\text{nom}} = 60.0\,\text{kWh}$ at $400\,\text{V}$ nominal.
- **Pack Mass:** $M_{\text{pack}} = 420.0\,\text{kg}$ per unit.
- **Pack Architecture:** 10 modules per pack (each $6.0\,\text{kWh}$, $42\,\text{kg}$).
- **Fleet Operating Age:** 6.5 years; average equivalent full cycles $\text{EFC} = 1,850$.

### Step 1: Gateway Diagnostic Screening Results
The fleet undergoes Stage 1 OBD telemetry harvesting and Stage 3 rapid electrical characterization:

| Cohort | Count | % of Fleet | Mean SoH | Mean Cell $\Delta V$ | Resistance Ratio ($R/R_0$) | Physical / Thermal Status | Initial Disposition |
|---|---:|---:|---:|---:|---:|---|---|
| **Group A** | 200 | 40% | $82.4\%$ | $14\,\text{mV}$ | $1.22$ | Intact casing, no fault DTCs | **Tier 1: Pack-Level BESS** |
| **Group B** | 175 | 35% | $74.2\%$ | $45\,\text{mV}$ | $1.48$ | Intact casing, cell imbalance | **Tier 2: Module Harvesting** |
| **Group C1** | 110 | 22% | $63.8\%$ | $30\,\text{mV}$ | $1.95$ | Past knee point, high $R_{\text{int}}$ | **Tier 3: Standard Recycling** |
| **Group C2** | 15 | 3% | $61.0\%$ | $85\,\text{mV}$ | $2.10$ | Swollen cells, over-temp DTC | **Tier 3: Critical Recycling** |

### Step 2: Hazardous Transport Cost Accounting
Transportation costs vary strictly by hazardous classification:
- **Groups A, B, and C1 (485 packs):** Classified as standard non-critical under ADR SP 377 / P909 or standard UN 3480.
  - Freight tariff: $\$180.00$ per pack (consolidated multi-pack dedicated flatbed with standard strapping and fire-retardant blankets).
- **Group C2 (15 packs):** Classified as critical damaged/defective under ADR SP 376 / P908.
  - Requires single-pack LP904 certified flame-containment boxes with vermiculite filling.
  - Packaging box amortization / rental: $\$450.00$ per pack.
  - Specialized hazmat transport freight surcharge: $\$300.00$ per pack.
  - Total freight & packaging for Group C2: $\$750.00$ per pack.

Total logistics expenditure across the fleet:
$$\text{Cost}_{\text{freight}} = (485 \times \$180.00) + (15 \times \$750.00) = \$87,300 + \$11,250 = \$98,550.00$$

### Step 3: Material Recovery Modeling for Black Mass (Recycling Baseline)
To evaluate disposition trade-offs, we calculate the benchmark **Net Recovery Value of Recycling ($NRV_{\text{recycle}}$)** per 420 kg pack:

**Commodity Spot Prices:**
- Nickel ($\text{Ni}$): $\$16.50\,\text{/kg}$
- Cobalt ($\text{Co}$): $\$28.00\,\text{/kg}$
- Lithium ($\text{Li}$ metal basis): $\$14.00\,\text{/kg}$
- Copper ($\text{Cu}$ foil): $\$9.20\,\text{/kg}$
- Aluminum ($\text{Al}$ casing/foil): $\$2.30\,\text{/kg}$

**Mass Fractions and Payability Factors:**
1. **Black Mass Yield:** $42.0\%$ of pack mass $= 420.0 \times 0.42 = 176.4\,\text{kg}$ of black mass.
   - Nickel content: $20.5\%$ of black mass $= 36.162\,\text{kg}$. Payability $= 75\%$.
     $$\text{Value}_{\text{Ni}} = 36.162 \times \$16.50 \times 0.75 = \$447.50$$
   - Cobalt content: $6.8\%$ of black mass $= 11.9952\,\text{kg}$. Payability $= 75\%$.
     $$\text{Value}_{\text{Co}} = 11.9952 \times \$28.00 \times 0.75 = \$251.90$$
   - Lithium content: $3.6\%$ of black mass $= 6.3504\,\text{kg}$. Payability $= 65\%$.
     $$\text{Value}_{\text{Li}} = 6.3504 \times \$14.00 \times 0.65 = \$57.79$$
2. **Physical Foil & Structural Scrap:**
   - Copper scrap: $7.5\%$ of total pack mass $= 31.5\,\text{kg}$. Payability $= 85\%$.
     $$\text{Value}_{\text{Cu}} = 31.5 \times \$9.20 \times 0.85 = \$246.33$$
   - Aluminum scrap: $18.0\%$ of total pack mass $= 75.6\,\text{kg}$. Payability $= 70\%$.
     $$\text{Value}_{\text{Al}} = 75.6 \times \$2.30 \times 0.70 = \$121.72$$

**Gross Recoverable Metal Value:**
$$\text{Gross Value} = \$447.50 + \$251.90 + \$57.79 + \$246.33 + \$121.72 = \$1,125.24\,\text{per pack}$$

**Deductions (Processing, Shredding, and Refining Fee):**
- Recycler treatment and refining charge (TC/RC) including inert shredding: $\$0.85\,\text{/kg}$ of total pack weight.
  $$\text{Processing Fee} = 420.0\,\text{kg} \times \$0.85 = \$357.00\,\text{per pack}$$

**Net Value at Recycling Gate (before transport):**
$$\text{Net Value}_{\text{gate}} = \$1,125.24 - \$357.00 = \$768.24\,\text{per pack}$$

- **Net Recovery Value for Standard Recycling (Group C1):**
  $$NRV_{\text{recycle, std}} = \$768.24 - \$180.00 = \$588.24\,\text{per pack}$$
- **Net Recovery Value for Critical Recycling (Group C2):**
  $$NRV_{\text{recycle, crit}} = \$768.24 - \$750.00 = \$18.24\,\text{per pack}$$

### Step 4: Tier 1 Pack-Level Repurposing Economics (Group A: 200 Packs)
- **Usable Remaining Energy:** $60.0\,\text{kWh} \times 0.824 = 49.44\,\text{kWh}$ per pack.
- **Secondary Market Selling Price:** $\$105.00\,\text{/kWh}$ of certified remaining capacity.
  $$\text{Gross Revenue}_{\text{pack}} = 49.44 \times \$105.00 = \$5,191.20$$
- **Refurbishment and Certification Costs:**
  - In-depth electrical testing and automated cycle validation: $\$450.00$
  - Open-source stationary BMS flash and harness integration: $\$150.00$
  - UL 1974 re-certification allocation and 3-year warranty reserve: $\$300.00$
  - Inbound transport: $\$180.00$
- **Total Cost per Pack:**
  $$\text{Cost}_{\text{Tier 1}} = \$450.00 + \$150.00 + \$300.00 + \$180.00 = \$1,080.00$$
- **Net Recovery Value (Tier 1):**
  $$NRV_{\text{Tier 1}} = \$5,191.20 - \$1,080.00 = \$4,111.20\,\text{per pack}$$
  *Advantage over direct standard recycling:* $\$4,111.20 - \$588.24 = +\$3,522.96$ per pack ($+599\%$).

### Step 5: Tier 2 Module Harvesting Economics (Group B: 175 Packs)
- 175 packs disassembled into $1,750$ modules ($6.0\,\text{kWh}$ nominal each).
- **Module Screening Yield:** Testing reveals $80.0\%$ of modules ($1,400$ units) pass Tier 2 criteria ($\text{SoH} \ge 75\%$). The remaining $20.0\%$ ($350$ modules) fail due to localized cell degradation and are routed to recycling.
- **Harvested Usable Capacity:**
  $$\text{Energy}_{\text{pass}} = 10\,\text{modules} \times 6.0\,\text{kWh} \times 0.742 \times 0.80 = 35.616\,\text{kWh per pack}$$
- **Module Resale Revenue (Telecom backup / Solar):** $\$90.00\,\text{/kWh}$.
  $$\text{Revenue}_{\text{modules}} = 35.616 \times \$90.00 = \$3,205.44$$
- **Scrap Credit from Failed Modules (20% to recycling):**
  $$\text{Scrap Credit} = 0.20 \times (\$1,125.24 - \$357.00) = \$153.65$$
- **Processing and Teardown Costs:**
  - Automated casing removal and busbar laser de-welding: $\$380.00$
  - Individual module testing and binning: $\$240.00$
  - Inbound transport: $\$180.00$
  $$\text{Total Cost}_{\text{Tier 2}} = \$380.00 + \$240.00 + \$180.00 = \$800.00$$
- **Net Recovery Value (Tier 2):**
  $$NRV_{\text{Tier 2}} = \$3,205.44 + \$153.65 - \$800.00 = \$2,559.09\,\text{per pack}$$
  *Advantage over direct standard recycling:* $\$2,559.09 - \$588.24 = +\$1,970.85$ per pack ($+335\%$).

### Step 6: Total Fleet Circular Recovery Value
Summarizing the optimal cascading disposition strategy across all 500 packs:

$$\begin{aligned}
\text{Fleet } NRV &= (200 \times \$4,111.20) + (175 \times \$2,559.09) + (110 \times \$588.24) + (15 \times \$18.24) \\
&= \$822,240.00 + \$447,840.75 + \$64,706.40 + \$273.60 \\
&= \mathbf{\$1,335,060.75}
\end{aligned}$$

**Comparison Against Counterfactual Scenarios:**
1. **Blind Total Recycling (Dump all 500 packs to recycling):**
   $$\text{Baseline } NRV = (485 \times \$588.24) + (15 \times \$18.24) = \$285,296.40 + \$273.60 = \mathbf{\$285,570.00}$$
   *The multi-tier screening engine captures an additional **$\$1,049,490.75$** ($+367\%$) in net asset recovery!*
2. **Blind Second-Life Commitment (Attempting to repurpose all 500 packs as BESS):**
   Deploying Groups B, C1, and C2 directly into BESS triggers catastrophic field failures. The 125 packs in Groups C1 and C2 would hit their degradation knee point within 6 months, causing thermal faults, string cutoffs, and field warranty replacement penalties averaging $\$8,500$ per deployed pack, completely wiping out initial operating profit.

---

## 7. Code, query, or calculation

The following self-contained Python decision engine implements the multi-tier battery disposition logic, incorporating black mass metal payability pricing, hazardous freight categorization, and second-life economic netting.

```python
from dataclasses import dataclass
from typing import Literal

@dataclass(frozen=True)
class BatteryPack:
    serial_id: str
    nominal_kwh: float
    weight_kg: float
    soh: float
    delta_v_cell_mv: float
    r_int_ratio: float
    thermal_fault: bool

Disposition = Literal["PACK_BESS", "MODULE_HARVEST", "RECYCLE_STD", "RECYCLE_CRITICAL"]

def evaluate_disposition(p: BatteryPack) -> tuple[Disposition, float, dict[str, float]]:
    # Black mass & metal payability model (NMC622)
    bm_kg = p.weight_kg * 0.42
    gross_metal = (
        (bm_kg * 0.205 * 16.50 * 0.75) +  # Nickel (75% payability)
        (bm_kg * 0.068 * 28.00 * 0.75) +  # Cobalt (75% payability)
        (bm_kg * 0.036 * 14.00 * 0.65) +  # Lithium (65% payability)
        (p.weight_kg * 0.075 * 9.20 * 0.85) +  # Copper (85% payability)
        (p.weight_kg * 0.180 * 2.30 * 0.70)    # Aluminum (70% payability)
    )
    recycle_fee = p.weight_kg * 0.85
    std_freight, crit_freight = 180.0, 750.0

    if p.thermal_fault:
        nrv = gross_metal - recycle_fee - crit_freight
        return "RECYCLE_CRITICAL", round(nrv, 2), {"freight": crit_freight, "gross": gross_metal}

    nrv_recycle_std = gross_metal - recycle_fee - std_freight

    # Tier 1: Pack-level stationary storage (BESS) repurposing
    if p.soh >= 0.80 and p.delta_v_cell_mv <= 20.0 and p.r_int_ratio <= 1.35:
        revenue = p.nominal_kwh * p.soh * 105.0  # $105/kWh remanufactured
        refurb_costs = 450.0 + 150.0 + 300.0     # Test, recertification, warranty
        nrv_pack = revenue - refurb_costs - std_freight
        if nrv_pack > nrv_recycle_std:
            return "PACK_BESS", round(nrv_pack, 2), {"revenue": revenue, "freight": std_freight}

    # Tier 2: Module-level harvesting for telecom/solar backup
    if p.soh >= 0.70 and p.r_int_ratio <= 1.65:
        harvest_kwh = p.nominal_kwh * p.soh * 0.80  # 80% module yield
        module_rev = harvest_kwh * 90.0             # $90/kWh module resale
        scrap_cred = (gross_metal - recycle_fee) * 0.20
        nrv_mod = module_rev + scrap_cred - 620.0 - std_freight  # $620 teardown/test
        if nrv_mod > nrv_recycle_std:
            return "MODULE_HARVEST", round(nrv_mod, 2), {"revenue": module_rev, "freight": std_freight}

    return "RECYCLE_STD", round(nrv_recycle_std, 2), {"gross": gross_metal, "freight": std_freight}

if __name__ == "__main__":
    fleet = [
        BatteryPack("BAT-001", 60.0, 420.0, 0.824, 14.0, 1.22, False),
        BatteryPack("BAT-002", 60.0, 420.0, 0.742, 45.0, 1.48, False),
        BatteryPack("BAT-003", 60.0, 420.0, 0.638, 30.0, 1.95, False),
        BatteryPack("BAT-004", 60.0, 420.0, 0.610, 85.0, 2.10, True),
    ]
    for pack in fleet:
        disp, nrv, meta = evaluate_disposition(pack)
        print(f"[{pack.serial_id}] SoH: {pack.soh*100:.1f}% -> {disp:<16} | NRV: ${nrv:>8.2f}")
```

### Execution Trace and Verification
Executing the code block yields the exact expected output matching the worked example calculations:
```text
[BAT-001] SoH: 82.4% -> PACK_BESS        | NRV: $ 4111.20
[BAT-002] SoH: 74.2% -> MODULE_HARVEST   | NRV: $ 2559.09
[BAT-003] SoH: 63.8% -> RECYCLE_STD      | NRV: $  588.24
[BAT-004] SoH: 61.0% -> RECYCLE_CRITICAL | NRV: $   18.24
```

---

## 8. Data, standards, and regulatory context

Battery reverse logistics operates under some of the most rigorous international treaties, environmental directives, and transport safety mandates in global commerce.

```text
========================================================================================================================
                                     GLOBAL REGULATORY & COMPLIANCE MATRIX
========================================================================================================================
Regulatory Body / Mandate       Primary Scope & Legal Enforceability                   Direct Operational Constraint
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
UN Model Regulations            Global Dangerous Goods Transport Classification.       Defines UN 3480 (lithium-ion cells/packs)
(Orange Book)                   Binding via ADR, US DOT 49 CFR, and IMDG.              and packaging standards P903, P908, P909.
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
ADR Special Provision 376       Mandatory road transport rules in 54 countries for     Requires certified flame-proof, pressure-
(ECE/TRANS/WP.15)               damaged, defective, or thermally unstable batteries.   relieved containers (LP904) for critical packs.
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
EU Battery Regulation           Mandatory across all 27 EU member states.              Mandates Digital Battery Passports (2027),
Regulation (EU) 2023/1542       Repeals Directive 2006/66/EC. Covers entire lifecycle. recycled content minimums, recycling efficiency.
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
US DOT 49 CFR § 173.185         Hazardous materials shipping in the United States.     Regulates ground and maritime carriage; strictly
(PHMSA Enforced)                Civil and criminal penalties for non-declaration.       forbids air transport of damaged/defective packs.
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
UL 1974                         Safety standard for evaluating, sorting, and grading   Mandates grading procedures, test bench protocols,
(ANSI Accredited)               batteries for second-life applications.                and quality management for remanufactured packs.
────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
IEC 62619 / ISO 6469-1          Industrial and secondary lithium battery safety.       Specifies electrical isolation resistance, short-
(International Standards)       Global benchmark for stationary storage safety.        circuit protection, and thermal runaway containment.
========================================================================================================================
```

### 1. The EU Battery Regulation (Regulation (EU) 2023/1542)
Entering into force across the European Union, this regulation replaces the old Battery Directive (2006/66/EC) with a directly applicable regulation establishing unprecedented circular supply chain mandates:

1. **Digital Battery Passport (DBP) (Mandatory February 2027):**
   - Applies to all electric vehicle batteries, light means of transport (LMT) batteries, and rechargeable industrial batteries $>2\,\text{kWh}$.
   - Stored in a decentralized, interoperable ledger accessible via a standardized QR code engraved on the battery casing.
   - Dynamic data fields must be updated across lifecycle: actual capacity, State of Health (SoH), internal resistance, round-trip energy efficiency, cumulative cycle count, history of deep discharge and thermal excursions.
   - Second-life operators who repurpose or remanufacture a battery assume legal manufacturer status. They must update the passport, certify a new State of Certified Energy ($\text{SOCE}$), and affix a new CE mark.
2. **Mandatory Minimum Recycled Content Targets:**
   New batteries placed on the EU market must contain verified percentages of active materials recovered from battery manufacturing scrap or post-consumer black mass:
   
   | Target Material | Target by 2031 (Art. 8) | Target by 2036 (Art. 8) |
   |---|---:|---:|
   | **Cobalt ($\text{Co}$)** | $16\%$ | $26\%$ |
   | **Lithium ($\text{Li}$)** | $6\%$ | $12\%$ |
   | **Nickel ($\text{Ni}$)** | $6\%$ | $15\%$ |
   | **Lead ($\text{Pb}$)** | $85\%$ | $85\%$ |

3. **Material Recovery Efficiency Mandates:**
   Hydrometallurgical and pyrometallurgical recycling facilities operating in or supplying the EU must hit strict mass-based recovery efficiencies:
   - **Lithium recovery:** Minimum $50\%$ by end of 2027; minimum $80\%$ by end of 2031.
   - **Cobalt, Copper, and Nickel recovery:** Minimum $90\%$ by end of 2027; minimum $95\%$ by end of 2031.
   - These thresholds render standard single-stage pyrometallurgical smelting (which loses lithium to slag) legally obsolete unless secondary slag leaching is deployed.

### 2. Hazardous Materials Shipping Regulations (ADR and US 49 CFR)
- **UN 3480 vs. UN 3481:** UN 3480 covers standalone lithium-ion batteries. When a battery is mounted inside an electric vehicle or within a mobile energy storage container, it ships under UN 3481 or UN 3536.
- **Air Transport Prohibition:** Under ICAO / IATA Dangerous Goods Regulations (DGR), damaged, defective, or waste batteries (SP 376 / SP 377) are strictly forbidden from carriage on both passenger and dedicated cargo aircraft.
- **Driver and Carrier Certification:** Road transport of Class 9 batteries exceeding ADR threshold quantities requires drivers holding an ADR training certificate with Class 9 endorsement, transport vehicles equipped with orange hazard plates, and emergency response instructions in writing.

---
### Digital Battery Passport (DBP) Mandatory Data Architecture

Under Article 77 and Annex XIII of Regulation (EU) 2023/1542, the Digital Battery Passport establishes seven mandatory, cryptographically verifiable data clusters:

| Data Cluster | Specific Mandatory Data Attributes | Access Level / Permissions |
|---|---|---|
| **1. General Battery & Manufacturer Information** | Unique identifier (UUID/GS1 Digital Link); manufacturer name, registered address; manufacturing date, geographic plant location; battery model and type code; original nominal capacity ($E_{\text{nom}}$ in kWh), nominal voltage ($V_{\text{nom}}$), and physical mass (kg). | **Public Access:** Readable via direct QR code scan by consumers, fleet operators, and customs authorities. |
| **2. Carbon Footprint Declaration** | Total cradle-to-gate carbon footprint (kg $\text{CO}_2\text{e}$ per kWh); carbon footprint across four lifecycle stages: raw material extraction, cell manufacturing, assembly/inbound logistics, and circular end-of-life recycling credits; URL link to third-party verification report. | **Public Access:** Must be published on the European Commission centralized battery registry. |
| **3. Supply Chain Due Diligence** | Verification report confirming compliance with OECD Due Diligence Guidance for Responsible Supply Chains of Minerals; independent third-party audit of cobalt, lithium, nickel, and natural graphite sourcing from conflict-affected and high-risk areas (CAHRAs). | **Restricted / Public:** Executive audit summary is public; underlying supplier audit logs restricted to market surveillance authorities. |
| **4. Material Composition & Recycled Content** | Full bill-of-materials chemical disclosure (hazardous substances, electrolyte solvents, cathode/anode active mineral mass); certified percentage of recycled cobalt, lithium, nickel, and lead incorporated into the cell active materials. | **Authorized Stakeholders:** Full chemical assay accessible to notified bodies and accredited recyclers. |
| **5. Performance & Durability Metrics** | Beginning-of-life certified rated capacity ($Ah$); capacity loss and internal resistance under standardized cycle life test conditions (ISO 12405 / IEC 62660); expected calendar service life under temperate and high-temperature conditions. | **Public / Commercial:** Standardized ratings are public; raw test cycle degradation logs accessible to second-life remanufacturers. |
| **6. Dynamic In-Use Battery State & Lifetime Telemetry** | Real-time / periodically synced State of Health (SoH); present internal resistance ($R_{\text{int}}$); cumulative cycle count (Equivalent Full Cycles); cumulative energy throughput (MWh); maximum and minimum temperature operating records; deep discharge and over-voltage fault event logs. | **Restricted Access:** Accessible only to the vehicle owner, authorized repair technicians, and registered second-life repurposers. |
| **7. Circularity, Dismantling, & Second Life Status** | Dismantling schematics (CAD drawings, high-voltage disconnect steps, explosive bolt locations); safety data sheets (SDS) and toxic gas emission profiles; operational status flag: "First Life", "Repurposed / Second Life", or "Waste Battery for Recycling"; State of Certified Energy (SOCE) for remanufactured units. | **Recyclers & Repurposers:** Teardown schematics restricted to accredited facilities; lifecycle status flag is public. |

## 9. Industry practice and case evidence

Real-world deployment of battery reverse logistics and second-life ecosystems illustrates the gap between academic circular models and industrial reality.

```text
=============================================================================================================
                                     INDUSTRIAL REVERSE ECOSYSTEM MATRIX
=============================================================================================================
Enterprise / Initiative     Operational Model               Industrial Scale               Key Empirical Finding
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Nissan & 4R Energy          Direct Pack & Module Repurpose  Namie Facility (Japan)         Automated robotic grading
(Sumitomo JV)               Leaf packs to solar BESS and    >2,000 packs/year              achieved 4-hour cycle time;
                            railway crossing backup power.                                 SoH yield averaged 81%.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Renault & Mobilize          Advanced Battery Storage (ABS)  Douai & Elhange (France)       Pack-level conversion avoided
                            Grid-scale frequency response   75 MWh operational capacity    teardown cost; OEM BMS software
                            using retired Zoe packs.                                       required custom gateway flash.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Redwood Materials           Closed-Loop Hydrometallurgy     Carson City, NV (USA)          Achieved >95% recovery for Ni,
(J.B. Straubel)             Direct gigafactory scrap and    >40,000 tons/year              Co, Li; refining back into
                            EoL pack chemical refining.                                    precursor cathode active materials.
─────────────────────────────────────────────────────────────────────────────────────────────────────────────
Umicore Battery Recycling   Combined Pyro-Hydro Smelting    Hoboken (Belgium)              Ultra-high throughput; required
                            Recycling of EV and e-waste     >7,000 tons/year               secondary hydromet step to recover
                            concentrates.                                                  lithium from smelting slag.
=============================================================================================================
```

### Case Evidence 1: Nissan & 4R Energy (Namie, Fukushima)
Founded in 2010 as a joint venture between Nissan and Sumitomo Corporation, 4R Energy established one of the world's first industrial-scale battery repurposing facilities in Namie, Japan.
- **Operational Reality:** Early manual inspection required up to 16 hours per pack to execute full charge-discharge cycles. 4R Energy developed a proprietary high-speed scanning algorithm that evaluates cell internal resistance and partial-state OCV curves, compressing diagnostic screening time to under 4 hours per pack.
- **Cascading Outcomes:** Decommissioned Nissan Leaf packs (24 kWh and 40 kWh) are graded into three commercial streams:
  - *Grade A (SoH $> 80\%$):* Re-manufactured as replacement packs for older Leaf vehicles.
  - *Grade B (SoH $60\%\text{--}80\%$):* Re-configured into stationary commercial solar storage systems and automated guided vehicle (AGV) power units for manufacturing plants.
  - *Grade C (SoH $< 60\%$ or damaged):* Repurposed into emergency backup power supplies for railway crossing signals.

### Case Evidence 2: Redwood Materials Closed-Loop Refining
Founded by former Tesla CTO J.B. Straubel, Redwood Materials operates commercial-scale mechanical shredding and hydrometallurgical recycling in Nevada.
- **Process Optimization:** Redwood bypasses intermediate trading brokers by collecting end-of-life battery packs directly from automotive OEMs (Ford, Toyota, Volvo, Audi) and manufacturing scrap from Panasonic's Gigafactory.
- **Empirical Recovery Evidence:** Documented recovery efficiencies consistently exceed $95\%$ for nickel, cobalt, copper, and lithium. Rather than selling crude black mass to overseas traders, Redwood refines the leached liquor directly into battery-grade precursor cathode active material (pCAM) and anode copper foil, creating a domestic closed-loop supply chain that qualifies for US Inflation Reduction Act (IRA) Section 30D clean vehicle tax credits.

### Documented Fact vs. Industry Inference
- **Documented Fact:** Tier 1 pack-level BESS repurposing requires significantly lower capital expenditure than cell-level harvesting because laser-welded busbar removal introduces extreme scrap rates ($8\%\text{--}15\%$ cell damage during teardown).
- **Inference (Supply Chain Modeling):** High lithium commodity spot prices favor direct hydrometallurgical recycling over second-life storage, because immediate cash conversion from black mass out-weighs the multi-year discounted warranty risk of second-life BESS installations.

---

## 10. Failure modes and red flags

Operating reverse battery networks without deep electrochemical and regulatory safeguards leads to catastrophic physical, financial, and legal failures.

| # | Failure Mode | Early Warning Sign | Root Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|---|
| 1 | **Thermal Runaway in Staging Area** | Localized temperature rise ($>5^\circ\text{C}$ above ambient); faint sweet chemical odor (evaporated carbonate electrolyte). | Internal micro-short caused by lithium dendrite penetration through damaged separator during rest. | Uncontained self-heating fire ($>900^\circ\text{C}$); toxic gas emission ($HF$, $CO$); facility total loss. | Continuous thermal imaging cameras in staging yards; automated submerged quench tanks; isolated fire bunkers. |
| 2 | **Knee-Point Blindness in Second Life** | Rapid acceleration in daily capacity loss ($dQ/dt$) after 3 months of mild stationary cycling. | Battery was deployed at or near its knee point; particle cracking triggered rapid lithium plating. | Premature BESS failure; contractual uptime SLA breach; high-cost emergency field pack replacement. | Enforce conservative knee margin threshold ($>15\%$); model $d^2Q/dN^2$ curvature; reject packs near knee. |
| 3 | **ADR SP 376 Carrier Rejection / Impoundment** | Road transport inspector notes swollen pack casing or lack of LP904 certified packaging markings. | Shipper attempted to move defective/swollen packs under standard UN 3480 freight tariffs. | Vehicle impoundment; regulatory fines ($>\$25,000$ per violation); shipper de-certification. | Formal pre-transport inspection gate; certified hazmat dangerous goods safety advisor (DGSA) sign-off. |
| 4 | **Premature Series String Disconnect** | BESS string trips offline when average string SoC is still at $35\%$. | Module capacity mismatch ($\Delta \text{SoH} > 5\%$) inside a series string; weakest cell hits lower cut-off voltage. | Loss of $30\%\text{--}40\%$ usable storage capacity; erratic power delivery; customer dispute. | Strict re-binning tolerance ($\Delta \text{SoH} \le \pm 2\%$); deploy active BMS balancing shuttling energy between cells. |
| 5 | **Busbar Weld Rupture During Teardown** | Sparks, terminal tear-out, or pouch puncture during module mechanical harvesting. | High mechanical shear stress during unbolting or de-welding of laser-fused aluminum/copper busbars. | Cell short-circuit; fire hazard; irreparable module damage; scrap rate increases past economic viability. | Automated fiber laser ablation; design for disassembly (bolted busbars); ultrasonic de-bonding protocols. |
| 6 | **Fluorine Contamination in Black Mass** | Hydrometallurgical refiner rejects black mass shipment or imposes $>\$500/\text{ton}$ penalty deduction. | Inadequate thermal drying during mechanical shredding leaves PVDF binder and $LiPF_6$ electrolyte salt intact. | Fluorine levels exceed $1.0\%$; causes severe equipment corrosion in hydrometallurgical autoclaves. | Low-temperature vacuum pyrolysis ($350\text{--}500^\circ\text{C}$) to decompose PVDF and strip fluorine prior to leaching. |
| 7 | **BMS Telemetry Spoofing / Data Corruption** | Diagnostic scan displays 98% SoH on a 7-year-old vehicle pack with heavy cosmetic chassis wear. | Vehicle owner or unscrupulous broker reset the BMS adaptive learning algorithms prior to vehicle handoff. | Over-payment for degraded core; misclassification into Tier 1; early customer warranty failure. | Never rely solely on BMS reported SoH; mandate physical gateway pulse (HPPC) and EIS impedance testing. |
| 8 | **Deep Discharged Cell Inversion** | Cell open-circuit voltage measured below $1.0\,\text{V}$ (or negative polarity). | Parasitic load from dormant electronics or internal self-discharge drove cell past zero-charge threshold. | Copper foil current collector dissolves into electrolyte; recharging forms copper dendrites and fires. | Mandatory cutoff: Any cell with $V_{\text{ocv}} < 1.5\,\text{V}$ is classified as hazardous scrap and never recharged. |
| 9 | **Regulatory Non-Compliance on Recycled Content** | Inability to clear new battery cells through EU customs under Regulation 2023/1542. | Lack of cryptographically auditable provenance tracing recycled cobalt/lithium back to certified recyclers. | Import ban; product recall; total forfeiture of EU commercial market access. | Adopt GS1 EPCIS 2.0 and interoperable Digital Battery Passport infrastructure linking refinery mass balances. |
| 10 | **Smelter Slag Lithium Stranding** | Hydrometallurgical recovery economics collapse due to processing low-grade pyrometallurgical matte. | Feeding whole packs into high-temperature pyrometallurgical smelters without pre-leaching. | Lithium and aluminum oxidize entirely into calcium silicate slag; lithium recovery drops to $0\%$. | Shift to front-end mechanical dismantling and direct hydrometallurgical leaching to guarantee $>90\%$ Li yield. |
| 11 | **Current Collector Foil Delamination** | Fine metallic flakes (copper or aluminum) clogging pneumatic shredder filters. | Micro-tearing during ultrasonic busbar de-bonding or excessive mechanical shear during cell harvesting. | Internal electrical shorting across cell jelly roll; fires during re-assembly; module scrap rate jumps $>25\%$. | Deploy non-contact infrared thermal de-bonding; enforce strict peel-angle tolerances during robotic harvesting. |
| 12 | **Cooling Channel Cross-Contamination** | Glycol residue detected on high-voltage terminal contacts; gradual isolation resistance decay. | Re-using pack chassis with breached internal coolant channels (aluminum micro-channel corrosion from old ethylene glycol). | Electrical arc tracking across pack busbars; dielectric breakdown ($R_{\text{iso}} < 100\,\Omega/\text{V}$); catastrophic fire in BESS. | Mandatory pressure-decay and vacuum helium leak testing on all fluid cooling circuits prior to second-life re-assembly. |

---

## 11. Assumptions and limits

Every quantitative model in battery reverse logistics rests on specific electrochemical and market boundary conditions:

| # | Assumption | Why Needed | Boundary Condition / What Breaks It | Validation Protocol |
|---|---|---|---|---|
| 1 | **Quasi-Linear Pre-Knee Degradation** | Enables extrapolation of remaining useful life (RUL) for second-life financial pro-formas. | High C-rate fast charging or low-temperature operation triggers premature non-linear lithium plating. | Validate through differential voltage analysis ($dV/dQ$) peak shift tracking. |
| 2 | **Stationary Metal Spot Prices & Payability** | Establishes the baseline Net Recovery Value ($NRV_{\text{recycle}}$) for arbitrage decisions. | Extreme commodity volatility (e.g., lithium carbonate swinging from $\$80,000/\text{t}$ to $\$12,000/\text{t}$). | Mark-to-market dynamic hedging; run Monte Carlo sensitivity on LME/Fastmarkets forward curves. |
| 3 | **Uniform Module Degradation Within Pack** | Assumes individual modules within an intact pack share approximately equal SoH. | Localized thermal gradients (e.g., cooling channel clogging) cause severe single-module degradation. | Mandatory individual module voltage and internal resistance screening during pack gateway triage. |
| 4 | **Non-Destructive Busbar Disassembly** | Assumes modules can be harvested from packs without damaging cell terminal seals. | Laser-welded aluminum busbars tear cell casing foil during mechanical de-processing. | Microscopic inspection of weld root; ultrasonic shear testing on pilot disassembly batches. |
| 5 | **Cryptographic BMS Key Availability** | Enables flashing open-source firmware onto OEM automotive battery controllers. | OEM locks BMS microprocessor with proprietary asymmetric encryption keys, blocking third-party access. | Maintain OEM diagnostic licensing agreements or budget complete BMS master/slave hardware replacement. |
| 6 | **Air-Tightness of Secondary Warranty Transfer** | Assumes the original automotive OEM is fully indemnified against second-life failure liabilities. | Ambiguous jurisdiction in contracts or failure to comply with EU 2023/1542 repurposing transfer protocols. | Legal audit of repurposing agreements; strict adherence to UL 1974 re-certification labeling. |
| 7 | **Ambient Temperature Control During Fast Screening** | Assumes measured ohmic and polarization resistance reflect intrinsic electrochemical state. | Testing executed in unheated warehouse ($T < 10^\circ\text{C}$); temperature drop doubles electrolyte resistance. | Climate-controlled test chambers maintained at $25.0^\circ\text{C} \pm 1.5^\circ\text{C}$ with 4-hour thermal soak. |
| 8 | **Zero Explosive Atmosphere During Milling** | Assumes mechanical shredding can be operated safely under inert gas blankets. | Oxygen ingress ($O_2 > 4\%$) or nitrogen purging compressor failure triggers solvent vapor ignition. | Triple-redundant zirconia oxygen sensors with automated interlocks and rapid $CO_2$ deluge suppression. |
| 9 | **Recycled Lithium Carbonate Market Liquidity** | Assumes chemical precursor salts ($Li_2CO_3, NiSO_4$) can be sold immediately at prevailing index prices. | Supply-demand imbalances, off-spec purity ($<99.5\%$), or cathode chemistry shifts toward sodium-ion. | Secure multi-year off-take agreements with cathode active material (pCAM) manufacturers with floor-pricing bands. |
| 10 | **Accurate SoC Initialization for Pulse Tests** | Assumes the pack can be set precisely to $50.0\% \pm 1.0\%$ SoC prior to running 30s HPPC screening pulses. | Hysteresis in flat-OCV chemistries (especially LFP) or uncalibrated Coulomb counting in degraded packs. | Perform 2-hour low-rate rest and temperature stabilization; use differential voltage ($dV/dQ$) anchor points. |

---

## 12. Exercises

### Exercise 1: Degradation Mechanism Identification
- **Type:** Recall & Diagnosis
- **Problem Statement:** A returned EV battery pack exhibits significant capacity fade ($24\%$ loss), but its high-frequency ohmic resistance ($R_s$) has only increased by $4\%$. However, low-rate incremental capacity analysis ($dQ/dV$) reveals that the area under the primary graphite lithiation peak has contracted by $25\%$, while peak positions have not shifted along the voltage axis. Identify the dominant physical degradation mode (LLI vs. LAM vs. Contact Corrosion) and explain whether this pack is suitable for Tier 1 BESS repurposing.
- **Deliverable:** 200-word written technical diagnosis naming the primary mechanism and disposition recommendation.
- **Acceptance Criteria:** Correctly identifies Loss of Lithium Inventory (LLI) without severe impedance growth; provides precise reasoning based on peak area versus peak shift.
- **Estimated Time:** 20 minutes.

### Exercise 2: ADR SP 376 Logistics Cost Evaluation
- **Type:** Quantitative Calculation
- **Problem Statement:** A reverse logistics coordinator must move a batch of 40 decommissioned 80 kWh battery packs (mass = 550 kg each) from a regional collision center to a central recycling plant (distance = 600 km). Visual inspection reveals 8 packs have cracked structural enclosures with evidence of electrolyte leakage, while 32 packs are mechanically intact.
  - Standard freight for intact packs (P909) costs $\$0.45$ per ton-km, plus $\$60.00$ handling per pack.
  - Critical damaged packs (ADR SP 376 / LP904) require dedicated flatbed transport at $\$3.20$ per vehicle-km (carrying a maximum of 4 LP904 boxes per truck trip), plus certified LP904 single-use blast-containment packaging boxes costing $\$850.00$ each.
  Calculate the total logistics expenditure to transport all 40 packs.
- **Deliverable:** Step-by-step arithmetic derivation of total transport costs, broken down by intact versus critical cohorts.
- **Acceptance Criteria:** Exact final dollar figure matching required packaging and freight constraints.
- **Estimated Time:** 25 minutes.

### Exercise 3: Black Mass Payability and Refining Arbitrage
- **Type:** Quantitative Calculation
- **Problem Statement:** An industrial recycler processes 10 metric tons (10,000 kg) of decommissioned NMC811 battery packs.
  - Black mass yield is $44.0\%$ of pack mass.
  - Assays show the black mass contains: $24.0\%$ Nickel, $4.0\%$ Cobalt, $4.2\%$ Lithium, and $38.0\%$ Graphite.
  - Prevailing metal spot prices: Nickel = $\$18.00/\text{kg}$, Cobalt = $\$30.00/\text{kg}$, Lithium = $\$15.00/\text{kg}$.
  - Refiner payability rates: Nickel = $78\%$, Cobalt = $78\%$, Lithium = $65\%$, Graphite = $0\%$ (uncredited).
  - Recycler processing fee (shredding + chemical treatment): $\$950.00$ per metric ton of incoming pack mass.
  Calculate the net revenue generated from the black mass refining of this 10-ton batch.
- **Deliverable:** Mathematical calculation showing gross metal values, payability deductions, processing fees, and final net recovery.
- **Acceptance Criteria:** Exact net recovery figure derived with zero arithmetic errors.
- **Estimated Time:** 30 minutes.

### Exercise 4: Second-Life Re-Binning Optimization
- **Type:** Optimization & Algorithm Design
- **Problem Statement:** You have harvested 12 battery modules to assemble a 48 V stationary telecom backup string consisting of 4 modules in series ($4\text{S}1\text{P}$). The measured capacities (in Ah) of the 12 candidate modules are:
  $$\{92.1, 88.4, 95.0, 84.2, 91.8, 89.0, 94.5, 85.0, 92.5, 96.0, 87.8, 91.5\}$$
  If modules must be grouped such that the capacity spread ($\Delta Q = Q_{\text{max}} - Q_{\text{min}}$) within any 4-module string is minimized:
  1. Select the optimal 4 modules to form the premium String 1.
  2. Compute the usable capacity of String 1 assuming string capacity is bounded by the weakest cell.
  3. Calculate the percentage of stranded capacity across the other 3 modules in String 1.
- **Deliverable:** Module assignment table, usable capacity calculation, and stranded energy loss percentage.
- **Acceptance Criteria:** Correctly identifies the subset minimizing variance; precise mathematical execution of stranded capacity.
- **Estimated Time:** 30 minutes.

### Exercise 5: EU Battery Regulation Compliance Audit
- **Type:** Regulatory Critique
- **Problem Statement:** A European energy storage startup purchases 500 retired EV battery packs from various automotive dismantling yards across Germany and France. The startup de-lids the packs, replaces the BMS with their own proprietary controller, encases them in a new containerized BESS, and markets them to commercial solar parks. The startup claims that because they are "reusing existing clean assets," they are exempt from EU Regulation 2023/1542 CE marking, recycled content tracking, and Digital Battery Passport registration. Audit this operational plan against Articles 13, 77, and 78 of Regulation (EU) 2023/1542.
- **Deliverable:** 300-word regulatory compliance audit identifying specific legal violations, liability transfers, and required remediation steps.
- **Acceptance Criteria:** Cites manufacturer liability transfer upon repurposing; details Digital Battery Passport obligations.
- **Estimated Time:** 35 minutes.

### Exercise 6: Gateway Triage Engine Design
- **Type:** System Design
- **Problem Statement:** Design a high-throughput, semi-automated battery gateway diagnostic station capable of processing 20 heavy EV packs per 8-hour shift. The design must specify:
  1. Physical layout and safety interlocks (blast walls, thermal cameras, fire deluge).
  2. Sequence of electrical tests (order of OCV, isolation resistance, HPPC, EIS).
  3. Cycle time budget per test to meet the 24-minute-per-pack takt time.
  4. Decision criteria for routing packs to Tier 1, Tier 2, or Tier 3.
- **Deliverable:** Complete operational specification document including a takt time table and system workflow diagram.
- **Acceptance Criteria:** Takt time adds up to $\le 24$ minutes; safety protocols meet ISO 6469-1 and NFPA 855 standards.
- **Estimated Time:** 45 minutes.

---
### Exercise 7: Randles Equivalent Circuit Parameter Extraction
- **Type:** Quantitative Modeling & Parameter Identification
- **Problem Statement:** During a 10-second gateway discharge screening pulse ($I_{\text{pulse}} = 100.0\,\text{A}$) on a 400 V nominal battery pack at $50\%$ SoC ($V_{\text{ocv}} = 398.50\,\text{V}$):
  - At $t = 10\,\text{ms}$ after pulse onset, the measured terminal voltage drops instantaneously to $V(0.01\,\text{s}) = 392.50\,\text{V}$.
  - At $t = 2.0\,\text{s}$, the voltage reaches $V(2.0\,\text{s}) = 388.50\,\text{V}$.
  - At $t = 10.0\,\text{s}$ (end of pulse), the voltage settles to $V(10.0\,\text{s}) = 384.50\,\text{V}$.
  Assuming a first-order approximation where the fast drop from $0.01\,\text{s}$ to $2.0\,\text{s}$ represents charge-transfer polarization ($R_1, C_1$) with time constant $\tau_1 \approx 0.65\,\text{s}$ (such that $e^{-2.0/0.65} \approx 0.046$, meaning $R_1$ has reached $\approx 95.4\%$ of its steady-state overpotential), and the subsequent drop from $2.0\,\text{s}$ to $10.0\,\text{s}$ represents slower diffusion polarization ($R_2$):
  1. Calculate the high-frequency ohmic resistance $R_0$ of the pack in $\text{m}\Omega$.
  2. Estimate the charge-transfer resistance $R_1$ in $\text{m}\Omega$.
  3. Estimate the slower diffusion overpotential resistance $R_2$ at $t = 10\,\text{s}$ in $\text{m}\Omega$.
  4. Compute the total dynamic resistance $R_{\text{total}}(10\,\text{s})$.
- **Deliverable:** Step-by-step arithmetic derivation of $R_0, R_1, R_2$, and $R_{\text{total}}$.
- **Acceptance Criteria:** Exact resistance values matching the stated pulse measurements within $0.5\,\text{m}\Omega$.
- **Estimated Time:** 25 minutes.

### Exercise 8: LFP vs. NMC Recycling Break-Even and Arbitrage Parity
- **Type:** Economic Arbitrage & Comparative Analysis
- **Problem Statement:** A recycler evaluates two 500 kg battery packs: Pack 1 is NMC622 (contains 42% black mass: 20.5% Ni, 6.8% Co, 3.6% Li; copper scrap 7.5%, aluminum scrap 18.0%); Pack 2 is LFP (contains 38% black mass: 0% Ni, 0% Co, 3.2% Li, 0% payable Fe/P; copper scrap 8.0%, aluminum scrap 16.0%).
  - Spot prices: Ni = $\$16.50/\text{kg}$, Co = $\$28.00/\text{kg}$, Li = $\$14.00/\text{kg}$, Cu = $\$9.20/\text{kg}$, Al = $\$2.30/\text{kg}$.
  - Payabilities: Ni 75%, Co 75%, Li 65%, Cu 85%, Al 70%.
  - Both packs incur an inert shredding and chemical treatment fee of $\$0.85/\text{kg}$ of pack mass ($\$425.00$), and standard dangerous goods transport of $\$180.00$.
  1. Calculate the Net Recovery Value of recycling the LFP pack ($NRV_{\text{recycle, LFP}}$).
  2. Calculate the Net Recovery Value of recycling the NMC622 pack ($NRV_{\text{recycle, NMC}}$).
  3. Explain the commercial implications for circular disposition strategy: why is LFP under intense economic pressure to enter second-life BESS storage, while NMC faces strong recycling pull?
- **Deliverable:** Mathematical comparison of both packs' net cash yields and a concise economic explanation.
- **Acceptance Criteria:** Exact net recovery values for both packs; clear identification of the LFP recycling deficit.
- **Estimated Time:** 30 minutes.

## 13. Self-check answers

### Solution to Exercise 1: Degradation Mechanism Identification
- **Diagnosis:** The dominant physical degradation mechanism is **Loss of Lithium Inventory (LLI)**.
- **Electrochemical Reasoning:**
  - The $24\%$ drop in capacity indicates substantial loss of cyclable charge carriers.
  - The near-constant high-frequency ohmic resistance ($+4\%$) confirms that the bulk electrolyte, separator porosity, and current collector contact interfaces remain intact, ruling out severe contact corrosion or complete electrolyte dry-out.
  - In Incremental Capacity Analysis ($dQ/dV$), the area under the phase transition peaks is directly proportional to the amount of active cyclable lithium participating in the phase change. A $25\%$ shrinkage in peak area without a horizontal voltage shift proves that the active electrode host matrices (graphite and cathode crystal lattices) are still structurally available, but cyclable lithium has been permanently trapped (predominantly in the passive SEI layer).
- **Disposition Recommendation:** Because ohmic resistance is stable and active material is intact, the battery has not reached its knee point. However, because capacity has degraded by $24\%$ ($\text{SoH} = 76\%$), it falls just below the typical Tier 1 threshold ($\text{SoH} \ge 80\%$). It should be routed to **Tier 2: Module Harvesting and Re-Binning**, or deployed in a derated stationary application where high C-rate power density is not required.

### Solution to Exercise 2: ADR SP 376 Logistics Cost Evaluation
1. **Intact Cohort (32 packs):**
   - Total mass $= 32 \times 550\,\text{kg} = 17,600\,\text{kg} = 17.6\,\text{metric tons}$.
   - Freight cost $= 17.6\,\text{tons} \times 600\,\text{km} \times \$0.45/\text{ton-km} = \$4,752.00$.
   - Handling cost $= 32\,\text{packs} \times \$60.00 = \$1,920.00$.
   - Subtotal Intact $= \$4,752.00 + \$1,920.00 = \$6,672.00$.
2. **Critical Damaged Cohort (8 packs):**
   - Requires certified single-use LP904 boxes: $8\,\text{boxes} \times \$850.00 = \$6,800.00$.
   - Transport capacity is 4 boxes per truck trip. Number of trips $= 8 / 4 = 2\,\text{trips}$.
   - Freight cost per trip $= 600\,\text{km} \times \$3.20/\text{km} = \$1,920.00$.
   - Total freight cost for 2 trips $= 2 \times \$1,920.00 = \$3,840.00$.
   - Subtotal Critical $= \$6,800.00 + \$3,840.00 = \$10,640.00$.
3. **Total Combined Logistics Expenditure:**
   $$\text{Total Cost} = \$6,672.00 + \$10,640.00 = \mathbf{\$17,312.00}$$
   *Notice that the 8 critical packs account for $61.5\%$ of the total logistics cost despite representing only $20\%$ of the units.*

### Solution to Exercise 3: Black Mass Payability and Refining Arbitrage
1. **Black Mass Production:**
   $$\text{Black Mass} = 10,000\,\text{kg} \times 0.44 = 4,400.0\,\text{kg}$$
2. **Contained Metal Quantities:**
   - Nickel: $4,400 \times 0.24 = 1,056.0\,\text{kg}$
   - Cobalt: $4,400 \times 0.04 = 176.0\,\text{kg}$
   - Lithium: $4,400 \times 0.042 = 184.8\,\text{kg}$
3. **Gross Payability Value (Metal Spot $\times$ Payability):**
   - Nickel: $1,056.0\,\text{kg} \times \$18.00 \times 0.78 = \$14,826.24$
   - Cobalt: $176.0\,\text{kg} \times \$30.00 \times 0.78 = \$4,118.40$
   - Lithium: $184.8\,\text{kg} \times \$15.00 \times 0.65 = \$1,801.80$
   - Gross Recovered Value $= \$14,826.24 + \$4,118.40 + \$1,801.80 = \$20,746.44$
4. **Processing & Treatment Fee:**
   $$\text{Processing Fee} = 10.0\,\text{tons} \times \$950.00/\text{ton} = \$9,500.00$$
5. **Net Revenue from Refining:**
   $$\text{Net Revenue} = \$20,746.44 - \$9,500.00 = \mathbf{\$11,246.44}$$

### Solution to Exercise 4: Second-Life Re-Binning Optimization
1. **Sorting Available Modules:**
   Sort the 12 measured module capacities in ascending order:
   $$\{84.2, 85.0, 87.8, 88.4, 89.0, 91.5, 91.8, 92.1, 92.5, 94.5, 95.0, 96.0\}$$
   To minimize capacity spread ($\Delta Q = Q_{\text{max}} - Q_{\text{min}}$) across 4 contiguous modules, evaluate candidate 4-element windows:
   - Window [84.2, 88.4]: $\Delta Q = 4.2\,\text{Ah}$
   - Window [85.0, 89.0]: $\Delta Q = 4.0\,\text{Ah}$
   - Window [87.8, 91.5]: $\Delta Q = 3.7\,\text{Ah}$
   - Window [88.4, 91.8]: $\Delta Q = 3.4\,\text{Ah}$
   - Window [89.0, 92.1]: $\Delta Q = 3.1\,\text{Ah}$
   - Window [91.5, 92.5]: $\{91.5, 91.8, 92.1, 92.5\} \implies \Delta Q = 92.5 - 91.5 = \mathbf{1.0\,\text{Ah}}$
   - Window [91.8, 94.5]: $\Delta Q = 2.7\,\text{Ah}$
   
   **Optimal String 1 Modules:** $\{91.5\,\text{Ah}, 91.8\,\text{Ah}, 92.1\,\text{Ah}, 92.5\,\text{Ah}\}$.
2. **Usable String Capacity:**
   In a series string ($4\text{S}1\text{P}$), usable capacity is bounded by the minimum cell/module:
   $$Q_{\text{usable}} = \min(91.5, 91.8, 92.1, 92.5) = \mathbf{91.5\,\text{Ah}}$$
3. **Stranded Capacity Calculation:**
   - Total installed capacity $= 91.5 + 91.8 + 92.1 + 92.5 = 367.9\,\text{Ah}$.
   - Total delivered capacity $= 4 \times 91.5\,\text{Ah} = 366.0\,\text{Ah}$.
   - Stranded capacity $= 367.9 - 366.0 = 1.9\,\text{Ah}$.
   - Percentage stranded $= (1.9 / 367.9) \times 100 = \mathbf{0.516\%}$.
   *By tightly binning within a 1.0 Ah spread, stranded capacity is suppressed to near zero.*

### Solution to Exercise 5: EU Battery Regulation Compliance Audit Rubric
A passing response must detail:
1. **Article 77 & 78 (Repurposing and Second Life Liability Transfer):** Under the EU Battery Regulation, any legal entity that subjects a waste or retired battery to repurposing or remanufacturing operations is legally deemed a **new manufacturer**. The startup cannot claim exemption. They assume full product liability, must draw up new EU technical documentation, conduct conformity assessments, issue a Declaration of Conformity, and affix a new **CE mark**.
2. **Article 13 & 77 (Digital Battery Passport):** The startup must establish and register a new Digital Battery Passport linked via QR code on the repurposed BESS enclosure. They must update the baseline capacity to reflect the new "State of Certified Energy" ($\text{SOCE}$) and provide transparent degradation data.
3. **Safety & Diligence Violations:** Bypassing CE marking and UL 1974 / IEC 62619 safety verification exposes the startup to immediate stop-sale orders, product seizure, and catastrophic criminal liability in the event of a commercial fire.

### Solution to Exercise 6: Gateway Triage Engine Design Rubric
A passing system design must include:
1. **Takt Time Budget ($\le 24$ minutes):**
   - Station 1: Unloading, visual inspection, barcode scan, laser swelling audit: $4\,\text{min}$.
   - Station 2: Automated high-voltage busbar contact and isolation resistance test ($R_{\text{iso}}$): $3\,\text{min}$.
   - Station 3: Automated 30-second HPPC pulse test (10s discharge, 10s rest, 10s charge): $5\,\text{min}$ (including automated pneumatic contact clamping).
   - Station 4: Multi-frequency EIS sweep ($1\,\text{kHz}$ to $0.1\,\text{Hz}$): $8\,\text{min}$.
   - Station 5: Automated decision routing and conveyor transfer: $4\,\text{min}$.
   - Total Cycle Time $= 24\,\text{minutes}$.
2. **Safety Architecture:** Reinforced concrete blast separation bays; overhead dual-wavelength infrared thermography with automated abort interlocks; floor-level emergency roll-out conveyor leading to an external water immersion quench tank in case of thermal excursion ($dT/dt > 2^\circ\text{C}/\text{min}$).

---
### Solution to Exercise 7: Randles Equivalent Circuit Parameter Extraction
1. **Ohmic Resistance ($R_0$):**
   The instantaneous voltage step occurs between $t = 0$ ($V_{\text{ocv}} = 398.50\,\text{V}$) and $t = 10\,\text{ms}$ ($V = 392.50\,\text{V}$):
   $$\Delta V_0 = 398.50\,\text{V} - 392.50\,\text{V} = 6.00\,\text{V}$$
   $$R_0 = \frac{\Delta V_0}{I_{\text{pulse}}} = \frac{6.00\,\text{V}}{100.0\,\text{A}} = 0.060\,\Omega = \mathbf{60.0\,\text{m}\Omega}$$
2. **Charge-Transfer Resistance ($R_1$):**
   Between $t = 10\,\text{ms}$ and $t = 2.0\,\text{s}$, voltage drops from $392.50\,\text{V}$ to $388.50\,\text{V}$:
   $$\Delta V_1 = 392.50\,\text{V} - 388.50\,\text{V} = 4.00\,\text{V}$$
   Since the first RC network has reached $\approx 95.4\%$ of its steady state ($1 - e^{-2.0/0.65} = 1 - 0.046 = 0.954$):
   $$I_{\text{pulse}} \cdot R_1 \approx \frac{4.00\,\text{V}}{0.954} = 4.193\,\text{V} \implies R_1 = \frac{4.193\,\text{V}}{100.0\,\text{A}} = 0.04193\,\Omega = \mathbf{41.93\,\text{m}\Omega}$$
3. **Diffusion Polarization Resistance ($R_2$ at $10\,\text{s}$):**
   Between $t = 2.0\,\text{s}$ and $t = 10.0\,\text{s}$, voltage drops from $388.50\,\text{V}$ to $384.50\,\text{V}$:
   $$\Delta V_2 = 388.50\,\text{V} - 384.50\,\text{V} = 4.00\,\text{V}$$
   $$R_2(10\,\text{s}) = \frac{\Delta V_2}{I_{\text{pulse}}} = \frac{4.00\,\text{V}}{100.0\,\text{A}} = 0.040\,\Omega = \mathbf{40.0\,\text{m}\Omega}$$
4. **Total Dynamic Resistance ($R_{\text{total}}$ at $10\,\text{s}$):**
   $$\Delta V_{\text{total}} = 398.50\,\text{V} - 384.50\,\text{V} = 14.00\,\text{V}$$
   $$R_{\text{total}} = \frac{14.00\,\text{V}}{100.0\,\text{A}} = 0.140\,\Omega = \mathbf{140.0\,\text{m}\Omega}$$
   *(Note: $R_0 + \Delta V_1/I + \Delta V_2/I = 60.0 + 40.0 + 40.0 = 140.0\,\text{m}\Omega$.)*

### Solution to Exercise 8: LFP vs. NMC Recycling Break-Even and Arbitrage Parity
1. **LFP Pack (500 kg) Recycling Economics:**
   - Black Mass mass $= 500 \times 0.38 = 190.0\,\text{kg}$.
   - Contained Lithium $= 190.0 \times 0.032 = 6.08\,\text{kg}$.
   - Payable Lithium Value $= 6.08\,\text{kg} \times \$14.00 \times 0.65 = \$55.33$.
   - Contained Copper $= 500 \times 0.08 = 40.0\,\text{kg}$.
   - Payable Copper Value $= 40.0\,\text{kg} \times \$9.20 \times 0.85 = \$312.80$.
   - Contained Aluminum $= 500 \times 0.16 = 80.0\,\text{kg}$.
   - Payable Aluminum Value $= 80.0\,\text{kg} \times \$2.30 \times 0.70 = \$128.80$.
   - **Gross Recoverable Value (LFP):** $\$55.33 + \$312.80 + \$128.80 = \$496.93$.
   - **Deductions:** Processing fee ($500 \times \$0.85 = \$425.00$) + Freight ($\$180.00$) $= \$605.00$.
   - **Net Recovery Value ($NRV_{\text{recycle, LFP}}$):**
     $$NRV_{\text{recycle, LFP}} = \$496.93 - \$605.00 = \mathbf{-\$108.07\,\text{(A net loss of \$108.07 per pack!)}}$$
2. **NMC622 Pack (500 kg) Recycling Economics:**
   - Black Mass mass $= 500 \times 0.42 = 210.0\,\text{kg}$.
   - Nickel: $210 \times 0.205 = 43.05\,\text{kg} \times \$16.50 \times 0.75 = \$532.74$.
   - Cobalt: $210 \times 0.068 = 14.28\,\text{kg} \times \$28.00 \times 0.75 = \$299.88$.
   - Lithium: $210 \times 0.036 = 7.56\,\text{kg} \times \$14.00 \times 0.65 = \$68.80$.
   - Copper: $500 \times 0.075 = 37.5\,\text{kg} \times \$9.20 \times 0.85 = \$293.25$.
   - Aluminum: $500 \times 0.18 = 90.0\,\text{kg} \times \$2.30 \times 0.70 = \$144.90$.
   - **Gross Recoverable Value (NMC):** $\$532.74 + \$299.88 + \$68.80 + \$293.25 + \$144.90 = \$1,339.57$.
   - **Deductions:** Processing fee ($\$425.00$) + Freight ($\$180.00$) $= \$605.00$.
   - **Net Recovery Value ($NRV_{\text{recycle, NMC}}$):**
     $$NRV_{\text{recycle, NMC}} = \$1,339.57 - \$605.00 = \mathbf{+\$734.57\,\text{per pack}}$$.
3. **Strategic Circular Implications:**
   - For **LFP**, recycling generates a negative cash flow ($-\$108.07$), meaning the battery owner or OEM must pay a "gate fee" / tipping fee to the recycler to process it. Conversely, LFP possesses a massive cycle life ($>4,000$ cycles). Therefore, LFP is under overwhelming economic pressure to enter **Second-Life BESS**, where its usable energy can be sold at $\$70	ext{--}\$100/	ext{kWh}$.
   - For **NMC**, recycling yields a handsome positive profit ($+\$734.57$ per pack) purely from contained nickel, cobalt, and copper. This creates an intense competitive threshold: a second-life BESS deployment must deliver risk-adjusted net margins well above $\$735$ to justify holding the asset, performing module teardown, and underwriting warranty liability.

## 14. Interview questions

### Question 1: Electrochemical Degradation Diagnostics
- **Prompt:** "How can a reverse logistics engineering team differentiate between Loss of Lithium Inventory (LLI) and Loss of Active Material (LAM) during rapid incoming battery inspection, and why does this distinction dictate second-life disposition?"
- **Strong Answer:** Differentiating LLI and LAM without destructive teardown relies on non-destructive differential electrochemical techniques—specifically Incremental Capacity Analysis ($dQ/dV$) and Differential Voltage Analysis ($dV/dQ$). LLI is characterized by a proportional shrinkage in peak area across phase transitions without a horizontal shift along the voltage axis, indicating that active lithium ions have been irreversibly consumed by continuous Solid Electrolyte Interphase (SEI) growth or lithium plating. LAM is characterized by asymmetric shrinkage of specific peaks corresponding to either the positive cathode crystal degradation ($LAM_{\text{PE}}$) or negative graphite anode particle cracking ($LAM_{\text{NE}}$), often accompanied by peak broadening and voltage shifts reflecting internal resistance growth. The distinction is critical for disposition: batteries suffering primarily from LLI with low impedance can operate safely and reliably in low C-rate stationary storage (Tier 1 BESS), whereas batteries suffering from severe LAM exhibit structural lattice instability, localized current crowding, and an imminent transition to the catastrophic degradation knee point, mandating immediate routing to hydrometallurgical recycling.
- **Red Flag / Weak Answer:** Confusing LLI with simple State of Charge, stating that resistance alone reveals the degradation mechanism, or claiming that visual inspection can identify internal crystal cracking.

### Question 2: Dangerous Goods Transport Regulations
- **Prompt:** "Under ADR and US DOT 49 CFR, what specific physical and electrical conditions classify a returned lithium-ion battery under Special Provision 376 as 'Critical Damaged/Defective', and what are the logistics cost implications?"
- **Strong Answer:** Under ADR SP 376 and 49 CFR § 173.185(f), a battery is classified as Critical Damaged/Defective if, under normal transport conditions, it is prone to enter rapid thermal runaway, produce toxic gases, or ignite. Specific diagnostic criteria include: visible pouch or prismatic swelling ($>8\text{--}10\%$), breached cell hermetic seals, electrolyte leakage or strong organic carbonate odor, physical enclosure penetration from vehicle collisions, persistent internal short-circuit DTCs with cell voltage below $1.5\,\text{V}$, or an onboard record of thermal runaway excursions ($T > 60^\circ\text{C}$). The logistics implications are severe: the battery cannot move via standard freight (P903/P909). It must be packaged under Packaging Instruction P908 or Large Packaging LP904 in certified Group I explosion-proof containers filled with non-combustible, pyro-absorbent cushioning (vermiculite or glass bubbles) with flame-arresting pressure vents. Packaging costs jump from $\$150$ to $\$800\text{--}\$1,200$ per container, transport is limited to certified ADR carriers with dedicated single-tier flatbeds, and air transport is strictly prohibited, increasing unit logistics costs by $300\%$ to $600\%$.
- **Red Flag / Weak Answer:** Believing any used or degraded battery is automatically critical, or claiming that wrapping a damaged battery in standard plastic wrap inside a cardboard box complies with dangerous goods regulations.

### Question 3: The Degradation Knee Point
- **Prompt:** "Explain the physical origin of the 'knee point' in lithium-ion battery aging curves, and explain how a reverse network optimization model should incorporate knee-point risk."
- **Strong Answer:** The knee point is the sudden non-linear transition from steady, linear capacity loss into rapid exponential capacity fade and impedance growth. Physically, it is triggered by the interplay between mechanical and electrochemical phenomena. During early life, degradation is dominated by sub-linear calendar SEI growth ($\sim t^{0.5}$). Over extended cycling, continuous volume changes cause micro-cracking of graphite and cathode particles, exposing fresh surfaces that consume cyclable lithium. Eventually, the effective negative-to-positive capacity ratio ($N/P$ ratio) drops locally below unity, causing the anode potential to drop to $0.0\,\text{V}$ versus $Li/Li^+$ during charging. This forces metallic lithium to plate directly onto the graphite surface rather than intercalating. Lithium dendrites accelerate SEI formation, block pores, starve the cell of electrolyte, and dramatically spike internal resistance. A reverse logistics optimization model must incorporate knee-point risk by enforcing a safety margin threshold on remaining useful life: rather than using a static cutoff (e.g., $\text{SoH} \ge 70\%$), the model must evaluate the second derivative of capacity loss ($d^2Q/dN^2$) and resistance growth rate ($dR/dN$). Packs within a defined margin of this inflection must be excluded from BESS deployments to avoid massive field replacement and warranty default costs.
- **Red Flag / Weak Answer:** Describing battery degradation as purely linear from $100\%$ down to $0\%$, or asserting that the knee point is caused by external software throttling.

### Question 4: Module-Level Re-Binning Economics
- **Prompt:** "Why do many industrial recyclers prefer to shred whole battery packs into black mass rather than harvesting intact modules for second-life applications, even when modules retain $>75\%$ SoH?"
- **Strong Answer:** Industrial operators face a trade-off between labor/disassembly friction, warranty liability, and metal commodity economics. Dismantling a pack to module level is labor-intensive, hazardous, and difficult to automate due to the hundreds of proprietary, non-standardized pack designs across OEMs. Laser-welded busbars require delicate mechanical or laser de-welding that yields an $8\%\text{--}15\%$ cell damage rate. Furthermore, harvested modules exhibit heterogeneous degradation; assembling them into a reliable stationary string requires extensive grading test time and strict re-binning ($\Delta \text{SoH} \le \pm 2\%$) to prevent premature string cutoff. The remanufacturer must then carry warranty and safety liability under UL 1974. Conversely, direct mechanical shredding and hydrometallurgical recycling is a continuous, automated process. When commodity prices for nickel, cobalt, and lithium are elevated, the immediate, unencumbered Net Recovery Value ($NRV$) from selling black mass salts can match or exceed the discounted, risk-adjusted margin of repurposed modules without carrying long-term field liability.
- **Red Flag / Weak Answer:** Assuming that remanufacturing is always more profitable than recycling, or failing to identify laser-welded busbars and non-standardized pack architectures as major economic bottlenecks.

### Question 5: EU Battery Passport Architecture
- **Prompt:** "What is the Digital Battery Passport mandated by EU Regulation 2023/1542, and how does it alter data exchange requirements across the forward and reverse supply chain?"
- **Strong Answer:** The Digital Battery Passport (DBP) is an interoperable electronic record linked to a unique QR code on all EV and industrial batteries $>2\,\text{kWh}$ placed on the EU market, taking effect in February 2027. It creates a cryptographically verifiable data thread spanning raw material sourcing, manufacturing, first-life operation, and circular disposition. Across the reverse chain, the DBP mandates access tiers: public data (general battery specifications, chemistry, carbon footprint), commercial stakeholder data (dismantling manuals, safety handling instructions, component teardown schematics for recyclers), and regulatory/audit data (exact recycled content percentages, supply chain due diligence reports). Crucially, the DBP requires dynamic data updates: the BMS must log State of Health, internal resistance, cycle throughput, and thermal history. When a battery is repurposed, the second-life operator must assume legal manufacturer status, record the new State of Certified Energy ($\text{SOCE}$), and re-register the passport. This eliminates information asymmetry during gateway triage.
- **Red Flag / Weak Answer:** Viewing the battery passport as a static PDF or simple paper certificate, or failing to recognize that repurposing requires updating the passport under new manufacturer liability.

### Question 6: Black Mass Metallurgical Valuation
- **Prompt:** "Walk me through the mathematical valuation of a metric ton of crude black mass derived from NMC622 cells. What are the key deductions between gross metal content and net cash realization?"
- **Strong Answer:** Valuation begins with a chemical assay to determine the dry mass percentages of payable metals: typically $\sim 20\%\,\text{Ni}$, $\sim 6.5\%\,\text{Co}$, and $\sim 3.5\%\,\text{Li}$ for NMC622. Multiplying these fractions by 1,000 kg yields the physical contained metal. Next, we multiply each contained quantity by its prevailing commodity spot price (LME for nickel and cobalt; Fastmarkets or S&P Platts for battery-grade lithium carbonate/hydroxide). We then apply **payability factors**, which reflect refiner recovery yields and market demand (typically $70\%\text{--}80\%$ for Ni and Co; $60\%\text{--}70\%$ for Li; $0\%$ for manganese and graphite). Summing these gives the Gross Payable Metal Value. Finally, we subtract critical deductions:
  1. Moisture and volatile content deductions (if black mass is assayed wet).
  2. Penalties for deleterious impurities (especially fluorine $>1.0\%$, copper $>0.5\%$, and aluminum $>1.5\%$).
  3. Treatment and Refining Charges (TC/RC), representing the refiner's chemical processing, acid consumption, and solvent extraction operating costs per ton.
  The remaining balance is the Net Cash Realization.
- **Red Flag / Weak Answer:** Assuming the recycler receives 100% of the commodity spot price for all contained elements, or omitting refiner payability discounts and treatment charges.

### Question 7: Thermal Runaway Propagation Mechanics
- **Prompt:** "What is the chain of physical reactions that constitutes thermal runaway in a lithium-ion battery, and what specific engineering controls are required in a reverse staging warehouse?"
- **Strong Answer:** Thermal runaway is an uncontrollable, exothermic self-heating reaction. It initiates when internal cell temperature reaches $80\text{--}120^\circ\text{C}$, causing the Solid Electrolyte Interphase (SEI) layer to decompose exothermically. With the passivation layer gone, exposed intercalated lithium in the graphite anode reacts directly with the organic carbonate electrolyte solvent, producing flammable gases ($C_2H_4, CH_4$) and driving temperatures past $130\text{--}150^\circ\text{C}$, at which point the polymer separator (PE/PP) melts, causing a massive hard internal short-circuit. Temperatures spike above $200^\circ\text{C}$, triggering cathode crystal breakdown, which releases gaseous oxygen ($O_2$) directly into the solvent-rich, high-temperature environment. This leads to explosive combustion, venting toxic hydrofluoric acid ($HF$) and carbon monoxide ($CO$), with temperatures exceeding $900^\circ\text{C}$. This heat transfers to adjacent cells, triggering cascading pack-level propagation. Warehouse controls mandate: NFPA 855-compliant separation distances; fire barrier blast walls; continuous dual-band infrared thermography; negative-pressure exhaust systems with scrubbers; and automated submerged water quench tanks capable of rapidly cooling packs below the runaway threshold.
- **Red Flag / Weak Answer:** Stating that lithium-ion battery fires are simple electrical fires that can be extinguished with standard Class A or dry powder fire extinguishers, or failing to understand that the cathode generates its own oxygen during runaway.

### Question 8: Impedance Diagnostics: EIS vs. Pulse Resistance
- **Prompt:** "Compare Electrochemical Impedance Spectroscopy (EIS) and Hybrid Pulse Power Characterization (HPPC) for industrial battery triage. When would you deploy each in a reverse supply chain network?"
- **Strong Answer:** HPPC measures DC overpotential by applying high-current square-wave pulses (e.g., 1C for 10s–30s) and measuring the instantaneous ohmic step ($\Delta V_0$) and time-dependent diffusion relaxation ($\Delta V_p$). It is computationally simple, robust, fast (taking under 1 minute), and directly reflects the battery's real-world power delivery and total internal resistance ($R_{\text{total}}$). However, it lumps multiple degradation mechanisms together. EIS applies small AC sinusoidal perturbations across a broad frequency domain ($10\,\text{kHz}$ to $0.01\,\text{Hz}$), deconvolving the total impedance into distinct electrochemical phenomena: high-frequency ohmic electrolyte resistance ($R_s$), mid-frequency SEI layer impedance ($R_{\text{sei}}$) and charge transfer resistance ($R_{\text{ct}}$), and low-frequency solid-state Warburg diffusion ($Z_w$). However, EIS requires precise, noise-isolated laboratory instrumentation and takes several minutes per cell/module. In a reverse supply chain network, HPPC is deployed at high-volume **Gateway Triage Nodes** for rapid, automated pass/fail screening. EIS is deployed at **Secondary Diagnostic & Remanufacturing Hubs** to perform root-cause failure analysis, detect localized lithium plating, and accurately match modules for precision second-life re-binning.
- **Red Flag / Weak Answer:** Claiming that EIS can be run in seconds on a moving conveyor belt, or claiming that a simple DC pulse provides insight into double-layer capacitance and SEI layer thickness.

---

## 15. Key terms

1. **State of Health (SoH):** Ratio of current usable discharge capacity to original factory rated nominal capacity ($Q_{\text{actual}} / Q_{\text{nominal}}$).
2. **Knee Point:** Non-linear inflection point in battery life where capacity fade shifts from sublinear calendar aging to rapid exponential decay.
3. **Loss of Lithium Inventory (LLI):** Permanent entrapment of cyclable lithium ions due to side reactions like SEI growth or metallic plating.
4. **Loss of Active Material (LAM):** Physical breakdown or isolation of electrode host materials via particle micro-cracking and delamination.
5. **Solid Electrolyte Interphase (SEI):** Passivation film on the graphite anode formed by electrolyte reduction; critical for stability but causes calendar LLI.
6. **Lithium Plating:** Deposition of metallic lithium on the anode surface during fast charging or low-temperature operation, posing dendrite short-circuit risks.
7. **Hybrid Pulse Power Characterization (HPPC):** Rapid DC test protocol evaluating instantaneous ohmic resistance and polarization impedance via current pulses.
8. **Electrochemical Impedance Spectroscopy (EIS):** AC frequency-sweep technique separating bulk ohmic, SEI, charge-transfer, and diffusion impedances.
9. **Nyquist Plot:** Argand diagram plotting imaginary impedance against real impedance across a frequency spectrum.
10. **Incremental Capacity Analysis (ICA):** Differential curve ($dQ/dV$ vs. $V$) identifying electrode phase changes and quantifying LLI versus LAM.
11. **Differential Voltage Analysis (DVA):** Inverted differential curve ($dV/dQ$ vs. $Q$) used to track individual electrode voltage behavior.
12. **Thermal Runaway:** Self-sustaining, exothermic chemical cascade releasing intense heat ($>900^\circ\text{C}$), toxic gases, and flammable solvents.
13. **Black Mass:** Concentrated powder recovered from mechanical shredding of cells containing cathode oxides, anode graphite, and residual salts.
14. **Hydrometallurgical Recycling:** Aqueous chemical leaching and solvent extraction reclaiming battery-grade metal salts at $>95\%$ efficiency.
15. **Pyrometallurgical Recycling:** High-temperature smelting process recovering nickel, cobalt, and copper matte while slagging lithium and aluminum.
16. **Payability:** Commercial percentage of prevailing metal spot price credited to scrap suppliers after refiner margin deductions.
17. **Treatment & Refining Charges (TC/RC):** Processing fees deducted by metallurgical refiners to convert crude black mass into chemical-grade salts.
18. **UN 3480:** United Nations Class 9 Dangerous Goods transport designation for standalone lithium-ion battery cells and packs.
19. **UN 3481:** UN Class 9 Dangerous Goods designation for lithium-ion batteries packed within or contained inside equipment.
20. **ADR Special Provision 376 (SP 376):** Road transport regulation governing damaged, defective, or leaking batteries prone to thermal runaway.
21. **Packaging Instruction P908 / LP904:** UN certified hazardous packaging specifications requiring explosion-proof, flame-arresting containers.
22. **ADR Special Provision 377 (SP 377):** Transport rules permitting carriage of non-critical waste lithium batteries for recycling under P909.
23. **Digital Battery Passport (DBP):** Interoperable digital lifecycle record mandated by EU Regulation 2023/1542 tracking provenance, SoH, and carbon footprint.
24. **State of Certified Energy (SOCE):** Legal metric established upon battery remanufacturing certifying the new usable energy content of a repurposed asset.
25. **Second-Life Battery Energy Storage System (BESS):** Stationary energy storage installation constructed from repurposed automotive battery packs or modules.
26. **UL 1974:** American National Standard governing sorting, grading, evaluating, and remanufacturing batteries for secondary use.
27. **IEC 62619:** International safety standard for secondary lithium cells and batteries used in industrial and stationary applications.
28. **Design for Disassembly (DfD):** Engineering methodology prioritizing bolted joints, accessible busbars, and standardized pack geometries to facilitate circular recovery.
29. **Stranded String Capacity:** Unusable energy left in healthy series-connected battery modules when the weakest module trips the BMS low-voltage cutoff.
30. **Inert Shredding:** Mechanical comminution executed under nitrogen, carbon dioxide, or argon gas blankets to prevent solvent-vapor explosion.

---

## 16. Sources

- **Birkl, C. R., Roberts, M. R., McTurk, E., Bruce, P. G., & Howey, D. A. (2017).** Degradation diagnostics for lithium ion cells. *Journal of Power Sources*, 341, 373–386. [DOI: 10.1016/j.jpowsour.2016.12.011]
- **European Parliament and Council of the European Union. (2023).** *Regulation (EU) 2023/1542 of 12 July 2023 concerning batteries and waste batteries, amending Directive 2008/98/EC and Regulation (EU) 2019/1020, and repealing Directive 2006/66/EC*. Official Journal of the European Union, L 191, 1–85.
- **Gaines, L. (2014).** The future of automotive lithium-ion battery recycling: Charting a sustainable course. *Sustainable Materials and Technologies*, 1–2, 2–7. [DOI: 10.1016/j.susmat.2014.10.001]
- **Martinez-Laserna, E., Gandiaga, I., Sarasketa-Zabala, E., Badeda, J., Stroe, D. I., Swierczynski, M., & Zhu, P. (2018).** Battery second life: Hype, hope or reality? A critical review of the state of the art and second-life battery applications. *Renewable and Sustainable Energy Reviews*, 93, 701–718. [DOI: 10.1016/j.rser.2018.04.035]
- **United Nations Economic Commission for Europe (UNECE). (2023).** *Agreement concerning the International Carriage of Dangerous Goods by Road (ADR 2023)*. ECE/TRANS/326, United Nations Publications, Geneva.
- **Underwriters Laboratories. (2018).** *ANSI/CAN/UL 1974: Standard for Evaluation for Repurposing Batteries*. UL Standards & Engagement, Northbrook, IL.
- **United States Department of Transportation (US DOT), Pipeline and Hazardous Materials Safety Administration (PHMSA). (2024).** *Title 49 of the Code of Federal Regulations (49 CFR) § 173.185: Lithium cells and batteries*. US Government Publishing Office.
- **International Electrotechnical Commission (IEC). (2022).** *IEC 62619: Secondary cells and batteries containing alkaline or other non-acid electrolytes — Safety requirements for secondary lithium cells and batteries, for use in industrial applications*. Edition 2.0, IEC, Geneva.
- **Redwood Materials. (2023).** *Closed-Loop Battery Recycling and Cathode Material Manufacturing: Annual Circularity Report*. Carson City, NV.
- **Sumitomo Corporation & Nissan Motor Co., Ltd. (2021).** *4R Energy: Decennial Technical Review of Second-Life Automotive Battery Deployment in Distributed Grid Storage*. Yokohama, Japan.

---

## 17. Links

- **Parent Module Index:** [`_index.md`](_index.md)
- **Sibling Topic 01 (Reverse Flows and Disposition):** [`01_reverse-flows-disposition.md`](01_reverse-flows-disposition.md)
- **Sibling Topic 02 (Return Unit Economics):** [`02_returns-unit-economics.md`](02_returns-unit-economics.md)
- **Sibling Topic 03 (Grading and Recovery Yield):** [`03_grading-yield.md`](03_grading-yield.md)
- **Sibling Topic 04 (Reverse Network Design):** [`04_reverse-network-design.md`](04_reverse-network-design.md)
- **Sibling Topic 05 (Refurbishment and Remanufacturing):** [`05_refurbishment-remanufacturing.md`](05_refurbishment-remanufacturing.md)
- **Sibling Topic 07 (Packaging Reuse and Deposit Return):** [`07_packaging-reuse-deposit.md`](07_packaging-reuse-deposit.md)
- **Sibling Topic 09 (Circular Metrics and Regulation):** [`09_circular-metrics-regulation.md`](09_circular-metrics-regulation.md)
- **Root Survey File:** [`../../../course/37_circular_economy_reverse_networks.md`](../../../course/37_circular_economy_reverse_networks.md)
- **Related Module (Spare Parts & Aftermarket):** [`../../05_labs_electives/31_aftermarket_spare_parts_service/_index.md`](../../05_labs_electives/31_aftermarket_spare_parts_service/_index.md)
