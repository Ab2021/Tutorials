# Dangerous Goods Logistics Engineering, Multi-Modal Hazmat Segregation, and Port Security Operations

> Module 41 — Maritime, Rail, and Air Cargo Operations Engineering
> Topic 09 of 10
> Estimated study time: 4.5 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 12 (Transportation Routing & Last Mile), Module 20 (Operations Research & Mathematical Programming), Topic 41.01 (Berth and Quay Crane Operations), Topic 41.02 (Terminal Yard and Gate), Topic 41.04 (Rail Yards and Shunting), Topic 41.05 (Air Cargo and ULD)
> Version 2026.5

*Multi-modal dangerous goods physics, IMO IMDG Code container vessel stowage segregation matrices, cellular separation distances, blast wave and vapor dispersion mechanics, IATA DGR air cargo constraints, NOTOC protocols, dry ice sublimation ventilation kinetics, 49 CFR Part 174 rail buffer car positioning optimization, ISPS Code and 33 CFR Part 105 MARSEC escalation dynamics, and Non-Intrusive Inspection (NII) dual-energy radiographic material discrimination algorithms.*

---

## 1. Why this topic exists

In global multi-modal transportation networks, Dangerous Goods (DG) and hazardous materials (hazmat) represent approximately 10% to 15% of all containerized maritime traffic, over 20% of North American rail ton-miles, and millions of time-critical air freight shipments annually. Modern industrial societies depend fundamentally on the uninterrupted distribution of volatile chemical precursors, compressed gases, flammable fuels, toxic reagents, corrosive acids, and high-energy materials. Semiconductor fabrication plants cannot operate without continuous supplies of ultra-pure silane, phosphine, and hydrofluoric acid. The global transition toward electric mobility and grid-scale storage requires moving millions of reactive lithium-ion battery cells across oceans and continents. Advanced life science therapies, viral vaccines, and biological specimens depend on continuous cryogenic distribution using dry ice. Heavy industry and agriculture rely on massive shipments of chlorine, anhydrous ammonia, and ammonium nitrate fertilizers.

Because these materials possess inherent chemical, radiological, and thermodynamic reactivity, a containment loss or uncontrolled chemical reaction is rarely localized. When hazardous materials fail in transit, the consequences scale non-linearly into catastrophic disasters that destroy capital assets, halt global supply corridors, and cause severe loss of life:

1. **Port Terminal Catastrophes:** On August 4, 2020, 2,750 metric tons of improperly stored ammonium nitrate detonated in the Port of Beirut, killing 218 people, injuring over 7,000, causing more than $15 billion in physical infrastructure destruction, and obliterating the grain silos that held 85% of Lebanon's grain reserves. On August 12, 2015, an uncontrolled fire at a container storage facility in the Port of Tianjin triggered the detonation of 800 metric tons of ammonium nitrate and nitrocellulose, killing 173 people, destroying 304 buildings and 12,428 vehicles, and generating $1.1 billion in direct insured losses.
2. **Containership Mid-Ocean Disasters:** On July 14, 2012, the 6,750 TEU container ship *MSC Flaminia* suffered an explosion in Cargo Hold 4 while crossing the North Atlantic. The explosion was triggered by the autocatalytic thermal polymerization of divinylbenzene (Class 9 / UN 1993), which had been stowed under-deck adjacent to heated heavy fuel oil (HFO) bunker tanks. The resulting fire burned for months, killed three crew members, and forced the total constructive loss and rebuilding of the vessel midsection.
3. **Rail Hazardous Consist Derailments:** On February 3, 2023, a 149-car Norfolk Southern freight train derailed 38 cars in East Palestine, Ohio. Eleven of the derailed tank cars carried hazardous materials, including five carrying 115,000 gallons of vinyl chloride monomer (UN 1086, Class 2.1). Fearing catastrophic boiling liquid expanding vapor explosions (BLEVE), incident commanders initiated an emergency manual breach and open-air flare burn, releasing massive plumes of phosgene and hydrogen chloride gas across the surrounding ecosystem and incurring cleanup and settlement liabilities exceeding $2.1 billion.
4. **Air Cargo In-Flight Hull Losses:** On September 3, 2010, UPS Airlines Flight 6, a Boeing 747-400 freighter, crashed shortly after departure from Dubai following an uncontrolled main-deck cargo fire initiated by thousands of improperly packed lithium-ion batteries. Dense toxic smoke entered the cockpit within three minutes, blinding the flight crew and disabling emergency life-support systems. Similar thermal runaway events destroyed Asiana Airlines Flight 991 (Boeing 747-400F) in July 2011.
5. **National Security and WMD Smuggling Vulnerabilities:** Maritime container ports process over 800 million TEUs annually. An unmonitored container provides a vector for the illicit introduction of Special Nuclear Material (SNM), dirty bombs (radiological dispersal devices), chemical agents, or energetic weapons into metropolitan centers. Under the SAFE Port Act of 2006 and the International Ship and Port Facility Security (ISPS) Code, global terminals must maintain layered security perimeters, credentialing architectures, and high-throughput Non-Intrusive Inspection (NII) radiographic screening.

The operations engineering problem is defined by four distinct operational stakeholders who face conflicting throughput and safety objectives:
- **The Vessel Stowage Planner:** Must maximize slot utilization and crane intensity on a 20,000 TEU vessel while strictly enforcing the combinatorial constraints of the IMO IMDG Code segregation matrix, ensuring incompatible oxidizers and flammables never share adjacent cell spaces, bulkheads, or holds.
- **The Flight Dispatcher and Loadmaster:** Must optimize high-yield cargo payloads while ensuring aircraft cargo hold dry ice sublimation rates do not overwhelm fresh air ventilation systems, preventing toxic hypercapnia in the cockpit and passenger cabin.
- **The Freight Rail Yardmaster:** Must assemble 10,000-ton manifest trains within tight yard departure windows while satisfying US 49 CFR Part 174 buffer car rules, inserting sacrificial non-hazardous railcars to protect locomotive crews from kinetic collision forces and chemical fireballs.
- **The Port Facility Security Officer (PFSO):** Must safeguard terminal perimeters against physical and asymmetric threats under the ISPS Code, managing vehicle queues and gate processing capacity when national authorities escalate security levels from MARSEC 1 to MARSEC 2 or 3.

Safety cannot be achieved by banning hazardous goods; modern industrial manufacturing would cease within days. The operations engineering mandate is therefore deterministic: supply chain systems must enforce strict mathematical separation distances, thermodynamic heat transfer limits, mechanical impact buffering, and automated radiographic anomaly detection across every modal handoff.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Classify and map** hazardous materials across UN Hazard Classes 1 through 9, Packing Groups I, II, and III, and explosive compatibility groups across the UN Model Regulations, IMO IMDG Code, ICAO/IATA DGR, and US 49 CFR.
2. **Formulate and solve** container vessel 3D bay-row-tier segregation matrices under IMDG Code Chapter 7.2, evaluating on-deck and under-deck cellular separation requirements ("Away from", "Separated from", "Separated by complete compartment", and "Separated longitudinally by intervening compartment").
3. **Model and calculate** aircraft cargo compartment dry ice ($CO_2$, UN 1845) sublimation kinetics, determining maximum permissible gross dry ice mass based on aircraft cabin ventilation rates, compartment volumes, and physiological $CO_2$ threshold limits.
4. **Validate and audit** IATA Dangerous Goods Notification to Captain (NOTOC) manifests, determining package net quantity limits, Cargo Aircraft Only (CAO) versus Passenger Aircraft (PAX) eligibility, and emergency response drill codes.
5. **Formulate mixed-integer linear programming (MILP)** constraints for freight train consist assembly under US 49 CFR Part 174, optimizing buffer car placement relative to lead locomotives, occupied cabooses/DPUs, and mechanical refrigeration ignition sources.
6. **Quantify port facility security queueing dynamics** under the ISPS Code and US 33 CFR Part 105, modeling terminal gate capacity degradation and truck dwell time divergence under MARSEC Level 1, 2, and 3 escalation protocols.
7. **Implement dual-energy radiographic Non-Intrusive Inspection (NII)** algorithms using the Beer-Lambert attenuation law to calculate transmission ratios and reconstruct effective atomic numbers ($Z_{\text{eff}}$) for automated cargo material discrimination.
8. **Establish risk-based screening decision thresholds** using Receiver Operating Characteristic (ROC) curves and Neyman-Pearson criteria to balance probability of threat detection ($P_D$) against false alarm rates ($P_{FA}$) at high-throughput port container gates.

---

## 3. The core mental model

Hazardous materials logistics and facility security operate at the intersection of **Physical Threat Kinetics**, **Modal Separation Constraints**, and **Layered Inspection Cordon Boundaries**.

```text
=============================================================================================================
                      MULTI-MODAL DANGEROUS GOODS & FACILITY SECURITY ARCHITECTURE
=============================================================================================================

[ PHYSICAL REACTION & THREAT VECTORS ]
  ├── Class 1: Blast Overpressure Wave       --> Delta_P_s = f(Z) where Z = R / W_TNT^(1/3)
  ├── Class 2/3: BLEVE & Radiant Heat Flux   --> q'' = (tau * eta * m_dot * Delta_H_c) / (4 * pi * R^2)
  ├── Class 5/4.1: Autocatalytic Runaway     --> Arrhenius kinetics dT/dt > Q_dissipation
  ├── Class 6/2.3: Toxic Vapor Plume Cloud   --> C(x,y,z) Gaussian dispersion plume
  ├── Class 7: Ionizing Gamma / Neutron Flux --> Phi(x) = Phi_0 * exp(-mu * x)
  └── Class 9: Li-Ion Thermal Runaway / CO2  --> Venting gas volume & compartment sublimation kinetics
                                  │
                                  ▼
[ MODAL CONTAINMENT & SEPARATION CONTROLS ]
  ├── MARITIME (IMDG Code Ch 7.2) ──► 3D Cellular Grid (Bay-Row-Tier)
  │                                    ├── Level 1: "Away from" (1 cell space / bulkhead)
  │                                    ├── Level 2: "Separated from" (2 cell spaces / steel bulkhead)
  │                                    ├── Level 3: "Complete compartment" (Deck/Hold isolation)
  │                                    └── Level 4: "Longitudinal compartment" (Intervening hold)
  ├── AIR FREIGHT (IATA DGR / NOTOC) ─► Compartment Mass Balance & Ventilation
  │                                    ├── Quantity Cutoffs: PAX vs Cargo Aircraft Only (CAO)
  │                                    ├── Dry Ice Mass Limit: M_max = (C_allow * Q * rho_CO2) / r_sub
  │                                    └── PIC Notification: NOTOC verification and ICAO Red Book drill codes
  └── RAIL FREIGHT (49 CFR Part 174) ─► Train Consist Position Optimization
                                       ├── Buffer Cars: B >= 5 non-hazmat cars from lead locomotive/DPU
                                       ├── Mechanical Isolation: Separation from reefer ignition sources
                                       └── Hump Yard Protection: Prohibition of uncoupling in motion
                                  │
                                  ▼
[ LAYERED FACILITY SECURITY & RADIOGRAPHIC SCREENING ]
  ├── ISPS Code / 33 CFR Part 105 ───► MARSEC Escalation Protocols (Gate Queueing Dynamics)
  │                                    ├── MARSEC 1: Baseline random screening (5-10% rate)
  │                                    ├── MARSEC 2: Heightened security (25-50% rate, gate delay surge)
  │                                    └── MARSEC 3: Total cordon lockdown (100% inspection / ops halt)
  └── NON-INTRUSIVE INSPECTION (NII) ─► Dual-Energy Radiographic Material Discrimination
                                       ├── Attenuation: I(E) = I_0(E) * exp(-mu(E) * x)
                                       ├── Dual-Energy Ratio: R = ln(I_H / I_0H) / ln(I_L / I_0L) = f(Z_eff)
                                       └── Automated Threat Recognition (ATR): High-Z SNM & Low-Z energetic ID
=============================================================================================================
```

```text
=============================================================================================================
             3D PHYSICAL SEPARATION & RADIOGRAPHIC INSPECTION GEOMETRY
=============================================================================================================

1. MARITIME CELLULAR GRID (Bay-Row-Tier)
      [Tier 86] [ ] [ ] [A] [ ] [ ] [B] [ ]  <-- On-Deck Hatch Cover
      [Tier 84] [ ] [ ] [ ] [ ] [ ] [ ] [ ]      (Delta_Row >= 2 for Segregation 2)
   ═══════════════════════════════════════════  <-- Weather Deck Plating
      [Tier 06] [ ] [ ] [ ] [ ] [ ] [ ] [ ]  <-- Under-Deck Cargo Hold
      [Tier 04] [ ] [C] [ ] [ ] [ ] [ ] [ ]      (Separated by Transverse Watertight Bulkhead)
                ├──Hold 1──┤  │  ├──Hold 2──┤
                              └── Solid Steel Bulkhead

2. RAIL CONSIST BUFFER CAR MECHANICAL CRUSH ZONE
   [LOCO 1]-[LOCO 2]-[LOCO 3]-[BUF 1]-[BUF 2]-[BUF 3]-[BUF 4]-[BUF 5]-[HAZ TANK]-[HAZ TANK]
   │<────── Head-End Power ──────>│<────── Sacrificial Crush Buffer ─────>│<── Hazardous Consist ──>│
                                         (49 CFR Part 174: B >= 5 cars)

3. DUAL-ENERGY X-RAY TRANSMISSION & ATOMIC NUMBER DISCRIMINATION
   [LINAC Source] ──► Interleaved Pulses (6 MeV / 9 MeV)
                            │
                            ▼
                    [Target Container] ──► Areal Density (rho * x)
                            │
                            ▼
                    [Detector Array]  ──► R = ln(I_9) / ln(I_6) = f(Z_eff)
                                           ├── Low Z (<= 10): Organics, C-4, Drugs (Orange)
                                           ├── Mid Z (11-26): Structural Metals, Steel (Blue)
                                           └── High Z (> 70): Lead Shielding, Uranium (Green/Black)
=============================================================================================================
```

### The Causal Mechanism
1. **The Source Term:** Every dangerous good contains a latent energy or toxicity source term. If breached, it radiates damage via blast overpressure ($\text{kPa}$), radiant thermal flux ($\text{kW/m}^2$), toxic concentration ($\text{ppm}$), or sub-deck gas accumulation ($\% \text{vol}$).
2. **The Modal Attenuation Envelope:** Because active containment can fail during collisions, derailments, or mechanical turbulence, safety depends on spatial attenuation. Separation distances ($R$) attenuate radiant heat flux by $1/R^2$ and blast pressure by $1/R^3$. In rail consists, buffer cars act as sacrificial mechanical crush zones that absorb kinetic energy and keep hazardous tank cars out of the derailment crumple zone. In air cargo, high fresh air exchange rates ($Q$) dilute sublimated toxic or asphyxiant gases before they migrate to passenger or flight crew compartments.
3. **The Information & Verification Barrier:** Physical separation fails if cargo is misdeclared, improperly packed, or maliciously introduced. Security architectures operate as informational and physical filters. Non-Intrusive Inspection (NII) interrogates container interiors at speed of light using dual-energy photon beams, reconstructing the atomic number of the contents without opening the container doors.

---

## 4. Definitions and vocabulary

| Term | Technical Definition | Common Operational Misuse / Misconception | Governing Standard / Units |
|---|---|---|---|
| **UN Number** | A four-digit serial number assigned by the UN Committee of Experts to identify hazardous substances. | Believing a UN number uniquely specifies a single formulation; many numbers are generic (e.g., UN 1993: Flammable Liquid, N.O.S.). | UN Model Regulations; dimensionless string. |
| **Proper Shipping Name (PSN)** | The standard legal chemical or descriptive name that describes the hazard in transport documentation. | Using commercial brand names or trade jargon instead of the mandatory PSN on shipping declarations. | IMDG Code Ch 3.1; IATA DGR Sec 4. |
| **Hazard Class & Division** | Numerical category (Classes 1–9) defining the primary physical or chemical hazard mechanism. | Assuming higher class numbers represent greater severity (e.g., assuming Class 9 is more dangerous than Class 1). | UN Orange Book / IMDG / 49 CFR; Class 1.1–9. |
| **Packing Group (PG)** | Degree of danger assigned to a substance: PG I (high danger), PG II (medium danger), PG III (low danger). | Confusing the structural strength rating of an outer box (UN packaging mark) with the chemical PG of the liquid inside. | UN Model Regulations; Roman numerals I, II, III. |
| **Compatibility Group** | Letter (A through S, excluding I, M, O, P, Q, R) assigned to Class 1 explosives defining safe co-stowage. | Assuming explosives with the same division (e.g., 1.1D and 1.1A) can be stowed together inside the same container. | IMDG Code Ch 2.1; letters A–S. |
| **IMDG Segregation 1 ("Away from")** | Separation allowing same hold/compartment provided packages or containers are spaced horizontally by $\ge 1$ cell. | Assuming containers stowed vertically above one another in the same cell stack satisfy "away from" requirements. | IMDG Code Ch 7.2; table code '1'. |
| **IMDG Segregation 2 ("Separated from")** | Requirement for vertical/longitudinal isolation by a complete steel bulkhead or $\ge 2$ container spaces. | Assuming a non-structural wire mesh cargo divider fulfills the definition of a complete steel bulkhead. | IMDG Code Ch 7.2; table code '2'. |
| **IMDG Segregation 3 ("Complete compartment")** | Total hold isolation; cargo must be in separate holds separated by a complete deck or transverse bulkhead. | Permitting stowage in adjacent hold bays when the intervening bulkhead contains non-gastight inspection hatches. | IMDG Code Ch 7.2; table code '3'. |
| **IMDG Segregation 4 ("Longitudinally separated")** | Isolation requiring an intervening full cargo hold or minimum longitudinal distance of 24 meters on-deck. | Assuming two container lengths (~12 m) fulfill the longitudinal separation distance for Class 1 versus Class 3. | IMDG Code Ch 7.2; table code '4'. |
| **NOTOC** | Notification to Captain: legal manifest provided to pilot-in-command detailing all DG positions and net masses. | Treating the NOTOC as a post-departure filing rather than a pre-flight flight-deck acceptance sign-off document. | ICAO Annex 18 / IATA DGR Sec 9.6; document. |
| **Cargo Aircraft Only (CAO)** | Mandatory package handling label for consignments forbidden on passenger-carrying aircraft. | Loading a CAO package onto a passenger wide-body belly hold assuming lower-deck cargo compartments are isolated. | IATA DGR Sec 7; categorical label. |
| **Dry Ice Sublimation Rate ($\dot{r}_{\text{sub}}$)** | Fraction of solid carbon dioxide transitioning to gas per unit time under standard packaging insulation. | Assuming dry ice mass remains constant in flight, neglecting cabin asphyxiation and cargo hold overpressurization. | $\%$ of initial mass per hour ($\text{hr}^{-1}$). |
| **Buffer Car** | A non-placarded rail car loaded with benign inert freight used to provide physical crush separation. | Using an empty uncleaned hazmat tank car with residue placards as a buffer car. | US 49 CFR § 174.85; integer car count. |
| **Key Train** | An FRA/AAR designated freight train carrying $\ge 20$ carloads of hazardous materials or $\ge 1$ car of toxic inhalation hazard. | Treating Key Trains under standard block-train line speeds, ignoring mandated $50\,\text{mph}$ corridor speed caps. | AAR Circular OT-55; operational class. |
| **Toxic Inhalation Hazard (TIH)** | Gases or volatile liquids that present extreme inhalation toxicity to humans (e.g., chlorine, phosgene). | Equating TIH solely with compressed gases; high-volatility liquids (e.g., titanium tetrachloride) are also TIH. | US 49 CFR § 171.8; Hazard Zone A–D. |
| **SADT** | Self-Accelerating Decomposition Temperature: lowest ambient temperature at which exothermic runaway occurs. | Stowing organic peroxides in uninsulated general containers exposed to direct tropical solar radiation. | UN Manual of Tests and Criteria; ${}^\circ\text{C}$. |
| **State of Charge (SoC)** | Residual electrical charge percentage of a lithium-ion cell relative to its rated electrical capacity. | Transporting bulk commercial lithium-ion cells at $100\%$ charge by air, violating the statutory $30\%$ limit. | IATA DGR Packing Instruction 965; $\%$. |
| **ISPS Code** | International Ship and Port Facility Security Code: SOLAS amendment mandating maritime security standards. | Treating ISPS as an administrative checklist rather than an operational security readiness framework. | IMO SOLAS Ch XI-2; statutory framework. |
| **MARSEC Level** | Three-tier maritime security alert condition (Level 1 Normal, Level 2 Heightened, Level 3 Imminent). | Believing a terminal can unilaterally downgrade its MARSEC level below the national Coast Guard directive. | US 33 CFR Part 105; integer level (1, 2, 3). |
| **PFSO** | Port Facility Security Officer: individual legally designated to maintain and execute the Security Plan. | Delegating statutory PFSO compliance sign-off authority to an uncertified third-party gate security contractor. | ISPS Code Part A/17; legal appointment. |
| **Declaration of Security (DoS)** | Formal agreement between ship and terminal allocating security responsibilities during interface. | Assuming a DoS is mandatory for every routine commercial port call under standard MARSEC Level 1 conditions. | ISPS Code Part A/5; signed legal instrument. |
| **Non-Intrusive Inspection (NII)** | Radiographic scanning of cargo transport units using high-energy X-ray or gamma-ray imaging systems. | Confusing standard single-energy density imaging with dual-energy atomic number ($Z_{\text{eff}}$) material discrimination. | ANSI N42.44; imaging modality. |
| **Effective Atomic Number ($Z_{\text{eff}}$)** | The equivalent atomic number of a compound or mixture governing its photon attenuation characteristics. | Assuming $Z_{\text{eff}}$ is simply the arithmetic average of atomic masses, ignoring energy-dependent cross-sections. | Dimensionless material index ($Z$). |
| **Mass Attenuation ($\mu/\rho$)** | The photon beam attenuation coefficient per unit mass density of the absorbing medium. | Assuming X-ray attenuation is governed solely by linear thickness without regard to mass density. | $\text{cm}^2/\text{g}$ or $\text{m}^2/\text{kg}$. |
| **Dual-Energy Ratio ($R$)** | The ratio of high-energy to low-energy X-ray linear attenuation through a target material volume. | Believing $R$ is constant across all linear accelerator beam energies; it is dependent on the spectral filtration pair. | Dimensionless transmission ratio. |
| **Automated Threat Recognition (ATR)** | Computer vision algorithm parsing radiographic images to detect anomalies, shielding, and explosives. | Assuming ATR replaces human adjudicators; ATR functions as an attention-cueing screener under Neyman-Pearson limits. | Algorithmic classifier; $P_D, P_{FA}$. |

---

## 5. The mechanics, step by step

Engineering dangerous goods transport and facility security requires a deterministic execution pipeline across classification, vessel stowage mathematics, air cabin mass balances, rail consist optimization, port gate queueing, and radiographic physics.

```text
=============================================================================================================
                           MULTI-MODAL DANGEROUS GOODS & SECURITY PIPELINE
=============================================================================================================
  [5.1 Classification & Compatibility] ──► UN Class, Packing Group, Compatibility Groups, SADT
                 │
                 ▼
  [5.2 Maritime Cellular Segregation]   ──► Bay-Row-Tier Math, Bulkhead Boundaries, Blast/Heat Flux
                 │
                 ▼
  [5.3 Air Cargo & Hold Ventilation]   ──► PAX vs CAO, NOTOC Drill Codes, Dry Ice Sublimation ODE
                 │
                 ▼
  [5.4 Freight Rail Consist Layout]    ──► 49 CFR Part 174 Buffer Car MILP, Hump Yard Restrictions
                 │
                 ▼
  [5.5 Port Security MARSEC Dynamics]  ──► Gate Inspection M/G/s Queues, Capacity Degradation
                 │
                 ▼
  [5.6 Non-Intrusive Radiography NII]  ──► Dual-Energy Transmission, Z_eff Reconstruction, ATR ROC
=============================================================================================================
```

### 5.1 Multi-Modal Classification and Chemical Compatibility
Every cargo shipment entering the international multi-modal chain must be classified across the UN Hazard Hierarchy:

- **Class 1: Explosives:** Divisions 1.1 (mass explosion hazard), 1.2 (projection hazard), 1.3 (fire/minor blast), 1.4 (minor hazard), 1.5 (very insensitive mass explosion substances), and 1.6 (extremely insensitive articles). Assigned compatibility groups A through S (excluding I, M, O, P, Q, R).
- **Class 2: Gases:** Division 2.1 (flammable gases, e.g., propane), Division 2.2 (non-flammable, non-toxic compressed gases, e.g., argon, nitrogen), and Division 2.3 (toxic inhalation hazard gases, e.g., chlorine, phosgene).
- **Class 3: Flammable Liquids:** Categorized by flash point (FP) and initial boiling point (IBP):
  - *Packing Group I:* $\text{FP} < 23^\circ\text{C}$ and $\text{IBP} \le 35^\circ\text{C}$ (extreme danger).
  - *Packing Group II:* $\text{FP} < 23^\circ\text{C}$ and $\text{IBP} > 35^\circ\text{C}$ (medium danger).
  - *Packing Group III:* $23^\circ\text{C} \le \text{FP} \le 60^\circ\text{C}$ and $\text{IBP} > 35^\circ\text{C}$ (minor danger).
- **Class 4: Reactive Solids:** Division 4.1 (flammable solids, self-reactive substances, and desensitized explosives), Division 4.2 (substances liable to spontaneous combustion, pyrophoric liquids/solids), Division 4.3 (substances which in contact with water emit flammable gases).
- **Class 5: Oxidizers and Organic Peroxides:** Division 5.1 (oxidizing substances that yield oxygen to stimulate combustion), Division 5.2 (organic peroxides containing the bivalent $-O-O-$ structure, subject to thermal runaway).
- **Class 6: Toxic and Infectious Substances:** Division 6.1 (toxic substances presenting oral, dermal, or inhalation toxicity measured by $LD_{50}$ / $LC_{50}$ thresholds), Division 6.2 (infectious biological substances Category A and B).
- **Class 7: Radioactive Material:** Radioactive packages categorized into White-I, Yellow-II, and Yellow-III based on surface radiation dose rate ($\text{mSv/h}$) and Transport Index (TI).
- **Class 8: Corrosive Substances:** Liquids or solids causing full-thickness destruction of intact skin tissue within prescribed exposure periods (PG I: $\le 3\,\text{min}$; PG II: $\le 14\,\text{days}$; PG III: steel/aluminum corrosion $> 6.25\,\text{mm/year}$).
- **Class 9: Miscellaneous Dangerous Substances:** Environmentally hazardous substances, genetically modified organisms, elevated temperature substances, and lithium-ion batteries (UN 3480, UN 3481).

#### Explosive Compatibility Groups Matrix
For Class 1 explosives, compatibility groups govern co-loading within the same Cargo Transport Unit (CTU). Compatibility Group A (primary explosive substance) cannot share a container with any other group. Group B (articles containing primary explosives, e.g., detonators) cannot be co-loaded with Group D (secondary detonating explosives, e.g., TNT, RDX), because accidental shock initiation of Group B would propagate an uncontained mass detonation into Group D. Group S articles (packaged such that hazardous effects are confined within the package) may be co-loaded with groups C, D, and E.

#### Thermal Decomposition and SADT Kinetics
For Class 5.2 organic peroxides and Class 4.1 self-reactive substances, transport safety is governed by the Self-Accelerating Decomposition Temperature (SADT). Decomposition follows Arrhenius reaction kinetics:
$$\frac{dT}{dt} = \frac{Q_{\text{gen}} - Q_{\text{diss}}}{m \cdot C_p} = \frac{\Delta H_r \cdot A \cdot e^{-\frac{E_a}{R T}} \cdot C^n - U \cdot A_{\text{surf}} (T - T_{\text{amb}})}{m \cdot C_p}$$
When ambient temperature rises such that heat generation exceeds conductive and convective dissipation ($Q_{\text{gen}} > Q_{\text{diss}}$), internal temperature accelerates exponentially, triggering deflagration or thermal explosion. Statutory thresholds mandate:
$$T_{\text{control}} \le T_{\text{SADT}} - 10^\circ\text{C} \quad \text{for } T_{\text{SADT}} \le 35^\circ\text{C}$$
$$T_{\text{emergency}} \le T_{\text{SADT}} - 5^\circ\text{C} \quad \text{for } T_{\text{SADT}} \le 35^\circ\text{C}$$

#### Lithium-Ion Battery Electro-Thermal Runaway Mechanics
Lithium-ion cells (UN 3480 standalone, UN 3481 packed with/in equipment) represent a severe hazard due to coupled electrochemical and thermal runaway mechanics. The failure sequence progresses through distinct physical stages:

1. **Initiation Phase:** Mechanical crushing, manufacturing defects (e.g., microscopic metal burrs), or external heating trigger localized internal electrical short circuits between cathode and anode layers.
2. **Solid Electrolyte Interphase (SEI) Breakdown:** At temperatures between $80^\circ\text{C}$ and $120^\circ\text{C}$, the protective SEI layer on the graphite anode decomposes exothermically:
   $$\text{SEI} \longrightarrow \text{Stable Inorganics} + \text{Flammable Hydrocarbon Gas} + \Delta H_1$$
3. **Cathode Decomposition and Oxygen Release:** Above $180^\circ\text{C}$ to $220^\circ\text{C}$, metal oxide cathodes (e.g., Nickel-Manganese-Cobalt NMC, Lithium Cobalt Oxide LCO) decompose, releasing free elemental oxygen:
   $$\text{Li}_{1-x}\text{CoO}_2 \longrightarrow \frac{1-x}{3}\text{Co}_3\text{O}_4 + \frac{x}{2}\text{Li}_2\text{O} + \frac{1-x}{6}\text{O}_2 \uparrow + \Delta H_2$$
4. **Combustion and Pressure Venting:** Released oxygen combusts organic carbonate electrolyte solvents (ethylene carbonate, dimethyl carbonate) inside the sealed cell casing. Internal pressure spikes until the safety burst disc ruptures, venting toxic, flammable gases ($CO, CO_2, H_2, CH_4, C_2H_4$) at rates between $0.8$ and $2.5\,\text{liters/Ah}$ with flame temperatures exceeding $900^\circ\text{C}$.

Under ICAO Annex 18 and IATA Packing Instruction 965, commercial lithium-ion cells shipped as cargo on freighters must not exceed a **30% State of Charge (SoC)**. At $\le 30\%$ SoC, the cell's internal electrochemical enthalpy is insufficient to drive cathode oxygen release during an internal short, preventing self-sustaining thermal runaway cascade across neighboring cells.

---

### 5.2 Maritime Container Vessel 3D Cellular Segregation Mechanics (IMDG Code Chapter 7.2)
On container vessels, locations are indexed via the standard six-digit Bay-Row-Tier coordinate system:
- **Bay ($B$):** Longitudinal position along the vessel length. Odd numbers ($01, 03, 05 \dots$) denote twenty-foot equivalent units (TEU). Even numbers ($02, 04, 06 \dots$) denote forty-foot equivalent units (FEU).
- **Row ($R$):** Transverse position across the vessel beam. Centerline is Row $00$. Starboard rows are odd ($01, 03, 05 \dots$); port rows are even ($02, 04, 06 \dots$).
- **Tier ($T$):** Vertical position. Under-deck tiers start at tank top: $02, 04, 06, 08 \dots$ On-deck tiers start at hatch cover: $82, 84, 86, 88 \dots$

```text
=============================================================================================================
             IMDG CELLULAR SEGREGATION RULES FOR FULLY CELLULAR CONTAINER SHIPS
=============================================================================================================
Segregation Level | On-Deck Longitudinal | On-Deck Transverse | Under-Deck Longitudinal | Under-Deck Transverse
------------------|----------------------|--------------------|-------------------------|----------------------
1: "Away from"    | >= 1 container pitch | >= 1 container row | >= 1 container pitch OR | >= 1 container row OR
                  | (One Bay gap)        | (One Row gap)      | separated by bulkhead   | separated by bulkhead
------------------|----------------------|--------------------|-------------------------|----------------------
2: "Separated     | >= 2 container pitch | >= 2 container row | Separated by a fire- and liquid-tight complete
   from"          | (Two Bay gaps)       | (Two Row gaps)     | transverse steel bulkhead (Different Holds)
------------------|----------------------|--------------------|-------------------------|----------------------
3: "Complete      | >= 2 container pitch | Prohibited in same | Separated vertically by a complete steel deck AND
   compartment"   | across hatch split   | hatch boundary     | horizontally by a complete steel bulkhead
------------------|----------------------|--------------------|-------------------------|----------------------
4: "Longitudinal  | >= 4 container pitch | Prohibited on same | Separated longitudinally by an entire intervening
   intervening"   | (min 24 meters)      | hatch cover        | complete cargo hold (Intervening Hold gap)
=============================================================================================================
```

#### Physical Blast and Thermal Boundary Formulations
Segregation distances prevent domino detonations or thermal ignition across container walls. Blast overpressure is governed by the Hopkinson-Cranz scaled distance $Z$:
$$Z = \frac{R}{W_{\text{TNT}}^{1/3}}$$
where $R$ is distance in meters and $W_{\text{TNT}}$ is effective explosive mass in kilograms. The incident blast peak overpressure $\Delta P_s$ in kilopascals is modeled by Sadovsky's formulation:
$$\Delta P_s = \frac{102}{Z} + \frac{403}{Z^2} + \frac{855}{Z^3} \quad [\text{kPa}]$$
Standard maritime container corrugated steel walls buckle at $\Delta P_s \ge 35\,\text{kPa}$ and suffer complete structural rupture at $\Delta P_s \ge 100\,\text{kPa}$.

For flammable pool fires on hatch covers, radiant thermal flux $q''$ arriving at an adjacent container wall is:
$$q'' = \frac{\tau \cdot \eta \cdot \dot{m} \cdot \Delta H_c}{4 \pi R^2} \quad \left[\frac{\text{kW}}{\text{m}^2}\right]$$
where $\tau$ is atmospheric transmissivity ($\approx 0.85$), $\eta$ is radiative fraction ($\approx 0.30$), $\dot{m}$ is burning rate ($\text{kg/s}$), $\Delta H_c$ is heat of combustion ($\text{kJ/kg}$), and $R$ is geometric separation distance ($\text{m}$). Safe operational separation requires $q'' \le 12.5\,\text{kW/m}^2$ to prevent autoignition of adjacent wooden dunnage and boiling vapor buildup in neighboring containers.

---

### 5.3 Air Cargo Hazmat Engineering and Compartment Ventilation Kinetics (IATA DGR)
Air transport introduces extreme environmental stresses: ambient pressure drops from $101.3\,\text{kPa}$ at sea level to $22.5\,\text{kPa}$ at FL350, generating severe outward pressure differentials ($\Delta P \approx 78\,\text{kPa}$) on sealed chemical drums, while cruising temperatures drop to $-55^\circ\text{C}$.

Under ICAO Annex 18 and IATA DGR, substances are restricted by aircraft type:
1. **Passenger Aircraft (PAX):** Restricted to small net package quantities (e.g., maximum $1.0\,\text{L}$ to $5.0\,\text{L}$ for Class 3 PG II). Bulk quantities are strictly prohibited.
2. **Cargo Aircraft Only (CAO):** Higher package mass allowances permitted, but strictly forbidden on passenger hulls. Packages must bear the bright orange CAO label and be stowed accessible to the flight crew or in certified Class C/Class E cargo compartments.
3. **Notification to Captain (NOTOC):** Mandatory manifest delivered to the pilot-in-command prior to departure detailing UN number, PSN, class, subsidiary risk, packing group, exact ULD position, net quantity, and ICAO Emergency Response Drill Code (e.g., Drill Code `3L`: Flammable liquid, spill/leak risk, extinguish with dry chemical/Halon, ventilate cockpit).

#### Dry Ice Sublimation Mass Balance
Dry ice (UN 1845, solid carbon dioxide) sublimes directly from solid to gas at temperatures above $-78.5^\circ\text{C}$. In an aircraft cargo hold of net volume $V$ ($\text{m}^3$) with fresh air ventilation rate $Q$ ($\text{m}^3/\text{hr}$), the transient mass balance differential equation for $CO_2$ concentration $C(t)$ (volume fraction) is:
$$\frac{dC(t)}{dt} = \frac{\dot{m}_{CO_2}}{\rho_{CO_2} V} - \left(\frac{Q}{V}\right) C(t)$$
where $\dot{m}_{CO_2} = M_{\text{ice}} \cdot \dot{r}_{\text{sub}}$ is gaseous $CO_2$ generation rate ($\text{kg/hr}$), $\dot{r}_{\text{sub}}$ is sublimation rate ($\text{hr}^{-1}$), and $\rho_{CO_2}$ is gas density at cabin pressure ($\approx 1.50\,\text{kg/m}^3$ at $82\,\text{kPa}$ and $20^\circ\text{C}$).

Integrating with initial background atmospheric concentration $C(0) = C_{\text{amb}} \approx 0.04\%$ ($400\,\text{ppm}$):
$$C(t) = C_{\text{amb}} e^{-\frac{Q}{V}t} + \frac{\dot{m}_{CO_2}}{\rho_{CO_2} Q} \left(1 - e^{-\frac{Q}{V}t}\right)$$

As $t \to \infty$, concentration reaches steady state:
$$C_{\text{ss}} = \frac{\dot{m}_{CO_2}}{\rho_{CO_2} Q} = \frac{M_{\text{ice}} \cdot \dot{r}_{\text{sub}}}{\rho_{CO_2} Q}$$

The Federal Aviation Administration (FAA) and aircraft manufacturer weight and balance manuals mandate that steady-state cargo hold $CO_2$ concentration cannot exceed $C_{\max} = 0.5\%$ ($5,000\,\text{ppm}$) during normal operations, and must remain below $3.0\%$ ($30,000\,\text{ppm}$) during single-engine ventilation failure. The maximum allowable dry ice mass $M_{\max}$ is bounded by:
$$M_{\max} = \frac{C_{\max} \cdot \rho_{CO_2} \cdot Q}{\dot{r}_{\text{sub}}}$$

If $Q = 900\,\text{m}^3/\text{hr}$, $\dot{r}_{\text{sub}} = 0.85\%/\text{hr} = 0.0085\,\text{hr}^{-1}$, and $\rho_{CO_2} = 1.48\,\text{kg/m}^3$:
$$M_{\max} = \frac{0.005 \times 1.48 \times 900}{0.0085} = 783.5\,\text{kg}$$

---

### 5.4 Rail Consist Marshalling & Buffer Car Optimization (US 49 CFR Part 174)
Federal regulations enforce deterministic spacing between placarded hazardous tank cars and train personnel/ignition sources to mitigate kinetic impact hazards during derailments and coupling shocks.

Let a freight train consist be modeled as a sequence of cars $i \in \{1, 2, \dots, N\}$, where $i = 1$ is the lead locomotive cab. Let binary parameters define the status of each railcar:
- $L_i = 1$ if car $i$ is an occupied locomotive cab; $0$ otherwise.
- $H_i = 1$ if car $i$ is a placarded loaded hazardous material tank car (e.g., Class 2.1, 2.3, 3, 1.1).
- $R_i = 1$ if car $i$ is an internal combustion ignition source (e.g., mechanical reefer unit).
- $S_i = 1$ if car $i$ carries a shiftable load (e.g., open flat car of lumber/steel rails).
- $C_i = 1$ if car $i$ is an occupied caboose or manned distributed power unit (DPU).

Under 49 CFR § 174.85, consist placement constraints are formulated as:

1. **Lead Locomotive Buffer Distance:** A placarded hazardous tank car must be separated from an occupied locomotive by at least $B = 5$ non-hazardous buffer cars:
   $$|i - j| \ge B + 1 \quad \forall i, j \text{ such that } H_i = 1, L_j = 1 \quad (\text{for } N \ge 30)$$
   If train length is insufficient ($N < 30$), the car must be placed as near to the middle of the train as possible, with never fewer than 1 buffer car ($|i - j| \ge 2$).
2. **Occupied Rear / Caboose Buffer Distance:**
   $$|i - k| \ge B + 1 \quad \forall i, k \text{ such that } H_i = 1, C_k = 1$$
3. **Mechanical Ignition Isolation:** A placarded tank car cannot be placed adjacent to a mechanical refrigeration unit or internal combustion engine:
   $$|i - r| \ge 2 \quad \forall i, r \text{ such that } H_i = 1, R_r = 1$$
4. **Shiftable Load Protection:** A placarded tank car cannot be placed adjacent to flat cars carrying lading prone to shifting under longitudinal buff-and-draft forces:
   $$|i - s| \ge 2 \quad \forall i, s \text{ such that } H_i = 1, S_s = 1$$

#### Mixed-Integer Linear Programming (MILP) Consist Assembly Formulation
In an automated rail classification yard, the yardmaster solves a combinatorial optimization problem to assemble an outbound train consist with minimal shunting operations. Let binary decision variable $x_{i,c} = 1$ if car $c \in \mathcal{C}$ is placed at train slot $i \in \{1, \dots, N\}$. Let $c_{\text{shunt}}(c, i)$ denote the switching cost of moving car $c$ from its classification track to consist position $i$:
$$\min \sum_{c \in \mathcal{C}} \sum_{i=1}^N c_{\text{shunt}}(c, i) \cdot x_{i,c}$$
subject to:
$$\sum_{c \in \mathcal{C}} x_{i,c} = 1 \quad \forall i \in \{1, \dots, N\} \quad (\text{Exact position occupancy})$$
$$\sum_{i=1}^N x_{i,c} = 1 \quad \forall c \in \mathcal{C} \quad (\text{Every car assigned once})$$
$$\sum_{k = \max(1, i-5)}^{\min(N, i+5)} \sum_{c \in \mathcal{H}} x_{k,c} \le M \left(1 - \sum_{c \in \mathcal{L}} x_{i,c}\right) \quad \forall i \in \{1, \dots, N\} \quad (\text{Lead Locomotive 5-car buffer})$$
$$\sum_{k = \max(1, i-1)}^{\min(N, i+1)} \sum_{c \in \mathcal{R}} x_{k,c} \le M \left(1 - \sum_{c \in \mathcal{H}} x_{i,c}\right) \quad \forall i \in \{1, \dots, N\} \quad (\text{Reefer 1-car isolation})$$
where $\mathcal{H}$ is the set of placarded hazmat cars, $\mathcal{L}$ is the set of occupied locomotives, $\mathcal{R}$ is the set of mechanical reefers, and $M$ is a sufficiently large integer constant.

In hump classification yards, cars placarded Class 1.1, 1.2, 2.3 (Zone A), or 4.2 are legally prohibited from being cut off while in motion ("humped") and cannot be coupled at speeds exceeding $4.0\,\text{mph}$ ($6.4\,\text{km/h}$) to prevent draft gear shear and tank head punctures.

---

### 5.5 Port Facility Security & MARSEC Escalation Dynamics (ISPS / 33 CFR Part 105)
The International Ship and Port Facility Security (ISPS) Code and US Coast Guard regulations (33 CFR Part 105) establish three Maritime Security (MARSEC) levels:

```text
=============================================================================================================
                  ISPS CODE & US 33 CFR PART 105 MARSEC ESCALATION PROTOCOLS
=============================================================================================================
Security Level | Threat State         | Operational Protocols & Terminal Gate Impacts
---------------|----------------------|----------------------------------------------------------------------
MARSEC Level 1 | Baseline / Normal    | Continuous perimeter access control; TWIC biometric badge checks;
               |                      | random physical screening of 5% to 10% of drayage trucks;
               |                      | standard automated terminal gate optical character recognition (OCR).
---------------|----------------------|----------------------------------------------------------------------
MARSEC Level 2 | Heightened / Alert   | Escalated screening of 25% to 50% of arriving trucks; 100% seal
               | (Specific Risk)      | integrity inspections; intensified waterside security boat patrols;
               |                      | closure of non-essential pedestrian gates; restricted berth access.
---------------|----------------------|----------------------------------------------------------------------
MARSEC Level 3 | Imminent / Attack    | Total facility lockdown; 100% physical and radiographic inspection of
               | Probable             | all inbound units OR complete cessation of cargo operations;
               |                      | armed waterside exclusion zones; naval/Coast Guard tactical control.
=============================================================================================================
```

#### Terminal Gate Queueing Under MARSEC Escalation
Let drayage trucks arrive at the port container gate following a Poisson process with arrival rate $\lambda$ ($\text{trucks/hour}$). Let the gate facility operate $s$ processing lanes.

Under baseline MARSEC Level 1, a fraction $(1 - p_1)$ of trucks undergo automated processing with mean service time $1/\mu_0 \approx 1.5\,\text{minutes}$ ($40\,\text{trucks/hr}$), while a random fraction $p_1 = 0.05$ undergo secondary physical inspection with mean service time $1/\mu_{\text{insp}} \approx 15.0\,\text{minutes}$ ($4\,\text{trucks/hr}$). The effective mean service rate $\mu_{\text{eff}}$ is:
$$E[S_1] = (1 - p_1)\frac{1}{\mu_0} + p_1 \frac{1}{\mu_{\text{insp}}} = 0.95(1.5) + 0.05(15.0) = 1.425 + 0.75 = 2.175\,\text{min} \implies \mu_{\text{eff},1} = 27.59\,\text{trucks/hr/lane}$$

When the Captain of the Port (COTP) escalates the facility to MARSEC Level 2, the inspection fraction increases to $p_2 = 0.35$:
$$E[S_2] = (1 - p_2)\frac{1}{\mu_0} + p_2 \frac{1}{\mu_{\text{insp}}} = 0.65(1.5) + 0.35(15.0) = 0.975 + 5.25 = 6.225\,\text{min} \implies \mu_{\text{eff},2} = 9.64\,\text{trucks/hr/lane}$$

Under MARSEC Level 2, lane capacity collapses by:
$$\Delta C_{\text{gate}} = \frac{27.59 - 9.64}{27.59} = 65.1\%$$
If the port gate operates $s = 6$ lanes and arrival rate is $\lambda = 120\,\text{trucks/hr}$:
- At MARSEC 1: System utilization $\rho_1 = \frac{\lambda}{s \cdot \mu_{\text{eff},1}} = \frac{120}{6 \times 27.59} = \frac{120}{165.5} = 0.725$ (Stable; mean queue $L_q \approx 1.8$ trucks).
- At MARSEC 2: System capacity is $s \cdot \mu_{\text{eff},2} = 6 \times 9.64 = 57.8\,\text{trucks/hr}$. Since $\lambda = 120 > 57.8$, utilization $\rho_2 = 2.07 > 1.0$.
The gate queue grows deterministically at rate $\frac{dL}{dt} = \lambda - s \mu_{\text{eff},2} = 120 - 57.8 = 62.2\,\text{trucks/hour}$, generating gridlock across adjacent arterial highways within 90 minutes.

#### M/G/s Heavy-Traffic Delay Multiplier
Beyond mean service degradation, the bimodal service distribution (fast OCR pass vs. slow physical unpack) introduces severe service time variance. Under the Allen-Cunneen approximation for multi-server $M/G/s$ queueing:
$$W_q \approx \frac{P_m}{s \mu_{\text{eff}} (1 - \rho)} \cdot \left(\frac{C_a^2 + C_s^2}{2}\right)$$
where $C_s^2 = \frac{\text{Var}(S)}{(E[S])^2}$ is the squared coefficient of variation of service time. Because $\text{Var}(S) = E[S^2] - (E[S])^2$, mixing 1.5-minute passes with 15-minute inspections inflates $C_s^2$ from $0.25$ to $> 2.45$. This structural variance doubles queue waiting time $W_q$ even before utilization $\rho$ approaches $1.0$.

---

### 5.6 Non-Intrusive Inspection (NII) & Dual-Energy Radiographic Screening Algorithms
High-energy X-ray gantry systems inspect containers using linear accelerators (LINAC) pulsing interleaved high-energy ($E_H = 9.0\,\text{MeV}$) and low-energy ($E_L = 6.0\,\text{MeV}$) photon beams.

As a mono-energetic photon beam passes through material of thickness $x$ and density $\rho$, transmitted intensity $I$ is governed by the Beer-Lambert law:
$$I = I_0 \exp\left(-\left(\frac{\mu}{\rho}\right) \cdot \rho x\right)$$
where $\mu/\rho$ is the mass attenuation coefficient ($\text{cm}^2/\text{g}$). At megavolt energy levels, three physical processes dominate total mass attenuation:
$$\frac{\mu}{\rho} = \left(\frac{\mu}{\rho}\right)_{\text{Photoelectric}} + \left(\frac{\mu}{\rho}\right)_{\text{Compton}} + \left(\frac{\mu}{\rho}\right)_{\text{Pair Production}}$$

1. **Photoelectric Absorption:** Cross-section scales as $\sigma_{\text{PE}} \propto \frac{Z^4}{E^3}$ (negligible above $1\,\text{MeV}$).
2. **Compton Scattering:** Incoherent scattering from electrons. Cross-section is independent of atomic number $Z$: $\sigma_C \propto \frac{Z}{A} \approx 0.5\,\text{const}$.
3. **Pair Production:** Creation of electron-positron pairs in the nuclear Coulomb field. Occurs only above threshold $E > 1.022\,\text{MeV}$, with cross-section scaling directly with atomic number: $\sigma_{\text{PP}} \propto \frac{Z^2}{A} \propto Z$.

#### Reconstructing Effective Atomic Number ($Z_{\text{eff}}$)
By measuring transmitted intensities at high and low beam energies, the dual-energy attenuation ratio $R$ cancels out total areal mass density ($\rho x$):
$$R = \frac{-\ln(I_H / I_{0,H})}{-\ln(I_L / I_{0,L})} = \frac{(\mu/\rho)_H}{(\mu/\rho)_L} = f(Z_{\text{eff}})$$

Because pair production increases with $Z$ at $9.0\,\text{MeV}$ while Compton scattering dominates at $6.0\,\text{MeV}$, the ratio $R$ increases monotonically with effective atomic number $Z_{\text{eff}}$:
- **Low-$Z$ (Organics, Explosives, Narcotics):** $Z_{\text{eff}} \le 10 \implies R \in [0.85, 0.95]$.
- **Mid-$Z$ (Inorganics, Structural Metals, Steel):** $10 < Z_{\text{eff}} \le 26 \implies R \in [0.98, 1.15]$.
- **High-$Z$ (Heavy Metals, Shielding, Uranium, Plutonium):** $Z_{\text{eff}} > 70 \implies R \ge 1.35$.

#### Neyman-Pearson Decision Theory & Automated Threat Recognition (ATR)
Automated Threat Recognition (ATR) systems evaluate container pixels using binary hypothesis testing:
$$H_0: \text{Lading is benign cargo} \quad \text{vs.} \quad H_1: \text{Lading contains contraband or illicit threat}$$
Let $x$ be the extracted radiographic feature vector (e.g., dual-energy ratio $R$, areal density $\rho x$, spatial texture). Under the Neyman-Pearson criterion, the decision threshold $\gamma$ is chosen to maximize the Probability of Detection ($P_D$) subject to an upper bound on False Alarm Rate ($P_{FA} \le \alpha$):
$$P_D(\gamma) = \int_{\gamma}^\infty p(x \mid H_1) dx, \quad P_{FA}(\gamma) = \int_{\gamma}^\infty p(x \mid H_0) dx \le \alpha$$
The trade-off across varying thresholds $\gamma$ traces the Receiver Operating Characteristic (ROC) curve. The integral under the curve represents the Area Under the Curve (AUC):
$$\text{AUC} = \int_0^1 P_D(P_{FA}) dP_{FA}$$
Operational NII systems achieve $\text{AUC} \ge 0.96$, balancing automated interdiction against secondary unpacking bay bottlenecks.

---

## 6. Worked example

An international chemical conglomerate ships an intermodal consolidation through a major container terminal and rail corridor, accompanied by an urgent air freight consignment. We evaluate safety compliance across all three modes.

### Part A: Maritime 3D Vessel Segregation
A 14,000 TEU ultra-large container vessel is loading cargo at the berth. Three containers are scheduled for stowage:
- **Container A:** 20-ft container, UN 1203, Gasoline (Class 3, Flammable Liquid, PG II).
- **Container B:** 20-ft container, UN 1479, Oxidizing Solid, N.O.S. (Class 5.1, Oxidizer, PG II).
- **Container C:** 20-ft container, UN 0004, Ammonium Picrate (Class 1.1D, Explosive).

#### 1. Segregation Requirements Lookup
Consulting the IMDG Code General Segregation Table:
- **Class 3 vs. Class 5.1:** Segregation code is `2` ("Separated from").
- **Class 1.1D vs. Class 3:** Segregation code is `4` ("Separated longitudinally by an intervening complete compartment or hold from").
- **Class 1.1D vs. Class 5.1:** Segregation code is `4`.

#### 2. Evaluating Planned Vessel Slots
The automated terminal operating system (TOS) proposes the following stowage slots:
- **Container A:** Stowed On-Deck at Bay 11, Row 03, Tier 84.
- **Container B:** Stowed On-Deck at Bay 11, Row 07, Tier 84.
- **Container C:** Stowed Under-Deck at Bay 27, Row 02, Tier 06.

```text
=============================================================================================================
             CONTAINERSHIP BAY 11 HATCH COVER TRANSVERSE STOWAGE PROFILE
=============================================================================================================
Tier 86 | [  ]   [  ]   [  ]   [  ]   [  ]   [  ]   [  ]   [  ]   [  ]   [  ]   [  ]   [  ]
Tier 84 | [  ]   [  ]   [ B ]  [  ]   [  ]   [  ]   [ A ]  [  ]   [  ]   [  ]   [  ]   [  ]
--------|----------------------------------------------------------------------------------------------------
Row ID  |  08     06     07     05     03     01     00     02     04     06     08     10
        | <────────── Starboard Rows (Odd) ──────────> | Ctr | <──────── Port Rows (Even) ─────────>
=============================================================================================================
Note: Container A is in Row 03; Container B is in Row 07. Intervening Row 05 provides physical buffer.
```

#### 3. Verification Mathematics
- **Container A to Container B (Class 3 vs. Class 5.1):**
  - Longitudinal Bay Delta: $|\text{Bay}_B - \text{Bay}_A| = |11 - 11| = 0$ container spaces.
  - Transverse Row Delta: Both containers are on starboard (odd rows). Row sequence is $01, 03, 05, 07$.
    The transverse spacing delta is $\Delta R = \frac{07 - 03}{2} = 2$ container spaces.
  - IMDG Rule for Segregation `2` On-Deck: Cargo must be separated by either (a) $\ge 2$ container pitches longitudinally AND $\ge 1$ container row transversely, OR (b) $\ge 2$ container rows transversely.
  - Evaluation: $\Delta R = 2 \ge 2$. **Compliant.** A physical buffer of one intervening container row (Row 05) separates the containers on the hatch cover.
- **Container A to Container C (Class 3 vs. Class 1.1D):**
  - Container A is on-deck in Hold 2 (Bays 09–13).
  - Container C is under-deck in Hold 4 (Bays 25–29).
  - Intervening space: Hold 3 (Bays 17–21) completely separates the two holds.
  - Longitudinal separation distance exceeds $24.0\,\text{meters}$ and spans a complete intervening cargo hold bounded by two watertight transverse structural bulkheads.
  - Evaluation: **Compliant with Segregation Level 4.**

---

### Part B: Air Freight Dry Ice Compartment Ventilation
A biomedical forwarder tenders a shipment containing $M_{\text{ice}} = 650.0\,\text{kg}$ of dry ice (UN 1845) packed in validated vacuum-insulated shipper boxes, loaded onto a Boeing 777F lower forward cargo compartment.

#### Operational Parameters:
- Lower forward hold volume: $V = 69.8\,\text{m}^3$.
- Aircraft fresh air ventilation delivery rate: $Q = 18.0\,\text{m}^3/\text{min} = 1,080.0\,\text{m}^3/\text{hr}$.
- Cruising cabin pressure: $P_{\text{cab}} = 82.5\,\text{kPa}$; Compartment temperature: $T = 18.0^\circ\text{C} = 291.15\,\text{K}$.
- Sublimation rate of insulated boxes: $\dot{r}_{\text{sub}} = 0.75\%/\text{hr} = 0.0075\,\text{hr}^{-1}$.
- Statutory maximum continuous $CO_2$ limit: $C_{\max} = 0.50\% = 0.0050$ volume fraction.
- Background atmospheric $CO_2$: $C_{\text{amb}} = 420\,\text{ppm} = 0.00042$.

#### 1. Density of Gaseous Carbon Dioxide
Using the ideal gas law:
$$\rho_{CO_2} = \frac{P_{\text{cab}} \cdot M_{\text{molar}}}{R_{\text{univ}} \cdot T} = \frac{82,500\,\text{Pa} \times 0.04401\,\text{kg/mol}}{8.3145\,\text{J/(mol}\cdot\text{K)} \times 291.15\,\text{K}} = \frac{3,630.8}{2,420.8} = 1.500\,\text{kg/m}^3$$

#### 2. Hourly Carbon Dioxide Generation Rate
$$\dot{m}_{CO_2} = M_{\text{ice}} \times \dot{r}_{\text{sub}} = 650.0\,\text{kg} \times 0.0075\,\text{hr}^{-1} = 4.875\,\text{kg/hr}$$
Volumetric gas generation rate:
$$\dot{V}_{CO_2} = \frac{\dot{m}_{CO_2}}{\rho_{CO_2}} = \frac{4.875\,\text{kg/hr}}{1.500\,\text{kg/m}^3} = 3.250\,\text{m}^3/\text{hr}$$

#### 3. Steady-State Hold Concentration
$$C_{\text{ss}} = C_{\text{amb}} + \frac{\dot{V}_{CO_2}}{Q} = 0.00042 + \frac{3.250\,\text{m}^3/\text{hr}}{1,080.0\,\text{m}^3/\text{hr}} = 0.00042 + 0.003009 = 0.003429 = 0.343\% \quad (3,429\,\text{ppm})$$
Evaluation: $C_{\text{ss}} = 3,429\,\text{ppm} \le 5,000\,\text{ppm}$. **Compliant.**

#### 4. Transient Build-Up Profile Over Flight Duration
Using the transient mass balance solution $C(t) = C_{\text{amb}} e^{-\frac{Q}{V}t} + \frac{\dot{m}_{CO_2}}{\rho_{CO_2} Q} (1 - e^{-\frac{Q}{V}t})$, with turnover rate $k = Q/V = 1,080.0 / 69.8 = 15.47\,\text{hr}^{-1}$:

```text
=============================================================================================================
              AIRCRAFT HOLD CO2 CONCENTRATION TRANSIENT DYNAMICS
=============================================================================================================
Time Elapsed (t) | Exponential Decay factor | Volumetric Fraction C(t) | Concentration (ppm) | Status
-----------------|--------------------------|--------------------------|---------------------|-------
t = 0 minutes    | e^(0) = 1.0000           | 0.000420                 |   420 ppm           | Baseline
t = 5 minutes    | e^(-1.289) = 0.2755      | 0.002600                 | 2,600 ppm           | Safe
t = 15 minutes   | e^(-3.868) = 0.0209      | 0.003366                 | 3,366 ppm           | Safe
t = 30 minutes   | e^(-7.736) = 0.0004      | 0.003428                 | 3,428 ppm           | Steady State
t = 120 minutes  | e^(-30.94) ~ 0.0000      | 0.003429                 | 3,429 ppm           | Steady State
=============================================================================================================
```

#### 5. Maximum Permissible Dry Ice Mass
$$M_{\max} = \frac{(C_{\max} - C_{\text{amb}}) \cdot \rho_{CO_2} \cdot Q}{\dot{r}_{\text{sub}}} = \frac{(0.0050 - 0.00042) \times 1.500 \times 1,080.0}{0.0075} = \frac{0.00458 \times 1,620.0}{0.0075} = \frac{7.4196}{0.0075} = 989.28\,\text{kg}$$
The $650.0\,\text{kg}$ consignment uses $650.0 / 989.28 = 65.7\%$ of maximum compartment ventilation capacity, leaving $339.28\,\text{kg}$ of dry ice buffer.

---

### Part C: Freight Rail Train Consist 49 CFR Part 174 Marshaling
A 75-car manifest train is assembling at an intermodal classification yard:
- Head end: 3 road locomotives (Positions 1, 2, 3).
- Hazmat block: 4 placarded loaded DOT-117 tank cars carrying UN 1086 (Vinyl Chloride, Class 2.1 Flammable Gas, Inhalation Hazard).
- General freight: Boxcars, covered hoppers, flat cars.
- Position 22: One mechanical refrigerator car (UN 1978 propane cooling engine running continuously).
- Rear end: Manned Distributed Power Unit (Position 75).

```text
=============================================================================================================
               75-CAR FREIGHT TRAIN CONSIST MARSHALING SCHEMATIC
=============================================================================================================
Pos 01-03: [LOCOMOTIVES 1, 2, 3]
Pos 04-09: [BUFFER CARS 1 to 6]       <-- 6 Non-Hazardous Cars (> 5 required under 49 CFR § 174.85)
Pos 10-13: [VINYL CHLORIDE TANK CARS] <-- Placarded Class 2.1 Flammable / Inhalation Hazard
Pos 14-21: [BUFFER CARS 7 to 14]      <-- 8 Non-Hazardous Cars (> 1 required from ignition source)
Pos 22:    [MECHANICAL REEFER CAR]    <-- Continuous Diesel / Propane Ignition Source
Pos 23-74: [GENERAL FREIGHT CARS]     <-- 52 Mixed Merchandise Cars
Pos 75:    [DISTRIBUTED POWER DPU]    <-- Manned Trailing Unit (> 5 cars from hazmat block)
=============================================================================================================
```

#### Execution of Placement Optimization:
1. **Locomotive Buffer Constraint:** Vinyl chloride tank cars must be separated from locomotive units 1–3 by $\ge 5$ non-hazardous cars. Earliest allowable position for the first hazmat car:
   $$\text{Pos}_{\min} = 3 + 5 + 1 = 9$$
2. **Mechanical Reefer Buffer Constraint:** Tank cars must be separated from Car 22 by at least 1 non-hazardous car ($|i - 22| \ge 2$). Positions 21 and 23 are strictly forbidden.
3. **Rear End DPU Buffer Constraint:** Tank cars must be separated from Car 75 by $\ge 5$ non-hazardous cars:
   $$\text{Pos}_{\max} = 75 - 5 - 1 = 69$$
4. **Selected Consist Assignment:** The 4 vinyl chloride tank cars are marshaled at **Positions 10, 11, 12, 13**.
   - Separation from lead locomotives: $10 - 3 - 1 = 6$ buffer cars ($> 5$). **Compliant.**
   - Separation from mechanical reefer: $22 - 13 - 1 = 8$ buffer cars ($> 1$). **Compliant.**
   - Separation from rear DPU: $75 - 13 - 1 = 61$ buffer cars ($> 5$). **Compliant.**

---

### Part D: Non-Intrusive Radiographic Inspection (NII) Adjudication
At the terminal exit gate, Container B is selected for dual-energy LINAC X-ray screening:
- Unattenuated beam intensity: $I_{0,H} = 20,000\,\text{counts}$, $I_{0,L} = 20,000\,\text{counts}$.
- Detectors record transmitted photon flux across two distinct cargo zones:
  - **Zone 1 (Bulk Declared Lading):** $I_H = 1,420\,\text{counts}$, $I_L = 1,060\,\text{counts}$.
  - **Zone 2 (Dense Internal Shielding Anomaly):** $I_H = 32\,\text{counts}$, $I_L = 210\,\text{counts}$.

#### 1. Calculating Dual-Energy Attenuation Ratios
For Zone 1:
$$\mu_H x = -\ln\left(\frac{1,420}{20,000}\right) = -\ln(0.0710) = 2.645$$
$$\mu_L x = -\ln\left(\frac{1,060}{20,000}\right) = -\ln(0.0530) = 2.937$$
$$R_{\text{Zone 1}} = \frac{2.645}{2.937} = 0.9006$$

For Zone 2:
$$\mu_H x = -\ln\left(\frac{32}{20,000}\right) = -\ln(0.0016) = 6.438$$
$$\mu_L x = -\ln\left(\frac{210}{20,000}\right) = -\ln(0.0105) = 4.556$$
$$R_{\text{Zone 2}} = \frac{6.438}{4.556} = 1.4131$$

#### 2. Material Discrimination and Atomic Number Reconstruction
Using the calibrated detector calibration equation:
$$Z_{\text{eff}} = 6.0 + 130.0 \cdot \max(0.0, R - 0.82)^{1.8}$$
- For Zone 1:
  $$Z_{\text{eff},1} = 6.0 + 130.0 \cdot (0.9006 - 0.82)^{1.8} = 6.0 + 130.0 \cdot (0.0806)^{1.8} = 6.0 + 130.0 \cdot 0.01077 = 7.40$$
  Material Classification: Low-$Z$ organic / nitrate oxidizer matrix ($Z_{\text{eff}} \approx 7.4$, consistent with declared UN 1479 inorganic nitrate).
- For Zone 2:
  $$Z_{\text{eff},2} = 6.0 + 130.0 \cdot (1.4131 - 0.82)^{1.8} = 6.0 + 130.0 \cdot (0.5931)^{1.8} = 6.0 + 130.0 \cdot 0.3897 = 56.66$$
  Material Classification: High-$Z$ heavy metal shielding ($Z_{\text{eff}} \approx 56.7$, indicating lead ($Z=82$) / tungsten ($Z=74$) shielding).
- Automated Threat Recognition Trigger: Zone 2 triggers a critical **High-$Z$ Special Nuclear Material (SNM) Shielding Alarm**. Container is interdicted, isolated in a radiation-shielded secondary inspection bay, and evaluated with gamma spectroscopy.

---

## 7. Code, query, or calculation

The following self-contained Python script validates multi-modal dangerous goods segregation, calculates aircraft dry ice steady-state concentrations, and reconstructs effective atomic numbers ($Z_{\text{eff}}$) from dual-energy radiographic transmission measurements.

```python
import math
from typing import Tuple

def check_imdg_segregation(req_level: int, on_deck: bool, d_long: int, d_trans: int) -> bool:
    """Verify cellular container separation satisfies IMDG Code 7.2 requirements."""
    if req_level == 1:  # Away from
        return (d_long >= 1 or d_trans >= 1)
    elif req_level == 2:  # Separated from
        return (d_long >= 2 or (d_trans >= 2 if on_deck else False))
    elif req_level == 3:  # Separated by complete compartment
        return d_long >= 3
    elif req_level == 4:  # Separated longitudinally by intervening compartment
        return d_long >= 4
    return True

def dry_ice_steady_state_ppm(mass_kg: float, subl_rate_hr: float, q_vent_m3_hr: float) -> Tuple[float, bool]:
    """Calculate steady-state cargo hold CO2 concentration and verify FAA 5,000 ppm limit."""
    rho_co2 = 1.48  # kg/m3 at 82.5 kPa cabin pressure and 20 C
    c_amb = 0.00042  # 420 ppm background
    co2_vol_flow = (mass_kg * subl_rate_hr) / rho_co2
    c_steady = c_amb + (co2_vol_flow / q_vent_m3_hr)
    c_ppm = c_steady * 1_000_000.0
    return c_ppm, (c_ppm <= 5000.0)

def dual_energy_zeff(i_h: float, i0_h: float, i_l: float, i0_l: float) -> Tuple[float, float, str]:
    """Reconstruct effective atomic number (Z_eff) and classify material from 6/9 MeV X-ray."""
    mu_h_x = -math.log(i_h / i0_h)
    mu_l_x = -math.log(i_l / i0_l)
    r = mu_h_x / mu_l_x
    z_eff = 6.0 + 130.0 * (max(0.0, r - 0.82) ** 1.8)
    if z_eff < 10.0:
        mat_class = "Organic / Energetic / Polymer"
    elif z_eff <= 26.0:
        mat_class = "Inorganic / Structural Metal"
    else:
        mat_class = "Heavy Metal / High-Z Shielding"
    return r, z_eff, mat_class

if __name__ == "__main__":
    # 1. IMDG Segregation Check: Level 2 on-deck with 2 transverse row gaps
    imdg_pass = check_imdg_segregation(req_level=2, on_deck=True, d_long=0, d_trans=2)
    print(f"IMDG Level 2 Segregation Pass: {imdg_pass}")

    # 2. Air Cargo Dry Ice Ventilation Check
    co2_ppm, safe_co2 = dry_ice_steady_state_ppm(650.0, 0.0075, 1080.0)
    print(f"Steady-State CO2: {co2_ppm:.1f} ppm | Safe (<5000 ppm): {safe_co2}")

    # 3. Dual-Energy Radiographic Material Classification
    r_val, z_val, classification = dual_energy_zeff(32.0, 20000.0, 210.0, 20000.0)
    print(f"Radiography Ratio R: {r_val:.4f} | Z_eff: {z_val:.2f} | Class: {classification}")
```

### Script Execution and Output
Executing the script with standard Python 3 (`python -c` or script file) yields the exact expected results:
```text
IMDG Level 2 Segregation Pass: True
Steady-State CO2: 3469.9 ppm | Safe (<5000 ppm): True
Radiography Ratio R: 1.4129 | Z_eff: 56.74 | Class: Heavy Metal / High-Z Shielding
```
The implementation is mathematically robust, free of third-party dependencies, and executes in less than 5 milliseconds.

---

## 8. Data, standards, and regulatory context

Dangerous Goods operations are strictly governed by mandatory international conventions and federal statutory codes. Compliance is non-negotiable; violations incur civil penalties, terminal operating license revocations, and criminal prosecution.

```text
=============================================================================================================
                      INTERNATIONAL AND FEDERAL HAZMAT & SECURITY GOVERNANCE
=============================================================================================================
Statutory Body / Standard       | Jurisdiction       | Scope and Key Operational Mandates
--------------------------------|--------------------|-------------------------------------------------------
IMO IMDG Code                   | Maritime Transport | Mandatory under SOLAS Chapter VII and MARPOL Annex III.
(Amendment 41-22 / 42-24)       | (Global Waters)    | Governs packaging, marking, labeling, placarding,
                                |                    | 3D vessel cellular segregation tables, and Stowage
                                |                    | Categories A through E.
--------------------------------|--------------------|-------------------------------------------------------
ICAO Annex 18 &                 | Commercial Aviation| Legally binding technical instructions for packaging,
IATA DGR (66th / 67th Editions) | (Global Airspace)  | net quantity limits per package, PAX vs CAO, NOTOC
                                |                    | pilot acceptance sign-off, and lithium battery limits.
--------------------------------|--------------------|-------------------------------------------------------
US 49 CFR Parts 100–185         | US Domestic        | Governed by PHMSA and FRA. Part 172 mandates shipping
(PHMSA / FRA Hazmat Regulations)| Multi-Modal & Rail | papers and placards. Part 174 mandates rail train
                                |                    | placement rules (§ 174.85) and buffer car counts.
--------------------------------|--------------------|-------------------------------------------------------
IMO ISPS Code                   | Port Facilities &  | Mandatory under SOLAS Chapter XI-2. Establishes PFSO,
(Part A Mandatory, Part B Rec.) | Commercial Ships   | Port Facility Security Plans (PFSP), Declarations of
                                |                    | Security (DoS), and MARSEC 1/2/3 escalation levels.
--------------------------------|--------------------|-------------------------------------------------------
US 33 CFR Part 105              | US Maritime        | US Coast Guard domestic implementation of ISPS Code.
(Maritime Security: Facilities) | Navigable Waters   | Mandates TWIC biometric credentialing, access control,
                                |                    | and vehicle screening ratios under MARSEC directives.
--------------------------------|--------------------|-------------------------------------------------------
ANSI N42.44 / N42.46 &          | Radiographic       | Technical standards defining minimum image resolution,
ISO 20836 Radiography Specs     | Inspection Systems | wire penetration, and effective atomic number (Z_eff)
                                |                    | discrimination accuracy for cargo container NII portals.
=============================================================================================================
```

### The Legal Role of the Notification to Captain (NOTOC)
Under ICAO Annex 18, Chapter 9, an airline cannot dispatch a flight carrying dangerous goods until the Pilot-in-Command (PIC) has signed the physical or certified digital NOTOC. The NOTOC legally confirms:
1. Exact container, pallet, or lower-deck compartment location.
2. Proper Shipping Name, UN/ID number, hazard class, and subsidiary risks.
3. Total net quantity of dangerous goods per package and aggregate mass.
4. Compliance with Cargo Aircraft Only (CAO) authorizations.
5. In-flight emergency drill codes established in ICAO Doc 9481 (*Emergency Response Guidance for Aircraft Incidents Involving Dangerous Goods*).

### International Trade Security Frameworks: CSI, C-TPAT, and WCO SAFE
Global maritime security couples facility physical access with upstream supply chain intelligence:
- **US SAFE Port Act of 2006 (Public Law 109-347):** Mandated 100% electronic manifest transmission 24 hours prior to foreign loading (the "24-Hour Rule") and established the statutory requirement for Non-Intrusive Inspection screening and radiation portal monitoring at major container gateways.
- **Container Security Initiative (CSI):** A bilateral program operated by US Customs and Border Protection (CBP) stationing multi-disciplinary targeting teams at over 60 foreign partner mega-ports (e.g., Shanghai, Singapore, Rotterdam) to pre-screen high-risk containers before they are laded onto US-bound vessels.
- **Customs-Trade Partnership Against Terrorism (CTPAT):** A voluntary public-private partnership verifying that commercial shippers, carriers, and forwarders enforce rigorous supply chain physical security, 7-point container inspection protocols, and high-security mechanical bolt seals (ISO 17712 certified).
- **World Customs Organization (WCO) SAFE Framework:** Establishes global standards for Authorized Economic Operator (AEO) mutual recognition, harmonizing electronic Advance Cargo Information (ACI) across international customs authorities.

---

## 9. Industry practice and case evidence

Operational discipline in dangerous goods and security engineering is grounded in forensic evidence from catastrophic industrial failures and modern automated defense deployments.

### 9.1 The 2015 Port of Tianjin Explosions (Failure of Yard Segregation)
On August 12, 2015, two massive explosions occurred at a container storage depot operated by Ruihai Logistics at the Port of Tianjin, China. Forensic investigation by the State Administration of Work Safety established that 61 metric tons of dry nitrocellulose (Class 4.1) stowed inside an unventilated freight container experienced thermal decomposition under summer temperatures reaching $36^\circ\text{C}$. The resulting fire breached adjacent containers illegally storing 800 metric tons of ammonium nitrate (Class 5.1), 500 metric tons of potassium nitrate, and 700 metric tons of sodium cyanide (Class 6.1). The catastrophic detonation registered as equivalent to $21.9\,\text{metric tons of TNT}$ and generated seismic shocks equivalent to a magnitude 2.9 earthquake. The disaster demonstrated that stacking incompatible oxidizers and toxic solids directly adjacent to self-reactive flammables obliterates physical containment boundaries. China's Maritime Safety Administration subsequently mandated real-time IoT temperature monitoring and strict 50-meter separation zones for container yard hazmat stacks.

### 9.2 The 2012 *MSC Flaminia* Containership Explosion (Sub-Deck Thermal Conduction)
In July 2012, while transiting from Charleston to Antwerp, Cargo Hold 4 of the *MSC Flaminia* exploded. The chemical culprit was three 40-foot tank containers carrying divinylbenzene (DVB), a monomer prone to exothermic runaway polymerization. The carrier stowed the DVB tank containers under-deck directly adjacent to heated ship fuel tanks used to maintain heavy fuel oil (HFO) at pumping viscosity ($50^\circ\text{C}$ to $60^\circ\text{C}$). The continuous conductive heat flux from the bunker bulkhead heated the uninsulated DVB above its Self-Accelerating Polymerization Temperature (SAPT), exhausting the chemical p-tert-butylcatechol inhibitor. The runaway reaction generated severe internal overpressure, venting flammable vapor into the enclosed cargo hold, which ignited upon contacting cargo hold electrical fixtures. This case led to revisions in IMDG Code Stowage Categories, mandating that heat-sensitive polymerizing substances be stowed "Away from sources of heat" with mandatory deck-level temperature monitoring.

### 9.3 The 2023 East Palestine Rail Derailment (Wayside Hot-Box Detection & Buffer Consist)
On February 3, 2023, Norfolk Southern freight train 32N derailed in East Palestine, Ohio. National Transportation Safety Board (NTSB) investigation report R-24-01 established that a mechanical roller bearing on Car 23 failed catastrophically, overheating to $253^\circ\text{F}$ ($140.6^\circ\text{C}$) above ambient temperature before derailing. The consist contained 141 cars, including 20 hazardous material cars. The hazardous cars derailed in a massive kinetic pileup because mid-train buff-and-draft forces crumpled the lightweight empty cars preceding them. While buffer cars protected the lead locomotives, the concentration of hazardous cars within the derailment impact zone triggered multiple punctures and an emergency vent-and-burn of five vinyl chloride cars. The incident resulted in FRA safety advisories mandating tighter wayside hot-box detector thresholds ($170^\circ\text{F}$ above ambient) and revised train makeup algorithms to distribute trailing tonnage evenly.

### 9.4 Automated Drive-Through (ADT) Radiographic Portals at the Port of Rotterdam
To process over 14 million TEUs annually without generating gate gridlock, the Port of Rotterdam and Dutch Customs deployed high-throughput Automated Drive-Through (ADT) gantry scanners. Prime mover trucks drive through the dual-energy linear accelerator portal at speeds between $8$ and $15\,\text{km/h}$. The LINAC pulses at $6/9\,\text{MeV}$, but suppresses beam firing while the driver cab transits the portal, initiating radiographic emission only after the cab passes the entry photocell array. The system captures dual-energy transmission imagery across the full 40-foot container length within 12 seconds, processing the radiography through Automated Threat Recognition (ATR) convolutional neural networks that segment cargo into 3D density and $Z_{\text{eff}}$ maps. This architecture screens over 120 containers per hour per lane while keeping driver radiation exposure strictly below statutory public limits ($< 0.1\,\mu\text{Sv}$ per scan).

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign / Red Flag | Physical / Operational Mechanism | Consequence | Engineering Mitigation & Control |
|---|---|---|---|---|
| **Undelivered / Missing NOTOC** | Flight crew receives loadsheet without dangerous goods annex at ramp gate. | Ground handler fails to sync warehouse acceptance data with departure control system. | Flight departs with undeclared DG; crew lacks emergency drill codes in fire. | Hard interlock in electronic flight bag (EFB); dispatch release blocked without PIC sign-off. |
| **Bunker Heat Degradation** | Stowage plan assigns Class 4.1 or Class 5.2 under-deck adjacent to fuel bunker. | Load planner ignores vessel tank heating coils; HFO heated to $60^\circ\text{C}$ conducts heat. | Bulkhead conducts thermal flux; cargo exceeds SADT and deflagrates. | TOS automated validation rule blocking DG assignment adjacent to heated fuel bulkheads. |
| **Incompatible Yard Co-Location** | Drayage driver drops Class 5.1 oxidizer container in same ground slot row as Class 3. | Terminal gate automation fails to cross-check IMDG segregation table during stack assignment. | Chemical leak or spill causes contact between oxidizer and fuel, triggering spontaneous ignition. | Real-time TOS spatial constraint engine enforcing 2-slot dynamic buffer zones in container yard. |
| **Dry Ice Hold Hypercapnia** | Flight crew reports headache, elevated breathing rate, or cabin gas sensor alarm. | Forwarder tenders dry ice with sub-standard packaging insulation ($\dot{r}_{\text{sub}} > 2\%/\text{hr}$). | Compartment $CO_2$ exceeds $5,000\,\text{ppm}$; flight crew suffers cognitive impairment. | Mandatory verification of packaging sublimation certification and ECS ventilation flow rates. |
| **Rail Buffer Car Omission** | Manifest train departs yard with placarded hazmat car coupled in Position 2. | Shunting crew cuts corners during humping or flat switching to meet departure timetable. | Hazmat car exposed to direct collision impact and locomotive engine fire. | Locomotive cab signaling and PTC interlock displaying consist violation alarm before brake release. |
| **Li-Ion Thermal Runaway** | Rapid temperature rise or VOC gas detection in ULD staging area. | Internal dendritic short circuit in low-quality or damaged cell propagates exothermically. | High-velocity flammable gas venting ($H_2, CO$), dense toxic smoke, unsuppressible fire. | Class C cargo holds with continuous smoke detection, Halon 1301 suppression, fire containment covers. |
| **MARSEC Gate Gridlock** | Coast Guard escalates to MARSEC 2; truck queue extends $> 2\,\text{km}$ onto highway. | Port gate fails to activate overflow inspection lanes as physical screening increases to $35\%$. | Terminal paralyzed; drayage operations halt; port access roads blocked by gridlock. | Dynamic gate reservation system throttling inbound appointments and activating auxiliary staging yards. |
| **Radiographic False Alarm Surge** | NII screener flags multiple containers of agricultural goods (cocoa, coffee, grains). | Natural density variations and trace organic minerals trigger low-$Z$ anomaly heuristics. | Secondary inspection bays overwhelmed; gate inspection dwell times surge by $300\%$. | Multi-point calibration curves and adaptive energy-dependent thresholding in ATR software. |
| **Undeclared Hazmat E-Commerce** | Package leaking unidentified liquid or exhibiting anomalous weight on sorter scale. | E-commerce shipper misdeclares lithium batteries, aerosols, or perfumes as general cargo. | Undeclared dangerous goods loaded into passenger aircraft holds lacking fire suppression. | High-speed dual-view CT parcel scanners deployed at air express hub induction belts. |
| **Weathered / Missing Placards** | Container arrives at rail interchange with faded, torn, or absent hazmat placards. | Extreme ocean spray and ultraviolet degradation peel placards from container exterior. | First responders in derailment misidentify chemical hazard, applying incorrect fire suppression. | Mandatory four-sided reflective vinyl placarding with permanent mechanical placard holders. |
| **Class 4.3 Water Reactivity** | Water-reactive shipment exposed to rain or terminal fire sprinkler deluge. | Container roof punctures allow rainwater ingress onto sodium metal or calcium carbide. | Release of highly flammable acetylene or hydrogen gas, triggering explosive ignition. | Hermetically sealed packaging; stowage in strictly dry under-deck locations away from deluge pipes. |

---

## 11. Assumptions and limits

| Theoretical Assumption | Operational / Engineering Purpose | Failure Scenario / Boundary Limit | Verification / Validation Test |
|---|---|---|---|
| **Constant Dry Ice Sublimation ($\dot{r}_{\text{sub}} = \text{const}$)** | Simplifies mass balance differential equation to linear first-order system. | Packaging outer temperature increases during tarmac ramp delay, causing $\dot{r}_{\text{sub}}$ to spike by $300\%$. | Environmental chamber thermal decay testing across $0^\circ\text{C}$ to $45^\circ\text{C}$ ambient ranges. |
| **Uniform Hold Air Mixing ($C(t)$ homogeneous)** | Enables lumped-parameter ODE modeling of cabin and cargo compartment gas dispersion. | Stagnant air pockets form behind contoured ULDs, causing localized $CO_2$ concentration to exceed $3.0\%$. | Multi-sensor gas tracer testing ($SF_6$) under varying environmental control system (ECS) fan speeds. |
| **Hopkinson-Cranz Blast Scaling** | Predicts blast overpressure from point-source explosive mass ($Z = R / W_{\text{TNT}}^{1/3}$). | Confined container hold geometry creates complex internal shock wave reflections and channeling. | Computational fluid dynamics (CFD) blast modeling accounting for container corrugated steel boundaries. |
| **Mono-Energetic Beam Approximation** | Simplifies dual-energy transmission mathematics to linear Beer-Lambert ratio. | Real LINACs emit continuous bremsstrahlung spectra, causing beam hardening through dense targets. | Beam-hardening polynomial correction algorithms calibrated with step-wedge steel/plastic phantoms. |
| **Deterministic Buffer Car Count** | Assumes 5 non-hazardous buffer cars guarantee structural protection for locomotive crew. | High-speed derailments ($> 50\,\text{mph}$) create massive pileups that crumple $> 10$ cars into the locomotive. | Non-linear dynamic crash simulations (LS-DYNA) modeling draft gear failure and derailment kinematics. |
| **Static Chemical Stability** | Assumes dangerous goods properties remain invariant during transit duration. | Monomer stabilizers deplete over time, triggering auto-polymerization after scheduled voyage delays. | Chemical shelf-life testing and mandatory maximum transit duration cutoffs for stabilized monomers. |
| **100% TWIC Badge Compliance** | Assumes physical credentialing eliminates unauthorized perimeter security breaches. | Credential cloning or tailgating allows unauthorized individuals to access secure terminal zones. | Biometric fingerprint/iris verification integrated with turnstile anti-tailgating optical barriers. |
| **Linear Gate Queueing Arrival Rates** | Assumes truck arrivals follow stationary Poisson process during security escalation. | Panicked drayage drivers rush to terminal gates before curfew, creating severe non-stationary arrival surges. | Time-varying queueing simulations and dynamic appointment cancellation mechanisms under MARSEC 2. |
| **Infinite Steel Bulkhead Thermal Sink** | Assumes ship transverse bulkheads absorb localized heat without transmitting destructive flux. | Prolonged container pool fire ($> 2\,\text{hours}$) heats bulkhead steel above $800^\circ\text{C}$, igniting cargo behind it. | Solas A-60 fire barrier structural insulation standards and bulkhead water boundary deluge cooling. |

---

## 12. Exercises

### Exercise 1: IMDG 3D Vessel Bay-Row-Tier Segregation Audit
A 10,000 TEU container vessel is planning stowage for four dangerous goods containers on Hatch Cover 08 (Bays 13–15):
- Container 1: 40-ft container, UN 1170, Ethanol (Class 3, PG II), Slot 14-04-84.
- Container 2: 40-ft container, UN 1944, Matches, Safety (Class 4.1, PG III), Slot 14-08-84.
- Container 3: 40-ft container, UN 3105, Organic Peroxide Type D, Liquid (Class 5.2), Slot 14-02-86.
- Container 4: 40-ft container, UN 2024, Mercury Compound, Liquid, N.O.S. (Class 6.1, PG I), Slot 14-06-84.

*Deliverable:* Construct the transverse and vertical separation distance matrix between all pairs. Using the IMDG Code General Segregation Table, identify all segregation violations and propose a compliant revised slot plan.
*Acceptance Criteria:* Document exact row and tier deltas; cite IMDG Table 7.2.4 requirements; provide a collision-free re-allocation.
*Estimated Time:* 45 minutes.

### Exercise 2: Aircraft Cargo Hold Dry Ice Sublimation Kinetics
A pharmaceutical shipper books 850 kg of dry ice (UN 1845) on an Airbus A350-900 lower forward cargo compartment.
- Compartment net volume: $V = 55.0\,\text{m}^3$.
- Fresh air ventilation exchange rate: $Q = 14.0\,\text{m}^3/\text{min}$.
- Cruise compartment pressure: $P = 80.0\,\text{kPa}$; Temperature: $T = 15.0^\circ\text{C}$.
- Packaging dry ice sublimation rate: $\dot{r}_{\text{sub}} = 0.95\%/\text{hr}$.
- Ambient background $CO_2$: $400\,\text{ppm}$. Statutory continuous limit: $5,000\,\text{ppm}$.

*Deliverable:* 
1. Calculate the density of gaseous $CO_2$ at cruising hold conditions.
2. Formulate the transient concentration equation $C(t)$ and compute concentration after 2 hours of flight.
3. Compute the final steady-state concentration $C_{\text{ss}}$ and evaluate compliance.
4. Calculate the maximum safe dry ice mass ($M_{\max}$) allowable in this compartment.
*Acceptance Criteria:* Show all algebraic steps and units; final answers must be accurate to within 1 ppm.
*Estimated Time:* 40 minutes.

### Exercise 3: Freight Rail Consist 49 CFR Part 174 Compliance Design
A Class I railroad is marshaling a 90-car manifest train from an automated classification yard. The consist includes:
- Lead power: 4 locomotives (Positions 1, 2, 3, 4).
- 6 loaded placarded tank cars carrying UN 1017 (Chlorine, Class 2.3 Toxic Inhalation Hazard, Hazard Zone B).
- 2 empty uncleaned tank cars with residue placards for UN 1203 (Gasoline, Class 3).
- 1 mechanical refrigerator car with diesel compressor (Position 40).
- 1 flat car loaded with structural steel beams prone to shifting (Position 65).
- Manned rear caboose (Position 90).

*Deliverable:* Formulate a mixed-integer linear programming (MILP) assignment problem or block schedule assigning positions to the 6 chlorine tank cars and 2 residue cars that minimizes total consist reshuffling while strictly satisfying US 49 CFR § 174.85 buffer car rules.
*Acceptance Criteria:* Explicitly verify buffer counts to lead power, rear caboose, mechanical reefer, and shiftable flat car. State the exact car positions assigned.
*Estimated Time:* 50 minutes.

### Exercise 4: Dual-Energy Radiographic Screening Adjudication
An automated NII gantry scanner records the following detector counts across three suspicious container pixels:
- Beam unattenuated flux: $I_{0,H} = 25,000$, $I_{0,L} = 25,000$.
- Pixel A: $I_H = 1,850$, $I_L = 1,320$.
- Pixel B: $I_H = 340$, $I_L = 390$.
- Pixel C: $I_H = 18$, $I_L = 165$.

*Deliverable:*
1. Calculate the dual-energy attenuation ratio $R$ for each pixel.
2. Reconstruct effective atomic number $Z_{\text{eff}}$ using $Z_{\text{eff}} = 6.0 + 130.0 \cdot \max(0.0, R - 0.82)^{1.8}$.
3. Classify each pixel into its material category (Organic, Structural Metal, High-$Z$ Shielding).
4. Identify which pixel warrants an immediate radiological secondary alarm and explain why.
*Acceptance Criteria:* Carry arithmetic to four decimal places; state specific threat implications for each material class.
*Estimated Time:* 35 minutes.

### Exercise 5: Forensic Critique of the *MSC Flaminia* Incident
Review the documented facts of the 2012 *MSC Flaminia* explosion.
*Deliverable:* Write a technical memorandum (400–500 words) critiquing the stowage planning failure. Analyze why the carrier's automated stowage software failed to prevent the tragedy, focusing on:
1. The physical mechanism of conductive heat transfer from heavy fuel oil bunker tanks.
2. The thermodynamic interaction between ambient temperature and chemical polymerization inhibitors.
3. Specific amendments made to the IMDG Code following the incident.
*Acceptance Criteria:* Address the Arrhenius kinetics of SAPT, bunker tank heating cycles, and Stowage Category revisions.
*Estimated Time:* 45 minutes.

### Exercise 6: Port Security Gate Queuing Under MARSEC Level 2 Escalation
A container terminal gate has $s = 8$ service lanes. Drayage trucks arrive at rate $\lambda = 160\,\text{trucks/hour}$. Baseline processing time is $1/\mu_0 = 1.8\,\text{minutes}$. Secondary physical inspection time is $1/\mu_{\text{insp}} = 16.0\,\text{minutes}$.
1. At MARSEC 1, secondary inspection fraction is $p_1 = 0.05$. Calculate system utilization $\rho_1$ and verify stability.
2. The Coast Guard elevates the port to MARSEC 2. The inspection fraction increases to $p_2 = 0.30$. Calculate effective service rate $\mu_{\text{eff},2}$ and system utilization $\rho_2$.
3. If the queue is unstable, calculate the rate of queue growth ($\text{trucks/hour}$) and determine how many additional inspection lanes must be opened to restore stability ($\rho < 0.85$).
*Deliverable:* Full queueing model derivation and numerical capacity plan.
*Acceptance Criteria:* Show all intermediate equations and lane calculations.
*Estimated Time:* 40 minutes.

### Exercise 7: Non-Intrusive Inspection Automated Threat Recognition ROC Thresholding
An Automated Threat Recognition (ATR) algorithm evaluates containers for illicit explosives. The detector decision metric $x$ follows normal distributions under benign and threat conditions:
- Benign cargo ($H_0$): $x \sim \mathcal{N}(\mu_0 = 1.0, \sigma_0^2 = 0.25) \implies \sigma_0 = 0.5$.
- Explosive threat ($H_1$): $x \sim \mathcal{N}(\mu_1 = 2.5, \sigma_1^2 = 0.25) \implies \sigma_1 = 0.5$.

*Deliverable:*
1. Formulate the Neyman-Pearson decision rule for threshold $\gamma$.
2. If regulatory policy mandates a Probability of Detection $P_D \ge 0.99$, calculate the required decision threshold $\gamma$.
3. Calculate the resulting False Alarm Rate ($P_{FA}$) at this threshold.
4. If the port scans 1,200 containers per day, how many false alarms must secondary inspection teams adjudicate daily?
*Acceptance Criteria:* Use standard normal cumulative distribution tables ($Z$-scores); show exact probability integrals.
*Estimated Time:* 40 minutes.

---

### Exercise 8: Lithium-Ion Battery Aircraft Hold Thermal Runaway Pressure Spike
A bulk cargo consignment of 2,500 Ah (nominal 3.7 V, 9.25 kWh total energy) of lithium-ion cells packed under Packing Instruction 965 Section IA undergoes uncontained thermal runaway inside a Boeing 777F lower forward cargo compartment.
- Compartment net volume: $V_{\text{hold}} = 69.8\,\text{m}^3$.
- Initial hold pressure: $P_0 = 82.5\,\text{kPa}$; Initial hold temperature: $T_0 = 20^\circ\text{C}$ ($293.15\,\text{K}$).
- Vented gas generation metric: $v_{\text{gas}} = 1.40\,\text{L/Ah}$ (measured at standard temperature and pressure: $P_{\text{STP}} = 101.325\,\text{kPa}, T_{\text{STP}} = 273.15\,\text{K}$).
- Vented gas temperature during ejection: $T_{\text{vent}} = 650^\circ\text{C} = 923.15\,\text{K}$.
- Structural hold decompression panel blowout differential pressure limit: $\Delta P_{\text{limit}} = 12.0\,\text{kPa}$.

*Deliverable:*
1. Calculate the total moles ($n_{\text{vent}}$) of gas ejected during the runaway event.
2. Assuming instantaneous mixing and that the hold remains sealed prior to blowout, calculate the theoretical peak internal hold pressure ($P_{\text{peak}}$) and the overpressure differential ($\Delta P = P_{\text{peak}} - P_0$) if the gas cools to an average mixed temperature of $T_{\text{mix}} = 120^\circ\text{C}$ ($393.15\,\text{K}$).
3. Determine whether the compartment liner structural blowout threshold ($\Delta P_{\text{limit}}$) is exceeded.
4. Calculate the minimum pressure relief vent area ($A_{\text{vent}}$) required to discharge the gas within $t_{\text{vent}} = 10\,\text{seconds}$ without exceeding $\Delta P_{\text{limit}}$, assuming sonic orifice flow:
   $$\dot{m} = C_d \cdot A_{\text{vent}} \cdot P_0 \sqrt{\frac{\gamma M}{R T_0} \left(\frac{2}{\gamma + 1}\right)^{\frac{\gamma+1}{\gamma-1}}}$$
   with discharge coefficient $C_d = 0.62$, ratio of specific heats $\gamma = 1.30$, and molar mass $M = 0.028\,\text{kg/mol}$.
*Acceptance Criteria:* State all thermodynamic state equations; carry numerical calculations to three significant digits; state airworthiness implications.
*Estimated Time:* 45 minutes.

---

## 13. Self-check answers

### Solution to Exercise 1: IMDG 3D Vessel Segregation Audit
1. **Pairwise IMDG General Segregation Table Requirements:**
   - Class 3 (UN 1170) vs. Class 4.1 (UN 1944): Segregation `1` ("Away from").
   - Class 3 (UN 1170) vs. Class 5.2 (UN 3105): Segregation `2` ("Separated from").
   - Class 3 (UN 1170) vs. Class 6.1 (UN 2024): No segregation required (`0` / safe to co-load unless subsidiary risks exist).
   - Class 4.1 (UN 1944) vs. Class 5.2 (UN 3105): Segregation `2` ("Separated from").
   - Class 4.1 (UN 1944) vs. Class 6.1 (UN 2024): Segregation `1` ("Away from").
   - Class 5.2 (UN 3105) vs. Class 6.1 (UN 2024): Segregation `1` ("Away from").

2. **Evaluating the Planned Slots:**
   - **Container 1 (Class 3 at 14-04-84) vs. Container 3 (Class 5.2 at 14-02-86):**
     - Bay Delta = 0. Row Delta: Row 04 to Row 02 $\implies \Delta R = 1$ container space.
     - Tier Delta: Tier 84 to Tier 86 $\implies \Delta T = 1$ tier (stacked directly on adjacent row).
     - Requirement for Segregation `2` On-Deck: Must be separated by $\ge 2$ container pitches longitudinally OR $\ge 2$ container rows transversely.
     - Actual spacing is only 1 row space transversely. **CRITICAL VIOLATION: Fails Segregation Level 2.**
   - **Container 3 (Class 5.2 at 14-02-86) vs. Container 4 (Class 6.1 at 14-06-84):**
     - Row Delta: Row 02 to Row 06 $\implies \Delta R = 2$ container spaces.
     - Requirement for Segregation `1`: $\ge 1$ container row space. Compliant.
   - **Container 2 (Class 4.1 at 14-08-84) vs. Container 3 (Class 5.2 at 14-02-86):**
     - Row Delta: Row 02 to Row 08 $\implies \Delta R = 3$ container spaces. Compliant with Segregation `2`.

3. **Compliant Re-allocation:**
   - Move Container 3 (Class 5.2) to **Slot 14-12-84** (Row 12).
   - Spacing from Container 1 (Row 04): Delta is 4 container rows ($> 2$). Spacing from Container 4 (Row 06): Delta is 3 container rows ($> 2$). Spacing from Container 2 (Row 08): Delta is 2 container rows ($\ge 2$). All IMDG segregation constraints are fully satisfied.

---

### Solution to Exercise 2: Air Cargo Dry Ice Ventilation
1. **Density of Gaseous Carbon Dioxide:**
   $$\rho_{CO_2} = \frac{P \cdot M}{R \cdot T} = \frac{80,000\,\text{Pa} \times 0.04401\,\text{kg/mol}}{8.3145\,\text{J/(mol}\cdot\text{K)} \times 288.15\,\text{K}} = \frac{3,520.8}{2,395.8} = 1.4696\,\text{kg/m}^3$$
2. **Volumetric Gas Generation and Ventilation Rate:**
   $$Q = 14.0\,\text{m}^3/\text{min} \times 60\,\text{min/hr} = 840.0\,\text{m}^3/\text{hr}$$
   $$\dot{m}_{CO_2} = 850.0\,\text{kg} \times 0.0095\,\text{hr}^{-1} = 8.075\,\text{kg/hr}$$
   $$\dot{V}_{CO_2} = \frac{\dot{m}_{CO_2}}{\rho_{CO_2}} = \frac{8.075}{1.4696} = 5.4947\,\text{m}^3/\text{hr}$$
   Air exchange turnover factor: $k = \frac{Q}{V} = \frac{840.0}{55.0} = 15.2727\,\text{hr}^{-1}$.
3. **Transient Concentration at $t = 2.0\,\text{hours}$:**
   $$C(t) = C_{\text{amb}} e^{-k t} + \frac{\dot{V}_{CO_2}}{Q}\left(1 - e^{-k t}\right)$$
   Since $k \cdot t = 15.2727 \times 2 = 30.55$, $e^{-30.55} \approx 5.4 \times 10^{-14} \approx 0$.
   The system reaches steady state well before 2 hours.
   $$C(2.0) \approx C_{\text{ss}} = 0.000400 + \frac{5.4947}{840.0} = 0.000400 + 0.006541 = 0.006941 = 0.6941\% \quad (6,941\,\text{ppm})$$
   **Evaluation:** $C_{\text{ss}} = 6,941\,\text{ppm} > 5,000\,\text{ppm}$. **UNSAFE AND ILLEGAL.** The flight cannot be dispatched with this payload under standard ventilation.
4. **Maximum Allowable Dry Ice Mass:**
   $$M_{\max} = \frac{(C_{\max} - C_{\text{amb}}) \cdot \rho_{CO_2} \cdot Q}{\dot{r}_{\text{sub}}} = \frac{(0.0050 - 0.00040) \times 1.4696 \times 840.0}{0.0095} = \frac{0.00460 \times 1,234.46}{0.0095} = \frac{5.6785}{0.0095} = 597.74\,\text{kg}$$
   The shipper must reduce dry ice content to $\le 597.7\,\text{kg}$ or utilize active refrigeration ULDs (e.g., Envirotainer / CSafe).

---

### Solution to Exercise 4: Dual-Energy Radiographic Screening
1. **Ratio Calculations:**
   - **Pixel A:** $\mu_H x = -\ln(1,850/25,000) = 2.6044$; $\mu_L x = -\ln(1,320/25,000) = 2.9412$.
     $$R_A = \frac{2.6044}{2.9412} = 0.8855$$
   - **Pixel B:** $\mu_H x = -\ln(340/25,000) = 4.2977$; $\mu_L x = -\ln(390/25,000) = 4.1605$.
     $$R_B = \frac{4.2977}{4.1605} = 1.0330$$
   - **Pixel C:** $\mu_H x = -\ln(18/25,000) = 7.2363$; $\mu_L x = -\ln(165/25,000) = 5.0207$.
     $$R_C = \frac{7.2363}{5.0207} = 1.4413$$

2. **Effective Atomic Number ($Z_{\text{eff}}$) Reconstruction:**
   - Pixel A: $Z_{\text{eff}} = 6.0 + 130.0 \cdot (0.8855 - 0.82)^{1.8} = 6.0 + 130.0 \cdot (0.0655)^{1.8} = 6.0 + 130.0 \cdot 0.00742 = 6.96$.
   - Pixel B: $Z_{\text{eff}} = 6.0 + 130.0 \cdot (1.0330 - 0.82)^{1.8} = 6.0 + 130.0 \cdot (0.2130)^{1.8} = 6.0 + 130.0 \cdot 0.06240 = 14.11$.
   - Pixel C: $Z_{\text{eff}} = 6.0 + 130.0 \cdot (1.4413 - 0.82)^{1.8} = 6.0 + 130.0 \cdot (0.6213)^{1.8} = 6.0 + 130.0 \cdot 0.4226 = 60.94$.

3. **Classification and Alarm Adjudication:**
   - **Pixel A ($Z_{\text{eff}} = 6.96$):** Low-$Z$ Organic / Polymer / Nitrate matrix (consistent with plastics, food, or energetic organic compounds).
   - **Pixel B ($Z_{\text{eff}} = 14.11$):** Mid-$Z$ Inorganic / Light Structural Metal (consistent with aluminum, silicon, or light steel containers).
   - **Pixel C ($Z_{\text{eff}} = 60.94$):** High-$Z$ Heavy Metal Shielding. Triggers an immediate **Radiological / Special Nuclear Material Alarm**. This indicates high-density lead ($Z=82$) or tungsten shielding configured to conceal illicit radioactive isotopes. Container must be halted at gate cordon.

---

### Solution to Exercise 6: Port Security Gate Queuing Under MARSEC 2
1. **MARSEC 1 Stability:**
   $$E[S_1] = 0.95(1.8) + 0.05(16.0) = 1.71 + 0.80 = 2.51\,\text{min} \implies \mu_{\text{eff},1} = \frac{60}{2.51} = 23.904\,\text{trucks/hr/lane}$$
   Total gate capacity: $C_1 = 8 \times 23.904 = 191.23\,\text{trucks/hr}$.
   Utilization: $\rho_1 = \frac{160}{191.23} = 0.837 < 1.0$. System is stable.
2. **MARSEC 2 Collapse:**
   $$E[S_2] = 0.70(1.8) + 0.30(16.0) = 1.26 + 4.80 = 6.06\,\text{min} \implies \mu_{\text{eff},2} = \frac{60}{6.06} = 9.901\,\text{trucks/hr/lane}$$
   Total gate capacity: $C_2 = 8 \times 9.901 = 79.21\,\text{trucks/hr}$.
   Utilization: $\rho_2 = \frac{160}{79.21} = 2.02 > 1.0$. The system is unstable.
3. **Queue Growth and Capacity Restoration:**
   Queue growth rate: $\frac{dL}{dt} = \lambda - C_2 = 160 - 79.21 = 80.79\,\text{trucks/hour}$.
   To achieve target utilization $\rho^* \le 0.85$:
   $$s^* \ge \frac{\lambda}{\rho^* \cdot \mu_{\text{eff},2}} = \frac{160}{0.85 \times 9.901} = \frac{160}{8.416} = 19.01 \implies s^* = 20\,\text{lanes}$$
   The terminal must activate $20 - 8 = 12$ auxiliary lanes or immediately implement drayage truck gate appointment reservation caps restricting arrivals to $\lambda_{\max} = 0.85 \times 79.21 = 67\,\text{trucks/hour}$.

---

### Solution to Exercise 7: Non-Intrusive Inspection ATR ROC Threshold
1. **Neyman-Pearson Threshold for $P_D \ge 0.99$:**
   Under $H_1$, $x \sim \mathcal{N}(2.5, 0.25) \implies \sigma = 0.5$.
   $$P_D = P(x \ge \gamma \mid H_1) = 1 - \Phi\left(\frac{\gamma - 2.5}{0.5}\right) \ge 0.99 \implies \Phi\left(\frac{\gamma - 2.5}{0.5}\right) \le 0.01$$
   From the standard normal $Z$-table, $\Phi(-2.326) = 0.01$.
   $$\frac{\gamma - 2.5}{0.5} = -2.326 \implies \gamma = 2.5 - (2.326 \times 0.5) = 2.5 - 1.163 = 1.337$$
2. **Resulting False Alarm Rate ($P_{FA}$):**
   Under $H_0$, $x \sim \mathcal{N}(1.0, 0.25) \implies \sigma = 0.5$.
   $$P_{FA} = P(x \ge 1.337 \mid H_0) = 1 - \Phi\left(\frac{1.337 - 1.0}{0.5}\right) = 1 - \Phi\left(\frac{0.337}{0.5}\right) = 1 - \Phi(0.674)$$
   From normal distribution tables, $\Phi(0.674) \approx 0.7498$.
   $$P_{FA} = 1 - 0.7498 = 0.2502 \quad (25.02\%)$$
3. **Operational Impact:**
   Daily false alarms requiring manual physical inspection:
   $$N_{\text{alarm}} = 1,200\,\text{containers/day} \times 0.2502 = 300.2 \approx 300\,\text{alarms/day}$$
   This demonstrates the classic ROC trade-off: forcing a $99\%$ detection probability with single-parameter thresholding generates 300 daily false alarms, overwhelming secondary inspection capacity. The engineering solution requires multi-view radiography or dual-stage automated anomaly fusion.

---

### Solution to Exercise 8: Lithium-Ion Battery Aircraft Hold Thermal Runaway
1. **Total Moles of Vented Gas:**
   Total volume of gas generated at standard temperature and pressure ($STP$):
   $$V_{\text{STP}} = 2,500\,\text{Ah} \times 1.40\,\text{L/Ah} = 3,500\,\text{L} = 3.50\,\text{m}^3$$
   Using the ideal gas law at STP ($P_{\text{STP}} = 101,325\,\text{Pa}, T_{\text{STP}} = 273.15\,\text{K}$):
   $$n_{\text{vent}} = \frac{P_{\text{STP}} \cdot V_{\text{STP}}}{R \cdot T_{\text{STP}}} = \frac{101,325 \times 3.50}{8.3145 \times 273.15} = \frac{354,637.5}{2,271.1} = 156.15\,\text{moles}$$

2. **Hold Overpressure Calculation:**
   Initial moles of air in the sealed hold:
   $$n_{\text{air},0} = \frac{P_0 \cdot V_{\text{hold}}}{R \cdot T_0} = \frac{82,500\,\text{Pa} \times 69.8\,\text{m}^3}{8.3145 \times 293.15\,\text{K}} = \frac{5,758,500}{2,437.4} = 2,362.56\,\text{moles}$$
   Total moles after thermal runaway venting:
   $$n_{\text{total}} = n_{\text{air},0} + n_{\text{vent}} = 2,362.56 + 156.15 = 2,518.71\,\text{moles}$$
   Internal hold pressure at mixed temperature $T_{\text{mix}} = 393.15\,\text{K}$ ($120^\circ\text{C}$):
   $$P_{\text{peak}} = \frac{n_{\text{total}} \cdot R \cdot T_{\text{mix}}}{V_{\text{hold}}} = \frac{2,518.71 \times 8.3145 \times 393.15}{69.8} = \frac{8,233,485}{69.8} = 117,958\,\text{Pa} = 117.96\,\text{kPa}$$
   The resulting overpressure differential across the cargo compartment ceiling liner is:
   $$\Delta P = P_{\text{peak}} - P_0 = 117.96\,\text{kPa} - 82.50\,\text{kPa} = 35.46\,\text{kPa}$$

3. **Structural Blowout Evaluation:**
   The calculated overpressure differential $\Delta P = 35.46\,\text{kPa}$ exceeds the certified decompression panel blowout limit $\Delta P_{\text{limit}} = 12.0\,\text{kPa}$ by:
   $$\frac{35.46}{12.0} = 2.955 \implies 195.5\% \text{ over structural limit}$$
   **Airworthiness Verdict:** The compartment decompression blowout panels will rupture violently within seconds of thermal runaway initiation. Dense toxic smoke, carbon monoxide, and unburned flammable gases will escape into the main cabin and cockpit floors, exactly replicating the failure mechanism of UPS Flight 6.

4. **Minimum Vent Area Calculation:**
   To maintain overpressure below $\Delta P_{\text{limit}} = 12.0\,\text{kPa}$, the relief venting mass rate must discharge the excess gas mass $m_{\text{vent}} = n_{\text{vent}} \cdot M = 156.15 \times 0.028 = 4.372\,\text{kg}$ over $t = 10.0\,\text{s}$:
   $$\dot{m}_{\text{req}} = \frac{4.372\,\text{kg}}{10.0\,\text{s}} = 0.4372\,\text{kg/s}$$
   Using the choked flow formula with $P_{\text{hold}} = P_0 + \Delta P_{\text{limit}} = 94,500\,\text{Pa}$:
   $$\psi = \sqrt{\frac{\gamma M}{R T_{\text{mix}}} \left(\frac{2}{\gamma + 1}\right)^{\frac{\gamma+1}{\gamma-1}}} = \sqrt{\frac{1.30 \times 0.028}{8.3145 \times 393.15} \left(\frac{2}{2.30}\right)^{\frac{2.30}{0.30}}} = \sqrt{1.1147 \times 10^{-5} \times (0.8696)^{7.667}} = \sqrt{1.1147 \times 10^{-5} \times 0.3392} = 1.944 \times 10^{-3}$$
   $$\dot{m} = C_d \cdot A_{\text{vent}} \cdot P_{\text{hold}} \cdot \psi \implies A_{\text{vent}} = \frac{\dot{m}_{\text{req}}}{C_d \cdot P_{\text{hold}} \cdot \psi} = \frac{0.4372}{0.62 \times 94,500 \times 1.944 \times 10^{-3}} = \frac{0.4372}{113.90} = 0.00384\,\text{m}^2 = 38.4\,\text{cm}^2$$
   A dedicated relief vent area of at least $38.4\,\text{cm}^2$ (e.g., a $6.2 \times 6.2\,\text{cm}$ opening) is required to prevent explosive structural failure.

---

## 14. Interview questions

### Question 1: Marine Stowage Segregation Physics
*Why does the IMDG Code differentiate between on-deck and under-deck segregation distances for the exact same chemical compatibility pair?*
- **What a strong answer contains:** The candidate explains that under-deck cargo holds are confined structural spaces where catastrophic failure mechanisms propagate rapidly: (1) Fire and explosion confinement prevents heat and pressure dissipation, causing localized deflagrations to build extreme overpressures; (2) Volatile flammable or toxic vapors released in an enclosed hold cannot dissipate through atmospheric dilution and pool in the bilges or circulate across the entire hold bay; (3) Firefighting access is restricted to remote carbon dioxide flooding systems, which cannot extinguish oxygen-releasing oxidizers (Class 5.1). On-deck stowage provides natural atmospheric dispersion of leaked vapors, radiative dissipation into the open air, and direct boundary water cooling via fire monitors. Consequently, under-deck stowage requires solid steel fire- and liquid-tight transverse bulkheads or intervening complete holds to achieve the same risk reduction that can be achieved on-deck across 1 or 2 container spaces.
- **What a weak answer reveals:** The candidate states that under-deck cargo is "harder to see" or assumes the difference is merely arbitrary administrative conservatism.

### Question 2: Dry Ice Aircraft Ventilation Engineering
*If an airline needs to double its dry ice vaccine transport capacity on a passenger wide-body flight, what specific operational variables can flight operations manipulate, and what are their physical limits?*
- **What a strong answer contains:** The candidate references the steady-state mass balance equation $M_{\max} = \frac{C_{\max} \cdot \rho_{CO_2} \cdot Q}{\dot{r}_{\text{sub}}}$. To increase $M_{\max}$, operations can: (1) Increase the Environmental Control System (ECS) fresh air ventilation rate $Q$ by setting engine bleed packs to "High Flow" mode (trading off a minor fuel burn penalty of $\approx 0.5\%$); (2) Partner with shippers to reduce the sublimation rate $\dot{r}_{\text{sub}}$ through validated vacuum-insulated packaging (VIP) or active refrigeration systems (reducing sublimation from $1.0\%/\text{hr}$ to $< 0.3\%/\text{hr}$); (3) Distribute the consignment across multiple isolated cargo compartments (e.g., splitting between forward, aft, and bulk holds to utilize independent ventilation zones). The candidate identifies the absolute boundary limit: engine bleed architecture limits maximum fresh air delivery, and single-engine failure contingency procedures require hold $CO_2$ to remain non-toxic if one air conditioning pack shuts down.
- **What a weak answer reveals:** The candidate suggests "opening a vent" or claims airlines can simply pack more ice because "carbon dioxide sinks to the floor."

### Question 3: Train Consist Placement under 49 CFR Part 174
*During rail switching operations, a yardmaster places five buffer cars between the lead locomotive and a block of loaded vinyl chloride tank cars. Does this satisfy 49 CFR § 174.85 for all possible train configurations?*
- **What a strong answer contains:** The candidate correctly states that while five buffer cars satisfies the lead locomotive separation rule for standard road trains ($N \ge 30$), compliance requires verifying several additional constraints: (1) The trailing end of the train must be verified: if there is an occupied caboose or manned Distributed Power Unit (DPU) at the rear, five non-hazardous buffer cars are also required between the hazmat cars and the occupied rear unit; (2) The internal train makeup must be verified: placarded tank cars cannot be placed adjacent to mechanical refrigeration cars (spark/combustion risk) or flat cars carrying shiftable lading (crush/puncture risk); (3) The buffer cars themselves must be inspected: they must be non-hazardous, non-placarded, and cannot be empty uncleaned hazmat tank cars displaying residue placards.
- **What a weak answer reveals:** The candidate says "yes" without checking rear-end power units, internal ignition sources, or the contents of the buffer cars.

### Question 4: Dual-Energy Radiographic Screening Physics
*How does dual-energy X-ray inspection discriminate between a 50 kg block of plastic explosive and a 50 kg block of structural steel, even if both exhibit the exact same thickness?*
- **What a strong answer contains:** The candidate explains that single-energy radiography measures only total beam attenuation ($I = I_0 e^{-\mu x}$), which conflates density, thickness, and material composition. Dual-energy radiography interrogates the material at two distinct photon energies (e.g., $6\,\text{MeV}$ and $9\,\text{MeV}$). At these energies, Compton scattering is proportional to electron density ($\sim Z/A \approx \text{const}$), while pair production cross-section scales with the square of the atomic number ($Z^2/A \propto Z$). By taking the logarithmic ratio of transmitted intensities $R = \frac{\ln(I_H / I_{0,H})}{\ln(I_L / I_{0,L})}$, physical thickness and mass density cancel out, leaving a pure function of the effective atomic number $Z_{\text{eff}}$. Plastic explosive (C-4, RDX) consists of low-$Z$ organic elements (carbon, hydrogen, nitrogen, oxygen: $Z_{\text{eff}} \approx 7$), yielding $R \approx 0.88–0.92$. Steel consists of iron ($Z = 26$), yielding $R \approx 1.05–1.10$. The system's automated classifier assigns distinct colors (orange for organic, blue/green for metals) based entirely on $Z_{\text{eff}}$.
- **What a weak answer reveals:** The candidate claims the system "measures the weight" or "uses radiation sensors to detect chemicals directly."

### Question 5: Port Security MARSEC Escalation Dynamics
*When a port facility elevates from MARSEC Level 1 to MARSEC Level 2, why does gate truck queueing often exhibit catastrophic non-linear delays rather than a modest linear increase?*
- **What a strong answer contains:** The candidate applies queueing theory ($M/M/s$ or $M/G/s$). Under MARSEC 1, intensive physical screening is applied to a small fraction ($p \approx 5\%$) of arrivals, maintaining gate utilization well below capacity ($\rho \approx 0.70$). Under MARSEC 2, the Coast Guard mandates screening ratios of $25\%$ to $50\%$. Because secondary physical inspections take 10 to 15 times longer than automated gate processing, the effective mean service time per truck surges. This causes the total arrival rate to exceed the degraded gate service capacity ($\lambda > s \cdot \mu_{\text{eff}}$). Once utilization $\rho \ge 1.0$, the queue transitions from stochastic steady state to deterministic accumulation, growing linearly without bound at rate $\lambda - s \mu_{\text{eff}}$. The candidate notes that mitigation requires pre-emptive demand throttling via mandatory appointment reservation quotas.
- **What a weak answer reveals:** The candidate attributes the delays solely to "slow security guards" and fails to model the queueing utilization threshold where $\rho \to 1$.

### Question 6: Lithium-Ion Battery Aircraft Risk Engineering
*Why did ICAO ban the carriage of bulk lithium-ion batteries (UN 3480) on passenger aircraft, and what technical standards govern their carriage on dedicated freighters?*
- **What a strong answer contains:** The candidate identifies the physical failure mechanism: internal dendritic shorts, manufacturing defects, or mechanical damage trigger thermal runaway in a single cell, releasing flammable organic electrolyte vapors ($C_2H_4, CO, H_2$) and oxygen from decomposing cathode oxides. Temperatures exceed $600^\circ\text{C}$, propagating runaway to adjacent cells. In an aircraft cargo hold, the Halon 1301 fire extinguishing agent suppresses open flames but cannot cool the internal chemical reaction or halt thermal runaway propagation. Furthermore, the vented gases generate explosive overpressures that can compromise cargo compartment liner integrity. Under ICAO Annex 18, UN 3480 standalone cells are banned on passenger flights. On dedicated freighters, they are restricted to: (1) Mandatory State of Charge (SoC) $\le 30\%$ at time of tender (drastically reducing electrochemical energy density and preventing thermal runaway propagation); (2) Carriage under Packing Instruction 965; (3) Fire containment covers or certified fire-resistant containers (FRCs).
- **What a weak answer reveals:** The candidate thinks lithium batteries catch fire because "they get too hot in the sun" and is unaware of the $30\%$ SoC regulatory ceiling.

### Question 7: High-Z Radiographic Screening Anomalies
*In high-energy container radiography, what constitutes a "High-Z Anomaly," and why is it treated as a potential national security emergency?*
- **What a strong answer contains:** A High-$Z$ anomaly occurs when the dual-energy transmission ratio indicates an effective atomic number $Z_{\text{eff}} \ge 70$ (e.g., lead $Z=82$, tungsten $Z=74$, uranium $Z=92$). Commercial containerized freight consists almost entirely of low-$Z$ agricultural/polymer goods ($Z < 10$) and mid-$Z$ structural metals ($Z \approx 26$). There are very few legitimate commercial reasons to transport thick, highly dense high-$Z$ metals inside general cargo containers. High-$Z$ materials are primary gamma-ray and neutron shields. When high-$Z$ shielding is detected surrounding an internal void or dense object, it suggests an intentional effort to shield and smuggle Special Nuclear Material (SNM, such as highly enriched uranium or weapons-grade plutonium) or a radiological dirty bomb to evade passive radiation portal monitors (RPM). It triggers an immediate facility security cordon and tactical team deployment.
- **What a weak answer reveals:** The candidate confuses high-$Z$ shielding with simple high density or assumes it refers to standard heavy machinery parts.

### Question 8: Self-Accelerating Decomposition Temperature (SADT)
*A chemical shipper books a tank container of organic peroxide with an SADT of 40°C on a summer containership voyage through the Suez Canal. What operational parameters govern safe stowage?*
- **What a strong answer contains:** The candidate details the Arrhenius reaction kinetics: organic peroxides contain unstable $-O-O-$ bonds. If heat generated by exothermic decomposition exceeds heat dissipated to the environment, internal temperature rises exponentially. Under UN Manual of Tests and Criteria, for a substance with an $\text{SADT} = 40^\circ\text{C}$: (1) The Control Temperature is $T_{\text{control}} \le \text{SADT} - 10^\circ\text{C} = 30^\circ\text{C}$; (2) The Emergency Temperature is $T_{\text{emergency}} \le \text{SADT} - 5^\circ\text{C} = 35^\circ\text{C}$. The shipment requires mechanically refrigerated temperature-controlled tank containers with dual independent refrigeration circuits and backup power generation. Stowage must be On-Deck, shielded from direct solar radiation (tropical sun can heat container roofs to $> 65^\circ\text{C}$), stowed "Away from sources of heat" (engine room casings, heated bunker tanks), and positioned for immediate emergency jettison over the ship's side if $T_{\text{emergency}}$ is breached.
- **What a weak answer reveals:** The candidate says "just keep it below 40°C" without understanding control temperatures, emergency thresholds, or jettison protocols.

### Question 9: UN Packaging Specification Testing
*Explain the differences between UN Packing Group I, II, and III performance tests under 49 CFR Part 178, and how they govern packaging selection for hazardous liquids.*
- **What a strong answer contains:** The candidate explains that UN specification packaging must withstand severe mechanical and hydraulic stresses verified through standardized testing: (1) Drop Testing (§ 178.603): PG I packages must survive a free drop from $1.8\,\text{meters}$ ($5.9\,\text{ft}$) onto an unyielding surface at $-18^\circ\text{C}$; PG II requires a $1.2\,\text{meter}$ drop; PG III requires a $0.8\,\text{meter}$ drop; (2) Hydrostatic Pressure Testing (§ 178.605): Packaging designed for liquids must withstand an internal hydraulic pressure gauge reading of not less than $250\,\text{kPa}$ for PG I, or $100\,\text{kPa}$ for PG II/III, or $1.5$ times the vapor pressure of the lading at $55^\circ\text{C}$; (3) Stacking Test (§ 178.606): Packaging must support a compressive load equivalent to a $3.0\,\text{meter}$ high stack of identical packages for 24 hours (or 28 days at $40^\circ\text{C}$ for plastics) without buckling or leakage. The candidate notes that a PG I substance must be enclosed in packaging bearing a UN specification mark starting with "X" (certified for PG I, II, III), whereas "Y" packages are limited to PG II and III, and "Z" packages are restricted to PG III.
- **What a weak answer reveals:** The candidate states that higher packing groups just mean "thicker cardboard" without knowing the drop test heights, pressure thresholds, or UN specification codes.

### Question 10: Dynamic TOS IMDG Segregation Algorithms
*How does an automated terminal operating system (TOS) maintain real-time IMDG segregation compliance during container discharge operations without statically reserving fixed yard blocks?*
- **What a strong answer contains:** The candidate describes dynamic spatial constraint propagation algorithms: (1) The TOS models the container yard as a 3D coordinate lattice (Block, Bay, Row, Tier); (2) When a dangerous goods container is assigned to a ground slot, the TOS constraint engine computes a dynamic 3D exclusion bounding box around that slot based on the container's UN hazard class and subsidiary risks (e.g., a 2-slot lateral and longitudinal exclusion zone for Class 5.1 oxidizers); (3) As general import containers arrive at the quay crane, the allocation engine evaluates available candidate yard slots against active exclusion zones via boolean collision masks, allowing benign non-hazardous containers to fill the buffer slots between incompatible DG units; (4) The system incorporates dynamic lookahead heuristics to ensure that discharging an export DG container in an adjacent bay does not create a retroactive segregation breach.
- **What a weak answer reveals:** The candidate suggests that terminals must maintain rigid, permanently segregated physical fences for every single chemical class, ignoring dynamic space optimization.

---

### Question 11: Boiling Liquid Expanding Vapor Explosion (BLEVE) Mechanics
*What thermodynamic sequence of events triggers a BLEVE in a liquefied gas rail tank car during a derailment fire, and what structural design requirements under DOT-117 specifications are engineered to prevent it?*
- **What a strong answer contains:** A BLEVE occurs when a pressure tank car containing a liquefied compressed gas (e.g., propane, vinyl chloride, anhydrous ammonia) is exposed to an external fire. The heat flux boils the liquid, raising vapor pressure and venting gas through the pressure relief valve (PRV). However, the tank shell in the vapor space (above the liquid level) is not cooled by liquid heat transfer. The steel tank wall heats above $550^\circ\text{C}$ to $650^\circ\text{C}$, losing over 50% of its yield and tensile strength. Under internal pressure, the weakened steel thins, bulges, and tears cataclysmically. The sudden drop to atmospheric pressure causes the entire superheated liquid mass to nucleate homogeneously and flash boil into vapor within milliseconds. The expanding vapor shatters the tank into rocket-like projectiles and ignites into a massive atmospheric fireball ($q'' > 100\,\text{kW/m}^2$). The candidate details DOT-117 mitigation: (1) Mandatory 9/16-inch thick TC-128 Grade B normalized steel shell; (2) Full-height 1/2-inch thick head shields on both tank ends to resist coupler punctures; (3) A 1/2-inch ceramic fiber thermal insulation blanket enclosed within an 11-gauge steel outer jacket capable of surviving a 100-minute pool fire ($800^\circ\text{C}$) and 30-minute torch fire ($1,200^\circ\text{C}$); (4) High-capacity re-closing pressure relief devices.
- **What a weak answer reveals:** The candidate confuses a BLEVE with a simple chemical gas explosion or believes that pressure relief valves alone guarantee complete tank protection.

### Question 12: Dual-View Radiography and Spatial Superposition
*Why are high-throughput port container screening systems transitioning from single-view line scanners to dual-view or multi-view radiographic geometries?*
- **What a strong answer contains:** Single-view transmission radiography projects a 3D cargo volume onto a single 2D detector plane. This creates two fatal imaging failure modes: (1) Spatial Superposition: An illicit contraband item (e.g., low-$Z$ plastic explosive or high-$Z$ nuclear shielding) may be physically aligned with a benign dense object (e.g., a steel machine tool or crate of ceramic tiles). The single-view detector measures only the combined path-integrated attenuation ($\int \mu dx$), making it mathematically impossible to separate the threat's atomic number from the background matrix; (2) Orientation Blind Spots: A sheet explosive or knife oriented parallel to the X-ray beam presents a razor-thin cross-section that escapes detection. Dual-view systems deploy two independent linear accelerators and detector arrays oriented at orthogonal angles (typically horizontal and vertical projections). By capturing two independent projection angles, the Automated Threat Recognition (ATR) software computes cross-view stereo triangulation, uncorrelating superposed layers, detecting thin sheet threats regardless of packing orientation, and improving the Area Under the Curve (AUC) from $\approx 0.91$ to $> 0.98$.
- **What a weak answer reveals:** The candidate states that dual-view is used "just to get a backup picture in case one camera breaks" or "to scan twice as fast."

---

## 15. Key terms

1. **UN Hazard Class:** Official numerical classification (Classes 1–9) defining the primary chemical or physical hazard of a substance in transit.
2. **Packing Group (PG):** Regulatory indicator of hazard severity: PG I (high), PG II (medium), PG III (low), dictating packaging durability.
3. **Compatibility Group:** Letter code (A–S) governing whether different Class 1 explosives may be safely stowed in the same cargo transport unit.
4. **IMDG Code:** International Maritime Dangerous Goods Code; mandatory IMO regulation under SOLAS Chapter VII for ocean freight.
5. **Bay-Row-Tier:** Six-digit coordinate system identifying the exact 3D cellular location of a container aboard a container vessel.
6. **Segregation Level 1 ("Away from"):** Minimum cellular separation requiring at least 1 container pitch or row space between incompatible units.
7. **Segregation Level 2 ("Separated from"):** Separation requiring a complete steel bulkhead under-deck or at least 2 container spaces on-deck.
8. **Segregation Level 3 ("Complete compartment"):** Total longitudinal hold isolation bounded by structural fire- and watertight bulkheads.
9. **Segregation Level 4 ("Longitudinal compartment"):** Extreme isolation requiring an entire intervening cargo hold or minimum 24-meter on-deck gap.
10. **Hopkinson-Cranz Scaled Distance ($Z$):** Dimensionless blast scaling factor relating distance to the cube root of equivalent TNT mass.
11. **Notification to Captain (NOTOC):** Statutory flight-deck manifest detailing all dangerous goods locations, net masses, and drill codes.
12. **Cargo Aircraft Only (CAO):** Consignments legally prohibited from carriage on passenger aircraft due to net quantity or hazard severity.
13. **Dry Ice Sublimation Rate ($\dot{r}_{\text{sub}}$):** Fractional rate at which solid carbon dioxide converts to gas within an insulated package ($\text{hr}^{-1}$).
14. **Environmental Control System (ECS):** Aircraft air conditioning and bleed-air delivery system providing cargo hold fresh air ventilation.
15. **State of Charge (SoC):** Percentage of rated electrical energy remaining in a lithium-ion cell; legally capped at $30\%$ for air cargo.
16. **49 CFR Part 174:** US Federal Railroad Administration hazmat safety regulations governing train consist marshaling and switching.
17. **Buffer Car:** Non-hazardous rail car loaded with benign freight placed to protect train crew from collision and hazmat reaction.
18. **Key Train:** North American rail designation for high-hazard freight consists subject to strict $50\,\text{mph}$ corridor speed caps.
19. **ISPS Code:** International Ship and Port Facility Security Code; mandatory SOLAS convention governing maritime counter-terrorism.
20. **MARSEC Levels:** Three-tiered maritime security posture: Level 1 (Normal), Level 2 (Heightened Risk), Level 3 (Imminent Attack).
21. **Port Facility Security Officer (PFSO):** Statutory individual certified to execute and maintain the terminal security plan.
22. **Declaration of Security (DoS):** Bilateral agreement between vessel master and terminal security officer defining shared protection tasks.
23. **Non-Intrusive Inspection (NII):** Radiographic imaging using high-energy X-rays or gamma rays to inspect cargo without destuffing.
24. **Beer-Lambert Law:** Exponential equation governing photon beam attenuation through matter: $I = I_0 e^{-\mu x}$.
25. **Mass Attenuation Coefficient ($\mu/\rho$):** Fundamental material property describing photon interaction probability per unit mass density.
26. **Effective Atomic Number ($Z_{\text{eff}}$):** Equivalent atomic number reconstructed via dual-energy radiography to determine material composition.
27. **Automated Threat Recognition (ATR):** Machine learning algorithm parsing radiographic imagery to identify high-$Z$ anomalies and explosives.
28. **SADT:** Self-Accelerating Decomposition Temperature; lowest ambient temperature at which self-heating reaction triggers thermal runaway.

---

## 16. Sources

1. International Maritime Organization (IMO). (2022). *International Maritime Dangerous Goods (IMDG) Code* (2022 Edition, Incorporating Amendment 41-22). International Maritime Organization.
2. International Civil Aviation Organization (ICAO). (2021). *The Safe Transport of Dangerous Goods by Air* (Annex 18 to the Convention on International Civil Aviation, 4th ed.). ICAO.
3. International Civil Aviation Organization (ICAO). (2023). *Emergency Response Guidance for Aircraft Incidents Involving Dangerous Goods* (Doc 9481-AN/928, 2023–2024 Edition). ICAO.
4. International Air Transport Association (IATA). (2025). *Dangerous Goods Regulations (DGR)* (66th ed.). International Air Transport Association.
5. United States Department of Transportation (US DOT). (2024). *Title 49, Code of Federal Regulations, Parts 100–185: Pipeline and Hazardous Materials Safety Administration*. Office of the Federal Register.
6. Federal Railroad Administration (FRA). (2023). *49 CFR Part 174: Carriage by Rail — Position in Train of Placarded Cars*. US Government Publishing Office.
7. United States Coast Guard (USCG). (2024). *Title 33, Code of Federal Regulations, Part 105: Maritime Security — Facilities*. Office of the Federal Register.
8. International Maritime Organization (IMO). (2003). *International Ship and Port Facility Security (ISPS) Code and SOLAS Amendments 2002*. International Maritime Organization.
9. National Transportation Safety Board (NTSB). (2024). *Derailment of Norfolk Southern Railway Freight Train 32N with Hazardous Materials Release, East Palestine, Ohio, February 3, 2023* (Accident Report NTSB/RAR-24/08). NTSB.
10. National Transportation Safety Board (NTSB). (2013). *Crash Following In-Flight Fire, UPS Airlines Flight 6, Boeing 747-44AF, N571UP, Dubai, United Arab Emirates, September 3, 2010* (Foreign Aircraft Accident Report). United Arab Emirates General Civil Aviation Authority / NTSB.
11. Federal Bureau of Maritime Casualty Investigation (BSU). (2014). *Investigation Report 255/12: Explosion and Fire aboard the MSC FLAMINIA on 14 July 2012*. Bundesstelle für Seeunfalluntersuchung.
12. American National Standards Institute (ANSI) & IEEE. (2021). *American National Standard Performance Criteria for Non-Intrusive Inspection Systems for Cargo and Vehicle Security* (ANSI N42.44-2021). IEEE.
13. Ogorodnikov, S., & Petrunin, V. (2002). Processing of dual-energy X-ray images in cargo inspection system. *Physical Review Special Topics - Accelerators and Beams*, 5(10), 104701. https://doi.org/10.1103/PhysRevSTAB.5.104701
14. Kinney, G. F., & Graham, K. J. (1985). *Explosive Shocks in Air* (2nd ed.). Springer-Verlag. https://doi.org/10.1007/978-3-642-86682-1
15. Gross, D. (2008). *Fundamentals of Queueing Theory* (4th ed.). John Wiley & Sons. https://doi.org/10.1002/9781118625651

---

## 17. Links

- **Parent Module Overview:** [`_index.md`](_index.md)
- **Sibling Topic 01 (Quayside & Berth Operations):** [`01_berth-quay-crane.md`](01_berth-quay-crane.md)
- **Sibling Topic 02 (Terminal Yard & Gate Operations):** [`02_terminal-yard-gate.md`](02_terminal-yard-gate.md)
- **Sibling Topic 03 (Rail Network & Path Capacity):** [`03_rail-network-paths.md`](03_rail-network-paths.md)
- **Sibling Topic 04 (Rail Yards & Shunting):** [`04_rail-yards-shunting.md`](04_rail-yards-shunting.md)
- **Sibling Topic 05 (Air Cargo and ULD Operations):** [`05_air-cargo-uld.md`](05_air-cargo-uld.md)
- **Sibling Topic 06 (Intermodal & Hinterland Connectivity):** [`06_intermodal-hinterland.md`](06_intermodal-hinterland.md)
- **Sibling Topic 07 (Disruption Management & Service Recovery):** [`07_disruption-recovery.md`](07_disruption-recovery.md)
- **Sibling Topic 08 (Terminal Simulation & Digital Twins):** [`08_terminal-simulation.md`](08_terminal-simulation.md)
- **Sibling Topic 10 (Terminal Automation & Capital Investment Decisions):** [`10_automation-capital.md`](10_automation-capital.md)
- **Global Freight, Sourcing, and International Trade:** [`../../06_domain_electives/36_global_freight_sourcing_trade/_index.md`](../../06_domain_electives/36_global_freight_sourcing_trade/_index.md)
- **Transportation Routing & Last Mile Operations:** [`../../03_commerce/12_transportation_routing_last_mile/_index.md`](../../03_commerce/12_transportation_routing_last_mile/_index.md)
- **Operations Research & Mathematical Programming:** [`../../02_methods/20_operations_research_mathematical_programming/_index.md`](../../02_methods/20_operations_research_mathematical_programming/_index.md)
- **Root Survey File:** [`../../../course/41_maritime_rail_air_operations.md`](../../../course/41_maritime_rail_air_operations.md)
