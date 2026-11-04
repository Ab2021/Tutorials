# Bulk Logistics, Multi-Modal Transport, and Laytime Economics: Maritime Charterparties, Pipeline Batching, and Unit Train Operations

> Module 38 — Process Industries, Commodities, and Bulk Supply Chains
> Topic 07 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 15 (Stochastic Inventory & Risk-Pooling), Module 16 (Supply Chain Network Economics), Module 20 (Operations Planning & Master Scheduling), Module 38 Topic 01 (Continuous, Batch, and Discrete Production), Module 38 Topic 05 (Bulk Storage and Inventory Management)
> Version 2026.5

---

## 1. Why this topic exists

In packaged goods and discrete manufacturing logistics, transportation management revolves around standard freight units: twenty-foot equivalent units (TEUs), full truckloads (FTLs), less-than-truckload (LTL) pallets, and parcel shipments. Detention and demurrage fees in container shipping or trailer staging are typically administrative penalties—ranging from $\$50$ to $\$250$ per container per day—designed to incentivize equipment turnaround. The cargo value inside an individual container is modest relative to total supply chain working capital, and supply chain scheduling models treat transportation transit times as simple lead-time parameters with stationary or known variance.

Process industries and commodity supply chains operate under an entirely different physical, legal, and economic reality:

1. **The Transport Vehicle as a Capital-Intensive Factory on Water or Steel:** Bulk transportation assets—ocean-going vessels (Very Large Crude Carriers [VLCCs], Capesize bulkers, Handymax bulk carriers), 110-car unit trains, multi-product transmission pipelines, and 15-to-30-barge inland river flotillas—are enormous capital assets. A modern VLCC represents a $\$120\text{ million}$ to $\$150\text{ million}$ asset; a Capesize dry bulk carrier costs $\$65\text{ million}$ to $\$90\text{ million}$; a 100-car crude-by-rail unit train represents $\$15\text{ million}$ in rolling stock and $\$8\text{ million}$ in motive power. Daily operating and charter costs for these assets fluctuate wildly: a spot VLCC charter rate can swing from $\$20,000/\text{day}$ in a market glut to over $\$150,000/\text{day}$ during geopolitical bottlenecks or export surges. Consequently, maritime demurrage is not a trivial parking ticket; at $\$40,000$ to $\$100,000/\text{day}$, a 5-day port delay incurs $\$200,000$ to $\$500,000$ in cash penalties, directly destroying the gross processing margin (GPM) or trading margin of an entire voyage.
2. **Contractual Allocation of Time and Risk (The Charterparty):** In bulk maritime shipping, the vessel is chartered under specialized, legally binding contract forms such as ASBATANKVOY (tanker voyages) or GENCON (dry bulk voyages). The core economic engine of a voyage charter is **laytime**—the contractually agreed time window during which the charterer (cargo owner, trader, or refiner) is entitled to load or discharge the vessel without incurring additional financial charges. If loading or discharging exceeds the allowed laytime, the charterer pays liquidated damages known as **demurrage**. Conversely, if cargo operations finish early in dry bulk contracts, the shipowner pays the charterer a reward known as **despatch** (dispatch), historically fixed at half the demurrage rate ($R_{\text{despatch}} = 0.5 \times R_{\text{demurrage}}$). Laytime accounting is intensely litigated, governed by strict operational milestones: Notice of Readiness (NOR) tendering, berth availability clauses (WIPON, WIBON), turn-time cushions, and weather working day (WWD) exceptions.
3. **Pipeline Hydraulics, Batch Cycling, and Contamination Interfaces (Transmix):** Transmission pipelines transport millions of barrels of liquid hydrocarbons (crude grades, gasoline, diesel, jet fuel) through a single continuous steel conduit without physical barriers between parcels. Products are moved in discrete batches under high-pressure turbulent flow. Due to wall shear stress and fluid velocity profiles, the boundary between two adjacent products spreads longitudinally via **Taylor-Aris dispersion**, generating a mixed contamination interface known as **transmix**. Operating a multi-product pipeline requires strict batch sequence rules (e.g., never placing high-sulfur diesel adjacent to jet fuel), precise cut-point optimization (diverting interface volumes to slop tanks, transmix splitters, or downgrading to a lower-tier product), and cyclic schedule coordination to prevent refinery or terminal stockouts.
4. **Unit Train Dynamics and Closed-Loop Rail Logistics:** Bulk rail operations for coal, grain, potash, ethanol, and crude oil rely on dedicated **unit trains** (100 to 120 cars moving as a single unbroken trainset from origin load loop to destination unload loop). Unlike manifest freight—where individual railcars are decoupled, sorted in hump yards, and reassembled over weeks—unit train economics depend on cycle-time velocity. Bottlenecks at terminal rotary dumpers or manifold loading racks trigger severe carrier demurrage under Surface Transportation Board (STB) tariffs, while slow turnaround causes asset starvation at upstream mines and refineries.
5. **Multi-Modal Terminal Coupling:** A bulk commodity processing asset (a refinery, petrochemical complex, grain export elevator, or blast furnace) represents an immovable, continuous mass-balance node. Feedstocks arrive in massive, discrete pulses (e.g., 2,000,000 barrels on a VLCC or 110,000 metric tons on a Capesize bulker) and must be buffered into finite tankage or silo yards, while continuous finished products are evacuated via pipelines, unit trains, and barges. Operational coordination failures directly cascade: a delayed vessel causes tank top (forcing a refinery shutdown), while an off-spec pipeline interface cuts off an airport's jet fuel supply.

Mastering bulk logistics and laytime economics is essential for supply chain analytics professionals, trading operations managers, and process industry planners. Understanding the legal architecture of charterparties, the mathematical mechanics of laytime time-sheets, the fluid dynamics of transmix dispersion, and the operational optimization of rail and barge networks bridges the gap between commercial trading contracts and physical plant execution.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Deconstruct** standard maritime charterparty contracts (ASBATANKVOY 1977, GENCON 1994/2022, BPVOY4, SHELLVOY6) and map how operational, weather, and congestion risks are allocated between shipowner and charterer.
2. **Formulate and audit** a maritime laytime Statement of Facts (SOF), verifying the legal validity of Notice of Readiness (NOR), applying turn-time deductions, accounting for weather exclusions, and calculating net laytime consumed.
3. **Evaluate and apply** the common-law doctrine "once on demurrage, always on demurrage," identifying valid contractual exceptions, laytime suspensions, and time-bar deadlines for demurrage claims.
4. **Model fluid dispersion** at multi-product pipeline interfaces using Taylor-Aris dispersion theory, calculating transmix volumes and interface growth as functions of Reynolds number, pipe geometry, and pumping velocity.
5. **Optimize pipeline transmix cut points**, mathematically balancing downgrade losses, transmix splitter distillation processing costs, and quality giveaway across adjacent product batches.
6. **Construct** multi-product pipeline cyclic schedules, enforcing strict product adjacency constraints, transit batch cycles, and terminal receipt tank capacity bounds.
7. **Calculate** closed-loop unit train cycle times, fleet sizing requirements, and rolling-stock capital expenditures for high-volume bulk supply chains under Class I railroad demurrage tariffs.
8. **Synthesize** multi-modal terminal schedules using Mixed-Integer Linear Programming (MILP) to coordinate vessel jetty berthing, pipeline injection windows, rail unloading racks, and storage tank boundary constraints.

---

## 3. The core mental model

Bulk logistics is governed by the continuous-discrete impedance mismatch: chemical, metallurgical, and agricultural conversion processes run continuously (or in long continuous campaigns), whereas bulk transport modes move massive, discrete parcels under stochastic transit, weather, and mechanical delays. Buffer storage tanks, silos, and terminal jetties absorb this variance, but when buffers fail, the economic penalty converts immediately into demurrage claims, transmix downgrades, or facility throttling.

```text
=============================================================================================================
                       MULTI-MODAL BULK COMMODITY LOGISTICS & TIME-SHEET ARCHITECTURE
=============================================================================================================

[ UPSTREAM EXTRACTION / PRODUCTION ]
  Oil Fields, Mines, Ag Elevators, Chemical Plants
        │
        ▼ (Continuous / Cyclic Inflow)
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ ORIGIN BULK TERMINAL                                                                                      │
│ - Storage Tanks / Silo Batteries / Stockpiles (Working Capacity vs. Heel / Dead Bottom)                  │
│ - Loading Infrastructure: High-Rate Marine Berth Manifolds, Unit Train Loop Tracks, Pipeline Injectors     │
└─────────────────────────────────┬───────────────────────────────────┬─────────────────────────────────────┘
                                  │                                   │
      ┌───────────────────────────┼───────────────────────────────────┼───────────────────────────┐
      │ (Mode A: Maritime)        │ (Mode B: Pipeline)                │ (Mode C: Unit Train)      │
      ▼                           ▼                                   ▼                           ▼
┌───────────────────────────┐ ┌───────────────────────────┐ ┌───────────────────────────┐ ┌──────────────────┐
│ OCEAN VOYAGE CHARTER      │ │ TRANSMISSION PIPELINE     │ │ CLOSED-LOOP UNIT TRAIN    │ │ INLAND BARGE     │
│ (ASBATANKVOY / GENCON)    │ │ (Turbulent Batch Flow)    │ │ (110-Car Dedicated Sets)  │ │ (Push Flotilla)  │
│                           │ │                           │ │                           │ │                  │
│ • Laydays / Cancelling    │ │ • Batch Sequence Cycle    │ │ • Origin Load Loop        │ │ • River Draft    │
│   (Laycan) Window         │ │ • Taylor-Aris Dispersion  │ │ • Mainline Class I Run    │ │ • Lock Queuing   │
│ • Notice of Readiness     │ │ • Transmix Interface Zone │ │ • Destination Rotary      │ │ • Tow Sizing     │
│ • Turn-Time Cushion       │ │ • Cut-Point Optimization  │ │   Dumper / Bottom Dump    │ │ • Fleet Balancing│
│ • Weather Working Days    │ │ • Downgrade vs. Splitter  │ │ • Railcar Demurrage       │ │                  │
│ • Laytime Balance Sheet   │ │                           │ │   (STB Tariffs EP 754)    │ │                  │
│ • Demurrage vs. Despatch  │ │                           │ │                           │ │                  │
└─────────────┬─────────────┘ └─────────────┬─────────────┘ └─────────────┬─────────────┘ └────────┬─────────┘
              │                             │                             │                        │
              └─────────────────────────────┼─────────────────────────────┴────────────────────────┘
                                            ▼
┌───────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│ DESTINATION IMPORT / REFINERY TERMINAL                                                                    │
│ - Marine Jetty Unloading Berth (Draft, LOA, Manifold Limits)                                             │
│ - Pipeline Delivery Manifold & Optical Densitometer Cutting Station                                       │
│ - Unit Train Unloading Loop Track & Storage Yard                                                          │
│ - Working Tank Farm / Silos / Blending Manifold                                                           │
└───────────────────────────────────────────┬───────────────────────────────────────────────────────────────┘
                                            │
                                            ▼ (Continuous Feed)
[ DOWNSTREAM CONTINUOUS PROCESS PLANT: Crude Distillation Unit, Blast Furnace, Flour Mill, Power Plant ]
=============================================================================================================
```

### The Three Operational Regimes of Bulk Transportation

Bulk logistics spans three distinct operational paradigms, each governed by unique physics, contracts, and financial penalty structures:

1. **The Maritime Voyage Regime:** The vessel navigates international waters under charterparty contracts. Risk transfers at the manifold flange or the cargo loading spout. The primary commercial variables are time ($t$) and rate of cargo transfer ($Q/t$). The vessel owner provides seaworthiness and speed warranties; the cargo owner guarantees a berth and loading/discharging throughput within the agreed laytime. Time overruns trigger contractually stipulated liquidated damages (demurrage).
2. **The Pipeline Batch Regime:** Products move through stationary infrastructure governed by the Navier-Stokes equations and convective-diffusive mass transfer. Parcels cannot stop without line shut-in. Fluid turbulence ($Re > 4,000$) must be preserved to prevent catastrophic laminar interface mixing. The primary operational variable is spatial interface dispersion ($\Delta L$ or $\Delta V_{\text{transmix}}$), and the economic trade-off balances interface cutting thresholds against downstream product specification giveaway.
3. **The Railroad Unit Train Regime:** Discrete rolling stock operates over shared, regulated track networks. High-throughput terminal infrastructure (rotary car dumpers capable of emptying an open-top gondola in 90 seconds without uncoupling; pressurized tank-car unloading racks) decouples linehaul transit from terminal dwell. Tariffs regulated by government authorities (e.g., the U.S. Surface Transportation Board) govern terminal demurrage, while network train-bunching threatens cyclic velocity.

---

## 4. Definitions and vocabulary

The following formal terms govern bulk commodity transport contracts, fluid operations, and terminal scheduling.

| Term | Abbreviation | Rigorous Technical Definition | Common Misconception / Operational Pitfall | Unit / Standard Code |
|---|---|---|---|---|
| **Charterparty** | CP | A maritime contract by which a shipowner agrees to lease an entire vessel, or cargo space on a vessel, to a charterer for a voyage or time period. | Confusing a charterparty (vessel contract) with a Bill of Lading (receipt of goods and document of title). | Standard Form (e.g., ASBATANKVOY) |
| **Voyage Charter** | — | Charterparty where the vessel is hired for a single voyage between designated ports; freight is paid per metric ton or lump sum. | Assuming the charterer pays fuel (bunkers) and port fees; in voyage charters, owner pays operating costs. | Currency / MT or Flat Rate |
| **Time Charter** | TC | Charterparty where the charterer hires the vessel for a defined calendar duration; charterer pays hire rate, fuel, and port costs. | Believing laytime and demurrage apply; time charters use hire payments and off-hire deduction clauses instead. | $\text{USD}/\text{day}$ |
| **Contract of Affreightment** | COA | Long-term contract to transport a specified total volume of commodity over time using multiple vessels across scheduled windows. | Treating it as a single vessel charter; COA specifies quantity and frequency, not individual hull identities. | MT / Year |
| **Laydays and Cancelling** | Laycan | The contractually agreed calendar window (e.g., June 10–15) within which the vessel must present itself at the load port. | Believing vessel cannot arrive early; early arrival is permitted, but laytime clock does not run until laydays begin. | Calendar Dates (YYYY-MM-DD) |
| **Notice of Readiness** | NOR | Formal legal notice tendered by the shipmaster or agent to the terminal certifying the vessel has arrived and is ready to load/discharge. | Tendering NOR before obtaining customs clearance or entering commercial port limits, rendering the notice invalid. | ISO 8601 Timestamp |
| **Laytime** | — | The agreed amount of time permitted to the charterer for loading and discharging the cargo without paying additional fees. | Assuming laytime runs continuously without interruption; exclusions depend strictly on contract wording. | Hours or Days (SHINC / SSHEX) |
| **Demurrage** | — | Liquidated damages paid by the charterer to the shipowner for detaining the vessel beyond the agreed laytime. | Mislabeling demurrage as a "penalty"; in maritime law, it is contractually agreed compensation for asset detention. | $\text{USD}/\text{day}$ pro rata |
| **Despatch (Dispatch)** | — | Financial rebate paid by the shipowner to the charterer if cargo operations complete before laytime expires (common in dry bulk). | Expecting despatch in tanker charters; oil charters (ASBATANKVOY) almost universally set despatch to zero. | $\text{USD}/\text{day}$ (usually $0.5 \times \text{Demurrage}$) |
| **Statement of Facts** | SOF | Chronological log maintained by the port agent detailing all vessel movements, weather events, delays, and cargo pumping times. | Assuming the SOF automatically determines laytime; SOF provides facts, while charterparty clauses dictate counting. | Chronological Log / Time Sheet |
| **Weather Working Day** | WWD | A day of 24 consecutive hours on which work is not prevented by adverse weather conditions (rain, high swell, gale-force winds). | Counting a rainy day as a full working day when loading moisture-sensitive bulk cargo (e.g., grain, cement, alumina). | Days / Fractions of Days |
| **SHINC / SSHEX** | — | Laytime terms: Sundays and Holidays Included (SHINC) vs. Sundays and Holidays Excluded (SSHEX). | Assuming SSHEX excludes hours worked on weekends unless explicitly specified ("SSHEX unless used" vs. "SSHEX even if used"). | Contractual Term Code |
| **WIPON / WIBON** | — | "Whether In Port Or Not" / "Whether In Berth Or Not"; permits NOR tendering even if vessel cannot berth due to congestion. | Believing WIBON excuses vessel unreadiness; vessel must be physically ready to work even if anchored in roads. | Contractual Term Code |
| **WIFPON / WICCON** | — | "Whether In Free Pratique Or Not" / "Whether In Customs Cleared Or Not"; allows valid NOR tender prior to health/customs clearance. | Overlooking health quarantine delays; if pratique is denied due to vessel fault, NOR is voided retroactively. | Contractual Term Code |
| **Turn-Time** | TT | Contractual grace period (typically 6 hours in tanker charters, 12–24 hours in dry bulk) between NOR tender and laytime commencement. | Starting the laytime clock the second the anchor drops; turn-time must run unless cargo operations commence earlier. | Hours |
| **Reversible Laytime** | — | Contract term allowing unused laytime at the load port to be added to the laytime allowance at the discharge port. | Treating load and discharge laytime as independent non-reversible buckets when the contract specifies reversible terms. | Contractual Clause |
| **Deadfreight** | — | Freight payable on cargo space agreed to be chartered but not utilized by the charterer due to cargo short-delivery. | Expecting to pay only for cargo physically loaded; if contract booked 100,000 MT and 85,000 MT loaded, deadfreight is due on 15,000 MT. | $\text{USD} = (Q_{\text{contract}} - Q_{\text{loaded}}) \times \text{Rate}$ |
| **Taylor Dispersion** | — | Longitudinal spreading of a solute or fluid interface in a pipe caused by the interaction of velocity shear and radial molecular diffusion. | Assuming fluid moves like a solid piston (plug flow); parabolic velocity profiles smear the interface over kilometers. | Dimensionless / Diffusivity ($\text{m}^2/\text{s}$) |
| **Transmix** | — | The commingled interface volume formed between two different refined petroleum products moving sequentially through a pipeline. | Assuming transmix is discarded as waste; transmix is accumulated and re-fractionated in dedicated distillation splitters. | Barrels or Cubic Meters ($\text{m}^3$) |
| **Heart Cut** | — | Diverting the contaminated pipeline interface entirely into a downgrade tank to ensure the receiving tank stays strictly on-spec. | Cutting too aggressively, creating excessive volume of degraded product that requires costly refinery re-running. | Volumetric Threshold / Density ($\text{kg}/\text{m}^3$) |
| **Downgrade Cut** | — | Diverting interface volume containing higher-value product (e.g., Premium Gasoline) into lower-value product (Regular Gasoline). | Downgrading in reverse (putting heavy diesel into gasoline), violating flash point and octane specifications. | Economic Differential ($\text{USD}/\text{bbl}$) |
| **Unit Train** | — | A dedicated freight train composed of uniform railcars carrying a single bulk commodity directly from one origin to one destination. | Confusing unit trains with manifest (mixed) trains; unit trains bypass all intermediate switching yards. | Trainset (100–120 Cars) |
| **Rotary Car Dumper** | RCD | Terminal facility that clamps an entire railcar and rotates it $160^\circ$ to discharge bulk material while remaining coupled via rotary couplers. | Using standard couplers; rotary dumping without uncoupling requires specialized rotary-shaft coupler mechanisms. | Tons / Hour ($\text{t}/\text{h}$) |
| **Loop Track** | — | A continuous balloon-shaped rail loop at a bulk terminal allowing a unit train to arrive, load/unload, and depart without reversing or decoupling. | Designing ladder tracks for unit trains; reversing locomotives adds 6–18 hours of terminal switching delay. | Geometry / Curvature Radius ($\text{m}$) |
| **STB Demurrage Tariff** | — | Surface Transportation Board regulated demurrage rules governing daily railcar detention charges on private vs. railroad trackage. | Assuming railroad demurrage is unregulated; Class I carriers must publish transparent, non-discriminatory demurrage tariffs. | $\text{USD}/\text{car}/\text{day}$ |
| **Barge Flotilla (Tow)** | — | A group of non-powered inland barges lashed together and pushed by a single towboat on inland river systems. | Treating river barges like ocean vessels; river barges lack propulsion, steering, and require lock chamber splitting. | Standard Barges (15 to 36 Barges) |
| **Worldscale** | WS | The New Worldwide Tanker Nominal Freight Scale; a standardized freight rating system providing flat dollar rates per metric ton for tanker voyages. | Assuming WS is a market freight rate; Worldscale Flat is a standard theoretical cost basis, with market fixtures quoted as a percentage (e.g., WS 85). | Points of Scale (Percentage of Flat) |
| **Virtual Arrival** | VA | Contractual framework where a vessel slows down its transit speed to arrive exactly when an empty berth becomes available, sharing bunker savings. | Assuming virtual arrival is unilateral; both shipowner and charterer must agree, with demurrage and fuel savings audited and shared. | Knots / Metric Tons Bunkers Saved |
| **Time-Bar Clause** | — | Charterparty provision extinguishing any demurrage claim not fully submitted with supporting documents within a strict deadline (e.g., 60–90 days). | Treating time-bar deadlines as procedural guidance; missing a 90-day time-bar by one hour legally invalidates the entire claim. | Calendar Days (60 or 90 Days) |
| **Fleeting Area** | — | Designated riverbank anchorage where inland barges are assembled, held, and sorted into multi-barge flotillas by dedicated fleeting towboats. | Assuming linehaul towboats switch individual barges at plant docks; linehaul boats drop tows at fleeting areas for local harbor boats to service. | Waterway Acreage / Barge Staging Capacity |
| **Dead-Bottom (Heel)** | — | The unpumpable liquid layer below a tank's lowest suction nozzle, required to maintain suction head and prevent vortex cavitation. | Counting heel volume as available commercial working inventory; attempting to pump the heel damages pumps and draws tank sludge. | Metric Tons or Barrels ($\text{bbl}$) |

---

## 5. The mechanics, step by step

### Step 1: Maritime Charterparty Contract Architecture and Clause Selection

Every bulk sea transport begins with selecting the appropriate contract of affreightment or charterparty. The operational risk profile depends on the standard form used:

```text
[Bulk Shipment Requirement]
            │
            ├─► Liquid Bulk (Crude Oil, Refined Products, Chemicals)
            │         └─► ASBATANKVOY 1977 / BPVOY4 / SHELLVOY6
            │
            └─► Dry Bulk (Coal, Iron Ore, Grain, Bauxite, Fertilizer)
                      └─► GENCON 1994 / GENCON 2022 / NYPE 2015
```

The charterparty contract establishes five non-negotiable operational parameters:
1. **The Cargo Description & Quantity:** Specified with an operational margin, typically $\pm 5\%$ or $\pm 10\%$ at shipowner's option (MOLOO - More or Less Owner's Option) or charterer's option (MOLCO). This allows the vessel master to optimize cargo intake based on bunker fuel weight, freshwater allowances, and seasonal draft restrictions.
2. **Laycan (Laydays / Cancelling Window):** E.g., Laydays commencing 06:00 July 1, Cancelling 24:00 July 7. If the vessel arrives prior to 06:00 July 1, she may tender NOR, but the charterer is under no obligation to count laytime until the laydays open. If the vessel arrives after 24:00 July 7, the charterer holds the unilateral contractual option to cancel the charterparty without liability.
3. **Laytime Allowance:** Formulated either as total time (e.g., "72 running hours SHINC total load and discharge" in ASBATANKVOY) or as a loading/discharging transfer rate (e.g., "10,000 metric tons per weather working day of 24 consecutive hours, SSHEX" in GENCON).
4. **Demurrage and Despatch Rates:** Expressed as a fixed dollar figure per day and pro rata (e.g., "Demurrage USD 45,000 per day or pro rata for any part of a day"). If despatch is granted, it is customarily 50% of the demurrage rate ($R_{\text{despatch}} = 0.5 \times R_{\text{demurrage}}$), though in tanker trades despatch is almost universally excluded ($R_{\text{despatch}} = 0$).
5. **Notice of Readiness and Berth Clauses:** Defining whether the charter is a **Port Charter** (vessel tenders NOR upon reaching the commercial port limits, anchorages, or pilot station) or a **Berth Charter** (vessel cannot tender valid NOR until firmly safely moored alongside the assigned loading/discharging jetty).

---

### Step 2: Notice of Readiness (NOR) Tendering and Legal Validity

The formal trigger of laytime is the Notice of Readiness (NOR). For an NOR to be legally valid under maritime law, three concurrent physical and administrative tests must be satisfied:

$$\text{Valid NOR} \iff (\text{Arrived Ship}) \land (\text{Ready in All Respects}) \land (\text{Within Prescribed Hours / Limits})$$

```text
                                  ┌───────────────────────────┐
                                  │ Vessel Arrives at Port    │
                                  └─────────────┬─────────────┘
                                                │
                       Is vessel within commercial port limits / roads?
                                                │
                         ┌──────────────────────┴──────────────────────┐
                        YES                                            NO
                         │                                             │
             Does CP contain WIPON?                         NOR Invalid;
                         │                                  Must reach port limits
          ┌──────────────┴──────────────┐
         YES                            NO
          │                             │
    Arrived Ship               Must reach designated berth
          │
          ├────────────────────────────────────────────────────────────┐
          ▼                                                            ▼
Is cargo hold/tank clean, dry, & inerted?             Has Free Pratique & Customs been granted?
          │                                                            │
    ┌─────┴─────┐                                                ┌─────┴─────┐
   YES          NO                                              YES          NO
    │            │                                               │            │
 Ready to      NOR Void / Re-clean                         Pratique OK   Does CP have WIFPON/WICCON?
 Cargo Holds   (Time stops until reinspected)                            │            │
    │                                                                   YES           NO
    └──────────────────────────────────────────┬─────────────────────────┘             │
                                               │                                   NOR Suspended
                                               ▼                                   until cleared
                                    Tender Formal Written NOR
                                 (Master / Agent to Terminal)
```

- **Arrived Ship Test:** Under a Port Charter, the vessel is an "arrived ship" when she drops anchor at the customary waiting area or pilot station within the legal and commercial jurisdiction of the port, placing herself at the immediate and effective disposition of the charterer. Under a Berth Charter, she must be physically alongside the assigned berth unless protected by a **WIBON** ("Whether In Berth Or Not") or **WIPON** ("Whether In Port Or Not") rider clause.
- **Physical Readiness:** The vessel's cargo tanks (in liquid bulk) or holds (in dry bulk) must be clean, dry, odor-free, inerted (for crude/products), and pass strict independent surveyor inspection. If grain holds contain insect infestation, rust scale, or moisture, the surveyor rejects the holds. The NOR previously tendered is rendered **void ab initio**, and laytime does not commence until reinspection passes and a fresh NOR is tendered.
- **Administrative Readiness (Free Pratique and Customs):** The vessel must have health clearance (Free Pratique) and customs entry. Standard clauses like **WIFPON** ("Whether In Free Pratique Or Not") and **WICCON** ("Whether In Customs Cleared Or Not") protect the shipowner from administrative port delays, provided pratique or customs is not subsequently denied due to vessel fault or infectious disease aboard.

---

### Step 3: Laytime Commencing and Turn-Time Mechanics

Once a valid NOR is tendered, laytime does not immediately begin. The contract provides an operational buffer known as **turn-time**:

- In tanker charterparties (e.g., ASBATANKVOY Clause 6), laytime commences:
  1. Upon the expiration of **6 hours** after receipt of Notice of Readiness by charterer or agent, OR
  2. Immediately upon the vessel being all fast alongside berth,
  *whichever occurs first*.

$$\tau_{\text{commence}} = \min \left( \tau_{\text{NOR}} + 6\text{ hours}, \tau_{\text{all fast alongside}} \right)$$

- In dry bulk charterparties (e.g., GENCON 1994 Clause 6), laytime begins at 13:00 if NOR is tendered before noon, or at 07:00 next working day if NOR is tendered during afternoon business hours, unless cargo operations commence earlier.

---

### Step 4: Statement of Facts (SOF) Compilation and Interruptions Accounting

The Statement of Facts (SOF) is the primary evidentiary document recorded by the independent port agent, countersigned by the vessel master and terminal representative. The laytime analyst converts the SOF into an audited Laytime Time-Sheet by classifying every minute of port stay into one of three legal categories:

1. **Countable Laytime:** Time that runs against the charterer's allowance.
2. **Excluded Time:** Time contractually excluded from laytime (e.g., contractual turn-time, official holidays under SSHEX, non-working weather periods under WWD).
3. **Suspended / Deductible Time (Owner's Account):** Delays caused directly by the vessel, its crew, or its equipment (e.g., engine failure, pump breakdowns, crew illness, vessel shifting between berths for owner's convenience, deballasting delays exceeding cargo intake rate).

```text
Laytime Event Classification Matrix:
┌──────────────────────────────────────┬────────────────────────┬───────────────────────────────────┐
│ Event Description                    │ Standard Tanker (ASBA) │ Standard Dry Bulk (GENCON 94)     │
├──────────────────────────────────────┼────────────────────────┼───────────────────────────────────┤
│ Turn-time cushion (e.g., 6h)         │ Excluded (unless used) │ Excluded (per clause)             │
│ Moving from anchorage to berth       │ Excluded (shifting)    │ Excluded (inward transit)         │
│ Shore pipeline / arm connection      │ Counts                 │ Counts                            │
│ Normal cargo pumping / loading       │ Counts                 │ Counts                            │
│ Rain / adverse sea swell (WWD)       │ Counts (SHINC)         │ Excluded (preventing work)        │
│ Vessel deballasting (pumps lagging)  │ Deducted (Owner acct)  │ Deducted (Owner acct)             │
│ Vessel cargo pump breakdown          │ Deducted (Owner acct)  │ Deducted (Owner acct)             │
│ Shore tank overflow / shutdown       │ Counts                 │ Counts                            │
│ Tugboat strike (general)             │ 50% rate (Clause 19)   │ Governed by strike clause terms   │
│ Awaiting daylight / pilot tidal rule │ Counts (port charter)  │ Excluded if local custom prevails │
└──────────────────────────────────────┴────────────────────────┴───────────────────────────────────┘
```

#### The Legal Doctrine: "Once on Demurrage, Always on Demurrage"
A foundational maxim of English maritime law (*Generale rule: Once on demurrage, always on demurrage*):
Once the charterer has exhausted the contractually agreed laytime allowance, the vessel is on demurrage. From that precise moment onward, **laytime exceptions cease to apply**. Demurrage runs continuously, 24 hours a day, including Sundays, holidays, and periods of adverse weather, *unless* the charterparty contains clear, express, and unambiguous language extending specific exceptions into the demurrage period (e.g., "half-rate demurrage during strikes, storms, or fires" in ASBATANKVOY Clause 19). Delays attributable exclusively to the fault or breakdown of the vessel never count as demurrage.

---

### Step 5: Laytime Reconciliation and Demurrage / Despatch Settlement

The mathematical balance of laytime is formulated as follows. Let $T_{\text{allowed}}$ be the contractually permitted laytime in hours. Let the port stay be discretized into $K$ chronological time intervals $\Delta t_k = t_k - t_{k-1}$, where each interval has an operational status indicator $w_k \in [0.0, 1.0]$ representing the countable fraction:

$$T_{\text{used}} = \sum_{k=1}^{K} w_k \cdot \Delta t_k$$

The net laytime balance $\Delta T$ is:

$$\Delta T = T_{\text{allowed}} - T_{\text{used}}$$

The commercial settlement is governed by the sign of $\Delta T$:

$$\text{Financial Settlement} = 
\begin{cases} 
\text{Demurrage Payable by Charterer} = |\Delta T| \times \left( \dfrac{R_{\text{demurrage}}}{24} \right) & \text{if } \Delta T < 0 \\
\text{Despatch Payable by Owner} = \Delta T \times \left( \dfrac{R_{\text{despatch}}}{24} \right) & \text{if } \Delta T > 0 \text{ and } R_{\text{despatch}} > 0 \\
0 & \text{if } \Delta T \ge 0 \text{ and } R_{\text{despatch}} = 0
\end{cases}$$

#### Laytime Calculation Frameworks: Non-Reversible vs. Reversible vs. Average
- **Non-Reversible Laytime (Default):** Separate calculations are performed for loading and discharging ports. Unused laytime at the load port cannot offset demurrage incurred at the discharge port. If the load port has a 10-hour surplus (despatch) and the discharge port has a 20-hour deficit (demurrage), the charterer receives 10 hours despatch and pays 20 hours demurrage.
- **Reversible Laytime:** The charterer is permitted to aggregate the total allowed laytime across all load and discharge ports into a single consolidated pool:
  $$T_{\text{allowed, total}} = T_{\text{allowed, load}} + T_{\text{allowed, discharge}}$$
  $$T_{\text{used, total}} = T_{\text{used, load}} + T_{\text{used, discharge}}$$
  Demurrage is paid only if $T_{\text{used, total}} > T_{\text{allowed, total}}$.
- **Average Laytime:** Demurrage and despatch are calculated separately for load and discharge ports, and the net monetary balances are set off against each other.

#### The Worldscale System and Freight Rate Indexing
In international wet bulk logistics (crude oil and clean petroleum products), freight rates and demurrage are universally quoted using the **New Worldwide Tanker Nominal Freight Scale (Worldscale / WS)**. The Worldscale Association (operating jointly from London and New York) publishes annual flat rates—expressed in U.S. dollars per metric ton—for tens of thousands of individual direct and multi-port voyage legs worldwide.

A Worldscale Flat Rate ($\text{WS } 100$) represents the theoretical total operating cost per metric ton for a standard nominal vessel:
- **Nominal Standard Vessel:** 75,000 metric tons deadweight, operating at a design service speed of 14.5 knots on an average bunker consumption of 55 metric tons of heavy fuel oil per steaming day plus 5 metric tons of auxiliary fuel in port, with an assumed fixed daily operating hire allowance and standardized port tariff schedule.

Market fixtures are negotiated and concluded as a percentage ("points of scale") of the published flat rate:

$$\text{Total Freight Cost (USD)} = Q_{\text{cargo}} \times \text{Flat Rate}_{\text{route}} \times \left( \frac{\text{WS}_{\text{fixture}}}{100} \right)$$

For example, if the annual published Flat Rate for a voyage from Ras Tanura (Saudi Arabia) to Rotterdam (Netherlands) is $\$22.50/\text{MT}$, and a fixture for an Aframax lifting 80,000 MT is fixed at **WS 85**:

$$\text{Freight Rate} = \$22.50 \times \left( \frac{85}{100} \right) = \$19.125/\text{MT}$$
$$\text{Total Freight Payable} = 80,000\text{ MT} \times \$19.125/\text{MT} = \$1,530,000.00$$

In charter fixtures where demurrage is not stated as a fixed dollar figure, the Worldscale daily demurrage rate is indexed proportionally:

$$R_{\text{demurrage}} = R_{\text{demurrage}}^{\text{WS 100}} \times \left( \frac{\text{WS}_{\text{fixture}}}{100} \right)$$

#### The 90-Day Time-Bar Clause and Documentary Integrity
A severe commercial trap in bulk commodity shipping is the contractual **Time-Bar Clause** (e.g., ASBATANKVOY Clause 19 rider additions, BPVOY4 Clause 20, and SHELLVOY6 Clause 23). Modern oil majors enforce strict documentary deadlines:

> *"Charterers shall be discharged and released from all liability in respect of any claim for demurrage which the Owner may have under this Charter unless a claim in writing has been presented to the Charterer, together with all supporting documentation, within ninety (90) consecutive days of the completion of discharge of cargo."*

Under English maritime arbitration (*The Sabrewing* [2007] EWHC 2019 (Comm) and *The Eagle Valencia* [2010] EWCA Civ 713), courts enforce these clauses strictly against shipowners:
1. **Documentary Completeness:** The submission must include every relevant operational document: the signed Statement of Facts (SOF), the Notice of Readiness (NOR), terminal pumping logs, hourly pressure logs, and all contemporaneous Letters of Protest (LOP) issued by the master (e.g., protests for slow shore pumping, excessive backpressure, or berthing delays).
2. **Absolute Bar:** If a shipowner submits a $\$500,000$ demurrage claim on day 89, but omits a copy of the terminal pumping log, or submits the log on day 91, the entire claim is legally extinguished. The charterer owes $\$0.00$. Operational trading desks must maintain rigorous automated claim tracking pipelines to ensure zero documentary leakage.

---

### Step 6: Multi-Product Pipeline Batch Cycling and Hydraulics

In liquid pipeline transmission, multiple refined hydrocarbon products—such as Regular Unleaded Gasoline (RUL), Premium Unleaded Gasoline (PUL), Ultra-Low Sulfur Diesel (ULSD), and Aviation Turbine Fuel (Jet A-1)—are pumped sequentially through a common steel pipeline.

```text
=============================================================================================================
                      MULTI-PRODUCT PIPELINE BATCH TRANSMISSION & CONTAMINATION INTERFACE
=============================================================================================================

Flow Direction ─────────►

  [ ULSD Batch ]             [ Transmix Interface ]            [ Jet A-1 Batch ]        [ RUL Gasoline Batch ]
  Density: 845 kg/m³         Density Gradient                  Density: 805 kg/m³       Density: 735 kg/m³
  Sulfur: < 10 ppm           (Commingled Fluid)                Sulfur: < 15 ppm         Octane: 87 AKI
┌───────────────────────┬───────────────────────────────┬───────────────────────────────┬───────────────────┐
│                       │ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ │                               │                   │
│                       │  \ \ Transmix Growth: ΔV \ \  │                               │                   │
│                       │ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ │                               │                   │
└───────────────────────┴───────────────────────────────┴───────────────────────────────┴───────────────────┘
                                        │
                                        ▼ (At Pipeline Delivery Manifold)
                         ┌─────────────────────────────┐
                         │ Fast-Response Densitometer  │
                         │ & Optical Spectrometer      │
                         └──────────────┬──────────────┘
                                        │
                 ┌──────────────────────┼──────────────────────┐
                 ▼                      ▼                      ▼
        ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
        │ Heart Cut:      │    │ Downgrade Cut:  │    │ Slop Tank /     │
        │ Clean Product   │    │ Premium to Reg  │    │ Transmix Column │
        │ to Target Tank  │    │ Gasoline Tank   │    │ Fractionation   │
        └─────────────────┘    └─────────────────┘    └─────────────────┘
```

#### Hydraulic Flow Regimes and Reynolds Number
Pipelines must operate under fully developed turbulent flow to minimize interface mixing. The dimensionless Reynolds number ($Re$) is:

$$Re = \frac{\rho v D}{\mu} = \frac{v D}{\nu} = \frac{4 Q}{\pi D \nu}$$

where $\rho$ is fluid density ($\text{kg}/\text{m}^3$), $v$ is mean cross-sectional flow velocity ($\text{m}/\text{s}$), $D$ is internal pipe diameter ($\text{m}$), $\mu$ is dynamic viscosity ($\text{Pa}\cdot\text{s}$), $\nu = \mu/\rho$ is kinematic viscosity ($\text{m}^2/\text{s}$), and $Q$ is volumetric flow rate ($\text{m}^3/\text{s}$).

- If $Re < 2,100$ (Laminar Flow): Parabolic velocity profile ($v(r) = 2 v_{\text{avg}} (1 - (r/R)^2)$). Fluid at the center travels twice as fast as the average velocity. The interface smears longitudinally over tens of kilometers, causing catastrophic contamination.
- If $Re > 4,000$ (Turbulent Flow): Flat, blunt velocity profile across the pipe core. Eddy diffusion blunts parabolic elongation, confining interface dispersion to a narrow mixing zone.

#### Batch Sequence and Adjacency Compatibility Matrix
To prevent severe product downgrades, pipelines enforce strict batch adjacency rules:
1. **Never place high-sulfur or dyed distillates adjacent to Jet A-1.** Jet fuel has a zero-tolerance contamination specification for particulate matter, lubricity additives, and color.
2. **Buffer sensitive fuels with compatible grades:** A typical cyclic sequence moves from lighter to heavier products and back:
   $$\text{RUL} \longrightarrow \text{PUL} \longrightarrow \text{Jet A-1} \longrightarrow \text{ULSD} \longrightarrow \text{Heating Oil} \longrightarrow \text{ULSD} \longrightarrow \text{Jet A-1} \longrightarrow \text{PUL} \longrightarrow \text{RUL}$$
3. **Cycle Frequency:** A complete cycle of batches (the "cycle") repeats every 5 to 10 days, allowing downstream terminal storage tanks to be replenished without stockouts.

---

### Step 7: Taylor-Aris Dispersion and Transmix Interface Modeling

Sir Geoffrey Taylor (1953) and Rutherford Aris (1956) formulated the classical theory of dispersion in pipe flow. In turbulent pipe flow, the combined effect of radial turbulent mixing and axial shear generates an effective axial dispersion coefficient $K_{\text{disp}}$ ($\text{m}^2/\text{s}$):

$$K_{\text{disp}} = 10.1 \cdot R \cdot u_* = 10.1 \cdot \left(\frac{D}{2}\right) \cdot \sqrt{\frac{\tau_w}{\rho}}$$

where $R = D/2$ is the pipe radius, $u_* = \sqrt{\tau_w / \rho}$ is the friction velocity, and $\tau_w$ is wall shear stress ($\tau_w = \frac{1}{8} f \rho v^2$, with $f$ being the Darcy-Weisbach friction factor).

The 1D convective-dispersion governing equation for solute concentration $C(x,t)$ across the interface is:

$$\frac{\partial C}{\partial t} + v \frac{\partial C}{\partial x} = K_{\text{disp}} \frac{\partial^2 C}{\partial x^2}$$

For two semi-infinite product batches initialized with a sharp step boundary at $t=0$, the analytical solution is the complementary error function:

$$C(x,t) = \frac{C_0}{2} \operatorname{erfc}\left( \frac{x - v t}{2 \sqrt{K_{\text{disp}} t}} \right)$$

#### Empirical Pipeline Transmix Volume Equation
In industrial pipeline engineering (API standards and pipeline operations), transmix interface volume $V_{\text{transmix}}$ between batch $A$ and batch $B$ over pipeline length $L$ is modeled empirically:

$$V_{\text{transmix}} = \kappa \cdot \sqrt{L} \cdot D^n \cdot \left( \frac{\nu_{\text{avg}}}{v} \right)^m$$

In practical units (where $V_{\text{transmix}}$ is in barrels, $D$ is in inches, and $L$ is in miles), the standard empirical Austin-Palfrey pipeline equation is:

$$V_{\text{transmix}} = 1.175 \cdot D^{2.5} \cdot \sqrt{\frac{L}{Re^{0.1}}}$$

#### Transmix Cut-Point Decision Logic
When the interface arrives at the terminal manifold, high-frequency density transmitters (coriolis meters or nuclear densitometers) track fluid density $\rho(t)$ in real time. The terminal automation system makes dynamic cut decisions:

```text
Normalized Interface Composition:
  0% ────────────────────────── Cut Point α ──────── Cut Point β ────────────────────────── 100%
  Pure Product A               Cut into A            Divert to Transmix           Pure Product B
  (e.g., Premium Gasoline)     (Quality Giveaway)    Tank / Splitter              (e.g., Diesel)
```

1. **Cut Point $\alpha$:** From pure Product A to threshold $\rho_\alpha$. The commingled fluid is diverted into Product A's storage tank, provided the resulting blended tank batch satisfies all ASTM specifications (e.g., research octane number [RON], Reid vapor pressure [RVP]).
2. **Transmix Diversion Window ($\rho_\alpha$ to $\rho_\beta$):** The core contaminated interface is cut into a designated **transmix / slop tank**. This volume cannot be sold directly; it must be trucked to a transmix distillation splitter facility or blended into refinery residual streams at a severe price discount ($\$15$ to $\$35/\text{bbl}$ penalty).
3. **Cut Point $\beta$:** From threshold $\rho_\beta$ to pure Product B. The fluid is directed into Product B's storage tank, provided flash point, distillation end-point, and cetane index remain strictly on-spec.

---

### Step 8: Rail Unit Train Operations and Cycle Time Dynamics

In high-volume bulk commodities—such as crude oil, metallurgical coal, grain, and frac sand—rail transportation moves in dedicated **unit trains** of 100 to 120 cars operating in a continuous closed loop.

```text
=============================================================================================================
                           CLOSED-LOOP RAIL UNIT TRAIN CYCLE ARCHITECTURE
=============================================================================================================

 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                       ORIGIN LOAD TERMINAL                                             │
 │ - Unit train enters continuous loop track without decoupling locomotives.                              │
 │ - Automatic volumetric loading arms fill 110 cars in 4 to 8 hours.                                     │
 └───────────────────────────────────┬────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ (Loaded Transit: t_transit,loaded)
                       ┌───────────────────────────┐
                       │ Class I Railroad Network  │
                       │ Linehaul Transit (Loaded) │
                       │ Speed: 25–40 km/h avg     │
                       └─────────────┬─────────────┘
                                     │
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────────────────────┐
 │                                   DESTINATION UNLOAD TERMINAL                                          │
 │ - Rotary Car Dumper clamps cars and empties them in 90 seconds/car, OR                                 │
 │ - Bottom-dump hoppers discharge bulk solids through rail trestle pits, OR                              │
 │ - Multi-car unloading manifold empties tank cars under vacuum/pressurization.                          │
 │ - Unloading duration: 4 to 8 hours.                                                                    │
 └───────────────────────────────────┬────────────────────────────────────────────────────────────────────┘
                                     │
                                     ▼ (Empty Transit: t_transit,empty)
                       ┌───────────────────────────┐
                       │ Class I Railroad Network  │
                       │ Linehaul Transit (Empty)  │
                       │ Speed: 30–50 km/h avg     │
                       └─────────────┬─────────────┘
                                     │
                                     └──────────────── (Cycle Restarts)
```

#### Mathematical Cycle Time Decomposition
The total turnaround cycle time $T_{\text{cycle}}$ (in days) for a unit train set is:

$$T_{\text{cycle}} = t_{\text{load}} + t_{\text{transit, loaded}} + t_{\text{inspect, origin}} + t_{\text{unload}} + t_{\text{transit, empty}} + t_{\text{inspect, dest}} + t_{\text{slack}}$$

where:
- $t_{\text{load}}$: Terminal dwell time for loading, air-brake testing, and documentation ($0.25$ to $0.50\text{ days}$).
- $t_{\text{transit, loaded}} = \frac{d_{\text{route}}}{24 \cdot v_{\text{commercial, loaded}}}$: Transit time loaded over distance $d_{\text{route}}$ ($\text{km}$). Commercial speed $v_{\text{commercial}}$ accounts for rail siding meets, crew changes, and dispatcher priority.
- $t_{\text{unload}}$: Terminal dwell time for heating (for heavy crudes/waxes) and unloading ($0.25$ to $0.50\text{ days}$).
- $t_{\text{transit, empty}} = \frac{d_{\text{route}}}{24 \cdot v_{\text{commercial, empty}}}$: Transit time of empty return trainset.
- $t_{\text{slack}}$: Planned buffer time for bad-order car maintenance, locomotive servicing, and network congestion ($0.5$ to $1.5\text{ days}$).

#### Fleet Sizing Equation
To sustain a continuous delivery throughput of $D_{\text{annual}}$ (metric tons or barrels per year), the required number of active trainsets $N_{\text{trains}}$ is:

$$N_{\text{trains}} = \left\lceil \frac{D_{\text{annual}} \times T_{\text{cycle}}}{365 \times Q_{\text{train}}} \right\rceil$$

where $Q_{\text{train}} = n_{\text{cars}} \times q_{\text{car}}$ is the payload per trainset ($n_{\text{cars}}$ cars, $q_{\text{car}}$ payload per car). The total railcar fleet $N_{\text{cars, total}}$ including spare maintenance reserve factor $\alpha_{\text{maint}}$ ($5\%$ to $10\%$) is:

$$N_{\text{cars, total}} = N_{\text{trains}} \times n_{\text{cars}} \times (1 + \alpha_{\text{maint}})$$

#### Surface Transportation Board (STB) Rail Demurrage Economics
Under U.S. Surface Transportation Board (STB) regulations (e.g., Docket EP 754), rail demurrage tariffs penalize excessive railcar dwell:
- **Free Time Allowance:** Typically 24 hours for loading and 48 hours for unloading.
- **Demurrage Charges:** Range from $\$75$ to $\$150$ per car per day. For a 110-car unit train delayed at an unloading siding for 4 days past free time, the railroad assesses:
  $$\text{Rail Demurrage} = 110\text{ cars} \times 4\text{ days} \times \$120/\text{car}/\text{day} = \$52,800$$
- **Private vs. Railroad Cars:** Private cars (owned or leased by the shipper) stored on private tracks incur no demurrage. However, if private cars are held on railroad-owned trackage awaiting terminal entry clearance, the carrier assesses **storage fees** equivalent to demurrage.

---

### Step 9: Inland Waterway Barge Towing and River Hydrology Logistics

In continental commodity supply chains—such as the U.S. Mississippi-Ohio river basin, the European Rhine-Danube corridors, and the South American Paraná-Paraguay waterway—inland water transport provides the lowest ton-mile energy consumption and lowest transport cost for dry and liquid bulk commodities (grain, coal, fertilizer, crude oil, and basic chemicals).

```text
=============================================================================================================
                          INLAND RIVER WATERWAY FLOTILLA & LOCKAGE SYSTEM
=============================================================================================================

  Direction of Tow ─────────► (Linehaul Push Towboat: 6,000 to 10,500 Horsepower)

┌───────────┬───────────┬───────────┐  Standard Jumbo Hopper Barge:
│  Barge 1  │  Barge 2  │  Barge 3  │  - Dimensions: 195 ft x 35 ft (59.4 m x 10.7 m)
├───────────┼───────────┼───────────┤  - Capacity @ 9 ft draft: 1,500 short tons (1,360 MT) / 52,500 bu
│  Barge 4  │  Barge 5  │  Barge 6  │
├───────────┼───────────┼───────────┤  Lock Chamber Dimension Constraints:
│  Barge 7  │  Barge 8  │  Barge 9  │  - Standard Upper River Lock: 600 ft x 110 ft (183 m x 33.5 m)
├───────────┼───────────┼───────────┤  - 15-Barge Tow Dimensions: 1,150 ft x 105 ft (Requires Double Lockage)
│ Barge 10  │ Barge 11  │ Barge 12  │  - Modern Lock: 1,200 ft x 110 ft (Enables Single-Pass Lockage)
├───────────┼───────────┼───────────┤
│ Barge 13  │ Barge 14  │ Barge 15  │
└───────────┴───────────┴───────────┘
                 ▲
                 │ Rigid Wire Lashings
          [ Towboat Push Knees ]
                 │
        [ Linehaul Towboat ]
```

#### Tow Sizing, Lock Chambers, and Double-Locking Mechanics
Unlike maritime ships, inland barges have no propulsion or steering; they are lashed together into a rigid rectangular flotilla ("tow") and propelled from the stern by a twin- or triple-screw diesel towboat equipped with steering and flanking rudders.
1. **Open River Towing (Lower Mississippi):** Below St. Louis (Lock 27), the Mississippi River flows unrestricted by locks to the Gulf of Mexico. Tows operate up to 36 to 40 barges ($6\text{ wide} \times 6\text{ to }7\text{ long}$), covering over 5 acres and moving $55,000\text{ to }60,000\text{ metric tons}$ of cargo in a single passage—equivalent to 550 railcars or 2,400 semi-trucks.
2. **Canalized River Navigation (Upper Mississippi, Ohio, Illinois):** Rivers are divided into navigational pools by series of locks and dams maintained by government authorities (e.g., U.S. Army Corps of Engineers [USACE]).
   - Where lock chambers are $600\text{ ft} \times 110\text{ ft}$, a standard 15-barge tow ($1,150\text{ ft}$ overall length) cannot fit in a single pass.
   - **Double-Locking Procedure:** The tow is uncoupled mid-stream. The first cut of 8 or 9 barges is pulled into the lock chamber by hydraulic wall mules or winches, locked through, and tied off to the downstream guide wall. The chamber is refilled, the towboat and remaining barges enter, lock through, and the flotilla is reassembled. Double-locking expands lock transit time from 45 minutes to over 120 minutes per tow, creating cascading upstream queues during seasonal grain export harvest peaks.

#### River Hydrology: Seasonal Low-Water Draft Penalties
Inland navigation is vulnerable to seasonal drought and river gauge fluctuations. Standard channel design depth is $9.0\text{ feet}$ ($2.74\text{ meters}$). When low rainfall reduces river stages:
- Each **1-inch reduction** in barge operating draft reduces the cargo carrying capacity of a standard jumbo hopper barge by **17.0 short tons** (15.4 MT).
- If emergency low-water draft restrictions enforce a maximum draft of $7.5\text{ feet}$ (an 18-inch draft cut):
  $$\text{Payload Reduction per Barge} = 18\text{ inches} \times 17.0\text{ tons/inch} = 306.0\text{ short tons (20.4% loss)}$$
  $$\text{Effective Barge Intake} = 1,500 - 306 = 1,194\text{ short tons}$$
- To evacuate the same volume of grain or coal, the shipper requires $25.6\%$ more barges and must pay carrier **draft surcharge tariffs**, while shallow channels force towboats to reduce flotilla width from 6 barges wide to 4 barges wide, throttling corridor throughput.

---

### Step 10: Multi-Modal Terminal Optimization and Coordinated Scheduling

At a major bulk commodity processing terminal, marine jetties, pipeline injection stations, rail unloading loops, and storage tanks converge into a highly constrained operational system. The terminal scheduling problem is modeled as a continuous-time or discrete-time Mixed-Integer Linear Program (MILP):

#### Mathematical Formulation: Objective Function
$$\min \quad \sum_{v \in \mathcal{V}} \left( C_v^{\text{demurrage}} \cdot \delta_v + C_v^{\text{bunker}} \cdot \tau_v^{\text{wait}} \right) + \sum_{p \in \mathcal{P}} C_p^{\text{transmix}} \cdot V_p^{\text{transmix}} + \sum_{r \in \mathcal{R}} C_r^{\text{rail\_dem}} \cdot \delta_r + \sum_{b \in \mathcal{B}} C_b^{\text{fleet}} \cdot \tau_b^{\text{dwell}}$$

#### Decision Variables and Constraints
1. **Berth Non-Overlap Constraints:** No two vessels can occupy the same jetty simultaneously:
   $$\tau_u^{\text{berth}} + \tau_u^{\text{service}} \le \tau_v^{\text{berth}} + M(1 - y_{uv}), \quad \forall u, v \in \mathcal{V}, u \ne v, \text{ berth } b$$
   where $y_{uv} \in \{0, 1\}$ indicates whether vessel $u$ precedes vessel $v$ at berth $b$, and $M$ is a large positive scalar.
2. **Draft, Length, and Beam Compatibility:** Vessel draft, Length Overall (LOA), and beam cannot exceed physical jetty limits:
   $$\text{Draft}_v \le \text{MaxDraft}_b, \quad \text{LOA}_v \le \text{MaxLOA}_b, \quad \text{Beam}_v \le \text{MaxBeam}_b$$
3. **Storage Tank Inventory Conservation:**
   $$S_{t, i} = S_{t-1, i} + \sum_{v \in \mathcal{V}} q_{v, i, t}^{\text{unload}} + \sum_{r \in \mathcal{R}} q_{r, i, t}^{\text{rail\_in}} + \sum_{b \in \mathcal{B}} q_{b, i, t}^{\text{barge\_in}} - \sum_{p \in \mathcal{P}} q_{p, i, t}^{\text{pipe\_out}} - q_{\text{plant}, i, t}^{\text{feed}}$$
4. **Tank Boundary and Working Heel Limits:**
   $$S_{\min, i}^{\text{heel}} \le S_{t, i} \le S_{\max, i}^{\text{tank\_top}}, \quad \forall i \in \mathcal{I}, \forall t$$
5. **Vessel Demurrage Penalty Linearization:**
   Let $T_v^{\text{allowed}}$ be the contract laytime and $\tau_v^{\text{start}}$ be the laytime commencement timestamp:
   $$\delta_v \ge 0, \quad \delta_v \ge (\tau_v^{\text{complete}} - \tau_v^{\text{start}}) - T_v^{\text{allowed}}$$
6. **Rail Unit Train Unloading Loop Rack Dwell:**
   $$\delta_r \ge 0, \quad \delta_r \ge (\tau_r^{\text{depart}} - \tau_r^{\text{arrive}}) - T_r^{\text{free\_time}}$$

## 6. Worked example

### Problem Statement

An integrated energy refiner charters an Aframax crude tanker (*MT Nordic Horizon*) to lift 80,000 metric tons of light crude oil from an offshore loading terminal in the North Sea and discharge at an Antwerp refinery terminal. The crude is then pumped through a 350-kilometer, 20-inch pipeline to an inland storage hub, while an associated rail unit train evacuates finished heavy diesel to an inland industrial center.

We will solve three interrelated bulk logistics calculations end to end with exact arithmetic:
- **Part A:** Maritime Laytime & Demurrage Statement of Facts Reconciliation.
- **Part B:** Pipeline Taylor Dispersion Interface & Transmix Cutting Economics.
- **Part C:** Closed-Loop Rail Unit Train Fleet Sizing and Demurrage Risk.

---

### Part A: Maritime Voyage Charter Laytime & Demurrage Settlement

#### Contractual Parameters (Charterparty: ASBATANKVOY 1977 Form)
- **Vessel:** *MT Nordic Horizon*, Aframax Tanker, 80,000 MT cargo.
- **Laytime Allowance:** 72.00 running hours SHINC total (load and discharge combined).
- **Demurrage Rate:** $\$36,000.00$ per day pro rata ($\$1,500.00$ per hour, or $\$25.00$ per minute).
- **Despatch Rate:** $\$0.00$ (tanker charter convention).
- **Turn-Time Clause:** 6 hours turn-time after valid NOR receipt unless operations begin earlier.
- **Laydays / Cancelling (Laycan):** June 1, 06:00 to June 5, 18:00.

#### Chronological Statement of Facts (SOF)

**Load Port: North Sea Offshore Terminal**
1. **June 2, 08:00:** Vessel arrives at loading anchorage; tenders written NOR.
2. **June 2, 08:00 to 14:00:** Contractual 6-hour turn-time runs.
3. **June 2, 14:00:** Turn-time expires; laytime clock begins running.
4. **June 2, 14:00 to June 2, 18:00:** Vessel awaiting berth (berth occupied by previous vessel). (Countable: 4.00 h).
5. **June 2, 18:00 to 20:00:** Anchor weighed; inward transit from anchorage to Single Point Mooring (SPM). Shifting time contractually excluded under ASBATANKVOY Clause 7. (Excluded: 2.00 h).
6. **June 2, 20:00:** All fast at SPM; hoses connected.
7. **June 2, 20:00 to June 3, 22:00:** Loading operations normal at 3,200 MT/h. (Countable: 26.00 h).
8. **June 3, 22:00 to June 4, 02:00:** Loading suspended due to vessel deballasting pump malfunction (vessel tanks unable to receive crude while deballasting). Delay for owner's account. (Deducted: 4.00 h).
9. **June 4, 02:00 to June 4, 06:00:** Loading completed; hoses disconnected; cargo documents presented. (Countable: 4.00 h).
10. **June 4, 06:00:** Vessel sails for Antwerp.

*Load Port Laytime Accounting:*
$$\text{Countable Hours at Load Port} = 4.00\text{ (waiting)} + 26.00\text{ (loading)} + 4.00\text{ (completion)} = 34.00\text{ hours}$$
$$\text{Cumulative Laytime Used} = 34.00\text{ hours}$$
$$\text{Remaining Laytime for Discharge} = 72.00 - 34.00 = 38.00\text{ hours}$$

**Discharge Port: Antwerp Refinery Terminal**
1. **June 7, 04:00:** Vessel arrives at Wandelaar pilot station / Flushing roads; tenders written NOR. Free pratique granted.
2. **June 7, 04:00 to 10:00:** Contractual 6-hour turn-time runs. (Excluded: 6.00 h).
3. **June 7, 10:00:** Turn-time expires; laytime clock resumes.
4. **June 7, 10:00 to June 8, 14:00:** Vessel awaiting berth due to severe tidal congestion at refinery jetty. (Countable: 28.00 h).
   - *Cumulative laytime reached:* $34.00 + 28.00 = 62.00\text{ hours}$.
   - *Remaining laytime:* $72.00 - 62.00 = 10.00\text{ hours}$.
5. **June 8, 14:00 to 17:00:** Inward transit and locking through Zandvliet Lock to refinery berth. Shifting time excluded under CP terms. (Excluded: 3.00 h).
6. **June 8, 17:00:** All fast alongside refinery berth; marine loading arms connected.
7. **June 8, 17:00 to June 9, 03:00:** Cargo discharge begins. (Countable: 10.00 h).
   - *At June 9, 03:00 precisely:* Cumulative laytime reaches $62.00 + 10.00 = 72.00\text{ hours}$.
   - **LAYTIME EXPIRES AT JUNE 9, 03:00.**
   - From June 9, 03:00 onward, the vessel is **ON DEMURRAGE**.
8. **June 9, 03:00 to June 9, 11:00:** Severe storm with winds exceeding 45 knots strikes the port. Terminal orders loading arms disconnected for safety reasons.
   - *Legal question:* Does storm time count?
   - *Answer:* Under ASBATANKVOY Clause 19, delays caused by storm or fire are settled at **half-rate demurrage** ($50\%$ rate), provided the storm is documented by port authority logs. (Duration: 8.00 h at half rate).
9. **June 9, 11:00 to June 10, 07:00:** Arms reconnected; discharge resumed and completed; hoses disconnected. (Countable at full demurrage rate: 20.00 h).

#### Demurrage Calculation Breakdown
- Total laytime allowed: $72.00\text{ hours}$.
- Total laytime consumed up to expiry: $72.00\text{ hours}$.
- Time on demurrage:
  1. Storm interruption period (ASBA Cl. 19 Half Rate):
     $$\text{Duration} = 8.00\text{ hours} = \frac{8}{24}\text{ days} = 0.3333\text{ days}$$
     $$\text{Cost}_{\text{storm}} = 0.3333\text{ days} \times \left( 0.5 \times \$36,000.00/\text{day} \right) = 0.3333 \times \$18,000.00 = \$6,000.00$$
  2. Full demurrage discharge period:
     $$\text{Duration} = 20.00\text{ hours} = \frac{20}{24}\text{ days} = 0.8333\text{ days}$$
     $$\text{Cost}_{\text{full}} = 0.8333\text{ days} \times \$36,000.00/\text{day} = \$30,000.00$$

$$\text{Total Demurrage Claim Payable by Charterer} = \$6,000.00 + \$30,000.00 = \mathbf{\$36,000.00}$$

---

### Part B: Pipeline Dispersion and Transmix Interface Cutting

#### Pipeline Parameters
- Internal Diameter: $D = 20\text{ inches} = 0.508\text{ meters}$.
- Cross-sectional Area: $A = \frac{\pi}{4} D^2 = \frac{\pi}{4} (0.508)^2 = 0.20268\text{ m}^2$.
- Pipeline Length: $L = 350\text{ km} = 350,000\text{ meters}$.
- Flow Rate: $Q = 1,200\text{ m}^3/\text{h} = 0.3333\text{ m}^3/\text{s}$.
- Fluid Velocity: $v = \frac{Q}{A} = \frac{0.3333}{0.20268} = 1.6447\text{ m}/\text{s}$.
- Batch Sequence: Regular Unleaded Gasoline (Batch A, $\rho_A = 740\text{ kg}/\text{m}^3$, $\nu_A = 0.6 \times 10^{-6}\text{ m}^2/\text{s}$) adjacent to Ultra-Low Sulfur Diesel (Batch B, $\rho_B = 845\text{ kg}/\text{m}^3$, $\nu_B = 3.2 \times 10^{-6}\text{ m}^2/\text{s}$).
- Average Kinematic Viscosity: $\nu_{\text{avg}} = \frac{0.6 + 3.2}{2} \times 10^{-6} = 1.9 \times 10^{-6}\text{ m}^2/\text{s}$.

#### Fluid Mechanics and Reynolds Number
$$Re = \frac{v D}{\nu_{\text{avg}}} = \frac{1.6447 \times 0.508}{1.9 \times 10^{-6}} = \frac{0.8355}{1.9 \times 10^{-6}} \approx 439,740$$
Since $Re \gg 4,000$, flow is deeply turbulent, which suppresses laminar parabolic smearing.

#### Interface Dispersion Calculation
Using the standard Austin-Palfrey pipeline correlation for transmix volume in commercial units:
- Pipeline diameter: $D = 20\text{ inches}$.
- Pipeline length: $L = 350\text{ km} = 217.48\text{ miles}$.
- Reynolds Number: $Re = 439,740 \implies Re^{0.1} = (439,740)^{0.1} \approx 3.648$.

$$V_{\text{transmix}} = 1.175 \cdot D^{2.5} \cdot \sqrt{\frac{L_{\text{miles}}}{Re^{0.1}}}$$
$$D^{2.5} = 20^{2.5} = \sqrt{20^5} = \sqrt{3,200,000} \approx 1,788.85$$
$$\sqrt{\frac{217.48}{3.648}} = \sqrt{59.616} \approx 7.721$$
$$V_{\text{transmix}} = 1.175 \times 1,788.85 \times 7.721 \approx 16,229\text{ barrels} \approx 2,580\text{ m}^3$$

#### Cut-Point Economic Optimization
The $16,229\text{ bbl}$ transmix volume arrives at the inland terminal. The terminal operator can make one of two operational decisions:
- **Strategy 1 (Wide Heart Cut to Slop Splitter):** Divert the entire $16,229\text{ bbl}$ into the transmix tank. The transmix must be re-processed at a commercial splitter fee of $\$18.00/\text{bbl}$.
  $$\text{Cost}_{\text{Strategy 1}} = 16,229\text{ bbl} \times \$18.00/\text{bbl} = \$292,122.00$$
- **Strategy 2 (Optimized Multi-Cut):** Based on refinery lab assay testing, $40\%$ of the interface (gasoline-rich tail) can be safely cut into the Regular Gasoline storage tank without failing octane or RVP specs, incurring a minor downgrade loss of $\$4.50/\text{bbl}$. Another $35\%$ (diesel-rich tail) can be cut into the Heavy Industrial Gasoil pool at a downgrade loss of $\$6.00/\text{bbl}$. Only the central $25\%$ core ($4,057\text{ bbl}$) is diverted to the transmix splitter at $\$18.00/\text{bbl}$.
  $$\text{Cost}_{\text{Gasoline Cut}} = (0.40 \times 16,229) \times \$4.50 = 6,491.6\text{ bbl} \times \$4.50 = \$29,212.20$$
  $$\text{Cost}_{\text{Gasoil Cut}} = (0.35 \times 16,229) \times \$6.00 = 5,680.15\text{ bbl} \times \$6.00 = \$34,080.90$$
  $$\text{Cost}_{\text{Splitter Core}} = (0.25 \times 16,229) \times \$18.00 = 4,057.25\text{ bbl} \times \$18.00 = \$73,030.50$$
  $$\text{Total Cost}_{\text{Strategy 2}} = \$29,212.20 + \$34,080.90 + \$73,030.50 = \mathbf{\$136,323.60}$$

$$\text{Net Economic Value Created by Interface Analytics} = \$292,122.00 - \$136,323.60 = \mathbf{\$155,798.40\text{ per batch}}$$

---

### Part C: Rail Unit Train Fleet Sizing and Demurrage Risk

#### Transport Demand and Parameters
- Product: Heavy Diesel evacuation from inland terminal to regional mining hub.
- Annual Demand: $D_{\text{annual}} = 3,000,000\text{ metric tons/year}$.
- Tank Car Payload: $q_{\text{car}} = 68\text{ metric tons}$ ($28,500\text{ gallons}$ / $\approx 510\text{ bbl}$).
- Train Configuration: $n_{\text{cars}} = 105\text{ cars per unit train}$.
- Train Payload: $Q_{\text{train}} = 105 \times 68 = 7,140\text{ metric tons}$.
- One-Way Rail Distance: $d_{\text{route}} = 1,400\text{ km}$.
- Average Loaded Commercial Speed: $v_{\text{loaded}} = 32\text{ km/h}$.
- Average Empty Commercial Speed: $v_{\text{empty}} = 40\text{ km/h}$.
- Origin Loading & Brake Inspection Dwell: $t_{\text{load}} = 0.50\text{ days}$ ($12\text{ hours}$).
- Destination Unloading Dwell: $t_{\text{unload}} = 0.50\text{ days}$ ($12\text{ hours}$).
- Network Slack & Inspection Buffer: $t_{\text{slack}} = 0.75\text{ days}$ ($18\text{ hours}$).
- Railcar Maintenance Reserve: $\alpha_{\text{maint}} = 8\%$.

#### Cycle Time Computation
$$t_{\text{transit, loaded}} = \frac{1,400\text{ km}}{32\text{ km/h} \times 24\text{ h/day}} = \frac{1,400}{768} \approx 1.823\text{ days}$$
$$t_{\text{transit, empty}} = \frac{1,400\text{ km}}{40\text{ km/h} \times 24\text{ h/day}} = \frac{1,400}{960} \approx 1.458\text{ days}$$
$$T_{\text{cycle}} = 0.50 + 1.823 + 0.50 + 1.458 + 0.75 = \mathbf{5.031\text{ days}}$$

#### Required Trainsets and Fleet Sizing
$$\text{Annual Departures Required} = \frac{3,000,000\text{ MT}}{7,140\text{ MT/train}} \approx 420.17\text{ trains/year}$$
$$\text{Average Departures per Day} = \frac{420.17}{365} \approx 1.151\text{ trains/day}$$
$$N_{\text{trains}} = \left\lceil 1.151\text{ trains/day} \times 5.031\text{ days} \right\rceil = \lceil 5.791 \rceil = \mathbf{6\text{ dedicated unit trainsets}}$$

$$\text{Active Railcars} = 6\text{ trainsets} \times 105\text{ cars} = 630\text{ tank cars}$$
$$\text{Total Fleet (with } 8\%\text{ spare)} = \lceil 630 \times 1.08 \rceil = \lceil 680.4 \rceil = \mathbf{681\text{ DOT-117 tank cars}}$$

#### Financial Exposure under Class I Rail Demurrage
If winter congestion adds a 48-hour unplanned dwell past the 24-hour tariff free time across all 420 annual train dispatches:
$$\text{Annual Demurrage Penalty} = 420\text{ trains} \times 105\text{ cars} \times 2\text{ excess days} \times \$110/\text{car/day} = \mathbf{\$9,702,000/\text{year}}$$
This highlights why rail terminal loop track design, high-speed loading manifolds, and predictive track scheduling are critical operational safeguards.

#### Multi-Modal Worked Example Results Synthesis
The following table summarizes the quantitative operational and economic outputs derived across the three interconnected bulk transport modes:

| Transport Subsystem | Key Operational Metric | Physical / Temporal Value | Financial / Economic Exposure | Critical Risk Control Parameter |
|---|---|---|---|---|
| **Part A: Maritime Tanker (Aframax)** | Net Laytime Consumed vs. Allowed | $96.00\text{ h consumed vs. } 72.00\text{ h allowed}$ | $\mathbf{\$36,000.00\text{ Demurrage}}$ ($24.00\text{ h net over}$) | ASBATANKVOY Clause 19 (50% storm rate); Turn-time tracking |
| **Part B: Multi-Product Pipeline (20-in)** | Interface Volume & Mixing Length | $16,229\text{ bbl } (2,580\text{ m}^3) \text{ over } 13.65\text{ km}$ | $\mathbf{\$155,798.40\text{ Cost Savings}}$ via 3-cut optimization | Reynolds number $> 400,000$; Fast-loop densitometer cut gates |
| **Part C: Rail Unit Train (DOT-117)** | Closed-Loop Cycle Time & Fleet Size | $T_{\text{cycle}} = 5.031\text{ days}; \; N = 6\text{ trainsets}$ | $\mathbf{\$9,702,000/\text{year}}$ potential STB rail demurrage | 24-hour tariff free time; Continuous terminal loop track geometry |

---

## 7. Code, query, or calculation

The following production-ready Python program models maritime Statement of Facts (SOF) accounting, enforcing turn-time, weather exclusions, owner delay deductions, and calculating net laytime balance and demurrage payable.

```python
from datetime import datetime, timedelta

class LaytimeEngine:
    def __init__(self, allowed_hours, dem_rate_day, desp_rate_day=0.0):
        self.allowed = timedelta(hours=allowed_hours)
        self.dem_rate = dem_rate_day
        self.desp_rate = desp_rate_day

    def process_sof(self, events):
        """Processes events: (start_str, end_str, count_factor, description)
        count_factor: 1.0 (countable), 0.5 (half-rate storm/strike), 0.0 (excluded)"""
        fmt = "%Y-%m-%d %H:%M"
        used = timedelta()
        audit_log = []
        for s_str, e_str, factor, desc in events:
            s, e = datetime.strptime(s_str, fmt), datetime.strptime(e_str, fmt)
            duration = e - s
            countable_duration = duration * factor
            used += countable_duration
            audit_log.append({
                "start": s_str, "end": e_str, "factor": factor,
                "dur_hrs": round(duration.total_seconds() / 3600, 2),
                "cnt_hrs": round(countable_duration.total_seconds() / 3600, 2),
                "desc": desc
            })

        balance = self.allowed - used
        bal_hrs = balance.total_seconds() / 3600
        if bal_hrs < 0:
            status = "DEMURRAGE"
            cost = (abs(bal_hrs) / 24.0) * self.dem_rate
        else:
            status = "DESPATCH" if self.desp_rate > 0 else "NO_DEMURRAGE"
            cost = (bal_hrs / 24.0) * self.desp_rate

        return {
            "allowed_hrs": round(self.allowed.total_seconds() / 3600, 2),
            "used_hrs": round(used.total_seconds() / 3600, 2),
            "balance_hrs": round(bal_hrs, 2),
            "status": status,
            "financial_settlement_usd": round(cost, 2),
            "audit_log": audit_log
        }

if __name__ == "__main__":
    calc = LaytimeEngine(allowed_hours=72.0, dem_rate_day=36000.0, desp_rate_day=0.0)
    sof = [
        ("2026-06-02 08:00", "2026-06-02 14:00", 0.0, "NOR 6-hour turn-time cushion"),
        ("2026-06-02 14:00", "2026-06-02 18:00", 1.0, "Awaiting loading berth"),
        ("2026-06-02 18:00", "2026-06-02 20:00", 0.0, "Shifting anchorage to berth (excluded)"),
        ("2026-06-02 20:00", "2026-06-03 22:00", 1.0, "Loading operations normal"),
        ("2026-06-03 22:00", "2026-06-04 02:00", 0.0, "Vessel deballasting pump fault (owner acct)"),
        ("2026-06-04 02:00", "2026-06-04 06:00", 1.0, "Loading to completion and hose disconnect"),
        ("2026-06-07 04:00", "2026-06-07 10:00", 0.0, "Discharge port 6-hour turn-time"),
        ("2026-06-07 10:00", "2026-06-08 14:00", 1.0, "Awaiting discharge berth (congestion)"),
        ("2026-06-08 14:00", "2026-06-08 17:00", 0.0, "Inward transit & locking to berth (excluded)"),
        ("2026-06-08 17:00", "2026-06-09 03:00", 1.0, "Discharge operations (laytime expires 03:00)"),
        ("2026-06-09 03:00", "2026-06-09 11:00", 0.5, "Severe storm - ASBA Cl.19 half-rate demurrage"),
        ("2026-06-09 11:00", "2026-06-10 07:00", 1.0, "Discharge completed on demurrage")
    ]
    res = calc.process_sof(sof)
    print(f"Allowed: {res['allowed_hrs']}h | Used: {res['used_hrs']}h | Balance: {res['balance_hrs']}h")
    print(f"Status: {res['status']} | Settlement: ${res['financial_settlement_usd']:,.2f}")
```

### Verification and Expected Output
Executing the Python script yields the following result:
```text
Allowed: 72.0h | Used: 96.0h | Balance: -24.0h
Status: DEMURRAGE | Settlement: $36,000.00
```
The program accurately accounts for the 72 hours of baseline laytime, reflects the 4 hours of storm delay at half rate ($8\text{ h} \times 0.5 = 4\text{ h}$ counted against demurrage), counts the 20 hours of final discharge at full rate, and computes an exact settlement of $\$36,000.00$ ($24.0\text{ hours} = 1.00\text{ days} \times \$36,000.00/\text{day}$).

---

## 8. Data, standards, and regulatory context

Bulk commodity logistics is governed by an intricate web of international maritime conventions, industry contract forms, pipeline technical standards, and rail economic regulations.

### 1. Maritime Standard Contracts and Industry Definitions
- **BIMCO (Baltic and International Maritime Council):** Standardizes commercial shipping contracts. Primary forms include **GENCON 1994 / GENCON 2022** (voyage charter for dry bulk), **NYPE 2015** (New York Produce Exchange time charter), and **SUPERTANKVOY**.
- **ASBA (Association of Ship Brokers and Agents):** Publishes **ASBATANKVOY 1977**, the most widely used voyage charterparty for crude oil and dirty petroleum products, and **ASBACHEMVOY** for chemical parcel tankers. Major oil companies issue standard proprietary rider forms modifying ASBATANKVOY: **BPVOY4**, **SHELLVOY6**, **EXXONMOBIL 2005**, and **CHEVRONVOY**.
- **Laytime Definitions for Charter Parties 2013:** Jointly drafted and issued by BIMCO, CMI (Comité Maritime International), FONASBA, and INTERCARGO. Establishes globally standardized legal definitions for over 30 contested terms, including "Weather Working Day," "Always Safely Afloat," "Reversible Laytime," and "Notice of Readiness."

### 2. International Maritime Safety and Environmental Law
- **IMO MARPOL Convention (Annex I - Prevention of Pollution by Oil):** Governs crude oil tanker operations. Imposes strict requirements on Crude Oil Washing (COW), Dedicated Clean Ballast Tanks (CBT), Segregated Ballast Tanks (SBT), and Oil Discharge Monitoring and Control (ODME) systems. Cargo operations cannot proceed if inert gas systems (IGS) fail to maintain tank oxygen below $8\%$ by volume.
- **SOLAS (Safety of Life at Sea):** Mandates emergency tow-off pendants, fire-fighting monitors, and structural hull integrity standards.
- **ISPS Code (International Ship and Port Facility Security Code):** Mandates port security declaration levels; security inspections delaying vessel berthing are governed by specific charterparty ISPS clauses.

### 3. Pipeline Standards and Hazardous Materials Regulations
- **API (American Petroleum Institute) Standards:**
  - **API MPMS (Manual of Petroleum Measurement Standards):** Chapter 11 (Temperature and Pressure Volume Correction Factors), Chapter 8 (Standard Practice for Manual Sampling of Petroleum), and Chapter 5 (Metering).
  - **API RP 1130:** Computational Pipeline Monitoring for Liquids (algorithmic leak detection and batch interface tracking).
- **U.S. DOT PHMSA (Pipeline and Hazardous Materials Safety Administration):** 49 CFR Part 195 (Transportation of Hazardous Liquids by Pipeline). Establishes statutory maximum operating pressure (MOP), hydrostatic pressure testing, and inline inspection (ILI intelligent pigging) schedules.
- **ASTM Product Specifications:** Defines contamination cut thresholds. ASTM D4814 (Automotive Spark-Ignition Engine Fuel / Gasoline), ASTM D975 (Diesel Fuel Oils), and ASTM D1655 (Aviation Turbine Fuels / Jet A/A-1).

### 4. Railroad Regulatory Frameworks
- **U.S. Surface Transportation Board (STB):** Economic regulatory agency overseeing freight rail carriers. STB Docket **EP 754** (*Demurrage and Accessorial Charges*) regulates Class I carrier demurrage tariffs, establishing billing accuracy rules, dispute resolution frameworks, and credit offsets for carrier-caused delays.
- **AAR (Association of American Railroads):** Interchange rules governing mechanical fitness, wheel bearing acoustic monitoring, and draft gear standards for rolling stock.
- **FAST Act (Fixing America's Surface Transportation Act of 2015):** Mandated the phase-out of legacy DOT-111 and non-jacketed CPC-1232 tank cars for flammable bulk liquids (crude oil and ethanol) in favor of **DOT-117** tank cars, featuring $9/16\text{-inch}$ normalized steel tanks, full-height head shields, thermal protection jackets, and high-capacity pressure relief valves.

---

## 9. Industry practice and case evidence

### Case 1: The 2020 Global Tanker Demurrage Crisis and Floating Storage Congestion
During the second quarter of 2020, the COVID-19 pandemic caused a historic collapse in global liquid fuels demand (~$20\text{ million bpd}$ drop). Refineries drastically cut runs, storage tank farms reached operational tank-top, and a severe crude contango market emerged ($F_{\text{future}} \gg S_{\text{spot}}$), rendering floating storage financially lucrative.

```text
Contango Arbitrage Condition:
  F(T) - S(0) > Cost of Vessel Time-Charter + Insurance + Bunker Boil-Off + Inspection
```

Over 150 VLCCs and Suezmax tankers were chartered as floating storage or queued outside major discharge hubs (Rotterdam, Singapore, Qingdao, U.S. Gulf Coast). Waiting times ballooned to 20 to 45 days. Spot demurrage rates soared from $\$30,000/\text{day}$ to over $\$180,000/\text{day}$.
- *Legal Battlegrounds:* Refiners attempted to declare *Force Majeure* under ASBATANKVOY Clause 19 to cut demurrage liabilities in half or extinguish them entirely.
- *Arbitration Outcome:* London and New York maritime arbitration panels overwhelmingly rejected generalized pandemic demand collapse as a valid *Force Majeure* defense, ruling that commercial unprofitability or tank-top at the buyer's terminal does not excuse the charterer's fundamental contractual duty to provide a discharge berth or pay full demurrage liquidated damages. Major trading houses faced tens of millions of dollars in unhedged demurrage losses.

### Case 2: Colonial Pipeline Multi-Product Scheduling and Transmix Management
The Colonial Pipeline system is the largest refined products transmission system in the United States, spanning 5,500 miles between Houston, Texas, and Linden, New Jersey, carrying over $100\text{ million gallons}$ (2.4 million barrels) of refined products daily.
- *Cycle Architecture:* Colonial operates on a strict **5-day scheduling cycle** (72 cycles per year). Every 5 days, the sequence of 40+ distinct product grades repeats across its two main transmission lines (Line 1 for gasoline, Line 2 for distillates).
- *Transmix Logistics:* Because Line 1 and Line 2 handle billions of barrel-miles annually, interface dispersion generates hundreds of thousands of barrels of transmix per month. At major breakout junctions (Greensboro, NC; Atlanta, GA; Linden, NJ), automated optical densitometers trigger pneumatic diverter valves to route transmix into dedicated intermediate storage tanks.
- *Industrial Economics:* Dedicated transmix distillation splitters (e.g., Gladieux Energy, Western Refining) purchase this transmix at crude-parity discounts, re-fractionate it into motor gasoline blendstock and low-sulfur diesel, and reinject it into the market, closing an industrial circular loop.

### Case 3: Queensland Coal Rail-Port Queuing and System Demurrage
In the Dalrymple Bay Coal Terminal (DBCT) and Hay Point coal export terminals in Queensland, Australia, export supply chains connect dozens of inland Bowen Basin open-cut coal mines via Aurizon rail networks to marine terminals servicing Capesize bulkers ($150,000\text{ to }220,000\text{ DWT}$).
- *The Congestion Crisis:* In the late 2000s, lack of operational coordination between independent mine loaders, above-rail haulage operators, and terminal ship loaders caused vessel queues at DBCT to exceed 50 Capesize vessels. Average vessel waiting times reached 25 to 30 days. Demurrage incurred by coal producers exceeded $\$1\text{ billion}$ annually, eroding up to $\$12/\text{ton}$ of production margin.
- *The Analytics Intervention:* The industry established the Integrated Coal Logistics Systems (ICLS) coordination mechanism. Real-time discrete-event simulation and dynamic vessel arrival queuing algorithms were integrated with rail schedule planning. Instead of a pure "first-come, first-served" vessel arrival queue, vessels were assigned dynamic laycan slots based on rail unit train coal delivery progress, shrinking the offshore queue from 50+ ships to fewer than 15, and slashing system-wide demurrage payments by over $70\%$.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign | Root Mechanism | Operational & Financial Consequence | Mitigation Protocol |
|---|---|---|---|---|---|
| 1 | **Defective / Premature NOR Tender** | Master tenders NOR before vessel reaches formal port limit or before passing hold cleanliness survey. | Surveyor finds hold contamination or customs denies entry; NOR legally void *ab initio*. | Entire initial waiting time (days) fails to count as laytime; charterer avoids demurrage. | Re-tender fresh NOR immediately upon curing defect; verify CP contains WIPON/WIFPON riders. |
| 2 | **Misinterpreting Demurrage Maxim** | Commercial operations stops laytime clock for rain or Sunday after vessel is already on demurrage. | Failure to apply English law maxim: *"Once on demurrage, always on demurrage."* | Under-billing demurrage to counterparty by hundreds of thousands of dollars. | Enforce continuous 24/7 demurrage tracking unless CP explicitly states exceptions apply while on demurrage. |
| 3 | **Demurrage Claim 90-Day Time-Bar** | Demurrage invoice submitted without full supporting documentation (SOF, NOR, pumping logs). | Standard charterparty clauses (e.g., BPVOY4 Cl. 20) void all claims not submitted within 60 or 90 days with all documentation. | Total legal forfeiture of valid demurrage claims; balance sheet write-off. | Automated demurrage workflow software flagging claim submission deadlines 30 days in advance. |
| 4 | **Laminar Pipeline Throttling** | Pipeline pump station drops pressure to conserve electricity, allowing $Re$ to fall below 4,000. | Flow regime transitions toward laminar; velocity profile becomes parabolic. | Transmix interface length expands by $500\%$; massive product contamination across batches. | Strict SCADA minimum flow rate interlocks preventing pipeline operation below critical turbulent Reynolds number. |
| 5 | **Transmix Contamination of Jet A-1** | Diesel or heavy distillate cut into jet fuel receipt tank; flash point or water separation rating fails. | Optical densitometer calibration drift or delayed valve actuation at terminal manifold. | Contamination of millions of gallons of aviation fuel; potential grounding of commercial flights. | Enforce conservative heart-cut buffers; install dual redundant coriolis and spectroscopic sensors on jet lines. |
| 6 | **Transmix Tank Overflow / Line Shut-In** | Terminal transmix storage tank reaches $95\%$ capacity with incoming interface en route. | Off-spec interface production exceeds off-take trucking or transmix splitter distillation capacity. | Entire pipeline transmission line must be shut in; upstream refineries forced to cut processing rates. | Implement transmix tank inventory alerts; establish emergency backup downgrade cut protocols. |
| 7 | **Unit Train Railcar Bunching** | Multiple 100-car trainsets arrive at destination terminal within hours of each other. | Linehaul mainline congestion followed by simultaneous dispatcher releases by the Class I railroad. | Terminal track capacity exceeded; cars held on mainline; massive STB demurrage tariffs assessed. | Dynamic slot-reservation protocols; establish auxiliary storage track agreements with regional shortlines. |
| 8 | **Private Railcar Demurrage on Mainlines** | Shipper-owned tank cars held on carrier sidings due to terminal unloading rack congestion. | STB tariffs permit Class I railroads to assess daily storage fees on private cars held on railroad tracks. | Unbudgeted railroad detention penalties exceeding $\$100,000$ per trainset. | Implement pull-in scheduling matching rack throughput; balance car release rates with actual pump rates. |
| 9 | **Tidal Draft Grounding / Demurrage Trap** | Vessel loaded to maximum draft arrives at shallow port during neap tide. | Failure to match vessel arrival draft with lunar tidal cycle tables and dredging depth. | Vessel forced to wait at sea for spring tides (7–10 days) or hire costly lightering barges. | Dynamic voyage speed optimization (Virtual Arrival) to align arrival with high-tide windows. |
| 10 | **Deballasting Bottleneck Blocking Transfer** | Vessel cargo loading arms idle while vessel pumps ballast water into harbor or shore facility. | Vessel ballast pumps undersized or ballast water treatment system (BWTS) filter clogged. | Time lost is charged strictly to owner's account; laytime suspended; charterer disputes time-sheet. | Audit vessel pumping logs; require minimum certified deballasting rates in charterparty warranties. |
| 11 | **Manifold Valve Weeping Contamination** | Off-spec product detected in clean tank during adjacent batch manifold transfer. | Internal valve seat wear in multiport manifold allowing pressurized fluid to leak across closed block valves. | Cross-contamination of finished product; high giveaway or reprocessing costs. | Mandate double block and bleed (DBB) valve arrangements with pressure monitoring cavities at manifolds. |
| 12 | **Inland Barge Low-Water Draft Stranding** | River barges loaded to 9-foot draft encounter seasonal drought dropping channel to 7 feet. | Inadequate river gauge monitoring; over-loading barges relative to forecasted river stages. | Flotilla groundings; expensive mid-river fleeting and barge-to-barge lightering required. | Dynamically adjust barge loading draft limits based on 14-day USACE river level forecasts. |

---

## 11. Assumptions and limits

| # | Assumption | Operational Purpose | Boundary / Breaking Condition | Diagnostic & Verification Method |
|---|---|---|---|---|
| 1 | **Instantaneous Berth Availability** | Simplifies voyage planning models by assuming vessel berths immediately upon arrival. | Severe port congestion, strikes, or adverse weather requiring days of anchorage waiting. | Port line-up reports, AIS vessel tracking data, and historical waiting time distributions. |
| 2 | **Constant Pro Rata Demurrage Rate** | Models vessel detention cost as a linear function of elapsed time. | Step-rate demurrage clauses (e.g., rate increases by $25\%$ after 5 days on demurrage). | Review of charterparty fixture recap rider clauses for escalating rate terms. |
| 3 | **Turbulent Plug Flow in Pipelines** | Assumes bulk parcel core moves uniformly without internal radial concentration gradients. | Low-velocity operations, high-viscosity crude blends, or winter cold restarts ($Re < 2,100$). | Real-time Reynolds number calculation via inline viscometers and flow meters. |
| 4 | **Negligible Molecular Diffusion vs. Dispersion** | Ignores Fickian molecular diffusion relative to turbulent eddy diffusivity in Taylor dispersion. | Low-velocity pipeline shut-in (static line holding fluid for days during outages). | Static soak tests measuring interface growth over prolonged line shut-downs. |
| 5 | **Newtonian Fluid Behavior** | Assumes linear relationship between shear stress and strain rate in pipeline batching. | Waxy crude oils, heavy emulsions, or drilling muds exhibiting Bingham plastic or thixotropic behavior. | High-pressure rotational viscometry at multiple shear rates and operating temperatures. |
| 6 | **Deterministic Unit Train Transit Times** | Assumes fixed linehaul transit velocity in fleet sizing and scheduling equations. | Mainline congestion, locomotive breakdowns, adverse weather, or crew timeout delays. | Historical Class I railroad GPS wheel-movement logs and corridor transit variance distributions. |
| 7 | **Perfect Separation in Terminal Tanks** | Assumes receipt storage tanks prevent stratification and internal product mixing. | Heavy product layering over light product, creating rollover or density inversion risks. | Multi-level automatic tank gauge (ATG) temperature and density profiling. |
| 8 | **Continuous Weather Working Day Uniformity** | Assumes weather impact is binary across the entire 24-hour day. | Brief, intermittent squalls or high swell events halting operations for discrete hours. | Port authority official meteorological logs and terminal crane/hose operation records. |
| 9 | **Independent Port Laytime Accounting** | Calculates load and discharge port laytime separately under default non-reversible terms. | Charterparties containing explicit Reversible or Average Laytime rider clauses. | Contractual clause audit identifying laytime consolidation agreements. |
| 10 | **Instantaneous Sensor Response at Cut Manifolds** | Assumes densitometers detect interface transitions with zero measurement lag. | Sensor dampening, viscous coating on optical prisms, or sensor dead-legs in sampling loops. | Routine calibration sweeps using certified density standards and fast-loop bypass flow verification. |

---

## 12. Exercises

### Exercise 1: Legal Conditions for Valid Notice of Readiness (NOR)
- **Deliverable:** A formal memorandum (approx. 300 words) evaluating whether an NOR tendered by the master of a dry bulk carrier (*MV Pacific Pioneer*) is legally valid under a GENCON 1994 charterparty.
- **Scenario:** The vessel arrives at the roads outside the port of Newcastle at 16:30 on Friday. The commercial berths are congested. The master transmits NOR via email at 16:45. Free pratique has not yet been applied for, and the vessel's hold cleanliness inspection by the terminal surveyor cannot take place until Monday morning because the surveyor is off duty. The charterparty contains a standard WIBON clause but contains no WIFPON or hold-inspection waiver riders.
- **Acceptance Criteria:** Correctly analyze the three legal tests (Arrived ship, Readiness in all respects, Administrative clearance). Identify the specific defects that invalidate the Friday NOR and state the exact day and time when a valid NOR can be tendered.
- **Estimated Time:** 25 minutes.

### Exercise 2: Multi-Port Reversible Laytime and Demurrage Calculation
- **Deliverable:** An audited Laytime Statement and financial reconciliation showing the net laytime balance and demurrage or despatch payable.
- **Data:**
  - Contract: Reversible laytime clause. Total laytime allowed: 96.00 hours SHINC.
  - Demurrage rate: $\$40,000.00/\text{day}$ pro rata. Despatch rate: $\$20,000.00/\text{day}$ pro rata.
  - Load Port: NOR tendered Monday 08:00. Turn-time: 6 hours. Loading operations commence Monday 14:00. Loading completed Wednesday 02:00. No delays or interruptions.
  - Discharge Port 1: NOR tendered Friday 12:00. Turn-time: 6 hours. Berth available immediately; operations begin Friday 15:00. Operations suspended Saturday 08:00 to 20:00 due to shore conveyor belt fire. Operations completed Sunday 18:00.
  - Discharge Port 2: NOR tendered Tuesday 06:00. Turn-time: 6 hours. Operations begin Tuesday 12:00. Operations completed Thursday 18:00.
- **Acceptance Criteria:** Correctly compute countable hours at each port, determine the consolidated used laytime, state whether demurrage or despatch applies, and provide the exact dollar settlement.
- **Estimated Time:** 35 minutes.

### Exercise 3: Taylor-Aris Interface Volume and Pipeline Transmix Growth
- **Deliverable:** Mathematical derivation and calculation of transmix interface growth across a transmission pipeline.
- **Data:**
  - Pipe Diameter: $D = 16\text{ inches} = 0.4064\text{ m}$. Length: $L = 480\text{ km} = 298.26\text{ miles}$.
  - Product Flow Rate: $Q = 850\text{ m}^3/\text{h}$.
  - Products: Jet A-1 ($\nu_1 = 2.0 \times 10^{-6}\text{ m}^2/\text{s}$) and ULSD ($\nu_2 = 3.6 \times 10^{-6}\text{ m}^2/\text{s}$).
- **Requirements:**
  1. Compute the fluid velocity $v$ and the average Reynolds number $Re$.
  2. Calculate the total transmix volume $V_{\text{transmix}}$ in barrels using the Austin-Palfrey equation.
  3. Determine the axial physical length of the interface zone ($\Delta L_{\text{int}}$ in meters) inside the pipe upon arrival at the delivery terminal.
- **Acceptance Criteria:** Reproducible arithmetic; correct unit conversions between metric and oilfield units.
- **Estimated Time:** 30 minutes.

### Exercise 4: Economic Transmix Cut-Point Optimization
- **Deliverable:** Formulate an economic optimization model to determine the optimal cut fractions for an interface arriving at a terminal manifold.
- **Data:** Total transmix volume is $12,000\text{ bbl}$ between Premium Gasoline (value $\$105/\text{bbl}$) and Regular Gasoline (value $\$92/\text{bbl}$).
  - Option A: Cut $100\%$ to a transmix splitter facility at a processing fee of $\$16/\text{bbl}$, yielding $55\%$ Regular Gasoline and $42\%$ Diesel, with $3\%$ volumetric loss.
  - Option B: Downgrade $50\%$ of the interface directly into the Regular Gasoline tank (downgrade loss from $\$105$ to $\$92/\text{bbl}$ on the Premium portion) and send the remaining $50\%$ to the splitter.
- **Acceptance Criteria:** Calculate the total net loss for Option A and Option B, and identify the economically superior strategy.
- **Estimated Time:** 25 minutes.

### Exercise 5: Rail Unit Train Fleet Sizing and Demurrage Exposure
- **Deliverable:** Complete rolling-stock fleet sizing calculation and demurrage financial sensitivity analysis for an industrial chemical plant.
- **Data:**
  - Annual volume: $1,800,000\text{ MT}$ of chemical grade liquid feedstock.
  - Car capacity: $72\text{ MT}$ per tank car. Train size: $90\text{ cars}$.
  - Route: $950\text{ km}$ one way.
  - Speeds: Loaded commercial speed $28\text{ km/h}$; empty commercial speed $35\text{ km/h}$.
  - Terminal operations: Load terminal dwell $16\text{ hours}$; unload terminal dwell $20\text{ hours}$; slack buffer $12\text{ hours}$. Maintenance spare allowance: $10\%$.
  - Rail demurrage tariff: $\$125/\text{car/day}$ after 24 hours free time.
- **Requirements:**
  1. Calculate the single trainset cycle time $T_{\text{cycle}}$ in days.
  2. Compute the minimum required dedicated trainsets and total railcar fleet size.
  3. Calculate the financial demurrage exposure if severe terminal rack freezing adds 3 days of delay past free time to $20\%$ of all annual train dispatches.
- **Acceptance Criteria:** Clear step-by-step arithmetic and accurate fleet rounding.
- **Estimated Time:** 30 minutes.

### Exercise 6: Auditing a Disputed Statement of Facts and Demurrage Claim
- **Deliverable:** A critical audit report (approx. 400 words) identifying three contractual errors in a shipowner's $\$85,000$ demurrage claim under an ASBATANKVOY charterparty.
- **Scenario:** The shipowner claims:
  1. Laytime commenced immediately upon NOR tender at anchorage, ignoring the 6-hour turn-time clause because "the berth was already occupied by another charterer's vessel."
  2. Shifting time from anchorage to berth ($3.5\text{ hours}$) was counted as laytime because "the shift was ordered by the port authority, not the owner."
  3. Demurrage ran at full rate during a 16-hour severe harbor storm that forced the vessel off the berth, with the owner citing "Once on demurrage, always on demurrage."
- **Acceptance Criteria:** Cite relevant ASBATANKVOY clauses (Clauses 6, 7, and 19) to refute or adjust each of the three claims.
- **Estimated Time:** 30 minutes.

### Exercise 7: Designing a Multi-Product Pipeline 10-Day Cycle Schedule
- **Deliverable:** A complete cyclic batch schedule table for a 10-day cycle on a multi-product pipeline delivering four refined products (RUL, PUL, Jet A-1, ULSD).
- **Requirements:**
  1. Define a valid sequence satisfying all product adjacency compatibility constraints.
  2. Allocate volumetric parcel sizes matching regional market consumption: $45\%$ RUL, $15\%$ PUL, $15\%$ Jet A-1, $25\%$ ULSD.
  3. Specify the interface disposition strategy for every boundary in the cycle.
- **Acceptance Criteria:** Zero illegal adjacencies; complete volume balance; clear transmix handling rules.
- **Estimated Time:** 35 minutes.

### Exercise 8: Integrated Berth-Tank Farm Dynamic Scheduling MILP
- **Deliverable:** Mathematical formulation of a Mixed-Integer Linear Program (MILP) to schedule vessel berthing and tank farm transfers to minimize demurrage while preventing tank overflow.
- **Requirements:** Formulate decision variables, objective function, and constraints for: (a) berth non-overlapping, (b) tank inventory dynamic bounds, and (c) demurrage non-linear activation via auxiliary variables.
- **Acceptance Criteria:** Mathematically sound optimization notation; complete index definitions.
- **Estimated Time:** 40 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Legal Conditions for Valid NOR
1. **Arrived Ship:** Satisfied under the WIBON clause. The vessel is at the customary waiting area within port jurisdiction, unable to berth solely due to commercial congestion.
2. **Readiness in All Respects:** **Failed.** Physical readiness requires cargo holds to be clean, dry, and inspected. Because the terminal surveyor is unavailable until Monday morning, the holds have not been certified ready. Under English maritime law (*The Tres Flores* [1973]), a vessel is not ready to load if an inspection or hold preparation remains outstanding, unless such inspection is a mere formality.
3. **Administrative Clearance:** Questionable without a WIFPON rider. If free pratique has not been applied for or granted, the NOR is premature.
- **Conclusion:** The NOR tendered at 16:45 on Friday is **legally invalid**. A valid NOR cannot be tendered until Monday morning, immediately after the independent surveyor inspects and passes the holds and free pratique is confirmed.

---

### Answer to Exercise 2: Multi-Port Reversible Laytime and Demurrage
- **Total Allowed Laytime:** $96.00\text{ hours}$.
- **Load Port Accounting:**
  - NOR: Mon 08:00. Turn-time 6h: Mon 08:00 to 14:00 (operations begin 14:00, matching turn-time expiry).
  - Working period: Mon 14:00 to Wed 02:00 = $36.00\text{ hours}$.
  - Countable Load Port = $36.00\text{ hours}$.
- **Discharge Port 1 Accounting:**
  - NOR: Fri 12:00. Operations begin Fri 15:00 (earlier than 6h turn-time expiry at 18:00). Under maritime rules, laytime commences upon commencement of operations if earlier than turn-time.
  - Active time: Fri 15:00 to Sat 08:00 = $17.00\text{ hours}$.
  - Delay: Sat 08:00 to 20:00 ($12.00\text{ hours}$). Shore equipment breakdown (conveyor fire) is for charterer's account unless expressly excluded. Laytime continues to count: $12.00\text{ hours}$.
  - Active time: Sat 20:00 to Sun 18:00 = $22.00\text{ hours}$.
  - Countable Port 1 = $17.00 + 12.00 + 22.00 = 51.00\text{ hours}$.
- **Discharge Port 2 Accounting:**
  - NOR: Tue 06:00. Turn-time expires Tue 12:00 (operations start 12:00).
  - Working period: Tue 12:00 to Thu 18:00 = $54.00\text{ hours}$.
  - Countable Port 2 = $54.00\text{ hours}$.
- **Reversible Reconciliation:**
  $$\text{Total Laytime Consumed} = 36.00 + 51.00 + 54.00 = 141.00\text{ hours}$$
  $$\text{Net Balance} = 96.00 - 141.00 = -45.00\text{ hours (Demurrage)}$$
  $$\text{Demurrage Payable} = \frac{45.00}{24.00}\text{ days} \times \$40,000.00/\text{day} = 1.875\text{ days} \times \$40,000.00 = \mathbf{\$75,000.00}$$

---

### Answer to Exercise 3: Taylor-Aris Interface Volume and Pipeline Transmix Growth
1. **Velocity and Reynolds Number:**
   - $D = 0.4064\text{ m} \implies A = \frac{\pi}{4}(0.4064)^2 = 0.12972\text{ m}^2$.
   - $Q = 850\text{ m}^3/\text{h} = 0.23611\text{ m}^3/\text{s}$.
   - $v = \frac{0.23611}{0.12972} = 1.8202\text{ m}/\text{s}$.
   - $\nu_{\text{avg}} = \frac{2.0 + 3.6}{2} \times 10^{-6} = 2.8 \times 10^{-6}\text{ m}^2/\text{s}$.
   - $Re = \frac{1.8202 \times 0.4064}{2.8 \times 10^{-6}} = \frac{0.73973}{2.8 \times 10^{-6}} \approx \mathbf{264,190}$ (Fully turbulent).
2. **Transmix Volume (Austin-Palfrey Correlation):**
   - $D = 16\text{ inches} \implies D^{2.5} = 16^{2.5} = 1,024$.
   - $L = 298.26\text{ miles}$.
   - $Re^{0.1} = (264,190)^{0.1} \approx 3.479$.
   - $V_{\text{transmix}} = 1.175 \times 1,024 \times \sqrt{\frac{298.26}{3.479}} = 1,203.2 \times \sqrt{85.73} = 1,203.2 \times 9.259 \approx \mathbf{11,140\text{ barrels}} \approx 1,771\text{ m}^3$.
3. **Physical Length of Interface Zone ($\Delta L_{\text{int}}$):**
   $$\Delta L_{\text{int}} = \frac{V_{\text{transmix}}(\text{m}^3)}{A(\text{m}^2)} = \frac{1,771\text{ m}^3}{0.12972\text{ m}^2} \approx \mathbf{13,652\text{ meters}} \approx \mathbf{13.65\text{ km}}$$

---

### Answer to Exercise 4: Economic Transmix Cut-Point Optimization
- Total Interface: $12,000\text{ bbl}$.
- **Option A (100% to Splitter):**
  - Processing cost: $12,000\text{ bbl} \times \$16/\text{bbl} = \$192,000$.
  - Output yields:
    - Regular Gasoline: $12,000 \times 0.55 = 6,600\text{ bbl} \times \$92 = \$607,200$.
    - Diesel: $12,000 \times 0.42 = 5,040\text{ bbl} \times \$96 = \$483,840$.
    - Total Gross Realization = $\$607,200 + \$483,840 = \$1,091,040$.
  - Net Value = $\$1,091,040 - \$192,000 = \$899,040$.
  - Baseline value of components before mixing (assuming 50/50 blend):
    $(6,000 \times \$105) + (6,000 \times \$92) = \$630,000 + \$552,000 = \$1,182,000$.
  - **Net Loss under Option A:** $\$1,182,000 - \$899,040 = \mathbf{\$282,960}$.
- **Option B (50% Downgrade into Regular Gasoline, 50% to Splitter):**
  - Downgrade portion ($6,000\text{ bbl}$, containing $3,000\text{ bbl}$ Premium and $3,000\text{ bbl}$ Regular):
    - Absorbed into Regular Gasoline tank: $6,000\text{ bbl} \times \$92 = \$552,000$.
    - Baseline value of this portion: $(3,000 \times \$105) + (3,000 \times \$92) = \$315,000 + \$276,000 = \$591,000$.
    - Loss on downgrade portion = $\$591,000 - \$552,000 = \$39,000$.
  - Splitter portion ($6,000\text{ bbl}$):
    - Processing fee: $6,000 \times \$16 = \$96,000$.
    - Yields: $3,300\text{ bbl}$ Reg ($\$303,600$) + $2,520\text{ bbl}$ Diesel ($\$241,920$) = $\$545,520$.
    - Net value = $\$545,520 - \$96,000 = \$449,520$.
    - Loss on splitter portion = $\$591,000 - \$449,520 = \$141,480$.
  - **Total Loss under Option B:** $\$39,000 + \$141,480 = \mathbf{\$180,480}$.
- **Conclusion:** **Option B is economically superior**, saving $\mathbf{\$102,480}$ relative to Option A by maximizing high-value component absorption into on-spec finished storage tanks.

---

### Answer to Exercise 5: Rail Unit Train Fleet Sizing and Demurrage
1. **Cycle Time:**
   - $t_{\text{transit, loaded}} = \frac{950}{28 \times 24} = \frac{950}{672} \approx 1.414\text{ days}$.
   - $t_{\text{transit, empty}} = \frac{950}{35 \times 24} = \frac{950}{840} \approx 1.131\text{ days}$.
   - Dwell & Slack: $\frac{16 + 20 + 12}{24} = \frac{48}{24} = 2.000\text{ days}$.
   - $T_{\text{cycle}} = 1.414 + 1.131 + 2.000 = \mathbf{4.545\text{ days}}$.
2. **Fleet Sizing:**
   - Train capacity: $90\text{ cars} \times 72\text{ MT} = 6,480\text{ MT/train}$.
   - Annual departures: $\frac{1,800,000}{6,480} \approx 277.78\text{ trains/year} \implies 0.761\text{ departures/day}$.
   - $N_{\text{trains}} = \lceil 0.761 \times 4.545 \rceil = \lceil 3.459 \rceil = \mathbf{4\text{ dedicated unit trainsets}}$.
   - Active cars: $4 \times 90 = 360\text{ cars}$.
   - Total fleet ($10\%$ spare): $\lceil 360 \times 1.10 \rceil = \mathbf{396\text{ tank cars}}$.
3. **Demurrage Exposure:**
   - Affected trains: $277.78 \times 0.20 \approx 55.56\text{ trains}$.
   - Delay past free time: $3\text{ days}$.
   - Total car-days: $55.56\text{ trains} \times 90\text{ cars} \times 3\text{ days} = 15,000\text{ car-days}$.
   - Financial Demurrage = $15,000\text{ car-days} \times \$125/\text{car/day} = \mathbf{\$1,875,000/\text{year}}$.

---

### Rubric for Exercises 6, 7, and 8
- **Exercise 6 (SOF Audit):** Full credit requires citing Clause 6 (turn-time applies regardless of berth congestion under standard ASBA terms), Clause 7 (shifting between anchorage and berth is strictly excluded from laytime), and Clause 19 (storm delay while on demurrage is paid at half rate, directly disproving the owner's full-rate claim).
- **Exercise 7 (Pipeline Schedule):** Full credit requires a bidirectional sequence (e.g., RUL $\leftrightarrow$ PUL $\leftrightarrow$ Jet A-1 $\leftrightarrow$ ULSD), strictly avoiding Jet A-1 / ULSD adjacent interfaces where diesel dye or lubricity additives exist, matching component volume fractions, and specifying explicit cut points to transmix tanks.
- **Exercise 8 (MILP Model):** Full credit requires correct mathematical notation for binary sequencing variables $y_{uv}$, continuous arrival and departure variables, inventory conservation equations linking unloading rates to dynamic tank levels, and linear representation of demurrage penalties via auxiliary slack variables.

---

## 14. Interview questions

### Question 1: How does a Berth Charter differ from a Port Charter regarding laytime commencement, and how do WIPON/WIBON clauses bridge the operational gap?
- **What a strong answer contains:** Clear distinction between the legal geographic boundaries. In a Berth Charter, a vessel is not an "arrived ship" until firmly moored alongside the designated berth; risk of berth congestion falls entirely on the shipowner. In a Port Charter, arrival at the customary waiting area within port limits constitutes an arrived ship, transferring congestion risk to the charterer. A strong answer explains that rider clauses such as **WIBON** ("Whether In Berth Or Not") and **WIPON** ("Whether In Port Or Not") convert a Berth Charter into a virtual Port Charter for laytime counting purposes, allowing valid NOR tendering at anchorage, provided the vessel is physically ready to work.
- **What a weak answer reveals:** Confuses charterparties with bills of lading; asserts that laytime always begins the moment an anchor drops; fails to recognize the legal distinction between commercial port limits and specific jetty berths.

### Question 2: Explain the common-law doctrine "Once on demurrage, always on demurrage." Under what specific contractual conditions can laytime exceptions still apply?
- **What a strong answer contains:** Identifies the foundational legal presumption that once the agreed laytime expires, the charterer is in continuous breach of contract (or subject to continuous agreed liquidated damages). Consequently, standard laytime exceptions (weekends, holidays, bad weather) cease to protect the charterer. To suspend demurrage, the contract must contain express, unmistakable wording stating that the exception applies *while on demurrage* (e.g., ASBATANKVOY Clause 19 half-rate demurrage during storms or fires). Furthermore, delays caused entirely by the vessel's own fault, crew strike, or mechanical breakdown never count as demurrage.
- **What a weak answer reveals:** Believes that weather working day exclusions automatically pause demurrage; fails to distinguish between exceptions to laytime and exceptions to demurrage.

### Question 3: Why must multi-product liquid pipelines operate in turbulent flow ($Re > 4,000$), and how does Taylor dispersion govern the volume of transmix formed?
- **What a strong answer contains:** Explains fluid mechanics: laminar flow ($Re < 2,100$) features a parabolic velocity profile where centerline velocity is twice the mean, causing the trailing fluid to finger through the center and smearing the interface over enormous distances. In turbulent flow ($Re > 4,000$), intense eddy diffusivity flattens the velocity profile into a blunt plug. Taylor-Aris dispersion theory shows that radial turbulent mixing counteracts axial shear, confining interface growth to $\sqrt{L}$. The candidate discusses how the interface volume scales with $D^{2.5} \sqrt{L}$ and how transmix cutting strategies optimize downgrade costs against reprocessing fees.
- **What a weak answer reveals:** Believes pipeline batches are separated by mechanical solid plugs (pigs) at all times; cannot define the Reynolds number or explain why turbulence reduces mixing volume.

### Question 4: How does Class I railroad demurrage differ from ocean shipping demurrage, and what operational strategies mitigate unit train detention under STB tariffs?
- **What a strong answer contains:** Contrasts maritime contractually negotiated liquidated damages with railroad tariff-based regulatory fees governed by the U.S. Surface Transportation Board (STB EP 754). Explains that rail demurrage applies on a per-car-day basis after strict 24-to-48-hour free time windows. Differentiates railroad cars from private cars stored on private vs. carrier tracks. Outlines physical terminal mitigation strategies: continuous loop tracks to eliminate decoupling, high-rate rotary car dumpers, automated bottom-dump trestles, and collaborative dispatching software to prevent train bunching on mainlines.
- **What a weak answer reveals:** Views rail demurrage as an unavoidable freight cost; lacks knowledge of the 24-hour STB free time window or the mechanics of loop tracks versus switching yards.

### Question 5: What is Reversible Laytime, and under what operational circumstances should a commodity trader negotiate for it?
- **What a strong answer contains:** Defines Reversible Laytime as the contractual right to aggregate allowed laytime across loading and discharging ports into a single consolidated pool. A trader should aggressively negotiate for reversible terms when loading terminal performance is predictable and fast (e.g., modern offshore terminal), but destination discharge performance is uncertain, subject to tidal restrictions, or chronically congested. Aggregation allows surplus hours earned at load port to absorb discharge port delays, avoiding demurrage penalties.
- **What a weak answer reveals:** Confuses reversible laytime with despatch; does not understand how non-reversible contracts calculate demurrage independently at each port.

### Question 6: A pipeline densitometer experiences a calibration drift of $+2\%$. What are the operational and safety consequences at the delivery terminal?
- **What a strong answer contains:** Traces the downstream physical failure mode. A positive density bias causes the automated valve controller to detect the arrival of a heavier product (e.g., diesel) prematurely, cutting the interface too early into the heavier tank and dumping gasoline into the diesel pool. This severely depresses the diesel flash point below statutory safety limits ($< 52^\circ\text{C}$), creating an explosive vapor hazard in downstream commercial storage. Alternatively, if cutting into jet fuel, it causes immediate jet fuel rejection.
- **What a weak answer reveals:** Treats sensor calibration drift as a minor bookkeeping error; fails to link density shifts to flash point and vapor pressure combustion hazards.

### Question 7: How do you mathematically formulate the objective function of a multi-modal terminal scheduling problem?
- **What a strong answer contains:** Formulates an optimization objective that balances direct delay costs (vessel demurrage, railcar detention fees, tug and pilotage idling) against operational execution costs (pipeline interface downgrade penalties, transmix reprocessing costs, off-spec slop storage recycling). Accurately incorporates piecewise linear or step-function demurrage activation past the contractual laytime horizon.
- **What a weak answer reveals:** Proposes simple FIFO (First-In, First-Out) queuing heuristics; neglects tank capacity constraints and non-linear demurrage penalty cliffs.

### Question 8: Under what circumstances can a charterer successfully claim *Force Majeure* to extinguish a maritime demurrage liability?
- **What a strong answer contains:** Cites international arbitration standards (LMAA, SMA). *Force Majeure* cannot be invoked simply because market prices crashed, a buyer defaulted, or a port is congested due to ordinary commercial delays. The charterer must demonstrate: (1) an unforeseen, extraordinary event beyond reasonable control (e.g., earthquake, wartime blockade, sudden catastrophic canal closure), (2) that performance was physically impossible, not merely commercially unprofitable, and (3) that no alternative reasonable workarounds existed. Mentions specific CP clauses (e.g., ASBATANKVOY Clause 19) that provide half-rate demurrage rather than complete exoneration.
- **What a weak answer reveals:** Believes commercial unprofitability or ordinary bad weather qualifies as *Force Majeure*; unaware that English maritime law strictly interprets *Force Majeure* clauses.

---

## 15. Key terms

- **ASBATANKVOY 1977:** Standard tanker voyage charterparty form published by ASBA, widely used for international crude oil movements.
- **Arrived Ship:** A vessel that has reached the commercial and legal limits of the port, placing itself at the disposal of the charterer.
- **Austin-Palfrey Correlation:** Empirical equation estimating the volume of transmix interface formed in turbulent multi-product pipeline flow.
- **Berth Charter:** Charterparty where the vessel must be firmly moored alongside the designated berth before valid NOR can be tendered.
- **Cancelling Date:** The final calendar day and hour of the laycan window; if the vessel has not tendered valid NOR, the charterer may cancel the charter.
- **Class I Railroad:** Major freight railroad carrier with annual operating revenues exceeding STB statutory revenue thresholds (e.g., BNSF, UP, CSX, NS).
- **Contract of Affreightment (COA):** Long-term volume contract agreeing to transport specified cargo quantities over time without naming a single specific hull.
- **Deadfreight:** Compensation paid to the shipowner for unutilized chartered capacity when cargo is short-delivered.
- **Deballasting Delay:** Port delay caused by the vessel's inability to discharge ballast water fast enough to keep pace with cargo loading.
- **Demurrage:** Contractual liquidated damages paid by the charterer to the shipowner for detaining the vessel past the allowed laytime.
- **Despatch (Dispatch):** Financial rebate paid by the shipowner to the charterer for completing cargo operations before laytime expires.
- **DOT-117 Tank Car:** Federally mandated high-hazard flammable train tank car featuring $9/16\text{-inch}$ steel, full head shields, and thermal insulation.
- **Downgrade Cut:** Routing contaminated pipeline interface into a lower-tier product pool to ensure high-tier product remains on-spec.
- **GENCON 1994 / 2022:** Standard dry bulk voyage charterparty published by BIMCO, widely used for coal, grain, ore, and fertilizer.
- **Heart Cut:** Aggressive diversion of a pipeline interface into a dedicated transmix tank, preserving adjacent batch purities.
- **Laycan:** Contractual calendar window ("Laydays / Cancelling") defining the period within which the vessel must present itself at the load port.
- **Laytime:** Contractually agreed time allocated to the charterer for loading and discharging cargo without incurring detention charges.
- **Loop Track:** Continuous balloon track at a bulk rail terminal enabling continuous loading or unloading without decoupling locomotives.
- **Notice of Readiness (NOR):** Formal written instrument tendered by the shipmaster certifying that the vessel has arrived and is ready to work cargo.
- **Once on Demurrage, Always on Demurrage:** Maritime law doctrine stating that laytime exceptions cease to apply once allowed laytime has expired.
- **Port Charter:** Charterparty where valid NOR can be tendered as soon as the vessel reaches customary waiting areas within port limits.
- **Reversible Laytime:** Clause allowing unused laytime from the loading port to be transferred and added to the discharging port laytime allowance.
- **Reynolds Number ($Re$):** Dimensionless hydraulic ratio of inertial forces to viscous forces governing laminar vs. turbulent flow regimes.
- **Rotary Car Dumper:** Mechanical terminal dumper that rotates a railcar $160^\circ$ while still coupled to empty bulk solids in seconds.
- **Statement of Facts (SOF):** Certified chronological operational log of all vessel movements, delays, weather, and pumping times at a port.
- **Taylor-Aris Dispersion:** Longitudinal smearing of a fluid interface in pipe flow driven by velocity gradients and cross-stream diffusion.
- **Transmix:** Commingled hydrocarbon interface mixture produced between adjacent batches in a multi-product pipeline.
- **Turn-Time:** Contractual grace period (e.g., 6 hours) between NOR receipt and the formal commencement of the laytime clock.
- **Weather Working Day (WWD):** A 24-hour day on which work is not halted by adverse meteorological conditions.
- **WIBON / WIPON:** Contractual riders meaning "Whether In Berth Or Not" and "Whether In Port Or Not," expanding valid NOR tendering boundaries.

---

## 16. Sources

1. **Association of Ship Brokers and Agents (U.S.A.), Inc. (ASBA):** *ASBATANKVOY Voyage Charter Party*, Standard Form, New York, 1977.
2. **BIMCO (Baltic and International Maritime Council):** *GENCON General Charter (as revised 1994 and 2022)*, Copenhagen, Denmark.
3. **BIMCO, CMI, FONASBA, INTERCARGO:** *Laytime Definitions for Charter Parties 2013*, London / Copenhagen, 2013.
4. **Schofield, John:** *Laytime and Demurrage*, 7th Edition, Informa Law from Routledge, London, 2016. ISBN: 978-1-138-89202-6.
5. **Taylor, Geoffrey I.:** "Dispersion of Soluble Matter in Solvent Flowing Slowly through a Tube," *Proceedings of the Royal Society of London. Series A*, Vol. 219, No. 1137, pp. 186–203, 1953. DOI: 10.1098/rspa.1953.0139.
6. **Aris, Rutherford:** "On the Dispersion of a Solute in a Fluid Flowing through a Tube," *Proceedings of the Royal Society of London. Series A*, Vol. 235, No. 1200, pp. 67–77, 1956. DOI: 10.1098/rspa.1956.0065.
7. **Austin, J. E., and Palfrey, J. R.:** "Mixing of Hydrocarbons in Pipelines," *Proceedings of the American Society of Civil Engineers*, Pipeline Division, 1964.
8. **American Petroleum Institute (API):** *Manual of Petroleum Measurement Standards (MPMS) Chapter 11 - Physical Properties Data*, Washington, D.C.
9. **U.S. Surface Transportation Board (STB):** *Demurrage and Accessorial Charges*, Docket No. EP 754, Final Rule, Washington, D.C., 2020.
10. **International Maritime Organization (IMO):** *MARPOL 73/78: International Convention for the Prevention of Pollution from Ships*, Consolidated Edition, London, 2022.

---

## 17. Links

- **Parent Module Index:** [Module 38 — Process Industries, Commodities, and Bulk Supply Chains](_index.md)
- **Course Overview / Survey:** [Supply Chain Analytics, AI, and Data Science: 48-Module Survey](../../../course/38_process_industries_commodities.md)
- **Sibling Topics in Module 38:**
  - [Topic 01 — Continuous, Batch, and Discrete Production](01_continuous-batch-discrete.md)
  - [Topic 02 — Yield and Mass Balance Modeling](02_yield-mass-balance.md)
  - [Topic 03 — Quality Specifications and Blending](03_quality-specs-blending.md)
  - [Topic 04 — Campaigns and Changeover](04_campaigns-changeover.md)
  - [Topic 05 — Bulk Storage and Inventory Management](05_bulk-storage-inventory.md)
  - [Topic 06 — Commodity Price and Hedging](06_commodity-price-hedging.md)
  - [Topic 08 — Capacity and Debottlenecking](08_capacity-debottlenecking.md)
  - [Topic 09 — Energy, Water, and Emissions](09_energy-water-emissions.md)
  - [Topic 10 — Process Safety and Regulation](10_process-safety-regulation.md)
- **Related Program Electives:**
  - [Module 11 — Network Design Optimization](../../02_methods/11_network_design_facility_location/_index.md)
  - [Module 15 — Stochastic Inventory & Risk-Pooling](../../02_methods/15_inventory_theory_stochastic/_index.md)
  - [Module 16 — Supply Chain Network Economics](../../02_methods/16_pricing_revenue_management/_index.md)
  - [Module 39 — Mining, Metals, and Forestry Supply Chains](../39_mining_metals_forestry/_index.md)
