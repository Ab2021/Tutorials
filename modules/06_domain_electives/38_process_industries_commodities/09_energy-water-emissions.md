# Energy, Water, and Emissions: Utility Integration, Water Pinch Analysis, and Carbon-Optimal Operations Planning

> Module 38 — Process Industries, Commodities, and Bulk Supply Chains
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 16 (Supply Chain Network Economics), Module 20 (Operations Planning & Master Scheduling), Module 38 Topic 01 (Continuous, Batch, and Discrete Production), Module 38 Topic 02 (Yield and Mass Balance Modeling), Module 38 Topic 08 (Capacity and Debottlenecking)
> Version 2026.5

---

## 1. Why this topic exists

In discrete assembly manufacturing, utilities such as compressed air, cooling water, and electricity are traditionally treated as indirect facility overheads. They are allocated across finished goods via fixed standard-cost absorption rates per machine-hour or labor-hour. A production planner scheduling an automotive stamping press or an electronics pick-and-place surface mount line does not recalculate the electrical grid heat rate, modulate cycle times based on hourly wholesale power LMP (Locational Marginal Price), or dynamically trade off boiler fuel consumption against wastewater treatment effluent penalties.

In the process and commodity transformation sectors—including chemicals, petrochemical refining, pulp and paper, base metal smelting, cement, industrial gases, and food ingredients—utilities are not secondary overheads. They are primary operating cost drivers and physical constraints that govern throughput, yield, and economic survival:

1. **Energy as a Co-Feedstock and Conversion Bottleneck:** A world-scale ethylene cracker, chlor-alkali electrolysis plant, or mechanical pulp mill consumes energy on the scale of hundreds of megawatts of electric power and hundreds of metric tons per hour of high-pressure steam. Utility costs frequently constitute 30% to 65% of net cash conversion costs. Furthermore, steam production is chemically and thermodynamically coupled to exothermic reaction headers, waste heat recovery boilers, and combined heat and power (CHP) turbines. If steam extraction drops below boiler turn-down limits or if cooling water exceeds thermal discharge permits, production must curtail regardless of product demand.
2. **Water Scarcity, Intake Limits, and Zero Liquid Discharge (ZLD):** Chemical and refining complexes process millions of cubic meters of water annually for reaction mediums, boiler feedwater (BFW), evaporative cooling towers, and multi-stage counter-current wash trains. In water-stressed industrial basins (such as the U.S. Gulf Coast, Northern India, Western Australia, or the Mediterranean), freshwater extraction permits impose hard volumetric caps. Furthermore, environmental regulations penalize or prohibit effluent discharge based on Total Dissolved Solids (TDS), Chemical Oxygen Demand (COD), heavy metals, and thermal discharge limits. Water reuse must be engineered systematically using **Water Pinch Analysis** to determine the thermodynamic minimum freshwater intake and minimum wastewater generation before capital is spent on biological or membrane wastewater plants.
3. **Volatile Wholesale Electricity and Demand Charges:** Process plants operate in deregulated power markets characterized by extreme intraday price volatility. Day-ahead and real-time wholesale power prices oscillate from negative values during surplus renewable generation (solar and wind oversupply) to scarcity caps exceeding $\$2,000/\text{MWh}$ during grid stress events. In addition, industrial tariffs impose steep monthly transmission and capacity demand charges ($\$10/\text{kW}$ to $\$35/\text{kW}$ of peak 15-minute demand). Running an energy-intensive unit (such as an Air Separation Unit compressor or an Electric Arc Furnace) across a single peak hour can incur hundreds of thousands of dollars in capacity ratchets that persist for an entire billing year. Production schedules must be dynamically co-optimized with electricity price signals and demand limits.
4. **Carbon Compliance Costs and Border Adjustments:** Industrial facilities operate under binding decarbonization frameworks, including the European Union Emissions Trading System (EU ETS Directive 2003/87/EC), California Cap-and-Trade (AB 32), and the EU Carbon Border Adjustment Mechanism (CBAM Regulation 2023/956). Direct combustion and calcination emissions (Scope 1) and purchased electricity/steam (Scope 2) face carbon pricing between $\$50$ and $\$120$ per metric ton of $\text{CO}_2$ equivalent ($\text{tCO}_2\text{e}$). Failing to embed marginal emissions factors and carbon allowance consumption into production scheduling results in dispatching high-emission, low-margin campaigns that incur financial penalties exceeding product gross margins.
5. **Decarbonization Capital and Operating Trade-offs:** Decarbonization pathways—such as process electrification via industrial heat pumps and Mechanical Vapor Recompression (MVR), green hydrogen fuel substitution, biomass co-firing, and post-combustion Carbon Capture, Utilization, and Storage (CCUS)—impose severe operating penalties. For instance, amine-based $\text{CO}_2$ capture requires $2.5\text{ to }3.5\text{ GJ}$ of thermal energy per ton of captured $\text{CO}_2$, consuming up to 30% of a plant's low-pressure steam capacity. Planners must evaluate the levelized cost of carbon abatement across complex, interconnected utility headers.

Treating energy, water, and carbon as decoupled, post-hoc utility accounting entries leads to sub-optimal campaign schedules, violated environmental discharge permits, punitive demand charges, and balance sheet exposure to carbon penalties. Mastering utility integration, water pinch curves, demand response scheduling, and emissions-constrained optimization is the foundation for resilient process supply chain operations.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Map and model** industrial utility balances, quantifying steam header enthalpy cascades, pressure-reduction valve (PRV) exergy losses, and combined heat and power (CHP) backpressure turbine power generation.
2. **Execute Water Pinch Analysis**, constructing limiting water profiles, concentration-mass load tables, and water composite curves to determine minimum freshwater intake and wastewater discharge targets.
3. **Synthesize Water Allocation Networks (WAN)**, solving linear programming transshipment and superstructure formulations to allocate process-to-process water reuse streams within maximum contaminant limits.
4. **Formulate Mixed-Integer Linear Programs (MILP)** for industrial demand response, optimizing multi-period production schedules, peak demand limits, and intermediate buffer storage under time-varying electricity tariffs.
5. **Quantify Scope 1, Scope 2 (location-based and market-based), and marginal grid carbon emissions**, incorporating carbon taxes, ETS allowance caps, and CBAM liabilities directly into production objective functions.
6. **Evaluate industrial decarbonization technologies**, calculating the Levelized Cost of Carbon Abatement (LCCA), energy penalties of carbon capture systems, and thermodynamic coefficients of performance (COP) for industrial heat pumps and MVR units.
7. **Identify operational failure modes and structural trade-offs**, detecting pinch violations, boiler cycling fatigue, cooling tower fouling, and demand ratchet penalties before execution.
8. **Audit compliance architectures**, aligning operational data models with ISO 50001 (Energy Management), ISO 14046 (Water Footprint), and the GHG Protocol Corporate Standard.

---

## 3. The core mental model

An integrated process plant is not an isolated collection of unit operations; it is a thermodynamically and chemically coupled network where mass and energy flows circulate through shared utility infrastructure. Feedstocks are converted into products, while utility systems supply heat, work, and mass-separating agents, simultaneously generating air emissions, solid wastes, and wastewater effluents.

```text
=============================================================================================================
             INTEGRATED PROCESS PLANT: ENERGY, WATER, AND EMISSIONS FLOW ARCHITECTURE
=============================================================================================================

ENERGY & UTILITY CASCADE (STEAM & POWER):
  Fuel Input (Gas, Biomass, H2) ──> [ High-Pressure Boiler ] ──> HP Steam (60-100 bar, 450-500°C)
                                                                       │
                                              ┌────────────────────────┴────────────────────────┐
                                              ▼                                                 ▼
                                     [ Backpressure Turbine ]                           [ PRV / De-superheater ]
                                     (Power Gen: W_cogen MW)                            (Pure Exergy Loss)
                                              │                                                 │
                                              └────────────────────────┬────────────────────────┘
                                                                       ▼
                                                              MP Steam (15-25 bar) ──> Process Reboilers
                                                                       │
                                              ┌────────────────────────┴────────────────────────┐
                                              ▼                                                 ▼
                                     [ Condensing Turbine ]                             [ Low-P Extraction ]
                                     (Power Gen: W_cond MW)                            (Strippers, Evaporators)
                                              │                                                 │
                                              ▼                                                 ▼
                                     Cooling Tower Duty                                 LP Steam (3-5 bar)
                                                                                                │
                                                                                                ▼
                                                                                        Condensate Return

WATER PINCH & REUSE NETWORK:
  Raw Freshwater Intake (F_FW) ──> [ Treatment / Demin Plant ] ──> Boiler Feedwater & Process Makeup
                                                                       │
                                                                       ▼
    Process 1: Scrubber Wash  ──── (Effluent C1) ───┐          [ Water Reuse Header ]
                                                   │                   │
    Process 2: Reactor Rinse  ──── (Effluent C2) ───┼──> (Pinch Filter) ┼──> Process 3: Stripper Wash
                                                   │                   │
    Process 3: Condenser Wash ──── (Effluent C3) ───┘                   ▼
                                                              Wastewater Effluent (F_WW)
                                                                       │
                                                                       ▼
                                                          [ Wastewater Treatment Plant ]
                                                          (Biological, RO, ZLD Evap)

CARBON & DEMAND RESPONSE SIGNALS:
  Grid Power Import (P_t MW)  <─── [ Real-Time Pricing (p_t) & Marginal Grid EF_t ] <─── ISO / Utility Market
  Peak Demand Ratchet (P_max) <─── [ 15-Minute Capacity Demand Charge Constraints ]
  Scope 1 Combustion + Rx     ───> [ On-Site Stacks ] ──> CEMS Data ──> [ EU ETS / Carbon Tax ]
  Scope 2 Grid Electricity    ───> [ Market-Based Accounting ]       ──> [ Scope 2 Liability ]
=============================================================================================================
```

### The Three Coupled Physical Engines

1. **The Thermodynamic Exergy Cascade (Combined Heat and Power):** High-temperature fuel combustion in industrial boilers generates high-pressure superheated steam ($60\text{ to }100\text{ bar}$). Rather than directly throttling steam to process pressure levels through pressure-reduction valves (which destroys exergy without producing work), steam expands through backpressure steam turbines to generate electrical or mechanical power. The extracted medium-pressure ($15\text{ to }25\text{ bar}$) and low-pressure ($3\text{ to }5\text{ bar}$) steam supplies thermal energy to reboilers, evaporators, and reactors. Condensate is recovered, polished, and recycled as boiler feedwater. The operational decision is balancing the ratio of on-site cogeneration versus wholesale grid power import based on fuel cost, turbine efficiency, and dynamic power prices.
2. **The Water Mass-Transshipment Cascade (Water Pinch):** Water acts as a mass-separating agent that extracts contaminants (e.g., salts, organics, suspended solids, acids) from process streams. In a conventional once-through layout, every unit operation draws virgin freshwater and discharges contaminated wastewater. Water Pinch Analysis treats the plant as a transshipment system where the effluent from a low-contaminant process (e.g., reactor final rinse or steam condensate) is redirected as the feed for a higher-contaminant process (e.g., raw gas scrubber or solids washing). The system is limited by the **Pinch Concentration**, which establishes the boundary below which freshwater must be supplied and above which wastewater must be rejected.
3. **The Environmental-Economic Scheduling Engine:** The operations schedule bridges physical thermodynamic flows and dynamic market regulations. The scheduling model decides hourly unit production rates, furnace firing, and utility boiler loads. It optimizes across three cost vectors: physical fuel/chemical consumption, time-varying grid electricity tariffs (including peak demand capacity charges), and carbon emissions liabilities (Scope 1 stack emissions and Scope 2 indirect grid emissions). Intermediate buffer inventory tanks (e.g., demineralized water tanks, intermediate chemical vessels, cryogenic liquid storage) decouple energy-intensive upstream units from steady downstream operations, enabling aggressive load shifting during high-price or high-emission grid intervals.

---

## 4. Definitions and vocabulary

| Term | Definition | Common Misuse | Unit / Standard |
|---|---|---|---|
| **Water Pinch Analysis** | Systematic methodology based on mass exchange integration to determine minimum freshwater intake and minimum wastewater discharge targets prior to designing the reuse network. | Confusing Water Pinch with simple end-of-pipe recycling or treating it as identical to thermal Heat Pinch. | $\text{t/h}$ or $\text{m}^3/\text{h}$; Wang & Smith (1994) |
| **Limiting Water Profile** | The maximum permissible contaminant concentration inlet and outlet specifications for a given water-using process stream. | Assuming processes have fixed water flowrates rather than fixed contaminant mass loads. | $\text{ppm}$, $\text{mg/L}$, or $\text{kg/t}$ |
| **Water Allocation Network (WAN)** | The physical topology of pipes, pumps, splitters, mixers, and treatment units that distributes freshwater, reuse water, and effluent across processes. | Mistaking the conceptual pinch target for the implemented piping network. | Dimensionless topology / P&ID |
| **Pinch Concentration ($C^*$)** | The contaminant concentration at which the freshwater supply line touches the limiting water composite curve, representing the thermodynamic mass-transfer bottleneck. | Believing water can be recycled freely across the pinch without increasing freshwater consumption. | $\text{ppm}$ or $\text{mg/L}$ |
| **Combined Heat and Power (CHP)** | Simultaneous generation of electrical/mechanical power and useful thermal energy from a single primary fuel source (cogeneration). | Treating electric power generation and steam generation as independent operating variables. | $\text{MW}$ electrical, $\text{t/h}$ steam; ISO 50001 |
| **Backpressure Turbine** | Steam turbine where exhaust steam is discharged at positive gauge pressure to supply industrial process heating headers. | Confusing with condensing turbines, which expand steam below atmospheric pressure to maximize electrical yield. | $\text{bar}$, $\text{MW}$; ASME PTC 6 |
| **Willans Line** | Linear or affine empirical relationship between steam turbine mass flowrate and electrical power output: $W = a \cdot F - b$. | Assuming turbine efficiency remains constant across all turn-down operating rates. | $\text{MW}$ vs. $\text{t/h}$ |
| **Pressure Reduction Valve (PRV)** | An isenthalpic throttling device that drops steam pressure without extracting mechanical work, accompanied by water de-superheating. | Assuming PRV throttling preserves exergy because enthalpy is conserved ($\Delta h = 0$). | $\text{bar}$, $\text{kJ/kg}$ |
| **Demand Response (DR)** | Intentional modulation of industrial electrical consumption in response to time-varying wholesale prices, grid stress signals, or incentive payments. | Confusing emergency curtailment with planned, continuous economic load shifting. | $\text{MW}$, $\text{MWh}$; FERC Order 2222 |
| **Capacity Demand Charge** | Monthly billing charge based on the maximum average electrical power drawn during any 15-minute or 30-minute interval within the billing period. | Focusing exclusively on volumetric energy charges ($\$/\text{MWh}$) while ignoring the peak ratchet ($\$/\text{kW}$). | $\$ /\text{kW}/\text{month}$ or $\$ /\text{MW}$ |
| **Scope 1 Emissions** | Direct greenhouse gas emissions from sources owned or controlled by the reporting organization (combustion, chemical reactions, calcination). | Omitting non-combustion fugitive and chemical process stoichiometry emissions. | $\text{tCO}_2\text{e}$; GHG Protocol |
| **Scope 2 (Location-Based)** | Indirect emissions from purchased electricity, steam, heating, or cooling calculated using average grid emissions intensity of the regional grid. | Using national average grid factors for localized operational dispatch decisions. | $\text{kgCO}_2\text{e}/\text{MWh}$; GHG Protocol |
| **Scope 2 (Market-Based)** | Indirect emissions calculated using contractual instruments (Power Purchase Agreements, RECs, Guarantees of Origin). | Assuming green contractual attributes reflect physical real-time zero-carbon electron delivery. | $\text{kgCO}_2\text{e}/\text{MWh}$; GHG Protocol |
| **Marginal Emissions Factor (MEF)** | The carbon intensity of the specific marginal power plant dispatched to satisfy the next increment of electrical load at a specific hour. | Using annual average grid carbon intensity to evaluate demand response scheduling benefits. | $\text{kgCO}_2\text{e}/\text{MWh}$; PJM / CAISO / ENTSO-E |
| **EU ETS Allowance (EUA)** | Tradable carbon compliance credit under the EU Emissions Trading System, granting the right to emit one metric ton of $\text{CO}_2\text{e}$. | Treating carbon allowances as a tax rather than a tradable financial asset with balance-sheet carrying costs. | $€/\text{tCO}_2\text{e}$; Directive 2003/87/EC |
| **Carbon Border Adjustment Mechanism (CBAM)** | EU regulatory tariff equalizing the carbon cost of imported carbon-intensive goods with domestic EU production under EU ETS. | Assuming CBAM applies only to direct emissions; Scope 2 and precursor embedded emissions are increasingly in scope. | $€/\text{tCO}_2\text{e}$; Regulation 2023/956 |
| **Specific Energy Consumption (SEC)** | Total net primary energy consumed per unit of marketable finished product output. | Comparing SEC across plants without standardizing for feedstock quality, ambient temperature, or battery limits. | $\text{GJ/t}$ or $\text{MWh/t}$ |
| **Specific Water Consumption (SWC)** | Total net freshwater withdrawn from external sources per unit of finished product output. | Measuring internal recirculation volume instead of net external freshwater withdrawal. | $\text{m}^3/\text{t}$ product; ISO 14046 |
| **Mechanical Vapor Recompression (MVR)** | Decarbonization technology where low-pressure evaporated vapor is compressed to higher pressure and temperature, recycling latent heat into reboilers. | Treating MVR as an ordinary condenser rather than an open-cycle industrial heat pump. | $\text{kW}$, $\text{COP}$; IEA Annex 35 |
| **Coefficient of Performance (COP)** | Ratio of useful thermal heat delivered to the electrical or mechanical work input in a heat pump or refrigeration cycle: $\text{COP} = Q_{heat} / W_{in}$. | Believing electric resistance heating ($\text{COP} \le 1.0$) is more efficient than heat pumps ($\text{COP} = 2.5\text{ to }5.0$). | Dimensionless ratio |
| **Levelized Cost of Carbon Abatement (LCCA)** | Net present value of incremental capital and operating expenditures divided by lifecycle greenhouse gas emissions avoided. | Evaluating abatement projects purely on CAPEX without incorporating dynamic utility price spreads. | $\$ /\text{tCO}_2\text{e}$ avoided |
| **Zero Liquid Discharge (ZLD)** | Wastewater treatment configuration where all effluent is purified and recycled, leaving only dry solid salts for disposal. | Ignoring the massive thermal and electrical energy penalty required to drive brine crystallizers. | $\text{t/h}$ brine; $\text{kWh/m}^3$ |

---

## 5. The mechanics, step by step

Industrial utility integration, water conservation, and emissions minimization require a unified optimization framework that spans thermodynamics, mass integration, electricity market dynamics, and environmental economics.

### 5.1 Industrial Utility Systems and Combined Heat and Power (CHP) Modeling

A large industrial site operates a hierarchical steam distribution system structured across multiple pressure headers:
- **High-Pressure (HP) Header:** Typically $60\text{ to }100\text{ bar}$, $450\text{ to }500^\circ\text{C}$, fed by fuel-fired utility boilers, waste heat boilers (WHB), and chemical recovery boilers.
- **Medium-Pressure (MP) Header:** Typically $15\text{ to }25\text{ bar}$, $250\text{ to }320^\circ\text{C}$, supplying high-temperature distillation reboilers, heavy preheaters, and steam stripping columns.
- **Low-Pressure (LP) Header:** Typically $3\text{ to }5\text{ bar}$, $140\text{ to }180^\circ\text{C}$, supplying general building heating, low-temperature reboilers, vacuum ejectors, and deaerators.

#### Steam Header Mass and Enthalpy Conservation
For each steam header $h \in \{\text{HP}, \text{MP}, \text{LP}\}$, mass conservation requires that total inflows equal total outflows:

$$\sum_{b \in \mathcal{B}_h} F_{b,t}^{\text{boil}} + \sum_{t' \in \mathcal{T}_{in,h}} F_{t',t}^{\text{turb,out}} + \sum_{p \in \mathcal{P}_{in,h}} F_{p,t}^{\text{PRV,out}} = \sum_{d \in \mathcal{D}_h} F_{d,t}^{\text{demand}} + \sum_{t'' \in \mathcal{T}_{out,h}} F_{t'',t}^{\text{turb,in}} + \sum_{p' \in \mathcal{P}_{out,h}} F_{p',t}^{\text{PRV,in}} + F_{h,t}^{\text{vent}}$$

where $F_{b,t}^{\text{boil}}$ is boiler steam generation, $F^{\text{turb}}$ represents turbine steam flows, $F^{\text{PRV}}$ represents bypass throttling flows, $F^{\text{demand}}$ is process demand, and $F^{\text{vent}}$ is atmospheric venting (loss).

Enthalpy conservation across the header ensures that steam leaving the header matches the mixed enthalpy of the entering streams:

$$H_h^{\text{header}} \cdot \sum F_{in,h} = \sum \left( F_{in,h,i} \cdot h_{in,h,i} \right) + Q_{\text{de-superheat}}$$

#### Turbine Power Generation: The Willans Line
Steam turbines convert the enthalpy drop between headers into mechanical or electrical power. Over normal operating ranges, the relationship between steam mass flow $F_t^{\text{turb}}$ and power generation $W_t$ follows the linear Willans formulation:

$$W_t = \eta_{\text{gen}} \cdot \left( F_t^{\text{turb}} \cdot \Delta h_{\text{isen}} \cdot \eta_{\text{isen}} - L_0 \right) = a_t \cdot F_t^{\text{turb}} - b_t \cdot Y_t^{\text{turb}}$$

where:
- $\Delta h_{\text{isen}} = h_{\text{in}} - h_{\text{out,isen}}$ is the ideal isentropic enthalpy drop ($\text{kJ/kg}$).
- $\eta_{\text{isen}}$ is the turbine isentropic efficiency ($0.70\text{ to }0.85$).
- $\eta_{\text{gen}}$ is the mechanical-electrical generator efficiency ($0.95\text{ to }0.98$).
- $L_0$ is the internal mechanical and aerodynamic no-load loss ($\text{MW}$).
- $Y_t^{\text{turb}} \in \{0, 1\}$ is a binary variable indicating turbine online status.
- $a_t$ is the marginal power generation slope ($\text{MWh/t}$ steam).
- $b_t$ is the fixed no-load intercept ($\text{MW}$).

When process steam demand on the LP header rises, additional steam can be passed through the backpressure turbine from the HP header, generating low-cost marginal power ($W_{\text{cogen}}$). If process steam demand falls, the plant faces a trade-off: throttle steam through PRVs (destroying exergy), vent LP steam to atmosphere (losing water and latent heat), or route steam through a condensing turbine (which generates power but rejects all latent heat into cooling water at low thermal efficiency).

---

### 5.2 Water Pinch Analysis: Limiting Profiles and Composite Curve Targeting

Water Pinch Analysis is an algorithmic, thermodynamics-based mass integration tool introduced by Wang & Smith (1994) to maximize water reuse and minimize freshwater withdrawal.

#### Step 1: Characterize Water-Using Operations
A water-using process operation $k \in \mathcal{K}$ removes a specific mass load of contaminant $\Delta m_k$ ($\text{kg/h}$) from a process stream. The process imposes limits on the maximum acceptable inlet contaminant concentration $C_{\text{in},k}^{\max}$ and the maximum allowable outlet contaminant concentration $C_{\text{out},k}^{\max}$:

$$\Delta m_k = F_k \cdot \left( C_{\text{out},k}^{\max} - C_{\text{in},k}^{\max} \right) \times 10^{-3} \quad [\text{kg/h}]$$

where $F_k$ is the limiting water flowrate ($\text{t/h}$) and concentrations are in parts per million ($\text{ppm} = \text{g/t} = \text{mg/kg}$). The limiting flowrate represents the absolute minimum water required by unit $k$ if it enters at $C_{\text{in},k}^{\max}$ and leaves fully saturated at $C_{\text{out},k}^{\max}$:

$$F_k^{\lim} = \frac{\Delta m_k}{\left( C_{\text{out},k}^{\max} - C_{\text{in},k}^{\max} \right) \times 10^{-3}}$$

#### Step 2: Construct the Limiting Water Composite Curve
To find the aggregate plant water demand profile:
1. Sort all concentration boundaries $\{C_{\text{in},k}^{\max}, C_{\text{out},k}^{\max}\}_{\forall k}$ in ascending order to form discrete concentration intervals $[C_n, C_{n+1}]$.
2. In each interval $n$, identify all processes $k \in \mathcal{K}_n$ that are active (i.e., $C_{\text{in},k}^{\max} \le C_n$ and $C_{\text{out},k}^{\max} \ge C_{n+1}$).
3. Calculate the aggregate water flowrate in interval $n$:
   $$F_n^{\text{int}} = \sum_{k \in \mathcal{K}_n} F_k^{\lim}$$
4. Calculate the contaminant mass load transferred in interval $n$:
   $$\Delta M_n = F_n^{\text{int}} \cdot (C_{n+1} - C_n) \times 10^{-3} \quad [\text{kg/h}]$$
5. Compute cumulative contaminant mass load $M(C)$ as a function of concentration:
   $$M(C_{n+1}) = M(C_n) + \Delta M_n, \quad M(C_0) = 0$$

Plotting Concentration $C$ (y-axis) versus Cumulative Contaminant Mass Load $M$ (x-axis) yields the **Limiting Water Composite Curve**.

```text
Concentration C (ppm)
  ^
  |                                                  / (Composite Curve)
  |                                            .---'
  |                                      .---'   * PINCH POINT (C*, M*)
  |                                .---'       /
  |                          .---'           /  Freshwater Supply Line
  |                    .---'               /    Slope = 1 / (F_FW,min * 10^-3)
  |              .---'                   /
  |        .---'                       /
  |  .---'                           /
  +--------------------------------+----------------------------> Mass Load M (kg/h)
  0                                M*                           M_total
```

#### Step 3: Freshwater Supply Line and Pinch Identification
Freshwater enters the plant with zero or near-zero contaminant concentration ($C_{\text{FW}} = 0$). The freshwater supply line is a straight line originating at the origin $(C=0, M=0)$ on the composite plot, with slope equal to:

$$\frac{dC}{dM} = \frac{1}{F_{\text{FW}} \times 10^{-3}}$$

For a water network to be physically feasible, water supplied to any process cannot have a contaminant concentration higher than the limiting curve at that mass load. Therefore, the freshwater operating line must lie **on or below** the limiting composite curve across all concentration intervals:

$$C_{\text{supply}}(M) \le C_{\text{composite}}(M) \quad \forall M \in [0, M_{\text{total}}]$$

Expressed in terms of freshwater flowrate at each breakpoint $j$:

$$F_{\text{FW}} \ge \frac{M_j}{C_j \times 10^{-3}} \quad \forall j$$

The **Minimum Freshwater Target** ($F_{\text{FW}}^{\min}$) is governed by the bottleneck point where this inequality is tightest:

$$F_{\text{FW}}^{\min} = \max_j \left\{ \frac{M_j}{C_j \times 10^{-3}} \right\}$$

The concentration $C^*$ and cumulative mass $M^*$ where this maximum occurs is the **Water Pinch Point**. The pinch divides the industrial system into two thermodynamically distinct zones:
- **Below the Pinch ($C \le C^*$):** Freshwater must be supplied. Water cannot be rejected to wastewater from this zone without penalizing the freshwater target.
- **Above the Pinch ($C \ge C^*$):** No freshwater should be introduced. All water demand above the pinch should be satisfied by reusing effluent streams from below the pinch. Effluent discharged above the pinch constitutes the plant's unavoidable wastewater effluent.

Assuming no net process water losses or gains, overall mass balance dictates the **Minimum Wastewater Discharge Target**:

$$F_{\text{WW}}^{\min} = F_{\text{FW}}^{\min} + \sum_{k} \Delta F_k^{\text{consumed/lost}} = F_{\text{FW}}^{\min}$$

---

### 5.3 Water Allocation Network (WAN) Superstructure Optimization

Once pinch targets ($F_{\text{FW}}^{\min}, F_{\text{WW}}^{\min}$) are established, engineers synthesize a physical Water Allocation Network. In complex plants with multiple contaminants (e.g., COD, TDS, and suspended solids), graphical pinch curves cannot capture cross-contaminant trade-offs. We formulate the WAN synthesis as a Linear Program (LP) or Mixed-Integer Linear Program (MILP) using a source-sink superstructure.

Let $\mathcal{S}$ be the set of water sources (effluents from processes, plus virgin freshwater) and $\mathcal{D}$ be the set of water demands (inlets to processes, plus effluent discharge to wastewater treatment).
- Let $F_{i,j}$ be the water flowrate routed from source $i \in \mathcal{S}$ to demand $j \in \mathcal{D}$ ($\text{t/h}$).
- Let $F_{\text{FW},j}$ be the freshwater routed directly to demand $j$.
- Let $F_{i,\text{WW}}$ be effluent from source $i$ discharged to the wastewater header.
- Let $C_{i,p}^{\text{src}}$ be the concentration of contaminant $p \in \mathcal{P}$ in source stream $i$.
- Let $C_{j,p}^{\max}$ be the maximum allowable concentration of contaminant $p$ entering demand unit $j$.

#### Optimization Formulation:
$$\min \quad \sum_{j \in \mathcal{D}} c_{\text{FW}} \cdot F_{\text{FW},j} + \sum_{i \in \mathcal{S}} c_{\text{WW}} \cdot F_{i,\text{WW}} + \sum_{i \in \mathcal{S}} \sum_{j \in \mathcal{D}} c_{\text{piping}} \cdot z_{i,j}$$

Subject to:
1. **Demand Flowrate Satisfaction:** Every unit operation $j$ receives its required volumetric water flow:
   $$F_{\text{FW},j} + \sum_{i \in \mathcal{S}} F_{i,j} = F_j^{\text{req}} \quad \forall j \in \mathcal{D}$$
2. **Source Flowrate Availability:** The total water extracted from source $i$ cannot exceed its generated flow:
   $$F_{i,\text{WW}} + \sum_{j \in \mathcal{D}} F_{i,j} = F_i^{\text{avail}} \quad \forall i \in \mathcal{S}$$
3. **Contaminant Concentration Limits at Demands:** Total contaminant entering unit $j$ divided by total flow cannot exceed $C_{j,p}^{\max}$. Multiplying through by total flow linearizes the constraint:
   $$\sum_{i \in \mathcal{S}} F_{i,j} \cdot C_{i,p}^{\text{src}} + F_{\text{FW},j} \cdot C_{\text{FW},p} \le C_{j,p}^{\max} \cdot F_j^{\text{req}} \quad \forall j \in \mathcal{D}, \; \forall p \in \mathcal{P}$$
4. **Piping Connection Constraints (MILP extension):** If pipeline investment cost is modeled:
   $$F_{i,j} \le M_{\text{flow}} \cdot z_{i,j}, \quad z_{i,j} \in \{0, 1\}$$

When solved without binary piping variables, the problem is a pure LP transshipment model that achieves the exact water pinch targets while handling dozens of simultaneous contaminant species.

---

### 5.4 Time-Varying Electricity Pricing, Demand Charges, and Industrial Demand Response

Process plants consuming tens or hundreds of megawatts of electric power are connected to high-voltage transmission networks. They face complex wholesale power tariff structures comprising:
1. **Volumetric Energy Charges:** Hourly Day-Ahead Market (DAM) Locational Marginal Prices ($p_t^{\text{DAM}}$) or 5-minute Real-Time Market (RTM) prices ($p_t^{\text{RTM}}$) in $\$ /\text{MWh}$.
2. **Capacity Demand Charges:** Monthly capacity fees ($\pi_{\text{cap}}$ in $\$ /\text{kW-month}$) assessed on the maximum 15-minute average power demand recorded during on-peak hours across the billing cycle:
   $$\text{Demand Charge} = \pi_{\text{cap}} \cdot \max_{t \in \mathcal{T}_{\text{peak}}} \left\{ P_t^{\text{grid}} \right\}$$
3. **Transmission Ratchet Clauses:** Many utilities stipulate that the billed capacity demand cannot drop below 80% of the peak demand recorded during the preceding 11 months.

#### Demand Response Mechanism via Intermediate Buffer Decoupling
Heavy continuous units (e.g., continuous polymerization reactors, glass float lines, crude distillation columns) cannot change feed rates rapidly without severe quality degradation or catalyst fouling. However, high-power auxiliary and preparation units can modulate load aggressively if separated by intermediate buffer storage:

```text
[ Raw Feedstock ] ──> [ High-Power Upstream Unit ] ──> [ Intermediate Buffer Tank ] ──> [ Continuous Downstream ] ──> [ Market ]
                        (P_t: 0 to P_max MW)              (Inventory: S_t tons)           (Constant Demand: D_t t/h)
                        Air Separation / Grinding         Cryogenic Liquid / Slurry       Chemical Reactor / Paper Machine
```

#### Multi-Period Demand Response MILP Formulation
For time periods $t \in \{1, \dots, T\}$ (typically $T=24$ or $T=168$ hourly steps):

**Decision Variables:**
- $u_t$: Production rate of upstream high-power unit in period $t$ ($\text{t/h}$).
- $s_t$: Inventory level in intermediate buffer tank at the end of period $t$ ($\text{t}$).
- $P_t^{\text{grid}}$: Grid electrical power imported in period $t$ ($\text{MW}$).
- $P^{\text{peak}}$: Peak electrical power demand established across the scheduling horizon ($\text{MW}$).
- $Y_t$: Binary operational status of the upstream unit ($1$ if running, $0$ if shutdown).

**Objective Function:**
$$\min \quad \sum_{t=1}^T \left( p_t^{\text{elec}} \cdot P_t^{\text{grid}} \cdot \Delta t + c_{\text{start}} \cdot v_t \right) + \pi_{\text{cap}} \cdot P^{\text{peak}}$$

**Constraints:**
1. **Buffer Tank Mass Inventory Conservation:**
   $$s_t = s_{t-1} + u_t \cdot \Delta t - D_t \cdot \Delta t \quad \forall t \in \{1, \dots, T\}$$
   $$s^{\min} \le s_t \le s^{\max} \quad \forall t$$
   $$s_T \ge s_{\text{target}}$$
2. **Upstream Unit Operating Limits and Ramping:**
   $$u^{\min} \cdot Y_t \le u_t \le u^{\max} \cdot Y_t \quad \forall t$$
   $$u_t - u_{t-1} \le \Delta u^{\text{ramp,up}} \quad \forall t$$
   $$u_{t-1} - u_t \le \Delta u^{\text{ramp,down}} \quad \forall t$$
3. **Power Consumption Coupling:**
   $$P_t^{\text{grid}} = \beta \cdot u_t + P_t^{\text{base}} - W_t^{\text{cogen}} \quad \forall t$$
   where $\beta$ is specific electrical consumption ($\text{MWh/t}$ product), $P_t^{\text{base}}$ is uncontrollable facility base load, and $W_t^{\text{cogen}}$ is on-site power generated by steam turbines.
4. **Peak Demand Tracking:**
   $$P^{\text{peak}} \ge P_t^{\text{grid}} \quad \forall t \in \mathcal{T}_{\text{peak}}$$

By solving this formulation, the plant automatically curtails production to $u^{\min}$ or zero during peak-price hours, relying on stored buffer inventory $s_t$ to satisfy downstream continuous demand $D_t$, and recharges inventory during off-peak hours when power prices are low or negative.

---

### 5.5 Carbon Accounting and Emissions Penalties in Operations Scheduling

Under mandatory compliance schemes (EU ETS, UK ETS, California WCI) and corporate science-based net-zero targets (SBTi), production schedules must internalize carbon emissions.

#### Emissions Classification in Process Operations
1. **Scope 1 Direct Combustion Emissions:**
   $$E_{1,t}^{\text{comb}} = \sum_{f \in \mathcal{F}} F_{f,t}^{\text{fuel}} \cdot \text{LHV}_f \cdot \text{EF}_f^{\text{comb}} \quad [\text{tCO}_2\text{e}]$$
   where $F_{f,t}^{\text{fuel}}$ is fuel mass or volumetric flow, $\text{LHV}_f$ is lower heating value ($\text{GJ/t}$ or $\text{GJ/Nm}^3$), and $\text{EF}_f^{\text{comb}}$ is the emission factor ($\text{tCO}_2\text{e/GJ}$).
2. **Scope 1 Direct Process / Stoichiometric Emissions:**
   In cement kilns, hydrogen steam methane reformers, and lime calcination, $\text{CO}_2$ is a chemical reaction by-product independent of fuel combustion:
   $$E_{1,t}^{\text{proc}} = \sum_{k} \gamma_k \cdot u_{k,t} \quad [\text{tCO}_2\text{e}]$$
   For example, limestone calcination ($\text{CaCO}_3 \to \text{CaO} + \text{CO}_2$) generates stoichiometrically $\gamma = 0.785\text{ tCO}_2/\text{t CaO}$.
3. **Scope 2 Indirect Electricity Emissions:**
   - **Location-Based Method:** Uses the regional grid average emissions factor:
     $$E_{2,t}^{\text{loc}} = P_t^{\text{grid}} \cdot \Delta t \cdot \text{EF}_{\text{grid,avg}} \quad [\text{tCO}_2\text{e}]$$
   - **Marginal Emissions Factor (MEF) Method for Operational Scheduling:** For operational dispatch, using average grid intensity causes misallocation. When a plant increases grid load by $1\text{ MWh}$, the grid does not dispatch nuclear or hydro; it ramps the marginal balancing generator (typically a natural gas combined-cycle or peaker turbine):
     $$E_{2,t}^{\text{marg}} = P_t^{\text{grid}} \cdot \Delta t \cdot \text{EF}_{t}^{\text{marginal}} \quad [\text{tCO}_2\text{e}]$$

#### Incorporating Carbon Costs into the Operational Objective Function
Under a cap-and-trade or carbon tax regime, total operating expenditure over the scheduling horizon includes carbon penalties:

$$\min \quad J = \sum_{t=1}^T \Big( \text{RawMaterialCost}_t + \text{FuelCost}_t + \text{ElecCost}_t + \text{WaterCost}_t + \lambda_{\text{CO2}} \cdot \mathcal{E}_t^{\text{net}} \Big)$$

where $\lambda_{\text{CO2}}$ is the carbon price ($\$ /\text{tCO}_2\text{e}$) and net taxable emissions $\mathcal{E}_t^{\text{net}}$ are:

$$\mathcal{E}_t^{\text{net}} = E_{1,t}^{\text{comb}} + E_{1,t}^{\text{proc}} + \alpha_{\text{Scope2}} \cdot E_{2,t} - \frac{\text{FreeAllowances}_t}{T}$$

Here, $\alpha_{\text{Scope2}} \in [0, 1]$ reflects whether purchased electricity is regulated under the local regime (e.g., $\alpha = 0$ in the EU ETS where power generators pay directly, but $\alpha = 1$ under California Cap-and-Trade or corporate carbon budgets).

When carbon prices rise to $\$80\text{ to }\$120/\text{tCO}_2$, the marginal cost ranking of alternative manufacturing trains inverts. In a petrochemical refinery, an older, direct gas-fired reboiler train can become more expensive than an electrified MVR train despite higher electricity prices.

---

### 5.6 Industrial Decarbonization Pathways and Abatement Economics

To achieve deep structural decarbonization, process supply chains evaluate four primary technological pathways:

```text
=============================================================================================================
                          PROCESS INDUSTRY DECARBONIZATION PATHWAY SPECTRUM
=============================================================================================================

1. PROCESS ELECTRIFICATION (HEAT PUMPS & MVR)
   Waste Heat Source (60-90°C) ──> [ Industrial Heat Pump (COP: 2.5-4.0) ] ──> Process Steam (120-160°C)
                                                   ^
                                     Electricity (MWh_elec)

2. HYDROGEN & LOW-CARBON FUEL SWITCHING
   Green Hydrogen (H2) / Biomethane ──> [ Dual-Fuel Burners / Furnaces ] ──> High-Temp Heat (>800°C)
   (Challenge: Flame velocity, NOx formation, pipeline embrittlement)

3. POST-COMBUSTION CARBON CAPTURE (CCUS)
   Flue Gas (Stack CO2) ──> [ Amine Absorption Column ] ──> Pure CO2 ──> [ Compression (100 bar) ] ──> Pipeline
                                     │
                                     ▼
                            [ Stripper Column ] <── Low-Pressure Steam Penalty (2.5 - 3.5 GJ/t CO2)

4. BIOMASS & CIRCULAR FEEDSTOCK INTEGRATION
   Agricultural Residues / Bio-naphtha ──> [ Gasification / Co-pyrolysis ] ──> Biogenic Hydrocarbons
=============================================================================================================
```

#### Techno-Economic Evaluation: Levelized Cost of Carbon Abatement (LCCA)
To rank decarbonization investments and operational retrofits, engineers calculate the Levelized Cost of Carbon Abatement:

$$\text{LCCA} = \frac{\Delta \text{CAPEX} \cdot \left[ \frac{r(1+r)^N}{(1+r)^N - 1} \right] + \Delta \text{OPEX}_{\text{annual}}}{\Delta \text{Emissions}_{\text{annual}}} \quad \left[ \frac{\$}{\text{tCO}_2\text{e avoided}} \right]$$

where:
- $\Delta \text{CAPEX}$ is the incremental capital expenditure for the low-carbon asset.
- $r$ is the weighted average cost of capital (discount rate).
- $N$ is asset operating life in years.
- $\Delta \text{OPEX}_{\text{annual}}$ is the change in annual operating costs (electricity, fuel, maintenance, catalysts, water).
- $\Delta \text{Emissions}_{\text{annual}}$ is net lifecycle $\text{tCO}_2\text{e}$ avoided per year.

An abatement option is economically viable if $\text{LCCA} < \lambda_{\text{CO2}}^{\text{market}} + \text{StrategicGreenPremium}$.

#### Industrial Heat Pump (IHP) Thermodynamics
Industrial heat pumps upgrade waste heat from cooling water or low-temperature effluent ($T_{\text{evap}}$) to useful process steam ($T_{\text{cond}}$). The theoretical Carnot efficiency sets the upper thermodynamic bound:

$$\text{COP}_{\text{Carnot}} = \frac{T_{\text{cond}} + 273.15}{(T_{\text{cond}} + 273.15) - (T_{\text{evap}} + 273.15)}$$

The real-world Second-Law efficiency factor ($\eta_{\text{II}} \approx 0.45\text{ to }0.55$) dictates the actual coefficient of performance:

$$\text{COP}_{\text{real}} = \eta_{\text{II}} \cdot \text{COP}_{\text{Carnot}}$$

For an evaporator capturing heat from $70^\circ\text{C}$ effluent to generate $130^\circ\text{C}$ LP steam ($\Delta T_{\text{lift}} = 60\text{ K}$):

$$\text{COP}_{\text{Carnot}} = \frac{403.15}{60} \approx 6.72, \quad \text{COP}_{\text{real}} = 0.50 \times 6.72 = 3.36$$

For every $1.0\text{ MWh}$ of electric power consumed, the heat pump delivers $3.36\text{ MWh}$ ($12.1\text{ GJ}$) of thermal heat to the steam header, eliminating $3.36 / \eta_{\text{boiler}}$ units of fossil fuel combustion.

---

## 6. Worked example

### 6.1 Problem Setup and Process Context

A continuous specialty chemical manufacturing complex in the Rhine-Meuse industrial basin operates three primary water-using washing, scrubbing, and reaction trains. The complex operates $8,400\text{ operating hours/year}$.

#### Current Plant Baseline (Once-Through Water Operation)
The plant tracks Chemical Oxygen Demand (COD) as the limiting contaminant. In the current layout, each unit operation draws pure freshwater ($C_{\text{FW}} = 0\text{ ppm}$) from municipal groundwater intake and discharges directly to the biological wastewater treatment plant (WWTP).

```text
Table 6.1: Water-Using Process Operation Specifications
+-----------+----------------------+-------------------+-------------------+--------------------+--------------------+
| Operation | Description          | Mass Load COD     | Max Inlet COD     | Max Outlet COD     | Limiting Water     |
| Index (k) |                      | Delta m_k (kg/h)  | C_in^max (ppm)    | C_out^max (ppm)    | Flowrate F_k (t/h) |
+-----------+----------------------+-------------------+-------------------+--------------------+--------------------+
| P1        | Acid Scrubber Wash   | 50.0              | 0                 | 500                | 100.0              |
| P2        | Polymer Reactor Rinse| 120.0             | 100               | 700                | 200.0              |
| P3        | Stripper Cond Wash   | 80.0              | 400               | 1,200              | 100.0              |
+-----------+----------------------+-------------------+-------------------+--------------------+--------------------+
| TOTAL     | Baseline Operations  | 250.0 kg/h        | —                 | —                  | 400.0 t/h          |
+-----------+----------------------+-------------------+-------------------+--------------------+--------------------+
```

Baseline Water Costs:
- Freshwater procurement and demineralization: $c_{\text{FW}} = \$1.80/\text{t}$.
- Wastewater treatment and effluent discharge fee: $c_{\text{WW}} = \$3.20/\text{t}$.
- Total combined water cost: $c_{\text{water}} = \$5.00/\text{t}$.
- Baseline hourly water operating cost: $400.0\text{ t/h} \times \$5.00/\text{t} = \$2,000.00/\text{h}$.
- Baseline annual water operating cost: $\$2,000.00/\text{h} \times 8,400\text{ h/yr} = \$16,800,000/\text{yr}$.

---

### 6.2 Step 1: Water Pinch Analysis and Minimum Flowrate Targeting

To determine the minimum freshwater intake and minimum wastewater discharge via Water Pinch Analysis:

1. **Identify Concentration Intervals:**
   Sort all limiting concentration boundaries: $\{0, 100, 400, 500, 700, 1,200\}\text{ ppm}$.
   This defines 5 distinct concentration intervals.

2. **Compute Active Streams, Interval Flowrates, and Mass Loads:**

```text
Table 6.2: Limiting Water Profile Interval Calculation
+----------+--------------------+----------------+-------------------+--------------------+--------------------+
| Interval | Concentration      | Active Process | Summed Interval   | Interval COD Mass  | Cumulative Mass    |
| (n)      | Range [C_n, C_n+1] | Streams        | Flow F_n^int (t/h)| Load Delta M (kg/h)| Load M(C) (kg/h)   |
+----------+--------------------+----------------+-------------------+--------------------+--------------------+
| Int 1    | 0 to 100 ppm       | P1             | 100.0             | 100.0 * 100 * 1e-3 | 10.0               |
|          |                    |                |                   | = 10.0 kg/h        |                    |
| Int 2    | 100 to 400 ppm     | P1, P2         | 100.0 + 200.0     | 300.0 * 300 * 1e-3 | 10.0 + 90.0        |
|          |                    |                | = 300.0           | = 90.0 kg/h        | = 100.0            |
| Int 3    | 400 to 500 ppm     | P1, P2, P3     | 100 + 200 + 100   | 400.0 * 100 * 1e-3 | 100.0 + 40.0       |
|          |                    |                | = 400.0           | = 40.0 kg/h        | = 140.0            |
| Int 4    | 500 to 700 ppm     | P2, P3         | 200.0 + 100.0     | 300.0 * 200 * 1e-3 | 140.0 + 60.0       |
|          |                    |                | = 300.0           | = 60.0 kg/h        | = 200.0            |
| Int 5    | 700 to 1,200 ppm   | P3             | 100.0             | 100.0 * 500 * 1e-3 | 200.0 + 50.0       |
|          |                    |                |                   | = 50.0 kg/h        | = 250.0            |
+----------+--------------------+----------------+-------------------+--------------------+--------------------+
| TOTAL    | 0 to 1,200 ppm     | —              | —                 | 250.0 kg/h         | 250.0 kg/h         |
+----------+--------------------+----------------+-------------------+--------------------+--------------------+
```

3. **Freshwater Flowrate Targeting at Breakpoints:**
The freshwater operating line starts at $(C=0, M=0)$. The required freshwater flowrate to satisfy concentration bounds at each breakpoint $j$ is:

$$F_{\text{FW},j}^{\text{req}} = \frac{M(C_j)}{C_j \times 10^{-3}}$$

```text
Table 6.3: Freshwater Flowrate Targeting Across Breakpoints
+------------------+--------------------------+------------------------------+
| Breakpoint C_j   | Cumulative COD Load M(C) | Required Freshwater Flowrate |
| (ppm)            | (kg/h)                   | F_FW,j^req (t/h)             |
+------------------+--------------------------+------------------------------+
| 100 ppm          | 10.0 kg/h                | 10.0 / 0.100 = 100.00 t/h    |
| 400 ppm          | 100.0 kg/h               | 100.0 / 0.400 = 250.00 t/h   |
| 500 ppm          | 140.0 kg/h               | 140.0 / 0.500 = 280.00 t/h   |
| 700 ppm          | 200.0 kg/h               | 200.0 / 0.700 = 285.71 t/h   |  <== PINCH POINT (C*, M*)
| 1,200 ppm        | 250.0 kg/h               | 250.0 / 1.200 = 208.33 t/h   |
+------------------+--------------------------+------------------------------+
```

#### Results of Pinch Targeting:
- **Pinch Concentration ($C^*$):** $700.0\text{ ppm}$ at cumulative load $M^* = 200.0\text{ kg/h}$.
- **Thermodynamic Minimum Freshwater Intake:** $F_{\text{FW}}^{\min} = 285.71\text{ t/h}$.
- **Thermodynamic Minimum Wastewater Discharge:** $F_{\text{WW}}^{\min} = 285.71\text{ t/h}$.
- **Effluent Contaminant Concentration:**
  $$C_{\text{effluent}} = \frac{250.0\text{ kg/h}}{285.71\text{ t/h} \times 10^{-3}} = 875.0\text{ ppm}$$
- **Volumetric Water Savings:**
  $$\Delta F_{\text{water}} = 400.0 - 285.71 = 114.29\text{ t/h} \quad (28.57\% \text{ reduction})$$
- **Economic Value of Water Pinch Reuse:**
  $$\text{Hourly Cost Savings} = 114.29\text{ t/h} \times \$5.00/\text{t} = \$571.45/\text{h}$$
  $$\text{Annual Water Cost Savings} = \$571.45/\text{h} \times 8,400\text{ h/yr} = \$4,800,180/\text{yr}$$

---

### 6.3 Step 2: Steam Header and Cogeneration Power Balance

The facility operates a natural gas boiler producing High-Pressure (HP) steam at $65\text{ bar}$, $480^\circ\text{C}$ ($h_{\text{HP}} = 3,370\text{ kJ/kg}$).
The plant requires:
- Medium-Pressure (MP) steam: $16\text{ bar}$ ($h_{\text{MP,sat}} = 2,794\text{ kJ/kg}$) at $50.0\text{ t/h}$.
- Low-Pressure (LP) steam: $4\text{ bar}$ ($h_{\text{LP,sat}} = 2,738\text{ kJ/kg}$) at $70.0\text{ t/h}$.
- Total steam generated by boiler: $F_{\text{boil}} = 50.0 + 70.0 = 120.0\text{ t/h}$.

#### Mode A: Pressure Reduction Valves (PRV Throttling Baseline)
If steam is expanded across PRVs with water de-superheating:
- Power generated = $0.0\text{ MW}$.
- High-grade exergy of $65\text{ bar}$ superheated steam is completely dissipated into low-pressure heat.

#### Mode B: Combined Heat and Power (CHP Backpressure Turbine Cascade)
The plant routes all $120.0\text{ t/h}$ through an extraction-backpressure steam turbine:
1. **Stage 1 (HP to MP Expansion):**
   - Steam flow: $120.0\text{ t/h} = 33.333\text{ kg/s}$.
   - Inlet enthalpy: $h_{\text{HP}} = 3,370\text{ kJ/kg}$.
   - Turbine exhaust enthalpy at MP header ($16\text{ bar}$, superheated): $h_{\text{MP,turb}} = 2,920\text{ kJ/kg}$.
   - Actual enthalpy drop: $\Delta h_1 = 3,370 - 2,920 = 450\text{ kJ/kg}$.
   - Generator electrical efficiency: $\eta_{\text{gen}} = 0.96$.
   - Stage 1 Power:
     $$W_1 = 33.333\text{ kg/s} \times 450\text{ kJ/kg} \times 0.96 = 14,400\text{ kW} = 14.40\text{ MW}$$
   - $50.0\text{ t/h}$ of MP steam is discharged to the process reboiler header.
2. **Stage 2 (MP to LP Expansion):**
   - Remaining steam flow: $70.0\text{ t/h} = 19.444\text{ kg/s}$.
   - Inlet enthalpy: $h_{\text{MP,turb}} = 2,920\text{ kJ/kg}$.
   - Exhaust enthalpy at LP header ($4\text{ bar}$): $h_{\text{LP,turb}} = 2,780\text{ kJ/kg}$.
   - Actual enthalpy drop: $\Delta h_2 = 2,920 - 2,780 = 140\text{ kJ/kg}$.
   - Stage 2 Power:
     $$W_2 = 19.444\text{ kg/s} \times 140\text{ kJ/kg} \times 0.96 = 2,613.3\text{ kW} = 2.61\text{ MW}$$
   - All $70.0\text{ t/h}$ enters the LP header to satisfy stripper and building demands.

#### Cogeneration Power Output and Value:
- **Total Cogenerated Power:** $W_{\text{cogen}} = 14.40 + 2.61 = 17.01\text{ MW}$.
- With industrial grid electricity priced at an average of $\$85.00/\text{MWh}$:
  $$\text{Hourly Electricity Value} = 17.01\text{ MW} \times \$85.00/\text{MWh} = \$1,445.85/\text{h}$$
  $$\text{Annual Cogeneration Power Value} = \$1,445.85/\text{h} \times 8,400\text{ h/yr} = \$12,145,140/\text{yr}$$

---

### 6.4 Step 3: 24-Hour Electricity Demand Response and Buffer Scheduling

The chemical site operates an upstream energy-intensive grinding and electrolysis unit that produces intermediate slurry feed for a downstream continuous reactor train.
- Downstream continuous reactor demand: $D_t = 10.0\text{ t/h}$ constant ($240.0\text{ t/day}$).
- Intermediate buffer tank capacity: $s^{\min} = 20.0\text{ t}$, $s^{\max} = 120.0\text{ t}$, initial buffer $s_0 = 50.0\text{ t}$, required ending buffer $s_{24} = 50.0\text{ t}$.
- Upstream unit max capacity: $u^{\max} = 22.0\text{ t/h}$. Specific electrical consumption: $\beta = 0.75\text{ MWh/t}$.
- Peak demand capacity charge: $\pi_{\text{cap}} = \$15.00/\text{MW-day}$.
- Carbon compliance price: $\lambda_{\text{CO2}} = \$60.00/\text{tCO}_2\text{e}$.

We evaluate two operational schedules across a 24-hour day-ahead market price profile:

```text
Table 6.4: 24-Hour Electricity Tariff and Grid Marginal Emissions Profile
+------+------------+------------------+------+------------+------------------+
| Hour | LMP ($/MWh)| Marg EF (t/MWh)  | Hour | LMP ($/MWh)| Marg EF (t/MWh)  |
+------+------------+------------------+------+------------+------------------+
| h00  | 32.00      | 0.32             | h12  | 85.00      | 0.42             |
| h01  | 30.00      | 0.30             | h13  | 80.00      | 0.40             |
| h02  | 28.00      | 0.28             | h14  | 85.00      | 0.43             |
| h03  | 29.00      | 0.29             | h15  | 105.00     | 0.52             |
| h04  | 35.00      | 0.35             | h16  | 150.00     | 0.67             |
| h05  | 55.00      | 0.48             | h17  | 180.00     | 0.72             |
| h06  | 95.00      | 0.62             | h18  | 165.00     | 0.69             |
| h07  | 140.00     | 0.68             | h19  | 120.00     | 0.58             |
| h08  | 160.00     | 0.70             | h20  | 85.00      | 0.46             |
| h09  | 145.00     | 0.66             | h21  | 60.00      | 0.38             |
| h10  | 110.00     | 0.55             | h22  | 45.00      | 0.34             |
| h11  | 90.00      | 0.45             | h23  | 35.00      | 0.32             |
+------+------------+------------------+------+------------+------------------+
```

#### Schedule 1: Baseline Flat Production
The upstream unit runs flat at $u_t = 10.0\text{ t/h}$ every hour.
- Power draw: $P_t = 10.0\text{ t/h} \times 0.75\text{ MWh/t} = 7.50\text{ MW}$ constant.
- Total daily electricity cost:
  $$\text{ElecCost} = 7.50\text{ MW} \times \sum_{t=0}^{23} \text{LMP}_t = 7.50 \times \$2,144.00 = \$16,080.00/\text{day}$$
- Total Scope 2 carbon emissions:
  $$\text{Emissions} = 7.50\text{ MW} \times \sum_{t=0}^{23} \text{EF}_t = 7.50 \times 11.61\text{ t/MWh} = 87.075\text{ tCO}_2\text{e}$$
- Carbon tax liability: $87.075\text{ t} \times \$60.00/\text{t} = \$5,224.50/\text{day}$.
- Capacity demand charge: $7.50\text{ MW} \times \$15.00/\text{MW} = \$112.50/\text{day}$.
- **Total Flat Schedule Daily Cost:**
  $$\$16,080.00 + \$5,224.50 + \$112.50 = \$21,417.00/\text{day}$$

#### Schedule 2: Demand Response and Carbon-Optimized Production
Solving the Linear Program allows the upstream unit to maximize production during cheap, low-carbon hours (h00–h04, h22–h23), throttle back during moderate hours, and shut down completely ($0.0\text{ t/h}$) during peak hours (h07–h09, h16–h18) while maintaining downstream flow from the buffer:
- Total daily production: $240.0\text{ t}$.
- Peak power demand established: $P^{\text{peak}} = 22.0\text{ t/h} \times 0.75 = 16.50\text{ MW}$.
- Optimized Electricity Cost: $\$8,511.00/\text{day}$.
- Optimized Scope 2 Emissions: $63.525\text{ tCO}_2\text{e}$ ($27.0\% \text{ reduction}$).
- Optimized Carbon Tax Liability: $63.525\text{ t} \times \$60.00/\text{t} = \$3,811.50/\text{day}$.
- Capacity Demand Charge: $16.50\text{ MW} \times \$15.00/\text{MW} = \$247.50/\text{day}$.
- **Total Optimized Schedule Daily Cost:**
  $$\$8,511.00 + \$3,811.50 + \$247.50 = \$12,570.00/\text{day}$$

#### Economic Synthesis:
```text
Table 6.5: Daily Operating Economics: Flat Baseline vs. Demand Response Optimized
+-------------------------------+-------------------+-----------------------+---------------------+
| Metric                        | Flat Baseline     | DR & Carbon Optimized | Net Daily Benefit   |
+-------------------------------+-------------------+-----------------------+---------------------+
| Volumetric Electricity Cost   | $16,080.00        | $8,511.00             | +$7,569.00 (-47.1%) |
| Carbon Emissions Liability    | $5,224.50         | $3,811.50             | +$1,413.00 (-27.0%) |
| Capacity Demand Charge        | $112.50           | $247.50               | -$135.00 (+120.0%)  |
+-------------------------------+-------------------+-----------------------+---------------------+
| Total Daily Operating Cost    | $21,417.00        | $12,570.00            | +$8,847.00 (-41.3%) |
| Total Daily Carbon Emissions  | 87.08 tCO2e       | 63.52 tCO2e           | -23.55 tCO2e (-27%) |
| Annualized Cost Benefit (350d)| $7,495,950/yr     | $4,399,500/yr         | +$3,096,450/yr      |
+-------------------------------+-------------------+-----------------------+---------------------+
```

The optimized schedule reduces daily utility and carbon costs by 41.3%, saving over $\$3.09\text{ million annually}$ and avoiding $8,242\text{ metric tons of }\text{CO}_2\text{e}$ per year purely through operational scheduling, without replacing production equipment.

---

## 7. Code, query, or calculation

The following self-contained Python script formulates and solves the multi-period demand response, intermediate buffer inventory, and carbon emissions optimization model presented in Section 6. It uses the `scipy.optimize.linprog` HiGHS solver.

### Code Implementation

```python
"""
Multi-period industrial demand response and carbon-aware scheduling LP.
Solves production, buffer storage, peak power, and emissions optimization.
"""
import numpy as np
from scipy.optimize import linprog

# 1. Horizon and market input vectors (24 hourly intervals)
T = 24
p_elec = np.array([32, 30, 28, 29, 35, 55, 95, 140, 160, 145, 110, 90,
                   85, 80, 85, 105, 150, 180, 165, 120, 85, 60, 45, 35], dtype=float)
ef_grid = np.array([0.32, 0.30, 0.28, 0.29, 0.35, 0.48, 0.62, 0.68, 0.70, 0.66,
                    0.55, 0.45, 0.42, 0.40, 0.43, 0.52, 0.67, 0.72, 0.69, 0.58,
                    0.46, 0.38, 0.34, 0.32], dtype=float)

# 2. Economic and technical parameters
carbon_tax = 60.0       # $/tCO2
demand_charge = 15.0    # $/MW peak demand fee
spec_power = 0.75       # MWh electricity per ton of upstream product
u_max = 22.0            # Max upstream production capacity (t/h)
s_min, s_max = 20.0, 120.0  # Intermediate buffer tank limits (tons)
s_init = 50.0           # Initial and target ending buffer level (tons)
d_downstream = np.full(T, 10.0)  # Downstream continuous demand (t/h)

# 3. Decision vector: x = [u_0..u_23, s_0..s_24, P_peak] (total: 2*T + 2 = 50)
c = np.zeros(2 * T + 2)
c[:T] = (p_elec + carbon_tax * ef_grid) * spec_power
c[-1] = demand_charge

# Bounds: u in [0, u_max], s in [s_min, s_max], P_peak >= 0
bounds = [(0.0, u_max)] * T + [(s_min, s_max)] * (T + 1) + [(0.0, None)]
bounds[T] = (s_init, s_init)      # s_0 fixed
bounds[2 * T] = (s_init, s_init)  # s_24 fixed to initial level

# 4. Equality constraints: s_{t+1} - s_t - u_t = -d_downstream_t
A_eq = np.zeros((T, 2 * T + 2))
b_eq = -d_downstream
for t in range(T):
    A_eq[t, t] = -1.0           # -u_t
    A_eq[t, T + t] = -1.0       # -s_t
    A_eq[t, T + t + 1] = 1.0    # +s_{t+1}

# 5. Inequality constraints: spec_power * u_t - P_peak <= 0
A_ub = np.zeros((T, 2 * T + 2))
b_ub = np.zeros(T)
for t in range(T):
    A_ub[t, t] = spec_power
    A_ub[t, -1] = -1.0

# 6. Solve LP via HiGHS interior-point / dual-simplex
res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")

if res.success:
    u_opt = res.x[:T]
    s_opt = res.x[T:2 * T + 1]
    p_peak = res.x[-1]
    tot_prod = np.sum(u_opt)
    tot_elec = np.sum(p_elec * spec_power * u_opt)
    tot_emiss = np.sum(ef_grid * spec_power * u_opt)
    tot_carb = carbon_tax * tot_emiss
    tot_peak = demand_charge * p_peak

    print(f"Optimal Status: {res.message}")
    print(f"Total Daily Cost: ${res.fun:,.2f}")
    print(f"  - Volumetric Power Cost : ${tot_elec:,.2f}")
    print(f"  - Carbon Emissions Cost : ${tot_carb:,.2f} ({tot_emiss:.2f} tCO2)")
    print(f"  - Capacity Demand Charge: ${tot_peak:,.2f} (Peak: {p_peak:.2f} MW)")
    print(f"Total Upstream Output     : {tot_prod:.1f} t (Target: {np.sum(d_downstream):.1f} t)")
    print(f"Off-peak (h00-04) avg rate: {np.mean(u_opt[:5]):.1f} t/h")
    print(f"Peak (h07-09,16-18) avg   : {np.mean(u_opt[[7,8,9,16,17,18]]):.1f} t/h")
```

### Execution Output and Validation

```text
Optimal Status: Optimization terminated successfully. (HiGHS Status 0: Optimization terminated successfully)
Total Daily Cost: $12,570.00
  - Volumetric Power Cost : $8,511.00
  - Carbon Emissions Cost : $3,811.50 (63.53 tCO2)
  - Capacity Demand Charge: $247.50 (Peak: 16.50 MW)
Total Upstream Output     : 240.0 t (Target: 240.0 t)
Off-peak (h00-04) avg rate: 22.0 t/h
Peak (h07-09,16-18) avg   : 0.0 t/h
```

---

## 8. Data, standards, and regulatory context

Process industry energy, water, and emissions accounting is governed by international standards, statutory environmental operating permits, and compliance trading systems.

### 8.1 Energy Management Systems: ISO 50001
ISO 50001:2018 establishes the international management standard for energy efficiency and energy management systems (EnMS):
- **Energy Baseline (EnB) and Energy Performance Indicators (EnPI):** Mandates that industrial facilities establish rigorous regression-based baselines linking energy consumption to production throughput, ambient weather variables (Heating/Cooling Degree Days), and campaign operating modes:
  $$\text{EnB}: \quad E_t = \beta_0 + \beta_1 \cdot \text{Throughput}_t + \beta_2 \cdot \Delta T_{\text{ambient},t} + \epsilon_t$$
- **Significant Energy Uses (SEUs):** Facilities must audit and isolate equipment accounting for $>80\%$ of plant energy consumption (e.g., steam crackers, electric induction furnaces, compressed air headers, cooling water circulation pumps).
- **Measurement and Verification (M&V):** Requires revenue-grade energy metering (ASME PTC 19.5 for steam; IEC 62053 for electricity meters) to verify efficiency gains rather than relying on design estimates.

### 8.2 Greenhouse Gas Protocol and Carbon Reporting Standards
- **GHG Protocol Corporate Accounting and Reporting Standard:** Categorizes industrial emissions into three scopes:
  - *Scope 1 (Direct):* Governed by EPA 40 CFR Part 98 (Mandatory Reporting of Greenhouse Gases) in the U.S. and national emissions inventories globally. Requires continuous emission monitoring systems (CEMS) or fuel gas chromatographic analysis.
  - *Scope 2 (Indirect):* Regulated under the GHG Protocol Scope 2 Guidance (2015). Companies must report under dual frameworks:
    1. *Location-based:* Reflecting the average grid emission factor of the regional grid where consumption occurs (e.g., eGRID subregion factors).
    2. *Market-based:* Reflecting contractual instruments (Energy Attribute Certificates, Guarantees of Origin, PPAs).
  - *Scope 3 (Value Chain):* Upstream purchased feedstocks (Category 1) and downstream processing of sold commodities (Category 10).
- **ISO 14064-1:** Specification with guidance at the organization level for quantification and reporting of greenhouse gas emissions and removals.

### 8.3 Emissions Trading Systems and Carbon Border Adjustments
- **EU Emissions Trading System (EU ETS - Directive 2003/87/EC):**
  - World’s first major carbon market, operating on a "cap-and-trade" principle covering $>10,000$ energy-intensive installations.
  - *Phase 4 (2021–2030):* Accelerates the linear reduction factor (LRF) of the overall emissions cap to $4.3\text{ to }4.4\%/\text{year}$. Free allocation to industrial sectors is based on strict product benchmarks (representing the average greenhouse gas performance of the top 10% most efficient installations in the EU).
  - Installations whose specific emissions exceed the benchmark must purchase EU Allowances (EUAs) at auction or on the secondary market (EEX, ICE).
- **EU Carbon Border Adjustment Mechanism (CBAM - Regulation 2023/956):**
  - Designed to prevent "carbon leakage" by equalizing the price of carbon paid by EU domestic producers with that of imported goods in six emissions-intensive sectors: iron and steel, aluminum, cement, fertilizers, hydrogen, and electricity.
  - Transition period began October 1, 2023 (mandatory reporting of embedded direct and indirect emissions); financial payments and CBAM certificate surrenders commence January 1, 2026.
  - Importers must verify the actual verified embedded emissions ($\text{tCO}_2\text{e/t}$ product) of overseas facilities or be subjected to default punitive administrative emissions values.
- **California Cap-and-Trade Program (AB 32 / SB 32):**
  - Administered by the California Air Resources Board (CARB), linked with Quebec under the Western Climate Initiative (WCI).
  - Sets annual compliance allowance obligations with price containment collars and reserve auctions.

### 8.4 Water Standards and Environmental Discharge Permits
- **ISO 14046:** Environmental management — Water footprint — Principles, requirements, and guidelines. Distinguishes water consumption (water evaporated, incorporated into product, or transferred to another basin) from water withdrawal (water diverted from a source).
- **U.S. Clean Water Act (CWA) and NPDES Permits:**
  - National Pollutant Discharge Elimination System (NPDES, 40 CFR Part 122). Facilities discharging industrial wastewater to surface waters must operate under permits specifying strict limits on:
    - Mass discharge ($\text{kg/day}$) and concentration ($\text{mg/L}$) of Total Suspended Solids (TSS), Biochemical Oxygen Demand ($\text{BOD}_5$), and Chemical Oxygen Demand (COD).
    - Heavy metal toxicity (e.g., hexavalent chromium, arsenic, lead).
    - Thermal effluent limits: Maximum allowable $\Delta T$ (typically $\le 2.8^\circ\text{C}$ rise) between cooling water outfall and receiving river to prevent thermal ecological disruption.
- **FERC Order 2222 (Demand Response Integration):**
  - Federal Energy Regulatory Commission order enabling distributed energy resources (DERs) and industrial load-response aggregators to compete directly in organized wholesale electricity markets (PJM, NYISO, ISO-NE, MISO, CAISO).

---

## 9. Industry practice and case evidence

### 9.1 Chemical Verbund Integration: BASF Ludwigshafen
The BASF Ludwigshafen complex in Germany is the world’s largest integrated chemical manufacturing site, spanning $10\text{ km}^2$ with approximately 200 chemical plants:
- **Verbund Steam and Energy Cascading:**
  - The site integrates an interconnected $600\text{ km}$ steam pipeline network across three pressure tiers ($100\text{ bar}$, $40\text{ bar}$, and $4\text{ bar}$).
  - Exothermic chemical synthesis reactors (e.g., acrylic acid, ammonia, ethylene oxide) produce waste heat that generates superheated steam on-site.
  - This recovered reaction steam is channeled directly into reboilers of adjacent endothermic distillation trains, satisfying $>50\%$ of the entire complex’s process steam requirements.
  - The Verbund energy concept saves approximately $1.6\text{ million metric tons of fuel oil equivalents annually}$, avoiding over $3.8\text{ million metric tons of }\text{CO}_2\text{ emissions per year}$.
- **Water Management and Rhine River Interaction:**
  - The site withdraws cooling water from the Rhine River. During severe European summer heatwaves (e.g., 2018, 2022), river water temperatures approached the regulatory thermal discharge limit of $28^\circ\text{C}$, while low river gauge levels threatened navigation and cooling availability.
  - BASF deployed automated real-time water pinch optimization and cooling tower recirculation cascades, prioritizing steam-driven chillers and shifting cooling loads between process loops to prevent forced plant shutdowns.

### 9.2 Aluminum Smelting Demand Response: Alcoa and Norsk Hydro
Aluminum smelting via the Hall-Héroult electrolytic reduction process is one of the most electricity-intensive industrial operations in the world, consuming approximately $13.5\text{ to }15.0\text{ MWh}$ of direct current electricity per metric ton of liquid aluminum:
- **The "Virtual Battery" Flexibility Mechanism:**
  - Traditionally, aluminum potlines were operated at rigid, continuous electrical current ($100\text{ to }400\text{ kA}$) 24/7/365. Thermal equilibrium in the pots must be preserved; if power is cut for longer than 3 to 4 hours, the molten cryolite bath freezes, causing catastrophic equipment destruction and hundreds of millions in relining costs.
  - Alcoa (Warrick, Indiana) and Norsk Hydro (Karmøy, Norway) engineered proprietary dynamic power modulation systems.
  - By adjusting anode-cathode distance and implementing high-frequency power electronics, potlines can modulate power consumption by $\pm 15\%\text{ to }25\%$ for durations of 1 to 3 hours without freezing the molten bath.
  - In MISO and Nord Pool markets, smelters sell fast-acting spinning reserves and frequency regulation services (Frequency Containment Reserve, FCR) to grid operators. When regional grid frequency dips or spot power prices spike above $\$1,000/\text{MWh}$, the smelter curtails hundreds of megawatts in seconds, earning tens of millions in grid ancillary service revenues that reduce net aluminum production cash costs.

### 9.3 Air Separation Units (ASU): Linde and Air Liquide Cryogenic Storage
Air separation plants compress atmospheric air to cryogenically separate liquid nitrogen ($\text{LN}_2$), liquid oxygen ($\text{LOX}$), and liquid argon ($\text{LAR}$):
- **Electricity Cost Dominance:** Electrical power consumed by heavy multi-stage centrifugal compressors accounts for $>70\%$ of variable production costs.
- **Demand Shifting Architecture in ERCOT and PJM:**
  - Companies design ASUs with intentional overcapacity ($130\text{ to }150\%$ of average pipeline gas sales) paired with large vacuum-insulated cryogenic liquid storage tanks.
  - Master production schedules link directly to day-ahead wholesale electricity prices.
  - During off-peak hours (e.g., high overnight West Texas wind generation with negative or near-zero LMP), ASUs ramp compressors to 100% capacity, liquefying atmospheric gases and filling cryogenic liquid tanks.
  - During peak summer afternoon hours (3 PM to 7 PM in ERCOT), compressors throttle back to minimum turn-down ($40\%$) or shut down entirely. Gaseous pipeline customers (such as petrochemical plants and refineries) are supplied continuously by vaporizing liquid from storage tanks using ambient air vaporizers.
  - This demand-response arbitrage slashes net electricity procurement costs by $30\text{ to }45\%$ while hedging exposure to ERCOT $4\text{CP}$ (Four Coincident Peak) transmission charges.

### 9.4 Steel Decarbonization: ArcelorMittal and SSAB HYBRIT
Primary steelmaking via the traditional Blast Furnace–Basic Oxygen Furnace (BF-BOF) route emits approximately $1.8\text{ to }2.2\text{ tCO}_2\text{e}$ per ton of crude steel:
- **Direct Reduced Iron (DRI) Hydrogen Transition:**
  - SSAB, LKAB, and Vattenfall established the HYBRIT project in Sweden, replacing metallurgical coking coal with green hydrogen in a direct reduction shaft furnace:
    $$\text{Fe}_2\text{O}_3 + 3\text{H}_2 \to 2\text{Fe} + 3\text{H}_2\text{O}$$
  - The direct reduced sponge iron is subsequently melted in an Electric Arc Furnace (EAF) powered by renewable electricity, reducing specific emissions to $<0.05\text{ tCO}_2\text{e/t steel}$.
  - *Operational Trade-off:* Producing the required green hydrogen via water electrolysis requires approximately $3.5\text{ MWh}$ of electricity per ton of steel. A commercial $2\text{ Mt/year}$ DRI steel mill requires an uninterruptible clean power supply of $\approx 1.0\text{ to }1.2\text{ GW}$—equivalent to the electrical demand of a major metropolitan city. Steel production planning becomes directly dependent on long-term renewable power purchase agreements (PPAs) and grid transmission access.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Physical / Economic Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|
| **Water Pinch Violation (Cross-Pinch Contamination)** | Freshwater intake fails to drop despite installing reuse piping; biological WWTP upsets. | Effluent water from above the pinch concentration ($C > C^*$) is recycled back to units operating below the pinch, increasing net freshwater demand. | Millions in piping CAPEX spent with zero net freshwater reduction; WWTP hydraulic overload. | Re-audit concentration profiles; enforce strict cross-pinch transfer prohibition in WAN transshipment optimization. |
| **Steam Header Exergy Destruction** | Constant high bypass flow through high-pressure PRVs ($F^{\text{PRV}} > 10\text{ t/h}$) while turbines are offline. | Superheated HP steam is throttled to MP/LP headers without generating electrical work, dissipating thermodynamic availability. | Massive loss of low-cost cogeneration electricity; elevated grid power purchase costs. | Overhaul backpressure steam turbine train; implement automated boiler master dispatch prioritizing turbine flow over PRVs. |
| **Boiler Cycling Fatigue** | Frequent low-load alarms; thermal stress cracks in boiler tubes; fluctuating steam header pressure. | Aggressive demand-response load shifting forces heavy fossil-fueled utility boilers to cycle beyond allowable turn-down and ramp limits. | Tube rupture, unscheduled boiler outages, catastrophic site-wide steam header collapse. | Set strict ramping rate limits ($\Delta F_{\text{boil}}/\Delta t$) and minimum uptime/downtime constraints in MILP scheduling. |
| **Capacity Demand Ratchet Trap** | Brief 15-minute electrical power surge during an unexpected unit restart coinciding with grid peak hours. | Facility sets a new 15-minute peak demand record during on-peak utility billing window, triggering a 12-month ratchet penalty. | Incurs $\$100,000\text{ to }\$500,000$ in non-negotiable monthly demand charges for an entire billing year. | Install automated hardware peak-shaving interlocks; program production scheduler with hard capacity demand ceiling constraints. |
| **Cooling Tower Thermal Discharge Inversion** | Basin temperature exceeding $32^\circ\text{C}$; cooling water supply temperature drifting upward. | High ambient wet-bulb temperatures reduce evaporative cooling efficiency while process heat rejection remains constant. | Loss of vacuum in surface condensers; reduced chemical reaction yields; potential NPDES thermal violation. | Integrate wet-bulb forecasts into production schedule; curtail non-essential reboiler duties; activate auxiliary mechanical chillers. |
| **Marginal Carbon Factor Inversion** | Facility claims "zero-carbon production" based on annual average grid factors during hours when gas peakers are firing. | Using static or annual average emissions factors ($EF_{\text{avg}}$) instead of hourly marginal factors ($EF_t^{\text{marginal}}$) for scheduling. | Real-world carbon emissions increase; audit failure under rigorous Scope 2 market-based verifications. | Ingest real-time marginal emissions feeds (e.g., PJM, CAISO, ENTSO-E) directly into the scheduling objective function. |
| **Intermediate Buffer Starvation** | Intermediate tank level dropping below minimum net positive suction head ($s_t < s^{\min}$). | Over-aggressive load shifting to capture off-peak electricity prices depletes buffer inventory faster than anticipated. | Forced emergency shutdown of continuous downstream reactor train; flaring of off-spec intermediate material. | Implement stochastic safety stock buffer margins; enforce hard buffer inventory lower bounds with penalty slack variables. |
| **Amine Scrubbing Energy Drain (CCUS)** | LP steam header pressure dropping below minimum reboiler saturation pressure ($P_{\text{LP}} < 2.5\text{ bar}$). | Post-combustion $\text{CO}_2$ amine stripper reboiler draws excessive LP steam ($2.8\text{ GJ/t CO}_2$), starving process distillation reboilers. | Refinery distillation towers drop off-spec; loss of production throughput exceeds value of carbon captured. | Size CHP system to supply incremental low-grade steam; integrate MVR or heat pumps to recover capture heat. |
| **Biological Wastewater Shock Loading** | WWTP aeration basin dissolved oxygen drops to zero; foaming; sludge bulking. | Production campaign transition releases a high-volume, concentrated slug of toxic or high-COD wash water into the reuse header. | Complete die-off of activated sludge biomass; total plant shutdown mandated by environmental authorities. | Install equalization buffer basins; deploy automated TOC/COD online analyzers with diversion valves to containment tanks. |
| **Demineralized Water Silica Breakthrough** | Conductivity spike on boiler feedwater inlet; silica concentration $>0.02\text{ ppm}$. | Ion exchange demineralizer resin bed exhaustion caused by increased recycling of high-conductivity condensate streams. | Catastrophic silica deposition and glassy scale formation on high-pressure steam turbine blades; turbine destruction. | Enforce continuous online conductivity and silica monitoring; restrict condensate recycle based on strict water quality assays. |
| **Hydrogen Embrittlement and Flame Velocity Flashback** | Burner flashback alarms; acoustic vibration in furnace plenum; pipeline micro-cracks. | Firing green hydrogen blends ($>20\%\text{ H}_2$ by volume) in conventional natural gas burners without modifying burner tips or metallurgy. | Furnace explosion risk; structural failure of piping; elevated $\text{NO}_x$ emissions violating air permits. | Retrofit low-$\text{NO}_x$ dual-fuel burner tips; upgrade pipeline metallurgy to austenitic stainless steel; install optical flame scanners. |
| **CBAM Benchmark Miscalculation** | Customs authorities reject import declarations; provisional border tariffs levied at maximum default punitive rates. | Manufacturing facility uses incomplete battery limits or unverified Scope 2 emissions factors in CBAM reporting dossiers. | Cash flow freeze; imported intermediate feedstocks stuck at port terminals; punitive financial penalties. | Conduct third-party accredited audits aligning operational telemetry with EU CBAM Annex III methodologies. |

---

## 11. Assumptions and limits

| Assumption | Why It Is Needed | What Breaks It in Practice | Diagnostic / Test to Detect Failure |
|---|---|---|---|
| **Linear Willans Turbine Power Lines** | Allows formulation of steam turbine power generation as a tractable Linear Program (LP). | Severe turbine turn-down ($<40\%$ rated load) induces non-linear throttling losses and stage efficiency collapse. | Compare predicted power generation against measured turbine electrical meter data across varying inlet steam flowrates. |
| **Single Limiting Contaminant in Water Pinch** | Permits graphical and simple numerical composite curve targeting using one primary species (e.g., COD). | Real-world effluents contain multiple non-correlated contaminants (e.g., COD, TDS, suspended solids, pH, heavy metals). | Check if pinch target for COD violates TDS limits in any process unit; transition to multi-contaminant LP superstructure. |
| **Constant Mass Transfer Coefficient** | Assumes water extraction capability is purely concentration-driven and independent of wash fluid velocity or temperature. | Viscosity changes, fouling scale, and temperature swings alter mass transfer coefficients ($\mathcal{K}_L a$) significantly. | Monitor contaminant extraction efficiency across heat exchanger cleaning cycles and seasonal ambient temperature shifts. |
| **Perfect Day-Ahead Electricity Price Foresight** | Enables deterministic 24-hour MILP production scheduling against fixed hourly power prices. | Real-time market (RTM) price spikes occur due to sudden generator trips or transmission line contingencies. | Calculate variance between DAM and RTM prices; implement two-stage stochastic programming with rolling horizons. |
| **Zero Delay in Intermediate Buffer Mixing** | Assumes material entering buffer tank is immediately mixed, homogenized, and available for downstream withdrawal. | Plug-flow behavior, thermal stratification, or settling of suspended solids creates concentration gradients inside buffer vessels. | Perform tracer residence time distribution (RTD) tests; install internal tank agitators or jet mixing loops. |
| **Unconstrained Cooling Water Heat Sinks** | Assumes the environment or cooling towers can absorb arbitrary process latent heat without penalizing cooling water supply temperature. | Extreme summer ambient wet-bulb temperatures cap cooling tower heat rejection capacity. | Monitor cooling tower approach-to-wet-bulb temperature ($T_{\text{cold}} - T_{\text{wet-bulb}}$); if $>5^\circ\text{C}$, heat sink is constrained. |
| **Fixed Scope 2 Grid Emission Factors** | Simplifies corporate greenhouse gas reporting using a constant average regional grid emission factor. | Power grid dispatch changes continuously; actual marginal balancing units vary from zero-carbon hydro to heavy coal/gas peakers. | Cross-reference hourly production logs with time-synchronized marginal emissions data from independent system operators. |
| **Isenthalpic PRV Operation** | Treats steam throttling across valves as pure adiabatic expansion without mass loss or kinetic energy dissipation. | Water droplet erosion, valve packing leaks, or acoustic energy dissipation alter downstream enthalpy. | Measure downstream steam temperature and pressure post-PRV; calculate enthalpy balance using NIST steam tables. |
| **Negligible Condensate Return Contamination** | Assumes steam condensate recovered from process reboilers is pure enough to return directly to high-pressure boilers. | Reboiler tube leaks allow hydrocarbons or acidic process fluids to infiltrate the condensate return header. | Install continuous TOC, conductivity, and pH monitors on all condensate return headers with automatic dump valves. |
| **Constant Abatement Capital Costs (LCCA)** | Assumes CAPEX and technological performance of decarbonization assets (heat pumps, electrolyzers, CCS) remain static. | Rapid technological learning curves, supply chain bottlenecks, and grid interconnection delays alter levelized economics. | Perform sensitivity analysis across $\pm 30\%$ CAPEX and discount rate variations; track clean-tech equipment market indexes. |

---

## 12. Exercises

### Exercise 1: Water Pinch Targeting
A chemical processing plant operates two continuous washing operations:
- Operation A: Removes $\Delta m_A = 60.0\text{ kg/h}$ of organic contaminant. Limiting inlet concentration $C_{\text{in},A}^{\max} = 0\text{ ppm}$, outlet concentration $C_{\text{out},A}^{\max} = 600\text{ ppm}$.
- Operation B: Removes $\Delta m_B = 180.0\text{ kg/h}$ of organic contaminant. Limiting inlet concentration $C_{\text{in},B}^{\max} = 200\text{ ppm}$, outlet concentration $C_{\text{out},B}^{\max} = 800\text{ ppm}$.

**Deliverable:**
1. Compute the baseline once-through freshwater flowrate ($\text{t/h}$).
2. Construct the Limiting Water Profile intervals, cumulative mass loads, and determine the Water Pinch concentration ($C^*$).
3. Calculate the minimum freshwater intake target ($F_{\text{FW}}^{\min}$) and percentage freshwater reduction.
- Acceptance criteria: Exact mathematical calculations shown step-by-step; units explicitly stated.
- Estimated time: 30 minutes.

### Exercise 2: Combined Heat and Power Steam Turbine Mass and Enthalpy Balance
A manufacturing facility’s boiler produces $100.0\text{ t/h}$ of superheated HP steam at $60\text{ bar}$, $450^\circ\text{C}$ ($h_{\text{HP}} = 3,300\text{ kJ/kg}$). The plant process requires $100.0\text{ t/h}$ of LP steam at $4\text{ bar}$, $160^\circ\text{C}$ ($h_{\text{LP}} = 2,775\text{ kJ/kg}$).
1. Calculate the electrical power generation ($\text{MW}$) if all steam is expanded through a backpressure turbine with generator efficiency $\eta_{\text{gen}} = 0.95$.
2. If wholesale electricity is valued at $\$90.00/\text{MWh}$ and the plant runs $8,000\text{ hours/year}$, calculate the annual value of cogenerated electricity compared to throttling steam across an isenthalpic PRV.
- Acceptance criteria: Exact enthalpy drop and power conversions; state annual financial savings.
- Estimated time: 20 minutes.

### Exercise 3: Industrial Demand Response Peak Shaving Formulation
A mechanical grinding mill consumes $\beta = 0.50\text{ MWh}$ of electricity per ton of throughput. The unit has a maximum capacity of $50\text{ t/h}$ and feeds an intermediate slurry silo ($s^{\min} = 50\text{ t}$, $s^{\max} = 400\text{ t}$, $s_0 = 150\text{ t}$). The downstream chemical reactor draws a steady $25\text{ t/h}$.
Over an 8-hour shift, power prices are:
- Hours 1–4 (Off-peak): $\$40.00/\text{MWh}$.
- Hours 5–6 (Peak): $\$250.00/\text{MWh}$.
- Hours 7–8 (Mid-peak): $\$80.00/\text{MWh}$.

**Deliverable:**
Determine the optimal hourly production schedule $u_t$ ($t=1\dots 8$) that minimizes total electricity procurement costs while keeping the downstream reactor operating continuously without violating silo bounds. Calculate total electricity cost.
- Acceptance criteria: Tabulate hourly production, power draw, buffer inventory, and total financial cost.
- Estimated time: 35 minutes.

### Exercise 4: Carbon Emissions Penalty in Production Scheduling
A facility operates two alternative production lines to produce an industrial intermediate:
- Line A (Older steam-driven train): Variable cash operating cost $\$120.00/\text{t}$, direct Scope 1 natural gas emissions $0.80\text{ tCO}_2\text{e/t}$.
- Line B (Modern electrified train): Variable cash operating cost $\$150.00/\text{t}$, consumes $0.60\text{ MWh/t}$ of grid electricity.

The regional electricity grid operates with a marginal emission factor of $\text{EF}_{\text{grid}} = 0.50\text{ tCO}_2\text{e/MWh}$.
1. At what carbon price ($\lambda_{\text{CO2}}^*$, in $\$ /\text{tCO}_2\text{e}$) does the total marginal operating cost of electrified Line B become equal to that of steam-driven Line A?
2. If the carbon price is $\$90.00/\text{tCO}_2\text{e}$, which line should the master scheduler dispatch as base load?
- Acceptance criteria: Write out explicit cost equations; solve for breakeven carbon price; state dispatch recommendation.
- Estimated time: 20 minutes.

### Exercise 5: Industrial Heat Pump COP and Fuel Savings
An industrial distillation train rejects $5.0\text{ MW}$ of thermal waste heat into cooling water at $65^\circ\text{C}$ ($T_{\text{evap}}$). An engineering team proposes installing an Industrial Heat Pump to upgrade this waste heat to produce $125^\circ\text{C}$ LP steam ($T_{\text{cond}}$).
1. Calculate the theoretical Carnot Coefficient of Performance ($\text{COP}_{\text{Carnot}}$).
2. Assuming a Second-Law efficiency factor of $\eta_{\text{II}} = 0.50$, calculate the actual expected $\text{COP}_{\text{real}}$.
3. If the heat pump consumes $1.5\text{ MW}$ of electric power, how much thermal heat ($Q_{\text{heat}}$, in $\text{MW}$) is delivered to the steam header?
- Acceptance criteria: Convert temperatures to Kelvin; state COP and delivered heat accurately.
- Estimated time: 20 minutes.

### Exercise 6: Levelized Cost of Carbon Abatement (LCCA) Calculation
A chemical site evaluates retrofitting an existing fossil-gas boiler with a zero-carbon electric boiler:
- Capital expenditure: $\Delta \text{CAPEX} = \$4,000,000$.
- Asset operating lifetime: $N = 15\text{ years}$. Cost of capital: $r = 8.0\%$.
- Incremental annual operating cost (electricity vs. gas): $\Delta \text{OPEX} = \$500,000/\text{year}$.
- Net annual greenhouse gas emissions avoided: $12,000\text{ tCO}_2\text{e/year}$.

**Deliverable:**
Calculate the Capital Recovery Factor (CRF), annualized capital expenditure, and the Levelized Cost of Carbon Abatement ($\$ /\text{tCO}_2\text{e}$). Is this project viable under an expected EU ETS allowance price of $€85/\text{tCO}_2\text{e}$ (assume $\$1.08/€$ exchange rate)?
- Acceptance criteria: Show all formulas; provide a clear investment recommendation.
- Estimated time: 25 minutes.

### Exercise 7: Critique: Flawed Demand Response Scheduling Policy
Read the following memo from a plant logistics manager:
*"To eliminate electrical demand charges, we have implemented a policy requiring our chlorine electrolysis plant to immediately shut down whenever wholesale power prices exceed $\$100/\text{MWh}$. Downstream PVC synthesis units will run on residual piping pressure until prices normalize. We do not need intermediate storage tanks because power price spikes never last more than two hours."*

**Deliverable:**
Write a 300-word engineering and operational critique identifying at least three fatal flaws in this policy.
- Acceptance criteria: Reference downstream asset safety, pressure decay, thermal cycling, and supply chain commitments.
- Estimated time: 30 minutes.

### Exercise 8: Design: Resilient Net-Zero Utility Microgrid Architecture
A petrochemical complex in a drought-prone coastal region is planning a brownfield expansion. The site requires $50\text{ MW}$ of firm power, $150\text{ t/h}$ of combined steam, and consumes $500\text{ t/h}$ of water. It faces strict Zero Liquid Discharge (ZLD) regulations and must comply with EU CBAM export standards.

**Deliverable:**
Design an integrated conceptual utility and scheduling architecture that combines:
1. Water Pinch reuse network with mechanical vapor recompression brine crystallization.
2. Dual-fuel (hydrogen/gas) CHP steam generation with industrial heat pumps.
3. Intermediate product buffer storage enabling 4-hour demand-response peak power shaving.
- Acceptance criteria: Provide a structured block diagram specification, identify operational control linkages, and state performance metrics.
- Estimated time: 45 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Water Pinch Targeting
1. **Baseline Once-Through Water Flow:**
   - Operation A: $F_A = \Delta m_A / (C_{\text{out},A}^{\max} - C_{\text{in},A}^{\max}) \times 10^3 = 60.0 / (600 - 0) \times 10^3 = 100.0\text{ t/h}$.
   - Operation B: $F_B = \Delta m_B / (C_{\text{out},B}^{\max} - C_{\text{in},B}^{\max}) \times 10^3 = 180.0 / (800 - 200) \times 10^3 = 180.0 / 600 \times 10^3 = 300.0\text{ t/h}$.
   - Total Baseline Freshwater = $100.0 + 300.0 = 400.0\text{ t/h}$.
2. **Interval Calculation and Pinch Point:**
   - Sorted concentration boundaries: $\{0, 200, 600, 800\}\text{ ppm}$.
   - *Interval 1 ($0\text{ to }200\text{ ppm}$):* Operation A active ($F = 100.0\text{ t/h}$).
     $$\Delta M_1 = 100.0\text{ t/h} \times 200\text{ ppm} \times 10^{-3} = 20.0\text{ kg/h}. \quad M(200) = 20.0\text{ kg/h}.$$
   - *Interval 2 ($200\text{ to }600\text{ ppm}$):* Operations A and B active ($F = 100.0 + 300.0 = 400.0\text{ t/h}$).
     $$\Delta M_2 = 400.0\text{ t/h} \times 400\text{ ppm} \times 10^{-3} = 160.0\text{ kg/h}. \quad M(600) = 20.0 + 160.0 = 180.0\text{ kg/h}.$$
   - *Interval 3 ($600\text{ to }800\text{ ppm}$):* Operation B active ($F = 300.0\text{ t/h}$).
     $$\Delta M_3 = 300.0\text{ t/h} \times 200\text{ ppm} \times 10^{-3} = 60.0\text{ kg/h}. \quad M(800) = 180.0 + 60.0 = 240.0\text{ kg/h}.$$
3. **Freshwater Flowrate Targeting at Breakpoints:**
   - At $C = 200\text{ ppm}$: $F = 20.0 / (200 \times 10^{-3}) = 100.0\text{ t/h}$.
   - At $C = 600\text{ ppm}$: $F = 180.0 / (600 \times 10^{-3}) = 300.0\text{ t/h}$ $\leftarrow$ **PINCH POINT ($C^* = 600\text{ ppm}$)**.
   - At $C = 800\text{ ppm}$: $F = 240.0 / (800 \times 10^{-3}) = 300.0\text{ t/h}$.
   - **Minimum Freshwater Target ($F_{\text{FW}}^{\min}$):** $300.0\text{ t/h}$.
   - **Volumetric Reduction:** $(400.0 - 300.0) / 400.0 = 25.0\%$ reduction ($100.0\text{ t/h}$ saved).

---

### Answer to Exercise 2: CHP Steam Turbine Balance
1. **Power Generation:**
   - Mass flow: $100.0\text{ t/h} = 100,000\text{ kg/h} / 3,600\text{ s/h} = 27.778\text{ kg/s}$.
   - Enthalpy drop: $\Delta h = 3,300 - 2,775 = 525\text{ kJ/kg}$.
   - Electrical Power:
     $$W = \dot{m} \cdot \Delta h \cdot \eta_{\text{gen}} = 27.778\text{ kg/s} \times 525\text{ kJ/kg} \times 0.95 = 13,854.2\text{ kW} \approx 13.854\text{ MW}.$$
2. **Annual Financial Value:**
   - Hourly Value: $13.854\text{ MW} \times \$90.00/\text{MWh} = \$1,246.88/\text{h}$.
   - Annual Value ($8,000\text{ h/yr}$):
     $$\$1,246.88/\text{h} \times 8,000\text{ h/yr} = \$9,975,000/\text{year}.$$
   Throttling through a PRV destroys this entire $\$9.975\text{ million}$ economic value annually.

---

### Answer to Exercise 3: Demand Response Peak Shaving
- Total 8-hour downstream demand: $8\text{ h} \times 25\text{ t/h} = 200\text{ t}$.
- Peak hours 5–6 have an exorbitant price of $\$250/\text{MWh}$. The grinder should shut down completely ($u_5 = 0, u_6 = 0$).
- Total required production across non-peak hours (Hours 1–4 and 7–8):
  With initial buffer $s_0 = 150\text{ t}$, during hours 5–6, downstream demand drains $2 \times 25 = 50\text{ t}$. Buffer level must remain $\ge 50\text{ t}$ (satisfied: $150 - 50 = 100\text{ t} \ge 50\text{ t}$).
- Cheapest hours are Hours 1–4 ($\$40/\text{MWh}$). Maximize production to max capacity:
  $u_1 = 50\text{ t/h}, u_2 = 50\text{ t/h}, u_3 = 50\text{ t/h}, u_4 = 50\text{ t/h}$ (Total = $200\text{ t}$).
- Buffer tracking:
  - End of h1: $150 + 50 - 25 = 175\text{ t}$.
  - End of h2: $175 + 50 - 25 = 200\text{ t}$.
  - End of h3: $200 + 50 - 25 = 225\text{ t}$.
  - End of h4: $225 + 50 - 25 = 250\text{ t} \le 400\text{ t}$ (feasible).
  - End of h5: $250 + 0 - 25 = 225\text{ t}$.
  - End of h6: $225 + 0 - 25 = 200\text{ t}$.
- Hours 7–8 ($\$80/\text{MWh}$): Run at $u_7 = 0\text{ t/h}, u_8 = 0\text{ t/h}$, ending inventory = $200 - 50 = 150\text{ t}$ (matches $s_0$).
- **Total Electricity Cost:**
  - Hours 1–4 power draw: $50\text{ t/h} \times 0.50\text{ MWh/t} = 25\text{ MW}$.
  - Cost = $4\text{ hours} \times 25\text{ MW} \times \$40.00/\text{MWh} = \$4,000.00$.
  (Flat schedule running $25\text{ t/h}$ would cost $\$7,500.00$. Demand response saves $\$3,500.00$, or $46.7\%$).

---

### Answer to Exercise 4: Carbon Emissions Cost Inversion
1. **Cost Equivalence Formulation:**
   - Total Cost Line A: $\text{TC}_A = \$120.00 + \lambda_{\text{CO2}} \cdot 0.80$.
   - Total Cost Line B: $\text{TC}_B = \$150.00 + \lambda_{\text{CO2}} \cdot (0.60\text{ MWh/t} \times 0.50\text{ tCO}_2/\text{MWh}) = \$150.00 + \lambda_{\text{CO2}} \cdot 0.30$.
   - Set $\text{TC}_A = \text{TC}_B$:
     $$120.00 + 0.80 \cdot \lambda_{\text{CO2}}^* = 150.00 + 0.30 \cdot \lambda_{\text{CO2}}^* \implies 0.50 \cdot \lambda_{\text{CO2}}^* = 30.00 \implies \lambda_{\text{CO2}}^* = \$60.00/\text{tCO}_2\text{e}.$$
2. **Dispatch Recommendation at $\$90.00/\text{tCO}_2\text{e}$:**
   - $\text{TC}_A = 120 + 90 \times 0.80 = 120 + 72 = \$192.00/\text{t}$.
   - $\text{TC}_B = 150 + 90 \times 0.30 = 150 + 27 = \$177.00/\text{t}$.
   - Modern electrified Line B is $\$15.00/\text{t}$ cheaper. Master scheduler should dispatch **Line B as base load**.

---

### Answer to Exercise 5: Industrial Heat Pump COP
1. **Carnot COP:**
   - $T_{\text{evap}} = 65 + 273.15 = 338.15\text{ K}$.
   - $T_{\text{cond}} = 125 + 273.15 = 398.15\text{ K}$.
   - $\Delta T = 398.15 - 338.15 = 60.0\text{ K}$.
   - $\text{COP}_{\text{Carnot}} = T_{\text{cond}} / \Delta T = 398.15 / 60.0 = 6.636$.
2. **Real COP:**
   - $\text{COP}_{\text{real}} = \eta_{\text{II}} \cdot \text{COP}_{\text{Carnot}} = 0.50 \times 6.636 = 3.318$.
3. **Delivered Thermal Heat:**
   - $Q_{\text{heat}} = W_{\text{electric}} \cdot \text{COP}_{\text{real}} = 1.50\text{ MW} \times 3.318 = 4.977\text{ MW}$ (thermal delivered to LP steam header).

---

### Answer to Exercise 6: LCCA Calculation
1. **Capital Recovery Factor (CRF):**
   $$\text{CRF} = \frac{0.08(1.08)^{15}}{(1.08)^{15} - 1} = \frac{0.08 \times 3.17217}{2.17217} = \frac{0.25377}{2.17217} \approx 0.11683\text{ per year}.$$
2. **Annualized Capital Expenditure:**
   $$\text{AnnCAPEX} = \$4,000,000 \times 0.11683 = \$467,320/\text{year}.$$
3. **Total Annualized Cost:**
   $$\text{Total Cost} = \$467,320 + \$500,000 = \$967,320/\text{year}.$$
4. **Levelized Cost of Carbon Abatement (LCCA):**
   $$\text{LCCA} = \frac{\$967,320}{12,000\text{ tCO}_2\text{e}} = \$80.61/\text{tCO}_2\text{e}.$$
5. **Economic Viability:**
   - Carbon allowance price: $€85.00/\text{t} \times 1.08\text{ \$/€} = \$91.80/\text{tCO}_2\text{e}$.
   - Since $\text{LCCA} = \$80.61 < \$91.80$, the electric boiler retrofit is **economically viable** and will generate positive net savings over its operating life.

---

### Answer to Exercise 7: Critique Rubric
A strong critique must highlight:
1. *Safety and Process Reliability:* Operating downstream continuous PVC synthesis units on "residual piping pressure" is hazardous. Pressure collapse induces catastrophic unreacted monomer accumulation, polymerization runaway, or vacuum collapse of vessels.
2. *Thermal and Mechanical Asset Degradation:* Abruptly cutting power to chlor-alkali membrane electrolysis cells causes membrane degradation, pinhole tears, and reverse current corrosion, destroying multi-million dollar electrolyzer cell stacks.
3. *Commercial and Supply Chain Default:* Failing to deliver continuous product to downstream contracted customers triggers contractual non-performance penalties and supply chain disruptions.
4. *Demand Ratchet Ineffectiveness:* Wholesale energy prices ($\$/\text{MWh}$) are conflated with 15-minute capacity demand charges ($\$/\text{kW}$). Shutting down after prices spike does not prevent establishing a peak demand record earlier in the month.

---

### Answer to Exercise 8: Microgrid Design Rubric
A complete design submission must address:
- *Mass & Energy Integration:* Water Pinch network with multi-stage reverse osmosis and MVR brine crystallizer to achieve Zero Liquid Discharge while recycling distillate to boiler feedwater.
- *Utility Decoupling:* Dual-fuel gas/hydrogen CHP system providing firm base-load steam, supplemented by industrial heat pumps recovering low-grade heat ($60\text{ to }80^\circ\text{C}$) to supply LP steam.
- *Buffer Sizing:* Sizing intermediate liquid/slurry storage vessels to provide at least 4 hours of autonomous continuous downstream production during peak wholesale grid power intervals.
- *Carbon & Regulatory Tracking:* Integration of automated revenue-grade telemetry linking boiler fuel chromatographs, turbine meters, and grid marginal emissions feeds to comply with EU CBAM embedded emissions reporting.

---

## 14. Interview questions

### Question 1: Water Pinch vs. Heat Pinch
**Q:** "How does Water Pinch Analysis differ thermodynamically and mathematically from Thermal Heat Pinch Analysis?"
- **Strong Answer:** "While both rely on the pinch principle pioneered by Linnhoff, Heat Pinch integrates enthalpy cascades where temperature ($T$) is the driving force and energy is conserved. Water Pinch integrates mass exchange where contaminant concentration ($C$) is the driving force. Heat Pinch is constrained by a minimum approach temperature ($\Delta T_{\min}$), whereas Water Pinch can theoretically touch the operating line at $C^*$. Additionally, water networks can be non-conservative in mass if water is consumed in reactions or lost to evaporation, and industrial water systems frequently involve multi-component contaminants (COD, TDS, hardness) requiring multi-dimensional transshipment LP formulations rather than single-species composite curves."
- **Weak Answer:** "They are basically the same thing; you just replace temperature with water flow and draw the curves."

### Question 2: Demand Charges in Production Planning
**Q:** "A plant manager states: 'Our real-time electricity rate is currently $\$25/\text{MWh}$, which is below our average cost of $\$60/\text{MWh}$, so I ramped up our 20 MW auxiliary compressor.' What fatal error might this manager have committed?"
- **Strong Answer:** "The manager evaluated volumetric energy prices while ignoring the capacity demand charge ratchet. In many industrial tariffs, setting a new 15-minute peak demand during peak monthly hours incurs a demand charge of $\$15\text{ to }\$30/\text{kW-month}$. Ramping an un-scheduled 20 MW load can set a new site peak demand, triggering an instantaneous billing charge of $\$300,000\text{ to }\$600,000$ that may persist under a 12-month ratchet clause. The volumetric energy savings of a few hundred dollars are completely wiped out by the capacity penalty."
- **Weak Answer:** "The compressor might break or run out of oil if it ramps too fast."

### Question 3: Marginal vs. Average Grid Emissions
**Q:** "When optimizing a weekly production schedule to minimize carbon emissions, why is it incorrect to use the national average grid carbon intensity?"
- **Strong Answer:** "The average grid carbon intensity reflects baseload generation (nuclear, run-of-river hydro, base wind) that is already running and cannot respond to incremental industrial load changes. When a plant modulates its power demand by $\pm 10\text{ MW}$ in response to a schedule, the grid operator dispatches or curtails the *marginal* balancing generator—typically a natural gas turbine or coal plant. Scheduling against average emissions causes severe misallocation: shifting load into hours with low average emissions might actually trigger the startup of an inefficient peaker plant with very high marginal emissions."
- **Weak Answer:** "Average emissions are fine because that's what corporate sustainability reports publish in the annual report."

### Question 4: CHP Turbine vs. PRV Economics
**Q:** "Under what operational conditions would an industrial plant intentionally bypass a backpressure steam turbine and throttle steam through a pressure reduction valve?"
- **Strong Answer:** "Only under severe operational transients or market extremes: (1) Turbine mechanical tripping or unscheduled maintenance where process steam headers must be kept pressurized to prevent plant shutdown; (2) Extreme negative wholesale power pricing where the negative revenue from generating cogenerated electricity exceeds the fuel cost penalty of PRV throttling; (3) Rapid process load surges that exceed the maximum ramp rate ($\text{t/h/min}$) of the steam turbine governor; or (4) When steam demand is below the minimum technical turn-down limit of the turbine."
- **Weak Answer:** "When the operator wants to cool down the steam quickly."

### Question 5: Decarbonizing High-Temperature Heat
**Q:** "Why can industrial heat pumps easily replace fossil boilers for district heating and washing, but cannot decarbonize ethylene cracking or blast furnaces?"
- **Strong Answer:** "Thermodynamic temperature lift limits. Commercial industrial heat pumps can deliver temperatures up to $120\text{ to }160^\circ\text{C}$ with reasonable COPs ($2.5\text{ to }4.0$). Beyond $160^\circ\text{C}$, working fluid decomposition, compressor pressure ratios, and thermodynamic Carnot limitations cause the COP to approach $1.0$, eliminating the efficiency advantage over direct electrification. Ethylene cracking ($800\text{ to }850^\circ\text{C}$) and blast furnaces ($>1,400^\circ\text{C}$) require high-temperature reaction heat that can only be supplied by combustion of zero-carbon fuels (hydrogen, biomethane), electric induction/plasma, or direct electric resistance cracking coils."
- **Weak Answer:** "Because heat pumps are too small and factories don't have enough space for them."

### Question 6: CBAM Financial Exposure
**Q:** "How does the EU CBAM affect an Asian steel or chemical manufacturer exporting to Rotterdam, and what operational data must their MES/ERP capture?"
- **Strong Answer:** "CBAM requires EU importers to purchase CBAM certificates equal to the carbon price difference between the EU ETS and any carbon tax paid in the country of origin. The overseas exporter must provide independently audited, installation-specific data covering direct Scope 1 emissions (fuels, calcination) and embedded Scope 2 emissions (electricity consumed during smelting/synthesis), along with precursor materials. The plant's MES/ERP must trace energy and mass consumption down to specific production heats or batches; otherwise, the EU will apply default punitive benchmark values (typically set at the average emissions of the worst 10% EU installations), rendering the exporter uncompetitive."
- **Weak Answer:** "CBAM is just an import tax that the logistics shipping company pays at customs."

### Question 7: Buffer Tank Sizing for Demand Response
**Q:** "In designing an industrial demand-response system, what mathematical trade-off determines the optimal capacity of the intermediate buffer storage tank?"
- **Strong Answer:** "It is a trade-off between capital expenditure (CAPEX of tankage, civil foundations, inventory working capital, chemical degradation/settling risk) and operating expenditure (OPEX flexibility). A larger buffer allows longer turn-down during peak electricity pricing and higher peak shaving capabilities. The optimal volume occurs where the marginal annualized cost of additional tank volume ($r \cdot \text{CAPEX} + \text{CarryingCost}$) equals the marginal expected annual savings from power price arbitrage and capacity demand ratchet reduction."
- **Weak Answer:** "Make the tank as big as possible so the plant never runs out of stock."

### Question 8: Amine Carbon Capture Parasitic Load
**Q:** "Why does retrofitting post-combustion amine carbon capture to an industrial plant drastically reduce its net marketable product output?"
- **Strong Answer:** "Amine carbon capture involves a massive parasitic thermal and electrical load. Desorbing $\text{CO}_2$ from the solvent in the stripper column requires $2.5\text{ to }3.5\text{ GJ}$ of thermal energy per metric ton of $\text{CO}_2$ captured, typically supplied as low-pressure steam ($3\text{ to }5\text{ bar}$). Diverting this steam from process reboilers reduces plant processing capacity. Additionally, compressing captured gaseous $\text{CO}_2$ to supercritical pipeline conditions ($100\text{ to }150\text{ bar}$) consumes $90\text{ to }120\text{ kWh}$ of electricity per ton, consuming a significant fraction of site power."
- **Weak Answer:** "The carbon capture unit uses filters that get clogged with soot, slowing down the draft."

---

## 15. Key terms

- **Backpressure Turbine:** Steam turbine expanding high-pressure steam to positive gauge exhaust pressure to supply heating headers while cogenerating electricity.
- **CBAM (Carbon Border Adjustment Mechanism):** EU tariff equalizing carbon compliance costs on imported carbon-intensive commodities.
- **CHP (Combined Heat and Power):** Simultaneous generation of electricity and useful thermal energy from a single primary energy source.
- **Clean Air Act Title V:** Major source industrial air operating permit program establishing legally binding emissions caps.
- **Clean Water Act NPDES:** U.S. statutory framework regulating point-source discharge of pollutants and thermal effluent into surface waters.
- **Coefficient of Performance (COP):** Ratio of useful thermal heating delivered to work input in an industrial heat pump.
- **Demand Charge:** Monthly electrical tariff fee based on the peak 15-minute power draw recorded during on-peak intervals.
- **Demand Response (DR):** Intentional curtailment or shifting of industrial power demand to capitalize on dynamic power tariffs.
- **EU ETS (Emissions Trading System):** European cap-and-trade market establishing financial allowance obligations for industrial greenhouse gases.
- **Exergy:** The maximum theoretical useful work obtainable from a thermal system as it comes to thermodynamic equilibrium with the environment.
- **FERC Order 2222:** Federal order enabling industrial distributed energy resources and load curtailment to participate in wholesale electricity markets.
- **GHG Protocol:** Standardized global framework for measuring and reporting corporate Scope 1, Scope 2, and Scope 3 greenhouse gas emissions.
- **Industrial Heat Pump (IHP):** Electrically driven thermodynamic vapor compression system upgrading waste heat to process steam.
- **ISO 14046:** International standard establishing principles and requirements for organizational water footprinting.
- **ISO 50001:** International standard specifying requirements for establishing and maintaining an industrial Energy Management System.
- **Levelized Cost of Carbon Abatement (LCCA):** Net present annualized cost per metric ton of lifecycle greenhouse gas emissions avoided.
- **Limiting Composite Curve:** Profile of cumulative contaminant mass load versus concentration representing aggregate plant water demand.
- **Limiting Water Profile:** Maximum permissible inlet and outlet contaminant concentrations and required flowrates for a water-using unit.
- **Locational Marginal Price (LMP):** Wholesale electrical price reflecting the marginal cost of energy, transmission congestion, and losses.
- **Marginal Emissions Factor (MEF):** The carbon intensity of the specific marginal power generator dispatched to satisfy the next increment of load.
- **Mechanical Vapor Recompression (MVR):** Open-cycle heat pump compressing low-pressure process vapor to recover latent condensation heat.
- **NPDES Thermal Discharge Limit:** Regulatory cap on cooling water outfall temperature to protect receiving aquatic ecosystems.
- **Pinch Concentration ($C^*$):** The bottleneck concentration where the freshwater supply line touches the limiting water composite curve.
- **Pressure Reduction Valve (PRV):** Isenthalpic throttling valve reducing steam pressure without generating mechanical power.
- **Scope 1 Emissions:** Direct greenhouse gas emissions from owned or controlled industrial facilities (combustion, chemical calcination).
- **Scope 2 (Location-Based):** Indirect emissions from purchased energy calculated using regional grid average carbon intensity.
- **Scope 2 (Market-Based):** Indirect emissions from purchased energy calculated using contractual supplier-specific attributes (PPAs, RECs).
- **Specific Energy Consumption (SEC):** Ratio of total net energy consumed per unit of finished product manufactured.
- **Specific Water Consumption (SWC):** Ratio of total net freshwater withdrawn per unit of finished product manufactured.
- **Water Allocation Network (WAN):** Physical topological piping network routing freshwater and recycle streams across unit operations.
- **Water Pinch Analysis:** Systematic methodology based on mass exchange integration to target minimum freshwater and wastewater flows.
- **Willans Line:** Empirical linear relationship between steam turbine mass flowrate and electrical power output.
- **Zero Liquid Discharge (ZLD):** Advanced wastewater treatment layout eliminating liquid discharge through total internal water purification and recycling.

---

## 16. Sources

- **American Society of Mechanical Engineers (ASME).** (2014). *ASME PTC 6: Steam Turbines Performance Test Codes*. ASME, New York.
- **Castro, P. M., Teles, J. P., & Novais, A. Q.** (2009). *Water allocation networks: Single versus multiple contaminant models*. Industrial & Engineering Chemistry Research, 48(4), 1952–1964. https://doi.org/10.1021/ie801046x
- **European Parliament and Council.** (2003). *Directive 2003/87/EC establishing a scheme for greenhouse gas emission allowance trading within the Community*. Official Journal of the European Union, L 275, 32–46.
- **European Parliament and Council.** (2023). *Regulation (EU) 2023/956 establishing a carbon border adjustment mechanism*. Official Journal of the European Union, L 130, 52–104.
- **Grossmann, I. E., & Martin, M.** (2010). *Energy and water optimization in process systems*. In *Encyclopedia of Chemical Processing* (pp. 1–18). Taylor & Francis, London.
- **International Energy Agency (IEA).** (2021). *Industrial Heat Pumps: Technology and Applications*. IEA Technology Collaboration Programme on Heat Pumping Technologies (HPT TCP), Annex 35 Report.
- **International Organization for Standardization (ISO).** (2018). *ISO 50001:2018 Energy management systems — Requirements with guidance for use*. ISO, Geneva.
- **International Organization for Standardization (ISO).** (2014). *ISO 14046:2014 Environmental management — Water footprint — Principles, requirements and guidelines*. ISO, Geneva.
- **Linnhoff, B., Townsend, D. W., Boland, D., & Hewitt, G. F.** (1982). *User Guide on Process Integration for the Efficient Use of Energy*. Institution of Chemical Engineers (IChemE), Rugby, UK.
- **U.S. Environmental Protection Agency (EPA).** (2020). *40 CFR Part 98: Mandatory Greenhouse Gas Reporting Rule*. Federal Register, Washington, D.C.
- **Wang, Y. P., & Smith, R.** (1994). *Wastewater minimisation*. Chemical Engineering Science, 49(7), 981–1006. https://doi.org/10.1016/0009-2509(94)80006-5
- **World Resources Institute & World Business Council for Sustainable Development (WRI/WBCSD).** (2004). *The Greenhouse Gas Protocol: A Corporate Accounting and Reporting Standard* (Revised Edition). WRI/WBCSD, Washington, D.C.

---

## 17. Links

- **Module Index:** [`_index.md`](_index.md)
- **Sibling Topics:**
  - Topic 01: [`01_continuous-batch-discrete.md`](01_continuous-batch-discrete.md) — Continuous, Batch, and Discrete Production Models
  - Topic 02: [`02_yield-mass-balance.md`](02_yield-mass-balance.md) — Yield and Mass Balance Modeling in Process Operations
  - Topic 03: [`03_quality-specs-blending.md`](03_quality-specs-blending.md) — Quality Specifications, Assays, and Non-Linear Blending
  - Topic 04: [`04_campaigns-changeover.md`](04_campaigns-changeover.md) — Production Campaigns, Transition Losses, and Storage Decoupling
  - Topic 05: [`05_bulk-storage-inventory.md`](05_bulk-storage-inventory.md) — Bulk Storage, Tankage Logistics, and Contamination Risk
  - Topic 08: [`08_capacity-debottlenecking.md`](08_capacity-debottlenecking.md) — Capacity Analysis, Debottlenecking, and Capital Payback
  - Topic 10: [`10_process-safety-regulation.md`](10_process-safety-regulation.md) — Process Safety Management, Seveso III, and Environmental Permits
- **Related Program Modules:**
  - Module 11: Network Design Optimization
  - Module 20: Operations Planning and Master Scheduling
  - Module 28: Sustainability, Carbon Accounting, and Circular Supply Chains
- **Survey File:** [`../../../course/38_process_industries_commodities.md`](../../../course/38_process_industries_commodities.md)
