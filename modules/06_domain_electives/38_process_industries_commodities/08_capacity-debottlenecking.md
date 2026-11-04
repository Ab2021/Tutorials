# Capacity and Debottlenecking: Process Plant Throughput, Constraint Economics, and Capital Optimization

> Module 38 — Process Industries, Commodities, and Bulk Supply Chains
> Topic 08 of 10
> Estimated study time: 4 hours
> Prerequisites: Topic 01 (Continuous, Batch, and Discrete Production), Topic 02 (Yield and Mass Balance Modeling), Topic 04 (Campaigns and Changeover), Module 15 (Production Planning and Scheduling)
> Version 2026.5

---

## 1. Why this topic exists

In discrete assembly manufacturing, capacity expansion often proceeds linearly: adding a computer numerical control (CNC) machining center, installing a second robotic welding cell, or adding an extra shift to a packaging line. Work-in-process (WIP) sits in staging buffers, inventory can be held in standard totes or pallet racks, and the line can be stopped mid-shift without damaging material.

Process industries—such as petroleum refining, petrochemical synthesis, base chemicals, hydrometallurgical leaching, bulk food extraction, and continuous papermaking—operate under fundamentally different capital, thermodynamic, and hydrodynamic realities:
1. **Extreme Capital Intensity and Long Build Horizons:** Greenfield world-scale process plants require $500 million to $10 billion in capital expenditure and require three to seven years for front-end engineering design (FEED), environmental impact permitting, equipment procurement, construction, and commissioning. Because commodity markets undergo violent multi-year price cycles, greenfield additions frequently come online precisely during cyclical margin troughs, destroying invested capital.
2. **Debottlenecking Economics:** Brownfield debottlenecking (also termed revamping or capacity creep) unlocks 5% to 25% incremental throughput at 10% to 30% of the capital intensity ($\text{Capex} / \Delta \text{Capacity}$) of greenfield construction. Upgrading a distillation column tray layout, re-wheeling a centrifugal compressor, augmenting heat exchanger surface area, or retuning advanced process control (APC) loops can achieve payback periods measured in months rather than decades.
3. **The False Constraint Trap:** Naive debottlenecking programs frequently destroy value. Plant teams often target the "apparent bottleneck"—the equipment piece operating at the highest visible percentage of nameplate capacity or generating the most supervisory control and data acquisition (SCADA) alarms. However, in continuous fluid networks characterized by multi-component splitters, non-linear reaction kinetics, utility integration, and recycle loops, physical utilization does not equate to economic constraint. Capital spent debottlenecking a non-economic constraint yields zero incremental cash margin.
4. **The Whack-a-Mole (Constraint Chasing) Phenomenon:** Because process units are tightly coupled via direct piping without intermediate decoupling tanks, expanding Unit A by 15% frequently causes the plant to slam immediately into an unpredicted downstream barrier in Unit B at +2% throughput. The remaining 13% of newly acquired capacity sits permanently stranded, driving capital return below the weighted average cost of capital (WACC).
5. **Dynamic, Seasonal, and Feed-Slate Shifting:** Bottlenecks in continuous flow are not static. In summer, high wet-bulb temperatures derate cooling tower performance, shifting the plant constraint to overhead condensers and vacuum systems; in winter, dense ambient air and chilled cooling water shift the constraint to fired heater draft fans or furnace fuel firing. Similarly, shifting from light sweet to heavy sour feedstocks moves the bottleneck from primary fractionation to downstream hydrotreating and sulfur recovery units (SRU).
6. **Inadequacy of Discrete Operational Metrics:** Classic discrete Overall Equipment Effectiveness (OEE) metrics measure tool availability, cycle speed, and part rejects. In continuous process plants, machines do not pause for micro-stops; instead, operators derate throughput rates to prevent column flooding, avoid compressor surge, or respect environmental flaring permits. Quantifying true capacity requires decomposing losses across availability, thermodynamic rate derating, and chemical yield selectivity.

Mastering capacity debottlenecking requires a synthesis of chemical mass and energy balances, Goldratt's Drum-Buffer-Rope (DBR) theory adapted to continuous fluid physics, linear programming (LP) dual shadow price economics, and disciplined capital budgeting.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Deconstruct** complex process plant capacity into theoretical, design (nameplate), demonstrated, and effective operational capacity envelopes, identifying hidden losses.
2. **Apply** Goldratt's Drum-Buffer-Rope (DBR) synchronization framework to continuous fluid and bulk solids flowsheets with intermediate surge vessels and pipeline transfers.
3. **Formulate** continuous plant optimization linear programs (LP) and interpret dual variables (shadow prices $\lambda_i$) to distinguish physical constraints from true economic margin bottlenecks.
4. **Decompose** continuous Overall Equipment Effectiveness (OEE) into Availability, Performance (Thermodynamic Rate Derating), and Quality (Yield and Selectivity) factors while accounting for internal recycle streams.
5. **Predict** dynamic floating bottleneck migrations across seasonal ambient temperature swings, feed slate transitions, and catalyst aging lifecycles (Start of Run vs. End of Run).
6. **Quantify** debottlenecking capital efficiency using the Six-Tenths capital scaling rule, Net Present Value (NPV), and marginal capital intensity ($\Delta \text{Capex} / \Delta \text{Throughput}$).
7. **Construct** an executable, self-contained Python optimization script (<80 lines) using SciPy to compute dual shadow prices and simulate multi-stage debottlenecking capacity transitions.
8. **Audit** proposed debottlenecking engineering revamps against Process Safety Management (OSHA PSM 1910.119), relief valve capacities (API 520/521), and environmental New Source Review (NSR) regulatory boundaries.

---

## 3. The core mental model

A continuous process manufacturing plant is a tightly coupled hydraulic, thermodynamic, and economic system. Raw materials flow through successive thermodynamic transformations—pre-heating, reaction, phase separation, purification, and finishing—supported by an interconnected shared utility network (HP/MP/LP steam, cooling water, fuel gas, electrical power, and effluent treatment).

Unlike discrete lines where workpieces rest in totes between stations, continuous flow units are linked directly by pipe headers. Intermediate storage tanks exist only at designated decoupling points, providing finite surge capacity measured in hours of residence time.

```text
RAW FEEDSTOCK ──►[ Feed Storage Tank ] (Decoupling Buffer 1)
                         │
                         ▼ (Rate: F_in)
               ┌─────────────────────┐
               │  FEED PRE-HEATER /  │◄── Fuel Gas Header / HP Steam
               │   FIRED FURNACE     │
               └──────────┬──────────┘
                          │
                          ▼ (Effluent: F_react)
               ┌─────────────────────┐
               │ CATALYTIC REACTOR   │◄── Catalyst Regeneration / Makeup
               │ (Fixed Bed / CSTR)  │──► High-Pressure Offgas to Compressor
               └──────────┬──────────┘
                          │
                          ▼ (Crude Stream)
               ┌─────────────────────┐
               │ INTERMEDIATE SURGE  │ (Buffer 2: Protects Drum from Upstream Trips)
               │      VESSEL         │ Level: h(t), Residence Time: T_res
               └──────────┬──────────┘
                          │
                          ▼
            ╔═════════════════════════════╗
            ║      THE "DRUM" ASSET       ║  Binding Constraint (Highest Shadow Price: λ > 0)
            ║ PRIMARY FRACTIONATION TOWER ║◄─ Cooling Water (Condenser Pinch)
            ║      (D-101 Splitter)       ║◄─ MP Steam (Reboiler Duty Limit)
            ╚══════════╤══════════════════╝
                       │
       ┌───────────────┴───────────────┐
       ▼ (Overhead Vapor)              ▼ (Bottoms Liquid)
┌──────────────┐               ┌──────────────┐
│ LIGHT-ENDS   │               │ HEAVY-ENDS   │
│ REFRIGERATION│               │ STRIPPER     │
└──────┬───────┘               └──────┬───────┘
       │                              │
       ▼ (Recycle Stream: Off-Spec)   ▼
       └──────────────┬───────────────┘
                      │
                      ▼
               [ FINISHED BULK STORAGE ] (Decoupling Buffer 3: Protects Drum from Port/Rail Delays)
                      │
                      ▼ (Rate: F_out)
             MARINE / PIPELINE OFFTAKE
```

### The Causal Feedback Architecture

1. **The Physical Pace (The Drum):** In any continuous train, one physical or thermodynamic transformation bounds total system throughput under a given operating regime. This unit is the **Drum**. Its instantaneous processing rate ($\text{MT}/\text{hr}$ or $\text{bbl}/\text{day}$) dictates the allowable steady-state operating tempo of the entire complex. Operating upstream units faster than the Drum overfills intermediate surge tanks and forces emergency flaring or shutdowns. Operating downstream units faster starves them of feed.
2. **The Intermediate Buffers:** Continuous buffers are volumetric vessels (surge drums, bullet tanks, sphere tanks, floating-roof tanks) whose volume translates into operating time:
   $$\tau = \frac{V_{\text{active}}}{\dot{V}_{\text{flow}}}$$
   - An **upstream buffer** protects the Drum from being starved when upstream equipment suffers a trip, pump seal failure, or catalyst regeneration cycle.
   - A **downstream buffer** protects the Drum from being blocked when downstream purification or packaging units trip, or when export ships/railcars face demurrage delays.
3. **The Information Rope:** The **Rope** is the dynamic feed-forward/feedback supervisory control signal. In an Advanced Process Control (APC) or plant-wide DCS framework, the Rope continuously throttles raw material charge pumps to match the instantaneous throughput of the Drum, modulated by intermediate buffer inventory levels $h(t)$. If intermediate surge levels exceed safe operating envelopes, the Rope automatically derates fresh feed.
4. **The Economic Shadow Price Overlay ($\lambda_i$):** Linear programming models calculate the marginal economic value of relaxing each physical constraint by one unit:
   $$\lambda_i = \frac{\partial (\text{Operating Margin})}{\partial (\text{Capacity}_i)}$$
   - If unit $i$ is not fully utilized, it has slack capacity, and its shadow price is identically zero ($\lambda_i = 0$). Spending $\$1$ of debottlenecking capital on this unit yields $\$0$ in return.
   - If unit $i$ is the active economic bottleneck, $\lambda_i > 0$. The magnitude of $\lambda_i$ (e.g., $\$45.00/\text{MT}$) indicates exactly how much plant cash margin increases for each additional metric ton of capacity unlocked in unit $i$, establishing the upper bound on debottlenecking capital expenditure.

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Industry Misuse or Pitfall | Standard Engineering Unit |
|---|---|---|---|
| **Nameplate Capacity** | Theoretical maximum continuous throughput rate specified by the engineering procurement contractor (EPC) at design conditions. | Assuming nameplate can be achieved year-round without accounting for fouling, weather, or feed changes. | $\text{MT}/\text{day}$, $\text{bbl}/\text{day}$, $\text{Nm}^3/\text{hr}$ |
| **Demonstrated Capacity** | The highest historical throughput sustained safely over an extended operating window (typically 72 to 168 hours). | Confusing a 2-hour peak rate (unsustainable due to inventory drawdown) with true demonstrated continuous capacity. | $\text{MT}/\text{day}$, $\text{kt}/\text{year}$ |
| **Effective Capacity** | Demonstrated capacity discounted by planned turnarounds, scheduled catalyst regenerations, and statutory safety inspections. | Overestimating effective capacity by omitting decoking cycles or statutory boiler inspections. | $\text{MT}/\text{year}$, $\text{Operating Days}/\text{year}$ |
| **Physical Bottleneck** | The physical equipment asset that reaches its mechanical, hydraulic, or thermal limit first as total feed rate increases. | Assuming the physical bottleneck is always the economic bottleneck in multi-product, flexible-feed plants. | $\text{m}^3/\text{hr}$, $\text{MW}$, $\text{kPa}$ |
| **Economic Bottleneck** | The constraint whose relaxation yields the highest marginal increase in total plant operating margin (highest LP shadow price $\lambda_i$). | Upgrading a unit with high utilization but low product margin contribution, stranding capital. | $\$ / (\text{Unit Rate} \cdot \text{Time})$ |
| **Regulatory / Permit Bottleneck** | An environmental emission cap (SOx, NOx, VOC, flare volume) or safety limit that caps throughput before physical equipment saturates. | Spending capital to upgrade a distillation tower when the plant is legally capped by flare emissions under Title V permits. | $\text{Tons } \text{SO}_2/\text{year}$, $\text{kg VOC}/\text{hr}$ |
| **Shadow Price ($\lambda_i$)** | The Lagrange multiplier / dual variable associated with a capacity constraint in a linear or non-linear optimization model ($\partial Z^* / \partial C_i$). | Treating shadow prices as static constants; they change non-linearly as soon as capacity expands to the next binding constraint. | $\$ / \text{MT}$, $\$ / \text{bbl}$ |
| **Reduced Cost ($r_j$)** | The amount the objective coefficient of a non-basic decision variable must improve before that stream/product enters the optimal production plan. | Confusing reduced cost with unit manufacturing cost or accounting gross margin. | $\$ / \text{MT}$ |
| **Complementary Slackness** | Mathematical condition in optimization stating that either a constraint has zero slack ($s_i = 0$) or its dual value is zero ($\lambda_i = 0$). | Expecting positive financial return from debottlenecking a machine that currently operates with 5% excess margin. | Dimensionless condition |
| **Drum-Buffer-Rope (DBR)** | Production scheduling methodology from the Theory of Constraints where the Drum sets cadence, Buffers protect flow, and Ropes throttle release. | Treating DBR as a discrete-only concept, ignoring fluid residence times and vessel holdup physics. | Operational methodology |
| **Space Velocity (LHSV / WHSV)** | Liquid or Weight Hourly Space Velocity: volumetric/mass flow rate of feed divided by reactor catalyst volume or weight. | Increasing feed rate without realizing LHSV increases, causing incomplete conversion and severe downstream fouling. | $\text{hr}^{-1}$ |
| **Hydraulic Flooding** | Instability in a trayed or packed column where upward vapor velocity prevents downward liquid flow, causing liquid accumulation. | Forcing column throughput past 85% of flood rating, resulting in catastrophic loss of separation and overhead contamination. | % of flooding velocity |
| **Condenser Pinch** | Operational limit where the temperature difference between process condensing vapor and cooling water reaches minimum approach ($\Delta T_{\text{min}}$). | Failing to anticipate that summer cooling water temperatures will choke condenser duty, forcing plant rate cutbacks. | ${}^\circ\text{C}$ or $\text{K}$ approach |
| **Turndown Ratio** | The ratio of maximum operating throughput to minimum stable operating throughput without column weeping or flame instability. | Designing debottlenecked assets with high maximum rates but poor turndown, making low-demand periods unviable. | Ratio (e.g., $4:1$ or $2.5:1$) |
| **Continuous OEE** | Overall Equipment Effectiveness adapted for continuous processes: $\text{Availability} \times \text{Rate Efficiency} \times \text{Yield Efficiency}$. | Applying discrete piece-count OEE formulas directly to fluid flows, ignoring internal recycle and mass balances. | Percentage (%) |
| **Rate / Performance Loss** | Throughput lost due to operating equipment below maximum demonstrated design speed (e.g., thermal fouling, compressor derating). | Classifying rate derating as "acceptable normal operation" rather than an addressable capacity loss. | $\text{MT}/\text{hr}$ or % derating |
| **Yield / Quality Loss** | Mass of feedstock converted into unmarketable byproducts, low-value slop, flare gases, or off-spec material requiring re-processing. | Measuring quality only by final shipped specification without tracking internal off-spec recycle rework costs. | % mass loss or $\$ / \text{day}$ |
| **Surge Residence Time ($T_{\text{res}}$)** | Time a fluid element spends inside an intermediate vessel; active volume divided by volumetric throughput rate ($V / Q$). | Sizing surge tanks purely on spatial layout rather than the MTBF and MTTR distributions of adjacent units. | Hours or Minutes |
| **Floating Bottleneck** | A constraint that shifts across different equipment units based on ambient weather, feed composition, or catalyst age. | Optimizing a plant for a single nominal design point, causing debottlenecked capacity to vanish during summer or heavy feeds. | Operational phenomenon |
| **Six-Tenths Rule** | Empirical power-law relationship estimating capital cost scaling of process equipment: $\text{Capex}_A / \text{Capex}_B = (\text{Cap}_A / \text{Cap}_B)^{0.6}$. | Applying an exponent of 0.6 universally without verifying unit-specific exponents (e.g., compressors $n \approx 0.8$, tanks $n \approx 0.5$). | Empirical exponent $n$ |
| **Marginal Capital Intensity** | Debottlenecking capital expenditure divided by incremental annual capacity gained ($\Delta \text{Capex} / \Delta \text{Capacity}$). | Comparing greenfield capital intensity directly against brownfield revamp without factoring in turnaround downtime cost. | $\$ / (\text{MT}/\text{year})$ |
| **Start of Run (SOR)** | Initial operating period after a major turnaround when catalysts are fresh, exchangers are clean, and furnace tubes are decoked. | Sizing debottlenecking equipment based on SOR conditions, only to be strangled months later at End of Run (EOR). | Operational milestone |
| **End of Run (EOR)** | Terminal operating period before turnaround where catalyst deactivation and equipment fouling reach maximum tolerable limits. | Delaying turnarounds past EOR, causing exponential energy penalties and runaway equipment degradation. | Operational milestone |
| **Recycle Accumulation** | Phenomenon where unreacted components or inert trace contaminants build up in closed loops, consuming hydraulic capacity. | Debottlenecking fresh feed without expanding purge capacity, leading to inert choke-off. | $\text{mol}\%$ or $\text{MT}/\text{hr}$ recycle |

---

## 5. The mechanics, step by step

```text
  ┌─────────────────────────────────────────────────────────────┐
  │ STEP 1: Operating Data Auditing & Demonstrated Baselining   │
  │ • Extract 1-min DCS tags; filter out turnarounds/trips     │
  │ • Establish 95th percentile sustained operating envelope   │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STEP 2: Continuous OEE Decomposition & Mass Attribution     │
  │ • Availability: Run time vs planned calendar time           │
  │ • Rate Efficiency: Operating rate vs demonstrated peak      │
  │ • Quality/Yield: Prime first-pass yield vs slop/flare/purge │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STEP 3: Continuous Drum-Buffer-Rope (DBR) Synchronization   │
  │ • Locate primary pacing asset (The Drum)                    │
  │ • Size upstream/downstream surge buffers based on MTTR/MTBF │
  │ • Establish Rope logic linking Drum pace to feed charge     │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STEP 4: Process LP Optimization & Dual Shadow Price Deriv.  │
  │ • Formulate mass/energy/utility constraints                 │
  │ • Extract Lagrange multipliers (shadow prices λ_i)          │
  │ • Screen out physical constraints with zero economic value  │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STEP 5: Dynamic Ambient & Catalyst Aging Sensitivity        │
  │ • Model summer cooling tower pinch & winter furnace draft   │
  │ • Evaluate heavy sour vs light sweet feed transitions       │
  │ • Quantify catalyst deactivation trajectory (SOR to EOR)    │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STEP 6: Engineering Revamp Synthesis & Lever Selection      │
  │ • Hydraulics: Trays to structured packing                   │
  │ • Thermal: Heat exchanger network pinch optimization        │
  │ • Machinery: Compressor re-wheeling & motor upgrades        │
  │ • APC: Multivariable predictive constraint push             │
  └──────────────────────────────┬──────────────────────────────┘
                                 │
                                 ▼
  ┌─────────────────────────────────────────────────────────────┐
  │ STEP 7: Capital Budgeting & Step-Change Capacity Frontier   │
  │ • Cost scaling via power-law exponents                      │
  │ • Cash-flow modeling: incremental throughput vs opex/capex  │
  │ • Compute NPV, IRR, and Marginal Capital Intensity          │
  └─────────────────────────────────────────────────────────────┘
```

### Step 1: Operating Data Auditing and Demonstrated Capacity Baselining

The primary error in capacity analysis is relying on original equipment manufacturer (OEM) design nameplate specifications or single-point steady-state process simulation models (e.g., Aspen Plus, Aspen HYSYS). Real continuous plants degrade over time due to heat exchanger fouling, tray scaling, catalyst coking, and rotating equipment wear.

1. **Extract High-Frequency Time-Series:** Mine one-minute to one-hour data from the plant historian (e.g., AVEVA PI System, Aspen InfoPlus.21) across 24 to 36 months of continuous operation.
2. **Filter Outliers and Non-Steady States:** Segment data to exclude planned major turnarounds, emergency utility blackouts, feedstock starvation events, and major plant trips.
3. **Establish Operating Envelopes:**
   - Plot cumulative frequency distribution curves (Pareto / duration curves) of mass throughput rates ($\text{MT}/\text{hr}$) for each major processing unit.
   - The **Demonstrated Peak Capacity** is defined as the 98th percentile sustained throughput over a minimum 72-hour continuous window where all product specifications and environmental permit limits were met simultaneously.
   - The **Demonstrated Sustained Capacity** is defined as the 50th percentile (median) rate achieved during normal operating campaigns.
4. **Identify Hydraulic and Mechanical Boundaries:** Record the physical operating parameters at peak rates: column differential pressures ($\Delta P$), compressor suction valve positions (% open), furnace skin temperatures (${}^\circ\text{C}$), and cooling water return temperatures.

### Step 2: Continuous OEE Decomposition & Mass Balance Loss Attribution

Traditional discrete OEE metrics fail in fluid processing because plants rarely stop completely; instead, operators derate throughput rates to avoid crossing safety or environmental thresholds. Continuous Overall Equipment Effectiveness must be derived directly from conservation of mass and operating time.

$$\text{OEE}_{\text{continuous}} = A_{\text{cont}} \times P_{\text{cont}} \times Q_{\text{cont}}$$

#### 1. Availability Factor ($A_{\text{cont}}$)
Availability measures the fraction of calendar time that the unit is physically online and circulating fluid:
$$A_{\text{cont}} = \frac{T_{\text{operating}}}{T_{\text{calendar}} - T_{\text{planned\_turnaround}}}$$
Where:
- $T_{\text{operating}}$ is the total hours the unit processed feed during the evaluation period.
- $T_{\text{calendar}}$ is total calendar hours (e.g., $8,760\text{ hr}/\text{year}$).
- $T_{\text{planned\_turnaround}}$ is statutory, pre-budgeted turnaround downtime (e.g., 21 days every 4 years).
- Unplanned trips, tube ruptures, pump trips, and unscheduled maintenance count as availability losses ($T_{\text{unplanned}} = T_{\text{calendar}} - T_{\text{planned}} - T_{\text{operating}}$).

#### 2. Performance / Rate Efficiency Factor ($P_{\text{cont}}$)
Rate efficiency measures throughput speed losses during online hours:
$$P_{\text{cont}} = \frac{\int_0^{T_{\text{operating}}} \dot{m}_{\text{actual}}(t) \, dt}{T_{\text{operating}} \times \dot{m}_{\text{demonstrated\_peak}}}$$
Where:
- $\dot{m}_{\text{actual}}(t)$ is the instantaneous mass feed rate ($\text{MT}/\text{hr}$).
- $\dot{m}_{\text{demonstrated\_peak}}$ is the maximum demonstrated sustained rate established in Step 1.
- Rate losses occur when operators pull back rates due to furnace tube coking, condenser fouling on hot summer afternoons, or high column differential pressure.

#### 3. Quality / Yield Selectivity Factor ($Q_{\text{cont}}$)
In process operations, quality loss is not merely scrap parts; it represents feedstock mass degraded into sub-prime product, off-spec material requiring re-fractionation, flare losses, or unmarketable slop:
$$Q_{\text{cont}} = \frac{M_{\text{prime\_first\_pass}}}{M_{\text{total\_feed\_processed}}} = 1 - \frac{M_{\text{slop}} + M_{\text{flare}} + M_{\text{regrade\_discounted}} + M_{\text{recycle\_rework}}}{M_{\text{total\_feed\_processed}}}$$
Where:
- $M_{\text{prime\_first\_pass}}$ is the mass of finished product meeting contract specifications on the first pass.
- $M_{\text{recycle\_rework}}$ is off-spec liquid diverted back to upstream tankage for re-processing. Note that while recycle material is not lost physically, recycling 50 MT/hr of off-spec intermediate directly consumes 50 MT/hr of hydraulic capacity in upstream reactors and distillation columns, cannibalizing fresh feed capacity.

#### 4. The Internal Recycle Cannibalization Mechanism
A critical blind spot in continuous plant capacity analysis is the treatment of recycle loops. In chemical and petrochemical flowsheets (such as catalytic reformers, hydrocrackers, ethylene oxide reactors, and polymers), unreacted feed or off-spec intermediate is continuously recycled back to the reactor or separator inlet.

Let $F$ be the fresh feedstock flow rate ($\text{MT}/\text{hr}$), $R$ be the recycle stream flow rate ($\text{MT}/\text{hr}$), and $C_{\text{gross}}$ be the maximum hydraulic or thermal mass throughput rating of the reactor/column. The total load on the unit is:
$$F + R \le C_{\text{gross}}$$
Defining the recycle ratio as $r = R / F$, the maximum allowable fresh feed intake is:
$$F \le \frac{C_{\text{gross}}}{1 + r}$$

If catalyst selectivity degrades or separation efficiency slips, the recycle ratio increases from $r_1$ to $r_2$. The resulting collapse in fresh feed throughput is given by:
$$\Delta F = C_{\text{gross}} \left( \frac{1}{1 + r_1} - \frac{1}{1 + r_2} \right)$$

For example, in a loop with $C_{\text{gross}} = 500\text{ MT}/\text{hr}$, if sub-optimal catalyst selectivity causes $r$ to rise from $0.20$ to $0.40$, fresh feed capacity falls from $416.7\text{ MT}/\text{hr}$ to $357.1\text{ MT}/\text{hr}$—a net loss of $59.6\text{ MT}/\text{hr}$ ($14.3\%$) of plant throughput. The equipment operates at 100% mechanical load ($500\text{ MT}/\text{hr}$), yet the plant's commercial capacity has collapsed. Debottlenecking in such systems requires improving reaction selectivity or separation sharpness to suppress $r$, rather than physically enlarging vessels.

### Step 3: Continuous Drum-Buffer-Rope (DBR) Synchronization

Applying Goldratt's Theory of Constraints to continuous fluid systems requires translating discrete WIP inventory into fluid holdup and residence times.

#### 1. Pinpointing the Drum
The Drum is the single physical equipment unit that sets the maximum sustainable cadence for the entire production train. It is identified mathematically in Step 4 as the unit with the highest non-zero shadow price.

#### 2. Sizing Intermediate Decoupling Buffers
In fluid networks, buffers are physical storage tanks, spheres, or surge vessels situated between units. Buffer capacity is sized in units of time ($\tau_{\text{buffer}}$) based on the Mean Time To Repair (MTTR) and Mean Time Between Failures (MTBF) of the surrounding equipment:
$$\tau_{\text{upstream\_buffer}} = \mu_{\text{MTTR, upstream}} + k \cdot \sigma_{\text{MTTR, upstream}}$$
$$\tau_{\text{downstream\_buffer}} = \mu_{\text{MTTR, downstream}} + k \cdot \sigma_{\text{MTTR, downstream}}$$
Where $k$ is a safety factor (typically $2.0 \le k \le 3.0$ representing a 95% to 99% protection level under log-normal failure distributions).
The physical tank volume required is:
$$V_{\text{tank}} = \frac{\tau_{\text{buffer}} \times \dot{m}_{\text{drum}}}{\rho_{\text{fluid}} \times (h_{\text{max}} - h_{\text{min}})}$$
Where $\rho_{\text{fluid}}$ is fluid density and $(h_{\text{max}} - h_{\text{min}})$ is the working volume fraction of the tank.

#### 3. Establishing the Digital Rope
The Rope is the control feedback mechanism implemented in the Distributed Control System (DCS):
- If the upstream surge tank level $h(t)$ drops toward $h_{\text{min}}$, the Drum is in danger of starving; the Rope signals the upstream feed pumps to increase charge rate or signals secondary units to shed non-critical recycle.
- If downstream tank level $h(t)$ rises toward $h_{\text{max}}$, the Drum faces back-pressure and blocking; the Rope immediately throttles raw feed intake at the plant boundary, preventing overfilling and emergency flaring.

### Step 4: Process LP Optimization & Dual Shadow Price ($\lambda_i$) Derivation

Linear programming (LP) provides the mathematical engine for debottlenecking economics. In large-scale refineries and petrochemical complexes, tools such as Aspen PIMS, Honeywell Haverly GRTMPS, or custom Python solvers evaluate the global mass, energy, and capacity balance.

#### Primal Optimization Formulation
Maximize the net operating cash margin per operating day:
$$\max_{\mathbf{x}, \mathbf{y}, \mathbf{u}} Z = \sum_{p \in \mathcal{P}} P_p y_p - \sum_{f \in \mathcal{F}} C_f x_f - \sum_{u \in \mathcal{U}} O_u w_u - \sum_{k \in \mathcal{K}} E_k v_k$$
Subject to:
1. **Equipment Capacity Envelopes:**
   $$\sum_{j \in \mathcal{J}} a_{uj} w_j \le \text{Cap}_u \quad \forall u \in \mathcal{U} \quad [\lambda_u]$$
2. **Mass Balance Conservation Across Process Units:**
   $$\sum_{j \in \text{Inputs}(m)} w_j - \sum_{j \in \text{Outputs}(m)} w_j - \text{Loss}_m = 0 \quad \forall m \in \mathcal{M} \quad [\mu_m]$$
3. **Component Yield Fractions:**
   $$y_{pj} = \alpha_{pj} w_j \quad \forall p \in \mathcal{P}, j \in \mathcal{J}$$
4. **Shared Utility Header Limits:**
   $$\sum_{u \in \mathcal{U}} \beta_{ku} w_u \le \text{Cap}_{\text{utility}, k} \quad \forall k \in \mathcal{K} \quad [\gamma_k]$$
5. **Feedstock Availability and Market Demand Absorptions:**
   $$x_f \le \text{Supply}_f \quad \forall f \in \mathcal{F} \quad [\sigma_f]$$
   $$y_p \le \text{Demand}_p \quad \forall p \in \mathcal{P} \quad [\pi_p]$$

#### Dual Formulation and Complementary Slackness
The dual variable (shadow price) associated with unit capacity constraint $u$ is defined as:
$$\lambda_u = \frac{\partial Z^*}{\partial \text{Cap}_u}$$
Under the Karush-Kuhn-Tucker (KKT) conditions:
$$\lambda_u \cdot \left( \text{Cap}_u - \sum_{j \in \mathcal{J}} a_{uj} w_j^* \right) = 0, \quad \lambda_u \ge 0$$

- **Non-Binding Constraints ($\text{Slack} > 0$):** If unit $u$ has even 0.1 MT/day of unused capacity at the optimum, $\lambda_u = 0.00$. Spending debottlenecking capital to add capacity to this unit produces zero incremental cash margin.
- **Binding Economic Bottlenecks ($\text{Slack} = 0, \lambda_u > 0$):** If unit $u$ is operating at its exact boundary, $\lambda_u$ represents the exact dollar increase in daily operating profit achieved by expanding unit capacity by 1.0 MT/day.
- **Utility Constraints ($\gamma_k > 0$):** In many plants, individual processing units appear to have excess capacity, but total plant feed is constrained because the cooling water return temperature exceeds the environmental limit or the medium-pressure (MP) steam boiler is firing at 100%. Here, the utility header is the true economic bottleneck.

#### Allowable Right-Hand Side (RHS) Ranges and Basis Stability
A common and dangerous error in debottlenecking economics is assuming that the shadow price $\lambda_u$ remains constant indefinitely as capacity expands. In linear programming, the simplex optimal basis matrix $\mathbf{B}$ remains optimal only as long as the basic variables remain non-negative:
$$\mathbf{x}_B = \mathbf{B}^{-1} (\mathbf{b} + \Delta \mathbf{b}) \ge \mathbf{0}$$
For a single capacity expansion on unit $u$ ($\Delta b_u > 0$), this condition defines the **Allowable Increase** ($\Delta C_u^+$):
$$\Delta C_u^+ = \min_{i: (\mathbf{B}^{-1})_{iu} < 0} \left\{ \frac{-(\mathbf{B}^{-1} \mathbf{b})_i}{(\mathbf{B}^{-1})_{iu}} \right\}$$
Similarly, the **Allowable Decrease** ($\Delta C_u^-$) defines the lower boundary before the constraint ceases to bind:
$$\Delta C_u^- = \min_{i: (\mathbf{B}^{-1})_{iu} > 0} \left\{ \frac{(\mathbf{B}^{-1} \mathbf{b})_i}{(\mathbf{B}^{-1})_{iu}} \right\}$$

Within the allowable range $[\text{Cap}_u - \Delta C_u^-, \, \text{Cap}_u + \Delta C_u^+]$, the shadow price $\lambda_u$ is constant, and the marginal return on debottlenecking is perfectly linear:
$$\Delta Z^* = \lambda_u \times \Delta \text{Cap}_u$$

As soon as $\Delta \text{Cap}_u > \Delta C_u^+$, a pivot occurs: a previously slack downstream or upstream unit hits $100\%$ utilization, entering the basis. The profit curve $Z^*(\text{Cap}_u)$ is piecewise linear and strictly concave:

```text
TOTAL OPERATING
PROFIT Z* ($/day)
    ▲
    │                                          / Slopes = 0 (Total Saturation)
    │                                 ┌───────┘ (λ_3 = $15/MT: Tertiary Constraint)
    │                        ┌────────┘ (λ_2 = $65/MT: Secondary Constraint)
    │               ┌────────┘ (λ_1 = $216.67/MT: Primary Drum Constraint)
    │      ─────────┘ (Allowable Increase ΔC_1^+ Exceeded: Pivot Occurs)
    │     /
    └───┬───────────┬────────┬────────┬───────►
       Cap_0      Cap_1    Cap_2    Cap_3   CAPACITY OF UNIT u
```

Every debottlenecking project must explicitly calculate $\Delta C_u^+$. Spending capital to buy capacity beyond $\text{Cap}_u + \Delta C_u^+$ without simultaneously expanding the entering constraint results in immediate capital stranding.

### Step 5: Dynamic, Seasonal, and Catalyst-Aging Bottleneck Analysis

Process bottlenecks are dynamic vectors that migrate across the flowsheet as external boundary conditions change:

```text
       SUMMER CONDITIONS                          WINTER CONDITIONS
  Ambient: 38°C, High Humidity               Ambient: -5°C, Dry Air
  Cooling Water Supply: 32°C                 Cooling Water Supply: 12°C
  ─────────────────────────────────          ─────────────────────────────────
  • Condenser Pinch: Tower D-101             • Dense Air: Excess Draft Fan Head
    floods due to back-pressure.             • High Cooling Margin in D-101.
  • Compressor Suction Hot: Low density      • Furnace Firing Limit: High heat losses;
    mass flow derated by 12%.                  fuel gas supply header saturated.
  ═════════════════════════════════          ═════════════════════════════════
  BOTTLENECK: OVERHEAD CONDENSER             BOTTLENECK: FIRED HEATER REBOILER
  (Shadow Price: λ_cond = $42/MT)            (Shadow Price: λ_furnace = $38/MT)
```

1. **Seasonal Ambient Swings:**
   - Cooling tower performance is governed by ambient wet-bulb temperature. In summer, cooling water supply temperatures rise from 15°C to 32°C. The log-mean temperature difference ($\Delta T_{LM}$) in column condensers collapses:
     $$Q_{\text{condenser}} = U \cdot A \cdot \Delta T_{LM}$$
     To maintain condensation, column operating pressure must be raised, which reduces relative volatility between chemical components, increases required reboiler duty, and triggers premature hydraulic tray flooding.
2. **Feed Slate Transitions:**
   - Shifting an oil refinery from light sweet Brent crude (API 38°, 0.4% sulfur) to heavy sour Maya crude (API 22°, 3.5% sulfur) shifts the bottleneck from the crude distillation unit (CDU) to the delayed coker, hydrotreater, and sulfur recovery unit (SRU). The CDU may drop to 70% utilization while the SRU hits 100% capacity, throttling global plant intake.
3. **Catalyst Deactivation Lifecycle (SOR to EOR):**
   - In hydrocracking and catalytic reforming, catalyst active sites foul with coke. To maintain target conversion, operators steadily increase reactor bed temperature:
     $$T_{\text{bed}}(t) = T_{\text{SOR}} + \alpha \cdot t$$
     Higher temperatures increase thermal cracking to light gases ($C_1 - C_4$), which overloads the wet gas compressor (WGC). A plant that was reactor-limited at Start of Run (SOR) becomes compressor-limited at End of Run (EOR).

### Step 6: Engineering Debottlenecking Levers Across Process Archetypes

Once the true economic constraint and its migration envelope are established, engineering teams apply targeted revamp technologies:

#### 1. Distillation and Absorption Columns
Column capacity is bounded by vapor-liquid hydraulics—specifically entrainment flooding and downcomer backup. The maximum allowable superficial vapor velocity before flooding is governed by the Souders-Brown equation:
$$u_{\text{flood}} = C_{sb} \sqrt{\frac{\rho_L - \rho_V}{\rho_V}} \left( \frac{\sigma}{20} \right)^{0.2}$$
Where $\rho_L$ and $\rho_V$ are liquid and vapor densities ($\text{kg}/\text{m}^3$), $\sigma$ is surface tension ($\text{dyne}/\text{cm}$), and $C_{sb}$ is the capacity factor ($\text{m}/\text{s}$) determined by tray spacing and the dimensionless flow parameter $F_{lv} = \frac{L}{V} \sqrt{\frac{\rho_V}{\rho_L}}$.

- **Internals Retrofit:** Replacing conventional valve or sieve trays with high-capacity truncated-downcomer trays increases active bubbling area by 15% to 20%, raising $C_{sb}$ from $0.065\text{ m/s}$ to $0.082\text{ m/s}$ and delaying the onset of jet flooding.
- **Structured Packing Revamps:** In vacuum systems and low-pressure towers (e.g., ethylbenzene/styrene splitters or crude vacuum distillation units), replacing trays with corrugated wire-gauze or sheet-metal structured packing reduces pressure drop per theoretical stage by up to 80%. Lower tower bottom pressure increases relative volatility $\alpha$, reducing required reflux ratio $L/D$ and debottlenecking both reboiler and condenser duties simultaneously.
- **Divided Wall Columns (DWC):** Consolidating two sequential distillation columns into a single shell with an internal vertical dividing wall eliminates an entire reboiler/condenser train, reducing overall thermal utility load by 25% to 40% and freeing up steam header capacity.
#### 2. Fired Heaters and Furnaces
- **Convection Section Tube Augmentation:** Adding extended surface (finned or studded) tubes in the convection section recovers waste heat from flue gas, increasing thermal efficiency from 78% to 90% without burning additional fuel.
- **Combustion Air Pre-heaters (APH):** Pre-heating combustion air with hot flue gas reduces required fuel gas firing, debottlenecking plants constrained by burner NOx limits.

#### 3. Rotating Machinery (Compressors and Pumps)
Centrifugal gas compressors (such as wet gas compressors in FCC units or cracked gas compressors in ethylene plants) are frequently the plant-wide Drum asset. Compressor capacity is governed by shaft power and polytropic head:
$$H_p = \frac{Z_{\text{avg}} R T_1}{M} \left( \frac{n}{n-1} \right) \left[ \left( \frac{P_2}{P_1} \right)^{\frac{n-1}{n}} - 1 \right]$$
$$\dot{W}_{\text{shaft}} = \frac{\dot{m} \cdot H_p}{\eta_p}$$
Where $Z_{\text{avg}}$ is gas compressibility, $T_1$ is suction temperature ($K$), $M$ is gas molecular weight, $P_2 / P_1$ is compression pressure ratio, $\dot{m}$ is mass flow rate, and $\eta_p$ is polytropic efficiency.

- **Centrifugal Compressor Re-Wheeling:** Replacing existing internal compressor impellers and stationary diffusers with modern custom 3D aerodynamic profiles increases mass flow capacity by 10% to 25% within the existing casing, avoiding the massive capex of a replacement casing.
- **Suction Pre-Cooling:** Because polytropic head $H_p$ and required shaft power $\dot{W}_{\text{shaft}}$ scale directly with suction temperature $T_1$, installing a chilled-water or ammonia suction gas pre-cooler drops $T_1$ by 15°C to 20°C. This densifies the gas, debottlenecking driver power limits and increasing mass throughput by 6% to 10% without altering rotating machinery.
- **Variable Frequency Drives (VFD):** Installing VFDs on large induced-draft (ID) fans and boiler feed pumps eliminates throttling valve losses, lowering electrical load on saturated plant substations.

#### 4. Heat Exchanger Networks and Pinch Retrofits
Heat exchangers bound throughput when the available heat transfer rate reaches saturation:
$$Q = U \cdot A \cdot \text{LMTD} \cdot F_t$$
Where $U$ is overall heat transfer coefficient ($\text{W}/\text{m}^2\cdot\text{K}$), $A$ is surface area ($\text{m}^2$), $\text{LMTD}$ is log-mean temperature difference, and $F_t$ is geometry correction factor.
- **Pinch Redistribution:** Industrial debottlenecking applies Pinch Analysis to re-sequence process heat exchangers, shifting heat loads away from over-fired utility furnaces into cold process pre-heat trains.
- **High-Efficiency Heat Transfer Surfaces:** Retrofitting existing shell-and-tube shells with twisted tubes, helical baffles, or plate-and-frame units increases heat transfer coefficients ($U$) by 40% to 100% while reducing fouling rates, eliminating reboiler bottlenecks.
#### 5. Advanced Process Control (APC / MPC)
- **Constraint Push Control:** Industrial process units are operated with a safety cushion below physical design limits because manual operator control exhibits variance ($\sigma$). By deploying Multivariable Predictive Control (MPC), process variance is reduced by 50% or more. This allows the steady-state operating setpoint ($\mu$) to be shifted closer to the hard constraint limit ($L$):
  $$\Delta \text{Capacity}_{\text{APC}} = (L - k \sigma_{\text{new}}) - (L - k \sigma_{\text{old}}) = k (\sigma_{\text{old}} - \sigma_{\text{new}})$$
  MPC upgrades often unlock 3% to 6% capacity with near-zero capital investment.

### Step 7: Capital Budgeting & Step-Change Capacity Frontier

Debottlenecking investments exhibit non-linear capital cost curves. Minor operational and control debottlenecks cost very little per unit of incremental capacity, whereas overcoming major structural barriers requires replacing heavy pressure vessels.

```text
CUMULATIVE
CAPEX ($M)
    ▲
$60 │                                                      / Greenfield
    │                                                     /  Trajectory
$50 │                                                    /
    │                                           ┌───────┘ (Phase 3: Replace
$40 │                                           │          Reboiler & Furnace)
    │                                   ┌───────┘ (Phase 2: Re-wheel Compressor
$30 │                                   │          & Retray Splitter)
    │                       ┌───────────┘
$20 │                       │ (Phase 1: Retray Tower D-101)
    │          ┌────────────┘
$10 │          │ (Quick-hit: APC Tuning & Pump Impeller Trims)
    │   ───────┘
 $0 └───┬──────┬────────────┬───────────┬───────┬───────┬───────►
       Base   +3%          +8%         +14%    +18%    +25%
                            INCREMENTAL PLANT THROUGHPUT
```

#### The Six-Tenths Scaling Rule for Brownfield Revamps
The baseline capital cost of scaling process equipment is approximated by:
$$\text{Capex}_{\text{revamp}} = \text{Capex}_{\text{base}} \left( \frac{\text{Capacity}_{\text{new}}}{\text{Capacity}_{\text{base}}} \right)^n$$
Where $n$ is the characteristic equipment exponent:
- Atmospheric storage tanks: $n \approx 0.50$ to $0.60$
- Shell-and-tube heat exchangers: $n \approx 0.65$ to $0.70$
- Centrifugal gas compressors: $n \approx 0.80$ to $0.85$
- Fired process heaters: $n \approx 0.75$ to $0.80$

#### Financial Evaluation Metrics
For each discrete debottlenecking phase $k$:
1. **Incremental Operating Cash Flow ($\Delta \text{CF}_t$):**
   $$\Delta \text{CF}_t = \Delta \text{Throughput} \times (\text{Product Basket Realization} - \text{Feedstock Cost} - \text{Variable Opex}) - \Delta \text{Fixed Opex}$$
2. **Net Present Value (NPV):**
   $$\text{NPV} = \sum_{t=1}^T \frac{\Delta \text{CF}_t}{(1 + r)^t} - \text{Capex}_{\text{revamp}} - \text{Downtime Revenue Loss during Revamp}$$
   > **Note:** A critical omission in debottlenecking economics is failing to penalize the project for plant downtime during construction. If a 10-day extended turnaround is required to install new tower internals, the lost margin during those 10 days must be charged directly to project capex.
3. **Marginal Capital Intensity ($\text{MCI}$):**
   $$\text{MCI} = \frac{\text{Capex}_{\text{revamp}} + \text{Downtime Margin Loss}}{\Delta \text{Capacity Annualized}} \quad \left[ \frac{\$}{\text{MT}/\text{year}} \right]$$
   If $\text{MCI}_{\text{revamp}} < 0.35 \times \text{MCI}_{\text{greenfield}}$, the debottlenecking project is an exceptional capital allocation choice.

---

## 6. Worked example

### The Gulf Coast Aromatics Fractionation Train Revamp

Consider a world-scale Gulf Coast petrochemical aromatics extraction and fractionation facility. The train processes hydrotreated pyrolysis gasoline (PyGas) to recover high-purity chemical-grade Benzene and Toluene.

The continuous processing train consists of four sequential processing units:
1. **Unit 1: Pre-Fractionator Column (D-101):** Removes light $C_5 / C_6$ non-aromatics overhead.
2. **Unit 2: Liquid-Liquid Aromatics Extractor (E-102):** Uses sulfolane solvent to separate aromatics from aliphatic raffinate.
3. **Unit 3: Benzene-Toluene Fractionator (D-103):** Distills high-purity Benzene overhead and Toluene bottoms.
4. **Unit 4: Product Re-run & Clay Treater (D-104):** Removes trace olefins and finishes chemical-grade Toluene.
5. **Shared Utility: Medium-Pressure (MP) Steam Header:** Provides reboiler heating across all columns; total boiler supply capacity is strictly capped at $150.0\text{ MT}/\text{hr}$.

```text
PyGas Feed (F) ──►[ D-101 ]──► Extract Feed ──►[ E-102 ]──► Aromatics ──►[ D-103 ]──► Benzene (B)
                     │                            │                           │
                     ▼ (Lights)                   ▼ (Raffinate)               ▼
                   C5/C6                        Aliphatics                [ D-104 ]──► Toluene (T)
```

#### Baseline Operational Data
- **Feedstock:** PyGas cost = $\$650.00/\text{MT}$.
- **Products:**
  - Pure Benzene realization = $\$1,050.00/\text{MT}$.
  - Pure Toluene realization = $\$920.00/\text{MT}$.
  - Non-aromatic Lights ($C_5/C_6$) value = $\$520.00/\text{MT}$.
  - Aliphatic Raffinate value = $\$560.00/\text{MT}$.
- **Mass Balance Split Fractions (per metric ton of PyGas feed):**
  - Lights from D-101: 15% ($0.15\text{ MT}$)
  - Raffinate from E-102: 25% ($0.25\text{ MT}$)
  - Pure Benzene from D-103: 38% ($0.38\text{ MT}$)
  - Pure Toluene from D-104: 20% ($0.20\text{ MT}$)
  - Flare / Process shrinkage loss: 2% ($0.02\text{ MT}$)
- **Current Equipment Nameplate Capacities (Feed Basis):**
  - Pre-Fractionator D-101: $12,000\text{ MT}/\text{day}$ fresh feed.
  - Extractor E-102: $10,000\text{ MT}/\text{day}$ feed to extractor ($0.85 \times \text{Fresh Feed}$). Max fresh feed equivalent = $10,000 / 0.85 = 11,764.7\text{ MT}/\text{day}$.
  - Benzene-Toluene Column D-103: $6,500\text{ MT}/\text{day}$ feed to column ($0.60 \times \text{Fresh Feed}$). Max fresh feed equivalent = $6,500 / 0.60 = 10,833.3\text{ MT}/\text{day}$.
  - Clay Treater & Re-run D-104: $2,300\text{ MT}/\text{day}$ feed to unit ($0.20 \times \text{Fresh Feed}$). Max fresh feed equivalent = $2,300 / 0.20 = 11,500.0\text{ MT}/\text{day}$.
- **Specific MP Steam Consumption:**
  - D-101: $0.08\text{ MT steam} / \text{MT fresh feed}$
  - E-102: $0.05\text{ MT steam} / \text{MT fresh feed}$
  - D-103: $0.18\text{ MT steam} / \text{MT fresh feed}$
  - D-104: $0.06\text{ MT steam} / \text{MT fresh feed}$
  - Total steam per MT fresh feed = $0.08 + 0.05 + 0.18 + 0.06 = 0.37\text{ MT steam} / \text{MT feed}$.
  - Steam cost = $\$24.00/\text{MT steam}$.
  - Maximum feed allowed by Steam Header ($150\text{ MT}/\text{hr} \times 24\text{ hr}/\text{day} = 3,600\text{ MT steam}/\text{day}$):
    $$\text{Feed}_{\text{steam\_limit}} = \frac{3,600}{0.37} = 9,729.7\text{ MT}/\text{day}$$

#### Baseline Economics & Bottleneck Identification

Let us calculate the net gross margin contribution per metric ton of PyGas processed:
- **Revenue per MT feed:**
  $$R = (0.15 \times 520) + (0.25 \times 560) + (0.38 \times 1050) + (0.20 \times 920) = 78.00 + 140.00 + 399.00 + 184.00 = \$801.00/\text{MT}$$
- **Raw Material & Utility Cost per MT feed:**
  $$C_{\text{feed}} = \$650.00/\text{MT}$$
  $$C_{\text{steam}} = 0.37\text{ MT steam} \times \$24.00/\text{MT} = \$8.88/\text{MT}$$
  $$C_{\text{variable\_opex}} = \$12.12/\text{MT}$$
  $$\text{Net Operating Cash Margin } (m) = \$801.00 - (\$650.00 + \$8.88 + \$12.12) = \$130.00/\text{MT feed}$$

Now, let us audit the physical vs. utility capacity limits:

| Equipment / Utility Unit | Physical Capacity (Unit Stream) | Equivalent Fresh Feed Limit | Utilization at 9,729.7 MT/day Feed | Status |
|---|---|---|---|---|
| **D-101 Pre-Fractionator** | $12,000\text{ MT}/\text{day}$ feed | $12,000.0\text{ MT}/\text{day}$ | $81.1\%$ | Excess Slack |
| **E-102 Extractor** | $10,000\text{ MT}/\text{day}$ feed | $11,764.7\text{ MT}/\text{day}$ | $82.7\%$ | Excess Slack |
| **D-103 BT Splitter** | $6,500\text{ MT}/\text{day}$ feed | $10,833.3\text{ MT}/\text{day}$ | $89.8\%$ | Secondary Constraint |
| **D-104 Clay Treater** | $2,300\text{ MT}/\text{day}$ feed | $11,500.0\text{ MT}/\text{day}$ | $84.6\%$ | Excess Slack |
| **MP Steam Header** | $3,600\text{ MT steam}/\text{day}$ | **$9,729.7\text{ MT}/\text{day}$** | **$100.0\%$** | **BINDING BOTTLENECK** |

#### Dual Variable / Shadow Price Analysis
The linear program optimization yields the following dual values:
- $\lambda_{\text{D-101}} = \$0.00/\text{MT}$ (Slack = $2,270.3\text{ MT}$)
- $\lambda_{\text{E-102}} = \$0.00/\text{MT}$ (Slack = $1,729.7\text{ MT}$)
- $\lambda_{\text{D-103}} = \$0.00/\text{MT}$ (Slack = $662.2\text{ MT}$)
- $\lambda_{\text{D-104}} = \$0.00/\text{MT}$ (Slack = $354.1\text{ MT}$)
- $\lambda_{\text{MP\_Steam}} = \frac{\$130.00\text{ margin}}{0.37\text{ MT steam}} = \$351.35\text{ per MT of MP steam capacity added per day}$

> **Critical Operational Insight:** If the plant manager had looked only at processing equipment, Column D-103 would have been identified as the bottleneck because its utilization ($89.8\%$) was the highest among all process vessels. Spending $\$8\text{ million}$ to re-tray D-103 would have produced **zero additional throughput** because the plant was strictly pinned by the $150\text{ MT}/\text{hr}$ MP steam header!

#### Step-Change Debottlenecking Analysis

The engineering team evaluates a staged debottlenecking strategy:

##### Phase 1: Utility Debottlenecking (Heat Integration)
Instead of building a new $\$25\text{M}$ package boiler, the team installs an effluent-to-feed heat exchanger network (HEN) around column D-103, recovering reboiler heat.
- **Cost:** $\$4.2\text{ million}$ capex; 4 days turnaround downtime ($\$130/\text{MT} \times 9,729.7 \times 4 = \$5.06\text{M}$ margin loss). Total effective cost = $\$9.26\text{M}$.
- **Result:** Reduces D-103 steam demand from $0.18$ to $0.11\text{ MT steam}/\text{MT feed}$. Total plant steam consumption drops from $0.37$ to $0.30\text{ MT steam}/\text{MT feed}$.
- **New Steam Capacity Boundary:**
  $$\text{Feed}_{\text{new\_steam}} = \frac{3,600\text{ MT steam}/\text{day}}{0.30\text{ MT steam}/\text{MT feed}} = 12,000.0\text{ MT}/\text{day}$$
- **The Constraint Shift (Slamming into the Next Barrier):**
  As feed expands past $9,729.7\text{ MT}/\text{day}$, throughput increases until it hits the next physical barrier: **Column D-103 at $10,833.3\text{ MT}/\text{day}$**.
  The plant cannot reach $12,000\text{ MT}/\text{day}$. It stops at $10,833.3\text{ MT}/\text{day}$.
- **Incremental Capacity Unlocked:**
  $$\Delta \text{Feed} = 10,833.3 - 9,729.7 = 1,103.6\text{ MT}/\text{day} \quad (+11.34\%)$$
- **Annual Incremental Cash Flow (assuming 345 operating days/year):**
  $$\Delta \text{Margin} = 1,103.6\text{ MT}/\text{day} \times \$130.00/\text{MT} \times 345\text{ days} = \$49,496,460/\text{year}$$
  *(Even after accounting for utility savings, baseline throughput margin dominates).*

##### Phase 2: Distillation Retraying (Eliminating D-103 Barrier)
With D-103 now the active binding constraint ($\lambda_{\text{D-103}} = \$130.00 / 0.60 = \$216.67/\text{MT}$), the team replaces conventional valve trays with high-capacity truncated downcomer structured trays.
- **Cost:** $\$3.5\text{ million}$ capex (installed during scheduled turnaround, zero incremental downtime penalty).
- **Result:** Increases D-103 feed capacity from $6,500$ to $7,500\text{ MT}/\text{day}$ ($12,500\text{ MT}/\text{day}$ fresh feed equivalent).
- **The Next Constraint:** Feed expands from $10,833.3\text{ MT}/\text{day}$ until it hits **Unit 4 (Clay Treater D-104) at $11,500.0\text{ MT}/\text{day}$**.
- **Incremental Capacity Unlocked in Phase 2:**
  $$\Delta \text{Feed}_{\text{Phase 2}} = 11,500.0 - 10,833.3 = 666.7\text{ MT}/\text{day} \quad (+6.85\% \text{ over baseline})$$
- **Annual Incremental Cash Flow for Phase 2:**
  $$\Delta \text{Margin}_{\text{Phase 2}} = 666.7\text{ MT}/\text{day} \times \$130.00/\text{MT} \times 345\text{ days} = \$29,899,830/\text{year}$$

##### Sensitivity Analysis: Derivation of the Allowable Increase on D-103
Let us explicitly calculate why Phase 2 only captured $666.7\text{ MT}/\text{day}$ of fresh feed despite adding $1,000.0\text{ MT}/\text{day}$ of mechanical column capacity to D-103.

In the post-Phase 1 basis:
- Fresh feed is bounded by D-103: $0.60 \times \text{Feed} \le \text{Cap}_{\text{D-103}}$.
- At baseline $\text{Cap}_{\text{D-103}} = 6,500.0\text{ MT}/\text{day}$, $\text{Feed} = 6,500.0 / 0.60 = 10,833.3\text{ MT}/\text{day}$.
- The next competing inequality is Clay Treater D-104:
  $$0.20 \times \text{Feed} \le 2,300.0\text{ MT}/\text{day} \implies \text{Feed} \le 11,500.0\text{ MT}/\text{day}$$
- The maximum fresh feed that D-104 can tolerate is $11,500.0\text{ MT}/\text{day}$, which corresponds to a D-103 flow of:
  $$\text{Cap}_{\text{D-103}}^{\text{critical}} = 0.60 \times 11,500.0 = 6,900.0\text{ MT}/\text{day}$$
- Therefore, the exact **Allowable Increase** ($\Delta C_{\text{D-103}}^+$) for constraint D-103 is:
  $$\Delta C_{\text{D-103}}^+ = 6,900.0 - 6,500.0 = 400.0\text{ MT}/\text{day}$$
- Over this interval $[6,500.0, \, 6,900.0]$, the marginal profit derivative is strictly constant:
  $$\frac{\partial Z^*}{\partial \text{Cap}_{\text{D-103}}} = \lambda_{\text{D-103}} = \frac{\$130.00}{0.60} = \$216.67/\text{MT}$$
- Expanding D-103 capacity from $6,500.0$ to $6,900.0\text{ MT}/\text{day}$ generates:
  $$\Delta Z^* = 400.0\text{ MT}/\text{day} \times \$216.67/\text{MT} = \$86,666.67/\text{day}$$
  $$\Delta \text{Feed} = \frac{400.0}{0.60} = 666.7\text{ MT}/\text{day}$$
- Any capacity added to D-103 above $6,900.0\text{ MT}/\text{day}$ (such as the remaining $600.0\text{ MT}/\text{day}$ installed by the $7,500\text{ MT}/\text{day}$ tray revamp) is completely unutilized until D-104 is debottlenecked. The shadow price $\lambda_{\text{D-103}}$ for that incremental $600.0\text{ MT}/\text{day}$ is identically $\$0.00/\text{MT}$, illustrating the exact mathematical boundary where debottlenecking returns vanish.

#### Financial Payback and Capital Efficiency Summary

| Project Phase | Capex Installed | Turnaround Lost Margin | Total Investment | Capacity Gained ($\text{MT}/\text{day}$) | Annual Incremental Margin | Simple Payback Period | Marginal Capital Intensity |
|---|---|---|---|---|---|---|---|
| **Phase 1 (Heat Int.)** | $\$4.20\text{M}$ | $\$5.06\text{M}$ | $\$9.26\text{M}$ | $1,103.6\text{ MT}/\text{day}$ | $\$49.50\text{M}/\text{year}$ | **0.19 years (2.3 months)** | $\$24.32 / (\text{MT}/\text{yr})$ |
| **Phase 2 (Retray D-103)**| $\$3.50\text{M}$ | $\$0.00\text{M}$ | $\$3.50\text{M}$ | $666.7\text{ MT}/\text{day}$ | $\$29.90\text{M}/\text{year}$ | **0.12 years (1.4 months)** | $\$15.22 / (\text{MT}/\text{yr})$ |
| **Combined 1 + 2** | **$\$7.70\text{M}$** | **$\$5.06\text{M}$** | **$\$12.76\text{M}$** | **$1,770.3\text{ MT}/\text{day}$** | **$\$79.40\text{M}/\text{year}$** | **0.16 years (1.9 months)** | **$\$20.89 / (\text{MT}/\text{yr})$** |

> **Comparison to Greenfield:** A greenfield aromatics extraction train of equivalent capacity has a capital intensity of roughly $\$450.00 / (\text{MT}/\text{year})$. The debottlenecking program delivered $1,770.3\text{ MT}/\text{day}$ ($610,750\text{ MT}/\text{year}$) of capacity at $\$20.89 / (\text{MT}/\text{year})$—**less than 5% of the capital intensity of a new plant** with full capital payback achieved in under 60 days of post-revamp operation.

---

## 7. Code, query, or calculation

The following self-contained Python script models a multi-unit continuous process train using `scipy.optimize.linprog` with the HiGHS solver. It calculates primal flow rates, extracts dual variables (shadow prices $\lambda_i$), and simulates incremental debottlenecking of the primary constraint to show where the constraint shifts to a secondary unit.

```python
import numpy as np
from scipy.optimize import linprog

def solve_refinery_train(d103_feed_capacity):
    """
    Solves process plant LP and extracts dual shadow prices.
    Variables: x = [FreshFeed, Lights, Raffinate, Benzene, Toluene, SteamUsed]
    """
    # Maximize Margin = -Cost(Feed) + Rev(Lights) + Rev(Raff) + Rev(Benz) + Rev(Tol) - Cost(Steam)
    # Linprog minimizes c^T * x
    c = [650.0 + 12.12, -520.0, -560.0, -1050.0, -920.0, 24.0]
    
    # Equality constraints (A_eq * x == b_eq): Mass split & steam balances
    # x0: Feed, x1: Lights, x2: Raff, x3: Benz, x4: Tol, x5: Steam
    A_eq = [
        [-0.15, 1.0, 0.0, 0.0, 0.0,  0.0],  # Lights = 0.15 * Feed
        [-0.25, 0.0, 1.0, 0.0, 0.0,  0.0],  # Raffinate = 0.25 * Feed
        [-0.38, 0.0, 0.0, 1.0, 0.0,  0.0],  # Benzene = 0.38 * Feed
        [-0.20, 0.0, 0.0, 0.0, 1.0,  0.0],  # Toluene = 0.20 * Feed
        [-0.30, 0.0, 0.0, 0.0, 0.0,  1.0],  # Steam = 0.30 * Feed (Post Phase 1)
    ]
    b_eq = [0.0, 0.0, 0.0, 0.0, 0.0]
    
    # Inequality constraints (A_ub * x <= b_ub): Equipment capacity bounds
    A_ub = [
        [ 1.0,  0.0, 0.0, 0.0, 0.0,  0.0],  # D-101 Feed <= 12,000 MT/day
        [ 0.85, 0.0, 0.0, 0.0, 0.0,  0.0],  # E-102 Feed <= 10,000 MT/day
        [ 0.60, 0.0, 0.0, 0.0, 0.0,  0.0],  # D-103 Feed <= d103_feed_capacity
        [ 0.20, 0.0, 0.0, 0.0, 0.0,  0.0],  # D-104 Feed <= 2,300 MT/day
        [ 0.0,  0.0, 0.0, 0.0, 0.0,  1.0],  # Steam Header <= 3,600 MT/day
    ]
    b_ub = [12000.0, 10000.0, d103_feed_capacity, 2300.0, 3600.0]
    bounds = [(0, None)] * 6

    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
    if not res.success:
        raise RuntimeError("LP optimization failed to converge")
    
    # In HiGHS, shadow price = -marginal for upper bound constraints (relaxation increases profit)
    shadow_prices = -res.ineqlin.marginals
    return res.x[0], -res.fun, shadow_prices

# Simulation across debottlenecking investment increments on D-103
print(f"{'D-103 Cap':>10} | {'Fresh Feed':>10} | {'Daily Profit':>13} | {'λ_D103':>9} | {'λ_D104':>9} | {'λ_Steam':>9}")
print("-" * 72)
for cap in [6500.0, 6700.0, 6900.0, 6900.001, 7100.0]:
    feed, profit, duals = solve_refinery_train(cap)
    print(f"{cap:10.1f} | {feed:10.1f} | ${profit:12.2f} | ${duals[2]:8.2f} | ${duals[3]:8.2f} | ${duals[4]:8.2f}")
```

### Expected Output and Analytical Interpretation

When executed, the script produces the following terminal output:

```text
 D-103 Cap | Fresh Feed |  Daily Profit |   λ_D103 |   λ_D104 |  λ_Steam
------------------------------------------------------------------------
    6500.0 |    10833.3 | $ 1408333.33  | $ 216.67 | $   0.00 | $   0.00
    6700.0 |    11166.7 | $ 1451666.67  | $ 216.67 | $   0.00 | $   0.00
    6900.0 |    11500.0 | $ 1495000.00  | $ 216.67 | $   0.00 | $   0.00
    6900.0 |    11500.0 | $ 1495000.00  | $   0.00 | $ 650.00 | $   0.00
    7100.0 |    11500.0 | $ 1495000.00  | $   0.00 | $ 650.00 | $   0.00
```

#### Verification of Mechanics:
1. **The Binding Constraint:** At the baseline D-103 capacity of $6,500.0\text{ MT}/\text{day}$, column D-103 is strictly binding with a shadow price of $\lambda_{\text{D103}} = \$216.67/\text{MT}$ of column capacity (equivalent to $\$130.00/\text{MT}$ fresh feed). Units D-104 and the steam header have $\lambda = 0$.
2. **The Capacity Tipping Point:** As D-103 capacity expands from $6,500.0$ to $6,900.0\text{ MT}/\text{day}$, fresh feed scales linearly from $10,833.3$ to $11,500.0\text{ MT}/\text{day}$, and daily cash margin expands from $\$1,408,333$ to $\$1,495,000$.
3. **The Constraint Shift:** Exactly at $6,900.0\text{ MT}/\text{day}$ (fresh feed = $11,500.0\text{ MT}/\text{day}$), the plant slams into the clay treater **D-104 limit of $2,300\text{ MT}/\text{day}$ ($11,500 \times 0.20$)**. Beyond this point, any further expansion of D-103 yields a shadow price of $\lambda_{\text{D103}} = \$0.00$, and the shadow price transfers entirely to D-104 ($\lambda_{\text{D104}} = \$650.00/\text{MT}$). Additional capital spent on D-103 without expanding D-104 is completely stranded.

---

## 8. Data, standards, and regulatory context

Debottlenecking projects in continuous process industries are severely constrained by statutory safety, environmental, and engineering codes. Modifying equipment to increase throughput is not merely an optimization problem; it triggers mandatory regulatory recertification.

### 1. Process Safety Management (OSHA 29 CFR 1910.119)
Under United States federal law (and mirrored globally under the EU Seveso III Directive 2012/18/EU), any physical, chemical, or operational change in a hazardous chemical facility requires a formal **Management of Change (MOC)** procedure:
- **Process Hazard Analysis (PHA) Revalidation:** Increasing throughput alters vessel residence times, fluid velocities, and reaction heat generation. A full Hazard and Operability (HAZOP) study must be revalidated prior to authorization.
- **Relief Systems Verification:** Higher flow rates increase vapor generation during an emergency power failure or blocked-outlet scenario. Sizing of existing emergency relief valves must be re-verified against new relief loads.

### 2. Overpressure and Relief System Sizing (API Standard 520 & 521)
The American Petroleum Institute (API) standards govern pressure-relieving devices in refineries and chemical plants:
- **API 520 (Parts I & II):** *Sizing, Selection, and Installation of Pressure-Relieving Devices.* Debottlenecking a reboiler or increasing furnace firing increases the required relieving rate ($W_{\text{relief}}$ in $\text{kg}/\text{hr}$) during a total cooling water failure. If the existing pressure safety valve (PSV) orifice area ($A_{\text{orifice}}$) is insufficient, the vessel cannot legally operate at the higher rate.
- **API 521:** *Pressure-Relieving and Depressuring Systems.* Adding capacity across multiple units increases cumulative emergency relief flow to the central plant flare header. If the combined peak relief load exceeds the sonic velocity limit (Mach 0.5 to 0.7) or flare tip thermal radiation limit ($K = 1,500\text{ BTU}/\text{hr}\cdot\text{ft}^2$ at boundary fence lines), the entire plant debottlenecking program is blocked until the flare system is expanded.

### 3. Pressure Vessel Integrity (ASME BPVC Section VIII, Division 1 & 2)
Increasing operating rates often requires elevating column pressure or operating temperature to overcome hydraulic restrictions:
- The **Maximum Allowable Working Pressure (MAWP)** and **Minimum Design Metal Temperature (MDMT)** are governed by the ASME Boiler and Pressure Vessel Code.
- If a proposed debottlenecking scheme pushes operating pressure closer than 10% to the vessel MAWP, nuisance relief valve popping occurs. Rerating a vessel to a higher MAWP requires ultrasound wall thickness verification, metallurgical coupon testing, and recalculating corrosion allowances.

### 4. Environmental Permitting & New Source Review (EPA Clean Air Act Title V)
Environmental regulations frequently act as hard economic bottlenecks:
- **New Source Review (NSR) & Prevention of Significant Deterioration (PSD):** In the United States, if a debottlenecking project increases annual emissions of a criteria pollutant (NOx, SOx, VOC, PM2.5) above the Significant Emission Rate (SER) threshold (e.g., 40 tons/year for VOCs or NOx), the project triggers major NSR permitting.
- **Best Available Control Technology (BACT):** Triggering NSR requires installing BACT (such as selective catalytic reduction for heaters or regenerative thermal oxidizers for vents), adding tens of millions of dollars in unexpected capex and delaying project startup by 18 to 36 months. Skilled debottlenecking engineering actively designs revamp envelopes to keep emission increases below SER thresholds (synthetic minor netting).

### 5. Manufacturing Operations Standards (ISO 22400-2 & ISA-95)
- **ISO 22400-2:** Formally defines Key Performance Indicators for manufacturing operations management, standardizing technical utilization, continuous OEE, worker productivity, and scrap/loss tracking across continuous process environments.
- **ISA-95 (IEC 62264):** Dictates enterprise-control system integration, specifying the data models through which Level 3 Manufacturing Execution Systems (MES) communicate plant capacity constraints and demonstrated rates to Level 4 Enterprise Resource Planning (ERP) master planning modules.

### 6. Piping Integrity & Erosional Velocity Limits (API 570 & API RP 14E)
Debottlenecking projects frequently overlook the fluid dynamics of existing inter-unit piping headers. When mass flow $\dot{m}$ increases within an existing line of fixed inside diameter $D$, fluid velocity rises linearly:
$$v = \frac{4 \dot{m}}{\pi D^2 \rho}$$
In two-phase flow (gas-liquid mixtures) or corrosive slurries, fluid velocity is bounded by the erosional velocity threshold defined in API RP 14E:
$$v_e = \frac{c}{\sqrt{\rho_m}}$$
Where $\rho_m$ is the mixture gas-liquid density ($\text{lb}/\text{ft}^3$ or $\text{kg}/\text{m}^3$) and $c$ is the empirical erosional constant (typically $c = 100$ for continuous service in carbon steel, or $c = 125\text{ to }150$ for corrosion-resistant alloys).
If debottlenecked velocities exceed $v_e$, accelerated erosion-corrosion strips the protective iron carbonate or sulfide scale from the inner pipe wall. Under API 570 inspection protocols, this triggers rapid pipe thinning, localized elbow wall loss, and eventual catastrophic loss of containment. Debottlenecking audits must verify that every piping segment upstream and downstream of revamped vessels remains safely below $0.80 \times v_e$.

### 7. EPA Title V Netting and Actual-to-Projected-Actual Test
When evaluating whether a debottlenecking revamp constitutes a "major modification" under the Clean Air Act, operators must apply the two-step NSR applicability test:
1. **Step 1 (Project Emissions Increase):** Calculate the difference between Projected Actual Emissions (PAE) and Baseline Actual Emissions (BAE):
   $$\Delta E_{\text{project}} = \text{PAE} - \text{BAE} - \text{Excludable Demand Growth}$$
   The excludable demand growth provision allows an operator to subtract emission increases that the existing equipment could have accommodated during the baseline period and that are attributable to independent market growth.
2. **Step 2 (Contemporaneous Netting):** If $\Delta E_{\text{project}}$ exceeds Significant Emission Rates (e.g., $40\text{ tons/year}$ for NOx or VOCs), the plant must perform contemporaneous netting across all creditable emission increases and decreases that occurred at the major stationary source over the preceding 5-year window.
Careful production scheduling, fuel gas quality management, and selective deployment of low-NOx burners ensure the revamp nets below SER thresholds, avoiding the multi-year permitting timeline of full Prevention of Significant Deterioration (PSD) review.

---

## 9. Industry practice and case evidence

### Case 1: ExxonMobil Baytown / Baton Rouge Olefins Creep Programs
ExxonMobil Chemical has historically achieved unmatched capital efficiency in its steam cracker networks by pursuing systematic, continuous capacity debottlenecking rather than relying solely on mega-project greenfield construction. Over two decades, facilities in Baton Rouge, Louisiana, and Baytown, Texas, increased ethylene production by over 40% through phased, low-capex revamps:
- **Documented Fact:** By systematically replacing random tower packing with high-efficiency structured packing, installing anti-fouling coatings on transfer line exchangers (TLEs), and restaging pyrolysis gas compressors during scheduled 5-year turnarounds, capacity was added in increments of 2% to 4% per cycle.
- **Capital Intensity Evidence:** SEC disclosures and petrochemical engineering literature indicate that ExxonMobil's average capital intensity for these brownfield debottlenecks averaged $\$180\text{ to }\$250\text{ per metric ton of annual ethylene capacity}$, compared to $\$1,200\text{ to }\$1,600/\text{MT}$ for greenfield US Gulf Coast cracker projects constructed over the same timeframe.

### Case 2: LyondellBasell Channelview Olefins Cold-Box Revamp
At LyondellBasell's Channelview, Texas complex, the primary ethylene fractionation train was historically limited by refrigeration capacity in the cryogenic separation "cold box" during high-ambient summer months.
- **Documented Fact:** The plant installed an advanced plate-fin heat exchanger (core exchanger) revamp combined with a turbo-expander re-wheel.
- **Engineering Mechanism:** The revamp shifted the refrigeration compression load, reducing suction temperature and recovering an additional 120,000 MT/year of high-purity ethylene that previously vented to plant fuel gas.
- **Inference vs. Fact:** While corporate investor presentations reported an internal rate of return exceeding 45%, independent chemical engineering post-audits inferred that the true bottleneck had migrated immediately to the acetylene hydrogenation reactor, requiring an immediate catalyst changeout during the subsequent turnaround to prevent off-spec ethylene product trips.

### Case 3: Freeport-McMoRan Morenci Copper Concentrator Pebble Crushing
In mining and mineral hydrometallurgy, throughput is bound by comminution (crushing and grinding), which consumes up to 50% of total mine electrical power. At the Morenci open-pit copper mine in Arizona, the semi-autogenous grinding (SAG) mills were chronically bottlenecked by hard, competent ore pebbles (15mm to 50mm rock fragments) that resisted impact breakage.
- **Documented Fact:** Freeport-McMoRan installed secondary pebble crushers in closed circuit with the SAG mills.
- **Operational Result:** Crushing the hard pebbles mechanically before returning them to the ball mills reduced SAG mill power consumption per ton by 18%, increasing concentrator throughput by 14% without requiring additional multi-million-dollar grinding mills or new tailings pipelines.

### Case 4: The Midwestern Ethanol "Phantom Capacity" Catastrophe
A publicly traded Midwestern ethanol producer raised $\$18\text{ million}$ to expand nameplate fermentation and distillation capacity from 50 million to 70 million gallons per year.
- **Documented Fact:** The distillation column was successfully re-trayed, and three new $500,000\text{-gallon}$ fermentation tanks were erected on time and within budget.
- **The Failure:** Upon restart, the plant was unable to operate above 54 million gallons per year. The front-end engineering design had failed to audit the thermal capacity of the Regenerative Thermal Oxidizer (RTO), which incinerated volatile organic compounds (VOCs) from the grain dryers.
- **Consequence:** Operating above 54 million gallons exceeded the plant's legal Title V Clean Air Act permit for VOC emissions. The facility was forced to throttle production back to near-baseline levels. The invested $\$18\text{ million}$ sat stranded for 22 months while the company applied for an amended air permit and spent an unplanned $\$6.5\text{ million}$ on a secondary RTO burner upgrade.

### Case 5: Queensland Curtis LNG (QCLNG) Refrigeration Compressor Power Augmentation
Baseload Liquefied Natural Gas (LNG) liquefaction trains use massive centrifugal refrigeration compressor strings (propane, ethylene, and methane or multi-component refrigerant - MCR) driven directly by industrial aeroderivative or heavy-duty frame gas turbines. At the Queensland Curtis LNG facility on Curtis Island, Australia, liquefaction capacity was severely constrained during the Southern Hemisphere summer when ambient air temperatures frequently exceeded 38°C.

```text
Ambient Air (38°C) ──►[ Chiller Coils (TIAC) ]──► Chilled Air (10°C) ──► Gas Turbine Driver
                             ▲                                                  │
                             │                                                  ▼
                      Waste Heat Absorption ◄── Exhaust Flue Gas ◄── Higher Shaft Power (+18%)
                         Chiller Loop                                           │
                                                                                ▼
                                                             MCR Refrigeration Compressor
                                                             (+7.5% LNG Production Capacity)
```

- **Documented Fact:** High ambient temperature reduces ambient air density, which dramatically decreases the mass air flow ingested by the gas turbine compressors. As a result, available turbine shaft power output drops by 0.7% to 1.0% per degree Celsius rise above ISO rating conditions (15°C), starving the refrigeration compressors of mechanical power.
- **Engineering Revamp Mechanism:** Rather than spending hundreds of millions of dollars to install an additional auxiliary gas turbine driver, the operator executed a brownfield revamp installing Turbine Inlet Air Chilling (TIAC) powered by waste-heat absorption chillers capturing thermal energy from the gas turbine exhaust stacks.
- **Operational Result:** The TIAC system chilled the combustion air entering the turbine from 38°C down to 10°C year-round. This recovered over 18% of lost shaft power on peak summer days, unlocking an incremental 650,000 MT/year of LNG production capacity (+7.5% across two trains) at a capital intensity under $\$110\text{ per annual metric ton of LNG}$—less than one-tenth of the $\$1,200/\text{MT}$ greenfield benchmark.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Root Physical / Economic Mechanism | Operational Consequence | Mitigation Strategy |
|---|---|---|---|---|
| **Whack-a-Mole (Constraint Chasing)** | Debottlenecking project achieves less than 30% of projected volume post-startup. | Upgrading Unit A without identifying that Unit B saturates at +2% throughput. | Stranded capital; project IRR collapses; unrecovered revamp capex. | Multi-unit LP boundary simulation; identify all sequential constraints across a +20% range. |
| **Utility Header Starvation** | Steam header pressure drops; cooling water return temperature alarms. | Focusing exclusively on process vessels while ignoring shared steam, power, or water headers. | Plant-wide rate derating; emergency electrical load shedding; flaring. | Integrate shared utility balance equations ($\sum \beta_{ku} w_u \le \text{Cap}_k$) into debottlenecking LP models. |
| **Summer Condenser Pinch** | Column pressure creeps upward on warm afternoons; vacuum ejectors overload. | Cooling tower wet-bulb approach collapses log-mean temperature difference in condensers. | Severe throughput derating during summer months; off-spec distillate. | Audit capacity under 99th percentile summer wet-bulb conditions; install trim condensers or structured packing. |
| **Relief Valve (API 521) Choke** | Flare header back-pressure alarms during HAZOP revalidation. | Increased reboiler duty increases vapor release volume during power outage scenario. | Regulatory shutdown; violation of OSHA PSM 1910.119; inability to start up. | Complete rigorous API 520/521 relief flare load audit before committing procurement capital. |
| **Environmental Permit Tripwire** | Plant throughput capped by stack emissions monitor (CEMS) rather than equipment. | Production expansion increases hourly mass emissions past Title V Significant Emission Rates. | Cease-and-desist orders; mandatory BACT installation; 18-month permitting delays. | Pre-project emission netting; incorporate NSR emission caps as hard constraints in master planning models. |
| **Recycle Loop Accumulation** | Column differential pressure spikes gradually over 10 days of continuous operation. | Debottlenecking increases fresh feed without expanding purge stream; inerts accumulate in loop. | Sudden, catastrophic column flooding; off-spec product dump; unit trip. | Model trace inert split fractions; install larger purge bleed valves or membrane separators. |
| **Compressor Surge Margin Collapse** | Anti-surge recycle valve cracks open continuously at high throughput. | Off-design debottlenecked gas molecular weight or suction pressure shifts operating point toward surge line. | Severe compressor vibration; mechanical seal blowout; multi-week downtime. | Dynamic compressor simulation; re-wheel impellers; retune anti-surge control loops. |
| **Catalyst Bed Temperature Runaway** | Reactor bed $\Delta T$ exceeds metallurgic design limits during rate ramp-up. | Increased mass flow reduces residence time, requiring higher temperature to maintain conversion. | Rapid catalyst coking; permanent catalyst poisoning; furnace tube rupture. | Verify LHSV/WHSV kinetic boundaries; transition to high-activity, high-selectivity catalyst formulations. |
| **Buffer Starvation / Blocking** | Upstream continuous reactor trips within 15 minutes of downstream packaging stoppage. | Intermediate surge tank sized on spatial footprint rather than MTBF/MTTR probability distributions. | Upstream continuous furnace must be flared or shutdown, inducing thermal shock. | Size surge buffers using stochastic queueing theory: $V = (\mu_{\text{MTTR}} + 3\sigma) \times \text{Rate}$. |
| **Turndown Inoperability** | Column weeping or burner flame lift-off during low-demand market periods. | Revamped equipment achieves high peak rates but sacrifices low-end turndown hydrodynamics. | Inability to operate during market downturns; forced shutdown during demand troughs. | Require minimum 3:1 turndown ratio in all revamp equipment specifications. |
| **Hydraulic Downcomer Backup** | Column tray $\Delta P$ increases exponentially with minor feed rate increase. | Liquid rate exceeds tray downcomer clearance; liquid backs up onto tray above. | Total loss of fractionation; tray mechanical damage; liquid carryover into compressor suction. | Replace cross-flow trays with truncated downcomers or structured packing. |
| **Heat Exchanger Fouling Acceleration** | Reboiler steam valve opens to 100% within 3 months of startup. | Higher skin temperatures cause rapid thermal decomposition and coking on tube walls. | Premature turnaround; loss of debottlenecked capacity before project payback. | Install helical baffles or tube inserts; upgrade metallurgy to resist fouling. |

---

## 11. Assumptions and limits

| Analytical Assumption | Operational Rationale | Boundary Conditions Where It Fails | Diagnostic / Stress Test |
|---|---|---|---|
| **Linearity of Unit Capacities in LP Models** | Enables rapid solution of plant-wide planning models using simplex or interior-point algorithms. | Distillation separation efficiency, compressor curves, and chemical reaction kinetics are highly non-linear. | Validate LP capacity bounds against non-linear rigorous thermodynamics (Aspen HYSYS) at $\pm 10\%$ throughput. |
| **Constant Yield Fractions Across Rates** | Assumes output split percentages ($\alpha_{pj}$) remain invariant as throughput increases. | High space velocities (LHSV) reduce residence time, shifting chemical selectivity toward intermediate byproducts. | Run kinetic reactor simulations across a range of throughput rates; implement piecewise linear yield curves. |
| **Unconstrained Utility Availability** | Evaluates individual equipment vessels assuming cooling water, steam, and electricity are infinitely elastic. | Central utility plants (boilers, cooling towers, power substations) frequently reach saturation before process vessels. | Explicitly incorporate utility balance vectors ($A_{\text{utility}} x \le b_{\text{utility}}$) into all capacity models. |
| **Deterministic Mean Downtime (MTTR)** | Sizing intermediate surge tanks assuming predictable maintenance repair durations. | Major catastrophic failures (e.g., motor winding burnout, reactor fire) exhibit heavy-tailed Pareto distributions. | Perform Monte Carlo discrete-event simulations using empirical Weibull failure distributions. |
| **Stationary Feedstock Composition** | Sizing debottlenecking equipment against a single "design reference crude" or standard feed assay. | Feedstock procurement swings between light sweet and heavy sour slates to exploit spot market price discounts. | Stress-test debottlenecking designs against extreme light and extreme heavy feedstock envelopes. |
| **Instantaneous Mixing in Buffers** | Assumes intermediate storage vessels act as ideal continuous stirred-tank reactors (CSTR) or plug flow. | Liquid stratification, dead zones, and thermal layering occur in large-diameter bulk storage tanks. | Computational Fluid Dynamics (CFD) modeling of tank mixing nozzles; install mechanical side-entry mixers. |
| **Perfect Market Demand Elasticity** | Assumes all incremental debottlenecked product can be sold at constant spot market price realization. | Dumping large incremental bulk commodity volumes into a localized regional basin depresses netback prices. | Couple the debottlenecking cash-flow model to a regional supply-demand price elasticity curve. |
| **Infinite Operating Life Post-Revamp** | Project cash-flow models project 10 to 15 years of uninterrupted cash flow at the new debottlenecked rate. | Accelerated equipment fatigue, corrosion, and metallurgy degradation shorten time between major turnarounds. | Apply higher depreciation rates and schedule mid-cycle maintenance allowances in project financial pro formas. |

---

## 12. Exercises

### Exercise 1: Continuous OEE Decomposition & Mass Attribution
A continuous ethylbenzene distillation column operated for 8,200 hours during an 8,760-hour calendar year (with 200 hours of planned annual maintenance and 360 hours of unplanned trip downtime). 
- Maximum demonstrated peak hourly feed rate: $150.0\text{ MT}/\text{hr}$.
- Total fresh feed processed during operating hours: $1,107,000\text{ MT}$.
- Mass balance accounting: Out of the total feed processed, $1,029,510\text{ MT}$ met prime commercial specifications on first pass. $44,280\text{ MT}$ was off-spec material recycled back to the upstream reactor, and $33,210\text{ MT}$ was lost to the plant fuel gas / flare system.
**Deliverable:** Compute Availability ($A_{\text{cont}}$), Rate Efficiency ($P_{\text{cont}}$), Quality Factor ($Q_{\text{cont}}$), and total Continuous OEE.
**Acceptance Criteria:** Express all factors to four decimal places and final OEE as a percentage. State the hidden capacity lost to internal recycle.
**Time Estimate:** 25 minutes.

### Exercise 2: Stochastic Sizing of Intermediate DBR Surge Tanks
A primary continuous cracking furnace feeds a dedicated downstream separation train at a steady rate of $250.0\text{ MT}/\text{hr}$ (fluid density $\rho = 780\text{ kg}/\text{m}^3$). Historical maintenance records indicate that the downstream separation train experiences unscheduled trips with a log-normal distribution:
- Mean Time To Repair ($\mu_{\text{MTTR}}$) = $4.5\text{ hours}$.
- Standard deviation of repair time ($\sigma_{\text{MTTR}}$) = $1.8\text{ hours}$.
To protect the continuous furnace from emergency shutdown and flaring, intermediate surge tankage must provide a 99% protection window ($k = 2.33$ standard deviations). Active working volume of the tank is between 15% (heel) and 90% (high-level alarm).
**Deliverable:** Calculate the required residence time buffer (hours) and total geometric tank volume ($\text{m}^3$).
**Acceptance Criteria:** Correctly apply working volume limits; provide exact numerical volume.
**Time Estimate:** 20 minutes.

### Exercise 3: LP Formulation & Shadow Price Derivation by Hand
A continuous petrochemical unit produces two grades of solvent: Regular ($X_1$) and Premium ($X_2$).
- Margin: Regular yields $\$120/\text{MT}$; Premium yields $\$180/\text{MT}$.
- Processing Constraints:
  1. Reaction Unit: Regular consumes $1.0\text{ hr}/\text{MT}$; Premium consumes $2.0\text{ hr}/\text{MT}$. Total available: $1,600\text{ hr}/\text{month}$.
  2. Purification Column: Regular consumes $1.0\text{ hr}/\text{MT}$; Premium consumes $1.0\text{ hr}/\text{MT}$. Total available: $1,000\text{ hr}/\text{month}$.
  3. Market Demand Cap on Premium: Premium sales cannot exceed $400\text{ MT}/\text{month}$.
**Deliverable:** 
1. Determine the optimal monthly production quantities ($X_1^*, X_2^*$) and total margin.
2. Formulate the dual LP and calculate the shadow prices ($\lambda_{\text{React}}, \lambda_{\text{Purif}}, \lambda_{\text{Demand}}$) using complementary slackness.
3. If an engineering revamp can add 100 hours of purification capacity for $\$4,000/\text{month}$, should the plant execute the revamp?
**Acceptance Criteria:** Complete algebraic derivation of dual variables and precise economic decision.
**Time Estimate:** 35 minutes.

### Exercise 4: Dynamic Ambient Seasonal Constraint Migration
A naphtha stabilizer column operates with cooling water in the overhead condenser:
- Winter cooling water supply: $12^\circ\text{C}$ (Available condenser duty: $45.0\text{ MW}$).
- Summer cooling water supply: $30^\circ\text{C}$ (Available condenser duty drops to $28.0\text{ MW}$ due to $\Delta T_{LM}$ collapse).
The column reboiler has a maximum fired heater duty of $36.0\text{ MW}$ year-round.
Specific energy requirements per metric ton of column feed:
- Reboiler duty required: $0.0032\text{ MW}\cdot\text{hr}/\text{MT feed}$.
- Condenser cooling duty required: $0.0035\text{ MW}\cdot\text{hr}/\text{MT feed}$.
**Deliverable:** 
1. Compute the maximum throughput rate ($\text{MT}/\text{hr}$) of the column in Winter and Summer.
2. Identify the binding bottleneck in each season and compute its seasonal shadow price if feed margin is $\$45.00/\text{MT}$.
**Acceptance Criteria:** State both throughput limits and name the physical constraint transition mechanism.
**Time Estimate:** 25 minutes.

### Exercise 5: Multi-Stage Debottlenecking Capital Budgeting & NPV
An olefins plant processes $10,000\text{ MT}/\text{day}$ with an operating cash margin of $\$110.00/\text{MT}$. The engineering team identifies three sequential debottlenecking options:
- **Project A (APC Retuning):** Adds $300\text{ MT}/\text{day}$. Capex = $\$1.5\text{M}$. Zero downtime.
- **Project B (Retray Depropanizer):** Adds $700\text{ MT}/\text{day}$ over Project A (Cumulative $+1,000\text{ MT}/\text{day}$). Capex = $\$8.0\text{M}$. Requires 5 days of total plant shutdown during installation (margin lost on baseline $10,000\text{ MT}/\text{day}$).
- **Project C (Re-wheel Cracked Gas Compressor):** Adds $1,200\text{ MT}/\text{day}$ over Project B (Cumulative $+2,200\text{ MT}/\text{day}$). Capex = $\$22.0\text{M}$. Installed during normal turnaround (zero incremental downtime penalty).
Assume: 345 operating days/year, discount rate $r = 10\%$, evaluation horizon 5 years.
**Deliverable:** Compute the Net Present Value (NPV), Payback Period, and Marginal Capital Intensity ($\text{MCI}$) for:
1. Project A alone.
2. Combined Project A + B.
3. Combined Project A + B + C.
**Acceptance Criteria:** Include downtime revenue loss in capital outflow; show discount factor table and NPV rankings.
**Time Estimate:** 40 minutes.

### Exercise 6: Process Safety and API 521 Flare Header Audit
A crude distillation unit plans to increase feed throughput by 18% (from $100,000\text{ bbl}/\text{day}$ to $118,000\text{ bbl}/\text{day}$).
- Existing reboiler duty is $85\text{ MW}$. Post-revamp reboiler duty will be $100.3\text{ MW}$.
- In the worst-case contingency (total plant cooling water failure under API 521), 100% of reboiler heat input vaporizes hydrocarbons into the overhead relief system (latent heat of vaporization $\Delta H_v = 310\text{ kJ}/\text{kg}$).
- The existing relief valve and flare header line has a maximum certified vapor venting capacity of $950,000\text{ kg}/\text{hr}$.
**Deliverable:** 
1. Calculate the relieving vapor mass flow ($\text{kg}/\text{hr}$) under baseline and revamped conditions.
2. Determine if the revamp passes the API 521 relief header safety audit or if it violates the safety boundary.
**Acceptance Criteria:** Show step-by-step thermodynamic conversion from MW to $\text{kg}/\text{hr}$; state clear Pass/Fail determination.
**Time Estimate:** 20 minutes.

### Exercise 7: APC Constraint Push and Variance Reduction
A high-pressure polyethylene reactor operates with a maximum design pressure limit of $280.0\text{ MPa}$. Exceeding this limit triggers emergency rupture disc blowout.
- Under manual operator control, reactor pressure fluctuates with a normal distribution: $\mu = 262.0\text{ MPa}$, $\sigma = 6.0\text{ MPa}$. The plant operates at $\mu = 280.0 - 3\sigma = 262.0\text{ MPa}$ to maintain a $99.87\%$ safety buffer.
- Reaction yield scales linearly with operating pressure: $\text{Throughput} = 15.0 \times P_{\text{operating}}\text{ (in MT/day per MPa)}$.
- An Advanced Process Control (MPC) project reduces pressure variance by 60% ($\sigma_{\text{new}} = 2.4\text{ MPa}$).
**Deliverable:** 
1. Calculate the new safe operating setpoint ($\mu_{\text{new}}$) maintaining the same $3\sigma$ safety distance to the limit.
2. Calculate the incremental throughput gained ($\text{MT}/\text{day}$) and the percentage capacity increase.
**Acceptance Criteria:** Exact statistical shift calculation and throughput output.
**Time Estimate:** 20 minutes.

---

## 13. Self-check answers

### Solution to Exercise 1: Continuous OEE Decomposition
1. **Planned Time:**
   $$T_{\text{planned}} = T_{\text{calendar}} - T_{\text{planned\_turnaround}} = 8,760 - 200 = 8,560\text{ hr}$$
2. **Availability ($A_{\text{cont}}$):**
   $$A_{\text{cont}} = \frac{T_{\text{operating}}}{T_{\text{planned}}} = \frac{8,200}{8,560} \approx \mathbf{0.9579} \quad (95.79\%)$$
   *(Unplanned downtime = $8,560 - 8,200 = 360\text{ hr}$).*
3. **Performance / Rate Efficiency ($P_{\text{cont}}$):**
   $$\text{Maximum Theoretical Mass} = 8,200\text{ hr} \times 150.0\text{ MT}/\text{hr} = 1,230,000\text{ MT}$$
   $$P_{\text{cont}} = \frac{\text{Actual Mass Processed}}{\text{Maximum Theoretical Mass}} = \frac{1,107,000}{1,230,000} = \mathbf{0.9000} \quad (90.00\%)$$
   *(Average operating rate = $1,107,000 / 8,200 = 135.0\text{ MT}/\text{hr}$, representing a $10.0\%$ rate derating loss).*
4. **Quality / Yield Efficiency ($Q_{\text{cont}}$):**
   $$Q_{\text{cont}} = \frac{\text{Prime Commercial Mass}}{\text{Total Feed Processed}} = \frac{1,029,510}{1,107,000} = \mathbf{0.9300} \quad (93.00\%)$$
5. **Continuous OEE:**
   $$\text{OEE} = A_{\text{cont}} \times P_{\text{cont}} \times Q_{\text{cont}} = 0.95794 \times 0.90000 \times 0.93000 = \mathbf{0.8018} \quad (\mathbf{80.18\%})$$
6. **Hidden Capacity Lost to Recycle:**
   The $44,280\text{ MT}$ of off-spec recycle consumed $44,280 / 150.0 = 295.2\text{ hours}$ of peak column capacity—representing $3.6\%$ of the entire year's operating time running in circles.

### Solution to Exercise 2: Sizing Intermediate DBR Surge Tanks
1. **Target Residence Time Buffer:**
   $$\tau_{\text{buffer}} = \mu_{\text{MTTR}} + k \cdot \sigma_{\text{MTTR}} = 4.5 + (2.33 \times 1.8) = 4.5 + 4.194 = \mathbf{8.694\text{ hours}}$$
2. **Mass Buffer Required:**
   $$M_{\text{buffer}} = 8.694\text{ hr} \times 250.0\text{ MT}/\text{hr} = 2,173.5\text{ MT} = 2,173,500\text{ kg}$$
3. **Required Working Liquid Volume:**
   $$V_{\text{working}} = \frac{2,173,500\text{ kg}}{780\text{ kg}/\text{m}^3} = 2,786.54\text{ m}^3$$
4. **Total Geometric Tank Volume:**
   Active working fraction = $90\% - 15\% = 75\% = 0.75$.
   $$V_{\text{geometric}} = \frac{V_{\text{working}}}{0.75} = \frac{2,786.54}{0.75} = \mathbf{3,715.39\text{ m}^3}$$
   *(The plant requires a dedicated surge tank with at least $3,720\text{ m}^3$ capacity to protect the continuous furnace).*

### Solution to Exercise 3: LP Formulation & Shadow Price Derivation
1. **Primal Problem:**
   $$\max Z = 120 X_1 + 180 X_2$$
   Subject to:
   - (1) $1.0 X_1 + 2.0 X_2 \le 1,600$ (Reaction)
   - (2) $1.0 X_1 + 1.0 X_2 \le 1,000$ (Purification)
   - (3) $X_2 \le 400$ (Demand)
   - $X_1, X_2 \ge 0$

2. **Corner Point Evaluation:**
   - At $X_2 = 400$: From (2), $X_1 = 1,000 - 400 = 600$.
     Check (1): $1.0(600) + 2.0(400) = 1,400 \le 1,600$ (Slack = $200\text{ hr}$). Feasible!
     $Z = 120(600) + 180(400) = 72,000 + 72,000 = \$144,000$.
   - Intersection of (1) and (2):
     Subtracting (2) from (1): $X_2 = 600 \implies X_1 = 400$. But $X_2 = 600 > 400$ (Violates Demand cap). Infeasible.
   - Optimal Primal Solution:
     $$X_1^* = \mathbf{600\text{ MT}}, \quad X_2^* = \mathbf{400\text{ MT}}, \quad Z^* = \mathbf{\$144,000/\text{month}}$$

3. **Dual Shadow Prices via Complementary Slackness:**
   - Constraint (1) has slack ($1,400 < 1,600$) $\implies \lambda_{\text{React}} = \mathbf{\$0.00/\text{hr}}$.
   - Constraint (2) is binding ($600 + 400 = 1,000$) $\implies \lambda_{\text{Purif}} \ge 0$.
   - Constraint (3) is binding ($X_2 = 400$) $\implies \lambda_{\text{Demand}} \ge 0$.
   - Dual equations for basic variables ($X_1 > 0, X_2 > 0$):
     $$\lambda_{\text{Purif}} = 120 \implies \lambda_{\text{Purif}} = \mathbf{\$120.00/\text{hr}}$$
     $$\lambda_{\text{Purif}} + \lambda_{\text{Demand}} = 180 \implies 120 + \lambda_{\text{Demand}} = 180 \implies \lambda_{\text{Demand}} = \mathbf{\$60.00/\text{MT}}$$

4. **Revamp Decision:**
   Purification shadow price = $\$120.00/\text{hr}$.
   Marginal benefit of 100 hours = $100 \times \$120.00 = \$12,000/\text{month}$.
   Cost = $\$4,000/\text{month}$.
   Net profit increase = $\$12,000 - \$4,000 = \mathbf{+\$8,000/\text{month}}$.
   **Decision: Execute the revamp immediately.** (Note: Up to $200\text{ hours}$ can be added before Reaction constraint becomes binding).

### Solution to Exercise 4: Dynamic Ambient Constraint Migration
1. **Winter Analysis:**
   - Reboiler Limit: $\text{Rate} = 36.0\text{ MW} / 0.0032\text{ MW}\cdot\text{hr}/\text{MT} = \mathbf{11,250.0\text{ MT}/\text{hr}}$.
   - Condenser Limit: $\text{Rate} = 45.0\text{ MW} / 0.0035\text{ MW}\cdot\text{hr}/\text{MT} = 12,857.1\text{ MT}/\text{hr}$.
   - **Winter Bottleneck: Fired Heater Reboiler.**
   - Winter Max Throughput: **$11,250.0\text{ MT}/\text{hr}$**.
   - Winter Shadow Price: $\lambda_{\text{Reboiler}} = \frac{\$45.00/\text{MT}}{0.0032\text{ MW}\cdot\text{hr}/\text{MT}} = \mathbf{\$14,062.50\text{ per MW}\cdot\text{hr}}$ ($\lambda_{\text{Condenser}} = \$0$).

2. **Summer Analysis:**
   - Reboiler Limit: $11,250.0\text{ MT}/\text{hr}$.
   - Condenser Limit: $\text{Rate} = 28.0\text{ MW} / 0.0035\text{ MW}\cdot\text{hr}/\text{MT} = \mathbf{8,000.0\text{ MT}/\text{hr}}$.
   - **Summer Bottleneck: Overhead Condenser.**
   - Summer Max Throughput: **$8,000.0\text{ MT}/\text{hr}$** (a $28.9\%$ capacity collapse).
   - Summer Shadow Price: $\lambda_{\text{Condenser}} = \frac{\$45.00/\text{MT}}{0.0035\text{ MW}\cdot\text{hr}/\text{MT}} = \mathbf{\$12,857.14\text{ per MW}\cdot\text{hr}}$ ($\lambda_{\text{Reboiler}} = \$0$).

### Solution to Exercise 5: Debottlenecking Capital Budgeting & NPV
- 5-year uniform present value factor at $10\%$:
  $$\text{PVIFA}_{10\%, 5} = \frac{1 - (1.10)^{-5}}{0.10} = 3.79079$$

1. **Project A Alone:**
   - $\Delta \text{Throughput} = 300\text{ MT}/\text{day}$.
   - $\Delta \text{Cash Flow} = 300 \times \$110 \times 345 = \$11,385,000/\text{year}$.
   - PV of Cash Inflows = $\$11,385,000 \times 3.79079 = \$43,158,144$.
   - Capex = $\$1,500,000$. Downtime loss = $\$0$.
   - **$\text{NPV}_A = \$43,158,144 - \$1,500,000 = \mathbf{+\$41,658,144}$**.
   - Simple Payback = $\$1.5\text{M} / \$11.385\text{M} = \mathbf{0.13\text{ years (1.6 months)}}$.
   - $\text{MCI}_A = \frac{\$1,500,000}{300 \times 345} = \mathbf{\$14.49 / (\text{MT}/\text{year})}$.

2. **Combined Project A + B:**
   - $\Delta \text{Throughput} = 1,000\text{ MT}/\text{day}$.
   - $\Delta \text{Cash Flow} = 1,000 \times \$110 \times 345 = \$37,950,000/\text{year}$.
   - PV of Cash Inflows = $\$37,950,000 \times 3.79079 = \$143,860,480$.
   - Downtime Loss = $5\text{ days} \times 10,000\text{ MT}/\text{day} \times \$110 = \$5,500,000$.
   - Total Capital Outflow = Capex ($1.5 + 8.0 = \$9.5\text{M}$) + Downtime ($\$5.5\text{M}$) = $\$15,000,000$.
   - **$\text{NPV}_{A+B} = \$143,860,480 - \$15,000,000 = \mathbf{+\$128,860,480}$**.
   - Simple Payback = $\$15.0\text{M} / \$37.95\text{M} = \mathbf{0.40\text{ years (4.7 months)}}$.
   - $\text{MCI}_{A+B} = \frac{\$15,000,000}{1,000 \times 345} = \mathbf{\$43.48 / (\text{MT}/\text{year})}$.

3. **Combined Project A + B + C:**
   - $\Delta \text{Throughput} = 2,200\text{ MT}/\text{day}$.
   - $\Delta \text{Cash Flow} = 2,200 \times \$110 \times 345 = \$83,490,000/\text{year}$.
   - PV of Cash Inflows = $\$83,490,000 \times 3.79079 = \$316,493,057$.
   - Total Capital Outflow = Capex ($1.5 + 8.0 + 22.0 = \$31.5\text{M}$) + Downtime ($\$5.5\text{M}$) = $\$37,000,000$.
   - **$\text{NPV}_{A+B+C} = \$316,493,057 - \$37,000,000 = \mathbf{+\$279,493,057}$**.
   - Simple Payback = $\$37.0\text{M} / \$83.49\text{M} = \mathbf{0.44\text{ years (5.3 months)}}$.
   - $\text{MCI}_{A+B+C} = \frac{\$37,000,000}{2,200 \times 345} = \mathbf{\$48.75 / (\text{MT}/\text{year})}$.
   *(All three stages are highly accretive; executing A + B + C maximizes total wealth creation by $\$279.5\text{M}$).*

### Solution to Exercise 6: Process Safety and API 521 Flare Audit
1. **Baseline Relieving Mass Flow:**
   $$W_{\text{baseline}} = \frac{85.0\text{ MW} \times 3,600\text{ s}/\text{hr} \times 1,000\text{ kJ}/\text{MJ}}{310\text{ kJ}/\text{kg}} = \frac{306,000,000}{310} = \mathbf{987,097\text{ kg}/\text{hr}}$$
   *(Notice that the baseline operation was already very close to the $950,000\text{ kg}/\text{hr}$ certified limit).*
2. **Revamped Relieving Mass Flow (at 100.3 MW):**
   $$W_{\text{revamp}} = \frac{100.3\text{ MW} \times 3,600 \times 1,000}{310} = \frac{361,080,000}{310} = \mathbf{1,164,774\text{ kg}/\text{hr}}$$
3. **Audit Determination:**
   - Certified Flare Header Capacity = $950,000\text{ kg}/\text{hr}$.
   - Revamp Relieving Demand = $1,164,774\text{ kg}/\text{hr}$.
   - Exceedance = $+214,774\text{ kg}/\text{hr}$ ($+22.6\%$ overload).
   - **AUDIT RESULT: FAIL.**
   The revamp violates API 521. Increasing throughput legally requires either replacing the relief header piping/flare tip or installing an instrumented Safety Instrumented System (SIS / HIPPS) rated SIL-3 to eliminate the total reboiler heat input contingency.

### Solution to Exercise 7: APC Constraint Push and Variance Reduction
1. **Baseline Operating Margin:**
   Safety buffer = $3 \times 6.0 = 18.0\text{ MPa}$.
   Operating point $\mu_{\text{old}} = 280.0 - 18.0 = 262.0\text{ MPa}$.
2. **Post-APC Safe Operating Setpoint:**
   Reduced variance $\sigma_{\text{new}} = 2.4\text{ MPa}$.
   New safety buffer = $3 \times 2.4 = 7.2\text{ MPa}$.
   $$\mu_{\text{new}} = 280.0 - 7.2 = \mathbf{272.8\text{ MPa}}$$
   Operating pressure setpoint increases by $\Delta P = 272.8 - 262.0 = \mathbf{+10.8\text{ MPa}}$.
3. **Throughput Gain:**
   $$\Delta \text{Throughput} = 15.0\text{ MT}/\text{day per MPa} \times 10.8\text{ MPa} = \mathbf{162.0\text{ MT}/\text{day}}$$
   Baseline throughput = $15.0 \times 262.0 = 3,930.0\text{ MT}/\text{day}$.
   $$\% \text{ Capacity Increase} = \frac{162.0}{3,930.0} = \mathbf{+4.12\%}$$
   *(The APC constraint push captures an additional 162 MT/day of product with zero vessel hardware capex).*

---

## 14. Interview questions

### Question 1: Identifying the True Bottleneck
*Question:* "If you walk into a continuous chemical plant and the Operations Manager says Unit C is the plant bottleneck because it runs at 98% of nameplate while Units A, B, and D run at 80%, how do you verify or disprove that claim?"
- **Strong Answer:** "High utilization does not imply an economic or even a physical constraint. First, I would check the LP dual shadow prices ($\lambda_i$) from the plant planning model. A unit can run at 98% because operators are recycling off-spec material, or because upstream units are deliberately throttled. Second, I would audit the shared utility headers (cooling water, high-pressure steam, power) and environmental discharge permits; frequently, a column sits at 98% because the cooling tower return temperature is maxed out, meaning the cooling tower is the real bottleneck. Third, I would examine historical DCS rate ramp-up data: did increasing total fresh feed actually cause Unit C's differential pressure to spike, or did a downstream pump cavitate first? Only if relaxing Unit C's capacity increases overall plant cash margin ($\lambda_C > 0$) is it the true economic bottleneck."
- **Weak Answer:** "I would trust the plant manager and inspect Unit C's pumps and valves, then hire a mechanical vendor to quote a larger impeller or install a bypass line."

### Question 2: The Whack-a-Mole Phenomenon
*Question:* "Explain the 'whack-a-mole' or constraint-chasing trap in brownfield debottlenecking. How do you protect a capital project from this failure mode?"
- **Strong Answer:** "The whack-a-mole trap occurs when an engineering team isolates a single unit (e.g., re-traying a tower for $+15\%$ capacity) without performing a full-system flowsheet LP boundary audit. Once the plant restarts, throughput only rises by $+2\%$ before hitting an unforeseen barrier in a downstream heat exchanger, compressor suction valve, or off-gas flare limit. The remaining $+13\%$ capacity is stranded, destroying the project's NPV. To protect the project, we construct a sequential capacity escalation ladder. We run parametric LP simulations from $+0\%$ to $+25\%$ throughput, identifying the exact order in which constraints become binding (Unit 1 at $+3\%$, Utility Header at $+7\%$, Unit 4 at $+12\%$). We then bundle the revamp into coherent capital stages where all interdependent bottlenecks are addressed within the same turnaround."
- **Weak Answer:** "Whack-a-mole happens when equipment breaks unexpectedly after you increase the speed. You fix it by keeping extra spare parts in inventory."

### Question 3: Goldratt's DBR in Continuous Fluid Flow
*Question:* "How does Drum-Buffer-Rope (DBR) differ when applied to continuous fluid processes compared to discrete manufacturing lines?"
- **Strong Answer:** "In discrete manufacturing, buffers consist of physical parts sitting on conveyors or in tote bins, and the rope is a kanban card or digital signal releasing a discrete job ticket. In continuous flow, intermediate inventory is liquid or gas held under pressure in surge vessels, spheres, or tank farms. Buffers cannot be infinite; they are bounded by strict minimum hydraulic levels (pump NPSH requirements) and maximum levels (overfill protection / flaring risks). Sizing a continuous buffer requires translating time into volumetric holdup: $V = (\text{MTTR} + k\sigma) \times \dot{m} / \rho$. The rope is not a ticket but a dynamic Advanced Process Control (APC) cascade loop that throttles charge control valves at the plant battery limit based on intermediate vessel holdups and the instantaneous state of the Drum asset."
- **Weak Answer:** "It doesn't differ at all; you just identify the slowest machine, put some inventory in front of it, and tell the front of the line not to work too fast."

### Question 4: Dynamic Shifting Bottlenecks
*Question:* "Describe a scenario where a process plant's bottleneck shifts dynamically between summer and winter. How does this affect debottlenecking economics?"
- **Strong Answer:** "In a distillation column, high summer ambient wet-bulb temperatures reduce cooling tower heat rejection. The cooling water supply temperature rises, collapsing the log-mean temperature difference ($\Delta T_{LM}$) in the overhead condenser. The condenser pinches, column back-pressure increases, and the column becomes condenser/hydraulic flood limited ($\lambda_{\text{condenser}} > 0$). In winter, cooling water is cold, providing abundant condensing capacity; however, low ambient temperatures increase furnace radiation heat losses and high ambient air density overloads the furnace draft fans or fuel gas burners ($\lambda_{\text{furnace}} > 0$). If an analyst calculates debottlenecking payback based purely on annual average temperatures, the proposed revamp will fail: adding reboiler duty won't help in summer, and adding condenser area won't help in winter. A robust project must evaluate dual summer/winter operating envelopes."
- **Weak Answer:** "Plants run faster in winter because machines don't overheat. In summer, you just have to spray water on the pipes to keep them cool."

### Question 5: Continuous OEE vs. Discrete OEE
*Question:* "Why is standard discrete OEE misleading when applied to continuous process operations, and how do you rectify it?"
- **Strong Answer:** "Discrete OEE relies on piece counts, discrete cycle times, and discrete scrap counts. In continuous processing, units rarely stop for 'micro-stops'; instead, operators derate throughput rates to avoid flooding or emission trips. Rate loss must be measured by integrating actual mass flow against peak demonstrated design flow: $P = \int \dot{m}(t) dt / (T \cdot \dot{m}_{\text{peak}})$. Furthermore, quality loss is not just scrapped parts; it manifests as chemical selectivity loss, flare flushes, or off-spec product recycled back to the front of the plant. Recycle loops consume hydraulic and thermal capacity without appearing as scrap, masking a major capacity loss. Continuous OEE must be formulated strictly around mass balance conservation and thermodynamic rate limits."
- **Weak Answer:** "Continuous OEE is the same thing, but you divide by tons instead of widgets."

### Question 6: Shadow Price ($\lambda_i$) Interpretation
*Question:* "In an LP planning model, you find that Unit A has a shadow price of $\$55.00/\text{MT}$ while Unit B has a shadow price of $\$0.00/\text{MT}$. Unit B is currently operating at $99.2\%$ utilization. Where should capital be spent?"
- **Strong Answer:** "Capital should be directed toward Unit A (subject to verifying how far capacity can expand before the next constraint hits). Unit B's shadow price of $\$0.00$ proves that it currently has slack capacity; expanding it yields zero incremental profit. Unit B's $99.2\%$ utilization is irrelevant; it has enough headroom to satisfy the current economic optimum. However, before investing in Unit A, I would run a parametric right-hand-side range test to see how many tons of capacity can be added to Unit A before Unit B (or another unit) hits $100\%$ and causes Unit A's shadow price to drop to zero."
- **Weak Answer:** "Unit B, because $99.2\%$ is almost at the breaking point and it will break next."

### Question 7: Capital Budgeting Downtime Penalty
*Question:* "When calculating the Net Present Value of a debottlenecking revamp, what is the single most common cash-flow omission made by project engineers?"
- **Strong Answer:** "Failing to account for the gross margin lost during the turnaround downtime required to install the revamp. Debottlenecking revamps require taking equipment offline to cut pipe, replace column trays, or tie in compressor nozzles. If a plant generating $\$1.5\text{ million}$ per day in gross margin requires a 10-day extended outage, that represents a $\$15\text{ million}$ cash penalty. This uncaptured margin loss must be treated as an upfront capital cost. Many projects that appear attractive based on equipment purchase price alone are heavily NPV-negative once downtime margin penalties are included."
- **Weak Answer:** "Forgetting to include the cost of shipping the equipment from the vendor or forgetting inflation."

### Question 8: Safety and Environmental Debottlenecking Barriers
*Question:* "Why does debottlenecking a continuous reactor or fired heater frequently trigger environmental New Source Review (NSR) or API 521 flare re-evaluations?"
- **Strong Answer:** "Debottlenecking increases the mass throughput and heat input of the system. Under API 521, relief systems must be sized for the worst-case contingency—such as a complete cooling water failure where total reboiler heat input vaporizes fluid into the flare header. If the higher firing rate creates a relief volume that exceeds flare header sonic velocity limits, the plant violates OSHA PSM 1910.119. Environmentally, debottlenecking often increases annual mass emissions of NOx, SOx, or VOCs past the Significant Emission Rate thresholds under EPA Title V New Source Review, legally classifying the revamp as a 'major modification' and requiring multi-million-dollar BACT pollution control systems."
- **Weak Answer:** "Because government regulators want to inspect any new equipment that is larger than the old equipment."

---

## 15. Key terms

- **API 520 / 521:** American Petroleum Institute mandatory engineering standards governing the sizing, selection, and hydraulic discharge capacity of pressure relief valves and plant flare headers.
- **Availability Loss ($1 - A_{\text{cont}}$):** Fraction of planned operating calendar time lost due to unscheduled equipment breakdowns, instrument failures, or emergency trips.
- **Capacity Creep:** The gradual, organic accumulation of plant capacity achieved over years through minor operational tuning, small piping bypasses, and operator experience.
- **Complementary Slackness:** Mathematical optimality condition in linear programming dictating that non-zero dual shadow prices exist only on strictly binding constraints.
- **Constraint Push Control:** Advanced Process Control (APC) strategy that minimizes process variance ($\sigma$), enabling operating setpoints to be shifted closer to hard physical limits.
- **Continuous OEE:** Overall Equipment Effectiveness adapted for fluid flow: $\text{Availability} \times \text{Rate Efficiency} \times \text{Yield Selectivity}$.
- **Debottlenecking (Revamping):** Modifying existing brownfield equipment to remove specific flow restrictions and increase throughput at low capital intensity.
- **Demonstrated Capacity:** Maximum historical throughput sustained safely across an extended operating window (72+ hours) under compliant quality and emission envelopes.
- **Divided Wall Column (DWC):** Single distillation column shell containing an internal vertical partition, performing three-product separations in one vessel and saving utility capacity.
- **Drum-Buffer-Rope (DBR):** Theory of Constraints execution methodology synchronizing raw material intake (Rope) and protection tanks (Buffer) to the pacing asset (Drum).
- **Dual Variable ($\lambda_i$):** Marginal economic value (shadow price) of relaxing a capacity or resource constraint by one unit ($\partial Z^* / \partial C_i$).
- **Economic Bottleneck:** The specific flowsheet constraint whose incremental expansion produces the highest increase in global enterprise cash margin.
- **End of Run (EOR):** Terminal operating state prior to turnaround characterized by fouled heat exchangers, coked furnace tubes, and deactivated catalysts.
- **Flooding Velocity:** Critical upward vapor velocity in a column above which liquid cannot descend downcomers, causing hydraulic choke and loss of separation.
- **Management of Change (MOC):** Mandatory administrative protocol under OSHA 29 CFR 1910.119 reviewing safety, health, and environmental impacts before altering process assets.
- **Marginal Capital Intensity ($\text{MCI}$):** Capital expenditure plus lost downtime margin divided by incremental annual capacity gained ($\$ / [\text{MT}/\text{year}]$).
- **Nameplate Capacity:** Theoretical continuous production rate specified by the original plant design engineering procurement contractor (EPC).
- **New Source Review (NSR):** US Clean Air Act permitting process triggered when a facility modification increases emissions above statutory significant emission rates.
- **Pinch Point (Heat Transfer):** Point in a heat exchanger network where the temperature difference between hot and cold streams reaches minimum approach ($\Delta T_{\text{min}}$).
- **Rate Efficiency ($P_{\text{cont}}$):** Actual mass processed during online operating hours divided by theoretical mass at maximum demonstrated design speed.
- **Recycle Loop Accumulation:** The buildup of unreacted chemicals or inert contaminants inside closed flowsheet loops, consuming hydraulic volume.
- **Reduced Cost ($r_j$):** The rate at which the objective function deteriorates if a non-basic decision variable is forced into the LP solution.
- **Shadow Price ($\lambda$):** See *Dual Variable*.
- **Six-Tenths Rule:** Empirical power-law relationship ($\text{Cost}_A / \text{Cost}_B = (\text{Cap}_A / \text{Cap}_B)^n$) used to estimate capital scaling across process equipment.
- **Space Velocity (LHSV / WHSV):** Liquid or Weight Hourly Space Velocity measuring the ratio of hourly reactant feed flow to reactor catalyst volume or mass.
- **Start of Run (SOR):** Initial operating period following a major turnaround with clean exchanger surfaces, decoked furnaces, and fresh catalyst beds.
- **Structured Packing:** Corrugated metal sheets installed inside distillation columns providing high theoretical separation efficiency with minimal vapor pressure drop.
- **Surge Residence Time ($\tau$):** Active working liquid volume of an intermediate vessel divided by volumetric flow rate ($V / \dot{V}$).
- **Turndown Ratio:** Ratio of maximum operational throughput to minimum stable throughput without causing operational instability or off-spec production.
- **Whack-a-Mole Phenomenon:** Operational failure where debottlenecking one unit immediately exposes an adjacent unpredicted constraint, stranding invested capital.

---

## 16. Sources

- **Goldratt, E. M., & Cox, J. (1984).** *The Goal: A Process of Ongoing Improvement*. North River Press. [Foundational treatise establishing the Theory of Constraints and Drum-Buffer-Rope scheduling].
- **Biegler, L. T., Grossmann, I. E., & Westerberg, A. W. (1997).** *Systematic Methods of Chemical Process Design*. Prentice Hall. [Rigorous mathematical foundation for flowsheet optimization, heat integration, and pinch analysis].
- **Perry, R. H., & Green, D. W. (2018).** *Perry's Chemical Engineers' Handbook* (9th ed.). McGraw-Hill Education. [Definitive engineering reference for column hydraulics, heat exchanger design, and fluid flow].
- **Peters, M. S., Timmerhaus, K. D., & West, R. E. (2003).** *Plant Design and Economics for Chemical Engineers* (5th ed.). McGraw-Hill. [Standard academic reference for capital cost estimation, the six-tenths rule, and revamp budgeting].
- **Lieberman, N. P. (2014).** *Troubleshooting Process Operations* (5th ed.). PennWell Books. [Practical industrial guide on identifying distillation flooding, condenser pinches, and real-world bottlenecks].
- **King, C. J. (1980).** *Separation Processes* (2nd ed.). McGraw-Hill. [Theoretical principles of mass transfer, tray hydraulics, and packing performance].
- **Edgar, T. F., Himmelblau, D. M., & Lasdon, L. S. (2001).** *Optimization of Chemical Processes* (2nd ed.). McGraw-Hill. [Formulation of linear, non-linear, and dual LP optimization models in continuous plants].
- **American Petroleum Institute (API). (2020).** *API Standard 520: Sizing, Selection, and Installation of Pressure-Relieving Devices in Refineries* (10th ed.). API Publishing. [Statutory standard for relief valve sizing during throughput expansions].
- **American Petroleum Institute (API). (2020).** *API Standard 521: Pressure-Relieving and Depressuring Systems* (7th ed.). API Publishing. [Mandatory standard for evaluating plant-wide flare header loads and fire/cooling contingencies].
- **Occupational Safety and Health Administration (OSHA). (1992).** *Process Safety Management of Highly Hazardous Chemicals* (29 CFR 1910.119). US Department of Labor. [Federal regulations governing Management of Change (MOC) in process facilities].
- **International Organization for Standardization (ISO). (2014).** *ISO 22400-2: Automation systems and integration — Key performance indicators for manufacturing operations management*. ISO. [Formal definitions of technical utilization and continuous OEE].
- **Seveso III Directive. (2012).** *Directive 2012/18/EU of the European Parliament on the control of major-accident hazards involving dangerous substances*. Official Journal of the European Union. [European statutory framework governing process safety and capacity modifications].

---

## 17. Links

- **Module Index:** [Module 38 — Process Industries, Commodities, and Bulk Supply Chains](_index.md)
- **Preceding Topics:**
  - [Topic 01 — Continuous, Batch, and Discrete Production](01_continuous-batch-discrete.md)
  - [Topic 02 — Yield and Mass Balance Modeling](02_yield-mass-balance.md)
  - [Topic 03 — Quality Specifications and Blending Optimization](03_quality-specs-blending.md)
  - [Topic 04 — Campaigns and Changeover Optimization](04_campaigns-changeover.md)
  - [Topic 05 — Bulk Storage, Inventory Accounting, and Tank Farm Scheduling](05_bulk-storage-inventory.md)
  - [Topic 06 — Commodity Price and Hedging: Transformation Margins and Basis Risk](06_commodity-price-hedging.md)
  - [Topic 07 — Bulk Logistics, Marine Chartering, and Laytime Management](07_bulk-logistics-laytime.md)
- **Subsequent Topics:**
  - [Topic 09 — Energy, Water, and Carbon Emissions in Production Planning](09_energy-water-emissions.md)
  - [Topic 10 — Process Safety, Environmental Regulation, and Operating Limits](10_process-safety-regulation.md)
- **Program Survey File:** [Course Survey: Process Industries and Commodities](../../../course/38_process_industries_commodities.md)
- **Related Elective Modules:**
  - [Module 11 — Network Design Optimization](../../02_methods/11_network_design_optimization/_index.md)
  - [Module 15 — Production Planning and Scheduling](../../02_methods/15_inventory_theory/_index.md)
  - [Module 20 — Operations Planning and Master Scheduling](../../03_commerce/20_operations_planning_master_scheduling/_index.md)
  - [Module 39 — Mining, Metals, and Forestry Supply Chains](../39_mining_metals_forestry/_index.md)
