# NPI and Ramp: Hardware Phase Gates, Yield Learning Curves, Tooling Lead Times, and Line Ramp Optimization

> Module 40 — Telecom, Technology Hardware, and Aerospace Supply Chains
> Topic 04 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 09 (Production Planning and Master Scheduling), Module 10 (Inventory Theory and Stochastic Control), Module 40 Topic 01 (Configuration, BOM, and Life Cycle), Topic 02 (Component Allocation under Shortage), Topic 03 (EMS and Contract Manufacturing)
> Version 2026.5

---

## 1. Why this topic exists

In telecommunications infrastructure, enterprise networking, aerospace avionics, and advanced technology hardware, New Product Introduction (NPI) represents the highest-risk phase of the product life cycle. Unlike software development, where features can be deployed incrementally and defects remediated via over-the-air patches, physical hardware development is fundamentally constrained by the laws of physics, metallurgy, thermodynamics, and long-cycle tooling fabrication. A dimensional interference in a high-pressure aluminum die-cast chassis, a thermal dissipation miscalculation on a 64-layer printed circuit board assembly (PCBA), or an undetected timing skew in a custom 4-nanometer Application-Specific Integrated Circuit (ASIC) cannot be patched post-assembly. Rectifying such errors after capital tooling has been cut or volume production has commenced requires millions of dollars in scrapped tooling, months of redesign delays, and catastrophic contract penalties.

High-technology hardware supply chains operate under an unforgiving operational reality: the window of peak market profitability is narrow, yet the physical lead times required to prepare manufacturing capacity are extraordinarily long. Sourcing long-lead semiconductors requires binding procurement commitments 26 to 52 weeks before the first finished device rolls off the production line. Machining multi-cavity injection molds and progressive stamping dies requires 12 to 24 weeks of capital tooling lead time before first parts can be sampled. Consequently, supply chain and operations executives are forced to commit tens of millions of dollars in non-cancelable capital expenditures and component inventory long before the product design has been fully validated or customer demand has stabilized.

When a hardware program transitions from low-volume engineering builds to mass production—a phase transition colloquially known as the "Valley of Death"—four systemic operational hazards converge:

1. **The Phase-Gate Discipline Breakdown:** Hardware engineering teams, facing aggressive launch targets, are routinely tempted to bypass rigorous qualification phase gates (Engineering Validation Test [EVT], Design Validation Test [DVT], and Production Validation Test [PVT]). Authorizing mass production tooling before freeze of the functional design results in extensive tool modifications, scrapped mold steel, and compounding schedule slips.
2. **The Yield Ramp Deficit:** Production yield does not instantaneously reach nominal targets upon factory launch. Early production lots suffer from steep defect rates driven by component tolerances, solder reflow thermal profiles, automated pick-and-place nozzle misalignments, and silicon parametric variation. If supply chain planners assume a mature 98% line yield when actual launch yield is 65%, the factory will fail to deliver up to 35% of its committed output, triggering stockouts, contractual service-level agreement (SLA) breaches, and assembly line starvation.
3. **The Non-Cancelable, Non-Returnable (NCNR) Liability Trap:** Long-lead semiconductor vendors demand binding NCNR purchase orders months in advance of production. If product yield collapses, or if early field testing reveals a critical flaw requiring an architectural redesign, the OEM remains contractually liable for 100% of the custom silicon and specialized components ordered. In volatile markets, NCNR liabilities on stalled NPI ramps routinely exceed hundreds of millions of dollars in excess and obsolescence (E&O) write-downs.
4. **The Line Balancing and Bottleneck Choke:** High-speed surface-mount technology (SMT) lines can place over 100,000 components per hour, but downstream functional testing, radio frequency (RF) calibration, and thermal burn-in chambers operate on cycles lasting minutes or hours. In early ramp, high defect fallout saturates test stations and rework loops, causing work-in-progress (WIP) to accumulate into unmanageable "bonepiles" that starve the final packaging line and halt factory throughput.

Mastering the mathematical mechanics of hardware phase gates, capital tooling qualification (T0, T1, T2 sampling), parametric yield learning curves, stochastic NCNR commitments, and line ramp optimization is essential for preventing catastrophic launch failures and protecting operating margins in technology hardware enterprises.

---

## 2. Learning outcomes

By completing this topic, you will be able to:

1. **Structure and govern hardware NPI phase gates** (Concept, EVT, DVT, PVT, Mass Production), defining quantitative exit criteria, build lot sizes, statistical process control metrics ($C_p, C_{pk}$), and First Article Inspection (FAI) gates.
2. **Model and optimize capital tooling qualification cycles**, establishing timelines for multi-cavity injection molds and stamping dies across T0 (first off-tool), T1 (dimensional/CMM verification), and T2 (production-intent capability) sampling milestones.
3. **Formulate and solve mathematical yield learning curves**, applying Wright's cumulative average model, Crawford's unit learning curve, and defect density learning formulations ($D(N)$) to forecast scrap rates and line output.
4. **Model semiconductor and assembly defect density**, deriving expected yields using Poisson, Murphy, Seeds, and Stapper (Negative Binomial) clustered-defect distributions as a function of active die area and process maturity.
5. **Calculate required production starts and scrap allowances** across multi-stage assembly routings using First-Pass Yield (FPY) and Roll-Through Yield (RTY) equations with rework feedback loops.
6. **Formulate NCNR procurement commitments under dual uncertainty**, optimizing order quantities for long-lead components facing both stochastic market demand and stochastic assembly yield.
7. **Optimize SMT and system assembly line balancing**, calculating takt time, station cycle times, line balancing efficiency, burn-in chamber capacity requirements, and bonepile rework equilibrium.
8. **Diagnose and mitigate industrial ramp crises**, identifying root causes of early-stage line starvation, testing bottlenecks, thermal chamber saturation, and supplier de-commitments.

---

## 3. The core mental model

The hardware NPI and ramp process is governed by the synchronized convergence of four distinct streams: **Design Maturity**, **Tooling & Fixture Readiness**, **Material Commitment Horizons**, and **Manufacturing Line Capability**.

At the beginning of an NPI cycle, product uncertainty is high, engineering changes occur daily, and capital commitment is minimal. As the program progresses through EVT, DVT, and PVT toward Mass Production (MP), design changes become exponentially more expensive, lead times expand, and financial exposure accelerates. The core objective of NPI supply chain analytics is to advance material and tooling investments just in time to support target market launch dates without exposing the balance sheet to catastrophic write-offs if qualification fails.

```text
====================================================================================================
                        HARDWARE NPI STAGE-GATE CONVERGENCE TIMELINE
====================================================================================================

PHASE:        CONCEPT / ARCH      EVT                 DVT                 PVT            MASS PROD (MP)
            [ Architecture ] -> [ Eng Validation ] -> [ Design Valid ] -> [ Prod Valid ] -> [ Full Ramp ]
----------------------------------------------------------------------------------------------------
TYPICAL DURATION: 12-24 Wks        8-12 Wks            8-12 Wks            6-8 Wks         Continuous
BUILD SIZE:       0-10 Units       50-200 Units        500-2,000 Units     2k-10k Units    10k-100k+/Wk
----------------------------------------------------------------------------------------------------
DESIGN        • System Spec       • PCB Layout A      • PCB Layout B      • PCB Rev C      • Design Frozen
MATURITY:     • Simulation        • 3D Printed Mockup • Soft Tool Parts   • Hard Tool Part • ECO Class I/II
              • Proof of Concept  • Hand Rework OK    • Environmental Test• Golden Samples   Only
----------------------------------------------------------------------------------------------------
TOOLING &     • Tooling DFM       • Soft Tooling Cut  • Hard Tooling Cut  • Multi-Cav T1   • T2 Approved
FIXTURES:     • Moldflow Sim      • Prototype Jigs    • T0 Sampling       • CMM / FAI Pass • Prod Fixtures
              • RF Simulation     • Benchtop Test     • Functional Fixt   • ICT/FCT Valid  • Gauge R&R Pass
----------------------------------------------------------------------------------------------------
MATERIAL &    • Feasibility BOM   • Short-Run Sourcing• NCNR POs Released • Prod Batches   • Continuous MRP
COMMITMENTS:  • Long-Lead ID      • Catalog Passives  • Long-Lead Silicon • Buffer Stocks  • Min-Max / VMI
              • ASIC Tapeout Auth • High Scrap Budget • Custom Chassis PO • Liability High • Low Scrap Pct
----------------------------------------------------------------------------------------------------
LINE YIELD    • N/A               • 20% - 50% FPY     • 50% - 80% FPY     • 80% - 92% FPY  • 95% - 99% FPY
& CAPACITY:   • R&D Lab Bench     • Manual SMT Line   • Pilot EMS Line    • Target MP Line • Multi-Shift
              • 0 Units/Day       • 5 Units/Day       • 50 Units/Day      • 500 Units/Day  • Takt Balanced
====================================================================================================
                                      CRITICAL DECISION GATES
              Gate 0:             Gate 1:             Gate 2:             Gate 3:          Gate 4:
              Capex & Tooling     DVT Tooling Cut     PVT Ramp Release    Mass Production  Full Capacity
              Authorization       & NCNR Silicon PO   & SMT Line Setup    Ramp Approval    Signoff
====================================================================================================
```

### The Causal Mechanics of the Convergence

1. **The Lead Time / Freeze Trade-off:** Long-lead items (ASICs, microprocessors, custom high-cavity tooling) dictate the schedule. If an ASIC requires a 36-week fabrication and packaging lead time, the tapeout decision and wafer order MUST be released during EVT, when the architectural logic is still subject to functional bug discovery. If bugs are found during DVT firmware integration, the company must decide whether to scrap the wafer lot (absorbing hundreds of thousands of dollars in NCNR write-downs) or implement complex software workarounds that degrade processing efficiency.
2. **The Tooling Transition Boundary:** Prototype parts during EVT are fabricated using rapid prototyping methods: CNC-machined aluminum billets, stereolithography (SLA) resin prints, and single-cavity soft aluminum tooling. While fast (2 to 4 weeks), these parts have different thermal expansion coefficients, surface finishes, and structural tolerances than production parts. Hard steel tooling (P20, H13, or hardened stainless steel) with 4, 8, or 16 cavities requires 12 to 20 weeks to machine, polish, and qualify. Hard tooling must be authorized no later than early DVT so that T0 and T1 samples are available to validate thermal and mechanical integrity before PVT.
3. **The Yield Ramp Curve:** Line yield follows a predictable learning trajectory governed by Wright's law and defect density reduction. Early EVT and DVT runs produce catastrophic fallout (20% to 50% scrap) because SMT pick-and-place programs, solder stencil apertures, reflow thermal profiles, and automated optical inspection (AOI) algorithms are uncalibrated. By PVT, production processes must achieve statistical process control ($C_{pk} \ge 1.33$) on critical-to-quality (CTQ) dimensions. In Mass Production, yield asymptotically approaches its theoretical upper bound ($Y_{\max} \approx 98\% - 99.5\%$).

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Industry Misuse / Pitfall | Standard Unit / Metric |
|---|---|---|---|
| **EVT (Engineering Validation Test)** | First physical build phase validating that core functional, electrical, and thermal architectures meet baseline technical specifications. | Confusing EVT with a production-ready trial; expecting high assembly yield or final cosmetic finishes on EVT units. | Units built (typically 50–200); First-Pass Yield (%) |
| **DVT (Design Validation Test)** | Second build phase verifying that the integrated product design meets environmental, regulatory, reliability, and cosmetic standards using production-intent materials. | Proceeding to DVT before resolving fundamental schematic or thermal architecture defects discovered in EVT. | Units built (500–2,000); Environmental pass rate (%) |
| **PVT (Production Validation Test)** | Final pre-production build phase executing at target mass production line speeds, with production tooling, final test fixtures, and standard line operators. | Treating PVT as an engineering experiment rather than a statistical verification of factory manufacturing capability and cycle times. | Units built (2,000–10,000); Process Capability ($C_{pk}$) |
| **MP (Mass Production)** | Continuous commercial manufacturing operating at planned takt time, nominal yield targets, and balanced multi-shift line capacity. | Declaring MP before achieving PVT exit criteria, causing massive scrap and test line gridlock under volume pressure. | Units per week / month; Run rate; OEE (%) |
| **First-Pass Yield (FPY)** | The percentage of manufactured units that complete a specific manufacturing or test process step correctly without requiring rework, re-test, or scrapping. | Conflating FPY with final line yield by ignoring units that passed only after being reworked off-line. | Percentage ($\%$) or ratio ($0 \le FPY \le 1$) |
| **Roll-Through Yield (RTY)** | The mathematical product of first-pass yields across all sequential manufacturing, assembly, and test operations in a production routing: $RTY = \prod_{k=1}^K FPY_k$. | Assuming high final yield implies a healthy process when individual intermediate stations have severe scrap and rework loops. | Percentage ($\%$) |
| **Defect Density ($D_0$)** | The average number of critical, fatal physical or electrical defects per unit area of active semiconductor silicon or PCBA surface. | Assuming defect density is constant across varying die sizes; ignoring defect clustering phenomena. | Defects per $\text{cm}^2$ or defects per million opportunities (DPMO) |
| **Wright's Learning Curve** | Empirical power-law relationship stating that cumulative average unit production time or cost decreases by a constant percentage every time cumulative production volume doubles. | Applying cost learning curves directly to assembly yield without adjusting for the theoretical asymptotic yield ceiling ($Y \le 100\%$). | Learning rate parameter ($b$); Progress ratio ($\phi = 2^{-b}$) |
| **Crawford's Unit Curve** | Learning curve formulation where the cost or labor of the specific $N$-th unit decreases as a power function of $N$, rather than the cumulative average of all units up to $N$. | Conflating Crawford's marginal unit curve with Wright's cumulative average curve, leading to severe underestimation of total early ramp scrap. | Exponent ($b$); Marginal unit value |
| **T0 Sampling** | The initial physical shot or part off a newly machined mold or die, inspected at the toolmaker to verify gross cavity fill, venting, and mold kinematics. | Demanding production cosmetic texture or tight dimensional tolerances on T0 parts before core steel adjustments are completed. | Dimensional deviation ($\text{mm}$); Visual defects |
| **T1 Sampling** | Parts produced after initial steel modifications, run on the intended manufacturing press, subjected to full First Article Inspection (FAI) and CMM measurement. | Approving T1 parts based solely on visual aesthetics without performing full dimensional Coordinate Measuring Machine (CMM) reports. | Inspection report pass rate ($\%$); CMM tolerance band |
| **T2 Sampling** | Production-intent parts run at nominal cycle times under mass production conditions, used to establish statistical process capability ($C_p, C_{pk}$) and cosmetic sign-off. | Skipping T2 sampling to accelerate schedule, resulting in dimensional drift and mold jamming during multi-cavity mass production. | $C_p, C_{pk}$ indices; Cavity-to-cavity variance |
| **Steel-Safe Condition** | Mold design strategy where steel is intentionally left uncut on internal core and cavity surfaces so that subsequent dimensional corrections require removing steel rather than welding. | Cutting tooling to nominal dimensions initially; adding steel via laser welding weakens mold integrity and shortens tool fatigue life. | Dimensional tolerance offset ($\text{mm}$) |
| **First Article Inspection (FAI)** | Formal metrology verification protocol (e.g., AS9102) certifying that 100% of engineered dimensions on a 2D engineering drawing match physical off-tool parts. | Sampling only 1 or 2 parts from an 8-cavity tool rather than measuring parts from every single cavity independently. | AS9102 Form 1, 2, 3 compliance; Out-of-spec count |
| **Gauge R&R** | Statistical evaluation of the repeatability and reproducibility of a measurement system to ensure measurement error is small relative to process tolerance. | Blaming manufacturing line variation for test failures when the test fixture or measurement gauge itself has $>30\%$ measurement error. | Percentage of tolerance ($\%R\&R < 10\%$ is acceptable) |
| **Process Capability ($C_{pk}$)** | Statistical measure quantifying how close a process is running to its specification limits relative to its natural process variability: $C_{pk} = \min\left(\frac{USL - \mu}{3\sigma}, \frac{\mu - LSL}{3\sigma}\right)$. | Measuring $C_{pk}$ on an unstable or non-normally distributed process; assuming $C_{pk} \ge 1.33$ guarantees zero defects. | Dimensionless ratio ($C_{pk} \ge 1.33$ standard; $\ge 1.67$ CTQ) |
| **NCNR Commitment** | Non-Cancelable, Non-Returnable commercial purchase order legally obligating the buyer to take delivery of and pay for 100% of committed components. | Treating an NCNR purchase order as an adjustable demand forecast; assuming suppliers will accept cancellation without full financial indemnity. | Currency value ($\$$) and committed units ($Q$) |
| **Liability Window** | The forward time horizon during which an OEM is financially exposed to raw material and work-in-progress inventory costs at a supplier or EMS partner. | Measuring liability only by finished goods inventory while ignoring un-cancellable wafer starts and raw material pipelines. | Calendar weeks or days |
| **Takt Time** | The required pace of manufacturing production necessary to match customer demand: $T_{\text{takt}} = \frac{\text{Available Net Operating Time}}{\text{Required Customer Demand}}$. | Confusing takt time (demand-driven target) with cycle time (physical station speed capability). | Seconds per unit ($\text{sec/unit}$) |
| **Line Balancing Efficiency** | Ratio of total work content across all line stations to the product of the number of stations and the bottleneck station cycle time: $\eta = \frac{\sum t_i}{M \cdot t_{\text{bottle}}}$. | Ignoring the impact of rework loops and scrap fallout on the effective arrival rate at downstream bottleneck stations. | Percentage ($\%$) |
| **Burn-In / HASS** | Highly Accelerated Stress Screening or thermal soak (e.g., 24–72 hours at $55^\circ\text{C}-85^\circ\text{C}$ under electrical load) to precipitate infant mortality failures. | Sizing burn-in chambers based on nominal steady-state cycle times without accounting for early ramp yield fallout and extended re-soak cycles. | Chamber slot capacity; Duration ($\text{hours}$) |
| **Bonepile** | Accumulation of defective, failed, or un-reworked PCBAs and systems waiting off-line for debug, component desoldering, or engineering disposition. | Failing to cap bonepile inventory; allowing high-value components to sit indefinitely until they become technically obsolete or damaged. | Units in bonepile; Dollar value ($\$$) of trapped WIP |
| **SMT Feeder Attrition Allowance** | The percentage of miniature passive components (e.g., 0201 or 0402 MLCCs) lost during automated tape-and-reel indexing, nozzle pickup, or vision reject. | Assuming 100% component delivery to boards; failing to add 0.5%–2.0% feeder attrition to NPI BOM procurement orders. | Percentage ($\%$) or parts-per-million (PPM) |
| **Reflow Thermal Profiling** | The empirical time-temperature curve (preheat, soak, reflow peak, liquidus time, cooling rate) measured via multivariant thermocouples across a PCBA. | Running a generic reflow profile across boards with divergent thermal mass, causing cold solder joints or component delamination. | Seconds above liquidus ($TAL$); Peak temp ($^\circ\text{C}$) |
| **Design for Assembly (DFA) Index** | Quantitative metric evaluating the ease and speed of manual or automated product assembly based on component count, insertion directions, and fasteners. | Designing complex internal cable routing and blind screw insertions that inflate assembly cycle times beyond planned takt time. | DFA score ($0$ to $100\%$); Assembly time ($\text{sec}$) |
| **ICT Bed-of-Nails Fixture** | Custom mechanical test fixture populated with spring-loaded pogo pins that contact dedicated test points on the unpowered or powered PCBA. | Releasing PCB layout revisions without preserving test point locations, rendering expensive ($15k–$40k) ICT fixtures obsolete. | Test coverage percentage ($\%$); Pin count |
| **ECO Effectivity Classification** | Formal categorization of Engineering Change Orders into Class I (form, fit, function, or safety impact) and Class II (minor, non-interchangeable adjustments). | Misclassifying a functional schematic fix as Class II to bypass customer notification or airworthiness re-certification audits. | Categorical (Class I vs. Class II) |
| **Near-Field EMI Pre-Scanning** | Diagnostic radio-frequency probing across PCB traces, clock oscillators, and chassis seams inside a benchtop shielded tent prior to formal anechoic testing. | Waiting until formal FCC/CE chamber testing during PVT to discover radiated emissions, forcing late-stage board redesign. | Decibels relative to microvolt ($\text{dB}\mu\text{V}$) |
| **Golden Sample** | Fully qualified physical unit certified by OEM and EMS engineering leaders as the benchmark standard for cosmetic finish, fit, and operational performance. | Losing physical custody of the golden unit or failing to produce duplicate golden samples for cross-plant calibration. | Serialized physical hardware artifact |
| **Moldflow Shear-Rate Analysis** | Computational fluid dynamics simulation calculating melt front velocity, injection pressure, shear heating, and air trap locations in mold cavities. | Relying on toolmaker intuition for gate sizing, resulting in resin degradation, high cosmetic gate blush, or fiber orientation warpage. | Shear rate ($\text{s}^{-1}$); Injection pressure ($\text{MPa}$) |

---

## 5. The mechanics, step by step

Managing an NPI program and its volume manufacturing ramp requires executing five interconnected analytical and operational procedures:

```text
[Step 1: Phase Gate Governance & Exit Metrics]
                     │
                     ▼
[Step 2: Tooling Qualification & Cavitation Economics]
                     │
                     ▼
[Step 3: Yield Learning Curve & Defect Density Modeling]
                     │
                     ▼
[Step 4: NCNR Purchase Commitments under Dual Uncertainty]
                     │
                     ▼
[Step 5: Line Ramp Balancing & Bottleneck Management]
```

---

### Step 1: NPI Phase Gate Governance and Quantitative Exit Criteria

Hardware development cannot be managed as an open-ended continuous sprint. It must be structured into sequential, gated phases with inviolable, quantitative exit criteria. Transitioning to a subsequent phase without satisfying exit gates incurs compounding financial and operational risk.

```text
====================================================================================================
                        PHASE GATE EXIT METRIC SCORECARD
====================================================================================================
Gate Metric              EVT Exit Gate            DVT Exit Gate            PVT Exit Gate (MP Launch)
----------------------------------------------------------------------------------------------------
Build Purpose            Functional POC           Environmental / Compl.   Factory Capability & Rate
Tooling Status           Soft / Prototype Tools   Hard Tools Cut (T0/T1)   Hard Tools Qualified (T2)
BOM Maturity             BOM Freeze: 85%          BOM Freeze: 98%          BOM Freeze: 100% (ECO Control)
First-Pass Yield (FPY)   $\ge 50\%$               $\ge 75\%$               $\ge 90\%$ (Final MP Target $\ge 98\%$)
Critical Dimensions      Rough tolerance check    FAI on all parts         $C_{pk} \ge 1.33$ ($1.67$ for CTQ)
Environmental / Rel.     Bench testing            HALT / HASS Passed       Full Reliability Qualified
Regulatory Compliance    Pre-scan compliance      Formal FCC/CE/NEBS Run   Certificates in Hand
SMT Placement Speed      Manual / Slow Speed      $50\%$ Production Speed  $100\%$ Target Takt Time
Test Coverage            ICT Partial Coverage     ICT 100% / FCT Prototype Full Automated ICT/FCT/Burn-In
Rework Capacity Ratio    $Rework/Fallout \ge 1.0$ $Rework/Fallout \ge 1.5$ $Rework/Fallout \ge 2.5$
====================================================================================================
```

#### The Gate Sign-Off Protocol
Each phase gate requires unanimous sign-off across five functional directors:
1. **VP of Hardware Engineering:** Validates schematic integrity, signal integrity, timing margins, and thermal performance.
2. **VP of Operations / Supply Chain:** Validates component lead times, supplier capacity, NCNR liability exposure, and dual-sourcing status.
3. **VP of Quality and Reliability:** Validates HALT (Highly Accelerated Life Testing), drop test, thermal shock, vibration, and environmental regulatory compliance.
4. **EMS Plant General Manager:** Validates SMT line readiness, feeder setup, test fixture repeatability (Gauge R&R $<10\%$), and labor training.
5. **Product Line General Manager (P&L Owner):** Validates total unit landed cost against target gross margin models.

If any gate metric fails—for example, if DVT thermal shock testing precipitates solder joint cracking under BGA packages—the gate is **held**. An Engineering Change Order (ECO) is raised, corrective actions are validated on an intermediate "DVT2" or "re-spin" build, and the timeline is adjusted before committing capital to mass production tooling and volume component inventory.

---

### Step 2: Capital Tooling Lifecycle and Cavitation Economics

Hardware products require custom physical tooling: injection molds for plastic bezels and internal structural frames, die-casting dies for aluminum RF filter cavities, progressive stamping dies for sheet-metal chassis, and automated assembly/test fixtures.

#### The Tooling Development Timeline
A production-grade hardened steel injection mold (e.g., H13 steel, Rockwell C hardness 48–52) requires a rigid 16-to-22 week qualification progression:

```text
Week 0          Wk 4           Wk 8         Wk 11        Wk 14        Wk 17        Wk 20
───┬──────────────┬──────────────┬────────────┬────────────┬────────────┬────────────┬───►
   │              │              │            │            │            │            │
Tooling PO     DFM & Mold-    CNC Roughing,  T0 Shot      T1 Shot      T2 Shot      Tool Release
Authorized     flow Complete  EDM & Polish   at Toolmaker at Molder    Production   to Mass Prod
(Capex spent)  (Steel freeze) (Steel-Safe)   (Gross fill) (Full FAI)   Run (Cpk)    (Gold Sample)
```

1. **Week 0–4: Design for Manufacturability (DFM) and Moldflow Simulation:** Engineering analyzes draft angles (minimum $1.0^\circ$ to $1.5^\circ$ for textured surfaces), wall thickness uniformity to prevent sink marks, gate locations to control knit lines, and cooling line placement. Tooling drawings are locked.
2. **Week 4–8: Steel Machining and EDM:** Mold bases are pre-machined. Cavities and cores are cut using 5-axis high-speed CNC milling and Electrical Discharge Machining (EDM). Critical dimensions are left in a **steel-safe condition** (extra metal on the mold core/cavity so parts are molded undersized on external walls and oversized on holes, allowing subsequent adjustments by cutting away steel rather than welding).
3. **Week 8–11: T0 Sampling (First Off-Tool Part):** The mold is mounted on a test press at the moldmaker's facility. Parts are shot using general resin to verify mechanical movement, side-action slider mechanics, lifter clearances, and basic cavity fill without flashing or short shots. Dimensions are measured for gross shrinkage divergence.
4. **Week 11–14: Tool Tuning and T1 Sampling:** Steel-safe areas are machined to nominal dimensions based on actual measured resin shrinkage. Polish is applied. T1 parts are molded using the specified production-grade resin and colorant on the designated contract manufacturer's injection press. A complete **First Article Inspection (FAI)** is conducted: Coordinate Measuring Machines (CMM) and optical comparators measure every dimension specified on the 2D engineering drawing across all mold cavities.
5. **Week 14–17: Texturing, Hardening, and T2 Sampling:** Chemical etching applies mold texturing (e.g., Mold-Tech MT-11010). Mold components undergo heat treatment for surface hardening. T2 parts are run under continuous mass production conditions at nominal cycle times (e.g., 300 continuous cycles). Metrology teams perform statistical process capability studies:
   $$C_p = \frac{USL - LSL}{6\sigma}, \quad C_{pk} = \min\left(\frac{USL - \mu}{3\sigma}, \frac{\mu - LSL}{3\sigma}\right)$$
   For release to volume production, critical dimensions must demonstrate $C_{pk} \ge 1.33$ (or $\ge 1.67$ for safety-critical dimensions).
6. **Week 17–20: "Golden Sample" Approval and Line Integration:** Molds are approved, master reference parts ("Golden Samples") are signed off by OEM quality executives, and the tool is officially transferred to mass production.

#### Cavitation Economics and Capex Optimization
Selecting the number of mold cavities ($M$) represents a fundamental trade-off between upfront non-recurring capital expenditure (Capex) and recurring piece-part manufacturing cost:

$$\text{Capex}(M) = C_{\text{base}} + C_{\text{cavity}} \cdot M^\gamma \quad (0.6 \le \gamma \le 0.85)$$

$$\text{Piece Price}(M) = \text{Material Cost} + \frac{\text{Machine Hourly Rate} \cdot T_{\text{cycle}}(M)}{3600 \cdot M} + \frac{\text{Direct Labor Hourly Rate} \cdot T_{\text{handling}}}{3600 \cdot M}$$

Where:
- $M$: Number of cavities (e.g., 1, 2, 4, 8, 16).
- $T_{\text{cycle}}(M)$: Molding cycle time (seconds). Larger multi-cavity tools require larger clamping tonnage presses, higher injection pressure, and longer cooling times, so $T_{\text{cycle}}(M)$ increases moderately with $M$.
- Higher cavitation dramatically lowers piece-part production costs by amortizing machine hourly run rates across $M$ units per shot.
- However, higher cavitation increases tool fabrication lead time by 4 to 8 weeks, elevates T1/T2 qualification complexity (balancing flow runners and gate pressures across all $M$ cavities simultaneously), and dramatically increases financial risk if a mid-DVT architectural engineering change renders the mold obsolete.

#### Tooling Metallurgy and Steel Selection Matrix
Selecting mold steel dictates tool life, thermal conductivity, polishability, and total program lead time:

```text
====================================================================================================
                        CAPITAL TOOLING METALLURGY MATRIX
====================================================================================================
Steel Grade       Hardness (HRC)  Shots (Fatigue Life)  Lead Time     Best Application
----------------------------------------------------------------------------------------------------
Al 7075-T6        N/A (150 HB)    5k - 25k shots        3 - 5 Wks     EVT rapid prototypes, low clamp
P20 Tool Steel    28 - 32 HRC     100k - 250k shots     8 - 12 Wks    DVT structural parts, soft steel
H13 Hard Steel    48 - 52 HRC     500k - 1,000k shots   14 - 18 Wks   PVT / MP high-volume enclosures
S136 Stainless    50 - 54 HRC     1,000k+ shots         16 - 20 Wks   Corrosive resins, optical finish
Tungsten Carbide  85 - 90 HRA     5,000k+ strokes       18 - 24 Wks   High-speed progressive stampings
====================================================================================================
```

#### Multi-Cavity Runner Pressure and Thermal Balancing
In multi-cavity injection molds ($M \ge 4$), melt front velocity through the hot or cold runner manifold induces non-Newtonian shear thinning. If gate pressures deviate across cavities:

$$\Delta P_{\text{gate}} = \max_{j,k} \frac{|P_j - P_k|}{\bar{P}} > 5.0\%$$

plastic packing density varies wildly. Cavity-to-cavity part weight disparity must be statistically verified at T1 sampling:

$$\Delta W = \frac{W_{\max} - W_{\min}}{\bar{W}} \le 1.5\%$$

If $\Delta W > 1.5\%$, parts from outer cavities suffer from sink marks and short shots, while parts from inner cavities develop severe flash, destroying process capability ($C_{pk} < 1.00$). Correcting runner balance requires rheological flow analysis and micro-EDM gate re-sizing.

---

### Step 3: Yield Learning Curve and Defect Density Modeling

Manufacturing yield during NPI is non-stationary; it improves systematically over cumulative production volume as engineers identify root causes of component failure, refine SMT placement programs, optimize solder paste stencil thickness, and calibrate automated test limits.

#### 1. Wright's Cumulative Learning Model vs. Crawford's Unit Model
Originally derived by T.P. Wright (1936) in aircraft assembly, the classical learning curve models unit labor or cost reduction as a power law of cumulative volume $x$:

$$\bar{Y}_{\text{cost}}(x) = Y_1 \cdot x^{-b}$$

Where $Y_1$ is the cost/hours of the first unit, $x$ is cumulative volume, and $b$ is the learning index related to the progress ratio $\phi$ (e.g., an 80% learning curve means $\phi = 0.80$):

$$b = -\frac{\ln(\phi)}{\ln(2)}$$

In Crawford's formulation (1944), the learning curve applies to the marginal cost of the $x$-th unit directly:

$$y_{\text{unit}}(x) = y_1 \cdot x^{-b}$$

Mathematically, the exact relationship between Crawford's marginal unit curve and Wright's cumulative average curve is derived by integrating the unit work content over cumulative production starts $N$:

$$T(N) = \sum_{x=1}^N y_{\text{unit}}(x) \approx \int_0^N y_1 x^{-b_c} \, dx = \left[ \frac{y_1}{1 - b_c} x^{1 - b_c} \right]_0^N = \frac{y_1}{1 - b_c} N^{1 - b_c}$$

The cumulative average unit labor or cost $\bar{Y}(N)$ across all $N$ units is therefore:

$$\bar{Y}(N) = \frac{T(N)}{N} = \frac{y_1}{1 - b_c} N^{-b_c}$$

Comparing this directly to Wright's cumulative average formulation $\bar{Y}(N) = Y_1 N^{-b_w}$, we observe that the power exponents are identical ($b_w = b_c = b$), while the first-unit baseline coefficient scales by the factor:

$$Y_1 = \frac{y_1}{1 - b}$$

In high-technology hardware NPI, conflating Crawford's marginal formulation with Wright's cumulative average model causes severe financial distortions. For an 80% learning rate ($\phi = 0.80$, $b = -\ln(0.8)/\ln(2) \approx 0.3219$):

$$\frac{Y_1}{y_1} = \frac{1}{1 - 0.3219} = \frac{1}{0.6781} \approx 1.4746$$

If a supply chain planner assumes an 80% Crawford unit curve but mistakenly calculates scrap budget using Wright's cumulative average equation, the total labor and scrap cost across early production lots will be underestimated by $47.5\%$, exhausting the NPI operating budget before DVT completion.

#### 2. Adapting Learning Curves to Hardware Manufacturing Yield
Directly applying cost learning formulations to manufacturing yield is physically invalid because yield is bounded from above ($0 \le Y(x) \le 1.0$). We model yield learning either via **defect rate reduction** or via **asymptotic yield improvement curves**.

##### A. Defect Density Learning Formulation
Yield fallout is driven by the density of fatal defects per unit area $D(x)$, which decays as cumulative assembly volume $x$ increases:

$$D(x) = D_\infty + (D_0 - D_\infty) \cdot x^{-\beta}$$

Where:
- $D_0$: Initial defect density during the first production lot.
- $D_\infty$: Asymptotic defect density achievable under mature mass production.
- $\beta > 0$: Defect learning rate parameter (typically $0.2 \le \beta \le 0.6$).
- $x$: Cumulative production starts (e.g., wafers or PCBAs).

##### B. Semiconductor and Microelectronic Yield Models
To convert defect density $D$ and active circuit/silicon area $A$ into expected yield, four foundational models are applied depending on the spatial distribution and clustering of defects:

```text
====================================================================================================
                        SEMICONDUCTOR & PCBA YIELD MODELS
====================================================================================================
Model Name         Mathematical Formulation                           Physical Assumption
----------------------------------------------------------------------------------------------------
Poisson Model      $Y = e^{-D \cdot A}$                               Defects distributed completely
(Elementary)                                                          at random (zero spatial clustering)
----------------------------------------------------------------------------------------------------
Murphy's Model     $Y = \left(\frac{1 - e^{-D \cdot A}}{D \cdot A}\right)^2$ Triangular distribution of defect
(Averaged)                                                            density across wafer / lot surface
----------------------------------------------------------------------------------------------------
Seeds' Model       $Y = e^{-\sqrt{D \cdot A}}$                        High-density localized clustering
(Empirical)                                                           with large defect-free regions
----------------------------------------------------------------------------------------------------
Stapper's Model    $Y = \left(1 + \frac{D \cdot A}{\alpha}\right)^{-\alpha}$ Negative Binomial distribution with
(Negative Binomial)                                                   clustering parameter $\alpha$ ($0.5 \le \alpha \le 2.0$)
====================================================================================================
```

In Stapper's Negative Binomial model:
- As the clustering parameter $\alpha \to \infty$, Stapper's model mathematically converges to the Poisson distribution:
  $$\lim_{\alpha \to \infty} \left(1 + \frac{D \cdot A}{\alpha}\right)^{-\alpha} = e^{-D \cdot A}$$
- For small $\alpha$ (e.g., $\alpha = 0.5$ or $1.0$), defects cluster tightly in localized regions (e.g., edge-of-wafer or localized SMT solder bridge zones). Defect clustering **increases** overall yield relative to the Poisson model because multiple defects land on the same already-defective die or board, leaving more units completely defect-free.

##### C. Asymptotic Manufacturing Yield Ramp Formulation
For discrete board-level assembly and final box-build, yield over ramp time $t$ (or cumulative lots $t$) is modeled via an asymptotic exponential or Gompertz progression:

$$Y(t) = Y_{\max} - (Y_{\max} - Y_0) \cdot e^{-\lambda \cdot t}$$

Where:
- $Y_0$: Initial First-Pass Yield at Week 1 of PVT/Ramp.
- $Y_{\max}$: Theoretical mature line yield (e.g., 0.985).
- $\lambda$: Engineering resolution velocity parameter ($\text{week}^{-1}$).

#### 3. Multi-Stage Assembly and Roll-Through Yield (RTY)
An electronic hardware manufacturing routing consists of $K$ sequential workstations:
1. SMT Line Top Side (Stencil $\to$ Pick-and-Place $\to$ Reflow $\to$ AOI)
2. SMT Line Bottom Side (Stencil $\to$ Pick-and-Place $\to$ Reflow $\to$ AOI)
3. In-Circuit Test (ICT) / Automated X-ray Inspection (AXI)
4. Functional Circuit Test (FCT) and RF Calibration
5. Thermal Stress Screening / Burn-In Chamber (HASS)
6. Final System Integration and Packout

The **Roll-Through Yield (RTY)** across all $K$ stations without rework is:

$$RTY = \prod_{k=1}^K FPY_k$$

If rework is permitted at station $k$, let:
- $FPY_k$: First-pass yield of station $k$.
- $Y_{rwk,k}$: Rework success probability (yield of the rework process itself).
- $\delta_k$: Maximum allowable rework cycles (typically $\delta = 1$ in high-reliability aerospace/telecom to prevent PCB trace delamination).

The effective yield of station $k$ with a single rework loop is:

$$Y_{eff,k} = FPY_k + (1 - FPY_k) \cdot Y_{rwk,k}$$

The cumulative effective line yield is:

$$Y_{eff,\text{line}} = \prod_{k=1}^K Y_{eff,k} = \prod_{k=1}^K \left[ FPY_k + (1 - FPY_k) \cdot Y_{rwk,k} \right]$$

To deliver a net production requirement of $N_{\text{good}}$ finished, shippable units, the factory must initiate input material starts $N_{\text{start}}$:

$$N_{\text{start}} = \left\lceil \frac{N_{\text{good}}}{Y_{eff,\text{line}}} \right\rceil$$

Total scrapped units generated across the routing is:

$$N_{\text{scrap}} = N_{\text{start}} - N_{\text{good}} = N_{\text{good}} \left( \frac{1 - Y_{eff,\text{line}}}{Y_{eff,\text{line}}} \right)$$

---

### Step 4: Long-Lead Material Sourcing and NCNR Purchase Commitment Mechanics

In technology hardware, critical component lead times ($L_{\text{lead}}$) frequently exceed customer order visibility ($H_{\text{vis}}$) by a factor of 4 to 8:

$$L_{\text{lead}} \gg H_{\text{vis}}$$

For instance, advanced 5G RF transceivers or AI accelerator ASICs have a 36-week wafer fabrication, packaging, and testing cycle. If the OEM operates with an 8-week firm customer order backlog, the supply chain is forced to commit to binding, Non-Cancelable, Non-Returnable (NCNR) purchase orders 28 weeks into the unforecasted future.

```text
|<─────────────────────── Component Lead Time L_lead (36 Weeks) ────────────────────────>|
|                                                           |<── Firm Backlog (8 Wks) ──>|
▲                                                           ▲                            ▲
PO Placed to Vendor                                         Firm Customer Order Bound    Finished Unit
(100% NCNR Liability Window: 28 Weeks)                      (Cancellation Penalized)     Shipped
```

#### Mathematical Formulation: The NPI Newsvendor under Dual Uncertainty
During NPI ramp, the procurement manager faces two simultaneous sources of stochastic variation:
1. **Demand Uncertainty:** Customer market adoption is random, $D \sim f_D(d)$, with cumulative distribution function $F_D(d)$, mean $\mu_D$, and variance $\sigma_D^2$.
2. **Manufacturing Yield Uncertainty:** The fraction of purchased components that survive assembly into functional systems is random, $Y \sim g_Y(y)$, with mean $\bar{Y}$ and variance $\sigma_Y^2$.

Let:
- $Q$: Quantity of NCNR components ordered at time $t - L_{\text{lead}}$.
- $c$: Purchase price per component paid to supplier.
- $p$: Selling price per finished hardware system ($p > c$).
- $s$: Salvage value of excess NCNR component ($s \ll c$; typically scrap metal or zero).
- $v$: Stockout / underage penalty per unit (lost gross margin $p - c$ plus contractual SLA penalties).

Total good finished units available for sale from an order $Q$ is $X = Y \cdot Q$.
The expected profit function $\Pi(Q)$ is formulated as:

$$\Pi(Q) = \mathbb{E}_{D, Y} \left[ p \cdot \min(D, Y \cdot Q) + s \cdot \max(0, Y \cdot Q - D) - c \cdot Q - v \cdot \max(0, D - Y \cdot Q) \right]$$

Defining economic underage cost $c_u$ and overage cost $c_o$:
- $c_u = p - c + v$ (marginal cost of being short one unit of demand).
- $c_o = c - s$ (marginal cost of having one excess unit of unused NCNR inventory).

If yield were deterministic ($Y = \bar{Y}$), the optimal order quantity $Q^*$ would follow the classical Newsvendor critical fractile:

$$F_D(\bar{Y} \cdot Q^*) = \frac{c_u}{c_u + c_o} \implies Q^* = \frac{F_D^{-1}\left(\frac{c_u}{c_u + c_o}\right)}{\bar{Y}}$$

Under **stochastic yield**, random yield fluctuations compound demand risk. By applying a second-order Taylor expansion around $(\mu_D, \bar{Y})$, the optimal order quantity $Q^*$ must be adjusted upward for expected yield loss and inflated by a variance penalty term:

$$Q^* \approx \frac{F_D^{-1}\left(\frac{c_u}{c_u + c_o}\right)}{\bar{Y}} + \frac{1}{2} \cdot \frac{c_o}{c_u + c_o} \cdot \frac{\sigma_Y^2}{\bar{Y}^3} \cdot f_D\left(\mu_D\right)$$

#### Cumulative NCNR Financial Liability Calculation
At any week $w$ in the NPI ramp, total corporate financial exposure to NCNR commitments across all bill-of-materials components $i \in \mathcal{B}$ is:

$$\text{Liability}_{\text{total}}(w) = \sum_{i \in \mathcal{B}} \sum_{\tau = w}^{w + L_i} \left[ PO_{i,\tau} \cdot c_i \cdot \left(1 - \theta_i(\tau - w)\right) \right] + \text{WIP}_{\text{EMS}}(w) + \text{FG}_{\text{stranded}}(w)$$

Where:
- $PO_{i,\tau}$: Purchase orders placed for component $i$ scheduled for delivery at week $\tau$.
- $c_i$: Contract unit price of component $i$.
- $\theta_i(\Delta t)$: Supplier cancellation mitigation factor ($0 \le \theta_i \le 1$). For raw un-wafered silicon or standard passives, suppliers may accept cancellation with a restocking fee ($\theta > 0$). For fully customized NCNR ASICs, $\theta = 0$ (100% legal liability).
- $\text{WIP}_{\text{EMS}}(w)$: Raw material and work-in-progress inventory currently held on the EMS factory floor for which the OEM is legally indemnified.
- $\text{FG}_{\text{stranded}}(w)$: Finished goods built that cannot be shipped due to missing companion components or failed regulatory sign-off.

#### The Semiconductor Value-Add Liability Curve
In reality, NCNR cancellation liability at a semiconductor foundry (e.g., TSMC, Intel Foundry Services) is not a static binary step function. Legal liability escalates continuously as the custom silicon advances through physical processing milestones. The value-add liability function $\mathcal{L}_{\text{ASIC}}(t)$ across elapsed fabrication time $t \in [0, T_{\text{lead}}]$ is modeled as:

$$\mathcal{L}_{\text{ASIC}}(t) = c_{\text{raw}} + (c_{\text{fab}} - c_{\text{raw}}) \cdot \left(\frac{\min(t, T_{\text{fab}})}{T_{\text{fab}}}\right)^{\gamma_{\text{fab}}} + c_{\text{pkg}} \cdot \mathbb{I}(t > T_{\text{fab}}) + c_{\text{test}} \cdot \mathbb{I}(t > T_{\text{fab}} + T_{\text{pkg}})$$

Where:
- $c_{\text{raw}}$: Raw silicon wafer ingot cost (typically $10\%–15\%$ of total finished die cost).
- $c_{\text{fab}}$: Cumulative front-end photolithography, ion implantation, and chemical vapor deposition cost ($50\%–65\%$).
- $c_{\text{pkg}}$: Advanced substrate packaging (e.g., 2.5D/3D chiplet CoWoS, flip-chip BGA) cost ($20\%–30\%$).
- $c_{\text{test}}$: Final wafer sort, automated test equipment (ATE), and burn-in screening cost ($5\%–10\%$).
- $\gamma_{\text{fab}} \approx 1.4$: Convex non-linear cost accumulation exponent reflecting the escalating capital intensity of late-stage metallization and chemical-mechanical planarization (CMP) layers.

Understanding this progression allows procurement negotiators to structure **milestone-cancellable contracts**: if an OEM discovers a fatal DVT architecture defect at week 12 of a 26-week lead time, terminating wafer processing before advanced packaging can salvage $35\%$ to $45\%$ of the total purchase order face value.

---

### Step 5: Manufacturing Line Balancing and Capacity Ramp Optimization

Transitioning an NPI design from low-rate pilot production to volume manufacturing requires balancing takt time, sizing test station clusters, and preventing bonepile saturation.

#### 1. Takt Time and Required Line Pace
Takt time ($T_{\text{takt}}$) is determined strictly by customer demand pace:

$$T_{\text{takt}} = \frac{T_{\text{net}}}{D_{\text{target}}}$$

Where:
- $T_{\text{net}}$: Net available operating time per shift or per day (total shift seconds minus planned maintenance, lunch, and shift-change breaks). For a 2-shift operation of 8 hours each with 60 minutes break per shift:
  $$T_{\text{net}} = 2 \times (8 - 1) \times 3600 = 50,400 \text{ seconds/day}$$
- $D_{\text{target}}$: Required daily production output of good units.

To produce $D_{\text{target}} = 1,000\text{ units/day}$, the line takt time must be:
$$T_{\text{takt}} = \frac{50,400}{1,000} = 50.4 \text{ seconds/unit}$$

#### 2. Station Cycle Time Balancing and Sizing
Let the manufacturing routing consist of $M$ sequential operations, each with task time $t_i$. If task time $t_i > T_{\text{takt}}$, multiple parallel machines or stations $k_i$ must be installed:

$$k_i = \left\lceil \frac{t_i}{T_{\text{takt}} \cdot \eta_i \cdot FPY_i} \right\rceil$$

Where:
- $\eta_i$: Machine operational availability / OEE (Overall Equipment Effectiveness, typically $0.85 \le \eta \le 0.92$).
- $FPY_i$: First-pass yield of station $i$. If a station has low yield, it must process extra volume to account for fallout, requiring additional capacity.

Line Balancing Efficiency ($\text{LBE}$) across the line is:

$$\text{LBE} = \frac{\sum_{i=1}^M t_i}{\left(\sum_{i=1}^M k_i\right) \cdot \max_i\left(\frac{t_i}{k_i}\right)}$$

#### 3. Sizing Thermal Stress and Burn-In Capacity via Little's Law
In high-reliability hardware (telecom base stations, servers, avionics), units must undergo extended thermal soak / burn-in screening to precipitate early infant mortality failures. While SMT placement takes seconds, burn-in duration ($T_{\text{burn}}$) ranges from 24 to 72 hours.

By Little's Law, the required burn-in chamber slot capacity ($WIP_{\text{burn}}$) is:

$$WIP_{\text{burn}} = \lambda_{\text{arrival}} \cdot T_{\text{burn}}$$

If the line feeds burn-in at an arrival rate of $\lambda_{\text{arrival}} = \frac{D_{\text{target}}}{FPY_{\text{pre-burn}}}$ units per hour:

$$\text{Required Slots } S^* = \left\lceil \frac{\left(\frac{D_{\text{target}}}{24 \cdot FPY_{\text{pre-burn}}}\right) \cdot T_{\text{burn}}}{A_{\text{chamber}} \cdot (1 - \rho_{\text{unusable}})} \right\rceil$$

Where $A_{\text{chamber}}$ is chamber availability (uptime) and $\rho_{\text{unusable}}$ is the slot loss factor due to thermal ramp-up and cool-down cycles.

#### 4. Bonepile Dynamics and Rework Equilibrium
Defective units identified at test stations exit the primary line into a rework loop. Let:
- $\lambda_{\text{fallout}}(t) = N_{\text{start}}(t) \cdot (1 - FPY(t))$: Arrival rate of failed units into the debug/rework queue.
- $\mu_{\text{rework}}$: Processing rate of the technical rework and debug engineering team (units per day).
- $B(t)$: Cumulative units trapped in the bonepile at time $t$.

The rate of bonepile accumulation is governed by the differential equation:

$$\frac{dB(t)}{dt} = \lambda_{\text{fallout}}(t) - \min\left(\mu_{\text{rework}}, B(t)\right)$$

During early NPI ramp (PVT to Week 4 of MP), if $\lambda_{\text{fallout}}(t) > \mu_{\text{rework}}$, the bonepile grows monotonically. Unchecked bonepile growth has catastrophic operational consequences:
1. High-value microprocessors and memory chips trapped in WIP cause working capital to explode.
2. If an ECO is released while units reside in the bonepile, the entire bonepile must be scrapped or undergo dual rework.
3. Defective components in the bonepile cannot be returned to suppliers for warranty credit within contractual Return Material Authorization (RMA) claim windows.

A mandatory NPI operating rule is the **Bonepile Cap Enforcement**: if $B(t) > B_{\max}$, primary SMT line starts are automatically throttled until debug capacity clears the backlog.

---

## 6. Worked example

### Problem Scenario
An enterprise telecommunications equipment OEM is launching a next-generation 5G Massive MIMO Active Antenna Unit (AAU). The product contains:
- 1 main digital processing board with a custom 5nm baseband ASIC and 4 high-bandwidth memory (HBM) modules.
- 1 64-channel RF transceiver board with power amplifiers.
- A high-precision die-cast aluminum heat sink chassis requiring hardened steel tooling.

The program is entering PVT and preparing for a 12-week Mass Production ramp to fulfill a Tier-1 mobile network operator's network deployment rollout.

```text
====================================================================================================
                        5G AAU NPI RAMP MODELING PARAMETERS
====================================================================================================
Parameter                                             Symbol          Assigned Value
----------------------------------------------------------------------------------------------------
Customer Demand (Target Delivered Good Units)         $D_{\text{tgt}}$ 24,000 units total over 12 wks
Weekly Demand Schedule (Weeks 1 to 12)                $d_t$           [500, 1000, 1500, 2000, 2500,
                                                                       2500, 2500, 2500, 2500, 2500,
                                                                       2000, 2000]
Available Operating Time per Week (5 days, 2 shifts)  $T_{\text{net}}$ 50,400 sec/day = 252,000 sec/wk
Tooling Cavitation Options:
  Option A: 2-Cavity Die-Cast Tool                    $M_A$           2 cavities; Capex = $180,000;
                                                                      Cycle time = 75 sec; Lead = 14 wks
  Option B: 4-Cavity Die-Cast Tool                    $M_B$           4 cavities; Capex = $340,000;
                                                                      Cycle time = 110 sec; Lead = 18 wks
Machine Hourly Rate for Die-Casting Press                             $120 / hour
Die-Cast Scrap Rate                                   $s_{\text{mold}}$ 4.0% (0.04)
Yield Learning Parameters:
  Initial First-Pass Yield (Week 1)                   $Y_0$           68.0% (0.68)
  Mature Target Line Yield (Asymptotic Ceiling)       $Y_{\max}$      96.0% (0.96)
  Engineering Resolution Velocity                     $\lambda$       0.32 week$^{-1}$
Manufacturing Routing Stations:
  Station 1: SMT Placement & Reflow (Top + Bottom)    $t_1$           42 seconds / board
  Station 2: Automated Optical Inspection (AOI)       $t_2$           18 seconds / board
  Station 3: In-Circuit Testing (ICT)                 $t_3$           35 seconds / board
  Station 4: RF Calibration & Functional Test (FCT)   $t_4$           180 seconds / unit
  Station 5: Thermal Burn-In Screening                $T_{\text{burn}}$ 48 hours continuous soak
  Station 6: Final System Assembly & Packout          $t_6$           60 seconds / unit
Rework Feasibility:
  SMT / ICT Defect Rework Yield                       $Y_{\text{rwk}}$ 85.0% (0.85)
  Burn-In Defect Rework Feasibility                                   0.0% (Failures scrapped)
Component Procurement Parameters:
  Custom Baseband ASIC Unit Cost                      $c_{\text{ASIC}}$ $220.00 (NCNR)
  Lead Time for ASIC Fabrication & Packaging          $L_{\text{ASIC}}$ 26 weeks
  Salvage Value of Excess ASIC                        $s$             $0.00
  Stockout Penalty per Unit (Lost Margin + SLA fine)  $v$             $850.00
  System Selling Price                                $p$             $2,400.00
Rework Engineering Team Capacity                      $\mu_{\text{rwk}}$ 150 units / week
====================================================================================================
```

---

### Step-by-Step Mathematical Calculations

#### 1. Tooling Cavitation Selection
Calculate weekly production capacity and part manufacturing cost for Option A (2-cavity) versus Option B (4-cavity).

For Option A (2 cavities, 75 sec cycle time):
$$\text{Shots per hour} = \frac{3600}{75} = 48 \text{ shots/hour}$$
$$\text{Gross parts per hour} = 48 \times 2 = 96 \text{ parts/hour}$$
$$\text{Net good parts per hour} = 96 \times (1 - 0.04) = 92.16 \text{ parts/hour}$$
$$\text{Weekly Capacity (70 operating hours/week)} = 70 \times 92.16 = 6,451 \text{ parts/week}$$
$$\text{Machine Run Cost per Part} = \frac{\$120 / \text{hour}}{92.16 \text{ parts/hour}} = \$1.302 \text{ per part}$$

For Option B (4 cavities, 110 sec cycle time):
$$\text{Shots per hour} = \frac{3600}{110} = 32.727 \text{ shots/hour}$$
$$\text{Gross parts per hour} = 32.727 \times 4 = 130.91 \text{ parts/hour}$$
$$\text{Net good parts per hour} = 130.91 \times (1 - 0.04) = 125.67 \text{ parts/hour}$$
$$\text{Weekly Capacity (70 operating hours/week)} = 70 \times 125.67 = 8,797 \text{ parts/week}$$
$$\text{Machine Run Cost per Part} = \frac{\$120 / \text{hour}}{125.67 \text{ parts/hour}} = \$0.955 \text{ per part}$$

**Decision Analysis:**
- Peak weekly customer demand is 2,500 units. Accounting for assembly yield fallout (initial yield ~68%), peak chassis demand will reach $\approx 2,500 / 0.85 = 2,941$ chassis/week.
- Option A provides 6,451 parts/week, which is $219\%$ of peak requirement.
- Cost savings per part for Option B is $\$1.302 - \$0.955 = \$0.347$.
- To amortize the $\$160,000$ Capex difference ($\$340,000 - \$180,000$), required volume is:
  $$\text{Breakeven Volume} = \frac{\$160,000}{\$0.347} = 461,095 \text{ units}$$
- The 12-week program volume is only 24,000 units (annual volume ~60,000 units). Furthermore, Option B requires an additional 4 weeks of tooling lead time (18 vs. 14 weeks), jeopardizing the PVT launch date.
- **Selection: Option A (2-Cavity Tool)** is quantitatively selected.

---

#### 2. Yield Ramp Curve and Weekly Line Yield
Calculate First-Pass Yield ($FPY_t$), effective yield with rework ($Y_{\text{eff},t}$), and required production starts for all 12 weeks:

$$FPY(t) = Y_{\max} - (Y_{\max} - Y_0) \cdot e^{-\lambda(t - 1)} = 0.96 - (0.96 - 0.68) \cdot e^{-0.32(t - 1)}$$

Assuming non-burn-in defects (representing 80% of all fallout) can be reworked with $Y_{\text{rwk}} = 0.85$:
$$Y_{\text{eff}}(t) = FPY(t) + \left( (1 - FPY(t)) \cdot 0.80 \cdot 0.85 \right) = FPY(t) + 0.68 \cdot (1 - FPY(t))$$
$$Y_{\text{eff}}(t) = 0.68 + 0.32 \cdot FPY(t)$$

Calculate values for selected milestone weeks:
- **Week 1 ($t = 1$):**
  $$FPY(1) = 0.96 - 0.28 \cdot e^0 = 0.6800 \quad (68.0\%)$$
  $$Y_{\text{eff}}(1) = 0.68 + 0.32 \cdot (0.68) = 0.8976 \quad (89.76\%)$$
  $$\text{Target Demand } d_1 = 500 \implies N_{\text{start}}(1) = \left\lceil \frac{500}{0.8976} \right\rceil = 558 \text{ starts}$$
  $$\text{Defect Fallout } = 558 - 500 = 58 \text{ units requiring debug/rework or scrap}$$

- **Week 4 ($t = 4$):**
  $$FPY(4) = 0.96 - 0.28 \cdot e^{-0.32 \times 3} = 0.96 - 0.28 \cdot e^{-0.96} = 0.96 - 0.28(0.3829) = 0.96 - 0.1072 = 0.8528 \quad (85.28\%)$$
  $$Y_{\text{eff}}(4) = 0.68 + 0.32 \cdot (0.8528) = 0.9529 \quad (95.29\%)$$
  $$\text{Target Demand } d_4 = 2,000 \implies N_{\text{start}}(4) = \left\lceil \frac{2,000}{0.9529} \right\rceil = 2,099 \text{ starts}$$

- **Week 8 ($t = 8$):**
  $$FPY(8) = 0.96 - 0.28 \cdot e^{-0.32 \times 7} = 0.96 - 0.28 \cdot e^{-2.24} = 0.96 - 0.28(0.1065) = 0.9302 \quad (93.02\%)$$
  $$Y_{\text{eff}}(8) = 0.68 + 0.32 \cdot (0.9302) = 0.9777 \quad (97.77\%)$$
  $$\text{Target Demand } d_8 = 2,500 \implies N_{\text{start}}(8) = \left\lceil \frac{2,500}{0.9777} \right\rceil = 2,558 \text{ starts}$$
- **Week 12 ($t = 12$):**
  $$FPY(12) = 0.96 - 0.28 \cdot e^{-0.32 \times 11} = 0.96 - 0.28 \cdot e^{-3.52} = 0.96 - 0.28(0.0296) = 0.9517 \quad (95.17\%)$$
  $$Y_{\text{eff}}(12) = 0.68 + 0.32 \cdot (0.9517) = 0.9845 \quad (98.45\%)$$
  $$\text{Target Demand } d_{12} = 2,000 \implies N_{\text{start}}(12) = \left\lceil \frac{2,000}{0.9845} \right\rceil = 2,032 \text{ starts}$$

```text
====================================================================================================
                        COMPLETE 12-WEEK RAMP SCHEDULE TABLE
====================================================================================================
Week (t)  Demand (d_t)  FPY (Y_t)  Eff. Yield (Y_eff)  Starts (N_start)  Fallout Units  Scrapped Units
----------------------------------------------------------------------------------------------------
W1        500           68.00%     89.76%              558               179            58
W2        1,000         75.66%     92.21%              1,085             265            85
W3        1,500         81.23%     93.99%              1,596             300            96
W4        2,000         85.28%     95.29%              2,099             309            99
W5        2,500         88.22%     96.23%              2,598             307            98
W6        2,500         90.35%     96.91%              2,580             250            80
W7        2,500         91.90%     97.41%              2,567             209            67
W8        2,500         93.02%     97.77%              2,558             179            58
W9        2,500         93.84%     98.03%              2,551             158            51
W10       2,500         94.43%     98.22%              2,546             142            46
W11       2,000         94.86%     98.36%              2,034             105            34
W12       2,000         95.17%     98.45%              2,032              99            32
----------------------------------------------------------------------------------------------------
TOTAL     24,000        —          —                   24,804            2,502          804
====================================================================================================
```
*Note:* Total starts required over the 12-week ramp is 24,804 units to deliver 24,000 good units. Cumulative scrap across the program is exactly 804 units ($3.24\%$ scrap allowance). Total un-reworked fallout requiring technical debugging is 2,502 units.

---

#### 3. Burn-In Chamber Sizing
Determine the required capacity for thermal burn-in screening.
- Peak weekly production starts occur at Week 5: $N_{\text{start}} = 2,598$ units/week.
- Burn-in duration: $T_{\text{burn}} = 48\text{ hours}$.
- Factory operates 5 days/week (120 operating hours/week).
- Chamber availability factor: $A_{\text{ch}} = 92.0\%$ (allowing 8% downtime for chamber loading, thermal ramp-up to $65^\circ\text{C}$, dwell, and cool-down).

The required active chamber slot capacity is:
$$\text{Throughput Arrival Rate } \lambda = \frac{2,598 \text{ units}}{120 \text{ hours}} = 21.65 \text{ units/hour}$$
$$\text{By Little's Law: } WIP_{\text{burn}} = \lambda \cdot T_{\text{burn}} = 21.65 \times 48 = 1,039.2 \text{ units}$$
$$\text{Required Physical Slots } S^* = \left\lceil \frac{1,039.2}{0.92} \right\rceil = 1,130 \text{ slots}$$

If standard industrial thermal chambers contain 192 test slots per chamber:
$$\text{Number of Chambers} = \left\lceil \frac{1,130}{192} \right\rceil = 6 \text{ chambers}$$
Total available slots across 6 chambers: $6 \times 192 = 1,152\text{ slots}$, providing a safe operating margin.

---

#### 4. Bottleneck Line Balancing and Takt Time
At peak production (Weeks 5 to 10), target output is 2,500 units/week over 252,000 net operating seconds:
$$T_{\text{takt}} = \frac{252,000}{2,500} = 100.8 \text{ seconds/unit}$$

Calculate required station quantities $k_i$ across all operations:
- **Station 1 (SMT Placement):** Task time $t_1 = 42\text{ sec}$.
  $$k_1 = \left\lceil \frac{42}{100.8 \times 0.90} \right\rceil = 1 \text{ dual-gantry SMT line}$$
- **Station 2 (AOI Inspection):** Task time $t_2 = 18\text{ sec}$.
  $$k_2 = \left\lceil \frac{18}{100.8 \times 0.95} \right\rceil = 1 \text{ AOI machine}$$
- **Station 3 (ICT Testing):** Task time $t_3 = 35\text{ sec}$.
  $$k_3 = \left\lceil \frac{35}{100.8 \times 0.90} \right\rceil = 1 \text{ ICT fixture}$$
- **Station 4 (RF Calibration & FCT):** Task time $t_4 = 180\text{ sec}$.
  $$k_4 = \left\lceil \frac{180}{100.8 \times 0.88 \times 0.90} \right\rceil = \left\lceil \frac{180}{79.83} \right\rceil = 3 \text{ parallel RF test stations}$$
- **Station 6 (Final Packout):** Task time $t_6 = 60\text{ sec}$.
  $$k_6 = \left\lceil \frac{60}{100.8 \times 0.95} \right\rceil = 1 \text{ packaging station}$$

**Bottleneck Analysis:** Station 4 (RF Calibration) requires 3 parallel RF shielded test enclosures with automated robotic handling. If only 2 stations were installed, station cycle time would be $180 / 2 = 90\text{ sec}$, but with OEE and re-test yield, maximum throughput would be restricted to $\approx 1,800\text{ units/week}$, creating a catastrophic line bottleneck that chokes the entire factory.

---

#### 5. NCNR Component Commitment and Financial Liability
The custom 5nm Baseband ASIC costs $c = \$220.00$ and has a 26-week lead time ($L = 26$).
At Week $-26$, the OEM must commit to a binding NCNR purchase order for the 12-week ramp.
- Total expected program starts: $\bar{N}_{\text{start}} = 24,804$ units.
- Underage cost: $c_u = (p - c) + v = (\$2,400 - \$220) + \$850 = \$3,030$ per unit.
- Overage cost: $c_o = c - s = \$220 - \$0 = \$220$ per unit.
- Critical Fractile:
  $$CF = \frac{c_u}{c_u + c_o} = \frac{3,030}{3,030 + 220} = \frac{3,030}{3,250} = 0.9323 \quad (93.23\%)$$

Due to high asymmetric shortage penalties ($c_u / c_o = 13.77$), the supply chain must buffer against both demand surges and early ramp yield fallout.
If total program demand across the 12 weeks has forecast standard deviation $\sigma_D = 2,500$ units and yield variance adds equivalent standard deviation $\sigma_Y \cdot \bar{N} = 650$ units:
$$\sigma_{\text{total}} = \sqrt{2,500^2 + 650^2} = 2,583 \text{ units}$$
$$z_{0.9323} = \Phi^{-1}(0.9323) = 1.493$$
$$Q^* = \bar{N}_{\text{start}} + z \cdot \sigma_{\text{total}} = 24,804 + (1.493 \times 2,583) = 24,804 + 3,856 = 28,660 \text{ ASICs}$$

**Total NCNR Financial Exposure:**
$$\text{Cash Liability} = 28,660 \times \$220.00 = \$6,305,200.00$$
This $\$6.31\text{M}$ commitment is binding 26 weeks before production. If the OEM's customer cancels the 5G rollout or delays deployment, the OEM has zero return rights and carries full financial liability.

---

## 7. Code, query, or calculation

The following self-contained Python script simulates the NPI production ramp, computes Wright/Crawford defect learning, tracks required starts, sizes the burn-in chamber bottleneck, and monitors bonepile accumulation:

```python
import math

def simulate_npi_ramp():
    demand = [500, 1000, 1500, 2000, 2500, 2500, 2500, 2500, 2500, 2500, 2000, 2000]
    y_0, y_max, lam = 0.68, 0.96, 0.32
    rework_yield = 0.85
    rework_capacity = 150  # Max units debugged/reworked per week
    burn_in_hrs, weekly_hrs, ch_slots = 48, 120, 192

    bonepile = 0
    cum_starts, cum_scrap = 0, 0
    print(f"{'Wk':>2} | {'Dem':>5} | {'FPY':>6} | {'Starts':>6} | {'Fallout':>7} | {'Bonepile':>8} | {'Chambers':>8}")
    print("-" * 58)

    for t, d in enumerate(demand, start=1):
        fpy = y_max - (y_max - y_0) * math.exp(-lam * (t - 1))
        # Effective yield: non-burn-in defects (80%) reworked at rework_yield
        y_eff = fpy + (1.0 - fpy) * 0.80 * rework_yield
        starts = math.ceil(d / y_eff)
        fallout = math.ceil(starts * (1.0 - fpy))
        scrapped = starts - d

        # Bonepile dynamic: arrivals are reworkable fallout (80%)
        rework_queue = math.ceil(fallout * 0.80)
        rework_done = min(rework_queue + bonepile, rework_capacity)
        bonepile = max(0, bonepile + rework_queue - rework_done)

        # Burn-in chambers required
        slots_needed = math.ceil((starts / weekly_hrs) * burn_in_hrs / 0.92)
        chambers_needed = math.ceil(slots_needed / ch_slots)

        cum_starts += starts
        cum_scrap += scrapped

        print(f"{t:2d} | {d:5d} | {fpy*100:5.1f}% | {starts:6d} | {fallout:7d} | {bonepile:8d} | {chambers_needed:8d}")

    print("-" * 58)
    print(f"Total Starts: {cum_starts:,} | Total Scrap: {cum_scrap:,} | Net Delivered: {sum(demand):,}")

if __name__ == "__main__":
    simulate_npi_ramp()
```

### Execution Output and Interpretation
```text
Wk |   Dem |    FPY | Starts | Fallout | Bonepile | Chambers
----------------------------------------------------------
 1 |   500 |  68.0% |    558 |     179 |        0 |        2
 2 |  1000 |  75.7% |   1085 |     265 |       62 |        3
 3 |  1500 |  81.2% |   1596 |     300 |      152 |        4
 4 |  2000 |  85.3% |   2099 |     309 |      250 |        5
 5 |  2500 |  88.2% |   2598 |     307 |      346 |        6
 6 |  2500 |  90.3% |   2580 |     250 |      396 |        6
 7 |  2500 |  91.9% |   2567 |     209 |      414 |        6
 8 |  2500 |  93.0% |   2558 |     179 |      408 |        6
 9 |  2500 |  93.8% |   2551 |     158 |      385 |        6
10 |  2500 |  94.4% |   2546 |     142 |      349 |        6
11 |  2000 |  94.9% |   2034 |     105 |      283 |        5
12 |  2000 |  95.2% |   2032 |      99 |      213 |        5
----------------------------------------------------------
Total Starts: 24,804 | Total Scrap: 804 | Net Delivered: 24,000
```

#### Analytical Insights from the Model
1. **The Bonepile Crisis:** Rework queue arrivals exceed rework technician throughput ($\mu = 150$) between Weeks 2 and 7. The bonepile peaks at Week 7 with **414 defective boards** trapped in off-line debug. This traps over $\$250,000$ in active WIP. Management must authorize temporary contractor rework technicians between Weeks 3 and 8 to suppress bonepile growth below 150 units.
2. **Thermal Chamber Saturation:** Sizing for steady-state volume would require only 5 chambers. However, during the peak ramp surge (Weeks 5–10), required slots reach 1,130, mandating exactly 6 chambers. Operating with 5 chambers would force a 15% reduction in factory output.

---

## 8. Data, standards, and regulatory context

Hardware NPI governance operates under a rigorous hierarchy of international engineering standards, reliability frameworks, and regulatory mandates:

```text
====================================================================================================
                        INTERNATIONAL HARDWARE NPI STANDARDS
====================================================================================================
Standard Body      Standard Reference                                 Operational Scope & Mandate
----------------------------------------------------------------------------------------------------
IPC                IPC-A-610H                                         Acceptability of Electronic Assemblies:
                                                                      Class 1 (General), Class 2 (Dedicated
                                                                      Service), Class 3 (High-Reliability / Telecom)
----------------------------------------------------------------------------------------------------
IPC                IPC-7351B                                          Generic Requirements for Surface Mount Design
                                                                      and Land Pattern Standard (DFM Rules)
----------------------------------------------------------------------------------------------------
JEDEC              JESD22-A104 / A110 / B111                          Reliability Stress Test Methods: Temperature
                                                                      Cycling, HAST, Board-Level Drop Test
----------------------------------------------------------------------------------------------------
Telcordia          GR-63-CORE / GR-1089-CORE                          NEBS (Network Equipment-Building System):
                                                                      Earthquake, Fire, Thermal Shock, EMC for Telecom
----------------------------------------------------------------------------------------------------
SAE Aerospace      AS9100D / AS9102 Rev C                             Aerospace First Article Inspection (FAI)
                                                                      Requirements and Metrology Verification
----------------------------------------------------------------------------------------------------
AIAG               AIAG PPAP (4th Edition)                            Production Part Approval Process: Level 1 to 5
                                                                      Part Submission Warrants (PSW) and $C_{pk}$ Audits
----------------------------------------------------------------------------------------------------
ISO                ISO 9001:2015 / ISO 13485:2016                    Design Transfer Protocols and Quality Management
                                                                      System Requirements for Mass Production Release
----------------------------------------------------------------------------------------------------
FCC / CE           FCC Title 47 CFR Part 15 / RED Directive           Electromagnetic Compatibility (EMC), RF Spectral
                   (2014/53/EU)                                       Emissions, and Telecom Equipment Type Approval
====================================================================================================
```

### Binding Regulatory Deadlines vs. Operational Conventions
1. **Mandatory Type Approval:** An enterprise telecommunications base station or wireless router cannot be legally energized or connected to commercial networks in the United States without formal FCC certification under Title 47 CFR Part 15/96. Securing FCC grants requires testing final, production-intent hardware (PVT units with production enclosures) in accredited anechoic chambers. Any post-PVT board layout revision voids the FCC grant, requiring a 6-to-10 week re-test cycle.
2. **First Article Inspection (AS9102):** In aerospace and high-reliability telecom, AS9102 Level 3 FAI requires 100% verification of all drawing dimensions for the first production run off hard tooling. Every single mold cavity must be independently serialized and measured. Parts from Cavity 1 cannot be used to certify Cavity 4.
3. **Hazardous Substances (RoHS 3 / REACH):** All components sourced during NPI must provide verifiable Full Material Declarations (FMD) certifying compliance with EU RoHS Directive 2015/863 (restricting lead, cadmium, mercury, hexavalent chromium, and phthalates). Sourcing non-compliant components on a prototype NPI build triggers factory quarantine and total batch rejection.
4. **Telcordia NEBS Level 3 Physical Architecture Qualification:** Tier-1 telecommunications carriers (e.g., AT&T, Verizon, Deutsche Telekom) contractually mandate that central office and cell site equipment achieve full NEBS Level 3 compliance prior to commercial deployment:
   - *Seismic Zone 4 Earthquake Testing:* Fully assembled racks and chassis loaded with production PCBAs are mounted on a tri-axial hydraulic shaker table. Synthesized earthquake acceleration waveforms (1.0g to 5.0g spectrum from 1 Hz to 50 Hz) are applied across all three orthogonal axes. Connectors must maintain electrical continuity with zero micro-interruptions ($>10 \text{ ns}$ disconnects cause carrier network packet drops).
   - *Environmental Thermal Soak:* Systems undergo continuous operation under extreme thermal envelopes ($-5^\circ\text{C}$ to $+55^\circ\text{C}$ continuous; short-term thermal excursions from $-40^\circ\text{C}$ to $+65^\circ\text{C}$ at $90\%$ relative humidity).
   - *Fire Spread Resistance:* Polymer components, fan trays, and circuit boards must meet UL 94 V-0 flammability ratings and undergo needle-flame self-extinguishing certification.
5. **JEDEC Stress and Mechanical Shock Regimes:**
   - *Thermal Fatigue Profiling (JESD22-A104):* Assembled PCBAs are subjected to 1,000 thermal cycles between $-40^\circ\text{C}$ and $+125^\circ\text{C}$ with a 15-minute dwell time. Solder joint fatigue is governed by the Coffin-Manson plastic strain relationship:
     $$N_f = C \cdot (\Delta \epsilon_p)^{-\beta_w} \cdot e^{\frac{E_a}{k_B T_{\max}}}$$
     where any thermal expansion mismatch between silicon BGA packages ($\approx 3 \text{ ppm}/^\circ\text{C}$) and FR4 laminates ($\approx 16 \text{ ppm}/^\circ\text{C}$) that shears solder balls causes immediate NPI qualification failure.
   - *Dynamic Shock and Drop Integrity (JESD22-B111):* Test vehicles undergo 30 consecutive drops at $1,500\text{g}$ peak acceleration with a $0.5\text{ ms}$ half-sine pulse to verify that corner BGA solder joints do not crack or crater into the PCB pad copper laminate.

---

## 9. Industry practice and case evidence

### Case 1: Apple Inc. — The Dual-Tooling and Capex-Funded EMS Ramp Model
**Documented Practice:** Apple operates the most aggressive consumer electronics hardware NPI ramp in industrial history, scaling production of flagship iPhones from zero to over 50 million units within a 12-week window each autumn.
- **Tooling Duplication and Parallel Lines:** To circumvent single-cavity cycle time constraints, Apple funds billions of dollars in custom tooling and CNC machining centers installed directly on the factory floors of its EMS partners (Foxconn, Pegatron, Luxshare). For critical structural components (such as unibody aluminum or titanium enclosures), Apple deploys thousands of custom CNC machines operating in parallel.
- **The EVT-to-PVT Compression:** Apple engineering teams reside on-site at EMS facilities in Shenzhen, Zhengzhou, and Chennai during DVT and PVT. Rather than sequential tooling sampling, Apple cuts duplicate "soft" and "hard" tooling simultaneously during EVT, sacrificing tooling capital to accelerate time-to-market by 8 to 12 weeks.
- **Analytical Inference:** Apple's ability to absorb tens of millions of dollars in discarded tooling scrap is economically viable only because their gross margins (>40%) and massive launch volumes make time-to-market delays exponentially more costly than discarded steel tooling.

### Case 2: Tesla Model 3 — "Production Hell" and the Automation Bottleneck (2017–2018)
**Documented Fact:** In mid-2017, Tesla launched the Model 3 aiming for an immediate ramp to 5,000 cars per week. By late 2017, the Fremont factory was producing fewer than 500 cars per week, threatening corporate liquidity.
- **The Mechanistic Cause:** Tesla attempted to bypass classical DVT/PVT soft-tooling stages by moving directly from computer simulations to fully automated mass production lines (the "machine that builds the machine"). Highly complex automated robotic conveyance systems jammed when handling non-rigid materials (wiring harnesses and battery packs). Furthermore, battery module assembly lines suffered severe thermal weld yield fallout, saturating rework loops.
- **The Resolution:** Elon Musk publicly acknowledged that "excessive automation at Tesla was a mistake." Tesla disassembled multi-million-dollar automated conveyance systems, installed manual assembly lines inside a temporary sprung-structure tent outside the Fremont factory, and recalibrated takt times to achieve the 5,000 unit/week threshold.
- **Takeaway:** Attempting to compress NPI by eliminating physical PVT line validation and replacing manual pilot learning with unproven high-speed automation leads directly to systemic line paralysis.

### Case 3: Cisco Systems — The 2021 Component Allocation and NCNR Liability Exposure
**Documented Fact:** During the 2020–2022 global semiconductor shortage, Cisco faced product backlogs exceeding $\$15$ billion due to shortages of specialized power management ICs, legacy microcontrollers, and custom networking ASICs with lead times exceeding 52 weeks.
- **The Operational Mechanism:** To secure foundry capacity, Cisco entered into multi-billion-dollar binding NCNR purchase commitments with semiconductor foundries and component distributors. As lead times stretched, customers placed duplicate orders across multiple networking vendors.
- **The Bullwhip Whiplash:** When global enterprise IT spending contracted in 2023, customer demand evaporated. Cisco was left with billions of dollars in binding NCNR purchase commitments and elevated component inventory on its balance sheet. Cisco was forced to slow factory intake, negotiate multi-million-dollar inventory deferral fees with contract manufacturers, and record significant excess and obsolescence (E&O) reserves.

---

## 10. Failure modes and red flags

| # | Failure Mode | Early Warning Sign | Root Physical / Operational Mechanism | Downstream Consequence | Mitigation / Engineering Action |
|---|---|---|---|---|---|
| 1 | **Premature Tooling Freeze** | High volume of ECOs issued while mold base steel is being machined. | Authorizing hard tooling during early EVT before completing thermal and shock testing. | Tooling must be welded, re-machined, or scrapped; schedule slips 8–16 weeks. | Enforce mandatory DVT gate sign-off before cutting production hard steel. |
| 2 | **Cavity-to-Cavity Imbalance** | Parts from Cavity 1 and 2 pass CMM, but Cavity 3 and 4 show flashing or short shots. | Thermal runner gradient or unequal gate pressures across multi-cavity injection molds. | High scrap rates, mold jamming, non-uniform shrinkage, and out-of-spec dimensions. | Perform multi-cavity pressure balancing; re-cut gate apertures; run independent cavity FAIs. |
| 3 | **Gauge R&R Rejection** | Test results vary wildly when the same golden board is re-tested on the same fixture. | Poor pogo-pin alignment, worn test probes, RF connector wear, or ground bounce in test jigs. | Good boards falsely rejected to bonepile; bad boards shipped to field. | Redesign test fixtures; enforce Gauge R&R $\%R\&R < 10\%$ before PVT authorization. |
| 4 | **Bonepile WIP Explosion** | Off-line debug area fills with stacks of failed PCBAs; repair queue exceeds 5 days of production. | SMT first-pass yield falls below forecast, while debug technician capacity is undersized. | Capital trapped in WIP; high risk of secondary damage during manual desoldering. | Implement SMT line throttling when bonepile hits cap; deploy dedicated debug engineers. |
| 5 | **Thermal Burn-In Saturation** | Units queue up in staging corridors waiting for open slots in environmental chambers. | Burn-in cycle time underestimated; early fallout requires extended re-soak cycles. | Line halts due to upstream WIP blockage; factory fails to meet weekly shipping rate. | Size burn-in chambers using Little's Law with $25\%$ headroom; optimize thermal soak profile. |
| 6 | **NCNR Silicon Stranding** | Millions of dollars in custom ASICs arrive at EMS warehouse, but cannot be assembled. | Companion passive component or mechanical chassis delayed due to tooling failure. | Cash conversion cycle collapses; high E&O inventory carrying costs. | Synchronize NCNR procurement horizons across the complete bill of materials. |
| 7 | **Solder Paste Aperture Bridging** | High solder bridge defect rate at AOI on 0.4mm pitch QFN/BGA packages. | Solder paste stencil thickness inappropriate for mixed fine-pitch and large power devices. | Low SMT first-pass yield; extensive manual touch-up causing thermal stress. | Utilize stepped stencils (electroformed / laser-cut) and optimize solder powder grain size (Type 4/5). |
| 8 | **Regulatory Type-Approval Failure** | Unit fails radiated emissions at 2.4 GHz or 5 GHz during formal FCC anechoic chamber testing. | Chassis seam RF leakage, inadequate grounding finger pressure, or clock harmonic radiation. | Product cannot be legally sold; mandatory mechanical redesign of chassis shielding. | Conduct near-field EMI pre-scans during EVT; design overlapping chassis labyrinth seams. |
| 9 | **Supplier De-commitment on Ramp** | Tier-2 component vendor reports $50\%$ allocation cut 4 weeks before mass production launch. | Supplier's own sub-tier raw material shortage or yields failing on new component revision. | Complete assembly line starvation; inability to ship finished systems. | Mandate secondary qualified sources on Approved Vendor List (AVL) before DVT exit. |
| 10 | **Solder Joint Fatigue in Field** | Units pass factory ICT and burn-in, but fail after 3 months of field thermal cycling. | Thermal expansion coefficient (CTE) mismatch between ceramic BGA substrate and FR4 PCB. | Catastrophic field recalls, warranty claims, and brand equity destruction. | Perform accelerated thermal cycling (JESD22-A104) and underfill critical large BGA components. |
| 11 | **SMT Micro-Attrition Starvation** | SMT line stops mid-shift because 0201 capacitor reel is empty despite theoretical BOM quantity matching starts. | High-speed pick-and-place nozzle pickup attrition (0.5%–1.5%) and tape indexing slippage on ultra-small passives. | Downstream PCBA line halted; partial assemblies sit un-reflowed on conveyance tracks. | Add mandatory 1.0%–2.0% feeder attrition over-ordering on all tape-and-reel passives. |
| 12 | **Latent ESD Degradation in Rework** | Field failures occur on PCBAs that were previously reworked off-line, while pristine first-pass units show zero defects. | Manual desoldering and hot-air rework conducted without verified ESD grounding or excessive thermal dwell time. | Gate oxide dielectric breakdown in CMOS silicon; early field failure after 500 operating hours. | Require continuous ground strap monitoring, temperature-controlled rework wands, and nitrogen shielding. |

---

## 11. Assumptions and limits

| # | Assumption | Operational Rationale | Failure Trigger / Boundary Condition | Verification & Sensitivity Test |
|---|---|---|---|---|
| 1 | **Stationary Defect Distribution** | Assumes defect mechanisms remain constant as volume scales, allowing learning curve projection. | Changes in raw material lots, solder paste chemical formulations, or operator shifts. | Monitor weekly Pareto charts of defect types; flag emergence of new failure modes. |
| 2 | **Normal Distribution of Dimensions** | Assumes physical dimensions follow Gaussian distribution for $C_p, C_{pk}$ calculation. | Tool wear, thermal drifting of molding machines, or multi-cavity multimodal distributions. | Test normality using Anderson-Darling or Shapiro-Wilk tests before computing $C_{pk}$. |
| 3 | **Unlimited Rework Capacity** | Planning models frequently assume all failed units can be reworked within the current shift. | Complex multi-layer BGA defects require specialized desoldering stations and expert debug. | Model bonepile as an $M/M/c$ queuing system with finite server capacity $\mu_{\text{rwk}}$. |
| 4 | **Fixed Tooling Life Cycle** | Assumes injection mold will run 500,000 to 1,000,000 shots before requiring refurbishment. | Highly abrasive glass-filled resins (e.g., $30\%$ glass-filled PBT) rapidly erode gate steel. | Track cumulative shot counter; perform scheduled preventative maintenance every 50k shots. |
| 5 | **Zero Post-PVT Design Changes** | Assumes BOM is 100% frozen at PVT exit and no further engineering modifications occur. | Critical field bugs, regulatory compliance failures, or component obsolescence notices. | Mandate ECO Class I freeze; require VP Operations sign-off for any post-PVT BOM change. |
| 6 | **Constant Component Purchase Cost** | Assumes supplier contractual pricing remains fixed during volume ramp up. | Purchase Price Variance (PPV), raw material surcharges (copper, resin, gold), or expedited air freight. | Track purchase price variance monthly; enforce price indexing clauses in supplier contracts. |
| 7 | **Independent Station Yields** | Assumes first-pass yield at Station $k$ is statistically independent of yield at Station $k-1$. | Marginal solder joints that pass AOI degrade under thermal stress and fail at ICT or Burn-in. | Compute conditional failure probabilities: $P(\text{Fail}_{ICT} \mid \text{Pass}_{AOI})$. |
| 8 | **Deterministic Supplier Lead Times** | Assumes components arrive exactly $L_{\text{lead}}$ weeks after purchase order transmission. | Port congestions, customs clearance delays, fab equipment breakdowns, or wafer yield crashes. | Model lead time as a stochastic variable $L \sim \text{Gamma}(\alpha, \beta)$; buffer with strategic safety stock. |
| 9 | **Zero Tooling Steel Fatigue Prior to Rated Life** | Assumes mold dimensional tolerances remain invariant throughout the NPI ramp volume. | Extreme injection pressures or abrasive glass-fiber filler eroding gate and runner geometry. | Track dimensional drift using weekly CMM control charts on critical part features. |
| 10 | **Rework Solder Joint Reliability Parity** | Assumes hand-soldered or hot-air reworked BGA/QFN components have equivalent fatigue life to reflowed components. | Multiple thermal excursions degrading intermetallic compound (IMC) thickness at pad interfaces. | Cross-section and shear-test sample reworked joints under thermal shock cycling. |

---

## 12. Exercises

### Exercise 1: Recall and Phase Gate Governance
State the primary objective, typical build quantity, and mandatory exit criteria for EVT, DVT, and PVT in enterprise hardware manufacturing. Explain why conducting PVT on prototype soft tooling violates quality standards.
- **Deliverable:** Structured comparison table and 1-page governance analysis.
- **Acceptance Criteria:** Correctly defines build quantities, tooling maturity, and yield targets for all three phases.
- **Estimated Time:** 30 minutes.

### Exercise 2: Semiconductor Defect Density and Yield Calculation
A newly designed 5G baseband processing ASIC has an active die area of $A = 2.40 \text{ cm}^2$.
1. During initial EVT wafer runs, fab defect density is measured at $D_0 = 0.55 \text{ defects/cm}^2$. Calculate expected wafer sort yield using:
   - The elementary Poisson model.
   - Stapper's Negative Binomial model with clustering parameter $\alpha = 1.20$.
2. Explain the physical reason for the yield disparity between the two models.
- **Deliverable:** Mathematical derivations, calculated percentage yields to two decimal places, and technical explanation.
- **Acceptance Criteria:** Correct application of exponential and negative binomial formulas; precise numerical results.
- **Estimated Time:** 45 minutes.

### Exercise 3: Multi-Stage Roll-Through Yield and Material Starts
An avionics flight computer routing consists of four sequential manufacturing stations:
- SMT Placement & Reflow: $FPY_1 = 92.0\%$; Rework yield $Y_{\text{rwk},1} = 80.0\%$ (all defects reworkable).
- Automated X-ray Inspection (AXI): $FPY_2 = 96.0\%$; No rework (structural solder voids; un-reworkable).
- In-Circuit Test (ICT): $FPY_3 = 94.0\%$; Rework yield $Y_{\text{rwk},3} = 75.0\%$ (all defects reworkable).
- Final Functional Test (FCT): $FPY_4 = 98.0\%$; No rework (scrap).
1. Calculate the raw Roll-Through Yield (RTY) without rework.
2. Calculate the effective line yield ($Y_{\text{eff}}$) with rework loops.
3. If customer contract demands exactly 5,000 finished flight computers, calculate required input board starts ($N_{\text{start}}$) and total scrapped boards.
- **Deliverable:** Step-by-step arithmetic formulas and final integer production quantities.
- **Acceptance Criteria:** Exact yield calculations; correct rounding up of discrete board starts.
- **Estimated Time:** 45 minutes.

### Exercise 4: Tooling Cavitation and Capex Optimization
A hardware OEM must manufacture an injection-molded enclosure for a smart IoT gateway. Annual product demand is 120,000 units over a 3-year product life cycle (total demand = 360,000 units). Two tooling options are available:
- **Option 1 (2-Cavity Mold):** Tooling Capex = $\$90,000$; Cycle time = 45 seconds; Press hourly rate = $\$85.00/\text{hour}$; Mold maintenance = $\$0.05/\text{unit}$.
- **Option 2 (4-Cavity Mold):** Tooling Capex = $\$165,000$; Cycle time = 62 seconds; Press hourly rate = $\$115.00/\text{hour}$; Mold maintenance = $\$0.08/\text{unit}$.
Material resin cost is $\$1.25$ per part for both options. Direct labor handling is $\$0.15$ per part.
1. Compute the total manufactured cost per unit (including amortized tooling Capex) for both options across the 360,000-unit program.
2. Determine the exact volume breakeven point where the 4-cavity mold becomes more economical than the 2-cavity mold.
- **Deliverable:** Cost equations, per-unit cost breakdown, and breakeven volume calculation.
- **Acceptance Criteria:** Accurate machine hourly amortization; correct algebraic breakeven solution.
- **Estimated Time:** 60 minutes.

### Exercise 5: NCNR Sourcing Optimization under Stochastic Demand
An OEM is procuring a proprietary optical transceiver module with an NCNR commitment for an upcoming telecom router launch. Lead time is 32 weeks.
- Contract purchase cost: $c = \$450.00$ per module.
- Finished router selling price: $p = \$3,500.00$.
- Underage penalty (lost margin plus SLA breach fine): $v = \$1,200.00$.
- Salvage value of excess transceiver: $s = \$50.00$.
- Demand is normally distributed with mean $\mu_D = 12,000$ units and standard deviation $\sigma_D = 3,200$ units.
- Assembly yield is expected to be $\bar{Y} = 94.0\%$ with standard deviation $\sigma_Y = 3.0\%$.
1. Calculate the critical fractile $CF$.
2. Compute the optimal NCNR order quantity $Q^*$ accounting for both demand and yield uncertainty.
3. Calculate the total financial liability exposure committed at Day 0.
- **Deliverable:** Complete Newsvendor calculation with step-by-step statistical formulas.
- **Acceptance Criteria:** Correct critical fractile; appropriate yield-adjusted standard deviation and order quantity.
- **Estimated Time:** 45 minutes.

### Exercise 6: Systems Design — Mitigating an NPI Ramp Crisis
You are appointed Director of Operations for an aerospace telecommunications manufacturer. Two weeks into a critical satellite receiver ramp, the following data is reported:
- Target daily output: 200 units/day.
- SMT placement starts: 280 units/day.
- Actual finished units packed: 110 units/day.
- Functional test first-pass yield: $62.0\%$ (target was $90.0\%$).
- Thermal burn-in chamber queue: 450 units waiting in hallway.
- Bonepile: 620 units accumulated; 2 debug technicians repairing 18 units/day each.
Develop a comprehensive 5-point emergency operational recovery plan to stabilize the factory within 14 days without violating quality or airworthiness standards.
- **Deliverable:** Executive operational intervention memorandum detailing tactical actions, resource reallocations, line balancing changes, and governance controls.
- **Acceptance Criteria:** Plan directly addresses the bottleneck, bonepile accumulation, and burn-in capacity constraints using quantitative arguments.
- **Estimated Time:** 60 minutes.

---

## 13. Self-check answers

### Answer to Exercise 2: Semiconductor Defect Density and Yield
**1. Poisson Model Calculation:**
$$Y_{\text{Poisson}} = e^{-D_0 \cdot A} = e^{-(0.55 \times 2.40)} = e^{-1.320} \approx 0.2671 \quad (26.71\%)$$

**2. Stapper Negative Binomial Model Calculation ($\alpha = 1.20$):**
$$Y_{\text{Stapper}} = \left(1 + \frac{D_0 \cdot A}{\alpha}\right)^{-\alpha} = \left(1 + \frac{0.55 \times 2.40}{1.20}\right)^{-1.20} = \left(1 + \frac{1.320}{1.20}\right)^{-1.20} = (1 + 1.10)^{-1.20} = (2.10)^{-1.20}$$
$$Y_{\text{Stapper}} = e^{-1.20 \cdot \ln(2.10)} = e^{-1.20 \cdot 0.74194} = e^{-0.89033} \approx 0.4105 \quad (41.05\%)$$

**3. Physical Explanation:**
The Stapper model yields $41.05\%$, which is $14.34\%$ higher than the Poisson model ($26.71\%$). The Poisson model assumes defects are scattered independently and uniformly across the entire wafer surface. In physical semiconductor fabrication, defects cluster in localized groups (due to particulate contamination or thermal gradients during photolithography/etching). When defects cluster, many defects fall on the exact same die, rendering that die defective only once. Consequently, large swaths of the remaining wafer surface remain entirely defect-free, producing a substantially higher net die yield.

---

### Answer to Exercise 3: Multi-Stage Roll-Through Yield and Starts
**1. Raw Roll-Through Yield (without rework):**
$$RTY = FPY_1 \cdot FPY_2 \cdot FPY_3 \cdot FPY_4 = 0.920 \times 0.960 \times 0.940 \times 0.980 = 0.8143 \quad (81.43\%)$$

**2. Effective Station Yields and Line Yield:**
- Station 1: $Y_{\text{eff},1} = FPY_1 + (1 - FPY_1) \cdot Y_{\text{rwk},1} = 0.920 + (0.080 \times 0.800) = 0.920 + 0.064 = 0.9840$
- Station 2: $Y_{\text{eff},2} = FPY_2 = 0.9600$ (no rework)
- Station 3: $Y_{\text{eff},3} = FPY_3 + (1 - FPY_3) \cdot Y_{\text{rwk},3} = 0.940 + (0.060 \times 0.750) = 0.940 + 0.045 = 0.9850$
- Station 4: $Y_{\text{eff},4} = FPY_4 = 0.9800$ (no rework)

$$Y_{\text{eff},\text{line}} = Y_{\text{eff},1} \cdot Y_{\text{eff},2} \cdot Y_{\text{eff},3} \cdot Y_{\text{eff},4} = 0.9840 \times 0.9600 \times 0.9850 \times 0.9800 = 0.9118 \quad (91.18\%)$$

**3. Required Production Starts:**
$$N_{\text{start}} = \left\lceil \frac{N_{\text{good}}}{Y_{\text{eff},\text{line}}} \right\rceil = \left\lceil \frac{5,000}{0.91182} \right\rceil = \lceil 5,483.54 \rceil = 5,484 \text{ starts}$$
$$\text{Total Scrapped Units} = N_{\text{start}} - N_{\text{good}} = 5,484 - 5,000 = 484 \text{ scrapped units}$$

---

### Answer to Exercise 4: Tooling Cavitation and Breakeven
**1. Per-Unit Running Cost Calculation:**
- **Option 1 (2-Cavity):**
  $$\text{Parts per hour} = \left(\frac{3600}{45}\right) \times 2 = 160 \text{ parts/hour}$$
  $$\text{Machine Run Cost} = \frac{\$85.00}{160} = \$0.53125 \text{ per part}$$
  $$\text{Variable Cost per Part } (v_1) = \text{Material} + \text{Machine} + \text{Labor} + \text{Maint} = 1.25 + 0.53125 + 0.15 + 0.05 = \$1.98125$$
  $$\text{Total Program Cost (360k units)} = \$90,000 + (360,000 \times 1.98125) = \$90,000 + \$713,250 = \$803,250$$
  $$\text{Total Landed Cost per Unit} = \frac{\$803,250}{360,000} = \$2.231 \text{ per unit}$$

- **Option 2 (4-Cavity):**
  $$\text{Parts per hour} = \left(\frac{3600}{62}\right) \times 4 = 232.258 \text{ parts/hour}$$
  $$\text{Machine Run Cost} = \frac{\$115.00}{232.258} = \$0.49514 \text{ per part}$$
  $$\text{Variable Cost per Part } (v_2) = 1.25 + 0.49514 + 0.15 + 0.08 = \$1.97514$$
  $$\text{Total Program Cost (360k units)} = \$165,000 + (360,000 \times 1.97514) = \$165,000 + \$711,050 = \$876,050$$
  $$\text{Total Landed Cost per Unit} = \frac{\$876,050}{360,000} = \$2.433 \text{ per unit}$$

**2. Breakeven Volume Analysis:**
$$Q_{\text{break}} = \frac{\text{Capex}_2 - \text{Capex}_1}{v_1 - v_2} = \frac{\$165,000 - \$90,000}{\$1.98125 - \$1.97514} = \frac{\$75,000}{\$0.00611} \approx 12,274,959 \text{ units}$$
**Conclusion:** The 4-cavity mold saves only $\$0.0061$ per unit in variable cost because higher machine hourly rates and mold maintenance offset cycle time gains. It requires over 12 million units to break even. For a 360,000-unit program, Option 1 (2-Cavity) saves $\$72,800$ in total program costs and is the superior economic decision.

---

### Answer to Exercise 5: NCNR Sourcing Optimization
**1. Critical Fractile:**
$$c_u = (p - c) + v = (\$3,500 - \$450) + \$1,200 = \$4,250$$
$$c_o = c - s = \$450 - \$50 = \$400$$
$$CF = \frac{c_u}{c_u + c_o} = \frac{4,250}{4,250 + 400} = \frac{4,250}{4,650} = 0.91398 \quad (91.40\%)$$

**2. Optimal Order Quantity:**
$$z_{0.9140} = \Phi^{-1}(0.9140) \approx 1.366$$
Adjusted standard deviation incorporating yield variance:
$$\sigma_{\text{eff}} = \sqrt{\sigma_D^2 + \left(\frac{\mu_D \cdot \sigma_Y}{\bar{Y}}\right)^2} = \sqrt{3,200^2 + \left(\frac{12,000 \times 0.03}{0.94}\right)^2} = \sqrt{10,240,000 + 146,672} = \sqrt{10,386,672} \approx 3,222.8$$
$$Q^* = \frac{\mu_D + z \cdot \sigma_{\text{eff}}}{\bar{Y}} = \frac{12,000 + (1.366 \times 3,222.8)}{0.94} = \frac{12,000 + 4,402.3}{0.94} = \frac{16,402.3}{0.94} \approx 17,450 \text{ units}$$

**3. Total Day-0 Financial Commitment:**
$$\text{Liability} = 17,450 \times \$450.00 = \$7,852,500.00$$

---

### Evaluation Rubric for Exercise 6: Systems Design
A passing plan must articulate five concrete operational interventions:
1. **Immediate SMT Input Throttling:** Throttle line starts from 280 to 180 units/day. Input starts currently exceed packout capacity (110 units/day) by 170 units/day, which is actively fueling the bonepile and burn-in bottleneck.
2. **Debug Team Capacity Surge:** Existing rework capacity is only $2 \times 18 = 36$ units/day, while fallout arrives at $280 \times 0.38 = 106$ units/day. Authorize temporary assignment of 4 experienced test engineering specialists to increase debug capacity to $>120$ units/day to drain the 620-unit bonepile.
3. **Burn-In Dynamic Management:** Chamber queue has 450 units waiting. Audit burn-in failure data: if infant mortality failures precipitate within the first 24 hours, request temporary engineering waiver to reduce soak duration from 48 hours to 36 hours for non-flight validation units, doubling chamber throughput.
4. **Root-Cause Defect Containment:** Conduct immediate Pareto analysis of the $38\%$ functional test fallout. If defects concentrate in specific BGA solder bridging or clock oscillator drift, execute immediate SMT stencil adjustment or component lot quarantine.
5. **Phase-Gate Governance Re-imposition:** Hold PVT exit sign-off. Daily cross-functional standups between OEM Quality, Operations, and EMS Plant GM to review bonepile burn-down rate and $C_{pk}$ trends.

---

## 14. Interview questions

### Question 1: Hardware Phase Gates and Readiness
*Question:* "How do you quantitatively distinguish between DVT and PVT, and under what conditions would you reject a program trying to enter PVT?"
- **Strong Answer:** DVT validates that the product design meets functional, environmental, and regulatory specifications using production-intent parts, focusing on design margin and reliability (HALT/HASS). PVT validates that the manufacturing factory and tooling can produce the product at target takt time, target yield, and standard labor cost. I would reject PVT entry if: (1) Class I ECOs remain open, (2) hard tooling has not achieved T1 FAI approval, (3) test fixture Gauge R&R exceeds 10%, (4) DVT first-pass yield is below $75\%$, or (5) critical dimensions fail to demonstrate $C_{pk} \ge 1.33$.
- **Weak Answer reveals:** Conflating product design bugs with factory process defects; believing PVT is just a "bigger DVT build" without understanding cycle time, takt time, or statistical process capability.

### Question 2: Defect Density Models in Hardware
*Question:* "Why does the Poisson yield model underestimate actual semiconductor and PCBA manufacturing yield, and what formulation should you use instead?"
- **Strong Answer:** The Poisson model ($Y = e^{-D A}$) assumes fatal defects are distributed uniformly and independently across the surface. In reality, defects cluster spatially due to localized process anomalies (e.g., thermal gradients, chemical vapor distribution, or particle contamination). When defects cluster, multiple defects hit the same die, leaving more total dies defect-free. To model this accurately, we use Stapper's Negative Binomial model: $Y = (1 + D A / \alpha)^{-\alpha}$, where $\alpha$ is the defect clustering parameter. As $\alpha \to \infty$, it converges to Poisson, but for physical fabs with clustering ($\alpha \approx 1$), Stapper provides an accurate, higher yield projection.
- **Weak Answer reveals:** Rote memorization of $e^{-D A}$ without understanding spatial defect physics, clustering phenomena, or the mathematical derivation of compound distributions.

### Question 3: Tooling Sampling Progression (T0, T1, T2)
*Question:* "Explain the concept of 'steel-safe condition' in injection mold tooling, and what specific metrology milestone separates T1 from T2 sampling?"
- **Strong Answer:** Steel-safe design leaves extra metal on mold cores and cavities during initial CNC machining. This ensures that when plastic parts are molded, external walls are undersized and internal holes are oversized. Correcting dimensional errors after initial sampling requires removing mold steel via CNC/EDM—which is fast and structurally sound—rather than welding steel back on, which degrades mold life and finish. T1 sampling is focused on dimensional compliance via 100% First Article Inspection (FAI) on a Coordinate Measuring Machine (CMM) across all cavities. T2 sampling runs under mass production conditions at full cycle speed to evaluate multi-cavity thermal balancing, mold wear, and statistical process capability ($C_p, C_{pk} \ge 1.33$).
- **Weak Answer reveals:** Thinking T0/T1/T2 are arbitrary marketing designations; unaware of mold fabrication metallurgy, shrinkage behavior, or CMM metrology.

### Question 4: NCNR Purchase Commitments under Shortages
*Question:* "If your custom ASIC supplier requires a 36-week NCNR commitment and your customer demand forecast has a standard deviation of 30%, how do you determine the order quantity?"
- **Strong Answer:** I formulate this as an asymmetric Newsvendor optimization problem under joint demand and yield uncertainty. I compute the underage cost $c_u$ (lost gross margin plus contractual SLA breach penalties) and overage cost $c_o$ (purchase cost minus scrap salvage). Because high-tech hardware has high margins and severe stockout penalties, the critical fractile $CF = c_u / (c_u + c_o)$ is typically high (often $85\% - 95\%$). I then inflate the order quantity by the expected yield loss ($1 / \bar{Y}$) and add a safety buffer proportional to the compound variance $\sigma_{\text{eff}} = \sqrt{\sigma_D^2 + (\mu_D \sigma_Y / \bar{Y})^2}$. Finally, I negotiate milestone release schedules (e.g., wafer fab release vs. final packaging release) to truncate the supplier's liability window.
- **Weak Answer reveals:** Ordering strictly to mean demand forecast; ignoring asymmetric stockout penalties; failing to account for assembly yield fallout in raw material procurement.

### Question 5: Managing an NPI Bonepile Crisis
*Question:* "During ramp, your SMT line yield drops to 70%, and a bonepile of 1,000 un-debugged boards accumulates. The factory manager wants to increase SMT starts to hit the weekly shipment quota. Do you approve?"
- **Strong Answer:** Absolutely not. Increasing SMT starts when yield has crashed accelerates bonepile growth and traps more working capital in un-shippable WIP. If the bonepile exceeds rework capacity, defective units sit for weeks, risking component obsolescence, component oxidation, and missed supplier warranty windows. My immediate actions would be: (1) Throttle or temporarily halt SMT line starts, (2) perform an urgent Pareto analysis on bonepile defect codes to find the root cause (e.g., solder paste stencil bridging or component lot defect), (3) reallocate test engineers to drain the rework queue, and (4) only resume production once the root cause is contained and first-pass yield recovers.
- **Weak Answer reveals:** Pushing volume into a broken process ("sweeping defects under the rug"); measuring factory performance purely by line starts rather than net good packout.

### Question 6: Line Balancing and Burn-In Sizing
*Question:* "How do you size a 48-hour burn-in process when upstream SMT cycle time is 45 seconds per board?"
- **Strong Answer:** SMT cycle time and burn-in sizing are decoupled by Little's Law: $WIP = \lambda \cdot T_{\text{burn}}$. First, determine required daily packout demand $D$, not instantaneous SMT speed. If daily demand is 1,000 units/day over 24 hours, the arrival rate is $\lambda = 41.67$ units/hour. For a 48-hour burn-in soak, required active WIP is $41.67 \times 48 = 2,000$ units. I then adjust for chamber availability (uptime $A \approx 90\%$) and pre-burn yield fallout ($FPY_{\text{pre}}$) to calculate required physical slot capacity: $S^* = 2,000 / (0.90 \times FPY_{\text{pre}})$. If chambers hold 200 units, I need $\lceil S^* / 200 \rceil$ chambers.
- **Weak Answer reveals:** Attempting to divide 48 hours by 45 seconds; confusing discrete station cycle time with continuous batch queuing and thermal soak capacity.

### Question 7: ECO Effectivity during Ramp
*Question:* "A critical design bug is discovered during PVT. How do you decide whether to execute an immediate scrap ECO or a phase-in run-out ECO?"
- **Strong Answer:** The decision hinges on whether the bug is safety/regulatory-critical (Class I) or functional/cost-critical (Class II). If the bug causes regulatory non-compliance (FCC emission breach) or safety/reliability hazards (solder fatigue in the field), an **immediate scrap ECO** is legally mandatory: all existing WIP and finished goods must be quarantined and scrapped or reworked regardless of cost. If it is a Class II performance or yield improvement, I conduct an economic run-out optimization: compare the cost of scrapping existing pipeline inventory and NCNR purchase commitments against the warranty risk or field patch cost of letting existing WIP run out.
- **Weak Answer reveals:** Assuming all ECOs can be phased in slowly to save money, or conversely, scrapping millions in inventory for minor cosmetic defects without evaluating run-out economics.

### Question 8: Test Fixture Repeatability (Gauge R&R)
*Question:* "Your contract manufacturer reports an 8% fallout rate at functional test. How do you determine whether the product is defective or the test fixture is failing?"
- **Strong Answer:** I conduct an immediate Gauge Repeatability and Reproducibility (Gauge R&R) study using ANOVA or X-bar/R charts. I take a set of 10 known, characterized boards (including borderline passing and failing units) and run them through the test fixture across 3 different operators, repeating the test 3 times each. If the measurement system variation (%R&R) exceeds $30\%$ of total tolerance, the test fixture is un-reproducible: failures are caused by pogo-pin resistance, RF cable calibration drift, or software threshold errors. If $\%R\&R < 10\%$, the measurement system is sound, and the fallout represents genuine product assembly defects.
- **Weak Answer reveals:** Immediately changing product design or blaming component suppliers without first verifying test fixture measurement system capability.

---

## 15. Key terms

- **Architecture Freeze:** Formal milestone locking system architecture, bus topologies, and core chipsets prior to detailed layout.
- **AS9102:** Standardized aerospace First Article Inspection documentation protocol verifying dimensional compliance.
- **Bonepile:** Physical inventory of defective assemblies waiting off-line for debug, component desoldering, or engineering disposition.
- **Burn-In:** High-temperature electrical stress screening designed to precipitate latent semiconductor infant mortality defects.
- **Cavitation:** The number of identical molding cavities machined into a single injection mold or die-casting tool.
- **Coordinate Measuring Machine (CMM):** High-precision metrology device utilizing physical probes or lasers to verify 3D part dimensions.
- **Crawford Learning Curve:** Unit-based learning curve formulation modeling the marginal labor or cost of the $N$-th produced unit.
- **Defect Clustering:** Physical tendency of microelectronic manufacturing defects to concentrate in localized spatial zones.
- **Design for Manufacturability (DFM):** Engineering practice of designing parts to optimize ease, speed, and cost of volume manufacturing.
- **DVT (Design Validation Test):** Phase gate verifying that product design meets all environmental, reliability, and regulatory requirements.
- **EVT (Engineering Validation Test):** Initial build phase validating basic functional and electrical performance of core schematics.
- **First Article Inspection (FAI):** Metrological verification that 100% of engineering drawing dimensions match off-tool parts.
- **First-Pass Yield (FPY):** Percentage of manufactured units passing a specific routing operation without rework or scrap.
- **Gauge R&R:** Statistical evaluation of the repeatability and reproducibility of a measurement system or test fixture.
- **HALT / HASS:** Highly Accelerated Life Testing (destructive limit testing) and Highly Accelerated Stress Screening (production screening).
- **In-Circuit Test (ICT):** Automated bed-of-nails fixture verifying passive component values, opens, shorts, and solder integrity.
- **Liability Window:** Forward calendar horizon during which an OEM is contractually liable for supplier material and WIP costs.
- **Line Balancing Efficiency:** Ratio of total assembly work content to the product of total stations and bottleneck cycle time.
- **Little's Law:** Queuing theory theorem ($L = \lambda W$) relating work-in-progress, arrival rate, and station soak/cycle time.
- **Murphy's Yield Model:** Microelectronic yield formulation assuming a triangular probability distribution of fatal defect density.
- **NCNR Commitment:** Non-Cancelable, Non-Returnable commercial purchase order legally binding the buyer to 100% of material cost.
- **Poisson Yield Model:** Elementary semiconductor yield formula assuming uniform, non-clustered, random defect distributions.
- **Process Capability ($C_{pk}$):** Index measuring how well a manufacturing process holds output within upper and lower specification limits.
- **PVT (Production Validation Test):** Pre-production build verifying factory line rate, production tooling, and mature cycle times.
- **Roll-Through Yield (RTY):** Mathematical product of first-pass yields across all sequential manufacturing routing stations.
- **Seeds' Yield Model:** Empirical microelectronic yield model accounting for high-density defect clustering ($Y = e^{-\sqrt{D A}}$).
- **Stapper's Model:** Negative Binomial yield model parameterizing defect clustering via parameter $\alpha$.
- **Steel-Safe Condition:** Toolmaking design practice leaving extra mold metal so subsequent dimensional tuning requires cutting steel.
- **T0 / T1 / T2 Sampling:** Sequential tooling qualification milestones: gross fill (T0), dimensional CMM (T1), and capability run (T2).
- **Takt Time:** Target manufacturing pace required to match customer demand ($T_{\text{net}} / D_{\text{target}}$).
- **Wright's Law:** Empirical learning curve formulation stating cumulative average unit cost falls by a constant percentage per volume doubling.

---

## 16. Sources

- AIAG. (2006). *Production Part Approval Process (PPAP)* (4th ed.). Automotive Industry Action Group.
- Crawford, J. R. (1944). *Learning Curve, Proration of Work Hours to Units*. Lockheed Aircraft Corporation.
- Cunningham, J. A. (1990). The Use and Evaluation of Yield Models in Integrated Circuit Manufacturing. *IEEE Transactions on Semiconductor Manufacturing*, 3(2), 60–71. https://doi.org/10.1109/66.53188
- IPC. (2020). *IPC-A-610H: Acceptability of Electronic Assemblies*. Association Connecting Electronics Industries.
- JEDEC. (2021). *JESD22-A104F: Temperature Cycling*. JEDEC Solid State Technology Association.
- Murphy, B. T. (1964). Cost of Large Integrated Chips: Analysis of Yield and Number of Good Chips per Wafer. *Electronics*, 37(24), 105–118.
- SAE International. (2014). *AS9102 Rev B: Aerospace First Article Inspection Requirement*. SAE Standards.
- Seeds, R. B. (1967). Yield and Cost Analysis of Bipolar LSI. *IEEE International Electron Devices Meeting (IEDM) Technical Digest*, 13, 12–15.
- Stapper, C. H. (1973). Defect Density for Small-Scale Integrated Circuits. *IEEE Transactions on Electron Devices*, ED-20(7), 655–657. https://doi.org/10.1109/T-ED.1973.17719
- Stapper, C. H. (1989). Large-Area Fault Clusters and Fault Tolerance in VLSI Circuits: A Review. *IBM Journal of Research and Development*, 33(2), 162–173. https://doi.org/10.1147/rd.332.0162
- Telcordia Technologies. (2017). *GR-63-CORE, Issue 5: NEBS Requirements: Physical Protection*. Ericsson Telecommunications Standards.
- Wright, T. P. (1936). Factors Affecting the Cost of Airplanes. *Journal of the Aeronautical Sciences*, 3(4), 122–128. https://doi.org/10.2514/8.155

---

## 17. Links

- [Module 40 Index (`_index.md`)](_index.md)
- [Module 40 Root Survey (`../../../course/40_telecom_technology_hardware.md`)](../../../course/40_telecom_technology_hardware.md)
- Sibling Topics:
  - [Topic 01 — Configuration, BOM, and Life Cycle](01_configuration-bom-lifecycle.md)
  - [Topic 02 — Component Allocation under Shortage](02_component-allocation.md)
  - [Topic 03 — EMS and Contract Manufacturing](03_ems-contract-manufacturing.md)
  - [Topic 05 — Telecom Network Rollout](05_telecom-network-rollout.md)
  - [Topic 06 — Device Supply and Channel Sell-Through](06_device-channel-sellthrough.md)
  - [Topic 09 — Obsolescence, LTB, and Product Teardown](09_obsolescence-last-time-buy.md)
- Prerequisite Modules:
  - [Module 09 — Production Planning and Master Scheduling](../../02_methods/09_production_planning_scheduling/_index.md)
  - [Module 10 — Inventory Theory and Stochastic Control](../../02_methods/10_inventory_theory_stochastic_control/_index.md)
  - [Module 14 — Supplier Relationship Management and Sourcing](../../03_commerce/14_supplier_relationship_management_sourcing/_index.md)
