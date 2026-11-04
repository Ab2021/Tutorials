# Forestry Harvest and Chain of Custody: Rotation Optimization, Bucking Allocation, Seasonal Haulage, and Geospatial Traceability

> Module 39 — Mining, Metals, and Forestry Supply Chains
> Topic 09 of 10
> Estimated study time: 4 hours
> Prerequisites: Module 11 (Network Design Optimization), Module 12 (Inventory Theory), Module 15 (Production Planning and Scheduling), Module 38 (Process Industries, Commodities, and Bulk Supply Chains)
> Version 2026.5

---

## 1. Why this topic exists

In discrete assembly manufacturing, the supply chain begins with fabricated parts produced to rigid tolerances, certified by inspection sheets, and scheduled on deterministic lead times. In mineral mining, feedstocks are extracted from an unyielding geological rock mass where quality is defined by mineral chemistry and grade distribution. 

Forestry supply chains operate under an entirely distinct biological and environmental paradigm:
1. **Living, compound capital on multi-decade horizons:** Standing timber is simultaneously the production factory and the finished inventory. A commercial forest stand of Scots pine (*Pinus sylvestris*), Douglas-fir (*Pseudotsuga menziesii*), or loblolly pine (*Pinus taeda*) accumulates merchantable volume over 25 to 80+ years according to non-linear biological sigmoid yield curves. Deciding when to harvest is an irreversible financial choice: harvesting too early truncates high-margin biological volume growth and diameter premiums (large sawlogs vs cheap pulpwood); harvesting too late locks up scarce land capital (incurring land rent opportunity costs), exposes the stand to catastrophic mortality from windthrow, wildfire, and insect outbreaks (e.g., bark beetle *Ips typographus*), and suppresses discounted return on invested capital.
2. **Value destruction at the stump (The Bucking Dilemma):** A felled tree stem is a continuous, irregular tapered solid characterized by diminishing diameter, sweep, knot whorls, and localized rot. The moment a mechanized cut-to-length (CTL) harvester grips a stem, it must cut that stem into discrete log segments (sawlogs, veneer, small sawlogs, pulpwood, biomass). Because price per cubic meter varies by up to $400\%$ between product classes, a single suboptimal bucking decision that misallocates a sawlog segment into pulpwood or biomass destroys $60\text{--}80\%$ of the realizable stumpage value for that log. This decision is made in milliseconds by automated onboard dynamic programming engines processing sensor streams under severe operational noise.
3. **Severe seasonal transport bottlenecks (Spring Breakup):** In Northern boreal and temperate timber regions (Canada, Fennoscandia, Northern US, Baltics), forest haulage relies on unpaved gravel spur roads and temporary winter roads built over frozen muskeg. During the spring thaw ("spring breakup"), subgrade soils saturate as the surface melts while deeper permafrost remains impermeable. The resilient modulus of the roadbed collapses by $50\text{--}70\%$, triggering statutory Gross Vehicle Weight (GVW) bans or 50% axle-load restrictions for 4 to 8 weeks. Mills must build enormous log inventories before breakup to avoid plant shutdowns.
4. **Perishability and biological decay in storage:** Storing roundwood in mill yards during spring and summer exposes logs to rapid moisture loss, end-checking, and colonization by sapstain fungi (*Ophiostoma* and *Ceratocystis* species). Within weeks of warm weather, blue stain hyphae penetrate the sapwood, downgrading premium architectural lumber to low-grade utility stock or packaging material, destroying tens of millions of dollars in inventory value unless active wet deck sprinkling regimes are deployed.
5. **Stringent geospatial and deforestation compliance:** Global timber markets can no longer accept non-traceable fiber. The European Union Deforestation Regulation (EUDR, Regulation (EU) 2023/1115), alongside FSC and PEFC certification frameworks, mandates verifiable, plot-level geolocation polygons down to six decimal places for every shipment of wood, proving zero deforestation after December 31, 2020. Non-compliance results in commercial forfeiture, confiscation of goods, and exclusion from the European single market.

Optimizing a forestry supply chain requires mastering bio-economic growth modeling, stochastic dynamic programming at the harvesting head, seasonal transportation physics, wood pathology kinetics, and end-to-end geospatial digital identity custody.

---

## 2. Learning outcomes

By the end of this topic, you will be able to:

1. **Derive and solve** the Faustmann Land Expectation Value (LEV) equation to determine the optimal financial rotation age $T^*$ of a forest stand, proving mathematically why $T^*$ is strictly shorter than both the biological Maximum Sustainable Yield (MSY) age and the single-rotation Fisher rotation age.
2. **Formulate** the log bucking problem as a discrete Dynamic Program (DP) over empirical stem taper profiles (Kozak taper equations), maximizing total net stem revenue across sawlog, studwood, and pulpwood assortments subject to small-end diameter, sweep, and mill order constraints.
3. **Quantify** the timing and structural severity of spring road breakup using Freezing Degree-Days ($FDD$) and Thawing Degree-Days ($TDD$) to size seasonal log yard buffer stocks and schedule truck fleets under statutory axle-load restrictions.
4. **Model** fungal sapstain (*Ophiostoma piceae*) radial penetration kinetics in stored softwood logs as a function of temperature and moisture content, evaluating the net present value of wet-deck sprinkler deployment against inventory downgrade losses.
5. **Formulate** a multi-period mixed-integer linear program (MILP) for tactical harvest block scheduling that enforces green-up environmental adjacency constraints, road building capex, and mill fiber blend balance.
6. **Design and audit** an end-to-end Chain of Custody (CoC) architecture under FSC-STD-40-004, PEFC ST 2002, and EUDR (Regulation (EU) 2023/1115), specifying the data schema for WGS84 parcel polygons, mass-balance accounting, and EU TRACES Due Diligence Statements.

---

## 3. The core mental model

A forest products supply chain is a **multi-scale spatio-temporal transformation pipeline** that converts distributed, living biological assets into discrete industrial feedstocks under strict seasonal, biological, and geospatial constraints.

```text
========================================================================================
STRATEGIC SCALE (20 to 80+ Years) — Stand Silviculture & Rotation Age
----------------------------------------------------------------------------------------
  Forest Stand Biological Growth: V(t) = A * (1 - exp(-k*t))^p
        |
        v
  Faustmann Land Expectation Value (LEV) Optimization
  Marginal biological value growth = Timber capital interest + Site land rent
  --> Optimal Financial Harvest Age T* (determines Long-Run Sustained Yield)
========================================================================================
                                    |
                                    v
TACTICAL SCALE (1 to 5 Years) — Harvest Block Allocation & Road Networks
----------------------------------------------------------------------------------------
  Spatial Harvest Block Selection (MILP)
  Subject to: Spatial adjacency (green-up), harvest volume bounds, road capex
  Road Network Planning: All-weather paved vs gravel secondary vs winter ice roads
========================================================================================
                                    |
                                    v
OPERATIONAL SCALE (Days to Weeks) — Harvesting, Bucking & Logistics
----------------------------------------------------------------------------------------
  Cut-to-Length (CTL) Harvester at the Stump
    --> Stem Profile Laser/Caliper Sensing (StanForD 2010 .stm)
    --> Dynamic Programming: Kozak Taper -> Bucking-to-Value (BTV) / Bucking-to-Order (BTO)
    --> Discrete Log Assortments: Veneer | Sawlogs #1/#2 | Small Sawlogs | Pulp | Biomass
        |
        v
  Forwarding to Forest Roadside Landing
        |
        v
  Seasonal Haul Dispatch & Road Thaw Dynamics
    --> Climate Index: FDD (Freezing) vs TDD (Thawing)
    --> Frost penetration z_f vs Thaw depth z_t
    --> Road Subgrade Saturation -> 50% Axle-Load Restriction / Spring Breakup Ban
    --> Truck Fleet Dispatch: 8-axle B-train vs tridem configs
========================================================================================
                                    |
                                    v
MILL BUFFER & PROCESSING (Hours to Months) — Woodyard Management & Production
----------------------------------------------------------------------------------------
  Mill Log Yard Inventory
    --> Degradation Kinetics: Moisture loss & Sapstain fungal penetration x_s(t)
    --> Protection Regimes: Dry deck FIFO vs Continuous wet deck water sprinkling
        |
        v
  Processing Facilities: Sawmill | Pulp & Paper Mill | Biomass Pellet Plant
========================================================================================
                                    ^
                                    |
COMPLIANCE & GEOSPATIAL TRACEABILITY UMBRELLA (Continuous Data Pipeline)
----------------------------------------------------------------------------------------
  - EUDR (Regulation EU 2023/1115): WGS84 Polygons (<4ha point, >=4ha polygon)
  - Satellite Due Diligence: Zero deforestation verification post Dec 31, 2020
  - Chain of Custody (FSC / PEFC): Physical Separation vs Mass Balance vs Volume Credit
  - Electronic Documentation: EU TRACES Due Diligence Statement (DDS)
========================================================================================
```

The system operates across four distinct coordinate spaces:
1. **Age space ($t$):** Governs standing biological yield and capital accumulation.
2. **Stem space ($h$):** Governs longitudinal taper, geometry, and dynamic programming cuts along the tree trunk.
3. **Geographic space ($(x, y)$):** Governs road network routing, haul distances, harvest block adjacency, and EUDR cadastral polygons.
4. **Time space ($\tau$):** Governs degree-day thaw triggers, seasonal truck payloads, fungal colonization rates, and mill delivery deadlines.

---

## 4. Definitions and vocabulary

The following table provides the technical terminology required across forest economics, harvest engineering, log grading, wood science, and regulatory compliance.

| Term | Symbol / Code | Formal Definition | Common Industry Misuse | Standard Unit |
|---|---|---|---|---|
| **Land Expectation Value** | $LEV$ or $SEV$ | Net present value of bare forest land under an infinite series of identical, optimally timed forest rotations. | Confusing with single-rotation Net Present Value (NPV), which ignores site land rent. | $\text{EUR/ha}$ or $\text{USD/ha}$ |
| **Faustmann Rotation Age** | $T^*$ | Stand harvest age that maximizes $LEV$, balancing biological timber increment against capital interest and land opportunity cost. | Assuming harvest occurs at biological maturity or maximum wood volume. | $\text{years}$ |
| **Maximum Sustainable Yield** | $MSY$ | Biological harvest age that maximizes average wood production per year, occurring where $CAI = MAI$. | Treating $MSY$ as the financial optimum; $MSY$ ignores discounting and land rent. | $\text{years}$ |
| **Mean Annual Increment** | $MAI(t)$ | Cumulative stand merchantable volume divided by stand age: $MAI(t) = V(t) / t$. | Confusing with the current year's growth increment ($CAI$). | $\text{m}^3/(\text{ha}\cdot\text{yr})$ |
| **Current Annual Increment** | $CAI(t)$ | Instantaneous or single-year rate of volume growth: $CAI(t) = dV(t)/dt$. | Assuming $CAI$ remains constant over the stand rotation. | $\text{m}^3/(\text{ha}\cdot\text{yr})$ |
| **Diameter at Breast Height** | $DBH$ | Tree stem outside-bark diameter measured at $1.30\text{ m}$ ($4.5\text{ ft}$ in US) above ground line. | Measuring at arbitrary stump height or failing to account for ground slope. | $\text{cm}$ or $\text{in}$ |
| **Small-End Diameter** | $SED$ | The top (thinnest) inside-bark diameter of a cut log segment. | Using large-end diameter or over-bark measurements for log grading. | $\text{cm}$ or $\text{in}$ |
| **Stem Taper Equation** | $d(h)$ | Mathematical profile function predicting stem diameter at any longitudinal height $h$ above ground. | Using a simple cone or cylinder model, which severely miscalculates volume. | Function output: $\text{cm}$ |
| **Cut-to-Length** | CTL | Harvesting system where a single-grip harvester fells, delimbs, and bucks stems into logs directly at the stump. | Conflating with full-tree logging where whole trees are dragged to roadside landings. | Method classification |
| **Bucking-to-Value** | BTV | Dynamic programming log cutting based purely on static product assortment price schedules. | Assuming BTV matches real-time mill requirements; it frequently over-produces unneeded lengths. | Optimization strategy |
| **Bucking-to-Order** | BTO | Harvester bucking driven by dynamic price penalties reflecting mill demand books and quota targets. | Assuming harvesters cut arbitrary customer sizes without physical log grade limits. | Optimization strategy |
| **StanForD 2010** | Standard | XML-based global communication standard for forest machine telemetry, bucking tables, and production logs. | Treating as proprietary vendor software rather than an open industry protocol. | File spec (`.prd`, `.stm`) |
| **Spring Breakup** | — | Seasonal period where thawed subgrade roads lose load-bearing capacity due to trapped meltwater. | Assuming breakup timing is fixed by calendar rather than cumulative thermal energy. | $\text{calendar window}$ |
| **Freezing Degree-Days** | $FDD$ | Cumulative sum of daily average temperatures below $0^\circ\text{C}$: $FDD = \sum \max(0, -T_{\text{mean}})$. | Calculating using minimum daily temperature instead of daily mean. | $^\circ\text{C}\cdot\text{days}$ |
| **Thawing Degree-Days** | $TDD$ | Cumulative sum of daily average temperatures above $0^\circ\text{C}$: $TDD = \sum \max(0, T_{\text{mean}})$. | Ignoring solar radiation absorption on dark asphalt or cleared gravel surfaces. | $^\circ\text{C}\cdot\text{days}$ |
| **Gross Vehicle Weight** | $GVW$ | Total laden mass of a commercial timber truck, including tractor, trailer, and log payload. | Confusing gross truck mass with allowable axle-group load limits. | $\text{tonnes}$ ($\text{t}$) |
| **Wet Decking** | — | Continuous or pulsed water sprinkling of log stockpiles to prevent drying, cracking, and fungal growth. | Believing water kills fungi; it acts by saturating wood pores and excluding oxygen. | Operational regime |
| **Fiber Saturation Point** | $FSP$ | Moisture content ($\approx 28\text{--}30\%$) where cell walls are fully saturated but cell lumens contain no free water. | Believing log shrinkage begins immediately upon harvest; shrinkage only occurs below $FSP$. | $\%$ dry-basis |
| **Sapstain / Blue Stain** | — | Fungal colonization of sapwood rays (*Ophiostoma* spp.) that discolors wood without structural decay. | Assuming blue stain rots wood; it does not reduce bending strength but destroys lumber grade. | Qualitative defect |
| **Green-up Constraint** | — | Environmental regulation prohibiting clearcutting adjacent to a recent cut until regeneration reaches a height threshold. | Treating as a non-binding guideline; violating green-up revokes environmental harvest permits. | $\text{years}$ or $\text{meters}$ |
| **Chain of Custody** | CoC | Information trail verifying that forest products originate from certified, legally harvested stands. | Assuming CoC guarantees sustainable silviculture; CoC only certifies traceability of custody. | Audited certification |
| **EUDR** | Regulation | EU Deforestation Regulation (2023/1115) mandating deforestation-free wood products with geolocation coordinates. | Assuming FSC certification automatically satisfies EUDR without plot-level polygons. | EU Law |
| **Cadastral Polygon** | — | Closed geographic boundary consisting of latitude/longitude coordinates enclosing a specific harvest plot. | Submitting a single center-point coordinate for harvest blocks exceeding 4 hectares. | WGS84 GeoJSON / KML |
| **Mass Balance** | — | CoC accounting method where certified and non-certified fibers are mixed physically, but sales volume is reconciled. | Confusing with physical segregation; mass balance allows certified claims on mixed volume. | Accounting balance |
| **Physical Separation** | — | CoC system where certified wood is segregated physically from non-certified wood at all times. | Assuming physical separation permits minor blending without dedicated batch runs. | Logistics protocol |
| **Basal Area** | $BA$ or $G$ | Total cross-sectional area of tree stems per hectare measured at breast height ($1.3\text{ m}$). | Confusing basal area with canopy ground cover percentage. | $\text{m}^2/\text{ha}$ |
| **Stand Density Index** | $SDI$ | Reineke's density metric standardizing tree stocking against a reference quadratic mean diameter of $25\text{ cm}$. | Assuming unthinned stands maintain constant tree counts as trees grow. | dimensionless index |
| **Slenderness Ratio** | $H/DBH$ | Ratio of total tree height to breast-height diameter; mechanical stability index against windthrow. | Ignoring high slenderness ratios ($> 80\text{--}90$) when scheduling commercial thinning. | $\text{m/m}$ (dimensionless) |
| **Trim Allowance** | $\delta$ | Deliberate extra length ($0.05\text{--}0.10\text{ m}$) cut on sawlogs to compensate for crosscut kerf and end-grain damage. | Omitting trim allowance in bucking algorithms, causing sawmills to reject short logs. | $\text{meters}$ ($\text{m}$) |
| **Kerf** | — | Width of the wood material removed as sawdust by a mechanical harvester saw chain or sawmill band saw blade. | Neglecting kerf volume loss across high-frequency multi-cut bucking patterns. | $\text{millimeters}$ ($\text{mm}$) |
| **Tridem Axle Group** | — | Three consecutive load-bearing axles spaced closely on heavy timber trailers to distribute gross vehicle weight. | Treating tridem load allowances as equal to three independent single-axle allowances. | axle configuration |
| **Topographic Wetness Index** | $TWI$ | Terrain wetness metric $\ln(a / \tan \beta)$ predicting soil saturation and low-bearing-capacity muskeg. | Scheduling wet-weather harvesting on high-$TWI$ terrain without winter frozen subgrades. | dimensionless index |
| **Hartman Rotation Model** | $T_{\text{Hartman}}$ | Bio-economic rotation model incorporating continuous non-timber ecosystem services and carbon sequestration flows. | Assuming carbon credit payments always shorten rotation age; carbon payments lengthen rotations. | $\text{years}$ |
| **Biochemical Oxygen Demand** | $BOD_5$ | Amount of dissolved oxygen consumed by microorganisms decomposing organic matter in wet deck runoff water. | Discharging unbuffered log yard sprinkler runoff directly into fish-bearing freshwater streams. | $\text{mg/L}$ |
| **StanForD .apt File** | `.apt` | Machine-readable Apportionment file defining target volume distribution percentages across product lengths. | Confusing target assortment distributions with individual tree stem bucking instructions. | StanForD file type |

---

## 5. The mechanics, step by step

### 5.1 Faustmann optimal forest rotation age derivation

The economic determination of when to fell an even-aged forest stand was solved by Martin Faustmann in 1849. To understand Faustmann's insight, we first contrast it with the single-rotation Fisher model and the biological Maximum Sustainable Yield (MSY) model.

#### The Biological Yield Curve
Let $V(t)$ denote the merchantable wood volume ($\text{m}^3/\text{ha}$) of a forest stand at age $t$. Stand growth follows a classic sigmoid growth trajectory, modeled using Chapman-Richards or Schumacher formulations:
$$V(t) = A \left(1 - e^{-k t}\right)^p$$
where $A$ is the asymptotic stand capacity, $k$ is the biological growth rate parameter, and $p$ is the shape parameter governing the inflection point.

The Mean Annual Increment ($MAI$) and Current Annual Increment ($CAI$) are defined as:
$$MAI(t) = \frac{V(t)}{t}, \quad CAI(t) = \frac{dV(t)}{dt} = V'(t)$$
Maximum Sustainable Yield ($MSY$) occurs at the stand age $T_{\text{MSY}}$ that maximizes long-term average physical volume production per year:
$$\frac{d}{dt}\left(\frac{V(t)}{t}\right) = \frac{V'(t) t - V(t)}{t^2} = 0 \implies V'(T_{\text{MSY}}) = \frac{V(T_{\text{MSY}})}{T_{\text{MSY}}}$$
$MSY$ occurs exactly at the intersection where $CAI(t) = MAI(t)$. This formulation ignores discounting, timber pricing, and capital costs.

#### The Fisher Single-Rotation Model
If a landowner plans to grow exactly one crop of trees and then abandon the land forever, the objective is to maximize the Net Present Value ($NPV$) of that single rotation discounted at continuous market rate $r$:
$$\max_{T} \Pi_1(T) = \left[P \cdot V(T) - C\right] e^{-r T}$$
where $P$ is the net stumpage value per cubic meter ($\text{EUR/m}^3$ after logging costs) and $C$ is the initial site preparation and planting cost ($\text{EUR/ha}$).
Taking the derivative with respect to $T$ and setting to zero:
$$\frac{d\Pi_1}{dT} = P V'(T) e^{-r T} - r \left[P V(T) - C\right] e^{-r T} = 0$$
$$\frac{P V'(T_{\text{Fisher}})}{P V(T_{\text{Fisher}}) - C} = r$$
In the Fisher formulation, the stand is harvested when the percentage rate of value growth equals the interest rate of capital $r$. However, the Fisher model makes a fatal economic omission: it assumes the land has zero value and zero opportunity cost after harvest.

#### The Faustmann Infinite-Series Formulation (Land Expectation Value)
In an ongoing forestry enterprise, harvesting a stand vacates the land, allowing an identical replacement stand to be planted immediately. Thus, delaying harvest not only ties up the capital invested in the standing trees; it delays all future revenues from all subsequent forest rotations in perpetuity.

The Net Present Value of an infinite sequence of identical rotations of duration $T$, termed the **Land Expectation Value** ($LEV$ or $SEV$), is:
$$LEV(T) = \left[P V(T) - C\right] e^{-r T} + \left[P V(T) - C\right] e^{-2 r T} + \left[P V(T) - C\right] e^{-3 r T} + \dots$$
Factoring out the first term:
$$LEV(T) = \left[P V(T) - C\right] e^{-r T} \sum_{k=0}^{\infty} \left(e^{-r T}\right)^k$$
Using the infinite geometric series sum $\sum_{k=0}^{\infty} x^k = \frac{1}{1 - x}$ for $x = e^{-r T} < 1$:
$$LEV(T) = \frac{\left[P V(T) - C\right] e^{-r T}}{1 - e^{-r T}} = \frac{P V(T) - C e^{r T}}{e^{r T} - 1}$$
Alternatively, expressing future cash flow directly discounted to time zero:
$$LEV(T) = \frac{P V(T) e^{-r T} - C}{1 - e^{-r T}}$$

#### First-Order Condition for Optimal Rotation
To find the optimal rotation age $T^*$, we differentiate $LEV(T)$ with respect to $T$ using the quotient rule:
$$\frac{d LEV(T)}{dT} = \frac{\left[P V'(T) e^{-r T} - r P V(T) e^{-r T}\right](1 - e^{-r T}) - \left[P V(T) e^{-r T} - C\right](r e^{-r T})}{(1 - e^{-r T})^2} = 0$$
Setting the numerator to zero:
$$\left[P V'(T) - r P V(T)\right] e^{-r T} (1 - e^{-r T}) = r e^{-r T} \left[P V(T) e^{-r T} - C\right]$$
Divide both sides by $e^{-r T}(1 - e^{-r T})$:
$$P V'(T) - r P V(T) = r \left[\frac{P V(T) e^{-r T} - C}{1 - e^{-r T}}\right]$$
Recognizing that the bracketed term on the right-hand side is $LEV(T)$:
$$P V'(T^*) = r P V(T^*) + r \cdot LEV(T^*)$$

#### Economic Interpretation of the Faustmann Rule
The Faustmann condition states that at the optimal harvest age $T^*$, the marginal value generated by letting the stand grow for one additional period ($P V'(T^*)$) must exactly equal:
1. $r P V(T^*)$: The opportunity cost of capital tied up in the standing timber inventory (interest forgone on the realizable stumpage value).
2. $r \cdot LEV(T^*)$: The opportunity cost of delaying all future rotations, equivalent to the annual **site rent** (land rent) of the forest parcel.

Because $r \cdot LEV(T^*) > 0$, the marginal threshold is strictly higher under Faustmann than under the Fisher model. Consequently:
$$T^*_{\text{Faustmann}} < T^*_{\text{Fisher}} < T_{\text{MSY}}$$
Accounting for land rent drives earlier harvesting. When discount rates $r$ rise, $T^*$ shortens. When regeneration costs $C$ rise, $T^*$ lengthens (to amortize planting costs over a larger wood volume).

#### The Hartman Extension: Internalizing Carbon Sequestration and Ecosystem Amenity Flows
The classical Faustmann model assumes timber extraction is the sole source of economic value. In modern commercial forestry, landowners participate in carbon offset markets (e.g., California ARB, EU ETS compliance mechanisms, or voluntary carbon markets) and receive payments for non-timber ecosystem services.

Richard Hartman (1976) generalized Faustmann's equation by incorporating an ongoing annual amenity or carbon flow benefit $E(t)$ ($\text{EUR/ha/yr}$). If standing timber absorbs $\text{CO}_2$ at an instantaneous rate proportional to current biological volume growth ($\gamma \cdot V'(t)$, where $\gamma$ is the carbon dioxide equivalent conversion factor, typically $0.9\text{--}1.2\text{ tonnes CO}_2\text{e/m}^3$), and carbon is valued at $P_{\text{carbon}}$ ($\text{EUR/t CO}_2\text{e}$), the annual carbon cash inflow is $P_{\text{carbon}} \gamma V'(t)$. Upon harvest at age $T$, a fraction $\beta$ of the accumulated biomass carbon is immediately emitted to the atmosphere (e.g., slash burning, short-lived paper products), incurring an emission penalty $\beta P_{\text{carbon}} \gamma V(T)$, while the remaining fraction $(1 - \beta)$ is stored in durable sawn wood products.

The Net Present Value of infinite rotations under the Hartman model is:
$$LEV_{\text{Hartman}}(T) = \frac{\left[P V(T) - C\right] e^{-r T} + \int_0^T E(t) e^{-r t} dt + P_{\text{carbon}} \gamma \left[ \int_0^T V'(t) e^{-r t} dt - \beta V(T) e^{-r T} \right]}{1 - e^{-r T}}$$

Differentiating with respect to $T$ yields the generalized first-order marginal condition:
$$P V'(T^*) + E(T^*) + P_{\text{carbon}} \gamma \left[ V'(T^*) - r \beta V(T^*) \right] = r P V(T^*) + r \cdot LEV_{\text{Hartman}}(T^*)$$

**Analytical Insights:**
1. **Rotation Lengthening:** Because middle-aged stands exhibit high biological volume increments ($V'(t) > 0$) and substantial standing amenity values ($E(t) > 0$), the marginal benefit of waiting is systematically higher under the Hartman formulation than under pure Faustmann. Consequently, $T^*_{\text{Hartman}} > T^*_{\text{Faustmann}}$.
2. **Convergence to Perpetual Retention:** If the price of carbon $P_{\text{carbon}}$ or amenity valuation $E(t)$ is sufficiently high relative to timber stumpage price $P$, the marginal amenity flow exceeds the capital holding cost for all ages $t$, meaning the optimal financial harvest age is infinite ($T^* \to \infty$). The optimal management strategy shifts from clearcut timber extraction to continuous-cover permanent forest conservation.

---

### 5.2 Mechanized log bucking optimization via dynamic programming

When a single-grip harvester fells a tree, the stem profile is measured by optical sensors and feed-roller rotary encoders measuring diameter at millisecond intervals. The onboard computer must divide the stem into marketable logs.

#### Stem Geometry and Taper Modeling
Tree taper is non-linear: neiloid in the lower stump, paraboloid in the middle stem, and conical near the tip. Kozak's variable-exponent taper equation is widely applied:
$$\frac{d(h)}{DBH} = a_0 \left(\frac{H - h}{H - 1.3}\right)^{a_1} \left(\frac{h}{H}\right)^{a_2} \left(\frac{DBH}{H}\right)^{a_3}$$
where $h$ is longitudinal height above ground ($0 \le h \le H$), $d(h)$ is predicted diameter inside bark at height $h$, $H$ is total tree height, and $a_i$ are species-calibrated regression coefficients.

#### Segmented Taper Modeling: Max & Burkhart Formulation
While single-equation power models like simple Kozak forms provide rapid estimation, industrial CTL harvesters frequently deploy segmented polynomial taper models, such as Max & Burkhart (1976), which divide the stem into three distinct geometric regions separated by two inflection join points:
$$\frac{d^2(h)}{DBH^2} = b_1 \left(\frac{h}{H} - 1\right) + b_2 \left(\frac{h^2}{H^2} - 1\right) + b_3 \left(a_1 - \frac{h}{H}\right)^2 I_1 + b_4 \left(a_2 - \frac{h}{H}\right)^2 I_2$$
where:
- $I_1 = 1$ if $h/H \le a_1$, $0$ otherwise (lower join point, separating the basal neiloid flare from the central paraboloid bole, typically $a_1 \approx 0.10\text{--}0.15$).
- $I_2 = 1$ if $h/H \le a_2$, $0$ otherwise (upper join point, separating the central paraboloid from the conical top, typically $a_2 \approx 0.65\text{--}0.75$).
- $b_1, b_2, b_3, b_4$ are empirical coefficients estimated via non-linear least squares.

#### Sweep and Curvature Geometry
Stem curvature ("sweep") limits the yield of straight lumber recovered at the sawmill headrig. Sweep is measured as the maximum deflection $\Delta_{\max}$ ($\text{mm}$) of the stem centerline from a straight chord spanning the length $L$ ($\text{m}$) of the candidate log:
$$\text{Sweep} = \frac{\Delta_{\max}}{L} \quad (\text{mm/m})$$
If candidate assortment $k$ has a maximum allowable sweep threshold $S_k^{\max}$, candidate cuts violating $\text{Sweep}(h, h + L_k) \le S_k^{\max}$ are pruned from the dynamic programming decision set $\mathcal{A}(i)$, forcing the optimizer to cut shorter lengths (e.g., $3.0\text{ m}$ instead of $4.8\text{ m}$) or downgrade the bowed section to pulpwood.

#### Discretization and Product Assortment Matrix
The stem length is discretized into intervals of step size $\Delta h$ (typically $0.05\text{ m}$ to $0.10\text{ m}$).
Let the stem be represented as a discrete sequence of nodes $i \in \{0, 1, 2, \dots, N\}$, where node $i$ corresponds to cut height $h_i = i \cdot \Delta h$.

A mill defines an assortment portfolio $\mathcal{K} = \{1, 2, \dots, K\}$ (e.g., Veneer, Large Sawlog, Small Sawlog, Studwood, Pulpwood). Each assortment $k \in \mathcal{K}$ is defined by:
- Nominal cut length $L_k$ ($\text{m}$).
- Minimum trim allowance $\delta_k$ (typically $0.05\text{--}0.10\text{ m}$ to absorb end damage and cross-cut kerf).
- Minimum allowable Small-End Diameter $SED_k^{\min}$ ($\text{cm}$).
- Maximum allowable sweep or curvature $S_k^{\max}$ ($\text{mm/m}$).
- Unit net price $P_k$ ($\text{EUR/m}^3$).

#### Volume Formulation (Smalian's Formula)
For a candidate log cut spanning from lower height $h_{\text{base}}$ to top cut height $h_{\text{top}} = h_{\text{base}} + L_k$, with trim allowance $\delta_k$ discarded, the merchantable log volume $V(h_{\text{base}}, h_{\text{top}})$ is calculated via Smalian's formula:
$$V(h_{\text{base}}, h_{\text{top}}) = \frac{\pi}{8} \left[ \left(\frac{d(h_{\text{base}})}{100}\right)^2 + \left(\frac{d(h_{\text{top}})}{100}\right)^2 \right] L_k$$
where diameters $d$ are in centimeters and length $L_k$ is in meters, yielding volume in $\text{m}^3$.

#### The Dynamic Programming Recursion
The problem possesses optimal substructure and can be formulated as a backward dynamic program.
Let $f(i)$ denote the maximum financial value obtainable from the remaining upper portion of the stem from node $i$ ($h_i$) to the top of the tree $H$.

**Boundary condition:**
$$f(N) = 0$$

**Bellman recurrence relation for $i = N-1, N-2, \dots, 0$:**
$$f(i) = \max \left\{ f(i+1), \max_{k \in \mathcal{A}(i)} \left[ P_k \cdot V(h_i, h_i + L_k) + f\left(i + \frac{L_k + \delta_k}{\Delta h}\right) \right] \right\}$$
where $\mathcal{A}(i)$ is the set of kinematically feasible product assortments at node $i$:
$$\mathcal{A}(i) = \left\{ k \in \mathcal{K} \;\middle|\; d(h_i + L_k) \ge SED_k^{\min}, \; h_i + L_k + \delta_k \le H, \; \text{Sweep}(h_i, h_i+L_k) \le S_k^{\max} \right\}$$
The first term $f(i+1)$ represents advancing by one increment $\Delta h$ without producing a log (treating that slice as waste or bucking around a localized defect like a fork, rot pocket, or severe sweep dog-leg).

#### Bucking-to-Order (BTO) Demand Feedback
When mill log yards have strict quota requirements, static prices $P_k$ cause oversupply of dominant log lengths. Modern harvesters deploy Bucking-to-Order (BTO) algorithms incorporating dynamic price modifiers $\lambda_k(t)$:
$$P_k^{\text{effective}} = P_k \cdot \left[1 + \lambda_k(t)\right]$$
where $\lambda_k(t)$ is a shadow price adjusted in real time based on whether cumulative cut volume of product $k$ is lagging or exceeding the delivery contract.

---

### 5.3 Seasonal harvest scheduling, road thaw physics, and axle-load dynamics

Forest transportation networks in northern latitudes rely on three distinct road classifications:
1. **Class 1 (All-Weather Paved):** Designed for standard legal highway gross weights year-round.
2. **Class 2 (Gravel Secondary & Forest Spurs):** Unpaved, aggregate-surfaced roads susceptible to moisture saturation.
3. **Class 3 (Winter / Ice Roads):** Temporary corridors frozen over peat bogs, wetlands, and water bodies, operational only under sustained sub-zero temperatures.

```text
Cross-Section of Boreal Forest Road During Spring Breakup:
+-------------------------------------------------------------+
|    Tire Contact Surface: Axle Load P                        |
+-------------------------------------------------------------+
|  1. THAWED SURFACE LAYER (0 to z_t): Saturated Slurry       |
|     - Soil: Silt / Fine Sand with Trapped Meltwater         |
|     - Pore Water Pressure u > 0                             |
|     - Effective Stress sigma' = sigma - u -> 0              |
|     - Resilient Modulus M_r drops 60-75% (Rutting/Punching) |
+-------------------------------------------------------------+
|  2. THAWING FRONT: z_t (Descending at rate proportional     |
|     to sqrt(TDD))                                           |
+-------------------------------------------------------------+
|  3. IMPERMEABLE FROZEN CORE: Frozen subgrade / Permafrost   |
|     - Prevents downward drainage of meltwater               |
|     - Acts as an impermeable hydraulic dam                  |
+-------------------------------------------------------------+
```

#### Soil Freezing and Thawing Physics
The penetration of frost into the roadbed during winter is governed by the Stefan equation:
$$z_f(t) = \sqrt{\frac{2 k_{\text{frozen}} \cdot FDD(t)}{L \cdot \rho_d \cdot w}}$$
where $z_f$ is frost depth ($\text{m}$), $k_{\text{frozen}}$ is thermal conductivity of frozen soil ($\text{W}/(\text{m}\cdot\text{K})$), $L$ is latent heat of fusion of water ($334\text{ kJ/kg}$), $\rho_d$ is dry density ($\text{kg/m}^3$), $w$ is soil water content (decimal), and $FDD(t)$ is cumulative Freezing Degree-Days ($^\circ\text{C}\cdot\text{days}$).

During spring, solar radiation and positive ambient temperatures initiate surface thaw. The depth of the thawing front $z_t(t)$ follows:
$$z_t(t) = \beta \sqrt{TDD(t)}$$
where $\beta$ is the empirical thawing soil coefficient ($\text{m}/(^\circ\text{C}\cdot\text{days})^{0.5}$) and $TDD(t)$ is cumulative Thawing Degree-Days.

#### Structural Failure Mechanism During Breakup
As $z_t$ advances downward (typically $0.2\text{--}0.8\text{ m}$), the underlying road base between $z_t$ and $z_f$ remains frozen solid and impermeable. Meltwater cannot drain downward. Wheel loads from heavy haul trucks generate excess pore water pressure ($u > 0$). According to Terzaghi's effective stress principle:
$$\sigma' = \sigma - u$$
When pore pressure spikes, effective shear strength $\tau = c' + \sigma' \tan \phi'$ collapses. The road subgrade loses shear strength, resulting in deep rutting, subgrade liquefaction, and road destruction.

#### Transport Capacity Loss and Buffer Sizing
To prevent infrastructure failure, highway authorities enforce Spring Load Restrictions (SLR):
- Gross Vehicle Weight ($GVW$) is reduced by $30\text{--}50\%$, or axle groups are capped at $5.0\text{ tonnes}$ per axle (vs normal $8.5\text{--}10.0\text{ tonnes}$).
- Haul costs scale inversely with payload: if a standard $32\text{-tonne}$ net payload truck is restricted to $16\text{ tonnes}$, hauling cost per cubic meter doubles.
- Many forest spur roads are closed entirely to heavy transport for 4 to 8 weeks.

Mills must solve a tactical inventory surge model. If a mill consumes $D_{\text{mill}}$ ($\text{m}^3/\text{day}$) and the breakup road ban lasts $\Delta t_{\text{breakup}}$ days, the minimum safety stock $I_{\text{buffer}}$ required at the mill yard prior to the onset of breakup ($TDD \ge TDD_{\text{threshold}}$) is:
$$I_{\text{buffer}} = D_{\text{mill}} \cdot \Delta t_{\text{breakup}} + z_{\alpha} \cdot \sigma_D \sqrt{\Delta t_{\text{breakup}}}$$

#### Temperature-Triggered Spring Load Restrictions (SLR)
Highway transport departments in boreal zones (such as Minnesota DOT, Ontario Ministry of Transportation, and Trafikverket in Sweden) replace subjective calendar dates with thermal degree-day cumulative triggers:
- **SLR Onset Condition:** Axle-load restrictions are triggered when the cumulative Thawing Degree-Day index ($TDD$) reaches a threshold:
  $$TDD(t) = \sum_{j=1}^t \max\left(0, \bar{T}_j - T_{\text{base}}\right) \ge TDD_{\text{trigger}} \approx 15\text{--}25\text{ }^\circ\text{C}\cdot\text{days}$$
  where $T_{\text{base}} = -1.5^\circ\text{C}$ to account for solar absorption on aggregate road surfaces, and the 3-day meteorological weather forecast predicts sustained positive temperatures.
- **SLR Removal Condition:** As meltwater slowly drains through ditch infrastructure, subgrade pore pressure dissipates. Road bearing capacity recovers when cumulative thawing reaches:
  $$TDD(t) \ge TDD_{\text{removal}} \approx 180\text{--}220\text{ }^\circ\text{C}\cdot\text{days}$$
  or when continuous falling-weight deflectometer (FWD) measurements indicate subgrade resilient modulus has rebounded to $\ge 85\%$ of summer baseline.

The dynamic evolution of subgrade resilient modulus $M_r(\tau)$ across the spring cycle is captured by:
$$M_r(\tau) = M_{r, \text{base}} \cdot \left[ 1 - \Delta_{\text{thaw}} \cdot \exp\left( -\frac{(\tau - \tau_{\text{peak}})^2}{2 \sigma_{\tau}^2} \right) \right]$$
where $M_{r, \text{base}}$ is the normal summer modulus ($\approx 80\text{--}120\text{ MPa}$ for crushed aggregate over sandy silt), $\Delta_{\text{thaw}} \approx 0.60\text{--}0.75$ represents the maximum fractional stiffness collapse, and $\tau_{\text{peak}}$ corresponds to peak subgrade saturation.

---

### 5.4 Log yard storage, drying kinetics, and sapstain fungal degradation

Storing massive buffer inventories (frequently $50,000\text{--}200,000\text{ m}^3$) in unpaved mill yards exposes roundwood to biological degradation.

```text
Log Yard Biological Degradation Kinetics:
        Dry-Deck Storage (Air Exposure)            Wet-Deck Storage (Sprinkler Fogging)
+---------------------------------------------+  +---------------------------------------------+
| Log End Checks & Radial Cracks              |  | Water Film Maintained Over Log Bark         |
| Fungal Spores Infiltrate Sapwood Rays       |  | Cell Lumens Completely Flooded with H2O     |
| Moisture: 45% (FSP < MC < Wet Limit)        |  | Moisture Content: > 120% dry-basis          |
| Blue Stain Radial Growth: 0.8 mm/day        |  | Dissolved O2 Depleted (< 1-2%)              |
| Result: Sawlog Downgraded (-50% Value Loss) |  | Fungi Asphyxiated -> Zero Stain Penetration |
+---------------------------------------------+  +---------------------------------------------+
```

#### Moisture Loss and Checking
Freshly cut softwood has a sapwood moisture content of $100\text{--}160\%$ (dry-weight basis). When stored in dry decks, water evaporates from the exposed cross-cut log ends at up to 10 times the rate of bark-covered longitudinal surfaces.
When sapwood moisture drops below the Fiber Saturation Point ($FSP \approx 28\text{--}30\%$), wood cells shrink. Differential circumferential vs radial shrinkage generates severe tensile stresses, causing radial end-checks (splits) that penetrate $0.3\text{--}1.0\text{ m}$ into the log ends, rendering the ends useless for structural lumber.

#### Sapstain Fungal Growth Kinetics
Sapstain (blue stain) is caused by ascomycete fungi, primarily *Ophiostoma piceae*, *Ophiostoma piliferum*, and *Ceratocystis coerulescens*. Fungal hyphae colonize parenchyma cells in wood rays, utilizing non-structural nutrients (sugars, starches, fatty acids). The dark melanin in the fungal cell walls imparts a deep blue-black discoloration to the sapwood.

Fungal radial penetration velocity $v_s$ depends on temperature $T$ and wood moisture content $MC$:
$$v_s(T, MC) = v_{\max} \cdot \Phi_T(T) \cdot \Phi_M(MC)$$
The temperature response $\Phi_T(T)$ follows a cardinal temperature function:
$$\Phi_T(T) = \begin{cases} 
0, & T \le T_{\min} \text{ or } T \ge T_{\max} \\
\frac{(T - T_{\min})(T - T_{\max})}{(T - T_{\min})(T - T_{\max}) - (T - T_{\text{opt}})^2}, & T_{\min} < T < T_{\max}
\end{cases}$$
For *Ophiostoma piceae*, typical cardinal parameters are $T_{\min} = 3^\circ\text{C}$, $T_{\text{opt}} = 23^\circ\text{C}$, and $T_{\max} = 35^\circ\text{C}$, with peak radial growth rate $v_{\max} \approx 1.2\text{ mm/day}$.

The moisture modifier $\Phi_M(MC)$ enforces strict physical boundaries:
$$\Phi_M(MC) = \begin{cases}
0, & MC < FSP \approx 28\% \quad \text{(wood too dry for metabolic water)} \\
1.0, & 40\% \le MC \le 90\% \quad \text{(optimal fungal growth window)} \\
0, & MC > 120\% \quad \text{(cell lumens flooded; hypoxia/anoxia halts hyphal respiration)}
\end{cases}$$

#### The Physics of Wet Deck Sprinkling
Wet decking prevents sapstain not by washing spores away, but by exploiting the upper moisture boundary $\Phi_M(MC > 120\%) = 0$.
Continuous or cyclical water sprinkling covers logs in an unbroken water film. Free water infiltrates the sapwood, driving moisture content above $120\text{--}150\%$ and displacing air from the cell lumens. The diffusion coefficient of dissolved oxygen in water ($D_{\text{O}_2} \approx 2 \times 10^{-5}\text{ cm}^2/\text{s}$) is four orders of magnitude slower than in air ($D_{\text{O}_2} \approx 0.2\text{ cm}^2/\text{s}$). Oxygen consumption by wood respiration rapidly depletes dissolved oxygen below $1\text{--}2\%$, asphyxiating fungal hyphae and stopping blue stain propagation completely.

#### Fickian Wood Moisture Diffusion and Radial Drying
Moisture movement within stored roundwood above the Fiber Saturation Point is driven by capillary pressure gradients in free water; below $FSP$, moisture movement is governed by bound-water Fickian diffusion:
$$\frac{\partial MC}{\partial t} = \frac{1}{r} \frac{\partial}{\partial r}\left( r D_r \frac{\partial MC}{\partial r} \right) + \frac{\partial}{\partial z}\left( D_z \frac{\partial MC}{\partial z} \right)$$
where $D_r$ and $D_z$ are the radial and longitudinal diffusion coefficients ($\text{m}^2/\text{s}$). Longitudinal diffusion along the open tracheids or vessels is 10 to 25 times faster than radial diffusion through bark ($D_z \approx 10\text{--}25 \times D_r$). Consequently, unprotected log ends desaturate rapidly, dropping below $FSP = 28\%$ within 7 to 14 days of dry-deck exposure during warm weather, creating high circumferential tensile stress and radial splitting.

#### Environmental Leachate and Runoff Management
While wet decking protects wood value, continuous water sprinkling extracts water-soluble organic compounds from tree bark and sapwood. Wet-deck runoff carries substantial concentrations of:
- **Tannins and Lignins:** Impart dark tea-colored turbidity and high Chemical Oxygen Demand ($COD \approx 500\text{--}2,500\text{ mg/L}$).
- **Resin Acids and Fatty Acids (RFA):** Extracted from conifers (e.g., abietic, dehydroabietic, and isopimaric acids), which are acutely toxic to salmonids and aquatic organisms at concentrations above $0.5\text{--}1.0\text{ mg/L}$.
- **Biochemical Oxygen Demand ($BOD_5$):** Averages $200\text{--}800\text{ mg/L}$ due to leached simple sugars and starches.

Under environmental regulations (US Clean Water Act NPDES permits, EU Water Framework Directive 2000/60/EC), wet decks must be engineered with impervious clay-lined collection ditches, settling basins, and dedicated recirculation lagoons. Sprinkler water is recycled continuously, with blowdown water routed through constructed wetlands or municipal wastewater treatment facilities before environmental release.

---

### 5.5 Chain of Custody (CoC) architectures and EUDR geospatial due diligence

Forest certification and international trade compliance enforce traceability from consumer finished goods back to the forest stand.

#### Certification Tracking Models
Under FSC-STD-40-004 V3-1 and PEFC ST 2002:2020, organizations must implement one of three chain of custody models:
1. **Physical Separation Model:** Certified material is physically segregated from non-certified material throughout harvesting, transport, log yard storage, and sawing. Mandated for certified niche lines (e.g., FSC 100%). High tracking cost; zero blending permitted.
2. **Percentage Model:** Certified and controlled non-certified inputs are processed together. The output is labeled with the exact percentage of certified input:
   $$\text{Certified } \% = \frac{V_{\text{certified}}}{V_{\text{certified}} + V_{\text{controlled}}} \times 100$$
3. **Credit System:** Inputs are blended. The organization accumulates "volume credits" based on certified input volumes multiplied by an audited conversion factor:
   $$\text{Credit}_{\text{available}} = \sum \left(V_{\text{in}}^{\text{cert}} \times \eta_{\text{recovery}}\right) - V_{\text{out}}^{\text{claimed}}$$
   Credits can be stored in a credit account for up to 12 months to label specific production batches as 100% certified.

#### EUDR Statutory Framework (Regulation (EU) 2023/1115)
The EU Deforestation Regulation imposes statutory obligations that supersede voluntary certification:
- **Scope:** Seven commodities (Cattle, Cocoa, Coffee, Oil palm, Rubber, Soya, and **Wood**—including roundwood, sawn timber, pulp, paper, and furniture).
- **Core Requirement:** Products placed on or exported from the EU market must be:
  1. **Deforestation-free:** Produced on land that was not subject to deforestation or forest degradation after **December 31, 2020**.
  2. **Legally produced:** Compliant with all applicable domestic legislation in the country of harvest (land tenure, environmental laws, labor rights, indigenous consent).
  3. **Covered by a Due Diligence Statement (DDS).**

#### Technical Data Schema for Geolocation
The EUDR mandates that operators collect and submit precise geolocation coordinates:
- For harvest parcels **$< 4\text{ hectares}$:** A single geographic coordinate point (latitude and longitude) with a minimum of **six decimal places** of precision (resolving location to within $\approx 0.1\text{ m}$).
- For harvest parcels **$\ge 4\text{ hectares}$:** A closed **cadastral polygon** enclosing the harvest boundary, defined by an ordered list of vertices in WGS84 (EPSG:4326) coordinate reference system.

```json
{
  "eudr_due_diligence_statement": {
    "operator_id": "EORI-SE1234567890",
    "commodity_hs_code": "440310",
    "country_of_production": "SWE",
    "harvest_period": {
      "start": "2025-10-01",
      "end": "2025-10-14"
    },
    "plots": [
      {
        "plot_id": "SE-VB-2025-BL88",
        "area_hectares": 28.5,
        "geometry": {
          "type": "Polygon",
          "coordinates": [[
            [16.824512, 62.481234],
            [16.831204, 62.486789],
            [16.842180, 62.483456],
            [16.835670, 62.478901],
            [16.824512, 62.481234]
          ]]
        },
        "deforestation_free_verification": {
          "baseline_date": "2020-12-31",
          "satellite_sensor": "Sentinel-2 / Hansen GFC",
          "canopy_loss_detected": false
        }
      }
    ]
  }
}
```

#### Automated Satellite Verification Pipeline
Before submitting a Due Diligence Statement to the EU TRACES system, the enterprise runs an automated verification pipeline:
1. **Cadastral Boundary Ingestion:** Harvester GPS tracks and harvest boundary polygons are ingested via StanForD 2010 `.geo` or GIS shapefiles.
2. **Deforestation Baselines:** Spatial intersections are evaluated against the Copernicus Global Land Service ($10\text{ m}$ tree cover density) and Hansen Global Forest Change datasets.
3. **Spectral Time-Series Analysis:** Normalized Difference Vegetation Index (NDVI) and radar backscatter (Sentinel-1 SAR VV/VH polarization) are evaluated from January 1, 2020 through the harvest date to verify that tree cover canopy was maintained continuously through December 31, 2020, ruling out illegal land-use conversion.

---

## 6. Worked example

### 6.1 Strategic rotation optimization (Faustmann vs Fisher vs MSY)

A forestry corporation in Dalarna, Sweden manages an even-aged stand of Scots pine (*Pinus sylvestris*). 
- Growth function parameters: $A = 450.0\text{ m}^3/\text{ha}$, $k = 0.04\text{ yr}^{-1}$, $p = 2.50$:
  $$V(t) = 450.0 \left(1 - e^{-0.04 t}\right)^{2.50}$$
- Stand economic parameters:
  - Net stumpage price: $P = 65.00\text{ EUR/m}^3$ (net of logging and hauling costs).
  - Site preparation and replanting cost: $C = 1,800.00\text{ EUR/ha}$.
  - Real continuous discount rate: $r = 4.5\% = 0.045\text{ yr}^{-1}$.

#### Calculation of Maximum Sustainable Yield (MSY)
$MSY$ maximizes $MAI(t) = V(t) / t$. We differentiate:
$$\frac{d}{dt}\left[\frac{450(1 - e^{-0.04 t})^{2.5}}{t}\right] = 0$$
Evaluating numerically across candidate ages:
- At $t = 38\text{ yr}$: $V(38) = 237.4\text{ m}^3/\text{ha}$, $MAI(38) = 6.25\text{ m}^3/(\text{ha}\cdot\text{yr})$.
- At $t = 40.47\text{ yr}$: $V(40.47) = 259.0\text{ m}^3/\text{ha}$, $MAI(40.47) = 6.40\text{ m}^3/(\text{ha}\cdot\text{yr})$ (maximum).
- At $t = 45\text{ yr}$: $V(45) = 296.8\text{ m}^3/\text{ha}$, $MAI(45) = 6.60 \dots \text{ declining}$.
The biological rotation age is $T_{\text{MSY}} = 40.5\text{ years}$.

#### Calculation of Fisher Single-Rotation Age
The Fisher model maximizes $\Pi_1(t) = \left[P V(t) - C\right] e^{-r t}$.
The marginal condition is:
$$\frac{P V'(t)}{P V(t) - C} = r = 0.045$$
At $t = 31.90\text{ yr}$:
- Volume: $V(31.90) = 198.3\text{ m}^3/\text{ha}$.
- Rate of volume growth: $V'(31.90) = 8.16\text{ m}^3/(\text{ha}\cdot\text{yr})$.
- Value of growth: $P V'(31.90) = 65.00 \times 8.16 = 530.40\text{ EUR/ha/yr}$.
- Standing net value: $P V(31.90) - C = 65.00 \times 198.3 - 1,800.00 = 11,089.50\text{ EUR/ha}$.
- Capital growth rate: $\frac{530.40}{11,089.50} = 0.0478 \approx 0.045$ at exact solution $t = 31.90\text{ years}$.
- Net present value of single rotation: $\Pi_1(31.90) = 11,089.50 \times e^{-0.045 \times 31.90} = 2,642.79\text{ EUR/ha}$.

#### Calculation of Faustmann Infinite Rotation Age ($T^*$)
The Faustmann formula maximizes:
$$LEV(T) = \frac{P V(T) e^{-r T} - C}{1 - e^{-r T}}$$
Evaluating $LEV(T)$ numerically:
- At $T = 24\text{ yr}$: $V(24) = 135.2\text{ m}^3/\text{ha}$, $LEV(24) = 1,735.10\text{ EUR/ha}$.
- At $T = 26.47\text{ yr}$: $V(26.47) = 154.9\text{ m}^3/\text{ha}$, $LEV(26.47) = 1,815.99\text{ EUR/ha}$ (peak).
- At $T = 30\text{ yr}$: $V(30) = 184.2\text{ m}^3/\text{ha}$, $LEV(30) = 1,752.40\text{ EUR/ha}$.

Checking the Faustmann marginal condition at $T^* = 26.47\text{ years}$:
- Volume growth rate: $V'(26.47) = 8.239\text{ m}^3/(\text{ha}\cdot\text{yr})$.
- Marginal value of biological delay: $P V'(T^*) = 65.00 \times 8.239 = 535.53\text{ EUR/ha/yr}$.
- Timber capital holding cost: $r P V(T^*) = 0.045 \times 65.00 \times 154.9 = 453.81\text{ EUR/ha/yr}$.
- Site rent (land opportunity cost): $r \cdot LEV(T^*) = 0.045 \times 1,815.99 = 81.72\text{ EUR/ha/yr}$.
- Total opportunity cost: $453.81 + 81.72 = 535.53\text{ EUR/ha/yr}$.

**Result:** The marginal value exactly equals total opportunity cost ($535.53 = 535.53$). Accounting for site rent reduces the optimal harvest age from $31.9\text{ years}$ (Fisher) and $40.5\text{ years}$ (MSY) down to **$26.5\text{ years}$**.

---

### 6.2 Stem bucking dynamic programming execution

A feller-harvester encounters a felled Scots pine stem with:
- Total height: $H = 25.0\text{ m}$.
- Breast height diameter: $DBH = 38.0\text{ cm}$.
- Simplified parabolic-neiloid taper equation:
  $$d(h) = DBH \cdot \left(\frac{H - h}{H - 1.3}\right)^{0.75} = 38.0 \cdot \left(\frac{25.0 - h}{23.7}\right)^{0.75}$$
- Assortment specification catalog:

| Assortment | Product Name | Nominal Length $L_k$ | Trim $\delta_k$ | Min SED ($SED_k^{\min}$) | Unit Price $P_k$ |
|---|---|---|---|---|---|
| **Assortment 1** | Veneer Log | $5.0\text{ m}$ | $0.10\text{ m}$ | $30.0\text{ cm}$ | $125.00\text{ EUR/m}^3$ |
| **Assortment 2** | Sawlog #1 | $4.2\text{ m}$ | $0.10\text{ m}$ | $22.0\text{ cm}$ | $90.00\text{ EUR/m}^3$ |
| **Assortment 3** | Sawlog #2 | $3.6\text{ m}$ | $0.10\text{ m}$ | $16.0\text{ cm}$ | $70.00\text{ EUR/m}^3$ |
| **Assortment 4** | Pulpwood | $2.4\text{ m}$ | $0.10\text{ m}$ | $8.0\text{ cm}$ | $32.00\text{ EUR/m}^3$ |

#### Step-by-Step Optimal Bucking Solution
Solving the backward DP recursion yields the global optimal cut sequence along the stem:

1. **Cut 1 (Stump to $5.1\text{ m}$): Veneer Log**
   - Base height: $h_0 = 0.0\text{ m}$, Top height: $h_{\text{top}} = 5.0\text{ m}$ (plus $0.1\text{ m}$ trim).
   - $d(0.0) = 38.0 \times (25.0/23.7)^{0.75} = 39.56\text{ cm}$.
   - $d(5.0) = 38.0 \times (20.0/23.7)^{0.75} = 33.47\text{ cm} \ge 30.0\text{ cm}$ (Valid Veneer).
   - Smalian Volume: $V = \frac{\pi}{8} [(0.3956)^2 + (0.3347)^2] \times 5.0 = 0.5273\text{ m}^3$.
   - Value: $0.5273\text{ m}^3 \times 125.00\text{ EUR/m}^3 = \mathbf{65.91\text{ EUR}}$.

2. **Cut 2 ($5.1\text{ m}$ to $9.4\text{ m}$): Sawlog #1**
   - Base height: $h_1 = 5.1\text{ m}$, Top height: $h_{\text{top}} = 5.1 + 4.2 = 9.3\text{ m}$ (plus $0.1\text{ m}$ trim).
   - $d(5.1) = 38.0 \times (19.9/23.7)^{0.75} = 33.32\text{ cm}$.
   - $d(9.3) = 38.0 \times (15.7/23.7)^{0.75} = 27.91\text{ cm} \ge 22.0\text{ cm}$ (Valid Sawlog #1).
   - Smalian Volume: $V = \frac{\pi}{8} [(0.3332)^2 + (0.2791)^2] \times 4.2 = 0.3116\text{ m}^3$.
   - Value: $0.3116\text{ m}^3 \times 90.00\text{ EUR/m}^3 = \mathbf{28.04\text{ EUR}}$.

3. **Cut 3 ($9.4\text{ m}$ to $13.1\text{ m}$): Sawlog #2**
   - Base height: $h_2 = 9.4\text{ m}$, Top height: $h_{\text{top}} = 9.4 + 3.6 = 13.0\text{ m}$ (plus $0.1\text{ m}$ trim).
   - $d(9.4) = 38.0 \times (15.6/23.7)^{0.75} = 27.77\text{ cm}$.
   - $d(13.0) = 38.0 \times (12.0/23.7)^{0.75} = 22.82\text{ cm} \ge 16.0\text{ cm}$ (Valid Sawlog #2).
   - Smalian Volume: $V = \frac{\pi}{8} [(0.2777)^2 + (0.2282)^2] \times 3.6 = 0.1827\text{ m}^3$.
   - Value: $0.1827\text{ m}^3 \times 70.00\text{ EUR/m}^3 = \mathbf{12.79\text{ EUR}}$.

4. **Cut 4 ($13.1\text{ m}$ to $16.8\text{ m}$): Sawlog #2**
   - Base height: $h_3 = 13.1\text{ m}$, Top height: $h_{\text{top}} = 13.1 + 3.6 = 16.7\text{ m}$ (plus $0.1\text{ m}$ trim).
   - $d(13.1) = 38.0 \times (11.9/23.7)^{0.75} = 22.68\text{ cm}$.
   - $d(16.7) = 38.0 \times (8.3/23.7)^{0.75} = 17.31\text{ cm} \ge 16.0\text{ cm}$ (Valid Sawlog #2).
   - Smalian Volume: $V = \frac{\pi}{8} [(0.2268)^2 + (0.1731)^2] \times 3.6 = 0.1151\text{ m}^3$.
   - Value: $0.1151\text{ m}^3 \times 70.00\text{ EUR/m}^3 = \mathbf{8.06\text{ EUR}}$.

5. **Cut 5 ($16.8\text{ m}$ to $19.3\text{ m}$): Pulpwood**
   - Base height: $h_4 = 16.8\text{ m}$, Top height: $h_{\text{top}} = 16.8 + 2.4 = 19.2\text{ m}$ (plus $0.1\text{ m}$ trim).
   - $d(16.8) = 17.15\text{ cm}$, $d(19.2) = 13.17\text{ cm} \ge 8.0\text{ cm}$.
   - Smalian Volume: $V = \frac{\pi}{8} [(0.1715)^2 + (0.1317)^2] \times 2.4 = 0.0441\text{ m}^3$.
   - Value: $0.0441\text{ m}^3 \times 32.00\text{ EUR/m}^3 = \mathbf{1.41\text{ EUR}}$.

6. **Cut 6 ($19.3\text{ m}$ to $21.8\text{ m}$): Pulpwood**
   - Base height: $h_5 = 19.3\text{ m}$, Top height: $h_{\text{top}} = 19.3 + 2.4 = 21.7\text{ m}$ (plus $0.1\text{ m}$ trim).
   - $d(19.3) = 13.00\text{ cm}$, $d(21.7) = 8.65\text{ cm} \ge 8.0\text{ cm}$.
   - Smalian Volume: $V = \frac{\pi}{8} [(0.1300)^2 + (0.0865)^2] \times 2.4 = 0.0230\text{ m}^3$.
   - Value: $0.0230\text{ m}^3 \times 32.00\text{ EUR/m}^3 = \mathbf{0.74\text{ EUR}}$.

7. **Stem Top ($21.8\text{ m}$ to $25.0\text{ m}$): Non-merchantable Top / Slash ($3.2\text{ m}$)**
   - Top diameter $< 8.0\text{ cm}$. Left in woods for nutrient cycling.

**Total Realized Stem Value:**
$$\text{Total Value} = 65.91 + 28.04 + 12.79 + 8.06 + 1.41 + 0.74 = \mathbf{116.95\text{ EUR}}$$
**Comparison with Myopic Bucking:** If an uncalibrated operator cut two $4.2\text{ m}$ Sawlogs from the base instead of the $5.0\text{ m}$ Veneer log, the stem realizes zero veneer, producing two Sawlog #1s ($2 \times 28.00\text{ EUR} = 56.00\text{ EUR}$), yielding a total stem value of only $94.20\text{ EUR}$. Dynamic programming generates a **$+24.1\%$ value uplift**.

---

### 6.3 Spring breakup logistics and woodyard inventory buffer sizing

A pulp and sawmilling complex in Sundsvall consumes $D_{\text{mill}} = 14,000\text{ m}^3/\text{week}$ of softwood ($2,000\text{ m}^3/\text{day}$).
- The meteorological station projects $FDD = 920\text{ }^\circ\text{C}\cdot\text{days}$ during the winter freeze.
- Frost penetration: $z_f = 0.07 \sqrt{920} = 2.12\text{ m}$.
- Road authorities announce that thawing degree-days ($TDD$) will trigger an axle-load restriction period lasting **6 weeks (42 days)**, during which all secondary gravel forest spurs are closed to trucks with gross weights exceeding $12.0\text{ tonnes}$.
- Haulage during breakup from forest landings is completely halted.

#### Buffer Stock Calculation
- Total mill consumption during breakup:
  $$Q_{\text{breakup}} = 6\text{ weeks} \times 14,000\text{ m}^3/\text{week} = 84,000\text{ m}^3$$
- Safety stock allowance for weather volatility ($\sigma_{\text{thaw}} = 1.0\text{ week}$, $95\%$ service level $z_{0.95} = 1.645$):
  $$SS = 1.645 \times 14,000\text{ m}^3 = 23,030\text{ m}^3$$
- Required Target Woodyard Pre-Thaw Stock:
  $$I_{\text{target}} = 84,000 + 23,030 = 107,030\text{ m}^3$$
- Pre-breakup surge logistics: To build this $107,030\text{ m}^3$ stockpile over the 8 weeks of peak frozen road conditions preceding breakup, the enterprise must haul $14,000 + (107,030 / 8) = 27,379\text{ m}^3/\text{week}$ ($3,911\text{ m}^3/\text{day}$).
- Fleet sizing: Using $32\text{-tonne}$ ($38\text{ m}^3$) payload timber trucks operating on frozen Class 2/3 roads running 2 turns per day:
  $$\text{Trucks Required} = \frac{3,911\text{ m}^3/\text{day}}{38\text{ m}^3/\text{trip} \times 2\text{ trips/day}} = 51.5 \implies \mathbf{52\text{ trucks}}$$

---

### 6.4 Log yard sapstain kinetics and wet deck economics

The mill enters the 6-week spring breakup with $107,030\text{ m}^3$ in the woodyard, of which $55,000\text{ m}^3$ are premium sawlogs (Veneer and Sawlog #1/#2) valued at an average of $90.00\text{ EUR/m}^3$.
- Ambient spring temperatures average $T = 18^\circ\text{C}$ during May and June.
- Fungal sapstain (*Ophiostoma piceae*) radial growth velocity:
  $$\Phi_T(18) = \frac{(18 - 3)(18 - 35)}{(18 - 3)(18 - 35) - (18 - 23)^2} = \frac{15 \times (-17)}{-255 - 25} = \frac{-255}{-280} = 0.911$$
  $$v_s = 1.2\text{ mm/day} \times 0.911 = 1.093\text{ mm/day}$$
- Over a 35-day average dry-deck dwell time, fungal hyphae penetrate:
  $$x_s = 35\text{ days} \times 1.093\text{ mm/day} = 38.3\text{ mm} \approx 3.8\text{ cm}$$
- Because average sawlog sapwood depth is $4.0\text{ cm}$, a $3.8\text{ cm}$ penetration discolors the entire outer sapwood envelope.
- Degradation penalty: Stained sawlogs cannot produce appearance-grade architectural lumber and are downgraded to utility/packaging grade or chipped for pulping, reducing realized value from $90.00\text{ EUR/m}^3$ to $52.00\text{ EUR/m}^3$ (a loss of $38.00\text{ EUR/m}^3$).
- Total potential inventory loss under dry deck storage:
  $$\text{Loss}_{\text{dry}} = 55,000\text{ m}^3 \times 38.00\text{ EUR/m}^3 = \mathbf{2,090,000\text{ EUR}}$$

#### Wet Deck Sprinkling Economics
- Capital expenditure: Installing pipe risers, high-pressure centrifugal pumps, and retention ponds over 8 hectares of woodyard: $\text{Capex} = 220,000\text{ EUR}$ (amortized over 10 years $= 22,000\text{ EUR/yr}$).
- Operating expenditure: Pumping electrical power, water recirculation screening, and nozzle maintenance for 45 days: $\text{Opex} = 38,000\text{ EUR/season}$.
- Sprinkling maintains sapwood moisture at $135\%$ ($>120\%$), creating hypoxia and preventing sapstain growth entirely ($x_s = 0\text{ mm}$).
- Net financial benefit:
  $$\text{Net Annual Benefit} = 2,090,000 - (22,000 + 38,000) = \mathbf{2,030,000\text{ EUR}}$$
  **Return on Investment (ROI):** $\frac{2,030,000}{60,000} = 33.8\times$ annual cost.

---

### 6.5 EUDR compliance audit and geospatial polygon due diligence

The enterprise prepares a commercial export shipment of $2,500\text{ m}^3$ of kiln-dried Scots pine sawn timber to an industrial joinery manufacturer in Germany.

#### Cadastral Harvest Block Verification
- Sourced Harvest Block: `SE-AC-2025-C104` (Västerbotten County, Sweden).
- Block Area: $24.8\text{ hectares}$ ($> 4.0\text{ ha}$, therefore requiring a full multi-point polygon).
- WGS84 Geolocation Polygon Coordinates (Ingested into EU TRACES DDS):
  - Vertex 1: $64.128450^\circ\text{ N}, 19.451205^\circ\text{ E}$
  - Vertex 2: $64.134210^\circ\text{ N}, 19.458920^\circ\text{ E}$
  - Vertex 3: $64.131050^\circ\text{ N}, 19.472140^\circ\text{ E}$
  - Vertex 4: $64.125600^\circ\text{ N}, 19.463800^\circ\text{ E}$
  - Vertex 5 (Closing): $64.128450^\circ\text{ N}, 19.451205^\circ\text{ E}$

#### Automated Compliance Audit Checks
1. **Deforestation Cut-Off Date Check:** Sentinel-2 multispectral infrared imagery from June 2020 and August 2020 establishes baseline Normalized Difference Vegetation Index: $NDVI_{\text{baseline}} = 0.81$ (dense conifer forest canopy). Imagery from July 2021 confirms $NDVI_{2021} = 0.82$. Canopy loss was zero as of December 31, 2020.
2. **Forest Degradation Check:** Forest management records confirm the stand was harvested under an approved silvicultural clearfelling plan with 10 retention trees per hectare and immediate artificial regeneration (replanting with Scots pine seedlings within 12 months), satisfying Swedish Forestry Act (*Skogsvårdslagen*) Section 5 and EUDR Article 2(7) non-degradation criteria.
3. **Chain of Custody Custody Log:** Physical segregation was maintained from harvester StanForD head ID `HVR-774`, to forwarder trip ID `FWD-12`, to roadside landing stack `STK-90`, through wet-deck bay `WD-3`, to sawmill shift batch `SM-401`.
4. **EU TRACES Submission:** Due Diligence Statement generated, signed electronically, and assigned EU TRACES Reference ID: `DDS-2025-SE-440310-99281A`. Goods cleared for export.

---

## 7. Code, query, or calculation

The following self-contained Python script implements the backward Dynamic Programming algorithm for optimal log bucking along a measured stem profile. It evaluates product assortments, enforces small-end diameter constraints, computes Smalian volumes, and reports the optimal cut sequence.

```python
import math

def stem_taper(h: float, h_tot: float = 25.0, dbh: float = 38.0) -> float:
    """Predict inside-bark diameter (cm) at height h (m) using power taper."""
    if h >= h_tot:
        return 0.0
    return dbh * ((h_tot - h) / (h_tot - 1.3)) ** 0.75

# Assortment catalog: (name, length_m, trim_m, min_sed_cm, price_eur_per_m3)
ASSORTMENTS = [
    ("Veneer",   5.0, 0.10, 30.0, 125.0),
    ("Sawlog_1", 4.2, 0.10, 22.0,  90.0),
    ("Sawlog_2", 3.6, 0.10, 16.0,  70.0),
    ("Pulpwood", 2.4, 0.10,  8.0,  32.0),
]

def optimize_bucking(h_tot: float = 25.0, dbh: float = 38.0, step: float = 0.1):
    n = int(round(h_tot / step))
    dp = [0.0] * (n + 1)
    choice = [None] * (n + 1)

    for i in range(n - 1, -1, -1):
        h_curr = i * step
        best_val = dp[i + 1]  # Option: advance 1 step (waste/trim)
        best_pick = None
        for name, lg, tr, min_sed, pr in ASSORTMENTS:
            nxt_i = i + int(round((lg + tr) / step))
            if nxt_i <= n:
                d_top = stem_taper(h_curr + lg, h_tot, dbh)
                if d_top >= min_sed:
                    d_base = stem_taper(h_curr, h_tot, dbh)
                    # Smalian volume: pi/8 * (D_base^2 + D_top^2) * Length
                    vol = (math.pi / 8.0) * ((d_base / 100.0) ** 2 + 
                                            (d_top / 100.0) ** 2) * lg
                    val = vol * pr + dp[nxt_i]
                    if val > best_val:
                        best_val = val
                        best_pick = (name, lg, d_base, d_top, vol, vol * pr, nxt_i)
        dp[i] = best_val
        choice[i] = best_pick

    # Trace forward
    cuts, idx = [], 0
    while idx < n:
        c = choice[idx]
        if c is None:
            idx += 1
        else:
            cuts.append(c)
            idx = c[6]
    return dp[0], cuts

if __name__ == "__main__":
    val, logs = optimize_bucking()
    print(f"Optimal Realized Stem Value: {val:.2f} EUR\n" + "-" * 62)
    print(f"{'Product':<10} {'Len(m)':<8} {'Base(cm)':<10} {'SED(cm)':<10} {'Vol(m3)':<10} {'Value(EUR)':<10}")
    for l in logs:
        print(f"{l[0]:<10} {l[1]:<8.1f} {l[2]:<10.1f} {l[3]:<10.1f} {l[4]:<10.3f} {l[5]:<10.2f}")
```

### Execution Output and Validation
```text
Optimal Realized Stem Value: 116.90 EUR
--------------------------------------------------------------
Product    Len(m)   Base(cm)   SED(cm)    Vol(m3)    Value(EUR)
Veneer     5.0      39.6       33.5       0.527      65.87     
Sawlog_1   4.2      33.3       27.9       0.312      28.05     
Sawlog_2   3.6      27.8       22.8       0.183      12.78     
Sawlog_2   3.6      22.7       17.3       0.115      8.05      
Pulpwood   2.4      17.1       13.2       0.044      1.41      
Pulpwood   2.4      13.1       8.7        0.023      0.74      
```

---

## 8. Data, standards, and regulatory context

Forest supply chains operate under intersecting forestry standards, international trade law, telecommunications protocols, and environmental statutes.

### 8.1 Forest Certification Standards (FSC & PEFC)
- **FSC-STD-40-004 V3-1 (Chain of Custody Certification):** Governs custody tracking across wood processors, traders, and manufacturers. Enforces strict input verification, volume bookkeeping, and non-entanglement with controversial sources (FSC-STD-40-005 Controlled Wood).
- **PEFC ST 2002:2020 (Chain of Custody of Forest and Tree Based Products):** The global standard governing PEFC claims. Mandates a documented Due Diligence System (DDS), annual mass balance audits, and supplier risk assessments covering legality, biodiversity conservation, and social welfare.

### 8.2 Statutory International Trade Regulations
- **EU Deforestation Regulation (EUDR, Regulation (EU) 2023/1115):** Binding European Union law entered into force in June 2023. Mandates geolocation polygon tracking for all forest holdings $\ge 4\text{ hectares}$ down to six decimal places. Imposes strict administrative liability: non-compliant goods face confiscation, exclusion from public procurement, and maximum administrative fines of at least $4\%$ of the operator's total annual EU turnover.
- **United States Lacey Act (16 U.S.C. §§ 3371–3378, 2008 Amendments):** Prohibits commerce in wood and plant products harvested in violation of foreign or domestic laws. Requires mandatory import declarations specifying genus, species, value, and country of harvest. Operates under strict criminal liability (felony penalties for knowing violations; misdemeanor for lack of due care).
- **Australian Illegal Logging Prohibition Act 2012 (ILPA):** Mandates due diligence processes prior to importing regulated timber products into Australia, requiring verification of harvest rights and payment of statutory royalties.

### 8.3 Machine Telemetry and Communications: StanForD 2010
Modern computerized CTL forestry machines communicate via **StanForD 2010** (Standard for Forest machine Data and Communication), an open, XML-based messaging standard maintained by Skogforsk (Sweden), Metsäteho (Finland), and global forest equipment OEMs (John Deere, Komatsu Forest, Ponsse):
- **`.prd` (Product Instruction):** Transmits assortment definitions, length/diameter matrices, trim tolerances, and price lists from mill scheduling systems to the harvester's onboard computer.
- **`.stm` (Stem Profile Data):** Records continuous optical caliper diameter measurements every $10\text{ cm}$ along each harvested stem, enabling mill-to-forest yield reconciliation and taper equation recalibration.
- **`.hpr` (Harvested Production Report):** Outputs the exact tally of cut logs by assortment, operator ID, time stamp, and GNSS coordinates, serving as the foundational raw ledger for Chain of Custody audits.

---

## 9. Industry practice and case evidence

### 9.1 Fennoscandian Integrated CTL Supply Chains (Stora Enso & Metsä Group)
In Sweden and Finland, over $98\%$ of industrial roundwood is harvested using Cut-to-Length (CTL) systems. Forest companies operate connected logistics networks:
- **Dynamic Bucking-to-Order:** Harvesting heads run real-time BTO software linked via cellular or satellite modems to central pulp and sawmill ERP systems. When sawmills experience surges in specific dimension orders (e.g., $4.8\text{ m} \times 20\text{ cm}$ framing timber for Japanese export), price modifiers $\lambda_k(t)$ are updated overnight across a fleet of 400+ harvesters, shifting harvest bucking patterns within 24 hours.
- **Spring Breakup GIS Routing:** Forest enterprises integrate soil moisture depth maps derived from airborne LiDAR digital elevation models (DEM) and topographic wetness indices (TWI). Timber stands located on high-bearing coarse gravel ridges are held in reserve and scheduled for harvesting during spring breakup, while deep-peat stands are scheduled strictly during mid-winter frozen periods.

### 9.2 North American Mega-Timber REITs (Weyerhaeuser & Rayonier)
In the US Pacific Northwest and US South, timberland real estate investment trusts (REITs) manage millions of acres under Faustmann-based harvest scheduling models:
- **Faustmann Valuation Under Price Volatility:** Harvest rotation ages are evaluated dynamically using real options analysis. When lumber composite prices spike, REITs accelerate harvest in stands between $24$ and $30$ years of age. During cyclical price troughs, stands are left standing to accumulate biological volume ($V(t)$), effectively utilizing the living forest as low-cost storage.
- **Southern Pine Beetle Wet Deck Operations:** During episodic Southern Pine Beetle (*Dendroctonus frontalis*) outbreaks, millions of tons of infested loblolly pine must be clearcut rapidly to halt beetle spread. Timber companies build continuous wet-deck storage yards covering 20+ hectares to store salvaged sawlogs under water spray for up to 12 months, preventing sapstain and blue stain fungal downgrades while orderly lumber manufacturing absorbs the volume shock.

### 9.3 Brazilian Short-Rotation Eucalyptus (Suzano & Klabin)
Brazilian pulp producers operate the world's most productive commercial forestry supply chains:
- **Ultra-Fast Faustmann Rotations:** Utilizing hybridized, cloned *Eucalyptus grandis* and *E. urophylla*, stands achieve Mean Annual Increments exceeding $45\text{--}55\text{ m}^3/(\text{ha}\cdot\text{yr})$. Due to high tropical biological growth rates and tropical soil conditions, the Faustmann optimal rotation age drops to **$6.0\text{ to }7.0\text{ years}$**.
- **Continuous Roundwood Pipeline:** Unlike boreal systems constrained by spring breakup, tropical eucalyptus harvesting runs 365 days per year. Dedicated high-capacity haul trucks (9-axle "rodotrens" carrying $74\text{-tonne}$ gross weights) transport roundwood directly from harvest blocks to coastal mega-pulp mills producing over $2.5\text{ million tonnes}$ of bleached kraft pulp per year per mill line.

### 9.4 EUDR Compliance Implementation (European Timber Importers)
Following the publication of Regulation (EU) 2023/1115, multinational paper and packaging producers established automated geospatial screening engines:
- **Polygon Ingestion:** Automated APIs ingest GIS boundary shapefiles directly from independent forest owners and national land registries.
- **Automated Cloud Computing Screening:** Platforms cross-reference uploaded polygons against the European Space Agency (ESA) WorldCover dataset and Sentinel-2 satellite archives. Polygons exhibiting $>10\%$ canopy loss between January 2021 and the harvest date are automatically flagged and placed on legal procurement holds, requiring on-site auditor verification before wood delivery tickets can be generated.

---

## 10. Failure modes and red flags

The table below catalogs operational, financial, and compliance failure modes across forest harvest and logistics chains.

| Failure Mode | Early Warning Sign | Root Physical / Operational Mechanism | Supply Chain Consequence | Engineering / Managerial Mitigation |
|---|---|---|---|---|
| **Harvester Head Diameter Calibration Drift** | Systematic discrepancies between harvester tally and sawmill scanner SED ($> 5\text{ mm}$ bias). | Feed roller mechanical wear, debarking spike dulling, or hydraulic caliper sensor drift. | Sawlogs cut under minimum SED are rejected at sawmill, downgrading to pulpwood ($60\%$ value loss). | Mandatory daily mechanical caliper calibration against steel test rings; automated scanner-to-harvester feedback. |
| **Myopic Bucking-to-Local-Value** | Log yard inventory accumulates unsellable lengths; chronic deficit of contracted lengths. | Harvester optimizer cuts highest local stem value without penalizing lengths in market surplus. | Working capital locked in slow-moving lumber; customer order fulfillment failure. | Implement StanForD 2010 BTO price modifiers ($\lambda_k(t)$) linked directly to mill sales order books. |
| **Spring Breakup Road Subgrade Rupture** | Surface ruts $> 10\text{ cm}$; pumping of wet mud slurry through aggregate surfacing. | Heavy truck traffic on saturated subgrade before thawed moisture drains ($u > 0, \sigma' \to 0$). | Spur road catastrophic failure; million-dollar road reconstruction capex; stranded truck fleets. | Enforce rigid $TDD$ thermal closure thresholds; monitor subgrade pore water pressure sensors. |
| **Mill Woodyard Starvation During Breakup** | Yard log inventory declines below 2 weeks of mill burn rate with 3 weeks of thaw remaining. | Inadequate pre-breakup haul surge or unpredicted early onset of warm spring temperatures. | Mill forced to curtail production or shut down ($50,000\text{--}150,000\text{ EUR/day}$ in idle fixed overhead). | Dynamic safety stock calculation incorporating degree-day forecasts; long-term gravel storage yard buffers. |
| **Sapstain / Blue Stain Degradation** | Greyish-blue discoloration radiating from end-grain along wood rays in stored pine logs. | Inadequate water spray coverage in log decks during warm weather ($T > 15^\circ\text{C}, MC < 90\%$). | Appearance-grade sawlogs downgraded to utility lumber or pulp chips ($40\text{--}60\%$ price write-down). | Continuous wet-deck water sprinkling maintaining $MC > 120\%$; strict FIFO inventory rotation. |
| **Radial End-Checking and Log Splitting** | Visible longitudinal cracks extending from log ends toward center. | Rapid moisture loss from cross-cut ends below $FSP$ ($28\%$), causing differential shrinkage. | Severe lumber recovery losses at the sawmill headrig; shortened board lengths ($15\text{--}25\%$ volume waste). | Apply paraffin wax or chemical end-sealants immediately after felling; keep decks under fogging nozzles. |
| **Salvage Log Volume Saturation** | Market stumpage prices collapse; sawmills overwhelmed with fire-scorched or beetle-killed logs. | Landscape-scale forest fire or bark beetle epidemic triggering emergency salvage clearcuts. | Mill sorting yards choked; blue stain and wood-borer insect infestation degrade salvage value. | Surge processing capacity; establish regional wet deck depots; export unpeeled logs under phytosanitary permits. |
| **Non-Conforming Fiber Contamination** | FSC/PEFC internal audit detects uncertified roundwood in certified production run. | Failure of physical segregation at debarker infeed or ledger errors in mass-balance credit accounts. | Revocation of FSC/PEFC Chain of Custody certificate; product recalls; loss of certified green premium. | Color-coded log painting at forest landings; automated RFID or barcode log tagging at scale. |
| **EUDR Geolocation Polygon Rejection** | EU TRACES system returns validation error: overlapping polygons or single points on parcels $\ge 4\text{ ha}$. | Field personnel submitting point coordinates; inaccurate handheld GPS coordinates crossing national boundaries. | Goods blocked at EU customs import border; demurrage charges; statutory compliance investigation. | Automated GIS pre-flight validation; verify polygon closure, WGS84 projection, and satellite canopy history. |
| **Spatial Green-Up Harvest Deadlock** | Forestry company unable to meet annual allowable cut (AAC) due to regulatory clearcut adjacency caps. | Failure to model multi-period spatial adjacency constraints in tactical linear programming models. | Harvesters forced to idle; contracted wood supply shortfalls; expensive equipment relocation. | Integrate Unit Restriction Models (URM) or Area Restriction Models (ARM) in tactical MILP harvest schedules. |

---

## 11. Assumptions and limits

Every quantitative model in forest supply chain management relies on operational and mathematical assumptions. The following ledger bounds their validity.

| Assumption | Analytical Role | Real-World Failure Condition | Boundary Testing & Verification |
|---|---|---|---|
| **Deterministic Faustmann Parameters** | Assumes timber prices $P$, planting costs $C$, and discount rate $r$ remain constant over 30–80 years. | Timber market volatility, real interest rate shocks, carbon tax regimes, and changing labor costs. | Apply stochastic dynamic programming or real options models with Geometric Brownian Motion (GBM) price processes. |
| **Continuous Rotation Invariance** | Assumes identical biological yield curves $V(t)$ across successive rotations in perpetuity. | Soil nutrient depletion from whole-tree harvesting, climate change shifts in species range, and pest epidemics. | Incorporate process-based growth models (e.g., 3-PG) that adjust yield curves for changing soil moisture and temperature. |
| **Axisymmetric Stem Geometry** | Assumes stem cross-sections are circular and taper smoothly without asymmetry. | Stem ovality, wind-induced reaction wood (compression/tension wood), basal butt flare, and sweep. | Use dual-axis caliper laser scanners on harvesters; introduce elliptical and sweep defect deduction penalties. |
| **Independent Product Pricing in Bucking** | Assumes log assortment unit prices ($P_k$) are independent of the quantities produced. | Sawmills face finite market depth; overproducing one dimension crashes the local market price. | Couple harvester bucking algorithms to mill order books via dynamic shadow prices ($\lambda_k(t)$). |
| **Uniform Stefan Frost Penetration** | Assumes soil thermal conductivity and moisture content are homogeneous across the road network. | Spatial variation in road gravel quality, ditch drainage performance, and insulating snow cover depth. | Install subgrade thermistor strings and time-domain reflectometry (TDR) moisture sensors on critical corridors. |
| **Step-Function Moisture Thresholds for Fungi** | Assumes sapstain growth is strictly zero below $FSP$ ($28\%$) and above $120\%$ moisture content. | Fungal strains exhibiting tolerance to mild hypoxia; localized surface drying in wet-deck outer log layers. | Sample sapwood cores across stacked decks; perform microscopic hyphal staining and culture assays. |
| **Linear Mass Balance Recovery in Pulping** | Assumes wood dry matter converts to pulp at a fixed recovery percentage ($\eta_{\text{recovery}}$). | Variations in wood basic density, lignin content, sapwood-to-heartwood ratio, and chemical decay. | Measure bone-dry mass and chip moisture continuously using radiometric or near-infrared (NIR) conveyor gauges. |
| **Sub-Meter GPS Accuracy Under Forest Canopy** | Assumes GNSS receivers on harvesters capture exact boundary vertices without multipath distortion. | Heavy conifer canopy attenuation, wet foliage, and steep valley terrain causing GPS drift of $5\text{--}15\text{ m}$. | Post-process kinematic (PPK) GNSS corrections; cross-reference harvester tracks with high-resolution aerial LiDAR. |

---

## 12. Exercises

### Exercise 1: Faustmann Rotation and Soil Rent Sensitivity (Calculation)
A plantation owner manages loblolly pine (*Pinus taeda*) in Georgia, USA.
- Yield curve: $V(t) = 380 \left(1 - e^{-0.06 t}\right)^{3.0} \text{ m}^3/\text{ha}$.
- Net stumpage price: $P = 45.00\text{ USD/m}^3$.
- Planting cost: $C = 1,200.00\text{ USD/ha}$.
- Discount rate: $r = 5.0\% = 0.05\text{ yr}^{-1}$.
1. Calculate the biological Maximum Sustainable Yield (MSY) rotation age.
2. Calculate the Faustmann Land Expectation Value ($LEV$) at stand ages $t = 20, 22, 24, 26, 28\text{ years}$. Identify the optimal rotation age $T^*$.
3. At $T^*$, verify that the marginal condition $P V'(T^*) = r P V(T^*) + r \cdot LEV(T^*)$ holds.
4. Calculate the financial penalty per hectare of delaying harvest from $T^*$ to $T_{\text{MSY}}$.

### Exercise 2: Kozak Stem Taper and Dynamic Programming Log Bucking (Calculation)
A harvested Douglas-fir stem has total height $H = 22.0\text{ m}$ and breast height diameter $DBH = 34.0\text{ cm}$.
Taper equation: $d(h) = 34.0 \cdot \left(\frac{22.0 - h}{20.7}\right)^{0.80}\text{ cm}$.
Candidate assortments:
- Sawlog A: Length $4.0\text{ m}$ ($+0.1\text{ m}$ trim), Min SED $= 20.0\text{ cm}$, Price $= 85.00\text{ USD/m}^3$.
- Sawlog B: Length $3.0\text{ m}$ ($+0.1\text{ m}$ trim), Min SED $= 14.0\text{ cm}$, Price $= 65.00\text{ USD/m}^3$.
- Pulpwood: Length $2.0\text{ m}$ ($+0.1\text{ m}$ trim), Min SED $= 8.0\text{ cm}$, Price $= 28.00\text{ USD/m}^3$.
1. Using backward dynamic programming with a step size of $\Delta h = 0.1\text{ m}$, evaluate candidate cut sequences starting from $h = 0.0\text{ m}$.
2. Determine the optimal sequence of log assortments, their lengths, SEDs, Smalian volumes, and total stem revenue.
3. Contrast the optimal solution against a greedy strategy that cuts Sawlog A whenever SED allows.

### Exercise 3: Spring Breakup Logistics and Pre-Thaw Sizing (Calculation)
A kraft pulp mill consumes $18,000\text{ tonnes}$ of green softwood chips per week (equivalent to $20,000\text{ m}^3$ of roundwood).
- Freezing degree-days reach $FDD = 1,100\text{ }^\circ\text{C}\cdot\text{days}$.
- The provincial transportation ministry imposes a mandatory 7-week spring road closure on all Class 2 aggregate roads leading to the mill.
- During the ban, secondary roads are restricted to $50\%$ legal axle weight, reducing truck net payload from $34\text{ tonnes}$ to $17\text{ tonnes}$, while freight rates per truck-trip remain fixed at $450.00\text{ USD/trip}$.
- The mill can build pre-thaw buffer inventory in its central yard, where log handling and holding cost is $4.50\text{ USD/m}^3$.
1. Formulate the trade-off between hauling at half-payload during breakup versus surging buffer stock prior to breakup.
2. Determine the cost-minimizing pre-breakup buffer stock volume.
3. Calculate the fleet size (trucks) required to haul the pre-breakup inventory over the 6 weeks of frozen conditions preceding the thaw, assuming each truck completes 1.5 trips per day on a 6-day work week.

### Exercise 4: Sapstain Fungal Growth and Wet Deck Capital Budgeting (Design & Critique)
A softwood sawmill holds $65,000\text{ m}^3$ of premium radiata pine sawlogs ($110.00\text{ USD/m}^3$) in unpaved log decks during spring and summer (average temperature $T = 21^\circ\text{C}$).
- In dry storage, sapstain fungi penetrate radially at $1.15\text{ mm/day}$.
- If penetration exceeds $20\text{ mm}$, the outer board yield is downgraded, destroying $42.00\text{ USD/m}^3$ in market value.
- The average storage dwell time is 40 days.
1. Determine whether logs stored in dry decks will exceed the critical $20\text{ mm}$ defect threshold.
2. Calculate the total expected inventory loss under dry decking.
3. Design a wet-deck sprinkler protection layout for an 8-hectare yard holding $65,000\text{ m}^3$. Specify pump head ($60\text{ m}$), flow rate ($45\text{ m}^3/\text{hr}$ per hectare), operating power ($75\text{ kW}$ pump), water recirculation system, and total seasonal cost (Capex amortized over 8 years + power + maintenance).
4. Compute the Net Present Value ($NPV$) and payback period of the wet-deck system.

### Exercise 5: Tactical Harvest Scheduling with Green-Up Adjacency (MILP Formulation)
A forestry enterprise manages 6 adjacent harvest blocks ($B_1, B_2, \dots, B_6$) over a 3-period planning horizon ($t \in \{1, 2, 3\}$).
- Harvest volume in block $i$ is $V_i$.
- Spatial adjacency matrix $A_{ij} = 1$ if block $i$ and block $j$ share a common boundary, $0$ otherwise.
- Environmental regulations enforce a strict **green-up constraint**: if block $i$ is harvested in period $t$, no adjacent block $j$ ($A_{ij} = 1$) may be harvested in period $t$ or period $t+1$.
- Mill demand requires total harvest volume in each period to lie within $[D_t^{\min}, D_t^{\max}]$.
1. Formulate the decision variables, objective function (maximizing discounted stumpage profit), and complete constraint matrix for this mixed-integer linear program.
2. Explain the difference between the Unit Restriction Model (URM) and Area Restriction Model (ARM).

### Exercise 6: EUDR Geospatial Traceability and Due Diligence System Architecture (Design)
Design the complete end-to-end data pipeline for a European timber importer procuring timber from 150 independent forest owners in Northern Sweden.
- Deliverable: System architecture document and JSON data exchange schema.
- Acceptance criteria: Must specify:
  1. Automated ingestion of StanForD 2010 `.geo` and `.hpr` telemetry files from harvesters.
  2. Polygon validation pipeline (WGS84 projection, self-intersection tests, polygon closure, vertex precision $\ge 6$ decimal places).
  3. Cloud-based satellite verification integration (Copernicus Sentinel-2 and Hansen Global Forest Change API) testing for post-2020 canopy loss.
  4. Generation and transmission of the automated Due Diligence Statement (DDS) to the EU TRACES system.

---

## 13. Self-check answers

### Answer to Exercise 1: Faustmann Rotation and Soil Rent Sensitivity
1. **Biological Maximum Sustainable Yield ($MSY$):**
   $$V(t) = 380 (1 - e^{-0.06 t})^3$$
   $$MAI(t) = \frac{V(t)}{t}$$
   Setting $\frac{d MAI}{dt} = 0 \implies V'(t) \cdot t - V(t) = 0$.
   Solving numerically: $T_{\text{MSY}} = \mathbf{28.7\text{ years}}$, with $V(28.7) = 212.4\text{ m}^3/\text{ha}$ and $MAI(28.7) = 7.40\text{ m}^3/(\text{ha}\cdot\text{yr})$.
2. **Faustmann LEV Calculation ($r = 0.05, P = 45, C = 1,200$):**
   - At $t = 20\text{ yr}$: $V(20) = 130.6\text{ m}^3/\text{ha}$. Net revenue $= 45 \times 130.6 = 5,877\text{ USD}$.
     $$LEV(20) = \frac{5,877 e^{-0.05 \times 20} - 1,200}{1 - e^{-0.05 \times 20}} = \frac{5,877(0.3679) - 1,200}{1 - 0.3679} = \frac{2,162.15 - 1,200}{0.6321} = \mathbf{1,522.15\text{ USD/ha}}$$
   - At $t = 22\text{ yr}$: $V(22) = 152.8\text{ m}^3/\text{ha}$. Net revenue $= 6,876\text{ USD}$.
     $$LEV(22) = \frac{6,876(0.3329) - 1,200}{1 - 0.3329} = \frac{2,289.02 - 1,200}{0.6671} = \mathbf{1,632.47\text{ USD/ha}}$$
   - At $t = 23.4\text{ yr}$ (peak): $V(23.4) = 168.1\text{ m}^3/\text{ha}$.
     $$LEV(23.4) = \mathbf{1,648.80\text{ USD/ha}}$$
   - At $t = 26\text{ yr}$: $V(26) = 194.5\text{ m}^3/\text{ha}$. Net revenue $= 8,752.50\text{ USD}$.
     $$LEV(26) = \frac{8,752.50(0.2725) - 1,200}{1 - 0.2725} = \frac{2,385.06 - 1,200}{0.7275} = \mathbf{1,628.95\text{ USD/ha}}$$
   - At $t = 28\text{ yr}$: $V(28) = 207.1\text{ m}^3/\text{ha}$.
     $$LEV(28) = \mathbf{1,570.12\text{ USD/ha}}$$
   **Optimal Faustmann rotation age:** $T^* = \mathbf{23.4\text{ years}}$.
3. **Verification of Faustmann Marginal Condition at $T^* = 23.4$:**
   - $V'(23.4) = 380 \times 3 (1 - e^{-0.06 \times 23.4})^2 (0.06 e^{-0.06 \times 23.4}) = 10.24\text{ m}^3/(\text{ha}\cdot\text{yr})$.
   - Marginal biological value: $P V'(23.4) = 45 \times 10.24 = 460.80\text{ USD/ha/yr}$.
   - Capital holding cost: $r P V(23.4) = 0.05 \times 45 \times 168.1 = 378.23\text{ USD/ha/yr}$.
   - Site rent: $r \cdot LEV(23.4) = 0.05 \times 1,648.80 = 82.44\text{ USD/ha/yr}$.
   - Total right-hand side: $378.23 + 82.44 = 460.67 \approx 460.80\text{ USD/ha/yr}$. Condition verified.
4. **Delay Penalty:** Delaying harvest from $T^* = 23.4\text{ yr}$ to $T_{\text{MSY}} = 28.7\text{ yr}$ reduces $LEV$ from $1,648.80\text{ USD/ha}$ to $1,542.30\text{ USD/ha}$, a net loss of **$106.50\text{ USD/ha}$** in land asset value.

---

### Answer to Exercise 2: Kozak Stem Taper and Dynamic Programming Bucking
1. **Stem Dimensions:** $H = 22.0\text{ m}, DBH = 34.0\text{ cm}$. Taper: $d(h) = 34.0 \cdot ((22.0 - h)/20.7)^{0.80}$.
   - At $h = 0.0\text{ m}$: $d(0) = 35.7\text{ cm}$.
   - At $h = 4.0\text{ m}$: $d(4.0) = 30.5\text{ cm} \ge 20.0\text{ cm}$ (Sawlog A valid).
   - At $h = 8.1\text{ m}$: $d(8.1) = 24.8\text{ cm} \ge 20.0\text{ cm}$ (Sawlog A valid).
   - At $h = 12.2\text{ m}$: $d(12.2) = 18.7\text{ cm} < 20.0\text{ cm}$ (Sawlog A invalid; Sawlog B valid).
   - At $h = 15.3\text{ m}$: $d(15.3) = 13.9\text{ cm} < 14.0\text{ cm}$ (Sawlog B invalid; Pulpwood valid).
   - At $h = 17.4\text{ m}$: $d(17.4) = 10.4\text{ cm} \ge 8.0\text{ cm}$ (Pulpwood valid).
   - At $h = 19.5\text{ m}$: $d(19.5) = 6.4\text{ cm} < 8.0\text{ cm}$ (Top discard).
2. **Optimal Cut Sequence:**
   - **Log 1:** Sawlog A ($0.0\text{ to }4.0\text{ m}$, cut at $4.1\text{ m}$). Base $= 35.7\text{ cm}$, SED $= 30.5\text{ cm}$.
     $$V_1 = \frac{\pi}{8} [(0.357)^2 + (0.305)^2] \times 4.0 = 0.3463\text{ m}^3 \times 85.00 = \mathbf{29.44\text{ USD}}$$
   - **Log 2:** Sawlog A ($4.1\text{ to }8.1\text{ m}$, cut at $8.2\text{ m}$). Base $= 30.4\text{ cm}$, SED $= 24.8\text{ cm}$.
     $$V_2 = \frac{\pi}{8} [(0.304)^2 + (0.248)^2] \times 4.0 = 0.2418\text{ m}^3 \times 85.00 = \mathbf{20.55\text{ USD}}$$
   - **Log 3:** Sawlog B ($8.2\text{ to }11.2\text{ m}$, cut at $11.3\text{ m}$). Base $= 24.7\text{ cm}$, SED $= 20.1\text{ cm}$.
     $$V_3 = \frac{\pi}{8} [(0.247)^2 + (0.201)^2] \times 3.0 = 0.1200\text{ m}^3 \times 65.00 = \mathbf{7.80\text{ USD}}$$
   - **Log 4:** Sawlog B ($11.3\text{ to }14.3\text{ m}$, cut at $14.4\text{ m}$). Base $= 20.0\text{ cm}$, SED $= 15.2\text{ cm}$.
     $$V_4 = \frac{\pi}{8} [(0.200)^2 + (0.152)^2] \times 3.0 = 0.0743\text{ m}^3 \times 65.00 = \mathbf{4.83\text{ USD}}$$
   - **Log 5:** Pulpwood ($14.4\text{ to }16.4\text{ m}$, cut at $16.5\text{ m}$). Base $= 15.1\text{ cm}$, SED $= 11.9\text{ cm}$.
     $$V_5 = \frac{\pi}{8} [(0.151)^2 + (0.119)^2] \times 2.0 = 0.0291\text{ m}^3 \times 28.00 = \mathbf{0.81\text{ USD}}$$
   - **Log 6:** Pulpwood ($16.5\text{ to }18.5\text{ m}$, cut at $18.6\text{ m}$). Base $= 11.8\text{ cm}$, SED $= 8.1\text{ cm}$.
     $$V_6 = \frac{\pi}{8} [(0.118)^2 + (0.081)^2] \times 2.0 = 0.0161\text{ m}^3 \times 28.00 = \mathbf{0.45\text{ USD}}$$
   - Top residue ($18.6\text{ to }22.0\text{ m}$): Discarded.
   **Total Optimal Stem Revenue:** $29.44 + 20.55 + 7.80 + 4.83 + 0.81 + 0.45 = \mathbf{63.88\text{ USD}}$.
3. **Greedy Comparison:** A greedy heuristic cutting Sawlog A whenever SED $\ge 20\text{ cm}$ attempts a third Sawlog A from $8.2$ to $12.2\text{ m}$, but fails because $d(12.2) = 18.7\text{ cm} < 20.0\text{ cm}$, forcing that entire section into Pulpwood. The greedy approach yields only $53.20\text{ USD}$ (a **$16.7\%$ value loss**).

---

### Answer to Exercise 3: Spring Breakup Logistics and Pre-Thaw Sizing
1. **Trade-off Analysis:**
   - Normal hauling cost per tonne: $\frac{450.00\text{ USD}}{34\text{ t}} = 13.24\text{ USD/tonne}$.
   - Breakup hauling cost per tonne (half-load): $\frac{450.00\text{ USD}}{17\text{ t}} = 26.47\text{ USD/tonne}$.
   - Extra transport cost incurred during breakup: $26.47 - 13.24 = 13.23\text{ USD/tonne}$.
   - Converted to roundwood equivalent ($\rho \approx 0.90\text{ t/m}^3$): $13.23 \times 0.90 = 11.91\text{ USD/m}^3$.
   - Central yard buffer holding cost: $4.50\text{ USD/m}^3$.
   - **Economic Rule:** Because the buffer holding cost ($4.50\text{ USD/m}^3$) is strictly lower than the haulage penalty ($11.91\text{ USD/m}^3$), the mill should **maximize pre-thaw buffer accumulation** and cease all hauling on restricted roads during breakup.
2. **Buffer Stock Sizing:**
   - Mill consumption during 7-week closure:
     $$Q = 7\text{ weeks} \times 20,000\text{ m}^3/\text{week} = \mathbf{140,000\text{ m}^3} \quad (126,000\text{ tonnes})$$
   - Adding a 1-week safety buffer ($20,000\text{ m}^3$) yields total pre-breakup target: **$160,000\text{ m}^3$**.
3. **Pre-Breakup Fleet Sizing:**
   - Total volume to haul over 6 frozen weeks: Regular mill consumption ($6 \times 20,000 = 120,000\text{ m}^3$) plus buffer build ($160,000\text{ m}^3$) $= 280,000\text{ m}^3$ ($252,000\text{ tonnes}$).
   - Weekly haul requirement: $\frac{252,000\text{ t}}{6\text{ weeks}} = 42,000\text{ t/week} = 7,000\text{ t/day}$ (6-day week).
   - Daily truck capacity: $1.5\text{ trips/day} \times 34\text{ t/trip} = 51.0\text{ tonnes/day/truck}$.
   - Fleet required: $\frac{7,000}{51.0} = 137.2 \implies \mathbf{138\text{ trucks}}$.

---

### Answer to Exercise 4: Sapstain Fungal Growth and Wet Deck Capital Budgeting
1. **Penetration Depth in Dry Decks:**
   $$x_s = 40\text{ days} \times 1.15\text{ mm/day} = \mathbf{46.0\text{ mm}}$$
   Because $46.0\text{ mm} > 20.0\text{ mm}$, stored sawlogs significantly exceed the critical defect threshold, causing extensive sapwood discoloration.
2. **Total Dry-Deck Inventory Loss:**
   $$\text{Loss} = 65,000\text{ m}^3 \times 42.00\text{ USD/m}^3 = \mathbf{2,730,000\text{ USD}}$$
3. **Wet-Deck System Design:**
   - Layout: 8 hectares, 80 sprinkler risers with impact nozzles ($0.56\text{ m}^3/\text{hr}$ per nozzle).
   - Total pumping rate: $8\text{ ha} \times 45\text{ m}^3/(\text{hr}\cdot\text{ha}) = 360\text{ m}^3/\text{hr}$.
   - System Capex: $280,000\text{ USD}$ (amortized over 8 years at $6\% = 45,089\text{ USD/yr}$).
   - System Opex (electrical power at $0.12\text{ USD/kWh}$ for $75\text{ kW}$ running $1,200\text{ hours}$ plus maintenance): $10,800 + 15,000 = 25,800\text{ USD/season}$.
   - Total annual cost: $45,089 + 25,800 = \mathbf{70,889\text{ USD/yr}}$.
4. **Financial Justification:**
   - Net Annual Avoided Loss: $2,730,000 - 70,889 = \mathbf{2,659,111\text{ USD/yr}}$.
   - Payback Period: $\frac{280,000}{2,659,111} = 0.105\text{ years} \approx \mathbf{38\text{ operating days}}$. The wet deck pays for itself in its very first season.

---

### Answer to Exercise 5: Tactical Harvest Scheduling with Green-Up Adjacency
1. **MILP Formulation:**
   - Let binary variable $x_{it} = 1$ if block $i$ is harvested in period $t$, $0$ otherwise.
   - Objective: Maximize total discounted profit:
     $$\max \sum_{t=1}^3 \sum_{i=1}^6 \frac{P_i \cdot V_i}{(1 + r)^t} x_{it}$$
   - **Single harvest constraint:** Each block can be harvested at most once:
     $$\sum_{t=1}^3 x_{it} \le 1 \quad \forall i \in \{1, \dots, 6\}$$
   - **Green-up spatial adjacency constraint:** For every pair of adjacent blocks $(i, j)$ with $A_{ij} = 1$:
     $$x_{it} + x_{jt} \le 1 \quad \forall t \in \{1, 2, 3\}$$
     $$x_{it} + x_{j, t+1} \le 1 \quad \forall t \in \{1, 2\}$$
     $$x_{jt} + x_{i, t+1} \le 1 \quad \forall t \in \{1, 2\}$$
   - **Periodic mill volume bounds:**
     $$D_t^{\min} \le \sum_{i=1}^6 V_i \cdot x_{it} \le D_t^{\max} \quad \forall t \in \{1, 2, 3\}$$
   - **Integrality:** $x_{it} \in \{0, 1\}$.
2. **URM vs ARM:**
   - **Unit Restriction Model (URM):** Forbids any two adjacent predetermined harvest units from being harvested simultaneously, regardless of their individual sizes.
   - **Area Restriction Model (ARM):** Allows adjacent units to be harvested concurrently provided the combined contiguous clearcut area does not exceed a statutory maximum footprint (e.g., $40\text{ hectares}$). ARM is more flexible but computationally harder, requiring generalized clique constraints or dynamic flow tracking.

---

### Answer to Exercise 6: EUDR Geospatial Traceability Architecture Rubric
- **Ingestion:** Directly ingests StanForD 2010 `.geo` and `.hpr` files via SFTP/REST API, extracting machine GNSS timestamps and operator credentials.
- **Validation:** Implements Shapely/GEOS library routines to verify:
  1. Coordinate Reference System is WGS84 (EPSG:4326).
  2. Polygon is valid, simple, and self-closing (first coordinate equals last).
  3. Coordinates carry at least 6 decimal places.
  4. Parcel area calculation matches cadastral land title within $\pm 2\%$.
- **Satellite Pipeline:** Connects to Sentinel-2 Harmonized surface reflectance bands (B4, B8) and Global Forest Watch APIs:
  - Computes pre-2021 baseline $NDVI$.
  - Confirms forest cover persisted through December 31, 2020.
  - Flags any polygon with $>5\%$ abrupt canopy loss between Jan 2021 and harvest date.
- **EU TRACES Integration:** Assembles the verified payload into the standard EUDR XML/JSON schema, signs using the enterprise eIDAS digital certificate, and transmits via the EU TRACES NT REST API, capturing the returned unique Due Diligence Statement (DDS) reference number.

---

## 14. Interview questions

### Question 1: How does the Faustmann rotation model differ fundamentally from the single-rotation Fisher model and the biological Maximum Sustainable Yield (MSY) model?
- **What a strong answer contains:** Explains that MSY is purely biological, maximizing physical mean annual increment ($MAI$) without regard to discounting or timber value. Shows that the Fisher model maximizes single-rotation NPV but treats land as worthless after felling. Highlights that Faustmann models land as perpetual capital, proving that optimal harvest age satisfies $P V'(T^*) = r P V(T^*) + r \cdot LEV(T^*)$. Emphasizes that because site land rent ($r \cdot LEV$) is strictly positive, Faustmann rotation age is strictly shorter than both Fisher and MSY rotation ages.
- **What a weak answer reveals:** Confuses financial optimization with biological maturity; believes stands should be harvested when growth ceases; fails to identify land rent as the driving opportunity cost.

### Question 2: In a mechanized cut-to-length (CTL) harvester, why is dynamic programming superior to greedy heuristic bucking?
- **What a strong answer contains:** Identifies that greedy bucking ("bucking for local grade") cuts the most valuable single log immediately from the butt, ignoring the downstream geometric consequences on the remaining stem. Explains that because stems taper non-linearly, a slightly shorter or lower-grade first cut may preserve sufficient diameter at higher heights to allow two high-margin sawlogs rather than one sawlog and a long stretch of low-margin pulpwood. Demonstrates that backward dynamic programming guarantees global value optimality along the stem length.
- **What a weak answer reveals:** Believes harvesters simply cut fixed lengths ordered by the mill; fails to recognize the trade-off between diameter taper, trim allowance, and product grade boundaries.

### Question 3: What is the physical mechanism of spring road breakup, and how does it constrain forest logistics?
- **What a strong answer contains:** Explains that surface thaw creates a saturated, undrained layer of soil trapped above an impermeable frozen subgrade. Describes how heavy truck axle loads generate positive pore water pressures ($u > 0$), reducing effective stress ($\sigma' = \sigma - u$) and collapsing the resilient modulus of the roadbed. Quantifies the impact: statutory 50% axle weight restrictions or complete road bans for 4–8 weeks, forcing mills to build massive pre-breakup roundwood buffer inventories.
- **What a weak answer reveals:** Attributes road damage simply to "mud" or "rain"; unaware of the frozen subgrade drainage barrier or pore pressure dynamics; fails to connect road physics to mill buffer inventory sizing.

### Question 4: How does wet decking preserve sawlog quality during summer storage, and why doesn't water rot the wood?
- **What a strong answer contains:** Details that wood-decay and sapstain fungi (*Ophiostoma* spp.) are obligate aerobes requiring oxygen for respiration. Explains that continuous water sprinkling maintains sapwood moisture content above $120\text{--}150\%$, flooding cell lumens and excluding air. Emphasizes that oxygen diffusion in water is 10,000 times slower than in air, inducing hypoxia/anoxia that halts fungal spore germination and hyphal growth. Mentions that wet decking also prevents radial end-checking by keeping wood above the Fiber Saturation Point ($FSP \approx 28\%$).
- **What a weak answer reveals:** Thinks water "washes away fungal spores" or "keeps the wood cold"; incorrectly believes that water causes immediate rot in logs (confusing saturated roundwood with partially damp processed lumber).

### Question 5: What are the three primary Chain of Custody (CoC) accounting systems under FSC and PEFC, and when must physical segregation be used?
- **What a strong answer contains:** Distinguishes between Physical Separation (segregating certified wood completely from forest to finished product), Percentage System (calculating the proportion of certified fiber in a mixed batch and labeling accordingly), and Credit System (accumulating volume credits from certified inputs to label a subset of output as 100% certified). Notes that Physical Separation is mandatory when selling FSC 100% single-source lines or when mixing with controversial non-certified wood is legally prohibited.
- **What a weak answer reveals:** Equates forest certification with simple invoice paperwork; unable to explain how volume credits or conversion recovery factors operate in a continuous pulp or paper mill.

### Question 6: What are the specific technical requirements for cadastral polygons under the EU Deforestation Regulation (EUDR)?
- **What a strong answer contains:** Cites Regulation (EU) 2023/1115. Explains that for parcels $< 4\text{ ha}$, a single point coordinate with at least 6 decimal places is required; for parcels $\ge 4\text{ ha}$, a closed multi-vertex polygon in WGS84 (EPSG:4326) coordinate system must be provided. Details that coordinates must prove zero deforestation after December 31, 2020 via satellite verification, and must be submitted alongside legal harvest rights under an electronic Due Diligence Statement (DDS) uploaded to EU TRACES.
- **What a weak answer reveals:** Believes an FSC certificate alone exempts a company from EUDR; unaware of the 4-hectare threshold, the 6-decimal-place precision mandate, or the December 31, 2020 cutoff date.

### Question 7: How does a harvester head's measurement error propagate through the timber supply chain?
- **What a strong answer contains:** Explains that length and diameter measurement errors directly distort dynamic programming bucking decisions. A $5\text{ mm}$ negative calibration error in diameter causes the harvester to cut sawlog lengths that actually fail the sawmill's minimum small-end diameter ($SED$) scanner threshold, resulting in automatic downgrading of entire log batches to pulpwood at the mill infeed (a 50–70% price loss). Emphasizes the need for daily manual caliper calibration checks and closed-loop data feedback between mill 3D scanners and harvester telemetry.
- **What a weak answer reveals:** Assumes harvester sensors are permanently accurate; fails to quantify the economic cost of mill-infeed log rejection.

### Question 8: How do spatial green-up constraints affect harvest scheduling economics?
- **What a strong answer contains:** Explains that green-up rules prevent adjacent forest compartments from being harvested until the previous clearcut reaches a mandated height or age threshold (e.g., 5–10 years). Points out that this spatial restriction fragments harvest operations, forces road networks to be built earlier and over longer distances, increases harvesting equipment relocation (lowbed) costs, and lowers the overall Net Present Value of the forest estate compared to unconstrained block scheduling.
- **What a weak answer reveals:** Views clearcutting restrictions solely as an environmental policy without understanding the mixed-integer mathematical complexity or the capital cost of accelerated road construction.

---

## 15. Key terms

- **Area Restriction Model (ARM):** Tactical spatial harvest model permitting adjacent cut blocks provided their total contiguous area does not exceed statutory clearcut caps.
- **Bucking-to-Order (BTO):** Harvester optimization driven by dynamic shadow price penalties reflecting current mill demand and delivery quotas.
- **Bucking-to-Value (BTV):** Harvester optimization maximizing immediate revenue against a static assortment price matrix.
- **Cadastral Polygon:** Closed multi-vertex boundary defining the precise geographic perimeter of a land parcel in WGS84 coordinates.
- **Current Annual Increment (CAI):** The instantaneous growth rate of a forest stand in volume per hectare per year: $dV(t)/dt$.
- **Cut-to-Length (CTL):** Mechanized harvesting system where a two-machine team (harvester and forwarder) fells, delimbs, bucks, and hauls logs to roadside landings.
- **Due Diligence Statement (DDS):** Statutory electronic declaration under EUDR certifying that timber is deforestation-free and legally harvested.
- **EUDR:** European Union Deforestation Regulation (Regulation (EU) 2023/1115) restricting trade in products linked to deforestation.
- **Faustmann Rotation Age ($T^*$):** Financial harvest age that maximizes the Land Expectation Value across an infinite series of rotations.
- **Fiber Saturation Point (FSP):** Wood moisture content ($\approx 28\text{--}30\%$) below which physical shrinkage, cell wall collapse, and radial end-checking begin.
- **Forwarder:** Heavy rubber-tired articulated transporter that carries bucked logs completely off the ground from forest stump to roadside landing.
- **Freezing Degree-Days (FDD):** Cumulative daily average temperatures below $0^\circ\text{C}$, used to predict ground frost depth.
- **Green-up Period:** Mandatory waiting period before a forest parcel adjacent to a recent clearcut can be harvested.
- **Gross Vehicle Weight (GVW):** Total laden mass of a commercial haul truck, governed by statutory road and bridge load ratings.
- **Kozak Taper Equation:** Variable-exponent regression equation predicting stem diameter inside bark at any height along a tree stem.
- **Land Expectation Value (LEV):** Net present value of bare forest land under infinite identical Faustmann rotations.
- **Maximum Sustainable Yield (MSY):** Harvest age maximizing average physical annual volume production, where $CAI = MAI$.
- **Mean Annual Increment (MAI):** Cumulative stand volume divided by stand age: $V(t)/t$.
- **Pore Water Pressure ($u$):** Pressure of groundwater held within soil voids, which increases under road thaw and reduces effective shear strength.
- **Resilient Modulus ($M_r$):** Measure of subgrade soil elasticity and load-bearing capacity under dynamic wheel loading.
- **Sapstain (Blue Stain):** Fungal discoloration of sapwood rays caused by *Ophiostoma* species without structural wood rot.
- **Single-Grip Harvester:** Forestry machine equipped with a processing head that fells, delimbs, measures, and bucks tree stems.
- **Smalian's Formula:** Volumetric formula calculating log volume based on the average cross-sectional area of base and top ends.
- **Small-End Diameter (SED):** Diameter inside bark at the thinnest end of a bucked log segment.
- **StanForD 2010:** Global XML standard governing data communication between forest harvesters and mill information systems.
- **Thawing Degree-Days (TDD):** Cumulative daily average temperatures above $0^\circ\text{C}$, used to predict the onset of spring road breakup.
- **Unit Restriction Model (URM):** Spatial harvest scheduling model strictly forbidding the simultaneous harvest of any two adjacent management blocks.
- **Wet Decking:** Log yard inventory management system deploying continuous water misting to maintain wood saturation and prevent fungal attack.

---

## 16. Sources

- **Faustmann, M. (1849).** "Calculation of the Value Which Forest Land and Immature Stands Possess for Forestry." *Allgemeine Forst- und Jagd-Zeitung*, 15, 441–455. (English translation in *Journal of Forest Economics*, 1(1), 1995, 7–44).
- **Kozak, A. (1988).** "A variable-exponent taper equation." *Canadian Journal of Forest Research*, 18(11), 1363–1368. doi:10.1139/x88-213.
- **Kozak, A. (2004).** "My last words on taper equations." *The Forestry Chronicle*, 80(4), 507–515. doi:10.5558/tfc80507-4.
- **Sessions, J. (1988).** "Making better tree-bucking decisions in the woods: an introduction to optimal bucking." *Journal of Forestry*, 86(10), 43–45.
- **Näsberg, M. (1985).** *Mathematical Programming Models for Optimal Log Bucking*. Linköping Studies in Science and Technology, Dissertation No. 132, Linköping University, Sweden.
- **Rönnqvist, M. (2003).** "Optimization in forestry." *Mathematical Programming*, 97(1–2), 267–284. doi:10.1007/s10107-003-0444-0.
- **Skogforsk (2010).** *StanForD 2010: Standard for Forest machine Data and Communication*. The Forestry Research Institute of Sweden, Uppsala.
- **Forest Stewardship Council (2021).** *FSC Standard for Chain of Custody Certification*. FSC-STD-40-004 V3-1 EN, FSC International Center, Bonn, Germany.
- **PEFC International (2020).** *Chain of Custody of Forest and Tree Based Products - Requirements*. PEFC ST 2002:2020, Programme for the Endorsement of Forest Certification, Geneva, Switzerland.
- **European Parliament and Council of the European Union (2023).** *Regulation (EU) 2023/1115 on the making available on the Union market and the export from the Union of certain commodities and products associated with deforestation and forest degradation*. Official Journal of the European Union, L 150/206.
- **Uzunovic, A., Yang, D. Q., Gignac, M., & Byrne, A. (2008).** "Assessment of sapstain and decay in stored softwood logs: Mechanisms, kinetics, and protection strategies." *Forest Products Journal*, 58(7/8), 12–21.
- **Clark, C. W. (1990).** *Mathematical Bioeconomics: The Optimal Management of Renewable Resources* (2nd ed.). John Wiley & Sons, New York.
- **Pearse, P. H. (1990).** *Introduction to Forestry Economics*. University of British Columbia Press, Vancouver.

---

## 17. Links

### Module Navigation
- **Module Index:** [Module 39 — Mining, Metals, and Forestry Supply Chains](_index.md)
- **Previous Topic:** [Topic 08 — Commodity Pricing, Quotation Periods, and Offtake Contracts](08_commodity-offtake.md)
- **Next Topic:** [Topic 10 — Social Licence, Environment, and Closure: Community, Water, Tailings, and Mine Site Rehabilitation](10_social-environment-closure.md)

### Related Sibling Topics
- [Topic 04 — Grade Blending and Specification Management](04_grade-blending.md)
- [Topic 06 — Pit-to-Port Logistics: Heavy-Haul Rail, Overland Conveyors, Slurry Hydraulics, and Terminal Demurrage](06_pit-to-port-logistics.md)
- [Topic 07 — Stockpiles and Bulk Inventory: Buffer, Blend, Degradation, and Working Capital](07_stockpiles-inventory.md)

### Program Foundations and Electives
- [Module 11 — Network Design Optimization](../../02_methods/11_network_design/_index.md)
- [Module 12 — Inventory Theory and Stochastic Control](../../02_methods/12_inventory_theory/_index.md)
- [Module 15 — Production Planning and Scheduling](../../02_methods/15_production_planning_scheduling/_index.md)
- [Module 38 — Process Industries, Commodities, and Bulk Supply Chains](../38_process_industries_commodities/_index.md)
- [Survey File: Mining, Metals, and Forestry Supply Chains](../../../course/39_mining_metals_forestry.md)
