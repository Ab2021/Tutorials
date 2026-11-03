# Humanitarian Needs Assessment

> Module 35 — Public Sector, Defence, and Humanitarian Logistics
> Topic 05 of 10
> Estimated study time: 3.5 hours
> Prerequisites: Topic 01 (Public Procurement Frameworks), Topic 04 (Strategic Stockpiles), Module 11 (Supply Chain Network Economics), Module 14 (Network Design)
> Version 2026.5

---

## 1. Why this topic exists

In commercial supply chains, demand sensing relies on automated point-of-sale (POS) scanners, electronic data interchange (EDI) purchase orders, enterprise resource planning (ERP) inventory reorder triggers, and longitudinal time-series signals. Inventory planners operate in stable environments where demand variability ($\sigma_D$) is characterized through stationary or seasonal statistical distributions, customer geolocations are fixed, transport infrastructure is maintained, and commercial banking rails settle transactions in milliseconds.

In rapid-onset humanitarian emergencies—triggered by tectonic earthquakes, Category 5 tropical cyclones, catastrophic riverine floods, or armed conflicts—this entire commercial demand architecture collapses instantaneously. Physical telecommunication towers are toppled, optical fiber trunks are severed, municipal registries and census archives are obliterated, road networks are cut by landslides or debris fields, and civilian populations flee their homes without notice. 

In this operational vacuum, humanitarian logisticians face the fatal paradox of disaster response: **the golden 72-hour window demands the immediate mobilization and dispatch of thousands of metric tons of life-saving relief supplies, yet baseline information regarding survivor geolocations, casualty counts, structural damage, water contamination, and nutritional deficits is fractured, contradictory, or completely absent.**

Historically, the absence of quantitative demand assessment frameworks produced catastrophic logistics failures:
1. **The Supply-Push Bullwhip:** In the 2004 Indian Ocean Tsunami and the 2010 Haiti Earthquake, well-intentioned donors inundated international airports with unsolicited bilateral donations (UBDs)—including winter overcoats sent to tropical islands, expired pharmaceuticals, high-heeled footwear, and canned meat sent into non-refrigerated disaster zones. These unsorted consignments choked airport aprons, blocked cargo handling equipment, consumed scarce warehouse floor space, and grounded aircraft carrying trauma surgical kits, water purification units, and heavy engineering gear.
2. **Nutritional and Water Caloric Deficits:** Sizing food aid without accounting for baseline demographics, physical labor requirements, ambient temperatures, or micronutrient balance causes excess mortality. Delivering dry grains without fuel, cooking vessels, or potable water triggers acute diarrheal epidemics among displaced children, turning a logistics supply effort into a vector for public health collapse.
3. **Local Market Annihilation via Inappropriate Modality:** Flooding a disaster zone with imported physical grain when local farmers and merchant networks still possess functional warehouses bankrupts indigenous traders, permanently destroys regional agricultural supply chains, and fosters chronic aid dependency. Conversely, injecting Cash and Voucher Assistance (CVA) in a closed market with severed supply corridors causes runaway hyperinflation, leaving impoverished non-beneficiaries unable to afford basic staples.

Humanitarian logistics is not the distribution of uncoordinated charity; it is a deterministic engineering discipline grounded in multi-sector rapid assessment pipelines, satellite-based geospatial damage estimation, internationally binding humanitarian standards (the Sphere Handbook), multi-echelon pipeline gap sizing, and local market absorption analytics.

This topic teaches the operational mechanics and mathematical models required to convert incomplete disaster intelligence into an executable, cost-optimized, and life-saving humanitarian supply pipeline.

---

## 2. Learning outcomes

By the end of this topic, you can:

1. **Synthesize** rapid crisis intelligence using the Multi-Cluster/Sector Initial Rapid Assessment (MIRA) framework, differentiating Phase 1 Preliminary Scenario Definitions (0–72 hours) from Phase 2 Multi-Sector Field Assessments (Days 3–14).
2. **Model** spatial population displacement and affected population counts by fusing baseline gridded population datasets (WorldPop, LandScan), Synthetic Aperture Radar (SAR) coherence damage proxy maps, and Nighttime Lights (VIIRS NTL) radiance decay.
3. **Predict** directional population displacement flux between disaster origins and host settlements using the Radiation Model for human mobility.
4. **Dimension** gross humanitarian commodity requirements across Water, Sanitation, and Hygiene (WASH), Food Security and Nutrition, and Shelter/Non-Food Items (NFI) strictly adhering to international Sphere Handbook standards.
5. **Engineer** emergency water supply systems, calculating water purification chemical dosages (NaDCC chlorination), bladder tank staging capacity, and water trucking fleet turnaround cycles ($N_{\text{trucks}}$).
6. **Formulate** dynamic multi-period Net Pipeline Gap (NPG) equations that track inventory runout, lead-time buffering, and inbound pipeline shipments to prevent catastrophic pipeline breaks.
7. **Calculate** local market absorption capacity using economic supply price elasticity ($E_s$), trader inventory restocking intervals, and inflation tolerance thresholds to determine the optimal split between Cash and Voucher Assistance (CVA) and physical in-kind commodity pipelines.
8. **Construct** an Emergency Market Mapping and Analysis (EMMA) critical market system diagram that isolates supply chain bottlenecks among producers, wholesalers, retailers, and logistical transport actors.
9. **Audit** humanitarian relief pipelines for unsolicited bilateral donations, airhead Maximum Aircraft on Ground (MOG) saturation, port demurrage spirals, and dietary/cultural commodity mismatches.
10. **Develop** an auditable Python decision engine that processes spatial damage indicators, sizes multi-sector Sphere relief requirements, evaluates market absorption boundaries, and calculates weekly net pipeline inventory deficits.

---

## 3. The core mental model

Humanitarian demand assessment and logistics pipeline sizing operates as a five-stage progressive filter. It transforms noisy, unstructured disaster signals into deterministic daily metric tonnages, transport schedules, and financial procurement envelopes.

```text
====================================================================================================
               HUMANITARIAN NEEDS ASSESSMENT & PIPELINE SIZING ARCHITECTURE
====================================================================================================

 [ TRIGGER: RAPID-ONSET DISASTER / CONFLICT SURGE ]
   • Earthquake Epicenter / Cyclone Track / Flood Extent / Frontline Hostilities
   • Infrastructure Disruption: Power grids, cell towers, bridges, port facilities destroyed
                       │
                       ▼
 [ STAGE 1: GEOSPATIAL & BASELINE POPULATION FUSION (0 - 72 HOURS) ]
   ├─ Pre-Crisis Demographics ──> COD-PS, WorldPop 100m Gridded Density, Census Data
   ├─ Remote Sensing Sensing ───> Sentinel-1 SAR Coherence Loss (Damage Proxy Map - DPM)
   ├─ Nighttime Lights Decay ───> VIIRS Day/Night Band (DNB) Normalized Radiance Reduction
   └─ Spatial Mobility Flux ────> Radiation Model for Displacement Distribution (T_ij)
                       │
                       ▼
 [ STAGE 2: MIRA FIELD ASSESSMENT & SEVERITY STRATIFICATION (DAYS 3 - 14) ]
   ├─ Primary Data Collection ──> Key Informant Interviews (KII), Direct Site Observation
   ├─ Cluster Harmonization ────> WASH, Food Security, Health, Shelter/NFI, Logistics
   └─ Severity Stratification ──> People in Need (PIN) across Severity Tiers (Tiers 1 to 5)
                       │
                       ▼
 [ STAGE 3: SPHERE STANDARDS & GROSS REQUIREMENT SIZING ]
   ├─ WASH Standard ────────────> 15 Liters/person/day potable water, 1 tap / 250 persons
   ├─ Water Trucking Fleet ─────> N_trucks = ceil( V_daily / (C_truck * Trips_per_day) )
   ├─ Food & Nutrition ─────────> 2,100 kcal/person/day (10-12% protein, 17-20% fat balance)
   └─ Shelter & NFI ────────────> 3.5 m² covered living area/person, 2 tarpaulins/household
                       │
                       ▼
 [ STAGE 4: LOCAL MARKET ABSORPTION & MODALITY SPLIT (EMMA / RAM) ]
   ├─ Market System Mapping ────> Wholesaler inventory, transport corridors, trader credit lines
   ├─ Price Elasticity Test ────> Es = (% ΔQs) / (% ΔP); Max tolerable local price inflation
   └─ Modality Allocation ──────> Cash & Voucher Assistance (CVA) vs. Physical In-Kind Aid
                       │
                       ▼
 [ STAGE 5: MULTI-ECHELON NET PIPELINE GAP & REPLENISHMENT DISPATCH ]
   ├─ Net Gap Equation ─────────> NPG(t) = Gross_Req(t) - [Stock(t-1) + Inbound(t) - Loss(t)]
   ├─ Lead Time Capitalization ─> Pipeline_Stock = Demand_Rate * (Lead_Time + Buffer)
   └─ Staging & Airhead Hubs ───> Regional Depots (UNHRD) -> Port of Entry -> Last-Mile FDPs
====================================================================================================
```

### The Causal Information Flow

1. **Baseline Ingestion:** Prior to disaster onset, logistics systems maintain pre-crisis Common Operational Datasets (CODs) covering administrative boundaries (COD-AB) and population statistics (COD-PS). When disaster strikes, spatial boundaries of physical impact are intersected with gridded population models to establish the baseline exposed population ($P_0$).
2. **Remote Damage Proxying:** Because physical ground reconnaissance is blocked during the first 48 hours, satellite Synthetic Aperture Radar (SAR) coherence degradation and optical building damage classifications establish an objective damage severity index ($S \in [0, 1]$). Concurrently, satellite nighttime lights (NTL) radiance drops measure electrical grid collapse and immediate human evacuation flux.
3. **Spatial Mobility Modeling:** Using the Radiation Model of human mobility, logistics planners estimate how displaced populations distribute across neighboring districts and urban hosting centers, avoiding the error of assuming survivors remain statically within destruction epicenters.
4. **MIRA Harmonization:** As inter-agency assessment teams deploy on Day 3, Key Informant Interviews (KII) and Community Group Discussions collect primary indicators using harmonized Humanitarian Exchange Language (HXL) standards. The assessed population is categorized into standardized severity tiers (1: None, 2: Stressed, 3: Severe, 4: Extreme, 5: Catastrophic) to establish the authoritative People in Need (PIN) metric.
5. **Sphere Requirement Dimensioning:** The PIN metric is multiplied by Sphere Handbook humanitarian standards. Survival water needs (15 L/person/day), minimum nutritional energy intake (2,100 kcal/person/day), and minimum covered shelter area (3.5 m²/person) establish the gross humanitarian demand curve over a rolling 30-, 60-, or 90-day planning horizon. Water delivery is operationalized through chemical treatment and water trucking fleet sizing.
6. **Economic Modality Optimization:** Before launching physical procurement of commodities, logistics economists evaluate market functionality using the Emergency Market Mapping and Analysis (EMMA) toolkit. If local commercial supply chains can restock quickly with positive supply elasticity ($E_s$), the agency injects Cash and Voucher Assistance (CVA). If local markets are destroyed, depleted, or physically cut off, the demand must be fulfilled through physical in-kind international supply pipelines.
7. **Dynamic Pipeline Gap Sizing:** Gross in-kind demand is netted against existing national strategic stockpiles, local pre-positioned inventory, and confirmed pipeline shipments. Factoring in multi-modal international freight lead times (airlift: 3–7 days, sea freight: 30–60 days, regional road: 10–20 days), logistics planners execute procurement call-offs to ensure inventory does not hit zero, avoiding a fatal "pipeline break."

---

## 4. Definitions and vocabulary

| Term | Formal Definition | Common Misconception | Metric / Unit / Standard |
|---|---|---|---|
| **MIRA** | Multi-Cluster/Sector Initial Rapid Assessment; an inter-agency process coordinated by UN OCHA to identify strategic humanitarian priorities in the first 72h–14 days. | Believing MIRA collects detailed household-level census data; it relies on purposive community key informants. | OCHA / IASC Standard Process |
| **Sphere Standards** | Globally recognized minimum operational benchmarks for humanitarian response across WASH, Food, Shelter, and Health. | Treating Sphere standards as aspirational targets; they represent non-negotiable legal/ethical survival minimums. | Sphere Handbook (2018 Edition) |
| **People in Need (PIN)** | The subset of an affected population whose life, physical integrity, or basic dignity is threatened without external humanitarian assistance. | Confusing total exposed population with PIN; not all exposed individuals require material assistance. | Integer count of individuals |
| **Damage Proxy Map (DPM)** | Geospatial raster generated from Synthetic Aperture Radar (SAR) phase coherence loss, indicating physical structural collapse. | Assuming DPM detects human casualties; it measures dielectric and physical structural surface changes. | Pixel Coherence Index $\Delta \gamma \in [0, 1]$ |
| **Nighttime Lights (NTL)** | Satellite measurement of nocturnal earth radiance (VIIRS Day/Night Band) used to proxy electrification and human presence. | Assuming zero light means complete abandonment; power grid failure drops NTL even if population remains. | Radiance ($\text{nW}\cdot\text{cm}^{-2}\cdot\text{sr}^{-1}$) |
| **Gross Humanitarian Requirement (GHR)** | The total physical quantity of a commodity required to support the target beneficiary population at Sphere standards over a defined time window. | Failing to multiply unit standards by household size, duration, and operational loss factors. | Metric Tons (MT), $\text{m}^3$, or Unit Kits |
| **Net Pipeline Gap (NPG)** | The net commodity deficit requiring immediate procurement or dispatch after subtracting current on-hand stock and confirmed inbound pipeline shipments. | Confusing gross demand with procurement need; ignoring stock in transit causes massive over-ordering. | Metric Tons (MT) per weekly bucket |
| **Pipeline Break** | An operational failure where available physical inventory at field distribution points reaches zero due to lead time stockouts. | Assuming a pipeline break only occurs if global stock is empty; transport bottlenecks create localized breaks. | Binary state or stockout duration (days) |
| **Pipeline Capitalization** | The total volume of inventory that must remain committed inside the transport pipeline to sustain a constant daily distribution rate. | Believing you only need to fund what is consumed; in-transit inventory must be funded and held simultaneously. | Currency ($) or Stock Volume ($\text{MT} \times \text{Lead Time}$) |
| **EMMA** | Emergency Market Mapping and Analysis; a toolkit evaluating the functionality of critical market systems following a disaster shock. | Viewing EMMA as a macroeconomic survey; it evaluates specific commodity value chains (e.g., wheat flour, CGI sheets). | IRC / Oxfam Standard |
| **Supply Price Elasticity ($E_s$)** | The percentage change in quantity supplied by local commercial traders divided by the percentage change in local market price. | Assuming post-disaster supply is infinitely elastic; physical road cuts make local supply completely inelastic ($E_s \approx 0$). | Dimensionless ratio: $\frac{\% \Delta Q_s}{\% \Delta P}$ |
| **Cash & Voucher Assistance (CVA)** | The direct provision of physical cash, mobile money transfers, or commodity vouchers to crisis-affected households. | Assuming CVA is universally cheaper than in-kind aid; in supply-constrained markets, CVA triggers hyperinflation. | Local Currency / USD per household-month |
| **Minimum Expenditure Basket (MEB)** | The monetary value required by a typical household to meet basic survival needs across food, shelter, water, and energy each month. | Treating MEB as identical to food poverty line; MEB includes non-food survival and hygiene items. | Currency / household / month |
| **Unsolicited Bilateral Donations (UBD)** | Donated relief items sent without official request from the receiving government or humanitarian logistics cluster. | Believing all donations help; UBDs clog airstrips, generate demurrage fees, and divert logistics labor. | Metric Tons of non-priority cargo |
| **Humanitarian Exchange Language (HXL)** | A standardized hashtag-based data convention embedded in spreadsheet headers to automate cross-agency data aggregation. | Mistaking HXL for a software platform; it is a lightweight tagging standard (e.g., `#affected +population`). | OCHA Data Standard |
| **Global Acute Malnutrition (GAM)** | The percentage of children aged 6–59 months suffering from acute malnutrition, measured by WHZ or MUAC. | Believing GAM measures chronic stunting; GAM reflects acute, immediate starvation and disease shock. | Percentage (%) of child cohort |
| **Mid-Upper Arm Circumference (MUAC)** | Rapid field screening measurement of the left upper arm circumference in children aged 6–59 months to diagnose acute malnutrition. | Assuming MUAC requires medical clinic infrastructure; non-specialists can measure MUAC using color-coded plastic strips. | Millimeters (mm); SAM < 115 mm |
| **Non-Food Items (NFI)** | Essential non-edible relief supplies required for survival, thermal protection, and human dignity (blankets, tarps, cooking sets). | Assuming NFIs are standardized across all climates; winterized NFIs differ radically from tropical kits. | Standard Kit Units / Household |
| **Food Consumption Score (FCS)** | Composite score based on dietary diversity, food frequency, and relative nutritional importance of food groups consumed over 7 days. | Treating FCS as a caloric measure; it measures dietary adequacy and nutrient diversity. | Dimensionless score (Poor: 0–21; Borderline: 21.5–35; Acceptable: >35) |
| **Coping Strategies Index (CSI)** | Numerical indicator measuring the frequency and severity of household behavioral adaptations to food shortages. | Assuming all coping strategies have equal weight; skipping meals is weighted far higher than buying cheaper food. | Weighted index score |
| **Maximum Aircraft on Ground (MOG)** | The maximum number of aircraft that can be accommodated simultaneously on an airfield's ramps and aprons for unloading. | Confusing runway length with MOG; a runway can handle landings while the apron is 100% full. | Integer count of aircraft parking slots |
| **Forward Distribution Point (FDP)** | The final physical node in the humanitarian logistics network where relief commodities are transferred directly to beneficiaries. | Assuming FDPs have warehouse storage; FDPs are often open-air temporary points requiring zero overnight stock. | Spatial geocode / Facility node |
| **ASYCUDA Relief Module** | Automated System for Customs Data module that fast-tracks clearance for accredited humanitarian relief consignments. | Believing emergency goods are exempt from documentation; cargo manifests and airway bills are still legally required. | UNCTAD Customs Standard |
| **Free Chlorine Residual (FCR)** | The concentration of available chlorine remaining in water after initial chemical disinfection demand is satisfied. | Assuming chlorine taste means toxic water; FCR protects treated water against recontamination in the home. | Milligrams per Liter (mg/L); target: 0.2–0.5 mg/L |

---

## 5. The mechanics, step by step

### Step 1: Pre-Crisis Demographic Profiling and Spatial Segmentation

Prior to disaster occurrence, logistics analytics systems maintain spatial database layers representing baseline vulnerability. 

1. **Common Operational Datasets (CODs):** Ingest official administrative boundary shapefiles (COD-AB) at Administrative Levels 0 (national), 1 (province/state), 2 (district), and 3 (sub-district). Concurrently, ingest COD-PS (Population Statistics), disaggregated by gender, age brackets (<5 years, 5–17, 18–59, 60+), and average household size ($H_{\text{size}}$).
2. **High-Resolution Gridded Population Fusion:** When administrative census boundaries are too coarse, ingest 100-meter gridded population density rasters from WorldPop or Oak Ridge National Laboratory LandScan. The baseline population $P_0(c)$ in any spatial polygon or administrative zone $c$ is:
   $$P_0(c) = \sum_{i \in c} \rho_i \cdot A_i$$
   where $\rho_i$ is the population density per hectare in grid cell $i$, and $A_i$ is the surface area of cell $i$.
3. **Critical Infrastructure Overlays:** Overlay spatial line and point vector layers representing primary supply corridors, bridges, seaports, airfields (runway length, maximum aircraft gross weight, apron parking capacity), electrical grid transmission nodes, and municipal water treatment plants.

### Step 2: Phase 1 MIRA (0–72 Hours) Satellite Sensing and Displacement Modeling

During the initial 72 hours, ground physical access is obstructed. Logistics planners must estimate affected population counts ($P_{\text{aff}}$) and displaced persons ($D$) using remote sensing indicators.

```text
+-----------------------+     +------------------------+     +------------------------+
| Sentinel-1 SAR Radar  |     | VIIRS Nighttime Lights |     | Mobile CDR Telemetry   |
| Coherence Loss Index  |     | Radiance Decay Ratio   |     | Tower Handoff Drop     |
| Delta Gamma in [0, 1] |     | R_NTL = L_post / L_pre |     | R_CDR = S_post / S_pre |
+-----------+-----------+     +-----------+------------+     +-----------+------------+
            |                             |                              |
            +-----------------------------+------------------------------+
                                          |
                                          v
                         +---------------------------------+
                         | Fused Displacement Coefficient   |
                         |  delta_c = w_d*S + w_l*(1-R_NTL)|
                         +---------------------------------+
                                          |
                                          v
                         +---------------------------------+
                         | Estimated Displaced Population  |
                         |  D(c) = P_0(c) * delta_c        |
                         +---------------------------------+
```

1. **Synthetic Aperture Radar (SAR) Damage Proxy Mapping (DPM):** Radar satellites (such as Sentinel-1 or COSMO-SkyMed) emit microwave pulses that penetrate clouds, dust, and darkness. By comparing the interferometric phase coherence between pre-disaster radar pairs ($\gamma_{\text{pre}}$) and a co-disaster radar pair ($\gamma_{\text{co}}$), remote sensing algorithms compute coherence loss:
   $$\Delta \gamma(i) = \max\Big(0, \, \gamma_{\text{pre}}(i) - \gamma_{\text{co}}(i)\Big)$$
   A structural damage severity index $S_c \in [0, 1]$ for zone $c$ is calculated by integrating severe coherence loss pixels over the urban footprint:
   $$S_c = \frac{1}{|U_c|} \sum_{i \in U_c} \mathbb{I}\Big(\Delta \gamma(i) \ge \tau_{\text{damage}}\Big)$$
   where $U_c$ is the set of urban building footprint pixels in zone $c$, and $\tau_{\text{damage}}$ is the calibrated structural damage threshold (typically $0.25 \le \tau_{\text{damage}} \le 0.40$).
2. **Nighttime Lights (NTL) Radiance Decay:** Nocturnal radiance data captured by the Visible Infrared Imaging Radiometer Suite (VIIRS) Day/Night Band (DNB) measures artificial lighting. For zone $c$, compute the post-disaster to pre-disaster radiance ratio:
   $$R_{\text{NTL}}(c) = \frac{\overline{L}_{\text{post}}(c) + \epsilon}{\overline{L}_{\text{pre}}(c) + \epsilon}$$
   where $\overline{L}(c)$ is the mean cloud-free radiance in $\text{nW}\cdot\text{cm}^{-2}\cdot\text{sr}^{-1}$, and $\epsilon = 0.1$ prevents division by zero. A drop toward zero ($R_{\text{NTL}} \to 0$) indicates electrical grid blackout and structural depopulation.
3. **Mobile Network Operator (MNO) Call Detail Records (CDR):** Where cellular tower generators survive, ingest anonymized cell-site telemetry. The active subscriber ratio is:
   $$R_{\text{CDR}}(c) = \frac{S_{\text{active, post}}(c)}{S_{\text{active, pre}}(c)}$$
4. **Fused Population Displacement Model:** The total displaced population $D(c)$ in zone $c$ is estimated by fusing these structural and behavioral signals:
   $$D(c) = P_0(c) \cdot \delta_c$$
   $$\delta_c = \min\Big(1.0, \, \omega_d \cdot S_c + \omega_l \cdot \max(0, 1 - R_{\text{NTL}}(c)) + \omega_c \cdot \max(0, 1 - R_{\text{CDR}}(c))\Big)$$
   where weights $\omega_d + \omega_l + \omega_c = 1.0$ reflect sensor reliability. In standard satellite operational protocols: $\omega_d = 0.50$, $\omega_l = 0.30$, and $\omega_c = 0.20$.

### Step 3: Directional Displacement Flux Modeling (The Radiation Model)

Displaced populations rarely remain in the rubble of destroyed homes; they evacuate toward host communities, public buildings, or spontaneous camps. To size logistics hubs correctly, planners must predict the directional migration flow $T_{ij}$ from destroyed origin zone $i$ to destination zone $j$.

Using the parameter-free **Radiation Model** (Simini et al., 2012):
$$T_{ij} = D(i) \cdot \frac{m_i \cdot n_j}{(m_i + s_{ij}) \cdot (m_i + n_j + s_{ij})}$$
where:
- $D(i)$ is the total displaced population originating in zone $i$.
- $m_i$ is the baseline population of origin zone $i$.
- $n_j$ is the absorptive capacity or baseline population of destination zone $j$.
- $s_{ij}$ is the total baseline population residing within the circular disc of radius $r_{ij}$ centered at $i$, excluding origin $i$ and destination $j$.

This formulation captures job, shelter, and family absorption opportunities, allowing logistics planners to position Forward Distribution Points (FDPs) in destination zones *before* massive informal camp clusters emerge.

### Step 4: Phase 2 MIRA (Days 3–14) Primary Assessment and Severity Stratification

Once field assessment teams access affected locations, secondary satellite estimates are ground-truthed using harmonized multi-sector assessment instruments.

1. **Sampling Architecture:** In rapid crises, classical simple random sampling (SRS) across households is logistically impossible. MIRA deploys a purposive two-stage stratified cluster sampling design:
   - *Primary Sampling Units (PSUs):* Populated settlements stratified by geospatial damage severity (High, Medium, Low).
   - *Secondary Units:* Key Informant Interviews (KII) conducted with community leaders, medical officers, market traders, and schoolteachers, supplemented by direct visual site observations.
2. **Severity Classification Matrix:** Inter-agency clusters evaluate each administrative unit on a standardized 5-point severity scale established by the Inter-Agency Standing Committee (IASC):
   - *Tier 1 (None/Minimal):* Basic services functional; population coping without external assistance.
   - *Tier 2 (Stressed):* Basic services strained; households reducing non-essential expenditures.
   - *Tier 3 (Severe):* Significant food and water consumption gaps; acute malnutrition spiking; survival assets depleted.
   - *Tier 4 (Extreme):* Extreme food deficits; high water-borne disease incidence; severe acute malnutrition (SAM) exceeding emergency thresholds; excess mortality.
   - *Tier 5 (Catastrophic):* Near-total collapse of survival infrastructure; starvation, destitution, and widespread death.
3. **People in Need (PIN) Aggregation:** The People in Need count in zone $c$ is the sum of populations residing in Tiers 3, 4, and 5:
   $$\text{PIN}(c) = \sum_{k=3}^{5} P_{c, k}$$
   where $P_{c, k}$ is the verified population in severity tier $k$. The target beneficiary population $B(c)$ is derived by applying a coverage factor $\alpha_c \in [0.8, 1.0]$ based on operational mandate and humanitarian access:
   $$B(c) = \alpha_c \cdot \text{PIN}(c)$$

### Step 4b: Nutritional Epidemiology & Therapeutic Pipeline Sizing (SMART Surveys & RUTF)

While MIRA provides initial multi-sector severity scoring, nutritional crises require precise epidemiological surveys executed via the **Standardized Monitoring and Assessment of Relief and Transitions (SMART)** methodology. SMART deploys two-stage cluster sampling measuring anthropometric status in children aged 6–59 months.

1. **Anthropometric Diagnostic Metrics:**
   - **Weight-for-Height Z-score (WHZ):** Measures acute wasting relative to the WHO 2006 Child Growth Standards:
     $$WHZ = \frac{w_i - \mu_{\text{ref}}(h_i)}{\sigma_{\text{ref}}(h_i)}$$
     where $w_i$ is the child's weight, $h_i$ is height/length, and $\mu_{\text{ref}}, \sigma_{\text{ref}}$ are median and standard deviation parameters from the WHO reference population.
   - **Global Acute Malnutrition (GAM):** The proportion of children with $WHZ < -2.0$ or bilateral pitting nutritional edema. A GAM rate $\ge 15.0\%$ constitutes an internationally recognized humanitarian emergency threshold.
   - **Severe Acute Malnutrition (SAM):** The proportion of children with $WHZ < -3.0$, bilateral edema, or Mid-Upper Arm Circumference $\text{MUAC} < 115\text{ mm}$. A SAM rate $\ge 2.0\%$ indicates imminent child mortality risk.
   - **Moderate Acute Malnutrition (MAM):** Children with $-3.0 \le WHZ < -2.0$ or $115\text{ mm} \le \text{MUAC} < 125\text{ mm}$.

2. **Therapeutic Feeding Pipeline Dimensioning:**
   Children diagnosed with SAM without medical complications are treated in community-based Outpatient Therapeutic Programmes (OTP) using **Ready-to-Use Therapeutic Food (RUTF)** (e.g., Plumpy'Nut: lipid-based paste providing $500\text{ kcal}$ per $92\text{ g}$ sachet).
   - The standard clinical treatment protocol spans an average of 6 to 8 weeks, consuming 1 carton of RUTF ($150\text{ sachets} = 13.8\text{ kg}$) per month, totaling $1.5\text{ cartons}$ ($20.7\text{ kg}$) per cured child.
   - Given target under-5 child cohort $P_{<5} \approx 0.18 \cdot P_{\text{displaced}}$, expected SAM prevalence rate $r_{\text{SAM}}$, and OTP programme coverage target $\alpha_{\text{OTP}} \approx 0.70$:
     $$N_{\text{SAM caseload}} = P_{<5} \cdot r_{\text{SAM}} \cdot \alpha_{\text{OTP}}$$
     $$\text{Gross RUTF Cartons} = N_{\text{SAM caseload}} \cdot 1.5 \cdot (1 + \phi_{\text{loss}})$$
     $$\text{Gross RUTF Tonnage} = \frac{\text{Gross RUTF Cartons} \cdot 13.8\text{ kg}}{1,000} \quad [\text{Metric Tons}]$$
   - Children with MAM are enrolled in Targeted Supplementary Feeding Programmes (TSFP) receiving Ready-to-Use Supplementary Food (RUSF) or Supercereal plus vegetable oil ($100\text{ g/child/day}$ for 90 days).

### Step 5: Dimensioning Gross Requirements via Sphere Standards

Humanitarian logisticians convert beneficiary counts $B(c)$ into physical commodity bills-of-materials (BOM) using the internationally binding standards of the Sphere Project.

```text
+---------------------------------------------------------------------------------------------------+
|                                 SPHERE OPERATIONAL BENCHMARKS                                     |
+====================+====================================+=========================================+
| Sector             | Core Survival Standard             | Engineering / Logistics Specification   |
+--------------------+------------------------------------+-----------------------------------------+
| Water (WASH)       | 15.0 Liters / person / day         | Max distance: 500 m; Max queue: 30 min  |
|                    |  • Drinking & cooking: 4–6 L/p/d    | Max 250 persons per tap (0.125 L/s)     |
|                    |  • Personal hygiene: 6 L/p/d       | Free Chlorine Residual: 0.5 mg/L source |
|                    |  • Household cleaning: 3–5 L/p/d   | Aquatabs: 1 tablet (67 mg NaDCC) / 20 L |
+--------------------+------------------------------------+-----------------------------------------+
| Sanitation (WASH)  | 1 latrine per 20 persons           | Max distance: 50 m from shelters        |
|                    | (Emergency phase peak: max 1:50)   | Min distance: 30 m from groundwater     |
+--------------------+------------------------------------+-----------------------------------------+
| Food & Nutrition   | 2,100 kcal / person / day          | Macronutrient energy distribution:      |
|                    |  • Adjusted for cold:              |  • Protein: 10–12% of total kcal        |
|                    |    +100 kcal per 5°C below 20°C    |  • Fat/Lipids: 17–20% of total kcal     |
|                    |  • Adjusted for heavy labor:       |  • Carbohydrates: Balance (~70%)        |
|                    |    +300 to +500 kcal/p/d           | Micronutrients: Vitamin A, Iron, Iodine |
+--------------------+------------------------------------+-----------------------------------------+
| Shelter & NFI      | 3.5 m² covered floor space/person  | Thermal insulation; vector protection   |
|                    | (Cold climates: 4.5–5.5 m²/person) | Standard Kit: 2 Tarps (4x5m) / family   |
|                    | Gross camp area: 45 m²/person      | 1 Kitchen Set, 2 Blankets/p, 2 Jerrycans|
+--------------------+------------------------------------+-----------------------------------------+
```

1. **WASH Gross Dimensioning:** For a target beneficiary population $B$ over planning window $T$ days:
   $$\text{Water Volume } V_{\text{water}} = B \cdot 15.0 \, \frac{\text{L}}{\text{person}\cdot\text{day}} \cdot T \quad [\text{Liters}]$$
   Converting to metric tons of water ($\rho = 1.0 \, \text{MT}/\text{m}^3$):
   $$\text{Mass}_{\text{water}} = \frac{V_{\text{water}}}{1,000} \quad [\text{Metric Tons}]$$
   Chemical water purification tablets (Sodium Dichloroisocyanurate - NaDCC, 67 mg active per 20 L):
   $$\text{Tablets}_{\text{NaDCC}} = \left\lceil \frac{V_{\text{water}}}{20} \right\rceil$$
2. **Water Trucking Fleet Dimensioning:** Delivering bulk water over damaged road networks requires calculating the fleet cycle turnaround time:
   $$T_{\text{cycle}} = T_{\text{fill}} + T_{\text{outbound}} + T_{\text{discharge}} + T_{\text{return}} \quad [\text{minutes}]$$
   For an operational working day $T_{\text{operating}}$ (typically 12 hours $= 720$ minutes), the number of delivery rotations per truck per day is:
   $$N_{\text{trips}} = \left\lfloor \frac{T_{\text{operating}}}{T_{\text{cycle}}} \right\rfloor$$
   Given tanker capacity $C_{\text{truck}}$ (typically $10\text{ m}^3$), the required active fleet size is:
   $$N_{\text{active trucks}} = \left\lceil \frac{V_{\text{daily}}}{C_{\text{truck}} \cdot N_{\text{trips}}} \right\rceil$$
   Accounting for mechanical fleet availability factor $\eta_{\text{avail}} \approx 0.80$:
   $$N_{\text{total fleet}} = \left\lceil \frac{N_{\text{active trucks}}}{\eta_{\text{avail}}} \right\rceil$$
3. **Food Basket Nutritional Dimensioning:** A standard emergency dry ration basket yielding 2,100 kcal/person/day comprises:
   - *Cereal Staple (Rice/Maize flour):* 400 g/p/d ($1,440\text{ kcal}$, $28\text{ g protein}$)
   - *Pulses (Lentils/Split peas):* 60 g/p/d ($204\text{ kcal}$, $15\text{ g protein}$)
   - *Fortified Vegetable Oil:* 30 g/p/d ($265\text{ kcal}$, $30\text{ g fat}$)
   - *Corn-Soya Blend (CSB+ / Supercereal):* 50 g/p/d ($190\text{ kcal}$, $8\text{ g protein}$)
   - *Iodized Salt:* 5 g/p/d ($0\text{ kcal}$)
   - *Total Unit Weight:* $u_{\text{food}} = 0.545\text{ kg/person/day}$; Total Energy: $2,099\text{ kcal/person/day}$.
   The gross food tonnage over $T$ days is:
   $$M_{\text{food}} = \frac{B \cdot 0.545 \cdot T}{1,000} \cdot (1 + \phi_{\text{loss}}) \quad [\text{Metric Tons}]$$
   where $\phi_{\text{loss}} \approx 0.05$ represents handling, spillage, and transit losses (5%).
4. **Shelter & NFI Dimensioning:** Shelter is provisioned at the household unit. For average household size $H_{\text{size}} = 5$:
   $$\text{Households } N_{\text{HH}} = \left\lceil \frac{B}{H_{\text{size}}} \right\rceil$$
   Gross tarpaulins (spec: $4\text{ m} \times 5\text{ m}$ reinforced plastic sheeting compliant with IFRC/ICRC/UNHCR specifications):
   $$\text{Tarpaulins} = 2 \cdot N_{\text{HH}}$$
   Family hygiene kits (soap, sanitary pads, jerrycans, detergent):
   $$\text{Hygiene Kits} = N_{\text{HH}} \cdot \left\lceil \frac{T}{30} \right\rceil$$

### Step 6: Local Market Absorption Capacity and Modality Decision Logic

Shipping physical food and shelter materials across global supply lines incurs massive freight costs and long lead times. Humanitarian logistics economists deploy the Emergency Market Mapping and Analysis (EMMA) methodology to test whether local commercial markets can meet demand via Cash and Voucher Assistance (CVA).

```text
+---------------------------------------------------------------------------------------------------+
|                               MODALITY SELECTION DECISION ENGINE                                  |
+---------------------------------------------------------------------------------------------------+
                                                  │
                                                  ▼
                       +------------------------------------------------------+
                       | Are physical supply corridors & transport functional?|
                       +--------------------------+---------------------------+
                                                  │
                                 YES ─────────────┴───────────── NO ───────> [ PHYSICAL IN-KIND AID ]
                                  │
                                  ▼
                       +------------------------------------------------------+
                       | Are financial transfer mechanisms (banks/mobile) up? |
                       +--------------------------+---------------------------+
                                                  │
                                 YES ─────────────┴───────────── NO ───────> [ PHYSICAL IN-KIND AID ]
                                  │
                                  ▼
                       +------------------------------------------------------+
                       | Can local merchant supply absorb demand surge?       |
                       | Test: Q_humanitarian <= Q_absorb                     |
                       +--------------------------+---------------------------+
                                                  │
                                 YES ─────────────┴───────────── NO
                                  │                              │
                                  ▼                              ▼
                       [ 100% CASH / VOUCHERS ]       [ HYBRID: CASH up to Q_absorb ]
                       [ (Zero In-Kind Freight)]      [ + IN-KIND for Deficit Volume]
```

1. **Market Supply Elasticity Formulation:** Let $Q_0$ be the pre-disaster monthly trading volume of staple commodities in the disaster zone, and $P_0$ be the baseline price per metric ton. The price elasticity of supply $E_s$ is:
   $$E_s = \frac{\Delta Q_s / Q_0}{\Delta P / P_0}$$
2. **Maximum Market Absorption Capacity ($Q_{\text{absorb}}$):** Local merchants can increase supply only if wholesale supply routes remain open and working capital is accessible. If the humanitarian cluster sets a maximum allowable post-disaster price inflation threshold $\pi_{\max} = \frac{\Delta P_{\max}}{P_0}$ (typically $10\%$ to $15\%$ to protect vulnerable host populations), the maximum surge capacity of the market is:
   $$\Delta Q_{\text{surge}} = Q_0 \cdot \Big(E_s \cdot \pi_{\max}\Big)$$
   Accounting for residual commercial demand from the non-displaced host population ($Q_{\text{comm}}$), the net absorption capacity available for humanitarian voucher redemption is:
   $$Q_{\text{absorb}} = \max\Big(0, \, (Q_0 + \Delta Q_{\text{surge}}) - Q_{\text{comm}}\Big)$$
3. **Modality Optimization Split:** For a gross commodity requirement $M_{\text{gross}}$:
   - If $M_{\text{gross}} \le Q_{\text{absorb}}$, allocate **$100\%$ to Cash & Voucher Assistance (CVA)**. Local commercial traders handle logistics; international shipping is zero.
   - If $M_{\text{gross}} > Q_{\text{absorb}}$, inject CVA up to $Q_{\text{absorb}}$, and allocate the remaining volume to **Physical In-Kind Logistics Pipeline**:
     $$M_{\text{in-kind}} = M_{\text{gross}} - Q_{\text{absorb}}$$
     $$M_{\text{cash}} = Q_{\text{absorb}}$$

### Step 7: Multi-Echelon Logistics Pipeline Net Gap Analysis

Physical in-kind requirements must be scheduled across multi-period planning buckets to prevent airport/port bottlenecks and warehouse stockouts.

```text
TIME BUCKETS:       Week 1          Week 2          Week 3          Week 4
Gross Demand:       [ GHR_1 ]       [ GHR_2 ]       [ GHR_3 ]       [ GHR_4 ]
                       ▲               ▲               ▲               ▲
                       │               │               │               │
Opening Stock:      [ S_0   ]       [ S_1   ]       [ S_2   ]       [ S_3   ]
Inbound Pipeline: + [ Arr_1 ]     + [ Arr_2 ]     + [ Arr_3 ]     + [ Arr_4 ]
Loss Allowance:   - [ L_1   ]     - [ L_2   ]     - [ L_3   ]     - [ L_4   ]
                       │               │               │               │
                       ▼               ▼               ▼               ▼
Closing Stock:      [ S_1   ]       [ S_2   ]       [ S_3   ]       [ S_4   ]
Deficit / Net Gap:  [ NPG_1 ]       [ NPG_2 ]       [ NPG_3 ]       [ NPG_4 ]
```

1. **The Dynamic Inventory Balance Equation:** For time bucket $t \in \{1, 2, \dots, T\}$ (where $t$ represents weekly increments):
   $$I_{\text{avail}}(t) = I(t - 1) + Q_{\text{inbound}}(t) - \Lambda(t)$$
   where $I(t-1)$ is closing inventory from the previous period, $Q_{\text{inbound}}(t)$ is confirmed shipment arrivals, and $\Lambda(t)$ is spoilage/loss in transit.
2. **Net Pipeline Gap ($NPG$):** If available stock is insufficient to satisfy the in-kind gross requirement $M_{\text{in-kind}}(t)$:
   $$NPG(t) = \max\Big(0, \, M_{\text{in-kind}}(t) - I_{\text{avail}}(t)\Big)$$
   The closing inventory $I(t)$ carries forward:
   $$I(t) = \max\Big(0, \, I_{\text{avail}}(t) - M_{\text{in-kind}}(t)\Big)$$
3. **Pipeline Stock Capitalization and Lead Time Buffering:** The international pipeline extends across multiple logistical legs:
   $$L_{\text{total}} = L_{\text{procure}} + L_{\text{freight}} + L_{\text{customs}} + L_{\text{hub}} + L_{\text{last-mile}}$$
   To guarantee continuous distributions without pipeline breaks under stochastic lead time variability ($\sigma_L$) and demand variability ($\sigma_D$), the required capitalized pipeline inventory is:
   $$\text{Pipeline Stock} = \overline{D}_{\text{daily}} \cdot L_{\text{total}} + Z_{SL} \cdot \sqrt{\overline{L}_{\text{total}} \cdot \sigma_D^2 + \overline{D}_{\text{daily}}^2 \cdot \sigma_L^2}$$
   where $Z_{SL}$ is the safety service level factor (for humanitarian survival commodities, $Z_{SL} \ge 2.33$, corresponding to a $99\%$ non-stockout probability).

---

## 6. Worked example

### The Scenario: Category 5 Tropical Cyclone "Zephyr"

On October 12, Category 5 Tropical Cyclone "Zephyr" made landfall across three coastal districts in Province Aurelia (Districts Alpha, Beta, Gamma). High-voltage electrical transmission grids were destroyed, bridges along Coastal Highway 1 collapsed, and municipal water pumping stations were inundated by a 4.5-meter storm surge.

### Step 1: Pre-Disaster Baseline Demographics
According to the National Statistics Office COD-PS dataset, the pre-crisis baseline population is:
- **District Alpha:** $P_0(\text{Alpha}) = 180,000$ persons ($36,000$ households, $H_{\text{size}} = 5.0$)
- **District Beta:** $P_0(\text{Beta}) = 120,000$ persons ($24,000$ households, $H_{\text{size}} = 5.0$)
- **District Gamma:** $P_0(\text{Gamma}) = 60,000$ persons ($12,000$ households, $H_{\text{size}} = 5.0$)
- **Total Exposed Population:** $P_0 = 360,000$ persons ($72,000$ households).

### Step 2: Phase 1 Satellite Remote Sensing & Displacement Estimation (Hour 48)
At Hour 48 post-landfall, the Copernicus Emergency Management Service and NASA Black Marble deliver satellite analysis:
- **District Alpha:** Structural damage index $S = 0.70$ (from Sentinel-1 SAR coherence loss); Nighttime lights radiance drop $1 - R_{\text{NTL}} = 0.75$.
- **District Beta:** Structural damage index $S = 0.45$; Nighttime lights radiance drop $1 - R_{\text{NTL}} = 0.50$.
- **District Gamma:** Structural damage index $S = 0.20$; Nighttime lights radiance drop $1 - R_{\text{NTL}} = 0.25$.

Using operational fusion weights ($\omega_d = 0.60$ for structural damage, $\omega_l = 0.40$ for NTL decay; MNO data unavailable due to cell tower battery exhaustion):
$$\delta_{\text{Alpha}} = (0.60 \cdot 0.70) + (0.40 \cdot 0.75) = 0.42 + 0.30 = 0.72$$
$$D(\text{Alpha}) = 180,000 \cdot 0.72 = 129,600 \text{ displaced persons}$$

$$\delta_{\text{Beta}} = (0.60 \cdot 0.45) + (0.40 \cdot 0.50) = 0.27 + 0.20 = 0.47$$
$$D(\text{Beta}) = 120,000 \cdot 0.47 = 56,400 \text{ displaced persons}$$

$$\delta_{\text{Gamma}} = (0.60 \cdot 0.20) + (0.40 \cdot 0.25) = 0.12 + 0.10 = 0.22$$
$$D(\text{Gamma}) = 60,000 \cdot 0.22 = 13,200 \text{ displaced persons}$$

$$\text{Total Displaced Population } D_{\text{total}} = 129,600 + 56,400 + 13,200 = 199,200 \text{ persons}$$
Target displaced households:
$$N_{\text{HH}} = \frac{199,200}{5.0} = 39,840 \text{ households}$$

### Step 3: Sizing 30-Day Emergency Relief Requirements via Sphere Standards
The humanitarian country team establishes a 30-day first-phase response window ($T = 30$ days) for the $199,200$ displaced survivors.

#### 1. WASH Requirement Sizing & Water Trucking Fleet Engineering
- **Daily Potable Water Demand:**
  $$V_{\text{daily}} = 199,200 \cdot 15.0 \text{ L/day} = 2,988,000 \text{ Liters/day} = 2,988 \text{ m}^3\text{/day} = 2,988 \text{ MT/day}$$
- **30-Day Cumulative Potable Water Demand:**
  $$V_{30\text{-day}} = 2,988 \cdot 30 = 89,640 \text{ m}^3 = 89,640 \text{ Metric Tons}$$
- **Point-of-Use Water Purification Tablets (Aquatabs 67mg, 1 tablet per 20 L):**
  $$\text{Tablets/day} = \frac{2,988,000}{20} = 149,400 \text{ tablets/day}$$
  $$\text{Tablets}_{30\text{-day}} = 149,400 \cdot 30 = 4,482,000 \text{ tablets}$$
- **Emergency Water Storage Bladder Tanks:**
  Assuming standard $10\text{ m}^3$ ($10,000\text{ L}$) flexible PVC bladder tanks distributed across planned camp zones, with twice-daily water truck refills:
  $$\text{Bladders Required} = \left\lceil \frac{2,988\text{ m}^3}{2 \times 10\text{ m}^3} \right\rceil = 150 \text{ bladder tanks}$$
- **Water Trucking Fleet Sizing ($N_{\text{trucks}}$):**
  Municipal treated water must be trucked from an inland water treatment works 25 km away over damaged roads.
  - Truck tanker capacity: $C_{\text{truck}} = 10\text{ m}^3$ ($10,000\text{ L}$).
  - Operational working window: 12 hours/day ($720\text{ minutes}$).
  - Turnaround cycle breakdown: Fill ($30\text{ min}$) + Outbound travel ($60\text{ min}$) + Discharge into bladders ($30\text{ min}$) + Return travel ($50\text{ min}$) $= 170\text{ minutes/trip}$.
  - Trips per truck per day: $\lfloor 720 / 170 \rfloor = 4\text{ trips/truck/day}$.
  - Daily delivery capacity per truck: $4 \times 10\text{ m}^3 = 40\text{ m}^3\text{/day}$.
  - Active trucks required: $\lceil 2,988 / 40 \rceil = 75\text{ active tankers}$.
  - Total fleet sizing with $80\%$ mechanical availability factor ($\eta = 0.80$):
    $$N_{\text{total}} = \left\lceil \frac{75}{0.80} \right\rceil = 94\text{ water tanker trucks}$$
  - Daily diesel fuel required (at $45\text{ L}/100\text{km}$ consumption, $200\text{ km/day/truck}$):
    $$\text{Fuel} = 75\text{ active trucks} \times 200\text{ km} \times 0.45\text{ L/km} = 6,750\text{ Liters of diesel/day}$$

#### 2. Food Security & Nutrition Requirement Sizing
Daily emergency ration yielding $2,100\text{ kcal/person/day}$:
- Cereal (Rice): $0.400\text{ kg/p/d}$
- Pulses (Lentils): $0.060\text{ kg/p/d}$
- Fortified Vegetable Oil: $0.030\text{ kg/p/d}$
- Corn-Soya Blend (CSB+): $0.050\text{ kg/p/d}$
- Total Daily Ration Weight: $0.540\text{ kg/person/day}$

**Daily Food Tonnage:**
$$\text{Mass}_{\text{food, daily}} = \frac{199,200 \cdot 0.540\text{ kg}}{1,000} = 107.568 \text{ MT/day}$$

**30-Day Gross Food Requirement:**
$$M_{\text{gross, food}} = 107.568 \cdot 30 = 3,227.04 \text{ Metric Tons}$$
- **Rice:** $199,200 \cdot 0.400 \cdot 30 / 1,000 = 2,390.40\text{ MT}$
- **Lentils:** $199,200 \cdot 0.060 \cdot 30 / 1,000 = 358.56\text{ MT}$
- **Vegetable Oil:** $199,200 \cdot 0.030 \cdot 30 / 1,000 = 179.28\text{ MT}$
- **CSB+:** $199,200 \cdot 0.050 \cdot 30 / 1,000 = 298.80\text{ MT}$
*(Check sum: $2,390.40 + 358.56 + 179.28 + 298.80 = 3,227.04\text{ MT}$)*

#### 3. Shelter and NFI Sizing
- **Sphere Living Area Requirement:** $3.5\text{ m}^2/\text{person} \times 5 = 17.5\text{ m}^2/\text{household}$.
- **Plastic Tarpaulins ($4\text{ m} \times 5\text{ m} = 20\text{ m}^2$):** $2\text{ tarpaulins/HH} \times 39,840\text{ HH} = 79,680\text{ tarpaulins}$.
- **Family Hygiene Kits (1-month duration):** $1\text{ kit/HH} \times 39,840\text{ HH} = 39,840\text{ kits}$.
- **Emergency Sanitation Latrines (Sphere 1:20 ratio):**
  $$\text{Latrines} = \frac{199,200}{20} = 9,960 \text{ latrines (initial emergency trench latrines capped at 1:50 = 3,984)}$$

### Step 4: Local Market Absorption Capacity Analysis (Rice Commodity)
Before ordering $2,390.40\text{ MT}$ of international rice shipments, the logistics cluster conducts an EMMA market assessment in the provincial capital of Aurelia.
- Baseline monthly commercial rice trade: $Q_0 = 2,500\text{ MT/month}$.
- Residual commercial demand from host population ($160,800$ non-displaced persons): $Q_{\text{comm}} = 1,800\text{ MT/month}$.
- Supply price elasticity of commercial wholesalers: $E_s = 0.50$.
- Maximum tolerable local rice price increase: $\pi_{\max} = 12\%$ ($0.12$).

**Maximum Market Surge Capacity:**
$$Q_{\max} = Q_0 \cdot \Big(1 + E_s \cdot \pi_{\max}\Big) = 2,500 \cdot \Big(1 + 0.50 \cdot 0.12\Big) = 2,500 \cdot 1.06 = 2,650 \text{ MT/month}$$

**Trader Capacity Available for Humanitarian Cash/Vouchers:**
$$Q_{\text{absorb}} = Q_{\max} - Q_{\text{comm}} = 2,650 - 1,800 = 850 \text{ Metric Tons}$$

**Modality Split Determination for Rice:**
- **Cash and Voucher Assistance (CVA):** $M_{\text{cash}} = 850.00\text{ MT}$ ($35.56\%$ of total rice requirement).
- **Physical In-Kind International Pipeline:** $M_{\text{in-kind}} = 2,390.40 - 850.00 = 1,540.40\text{ MT}$ ($64.44\%$).

### Step 5: Multi-Week In-Kind Net Pipeline Gap Calculation
The in-kind requirement of $1,540.40\text{ MT}$ is scheduled over 4 weekly operational cycles ($385.10\text{ MT/week}$).
- Regional UN Humanitarian Response Depot (UNHRD) initial pre-positioned stock: $I(0) = 250.0\text{ MT}$.
- Inbound confirmed shipments from international ports:
  - Week 1: $0.0\text{ MT}$ (vessel in transit)
  - Week 2: $300.0\text{ MT}$ (first air charter & fast feeder arrival)
  - Week 3: $500.0\text{ MT}$ (breakbulk ocean shipment discharge)
  - Week 4: $500.0\text{ MT}$ (second ocean shipment discharge)

```text
+===================================================================================================+
|                     DYNAMIC IN-KIND NET PIPELINE GAP SCHEDULE (RICE - MT)                        |
+======+===============+==================+================+=============+==============+===========+
| Week | Opening Stock | Inbound Arrival  | Total Available| Weekly Req  | Closing Stock| Net Gap   |
+======+===============+==================+================+=============+==============+===========+
| 1    | 250.0 MT      |   0.0 MT         | 250.0 MT       | 385.1 MT    |   0.0 MT     | 135.1 MT  |
| 2    |   0.0 MT      | 300.0 MT         | 300.0 MT       | 385.1 MT    |   0.0 MT     |  85.1 MT  |
| 3    |   0.0 MT      | 500.0 MT         | 500.0 MT       | 385.1 MT    | 114.9 MT     |   0.0 MT  |
| 4    | 114.9 MT      | 500.0 MT         | 614.9 MT       | 385.1 MT    | 229.8 MT     |   0.0 MT  |
+======+===============+==================+================+=============+==============+===========+
```

### Analysis of the Operational Result
1. **Week 1 Stockout:** Available opening stock of $250.0\text{ MT}$ is exhausted against the $385.1\text{ MT}$ requirement, creating an immediate first-week pipeline deficit of $135.1\text{ MT}$ ($35.1\%$ ration shortfall). Logistics must authorize emergency military airlift or borrow from national security reserves.
2. **Week 2 Deficit:** With $300.0\text{ MT}$ arriving, available stock remains $85.1\text{ MT}$ below gross weekly demand.
3. **Week 3 Recovery & Capitalization:** The arrival of $500.0\text{ MT}$ fully eliminates the pipeline gap, satisfying the $385.1\text{ MT}$ demand and leaving a positive closing buffer of $114.9\text{ MT}$.
4. **Week 4 Pipeline Health:** A second $500.0\text{ MT}$ arrival expands the closing stock to $229.8\text{ MT}$, establishing an 18-day operating safety buffer entering the second month.

---

## 7. Code, query, or calculation

The following self-contained Python program executes the displacement fusion model, calculates multi-sector Sphere gross requirements, evaluates local market absorption capacity, and simulates the multi-week net pipeline gap.

```python
import numpy as np

def run_humanitarian_sizing():
    # 1. Baseline Population and Geospatial Damage Indicators
    districts = ["Alpha", "Beta", "Gamma"]
    p0 = np.array([180000, 120000, 60000])
    dmg_sar = np.array([0.70, 0.45, 0.20])
    ntl_drop = np.array([0.75, 0.50, 0.25])
    w_dmg, w_ntl = 0.60, 0.40

    # Geospatial Displacement Fusion
    disp_rate = w_dmg * dmg_sar + w_ntl * ntl_drop
    displaced = (p0 * disp_rate).astype(int)
    total_disp = int(np.sum(displaced))
    hh_count = int(np.ceil(total_disp / 5.0))

    # 2. Sphere Sizing (30 Days)
    days = 30
    water_m3 = (total_disp * 15.0 * days) / 1000.0
    food_mt = (total_disp * 0.540 * days) / 1000.0
    rice_mt = (total_disp * 0.400 * days) / 1000.0
    tarpaulins = hh_count * 2

    # 3. Market Absorption Capacity (Rice)
    base_trade, comm_demand, es, max_inf = 2500.0, 1800.0, 0.50, 0.12
    q_max = base_trade * (1.0 + es * max_inf)
    q_absorb = max(0.0, q_max - comm_demand)
    cash_rice_mt = min(rice_mt, q_absorb)
    inkind_rice_mt = rice_mt - cash_rice_mt

    # 4. Multi-Week Net Pipeline Gap for In-Kind Rice
    weeks = 4
    weekly_req = inkind_rice_mt / weeks
    inbound_pipeline = [0.0, 300.0, 500.0, 500.0]
    stock = 250.0  # Opening stock in MT

    pipeline_log = []
    for w in range(weeks):
        arr = inbound_pipeline[w]
        avail = stock + arr
        gap = max(0.0, weekly_req - avail)
        stock = max(0.0, avail - weekly_req)
        pipeline_log.append((w + 1, arr, avail, weekly_req, stock, gap))

    print(f"Displaced Survivors: {total_disp:,} across {hh_count:,} households")
    print(f"Sphere Demands: Water={water_m3:,.0f} m3 | Food={food_mt:,.2f} MT | Tarps={tarpaulins:,}")
    print(f"Market Absorption: {q_absorb:.1f} MT | Cash Share={cash_rice_mt:.1f} MT | In-Kind={inkind_rice_mt:.1f} MT")
    print("--- Dynamic In-Kind Pipeline Simulation ---")
    for w, arr, avail, req, close_s, npg in pipeline_log:
        print(f"Wk {w}: Inbound={arr:5.1f} MT | Avail={avail:5.1f} MT | "
              f"Req={req:5.1f} MT | Close={close_s:5.1f} MT | Gap={npg:5.1f} MT")

if __name__ == "__main__":
    run_humanitarian_sizing()
```

### Verification and Expected Output

When executed in a standard Python 3.10+ environment with NumPy, the script produces:

```text
Displaced Survivors: 199,200 across 39,840 households
Sphere Demands: Water=89,640 m3 | Food=3,227.04 MT | Tarps=79,680
Market Absorption: 850.0 MT | Cash Share=850.0 MT | In-Kind=1540.4 MT
--- Dynamic In-Kind Pipeline Simulation ---
Wk 1: Inbound=  0.0 MT | Avail=250.0 MT | Req=385.1 MT | Close=  0.0 MT | Gap=135.1 MT
Wk 2: Inbound=300.0 MT | Avail=300.0 MT | Req=385.1 MT | Close=  0.0 MT | Gap= 85.1 MT
Wk 3: Inbound=500.0 MT | Avail=500.0 MT | Req=385.1 MT | Close=114.9 MT | Gap=  0.0 MT
Wk 4: Inbound=500.0 MT | Avail=614.9 MT | Req=385.1 MT | Close=229.8 MT | Gap=  0.0 MT
```

The programmatic output validates the hand calculations: exactly $199,200$ displaced persons requiring $89,640\text{ m}^3$ of water and $3,227.04\text{ MT}$ of emergency food rations. The economic market engine allocates $850.0\text{ MT}$ to local cash assistance, leaving $1,540.4\text{ MT}$ for physical supply pipelines. The pipeline simulation demonstrates the exact $135.1\text{ MT}$ Week 1 stockout and subsequent recovery in Week 3.

---

## 8. Data, standards, and regulatory context

Humanitarian demand assessment and logistics pipeline engineering operate within a structured international governance architecture designed to ensure dignity, equity, technical quality, and financial transparency.

```text
====================================================================================================
               HUMANITARIAN STANDARDS, REGULATORY BODIES & DATA ARCHITECTURES
====================================================================================================

 [ TECHNICAL & QUALITY STANDARDS ]
   • The Sphere Project (2018): Universal Minimum Standards in Humanitarian Response
   • Core Humanitarian Standard (CHS) on Quality and Accountability
   • WHO / UNICEF Joint Monitoring Programme (JMP) for Water Supply and Sanitation
   • Codex Alimentarius: Specifications for Ready-to-Use Therapeutic Foods (RUTF) and Fortified Grains

 [ COORDINATION & ASSESSMENT MANDATES ]
   • UN General Assembly Resolution 46/182: Inter-Agency Standing Committee (IASC) Coordination
   • IASC Transformative Agenda: Humanitarian Programme Cycle (HPC) & MIRA Guidelines
   • Global Logistics Cluster (led by UN World Food Programme - WFP)
   • Global Emergency Shelter, WASH, and Food Security Clusters

 [ INTEROPERABILITY & DATA STANDARDS ]
   • UN OCHA Common Operational Datasets (COD-AB: Boundaries; COD-PS: Population Statistics)
   • Humanitarian Exchange Language (HXL): Standardized data tagging for automated exchange
   • OCHA Financial Tracking Service (FTS) & International Aid Transparency Initiative (IATI)
   • Copernicus Emergency Management Service (EMS) & International Charter 'Space and Major Disasters'

 [ DONOR COMPLIANCE & REPAIR FRAMEWORKS ]
   • USAID Bureau for Humanitarian Assistance (BHA) Emergency Application Guidelines
   • European Civil Protection and Humanitarian Aid Operations (DG ECHO) Single Form Framework
   • Good Humanitarian Donorship (GHD) Principles (Un-earmarked, flexible multi-year funding)
====================================================================================================
```

### 1. The Sphere Handbook (2018 Edition)
The Sphere standards represent the global legal and technical baseline for humanitarian action. Governed by the Humanitarian Charter, Protection Principles, and the Core Humanitarian Standard, Sphere codifies binding survival metrics:
- **WASH Standard 2.1 (Water Access):** Minimum 15 liters per person per day for drinking, cooking, and hygiene; maximum 500-meter transit distance; water point density not exceeding 250 persons per tap at a minimum flow rate of 0.125 L/s. Free residual chlorine must be at least 0.5 mg/L at source after 30 minutes contact time and 0.2 mg/L at point of consumption.
- **WASH Standard 3.1 (Excreta Management):** Minimum 1 latrine per 20 persons in settled camps (emergency peak capped at 1:50); maximum 50 meters distance from dwellings; minimum 30 meters distance from groundwater sources, with pit base sitting at least 1.5 meters above seasonal high water table.
- **Food Security Standard 2.1 (Nutritional Requirements):** 2,100 kcal per person per day baseline intake. Protein must provide 10% to 12% of total energy; fat must provide 17% to 20%. Rations must contain adequate micronutrients to prevent outbreaks of scurvy (Vitamin C), pellagra (Niacin), beriberi (Thiamine), and xerophthalmia (Vitamin A).
- **Shelter Standard 3.1 (Covered Living Space):** Minimum 3.5 m² of covered living area per person in warm climates, excluding external cooking, verandas, and sanitation. In cold climates or urban settings where cooking occurs inside, the standard increases to 4.5–5.5 m² per person. Total planned settlement area must provide 45 m² per person, incorporating access roads, clinics, schools, sanitation buffers, and distribution nodes.

### 2. Common Operational Datasets (CODs) and HXL
UN OCHA mandates that all inter-agency assessments utilize official Common Operational Datasets:
- **COD-AB (Administrative Boundaries):** Canonical geographic boundary polygons with standardized P-codes (Place Codes). Every assessment record must bind to an authoritative P-code to prevent geographic ambiguity.
- **COD-PS (Population Statistics):** Official baseline population figures agreed upon by national authorities and the humanitarian country team.
- **Humanitarian Exchange Language (HXL):** An international data standard adding human- and machine-readable hashtags to spreadsheet headers (e.g., `#affected +population`, `#adm1 +code`, `#indicator +water +liters`). HXL eliminates schema reconciliation friction during multi-cluster pipeline consolidation.

### 3. Donor Regulations: USAID/BHA and DG ECHO
- **USAID Bureau for Humanitarian Assistance (BHA):** Requires implementing partners to submit detailed Procurement Plans, Pipeline Health Summaries, and Bellmon Amendment market impact analyses. The Bellmon Amendment prohibits the distribution of US in-kind food commodities if the distribution will create a substantial disincentive to local domestic production or disrupt local commercial marketing.
- **DG ECHO (European Commission):** Requires adherence to the "Cash by Default" policy under its Common Assessment Framework. Implementing partners must provide quantitative economic evidence (such as EMMA market assessments) to justify physical in-kind procurement over cash transfers.

### 4. The Humanitarian Programme Cycle (HPC) & Appeal Funding Calendars
Inter-agency assessment data directly drives statutory UN funding appeals under the IASC Transformative Agenda:
- **Flash Appeal (Hours 72–120):** Formulated by the UN Resident/Humanitarian Coordinator (RC/HC) and cluster lead agencies immediately following MIRA Phase 1. Sizing is preliminary and covers immediate 3- to 6-month survival requirements.
- **Central Emergency Response Fund (CERF) Rapid Response Grants:** UN OCHA releases un-earmarked emergency cash allocations ($10M–$25M) within 48–72 hours of sudden-onset disasters based on preliminary MIRA displacement figures. These funds capitalize the emergency pipeline, financing immediate air charters and UNHRD stockpile call-offs.
- **Revised Flash Appeal / Humanitarian Response Plan (Day 30):** Incorporates MIRA Phase 2 primary field assessment data, confirmed census counts, and detailed EMMA market capacity findings, converting preliminary gross estimates into verified project envelopes.

### 5. International Customs Treaties: ASYCUDA Relief & Istanbul Convention
Emergency pipelines frequently stall at national borders due to tariff, sanitary, and fumigation inspections. International law provides legal instruments to bypass standard customs delays:
- **WCO Istanbul Convention (Annex B.9):** The World Customs Organization Convention on Temporary Admission establishes mandatory duty-free, tax-free temporary admission for humanitarian relief consignments, telecommunications equipment, and medical field hospitals without requiring cash security deposits.
- **UN OCHA / WCO Model Customs Agreement:** Codifies pre-disaster bilateral protocols where national governments agree to grant immediate electronic pre-arrival clearance (via the UNCTAD ASYCUDA Relief Module), waive import licensing for certified UN/NGO relief consignments, and permit physical cargo release prior to final customs declaration processing.

---

## 9. Industry practice and case evidence

### Case 1: 2010 Haiti Earthquake — Airport Paralysis and Unsolicited Donations
Following the 7.0-magnitude earthquake in Port-au-Prince on January 12, 2010, the humanitarian logistics pipeline suffered one of the most severe logistical collapses in modern history.

```text
+---------------------------------------------------------------------------------------------------+
|                        HAITI 2010 AIRPORT GRIDLOCK VS DEMAND COLLAPSE                             |
+---------------------------------------------------------------------------------------------------+
  Runway Capacity: 1 Single 3,040 m Runway (Toussaint Louverture International Airport - MTPP)
  Pre-Disaster Air Traffic: ~25 commercial flights / day
  Post-Disaster Flight Requests: >200 cargo & military flights / day
  
  [ UNREGULATED SUPPLY-PUSH FLIGHTS ] ───> Choked 10 available aircraft parking spots
  [ UNSORTED DONATIONS (UBDs) ] ─────────> Choked ramp space; cargo handlers spent 14 hours/day
                                           sorting non-essential clothing and expired drugs
  [ CONSEQUENCE ] ───────────────────────> Dozens of life-saving medical & water purification flights
                                           diverted to Santo Domingo (Dominican Republic), adding
                                           48 hours of transit time over damaged mountain roads
+---------------------------------------------------------------------------------------------------+
```

- **Documented Fact:** The single runway and ten-slot apron at Port-au-Prince airport became saturated within 36 hours. Over 1,000 flight clearance requests backed up. Flights carrying US Air Force Special Tactics airfield management teams, surgical field hospitals, and water purification units were forced into holding patterns or diverted to Santo Domingo.
- **Failure Mechanism:** The absence of a centralized Phase 1 MIRA demand assessment resulted in an uncoordinated "supply-push" wave. International NGOs and bilateral donors shipped unsorted clothing, frozen foods requiring cold chains that did not exist, and incompatible medical supplies. 
- **Lessons Learned:** This disaster catalyzed the formalization of the **Global Logistics Cluster** flight slot-booking system and the mandate that military air traffic control must enforce strict landing prioritization tied directly to validated cluster demand.

### Case 2: 2013 Typhoon Haiyan (Yolanda), Philippines — Multi-Island Logistics Bottlenecks
On November 8, 2013, Super Typhoon Haiyan struck the Eastern Visayas with 315 km/h sustained winds and a 5-meter storm surge, destroying Tacloban City and severing transport links across 40 provinces.

- **Documented Fact:** The MIRA multi-sector assessment was deployed across 9 provinces within 72 hours. Initial satellite radar and optical imagery rapidly identified that Tacloban airport runway was intact but the terminal, fuel tanks, and communication systems were annihilated. Coastal Highway 1 was blocked by capsized ships, downburst timber, and severed power lines.
- **Logistics Architecture:** The Logistics Cluster established a multi-hub staging architecture. Sea-lift and C-130 airbridge corridors originated from national staging hubs in Manila and Cebu, funneled into regional forward staging bases in Tacloban, Guiuan, and Ormoc, and transferred to landing craft mechanized (LCM) barges for island-hopping last-mile distribution.
- **Synthesis:** Haiyan demonstrated the successful operational integration of satellite Damage Proxy Maps to identify functional airheads, combined with immediate Sphere WASH deployment: the cluster trucked mobile water treatment units (WTUs) to treat seawater and river sources directly in Tacloban, delivering 15 L/person/day within 5 days.

### Case 3: 2015 Nepal Earthquake — EMMA Market Recovery and Cash Deployment
On April 25, 2015, a 7.8-magnitude earthquake devastated central Nepal, destroying over 600,000 houses across rugged Himalayan topography.

- **Documented Fact:** In Kathmandu Valley and lower foothills, the Emergency Market Mapping and Analysis (EMMA) assessment executed by Oxfam and the Cash Learning Partnership (CALP) revealed that while road cuts isolated high mountain villages, local wholesalers and urban supply corridors in the Terai plains remained fully functional with high supply elasticity ($E_s \ge 0.80$).
- **Operational Execution:** Humanitarian agencies implemented a dual-modality strategy. For inaccessible mountain zones (Gorkha, Langtang), helicopters and porter caravans delivered physical in-kind corrugated galvanized iron (CGI) sheets and high-energy biscuits. In accessible valleys and urban centers, agencies distributed unconditional cash transfers totaling over $35 million.
- **Outcome:** The cash injections stimulated local brick kilns and hardware merchants, preventing price gouging and accelerating permanent housing reconstruction 40% faster than in-kind aid pipelines could achieve.

### Case 4: 2022 Pakistan Floods — Mega-Inundation and Rail System Severing
From June to August 2022, unprecedented monsoon downpours and glacial lake outburst floods inundated one-third of Pakistan, affecting 33 million people and destroying 2.2 million homes.

- **Documented Fact:** Over 9,000 kilometers of road networks and 45 bridges were washed away, severing the primary north-south rail corridor (ML-1) linking Karachi seaport to Punjab and Khyber Pakhtunkhwa.
- **Logistics Innovation:** UN OCHA and the Logistics Cluster utilized Sentinel-1 SAR flood extent mapping updated every 48 hours to dynamically redirect supply pipelines. Because road and rail were paralyzed, the humanitarian pipeline mobilized an amphibious last-mile operation: deploying over 400 flat-bottom motorized boats, airlifting mobile water purification systems, and establishing forward floating distribution hubs in flooded Sindh province.

### Case 5: 2023 Türkiye-Syria Earthquake — Satellite-Driven Displacement Sensing
On February 6, 2023, twin 7.8 and 7.5 magnitude earthquakes destroyed over 200,000 buildings across 11 provinces in southern Türkiye and northern Syria.

- **Documented Fact:** The Copernicus Emergency Management Service and UNOSAT produced building damage proxy maps using Sentinel-1 SAR interferometry and optical high-resolution VHR sensors within 48 hours. By intersecting SAR coherence loss with WorldPop 100-meter gridded population datasets, UN OCHA and AFAD accurately estimated that over 2.7 million individuals had been displaced before ground assessment teams could survey rubble-blocked streets in Antakya and Kahramanmaraş.
- **Operational Impact:** This satellite-derived demand sensing immediately dimensioned the winterized shelter pipeline: calculating the need for 250,000 thermal family tents, 1.5 million winterized blankets, and mobile container latrines, allowing international airbridges to dispatch pre-sized relief kits directly to Adana and Gaziantep airports without waiting for week-long ground census counts.

---

## 10. Failure modes and red flags

| Failure Mode | Early Warning Sign | Root Mechanism | Operational Consequence | Engineering Mitigation |
|---|---|---|---|---|
| **Supply-Push Dump (UBDs)** | Cargo aircraft landing without slot authorization or cluster consignee documentation. | Donor domestic political pressure to show immediate action without validating cluster demand. | Airport apron gridlock; demurrage fees; logistics labor diverted to sort garbage. | Mandatory Slot Allocation Systems; legal rejection and customs seizure of unmanifested humanitarian cargo. |
| **Caloric / Energy Deficit** | Displaced population reporting weakness; child MUAC indicators deteriorating. | Using dry grain mass without calculating cooking fuel, milling requirements, or macronutrient balance. | Acute malnutrition (GAM/SAM) spikes despite high physical tonnage delivered. | Standardize ration baskets to Sphere 2,100 kcal standard with balanced 10–12% protein and 17–20% lipids. |
| **Market Hyperinflation** | Local market staple prices jumping >20% within 48 hours of cash voucher distribution. | Injecting Cash & Voucher Assistance into an inelastic or physically severed local market ($E_s \approx 0$). | Host community impoverished; cash purchasing power collapses; vendor hoarding. | Conduct EMMA supply elasticity assessments; cap cash injections at $Q_{\text{absorb}}$; deploy in-kind aid for deficit. |
| **Satellite Cloud / Radar Blindspots** | Zero damage reported in steep mountain valleys or monsoon zones. | Optical satellite sensors occluded by 100% cloud cover; radar layover/shadowing in steep terrain. | Critical mountain populations abandoned during the first 72 hours. | Switch to L-band Synthetic Aperture Radar (SAR); deploy low-altitude UAVs and key informant radio networks. |
| **Baseline Census Obsoletion** | Relief supply deliveries fall 50% short of arriving crowds at distribution points. | Using 10-year-old static census data that misses rapid peri-urban migration or seasonal labor flows. | Rioting at distribution sites; security forces firing tear gas; stockouts of critical supplies. | Calibrate baseline census with recent WorldPop gridded models and pre-crisis cell phone density maps. |
| **Pipeline Break Stockout** | Field warehouse inventory reaches zero; distribution centers turn away families. | Underestimating multi-modal international freight and customs clearance lead times ($L_{\text{total}}$). | Morbidity and mortality rates spike; loss of community trust; operational shutdown. | Capitalize pipeline with safety stock ($Z_{SL} \ge 2.33$); establish advance contracts with regional UNHRD depots. |
| **Port / Airhead Congestion** | Ocean containers waiting 25+ days for berth space; cargo aircraft holding in air. | Inflow of relief cargo exceeds Maximum Aircraft on Ground (MOG) or port container handling capacity. | Vessel demurrage penalties ($10,000–$50,000/day); supply pipeline paralyzed at entry gate. | Impose port throughput constraints ($C_{\text{port}}$) in optimization models; divert secondary flows to secondary ports. |
| **Cultural Dietary Incompatibility** | Distributed canned food or whole grains dumped or traded on black market for pennies. | Procuring commodities violating religious restrictions (pork) or requiring unfamiliar preparation. | Complete nutritional failure; massive financial waste; local community resentment. | Align relief procurement strictly with cluster-approved national dietary habits and cultural standards. |
| **Sanitation Groundwater Contamination** | Cholera or acute watery diarrhea (AWD) outbreak in displaced settlement within 14 days. | Emergency pit latrines dug within 30 meters of shallow hand-dug wells or above water tables. | Rapid epidemic spread through camp water supply; high mortality among children under 5. | Enforce Sphere WASH rules: latrines $\ge 30\text{ m}$ from water sources and $\ge 1.5\text{ m}$ above groundwater table. |
| **Double Counting & Ghost Demand** | Sum of cluster agency funding appeals exceeds total regional population by 200%. | Competing NGOs assessing identical displacement camps and publishing separate uncoordinated appeals. | Donor fatigue; misallocation of capital away from truly underserved geographic districts. | Enforce OCHA Humanitarian Exchange Language (HXL) tagging and centralized inter-agency MIRA registry. |
| **Fuel Pipeline Starvation** | Water treatment pumps, generator-backed vaccine fridges, and delivery trucks halting. | Focusing entirely on food/shelter cargo without contracting commercial diesel supply lines. | Total paralysis of all downstream logistics operations within 7 days of disaster shock. | Treat diesel fuel as a mandatory Tier-1 relief commodity; calculate and pre-position bulk fuel bladders. |
| **Airhead Cubing-Out** | Aircraft cargo holds reaching 100% volumetric capacity while carrying only 30% payload weight. | Shipping bulky, low-density NFIs (mosquito nets, plastic jerrycans) on weight-chartered flights. | Massive air charter freight expenditure; underutilization of expensive aircraft payload capacity. | Enforce IATA volumetric ratio guidelines ($1\text{ m}^3 = 167\text{ kg}$); pack dense cargo (cereal) with bulky NFIs. |
| **Secondary Market Diversion** | Relief-branded vegetable oil and CSB+ appearing in commercial markets 100 km away. | Unmonitored bulk distributions without biometric verification or household ration cards. | Aid captured by warlords, corrupt officials, or commercial syndicates; intended beneficiaries starve. | Deploy biometric registration (UNHCR BIMS / WFP SCOPE) and serialized barcode ration voucher scanning. |

---

## 11. Assumptions and limits

| Assumption / Boundary | Operational Rationale | Real-World Failure Condition | Diagnostic & Stress Test |
|---|---|---|---|
| **Constant Household Size ($H_{\text{size}} = 5$)** | Simplifies rapid bulk procurement of standardized family kits (tarpaulins, kitchen sets, hygiene packs). | Family fragmentation occurs during panic; unattached orphans and single elderly individuals receive inadequate aid. | Sample household size variance during Phase 2 MIRA; deploy individual-sized contingency kits (10% of total). |
| **Linear Supply Elasticity ($E_s$)** | Assumes local merchant supply response scales proportionally with price increases up to $\pi_{\max}$. | Road closures, bridge collapses, or supplier credit freezes make supply completely inelastic ($E_s = 0$). | Conduct weekly trader inventory and supply-chain corridor checks; verify fuel availability for merchant trucks. |
| **Homogeneous Vulnerability** | Assumes all displaced individuals in a designated severity tier experience identical survival deficits. | Marginalized ethnic minorities, pregnant women, and disabled individuals suffer acute disparities. | Stratify MIRA Key Informant Interviews by protection and gender; cross-tabulate with vulnerability indicators. |
| **Sphere Caloric Invariance** | 2,100 kcal baseline assumed sufficient for average individual survival across all environments. | Sub-zero ambient temperatures dramatically increase metabolic thermogenesis requirements. | Calculate temperature adjustment: add $100\text{ kcal/person/day}$ for every $5^\circ\text{C}$ drop below $20^\circ\text{C}$. |
| **Stable Cellular Infrastructure** | Mobile Call Detail Records (CDR) assumed available to monitor population displacement flux. | Physical cell tower destruction, fiber line severing, and backup diesel generator theft. | Verify cell-tower ping density before trusting CDR telemetry; cross-validate with satellite NTL radiance data. |
| **Frictionless Customs Clearance** | Models assume humanitarian relief flights and cargo containers clear borders within 24–48 hours. | National authorities impose bureaucratic import permits, radiation inspections, or fumigation requirements. | Pre-negotiate ASYCUDA humanitarian customs fast-track protocols and bilateral SOFA/MOU exemptions. |
| **Continuous Ground Water Recharge** | Sphere WASH assumes borehole drilling or river trucking can sustain 15 L/person/day indefinitely. | Drought conditions, seasonal dry riverbeds, or saline aquifer intrusion deplete water sources. | Conduct hydrogeological yield testing; install water meters on boreholes; budget for long-distance water tankering. |
| **Zero Secondary Migration** | Models assume displaced populations remain static in initial concentration camps during the 30-day window. | Secondary displacement occurs due to aftershocks, security threats, or rumors of aid distribution elsewhere. | Track mobile CDR trends weekly; deploy rapid visual UAV sweeps to detect spontaneous camp formations. |
| **Uniform Household Cooking Fuel** | Sizing dry rations assumes beneficiary households possess cooking stoves, firewood, and cooking pots. | In flood or urban rubble zones, zero cooking fuel exists; families cannot boil dry beans or raw rice. | Deploy Ready-to-Eat (RTE) rations or High Energy Biscuits (HEB) during Days 1–7; distribute fuel briquettes. |
| **Trader Credit Line Stability** | Assumes wholesalers can secure commercial bank credit to purchase surge inventory from capital cities. | Local banking liquidity freezes, cash runs deplete ATMs, and suppliers demand physical cash on delivery. | Conduct EMMA financial service provider (FSP) audits; provide emergency liquidity credit guarantees to key wholesalers. |
| **Infinite Airfield Fuel Availability** | Assumes international cargo aircraft can refuel at the disaster airhead for return flights. | Airfield jet fuel (Jet-A1) tanks are contaminated, destroyed, or reserved exclusively for search-and-rescue helicopters. | Require incoming cargo aircraft to "tanker" sufficient fuel for round-trip return flights without refuel. |

---

## 12. Exercises

### Exercise 1: Multi-District Satellite Displacement Fusion
A 7.5-magnitude earthquake strikes Province Solaria. The baseline population of District North is $240,000$ persons ($48,000$ households). At Hour 36, satellite analysis reveals:
- Synthetic Aperture Radar (SAR) structural damage severity index: $S = 0.65$.
- VIIRS Nighttime Lights post-to-pre radiance ratio: $R_{\text{NTL}} = 0.30$.
- Mobile CDR active subscriber ratio: $R_{\text{CDR}} = 0.40$.

**Deliverable:** Calculate the estimated displaced population $D(\text{North})$ using operational fusion weights $\omega_d = 0.50$, $\omega_l = 0.30$, and $\omega_c = 0.20$. State the number of displaced households assuming $H_{\text{size}} = 5.0$.
- *Acceptance Criteria:* Displaced population and household counts calculated exactly.
- *Estimated Time:* 15 minutes.

### Exercise 2: Sphere Standards Multi-Sector Bill-of-Materials
Using the displaced population calculated in Exercise 1, dimension a 30-day first-phase emergency response bill-of-materials adhering strictly to Sphere standards.
1. Total potable water volume (in $\text{m}^3$) and daily requirement (in metric tons).
2. Total quantity of $67\text{ mg}$ NaDCC water purification tablets (each tablet treats $20\text{ L}$).
3. Total food tonnage (in metric tons) using the standard $0.540\text{ kg/person/day}$ emergency dry ration, disaggregated into rice ($0.400\text{ kg}$), pulses ($0.060\text{ kg}$), fortified oil ($0.030\text{ kg}$), and CSB+ ($0.050\text{ kg}$).
4. Total quantity of standard $4\text{ m} \times 5\text{ m}$ plastic tarpaulins ($2$ per household).

**Deliverable:** Tabulated multi-sector bill-of-materials with exact numerical units.
- *Acceptance Criteria:* All mathematical conversions correct; zero unit errors.
- *Estimated Time:* 20 minutes.

### Exercise 3: Cold-Climate Caloric Adjustment
An emergency displacement occurs during winter in high-altitude terrain where the mean ambient temperature is $-5^\circ\text{C}$. The baseline Sphere caloric standard is $2,100\text{ kcal/person/day}$ at $20^\circ\text{C}$, with a mandatory increase of $100\text{ kcal/person/day}$ for every $5^\circ\text{C}$ drop below $20^\circ\text{C}$.

**Deliverable:** 
1. Calculate the adjusted daily caloric requirement per person.
2. If the additional calories are supplied entirely by increasing the cereal ration ($360\text{ kcal}$ per $100\text{ g}$), calculate the revised daily cereal ration weight (in grams) and the percentage increase in total cereal pipeline tonnage.
- *Acceptance Criteria:* Caloric adjustment formula applied correctly; ration weight derived accurately.
- *Estimated Time:* 15 minutes.

### Exercise 4: Local Market Absorption Capacity & Inflation Boundary
In District North (from Exercise 1), the gross 30-day rice requirement must be procured. The local commercial rice market operates under the following parameters:
- Baseline monthly trading volume: $Q_0 = 3,000\text{ MT/month}$.
- Residual commercial demand from non-displaced host population: $Q_{\text{comm}} = 2,200\text{ MT/month}$.
- Supply price elasticity: $E_s = 0.45$.
- Maximum tolerable local price inflation: $\pi_{\max} = 10\%$ ($0.10$).

**Deliverable:**
1. Calculate the maximum market surge capacity $Q_{\max}$.
2. Calculate the net market absorption capacity $Q_{\text{absorb}}$ available for humanitarian voucher redemption.
3. Determine the optimal split between Cash & Voucher Assistance (CVA) and physical in-kind international shipping for the required rice volume.
- *Acceptance Criteria:* Elasticity equation applied correctly; modality split expressed in metric tons and percentages.
- *Estimated Time:* 20 minutes.

### Exercise 5: Dynamic Net Pipeline Gap Scheduling
An agency must deliver $1,200\text{ MT}$ of in-kind relief grain over 4 weekly cycles ($300\text{ MT/week}$).
- Opening stock in local warehouse at Week 1: $150\text{ MT}$.
- Inbound confirmed shipments: Week 1: $0\text{ MT}$; Week 2: $200\text{ MT}$; Week 3: $400\text{ MT}$; Week 4: $500\text{ MT}$.
- Transit and handling loss allowance: $5\%$ applied to all inbound arriving shipments.

**Deliverable:** Construct a complete 4-week pipeline balance table displaying Opening Stock, Net Arriving Inbound, Total Available, Weekly Requirement, Closing Stock, and Net Pipeline Gap for each week.
- *Acceptance Criteria:* Inventory balance equations satisfied; losses accounted for; stockouts flagged.
- *Estimated Time:* 25 minutes.

### Exercise 6: Pipeline Capitalization & Lead Time Safety Sizing
A remote humanitarian field operation consumes an average of $25\text{ MT/day}$ of therapeutic food. The international multimodal supply pipeline exhibits a mean lead time of $\overline{L} = 45\text{ days}$ with a standard deviation of $\sigma_L = 8\text{ days}$. Daily consumption demand variability is $\sigma_D = 4\text{ MT/day}$. The humanitarian country team mandates a $99\%$ service level ($Z_{SL} = 2.33$) to prevent acute starvation.

**Deliverable:**
1. Calculate the required safety stock (in metric tons).
2. Calculate the total capitalized pipeline inventory (on-order plus safety stock) that must be funded and maintained.
- *Acceptance Criteria:* Safety stock formula under stochastic lead time and demand applied correctly.
- *Estimated Time:* 20 minutes.

### Exercise 7: Water Trucking Fleet Dimensioning and Fuel Burn
A displaced camp of $75,000$ persons requires $15\text{ L/person/day}$ of drinking and hygiene water. Water must be hauled from a spring $20\text{ km}$ away.
- Tanker truck capacity: $C_{\text{truck}} = 12\text{ m}^3$ ($12,000\text{ L}$).
- Daily operating window: $10\text{ hours/day}$ ($600\text{ minutes}$).
- Turnaround cycle times: Loading $= 25\text{ min}$; Outbound loaded trip $= 45\text{ min}$; Discharge $= 20\text{ min}$; Return empty trip $= 35\text{ min}$; Driver rest/inspection $= 15\text{ min}$.
- Fleet availability factor: $\eta_{\text{avail}} = 0.75$.
- Tanker fuel economy: $0.50\text{ Liters of diesel per km}$ (round trip $= 40\text{ km}$).

**Deliverable:**
1. Calculate daily water requirement in $\text{m}^3$.
2. Calculate round-trip cycle time $T_{\text{cycle}}$ and trips per truck per day.
3. Calculate the number of active tanker trucks and total fleet size required.
4. Calculate total daily diesel fuel required to sustain the water trucking operation.
- *Acceptance Criteria:* Accurate queuing and turnaround arithmetic; fuel consumption calculated correctly.
- *Estimated Time:* 25 minutes.

### Exercise 8: Critique of Unsolicited Bilateral Donations (UBD)
A non-governmental organization receives an unannounced donation of 100 metric tons of canned beef stew (in steel cans packed in wooden crates, weight: $1.2\text{ kg/can}$, volume: $1.8\text{ m}^3\text{/MT}$) shipped to a tropical island disaster response where $85\%$ of the affected population is vegetarian and airhead apron space is restricted to two C-130 aircraft.

**Deliverable:** Write a professional logistics critique (300–400 words) analyzing the operational impact of accepting this donation. Address airport slot utilization, warehouse cubic capacity, dietary compliance, and alternative cash-equivalent opportunity cost.
- *Acceptance Criteria:* Identifies at least four distinct supply chain failure mechanisms.
- *Estimated Time:* 25 minutes.

### Exercise 9: Airfreight Palletization & Volumetric Weight Optimization
An emergency air charter (Boeing 777F, maximum structural payload: $102\text{ MT}$, maximum cargo volume: $650\text{ m}^3$) is booked to deploy relief supplies.
- Commodity 1: High Energy Biscuits (HEB) — Density: $600\text{ kg/m}^3$.
- Commodity 2: Family Hygiene Kits — Density: $150\text{ kg/m}^3$.
IATA air cargo volumetric conversion standard: $1\text{ m}^3 = 167\text{ kg}$ chargeable weight.

**Deliverable:**
1. Determine whether each commodity is weight-limited or volume-limited (cubing out).
2. Formulate the linear programming blending problem that maximizes total relief payload weight without exceeding aircraft structural payload ($102\text{ MT}$) or usable volume ($650\text{ m}^3$). Solve for the optimal tonnage of HEB and Hygiene Kits.
- *Acceptance Criteria:* Linear constraints formulated and simultaneous solution computed.
- *Estimated Time:* 30 minutes.

### Exercise 10: Directional Displacement Modeling via the Radiation Model
An earthquake destroys rural District A (population: $m_A = 100,000$; displaced: $D_A = 40,000$). Two candidate host cities exist:
- City B: Population $n_B = 150,000$; distance $r_{AB} = 30\text{ km}$.
- City C: Population $n_C = 300,000$; distance $r_{AC} = 60\text{ km}$.
The population residing in the circle of radius $30\text{ km}$ around A (excluding A and B) is $s_{AB} = 50,000$. The population residing in the circle of radius $60\text{ km}$ around A (excluding A and C) is $s_{AC} = 220,000$.

**Deliverable:**
1. Using the Radiation Model, calculate the predicted displaced population moving from A to City B ($T_{AB}$) and from A to City C ($T_{AC}$).
2. Determine which city requires the larger forward humanitarian logistics hub.
- *Acceptance Criteria:* Radiation model formula evaluated precisely.
- *Estimated Time:* 25 minutes.

---

## 13. Self-check answers

### Answer to Exercise 1: Multi-District Satellite Displacement Fusion
1. **Fused Displacement Rate ($\delta_{\text{North}}$):**
   $$\delta_{\text{North}} = \omega_d \cdot S + \omega_l \cdot (1 - R_{\text{NTL}}) + \omega_c \cdot (1 - R_{\text{CDR}})$$
   $$\delta_{\text{North}} = (0.50 \cdot 0.65) + (0.30 \cdot (1 - 0.30)) + (0.20 \cdot (1 - 0.40))$$
   $$\delta_{\text{North}} = (0.50 \cdot 0.65) + (0.30 \cdot 0.70) + (0.20 \cdot 0.60)$$
   $$\delta_{\text{North}} = 0.325 + 0.210 + 0.120 = 0.655 \quad (65.5\%)$$
2. **Displaced Population Count:**
   $$D(\text{North}) = 240,000 \cdot 0.655 = 157,200 \text{ persons}$$
3. **Displaced Households ($H_{\text{size}} = 5.0$):**
   $$N_{\text{HH}} = \frac{157,200}{5.0} = 31,440 \text{ households}$$

### Answer to Exercise 2: Sphere Standards Multi-Sector Bill-of-Materials
1. **WASH Potable Water Sizing:**
   - Daily Water: $157,200 \times 15.0\text{ L/day} = 2,358,000\text{ L/day} = 2,358\text{ m}^3\text{/day} = 2,358\text{ MT/day}$.
   - 30-Day Cumulative Water: $2,358\text{ m}^3\text{/day} \times 30\text{ days} = 70,740\text{ m}^3$ ($70,740\text{ Metric Tons}$).
2. **Water Purification Tablets (Aquatabs 67mg, 1 per 20 L):**
   $$\text{Tablets}_{30\text{-day}} = \frac{70,740,000\text{ L}}{20\text{ L/tab}} = 3,537,000 \text{ tablets}$$
3. **Food Security Emergency Ration (30 Days):**
   - Gross Daily Food: $157,200 \times 0.540\text{ kg} / 1,000 = 84.888\text{ MT/day}$.
   - Gross 30-Day Food: $84.888 \times 30 = 2,546.64\text{ MT}$.
   - **Rice ($0.400\text{ kg/p/d}$):** $157,200 \times 0.400 \times 30 / 1,000 = 1,886.40\text{ MT}$.
   - **Pulses ($0.060\text{ kg/p/d}$):** $157,200 \times 0.060 \times 30 / 1,000 = 282.96\text{ MT}$.
   - **Vegetable Oil ($0.030\text{ kg/p/d}$):** $157,200 \times 0.030 \times 30 / 1,000 = 141.48\text{ MT}$.
   - **CSB+ ($0.050\text{ kg/p/d}$):** $157,200 \times 0.050 \times 30 / 1,000 = 235.80\text{ MT}$.
   *(Check sum: $1,886.40 + 282.96 + 141.48 + 235.80 = 2,546.64\text{ MT}$)*
4. **Shelter Plastic Tarpaulins:**
   $$\text{Tarpaulins} = 31,440\text{ HH} \times 2 = 62,880 \text{ standard } 4\text{m} \times 5\text{m tarpaulins}$$

### Answer to Exercise 3: Cold-Climate Caloric Adjustment
1. **Adjusted Caloric Intake:**
   - Temperature drop below baseline ($20^\circ\text{C}$): $\Delta T = 20^\circ\text{C} - (-5^\circ\text{C}) = 25^\circ\text{C}$.
   - Number of $5^\circ\text{C}$ decrements: $\frac{25}{5} = 5$ intervals.
   - Additional caloric requirement: $5 \times 100\text{ kcal} = +500\text{ kcal/person/day}$.
   - **Adjusted Standard:** $2,100 + 500 = 2,600\text{ kcal/person/day}$.
2. **Revised Cereal Ration Weight:**
   - Cereal energy density: $360\text{ kcal} / 100\text{ g} = 3.6\text{ kcal/g}$.
   - Additional cereal required: $\frac{500\text{ kcal}}{3.6\text{ kcal/g}} = 138.89\text{ grams/person/day}$.
   - Revised cereal ration: $400\text{ g} + 138.89\text{ g} = 538.89\text{ g/person/day}$.
   - Percentage increase in cereal pipeline: $\frac{138.89}{400} \times 100\% = 34.72\%$.

### Answer to Exercise 4: Local Market Absorption Capacity & Inflation Boundary
1. **Maximum Market Throughput ($Q_{\max}$):**
   $$Q_{\max} = Q_0 \cdot \Big(1 + E_s \cdot \pi_{\max}\Big) = 3,000 \cdot \Big(1 + 0.45 \cdot 0.10\Big) = 3,000 \cdot 1.045 = 3,135 \text{ MT/month}$$
2. **Net Market Absorption Capacity ($Q_{\text{absorb}}$):**
   $$Q_{\text{absorb}} = Q_{\max} - Q_{\text{comm}} = 3,135 - 2,200 = 935 \text{ MT/month}$$
3. **Modality Allocation for Rice ($1,886.40\text{ MT}$ total):**
   - **Cash & Voucher Assistance (CVA):** $935.00\text{ MT}$ ($49.57\%$ of total rice requirement).
   - **In-Kind Logistics Pipeline:** $1,886.40 - 935.00 = 951.40\text{ MT}$ ($50.43\%$).

### Answer to Exercise 5: Dynamic Net Pipeline Gap Scheduling
Net arriving inbound accounts for $5\%$ loss: $\text{Arriving}_{\text{net}} = \text{Shipment} \times 0.95$.
- Week 1: $0 \times 0.95 = 0.0\text{ MT}$
- Week 2: $200 \times 0.95 = 190.0\text{ MT}$
- Week 3: $400 \times 0.95 = 380.0\text{ MT}$
- Week 4: $500 \times 0.95 = 475.0\text{ MT}$

```text
+======+===============+==================+================+=============+==============+===========+
| Week | Opening Stock | Net Inbound Arriv| Total Available| Weekly Req  | Closing Stock| Net Gap   |
+======+===============+==================+================+=============+==============+===========+
| 1    | 150.0 MT      |    0.0 MT        | 150.0 MT       | 300.0 MT    |   0.0 MT     | 150.0 MT  |
| 2    |   0.0 MT      |  190.0 MT        | 190.0 MT       | 300.0 MT    |   0.0 MT     | 110.0 MT  |
| 3    |   0.0 MT      |  380.0 MT        | 380.0 MT       | 300.0 MT    |  80.0 MT     |   0.0 MT  |
| 4    |  80.0 MT      |  475.0 MT        | 555.0 MT       | 300.0 MT    | 255.0 MT     |   0.0 MT  |
+======+===============+==================+================+=============+==============+===========+
```
*Summary:* Severe stockouts occur in Week 1 ($150.0\text{ MT}$ deficit) and Week 2 ($110.0\text{ MT}$ deficit). Full operational stability is achieved in Week 3, closing Week 4 with a healthy $255.0\text{ MT}$ safety stock buffer.

### Answer to Exercise 6: Pipeline Capitalization & Lead Time Safety Sizing
1. **Safety Stock Formulation:**
   $$\sigma_{\text{pipeline}} = \sqrt{\overline{L} \cdot \sigma_D^2 + \overline{D}^2 \cdot \sigma_L^2}$$
   $$\sigma_{\text{pipeline}} = \sqrt{45 \cdot (4)^2 + (25)^2 \cdot (8)^2} = \sqrt{45 \cdot 16 + 625 \cdot 64} = \sqrt{720 + 40,000} = \sqrt{40,720} \approx 201.79 \text{ MT}$$
   $$\text{Safety Stock} = Z_{SL} \cdot \sigma_{\text{pipeline}} = 2.33 \cdot 201.79 \approx 470.17 \text{ MT}$$
2. **Total Capitalized Pipeline Inventory:**
   $$\text{Pipeline In-Transit} = \overline{D} \cdot \overline{L} = 25 \text{ MT/day} \times 45 \text{ days} = 1,125.0 \text{ MT}$$
   $$\text{Total Capitalized Stock} = 1,125.0 + 470.17 = 1,595.17 \text{ MT}$$

### Answer to Exercise 7: Water Trucking Fleet Dimensioning and Fuel Burn
1. **Daily Water Requirement:**
   $$V_{\text{daily}} = 75,000 \times 15.0\text{ L/day} = 1,125,000\text{ L/day} = 1,125\text{ m}^3\text{/day}$$
2. **Cycle Time and Daily Trips:**
   $$T_{\text{cycle}} = 25 + 45 + 20 + 35 + 15 = 140\text{ minutes/trip}$$
   $$\text{Trips per truck/day} = \left\lfloor \frac{600\text{ min}}{140\text{ min}} \right\rfloor = 4\text{ trips/day}$$
   Daily delivery per tanker $= 4 \times 12\text{ m}^3 = 48\text{ m}^3\text{/day}$.
3. **Fleet Size Sizing:**
   $$\text{Active Tankers} = \left\lceil \frac{1,125}{48} \right\rceil = 24\text{ active tanker trucks}$$
   $$\text{Total Fleet Size} = \left\lceil \frac{24}{0.75} \right\rceil = 32\text{ tanker trucks in fleet}$$
4. **Daily Diesel Consumption:**
   Daily distance per active truck $= 4\text{ trips} \times 40\text{ km/trip} = 160\text{ km/day}$.
   $$\text{Total Fuel} = 24\text{ active trucks} \times 160\text{ km} \times 0.50\text{ L/km} = 1,920\text{ Liters of diesel/day}$$

### Answer to Exercise 8: Critique of Unsolicited Bilateral Donations (UBD)
*Grading Rubric:* Strong critique must identify:
1. **Dietary Non-Compliance:** $85\%$ vegetarian population means beef stew will be rejected or traded for pennies, failing to address nutritional deficits.
2. **Volumetric & Weight Inefficiency:** Canned food with heavy water content ($1.2\text{ kg/can}$) and wooden crates consumes massive payload and warehouse volume ($1.8\text{ m}^3\text{/MT}$) compared to dense grains or high-energy biscuits ($0.8\text{ m}^3\text{/MT}$).
3. **Airhead Slot Cannibalization:** Unsolicited cargo flights consume scarce C-130 landing slots and ramp handling equipment, delaying surgical teams and water purification gear.
4. **Economic Opportunity Cost:** Ocean freight and handling costs for $100\text{ MT}$ of canned beef often exceed $\$150,000$, which could have procured $300\text{ MT}$ of regionally sourced grain or funded cash vouchers.

### Answer to Exercise 9: Airfreight Palletization & Volumetric Weight Optimization
1. **Volumetric Classification:**
   - IATA benchmark density: $167\text{ kg/m}^3$.
   - High Energy Biscuits ($600\text{ kg/m}^3 > 167\text{ kg/m}^3$): **Weight-limited** (dense cargo).
   - Family Hygiene Kits ($150\text{ kg/m}^3 < 167\text{ kg/m}^3$): **Volume-limited** (cubing out).
2. **Linear Programming Optimization:**
   Let $x_1$ be tonnage of HEB (MT) and $x_2$ be tonnage of Hygiene Kits (MT).
   Maximize Payload: $W = x_1 + x_2$
   Subject to:
   - Structural Payload Constraint: $x_1 + x_2 \le 102\text{ MT}$
   - Volumetric Capacity Constraint: $\frac{x_1}{0.600} + \frac{x_2}{0.150} \le 650\text{ m}^3 \implies 1.667 x_1 + 6.667 x_2 \le 650$
   Solving simultaneously at the intersection:
   From payload: $x_1 = 102 - x_2$.
   Substitute into volume: $1.667(102 - x_2) + 6.667 x_2 = 650$
   $170.03 + 5.0 x_2 = 650 \implies 5.0 x_2 = 479.97 \implies x_2 = 96.0\text{ MT of Hygiene Kits}$.
   $x_1 = 102 - 96.0 = 6.0\text{ MT of HEB}$.
   Check Volume: $\frac{6.0}{0.600} + \frac{96.0}{0.150} = 10.0 + 640.0 = 650.0\text{ m}^3$ (exactly hits $650\text{ m}^3$ and $102\text{ MT}$).

### Answer to Exercise 10: Directional Displacement Modeling via the Radiation Model
1. **Radiation Model Calculations:**
   - **Origin District A:** $m_A = 100,000$; $D_A = 40,000$.
   - **Flow to City B ($T_{AB}$):**
     $$T_{AB} = D_A \cdot \frac{m_A \cdot n_B}{(m_A + s_{AB})(m_A + n_B + s_{AB})}$$
     $$T_{AB} = 40,000 \cdot \frac{100,000 \cdot 150,000}{(100,000 + 50,000)(100,000 + 150,000 + 50,000)}$$
     $$T_{AB} = 40,000 \cdot \frac{15,000,000,000}{150,000 \cdot 300,000} = 40,000 \cdot \frac{15 \cdot 10^9}{45 \cdot 10^9} = 40,000 \cdot \frac{1}{3} \approx 13,333 \text{ persons}$$
   - **Flow to City C ($T_{AC}$):**
     $$T_{AC} = D_A \cdot \frac{m_A \cdot n_C}{(m_A + s_{AC})(m_A + n_C + s_{AC})}$$
     $$T_{AC} = 40,000 \cdot \frac{100,000 \cdot 300,000}{(100,000 + 220,000)(100,000 + 300,000 + 220,000)}$$
     $$T_{AC} = 40,000 \cdot \frac{30,000,000,000}{320,000 \cdot 620,000} = 40,000 \cdot \frac{30 \cdot 10^9}{198.4 \cdot 10^9} = 40,000 \cdot 0.1512 \approx 6,048 \text{ persons}$$
2. **Logistics Staging Decision:**
   City B will absorb more than double the displaced survivors of City C ($13,333$ vs $6,048$), despite City C having twice the baseline population. This is because City B is geographically closer and has fewer intervening population intervening opportunities ($s_{AB} = 50,000$ vs $s_{AC} = 220,000$). The primary forward logistics distribution hub must be established at **City B**.

---

## 14. Interview questions

### Question 1: In the first 48 hours of a sudden-onset disaster, why can you not simply distribute survey teams to conduct household-level needs assessments?
- **Strong Answer:** "Household surveys during the Golden 72 Hours represent a fatal operational error. Physical transport infrastructure—bridges, roads, communication masts—is severed, making broad spatial access impossible. Deploying field surveyors across destroyed terrain exposes staff to secondary hazards (aftershocks, flash floods) and yields a statistically biased sample of only easily accessible fringe locations. Furthermore, household sampling requires weeks to code, clean, and analyze, delaying life-saving logistics dispatches. Industry practice dictates executing a **Phase 1 MIRA Preliminary Scenario Definition** within 72 hours, utilizing pre-crisis baseline demographics (COD-PS, WorldPop) fused with satellite Synthetic Aperture Radar (SAR) coherence damage proxy maps and Nighttime Lights radiance drops to rapidly dimension bulk commodity requirements, reserving primary household sampling for subsequent phases."
- **Weak Answer:** "Because we don't have enough clipboards and tablets, and people are too traumatized to answer detailed survey questions immediately after a disaster."

### Question 2: How do you evaluate whether to deploy Cash and Voucher Assistance (CVA) versus physical in-kind commodity shipments in an emergency food response?
- **Strong Answer:** "The modality decision is governed by local market absorption capacity, evaluated through the Emergency Market Mapping and Analysis (EMMA) framework. We evaluate three non-negotiable physical gates: (1) physical safety and corridor access to markets; (2) functionality of financial infrastructure (mobile money networks, operating banks, liquidity); and (3) merchant supply elasticity ($E_s = \frac{\% \Delta Q_s}{\% \Delta P}$). If wholesalers have intact supply lines and can expand inventory without triggering local price inflation beyond an established threshold (typically $\le 10–15\%$), CVA is superior because it eliminates international maritime freight costs, accelerates delivery, and stimulates local economic recovery. However, if supply elasticity is zero ($E_s \approx 0$) due to severed external supply routes or destroyed merchant warehouses, injecting cash triggers immediate local hyperinflation, harming both beneficiaries and non-beneficiary host families. In that scenario, physical in-kind commodity pipelines are mandatory."
- **Weak Answer:** "Cash is always better because it gives beneficiaries dignity and saves shipping costs, so we should always distribute cash unless donors specifically force us to deliver food."

### Question 3: Explain the difference between Gross Humanitarian Requirement and Net Pipeline Gap, and why confusing them causes logistics failures.
- **Strong Answer:** "The Gross Humanitarian Requirement (GHR) is the theoretical commodity volume required to support the target population at Sphere standards over a defined time horizon ($GHR = \text{Beneficiaries} \times \text{Standard} \times \text{Time}$). The Net Pipeline Gap (NPG) is the operational procurement deficit: $NPG(t) = GHR(t) - [I(t-1) + Q_{\text{inbound}}(t) - \Lambda(t)]$, which subtracts opening on-hand inventory, confirmed arriving shipments, and local in-country procurements while accounting for transit losses. Confusing the two causes catastrophic supply chain bullwhips. If an agency treats GHR as its procurement order, it ignores stock already pre-positioned in regional hubs (like UNHRD) and shipments already afloat. This results in massive over-ordering, port and airport apron gridlock, warehouse overflow, container demurrage penalties, and eventually commodity expiration in tropical warehouses."
- **Weak Answer:** "Gross requirement is what everyone needs, and net pipeline gap is just the money we have left in our donor grant budget."

### Question 4: Under Sphere standards, what is the survival water requirement, and how must logistics dimension the supply pipeline beyond simple water trucking?
- **Strong Answer:** "Sphere Standard 2.1 establishes a non-negotiable minimum of 15 liters per person per day for drinking, cooking, and basic hygiene. Logistically, delivering $15\text{ L/p/d}$ exclusively via water trucking is physically unsustainable for large displaced populations ($100,000$ people require $1,500\text{ metric tons}$ of water daily, equivalent to 150 ten-ton water tankers operating continuous rotations over damaged roads). Therefore, humanitarian logistics dimensions a multi-layered WASH pipeline: (1) Immediate airlift of point-of-use water purification tablets (such as NaDCC Aquatabs, 1 tablet per 20 L) and collapsible 10-liter or 20-liter jerrycans (2 per household); (2) Deployment of modular, skid-mounted Water Treatment Units (WTUs) using flocculation, sand filtration, and chlorination to treat local surface water; (3) Airfreight of flexible PVC bladder tanks ($5\text{ m}^3$ to $10\text{ m}^3$) and rapid distribution tapstands (max 250 persons per tap, max 500 meters walking distance) to establish decentralized distribution nodes."
- **Weak Answer:** "Sphere says everyone needs 2 liters of drinking water a day, so we just buy pallets of bottled water from commercial supermarkets and fly them in."

### Question 5: What is an Unsolicited Bilateral Donation (UBD), and what operational protocol should a humanitarian logistics officer enforce when one arrives?
- **Strong Answer:** "An Unsolicited Bilateral Donation is relief cargo dispatched into a disaster zone without a prior formal request, cluster validation, or consignee agreement from national authorities or the Logistics Cluster. UBDs typically consist of unsorted clothing, expired medicines, culturally inappropriate canned foods, or broken medical equipment. When a UBD flight lands, it consumes scarce Maximum Aircraft on Ground (MOG) parking spots, ties up cargo forklifts, and clogs warehouse floor space. A professional logistics officer must enforce strict pre-clearance flight slot allocation protocols. If an unmanifested UBD arrives, it must be shunted to secondary staging aprons, refused customs exemption, and prioritized behind validated life-saving cluster cargo. If goods cannot be verified by health or food safety authorities, the logistics officer must recommend formal customs rejection or controlled destruction rather than absorbing sorting and storage costs."
- **Weak Answer:** "It is a generous donation from another country. We should always accept it, thank the donor on social media, and find someone in the camp who can use it."

### Question 6: How do Synthetic Aperture Radar (SAR) coherence maps identify building damage through cloud cover and darkness?
- **Strong Answer:** "Optical satellite sensors rely on reflected sunlight in the visible and near-infrared spectrum and are rendered useless by cloud cover, atmospheric smoke, and nocturnal darkness. In contrast, Synthetic Aperture Radar (SAR) satellites (e.g., Sentinel-1) actively emit microwave electromagnetic pulses at calibrated wavelengths (e.g., C-band, $\lambda \approx 5.6\text{ cm}$) and measure both the amplitude and the phase of the backscattered wave. In an intact urban area, buildings act as stable corner reflectors, maintaining a consistent phase relationship across repeat-pass satellite orbits, resulting in high interferometric coherence ($\gamma \approx 1.0$). When buildings collapse or experience severe structural deformation, the geometric arrangement of scatterers changes randomly, causing decorrelation and phase coherence collapse ($\gamma \to 0$). By calculating the coherence loss ($\Delta \gamma = \gamma_{\text{pre}} - \gamma_{\text{co}}$), remote sensing algorithms generate Damage Proxy Maps (DPMs) that pinpoint urban collapse zones within hours of satellite overpass, regardless of weather."
- **Weak Answer:** "SAR takes a high-resolution color photograph using special infrared night-vision lenses that can see through clouds."

### Question 7: Why must a humanitarian food ration be evaluated for macronutrient balance and ambient temperature, rather than simply meeting a gross caloric threshold?
- **Strong Answer:** "A survival ration cannot merely deliver 2,100 calories of pure refined sugar or carbohydrates. The human body requires structural macronutrient balance to prevent severe physiological decline: 10% to 12% of total energy must derive from protein (52.5–63 grams) to prevent muscle catabolism and sustain immune function, and 17% to 20% must derive from lipids/fats (40–46.7 grams) to enable absorption of fat-soluble vitamins (A, D, E, K) and provide caloric density. Furthermore, the baseline 2,100 kcal standard assumes light activity at an ambient temperature of $20^\circ\text{C}$. In winter or high-altitude operations, the human body must burn additional calories to maintain core body temperature through metabolic thermogenesis. Under Sphere guidelines, logisticians must add $100\text{ kcal/person/day}$ for every $5^\circ\text{C}$ drop below $20^\circ\text{C}$. Sizing a warm-climate ration for a sub-zero winter emergency results in progressive hypothermia, immune collapse, and surging mortality among vulnerable populations."
- **Weak Answer:** "Calories are calories; as long as the total number on the food packaging adds up to 2,100, the body burns them the same way."

### Question 8: How do you mathematically calculate the required capitalized pipeline inventory for a humanitarian supply corridor under stochastic lead times?
- **Strong Answer:** "Capitalized pipeline inventory represents the total committed inventory required to sustain daily field distributions without experiencing stockouts. It consists of two components: expected cycle pipeline stock and safety stock. Cycle stock is $\overline{D} \cdot \overline{L}$, where $\overline{D}$ is the average daily consumption rate and $\overline{L}$ is the average total lead time across all procurement, shipping, customs, and staging legs. Under stochastic conditions where daily demand has standard deviation $\sigma_D$ and lead time has standard deviation $\sigma_L$, the combined variance of lead-time demand is $\sigma_{\text{pipeline}}^2 = \overline{L} \cdot \sigma_D^2 + \overline{D}^2 \cdot \sigma_L^2$. The required safety buffer is $Z_{SL} \cdot \sigma_{\text{pipeline}}$, where $Z_{SL}$ represents the standard normal service level factor. For critical survival commodities, we enforce $Z_{SL} \ge 2.33$ ($99\%$ non-stockout probability). Therefore, total capitalized stock is $\text{Stock} = \overline{D} \cdot \overline{L} + Z_{SL} \cdot \sqrt{\overline{L} \cdot \sigma_D^2 + \overline{D}^2 \cdot \sigma_L^2}$."
- **Weak Answer:** "You just take the daily demand and multiply it by the lead time, then add 10% on top just to be safe."

### Question 9: What is the Maximum Aircraft on Ground (MOG) constraint, and how does it dictate humanitarian airlift schedules?
- **Strong Answer:** "MOG is the maximum number of aircraft that can be physically accommodated simultaneously on an airfield's aprons, taxiways, and unloading ramps. It is subdivided into parking MOG (physical footprint) and working MOG (the number of aircraft that ground handling crews, K-loaders, and forklifts can actively offload at once). Even if an airport runway can handle 30 landings per hour, if the working MOG is 2, any arrival rate exceeding 2 aircraft per ground-turnaround window ($2–3\text{ hours}$ for widebodies) results in aircraft holding patterns, fuel diversions, or ramp gridlock. Logistics planners must enforce a strict slot-allocation reservation system, sequencing aircraft arrivals so that hourly inflow does not exceed the airfield's cargo offload rate ($C_{\text{offload}} = \text{Working MOG} \times \frac{\text{Payload}}{\text{Turnaround Time}}$)."
- **Weak Answer:** "MOG is the maximum weight an aircraft can carry when landing on a dirt runway."

### Question 10: How does the Emergency Market Mapping and Analysis (EMMA) toolkit protect non-beneficiary host community members?
- **Strong Answer:** "When an agency injects massive cash or voucher assistance into a post-disaster economy without assessing market capacity, aggregate purchasing power surges instantly. If local merchant supply is constrained or inelastic ($E_s \approx 0$), traders respond by increasing prices to clear the market. While cash beneficiaries may survive because their vouchers subsidize higher prices, poor and working-class families in the non-beneficiary host community—who receive no humanitarian aid—are suddenly priced out of basic staples like rice, wheat, and cooking oil. EMMA specifically calculates the market absorption threshold $Q_{\text{absorb}} = Q_{\text{base}}(1 + E_s \cdot \pi_{\max}) - Q_{\text{comm}}$. By capping cash transfers at this volume and importing physical in-kind aid for any remaining deficit, EMMA prevents price hyperinflation, ensuring that the humanitarian intervention does no harm to the host community's cost of living."
- **Weak Answer:** "EMMA protects host communities by making sure that NGOs buy tea and snacks from local shopkeepers during assessment meetings."

### Question 11: How do you dimension a Ready-to-Use Therapeutic Food (RUTF) supply pipeline following a nutrition survey that reports a 4.0% Severe Acute Malnutrition (SAM) rate?
- **Strong Answer:** "We dimension the therapeutic pipeline through epidemiological cohort modeling rather than gross population multiplication. First, we isolate the demographic cohort of children aged 6–59 months, which typically constitutes approximately 18% of a displaced developing-world population ($P_{<5} = 0.18 \cdot P_{\text{displaced}}$). Second, we calculate the active SAM caseload applying the survey rate ($r_{\text{SAM}} = 0.04$) and an Outpatient Therapeutic Program (OTP) geographic and clinical coverage target (typically $\alpha_{\text{OTP}} = 70\%$ under Sphere standards), adding an incidence factor ($1.6$) to account for newly incident cases over a 6-month planning horizon: $\text{Caseload} = P_{<5} \cdot r_{\text{SAM}} \cdot 1.6 \cdot \alpha_{\text{OTP}}$. Third, we translate clinical caseload into supply units: the standard WHO/UNICEF OTP protocol requires an average of 1.5 cartons ($225\text{ sachets} = 20.7\text{ kg}$) per treated child over an 8-week recovery course. Finally, we multiply caseload by $1.5\text{ cartons}$, add a $5\%$ pipeline buffer for damaged packaging or transport losses, and convert to metric tons ($1\text{ carton} = 13.8\text{ kg}$). This gives the precise volumetric requirement for sea freight container stuffing or charter flights."
- **Weak Answer:** "You just take 4% of the whole population and buy 4% worth of peanut butter packets."

### Question 12: How do you structure a multi-echelon staging network connecting UN Humanitarian Response Depots (UNHRD), national staging hubs, and last-mile Forward Distribution Points (FDPs)?
- **Strong Answer:** "A humanitarian staging network operates as a three-echelon push-pull inventory system. Echelon 1 consists of global pre-positioning hubs (the UNHRD network in Brindisi, Dubai, Panama, Accra, and Subang), which hold strategic stockpiles of universal, standardized capital assets (mobile storage units, water bladders, high-energy biscuits, trauma kits) under a central push strategy with guaranteed 24–48 hour international air dispatch. Echelon 2 consists of National Staging Hubs established at the primary entry seaport or airport in the affected country (e.g., Manila or Cebu during Typhoon Haiyan); here, international bulk cargo is deconsolidated, customs-cleared under ASYCUDA fast-track protocols, and repackaged into multi-commodity household kits. Echelon 3 consists of Forward Logistics Bases (FLBs) located within 2 to 4 hours trucking distance of crisis zones, which act as cross-docking buffers holding 3 to 7 days of safety stock. From the FLBs, daily distribution convoys push supplies to temporary Forward Distribution Points (FDPs) for same-day distribution to beneficiaries, ensuring zero overnight stock remains at unprotected last-mile nodes."
- **Weak Answer:** "You store all the aid in one big warehouse near the capital city and have beneficiaries take buses to come pick up their boxes."

---

## 15. Key terms

- **ASYCUDA Relief Module:** Automated customs system module that accelerates clearance for accredited humanitarian emergency consignments.
- **Common Operational Datasets (CODs):** Authoritative geographic and demographic reference data approved by the UN Humanitarian Country Team.
- **Coping Strategies Index (CSI):** A standardized indicator measuring household behavioral adjustments to food shortages.
- **Damage Proxy Map (DPM):** A spatial raster displaying physical structural damage derived from satellite Synthetic Aperture Radar (SAR) coherence change.
- **Emergency Market Mapping and Analysis (EMMA):** An operational toolkit for assessing post-disaster functionality of critical market systems.
- **Food Consumption Score (FCS):** A frequency-weighted composite score measuring household dietary diversity and nutrient intake.
- **Forward Distribution Point (FDP):** The final physical node in the supply chain where relief goods are handed directly to beneficiaries.
- **Free Chlorine Residual (FCR):** The concentration of active chlorine remaining in treated water to prevent secondary biological contamination.
- **Global Acute Malnutrition (GAM):** A population-level nutritional status metric measuring the percentage of children aged 6–59 months suffering from acute wasting.
- **Gross Humanitarian Requirement (GHR):** The total theoretical volume of relief supplies needed to support an affected population at Sphere standards.
- **Humanitarian Exchange Language (HXL):** A lightweight metadata tagging convention embedded in spreadsheet headers to standardize humanitarian data exchange.
- **Inter-Agency Standing Committee (IASC):** The primary UN forum for inter-agency coordination of humanitarian assistance.
- **Maximum Aircraft on Ground (MOG):** The maximum number of aircraft an airfield can accommodate simultaneously for parking, offloading, and servicing.
- **Mid-Upper Arm Circumference (MUAC):** A rapid anthropometric field screening measurement used to identify severe and moderate acute malnutrition.
- **Minimum Expenditure Basket (MEB):** The minimum financial expenditure required for a household to meet essential survival needs monthly.
- **Multi-Cluster/Sector Initial Rapid Assessment (MIRA):** A joint multi-agency assessment executed within 72h–14 days of crisis onset.
- **Net Pipeline Gap (NPG):** The unmet commodity demand calculated as gross requirements minus opening inventory, inbound shipments, and local purchases.
- **Nighttime Lights (NTL):** Nocturnal earth radiance data collected by satellite (VIIRS DNB) to detect electrical power failure and human displacement.
- **People in Need (PIN):** The verified number of individuals whose survival or basic human dignity is endangered without humanitarian assistance.
- **Pipeline Break:** A catastrophic logistics stockout where inventory at field distribution points drops to zero before replenishment arrives.
- **Pipeline Capitalization:** The total financial or volumetric inventory that must be maintained in transit to support continuous distribution.
- **Radiation Model for Mobility:** A parameter-free physical model predicting population migration flows based on employment and absorption capacity.
- **Ready-to-Use Therapeutic Food (RUTF):** High-energy, lipid-based fortified peanut-paste rations used to treat Severe Acute Malnutrition without water preparation.
- **Severe Acute Malnutrition (SAM):** A life-threatening condition characterized by extreme muscle wasting (MUAC < 115 mm or WHZ < -3).
- **Sodium Dichloroisocyanurate (NaDCC):** The standard effervescent chlorine-releasing chemical compound used in water purification tablets (Aquatabs).
- **Sphere Standards:** Universally recognized humanitarian operational minimums across WASH, food, shelter, and health.
- **Supply Price Elasticity ($E_s$):** The ratio of the percentage change in merchant supply quantity to the percentage change in local market price.
- **Unsolicited Bilateral Donations (UBD):** Inappropriate or uncoordinated relief cargo dispatched into a disaster zone without official request.
- **Visible Infrared Imaging Radiometer Suite (VIIRS):** Satellite sensor providing Day/Night Band nocturnal radiance data.
- **Water Treatment Unit (WTU):** A deployable, skid-mounted modular engineering system that pumps, coagulates, filters, and chlorinates surface water.

---

## 16. Sources

1. **Sphere Association (2018).** *The Sphere Handbook: Humanitarian Charter and Minimum Standards in Humanitarian Response*, 4th Edition. Geneva, Switzerland. ISBN: 978-1-908182-41-8.
2. **Inter-Agency Standing Committee (IASC) (2015).** *Multi-Cluster/Sector Initial Rapid Assessment (MIRA) Guidance*, UN OCHA Humanitarian Programme Cycle Steering Group. Geneva.
3. **Emergency Market Mapping and Analysis (EMMA) Project (2010).** *EMMA Toolkit: Emergency Market Mapping and Analysis for Practitioners*. Practical Action Publishing / Oxfam GB. ISBN: 978-1-85339-702-8.
4. **World Food Programme (WFP) (2023).** *Emergency Food Security Assessment (EFSA) Handbook*, 3rd Edition. WFP Emergency Preparedness and Response Division, Rome.
5. **The CALP Network (2022).** *The State of the World's Cash 2022: Cash and Voucher Assistance in Humanitarian Aid*. Cash Learning Partnership, Oxford, UK.
6. **Simini, F., González, M. C., Maritan, A., & Barabási, A. L. (2012).** "A Universal Model for Mobility and Migration Patterns." *Nature*, 484(7392), 96–100. DOI: 10.1038/nature10856.
7. **Yun, S. H. et al. (2015).** "Damage Proxy Map from Interferometric Synthetic Aperture Radar Coherence: Application to the 2015 Gorkha, Nepal Earthquake." *Earth, Planets and Space*, 67(1), 173. DOI: 10.1186/s40623-015-0343-7.
8. **WorldPop (2021).** *Global High Resolution Population Denominators Project*. School of Geography and Environmental Science, University of Southampton. DOI: 10.5258/SOTON/WP00645.
9. **Thomas, A. S. & Kopczak, L. R. (2005).** *From Logistics to Supply Chain Management: The Path Forward in the Humanitarian Sector*. Fritz Institute, San Francisco, CA.
10. **Kovács, G. & Spens, K. M. (2007).** "Humanitarian Logistics in Disaster Relief Operations." *International Journal of Physical Distribution & Logistics Management*, 37(2), 99–114. DOI: 10.1108/09600030710734820.
11. **Van Wassenhove, L. N. (2006).** "Humanitarian Aid Logistics: Supply Chain Management in High Gear." *Journal of the Operational Research Society*, 57(5), 475–489. DOI: 10.1057/palgrave.jors.2602125.
12. **Apte, A. (2009).** "Humanitarian Logistics: A New Field of Research and Education." *Foundations and Trends in Technology, Information and Operations Management*, 3(1), 1–100. DOI: 10.1561/0200000014.

---

## 17. Links

- **Parent Module Index:** [Module 35: Public Sector, Defence, and Humanitarian Logistics](_index.md)
- **Module Survey File:** [Module 35 Survey File: Public Sector, Defence, and Humanitarian Logistics](../../../course/35_public_sector_defence_humanitarian.md)
- **Sibling Topics:**
  - [Topic 01: Public Procurement Frameworks](01_public-procurement-frameworks.md)
  - [Topic 02: Whole-Life Cost and MEAT Evaluation](02_whole-life-cost-meat.md)
  - [Topic 03: Defence Availability and Performance-Based Logistics](03_defence-availability-pbl.md)
  - [Topic 04: Strategic Stockpiles](04_strategic-stockpiles.md)
  - [Topic 06: Pipeline and Pre-Positioning](06_pipeline-prepositioning.md)
  - [Topic 07: Last Mile in Fragile Settings](07_last-mile-fragile-settings.md)
  - [Topic 08: Cash versus In-Kind Assistance](08_cash-versus-in-kind.md)
- **Cross-Module References:**
  - [Module 11: Supply Chain Network Economics](../../02_methods/11_network_design/_index.md)
  - [Module 14: Network Design & Facility Location](../../02_methods/14_network_design_optimization/_index.md)
