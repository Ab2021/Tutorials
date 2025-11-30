# Climate Risk Modeling (Part 1) - Physical & Transition Risks - Theoretical Deep Dive

## Overview
"The 100-year flood is now happening every 10 years."
Climate Change is breaking the fundamental assumption of Insurance: **Stationarity**.
(The past is no longer a predictor of the future).
Insurers must move from **Backward-Looking** Actuarial Models to **Forward-Looking** Climate Models.

---

## 1. Conceptual Foundation

### 1.1 The Two Types of Climate Risk (TCFD)

1.  **Physical Risk:** Damage from the weather itself.
    *   *Acute:* Hurricanes, Wildfires, Floods.
    *   *Chronic:* Sea Level Rise, Heat Stress (Crop Failure).
2.  **Transition Risk:** Financial loss from the shift to a Low-Carbon Economy.
    *   *Policy:* Carbon Tax makes Coal Plants uninsurable.
    *   *Tech:* EVs replace ICE vehicles (Gas Stations become stranded assets).

### 1.2 Catastrophe (Cat) Modeling vs. Climate Modeling

*   **Cat Models (Traditional):**
    *   *Horizon:* 1 year (Renewable).
    *   *Data:* Historical (1900-2020).
    *   *Goal:* Solvency (Can we survive a Category 5 this year?).
*   **Climate Models (GCMs):**
    *   *Horizon:* 10-50 years.
    *   *Data:* Physics (Thermodynamics).
    *   *Goal:* Strategy (Should we exit the Florida market by 2030?).

---

## 2. Mathematical Framework

### 2.1 The RCP Scenarios (Representative Concentration Pathways)

*   **RCP 2.6:** Aggressive Mitigation (Paris Agreement). Temp +1.5°C.
*   **RCP 4.5:** Moderate Mitigation. Temp +2.5°C.
*   **RCP 8.5:** Business as Usual (High Emissions). Temp +4.5°C.
*   **Modeling Task:** Calculate the **Average Annual Loss (AAL)** for a portfolio under RCP 8.5 in the year 2050.

### 2.2 Vulnerability Functions

$$ \text{DamageRatio} = f(\text{HazardIntensity}, \text{BuildingCharacteristics}) $$

*   *Example (Wildfire):*
    *   If `DistanceToVegetation` < 10m AND `RoofMaterial` = Wood:
        *   Damage Ratio = 100% (Total Loss).
    *   If `DistanceToVegetation` > 30m AND `RoofMaterial` = Tile:
        *   Damage Ratio = 5%.

---

## 3. Theoretical Properties

### 3.1 Non-Stationarity

*   **Concept:** The probability distribution of events $P(X)$ changes over time $t$.
    $$ P(X_t) \neq P(X_{t-1}) $$
*   **Implication:** You cannot just take the average of the last 20 years. You must apply a **Climate Loading Factor**.
    *   *Premium* = *Base Rate* $\times$ (1 + *ClimateTrend*).

### 3.2 Correlation of Risks (Tail Dependence)

*   **Scenario:** A "Heat Dome" causes:
    1.  Wildfires (Property Claims).
    2.  Crop Failure (Ag Claims).
    3.  Power Grid Failure (Business Interruption).
    4.  Heat Stroke (Health/Life Claims).
*   **Result:** Diversification fails. Everything goes wrong at once.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Flood Risk Scoring (Python + GeoPandas)

```python
import geopandas as gpd
import rasterio

# 1. Load Portfolio (Lat/Lon)
properties = gpd.read_file("portfolio.geojson")

# 2. Load Flood Map (Raster - 100 Year Return Period)
# Value 1 = Flooded, 0 = Dry
flood_map = rasterio.open("flood_map_2050_rcp85.tif")

# 3. Sample Raster at Points
coords = [(x,y) for x, y in zip(properties.geometry.x, properties.geometry.y)]
properties['flood_depth'] = [x[0] for x in flood_map.sample(coords)]

# 4. Calculate Risk Score
def get_risk(depth):
    if depth > 1.0: return "Critical" # > 1 meter
    if depth > 0.1: return "High"
    return "Low"

properties['risk_level'] = properties['flood_depth'].apply(get_risk)
print(properties.groupby('risk_level').size())
```

### 4.2 Carbon Footprint Calculator (Transition Risk)

*   **Scope 3 Emissions:** Insurers are responsible for the emissions of the companies they insure (Insurance-Associated Emissions).
*   **Formula:**
    $$ \text{InsurerEmissions} = \sum \frac{\text{Premium}_i}{\text{CompanyRevenue}_i} \times \text{CompanyEmissions}_i $$

---

## 5. Evaluation & Validation

### 5.1 Stress Testing (CBES)

*   **Climate Biennial Exploratory Scenario (Bank of England):**
    *   *Scenario:* "Late Action" (Policy changes happen abruptly in 2030).
    *   *Test:* Does the insurer have enough capital to absorb the shock of stranded assets (Coal stocks crash) + Physical losses?

### 5.2 Model Blending

*   **Method:** Don't trust one model. Blend 3 (e.g., RMS, AIR, KatRisk).
*   **Why?** Climate science is uncertain. Ensembling reduces variance.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The "Protection Gap"**
    *   *Reality:* As risks become "Uninsurable" (e.g., California Wildfire), insurers withdraw.
    *   *Result:* The risk transfers to the State (Fair Plans) or the Homeowner.
    *   *Ethics:* Is it ethical to abandon a market?

2.  **Trap: False Precision**
    *   *Mistake:* "The flood depth will be exactly 1.24 meters in 2050."
    *   *Reality:* The uncertainty bounds are huge (+/- 50%). Use *ranges*, not points.

---

## 7. Advanced Topics & Extensions

### 7.1 Resilience Bonds

*   **Concept:** A Cat Bond where the principal is used to *build a sea wall* instead of just sitting in an escrow account.
*   **Benefit:** Reduces the risk *before* the disaster happens.

### 7.2 Parametric Climate Insurance

*   **Product:** Heat Stress Insurance for Dairy Cows.
*   **Trigger:** If Temp > 35°C for 3 days.
*   **Payout:** Farmer buys fans/misters for the cows.

---

## 8. Regulatory & Governance Considerations

### 8.1 TCFD Disclosures

*   **Requirement:** Task Force on Climate-related Financial Disclosures.
*   **Pillars:**
    1.  **Governance:** Does the Board oversee climate risk?
    2.  **Strategy:** What is the impact of a 2°C scenario?
    3.  **Risk Management:** How is climate integrated into underwriting?
    4.  **Metrics:** What is the Weighted Average Carbon Intensity (WACI)?

---

## 9. Practical Example

### 9.1 Worked Example: The "Wildfire Score"

**Scenario:** Insuring a Cabin in Lake Tahoe.
*   **Traditional Model:** "It's in a forest. Rate = High."
*   **Climate Model (Zesty.ai / Kettle):**
    *   **Vegetation Density:** High (Satellite View).
    *   **Slope:** Steep (Fire climbs fast).
    *   **Wind Corridor:** Yes.
    *   **Defensible Space:** No (Trees touch the roof).
*   **Decision:** Decline.
*   **Mitigation:** "If you cut the trees back 30 feet (Defensible Space), we will Quote."

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Physical Risk** destroys assets. **Transition Risk** destroys business models.
2.  **Stationarity is Dead.** History is a poor guide.
3.  **Granularity** (Address-level) is essential.

### 10.2 When to Use This Knowledge
*   **Chief Investment Officer:** "Should we divest from Oil & Gas bonds?"
*   **Underwriter:** "Why is this flood map different from FEMA's?" (FEMA is backward-looking).

### 10.3 Critical Success Factors
1.  **Data:** High-resolution satellite imagery.
2.  **Science:** Partnership with academia (Climate Scientists).

### 10.4 Further Reading
*   **Geneva Association:** "Climate Change and the Insurance Industry".

---

## Appendix

### A. Glossary
*   **RCP:** Representative Concentration Pathway.
*   **TCFD:** Task Force on Climate-related Financial Disclosures.
*   **AAL:** Average Annual Loss.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Emissions** | $\sum \frac{P}{R} \times E$ | Scope 3 |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
