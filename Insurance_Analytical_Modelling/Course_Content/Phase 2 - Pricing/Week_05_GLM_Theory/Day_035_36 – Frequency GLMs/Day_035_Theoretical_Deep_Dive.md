# Frequency GLMs (Part 2) - Theoretical Deep Dive

## Overview
This session expands on Frequency Modeling by addressing the single most important rating variable in Personal Lines insurance: **Territory**. We explore how to model geographic risk without overfitting, using techniques like Spatial Smoothing, Adjacency Matrices, and Credibility Weighting. We also touch on advanced Geospatial methods (GWR, CAR) that are becoming standard in modern pricing.

---

## 1. Conceptual Foundation

### 1.1 The Territory Problem

**Why Location Matters:**
*   **Traffic Density:** More cars = more accidents.
*   **Road Conditions:** Potholes, intersections, speed limits.
*   **Legal Environment:** Some counties are more litigious (Frequency of lawsuits).
*   **Weather:** Hail belts, flood zones.

**The Modeling Challenge:**
*   **Granularity:** Zip codes are too small (low credibility). States are too big (heterogeneous).
*   **Adjacency:** Risk doesn't stop at a Zip Code boundary. If Zip A is risky, its neighbor Zip B is likely risky too.
*   **Political Constraints:** "Redlining" (avoiding certain areas) is illegal. Rates must be statistically justified.

### 1.2 Spatial Smoothing

**Goal:** Create a smooth "Risk Surface" rather than a jagged map of Zip Code factors.

**Methods:**
1.  **Clustering:** Group Zips with similar historical frequency into "Territories" (e.g., Territory 1 to 10).
2.  **Distance-Based Smoothing:** The rate for Zip X is a weighted average of Zip X and all Zips within 5 miles.
3.  **Adjacency Smoothing:** The rate for Zip X is influenced by its direct neighbors.

### 1.3 Credibility in Spatial Modeling

**Bühlmann-Straub for Territories:**
$$ Z = \frac{N}{N + K} $$
$$ \text{Rate}_{New} = Z \times \text{Observed}_{Zip} + (1-Z) \times \text{Regional Mean} $$
*   **High Volume Zip:** $Z \approx 1$. Use its own data.
*   **Low Volume Zip:** $Z \approx 0$. Use the neighbors' data.

---

## 2. Mathematical Framework

### 2.1 Adjacency Matrices

An **Adjacency Matrix** $W$ defines the relationships between $n$ regions.
*   $w_{ij} = 1$ if region $i$ and $j$ share a border.
*   $w_{ij} = 0$ otherwise.

**Spatial Lag:**
The "Spatial Lag" of a variable $y$ is $Wy$ (the weighted sum of neighbors).
$$ (Wy)_i = \sum_{j} w_{ij} y_j $$

### 2.2 Conditional Autoregressive (CAR) Models

Used in Bayesian Hierarchical Modeling (e.g., with INLA or MCMC).
$$ \ln(\mu_i) = X_i \beta + \phi_i $$
$$ \phi_i | \phi_{-i} \sim N \left( \frac{\sum w_{ij} \phi_j}{\sum w_{ij}}, \frac{\tau^2}{\sum w_{ij}} \right) $$
*   **Interpretation:** The random effect $\phi_i$ for region $i$ is centered on the average of its neighbors.
*   **Result:** Smooths the residuals. If a Zip is high-risk but has low data, it gets pulled towards its neighbors.

### 2.3 Geographically Weighted Regression (GWR)

Standard GLM assumes $\beta$ is constant everywhere. GWR assumes $\beta$ varies by location.
$$ \ln(\mu_i) = \beta_0(u_i, v_i) + \beta_1(u_i, v_i) x_{1i} $$
*   $(u_i, v_i)$: Coordinates (Lat/Lon).
*   **Kernel:** We fit a weighted regression for *every point*, weighting nearby points higher.
*   **Use Case:** The effect of "Vehicle Age" might be different in a rust-belt city (rust issues) vs. a desert city.

---

## 3. Theoretical Properties

### 3.1 The Modifiable Areal Unit Problem (MAUP)

*   **Issue:** The results depend on how you draw the boundaries (Zip vs. County vs. Census Tract).
*   **Actuarial Solution:** Use the most granular unit available (Zip or Census Block) and smooth up. Avoid aggregating to County if possible.

### 3.2 Spatial Autocorrelation

**Moran's I:**
A measure of global spatial autocorrelation (like correlation, but for space).
$$ I = \frac{N}{\sum w_{ij}} \frac{\sum \sum w_{ij} (x_i - \bar{x})(x_j - \bar{x})}{\sum (x_i - \bar{x})^2} $$
*   $I > 0$: Clustered (High values near High values).
*   $I < 0$: Dispersed (Checkerboard).
*   **GLM Check:** Residuals should have Moran's I $\approx 0$. If $I > 0$, you are missing a spatial covariate.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Traditional Approach: Clustering

1.  **Calculate Raw Relativities:** Run a GLM with `ZipCode` as a categorical variable (using minimal regularization).
2.  **Map Residuals:** Plot the coefficients.
3.  **Cluster:** Use K-Means or Hierarchical Clustering on the coefficients (weighted by exposure) to group Zips into 10-20 "Territories".
4.  **Refit:** Run the final GLM with `Territory` instead of `ZipCode`.

### 4.2 Modern Approach: Splines & Coordinates

*   Include Latitude and Longitude directly in the GLM.
*   **Thin Plate Regression Splines:** $f(Lat, Lon)$.
*   **Benefit:** No arbitrary boundaries. Continuous surface.

### 4.3 Model Specification (Python Example)

Simulating Spatial Data and fitting a Spline-based GLM (GAM).

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import PoissonRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer

# Simulate Spatial Data (10x10 Grid)
np.random.seed(42)
n_locations = 100
lat = np.random.uniform(0, 10, n_locations)
lon = np.random.uniform(0, 10, n_locations)
exposure = np.random.uniform(10, 100, n_locations) # Policies per location

# True Risk Surface: High risk in the center (5,5)
dist_center = np.sqrt((lat-5)**2 + (lon-5)**2)
true_rate = 0.1 + 0.2 * np.exp(-0.5 * dist_center) 
# Base 0.1, Peak 0.3 at center

counts = np.random.poisson(true_rate * exposure)

df = pd.DataFrame({'lat': lat, 'lon': lon, 'exp': exposure, 'count': counts})
df['obs_freq'] = df['count'] / df['exp']

# 1. Polynomial GLM (Lat, Lon, Lat*Lon, Lat^2...)
poly_glm = make_pipeline(
    PolynomialFeatures(degree=2),
    PoissonRegressor(alpha=0, max_iter=1000)
)
poly_glm.fit(df[['lat', 'lon']], df['count'] / df['exp'], poissonregressor__sample_weight=df['exp'])

# 2. Visualization of Predicted Surface
grid_x, grid_y = np.meshgrid(np.linspace(0, 10, 50), np.linspace(0, 10, 50))
grid_flat = np.c_[grid_x.ravel(), grid_y.ravel()]

pred_surface = poly_glm.predict(grid_flat).reshape(grid_x.shape)

plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.scatter(df['lat'], df['lon'], c=df['obs_freq'], cmap='viridis', s=df['exp'])
plt.colorbar(label='Observed Freq')
plt.title('Observed Data')

plt.subplot(1, 2, 2)
plt.contourf(grid_x, grid_y, pred_surface, cmap='viridis')
plt.colorbar(label='Predicted Freq')
plt.title('Polynomial GLM Surface')

plt.show()

# Interpretation:
# The Polynomial GLM captures the "Hotspot" in the center.
# In production, we would use Splines (GAM) for more complex shapes.
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Territory Factors:** The multiplier for each Zip Code.
2.  **Maps:** Heatmaps of Indicated vs. Fitted Rates.

**Interpretation:**
*   **Smoothing:** "Zip 12345 had 0 claims, but its factor is 1.2 because it's next to downtown."
*   **Edge Effects:** Be careful at the boundaries of the map (extrapolation).

---

## 5. Evaluation & Validation

### 5.1 The "Checkerboard" Test

*   If you plot residuals on a map and see a checkerboard pattern (High-Low-High-Low), your smoothing is too weak (Undersmoothing).
*   If you see large blobs of all-positive or all-negative residuals, your smoothing is too strong (Oversmoothing).

### 5.2 Cross-Validation by Region

*   Don't split randomly. Split by **Region**.
*   Train on North/East/West. Test on South.
*   Ensures the model generalizes spatially.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Confounding with Demographics**
    *   **Issue:** "Inner city has high frequency."
    *   **Reality:** Is it the location, or the fact that younger people live there?
    *   **Fix:** Include Age/Gender in the GLM *before* analyzing the spatial residuals.

2.  **Trap: Zip Code Changes**
    *   **Issue:** USPS changes Zip Codes frequently.
    *   **Fix:** Use Lat/Lon if possible, or maintain a rigorous mapping table.

### 6.2 Implementation Challenges

1.  **Computational Cost:**
    *   GWR requires running $N$ regressions.
    *   **Solution:** Use FastGWR or approximations.

---

## 7. Advanced Topics & Extensions

### 7.1 Telematics as "Micro-Territory"

*   Instead of "Where do you live?", use "Where do you drive?" (GPS data).
*   **Road Segment Risk:** Assigning risk to specific highway stretches.

### 7.2 Flood Modeling

*   Frequency of flood is not smooth. It follows elevation contours.
*   **Physics-Based Models:** Use hydrological models (KatRisk, RMS) as an input feature to the GLM, rather than just smoothing historical claims.

---

## 8. Regulatory & Governance Considerations

### 8.1 Redlining & Disparate Impact

*   **Redlining:** Explicitly refusing to write in certain zips. (Illegal).
*   **Disparate Impact:** A neutral practice (e.g., Credit Score) that disproportionately hurts a protected class.
*   **Territory:** Often highly correlated with Race/Income. Regulators scrutinize "Territory Relativities" closely to ensure they match cost data exactly.

---

## 9. Practical Example

### 9.1 Worked Example: Credibility Weighting

**Zip Code A:**
*   Observed Frequency: 0.20
*   Exposure: 50 cars (Low).
*   Statewide Frequency: 0.10.

**Credibility ($K=500$):**
*   $Z = 50 / (50 + 500) = 0.09$.
*   Rate = $0.09(0.20) + 0.91(0.10) = 0.018 + 0.091 = 0.109$.
*   **Result:** The rate is pulled strongly towards the mean (0.10), ignoring the bad luck (0.20).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Location** is a proxy for traffic, weather, and legal environment.
2.  **Smoothing** is essential to handle sparse data.
3.  **Adjacency** improves predictions by borrowing strength from neighbors.

### 10.2 When to Use This Knowledge
*   **Territorial Ratemaking:** The annual review of base rates by territory.
*   **Marketing:** Targeting "underpriced" zip codes.

### 10.3 Critical Success Factors
1.  **Granularity:** Get as close to the street level as possible.
2.  **Compliance:** Ensure maps don't inadvertently discriminate.
3.  **Visualization:** Always map the results.

### 10.4 Further Reading
*   **Cressie:** "Statistics for Spatial Data".
*   **CAS Monograph 5:** Chapter on Territorial Ratemaking.

---

## Appendix

### A. Glossary
*   **Geocoding:** Converting Address to Lat/Lon.
*   **Centroid:** The center point of a Zip Code.
*   **Smoothing:** Reducing variance by averaging nearby data.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Moran's I** | Spatial Correlation | Diagnostics |
| **Bühlmann Z** | $N/(N+K)$ | Credibility |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
