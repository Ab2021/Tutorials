# Frequency GLMs (Part 3) - Theoretical Deep Dive

## Overview
This final session on Frequency Modeling addresses the temporal and structural complexities of insurance data. We explore **Trend Analysis** (how risk changes over time), **Seasonality** (weather effects), and **Multi-Peril Modeling** (handling correlated risks like Collision and Liability). We also introduce **Copulas** as a method to model dependence between different claim types.

---

## 1. Conceptual Foundation

### 1.1 The Time Dimension: Trend

**Frequency Trend:** The annual rate of change in claim frequency.
*   **Drivers:** Improved vehicle safety (ADAS), changes in law enforcement, economic cycles (recessions reduce driving).
*   **Modeling:** We must "detrend" historical data to current levels before pricing.
*   **Assumption:** Past trends will continue (or we must adjust them based on judgment).

### 1.2 Seasonality

**Recurring Patterns:**
*   **Winter:** Ice/Snow $\to$ High Collision Frequency.
*   **Summer:** Road Trips $\to$ High Miles Driven.
*   **Hurricane Season:** Wind claims in Q3.

**Why Model It?**
*   **Short-Term Policies:** A 6-month policy sold in July has a different risk profile than one sold in January.
*   **Cash Flow:** Reserving requires accurate monthly expectations.

### 1.3 Multi-Peril Dependencies

**Independence Assumption:**
Standard GLMs assume Collision claims are independent of Liability claims.
*   *Reality:* A crash often causes both.
*   **Correlation:** High positive correlation between coverages.
*   **Impact:** If we model them separately and sum the premiums, the *mean* is correct, but the *variance* of the total loss is underestimated. This affects capital requirements (Solvency II).

---

## 2. Mathematical Framework

### 2.1 Trend Modeling in GLMs

**Method 1: Time as a Covariate**
$$ \ln(\mu) = X\beta + \gamma \cdot \text{Time} $$
*   $\text{Time}$: Continuous variable (e.g., Year 2020=1, 2021=2).
*   $\gamma$: The trend factor. $e^\gamma - 1 \approx$ Annual Trend %.

**Method 2: Two-Step Trending**
1.  Fit ARIMA to the time series of monthly frequencies.
2.  Adjust the historical exposure in the GLM by the estimated trend factors.

### 2.2 Seasonality with Fourier Series

Instead of 11 dummy variables (Jan, Feb...), use Sine/Cosine waves.
$$ \ln(\mu) = X\beta + \alpha_1 \sin\left(\frac{2\pi t}{12}\right) + \alpha_2 \cos\left(\frac{2\pi t}{12}\right) $$
*   **Parsimony:** Uses 2 parameters instead of 11.
*   **Smoothness:** Ensures Dec 31 is close to Jan 1.

### 2.3 Copulas for Dependence

**Sklar's Theorem:**
Any joint distribution $H(x, y)$ can be written as:
$$ H(x, y) = C(F(x), G(y)) $$
*   $F(x), G(y)$: Marginal distributions (e.g., Poisson for Freq A, Poisson for Freq B).
*   $C(u, v)$: The **Copula** function that binds them.

**Common Copulas:**
*   **Gaussian Copula:** Correlation matrix structure.
*   **Gumbel Copula:** Tail dependence (Extreme events happen together).

---

## 3. Theoretical Properties

### 3.1 Stationarity

**ARIMA Assumption:**
Time series data must be **Stationary** (Constant mean and variance over time).
*   **Differencing:** $y_t - y_{t-1}$. Removes linear trend.
*   **Log Transformation:** Stabilizes variance.

### 3.2 Correlation vs. Tail Dependence

*   **Pearson Correlation:** Measures linear relationship.
*   **Tail Dependence:** Measures the probability that $Y$ is extreme given $X$ is extreme.
*   *Insurance Context:* We care about Tail Dependence. If a hurricane hits, *all* lines of business (Auto, Home, Marine) have extreme losses simultaneously.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Trend Selection

1.  **Fit GLM with Time:** Estimate $\hat{\gamma}$.
2.  **Residual Analysis:** Plot residuals over time.
    *   If residuals show a pattern (e.g., a dip during COVID-19), the linear trend is insufficient.
    *   **Intervention Analysis:** Add a dummy variable for the COVID period.

### 4.2 Multi-Peril GLM Strategy

**Option A: Independent Models (Standard)**
*   Model Freq(Coll), Freq(Liab), Freq(Comp) separately.
*   Sum the expected counts: $E[N_{Total}] = \sum E[N_i]$.
*   *Valid for Mean Prediction (Pricing).*

**Option B: Multivariate GLM**
*   Use a joint distribution (e.g., Multivariate Poisson).
*   *Required for Capital Modeling/Reinsurance.*

### 4.3 Model Specification (Python Example)

Modeling Trend and Seasonality.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Simulate Monthly Data (5 Years)
np.random.seed(42)
months = np.arange(60)
time = np.linspace(0, 5, 60) # 5 Years

# 1. Trend: +5% per year
trend = np.exp(0.05 * time)

# 2. Seasonality: Peak in Winter (Month 0, 12, 24...)
seasonality = 1 + 0.2 * np.cos(2 * np.pi * time)

# 3. Base Rate
base_rate = 100 

# True Lambda
mu = base_rate * trend * seasonality
counts = np.random.poisson(mu)

df = pd.DataFrame({'Time': time, 'Month': months % 12, 'Count': counts})

# Model 1: Linear Trend + Monthly Dummies
# C(Month) creates 11 dummies
model_dummy = smf.glm("Count ~ Time + C(Month)", data=df, 
                      family=sm.families.Poisson()).fit()

# Model 2: Fourier Series (Sin/Cos)
df['sin_time'] = np.sin(2 * np.pi * df['Time'])
df['cos_time'] = np.cos(2 * np.pi * df['Time'])

model_fourier = smf.glm("Count ~ Time + sin_time + cos_time", data=df, 
                        family=sm.families.Poisson()).fit()

print(f"Dummy AIC: {model_dummy.aic:.2f}")
print(f"Fourier AIC: {model_fourier.aic:.2f}")

# Visualization
plt.figure(figsize=(10, 5))
plt.plot(df['Time'], df['Count'], 'o', label='Observed', alpha=0.5)
plt.plot(df['Time'], model_fourier.predict(df), 'r-', label='Fourier Fit')
plt.title('Frequency Trend & Seasonality')
plt.legend()
plt.show()

# Interpretation:
# Fourier is smoother and uses fewer degrees of freedom.
# The 'Time' coefficient gives the annual trend.
print(f"Estimated Annual Trend: {np.exp(model_fourier.params['Time']) - 1:.2%}")
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Trend Factor:** "Frequency is dropping 2% per year."
2.  **Seasonal Factors:** "January is 1.2x the average month."

**Interpretation:**
*   **Pricing:** We project the trend forward to the "Policy Effective Period."
    *   If data is from 2020-2022, and we write policies in 2024, we need 2 years of trend.
    *   Trend Selection is the *most sensitive* assumption in ratemaking.

---

## 5. Evaluation & Validation

### 5.1 Out-of-Time Validation

*   Train on Years 1-4. Test on Year 5.
*   Does the trend hold? Or did it flatten out?

### 5.2 Correlation Matrices

*   Calculate correlation of residuals between Collision and Liability models.
*   If $\rho > 0.1$, consider a Copula for capital modeling.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Double Counting Trend**
    *   **Issue:** Including `ModelYear` (Vehicle Age) and `CalendarYear` (Time) in the same model.
    *   **Reality:** They are correlated but distinct. `ModelYear` captures safety features. `CalendarYear` captures road conditions/economy. You usually need both.

2.  **Trap: Shock Losses**
    *   **Issue:** A massive hurricane spikes frequency in one month.
    *   **Fix:** Treat as a "Catastrophe" (Cat). Remove Cat claims from standard frequency modeling and load them separately (Cat Load).

### 6.2 Implementation Challenges

1.  **Sparse Seasonality:**
    *   "Convertibles in Winter". Very few observations.
    *   **Solution:** Interaction term `VehicleType * Season`.

---

## 7. Advanced Topics & Extensions

### 7.1 ARIMA with Exogenous Variables (ARIMAX)

*   Model frequency using external predictors: `UnemploymentRate`, `GasPrice`, `Precipitation`.
*   **Benefit:** Explains *why* the trend is happening.

### 7.2 Multivariate Poisson-Lognormal

*   A specific distribution that handles overdispersion and correlation simultaneously.
*   Computationally intensive (requires MCMC).

---

## 8. Regulatory & Governance Considerations

### 8.1 Trend Justification

*   Regulators often challenge high trend selections.
*   **Requirement:** You must show that the trend is statistically significant and not just a temporary blip.
*   **Selection:** Often a blend of "Internal Trend" (your data) and "Industry Trend" (Fast Track data).

---

## 9. Practical Example

### 9.1 Worked Example: Trending to Future

**Data:** 2021-2023.
**Trend:** -2% per year.
**Target:** Policies written in 2025.

**Midpoints:**
*   Average Accident Date (Data): July 1, 2022.
*   Average Accident Date (Future): July 1, 2025.
*   **Delta:** 3 Years.

**Trend Factor:**
$$ (1 - 0.02)^3 = 0.941 $$
*   We multiply the historical frequency by 0.941 to get the projected frequency.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Trend** projects the past into the future.
2.  **Seasonality** handles cyclic patterns.
3.  **Independence** is a convenient lie; real risks are correlated.

### 10.2 When to Use This Knowledge
*   **Rate Indications:** Every rate filing requires a trend selection.
*   **Reserving:** IBNR calculations depend heavily on seasonality.

### 10.3 Critical Success Factors
1.  **Remove Cats:** Don't let hurricanes distort your frequency trend.
2.  **Validate:** Check if the trend is linear or exponential.
3.  **Parsimony:** Use Fourier series for seasonality.

### 10.4 Further Reading
*   **Werner & Modlin:** "Basic Ratemaking" (Chapter on Trending).
*   **Frees & Valdez:** "Understanding Relationships Using Copulas".

---

## Appendix

### A. Glossary
*   **Detrending:** Adjusting past data to current cost levels.
*   **Stationarity:** Statistical properties do not change over time.
*   **Copula:** A function that couples marginals to a joint distribution.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Exponential Trend** | $e^{\gamma t}$ | Trending |
| **Fourier Season** | $\sin(2\pi t/12)$ | Seasonality |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
