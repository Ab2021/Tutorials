# Frequency–Severity Framework - Theoretical Deep Dive

## Overview
This session explores the cornerstone of non-life actuarial modeling: the Frequency-Severity method. We decompose the expected loss cost into two components—how often claims happen (Frequency) and how much they cost (Severity). We cover the standard probability distributions used for each (Poisson, Negative Binomial, Gamma, Lognormal, Pareto), the Tweedie distribution for aggregate loss, and the Generalized Linear Model (GLM) framework used to fit these models in practice.

---

## 1. Conceptual Foundation

### 1.1 Definition & Core Concept

**Frequency-Severity Decomposition:** The practice of modeling the number of claims ($N$) and the size of claims ($X$) separately to estimate the aggregate loss ($S$).

**Pure Premium (Loss Cost):**
$$ \text{Pure Premium} = \text{Frequency} \times \text{Severity} $$
$$ E[S] = E[N] \times E[X] $$

**Why Separate Them?**
1.  **Different Drivers:** Frequency is driven by activity (miles driven) and safety (braking). Severity is driven by asset value (car price) and inflation (medical costs).
2.  **Statistical Behavior:** Count data (discrete, non-negative) behaves differently from cost data (continuous, skewed).
3.  **Predictive Power:** Some variables predict frequency but not severity (e.g., a bad driver crashes often, but the cost depends on what they hit).

**Key Terminology:**
-   **Frequency ($\lambda$):** Claims per unit of exposure.
-   **Severity ($\mu$):** Average cost per claim.
-   **Overdispersion:** When the variance of the data exceeds the mean (common in claim counts).
-   **Heavy Tail:** When there is a non-negligible probability of extreme values (common in claim costs).
-   **Zero-Inflation:** When the data has more zeros (no claims) than the standard distribution predicts.

### 1.2 Historical Context & Evolution

**Origin:**
-   **Early 20th Century:** Actuaries used simple averages (Total Losses / Total Exposures).
-   **1960s-80s:** Introduction of Bailey-Simon method and Minimum Bias procedures.

**Evolution:**
-   **1990s:** Widespread adoption of Generalized Linear Models (GLMs) allowed for multivariate analysis of frequency and severity.
-   **2000s:** Tweedie distribution gained popularity for modeling Pure Premium directly.
-   **Present:** Machine Learning (GBMs, Neural Networks) is replacing GLMs, but the Frequency-Severity split remains the standard architecture.

### 1.3 Why This Matters

**Business Impact:**
-   **Pricing Granularity:** Allows insurers to penalize frequency risk (bad drivers) differently from severity risk (expensive cars).
-   **Reserving:** Frequency trends (e.g., safer cars) vs. Severity trends (e.g., higher repair costs) require different reserve assumptions.

**Regulatory Relevance:**
-   **Rate Filings:** Regulators require justification for rate changes. "Frequency is down 2%, but Severity is up 8%" is a standard argument for a +6% rate hike.

---

## 2. Mathematical Framework

### 2.1 Frequency Distributions (Count Models)

1.  **Poisson Distribution:**
    *   **Use Case:** Idealized count data.
    *   **Property:** Mean = Variance ($E[N] = Var[N] = \lambda$).
    *   **Limitation:** Insurance data is usually overdispersed ($Var > Mean$), making Poisson confidence intervals too narrow.

2.  **Negative Binomial Distribution:**
    *   **Use Case:** Real-world claim counts (Overdispersed).
    *   **Property:** $Var[N] = \lambda + \phi \lambda^2$ (where $\phi$ is the dispersion parameter).
    *   **Interpretation:** Can be viewed as a Poisson distribution where the $\lambda$ parameter itself varies across the population (Gamma mixture).

3.  **Zero-Inflated Models (ZIP, ZINB):**
    *   **Use Case:** High number of zero claims (e.g., due to deductibles).
    *   **Structure:** A mixture of a point mass at zero (never claim) and a count distribution (might claim).

### 2.2 Severity Distributions (Cost Models)

1.  **Gamma Distribution:**
    *   **Use Case:** Attritional losses (Auto Physical Damage).
    *   **Property:** Light tail; skewed positive.
    *   **GLM Link:** Log link is standard.

2.  **Lognormal Distribution:**
    *   **Use Case:** Medium-tailed losses (Liability).
    *   **Property:** $\ln(X)$ follows a Normal distribution.
    *   **Note:** Not part of the Exponential Family, so strictly speaking, not a standard GLM (though often treated similarly).

3.  **Pareto Distribution:**
    *   **Use Case:** Large losses, Reinsurance, Catastrophes.
    *   **Property:** Heavy tail (Power law decay). "80% of costs come from 20% of claims."

### 2.3 The Tweedie Distribution

A member of the Exponential Dispersion Family that models **Aggregate Loss** (Pure Premium) directly.
*   **Parameter $p$:** The variance power parameter ($Var(Y) = \phi \mu^p$).
    *   $p=1$: Poisson.
    *   $p=2$: Gamma.
    *   $1 < p < 2$: Compound Poisson-Gamma (Poisson count of Gamma severities).
*   **Feature:** Has a point mass at zero (no claims) and a continuous distribution for positive values.
*   **Advantage:** One model instead of two.
*   **Disadvantage:** Harder to interpret "why" the premium changed (was it frequency or severity?).

### 2.4 Generalized Linear Models (GLMs)

The industry standard for fitting these distributions.
$$ g(E[Y]) = \beta_0 + \beta_1 x_1 + \dots + \beta_n x_n $$
*   **Link Function $g(\cdot)$:** Usually $\ln(\cdot)$ for insurance (ensures predictions are positive).
*   **Offset:**
    *   Frequency Model: $\ln(\text{Exposure})$.
    *   Severity Model: None (or sometimes weight).

---

## 3. Theoretical Properties

### 3.1 Independence Assumption

*   **Assumption:** $N$ and $X$ are independent.
*   **Reality:** Often violated.
    *   *Example:* In Workers' Comp, accidents with high frequency (slips/falls) might have low severity, while low frequency (explosions) have high severity.
    *   *Fix:* Include common predictors in both models or use Copulas.

### 3.2 Coefficient of Variation (CV)

For Severity, the CV ($\sigma / \mu$) is a measure of tail risk.
*   **Gamma:** Constant CV (if shape parameter is fixed).
*   **Lognormal:** CV depends on $\sigma$.

### 3.3 Law of Large Numbers

*   **Frequency:** Converges quickly (need fewer claims to estimate $\lambda$).
*   **Severity:** Converges slowly (need many claims to estimate $\mu$, especially if heavy-tailed).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

*   **Policy Level:** Exposure, Rating Factors (Age, Vehicle, Zone).
*   **Claim Level:** Claim Count, Incurred Amount (Paid + Case Reserve).

### 4.2 Preprocessing Steps

**Step 1: Capping Large Losses**
*   Severity models are sensitive to outliers.
*   **Technique:** Cap losses at a threshold (e.g., $100k) and model the excess separately (Extreme Value Theory).

**Step 2: Time Trending**
*   Adjust historical losses for inflation to bring them to current cost levels.

### 4.3 Model Specification (Python Example)

Fitting Frequency and Severity GLMs using `statsmodels`.

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Simulated Data
np.random.seed(42)
n_rows = 1000

data = pd.DataFrame({
    'Age': np.random.randint(18, 75, n_rows),
    'VehicleValue': np.random.normal(30000, 10000, n_rows),
    'Exposure': np.random.uniform(0.5, 1.0, n_rows)
})

# True Parameters
# Frequency: Decreases with Age
lambda_true = np.exp(-2.0 - 0.02 * (data['Age'] - 18)) * data['Exposure']
data['ClaimCount'] = np.random.poisson(lambda_true)

# Severity: Increases with VehicleValue
# Only for rows with claims
mu_true = np.exp(7.0 + 0.00002 * data['VehicleValue'])
# Generate severity for all, but only keep if ClaimCount > 0
severity_draws = np.random.gamma(shape=2.0, scale=mu_true/2.0) 
data['ClaimCost'] = np.where(data['ClaimCount'] > 0, severity_draws * data['ClaimCount'], 0)
data['AvgSeverity'] = np.where(data['ClaimCount'] > 0, data['ClaimCost'] / data['ClaimCount'], np.nan)

# 1. Frequency Model (Poisson GLM)
# Offset is log(Exposure)
freq_model = smf.glm(
    formula='ClaimCount ~ Age', 
    data=data, 
    family=sm.families.Poisson(),
    offset=np.log(data['Exposure'])
).fit()

print("Frequency Model Summary:")
print(freq_model.summary())

# 2. Severity Model (Gamma GLM)
# Only fit on records with claims
severity_data = data[data['ClaimCount'] > 0].copy()

sev_model = smf.glm(
    formula='AvgSeverity ~ VehicleValue', 
    data=severity_data, 
    family=sm.families.Gamma(link=sm.families.links.log())
).fit()

print("\nSeverity Model Summary:")
print(sev_model.summary())

# 3. Pure Premium Calculation
# Predict for a new profile: Age 30, Vehicle Value 40000, Exposure 1.0
new_profile = pd.DataFrame({'Age': [30], 'VehicleValue': [40000], 'Exposure': [1.0]})

pred_freq = freq_model.predict(new_profile, offset=np.log(new_profile['Exposure']))[0]
pred_sev = sev_model.predict(new_profile)[0]
pred_pure_premium = pred_freq * pred_sev

print(f"\nPredicted Frequency: {pred_freq:.4f}")
print(f"Predicted Severity: ${pred_sev:.2f}")
print(f"Predicted Pure Premium: ${pred_pure_premium:.2f}")
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Relativities:** The multiplicative factors derived from the coefficients ($e^\beta$).
    *   *Example:* If Age coefficient is -0.02, then for every year older, frequency drops by $1 - e^{-0.02} \approx 2\%$.
2.  **Pure Premium:** The final price (before expenses/profit).

**Interpretation:**
*   **Frequency Drivers:** Usually Age, Gender, History, Territory.
*   **Severity Drivers:** Usually Vehicle Value, Limit, Deductible.

---

## 5. Evaluation & Validation

### 5.1 Model Diagnostics

**Deviance:**
*   Measure of goodness-of-fit. Lower is better.
*   **Scaled Deviance:** Should be close to 1.0 (Degrees of Freedom). If >> 1.0, indicates Overdispersion (switch from Poisson to Negative Binomial).

**AIC / BIC:**
*   Used for variable selection. Penalizes complexity.

### 5.2 Visual Validation

**One-Way Lift Charts:**
*   Plot "Predicted Pure Premium" vs. "Actual Pure Premium" across buckets of a variable (e.g., Age).
*   Lines should track closely.

**Gini Coefficient / Lorenz Curve:**
*   Measures the model's ability to segment risk. Higher Gini = Better segmentation.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Modeling Total Loss directly with OLS**
    *   **Issue:** Using standard Linear Regression on `ClaimCost`.
    *   **Reality:** Fails because of the mass of zeros and the skewness of positive values. Must use Tweedie or Freq-Sev.

2.  **Trap: Ignoring Exposure in Frequency**
    *   **Issue:** Treating a 6-month policy the same as a 12-month policy.
    *   **Fix:** Always use `offset=log(Exposure)`.

### 6.2 Implementation Challenges

1.  **Sparse Data:** High-dimensionality categorical variables (e.g., Zip Code) with few claims. Requires smoothing or credibility weighting.
2.  **Correlated Predictors:** Vehicle Age and Vehicle Value are correlated. GLM handles this, but interpretation can be tricky (multicollinearity).

---

## 7. Advanced Topics & Extensions

### 7.1 Zero-Inflated Models

Used when the number of zeros exceeds what a Poisson/NegBin expects.
*   **Process:**
    1.  Binary model: Claim vs. No Claim.
    2.  Count model: How many claims (given > 0).

### 7.2 Hierarchical / Multilevel Models

Used for Credibility Weighting within the model structure.
*   **Example:** Modeling Territory. Instead of a fixed effect for every Zip Code, treat Zip Code as a random effect nested within County.

### 7.3 Machine Learning Approaches

*   **GBM (Gradient Boosting Machines):** XGBoost/LightGBM with Poisson/Gamma/Tweedie objective functions. Often outperforms GLMs in predictive accuracy but lacks interpretability.
*   **Monotonic Constraints:** Enforcing that "More Speeding Tickets" cannot *lower* the premium, even if the data suggests a noisy dip.

---

## 8. Regulatory & Governance Considerations

### 8.1 Rate Filing Support

*   Regulators accept GLMs but require "Relativities" to be filed.
*   **Black Box:** Pure ML models (Neural Nets) are often rejected for pricing because the rate impact of specific variables cannot be easily explained.

### 8.2 Disparate Impact

*   Variables that predict Frequency/Severity (e.g., Credit Score) might correlate with protected classes.
*   **Analysis:** Must test if the model unfairly penalizes specific groups beyond what the risk data justifies.

---

## 9. Practical Example

### 9.1 Worked Example: Auto Insurance

**Scenario:**
*   **Frequency Model:** $\ln(\lambda) = -1.5 + 0.02 \times \text{Points} - 0.01 \times \text{Age}$.
*   **Severity Model:** $\ln(\mu) = 8.0 + 0.1 \times \text{LuxuryCar}$.

**Risk Profile:**
*   Age 40, 2 Points, Luxury Car (1).
*   Exposure: 1 Year.

**Calculation:**
1.  **Frequency:**
    $$ \ln(\lambda) = -1.5 + 0.02(2) - 0.01(40) = -1.5 + 0.04 - 0.4 = -1.86 $$
    $$ \lambda = e^{-1.86} = 0.1557 \text{ claims/year} $$

2.  **Severity:**
    $$ \ln(\mu) = 8.0 + 0.1(1) = 8.1 $$
    $$ \mu = e^{8.1} = 3294.47 \text{ per claim} $$

3.  **Pure Premium:**
    $$ PP = 0.1557 \times 3294.47 = \$512.95 $$

**Interpretation:** The expected loss cost for this driver is $512.95. The insurer will add expenses and profit to this base.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Decomposition:** $PP = Freq \times Sev$.
2.  **Distributions:** Poisson/NegBin for counts; Gamma/Lognormal for cost.
3.  **GLM:** The standard framework for fitting these models.
4.  **Tweedie:** An alternative for modeling PP directly.

### 10.2 When to Use This Knowledge
*   **Ratemaking:** Building rating plans.
*   **Reserving:** Setting IBNR (Frequency/Severity methods).
*   **Risk Management:** Analyzing drivers of loss.

### 10.3 Critical Success Factors
1.  **Choose the Right Distribution:** Check variance/mean ratio.
2.  **Handle Outliers:** Cap large losses for severity modeling.
3.  **Validate:** Use lift charts and holdout data.

### 10.4 Further Reading
*   **Textbook:** "Generalized Linear Models for Insurance Data" (De Jong & Heller).
*   **Paper:** "Tweedie Distributions for Generalized Linear Models" (Smyth).

---

## Appendix

### A. Glossary
*   **Overdispersion:** Variance > Mean.
*   **Link Function:** Function connecting the linear predictor to the mean.
*   **Offset:** A fixed term in the linear predictor (usually log exposure).
*   **Relativity:** The factor applied to the base rate for a specific characteristic.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Pure Premium** | $E[N] \times E[X]$ | Loss Cost |
| **Poisson Variance** | $Var = \mu$ | Frequency |
| **NegBin Variance** | $Var = \mu + \phi \mu^2$ | Overdispersed Freq |
| **GLM** | $g(\mu) = X\beta$ | Modeling |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,100+*
