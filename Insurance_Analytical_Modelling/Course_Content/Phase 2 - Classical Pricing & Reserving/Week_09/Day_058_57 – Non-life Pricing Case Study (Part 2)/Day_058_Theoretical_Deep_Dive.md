# Non-life Pricing Case Study (Part 2) - Theoretical Deep Dive

## Overview
The "One Rate Fits All" era is dead. Modern pricing uses **Generalized Linear Models (GLMs)** to tailor the rate to the individual risk. We model **Frequency** (Poisson) and **Severity** (Gamma) separately, then combine them to get the Pure Premium. This session covers the math, the code, and the art of **Feature Selection**.

---

## 1. Conceptual Foundation

### 1.1 Why GLMs?

*   **Linear Regression (OLS):** Assumes $Y \sim \text{Normal}$.
    *   *Problem:* Claims are non-negative and skewed. OLS can predict negative premiums.
*   **GLM:** Allows $Y$ to follow any Exponential Family distribution.
    *   *Link Function:* Connects the linear predictor $X\beta$ to the mean $\mu$.

### 1.2 The Frequency-Severity Split

Instead of modeling Pure Premium directly (Tweedie), we often split it:
1.  **Frequency Model:** Predicts claim count per exposure.
    *   *Distribution:* Poisson (or Negative Binomial).
    *   *Link:* Log.
2.  **Severity Model:** Predicts average cost per claim.
    *   *Distribution:* Gamma.
    *   *Link:* Log.
3.  **Pure Premium:** $E[PP] = E[Freq] \times E[Sev]$.

### 1.3 Link Functions

*   **Log Link:** $g(\mu) = \ln(\mu) = X\beta \implies \mu = e^{X\beta}$.
    *   *Multiplicative Structure:* Rate = Base $\times$ Factor1 $\times$ Factor2.
    *   This matches standard insurance rating plans.

---

## 2. Mathematical Framework

### 2.1 Poisson Regression (Frequency)

$$ P(Y=k) = \frac{e^{-\lambda} \lambda^k}{k!} $$
*   $\lambda = \text{Exposure} \times e^{X\beta}$.
*   **Offset:** We use $\ln(\text{Exposure})$ as an offset term in the linear predictor to handle varying exposure periods (e.g., 6 months vs. 12 months).

### 2.2 Gamma Regression (Severity)

$$ f(y) = \frac{1}{\Gamma(k)\theta^k} y^{k-1} e^{-y/\theta} $$
*   Mean $\mu = k\theta$. Variance $V = \mu^2 / k$.
*   **Constant Coefficient of Variation:** Gamma assumes the standard deviation grows linearly with the mean. (Big claims vary more).

### 2.3 Tweedie (Pure Premium)

*   Models the mass at zero (no claims) and the continuous positive distribution (claims) simultaneously.
*   $Y \sim Tw_p(\mu, \phi)$.
*   $1 < p < 2$.
*   *Advantage:* One model instead of two.
*   *Disadvantage:* Harder to interpret "Frequency drivers" vs "Severity drivers".

---

## 3. Theoretical Properties

### 3.1 Deviance

*   **OLS:** Minimizes Sum of Squared Errors (SSE).
*   **GLM:** Minimizes **Deviance** (Difference between Log-Likelihood of saturated model and current model).
*   **AIC/BIC:** Used for model selection (penalizing complexity).

### 3.2 Overdispersion

*   **Poisson Assumption:** Mean = Variance.
*   **Reality:** Variance > Mean (Overdispersion).
*   **Fix:** Use **Negative Binomial** or **Quasi-Poisson**.
    *   *Quasi-Poisson:* Estimates a dispersion parameter $\phi$ to correct the standard errors.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Fitting GLMs in Python (statsmodels)

```python
import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Data: Policy Level
# ClaimCount, ClaimAmount, Exposure, Age, VehicleType
data = pd.DataFrame({
    'ClaimCount': [0, 1, 0, 2, 0],
    'ClaimAmount': [0, 5000, 0, 12000, 0],
    'Exposure': [1.0, 0.5, 1.0, 1.0, 0.8],
    'Age': [25, 40, 30, 50, 22],
    'VehicleType': ['Sedan', 'SUV', 'Sedan', 'Truck', 'Sedan']
})

# 1. Frequency Model (Poisson)
# Offset: log(Exposure)
# Formula: ClaimCount ~ Age + VehicleType
freq_model = smf.glm(
    'ClaimCount ~ Age + VehicleType',
    data=data,
    offset=np.log(data['Exposure']),
    family=sm.families.Poisson(link=sm.families.links.log())
).fit()

print("Frequency Model Summary:")
print(freq_model.summary())

# 2. Severity Model (Gamma)
# Only train on claims > 0
severity_data = data[data['ClaimAmount'] > 0].copy()
severity_data['AvgClaim'] = severity_data['ClaimAmount'] / severity_data['ClaimCount']

# Weights: ClaimCount (More claims = more precision on the average)
sev_model = smf.glm(
    'AvgClaim ~ Age + VehicleType',
    data=severity_data,
    family=sm.families.Gamma(link=sm.families.links.log()),
    freq_weights=severity_data['ClaimCount']
).fit()

print("\nSeverity Model Summary:")
print(sev_model.summary())

# 3. Prediction
# New Policy
new_policy = pd.DataFrame({'Age': [35], 'VehicleType': ['SUV'], 'Exposure': [1.0]})

pred_freq = freq_model.predict(new_policy, offset=np.log(new_policy['Exposure']))
pred_sev = sev_model.predict(new_policy)
pure_premium = pred_freq * pred_sev

print(f"\nPredicted Pure Premium: ${pure_premium[0]:.2f}")
```

### 4.2 Feature Selection (One-Way Analysis)

*   Before modeling, we check the univariate relationship.
*   **Plot:** Average Frequency vs. Age.
*   **Binning:** Group Age into buckets (16-20, 21-25, etc.) to see non-linearities.

```python
# One-Way Analysis Helper
def one_way_summary(df, feature, target, exposure):
    summary = df.groupby(feature).agg({
        target: 'sum',
        exposure: 'sum'
    })
    summary['Observed'] = summary[target] / summary[exposure]
    return summary

# Example usage
# one_way_summary(data, 'AgeBucket', 'ClaimCount', 'Exposure')
```

---

## 5. Evaluation & Validation

### 5.1 Lift Charts (Gini)

*   Sort policies by Predicted Pure Premium (Low to High).
*   Group into 10 deciles.
*   Plot **Predicted Loss Ratio** vs. **Actual Loss Ratio**.
*   **Good Model:** Steep slope. (Identifies low risk and high risk correctly).
*   **Bad Model:** Flat line. (Predictions are random).

### 5.2 Double Lift Charts

*   Compare Model A (Old) vs. Model B (New).
*   Sort by the *ratio* of Model B / Model A.
*   If Model B is better, the actual loss ratio should track Model B's predictions in the sorted buckets.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Modeling Total Loss directly**
    *   **Issue:** Using `ClaimAmount` as target in Poisson.
    *   **Reality:** Poisson is for *integers*. Use Tweedie for total loss, or split Freq/Sev.

2.  **Trap: Correlation of Features**
    *   **Issue:** Age and YearsLicensed are 99% correlated.
    *   **Result:** Coefficients blow up (Multicollinearity).
    *   **Fix:** Drop one, or use PCA, or Regularization (Elastic Net).

### 6.2 Implementation Challenges

1.  **Categorical Levels:**
    *   Zip Code has 40,000 levels.
    *   **Solution:** Target Encoding (replace Zip with historical loss cost of that Zip) or Spatial Smoothing (Credibility).

---

## 7. Advanced Topics & Extensions

### 7.1 GAMs (Generalized Additive Models)

*   GLM assumes linearity: $\beta \cdot \text{Age}$.
*   **GAM:** Allows non-linear splines: $f(\text{Age})$.
*   *Benefit:* Captures the "U-shape" of Age (High risk for young and old) automatically without manual binning.

### 7.2 GBMs (Gradient Boosting)

*   XGBoost/LightGBM.
*   Often used as a benchmark or to find interactions (e.g., Age * VehicleType).
*   *Regulatory:* Harder to explain, but getting accepted ("Monotonic Constraints").

---

## 8. Regulatory & Governance Considerations

### 8.1 Fairness & Bias

*   **Proxy Discrimination:** Using "Credit Score" might be a proxy for Race.
*   **Disparate Impact:** Even if the variable is neutral (Zip Code), does it result in higher rates for protected classes?
*   **Testing:** You must run bias tests on the final rates.

---

## 9. Practical Example

### 9.1 Worked Example: The "Red Car" Myth

**Scenario:**
*   Data shows Red Cars have higher frequency.
*   **GLM:** When we control for "Age" and "Vehicle Type" (Sports Car), the "Color" coefficient becomes insignificant.
*   **Conclusion:** Red cars don't cause accidents. Young people drive red sports cars.
*   **Action:** Do not include Color in the rating plan.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Poisson** for Frequency, **Gamma** for Severity.
2.  **Log Link** creates multiplicative rates.
3.  **Offset** handles exposure duration.

### 10.2 When to Use This Knowledge
*   **Pricing:** Building the core rating algorithm.
*   **Underwriting:** Identifying bad segments.

### 10.3 Critical Success Factors
1.  **Clean Data:** Outliers in Severity (one \$1M claim) can ruin the Gamma model. Cap large losses!
2.  **Validate:** Use Lift Charts to prove the model works.
3.  **Explain:** Can you explain to a regulator why "Credit Score" predicts risk?

### 10.4 Further Reading
*   **Goldburd et al.:** "Generalized Linear Models for Insurance Rating" (CAS Monograph 5).
*   **Anderson et al.:** "A Practitioner's Guide to Generalized Linear Models".

---

## Appendix

### A. Glossary
*   **Offset:** A term with coefficient fixed at 1.
*   **Deviance:** Measure of goodness-of-fit.
*   **Relativity:** $e^\beta$.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Poisson PMF** | $e^{-\lambda}\lambda^k/k!$ | Frequency |
| **Log Link** | $\ln(\mu) = X\beta$ | GLM Structure |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
