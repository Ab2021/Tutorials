# Severity GLMs (Part 1) - Theoretical Deep Dive

## Overview
We now shift our focus from "How many claims?" (Frequency) to "How much does each claim cost?" (Severity). This session introduces the **Gamma GLM**, the industry standard for modeling average claim size. We explore why the Gamma distribution is uniquely suited for insurance losses due to its **Constant Coefficient of Variation (CV)** property, and why the **Log Link** is almost always preferred over the canonical Inverse Link.

---

## 1. Conceptual Foundation

### 1.1 The Nature of Severity Data

**Claim Severity ($Y$):** The cost of a single claim (or average cost per claim).
*   **Continuous & Positive:** Claims are always $>0$ (usually).
*   **Right-Skewed:** Most claims are small (fender benders), but a few are massive (total losses).
*   **Heteroscedastic:** The variability of losses increases with the size of the loss. A \$500 claim might vary by \$50. A \$50,000 claim might vary by \$10,000.

**Why not OLS?**
*   OLS assumes constant variance ($\sigma^2$).
*   If we use OLS on raw dollars, the massive variance of high-value claims will dominate the least-squares minimization, causing the model to fit the tail and ignore the body.

### 1.2 The Gamma Distribution

The Gamma distribution is defined by Shape ($\alpha$) and Scale ($\theta$).
$$ f(y) = \frac{1}{\Gamma(\alpha)\theta^\alpha} y^{\alpha-1} e^{-y/\theta} $$
*   **Mean:** $\mu = \alpha \theta$.
*   **Variance:** $V = \alpha \theta^2 = \frac{\mu^2}{\alpha}$.
*   **Variance Function:** $V(\mu) \propto \mu^2$.

**Constant Coefficient of Variation (CV):**
$$ CV = \frac{\sqrt{Variance}}{Mean} = \frac{\sqrt{\mu^2/\alpha}}{\mu} = \frac{1}{\sqrt{\alpha}} $$
*   Since $\alpha$ is constant in a standard GLM, the CV is constant.
*   **Interpretation:** The "relative error" is constant. We predict small claims with the same *percentage* accuracy as large claims. This aligns perfectly with insurance reality.

### 1.3 Link Functions: Log vs. Inverse

**Canonical Link (Inverse):** $g(\mu) = 1/\mu$.
*   **Pros:** Mathematical simplicity (Newton-Raphson converges fast).
*   **Cons:** Can predict negative values if $X\beta$ crosses zero. Hard to interpret (reciprocal of cost?).

**Log Link:** $g(\mu) = \ln(\mu)$.
*   **Pros:** Guarantees $\mu > 0$. Coefficients are multiplicative factors (easy for rating).
*   **Cons:** Not canonical (slightly slower convergence, though irrelevant with modern computers).
*   **Verdict:** **Always use Log Link** for pricing.

---

## 2. Mathematical Framework

### 2.1 The Gamma GLM Structure

$$ Y_i \sim \text{Gamma}(\mu_i, \phi) $$
$$ \ln(\mu_i) = \beta_0 + \beta_1 x_{1i} + \dots + \beta_p x_{pi} $$
$$ \text{Var}(Y_i) = \phi \mu_i^2 / w_i $$

*   $\phi$: Dispersion parameter ($1/\alpha$).
*   $w_i$: Weight (usually Claim Count).

### 2.2 Weights in Severity Models

**Frequency Weights:** Exposure (Car-Years).
**Severity Weights:** Claim Count.
*   *Why?* An observation representing the average of 10 claims is more reliable (lower variance) than an observation of 1 claim.
*   **Variance of Average:** $\text{Var}(\bar{Y}) = \frac{\text{Var}(Y)}{n}$.
*   Therefore, weight $w_i = n_i$ (Number of Claims).

### 2.3 Large Loss Truncation

Gamma models struggle with extreme tails (e.g., a \$1M liability claim).
*   **Capping:** We usually truncate losses at a threshold (e.g., \$100k or \$250k) for the "Attritional" model.
*   **Excess:** Losses above the cap are modeled separately (Excess Severity) or loaded as a flat factor.

---

## 3. Theoretical Properties

### 3.1 Relationship to Lognormal

*   **Lognormal Regression:** $\ln(Y) \sim N(X\beta, \sigma^2)$. (OLS on logged target).
*   **Gamma GLM:** $Y \sim \text{Gamma}$.
*   **Difference:**
    *   Lognormal minimizes squared error of *logs*. It is biased when transforming back to dollars ($E[Y] = e^{\mu + \sigma^2/2}$).
    *   Gamma minimizes squared percentage error of *dollars*. It is unbiased for the mean.
    *   **Actuarial Preference:** Gamma is generally preferred because it preserves the total loss ($ \sum \hat{y} = \sum y $).

### 3.2 The Variance Assumption Check

*   **Plot:** Squared Residuals vs. Fitted Values (Log-Log scale).
*   **Slope:**
    *   Slope $\approx 0 \implies$ Constant Variance (Normal/OLS).
    *   Slope $\approx 1 \implies$ Variance $\propto$ Mean (Poisson).
    *   Slope $\approx 2 \implies$ Variance $\propto$ Mean$^2$ (Gamma).
    *   Slope $\approx 3 \implies$ Variance $\propto$ Mean$^3$ (Inverse Gaussian).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Preparation for Severity

1.  **Filter:** Keep only records with `ClaimCount > 0`.
2.  **Target:** `AverageSeverity = IncurredLoss / ClaimCount`.
3.  **Weight:** `ClaimCount`.
4.  **Capping:** `Loss = min(Loss, 250000)`.

### 4.2 Model Specification (Python Example)

Comparing Gamma GLM vs. OLS on Log(Loss).

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Simulate Severity Data (Gamma)
np.random.seed(42)
n = 1000
age = np.random.uniform(18, 60, n)
is_luxury = np.random.binomial(1, 0.2, n)

# True Mean: Luxury cars cost 2x to repair
mu = np.exp(7 + 0.01 * age + 0.7 * is_luxury) 
# Base ~1100, +1% per year, +70% for Luxury

# Generate Gamma distributed losses
# Shape=2 (CV ~ 0.7)
shape = 2
scale = mu / shape
losses = np.random.gamma(shape, scale)

df = pd.DataFrame({'Loss': losses, 'Age': age, 'Luxury': is_luxury})

# 1. OLS on Log(Loss) - "Lognormal Regression"
df['log_loss'] = np.log(df['Loss'])
model_ols = smf.ols("log_loss ~ Age + Luxury", data=df).fit()

# 2. Gamma GLM (Log Link)
model_gamma = smf.glm("Loss ~ Age + Luxury", data=df,
                      family=sm.families.Gamma(link=sm.families.links.log())).fit()

print("OLS Coefs (Log Scale):")
print(model_ols.params)
print("\nGamma Coefs (Log Scale):")
print(model_gamma.params)

# Predictions
pred_ols_log = model_ols.predict(df)
pred_ols_raw = np.exp(pred_ols_log) # Biased! Needs smearing factor.
pred_gamma = model_gamma.predict(df)

print(f"\nTotal Actual Loss: {df['Loss'].sum():,.0f}")
print(f"Total Gamma Pred:  {pred_gamma.sum():,.0f} (Unbiased)")
print(f"Total OLS Pred:    {pred_ols_raw.sum():,.0f} (Biased Low)")

# Residual Plot (Variance Check)
resid = model_gamma.resid_pearson
plt.scatter(np.log(pred_gamma), resid, alpha=0.5)
plt.title("Gamma Pearson Residuals vs Log(Fitted)")
plt.xlabel("Log(Fitted Severity)")
plt.ylabel("Pearson Residual")
plt.axhline(0, color='r')
plt.show()

# Interpretation:
# The Gamma model sums to the total actual loss.
# The OLS model (unadjusted) underpredicts because E[exp(X)] != exp(E[X]).
```

### 4.3 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Coefficients:** "Luxury cars are 1.7x more expensive to fix."
2.  **Dispersion ($\phi$):** Indicates the volatility of claims.

**Interpretation:**
*   **Age Effect:** Often flatter in Severity than Frequency. Old people crash less (Freq), but when they do, the cost is similar (Sev).
*   **Vehicle Type:** The dominant driver of Severity.

---

## 5. Evaluation & Validation

### 5.1 Bias Tests

*   **Decile Plot:** Sort data by predicted severity. Check if Actual $\approx$ Predicted in every decile.
*   **Variable Bias:** Check Actual vs. Predicted by `VehicleMake`. If "BMW" is consistently underpredicted, you need a better vehicle grouping.

### 5.2 The "Smearing" Factor

If you *must* use Lognormal Regression (OLS on logs):
*   Calculate residuals $e_i$.
*   Smearing Factor $S = \text{mean}(e^{e_i})$.
*   Prediction $\hat{y} = \hat{y}_{raw} \times S$.
*   *Gamma GLM avoids this mess entirely.*

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Including Zero Claims**
    *   **Issue:** Running Severity on the whole dataset (including non-claims).
    *   **Reality:** Severity is conditional on a claim occurring ($Y | N > 0$). Zeros belong in the Frequency model.

2.  **Trap: Weighting by Exposure**
    *   **Issue:** Using `Exposure` as weight in Severity.
    *   **Reality:** A claim is a claim. It doesn't matter if it came from a 6-month or 12-month policy. Weight by `ClaimCount`.

### 6.2 Implementation Challenges

1.  **Convergence:**
    *   Gamma with Log Link is not canonical. It can diverge if initial values are bad.
    *   **Fix:** Initialize with the mean of $y$.

---

## 7. Advanced Topics & Extensions

### 7.1 Inverse Gaussian Distribution

*   If the tail is *heavier* than Gamma (Variance $\propto \mu^3$).
*   Used for Workers Comp or Liability where tails are fat.

### 7.2 Tweedie Distribution

*   Models Pure Premium directly (Frequency $\times$ Severity).
*   Avoids the two-model split.
*   *More on this in Day 39.*

---

## 8. Regulatory & Governance Considerations

### 8.1 Inflation Adjustment

*   Severity trends upward with inflation (CPI, Medical CPI).
*   **Requirement:** You must trend historical losses to the future "Cost Level" before modeling.
*   **Trend Factor:** $1.05^3$ for 3 years of 5% inflation.

---

## 9. Practical Example

### 9.1 Worked Example: Severity Prediction

**Model:**
*   Intercept: 7.0 (Base $\approx$ \$1,096)
*   New Car: +0.20
*   Urban: +0.10

**Scenario:**
*   New Car in Urban area.

**Calculation:**
1.  Linear Predictor: $7.0 + 0.2 + 0.1 = 7.3$.
2.  Prediction: $e^{7.3} = \$1,480$.

**Contrast with Frequency:**
*   Frequency might be $0.10$.
*   Pure Premium = $0.10 \times 1480 = \$148$.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Gamma** handles the spread of dollar amounts.
2.  **Log Link** keeps predictions positive and multiplicative.
3.  **Constant CV** is the magic property that fits insurance data.

### 10.2 When to Use This Knowledge
*   **Pricing:** The standard Severity model.
*   **Claims Analytics:** Predicting the final cost of an open claim.

### 10.3 Critical Success Factors
1.  **Cap Large Losses:** Don't let one \$1M claim distort the model.
2.  **Weight Correctly:** Use Claim Count, not Exposure.
3.  **Trend Data:** Adjust for inflation first.

### 10.4 Further Reading
*   **McCullagh & Nelder:** Chapter on Gamma Models.
*   **CAS Study Note:** "Basic Ratemaking - Severity".

---

## Appendix

### A. Glossary
*   **Heteroscedasticity:** Variance changes with the mean.
*   **Coefficient of Variation (CV):** Standard Deviation / Mean.
*   **Canonical Link:** The link that simplifies the math (Inverse for Gamma).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Gamma Variance** | $\mu^2 / \alpha$ | Assumption |
| **Log Link** | $\ln(\mu) = X\beta$ | Structure |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
