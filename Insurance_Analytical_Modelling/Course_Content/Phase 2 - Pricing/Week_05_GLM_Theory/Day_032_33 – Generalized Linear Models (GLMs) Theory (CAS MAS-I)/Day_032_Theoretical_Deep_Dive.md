# Generalized Linear Models (GLMs) Theory (Part 2) - Theoretical Deep Dive

## Overview
Building on the GLM framework, this session focuses on the critical art of **Model Selection and Diagnostics**. How do we know if our model is "good"? How do we choose between two competing models? We explore statistical criteria like AIC/BIC, Deviance Analysis, and advanced residual checks (Anscombe, Quantile) to ensure our pricing models are robust, stable, and predictive.

---

## 1. Conceptual Foundation

### 1.1 The Trade-off: Bias vs. Variance

**Model Selection** is the process of finding the "Goldilocks" zone:
*   **Underfitting (High Bias):** The model is too simple. It misses real patterns (e.g., ignoring that young drivers are risky).
*   **Overfitting (High Variance):** The model is too complex. It memorizes noise (e.g., "Drivers named 'Bob' in Zip Code 90210 are safe").

**Goal:** Minimize the expected prediction error on *unseen* data.

### 1.2 Goodness of Fit vs. Parsimony

*   **Goodness of Fit:** How well does the model explain the training data? (Measured by Deviance).
*   **Parsimony:** Is the model simple enough to be stable and interpretable?
*   **The Penalty:** We penalize complex models. A variable must add *enough* signal to justify its existence.

### 1.3 Residuals in GLMs

In OLS, residuals ($y - \hat{y}$) are simple. In GLMs, they are tricky because:
1.  **Heteroscedasticity:** Variance changes with the mean. A \$100 error on a \$500 claim is huge; on a \$50,000 claim, it's noise.
2.  **Skewness:** The raw residuals are not normally distributed.

**Solution:** We transform residuals (Pearson, Anscombe, Quantile) to make them look "Normal" so we can visually inspect them.

---

## 2. Mathematical Framework

### 2.1 Information Criteria (AIC & BIC)

These metrics estimate the relative quality of statistical models.

**Akaike Information Criterion (AIC):**
$$ AIC = -2\ln(\hat{L}) + 2k $$
*   $\hat{L}$: Maximized Likelihood.
*   $k$: Number of parameters.
*   **Use:** General purpose model selection. Favors slightly more complex models than BIC.

**Bayesian Information Criterion (BIC):**
$$ BIC = -2\ln(\hat{L}) + k\ln(n) $$
*   $n$: Number of observations.
*   **Penalty:** The penalty $k\ln(n)$ is stronger than $2k$ (for $n > 8$).
*   **Use:** Favors simpler models. Better for large datasets (like insurance) to prevent overfitting.

**Rule of Thumb:**
*   $\Delta AIC < 2$: Models are indistinguishable.
*   $\Delta AIC > 10$: Strong evidence for the model with lower AIC.

### 2.2 Deviance Analysis

**Deviance ($D$):**
$$ D = 2(\mathcal{L}_{sat} - \mathcal{L}_{model}) $$

**Likelihood Ratio Test (LRT):**
Used to compare **Nested Models** (Model A is a subset of Model B).
*   $H_0$: The simpler model (A) is sufficient.
*   Statistic: $\chi^2 = D_A - D_B$.
*   Degrees of Freedom: $df = k_B - k_A$.
*   If $\chi^2 > \text{Critical Value}$, reject $H_0$ (The complex model is significantly better).

### 2.3 Types of Residuals

1.  **Raw Residual:** $r_i = y_i - \hat{\mu}_i$. (Hard to interpret).
2.  **Pearson Residual:**
    $$ r_{P,i} = \frac{y_i - \hat{\mu}_i}{\sqrt{V(\hat{\mu}_i)}} $$
    *   Standardizes by the standard deviation.
    *   Sum of squares $\approx$ Pearson Chi-Square Statistic.
3.  **Anscombe Residual:**
    *   A non-linear transformation $A(y)$ designed to make the distribution of residuals closer to Normal.
    *   Preferred for Gamma/Poisson regression plots.
4.  **Quantile Residuals (Dunn-Smyth):**
    *   The "Gold Standard" for discrete distributions (Poisson/Negative Binomial).
    *   Inverts the CDF to produce perfectly Normal residuals if the model is correct.

---

## 3. Theoretical Properties

### 3.1 Hypothesis Testing for Coefficients

**Wald Test:**
$$ z = \frac{\hat{\beta}_j}{SE(\hat{\beta}_j)} $$
*   Tests $H_0: \beta_j = 0$.
*   Approximates the LRT but is computationally cheaper (don't need to refit the model).
*   **Warning:** Can be unreliable for rare events (Hauck-Donner effect).

### 3.2 Dispersion Parameter Estimation

For Poisson, we assume $\text{Var} = \mu$ ($\phi=1$).
If $\text{Var} > \mu$, we have **Overdispersion**.
*   Estimate $\hat{\phi} = \frac{\chi^2}{n-k}$.
*   If $\hat{\phi} \gg 1$ (e.g., 1.5 or 2.0), the standard errors are understated.
*   **Fix:** Multiply SE by $\sqrt{\hat{\phi}}$ or use Negative Binomial.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Stepwise Selection

**Forward Selection:**
1.  Start with Null Model (Intercept only).
2.  Test all variables. Add the one with the lowest AIC.
3.  Repeat until AIC stops dropping.

**Backward Elimination:**
1.  Start with Full Model.
2.  Remove the variable with the highest p-value (or least impact on AIC).
3.  Repeat.

### 4.2 Diagnostic Plots

1.  **Residuals vs. Fitted:** Checks for non-linearity or heteroscedasticity.
2.  **Q-Q Plot:** Checks if residuals follow the assumed distribution (Normal for Anscombe/Quantile).
3.  **Scale-Location:** Checks homoscedasticity.
4.  **Cook's Distance:** Identifies influential points (outliers that pull the regression line).

### 4.3 Model Specification (Python Example)

Comparing Models using AIC/BIC and analyzing residuals.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
from scipy import stats

# Simulate Data: Poisson with some noise
np.random.seed(42)
n = 1000
x1 = np.random.normal(0, 1, n)
x2 = np.random.normal(0, 1, n) # Noise variable
exposure = np.random.uniform(0.5, 1.0, n)

# True Model: depends only on x1
true_mu = np.exp(-1 + 0.5 * x1) * exposure
y = np.random.poisson(true_mu)

df = pd.DataFrame({'y': y, 'x1': x1, 'x2': x2, 'exposure': exposure})
df['log_exp'] = np.log(df['exposure'])

# Model 1: Simple (Correct)
model1 = smf.glm("y ~ x1", data=df, offset=df['log_exp'],
                 family=sm.families.Poisson()).fit()

# Model 2: Complex (Overfit with x2)
model2 = smf.glm("y ~ x1 + x2", data=df, offset=df['log_exp'],
                 family=sm.families.Poisson()).fit()

# 1. Compare AIC/BIC
print(f"Model 1 AIC: {model1.aic:.2f}, BIC: {model1.bic_llf:.2f}")
print(f"Model 2 AIC: {model2.aic:.2f}, BIC: {model2.bic_llf:.2f}")

# Likelihood Ratio Test
# D_diff = 2 * (LL_complex - LL_simple)
# statsmodels stores loglike
ll_diff = 2 * (model2.llf - model1.llf)
p_value = 1 - stats.chi2.cdf(ll_diff, df=1)
print(f"LRT p-value: {p_value:.4f}")
# Expect p-value > 0.05 (Fail to reject H0: x2 is useless)

# 2. Residual Analysis (Anscombe)
resid_anscombe = model1.resid_anscombe
fitted_val = model1.predict(df, offset=df['log_exp'])

plt.figure(figsize=(12, 5))

# Residuals vs Fitted
plt.subplot(1, 2, 1)
plt.scatter(fitted_val, resid_anscombe, alpha=0.5)
plt.axhline(0, color='r', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Anscombe Residuals')
plt.title('Residuals vs Fitted')

# Q-Q Plot
plt.subplot(1, 2, 2)
stats.probplot(resid_anscombe, dist="norm", plot=plt)
plt.title('Q-Q Plot of Anscombe Residuals')

plt.show()

# Interpretation:
# If the Q-Q plot lies on the line, the Poisson assumption is reasonable.
# If Residuals vs Fitted shows a "fan" shape, we might have unmodeled heterogeneity.
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **AIC/BIC:** Lower is better.
2.  **Deviance:** Used for $\chi^2$ tests.
3.  **Residual Plots:** Visual confirmation of assumptions.

**Interpretation:**
*   **AIC vs BIC:** If AIC selects Model 2 but BIC selects Model 1, prefer Model 1 (Parsimony). Insurance datasets are large, so BIC is often safer.
*   **LRT:** "Adding `VehicleColor` reduced deviance by 50 with 10 degrees of freedom. $\chi^2_{10}$ critical value is 18.3. Significant improvement."

---

## 5. Evaluation & Validation

### 5.1 Cross-Validation

**K-Fold CV:**
1.  Split data into $K=5$ folds.
2.  Train on 4, Test on 1.
3.  Calculate Out-of-Sample Deviance or Gini.
4.  Average the results.
*   **Why?** AIC/BIC are approximations. CV measures *actual* predictive performance.

### 5.2 Stability Testing

*   **Bootstrap:** Resample the data with replacement and refit.
*   Check the distribution of $\beta$ coefficients.
*   If `Age_Young` coefficient swings from 0.2 to 0.8, the variable is unstable and should be removed or grouped.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Comparing AIC across different datasets**
    *   **Issue:** "Model A on Dataset 1 has AIC 1000. Model B on Dataset 2 has AIC 900. Model B is better."
    *   **Reality:** AIC is scale-dependent. You can only compare models fitted to the *exact same target vector*.

2.  **Trap: Ignoring Exposure in Residuals**
    *   **Issue:** Plotting residuals without adjusting for exposure.
    *   **Reality:** High exposure points will naturally have larger raw residuals. Use Pearson/Anscombe to standardize.

### 6.2 Implementation Challenges

1.  **Large Datasets:**
    *   Calculating Cook's Distance requires matrix inversion ($O(n^3)$).
    *   **Solution:** Use approximations or run diagnostics on a sample.

---

## 7. Advanced Topics & Extensions

### 7.1 Quantile Residuals

*   For discrete data (Poisson), Anscombe residuals are still not perfectly Normal.
*   **Randomized Quantile Residuals:**
    *   $u_i = F(y_i; \hat{\mu}_i)$. (CDF value).
    *   For discrete $y$, $u_i$ is a random value in $[F(y-1), F(y)]$.
    *   $r_Q = \Phi^{-1}(u_i)$.
    *   These are *exactly* Normal if the model is correct.

### 7.2 Variable Selection with Lasso (GLMNet)

*   Instead of Stepwise (Greedy), use Regularization ($L_1$ penalty).
*   Shrinks coefficients to exactly zero.
*   **Benefit:** Handles high-dimensional data (e.g., hundreds of vehicle symbols) better than Stepwise.

---

## 8. Regulatory & Governance Considerations

### 8.1 "Black Box" vs. "Glass Box"

*   Stepwise selection is a "Glass Box" (we see the steps).
*   Lasso is slightly more opaque but still interpretable.
*   **Regulator View:** They prefer models where every variable has a clear, intuitive reason for being there. "The algorithm picked it" is a weak defense.

---

## 9. Practical Example

### 9.1 Worked Example: Deviance Table

**Scenario:**
*   **Null Model:** Deviance = 10,000, df = 999.
*   **Model A (+Age):** Deviance = 9,500, df = 995 (4 parameters).
*   **Model B (+Age + Gender):** Deviance = 9,490, df = 994 (1 parameter).

**Analysis:**
1.  **Null vs A:** $\Delta D = 500$, $\Delta df = 4$. Highly significant. Age is good.
2.  **A vs B:** $\Delta D = 10$, $\Delta df = 1$. $\chi^2_{1, 0.95} = 3.84$.
    *   $10 > 3.84$. Gender is significant.
    *   **AIC Check:** $AIC = D + 2k$.
        *   $\Delta AIC = \Delta D - 2(\Delta k) = 10 - 2(1) = 8$.
        *   AIC drops by 8. Model B is better.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **AIC/BIC** balance fit and complexity.
2.  **Deviance** is the yardstick for GLM fit.
3.  **Anscombe/Quantile Residuals** are needed to diagnose non-normal errors.

### 10.2 When to Use This Knowledge
*   **Model Iteration:** Deciding whether to add that new telematics variable.
*   **Peer Review:** Challenging a colleague's complex model.

### 10.3 Critical Success Factors
1.  **Visual Inspection:** Don't rely solely on AIC numbers. Look at the plots.
2.  **Parsimony:** When in doubt, choose the simpler model.
3.  **Business Logic:** Statistical significance $\neq$ Business significance.

### 10.4 Further Reading
*   **Dunn & Smyth:** "Randomized Quantile Residuals".
*   **Burnham & Anderson:** "Model Selection and Multimodel Inference".

---

## Appendix

### A. Glossary
*   **Parsimony:** The principle that the simplest explanation is usually the best.
*   **Saturated Model:** A model that fits the data perfectly (1 param per data point).
*   **Nested Model:** A model that can be obtained by constraining parameters of a larger model to zero.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **AIC** | $-2\ln(L) + 2k$ | Selection |
| **BIC** | $-2\ln(L) + k\ln(n)$ | Selection (Large N) |
| **Pearson Resid** | $(y-\mu)/\sqrt{V(\mu)}$ | Diagnostics |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
