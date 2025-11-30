# Frequency GLMs (Part 1) - Theoretical Deep Dive

## Overview
This session focuses on the practical application of GLMs to **Claim Frequency**. We explore the nuances of modeling count data, specifically the choice between **Poisson** and **Negative Binomial** distributions. We also tackle the critical concept of **Exposure Offsets** and handle the common issue of **Zero-Inflation** (too many policyholders with no claims).

---

## 1. Conceptual Foundation

### 1.1 The Nature of Frequency Data

**Claim Count ($N$):** A non-negative integer ($0, 1, 2, \dots$).
*   **Rare Events:** Most policies have 0 claims. Some have 1. Very few have 2+.
*   **Exposure:** A policy active for 1 day has a much lower chance of claiming than one active for 365 days.

**Why not OLS?**
*   OLS predicts continuous values (e.g., 0.35 claims).
*   OLS assumes constant variance (Homoscedasticity). In counts, variance usually grows with the mean.
*   OLS can predict negative counts.

### 1.2 The Poisson Distribution

The starting point for all frequency modeling.
$$ P(N=k) = \frac{e^{-\lambda}\lambda^k}{k!} $$
*   **Mean:** $E[N] = \lambda$.
*   **Variance:** $Var(N) = \lambda$.
*   **Equidispersion:** Mean = Variance.

**The Offset:**
We model the *Rate* ($\mu$), but observe the *Count* ($N$).
$$ \lambda = \mu \cdot \text{Exposure} $$
$$ \ln(\lambda) = \ln(\mu) + \ln(\text{Exposure}) $$
$$ \ln(\lambda) = X\beta + \text{Offset} $$

### 1.3 Overdispersion

**The Problem:** In real insurance data, Variance > Mean.
*   **Why?** Unobserved heterogeneity. Some drivers are inherently riskier than our variables capture.
*   **Consequence:** Poisson standard errors are too small. We find "significance" where there is none.

**The Solution:**
1.  **Quasi-Poisson:** Scale the standard errors by $\sqrt{\phi}$.
2.  **Negative Binomial:** A mixture of Poisson and Gamma. It has a variance function $V(\mu) = \mu + \alpha \mu^2$.

---

## 2. Mathematical Framework

### 2.1 Negative Binomial (NB) Regression

**Derivation:**
Assume $N \sim \text{Poisson}(\lambda)$, but $\lambda$ itself is random (Gamma distributed).
The resulting marginal distribution of $N$ is Negative Binomial.

**Variance Function:**
$$ Var(N) = \mu + \alpha \mu^2 $$
*   $\alpha$: Dispersion parameter.
*   If $\alpha \to 0$, NB converges to Poisson.
*   If $\alpha > 0$, NB handles overdispersion.

### 2.2 Zero-Inflated Models (ZIP / ZINB)

**The Concept:**
Zeros come from two sources:
1.  **Structural Zeros:** Impossible to have a claim (e.g., Fraudulent policy, coverage not active).
2.  **Sampling Zeros:** Just lucky (Poisson chance).

**The Model:**
$$ P(N=0) = \pi + (1-\pi)e^{-\lambda} $$
$$ P(N=k) = (1-\pi)\frac{e^{-\lambda}\lambda^k}{k!} \quad \text{for } k > 0 $$
*   $\pi$: Probability of being in the "Zero State" (Logit model).
*   $\lambda$: Mean of the "Count State" (Poisson model).

### 2.3 Hurdle Models

Similar to ZIP, but handles zeros and non-zeros completely separately.
1.  **Bernoulli:** Did you have a claim? (Yes/No).
2.  **Truncated Poisson:** Given you had a claim, how many? ($1, 2, \dots$).

---

## 3. Theoretical Properties

### 3.1 Exposure: Weight vs. Offset

**Offset Approach (Standard):**
*   Target: Count ($N$).
*   Offset: $\ln(\text{Exposure})$.
*   Prediction: $\hat{N}$. Divide by Exposure to get Rate.

**Weight Approach:**
*   Target: Frequency ($N / \text{Exposure}$).
*   Weight: Exposure.
*   **Equivalence:** These produce identical coefficients $\beta$ in standard software (SAS/R/Python) *if* the software handles weights as "replication weights" or "inverse variance weights" correctly.
*   *Preference:* Offset is safer and more standard in Actuarial Science.

### 3.2 The Variance-Mean Relationship

*   **Poisson:** Linear ($V \propto \mu$).
*   **Negative Binomial:** Quadratic ($V \propto \mu^2$).
*   **Visual Check:** Plot $(y-\hat{y})^2$ vs. $\hat{y}$.
    *   If linear slope $\approx 1$, Poisson is okay.
    *   If quadratic slope, use NB.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Preparation for Frequency

1.  **Aggregation:** Group by Policy-Term.
2.  **Exposure Calculation:**
    *   `Exposure = (EndDate - StartDate) / 365.25`.
    *   Handle cancellations and mid-term adjustments.
3.  **Target:** `ClaimCount`.

### 4.2 Model Specification (Python Example)

Comparing Poisson, Negative Binomial, and ZIP.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.discrete.count_model import ZeroInflatedPoisson

# Simulate Overdispersed & Zero-Inflated Data
np.random.seed(42)
n = 2000
exposure = np.random.uniform(0.5, 1.0, n)
x = np.random.normal(0, 1, n)

# True Process:
# 1. Zero Inflation (20% structural zeros)
pi = 0.2
is_structural_zero = np.random.binomial(1, pi, n)

# 2. Poisson Mean (depends on x)
mu = np.exp(-1 + 0.5 * x) * exposure

# 3. Negative Binomial Heterogeneity (Gamma noise)
# Gamma mean=1, variance=0.5
gamma_noise = np.random.gamma(shape=2, scale=0.5, size=n)
lambda_i = mu * gamma_noise

# Generate Counts
counts = np.random.poisson(lambda_i)
counts[is_structural_zero == 1] = 0

df = pd.DataFrame({'Count': counts, 'x': x, 'Exposure': exposure})
df['log_exp'] = np.log(df['Exposure'])

print(f"Variance/Mean Ratio: {df['Count'].var() / df['Count'].mean():.2f}")
# Expect > 1 (Overdispersion)

# 1. Poisson GLM
poisson_model = smf.glm("Count ~ x", data=df, offset=df['log_exp'],
                        family=sm.families.Poisson()).fit()
print("\nPoisson AIC:", poisson_model.aic)

# 2. Negative Binomial GLM
# Statsmodels NB requires estimating alpha (alpha parameter).
# We use NegativeBinomial family (NB2).
nb_model = smf.glm("Count ~ x", data=df, offset=df['log_exp'],
                   family=sm.families.NegativeBinomial(alpha=0.5)).fit()
# Note: In real life, we iterate to find optimal alpha.
print("Negative Binomial AIC:", nb_model.aic)

# 3. Zero-Inflated Poisson (ZIP)
# Statsmodels ZIP is not a GLM object, has different syntax.
# exog_infl is the predictor for the zero component (logit).
zip_model = ZeroInflatedPoisson(df['Count'], df[['x']], 
                                exog_infl=np.ones((n, 1)), # Constant zero probability
                                offset=df['log_exp']).fit(disp=0)
print("ZIP AIC:", zip_model.aic)

# Interpretation:
# Compare AICs. The NB or ZIP should beat the plain Poisson.
```

### 4.3 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Coefficients:** Impact on Frequency.
2.  **Dispersion Parameter ($\alpha$):** Magnitude of overdispersion.
3.  **Zero-Inflation Probability ($\pi$):** "20% of our book is risk-free."

**Interpretation:**
*   **NB vs Poisson:** If NB fits better, it means there is unexplained variance (e.g., missing variables like "Driver Aggressiveness").
*   **ZIP:** If ZIP fits better, we might have a segment of customers who simply *never* claim (e.g., they pay small claims out of pocket to avoid rate hikes).

---

## 5. Evaluation & Validation

### 5.1 Rootogram

*   A visual tool to compare Observed vs. Expected Counts.
*   **Hanging Rootogram:** Bars hang from the theoretical curve. If they touch the x-axis, the fit is good.
*   **Check:** Does the model predict enough Zeros? Does it capture the tail (2, 3 claims)?

### 5.2 Vuong Test

*   Statistical test to compare non-nested models (e.g., ZIP vs. Standard Poisson).
*   $V > 1.96$: ZIP is significantly better.
*   $V < -1.96$: Standard Poisson is better.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Interpreting Offset Coefficient**
    *   **Issue:** "The coefficient for Log(Exposure) is 0.9."
    *   **Reality:** It *must* be 1.0. If you estimate it and it's not 1.0, your assumption of proportionality is wrong (or you have data issues). Always fix it to 1.

2.  **Trap: Zero-Inflation Overuse**
    *   **Issue:** Using ZIP just because there are many zeros.
    *   **Reality:** Poisson naturally predicts many zeros if $\lambda$ is low. Only use ZIP if there are *more* zeros than Poisson predicts.

### 6.2 Implementation Challenges

1.  **Convergence of NB:**
    *   Estimating $\alpha$ and $\beta$ simultaneously can be unstable.
    *   **Fix:** Profile Likelihood (fix $\alpha$, fit $\beta$, optimize $\alpha$, repeat).

---

## 7. Advanced Topics & Extensions

### 7.1 Time-Dependent Covariates

*   What if a driver moves Zip Code halfway through the policy?
*   **Solution:** Split the policy into two rows (Exposure 0.5 each) with different Zip Codes. GLM handles this naturally.

### 7.2 Multi-Peril Models

*   Modeling Collision, Liability, and Comprehensive separately.
*   **Correlation:** Errors might be correlated (a bad driver crashes *and* gets sued).
*   **Copulas:** Used to join the marginal frequency distributions.

---

## 8. Regulatory & Governance Considerations

### 8.1 "Optimization" of Exposure

*   Some insurers try to "optimize" the exposure definition (e.g., per mile vs. per year).
*   **Regulator:** Requires strict proof that the new exposure base is more accurate and not discriminatory.

---

## 9. Practical Example

### 9.1 Worked Example: Rate Calculation

**Model:** Negative Binomial.
*   Intercept: -1.5
*   Sports Car: +0.4
*   Offset: Log(Exposure)

**Scenario:**
*   Driver with Sports Car.
*   Policy Term: 6 months (0.5).

**Calculation:**
1.  Linear Predictor: $\eta = -1.5 + 0.4 = -1.1$.
2.  Rate (Annual): $\mu = e^{-1.1} = 0.3329$.
3.  Expected Count: $\lambda = 0.3329 \times 0.5 = 0.166$.

**Pricing:**
*   If Average Severity = \$5,000.
*   Pure Premium = $0.166 \times 5000 = \$832$.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Poisson** is the baseline.
2.  **Negative Binomial** fixes Overdispersion.
3.  **Offset** is mandatory for varying exposure.

### 10.2 When to Use This Knowledge
*   **Auto/Home Pricing:** The standard approach.
*   **Commercial Lines:** Often use NB due to high heterogeneity.

### 10.3 Critical Success Factors
1.  **Data Granularity:** Split policies by coverage and location changes.
2.  **Distribution Check:** Don't assume Poisson. Test it.
3.  **Zero Check:** Ensure you aren't under-predicting the "safe" drivers.

### 10.4 Further Reading
*   **Hilbe:** "Negative Binomial Regression".
*   **Dean et al.:** "Testing for Overdispersion in Poisson Regression".

---

## Appendix

### A. Glossary
*   **Overdispersion:** Variance > Mean.
*   **Structural Zero:** A zero count that *must* be zero (not random).
*   **Exposure:** The unit of risk (Car-Year, House-Year).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **NB Variance** | $\mu + \alpha \mu^2$ | Heterogeneity |
| **ZIP Prob(0)** | $\pi + (1-\pi)e^{-\lambda}$ | Zero Inflation |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
