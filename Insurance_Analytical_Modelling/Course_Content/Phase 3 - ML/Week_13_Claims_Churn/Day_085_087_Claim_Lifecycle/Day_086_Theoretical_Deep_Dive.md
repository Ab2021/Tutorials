# Claim Lifecycle & Severity Development (Part 2) - Severity Modeling & Distributions - Theoretical Deep Dive

## Overview
"Frequency is how often it happens. Severity is how much it hurts."
While frequency is often Poisson, severity is the wild west of statistics.
It is skewed, heavy-tailed, and driven by inflation.
This day focuses on **Severity Distributions** (Gamma, Lognormal, Pareto) and **Stochastic Modeling** of claim costs.

---

## 1. Conceptual Foundation

### 1.1 The Nature of Severity

*   **Positivity:** Claims are always $> 0$. (Normal distribution is bad because it allows negatives).
*   **Skewness:** Most claims are small (fender benders). A few are huge (total loss).
*   **Heavy Tails:** The probability of a \$1M claim is higher than a Gaussian model predicts.

### 1.2 Key Distributions

1.  **Gamma:** Good for "Attritional" claims (e.g., Physical Damage). Light tail.
2.  **Lognormal:** The workhorse. Good for "Medium" tails (e.g., Bodily Injury).
3.  **Pareto:** The "Disaster" distribution. Good for "Large" losses (e.g., Liability, Catastrophe).

---

## 2. Mathematical Framework

### 2.1 The Gamma Distribution

$$ f(x) = \frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x} $$
*   **Shape ($\alpha$):** Controls skewness.
*   **Rate ($\beta$):** Controls scale.
*   **GLM Link:** Log Link. $E[Y] = \exp(X\beta)$.
*   **Variance:** $Var(Y) = \phi E[Y]^2$ (Constant Coefficient of Variation).

### 2.2 The Lognormal Distribution

If $Y \sim Lognormal(\mu, \sigma)$, then $\ln(Y) \sim Normal(\mu, \sigma)$.
*   **Fitting:** Simply take the log of claim amounts and fit a Normal distribution (OLS).
*   **Property:** Multiplicative effects (Inflation adds to $\mu$).

### 2.3 The Pareto Distribution (Type II / Lomax)

$$ f(x) = \frac{\alpha \lambda^\alpha}{(x + \lambda)^{\alpha+1}} $$
*   **Alpha ($\alpha$):** Tail index. Lower $\alpha$ = Heavier tail.
*   **Use:** Excess of Loss pricing. "What is the probability a claim exceeds \$1M?"

---

## 3. Theoretical Properties

### 3.1 Coefficient of Variation (CV)

*   **Formula:** $CV = \frac{\sigma}{\mu}$.
*   **Interpretation:** Volatility per unit of mean.
*   **Gamma:** CV is constant ($1/\sqrt{\alpha}$).
*   **Lognormal:** CV depends on $\sigma$.

### 3.2 Limited Expected Value (LEV)

*   **Definition:** $E[min(X, L)]$. The expected cost to the insurer if there is a policy limit $L$.
*   **Use:** Pricing deductibles and limits.
    *   $Premium = Frequency \times (LEV(Limit) - LEV(Deductible))$.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Fitting Distributions (Scipy)

```python
import scipy.stats as stats
import matplotlib.pyplot as plt
import numpy as np

# Data: Claim Severities
claims = df['incurred_amount'].values

# 1. Fit Gamma
alpha_g, loc_g, beta_g = stats.gamma.fit(claims, floc=0)

# 2. Fit Lognormal
shape_l, loc_l, scale_l = stats.lognorm.fit(claims, floc=0)

# 3. Compare AIC (Akaike Information Criterion)
# Lower is better
def calculate_aic(dist, params, data):
    ll = dist.logpdf(data, *params).sum()
    k = len(params)
    return 2*k - 2*ll

aic_gamma = calculate_aic(stats.gamma, (alpha_g, loc_g, beta_g), claims)
aic_lognorm = calculate_aic(stats.lognorm, (shape_l, loc_l, scale_l), claims)

print(f"Gamma AIC: {aic_gamma:.0f}")
print(f"Lognormal AIC: {aic_lognorm:.0f}")
```

### 4.2 GLM for Severity (Tweedie)

*   **Tweedie:** A compound Poisson-Gamma distribution.
*   **Power Parameter ($p$):**
    *   $p=1$: Poisson (Frequency).
    *   $p=2$: Gamma (Severity).
    *   $1 < p < 2$: Compound Poisson (Pure Premium).

```python
import statsmodels.api as sm

# Gamma GLM for Average Severity
# Weights = Claim Count (to handle averaging)
glm_gamma = sm.GLM(
    df['avg_severity'],
    df[['age', 'vehicle_value', 'is_urban']],
    family=sm.families.Gamma(link=sm.families.links.log()),
    var_weights=df['claim_count']
).fit()

print(glm_gamma.summary())
```

---

## 5. Evaluation & Validation

### 5.1 QQ Plots

*   **Visual Check:** Plot Empirical Quantiles vs. Theoretical Quantiles.
*   **Tail Check:** Does the model capture the top 1% of claims? (Lognormal often fails here, needing a Pareto splice).

### 5.2 Gini Coefficient for Severity

*   **Concept:** Does the model differentiate between "Expensive" and "Cheap" claims?
*   **Note:** Severity is harder to predict than frequency. Gini scores are usually lower (e.g., 0.15 vs 0.30 for frequency).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 The "Zero" Problem

*   **Issue:** Severity models (Gamma) cannot handle $Y=0$.
*   **Context:** Closed Without Payment (CWP) claims have 0 severity.
*   **Fix:** Filter out CWPs before modeling severity. Model CWP probability separately (Logistic Regression).

### 6.2 Inflation Bias

*   **Issue:** Fitting a model to 10 years of data without adjusting for inflation.
*   **Result:** The model underestimates today's cost.
*   **Fix:** "On-Leveling". Adjust past claims to current dollar values using CPI or Medical Inflation indices *before* fitting.

---

## 7. Advanced Topics & Extensions

### 7.1 Spliced Distributions

*   **Concept:** Use Gamma for the body (<\$100k) and Pareto for the tail (>\$100k).
*   **Method:** "Composite Models".
*   **Benefit:** Best of both worlds. Stability for small claims, safety for large claims.

### 7.2 Mixture Models

*   **Concept:** The data comes from two subpopulations.
    *   Type A: Fender Benders (Mean \$2k).
    *   Type B: Total Losses (Mean \$30k).
*   **Model:** Gaussian Mixture Model (GMM) on log-claims.
    *   $f(x) = w_1 f_1(x) + w_2 f_2(x)$.

---

## 8. Regulatory & Governance Considerations

### 8.1 Trend Selection

*   **Regulation:** Regulators scrutinize the "Trend Factor" (Inflation assumption).
*   **Risk:** If you assume 2% inflation but medical costs rise 8%, you are insolvent.

---

## 9. Practical Example

### 9.1 The "Total Loss" Predictor

**Scenario:** Auto Physical Damage.
**Data:** Vehicle Age, Make, Model, Point of Impact.
**Model:**
1.  **Classifier:** Is it a Total Loss? (Yes/No).
2.  **Regressor:** If No, what is the Repair Cost?
3.  **Logic:** If Yes, Cost = Vehicle Value (ACV).
**Result:** Much more accurate than a single Gamma GLM, because Total Losses follow a different physics (Capped at ACV) than Repairs (Uncapped up to ACV).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Gamma** for body, **Pareto** for tail.
2.  **Lognormal** is the standard starting point.
3.  **Inflation** must be removed (on-leveling) before modeling.

### 10.2 When to Use This Knowledge
*   **Pricing:** Setting Pure Premiums.
*   **Reserving:** Stochastic Reserving (Bootstrapping).

### 10.3 Critical Success Factors
1.  **Tail Fit:** Don't ignore the top 1%. That's where the risk is.
2.  **Segmentation:** Modeling "Fire" and "Theft" together is bad. They have different severity distributions.

### 10.4 Further Reading
*   **Klugman, Panjer, Willmot:** "Loss Models: From Data to Decisions" (The Bible of Severity).
*   **Parodi:** "Pricing in General Insurance".

---

## Appendix

### A. Glossary
*   **ACV:** Actual Cash Value (Replacement Cost - Depreciation).
*   **MFL:** Maximum Foreseeable Loss.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Gamma PDF** | $\frac{\beta^\alpha}{\Gamma(\alpha)} x^{\alpha-1} e^{-\beta x}$ | Attritional Severity |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
