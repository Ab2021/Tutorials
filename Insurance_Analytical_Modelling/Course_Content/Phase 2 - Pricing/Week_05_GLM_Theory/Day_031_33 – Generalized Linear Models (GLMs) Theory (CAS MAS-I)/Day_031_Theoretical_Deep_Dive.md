# Generalized Linear Models (GLMs) Theory (Part 1) - Theoretical Deep Dive

## Overview
This session introduces the Generalized Linear Model (GLM), the workhorse of modern actuarial pricing. We move beyond Ordinary Least Squares (OLS) to a framework that can handle the skewed, non-negative, and heteroscedastic nature of insurance data. We explore the three components of a GLM: the Random Component (Exponential Family), the Systematic Component (Linear Predictor), and the Link Function.

---

## 1. Conceptual Foundation

### 1.1 The Limitations of OLS for Insurance

**Ordinary Least Squares (OLS)** assumes:
1.  **Normality:** Errors are normally distributed ($Y \sim N(\mu, \sigma^2)$).
2.  **Homoscedasticity:** Variance is constant ($\sigma^2$) regardless of the mean.
3.  **Identity Link:** The relationship is linear ($Y = X\beta + \epsilon$).

**Why OLS Fails for Insurance:**
*   **Skewness:** Claims are highly right-skewed. A Normal distribution allows for negative claims (impossible) and underestimates the tail.
*   **Heteroscedasticity:** Risky drivers (high mean frequency) have *more variable* outcomes than safe drivers. Variance is not constant; it grows with the mean.
*   **Multiplicative Nature:** Risk factors usually multiply. A bad driver in a bad car is *much* worse than the sum of the parts. OLS is additive.

### 1.2 The GLM Framework

A GLM generalizes OLS by allowing:
1.  **Random Component:** The response $Y$ can follow any distribution in the **Exponential Family** (Normal, Poisson, Gamma, Binomial, Inverse Gaussian).
2.  **Systematic Component:** A linear predictor $\eta = X\beta$.
3.  **Link Function:** A monotonic function $g(\cdot)$ connecting the mean $\mu = E[Y]$ to the linear predictor: $\eta = g(\mu)$.

**Key Insight:** We do not transform the *data* (like taking $\ln(Y)$ in OLS). We transform the *expectation* ($\ln(E[Y])$). This preserves the mean structure.

### 1.3 Why This Matters

**Business Impact:**
*   **Accuracy:** GLMs capture the true shape of risk.
*   **Fairness:** By modeling the mean directly, GLMs produce unbiased estimates (Total Predicted Loss = Total Actual Loss).
*   **Interpretability:** The multiplicative structure (Log Link) aligns with standard rating plans (Base Rate $\times$ Factor A $\times$ Factor B).

---

## 2. Mathematical Framework

### 2.1 The Exponential Family

A distribution is in the Exponential Family if its PDF can be written as:
$$ f(y; \theta, \phi) = \exp \left( \frac{y\theta - b(\theta)}{a(\phi)} + c(y, \phi) \right) $$

*   $\theta$: Canonical Parameter (related to the mean).
*   $\phi$: Dispersion Parameter (related to the variance).
*   $b(\theta)$: Cumulant function.

**Key Properties:**
1.  **Mean:** $E[Y] = \mu = b'(\theta)$.
2.  **Variance:** $Var(Y) = b''(\theta) \cdot a(\phi)$.
3.  **Variance Function:** $V(\mu) = b''(\theta)$. This defines how variance relates to the mean.

### 2.2 Common Distributions in Insurance

| Distribution | Domain | Variance Function $V(\mu)$ | Typical Use |
| :--- | :--- | :--- | :--- |
| **Normal** | $(-\infty, \infty)$ | $1$ (Constant) | General stats (rare in pricing) |
| **Poisson** | $0, 1, 2, \dots$ | $\mu$ | Claim Frequency |
| **Gamma** | $(0, \infty)$ | $\mu^2$ | Claim Severity |
| **Inverse Gaussian** | $(0, \infty)$ | $\mu^3$ | Severity (Heavy Tail) |
| **Binomial** | $0, 1, \dots, n$ | $\mu(1-\mu)$ | Retention / Renewal |
| **Tweedie** | $[0, \infty)$ | $\mu^p$ ($1 < p < 2$) | Pure Premium |

### 2.3 Link Functions

The link function $g(\mu) = \eta$ maps the mean (domain of $Y$) to the linear predictor $(-\infty, \infty)$.

**Common Links:**
1.  **Identity:** $g(\mu) = \mu$. (OLS).
2.  **Log:** $g(\mu) = \ln(\mu) \implies \mu = e^{X\beta}$. (Multiplicative).
    *   Ensures $\mu > 0$.
    *   Standard for Poisson/Gamma.
3.  **Logit:** $g(\mu) = \ln(\frac{\mu}{1-\mu})$. (Logistic).
    *   Maps $(0, 1)$ to $(-\infty, \infty)$.
    *   Standard for Binomial.
4.  **Canonical Link:** The link that makes $\theta = \eta$.
    *   Poisson Canonical: Log.
    *   Gamma Canonical: Reciprocal ($1/\mu$). *Note: Actuaries rarely use the reciprocal link for Gamma because it doesn't guarantee positivity. We prefer Log.*

---

## 3. Theoretical Properties

### 3.1 Maximum Likelihood Estimation (MLE)

GLMs are fitted using MLE. We maximize the Log-Likelihood:
$$ \mathcal{L}(\beta) = \sum_{i=1}^n \ln f(y_i; \theta_i, \phi) $$

**Iteratively Reweighted Least Squares (IRLS):**
*   There is no closed-form solution (unlike OLS $(X^TX)^{-1}X^Ty$).
*   We use the Newton-Raphson method, which is equivalent to running a weighted regression iteratively until convergence.

### 3.2 Deviance

Deviance is the GLM equivalent of "Sum of Squared Errors."
$$ D = 2 \left( \mathcal{L}(\text{Saturated}) - \mathcal{L}(\text{Model}) \right) $$
*   **Saturated Model:** A model with one parameter for every observation (perfect fit).
*   **Scaled Deviance:** $D^* = D / \phi$.
*   **Goal:** Minimize Deviance.

### 3.3 Offset

An **Offset** is a term in the linear predictor with a fixed coefficient of 1.
$$ \ln(\mu) = X\beta + \ln(\text{Exposure}) $$
$$ \mu = \text{Exposure} \cdot e^{X\beta} $$
*   Crucial for modeling rates (Frequency per year). If a policy is only active for 6 months, expected counts should be half.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

*   **Target:** Response variable (Count, Cost).
*   **Features:** Rating variables (Age, Zip, etc.).
*   **Weight/Exposure:** Duration of the policy.

### 4.2 Preprocessing Steps

**Step 1: One-Hot Encoding**
*   GLMs require numeric input. Categorical variables must be dummy-encoded.
*   **Base Level:** One level is dropped to avoid multicollinearity (The "Reference" level).

**Step 2: Log-Transformation of Exposure**
*   Create `log_exposure` column for the offset.

### 4.3 Model Specification (Python Example)

Comparing OLS and Poisson GLM on Count Data.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Simulate Count Data (Poisson)
np.random.seed(42)
n = 1000
age = np.random.uniform(18, 60, n)
is_male = np.random.binomial(1, 0.5, n)
exposure = np.random.uniform(0.5, 1.0, n)

# True Lambda: exp(-2 + 0.02*Age + 0.3*Male) * Exposure
true_mu = np.exp(-2 + 0.02 * age + 0.3 * is_male) * exposure
counts = np.random.poisson(true_mu)

df = pd.DataFrame({'Count': counts, 'Age': age, 'Male': is_male, 'Exposure': exposure})

# 1. OLS (Linear Regression)
# Count ~ Age + Male
ols_model = smf.ols("Count ~ Age + Male", data=df).fit()
print("OLS Summary:")
print(ols_model.summary())

# Check Predictions
pred_ols = ols_model.predict(df)
print(f"\nMin OLS Prediction: {pred_ols.min():.4f}")
# OLS might predict negative counts for young people with low exposure!

# 2. Poisson GLM
# Count ~ Age + Male + offset(log(Exposure))
df['log_exp'] = np.log(df['Exposure'])
glm_model = smf.glm("Count ~ Age + Male", data=df, 
                    offset=df['log_exp'],
                    family=sm.families.Poisson(link=sm.families.links.log())).fit()

print("\nGLM Summary:")
print(glm_model.summary())

# Check Predictions
pred_glm = glm_model.predict(df, offset=df['log_exp'])
print(f"Min GLM Prediction: {pred_glm.min():.4f}")

# Visualization: Residuals
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(pred_ols, df['Count'] - pred_ols, alpha=0.5)
plt.title('OLS Residuals (Heteroscedastic)')
plt.xlabel('Predicted')
plt.ylabel('Residual')
plt.axhline(0, color='r')

plt.subplot(1, 2, 2)
# Pearson Residuals for GLM
resid_pearson = glm_model.resid_pearson
plt.scatter(pred_glm, resid_pearson, alpha=0.5)
plt.title('GLM Pearson Residuals (More Constant)')
plt.xlabel('Predicted')
plt.ylabel('Pearson Residual')
plt.axhline(0, color='r')

plt.show()

# Interpretation:
# OLS residuals fan out (variance increases with mean).
# GLM Pearson residuals are standardized by the variance function V(mu)=mu, so they look more constant.
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Coefficients:** $\beta$.
2.  **Standard Errors:** Uncertainty in $\beta$.
3.  **z-score / p-value:** Significance.

**Interpretation:**
*   **OLS:** "Being Male adds 0.15 claims." (Additive).
*   **GLM (Log Link):** "Being Male multiplies frequency by $e^{0.3} = 1.35$." (Multiplicative).
*   **Actuarial Preference:** Multiplicative is better because a 35% surcharge applies correctly whether the base rate is high (Urban) or low (Rural).

---

## 5. Evaluation & Validation

### 5.1 Residual Analysis

**Pearson Residuals:**
$$ r_p = \frac{y - \hat{\mu}}{\sqrt{V(\hat{\mu})}} $$
*   Should have mean 0 and variance 1 (roughly).
*   Used to check the variance assumption.

**Anscombe Residuals:**
*   Transformed residuals that are closer to Normal. Better for visual inspection.

### 5.2 Goodness of Fit

**AIC (Akaike Information Criterion):**
$$ AIC = -2\ln(\mathcal{L}) + 2k $$
*   Penalizes complexity ($k$ parameters).
*   Used to compare non-nested models.

**Likelihood Ratio Test:**
*   Used to compare nested models (e.g., Model A vs. Model A + Age).
*   Statistic $\sim \chi^2$.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Using R-squared**
    *   **Issue:** $R^2$ is an OLS concept (Variance explained). It is meaningless in Poisson/Gamma GLMs.
    *   **Fix:** Use Deviance Explained or Gini Index.

2.  **Trap: Canonical Link Blindness**
    *   **Issue:** Using the Inverse Link for Gamma just because it's canonical.
    *   **Reality:** It can predict negative values if the linear predictor goes negative.
    *   **Fix:** Always use Log Link for Gamma in pricing.

### 6.2 Implementation Challenges

1.  **Convergence Failures:**
    *   If predictors are perfectly correlated (Rank Deficient), IRLS may fail.
    *   **Solution:** Drop redundant variables or use regularization.

---

## 7. Advanced Topics & Extensions

### 7.1 Overdispersion

*   **Poisson Assumption:** Mean = Variance ($\mu = \sigma^2$).
*   **Reality:** Variance > Mean (Overdispersion).
*   **Fix:** Use **Quasi-Poisson** (estimates $\phi > 1$) or **Negative Binomial** (adds a parameter).

### 7.2 Zero-Inflated Models

*   If there are *too many* zeros (more than Poisson predicts).
*   **ZIP (Zero-Inflated Poisson):** A mixture of a point mass at zero (Perfect Drivers) and a Poisson (Normal Drivers).

---

## 8. Regulatory & Governance Considerations

### 8.1 Statistical Significance vs. Business Logic

*   A variable might be statistically significant (p < 0.05) but counter-intuitive (e.g., "Newer cars have higher frequency").
*   **Regulator:** Will ask "Why?"
*   **Answer:** Maybe newer cars are driven more miles. If you don't have mileage in the model, Car Age proxies for it.
*   **Action:** You might remove the variable even if significant, to avoid "Counter-intuitive Rating."

---

## 9. Practical Example

### 9.1 Worked Example: Manual Calculation

**Scenario:**
*   Intercept $\beta_0 = -1.0$.
*   Age < 25 $\beta_1 = 0.5$.
*   Exposure = 0.5 years.

**Prediction:**
1.  **Linear Predictor:** $\eta = -1.0 + 0.5 + \ln(0.5) = -0.5 - 0.693 = -1.193$.
2.  **Mean:** $\mu = e^{-1.193} = 0.303$.
3.  **Rate:** $0.303 / 0.5 = 0.606$ claims per year.

**Check:**
*   Base Rate = $e^{-1.0} = 0.368$.
*   Relativity = $e^{0.5} = 1.648$.
*   Annual Rate = $0.368 \times 1.648 = 0.606$. Matches.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **GLM = Random + Systematic + Link.**
2.  **Exponential Family** defines the variance structure.
3.  **Log Link** provides the multiplicative factors needed for insurance rating.

### 10.2 When to Use This Knowledge
*   **Pricing:** The standard for Rate Indication.
*   **Reserving:** Stochastic Reserving (Mack) is a form of GLM.

### 10.3 Critical Success Factors
1.  **Choose the Right Distribution:** Poisson for counts, Gamma for severity.
2.  **Handle Exposure:** Always use the offset.
3.  **Check Residuals:** Ensure the variance assumption holds.

### 10.4 Further Reading
*   **McCullagh & Nelder:** "Generalized Linear Models" (The Bible of GLMs).
*   **CAS Exam MAS-I Syllabus.**

---

## Appendix

### A. Glossary
*   **Canonical Link:** The natural link function for a distribution.
*   **Deviance:** Goodness of fit measure.
*   **IRLS:** Iteratively Reweighted Least Squares.
*   **Homoscedasticity:** Constant variance.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **GLM Structure** | $g(\mu) = X\beta$ | Model Def |
| **Poisson Variance** | $V(\mu) = \mu$ | Assumption |
| **Gamma Variance** | $V(\mu) = \mu^2$ | Assumption |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
