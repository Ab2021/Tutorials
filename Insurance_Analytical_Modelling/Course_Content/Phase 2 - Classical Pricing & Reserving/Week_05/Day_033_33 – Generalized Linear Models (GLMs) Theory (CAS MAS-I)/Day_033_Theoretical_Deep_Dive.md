# Generalized Linear Models (GLMs) Theory (Part 3) - Theoretical Deep Dive

## Overview
In this final theory session on GLMs, we tackle the complexities of real-world modeling. We move beyond simple "Main Effects" models to handle **Interactions**, **Non-Linearity (Polynomials)**, and **Categorical Encoding**. We also introduce **Regularization (Elastic Net)**, a modern bridge between classical GLMs and Machine Learning, essential for handling high-dimensional data.

---

## 1. Conceptual Foundation

### 1.1 Beyond Main Effects

**Main Effects Assumption:** The effect of Age is independent of the effect of Gender.
*   *Reality:* Young Males might be much riskier than the sum of "Young" + "Male" would suggest.
*   **Interaction Term:** A multiplier that activates only when *both* conditions are true.

### 1.2 Categorical Variables in GLMs

GLMs require numerical input. We must encode categories (e.g., "Vehicle Type: Sedan, SUV, Truck").
*   **Dummy Encoding (One-Hot):** Creates binary columns.
*   **Base Level (Reference):** The category left out. All other coefficients are *relative* to this base.
    *   *Choice of Base:* Should be the largest group (most stable) or the "Standard" risk (e.g., Age 30-40).

### 1.3 The Role of Offsets and Weights

**Offset:**
*   A variable with a fixed coefficient of 1.0.
*   Used for **Exposure**. We model $\frac{Claims}{Exposure}$, but fit $Claims$ with offset $\ln(Exposure)$.

**Weight:**
*   Used for **Credibility/Variance**.
*   A record representing 100 policies should have $100\times$ the influence of a single policy.
*   In GLMs, weights scale the variance: $\text{Var}(Y) = \frac{\phi V(\mu)}{\omega}$.

---

## 2. Mathematical Framework

### 2.1 Interactions

**Mathematical Form:**
$$ \ln(\mu) = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \beta_3 (x_1 \cdot x_2) $$

**Interpretation (Log Link):**
*   Base: $e^{\beta_0}$
*   Effect of $x_1$: $e^{\beta_1}$
*   Effect of $x_2$: $e^{\beta_2}$
*   **Interaction:** $e^{\beta_3}$.
*   Total Effect = Base $\times$ $x_1$ Factor $\times$ $x_2$ Factor $\times$ Interaction Factor.

**Example:**
*   Young Factor = 1.5
*   Male Factor = 1.2
*   Young $\times$ Male Interaction = 1.1
*   Total = $1.5 \times 1.2 \times 1.1 = 1.98$. (Without interaction, it would be $1.8$).

### 2.2 Polynomials & Splines

**Polynomials:**
$$ \ln(\mu) = \beta_0 + \beta_1 \text{Age} + \beta_2 \text{Age}^2 $$
*   Captures U-shaped curves (e.g., Young and Old are risky, Middle is safe).

**Splines (GAMs):**
*   Piecewise polynomials joined at "knots."
*   More flexible than global polynomials.

### 2.3 Regularization (Elastic Net)

**The Problem:**
*   If we have 1000 zip codes, OLS/GLM will overfit (high variance).
*   Some zip codes have 1 claim. The coefficient will be extreme ($-\infty$ or $+\infty$).

**The Solution:** Penalize the magnitude of coefficients.
$$ \min_{\beta} - \mathcal{L}(\beta) + \lambda \left( \alpha \sum |\beta_j| + (1-\alpha) \sum \beta_j^2 \right) $$

*   **Lasso ($\alpha=1$):** $L_1$ penalty. Forces coefficients to **Zero**. (Feature Selection).
*   **Ridge ($\alpha=0$):** $L_2$ penalty. Shrinks coefficients towards zero. (Handles Multicollinearity).
*   **Elastic Net ($0 < \alpha < 1$):** Best of both.

---

## 3. Theoretical Properties

### 3.1 Aliasing (Multicollinearity)

**Perfect Aliasing:**
*   $x_1 + x_2 = 1$ (e.g., Male + Female = 1).
*   The matrix $X^TX$ is singular. The model cannot be solved.
*   **Fix:** Drop one column (Base Level).

**Near Aliasing:**
*   Variables are highly correlated (e.g., Vehicle Value and Horsepower).
*   Coefficients become unstable (large standard errors).
*   **Fix:** Ridge Regression or remove one variable.

### 3.2 Canonical Link vs. Non-Canonical

*   **Canonical:** Simplifies the math (Sufficient Statistics).
*   **Non-Canonical:** Often needed for business logic.
    *   *Example:* Binomial Canonical is Logit. But in retention modeling, we might use Probit or Log-Log.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Categorical Encoding Strategies

1.  **One-Hot:** Standard. Good for low cardinality (< 50 levels).
2.  **Target Encoding (Mean Encoding):**
    *   Replace "Zip Code" with "Average Loss in Zip Code".
    *   **Risk:** Target Leakage. Must use Cross-Validation or Smoothing.
3.  **Frequency Encoding:** Replace category with its count.

### 4.2 Model Specification (Python Example)

Modeling Interactions and using Regularized GLM (GLMNet style).

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import PoissonRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Simulate Data
np.random.seed(42)
n = 1000
age = np.random.uniform(18, 60, n)
gender = np.random.choice(['M', 'F'], n)
exposure = np.random.uniform(0.5, 1.0, n)

# Interaction: Young Males are extra risky
is_young = (age < 25).astype(int)
is_male = (gender == 'M').astype(int)
interaction = is_young * is_male

true_mu = np.exp(-2 + 0.5*is_young + 0.2*is_male + 0.4*interaction) * exposure
y = np.random.poisson(true_mu)

df = pd.DataFrame({'y': y, 'age': age, 'gender': gender, 'exposure': exposure})
df['log_exp'] = np.log(df['exposure'])

# 1. Statsmodels with Interaction
# Formula: age * gender includes main effects AND interaction
model_int = smf.glm("y ~ age * gender", data=df, offset=df['log_exp'],
                    family=sm.families.Poisson()).fit()

print("Interaction Model Summary:")
print(model_int.summary())

# 2. Regularized GLM (Scikit-Learn)
# Scikit-Learn penalizes the intercept by default, which is bad for GLMs. 
# We usually don't penalize intercept.
# Also, sklearn doesn't support 'offset' directly in fit() for some versions, 
# but PoissonRegressor does not have an offset param in older versions.
# Workaround: Move offset to LHS? No, Poisson doesn't work like that.
# Modern sklearn (v1.0+) has sample_weight, but offset is different.
# Actually, for Poisson, we can model Rate = y / exposure with sample_weight=exposure?
# No, that's for Frequency.
# Correct Sklearn approach: Use log(exposure) as a fixed feature with coef=1? No.
# We will model Frequency (y/exposure) and weight by exposure.

df['freq'] = df['y'] / df['exposure']

# Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), ['age']),
        ('cat', OneHotEncoder(drop='first'), ['gender'])
    ])

# Poisson Regressor with L2 penalty (Alpha)
# Note: Sklearn uses 'alpha' for lambda (strength).
reg_glm = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', PoissonRegressor(alpha=1.0, max_iter=300))
])

# Fit
# We weight by exposure to simulate the offset effect for rates
reg_glm.fit(df[['age', 'gender']], df['freq'], regressor__sample_weight=df['exposure'])

print("\nRegularized Coefficients:")
print(reg_glm.named_steps['regressor'].coef_)
print(reg_glm.named_steps['regressor'].intercept_)

# Interpretation:
# The coefficients are "shrunk" compared to the raw GLM.
# This helps if we had 100 features.
```

### 4.3 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Interaction Coefficient:** Is it significant?
    *   If `Age:Gender[T.M]` is positive, the effect of Age depends on Gender.
2.  **Regularization Path:** How coefficients change as we increase $\lambda$.

**Interpretation:**
*   **Hierarchy Principle:** If an interaction $x_1 \cdot x_2$ is significant, you *must* include the main effects $x_1$ and $x_2$, even if they are not significant.
*   **Lasso:** "The model dropped `VehicleColor`. It deemed it irrelevant."

---

## 5. Evaluation & Validation

### 5.1 Interaction Plots

*   Plot "Predicted Frequency vs. Age" for Males and Females separately.
*   **Parallel Lines:** No interaction (Additive in Log scale).
*   **Crossing/Diverging Lines:** Interaction exists.

### 5.2 Stability of Categorical Levels

*   If a category has high standard error, merge it with the Base Level or a similar group.
*   **Credibility Weighting:** $\hat{\beta}_{adj} = Z \hat{\beta}_{GLM} + (1-Z) \hat{\beta}_{Prior}$.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: The Dummy Variable Trap**
    *   **Issue:** Including all levels of a category plus the intercept.
    *   **Result:** Perfect multicollinearity.
    *   **Fix:** Always drop one level (`drop='first'`).

2.  **Trap: Interpreting Polynomials**
    *   **Issue:** "The coefficient for $Age^2$ is 0.001."
    *   **Reality:** Small coefficients on squared terms can have huge effects at high values of Age. Always visualize the curve.

### 6.2 Implementation Challenges

1.  **Sparse Interactions:**
    *   "Young Drivers" $\times$ "Ferrari".
    *   You might have 0 observations. The model will crash or output garbage.
    *   **Solution:** Check cross-tabs before modeling.

---

## 7. Advanced Topics & Extensions

### 7.1 GAMs (Generalized Additive Models)

*   Instead of guessing polynomials ($x, x^2, x^3$), let the data speak.
*   $g(\mu) = \beta_0 + f_1(x_1) + f_2(x_2)$.
*   $f(x)$ are smooth spline functions.
*   **PyGAM** or R's **mgcv** are standard tools.

### 7.2 Gradient Boosting (GBM) as a GLM

*   XGBoost/LightGBM with `objective='poisson'` is essentially a GLM where the linear predictor is replaced by a sum of trees.
*   Automatically finds interactions and non-linearities.
*   **Hybrid Approach:** Use GBM to find interactions, then hard-code them into a GLM for regulatory approval.

---

## 8. Regulatory & Governance Considerations

### 8.1 "Unfair" Interactions

*   **Scenario:** `CreditScore` $\times$ `ZipCode`.
*   **Regulator:** "Are you using Zip Code as a proxy for Race?"
*   **Rule:** Some states ban specific interactions or require justification for every single one.

---

## 9. Practical Example

### 9.1 Worked Example: Base Level Selection

**Variable:** Vehicle Use.
*   Levels: Pleasure (80%), Commute (15%), Business (4%), Farm (1%).

**Choice:**
*   **Bad Base:** Farm. (Smallest group. All other coefficients will have huge standard errors).
*   **Good Base:** Pleasure. (Largest group. Stable reference).

**Interpretation:**
*   If Base = Pleasure, and Commute Coef = 0.10.
*   "Commuters are 10% riskier than Pleasure drivers."

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Interactions** capture complexity.
2.  **Offsets** handle exposure.
3.  **Regularization** handles high dimensionality.

### 10.2 When to Use This Knowledge
*   **Refining the Model:** When the simple model fails to capture the "Young Male" effect.
*   **Modern Pricing:** Using Elastic Net for "Symboling" (thousands of car models).

### 10.3 Critical Success Factors
1.  **Check Significance:** Don't add interactions just because you can.
2.  **Visualize:** Plot the interaction effects.
3.  **Standardize:** Scale numeric variables before using Regularization.

### 10.4 Further Reading
*   **Hastie, Tibshirani, Friedman:** "The Elements of Statistical Learning".
*   **Goldburd et al.:** "GLMs for Insurance Rating" (Chapter on Interactions).

---

## Appendix

### A. Glossary
*   **Interaction:** The effect of one variable depends on another.
*   **Offset:** A fixed component of the linear predictor.
*   **Lasso:** L1 Regularization.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Interaction** | $\beta_3 x_1 x_2$ | Synergy |
| **Elastic Net** | $\lambda(\alpha L_1 + (1-\alpha)L_2)$ | Penalty |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
