# Mini Project (Part 2): Pricing Model Implementation - Theoretical Deep Dive

## Overview
In Part 2 of the Mini Project, we move from data preparation to the core of actuarial pricing: Generalized Linear Models (GLMs). We will build a Frequency-Severity model using Poisson and Gamma regression, explore the Tweedie distribution for pure premium modeling, and validate our models using industry-standard metrics like Lift Charts and the Gini Index.

---

## 1. Conceptual Foundation

### 1.1 Why GLMs?

**The Problem with OLS:**
Ordinary Least Squares (Linear Regression) assumes:
1.  Errors are Normally distributed.
2.  Variance is constant (Homoscedasticity).
3.  Target variable can be negative.

**Insurance Data Reality:**
1.  **Skewed:** Claims are highly right-skewed (many small, few large).
2.  **Heteroscedastic:** Variance increases with the mean (riskier policies have more volatile claims).
3.  **Non-Negative:** Claims cannot be negative.

**The GLM Solution:**
GLMs allow us to specify:
*   **Error Structure:** Poisson (Frequency), Gamma (Severity), Tweedie (Pure Premium).
*   **Link Function:** Log Link ($\ln(\mu) = X\beta$) ensures predictions are always positive.

### 1.2 The Frequency-Severity Approach

**Frequency Model:**
*   **Target:** Claim Count ($N$).
*   **Distribution:** Poisson (or Overdispersed Poisson / Negative Binomial).
*   **Offset:** $\ln(\text{Exposure})$. We model claims *per unit of exposure*.

**Severity Model:**
*   **Target:** Average Claim Cost ($Y$).
*   **Distribution:** Gamma.
*   **Weight:** Claim Count. (A severity observed from 10 claims is more reliable than from 1 claim).

**Pure Premium:**
$$ \text{Pure Premium} = \text{Frequency} \times \text{Severity} $$

### 1.3 The Tweedie Approach

**Direct Modeling:**
*   **Target:** Pure Premium (Loss / Exposure).
*   **Distribution:** Tweedie ($p \approx 1.5$).
*   **Advantage:** Single model to maintain. Handles the "Zero Mass" (policies with no claims) naturally.
*   **Disadvantage:** Harder to interpret "Frequency drivers" vs "Severity drivers."

---

## 2. Mathematical Framework

### 2.1 Poisson Regression (Frequency)

**Probability Mass Function:**
$$ P(N=k) = \frac{e^{-\lambda} \lambda^k}{k!} $$
**Link Function:**
$$ \ln(\lambda) = \beta_0 + \beta_1 x_1 + \dots + \beta_p x_p + \ln(\text{Exposure}) $$
**Interpretation of Coefficients:**
*   If $\beta_1 = 0.693$ for "Male", then $e^{0.693} = 2.0$.
*   Males have $2\times$ the frequency of the base class, holding all else constant.

### 2.2 Gamma Regression (Severity)

**Probability Density Function:**
$$ f(y) = \frac{1}{\Gamma(\alpha)\theta^\alpha} y^{\alpha-1} e^{-y/\theta} $$
**Variance Function:** $V(\mu) = \mu^2 / \nu$. (Variance is proportional to Mean squared).

### 2.3 Model Validation Metrics

**AIC / BIC:**
*   Penalized Likelihood measures. Lower is better.
*   Used for variable selection (Stepwise Regression).

**Deviance:**
*   A measure of "Goodness of Fit" (similar to SSE in OLS).
*   Scaled Deviance should be close to Degrees of Freedom.

---

## 3. Theoretical Properties

### 3.1 The Gini Index (Lorenz Curve)

**Concept:**
*   Sort policies by Predicted Risk (Low to High).
*   Plot Cumulative Exposure (X-axis) vs. Cumulative Loss (Y-axis).
*   **Line of Equality:** If the model is random, the plot is a $45^\circ$ line.
*   **Lorenz Curve:** The actual curve. It bows under the line.
*   **Gini:** $2 \times \text{Area between Line and Curve}$.
*   **Range:** 0 (Random) to 1 (Perfect). Good models are usually 0.3 - 0.4.

### 3.2 Lift Charts (Decile Analysis)

**Steps:**
1.  Sort test data by Predicted Pure Premium.
2.  Bin into 10 equal buckets (Deciles).
3.  Calculate "Actual Pure Premium" for each bucket.
4.  Plot Predicted vs. Actual.

**Interpretation:**
*   **Monotonicity:** Does Actual Loss increase steadily from Decile 1 to 10? (Good).
*   **Steepness:** The steeper the slope, the better the segmentation.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

*   **Cleaned DataFrame:** From Part 1.
*   **One-Hot Encoding:** Categorical variables must be converted to numeric dummies (unless using H2O or specialized libraries).

### 4.2 Preprocessing Steps

**Step 1: Train/Test Split**
*   Crucial to prevent overfitting.

**Step 2: Offset Creation**
*   For Poisson, create column `log_exposure = np.log(df['Exposure'])`.

### 4.3 Model Specification (Python Example)

Building Frequency and Severity GLMs using `statsmodels`.

```python
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# Load Data (Simulated from Part 1)
# Assume df exists with columns: ClaimCount, ClaimAmount, Exposure, AgeBand, Gender, VehicleAge

# 1. Frequency Model (Poisson)
# Formula: Count ~ AgeBand + Gender + VehicleAge + offset(log(Exposure))
df['log_exposure'] = np.log(df['Exposure'])

freq_formula = "ClaimCount ~ C(AgeBand) + C(Gender) + VehicleAge"
freq_model = smf.glm(formula=freq_formula, 
                     data=df, 
                     offset=df['log_exposure'],
                     family=sm.families.Poisson(link=sm.families.links.log())).fit()

print("Frequency Model Summary:")
print(freq_model.summary())

# 2. Severity Model (Gamma)
# Only train on records with Claims > 0
claims_df = df[df['ClaimCount'] > 0].copy()
claims_df['AvgClaimCost'] = claims_df['ClaimAmount'] / claims_df['ClaimCount']

sev_formula = "AvgClaimCost ~ C(AgeBand) + C(Gender)" # Maybe VehicleAge doesn't matter for severity
sev_model = smf.glm(formula=sev_formula,
                    data=claims_df,
                    weights=claims_df['ClaimCount'], # Weight by number of claims
                    family=sm.families.Gamma(link=sm.families.links.log())).fit()

print("\nSeverity Model Summary:")
print(sev_model.summary())

# 3. Prediction (Pure Premium)
# Predict on FULL dataset (Test set in reality)
df['PredFreq'] = freq_model.predict(df, offset=df['log_exposure'])
# Note: predict() gives expected count. Divide by exposure to get frequency rate?
# Actually, statsmodels predict() with offset gives the Count. 
# We want Rate = Count / Exposure.
df['PredFreqRate'] = freq_model.predict(df, offset=0) # Predict with offset=0 gives rate per unit

df['PredSev'] = sev_model.predict(df)
df['PredPurePrem'] = df['PredFreqRate'] * df['PredSev']

# 4. Validation: Lift Chart
# Sort by Prediction
df = df.sort_values('PredPurePrem')
df['Decile'] = pd.qcut(df['PredPurePrem'], 10, labels=False)

lift = df.groupby('Decile').agg({
    'PredPurePrem': 'mean',
    'ClaimAmount': 'sum',
    'Exposure': 'sum'
}).reset_index()

lift['ActualPurePrem'] = lift['ClaimAmount'] / lift['Exposure']
lift['AvgPredPurePrem'] = lift['PredPurePrem'] # Already a rate

print("\nLift Chart Data:")
print(lift[['Decile', 'AvgPredPurePrem', 'ActualPurePrem']])

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(lift['Decile'], lift['ActualPurePrem'], marker='o', label='Actual')
plt.plot(lift['Decile'], lift['AvgPredPurePrem'], marker='x', linestyle='--', label='Predicted')
plt.title('Double Lift Chart')
plt.xlabel('Decile (Low Risk -> High Risk)')
plt.ylabel('Pure Premium')
plt.legend()
plt.show()

# Interpretation:
# If the "Actual" line tracks the "Predicted" line closely, the model is calibrated.
# If "Actual" is flat while "Predicted" goes up, the model is overfitting (finding noise).
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Coefficients (Relativities):** The multipliers for the rating plan.
    *   Example: `C(AgeBand)[T.18-25] coef = 0.69`. Relativity = $e^{0.69} \approx 2.0$.
2.  **p-values:** Is the variable statistically significant?
3.  **Deviance:** Model fit statistic.

**Interpretation:**
*   **Base Level:** The intercept represents the base risk (e.g., Adult Female, Vehicle Age 0).
*   **Relativities:** Multiplicative factors applied to the base rate.

---

## 5. Evaluation & Validation

### 5.1 The "Double Lift" Chart

*   Compare **Model A** (Current Rating Plan) vs. **Model B** (New GLM).
*   Sort by the *Ratio* of Model A / Model B.
*   If Model B is better, it should identify that the policies where Ratio > 1 are actually safer, and Ratio < 1 are actually riskier.

### 5.2 Stability Testing

*   Bootstrap the training data and refit the GLM 100 times.
*   Check the variance of the coefficients.
*   If `AgeBand_18-25` varies from 1.5 to 2.5, the estimate is unstable.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Overfitting to the Tail**
    *   **Issue:** A few large claims drive the Gamma regression.
    *   **Fix:** Cap large losses (e.g., at $50k) for the basic pricing model. Price the excess layer separately.

2.  **Trap: Correlation vs. Interaction**
    *   **Issue:** Assuming Age and VehicleAge are independent.
    *   **Reality:** Young people drive old cars.
    *   **Fix:** Check for interactions or use non-linear models (GBM) to detect them, then add to GLM.

### 6.2 Implementation Challenges

1.  **Categorical Levels:**
    *   "Vehicle Model" has 2,000 levels. GLM will explode.
    *   **Solution:** Target Encoding or Credibility Weighting to group them.

---

## 7. Advanced Topics & Extensions

### 7.1 Regularized GLM (Elastic Net)

*   Adds a penalty term ($\lambda \sum \beta^2$) to the likelihood.
*   **L1 (Lasso):** Shrinks coefficients to zero (Feature Selection).
*   **L2 (Ridge):** Shrinks coefficients towards zero (Stability).
*   Useful when you have hundreds of correlated variables.

### 7.2 GAMs (Generalized Additive Models)

*   Instead of linear terms ($\beta x$), use smooth functions $f(x)$ (Splines).
*   Allows for non-linear relationships (e.g., Age curve is U-shaped) without manual binning.

---

## 8. Regulatory & Governance Considerations

### 8.1 Rate Filing

*   You cannot just upload a Python script to the regulator.
*   You must produce a **Rating Manual**: Base Rate $\times$ Age Factor $\times$ Territory Factor.
*   GLMs are preferred over "Black Box" ML models (XGBoost) because they output explicit factors.

---

## 9. Practical Example

### 9.1 Worked Example: Interpreting Coefficients

**Output:**
*   Intercept: -1.20
*   Age_Young: 0.50
*   Gender_M: 0.10

**Calculation:**
*   Base Frequency = $e^{-1.20} = 0.301$ (30.1%).
*   Young Male Frequency = $\exp(-1.20 + 0.50 + 0.10) = \exp(-0.60) = 0.548$ (54.8%).
*   **Relativity:** Young Male is $0.548 / 0.301 = 1.82$ times riskier than the Base.
*   Check: $e^{0.5} \times e^{0.1} = 1.648 \times 1.105 = 1.82$. (Multiplicative structure).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Poisson** for Frequency, **Gamma** for Severity.
2.  **Log Link** ensures positive predictions and multiplicative factors.
3.  **Lift Charts** prove the model works.

### 10.2 When to Use This Knowledge
*   **Pricing:** The bread and butter of P&C actuarial work.
*   **Underwriting:** Identifying bad risks to decline.
*   **Marketing:** Targeting the "decile 1" (lowest risk) customers.

### 10.3 Critical Success Factors
1.  **Validate Out-of-Sample:** Never trust training error.
2.  **Check the Residuals:** Look for patterns you missed.
3.  **Keep it Simple:** A simple GLM is better than a complex one if the lift is similar (Occam's Razor).

### 10.4 Further Reading
*   **Anderson et al.:** "A Practitioner's Guide to Generalized Linear Models".
*   **CAS Exam 8 Syllabus:** Advanced Ratemaking.

---

## Appendix

### A. Glossary
*   **GLM:** Generalized Linear Model.
*   **Link Function:** The function connecting the linear predictor to the mean.
*   **Deviance:** A measure of error.
*   **Offset:** The exposure term in the model.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Poisson PMF** | $e^{-\lambda}\lambda^k/k!$ | Frequency |
| **Log Link** | $\ln(\mu) = X\beta$ | Prediction |
| **AIC** | $2k - 2\ln(L)$ | Selection |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,000+*
