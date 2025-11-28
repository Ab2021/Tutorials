# Mini Project (Part 1): Data Simulation & Exploration - Theoretical Deep Dive

## Overview
This session kicks off the Phase 1 Mini Project, an end-to-end actuarial modeling exercise. Part 1 focuses on the foundation: Data. We explore techniques for simulating realistic insurance datasets (Policy and Claims), performing Exploratory Data Analysis (EDA) to understand risk characteristics, and engineering features for downstream modeling.

---

## 1. Conceptual Foundation

### 1.1 The Role of Synthetic Data

**Why Simulate?**
*   **Privacy:** Real insurance data is PII (Personally Identifiable Information) and highly regulated (GDPR/HIPAA).
*   **Stress Testing:** We can create "bad" scenarios (e.g., a hurricane) that haven't happened yet in history.
*   **Model Validation:** Since we know the *true* underlying parameters (Ground Truth), we can check if our GLMs recover them correctly.

**Components of Insurance Data:**
1.  **Policy Data (Exposure):** Who is insured? (Age, Vehicle, Limit, Deductible).
2.  **Claims Data (Losses):** What happened? (Date of Loss, Amount, Cause).

### 1.2 Exploratory Data Analysis (EDA) in Insurance

**Actuarial EDA Goals:**
*   **Data Integrity:** Are there negative premiums? Claims without policies? Dates of birth in the future?
*   **Univariate Analysis:** What is the distribution of Age? Is it skewed?
*   **Bivariate Analysis:** Does Claim Frequency increase with Age? (The "One-Way Analysis").
*   **Correlations:** Is Vehicle Value correlated with Driver Age? (Multicollinearity check).

### 1.3 Feature Engineering

**Transforming Raw Data into Risk Factors:**
*   **Binning:** Converting continuous Age into "Age Bands" (16-21, 22-30, etc.) to capture non-linear risk.
*   **Interactions:** `Age * Gender` (Young Males are riskier than Young Females, but Old Males might be safer than Old Females).
*   **Vehicle Symboling:** Grouping thousands of car models into "Risk Groups" based on physical characteristics (Weight, Horsepower).

---

## 2. Mathematical Framework

### 2.1 Simulation Algorithms

**Inverse Transform Sampling:**
To generate a random variable $X$ from CDF $F(x)$:
1.  Generate $U \sim \text{Uniform}(0, 1)$.
2.  Compute $X = F^{-1}(U)$.
*   *Example:* Generating Exponential claims. $X = -\frac{1}{\lambda} \ln(1-U)$.

**Thinning (Acceptance-Rejection):**
Used for generating non-homogeneous Poisson processes (e.g., seasonality in claims).

### 2.2 Frequency-Severity Decomposition

**Frequency ($N$):**
*   Usually Poisson($\lambda$) or Negative Binomial($r, p$).
*   $\lambda$ depends on covariates: $\lambda_i = \exp(X_i \beta)$.

**Severity ($Y$):**
*   Usually Gamma($\alpha, \theta$) or Lognormal($\mu, \sigma$).
*   Mean $\mu_i = \exp(X_i \gamma)$.

**Total Loss ($S$):**
$$ S = \sum_{j=1}^N Y_j $$

### 2.3 One-Way Analysis (Relativities)

Comparing the "Actual Loss Ratio" or "Frequency" across levels of a factor.
$$ \text{Relativity}_k = \frac{\text{Mean Frequency for Group } k}{\text{Overall Mean Frequency}} $$
*   *Visual:* A bar chart showing how risk varies by Age Band.

---

## 3. Theoretical Properties

### 3.1 Data Quality Issues

*   **Right Censoring:** Recent policies haven't had time to claim yet (IBNR).
*   **Left Truncation:** Deductibles hide small losses.
*   **Missingness:** "Not Indicated" values in optional fields.

### 3.2 Correlation vs. Causation

*   **Red Cars:** Red cars might have higher frequency, but is it the color? Or is it that aggressive drivers buy red cars?
*   **Actuarial Stance:** We care about *prediction* (Correlation), but for *fairness* (Regulation), we often need Causation.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Data Requirements

*   **Policy Table:** PolicyID, StartDate, EndDate, Premium, Age, Gender, VehicleType.
*   **Claims Table:** ClaimID, PolicyID, DateOfLoss, PaidAmount, IncurredAmount.

### 4.2 Preprocessing Steps

**Step 1: Earned Exposure Calculation**
*   Convert Start/End dates into "Exposure Years" for the analysis period (e.g., Calendar Year 2023).
*   *Formula:* $\text{Exposure} = \frac{\min(\text{End}, \text{ValDate}) - \max(\text{Start}, \text{Jan1})}{365}$.

**Step 2: Joining**
*   Left Join Policy Data with Claims Data.
*   Fill non-claimants with Zero Loss.

### 4.3 Model Specification (Python Example)

Generating a synthetic Auto Insurance dataset.

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(42)
n_policies = 10000

# 1. Simulate Policy Data
ids = np.arange(n_policies)
ages = np.random.randint(18, 80, n_policies)
genders = np.random.choice(['M', 'F'], n_policies)
vehicle_age = np.random.randint(0, 20, n_policies)

# Create DataFrame
df = pd.DataFrame({
    'PolicyID': ids,
    'Age': ages,
    'Gender': genders,
    'VehicleAge': vehicle_age,
    'Exposure': np.random.uniform(0.5, 1.0, n_policies) # Partial year exposure
})

# 2. Define True Risk Parameters (Ground Truth)
# Base Frequency = 10%
# Age Factor: Young (18-25) are 2x riskier
# Gender Factor: Males are 1.2x riskier
def get_lambda(row):
    base_freq = 0.10
    age_factor = 2.0 if row['Age'] < 25 else 1.0
    gender_factor = 1.2 if row['Gender'] == 'M' else 1.0
    return base_freq * age_factor * gender_factor * row['Exposure']

df['TrueLambda'] = df.apply(get_lambda, axis=1)

# 3. Simulate Claims (Frequency)
# Poisson Draw
df['ClaimCount'] = np.random.poisson(df['TrueLambda'])

# 4. Simulate Severity (Gamma)
# Mean severity = $2000. Shape=2, Scale=1000.
# Assume Severity is independent of covariates for simplicity
shape, scale = 2.0, 1000.0

def get_severity(count):
    if count == 0: return 0
    return np.sum(np.random.gamma(shape, scale, count))

df['ClaimAmount'] = df['ClaimCount'].apply(get_severity)

# 5. Exploratory Data Analysis (EDA)

# Binning Age
df['AgeBand'] = pd.cut(df['Age'], bins=[17, 25, 40, 60, 100], labels=['18-25', '26-40', '41-60', '60+'])

# One-Way Analysis: Frequency by Age Band
one_way = df.groupby('AgeBand').agg({
    'ClaimCount': 'sum',
    'Exposure': 'sum'
}).reset_index()

one_way['Frequency'] = one_way['ClaimCount'] / one_way['Exposure']

print("One-Way Analysis (Frequency by Age):")
print(one_way)

# Visualization
plt.figure(figsize=(10, 5))
sns.barplot(x='AgeBand', y='Frequency', data=one_way, palette='viridis')
plt.title('Observed Claim Frequency by Age Band')
plt.ylabel('Frequency (Claims / Exposure)')
plt.show()

# Correlation Matrix
corr = df[['Age', 'VehicleAge', 'ClaimCount', 'ClaimAmount']].corr()
print("\nCorrelation Matrix:")
print(corr)

# Interpretation:
# We should see a clear spike in Frequency for the 18-25 bucket.
# This confirms our simulation logic worked and prepares us for modeling.
```

### 4.4 Model Outputs & Interpretation

**Primary Outputs:**
1.  **Cleaned Dataset:** Ready for GLM (Part 2).
2.  **EDA Report:** Identifying key drivers of risk (Age, Gender).
3.  **Bad Data Flags:** Identifying records to exclude.

**Interpretation:**
*   **Non-Monotonicity:** If the "60+" group has higher frequency than "41-60", is it real (seniors driving worse) or noise?
*   **Sparse Segments:** If "VehicleAge > 15" has only 10 policies, we can't trust the average.

---

## 5. Evaluation & Validation

### 5.1 Sanity Checks

*   **Avg Frequency:** Should be around 10-15% for Auto.
*   **Avg Severity:** Should be around $3k-$5k.
*   **Pure Premium:** Freq $\times$ Sev.

### 5.2 Train/Test Split

*   **Random Split:** 70/30.
*   **Time-Based Split:** Train on 2022, Test on 2023. (Better for insurance to test "out of time" performance).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Target Leakage**
    *   **Issue:** Including "Post-Loss" variables in the model (e.g., "Days to Report").
    *   **Reality:** You don't know "Days to Report" when you write the policy.
    *   **Fix:** Only use variables known at the Quote Date.

2.  **Trap: Zero-Inflation**
    *   **Issue:** 90% of policies have Zero claims.
    *   **Reality:** Standard OLS regression fails here.
    *   **Fix:** Use Tweedie GLM or Zero-Inflated Poisson models.

### 6.2 Implementation Challenges

1.  **Large Data:**
    *   Insurance datasets can be millions of rows.
    *   **Solution:** Use `polars` or `dask` instead of `pandas` for big data.

---

## 7. Advanced Topics & Extensions

### 7.1 SMOTE (Synthetic Minority Over-sampling Technique)

*   Used for Fraud Detection (where fraud is rare, < 1%).
*   Generates synthetic examples of the minority class to balance the dataset.

### 7.2 Geocoding

*   Turning "Zip Code" into "Latitude/Longitude" or "Distance to Coast."
*   Crucial for Property pricing.

---

## 8. Regulatory & Governance Considerations

### 8.1 Unfair Discrimination

*   **Protected Classes:** Race, Religion, Origin.
*   **Proxies:** Regulators check if "Zip Code" is just a proxy for Race.
*   **EDA Check:** We must check if our model inputs are legally allowed.

---

## 9. Practical Example

### 9.1 Worked Example: Feature Engineering "Vehicle Power"

**Scenario:**
*   Raw Data: `Horsepower` (HP) and `Weight` (Lbs).
*   **Hypothesis:** High HP in a light car is dangerous. Low HP in a heavy car is safe.

**Feature Creation:**
*   `PowerToWeight` = `Horsepower` / `Weight`.
*   **Binning:** Deciles of PowerToWeight.
*   **Check:** Run One-Way Analysis on the Deciles.
*   **Result:** Top decile has 2x frequency. This is a powerful feature.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Simulation** allows us to control the ground truth.
2.  **EDA** validates our assumptions and finds dirty data.
3.  **Feature Engineering** turns raw fields into predictive signals.

### 10.2 When to Use This Knowledge
*   **Every Project:** Data prep is 80% of the work.
*   **Pricing:** Building the rating plan.
*   **Reserving:** Checking for changes in mix of business.

### 10.3 Critical Success Factors
1.  **Know the Business:** Don't just look at stats. Know that "Symbol 7" means "Hired Auto."
2.  **Visualize:** A plot is worth 1000 tables.
3.  **Document:** Record every cleaning step for the regulator.

### 10.4 Further Reading
*   **Goldburd et al.:** "Generalized Linear Models for Insurance Rating" (CAS Monograph 5).
*   **Kaggle:** "Prudential Life Insurance Assessment" competition kernels.

---

## Appendix

### A. Glossary
*   **EDA:** Exploratory Data Analysis.
*   **One-Way:** Univariate analysis of risk.
*   **Exposure:** Time on risk (0 to 1 year).
*   **Pure Premium:** Loss Cost per unit of exposure.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Exposure** | $(End - Start)/365$ | Normalization |
| **Frequency** | $Count / Exposure$ | Risk Metric |
| **Severity** | $Loss / Count$ | Cost Metric |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 1,000+*
