# Regulatory & Fairness Constraints - Theoretical Deep Dive

## Overview
The final session of Phase 2 addresses the "Third Rail" of insurance modeling: **Ethics and Regulation**. A model can be statistically perfect (high Gini) but illegal or unethical. We explore the tension between **Actuarial Fairness** (price = risk) and **Social Fairness** (equal access), covering **Protected Classes**, **Disparate Impact**, and the modern toolkit for **Bias Mitigation**.

---

## 1. Conceptual Foundation

### 1.1 The Two Definitions of Fairness

**1. Actuarial Fairness:**
*   "It is fair to charge higher rates to groups that have higher losses."
*   *Example:* Young men crash more than young women. Therefore, charging men more is "fair" because it reflects reality.

**2. Social Fairness (Solidarity):**
*   "It is unfair to penalize individuals for traits they cannot control."
*   *Example:* Charging more based on Race or Ethnicity is illegal and immoral, even if the data shows a correlation.
*   **The Conflict:** What about Credit Score? It predicts risk, but it correlates with Race/Income. Is it fair?

### 1.2 Protected Classes

Traits you generally **cannot** use in pricing (varies by jurisdiction):
*   **Race, Religion, National Origin:** Universally banned.
*   **Gender:** Banned in EU (2012), CA, HI, MA, MI, MT, NC, PA.
*   **Credit Score:** Banned/Restricted in CA, MA, HI, MD, MI.
*   **Marital Status:** Restricted in some states.

### 1.3 Disparate Treatment vs. Disparate Impact

**Disparate Treatment (Intentional):**
*   Explicitly using "Race" as a variable in the GLM.
*   *Verdict:* Illegal everywhere.

**Disparate Impact (Unintentional):**
*   Using a neutral variable (e.g., "Zip Code") that highly correlates with a protected class (Race).
*   *Verdict:* The legal battleground.
*   **The Standard:** Is there a "Legitimate Business Necessity"? Is there a less discriminatory alternative?

---

## 2. Mathematical Framework

### 2.1 Fairness Metrics

How do we measure if a model is biased?
Let $Y=1$ be the favorable outcome (Low Rate), $S=1$ be the protected group (e.g., Minority).

**1. Demographic Parity:**
$$ P(\hat{Y}=1 | S=1) = P(\hat{Y}=1 | S=0) $$
*   The average rate for Group S should equal the average rate for Group Non-S.
*   *Critique:* Ignores risk differences. If Group S actually crashes more, this forces cross-subsidy.

**2. Equalized Odds:**
$$ P(\hat{Y}=1 | Y=1, S=1) = P(\hat{Y}=1 | Y=1, S=0) $$
*   Among people who *are* safe risks ($Y=1$), the model should treat S and Non-S equally.
*   *Preferred by Data Scientists:* It allows for different base rates if the underlying risk is different.

### 2.2 Proxy Analysis

**Linear Regression Check:**
1.  Regress the "Suspect Variable" (e.g., Credit Score) against Protected Classes.
    $$ \text{Credit} = \beta_0 + \beta_1 \text{Race} + \epsilon $$
2.  If $R^2$ is high, Credit is a proxy for Race.

---

## 3. Theoretical Properties

### 3.1 The Efficiency-Fairness Trade-off

*   **Efficiency:** Maximizing accuracy (Gini).
*   **Fairness:** Minimizing Disparate Impact.
*   **The Pareto Frontier:** You usually cannot have both. Removing Credit Score might drop Gini by 5 points.
    *   *Society's Choice:* How much accuracy are we willing to sacrifice for fairness?

### 3.2 Redlining

**Historical Redlining:** Drawing red lines on a map around minority neighborhoods and refusing to write insurance.
**Digital Redlining:**
*   Using "Zip Code" in a GLM where the relativity for the minority zip code is 3.0x the white zip code.
*   *Regulatory Check:* Regulators look at maps of your residuals. If you consistently overcharge minority areas, you are in trouble.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Bias Detection in Python

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Simulate Data
np.random.seed(42)
n = 10000
# Protected Class (0=Majority, 1=Minority)
protected = np.random.binomial(1, 0.3, n) 

# True Risk (Correlated with Protected due to systemic issues)
# Minority group has slightly higher risk in this synthetic example
risk_score = np.random.normal(100, 20, n) + 10 * protected 

# Proxy Variable (e.g., Credit Score) - Highly correlated with Protected
# Lower Credit = Higher Risk
credit_score = 800 - 2 * risk_score + np.random.normal(0, 10, n)

# Model Prediction (Linear Regression on Credit)
# We don't use 'protected', but we use 'credit'
pred_premium = 1000 - 1.5 * (credit_score - 600)

df = pd.DataFrame({
    'Protected': protected,
    'Risk': risk_score,
    'Credit': credit_score,
    'Premium': pred_premium
})

# 1. Disparate Impact Analysis (Average Premium)
avg_prem = df.groupby('Protected')['Premium'].mean()
print("Average Premium by Group:")
print(avg_prem)

# Impact Ratio: Minority / Majority
impact_ratio = avg_prem[1] / avg_prem[0]
print(f"Impact Ratio: {impact_ratio:.2f}")
# If Ratio > 1.10, regulators might flag it.

# 2. Residual Analysis (Equalized Odds)
# Does the model overcharge Minority risks *relative to their true risk*?
df['Residual'] = df['Premium'] - (10 * df['Risk']) # Assume 10*Risk is "Fair" Price
avg_resid = df.groupby('Protected')['Residual'].mean()

print("\nAverage Residual (Overpricing) by Group:")
print(avg_resid)

# Visualization
sns.boxplot(x='Protected', y='Premium', data=df)
plt.title('Premium Distribution by Protected Class')
plt.show()

# Interpretation:
# Even though we didn't use 'Protected', the 'Premium' is higher for Group 1.
# This is because 'Credit' is a proxy.
```

### 4.2 Mitigation Strategies

1.  **Drop the Variable:** Remove Credit Score. (Simple, but hurts accuracy).
2.  **Orthogonalization:**
    *   Regress Credit on Race.
    *   Take the *Residual* (The part of Credit NOT explained by Race).
    *   Use the Residual in the pricing model.
    *   *Result:* You keep the "risk" signal of Credit but remove the "race" signal.

---

## 5. Evaluation & Validation

### 5.1 The "Control Variable" Method

*   Train a model *with* the Protected Class as a feature.
*   Train a model *without* it.
*   Compare the coefficients of other variables.
*   If the coefficient for "Zip Code" jumps when you remove "Race", then Zip Code is picking up the Race signal.

### 5.2 Fairness-Aware Machine Learning

*   **Adversarial Debiasing:** A neural network with two heads.
    *   Head 1: Predicts Loss (Minimize Error).
    *   Head 2: Predicts Race (Maximize Error).
    *   *Goal:* Learn features that predict loss but *cannot* predict race.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: "We don't collect Race data"**
    *   **Issue:** Insurers often don't ask for race.
    *   **Consequence:** You can't test for bias if you don't have the labels.
    *   **Solution:** BISG (Bayesian Improved Surname Geocoding). Infer race from Name + Zip Code to perform disparate impact testing.

2.  **Trap: Causation vs. Correlation**
    *   **Issue:** "Credit score predicts loss, so it's fair."
    *   **Counter:** Does low credit *cause* accidents? Or is it just a proxy for poverty/stress?
    *   **Trend:** Regulators are demanding *Causal* justification for rating factors.

### 6.2 Implementation Challenges

1.  **State Heterogeneity:**
    *   New York: "No Credit Scoring."
    *   Florida: "Credit Scoring is fine."
    *   **Result:** You need different models for different states.

---

## 7. Advanced Topics & Extensions

### 7.1 Causal Inference Graphs (DAGs)

*   Mapping the causal flow: Race $\to$ Income $\to$ Credit $\to$ Loss.
*   **Intervention:** We want to block the path Race $\to$ Premium, but keep Risk $\to$ Premium.

### 7.2 Telematics as the Great Equalizer

*   **Promise:** Driving behavior (Hard Braking, Speeding) is color-blind.
*   **Reality:** Even Telematics can be biased (e.g., if minority neighborhoods have more potholes or traffic, leading to more "events").

---

## 8. Regulatory & Governance Considerations

### 8.1 The "Black Box" Ban

*   Some states (e.g., Colorado) are passing laws requiring AI systems to be explainable and tested for bias.
*   **Impact:** GBMs and Neural Nets are hard to approve. GLMs are preferred because you can point to the exact coefficient.

### 8.2 Third-Party Data

*   If you buy a "Marketing Score" from a vendor, *you* are responsible for its bias. You cannot blame the vendor.

---

## 9. Practical Example

### 9.1 Worked Example: The "Price Optimization" Ban

**Scenario:**
*   Model says: "Elderly people are less likely to switch insurers."
*   Optimization: "Charge elderly people +10% loyalty penalty."
*   **Outcome:** This is technically "accurate" (they will pay it), but regulators define it as **Unfair Discrimination**.
*   **Rule:** Price must be based on *Cost*, not *Willingness to Pay*.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Disparate Impact** is the main risk.
2.  **Proxies** (Credit, Zip) are the main culprits.
3.  **Fairness** is a trade-off with Accuracy.

### 10.2 When to Use This Knowledge
*   **Every Filing:** You must certify that rates are not unfairly discriminatory.
*   **Model Review:** The first question a Chief Actuary asks: "Is this variable a proxy?"

### 10.3 Critical Success Factors
1.  **Test for Bias:** Don't wait for the regulator to find it.
2.  **Document Justification:** Have a "Business Necessity" paper ready for every variable.
3.  **Stay Updated:** Regulations change monthly.

### 10.4 Further Reading
*   **O'Neil:** "Weapons of Math Destruction".
*   **CAS Research Paper:** "Methods for Quantifying Discriminatory Effects".

---

## Appendix

### A. Glossary
*   **BISG:** Bayesian Improved Surname Geocoding (Race imputation).
*   **Redlining:** Geographic discrimination.
*   **Solidarity:** The principle that risk should be shared across society.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Demographic Parity** | $E[\hat{Y}|S=1] = E[\hat{Y}|S=0]$ | Fairness Metric |
| **Impact Ratio** | $\mu_{minority} / \mu_{majority}$ | Disparate Impact |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
