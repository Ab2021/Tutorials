# Customer Churn Prediction (Part 1) - Survival Analysis - Theoretical Deep Dive

## Overview
"It's not *if* they will leave, but *when*."
Traditional Churn Prediction asks: "Will they churn in the next 30 days?" (Binary Classification).
**Survival Analysis** asks: "How long until they churn?" (Time-to-Event).
This shift allows us to calculate the **Customer Lifetime** and intervene at the right moment.

---

## 1. Conceptual Foundation

### 1.1 The Flaw of Binary Classification

*   **Scenario:** Customer A churns on Day 31. Customer B churns on Day 365.
*   **Binary Model (30-day window):** Both are labeled "0" (Did not churn in 30 days).
*   **Result:** The model treats a loyal customer (B) the same as a risky customer (A).
*   **Survival Analysis:** Distinguishes between Day 31 and Day 365.

### 1.2 Censoring (The "Unknown" Future)

*   **Right-Censoring:** We observe a customer for 12 months. They are still active.
    *   *Do we know when they will churn?* No.
    *   *Do we know they survived at least 12 months?* Yes.
*   **Impact:** Standard Regression (MSE) fails because we don't have the "Target" ($Y$) for active customers. Survival models use this partial information.

---

## 2. Mathematical Framework

### 2.1 The Survival Function $S(t)$

*   **Definition:** Probability that a customer survives longer than time $t$.
    $$ S(t) = P(T > t) $$
*   **Properties:**
    *   $S(0) = 1$ (Everyone is alive at start).
    *   $S(\infty) = 0$ (Everyone eventually churns).
    *   Non-increasing.

### 2.2 The Hazard Function $h(t)$

*   **Definition:** Instantaneous rate of failure at time $t$, given survival up to $t$.
    $$ h(t) = \lim_{\Delta t \to 0} \frac{P(t \le T < t + \Delta t | T \ge t)}{\Delta t} $$
*   *Interpretation:* "The risk of dying right now."

### 2.3 Kaplan-Meier Estimator

*   **Non-Parametric:** Makes no assumptions about the distribution.
*   **Formula:**
    $$ \hat{S}(t) = \prod_{t_i \le t} \left( 1 - \frac{d_i}{n_i} \right) $$
    *   $d_i$: Number of churns at time $t_i$.
    *   $n_i$: Number of customers at risk just before $t_i$.

---

## 3. Theoretical Properties

### 3.1 Cox Proportional Hazards Model (CPH)

*   **Semi-Parametric:** Combines a baseline hazard $h_0(t)$ with covariates $X$.
*   **Formula:**
    $$ h(t|X) = h_0(t) \exp(\beta_1 X_1 + ... + \beta_k X_k) $$
*   **Assumption:** Proportional Hazards.
    *   If $\beta_1 > 0$, increasing $X_1$ increases the hazard by a constant factor *at all times*.
    *   *Example:* "Smokers are 2x more likely to die than Non-Smokers at age 30, and also at age 80."

### 3.2 Time-Varying Covariates

*   **Problem:** Standard Cox assumes $X$ is constant (e.g., Gender).
*   **Reality:** $X$ changes (e.g., Premium increases, Claims filed).
*   **Solution:** Extended Cox Model (Start-Stop format).

---

## 4. Modeling Artifacts & Implementation

### 4.1 Python Implementation (Lifelines)

```python
from lifelines import KaplanMeierFitter, CoxPHFitter
import pandas as pd

# 1. Data: Duration (T) and Event (E)
df = pd.DataFrame({
    'T': [5, 10, 12, 50, 2], # Months
    'E': [1, 1, 0, 0, 1],     # 1=Churn, 0=Censored
    'Premium': [100, 120, 100, 90, 200]
})

# 2. Kaplan-Meier (Visualizing the Curve)
kmf = KaplanMeierFitter()
kmf.fit(df['T'], event_observed=df['E'])
kmf.plot_survival_function()

# 3. Cox PH (Feature Importance)
cph = CoxPHFitter()
cph.fit(df, duration_col='T', event_col='E')
cph.print_summary()

# 4. Prediction
# Predict median survival time for a new customer
cph.predict_median(df.iloc[0:1])
```

### 4.2 Checking Assumptions

*   **Schoenfeld Residuals:** Test if the Proportional Hazards assumption holds.
    *   If p-value < 0.05, the assumption is violated.
    *   *Fix:* Stratify the variable (e.g., fit separate models for "High Risk" vs "Low Risk").

---

## 5. Evaluation & Validation

### 5.1 Concordance Index (C-Index)

*   **Definition:** Probability that, for a random pair of customers, the model correctly predicts who churns first.
*   **Range:** 0.5 (Random) to 1.0 (Perfect).
*   **Typical Good Score:** > 0.7.

### 5.2 Brier Score (Time-Dependent)

*   **Definition:** MSE of the predicted survival probability vs. actual status at time $t$.
*   *Goal:* Minimize Brier Score.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Immortal Time Bias**
    *   *Scenario:* You define a variable "Had a Claim".
    *   *Problem:* You have to *survive* long enough to have a claim.
    *   *Result:* The model thinks "Having a Claim" *increases* survival (because short-lived customers didn't have time to claim).
    *   *Fix:* Use Time-Varying Covariates properly.

2.  **Trap: Competing Risks**
    *   *Scenario:* Customer "Dies" (Death Claim) vs. Customer "Cancels" (Lapse).
    *   *Problem:* Standard Survival treats Death as "Censored" for Lapse.
    *   *Fix:* Fine-Gray Model (Competing Risks Regression).

---

## 7. Advanced Topics & Extensions

### 7.1 Random Survival Forests (RSF)

*   **Idea:** Ensemble of trees where each leaf node calculates a Kaplan-Meier curve.
*   **Pros:** Handles non-linearities and interactions automatically.
*   **Cons:** Harder to interpret than Cox Coefficients.

### 7.2 DeepSurv

*   **Idea:** Replace the linear part of Cox ($\beta X$) with a Neural Network.
*   **Formula:** $h(t|X) = h_0(t) \exp(f_{NN}(X))$.

---

## 8. Regulatory & Governance Considerations

### 8.1 "Price Walking" Detection

*   **Analysis:** Use Survival Analysis to see if "Tenure" is a predictor of "Higher Price".
*   **Regulation:** In many jurisdictions, penalizing loyalty is illegal.

---

## 9. Practical Example

### 9.1 Worked Example: The "Renewal Cliff"

**Scenario:**
*   **Observation:** Kaplan-Meier curve shows a steep drop at Month 12 and Month 24.
*   **Hypothesis:** Price increases at renewal are driving churn.
*   **Model:** Cox PH with `Price_Change_pct` as a variable.
*   **Result:** Hazard Ratio for `Price_Change` is 1.5. (10% price hike = 50% higher risk).
*   **Action:**
    *   **Segment A (High Elasticity):** Cap price increase at 5%.
    *   **Segment B (Low Elasticity):** Allow 10% increase.
*   **Outcome:** Retention at Month 12 improves by 8%.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Censoring** saves data.
2.  **Hazard Ratio** tells you the risk multiplier.
3.  **Time** is the target variable.

### 10.2 When to Use This Knowledge
*   **Actuarial:** "What is the expected duration of this policy block?"
*   **Marketing:** "When should we send the 'Thank You' gift?" (Before the median churn time).

### 10.3 Critical Success Factors
1.  **Data Hygiene:** Start/Stop dates must be precise.
2.  **Assumption Checking:** Don't blindly trust Cox PH. Check residuals.

### 10.4 Further Reading
*   **Davidson-Pilon:** "CamDavidsonPilon/lifelines" (Documentation).

---

## Appendix

### A. Glossary
*   **KM:** Kaplan-Meier.
*   **CPH:** Cox Proportional Hazards.
*   **HR:** Hazard Ratio.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Cox Hazard** | $h_0(t)e^{\beta X}$ | Risk Modeling |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
