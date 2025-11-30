# Advanced Reserving (Machine Learning) - Theoretical Deep Dive

## Overview
Traditional reserving (Chain Ladder) aggregates data into triangles, destroying valuable information. **Individual Claims Reserving (ICR)** uses the full granularity of the data. We explore how **Machine Learning** (XGBoost, Neural Networks) and **Survival Analysis** are revolutionizing reserving by predicting the lifecycle of *each* claim individually.

---

## 1. Conceptual Foundation

### 1.1 The Granularity Gap

**Aggregate (Triangle):**
*   "The average claim from 2020 developed by 10%."
*   Ignores: Claimant age, injury type, attorney involvement, adjustor notes.

**Individual (Micro):**
*   "Claim #123 (Back Injury, Attorney Rep, 45yo Male) has a 80% chance of staying open for 2 years."
*   **Promise:** Better accuracy, faster reaction to trend shifts (e.g., a specific lawyer filing more suits).

### 1.2 Survival Analysis in Reserving

*   **Time-to-Event:** We model the time until "Claim Closure" or "Next Payment".
*   **Censoring:** Open claims are "Right Censored" (we know they lasted *at least* $t$ months).
*   **Hazard Function:** The instantaneous probability of closing at time $t$, given it was open until $t$.

### 1.3 Machine Learning for Case Reserves

*   **Goal:** Predict the *Ultimate Loss* given the current snapshot of features.
*   **Target:** $Y = \text{Ultimate Loss}$.
*   **Features:** $X = [\text{Paid to Date}, \text{Case Reserve}, \text{Injury Code}, \text{Description Text}, \dots]$.
*   **Algorithm:** Gradient Boosted Trees (XGBoost/LightGBM) are the industry standard for tabular data.

---

## 2. Mathematical Framework

### 2.1 Cox Proportional Hazards Model

$$ h(t | X) = h_0(t) \cdot \exp(\beta_1 X_1 + \dots + \beta_k X_k) $$
*   $h_0(t)$: Baseline hazard (how claims generally close over time).
*   $\exp(\beta X)$: How features shift the baseline (e.g., Attorney involvement reduces closure speed by 50%).

### 2.2 The "Chain Ladder" Neural Network

*   **Input:** A sequence of past payments $[P_1, P_2, \dots, P_t]$.
*   **Architecture:** Recurrent Neural Network (RNN) or LSTM.
*   **Output:** Predicted sequence $[P_{t+1}, \dots, P_{ult}]$.
*   *Advantage:* Can learn non-linear development patterns that standard Chain Ladder misses.

---

## 3. Theoretical Properties

### 3.1 IBNR in a Micro-World

*   **Pure IBNR:** We don't know the claim exists.
    *   *Model:* Frequency GLM (Poisson) to predict *count* of late-reported claims per day.
*   **IBNER (Development):** We know the claim, but the amount will change.
    *   *Model:* Regression (XGBoost) on the difference between Ultimate and Current Incurred.

### 3.2 Feature Importance (SHAP)

*   **Explainability:** Actuaries need to know *why* the reserve increased.
*   **SHAP Values:** "The reserve increased by \$5k because 'Attorney' changed from No to Yes."
*   This builds trust with Claims Adjusters.

---

## 4. Modeling Artifacts & Implementation

### 4.1 Survival Analysis (Python `lifelines`)

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter, CoxPHFitter

# Data: Individual Claims
# 'T': Duration (Months)
# 'E': Event (1=Closed, 0=Open/Censored)
data = pd.DataFrame({
    'T': [5, 12, 24, 3, 30, 10, 15, 6],
    'E': [1, 1, 0, 1, 0, 1, 1, 0],
    'Attorney': [0, 1, 1, 0, 1, 0, 1, 0],
    'Age': [30, 50, 45, 25, 60, 35, 40, 28]
})

# 1. Kaplan-Meier Curve (Non-Parametric)
kmf = KaplanMeierFitter()
kmf.fit(data['T'], event_observed=data['E'])

plt.figure(figsize=(8, 5))
kmf.plot_survival_function()
plt.title("Probability of Claim Remaining Open")
plt.xlabel("Months since Report")
plt.ylabel("Survival Probability")
plt.grid(True)
plt.show()

# 2. Cox Proportional Hazards (Regression)
cph = CoxPHFitter()
cph.fit(data, duration_col='T', event_col='E')

print("\nCox Model Summary:")
cph.print_summary()

# Interpretation:
# If Attorney coef is -0.5, exp(-0.5) = 0.60.
# Claims with Attorneys close 40% slower than baseline.
```

### 4.2 XGBoost for Ultimate Loss

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Data: Closed Claims (Training Set)
# Features: [Paid_at_12, Case_at_12, Age, Injury_Code]
# Target: Ultimate_Payment

X = np.random.rand(1000, 4) # Synthetic Features
y = X[:, 0] * 1.5 + X[:, 1] * 0.5 + np.random.normal(0, 0.1, 1000) # Synthetic Target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Fit XGBoost
model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100)
model.fit(X_train, y_train)

# Predict
preds = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, preds))
print(f"RMSE: {rmse:.3f}")

# Feature Importance
xgb.plot_importance(model)
plt.show()

# Application to Open Claims:
# 1. Take all currently open claims.
# 2. Construct feature vector X (current state).
# 3. Predict Y (Ultimate).
# 4. Reserve = Sum(Predicted Y) - Sum(Paid to Date).
```

---

## 5. Evaluation & Validation

### 5.1 The "Aggregate" Check

*   Sum the micro-reserves for Accident Year 2020.
*   Compare to the Chain Ladder reserve.
*   **Divergence:** If Micro says \$50M and CL says \$80M, investigate.
    *   Maybe CL is over-reacting to a few large claims.
    *   Maybe Micro is missing a systemic trend (Inflation) that isn't in the features.

### 5.2 Time-Based Validation

*   Train on claims closed before 2022.
*   Test on claims closed in 2023.
*   **Drift:** If the model fails on 2023, the claims environment has changed (Concept Drift).

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Data Leakage**
    *   **Issue:** Using "Total Paid" as a feature.
    *   **Reality:** Total Paid *is* the target (or part of it). You can only use "Paid *to Date*".

2.  **Trap: The "Reopening" Problem**
    *   **Issue:** Survival models assume "Closed" is final.
    *   **Reality:** Claims reopen.
    *   **Fix:** Model "Time to Next Payment" instead of "Time to Closure". Or use a Multi-State Model (Open $\leftrightarrow$ Closed).

### 6.2 Implementation Challenges

1.  **Data Quality:**
    *   ICR requires clean snapshots of data at specific dates (e.g., "What was the Case Reserve on Jan 1, 2019?").
    *   Most systems overwrite history. You need a **Transaction Log**.

---

## 7. Advanced Topics & Extensions

### 7.1 NLP on Adjuster Notes

*   **Text:** "Claimant is very litigious. Hired Dewey, Cheatem & Howe."
*   **Model:** BERT / LLM to extract sentiment or flags.
*   **Impact:** A "Litigation Risk Score" feature often boosts model accuracy significantly.

### 7.2 Hierarchical Compartmental Models

*   Physics-based approach.
*   Claims flow between "compartments" (Reported $\to$ Paid $\to$ Closed).
*   Differential equations describe the flow rates.

---

## 8. Regulatory & Governance Considerations

### 8.1 "Black Box" Concerns

*   Regulators are skeptical of "The Algo says \$50M."
*   **Hybrid Approach:** Use Chain Ladder for the official books. Use ML for the "Best Estimate" internal view or to allocate reserves to specific segments.

---

## 9. Practical Example

### 9.1 Worked Example: The Triage Model

**Scenario:**
*   New claims arrive.
*   **Model:** Predicts probability of "Jumper" (Low initial reserve $\to$ High ultimate payment).
*   **Action:** If Prob > 80%, assign to Senior Adjuster immediately.
*   **Result:** Early intervention reduces the ultimate cost. The model *changes* the outcome (Heisenberg effect).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Micro-Reserving** uses the transaction log.
2.  **Survival Analysis** models the timeline.
3.  **ML** predicts the severity.

### 10.2 When to Use This Knowledge
*   **Claims Triage:** Operational efficiency.
*   **Segment Analysis:** "Are claims with Attorney X costing more?"

### 10.3 Critical Success Factors
1.  **Data Engineering:** Reconstructing historical snapshots is 80% of the work.
2.  **Hybrid Validation:** Always benchmark against Chain Ladder.
3.  **Explainability:** Use SHAP values to sell it to the Claims Department.

### 10.4 Further Reading
*   **Wüthrich (2018):** "Machine Learning in Individual Claims Reserving".
*   **Taylor:** "Loss Reserving: An Actuarial Perspective" (Micro chapters).

---

## Appendix

### A. Glossary
*   **Censoring:** Incomplete observation of a timeline.
*   **Hazard Rate:** Instantaneous risk.
*   **Snapshot Date:** The "as of" date for feature engineering.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Cox Hazard** | $h_0(t)e^{\beta X}$ | Survival Regression |
| **MSE** | $\frac{1}{n}\sum(y-\hat{y})^2$ | ML Loss Function |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
