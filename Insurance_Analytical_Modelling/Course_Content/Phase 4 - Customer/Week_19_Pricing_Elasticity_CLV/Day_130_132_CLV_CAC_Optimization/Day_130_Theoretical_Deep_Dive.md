# Customer Churn Prediction (Part 2) - Advanced Techniques & Case Study - Theoretical Deep Dive

## Overview
"Predicting churn is easy. Preventing it is hard."
Yesterday, we used Survival Analysis to ask "When?".
Today, we use **XGBoost** to ask "Who?" and **Uplift Modeling** to ask "Who is persuadable?".
We conclude Phase 4 with a full-stack **Churn Prevention System**.

---

## 1. Conceptual Foundation

### 1.1 The Churn Taxonomy

1.  **Voluntary Churn:** Customer cancels (Price, Service, Competitor). *Targetable.*
2.  **Involuntary Churn:** Payment failure, Death, Moving out of state. *Not Targetable.*
3.  **Rotational Churn:** Customer switches every year for a bonus (The "Switcher"). *Hard to stop.*

### 1.2 Uplift Modeling (The "Why")

*   **Traditional Churn Model:** Targets high-risk customers.
    *   *Risk:* You might target a "Sleeping Dog" (someone who forgot they subscribed) and remind them to cancel.
*   **Uplift Model:** Targets customers where:
    $$ P(\text{Stay} | \text{Treatment}) - P(\text{Stay} | \text{Control}) > 0 $$

---

## 2. Mathematical Framework

### 2.1 XGBoost for Churn

*   **Objective:** Minimize LogLoss (Binary Classification).
*   **Key Features:**
    *   **Lag Features:** `calls_last_30d`, `claims_last_90d`.
    *   **Delta Features:** `premium_increase_pct` (Current vs. Last Year).
    *   **Interaction:** `tenure * claims_count` (New customers with claims are high risk).

### 2.2 The Transformed Outcome Method (Uplift)

*   **Goal:** Predict Uplift $\tau(x)$.
*   **Transformation:** Define a new target variable $Z$:
    *   $Z = 2$ if (Treated & Stayed) or (Control & Churned).
    *   $Z = -2$ if (Treated & Churned) or (Control & Stayed).
*   **Result:** Training a regression model on $Z$ estimates $\tau(x)$.

---

## 3. Theoretical Properties

### 3.1 Feature Importance (SHAP)

*   **Global Importance:** "Price Increase" is the #1 driver of churn.
*   **Local Importance:** "For Customer John, the #1 driver was a 'Denied Claim'."
*   *Action:* The Call Center agent sees the Local SHAP values and says: "I see you had a claim denied..."

### 3.2 The "Feedback Loop" of Intervention

*   **Scenario:** You predict Churn. You offer a discount. The customer stays.
*   **Data Pollution:** The training data now shows "High Risk Customer -> Stayed".
*   **Fix:** You must record the **Treatment** as a feature in future models, or exclude treated customers from training.

---

## 4. Modeling Artifacts & Implementation

### 4.1 XGBoost Pipeline

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split

# 1. Feature Engineering (Sliding Window)
df['calls_trend'] = df['calls_last_30d'] / (df['calls_last_90d'] / 3)

# 2. Train XGBoost
model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    scale_pos_weight=10 # Handle Class Imbalance (Churn is rare)
)
model.fit(X_train, y_train)

# 3. Evaluation
y_pred = model.predict_proba(X_test)[:, 1]
print(f"AUC: {roc_auc_score(y_test, y_pred)}")
```

### 4.2 Uplift Model (CausalML)

```python
from causalml.inference.meta import XGBoostRegressor
from causalml.inference.meta import BaseXRegressor

# 1. T-Learner (Two Models)
# Model 0: Predicts outcome for Control group
# Model 1: Predicts outcome for Treatment group
learner = BaseXRegressor(learner=XGBoostRegressor())
uplift = learner.fit_predict(X=X, treatment=treatment, y=outcome)

# 2. Select Top Decile
df['uplift_score'] = uplift
target_list = df.sort_values('uplift_score', ascending=False).head(1000)
```

---

## 5. Evaluation & Validation

### 5.1 Qini Coefficient (Area Under Uplift Curve)

*   **Random Targeting:** Diagonal line.
*   **Model Targeting:** Curve above the diagonal.
*   **Qini:** Area between Model and Random.

### 5.2 Dollar Value of Retention

*   **Metric:** Saved CLV.
    $$ \text{Value} = \sum_{\text{Saved}} (\text{CLV} - \text{Cost of Offer}) $$

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Leakage from the Future**
    *   *Scenario:* Feature `days_since_last_payment`.
    *   *Problem:* If they churned, this value grows indefinitely. It "leaks" the label.
    *   *Fix:* Cut off all features at the "Observation Date" (e.g., 30 days before churn).

2.  **Trap: The "Save" Rate Illusion**
    *   *Scenario:* Agent claims "I saved 50% of callers!"
    *   *Reality:* Those 50% were bluffing and weren't going to cancel anyway.
    *   *Fix:* Measure against a **Control Group** who were *not* offered the save.

---

## 7. Advanced Topics & Extensions

### 7.1 Sequence Modeling (Transformers)

*   **BERT for Churn:** Treat the sequence of customer touchpoints (`Web_Visit -> Call -> App_Login`) as a "sentence".
*   **Attention:** The model learns that `Call -> Denied_Claim` is a critical phrase.

### 7.2 Social Network Churn

*   **Viral Churn:** If a "Family Head" churns, the whole family churns.
*   **Graph Features:** `neighbors_churned_count`.

---

## 8. Regulatory & Governance Considerations

### 8.1 Fairness in Retention Offers

*   **Risk:** You offer discounts only to "High Income" areas because they have higher Uplift.
*   **Regulation:** This is Redlining.
*   **Constraint:** Ensure equal opportunity for retention offers across protected classes.

---

## 9. Practical Example

### 9.1 Worked Example: The "Pre-Emptive" Strike

**Scenario:**
*   **Trigger:** Customer logs into the "Cancel Policy" page.
*   **Real-Time Score:**
    *   Churn Prob: 90%.
    *   CLV: \$5,000.
    *   Uplift (Discount): High.
*   **Action:**
    *   Website displays: "Wait! Before you go, here is a 10% discount for being a loyal customer."
*   **Result:** Customer accepts.
*   **Backend:** System tags this as a "Retained via Discount" event (for future training).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **XGBoost** finds the risk.
2.  **Uplift** finds the opportunity.
3.  **Intervention** must be timely.

### 10.2 When to Use This Knowledge
*   **Operations:** "Who should the 'Save Team' call today?"
*   **Strategy:** "Is our price increase causing churn?"

### 10.3 Critical Success Factors
1.  **Speed:** Churn prediction is a "Real-Time" problem. Predicting churn *after* they leave is useless.
2.  **Experimentation:** You must constantly A/B test your retention offers.

### 10.4 Further Reading
*   **Uber Engineering:** "Causal Inference at Uber" (Uplift modeling at scale).

---

## Appendix

### A. Glossary
*   **Uplift:** Incremental benefit of an action.
*   **LogLoss:** Standard metric for probability accuracy.
*   **SHAP:** Shapley Additive Explanations.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Uplift** | $E[Y|T] - E[Y|C]$ | Targeting |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
