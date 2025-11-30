# Uplift Modelling (Part 1) - Theoretical Deep Dive

## Overview
Traditional Churn Models answer: "Who will leave?"
**Uplift Models** answer: "Who will stay *if we give them a discount*?"
In Insurance, this distinction is worth millions. Sending a discount to someone who was going to stay anyway is a waste. Sending a "We miss you" email to a "Sleeping Dog" might actually *cause* them to leave.

---

## 1. Conceptual Foundation

### 1.1 The Four Quadrants

We segment customers based on their reaction to a Treatment (e.g., Discount, Call).

| Segment | Behavior without Treatment | Behavior with Treatment | Action |
| :--- | :--- | :--- | :--- |
| **Persuadables** | Churn | Stay | **Target** (High Priority) |
| **Sure Things** | Stay | Stay | **Ignore** (Waste of money) |
| **Lost Causes** | Churn | Churn | **Ignore** (Waste of money) |
| **Sleeping Dogs** | Stay | Churn | **Avoid** (Do not wake them!) |

*   *Insurance Example:* A customer has auto-renewal on. If you send them an email "Check your renewal price!", they might realize it went up and switch. They are a **Sleeping Dog**.

### 1.2 Churn Prediction vs. Uplift

*   **Churn Model:** $P(Y=Churn | X)$
*   **Uplift Model:** $P(Y=Stay | X, T=1) - P(Y=Stay | X, T=0)$
    *   $T=1$: Treated (Received Discount).
    *   $T=0$: Control (No Discount).

---

## 2. Mathematical Framework

### 2.1 Conditional Average Treatment Effect (CATE)

$$ \tau(x) = E[Y | X=x, T=1] - E[Y | X=x, T=0] $$

*   $\tau(x)$: The "Uplift" for customer with features $x$.
*   **Goal:** Maximize $\sum \tau(x)$ for the targeted population.

### 2.2 The Fundamental Problem of Causal Inference

*   We can never observe *both* outcomes for the same person.
*   If Alice received the discount, we don't know what she would have done *without* it.
*   *Solution:* Randomized Control Trials (RCT) + Statistical Modeling.

---

## 3. Theoretical Properties

### 3.1 Meta-Learners

Algorithms that use standard ML models (XGBoost, etc.) to estimate CATE.

1.  **S-Learner (Single):**
    *   Train one model: $Y \sim f(X, T)$.
    *   Prediction: $\hat{\tau}(x) = f(x, 1) - f(x, 0)$.
    *   *Pros:* Simple. *Cons:* Often biases $\tau$ towards zero.

2.  **T-Learner (Two):**
    *   Train two models:
        *   $\mu_1(x)$ on Treated group.
        *   $\mu_0(x)$ on Control group.
    *   Prediction: $\hat{\tau}(x) = \mu_1(x) - \mu_0(x)$.
    *   *Pros:* Flexible. *Cons:* High variance if sample sizes differ.

3.  **X-Learner (Cross):**
    *   Complex 3-stage process. Best for imbalanced data (e.g., 99% Control, 1% Treated).

---

## 4. Modeling Artifacts & Implementation

### 4.1 CausalML (Uber's Library)

```python
from causalml.inference.meta import BaseXRegressor
from xgboost import XGBRegressor

# 1. Data
X = df[['Age', 'Premium', 'Tenure']]
T = df['Treatment_Flag'] # 0 or 1
y = df['Renewed_Flag']   # 0 or 1

# 2. Learner
learner = BaseXRegressor(learner=XGBRegressor())

# 3. Estimate Uplift
uplift = learner.fit_predict(X, T, y)
df['uplift_score'] = uplift

# 4. Action
# Target top 10% of uplift_score
target_customers = df.nlargest(1000, 'uplift_score')
```

### 4.2 EconML (Microsoft's Library)

*   Focuses on Econometrics + ML.
*   Good for "Double Machine Learning" (DML).

---

## 5. Evaluation & Validation

### 5.1 Why Accuracy/AUC doesn't work

*   We don't have "Ground Truth" for Uplift (because of the Fundamental Problem).
*   We can't calculate RMSE between $\hat{\tau}$ and $\tau$.

### 5.2 Qini Curve & AUUC

*   **Qini Curve:**
    *   Sort customers by predicted Uplift (High to Low).
    *   Plot Cumulative Incremental Gains.
    *   *Interpretation:* If the curve is above the diagonal, the model is better than random targeting.
*   **AUUC (Area Under Uplift Curve):** Similar to AUC, but for Uplift.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Targeting High Churn Risk**
    *   "Alice has 90% churn risk. Let's send her a gift."
    *   *Reality:* Alice might be a "Lost Cause". The gift costs money and she leaves anyway.
    *   *Fix:* Target High *Uplift*, not High *Risk*.

2.  **Trap: Sleeping Dogs**
    *   "Let's email everyone."
    *   *Reality:* You wake up the Sleeping Dogs.
    *   *Fix:* Check for *negative* uplift scores.

### 6.2 Implementation Challenges

1.  **Data Requirements:**
    *   You NEED Randomized Data (RCT).
    *   You cannot train an Uplift model on purely observational data without strong assumptions (Unconfoundedness).
    *   *Fix:* Run a random holdout group in your campaigns.

---

## 7. Advanced Topics & Extensions

### 7.1 Uplift Decision Trees

*   Instead of minimizing Gini Impurity (Class purity), we maximize **Distribution Divergence** (Difference between Treatment and Control distributions).
*   *Library:* `CausalML` supports Uplift Random Forests.

### 7.2 Cost-Sensitive Uplift

*   **Profit:** $Value \times P(Stay|T) - Cost - Value \times P(Stay|C)$.
*   We optimize for *Net Value*, not just retention probability.

---

## 8. Regulatory & Governance Considerations

### 8.1 Price Optimization

*   **Risk:** Using Uplift to charge higher prices to people who are "Sure Things" (Price Elasticity).
*   **Regulation:** Banned in UK (FCA) and many US states. "Dual Pricing" is illegal.
*   **Safe Use:** Use Uplift for *Retention Offers* (Discounts), not for *Base Pricing* (Increases).

---

## 9. Practical Example

### 9.1 Worked Example: The "10% Off" Campaign

**Scenario:**
*   Insurer wants to retain customers.
*   **RCT:**
    *   Group A (Random 10k): No Offer. Retention = 80%.
    *   Group B (Random 10k): 10% Discount. Retention = 85%.
    *   Average Uplift = 5%.
*   **Uplift Model:**
    *   Finds a segment (Young Drivers) where Uplift = 15%.
    *   Finds a segment (Seniors) where Uplift = 0%.
*   **Action:** Send discount ONLY to Young Drivers.
*   **Result:** Same retention gain, but 50% less cost (discounts given).

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Persuadables** are the only ones worth paying for.
2.  **Sleeping Dogs** must be left alone.
3.  **CATE** is the metric, not Accuracy.

### 10.2 When to Use This Knowledge
*   **Marketing:** Coupon targeting.
*   **Claims:** "Should we call this claimant or just pay?" (Settlement Uplift).

### 10.3 Critical Success Factors
1.  **Randomization:** Start with an A/B test to generate training data.
2.  **Negative Uplift:** Always look for it.

### 10.4 Further Reading
*   **Gutierrez & Gerardy:** "Causal Inference and Uplift Modeling: A Review and Assessment".

---

## Appendix

### A. Glossary
*   **CATE:** Conditional Average Treatment Effect.
*   **RCT:** Randomized Control Trial.
*   **Counterfactual:** The "What If" scenario.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Uplift** | $P(Y|X, T=1) - P(Y|X, T=0)$ | Definition |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
