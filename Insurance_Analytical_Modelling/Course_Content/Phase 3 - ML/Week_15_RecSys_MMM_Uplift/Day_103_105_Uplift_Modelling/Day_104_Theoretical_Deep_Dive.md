# Uplift Modelling (Part 2) - Uplift Trees & Random Forests - Theoretical Deep Dive

## Overview
Yesterday, we used Meta-Learners (S/T/X) which wrap around standard models. Today, we modify the algorithm itself. **Uplift Trees** change the *splitting criterion* of a Decision Tree to directly maximize the difference between Treatment and Control.

---

## 1. Conceptual Foundation

### 1.1 Standard Tree vs. Uplift Tree

*   **Standard Tree (CART):** Splits to maximize *purity* (e.g., Gini Impurity).
    *   Goal: Separate "Churners" from "Non-Churners".
*   **Uplift Tree:** Splits to maximize *divergence* between Treatment and Control.
    *   Goal: Separate "People who react to discount" from "People who don't".

### 1.2 The Splitting Logic

*   **Node A:** Treatment Retention = 80%, Control Retention = 80%. (Uplift = 0%).
*   **Node B:** Treatment Retention = 90%, Control Retention = 60%. (Uplift = 30%).
*   **Action:** The tree will try to isolate Node B.

---

## 2. Mathematical Framework

### 2.1 Distribution Divergence Measures

To decide where to split, we measure the "distance" between the Treatment distribution ($P^T$) and Control distribution ($P^C$) in the child nodes.

1.  **Kullback-Leibler (KL) Divergence:**
    $$ D_{KL}(P^T || P^C) = \sum P^T(y) \log \frac{P^T(y)}{P^C(y)} $$
    *   Measures how different the two distributions are.

2.  **Euclidean Distance:**
    $$ D_{Euc} = \sum (P^T(y) - P^C(y))^2 $$

3.  **Chi-Squared Statistic:**
    *   Standard statistical test for independence.

### 2.2 Honest Estimation

*   **Problem:** If we use the same data to build the tree structure *and* estimate the leaf values, we overfit. We find "fake" uplift.
*   **Solution:** Split training data into:
    1.  **Subsample A:** Build the tree structure (Splits).
    2.  **Subsample B:** Estimate the Uplift in the leaves.

---

## 3. Theoretical Properties

### 3.1 Uplift Random Forest

*   **Ensemble:** Just like a standard Random Forest, we build 100+ Uplift Trees and average their predictions.
*   **Benefit:** Reduces variance. Uplift signal is often weak and noisy; averaging helps significantly.

### 3.2 Feature Importance in Uplift

*   Which features drive *persuadability*?
*   *Example:* "Age" might be important for Churn, but "Tenure" might be important for Uplift (New customers react to discounts; old customers don't).

---

## 4. Modeling Artifacts & Implementation

### 4.1 CausalML Implementation

```python
from causalml.inference.tree import UpliftRandomForestClassifier
import pandas as pd

# 1. Prepare Data
# X: Features, T: Treatment (0/1), y: Outcome (0/1)
X = df[['Age', 'Premium', 'Tenure']]
treatment = df['treatment_group'].apply(lambda x: 1 if x == 'discount' else 0)
y = df['renewed']

# 2. Train Uplift Random Forest
uplift_model = UpliftRandomForestClassifier(
    n_estimators=100,
    evaluationFunction='KL',  # Use KL Divergence for splitting
    control_name='0'
)

uplift_model.fit(X.values, treatment=treatment.values, y=y.values)

# 3. Predict Uplift
pred_uplift = uplift_model.predict(X.values)
```

### 4.2 Visualizing the Tree

*   We can plot a single tree to understand the logic.
*   *Root Node:* "Is Premium > \$1000?"
    *   *Yes:* High Uplift (Price sensitive).
    *   *No:* Low Uplift.

---

## 5. Evaluation & Validation

### 5.1 The Qini Curve

Since we can't see ground truth, we use the **Qini Curve**.

1.  **Rank:** Sort customers by predicted Uplift (High to Low).
2.  **Bin:** Divide into deciles (Top 10%, Next 10%...).
3.  **Calculate Cumulative Gain:**
    $$ G(k) = N_t(k) \times (R_t(k) - R_c(k)) $$
    *   $N_t(k)$: Number of treated people in top $k$ percent.
    *   $R_t(k)$: Retention rate of treated people in top $k$.
    *   $R_c(k)$: Retention rate of control people in top $k$.

### 5.2 AUUC (Area Under Uplift Curve)

*   **Perfect Model:** Identifies all Persuadables first. Curve goes up steeply.
*   **Random Model:** Curve follows the diagonal.
*   **AUUC:** The area between the Model Curve and the Random Diagonal.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: High Variance in Leaves**
    *   If a leaf has 10 Treated people and 2 Control people, the Uplift estimate is garbage.
    *   *Fix:* Enforce `min_samples_leaf` to be high (e.g., 100+ per group).

2.  **Trap: Treatment Imbalance**
    *   If 99% of data is Control, the tree struggles to find Treated samples to calculate divergence.
    *   *Fix:* Stratified sampling or re-weighting.

---

## 7. Advanced Topics & Extensions

### 7.1 Contextual Treatment Selection

*   What if we have *multiple* treatments? (10% off, Free Gift, Call).
*   **Multi-Treatment Uplift RF:** Predicts a vector of uplifts $[\tau_{10\%}, \tau_{Gift}, \tau_{Call}]$.
*   *Action:* Choose $\text{argmax}(\tau_i)$.

### 7.2 Causal Forest (EconML)

*   Uses "Orthogonalization" to remove the effect of confounders before estimating uplift.
*   More robust for observational data (non-RCT).

---

## 8. Regulatory & Governance Considerations

### 8.1 Disparate Impact

*   If the Uplift Model targets only "Young People" for discounts, is that Age Discrimination?
*   *Check:* Calculate the average discount rate per protected class.

---

## 9. Practical Example

### 9.1 Worked Example: The "Qini" Calculation

**Scenario:**
*   **Top 10% (Predicted High Uplift):**
    *   Treated: 100 people, 90 renewed. (90%).
    *   Control: 100 people, 60 renewed. (60%).
    *   **Uplift:** 30%. (Great!)
*   **Bottom 10% (Predicted Low Uplift):**
    *   Treated: 100 people, 80 renewed.
    *   Control: 100 people, 80 renewed.
    *   **Uplift:** 0%. (Correctly identified as Sure Things).
*   **Conclusion:** The model successfully ranked customers.

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **Uplift Trees** optimize for divergence, not purity.
2.  **Honest Estimation** prevents overfitting.
3.  **Qini Curve** is the standard metric.

### 10.2 When to Use This Knowledge
*   **Campaign Optimization:** "Who gets the coupon?"
*   **Personalization:** "Which offer works best?"

### 10.3 Critical Success Factors
1.  **Sample Size:** You need enough data in *both* Treatment and Control groups.
2.  **Stability:** Check if the Qini curve holds up on a future time period.

### 10.4 Further Reading
*   **Rzepakowski & Jaroszewicz:** "Decision trees for uplift modeling with single and multiple treatments".

---

## Appendix

### A. Glossary
*   **Divergence:** Difference between distributions.
*   **Honest Tree:** A tree built on one subset and estimated on another.

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **Qini Coefficient** | Area / Ideal Area | Performance |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
