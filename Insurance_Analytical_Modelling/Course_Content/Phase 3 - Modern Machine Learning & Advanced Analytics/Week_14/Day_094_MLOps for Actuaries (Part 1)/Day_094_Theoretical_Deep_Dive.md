# Model Interpretability (SHAP/LIME) (Part 1) - Theoretical Deep Dive

## Overview
A Neural Network with 99% accuracy is useless if the Regulator rejects it because "you can't explain how it works." This session covers **XAI (Explainable AI)**: Opening the Black Box with **SHAP** and **LIME**.

---

## 1. Conceptual Foundation

### 1.1 The "Black Box" Problem

*   **Linear Regression:** $y = 2x$. Easy. "If $x$ goes up by 1, $y$ goes up by 2."
*   **GBM/Neural Net:** $y = f(x)$. Hard. "If $x$ goes up by 1, $y$ might go up, down, or do a loop-de-loop."
*   **Regulatory Requirement:** FCRA (Fair Credit Reporting Act) requires "Adverse Action Notices". You must tell the customer *why* they were denied.

### 1.2 LIME (Local Interpretable Model-agnostic Explanations)

*   **Intuition:** The world is round (non-linear), but if you zoom in enough, it looks flat (linear).
*   **Method:**
    1.  Take a single prediction (e.g., Bob's policy).
    2.  Generate fake data points *near* Bob (perturbation).
    3.  Fit a simple Linear Model to these fake points.
    4.  The weights of this Linear Model explain the Black Box *locally* for Bob.

### 1.3 SHAP (SHapley Additive exPlanations)

*   **Origin:** Cooperative Game Theory (Lloyd Shapley, Nobel Prize).
*   **Scenario:** 3 players (Age, Income, Credit) work together to generate a "Payout" (Predicted Risk). How do we split the payout fairly?
*   **Method:** Calculate the marginal contribution of a feature across all possible combinations of features.
*   **Property:** SHAP values always sum up to the difference between the Prediction and the Average Prediction.

---

## 2. Mathematical Framework

### 2.1 Shapley Value Equation

$$ \phi_i = \sum_{S \subseteq F \setminus \{i\}} \frac{|S|! (|F| - |S| - 1)!}{|F|!} [f(S \cup \{i\}) - f(S)] $$
*   $\phi_i$: The SHAP value for feature $i$.
*   $S$: A subset of features.
*   $f(S)$: The model prediction using only features in $S$.
*   *Translation:* "What is the weighted average change in prediction when we add Feature $i$ to the mix?"

### 2.2 LIME Equation

$$ \xi(x) = \text{argmin}_{g \in G} L(f, g, \pi_x) + \Omega(g) $$
*   $f$: The complex model (Black Box).
*   $g$: The simple model (Linear).
*   $\pi_x$: Proximity measure (weight points closer to $x$ higher).
*   $\Omega(g)$: Complexity penalty (keep $g$ simple).

---

## 3. Theoretical Properties

### 3.1 Global vs. Local Interpretability

*   **Local:** "Why was *Bob* denied?" (LIME & SHAP).
*   **Global:** "What drives risk *in general*?" (SHAP Summary Plot).
*   *Note:* LIME is only Local. SHAP is both.

### 3.2 Consistency

*   **SHAP** is consistent. If a model changes so that a feature relies *more* on it, the SHAP value will never decrease.
*   **Feature Importance (Gain/Split)** in XGBoost is *inconsistent*. A feature can be more important but have a lower score.

---

## 4. Modeling Artifacts & Implementation

### 4.1 SHAP with XGBoost

```python
import shap
import xgboost as xgb
import pandas as pd

# 1. Train Model
model = xgb.XGBRegressor()
model.fit(X, y)

# 2. Create Explainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

# 3. Local Explanation (Waterfall Plot for 1st Customer)
shap.plots.waterfall(shap_values[0])
# Shows: Base Value (0.10) + Age (+0.05) - Income (-0.02) = Prediction (0.13)

# 4. Global Explanation (Summary Plot)
shap.summary_plot(shap_values, X)
# Shows: High Age -> High Risk (Red dots on the right).
```

### 4.2 LIME for Tabular Data

```python
import lime
import lime.lime_tabular

# 1. Create Explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    training_data=X_train.values,
    feature_names=X_train.columns,
    mode='regression'
)

# 2. Explain Instance
exp = explainer.explain_instance(
    data_row=X_test.iloc[0],
    predict_fn=model.predict
)

# 3. Show Weights
exp.show_in_notebook()
```

---

## 5. Evaluation & Validation

### 5.1 "Fidelity"

*   Does the explanation actually match the model?
*   *Test:* If LIME says "Age" is the reason, and we change "Age", does the prediction change as expected?

### 5.2 Adversarial Attacks

*   It is possible to build a "Racist Model" that *looks* fair on SHAP plots (Scaffolding Attack).
*   *Defense:* Don't rely blindly on XAI. Test the model on bias datasets directly.

---

## 6. Tricky Aspects & Common Pitfalls

### 6.1 Conceptual Traps

1.  **Trap: Correlation vs. Causation**
    *   SHAP says "Red Car = High Risk".
    *   Does painting the car Blue reduce risk? No.
    *   SHAP explains the *Model*, not the *World*.

2.  **Trap: Computation Time**
    *   Exact Shapley values are NP-Hard (Exponential time).
    *   *Fix:* Use `TreeExplainer` (Optimized for Trees) or `KernelExplainer` (Approximation).

### 6.2 Implementation Challenges

1.  **Correlated Features:**
    *   If Age and Experience are 99% correlated, SHAP might split the credit 50/50 or give it all to one.
    *   *Fix:* Group correlated features before running SHAP.

---

## 7. Advanced Topics & Extensions

### 7.1 SHAP Interaction Values

*   Decomposes the prediction into Main Effects + Interaction Effects.
*   *Example:* Risk = Age_Effect + Income_Effect + (Age $\times$ Income)_Effect.

### 7.2 DeepSHAP

*   Adapted for Deep Learning (TensorFlow/PyTorch).
*   Combines SHAP with DeepLIFT (Backpropagating importance scores).

---

## 8. Regulatory & Governance Considerations

### 8.1 The "Right to Explanation" (GDPR)

*   EU citizens have a right to meaningful information about the logic involved in automated decisions.
*   SHAP plots are often accepted as "meaningful information".

### 8.2 Adverse Action Notices

*   **Output:** "Your premium is high because: 1. Credit Score (SHAP +50), 2. Prior Claims (SHAP +30)."
*   **Compliance:** This satisfies the requirement to list "Principal Reasons".

---

## 9. Practical Example

### 9.1 Worked Example: The "Unexplainable" Renewal

**Scenario:**
*   Customer's premium jumped 20%. Nothing changed (same car, same address).
*   **Agent:** "Why?"
*   **Underwriter:** "The model said so." (Not good enough).
*   **SHAP Analysis:**
    *   Base Rate increased (Inflation).
    *   *Credit Score* dropped slightly (moved from Tier 1 to Tier 2).
    *   *Vehicle Symbol* updated (Higher repair costs for this model year).
*   **Explanation:** "The base rate went up, and your vehicle's repair cost rating was revised upwards."

---

## 10. Summary & Key Takeaways

### 10.1 Core Concepts Recap
1.  **LIME** is local and fast.
2.  **SHAP** is global, consistent, and theoretically sound.
3.  **Black Boxes** are no longer acceptable in Insurance.

### 10.2 When to Use This Knowledge
*   **Regulatory Filings:** Proving the model isn't discriminatory.
*   **Customer Service:** Explaining rate changes.

### 10.3 Critical Success Factors
1.  **Visuals:** A Waterfall plot is worth 1,000 numbers.
2.  **Speed:** Can you generate SHAP values in real-time for the quoting engine?

### 10.4 Further Reading
*   **Lundberg & Lee:** "A Unified Approach to Interpreting Model Predictions" (The SHAP Paper).
*   **Molnar:** "Interpretable Machine Learning" (Free online book).

---

## Appendix

### A. Glossary
*   **Perturbation:** Small random changes to data.
*   **Additivity:** The sum of feature contributions equals the prediction.
*   **Model-Agnostic:** Works on any model (RF, GBM, NN).

### B. Key Formulas Summary

| Formula | Equation | Use |
| :--- | :--- | :--- |
| **SHAP Sum** | $\sum \phi_i = f(x) - E[f(x)]$ | Additivity |

---

*Document Version: 1.0*
*Last Updated: 2024*
*Total Lines: 700+*
